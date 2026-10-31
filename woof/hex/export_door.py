"""``woof hex export``: a forecast's history on a lat-lon grid, without drawing.

Until this door the only way a hex history became a regular latitude-longitude
NetCDF file was inside ``render``: the converter wrote the frames into the
render scratch, the renderer drew them, and the scratch was removed.  A user who
wanted the numbers -- to feed another model, to score with their own tool, to
open in any NetCDF reader -- had no command for it (woof global has had
``export`` since 0.1).

What it does, and nothing more: it runs the same Rust converter the render
door runs (``rw_mpas_convert``, resolved the same way), over every history
frame of a run, onto the window asked for, and files each frame as
``<out>/<view>/<valid-day>/<frame>``.  The frames are WRF-shaped
(``wrfout_d01_<valid>``: XLAT/XLONG on a regular grid, the WRF names for the
fields, the MPAS provenance in global attributes), which is what every WRF
reader and the WOOF engine's own wrfout readers open.  The converter's own
JSON receipt -- mesh, weights and per-frame output digests and the WRF fields
the history did not carry -- is kept beside them as ``export-receipt.json``.

The window vocabulary is the render door's: ``global`` (the whole sphere at the
mesh's own spacing), ``global-0.25``, ``mesh``, ``focus``, ``region:`` and
``lambert:`` specs.  ``--remap product`` (the default) interpolates the state
linearly and remaps accumulations conservatively; ``nearest`` copies the nearest
cell, which is exact at cell centres and steps at cell edges.

A variable-resolution mesh's ``global`` export is a stack of grids, one per
band of the mesh's own spacing (:func:`export_plan`): the whole sphere at the
outer mesh's coarse spacing (``wrfout_d01``), a box for every ring at that
ring's spacing (``d02`` ...), and the refined core at its own spacing (a
lat-lon box too, filed as the ``mesh`` view, the last domain), so every file
carries the CF 1-D axes of a lat-lon frame.  Each region is written at the detail the
model has there, the way a WRF run writes its parent and its nests.  THE
BREAKAGE THIS PREVENTS (an Arizona 96 -> 3 km zoom globe, 800 km core,
36 h, 8 x RTX PRO 6000, woof hex 0.4.10, 2026-10-30): ``global`` laid the
whole sphere down at the refined core's median spacing, capped at 9 million
points (9.4 km), so the planet's 96 km cells were interpolated onto a grid
about 100 times finer than the model, and the export wrote 252 GB of frames
(7.3 GB an hour; 19 GB an hour uncompressed on 0.4.9) after the cards had gone
idle, without the 3 km core's own detail in any of them.

The export can also run beside the forecast (``--follow``), each frame the
moment it is closed, and the one-shot export after the forecast then takes the
frames the follower already wrote (``export-ready.jsonl``: same grid, the
history unchanged) instead of converting them again.  The doors start that
follower themselves when ``WOOF_HEX_EXPORT_LIVE`` names the export
(:func:`live_export_command`).

No numerics are done here: Python schedules, Rust converts.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any

from . import render_door
from .errors import MpasPortError

RECEIPT_NAME = "export-receipt.json"
RECEIPT_SCHEMA = "gpuwm-hex.export/v1"


class ExportDoorError(MpasPortError):
    """The export door refuses, and says why."""


def _valid_day(frame: Path) -> str:
    """``YYYY-MM-DD`` from a ``wrfout_d01_YYYY-MM-DD_HH_MM_SS`` frame name."""

    parts = frame.name.split("_")
    for token in parts:
        if len(token) == 10 and token[4] == "-" and token[7] == "-":
            return token
    return "undated"


READY_NAME = "export-ready.jsonl"
HISTORY_READY_NAME = "history-ready.jsonl"


def _history_ready_paths(forecast_dir: Path) -> list[Path]:
    """Frames the running forecast has closed, in the order it listed them.

    ``history-ready.jsonl`` is appended by every forecast driver after a frame
    is durable (``{"step", "path"}``, or the products worker's longer row);
    a line still being written is skipped until it is whole.  An executor that
    keeps no such list (the variable-resolution door's) is followed by its
    durable sidecars instead (``_durable_frames``)."""

    try:
        text = (forecast_dir / HISTORY_READY_NAME).read_text(encoding="utf-8")
    except OSError:
        return _durable_frames(forecast_dir)
    seen: list[Path] = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        path = row.get("path") if isinstance(row, dict) else None
        if path and Path(path) not in seen:
            seen.append(Path(path))
    return seen


HISTORY_DURABLE_GLOB = "*history.*.nc.durable.json"


def _durable_frames(forecast_dir: Path) -> list[Path]:
    """Closed frames by their durable sidecars, in valid-time (name) order.

    THE BREAKAGE THIS PREVENTS: the variable-resolution door's executor writes
    each frame and then its ``<frame>.durable.json`` sidecar, but no
    ``history-ready.jsonl``, so ``--follow`` beside it waited for a list that
    never came and exported nothing (2026-10-23).  The sidecar is written
    after the frame, so a frame whose sidecar names its full size is done (the
    same rule as hex-fast-start c21eeba2's first look).  Frames are taken in
    order and the scan stops at the first one not yet whole, so a later frame
    is never exported ahead of an earlier one."""

    frames: list[Path] = []
    for sidecar in sorted(forecast_dir.glob(HISTORY_DURABLE_GLOB)):
        frame = sidecar.with_name(sidecar.name[: -len(".durable.json")])
        try:
            whole = frame.stat().st_size == int(json.loads(sidecar.read_text(encoding="utf-8"))["bytes"])
        except (OSError, ValueError, KeyError, TypeError):
            whole = False
        if not whole:
            break
        frames.append(frame)
    return frames


def _history_folder(forecast_dir: Path) -> Path:
    """The folder holding ``history-ready.jsonl``: the run folder or a door's ``forecast/``."""

    for candidate in (forecast_dir, forecast_dir / "forecast"):
        if (candidate / HISTORY_READY_NAME).is_file():
            return candidate
    for depth in ("*", "*/*"):
        found = sorted(forecast_dir.glob(f"{depth}/{HISTORY_READY_NAME}"))
        if found:
            return found[0].parent
    for candidate in (forecast_dir, forecast_dir / "forecast"):
        if next(candidate.glob(HISTORY_DURABLE_GLOB), None) is not None:
            return candidate
    return forecast_dir


_STAMP = __import__("re").compile(r"(\d{4}-\d{2}-\d{2})[_T](\d{2})[.:_-]?(\d{2})[.:_-]?(\d{2})")


def history_valid_time(path: Path) -> _dt.datetime | None:
    """The valid time a history file's name carries (``...YYYY-MM-DD_HH.MM.SS.nc``), or None."""

    match = None
    for match in _STAMP.finditer(Path(path).name):
        pass
    if match is None:
        return None
    day, hh, mm, ss = match.groups()
    return _dt.datetime.strptime(f"{day} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")


def _follow_from(text: str | None) -> _dt.datetime | None:
    if not text:
        return None
    return _dt.datetime.strptime(text.replace("T", "_").rstrip("Z"), "%Y-%m-%d_%H:%M:%S")


#: Kilometres in one degree of latitude on the mesh's sphere (6371.229 km).
KM_PER_DEGREE = math.pi * 6371.229 / 180.0

#: The lat-lon spacings (degrees) a stack grid takes: every one divides 180 and 360, so a box snapped to its
#: multiples passes the converter's exact-partition check and a whole-sphere grid closes on itself.
STACK_SPACINGS_DEG = (2.0, 1.0, 0.75, 0.5, 0.4, 0.25, 0.2, 0.125, 0.1, 0.08, 0.05, 0.04, 0.025, 0.02,
                      0.0125, 0.01, 0.008, 0.005, 0.004, 0.0025)

#: The converter's limit on an explicit lat-lon grid (rw-mpas MESH_FOCUS_MAX_POINTS_PER_SIDE).
MAX_POINTS_PER_SIDE = 2000

#: Cells a spacing band needs before it gets a grid of its own: a few transition cells between two rings are
#: not a ring.
MIN_BAND_CELLS = 500


def stack_spacing_degrees(spacing_km: float) -> float:
    """The coarsest stack spacing that is no coarser than ``spacing_km`` along a meridian: a grid at it keeps every
    cell of that spacing (the 96 km outer globe at 0.75 degrees, 83 km; a 6 km ring at 0.05 degrees, 5.6 km)."""

    for degrees in STACK_SPACINGS_DEG:
        if degrees * KM_PER_DEGREE <= spacing_km * (1.0 + 1e-9):
            return degrees
    return STACK_SPACINGS_DEG[-1]


def _mesh_cells(mesh):
    """(latitude deg, longitude deg, spacing km) of every cell, as numpy arrays, or None when unreadable."""

    try:
        import numpy as np
        from netCDF4 import Dataset

        from .netcdf_lock import NETCDF
        with NETCDF, Dataset(mesh) as grid:
            area = np.asarray(grid["areaCell"][:], dtype=np.float64)
            lat = np.degrees(np.asarray(grid["latCell"][:], dtype=np.float64))
            lon = np.degrees(np.asarray(grid["lonCell"][:], dtype=np.float64))
            radius = float(getattr(grid, "sphere_radius", 1.0) or 1.0)
    except (ImportError, OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    scale = 6371.229 / radius if radius < 1000.0 else 0.001  # a unit sphere's areas, else metres
    spacing = np.sqrt(2.0 * area / math.sqrt(3.0)) * scale
    return lat, (lon + 180.0) % 360.0 - 180.0, spacing


def _fmt(value: float) -> str:
    return f"{round(value, 9):.9f}".rstrip("0").rstrip(".")


def _latlon_box(south: float, north: float, west: float, east: float, degrees: float) -> tuple[str, int, int, float]:
    """The ``latlon:`` window over a box snapped outward to multiples of ``degrees`` (coarsened until it fits the
    converter's points-a-side limit): (window, nx, ny, degrees). A box that wraps the planet or reaches a pole takes
    every longitude."""

    while True:
        s = max(-90.0, math.floor(south / degrees + 1e-9) * degrees)
        n = min(90.0, math.ceil(north / degrees - 1e-9) * degrees)
        w = math.floor(west / degrees + 1e-9) * degrees
        e = math.ceil(east / degrees - 1e-9) * degrees
        if e - w >= 360.0 - degrees or s <= -90.0 + degrees or n >= 90.0 - degrees:
            w, e = -180.0, 180.0 - degrees
        nx = int(round((e - w) / degrees)) + 1
        ny = int(round((n - s) / degrees)) + 1
        if max(nx, ny) <= MAX_POINTS_PER_SIDE or degrees >= STACK_SPACINGS_DEG[0]:
            return f"latlon:{_fmt(s)},{_fmt(n)},{_fmt(w)},{_fmt(e)},{_fmt(degrees)}", nx, ny, degrees
        degrees = next(d for d in reversed(STACK_SPACINGS_DEG) if d > degrees)


def export_plan(window: str, mesh, view: str | None = None) -> list[dict[str, Any]]:
    """The grids an export writes: one for every window but a variable-resolution mesh's ``global``.

    Such a mesh is cut into bands of its own spacing, by doubling from the refined core's (3, 6, 12, 24, 48 and
    96 km on a 96 -> 3 km zoom globe). The outermost band is the whole sphere on the lat-lon grid of its spacing
    (domain 1, view ``global``); every band between has a lat-lon box over the ground it and the finer bands inside
    it cover, at its spacing (domains 2.., views ``ring-<km>km``, coarse to fine); the refined core is a lat-lon
    box at the core's spacing the same way (the last domain, view ``mesh``), or the converter's Lambert ``mesh``
    window when no lat-lon box holds it at that spacing (a polar core). Each region is written at the detail the model has there, never interpolated finer: see the module
    notes for the breakage. A quasi-uniform mesh (95th percentile spacing within twice the 5th), an unreadable
    mesh, an explicit ``--view`` and every other window keep one grid, as before."""

    single = [{"domain": 1, "view": view or render_door.default_view(window),
               "window": render_door.converter_window(window), "spacing_km": None, "why": None}]
    if window != "global" or view or mesh is None:
        return single
    cells = _mesh_cells(mesh)
    if cells is None:
        return single
    import numpy as np

    lat, lon, spacing = cells
    ordered = np.sort(spacing)
    count = len(ordered)
    if count == 0:
        return single
    low, high = float(ordered[count // 20]), float(ordered[(19 * count) // 20])
    from .products_door import VARIABLE_RESOLUTION_SPACING_RATIO
    if low <= 0.0 or high / low < VARIABLE_RESOLUTION_SPACING_RATIO:
        return single
    finest = float(ordered[count // 100])
    band = np.rint(np.log2(np.maximum(spacing, finest) / finest)).astype(int)
    present = [int(b) for b in np.unique(band) if int((band == b).sum()) >= MIN_BAND_CELLS]
    if len(present) < 2:
        return single
    outer, core = present[-1], present[0]
    band_km = {b: float(np.median(spacing[band == b])) for b in present}
    # The refined core's centre: the mean of its cells on the unit sphere.
    inner = band == core
    phi, lam = np.radians(lat[inner]), np.radians(lon[inner])
    x, y = float(np.mean(np.cos(phi) * np.cos(lam))), float(np.mean(np.cos(phi) * np.sin(lam)))
    centre_lon = math.degrees(math.atan2(y, x))
    globe_degrees = stack_spacing_degrees(band_km[outer])
    globe, nx, ny, globe_degrees = _latlon_box(-90.0, 90.0, -180.0, 180.0, globe_degrees)
    plan = [{"domain": 1, "view": "global", "window": globe, "spacing_km": round(band_km[outer], 3),
             "grid_km": round(globe_degrees * KM_PER_DEGREE, 3), "points": [nx, ny],
             "why": f"the outer mesh ({band_km[outer]:.1f} km cells) on the whole sphere at {_fmt(globe_degrees)} "
                    f"degrees"}]
    def band_box(b: int) -> tuple[str, int, int, float, float]:
        """The lat-lon box over band ``b`` and every finer band, one grid step past its outermost cell, at the
        band's stack spacing: (window, nx, ny, degrees taken, degrees asked)."""
        inside = band <= b
        dlon = (lon[inside] - centre_lon + 180.0) % 360.0 - 180.0
        asked = stack_spacing_degrees(band_km[b])
        box, nx, ny, degrees = _latlon_box(float(lat[inside].min()) - asked, float(lat[inside].max()) + asked,
                                           centre_lon + float(dlon.min()) - asked,
                                           centre_lon + float(dlon.max()) + asked, asked)
        return box, nx, ny, degrees, asked

    for b in reversed(present[1:-1]):
        box, nx, ny, degrees, _asked = band_box(b)
        plan.append({"domain": len(plan) + 1, "view": f"ring-{band_km[b]:.0f}km", "window": box,
                     "spacing_km": round(band_km[b], 3), "grid_km": round(degrees * KM_PER_DEGREE, 3),
                     "points": [nx, ny],
                     "why": f"the {band_km[b]:.1f} km ring and everything inside it at {_fmt(degrees)} degrees"})
    # The refined core is a lat-lon box at its own spacing too (the folder keeps the name `mesh`, the refined core's
    # view), so the converter writes the CF axes of every lat-lon frame (lat, lon, Time, bottom_top,
    # bottom_top_stag, soil_layers_stag), as the single whole-sphere file did. THE BREAKAGE THIS PREVENTS (the
    # independent check of this stack, RTX 5090, 2026-10-30): the core went on the converter's `mesh` window, a Lambert
    # grid, which has no 1-D axes, so the 3 km file lacked those six variables, and readers that open a frame by its
    # 1-D coordinates (the ArWen engine taking a hex parent, CF tools) could not open the core at all. Only a core a
    # lat-lon box cannot hold at its spacing (one that reaches a pole, or wider than the converter's 2000 points a
    # side) keeps the Lambert grid, which would otherwise be coarsened past the cells it holds.
    box, nx, ny, degrees, asked = band_box(core)
    if degrees <= asked:
        plan.append({"domain": len(plan) + 1, "view": "mesh", "window": box, "spacing_km": round(band_km[core], 3),
                     "grid_km": round(degrees * KM_PER_DEGREE, 3), "points": [nx, ny],
                     "why": f"the refined core ({band_km[core]:.2f} km cells) at {_fmt(degrees)} degrees"})
    else:
        plan.append({"domain": len(plan) + 1, "view": "mesh", "window": render_door.converter_window("mesh"),
                     "spacing_km": round(band_km[core], 3), "grid_km": None, "points": None,
                     "why": f"the refined core ({band_km[core]:.2f} km cells) on the refined-core maps' Lambert grid: "
                            f"a lat-lon box at {_fmt(asked)} degrees would need {max(nx, ny) * degrees / asked:.0f} "
                            f"points a side, past the converter's {MAX_POINTS_PER_SIDE}"})
    return plan


def _domain_name(frame: Path, domain: int) -> str:
    """``wrfout_d01_<valid>`` as the stack's ``wrfout_dNN_<valid>``, WRF's name for domain NN."""

    return re.sub(r"^wrfout_d\d{2}_", f"wrfout_d{domain:02d}_", frame.name)


def _same_file(path) -> str:
    """One name for a history however it was reached. THE BREAKAGE THIS PREVENTS (RTX 5090, 2026-10-30): the forecast
    door listed its frames in history-ready.jsonl under the resolved path while their durable sidecars, read first,
    named them through a symlinked folder, so the follower took f000 twice and refused its own frame."""
    return os.path.realpath(str(path))


def _history_identity(path) -> list[int] | None:
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


def _ready_rows(out: Path) -> list[dict[str, Any]]:
    try:
        text = (out / READY_NAME).read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _reusable(rows: list[dict[str, Any]], entry: dict[str, Any], *, field_set: str, remap: str,
              nc_format: str) -> dict[str, dict[str, Any]]:
    """{history path: ready row} of the frames a follower (``--follow`` beside the forecast) already exported onto
    this grid: same window, domain, field set, remap and format, the history unchanged since (size and mtime), the
    frame whole."""

    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (row.get("window"), row.get("field_set"), row.get("remap"), row.get("nc_format"), row.get("domain", 1)) != (
                entry["window"], field_set, remap, nc_format, entry["domain"]):
            continue
        history, frame = row.get("history"), row.get("frame")
        if not history or not frame or row.get("history_identity") != _history_identity(history):
            continue
        try:
            if Path(frame).stat().st_size != int(row.get("bytes", -1)):
                continue
        except (OSError, TypeError, ValueError):
            continue
        found[_same_file(history)] = row
    return found


def _convert_and_file(entry: dict[str, Any], histories: list[Path], *, args, mesh: Path, init,
                      start: str | None, scratch: Path, out: Path, remap: str, convert_exe: Path,
                      jobs: int, replaceable: frozenset = frozenset()) -> tuple[list[dict[str, Any]], dict, list[str]]:
    """Convert ``histories`` onto one grid of the plan and file each frame as ``<out>/<view>/<day>/wrfout_dNN_*``
    (a rename: scratch sits beside ``--out``). A frame already there is replaced only with ``--overwrite`` or when it
    is one a follower wrote (``replaceable``: its history changed since). Returns the delivered rows (each with the
    history it came from), the converter's report and its command."""

    scratch = Path(scratch) / f"d{entry['domain']:02d}"
    shutil.rmtree(scratch, ignore_errors=True)
    report, command = render_door.convert_frames(
        convert_exe, histories=histories, mesh=mesh, scratch=scratch, window=entry["window"],
        field_set=args.field_set, simulation_start=start, nc_format=args.nc_format,
        init=init, model_label=args.model_label, remap=remap, jobs=jobs)
    by_valid = {}
    for history in histories:
        valid = history_valid_time(history)
        if valid is not None:
            by_valid[valid] = history
    by_name = {Path(row.get("output", "")).name: row for row in report.get("frames", [])}
    rows = []
    for frame in sorted((scratch / "frames").glob("wrfout_*")):
        source = by_name.get(frame.name, {})
        target = out / entry["view"] / _valid_day(frame) / _domain_name(frame, entry["domain"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not args.overwrite and str(target) not in replaceable:
                raise ExportDoorError(f"{target} exists; pass --overwrite to replace it")
            target.unlink()
        shutil.move(str(frame), str(target))
        try:
            valid = _dt.datetime.strptime(str(source.get("valid_time", ""))[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            valid = None
        history = by_valid.get(valid) if valid is not None else None
        if history is None and len(histories) == 1:
            history = histories[0]
        rows.append({"frame": str(target), "bytes": target.stat().st_size, "domain": entry["domain"],
                     "view": entry["view"], "window": entry["window"], "sha256": source.get("output_sha256"),
                     "valid_time": source.get("valid_time"), "history": str(history) if history else None,
                     "history_identity": _history_identity(history) if history else None})
    if not args.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    return rows, report, command


def run_follow(args: argparse.Namespace) -> int:
    """``export --follow``: export each frame the moment the forecast closes it.

    THE BREAKAGE THIS PREVENTS: a WOOF domain driven by a hex parent could only
    start after the whole hex forecast had run, because the export read a finished
    run.  A nest fed this way waits hours for boundaries it needs one at a time.
    Followed, each closed frame is converted onto the window at once and filed by an
    atomic rename, then listed in ``<out>/export-ready.jsonl`` (frame path, valid time,
    bytes, sha256, the grid and the history it came from), which a consumer
    (``python -m woof.local_posting``) reads as the frame's ready signal and the
    one-shot export after the forecast reads to take the frame as it is.  Same
    converter, grids and files as the one-shot export; only the timing differs.

    Every frame closed since the last pass is converted together (the converter
    takes them side by side), so a follower that fell behind catches up in one pass
    instead of one frame at a time.  THE BREAKAGE THIS PREVENTS (an Arizona
    zoom globe on 8 cards, 2026-10-30): the forecast closed a frame every 22 s, and
    a follower that converts one frame per pass is level with it only while one
    conversion takes less than that.
    """

    started = time.monotonic()
    if args.forecast_dir is None:
        raise ExportDoorError("--follow watches a forecast folder; give RUN_DIR (the forecast's --out)")
    if not args.mesh or not args.init:
        raise ExportDoorError("--follow runs beside a forecast whose receipt is not written yet; "
                              "give --mesh GRID.nc and --init INIT.nc")
    if not args.simulation_start:
        raise ExportDoorError("--follow needs --simulation-start YYYY-MM-DD_HH:MM:SS (the forecast's start), "
                              "because lead hours are numbered from it before any receipt exists")
    if not args.follow_frames or args.follow_frames < 1:
        raise ExportDoorError("--follow needs --follow-frames N, the number of history frames the forecast writes")
    forecast_dir = Path(args.forecast_dir)
    mesh, init = Path(args.mesh), Path(args.init)
    out = Path(args.out) if args.out else forecast_dir / "export"
    scratch_root = Path(args.scratch) if args.scratch else out.parent / (out.name + ".export-scratch")
    render_door.refuse_scratch_inside_out(scratch_root, out)
    plan = export_plan(args.window, mesh, args.view)
    remap = render_door.resolve_remap(args.remap, composite=False)
    convert_exe = render_door.resolve_convert_exe(args.convert_exe)
    out.mkdir(parents=True, exist_ok=True)
    ready_path = out / READY_NAME
    quiet_limit = float(args.follow_timeout_minutes) * 60.0
    # A nest added while the parent runs (woof hex add-nest) starts at a frame
    # the parent closed earlier: frames before it are not the nest's.
    follow_from = _follow_from(getattr(args, "follow_from", None))
    # --follow-every-minutes: only frames on the consumer's lead grid
    # (follow_from, or the simulation start, plus k * M).  Taking every frame
    # of a parent that writes more often than the nest's boundary interval
    # stopped --follow-frames short of the nest's end (release review 30,
    # finding 2).
    every = getattr(args, "follow_every_minutes", None)
    every_s = None if not every else float(every) * 60.0
    anchor = follow_from or _follow_from(args.simulation_start)
    last_progress = time.monotonic()
    print(f"EXPORT follow frames={args.follow_frames} window={plan[0]['window']} grids={len(plan)} out={out}",
          flush=True)
    done: list[dict[str, Any]] = []
    exported = 0
    converted: set[str] = set()
    while exported < args.follow_frames:
        pending, listed = [], set()
        for p in _history_ready_paths(_history_folder(forecast_dir)):
            if _same_file(p) in converted or _same_file(p) in listed or _before(p, follow_from):
                continue
            listed.add(_same_file(p))
            if every_s is not None and not _on_grid(p, anchor, every_s):
                converted.add(_same_file(p))  # between two of the consumer's leads: never its frame
                continue
            pending.append(p)
        pending = pending[: args.follow_frames - exported]
        if not pending:
            if time.monotonic() - last_progress > quiet_limit:
                raise ExportDoorError(
                    f"--follow: {exported} of {args.follow_frames} frames, and the forecast listed no new "
                    f"frame for {args.follow_timeout_minutes:g} min")
            time.sleep(float(args.follow_poll_seconds))
            continue
        began = time.monotonic()
        rows: list[dict[str, Any]] = []
        for entry in plan:
            delivered, _report, _command = _convert_and_file(
                entry, pending, args=args, mesh=mesh, init=init, start=args.simulation_start,
                scratch=scratch_root / f"pass-{exported:04d}", out=out, remap=remap, convert_exe=convert_exe,
                jobs=1)
            rows += delivered
        if len(rows) != len(pending) * len(plan):
            raise ExportDoorError(f"rw_mpas_convert wrote {len(rows)} frames for {len(pending)} histories on "
                                  f"{len(plan)} grids; expected {len(pending) * len(plan)}")
        seconds = round(time.monotonic() - began, 3)
        order = {str(p): i for i, p in enumerate(pending)}
        rows.sort(key=lambda row: (order.get(str(row.get("history")), len(order)), row["domain"]))
        with ready_path.open("a", encoding="utf-8") as stream:
            for row in rows:
                row.update(field_set=args.field_set, remap=remap, nc_format=args.nc_format, seconds=seconds,
                           ready_unix=round(time.time(), 3))
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        converted.update(_same_file(p) for p in pending)
        done += rows
        exported += len(pending)
        last_progress = time.monotonic()
        for history in pending:
            print(f"EXPORT frame {order[str(history)] + exported - len(pending) + 1}/{args.follow_frames} "
                  f"{history.name} {seconds} s ({len(pending)} together, {len(plan)} grids)", flush=True)
    value = {
        "schema": RECEIPT_SCHEMA, "mode": "follow",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "forecast": str(forecast_dir), "mesh": str(mesh), "init": str(init),
        "simulation_start": args.simulation_start, "window": plan[0]["window"], "view": plan[0]["view"],
        "stack": plan, "field_set": args.field_set, "remap": remap, "nc_format": args.nc_format,
        "frames": done, "bytes": int(sum(row["bytes"] for row in done)),
        "seconds": round(time.monotonic() - started, 3),
    }
    (out / RECEIPT_NAME).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.keep_scratch:
        shutil.rmtree(scratch_root, ignore_errors=True)
    print(f"EXPORT follow done frames={exported} files={len(done)} bytes={value['bytes']} "
          f"seconds={value['seconds']}", flush=True)
    return 0


def _on_grid(path: Path, anchor: _dt.datetime, every_s: float) -> bool:
    valid = history_valid_time(path)
    if valid is None:
        raise ExportDoorError(f"--follow-every-minutes needs the valid time in the history name; {path.name} has none")
    offset = (valid - anchor).total_seconds()
    return offset >= 0 and abs(offset / every_s - round(offset / every_s)) < 1e-9


def _before(path: Path, start: _dt.datetime | None) -> bool:
    if start is None:
        return False
    valid = history_valid_time(path)
    if valid is None:
        raise ExportDoorError(f"--follow-from needs the valid time in the history name; {path.name} has none")
    return valid < start


def run_export(args: argparse.Namespace) -> int:
    from .products_door import (FORECAST_RECEIPT, discover_histories, forecast_receipt,
                                receipt_text, simulation_start)

    if getattr(args, "follow", False):
        return run_follow(args)
    started = time.monotonic()
    forecast_dir = Path(args.forecast_dir) if args.forecast_dir else None
    receipt = forecast_receipt(forecast_dir)
    if args.history:
        histories = [Path(p) for p in args.history]
    elif forecast_dir is not None:
        histories = discover_histories(forecast_dir)
    else:
        raise ExportDoorError("give the forecast folder (the forecast door's --out) or --history FILE...")
    if not histories:
        raise ExportDoorError(f"no history frames in {forecast_dir}; nothing to export")
    for path in histories:
        if not path.is_file():
            raise ExportDoorError(f"history frame {path} is missing")
    mesh_text = args.mesh or receipt_text(receipt, "mesh", "grid")
    if not mesh_text:
        raise ExportDoorError(
            f"no mesh: --mesh was not given and the forecast folder carries no {FORECAST_RECEIPT} "
            "naming the grid; the converter places the mesh's cells on the lat-lon grid from it")
    mesh = Path(mesh_text)
    if not mesh.is_file():
        raise ExportDoorError(f"the mesh {mesh} is missing; pass --mesh GRID.nc")
    init_text = args.init or receipt_text(receipt, "init", "path")
    init = Path(init_text) if init_text else None
    if init is not None and not init.is_file():
        raise ExportDoorError(
            f"the init {init} is missing; zgrid, the heights behind every pressure-level field, "
            "lives there; pass --init INIT.nc")
    start = simulation_start(args.simulation_start, forecast_dir, histories)
    if args.out:
        out = Path(args.out)
    elif forecast_dir is not None:
        out = forecast_dir / "export"
    else:
        raise ExportDoorError("no --out: with --history there is no forecast folder to file the export beside")
    scratch = Path(args.scratch) if args.scratch else out.parent / (out.name + ".export-scratch")
    render_door.refuse_scratch_inside_out(scratch, out)
    plan = export_plan(args.window, mesh, args.view)
    remap = render_door.resolve_remap(args.remap, composite=False)
    convert_exe = render_door.resolve_convert_exe(args.convert_exe)
    print(f"EXPORT frames={len(histories)} window={plan[0]['window']} grids={len(plan)} remap={remap} out={out}",
          flush=True)
    for entry in plan:
        if entry.get("why"):
            print(f"EXPORT GRID d{entry['domain']:02d} {entry['view']} {entry['window']}: {entry['why']}", flush=True)
    ready = _ready_rows(out)
    followed = frozenset(str(row.get("frame")) for row in ready if row.get("frame"))
    delivered: list[dict[str, Any]] = []
    converters: list[dict[str, Any]] = []
    for entry in plan:
        reuse = _reusable(ready, entry, field_set=args.field_set, remap=remap, nc_format=args.nc_format)
        taken = [reuse[_same_file(h)] for h in histories if _same_file(h) in reuse]
        todo = [h for h in histories if _same_file(h) not in reuse]
        rows: list[dict[str, Any]] = []
        if todo:
            rows, report, command = _convert_and_file(
                entry, todo, args=args, mesh=mesh, init=init, start=start, scratch=scratch, out=out,
                remap=remap, convert_exe=convert_exe, jobs=max(1, int(args.jobs)), replaceable=followed)
            if not rows:
                raise ExportDoorError("rw_mpas_convert finished and wrote no frame; nothing was exported")
            converters.append({"domain": entry["domain"], "command": command, "report": report})
            # Listed like a follower's frames, so an export step run again takes them instead of refusing them.
            out.mkdir(parents=True, exist_ok=True)
            with (out / READY_NAME).open("a", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps({**row, "field_set": args.field_set, "remap": remap,
                                             "nc_format": args.nc_format, "ready_unix": round(time.time(), 3)},
                                            sort_keys=True) + "\n")
        entry["converted"], entry["reused"] = len(rows), len(taken)
        entry["bytes"] = int(sum(int(row["bytes"]) for row in [*taken, *rows]))
        print(f"EXPORT GRID d{entry['domain']:02d} {entry['view']}: {len(rows)} converted, {len(taken)} taken from "
              f"the follower, {entry['bytes']} bytes", flush=True)
        delivered += [{k: row.get(k) for k in ("frame", "bytes", "sha256", "valid_time", "domain", "view")}
                      for row in [*taken, *rows]]
    delivered.sort(key=lambda row: (row["domain"], Path(row["frame"]).name))
    value = {
        "schema": RECEIPT_SCHEMA,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "forecast": str(forecast_dir) if forecast_dir else None,
        "histories": [str(p) for p in histories],
        "mesh": str(mesh),
        "init": str(init) if init else None,
        "simulation_start": start,
        "window": plan[0]["window"],
        "view": plan[0]["view"],
        "stack": plan,
        "field_set": args.field_set,
        "remap": remap,
        "nc_format": args.nc_format,
        "frames": delivered,
        "bytes": int(sum(int(row["bytes"]) for row in delivered)),
        # The first grid's converter, as before the stack; every grid's is in "converters".
        "converter": converters[0] if converters else None,
        "converters": converters,
        "seconds": round(time.monotonic() - started, 3),
    }
    # The any-point column store, every output hour (column_store.py): from the frames just delivered, one store per
    # domain of the stack (<out>/columns/dNN/).
    from .column_store import write as write_columns
    value["columns"] = write_columns([Path(row["frame"]) for row in delivered], out / "columns",
                                     field_set=args.field_set)
    out.mkdir(parents=True, exist_ok=True)
    (out / RECEIPT_NAME).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"EXPORT done frames={len(delivered)} bytes={value['bytes']} receipt={out / RECEIPT_NAME} "
          f"seconds={value['seconds']}", flush=True)
    return 0


#: The export a door runs beside its forecast: the export's own flags as a JSON list (``["--out", DIR, "--window",
#: "global", "--field-set", "surface"]``), the forecast folder, mesh, init, start and frame count being the door's.
LIVE_EXPORT_ENVIRONMENT = "WOOF_HEX_EXPORT_LIVE"
#: The CPU priority (nice) the live export runs at: the forecast is on the card and its host threads come first.
LIVE_EXPORT_NICE = 10


def live_export_command(forecast: Path, *, grid, init, start_time: str, frames: int,
                        environ=None) -> list[str] | None:
    """The ``export --follow`` command a door starts beside its forecast, or None when ``WOOF_HEX_EXPORT_LIVE``
    names no export (unset, empty or unreadable: the export then runs after the forecast, as before).

    THE BREAKAGE THIS PREVENTS (an Arizona 96 -> 3 km zoom globe, 36 h, 8 x RTX PRO 6000, 2026-10-30): the
    site's export ran after the forecast and its maps, 6 min 22 s on the host's CPUs while the 8 cards sat idle.
    Beside the forecast, each frame is exported the moment it is closed, and the export step after it takes those
    frames as they are (``export-ready.jsonl``)."""

    import sys

    text = (os.environ if environ is None else environ).get(LIVE_EXPORT_ENVIRONMENT, "").strip()
    if not text:
        return None
    try:
        flags = json.loads(text)
    except ValueError:
        return None
    if not isinstance(flags, list) or not all(isinstance(item, str) for item in flags):
        return None
    banned = {"--follow", "--mesh", "--init", "--simulation-start", "--follow-frames", "--history"}
    flags = [item for item in flags if item.split("=", 1)[0] not in banned]
    argv = [sys.executable, "-m", "woof.hex.cli", "export", str(forecast), "--follow", "--mesh", str(grid),
            "--init", str(init), "--simulation-start", str(start_time), "--follow-frames", str(int(frames)), *flags]
    nice = shutil.which("nice")
    return [nice, "-n", str(LIVE_EXPORT_NICE), *argv] if nice else argv


class LiveExport:
    """``export --follow`` beside a door's forecast (:func:`live_export_command`). Nothing here can fail the run: a
    follower that fails, lags or is stopped leaves the export step after the forecast to convert what it did not."""

    def __init__(self, forecast: Path, *, grid, init, start_time: str, frames: int, log: Path):
        self.argv = live_export_command(Path(forecast), grid=grid, init=init, start_time=start_time, frames=frames)
        self.log = Path(log)
        self.proc = None
        self.began = None

    def start(self) -> "LiveExport":
        if self.argv is None:
            return self
        import subprocess

        try:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with self.log.open("w", encoding="utf-8") as stream:
                self.proc = subprocess.Popen(self.argv, stdout=stream, stderr=subprocess.STDOUT)
            self.began = time.monotonic()
            print(f"STAGE live export beside the forecast (log {self.log})", flush=True)
        except OSError as error:
            print(f"STAGE live export not started ({error}); the export step converts every frame", flush=True)
            self.proc = None
        return self

    def finish(self, *, forecast_ok: bool, timeout: float = 900.0) -> dict[str, Any]:
        """The forecast has ended: let the follower export the frames it closed last (up to ``timeout`` s), or stop it
        at once when the forecast failed."""

        if self.proc is None:
            return {"status": "off" if self.argv is None else "not-started"}
        import subprocess

        if not forecast_ok:
            self.proc.terminate()
        try:
            rc = self.proc.wait(timeout=timeout if forecast_ok else 30)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                rc = self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                rc = self.proc.wait()
        return {"status": "done" if rc == 0 else "stopped", "returncode": rc, "log": str(self.log),
                "seconds": round(time.monotonic() - (self.began or time.monotonic()), 1)}


def add_export_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("forecast_dir", nargs="?", default=None, metavar="RUN_DIR",
                        help="the forecast door's --out folder (its receipt names the mesh and init)")
    parser.add_argument("--history", nargs="+", default=None, metavar="FILE",
                        help="history frames, instead of a forecast folder")
    parser.add_argument("--mesh", default=None, metavar="FILE", help="the run's grid (default: the receipt's)")
    parser.add_argument("--init", default=None, metavar="FILE", help="the run's init (default: the receipt's)")
    parser.add_argument("--out", default=None, metavar="DIR", help="default: <RUN_DIR>/export")
    parser.add_argument("--window", default="global", type=render_door.window_argument,
                        help="global (the sphere at the mesh's spacing; on a variable-resolution mesh a stack "
                             "of grids, each band of the mesh at its own spacing: wrfout_d01 the sphere at the "
                             "outer spacing, d02.. each ring, the last the refined core), global-0.25, mesh, focus, "
                             "region:CLAT,CLON,W_KM,H_KM, region-latlon:CLAT,CLON,W_KM,H_KM (a regular "
                             "lat-lon box at any latitude: the window a WOOF domain takes as its parent, "
                             "with --field-set full) or lambert:... (default global)")
    parser.add_argument("--view", default=None, metavar="NAME",
                        help="folder name under --out (default: the window's view name)")
    parser.add_argument("--field-set", default="full", choices=("full", "surface"))
    parser.add_argument("--remap", default="product", choices=("product", "nearest"),
                        help="product: linear state, conservative accumulations (default); nearest: "
                             "the nearest cell's value")
    parser.add_argument("--nc-format", default="cdf5", choices=("cdf2", "cdf5"),
                        help="NetCDF variant (default cdf5: a global frame passes the CDF-2 size limit)")
    parser.add_argument("--simulation-start", default=None, metavar="YYYY-MM-DD_HH:MM:SS")
    parser.add_argument("--model-label", default=render_door.DEFAULT_MODEL_LABEL)
    parser.add_argument("--jobs", type=int, default=1, metavar="N", help="frames converted at once")
    parser.add_argument("--scratch", default=None, metavar="DIR")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--convert-exe", default=None, metavar="FILE")
    parser.add_argument("--follow", action="store_true",
                        help="export each frame as the running forecast closes it (history-ready.jsonl), "
                             "listing it in <out>/export-ready.jsonl; needs --mesh, --init, "
                             "--simulation-start and --follow-frames")
    parser.add_argument("--follow-frames", type=int, default=None, metavar="N",
                        help="with --follow: the number of history frames the forecast writes")
    parser.add_argument("--follow-timeout-minutes", type=float, default=240.0, metavar="MIN",
                        help="with --follow: stop when the forecast lists no new frame for this long (default 240)")
    parser.add_argument("--follow-poll-seconds", type=float, default=1.0, metavar="S")
    parser.add_argument("--follow-from", default=None, metavar="YYYY-MM-DD_HH:MM:SS",
                        help="with --follow: skip the forecast's frames valid before this time "
                             "(a nest added while the parent runs starts here)")
    parser.add_argument("--follow-every-minutes", type=float, default=None, metavar="M",
                        help="with --follow: export only frames valid at --follow-from (or the simulation "
                             "start) plus a whole multiple of M minutes, the consumer's boundary interval "
                             "(default: every frame)")


def add_export_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "export",
        help="a forecast's history on a lat-lon (or Lambert) grid as NetCDF, without drawing",
        description=__doc__.split("\n\n")[0])
    add_export_arguments(parser)
    parser.set_defaults(handler=run_export)
