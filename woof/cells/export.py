"""``woof cells export``: WOOF-ArWen history as titan-rs volumes.

titan reads Cartesian volumes on one height ladder shared by every
column; a model writes its fields on terrain-following levels whose
heights differ column by column and frame by frame.  The exporter is
that resampling and nothing more:

* **The ladder** is a fixed set of cell-centre heights above sea level,
  by default every 250 m from 250 m to 18 000 m (72 levels), spelled
  ``BOTTOM:TOP:STEP`` in metres.  It is the volume's ``z_levels_m``, so
  the choice travels inside the file the engine reads.
* **The rule** is linear interpolation in the field's own units (dBZ
  for reflectivity, degrees C for temperature) between the two model
  mass levels that bracket each ladder height.  A ladder level below the
  lowest mass level but at or above the terrain takes the lowest level's
  value; a level below the terrain, or above the highest mass level, is
  missing (NaN), which titan treats as no data rather than as no echo.
  Linear-in-dBZ is the convention the radar CAPPI gridders titan was
  validated against use, and the model's dBZ field is already
  logarithmic, so interpolating it in linear Z would weight the bright
  cell of each pair.
* **The horizontal grid** is the model's own: cell ``(x, y)`` of the
  volume is mass point ``(west_east=x, south_north=y)``, the spacing is
  the file's ``DX``/``DY``, and the origin is placed so the domain's
  centre is ``(0, 0)`` in the volume's projected metres.  The map
  projection the model ran on is recorded in the volume's projection
  string (``MAP_PROJ``, true latitudes, standard longitude, centre) and
  in the receipt, and the catalog puts titan's metres back on the
  model's ``XLAT``/``XLONG`` by that rule.

Output is deterministic: the same frames on the same ladder give the
same bytes, and a test pins the stream's digest.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from woof.cells.columns import (ColumnFrame, ColumnsError, expand_inputs,
                                 open_frames, partition_inputs)
from woof.cells.titan_volume import StreamWriter, TitanVolume

#: The default ladder, metres above sea level: bottom, top, step.
DEFAULT_LADDER = "250:18000:250"

#: What the receipt and the volume's metadata say the resampling is.
INTERPOLATION_RULE = (
    "linear in the field's own units between the bracketing model mass "
    "levels; constant below the lowest mass level down to the terrain; "
    "NaN (missing) below the terrain and above the highest mass level")

#: The stream every export writes, and the receipt beside it.
STREAM_NAME = "input.tfs"
RECEIPT_NAME = "export-receipt.json"
RECEIPT_SCHEMA = "gpuwm-cells-export/1"


class ExportError(RuntimeError):
    """An export that cannot be made, named by what stops it."""


def parse_ladder(text: str) -> np.ndarray:
    """``BOTTOM:TOP:STEP`` in metres -> strictly increasing centre heights."""

    try:
        bottom, top, step = (float(part) for part in str(text).split(":"))
    except ValueError:
        raise ExportError(
            f"--ladder {text!r} is not BOTTOM:TOP:STEP in metres, e.g. "
            f"{DEFAULT_LADDER}") from None
    if step <= 0 or top <= bottom:
        raise ExportError(
            f"--ladder {text!r}: STEP must be positive and TOP above BOTTOM")
    count = int(np.floor((top - bottom) / step + 1e-9)) + 1
    ladder = bottom + step * np.arange(count, dtype=np.float64)
    if count < 2:
        raise ExportError(
            f"--ladder {text!r} has {count} level; titan needs a column")
    return ladder


def ladder_text(ladder: np.ndarray) -> str:
    step = float(ladder[1] - ladder[0]) if len(ladder) > 1 else 0.0
    return f"{ladder[0]:g}:{ladder[-1]:g}:{step:g}"


def interpolate_columns(values: np.ndarray, z_mass: np.ndarray,
                        terrain_m: np.ndarray, ladder: np.ndarray
                        ) -> np.ndarray:
    """Resample ``values`` (nz, ny, nx) at ``z_mass`` onto ``ladder``.

    Vectorised over the horizontal grid one ladder level at a time: the
    bracketing index is the count of mass levels below the target height
    (mass heights increase upward in every column), and the two-point
    linear weight follows from the bracketing heights.
    """

    values = np.asarray(values, dtype=np.float32)
    z = np.asarray(z_mass, dtype=np.float32)
    terrain = np.asarray(terrain_m, dtype=np.float32)
    nz = z.shape[0]
    if nz < 2:
        raise ExportError("a column needs at least two mass levels to bracket")
    out = np.full((len(ladder), *z.shape[1:]), np.nan, dtype=np.float32)
    lowest = values[0]
    for index, height in enumerate(ladder):
        h = np.float32(height)
        below = (z < h).sum(axis=0)
        lower = np.clip(below - 1, 0, nz - 2)
        upper = lower + 1
        z0 = np.take_along_axis(z, lower[None], axis=0)[0]
        z1 = np.take_along_axis(z, upper[None], axis=0)[0]
        v0 = np.take_along_axis(values, lower[None], axis=0)[0]
        v1 = np.take_along_axis(values, upper[None], axis=0)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = (h - z0) / (z1 - z0)
        level = v0 + weight * (v1 - v0)
        under_first = below == 0
        level = np.where(under_first, np.where(terrain <= h, lowest, np.nan),
                         level)
        level = np.where(below >= nz, np.nan, level)
        out[index] = level
    return out


def projection_string(frame: ColumnFrame) -> str:
    """The volume's projection field: the model's own map, spelled out."""

    parts = ["ARWEN_GRID"]
    for name, value in frame.projection.items():
        parts.append(f"{name}={value:g}" if isinstance(value, float)
                     else f"{name}={value}")
    parts.append("origin=domain_centre")
    parts.append("x=west_east*DX y=south_north*DY")
    return " ".join(parts)


def frame_to_volume(frame: ColumnFrame, ladder: np.ndarray, *,
                    temperature: bool = True) -> TitanVolume:
    """One column frame as a titan volume on ``ladder``."""

    if not np.isfinite(frame.dx_m) or frame.dx_m <= 0:
        raise ExportError(
            f"{frame.path}: DX is {frame.dx_m!r}; titan needs a positive "
            f"horizontal spacing to measure area and volume")
    reflectivity = interpolate_columns(
        frame.refl_dbz, frame.z_mass, frame.terrain_m, ladder)
    optional: dict[str, np.ndarray] = {}
    if temperature:
        optional["temperature"] = interpolate_columns(
            frame.temperature_k - np.float32(273.15), frame.z_mass,
            frame.terrain_m, ladder)
    nx, ny = frame.nx, frame.ny
    domain = frame.domain or "native_grid"
    return TitanVolume(
        timestamp_ms=frame.timestamp_ms, nx=nx, ny=ny,
        z_levels_m=np.asarray(ladder, dtype=np.float64),
        origin_x_m=-0.5 * nx * frame.dx_m, origin_y_m=-0.5 * ny * frame.dy_m,
        dx_m=float(frame.dx_m), dy_m=float(frame.dy_m),
        projection=projection_string(frame),
        source=f"{domain} {frame.path.name} t{frame.time_index}",
        reflectivity=reflectivity, optional=optional)


@dataclass
class ExportedFrame:
    path: str
    time_index: int
    valid: str
    timestamp_ms: int
    source: str
    finite_cells: int
    max_dbz: float | None
    seconds: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_series(inputs, out_dir: Path | str, *, ladder: str = DEFAULT_LADDER,
                  temperature: bool = True,
                  progress: Callable[[str], None] | None = None,
                  executable: Path | None = None) -> dict:
    """Write ``input.tfs`` and its receipt for a wrfout series.

    Frames are pushed to the stream as each is resampled, so the peak is
    one frame; the frame count is taken up front from the files'
    ``Time`` dimensions, which is what the stream header needs.
    """

    say = progress or (lambda _text: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    levels = parse_ladder(ladder)
    paths = expand_inputs(inputs)
    started = time.perf_counter()
    from woof import netcdf_bridge
    usable, skipped = partition_inputs(paths, executable=executable)
    for entry in skipped:
        say(f"skipped {Path(entry['path']).name}: {entry['reason']} (initial frame)")
    if not usable:
        raise ExportError(
            f"none of the {len(paths)} wrfout files carries REFL_10CM, so "
            f"there is no reflectivity to identify a cell in; run with a "
            f"microphysics scheme that writes it (every WOOF-ArWen scheme does)")
    paths = usable
    counts: list[int] = []
    for path in paths:
        with netcdf_bridge.open_dataset(path, executable=executable) as dataset:
            counts.append(int(dataset.variables["REFL_10CM"].shape[0]))
    total = sum(counts)
    stream = out / STREAM_NAME
    writer = StreamWriter(stream, total)
    frames: list[ExportedFrame] = []
    grid: dict | None = None
    seen: set[int] = set()
    try:
        for path in paths:
            for frame in open_frames(path, executable=executable):
                tick = time.perf_counter()
                if frame.timestamp_ms in seen:
                    raise ExportError(
                        f"{path} repeats the valid time {frame.stamp} already "
                        f"exported; titan refuses two frames at one instant, "
                        f"so name each file once")
                seen.add(frame.timestamp_ms)
                volume = frame_to_volume(frame, levels, temperature=temperature)
                writer.push(volume)
                finite = int(np.isfinite(volume.reflectivity).sum())
                peak = (float(np.nanmax(volume.reflectivity)) if finite
                        else None)
                if grid is None:
                    grid = {
                        "domain": frame.domain, "grid_id": frame.grid_id,
                        "nx": frame.nx, "ny": frame.ny,
                        "model_levels": frame.nz, "dx_m": frame.dx_m,
                        "dy_m": frame.dy_m, "projection": frame.projection,
                        "projection_string": volume.projection,
                        "origin_x_m": volume.origin_x_m,
                        "origin_y_m": volume.origin_y_m,
                        "lat_range": [float(frame.lat.min()), float(frame.lat.max())],
                        "lon_range": [float(frame.lon.min()), float(frame.lon.max())],
                    }
                elif (frame.nx, frame.ny) != (grid["nx"], grid["ny"]):
                    raise ExportError(
                        f"{path} is {frame.nx}x{frame.ny} but the series "
                        f"started at {grid['nx']}x{grid['ny']}; one titan "
                        f"stream is one grid, so export each domain on its own")
                frames.append(ExportedFrame(
                    path=str(path), time_index=frame.time_index,
                    valid=frame.stamp, timestamp_ms=frame.timestamp_ms,
                    source=volume.source, finite_cells=finite,
                    max_dbz=peak, seconds=time.perf_counter() - tick))
                say(f"exported {frame.stamp} ({frame.domain or 'native_grid'}, "
                    f"max {peak if peak is None else round(peak, 1)} dBZ, "
                    f"{frames[-1].seconds:.1f} s)")
    except Exception:
        writer.abandon()
        raise
    writer.finish()
    seconds = time.perf_counter() - started
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stream": str(stream),
        "stream_sha256": _sha256(stream),
        "stream_bytes": stream.stat().st_size,
        "reader": "rw_netcdf via woof.netcdf_bridge",
        "ladder": {
            "spelling": ladder_text(levels),
            "levels_m_msl": [float(v) for v in levels],
            "count": int(len(levels)),
            "reference": "metres above sea level, cell centres",
        },
        "interpolation": INTERPOLATION_RULE,
        "fields": {
            "reflectivity": "REFL_10CM, dBZ, on the ladder",
            **({"temperature": "T and P+PB -> temperature, degrees C, on the ladder"}
               if temperature else {}),
        },
        "grid": grid,
        "frames": [frame.__dict__ for frame in frames],
        "frame_count": len(frames),
        "skipped": skipped,
        "wall_seconds": round(seconds, 3),
    }
    (out / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    say(f"wrote {stream} ({len(frames)} frames, {seconds:.1f} s)")
    return receipt


__all__ = [
    "DEFAULT_LADDER", "ExportError", "INTERPOLATION_RULE", "RECEIPT_NAME",
    "STREAM_NAME", "export_series", "frame_to_volume", "interpolate_columns",
    "ladder_text", "parse_ladder", "projection_string",
]
