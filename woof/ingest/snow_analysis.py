"""Observed snow at the start: an analysis replaces the forcing's snowpack.

Why this module exists
----------------------
Spring snow started far too light from every forcing.  Measured on the raw
source grids against SNOTEL (2023-04-27 12Z; ERA5 2023-04-12 12Z):

* GFS 0.25 deg: 4.0 mm SWE at 32 Sierra sites (observed 1242 mm) and 7.9 mm
  at 51 San Juan sites (observed 530 mm);
* HRRR f00: 531 mm in the Sierra (43 %) at a raw density of 294 kg/m3
  (observed 507), and 20 mm in the San Juans (4 %);
* ERA5: 271 mm at 43 Sierra sites (observed 1188 mm).

Our ingest carries each source faithfully (initial SNOW over raw source SWE
is 1.00 on the ERA5 route; the HRRR and GFS starts match their raw grids
within interpolation), and the source orography sits within 50-300 m of the
sites, so neither units, field choice nor elevation explain it: the forcings
themselves hold that little snow.  A Noah-MP forecast started on them melts
a veneer instead of the April pack.

What it does
------------
Rows of ``woof/data/snow/snow-analysis.v1.json`` (SNODAS over CONUS today)
name an observed daily snow analysis, where to fetch it and how to read it.
For the newest analysis at or before the start (within the row's
``max_age_hours``), every target land cell the analysis covers takes the
analysis's grid-cell mean snow water (kg m-2) and snow depth (m), averaged
over the analysis pixels inside the cell's footprint.  Cells it does not
cover keep the forcing's snow unchanged, with the forcing's own depth rule.

``WOOF_SNOW_ANALYSIS=off`` (or ``source``) is the opt-out: the forcing's
snow, byte-identical to before this module existed.  An analysis that
cannot be fetched or read falls back to the forcing's snow for that run and
says so on stderr; it never stops a preparation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import gzip
import json
import math
import os
from pathlib import Path
import sys
import tarfile
from typing import Mapping

import numpy as np

_TABLE = Path(__file__).resolve().parent.parent / "data" / "snow" / "snow-analysis.v1.json"
_SCHEMA = "gpuwm-snow-analysis-v1"
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
#: real.exe's depth rule for snow water with no depth (module_initialize_real.F
#: 517-543): 5:1 liquid-to-snow, a 200 kg m-3 initial density.  Written as
#: ``snow / 1000.0 * 5.0`` exactly as woof.ingest.soil applies it.
_DEPTH_RULE_DIVISOR = 1000.0
_DEPTH_RULE_FACTOR = 5.0

_RULES: dict | None = None


def load_rules(path: Path | None = None) -> dict:
    global _RULES
    if path is None and _RULES is not None:
        return _RULES
    rules = json.loads(Path(path or _TABLE).read_text(encoding="utf-8"))
    if rules.get("schema") != _SCHEMA:
        raise ValueError(f"{path or _TABLE}: schema is not {_SCHEMA}")
    for row in rules["analyses"]:
        if row.get("format") != "nohrsc-raster-v1":
            raise ValueError(f"snow analysis {row.get('id')}: format "
                             f"{row.get('format')!r} has no reader")
        if set(row["members"]) != {"snow_water", "snow_depth"}:
            raise ValueError(f"snow analysis {row['id']}: members must be "
                             "snow_water and snow_depth")
    if path is None:
        _RULES = rules
    return rules


def _note(message: str, *, warning: bool = False) -> None:
    print(("warning: " if warning else "note: ") + message, file=sys.stderr, flush=True)


def _disabled(rules) -> bool:
    return os.environ.get(rules["opt_out_env"], "").strip().lower() in set(
        rules.get("opt_out_values", ("off",)))


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _fill(template: str, date: datetime) -> str:
    # %b is locale dependent; the archive's folder names are English.
    return template.replace("{date:%b}", _MONTHS[date.month - 1]).format(date=date)


def analysis_times(row, start: datetime) -> list[datetime]:
    """Every analysis at or before ``start`` within the row's max age, newest first.

    The newest one may not be published yet: NSIDC posts the 06Z SNODAS
    analysis at about 14:00Z (measured 2026-11-07 and 11-08, Last-Modified
    14:00:24Z), so a 12Z start asks for an analysis that is not on the
    archive.  The caller falls back to the previous one instead of dropping
    to the forcing's snow (defect: every real-time 06Z-13Z start lost the
    analysis to a 404).
    """

    start = _utc(start)
    candidate = start.replace(hour=int(row["analysis_hour_utc"]), minute=0,
                              second=0, microsecond=0)
    if candidate > start:
        candidate -= timedelta(days=1)
    first = datetime.fromisoformat(row["valid_from"].replace("Z", "+00:00"))
    out = []
    while (start - candidate <= timedelta(hours=float(row["max_age_hours"]))
           and candidate >= first):
        out.append(candidate)
        candidate -= timedelta(days=1)
    return out


def analysis_time(row, start: datetime) -> datetime | None:
    """The newest analysis at or before ``start``, or None past max age."""

    times = analysis_times(row, start)
    return times[0] if times else None


def _cache_root() -> Path:
    explicit = os.environ.get("WOOF_SNOW_ANALYSIS_CACHE")
    if explicit:
        return Path(explicit)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "woof" / "snow-analysis"


def _fetch(row, date: datetime, root: Path) -> Path:
    folder = root / row["id"]
    destination = folder / _fill(row["transport"]["file"], date)
    if destination.is_file() and tarfile.is_tarfile(destination):
        return destination
    folder.mkdir(parents=True, exist_ok=True)
    from urllib.request import Request, urlopen

    url = _fill(row["transport"]["url"], date)
    part = destination.with_name(destination.name + f".part{os.getpid()}")
    request = Request(url, headers={"User-Agent": "gpuwm-snow-analysis"})
    with urlopen(request, timeout=120) as response, part.open("wb") as out:
        while True:
            block = response.read(1 << 22)
            if not block:
                break
            out.write(block)
    if not tarfile.is_tarfile(part):
        part.unlink()
        raise ValueError(f"{url} is not a tar archive")
    os.replace(part, destination)
    return destination


def _header(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def read_nohrsc_member(archive: tarfile.TarFile, stem: str):
    """``(values in metres (ny, nx) float64, NaN for no data; header dict)``."""

    names = archive.getnames()

    def member(suffix):
        hits = [n for n in names if Path(n).name.startswith(stem) and n.endswith(suffix)]
        if not hits:
            raise KeyError(f"{stem}{suffix} is not in the archive")
        return gzip.decompress(archive.extractfile(hits[0]).read())

    head = _header(member(".txt.gz").decode("latin-1"))
    ny, nx = int(head["Number of rows"]), int(head["Number of columns"])
    if int(head["Data bytes per pixel"]) != 2:
        raise ValueError(f"{stem}: only 2-byte rasters are read")
    raw = np.frombuffer(member(".dat.gz"), ">i2")
    if raw.size != ny * nx:
        raise ValueError(f"{stem}: {raw.size} pixels, header says {ny} x {nx}")
    raw = raw.reshape(ny, nx)
    units = head["Data units"].lower()
    if not units.startswith("meters"):
        raise ValueError(f"{stem}: units {head['Data units']!r} are not metres")
    divisor = float(units.split("/", 1)[1]) if "/" in units else 1.0
    slope = float(head.get("Data slope", 1.0))
    intercept = float(head.get("Data intercept", 0.0))
    nodata = float(head["No data value"])
    values = (raw.astype(np.float64) * slope + intercept) / divisor
    values[(raw == nodata) | (raw < 0)] = np.nan
    return values, head


def _cell_spacing_m(lat: np.ndarray, lon: np.ndarray) -> float:
    """The target grid spacing, from its own mass points (median neighbour)."""

    if lat.ndim != 2 or min(lat.shape) < 2:
        return 0.0
    p1, p2 = np.radians(lat[:, :-1]), np.radians(lat[:, 1:])
    dl = np.radians(lon[:, 1:] - lon[:, :-1])
    cosine = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return float(np.median(6371000.0 * np.arccos(np.clip(cosine, -1.0, 1.0))))


def sample_footprint(values: np.ndarray, head: Mapping, lat, lon, spacing_m: float):
    """Mean of the valid analysis pixels inside each target cell's footprint.

    The footprint is ``floor(spacing / (2 * pixel))`` pixels each side of the
    nearest pixel along each axis, pixel widths in metres at the cell's
    latitude.  Returns ``(mean (lat's shape), found mask)``.
    """

    lat = np.asarray(lat, dtype=np.float64)
    lon = ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0
    x0 = float(head["Benchmark x-axis coordinate"])
    y0 = float(head["Benchmark y-axis coordinate"])
    rx = float(head["X-axis resolution"])
    ry = float(head["Y-axis resolution"])
    ny, nx = values.shape
    col = np.rint((lon - x0) / rx).astype(np.int64)
    row = np.rint((y0 - lat) / ry).astype(np.int64)
    inside = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    py = math.radians(ry) * 6371000.0
    px = math.radians(rx) * 6371000.0 * np.maximum(np.cos(np.radians(lat)), 0.05)
    hy = int(spacing_m // (2.0 * py))
    hx_cell = np.floor(spacing_m / (2.0 * px)).astype(np.int64)
    hx = int(hx_cell.max()) if hx_cell.size else 0
    total = np.zeros(lat.shape)
    count = np.zeros(lat.shape)
    for dj in range(-hy, hy + 1):
        jj = np.clip(row + dj, 0, ny - 1)
        for di in range(-hx, hx + 1):
            ii = np.clip(col + di, 0, nx - 1)
            v = values[jj, ii]
            ok = inside & (abs(di) <= hx_cell) & np.isfinite(v)
            total += np.where(ok, v, 0.0)
            count += ok
    found = count > 0
    return np.where(found, total / np.where(found, count, 1.0), np.nan), found


def apply_snow_analysis(fields, *, context, landmask=None, lake_mask=None):
    """Replace the forcing's SNOW/SNOWH with an observed analysis where it covers.

    ``context`` is :class:`woof.ingest.land_state.LandStateContext` (where
    and when).  Returns ``(fields, receipt)``; ``fields`` is the input
    mapping itself whenever nothing is replaced.
    """

    rules = load_rules()
    if _disabled(rules):
        return fields, {"analysis": f"off ({rules['opt_out_env']})"}
    if context is None:
        return fields, {"analysis": "none: this route passed no domain context"}
    from woof.ingest.soil import _host

    start = _utc(context.valid_time)
    reasons = []
    for row in rules["analyses"]:
        dates = analysis_times(row, start)
        if not dates:
            reasons.append(f"{row['id']}: no analysis within {row['max_age_hours']} h "
                           f"before {start:%Y-%m-%d %H}Z (archive from {row['valid_from']})")
            continue
        box = row["coverage_box"]
        lat = np.asarray(context.latitude, dtype=np.float64)
        lon = ((np.asarray(context.longitude, dtype=np.float64) + 180.0) % 360.0) - 180.0
        if not ((lat >= box["lat"][0]) & (lat <= box["lat"][1])
                & (lon >= box["lon"][0]) & (lon <= box["lon"][1])).any():
            reasons.append(f"{row['id']}: the domain lies outside its coverage")
            continue
        read = None
        tried = []
        for date in dates:
            try:
                path = _fetch(row, date.replace(tzinfo=None), _cache_root())
                with tarfile.open(path) as archive:
                    swe_m, head = read_nohrsc_member(archive, _fill(row["members"]["snow_water"], date))
                    depth_m, head_d = read_nohrsc_member(archive, _fill(row["members"]["snow_depth"], date))
                if swe_m.shape != depth_m.shape or any(
                        head[k] != head_d[k] for k in ("Benchmark x-axis coordinate",
                                                       "Benchmark y-axis coordinate",
                                                       "X-axis resolution", "Y-axis resolution")):
                    raise ValueError("snow water and snow depth rasters are on different grids")
                spacing = _cell_spacing_m(lat, lon)
                swe, found = sample_footprint(swe_m, head, lat, lon, spacing)
                depth, _ = sample_footprint(depth_m, head, lat, lon, spacing)
            except Exception as error:  # try the previous analysis, then the forcing, loudly
                tried.append(f"{date:%Y-%m-%d %H}Z {type(error).__name__}: {error}")
                continue
            read = date
            break
        if read is None:
            reasons.append(f"{row['id']}: " + "; ".join(tried))
            _note(f"snow: analysis {row['id']} unavailable for this run "
                  f"({'; '.join(tried)}); the forcing's snow stays", warning=True)
            continue
        date = read
        if tried:
            _note(f"snow: {row['id']} newest analysis not readable ({'; '.join(tried)}); "
                  f"using {date:%Y-%m-%d %H}Z", warning=True)
        if "SNOW" in fields:
            snow = _host(fields["SNOW"])
            source_field = "SNOW"
        elif "SNOW_EC" in fields:
            snow = 1000.0 * _host(fields["SNOW_EC"])
            source_field = "SNOW_EC"
        else:
            snow = np.zeros(lat.shape)
            source_field = None
        if snow.shape != lat.shape:
            reasons.append(f"{row['id']}: snow field shape {snow.shape} is not the grid {lat.shape}")
            continue
        if "SNOWH" in fields:
            snowh = _host(fields["SNOWH"])
        else:
            # The forcing's own depth rule, applied now, so a cell the
            # analysis does not cover keeps exactly the depth it had.
            snowh = snow / _DEPTH_RULE_DIVISOR * _DEPTH_RULE_FACTOR
        land_source = landmask if landmask is not None else fields.get(
            "LANDMASK", fields.get("LANDSEA"))
        land = (np.ones(lat.shape, bool) if land_source is None
                else _host(land_source) >= 0.5)
        if lake_mask is not None:
            land &= ~_host(lake_mask).astype(bool)
        covered = land & found
        if not covered.any():
            reasons.append(f"{row['id']}: covers none of this domain's land")
            continue
        new_swe = np.maximum(np.nan_to_num(swe) * 1000.0, 0.0)
        new_depth = np.where(new_swe > 0.0, np.maximum(np.nan_to_num(depth), 0.0), 0.0)
        # A footprint mean can pair snow water with too little depth (a
        # density above water); real.exe's 200 kg m-3 rule stands in there.
        thin = (new_swe > 0.0) & (new_depth * 1000.0 < new_swe)
        new_depth = np.where(thin, new_swe / _DEPTH_RULE_DIVISOR * _DEPTH_RULE_FACTOR,
                             new_depth)
        before = float(snow[covered].mean())
        out = dict(fields)
        out["SNOW"] = np.where(covered, new_swe, snow)
        out["SNOWH"] = np.where(covered, new_depth, snowh)
        dense = covered & (new_depth > 0.05)
        receipt = {
            "analysis": row["id"],
            "analysis_time": f"{date:%Y-%m-%dT%H}Z",
            "newer_analyses_unreadable": tried,
            "file": str(path),
            "replaced": source_field,
            "footprint_spacing_m": round(spacing, 1),
            "land_cells_covered": int(np.count_nonzero(covered)),
            "land_cells_not_covered": int(np.count_nonzero(land & ~found)),
            "swe_mean_before_kg_m2": round(before, 2),
            "swe_mean_after_kg_m2": round(float(new_swe[covered].mean()), 2),
            "density_median_kg_m3": (round(float(np.median(new_swe[dense] / new_depth[dense])), 1)
                                     if dense.any() else None),
            "depth_rule_cells": int(np.count_nonzero(thin & covered)),
        }
        _note(f"snow: {row['id']} {date:%Y-%m-%d %H}Z on {receipt['land_cells_covered']} of "
              f"{int(np.count_nonzero(land))} land cell(s); mean SWE {before:.1f} -> "
              f"{receipt['swe_mean_after_kg_m2']:.1f} kg/m2 (woof/data/snow/snow-analysis.v1.json)")
        return out, receipt
    return fields, {"analysis": "none: " + "; ".join(reasons)}


__all__ = ["analysis_time", "analysis_times", "apply_snow_analysis", "load_rules",
           "read_nohrsc_member", "sample_footprint"]
