"""``woof cells catalog``: one row per cell per frame, titan plus WOOF-ArWen.

titan supplies the object: identity, track, age, area, centroid, echo
tops, trend, forecast footprint.  WOOF-ArWen supplies what a radar never
sees and a model always has: the updraft, the cloud's top and base and
their temperatures, the freezing and supercooled levels, the
supercooled liquid water.  The catalog joins the two on the cell's own
footprint -- the grid columns titan's 3-D voxels project onto -- so the
model attributes are sampled where the cell is, not at a box around it.

Every column name carries its unit; :data:`COLUMNS` carries the
provenance beside it, and the JSON catalog embeds that table so a
reader never has to guess which program a number came from.  titan's
numbers pass through untouched.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

from woof.cells.columns import (ColumnFrame, expand_inputs, open_frames,
                                 partition_inputs)
from woof.cells.titan import Bundle
from woof.cells.titan_volume import cell_xyz

CATALOG_SCHEMA = "gpuwm-cells-catalog/1"

#: Feet per minute in one metre per second.
FT_PER_MIN_PER_MPS = 196.85

#: Cloud: any condensate above this mixing ratio, kg/kg.
CLOUD_THRESHOLD_KG_KG = 1.0e-6

#: The isotherm levels the cell catalog records, degrees C.
ISOTHERMS_C = (0.0, -5.0, -10.0, -15.0, -20.0)

#: Column name -> (unit, provenance).  The order here is the CSV order.
COLUMNS: dict[str, tuple[str, str]] = {
    "valid_time": ("UTC, ISO 8601", "WOOF-ArWen frame valid time"),
    "timestamp_ms": ("ms since 1970-01-01T00:00:00Z", "titan frame timestamp"),
    "frame_index": ("1", "position in the exported series, from 0"),
    "domain": ("text", "WOOF-ArWen GRID_ID as dNN"),
    "object_id": ("1", "titan object id within the frame"),
    "track_id": ("1", "titan track id (identity through time)"),
    "is_birth": ("bool", "titan: first observation of this track"),
    "track_status": ("text", "titan tracks.json status"),
    "lifetime_so_far_s": ("s", "titan: frame time minus the track's first observation"),
    "track_seen_scans": ("count", "titan tracks.json seen_scans (whole track)"),
    "parent_track_ids": ("list", "titan lineage: tracks this one split or merged from"),
    "child_track_ids": ("list", "titan lineage: tracks that split or merged from this one"),
    "centroid_x_m": ("m, east of the domain centre", "titan geometric centroid"),
    "centroid_y_m": ("m, north of the domain centre", "titan geometric centroid"),
    "centroid_lat": ("degrees N", "WOOF-ArWen XLAT at titan's centroid (bilinear)"),
    "centroid_lon": ("degrees E", "WOOF-ArWen XLONG at titan's centroid (bilinear)"),
    "projected_area_km2": ("km^2", "titan"),
    "volume_km3": ("km^3", "titan"),
    "max_dbz": ("dBZ", "titan"),
    "mean_dbz": ("dBZ", "titan"),
    "base_height_m_msl": ("m MSL", "titan base_height_m"),
    "top_height_m_msl": ("m MSL", "titan top_height_m (envelope top)"),
    "echo_top_18dbz_m_msl": ("m MSL", "titan"),
    "echo_top_30dbz_m_msl": ("m MSL", "titan"),
    "echo_top_40dbz_m_msl": ("m MSL", "titan"),
    "echo_top_50dbz_m_msl": ("m MSL", "titan"),
    "vil_max_kg_m2": ("kg/m^2", "titan"),
    "mesh_mm": ("mm", "titan maximum estimated hail size"),
    "mean_temperature_c": ("degrees C", "titan, over the exported temperature field"),
    "motion_east_mps": ("m/s", "titan tracker (Kalman) velocity state at this scan; see trend_* for the fitted motion"),
    "motion_north_mps": ("m/s", "titan tracker (Kalman) velocity state at this scan"),
    "motion_speed_mps": ("m/s", "titan tracker (Kalman) velocity state at this scan"),
    "motion_direction_to_deg": ("degrees clockwise from north, toward", "titan tracker (Kalman) velocity state at this scan"),
    "trend_east_mps": ("m/s", "titan forecast x_trend_mps: robust fit of the track's recent positions"),
    "trend_north_mps": ("m/s", "titan forecast y_trend_mps"),
    "trend_speed_mps": ("m/s", "hypot(trend_east_mps, trend_north_mps)"),
    "trend_direction_to_deg": ("degrees clockwise from north, toward", "from the trend components"),
    "trend_area_log_per_min": ("ln(km^2)/min", "titan forecast area_log_trend_per_min"),
    "trend_volume_log_per_min": ("ln(km^3)/min", "titan forecast volume_log_trend_per_min"),
    "trend_top_m_per_min": ("m/min", "titan forecast top_trend_m_per_min"),
    "trend_max_dbz_per_min": ("dBZ/min", "titan forecast max_dbz_trend_per_min"),
    "trend_history_points": ("count", "titan forecast history_points"),
    "trend_method": ("text", "titan forecast method"),
    "forecast_leads_s": ("s, list", "titan forecast lead times in this row"),
    "footprint_columns": ("count", "grid columns under titan's voxels"),
    "footprint_source": ("text", "voxels (titan 3-D cells) or polygon (rasterised footprint)"),
    "peak_w_mps": ("m/s", "WOOF-ArWen W, max over footprint columns and all w levels"),
    "peak_w_ft_min": ("ft/min", "peak_w_mps x 196.85"),
    "peak_w_height_m_msl": ("m MSL", "WOOF-ArWen w-level height of the peak"),
    "peak_w_lat": ("degrees N", "WOOF-ArWen XLAT of the peak's column"),
    "peak_w_lon": ("degrees E", "WOOF-ArWen XLONG of the peak's column"),
    "min_w_mps": ("m/s", "WOOF-ArWen W, min over footprint columns (downdraft)"),
    "cloud_top_m_msl": ("m MSL", "WOOF-ArWen: highest mass level with QCLOUD+QICE > 1e-6 kg/kg, max over footprint"),
    "cloud_top_temperature_c": ("degrees C", "WOOF-ArWen temperature at that level in that column"),
    "cloud_base_m_msl": ("m MSL", "WOOF-ArWen: lowest mass level with QCLOUD+QICE > 1e-6 kg/kg, min over footprint"),
    "cloud_base_temperature_c": ("degrees C", "WOOF-ArWen temperature at that level in that column"),
    "cloud_depth_m": ("m", "cloud_top_m_msl minus cloud_base_m_msl"),
    "freezing_level_m_msl": ("m MSL", "WOOF-ArWen: footprint mean of the lowest 0 C crossing (linear between mass levels)"),
    "level_minus5c_m_msl": ("m MSL", "WOOF-ArWen: footprint mean of the lowest -5 C crossing"),
    "level_minus10c_m_msl": ("m MSL", "WOOF-ArWen: footprint mean of the lowest -10 C crossing"),
    "level_minus15c_m_msl": ("m MSL", "WOOF-ArWen: footprint mean of the lowest -15 C crossing"),
    "level_minus20c_m_msl": ("m MSL", "WOOF-ArWen: footprint mean of the lowest -20 C crossing"),
    "slwp_max_kg_m2": ("kg/m^2", "WOOF-ArWen: sum of QCLOUD x rho x dz over levels with T < 0 C, max over footprint columns"),
    "slwp_mean_kg_m2": ("kg/m^2", "the same, mean over footprint columns"),
}

CSV_NAME = "catalog.csv"
JSON_NAME = "catalog.json"
GEOJSON_NAME = "cells.geojson"
OVERLAY_DIR = "overlays"


class CatalogError(RuntimeError):
    """A catalog that cannot be joined, named by the mismatch."""


# -- geometry ---------------------------------------------------------------

def grid_to_latlon(lat: np.ndarray, lon: np.ndarray, i, j
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear ``XLAT``/``XLONG`` at fractional grid indices ``(i, j)``.

    ``i`` runs west_east, ``j`` south_north; both are clipped to the grid
    so a titan centroid a hair outside (a footprint touching the edge)
    still lands on the map.
    """

    ny, nx = lat.shape
    fi = np.clip(np.asarray(i, dtype=np.float64), 0.0, nx - 1.0)
    fj = np.clip(np.asarray(j, dtype=np.float64), 0.0, ny - 1.0)
    i0 = np.clip(np.floor(fi).astype(int), 0, max(nx - 2, 0))
    j0 = np.clip(np.floor(fj).astype(int), 0, max(ny - 2, 0))
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    wi = fi - i0
    wj = fj - j0

    def mix(grid):
        return ((1 - wi) * (1 - wj) * grid[j0, i0] + wi * (1 - wj) * grid[j0, i1]
                + (1 - wi) * wj * grid[j1, i0] + wi * wj * grid[j1, i1])

    lon_east = np.where(lon < 0, lon + 360.0, lon) if np.ptp(lon) > 180 else lon
    out_lon = mix(lon_east)
    if np.ptp(lon) > 180:
        out_lon = np.where(out_lon > 180.0, out_lon - 360.0, out_lon)
    return mix(lat), out_lon


class Projector:
    """titan's projected metres <-> the model's grid and map."""

    def __init__(self, frame: ColumnFrame, origin_x_m: float,
                 origin_y_m: float, dx_m: float, dy_m: float):
        self.lat = frame.lat
        self.lon = frame.lon
        self.origin_x = float(origin_x_m)
        self.origin_y = float(origin_y_m)
        self.dx = float(dx_m)
        self.dy = float(dy_m)
        self.nx = frame.nx
        self.ny = frame.ny

    def to_index(self, x_m, y_m) -> tuple[np.ndarray, np.ndarray]:
        """Fractional ``(i, j)`` mass-point indices of projected metres."""

        i = (np.asarray(x_m, dtype=np.float64) - self.origin_x) / self.dx - 0.5
        j = (np.asarray(y_m, dtype=np.float64) - self.origin_y) / self.dy - 0.5
        return i, j

    def to_latlon(self, x_m, y_m) -> tuple[np.ndarray, np.ndarray]:
        i, j = self.to_index(x_m, y_m)
        return grid_to_latlon(self.lat, self.lon, i, j)

    def ring_to_latlon(self, ring) -> list[list[float]]:
        """A titan ring (metres) as ``[[lat, lon], ...]``."""

        pts = np.asarray(ring, dtype=np.float64)
        if pts.size == 0:
            return []
        lat, lon = self.to_latlon(pts[:, 0], pts[:, 1])
        return [[round(float(a), 5), round(float(b), 5)]
                for a, b in zip(lat, lon)]

    def footprint_rings(self, footprint) -> list[list[list[list[float]]]]:
        """Every part's rings, lat/lon, for a Polygon or MultiPolygon."""

        if not footprint:
            return []
        kind = footprint.get("type")
        coords = footprint.get("coordinates") or []
        parts = [coords] if kind == "Polygon" else coords
        return [[self.ring_to_latlon(ring) for ring in part] for part in parts]

    def geojson_geometry(self, footprint) -> dict | None:
        """The same footprint as GeoJSON (lon, lat order) for map tools."""

        if not footprint:
            return None
        parts = self.footprint_rings(footprint)
        swapped = [[[[p[1], p[0]] for p in ring] for ring in part]
                   for part in parts]
        if footprint.get("type") == "Polygon":
            return {"type": "Polygon", "coordinates": swapped[0] if swapped else []}
        return {"type": "MultiPolygon", "coordinates": swapped}


def rasterise_footprint(footprint, projector: Projector) -> np.ndarray:
    """Column indices ``y * nx + x`` whose centres fall inside a footprint.

    The fallback for a bundle written with ``retain_voxels=false``;
    even-odd rule on cell centres, which titan's own cell-edge rings
    never pass through (FORMATS.md).
    """

    if not footprint:
        return np.zeros(0, dtype=np.int64)
    kind = footprint.get("type")
    coords = footprint.get("coordinates") or []
    parts = [coords] if kind == "Polygon" else coords
    inside_any = np.zeros((projector.ny, projector.nx), dtype=bool)
    xs = projector.origin_x + (np.arange(projector.nx) + 0.5) * projector.dx
    ys = projector.origin_y + (np.arange(projector.ny) + 0.5) * projector.dy
    cx, cy = np.meshgrid(xs, ys)
    for part in parts:
        inside = np.zeros_like(inside_any)
        for ring in part:
            pts = np.asarray(ring, dtype=np.float64)
            if len(pts) < 4:
                continue
            x0, y0 = pts[:-1, 0], pts[:-1, 1]
            x1, y1 = pts[1:, 0], pts[1:, 1]
            crossings = np.zeros_like(inside)
            for ax, ay, bx, by in zip(x0, y0, x1, y1):
                if ay == by:
                    continue
                spans = ((ay > cy) != (by > cy))
                xint = ax + (cy - ay) * (bx - ax) / (by - ay)
                crossings ^= spans & (cx < xint)
            inside ^= crossings
        inside_any |= inside
    return np.flatnonzero(inside_any.ravel())


def footprint_columns(obj: dict, nx: int, ny: int, projector: Projector
                      ) -> tuple[np.ndarray, str]:
    """Linear column indices under one titan object, and where they came from."""

    voxels = obj.get("voxels") or []
    if voxels:
        x, y, _z = cell_xyz(nx, ny, np.asarray(voxels, dtype=np.int64))
        return np.unique(y * nx + x), "voxels"
    return rasterise_footprint(obj.get("footprint"), projector), "polygon"


# -- ArWen attributes over a column set -------------------------------------

def isotherm_heights(temperature_k: np.ndarray, z_mass: np.ndarray,
                     target_c: float) -> np.ndarray:
    """Height of the lowest crossing of ``target_c`` in each column.

    ``temperature_k`` and ``z_mass`` are ``(nz, n)``.  Linear between the
    bracketing mass levels; a column already colder than the target at
    its lowest level reports that level's height; a column never that
    cold reports NaN.
    """

    target = target_c + 273.15
    t = np.asarray(temperature_k, dtype=np.float64)
    z = np.asarray(z_mass, dtype=np.float64)
    out = np.full(t.shape[1], np.nan)
    above = t[:-1] > target
    below = t[1:] <= target
    crossing = above & below
    has = crossing.any(axis=0)
    k = np.argmax(crossing, axis=0)
    cols = np.arange(t.shape[1])
    t0 = t[k, cols]
    t1 = t[k + 1, cols]
    z0 = z[k, cols]
    z1 = z[k + 1, cols]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (target - t0) / (t1 - t0)
    out[has] = (z0 + frac * (z1 - z0))[has]
    at_surface = t[0] <= target
    out[at_surface] = z[0, at_surface]
    return out


def column_attributes(frame: ColumnFrame, cols: np.ndarray) -> dict:
    """Every WOOF-ArWen attribute of one footprint, sampled on ``cols``."""

    if cols.size == 0:
        return {}
    ny, nx = frame.ny, frame.nx
    ys, xs = cols // nx, cols % nx
    w = frame.w_mps[:, ys, xs]                    # (nz+1, n)
    z_w = frame.z_w[:, ys, xs]
    peak_flat = int(np.argmax(w))
    peak_level, peak_col = divmod(peak_flat, w.shape[1])
    peak = float(w[peak_level, peak_col])
    t = frame.temperature_k[:, ys, xs].astype(np.float64)
    z = frame.z_mass[:, ys, xs].astype(np.float64)
    condensate = frame.qcloud[:, ys, xs] + frame.qice[:, ys, xs]
    cloudy = condensate > CLOUD_THRESHOLD_KG_KG
    has_cloud = cloudy.any(axis=0)
    nz = cloudy.shape[0]
    top_k = nz - 1 - np.argmax(cloudy[::-1], axis=0)
    base_k = np.argmax(cloudy, axis=0)
    ncol = np.arange(cols.size)
    top_z = np.where(has_cloud, z[top_k, ncol], np.nan)
    base_z = np.where(has_cloud, z[base_k, ncol], np.nan)
    out: dict = {
        "footprint_columns": int(cols.size),
        "peak_w_mps": peak,
        "peak_w_ft_min": peak * FT_PER_MIN_PER_MPS,
        "peak_w_height_m_msl": float(z_w[peak_level, peak_col]),
        "peak_w_lat": float(frame.lat[ys[peak_col], xs[peak_col]]),
        "peak_w_lon": float(frame.lon[ys[peak_col], xs[peak_col]]),
        "min_w_mps": float(np.min(w)),
    }
    if has_cloud.any():
        top_col = int(np.nanargmax(top_z))
        base_col = int(np.nanargmin(base_z))
        out["cloud_top_m_msl"] = float(top_z[top_col])
        out["cloud_top_temperature_c"] = float(t[top_k[top_col], top_col] - 273.15)
        out["cloud_base_m_msl"] = float(base_z[base_col])
        out["cloud_base_temperature_c"] = float(t[base_k[base_col], base_col] - 273.15)
        out["cloud_depth_m"] = out["cloud_top_m_msl"] - out["cloud_base_m_msl"]
    else:
        for name in ("cloud_top_m_msl", "cloud_top_temperature_c",
                     "cloud_base_m_msl", "cloud_base_temperature_c",
                     "cloud_depth_m"):
            out[name] = None
    names = {0.0: "freezing_level_m_msl", -5.0: "level_minus5c_m_msl",
             -10.0: "level_minus10c_m_msl", -15.0: "level_minus15c_m_msl",
             -20.0: "level_minus20c_m_msl"}
    for target in ISOTHERMS_C:
        heights = isotherm_heights(t, z, target)
        finite = heights[np.isfinite(heights)]
        out[names[target]] = float(finite.mean()) if finite.size else None
    rho = frame.density_kg_m3()[:, ys, xs]
    dz = frame.layer_thickness_m()[:, ys, xs]
    supercooled = (frame.temperature_k[:, ys, xs] < 273.15)
    slwp = (np.where(supercooled, frame.qcloud[:, ys, xs] * rho * dz, 0.0)
            .sum(axis=0))
    out["slwp_max_kg_m2"] = float(slwp.max())
    out["slwp_mean_kg_m2"] = float(slwp.mean())
    return out


def peak_w_direct(frame: ColumnFrame, cols: np.ndarray) -> float:
    """The reference computation a test holds the catalog against."""

    ys, xs = cols // frame.nx, cols % frame.nx
    return float(frame.w_mps[:, ys, xs].max())


# -- the join ---------------------------------------------------------------

def _num(value):
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _trend_speed(forecast: dict) -> float | None:
    vx, vy = forecast.get("x_trend_mps"), forecast.get("y_trend_mps")
    if vx is None or vy is None:
        return None
    return float(math.hypot(vx, vy))


def _trend_direction(forecast: dict) -> float | None:
    """Bearing the cell moves TOWARD, degrees clockwise from north."""

    vx, vy = forecast.get("x_trend_mps"), forecast.get("y_trend_mps")
    if vx is None or vy is None or (vx == 0 and vy == 0):
        return None
    return float(math.degrees(math.atan2(vx, vy)) % 360.0)


def _lineage(bundle: Bundle, track_id: int) -> tuple[list[int], list[int]]:
    track = bundle.tracks.get(track_id) or {}
    return ([int(v) for v in track.get("parent_ids", [])],
            [int(v) for v in track.get("child_ids", [])])


def _fmt_list(values) -> str:
    return ";".join(str(v) for v in values)


def cell_row(bundle: Bundle, frame_json: dict, obj: dict, frame: ColumnFrame,
             frame_index: int, projector: Projector) -> tuple[dict, dict]:
    """One catalog row and its footprint geometry (lat/lon rings)."""

    assignment = bundle.track_of(frame_json).get(int(obj["object_id"]), {})
    track_id = assignment.get("track_id")
    track = bundle.tracks.get(int(track_id), {}) if track_id is not None else {}
    metrics = obj.get("metrics", {})
    centroid = obj.get("centroid", {})
    cx, cy = centroid.get("x_m"), centroid.get("y_m")
    lat, lon = (projector.to_latlon(cx, cy) if cx is not None and cy is not None
                else (np.nan, np.nan))
    state = bundle.active_track_state(frame_json).get(
        int(track_id), {}) if track_id is not None else {}
    forecast = bundle.forecasts_of(frame_json).get(
        int(track_id), {}) if track_id is not None else {}
    parents, children = (_lineage(bundle, int(track_id))
                         if track_id is not None else ([], []))
    cols, source = footprint_columns(obj, frame.nx, frame.ny, projector)
    row: dict = {
        "valid_time": frame.valid.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timestamp_ms": int(frame_json["timestamp_ms"]),
        "frame_index": frame_index,
        "domain": frame.domain,
        "object_id": int(obj["object_id"]),
        "track_id": None if track_id is None else int(track_id),
        "is_birth": bool(assignment.get("is_birth", False)),
        "track_status": track.get("status"),
        "lifetime_so_far_s": (None if not track else
                              (int(frame_json["timestamp_ms"]) - int(track["created_ms"])) / 1000.0),
        "track_seen_scans": track.get("seen_scans"),
        "parent_track_ids": parents,
        "child_track_ids": children,
        "centroid_x_m": _num(cx),
        "centroid_y_m": _num(cy),
        "centroid_lat": _num(float(lat)),
        "centroid_lon": _num(float(lon)),
        "projected_area_km2": _num(metrics.get("projected_area_km2")),
        "volume_km3": _num(metrics.get("volume_km3")),
        "max_dbz": _num(metrics.get("max_dbz")),
        "mean_dbz": _num(metrics.get("mean_dbz")),
        "base_height_m_msl": _num(metrics.get("base_height_m")),
        "top_height_m_msl": _num(metrics.get("top_height_m")),
        "echo_top_18dbz_m_msl": _num(metrics.get("echo_top_18dbz_m")),
        "echo_top_30dbz_m_msl": _num(metrics.get("echo_top_30dbz_m")),
        "echo_top_40dbz_m_msl": _num(metrics.get("echo_top_40dbz_m")),
        "echo_top_50dbz_m_msl": _num(metrics.get("echo_top_50dbz_m")),
        "vil_max_kg_m2": _num(metrics.get("vil_max_kg_m2")),
        "mesh_mm": _num(metrics.get("mesh_mm")),
        "mean_temperature_c": _num(metrics.get("mean_temperature_c")),
        "motion_east_mps": _num(state.get("vx_mps")),
        "motion_north_mps": _num(state.get("vy_mps")),
        "motion_speed_mps": _num(state.get("speed_mps")),
        "motion_direction_to_deg": _num(state.get("direction_deg")),
        "trend_east_mps": _num(forecast.get("x_trend_mps")),
        "trend_north_mps": _num(forecast.get("y_trend_mps")),
        "trend_speed_mps": _trend_speed(forecast),
        "trend_direction_to_deg": _trend_direction(forecast),
        "trend_area_log_per_min": _num(forecast.get("area_log_trend_per_min")),
        "trend_volume_log_per_min": _num(forecast.get("volume_log_trend_per_min")),
        "trend_top_m_per_min": _num(forecast.get("top_trend_m_per_min")),
        "trend_max_dbz_per_min": _num(forecast.get("max_dbz_trend_per_min")),
        "trend_history_points": forecast.get("history_points"),
        "trend_method": forecast.get("method"),
        "forecast_leads_s": [int(p["lead_seconds"]) for p in forecast.get("points", [])],
        "footprint_source": source,
    }
    row.update(column_attributes(frame, cols))
    geometry = {
        "footprint": projector.footprint_rings(obj.get("footprint")),
        "geojson": projector.geojson_geometry(obj.get("footprint")),
        "forecast": [],
    }
    for point in forecast.get("points", []):
        px, py = point.get("x_m"), point.get("y_m")
        plat, plon = (projector.to_latlon(px, py)
                      if px is not None and py is not None else (np.nan, np.nan))
        geometry["forecast"].append({
            "lead_s": int(point["lead_seconds"]),
            "valid_time_ms": point.get("valid_time_ms"),
            "centroid_lat": _num(float(plat)),
            "centroid_lon": _num(float(plon)),
            "projected_area_km2": _num(point.get("projected_area_km2")),
            "max_dbz": _num(point.get("max_dbz")),
            "top_height_m_msl": _num(point.get("top_height_m")),
            "confidence": _num(point.get("confidence")),
            "footprint": projector.footprint_rings(point.get("footprint")),
        })
    return row, geometry


# -- overlays for the renderer ---------------------------------------------

#: Distinct, print-safe colours cycled by track id.
PALETTE = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b",
           "#e377c2", "#17becf", "#bcbd22", "#7f7f7f")


def track_colour(track_id: int | None) -> str:
    return PALETTE[(0 if track_id is None else int(track_id)) % len(PALETTE)]


def overlay_document(rows: list[dict], geometries: list[dict], *,
                     forecast_lead_s: int | None = None) -> dict:
    """The renderer's ``--overlays`` JSON for one frame's cells.

    Each cell's footprint rings become closed lines coloured by track;
    a label at the centroid names the track, its lifetime, and its peak
    updraft; the forecast footprint at ``forecast_lead_s`` (the first
    lead when None) is drawn thinner in the same colour.
    """

    lines: list[dict] = []
    labels: list[dict] = []
    for row, geometry in zip(rows, geometries):
        colour = track_colour(row.get("track_id"))
        for part in geometry.get("footprint", []):
            for ring in part:
                if len(ring) >= 4:
                    lines.append({"points": ring, "color": colour, "width": 3,
                                  "closed": True})
        chosen = None
        for point in geometry.get("forecast", []):
            if forecast_lead_s is None or point["lead_s"] == forecast_lead_s:
                chosen = point
                break
        if chosen is not None:
            for part in chosen.get("footprint", []):
                for ring in part:
                    if len(ring) >= 4:
                        lines.append({"points": ring, "color": colour,
                                      "width": 1, "closed": True})
        if row.get("centroid_lat") is not None:
            age = row.get("lifetime_so_far_s")
            peak = row.get("peak_w_mps")
            text = f"t{row.get('track_id')}"
            if age is not None:
                text += f" {age / 60.0:.0f}min"
            if peak is not None:
                text += f" {peak:.0f}m/s"
            labels.append({"lat": row["centroid_lat"], "lon": row["centroid_lon"],
                           "text": text})
    return {"lines": lines, "labels": labels}


# -- the door's work --------------------------------------------------------

def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return _fmt_list(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_catalog(rows: list[dict], geometries: list[dict], out_dir: Path,
                  *, receipt: dict) -> dict[str, str]:
    """``catalog.csv``, ``catalog.json``, ``cells.geojson`` under ``out_dir``."""

    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(COLUMNS)
    csv_path = out_dir / CSV_NAME
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for row in rows:
            writer.writerow([_csv_value(row.get(name)) for name in names])
    json_path = out_dir / JSON_NAME
    document = {
        "schema": CATALOG_SCHEMA,
        "columns": {name: {"unit": unit, "provenance": provenance}
                    for name, (unit, provenance) in COLUMNS.items()},
        "receipt": receipt,
        "rows": [dict(row, geometry=geometry)
                 for row, geometry in zip(rows, geometries)],
    }
    json_path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    features = []
    for row, geometry in zip(rows, geometries):
        if geometry.get("geojson") is None:
            continue
        features.append({"type": "Feature", "geometry": geometry["geojson"],
                         "properties": {k: v for k, v in row.items()}})
    geojson_path = out_dir / GEOJSON_NAME
    geojson_path.write_text(json.dumps(
        {"type": "FeatureCollection",
         "properties": {"coordinate_space": "WGS84 lon/lat from WOOF-ArWen XLAT/XLONG",
                        "schema": CATALOG_SCHEMA},
         "features": features}) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path),
            "geojson": str(geojson_path)}


def write_overlays(rows: list[dict], geometries: list[dict], out_dir: Path
                   ) -> list[str]:
    """One renderer overlay file per frame, named by the valid stamp."""

    overlay_dir = out_dir / OVERLAY_DIR
    overlay_dir.mkdir(parents=True, exist_ok=True)
    by_frame: dict[int, tuple[list[dict], list[dict]]] = {}
    stamps: dict[int, str] = {}
    for row, geometry in zip(rows, geometries):
        key = int(row["timestamp_ms"])
        by_frame.setdefault(key, ([], []))
        by_frame[key][0].append(row)
        by_frame[key][1].append(geometry)
        stamps[key] = row["valid_time"]
    written: list[str] = []
    for key in sorted(by_frame):
        stamp = stamps[key].replace(":", "").replace("-", "")
        path = overlay_dir / f"cells_{stamp}.json"
        path.write_text(json.dumps(overlay_document(*by_frame[key]))
                        + "\n", encoding="utf-8")
        written.append(str(path))
    return written


def build_catalog(inputs, bundle_dir: Path | str, out_dir: Path | str, *,
                  progress: Callable[[str], None] | None = None,
                  executable: Path | None = None,
                  export_receipt: dict | None = None) -> dict:
    """Join a titan bundle to its wrfout series; write the catalog files."""

    say = progress or (lambda _text: None)
    started = time.perf_counter()
    bundle = Bundle.load(bundle_dir)
    by_time = {int(frame["timestamp_ms"]): frame for frame in bundle.frames}
    paths, skipped = partition_inputs(expand_inputs(inputs), executable=executable)
    for entry in skipped:
        say(f"skipped {Path(entry['path']).name}: {entry['reason']} (initial frame)")
    if not paths:
        raise CatalogError(
            "none of the wrfout files carries REFL_10CM; there is nothing "
            "titan could have identified in them")
    grid = (export_receipt or {}).get("grid") or {}
    rows: list[dict] = []
    geometries: list[dict] = []
    matched = 0
    frame_index = 0
    for path in paths:
        for frame in open_frames(path, executable=executable):
            frame_json = by_time.get(frame.timestamp_ms)
            if frame_json is None:
                say(f"no titan frame at {frame.stamp}; skipped")
                frame_index += 1
                continue
            matched += 1
            origin_x = grid.get("origin_x_m", -0.5 * frame.nx * frame.dx_m)
            origin_y = grid.get("origin_y_m", -0.5 * frame.ny * frame.dy_m)
            projector = Projector(frame, origin_x, origin_y, frame.dx_m, frame.dy_m)
            objects = frame_json.get("objects", [])
            for obj in objects:
                row, geometry = cell_row(bundle, frame_json, obj, frame,
                                         frame_index, projector)
                rows.append(row)
                geometries.append(geometry)
            say(f"{frame.stamp}: {len(objects)} cells")
            frame_index += 1
    if matched == 0:
        raise CatalogError(
            f"none of the {frame_index} wrfout frames matches a frame in "
            f"{bundle.path} by timestamp; the bundle was analysed from a "
            f"different series, so export and analyze this one first")
    seconds = time.perf_counter() - started
    receipt = {
        "schema": CATALOG_SCHEMA,
        "written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bundle": str(bundle.path),
        "titan_summary": bundle.summary,
        "titan_config": bundle.resolved_config,
        "wrfout": [str(p) for p in paths],
        "frames_matched": matched,
        "frames_total": frame_index,
        "skipped": skipped,
        "rows": len(rows),
        "tracks": len({r["track_id"] for r in rows if r["track_id"] is not None}),
        "cloud_threshold_kg_kg": CLOUD_THRESHOLD_KG_KG,
        "ft_per_min_per_mps": FT_PER_MIN_PER_MPS,
        "wall_seconds": round(seconds, 3),
    }
    out = Path(out_dir)
    files = write_catalog(rows, geometries, out, receipt=receipt)
    files["overlays"] = write_overlays(rows, geometries, out)
    receipt["files"] = files
    say(f"catalog: {len(rows)} rows over {matched} frames, "
        f"{receipt['tracks']} tracks ({seconds:.1f} s)")
    return receipt


__all__ = [
    "CATALOG_SCHEMA", "CLOUD_THRESHOLD_KG_KG", "COLUMNS", "CSV_NAME",
    "CatalogError", "FT_PER_MIN_PER_MPS", "GEOJSON_NAME", "ISOTHERMS_C",
    "JSON_NAME", "OVERLAY_DIR", "Projector", "build_catalog", "cell_row",
    "column_attributes", "footprint_columns", "grid_to_latlon",
    "isotherm_heights", "overlay_document", "peak_w_direct",
    "rasterise_footprint", "track_colour", "write_catalog", "write_overlays",
]
