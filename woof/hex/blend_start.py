"""A blended start: a regional source's analysis where it exists, a global source's everywhere else.

    woof hex init-source --source hrrr-gfs --cycle 2023-03-31T12 --out-dir INIT \\
        --grid MESH.grid.nc --static MESH.static.nc --vertical-spec tc55-v1.json

THE BREAKAGE THIS CLOSES.  A zoom globe (WOOF's default hex shape: 96 -> 3 km, 400 km core) needs meteorology on
the whole sphere, and HRRR covers CONUS only, so every zoom globe over CONUS started from GFS.  The project ruling of
2026-10-29 ("start CONUS cycles from HRRR") is measured: a GFS cold start builds a sharp low-level-jet edge and a false
squall line within an hour, while a HRRR start matches HRRR.  The HRRR start door (the ``hrrr`` row) initialises a
limited-area cull only, so before this module no global or zoom mesh could start from HRRR anywhere.

THE METHOD, each step an existing door of this tree, nothing re-implemented:

1. **outside**: the outside row's own ``init-source`` on the whole mesh (its acquire, decode, init, cold-soil remap
   and suite species), unchanged -- this is the init a ``--source gfs`` run gets.
2. **the inside region**: the largest lat-lon box inside the inside source's native grid (the intermediate door's
   own :func:`woof.hex.hrrr_intermediate.clip_to_source`), shrunk so that the cull's seven boundary rings, which
   ``rw_mpas_mesh --cull-parent`` adds OUTSIDE the region, stay inside that box too (hop distance on the parent's
   own ``cellsOnCell``).
3. **the cut**: ``rw_mpas_mesh --cull-parent`` cuts the parent's grid, static and vertical artifact by that box,
   the cull door's method, so the cut's terrain and vertical grid ARE the parent's, cell for cell.
4. **inside**: the intermediate door over the cut (``intermediate --grid``), then the init door on the cut with the
   cut vertical artifact, the way the regional HRRR door starts a limited-area region.
5. **blend**: every floating-point field the two inits share becomes ``outside + w * (inside - outside)`` on the
   parent's cells, edges and vertices.  ``w`` is 0 outside the cut and on its boundary rings, and rises as a
   smoothstep over ``taper_km`` of distance from them, so it is 1 well inside the inside source's domain.  An
   edge takes the mean of its two cells' weights, a vertex the mean of its three.  The atmosphere, the soil and
   the hydrometeors take the same weight, so each column's soil and condensate come from the same source as the
   air above it.  Where the two inits agree (terrain, the vertical grid, every static field) nothing is written.

The row is data (``data/init-sources.v1.json``, the ``blend`` block names the two rows and the taper), so another
regional analysis inside another global one is a row, never a code path (the arbitrary acceptance test).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import MpasPortError

BLEND_SCHEMA = "gpuwm-hex.blend-start/v1"
EARTH_RADIUS_KM = 6371.229
#: Boundary rings ``rw_mpas_mesh --cull-parent`` adds outside a region (MPAS-Limited-Area v2.2: bdyMask 1..7).
CULL_RINGS = 7
#: Degrees the cut keeps inside the inscribed box: the intermediate door's 16-point stencil (2 points of 0.025
#: degrees) plus its 30 km margin (0.42 degrees of longitude at 50N), so the door's box over the cut lies inside
#: the inscribed box whole and it never has to trim.  THE BREAKAGE (a test host, 2026-10-29): at a 0.2 degree inset the
#: door's box reached south of HRRR's south-west corner and its trim gave up the west side, which had no ground.
BOX_INSET_DEG = 0.6
MESH_DIMS = {"nCells": "Cell", "nEdges": "Edge", "nVertices": "Vertex"}
FALLBACKS = ("refuse", "outside")


class BlendRefusal(MpasPortError):
    """A blended start cannot be built, and the message says why."""


def _log(message: str) -> None:
    print(f"blend-start: {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# the row
# ---------------------------------------------------------------------------
def blend_block(row: Mapping[str, Any]) -> dict[str, Any]:
    """The row's ``blend`` declaration, checked: two row ids and a positive taper."""

    block = dict(row.get("blend") or {})
    unknown = set(block) - {"inside", "outside", "taper_km", "lid_taper_m", "about"}
    if unknown:
        raise BlendRefusal(f"source row {row.get('id')!r}: blend keys {sorted(unknown)} are unknown")
    for key in ("inside", "outside"):
        if not str(block.get(key, "")).strip():
            raise BlendRefusal(f"source row {row.get('id')!r}: blend.{key} must name a source row")
    taper = block.get("taper_km")
    if not isinstance(taper, (int, float)) or not taper > 0:
        raise BlendRefusal(f"source row {row.get('id')!r}: blend.taper_km must be a positive distance in km")
    lid = block.get("lid_taper_m")
    if lid is not None and (not isinstance(lid, list) or len(lid) != 2
                            or not all(isinstance(v, (int, float)) for v in lid) or not 0 < lid[0] < lid[1]):
        raise BlendRefusal(f"source row {row.get('id')!r}: blend.lid_taper_m is [bottom, top] in metres, bottom < top")
    return block


def effective_row(table: Mapping[str, Any], row: Mapping[str, Any]) -> Mapping[str, Any]:
    """The row whose cycles, posting, Vtable and forecast horizon a door checks a start against: a blend's outside
    (global) row, which is posted last and runs longest; any other row itself."""

    if not row.get("blend"):
        return row
    from .init_sources import resolve_source

    return resolve_source(dict(table), blend_block(row)["outside"])


def blend_rows(table: Mapping[str, Any], row: Mapping[str, Any], accepted=()) -> tuple[dict, dict]:
    """The outside (global) and inside (limited-area) rows a blend row names."""

    from .init_sources import resolve_source

    block = blend_block(row)
    outside = resolve_source(dict(table), block["outside"], accepted)
    inside = resolve_source(dict(table), block["inside"], accepted, coverage="limited_area")
    if outside.get("blend") or inside.get("blend"):
        raise BlendRefusal(f"source row {row['id']!r}: a blend names two plain rows, not another blend")
    intermediate = str((inside.get("intermediate") or {}).get("row", "")).strip()
    if not intermediate:
        raise BlendRefusal(f"source row {row['id']!r}: inside row {inside['id']!r} names no intermediate row")
    return outside, inside


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def inscribed_box(intermediate_row: str, *, spacing_deg: float = 0.05, inset_deg: float = BOX_INSET_DEG,
                  ) -> dict[str, float]:
    """The largest lat-lon box (by the intermediate door's own trim) inside a projected source's native grid,
    pulled in by ``inset_deg`` on every side.  Longitudes are degrees east in -180..180."""

    from .hrrr_intermediate import (LatLonTarget, _native_positions, _source_window_limits, clip_to_source,
                                    source_row)

    from .packaged_intermediate import handles as has_packaged_intermediate, source_table_row, coverage_window
    packaged = has_packaged_intermediate(intermediate_row)
    if packaged:
        window = coverage_window(source_table_row(intermediate_row))
        if window is None or not hasattr(window, "grid"):
            raise BlendRefusal(f"{intermediate_row} declares no projected coverage window")
        grid, x0, x1, y0, y1 = window.grid(), 0., window.nx - 1.000001, 0., window.ny - 1.000001
    else:
        row = source_row(intermediate_row)
        grid, x0, x1, y0, y1 = _source_window_limits(row)
    # Where the native grid is: every 1 degree point of the sphere that maps inside it.
    lat, lon = np.meshgrid(np.arange(-89.0, 90.0, 1.0), np.arange(-180.0, 180.0, 1.0), indexing="ij")
    x, y = _native_positions(grid, lat.ravel(), lon.ravel())
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    if not inside.any():
        raise BlendRefusal(f"no 1-degree point of the sphere lies inside {intermediate_row}'s native grid")
    la, lo = lat.ravel()[inside], lon.ravel()[inside]
    south, north = float(la.min()) - 1.0, float(la.max()) + 1.0
    west, east = float(lo.min()) - 1.0, float(lo.max()) + 1.0
    if east - west >= 300.0:
        raise BlendRefusal(f"{intermediate_row}'s native grid straddles the antimeridian; this box rule does not")
    target = LatLonTarget(south=south, west=west, dlat=spacing_deg, dlon=spacing_deg,
                          nx=int(round((east - west) / spacing_deg)) + 1,
                          ny=int(round((north - south) / spacing_deg)) + 1)
    if packaged:
        # Trim geometrical edges using the row's coverage API. No meteorological arrays enter this calculation.
        box = {"south": south, "north": north, "west": west, "east": east}
        for _ in range(4 * (target.nx + target.ny)):
            la = np.arange(box["south"], box["north"] + spacing_deg / 2, spacing_deg)
            lo = np.arange(box["west"], box["east"] + spacing_deg / 2, spacing_deg)
            lat_edge = np.concatenate([la, la, np.full(lo.size, box["south"]), np.full(lo.size, box["north"])])
            lon_edge = np.concatenate([np.full(la.size, box["west"]), np.full(la.size, box["east"]), lo, lo])
            x, y = grid.latlon_to_ij(lat_edge, lon_edge)
            excess = {"west": float((1-x).max()), "east": float((x-window.nx+1e-6).max()),
                      "south": float((1-y).max()), "north": float((y-window.ny+1e-6).max())}
            side = max(excess, key=excess.get)
            if excess[side] <= 0:
                break
            box[side] += spacing_deg if side in ("south", "west") else -spacing_deg
            if box["south"] >= box["north"] or box["west"] >= box["east"]:
                raise BlendRefusal(f"{intermediate_row} has no inscribed box")
        else:
            raise BlendRefusal(f"{intermediate_row}'s inscribed box did not settle")
        target = LatLonTarget(south=box["south"], west=box["west"], dlat=spacing_deg, dlon=spacing_deg,
                              nx=round((box["east"]-box["west"])/spacing_deg)+1,
                              ny=round((box["north"]-box["south"])/spacing_deg)+1)
    else:
        target, _ = clip_to_source(target, row)
    box = {"south": target.south + inset_deg, "north": target.north - inset_deg,
           "west": target.west + inset_deg, "east": target.east - inset_deg}
    if not (box["north"] > box["south"] and box["east"] > box["west"]):
        raise BlendRefusal(f"{intermediate_row}'s inscribed box {box} is empty after its {inset_deg} degree inset")
    return {key: round(float(value), 6) for key, value in box.items()}


def _in_box(lat: np.ndarray, lon: np.ndarray, box: Mapping[str, float]) -> np.ndarray:
    lon = (np.asarray(lon) + 180.0) % 360.0 - 180.0
    return (lat >= box["south"]) & (lat <= box["north"]) & (lon >= box["west"]) & (lon <= box["east"])


def cap_inside_box(centre: Sequence[float], radius_km: float, box: Mapping[str, float]) -> bool:
    """Whether a spherical cap lies inside a lat-lon box (its rim sampled every 2 degrees of azimuth)."""

    lat0, lon0 = math.radians(float(centre[0])), math.radians(float(centre[1]))
    delta = float(radius_km) / EARTH_RADIUS_KM
    azimuth = np.radians(np.arange(0.0, 360.0, 2.0))
    lat = np.arcsin(np.sin(lat0) * np.cos(delta) + np.cos(lat0) * np.sin(delta) * np.cos(azimuth))
    lon = lon0 + np.arctan2(np.sin(azimuth) * np.sin(delta) * np.cos(lat0),
                            np.cos(delta) - np.sin(lat0) * np.sin(lat))
    rim = _in_box(np.degrees(lat), np.degrees(lon), box)
    return bool(rim.all() and _in_box(np.array([float(centre[0])]), np.array([float(centre[1])]), box).all())


def hop_distance(seed: np.ndarray, neighbours: np.ndarray, counts: np.ndarray, limit: int) -> np.ndarray:
    """Fewest ``cellsOnCell`` steps from each cell to a seed cell (0 on seeds), capped at ``limit``.

    ``neighbours`` is zero-based (n, maxEdges); slots past ``counts`` and negative entries are unused."""

    seed = np.asarray(seed, dtype=bool)
    used = np.arange(neighbours.shape[1])[None, :] < np.asarray(counts)[:, None]
    graph = np.where(used & (neighbours >= 0) & (neighbours < seed.size), neighbours, -1)
    distance = np.full(seed.size, limit, dtype=np.int32)
    distance[seed] = 0
    frontier = np.flatnonzero(seed)
    for step in range(1, limit):
        if frontier.size == 0:
            break
        reached = graph[frontier].ravel()
        reached = np.unique(reached[reached >= 0])
        reached = reached[distance[reached] == limit]
        distance[reached] = step
        frontier = reached
    return distance


def cut_box(lat: np.ndarray, lon: np.ndarray, hops: np.ndarray, box: Mapping[str, float],
            need: int) -> dict[str, float]:
    """``box`` with each side moved in until every cell inside it is at least ``need`` hops from the cells
    outside the original box (each offending cell is given to its nearest side)."""

    lon = (np.asarray(lon) + 180.0) % 360.0 - 180.0
    out = dict(box)
    for _ in range(64):
        bad = _in_box(lat, lon, out) & (hops < need)
        if not bad.any():
            return {key: round(float(value), 6) for key, value in out.items()}
        bl, bo = lat[bad], lon[bad]
        cos = np.cos(np.radians(bl))
        gaps = np.stack([bl - out["south"], out["north"] - bl, (bo - out["west"]) * cos,
                         (out["east"] - bo) * cos])
        side = np.argmin(gaps, axis=0)
        eps = 1e-4
        if (side == 0).any():
            out["south"] = float(bl[side == 0].max()) + eps
        if (side == 1).any():
            out["north"] = float(bl[side == 1].min()) - eps
        if (side == 2).any():
            out["west"] = float(bo[side == 2].max()) + eps
        if (side == 3).any():
            out["east"] = float(bo[side == 3].min()) - eps
        if not (out["north"] > out["south"] and out["east"] > out["west"]):
            raise BlendRefusal(f"the box {dict(box)} leaves no cell {need} rings from its edge on this mesh")
    raise BlendRefusal(f"the cut box did not settle inside {dict(box)}")


def _mesh(path: Path) -> dict[str, np.ndarray]:
    from netCDF4 import Dataset

    with Dataset(str(path)) as dataset:
        dataset.set_auto_maskandscale(False)
        variables = dataset.variables
        return {
            "lat": np.degrees(np.asarray(variables["latCell"][:], dtype=np.float64)),
            "lon": np.degrees(np.asarray(variables["lonCell"][:], dtype=np.float64)),
            "cellsOnCell": np.asarray(variables["cellsOnCell"][:], dtype=np.int64) - 1,
            "nEdgesOnCell": np.asarray(variables["nEdgesOnCell"][:], dtype=np.int64),
        }


def plan_cut(grid: Path, box: Mapping[str, float], *, guard_rings: int = 1) -> dict[str, Any]:
    """The lat-lon box the cut is made by, so the cut AND its seven rings lie inside ``box``."""

    mesh = _mesh(grid)
    outside = ~_in_box(mesh["lat"], mesh["lon"], box)
    need = CULL_RINGS + guard_rings
    hops = hop_distance(outside, mesh["cellsOnCell"], mesh["nEdgesOnCell"], need + 1)
    region = cut_box(mesh["lat"], mesh["lon"], hops, box, need)
    cells = int(_in_box(mesh["lat"], mesh["lon"], region).sum())
    if cells == 0:
        raise BlendRefusal(f"no cell of {grid} lies in the inside source's box {dict(box)}")
    return {"region": {"kind": "lat_lon_box", "lat_deg": [region["south"], region["north"]],
                       "lon_deg": [region["west"], region["east"]]},
            "box": dict(box), "region_box": region, "region_cells": cells, "parent_cells": int(outside.size)}


def decoder_box_fits(grid: Path, intermediate_row: str) -> bool:
    """Check the actual cut plus the decoder's margin, before static cuts or weather decoding.

    A geometric region and its marked seven rings can extend farther than the graph estimate on
    coarse cells. The native HRRR road clips its box; the packaged road correctly refuses it.
    """
    from .packaged_intermediate import handles, coverage_window, source_table_row
    if not handles(intermediate_row):
        return True  # The existing native source road trims its own decoder window.
    from .hrrr_intermediate import target_from_mesh
    target, _ = target_from_mesh(grid, None)
    window = coverage_window(source_table_row(intermediate_row))
    if window is None:
        return True
    la = target.south + np.arange(target.ny) * target.dlat
    lo = target.west + np.arange(target.nx) * target.dlon
    lat = np.concatenate([la, la, np.full(lo.size, la[0]), np.full(lo.size, la[-1])])
    lon = np.concatenate([np.full(la.size, lo[0]), np.full(la.size, lo[-1]), lo, lo])
    x, y = window.grid().latlon_to_ij(lat, lon)
    return bool(((x >= 1) & (x < window.nx) & (y >= 1) & (y < window.ny)).all())


# ---------------------------------------------------------------------------
# the blend
# ---------------------------------------------------------------------------
def _coordinates(dataset, kind: str) -> np.ndarray:
    return np.ascontiguousarray(np.stack(
        [np.asarray(dataset.variables[f"{axis}{kind}"][:], dtype=np.float64) for axis in "xyz"], axis=1))


def _element_index(small: np.ndarray, large: np.ndarray, kind: str) -> np.ndarray:
    """Where each element of a cut sits in its parent: the element with the same three coordinates (a cut copies
    coordinates byte for byte)."""

    where = {row.tobytes(): index for index, row in enumerate(large)}
    found = np.fromiter((where.get(row.tobytes(), -1) for row in small), dtype=np.int64, count=len(small))
    if np.any(found < 0):
        raise BlendRefusal(f"{int(np.sum(found < 0))} {kind.lower()}s of the inside init are not in the outside "
                           "init's mesh; the inside init is not a cut of this mesh")
    return found


def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def cell_weights(parent_xyz: np.ndarray, interior: np.ndarray, taper_km: float) -> np.ndarray:
    """0 off the interior, else the smoothstep of the distance to the nearest non-interior cell over ``taper_km``."""

    from scipy.spatial import cKDTree

    weight = np.zeros(interior.size, dtype=np.float64)
    if not interior.any():
        return weight
    unit = parent_xyz / np.linalg.norm(parent_xyz, axis=1)[:, None]
    if interior.all():
        weight[:] = 1.0
        return weight
    chord, _ = cKDTree(unit[~interior]).query(unit[interior], workers=max(1, int(os.environ.get("RAYON_NUM_THREADS", "8"))))
    distance = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    weight[interior] = smoothstep(distance / float(taper_km))
    return weight


def _grid_geometry(path: Path) -> dict[str, np.ndarray]:
    from netCDF4 import Dataset

    with Dataset(str(path)) as dataset:
        dataset.set_auto_maskandscale(False)
        geometry = {kind: _coordinates(dataset, kind) for kind in MESH_DIMS.values()}
        for name in ("cellsOnEdge", "cellsOnVertex", "bdyMaskCell"):
            if name in dataset.variables:
                geometry[name] = np.asarray(dataset.variables[name][:], dtype=np.int64)
    return geometry


def lid_factor(z: np.ndarray, lid_m: Sequence[float] | None) -> np.ndarray:
    """1 below ``lid_m[0]``, 0 above ``lid_m[1]``, a smoothstep between (metres above sea level)."""

    if lid_m is None:
        return np.ones_like(z, dtype=np.float64)
    bottom, top = float(lid_m[0]), float(lid_m[1])
    return 1.0 - smoothstep((np.asarray(z, dtype=np.float64) - bottom) / (top - bottom))


def blend_inits(outside: Path, inside: Path, out: Path, *, taper_km: float, parent_grid: Path,
                cut_grid: Path, lid_m: Sequence[float] | None = None) -> dict[str, Any]:
    """Write ``out``: ``outside`` with ``w * (inside - outside)`` added on every float field the two share.

    ``inside`` is an init on ``cut_grid``, a cut of ``parent_grid`` (the mesh of ``outside``) whose
    ``bdyMaskCell`` marks the cut's rings.  Elements are matched by their coordinates in the two grid files."""

    from netCDF4 import Dataset

    started = time.perf_counter()
    parent_geometry, cut_geometry = _grid_geometry(parent_grid), _grid_geometry(cut_grid)
    if "bdyMaskCell" not in cut_geometry:
        raise BlendRefusal(f"{cut_grid} carries no bdyMaskCell; the blend weights start from the cut's rings")
    index = {kind: _element_index(cut_geometry[kind], parent_geometry[kind], kind) for kind in MESH_DIMS.values()}
    interior = np.zeros(len(parent_geometry["Cell"]), dtype=bool)
    interior[index["Cell"][cut_geometry["bdyMaskCell"] == 0]] = True
    cell_w = cell_weights(parent_geometry["Cell"], interior, taper_km)
    cells_on_edge = parent_geometry["cellsOnEdge"] - 1
    cells_on_vertex = parent_geometry["cellsOnVertex"] - 1
    partial = out.with_name(out.name + ".partial")
    shutil.copyfile(outside, partial)
    changed: dict[str, Any] = {}
    unchanged: list[str] = []
    only_inside: list[str] = []
    categorical: list[str] = []
    with Dataset(str(partial), "r+") as parent, Dataset(str(inside)) as cut:
        parent.set_auto_maskandscale(False)
        cut.set_auto_maskandscale(False)
        for dimension, kind in MESH_DIMS.items():
            if (len(parent.dimensions[dimension]) != len(parent_geometry[kind])
                    or len(cut.dimensions[dimension]) != len(cut_geometry[kind])):
                raise BlendRefusal(f"{dimension} of the inits differs from their grids'; they are not one mesh")
        # A producer can otherwise silently blend L60 values on two different coordinate surfaces.
        for name in ("nVertLevels", "nVertLevelsP1"):
            if name in parent.dimensions and (name not in cut.dimensions or
                    len(parent.dimensions[name]) != len(cut.dimensions[name])):
                raise BlendRefusal(f"{name} differs between the whole init and cut init")
        if "zgrid" not in parent.variables or "zgrid" not in cut.variables:
            raise BlendRefusal("both inits must carry zgrid to prove they share the vertical grid")
        original_z = np.asarray(parent.variables["zgrid"][:])[index["Cell"]]
        if not np.array_equal(original_z, np.asarray(cut.variables["zgrid"][:])):
            raise BlendRefusal("the cut init changed the whole init's vertical grid; refusing to mix levels")
        vertical: dict[tuple[str, str], np.ndarray] = {}
        if lid_m is not None:
            # The inside source's top: its analysis is used below lid_m[0] and the outside's above lid_m[1].
            # THE BREAKAGE (a test host, 2026-10-29, default zoom): HRRR's lid is under the 30 km top, so the init
            # extrapolates it there and the top level's theta sat 16 K (mean) from GFS's in full-weight cells.
            if "zgrid" not in parent.variables:
                raise BlendRefusal(f"{outside} carries no zgrid; the lid taper is by height")
            zgrid = np.asarray(parent.variables["zgrid"][:], dtype=np.float64)
            per_cell = {"nVertLevelsP1": lid_factor(zgrid, lid_m),
                        "nVertLevels": lid_factor(0.5 * (zgrid[:, 1:] + zgrid[:, :-1]), lid_m)}
            for level, factor in per_cell.items():
                padded_factor = np.concatenate([factor, np.ones((1, factor.shape[1]))])
                vertical[("Cell", level)] = factor
                vertical[("Edge", level)] = padded_factor[np.where(cells_on_edge >= 0, cells_on_edge, -1)].mean(axis=1)
                vertical[("Vertex", level)] = padded_factor[np.where(cells_on_vertex >= 0, cells_on_vertex,
                                                                     -1)].mean(axis=1)
        padded = np.concatenate([cell_w, [0.0]])
        weights = {
            "Cell": cell_w,
            "Edge": padded[np.where(cells_on_edge >= 0, cells_on_edge, -1)].mean(axis=1),
            "Vertex": padded[np.where(cells_on_vertex >= 0, cells_on_vertex, -1)].mean(axis=1),
        }
        for name, variable in cut.variables.items():
            if name not in parent.variables:
                only_inside.append(name)
                continue
            target = parent.variables[name]
            kinds = [dim for dim in variable.dimensions if dim in MESH_DIMS]
            if variable.dimensions != target.dimensions or len(kinds) != 1:
                continue
            if not np.issubdtype(target.dtype, np.floating):
                # Integer fields are the mesh's own (connectivity, indices, categories from the static): a cut
                # renumbers connectivity, and a category is never averaged.
                continue
            kind = MESH_DIMS[kinds[0]]
            axis = variable.dimensions.index(kinds[0])
            where = index[kind]
            weight = weights[kind][where]
            values = np.asarray(target[:])
            selector = [slice(None)] * values.ndim
            selector[axis] = where
            selector = tuple(selector)
            base = values[selector].astype(np.float64)
            incoming = np.asarray(variable[:], dtype=np.float64)
            difference = incoming - base
            active = weight > 0.0
            shape = [1] * values.ndim
            shape[axis] = weight.size
            full_weight = weight.reshape(shape)
            level = next((dim for dim in ("nVertLevels", "nVertLevelsP1") if dim in variable.dimensions), None)
            if level is not None and (kind, level) in vertical:
                factor = vertical[(kind, level)][where]
                level_axis = variable.dimensions.index(level)
                factor_shape = [1] * values.ndim
                factor_shape[axis], factor_shape[level_axis] = factor.shape
                if level_axis < axis:
                    factor = factor.T
                    factor_shape[level_axis], factor_shape[axis] = factor.shape
                full_weight = full_weight * factor.reshape(factor_shape)
            if not np.any(difference[tuple(slice(None) if d != axis else active for d in range(values.ndim))]):
                unchanged.append(name)
                continue
            if not np.isfinite(incoming[tuple(slice(None) if d != axis else active for d in range(values.ndim))]).all():
                raise BlendRefusal(f"the inside init's {name} is not finite where it is blended in")
            if np.array_equal(base, np.round(base)) and np.array_equal(incoming, np.round(incoming)):
                # A flag held as a float (xland 1/2, snowc 0/1, a 0/1 ice field) is never averaged: each element
                # takes the source that weighs more.  THE BREAKAGE (a test host, 2026-10-29, default zoom): a linear
                # blend wrote xland 1.012 on 21 Great Lakes cells where HRRR has ice and GFS has none.
                blended = np.where(full_weight >= 0.5, incoming, base)
                categorical.append(name)
            else:
                blended = base + full_weight * difference
            values[selector] = blended.astype(values.dtype)
            target[:] = values
            change = np.abs(difference) * (full_weight > 0)
            changed[name] = {"kind": kind.lower(), "max_abs_inside_minus_outside": float(change.max())}
    os.replace(partial, out)
    full = cell_w >= 1.0
    return {
        "taper_km": float(taper_km),
        "lid_taper_m": list(lid_m) if lid_m is not None else None,
        "cells": {"parent": int(cell_w.size), "cut": int(index["Cell"].size),
                  "cut_interior": int(interior.sum()), "weight_one": int(full.sum()),
                  "weight_between": int(((cell_w > 0) & ~full).sum()), "weight_zero": int((cell_w <= 0).sum())},
        "edges_blended": int((weights["Edge"] > 0).sum()),
        "vertices_blended": int((weights["Vertex"] > 0).sum()),
        "fields_blended": changed,
        "fields_switched_not_averaged": sorted(categorical),
        "fields_equal": sorted(unchanged),
        "fields_only_inside": sorted(only_inside),
        "seconds": round(time.perf_counter() - started, 3),
        "_cell_weights": cell_w,
    }


def core_weight(grid: Path, weights: np.ndarray, centre: Sequence[float], radius_km: float) -> dict[str, Any]:
    """The smallest and mean blend weight over the cells within ``radius_km`` of ``centre``."""

    mesh = _mesh(grid)
    lat, lon = np.radians(mesh["lat"]), np.radians(mesh["lon"])
    lat0, lon0 = math.radians(float(centre[0])), math.radians(float(centre[1]))
    cos_arc = np.sin(lat) * math.sin(lat0) + np.cos(lat) * math.cos(lat0) * np.cos(lon - lon0)
    distance = EARTH_RADIUS_KM * np.arccos(np.clip(cos_arc, -1.0, 1.0))
    core = distance <= float(radius_km)
    if not core.any():
        return {"cells": 0}
    return {"centre_deg": [float(centre[0]), float(centre[1])], "radius_km": float(radius_km),
            "cells": int(core.sum()), "min_weight": float(weights[core].min()),
            "mean_weight": float(weights[core].mean())}


# ---------------------------------------------------------------------------
# choosing it: the zoom globe's default
# ---------------------------------------------------------------------------
AUTO = "auto"


def core_cap(spec: Mapping[str, Any]) -> tuple[tuple[float, float] | None, float]:
    """The refined core of a resolution spec as a cap: its finest region's centre and reach (radius plus its
    transition).  A polygon or box core is the cap about its vertices' mean that holds every vertex.
    ``(None, 0)`` for a uniform globe."""

    regions = list(spec.get("regions") or [])
    if not regions:
        return None, 0.0
    finest = min(regions, key=lambda region: float(region["spacing_km"]))
    shape = finest.get("shape") or {}
    transition = float(finest.get("transition_km") or
                       float(finest.get("transition_cells") or 0) * float(finest["spacing_km"]))
    if shape.get("kind") == "cap":
        lat, lon = (float(v) for v in shape["center_deg"])
        return (lat, lon), float(shape["radius_km"]) + transition
    if shape.get("kind") == "lat_lon_box":
        (south, north), (west, east) = shape["lat_deg"], shape["lon_deg"]
        vertices = [(south, west), (south, east), (north, west), (north, east)]
    else:
        vertices = [(float(a), float(b)) for a, b in shape.get("vertices_deg") or ()]
    if not vertices:
        return None, 0.0
    lat = np.radians([v[0] for v in vertices])
    lon = np.radians([v[1] for v in vertices])
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)
    mean = xyz.mean(axis=0)
    mean /= np.linalg.norm(mean)
    reach = float(np.max(np.arccos(np.clip(xyz @ mean, -1.0, 1.0)))) * EARTH_RADIUS_KM
    centre = (math.degrees(math.asin(mean[2])), math.degrees(math.atan2(mean[1], mean[0])))
    return centre, reach + transition


def auto_source(table: Mapping[str, Any], centre: Sequence[float] | None, radius_km: float,
                lead_hours: int = 0) -> tuple[str | None, str]:
    """The start a zoom globe gets when its door names none: the first blend row whose inside source's box holds
    the whole refined core (``centre``, ``radius_km``), else that row's outside (global) source.  ``(None, why)``
    when the table has no blend row."""

    blends = [row for row in table["sources"] if row.get("blend") and row.get("status") != "refused"]
    if not blends:
        return None, "the table has no blend row"
    default = blend_block(blends[0])["outside"]
    if centre is None:
        return default, "the mesh has no refined core, so no regional analysis covers it"
    reasons = []
    for row in blends:
        block = blend_block(row)
        inside = next((r for r in table["sources"] if r["id"] == block["inside"]), None)
        intermediate = str(((inside or {}).get("intermediate") or {}).get("row", "")).strip()
        horizon = ((inside or {}).get("forecast") or {}).get("max_hours")
        if horizon is not None and lead_hours > int(horizon):
            reasons.append(f"{row['id']}: {block['inside']} posts no forecast hour {lead_hours}")
            continue
        try:
            box = inscribed_box(intermediate)
        except Exception as error:  # an engine without the source's grid: the default stays the global source
            reasons.append(f"{row['id']}: {type(error).__name__}: {error}")
            continue
        if cap_inside_box(centre, radius_km, box):
            return row["id"], (f"the refined core ({radius_km:g} km about {float(centre[0]):g}, "
                               f"{float(centre[1]):g}) lies inside {block['inside']}'s box {box}")
        reasons.append(f"{row['id']}: the core reaches outside {block['inside']}'s box {box}")
    return default, "; ".join(reasons)


# ---------------------------------------------------------------------------
# the door
# ---------------------------------------------------------------------------
def _child_arguments(arguments: argparse.Namespace, **changes: Any) -> argparse.Namespace:
    values = dict(vars(arguments))
    values.update(changes)
    return argparse.Namespace(**values)


def _capsule_of(init: Path) -> Path | None:
    """The vertical artifact the outside init was built from (its provenance's ``--capsule``)."""

    receipt = init.with_name(init.name + ".provenance.json")
    try:
        argv = json.loads(receipt.read_text(encoding="utf-8"))["engine_argv"]
        path = Path(argv[argv.index("--capsule") + 1])
    except (OSError, KeyError, ValueError, IndexError, json.JSONDecodeError):
        return None
    return path if path.is_file() else None


def _cli(argv: list[str]) -> int:
    from .cli import main as cli_main

    _log("$ woof hex " + " ".join(str(a) for a in argv))
    return int(cli_main([str(a) for a in argv]))


def _staged_grib(arguments: argparse.Namespace, names: Sequence[str]) -> Path | None:
    for directory in getattr(arguments, "staged", None) or ():
        directory = Path(directory).expanduser().absolute()
        if all((directory / name).is_file() for name in names):
            return directory
    return None


def _prefetch_inside(inside_dir: Path, intermediate: str, cycle: datetime, hour: int, workers: int,
                     arguments: argparse.Namespace) -> dict[str, Any]:
    """The inside source's files for the start hour, fetched while the mesh is built (the intermediate door's
    own fetch, :func:`woof.hex.hrrr_intermediate._ensure_hour`)."""

    from types import SimpleNamespace

    from .hrrr_intermediate import _ensure_hour, source_row

    from .packaged_intermediate import handles as has_packaged_intermediate
    if has_packaged_intermediate(intermediate):
        from .regional_door import _prefetch_source
        started = time.perf_counter()
        grib = inside_dir / "grib"
        rc = _prefetch_source([intermediate, str(grib), cycle.strftime("%Y-%m-%dT%H"),
                               str(inside_dir), str(workers), str(hour)])
        if rc:
            raise BlendRefusal(f"{intermediate} prefetch exited {rc}")
        return {"grib_dir": str(grib), "staged": False, "seconds": round(time.perf_counter()-started, 3)}
    row = source_row(intermediate)
    names = [path.name for step in (0, 1) for path in row.hour_files(Path("."), cycle, hour + step)]
    staged = _staged_grib(arguments, names)
    started = time.perf_counter()
    if staged is not None:
        return {"grib_dir": str(staged), "staged": True, "seconds": 0.0}
    request = SimpleNamespace(source=row, grib_dir=inside_dir / "grib", cycle=cycle, out_dir=inside_dir,
                              workers=workers)
    # The decoder series takes two contiguous hours (hrrr_intermediate: "at least two contiguous hours").
    roots = {hour + step: str(_ensure_hour(request, hour + step, _log)) for step in (0, 1)}
    return {"grib_dirs": roots, "staged": False, "seconds": round(time.perf_counter() - started, 3)}


def run_blend(arguments: argparse.Namespace, table: Mapping[str, Any], row: Mapping[str, Any]) -> int:
    """``init-source`` for a blend row: the outside row's init on the whole mesh, the inside row's on a cut of it,
    blended.  Honours ``--stop-after acquire|decode`` and ``--from-decoded`` the way every row does."""

    from .init_sources import Cycle, admit_cycle, run_init_source_command

    started = time.time()
    accepted = getattr(arguments, "accept_licence", ())
    block = blend_block(row)
    outside_row, inside_row = blend_rows(table, row, accepted)
    intermediate = inside_row["intermediate"]["row"]
    fallback = getattr(arguments, "blend_fallback", None) or "refuse"
    if fallback not in FALLBACKS:
        raise BlendRefusal(f"--blend-fallback {fallback!r} is not one of {FALLBACKS}")
    if getattr(arguments, "member", None) is not None:
        raise BlendRefusal(f"{row['id']} is a deterministic start; --member names an ensemble row's member")
    if getattr(arguments, "case_template", None) is not None:
        raise BlendRefusal(f"{row['id']}: --case-template orders a single row's init; order the blended init.nc")
    source_cycle = Cycle.parse(arguments.cycle)
    lead = int(getattr(arguments, "lead_hours", 0) or 0)
    admit_cycle(inside_row, source_cycle)
    valid = source_cycle.when + timedelta(hours=lead)
    outside_cycle = Cycle.parse(getattr(arguments, "boundary_cycle", None) or arguments.cycle)
    from .regional_door import admit_boundary_cycle
    admit_boundary_cycle(outside_row, outside_cycle)
    outside_lead = int((valid - outside_cycle.when).total_seconds() // 3600)
    if outside_lead < 0:
        raise BlendRefusal(f"{outside_row['id']} outside cycle {outside_cycle.start_time} is later than the start")
    from .init_sources import template_context
    template_context(outside_cycle, outside_row, None, outside_lead)
    out_dir = arguments.out_dir.expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    outside_dir, inside_dir = out_dir / "outside", out_dir / "inside"
    inside_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "init-source-receipt.json"
    prior = {}
    if getattr(arguments, "from_decoded", False) and receipt_path.is_file():
        prior = json.loads(receipt_path.read_text(encoding="utf-8"))
        if prior.get("source") != row["id"] or prior.get("cycle") != source_cycle.start_time:
            raise BlendRefusal(f"{receipt_path} was written for another request; --from-decoded reads only its own")
    receipt: dict[str, Any] = {
        "schema": "gpuwm-hex.init-source-receipt/v1", "blend_schema": BLEND_SCHEMA,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "table": {"path": table["_path"], "sha256": table["_sha256"]},
        "source": row["id"], "member": None, "cycle": source_cycle.start_time,
        "source_cycle": source_cycle.start_time, "lead_hours": lead,
        "valid_time": valid.strftime("%Y-%m-%d_%H:%M:%S"), "row": dict(row),
        "run_switches": table["run_switches"],
        "parallelism": {"workers": int(getattr(arguments, "jobs", None) or 8),
                        "rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
                        "omp_threads": os.environ.get("OMP_NUM_THREADS")},
        "blend": {"outside": outside_row["id"], "inside": inside_row["id"], "taper_km": block["taper_km"],
                  "fallback": fallback, "inside_cycle": source_cycle.start_time,
                  "outside_cycle": outside_cycle.start_time, "outside_lead_hours": outside_lead},
    }
    stages: dict[str, float] = dict(prior.get("stage_seconds") or {})
    if prior.get("blend", {}).get("prefetch"):
        receipt["blend"]["prefetch"] = prior["blend"]["prefetch"]

    def write_receipt() -> None:
        receipt["stage_seconds"] = stages
        receipt["seconds"] = round(time.time() - started, 3)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
                                encoding="utf-8")

    # 1. the outside row, every stage of it, through its own door.
    t0 = time.time()
    outside_args = _child_arguments(arguments, source=outside_row["id"], out_dir=outside_dir,
                                    cycle=outside_cycle.when.strftime("%Y-%m-%dT%H"), lead_hours=outside_lead,
                                    blend_fallback=None, boundary_cycle=None)
    rc = run_init_source_command(outside_args)
    stages["outside" if arguments.stop_after not in ("acquire", "decode") else "outside_decode"] = \
        round(time.time() - t0, 3)
    if rc != 0:
        receipt["failed"] = f"the outside row {outside_row['id']} exited {rc}"
        write_receipt()
        return rc
    outside_receipt = json.loads((outside_dir / "init-source-receipt.json").read_text(encoding="utf-8"))
    receipt["outside"] = {key: outside_receipt.get(key) for key in
                          ("source", "products", "decode", "init", "init_argv", "stage_seconds", "seconds",
                           "effective_first_guess")}
    receipt["first_guess"] = {"outside": outside_receipt.get("effective_first_guess", outside_row["first_guess"])}
    if arguments.stop_after in ("acquire", "decode"):
        # The inside source's files need no mesh either: they come down beside the mesh and static.
        t0 = time.time()
        try:
            receipt["blend"]["prefetch"] = _prefetch_inside(
                inside_dir, intermediate, source_cycle.when.replace(tzinfo=None), lead, int(arguments.jobs or 8),
                arguments)
        except Exception as error:  # the full stage fetches again and decides there
            receipt["blend"]["prefetch"] = {"failed": f"{type(error).__name__}: {error}"}
            _log(f"the {inside_row['id']} prefetch failed ({error}); the init stage fetches it again")
        stages["inside_prefetch"] = round(time.time() - t0, 3)
        write_receipt()
        return 0

    outside_init = outside_dir / "init.nc"
    init = out_dir / "init.nc"
    # The zoom door discovers producer drag beside its final init. Keep that contract when the
    # whole-mesh producer ran in the outside subdirectory, or hrrrv5 loses its GSL statistics.
    drag = outside_dir / "ugwp_oro_data.nc"
    if drag.is_file():
        shutil.copyfile(drag, out_dir / drag.name)
        receipt["drag"] = {"path": str(out_dir / drag.name), "source": str(drag)}
    try:
        result = _build_inside_and_blend(arguments, row, block, inside_row, intermediate, source_cycle, lead,
                                         valid, outside_init, inside_dir, init, receipt, stages)
    except Exception as error:
        receipt["failed"] = f"{type(error).__name__}: {error}"
        if fallback != "outside":
            write_receipt()
            raise
        # Chosen by the zoom door's default (auto), not by name: a run still starts, from the outside row, and
        # says so here, on stderr and in the receipt.
        _log(f"FALLBACK: the {inside_row['id']} half failed ({type(error).__name__}: {error}); this init is "
             f"{outside_row['id']} alone")
        shutil.copyfile(outside_init, init)
        provenance = outside_init.with_name(outside_init.name + ".provenance.json")
        if provenance.is_file():
            shutil.copyfile(provenance, init.with_name(init.name + ".provenance.json"))
        receipt["blend"]["fell_back_to"] = outside_row["id"]
        receipt["init"] = {"path": str(init), "provenance": str(init.with_name(init.name + ".provenance.json"))}
        write_receipt()
        return 0
    receipt["blend"].update(result)
    receipt["init"] = {"path": str(init), "provenance": str(init.with_name(init.name + ".provenance.json"))}
    write_receipt()
    print(json.dumps({"source": row["id"], "init": str(init), "seconds": receipt["seconds"],
                      "cells": result["weights"]["cells"]}, indent=2))
    return 0


def _build_inside_and_blend(arguments, row, block, inside_row, intermediate, source_cycle, lead, valid,
                            outside_init: Path, inside_dir: Path, init: Path, receipt, stages) -> dict[str, Any]:
    from .cull_door import carry_lineage, cull_one, resolve_mesh_engine

    result: dict[str, Any] = {}
    grid = Path(arguments.grid).expanduser().absolute()
    static = Path(arguments.static).expanduser().absolute()
    # 2. the box the inside source covers, and the cut that keeps its rings inside it.
    t0 = time.time()
    box = inscribed_box(intermediate)
    plan = plan_cut(grid, box)
    region_path = inside_dir / "cut-region.json"
    region_path.write_text(json.dumps(plan["region"], indent=2) + "\n", encoding="utf-8")
    result["cut_plan"] = plan
    stages["inside_plan"] = round(time.time() - t0, 3)
    # 3. the cut: grid, static and vertical artifact, all by the same region of the same mesh.
    t0 = time.time()
    from .physics_default import init_producer
    suite = getattr(arguments, "physics_suite", None)
    producer = init_producer(suite) if suite else None
    capsule = outside_init if producer else _capsule_of(outside_init)
    if capsule is None:
        capsule = outside_init
        _log(f"the outside init names no vertical artifact on disk; the cut takes its vertical grid from {capsule}")
    engine = resolve_mesh_engine(None)
    cut_grid, cut_static, cut_vertical = (inside_dir / "cut.grid.nc", inside_dir / "cut.static.nc",
                                          inside_dir / "cut.vertical.nc")
    attempts = []
    for guard in (1, 3, 5, 7):
        if guard != 1:
            plan = plan_cut(grid, box, guard_rings=guard)
            region_path.write_text(json.dumps(plan["region"], indent=2) + "\n", encoding="utf-8")
        cut = cull_one(engine, grid, region_path, cut_grid, clobber=True)
        fits = decoder_box_fits(cut_grid, intermediate)
        attempts.append({"guard_rings": guard, "decoder_box_fits": fits, "seconds": cut.get("seconds"),
                         "bytes": cut_grid.stat().st_size})
        if fits:
            break
    else:
        raise BlendRefusal(f"{inside_row['id']}'s cut and decoder margin do not fit inside its source map")
    result["cut_plan"] = plan
    result["cut_attempts"] = attempts
    cuts = {"grid": cut}
    cuts["grid"]["lineage"] = carry_lineage(grid, cut_grid, drives_boundaries=False)
    cuts["static"] = cull_one(engine, outside_init if producer else static, region_path, cut_static, clobber=True, provenance_grid=cut_grid,
                              mark_parent=grid)
    cuts["vertical"] = cull_one(engine, capsule, region_path, cut_vertical, clobber=True,
                                provenance_grid=cut_grid, mark_parent=grid)
    result["cut"] = {name: {key: value for key, value in item.items() if key in ("out", "seconds", "receipt")}
                     for name, item in cuts.items()}
    result["cut"]["vertical_parent"] = str(capsule)
    stages["inside_cut"] = round(time.time() - t0, 3)
    # 4. the inside source on the cut: its intermediate, then its init with the cut's vertical grid.
    t0 = time.time()
    met_dir = inside_dir / "met"
    prefetch = (receipt["blend"].get("prefetch") or {})
    grib_dir = inside_dir / "grib"
    if prefetch.get("grib_dir"):
        grib_dir = Path(prefetch["grib_dir"])
    elif prefetch.get("grib_dirs"):
        # The prefetch fetched each hour into its own folder; the door reads one: link them side by side.
        from .hrrr_intermediate import source_row

        grib_dir.mkdir(parents=True, exist_ok=True)
        for hour, folder in prefetch["grib_dirs"].items():
            for path in source_row(intermediate).hour_files(Path(folder), source_cycle.when.replace(tzinfo=None),
                                                            int(hour)):
                if path.is_file() and not (grib_dir / path.name).exists():
                    os.link(path, grib_dir / path.name)
    # The decoder series takes two contiguous hours; the init reads the start hour's file.
    argv = ["intermediate", "--source", intermediate, "--grid", cut_grid, "--grib-dir", grib_dir,
            "--cycle", source_cycle.when.strftime("%Y-%m-%dT%H"), "--hours", str(lead),
            "--out-dir", met_dir,
            "--workers", str(min(13, int(getattr(arguments, "jobs", None) or 8)))]
    if getattr(arguments, "physics_suite", None):
        argv += ["--physics-suite", arguments.physics_suite]
    if _cli(argv) != 0:
        raise BlendRefusal(f"the intermediate door refused {inside_row['id']} over the cut ({met_dir})")
    stages["inside_intermediate"] = round(time.time() - t0, 3)
    t0 = time.time()
    intermediate_receipt = json.loads((met_dir / "intermediate-receipt.json").read_text(encoding="utf-8"))
    stamp = valid.strftime("%Y-%m-%d_%H")
    mets = [Path(item["path"]) for item in intermediate_receipt["files"] if Path(item["path"]).name.endswith(stamp)]
    if len(mets) != 1:
        raise BlendRefusal(f"the intermediate door wrote no single file valid at {stamp} in {met_dir}")
    guess = dict(inside_row["first_guess"])
    measured = intermediate_receipt.get("levels") or {}
    if measured.get("count"):
        guess["nfglevels"] = int(measured["count"])
    if (measured.get("init_switches") or {}).get("--nfgsoillevels"):
        guess["nfgsoillevels"] = int(measured["init_switches"]["--nfgsoillevels"])
    receipt.setdefault("first_guess", {})["inside"] = guess
    switches = receipt["run_switches"]
    inside_init = inside_dir / "init.nc"
    argv = ["init", "--met", mets[0], "--static", cut_static, "--capsule", cut_vertical, "--reference",
            cut_vertical, "--out", inside_init, "--start-time", valid.strftime("%Y-%m-%d_%H:%M:%S"),
            "--nfglevels", str(guess["nfglevels"]), "--nfgsoillevels", str(guess["nfgsoillevels"]),
            "--use-spechumd", guess["use_spechumd"],
            "--extrap-airtemp", switches["extrap_airtemp"],
            "--theta-adv-order", str(switches["theta_adv_order"]),
            "--coef-3rd-order", repr(float(switches["coef_3rd_order"])),
            "--virtual-factor", switches["virtual_factor"],
            "--deep-soil-moisture", switches["deep_soil_moisture"],
            "--landuse-table", switches["landuse_table"],
            "--frac-seaice", switches["frac_seaice"],
            "--tsk-seaice-threshold", repr(float(switches["tsk_seaice_threshold"])),
            "--oned-underflow", switches["oned_underflow"]]
    for name in ("physics_suite", "physics_profile"):
        if getattr(arguments, name, None):
            argv += ["--" + name.replace("_", "-"), getattr(arguments, name)]
    if producer:
        from .mpasphys.rrfs.cold_start_rrfs import init_on_template
        result["inside_producer"] = init_on_template(
            suite=suite, template=cut_vertical, met=mets[0], out=inside_init,
            start_time=valid.strftime("%Y-%m-%d_%H:%M:%S"),
            nfglevels=int(guess["nfglevels"]), nfgsoillevels=int(guess["nfgsoillevels"]))
    else:
        if _cli(argv) != 0:
            raise BlendRefusal(f"the init door refused {inside_row['id']} on the cut ({inside_init})")
        result["inside_init_argv"] = [str(a) for a in argv]
    stages["inside_init"] = round(time.time() - t0, 3)
    # 5. the blend.
    t0 = time.time()
    weights = blend_inits(outside_init, inside_init, init, taper_km=float(block["taper_km"]), parent_grid=grid,
                          cut_grid=cut_grid, lid_m=block.get("lid_taper_m"))
    cell_w = weights.pop("_cell_weights")
    result["weights"] = weights
    core = getattr(arguments, "blend_core", None)
    if core:
        result["core"] = core_weight(grid, cell_w, core[0], core[1])
    np.save(inside_dir / "cell-weights.npy", cell_w.astype(np.float32))
    provenance = outside_init.with_name(outside_init.name + ".provenance.json")
    if provenance.is_file():
        document = json.loads(provenance.read_text(encoding="utf-8"))
        from .init_door import _described

        document["blend"] = {"schema": BLEND_SCHEMA, "outside_init": document.get("output"),
                             "inside_init": _described(inside_init), "taper_km": float(block["taper_km"]),
                             "cells": weights["cells"]}
        document["output"] = _described(init)
        init.with_name(init.name + ".provenance.json").write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    stages["blend"] = round(time.time() - t0, 3)
    # The door's caller removes spent inputs after its figures and numerical receipt are recorded.
    result["spent_inputs"] = [str(outside_init), str(inside_init), str(cut_vertical), str(met_dir)]
    _log(f"blended {weights['cells']['weight_one']:,} cells at full {inside_row['id']} weight, "
         f"{weights['cells']['weight_between']:,} in the {block['taper_km']:g} km taper, "
         f"{weights['cells']['weight_zero']:,} {receipt['blend']['outside']} alone")
    return result
