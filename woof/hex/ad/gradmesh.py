# SPDX-License-Identifier: Apache-2.0
"""The gradient taken on a coarser companion mesh of the same region and carried back (``--gradient-mesh``).

A regional mesh may name, in the door table's ``gradient_meshes`` section, coarser companion meshes of the same region.
Each companion is table data: its mesh files, its recorded plan, the map between the two meshes (a fixed sparse
operator, coarse cell -> native cell, HXCSR1) and the list of targets the coarse route was shown on. A door asked for
``--gradient-mesh NAME`` runs the same command on the companion mesh and carries the gradient to the native mesh;
``native`` runs the route of the mesh itself, unchanged; ``auto`` reads the mesh's own default in the table, so
changing a default is one table entry. The receipt and the output files name the mesh asked for, the mesh the gradient
was made on, the native mesh it was carried to, the method of carrying, and that the answer check was made on the
coarse operator: a carried gradient never reads as a native one.

The carrying is the one the coarse route was shown with: a gradient is a covector, so it
moves as a density, g_native = area_native * P (g_coarse / area_coarse), P the barycentric interpolation of a coarse
cell field to the native cells on the coarse mesh's Delaunay triangles (rows sum to 1), then zero in the native
boundary zone (bdyMaskCell > 0), where the comparison was made with the zone set to zero.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

CSR_MAGIC = b"HXCSR1\0\0"


class GradientMeshError(RuntimeError):
    """The coarse route cannot be taken for this door, mesh or target, with the reason named."""


# ----------------------------------------------------------------------------------------------------- table
def entry_for(section: dict, mesh: str | None) -> tuple[str | None, dict | None]:
    """The table entry of a native mesh (by its name or one of its ``names``), or (None, None)."""
    if not mesh:
        return None, None
    for key, entry in (section.get("by_mesh") or {}).items():
        if mesh == key or mesh in (entry.get("names") or []):
            return key, entry
    return None, None


def companion(entry: dict, name: str) -> tuple[str | None, dict | None]:
    """A companion of an entry by its key (``6km``) or its mesh name."""
    for key, comp in (entry.get("companions") or {}).items():
        if name == key or name == comp.get("mesh"):
            return key, comp
    return None, None


# ---------------------------------------------------------------------------------------------------- targets
EARTH_RADIUS_KM = 6371.0


def _km(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    a, b, c, d = (math.radians(x) for x in (lat0, lon0, lat1, lon1))
    cosine = math.sin(a) * math.sin(c) + math.cos(a) * math.cos(c) * math.cos(d - b)
    return EARTH_RADIUS_KM * math.acos(max(-1.0, min(1.0, cosine)))


def _floats(text: str, n: int, word: str) -> list[float]:
    try:
        out = [float(x) for x in str(text).split(",")]
    except ValueError:
        out = []
    if len(out) != n:
        raise GradientMeshError(f"{word} {text!r}: give {n} numbers separated by commas")
    return out


def describe(shown: dict) -> str:
    """One shown target in words: the table's own ``what``, else made from its fields."""
    if shown.get("what"):
        return str(shown["what"])
    return json.dumps({k: v for k, v in shown.items() if k not in ("evidence", "measured")}, sort_keys=True)


def misses(shown: dict, target: dict, check_start: bool = True) -> list[str]:
    """What of ``target`` (metric, box, circle, levels, window_s, start) lies outside one shown target; [] when it
    lies inside."""
    out: list[str] = []
    if target.get("metric") not in (shown.get("metrics") or []):
        out.append(f"metric {target.get('metric')} (shown: {', '.join(shown.get('metrics') or []) or 'none'})")
    region = shown.get("region") or {}
    if target.get("box"):
        if "box" not in region:
            out.append("a --box region (shown on a circle)" if "circle" in region else "a --box region")
        else:
            got = _floats(target["box"], 4, "--box")
            if any(abs(a - b) > float(region.get("edge_within_deg", 0.0)) + 1e-9 for a, b in zip(got, region["box"])):
                out.append(f"box {target['box']} (shown: {','.join(f'{x:g}' for x in region['box'])})")
    elif target.get("circle"):
        if "circle" not in region:
            out.append("a --circle region")
        else:
            lat, lon, radius = _floats(target["circle"], 3, "--circle")
            lat0, lon0, _ = region["circle"]
            far = _km(lat0, lon0, lat, lon)
            if far > float(region.get("center_within_km", 0.0)) + 1e-9:
                out.append(f"a circle centred {far:.0f} km from the shown centre {lat0:g},{lon0:g} (shown within "
                           f"{float(region.get('center_within_km', 0.0)):g} km)")
            lo, hi = region.get("radius_km") or [region["circle"][2], region["circle"][2]]
            if not float(lo) - 1e-9 <= radius <= float(hi) + 1e-9:
                out.append(f"radius {radius:g} km (shown: {float(lo):g} to {float(hi):g} km)")
    elif not region.get("every_cell"):
        out.append("every cell of the mesh as the region (shown on a smaller region)")
    levels = shown.get("levels", "all")
    if levels == "all":
        if target.get("levels"):
            out.append(f"levels {target['levels']} (shown on every level)")
    elif str(target.get("levels") or "") not in [str(x) for x in (levels if isinstance(levels, list) else [levels])]:
        out.append(f"levels {target.get('levels') or 'every level'} (shown: {levels})")
    lo, hi = shown.get("window_s") or [0.0, 0.0]
    w = float(target.get("window_s") or 0.0)
    if not float(lo) - 1e-6 <= w <= float(hi) + 1e-6:
        out.append(f"a {w:g} s window (shown: {float(lo):g} to {float(hi):g} s)")
    if check_start:
        starts = shown.get("starts") or []
        if target.get("start") is None:
            out.append("a start the plans do not name (shown: " + ", ".join(starts) + ")")
        elif target["start"] not in starts:
            out.append(f"start {target['start']} (shown: {', '.join(starts) or 'none'})")
    return out


def shown_target(comp: dict, target: dict, check_start: bool = True) -> dict:
    """The shown target ``target`` lies inside, or a named refusal listing every shown target."""
    shown = comp.get("shown") or []
    best = None
    for entry in shown:
        why = misses(entry, target, check_start)
        if not why:
            return entry
        if best is None or len(why) < len(best):
            best = why
    listed = "; ".join(describe(s) for s in shown) or "none yet"
    raise GradientMeshError(f"the coarse route was not shown on this target: {', '.join(best or ['no shown target'])}. "
                            f"It was shown on: {listed}")


# ------------------------------------------------------------------------------------------------------ plans
def plan_start(plan: Path | None) -> str | None:
    """The start time a plan package names (plan.json physics.state_xtime, else from-state.json window_start), as
    YYYY-MM-DDTHH:MM:SS, or None."""
    if plan is None:
        return None
    from datetime import datetime
    plan = Path(plan)
    try:
        stamp = json.loads((plan / "plan.json").read_text(encoding="utf-8"))["physics"]["state_xtime"]
        return datetime.strptime(str(stamp).strip("\0 ")[:19], "%Y-%m-%d_%H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S")
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        stamp = json.loads((plan / "from-state.json").read_text(encoding="utf-8"))["window_start"]
        return datetime.strptime(str(stamp)[:19], "%Y-%m-%d_%H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S")
    except (OSError, ValueError, KeyError, TypeError):
        return None


def coarse_steps(window_s: float, dt_s: float) -> int:
    """Steps of the companion mesh that cover the native window: the whole number of its steps at or above it (the
    route was shown with 30 steps of 30 s for 59 of 15 s)."""
    if not dt_s or dt_s <= 0:
        raise GradientMeshError(f"the companion mesh names no timestep ({dt_s!r})")
    return max(1, int(math.ceil(float(window_s) / float(dt_s) - 1e-9)))


# -------------------------------------------------------------------------------------------------------- map
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_map(path: Path, *, rows: int, cols: int, sha256: str | None) -> dict:
    """The map coarse cell -> native cell (HXCSR1: magic, rows, cols, nnz as u64, then ptr u64[rows+1], col u32[nnz],
    weight f64[nnz]), checked against the table's sha256 and the two meshes' cell counts. THE BREAKAGE THIS PREVENTS:
    a map made for another cut of the region (other cells, other weights) would carry a gradient onto the wrong
    places with no error."""
    import numpy as np
    path = Path(path)
    if not path.is_file():
        raise GradientMeshError(f"the map between the meshes, {path}, is not a file")
    got = sha256_file(path)
    if sha256 and got != sha256:
        raise GradientMeshError(f"the map {path} has sha256 {got}; the table pins {sha256}: another map's carrying "
                                "is not this companion's")
    data = path.read_bytes()
    if data[:8] != CSR_MAGIC:
        raise GradientMeshError(f"{path} is not an HXCSR1 map")
    n_rows, n_cols, nnz = struct.unpack_from("<3Q", data, 8)
    if (n_rows, n_cols) != (int(rows), int(cols)):
        raise GradientMeshError(f"the map {path} takes {n_cols:,} coarse cells to {n_rows:,} native cells; the meshes "
                                f"have {int(cols):,} and {int(rows):,}")
    at = 32
    ptr = np.frombuffer(data, "<u8", n_rows + 1, at).astype(np.int64)
    at += 8 * (n_rows + 1)
    col = np.frombuffer(data, "<u4", nnz, at).astype(np.int64)
    at += 4 * nnz
    weight = np.frombuffer(data, "<f8", nnz, at)
    if int(ptr[-1]) != int(nnz) or (nnz and int(col.max()) >= n_cols):
        raise GradientMeshError(f"the map {path} is not whole (row pointers or columns out of range)")
    return {"path": str(path), "sha256": got, "rows": int(n_rows), "cols": int(n_cols), "nnz": int(nnz),
            "ptr": ptr, "col": col, "weight": weight}


def apply_map(op: dict, field):
    """y[level, r] = sum_k w[k] x[level, col[k]] over native cell r (float64), for a (levels, coarse cells) field."""
    import numpy as np
    x = np.asarray(field, np.float64)
    if x.ndim != 2 or x.shape[1] != op["cols"]:
        raise GradientMeshError(f"a field of shape {tuple(x.shape)} does not sit on the map's {op['cols']:,} coarse "
                                "cells")
    row_of = np.repeat(np.arange(op["rows"]), np.diff(op["ptr"]))
    y = np.empty((x.shape[0], op["rows"]), np.float64)
    for level in range(x.shape[0]):
        y[level] = np.bincount(row_of, weights=op["weight"] * x[level, op["col"]], minlength=op["rows"])
    return y


def carry_gradient(op: dict, gradient, area_coarse, area_native, inner=None):
    """A coarse-mesh gradient (covector, per cell) carried to the native mesh as a density:
    area_native * P (gradient / area_coarse), zero where ``inner`` is False (the native boundary zone)."""
    import numpy as np
    ac = np.asarray(area_coarse, np.float64).ravel()
    an = np.asarray(area_native, np.float64).ravel()
    if (ac.mean() < 1.0) != (an.mean() < 1.0):   # one mesh on the unit sphere, the other in m2: refuse the mix
        raise GradientMeshError(f"the two meshes' cell areas are in different units (coarse mean {ac.mean():.3e}, "
                                f"native mean {an.mean():.3e})")
    out = apply_map(op, np.asarray(gradient, np.float64) / ac[None, :]) * an[None, :]
    if inner is not None:
        out[:, ~np.asarray(inner, bool)] = 0.0
    return out


def boundary_inner(grid: Path, cells: int):
    """True for the native cells outside the limited-area boundary zone (bdyMaskCell == 0); all True when the grid
    has no mask."""
    import numpy as np
    from woof.netcdf_bridge import open_dataset
    with open_dataset(grid) as ds:
        if "bdyMaskCell" not in ds.variables:
            return np.ones(int(cells), bool)
        mask = np.asarray(ds.variables["bdyMaskCell"][:]).ravel()
    if mask.size != int(cells):
        raise GradientMeshError(f"grid {grid} has a bdyMaskCell of {mask.size} cells; the native mesh has {cells}")
    return mask == 0
