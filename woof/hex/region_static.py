"""A limited-area region's parent static, built from the cut's halo and the parent's terrain.

THE BREAKAGE THIS PREVENTS (WOOF customer-shape canary gate, 2026-10-24, rows
hex-region cold run_5b05d795abf04906 and warm run_6a1b556a3f26b66e): a 600 km
region spent 15.5 min of its 19.8 and 21.8 min click to first map in
"Preparing the region", on every new box.  ``country``'s region prepare built
the static of the WHOLE variable-resolution parent globe (45,227 cells, mostly
at the 120 km background) only so it could initialize the parent on a closed
sphere and cut about 1,600 cells out of it.  That static reads every
30-arc-second dataset of the whole globe, about 29 GB of WPS_GEOG, and a new
box pulls its root volume lazily from the snapshot at about 30 MB/s; on a
warm 16-vCPU box the arithmetic alone was about 4 min.

What the cut takes from the parent is the same with far less work:

* the cut's own static values are its cells' pixels, their first rings and the
  values carried to land cells with no pixel of their own.  They are built on a
  HALO, a wider cut of the parent, exactly as ``mesh_point``'s halo route
  builds a point region's static, and held to the same proof
  (:func:`woof.hex.mesh_point.check_halo_static`: the cut lies more rings inside
  the halo's rim than the builder carried values plus the stencils);
* the cut's vertical grid is smoothed level on level over the parent's terrain
  (about 3,000 passes at the top, ``woof.hex.vertical``), so it depends on
  terrain far beyond the cut.  ``rw_mpas_static --terrain-only`` writes the
  parent's exact terrain and its geometry (``ter``, the scaled coordinates,
  lengths, areas, weights and ``deriv_two``, the same bytes the full build
  writes) from the terrain dataset alone.

The parent static the door initializes is those two put together: the
terrain overlay and the parent grid's topology everywhere, the halo's values
on every halo cell, edge and vertex, and outside the halo the nearest halo
element's value.  Those outside values feed only parent columns the cull
throws away; inside the cut every variable is the full build's.  When the halo
check does not hold (twice, the second time twice as wide), the door builds
the whole parent's static as before and says why.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

EARTH_RADIUS_KM = 6371.229
#: Halo beyond the cut's farthest cell: the static's stencil rings and carried
#: values (``check_halo_static`` measures the reach on every build and the
#: halo is doubled once when it does not hold).
DEFAULT_HALO_MARGIN_KM = 600.0
MESH_DIMS = ("nCells", "nEdges", "nVertices")
KIND = {"nCells": "Cell", "nEdges": "Edge", "nVertices": "Vertex"}


def cut_reach_km(grid: Path, center_deg: tuple[float, float]) -> float:
    """The great-circle distance from the region's centre to the farthest cell of a cut grid."""

    import numpy as np
    from netCDF4 import Dataset

    with Dataset(str(grid)) as dataset:
        dataset.set_auto_maskandscale(False)
        lat = np.asarray(dataset.variables["latCell"][:], dtype=np.float64)
        lon = np.asarray(dataset.variables["lonCell"][:], dtype=np.float64)
    lat0, lon0 = (math.radians(v) for v in center_deg)
    cosine = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)
    return float(EARTH_RADIUS_KM * np.arccos(np.clip(cosine, -1.0, 1.0)).max())


def _coords(path: Path, kind: str):
    import numpy as np
    from netCDF4 import Dataset

    with Dataset(str(path)) as dataset:
        dataset.set_auto_maskandscale(False)
        xyz = np.stack([np.asarray(dataset.variables[f"{axis}{kind}"][:], dtype=np.float64) for axis in "xyz"], axis=1)
    return xyz / np.linalg.norm(xyz, axis=1, keepdims=True)


#: Topology tables padded past an element's own count, and the count that says where the padding starts.
PADDED = {"nCells": "nEdgesOnCell", "nEdges": "nEdgesOnEdge"}


def _pad_like_halo(values, name: str, var, grid, halo) -> None:
    """Fill a topology table's padded slots the way the static builder does, read off the halo static.

    The grid file repeats an entry in the slots past ``nEdgesOnCell``; gpuwm 2.9.7's builder writes 0 there
    (measured on this row: ``cellsOnCell`` and ``edgesOnCell`` in 180,920 padded slots of the parent).  When
    every padded slot of the halo's table holds one value, the parent's padded slots take it; otherwise the
    builder kept the grid's padding and so does this.
    """

    import numpy as np

    count = PADDED.get(var.dimensions[0]) if var.ndim == 2 else None
    if count is None or count not in grid.variables or count not in halo.variables:
        return
    width = np.arange(var.shape[1])[None, :]
    halo_pad = width >= np.asarray(halo.variables[count][:])[:, None]
    seen = np.unique(np.asarray(var[:])[halo_pad])
    if seen.size == 1:
        values[width >= np.asarray(grid.variables[count][:])[:, None]] = seen[0]


def assemble_parent_static(*, parent_grid: Path, halo_grid: Path, halo_static: Path, terrain: Path,
                           out: Path) -> dict[str, Any]:
    """Write the parent static: terrain overlay + parent topology everywhere, halo values on the halo.

    Every variable of the halo static is written, in its order, at the parent's size.  A variable the
    terrain overlay holds is the overlay's (the full build's own bytes); one the parent grid holds is the
    grid's (topology, cast to the static's type); a boundary mask is zero (the parent is a closed sphere);
    any other mesh variable is the halo's on the halo's elements and the nearest halo element's beyond it.
    """

    import numpy as np
    from netCDF4 import Dataset
    from scipy.spatial import cKDTree

    from .mesh_point import _element_index

    started = time.perf_counter()
    tmp = out.with_name(out.name + ".part")
    tmp.unlink(missing_ok=True)
    where: dict[str, Any] = {}
    nearest: dict[str, Any] = {}
    counts: dict[str, int] = {}
    from .netcdf_define import deferred_definition

    # Every variable and attribute is defined in one define session before any data is written: defined one at a
    # time after the first variable's data, each definition moved the whole data section again (26 s of one core
    # on a 3 km region's parent, 2026-10-27).  The file's bytes are the same (woof.hex.netcdf_define).
    with Dataset(str(parent_grid)) as grid, Dataset(str(halo_static)) as halo, Dataset(str(terrain)) as overlay, \
            deferred_definition(Dataset)(str(tmp), "w", format=halo.data_model, keepweakref=True) as dst:
        for source in (grid, halo, overlay):
            source.set_auto_maskandscale(False)
        dst.set_auto_maskandscale(False)
        for dim in MESH_DIMS:
            counts[dim] = len(grid.dimensions[dim])
            kind = KIND[dim]
            where[dim] = _element_index(halo_grid, parent_grid, kind)
            outside = np.ones(counts[dim], dtype=bool)
            outside[where[dim]] = False
            parent_xyz = _coords(parent_grid, kind)
            _, k = cKDTree(parent_xyz[where[dim]]).query(parent_xyz[outside], workers=-1)
            nearest[dim] = (outside, np.asarray(k, dtype=np.int64))
        for name, dim in halo.dimensions.items():
            size = counts.get(name, None if dim.isunlimited() else len(dim))
            dst.createDimension(name, size)
        # The halo is a limited-area file and stores its coordinates in binary32; the parent's are the terrain
        # overlay's binary64 (the full build's own), and so is the attribute that says which.
        attrs = {key: halo.getncattr(key) for key in halo.ncattrs()}
        attrs.update({key: overlay.getncattr(key) for key in overlay.ncattrs() if key.startswith("rw_coordinate_")})
        sources: dict[str, str] = {}
        defined = {}
        with dst.definition_batch():
            dst.setncatts(attrs)
            for name, var in halo.variables.items():
                dtype = overlay.variables[name].dtype if name in overlay.variables else var.dtype
                out_var = dst.createVariable(name, dtype, var.dimensions)
                out_var.setncatts({key: var.getncattr(key) for key in var.ncattrs()})
                defined[name] = out_var
            dst.setncattr("rw_hex_region_parent_static", json.dumps({
                "schema": "gpuwm-hex.region-parent-static/v1",
                "note": "terrain overlay and parent topology everywhere; halo static on the halo, nearest halo "
                        "element beyond it (only columns the cull discards)",
                "halo_static": str(halo_static), "terrain": str(terrain), "parent_grid": str(parent_grid)}))
        for name, var in halo.variables.items():
            out_var = defined[name]
            mesh_dim = var.dimensions[0] if var.dimensions and var.dimensions[0] in MESH_DIMS else None
            if name in overlay.variables:
                values, sources[name] = np.asarray(overlay.variables[name][:]), "terrain"
            elif name in grid.variables and mesh_dim is not None:
                values, sources[name] = np.asarray(grid.variables[name][:]).astype(var.dtype), "grid"
                _pad_like_halo(values, name, var, grid, halo)
            elif name.startswith("bdyMask") and mesh_dim is not None:
                values, sources[name] = np.zeros(out_var.shape, dtype=var.dtype), "closed sphere"
            elif mesh_dim is not None:
                source = np.asarray(var[:])
                values = np.empty((counts[mesh_dim], *source.shape[1:]), dtype=var.dtype)
                values[where[mesh_dim]] = source
                outside, k = nearest[mesh_dim]
                values[outside] = source[k]
                sources[name] = "halo"
            else:
                values, sources[name] = np.asarray(var[:]), "halo"
            out_var[...] = values
    tmp.replace(out)
    return {"seconds": round(time.perf_counter() - started, 2), "sources": sources,
            "halo_elements": {dim: int(where[dim].size) for dim in MESH_DIMS}, "parent_elements": counts}


def build_parent_static(*, parent_grid: Path, region: Path, center_deg: tuple[float, float], geog: Path | None,
                        mesh_exe: Path, static_exe: Path, out: Path, receipt: Path, work: Path,
                        run: Callable[[list[str]], Any], margin_km: float = DEFAULT_HALO_MARGIN_KM,
                        log=print) -> dict[str, Any]:
    """Build ``out`` (the parent static the region initializes) from a halo and the terrain overlay.

    ``run`` builds an ``rw_mpas_static`` argv (the door's prep-store route).  Returns the record; its
    ``built`` is False, with ``why``, when the halo did not hold and the caller must build the whole
    parent's static.
    """

    from .cull_door import carry_lineage, cull_one
    from .mesh_point import check_halo_static

    started = time.perf_counter()
    record: dict[str, Any] = {"schema": "gpuwm-hex.region-parent-static-build/v1", "built": False, "legs": {}}

    def leg(name: str, action: Callable[[], Any]) -> Any:
        t0 = time.perf_counter()
        try:
            return action()
        finally:
            record["legs"][name] = round(time.perf_counter() - t0, 2)

    geog_args = ["--geog", str(geog)] if geog else []
    cut = work / "region-cut.grid.nc"
    leg("cut-probe", lambda: cull_one(mesh_exe, parent_grid, region, cut, clobber=True))
    reach = cut_reach_km(cut, center_deg)
    record["cut_reach_km"] = round(reach, 1)
    halo_grid, halo_static = work / "halo.grid.nc", work / "halo.static.nc"
    halo_receipt = work / "halo.static-receipt.json"
    checks = []
    held = None
    for attempt, margin in enumerate((margin_km, 2.0 * margin_km)):
        halo_region = work / "halo-region.json"
        halo_region.write_text(json.dumps({"kind": "cap", "center_deg": list(center_deg),
                                           "radius_km": reach + margin}), encoding="utf-8")
        for stale in (halo_grid, halo_static, halo_receipt):
            stale.unlink(missing_ok=True)
        leg(f"halo-grid-{attempt}", lambda: cull_one(mesh_exe, parent_grid, halo_region, halo_grid, clobber=True))
        # The parent grid's attributes onto the halo, as mesh_point's halo route does: among them the binary64
        # coordinate declaration, without which the builder takes a cut's coordinates as binary32 and its operators
        # and pixel binning differ from the parent's in the last bits (measured on this row: defc_a/b and
        # cell_gradient_coef_x/y in 1,605 of the cut's 1,629 cells, green fraction and soil in 30).
        carry_lineage(parent_grid, halo_grid, drives_boundaries=False)
        leg(f"halo-static-{attempt}", lambda: run([str(static_exe), "--grid", str(halo_grid), *geog_args,
                                                   "--out", str(halo_static), "--receipt", str(halo_receipt),
                                                   "--clobber"]))
        check = check_halo_static(halo_grid=halo_grid, parent_grid=parent_grid, static_receipt=halo_receipt,
                                  cut_grid=cut, log=log)
        check["margin_km"] = margin
        checks.append(check)
        if check.get("agreed"):
            held = margin
            break
        log(f"region static: the {margin:g} km halo does not hold ({check.get('why')})")
    record["halo_checks"] = checks
    if held is None:
        record["why"] = checks[-1].get("why")
        record["seconds"] = round(time.perf_counter() - started, 2)
        return record
    terrain = work / "terrain.static.nc"
    leg("terrain", lambda: run([str(static_exe), "--grid", str(parent_grid), *geog_args, "--out", str(terrain),
                                "--receipt", str(work / "terrain.static-receipt.json"), "--terrain-only", "--clobber"]))
    record["assembly"] = leg("assemble", lambda: assemble_parent_static(
        parent_grid=parent_grid, halo_grid=halo_grid, halo_static=halo_static, terrain=terrain, out=out))
    # The static receipt the parent's row names: the halo's (the builder's own), with how the parent was made.
    halo_doc = json.loads(halo_receipt.read_text(encoding="utf-8"))
    halo_doc["hex_region_parent_static"] = {k: v for k, v in record.items() if k != "assembly"}
    receipt.write_text(json.dumps(halo_doc, indent=2, default=str) + "\n", encoding="utf-8")
    record.update(built=True, halo_margin_km=held, seconds=round(time.perf_counter() - started, 2))
    for scratch in (terrain, cut, Path(f"{cut}.cull-receipt.json")):
        scratch.unlink(missing_ok=True)
    return record
