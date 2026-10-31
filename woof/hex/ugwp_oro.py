"""The GSL drag's orographic statistics file (``<mesh>.ugwp_oro_data.nc``) for any hex mesh.

THE BREAKAGE THIS PREVENTS: the RRFS-shape suite (``--physics-suite
rrfs_v2_dev``, and any selection of ``config_gwdo_scheme='bl_ugwp_gwdo'``)
runs the GSL orographic drag, which reads twenty per-cell statistics (var2d,
con, oa1-4, ol1-4 at large and small scale) from the mesh's
``ugwp_oro_data.nc``.  RRFS publishes that file only for its own meshes
(conus3km, conus12km), so on every other mesh -- a global, a zoom globe, a
region, a point mesh, a storm-following member -- the forecast refused before
its first step: "--orographic-statistics names a missing file
.../region-adb5903adc0b.ugwp_oro_data.nc" (3 km region from GFS 2026-10-23
18Z f348, box L, 2026-10-24).

This module builds the file for any mesh, from the 30-arc-second terrain the
static already reads (``topo_gmted2010_30s``), with the Rust generator
``rw_hex_ugwp_oro`` (rust/ugwp-oro: the method, its papers, every core).  The
grid is read through the engine's Rust NetCDF bridge; the twenty fields are
written beside the grid with RRFS's own names, units and long names.

Per cell, the statistics box has the side of the cell's nominal spacing,
MPAS's ``len_disp / meshDensity**0.25`` (``nominalMinDc`` and
``meshDensity`` on the grid), which on RRFS's meshes is 12 km and 3 km; where
a grid carries no usable density the cell's own mean ``dcEdge`` is used.

Validated against RRFS's published files on RRFS's own meshes (conus12km,
157,859 cells; conus3km, 2,341,818 cells): docs/physics-suites.md has the
table.  The file is a product of the mesh and the terrain alone, so it is
stored in the per-mesh preparation store with the static
(``woof.hex.prep_store``, kind ``ugwp_oro``) and fetched, never rebuilt, where
the store holds it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "gpuwm-hex.ugwp-oro-data/v1"
#: Bump when the method (rust/ugwp-oro defaults) changes: it keys the store.
METHOD = "hex-ugwp-oro/1"
TOPO_DATASET = "topo_gmted2010_30s"
BINARY_ENV = "WOOF_HEX_RW_UGWP_ORO"
FILE_SUFFIX = ".ugwp_oro_data.nc"
EARTH_RADIUS_M = 6371229.0

#: name -> (units, long_name), exactly as RRFS's files carry them.
VARIABLES: dict[str, tuple[str, str]] = {}
for _scale, _what in (("ls", "meso-scale GWD"), ("ss", "small-scale GWD")):
    VARIABLES[f"var2d{_scale}"] = ("m", f"standard deviation of subgrid-scale orography ({_what})")
    VARIABLES[f"con{_scale}"] = ("unitless", f"orographic convexity ({_what})")
    for _k, _flow in enumerate(("westerly", "southerly", "south-westerly", "north-westerly"), start=1):
        VARIABLES[f"oa{_k}{_scale}"] = (
            "unitless", f"asymmetry of subgrid-scale orography for {_flow} flow ({_what})")
    for _k, _flow in enumerate(("westerly", "southerly", "south-westerly", "north-westerly"), start=1):
        VARIABLES[f"ol{_k}{_scale}"] = ("unitless", f"effective orographic length for {_flow} flow ({_what})")
ORDER = tuple(VARIABLES)


class UgwpOroError(RuntimeError):
    """The statistics could not be built; the message names what is missing."""


def binary() -> Path:
    """The generator: $WOOF_HEX_RW_UGWP_ORO, the wheel's libexec, a source-tree build, then PATH."""
    name = "rw_hex_ugwp_oro" + (".exe" if os.name == "nt" else "")
    explicit = os.environ.get(BINARY_ENV)
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise UgwpOroError(f"{BINARY_ENV}={path} does not exist")
        return path
    here = Path(__file__).resolve()
    for candidate in (here.with_name("libexec") / name,
                      here.parents[2] / "rust/ugwp-oro/target/release" / name):
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    raise UgwpOroError(
        "rw_hex_ugwp_oro is not installed (it ships in the woof hex wheel's woof/hex/libexec). "
        f"From a source tree: cargo build --release --manifest-path rust/ugwp-oro/Cargo.toml, or set {BINARY_ENV}.")


def topo_dir(geog: str | Path) -> Path:
    from .prep_store import resolve_geog_dataset_dir
    where = resolve_geog_dataset_dir(Path(geog).expanduser(), TOPO_DATASET)
    if not (where / "index").is_file():
        raise UgwpOroError(
            f"{where} has no index: the orographic statistics read the 30-arc-second terrain "
            f"({TOPO_DATASET}); stage it with `woof fetch-geog --datasets mesh`")
    return where


def topo_fingerprint(geog: str | Path) -> dict[str, Any]:
    """Every tile's name and size plus the index digest (as prep_store fingerprints a geography)."""
    where = topo_dir(geog)
    listing = sorted([p.name, p.stat().st_size] for p in where.iterdir() if p.is_file())
    blob = json.dumps(listing, separators=(",", ":")).encode()
    return {"dataset": TOPO_DATASET, "files": hashlib.sha256(blob).hexdigest(), "n_files": len(listing),
            "index_sha256": hashlib.sha256((where / "index").read_bytes()).hexdigest()}


def default_path(grid: str | Path, mesh: str) -> Path:
    """Where the forecast door looks: ``<mesh>.ugwp_oro_data.nc`` beside the grid (RRFS's layout)."""
    return Path(grid).parent / f"{mesh}{FILE_SUFFIX}"


def _attribute(dataset, name):
    for get in (lambda: dataset.getncattr(name), lambda: getattr(dataset, name)):
        try:
            return get()
        except Exception:
            continue
    return None


def cell_boxes(grid: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """latCell, lonCell (rad) and each cell's statistics box side (m), read through the Rust bridge."""
    from woof.netcdf_bridge import open_dataset

    dataset = open_dataset(Path(grid))
    radius = _attribute(dataset, "sphere_radius")
    radius = float(radius) if radius is not None else 1.0
    scale = EARTH_RADIUS_M if radius < 10.0 else EARTH_RADIUS_M / radius
    lat = np.ascontiguousarray(np.asarray(dataset.variables["latCell"][:], dtype="<f8"))
    lon = np.ascontiguousarray(np.asarray(dataset.variables["lonCell"][:], dtype="<f8"))
    n = lat.size
    edges = np.asarray(dataset.variables["edgesOnCell"][:], dtype=np.int64) - 1
    count = np.asarray(dataset.variables["nEdgesOnCell"][:], dtype=np.int64)
    dc = np.asarray(dataset.variables["dcEdge"][:], dtype=np.float64) * scale
    valid = (np.arange(edges.shape[1])[None, :] < count[:, None]) & (edges >= 0) & (edges < dc.size)
    local = np.where(valid, dc[np.clip(edges, 0, dc.size - 1)], 0.0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    side = local.copy()
    how = {"local_mean_dcEdge": int(n), "nominal": 0}
    names = set(dataset.variables)
    if "meshDensity" in names and "nominalMinDc" in names:
        nominal = float(np.asarray(dataset.variables["nominalMinDc"][:]).reshape(-1)[0]) * scale
        density = np.asarray(dataset.variables["meshDensity"][:], dtype=np.float64)
        if nominal > 0 and density.shape == (n,) and np.all(density > 0):
            spacing = nominal / density ** 0.25
            # MPAS's own spacing where the grid's density follows its convention
            # (spacing ~ density**-1/4); the cell's own edges where it does not.
            agree = (spacing > 0.75 * local) & (spacing < 1.33 * local)
            side = np.where(agree, spacing, local)
            how = {"local_mean_dcEdge": int((~agree).sum()), "nominal": int(agree.sum()),
                   "nominal_min_dc_m": nominal}
    if not np.all(np.isfinite(side) & (side > 0)):
        raise UgwpOroError(f"{grid}: a cell has no positive spacing (edgesOnCell/dcEdge)")
    return lat, lon, np.ascontiguousarray(side, dtype="<f8"), how


def write(path: Path, fields: np.ndarray, *, provenance: Mapping[str, Any]) -> None:
    """The twenty fields as RRFS lays them out: float32 [nCells], RRFS's names, units and long names."""
    import netCDF4

    n = fields.shape[1]
    partial = path.with_name(path.name + ".partial")
    partial.unlink(missing_ok=True)
    with netCDF4.Dataset(partial, "w", format="NETCDF3_64BIT_OFFSET") as out:
        out.createDimension("nCells", n)
        for k, name in enumerate(ORDER):
            units, long_name = VARIABLES[name]
            var = out.createVariable(name, "f4", ("nCells",))
            var.units = units
            var.long_name = long_name
            var[:] = fields[k]
        out.Conventions = "MPAS"
        out.source = "woof hex rw_hex_ugwp_oro"
        out.method = METHOD
        out.provenance = json.dumps(provenance, sort_keys=True)
    os.replace(partial, path)


def build(grid: str | Path, out: str | Path, geog: str | Path, *, receipt: str | Path | None = None,
          threads: int | None = None) -> dict[str, Any]:
    """Build ``out`` for ``grid`` from the terrain under ``geog``; returns (and writes) the receipt."""
    grid, out = Path(grid), Path(out)
    started = time.perf_counter()
    lat, lon, side, how = cell_boxes(grid)
    read_s = time.perf_counter() - started
    exe = binary()
    with tempfile.TemporaryDirectory(prefix="ugwp-oro-", dir=out.parent) as tmp:
        cells, values = Path(tmp) / "cells.bin", Path(tmp) / "stats.bin"
        with open(cells, "wb") as stream:
            stream.write(np.uint64(lat.size).tobytes())
            for array in (lat, lon, side):
                stream.write(array.tobytes())
        argv = [str(exe), "--cells", str(cells), "--topo", str(topo_dir(geog)), "--out", str(values)]
        if threads:
            argv += ["--threads", str(int(threads))]
        result = subprocess.run(argv, capture_output=True, text=True)
        if result.returncode != 0:
            raise UgwpOroError(f"rw_hex_ugwp_oro exited {result.returncode}: {result.stderr.strip()}")
        stats = json.loads(result.stdout)
        fields = np.fromfile(values, dtype="<f4").reshape(len(ORDER), lat.size)
    if list(stats["variables"]) != list(ORDER):
        raise UgwpOroError(f"rw_hex_ugwp_oro wrote {stats['variables']}, expected {list(ORDER)}")
    if not np.all(np.isfinite(fields)):
        raise UgwpOroError("rw_hex_ugwp_oro wrote non-finite statistics")
    provenance = {"schema": SCHEMA, "method": METHOD, "grid": grid.name,
                  "terrain": topo_fingerprint(geog), "box_side": how}
    write(out, fields, provenance=provenance)
    record = {**provenance, "out": str(out), "cells": int(lat.size), "generator": stats,
              "generator_binary": str(exe),
              "seconds": {"grid_read": round(read_s, 2), "total": round(time.perf_counter() - started, 2)},
              "summary": {name: {"mean": float(fields[k].mean()), "max": float(fields[k].max()),
                                 "nonzero_fraction": float(np.mean(fields[k] != 0))}
                          for k, name in enumerate(ORDER) if name in ("var2dls", "var2dss", "conss", "ol1ss")}}
    if receipt is not None:
        Path(receipt).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def ensure(grid: str | Path, out: str | Path, geog: str | Path | None = None, *,
           receipt: str | Path | None = None, log=None) -> dict[str, Any]:
    """``out`` for ``grid``: kept if present, else fetched from the preparation store, else built (and staged).

    ``geog`` defaults to the root the static builder reads (``mesh_point.resolve_geog``).
    """
    out = Path(out)
    if out.is_file():
        return {"source": "present", "out": str(out)}
    if geog is None:
        from .mesh_point import PointPlanRefusal, resolve_geog
        try:
            geog = resolve_geog(None)
        except PointPlanRefusal as error:
            # The door turns this into its own refusal naming the statistics.
            raise UgwpOroError(str(error)) from error
    from . import prep_store
    say = log or (lambda text: print(text, file=sys.stderr, flush=True))
    started = time.perf_counter()
    got = prep_store.ugwp_oro(Path(grid), Path(geog), out,
                              receipt=None if receipt is None else Path(receipt),
                              builder=lambda: build(grid, out, geog, receipt=receipt))
    say(f"ugwp_oro_data: {out.name} {got['source']} in {time.perf_counter() - started:.1f} s")
    return {**got, "out": str(out)}


def suite_reads_ugwp(suite: str | None, physics: Sequence[str] = ()) -> bool:
    """True when the named MPAS-physics suite (or a --physics override) selects the GSL drag."""
    for item in physics:
        key, _, value = str(item).partition("=")
        if key.strip() == "config_gwdo_scheme":
            return value.strip() == "bl_ugwp_gwdo"
    if not suite:
        return False
    from importlib.resources import files
    table = json.loads(files("woof.hex").joinpath("data/profiles/mpas-physics-suites.v1.json").read_text())
    rows = table.get("suites", [])
    for row in rows:
        if row.get("name") == suite:
            return row.get("selectors", {}).get("config_gwdo_scheme") == "bl_ugwp_gwdo"
    # MPAS's own suites and the RRFS fork's (hrrrv5, the rrfs_* suites) declare their options in mpas_suites.
    # THE BREAKAGE THIS PREVENTS (hrrrv5 hex region, 2026-10-29): only the column rows above were read, so a
    # door's prepare stage built nothing for hrrrv5 and its forecast died on the drag's Registry zeros.
    options = (table.get("mpas_suites") or {}).get("suites", {}).get(suite)
    if isinstance(options, dict):
        return options.get("config_gwdo_scheme") == "bl_ugwp_gwdo"
    return False


def forecast_args_read_ugwp(argv: Sequence[str]) -> bool:
    """Does a forecast argv (``--physics-suite X``, ``--physics K=V``, ``--forecast-arg=...``) select the GSL drag?"""
    suite, physics = None, []
    items = []
    for item in argv:
        text = str(item)
        if text.startswith("--forecast-arg="):
            text = text.split("=", 1)[1]
        items.append(text)
    i = 0
    while i < len(items):
        text = items[i]
        flag, eq, value = text.partition("=")
        if flag in ("--physics-suite", "--physics"):
            if not eq:
                value = items[i + 1] if i + 1 < len(items) else ""
                i += 1
            if flag == "--physics-suite":
                suite = value
            else:
                physics.append(value)
        i += 1
    return suite_reads_ugwp(suite, physics)


def prepare_for_forecast(forecast_argv: Sequence[str], grid: str | Path, geog: str | Path | None = None,
                         *, log=None) -> dict[str, Any] | None:
    """At a door's prepare stage: build (or fetch) the file the planned forecast will read, before any card.

    None when the planned forecast's physics reads no GSL drag statistics, or
    when it names its own ``--orographic-statistics``.
    """
    argv = [str(a) for a in forecast_argv]
    if "--orographic-statistics" in argv or not forecast_args_read_ugwp(argv):
        return None
    mesh = argv[argv.index("--mesh") + 1] if "--mesh" in argv else Path(grid).name.split(".")[0]
    return ensure(grid, default_path(grid, mesh), geog, log=log)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m woof.hex.ugwp_oro", description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="default: <grid dir>/<--mesh>.ugwp_oro_data.nc")
    parser.add_argument("--mesh", help="mesh name for the default output path")
    parser.add_argument("--geog", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--no-store", action="store_true", help="build here; do not consult the preparation store")
    args = parser.parse_args(argv)
    out = args.out or (default_path(args.grid, args.mesh) if args.mesh else None)
    if out is None:
        parser.error("--out or --mesh is required")
    if args.no_store:
        from .mesh_point import resolve_geog
        record = build(args.grid, out, resolve_geog(args.geog), receipt=args.receipt)
    else:
        record = ensure(args.grid, out, args.geog, receipt=args.receipt)
    print(json.dumps(record, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
