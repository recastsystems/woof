"""Per-rank owned-slice NetCDF output (zero rank-0 assembly) plus offline stitch and 0-ULP compare.

Enabled via ``WOOF_HEX_PER_RANK_OUTPUT=1`` (off by default, so the gathered single-file path stays unchanged).
Eliminates rank-0 TCP frame gathers and full-globe host assembly spikes that OOM large global runs (e.g. 10 km
x1.5898242 at 5.90M cells).  Every rank writes its own owned entities to ``<stem>.rankNNNN.nc`` beside a one-time
``rank-map.rankNNNN.npz`` carrying 64-bit global entity IDs (``cell_l2g``, ``edge_l2g``, ``vertex_l2g``).
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .partition_v842 import _REFERENCES

ENVIRONMENT = "WOOF_HEX_PER_RANK_OUTPUT"
_ENTITY_MAPS = {
    "nCells": ("cell_l2g", "n_owned_cells"),
    "nEdges": ("edge_l2g", "n_owned_edges"),
    "nVertices": ("vertex_l2g", "n_owned_vertices"),
}


def enabled() -> bool:
    """True when per-rank owned-slice output is requested (default False)."""
    return os.environ.get(ENVIRONMENT, "0").strip().lower() in ("1", "true", "yes", "on")


def rank_target(path: str | Path, rank: int) -> Path:
    """Rewrite ``history.2024-05-21_19.00.00.nc`` -> ``history.2024-05-21_19.00.00.rank0000.nc``."""
    target = Path(path)
    tag = f".rank{int(rank):04d}"
    if target.name.endswith(f"{tag}{target.suffix}"):
        return target
    if target.suffix:
        return target.with_name(f"{target.stem}{tag}{target.suffix}")
    return target.with_name(f"{target.name}{tag}")


def rank_map_path(out_dir: str | Path, rank: int) -> Path:
    return Path(out_dir) / f"rank-map.rank{int(rank):04d}.npz"


def write_rank_map(out_dir: str | Path, rank: int, partition: Any, global_layout: Any, *, n_ranks: int) -> Path:
    """Write this rank's 64-bit owned local-to-global entity maps once per run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = rank_map_path(out, rank)
    n_cells = int(partition.n_owned_cells)
    n_edges = int(partition.n_owned_edges)
    n_verts = int(partition.n_owned_vertices)
    cell_l2g = np.ascontiguousarray(np.asarray(partition.cell_l2g, dtype=np.int64)[:n_cells])
    edge_l2g = np.ascontiguousarray(np.asarray(partition.edge_l2g, dtype=np.int64)[:n_edges])
    vertex_l2g = np.ascontiguousarray(np.asarray(partition.vertex_l2g, dtype=np.int64)[:n_verts])
    meta = dict(
        schema="gpuwm-hex.per-rank-output-map/v1",
        rank=int(rank),
        n_ranks=int(n_ranks),
        owned=dict(nCells=n_cells, nEdges=n_edges, nVertices=n_verts),
        global_dimensions={
            k: int(global_layout.dimensions[k])
            for k in ("nCells", "nEdges", "nVertices")
            if k in global_layout.dimensions
        },
    )
    tmp = path.with_name(path.name + ".partial.npz")
    np.savez(
        tmp,
        cell_l2g=cell_l2g,
        edge_l2g=edge_l2g,
        vertex_l2g=vertex_l2g,
        meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    os.replace(tmp, path)
    return path


def rank_global_attrs(rank: int, n_ranks: int, global_layout: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "hex_per_rank_output": 1,
        "hex_rank": int(rank),
        "hex_n_ranks": int(n_ranks),
    }
    for dim in ("nCells", "nEdges", "nVertices"):
        if dim in global_layout.dimensions:
            attrs[f"hex_global_{dim}"] = int(global_layout.dimensions[dim])
    return attrs


def remap_owned_connectivity(
    snapshot: Mapping[str, Any],
    partition: Any,
    rows: Mapping[str, Any],
    global_layout: Any,
) -> dict[str, Any]:
    """Remap local 1-based mesh connectivity fields in a rank's owned snapshot back to global 1-based IDs."""
    from ..output import HistoryField

    out: dict[str, Any] = {}
    for name, item in snapshot.items():
        if not isinstance(item, HistoryField) or name not in rows:
            out[name] = item
            continue
        row, _ = rows[name]
        referent = _REFERENCES.get(row["name"]) if row["pool"] == "mesh" else None
        if referent is None:
            out[name] = item
            continue
        values = np.asarray(item.values)
        l2g = np.asarray(getattr(partition, _ENTITY_MAPS[referent][0]), dtype=np.int64)
        remapped = values.copy()
        positive = remapped > 0
        if np.any(positive):
            local = remapped[positive].astype(np.int64, copy=False)
            in_range = local <= len(l2g)
            clamped = np.minimum(local, max(1, len(l2g))) - 1
            global_dummy = int(global_layout.dimensions[referent]) + 1
            mapped = np.where(in_range, l2g[clamped] + 1 if len(l2g) else global_dummy, global_dummy)
            remapped[positive] = mapped.astype(remapped.dtype)
        out[name] = replace(item, values=remapped)
    return out


def _load_map(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        return dict(
            meta=meta,
            nCells=np.asarray(data["cell_l2g"], dtype=np.int64),
            nEdges=np.asarray(data["edge_l2g"], dtype=np.int64),
            nVertices=np.asarray(data["vertex_l2g"], dtype=np.int64),
        )


def _raw_words(array: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(array)
    if arr.dtype.kind == "f" and arr.dtype.itemsize in (4, 8):
        return arr.view(np.uint32 if arr.dtype.itemsize == 4 else np.uint64)
    return arr


def stitch_rank_files(
    rank_paths: Sequence[str | Path],
    map_paths: Sequence[str | Path],
    out_path: str | Path,
    *,
    netcdf_format: str = "NETCDF3_64BIT_DATA",
) -> Path:
    """Stitch per-rank owned NetCDF shards into a single global 64-bit NetCDF file."""
    import netCDF4

    if len(rank_paths) != len(map_paths) or not rank_paths:
        raise ValueError("rank_paths and map_paths must be non-empty and equal length")
    maps = [_load_map(p) for p in map_paths]
    global_dims = dict(maps[0]["meta"]["global_dimensions"])
    datasets = [netCDF4.Dataset(str(p), "r") for p in rank_paths]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".partial")
    try:
        ref = datasets[0]
        with netCDF4.Dataset(str(tmp), "w", format=netcdf_format) as dst:
            for attr in ref.ncattrs():
                if not attr.startswith("hex_"):
                    dst.setncattr(attr, ref.getncattr(attr))
            for dname, dim in ref.dimensions.items():
                if dim.isunlimited():
                    dst.createDimension(dname, None)
                elif dname in global_dims:
                    dst.createDimension(dname, int(global_dims[dname]))
                else:
                    dst.createDimension(dname, len(dim))
            for vname, var in ref.variables.items():
                dims = tuple(var.dimensions)
                axes = [(i, d) for i, d in enumerate(dims) if d in _ENTITY_MAPS]
                out_var = dst.createVariable(vname, var.datatype, dims)
                for attr in var.ncattrs():
                    out_var.setncattr(attr, var.getncattr(attr))
                if not axes:
                    out_var[...] = np.asarray(var[...])
                    continue
                if len(axes) > 1:
                    raise ValueError(f"{vname}: more than one decomposed entity axis: {dims}")
                axis, entity = axes[0]
                shape = [int(global_dims[d]) if d in global_dims else int(var.shape[i]) for i, d in enumerate(dims)]
                assembled = np.empty(shape, dtype=var.dtype)
                for ds, rmap in zip(datasets, maps):
                    ids = rmap[entity]
                    idx = [slice(None)] * len(shape)
                    idx[axis] = ids
                    assembled[tuple(idx)] = np.asarray(ds.variables[vname][...])
                out_var[...] = assembled
        os.replace(tmp, out_path)
        return out_path
    finally:
        for ds in datasets:
            ds.close()


def compare_sharded_vs_global(
    rank_paths: Sequence[str | Path],
    map_paths: Sequence[str | Path],
    global_path: str | Path,
) -> dict[str, Any]:
    """Compare per-rank NetCDF shards directly against a global NetCDF file word-for-word (0-ULP)."""
    import netCDF4

    maps = [_load_map(p) for p in map_paths]
    datasets = [netCDF4.Dataset(str(p), "r") for p in rank_paths]
    gds = netCDF4.Dataset(str(global_path), "r")
    try:
        total_words = 0
        diff_words = 0
        per_var: dict[str, dict[str, int]] = {}
        shared_vars = sorted(set(gds.variables) & set(datasets[0].variables))
        for vname in shared_vars:
            gvar = gds.variables[vname]
            dims = tuple(gvar.dimensions)
            gvals = _raw_words(np.asarray(gvar[...]))
            axes = [(i, d) for i, d in enumerate(dims) if d in _ENTITY_MAPS]
            if not axes:
                rvals = _raw_words(np.asarray(datasets[0].variables[vname][...]))
                n_diff = int(np.count_nonzero(rvals != gvals))
                n_tot = int(gvals.size)
            else:
                axis, entity = axes[0]
                n_diff = 0
                n_tot = int(gvals.size)
                seen = 0
                for ds, rmap in zip(datasets, maps):
                    ids = rmap[entity]
                    idx = [slice(None)] * gvals.ndim
                    idx[axis] = ids
                    rvals = _raw_words(np.asarray(ds.variables[vname][...]))
                    gslice = gvals[tuple(idx)]
                    n_diff += int(np.count_nonzero(rvals != gslice))
                    seen += int(rvals.size)
                if seen != n_tot:
                    raise ValueError(f"{vname}: shard word count {seen} != global {n_tot}")
            total_words += n_tot
            diff_words += n_diff
            if n_diff:
                per_var[vname] = dict(words=n_tot, differing=n_diff)
        return dict(
            variables_checked=len(shared_vars),
            total_words=int(total_words),
            differing_words=int(diff_words),
            bit_exact=(diff_words == 0),
            differing_variables=per_var,
        )
    finally:
        gds.close()
        for ds in datasets:
            ds.close()
