"""Per-rank partitioned start shards and parent host-memory release (zero whole-globe retention).

Two complementary modes:
1. ``WOOF_HEX_PER_RANK_START=1`` (diet mode on any plain ``init.nc``): serializes payload projection before
   global dynamics load so the parent never holds both global copies simultaneously, skips global
   ``SharedObjects(source_mesh)`` duplication, and runs ``gc.collect() + malloc_trim(0)`` immediately after cutting
   this host's local ranks.
2. ``WOOF_HEX_START_SHARDS=<dir>`` (true zero-whole-globe start): loads pre-cut per-rank start shards
   (``rankNNNN/start.pkl`` + ``rankNNNN/pool/``) for ONLY this host's ``local_ranks``, verified against 64-bit
   partition IDs, rank 0 and joined hosts never load the full globe into RAM.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
from typing import Any, Sequence

import numpy as np

ENV_START = "WOOF_HEX_PER_RANK_START"
ENV_SHARDS = "WOOF_HEX_START_SHARDS"
ENV_SAVE_SHARDS = "WOOF_HEX_SAVE_START_SHARDS"
SCHEMA = "gpuwm-hex.per-rank-start-shards/v1"


def enabled() -> bool:
    if os.environ.get(ENV_SHARDS):
        return True
    return os.environ.get(ENV_START, "0").strip().lower() in ("1", "true", "yes", "on")


def refuse_unsupported(layout: Any, config: Any = None) -> None:
    """Refuse the per-rank start by name where a rank still needs the global mesh it lets go of.

    THE BREAKAGE THIS PREVENTS (a multi-host edge case found on 20 Nov 2026, conus3km): with the word on,
    a limited-area run stopped in open_limited_area_dycore ("mesh has no MPAS field 'bdyMaskCell'"), because each
    rank's limited-area dycore reads its boundary masks from that mesh.  The RRFS fork's helicity, gathered in one
    card's vertex order on several ranks (WOOF_HEX_UH_RANK_ORDER, the default), reads it too, and stopped at the first
    diagnostics frame.  Both now stop before any work, naming the word.
    """
    if not enabled():
        return
    word = ENV_SHARDS if os.environ.get(ENV_SHARDS) else ENV_START
    configs = getattr(layout, "configs", None) or {}
    if configs.get("config_apply_lbcs", False):
        raise ValueError(f"{word} is refused on a limited-area mesh: it lets go of the global mesh each rank's "
                         "limited-area dycore reads its boundary masks (bdyMaskCell) from. Unset it for this run")
    if config is None:
        return
    try:
        from .route_table import output_diagnostics_type
        diagnostics = output_diagnostics_type(config)
    except Exception:
        return
    if getattr(diagnostics, "__name__", "") == "ForkOutputDiagnostics":
        from .rrfs.output_diagnostics_rrfs import UH_RANK_ORDER_WORD, uh_rank_order
        if uh_rank_order() != "native":
            raise ValueError(f"{word} is refused on this route: its updraft helicity is gathered in one card's "
                             "vertex order, read from the global mesh the word lets go of. Unset it, or set "
                             f"{UH_RANK_ORDER_WORD}=native")


def shard_dir() -> Path | None:
    raw = os.environ.get(ENV_SHARDS)
    return Path(raw).expanduser().resolve() if raw and raw.strip() else None


def save_shard_dir() -> Path | None:
    raw = os.environ.get(ENV_SAVE_SHARDS)
    return Path(raw).expanduser().resolve() if raw and raw.strip() else None


def partition_digest(partition: Any) -> str:
    h = hashlib.sha256()
    for attr in ("cell_l2g", "edge_l2g", "vertex_l2g"):
        arr = np.ascontiguousarray(np.asarray(getattr(partition, attr), dtype=np.int64))
        h.update(arr.tobytes())
    h.update(
        f"{int(partition.n_owned_cells)}:{int(partition.n_owned_edges)}:{int(partition.n_owned_vertices)}".encode()
    )
    return h.hexdigest()


def release_host_memory() -> tuple[float, float]:
    """Run GC + glibc malloc_trim(0); return (rss_before_gib, rss_after_gib)."""
    def _rss() -> float:
        try:
            with open("/proc/self/status") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1048576.0
        except OSError:
            pass
        return float("nan")

    before = _rss()
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    return before, _rss()


def save_rank_start_shards(
    out_dir: str | Path,
    partitions: Sequence[Any],
    prepared: Sequence[tuple[Any, Path, Any, float]],
    *,
    global_dimensions: dict[str, int],
    field_indices: Any = None,
) -> Path:
    """Persist per-rank cut inputs and projected pools so future runs start without loading the full globe."""
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    ranks_meta = []
    for rank, (part, item) in enumerate(zip(partitions, prepared)):
        if item is None:
            continue
        local_layout, local_payload, local_inputs, _ = item
        rdir = root / f"rank{rank:04d}"
        if rdir.exists():
            shutil.rmtree(rdir)
        rdir.mkdir(parents=True, exist_ok=True)
        # Drop any global back-pointer so the shard holds ONLY this rank's local mesh!
        local_mesh = local_inputs[0]
        if getattr(local_mesh, "partition_source_mesh", None) is not None:
            local_mesh.partition_source_mesh = None
        pool_dst = rdir / "pool"
        shutil.copytree(local_payload, pool_dst)
        start_pkl = rdir / "start.pkl"
        with start_pkl.open("wb") as handle:
            pickle.dump(dict(local_layout=local_layout, local_inputs=local_inputs), handle, protocol=5)
        ranks_meta.append(
            dict(
                rank=rank,
                partition_sha256=partition_digest(part),
                n_owned_cells=int(part.n_owned_cells),
                n_local_cells=int(part.n_local_cells),
                start_bytes=int(start_pkl.stat().st_size),
            )
        )
    if field_indices is not None:
        with (root / "field_indices.pkl").open("wb") as handle:
            pickle.dump(field_indices, handle, protocol=5)
    manifest = dict(
        schema=SCHEMA,
        ranks=len(partitions),
        global_dimensions={str(k): int(v) for k, v in global_dimensions.items()},
        has_field_indices=(field_indices is not None),
        shards=ranks_meta,
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


def load_field_indices(root: str | Path) -> Any:
    with (Path(root).resolve() / "field_indices.pkl").open("rb") as handle:
        return pickle.load(handle)


def load_rank_start_shard(
    root: str | Path,
    rank: int,
    partition: Any,
    *,
    configs_before: dict[str, Any] | None = None,
) -> tuple[Any, Path, Any, float]:
    """Load one rank's pre-cut start shard directly (zero global init.nc load)."""
    began = time.perf_counter()
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported start shard schema: {manifest.get('schema')!r}")
    by_rank = {int(s["rank"]): s for s in manifest["shards"]}
    if int(rank) not in by_rank:
        raise ValueError(f"start shard directory {root} does not contain rank {rank}")
    entry = by_rank[int(rank)]
    expected = partition_digest(partition)
    if entry.get("partition_sha256") != expected:
        raise ValueError(
            f"start shard rank {rank} partition mismatch ({entry.get('partition_sha256')} != {expected})"
        )
    rdir = root / f"rank{int(rank):04d}"
    with (rdir / "start.pkl").open("rb") as handle:
        data = pickle.load(handle)
    local_layout = data["local_layout"]
    local_layout.samehost_partition = True
    if configs_before is not None:
        local_layout.configs = dict(configs_before)
    local_inputs = data["local_inputs"]
    local_payload = rdir / "pool"
    return local_layout, local_payload, local_inputs, time.perf_counter() - began
