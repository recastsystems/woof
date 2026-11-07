"""Byte-only owned-region reconstruction for N-rank forecast boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import math
import operator
import struct
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

from woof.hex.partition_executor_v841 import (
    ExecutorError, classify_partition_axis, owned_block, scatter_owned_axis,
)
from woof.hex.partition_samehost_tables_v841 import build_all

__all__ = ["collect_owned", "reconstruct_boundary", "assemble_owned_arrays"]

_PLURAL = {"cell": "cells", "edge": "edges", "vertex": "vertices"}
_PADDED_CARRIERS = frozenset(
    f"__backend__/seam/arrays/driver/{name}/ru"
    for name in ("pbl_tendencies", "radiation_tendencies", "cumulus_tendencies")
)
_EXPECTED_METADATA = (
    "identity/constructor_identity_sha256",
    "identity/n_columns",
)
_ATMO_FIELDS = (
    ("rho", "state", "cell"),
    ("rho_theta", "state", "cell"),
    ("rho_u", "state", "edge"),
    ("rho_w", "state", "cell"),
    ("scalars", "state", "cell"),
    ("theta_m", "saved", "cell"),
    ("exner", "saved", "cell"),
    ("density_perturbation", "saved", "cell"),
    ("rho_theta_perturbation", "saved", "cell"),
    ("pressure_perturbation", "saved", "cell"),
    ("normal_velocity", "saved", "edge"),
    ("vertical_velocity", "saved", "cell"),
)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if type(value).__module__.split(".")[0] == "cupy":
        import cupy
        return cupy.asnumpy(value)
    return np.ascontiguousarray(np.asarray(value))


def _walk_leaves(prefix: str, item: Any, arrays: dict, scalars: dict) -> None:
    if isinstance(item, Mapping):
        for key in sorted(item):
            _walk_leaves(f"{prefix}/{key}" if prefix else str(key), item[key], arrays, scalars)
    elif isinstance(item, np.ndarray) or type(item).__module__.split(".")[0] == "cupy":
        arrays[prefix] = item
    elif isinstance(item, (str, int, float, bool)) or item is None:
        scalars[prefix] = item
    elif isinstance(item, (tuple, list)):
        for index, child in enumerate(item):
            _walk_leaves(f"{prefix}/{index}", child, arrays, scalars)
    else:
        scalars[prefix] = repr(item)


def _proof_module(proof: Any) -> Any:
    if proof is not None:
        return proof
    # The public forecast door loads its forecast module under this name.
    # Reuse that live module's bound proof rather than another import identity.
    for name in ("v841_forecast", "run_cuda_v841_forecast"):
        forecast = sys.modules.get(name)
        if forecast is not None and hasattr(forecast, "proof"):
            return forecast.proof
    return importlib.import_module("woof.hex.drivers.run_cuda_v841_full_physics_x4")


def collect_owned(layouts: Sequence[Any], rank: int, stack: Mapping[str, Any]) -> dict[str, Any]:
    """Keep owned bytes, native garbage, and named staggered artifacts distinct.

    Native regional state has one garbage cell and edge after each local
    entity extent. Physics runs that padded column too, so its backend fields
    include that tail. It must be retained in the global hash and must agree
    on every rank. Only the three unconsumed staggered-u carriers are artifacts.
    """
    layout = layouts[rank]
    atmosphere = stack["driver"].atmosphere
    regional = int(atmosphere.state.rho.shape[-1]) == int(layout.n_local_cells) + 1
    entries, blobs = [], []

    def add(path, array, kind, axis, *, padded=False, native_padding=False):
        host = _to_numpy(array)
        tail = (2 if regional else 1) if padded else int(native_padding)
        if kind == "none":
            block = host
        else:
            _, owned, local, _ = _entity_info(layout, kind)
            expected = local + tail
            if host.shape[axis] != expected:
                raise ExecutorError(f"{path}: local entity extent {host.shape[axis]} != {expected}")
            block = owned_block(host, axis, owned)
            if tail:
                slices = [slice(None)] * host.ndim
                slices[axis] = slice(-tail, None)
                block = np.ascontiguousarray(np.concatenate((block, host[tuple(slices)]), axis=axis))
        entries.append({
            "path": path, "dtype": block.dtype.str, "shape": list(block.shape),
            "kind": kind, "axis": axis, "padded": bool(padded),
            "native_padding": bool(native_padding),
        })
        blobs.append(block.tobytes(order="C"))

    for name, owner, kind in _ATMO_FIELDS:
        source = atmosphere.state if owner == "state" else atmosphere.saved
        array = getattr(source, name)
        add(f"__atmosphere__/{owner}/{name}", array, kind, array.ndim - 1,
            native_padding=regional)

    arrays, scalars = {}, {}
    # The wave state is the run's one whole-mesh service on rank 0 (waves.attach_arwen), not a partitioned
    # field: it has no owned slice to assemble. THE BREAKAGE THIS PREVENTS: with waves on and more than one
    # rank, rank 0's restart state carried the spectrum and the others only a marker, and the boundary writer
    # stopped the run ("boundary entry paths diverge: only on rank 0 ['__backend__/waves/forcing/psfc', ...]").
    state = dict(stack["backend"].restart_state())
    state.pop("waves", None)
    _walk_leaves("", state, arrays, scalars)
    for path in sorted(arrays):
        host = _to_numpy(arrays[path])
        full_path = f"__backend__/{path}"
        counts = {kind: _entity_info(layout, kind)[2] for kind in _PLURAL}
        if full_path in _PADDED_CARRIERS and host.ndim and host.shape[-1] == counts["cell"] + (2 if regional else 1):
            add(full_path, host, "cell", host.ndim - 1, padded=True)
            continue
        # Prefer native padded extents on regional fields, preserving both the
        # leading column axes and trailing MPAS field axes.
        matches = []
        if regional and host.ndim:
            for axis in dict.fromkeys((host.ndim - 1, 0)):
                matches = [(kind, axis) for kind, count in counts.items() if host.shape[axis] == count + 1]
                if matches:
                    break
            if len(matches) > 1:
                raise ExecutorError(f"{full_path}: native padded entity class is ambiguous")
        if matches:
            kind, axis = matches[0]
            add(full_path, host, kind, axis, native_padding=True)
            continue
        kind, axis = classify_partition_axis(
            host.shape, n_local_cells=counts["cell"],
            n_local_edges=counts["edge"], n_local_vertices=counts["vertex"],
        )
        padded = False
        if kind == "none" and host.ndim:
            for candidate, count in counts.items():
                if host.shape[-1] == count + 1:
                    kind, axis, padded = candidate, host.ndim - 1, True
                    break
        add(full_path, host, kind, axis, padded=padded)
    return {"entries": entries, "blobs": blobs, "backend_scalars": scalars,
            "time_seconds": float(atmosphere.state.time_seconds)}


def _entity_info(layout: Any, kind: str) -> tuple[np.ndarray, int, int, int]:
    plural = _PLURAL[kind]
    return (
        getattr(layout, f"{kind}_l2g"),
        int(getattr(layout, f"n_owned_{plural}")),
        int(getattr(layout, f"n_local_{plural}")),
        int(getattr(layout, f"_global_{plural}")),
    )


def _contiguous(block: np.ndarray) -> np.ndarray:
    return block.copy() if block.ndim == 0 else np.ascontiguousarray(block)


def _decode_collected(collected: Mapping[str, Any], rank: int) -> dict[str, Any]:
    decoded = "header" in collected
    header = collected["header"] if decoded else collected
    entries = header["entries"]
    if not decoded and len(entries) != len(collected["blobs"]):
        raise ExecutorError(f"rank {rank}: entry and blob counts differ")
    blocks: dict[str, np.ndarray] = {}
    by_path: dict[str, dict[str, Any]] = {}
    for position, original in enumerate(entries):
        entry = dict(original)
        path = entry["path"]
        if not isinstance(path, str) or not path.startswith(("__atmosphere__/", "__backend__/")):
            raise ExecutorError(f"rank {rank}: unrecognized boundary path {path!r}")
        if path in by_path:
            raise ExecutorError(f"rank {rank}: duplicate boundary path {path!r}")
        try:
            dtype = np.dtype(entry["dtype"])
            shape = tuple(operator.index(extent) for extent in entry["shape"])
        except (TypeError, ValueError) as exc:
            raise ExecutorError(f"rank {rank} {path}: invalid dtype or shape") from exc
        if dtype.hasobject or any(extent < 0 for extent in shape):
            raise ExecutorError(f"rank {rank} {path}: object dtype or negative shape")
        if decoded:
            block = np.asarray(collected["blocks"][path])
            if block.dtype != dtype or block.shape != shape:
                raise ExecutorError(f"rank {rank} {path}: block differs from its dtype/shape header")
        else:
            blob = collected["blobs"][position]
            nbytes = math.prod(shape) * dtype.itemsize
            if len(blob) != nbytes:
                raise ExecutorError(
                    f"rank {rank} {path}: blob has {len(blob)} bytes, header requires {nbytes}"
                )
            block = np.frombuffer(blob, dtype=dtype).reshape(shape)
        kind = entry["kind"]
        if kind not in (*_PLURAL, "none"):
            raise ExecutorError(f"rank {rank} {path}: unknown entity kind {kind!r}")
        axis = entry["axis"]
        if kind == "none":
            if axis is not None or entry.get("padded", False) or entry.get("native_padding", False):
                raise ExecutorError(f"rank {rank} {path}: a constant has an entity axis or pad")
        else:
            try:
                axis = operator.index(axis)
            except TypeError as exc:
                raise ExecutorError(f"rank {rank} {path}: entity axis must be an integer") from exc
            if not 0 <= axis < len(shape):
                raise ExecutorError(f"rank {rank} {path}: entity axis {axis} is outside shape {shape}")
        entry.update(dtype=dtype.str, shape=list(shape), axis=axis)
        if entry.get("padded", False) and entry.get("native_padding", False):
            raise ExecutorError(f"rank {rank} {path}: native garbage cannot also be an artifact")
        by_path[path] = entry
        blocks[path] = _contiguous(block)
    if decoded and set(collected["blocks"]) != set(by_path):
        raise ExecutorError(f"rank {rank}: decoded block paths differ from entry paths")
    scalars = header["backend_scalars"]
    if not isinstance(scalars, Mapping):
        raise ExecutorError(f"rank {rank}: backend scalar metadata is not a mapping")
    time_seconds = float(header["time_seconds"])
    if not math.isfinite(time_seconds):
        raise ExecutorError(f"rank {rank}: model time is not finite")
    return {"entries": by_path, "blocks": blocks, "scalars": dict(scalars), "time": time_seconds}


def _validate_pair_schema(path: str, entries: Sequence[Mapping[str, Any]], blocks: Sequence[np.ndarray]) -> tuple[str, int | None, bool, bool]:
    first = entries[0]
    kind, axis, padded = first["kind"], first["axis"], bool(first.get("padded", False))
    native_padding = bool(first.get("native_padding", False))
    shape = blocks[0].shape
    for rank, (entry, block) in enumerate(zip(entries, blocks)):
        if (entry["kind"], entry["axis"], bool(entry.get("padded", False)), bool(entry.get("native_padding", False))) != (kind, axis, padded, native_padding):
            raise ExecutorError(f"rank {rank} {path}: entity kind/axis/pad schema differs")
        if block.dtype != blocks[0].dtype:
            raise ExecutorError(f"rank {rank} {path}: dtype differs between ranks")
        if len(block.shape) != len(shape) or any(
            extent != shape[index]
            for index, extent in enumerate(block.shape)
            if index != axis
        ):
            raise ExecutorError(f"rank {rank} {path}: non-entity shape differs between ranks")
    return kind, axis, padded, native_padding


def _assemble_blocks(layouts: Sequence[Any], blocks: Sequence[np.ndarray], kind: str, axis: int, *, owned_only: bool, native_padding: bool = False) -> np.ndarray:
    shape = list(blocks[0].shape)
    shape[axis] = _entity_info(layouts[0], kind)[3] + int(native_padding)
    result = np.empty(tuple(shape), dtype=blocks[0].dtype)
    for rank, (layout, block) in enumerate(zip(layouts, blocks)):
        l2g, owned, local, _global = _entity_info(layout, kind)
        allowed = (owned + int(native_padding),) if owned_only else (owned + int(native_padding), local + int(native_padding))
        if block.shape[axis] not in allowed:
            raise ExecutorError(
                f"rank {rank} {kind}: entity axis {block.shape[axis]} is not an expected extent {allowed}"
            )
        slices = [slice(None)] * block.ndim
        slices[axis] = slice(0, owned)
        scatter_owned_axis(result, block[tuple(slices)], np.asarray(l2g), owned, axis)
        if native_padding:
            slices[axis] = slice(-1, None)
            tail = block[tuple(slices)]
            if rank == 0:
                result[tuple(slices)] = tail
            elif tail.tobytes() != result[tuple(slices)].tobytes():
                raise ExecutorError(f"rank {rank} {kind}: native garbage differs bytewise between ranks")
    return result


def reconstruct_boundary(layouts: Sequence[Any], collected_sequence: Sequence[Mapping[str, Any]], proof: Any = None) -> dict[str, Any]:
    """Fingerprint the full owned union in the existing single-card schema."""

    from woof.hex.cuda_dualrun import fingerprint_atmosphere

    layouts = tuple(layouts)
    collected_sequence = tuple(collected_sequence)
    if len(layouts) != len(collected_sequence):
        raise ExecutorError("boundary collection count differs from rank count")
    build_all(layouts)
    decoded = [_decode_collected(item, rank) for rank, item in enumerate(collected_sequence)]
    first = decoded[0]
    atmosphere_fields = _ATMO_FIELDS
    expected_atmosphere = {
        f"__atmosphere__/{owner}/{name}": kind
        for name, owner, kind in atmosphere_fields
    }
    actual_atmosphere = {path for path in first["entries"] if path.startswith("__atmosphere__/")}
    if actual_atmosphere != set(expected_atmosphere):
        raise ExecutorError("boundary does not cover the required atmosphere field schema")
    for path, kind in expected_atmosphere.items():
        entry = first["entries"][path]
        if entry["kind"] != kind or entry["axis"] != len(entry["shape"]) - 1 or entry.get("padded", False):
            raise ExecutorError(f"{path}: atmosphere entity schema differs from the collector law")
    metadata_diffs = []
    for rank, item in enumerate(decoded[1:], 1):
        if struct.pack("!d", item["time"]) != struct.pack("!d", first["time"]):
            raise ExecutorError(f"rank {rank}: ranks disagree about model time at a boundary")
        if set(item["entries"]) != set(first["entries"]):
            raise ExecutorError(f"rank {rank}: boundary entry paths diverge: only on rank 0 "
                                f"{sorted(set(first['entries']) - set(item['entries']))}, only on this rank "
                                f"{sorted(set(item['entries']) - set(first['entries']))}")
        if set(item["scalars"]) != set(first["scalars"]):
            raise ExecutorError(f"rank {rank}: backend scalar metadata paths diverge")
        for path in sorted(first["scalars"]):
            mine, theirs = first["scalars"][path], item["scalars"][path]
            same = json.dumps(mine, sort_keys=True) == json.dumps(theirs, sort_keys=True)
            if not same:
                if not any(marker in path for marker in _EXPECTED_METADATA):
                    raise ExecutorError(f"rank {rank}: backend scalar metadata differs at {path!r}")
                metadata_diffs.append({"rank": rank, "path": path, "rank0": mine, "value": theirs})

    assembled = {}
    artifacts = {}
    deferred = []
    for path in sorted(first["entries"]):
        entries = [item["entries"][path] for item in decoded]
        blocks = [item["blocks"][path] for item in decoded]
        kind, axis, padded, native_padding = _validate_pair_schema(path, entries, blocks)
        if padded:
            if path not in _PADDED_CARRIERS or kind != "cell" or axis != blocks[0].ndim - 1:
                raise ExecutorError(f"{path}: padded leaf is not a named staggered-u carrier")
            artifact = {"dtype": blocks[0].dtype.str, "kind": kind, "law": "batch-order staggered-u carrier; unconsumed by the seam"}
            for rank, (layout, block) in enumerate(zip(layouts, blocks)):
                owned = _entity_info(layout, kind)[1]
                regional = bool(first["entries"]["__atmosphere__/state/rho"].get("native_padding", False))
                tail = 2 if regional else 1
                if block.shape[axis] != owned + tail:
                    raise ExecutorError(f"rank {rank} {path}: padded carrier is not owned extent plus {tail}")
                artifact[f"rank{rank}_shape"] = list(block.shape)
                artifact[f"rank{rank}_sha256"] = hashlib.sha256(block.tobytes()).hexdigest()
            artifacts[path[len("__backend__/"):]] = artifact
        elif kind == "none":
            if any(block.tobytes() != blocks[0].tobytes() for block in blocks[1:]):
                raise ExecutorError(f"{path}: non-entity constants differ bytewise between ranks")
            assembled[path] = blocks[0]
        else:
            assert axis is not None
            assembled[path] = None  # keeps the sorted insertion order
            deferred.append((path, blocks, kind, axis, native_padding))
    gathered = _parallel_fields(
        lambda item: _assemble_blocks(layouts, item[1], item[2], item[3], owned_only=True,
                                      native_padding=item[4]), deferred)
    for (path, *_), value in zip(deferred, gathered):
        assembled[path] = value

    state = SimpleNamespace(time_seconds=first["time"])
    saved = SimpleNamespace()
    for name, owner, _kind in atmosphere_fields:
        path = f"__atmosphere__/{owner}/{name}"
        if path not in assembled:
            raise ExecutorError(f"boundary is missing required atmosphere field {path!r}")
        setattr(state if owner == "state" else saved, name, assembled[path])
    backend = {
        path[len("__backend__/"):]: value
        for path, value in assembled.items()
        if path.startswith("__backend__/")
    }
    if set(backend) & set(first["scalars"]):
        raise ExecutorError("backend array and scalar paths overlap")
    backend.update(first["scalars"])
    record = {
        "atmosphere": fingerprint_atmosphere(SimpleNamespace(state=state, saved=saved)),
        "backend": _proof_module(proof).fingerprint_nested_arrays(backend),
    }
    if artifacts:
        record["batch_order_artifacts"] = artifacts
    if metadata_diffs:
        record["rank_metadata_divergences"] = metadata_diffs
    return record


def assemble_owned_arrays(layouts: Sequence[Any], mappings: Sequence[Mapping[str, Any]], *, kinds: Mapping[str, str] | None = None,
                          native_pads: Sequence[str] = (), consume: bool = False) -> dict[str, np.ndarray]:
    """Assemble trailing-axis history arrays from local or owned-only blocks.

    ``kinds`` maps field names to cell/edge/vertex, or none for constants.
    Without an override, an entity class is inferred only when every rank's
    trailing extent names one unambiguous class.
    """

    layouts = tuple(layouts)
    mappings = tuple(mappings)
    if len(layouts) != len(mappings):
        raise ExecutorError("history collection count differs from rank count")
    build_all(layouts)
    paths = set(mappings[0])
    if any(set(mapping) != paths for mapping in mappings[1:]):
        raise ExecutorError("history field paths diverge between ranks")
    if kinds is not None and set(kinds) - paths:
        raise ExecutorError("history kind overrides name an absent field")
    if set(native_pads) - paths:
        raise ExecutorError("native padding overrides name an absent field")
    ordered = sorted(paths)
    # Each field is gathered independently; with the GIL disabled the numpy
    # scatters run in parallel.  Results are placed in the same sorted order.
    def assemble(path):
        result = _assemble_one(path, layouts, mappings, kinds, native_pads)
        if consume:
            # Each rank field dies after its scatter, before the next frame.
            for mapping in mappings:
                mapping.pop(path)
        return result
    results = _parallel_fields(assemble, ordered)
    return dict(zip(ordered, results))


def _gather_threads() -> int:
    import os
    text = os.environ.get("WOOF_HEX_GATHER_THREADS", "16")
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise ExecutorError("WOOF_HEX_GATHER_THREADS must be a positive integer")
    return value


def _parallel_fields(function, items):
    """Map ``function`` over ``items`` on WOOF_HEX_GATHER_THREADS threads, in order.

    Breakage it prevents: the coordinator gathered every history field one
    after another (about 27 s per 15 km frame with every rank stalled, 8-card RTX PRO 6000 host
    smoke 2026-10-10).  Each field's gather writes only its own output array.
    """
    items = list(items)
    workers = min(_gather_threads(), len(items))
    if workers <= 1:
        return [function(item) for item in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hex-gather") as pool:
        return list(pool.map(function, items))


def _assemble_one(path, layouts, mappings, kinds, native_pads):
    """One field of :func:`assemble_owned_arrays` (unchanged law, now per field)."""
    blocks = [np.asarray(mapping[path]) for mapping in mappings]
    if any(block.dtype.hasobject for block in blocks):
        raise ExecutorError(f"{path}: history arrays cannot have object dtype")
    kind = (kinds or {}).get(path)
    pad = None
    if path in native_pads:
        if kind not in _PLURAL:
            raise ExecutorError(f"{path}: native padding needs an explicit entity kind")
        for rank, (layout, block) in enumerate(zip(layouts, blocks)):
            if not block.ndim or block.shape[-1] != _entity_info(layout, kind)[2] + 1:
                raise ExecutorError(f"rank {rank} {path}: native garbage extent differs")
        pad = _contiguous(blocks[0][..., -1:])
        if any(block.dtype != pad.dtype or block[..., -1:].shape != pad.shape
               or _contiguous(block[..., -1:]).tobytes() != pad.tobytes() for block in blocks[1:]):
            raise ExecutorError(f"{path}: native garbage columns differ bytewise")
        blocks = [block[..., :-1] for block in blocks]
    if kind is None:
        candidates = []
        for candidate in _PLURAL:
            if all(
                block.ndim > 0 and block.shape[-1] in _entity_info(layout, candidate)[1:3]
                for layout, block in zip(layouts, blocks)
            ):
                candidates.append(candidate)
        if len(candidates) > 1:
            raise ExecutorError(f"{path}: history entity class is ambiguous; supply its kind")
        kind = candidates[0] if candidates else "none"
    if kind not in (*_PLURAL, "none"):
        raise ExecutorError(f"{path}: unknown history entity kind {kind!r}")
    axis = None if kind == "none" else blocks[0].ndim - 1
    if axis is not None and axis < 0:
        raise ExecutorError(f"{path}: a scalar has no history entity axis")
    entries = [{"kind": kind, "axis": axis, "padded": False} for _ in layouts]
    _validate_pair_schema(path, entries, blocks)
    if kind == "none":
        if any(block.tobytes() != blocks[0].tobytes() for block in blocks[1:]):
            raise ExecutorError(f"{path}: non-entity history constants differ bytewise")
        return _contiguous(blocks[0])
    assert axis is not None
    assembled = _assemble_blocks(layouts, blocks, kind, axis, owned_only=False)
    if pad is not None:
        assembled = np.ascontiguousarray(np.concatenate((assembled, pad), axis=axis))
    return assembled
