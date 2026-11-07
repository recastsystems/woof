"""CPU checks for N-rank boundary and history byte reconstruction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from woof.hex.cuda_dualrun import fingerprint_atmosphere
from woof.hex.partition_assets_v841 import PartitionLayout
from woof.hex.partition_executor_v841 import ExecutorError
from compare_2gpu_invariance_v841 import compare_boundary_records
import run_cuda_v841_forecast_2gpu as legacy
from woof.hex.drivers.samehost_identity_v841 import assemble_owned_arrays, collect_owned, reconstruct_boundary


def _layouts(n_ranks):
    result = []
    for rank in range(n_ranks):
        vectors = {}
        counts = {}
        for kind, total, halo in (("cell", 12, 2), ("edge", 24, 3), ("vertex", 36, 3)):
            owned = total // n_ranks
            vector = np.arange(rank * owned, (rank + 1) * owned, dtype=np.int32)
            if n_ranks > 1:
                peer_start = ((rank + 1) % n_ranks) * owned
                vector = np.concatenate((vector, np.arange(peer_start, peer_start + halo, dtype=np.int32)))
            vectors[kind] = vector
            counts[kind] = owned
        result.append(
            PartitionLayout(
                partition=rank,
                n_owned_cells=counts["cell"],
                n_owned_edges=counts["edge"],
                n_owned_vertices=counts["vertex"],
                cell_l2g=vectors["cell"],
                edge_l2g=vectors["edge"],
                vertex_l2g=vectors["vertex"],
                cell_ring=np.zeros(vectors["cell"].size, dtype=np.int16),
                cell_row_complete=np.ones(vectors["cell"].size, dtype=bool),
                edge_row_complete=np.ones(vectors["edge"].size, dtype=bool),
                vertex_row_complete=np.ones(vectors["vertex"].size, dtype=bool),
                _global_cells=12,
                _global_edges=24,
                _global_vertices=36,
            )
        )
    return result


def _truth():
    state = SimpleNamespace(time_seconds=120.0)
    saved = SimpleNamespace()
    for name, owner, kind in legacy._ATMO_FIELDS:
        count = 24 if kind == "edge" else 12
        shape = (6, 2, count) if name == "scalars" else (2, count)
        value = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        setattr(state if owner == "state" else saved, name, value)
    backend = {
        "soil": np.arange(36, dtype=np.float32).reshape(12, 3),
        "edge": np.arange(48, dtype=np.float32).reshape(2, 24),
        "constant": np.array([0, 0x80000000, 0x3F800000], dtype=np.uint32).view(np.float32),
        "identity": {"constructor_identity_sha256": "whole", "n_columns": 12},
        "schema": "test-backend/v1",
        "step": 1,
    }
    return SimpleNamespace(state=state, saved=saved), backend


def _collections(layouts, *, padded=False):
    atmosphere, backend = _truth()
    collections = []
    for rank, layout in enumerate(layouts):
        local_state = SimpleNamespace(time_seconds=120.0)
        local_saved = SimpleNamespace()
        for name, owner, kind in legacy._ATMO_FIELDS:
            source = getattr(atmosphere, owner)
            value = getattr(source, name)[..., getattr(layout, f"{kind}_l2g")].copy()
            value[..., getattr(layout, f"n_owned_{kind}s"):] = np.float32(-42)
            setattr(local_state if owner == "state" else local_saved, name, value)
        local_backend = {
            "soil": backend["soil"][layout.cell_l2g, :],
            "edge": backend["edge"][..., layout.edge_l2g],
            "constant": backend["constant"],
            "identity": {"constructor_identity_sha256": f"rank{rank}", "n_columns": layout.n_local_cells},
            "schema": backend["schema"],
            "step": backend["step"],
        }
        if padded:
            local_backend["seam"] = {
                "arrays": {
                    "driver/pbl_tendencies/ru": np.full((2, 1, layout.n_local_cells + 1), rank + 1, dtype=np.float32)
                }
            }
        stack = {
            "driver": SimpleNamespace(atmosphere=SimpleNamespace(state=local_state, saved=local_saved)),
            "backend": SimpleNamespace(restart_state=lambda payload=local_backend: payload),
        }
        collections.append(collect_owned(layouts, rank, stack))
    return collections, atmosphere, backend


@pytest.mark.parametrize("n_ranks", [1, 2, 3, 4])
def test_full_boundary_reconstruction_matches_one_card_hashes(n_ranks):
    layouts = _layouts(n_ranks)
    collected, atmosphere, backend = _collections(layouts)
    record = reconstruct_boundary(layouts, collected, proof=legacy.proof)
    reference = {
        "atmosphere": fingerprint_atmosphere(atmosphere),
        "backend": legacy.proof.fingerprint_nested_arrays(backend),
    }
    outcome = compare_boundary_records(record, reference)
    assert outcome["state_invariant"]
    assert outcome["unexpected_metadata_diffs"] == []
    assert len(record.get("rank_metadata_divergences", [])) == n_ranks - 1


def test_two_rank_reconstruction_matches_legacy_record():
    layouts = _layouts(2)
    collected, _, _ = _collections(layouts)
    courier = legacy.OwnedRegionCourier(layouts, 0, None)
    shipment = courier._decode(courier._encode(collected[1]))
    old = legacy.reconstruct_boundary_record(courier, collected[0], shipment)
    candidate = reconstruct_boundary(layouts, [collected[0], shipment], proof=legacy.proof)
    assert candidate["atmosphere"] == old["atmosphere"]
    assert candidate["backend"] == old["backend"]


def test_default_proof_uses_dynamically_bound_forecast_module(monkeypatch):
    layouts = _layouts(3)
    collected, _, _ = _collections(layouts)
    calls = []

    def bound_fingerprint(value):
        calls.append(value)
        return legacy.proof.fingerprint_nested_arrays(value)

    monkeypatch.setitem(sys.modules, "v841_forecast", SimpleNamespace(proof=SimpleNamespace(fingerprint_nested_arrays=bound_fingerprint)))
    reconstruct_boundary(layouts, collected)
    assert len(calls) == 1


def test_named_padded_carrier_is_excluded_with_every_rank_hash():
    layouts = _layouts(3)
    collected, atmosphere, backend = _collections(layouts, padded=True)
    record = reconstruct_boundary(layouts, collected, proof=legacy.proof)
    path = "seam/arrays/driver/pbl_tendencies/ru"
    assert path not in record["backend"]["arrays"]
    artifact = record["batch_order_artifacts"][path]
    for rank, item in enumerate(collected):
        position = next(i for i, entry in enumerate(item["entries"]) if entry["path"] == f"__backend__/{path}")
        assert artifact[f"rank{rank}_sha256"] == hashlib.sha256(item["blobs"][position]).hexdigest()
        assert artifact[f"rank{rank}_shape"] == [2, 1, layouts[rank].n_owned_cells + 1]
    backend = dict(backend)
    backend[path] = np.zeros((2, 1, 13), dtype=np.float32)
    reference = {"atmosphere": fingerprint_atmosphere(atmosphere), "backend": legacy.proof.fingerprint_nested_arrays(backend)}
    assert compare_boundary_records(record, reference)["state_invariant"]


@pytest.mark.parametrize(
    "mutation, reason",
    [
        ("time", "disagree about model time"),
        ("signed_zero_time", "disagree about model time"),
        ("paths", "entry paths diverge"),
        ("duplicate", "duplicate boundary path"),
        ("blob_count", "entry and blob counts differ"),
        ("blob_size", "header requires"),
        ("kind", "entity kind/axis/pad schema differs"),
        ("axis", "entity kind/axis/pad schema differs"),
        ("dtype", "dtype differs"),
        ("leading_shape", "non-entity shape differs"),
        ("owned_extent", "not an expected extent"),
        ("constant_bytes", "constants differ bytewise"),
        ("metadata_schema", "scalar metadata paths diverge"),
        ("metadata_value", "scalar metadata differs"),
    ],
)
def test_boundary_structure_and_clock_fail_closed(mutation, reason):
    layouts = _layouts(3)
    collected, _, _ = _collections(layouts)
    item = collected[1]
    if mutation == "time":
        item["time_seconds"] = 121.0
    elif mutation == "signed_zero_time":
        for payload in collected:
            payload["time_seconds"] = 0.0
        item["time_seconds"] = -0.0
    elif mutation == "paths":
        item["entries"][-1]["path"] += "/renamed"
    elif mutation == "duplicate":
        item["entries"].append(deepcopy(item["entries"][0]))
        item["blobs"].append(item["blobs"][0])
    elif mutation == "blob_count":
        item["blobs"].pop()
    elif mutation == "blob_size":
        item["blobs"][0] = item["blobs"][0][:-1]
    elif mutation in ("kind", "axis", "dtype", "leading_shape", "owned_extent"):
        entry = item["entries"][0]
        if mutation == "kind":
            entry["kind"] = "edge"
        elif mutation == "axis":
            entry["axis"] = 0
        elif mutation == "dtype":
            entry["dtype"] = "<i4"
        else:
            entry["shape"][0 if mutation == "leading_shape" else 1] += 1
            item["blobs"][0] = np.zeros(entry["shape"], dtype=np.float32).tobytes()
    elif mutation == "constant_bytes":
        position = next(i for i, entry in enumerate(item["entries"]) if entry["path"] == "__backend__/constant")
        item["blobs"][position] = bytes(len(item["blobs"][position]))
    elif mutation == "metadata_schema":
        item["backend_scalars"]["extra"] = "value"
    elif mutation == "metadata_value":
        item["backend_scalars"]["schema"] = "changed"
    with pytest.raises(ExecutorError, match=reason):
        reconstruct_boundary(layouts, collected, proof=legacy.proof)


def test_unknown_padded_array_cannot_be_silently_excluded():
    layouts = _layouts(3)
    collected, _, _ = _collections(layouts, padded=True)
    for item in collected:
        item["entries"][-2]["path"] = "__backend__/unexpected_padded_state"
    with pytest.raises(ExecutorError, match="not a named staggered-u carrier"):
        reconstruct_boundary(layouts, collected, proof=legacy.proof)


def test_boundary_ownership_cover_and_collection_count_are_required():
    layouts = _layouts(3)
    collected, _, _ = _collections(layouts)
    with pytest.raises(ExecutorError, match="collection count differs"):
        reconstruct_boundary(layouts, collected[:-1], proof=legacy.proof)
    broken = list(layouts)
    broken[0] = replace(broken[0], n_owned_vertices=0)
    with pytest.raises(ExecutorError, match="vertices: global index .* has no owner"):
        reconstruct_boundary(broken, collected, proof=legacy.proof)


@pytest.mark.parametrize("owned_only", [False, True])
def test_history_reconstruction_preserves_all_bits_and_ignores_halos(owned_only):
    layouts = _layouts(3)
    # Use raw float bit patterns, including signed zero and a NaN payload.
    truth = np.array([0, 0x80000000, 0x7FC00001, 0x3F800000] * 3, dtype=np.uint32).view(np.float32)
    mappings = []
    for layout in layouts:
        local = truth[layout.cell_l2g].copy()
        if owned_only:
            local = local[:layout.n_owned_cells]
        else:
            local[layout.n_owned_cells:] = 42.0
        mappings.append({"field": local, "constant": np.array(5, dtype=np.int32)})
    result = assemble_owned_arrays(layouts, mappings, kinds={"field": "cell", "constant": "none"})
    assert result["field"].tobytes() == truth.tobytes()
    assert result["constant"].shape == ()
    assert result["constant"].tobytes() == mappings[0]["constant"].tobytes()


@pytest.mark.parametrize("owned_only", [False, True])
def test_consuming_history_reconstruction_releases_inputs_and_preserves_bits(owned_only):
    layouts = _layouts(3)
    truth = np.array([0, 0x80000000, 0x7FC00001, 0x3F800000] * 3, dtype=np.uint32).view(np.float32)
    mappings = []
    for layout in layouts:
        local = truth[layout.cell_l2g].copy()
        if owned_only:
            local = local[:layout.n_owned_cells]
        mappings.append({"field": local, "constant": np.array(5, dtype=np.int32)})
    result = assemble_owned_arrays(layouts, mappings, kinds={"field":"cell", "constant":"none"}, consume=True)
    assert result["field"].tobytes() == truth.tobytes()
    assert result["constant"].tobytes() == np.array(5,dtype=np.int32).tobytes()
    assert all(not mapping for mapping in mappings)


def test_history_infers_cell_and_edge_trailing_axes():
    layouts = _layouts(3)
    cells = np.arange(24, dtype=np.float32).reshape(2, 12)
    edges = np.arange(48, dtype=np.float32).reshape(2, 24)
    mappings = [{"cell": cells[..., layout.cell_l2g], "edge": edges[..., layout.edge_l2g]} for layout in layouts]
    result = assemble_owned_arrays(layouts, mappings)
    assert result["cell"].tobytes() == cells.tobytes()
    assert result["edge"].tobytes() == edges.tobytes()


@pytest.mark.parametrize("mutation, reason", [("dtype", "dtype differs"), ("shape", "non-entity shape differs"), ("constant", "constants differ bytewise"), ("paths", "field paths diverge")])
def test_history_schema_and_constants_are_required(mutation, reason):
    layouts = _layouts(3)
    mappings = [{"field": np.zeros((2, layout.n_local_cells), dtype=np.float32), "constant": np.array([5, 6], dtype=np.int32)} for layout in layouts]
    if mutation == "dtype":
        mappings[1]["field"] = mappings[1]["field"].astype(np.int32)
    elif mutation == "shape":
        mappings[1]["field"] = mappings[1]["field"][:1]
    elif mutation == "constant":
        mappings[1]["constant"][0] = 7
    elif mutation == "paths":
        mappings[1].pop("constant")
    with pytest.raises(ExecutorError, match=reason):
        assemble_owned_arrays(layouts, mappings, kinds={"field": "cell", "constant": "none"})


def _regional_collections(layouts, *, staggered=False):
    atmosphere, backend = _truth()
    for name, owner, kind in legacy._ATMO_FIELDS:
        target = atmosphere.state if owner == "state" else atmosphere.saved
        value = getattr(target, name)
        tail = np.zeros(value.shape[:-1] + (1,), dtype=np.float32)
        tail.view(np.uint32)[...] = 0x80000000
        setattr(target, name, np.concatenate((value, tail), axis=-1))
    backend["soil"] = np.concatenate((backend["soil"], np.zeros((1, 3), dtype=np.float32)), axis=0)
    backend["edge"] = np.concatenate((backend["edge"], np.zeros((2, 1), dtype=np.float32)), axis=-1)
    backend["identity"]["n_columns"] = 13
    collected = []
    for rank, layout in enumerate(layouts):
        local_state = SimpleNamespace(time_seconds=120.)
        local_saved = SimpleNamespace()
        for name, owner, kind in legacy._ATMO_FIELDS:
            source = getattr(atmosphere.state if owner == "state" else atmosphere.saved, name)
            local = np.concatenate((source[..., getattr(layout, f"{kind}_l2g")], source[..., -1:]), axis=-1)
            owned = getattr(layout, f"n_owned_{kind}s")
            local[..., owned:-1] = -42.
            setattr(local_state if owner == "state" else local_saved, name, local)
        payload = {
            "soil": np.concatenate((backend["soil"][layout.cell_l2g], backend["soil"][-1:]), axis=0),
            "edge": np.concatenate((backend["edge"][..., layout.edge_l2g], backend["edge"][..., -1:]), axis=-1),
            "constant": backend["constant"],
            "identity": {"constructor_identity_sha256": f"rank{rank}", "n_columns": layout.n_local_cells + 1},
            "schema": backend["schema"], "step": backend["step"],
        }
        if staggered:
            payload["seam"] = {"arrays": {
                f"driver/{name}/ru": np.full((2, 1, layout.n_local_cells + 2), rank + 1., dtype=np.float32)
                for name in ("pbl_tendencies", "radiation_tendencies", "cumulus_tendencies")
            }}
        stack = {
            "driver": SimpleNamespace(atmosphere=SimpleNamespace(state=local_state, saved=local_saved)),
            "backend": SimpleNamespace(restart_state=lambda payload=payload: payload),
        }
        collected.append(collect_owned(layouts, rank, stack))
    return collected, atmosphere, backend


@pytest.mark.parametrize("n_ranks", [1, 2, 3, 4])
def test_native_garbage_is_in_full_atmosphere_and_backend_hashes(n_ranks):
    layouts = _layouts(n_ranks)
    collected, atmosphere, backend = _regional_collections(layouts)
    record = reconstruct_boundary(layouts, collected, proof=legacy.proof)
    reference = {"atmosphere": fingerprint_atmosphere(atmosphere),
                 "backend": legacy.proof.fingerprint_nested_arrays(backend)}
    assert compare_boundary_records(record, reference)["state_invariant"]
    assert record["atmosphere"] == reference["atmosphere"]
    assert "batch_order_artifacts" not in record
    for path in ("soil", "edge"):
        assert record["backend"]["arrays"][path] == reference["backend"]["arrays"][path]
    assert all(entry["native_padding"] for entry in collected[0]["entries"]
               if entry["path"].startswith("__atmosphere__/"))


@pytest.mark.parametrize("path", ["__atmosphere__/state/rho", "__backend__/soil", "__backend__/edge"])
def test_native_garbage_divergence_is_never_silently_dropped(path):
    layouts = _layouts(3)
    collected, _, _ = _regional_collections(layouts)
    position = next(i for i, entry in enumerate(collected[1]["entries"]) if entry["path"] == path)
    entry = collected[1]["entries"][position]
    array = np.frombuffer(collected[1]["blobs"][position], dtype=np.dtype(entry["dtype"])).reshape(entry["shape"]).copy()
    slices = [slice(None)] * array.ndim
    slices[entry["axis"]] = slice(-1, None)
    # Changing only the sign of a zero still changes the native byte contract.
    array[tuple(slices)].view(np.uint32)[...] ^= np.uint32(0x80000000)
    collected[1]["blobs"][position] = array.tobytes()
    with pytest.raises(ExecutorError, match="native garbage differs bytewise"):
        reconstruct_boundary(layouts, collected, proof=legacy.proof)


def test_regional_staggered_carriers_have_two_named_artifact_tail_slots():
    layouts = _layouts(3)
    collected, _, _ = _regional_collections(layouts, staggered=True)
    record = reconstruct_boundary(layouts, collected, proof=legacy.proof)
    assert len(record["batch_order_artifacts"]) == 3
    for path, artifact in record["batch_order_artifacts"].items():
        assert path not in record["backend"]["arrays"]
        for rank, layout in enumerate(layouts):
            assert artifact[f"rank{rank}_shape"][-1] == layout.n_owned_cells + 2
