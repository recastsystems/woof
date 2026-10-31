"""The content-keyed host-preparation cache returns exactly what the function returned."""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pytest

from woof.hex import prep_cache, numerics
from woof.hex.damping_v841 import build_v841_vertical_velocity_damping
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mixing_v841 import initialize_deformation_weights_v841
from woof.hex.transport import build_advection_coefficients
from test_mesh_validation import graded_sphere_mesh


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "prep"
    monkeypatch.setenv(prep_cache.ENVIRONMENT, str(root))
    prep_cache._memory.clear()
    prep_cache._ledger.clear()
    yield root
    prep_cache._memory.clear()
    prep_cache._ledger.clear()


def _bits(array):
    array = np.asarray(array)
    return array.dtype.str, array.shape, array.tobytes(order="A"), array.flags.f_contiguous


def _events(label):
    return [row["event"] for row in prep_cache._ledger if row.get("label") == label]


def _zgrid(dtype=np.float32, cells=37, levels=12):
    rng = np.random.default_rng(7)
    base = rng.uniform(0.0, 900.0, cells).astype(dtype)
    steps = rng.uniform(200.0, 2000.0, (levels, cells)).astype(dtype)
    return np.concatenate([base[None, :], base[None, :] + np.cumsum(steps, axis=0, dtype=dtype)]).astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_damping_hit_is_bitwise_the_computation(cache_root, dtype):
    zgrid = _zgrid(dtype)
    kwargs = dict(xnutr=0.2, damping_start_z=float(zgrid[-1].min() * 0.4))
    reference = build_v841_vertical_velocity_damping.__wrapped_prep_cache__(zgrid, **kwargs)
    first = build_v841_vertical_velocity_damping(zgrid, **kwargs)
    second = build_v841_vertical_velocity_damping(zgrid, **kwargs)
    prep_cache._memory.clear()
    third = build_v841_vertical_velocity_damping(zgrid, **kwargs)
    assert _bits(first) == _bits(reference) == _bits(second) == _bits(third)
    assert _events("v841-vertical-velocity-damping") == ["miss", "hit-memory", "hit-disk"]


def test_returned_arrays_are_private_copies(cache_root):
    zgrid = _zgrid()
    kwargs = dict(xnutr=0.2, damping_start_z=float(zgrid[-1].min() * 0.4))
    first = build_v841_vertical_velocity_damping(zgrid, **kwargs)
    expected = first.copy()
    first[...] = -1.0
    second = build_v841_vertical_velocity_damping(zgrid, **kwargs)
    second[...] = -2.0
    third = build_v841_vertical_velocity_damping(zgrid, **kwargs)
    assert _bits(third) == _bits(expected)


def test_any_input_byte_changes_the_key(cache_root):
    zgrid = _zgrid()
    kwargs = dict(xnutr=0.2, damping_start_z=float(zgrid[-1].min() * 0.4))
    build_v841_vertical_velocity_damping(zgrid, **kwargs)
    moved = zgrid.copy()
    moved.view(np.uint32)[5, 3] ^= 1  # one ULP in one interface height
    result = build_v841_vertical_velocity_damping(moved, **kwargs)
    reference = build_v841_vertical_velocity_damping.__wrapped_prep_cache__(moved, **kwargs)
    assert _bits(result) == _bits(reference)
    build_v841_vertical_velocity_damping(zgrid, xnutr=0.25, damping_start_z=kwargs["damping_start_z"])
    assert _events("v841-vertical-velocity-damping") == ["miss", "miss", "miss"]


def test_refusal_is_never_stored(cache_root):
    zgrid = _zgrid()
    with pytest.raises(ConfigurationRefusal):
        build_v841_vertical_velocity_damping(zgrid, xnutr=2.0, damping_start_z=1000.0)
    with pytest.raises(ConfigurationRefusal):
        build_v841_vertical_velocity_damping(zgrid, xnutr=2.0, damping_start_z=1000.0)
    assert not list(cache_root.rglob("manifest.json"))


def test_damaged_entry_is_discarded_and_recomputed(cache_root):
    zgrid = _zgrid()
    kwargs = dict(xnutr=0.2, damping_start_z=float(zgrid[-1].min() * 0.4))
    reference = build_v841_vertical_velocity_damping(zgrid, **kwargs)
    (stored,) = [p for p in cache_root.rglob("0.npy")]
    raw = bytearray(stored.read_bytes())
    raw[-1] ^= 0xFF
    stored.write_bytes(bytes(raw))
    prep_cache._memory.clear()
    again = build_v841_vertical_velocity_damping(zgrid, **kwargs)
    assert _bits(again) == _bits(reference)
    assert _events("v841-vertical-velocity-damping") == ["miss", "miss"]
    assert any(row.get("event") == "discarded" for row in prep_cache._ledger)


def test_off_switch_runs_the_function(tmp_path, monkeypatch):
    monkeypatch.setenv(prep_cache.ENVIRONMENT, "off")
    prep_cache._ledger.clear()
    zgrid = _zgrid()
    build_v841_vertical_velocity_damping(zgrid, xnutr=0.2, damping_start_z=float(zgrid[-1].min() * 0.4))
    assert prep_cache._ledger == []
    assert prep_cache.receipt()["enabled"] is False


def test_mesh_dataclass_results_round_trip(cache_root):
    mesh = graded_sphere_mesh(level=1)
    reference = initialize_deformation_weights_v841.__wrapped_prep_cache__(mesh)
    first = initialize_deformation_weights_v841(mesh)
    prep_cache._memory.clear()
    second = initialize_deformation_weights_v841(mesh)
    for name in ("coef_c2", "coef_s2", "coef_cs"):
        assert _bits(getattr(first, name)) == _bits(getattr(reference, name))
        assert _bits(getattr(second, name)) == _bits(getattr(reference, name))
    assert type(second) is type(reference)
    n_edges = int(np.asarray(mesh.cellsOnEdge).shape[0])
    stencil = 2 * int(np.asarray(mesh.cellsOnCell).shape[1]) + 3
    deriv = np.random.default_rng(11).normal(size=(n_edges, 2, stencil)).astype(np.float32)
    coefficients = build_advection_coefficients(mesh, deriv)
    prep_cache._memory.clear()
    again = build_advection_coefficients(mesh, deriv)
    direct = build_advection_coefficients.__wrapped_prep_cache__(mesh, deriv)
    for name in ("adv_coefs", "adv_coefs_3rd", "n_adv_cells_for_edge", "adv_cells_for_edge"):
        assert _bits(getattr(again, name)) == _bits(getattr(direct, name)) == _bits(getattr(coefficients, name))
    assert again.horizontal_order == direct.horizontal_order


def test_mesh_state_change_misses(cache_root):
    mesh = graded_sphere_mesh(level=1)
    initialize_deformation_weights_v841(mesh)
    mesh.arrays["xCell"] = np.array(mesh.arrays["xCell"], copy=True)
    mesh.arrays["xCell"].view(np.uint64 if mesh.arrays["xCell"].dtype == np.float64 else np.uint32)[0] ^= 1
    initialize_deformation_weights_v841(mesh)
    assert _events("v841-deformation-weights") == ["miss", "miss"]


def test_foreign_argument_is_uncacheable(cache_root):
    @dataclass
    class Foreign:
        value: int

    calls = []

    @prep_cache.cached("unit-foreign")
    def probe(thing):
        calls.append(thing)
        return np.arange(3)

    probe(Foreign(1))
    probe(Foreign(1))
    assert len(calls) == 2
    assert _events("unit-foreign") == ["uncacheable", "uncacheable"]


def test_fortran_order_and_read_only_flags_survive(cache_root):
    @prep_cache.cached("unit-order")
    def probe(n):
        value = np.asfortranarray(np.arange(n * 3, dtype=np.float32).reshape(n, 3))
        frozen = np.arange(4, dtype=np.int64)
        frozen.flags.writeable = False
        return value, frozen, np.float32(1.5), {"n": n}

    first = probe(4)
    prep_cache._memory.clear()
    second = probe(4)
    assert second[0].flags.f_contiguous and not second[0].flags.c_contiguous
    assert _bits(second[0]) == _bits(first[0])
    assert second[1].flags.writeable is False
    assert second[2] == np.float32(1.5) and type(second[2]) is np.float32
    assert second[3] == {"n": 4}
    assert _events("unit-order") == ["miss", "hit-disk"]


def test_receipt_records_each_call(cache_root):
    zgrid = _zgrid()
    kwargs = dict(xnutr=0.2, damping_start_z=float(zgrid[-1].min() * 0.4))
    build_v841_vertical_velocity_damping(zgrid, **kwargs)
    build_v841_vertical_velocity_damping(zgrid, **kwargs)
    receipt = prep_cache.receipt()
    assert receipt["enabled"] and receipt["root"] == str(cache_root)
    assert [row["event"] for row in receipt["calls"]] == ["miss", "hit-memory"]
    json.dumps(receipt)


def test_salt_separates_bridge_builds(cache_root):
    build = ["a" * 64]
    calls = []

    @prep_cache.cached("unit-salt", salt=lambda: build[0])
    def probe(n):
        calls.append(n)
        return np.arange(n)

    probe(3)
    probe(3)
    build[0] = "b" * 64
    probe(3)
    assert calls == [3, 3]
    assert _events("unit-salt") == ["miss", "hit-memory", "miss"]


def test_unresolvable_salt_runs_the_function(cache_root):
    def missing():
        raise FileNotFoundError("no bridge")

    @prep_cache.cached("unit-missing-salt", salt=missing)
    def probe(n):
        return np.arange(n)

    assert probe(2).tolist() == [0, 1]
    assert _events("unit-missing-salt") == ["uncacheable"]


# ---------------------------------------------------------------------------
# hooks: the binding of every name a cached function reaches is in the key
# ---------------------------------------------------------------------------
from woof.hex import native_geometry_math  # noqa: E402
from woof.hex.mesh import Mesh, _looks_like_connectivity  # noqa: E402


def _damping_case():
    zgrid = _zgrid()
    return zgrid, dict(xnutr=0.2, damping_start_z=float(zgrid[-1].min() * 0.4))


# The float32 damping builder evaluates the selected geometry authority through
# ``woof.hex.native_geometry_math.sin`` (dycore-match); the float64 path still
# reaches the Rust table through a function-local import, which the fresh-process
# keying test below covers.
def _zero_sin(phase):
    return np.zeros_like(np.asarray(phase))


def test_rebinding_a_hook_outside_hexcore_bypasses_the_cache(cache_root, monkeypatch):
    # This hook contract applies to the fast host-libm path.
    monkeypatch.setattr(numerics, "_mode", "fast")
    zgrid, kwargs = _damping_case()
    default = build_v841_vertical_velocity_damping(zgrid, **kwargs)
    build_v841_vertical_velocity_damping(zgrid, **kwargs)
    monkeypatch.setattr(native_geometry_math, "sin", _zero_sin)
    rebound = build_v841_vertical_velocity_damping(zgrid, **kwargs)
    direct = build_v841_vertical_velocity_damping.__wrapped_prep_cache__(zgrid, **kwargs)
    assert _bits(rebound) == _bits(direct)
    assert _bits(rebound) != _bits(default)
    assert _events("v841-vertical-velocity-damping") == ["miss", "hit-memory", "uncacheable"]
    (row,) = [r for r in prep_cache._ledger if r.get("event") == "uncacheable"]
    assert "woof.hex.native_geometry_math.sin" in row["reason"]


def test_rebinding_a_hook_to_another_trusted_function_misses(cache_root, monkeypatch):
    # This hook contract applies to the fast host-libm path.
    monkeypatch.setattr(numerics, "_mode", "fast")
    zgrid, kwargs = _damping_case()
    build_v841_vertical_velocity_damping(zgrid, **kwargs)
    original = native_geometry_math.sin
    monkeypatch.setattr(native_geometry_math, "sin", np.sin)
    rebound = build_v841_vertical_velocity_damping(zgrid, **kwargs)
    direct = build_v841_vertical_velocity_damping.__wrapped_prep_cache__(zgrid, **kwargs)
    assert _bits(rebound) == _bits(direct)
    # Restore the hook only (monkeypatch.undo would also drop the cache root).
    monkeypatch.setattr(native_geometry_math, "sin", original)
    build_v841_vertical_velocity_damping(zgrid, **kwargs)
    assert _events("v841-vertical-velocity-damping") == ["miss", "miss", "hit-memory"]


def test_a_function_local_import_is_keyed_whether_or_not_it_ran_before():
    # The key of one computation must not depend on what else the process imported first.
    import subprocess
    import sys

    probe = (
        "import sys\n"
        "from woof.hex import prep_cache, damping_v841\n"
        "fresh = 'woof.hex.rust_hostprep' not in sys.modules\n"
        "tokens, _ = prep_cache.binding_tokens(damping_v841.build_v841_vertical_velocity_damping.__wrapped_prep_cache__)\n"
        "print(fresh, len(tokens), any(t.startswith('woof.hex.rust_hostprep.vertical_velocity_damping=') for t in tokens))\n"
    )
    import os
    from pathlib import Path

    # conftest's sys.path insertion is local to the parent. A source/sdist
    # probe must key the same package, even beside an older installed wheel.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(native_geometry_math.__file__).resolve().parents[1]) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            check=True, env=env)
    fresh, count, reached = result.stdout.split()
    assert fresh == "True", "the probe must start without woof.hex.rust_hostprep imported"
    tokens, _ = prep_cache.binding_tokens(build_v841_vertical_velocity_damping.__wrapped_prep_cache__)
    assert reached == "True" and int(count) == len(tokens)


def test_a_transparent_wrapper_is_keyed_as_what_it_wraps(cache_root, monkeypatch):
    # This hook contract applies to the fast host-libm path.
    monkeypatch.setattr(numerics, "_mode", "fast")
    import functools

    zgrid, kwargs = _damping_case()
    build_v841_vertical_velocity_damping(zgrid, **kwargs)
    original = native_geometry_math.sin

    @functools.wraps(original)
    def timed(*args, **kw):
        return original(*args, **kw)

    timed.__prep_cache_transparent__ = True
    monkeypatch.setattr(native_geometry_math, "sin", timed)
    build_v841_vertical_velocity_damping(zgrid, **kwargs)
    assert _events("v841-vertical-velocity-damping") == ["miss", "hit-memory"]


def test_every_shipped_cached_function_walks_cleanly():
    from woof.hex import damping_v841, driver, mixing_v841, transport, vertical_spec

    functions = [
        damping_v841.build_v841_vertical_velocity_damping,
        driver._frozen_vertical_damping,
        mixing_v841.initialize_deformation_weights_v841,
        transport.build_advection_coefficients,
        vertical_spec.build_vertical_from_spec,
    ]
    for function in functions:
        tokens, _ = prep_cache.binding_tokens(function.__wrapped_prep_cache__, [Mesh])
        assert tokens, function
    tokens, _ = prep_cache.binding_tokens(
        damping_v841.build_v841_vertical_velocity_damping.__wrapped_prep_cache__
    )
    assert "woof.hex.native_geometry_math.sin=woof.hex.native_geometry_math:_intrinsic.<locals>.evaluate" in tokens


# ---------------------------------------------------------------------------
# paths: the same mesh bytes anywhere are the same key
# ---------------------------------------------------------------------------
def _write_mesh(path, mesh):
    from netCDF4 import Dataset

    with Dataset(path, "w", format="NETCDF4") as dataset:
        for name, value in sorted(mesh.attrs.items()):
            dataset.setncattr(name, value)
        sizes = {}
        for name, array in sorted(mesh.arrays.items()):
            array = np.asarray(array)
            dims = []
            for size in array.shape:
                dim = sizes.setdefault(size, f"n{size}")
                if dim not in dataset.dimensions:
                    dataset.createDimension(dim, size)
                dims.append(dim)
            if _looks_like_connectivity(name, array):
                array = np.where(array < 0, 0, array + 1).astype(array.dtype)
            dataset.createVariable(name, array.dtype, tuple(dims))[...] = array


@pytest.fixture
def moved_mesh(tmp_path):
    import os
    import shutil

    first = tmp_path / "case-a"
    second = tmp_path / "elsewhere" / "case-b"
    first.mkdir()
    second.mkdir(parents=True)
    _write_mesh(first / "grid.nc", graded_sphere_mesh(level=1))
    try:
        os.link(first / "grid.nc", second / "grid.nc")
    except OSError:
        shutil.copyfile(first / "grid.nc", second / "grid.nc")
    return (Mesh.from_netcdf(first / "grid.nc", validate=False),
            Mesh.from_netcdf(second / "grid.nc", validate=False))


def test_the_same_mesh_at_another_path_hits(cache_root, moved_mesh):
    here, there = moved_mesh
    assert here.grid_path != there.grid_path
    reference = initialize_deformation_weights_v841(here)
    prep_cache._memory.clear()
    moved = initialize_deformation_weights_v841(there)
    for name in ("coef_c2", "coef_s2", "coef_cs"):
        assert _bits(getattr(moved, name)) == _bits(getattr(reference, name))
    assert _events("v841-deformation-weights") == ["miss", "hit-disk"]
    assert len(list(cache_root.rglob("manifest.json"))) == 1


def test_a_function_that_reads_the_path_is_keyed_on_it(cache_root, moved_mesh):
    here, there = moved_mesh

    @prep_cache.cached("unit-reads-path")
    def probe(mesh):
        return np.frombuffer(mesh.grid_path.encode("utf-8"), dtype=np.uint8).copy()

    assert bytes(probe(here)) == here.grid_path.encode("utf-8")
    assert bytes(probe(there)) == there.grid_path.encode("utf-8")
    assert _events("unit-reads-path") == ["miss", "miss"]


# ---------------------------------------------------------------------------
# CPU dispatch, memory, size
# ---------------------------------------------------------------------------
def test_numpy_cpu_dispatch_is_in_the_producer_digest(monkeypatch):
    prep_cache.producer_digest.cache_clear()
    here = prep_cache.producer_digest()
    assert "enabled" in prep_cache.numpy_cpu_dispatch()
    monkeypatch.setattr(prep_cache, "numpy_cpu_dispatch", lambda: '{"baseline": [], "enabled": []}')
    prep_cache.producer_digest.cache_clear()
    try:
        assert prep_cache.producer_digest() != here
    finally:
        monkeypatch.undo()
        prep_cache.producer_digest.cache_clear()


def test_a_stored_result_is_not_held_twice(cache_root):
    zgrid, kwargs = _damping_case()
    build_v841_vertical_velocity_damping(zgrid, **kwargs)
    (held,) = prep_cache._memory.values()
    assert "arrays" not in held and held["disk"] is True


def test_the_disk_layer_keeps_to_its_cap(cache_root, monkeypatch):
    @prep_cache.cached("unit-cap")
    def probe(n):
        return np.full(64 * 1024, n, dtype=np.float64)  # 512 KiB each

    monkeypatch.setenv(prep_cache.CAP_ENVIRONMENT, str(1.2 * 1024 * 1024 / (1 << 30)))
    for n in range(5):
        prep_cache._memory.clear()  # as five separate runs
        probe(n)
    kept = [path.parent.name for path in cache_root.rglob("manifest.json")]
    assert len(kept) == 2
    assert any(row.get("event") == "evicted" for row in prep_cache._ledger)
    prep_cache._memory.clear()
    assert probe(4)[0] == 4.0
    assert _events("unit-cap")[-1] == "hit-disk"


def test_eviction_never_removes_an_entry_this_process_holds(cache_root, monkeypatch):
    @prep_cache.cached("unit-held")
    def probe(n):
        return np.full(64 * 1024, n, dtype=np.float64)

    monkeypatch.setenv(prep_cache.CAP_ENVIRONMENT, str(1.0 / (1 << 30)))
    for n in range(3):
        probe(n)
    assert len(list(cache_root.rglob("manifest.json"))) == 3


def test_a_reader_the_cached_function_never_calls_does_not_key_it(cache_root, moved_mesh, monkeypatch):
    # The same-host measurement tool rebinds woof.hex.mesh.Dataset to a guarded
    # proxy.  Mesh.from_netcdf uses it; the deformation weights never call it,
    # so the call stays cacheable (measured 2026-10-05: under the
    # tool, three tables went uncacheable when every Mesh method was walked).
    from woof.hex import mesh as mesh_module

    here, _ = moved_mesh

    def guarded_dataset(*args, **kwargs):
        raise AssertionError("not called by the cached computation")

    initialize_deformation_weights_v841(here)
    monkeypatch.setattr(mesh_module, "Dataset", guarded_dataset, raising=False)
    prep_cache._memory.clear()
    initialize_deformation_weights_v841(here)
    assert _events("v841-deformation-weights") == ["miss", "hit-disk"]


# ---------------------------------------------------------------------------
# a same-host rank's local mesh names the whole mesh it was cut from
# ---------------------------------------------------------------------------
def _whole_mesh(level):
    # An array the key can never encode: a call keyed on this mesh is uncacheable, one keyed without it is not.
    whole = graded_sphere_mesh(level=level)
    whole.arrays["unkeyable"] = np.array([object()], dtype=object)
    return whole


def test_a_rank_mesh_is_keyed_without_the_mesh_it_was_cut_from(cache_root):
    # THE BREAKAGE (RRFS hrrrv5 limited area, mv3km 4 same-host ranks, V8X6000, 2026-10-29): the cached
    # advection coefficients and deformation weights of a rank's local mesh hashed the whole mesh it was cut
    # from (``partition_source_mesh``, a shared copy), so every rank process held all of it resident.
    local = graded_sphere_mesh(level=1)
    local.partition_source_mesh = _whole_mesh(2)
    weights = initialize_deformation_weights_v841(local)
    n_edges = int(np.asarray(local.cellsOnEdge).shape[0])
    stencil = 2 * int(np.asarray(local.cellsOnCell).shape[1]) + 3
    deriv = np.random.default_rng(11).normal(size=(n_edges, 2, stencil)).astype(np.float32)
    build_advection_coefficients(local, deriv)
    assert _events("advection-coefficients") == ["miss"]
    # Another whole mesh behind the same local arrays is the same entry, with the same words.
    local.partition_source_mesh = _whole_mesh(3)
    prep_cache._memory.clear()
    again = initialize_deformation_weights_v841(local)
    assert _events("v841-deformation-weights") == ["miss", "hit-disk"]
    for name in ("coef_c2", "coef_s2", "coef_cs", "coef_c", "coef_s"):
        assert _bits(getattr(again, name)) == _bits(getattr(weights, name))


def test_a_read_provenance_field_keys_that_field_only(cache_root, moved_mesh):
    here, there = moved_mesh
    here.partition_source_mesh = _whole_mesh(2)

    @prep_cache.cached("unit-reads-path-only")
    def probe(mesh):
        return np.frombuffer(mesh.grid_path.encode("utf-8"), dtype=np.uint8).copy()

    assert bytes(probe(here)) == here.grid_path.encode("utf-8")
    assert bytes(probe(there)) == there.grid_path.encode("utf-8")
    assert _events("unit-reads-path-only") == ["miss", "miss"]
