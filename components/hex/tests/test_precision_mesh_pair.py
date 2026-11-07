"""Strict validation remains on every changed carrier in a precision pair."""
import copy
import hashlib
import json
import threading

import numpy as np
import pytest

from woof.hex import mesh as module
from woof.hex.mesh import Mesh, MeshValidationError
from test_mesh_validation import graded_sphere_mesh


@pytest.mark.parametrize('producer_pi', [module.GRID_ROTATE_DEFAULT_REAL_PI,
                                       module.GRID_ROTATE_X4_REAL_PI])
def test_reconcile_both_exact_pi_signatures_without_changing_geometry(producer_pi):
    mesh = graded_sphere_mesh(level=1, zoom=1.0, dtype=np.float64)
    original = {k: v.tobytes() for k, v in mesh.arrays.items() if not k.startswith('lon')}
    for entity in ('Cell', 'Edge', 'Vertex'):
        mesh.arrays['lon'+entity] = module._grid_rotate_convert_xl_longitude(
            mesh.arrays['x'+entity], mesh.arrays['y'+entity], producer_pi=producer_pi)[0]
    evidence = module.reconcile_grid_rotate_longitudes(mesh)
    assert evidence['producer_default_real_pi'] == producer_pi
    assert all(mesh.arrays[k].tobytes() == v for k, v in original.items())
    mesh.validate()


def test_x4_signature_rejects_corruption_and_mixed_producers():
    for corrupt in (False, True):
        mesh = graded_sphere_mesh(level=1, zoom=1.0, dtype=np.float64)
        for entity in ('Cell', 'Edge', 'Vertex'):
            pi = (module.GRID_ROTATE_DEFAULT_REAL_PI if not corrupt and entity == 'Edge'
                  else module.GRID_ROTATE_X4_REAL_PI)
            mesh.arrays['lon'+entity] = module._grid_rotate_convert_xl_longitude(
                mesh.arrays['x'+entity], mesh.arrays['y'+entity], producer_pi=pi)[0]
        if corrupt:
            mesh.arrays['lonCell'][0] += 1e-6
        before = {k: mesh.arrays[k].copy() for k in ('lonCell','lonEdge','lonVertex')}
        with pytest.raises(MeshValidationError, match='mixed producer|pinned producer formula'):
            module.reconcile_grid_rotate_longitudes(mesh)
        assert all(np.array_equal(mesh.arrays[k], v) for k, v in before.items())


@pytest.mark.parametrize("layout", ["vector", "reversed", "fortran", "scalar", "empty"])
def test_blocked_ulp_check_preserves_special_values_and_shapes(monkeypatch, layout):
    from woof.hex import host_parallel
    monkeypatch.setattr(host_parallel, "MIN_PARALLEL_ELEMENTS", 0)
    monkeypatch.setattr(host_parallel, "MIN_BLOCK_COLUMNS", 2)
    authority = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 1e-310])
    candidate = np.nextafter(authority, np.inf)
    if layout == "reversed":
        candidate, authority = candidate[::-1], authority[::-1]
    elif layout == "fortran":
        candidate, authority = (np.asfortranarray(a.reshape(2, 4)) for a in (candidate, authority))
    elif layout == "scalar":
        candidate, authority = np.array(1.0), np.array(1.0)
    elif layout == "empty":
        candidate, authority = np.empty((0, 3)), np.empty((0, 3))
    before = [a.tobytes() for a in (candidate, authority)]
    with np.errstate(all="ignore"):
        monkeypatch.setenv("WOOF_HEX_HOST_COPY_THREADS", "1")
        serial = module._maximum_ulp_error(candidate, authority)
        monkeypatch.setenv("WOOF_HEX_HOST_COPY_THREADS", "4")
        parallel = module._maximum_ulp_error(candidate, authority)
    assert np.array(serial).tobytes() == np.array(parallel).tobytes()
    assert [a.tobytes() for a in (candidate, authority)] == before
    assert module._maximum_ulp_error(np.empty(2), np.empty(3)) == float("inf")


@pytest.mark.parametrize("reversed_view", [False, True])
def test_blocked_longitude_branches_and_cartesian_rows_are_exact(monkeypatch, reversed_view):
    from woof.hex import host_parallel
    monkeypatch.setattr(host_parallel, "MIN_PARALLEL_ELEMENTS", 0)
    monkeypatch.setattr(host_parallel, "MIN_BLOCK_COLUMNS", 2)
    x = np.array([1, -1, -1, 1, 0, 0, 1, -1, 0, -0., 1e-11, -1e-11], dtype="f8")
    y = np.array([1, 1, -1, -1, 1, -1, 0, -0., 0, -0., 1, -1], dtype="f8")
    if reversed_view:
        x, y = x[::-1], y[::-1]
    before = x.tobytes(), y.tobytes()
    monkeypatch.setenv("WOOF_HEX_HOST_COPY_THREADS", "1")
    serial = module._grid_rotate_convert_xl_longitude(x, y)
    latitude = np.linspace(-np.pi / 2, np.pi / 2, x.size)
    expected = 6371229.0 * np.stack(
        (np.cos(latitude) * np.cos(serial[0]), np.cos(latitude) * np.sin(serial[0]), np.sin(latitude)),
        axis=1,
    )
    monkeypatch.setenv("WOOF_HEX_HOST_COPY_THREADS", "4")
    parallel = module._grid_rotate_convert_xl_longitude(x, y)
    assert [a.tobytes() for a in serial] == [a.tobytes() for a in parallel]
    xyz = module._latitude_longitude_xyz(latitude, parallel[0], 6371229.0)
    assert xyz.tobytes() == expected.tobytes()
    assert xyz.flags.c_contiguous and xyz.flags.owndata
    assert (x.tobytes(), y.tobytes()) == before


def test_blocked_geometry_uses_workers_and_inherits_error_policy(monkeypatch):
    from woof.hex import host_parallel
    monkeypatch.setenv("WOOF_HEX_HOST_COPY_THREADS", "4")
    monkeypatch.setattr(host_parallel, "MIN_PARALLEL_ELEMENTS", 0)
    monkeypatch.setattr(host_parallel, "MIN_BLOCK_COLUMNS", 2)
    seen = []
    original = np.spacing

    def spacing(value):
        seen.append((threading.current_thread().name, np.geterr()))
        return original(value)

    monkeypatch.setattr(np, "spacing", spacing)
    with np.errstate(all="raise"):
        assert module._maximum_ulp_error(np.ones(16), np.ones(16)) == 0.0
    assert seen and all(name.startswith("hex-host-copy") for name, _ in seen)
    assert all(policy == dict(divide="raise", over="raise", under="raise", invalid="raise") for _, policy in seen)


@pytest.mark.parametrize("value", [
    np.array([0, 0x8000000000000000, 0x7ff8000000000007], dtype="u8").view("f8"),
    np.arange(15, dtype=">f8")[::-2],
    np.asfortranarray(np.arange(12, dtype="i4").reshape(3, 4)),
    np.zeros(3, dtype=[("count", "i4"), ("value", ">f8")]),
    np.arange(3).astype("datetime64[ns]"),
    np.array(2.0), np.empty((0, 3)), np.empty(3, dtype="V0"),
    np.array(["kept", None], dtype=object),
])
def test_mesh_array_digest_keeps_dtype_shape_and_complete_c_bytes(value):
    array = np.ascontiguousarray(np.asarray(value))
    expected = hashlib.sha256()
    expected.update(str(array.dtype).encode("ascii"))
    expected.update(b"\0")
    expected.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    expected.update(b"\0")
    expected.update(array.tobytes(order="C"))
    before = value.tobytes()
    assert module._sha256_array(value) == expected.hexdigest()
    assert value.tobytes() == before


@pytest.mark.parametrize("workers,large", [(1, True), (4, False), (4, True)])
def test_mesh_array_digest_group_keeps_order_and_worker_bounds(monkeypatch, workers, large):
    monkeypatch.setenv("WOOF_HEX_HOST_COPY_THREADS", str(workers))
    monkeypatch.setattr(module, "_PARALLEL_ARRAY_DIGEST_BYTES", 0 if large else 1000)
    original = module._sha256_array
    seen = []
    pair_started = threading.Barrier(2)

    def fingerprint(value):
        seen.append(threading.current_thread().name)
        if workers > 1 and large:
            pair_started.wait(timeout=5)
        return original(value)

    values = {name: np.array([i]) for i, name in enumerate(("second", "first"))}
    expected = [(name, original(value)) for name, value in values.items()]
    monkeypatch.setattr(module, "_sha256_array", fingerprint)
    with module._array_digest_items(values) as results:
        assert list(results) == expected
    if workers > 1 and large:
        assert all(name.startswith("hex-mesh-array-digest") for name in seen)
    else:
        assert seen == [threading.current_thread().name] * 2


def test_mesh_digest_comparison_precedes_later_hash_error_and_joins(monkeypatch):
    monkeypatch.setenv("WOOF_HEX_HOST_COPY_THREADS", "2")
    monkeypatch.setattr(module, "_PARALLEL_ARRAY_DIGEST_BYTES", 0)
    started = threading.Event()
    finished = []

    def fingerprint(value):
        try:
            if value[0] == 0:
                assert started.wait(timeout=5)
                return "mismatch"
            started.set()
            raise RuntimeError("later array failed")
        finally:
            finished.append(int(value[0]))

    monkeypatch.setattr(module, "_sha256_array", fingerprint)
    with pytest.raises(ValueError, match="first comparison failed"):
        with module._array_digest_items({"first": np.array([0]), "later": np.array([1])}) as results:
            for name, actual in results:
                if actual != "expected":
                    raise ValueError("first comparison failed")
    assert sorted(finished) == [0, 1]


@pytest.mark.parametrize("field,category", [("xCell", "Cartesian"), ("latCell", "latitude")])
def test_reconciliation_still_refuses_changed_carriers_and_restores_longitudes(monkeypatch, field, category):
    mesh = graded_sphere_mesh(level=1, zoom=1.0, dtype=np.float64)
    originals = {name: mesh.arrays[name] for name in ("lonCell", "lonEdge", "lonVertex")}
    sources = dict(mesh.variable_sources)
    monkeypatch.setenv("WOOF_HEX_HOST_COPY_THREADS", "4")
    monkeypatch.setattr(module, "_PARALLEL_ARRAY_DIGEST_BYTES", 0)

    def changed(self):
        self.arrays[field][0] += 1.0
        return self

    monkeypatch.setattr(Mesh, "validate", changed)
    with pytest.raises(RuntimeError, match=f"changed {category} field {field}"):
        module.reconcile_grid_rotate_longitudes(mesh)
    assert all(mesh.arrays[name] is value for name, value in originals.items())
    assert mesh.variable_sources == sources


@pytest.fixture
def pair(tmp_path, monkeypatch):
    grid, static = tmp_path / "grid.nc", tmp_path / "static.nc"
    grid.write_bytes(b"synthetic grid")
    static.write_bytes(b"synthetic static")
    output = graded_sphere_mesh(level=1, zoom=1.0, dtype=np.float64)
    for entity in ("Cell", "Edge", "Vertex"):
        output.arrays[f"indexTo{entity}ID"] = np.arange(output.arrays[f"x{entity}"].size) + 1
    dynamics = copy.deepcopy(output)
    for mesh in (output, dynamics):
        mesh.grid_path = grid
        mesh.variable_sources.update({name: "grid" for name in mesh.arrays})
    def load(cls, grid_path, static_path=None, **kw):
        assert not kw["validate"]
        return copy.deepcopy(output if static_path is None else dynamics)
    monkeypatch.setattr(Mesh, "from_netcdf", classmethod(load))
    return grid, static, output, dynamics


@pytest.mark.parametrize("longitude_shift", [0.0, 2 * np.pi])
def test_output_matches_its_strictly_validated_carriers(pair, monkeypatch, longitude_shift):
    grid, static, original, _ = pair
    for entity in ("Cell", "Edge", "Vertex"):
        original.arrays[f"lon{entity}"] += longitude_shift
    validate = Mesh.validate
    validated = []
    def capture(mesh):
        result = validate(mesh)
        validated.append((mesh, {k: (v.dtype, v.shape, v.tobytes()) for k, v in mesh.arrays.items()}))
        return result
    monkeypatch.setattr(Mesh, "validate", capture)
    dynamics, output, evidence = module.load_precision_preserving_mesh_pair(grid, static)
    output_checks = [arrays for mesh, arrays in validated if mesh is output]
    assert len(output_checks) == 1
    assert output_checks[0] == {k: (v.dtype, v.shape, v.tobytes()) for k, v in output.arrays.items()}
    assert len([mesh for mesh, _ in validated if mesh is dynamics]) == 2
    validate(output)
    validate(dynamics)
    assert evidence["merged_validation_without_tolerance_change"] == "passed"
    assert output.provenance["precision_preserving_mesh_pair"] is evidence
    assert dynamics.provenance["precision_preserving_mesh_pair"] is evidence
    for name in output.arrays:
        assert not np.shares_memory(output.arrays[name], dynamics.arrays[name])


def test_invalid_grid_still_refused(pair):
    grid, static, output, _ = pair
    output.arrays["areaCell"][0] = -1.0
    with pytest.raises(MeshValidationError, match="areaCell"):
        module.load_precision_preserving_mesh_pair(grid, static)


def test_invalid_static_metrics_still_refused_after_overlay(pair):
    grid, static, _, dynamics = pair
    dynamics.arrays["areaCell"][0] = -1.0
    with pytest.raises(MeshValidationError, match="areaCell"):
        module.load_precision_preserving_mesh_pair(grid, static)


def test_changed_topology_still_refused(pair):
    grid, static, _, dynamics = pair
    dynamics.arrays["cellsOnEdge"][0] = dynamics.arrays["cellsOnEdge"][0, ::-1]
    with pytest.raises(MeshValidationError, match="topology differs"):
        module.load_precision_preserving_mesh_pair(grid, static)


def test_failed_raw_geometry_is_repaired_and_revalidated(pair):
    grid, static, _, dynamics = pair
    dynamics.arrays["xCell"] *= 0.5
    repaired, output, evidence = module.load_precision_preserving_mesh_pair(grid, static)
    assert evidence["raw_static_overlay"]["status"] == "failed"
    assert repaired.arrays["xCell"].tobytes() == output.arrays["xCell"].tobytes()
    repaired.validate()


@pytest.mark.parametrize("workers", [1, 4])
def test_grid_is_decoded_once_without_sharing_carriers(tmp_path, monkeypatch, workers):
    grid, static = tmp_path / "grid.nc", tmp_path / "static.nc"
    grid.write_bytes(b"grid payload")
    static.write_bytes(b"static payload")
    original = graded_sphere_mesh(level=1, zoom=1.0, dtype=np.float64)
    for entity in ("Cell", "Edge", "Vertex"):
        original.arrays[f"indexTo{entity}ID"] = np.arange(original.arrays[f"x{entity}"].size) + 1
    stored = {
        name: value + 1 if module._looks_like_connectivity(name, value) else value.copy()
        for name, value in original.arrays.items()
    }
    stored["carrier"] = np.asfortranarray(np.arange(18).reshape(3, 6))
    attrs = dict(original.attrs, file_id="grid", vector=np.arange(3, dtype="i2"))
    variable_attrs = {"carrier": {"vector": np.arange(2, dtype="i4")}}
    calls = []

    def read(path):
        calls.append(path)
        if path == grid:
            return copy.deepcopy((original.dimensions, attrs, stored, {}, variable_attrs))
        assert path == static
        # Leave grid-only topology, metrics and a Fortran-order carrier intact.
        return {}, {"parent_id": "grid"}, {"ter": np.zeros(original.nCells)}, {}, {}

    monkeypatch.setattr(module, "_read_netcdf", read)
    monkeypatch.setenv("WOOF_HEX_HOST_COPY_THREADS", str(workers))
    dynamics, output, evidence = module.load_precision_preserving_mesh_pair(grid, static)
    assert calls == [grid, static]
    assert evidence["raw_static_overlay"]["status"] == "passed"
    assert output.arrays["carrier"].flags.f_contiguous
    assert dynamics.arrays["carrier"].flags.f_contiguous
    for name in output.arrays:
        assert not np.shares_memory(output.arrays[name], dynamics.arrays[name]), name
        assert dynamics.arrays[name].tobytes() == output.arrays[name].tobytes(), name
    for name in original.arrays:
        expected = (module.normalize_longitudes(original.arrays[name])
                    if name in ("lonCell", "lonEdge", "lonVertex") else original.arrays[name])
        assert dynamics.arrays[name].tobytes() == expected.tobytes(), name
    output.arrays["carrier"][0, 0] = -10
    output.grid_attrs["vector"][0] = -11
    output.variable_attrs["carrier"]["vector"][0] = -12
    assert dynamics.arrays["carrier"][0, 0] == 0
    assert dynamics.grid_attrs["vector"][0] == 0
    assert dynamics.variable_attrs["carrier"]["vector"][0] == 0


def test_input_digests_overlap_construction_and_preserve_complete_pair(pair, monkeypatch):
    import threading
    from woof.hex import file_digest, mesh_snapshot

    grid, static, _, _ = pair
    monkeypatch.setenv("WOOF_HEX_DIGEST_CACHE", "off")
    monkeypatch.setenv("WOOF_HEX_HOST_COPY_THREADS", "4")
    expected = module.load_precision_preserving_mesh_pair(grid, static)
    monkeypatch.setattr(file_digest, "MIN_CACHED_BYTES", 0)
    barrier = threading.Barrier(3, timeout=5)
    caller = threading.get_ident()
    hash_file, from_netcdf = module._sha256_file, Mesh.from_netcdf
    reads = []

    def fingerprint(path):
        assert threading.get_ident() != caller
        barrier.wait()
        return hash_file(path)

    def read(cls, grid_path, static_path=None, **kwargs):
        assert threading.get_ident() == caller
        reads.append(static_path)
        if static_path is None:
            barrier.wait()
        return from_netcdf(grid_path, static_path, **kwargs)

    monkeypatch.setattr(module, "_sha256_file", fingerprint)
    monkeypatch.setattr(Mesh, "from_netcdf", classmethod(read))
    actual = module.load_precision_preserving_mesh_pair(grid, static)
    assert reads == [None, static]
    manifests = []
    for carriers in (expected, actual):
        encoder = mesh_snapshot._Encoder()
        structure = encoder.encode((dict(vars(carriers[0])), dict(vars(carriers[1])), carriers[2]))
        manifests.append((structure, [(a.dtype.str, a.shape, a.flags.c_contiguous,
                                      a.flags.f_contiguous, a.flags.writeable,
                                      a.tobytes()) for a in encoder.arrays]))
    assert manifests[0] == manifests[1]


@pytest.mark.parametrize("workers,threshold", [(1, 0), (4, 1 << 30)])
def test_small_or_single_worker_pair_keeps_synchronous_hashes(pair, monkeypatch, workers, threshold):
    import threading
    from woof.hex import file_digest

    grid, static, _, _ = pair
    monkeypatch.setenv("WOOF_HEX_HOST_COPY_THREADS", str(workers))
    monkeypatch.setattr(file_digest, "MIN_CACHED_BYTES", threshold)
    original = module._sha256_file
    caller = threading.get_ident()
    calls = []

    def fingerprint(path):
        assert threading.get_ident() == caller
        calls.append(path)
        return original(path)

    monkeypatch.setattr(module, "_sha256_file", fingerprint)
    module.load_precision_preserving_mesh_pair(grid, static)
    assert calls == [grid, static]


def test_changed_file_is_rehashed_at_its_original_consumption_point(pair, monkeypatch):
    import hashlib
    import threading
    from woof.hex import file_digest

    grid, static, _, _ = pair
    monkeypatch.setenv("WOOF_HEX_HOST_COPY_THREADS", "4")
    monkeypatch.setenv("WOOF_HEX_DIGEST_CACHE", "off")
    monkeypatch.setattr(file_digest, "MIN_CACHED_BYTES", 0)
    ready = threading.Event()
    original = module._sha256_file
    calls = []

    def fingerprint(path):
        value = original(path)
        calls.append(path)
        if path == grid:
            ready.set()
        return value

    def build(grid_path, static_path, *, _file_digest):
        assert ready.wait(5)
        grid.write_bytes(b"changed complete grid authority")
        return _file_digest(grid), _file_digest(static)

    monkeypatch.setattr(module, "_sha256_file", fingerprint)
    monkeypatch.setattr(module, "_load_precision_preserving_mesh_pair", build)
    result = module.load_precision_preserving_mesh_pair(grid, static)
    assert result == (hashlib.sha256(grid.read_bytes()).hexdigest(),
                      hashlib.sha256(static.read_bytes()).hexdigest())
    assert calls.count(grid) == 2


@pytest.mark.parametrize("consume", [False, True])
def test_background_hash_failure_keeps_admission_error_order(pair, monkeypatch, consume):
    from woof.hex import file_digest

    grid, static, _, _ = pair
    monkeypatch.setenv("WOOF_HEX_HOST_COPY_THREADS", "4")
    monkeypatch.setattr(file_digest, "MIN_CACHED_BYTES", 0)

    def fingerprint(path):
        raise PermissionError("input digest refused")

    def build(grid_path, static_path, *, _file_digest):
        if consume:
            return _file_digest(grid)
        raise MeshValidationError("invalid grid before digest consumption")

    monkeypatch.setattr(module, "_sha256_file", fingerprint)
    monkeypatch.setattr(module, "_load_precision_preserving_mesh_pair", build)
    error, message = ((PermissionError, "input digest refused") if consume else
                      (MeshValidationError, "invalid grid before digest consumption"))
    with pytest.raises(error, match=message):
        module.load_precision_preserving_mesh_pair(grid, static)


@pytest.mark.parametrize("layout", ["vector", "strided", "fortran", "scalar", "empty"])
def test_blocked_longitude_receipt_matches_whole_array(layout, monkeypatch):
    from woof.hex import host_parallel
    monkeypatch.setenv("WOOF_HEX_HOST_COPY_THREADS", "4")
    monkeypatch.setattr(host_parallel, "MIN_PARALLEL_ELEMENTS", 0)
    source = np.linspace(-2 * np.pi, 2 * np.pi, 32769)
    if layout == "strided":
        source = source[::-2]
    elif layout == "fortran":
        source = np.asfortranarray(np.tile(source, (3, 1)))
    elif layout == "scalar":
        source = np.asarray(-0.0)
    elif layout == "empty":
        source = np.empty((0, 3))
    candidate = (source + np.pi) % (2.0 * np.pi) - np.pi
    before = source.tobytes(), candidate.tobytes()
    expected = {
        "method": "(longitude + pi) modulo 2pi minus pi",
        "sin_max_abs_gap": float(np.max(np.abs(np.sin(candidate) - np.sin(source)), initial=0.0)),
        "cos_max_abs_gap": float(np.max(np.abs(np.cos(candidate) - np.cos(source)), initial=0.0)),
        "trig_equivalence_atol": module.LONGITUDE_TRIG_EQUIVALENCE_ATOL,
    }
    assert module.validate_longitude_normalization(source, candidate) == expected
    assert (source.tobytes(), candidate.tobytes()) == before


@pytest.mark.parametrize("failure", ["shape", "finite", "mapping", "trig"])
def test_blocked_longitude_errors_keep_whole_array_precedence(failure, monkeypatch):
    from woof.hex import host_parallel
    monkeypatch.setenv("WOOF_HEX_HOST_COPY_THREADS", "4")
    monkeypatch.setattr(host_parallel, "MIN_PARALLEL_ELEMENTS", 0)
    source = np.linspace(-np.pi, np.pi, 32769)
    candidate = (source + np.pi) % (2.0 * np.pi) - np.pi
    if failure == "shape":
        candidate = candidate[:-1]
        message = "longitude normalization requires same-shaped finite arrays"
    elif failure == "finite":
        candidate[0] = 100.0  # Earlier block has a mapping/range error.
        source[-1] = np.nan  # Finite admission must still win globally.
        message = "longitude normalization requires same-shaped finite arrays"
    elif failure == "mapping":
        candidate[-1] = 100.0
        message = "longitude normalization is not the exact modulo mapping"
    else:
        source[-1] = 1e16
        candidate[-1] = (source[-1] + np.pi) % (2.0 * np.pi) - np.pi
        message = "longitude modulo mapping is not trig-equivalent"
    with pytest.raises(ValueError) as error:
        module.validate_longitude_normalization(source, candidate)
    assert str(error.value) == message
