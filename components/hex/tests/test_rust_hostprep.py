"""Exact scalar-order parity for the default forecast host bridge."""
from __future__ import annotations

import numpy as np
import pytest
from itertools import product

from woof.hex import rust_hostprep
from woof.hex import doctor, engines
from woof.hex.terrain import build_terrain_coupling
import _hostprep_authority as authority
from test_mesh_validation import graded_sphere_mesh
from test_regional_mesh_admission import cull_regional_mesh


def assert_bits(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(f"u{actual.dtype.itemsize}"),
                                  expected.view(f"u{expected.dtype.itemsize}"))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("coefficient", [0.0, 0.25, 0.3])
@pytest.mark.parametrize("regional", [False, True])
def test_terrain_metrics_bits(dtype, coefficient, regional):
    mesh = graded_sphere_mesh(level=1)
    counts = np.asarray(mesh.nEdgesOnCell, dtype=np.int64)
    edges = np.asarray(mesh.edgesOnCell, dtype=np.int64)
    cells = np.asarray(mesh.cellsOnEdge, dtype=np.int64).copy()
    if regional:
        cells[0, 1] = -1
        # The absent cell no longer uses the cut edge.
        missing_cell = int(mesh.cellsOnEdge[0, 1])
        used = list(edges[missing_cell, :counts[missing_cell]])
        used.remove(0)
        counts[missing_cell] -= 1
        edges = edges.copy()
        edges[missing_cell] = -1
        edges[missing_cell, :len(used)] = used
    rng = np.random.default_rng(20261004)
    zb = rng.normal(size=(7, 2, len(cells))).astype(dtype)
    zb3 = rng.normal(size=zb.shape).astype(dtype)
    zb[0, :, 0] = [0.0, -0.0]
    zb3[0, :, 0] = [-0.0, 0.0]
    options = dict(n_edges_on_cell=counts, edges_on_cell=edges,
        cells_on_edge=cells, zb=zb, zb3=zb3,
        config_coef_3rd_order=coefficient, allow_regional_sentinels=regional)
    expected = authority.build_terrain_coupling(**options)
    actual = build_terrain_coupling(**options)
    for name in ("n_edges_on_cell", "edges_on_cell", "cells_on_edge",
                 "edges_on_cell_sign", "zb_cell", "zb3_cell"):
        assert_bits(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("regional", [False, True])
def test_threaded_terrain_blocks_match_scalar_authority(monkeypatch, dtype, regional):
    # conus3km start-up (2026-10-29): mode 0 runs in-process on cell blocks; tiny blocks on four threads
    # must write the authority's words, from native (edge, side, level) storage as the driver passes it.
    from woof.hex import host_parallel
    monkeypatch.setattr(host_parallel, 'MIN_PARALLEL_ELEMENTS', 0)
    monkeypatch.setattr(host_parallel, 'MIN_BLOCK_COLUMNS', 7)
    monkeypatch.setenv(host_parallel.ENVIRONMENT, '4')
    test_terrain_metrics_bits(dtype, 0.3, regional)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_threaded_momentum_blocks_match_scalar_authority(monkeypatch, dtype):
    # conus3km start-up (2026-10-29): mode 1 runs in-process on cell blocks with the bridge's operation order.
    from woof.hex import host_parallel
    monkeypatch.setattr(host_parallel, 'MIN_PARALLEL_ELEMENTS', 0)
    monkeypatch.setattr(host_parallel, 'MIN_BLOCK_COLUMNS', 3)
    monkeypatch.setenv(host_parallel.ENVIRONMENT, '4')
    monkeypatch.setattr(rust_hostprep, '_run', lambda **kw: pytest.fail('mode 1 must not need the bridge'))
    test_initialized_vertical_momentum_bits(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_initialized_vertical_momentum_bits(dtype):
    rng = np.random.default_rng(20261004)
    nc, ne, me, nl = 13, 31, 7, 9
    counts = rng.integers(0, me + 1, size=nc, dtype=np.int64)
    edges = rng.integers(0, ne, size=(nc, me), dtype=np.int64)
    ru = rng.normal(size=(nl, ne)).astype(dtype)
    ru[:2, :2] = [[0.0, -0.0], [-0.0, 0.0]]
    arrays = dict(counts=counts, edges=edges,
        fzm=rng.normal(size=nl).astype(dtype), fzp=rng.normal(size=nl).astype(dtype),
        ru=ru, interface_zz=rng.normal(size=(nl + 1, nc)).astype(dtype),
        signs=rng.choice([-1.0, 1.0], size=(nc, me)).astype(dtype),
        zb_cell=rng.normal(size=(nl + 1, nc, me)).astype(dtype),
        zb3_cell=rng.normal(size=(nl + 1, nc, me)).astype(dtype),
        rw=rng.normal(size=(nl + 1, nc)).astype(dtype))
    arrays["fzm"][:2] = dtype(1.0)
    arrays["fzp"][:2] = dtype(0.0)
    expected = authority.correct_vertical_momentum(**arrays)
    assert_bits(rust_hostprep.correct_vertical_momentum(**arrays), expected)


def topology_arguments(mesh, regional=False):
    return dict(n_vertices=mesh.dimensions["nVertices"],
        max_edges2=mesh.dimensions["maxEdges2"], vertex_degree=mesh.dimensions["vertexDegree"],
        regional=regional, counts=mesh.nEdgesOnCell.copy(), edges=mesh.edgesOnCell.copy(),
        cells_on_cell=mesh.cellsOnCell.copy(), vertices_on_cell=mesh.verticesOnCell.copy(),
        cells_on_edge=mesh.cellsOnEdge.copy(), vertices_on_edge=mesh.verticesOnEdge.copy(),
        cells_on_vertex=mesh.cellsOnVertex.copy(), edges_on_vertex=mesh.edgesOnVertex.copy(),
        edge_counts=mesh.nEdgesOnEdge.copy(), edges_on_edge=mesh.edgesOnEdge.copy())


@pytest.mark.parametrize("regional", [False, True])
@pytest.mark.parametrize("mutation", [None, "cell_duplicate", "edge_duplicate",
    "vertex_duplicate", "stencil_duplicate", "mixed_errors", "disconnected"])
def test_topology_error_bytes(regional, mutation):
    parent = graded_sphere_mesh(level=3)
    mesh = cull_regional_mesh(parent, seed_cell=0, disk_hops=12) if regional else parent
    arrays = topology_arguments(mesh, regional)
    if mutation in ("cell_duplicate", "mixed_errors"):
        arrays["edges"][:8, 1] = arrays["edges"][:8, 0]
    if mutation in ("edge_duplicate", "mixed_errors"):
        arrays["cells_on_edge"][:8, 1] = arrays["cells_on_edge"][:8, 0]
    if mutation in ("vertex_duplicate", "mixed_errors"):
        arrays["cells_on_vertex"][:8, 1] = arrays["cells_on_vertex"][:8, 0]
    if mutation in ("stencil_duplicate", "mixed_errors"):
        arrays["edges_on_edge"][:8, 1] = arrays["edges_on_edge"][:8, 0]
    if mutation == "disconnected":
        arrays["cells_on_cell"][:] = -1
    expected = authority.topology_errors(**arrays)
    assert rust_hostprep.topology_errors(**arrays) == expected
    if mutation is None:
        assert expected == []
    else:
        assert expected


@pytest.mark.parametrize("mutation", ["edge_range", "duplicate", "reciprocity"])
def test_terrain_refusal_messages_match(mutation):
    mesh = graded_sphere_mesh(level=1)
    counts = mesh.nEdgesOnCell.copy()
    edges = mesh.edgesOnCell.copy()
    cells = mesh.cellsOnEdge.copy()
    if mutation == "edge_range":
        edges[0, 0] = len(cells)
    elif mutation == "duplicate":
        edges[0, 1] = edges[0, 0]
    else:
        cells[int(edges[0, 0])] = [1, 2]
    arrays = dict(n_edges_on_cell=counts, edges_on_cell=edges, cells_on_edge=cells,
        zb=np.zeros((3, 2, len(cells)), dtype=np.float32),
        zb3=np.zeros((3, 2, len(cells)), dtype=np.float32), config_coef_3rd_order=0.25)
    with pytest.raises(ValueError) as expected:
        authority.build_terrain_coupling(**arrays)
    with pytest.raises(ValueError) as actual:
        build_terrain_coupling(**arrays)
    assert str(actual.value) == str(expected.value)


def test_doctor_reports_the_default_host_preparation_bridge(monkeypatch):
    assert engines.HOSTPREP in engines.ENGINES
    monkeypatch.setattr(engines, "gpuwm_bundles", lambda spec: True)
    monkeypatch.setattr(engines, "locate", lambda spec: (None, "not staged"))
    findings = doctor.check_engines()
    required = [finding for finding in findings
                if finding.subject.startswith("rw_mpas_hostprep ")]
    assert len(required) == 1
    assert required[0].status == doctor.MISSING
    assert required[0].required
    assert "fetch-bridges" in required[0].remedy


@pytest.mark.parametrize("mutation", ["duplicate_before_range", "reciprocity_before_range",
                                    "earlier_cell_before_later_range"])
def test_mixed_terrain_refusals_keep_cell_slot_priority(mutation):
    counts = np.asarray([3, 0], dtype=np.int64)
    cells = np.asarray([[0, 1]], dtype=np.int64)
    edges = np.asarray([[0, 0, 1], [-1, -1, -1]], dtype=np.int64)
    if mutation == "reciprocity_before_range":
        counts = np.asarray([3, 0, 0], dtype=np.int64)
        cells = np.asarray([[1, 2]], dtype=np.int64)
        edges = np.asarray([[0, 1, 0], [-1, -1, -1], [-1, -1, -1]], dtype=np.int64)
    elif mutation == "earlier_cell_before_later_range":
        counts = np.asarray([2, 1], dtype=np.int64)
        edges = np.asarray([[0, 0, -1], [1, -1, -1]], dtype=np.int64)
    arrays = dict(n_edges_on_cell=counts, edges_on_cell=edges, cells_on_edge=cells,
        zb=np.zeros((3, 2, 1), dtype=np.float32), zb3=np.zeros((3, 2, 1), dtype=np.float32),
        config_coef_3rd_order=0.25)
    with pytest.raises(ValueError) as expected:
        authority.build_terrain_coupling(**arrays)
    with pytest.raises(ValueError) as actual:
        build_terrain_coupling(**arrays)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("widths", list(product((np.float32, np.float64), repeat=8)))
def test_every_mixed_momentum_field_dtype_preserves_scalar_promotions(widths):
    rng = np.random.default_rng(104)
    nl, ne, nc, me = 4, 5, 3, 3
    shapes = [(nl,), (nl,), (nl, ne), (nl + 1, nc), (nc, me),
              (nl + 1, nc, me), (nl + 1, nc, me), (nl + 1, nc)]
    names = ("fzm", "fzp", "ru", "interface_zz", "signs", "zb_cell", "zb3_cell", "rw")
    arrays = {name: rng.normal(size=shape).astype(dtype)
              for name, shape, dtype in zip(names, shapes, widths)}
    arrays["ru"][:2, :2] = [[0.0, -0.0], [-0.0, 0.0]]
    arrays["fzm"][:2] = 1.0
    arrays["fzp"][:2] = 0.0
    arrays.update(counts=np.asarray([3, 2, 1], dtype=np.int64),
                  edges=np.asarray([[0, 1, 2], [3, 4, -1], [2, -1, -1]], dtype=np.int64))
    assert_bits(rust_hostprep.correct_vertical_momentum(**arrays),
                authority.correct_vertical_momentum(**arrays))


def test_mixed_vertical_weight_keeps_the_store_rounding_bit():
    arrays = dict(counts=np.asarray([1], dtype=np.int64), edges=np.asarray([[0]], dtype=np.int64),
        fzm=np.asarray([1, 1.00000003], dtype=np.float64), fzp=np.zeros(2, dtype=np.float32),
        ru=np.ones((2, 1), dtype=np.float32), interface_zz=np.ones((3, 1), dtype=np.float32),
        signs=np.ones((1, 1), dtype=np.float32), zb_cell=np.ones((3, 1, 1), dtype=np.float32),
        zb3_cell=np.zeros((3, 1, 1), dtype=np.float32), rw=np.full((3, 1), 2, dtype=np.float32))
    expected = authority.correct_vertical_momentum(**arrays)
    assert expected.view(np.uint32)[1, 0] == 0x3f7fffff
    assert_bits(rust_hostprep.correct_vertical_momentum(**arrays), expected)
