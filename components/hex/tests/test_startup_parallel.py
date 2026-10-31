"""fix/rrfs-startup-parallel: the start-up pieces taken off one core write the words they wrote on it."""
from __future__ import annotations

import numpy as np
import pytest

from woof.hex import host_parallel, prep_cache
from woof.hex.diagnostics import edge_signs_on_vertices
from woof.hex.mpasphys.framework_backend_v842 import FrameworkPhysics


def _loop_edge_signs(edges, vertices):
    signs = np.zeros(edges.shape, dtype=np.float64)
    for vertex in range(edges.shape[0]):
        for slot, edge in enumerate(edges[vertex]):
            if edge >= 0:
                signs[vertex, slot] = 1.0 if vertices[edge, 1] == vertex else -1.0
    return signs


class _Mesh:
    def __init__(self, **arrays):
        self.arrays = arrays

    def __getattr__(self, name):
        try:
            return self.__dict__['arrays'][name]
        except KeyError:
            raise AttributeError(name) from None


def test_edge_signs_match_the_loop():
    rng = np.random.default_rng(7)
    n_vertices, n_edges = 2000, 3000
    vertices = rng.integers(0, n_vertices, size=(n_edges, 2)).astype(np.int32)
    edges = rng.integers(0, n_edges, size=(n_vertices, 3)).astype(np.int32)
    edges[rng.random(edges.shape) < 0.1] = -1
    # Make some vertices the second vertex of their edges, so both signs occur.
    for vertex in range(0, n_vertices, 3):
        if edges[vertex, 0] >= 0:
            vertices[edges[vertex, 0], 1] = vertex
    mesh = _Mesh(edgesOnVertex=edges, verticesOnEdge=vertices)
    got = edge_signs_on_vertices(mesh)
    want = _loop_edge_signs(edges.astype(np.int64), vertices.astype(np.int64))
    assert got.dtype == np.float64 and got.shape == want.shape
    assert got.tobytes() == want.tobytes()
    assert {1.0, -1.0, 0.0} <= set(np.unique(got).tolist())


@pytest.mark.parametrize("order", ["F", "C"])
def test_copy_into_writes_the_same_bytes(monkeypatch, order):
    monkeypatch.setattr(host_parallel, "MIN_PARALLEL_ELEMENTS", 16)
    monkeypatch.setattr(host_parallel, "MIN_BLOCK_COLUMNS", 7)
    monkeypatch.setenv(host_parallel.ENVIRONMENT, "4")
    rng = np.random.default_rng(1)
    words = rng.integers(0, 2**32, size=(5, 37, 41), dtype=np.uint32)
    words[0, 0, :4] = [0x7FC00001, 0x7F800001, 0xFFC12345, 0x00000001]   # NaN payloads, subnormal
    source = np.asfortranarray(words.view(np.float32)) if order == "F" else words.view(np.float32)
    target = np.zeros(source.shape, np.float32, order="F")
    host_parallel.copy_into(target, source)
    assert target.view(np.uint32).tobytes(order="F") == source.view(np.uint32).tobytes(order="F")
    out = host_parallel.copied(source, order="F")
    assert out.flags.f_contiguous and out.view(np.uint32).tobytes(order="F") == source.view(np.uint32).tobytes(order="F")


def test_adopt_field_holds_put_field_words():
    rng = np.random.default_rng(3)
    value = np.asfortranarray(rng.random((4, 6, 9), dtype=np.float32))
    put, adopt = FrameworkPhysics(), FrameworkPhysics()
    for level in (1, 2):
        put.put_field('state', 'theta_m', value, level=level, nlevels=2)
    adopt.adopt_field('state', 'theta_m', value.copy(order='F'), nlevels=2)
    assert put._fields['state', 'theta_m'] == adopt._fields['state', 'theta_m']
    for level in (1, 2):
        a = put.get_field('state', 'theta_m', level=level)
        b = adopt.get_field('state', 'theta_m', level=level)
        assert a.tobytes(order='F') == b.tobytes(order='F')
    # The time levels are independent storage, as put_field's are.
    adopt.field_storage('state', 'theta_m', level=1)[0, 0, 0] = 5.0
    assert adopt.get_field('state', 'theta_m', level=2)[0, 0, 0] == value[0, 0, 0]
    # A C-ordered buffer is not adopted; it is stored as put_field stores it.
    other = FrameworkPhysics()
    c_value = np.ascontiguousarray(value)
    other.adopt_field('state', 'w', c_value)
    assert other.field_storage('state', 'w') is not c_value
    assert other.get_field('state', 'w').tobytes(order='F') == value.tobytes(order='F')


def test_block_digests_cover_every_byte(monkeypatch):
    monkeypatch.setattr(prep_cache, "_PARALLEL_BYTES", 64)
    monkeypatch.setattr(prep_cache, "_BLOCK_BYTES", 256)
    rng = np.random.default_rng(5)
    arrays = [rng.random(1000), np.asfortranarray(rng.random((30, 20))), np.arange(10), rng.random(40)]
    digests = prep_cache._array_digests(arrays)
    assert digests == [prep_cache._array_digest(a) for a in arrays]
    changed = [a.copy() for a in arrays]
    changed[0][999] += 1.0
    changed[1][29, 19] += 1.0
    again = prep_cache._array_digests(changed)
    assert again[0] != digests[0] and again[1] != digests[1] and again[2:] == digests[2:]


@pytest.mark.parametrize("shape,axis", [((55, 5003), -1), ((7, 9, 4001), -1), ((4001, 6), 0), ((9001,), 0)])
def test_pad_blocks_the_longest_axis_and_writes_the_concatenate(monkeypatch, shape, axis):
    from woof.hex import cuda_regional_forecast_v841 as regional
    monkeypatch.setattr(host_parallel, "MIN_PARALLEL_ELEMENTS", 16)
    monkeypatch.setattr(host_parallel, "MIN_BLOCK_COLUMNS", 64)
    monkeypatch.setenv(host_parallel.ENVIRONMENT, "4")
    calls = []
    real = host_parallel.run_blocks
    monkeypatch.setattr(host_parallel, "run_blocks",
                        lambda function, columns, **kw: (calls.append(columns), real(function, columns, **kw)))
    data = np.random.default_rng(11).random(shape).astype(np.float32)
    got = regional._padded(data, axis, 0.0)
    pad_shape = list(shape)
    pad_shape[axis] = 1
    want = np.concatenate([data, np.zeros(pad_shape, np.float32)], axis=axis)
    assert got.shape == want.shape and got.tobytes() == want.tobytes()
    assert calls == [max(shape)]
