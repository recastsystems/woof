"""Per-launch host checks run once (VRFS 8-card r700, 2026-10-28).

DeviceMesh.validate and CudaV841Context.validate ran in full on every
dycore launch, and the forecast writer and output diagnostics expanded every
stream's membership again on every step and per diagnostic field queried.
Each is now done once; these tests pin that a repeat is free and that any
change is still caught.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.streams_v842 import STREAM_NAMES, StreamFieldCache, registry_stream


def _streams():
    return {name: registry_stream(name) for name in STREAM_NAMES}


def test_stream_field_cache_matches_expansion_and_follows_changes(monkeypatch):
    streams = _streams()
    packages = ('mp_thompson_in', 'cu_ntiedtke_in', 'bl_mynn_in')
    cache = StreamFieldCache()
    for stream in streams.values():
        assert cache.fields(streams, packages, stream) == stream.fields(packages, streams=streams)
    calls = []
    original = type(streams['output']).fields
    monkeypatch.setattr(type(streams['output']), 'fields',
                        lambda self, *a, **k: calls.append(self) or original(self, *a, **k))
    expansions = lambda target: sum(call is target for call in calls)
    for stream in streams.values():
        cache.fields(streams, packages, stream)
    assert calls == []
    # Another package set, another object under a name, or a stream outside the mapping: expanded again.
    cache.fields(streams, packages + ('sf_mynn_in',), streams['output'])
    assert expansions(streams['output']) == 1
    streams['output'] = replacement = registry_stream('output')
    cache.fields(streams, packages, replacement)
    cache.fields(streams, packages, replacement)
    assert expansions(replacement) == 1
    outside = registry_stream('diagnostics')
    cache.fields(streams, packages, outside)
    cache.fields(streams, packages, outside)
    assert expansions(outside) == 2


def _cupy():
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('no CUDA device')
    except Exception as error:  # pragma: no cover - no driver
        pytest.skip(f'no CUDA device: {error}')
    return cp


def _device_mesh(cp):
    from woof.hex.cuda_backend.containers import DeviceMesh, TransferStats

    nc, ne, nv, me, me2, vd = 4, 6, 3, 3, 5, 3
    i = lambda *shape: cp.zeros(shape, dtype=cp.int32)
    f = lambda *shape: cp.zeros(shape, dtype=cp.float32)
    return DeviceMesh(
        cells_on_edge=i(ne, 2), edges_on_cell=i(nc, me), n_edges_on_cell=i(nc),
        cells_on_cell=i(nc, me), vertices_on_edge=i(ne, 2), edges_on_edge=i(ne, me2),
        n_edges_on_edge=i(ne), vertices_on_cell=i(nc, me), edges_on_vertex=i(nv, vd),
        cells_on_vertex=i(nv, vd), edge_sign_on_cell=f(nc, me), weights_on_edge=f(ne, me2),
        dc_edge=f(ne), dv_edge=f(ne), area_cell=f(nc), area_triangle=f(nv),
        kite_areas_on_vertex=f(nv, vd), lat_cell=f(nc), lon_cell=f(nc), lat_edge=f(ne),
        lon_edge=f(ne), angle_edge=f(ne), mesh_density=f(nc), defc_a=f(nc, me), defc_b=f(nc, me),
        f_vertex=f(nv), f_edge=f(ne), spec_zone_mask_edge=f(ne), n_cells=nc, n_edges=ne,
        n_vertices=nv, max_edges=me, max_edges2=me2, vertex_degree=vd, nominal_min_dc=1.0,
        dtype=np.dtype(np.float32), index_dtype=np.dtype(np.int32), h2d=TransferStats(0, 0.0))


def test_mesh_and_context_validate_once_and_recheck_any_change(monkeypatch):
    cp = _cupy()
    from woof.hex import cuda_v841
    from woof.hex.cuda_backend import containers

    mesh = _device_mesh(cp)
    calls = []
    original = containers.require_resident_array
    counting = lambda name, *a, **k: calls.append(name) or original(name, *a, **k)
    monkeypatch.setattr(containers, 'require_resident_array', counting)
    mesh.validate()
    assert len(calls) == 28
    mesh.validate()
    assert len(calls) == 28
    mesh.dc_edge = cp.zeros(6, dtype=cp.float32)      # another array: checked in full again
    mesh.validate()
    assert len(calls) == 56
    mesh.dc_edge = cp.zeros(6, dtype=cp.float64)      # a wrong one is still refused
    with pytest.raises(TypeError):
        mesh.validate()
    mesh.dc_edge = cp.zeros(6, dtype=cp.float32)
    mesh.validate()
    mesh.dc_edge.shape = (3, 2)                       # an in-place shape change is seen
    with pytest.raises(ValueError):
        mesh.validate()

    nlev = 2
    context = cuda_v841.CudaV841Context(
        inv_area_cell=cp.ones(4, cp.float32), inv_area_triangle=cp.ones(3, cp.float32),
        inv_dc_edge=cp.ones(6, cp.float32), inv_dv_edge=cp.ones(6, cp.float32),
        etp=cp.ones(nlev, cp.float32), etm=cp.ones(nlev, cp.float32),
        ewp=cp.ones(nlev + 1, cp.float32), ewm=cp.ones(nlev + 1, cp.float32),
        u_init=cp.ones(nlev, cp.float32), v_init=cp.ones(nlev, cp.float32),
        h2d=containers.TransferStats(0, 0.0))
    calls.clear()
    monkeypatch.setattr(cuda_v841, 'require_resident_array', counting)
    dims = dict(n_vert_levels=nlev, n_cells=4, n_edges=6, n_vertices=3)
    context.validate(**dims)
    context.validate(**dims)
    assert len(calls) == 10
    with pytest.raises(ValueError):                   # other dimensions: checked (and refused) again
        context.validate(**{**dims, 'n_cells': 5})
