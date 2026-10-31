import os
from types import SimpleNamespace
import numpy as np
import pytest
from woof.hex.moving_velocity import recover_vertical_velocity
from woof.hex import moving_velocity_cuda as cuda


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('fmad', ['false', 'true'])
def test_device_w_keeps_host_slot_words_with_ftz_enabled(monkeypatch, dtype, fmad):
    monkeypatch.setattr(cuda, 'OPTIONS', (f'--fmad={fmad}', '--ftz=true'))
    cuda._kernel.cache_clear()
    rng = np.random.default_rng(4599)
    nc, nl, width = 37, 55, 4
    cells = np.column_stack((np.arange(nc), (np.arange(nc)+1)%nc))
    edges = np.column_stack([(np.arange(nc)+i)%nc for i in range(width)])
    counts = np.arange(nc)%5
    edges[np.arange(width)[None, :] >= counts[:, None]] = -1
    arrays = dict(nEdgesOnCell=counts, edgesOnCell=edges, cellsOnEdge=cells)
    state = SimpleNamespace(rho=rng.uniform(.1, 2, (nl, nc)).astype(dtype),
        rho_u=rng.normal(size=(nl, nc)).astype(dtype), rho_w=rng.normal(size=(nl+1, nc)).astype(dtype))
    state.rho_u[:, 1] = dtype(-0.)
    state.rho_u[:, 3] *= dtype(1e-40)
    state.rho_w[:, 4] *= dtype(1e-40)
    vertical = SimpleNamespace(zz=rng.uniform(.5, 1.5, (nl, nc)).astype(dtype),
        fzm=np.full(nl, .42, dtype), fzp=np.full(nl, .58, dtype), cf1=1.2, cf2=-.3, cf3=.1)
    terrain = SimpleNamespace(zb_cell=rng.normal(size=(nl+1, nc, width)).astype(dtype),
        zb3_cell=rng.normal(size=(nl+1, nc, width)).astype(dtype))
    expected = recover_vertical_velocity(state, vertical, arrays, terrain)
    weights = SimpleNamespace(backend='cuda')
    try:
        assert cuda.prepare(weights, vertical, arrays, terrain)
        actual = recover_vertical_velocity(state, vertical, arrays, terrain, device_map=weights)
        assert actual.tobytes() == expected.tobytes()
        cached = weights._cuda_velocity_geometry
        recover_vertical_velocity(state, vertical, arrays, terrain, device_map=weights)
        assert cached is weights._cuda_velocity_geometry
    finally:
        cuda._kernel.cache_clear()
