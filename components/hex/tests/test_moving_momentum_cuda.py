import os
from types import SimpleNamespace
import numpy as np
import pytest

from woof.hex.moving_state import reconstruct_cartesian, mass_flux_divergence
from woof.hex import moving_momentum_cuda as cuda


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
@pytest.mark.parametrize('fmad', ['false','true'])
@pytest.mark.parametrize('dtype', [np.float32,np.float64])
def test_gpu_reconstruction_and_divergence_keep_slot_order_words(monkeypatch, fmad, dtype):
    pytest.importorskip('cupy')
    monkeypatch.setattr(cuda,'OPTIONS',(f'--fmad={fmad}', '--ftz=true'))
    cuda._kernels.cache_clear()
    rng = np.random.default_rng(1031)
    n, levels = 37, 55
    c = np.arange(n)
    arrays = dict(edgesOnCell=np.column_stack(((c-1)%n,c,(c+3)%n,np.full(n,-1))),
        nEdgesOnCell=c%4, cellsOnEdge=np.column_stack((c,(c+1)%n)),
        coeffs_reconstruct=rng.normal(size=(n,4,3)),dvEdge=rng.uniform(.1,2.,n),areaCell=rng.uniform(.1,2.,n))
    normal = rng.normal(size=(levels,n)).astype(dtype)
    normal[:,0] = -0.;normal[::2,1] *= dtype(1e-30)
    rho = rng.uniform(.1,2.,(levels,n)).astype(dtype)
    w = SimpleNamespace()
    try:
        expected = reconstruct_cartesian(normal,arrays)*rho[:,:,None]
        assert cuda.reconstruct_mass(w,normal,rho,arrays).tobytes() == expected.tobytes()
        for scale in (1.,.25):
            flux = (normal*scale).astype(dtype)
            expected = mass_flux_divergence(flux,arrays)
            assert cuda.divergence(w,flux,arrays,kind='source').tobytes() == expected.tobytes()
        cached = w._cuda_momentum_geometry['source']
        cuda.prepare(w,arrays,arrays)
        assert w._cuda_momentum_geometry['source'] is cached
    finally:
        cuda._kernels.cache_clear()


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
@pytest.mark.parametrize('fmad', ['false','true'])
@pytest.mark.parametrize('order', ['C','F'])
def test_tangent_and_edge_projection_keep_einsum_words_and_missing_cell_zero(monkeypatch, fmad, order):
    monkeypatch.setattr(cuda, 'OPTIONS', (f'--fmad={fmad}', '--ftz=true'))
    cuda._kernels.cache_clear()
    rng=np.random.default_rng(405)
    nc,nl,ne=37,55,53
    centres=rng.normal(size=(nc,3))
    cells=rng.integers(0,nc,size=(ne,2));cells[::7,1]=-1
    normals=np.array(rng.normal(size=(ne,3)),order=order)
    arrays=dict(xCell=centres[:,0],yCell=centres[:,1],zCell=centres[:,2],
                cellsOnEdge=cells,edgeNormalVectors=normals)
    vector=rng.normal(size=(nl,nc,3))
    vector[:,0]=-0.;vector[:,1:4]*=1.e-40
    centres=centres/np.linalg.norm(centres,axis=-1,keepdims=True)
    radial=np.einsum('lcj,cj->lc',vector,centres)
    tangent=vector-radial[:,:,None]*centres
    gather=np.where(cells<0,nc,cells)
    padded=np.concatenate((tangent,np.zeros_like(tangent[:,:1])),axis=1)
    flux=np.einsum('lej,ej->le',(padded[:,gather[:,0]]+padded[:,gather[:,1]])*.5,normals).astype(np.float32)
    try:
        actual,actual_radial,_,actual_gather,actual_flux=cuda.project_edges(SimpleNamespace(),vector,arrays)
        for a,b in [(actual,tangent),(actual_radial,radial),(actual_flux,flux)]:
            assert a.tobytes()==b.tobytes()
        np.testing.assert_array_equal(actual_gather,gather)
    finally:
        cuda._kernels.cache_clear()
