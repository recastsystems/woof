"""The admitted class's GPU operators keep native insertion and rounding order."""
import os
from types import SimpleNamespace
import numpy as np
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA')!='1',reason='CUDA lease required')


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('order',[2,3,4])
@pytest.mark.parametrize('fmad',['false','true'])
def test_native_advection_words(dtype,order,fmad,monkeypatch):
    from woof.hex import moving_advection_cuda as cuda
    from woof.hex.transport import build_advection_coefficients
    rng=np.random.default_rng(176)
    n=31;c=np.arange(n)
    counts=np.full(n,2,np.int64);counts[3::7]=0
    mesh=SimpleNamespace(arrays=dict(nEdgesOnCell=counts,
        edgesOnCell=np.column_stack(((c-1)%n,c)),
        cellsOnCell=np.column_stack(((c-1)%n,(c+1)%n)),
        cellsOnEdge=np.column_stack((c,(c+1)%n)),
        indexToCellID=rng.permutation(n)+1,
        deriv_two=rng.normal(size=(n,2,8)).astype(dtype),
        dcEdge=rng.uniform(100.,1e5,n).astype(dtype),
        dvEdge=rng.uniform(100.,1e5,n).astype(dtype)))
    mesh.arrays['deriv_two'][::3]=dtype(-0.)
    mesh.arrays['dvEdge'][0]=dtype(-1.)
    args=dict(config_scalar_adv_order=order,n_vert_levels=55)
    expected=build_advection_coefficients(mesh,source_order_v841=True,**args)
    monkeypatch.setattr(cuda,'OPTIONS',('--fmad='+fmad,));cuda._kernel.cache_clear()
    actual=cuda.build(mesh,**args)
    for name in ('adv_coefs','adv_coefs_3rd','n_adv_cells_for_edge',
                 'adv_cells_for_edge','high_order_advection_mask'):
        assert getattr(actual,name).tobytes()==getattr(expected,name).tobytes(),name
