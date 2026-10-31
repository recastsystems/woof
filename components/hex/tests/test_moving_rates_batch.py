import os
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from woof.hex.moving_remap import ConservativeMap
from woof.hex.moving_vertical import HeightConservativeMap
from woof.hex.moving_physics import carry_horizontal_vector
from woof.hex.mpasphys.moving_rates_batch_v842 import remap


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1',reason='requires an owned CUDA card')
def test_pinned_rate_uploads_keep_all_bits_layouts_and_dtype_groups(monkeypatch):
    import cupy as cp
    from woof.hex.mpasphys import moving_rates_batch_v842 as rates
    rates.prepare_uploads(SimpleNamespace(source_area=np.ones(3),source_dzw=np.ones(3)))
    def no_more_pinning(*args):
        raise AssertionError('prepared rate uploads must not pin pages during the move')
    monkeypatch.setattr(cp.cuda,'alloc_pinned_memory',no_more_pinning)
    calls=[]
    def transfer(value):
        calls.append(value.nbytes)
        return cp.asarray(value)
    device=SimpleNamespace(cuda=cp.cuda,asarray=transfer)
    bits=np.resize(np.array([0x80000000,0x7fc12345,1,0x3f800001,0,0x7fa54321],np.uint32),9).reshape(3,3)
    pools=dict(a=np.array(bits.view(np.float32),order='C'),b=np.array(bits.view(np.float32),order='F'),
               e=np.array(bits.view(np.float32),order='F'),
               c=np.arange(9,dtype=np.float64).reshape(3,3),d=np.arange(18,dtype=np.float64).reshape(3,6)[:,::2])
    actual,staging=rates._upload_inputs(device,pools,list(pools),3)
    cp.cuda.get_current_stream().synchronize()
    assert len(calls)==len(staging)==2
    for key,expected in pools.items():
        assert actual[key].shape==expected.shape and actual[key].dtype==expected.dtype
        assert cp.asnumpy(actual[key]).tobytes()==expected.tobytes()


@pytest.mark.parametrize('backend', ['cpu','cuda'])
@pytest.mark.parametrize('batch_mib', ['0','256'])
@pytest.mark.parametrize('dtype', [np.float32,np.float64])
@pytest.mark.parametrize('order', ['C','F'])
def test_batched_rates_and_vector_rotations_keep_each_fields_words(monkeypatch, backend, batch_mib, dtype, order):
    if backend == 'cuda' and os.environ.get('HEX_TEST_CUDA') != '1':
        pytest.skip('requires an owned CUDA card')
    monkeypatch.setenv('WOOF_HEX_MOVING_RATE_BATCH_MIB', batch_mib)
    rng = np.random.default_rng(3010)
    ns, nt, levels = 19, 17, 9
    matrix = np.zeros((nt,ns))
    for r in range(nt):matrix[r,[r,(r+1)%ns]] = [.75,.25]
    h = ConservativeMap(csr_matrix(matrix), np.ones(ns), np.ones(nt), {})
    z = np.linspace(0.,10000.,levels+1)[:,None]*np.ones((1,ns))
    target = np.linspace(0.,10000.,levels+1)[:,None]*np.ones((1,nt))
    target += (1-target/10000.)*rng.uniform(-200.,200.,nt)
    w = HeightConservativeMap(h,z,target,np.diff(z[:,0]),np.diff(target[:,0]),
                             endpoint_alignment=False,backend=backend)
    source = SimpleNamespace(arrays=dict(latCell=rng.uniform(-1.,1.,ns),lonCell=rng.uniform(-3.,3.,ns)))
    dest = SimpleNamespace(arrays=dict(latCell=rng.uniform(-1.,1.,nt),lonCell=rng.uniform(-3.,3.,nt)))
    pools = {k:np.array(rng.normal(size=(levels,ns)),dtype=dtype,order=order) for k in ('a','b','u','v')}
    pools['a'][:,0] = -0.;pools['b'][::2,1] *= dtype(1e-30)
    pools['unset'] = pools['a'].copy();pools['unset'][0,0] = np.nan
    pools['category'] = np.zeros((levels,ns),dtype=np.int32)
    expected = {k:w.layer_intensive(pools[k]) for k in ('a','b')}
    expected['u'],expected['v'] = carry_horizontal_vector(pools['u'],pools['v'],w,source,dest,atmospheric=True)
    actual, stats = remap(w,pools,list(pools),[('u','v')],source,dest)
    assert set(actual) == set(expected)
    for k in expected:
        assert actual[k][0] is True
        assert actual[k][1].tobytes() == expected[k].tobytes()
    assert stats['fields'] == 4 and stats['channels'] == 5
    assert stats['batches'] == (5 if batch_mib == '0' else 1)
