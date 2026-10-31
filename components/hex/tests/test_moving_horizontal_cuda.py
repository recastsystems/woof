"""Moving land fields retain every word of the sparse reference."""
import os

import numpy as np
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA')!='1',reason='CUDA lease required')


@pytest.mark.parametrize('shape',[(43,),(4,43),(3,4,43)])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('fmad',['false','true'])
def test_ordered_sparse_words(shape,dtype,fmad,monkeypatch):
    from types import SimpleNamespace
    from scipy.sparse import csr_matrix
    from woof.hex import moving_horizontal_cuda as cuda
    from woof.hex.moving_remap import ConservativeMap
    rng=np.random.default_rng(27)
    # Unsorted rows, duplicate donors and an empty row all keep CSR's order.
    counts=np.array([5,0,1,7,2,4,3,12,1,6,3])
    ptr=np.concatenate(([0],np.cumsum(counts)))
    data=rng.uniform(.1,10.,ptr[-1])
    ids=rng.integers(0,43,ptr[-1])
    matrix=csr_matrix((data,ids,ptr),shape=(len(counts),43))
    w=SimpleNamespace(source_area=np.ones(43),target_area=rng.uniform(.1,2.,len(counts)),overlap=matrix)
    values=(rng.normal(size=shape)*100.).astype(dtype)
    monkeypatch.setattr(cuda,'OPTIONS',('--fmad='+fmad,))
    cuda._kernel.cache_clear()
    expected=ConservativeMap.intensive(w,values)
    actual=cuda.intensive(w,values)
    np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))
    assert cuda.intensive(w,values).tobytes()==expected.tobytes()
    with pytest.raises(ValueError,match='finite-value'):
        cuda.intensive(w,np.full(shape,np.nan))
