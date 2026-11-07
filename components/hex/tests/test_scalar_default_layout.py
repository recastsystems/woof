"""Layout cache guards for the shared edge launch ABI."""
import gc
from types import SimpleNamespace
import numpy as np
import pytest
from woof.hex import cuda_transport_v841 as transport


def test_stencil_extent_stream_and_lifetime(monkeypatch):
    stream = SimpleNamespace(ptr=1)
    thread = SimpleNamespace(id=1)
    monkeypatch.setattr(transport, '_thread_id', lambda: thread.id)
    cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda:SimpleNamespace(id=0),
         get_current_stream=lambda:stream),ascontiguousarray=np.ascontiguousarray,zeros=np.zeros)
    monkeypatch.setattr(transport,'_cupy',lambda:cp)
    monkeypatch.setattr(transport,'_EDGE_STENCILS',{})
    arrays=(np.arange(12,dtype=np.int32).reshape(3,4),np.arange(12,dtype=np.float32).reshape(3,4))
    result=transport._stencil_layout(arrays,4)
    for a,b in zip(arrays,result):
        np.testing.assert_array_equal(a.T,b[:,:-1]);assert np.all(b[:,-1]==0)
    assert transport._stencil_layout(arrays,4) is result
    exact=transport._stencil_layout(arrays,3)
    np.testing.assert_array_equal(exact[0],arrays[0].T)
    stream.ptr=2
    stream_result=transport._stencil_layout(arrays,4)
    assert stream_result is not result
    thread.id=2
    assert transport._stencil_layout(arrays,4) is not stream_result
    with pytest.raises(ValueError,match='extent'):transport._stencil_layout(arrays,5)
    with pytest.raises(ValueError,match='extent'):transport._stencil_layout(arrays,2)
    del a,arrays
    gc.collect()
    assert not transport._EDGE_STENCILS

    singleton=(np.arange(4,dtype=np.float32).reshape(1,4),)
    singleton_result=transport._stencil_layout(singleton,1)
    assert not np.shares_memory(singleton_result[0],singleton[0])
    del singleton
    gc.collect()
    assert not transport._EDGE_STENCILS
