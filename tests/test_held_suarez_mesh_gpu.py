"""Resident mesh ownership and continuation, independent of oracle arithmetic."""
import pytest
from conftest import requires_gpu


@requires_gpu
@pytest.mark.gpu
def test_private_streams_restart_and_unsolved_edges_without_stage_allocation():
    import cupy as cp
    import numpy as np
    from woof.core.held_suarez_mesh import HeldSuarezMeshMomentum
    topology=cp.asarray([[0,1],[1,2],[99,99]],dtype=cp.int32)
    base=cp.asarray([[100000.]*3,[50000.]*3],dtype=cp.float32)
    pressure=cp.zeros_like(base)
    wind=cp.asarray([[2.,-3.,9.],[2.,-3.,9.]],dtype=cp.float32)
    def binding():
        operator=HeldSuarezMeshMomentum()
        operator.bind(topology,base,solved_edges=2)
        return operator
    operators=[binding(),binding()]
    states=[cp.full_like(wind,1.),cp.full_like(wind,-1.)]
    references=[v.copy() for v in states]
    streams=[cp.cuda.Stream(non_blocking=True),cp.cuda.Stream(non_blocking=True)]
    cp.cuda.get_current_stream().synchronize()
    # Warm launches before measuring the persistent device allocation.
    for op,value in zip(operators,references):op.apply(value,wind,pressure)
    for value,initial in zip(references,(1.,-1.)):value.fill(initial)
    cp.cuda.get_current_stream().synchronize()
    used=cp.get_default_memory_pool().used_bytes()
    for _ in range(5):
        for stream,op,value in zip(streams,operators,states):
            with stream:op.apply(value,wind,pressure)
    for stream in streams:stream.synchronize()
    assert cp.get_default_memory_pool().used_bytes()==used
    # A restored member rebinds its immutable topology and base pressure.
    continued=[value.copy() for value in states]
    fresh=[binding(),binding()]
    for _ in range(5):
        for op,value in zip(fresh,continued):op.apply(value,wind,pressure)
    for _ in range(10):
        for op,value in zip(operators,references):op.apply(value,wind,pressure)
    for original,actual,expected in zip((1.,-1.),continued,references):
        np.testing.assert_array_equal(cp.asnumpy(actual).view('u4'),cp.asnumpy(expected).view('u4'))
        np.testing.assert_array_equal(cp.asnumpy(actual[:,2]),original)
        assert float(actual[0,0])<original
        assert float(actual[0,1])>original
