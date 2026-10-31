import os
from types import SimpleNamespace
import numpy as np
import pytest
from woof.hex.mpasphys.partition_v842 import _maps, _project_buffer


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
@pytest.mark.parametrize('order', ['C','F'])
def test_device_class_projection_transports_all_native_words_and_dummy_axes(order):
    from woof.hex.moving_pool_projection_cuda import prepare, project
    layout=SimpleNamespace(dimensions=dict(nCells=6,nEdges=5,nVertices=4))
    first=SimpleNamespace(cell_l2g=np.array([3,1,4]),n_owned_cells=2,
        edge_l2g=np.array([2,0]),n_owned_edges=1,vertex_l2g=np.array([1,3]),n_owned_vertices=1)
    second=SimpleNamespace(cell_l2g=np.array([0,2,5,1]),n_owned_cells=3,
        edge_l2g=np.array([4,1,3]),n_owned_edges=2,vertex_l2g=np.array([0,2,3]),n_owned_vertices=2)
    maps=[_maps(layout,p) for p in (first,second)]
    gpu=prepare(maps,layout)
    cases=[('real',['nVertLevels','nCells'],(3,7)),
           ('integer',['nCellsSolve','nEdgesAll'],(6,5)),('real',[],()),
           ('real',['nVertLevels','nCellsAll'],(3,6))]
    for kind,dims,shape in cases:
        count=int(np.prod(shape))
        bits=np.array((np.arange(count,dtype=np.uint32)+np.uint32(0x7f800001)).reshape(shape),order=order)
        bits.flat[0]=0x80000000
        value=bits.view(np.float32 if kind=='real' else np.int32)
        row=dict(pool='state',name='carried',kind=kind,dimensions=dims,shape=list(shape))
        expected=[_project_buffer(value,row,layout,m) for m in maps]
        actual=project(value,row,layout,maps,gpu)
        for a,b in zip(actual,expected):
            assert a.shape==b.shape and a.dtype==b.dtype and a.flags.f_contiguous
            assert a.tobytes(order='F')==b.tobytes(order='F')
        assert not np.shares_memory(actual[0],actual[1])
