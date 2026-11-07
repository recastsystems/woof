"""Real recovery kernels must preserve words when their dead inputs hold outputs."""
import numpy as np
import pytest
from woof.hex.cuda_driver import _CUDA_SOURCE

@pytest.fixture(scope='module')
def kernels():
    cp=pytest.importorskip('cupy')
    if not cp.cuda.runtime.getDeviceCount():pytest.skip('CUDA required')
    module=cp.RawModule(code=_CUDA_SOURCE,options=('--fmad=false','--ftz=true','--std=c++17'))
    return cp,module

@pytest.mark.gpu
@pytest.mark.parametrize('stage',[1,2,3])
@pytest.mark.parametrize('heating',[0,1])
def test_recovery_input_output_aliases_are_word_identical(kernels,stage,heating):
    cp,module=kernels
    rng=np.random.default_rng(842)
    nz,nc,ne=7,257,771
    def array(n,lo=.001,hi=.1):return cp.asarray(rng.uniform(lo,hi,n).astype(np.float32))
    n=nz*nc
    # The same source, launch and operands; only output addresses differ.
    inputs=[array(n),array(n),array(n),array(n),array(n),array(n,1,2),
            array(n,300,400),array(n,.9,1.1),array(n,1,2),array(n,.9,1.1),array(n)]
    for a in inputs[:4]:a[:4]=cp.asarray(np.array([0,1,0x80000000,0x007fffff],np.uint32).view(np.float32))
    def cells(alias):
        args=[a.copy() for a in inputs]
        out=[cp.empty(n,np.float32) for _ in range(7)]
        if alias:out[:2]=args[2:4]
        module.get_function('recover_cells_f32')(((n+127)//128,),(128,),
            (np.int32(nz),np.int32(nc),np.int32(stage),np.int32(1),
             np.float32(287),np.float32(1004.5),np.float32(100000),
             *args[:4],np.int32(heating),np.float32(18),*args[4:],*out))
        return out
    for expected,actual in zip(cells(False),cells(True)):
        np.testing.assert_array_equal(cp.asnumpy(expected).view(np.uint32),cp.asnumpy(actual).view(np.uint32))
    for name,count,extent,interface in [('recover_edges_f32',ne,nz*ne,False),('recover_interfaces_f32',nc,(nz+1)*nc,True)]:
        saved,increment,average=[array(extent,-10,10) for _ in range(3)]
        preserved=increment.copy()
        outs=[]
        for alias in (False,True):
            mean=average.copy();state=cp.empty_like(saved);flux=mean if alias else cp.empty_like(saved)
            module.get_function(name)(((extent+127)//128,),(128,),
                (np.int32(nz),np.int32(count),np.int32(4),saved,increment,mean,state,flux))
            outs.append((state,flux))
        for expected,actual in zip(*outs):
            np.testing.assert_array_equal(cp.asnumpy(expected).view(np.uint32),cp.asnumpy(actual).view(np.uint32))
        np.testing.assert_array_equal(cp.asnumpy(increment).view(np.uint32),cp.asnumpy(preserved).view(np.uint32))
