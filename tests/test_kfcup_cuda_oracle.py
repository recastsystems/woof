"""Device KF-CuP checks, run only through the sprint card queue."""
import os
import numpy as np
import pytest
from woof.verify.kfcup_oracle import (
    configs, load_capture, pack_column_inputs, expected_outputs,
    unpack_outputs, differing_words, _rows, word,
)
pytestmark = pytest.mark.gpu


def assert_fields(got, expected):
    strict=os.environ.get('GPUWM_WRF_EXACT')=='1'
    differences=differing_words(got,expected)
    print('DEVICE_DIFFERING_WORDS', {k:v for k,v in differences.items() if v})
    if strict:
        assert not any(differences.values()),differences
    else:
        # Record the observed default error separately from strict acceptance.
        for key,exp in expected.items():
            if exp.dtype==np.float32:
                np.testing.assert_allclose(got[key],exp,rtol=5e-4,atol=1e-8,
                                           err_msg=key)
            else:
                np.testing.assert_array_equal(got[key],exp,err_msg=key)


@pytest.mark.parametrize('cfg',configs(),ids=lambda c:c.name)
def test_cuda_kfcup_columns(cfg):
    import cupy as cp
    from woof.core.kernels import get_kernel
    from woof.core.kf import load_kf_table, _device_table
    cap=load_capture(cfg.index)
    buffers={k:cp.asarray(v) for k,v in pack_column_inputs(cap).items()}
    tab=load_kf_table()
    blocks=(cap.ncol+31)//32
    per_thread=148*(cap.nz+1)+cfg.numbins*cfg.numbins+2*cfg.numbins
    ws=cp.empty(blocks*32*per_thread,dtype=cp.float32)
    status=cp.zeros(1,dtype=cp.int32)
    get_kernel('kfcup','kfcup_column')((blocks,),(32,),(
        *(buffers[k] for k in ('lev_in','sca_in','kpbl','held','iheld','lev_out')),
        *_device_table(),
        np.float32(tab.pressure_top),np.float32(tab.pressure_reciprocal),
        np.float32(tab.thetae_reciprocal),np.float32(cfg.dt),np.float32(cfg.dx),
        np.int32(cfg.stepcu),np.int32(cfg.numbins),np.float32(cfg.thbinsize),
        np.float32(cfg.rbinsize),np.float32(cfg.mindeepfreq),np.float32(cfg.minshallowfreq),
        np.int32(cfg.flags),ws,status,np.int32(cap.nz),np.int32(cap.ncol),np.int32(0)))
    cp.cuda.get_current_stream().synchronize()
    assert int(status.get()[0])==0
    got=unpack_outputs({k:cp.asnumpy(v) for k,v in buffers.items()})
    assert_fields(got,expected_outputs(cap))


@pytest.mark.parametrize('cfg',configs(),ids=lambda c:c.name)
def test_cuda_kfcup_history(cfg):
    import cupy as cp
    from woof.core.kernels import get_kernel
    cap=load_capture(cfg.index)
    avg=cp.asarray(cap.level_in['w0avg'][:cap.nz].copy())
    w=cp.asarray(cap.level_in['w'].copy())
    get_kernel('kfcup','kfcup_trigger_history')(((avg.size+255)//256,),(256,),
        (avg,w,np.float32(2*cfg.stepcu),np.int32(cap.ncol),np.int32(cap.nz)))
    assert_fields({'w0avg':avg.get()},{'w0avg':cap.level_out['w0avg']})


@pytest.mark.parametrize('icloud,forced',[(1,False),(1,True),(3,False),(3,True)])
def test_cuda_kfcup_radiation(icloud,forced):
    import cupy as cp
    from woof.core.kernels import get_kernel
    rows=sorted((r for r in _rows('kfcup-rad-levels.csv')
                 if int(r['icloud'])==icloud and (r['shallowcu_forced_ra']=='T')==forced),
                 key=lambda r:(int(r['k']),int(r['case'])))
    ncol=max(int(r['case']) for r in rows)
    nz=max(int(r['k']) for r in rows)
    v={key:cp.asarray(np.array([word(r[key]) for r in rows],np.float32))
       for key in ('cldfra1','qc_in','cldfra_cup_in')}
    flags=cp.asarray(np.array([int(r['cldfra1_flag']) for r in rows],np.int32))
    held=[cp.asarray(np.array([word(r[key]) for r in rows[:ncol]],np.float32))
          for key in ('shall','cubot','cutop')]
    get_kernel('kfcup','kfcup_radiation_merge')(((ncol*nz+255)//256,),(256,),
        (v['cldfra1'],v['qc_in'],v['cldfra_cup_in'],flags,*held,
         np.int32(icloud),np.int32(forced),np.int32(ncol),np.int32(nz)))
    got={k:v[s].get() for k,s in [('cldfra','cldfra1'),('qc','qc_in'),('cldfra_cup','cldfra_cup_in')]}
    expected={k:np.array([word(r[k]) for r in rows],np.float32)
              for k in ('cldfra','qc','cldfra_cup')}
    assert_fields(got,expected)


def test_cuda_kfcup_default_route_preserves_subnormal_rounding():
    import cupy as cp
    from woof.core.kernels import get_kernel
    avg=cp.asarray(np.array([[3]],np.uint32).view(np.float32))
    w=cp.zeros((2,1),cp.float32)
    get_kernel('kfcup','kfcup_trigger_history')((1,),(32,),
        (avg,w,np.float32(2),np.int32(1),np.int32(1)))
    np.testing.assert_array_equal(avg.get().view(np.uint32),np.array([[2]],np.uint32))
