"""Streamed ozone must preserve every interpolation word and calendar edge."""
import numpy as np
import pytest

from woof.hex.mpasphys import radshared_v842 as ozone


def test_streamed_kernel_keeps_calendar_arithmetic():
    original=ozone.CUDA_SOURCE.split('void ozone_time(',1)[1].split('\n}',1)[0]
    streamed=ozone.CUDA_SOURCE.split('void ozone_time_pair(',1)[1].split('\n}',1)[0]
    assert original.split('    const int dates',1)[1].split('    out[idx]',1)[0] == streamed.split('    const int dates',1)[1].split('    out[idx]',1)[0]


@pytest.mark.gpu
def test_two_month_interpolation_is_word_identical():
    cp=pytest.importorskip('cupy')
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip('CUDA required for interpolation word comparison')
    rng=np.random.default_rng(841)
    host=rng.uniform(0,1e-5,(12,59,17)).astype(np.float32)
    host[0,0,:4]=np.array([0,1,0x80000000,0x007fffff],dtype=np.uint32).view(np.float32)
    monthly=cp.asarray(host.transpose(2,1,0).copy())
    # Calendar boundaries, fractional rounding, cyclic years and leap-day input.
    dates=[0,15,44,74,104,135,165,196,227,257,288,318,349,364,365,730]
    for julian in [np.float32(d+delta) for d in dates for delta in (-.0001,0,.25,.9999)]:
        months=ozone._ozone_month_pair(julian)
        pair=cp.asarray(host[list(months)].copy())
        old=ozone.o3climatology_from_MPAS(julian,monthly)
        new=cp.empty_like(old)
        ozone._launch('ozone_time_pair',new.size,(np.int32(new.size),np.int32(17),julian,pair,new))
        np.testing.assert_array_equal(cp.asnumpy(new).view(np.uint32),cp.asnumpy(old).view(np.uint32),err_msg=str(julian))
