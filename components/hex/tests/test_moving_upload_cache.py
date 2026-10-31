import os
import numpy as np
import pytest
from woof.hex.moving_upload_cache import UploadCache


def test_prediction_cache_can_only_be_claimed_once():
    cache = UploadCache()
    assert cache.claim() is cache
    assert cache.claim() is None


def test_fresh_noncontiguous_input_keeps_dtype_without_cuda():
    source=np.arange(60,dtype=np.float64).reshape(6,10)[:,::2]
    actual=UploadCache().upload(np,'mesh','field',source)
    assert actual.dtype==source.dtype and actual.flags.c_contiguous
    assert actual.tobytes()==source.tobytes()


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
@pytest.mark.parametrize('dtype', ['u4','u8'])
def test_incremental_upload_preserves_changed_bits_strides_and_shape(dtype):
    import cupy as cp
    cache = UploadCache()
    bits = np.arange(1024, dtype=dtype).reshape(32,32)
    value = np.array(bits.view('f'+dtype[-1]), order='F')
    first = cache.upload(cp, 'mesh', 'value', value)
    cp.cuda.get_current_stream().synchronize()
    # Sparse path includes signed zero, subnormal and a noncanonical NaN.
    changed = bits.copy()
    changed.flat[0] = 1 << (8*changed.dtype.itemsize-1)
    changed.flat[13] = 1
    changed.flat[91] = 0x7fc01234 if dtype == 'u4' else 0x7ff8000000001234
    second = cache.upload(cp, 'mesh', 'value', changed.view(value.dtype))
    assert second is first
    np.testing.assert_array_equal(cp.asnumpy(second).view(dtype), changed)
    # Dense and arbitrary-stride paths refresh every word in logical order.
    dense = changed[::-1,::-1]
    assert cache.upload(cp, 'mesh', 'value', dense.view(value.dtype)) is first
    np.testing.assert_array_equal(cp.asnumpy(first).view(dtype), dense)
    other = cache.upload(cp, 'state', 'value', value)
    assert other is not first
    fresh = cache.upload(cp, 'mesh', 'value', value[:16])
    assert fresh.shape == (16,32) and fresh is not first
    assert cache.receipt()['reused_bytes'] == 2*value.nbytes
    assert cache.receipt()['changed_bytes'] >= 3*value.dtype.itemsize


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
def test_admitted_atmosphere_matches_fresh_upload_after_prediction_changes():
    from dataclasses import fields
    import cupy as cp
    from test_device_field_upload import atmosphere_fixture
    from woof.hex.cuda_backend.containers import DeviceAtmosphere
    sources = atmosphere_fixture()
    cache = UploadCache()
    predicted = DeviceAtmosphere.from_host(*sources, upload_cache=cache)
    sources[0].arrays['areaCell'][5] = 4.25
    sources[0].arrays['cellsOnEdge'][5,0] = 1
    for owner, name in [(sources[1],'rho'),(sources[2],'zz'),
                        (sources[3],'exner_base'),(sources[4],'exner'),(sources[5],'zb_cell')]:
        getattr(owner,name).flat[4] = -0.
    admitted = DeviceAtmosphere.from_host(*sources, upload_cache=cache.claim())
    fresh = DeviceAtmosphere.from_host(*sources)
    for name in ('mesh','state','vertical','reference','saved','terrain'):
        actual, expected = getattr(admitted,name), getattr(fresh,name)
        for descriptor in fields(expected):
            field = descriptor.name
            value = getattr(expected,field)
            if isinstance(value,cp.ndarray):
                other = getattr(actual,field)
                assert other.shape == value.shape and other.dtype == value.dtype
                assert cp.asnumpy(other).tobytes() == cp.asnumpy(value).tobytes(), (name,field)
    assert admitted.state.rho is not predicted.state.rho
    assert admitted.reference.exner_base is predicted.reference.exner_base
    assert not any(key[0] in ('DevicePrognosticState.from_host','DeviceSavedDiagnostics.from_host')
                   for key in cache.arrays)
    assert not cache.arrays
    assert cache.receipt()['reused_bytes'] > 0
