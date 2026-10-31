"""Word preservation for the large CopyPlan transpose and mixed fallback rows."""
import numpy as np
import pytest


def test_large_transpose_tail_tiles_and_mixed_rows():
    cp = pytest.importorskip('cupy')
    from woof.hex.mpasphys.rrfs.glue_plan import CopyPlan
    # Exercise incomplete tiles in both dimensions and preserve NaN payloads,
    # signed zero and arbitrary words without floating-point interpretation.
    host = np.arange(32769 * 33, dtype=np.uint32).reshape(32769, 33)
    host.flat[:4] = [0x80000000, 0x7fc01234, 0x7f800001, 0xffffffff]
    source = cp.array(host, order='F')
    backing = cp.full((32769, 35), 0x12345678, dtype=cp.uint32)
    target = backing[:, 1:34]
    broadcast = cp.empty((3, 7), cp.uint32)
    values = cp.arange(7, dtype=cp.uint32)
    filled = cp.empty((5, 9), cp.uint32)
    plan = CopyPlan('large transpose, padded destination, mixed fallback')
    plan.copy(target, source)
    plan.copy(broadcast, values)
    plan.fill(filled, np.uint32(0x87654321))
    plan.build()
    assert plan._tiled
    plan()
    np.testing.assert_array_equal(cp.asnumpy(target), host)
    np.testing.assert_array_equal(cp.asnumpy(backing[:, [0, 34]]), 0x12345678)
    np.testing.assert_array_equal(cp.asnumpy(broadcast), np.broadcast_to(np.arange(7), (3, 7)))
    np.testing.assert_array_equal(cp.asnumpy(filled), 0x87654321)


def test_small_transpose_retains_original_kernel():
    cp = pytest.importorskip('cupy')
    from woof.hex.mpasphys.rrfs.glue_plan import CopyPlan
    source = cp.arange(59 * 2562, dtype=cp.uint32).reshape(59, 2562).T
    target = cp.empty(source.shape, cp.uint32)
    plan = CopyPlan('x1.2562')
    plan.copy(target, source)
    plan.build()
    assert not plan._tiled
    plan()
    np.testing.assert_array_equal(cp.asnumpy(target), cp.asnumpy(source))
