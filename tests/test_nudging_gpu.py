"""Run explicitly inside a GPU pool lease: WOOF_DOWNSCALE_GPU_TEST=1 pytest ..."""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("WOOF_DOWNSCALE_GPU_TEST") != "1",
                                reason="requires an explicit GPU pool lease")


def test_captured_four_field_operator():
    import cupy as cp
    from woof.spectral_nudging import NudgingConfig, build_nudger
    from woof.spectral_nudging.gpu import FieldBinding
    config = NudgingConfig(enabled=True, cutoff_x=1, cutoff_y=1,
                           relaxation_seconds=100., minimum_height_m=0.,
                           ramp_depth_m=1000., boundary_cells=0)
    bindings, initial = {}, {}
    for i, name in enumerate(("u", "v", "temperature", "geopotential")):
        shape = (3 + (i == 3), 12 + (i == 1), 16 + (i == 0))
        x = np.arange(shape[-1], dtype=np.float32)[None, None, :]
        wave = np.cos(2 * np.pi * x / shape[-1]) + .5 * np.cos(8 * np.pi * x / shape[-1])
        value = np.broadcast_to(wave, shape).copy()
        height = np.full(shape, 2000., dtype=np.float32)
        height[0] = 0.
        initial[name] = value.copy()
        bindings[name] = FieldBinding(cp.asarray(value), cp.zeros(shape, dtype=cp.float32),
                                     cp.full(shape, 2., dtype=cp.float32), cp.asarray(height),
                                     cp.zeros(shape[-2:], dtype=cp.float32))
    plan = build_nudger(config, bindings, elapsed_seconds=0., dt_seconds=10.,
                        interval_start=0., interval_end=20.)
    for name, binding in bindings.items():
        np.testing.assert_array_equal(binding.values.get(), initial[name])
    plan.enqueue()
    plan.enqueue()
    cp.cuda.get_current_stream().synchronize()
    plan.check_at_seam()
    for name, binding in bindings.items():
        before = initial[name]
        n = before.shape[-1]
        low = np.cos(2 * np.pi * np.arange(n, dtype=np.float32) / n)
        # midpoint analyses: 0.5 then 1.5; two Euler steps of fraction .1.
        expected = before.copy()
        expected[1:] += (-.19 * low + .195)[None, None, :]
        np.testing.assert_allclose(binding.values.get(), expected, atol=2.e-6, rtol=2.e-6)
        np.testing.assert_array_equal(binding.values.get()[0].view(np.uint32), before[0].view(np.uint32))
    snapshots = {name: b.values.get() for name, b in bindings.items()}
    plan.enqueue()  # outside interval: all four fields unchanged, error recorded
    cp.cuda.get_current_stream().synchronize()
    with pytest.raises(RuntimeError, match="escaped"):
        plan.check_at_seam()
    for name, binding in bindings.items():
        np.testing.assert_array_equal(binding.values.get(), snapshots[name])
