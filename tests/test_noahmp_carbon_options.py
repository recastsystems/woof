"""Actual dynamic vegetation state must reach its next CUDA physics call."""
import pytest
from conftest import requires_gpu


@requires_gpu
@pytest.mark.parametrize("dveg", [2, 5, 6])
def test_carbon_pools_are_initialized_consumed_and_written(dveg):
    import cupy as cp
    from test_noahmp_runtime import _build
    from woof.core.dycore import step
    from woof.core.noahmp_carbon_slab import CARBON_STATE_2D
    from woof.io.wrfout import state_frame

    a, cfg, driver = _build(nx=6, ny=4, nz=10, dveg=dveg)
    b, _, other = _build(nx=6, ny=4, nz=10, dveg=dveg)
    # A stock cold start must produce real leaf/root and soil carbon pools.
    for name in ("lfmassxy", "rtmassxy", "stblcpxy", "fastcpxy"):
        assert bool(cp.any(driver.fields[name] > 0))
    before = driver.fields["lfmassxy"].copy()
    other.fields["lfmassxy"][:, :4] *= cp.float32(1.4)
    step(a, cfg)
    step(b, cfg)
    assert not bool(cp.array_equal(before, driver.fields["lfmassxy"]))
    assert not bool(cp.array_equal(driver.fields["lai"], other.fields["lai"]))
    assert not bool(cp.array_equal(driver.fields["lfmassxy"], other.fields["lfmassxy"]))
    frame = state_frame(a)
    for name in (*CARBON_STATE_2D, "neexy", "gppxy", "nppxy"):
        assert name[:-2].upper() in frame
        assert bool(cp.isfinite(driver.fields[name]).all())
