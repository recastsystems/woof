"""The PX fractional ice wrapper feeds the production RUC land step."""
import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("fractional_seaice", [0, 1])
def test_px_ruc_wrapper_and_post_land_blend(fractional_seaice):
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.physics_inventory import PX_SFCLAY_SEA, PX_SFCLAY_STATE
    from test_ruc_lsm_fused import _harness

    bench = _harness()
    bench.RunConfig = lambda **kw: RunConfig(**{
        **kw, "sf_sfclay_physics": 7, "bl_pbl_physics": 0,
        "fractional_seaice": fractional_seaice})
    state, cfg, driver, atmosphere, cold = bench.build(97, 61, 20, 9, "mixed", 11)
    threshold = 0.02 if fractional_seaice else 0.5
    assert driver.ruc_params.xice_threshold == threshold
    for step in (1, 2):
        state.elapsed_seconds = (step - 1) * cfg.dt
        bench.forcing(driver, step, 11, (61, 97), cold)
        driver._run_px_sfclay(atmosphere, cfg)
        census = bench.call(driver, atmosphere, step)
        assert census["land"] > 0 and census["sea_ice"] > 0
        for name in (*PX_SFCLAY_STATE,
                     *(f"{key}_sea" for key in PX_SFCLAY_SEA),
                     "t2", "q2", "tsk"):
            assert bool(cp.all(cp.isfinite(driver.fields[name]))), name
        assert bool(cp.any(driver.fields["ust"] > np.float32(0)))
