"""An unsaturated, cloud-free column must not acquire microphysics heat."""
import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("moist_theta", [0, 1])
def test_clear_column_has_no_repeated_adapter_heating(moist_theta):
    import cupy as cp
    from woof.core import constants as c, microphysics
    from woof.physics_registry import consumer_row_for_selector
    from test_mp_spec_zone_ring import _square_moist_state

    state, cfg = _square_moist_state(ny=4, nx=5, nz=12, mp=55,
                                    dt=12, specified=False)
    row = consumer_row_for_selector("microphysics", "native_column", 55)
    for name in row["transport"]:
        getattr(state, name).fill(0)
    state.qv.fill(np.float32(0.001))
    state.p.fill(np.float32(90000))
    state.use_theta_m = moist_theta
    full = cp.full(state.p.shape, np.float32(320))
    if moist_theta:
        full *= np.float32(1) + np.float32(c.RVOVRD) * state.qv
    state.thp[...] = full - state.thb[:, None, None]
    before = state.thp.copy()
    for _ in range(20):
        microphysics.apply(state, cfg, 12, refl_10cm_due=False)
    assert float(cp.max(cp.abs(state.thp - before))) < 0.001
    assert float(cp.max(cp.abs(state.h_diabatic))) < 1e-5
    assert bool(cp.all(state.qv == np.float32(0.001)))
