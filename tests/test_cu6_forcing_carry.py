"""WRF set_tend keeps the final coupled transport for the next CU6 call."""
from types import SimpleNamespace as NS
import numpy as np


def test_cu6_retains_last_transport_without_early_mass_division(monkeypatch):
    from woof.core import moist
    monkeypatch.setattr(moist, 'WRF_EXACT', True)
    state = NS(rqvften=np.zeros((2, 2, 2), np.float32), has_msf=True,
        msft=np.full((2, 2), np.float32(1.03)),
        c1h=np.full(2, np.float32(.75)), c2h=np.full(2, np.float32(25000.)))
    cfg = NS(cu_physics=6)
    initial_mass=np.full((2,2), np.float32(90000.))
    for stage in (1, 2, 3):
        native_transport=np.full((2,2,2), np.float32(stage*1.25))
        moist._capture_advective_qv_forcing(state, native_transport, initial_mass, cfg=cfg)
    expected=native_transport*state.msft[None]
    np.testing.assert_array_equal(state.rqvften.view('u4'), expected.view('u4'))
    # At the next physics call the actual carried MUTS, not initial_mass,
    # supplies phy_prep_part2's divisor. The adapter is checked by the
    # native step-2 captured qvften replay, not an invented forecast rate.
    assert np.all(state.rqvften != expected / initial_mass)
