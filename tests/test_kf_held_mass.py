"""Held KF rates follow native calculate_phy_tend and phy_prep_part2."""
from types import SimpleNamespace

import numpy as np

from woof.core.physics import PhysicsDriver


def test_held_rate_keeps_both_mass_rounds_without_history_fields():
    shape = (2, 1, 1)
    start = np.array([[7.3]], dtype="f4")
    end = np.array([[7.3003]], dtype="f4")
    state = SimpleNamespace(c1h=np.array([1., .5], dtype="f4"),
                            c2h=np.array([2., 3.], dtype="f4"), total_mu=lambda: start)
    driver = PhysicsDriver.__new__(PhysicsDriver)
    rate = np.array([.00012349, -.0000003541], dtype="f4").reshape(shape)
    driver.fields = {}
    driver.cu_rates = {"rthcuten": rate.copy(), "rqvcuten": rate.copy()}
    cfg = SimpleNamespace(cu_physics=1, bl_pbl_physics=0)
    weight = state.c1h[:, None, None] * start + state.c2h[:, None, None]
    coupled = rate * weight
    expected = coupled / (state.c1h[:, None, None] * end + state.c2h[:, None, None])
    for _ in range(2):
        driver.couple_cumulus_history(state, cfg)
        for actual in driver.cu_rates.values():
            np.testing.assert_array_equal(actual.view("u4"), coupled.view("u4"))
        driver.finish_history_mass(state, cfg, muts=end)
        for actual in driver.cu_rates.values():
            np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))
        coupled = expected * weight
        expected = coupled / (state.c1h[:, None, None] * end + state.c2h[:, None, None])
