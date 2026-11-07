"""EEPS carriers through resident nest edges and member-local execution."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_eeps_member_physics_keeps_independent_carriers():
    from test_ensemble_physics_execution_gpu import (
        test_member_local_real_driver_and_microphysics_match_each_independent_call)
    test_member_local_real_driver_and_microphysics_match_each_independent_call(
        2, 8, 16, 1, 0)


def test_eeps_eta_retains_native_surface_owned_zero_tke():
    import cupy as cp
    from test_restart import _physics_state
    from woof.core.dycore import run_steps
    state, cfg, driver = _physics_state(cp, bl_pbl_physics=16, sf_sfclay_physics=2)
    assert not bool(cp.any(driver.fields["tke_myj"]))
    run_steps(state, cfg, 3)
    assert not bool(cp.any(driver.fields["tke_myj"]))
    assert float(cp.max(state.pek_adv)) > 0.
    assert bool(cp.all(cp.isfinite(state.thp)))


@pytest.mark.parametrize("feedback", [0, 1])
def test_live_eeps_pair_crosses_nest_force_and_feedback(feedback):
    import cupy as cp
    from test_eeps_driver_seam import _driver_cfg, _driver_state, _skin
    from test_nest_coupler import _clock
    from woof.core.dycore import run_steps
    from woof.core.model import FeedbackScratch
    from woof.core.nest import NestCoupler
    from woof.core.physics import initialize_physics, DECLARED_CONSTANT_GLW_WM2

    pcfg = replace(_driver_cfg(nx=37, ny=37, nz=6), grid_id=1, specified=True)
    ccfg = replace(_driver_cfg(nx=27, ny=27, nz=6, dt=2.),
                   grid_id=2, nested=True, specified=False)
    states = []
    for cfg in (pcfg, ccfg):
        state = _driver_state(cfg)
        initialize_physics(state, cfg, landmask=1., tsk=_skin(cfg),
                           glw=DECLARED_CONSTANT_GLW_WM2, swdown=600.)
        # Warm both closures before this test attaches the nested edge.
        run_steps(state, replace(cfg, nested=False, specified=False), 1)
        states.append(state)
    parent = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1, parent_id=0, parent_grid_ratio=1,
                            i_parent_start=1, j_parent_start=1, run=pcfg),
        state=states[0], clock=_clock(1, 0, step_ticks=6, dt=6., advanced=True))
    child = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=2, parent_id=1, parent_grid_ratio=3,
                            i_parent_start=5, j_parent_start=5, run=ccfg),
        state=states[1], parent=parent,
        clock=_clock(2, 1, step_ticks=2, dt=2.))
    child.clock.prepare_step()
    coupler = NestCoupler(child, feedback=feedback, smooth_option=0)
    coupler.force(child)
    assert {"pek_adv", "pep_adv"} <= coupler._last_tables.keys()
    for name in ("pek_adv", "pep_adv"):
        for endpoint in coupler._last_tables[name].values():
            assert all(bool(cp.all(cp.isfinite(array))) for array in endpoint)
    run_steps(child.state, ccfg, 3)
    assert bool(cp.all(cp.isfinite(child.state.pek_adv)))
    before = {name: getattr(parent.state, name).get().tobytes()
              for name in ("pek_adv", "pep_adv")}
    # Distinct live child words must reach the parent's restriction window.
    child.state.pek_adv.fill(np.float32(.25))
    child.state.pep_adv.fill(np.float32(.005))
    child.clock.ticks = parent.clock.ticks
    coupler.feedback_prepare(child, FeedbackScratch())
    coupler.feedback_commit(child)
    coupler.feedback_finalize(child)
    for name in before:
        values = getattr(parent.state, name).get()
        assert np.isfinite(values).all()
        assert (values.tobytes() != before[name]) is bool(feedback), name
