"""Run PX on both domains through the production nest executor."""
from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("feedback", [0, 1])
@pytest.mark.parametrize("pbl,sfclay", [(9, 7), (7, 7), (7, 1), (7, 91)])
def test_px_surface_layer_advances_on_both_sides_of_nest(feedback, pbl, sfclay, monkeypatch):
    import cupy as cp
    from woof.core.model import execute_experiment
    from woof.core.physics_inventory import PX_SFCLAY_STATE
    from woof.io.history_layout import live_state_history_fields
    from tilestream import test_nest_executor as executor
    from tilestream.test_moving_nest import (
        build_parent, child_cfg, domain_boundaries, parent_cfg)

    monkeypatch.setattr(executor, "I_PARENT_START", 41)
    monkeypatch.setattr(executor, "J_PARENT_START", 41)
    monkeypatch.setattr(executor, "PARENT_DT", 6.0)
    cfg = parent_cfg(nx=96, ny=96, nz=24, dt=6.0, mp_physics=6,
                     bl_pbl_physics=pbl, sf_sfclay_physics=sfclay,
                     sf_surface_physics=2, cu_physics=0,
                     ra_lw_physics=90, ra_sw_physics=90)
    boundaries = domain_boundaries(cfg)
    parent, geography = build_parent(cfg, boundaries=boundaries)
    exp = replace(executor.experiment(
        cfg, child_cfg(cfg, nx=48, ny=48), steps=2), feedback=feedback)
    model = executor.assemble(exp, parent, geography.grid, feedback=feedback)
    if feedback:
        executor.initial_feedback(model)
    result = execute_experiment(model, validate_state=True,
                                pool_trim_per_period=False)
    assert result.steps > 0 and result.forces > 0
    coupler = model.node(2).coupler
    assert coupler.force_count > 0
    assert (coupler.feedback_count > 0) == bool(feedback)
    for node in (model.root, model.node(2)):
        physics = node.state.physics
        if sfclay == 7:
            assert physics.scheme_dispatch["sf_sfclay_physics"] == "_run_px_sfclay"
        if pbl == 7:
            assert physics.scheme_dispatch["bl_pbl_physics"] == "_run_acm2"
            assert physics.call_counts["ysu"] >= 2
            for key in ("pblh", "kpbl", "rmol", "regime", "exch_h", "exch_m"):
                assert bool(cp.all(cp.isfinite(physics.fields[key]))), (node.grid_id, key)
        assert physics.call_counts["sfclay"] >= 2
        history = live_state_history_fields(node.state)
        for key in PX_SFCLAY_STATE if sfclay == 7 else ():
            assert key in physics.fields
            assert bool(cp.all(cp.isfinite(physics.fields[key]))), (node.grid_id, key)
            assert key.upper() in history or key.upper() in {
                "HFX", "QFX", "LH", "PSIM", "PSIH", "UST", "U10", "V10"}
        assert bool(cp.any(physics.fields["ust"] > np.float32(0)))
