"""GFS 3/3 through the ordinary tree executor and both feedback modes."""
from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def test_gfs_pair_nests_keep_one_way_parent_words_and_apply_two_way_feedback(monkeypatch):
    import cupy as cp
    from tilestream import test_moving_nest as moving
    from tilestream import test_nest_executor as executor
    from woof.core.model import execute_experiment
    from woof.ensemble.state_sha import live_state_sha256
    from woof.wrf_exact import ENABLED as wrf_exact

    monkeypatch.setattr(executor, "PARENT_DT", 15.0)
    monkeypatch.setattr(executor, "I_PARENT_START", 20)
    monkeypatch.setattr(executor, "J_PARENT_START", 20)
    cfg = moving.parent_cfg(nx=64, ny=64, nz=24, dt=15.0,
                            run_seconds=60.0, bl_pbl_physics=3,
                            sf_sfclay_physics=3, cu_physics=0)
    child_cfg = moving.child_cfg(cfg, nx=27, ny=27)

    def run(nested, feedback=0, *, match_force=False):
        boundaries = moving.domain_boundaries(cfg, seconds=21600)
        state, geography = moving.build_parent(cfg, boundaries=boundaries)
        exp = replace(executor.experiment(
            cfg, child_cfg if nested else None, steps=4), feedback=feedback)
        model = executor.assemble(exp, state, geography.grid, feedback=feedback)
        if feedback:
            executor.initial_feedback(model)
        force_calls = []
        steppers = None
        if match_force:
            from woof.core.dycore import step
            from woof.core.nest_round_trip import couple_round_trip

            def matched_step(state, run_cfg, **options):
                result = step(state, run_cfg, **options)
                # med_force_domain couples and uncouples the parent even
                # with feedback=0 (mediation_force_domain.F:111-206).
                # Match those strict FP32 operations in the solo control.
                couple_round_trip(state, run_cfg)
                force_calls.append(1)
                return result

            steppers = {1: matched_step}
        execute_experiment(model, validate_state=True,
                           pool_trim_per_period=False, steppers=steppers)
        if match_force:
            assert len(force_calls) == model.root.clock.step_count == 4
        if nested:
            assert model.node(2).coupler.force_count == 4
        cp.cuda.get_current_stream().synchronize()
        result = live_state_sha256(model.root.state)
        for node in model.walk_parent_first():
            driver = node.state.physics
            assert node.cfg.run.bl_pbl_physics == 3
            assert node.cfg.run.sf_sfclay_physics == 3
            assert node.clock.step_count >= 4
            for name in ("pblh", "ust", "gz1oz0", "psim", "psih"):
                assert np.isfinite(cp.asnumpy(driver.fields[name])).all(), name
            assert np.any(cp.asnumpy(driver.fields["pblh"]) > 0)
        return result

    unforced = run(False)
    alone = run(False, match_force=True) if wrf_exact else unforced
    one_way = run(True, 0)
    two_way = run(True, 1)
    assert one_way == alone, "one-way child changed its GFS parent's words"
    assert two_way != one_way, "two-way GFS feedback was inactive"
    if wrf_exact:
        assert alone != unforced, "strict force round-trip control was inactive"
