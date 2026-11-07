"""Exercise P3 moment ownership across real nest and ensemble transactions."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu


@requires_gpu
@pytest.mark.parametrize("feedback", [0, 1])
def test_prognostic_nc_nest_forcing_and_feedback(feedback):
    import cupy as cp
    from test_feedback import _domain, _seed_state
    from test_nest_coupler import _clock
    from woof.core.model import FeedbackScratch
    from woof.core.nest import NestCoupler
    from woof.core.state import DomainState

    parent_cfg = _domain(1, 0, nx=14, ny=14)
    child_cfg = _domain(2, 1, nx=12, ny=12, ratio=4)
    parent_cfg = replace(parent_cfg, run=replace(parent_cfg.run, mp_physics=51))
    child_cfg = replace(child_cfg, run=replace(child_cfg.run, mp_physics=51))
    parent_state, child_state = DomainState(parent_cfg.run), DomainState(child_cfg.run)
    _seed_state(parent_state, child=False)
    _seed_state(child_state, child=True)
    for index, name in enumerate(("nc",), start=1):
        getattr(parent_state, name).fill(index * 16)
        getattr(child_state, name).fill(index * 32)
        pattern = cp.arange(1, 17, dtype=cp.float32).reshape(4, 4) * index
        getattr(child_state, name)[:, 4:8, 4:8] = pattern
    parent_clock = _clock(1, 0, step_ticks=4, dt=30., advanced=True)
    child_clock = _clock(2, 1, step_ticks=1, dt=7.5)
    parent = SimpleNamespace(cfg=parent_cfg, state=parent_state, clock=parent_clock)
    child = SimpleNamespace(cfg=child_cfg, state=child_state, clock=child_clock, parent=parent)
    coupler = NestCoupler(child, feedback=feedback)
    coupler.force(child)
    assert child_state.lateral_boundaries is not None
    assert coupler.force_count == 1
    # Reach the synchronization point, then exercise the production transaction.
    for _ in range(4):
        child_clock.advance()
    before = {name: cp.asnumpy(getattr(parent_state, name)) for name in ("nc",)}
    scratch = FeedbackScratch()
    coupler.feedback_prepare(child, scratch)
    if feedback:
        assert {"nc"} <= set(scratch.payload["kinds"])
    coupler.feedback_commit(child)
    coupler.feedback_finalize(child)
    for index, name in enumerate(("nc",), start=1):
        actual = cp.asnumpy(getattr(parent_state, name))
        expected = before[name].copy()
        if feedback:
            expected[:, 4, 4] = np.float32(index * 8.5)
        np.testing.assert_array_equal(actual, expected)


@requires_gpu
def test_prognostic_nc_ensemble_packs_independent_device_carriers():
    import cupy as cp
    from test_ensemble_batch_state import config, prepared
    from woof.ensemble.batch_state import BatchedDomainState

    cfg = config(moist=True, mp_physics=51)
    members = [prepared(cfg), prepared(cfg)]
    for index, member in enumerate(members, start=1):
        member.arrays["nc"].fill(index * 32)
    batch = BatchedDomainState.from_prepared(members, array_module=cp,
                                            available_bytes=10**7)
    for name in ("nc",):
        for index, member in enumerate(members):
            np.testing.assert_array_equal(cp.asnumpy(batch.member_view(name, index)),
                                          member.arrays[name])
        batch.member_view(name, 0).fill(0)
        np.testing.assert_array_equal(cp.asnumpy(batch.member_view(name, 1)), members[1].arrays[name])
