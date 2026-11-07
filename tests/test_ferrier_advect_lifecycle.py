"""Exercise MP15 payloads through the production lifecycle operators."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

@pytest.mark.gpu
@pytest.mark.parametrize('feedback', [0, 1])
def test_qrimef_crosses_the_production_feedback_operator(feedback):
    import cupy as cp
    from test_feedback import _domain, _seed_state
    from woof.core.state import DomainState
    from woof.core.model import FeedbackScratch
    from woof.core.nest import NestCoupler
    from woof.core.nest_fields import nest_field_kinds
    from woof.core.clock import DomainClock
    domains = [_domain(1, 0, nx=14, ny=14),
               _domain(2, 1, nx=9, ny=9, ratio=3)]
    domains = [replace(d, run=replace(d.run, nz=4, mp_physics=15)) for d in domains]
    states = [DomainState(d.run) for d in domains]
    for i, s in enumerate(states):
        _seed_state(s, child=bool(i))
        s.qi.fill(np.float32(.001))
        s.qrimef.fill(np.float32(.008 if i else .002))
    clocks = [DomainClock(SimpleNamespace(step_ticks=dt, dt_fp32=np.float32(dt)),
                          tick_den=1, run_ticks=60) for dt in (30, 10)]
    clocks[0].ticks, clocks[0].step_count = 30, 1
    parent = SimpleNamespace(cfg=domains[0], state=states[0], clock=clocks[0])
    child = SimpleNamespace(cfg=domains[1], state=states[1], clock=clocks[1], parent=parent)
    coupler = NestCoupler(child, feedback=feedback)
    assert 'qrimef' in nest_field_kinds(domains[1].run)
    coupler.force(child)
    for value, tendency in coupler._last_tables['qrimef'].values():
        np.testing.assert_allclose(value.get(), .008 * 128, rtol=1.e-6)
        np.testing.assert_allclose(tendency.get(), (.002 - .008) * 128 / 30,
                                   rtol=1.e-6)
    clocks[1].ticks = 30
    before = states[0].qrimef.get()
    scratch = FeedbackScratch()
    coupler.feedback_prepare(child, scratch)
    coupler.feedback_commit(child)
    coupler.feedback_finalize(child)
    actual = states[0].qrimef.get()
    if feedback:
        assert actual[0, 4, 4] == pytest.approx(.008, abs=1.e-9)
        np.testing.assert_array_equal(actual[:, :3], before[:, :3])
    else:
        np.testing.assert_array_equal(actual, before)


def test_qrimef_checkpoint_restores_all_words(tmp_path, monkeypatch):
    from test_restart import _cfg, _shim_driver_state, _fill_setup, _fill_serialized
    from woof.io import restart
    cfg = _cfg(moist=True, mp_physics=15)
    state, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, seed=15)
    expected = state.qrimef.tobytes()
    path = restart.write_restart(tmp_path/'mp15.npz', state, cfg)
    fresh, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)
    assert fresh.qrimef.tobytes() == expected
    assert fresh.qrimef0 is not None


def test_offline_parent_reads_the_advected_scalar():
    from woof.offline_child import _scheme_wrf_to_state, _offline_child_mp_physics
    assert _scheme_wrf_to_state(15)['QRIMEF'] == 'qrimef'
    assert 15 in _offline_child_mp_physics()


def test_restart_inventory_keeps_the_pbl_ice_tendency():
    from woof.config import RunConfig
    from woof.core.physics import _pbl_optional_tendency_components
    from woof.core.physics_inventory import REFL_10CM_MICROPHYSICS
    cfg = RunConfig(nx=8, ny=6, nz=4, dx=12000., dy=12000.,
                    ztop=10000., dt=1., run_seconds=2., moist=True,
                    mp_physics=15, bl_pbl_physics=1, sf_sfclay_physics=1,
                    sf_surface_physics=2)
    assert 'rqi' in _pbl_optional_tendency_components(cfg)
    assert 15 in REFL_10CM_MICROPHYSICS


def test_native_restart_reads_rime_and_the_real_cases_drag_inputs():
    from woof.ingest.wrfrst import WRFRST_DISPOSITIONS
    for name in ('QRIMEF', 'VAR', 'CON', 'OA1', 'OA2', 'OA3', 'OA4',
                 'OL1', 'OL2', 'OL3', 'OL4'):
        assert WRFRST_DISPOSITIONS[name].kind == 'read'
    for name in ('DTAUX3D', 'DTAUY3D', 'DUSFCG', 'DVSFCG'):
        assert WRFRST_DISPOSITIONS[name].kind == 'rebuilt'


@pytest.mark.gpu
def test_batched_members_own_independent_rime_payloads():
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.state import DomainState
    from woof.core.device_inventory import state_array_shapes
    from woof.ensemble.batch_state import BatchedDomainState, PreparedHostMember
    cfg = RunConfig(nx=8, ny=6, nz=4, dx=12000., dy=12000.,
                    ztop=10000., dt=1., run_seconds=2., moist=True, mp_physics=15)
    inputs = []
    for member in range(3):
        state = DomainState(cfg, array_module=np)
        state.qi.fill(np.float32(.001))
        state.qrimef.fill(np.float32(.002 * (member + 1)))
        shapes = state_array_shapes(cfg)
        scalars = {k: getattr(state, k) for k in (
            'mub', 'p_top', 'cf1', 'cf2', 'cf3', 'cfn', 'cfn1',
            'has_msf', 'rotational', 'elapsed_seconds')}
        clock = dict(ticks=0, step_ticks=1, tick_den=1, run_ticks=2,
                     step_count=0, dt_fp32=np.float32(1), dtbc_fp32=np.float32(0))
        inputs.append(PreparedHostMember(cfg,
            {k: getattr(state, k).copy() for k in shapes}, scalars, clock))
    batch = BatchedDomainState.from_prepared(inputs, array_module=cp,
                                            available_bytes=1 << 26)
    batch.member_view('qrimef', 1).fill(np.float32(.009))
    for member, expected in enumerate((.002, .009, .006)):
        np.testing.assert_array_equal(batch.member_view('qrimef', member).get(),
            np.full((4, 6, 8), np.float32(expected)))
