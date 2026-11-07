"""Production coupling checks, run only through the card queue."""
import cupy as cp
import pytest
from woof.config import RunConfig
from woof.core.kf import kf_phase_mode_for_microphysics, KFPhaseMode
from woof.core.physics import _cumulus_optional_tendency_components, _sr_roundoff_envelope
from woof.wrf_physics_inventory import stock_wrf_physics_inventory

pytestmark = pytest.mark.gpu


def test_initial_history_keeps_native_mp7_reflectivity_before_first_step():
    import numpy as np
    from types import SimpleNamespace
    from woof.core.refl import analysis_refl_10cm
    from woof.core.state import DomainState
    cfg = RunConfig(nx=16, ny=16, nz=40, dx=3000.0, dy=3000.0,
                    ztop=20000.0, dt=6.0, run_seconds=18.0,
                    moist=True, mp_physics=7)
    state = DomainState(cfg)
    state.qh.fill(np.float32(0.001))
    before = state.qh.copy()
    frame = analysis_refl_10cm(state, cfg)
    assert isinstance(frame, cp.ndarray)
    assert frame.shape == state.qv.shape
    assert bool(cp.all(frame.view(cp.uint32) == 0))
    assert bool(cp.array_equal(state.qh, before))
    # The store-backed history route must return the whole requested shape,
    # even when the resident template contains only one bounded slab.
    host = analysis_refl_10cm(
        SimpleNamespace(qv=np.zeros((40, 1, 16), np.float32)), cfg,
        shape=(40, 75, 16), array_module=np)
    assert host.shape == (40, 75, 16)
    assert not np.any(host.view(np.uint32))


@pytest.mark.parametrize('cumulus', [1, 3])
def test_cumulus_feedback_keeps_the_packages_separate_ice_and_snow(cumulus):
    inventory = stock_wrf_physics_inventory(7)
    fields = {f.registry_name for f in inventory.wrfinput_fields}
    assert {'qi', 'qs', 'qh'} <= fields
    cfg = RunConfig(nx=16, ny=16, nz=40, dx=3000.0, dy=3000.0,
                    ztop=20000.0, dt=6.0, run_seconds=18.0,
                    moist=True, mp_physics=7, cu_physics=cumulus)
    assert kf_phase_mode_for_microphysics(7) == KFPhaseMode.SEPARATE_ICE_SNOW
    assert _cumulus_optional_tendency_components(cfg) == ('rqr', 'rqi', 'rqs')


def test_sr_bound_is_roundoff_only_and_does_not_clip_a_field():
    import numpy as np
    upper, ulps, loops = _sr_roundoff_envelope(7, 18.0)
    assert ulps == 4
    assert upper.view(np.uint32) == np.float32(1.0).view(np.uint32) + 4
    values = cp.array([1.0, upper], dtype=cp.float32)
    before = values.copy()
    assert bool(cp.all(values <= upper))
    assert bool(cp.array_equal(values, before))


def test_hail_mass_enters_every_acoustic_face_loading():
    import numpy as np
    from woof.core.state import DomainState
    from woof.core.acoustic import prepare_moist_cq
    cfg = RunConfig(nx=16, ny=16, nz=40, dx=3000.0, dy=3000.0,
                    ztop=20000.0, dt=6.0, run_seconds=18.0,
                    moist=True, mp_physics=7)
    state = DomainState(cfg)
    state.qh.fill(np.float32(0.1))
    cqu, cqv, cqw = prepare_moist_cq(state, cfg)[:3]
    expected = np.float32(1.0) / (np.float32(1.0) + np.float32(0.1))
    for field in (cqu, cqv, cqw[1:-1]):
        np.testing.assert_array_equal(cp.asnumpy(field), expected)
    np.testing.assert_array_equal(cp.asnumpy(cqw[[0, -1]]), np.float32(1.0))


def test_existing_hail_packages_keep_their_complete_transport_inventory():
    from dataclasses import replace
    from woof.core.state import DomainState
    from woof.core.moist import extra_moist_species, NSSL_SPECIES, NSSL_MASS_SPECIES, MY2_SPECIES
    cfg = RunConfig(nx=16, ny=16, nz=40, dx=3000.0, dy=3000.0,
                    ztop=20000.0, dt=6.0, run_seconds=18.0, moist=True, mp_physics=7)
    assert extra_moist_species(DomainState(cfg)) == NSSL_MASS_SPECIES
    for selector, expected in ((9, MY2_SPECIES), (18, NSSL_SPECIES)):
        state = DomainState(replace(cfg, mp_physics=selector))
        assert extra_moist_species(state) == expected


def test_ensemble_hail_loading_matches_independent_scalar_members():
    import numpy as np
    from woof.core import acoustic as scalar
    from woof.ensemble import batch_acoustic as batch
    from test_ensemble_batch_acoustic_gpu import _pack_physical, _same_words

    state, references, cfg, _ = _pack_physical(2, moist=True, mp_physics=7)
    batch.prepare_moist_cq(state, cfg)
    clear = state.scratch_member_view('acoustic_cqu', 0).copy()
    for member, reference in enumerate(references):
        hail = np.float32(0.02 * (member + 1))
        state.member_view('qh', member).fill(hail)
        reference.qh.fill(hail)
    batch.prepare_moist_cq(state, cfg)
    assert bool(cp.all(state.scratch_member_view('acoustic_cqu', 0) < clear))
    for member, reference in enumerate(references):
        expected = scalar.prepare_moist_cq(reference, cfg)
        for slot, value in zip(('acoustic_cqu', 'acoustic_cqv', 'acoustic_cqw'),
                               expected[:3], strict=True):
            _same_words(cp, state.scratch_member_view(slot, member), value)


def test_restart_preserves_nonzero_hail_and_carried_heating(tmp_path):
    import numpy as np
    from woof.core import microphysics
    from woof.core.physics import initialize_physics
    from woof.io.restart import write_restart, restore_restart
    from test_mp_spec_zone_ring import _square_moist_state, _mp_state_fields

    state, cfg = _square_moist_state(mp=7)
    initialize_physics(state, cfg, xland=1.0)
    microphysics.apply(state, cfg, cfg.dt)
    state.qh[...] = cp.linspace(1e-5, 2e-4, state.qh.size,
                               dtype=cp.float32).reshape(state.qh.shape)
    state.acphysf.fill(np.float32(0.125))
    carried = {name: cp.asnumpy(getattr(state, name)).view(np.uint32).copy()
               for name in _mp_state_fields(state, cfg)}
    path = tmp_path / 'mp7-restart.npz'
    write_restart(path, state, cfg)
    resumed, resumed_cfg = _square_moist_state(mp=7)
    initialize_physics(resumed, resumed_cfg, xland=1.0)
    restore_restart(path, resumed, resumed_cfg)
    for name, expected in carried.items():
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(resumed, name)).view(np.uint32), expected)
    microphysics.apply(state, cfg, cfg.dt)
    microphysics.apply(resumed, resumed_cfg, resumed_cfg.dt)
    for name in carried:
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(resumed, name)).view(np.uint32),
            cp.asnumpy(getattr(state, name)).view(np.uint32))


@pytest.mark.parametrize('feedback', [0, 1])
def test_nested_members_transport_hail_with_both_native_feedback_modes(feedback):
    """Real resident and batch edge kernels agree, including nonzero hail."""
    import numpy as np
    from woof.core.device_inventory import state_array_shapes
    from test_ensemble_batch_nesting_gpu import _fixture, _same

    parent, child, nodes, edge = _fixture(
        2, ratio=3, mapped=True, smooth=0, mp_physics=7, feedback=feedback)
    assert 'qh' in edge.fields
    for member, (pnode, cnode, _) in enumerate(nodes):
        # Member-specific endpoints catch missing hail and member broadcasts.
        pnode.state.qh.fill(np.float32((member + 1) * 0.0001))
        parent.member_view('qh', member)[:] = pnode.state.qh
        cnode.state.qh.fill(np.float32((member + 1) * 0.0003))
        child.member_view('qh', member)[:] = cnode.state.qh
    before = [cp.asnumpy(p.state.qh).copy() for p, _, _ in nodes]
    pclock = tuple(p.clock for p, _, _ in nodes)
    cclock = tuple(c.clock for _, c, _ in nodes)
    tables = edge.force(parent_clocks=pclock, child_clocks=cclock)
    assert 'qh' in tables
    for member, (pnode, cnode, _) in enumerate(nodes):
        cnode.coupler.force(cnode)
        for side, values in tables['qh'].items():
            for actual, expected in zip(values, cnode.coupler._last_tables['qh'][side], strict=True):
                _same(cp, actual[member], expected)
        cnode.clock.ticks = pnode.clock.ticks
    # Exact FORCE itself rounds parent fields; feedback=0 preserves this endpoint.
    before = [cp.asnumpy(p.state.qh).copy() for p, _, _ in nodes]
    if feedback:
        edge.feedback_prepare(parent_clocks=pclock, child_clocks=cclock)
        edge.feedback_commit()
        edge.feedback_finalize()
        for _, cnode, scratch in nodes:
            cnode.coupler.feedback_prepare(cnode, scratch)
            cnode.coupler.feedback_commit(cnode)
            cnode.coupler.feedback_finalize(cnode)
    for member, (pnode, cnode, _) in enumerate(nodes):
        for batch, scalar in ((parent, pnode.state), (child, cnode.state)):
            for name in state_array_shapes(batch.cfg):
                _same(cp, batch.member_view(name, member), getattr(scalar, name))
        if feedback:
            assert np.any(cp.asnumpy(pnode.state.qh) != before[member])
        else:
            np.testing.assert_array_equal(cp.asnumpy(pnode.state.qh), before[member])
