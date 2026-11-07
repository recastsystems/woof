"""CCN continuation through the actual CUDA physics and restart inventory.

This tests durable state and member isolation. It is not a native WRF oracle.
"""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
from conftest import requires_gpu

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools/nssl2_wrf461_variant_oracle'))
import cases


def make_state(cccn):
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.state import DomainState
    from woof.core.physics import initialize_physics
    from woof.io import restart
    cfg = RunConfig(nx=6, ny=6, nz=40, dx=1000., dy=1000., ztop=10000.,
                    dt=6., run_seconds=48., moist=True, mp_physics=18,
                    nssl_cccn=cccn)
    state = DomainState(cfg)
    # Give every serialized setup plane a reproducible image; the physics
    # fields below then receive the same six-regime meteorology per row.
    for name in restart.STATE_SETUP_ARRAYS:
        a = getattr(state, name, None)
        if a is not None:
            a[...] = 0
    mode = replace(cases.MODE_BY_LABEL['cccn-low'], cccn=cccn)
    values, env = cases.build_case(mode)
    def cell(a):
        return cp.asarray(np.ascontiguousarray(np.repeat(a.T[:, None, :], 6, axis=1)))
    for name in cases.STATE_FIELDS:
        a = getattr(state, name, None)
        if a is not None:
            a[...] = cell(values[name])
    state.thb[...] = 300.
    state.thp[...] = cell(values['theta']) - cp.float32(300.)
    state.p[...] = cell(env['pressure'])
    state.alt[...] = cp.float32(1.) / cell(env['rho'])
    state.phb[...] = cp.arange(41, dtype=cp.float32) * cp.float32(cases.DZ*9.81)
    state.php[...] = 0.
    state.w[:-1] = cell(env['w'])
    state.w[-1] = state.w[-2]
    state.mub2d[...] = 90000.
    state.mup[...] = 0.
    driver = initialize_physics(state, cfg)
    return cfg, state, driver


def advance(state, driver, cfg, steps):
    from woof.core import microphysics
    from woof.core.refl import consume_refl_10cm
    for _ in range(steps):
        result = microphysics.apply(state, cfg, cfg.dt, refl_10cm_due=True)
        driver.accept_microphysics(result, dt=cfg.dt)
        consume_refl_10cm(state)
        state.elapsed_seconds += cfg.dt


def images(state):
    from woof.io import restart
    manifest = {**restart.state_manifest(state), **restart._scratch_manifest(state),
                **restart._driver_manifest(state.physics)}
    return {k: restart._host(v).tobytes() for k,v in sorted(manifest.items())}


@pytest.mark.gpu
@requires_gpu
def test_actual_nondefault_cuda_continuation_and_wrong_coefficient_refusal(tmp_path):
    from woof.io import restart
    cfg, straight, sd = make_state(1.e7)
    advance(straight, sd, cfg, 8)
    _, split, pd = make_state(1.e7)
    advance(split, pd, cfg, 3)
    path = restart.write_restart(tmp_path/'cccn.npz', split, cfg)
    _, resumed, rd = make_state(1.e7)
    binding = rd.nssl2_binding
    restart.restore_restart(path, resumed, cfg)
    assert rd.nssl2_binding is binding
    assert images(resumed) == images(split)
    advance(resumed, rd, cfg, 5)
    assert images(resumed) == images(straight)
    _, incompatible, _ = make_state(0.5e9)
    original = images(incompatible)
    with pytest.raises(ValueError, match='config|identity|coefficient|digest'):
        restart.restore_restart(path, incompatible, replace(cfg, nssl_cccn=0.5e9))
    assert images(incompatible) == original


@pytest.mark.gpu
@requires_gpu
def test_nondefault_ccn_tree_checkpoint_continues_bitwise(tmp_path):
    """Publish and restore the actual checkpoint generation, then advance."""
    from datetime import datetime, timedelta
    from types import SimpleNamespace
    from woof.core.model import DomainNode, ExperimentState, ModelRuntimeStatus
    from woof.experiment import DomainConfig
    from woof.io import restart

    def tree(cfg, state, steps):
        dc = DomainConfig(grid_id=1, parent_id=0, i_parent_start=1,
                          j_parent_start=1, parent_grid_ratio=1,
                          parent_time_step_ratio=1, history_interval_s=6.,
                          time_step=6, run=cfg)
        clock = SimpleNamespace(
            ticks=steps*6, tick_den=1, step_count=steps, dtbc_fp32=np.float32(0.),
            spec=SimpleNamespace(grid_id=1, start_ticks=0, step_ticks=6),
            history_due=lambda: False)
        node = DomainNode(dc, None, state, clock, None, [], None)
        model = ExperimentState(node, {1:node}, SimpleNamespace(
            period_ticks=6, clock=SimpleNamespace(
                start_time=datetime(2024,5,25,18), tick_den=1, run_ticks=48)),
            None, 'nssl-cccn-continuation-component')
        model._runtime_status = ModelRuntimeStatus()
        return model

    cfg, straight, sd = make_state(1.e7)
    advance(straight, sd, cfg, 8)
    _, split, pd = make_state(1.e7)
    advance(split, pd, cfg, 3)
    source = tree(cfg,split,3)
    path = restart.write_tree_restart(tmp_path,source,
                                    datetime(2024,5,25,18)+timedelta(seconds=18))
    _, resumed, rd = make_state(1.e7)
    model = tree(cfg,resumed,0)
    restart.restore_tree_restart(path,model)
    assert model.root.clock.ticks == 18
    assert model.root.clock.step_count == 3
    assert images(resumed) == images(split)
    advance(resumed,rd,cfg,5)
    assert images(resumed) == images(straight)


@pytest.mark.gpu
@requires_gpu
def test_member_executor_preserves_each_cccn_and_matches_independent_physics():
    from woof.ensemble.physics_execution import MemberPhysicsBinding, MemberPhysicsExecutor
    from woof.core.refl import consume_refl_10cm
    coefficients = (1.e7, 5.e10)
    independent, members = [], []
    for value in coefficients:
        cfg, state, driver = make_state(value)
        advance(state, driver, cfg, 2)
        independent.append(images(state))
        members.append(make_state(value))
    executor = MemberPhysicsExecutor([
        MemberPhysicsBinding(state, cfg, driver, i)
        for i,(cfg,state,driver) in enumerate(members)])
    for _ in range(2):
        executor.apply_microphysics(refl_10cm_due=(True, True))
        for cfg,state,driver in members:
            consume_refl_10cm(state)
            state.elapsed_seconds += cfg.dt
    for (_,state,_), expected in zip(members, independent):
        assert images(state) == expected
    assert images(members[0][1])['state/qnn'] != images(members[1][1])['state/qnn']


@pytest.mark.gpu
@requires_gpu
def test_nondefault_ccn_forcing_carries_every_active_moment():
    import cupy as cp
    from types import SimpleNamespace
    from test_nest_coupler import _nodes, _DeviceState
    from woof.core.nest import NestCoupler
    from woof.core.nssl2_contract import DEFAULT_MODE
    from woof.verify.npref import np_nest_force
    hp, hc = _nodes()
    for node in (hp, hc):
        node.cfg.run = replace(node.cfg.run, moist=True, mp_physics=18, nssl_cccn=1.e7)
        for i,name in enumerate(DEFAULT_MODE.transported_fields):
            value = np.full(node.state.thp.shape, np.float32((i+1)*1.e-6))
            if name == 'qnn':
                value[...] = np.float32(1.e7/1.225)
                value += np.arange(value.shape[-1], dtype=np.float32)[None,None,:]*8
            setattr(node.state, name, value)
    parent = SimpleNamespace(cfg=hp.cfg, state=_DeviceState(hp.state, cp), clock=hp.clock)
    child = SimpleNamespace(cfg=hc.cfg, state=_DeviceState(hc.state, cp), clock=hc.clock, parent=parent)
    coupler = NestCoupler(child, feedback=0)
    expected = np_nest_force(
        hp.state, hc.state, coupler.registrations,
        field_names=DEFAULT_MODE.transported_fields,
        parent_dt_fp32=parent.clock.spec.dt_fp32,
        parent_interval_ticks=parent.clock.spec.step_ticks,
        spec_zone=1, relax_zone=4, spec_bdy_width=5, dtype=np.float32)
    coupler.force(child)
    for name in DEFAULT_MODE.transported_fields:
        for side in ('west','east','south','north'):
            for actual,want in zip(coupler._last_tables[name][side], expected[name][side]):
                np.testing.assert_array_equal(cp.asnumpy(actual).view(np.uint32),
                                              np.asarray(want,np.float32).view(np.uint32),
                                              err_msg=f'{name}/{side}')


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize('feedback', [0,1])
def test_nondefault_ccn_feedback_carries_every_active_moment(feedback):
    import cupy as cp
    from types import SimpleNamespace
    from test_feedback import _domain, _seed_state
    from woof.core.state import DomainState
    from woof.core.nest import NestCoupler
    from woof.core.model import FeedbackScratch
    from woof.core.nest_interp import feedback_parent_bounds
    from woof.core.nssl2_contract import DEFAULT_MODE
    pc = _domain(1,0,nx=14,ny=14)
    cc = _domain(2,1,nx=12,ny=12,ratio=4)
    pc = replace(pc,run=replace(pc.run,mp_physics=18,nssl_cccn=1.e7))
    cc = replace(cc,run=replace(cc.run,mp_physics=18,nssl_cccn=1.e7))
    ps,cs = DomainState(pc.run),DomainState(cc.run)
    _seed_state(ps,child=False);_seed_state(cs,child=True)
    for i,name in enumerate(DEFAULT_MODE.transported_fields):
        # Dyadic constants have an exact float32 sixteen-cell restriction.
        # A decimal constant can change by one ULP in copy_fcn's ordered
        # additions even though every child cell carries the same value.
        getattr(ps,name)[...] = cp.float32((i+1)*2**-6)
        getattr(cs,name)[...] = cp.float32((i+1)*2**-10)
    ps.qnn[...] = cp.float32(1.e7/1.225)
    cs.qnn[...] = cp.float32(2.e6)
    clock=SimpleNamespace(ticks=0)
    parent=SimpleNamespace(cfg=pc,state=ps,clock=clock)
    child=SimpleNamespace(cfg=cc,state=cs,clock=clock,parent=parent)
    coupler=NestCoupler(child,feedback=feedback)
    before={name:cp.asnumpy(getattr(ps,name)).copy() for name in DEFAULT_MODE.transported_fields}
    want={name:cp.asnumpy(getattr(cs,name))[0,0,0] for name in DEFAULT_MODE.transported_fields}
    scratch=FeedbackScratch()
    coupler.feedback_prepare(child,scratch)
    coupler.feedback_commit(child)
    coupler.feedback_finalize(child)
    reg=coupler.registrations['m']
    il,ih,jl,jh=feedback_parent_bounds(reg,spec_zone=cc.run.spec_zone)
    for name in DEFAULT_MODE.transported_fields:
        expected=before[name]
        if feedback:
            expected[:,jl:jh+1,il:ih+1]=want[name]
        np.testing.assert_array_equal(cp.asnumpy(getattr(ps,name)).view(np.uint32),
                                      expected.view(np.uint32),err_msg=name)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize('source_mp', [8, 6])
def test_mixed_edge_uses_target_cccn_in_both_directions(source_mp):
    """Both the ratified and matrix kernels seed the destination domain.

    WRF has no mixed-scheme nest equivalent. Check this extension's CCN
    closure against the native calcnfromq arithmetic, independently on CPU.
    """
    import cupy as cp
    from types import SimpleNamespace
    from woof.config import RunConfig
    from woof.core.microphysics_transition import (
        resolve_microphysics_transition, resolve_reverse_microphysics_transition,
        launch_microphysics_edge_field)
    from woof.core.nssl2_contract import nssl2_background_ccn_per_kg
    base = RunConfig(nx=6, ny=6, nz=4, dx=1000., dy=1000., ztop=10000.,
                     dt=3., run_seconds=9., moist=True)
    target = replace(base, mp_physics=18, nssl_cccn=-1.e7)
    other = replace(base, mp_physics=source_mp)
    forward = resolve_microphysics_transition(other, target)
    reverse = resolve_reverse_microphysics_transition(target, other)
    shape = (4,6,6)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    alt = np.float32(.8) + x*np.float32(.002)
    cloud = np.resize(np.array([0., 1e-14, 1e-9, 2e-8, 1e-5, 1e-4], np.float32), shape)
    state = SimpleNamespace(
        alt=cp.asarray(alt), qc=cp.asarray(cloud), qv=cp.full(shape,.007,cp.float32),
        qr=cp.zeros(shape,cp.float32), qi=cp.zeros(shape,cp.float32),
        qs=cp.zeros(shape,cp.float32), qg=cp.zeros(shape,cp.float32),
        mub2d=cp.full(shape[1:],90000.,cp.float32), mup=cp.zeros(shape[1:],cp.float32),
        c1h=cp.ones(shape[0],cp.float32), c2h=cp.zeros(shape[0],cp.float32))
    rho = np.float32(1.)/alt
    inv = 1./rho.astype(np.float64)
    seed = nssl2_background_ccn_per_kg(-1.e7)
    concentration = (np.float64(seed)*rho.astype(np.float64)).astype(np.float32)
    cloud_number = np.zeros(shape,np.float32)
    active = cloud > np.float32(1e-8)
    cloud_number[active] = np.minimum(seed, cloud[active]*np.float32(327479132160.))*rho[active]
    concentration -= cloud_number
    expected = {'qnn': (concentration.astype(np.float64)*inv).astype(np.float32),
                'qndrop': (cloud_number.astype(np.float64)*inv).astype(np.float32)}
    for contract in (forward,reverse):
        assert contract.target_nssl_cccn == 1.e7
        assert contract.receipt()['nssl2_background_ccn_per_kg'] == float(seed)
        for field,want in expected.items():
            out=cp.empty(shape,cp.float32)
            launch_microphysics_edge_field(contract,state,field,out=out,coupled=False)
            np.testing.assert_array_equal(cp.asnumpy(out).view(np.uint32),want.view(np.uint32))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize('placement', ['host', 'streamed'])
def test_initial_nssl_history_slabs_match_resident_moments(monkeypatch, placement):
    import cupy as cp
    from types import SimpleNamespace
    from woof import wrf_exact
    from woof.core.refl import analysis_refl_10cm
    monkeypatch.setattr(wrf_exact, 'ENABLED', False)
    cfg, resident, _ = make_state(1.e7)
    # More than 64 rows exercises two independent staging slabs. Rain/ice,
    # number and volume moments vary with height and horizontal position.
    names=('p','thb','thp','alt','qv','qc','qr','qi','qs','qg','qh',
           'qnr','qni','qns','qng','qnh','qvolg','qvolh')
    for name in names:
        a=getattr(resident,name)
        if a.ndim == 3:
            setattr(resident,name,cp.tile(a,(1,13,1)))
    shape=resident.qv.shape
    expected=cp.asnumpy(analysis_refl_10cm(resident,cfg=cfg))
    original={name:cp.asnumpy(getattr(resident,name)) for name in names}
    host=SimpleNamespace(**{name:value.copy() for name,value in original.items()})
    if placement=='streamed':
        host._streamed_domain=SimpleNamespace(
            store={'state/'+name:value for name,value in original.items() if name!='thb'},
            _geography={'setup/thb':original['thb']})
        for name in names:
            setattr(host,name,np.full((shape[0],1,shape[2]),-999,np.float32))
    actual=analysis_refl_10cm(host,shape=shape,cfg=cfg)
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    for name,value in original.items():
        live=(host._streamed_domain._geography['setup/thb'] if name=='thb' else
              host._streamed_domain.store['state/'+name]) if placement=='streamed' else getattr(host,name)
        np.testing.assert_array_equal(live.view(np.uint32),value.view(np.uint32))
