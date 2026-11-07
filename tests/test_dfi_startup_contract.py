"""Signed startup clocks and native DFI restrictions before any GPU writes."""
from dataclasses import replace
from datetime import datetime
import numpy as np
import pytest
from woof.config import RunConfig
from woof.core.dfi_graph import PassInterval, DfiBoundaryClock, CompiledModelPass
from woof.core.dfi_model import backward_configuration
from woof.dfi_control import validate_dfi, import_dfi


def test_backward_fractional_clock_keeps_signed_native_fraction():
    interval=PassInterval(0,-120,-5,3)
    clock=DfiBoundaryClock(interval,np.float32(-5)/np.float32(3))
    times=[]
    for _ in range(interval.steps):
        times.append(clock.elapsed_seconds_fp32)
        clock.prepare_step();clock.advance()
    assert interval.steps==24
    # WRF converts whole seconds and remainder separately; direct -5/3
    # rounds one word lower and must not replace that native clock.
    assert times[1].view(np.uint32)==0xbfd55556
    assert clock.elapsed_seconds==-40
    assert clock.dtbc_fp32 < 0 and abs(clock.dtbc_fp32+40) < 1e-4


@pytest.mark.parametrize('start,stop,step,den',[(0,-120,15,1),(0,120,0,1),
 (0,121,15,1),(0,120,15,0),(-(1<<63),1,1,1)])
def test_invalid_clock_refuses_before_submission(start,stop,step,den):
    with pytest.raises(ValueError):PassInterval(start,stop,step,den)


def test_partial_driver_failure_cannot_submit_again(monkeypatch):
    calls=[]
    def fail(*args,**kwargs):
        calls.append(1)
        raise RuntimeError('driver error')
    monkeypatch.setattr('woof.core.dfi_runtime.submit_compiled_pass',fail)
    p=CompiledModelPass(None,None,None,None,None,PassInterval(0,120,15),[])
    with pytest.raises(RuntimeError,match='driver error'):p.submit()
    with pytest.raises(RuntimeError,match='only once'):p.submit()
    assert calls==[1]


def test_backward_disables_native_physics_without_mutating_forecast():
    cfg=RunConfig(nx=12,ny=12,nz=8,dx=3000,dy=3000,ztop=15000,dt=15,run_seconds=3600)
    backward=backward_configuration(cfg,dt=-15,moist_adv_opt=2)
    assert backward.dt==-15 and backward.diff_opt==0 and backward.km_opt==1
    assert backward.sf_ocean_physics==0 and backward.fractional_seaice==0
    assert backward.cu_diag==0 and backward.ra_physics==0 and backward.mp_physics==0
    assert backward.moist_adv_opt==2
    assert cfg.dt>0 and cfg.moist_adv_opt!=2


def test_native_broken_filter_is_an_explicit_optin_with_nyquist_validation():
    cfg=replace(RunConfig(nx=12,ny=12,nz=8,dx=3000,dy=3000,ztop=15000,dt=15,run_seconds=3600),dfi_opt=3,dfi_nfilter=8,
        dfi_bckstop_seconds=-120,dfi_fwdstop_seconds=120)
    validate_dfi(cfg)
    from woof.dfi_control import registry_parameters, RHO_LABEL
    assert registry_parameters()['dfi_nfilter']['enum']==list(range(9))
    assert RHO_LABEL in registry_parameters()['dfi_nfilter']['warnings'][0]
    with pytest.raises(ValueError,match='Nyquist'):
        validate_dfi(replace(cfg,dfi_cutoff_seconds=30))


def test_native_restart_does_not_request_another_filter():
    assert import_dfi({'dfi_control':{'dfi_opt':[3]},
        'time_control':{'restart':[True]}},error=ValueError)=={}


def test_phase_phy_init_retains_surface_diagnostics_but_resets_held_tendencies(monkeypatch):
    from types import SimpleNamespace
    from woof.core.dfi_model import DfiPhysicsInitialization
    names=('t2','q2','th2','u10','v10','psfc')
    old={name:np.arange(4,dtype=np.float32).reshape(2,2)+index
         for index,name in enumerate(names)}
    old['held_tendency']=np.full((2,2),7,dtype=np.float32)
    state=SimpleNamespace(physics=SimpleNamespace(fields=old),
                          _dfi_remember_physics=True)
    fresh=SimpleNamespace(fields={name:np.zeros_like(value) for name,value in old.items()})
    def initialize(state,cfg,**inputs):
        state.physics=fresh
        return fresh
    monkeypatch.setattr('woof.core.physics.initialize_physics',initialize)
    factory=DfiPhysicsInitialization.__new__(DfiPhysicsInitialization)
    factory.state=state;factory.driver=state.physics
    factory.inputs={};factory.radii={};factory.restore_base=lambda cfg:None
    assert factory.reinitialize(object(),epoch=0) is fresh
    for name in names:
        assert np.array_equal(fresh.fields[name].view(np.uint32),old[name].view(np.uint32))
    assert not fresh.fields['held_tendency'].any()
    assert state._dfi_remember_physics is True


def test_dfl_alarms_follow_native_filtered_midpoint():
    from woof.core.clock import DomainClock, DomainTicks
    from woof.checkpoint_schedule import resolve
    from datetime import datetime
    spec=DomainTicks(grid_id=1,parent_id=0,parent_time_step_ratio=1,
        step_ticks=15,dt_fp32=np.float32(15),history_ticks=3600,restart_ticks=3600,
        radt_ticks=None,stepra=None,cudt_ticks=None,stepcu=None,bldt_ticks=None,stepbl=None)
    clock=DomainClock(spec,1,10800)
    clock.dfi_start_ticks=60
    for tick,history,restart in ((60,True,False),(3600,False,False),
                                (3660,True,True),(7260,True,True),
                                (10800,False,False)):
        clock.ticks=tick
        assert clock.history_due()==history
        assert clock.restart_due()==restart
    args=dict(start_time=datetime(2026,10,16,9),run_ticks=10800,
        restart_ticks=3600,tick_den=1,restart_origin_ticks=60)
    assert resolve(('end',),**args).keeps(10800)
    assert resolve(('2026-10-16T10:01:00',),**args).keeps(3660)
    with pytest.raises(ValueError,match='not a restart_interval'):
        resolve(('2026-10-16T10:00:00',),**args)


def test_dfi_memory_is_absent_off_and_scales_with_actual_fields():
    from woof.experiment import experiment_from_run_config
    from woof.core.dfi_memory import startup_shapes
    cfg=RunConfig(nx=12,ny=12,nz=8,dx=3000,dy=3000,ztop=15000,dt=15,run_seconds=3600)
    domain=experiment_from_run_config(cfg,datetime(2026,10,16,9)).root
    assert startup_shapes(domain)=={}
    active=replace(domain,run=replace(cfg,dfi_opt=3,
        dfi_bckstop_seconds=-120,dfi_fwdstop_seconds=120))
    shapes=startup_shapes(active)
    assert shapes['saved/u_2']==(8,12,13)
    assert shapes['saved/ph_2']==(9,12,12)
    assert 'history_snapshots_bound' not in shapes
    history=startup_shapes(replace(active,run=replace(active.run,dfi_write_dfi_history=True)))
    assert history['history_snapshots_bound'][0]>0


def test_dfl_initial_reflectivity_uses_filtered_forecast_start():
    from types import SimpleNamespace
    from woof.core.refl import domain_start_ticks_of,refl_10cm_stash_is_due
    node=SimpleNamespace(clock=SimpleNamespace(
        spec=SimpleNamespace(start_ticks=0),dfi_start_ticks=60))
    origin=domain_start_ticks_of(node)
    assert origin==60
    assert not refl_10cm_stash_is_due(60,domain_start_ticks=origin)
    assert refl_10cm_stash_is_due(3660,domain_start_ticks=origin)


def test_prepared_start_uploads_noah_constants_without_a_model_step(monkeypatch):
    from types import SimpleNamespace
    from woof.core.dfi_initialization import _prepare_model_graph
    from woof.ingest.soil import NOAH_LAYER_THICKNESS_M
    calls = []
    monkeypatch.setattr('woof.core.wsm6.prepare_model_graph',
                        lambda cfg: calls.append(('microphysics', cfg.mp_physics)))
    monkeypatch.setattr('woof.core.noah._device_tables',
                        lambda params, dzs: calls.append(('noah', params, dzs)))
    monkeypatch.setattr('woof.physics_compat.thompson_table_root', lambda: 'tables')
    monkeypatch.setattr('woof.core.thompson_runtime.load_classic_device_tables',
                        lambda root: calls.append(('thompson', root)))
    cfg = SimpleNamespace(mp_physics=8, sf_surface_physics=2)
    params = object()
    state = SimpleNamespace(physics=SimpleNamespace(noah_params=params))
    _prepare_model_graph(state, cfg)
    assert calls == [('microphysics', 8), ('thompson', 'tables'),
                     ('noah', params, NOAH_LAYER_THICKNESS_M)]
    calls.clear()
    _prepare_model_graph(SimpleNamespace(physics=None),
                         SimpleNamespace(mp_physics=0, sf_surface_physics=0))
    assert calls == [('microphysics', 0)]
