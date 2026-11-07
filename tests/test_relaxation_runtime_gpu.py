"""Executed resident lifecycle coverage for WRF selector 31."""
from datetime import datetime
import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [requires_gpu, pytest.mark.gpu]


def _build(shortwave=0):
    import cupy as cp
    from test_restart import _physics_state
    return _physics_state(cp, ra_physics=0, ra_lw_physics=31,
                          ra_sw_physics=shortwave, cu_physics=0)


@pytest.mark.parametrize('shortwave',(0,1))
def test_host_geography_and_held_rate_survive_checkpoint(tmp_path, shortwave):
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _assert_restart_equal
    state,cfg,driver=_build(shortwave)
    # Real-input builders hand initialize_physics NumPy geography.
    assert isinstance(driver.radiation_callable.latitude_deg,cp.ndarray)
    run_steps(state,cfg,40)
    reference=restart.write_restart(tmp_path/'reference.npz',state,cfg)
    split,split_cfg,_=_build(shortwave)
    run_steps(split,split_cfg,20)
    checkpoint=restart.write_restart(tmp_path/'checkpoint.npz',split,split_cfg)
    resumed,resumed_cfg,_=_build(shortwave)
    restart.restore_restart(checkpoint,resumed,resumed_cfg)
    run_steps(resumed,resumed_cfg,20)
    actual=restart.write_restart(tmp_path/'resumed.npz',resumed,resumed_cfg)
    _assert_restart_equal(reference,actual)
    assert resumed.physics.call_counts['radiation']==7
    assert bool(cp.any(resumed.physics.rthratenlw != 0))


@pytest.mark.parametrize('feedback',(0,1))
@pytest.mark.parametrize('shortwave',(0,1))
def test_both_nest_feedback_modes(monkeypatch,feedback,shortwave):
    import cupy as cp
    from tilestream import test_moving_nest as moving, test_nest_executor as executor
    from woof.core.model import execute_experiment
    from woof.core.nest import NestCoupler
    from woof.verify.cases.nest_ideal_common import assemble_idealized_tree
    monkeypatch.setattr(executor,'I_PARENT_START',21)
    monkeypatch.setattr(executor,'J_PARENT_START',21)
    monkeypatch.setattr(executor,'PARENT_DT',6.)
    parent_cfg=moving.parent_cfg(nx=48,ny=48,nz=16,dt=6.,
        ra_physics=0,ra_lw_physics=31,ra_sw_physics=shortwave,radt_minutes=.1,
        sf_sfclay_physics=1,nwp_diagnostics=0)
    child_cfg=moving.child_cfg(parent_cfg,nx=24,ny=24)
    boundaries=moving.domain_boundaries(parent_cfg,seconds=60.)
    parent,_=moving.build_parent(parent_cfg,boundaries=boundaries,warmup=0)
    experiment=executor.experiment(parent_cfg,child_cfg,steps=4)
    model=assemble_idealized_tree(experiment,parent,
        coupler_factory=lambda node:NestCoupler(node,feedback=feedback),
        domain_preparer=moving._preparer)
    if feedback:executor.initial_feedback(model)
    execution=execute_experiment(model,validate_state=True,pool_trim_per_period=False)
    assert execution.steps==16
    child=model.node(2)
    assert child.coupler.force_count>0
    assert (child.coupler.feedback_count>0)==bool(feedback)
    adapters=[]
    for node in (model.root,child):
        driver=node.state.physics
        assert driver.call_counts['radiation']>=4
        assert bool(cp.any(driver.rthratenlw!=0))
        for name in ('u','v','thp','p'):
            assert bool(cp.all(cp.isfinite(getattr(node.state,name))))
        adapters.append(driver.radiation_callable)
    from woof.core.radiation_composition import radiation_adapters
    leaves=[next(item for item in radiation_adapters(adapter)
                 if hasattr(item,'_relaxation_heating')) for adapter in adapters]
    assert leaves[0]._relaxation_heating.data.ptr!=leaves[1]._relaxation_heating.data.ptr


@pytest.mark.parametrize('shortwave',(0,1))
def test_members_on_private_streams_match_serial(tmp_path,shortwave):
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _assert_restart_equal
    references=[]
    for member in range(2):
        state,cfg,_=_build(shortwave)
        state.thp+=cp.float32(member*.1)
        run_steps(state,cfg,4)
        references.append(restart.write_restart(tmp_path/f'serial-{member}.npz',state,cfg))
    members=[]
    for member in range(2):
        state,cfg,driver=_build(shortwave)
        state.thp+=cp.float32(member*.1)
        members.append((state,cfg,driver,cp.cuda.Stream(non_blocking=True)))
    ready=cp.cuda.Event();ready.record()
    for state,cfg,_,stream in members:
        stream.wait_event(ready)
        with stream:run_steps(state,cfg,4)
    from woof.core.radiation_composition import radiation_adapters
    pointers=[next(item for item in radiation_adapters(driver.radiation_callable)
                   if hasattr(item,'_relaxation_heating'))._relaxation_heating.data.ptr
              for _,_,driver,_ in members]
    assert len(set(pointers))==2
    for member,(state,cfg,_,stream) in enumerate(members):
        stream.synchronize()
        actual=restart.write_restart(tmp_path/f'concurrent-{member}.npz',state,cfg)
        _assert_restart_equal(references[member],actual)


@pytest.mark.parametrize('shortwave',(0,1))
def test_tiled_forcing_matches_resident_across_reused_buffers(shortwave):
    import cupy as cp
    from dataclasses import replace
    from types import SimpleNamespace
    from tilestream import harness
    from tilestream.physics_inventory import carrier_manifest
    from woof.core import streaming
    from woof.core.physics import initialize_physics
    from woof.core.radiation_composition import make_radiation
    from woof.core.dycore import step
    from test_radiation_composition import _cfg
    cfg = replace(_cfg(31, shortwave), nx=48, ny=40, dt=1., radt=.05, terrain_opt=1)
    geo = harness.make_geography(cfg)
    start = datetime(2001,6,15,12)
    def build():
        state, _ = harness.make_physics_state(cfg, 123, geography=geo)
        initialize_physics(state, cfg,
            radiation=make_radiation(cfg, start, geo.lat, geo.lon),
            radiation_start_time=start, radiation_latitude=geo.lat,
            radiation_longitude=geo.lon)
        return state
    resident, tiled = build(), build()
    run = streaming.make_stepper(tiled, cfg,
        streaming.StreamingOptions(mode="on", tile_nx=24, tile_ny=20,
                                   nbuffers=2, store="host"),
        build=streaming.prepared_domain_builder(
            SimpleNamespace(state=tiled, cfg=SimpleNamespace(run=cfg), parent=None),
            check_geography=True))
    assert len(run.tiled_run.specs) == 4
    for _ in range(4):
        step(resident, cfg)
        run(tiled, cfg)
        for name, field in carrier_manifest(resident).items():
            expected = cp.asnumpy(field) if isinstance(field, cp.ndarray) else np.asarray(field)
            np.testing.assert_array_equal(run.store[name], expected, err_msg=name)
