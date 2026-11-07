"""Exercise the complete CAM driver through actual GPU lifecycle seams."""
import numpy as np
import pytest
cp=pytest.importorskip('cupy')
try:
    if cp.cuda.runtime.getDeviceCount()<1: raise RuntimeError('no device')
except Exception:
    pytest.skip('no CUDA device',allow_module_level=True)

def test_cam_held_radiation_and_carriers_resume_bitwise(tmp_path):
    from test_restart import _physics_state, _assert_restart_equal
    from woof.core.dycore import run_steps
    from woof.io import restart
    def build():
        return _physics_state(cp,ra_physics=0,ra_lw_physics=3,ra_sw_physics=3)
    state,cfg,_=build()
    run_steps(state,cfg,40)
    reference=restart.write_restart(tmp_path/'reference.npz',state,cfg)
    split,cfg_split,_=build()
    run_steps(split,cfg_split,20)
    checkpoint=restart.write_restart(tmp_path/'split.npz',split,cfg_split)
    restored,cfg_restored,_=build()
    restart.restore_restart(checkpoint,restored,cfg_restored)
    run_steps(restored,cfg_restored,20)
    resumed=restart.write_restart(tmp_path/'resumed.npz',restored,cfg_restored)
    _assert_restart_equal(reference,resumed)
    assert restored.physics.call_counts['radiation']==7
    assert restored.physics.radiation_callable.update_count==7
    assert bool(cp.all(restored.physics.fields['lwupt'] > 0))


@pytest.mark.parametrize('feedback',(0,1))
def test_cam_runs_in_both_nested_feedback_modes(monkeypatch,feedback):
    from tilestream import test_moving_nest as moving, test_nest_executor as executor
    from woof.core.model import execute_experiment
    from woof.core.nest import NestCoupler
    from woof.verify.cases.nest_ideal_common import assemble_idealized_tree
    monkeypatch.setattr(executor,'I_PARENT_START',21)
    monkeypatch.setattr(executor,'J_PARENT_START',21)
    monkeypatch.setattr(executor,'PARENT_DT',6.)
    parent_cfg=moving.parent_cfg(nx=48,ny=48,nz=16,dt=6.,
        ra_physics=0,ra_lw_physics=3,ra_sw_physics=3,radt_minutes=.1,
        sf_sfclay_physics=1,nwp_diagnostics=0)
    child_cfg=moving.child_cfg(parent_cfg,nx=24,ny=24)
    boundaries=moving.domain_boundaries(parent_cfg,seconds=60.)
    parent,geo=moving.build_parent(parent_cfg,boundaries=boundaries,warmup=0)
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
        assert bool(cp.any(driver.fields['lwupt']>0))
        for name in ('swupt','lwupt','taucldc','taucldi','cemiss','swcf','lwcf'):
            assert bool(cp.all(cp.isfinite(driver.fields[name])))
        adapters.append(driver.radiation_callable)
    assert adapters[0]._retained['abstot_3d'].data.ptr!=adapters[1]._retained['abstot_3d'].data.ptr


def test_cam_independent_members_match_serial_on_private_cuda_streams(tmp_path):
    from test_restart import _physics_state, _assert_restart_equal
    from woof.io import restart
    references=[]
    for member in range(2):
        state,cfg,driver=_physics_state(cp,ra_physics=0,ra_lw_physics=3,ra_sw_physics=3)
        state.qc*=cp.float32(1+member)
        driver.compute(state,cfg)
        references.append(restart.write_restart(tmp_path/f'serial-{member}.npz',state,cfg))
    members=[]
    for member in range(2):
        state,cfg,driver=_physics_state(cp,ra_physics=0,ra_lw_physics=3,ra_sw_physics=3)
        state.qc*=cp.float32(1+member)
        members.append((state,cfg,driver,cp.cuda.Stream(non_blocking=True)))
    ready=cp.cuda.Event();ready.record()
    for state,cfg,driver,stream in members:
        stream.wait_event(ready)
        with stream:driver.compute(state,cfg)
    for member,(state,cfg,driver,stream) in enumerate(members):
        stream.synchronize()
        actual=restart.write_restart(tmp_path/f'concurrent-{member}.npz',state,cfg)
        _assert_restart_equal(references[member],actual)

@pytest.mark.parametrize('pair',[(3,1),(1,3),(3,0),(0,3)])
def test_cam_independent_spectra_and_checkpoint(pair,tmp_path):
    from test_restart import _physics_state,_assert_restart_equal
    from woof.io import restart
    from woof.core.radiation_composition import radiation_adapters
    state,cfg,driver=_physics_state(cp,ra_physics=0,ra_lw_physics=pair[0],ra_sw_physics=pair[1])
    driver.compute(state,cfg)
    cam=next(a for a in radiation_adapters(driver.radiation_callable) if type(a).__name__=='Cam3Radiation')
    assert cam.longwave==(pair[0]==3) and cam.shortwave==(pair[1]==3)
    assert bool(cp.all(cp.isfinite(driver.rthraten)))
    first=restart.write_restart(tmp_path/'before.npz',state,cfg)
    restored,cfg2,_=_physics_state(cp,ra_physics=0,ra_lw_physics=pair[0],ra_sw_physics=pair[1])
    restart.restore_restart(first,restored,cfg2)
    second=restart.write_restart(tmp_path/'after.npz',restored,cfg2)
    _assert_restart_equal(first,second)


@pytest.mark.parametrize('pair', [(3, 3), (3, 1), (1, 3)])
def test_cam_tile_twin_keeps_native_tables_policy_and_private_state(monkeypatch, pair):
    from test_restart import _physics_state
    from woof.core import streaming
    from woof.core.radiation_composition import radiation_adapters
    state, cfg, driver = _physics_state(
        cp, ra_physics=0, ra_lw_physics=pair[0], ra_sw_physics=pair[1], icloud=2)
    driver.compute(state, cfg)
    domain = driver.radiation_callable
    cam = next(a for a in radiation_adapters(domain) if type(a).__name__ == 'Cam3Radiation')
    assert cam.update_count > 0 and cam._retained
    latitude = cp.full((2, 3), 41., cp.float32)
    longitude = cp.full((2, 3), -93., cp.float32)
    # The constructor must use the domain's resolved tables, even when the
    # environment would select another root for a fresh forecast.
    monkeypatch.setenv('WOOF_CAM_TABLE_ROOT', '/unavailable-cam-tile-tables')
    twins = [streaming._tile_scheme(domain, latitude, longitude) for _ in range(2)]
    adapters = [next(a for a in radiation_adapters(twin)
                     if type(a).__name__ == 'Cam3Radiation') for twin in twins]
    for twin in adapters:
        assert twin is not cam and twin.tables is cam.tables
        assert twin.tables.root == cam.tables.root
        assert (twin.icloud, twin.column_chunk, twin.longwave, twin.shortwave) == (
            cam.icloud, cam.column_chunk, cam.longwave, cam.shortwave)
        assert twin.start_time == cam.start_time
        cp.testing.assert_array_equal(twin.latitude_deg, latitude)
        cp.testing.assert_array_equal(twin.longitude_deg, longitude)
        assert twin._retained == {} and twin._abs_time is None and twin.update_count == 0
    assert adapters[0]._retained is not adapters[1]._retained
    assert adapters[0]._ozone.data.ptr != adapters[1]._ozone.data.ptr
