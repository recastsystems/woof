import numpy as np
import pytest
from conftest import requires_gpu
from test_physics_driver import _full_state
from woof.io.restart import write_restart, restore_restart

@pytest.mark.gpu
@requires_gpu
def test_skin_advances_on_held_surface_steps_and_restores(tmp_path):
    import cupy as cp
    state,cfg,driver = _full_state(sst_skin=1, bldt=1., sf_surface_physics=0, bl_pbl_physics=0)
    driver.fields["gsw"][...] = cp.float32(1000.)
    driver.fields["glw"][...] = cp.float32(400.)
    driver.compute(state,cfg)
    water = driver.fields["xland"] > 1.5
    first = driver.fields["dtw"].copy()
    calls = driver.call_counts["sfclay"]
    state.elapsed_seconds = cfg.dt
    driver.compute(state,cfg)
    assert driver.call_counts["sfclay"] == calls
    assert bool(cp.any(driver.fields["dtw"][water] != first[water]))
    path=write_restart(tmp_path/"skin.npz",state,cfg)
    resumed,rcfg,rdriver = _full_state(sst_skin=1,bldt=1.,sf_surface_physics=0,bl_pbl_physics=0)
    restore_restart(path,resumed,rcfg)
    for key in ("sst","sstsk","dtw","tsk","tslb"):
        np.testing.assert_array_equal(cp.asnumpy(rdriver.fields[key]).view(np.uint32),cp.asnumpy(driver.fields[key]).view(np.uint32))
    state.elapsed_seconds = resumed.elapsed_seconds = 2*cfg.dt
    driver.compute(state,cfg)
    rdriver.compute(resumed,rcfg)
    for key in ("sstsk","dtw","tsk"):
        np.testing.assert_array_equal(cp.asnumpy(rdriver.fields[key]).view(np.uint32),cp.asnumpy(driver.fields[key]).view(np.uint32))

@pytest.mark.gpu
@requires_gpu
def test_bulk_sst_snapshot_does_not_alias_device_skin_temperature():
    import cupy as cp
    from woof.core.physics import initialize_physics
    state,cfg,old = _full_state(sst_skin=1,sf_surface_physics=0,bl_pbl_physics=0)
    temperature=old.fields["tsk"].copy()
    driver=initialize_physics(state,cfg,landmask=0.,tsk=temperature,sst=temperature,glw=400.,swdown=1000.)
    assert driver.fields["sst"].data.ptr != driver.fields["tsk"].data.ptr
    bulk=driver.fields["sst"].copy()
    driver.fields["gsw"][...] = cp.float32(1000.)
    driver.compute(state,cfg)
    np.testing.assert_array_equal(cp.asnumpy(driver.fields["sst"]).view(np.uint32),cp.asnumpy(bulk).view(np.uint32))
    assert bool(cp.any(driver.fields["tsk"] != bulk))
