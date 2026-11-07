"""QNSE admission, cadence and byte-preserving restart through the real driver."""
from __future__ import annotations
import numpy as np
import pytest
from conftest import requires_gpu


def _build(mfshconv=1, bldt=0.0):
    import cupy as cp
    from woof.config import RunConfig, validate_run_config
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics
    cfg = validate_run_config(RunConfig(nx=8, ny=6, nz=40, dx=3000., dy=3000.,
        ztop=16000., dt=6., run_seconds=60., time_step_sound=4, moist=True,
        mp_physics=6, sf_sfclay_physics=4, sf_surface_physics=2,
        bl_pbl_physics=4, mfshconv=mfshconv, bldt=bldt))
    coord = make_vertical_coord(cfg.nz, stretch=3.2)
    def theta(z):
        z=np.asarray(z,np.float64)
        return np.where(z<1500.,300.,np.where(z<1700.,300.+.03*(z-1500.),306.+.0045*(z-1700.)))
    def qv(z):
        z=np.asarray(z,np.float64)
        return np.where(z<1500.,.0135,np.maximum(.0135-6.e-6*(z-1500.),1.e-5))
    base=make_base_state(coord,theta,p_surf=cfg.p_surf,ztop=cfg.ztop)
    state=init_moist_balanced(cfg,coord,base,qv)
    state.u[...] = cp.float32(7.)
    state.v[...] = cp.float32(1.5)
    land=np.ones((cfg.ny,cfg.nx));land[:,-2:]=0
    tsk=np.where(land,301.,297.)
    soil_t=np.stack([tsk-.5,tsk-1.,tsk-1.5,tsk-2.])
    soil_m=np.full((4,cfg.ny,cfg.nx),.3);soil_m[:,land==0]=1.
    driver=initialize_physics(state,cfg,landmask=land,tsk=tsk,
        soil_temperature=soil_t,soil_moisture=soil_m,liquid_moisture=soil_m,
        ivgtyp=np.where(land,10,17),isltyp=np.where(land,6,14),vegfra=55.,
        tmn=287.,swdown=600.,glw=330.,pblh=500.)
    return state,cfg,driver


@requires_gpu
@pytest.mark.parametrize("mfshconv",[0,1])
@pytest.mark.parametrize("bldt",[0.,.3])
def test_qnse_cadence_and_restart_preserve_every_word(tmp_path,mfshconv,bldt):
    import cupy as cp
    from woof.core.dycore import step
    from woof.io.restart import write_restart,restore_restart
    from woof.core.physics_inventory import MFSHCONV_STATE_3D,MFSHCONV_OUTPUTS_3D
    state,cfg,driver=_build(mfshconv,bldt)
    assert driver.scheme_dispatch["bl_pbl_physics"] == "_run_qnse_pbl"
    assert driver.scheme_dispatch["sf_sfclay_physics"] == "_run_myj_sfclay"
    assert np.all(cp.asnumpy(driver.fields["tke_myj"]) == np.float32(.01))
    assert ("rc_mf" in driver.fields) == bool(mfshconv)
    for _ in range(6): step(state,cfg)
    assert driver.call_counts["ysu"] > 0
    path=tmp_path/"qnse.npz"
    write_restart(path,state,cfg)
    fresh,fresh_cfg,fresh_driver=_build(mfshconv,bldt)
    restore_restart(path,fresh,fresh_cfg)
    def identical():
        # Compare every checkpoint array, including acoustic and held
        # carriers outside the selected state-field list below. The JSON
        # header differs only in its creation timestamp.
        from woof.io.restart import _HEADER_KEY
        original_path = tmp_path / "all-original.npz"
        restored_path = tmp_path / "all-restored.npz"
        write_restart(original_path, state, cfg)
        write_restart(restored_path, fresh, fresh_cfg)
        with np.load(original_path, allow_pickle=False) as original, \
                np.load(restored_path, allow_pickle=False) as restored:
            assert set(original.files) == set(restored.files)
            for key in original.files:
                if key == _HEADER_KEY:
                    import json
                    left = json.loads(original[key].tobytes())
                    right = json.loads(restored[key].tobytes())
                    assert "created" in left and "created" in right
                    left.pop("created")
                    right.pop("created")
                    assert left == right, "restart header changed beyond creation timestamp"
                    continue
                a, b = original[key], restored[key]
                assert a.dtype == b.dtype and a.shape == b.shape, key
                assert a.tobytes() == b.tobytes(), key
        for name,a in driver.fields.items():
            if isinstance(a,cp.ndarray):
                aa=cp.asnumpy(a);bb=cp.asnumpy(fresh_driver.fields[name])
                assert aa.tobytes()==bb.tobytes(),name
        for name in ("u","v","w","thp","mup","qv","qc"):
            assert cp.asnumpy(getattr(state,name)).tobytes()==cp.asnumpy(getattr(fresh,name)).tobytes(),name
    identical()
    step(state,cfg);step(fresh,fresh_cfg)
    identical()
    for name in ("tke_myj","exch_h","exch_m","pblh","t2","u10","v10"):
        assert np.isfinite(cp.asnumpy(driver.fields[name])).all(),name
    if mfshconv:
        for name in MFSHCONV_STATE_3D+MFSHCONV_OUTPUTS_3D:
            assert np.isfinite(cp.asnumpy(driver.fields[name])).all(),name
    assert not np.all(cp.asnumpy(driver.fields["pblh"])==500.)


@requires_gpu
def test_qnse_history_contains_its_carried_and_updraft_fields():
    from woof.core.dycore import step
    from woof.io.wrfout import state_frame
    state,cfg,driver=_build(1)
    step(state,cfg)
    frame=state_frame(state)
    for name in ("TKE_PBL","EL_PBL","RC_MF","MASS_FLUX","ENTR","DETR","THL_UP","THV_UP","RT_UP","RV_UP","RC_UP","U_UP","V_UP","FRAC_UP"):
        assert name in frame,name
        assert np.isfinite(np.asarray(frame[name])).all(),name


@requires_gpu
def test_qnse_stepbl_holds_tke_between_due_calls():
    import cupy as cp
    from woof.core.dycore import step
    state,cfg,driver=_build(1,.3)
    assert driver.stepbl==3 and driver.bldt_seconds==18.
    step(state,cfg)  # WRF always calls NTSD=1.
    held=cp.asnumpy(driver.fields["tke_myj"]).tobytes()
    calls=driver.call_counts["ysu"]
    step(state,cfg)  # NTSD=2 is not due.
    assert driver.call_counts["ysu"]==calls
    assert cp.asnumpy(driver.fields["tke_myj"]).tobytes()==held
    step(state,cfg)  # NTSD=3 is due: multiples of STEPBL, not elapsed intervals from call 1.
    assert driver.call_counts["ysu"]==calls+1
    assert cp.asnumpy(driver.fields["tke_myj"]).tobytes()!=held
    held=cp.asnumpy(driver.fields["tke_myj"]).tobytes()
    for _ in range(2):
        step(state,cfg)
        assert driver.call_counts["ysu"]==calls+1
        assert cp.asnumpy(driver.fields["tke_myj"]).tobytes()==held
    step(state,cfg)
    assert driver.call_counts["ysu"]==calls+2
