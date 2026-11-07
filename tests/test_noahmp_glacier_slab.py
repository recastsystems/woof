"""The resident driver preserves the complete glacier arm and captures on GPU."""
import numpy as np
import pytest
from conftest import requires_gpu


@requires_gpu
@pytest.mark.parametrize('snow_age',['wrf_461','wrf_470'])
def test_resident_glacier_driver_matches_existing_native_arm(snow_age):
    import cupy as cp
    from test_noahmp_runtime import _build, _ICE
    from woof.core.physics import _prepare_atmosphere
    from woof.core.noahmp_runtime import _glacier_lsm_step, _glacier_evaluate_on_device
    from woof.core.noahmp_glacier_slab import run_glacier_slab
    from woof.core.health_ledger import HealthLedger, deferring
    old=_build(nx=6,ny=4,vegtyp=_ICE)
    new=_build(nx=6,ny=4,vegtyp=_ICE)
    fields=old[2].fields;other=new[2].fields
    active=cp.zeros((4,6),dtype=cp.bool_);active[1:3,2:5]=True
    atmosphere=_prepare_atmosphere(old[0]);cosz=cp.full((4,6),.6,dtype=cp.float32)
    dzs=old[2].noahmp_soil_thickness_m
    for unused in range(3):
        _glacier_lsm_step(fields,atmosphere,active,params=old[2].noahmp_params,
            coszen=cosz,dt=12.,dzs=dzs,evaluator=_glacier_evaluate_on_device,
            noahmp_snow_age=snow_age)
        zsoil=-cp.cumsum(cp.asarray(dzs,dtype=cp.float32))
        ledger=HealthLedger()
        with deferring(ledger):
            run_glacier_slab(other,atmosphere,active,coszen=cosz,zsoil=zsoil,dt=12.,
                noahmp_snow_age=snow_age)
        ledger.drain()
        for name,value in fields.items():
            if isinstance(value,cp.ndarray):
                assert value.get().tobytes()==other[name].get().tobytes(),name


@requires_gpu
@pytest.mark.parametrize('options',[dict(opt_btr=2,opt_rsf=2),dict(opt_alb=1,opt_btr=3,opt_rsf=3), dict(opt_rsf=4),
    *(dict(opt_snf=v) for v in (2,3,4,5)), *(dict(dveg=v) for v in (1,3)),
    dict(noahmp_output=3,noahmp_acc_dt=.4)])
def test_selected_glacier_step_captures_without_host_transfers(options,monkeypatch):
    import cupy as cp
    from test_noahmp_runtime import _build,_GRASSLAND,_ICE
    from woof.core.physics import _prepare_atmosphere
    from woof.core.health_ledger import HealthLedger,deferring
    vegetation=np.full((4,6),_GRASSLAND);vegetation[1,2]=_ICE
    state,cfg,driver=_build(nx=6,ny=4,vegtyp=vegetation,**options)
    atmosphere=_prepare_atmosphere(state);ledger=HealthLedger()
    stream=cp.cuda.Stream(non_blocking=True);cp.cuda.get_current_stream().synchronize()
    with stream,deferring(ledger):
        for unused in range(2):driver._run_noahmp(atmosphere,cfg,2)
        stream.synchronize()
        def refuse(*args,**kwargs):raise AssertionError('glacier host transfer')
        monkeypatch.setattr(cp,'asnumpy',refuse)
        stream.begin_capture();driver._run_noahmp(atmosphere,cfg,2)
        graph=stream.end_capture();graph.launch(stream)
    stream.synchronize();monkeypatch.undo();ledger.drain()
    assert float(driver.fields['snowc'][1,2].get())==1.
    assert float(driver.fields['z0'][1,2].get())==pytest.approx(.002)
