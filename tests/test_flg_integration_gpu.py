"""Live FLG checkpoint, member ownership, and existing nest transport gates."""
from dataclasses import replace
import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def _state(cp):
    from test_restart import _physics_state
    return _physics_state(cp, mp_physics=6, cu_physics=0, ra_physics=0,
                          ra_lw_physics=7, ra_sw_physics=7, ztop=16000.)


def test_flg_off_cadence_checkpoint_preserves_next_radiation_call(tmp_path):
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _assert_restart_equal
    a,ca,da=_state(cp)
    run_steps(a,ca,8)
    reference=restart.write_restart(tmp_path/'reference.npz',a,ca)
    b,cb,db=_state(cp)
    run_steps(b,cb,4)
    assert db.call_counts['radiation']==1
    mid=restart.write_restart(tmp_path/'mid.npz',b,cb)
    c,cc,dc=_state(cp)
    restart.restore_restart(mid,c,cc)
    run_steps(c,cc,4)
    resumed=restart.write_restart(tmp_path/'resumed.npz',c,cc)
    _assert_restart_equal(reference,resumed)
    assert dc.call_counts['radiation']==da.call_counts['radiation']==2
    assert bool(cp.any(dc.rthratenlw!=0.))


def test_flg_member_local_plan_and_device_storage_are_isolated():
    import cupy as cp
    from woof.core.physics import _prepare_atmosphere
    from woof.ensemble.suite_capabilities import plan_suite
    a,ca,da=_state(cp); b,cb,db=_state(cp)
    plan=plan_suite(ca,members=2)
    radiation=next(row for row in plan.components if row.component=='radiation')
    assert radiation.implemented and radiation.option_id=='fu-liou-gu'
    assert plan.mode=='member_local'
    da._run_radiation(_prepare_atmosphere(a),a,ca)
    before=cp.asnumpy(da.rthratenlw)
    b.qc[2:6]*=cp.float32(2.)
    db._run_radiation(_prepare_atmosphere(b),b,cb)
    assert da.radiation_callable.column.workspace.data.ptr != db.radiation_callable.column.workspace.data.ptr
    assert da.radiation_callable.surface.data.ptr != db.radiation_callable.surface.data.ptr
    np.testing.assert_array_equal(cp.asnumpy(da.rthratenlw),before)
    assert not np.array_equal(cp.asnumpy(db.rthratenlw),before)


def test_flg_one_way_nest_uses_existing_device_force_tables(monkeypatch):
    import test_nest_coupler as fixture
    original=fixture._run
    def selected(*args,**kwargs):
        return replace(original(*args,**kwargs),ra_lw_physics=7,ra_sw_physics=7)
    monkeypatch.setattr(fixture,'_run',selected)
    fixture.test_gpu_force_tables_match_np_nest_force_real_emulation()


def test_flg_two_way_nest_uses_existing_device_restriction(monkeypatch):
    import test_feedback as fixture
    original=fixture._domain
    def selected(*args,**kwargs):
        domain=original(*args,**kwargs)
        return replace(domain,run=replace(domain.run,ra_lw_physics=7,ra_sw_physics=7,nz=16,mp_physics=6))
    monkeypatch.setattr(fixture,'_domain',selected)
    fixture.test_feedback_restricts_exact_mass_faces_and_leaves_rim_untouched(3)


def test_mixed_flg_retains_native_rounding_and_passive_sw_case():
    import cupy as cp
    from datetime import datetime
    from types import SimpleNamespace
    from woof.core.flg_composition import NativeFLGComposition
    from woof.core.physics import RadiationResult
    from woof.config import RunConfig
    shape=(4,1,1); surface=(1,1); calls=[]
    def slab(v):return cp.full(shape,v,cp.float32)
    def plane(v):return cp.full(surface,v,cp.float32)
    class Leaf:
        publishes_olr=True
        diagnostics=()
        heat_total=cp.full((1,4),1.e8,cp.float32)
        def __init__(self,name):self.name=name
        def __call__(self,**kwargs):
            calls.append(self.name)
            return RadiationResult(rthratenlw=slab(1.e8),rthratensw=slab(1.),
                swdown=plane(300.),glw=plane(250.),gsw=plane(270.),coszen=plane(.5),
                olr=plane(200.),swddir=plane(100.),swddif=plane(200.),
                flux_diagnostics={n:plane(v) for n,v in [('SWDDNI',99.),('SWDDIR',99.),('SWDDIF',99.)]})
    atmosphere={'pressure':slab(70000.)}
    cfg=RunConfig(nx=1,ny=1,nz=4,dx=30000.,dy=30000.,ztop=20000.,dt=180.,run_seconds=0.,ra_lw_physics=7,ra_sw_physics=5)
    args=dict(atmosphere=atmosphere,fields={'glw':plane(250.)},state=SimpleNamespace(elapsed_seconds=0.),cfg=cfg)
    combined=NativeFLGComposition(datetime(2000,1,1),plane(40.),plane(-90.),
        longwave_adapter=Leaf('LW7'),shortwave_adapter=Leaf('SW5'),flg_longwave=True)
    result=combined(**args)
    assert calls==['LW7','SW5']
    np.testing.assert_array_equal(cp.asnumpy(result.rthratensw),0.)
    np.testing.assert_array_equal(cp.asnumpy(result.flux_diagnostics['SWDDNI']),200.)
    calls.clear()
    passive=NativeFLGComposition(datetime(2000,1,1),plane(40.),plane(-90.),
        longwave_adapter=Leaf('LW5'),shortwave_adapter=None,flg_longwave=False)
    cfg=replace(cfg,ra_lw_physics=5,ra_sw_physics=7)
    result=passive(**dict(args,cfg=cfg))
    assert calls==['LW5']
    np.testing.assert_array_equal(cp.asnumpy(result.rthratensw),0.)
    np.testing.assert_array_equal(cp.asnumpy(result.swdown),0.)
