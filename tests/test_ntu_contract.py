"""NTU allocation, history, checkpoint and oracle refusal checks."""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import netCDF4
import numpy as np
import pytest
from woof.config import RunConfig, validate_run_config
from woof.core.ntu import NTU_SPECIES, NTU_SCALARS, KMAX_TIERS, kmax_for
from woof.core.device_inventory import state_array_shapes
from woof.core.nest_fields import nest_field_kinds
from woof.io.history_layout import live_state_history_fields
from woof.io.wrf_output_schema import NTU_OUTPUT_FIELDS
from woof.io.wrfout import WrfoutWriter
from woof.io import restart
from test_restart import _shim_state, _fill_setup

ROOT=Path(__file__).resolve().parents[1]
def config(**kw):
    return RunConfig(nx=6,ny=4,nz=5,dx=2000,dy=2000,ztop=8000,dt=1,run_seconds=0,moist=True,mp_physics=56,**kw)


def test_ntu_allocation_and_checkpoint_classification(monkeypatch):
    cfg=config(); state=_shim_state(cfg,monkeypatch); declared=state_array_shapes(cfg)
    for name in NTU_SPECIES:
        assert getattr(state,name).shape==declared[name]
        assert restart.classify_state_attr(name)=='serialize'
        assert restart.classify_state_attr(name+'0')=='rebuild'
        assert name in nest_field_kinds(cfg)
    assert len(NTU_SCALARS)==21 and len(NTU_SPECIES)==25
    from woof.ingest.lateral_bc import COUPLED_SCALAR_STATE_FIELDS
    assert set(NTU_SPECIES)<=COUPLED_SCALAR_STATE_FIELDS
    from woof.physics_registry import physics_registry
    assert physics_registry()['parameters']['ntu_ccnty']['enum']==[1,2,3,4]


def test_ntu_ensemble_pack_keeps_each_members_prognostics():
    from test_ensemble_batch_state import prepared, pack
    from woof.ensemble.batch_dycore import member_domain_view
    cfg=replace(config(),dt=3.0,run_seconds=30.0)
    members=(prepared(cfg),prepared(cfg))
    for member,row in enumerate(members):
        for index,name in enumerate(NTU_SPECIES):
            row.arrays[name].fill(np.float32((member+1)*(index+1)*0.125))
            row.arrays[name].reshape(-1)[0]=np.float32(-0.0)
    batch=pack(members)
    for member,row in enumerate(members):
        view=member_domain_view(batch,member)
        for name in NTU_SPECIES:
            assert getattr(view,name).tobytes()==row.arrays[name].tobytes()
    member_domain_view(batch,0).qrcn.fill(7)
    assert batch.member_view('qrcn',1).tobytes()==members[1].arrays['qrcn'].tobytes()


def test_ntu_initial_reflectivity_uses_the_ntu_distribution(monkeypatch):
    from woof.core import ntu,state_reflectivity
    state=object(); sentinel=object();cfg=config()
    calls=[]
    def diagnose(s,**kwargs):
        calls.append((s,kwargs));return sentinel
    monkeypatch.setattr(ntu,'reflectivity',diagnose)
    assert state_reflectivity.diagnose(state,cfg) is sentinel
    assert calls==[(state,{'cfg':cfg})]


from conftest import requires_gpu

@requires_gpu
def test_ntu_restart_keeps_each_tracer_word_and_binds_aerosol(tmp_path):
    import cupy as cp
    from woof.core.state import DomainState
    def fill_setup(state):
        for index, name in enumerate(restart.STATE_SETUP_ARRAYS):
            array=getattr(state,name)
            array[...] = cp.arange(array.size,dtype=cp.float64).reshape(array.shape).astype(cp.float32)*cp.float32(0.01*(index+1))
    cfg=config(); source=DomainState(cfg); fill_setup(source)
    for i,name in enumerate(NTU_SPECIES):
        value=getattr(source,name); value[...] = cp.arange(value.size,dtype=cp.float32).reshape(value.shape)*cp.float32(i+1)
        value.reshape(-1)[0]=np.float32(-0.0); value.reshape(-1)[-1]=np.float32(1e-40)
    path=restart.write_restart(tmp_path/'ntu.npz',source,cfg)
    fresh=DomainState(cfg); fill_setup(fresh); restart.restore_restart(path,fresh,cfg)
    for name in NTU_SPECIES: assert cp.asnumpy(getattr(source,name)).tobytes()==cp.asnumpy(getattr(fresh,name)).tobytes()
    changed=replace(cfg,ntu_ccnty=3); other=DomainState(changed); fill_setup(other)
    with pytest.raises(restart.RestartMismatchError): restart.restore_restart(path,other,changed)


def test_ntu_history_publishes_each_extra_moment(tmp_path):
    state=SimpleNamespace(**{attr:np.full((5,4,6),i+0.25,dtype=np.float32) for i,(_,attr) in enumerate(NTU_SCALARS)})
    fields=live_state_history_fields(state)
    path=tmp_path/'history'
    with WrfoutWriter(path,nx=6,ny=4,nz=5,dx=2000,dy=2000,field_schema=fields) as writer: writer.write_frame('2026-10-08_00:00:00',fields)
    with netCDF4.Dataset(path) as ds:
        for attr,row in NTU_OUTPUT_FIELDS.items():
            var=ds.variables[row.netcdf_name]
            assert var.dimensions==('Time','bottom_top','south_north','west_east')
            assert var.units==row.units
            assert np.asarray(var[0]).tobytes()==getattr(state,attr).tobytes()
    with netCDF4.Dataset(path) as ds:
        assert len(fields)==len(NTU_SCALARS)
        for name,value in fields.items():assert np.asarray(ds.variables[name][0]).tobytes()==value.tobytes(),name
    assert 'AS' in fields and 'AS_' not in fields


@pytest.mark.parametrize('nz,tier',[(2,64),(64,64),(65,128),(128,128),(129,256),(256,256)])
def test_tier_bounds(nz,tier): assert kmax_for(nz)==tier

def test_deeper_column_is_refused():
    with pytest.raises(ValueError,match='overrun'): kmax_for(257)


@pytest.mark.parametrize('mode',['equal','onebit','signedzero','nan','truncated'])
def test_oracle_comparison_returns_a_failing_exit(tmp_path,mode):
    meta={'A':dict(nz=2,columns=['edge'],steps=1)}; (tmp_path/'meta.json').write_text(json.dumps(meta))
    a=np.ones(5*2+29*2+9,dtype='<f4'); b=a.copy()
    if mode=='onebit': b.view(np.uint32)[19]^=np.uint32(1)
    if mode=='signedzero': a[19]=0; b[19]=-0.0
    if mode=='nan': a[19]=b[19]=np.nan
    if mode=='truncated': b=b[:-1]
    a.tofile(tmp_path/'a.bin'); b.tofile(tmp_path/'b.bin')
    result=subprocess.run([sys.executable,str(ROOT/'tools/ntu_wrf461_oracle/compare.py'),str(tmp_path/'meta.json'),'A',str(tmp_path/'a.bin'),str(tmp_path/'b.bin')],capture_output=True,text=True)
    assert (result.returncode==0)==(mode=='equal'),result.stdout+result.stderr


def test_fast_default_keeps_strict_math_in_the_proof_arm():
    from woof.core.ntu import module_source, compile_options
    from woof import numerics
    source=module_source(64)
    if numerics.mode() == "strict":
        assert '#define NTU_ERFF(x)   ntu_erff(x)' in source
    else:
        assert '#define NTU_ERFF(x) erff(x)' in source
        assert '#define NTU_POWF(x, y) powf((x), (y))' in source
    assert '--device-debug' not in compile_options()
    assert not any('Ofast-compile' in option for option in compile_options())
    assert '--ftz=false' in compile_options()




def test_ntu_input_inventory_restores_every_prognostic():
    from woof.ingest.wrfinput import active_moisture_inventory, active_moisture_map
    cfg=config(); required,allowed=active_moisture_inventory(cfg)
    fields=active_moisture_map(cfg)
    assert required==allowed
    assert set(fields.values())=={'qv','qc','qr'}|set(NTU_SPECIES)
    assert fields['AS']=='as_'
    assert all(row.netcdf_name in required for row in NTU_OUTPUT_FIELDS.values())


@requires_gpu
def test_ntu_evolved_column_restart_continues_word_identically(tmp_path):
    """The real NTU launcher evolves the complete restored scalar package."""
    import cupy as cp
    from woof.core import ntu
    from woof.core.state import DomainState
    from tools.ntu_wrf461_oracle.make_columns import make_column
    from tools.ntu_wrf461_oracle.run_gpu import ENGINE
    cfg=RunConfig(nx=6,ny=4,nz=40,dx=2000,dy=2000,ztop=20000,dt=20,
                  run_seconds=0,moist=True,mp_physics=56)
    rng=np.random.default_rng(18)
    regimes=('deep_convective','stratiform','warm_rain','mixed_phase',
             'cold_snow','hail','high_terrain','edge')
    columns=[make_column(regime,variant,40,rng) for regime in regimes for variant in (0,1,2)]
    def data(name):
        shape=(41,4,6) if name in ('W','PHB','PH') else (40,4,6)
        return cp.asarray(np.stack([column[name] for column in columns],axis=1).astype('f4').reshape(shape))
    def setup(state):
        for index,name in enumerate(restart.STATE_SETUP_ARRAYS):
            array=getattr(state,name)
            array[...] = cp.arange(array.size,dtype=cp.float32).reshape(array.shape)*cp.float32(0.01*(index+1))
        state.thb[...] = 0
    source=DomainState(cfg);setup(source)
    # thb=0 makes the serialized perturbation equal to full theta here.
    source.thp[...] = data('TH')
    for name,attr in ENGINE.items():
        if attr in ('qv','qc','qr')+ntu.NTU_SPECIES:getattr(source,attr)[...] = data(name)
    fixed={name:data(key) for name,key in (('p','P'),('dz','DZ'),('pii','PII'),('w','W'))}
    ntu.launch_init(data('PHB'),data('PH'),fixed['p'],cp.zeros_like(source.qv),
                    data('ALT'),source.qv,[getattr(source,n) for n in ntu.NTU_AEROSOL],
                    ccnty=2,restart=False)
    def step(state):
        fields=dict(fixed,theta=state.thp)
        fields.update({name:getattr(state,name) for name in ('qv','qc','qr')+ntu.NTU_SPECIES})
        mapping={'rainnc':'mp_rainnc','rainncv':'mp_rainncv','snownc':'mp_snownc',
                 'snowncv':'mp_snowncv','graupelnc':'mp_graupelnc','graupelncv':'mp_graupelncv',
                 'hailnc':'mp_hailnc','hailncv':'mp_hailncv','sr':'mp_sr'}
        surface={name:state.scratch((4,6),slot) for name,slot in mapping.items()}
        ntu.launch(fields,surface,20,ccnty=2)
    before={name:cp.asnumpy(getattr(source,name)).tobytes() for name in ntu.NTU_SPECIES}
    step(source)
    assert sum(cp.asnumpy(getattr(source,name)).tobytes()!=before[name] for name in ntu.NTU_SPECIES)>=10
    checkpoint=restart.write_restart(tmp_path/'evolved-ntu.npz',source,cfg)
    resumed=DomainState(cfg);setup(resumed);restart.restore_restart(checkpoint,resumed,cfg)
    for _ in range(2):step(source);step(resumed)
    for name in ('thp','qv','qc','qr')+ntu.NTU_SPECIES:
        assert cp.asnumpy(getattr(source,name)).tobytes()==cp.asnumpy(getattr(resumed,name)).tobytes(),name
    for name in ntu.PRECIPITATION_SLOTS:
        assert cp.asnumpy(source.scratch((4,6),name)).tobytes()==cp.asnumpy(resumed.scratch((4,6),name)).tobytes(),name


def test_ntu_legacy_radiation_uses_ice_and_snow_membership():
    from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags, _MP_DECLARES_RADII
    assert legacy_cloud_fraction_flags(56)==(True,True)
    # WRF computes radii in radiation when use_mp_re does not name NTU.
    assert _MP_DECLARES_RADII[56] is False


def test_ntu_forecast_receipt_includes_all_number_moments():
    from tools.ntu_wrf461_oracle.forecast_receipt import field_names
    names=field_names()
    assert len(names)==29 and len(set(names))==29
    assert {'QNCLOUD','QNRAIN','QNICE','QNSNOW','QNGRAUPEL','QNHAIL'}<=set(names)


def test_native_source_identity_rejects_modified_physics():
    from woof.io.restart import MICROPHYSICS_ALGORITHM_IDENTITIES
    identity=MICROPHYSICS_ALGORITHM_IDENTITIES[56]
    assert 'v4.7.1' in identity
    assert 'fixes-f0-f6' not in identity


@pytest.mark.parametrize('previous',[
    'ntu-multi-moment-wrf-v4.6.1-v2-float32-log10-cos-3mom-ice-shape-density-prognostic-aerosol-fixes-f0-f6',
    'ntu-multi-moment-wrf-v4.6.1-v3-own-erf-device-debug-log10-cos-3mom-ice-shape-density-prognostic-aerosol-fixes-f0-f6'])
def test_ntu_restart_refuses_the_previous_erf_arithmetic(tmp_path,monkeypatch,previous):
    cfg=config()
    source=_shim_state(cfg,monkeypatch);_fill_setup(source)
    with monkeypatch.context() as old:
        old.setitem(restart.MICROPHYSICS_ALGORITHM_IDENTITIES,56,previous)
        path=restart.write_restart(tmp_path/'previous-ntu.npz',source,cfg)
    fresh=_shim_state(cfg,monkeypatch);_fill_setup(fresh)
    with pytest.raises(restart.RestartMismatchError):
        restart.restore_restart(path,fresh,cfg)


@requires_gpu
def test_ntu_same_scheme_nest_forces_every_prognostic():
    import cupy as cp
    from test_nest_coupler import _nodes,_DeviceState
    from woof.core.nest import NestCoupler
    from woof.verify.npref import np_nest_force
    hp,hc=_nodes()
    for node in (hp,hc):
        node.cfg.run=replace(node.cfg.run,mp_physics=56,moist=True,moist_cq=True)
        shape=node.state.qv.shape
        for i,name in enumerate(NTU_SPECIES):
            value=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)
            setattr(node.state,name,(value+np.float32(i+1))*np.float32(1e-6))
    parent=SimpleNamespace(cfg=hp.cfg,state=_DeviceState(hp.state,cp),clock=hp.clock)
    child=SimpleNamespace(cfg=hc.cfg,state=_DeviceState(hc.state,cp),parent=parent,clock=hc.clock)
    coupler=NestCoupler(child)
    fields=('qv','qc','qr')+NTU_SPECIES
    expected=np_nest_force(hp.state,hc.state,coupler.registrations,field_names=fields,
                           parent_dt_fp32=parent.clock.spec.dt_fp32,
                           parent_interval_ticks=parent.clock.spec.step_ticks,
                           spec_zone=1,relax_zone=4,spec_bdy_width=5,dtype=np.float32)
    before={name:cp.asnumpy(getattr(parent.state,name)).tobytes() for name in fields}
    coupler.force(child)
    assert set(fields)<=set(coupler._last_tables)
    for name in fields:
        for side in ('west','east','south','north'):
            for got,want in zip(coupler._last_tables[name][side],expected[name][side]):
                np.testing.assert_array_equal(cp.asnumpy(got).view(np.uint32),
                                              np.asarray(want,dtype=np.float32).view(np.uint32),
                                              err_msg=f'{name}/{side}')
        assert cp.asnumpy(getattr(parent.state,name)).tobytes()==before[name]


@requires_gpu
@pytest.mark.parametrize('feedback',[0,1])
def test_ntu_nest_feedback_keeps_or_restricts_every_prognostic(feedback):
    import cupy as cp
    from test_nest_coupler import _nodes,_DeviceState
    from woof.core.nest import NestCoupler
    from woof.core.model import FeedbackScratch
    from woof.verify.npref import np_copy_fcn
    hp,hc=_nodes()
    fields=('qv','qc','qr')+NTU_SPECIES
    for node,offset in ((hp,1),(hc,4)):
        node.cfg.run=replace(node.cfg.run,mp_physics=56,moist=True,moist_cq=True)
        for index,name in enumerate(fields):
            setattr(node.state,name,np.full_like(node.state.qv,np.float32((index+offset)*0.125)))
    parent=SimpleNamespace(cfg=hp.cfg,state=_DeviceState(hp.state,cp),clock=hp.clock)
    child=SimpleNamespace(cfg=hc.cfg,state=_DeviceState(hc.state,cp),parent=parent,clock=hc.clock)
    child.clock.ticks=parent.clock.ticks
    coupler=NestCoupler(child,feedback=feedback,smooth_option=0)
    scratch=FeedbackScratch()
    coupler.feedback_prepare(child,scratch)
    if feedback:
        assert set(fields)<=set(scratch.payload['kinds'])
    else:
        assert scratch.payload is None
    coupler.feedback_commit(child)
    for name in fields:
        before=getattr(hp.state,name)
        expected=np_copy_fcn(before,getattr(hc.state,name),i_parent_start=hc.cfg.i_parent_start,
            j_parent_start=hc.cfg.j_parent_start,nri=3,nrj=3) if feedback else before
        actual=cp.asnumpy(getattr(parent.state,name))
        np.testing.assert_allclose(actual,expected,rtol=4*np.finfo(np.float32).eps,atol=0,err_msg=name)
        if feedback:
            assert np.any(actual!=before),name
        else:
            assert actual.tobytes()==before.tobytes(),name


def test_ntu_held_mixing_tendencies_have_lifetime_rows():
    from woof.core.preflight import scratch_slot_registry,scratch_slot_lifetime
    cfg=config(km_opt=4,diff_6th_opt=2,smdiv=0.1)
    slots=scratch_slot_registry(cfg)
    for name in ('qv','qc','qr')+NTU_SPECIES:
        slot='smag_r'+name
        assert slot in slots
        assert scratch_slot_lifetime(slot) is not None,slot


@pytest.mark.parametrize('population', [1, 2, 3, 4])
def test_ntu_population_binds_only_ntu_checkpoint_identity(population):
    active = replace(config(), ntu_ccnty=population)
    assert restart.configuration_echo(active)['ntu_ccnty'] == population
    active_digest = restart._configuration_digest_values(restart.configuration_echo(active))
    assert active_digest['ntu_ccnty'] == population
    inactive = replace(active, mp_physics=0)
    reference = replace(inactive, ntu_ccnty=2)
    assert restart.configuration_echo(inactive) == restart.configuration_echo(reference)
    assert 'ntu_ccnty' not in restart.configuration_echo(inactive)
