"""Native WRF handoff oracles; CUDA checks require the card queue."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
import pytest
from conftest import requires_gpu
ROOT=Path('tests/data/oracles/mskf/handoffs')

@pytest.mark.parametrize('icloud',[1,3])
def test_mskf_radiation_view_leaves_one_complete_feedback_merge(monkeypatch,icloud):
    from woof.core import cloud_fraction
    feedback={'qc_cu':object()}
    atmosphere={'cumulus_feedback':feedback}
    fields={}
    state=object()
    cfg=SimpleNamespace(cu_physics=11,cu_rad_feedback=True,mp_physics=40,
                        icloud=icloud)
    calls=[]
    def fraction3(a,f,s,c):
        calls.append((a,f,s,c))
        return cloud_fraction.RadiationCloudView(a,f,s,'fraction3')
    monkeypatch.setattr(cloud_fraction,'prepare_cloud_fraction3',fraction3)
    view=cloud_fraction.prepare_radiation_clouds(atmosphere,fields,state,cfg)
    assert view.atmosphere['cumulus_feedback'] is feedback
    assert view.state is state and view.fields is fields
    assert len(calls)==(icloud==3)
    assert view.fraction==('fraction3' if icloud==3 else None)

def test_handoff_fixture_integrity():
    meta=json.loads((ROOT/'manifest.json').read_text())
    assert meta['columns']==64 and meta['months']==8
    assert hashlib.sha256((ROOT/'handoffs.npz').read_bytes()).hexdigest()==meta['fixture_sha256']

def test_mskf_config_guards():
    from dataclasses import replace
    from woof.config import RunConfig,validate_run_config
    cfg=RunConfig(nx=24,ny=24,nz=40,ztop=16000.,run_seconds=60.,dx=3000.,dy=3000.,dt=10.,moist=True,cu_physics=11,mp_physics=40)
    validate_run_config(cfg)
    for arm in (1,2):
        validate_run_config(replace(cfg,aercu_opt=arm))
    with pytest.raises(ValueError,match='check_a_mundo'):validate_run_config(replace(cfg,km_opt=5))
    with pytest.raises(ValueError,match='requires cu_physics=11 and mp_physics=40'):
        validate_run_config(replace(cfg,aercu_opt=1,mp_physics=10))

def test_native_warm_start_preserves_hail_duration_and_starts_absent_windows():
    from woof.ingest.wrfrst import scratch_restore_plan
    assert scratch_restore_plan('hailcast_wdur',11,{'HAILCAST_WDUR'})==('fill','HAILCAST_WDUR')
    assert scratch_restore_plan('hailcast_wup_mask',11,{'HAILCAST_WUP_MASK'})==('fill','HAILCAST_WUP_MASK')
    plan=scratch_restore_plan('hailcast_runmax',11,{'HAILCAST_WDUR'})
    assert plan[0:2]==('cold',0.0) and 'restart instant' in plan[2]

def test_aerosol_restart_classification():
    from woof.io.restart import classify_state_attr
    assert classify_state_attr('aerocu')=='serialize'
    for name in ('aeromcu','aeropcu','aerosol_start_time'):assert classify_state_attr(name)=='setup'


@pytest.fixture
def wrf_exact_numerics(monkeypatch):
    """These words are graded against native WRF, a strict-numerics claim.

    0d68b73458 made the shared direct route fast by default (--fmad=true
    --ftz=true unless WRF-exact is on). Under the fast default both handoffs
    differ from native by 1-3 ULP (radiation: 624 of 3584 words), and pass
    0-different with GPUWM_NUMERICS=strict GPUWM_WRF_EXACT=1. Pin that mode
    instead of running the proof in whatever mode the process happens to be.
    """
    import woof.numerics as numerics
    import woof.wrf_exact as exact
    from woof.core.kernels import no_ftz
    monkeypatch.setenv("GPUWM_WRF_EXACT", "1")
    monkeypatch.setattr(exact, "ENABLED", True)
    monkeypatch.setattr(numerics, "_mode", "strict")
    no_ftz.get_kernel_no_ftz.cache_clear()
    no_ftz.load_module_no_ftz.cache_clear()
    yield
    no_ftz.get_kernel_no_ftz.cache_clear()
    no_ftz.load_module_no_ftz.cache_clear()

@requires_gpu
def test_time_and_pressure_interpolation_matches_native_wrf(wrf_exact_numerics):
    import cupy as cp
    from woof.core.kernels.no_ftz import get_kernel_no_ftz
    fx=np.load(ROOT/'handoffs.npz')
    pressure=cp.asarray(fx['pressure']);nz,ncol=pressure.shape
    out=cp.empty((10,nz,ncol),cp.float32)
    kernel=get_kernel_no_ftz('cumulus_aerosol','cumulus_aerosol')
    for day,expected in zip(fx['days'],fx['aerosol']):
        status=cp.zeros(1,cp.int32)
        kernel((1,),(128,),(cp.asarray(fx['monthly']),cp.asarray(fx['pressure_monthly']),pressure,out,status,day,np.int32(30),np.int32(nz),np.int32(ncol)))
        assert int(status.get()[0])==0
        np.testing.assert_array_equal(out.get().view('u4'),expected.view('u4'))
    pm=cp.asarray(fx['pressure_monthly']);pm[:,5]=pm[:,4]
    kernel((1,),(128,),(cp.asarray(fx['monthly']),pm,pressure,out,status,np.float32(0),np.int32(30),np.int32(nz),np.int32(ncol)))
    assert int(status.get()[0])!=0

@requires_gpu
def test_radiation_feedback_matches_native_wrf(wrf_exact_numerics):
    import cupy as cp
    from woof.core.cumulus_feedback import merge_cumulus_feedback
    fx=np.load(ROOT/'handoffs.npz')
    dev=lambda n:cp.asarray(fx[n])
    feedback={key:dev(source) for key,source in dict(cldfra_dp='deep',cldfra_sh='shallow',qc_cu='cu_qc',qi_cu='cu_qi',qs_cu='cu_qs',efcs='cu_ec',efis='cu_ei',efss='cu_es').items()}
    qc,qi,qs,cloud,radii=merge_cumulus_feedback(dev('qc'),dev('qi'),dev('qs'),dev('cloud'),feedback,aerosol=True,radii={'effc':dev('ec'),'effi':dev('ei'),'effs':dev('es')})
    arrays=[qc,qi,qs,cloud]+[radii[n]*np.float32(1e-6) for n in ('effc','effi','effs')]
    actual=np.stack([x.get() for x in arrays])
    np.testing.assert_array_equal(actual.view('u4'),fx['radiation'].view('u4'))

@pytest.mark.parametrize("big_endian",[False,True])
@pytest.mark.parametrize("fixture",["handoffs","wrapped-handoffs"])
def test_rust_horizontal_forcing_matches_native_wrf(tmp_path,fixture,big_endian):
    import os,subprocess
    binary=os.environ.get('WOOF_AEROSOL_GRID') or 'tools/aerosol_grid/target/release/aerosol-grid'
    if not Path(binary).is_file():pytest.skip('build the Rust aerosol provider on the CPU box first')
    horizontal_root=ROOT.parent/fixture
    fx=np.load(horizontal_root/'horizontal.npz')
    fx['coordinates'].astype('<f4').tofile(tmp_path/'grid.bin')
    asset=horizontal_root/'toy-cesm.dat'
    if big_endian:
        data=asset.read_bytes()
        asset=tmp_path/'big-endian.dat'
        asset.write_bytes(b''.join(data[i:i+4][::-1] for i in range(0,len(data),4)))
    subprocess.run([binary,str(asset),str(tmp_path/'grid.bin'),str(tmp_path/'output.bin')],check=True)
    got=np.fromfile(tmp_path/'output.bin','<f4').reshape(fx['expected'].shape)
    np.testing.assert_array_equal(got.view('u4'),fx['expected'].view('u4'))
    bad=tmp_path/'bad.dat';bad.write_bytes(b'bad')
    assert subprocess.run([binary,str(bad),str(tmp_path/'grid.bin'),str(tmp_path/'bad.bin')],capture_output=True).returncode!=0


def test_positive_aerosol_inventory_prices_native_monthly_and_scratch():
    from woof.config import RunConfig
    from woof.core.device_inventory import state_array_shapes
    from woof.core.preflight import scratch_slot_registry
    cfg=RunConfig(nx=24,ny=24,nz=40,ztop=16000.,run_seconds=60.,dx=3000.,dy=3000.,dt=10.,moist=True,cu_physics=11,mp_physics=40,aercu_opt=2)
    shapes=state_array_shapes(cfg)
    assert shapes['aeromcu']==(10,12,30,24,24)
    assert shapes['aeropcu']==(12,30,24,24)
    assert shapes['aerocu']==(10,40,24,24)
    assert scratch_slot_registry(cfg)['morr_aero_kzh_up']==(40,24,24)


def test_public_aerosol_options_are_opt_in_with_cost_and_rewrite_warning():
    from woof.physics_registry import physics_registry
    registry=physics_registry()
    assert registry['parameters']['aercu_opt']['enum'] == [0,1,2]
    multiplier = registry['parameters']['aercu_fct']
    assert multiplier['implemented'] is True
    assert 'unimplemented_reason' not in multiplier
    mp40 = registry['components']['microphysics']['options']['morrison-aero-mp40']
    assert mp40['aerosol_state']['forecast_input_active'] is True
    assert registry['parameters']['aercu_opt']['default'] == 0
    assert 'aercu_opt' not in mp40['constraints']['required_settings']
    warning = ' '.join(mp40['warnings'])
    assert 'opt-in' in warning and 'written from papers, not bitwise to WRF' in warning
    assert 'aercu_opt=0' in warning and 'x/' in warning
    row=registry['parameters']['cu_rad_feedback']
    assert row['implemented'] is True
    assert 'unimplemented_reason' not in row


@pytest.mark.parametrize('aerosol_option',[1,2])
def test_public_plan_admits_explicit_positive_aerosol_options(aerosol_option):
    from woof.physics_registry import (PLAN_SCHEMA,physics_registry,
                                        registry_sha256,validate_physics_plan)
    registry=physics_registry()
    # Supply a test transport for a stock-namelist parameter. Every physics
    # option, parameter specification and composition constraint stays shipped.
    route=dict(registry['runner_routes']['tools.prepared_single_domain_forecast'])
    route['allowed_parameter_keys']=['aercu_opt']
    registry['runner_routes']['mskf-test-transport']=route
    plan={'schema':PLAN_SCHEMA,'plan_id':'mskf-monthly-aerosol',
          'registry_sha256':registry_sha256(registry),
          'context':{'runner_id':'mskf-test-transport','source_id':'hrrr',
                     'topology_id':'single-domain-v1'},
          'domains':[{'domain_id':'d01',
                      'template_id':'wsm6-ysu-mm5-noah-no-radiation-v1',
                      'components':{'cumulus':'multiscale-kf',
                                    'microphysics':'morrison-aero-mp40'},
                      'parameters':{'aercu_opt':aerosol_option}}],
          'edges':[]}
    result=validate_physics_plan(plan,registry=registry)
    assert result['launchable'], result['errors']
    plan['domains'][0]['parameters']['aercu_opt'] = 0
    control = validate_physics_plan(plan,registry=registry)
    assert control['launchable'], control['errors']


def test_mskf_held_momentum_and_activation_work_are_classified():
    from woof.io.restart import classify_scratch_slot
    for name in ('cu_rucuten','cu_rvcuten'):assert classify_scratch_slot(name)=='serialize'
    for name in ('morr_aero_work','morr_aero_ccn','morr_aero_kzh_up'):assert classify_scratch_slot(name)=='rebuild'


@pytest.mark.parametrize('start,elapsed,day',[
 ('2024-01-01T00:00:00+00:00',0,0),
 ('2024-02-28T23:00:00+00:00',7200,59+1/24),
 ('2024-12-31T21:00:00+00:00',10800,0),
 ('2024-05-21T21:00:00+00:00',300,141+21/24+5/1440)])
def test_forcing_calendar_handles_elapsed_time_and_year_rollover(start,elapsed,day):
    import datetime
    from woof.core.cumulus_aerosol import forcing_julian
    epoch=datetime.datetime.fromisoformat(start).timestamp()
    assert forcing_julian(epoch,elapsed).view('u4')==np.float32(day).view('u4')


def test_retained_cloud_feedback_round_trips_with_cumulus_history():
    from woof.io.restart import _cumulus_history_manifest,_restore_cumulus_history
    state=object()
    source=SimpleNamespace(w0avg=np.arange(8,dtype='f4').reshape(2,2,2),cldfra_dp=np.full((2,2,2),.25,dtype='f4'),cu_act=np.ones((2,2),dtype='i4'))
    stored={k:v.copy() for k,v in _cumulus_history_manifest(source).items()}
    assert set(stored)=={'cumulus/w0avg','cumulus/cldfra_dp','cumulus/cu_act'}
    target=SimpleNamespace(w0avg=np.zeros((2,2,2),'f4'),cldfra_dp=np.zeros((2,2,2),'f4'),cu_act=np.zeros((2,2),'i4'))
    original=target.cldfra_dp
    _restore_cumulus_history(stored,target,state,75.,np.asarray)
    for key,value in stored.items():np.testing.assert_array_equal(getattr(target,key.split('/')[1]),value)
    assert target.cldfra_dp is original and target._history_state is state and target._history_time==75.

@pytest.mark.parametrize('minutes',[0.0,1.0,7.5])
@pytest.mark.parametrize('trigger',[1])
def test_public_namelist_preserves_mskf_cumulus_cadence(tmp_path,minutes,trigger):
    import re
    from test_namelist_import import INPUT_TEXT,_pair
    from woof.namelist_import import import_namelists
    text=re.sub(r'(?m)^.*mp_physics\s*=.*$', ' mp_physics = 40,40,',INPUT_TEXT)
    text=re.sub(r'(?m)^.*cu_physics\s*=.*$', ' cu_physics = 11,11,',text)
    text=re.sub(r'(?m)^.*cudt\s*=.*$', f' cudt = {minutes},{minutes},',text)
    if ' cudt =' not in text:text=text.replace('&physics',f'&physics\n cudt = {minutes},{minutes},')
    text=text.replace('&physics',f'&physics\n kfeta_trigger = {trigger},')
    toml,report=import_namelists(*_pair(tmp_path,inp=text))
    import tomllib
    from woof.experiment import build_experiment
    exp=build_experiment(tomllib.loads(toml),source=str(tmp_path/'namelist.input'))
    # Native STEPCU uses NINT(CUDT*60/DT). The 60 s parent rounds
    # 7.5 minutes to eight steps; the 15 s child hits 7.5 exactly.
    expected = [8.0, 7.5] if minutes == 7.5 else [minutes, minutes]
    assert [dc.run.cudt_minutes for dc in exp.domains] == expected
    assert not any('cudt' in str(row) for row in report.dropped)


def test_inactive_aerosol_keys_preserve_checkpoint_configuration_bytes():
    from dataclasses import replace
    from woof.config import RunConfig
    from woof.io.restart import configuration_echo
    cfg=RunConfig(nx=24,ny=24,nz=40,ztop=16000.,run_seconds=60.,dx=3000.,dy=3000.,dt=10.,moist=True,cu_physics=11,mp_physics=40)
    echo=configuration_echo(cfg)
    assert 'aercu_opt' not in echo and 'aercu_fct' not in echo
    active=configuration_echo(replace(cfg,aercu_opt=2,aercu_fct=.5))
    assert active['aercu_opt']==2 and active['aercu_fct']==.5


def test_activation_restart_inventory_has_one_reader_writer_authority():
    from woof import state_serialization_contract as contract
    from woof.io import restart
    from woof.core.morrison_aero import AERO_DIAGNOSTIC_FIELDS
    assert restart.STATE_SERIALIZED_ATTRS is contract.STATE_SERIALIZED_ATTRS
    assert set(AERO_DIAGNOSTIC_FIELDS)<=set(contract.STATE_SERIALIZED_ATTRS)


def test_mskf_history_planner_and_live_carriers_agree():
    from dataclasses import replace
    from woof.config import RunConfig
    from woof.core.column_cumulus import MskfColumnCumulus
    from woof.core.mskf import OUTPUT_3D, OUTPUT_2D
    from woof.io.history_layout import produced_history_shapes
    from woof.io.wrf_output_schema import MSKF_OUTPUT_FIELDS, HISTORY_FIELDS_BY_NETCDF_NAME
    cfg=RunConfig(nx=24,ny=24,nz=40,ztop=16000.,run_seconds=60.,dx=3000.,dy=3000.,dt=10.,moist=True,cu_physics=11,mp_physics=6)
    adapter=object.__new__(MskfColumnCumulus)
    for key in (*OUTPUT_3D,'w0avg'):
        setattr(adapter,key,np.zeros((40,24,24),'f4'))
    for key in OUTPUT_2D:
        setattr(adapter,key,np.zeros((24,24),'f4'))
    adapter.cu_act=np.zeros((24,24),'i4')
    live=adapter.history_fields()
    planned=produced_history_shapes(cfg)
    assert len(live)==35
    for key,schema in MSKF_OUTPUT_FIELDS.items():
        assert live[schema.netcdf_name] is getattr(adapter,key)
        assert planned[schema.netcdf_name]==live[schema.netcdf_name].shape
        assert HISTORY_FIELDS_BY_NETCDF_NAME[schema.netcdf_name].dtype==live[schema.netcdf_name].dtype.str[1:]
    assert 'CU_UAF' in live and 'AINC_FRAC' not in live
    assert 'CU_UAF' not in produced_history_shapes(replace(cfg,cu_physics=0))


def test_mskf_relocation_restores_every_module_carrier_in_place():
    from woof.core.column_cumulus import MskfColumnCumulus
    from woof.core.mskf import OUTPUT_3D,OUTPUT_2D
    from woof.core.physics_continuation import capture_continuation,restore_continuation
    from woof.io.restart import CUMULUS_CALLABLE_ARRAYS
    state=SimpleNamespace(w=np.zeros((4,2,3),'f4'),existing_scratch=lambda _:None)
    source=object.__new__(MskfColumnCumulus)
    target=object.__new__(MskfColumnCumulus)
    for key in (*OUTPUT_3D,*OUTPUT_2D,'w0avg','cu_act'):
        shape=(3,2,3) if key in (*OUTPUT_3D,'w0avg') else (2,3)
        dtype='i4' if key=='cu_act' else 'f4'
        setattr(source,key,np.full(shape,3,dtype))
        setattr(target,key,np.zeros(shape,dtype))
    owned = {*OUTPUT_3D, *OUTPUT_2D, 'w0avg', 'cu_act'}
    assert owned <= CUMULUS_CALLABLE_ARRAYS
    held={key:getattr(target,key) for key in owned}
    captured=capture_continuation(state,SimpleNamespace(cumulus_callable=source))
    receipt=restore_continuation(state,SimpleNamespace(cumulus_callable=target),captured)
    for key in owned:
        assert getattr(target,key) is held[key]
        np.testing.assert_array_equal(getattr(target,key),getattr(source,key))
    assert target._history_state is state and target._history_time is None
    assert receipt['w0avg_moved'] is True
    assert len(receipt['slots_moved'])==34


def test_native_restart_reads_mskf_feedback_and_keeps_held_rain():
    from woof.ingest.wrfrst import _disposition,scratch_restore_plan
    from woof.io.wrf_output_schema import MSKF_OUTPUT_FIELDS
    cfg=SimpleNamespace(cu_physics=11,mp_physics=6)
    for key,schema in MSKF_OUTPUT_FIELDS.items():
        if key!='cu_act':assert _disposition(schema.netcdf_name,cfg).kind=='read'
    for key,source in [('cu_nca','NCA'),('cu_pratec','PRATEC'),('cu_raincv','RAINCV'),('cu_rucuten','RUCUTEN'),('cu_rvcuten','RVCUTEN')]:
        assert scratch_restore_plan(key,11)==('fill',source)
    assert scratch_restore_plan('cu_nca',6)[0]=='cold'
    # The merged common cloud-fraction producer owns the native field
    # even without an active MSKF adapter.
    assert _disposition('CLDFRA_DP',SimpleNamespace(cu_physics=0,mp_physics=6)).kind=='read'


@pytest.mark.parametrize('aercu,edrates',[(0,0),(0,1),(1,1)])
def test_native_restart_allocates_clouds_before_binding_w0avg(monkeypatch,aercu,edrates):
    import sys
    from woof.core.column_cumulus import MskfColumnCumulus
    from woof.ingest import wrfrst
    from woof.io import restart
    from woof.io.wrf_output_schema import MSKF_OUTPUT_FIELDS
    from woof.core.mskf import AEROSOL_OUTPUTS,EDRATE_OUTPUTS
    # Exercise the real adapter allocation and overlay with host arrays.
    monkeypatch.setitem(sys.modules,'cupy',SimpleNamespace(zeros=np.zeros,
        float32=np.float32,int32=np.int32,asarray=np.asarray))
    state=SimpleNamespace(w=np.zeros((4,2,3),'f4'),thp=np.zeros((3,2,3),'f4'))
    adapter=MskfColumnCumulus()
    inactive=(set(AEROSOL_OUTPUTS) if not aercu else set()) | (set(EDRATE_OUTPUTS) if not edrates else set())
    raw={schema.netcdf_name:np.full((3,2,3) if key in (*adapter.contract['outputs_3d'],'w0avg') else (2,3),i+1,'f4')
         for i,(key,schema) in enumerate(MSKF_OUTPUT_FIELDS.items()) if key!='cu_act' and key not in inactive}
    restored=SimpleNamespace(raw=raw,warm_start=SimpleNamespace(clock=SimpleNamespace(elapsed_seconds=900.)))
    monkeypatch.setattr(wrfrst,'_restored_state_reasons',lambda cfg:{})
    monkeypatch.setattr(restart,'state_manifest',lambda state:{})
    monkeypatch.setattr(restart,'_scratch_manifest',lambda state:{})
    monkeypatch.setattr(restart,'_driver_manifest',lambda driver:restart._cumulus_history_manifest(driver.cumulus_callable))
    monkeypatch.setattr(wrfrst,'_recompose_held_tendencies',lambda *a:None)
    monkeypatch.setattr(wrfrst,'_warm_start_driver_counters',lambda *a:None)
    monkeypatch.setattr(wrfrst,'_record_eos_checks',lambda *a:None)
    report=wrfrst.apply_warm_start_physics(state,SimpleNamespace(cumulus_callable=adapter),restored,SimpleNamespace(cu_physics=11,diff_opt=0,aercu_opt=aercu,kf_edrates=edrates,mean_diag=0,lightning_option=0))
    assert len(report.filled)==34-len(inactive)
    assert len(report.rebuilt)==1+len(inactive)
    for key,schema in MSKF_OUTPUT_FIELDS.items():
        if key in inactive:assert not getattr(adapter,key).any()
        elif key!='cu_act':np.testing.assert_array_equal(getattr(adapter,key),raw[schema.netcdf_name])
    adapter.ensure_trigger_history(state)
    np.testing.assert_array_equal(adapter.cldfra_dp,raw['CLDFRA_DP'])
    assert adapter._history_time==900. and not adapter.cu_act.any()


def test_physics_driver_builds_mskf_with_the_kf_family_contract():
    """woof.core.physics builds KF_FAMILY_CU_SCHEMES adapters as
    factory(scheme=..., kf_edrates=...); MSKF took only ``selector`` and every
    cu_physics=11 forecast died at physics init with a TypeError, 0 steps."""
    import importlib
    from woof.config import CU_STOCK_CALLABLES, KF_FAMILY_CU_SCHEMES
    assert 11 in KF_FAMILY_CU_SCHEMES
    module_name, _, class_name = CU_STOCK_CALLABLES[11].rpartition(".")
    factory = getattr(importlib.import_module(module_name), class_name)
    adapter = factory(scheme=11, kf_edrates=True)
    assert adapter.kf_edrates is True
    assert adapter.contract == factory().contract


@requires_gpu
def test_mskf_production_factory_uses_mskf_constructor():
    """The KF family contains independent adapters with different constructors."""
    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics
    from woof.core.column_cumulus import MskfColumnCumulus
    cfg = RunConfig(nx=3, ny=1, nz=16, dx=30000., dy=30000.,
                    ztop=9000., dt=60., run_seconds=0., moist=True,
                    mp_physics=6, cu_physics=11, kf_edrates=1)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 300. + .003*np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base,
                               lambda z: .010*np.exp(-np.asarray(z)/2200.))
    assert isinstance(initialize_physics(state, cfg).cumulus_callable, MskfColumnCumulus)
