"""Native AFWA threshold failure reaches the existing deferred health drain."""
from types import SimpleNamespace
import numpy as np
import pytest
from woof.core import afwa_diagnostics as afwa

def test_literal_native_first_time_and_restart_alarm():
    cfg=SimpleNamespace(afwa_bad_data_check=1,dt=12.,output_interval_s=24.)
    state=SimpleNamespace(elapsed_seconds=12.,domain_start_offset=0.)
    assert not afwa.bad_check_due(state,cfg)
    state.elapsed_seconds=36.
    assert afwa.bad_check_due(state,cfg)
    state.elapsed_seconds=108.;state.domain_start_offset=96.
    assert afwa.bad_check_due(state,cfg)

@pytest.mark.gpu
def test_device_native_thresholds_and_deferred_failure():
    import cupy as cp
    from woof.core.health_ledger import HealthLedger,deferring
    cfg=SimpleNamespace(afwa_bad_data_check=1,dt=1.,output_interval_s=1.,nx=4,ny=1,nz=1)
    arrays={'afwa_u':cp.asarray([300.,-300.,np.nan,0.],dtype='f4'),
            'afwa_v':cp.asarray([-300.,300.,0.,np.nan],dtype='f4'),
            'afwa_theta':cp.asarray([99.,9999.,300.,300.],dtype='f4'),
            'afwa_bad_status':cp.zeros(4,dtype='f4')}
    state=SimpleNamespace(elapsed_seconds=2.,domain_start_offset=0.,existing_scratch=arrays.get)
    ledger=HealthLedger()
    with deferring(ledger):afwa.check_bad_data(state,cfg)
    ledger.drain()
    for key,value in (('afwa_u',301.),('afwa_u',-301.),('afwa_v',301.),('afwa_v',-301.),('afwa_theta',98.),('afwa_theta',10000.)):
        old=arrays[key][0].copy();arrays[key][0]=value
        with deferring(ledger):afwa.check_bad_data(state,cfg)
        with pytest.raises(FloatingPointError,match='AFWA Diagnostics: model winds'):
            ledger.drain()
        arrays[key][0]=old

def test_scalar_check_admission_preserves_native_main_package_gate():
    from pathlib import Path
    import importlib.util,tomllib
    from woof.fortran_namelist import parse_namelist_text
    from woof.namelist_import import import_parsed_namelists
    path=Path(__file__).parents[1]/'tools/diag_wrf471_oracle/census.py'
    spec=importlib.util.spec_from_file_location('check_census',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fixture,_=module.fixture()
    def imported(controls):
        inp=parse_namelist_text(fixture['INPUT_TEXT']);inp['afwa']=controls
        text,_=import_parsed_namelists(parse_namelist_text(fixture['WPS_TEXT']),inp,
            wps_path='namelist.wps',input_path='namelist.input',name='afwa-check',source_text=None)
        return tomllib.loads(text)['shared']
    assert 'afwa_diag_opt' not in imported({'afwa_bad_data_check':[1]})
    values=imported({'afwa_therm_opt':[0,1],'afwa_bad_data_check':[2]})
    assert values['afwa_bad_data_check']==2 and values['afwa_diag_opt']==1

@pytest.mark.gpu
@pytest.mark.parametrize('finish',['generic','classic','adapter'])
@pytest.mark.parametrize('no_heating',[0,1])
def test_native_check_keeps_raw_theta_before_heating_limit(finish,no_heating):
    import cupy as cp
    from woof.core.microphysics import moist_physics_finish
    from woof.core.thompson_classic import launch_classic_finish
    from woof.core.thompson import launch_adapter_finish
    cfg=SimpleNamespace(afwa_diag_opt=1,afwa_bad_data_check=1,dt=1.,
        output_interval_s=1.,mp_tend_lim=.1,no_mp_heating=no_heating)
    shape=(1,1,1);raw=cp.full(shape,10000.,dtype='f4')
    held=cp.zeros(shape,dtype='f4')
    state=SimpleNamespace(thp=cp.zeros(shape,dtype='f4'),
        h_diabatic=cp.full(shape,300.,dtype='f4'),elapsed_seconds=2.,
        domain_start_offset=0.,existing_scratch=lambda key:held)
    saved=state.h_diabatic.copy()
    if finish=='generic':moist_physics_finish(state,cfg,raw,1.)
    elif finish=='classic':launch_classic_finish(raw,state,saved,cfg,1.,size=1)
    else:
        surface=cp.zeros((1,1),dtype='f4');theta=cp.empty_like(raw)
        launch_adapter_finish(raw,cp.ones_like(raw),theta,state.thp,
            state.h_diabatic,surface,surface,surface,surface,cfg,1.)
        afwa.capture_microphysics_theta(state,cfg,theta)
    assert float(held.item())==10000.
    assert float(state.thp.item())==pytest.approx(0. if no_heating else .1)

@pytest.mark.gpu
@pytest.mark.parametrize('full_base',[False,True])
def test_native_check_excluded_ring_uses_restored_pre_mp_theta(full_base):
    import cupy as cp
    from woof.core.physics_inventory import spec_zone_ring_slices
    shape=(2,4,5);held=cp.full(shape,10000.,dtype='f4')
    base=cp.full(shape if full_base else (2,),300.,dtype='f4')
    state=SimpleNamespace(thp=cp.ones(shape,dtype='f4'),thb=base,
        elapsed_seconds=2.,domain_start_offset=0.,existing_scratch=lambda key:held)
    cfg=SimpleNamespace(afwa_diag_opt=1,afwa_bad_data_check=1,dt=1.,output_interval_s=1.)
    afwa.restore_theta_ring(state,cfg,spec_zone_ring_slices(4,5,1))
    actual=cp.asnumpy(held);expected=np.full(shape,301.,dtype='f4')
    expected[:,1:-1,1:-1]=10000.
    assert np.array_equal(actual,expected)
