"""Admission preserves older documents and binds CAM controls to selectors 3/3."""
from dataclasses import replace
from datetime import datetime
import pytest
from woof.config import RunConfig,validate_run_config
from woof.experiment import experiment_from_run_config,domain_config_document
from woof.physics_registry import physics_registry

def cfg(**kw):
    return RunConfig(nx=12,ny=12,nz=48,dx=3000.,dy=3000.,ztop=20000.,dt=10.,run_seconds=60.,ra_lw_physics=3,ra_sw_physics=3,**kw)

def test_cam_metadata_admission_and_control_binding():
    c=validate_run_config(cfg())
    row=physics_registry()['components']['radiation']['options']['wrf-cam3']
    assert row['implemented'] and row['selectors']=={'ra_lw_physics':3,'ra_sw_physics':3}
    assert row['consumers']['stock_callable_class']=='woof.core.cam3.Cam3Radiation'
    doc=domain_config_document(experiment_from_run_config(c,datetime(2024,5,21)).root)['run']
    assert 'ghg_input' not in doc and c.ghg_input == 0
    assert doc['cam_abs_freq_s']==21600.
    older=domain_config_document(experiment_from_run_config(replace(c,ra_lw_physics=1,ra_sw_physics=1),datetime(2024,5,21)).root)['run']
    assert 'ghg_input' not in older and 'cam_abs_freq_s' not in older
    explicit=domain_config_document(experiment_from_run_config(replace(c,ra_lw_physics=4,ra_sw_physics=4,ghg_input=1),datetime(2024,5,21)).root)['run']
    assert explicit['ghg_input']==1

@pytest.mark.parametrize('overrides',[{'ghg_input':2},{'cam_abs_freq_s':-1}])
def test_cam_rejects_unimplemented_controls(overrides):
    with pytest.raises(ValueError):validate_run_config(replace(cfg(),**overrides))

@pytest.mark.parametrize('pair',[(3,3),(3,1),(1,3),(3,0)])
def test_cam_namelist_preserves_native_selectors_and_gas_controls(tmp_path,pair):
    import re,tomllib
    from test_namelist_import import INPUT_TEXT,WPS_TEXT
    from woof.namelist_import import import_namelists
    text=INPUT_TEXT
    for key,selector in zip(('ra_lw_physics','ra_sw_physics'),pair):
        text=re.sub(r'(?m)^(\s*'+key+r'\s*=)[^\n]+',r'\g<1> '+str(selector)+', '+str(selector)+',',text)
    text=text.replace('&physics','&physics\n ghg_input=1,\n cam_abs_freq_s=7200.,',1)
    inp=tmp_path/'namelist.input';wps=tmp_path/'namelist.wps';inp.write_text(text);wps.write_text(WPS_TEXT)
    output,_=import_namelists(wps,inp)
    parsed=tomllib.loads(output)
    assert f'ra_lw_physics = {pair[0]}' in output and f'ra_sw_physics = {pair[1]}' in output
    assert 'ghg_input = 1' in output and 'cam_abs_freq_s = 7200' in output


def test_absorption_schedule_matches_fixed_and_adaptive_source_boundaries():
    from woof.core.cam3 import absorption_due
    assert [absorption_due(t,15.,60.,100.,first=t==0) for t in (0,60,120,180,240)] == [True,False,True,False,True]
    assert absorption_due(99.,2.,60.,100.,first=False,adaptive=True)
    assert not absorption_due(101.,2.,60.,100.,first=False,adaptive=True)
    assert absorption_due(101.,2.,60.,0.,first=False,adaptive=True)
