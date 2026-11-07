"""AFWA native clocks, domain normalization and real carrier lifetimes."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import numpy as np
import pytest
from woof.afwa_config import DEFAULTS, drop_inert
from woof.core import afwa_diagnostics as afwa
from woof.io import restart
from test_restart import _cfg, _shim_state, _fill_setup


def test_afwa_checkpoint_preserves_output_and_rebuilds_work(monkeypatch,tmp_path):
    cfg=_cfg(moist=True,afwa_diag_opt=1,afwa_therm_opt=1,afwa_ptype_opt=1,afwa_radar_opt=1,afwa_vil_opt=1,afwa_vis_opt=1,afwa_icing_opt=1,afwa_buoy_opt=1,mp_physics=16)
    src=_shim_state(cfg,monkeypatch);_fill_setup(src)
    expected={}
    for i,name in enumerate(afwa.PERSISTENT):
        x=src.existing_scratch(name)
        x[...] = np.arange(x.size,dtype='f4').reshape(x.shape)+np.float32(i+.25)
        expected[name]=x.tobytes()
    path=restart.write_restart(tmp_path/'afwa.npz',src,cfg)
    dst=_shim_state(cfg,monkeypatch);_fill_setup(dst)
    restart.restore_restart(path,dst,cfg)
    assert all(dst.existing_scratch(k).tobytes()==v for k,v in expected.items())
    with np.load(path,allow_pickle=False) as data:
        assert all('scratch/'+k in data.files for k in afwa.PERSISTENT)
        assert all('scratch/'+k not in data.files for k in afwa.WORK)


def test_native_restart_has_exact_sources():
    from woof.ingest import wrfrst
    for name in afwa.PERSISTENT:
        assert wrfrst.WRFRST_DISPOSITIONS[wrfrst.SCRATCH_SOURCES[name]].kind=='read'
        assert wrfrst.scratch_restore_plan(name,0)==('fill',wrfrst.SCRATCH_SOURCES[name])
    for name in afwa.WORK:
        assert wrfrst.scratch_restore_plan(name,0)[0]=='cold'


@pytest.mark.parametrize('now,epoch,expected',[(0,0,False),(3588,0,True),(3600,0,False),(1795,0,False),(5388,1800,True),(5400,1800,False)])
def test_alarm_uses_domain_start_and_end_of_step(now,epoch,expected):
    state=SimpleNamespace(elapsed_seconds=now,domain_start_offset=epoch)
    assert afwa.output_due(state,SimpleNamespace(output_interval_s=3600,dt=12))==expected


def test_off_identity_and_allocation_are_inert():
    values={'kept':3,**DEFAULTS};drop_inert(values);assert values=={'kept':3}
    assert afwa.scratch_shapes(_cfg())=={}


def test_native_any_domain_thermal_enables_all_domain_base():
    from woof.fortran_namelist import parse_namelist_text
    from woof.namelist_import import import_parsed_namelists
    import tomllib
    path=Path(__file__).parents[1]/'tools/diag_wrf471_oracle/census.py'
    spec=importlib.util.spec_from_file_location('afwa_census',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fixture,_=module.fixture();wps=parse_namelist_text(fixture['WPS_TEXT']);inp=parse_namelist_text(fixture['INPUT_TEXT'])
    inp['afwa']={'afwa_therm_opt':[0,1]}
    text,_=import_parsed_namelists(wps,inp,wps_path='namelist.wps',input_path='namelist.input',name='afwa',source_text=None)
    values=tomllib.loads(text)['shared']
    assert values['afwa_diag_opt']==values['afwa_therm_opt']==1


def test_dry_native_undefined_mslp_operand_is_named():
    from woof.afwa_config import validate
    with pytest.raises(ValueError,match='qvapr.*undefined'):
        validate(_cfg(afwa_diag_opt=1,moist=False))


@pytest.mark.parametrize('selector',tuple(afwa.PACKAGES))
def test_each_package_allocates_its_real_history_and_only_needed_profiles(selector):
    cfg=_cfg(moist=True,mp_physics=8,afwa_diag_opt=1,**({selector:1} if selector!='afwa_diag_opt' else {}))
    shapes=afwa.scratch_shapes(cfg)
    assert set(afwa.PACKAGES[selector]) <= set(shapes)
    profile=selector not in ('afwa_diag_opt','afwa_therm_opt','afwa_icing_opt')
    assert ('afwa_temperature' in shapes)==profile
    assert ('afwa_ze' in shapes)==(selector in ('afwa_radar_opt','afwa_vil_opt'))


def test_native_ptype_parameters_survive_stock_namelist_import():
    from woof.fortran_namelist import parse_namelist_text
    from woof.namelist_import import import_parsed_namelists
    import tomllib
    path=Path(__file__).parents[1]/'tools/diag_wrf471_oracle/census.py'
    spec=importlib.util.spec_from_file_location('afwa_parameter_census',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fixture,_=module.fixture()
    inp=parse_namelist_text(fixture['INPUT_TEXT'])
    inp['afwa']={'afwa_ptype_opt':[0,1],'afwa_ptype_ccn_tmp':[260.],'afwa_ptype_tot_melt':[55.]}
    text,_=import_parsed_namelists(parse_namelist_text(fixture['WPS_TEXT']),inp,wps_path='namelist.wps',input_path='namelist.input',name='afwa',source_text=None)
    values=tomllib.loads(text)['shared']
    assert values['afwa_diag_opt']==values['afwa_ptype_opt']==1
    assert values['afwa_ptype_ccn_tmp']==260. and values['afwa_ptype_tot_melt']==55.


def test_missing_native_hydrometeor_and_dust_couplings_are_named():
    from woof.afwa_config import validate
    with pytest.raises(ValueError,match='QR, QS and QG'):
        validate(_cfg(moist=True,mp_physics=1,afwa_diag_opt=1,afwa_radar_opt=1))
    with pytest.raises(ValueError,match='five-bin dust concentration'):
        validate(_cfg(moist=True,mp_physics=8,afwa_diag_opt=1,afwa_vis_opt=1,chem_sets='gocart'))
