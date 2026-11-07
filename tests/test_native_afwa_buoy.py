"""Native parcel-selector normalization and continuation admission."""
from woof.afwa_config import drop_inert
from woof.core import afwa_diagnostics as afwa
from test_restart import _cfg

def test_buoyancy_off_keeps_existing_afwa_continuation_identity():
    old={'afwa_diag_opt':1,'afwa_vil_opt':1}
    new={**old,'afwa_buoy_opt':0}
    drop_inert(old);drop_inert(new)
    assert old==new

def test_buoyancy_work_and_both_parcels_have_disjoint_lifetimes():
    cfg=_cfg(moist=True,mp_physics=8,afwa_diag_opt=1,afwa_buoy_opt=1)
    shapes=afwa.scratch_shapes(cfg)
    assert len(afwa.BUOY_HISTORY)==7
    assert all(shapes[k]==(2,cfg.nz,cfg.ny,cfg.nx) for k in afwa.BUOY_WORK[:3])
    assert shapes['afwa_buoy_meta']==(2,6,cfg.ny,cfg.nx)
    assert set(afwa.BUOY_HISTORY)<=set(afwa.PERSISTENT)
    assert not set(afwa.BUOY_WORK)&set(afwa.PERSISTENT)

def test_stock_buoyancy_selector_uses_native_any_domain_normalization():
    import importlib.util,tomllib
    from pathlib import Path
    from woof.fortran_namelist import parse_namelist_text
    from woof.namelist_import import import_parsed_namelists
    path=Path(__file__).parents[1]/'tools/diag_wrf471_oracle/census.py'
    spec=importlib.util.spec_from_file_location('buoy_census',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    fixture,_=m.fixture();inp=parse_namelist_text(fixture['INPUT_TEXT'])
    inp['afwa']={'afwa_buoy_opt':[0,1]}
    text,_=import_parsed_namelists(parse_namelist_text(fixture['WPS_TEXT']),inp,
        wps_path='namelist.wps',input_path='namelist.input',name='buoy',source_text=None)
    values=tomllib.loads(text)['shared']
    assert values['afwa_diag_opt']==values['afwa_buoy_opt']==1
