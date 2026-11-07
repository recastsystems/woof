"""The public doors must retain native sea-ice settings and require their data."""
from dataclasses import replace
import numpy as np
import pytest
from woof.config import RunConfig, validate_run_config

def config(**kw):
    return replace(RunConfig(nx=12,ny=12,nz=8,dx=3000.,dy=3000.,ztop=12000.,
        dt=6.,run_seconds=60.,sf_surface_physics=2,sf_sfclay_physics=91,
        bl_pbl_physics=1),**kw)

@pytest.mark.parametrize('land',[2,4])
@pytest.mark.parametrize('knobs',[
    dict(fractional_seaice=1),dict(seaice_albedo_opt=1),dict(seaice_albedo_opt=2),
    dict(seaice_thickness_opt=1),dict(seaice_snowdepth_opt=1)])
def test_noah_and_noahmp_keep_native_selectors(land,knobs,tmp_path):
    import tomllib
    from test_namelist_import import INPUT_TEXT,_pair
    from woof.namelist_import import import_namelists
    inp=INPUT_TEXT.replace('sf_surface_physics = 2, 2',
                           f'sf_surface_physics = {land}, {land}')
    inp=inp.replace(' bldt = 0, 0,',' num_soil_layers = 4,\n bldt = 0, 0,')
    inp=inp.replace('&physics','&physics\n'+''.join(
        f' {key} = {value},\n' for key,value in knobs.items()),1)
    text,report=import_namelists(*_pair(tmp_path,inp=inp))
    shared=tomllib.loads(text)['shared']
    assert all(shared[k]==v for k,v in knobs.items())
    assert not set(knobs).intersection(s.key for s in report.substitutions)
    validate_run_config(config(sf_surface_physics=land,**knobs))

@pytest.mark.parametrize('option,field',[
    ('seaice_albedo_opt','albsi'),('seaice_thickness_opt','icedepth'),
    ('seaice_snowdepth_opt','snowsi')])
def test_prescribed_options_require_their_native_input(option,field):
    from woof.core.seaice_options import attach_inputs
    cfg=config(**{option:2 if field=='albsi' else 1})
    def as_2d(value,shape,name):
        return np.broadcast_to(np.asarray(value,np.float32),shape)
    with pytest.raises(ValueError,match=field.upper()):
        attach_inputs({},cfg,{},(2,2),as_2d)
    result={}
    attach_inputs(result,cfg,{field:np.ones((2,2),np.float32)},(2,2),as_2d)
    assert np.array_equal(result[field],np.ones((2,2),np.float32))

def test_off_values_leave_restart_identity_unchanged():
    from woof.io.restart import _drop_default_off_run_keys
    names=('seaice_albedo_opt','seaice_thickness_opt','seaice_snowdepth_opt',
           'seaice_thickness_default','seaice_snowdepth_min','seaice_snowdepth_max')
    cfg=config();values={k:getattr(cfg,k) for k in names}
    _drop_default_off_run_keys(values)
    assert not values
    values={'seaice_albedo_opt':2};_drop_default_off_run_keys(values)
    assert values=={'seaice_albedo_opt':2}

@pytest.mark.parametrize("land", [2, 4])
def test_fractional_mm5_uses_noah_ice_parameters(monkeypatch, land):
    from types import SimpleNamespace
    from woof.core.physics import PhysicsDriver
    import woof.core.noah_fractional_seaice as fractional
    calls = []
    monkeypatch.setattr(fractional, "mm5", lambda *args: calls.append(args))
    driver = SimpleNamespace(fields={}, ruc_params=None)
    cfg = config(sf_surface_physics=land, sf_sfclay_physics=1, fractional_seaice=1)
    PhysicsDriver._run_sfclay(driver, {}, cfg)
    assert len(calls) == 1
