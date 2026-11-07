"""Distribution coefficients, native address bounds and durable identity."""
from dataclasses import asdict, replace
import math
import pytest
from woof.config import RunConfig, validate_run_config
from woof.core.nssl2_shape import resolved_shapes
from woof.io.restart import _configuration_digest_values


def config(**kw):
    return RunConfig(nx=6,ny=6,nz=8,dx=1000.,dy=1000.,ztop=8000.,
                     dt=2.,run_seconds=4.,moist=True,mp_physics=18,**kw)


@pytest.mark.parametrize('alpha',[-.5,0.,1.,2.5,15.,20.])
def test_native_gamma_address_bounds_allow_more_than_lookup_clamp(alpha):
    c=config(nssl_alphah=alpha,nssl_alphahl=alpha)
    validate_run_config(c)
    assert resolved_shapes(c.nssl_alphah,c.nssl_alphahl)==(alpha,alpha)


@pytest.mark.parametrize('alpha',[-1.,-2.,math.inf,math.nan,23.,1e100])
def test_undefined_or_out_of_array_shape_refused(alpha):
    with pytest.raises(ValueError,match='nssl_alpha'):
        validate_run_config(config(nssl_alphah=alpha))


def test_default_shapes_preserve_legacy_checkpoint_digest():
    current=asdict(config());legacy=dict(current)
    legacy.pop('nssl_alphah');legacy.pop('nssl_alphahl')
    assert _configuration_digest_values(current)==_configuration_digest_values(legacy)


@pytest.mark.parametrize('field',['nssl_alphah','nssl_alphahl'])
def test_each_nondefault_shape_binds_checkpoint_identity(field):
    current=asdict(config());changed=asdict(config(**{field:2.5}))
    assert _configuration_digest_values(current)!=_configuration_digest_values(changed)


def test_namelist_import_reaches_both_shape_values(tmp_path):
    from test_namelist_import import WPS_TEXT,INPUT_TEXT
    from woof.namelist_import import import_namelists
    text=INPUT_TEXT.replace('mp_physics = 10, 10,','mp_physics = 18, 18,')
    text=text.replace('&physics\n','&physics\n nssl_alphah = 2.5,\n nssl_alphahl = 0.0,\n')
    wps,inp=tmp_path/'namelist.wps',tmp_path/'namelist.input'
    wps.write_text(WPS_TEXT);inp.write_text(text)
    toml,_=import_namelists(wps,inp)
    assert 'nssl_alphah = 2.5' in toml and 'nssl_alphahl = 0' in toml


@pytest.mark.parametrize('field,value',[('nssl_alphah',2.5),('nssl_alphahl',0.),('nssl_cccn',1e7)])
def test_raw_restart_omitted_defaults_resume_and_overrides_refuse(field,value):
    from woof.io.restart import _require_config_match,RestartMismatchError
    default=config();stored=asdict(default)
    for key in ('nssl_alphah','nssl_alphahl','nssl_cccn'):stored.pop(key)
    _require_config_match(stored,default,'omitted-native-defaults.npz')
    with pytest.raises(RestartMismatchError,match=field):
        _require_config_match(stored,replace(default,**{field:value}),'omitted-native-defaults.npz')


@pytest.mark.parametrize('field,default',[('nssl_alphah',0.),('nssl_alphahl',1.)])
@pytest.mark.parametrize('reverse',[False,True])
def test_prepared_legacy_shapes_tolerate_defaults_only(field,default,reverse):
    from woof.ingest.prepared_cache import compare_prepared_domain_config,undelayed_identity_defaults
    for value,accepted in [(default,True),(2.5,False)]:
        old={'run':{'mp_physics':18}};new={'run':{'mp_physics':18,field:value}}
        a,b=(new,old) if reverse else (old,new)
        tolerated,different=compare_prepared_domain_config(a,b,not_in_use=undelayed_identity_defaults(None))
        assert tolerated==(['run.'+field] if accepted else [])
        assert different==([] if accepted else ['run.'+field])
