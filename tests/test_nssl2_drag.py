"""Native integer coefficients remain durable without a fabricated 0..6 range."""
from dataclasses import asdict,replace
import pytest
import numpy as np
from test_nssl2_shapes import config
from woof.config import validate_run_config
from woof.core.nssl2_shape import resolved_shapes,resolved_drags,shape_defines
from woof.io.restart import _require_config_match,RestartMismatchError,_configuration_digest_values

@pytest.mark.parametrize('pair',[(6,6),(2,4),(0,-1),(-2147483648,-2147483648)])
def test_native_integer_coefficient_words(pair):
    assert resolved_drags(*pair)==pair
    c=config(nssl_icdx=pair[0],nssl_icdxhl=pair[1]);validate_run_config(c)
    coefficients=resolved_shapes(0.,1.,*pair)
    assert coefficients==((0.,1.) if pair==(6,6) else (0.,1.,*pair))
    definitions=dict(shape_defines(coefficients))
    if pair!=(6,6):
        assert definitions['NSSL2_DRAG_G_WORD']==(pair[0]&0xffffffff)+1
        assert definitions['NSSL2_DRAG_H_WORD']==(pair[1]&0xffffffff)+1

@pytest.mark.parametrize('value',[True,1.5,6.,2**31,-2**31-1,'2'])
def test_non_native_integer_refused(value):
    with pytest.raises(ValueError,match='INTEGER'):validate_run_config(config(nssl_icdx=value))

@pytest.mark.parametrize('field',['nssl_icdx','nssl_icdxhl'])
def test_raw_restart_default_implicit_and_changed_law_refused(field):
    c=config();stored=asdict(c)
    stored.pop('nssl_icdx');stored.pop('nssl_icdxhl')
    _require_config_match(stored,c,'legacy-native-drag-defaults.npz')
    assert _configuration_digest_values(stored)==_configuration_digest_values(asdict(c))
    changed=replace(c,**{field:4})
    assert _configuration_digest_values(asdict(changed))!=_configuration_digest_values(stored)
    with pytest.raises(RestartMismatchError,match=field):_require_config_match(stored,changed,'changed-native-drag.npz')

@pytest.mark.parametrize('field',['nssl_icdx','nssl_icdxhl'])
@pytest.mark.parametrize('reverse',[False,True])
def test_prepared_drag_defaults_only(field,reverse):
    from woof.ingest.prepared_cache import compare_prepared_domain_config,undelayed_identity_defaults
    for value,accepted in [(6,True),(4,False)]:
        old={'run':{'mp_physics':18}};new={'run':{'mp_physics':18,field:value}}
        a,b=(new,old) if reverse else (old,new)
        tolerated,different=compare_prepared_domain_config(a,b,not_in_use=undelayed_identity_defaults(None))
        assert tolerated==(['run.'+field] if accepted else [])
        assert different==([] if accepted else ['run.'+field])

def test_stock_namelist_import_reaches_both_drag_laws(tmp_path):
    from test_namelist_import import WPS_TEXT,INPUT_TEXT
    from woof.namelist_import import import_namelists
    text=INPUT_TEXT.replace('mp_physics = 10, 10,','mp_physics = 18, 18,')
    text=text.replace('&physics\n','&physics\n nssl_icdx = 2,\n nssl_icdxhl = 4,\n')
    wps,inp=tmp_path/'namelist.wps',tmp_path/'namelist.input'
    wps.write_text(WPS_TEXT);inp.write_text(text)
    toml,_=import_namelists(wps,inp)
    assert 'nssl_icdx = 2' in toml and 'nssl_icdxhl = 4' in toml

@pytest.mark.parametrize('pair',[(7,6),(6,2),(6,7),(2147483647,6),(7,2),(2,2)])
def test_native_slab_cd_carry_admitted(pair):
    from woof.core.nssl2_shape import requires_cd_carry
    assert resolved_drags(*pair)==pair
    validate_run_config(config(nssl_icdx=pair[0],nssl_icdxhl=pair[1]))
    assert requires_cd_carry(resolved_shapes(0.,1.,*pair))


def test_gathered_cd_scratch_is_priced_and_rebuilt():
    from woof.core.preflight import scratch_slot_registry
    from woof.io.restart import REBUILT_SCRATCH_SLOTS
    default=config()
    carry=config(nssl_icdx=7,nssl_icdxhl=2)
    assert 'nssl2_drag_cd_carry' not in scratch_slot_registry(default)
    assert scratch_slot_registry(carry)['nssl2_drag_cd_carry']==(7,9,6,6)
    assert 'nssl2_drag_cd_carry' in REBUILT_SCRATCH_SLOTS


@pytest.mark.parametrize('hail,planes', [(0,22), (2,27)])
def test_one_moment_prices_only_its_allocated_workspace(hail, planes):
    from woof.core.preflight import scratch_slot_registry
    one = config(nssl_2moment_on=0, nssl_hail_on=hail,
                 nssl_density_on=1 if hail else 0,
                 nssl_icdx=7, nssl_icdxhl=2)
    slots = scratch_slot_registry(one)
    assert slots['nssl1_driver_state'] == (planes,8,6,6)
    assert not any(n in slots for n in ('nssl2_drag_cd_carry',
        'nssl2_driver_state', 'nssl2_nucond_ss'))


def test_cd_carry_reusable_workspace_metadata():
    from woof.core.nssl2_driver_support import (
        NSSL2DriverWorkspace, validate_nssl2_driver_workspace)
    from dataclasses import replace
    shape = (4, 2, 3)
    workspace = NSSL2DriverWorkspace(
        state=np.zeros((16, *shape), np.float32),
        category_surface_export=np.zeros((5, 2, 3), np.float32),
        ignored_accumulator=np.zeros((2, 3), np.float32), shape=shape,
        drag_cd=np.empty((7, 5, 2, 3), np.float32))
    validate_nssl2_driver_workspace(workspace, shape)
    with pytest.raises(ValueError, match='workspace.drag_cd must have shape'):
        validate_nssl2_driver_workspace(replace(workspace,
            drag_cd=np.empty((7, 4, 2, 3), np.float32)), shape)
