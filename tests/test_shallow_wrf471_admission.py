"""WRF shallow selectors reach the same native pairings at both doors."""
import numpy as np
import pytest


def deng_input(text):
    return text.replace('bl_pbl_physics = 11, 11','bl_pbl_physics = 2, 2').replace(
        'sf_sfclay_physics = 91, 91','sf_sfclay_physics = 2, 2')


def test_deng_domain_import_and_run_agree(tmp_path):
    from test_namelist_import import _import_with,_load
    text,_=_import_with(tmp_path,extra_physics=' shcu_physics = 5, 0,\n icloud_bl = 0, 1,\n',inp_filter=deng_input)
    exp=_load(tmp_path,text)
    assert (exp.domains[0].run.shcu_physics,exp.domains[0].run.icloud_bl)==(5,0)
    assert (exp.domains[1].run.shcu_physics,exp.domains[1].run.icloud_bl)==(0,1)


@pytest.mark.parametrize('cloud',['',' icloud_bl = 1, 1,\n'])
def test_deng_import_refuses_native_cloud_default(tmp_path,cloud):
    from test_namelist_import import _import_with
    with pytest.raises(ValueError,match='Deng requires explicit icloud_bl=0'):
        _import_with(tmp_path,extra_physics=' shcu_physics = 5, 0,\n'+cloud,inp_filter=deng_input)


def test_deng_integer_cloud_levels_have_device_bounds():
    from woof.core.health import gpu_integer_policy,collect_state_fields
    from types import SimpleNamespace
    fields={n:np.ones((2,2),np.int32) for n in ('ltopb','kdcldtop','kdcldbas')}
    state=SimpleNamespace(p=np.zeros((8,2,2),np.float32),physics=SimpleNamespace(fields=fields))
    for name in fields:
        assert gpu_integer_policy('surface.'+name,np.int32) is not None
    # Production collection obtains the domain-dependent upper bound.
    rows=collect_state_fields(state)
    for row in rows:
        if row.name.startswith('surface.') and row.name[8:] in fields:
            assert row.rule.lower==0 and row.rule.upper==8
