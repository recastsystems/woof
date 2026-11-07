from dataclasses import replace
from pathlib import Path
import hashlib
import numpy as np
import pytest
from woof.config import RunConfig, validate_irrigation_config
from woof.core.kernels import module_source
from woof.table_assets import irrigation_static_manifest


def config(**kw):
    base = RunConfig(nx=8, ny=8, nz=5, dx=2000., dy=2000., ztop=8000., dt=10., run_seconds=60., moist=True, mp_physics=6, sf_surface_physics=2)
    return replace(base, **kw)


def test_irrigation_kernel_exists():
    source = module_source('irrigation')
    assert 'irrigation_column' in source


@pytest.mark.parametrize('option',[1,2,3])
def test_each_wrf_package_is_admitted(option):
    validate_irrigation_config(config(sf_surf_irr_scheme=option,irr_num_hours=4))


@pytest.mark.parametrize('kw,reason',[
    ({'sf_surf_irr_scheme':1,'sf_surface_physics':4,'irr_num_hours':4},'requires Noah'),
    ({'sf_surf_irr_scheme':2},'irr_num_hours'),
    ({'sf_surf_irr_scheme':3,'irr_num_hours':4,'irr_freq':0},'modulo divisor'),
    ({'sf_surf_irr_scheme':2,'irr_num_hours':4,'irr_ph':3},'undefined'),
])
def test_refusals_name_the_broken_calculation(kw,reason):
    base=RunConfig(nx=8,ny=8,nz=5,dx=2000.,dy=2000.,ztop=8000.,dt=10.,run_seconds=60.,moist=True,mp_physics=6,sf_surface_physics=2)
    with pytest.raises(ValueError,match=reason):
        validate_irrigation_config(replace(base,**kw))


@pytest.mark.parametrize('missing',['IRRIGATION','IRR_PH_FIELD'])
def test_missing_static_refuses_with_its_field_name(missing):
    cfg=config(sf_surf_irr_scheme=1,irr_num_hours=4)
    fields={'IRRIGATION':np.ones((8,8),np.float32),'IRR_PH_FIELD':np.zeros((8,8),np.int32)}
    fields.pop(missing)
    with pytest.raises(ValueError,match=missing):
        irrigation_static_manifest(cfg,fields)


def test_static_manifest_covers_runtime_words_and_changes_with_one_bit():
    cfg=config(sf_surf_irr_scheme=1,irr_num_hours=4)
    raw={'IRRIGATION':np.ones((8,8),np.float32),'IRR_PH_FIELD':np.zeros((8,8),np.int32)}
    fields,first=irrigation_static_manifest(cfg,raw)
    assert first['IRRIGATION']['sha256']==hashlib.sha256(fields['IRRIGATION'].tobytes()).hexdigest()
    raw['IRRIGATION'][0,0]=np.nextafter(np.float32(1),np.float32(0))
    _,second=irrigation_static_manifest(cfg,raw)
    assert first['IRRIGATION']['sha256']!=second['IRRIGATION']['sha256']


def test_zero_selector_needs_no_fields():
    assert irrigation_static_manifest(config(),{})==({}, {})


def test_metadata_owns_namelist_parameters():
    from woof.physics_registry import physics_registry
    from woof.wrf_namelist_registry import wrf_namelist_keys
    reg=physics_registry()
    for key in ('sf_surf_irr_scheme','irr_daily_amount','irr_start_hour','irr_num_hours',
                'irr_start_julianday','irr_end_julianday','irr_freq','irr_ph'):
        assert key in reg['parameters']
        assert reg['parameters'][key]['default']==getattr(config(),key)
        assert ('physics',key) in wrf_namelist_keys()


def test_native_irrigation_field_inventory_includes_dimensions():
    from woof.ingest.wrfinput import OPTIONAL_WRFINPUT, WRFINPUT_DIMENSIONS
    for key in ('IRRIGATION','IRR_PH_FIELD'):
        assert key in OPTIONAL_WRFINPUT
        assert WRFINPUT_DIMENSIONS[key]==('south_north','west_east')


@pytest.mark.parametrize('hours',[0,4])
def test_plan_review_uses_the_runtime_irrigation_denominator_rule(hours):
    from test_authority_agreement import (_permissive_registry,_single_domain_plan,
                                          _base_template_id,_PERMISSIVE_RUNNER)
    from woof.physics_registry import validate_physics_plan
    registry=_permissive_registry()
    registry['runner_routes'][_PERMISSIVE_RUNNER]['allowed_parameter_keys']=['sf_surf_irr_scheme','irr_num_hours']
    plan=_single_domain_plan(registry,_PERMISSIVE_RUNNER,'any-source',
        _base_template_id(registry),parameters={'sf_surf_irr_scheme':1,'irr_num_hours':hours})
    report=validate_physics_plan(plan,registry=registry)
    issues=[e for e in report['errors'] if e['code']=='irrigation-pairing']
    assert bool(issues)==(hours==0)


def test_phase_static_has_a_gpu_health_integer_policy():
    from woof.core.health import gpu_integer_policy,rule_for_field
    assert gpu_integer_policy('surface.irr_ph_field',np.int32)>0
    rule=rule_for_field('surface.irr_ph_field')
    assert rule.lower==0
    assert rule.upper==np.iinfo(np.int32).max


def test_calendar_casts_day_one_before_subtracting_it():
    from datetime import datetime
    from woof.table_assets import irrigation_calendar_scalars
    from woof.core.swint import calendar_scalars
    jul,_,_=irrigation_calendar_scalars(datetime(2024,1,1,2,24),0)
    assert jul==np.float32(0.10000002384185791)
    old,_,_=calendar_scalars(datetime(2024,1,1,2,24),0)
    assert jul.view(np.uint32)!=old.view(np.uint32)


def test_calendar_matches_the_public_wrf_date_leaf_fixture():
    from datetime import datetime
    from woof.table_assets import irrigation_calendar_scalars
    root=Path(__file__).parents[1]/'tools'/'irrigation_wrf461_oracle'/'fixtures'
    ref=np.loadtxt(root/'clock-oracle.txt')[:,1:].astype(np.float32)
    got=[]
    for line in (root/'clock-cases.txt').read_text().splitlines():
        _,start,_,elapsed=line.split()
        got.append(irrigation_calendar_scalars(datetime.strptime(start,'%Y-%m-%d_%H:%M:%S.%f'),float(elapsed)))
    assert np.array_equal(np.asarray(got,np.float32).view(np.uint32),ref.view(np.uint32))
