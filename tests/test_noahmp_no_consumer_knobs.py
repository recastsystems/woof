"""All Noah-MP output selectors now have a native consumer at the real doors."""
from pathlib import Path
import importlib,sys
import pytest
from woof.config import RunConfig,validate_run_config,NOAHMP_OPTIONS_WITHOUT_CONSUMER

def _sibling(name):
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    return importlib.import_module(name)

def _noahmp_cfg(**kw):
    return RunConfig(nx=12,ny=12,nz=8,dx=3000.,dy=3000.,ztop=12000.,dt=6.,run_seconds=60.,moist=True,sf_surface_physics=4,sf_sfclay_physics=1,bl_pbl_physics=1,**kw)

def test_old_declared_only_rows_have_consumers():
    assert not NOAHMP_OPTIONS_WITHOUT_CONSUMER

@pytest.mark.parametrize('level',[0,1,2,3])
def test_output_level_at_both_real_doors(level,tmp_path,capsys):
    validate_run_config(_noahmp_cfg(noahmp_output=level,noahmp_acc_dt=60.))
    toml,report=_sibling('test_namelist_import')._import_with(tmp_path,extra_input=f'&noah_mp\n noahmp_output={level}, noahmp_acc_dt=60.,\n/\n')
    rows={(r.section,r.key):r for r in report.fixed}
    assert rows['noah_mp','noahmp_output'].fixed_value==level
    assert rows['noah_mp','noahmp_acc_dt'].fixed_value==60.
    assert 'reaches no woof code' not in capsys.readouterr().err

@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf'),'60',True])
def test_invalid_accumulation_interval_names_the_field(value):
    with pytest.raises(ValueError,match='noahmp_acc_dt|where Noah-MP takes a float'):
        validate_run_config(_noahmp_cfg(noahmp_acc_dt=value))

@pytest.mark.parametrize('name,value',[('opt_run',9),('opt_crop',3),('noahmp_output',4)])
def test_missing_branch_still_refuses(name,value):
    with pytest.raises(ValueError,match='outside the admitted Noah-MP'):
        validate_run_config(_noahmp_cfg(**{name:value}))

@pytest.mark.parametrize('name,value',[('opt_run',9),('opt_crop',3),('noahmp_output',4)])
def test_importer_refuses_an_invalid_native_option(name,value,tmp_path):
    with pytest.raises(ValueError,match=name):
        _sibling('test_namelist_import')._import_with(tmp_path,extra_input=f'&noah_mp\n {name}={value},\n/\n')

@pytest.mark.parametrize('name,value',[('opt_run',9),('opt_crop',3),('noahmp_output',4)])
def test_registry_refuses_an_invalid_native_option(name,value):
    m=_sibling('test_physics_registry');plan=m._single_plan(m.NOAHMP_PROFILE_ID)
    plan['domains'][0]['parameters']={name:value}
    from woof.physics_registry import validate_physics_plan
    result=validate_physics_plan(plan)
    assert any(name in r['message'] for r in result['errors']),result

@pytest.mark.parametrize('level',[0,1,2,3])
def test_registry_admits_same_native_output_levels(level):
    m=_sibling('test_physics_registry');plan=m._single_plan(m.NOAHMP_PROFILE_ID)
    plan['domains'][0]['parameters']={'noahmp_output':level,'noahmp_acc_dt':60.}
    from woof.physics_registry import validate_physics_plan
    result=validate_physics_plan(plan)
    assert not [r for r in result['errors'] if any(k in r['message'] for k in ('noahmp_output','noahmp_acc_dt'))],result['errors']

# Release census replacements (reviewed against the merged sources).
# Census: test_a_knob_whose_other_branch_was_never_transcribed_is_still_refused is covered by test_missing_branch_still_refuses.
# Census: test_a_knob_with_no_consumer_runs_and_says_it_reaches_nothing is covered by test_output_level_at_both_real_doors.
# Census: test_plan_review_does_not_refuse_a_knob_with_no_consumer is covered by test_registry_admits_same_native_output_levels.
# Census: test_plan_review_still_refuses_the_knob_whose_branch_is_missing is covered by test_registry_refuses_an_invalid_native_option.
# Census: test_the_admitted_value_of_a_no_consumer_knob_says_nothing is covered by test_output_level_at_both_real_doors.
# Census: test_the_importer_agrees_with_the_run_door_about_the_same_knob is covered by test_output_level_at_both_real_doors.
# Census: test_the_importer_reads_the_verdict_from_the_config_table is covered by test_output_level_at_both_real_doors.
# Census: test_the_importer_still_refuses_the_knob_whose_branch_is_missing is covered by test_importer_refuses_an_invalid_native_option.
# Census: test_the_no_consumer_set_is_derived_from_the_evidence_it_cites is covered by test_old_declared_only_rows_have_consumers.
# Census: test_the_wrong_type_is_still_refused_and_names_the_type is covered by test_invalid_accumulation_interval_names_the_field.
