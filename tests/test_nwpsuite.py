"""CPU acceptance checks for the declarative suite contract."""
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from woof.nwpsuite.cli import apply_what_if, initialize, write_suite
from woof.nwpsuite.hardware import box_from_check
from woof.nwpsuite.planner import plan_suite
from woof.nwpsuite.registry import DATA, ROOT, Registries, SuiteRefusal
from woof.nwpsuite.validator import load_suite, validate_suite

def example(count=2):
    return load_suite(ROOT / f'configs/suites/us-{count}box.toml')

def assert_timeline(plan):
    for box in plan['boxes']:
        for block in box['blocks']:
            assert block['card_hours'] <= block['budget_card_hours'] + 1e-7
            for intervals in block['lanes'].values():
                assert all(a[1] <= b[0]+1e-7 for a,b in zip(intervals,intervals[1:]))
                assert all(0<=a<b<=21600 for a,b in intervals)
    for job in plan['jobs']:
        assert job['end_s']<=job['deadline_s']+1e-7
        assert job['start_s']>=job['ready_offset_s']

@pytest.mark.parametrize('count,totals',[(2,[138.7333333333,131.2733333333]),(1,[138.2244444444])])
def test_budgets_and_all_card_lanes(count,totals):
    suite=example(count); plan=plan_suite(suite)
    assert plan['admitted'],plan['refusals']
    assert [b['daily_card_hours'] for b in plan['boxes']]==pytest.approx(totals)
    assert plan==plan_suite(suite)
    assert_timeline(plan)
    assert len([j for j in plan['jobs'] if j['row']=='cam-det'])==24
    assert all(j['end_s']<=j['deadline_s'] for j in plan['jobs'])

def test_projected_row_is_measurement_only():
    suite=example(); plan=plan_suite(suite)
    assert [s['row'] for s in plan['measurement_slots']]==['global-15-3']
    assert not plan['measurement_slots'][0]['launch_allowed']
    assert not any(j['row']=='global-15-3' for j in plan['jobs'])
    next(r for r in suite['system'] if r['id']=='global-15-3')['admit']='when-fits'
    plan=plan_suite(suite)
    assert not plan['admitted']
    assert any('projected' in r['reason'] for r in plan['refusals'])

def test_fine_candidate_refuses_with_numbers():
    suite=apply_what_if(example(),'rrfs-3km.enabled=true')
    plan=plan_suite(suite)
    refusal=next(r for r in plan['refusals'] if r['row']=='rrfs-3km')
    assert not plan['admitted'] and plan['exit_code']==2
    assert refusal['rate']=='lam-fine-8'
    assert refusal['required']>refusal['budget']
    assert refusal['overrun']>0

@pytest.mark.parametrize('mutation,text',[
    (lambda s:s.update(typo=True),'typo'),
    (lambda s:s['system'][0]['cycle'].update(lead_typo='1h'),'lead_typo'),
    (lambda s:s.update(region='missing-region'),'missing-region'),
    (lambda s:s['system'][0].update(recipe='missing-recipe'),'missing-recipe'),
    (lambda s:s['system'][0].update(ic=['missing-source']),'missing-source'),
    (lambda s:s['system'][0].update(mesh='missing-mesh'),'missing-mesh'),
    (lambda s:s['system'][0].update(physics={'missing_scheme':1}),'missing_scheme'),
    (lambda s:s['system'][0].update(rate='missing-rate'),'missing-rate'),
    (lambda s:s['system'][0].update(da={'method':'none','rate':'missing-da-rate'}),'missing-da-rate'),
    (lambda s:s['system'][0].update(shape='missing-shape'),'missing-shape'),
    (lambda s:s['system'][0].update(perturb={'recipe':'missing-ensemble-recipe'}),'missing-ensemble-recipe'),
    (lambda s:s['system'][0].update(perturb={'stochastic':['missing-stochastic']}),'missing-stochastic'),
])
def test_names_and_unknown_keys(mutation,text):
    suite=example(); mutation(suite)
    with pytest.raises(SuiteRefusal,match=text): validate_suite(suite)

def test_inheritance_and_per_hour_overrides():
    suite=example(); cam=next(r for r in suite['system'] if r['id']=='cam-ens')
    assert cam['recipe']=='hrrr_v4_gsd41' and cam['cycle']['lead']=='18h' and cam['at']=={}
    plan=plan_suite(suite)
    jobs=[j for j in plan['jobs'] if j['row']=='cam-det']
    assert [j['end_s']-j['start_s'] for j in jobs[:2]]==[6240,2340]
    assert all(j['dependencies'] for j in plan['jobs'] if j['row']=='cam-ens')

def test_extends_parent_changes_flow_to_child(tmp_path):
    path=tmp_path/'file.toml'
    path.write_text('schema="gpuwm.suite.v1"\nname="test"\nregion="US"\nextends="template:agency-2box"\n[[system]]\nid="cam-det"\nic=["gfs"]\n')
    suite=load_suite(path)
    assert next(r for r in suite['system'] if r['id']=='cam-ens')['ic']==['gfs']

def test_cycles_duplicates_and_bad_at():
    suite=example(); suite['system'][0]['from']=suite['system'][0]['id']
    with pytest.raises(SuiteRefusal,match='from cycle'): validate_suite(suite)
    suite=example(); suite['system'].append(deepcopy(suite['system'][0]))
    with pytest.raises(SuiteRefusal,match='duplicate'): validate_suite(suite)
    suite=example(); suite['system'][0]['at']={'00':{'ic':['bad-at-source']}}
    with pytest.raises(SuiteRefusal,match='bad-at-source'): validate_suite(suite)

def test_what_if_is_validated_and_changes_budget():
    suite=example(); base=plan_suite(suite)
    changed=apply_what_if(suite,'cam-ens.members.perturbed=5')
    plan=plan_suite(changed)
    assert plan['daily_card_hours']<base['daily_card_hours']
    with pytest.raises(SuiteRefusal,match='unknown'): apply_what_if(example(),'cam-ens.membres=7')
    with pytest.raises(SuiteRefusal,match='unknown'): apply_what_if(example(),'cam-det.cards.typo=7')

def test_memory_refuses_on_cpu_without_card():
    suite=initialize('PH'); suite['box'][0]['memory_mib']=1
    plan=plan_suite(suite)
    assert not plan['admitted']
    assert any(r['unit']=='MiB/card' and r['overrun']>0 for r in plan['refusals'])

def test_short_inventory_names_card_shortage():
    suite=initialize('PH')
    box=suite['box'][0]
    box['cards']=1
    box['inventory']=[{'index':7,'name':box['card_model'],'memory_mib':box['memory_mib']}]
    plan=plan_suite(suite)
    refusal=next(r for r in plan['refusals'] if r['unit']=='cards')
    assert not plan['admitted'] and plan['exit_code']==2
    assert refusal['budget']==1 and refusal['required']==2 and refusal['overrun']==1
    assert 'eligible card count' in refusal['message']

def test_generated_meshes_have_explicit_runtime_bindings():
    suite=example(); plan=plan_suite(suite)
    for mesh in ('patch48-3','x1.655362','x1.2621442'):
        jobs=[j for j in plan['jobs'] if j['phase']=='forecast' and j['settings'].get('mesh')==mesh]
        assert jobs
        for job in jobs:
            assert job['env']['WOOF_HEX_MESH_ROWS']=='{prepared.mesh_rows}'
            assert 'prepared.mesh_rows' in job['bindings'] and not job['execution_ready']
    row=next(r for r in suite['system'] if r.get('mesh')=='patch48-3')
    row.setdefault('prepared',{})['mesh_rows']='/prepared/rows.json'
    jobs=[j for j in plan_suite(suite)['jobs'] if j['row']==row['id']]
    assert jobs and all(j['env']['WOOF_HEX_MESH_ROWS']=='/prepared/rows.json' for j in jobs)

@pytest.mark.parametrize('key,value',[
    ('all_in_s_per_fh',None),('all_in_s_per_fh',0),('all_in_s_per_fh',True),
    ('startup_s',None),('startup_s',-1),('peak_mib_per_card',float('nan')),
    ('reservation_floor_s',-1),('host_memory_mib',None),('capacity_bound_mib',-1),
    ('cards',0),('cards',2.5),('cards',True),('source',''),
])
def test_invalid_rate_edit_refuses_named_key(tmp_path,key,value):
    data=json.loads((DATA/'rates.v1.json').read_text())
    data['rates'][0][key]=value
    path=tmp_path/'bad-rates.json'; path.write_text(json.dumps(data))
    suite=example(); suite['policy']['rates']=str(path)
    with pytest.raises(SuiteRefusal,match=re.escape(key)):
        plan_suite(suite)

def test_missing_rate_id_refuses_without_key_error(tmp_path):
    data=json.loads((DATA/'rates.v1.json').read_text())
    del data['rates'][0]['id']
    path=tmp_path/'bad-rates.json'; path.write_text(json.dumps(data))
    suite=example(); suite['policy']['rates']=str(path)
    with pytest.raises(SuiteRefusal,match='provenance'):
        plan_suite(suite)

def test_validate_resolves_rate_path_relative_to_suite(tmp_path):
    path=tmp_path/'suite.toml'
    shutil.copyfile(DATA/'rates.v1.json',tmp_path/'local-rates.json')
    path.write_text('schema="gpuwm.suite.v1"\nname="test"\nregion="US"\nextends="template:agency-2box"\n[policy]\nrates="local-rates.json"\n')
    suite=load_suite(path)
    assert plan_suite(suite,base=tmp_path)['admitted']
    assert plan_suite(apply_what_if(suite,'cam-ens.members.perturbed=5',base=tmp_path),base=tmp_path)['admitted']
    result=subprocess.run([sys.executable,'-m','woof','suite','plan',str(path),'--json','--what-if','cam-ens.members.perturbed=5'],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stderr+result.stdout
    assert json.loads(result.stdout)['admitted']
    rates=json.loads((tmp_path/'local-rates.json').read_text())
    rates['rates']=[row for row in rates['rates'] if row['id']!='global-fine-8']
    (tmp_path/'local-rates.json').write_text(json.dumps(rates))
    with pytest.raises(SuiteRefusal,match='global-fine-8'):
        load_suite(path)

def test_headroom_refuses_busy_block():
    suite=example(); suite['policy']['headroom']=.95
    plan=plan_suite(suite)
    assert not plan['admitted'] and any('headroom' in r['reason'] for r in plan['refusals'])

def test_override_member_selection_survives_to_door():
    suite=example(); suite['override']=[{'system':'cam-ens','set':{'members[3..5].physics.cu_physics':0}}]
    validate_suite(suite)
    plan=plan_suite(suite)
    selected=[j for j in plan['jobs'] if j['row']=='cam-ens' and j['cycle_hour']==0]
    assert len(selected)==6,plan['refusals']
    assert ['cu_physics' in j['settings'].get('physics',{}) for j in selected]==[False,False,False,True,True,True]

def test_unpriceable_scheme_change_refuses_instead_of_crashing():
    suite=example(); suite['override']=[{'system':'cam-ens','set':{'members[3..5].physics.mp':8}}]
    plan=plan_suite(suite)
    assert not plan['admitted']
    assert any(r['row']=='cam-ens' and 'CPU memory admission unavailable' in r['reason'] for r in plan['refusals'])
    assert plan['overrides']==suite['override']

def test_arbitrary_region_and_wheel_templates(tmp_path):
    target=tmp_path/'registries'; shutil.copytree(DATA,target)
    path=target/'regions.v1.json'; regions=json.loads(path.read_text())
    regions['regions']['ZZ_TEST']=deepcopy(regions['regions']['PH'])
    path.write_text(json.dumps(regions))
    suite=initialize('ZZ_TEST',registry_dir=target)
    output=tmp_path/'suite.toml'; write_suite(output,suite)
    plan=plan_suite(load_suite(output))
    assert plan['admitted'],plan['refusals']
    assert plan['region']=='ZZ_TEST'
    for template in ('agency-1box','agency-2box','agency-generic'):
        assert plan_suite(load_suite(DATA/'templates'/f'{template}.toml'))['admitted']

def test_hardware_receipt_uses_physical_indices(tmp_path):
    path=tmp_path/'box.json'; data=dict(host='test-host',gpuwm_version='2.8.9',host_ram_bytes=768*2**30,
        devices=[dict(index=i,uuid=str(i),name='NVIDIA RTX PRO 6000 Blackwell Server Edition',memory_total_bytes=97887*1048576) for i in range(3,11)])
    path.write_text(json.dumps(data))
    box=box_from_check(path,'a')
    assert box['cards']==8 and box['inventory'][0]['index']==3
    suite=initialize('PH',str(path)); plan=plan_suite(suite)
    assert plan['admitted'] and all(set(j['card_ids'])<=set(range(3,11)) for j in plan['jobs'])
    assert all(j['logical_card_ids']==list(range(len(j['card_ids']))) for j in plan['jobs'])
    job=next(j for j in plan['jobs'] if '--samehost-devices' in j['argv'])
    assert job['argv'][job['argv'].index('--samehost-devices')+1]=='0,1'
    assert '--samehost-layouts' in job['argv']
    path.write_text('{}')
    with pytest.raises(SuiteRefusal,match='no device inventory'): box_from_check(path,'a')

def test_generic_code_has_no_country_or_suite_branch():
    regions=json.loads((DATA/'regions.v1.json').read_text())['regions']
    suite_names=[p.stem for p in (DATA/'templates').glob('*.toml')]
    identifiers=[*regions,*suite_names,'us-1box','us-2box']
    for path in (ROOT/'woof/nwpsuite').glob('*.py'):
        code=path.read_text()
        for identifier in identifiers:
            assert not re.search(r'(?<![\w-])'+re.escape(identifier)+r'(?![\w-])',code),(path,identifier)

@pytest.mark.parametrize('arguments,code',[
    (['suite','validate','configs/suites/us-2box.toml','--json'],0),
    (['suite','plan','configs/suites/us-1box.toml','--json'],0),
    (['suite','plan','configs/suites/us-2box.toml','--json','--what-if','rrfs-3km.enabled=true'],2),
    (['suite','plan','configs/suites/us-2box.toml','--emit-commands'],0),
])
def test_normal_cli(arguments,code):
    result=subprocess.run([sys.executable,'-m','woof',*arguments],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==code,result.stderr+result.stdout[:2000]
    if '--json' in arguments: assert isinstance(json.loads(result.stdout),dict)
    else: assert 'woof run-plan' in result.stdout and 'woof hex forecast' in result.stdout

def test_bare_box_check_does_not_open_cuda():
    result=subprocess.run([sys.executable,'-m','woof','check','--json'],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    data=json.loads(result.stdout)
    assert data['schema']=='gpuwm.box.check.v1' and not data['readiness']['collected']
    assert 'no CUDA context' in data['basis']

def test_init_cli_and_what_if_json(tmp_path):
    output=tmp_path/'country.toml'
    run=subprocess.run([sys.executable,'-m','woof','suite','init','--region','PH','-o',str(output)],cwd=ROOT,text=True,capture_output=True)
    assert run.returncode==0,run.stderr
    run=subprocess.run([sys.executable,'-m','woof','suite','plan',str(output),'--json'],cwd=ROOT,text=True,capture_output=True)
    assert run.returncode==0,run.stderr
    assert json.loads(run.stdout)['region']=='PH'
    run=subprocess.run([sys.executable,'-m','woof','suite','plan','configs/suites/us-2box.toml','--json','--what-if','cam-ens.members.perturbed=5'],cwd=ROOT,text=True,capture_output=True)
    assert run.returncode==0,run.stderr
    assert json.loads(run.stdout)['what_if']['delta_daily_card_hours']<0

def test_references_in_region_and_unpriced_da_refuse(tmp_path):
    target=tmp_path/'data'; shutil.copytree(DATA,target)
    path=target/'regions.v1.json'; data=json.loads(path.read_text())
    data['regions']['PH']['verify']['surface']=['missing-network']; path.write_text(json.dumps(data))
    with pytest.raises(SuiteRefusal,match='missing-network'): initialize('PH',registry_dir=target)
    data['regions']['PH']['verify']['surface']=['PH__ASOS']
    data['regions']['PH']['profiles']['agency-generic']='missing-template'
    path.write_text(json.dumps(data))
    with pytest.raises(SuiteRefusal,match='missing-template'): initialize('PH',registry_dir=target)

def test_door_lead_alias_is_priced_before_placement():
    suite=example(); suite['override']=[{'system':'global-det','set':{'door.hours':24}}]
    validate_suite(suite); plan=plan_suite(suite)
    assert plan['admitted'],plan['refusals']
    jobs=[j for j in plan['jobs'] if j['row']=='global-det' and j['phase']=='forecast']
    assert all(j['settings']['cycle']['lead']=='24h' for j in jobs)
    assert all(j['argv'][j['argv'].index('--hours')+1]=='24.0' for j in jobs)
    assert plan['daily_card_hours']<plan_suite(example())['daily_card_hours']

def test_member_door_flag_reaches_distinct_argv():
    suite=example(); suite['override']=[{'system':'global-ens','set':{'members[3..5].door.out':'/scratch/selected'}}]
    validate_suite(suite); plan=plan_suite(suite)
    assert plan['admitted'],plan['refusals']
    jobs=[j for j in plan['jobs'] if j['row']=='global-ens' and j['cycle_hour']==0]
    assert [j['argv'][j['argv'].index('--out')+1]=='/scratch/selected' for j in jobs]==[i in (3,4,5) for i in range(21)]

def test_unknown_unpriced_and_placement_door_flags_refuse():
    suite=example(); suite['override']=[{'system':'global-det','set':{'door.not-a-flag':1}}]
    with pytest.raises(SuiteRefusal,match='not-a-flag'): validate_suite(suite)
    suite['override']=[{'system':'global-det','set':{'door.dt':'30'}}]
    validate_suite(suite); plan=plan_suite(suite)
    assert not plan['admitted'] and any('matching measured rate' in r['reason'] for r in plan['refusals'])
    suite['override']=[{'system':'global-det','set':{'door.samehost-devices':'7'}}]
    with pytest.raises(SuiteRefusal,match='reserved lanes'): validate_suite(suite)
    suite['override']=[{'system':'cam-det','set':{'door.no-memory-gate':True}}]
    with pytest.raises(SuiteRefusal,match='silent drops'): validate_suite(suite)

def test_cam_door_cards_alias_and_source_binding():
    suite=example(); suite['override']=[{'system':'cam-det','set':{'door.devices':4,'door.keep-checkpoints':2}}]
    validate_suite(suite); plan=plan_suite(suite)
    assert plan['admitted'],plan['refusals']
    job=next(j for j in plan['jobs'] if j['row']=='cam-det')
    assert job['argv'][:2]==['woof','go'] and '--keep-checkpoints' in job['argv']
    assert job['config_edits']['fetch.source']=='{selected.fetch.source}'
    assert job['source_selection']['ic']==['hrrr']
    # woof go forces a CAM row's boundaries from its one [fetch] source.
    assert job['source_selection']['lbc']==['hrrr']
    suite['override']=[{'system':'cam-ens','set':{'door.devices':1}}]
    validate_suite(suite); plan=plan_suite(suite)
    assert not plan['admitted'] and any('rate' in r['reason'] for r in plan['refusals'])

def test_override_lead_source_and_type_validation():
    suite=example(); suite['override']=[{'system':'global-det','set':{'door.init-source':'missing-source'}}]
    with pytest.raises(SuiteRefusal,match='missing-source'): validate_suite(suite)
    suite['override']=[{'system':'global-det','set':{'door.hours':'24'}}]
    with pytest.raises(SuiteRefusal,match='invalid door flag'): validate_suite(suite)
    suite['override']=[{'system':'global-ens','set':{'members[1].door.cycle-hours':24}}]
    with pytest.raises(SuiteRefusal,match='separate row'): validate_suite(suite)

def test_measured_door_flag_is_only_a_rate_table_edit(tmp_path):
    data=json.loads((DATA/'rates.v1.json').read_text())
    for row in data['rates']:
        if row['id']=='global-fine-8': row['door_flags']={'dt':'30'}
    path=tmp_path/'rates.json'; path.write_text(json.dumps(data))
    suite=example(); suite['policy']['rates']=str(path)
    suite['override']=[{'system':'global-det','set':{'door.dt':'30'}}]
    validate_suite(suite); plan=plan_suite(suite)
    assert plan['admitted'],plan['refusals']
    job=next(j for j in plan['jobs'] if j['row']=='global-det' and j['phase']=='forecast')
    assert job['argv'][job['argv'].index('--dt')+1]=='30'
    from woof.nwpsuite.overrides import argv_flags
    shape=Registries().shapes['global-uniform']
    assert argv_flags([],{'door_flags':{'fullstep-local-timestep':False}},shape)==['--no-fullstep-local-timestep']
    suite['override']=[{'system':'global-det','set':{'door.mesh':'missing-mesh'}}]
    with pytest.raises(SuiteRefusal,match='missing-mesh'): validate_suite(suite)


def test_every_suite_verb_has_a_help_line():
    # `woof suite --help` listed only `init`; the other nine doors had no line.
    import argparse
    from woof.nwpsuite.cli import register_cli

    root = argparse.ArgumentParser(prog="woof")
    register_cli(root.add_subparsers())
    suite = root._subparsers._group_actions[0].choices["suite"]
    verbs = suite._subparsers._group_actions[0]
    helped = {action.dest: action.help for action in verbs._choices_actions}
    assert set(helped) == {"init", "run", "status", "events", "halt", "resume",
                           "apply", "rehearse", "validate", "plan"}
    assert all(helped.values()), helped

def test_every_table_citation_resolves_inside_the_shipped_tree():
    """The suite tables ship in the public wheel. A citation of an out-of-tree
    report folder cannot be checked by a reader, and it carried internal
    folder names (and the private DA build row) into the public tables; the
    DA leak scan names one of them. Each citation is an in-tree file, a
    woof hex source line pinned by references.v1.json's hex_commit, or a
    receipt in receipts.v1.json, and every receipt is cited."""
    receipts = json.loads((DATA / 'receipts.v1.json').read_text(encoding='utf-8'))['receipts']
    anchor = 'woof/data/nwpsuite/receipts.v1.json#'
    cited = set()

    def check(value, where):
        if value.startswith(anchor):
            assert value[len(anchor):] in receipts, where
            cited.add(value[len(anchor):])
        elif not value.startswith(('hex@', 'tree/src/')):
            assert (ROOT / value.rsplit(':', 1)[0]).is_file(), (where, value)

    rates = json.loads((DATA / 'rates.v1.json').read_text(encoding='utf-8'))['rates']
    for row in rates:
        for key in ('source', 'build_source', 'memory_source', 'startup_source', 'reservation_basis'):
            if key in row:
                check(row[key], (row['id'], key))
    references = json.loads((DATA / 'references.v1.json').read_text(encoding='utf-8'))['registries']
    for group, rows in references.items():
        for name, row in rows.items():
            if isinstance(row, dict) and 'source' in row:
                check(row['source'], (group, name))
    for template in (DATA / 'templates').glob('*.toml'):
        for value in re.findall(re.escape(anchor) + r'[A-Za-z0-9.-]+', template.read_text(encoding='utf-8')):
            check(value, template.name)
    assert cited == set(receipts)
