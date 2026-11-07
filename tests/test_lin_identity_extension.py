from copy import deepcopy
from woof.physics_registry import physics_registry, registry_physics_part_sources, registry_physics_parts

def selected(mp):
    return {'microphysics':mp,'pbl':'mynn'}

def legacy_registry(registry):
    old=deepcopy(registry)
    old['components']['microphysics']['options'].pop('lin-mp2')
    for policy in old['transitions'].values():
        policy['cross_options']=[r for r in policy['cross_options'] if 'lin-mp2' not in (r['parent_option_id'],r['child_option_id'])]
    old['components']['pbl']['options']['mynn']['extensions']['supplied_moisture_species']['flag_qs_true_microphysics_selectors'].remove(2)
    return old

def test_unused_additive_option_retains_established_transition_and_species_parts():
    current=physics_registry();old=legacy_registry(current)
    now=registry_physics_parts(current,options=selected('wsm6-mp6'))
    prior=registry_physics_parts(old,options=selected('wsm6-mp6'))
    assert now['transitions']==prior['transitions']
    assert now['components.pbl.options.mynn']==prior['components.pbl.options.mynn']

def test_selected_new_option_binds_its_full_transition_and_snow_contract():
    current=physics_registry()
    parts=registry_physics_part_sources(current,options=selected('lin-mp2'))
    # 110 rows less the 20 touching WSM7 (identity extension, not selected here).
    assert len(parts['transitions']['microphysics-one-way-v1']['cross_options'])==90
    assert 2 in parts['components.pbl.options.mynn']['extensions']['supplied_moisture_species']['flag_qs_true_microphysics_selectors']

def test_changed_established_edge_and_classification_are_not_ignored():
    current=physics_registry();changed=deepcopy(current)
    row=next(r for r in changed['transitions']['microphysics-one-way-v1']['cross_options'] if 'lin-mp2' not in (r['parent_option_id'],r['child_option_id']))
    row['required_child_settings']['moist']=False
    changed['components']['pbl']['options']['mynn']['extensions']['supplied_moisture_species']['flag_qs_true_microphysics_selectors'].remove(6)
    before=registry_physics_parts(current,options=selected('wsm6-mp6'));after=registry_physics_parts(changed,options=selected('wsm6-mp6'))
    assert before['transitions']!=after['transitions']
    assert before['components.pbl.options.mynn']!=after['components.pbl.options.mynn']

def test_whole_registry_keeps_every_declared_transition():
    # 110 rows through 2.9.12, plus the 22 SBU-YLin (mp13) one-way rows the 3.0
    # port added; every earlier row is unchanged (additive, measured 2026-11-02).
    assert len(registry_physics_part_sources(physics_registry())['transitions']['microphysics-one-way-v1']['cross_options'])==132
