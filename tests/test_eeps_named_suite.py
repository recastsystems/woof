from woof.physics_registry import physics_registry

SUITE='wsm6-eeps-mm5-noah-dudhia-sw-v1'

def test_eeps_named_composition_is_declared_on_its_supported_routes():
    r=physics_registry();v=r['templates'][SUITE]
    assert v['components']['pbl']=='eeps'
    assert v['components']['microphysics']=='wsm6-mp6'
    assert v['maturity']=='implemented-unverified'
    for name in ['tools.prepared_domain_tree_forecast','tools.prepared_single_domain_forecast']:
        lists=r['runner_routes'][name]['source_template_ids'].values()
        assert any(SUITE in x for x in lists)
    assert SUITE in r['runner_routes']['tools.hrrr_single_domain_benchmark']['refused_template_ids']


def test_named_eeps_suite_is_offered_and_resolved_at_the_prepared_door():
    from woof.physics_compat import single_domain_runtime_switches
    from woof.prepared_single_domain_forecast import runner_capabilities

    route = physics_registry()['runner_routes'][
        'tools.prepared_single_domain_forecast']
    capabilities = runner_capabilities()
    for source, declared in route['source_template_ids'].items():
        if SUITE in declared:
            assert SUITE in capabilities['source_profiles'][source][
                'physics_profile_ids']
    switches = single_domain_runtime_switches(SUITE)
    assert switches['bl_pbl_physics'] == 16
    assert switches['ra_sw_physics'] == 1
    assert switches['ra_lw_physics'] == 0
