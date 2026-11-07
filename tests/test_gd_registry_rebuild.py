def test_registry_rebuild_registers_gd_without_a_generated_seed_row():
    import copy
    from woof.physics_registry import physics_registry
    from tools.build_registry import build
    seed = copy.deepcopy(physics_registry())
    seed["components"]["cumulus"]["options"].pop("grell-devenyi", None)
    rebuilt = build(build(seed))
    row = rebuilt["components"]["cumulus"]["options"]["grell-devenyi"]
    assert row["selectors"] == {"cu_physics": 93}
    assert row["consumers"]["stock_callable_class"] == "woof.core.gf.GrellDevenyi"
    assert build(copy.deepcopy(rebuilt)) == rebuilt


def test_named_gd_composition_reaches_the_prepared_runner():
    from woof.physics_compat import GRELL_DEVENYI_PROFILE_ID, single_domain_runtime_switches
    from woof.prepared_single_domain_forecast import runner_capabilities
    from tools.hrrr_single_domain_benchmark import runner_capabilities as native_capabilities
    switches = single_domain_runtime_switches(GRELL_DEVENYI_PROFILE_ID)
    assert switches["cu_physics"] == 93
    assert switches["cu_diag"] == 1
    for source in runner_capabilities()["source_profiles"].values():
        if source["physics_profile_ids"]:
            assert GRELL_DEVENYI_PROFILE_ID in source["physics_profile_ids"]
    assert GRELL_DEVENYI_PROFILE_ID not in native_capabilities()["physics_profile_ids"]
