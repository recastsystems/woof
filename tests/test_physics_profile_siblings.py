"""Runner offers and published sibling templates use the same ordering."""
import pytest
from woof.physics_profile_siblings import PROFILE_SIBLINGS, with_profile_siblings
from woof.prepared_single_domain_forecast import (
    _EXPERT_PROFILE_IDS,
    _SOURCE_PHYSICS_PROFILES,
    _VERIFIED_SOURCE_PHYSICS_PROFILES,
)
from woof.physics_compat import COMPOSITION_SUITE_PROFILE_IDS
from woof.physics_registry import expert_template_ids_for_source, physics_registry


def test_siblings_do_not_create_a_first_offer_or_move_an_unrelated_profile():
    assert with_profile_siblings(()) == ()
    assert with_profile_siblings(('unrelated-a', 'unrelated-b')) == (
        'unrelated-a', 'unrelated-b')
    assert with_profile_siblings(('a', 'base', 'b'), siblings=(('base', 'new'),)) == (
        'a', 'base', 'new', 'b')


def test_chained_sibling_order_is_idempotent():
    base = PROFILE_SIBLINGS[0][0]
    expanded = with_profile_siblings(('first', base, 'last'))
    assert expanded == ('first', base, PROFILE_SIBLINGS[1][1],
                        PROFILE_SIBLINGS[2][1], PROFILE_SIBLINGS[0][1], 'last')
    assert with_profile_siblings(expanded) == expanded


def test_new_offers_preserve_every_previous_source_order_and_default():
    for source, verified in _VERIFIED_SOURCE_PHYSICS_PROFILES.items():
        old = (tuple(p for p in verified if p not in _EXPERT_PROFILE_IDS)
               + COMPOSITION_SUITE_PROFILE_IDS + _EXPERT_PROFILE_IDS) if verified else verified
        current = _SOURCE_PHYSICS_PROFILES[source]
        # The port line adds named suites; every earlier offer keeps its order.
        added = tuple(p for p in current if p not in old)
        assert set(added) <= {
            "lin-mp2-ysu-mm5-noah-rrtmg-legacy-implemented-unverified-v1",
            "morrison-mp10-ysu-mm5-slab-kf-rte-rrtmgp-v1"}, (source, added)
        assert tuple(p for p in current if p not in added) == old, source
        if old:
            assert current[0] == old[0], source


def test_published_single_domain_offers_match_every_live_source_menu():
    route = physics_registry()['runner_routes']['tools.prepared_single_domain_forecast']
    for source, offered in _SOURCE_PHYSICS_PROFILES.items():
        normal = list(route['source_template_ids'].get(source, ()))
        assert normal + expert_template_ids_for_source(route, source) == list(offered), source


def test_slab_sibling_resolves_five_layers_without_claiming_a_native_benchmark():
    from woof.physics_compat import single_domain_runtime_switches
    from woof.prepared_single_domain_forecast import SLAB_PHYSICS_PROFILE

    switches = single_domain_runtime_switches(SLAB_PHYSICS_PROFILE)
    assert (switches['sf_surface_physics'], switches['num_soil_layers']) == (1, 5)
    route = physics_registry()['runner_routes']['tools.hrrr_single_domain_benchmark']
    assert all(SLAB_PHYSICS_PROFILE not in offered
               for offered in route['source_template_ids'].values())


def test_native_siblings_keep_their_existing_namelist_contracts():
    from tools.hrrr_single_domain_benchmark import _native_hrrr_profile_contract

    route = physics_registry()['runner_routes']['tools.hrrr_single_domain_benchmark']
    for _, sibling in PROFILE_SIBLINGS:
        assert any(sibling in declared for declared in route['source_template_ids'].values())
        assert sibling not in route['refused_template_ids']
        assert _native_hrrr_profile_contract(sibling)['physics']['bl_pbl_physics'] == 5.0
    reason = route['refused_template_ids'][COMPOSITION_SUITE_PROFILE_IDS[2]]
    assert 'no native run of this composition' in reason
    assert 'namelist contract' in reason
    assert 'tools.prepared_single_domain_forecast' in reason
    assert 'tools.prepared_domain_tree_forecast' in reason


@pytest.mark.parametrize('profile', [sibling for _, sibling in PROFILE_SIBLINGS])
def test_existing_named_native_builder_paths_still_build_the_selected_run(profile):
    from dataclasses import replace
    from woof.experiment import VerticalConfig, build_experiment
    from woof.ingest.hrrr_target import HrrrTargetDomain
    from tools.hrrr_single_domain_benchmark import _experiment_tables

    target = replace(HrrrTargetDomain.legacy_500x500(), nx=50, ny=50, nz=12)
    vertical = VerticalConfig(eta_levels=tuple(1 - i / 12 for i in range(13)),
                              p_top=5000., hybrid_opt=2, etac=.2)
    raw, _ = _experiment_tables(vertical, target=target, run_seconds=60,
                                physics_profile=profile)
    cfg = build_experiment(raw, source='named native builder control').root.run
    assert (cfg.bl_pbl_physics, cfg.sf_sfclay_physics, cfg.sf_surface_physics) == (5, 5, 3)
    assert cfg.ra_rrtmg_variant == 'rrtmg_legacy'
    assert cfg.mp_physics == (28 if profile == PROFILE_SIBLINGS[0][1] else 8)
    # The named source builder selects solar albedo for all three profiles.
    assert cfg.alb_sol == 1
