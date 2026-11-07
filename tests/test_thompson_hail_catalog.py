"""The hail selector has a named menu entry and coherent radiation overrides."""
import pytest
from woof import physics_catalog
import test_physics_combination_matrix as matrix

HAIL = "thompson-graupel-hail-mp38"


def test_hail_has_factual_catalog_words():
    words = physics_catalog.table()["descriptions"]["microphysics"][HAIL]
    assert words and "hail" in words.lower()


def test_hail_has_a_registered_composition():
    templates = matrix.REGISTRY["templates"]
    assert any(row["components"]["microphysics"] == HAIL
               for row in templates.values())


@pytest.mark.parametrize("microphysics", ["thompson-aerosol-mp28", HAIL])
@pytest.mark.parametrize("radiation", ["off", "dudhia-shortwave", "wrf-rrtm-dudhia"])
def test_component_override_replaces_the_templates_radiation_receipt(microphysics, radiation):
    choice = dict(matrix.anchor_components(), microphysics=microphysics, radiation=radiation)
    route_id = "tools.prepared_single_domain_forecast"
    plan = matrix.plan_for(route_id, matrix.REGISTRY["runner_routes"][route_id], "aifs", choice)
    report = matrix.validate_physics_plan(plan)
    assert report["launchable"], report["errors"]
    for row in report["resolved_domains"]:
        settings = row["settings"]
        assert settings["wrf_rrtmg_compatibility"] == "none"
        assert matrix.run_door(settings, nested=False) is None


@pytest.mark.parametrize("radiation", ["off", "dudhia-shortwave", "wrf-rrtm-dudhia"])
def test_explicit_domain_receipt_is_retained_and_refused_when_inconsistent(radiation):
    choice = dict(matrix.anchor_components(), microphysics=HAIL, radiation=radiation)
    route_id = "tools.prepared_single_domain_forecast"
    plan = matrix.plan_for(route_id, matrix.REGISTRY["runner_routes"][route_id], "aifs", choice)
    token = "wrf-rrtmg-4-4-legacy-v1"
    plan["domains"][0]["parameters"] = {"wrf_rrtmg_compatibility": token}
    report = matrix.validate_physics_plan(plan)
    settings = report["resolved_domains"][0]["settings"]
    assert settings["wrf_rrtmg_compatibility"] == token
    assert "requires the resolved 4/4 pair" in matrix.run_door(settings, nested=False)


def test_hail_override_replaces_the_fork_only_snow_setting():
    template = matrix.REGISTRY["templates"]["thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"]
    choice = dict(template["components"], microphysics=HAIL)
    route_id = "tools.prepared_domain_tree_forecast"
    plan = matrix.plan_for(route_id, matrix.REGISTRY["runner_routes"][route_id], "era5", choice)
    report = matrix.validate_physics_plan(plan)
    assert report["launchable"], report["errors"]
    for index, row in enumerate(report["resolved_domains"]):
        assert row["settings"]["thompson_version"] == "wrf_461"
        assert row["settings"]["thompson_fork_snow_fall"] == "blend"
        assert matrix.run_door(row["settings"], nested=index > 0) is None
