"""Cloud feedback review must match the selected radiation consumer."""
import pytest
from woof.config import validate_run_config
from woof.physics_registry import physics_registry, validate_physics_plan
from test_authority_agreement import _tree_plan, _run_config

@pytest.mark.parametrize("template,feedback,accepted,radiation", [
    ("wsm6-ysu-mm5-noah-nsas-rrtmg-legacy-v1", False, True, None),
    ("wsm6-ysu-mm5-noah-nsas-rrtmg-legacy-v1", True, True, None),
    ("thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1", True, True, None),
    ("wsm6-ysu-mm5-noah-nsas-rrtmg-legacy-v1", True, True, "new-goddard"),
    ("wsm6-ysu-mm5-noah-nsas-rrtmg-legacy-v1", True, True, "off"),
    ("wsm6-ysu-mm5-noah-nsas-rrtmg-legacy-v1", True, False, "analytic-clear-sky"),
])
def test_bmj_feedback_plan_and_runtime_agree(template, feedback, accepted, radiation):
    registry = physics_registry()
    components = {"cumulus": "betts-miller-janjic"}
    if radiation is not None:
        components["radiation"] = radiation
    plan = _tree_plan(registry, "tools.prepared_domain_tree_forecast", "gfs",
                      template, components=components,
                      parameters={"bmj_rad_feedback": feedback})
    report = validate_physics_plan(plan)
    assert report["launchable"] is accepted, report["errors"]
    for domain in report["resolved_domains"]:
        cfg = _run_config(domain["settings"], nested=False)
        if accepted:
            assert validate_run_config(cfg) is cfg
        else:
            assert "component-conditional-refusal" in {e["code"] for e in report["errors"]}
            with pytest.raises(ValueError, match="does not merge the carried convective cloud"):
                validate_run_config(cfg)
