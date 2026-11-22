"""Plan review and the run door decide the free per-domain settings alike.

The reachable-plan walk of tests/test_authority_agreement.py found 740 of
84,932 plans on the prepared-tree route (1,468 domains) that the two doors
decided differently (2026-11-12): bmj_rad_feedback without BMJ, obs_nudge_opt=1 with
the default max_obs=0, shalconv=0 outside SAS, icloud_bl=0, tke_budget=1
outside km_opt=2 or MYNN, and grav_settling on the mp_physics=28 templates,
where plan review's refusal carried a code the comparison did not know.
One plan per family, through both doors and the walk's own comparison, so
each case fails if its fix is taken away.  The whole walk stays in
test_authority_agreement.py.
"""

import pytest

from woof.config import validate_run_config
from woof.physics_registry import physics_registry, validate_physics_plan
from test_authority_agreement import _compare, _run_config, _tree_plan

ROUTE = "tools.prepared_domain_tree_forecast"


def _review(source, template, parameters):
    registry = physics_registry()
    plan = _tree_plan(registry, ROUTE, source, template,
                      parameters=parameters)
    report = validate_physics_plan(plan)
    found: list[str] = []
    _compare(f"{template} {parameters}", report, found, registry=registry)
    return report, found


@pytest.mark.parametrize("source,template,parameters,code,run_says", [
    ("hrrr", "morrison-aero-mp40-ysu-mm5-noah-rrtmg-legacy-v1",
     {"bmj_rad_feedback": True}, "bmj-feedback-dependency",
     "bmj_rad_feedback requires cu_physics=2"),
    ("hrrr", "wsm6-ysu-mm5-noah-nsas-rrtmg-legacy-v1",
     {"obs_nudge_opt": 1}, "observation-nudging-settings",
     "max_obs must be a positive integer"),
    ("hrrr", "thompson-aerosol-mp28-myj-eta-noah-rte-rrtmgp-v1",
     {"shalconv": 0}, "shalconv-consumer",
     "shalconv is consumed only by cu_physics=94"),
    ("hrrr", "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1",
     {"icloud_bl": 0}, "mynn-option-identity",
     "icloud_bl=0 is outside the admitted MYNN option identity"),
    ("hrrr", "ferrier-aligo-advect-mp15-ysu-mm5-noah-rrtmg-legacy-v1",
     {"tke_budget": 1}, "tke-budget-closure",
     "tke_budget=1 requires km_opt=2 or bl_pbl_physics=5"),
    ("era5", "thompson-aerosol-mp28-myj-eta-noah-rte-rrtmgp-v1",
     {"grav_settling": 1}, "fog-kfeta-pairing",
     "grav_settling>0 with mp_physics=28/29"),
])
def test_plan_review_refuses_what_the_run_refuses(source, template,
                                                  parameters, code, run_says):
    report, found = _review(source, template, parameters)
    assert found == []
    assert report["launchable"] is False
    assert code in {error["code"] for error in report["errors"]}
    for index, domain in enumerate(report["resolved_domains"]):
        with pytest.raises(ValueError, match=run_says):
            validate_run_config(_run_config(domain["settings"],
                                            nested=index > 0))


@pytest.mark.parametrize("template", [
    "goddard4ice-mp7-ysu-mm5-noah-rrtmg-legacy-v1",
    "thompson-mp8-ysu-mm5-noah-grell-devenyi-rte-rrtmgp-v1",
    "wsm6-pbl-off-mm5-noah-smagorinsky-3d-v1",
])
def test_icloud_bl_zero_runs_outside_mynn(template):
    """Outside MYNN no reader takes icloud_bl, so 0 runs, on both doors.

    The run door used to refuse it as "outside the admitted MYNN option
    identity" on domains with no MYNN, where no MYNN branch can be
    substituted; that refusal named no breakage and is gone.
    """
    report, found = _review("hrrr", template, {"icloud_bl": 0})
    assert found == []
    assert report["launchable"] is True, report["errors"]
    for index, domain in enumerate(report["resolved_domains"]):
        cfg = _run_config(domain["settings"], nested=index > 0)
        assert cfg.bl_pbl_physics != 5 and cfg.icloud_bl == 0
        assert validate_run_config(cfg) is cfg


@pytest.mark.parametrize("source,template", [
    ("hrrr", "wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1"),
    ("era5", "thompson-mp8-mynn-mynn-ruc-monthly-rrtmg-legacy-v1"),
])
def test_icloud_bl_zero_on_mynn_is_judged_on_the_generation_the_run_holds(
        source, template):
    """RunConfig runs icloud_bl=0 on default MYNN as WRF 4.7.1's MYNN.

    Plan review must judge the identity on that generation, not on the
    unresolved wrf_461: judged on the raw setting it refused 28 reachable
    plans (56 domains) the run starts.
    """
    report, found = _review(source, template, {"icloud_bl": 0})
    assert found == []
    assert report["launchable"] is True, report["errors"]
    for index, domain in enumerate(report["resolved_domains"]):
        cfg = _run_config(domain["settings"], nested=index > 0)
        assert cfg.bl_pbl_physics == 5 and cfg.bl_mynn_version == "wrf_471"
        assert validate_run_config(cfg) is cfg


def test_observation_nudging_with_a_capacity_is_decided_alike():
    """The obs law is shared, not a blanket refusal of obs_nudge_opt=1."""
    report, found = _review("hrrr", "wsm6-ysu-mm5-noah-nsas-rrtmg-legacy-v1",
                            {"obs_nudge_opt": 1, "max_obs": 150000})
    assert found == []
    assert "max_obs must be a positive integer" not in " ".join(
        error["message"] for error in report["errors"])


def test_plan_review_leaves_the_grid_question_to_the_run():
    """map_proj is a grid fact a plan does not have; only the run asks it."""
    from types import SimpleNamespace
    from woof.config import RunConfig
    from woof.fddaobs.config import validate, validate_settings

    cfg = RunConfig(nx=20, ny=20, nz=20, dx=3000.0, dy=3000.0, ztop=20000.0,
                    dt=6.0, run_seconds=60.0, obs_nudge_opt=1, max_obs=1000,
                    obs_twindo=0.6667, obs_rinxy=240.0)
    fields = {name: getattr(cfg, name) for name in cfg.__dataclass_fields__}
    fields["map_proj"] = 6
    view = SimpleNamespace(**fields)
    validate_settings(view)
    with pytest.raises(ValueError, match="regional WRF Lambert"):
        validate(view)
