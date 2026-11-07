"""Regression for the two formerly reversed WRF 4.7.1 option doors."""
import re
import tomllib
import pytest
from woof.namelist_import import import_namelists, NamelistRefusal
from woof.experiment import build_experiment
from test_namelist_import import INPUT_TEXT, _pair


def imported(tmp_path, *, gs='0,0', trigger=1, cu=1, mp=6):
    text = re.sub(r'(?m)^.*mp_physics\s*=.*$', f' mp_physics = {mp},{mp},', INPUT_TEXT)
    text = re.sub(r'(?m)^.*cu_physics\s*=.*$', f' cu_physics = {cu},{cu},', text)
    text = text.replace('&physics', f'&physics\n grav_settling = {gs},\n kfeta_trigger = {trigger},')
    return import_namelists(*_pair(tmp_path, inp=text))


@pytest.mark.parametrize('gs', [1, 2])
@pytest.mark.parametrize('mp', [6, 10])
def test_shared_fog_reaches_both_domains(tmp_path, gs, mp):
    text, report = imported(tmp_path, gs=f'{gs},{gs}', mp=mp)
    exp = build_experiment(tomllib.loads(text), source="fog-kfeta-regression")
    assert [dc.run.grav_settling for dc in exp.domains] == [gs, gs]
    assert not any(row.key == 'grav_settling' for row in report.dropped)


def test_fog_column_keeps_native_default_in_omitted_tail(tmp_path):
    text, _ = imported(tmp_path, gs='2')
    assert [dc.run.grav_settling for dc in build_experiment(tomllib.loads(text), source="fog-kfeta-regression").domains] == [2, 0]


@pytest.mark.parametrize('trigger', [2, 3])
def test_kf_eta_trigger_reaches_run_config(tmp_path, trigger):
    text, report = imported(tmp_path, trigger=trigger)
    assert [dc.run.kfeta_trigger for dc in build_experiment(tomllib.loads(text), source="fog-kfeta-regression").domains] == [trigger]*2
    assert not any(row.key == 'kfeta_trigger' for row in report.dropped)


@pytest.mark.parametrize('trigger', [2, 3])
@pytest.mark.parametrize('cu', [0, 11])
def test_other_cumulus_cannot_silently_drop_trigger(tmp_path, trigger, cu):
    with pytest.raises(NamelistRefusal, match='only KF_eta_PARA consumes'):
        imported(tmp_path, trigger=trigger, cu=cu)


@pytest.mark.parametrize('gs', [1, 2])
def test_native_thompson_override_is_named(tmp_path, gs):
    with pytest.raises(NamelistRefusal, match='module_check_a_mundo.F:2474-2487'):
        imported(tmp_path, gs=f'{gs},{gs}', mp=28)


@pytest.mark.parametrize('trigger', [2, 3])
def test_plan_and_run_admit_native_kf_triggers(trigger):
    from woof.physics_registry import validate_physics_plan
    from test_physics_registry import _uniform_tree
    plan = _uniform_tree()
    for domain in plan['domains']:
        domain.update(template_id=f'thompson-mp8-ysu-mm5-noah-kf-trigger{trigger}-rte-rrtmgp-v1',
                      parameters={'grav_settling': 2})
    report = validate_physics_plan(plan)
    assert report['launchable'], report['errors']
    assert all(domain['settings']['kfeta_trigger'] == trigger
               and domain['settings']['grav_settling'] == 2
               for domain in report['resolved_domains'])


def test_plan_refuses_different_global_kf_triggers():
    from woof.physics_registry import validate_physics_plan
    from test_physics_registry import _uniform_tree
    plan = _uniform_tree()
    for domain, trigger in zip(plan['domains'], (2, 3)):
        domain['template_id'] = f'thompson-mp8-ysu-mm5-noah-kf-trigger{trigger}-rte-rrtmgp-v1'
    report = validate_physics_plan(plan)
    assert not report['launchable']
    assert any(issue['code'] == 'nonuniform-shared-setting'
               and 'kfeta_trigger' in issue['message'] for issue in report['errors'])


@pytest.mark.parametrize('gs', [1, 2])
def test_plan_names_native_thompson_override(gs):
    from woof.physics_registry import physics_registry, validate_physics_plan
    from test_physics_registry import _uniform_tree
    registry = physics_registry()
    aero = next(key for key, option in registry['components']['microphysics']['options'].items()
                if option.get('selectors', {}).get('mp_physics') == 28)
    plan = _uniform_tree()
    plan['domains'][0].update(components={'microphysics': aero},
                              parameters={'grav_settling': gs})
    report = validate_physics_plan(plan)
    assert not report['launchable']
    assert any(issue['code'] == 'fog-kfeta-pairing'
               and 'module_check_a_mundo.F:2474-2487' in issue['message']
               for issue in report['errors'])


@pytest.mark.parametrize('gs', [1, 2])
def test_pbl_only_fog_does_not_apply_the_skipped_surface_landuse_gate(gs):
    from types import SimpleNamespace
    from woof.core.fog_settling import build_fog_settling
    cfg = SimpleNamespace(grav_settling=gs, mp_physics=6,
                          bl_pbl_physics=1, sf_sfclay_physics=0)
    carrier = build_fog_settling(cfg, 'SIB')
    assert carrier.grav_settling == gs
    assert carrier.categories is None


@pytest.mark.parametrize("key,value", [("grav_settling", 1), ("kfeta_trigger", 2)])
def test_legacy_default_restart_resumes_and_active_fog_kf_changes_refuse(key, value):
    from dataclasses import asdict, replace
    from woof.config import RunConfig
    from woof.io.restart import (
        RestartMismatchError, _drop_default_off_run_keys, _require_config_match)
    cfg = RunConfig(nx=8, ny=8, nz=49, dx=3000., dy=3000.,
                    ztop=20000., dt=12., run_seconds=120.)
    stored = asdict(cfg)
    _drop_default_off_run_keys(stored)
    assert "grav_settling" not in stored and "kfeta_trigger" not in stored
    _require_config_match(stored, cfg, "legacy-default")
    with pytest.raises(RestartMismatchError, match=key):
        _require_config_match(stored, replace(cfg, **{key: value}), "active-change")
