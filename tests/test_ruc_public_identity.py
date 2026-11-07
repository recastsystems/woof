"""Generic selector defaults retain pre-field public and trajectory bytes."""
from dataclasses import fields, make_dataclass, replace
from datetime import datetime
import json

import pytest

from woof.config import CHEM_RUN_FIELDS, FIRE_RUN_FIELDS, RunConfig
from woof.core.model import restart_identity_payload
from woof.experiment import (
    DomainConfig, ExperimentConfig, VerticalConfig,
    domain_config_document, experiment_config_document)
from woof.ingest.prepared_cache import effective_prepared_domain_config
from woof.io import restart


OPTIONS = {"ruc_irrigation": "wrf_45", "ruc_qvg_cold_start": "air",
           "ruc_2m_diagnostic": "log_profile", "ruc_snow": "wrf_45"}
# The historical schema predates the RUC forms and the subsequently
# appended radiation selectors. Keep this snapshot independent of the
# emitter's omission table, so the full byte comparisons retain their pin.
PRE_FIELD_OMISSIONS = frozenset(OPTIONS) | {"swint_opt", "aer_opt", "alb_sol"}


def _experiment(**overrides):
    cfg = RunConfig(nx=6, ny=4, nz=5, dx=3000., dy=3000., ztop=20000.,
                    dt=20., run_seconds=0., **overrides)
    domain = DomainConfig(grid_id=1, parent_id=0, i_parent_start=1,
        j_parent_start=1, parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=60., run=cfg, time_step=20)
    return ExperimentConfig(name="generic", start_time=datetime(2026, 1, 1),
        run_seconds=0., vertical=VerticalConfig(
            eta_levels=(1., .8, .6, .4, .2, 0.), p_top=5000., hybrid_opt=2, etac=.2),
        restart_interval_s=0., domains=(domain,))


def _historical_schema(exp):
    domain = exp.domains[0]
    # The historical public schema predates chemistry and these neutral selectors.
    assert domain.run.chem_sets == ""
    neutral = {"swint_opt": 0, "aer_opt": 0, "alb_sol": 0}
    for name, default in neutral.items():
        assert getattr(domain.run, name) == default
    # The keyword-only fire block (development branch) postdates it too.
    omitted = (set(PRE_FIELD_OMISSIONS) | set(CHEM_RUN_FIELDS)
               | set(FIRE_RUN_FIELDS))
    old_fields = [field for field in fields(domain.run) if field.name not in omitted]
    # ExperimentConfig still asks every run whether it carries a fire, so
    # the stand-in answers ifire 0 and no fire smoke without making them
    # fields (fields are what the documents serialize).
    old_type = make_dataclass("HistoricalRunConfig", [(field.name, field.type) for field in old_fields],
                              namespace={"ifire": 0, "fire_smoke": False})
    old_run = old_type(**{field.name: getattr(domain.run, field.name) for field in old_fields})
    return replace(exp, domains=(replace(domain, run=old_run),))


def _bytes(document):
    return json.dumps(document, default=str, allow_nan=False).encode("utf-8")


def test_generic_forms_keep_complete_pre_field_public_and_trajectory_bytes():
    # This is a non-RUC configuration. RUC's v6 algorithm identity still
    # independently refuses earlier ambiguous RUC continuation headers.
    exp = _experiment()
    old = _historical_schema(exp)
    assert exp.domains[0].run.sf_surface_physics == 0
    assert tuple(getattr(exp.domains[0].run, name)
                 for name in ("swint_opt", "aer_opt", "alb_sol")) == (0, 0, 0)
    assert _bytes(domain_config_document(exp.domains[0])) == _bytes(domain_config_document(old.domains[0]))
    assert _bytes(experiment_config_document(exp)) == _bytes(experiment_config_document(old))
    assert _bytes(restart_identity_payload(exp)) == _bytes(restart_identity_payload(old))


@pytest.mark.parametrize("name,value", OPTIONS.items())
def test_moved_form_binds_public_echo_and_trajectory_but_reuses_preparation(name, value):
    exp = _experiment(**{name: value})
    old = _historical_schema(exp)
    assert domain_config_document(exp.domains[0])["run"][name] == value
    assert experiment_config_document(exp)["domains"][0]["run"][name] == value
    assert restart.configuration_echo(exp.domains[0].run)[name] == value
    assert _bytes(restart_identity_payload(exp)) != _bytes(restart_identity_payload(old))
    assert effective_prepared_domain_config(domain_config_document(exp.domains[0])) \
        == effective_prepared_domain_config(domain_config_document(old.domains[0]))
