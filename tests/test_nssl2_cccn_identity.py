"""A default CCN coefficient preserves pre-coefficient checkpoint identity."""
from dataclasses import asdict
from woof.config import RunConfig
from woof.core.model import restart_identity_payload
from woof.io.restart import _configuration_digest_values
from woof.ingest.prepared_cache import (
    compare_prepared_domain_config, undelayed_identity_defaults,
)
import pytest


@pytest.fixture(autouse=True)
def identity_document_seam(monkeypatch):
    # This unit check grades the identity normalizer's complete document,
    # including the field-absent document written by the previous version.
    import woof.experiment
    monkeypatch.setattr(woof.experiment, 'experiment_config_document', lambda value: value)


def config(value=0.5e9):
    return RunConfig(nx=6,ny=6,nz=8,dx=1000.,dy=1000.,ztop=8000.,dt=2.,
                     run_seconds=4.,moist=True,mp_physics=18,nssl_cccn=value)


def document(run):
    return {'domains':[{'run':run}]}


def test_default_experiment_identity_equals_legacy_absent_field():
    current=asdict(config())
    legacy=dict(current)
    legacy.pop('nssl_cccn')
    assert restart_identity_payload(document(current)) == restart_identity_payload(document(legacy))


def test_default_checkpoint_digest_equals_legacy_absent_field():
    current=asdict(config())
    legacy=dict(current)
    legacy.pop('nssl_cccn')
    assert _configuration_digest_values(current) == _configuration_digest_values(legacy)


def test_nondefault_cccn_binds_checkpoint_and_experiment():
    current=asdict(config(1e7))
    legacy=dict(current)
    legacy.pop('nssl_cccn')
    assert _configuration_digest_values(current) != _configuration_digest_values(legacy)
    assert restart_identity_payload(document(current)) != restart_identity_payload(document(legacy))


@pytest.mark.parametrize('value,accepted', [(0.5e9, True), (1e7, False)])
@pytest.mark.parametrize('reverse', [False, True])
def test_legacy_prepared_header_tolerates_only_default_coefficient(value, accepted, reverse):
    legacy = {'run': {'mp_physics': 18}}
    current = {'run': {'mp_physics': 18, 'nssl_cccn': value}}
    cached, live = (current, legacy) if reverse else (legacy, current)
    tolerated, differing = compare_prepared_domain_config(
        cached, live, not_in_use=undelayed_identity_defaults(None))
    assert tolerated == (['run.nssl_cccn'] if accepted else [])
    assert differing == ([] if accepted else ['run.nssl_cccn'])
