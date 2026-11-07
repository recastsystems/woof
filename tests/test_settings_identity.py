"""Settings intake preserves old public bytes and binds active controls."""
from dataclasses import fields, make_dataclass, replace
import json

import pytest

from woof.experiment import domain_config_document, experiment_config_document
from woof.config import CHEM_RUN_FIELDS, FIRE_RUN_FIELDS
from woof.io import restart
from test_ruc_public_identity import _experiment


ADDED = frozenset({"use_theta_m", "tke_adv_opt", "sfs_opt", "m_opt",
                   "tke_mix2_off", "tke_mix6_off"})


def _bytes(document):
    return json.dumps(document, default=str, allow_nan=False).encode("utf-8")


def test_default_settings_keep_complete_pre_intake_public_bytes():
    current = _experiment()
    domain = current.domains[0]
    # These older optional blocks also omit their neutral fields. A stand-in
    # without the RunConfig type must represent their emitted schema.
    omitted = ADDED | set(CHEM_RUN_FIELDS) | set(FIRE_RUN_FIELDS)
    legacy_fields = [f for f in fields(domain.run) if f.name not in omitted]
    legacy_type = make_dataclass("PreSettingsRunConfig",
                                [(f.name, f.type) for f in legacy_fields],
                                namespace={"ifire": 0, "fire_smoke": False})
    legacy_run = legacy_type(**{f.name: getattr(domain.run, f.name)
                               for f in legacy_fields})
    legacy = replace(current, domains=(replace(domain, run=legacy_run),))
    assert _bytes(domain_config_document(domain)) == _bytes(
        domain_config_document(legacy.domains[0]))
    assert _bytes(experiment_config_document(current)) == _bytes(
        experiment_config_document(legacy))
    assert not ADDED.intersection(restart.configuration_echo(domain.run))


@pytest.mark.parametrize("name,value", [("use_theta_m", 1),
                                      ("tke_mix2_off", True),
                                      ("tke_mix6_off", True)])
def test_active_settings_bind_public_and_checkpoint_identity(name, value):
    default = _experiment(km_opt=5, bl_pbl_physics=0)
    domain = default.domains[0]
    active_run = replace(domain.run, **{name: value})
    active = replace(default, domains=(replace(domain, run=active_run),))
    assert domain_config_document(active.domains[0])["run"][name] == value
    assert _bytes(experiment_config_document(active)) != _bytes(
        experiment_config_document(default))
    assert restart.configuration_echo(active_run)[name] == value
    with pytest.raises(restart.RestartMismatchError, match=name):
        restart._require_config_match(restart.configuration_echo(domain.run),
                                      active_run, "pre-settings.npz")
