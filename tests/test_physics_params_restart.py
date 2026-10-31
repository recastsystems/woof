"""Parameter sets bind trajectory identity without expanding default echoes."""
from dataclasses import asdict, replace
import json

from woof import experiment, physics_params as pp
from woof.core.model import restart_identity_payload
from woof.io.restart import configuration_echo
from test_physics_params import (
    _EXPERIMENT, _build, _fresh_process_binding as fresh_binding,
)


def _bytes(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      default=str).encode()


def test_absent_set_preserves_emitted_config_bytes_and_checkpoint_echo():
    exp = _build(_EXPERIMENT)
    previous = experiment._public_config_value(exp)
    previous.pop("physics_params")
    # These current default-off controls predate this feature.
    previous.pop("devices")
    previous.pop("simulated_radar")
    # [radar_heating] (e30746bd1) is a later default-off control that
    # experiment_config_document drops when off, the same way.
    previous.pop("radar_heating")
    # [nudging] (labs/nudging, 2.9.11) is another default-off control the
    # document drops when absent.
    previous.pop("nudging", None)
    assert _bytes(experiment.experiment_config_document(exp)) == _bytes(previous)
    assert "physics_params" not in restart_identity_payload(exp)
    assert "physics_params" not in configuration_echo(exp.root.run)


def test_set_identity_echoes_values_registry_and_digest():
    exp = _build(_EXPERIMENT)
    before = restart_identity_payload(exp)
    run_echo = configuration_echo(exp.root.run)
    pset = pp.make_set("member", {"mynn.prandtl": 0.8})
    member = replace(exp, physics_params=pset)
    document = experiment.experiment_config_document(member)
    identity = restart_identity_payload(member)
    assert document["physics_params"] == pp.document(pset)
    assert identity["physics_params"] == pp.document(pset)
    assert configuration_echo(member.root.run) == run_echo
    assert {key: value for key, value in identity.items()
            if key != "physics_params"} == before
    changed = replace(member, physics_params=pp.make_set("member", {"mynn.prandtl": 0.9}))
    assert _bytes(restart_identity_payload(changed)) != _bytes(identity)
    assert asdict(member.root.run) == asdict(exp.root.run)
