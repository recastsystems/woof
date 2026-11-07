"""Disabled previous-KF diagnostic settings preserve older configuration bytes."""
from woof.io.restart import _configuration_digest_values, _drop_default_off_run_keys

def test_disabled_profile_does_not_change_legacy_configuration_identity():
    old = {"cu_physics": 1, "nx": 40}
    new = {**old, "kf_edrates": 0}
    assert _configuration_digest_values(old) == _configuration_digest_values(new)
    _drop_default_off_run_keys(new)
    assert new == old

def test_enabled_profile_remains_declared():
    values = {"cu_physics": 99, "kf_edrates": 1}
    _drop_default_off_run_keys(values)
    assert values["kf_edrates"] == 1
