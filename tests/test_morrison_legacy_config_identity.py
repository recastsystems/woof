"""New package defaults preserve legacy echo and digest bytes."""
from woof.io.restart import _configuration_digest_values, _drop_default_off_run_keys

def test_inactive_package_defaults_preserve_legacy_config():
    old = {"mp_physics": 8, "nx": 40, "morr_rimed_ice": 1}
    new = {**old, "morr_rimed_ice": 1, "aercu_opt": 0}
    assert _configuration_digest_values(new) == _configuration_digest_values(old)
    _drop_default_off_run_keys(new)
    assert new == old

def test_active_or_changed_package_controls_stay_bound():
    for values in [{"mp_physics": 40, "morr_rimed_ice": 1, "aercu_opt": 1}, {"mp_physics": 8, "morr_rimed_ice": 2, "aercu_opt": 1}]:
        expected = values.copy()
        _drop_default_off_run_keys(values)
        assert values == expected


def test_active_package_omits_neutral_aerosol_control():
    values = {"mp_physics": 40, "aercu_opt": 0}
    _drop_default_off_run_keys(values)
    assert values == {"mp_physics": 40}
