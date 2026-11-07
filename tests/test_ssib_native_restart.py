"""Native SSiB restart inventory and inactive-scheme isolation."""
from types import SimpleNamespace
from woof.ingest.wrfrst import _disposition, _ssib_restart_names, fields_restore_plan
from woof.core.ssib import SSIB_REGISTRY_STATE


def test_every_ssib_native_carrier_is_read_and_mapped():
    cfg = SimpleNamespace(sf_surface_physics=8)
    aliases = _ssib_restart_names()
    names = {aliases.get(n, n.upper()) for n in SSIB_REGISTRY_STATE}
    for name in SSIB_REGISTRY_STATE:
        native = aliases.get(name, name.upper())
        assert _disposition(native, cfg).kind == 'read', name
        assert fields_restore_plan(name, names, cfg) == ('fill', native)
    assert fields_restore_plan('_ssib_status', names, cfg)[:2] == ('cold', 0)
    assert fields_restore_plan('xice_save', names, cfg)[:2] == ('copy', 'fields/xice')
    assert fields_restore_plan('tsk_save', names, cfg)[:2] == ('copy', 'fields/tsk')
    assert fields_restore_plan('surface_rainncv', {'RAINNCV'}, cfg) == ('fill', 'RAINNCV')


def test_ssib_restart_rules_do_not_change_noah():
    cfg = SimpleNamespace(sf_surface_physics=2)
    assert _disposition('SIBECT', cfg) is None
    assert _disposition('SWVISDIR', cfg).kind == 'inactive'
    assert fields_restore_plan('ssib_ect', {'SIBECT'}, cfg) is None
