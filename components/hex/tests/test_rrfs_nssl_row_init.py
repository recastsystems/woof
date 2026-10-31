"""A renamed row still selects its initializer without scheme-specific code."""
from types import SimpleNamespace


def test_microphysics_initializer_follows_the_row(monkeypatch):
    from woof.hex import physics_suites
    from woof.hex.mpasphys.rrfs import nssl_rrfs
    from woof.hex.mpasphys.rrfs.framework_init_rrfs import FrameworkInitializationRRFS
    row = physics_suites.load_table()['mpasphys_rrfs']['slots']['microphysics']['cuda_entries']['mp_nssl2m']
    assert row['initialize'] == 'initialize'
    table = {'alternate_route': {'slots': {'microphysics': {
        'cuda_entries': {'renamed_scheme': row}}}}}
    monkeypatch.setattr(physics_suites, 'load_table', lambda: table)
    calls = []
    monkeypatch.setattr(nssl_rrfs, 'initialize', calls.append)
    provider = object()
    target = SimpleNamespace(configs={'config_microp_scheme': 'renamed_scheme'},
        session=SimpleNamespace(config=SimpleNamespace(table_key='alternate_route'),
                                device_pools=provider))
    FrameworkInitializationRRFS.initialize_microphysics(target)
    assert calls == [provider]
