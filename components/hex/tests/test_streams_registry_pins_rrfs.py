"""Coexisting pin contracts and the actual 44c0849 Registry C output."""
import os
from pathlib import Path
import re

import pytest

from woof.hex import streams_rrfs as streams
from test_streams_native_registry_rrfs import _native_fields
from stream_registry_source_rrfs import DECLARED_UNITS


@pytest.fixture
def contract():
    return streams.RegistryContract(Path(streams.__file__).with_name('streams_rrfs_44c0849_registry.json'),
                                    timelevel_options={'reject_step_after_stop': True})


def test_aod550_declares_the_cf_dimensionless_unit_the_converter_requires(contract):
    # The fork's Registry says '-'; rw_mpas_convert requires '1' of an AOD product (stream_registry_source_rrfs
    # DECLARED_UNITS). Everything else about the field is the fork's.
    aod = contract.field_definition('aod550')
    assert aod['attrs']['units'] == '1'
    assert aod['dimensions'] == ['nCells', 'Time']
    assert aod['attrs']['long_name'] == 'aerosol optical depth (2D) from all aerosols'


def test_old_and_new_streams_keep_their_own_metadata(contract, tmp_path):
    old = streams.registry_stream('restart')
    new = contract.registry_stream('restart')
    assert 'defc_a' in old.fields(streams._all_packages())
    assert 'defc_a' not in new.fields(streams._all_packages(contract.data))
    assert 'tke' in new.fields(streams._all_packages(contract.data))
    with pytest.raises(ValueError, match='unknown fork Registry field'):
        old.field_definition('prate_max')
    deck = tmp_path/'streams.atmosphere'
    deck.write_text('<streams><stream name="rates" type="output" filename_template="rates.nc" '
                    'output_interval="00:01:00"><var name="prate_max"/></stream></streams>')
    parsed = contract.parse_streams(deck)
    assert parsed['rates'].fields(streams._all_packages(contract.data)) == ('prate_max',)
    assert parsed['rates'].field_definition('prate_max')['dimensions'] == ['nCells', 'Time']
    with pytest.raises(ValueError, match='unknown fork Registry field'):
        streams.parse_streams(deck)


@pytest.mark.parametrize('physics', [False, True])
def test_new_pin_native_registry(contract, physics):
    root = os.environ.get('RRFS_44C0849_NATIVE_REGISTRY_GENERATED')
    if not root:
        pytest.skip('44c0849 compiled Registry artifacts not supplied')
    generated = Path(root)/('full' if physics else 'dry')
    native = _native_fields(generated/'structs_and_variables.inc')
    assert set(native) == set(contract.data['fields' if physics else 'dry_fields'])
    for name, definition in native.items():
        ours = contract.field_definition(name, physics=physics)
        if physics and name in DECLARED_UNITS:
            definition = {**definition, 'attrs': {**definition['attrs'], 'units': DECLARED_UNITS[name]}}
        assert definition == {k: ours[k] for k in ('type', 'dimensions', 'attrs')}, name

    # Compare the stream membership emitted by MPAS's C tool, including every
    # individual package and the empty/full package selections.
    memberships = {}
    gate = ''
    for line in (generated/'setup_immutable_streams.inc').read_text().splitlines():
        create = re.search(r"create_stream\(manager, '([^']+)'", line)
        if create:
            memberships[create[1]] = []
        packages = re.search(r"write\(packages,'\(a\)'\) '([^']*)'", line)
        if packages:
            gate = packages[1]
        member = re.search(r"add_(field|pool)\(manager, '([^']+)', '([^']+)'(.*)\)", line)
        if member:
            memberships[member[2]].append((member[3], gate if 'packages=packages' in member[4] else ''))
    groups = contract.data['groups' if physics else 'dry_groups']
    all_packages = streams._all_packages(contract.data)
    for active in (frozenset(), all_packages, *(frozenset([p]) for p in sorted(all_packages))):
        for name, members in memberships.items():
            ours = contract.registry_stream(name, physics=physics)
            if not streams.package_active(ours.options.get('packages', ''), active):
                continue
            expected = set()
            for member, gate in members:
                if not streams.package_active(gate, active):
                    continue
                for variable in groups.get(member, (member,)):
                    definition = contract.field_definition(variable, physics=physics)
                    if all(streams.package_active(p, active) for p in definition['gates']):
                        expected.add(variable)
            assert set(ours.fields(active, physics=physics)) == expected, (name, active)


def test_new_pin_metadata_regenerates(contract):
    from stream_registry_source_rrfs import metadata
    ref = os.environ.get('RRFS_44C0849_SOURCE')
    if not ref:
        pytest.skip('44c0849 source artifact not supplied')
    assert metadata(Path(ref)/'src/core_atmosphere/Registry.xml') == contract.data


@pytest.mark.parametrize('spec', ['0-3-15m 4-24 27-36-3','1 2 3 6 9 12 24',
    '0m 15m 1h3m 2 3','1h30m-2h-15m','0-90s-30s','0-1D-3h',
    '0-1-2','4-2','0-6-0','0  1  2','0-30m-0.5m','0-2-1e-2h', '4-5-2'])
def test_native_timelevel_words(contract, spec):
    import subprocess
    native = os.environ.get('RRFS_44C0849_NATIVE_TIMELEVELS')
    if not native:
        pytest.skip('44c0849 native timelevel artifact not supplied')
    for seconds in (0,1,29,30,59,60,899,900,1800,3599,3600,5400,7200,10800,21600,86400,129600):
        run = subprocess.run([native, spec, str(seconds)], text=True, capture_output=True, timeout=5)
        try:
            ours = contract.timelevel_query(spec, seconds)
        except ValueError:
            assert run.returncode != 0
            continue
        assert run.returncode == 0, run.stderr
        rows = [[int(word)&0xffffffff for word in line.split()] for line in run.stdout.splitlines()]
        assert rows[:-1] == ours['words']
        assert rows[-1][1:] == ours['query_words']
        assert (rows[-1][0] == 0) == ours['matched']


def test_timelevel_validation_is_bound_to_pin(contract, tmp_path):
    assert streams.timelevel_query('0-1-2', 0)['matched']
    with pytest.raises(ValueError, match='start-stop-step'):
        contract.timelevel_query('0-1-2', 0)
    deck = tmp_path/'streams.atmosphere'
    deck.write_text('<streams><stream name="rates" type="output" filename_template="rates.nc" '
                    'output_timelevels="0-1-2"><var name="xtime"/></stream></streams>')
    assert streams.parse_streams(deck)['rates'].timelevel_query(0)['matched']
    with pytest.raises(ValueError, match='start-stop-step'):
        contract.parse_streams(deck)
