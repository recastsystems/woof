"""Output boundary contracts and real Rust NetCDF round trips.

Synthetic words exercise storage only. These are not forecast identity proofs.
"""
from datetime import datetime, timedelta
import os
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys.forecast_v842 import Stream
from woof.hex.mpasphys.streams_output_v842 import PhysicsStreams
from woof.hex.restart_v842 import inspect_stream, read_stream
from woof.hex.stream_headers_v842 import compare_headers


class DeviceArray(np.ndarray):
    def get(self, order='C', *, out=None, blocking=True):
        if out is not None:
            assert blocking
            np.copyto(out, np.asarray(self))
            return out
        return np.array(self, copy=True, order=order, subok=False)


def session(*, resident):
    start = datetime(2023, 3, 31, 12)
    words = np.resize(np.array([0x80000000, 0x7fc12345, 1, 0x3f800000], np.uint32), (2, 2, 4))
    fields = {('state', 'scalars'): words.view(np.float32),
              ('state', 'Time'): np.array(16777216, np.float32)}
    rows = [dict(name='scalars', pool='state', pool_name='scalars', components=['qv', 'qc'],
                 dimensions=['num_scalars', 'nVertLevels', 'nCells'], kind='real'),
            dict(name='Time', pool='state', pool_name='Time', components=[], dimensions=[], kind='real'),
            dict(name='initial_time', pool='state', pool_name='initial_time', components=[],
                 dimensions=[], kind='character'),
            dict(name='xtime', pool='state', pool_name='xtime', components=[],
                 dimensions=[], kind='character')]
    layout = SimpleNamespace(fields=rows, packages=frozenset({'mp_wsm6_in'}),
        dimensions=dict(nCells=3, nVertLevels=2, num_scalars=2),
        configs=dict(config_start_time=start.strftime('%Y-%m-%d_%H:%M:%S'),
                     config_do_restart=False, config_do_DAcycling=False))
    reads = []

    def get(pool, name, *, level=1):
        reads.append((pool, name))
        if resident and name != 'Time':
            raise AssertionError('resident output must not round trip through host pools')
        return fields[pool, name].copy()

    native = SimpleNamespace(get_field=get,
        get_text_field=lambda *a, **kw: start.strftime('%Y-%m-%d_%H:%M:%S'))
    views = SimpleNamespace(resident=resident, cp=np, level=2,
        field=lambda pool, name: fields[pool, name].view(DeviceArray),
        transfers=dict(publication_download_bytes=0))
    return SimpleNamespace(native=native, layout=layout, device_pools=views), fields, reads, start


def stream():
    return Stream('output', ('output',), 'history.nc', 'none', '00:01:00',
                  ('scalars', 'Time', 'xtime', 'initial_time'))


def test_default_streams_retain_every_active_registry_member(tmp_path):
    from woof.hex.mpasphys import resolve_namelist
    from woof.hex.mpasphys.forecast_v842 import read_streams, write_default_streams
    from woof.hex.mpasphys.framework_backend_v842 import registry_source
    from woof.hex.mpasphys.pool_v842 import RegistryLayout
    from woof.hex.mpasphys.resident_output_diagnostics_v842 import PRODUCED
    from woof.hex.mpasphys.streams_output_v842 import output_definitions, output_inventory

    text = '&physics\nconfig_physics_suite="convection_permitting"\n/'
    layout = RegistryLayout(registry_source(), resolve_namelist(text), text,
        dict(nCells=2, nEdges=3, nVertices=2, maxEdges=6, maxEdges2=12,
             vertexDegree=3, nVertLevels=55))
    path = write_default_streams(layout, tmp_path/'streams.atmosphere',
                                 history_minutes=60, checkpoint_hours='1')
    streams = read_streams(path, layout)
    missing = output_inventory(streams, layout, produced=PRODUCED)
    assert all(not fields for fields in missing.values())
    definitions = output_definitions(streams, layout, missing)
    arrays = {row['name']: tuple(row['components']) for row in layout.fields if row['components']}
    for selected in streams:
        expected = tuple(n for name in selected.fields for n in arrays.get(name, (name,)))
        assert definitions[selected.name].fields(layout.packages, streams=definitions) == expected


@pytest.mark.parametrize('resident', [False, True])
def test_independent_gravity_wave_diagnostics_reach_rust_output(tmp_path, bridge, resident):
    """YSU may request dormant UGWP diagnostics without enabling UGWP physics.

    Native selects the output package independently (core_interface.F:249-253).
    Its initialized diagnostic words must survive stream selection, resident
    capture and encoding, while Time follows the committed framework clock.
    """
    from dataclasses import replace
    from woof.hex.mpasphys import resolve_namelist
    from woof.hex.mpasphys.forecast_v842 import read_streams, write_default_streams
    from woof.hex.mpasphys.framework_backend_v842 import registry_source
    from woof.hex.mpasphys.pool_v842 import RegistryLayout

    state, values, reads, start = session(resident=resident)
    text = '&physics\nconfig_physics_suite="convection_permitting"\nconfig_ugwp_diags=.true.\n/'
    layout = RegistryLayout(registry_source(), resolve_namelist(text), text,
        dict(nCells=3, nEdges=4, nVertices=3, maxEdges=6, maxEdges2=12,
             vertexDegree=3, nVertLevels=55))
    assert 'ugwp_diags_stream' in layout.packages
    assert 'ugwp_orog_stream' not in layout.packages
    state.layout = layout
    layout.configs['config_start_time'] = start.strftime('%Y-%m-%d_%H:%M:%S')
    path = write_default_streams(layout, tmp_path/'streams.atmosphere')
    selected = next(s for s in read_streams(path, layout) if s.name == 'diag_ugwp')
    selected = replace(selected, output_interval='03:00:00', filename_interval='03:00:00')
    rows = {row['name']: row for row in layout.fields}
    diagnostic_names = [name for name in selected.fields if rows[name]['pool'] == 'diag_physics']
    assert len(diagnostic_names) == 19 and len(selected.fields) == 22
    for name in diagnostic_names:
        row = rows[name]
        value = np.zeros(row['shape'], np.float32)
        value[..., -1] = -999  # Allocated dummy cells must never reach disk.
        values[row['pool'], row['pool_name']] = value
    committed = [start]
    state.native.clock_state = SimpleNamespace(current_time=SimpleNamespace(
        stamp=lambda: committed[0].strftime('%Y-%m-%d_%H:%M:%S')))
    receipt = {}
    with PhysicsStreams(state, (selected,), tmp_path, final_time=start+timedelta(hours=3)) as writer:
        assert writer.missing == {'diag_ugwp': []}
        writer.write(step=0, dt=1200, receipt=receipt)
        committed[0] += timedelta(hours=3)
        values['state', 'Time'][...] = 10800
        writer.write(step=999, dt=1200, receipt=receipt)
    files = sorted(tmp_path.glob('diag_ugwp.*.nc'))
    assert len(files) == 2 and len(receipt['streams']) == 2
    for index, file in enumerate(files):
        header = inspect_stream(file, scratch=tmp_path/'inspect')
        assert header['format'] == 5
        with read_stream(file, scratch=tmp_path/'read') as output:
            assert list(output.variables) == list(selected.fields)
            assert output.variables['Time'].values[0] == (16777216 if index == 0 else 10800)
            assert output.variables['xtime'].text == ((start+timedelta(hours=index*3)).strftime('%Y-%m-%d_%H:%M:%S'),)
            for name in diagnostic_names:
                variable = output.variables[name]
                assert variable.values.shape == (1, 3, *rows[name]['shape'][:-1])
                assert np.all(variable.values.view(np.uint32) == 0)
                assert variable.attrs['units'] == rows[name]['units']
    if resident:
        assert reads == [('state', 'Time'), ('state', 'Time')]
        assert state.device_pools.transfers['publication_download_bytes'] == 2*3*(11*55+8)*4


@pytest.fixture
def bridge():
    if not os.environ.get('WOOF_HEX_RW_MPAS_STREAM'):
        pytest.skip('requires rw_mpas_stream with ordered metadata support')


@pytest.mark.parametrize('resident,pinned', [(False, False), (True, False), (True, True)])
def test_selected_words_are_frozen_trimmed_and_appended(tmp_path, bridge, resident, pinned):
    state, fields, reads, start = session(resident=resident)
    allocations = []
    if pinned:
        def allocate(size):
            # CuPy may return a larger cached allocation than requested.
            allocations.append(size)
            return bytearray(size + 128)
        state.device_pools.cp = SimpleNamespace(uint32=np.uint32,
            cuda=SimpleNamespace(alloc_pinned_memory=allocate))
    before = fields['state', 'scalars'][..., :3].copy()
    receipt = {}
    with PhysicsStreams(state, (stream(),), tmp_path, final_time=start+timedelta(minutes=2)) as writer:
        assert writer.required(start)
        writer.write(step=0, dt=60, receipt=receipt)
        fields['state', 'scalars'][...] = 7
        fields['state', 'Time'][...] = 16777218
        assert not writer.required(start)
        writer.write(step=1, dt=60, receipt=receipt)
    assert state.device_pools.level == 2
    assert [(r['record_index'], r['writer']) for r in receipt['streams']] == [(0, 'rw_mpas_stream'), (1, 'rw_mpas_stream')]
    assert inspect_stream(tmp_path/'history.nc', scratch=tmp_path/'inspect')['format'] == 5
    with read_stream(tmp_path/'history.nc', scratch=tmp_path/'read') as output:
        assert output.variables['qv'].attrs['long_name'] == 'Water vapor mixing ratio'
        assert output.variables['Time'].attrs['standard_name'] == 'time'
        assert len(output.global_attrs['file_id']) == 10
        for index, name in enumerate(('qv', 'qc')):
            actual = output.variables[name].values
            np.testing.assert_array_equal(actual[0].view(np.uint32), before[index].T.view(np.uint32))
            np.testing.assert_array_equal(actual[1], 7)
        np.testing.assert_array_equal(output.variables['Time'].values, [16777216, 16777218])
        assert output.variables['xtime'].text == ('2023-03-31_12:00:00', '2023-03-31_12:01:00')
    if resident:
        assert reads == [('state', 'Time'), ('state', 'Time')]
        assert state.device_pools.transfers['publication_download_bytes'] == before.nbytes * 2
    if pinned:
        # One pinned set, or two when the second write found the first frame still being encoded
        # (PhysicsStreams.FRAMES_IN_FLIGHT); a set is never allocated twice for the same frame.
        assert allocations in ([before.nbytes], [before.nbytes] * 2)
        assert not writer._host_storage


def test_background_failure_propagates_on_close(tmp_path):
    state, _, _, start = session(resident=True)
    writer = PhysicsStreams(state, (stream(),), tmp_path, final_time=start+timedelta(minutes=2))

    def fail(*args):
        raise OSError('encoder failed')

    writer._encode = fail
    writer.write(step=0, dt=60, receipt={})
    with pytest.raises(OSError, match='encoder failed'):
        writer.close()
    assert not list(tmp_path.glob('*.nc'))


@pytest.mark.parametrize('resident', [False, True])
def test_microphysics_diagnostics_reach_output_from_their_authority(tmp_path, bridge, resident):
    """Humidity and surface-pressure tendency already belong to microphysics.

    Exercise both pool families and preserve the selected device words. A
    resident frame must use changed device values even when host pools lag.
    An enabled stream must reject allocated but uncomputed diagnostics.
    """
    from dataclasses import replace
    state, fields, reads, start = session(resident=resident)
    rh = np.array([[0x80000000, 0x7fc12345, 0x42c80000],
                   [1, 0x3f800000, 0x42a00000]], np.uint32).view(np.float32)
    tendency = np.array([0x80000000, 1, 0xbf800000], np.uint32).view(np.float32)
    for pool, name, dims, value in (
            ('diag', 'relhum', ['nVertLevels', 'nCells'], rh),
            ('tend', 'tend_sfc_pressure', ['nCells'], tendency)):
        fields[pool, name] = np.pad(value, [(0, 0)]*(value.ndim-1)+[(0, 1)])
        state.layout.fields.append(dict(name=name, pool=pool, pool_name=name,
            dimensions=dims, kind='real', components=[]))
    state.layout.fields.append(dict(name='ke', pool='diag', pool_name='ke',
        dimensions=['nVertLevels', 'nCells'], kind='real', components=[]))
    if resident:
        device = {key: value.copy().view(DeviceArray) for key, value in fields.items()}
        fields['diag', 'relhum'].fill(-1)
        fields['tend', 'tend_sfc_pressure'].fill(-1)
        state.device_pools.field = lambda pool, name: device[pool, name]
    definition = replace(stream(), fields=('relhum', 'tend_sfc_pressure', 'ke'))
    with pytest.raises(ValueError, match='output fields lack computed authority: output: ke'):
        PhysicsStreams(state, (definition,), tmp_path,
                       final_time=start+timedelta(minutes=2))
    assert not (tmp_path/'history.nc').exists()
    definition = replace(definition, fields=('relhum', 'tend_sfc_pressure'))
    with PhysicsStreams(state, (definition,), tmp_path,
                        final_time=start+timedelta(minutes=2)) as writer:
        assert writer.missing['output'] == []
        writer.write(step=0, dt=60, receipt={})
    with read_stream(tmp_path/'history.nc', scratch=tmp_path/'read') as output:
        assert list(output.variables) == ['relhum', 'tend_sfc_pressure']
        np.testing.assert_array_equal(output.variables['relhum'].values[0].view(np.uint32),
                                      rh.T.view(np.uint32))
        np.testing.assert_array_equal(output.variables['tend_sfc_pressure'].values[0].view(np.uint32),
                                      tendency.view(np.uint32))
        assert output.variables['relhum'].attrs['units'] == 'percent'
        assert output.variables['tend_sfc_pressure'].attrs['units'] == 'Pa s^{-1}'
    if resident:
        assert reads == []
        assert state.device_pools.transfers['publication_download_bytes'] == rh.nbytes+tendency.nbytes


def test_header_gate_detects_order_type_and_attribute_changes(tmp_path, bridge):
    from dataclasses import replace
    from woof.hex.output import HistoryField
    from woof.hex.restart_v842 import write_stream
    from woof.hex.streams_v842 import Stream as Definition
    state, fields, _, start = session(resident=False)
    definition = Definition('output', dict(type='output', filename_template='history.nc'),
                            (('var', 'Time', ''), ('var', 'qv', ''), ('var', 'xtime', '')))
    mesh = SimpleNamespace(dimensions=state.layout.dimensions, arrays={}, attrs={})
    values = {'qv': HistoryField(fields['state', 'scalars'][0, :, :3], ('nVertLevels', 'nCells'))}
    write_stream(tmp_path/'native.nc', definition, mesh, values, start, initial_time=start)
    write_stream(tmp_path/'ours.nc', definition, mesh, values, start, initial_time=start)
    same = compare_headers(tmp_path/'native.nc', tmp_path/'ours.nc', scratch=tmp_path/'inspect')
    assert same['passed'] and same['byte_identical']
    reordered = replace(definition, members=tuple(reversed(definition.members)))
    write_stream(tmp_path/'reordered.nc', reordered, mesh, values, start, initial_time=start,
                 field_attrs={'qv': {'units': 'changed'}})
    changed = compare_headers(tmp_path/'native.nc', tmp_path/'reordered.nc', scratch=tmp_path/'inspect')
    assert not changed['passed'] and not changed['byte_identical']
    assert {'variable_order', 'dimension_order', 'qv.attribute_words'} <= {d['item'] for d in changed['differences']}


def test_full_output_gate_rejects_missing_variables_and_changed_data(tmp_path, bridge):
    from dataclasses import replace
    from woof.hex.output import HistoryField
    from woof.hex.restart_v842 import write_stream
    from woof.hex.streams_v842 import Stream as Definition
    from woof.hex.verify_mpas import compare_sample_headers, stream_header_failures
    state, fields, _, start = session(resident=False)
    definition = Definition('output', dict(type='output', filename_template='history.nc'),
                            (('var', 'Time', ''), ('var', 'qv', ''), ('var', 'xtime', '')))
    mesh = SimpleNamespace(dimensions=state.layout.dimensions, arrays={}, attrs={})
    qv = fields['state', 'scalars'][0, :, :3]
    values = {'qv': HistoryField(qv, ('nVertLevels', 'nCells'))}
    native = tmp_path/'native.nc'
    write_stream(native, definition, mesh, values, start, initial_time=start)
    partial = replace(definition, members=tuple(m for m in definition.members if m[1] != 'qv'))
    write_stream(tmp_path/'partial.nc', partial, mesh, values, start, initial_time=start)
    changed = {'qv': HistoryField(qv + np.float32(1), ('nVertLevels', 'nCells'))}
    write_stream(tmp_path/'changed.nc', definition, mesh, changed, start, initial_time=start)
    def grade(path):
        return compare_sample_headers([dict(strict=str(native), hex=str(path),
            stream='output', steps=[0])], scratch=tmp_path/'inspect')
    assert stream_header_failures(grade(native)) == []
    # Projecting out native-only declarations hid this incomplete inventory.
    assert compare_headers(native, tmp_path/'partial.nc', scratch=tmp_path/'inspect',
                           written_fields_only=True)['passed']
    missing = grade(tmp_path/'partial.nc')
    assert missing[0]['scope'] == 'complete file' and not missing[0]['passed']
    assert 'stream header differs from native' in stream_header_failures(missing)[0]
    words = grade(tmp_path/'changed.nc')
    assert words[0]['passed'] and not words[0]['byte_identical']
    assert stream_header_failures(words) == []  # The Rust data-word comparison rejects this change.


def test_gpu_route_selects_ported_framework_and_native_reference_remains_explicit():
    from woof.hex.mpasphys.driver_v842 import resolve_namelist
    from woof.hex.mpasphys.runtime_v842 import framework_only
    text = "&physics config_physics_suite='convection_permitting' /"
    gpu = resolve_namelist(text, default_backend='hex-cuda')
    assert framework_only(gpu)
    referee = resolve_namelist(text, default_backend='native-cpu')
    assert not framework_only(referee)
    hybrid = resolve_namelist(text, default_backend='hex-cuda', backends={'microphysics': 'native-cpu'})
    assert not framework_only(hybrid)
    inactive = resolve_namelist("&physics config_physics_suite='none' /", default_backend='native-cpu')
    assert framework_only(inactive)


def test_gpu_route_with_sst_update_keeps_the_ported_framework():
    # hex-hybrid-longrange (2026-10-26): config_sst_update on the GPU route left the surface-update
    # service on native-cpu, so the route asked for the native MPAS library a user install lacks.
    from woof.hex.mpasphys.driver_v842 import resolve_namelist
    from woof.hex.mpasphys.runtime_v842 import framework_only
    text = "&physics config_physics_suite='convection_permitting' config_sst_update=.true. /"
    gpu = resolve_namelist(text, default_backend='hex-cuda')
    assert str(gpu.service_backends['surface_update']) == 'hex-cuda'
    assert framework_only(gpu)
    pinned = resolve_namelist(text + chr(10) + "&hex_mpasphys config_surface_update_backend='native-cpu' /",
                              default_backend='hex-cuda')
    assert str(pinned.service_backends['surface_update']) == 'native-cpu'
    assert not framework_only(pinned)
    assert not framework_only(resolve_namelist(text, default_backend='native-cpu'))
    # Without a surface series the receipt keeps no service row, as before.
    plain = resolve_namelist("&physics config_physics_suite='convection_permitting' /", default_backend='hex-cuda')
    assert 'service_backends' not in plain.receipt()


def test_stream_time_comes_from_committed_framework_clock(tmp_path):
    from woof.hex.mpasphys.framework_calendar_v842 import PhysicsClock
    state, _, _, start = session(resident=False)
    state.native.clock_state = PhysicsClock(start='2023-03-31_12:00:00',
        stop='2023-03-31_13:00:00', dt=60, calendar='gregorian')
    with PhysicsStreams(state, (stream(),), tmp_path, final_time=start+timedelta(hours=1)) as output:
        state.native.clock_state.advance()
        assert output.current_time(step=999, dt=1) == start+timedelta(minutes=1)


@pytest.mark.parametrize('direct', [False, True])
def test_restart_struct_retains_native_variable_array_order(tmp_path, direct):
    """Native hl-3 restart variables 82-93: scalars precede u_init..qv_init.

    Allocation order comes from MPAS src/tools/registry/gen_inc.c:1944-1952.
    The Registry XML and decoded field list can interleave ordinary variables
    before a variable array, so declaration order alone changes restart headers.
    """
    import xml.etree.ElementTree as ET
    from woof.hex.mpasphys.forecast_v842 import read_streams
    members = ('<var name="u_init"/><var name="v_init"/><var_array name="scalars"/>'
               '<var name="t_init"/><var name="qv_init"/>' if direct else '<var_struct name="state"/>')
    root = ET.fromstring('<registry><streams><stream name="restart">'+members+
                         '</stream></streams></registry>')
    rows = [dict(name=name, pool='state', components=[])
            for name in ('u_init', 'v_init', 't_init', 'qv_init')]
    rows.insert(2, dict(name='scalars', pool='state', components=['qv', 'qc']))
    layout = SimpleNamespace(root=root, fields=rows, enabled=lambda node: True, packages=set())
    path = tmp_path/'streams.atmosphere'
    path.write_text('<streams><immutable_stream name="restart" type="output" '
                    'filename_template="restart.nc"/></streams>')
    definition, = read_streams(path, layout)
    assert definition.fields == ('scalars', 'u_init', 'v_init', 't_init', 'qv_init')


@pytest.mark.parametrize('seed,expected', [
    (0, ['jy9757acx1', 'eve7nmoigs']),
    (1, ['jy9757acx1', 'eve7nmoigs']),
    (1760000000, ['z59r8fo96c', '3uobli7y3e']),
    (0x80000000, ['doknas2fth', '5hjkl2f3z5']),
    (0xffffffff, ['fssxk6mf8k', 'gl671laxni'])])
def test_native_file_ids_retain_gnu_rand_words(seed, expected):
    # Independent native random_id.c witness is in the lane proof receipts.
    from woof.hex.mpasphys.stream_metadata_v842 import NativeFileIds
    ids = NativeFileIds(seed)
    assert [ids.next(), ids.next()] == expected


def test_input_metadata_reaches_writer_without_reference_output(tmp_path, bridge):
    import xml.etree.ElementTree as ET
    from dataclasses import replace
    from woof.hex.output import HistoryField
    from woof.hex.restart_v842 import write_stream
    from woof.hex.streams_v842 import Stream as Definition
    state, fields, _, start = session(resident=False)
    state.layout.root = ET.fromstring('<registry model="mpas" core="atmosphere" version="8.4.2"/>')
    state.layout.config_kinds = dict(config_dt='real', config_number_of_blocks='integer',
                                   config_do_restart='logical')
    state.layout.configs.update(config_dt=60., config_number_of_blocks=1)
    state.output_config_snapshot = dict(state.layout.configs)
    # Core hydration may resolve an originally unspecified config after native
    # streams have copied its namelist value. Output retains the earlier word.
    state.layout.configs['config_dt'] = 120.
    definition = Definition('input', dict(type='output', filename_template='input.nc'),
                            (('var', 'qv', ''),))
    mesh = SimpleNamespace(dimensions=state.layout.dimensions, arrays={}, attrs={})
    write_stream(tmp_path/'input.nc', definition, mesh,
        {'qv': HistoryField(fields['state', 'scalars'][0, :, :3], ('nVertLevels', 'nCells'))},
        start, initial_time=start, field_attrs={'qv': {'units': 'input units', 'extra': 'unallocated'}},
        global_attrs=dict(file_id='parentfile', parent_id='ancestor\n', mesh_spec='1.1',
                          sphere_radius={'dtype': 'f4', 'values': 6371229.}, arbitrary='not forwarded'))
    input_stream = replace(stream(), name='input', directions=('input',), fields=('qv',),
                           input_file=str(tmp_path/'input.nc'))
    with PhysicsStreams(state, (input_stream, stream()), tmp_path,
                        final_time=start+timedelta(minutes=2)) as writer:
        writer.write(step=0, dt=60, receipt={})
        seed = writer.metadata_receipt['file_id_seed']
        writer.write(step=1, dt=60, receipt={})
    header = inspect_stream(tmp_path/'history.nc', scratch=tmp_path/'inspect')
    assert header['global_attribute_order'][-1] == 'file_id'
    assert header['global_attrs']['parent_id'] == 'parentfile\nancestor\n'
    assert header['global_attrs']['config_dt'] == {'dtype': 'f4', 'values': [60.]}
    assert header['global_attrs']['config_number_of_blocks'] == {'dtype': 'i4', 'values': [1]}
    assert header['global_attrs']['config_do_restart'] == 'NO'
    assert 'arbitrary' not in header['global_attrs']
    assert header['variables']['qv']['attrs']['units'] == 'input units'
    assert 'extra' not in header['variables']['qv']['attrs']
    assert header['variables']['qv']['attribute_order'] == ['units', 'long_name']
    from woof.hex.mpasphys.stream_metadata_v842 import NativeFileIds
    assert header['global_attrs']['file_id'] == NativeFileIds(seed).next()


@pytest.mark.parametrize('sphere,periodic,expected_periodic', [('YES', 'YES', 'NO'),
                                                           ('NO', 'YES', 'YES'),
                                                           ('NO', 'NO', 'NO')])
def test_metadata_preserves_native_geometry_rules(sphere, periodic, expected_periodic):
    import xml.etree.ElementTree as ET
    from woof.hex.mpasphys.stream_metadata_v842 import global_attributes
    layout = SimpleNamespace(root=ET.fromstring('<registry model="mpas" core="atmosphere" version="8.4.2"/>'),
                             config_kinds={}, configs={})
    attrs = global_attributes(layout, dict(global_attrs=dict(on_a_sphere=sphere,
        is_periodic=periodic, x_period={'dtype': 'f8', 'values': [123.]}, y_period=456.)))
    assert attrs['is_periodic'] == expected_periodic
    assert attrs['x_period'] == {'dtype': 'f4', 'values': 123. if expected_periodic == 'YES' else 0.}
    assert attrs['sphere_radius'] == {'dtype': 'f4', 'values': 1.}
    assert attrs['parent_id'] == ''
    assert attrs['mesh_spec'] == '0.0'


@pytest.mark.parametrize('resident', [False, True])
def test_resident_diagnostic_updates_between_outputs_and_resets_after_capture(tmp_path, bridge, resident):
    """A maximum must include intervening steps, then reset after its snapshot.

    The provider holds authoritative device words while host pools stay stale.
    Use the real asynchronous Rust writer to catch premature accumulator resets.
    """
    from dataclasses import replace
    state, _, reads, start = session(resident=resident)
    state.layout.fields.append(dict(name='ke', pool='diag', pool_name='ke',
        dimensions=['nVertLevels', 'nCells'], kind='real', components=[]))
    definition = replace(stream(), fields=('ke', 'xtime'))

    class Diagnostics:
        produced = {'ke'}
        accumulates = True

        def __init__(self):
            self.value = np.zeros((2, 4), np.float32).view(DeviceArray)
            self.events = []

        def bind(self, writer):
            self.writer = writer

        def before_write(self, *, step, when):
            self.events.append(('update', step))
            self.value += step + 1
            self.when = when

        def after_snapshot(self):
            assert self.writer.alarms.due('output', self.when)
            self.events.append(('reset', self.when))
            self.value.fill(0)

        def field(self, pool, name):
            assert (pool, name) == ('diag', 'ke')
            return self.value

        def receipt(self):
            return dict(events=len(self.events))

    diagnostics = Diagnostics()
    receipt = {}
    with PhysicsStreams(state, (definition,), tmp_path,
            final_time=start+timedelta(minutes=1), diagnostics=diagnostics) as writer:
        assert writer.missing == {'output': []}
        for step in range(3):
            # Accumulation alone no longer forces a publication: step 1 has
            # no due stream, and the forecast defers its maxima update to the
            # next synchronize (ResidentOutputDiagnostics.defer). write still
            # folds them when called.
            assert writer.required(start+timedelta(seconds=step*30)) is (step != 1)
            writer.write(step=step, dt=30, receipt=receipt)
    assert diagnostics.events == [('update', 0), ('reset', start), ('update', 1),
        ('update', 2), ('reset', start+timedelta(minutes=1))]
    with read_stream(tmp_path/'history.nc', scratch=tmp_path/'read') as output:
        np.testing.assert_array_equal(output.variables['ke'].values[0], 1)
        np.testing.assert_array_equal(output.variables['ke'].values[1], 5)
    assert reads == []
    assert len(receipt['output_capture']) == 2
    assert state.device_pools.transfers['publication_download_bytes'] == 2*2*3*4


def test_nonresident_diagnostics_refresh_current_host_authority():
    from woof.hex.mpasphys.resident_output_diagnostics_v842 import ResidentOutputDiagnostics
    state, fields, _, _ = session(resident=False)
    provider = ResidentOutputDiagnostics(state, SimpleNamespace())
    def stale_view(pool, name):
        raise AssertionError('cached slot views can predate host coupling writes')
    state.device_pools.field = stale_view
    first = provider.current('state', 'scalars').copy()
    np.testing.assert_array_equal(first.view(np.uint32), fields['state', 'scalars'].view(np.uint32))
    fields['state', 'scalars'][...] = 23
    np.testing.assert_array_equal(provider.current('state', 'scalars'), 23)
    assert state.device_pools.level == 2


def test_restart_momentum_and_initialized_words_use_their_authority(tmp_path, bridge):
    """Real Rust readback catches stale pools and rounded momentum rebuilding.

    Only diagnostic scheduling is bypassed. The production field provider,
    immutable capture and asynchronous encoder all execute normally.
    """
    from dataclasses import replace
    from woof.hex.mpasphys.resident_output_diagnostics_v842 import ResidentOutputDiagnostics
    state, fields, reads, start = session(resident=True)
    state.layout.dimensions.update(nEdges=4, nVertices=3, nVertLevelsP1=3)
    words = np.array([0x80000000, 0x7fc12345, 1, 0x3f800000], np.uint32)
    initial = {}
    for name, dims, shape in (
            ('ru', ['nVertLevels', 'nEdges'], (2, 4)),
            ('rw', ['nVertLevelsP1', 'nCells'], (3, 3)),
            ('theta_base', ['nVertLevels', 'nCells'], (2, 3)),
            ('circulation', ['nVertLevels', 'nVertices'], (2, 3))):
        value = np.resize(words, shape).view(np.float32)
        initial[name] = value.copy()
        fields['diag', name] = np.pad(value, [(0, 0), (0, 1)])
        state.layout.fields.append(dict(name=name, pool='diag', pool_name=name,
            dimensions=dims, kind='real', components=[]))
    momentum = SimpleNamespace(rho_u=initial['ru'].copy().view(DeviceArray),
                               rho_w=initial['rw'].copy().view(DeviceArray))
    dycore = SimpleNamespace(atmosphere=SimpleNamespace(state=momentum))
    provider = ResidentOutputDiagnostics(state, dycore)
    # Emulate initialized host allocations. Mutable momentum pools stay stale.
    def get(pool, name, *, level=1):
        assert name in ('theta_base', 'circulation')
        reads.append((pool, name))
        return fields[pool, name].copy()
    state.native.get_field = get
    state.device_pools.cp = SimpleNamespace(asarray=lambda x: np.asarray(x).view(DeviceArray),
        ascontiguousarray=lambda x: x.copy().view(DeviceArray), uint32=np.uint32)
    provider.cp = state.device_pools.cp
    provider.bind = lambda writer: None
    provider.before_write = lambda **kw: None
    provider.after_snapshot = lambda: None
    provider.accumulates = False
    definition = replace(stream(), fields=(*initial, 'xtime'))
    receipt = {}
    with PhysicsStreams(state, (definition,), tmp_path,
            final_time=start+timedelta(minutes=1), diagnostics=provider) as writer:
        assert writer.missing == {'output': []}
        writer.write(step=0, dt=60, receipt=receipt)
        momentum.rho_u.fill(19)
        momentum.rho_w.fill(-23)
        for name in initial:
            fields['diag', name].fill(-999)
        writer.write(step=1, dt=60, receipt=receipt)
    with read_stream(tmp_path/'history.nc', scratch=tmp_path/'read') as output:
        assert list(output.variables) == [*initial, 'xtime']
        for name, value in initial.items():
            actual = output.variables[name].values
            np.testing.assert_array_equal(actual[0].view(np.uint32), value.T.view(np.uint32))
            if name in ('ru', 'rw'):
                np.testing.assert_array_equal(actual[1], 19 if name == 'ru' else -23)
            else:
                np.testing.assert_array_equal(actual[1].view(np.uint32), value.T.view(np.uint32))
    assert reads == [('diag', 'theta_base'), ('diag', 'circulation')]


def test_acoustic_restart_words_are_retained_only_for_due_frames(tmp_path, bridge):
    """Keep the exact final perturbations, then release after immutable capture.

    A committed framework clock wins over an unrelated loop counter. Ordinary
    steps retain no acoustic allocation. Stale host pools cannot supply output.
    """
    from dataclasses import replace
    from woof.hex.mpasphys.resident_output_diagnostics_v842 import ResidentOutputDiagnostics
    state, fields, reads, start = session(resident=True)
    state.layout.dimensions.update(nEdges=4, nVertLevelsP1=3)
    initial = {}
    for name, dims, shape in (
            ('ru_p', ['nVertLevels', 'nEdges'], (2, 4)),
            ('rw_p', ['nVertLevelsP1', 'nCells'], (3, 3))):
        initial[name] = np.zeros(shape, np.float32)
        fields['diag', name] = np.pad(initial[name], [(0, 0), (0, 1)])
        state.layout.fields.append(dict(name=name, pool='diag', pool_name=name,
            dimensions=dims, kind='real', components=[]))
    def get(pool, name, *, level=1):
        reads.append((pool, name))
        assert name in initial
        return fields[pool, name].copy()
    state.native.get_field = get
    state.device_pools.cp = SimpleNamespace(asarray=lambda x: np.asarray(x).view(DeviceArray),
        ascontiguousarray=lambda x: x.copy().view(DeviceArray), uint32=np.uint32)
    committed = [start]
    state.native.clock_state = SimpleNamespace(current_time=SimpleNamespace(
        stamp=lambda: committed[0].strftime('%Y-%m-%d_%H:%M:%S')))
    provider = ResidentOutputDiagnostics(state, SimpleNamespace())
    provider.convective = SimpleNamespace(reset=lambda: None)
    def bind(writer):
        provider.writer = writer
    def before_write(*, step, when):
        provider.last_step = step
    provider.bind = bind
    provider.before_write = before_write
    provider.accumulates = False
    definition = replace(stream(), output_interval='00:02:00', fields=(*initial, 'xtime'))
    words = np.array([0x80000000, 1, 0x7fc23456, 0x3f800000], np.uint32)
    acoustic = {name: np.resize(words, value.shape).view(np.float32).view(DeviceArray)
                for name, value in initial.items()}
    expected = {name: value.copy() for name, value in acoustic.items()}
    with PhysicsStreams(state, (definition,), tmp_path,
            final_time=start+timedelta(minutes=2), diagnostics=provider) as writer:
        assert writer.missing == {'output': []}
        writer.write(step=0, dt=60, receipt={})
        assert provider.prepare_step(step=1, dt=60) is None
        assert not provider.acoustic
        committed[0] += timedelta(minutes=1)
        assert not writer.write(step=1, dt=60, receipt={})
        callback = provider.prepare_step(step=999, dt=60)
        assert callback is not None
        callback(**acoustic)
        assert all(provider.field('diag', name) is value for name, value in acoustic.items())
        assert provider.acoustic_captures == 1
        for name in initial:
            fields['diag', name].fill(-999)
        committed[0] += timedelta(minutes=1)
        writer.write(step=2, dt=60, receipt={})
        assert not provider.acoustic
        with pytest.raises(RuntimeError, match='final acoustic output was not retained'):
            provider.field('diag', 'ru_p')
        for value in acoustic.values():
            value.fill(-1)
    assert not provider.buffers and not provider.acoustic
    with read_stream(tmp_path/'history.nc', scratch=tmp_path/'read') as output:
        for name in initial:
            actual = output.variables[name].values
            np.testing.assert_array_equal(actual[0], 0)
            np.testing.assert_array_equal(actual[1].view(np.uint32),
                                          expected[name].T.view(np.uint32))
    assert reads == [('diag', 'ru_p'), ('diag', 'rw_p')]


def test_a_route_declared_omission_leaves_the_hex_written_default_stream(tmp_path):
    """hrrrv5 hex region (2026-10-29): the RRFS fork's Registry lists the PV diagnostics in its default output stream,
    the RRFS route computes none, and every suite-named run with no deck refused at its first write.  The route row's
    default_streams_omit drops them from the streams hex writes; the row says why, and a deck is still held to them."""
    import json
    from pathlib import Path
    import xml.etree.ElementTree as ET

    from woof.hex.mpasphys.forecast_v842 import write_default_streams

    registry = ET.fromstring(
        '<registry><streams><stream name="output" type="output" filename_template="history.nc" '
        'output_interval="6:00:00" runtime_format="separate_file"><var name="theta"/><var name="ertel_pv"/>'
        '</stream></streams></registry>')
    layout = SimpleNamespace(root=registry, fields=[])
    write_default_streams(layout, tmp_path / 'plain' / 'streams.atmosphere')
    assert (tmp_path / 'plain' / 'stream_list.atmosphere.output').read_text().split() == ['theta', 'ertel_pv']
    table = json.loads((Path(__file__).resolve().parents[1] / '../../woof/hex/data/profiles/'
                        'mpas-physics-suites.v1.json').read_text(encoding='utf-8'))
    omit = table['mpasphys_rrfs']['default_streams_omit']
    write_default_streams(layout, tmp_path / 'rrfs' / 'streams.atmosphere', omit=omit)
    assert (tmp_path / 'rrfs' / 'stream_list.atmosphere.output').read_text().split() == ['theta']
    assert table['mpasphys_rrfs_44c0849']['default_streams_omit'] == omit


def test_a_default_stream_member_with_a_zero_length_dimension_is_left_out(tmp_path):
    """hrrrv5 hex region (2026-10-29): the fork's chemistry Registry puts ddvel (nChem x nCells, no package) in the
    default output stream; nChem is num_chem = 0 without smoke and the first write refused.  Only the unlimited
    dimension may be zero-length in a NetCDF file, so the default streams hex writes leave such a member out."""
    import xml.etree.ElementTree as ET

    from woof.hex.mpasphys.forecast_v842 import write_default_streams

    registry = ET.fromstring(
        '<registry><streams><stream name="output" type="output" filename_template="history.nc" '
        'output_interval="6:00:00" runtime_format="separate_file"><var name="theta"/><var name="ddvel"/>'
        '</stream></streams></registry>')
    fields = [{'name': 'theta', 'dimensions': ['nVertLevels', 'nCells']},
              {'name': 'ddvel', 'dimensions': ['nChem', 'nCells']}]
    for chem, expected in ((0, ['theta']), (2, ['theta', 'ddvel'])):
        layout = SimpleNamespace(root=registry, fields=fields,
                                 dimensions={'nVertLevels': 59, 'nCells': 10, 'nChem': chem})
        write_default_streams(layout, tmp_path / str(chem) / 'streams.atmosphere')
        assert (tmp_path / str(chem) / 'stream_list.atmosphere.output').read_text().split() == expected


def test_an_explicitly_bound_registry_input_stream_is_read_once_at_the_start(tmp_path):
    """RRFS v2 hrrrv5 on WOOF (RTX 5090 worker proof, 2026-10-30): the suite-named forecast refused
    --stream-input invariant=FILE as not an active input stream, because the RRFS fork's Registry declares
    invariant immutable, in_defaults=false and input_interval=none, so the default streams left it out.
    A registered input stream the run binds is now written at initial_only; unbound it stays out as before
    (WOOF row rrfs-explicit-input-streams-v1 retires on this release)."""
    import xml.etree.ElementTree as ET

    from woof.hex.mpasphys.forecast_v842 import write_default_streams

    registry = ET.fromstring(
        '<registry><streams><stream name="input" type="input" filename_template="init.nc" input_interval="initial_only"'
        ' immutable="true"/><stream name="invariant" type="input" filename_template="invariant.nc" '
        'input_interval="none" immutable="true" in_defaults="false"/></streams></registry>')
    layout = SimpleNamespace(root=registry, fields=[])

    def streams(path, **kwargs):
        write_default_streams(layout, path, **kwargs)
        return {e.get('name'): e for e in ET.parse(path).getroot()}

    unbound = streams(tmp_path / 'plain' / 'streams.atmosphere')
    assert 'invariant' not in unbound and 'input' in unbound
    bound = streams(tmp_path / 'bound' / 'streams.atmosphere', bound_inputs={'invariant': 'invariant.nc'})
    assert bound['invariant'].get('input_interval') == 'initial_only'
    assert bound['input'].get('input_interval') == 'initial_only'
