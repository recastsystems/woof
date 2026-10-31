"""Complete real-record ingress and append workflows through the native pool.

This exercises service callbacks on original native inputs/outputs. Combined
filenames are explicit replay controls, not an operational forecast claim.
"""
from dataclasses import replace
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np

from woof.hex.mpasphys.native_backend import NativePhysics
from woof.hex.mpasphys.pool_v842 import RegistryLayout
from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist, specification
from woof.hex.output_rrfs import StreamSession
from woof.hex.restart_rrfs import read_stream, RestartControl
from woof.hex.stream_authority_rrfs import NativePoolStreamAuthority
from woof.hex.streams_rrfs import parse_streams, StreamAlarms
from replay_rrfs_native_io import sha, native_packages


def allocate(native, boundary, names):
    selected = {(boundary.bindings[n][0]['pool'], boundary.bindings[n][0]['pool_name'])
                for n in names if n in boundary.bindings}
    for row in boundary.layout.fields:
        key = (row['pool'], row['pool_name'])
        if key not in selected:
            continue
        if row['kind'] == 'character':
            native.put_text_field(*key, '', nlevels=row['nlevels'])
        else:
            native.put_field(*key, np.zeros(row['shape'],
                dtype='f4' if row['kind'] == 'real' else 'i4', order='F'),
                nlevels=row['nlevels'])


def equal_fields(left, right):
    if set(left.variables) != set(right.variables):
        raise ValueError('record variable inventory differs')
    words = 0
    for name, a in left.variables.items():
        b = right.variables[name]
        if (a.dimensions != b.dimensions or a.values.shape != b.values.shape or
                a.values.dtype != b.values.dtype or a.attrs != b.attrs or
                a.values.tobytes() != b.values.tobytes()):
            raise ValueError('record word/axis/attribute mismatch: ' + name)
        words += a.values.size
    return words


def replay_append(source, output, receipt, library):
    manifest = json.loads((source / 'MANIFEST.json').read_text())
    streams = parse_streams(source / 'streams.atmosphere')
    packages, package_authority = native_packages(source, manifest, streams)
    namelist = (source / 'namelist.atmosphere').read_text()
    dimensions = {}
    for name in ('init.nc', 'invariant.nc', 'ugwp_oro_data.nc'):
        path = library.parent / 'inputs' / manifest['inputs'][name]
        assert sha(path) == manifest['inputs'][name]
        dimensions.update(json.loads(subprocess.check_output(
            [str(library.parent / 'rw_mpas_physics_pool'), '--inspect', str(path)], text=True)))
    groups = {}
    for name, digest in manifest['outputs'].items():
        if not name.endswith('.nc'):
            continue
        assert sha(source / name) == digest
        with read_stream(source / name, scratch=output, select=('xtime', 'initial_time')) as state:
            at = datetime.strptime(state.variables['xtime'].text[0], '%Y-%m-%d_%H:%M:%S')
            initial = datetime.strptime(state.variables['initial_time'].text[0], '%Y-%m-%d_%H:%M:%S')
        stream = next(s for s in streams.values() if 'output' in s.options['type'].split(';')
                      and s.filename(at, initial) == name)
        groups.setdefault(stream.name, []).append((at, name, initial))
    rows = []
    for stream_name, records in groups.items():
        records.sort()
        # The replay uses native fields, native intervals and native clobber
        # policy. Only filename rotation is disabled to exercise open files.
        original = streams[stream_name]
        stream = replace(original, options=dict(original.options,
            filename_interval='none', filename_template=stream_name + '.nc'))
        selected_streams = {name: replace(s, options=dict(s.options, type='none'))
                            for name, s in streams.items()}
        selected_streams[stream_name] = stream
        _, _, initial = records[0]
        start = datetime.strptime(manifest['resolved_configuration']['resolved_settings']['config_start_time'],
                                  '%Y-%m-%d_%H:%M:%S')
        end = records[-1][0]
        control = RestartControl.from_namelist(namelist)
        session = StreamSession(selected_streams,
            StreamAlarms(selected_streams, start, initial, end, active_packages=packages,
                do_restart=control.config_do_restart, do_DAcycling=control.config_do_DAcycling),
            control, output, random_seed=1234567)
        for index, (at, name, _) in enumerate(records):
            with read_stream(source / name, scratch=output) as state:
                dims = dict(dimensions, **state.dimensions)
                layout = RegistryLayout(Path(os.environ['RRFS_MPAS_REF']),
                    resolve_namelist(namelist), namelist, dims)
                attrs = dict(state.global_attrs)
                attrs.pop('file_id', None)
                field_attrs = {n: v.attrs for n, v in state.variables.items()}
                with NativePhysics(library, test_only=True, pin_set=specification()['pin_set']) as native:
                    boundary = NativePoolStreamAuthority(native, layout, active_packages=packages,
                        streams=selected_streams, global_attrs=attrs, input_directory=source,
                        field_attributes=field_attrs)
                    allocate(native, boundary, state.variables)
                    boundary.read_file(source / name, tuple(state.variables), record=0, scratch=output)
                    indices = boundary.record_indices(session, at, scratch=output)
                    assert indices == ({} if index == 0 else {stream_name: index})
                    written = session.publish_due(at, boundary.mesh, boundary.capture_stream,
                        global_attrs=attrs, scratch=output, record_indices=indices)
                    assert written == (output / (stream_name + '.nc'),)
            with read_stream(source / name, scratch=output, record=0) as expected, \
                    read_stream(written[0], scratch=output, record=index) as actual:
                words = equal_fields(expected, actual)
            rows.append(dict(stream=stream_name, native_file=name, native_sha256=sha(source / name),
                             record=index, words=words, differing_words=0, passed=True))
        with read_stream(output / (stream_name + '.nc'), scratch=output, select=('xtime',)) as state:
            assert len(state.variables['xtime'].text) == len(records)
            assert session._random_ordinal == 1
    result = dict(schema='r11-real-native-pool-append/v1', passed=True,
        whole_model_proof=False, scope='Native fields; only filename rotation disabled for replay',
        native_manifest_sha256=sha(source / 'MANIFEST.json'), package_authority=package_authority,
        reference_labels=manifest.get('reference_labels', []),
        init_derivative_authorities=manifest.get('init_derivative_authorities', []),
        records=rows, words=sum(r['words'] for r in rows), differing_words=0)
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    return result


def replay_external(args):
    dependency = json.loads(args.authority.read_text())
    assert sha(Path(dependency['publication_path'])) == dependency['publication_sha256']
    library = Path(os.environ['RRFS_NATIVE_STREAM_LIBRARY'])
    assert sha(library) == dependency['native_library_sha256']
    rows = []
    for case in dependency['cases']:
        source = args.authority.parent.parent / 'artifacts/rrfs-native-stream-pool/inputs'
        path = source / case['input_sha256']
        assert sha(path) == case['input_sha256']
        deck = Path(os.environ['RRFS_PUBLIC_CI_DECKS']) / case['deck']
        for name, digest in case['deck_files'].items():
            assert sha(deck / name) == digest
        streams = parse_streams(deck / 'streams.atmosphere')
        stream = streams['surface']
        namelist = (deck / 'namelist.atmosphere').read_text()
        with read_stream(path, scratch=args.output, select=('xtime',)) as clock:
            stamps = clock.variables['xtime'].text
        times = [datetime.strptime(t, '%Y-%m-%d_%H:%M:%S') for t in stamps]
        with read_stream(path, scratch=args.output) as state:
            dims = dict(state.dimensions)
            field_attrs = {n: v.attrs for n, v in state.variables.items()}
        # Complete Registry allocation dimensions come from the actual paired init.
        init = source / case['init_sha256']
        assert sha(init) == case['init_sha256']
        dims.update(json.loads(subprocess.check_output(
            [str(library.parent / 'rw_mpas_physics_pool'), '--inspect', str(init)], text=True)))
        layout = RegistryLayout(Path(os.environ['RRFS_MPAS_REF']), resolve_namelist(namelist), namelist, dims)
        session = StreamSession({stream.name: stream}, None, RestartControl(False, False), args.output)
        session.alarms = SimpleNamespace(initial_time=times[0])
        input_dir = args.output / case['deck']
        input_dir.mkdir()
        (input_dir / stream.filename(times[0], times[0])).symlink_to(path)
        targets = [times[0] - timedelta(seconds=1), times[-1] + timedelta(seconds=1)]
        targets += times + [t + (u-t)/2 for t, u in zip(times, times[1:])]
        names = stream.fields((), streams=streams)
        with NativePhysics(library, test_only=True, pin_set=specification()['pin_set']) as native:
            boundary = NativePoolStreamAuthority(native, layout, active_packages=(),
                streams=streams, global_attrs={}, input_directory=input_dir, field_attributes=field_attrs)
            allocate(native, boundary, names)
            for when in sorted(targets):
                request = f'{len(stamps)} 102\n{when:%Y-%m-%d_%H:%M:%S}\n' + '\n'.join(stamps) + '\n'
                response = subprocess.check_output([str(args.seek)], input=request, text=True).strip()
                record = int(response.split('|')[0]) - 1
                if record < 0:
                    snapshots = boundary.capture_stream(stream)
                    try:
                        boundary.read_stream_latest_before(stream, when, session, scratch=args.output)
                    except ValueError as error:
                        assert 'no native record' in str(error)
                    else:
                        raise AssertionError('native seek refusal was ignored')
                    after = boundary.capture_stream(stream)
                    assert all(snapshots[n].values.tobytes() == after[n].values.tobytes() for n in names)
                    rows.append(dict(case=case['deck'], when=when.isoformat(), native_seek=response,
                                     refused=True, words=0, differing_words=0, passed=True))
                    continue
                restored = boundary.read_stream_latest_before(stream, when, session, scratch=args.output)
                assert frozenset(restored) == frozenset(names)
                captured = boundary.capture_stream(stream)
                with read_stream(path, scratch=args.output, select=names, record=record) as expected:
                    words = 0
                    for n in names:
                        assert expected.variables[n].values.tobytes() == captured[n].values.tobytes(), n
                        words += expected.variables[n].values.size
                rows.append(dict(case=case['deck'], when=when.isoformat(), native_seek=response,
                                 record=record, words=words, differing_words=0, passed=True))
    result = dict(schema='r11-real-native-pool-external-records/v1', passed=True,
        whole_model_proof=False, scope='Whole fork native seek on real clocks; native pool real surface fields',
        authority_sha256=sha(args.authority), seek_sha256=sha(args.seek), cases=rows,
        words=sum(r['words'] for r in rows), differing_words=0)
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('authority', 'seek', 'output', 'receipt'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raise SystemExit(replay_external(args))
