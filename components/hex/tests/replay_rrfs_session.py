"""Replay the complete stream session on witnessed strict native forecast fields.

The atmosphere state stays native: this is stream-layer evidence, not a Hex
forecast. Native wall-clock random seeds are inferred from authenticated file
IDs, then the actual session generates IDs, schedules streams and writes them.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timedelta
import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from woof.hex.output import HistoryField
from woof.hex.output_rrfs import StreamSession, native_file_id
from woof.hex.stream_authority_rrfs import NativePoolStreamAuthority
from woof.hex.mpasphys.native_backend import NativePhysics
from woof.hex.mpasphys.pool_v842 import RegistryLayout
from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist, specification
from woof.hex.restart_rrfs import read_stream, done_marker_enabled, write_done_marker
from woof.hex.streams_rrfs import parse_streams
from replay_rrfs_native_io import native_packages, sha
from replay_rrfs_native_pair import replay as compare_pair


def replay(args):
    source = args.native_run.resolve(strict=True)
    manifest = json.loads((source / 'MANIFEST.json').read_text())
    streams = parse_streams(source / 'streams.atmosphere')
    packages, package_authority = native_packages(source, manifest, streams)
    clock_files = {}
    initial = None
    for name, digest in manifest['outputs'].items():
        if not name.endswith('.nc'):
            continue
        if sha(source / name) != digest:
            raise ValueError('native file changed: ' + name)
        with read_stream(source / name, scratch=args.output.parent, select=('xtime', 'initial_time')) as clock:
            stamps = clock.variables['xtime'].text
            initial_text = clock.variables['initial_time'].text[0]
        if len(stamps) != 1:
            raise ValueError('this complete workflow session requires one record per native file')
        initial = datetime.strptime(initial_text, '%Y-%m-%d_%H:%M:%S')
        stamp = datetime.strptime(stamps[0], '%Y-%m-%d_%H:%M:%S')
        candidates = [s for s in streams.values() if 'output' in s.options['type'].split(';')
                      and s.filename(stamp, initial) == name]
        if len(candidates) != 1:
            raise ValueError('one native stream binding required')
        clock_files.setdefault(stamp, {})[candidates[0].name] = name
    settings = manifest['resolved_configuration']['resolved_settings']
    library = Path(os.environ['RRFS_NATIVE_STREAM_LIBRARY']).resolve(strict=True)
    if sha(library) != 'efb9764c714901e9c8f37344b0d879756c74b353f1abf24e23689a1e242bee3e':
        raise ValueError('authenticated r2 fork native pool library changed')
    pool_transfers = []
    reader = library.parent / 'rw_mpas_physics_pool'
    if sha(reader) != '1106a18b743323eb95ae106119bf78a639ff717d3486c13b499c11f68dbcee56':
        raise ValueError('native Registry dimension reader changed')
    input_dimensions, dimension_authorities = {}, []
    for name in ('init.nc', 'invariant.nc', 'ugwp_oro_data.nc'):
        digest = manifest['inputs'][name]
        path = library.parent / 'inputs' / digest
        if sha(path) != digest:
            raise ValueError('native Registry dimension input changed: ' + name)
        input_dimensions.update(json.loads(subprocess.check_output(
            [str(reader), '--inspect', str(path)], text=True)))
        dimension_authorities.append(dict(name=name, sha256=digest))
    start = datetime.strptime(settings['config_start_time'], '%Y-%m-%d_%H:%M:%S')
    args.output.mkdir(parents=True, exist_ok=False)
    session = StreamSession.from_namelist((source / 'namelist.atmosphere').read_text(),
        source / 'streams.atmosphere', start_time=start, initial_time=initial,
        final_time=max(clock_files), output_directory=args.output, active_packages=packages)
    first = next(name for name in session.streams if session.alarms.due(name, min(clock_files)))
    first_path = source / clock_files[min(clock_files)][first]
    with read_stream(first_path, scratch=args.output) as state:
        first_id = state.global_attrs['file_id']
    opened = re.search(r'Opened at (\d{4}/\d\d/\d\d \d\d:\d\d:\d\d)',
                       (source / 'log.atmosphere.0000.out').read_text()).group(1)
    seed_clock = int(datetime.strptime(opened, '%Y/%m/%d %H:%M:%S').timestamp())
    seeds = [seed for seed in range(seed_clock - 10, seed_clock + 11)
             if native_file_id(seed, 0) == first_id]
    if len(seeds) != 1:
        raise ValueError('native file ID did not authenticate exactly one logged wall-clock seed')
    session.random_seed = seeds[0]
    rows = []
    at = start
    dt = timedelta(seconds=float(settings['config_dt']))
    while at <= max(clock_files):
        authority = clock_files.get(at, {})
        with ExitStack() as stack:
            states = {name: stack.enter_context(read_stream(source / filename, scratch=args.output))
                      for name, filename in authority.items()}
            if states:
                dimensions = dict(input_dimensions)
                for state in states.values(): dimensions.update(state.dimensions)
                attrs = dict(next(iter(states.values())).global_attrs)
                attrs.pop('file_id', None)
                attrs.pop('output_interval', None)
                mesh = SimpleNamespace(dimensions=dimensions, arrays={}, attrs={})
                layout = RegistryLayout(Path(os.environ['RRFS_MPAS_REF']),
                    resolve_namelist((source / 'namelist.atmosphere').read_text()),
                    (source / 'namelist.atmosphere').read_text(), dimensions)
                def capture(stream):
                    state = states[stream.name]
                    # Use the actual native ABI rather than a fake pool or a
                    # direct file-to-file capture. Unselected allocated slots
                    # are not compared and are not forecast evidence.
                    with NativePhysics(library, test_only=True,
                            pin_set=specification()['pin_set']) as native:
                        boundary = NativePoolStreamAuthority(native, layout,
                            active_packages=packages, streams=session.streams,
                            global_attrs=attrs, input_directory=source,
                            field_attributes={name: var.attrs for name, var in state.variables.items()})
                        names = tuple(state.variables)
                        selected = {(boundary.bindings[name][0]['pool'],
                                     boundary.bindings[name][0]['pool_name'])
                                    for name in names if name in boundary.bindings}
                        for row in layout.fields:
                            key = (row['pool'], row['pool_name'])
                            if key not in selected:
                                continue
                            if row['kind'] == 'character':
                                native.put_text_field(*key, '', nlevels=row['nlevels'])
                            else:
                                native.put_field(*key, np.zeros(row['shape'],
                                    dtype='f4' if row['kind'] == 'real' else 'i4', order='F'),
                                    nlevels=row['nlevels'])
                        restored = boundary.read_file(source / authority[stream.name],
                            names, record=0, scratch=args.output)
                        if frozenset(restored) != frozenset(names):
                            raise ValueError('native pool boundary omitted real fields')
                        captured = boundary.capture_stream(stream)
                        pool_transfers.append(dict(file=authority[stream.name],
                            restored_fields=len(restored), captured_fields=len(captured),
                            native_pool_fields=len(selected)))
                        return captured
                written = session.publish_due(at, mesh, capture, global_attrs=attrs, scratch=args.output,
                                               restart_timestamp_path=args.output / 'restart_timestamp')
                expected = {authority[name] for name in authority}
                actual = {path.name for path in written}
                if expected != actual:
                    raise ValueError('native/session publication mismatch: ' + str((at, expected, actual)))
                rows.append(dict(time=at.isoformat(), files=sorted(actual)))
            elif any(session.alarms.due(name, at) for name in session.streams):
                raise ValueError('session produced an output absent from native')
        at += dt
    # The generic strict-native comparator authenticates words. This manifest
    # is only a transport index: the receipt explicitly states the native-state
    # scope, and never asserts the replay is an atmosphere executable.
    transported = dict(schema='rrfs-native-stream-session-transport/v1',
        source_native_manifest_sha256=sha(source / 'MANIFEST.json'),
        reference_labels=manifest.get('reference_labels', []),
        init_derivative_authorities=manifest.get('init_derivative_authorities', []),
        transport_completed=True, whole_model_proof=False,
        outputs={p.name: sha(p) for p in args.output.glob('*.nc')})
    (args.output / 'MANIFEST.json').write_text(json.dumps(transported, indent=2) + '\n')
    compared_receipt = args.receipt.with_suffix('.words.json')
    status = compare_pair(SimpleNamespace(left=source, right=args.output,
        right_continuous=False, right_transport=True, comparator=args.comparator, receipt=compared_receipt))
    attrs_identical = True
    markers = []
    for files in clock_files.values():
        for stream_name, name in files.items():
            with read_stream(source / name, scratch=args.output) as left, read_stream(args.output / name, scratch=args.output) as right:
                attrs_identical &= (left.global_attrs == right.global_attrs and
                    {key: var.attrs for key, var in left.variables.items()} ==
                    {key: var.attrs for key, var in right.variables.items()})
            native_marker = source / (name + '.done')
            encoded_marker = args.output / (name + '.done')
            expected = done_marker_enabled(streams[stream_name])
            marker = dict(file=name + '.done', expected=expected,
                passed=native_marker.exists() == encoded_marker.exists() == expected)
            if expected:
                if sha(native_marker) != manifest['outputs'][native_marker.name]:
                    raise ValueError('native marker changed: ' + native_marker.name)
                witnessed = datetime.strptime(native_marker.read_text(), '%Y-%m-%d %H:%M:%S\n')
                marker['replay_wall_clock'] = encoded_marker.read_text().strip()
                write_done_marker(args.output / name, now=witnessed)
                marker.update(native_wall_clock=witnessed.isoformat(),
                    same_instant_bytes_identical=native_marker.read_bytes() == encoded_marker.read_bytes())
                marker['passed'] &= marker['same_instant_bytes_identical']
            markers.append(marker)
    result = dict(schema='rrfs-authenticated-native-stream-session/v1', whole_model_proof=False,
        scope='Actual session publication on native state, Rust words/IDs, no Hex atmosphere executed',
        native_manifest_sha256=sha(source / 'MANIFEST.json'), process_count=1,
        reference_labels=manifest.get('reference_labels', []),
        init_derivative_authorities=manifest.get('init_derivative_authorities', []),
        matched_native_seed=session.random_seed, seed_witness_file=first_path.name,
        generated_file_ids=session._random_ordinal, publications=rows,
        native_pool_library_sha256=sha(library), native_pool_transfers=pool_transfers,
        native_registry_dimension_inputs=dimension_authorities,
        all_global_attributes_identical=attrs_identical, words_receipt=compared_receipt.name,
        package_authority=package_authority, done_markers=markers,
        marker_scope='Native presence and exact serialization at the same witnessed instant; execution wall clocks differ',
        passed=status == 0 and attrs_identical and all(marker['passed'] for marker in markers))
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('native-run', 'output', 'comparator', 'receipt'):
        parser.add_argument('--' + key, type=Path, required=True)
    raise SystemExit(replay(parser.parse_args()))
