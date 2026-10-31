"""Replay RRFS stream word transport on a stated set of real r1 tap fields.

The surface scheme's post-call words are INPUTS to the stream service. This
harness does not replay that scheme or claim a forecast. Native stream timing,
XML, packages and warm-start policy are independently compared by the complete
compiled-native stream test suite. Replay clock coordinates below exercise the
service; they are not attributed to the captured forecast's valid times.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from woof.hex.output import HistoryField
from woof.hex.output_rrfs import StreamSession
from woof.hex.restart_rrfs import RestartControl, read_restart, read_stream
from woof.hex.streams_rrfs import Stream, StreamAlarms, field_definition

PUBLICATION_SHA = '6694ed48bc5ebcac9da3b2987f459c84dcd6322e5edb425284484164f246fee7'
ALLOCATION_PACKAGES = frozenset({'bl_mynnedmf_in'})
FIELDS = {'t2': 't2m', 'th2': 'th2m', 'q2': 'q2'}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def call(oracle, operation, *args):
    result = subprocess.run([str(oracle), operation, *map(str, args)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout) if result.stdout.strip() else None


def replay(args):
    root = args.capture_root.resolve(strict=True)
    if sha(root / 'publication.json') != PUBLICATION_SHA:
        raise ValueError('r1 native publication changed')
    publication = json.loads((root / 'publication.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    manifests = sorted(root.glob('*.manifest.json'))
    if len(manifests) != 2:
        raise ValueError('the stated replay requires both rank-zero step-one and step-three GFS captures')
    rows = []
    for ordinal, manifest_path in enumerate(manifests):
        if sha(manifest_path) != publication['members'][manifest_path.name]['sha256']:
            raise ValueError('r1 native manifest changed')
        manifest = json.loads(manifest_path.read_text())
        if manifest['reference'] != 'rrfs' or manifest['process_count'] != 1:
            raise ValueError('native RRFS rank-one authority required')
        capture = manifest_path.name.removesuffix('.manifest.json')
        directory = args.output / capture
        directory.mkdir()
        fields, originals, metadata = {}, {}, {}
        for native_name, stream_name in FIELDS.items():
            filename = f'{capture}.post.{native_name}.bin'
            source = root / 'captures' / filename
            if sha(source) != manifest['files'][filename]['sha256']:
                raise ValueError('r1 captured field changed: ' + filename)
            raw = directory / (native_name + '.raw')
            header = call(args.oracle, 'header', source, '-')
            if header['code'] != 1 or len(header['shape']) != 1 or header['mpi_rank'] != 0:
                raise ValueError('stated native surface capture must be a rank-one RKIND field')
            call(args.oracle, 'raw', source, raw)
            definition = field_definition(stream_name)
            dims = tuple(d for d in definition['dimensions'] if d != 'Time')
            if dims != ('nCells',):
                raise ValueError('native Registry surface field layout changed')
            # Views on Rust-decoded payload; no field decode or arithmetic here.
            array = np.memmap(raw, mode='r', dtype='=f4', shape=tuple(header['shape']))
            fields[stream_name] = HistoryField(array, dims)
            originals[stream_name] = source
            metadata[stream_name] = header
        shape = {tuple(field.values.shape) for field in fields.values()}
        if len(shape) != 1:
            raise ValueError('native surface field extents disagree')
        mesh = SimpleNamespace(dimensions={'nCells': next(iter(shape))[0]}, arrays={}, attrs={})
        members = tuple(('var', name, '') for name in ('xtime', 'initial_time', *fields))
        base = dict(type='input;output', input_interval='initial_only',
                    output_interval='01:00:00', filename_interval='output',
                    output_timelevels='0 1 3', clobber_mode='truncate',
                    precision='single', io_type='pnetcdf', output_done_marker='yes', gattr_update='yes')
        streams = {name: Stream(name, dict(base, filename_template=name + '.$Y-$M-$D_$h.$m.$s.nc'), members)
                   for name in ('output', 'da_state', 'restart')}
        start = datetime(2000, 1, 1)  # Replay coordinate, not a forecast attribution.
        for mode, control in (('cold', RestartControl()), ('cycling', RestartControl(False, True)),
                              ('restart', RestartControl(True, False))):
            out = directory / mode
            alarms = StreamAlarms(streams, start, start, start + timedelta(hours=3),
                                  do_restart=control.config_do_restart,
                                  do_DAcycling=control.config_do_DAcycling,
                                  active_packages=ALLOCATION_PACKAGES)
            session = StreamSession(streams, alarms, control, out, random_seed=123456)
            emitted = []
            for lead in (0, 1, 3):
                time = start + timedelta(hours=lead)
                paths = session.publish_due(time, mesh, lambda stream: fields, scratch=directory)
                if mode == 'cycling' and lead == 0 and any(p.name.startswith('da_state.') for p in paths):
                    raise AssertionError('DA cycling emitted an initial mpasout')
                for path in paths:
                    with read_stream(path, scratch=directory, record=0) as state:
                        for name, source in originals.items():
                            decoded = directory / 'decoded.raw'
                            assert state.variables[name].values.size == metadata[name]['shape'][0]
                            state.variables[name].values.tofile(decoded)
                            compared = call(args.oracle, 'compare-raw', source, decoded)
                            if compared['bitwise_equal'] is not True or compared['different_words'] != 0:
                                raise AssertionError(compared)
                            rows.append(dict(capture=capture, mode=mode, replay_lead=lead,
                                             stream=path.name.split('.')[0], field=name,
                                             input_sha256=sha(source), metadata=metadata[name], comparison=compared))
                    if not path.with_name(path.name + '.done').is_file():
                        raise AssertionError('done marker absent')
                    emitted.append(path.name)
            # DA/restart ingress must also return the complete real field words.
            for name in ('da_state', 'restart'):
                source = out / streams[name].filename(start + timedelta(hours=1), start)
                with read_restart(source, valid_time=start + timedelta(hours=1), scratch=directory,
                                  stream=streams[name], streams=streams, authority_layout=True,
                                  active_packages=ALLOCATION_PACKAGES) as state:
                    for field, original in originals.items():
                        decoded = directory / 'decoded.raw'
                        assert state.variables[field].values.size == metadata[field]['shape'][0]
                        state.variables[field].values.tofile(decoded)
                        compared = call(args.oracle, 'compare-raw', original, decoded)
                        if not compared['bitwise_equal'] or compared['different_words']:
                            raise AssertionError(compared)
                        rows.append(dict(capture=capture, mode=mode, stream=name, operation='ingress',
                                         field=field, input_sha256=sha(original), comparison=compared))
        for field in fields.values():
            field.values._mmap.close()
    result = dict(schema='rrfs-real-native-tap-stream-replay/v1',
                  native_publication_sha256=sha(root / 'publication.json'), process_count=1,
                  allocation_packages=sorted(ALLOCATION_PACKAGES),
                  allocation_scope='Registry allocation gate for captured arrays; no claim about the original resolved physics suite',
                  stated_cases=[p.name for p in manifests], fields=list(FIELDS.values()),
                  words=sum(r['comparison']['words'] for r in rows), different_words=0,
                  all_output_words_identical=True, rows=rows,
                  scope='Stream service preservation of real native surface fields; synthetic replay clock coordinates',
                  native_scheme_replayed=False, actual_rrfs_forecast=False)
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    replay(parser.parse_args())
