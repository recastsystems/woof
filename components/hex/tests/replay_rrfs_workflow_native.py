"""Exercise workflow streams and DA ingress with the published strict fork.

This uses the native CPU stand-in, not a Hex forecast. It fills the original
workflow template as exrrfs_fcst.sh:69-100 does, and preserves every stream list.
"""
import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
from datetime import datetime

from woof.hex.streams_rrfs import parse_streams


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(args):
    original = json.loads((args.native_deck.parent / 'INDEX.json').read_text())
    row = next(r for r in original['decks'] if r['deck'] == args.native_deck.name)
    for name, digest in row['files'].items():
        if sha(args.native_deck / name) != digest:
            raise ValueError('published original deck changed: ' + name)
    original_start = datetime.strptime(row['resolved_settings']['config_start_time'], '%Y-%m-%d_%H:%M:%S')
    start = datetime.strptime(args.start_time, '%Y-%m-%d_%H:%M:%S') if args.start_time else original_start
    template = args.workflow / 'parm/streams.atmosphere'
    text = template.read_text()
    text = text.replace('output_interval="@mpasout_interval@"',
                        'output_timelevels="' + args.output_hours + '"')
    values = dict(restart_interval=args.interval, history_interval=args.interval,
                  diag_interval=args.interval, lbc_interval=args.lbc_hours)
    for name, value in values.items():
        text = text.replace('@' + name + '@', str(value))
    if '@' in text:
        raise ValueError('unfilled workflow stream placeholder')
    args.work_root.mkdir(parents=True, exist_ok=False)
    decks = args.work_root / 'decks'
    decks.mkdir()
    deck = decks / ('warm' if args.warm_state else 'cold')
    deck.mkdir()
    source_text = (args.native_deck / 'namelist.atmosphere').read_text()
    source_text = source_text.replace('&restart', '&restart\n config_do_DAcycling = ' +
                                     ('.true.' if args.warm_state else '.false.'), 1)
    jedi_da = args.jedi_da == 'true'
    source_text += '\n&assimilation\n config_jedi_da = ' + ('.true.' if jedi_da else '.false.') + '\n/\n'
    overlays = dict(config_jedi_da=jedi_da, config_do_DAcycling=bool(args.warm_state))
    for key, value in (('config_start_time', args.start_time),
                       ('config_run_duration', args.duration),
                       ('config_do_restart', True if args.restart_state else None)):
        if value is not None:
            literal = '.true.' if value is True else "'" + value + "'"
            source_text, count = re.subn(r'(?mi)^(\s*' + key + r'\s*=\s*).*$',
                                         lambda m: m[1] + literal, source_text)
            if count != 1:
                raise ValueError('one original namelist option required: ' + key)
            overlays[key] = value
    (deck / 'namelist.atmosphere').write_text(source_text)
    (deck / 'streams.atmosphere').write_text(text)
    target_lists = deck / 'stream_list'
    target_lists.mkdir()
    for path in (args.workflow / 'fix/stream_list/hrrrv5').iterdir():
        if path.is_file():
            shutil.copyfile(path, target_lists / path.name)
    updated = copy.deepcopy(row)
    updated['deck'] = deck.name
    updated['resolved_settings'].update(overlays)
    updated['files'] = {str(p.relative_to(deck)): sha(p) for p in deck.rglob('*') if p.is_file()}
    original['decks'] = [updated]
    original['runtime'] = dict(original['runtime'], output_timelevels=args.output_hours,
        output_interval=args.interval, restart_interval=args.interval,
        workflow_stream_template_sha256=sha(template))
    (decks / 'INDEX.json').write_text(json.dumps(original, indent=2) + '\n')
    case = json.loads(args.case.read_text())
    old_streams = parse_streams(args.native_deck / 'streams.atmosphere')
    new_streams = parse_streams(deck / 'streams.atmosphere')
    files = {}
    for name, stream in new_streams.items():
        if 'input' not in stream.options['type'].split(';') or name in ('restart', 'iau', 'da_state'):
            continue
        old_name = 'input' if name == 'invariant' else name
        if old_name not in old_streams:
            continue
        old_filename = old_streams[old_name].filename(original_start, original_start)
        if old_filename in case['files']:
            files[stream.filename(start, start)] = case['files'][old_filename]
    if args.warm_state:
        files[new_streams['da_state'].filename(start, start)] = str(args.warm_state.resolve(strict=True))
    if args.restart_state:
        files[new_streams['restart'].filename(start, start)] = str(args.restart_state.resolve(strict=True))
    own_case = args.work_root / 'case.json'
    own_case.write_text(json.dumps(dict(files=files,
        scope='Native CPU stand-in for workflow stream and DA transport; no Hex forecast'), indent=2) + '\n')
    return deck, own_case, dict(template_sha256=sha(template),
        original_deck_index_sha256=sha(args.native_deck.parent / 'INDEX.json'),
        original_namelist_sha256=sha(args.native_deck / 'namelist.atmosphere'),
        workflow_lists_sha256={p.name: sha(p) for p in target_lists.iterdir()},
        output_timelevels=args.output_hours, output_interval=args.interval,
        scientific_overlays=overlays,
        input_mapping=files, workflow_template_edited=False)


def replay(args):
    deck, case, authority = prepare(args)
    out = args.work_root / 'forecast'
    argv = ['python', str(args.native_runner), '--build', str(args.build),
            '--deck', str(deck), '--case', str(case), '--out', str(out),
            '--mpi', str(args.mpi), '--ranks', '1']
    for path in args.tables:
        argv += ['--tables', str(path)]
    with (args.work_root / 'controller.log').open('w') as log:
        status = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT).returncode
    manifest = json.loads((out / 'MANIFEST.json').read_text()) if (out / 'MANIFEST.json').exists() else None
    receipt = dict(schema='rrfs-workflow-native-stream-attempt/v1',
        authority=authority, run_host=os.uname().nodename, process_count=1,
        argv=argv, exit_status=status, native_manifest=manifest,
        no_initial_mpasout_publication=(not any(n.startswith('mpasout.') and
            n.endswith(args.warm_state.name.removeprefix('mpasout.'))
            for n in manifest['outputs']) if args.warm_state and manifest else None),
        scope='Strict native CPU stand-in; complete workflow stream/DA feature attempt',
        whole_model_proof=False)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    return 0 if status == 0 and manifest and manifest['completed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('native-runner', 'native-deck', 'workflow', 'case', 'build', 'mpi', 'work-root', 'receipt'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--tables', type=Path, action='append', required=True)
    parser.add_argument('--warm-state', type=Path)
    parser.add_argument('--restart-state', type=Path)
    parser.add_argument('--start-time')
    parser.add_argument('--duration')
    parser.add_argument('--jedi-da', choices=('true', 'false'), default='true')
    for key in ('output-hours', 'interval', 'lbc-hours'):
        parser.add_argument('--' + key, required=True)
    raise SystemExit(replay(parser.parse_args()))
