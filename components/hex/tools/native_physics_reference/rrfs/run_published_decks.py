#!/usr/bin/env python3
"""Run whole native deck groups from authenticated published initializer outputs.

All meshes, cases, clocks, schemes, soil companions and recipes are request
data. The existing native runner computes fields and seals every completed
forecast; this launcher transfers and verifies original bytes only.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

spec = importlib.util.spec_from_file_location('rrfs_published_replay',
                                            Path(__file__).with_name('replay_io.py'))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def authenticate(directory, pin, names=None):
    sha = replay.load('run_forecast').sha
    if sha(directory / 'PUBLICATION.json') != pin:
        raise ValueError('native publication authority changed')
    publication = json.loads((directory / 'PUBLICATION.json').read_text())
    selected = list(publication['files']) if names is None else names
    for name in selected:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or name not in publication['files']:
            raise ValueError('unsafe or unsealed native publication member')
        if sha(directory / name) != publication['files'][name]:
            raise ValueError('published native member bytes changed')
    return {name: publication['files'][name] for name in selected}


def acquire(host, source, directory, pin=None, names=None):
    sha = replay.load('run_forecast').sha
    directory.mkdir(parents=True, exist_ok=False)
    subprocess.run(['rsync', '-a', host + ':' + source + '/PUBLICATION.json',
                    str(directory) + '/'], check=True)
    digest = sha(directory / 'PUBLICATION.json')
    if pin is not None and digest != pin:
        raise ValueError('native publication authority changed')
    publication = json.loads((directory / 'PUBLICATION.json').read_text())
    selected = list(publication['files']) if names is None else names
    for name in selected:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or name not in publication['files']:
            raise ValueError('unsafe or unsealed native publication member')
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync', '-a', '--sparse', host + ':' + source + '/' + name, str(target)], check=True)
    files = authenticate(directory, digest, selected)
    return dict(path=source, publication_sha256=digest, selected_files=files)


def expand_replays(rows, phases):
    """Bind explicit repeat/rank/restart phases to each authenticated deck.

    Physics options and clocks remain request data. Dependencies always refer
    to an earlier phase of the same deck and recipe, never another forecast.
    Additional partition inputs require their own byte authority.
    """
    if not phases:
        raise ValueError('explicit native replay phases required')
    sha = replay.load('run_forecast').sha
    seen = set()
    for phase in phases:
        name = phase['name']
        if (not name or Path(name).name != name or name in ('.', '..') or name in seen
                or type(phase['ranks']) is not int or phase['ranks'] < 1):
            raise ValueError('unique simple replay names and positive ranks required')
        dependencies = set(phase.get('from_outputs', {})) | set(phase.get('compare', {}))
        if not dependencies <= seen:
            raise ValueError('replay dependency must name an earlier same-deck phase')
        if any(mode not in ('rerun', 'ranks') for mode in phase.get('compare', {}).values()):
            raise ValueError('native replay comparison mode is unsupported')
        for target, authority in phase.get('input_files', {}).items():
            if (not target or Path(target).name != target or target in ('.', '..')
                    or sha(Path(authority['path'])) != authority['sha256']):
                raise ValueError('native replay partition input authority changed')
        seen.add(name)
    expanded = []
    for row in rows:
        for phase in phases:
            selected = copy.deepcopy(row)
            selected['name'] = row['name'] + '-' + phase['name']
            selected['ranks'] = phase['ranks']
            for group, overrides in phase.get('overlay', {}).items():
                if group not in ('settings', 'stream_variables', 'streams', 'stream_list_directory'):
                    raise ValueError('unknown native replay configuration overlay')
                if group == 'stream_list_directory':
                    selected['overlay'][group] = overrides
                elif group == 'streams':
                    for stream, values in overrides.items():
                        selected['overlay'].setdefault(group, {}).setdefault(stream, {}).update(values)
                else:
                    selected['overlay'].setdefault(group, {}).update(overrides)
            for target, authority in phase.get('input_files', {}).items():
                if target in selected['inputs']:
                    raise ValueError('replay extra input cannot replace initialized weather')
                selected['inputs'][target] = authority['path']
            selected['from_outputs'] = {row['name'] + '-' + name: dict(files)
                                        for name, files in phase.get('from_outputs', {}).items()}
            selected['compare'] = {row['name'] + '-' + name: mode
                                  for name, mode in phase.get('compare', {}).items()}
            expanded.append(selected)
    return expanded


def run(request, out):
    matrix = replay.load('replay_matrix')
    sha = replay.load('run_forecast').sha
    root = Path(request['work_root']).resolve(strict=True)
    if out.exists() or not out.resolve().is_relative_to(root):
        raise ValueError('published deck run requires new owned scratch')
    if any(sha(Path(path)) != digest for path, digest in request['input_hashes'].items()):
        raise ValueError('published deck request authority changed')
    names = [group['name'] for group in request['groups']]
    if not names or len(names) != len(set(names)) or any(
            not name or Path(name).name != name or name in ('.', '..') for name in names):
        raise ValueError('explicit unique simple native group identifiers required')
    out.mkdir(parents=True)
    (out / 'REQUEST.json').write_text(json.dumps(request, indent=2) + '\n')
    host = request['archive']['host']
    results = []

    def record():
        summary = dict(schema='mpas.rrfs-published-native-deck-groups/v1', scope=request['scope'],
                       requested=len(names), results=results,
                       all_attempts_finished=len(results) == len(names) and all(
                           result['all_attempts_finished'] for result in results),
                       all_native_completed=len(results) == len(names) and all(
                           result['all_native_completed'] for result in results))
        (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary

    record()
    tables = {}
    table_authorities = {}
    for recipe, table in request['table_publications'].items():
        directory = Path(table['local'])
        table_authorities[recipe] = dict(path=table['path'],
            publication_sha256=table['publication_sha256'],
            selected_files=authenticate(directory, table['publication_sha256']))
        tables[recipe] = [*request['base_tables'], str(directory)]
    for group in request['groups']:
        scratch = out / (group['name'] + '-inputs')
        scratch.mkdir()
        final_publication = group['cases'][-1]['path'] + '/PUBLICATION.json'
        while True:
            status = subprocess.run(['ssh', '-o', 'BatchMode=yes', host,
                                     shlex.join(['test', '-f', final_publication])]).returncode
            if status == 0:
                break
            if status != 1:
                raise RuntimeError('native archive availability check failed')
            time.sleep(30)
        inputs, authorities = {}, {}
        for position, case in enumerate(group['cases']):
            directory = scratch / ('case-' + str(position))
            selected = ['MANIFEST.json', *case['files'].values()]
            authorities[case['path']] = acquire(host, case['path'], directory,
                                                case.get('publication_sha256'), selected)
            manifest = json.loads((directory / 'MANIFEST.json').read_text())
            if not manifest['completed'] or manifest['exit_status'] != 0 or (
                    manifest['binary_sha256'] != request['init_binary_sha256']):
                raise ValueError('published native initializer did not complete with the pinned binary')
            if any(manifest['row'].get(key) != value for key, value in case['row_authority'].items()):
                raise ValueError('native initializer scientific request changed')
            for target, name in case['files'].items():
                if target in inputs or Path(target).name != target or target in ('.', '..'):
                    raise ValueError('duplicate or unsafe native input name')
                if manifest['outputs'][name] != sha(directory / name):
                    raise ValueError('native initializer output differs from its manifest')
                inputs[target] = str(directory / name)
        for position, companion in enumerate(group.get('ancillary_publications', [])):
            directory = scratch / ('ancillary-' + str(position))
            selected = list(companion['files'].values())
            authorities[companion['path']] = acquire(host, companion['path'], directory,
                companion['publication_sha256'], selected)
            for target, name in companion['files'].items():
                if target in inputs or Path(target).name != target or target in ('.', '..'):
                    raise ValueError('duplicate or unsafe published companion name')
                inputs[target] = str(directory / name)
        rows = []
        for position, deck_group in enumerate(group['deck_publications']):
            directory = scratch / ('decks-' + str(position))
            selected = ['INDEX.json', *deck_group['files']]
            authorities[deck_group['path']] = acquire(host, deck_group['path'], directory,
                                                     deck_group['publication_sha256'], selected)
            index = json.loads((directory / 'INDEX.json').read_text())
            for name in deck_group['selected_names']:
                row = next(row for row in index['decks'] if row['deck'] == name)
                if row['resolved_settings']['num_soil_layers'] != group['soil_levels']:
                    raise ValueError('native deck and initializer soil companions differ')
                for recipe in group['recipes']:
                    label = group['name'] + '-' + name + '-' + recipe
                    native_row = dict(name=label, mesh=group['mesh'], date=group['date'],
                        source_deck=str(directory / name), source_index_sha256=sha(directory / 'INDEX.json'),
                        overlay=copy.deepcopy(group['overlay']), inputs=inputs, ranks=group['ranks'],
                        build=request['builds'][recipe],
                        archive_build=request.get('archive_builds', request['builds'])[recipe],
                        tables=tables[recipe])
                    if group.get('readonly_inputs'):
                        native_row['readonly_inputs'] = list(group['readonly_inputs'])
                    if group.get('output_archive'):
                        native_row['output_archive'] = dict(group['output_archive'],
                            host=host,stage=str(Path(request['archive']['staging'])/label),
                            comparator=request['comparator'],work_root=str(root),
                            limit_bytes=request['limit_bytes'])
                    rows.append(native_row)
        if 'replays' in group:
            rows = expand_replays(rows, group['replays'])
        hashes = {path: sha(Path(path)) for row in rows for path in row['inputs'].values()}
        hashes.update({str(path): sha(path) for directories in tables.values() for directory in directories
                       for path in Path(directory).iterdir() if path.is_file()})
        configured = {key: request[key] for key in ('scope', 'workflow', 'source', 'mpi', 'archive',
                      'work_root', 'limit_bytes', 'reserve_bytes', 'comparator')}
        configured.update(runs=rows, input_hashes=hashes, input_publications=authorities,
                          table_authorities=table_authorities)
        result = matrix.matrix(configured, out / group['name'])
        results.append(dict(name=group['name'], summary_sha256=sha(out / group['name'] / 'SUMMARY.json'),
                            all_attempts_finished=result['all_attempts_finished'],
                            all_native_completed=result['all_native_completed']))
        record()
        # Published case/deck members remain immutable on the archive host.
        # Retire only transferred NC inputs after every native row has finished.
        for source in inputs.values():
            path = Path(source)
            if sha(path) != hashes[source] or path.is_symlink():
                raise ValueError('native input scratch changed before retirement')
        for source in inputs.values():
            Path(source).unlink()
        print(json.dumps(results[-1]), flush=True)
    return record()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.request.read_text()), args.out)
    raise SystemExit(0 if result['all_native_completed'] else 1)
