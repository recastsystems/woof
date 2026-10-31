#!/usr/bin/env python3
"""Generate and publish every explicitly requested mesh/date deck group."""
import argparse
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rrfs_deck_matrix', HERE / 'generate_decks.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)
sys.path.insert(0, str(HERE.parent))
from publish_cache import publish, sha


def matrix(source, workflow, request, work, reference, receipt):
    specifications = json.loads((HERE / 'deck-specs.json').read_text())
    groups, entries = [], []
    for row in request['groups']:
        settings = copy.deepcopy(specifications)
        if 'selected_names' in row:
            settings['selected_names'] = row['selected_names']
        destination = reference / 'decks' / row['name']
        if row.get('existing'):
            if Path(row['existing']).resolve() != destination.resolve():
                raise ValueError('existing group does not match its publication path')
            publication = json.loads((destination / 'PUBLICATION.json').read_text())
            for name, digest in publication['files'].items():
                if sha(destination / name) != digest:
                    raise ValueError('existing published deck bytes changed')
            index = json.loads((destination / 'INDEX.json').read_text())
            if index['runtime'] != row['runtime']:
                raise ValueError('existing group differs from requested runtime')
            product = dict(path=str(destination), publication_sha256=sha(destination / 'PUBLICATION.json'))
        else:
            directory = work / row['name']
            index = generator.generate(source, workflow, directory, row['runtime'], settings)
            product = publish(directory, destination, ['INDEX.json', '**/*'], reference / 'READY.md',
                f'{len(index["decks"])} generated primary RRFS decks; mesh/date/runtime and companions in INDEX.json; '
                'run rrfs/run_forecast.py with matching strict-init cases and audited binary; forecast identity pending')
        groups.append(dict(name=row['name'], runtime=index['runtime'], publication=product,
                           index_sha256=sha(destination / 'INDEX.json')))
        for deck in index['decks']:
            entries.append(dict(group=row['name'], path=str(destination / deck['deck']), **deck))
        receipt.write_text(json.dumps(dict(completed=False, groups=groups), indent=2) + '\n')
    result = dict(schema='mpas.rrfs-native-deck-matrix/v1', request=request, groups=groups, decks=entries,
                  forecast_identity='pending; deck generation is not a forecast receipt')
    index_name = request.get('index_name', 'INDEX.json')
    if Path(index_name).name != index_name or index_name in ('.', '..'):
        raise ValueError('simple deck index filename required')
    target = reference / 'decks' / index_name
    if target.exists():
        raise ValueError('refusing to replace published deck matrix')
    partial = target.with_name('INDEX.json.partial')
    partial.write_text(json.dumps(result, indent=2) + '\n')
    partial.rename(target)
    target.chmod(0o444)
    with (reference / 'READY.md').open('a') as ready:
        fcntl.flock(ready, fcntl.LOCK_EX)
        ready.write(f'Primary generated deck matrix: `{target}`; sha256 `{sha(target)}`; '
                    'each group names immutable deck paths, hashes, runtime and forced companions; '
                    'use rrfs/run_forecast.py with audited binary and matching strict-init inputs.\n')
    receipt.write_text(json.dumps(dict(completed=True, index=str(target), sha256=sha(target),
        groups=groups, decks=len(entries)), indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'workflow', 'request', 'work', 'reference', 'receipt'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    result = matrix(args.source.resolve(), args.workflow.resolve(), json.loads(args.request.read_text()),
                    args.work.resolve(), args.reference.resolve(), args.receipt.resolve())
    print(json.dumps(dict(groups=len(result['groups']), decks=len(result['decks']))))
