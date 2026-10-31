#!/usr/bin/env python3
"""Run all ready native forecast rows, publishing every successful whole run.

Explicit request rows bind builds, decks, cases, tables and restart parents.
Missing products never block unrelated ready rows. Failed attempts remain
receipts and are not retried under the same directory or published as success.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location('rrfs_matrix_' + name, HERE / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


native = module('run_forecast')
publication = module('publish_forecast')
cases = module('run_case_matrix')


class PendingInputs(Exception):
    pass


def inputs(row, results):
    files = {}
    for name, source in row['files'].items():
        if isinstance(source, dict):
            parent = results.get(source['run'], {})
            if parent.get('state') != 'completed':
                raise PendingInputs('restart parent has not completed: ' + source['run'])
            source = str(Path(parent['publication']['path']) / source['file'])
        if not Path(source).is_file():
            raise PendingInputs('case input absent: ' + str(source))
        files[name] = source
    return dict(files=files, source_case=row.get('source_case'), authority=row.get('authority'))


def verify_publication(directory):
    path = directory / 'PUBLICATION.json'
    if not path.is_file():
        raise PendingInputs('publication absent: ' + str(directory))
    receipt = json.loads(path.read_text())
    for name, digest in receipt['files'].items():
        if native.sha(directory / name) != digest:
            raise ValueError('immutable dependency changed: ' + str(directory / name))
    return native.sha(path)


def publication_root(directory):
    """A table subdirectory belongs to its immutable enclosing cache."""
    for candidate in (directory, *directory.parents):
        if (candidate / 'PUBLICATION.json').is_file():
            return candidate
    raise PendingInputs('publication absent: ' + str(directory))


def execute(request, work, reference, receipt, once=False):
    rows = request['rows']
    if not rows or len({r['name'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique forecast rows required')
    for row in rows:
        if Path(row['name']).name != row['name'] or row['name'] in ('.', '..'):
            raise ValueError('simple unique run names required')
    results = {}
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous['request'] != request:
            raise ValueError('forecast matrix receipt belongs to a different request')
        results = previous['results']
    checked = {}
    while True:
        attempted = False
        for row in rows:
            if results.get(row['name'], {}).get('state') in ('completed', 'failed'):
                continue
            result = dict(state='pending')
            directory = work / row['name']
            try:
                build_receipt = Path(row['build_receipt'])
                if not build_receipt.is_file():
                    raise PendingInputs('native build receipt absent: ' + str(build_receipt))
                build = Path(json.loads(build_receipt.read_text())['path'])
                dependencies = [build, Path(row['deck']).parent, *map(publication_root, map(Path, row['tables'])),
                                *map(Path, row.get('case_publications', []))]
                for dependency in dependencies:
                    if dependency not in checked:
                        checked[dependency] = verify_publication(dependency)
                case = inputs(row, results)
                if directory.exists():
                    raise ValueError('attempt directory exists without a terminal matrix receipt; inspect it before a new request')
                cases.memory_ready()
                attempted = True
                manifest = native.forecast(build, Path(row['deck']), case, list(map(Path, row['tables'])),
                                           directory, Path(row['mpi']), row['ranks'])
                result.update(state='completed' if manifest['completed'] else 'failed',
                              exit_status=manifest['exit_status'], manifest=str(directory / 'MANIFEST.json'),
                              manifest_sha256=native.sha(directory / 'MANIFEST.json'),
                              dependency_publications={str(p):checked[p] for p in dependencies})
                if manifest['completed']:
                    result['publication'] = publication.publish_forecast(directory, reference, row['mesh'],
                                                                         row['date'], row['label'])
            except PendingInputs as error:
                result['reason'] = str(error)
            except Exception as error:
                result.update(state='failed', error=type(error).__name__ + ': ' + str(error))
            results[row['name']] = result
            receipt.write_text(json.dumps(dict(schema='mpas.rrfs-native-forecast-matrix/v1', request=request,
                completed=all(results.get(r['name'], {}).get('state') == 'completed' for r in rows),
                results=results), indent=2) + '\n')
            if result['state'] != 'pending':
                print(json.dumps(dict(name=row['name'], **result)), flush=True)
        pending = any(result['state'] == 'pending' for result in results.values())
        if once or not pending:
            return all(result['state'] == 'completed' for result in results.values())
        if not attempted:
            time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('request', 'work', 'reference', 'receipt'):
        parser.add_argument('--' + key, required=True, type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    ok = execute(json.loads(args.request.read_text()), args.work.resolve(), args.reference.resolve(),
                 args.receipt.resolve(), args.once)
    sys.exit(0 if ok else 1)
