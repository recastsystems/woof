#!/usr/bin/env python3
"""Run explicit whole native forecasts, comparisons and restart dependencies.

Uses the existing fork runner, indexed deck generator, archive receiver and
Rust comparator. Every attempt, including native failures and word differences,
is recorded. All scheme, mesh, clock, rank and scientific choices are request
data. Completed outputs are published before SHA-guarded scratch retirement.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

spec = importlib.util.spec_from_file_location('rrfs_matrix_replay_io',
                                            Path(__file__).with_name('replay_io.py'))
replay_io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay_io)
load = replay_io.load


def matrix(request, out):
    runner, overlay, identity = [load(name) for name in ('run_forecast', 'workflow_deck', 'identity')]
    sha = runner.sha
    root = Path(request['work_root']).resolve(strict=True)
    if out.exists() or not out.resolve().is_relative_to(root):
        raise ValueError('forecast matrix requires new owned output scratch')
    rows = request['runs']
    names = [row['name'] for row in rows]
    if not rows or len(names) != len(set(names)) or any(
            not n or n in ('.', '..') or Path(n).name != n for n in names):
        raise ValueError('explicit unique simple forecast identifiers required')
    for row in rows:
        for dependency in [*row.get('from_outputs', {}), *row.get('compare', {})]:
            if dependency not in names[:names.index(row['name'])]:
                raise ValueError('forecast dependency must precede its consumer')
    if any(sha(Path(p)) != digest for p, digest in request['input_hashes'].items()):
        raise ValueError('native forecast matrix input authority changed')
    mpi = Path(request['mpi'])
    os.environ['LD_LIBRARY_PATH'] = str(mpi.parent.parent / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    out.mkdir(parents=True)
    (out / 'REQUEST.json').write_text(json.dumps(request, indent=2) + '\n')
    results, runs, retired = [], {}, []

    def record():
        summary = dict(schema='mpas.rrfs-native-whole-forecast-matrix/v1', scope=request['scope'],
                       requested=len(rows), results=results, retired=retired,
                       all_attempts_finished=len(results) == len(rows),
                       all_native_completed=len(results) == len(rows) and all(
                           r.get('completed') and r.get('publication') and not r.get('error') for r in results))
        (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary

    def publish(run, row, manifest):
        archive = request['archive']
        selected = {'MANIFEST.json', *manifest['outputs'], *manifest['resolved_configuration']['files']}
        listing = run.parent / 'transfer-files.txt'
        stage = Path(archive['staging']) / row['name']
        archived = manifest['case'].get('output_archive')
        already = set()
        if archived:
            witness = json.loads((run/'ARCHIVED-OUTPUTS.json').read_text())
            already = set(witness['files'])
            if Path(archived['stage']) != stage:
                raise ValueError('closed-byte archive and final publication staging differ')
        listing.write_text('\n'.join(sorted(selected - already)) + '\n')
        subprocess.run(['rsync', '-a', '--mkpath', '--files-from=' + str(listing), str(run) + '/',
                        archive['host'] + ':' + str(stage) + '/'], check=True)
        expected = dict(manifest_sha256=sha(run / 'MANIFEST.json'), mesh=row['mesh'],
                        date=row['date'], label=row['name'])
        argv = ['python', archive['receiver'], '--source', str(stage), '--reference', archive['reference'],
                '--build', row.get('archive_build', row['build']), '--retire-output-root', archive['work_root']]
        command = ('export TMPDIR=' + shlex.quote(archive['tmpdir']) + '; . ' +
                   shlex.quote(archive['environment']) + '; taskset -c ' + shlex.quote(archive['cpuset']) +
                   ' ' + shlex.join(argv))
        response = subprocess.run(['ssh', '-o', 'BatchMode=yes', archive['host'], command],
                                  input=json.dumps(expected), capture_output=True, text=True, check=True)
        publication = json.loads(response.stdout)
        (run.parent / 'PUBLICATION.json').write_text(json.dumps(publication, indent=2) + '\n')
        return publication

    for position, row in enumerate(rows):
        result = dict(name=row['name'], completed=False)
        results.append(result)
        record()
        work = out / row['name']
        try:
            if any(not runs.get(dependency, {}).get('completed')
                   for dependency in row.get('from_outputs', {})):
                raise ValueError('native restart dependency did not complete; no replacement input invented')
            while int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                           if line.startswith('MemAvailable:'))) * 1024 < 16 * 1024 ** 3:
                time.sleep(15)
            source_deck = Path(row['source_deck'])
            if sha(source_deck.parent / 'INDEX.json') != row['source_index_sha256']:
                raise ValueError('native deck index changed')
            deck = overlay.generate(source_deck, Path(request['workflow']), Path(request['source']),
                                    work / 'decks', row['overlay'])
            files = dict(row['inputs'])
            for dependency, mapping in row.get('from_outputs', {}).items():
                prior = runs[dependency]
                for target, name in mapping.items():
                    path = prior['path'] / name
                    if sha(path) != prior['manifest']['outputs'][name]:
                        raise ValueError('native restart output authority changed')
                    files[target] = str(path)
            usage = int(subprocess.check_output(['du', '-s', '-B1', str(root)], text=True).split()[0])
            copied = {n:p for n,p in files.items() if n not in row.get('readonly_inputs', [])}
            if usage + sum(Path(p).stat().st_size for p in copied.values()) + request['reserve_bytes'] > request['limit_bytes']:
                raise ValueError('whole native forecast exceeds owned scratch reservation')
            run = work / 'forecast'
            case = dict(files=files, scope=request['scope'])
            if row.get('readonly_inputs'):
                case['readonly_inputs'] = row['readonly_inputs']
            if row.get('output_archive'):
                case['output_archive'] = row['output_archive']
            manifest = runner.forecast(Path(row['build']), deck, case,
                                       [Path(p) for p in row['tables']], run, mpi, row['ranks'])
            result.update(completed=manifest['completed'], exit_status=manifest['exit_status'],
                          manifest_sha256=sha(run / 'MANIFEST.json'))
            runs[row['name']] = dict(path=run, manifest=manifest, completed=manifest['completed'])
            if manifest['completed']:
                result['publication'] = publish(run, row, manifest)
            result['comparisons'] = {}
            for reference, mode in row.get('compare', {}).items():
                if not manifest['completed'] or not runs.get(reference, {}).get('completed'):
                    result['comparisons'][reference] = dict(not_run='native forecast incomplete')
                    continue
                measured = identity.witness(runs[reference]['path'], run, Path(request['comparator']), mode)
                path = work / (reference + '-' + mode + '.json')
                path.write_text(json.dumps(measured, indent=2) + '\n')
                result['comparisons'][reference] = dict(sha256=sha(path),
                    all_identical=measured['all_identical'], all_file_bytes_identical=measured['all_file_bytes_identical'],
                    first_difference=measured['first_difference'])
        except Exception as exc:
            result['error'] = type(exc).__name__ + ': ' + str(exc)
        record()
        # Every future consumer, whether restart or comparison, keeps its exact
        # source output alive. Authenticate all private members before deleting.
        future = {d for later in rows[position + 1:]
                  for d in [*later.get('from_outputs', {}), *later.get('compare', {})]}
        for previous in results:
            name = previous['name']
            if name in future or previous.get('retired') or not previous.get('publication'):
                continue
            native = runs[name]
            manifest, run = native['manifest'], native['path']
            candidates = [p for p in run.glob('*.nc') if not p.is_symlink()]
            checked = []
            for p in candidates:
                digest = manifest['outputs'].get(p.name) or manifest['inputs'].get(p.name)
                if not digest or sha(p) != digest:
                    raise ValueError('native matrix scratch changed before retirement')
                checked.append(dict(file=p.name, sha256=digest, bytes=p.stat().st_size))
            for p in candidates:
                p.unlink()
            previous['retired'] = True
            retired.append(dict(name=name, publication=previous['publication'], files=checked))
        record()
        print(json.dumps(result), flush=True)
    return record()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = matrix(json.loads(args.request.read_text()), args.out)
    raise SystemExit(0 if result['all_native_completed'] else 1)
