#!/usr/bin/env python3
"""Replay the explicitly requested, unmodified upstream create_region oracle.

The upstream tool computes the mesh subset. This wrapper only authenticates
inputs, launches that tool, and records original output hashes and headers.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publish_cache import sha


def replay(request, out):
    source = Path(request['source']).resolve(strict=True)
    entry = source / request['entry']
    inputs = {name:Path(row['path']).resolve(strict=True) for name,row in request['inputs'].items()}
    for name, path in inputs.items():
        if sha(path) != request['inputs'][name]['sha256']:
            raise ValueError('native regional mesh input changed: ' + name)
    for name, digest in request['source_files'].items():
        path = source / name
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(path) != digest:
            raise ValueError('upstream create_region source changed')
    if request['entry'] not in request['source_files'] or not request['expected_outputs']:
        raise ValueError('native region entry and outputs must be explicitly pinned')
    if any(Path(name).name != name for name in request['expected_outputs']):
        raise ValueError('native region outputs must be simple filenames')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'REQUEST.json').write_text(json.dumps(request, indent=2) + '\n')
    results = []
    for number in (1, 2):
        directory = out / ('run-' + str(number)); directory.mkdir()
        argv = [sys.executable, str(entry), str(inputs['points']), str(inputs['grid']), '-v', '3']
        with (directory / 'native.stdout.log').open('w') as log:
            status = subprocess.run(argv, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1')).returncode
        outputs = {name:sha(directory/name) for name in request['expected_outputs']
                   if (directory/name).is_file()}
        completed = status == 0 and set(outputs) == set(request['expected_outputs'])
        row = dict(exit_status=status, completed=completed, outputs=outputs, argv=argv,
                   host=os.uname().nodename, cpu_affinity=sorted(os.sched_getaffinity(0)))
        results.append(row)
        (directory / 'MANIFEST.json').write_text(json.dumps(row, indent=2) + '\n')
        if not completed:
            break
        for name in request['expected_outputs']:
            if name.endswith('.nc'):
                with (directory / (name + '-header.txt')).open('w') as header:
                    subprocess.run(['ncdump', '-h', str(directory/name)], stdout=header, check=True)
    for name, path in inputs.items():
        if sha(path) != request['inputs'][name]['sha256']:
            raise ValueError('native region run modified an input')
    complete = len(results) == 2 and all(row['completed'] for row in results)
    summary = dict(schema='mpas.rrfs-native-create-region-repeat/v1',
                   source_commit=request['source_commit'], source_files=request['source_files'],
                   inputs=request['inputs'], results=results, completed=complete,
                   all_output_bytes_identical=complete and results[0]['outputs'] == results[1]['outputs'])
    (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = replay(json.loads(args.request.read_text()), args.out.resolve())
    print(json.dumps(result))
    raise SystemExit(0 if result['completed'] and result['all_output_bytes_identical'] else 1)
