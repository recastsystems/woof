#!/usr/bin/env python3
"""Execute generated initialization rows; keep each real failure as evidence.

Weather decoding and interpolation are exclusively native ungrib/init. A
completed decode can be reused only after its input and output hashes match.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rrfs_matrix_cache', HERE / 'cache_cases.py')
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


def memory_ready():
    while True:
        available = next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        if int(available) * 1024 >= 16 * 1024**3:
            return
        time.sleep(15)


def verified_decode(row, directory, ungrib, vtable):
    manifest = directory / 'MANIFEST.json'
    if not manifest.is_file():
        memory_ready()
        return cache.decode(row, directory, ungrib, vtable)
    receipt = json.loads(manifest.read_text())
    expected = {Path(p).name: cache.sha(Path(p)) for parts in row['gribs'] for p in parts}
    previous = {Path(p).name: digest for p, digest in receipt['inputs'].items()}
    if len(previous) != len(receipt['inputs']) or len(expected) != sum(map(len, row['gribs'])):
        raise ValueError('ambiguous decode input basenames')
    if (previous != expected or receipt['ungrib_sha256'] != cache.sha(ungrib)
            or receipt['vtable_sha256'] != cache.sha(vtable)
            or any(receipt['row'][key] != row[key] for key in ('start', 'stop', 'interval_seconds'))):
        raise ValueError('completed decode differs from requested inputs/toolchain/clock')
    if not receipt['intermediates'] or any(cache.sha(directory / name) != digest
                                         for name, digest in receipt['intermediates'].items()):
        raise ValueError('completed native intermediate bytes changed')
    return directory


def run(index, work, workflow, build, mpi, ungrib, vtable, ready, receipt, decode_only=False):
    results = []
    stages = [('decode', row) for row in index.get('decode', [])]
    if not decode_only:
        stages += [('case', row) for row in index.get('cases', [])]
    for kind, row in stages:
        directory = work / ('decoded' if kind == 'decode' else '') / row['name']
        result = dict(kind=kind, name=row['name'], path=str(directory), completed=False)
        try:
            if kind == 'decode':
                # Inputs are downloaded atomically to their final filenames.
                while any(not Path(p).is_file() for parts in row['gribs'] for p in parts):
                    time.sleep(15)
                if index.get('input_verification_directory'):
                    verification = Path(index['input_verification_directory'])
                    for parts in row['gribs']:
                        for source in parts:
                            marker = verification / (Path(source).name + '.VERIFIED.json')
                            while not marker.is_file():
                                time.sleep(15)
                            proven = json.loads(marker.read_text())
                            if not proven['verified'] or proven['sha256'] != cache.sha(Path(source)):
                                raise ValueError('public input differs from its S3 checksum witness')
                verified_decode(row, directory, ungrib, vtable)
            else:
                memory_ready()
                cache.verified_case(row, directory, workflow, build, mpi)
                if 'publish' in row:
                    result['publication'] = cache.publish_case(directory, Path(row['publish']), ready, row['description'])
            result['completed'] = True
            result['manifest_sha256'] = cache.sha(directory / 'MANIFEST.json')
        except Exception as error:
            result['error'] = type(error).__name__ + ': ' + str(error)
            if (directory / 'MANIFEST.json').is_file():
                result['manifest_sha256'] = cache.sha(directory / 'MANIFEST.json')
        results.append(result)
        receipt.write_text(json.dumps(dict(schema='mpas.rrfs-native-case-matrix-run/v1',
            attempted=len(results), requested=len(stages),
            completed=len(results) == len(stages) and all(r['completed'] for r in results),
            rows=results), indent=2) + '\n')
        print(json.dumps(result), flush=True)
    return all(row['completed'] for row in results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'work', 'workflow', 'ungrib', 'vtable', 'ready', 'receipt'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--mpi', type=Path)
    parser.add_argument('--decode-only', action='store_true')
    args = parser.parse_args()
    if not args.decode_only and (not args.build or not args.mpi):
        parser.error('native case stages require --build and --mpi')
    ok = run(json.loads(args.index.read_text()), args.work.resolve(), args.workflow.resolve(),
             args.build.resolve() if args.build else None, args.mpi.resolve() if args.mpi else None,
             args.ungrib.resolve(), args.vtable.resolve(), args.ready.resolve(), args.receipt.resolve(),
             args.decode_only)
    sys.exit(0 if ok else 1)
