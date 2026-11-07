#!/usr/bin/env python3
"""Fingerprint complete native checkpoint array bytes for an OFF/base comparison.

ZIP timestamps and the provenance header are excluded. All other NPY entries,
including continuation state and their dtype/shape headers, are hashed exactly.
This is byte verification, not a numerical reduction or forecast qualification.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

HEADER = '__gpuwm_restart_header__.npy'


def fingerprint(path, workers):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if HEADER not in names or len(names) != len(set(names)):
            raise ValueError('expected one native checkpoint header and unique entries')
        if any(not name.endswith('.npy') for name in names):
            raise ValueError('unexpected non-array checkpoint entry')
        import numpy as np
        header = json.loads(np.load(io.BytesIO(archive.read(HEADER)), allow_pickle=False).tobytes())
        if 'elapsed_seconds' not in header:
            raise ValueError('native checkpoint has no clock')
        clock_placement = {name: header.get(name) for name in ('elapsed_seconds', 'placement', 'grid_id')}
        names = sorted(name for name in names if name != HEADER)
    if not names:
        raise ValueError('empty native checkpoint')

    def digest(name):
        with zipfile.ZipFile(path) as archive, archive.open(name) as stream:
            value = hashlib.file_digest(stream, 'sha256').hexdigest()
            size = archive.getinfo(name).file_size
        return name, dict(sha256=value, bytes=size)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        arrays = dict(pool.map(digest, names))
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('checkpoint changed during fingerprinting')
    return dict(path=str(path), archive_bytes=after.st_size, clock_placement=clock_placement, arrays=arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 60:
        parser.error('workers must be 1..60')
    tick = time.perf_counter()
    rows = [fingerprint(path, args.workers) for path in args.input]
    result = dict(schema='gpuwm-native-checkpoint-fingerprints/1', checkpoints=rows,
                  scope='all native NPY array bytes; excludes provenance JSON header',
                  ordering='explicit input domain order', workers=args.workers,
                  wall_seconds=time.perf_counter()-tick)
    if args.compare:
        prior = json.loads(args.compare.read_text())
        if prior.get('schema') != result['schema']:
            raise ValueError('unsupported comparison schema')
        previous = prior['checkpoints']
        if len(rows) != len(previous):
            raise ValueError('checkpoint domain counts differ')
        mismatch = []
        for i, (a, b) in enumerate(zip(previous, rows)):
            if a.get('clock_placement') != b['clock_placement']:
                mismatch.append(dict(domain_index=i, array='clock_placement'))
            for name in sorted(set(a['arrays']) | set(b['arrays'])):
                if a['arrays'].get(name) != b['arrays'].get(name):
                    mismatch.append(dict(domain_index=i, array=name))
        result.update(bitwise_equal=not mismatch, mismatches=mismatch,
                      comparison=str(args.compare.resolve()))
    temporary = args.output.with_suffix(args.output.suffix + '.part')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(args.output)
    if result.get('bitwise_equal') is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
