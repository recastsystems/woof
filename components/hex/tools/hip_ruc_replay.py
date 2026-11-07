#!/usr/bin/env python3
"""Replay captured resident RUC land arguments without changing physics.

The capture contains manifest-checked .npy files, including optional-presence
flags. Run each backend in a fresh process. The reference is the unchanged
CUDA launch's output; this tool compares words, including NaN payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def load_manifest(folder):
    rows = json.loads((folder / 'manifest.json').read_text())
    result = {}
    for name, row in rows.items():
        path = folder / row['file']
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError(f'capture hash or size differs: {path}')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if list(array.shape) != row['shape'] or str(array.dtype) != row['dtype']:
            raise ValueError(f'capture array metadata differs: {path}')
        result[name] = array
    return result


def attributes(kernel):
    from cupy.cuda import driver
    return {name.removeprefix('CU_FUNC_ATTRIBUTE_').lower():
            driver.funcGetAttribute(getattr(driver, name), kernel.ptr)
            for name in ('CU_FUNC_ATTRIBUTE_NUM_REGS', 'CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES',
                         'CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES', 'CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK')}


def save_manifest(folder, arrays):
    folder.mkdir(parents=True, exist_ok=False)
    rows = {}
    for name, array in arrays.items():
        path = folder / (name + '.npy')
        np.save(path, array.get())
        rows[name] = dict(file=path.name, shape=list(array.shape), dtype=str(array.dtype),
                          bytes=path.stat().st_size, sha256=digest(path))
    (folder / 'manifest.json').write_text(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=('candidate', 'legacy', 'tiled-noinline'), default='candidate')
    parser.add_argument('--reserve-bytes', type=int, default=0,
                        help='hold the same pressure allocation in legacy and candidate experiments')
    parser.add_argument('--write-reference', action='store_true',
                        help='create the missing reference using the unchanged legacy CUDA source')
    parser.add_argument('--repeat', type=int, default=1,
                        help='reset inputs outside each timed launch; report every timing and median')
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    if args.write_reference and (args.variant != 'legacy' or os.environ.get('WOOF_BACKEND') != 'cuda'):
        parser.error('reference creation requires --variant legacy and WOOF_BACKEND=cuda')

    import woof
    import cupy as cp
    from woof.hip_backend import install, device_info
    install()
    from woof.hex.mpasphys.rrfs import ruc_lsm_8c9f1a8 as ruc
    from woof.hex.mpasphys.rrfs.resident_ruc_rrfs import _Boundary

    if args.variant == 'legacy':
        ruc.CUDA_SOURCE = (args.capture / 'source-ruc.cu').read_text()
        ruc.workspace_columns = lambda entry, n: n
    elif args.variant == 'tiled-noinline':
        ruc.CUDA_SOURCE = ruc.CUDA_SOURCE.replace(
            '#define RUCDEV __device__ __forceinline__', '#define RUCDEV __device__')
    launch = json.loads((args.capture / 'launch.json').read_text())
    inputs = load_manifest(args.capture / 'inputs')
    reference = None if args.write_reference else load_manifest(args.capture / 'reference')
    entry = launch['entry']
    layout = ruc.ARGUMENTS[entry]
    device = {field['name']: cp.asarray(inputs[field['name']]) for field in layout}
    optional = [field for field in layout if field['optional']]
    present = {}
    for index, field in enumerate(optional):
        flags = inputs[f'presence-{index}']
        if not np.all(flags == flags[0]):
            raise ValueError('resident presence flags must be uniform')
        present[field['name']] = bool(flags[0])
    scalars = {name: int(inputs[name][0]) for name in
               ('nsl', 'nsoillevels', 'nlcat', 'nscat') if name in inputs}
    tables = cp.asarray(inputs['tables'])
    errors = cp.zeros(1, cp.int32)
    bound = _Boundary(entry, device, tables, None, errors,
                      present=present, scalars=scalars)
    if args.variant == 'legacy':
        bound.hip_tiles = False
    if bound.n != launch['columns'] or bound.rows != launch['rows']:
        raise ValueError(f'replay extent differs: {bound.n}/{bound.rows}, {launch}')
    workspace = cp.empty((bound.rows, bound.work_columns), cp.float32)
    pressure = cp.empty(args.reserve_bytes, cp.uint8) if args.reserve_bytes else None
    if pressure is not None:
        pressure.fill(0)
    cp.cuda.get_current_stream().synchronize()
    free_before, total = cp.cuda.runtime.memGetInfo()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.prelaunch.json').write_text(json.dumps(dict(
        variant=args.variant, reserve_bytes=args.reserve_bytes, free_before=free_before,
        card_bytes=total, columns=bound.n, arena_bytes=workspace.nbytes,
        kernel_attributes=attributes(bound.kernel), device=device_info()), indent=2))
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    timings = []
    for index in range(args.repeat):
        if index:
            for name, array in device.items():
                array.set(inputs[name])
            errors.fill(0)
            cp.cuda.get_current_stream().synchronize()
        start = time.perf_counter()
        begin.record()
        bound(workspace)
        end.record()
        end.synchronize()
        timings.append(dict(kernel_milliseconds=cp.cuda.get_elapsed_time(begin, end),
                            launch_wall_seconds=time.perf_counter() - start))
        if args.write_reference and index == 0:
            save_manifest(args.capture / 'reference', dict(device, errors=errors))
    compared = {}
    for name, expected in (reference or {}).items():
        actual = errors.get() if name == 'errors' else device[name].get()
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f'reference shape or type differs: {name}')
        raw, wanted = actual.view(np.uint32), expected.view(np.uint32)
        different = int(np.count_nonzero(raw != wanted))
        row = dict(words=int(raw.size), different_words=different,
                   sha256=hashlib.sha256(actual.tobytes()).hexdigest())
        if different:
            offset = int(np.flatnonzero((raw != wanted).ravel())[0])
            row['first'] = dict(offset=offset, actual=f'{raw.ravel()[offset]:08x}',
                                expected=f'{wanted.ravel()[offset]:08x}')
        compared[name] = row
    result = dict(exact=all(row['different_words'] == 0 for row in compared.values()) if reference else None,
                  reference_created=args.write_reference,
                  device=device_info(), columns=bound.n, arena_rows=bound.rows,
                  arena_bytes=workspace.nbytes, free_before=free_before,
                  card_bytes=total, numerics=os.environ.get('GPUWM_NUMERICS'),
                  variant=args.variant, reserve_bytes=args.reserve_bytes,
                  source_sha256=ruc.source_digest(), kernel_attributes=attributes(bound.kernel),
                  timings=timings,
                  kernel_milliseconds=statistics.median(row['kernel_milliseconds'] for row in timings),
                  compared=compared)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'compared'}, indent=2))
    return 0 if result['exact'] or args.write_reference else 1


if __name__ == '__main__':
    raise SystemExit(main())
