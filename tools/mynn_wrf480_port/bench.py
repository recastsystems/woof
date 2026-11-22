#!/usr/bin/env python3
"""Device time of the shipped wrf_480 column launches on real native columns.

Retained spun-up native input columns (PBL calls 10, 30 and 60 by default)
are tiled to the forecast column count and run through the shipped image,
fast numerics unless --strict. Each sample restores pristine inputs first
(outside the timed region) and times only the column launches with CUDA
events. With the phase split, each phase is timed on its own as well. This
is a separate observer, never a reported forecast run.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('captures', type=Path)
p.add_argument('--ncol', type=int, default=50200)
p.add_argument('--calls', default='0010,0030,0060')
p.add_argument('--reps', type=int, default=12)
p.add_argument('--strict', action='store_true')
p.add_argument('--out', type=Path)
a = p.parse_args()
if a.strict:
    os.environ['GPUWM_WRF_EXACT'] = '1'
import cupy as cp
from woof.core import mynn480_gpu as m

meta = m.schema(49)
fields = {n: f for n, f in meta['arguments'].items() if f['type'] != 'character'}
paths = sorted(q for q in a.captures.glob('*.input.bin')
               if q.name.split('.')[0].rsplit('-', 1)[1] in a.calls.split(','))
assert paths, 'no captures'


def read(path):
    raw = path.read_bytes(); offset = 0; result = {}
    for name, f in fields.items():
        count = f['count']
        if name in ('chem1', 'settle1'):
            count = meta['nz'] * int(result['nchem'][0])
        elif name == 'vdep':
            count = int(result['ndvel'][0])
        dtype = '>f4' if f['slab'] == 'f32' else '>i4'
        result[name] = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).astype(dtype[1:])
        offset += count * 4
    return result


base = {s: np.empty((meta['slab_fields'][k], len(paths)), dtype=s)
        for s, k in (('float32', 'f32'), ('int32', 'i32'))}
for column, path in enumerate(paths):
    values = read(path)
    for name, f in fields.items():
        slab = base['float32' if f['slab'] == 'f32' else 'int32']
        slab[f['offset']:f['offset'] + f['count'], column] = values[name][:f['count']]
    for name in ('nchem', 'ndvel'):
        base['int32'][fields[name]['offset'], column] = 1
n = a.ncol
tile = np.arange(n) % len(paths)
pristine = {s: cp.asarray(np.ascontiguousarray(v[:, tile])) for s, v in base.items()}
work = {s: cp.empty_like(v) for s, v in pristine.items()}
arena = cp.zeros(n * meta['scratch_bytes'], dtype=cp.uint8)
module, kernels = m.image(49, cp.cuda.Device().id, a.strict)[:2]
plan = m.launch_plan(module, a.strict)
threads = 32 if a.strict else 128
args = (np.int32(n), work['float32'], work['int32'], arena)
samples = {name: [] for name, _, _ in plan}
samples['total'] = []
for rep in range(a.reps + 2):
    for s in work:
        work[s][...] = pristine[s]
    events = [cp.cuda.Event() for _ in range(len(plan) + 1)]
    events[0].record()
    for i, (name, kernel, ysplit) in enumerate(plan):
        kernel(((n + threads - 1) // threads, ysplit), (threads,), args)
        events[i + 1].record()
    events[-1].synchronize()
    if rep < 2:
        continue
    for i, (name, _, _) in enumerate(plan):
        samples[name].append(cp.cuda.get_elapsed_time(events[i], events[i + 1]))
    samples['total'].append(cp.cuda.get_elapsed_time(events[0], events[-1]))
report = dict(ncol=n, columns=len(paths), strict=a.strict,
              device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              median_ms={k: float(np.median(v)) for k, v in samples.items()},
              min_ms={k: float(np.min(v)) for k, v in samples.items()})
print(json.dumps(report, indent=1))
if a.out:
    a.out.write_text(json.dumps(dict(report, samples=samples), indent=1) + '\n')
