#!/usr/bin/env python3
"""Compare retained real native columns with the generated GPU candidate.

All comparisons use raw 32-bit words. No tolerance or automatic exclusions.
The output lists every differing field; it cannot qualify a partial match.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('work', type=Path)
p.add_argument('captures', type=Path)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--set', action='append', default=[])
p.add_argument('--limit', type=int, default=0)
p.add_argument('--image', type=Path)
p.add_argument('--native-only', action='store_true')
p.add_argument('--host', action='store_true', help='Debug lowering on CPU, never a GPU qualification')
a = p.parse_args()
meta = json.loads((a.work / 'schema.json').read_text())
fields = {n: f for n, f in meta['arguments'].items() if f['type'] != 'character'}
overrides = dict(item.split('=', 1) for item in a.set)
for name in overrides:
    if name not in fields or fields[name]['dims']:
        raise ValueError('Override must name a native scalar: ' + name)
paths = sorted(a.captures.glob('*.input.bin'))
if a.limit:
    paths = paths[:a.limit]
if not paths:
    raise ValueError('No retained native captures')
a.out.mkdir(parents=True, exist_ok=True)


def read(path):
    raw = path.read_bytes()
    offset = 0
    result = {}
    for name, f in fields.items():
        count = f['count']
        if name == 'chem':
            count = meta['nz'] * int(result['nchem'][0])
        elif name == 'vdep':
            count = int(result['ndvel'][0])
        dtype = '>f4' if f['slab'] == 'f32' else '>i4'
        result[name] = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).astype(dtype[1:])
        offset += count * 4
    if offset != len(raw):
        raise ValueError((path, offset, len(raw)))
    return result


n = len(paths)
slabs = {s: np.empty((count, n), dtype=s) for s, count in
         (('float32', meta['slab_fields']['f32']), ('int32', meta['slab_fields']['i32']))}
references = []
for column, path in enumerate(paths):
    values = read(path)
    for name, value in overrides.items():
        values[name][0] = (value.lower() in ('true', '.true.', '1')
                          if fields[name]['type'] == 'logical' else float(value))
    for name, f in fields.items():
        slab = slabs['float32' if f['slab'] == 'f32' else 'int32']
        slab[f['offset']:f['offset'] + f['count'], column] = values[name][:f['count']]
    # No chemical solve is admitted. Its unused columns remain native input
    # aliases; the device work array has one inactive chemical column.
    assert not values['mix_chem'][0]
    for name in ('nchem', 'ndvel'):
        slabs['int32'][fields[name]['offset'], column] = 1
    reference = a.out / (path.stem + '.native.bin')
    command = [str((a.work / 'referee').resolve()), str(path.resolve()), str(reference.resolve())]
    for name, value in overrides.items():
        command += [name, value]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    references.append(read(reference))

if a.native_only:
    if overrides:
        raise ValueError('Capture witness compares the unchanged native configuration')
    report = dict(columns=n, differing=0, fields={}, witness='native standalone repeated call')
    for column, path in enumerate(paths):
        witness = a.out / (path.stem + '.repeat.native.bin')
        subprocess.run([str((a.work/'referee').resolve()), str(path.resolve()),
                        str(witness.resolve())], check=True, stdout=subprocess.DEVNULL)
        captured = read(witness)
        for name, f in fields.items():
            if f['intent'] == 'in':
                continue
            got, expected = references[column][name], captured[name]
            diff = int(np.count_nonzero(got.view('u4') != expected.view('u4')))
            report['differing'] += diff
            if diff:
                report['fields'].setdefault(name, []).append(dict(column=column, differing=diff))
    (a.out/'native-capture-witness.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(1 if report['differing'] else 0)

started = time.monotonic()
if a.host:
    import ctypes
    library=ctypes.CDLL(str((a.work/'host.so').resolve()))
    kernel=library.mynn471_step
    kernel.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    host={s:v.copy() for s,v in slabs.items()}
    for c in range(n):
        f=np.ascontiguousarray(host['float32'][:,c])
        i=np.ascontiguousarray(host['int32'][:,c])
        kernel(1,f.ctypes.data,i.ctypes.data)
        host['float32'][:,c]=f;host['int32'][:,c]=i
else:
    import cupy as cp
    from woof.core.kernels import compile_module_no_ftz
    from source import cuda_source
    source=cuda_source(a.work)
if not a.host and a.image:
    import hashlib
    manifest = json.loads(a.image.with_suffix('.json').read_text())
    assert manifest['mode'] == 'strict'
    assert manifest['source_sha256'] == hashlib.sha256(source.encode()).hexdigest()
    assert manifest['image_sha256'] == hashlib.sha256(a.image.read_bytes()).hexdigest()
    assert manifest['schema_sha256'] == hashlib.sha256((a.work/'schema.json').read_bytes()).hexdigest()
    module = cp.cuda.function.Module()
    module.load(a.image.read_bytes())
elif not a.host:
    module = compile_module_no_ftz('mynn471', source)
if not a.host:
    kernel = module.get_function('mynn471_step')
    print('compile_or_load_seconds', time.monotonic() - started, flush=True)
    device = {s: cp.asarray(v) for s, v in slabs.items()}
    args=(np.int32(n), device['float32'], device['int32'])
    if meta.get('scratch_bytes'):
        scratch=cp.empty(n*meta['scratch_bytes'], dtype=cp.uint8)
        args+= (scratch,)
    print('device_free_before_launch', cp.cuda.runtime.memGetInfo()[0],
          'explicit_scratch_bytes',n*meta.get('scratch_bytes',0),flush=True)
    kernel(((n + 31) // 32,), (32,), args)
    cp.cuda.get_current_stream().synchronize()
    host = {s: cp.asnumpy(v) for s, v in device.items()}
column_differences=np.zeros(n,dtype=np.int64)
report = dict(columns=n, overrides=overrides, compared=0, differing=0, fields={})
for name, f in fields.items():
    if f['intent'] == 'in':
        continue
    slab = host['float32' if f['slab'] == 'f32' else 'int32']
    got = np.ascontiguousarray(slab[f['offset']:f['offset'] + f['count']].T)
    expected = np.stack([r[name] for r in references])
    if name == 'chem' and expected.shape[1] != f['count']:
        # The admitted wrapper has mix_chem=False. Extra chemical input
        # columns alias their input, just as the unmodified native does.
        # Compare those words too; never mark them as exclusions.
        assert all(not read(path)['mix_chem'][0] for path in paths)
        complete = np.stack([read(path)[name] for path in paths])
        complete[:, :f['count']] = got
        got = complete
    different = got.view('u4') != expected.view('u4')
    column_differences += different.sum(axis=1)
    report['compared'] += different.size
    report['differing'] += int(different.sum())
    detail = dict(words=different.size, differing=int(different.sum()))
    if different.any():
        idx = np.argwhere(different)[0]
        detail['first'] = dict(column=int(idx[0]), level=int(idx[1]),
                               gpu=float(got[tuple(idx)]), native=float(expected[tuple(idx)]),
                               gpu_word=int(got.view('u4')[tuple(idx)]),
                               native_word=int(expected.view('u4')[tuple(idx)]))
    report['fields'][name] = detail
report['column_differences']={path.name:int(count) for path,count in zip(paths,column_differences)}
(a.out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
raise SystemExit(1 if report['differing'] else 0)
