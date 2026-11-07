"""Replay real native hydrostatic acoustic initialization and evolved calls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.relaxation_wrf471_oracle.replay import grade


def read_capture(path):
    raw = Path(path).read_bytes()
    magic, nx, ny, nz, step = map(int, np.frombuffer(raw, '<i4', 5))
    if magic != 3107 or min(nx, ny, nz) < 1:
        raise ValueError('invalid native hydrostatic acoustic capture')
    offset = 20

    def take(shape, trans=None):
        nonlocal offset
        count = int(np.prod(shape))
        value = np.frombuffer(raw, '<f4', count, offset).reshape(shape, order='F')
        offset += 4*count
        return value.transpose(trans) if trans is not None else value

    constants = take((2,))
    profiles = np.stack([take((nz,)) for _ in range(4)])
    fields = {name: take((nx, ny), (1, 0)) for name in ('mu', 'mut')}
    fields.update({name: take((nx, nz, ny), (1, 2, 0))
                   for name in ('alt', 'theta_coupled', 'theta_ref', 'c2a')})
    fields['ph'] = take((nx, nz+1, ny), (1, 2, 0))
    fields['pm1'] = take((nx, nz, ny), (1, 2, 0))
    expected = {name: take((nx, nz, ny), (1, 2, 0)) for name in ('p', 'al')}
    expected['ph'] = take((nx, nz+1, ny), (1, 2, 0))
    expected['pm1'] = take((nx, nz, ny), (1, 2, 0))
    if offset != len(raw):
        raise ValueError('native hydrostatic acoustic capture length mismatch')
    return (nx, ny, nz), step, constants, profiles, fields, expected, dict(
        path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def run(directory):
    import cupy as cp
    from woof.core.global_hydrostatic import HydrostaticAcoustic
    from woof.certify.kernel_manifest import kernel_manifest
    calls = []
    for path in sorted(Path(directory).glob('acoustic-*.bin')):
        dims, step, constants, profiles, fields, expected, capture = read_capture(path)
        device = lambda v: cp.asarray(np.ascontiguousarray(v))
        data = {name: device(v) for name, v in fields.items()}
        data.update(p=cp.empty_like(data['alt']), al=cp.empty_like(data['alt']))
        operator = HydrostaticAcoustic(*dims, profiles=device(profiles),
                                       t0=constants[0], smdiv=constants[1])
        operator.apply(**data, first=step == 0)
        results = {name: grade(cp.asnumpy(data[name]), value)
                   for name, value in expected.items()}
        calls.append(dict(capture=capture, native_step=step, outputs=results))
    if len(calls) != 2 or not any(c['native_step'] == 0 for c in calls) or not any(c['native_step'] > 0 for c in calls):
        raise ValueError('native acoustic initialization and evolved calls required')
    outputs = [r for call in calls for r in call['outputs'].values()]
    return dict(calls=calls, words=sum(r['words'] for r in outputs),
                different_words=sum(r['different_words'] for r in outputs),
                passed=all(r['different_words'] == 0 for r in outputs),
                kernel_manifest=kernel_manifest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture_directory', type=Path)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    result = run(args.capture_directory)
    args.receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['calls'], indent=2))
    raise SystemExit(0 if result['passed'] else 1)
