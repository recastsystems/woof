"""Replay evolved native scalar/u/v transport including polar loop bounds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.relaxation_wrf471_oracle.replay import grade


def read_capture(path):
    raw = Path(path).read_bytes()
    magic, nx, ny, nz, kind, vorder = map(int, np.frombuffer(raw, '<i4', 6))
    if magic != 3111 or min(nx, ny, nz) < 1 or kind not in (1, 2, 3):
        raise ValueError('invalid native global transport capture')
    kind = {1: 'u', 2: 'v', 3: 'scalar'}[kind]
    nxs, nys = nx+int(kind == 'u'), ny+int(kind == 'v')
    offset = 24

    def take(shape, trans=None):
        nonlocal offset
        count = int(np.prod(shape))
        value = np.frombuffer(raw, '<f4', count, offset).reshape(shape, order='F')
        offset += 4*count
        return value.transpose(trans) if trans is not None else value

    spacing = take((2,))
    profiles = np.stack([take((nz+1,)) for _ in range(3)])
    fields = dict(metric=take((nxs, nys), (1, 0)), vx=take((nx, ny+1), (1, 0)))
    for name, shape in (('q', (nx+7, nz, nys)), ('ru', (nx+7, nz, ny)),
                        ('rv', (nx+7, nz, ny+1)), ('ww', (nx+7, nz+1, ny))):
        fields[name] = take(shape, (1, 2, 0))
    fields['tendency'] = take((nxs, nz, nys), (1, 2, 0))
    expected = take((nxs, nz, nys), (1, 2, 0))
    if offset != len(raw):
        raise ValueError('native global transport capture length mismatch')
    return (nx, ny, nz), kind, vorder, spacing, profiles, fields, expected, dict(
        path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def run(directory):
    import cupy as cp
    from woof.core.global_advection import GlobalAdvection
    from woof.certify.kernel_manifest import kernel_manifest
    calls = []
    for path in sorted(Path(directory).glob('*.bin')):
        dims, kind, vorder, spacing, profiles, fields, expected, capture = read_capture(path)
        device = lambda value: cp.asarray(np.ascontiguousarray(value))
        data = {name: device(value) for name, value in fields.items()}
        operator = GlobalAdvection(*dims, profiles=device(profiles))
        operator.apply(**data, kind=kind, rdx=spacing[0], rdy=spacing[1], vorder=vorder)
        output = grade(cp.asnumpy(data['tendency']), expected)
        calls.append(dict(kind=kind, vorder=vorder, capture=capture, output=output))
    if {c['kind'] for c in calls} != {'scalar', 'u', 'v'} or len(calls) != 3:
        raise ValueError('native scalar/u/v transport calls required')
    return dict(calls=calls, words=sum(c['output']['words'] for c in calls),
                different_words=sum(c['output']['different_words'] for c in calls),
                passed=all(c['output']['different_words'] == 0 for c in calls),
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
