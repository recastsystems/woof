"""Replay an evolved native hydrostatic vertical velocity module call."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.relaxation_wrf471_oracle.replay import grade


def read_capture(path):
    raw = Path(path).read_bytes()
    magic, nx, ny, nz = map(int, np.frombuffer(raw, '<i4', 4))
    if magic != 3108 or min(nx, ny, nz) < 1:
        raise ValueError('invalid native diagnosed W capture')
    offset = 16

    def take(shape, trans=None):
        nonlocal offset
        count = int(np.prod(shape))
        value = np.frombuffer(raw, '<f4', count, offset).reshape(shape, order='F')
        offset += 4*count
        return value.transpose(trans) if trans is not None else value

    constants = take((7,))
    profiles = np.stack([take((nz+1,)) for _ in range(2)])
    fields = {name: take((nx, ny), (1, 0)) for name in ('mut', 'msftx', 'msfty')}
    fields['height_halo'] = take((nx+2, ny+2), (1, 0))
    # Native W only reads its first three wind levels at the lower boundary.
    # Zero-pad unused levels; the captured words themselves are unchanged.
    for name, shape in (('u', (nz, ny, nx+1)), ('v', (nz, ny+1, nx))):
        fields[name] = np.zeros(shape, np.float32)
        fields[name][:3] = take((shape[2], 3, shape[1]), (1, 2, 0))
    fields.update({name: take((nx, nz+1, ny), (1, 2, 0))
                   for name in ('ph_tend', 'ph_new', 'ph_old')})
    expected = take((nx, nz+1, ny), (1, 2, 0))
    if offset != len(raw):
        raise ValueError('native diagnosed W capture length mismatch')
    return (nx, ny, nz), constants, profiles, fields, expected, dict(
        path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def run(path):
    import cupy as cp
    from woof.core.global_hydrostatic import HydrostaticVerticalVelocity
    from woof.certify.kernel_manifest import kernel_manifest
    dims, constants, profiles, fields, expected, capture = read_capture(path)
    device = lambda v: cp.asarray(np.ascontiguousarray(v))
    data = {name: device(v) for name, v in fields.items()}
    data['w'] = cp.empty_like(data['ph_new'])
    operator = HydrostaticVerticalVelocity(*dims, constants=device(constants), profiles=device(profiles))
    operator.apply(**data)
    result = grade(cp.asnumpy(data['w']), expected)
    return dict(capture=capture, output=result, words=result['words'],
                different_words=result['different_words'],
                passed=result['different_words'] == 0, kernel_manifest=kernel_manifest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    result = run(args.capture)
    args.receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['output'], indent=2))
    raise SystemExit(0 if result['passed'] else 1)
