"""Replay an evolved native hydrostatic diagnostic module capture."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.relaxation_wrf471_oracle.replay import grade


def read_capture(path):
    raw = Path(path).read_bytes()
    magic, nx, ny, nz, nmoist, theta_m, hypso, _ = map(int, np.frombuffer(raw, '<i4', 8))
    if magic != 3104:
        raise ValueError("not a native hydrostatic diagnostic capture")
    offset = 32

    def take(shape, trans=None):
        nonlocal offset
        count = int(np.prod(shape))
        a = np.frombuffer(raw, '<f4', count, offset).reshape(shape, order='F')
        offset += 4*count
        return a.transpose(trans) if trans is not None else a

    constants = take((6,))
    names = ('c1', 'c2', 'c3h', 'c4h', 'c3f', 'c4f', 'dnw', 'rdnw', 'rdn')
    profiles = np.zeros((9, nz+1), np.float32)
    for k, name in enumerate(names):
        size = nz+1 if name in ('c3f', 'c4f') else nz
        profiles[k, :size] = take((size,))
    fields = {n: take((nx, ny), (1, 0)) for n in ('mu', 'muts')}
    fields.update({n: take((nx, nz, ny), (1, 2, 0)) for n in ('alb', 'pb', 't')})
    fields.update({n: take((nx, nz+1, ny), (1, 2, 0)) for n in ('ph', 'phb')})
    fields['moist'] = take((nx, nz, ny, nmoist), (3, 1, 2, 0))
    outputs = {n: take((nx, nz, ny), (1, 2, 0)) for n in ('p', 'al')}
    outputs['ph'] = take((nx, nz+1, ny), (1, 2, 0))
    if offset != len(raw):
        raise ValueError("invalid native diagnostic capture length")
    return (nx, ny, nz), constants, profiles, fields, outputs, dict(
        nmoist=nmoist, use_theta_m=theta_m, hypsometric_opt=hypso), dict(
        path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def run(path):
    import cupy as cp
    from woof.core.global_hydrostatic import HydrostaticDiagnostics
    from woof.certify.kernel_manifest import kernel_manifest
    dims, constants, profiles, fields, expected, kw, capture = read_capture(path)
    device = lambda v: cp.asarray(np.ascontiguousarray(v))
    data = {n: device(v) for n, v in fields.items()}
    data.update(p=cp.empty_like(data['t']), al=cp.empty_like(data['t']),
                full_pressure=cp.empty_like(data['t']), alt=cp.empty_like(data['t']))
    operator = HydrostaticDiagnostics(*dims, constants=device(constants), profiles=device(profiles), **kw)
    operator.apply(**data)
    results = {n: grade(cp.asnumpy(data[n]), v) for n, v in expected.items()}
    return dict(capture=capture, configuration=kw, outputs=results,
                words=sum(v['words'] for v in results.values()),
                different_words=sum(v['different_words'] for v in results.values()),
                passed=all(v['different_words'] == 0 for v in results.values()),
                kernel_manifest=kernel_manifest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    result = run(args.capture)
    args.receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['outputs'], indent=2))
    raise SystemExit(0 if result['passed'] else 1)
