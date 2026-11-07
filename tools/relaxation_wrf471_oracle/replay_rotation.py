"""Replay native evolved directional Coriolis and polar curvature together."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.relaxation_wrf471_oracle.replay import grade


def read_capture(path):
    raw = Path(path).read_bytes()
    magic, nx, ny, nz = map(int, np.frombuffer(raw, '<i4', 4))
    if magic != 3110 or min(nx, ny, nz) < 1:
        raise ValueError('invalid native rotation capture')
    offset = 16

    def take(shape, trans=None):
        nonlocal offset
        count = int(np.prod(shape))
        value = np.frombuffer(raw, '<f4', count, offset).reshape(shape, order='F')
        offset += 4*count
        return value.transpose(trans) if trans is not None else value

    metrics = [take(shape, (1, 0)) for shape in
               ((nx, ny), (nx, ny), (nx+1, ny), (nx+1, ny), (nx, ny+1), (nx, ny+1))]
    geography = [take((nx, ny), (1, 0)) for _ in range(4)]
    profiles = np.stack([take((nz+1,)) for _ in range(2)])
    ushape, vshape, wshape = (nx+1, nz, ny), (nx, nz, ny+1), (nx, nz+1, ny)
    fields = {name: take(shape, (1, 2, 0)) for name, shape in
              zip(('ru', 'rv', 'rw', 'ru_tend', 'rv_tend', 'rw_tend'),
                  (ushape, vshape, wshape)*2)}
    halo_geo = take((3*ny,))
    halo_rv = take((nz*(ny+1),))
    halo_rw = take(((nz+1)*ny,))
    constants = take((2,))
    geography.append(take((nx, ny), (1, 0)))
    halo_lat = take((ny,))
    fields['zonal_halo'] = np.concatenate((halo_geo, halo_lat, halo_rv, halo_rw))
    fields['u'] = take(ushape, (1, 2, 0))
    fields['v'] = take(vshape, (1, 2, 0))
    expected = {name: take(shape, (1, 2, 0)) for name, shape in
                zip(('ru_tend', 'rv_tend', 'rw_tend'), (ushape, vshape, wshape))}
    if offset != len(raw):
        raise ValueError('native rotation capture length mismatch')
    return (nx, ny, nz), metrics, geography, profiles, constants, fields, expected, dict(
        path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def run(path):
    import cupy as cp
    from woof.core.global_rotation import GlobalRotation
    from woof.certify.kernel_manifest import kernel_manifest
    dims, metrics, geography, profiles, constants, fields, expected, capture = read_capture(path)
    device = lambda v: cp.asarray(np.ascontiguousarray(v))
    data = {name: device(value) for name, value in fields.items() if name != 'zonal_halo'}
    operator = GlobalRotation(*dims, metrics=[device(v) for v in metrics],
                              geography=[device(v) for v in geography], profiles=device(profiles),
                              reradius=constants[0], degrad=constants[1],
                              zonal_halo=device(fields['zonal_halo']))
    operator.apply(**data, refresh_zonal_halo=False)
    outputs = {name: grade(cp.asnumpy(data[name]), value) for name, value in expected.items()}
    return dict(capture=capture, outputs=outputs, words=sum(r['words'] for r in outputs.values()),
                different_words=sum(r['different_words'] for r in outputs.values()),
                passed=all(r['different_words'] == 0 for r in outputs.values()),
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
