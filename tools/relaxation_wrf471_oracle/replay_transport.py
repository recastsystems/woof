"""Replay native hydrostatic acoustic UV and mass/theta transport calls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.relaxation_wrf471_oracle.replay import grade


def read_capture(path):
    raw = Path(path).read_bytes()
    magic, nx, ny, nz = map(int, np.frombuffer(raw, '<i4', 4))
    if magic not in (3112, 3113) or min(nx, ny, nz) < 1:
        raise ValueError('invalid global acoustic transport capture')
    offset = 16

    def take(shape, transpose=None):
        nonlocal offset
        count = int(np.prod(shape))
        value = np.frombuffer(raw, '<f4', count, offset).reshape(shape, order='F')
        offset += count*4
        return value.transpose(transpose) if transpose else value

    def field(levels, rows=ny, cols=nx):
        return take((cols, levels, rows), (1, 2, 0))

    def plane(rows=ny, cols=nx):
        return take((cols, rows), (1, 0))

    constants = take((4,))
    if magic == 3112:
        profiles = np.zeros((6, nz+1), '<f4')
        for k in range(2):
            profiles[k] = take((nz+1,))
        metrics = [np.zeros((ny, nx), '<f4') for _ in range(2)]
        metrics += [plane(ny, nx+1) for _ in range(2)]
        metrics += [plane(ny+1, nx) for _ in range(3)]
        data = dict(muu=plane(ny, nx+1), muv=plane(ny+1, nx),
                    mudf=plane(ny+2, nx+2))
        for name, levels in (('ph', nz+1), ('alt', nz), ('p', nz), ('pb', nz), ('al', nz)):
            data[name] = field(levels, ny+2, nx+2)
        data.update(ru_tend=field(nz, ny, nx+1), rv_tend=field(nz, ny+1, nx),
                    u=field(nz, ny, nx+1), v=field(nz, ny+1, nx))
        expected = dict(u=field(nz, ny, nx+1), v=field(nz, ny+1, nx))
    else:
        profiles = np.stack([take((nz+1,)) for _ in range(6)])
        metrics = [plane() for _ in range(2)]
        metrics += [plane(ny, nx+1) for _ in range(2)]
        metrics += [plane(ny+1, nx) for _ in range(3)]
        data = dict(muu=plane(ny, nx+1), muv=plane(ny+1, nx), mut=plane())
        data.update(u=field(nz, ny, nx+1), v=field(nz, ny+1, nx),
                    u_ref=field(nz, ny, nx+1), v_ref=field(nz, ny+1, nx),
                    theta_ref=field(nz, ny+2, nx+2), ww_ref=field(nz+1),
                    theta_tend=field(nz), mu_tend=plane(), mu=plane(),
                    ww=field(nz+1), theta=field(nz))
        expected = {name: plane() for name in ('mu', 'muts', 'muave', 'mudf')}
        expected.update(ww=field(nz+1), theta=field(nz), theta_ave=field(nz))
        data.update(muts=np.empty_like(data['mu']), muave=np.empty_like(data['mu']),
                    mudf=np.empty_like(data['mu']), theta_ave=np.empty_like(data['theta']),
                    divv=np.empty_like(data['theta']))
    if offset != len(raw):
        raise ValueError('native acoustic transport length mismatch')
    return (nx, ny, nz), magic, constants, profiles, metrics, data, expected, dict(
        path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def run(directory):
    import cupy as cp
    from woof.core.global_acoustic import GlobalAcousticTransport
    from woof.certify.kernel_manifest import kernel_manifest
    calls = []
    for path in sorted(Path(directory).glob('*.bin')):
        dims, magic, constants, profiles, metrics, data, expected, capture = read_capture(path)
        device = lambda value: cp.asarray(np.ascontiguousarray(value))
        arrays = {name: device(value) for name, value in data.items()}
        operator = GlobalAcousticTransport(*dims, profiles=device(profiles),
            metrics=tuple(map(device, metrics)), rdx=constants[0], rdy=constants[1],
            **({'emdiv': constants[3]} if magic == 3112 else {'epssm': constants[3]}))
        method = operator.advance_uv if magic == 3112 else operator.advance_mass_theta
        method(dt=constants[2], **arrays)
        outputs = {name: grade(cp.asnumpy(arrays[name]), value) for name, value in expected.items()}
        calls.append(dict(kind='uv' if magic == 3112 else 'mass_theta', capture=capture, outputs=outputs))
    if {c['kind'] for c in calls} != {'uv', 'mass_theta'} or len(calls) != 2:
        raise ValueError('native UV and mass/theta transport calls required')
    results = [r for c in calls for r in c['outputs'].values()]
    return dict(calls=calls, words=sum(r['words'] for r in results),
                different_words=sum(r['different_words'] for r in results),
                passed=all(r['different_words'] == 0 for r in results),
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
