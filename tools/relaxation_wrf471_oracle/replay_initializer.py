"""Replay a real native global analytic initializer, including RNG input."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.relaxation_wrf471_oracle.replay import grade


def read_capture(path):
    raw = Path(path).read_bytes()
    magic, nx, ny, nz, hybrid, theta_m, _, _ = np.frombuffer(raw, '<i4', 8)
    if magic not in (3101, 3102):
        raise ValueError("not a native global initializer capture")
    offset = 32

    def take(shape):
        nonlocal offset
        n = int(np.prod(shape))
        result = np.frombuffer(raw, '<f4', n, offset).reshape(shape, order='F')
        offset += 4*n
        return result

    constants = take((20,))
    local_surface_weights = take((3,)) if magic == 3102 else None
    fields = {}
    for name in ('znw', 'c1f', 'c2f', 'c3f', 'c4f'):
        fields[name] = take((nz+1,))
    for name in ('znu', 'dnw', 'rdnw', 'c1h', 'c2h', 'c3h', 'c4h'):
        fields[name] = take((nz,))
    for name in ('dn', 'rdn', 'fnp', 'fnm'):
        fields[name] = take((nz-1,))
    for name in ('xlat', 'xlong', 'msftx', 'msfty', 'f', 'e', 'mub', 'mu_2', 'tsk', 'tmn'):
        fields[name] = take((nx, ny)).T
    for name in ('msfvx', 'msfvy', 'msfvx_inv'):
        fields[name] = take((nx, ny+1)).T
    random = take((nx, nz, ny)).transpose(1, 2, 0)
    for name in ('pb', 't_init', 'alb', 't_2', 'p', 'al'):
        fields[name] = take((nx, nz, ny)).transpose(1, 2, 0)
    for name in ('phb', 'ph_2'):
        fields[name] = take((nx, nz+1, ny)).transpose(1, 2, 0)
    if offset != len(raw):
        raise ValueError(f"consumed {offset} of {len(raw)} bytes")
    fields['scalars'] = constants[12:]
    return tuple(map(int, (nx, ny, nz, hybrid))), constants[:12], random, fields, dict(
        path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
        schema=int(magic), use_theta_m=int(theta_m),
        local_surface_weight_bits=(None if local_surface_weights is None
                                   else local_surface_weights.view(np.uint32).tolist()))


def run(path, undefined_witness=None):
    import cupy as cp
    from woof.core.global_ideal_init import initialize
    from woof.certify.kernel_manifest import kernel_manifest
    dims, c, random, native, capture = read_capture(path)
    actual = initialize(*dims[:3], hybrid_opt=dims[3], constants=c,
                        random_samples=cp.asarray(np.ascontiguousarray(random)))
    outputs = {}
    for name, expected in native.items():
        value = actual[name]
        if name in ('dn', 'rdn', 'fnp', 'fnm'):
            value = value[1:dims[2]]  # native endpoint is undefined
        elif expected.ndim == 1:
            value = value[:expected.size]
        outputs[name] = grade(cp.asnumpy(value), expected)
    # Native dummy_decl.inc declares cf1/2/3 as local scalars, not arguments.
    # Its initializer assigns grid%cf1/2/3 but reads the uninitialized locals
    # in TSK. A compiler poison run witnesses the undefined residue. The GPU
    # deliberately uses the computed grid weights, giving a defined surface.
    residue = {}
    witness = None
    if undefined_witness is not None:
        wd, wc, wr, wf, witness = read_capture(undefined_witness)
        weights = np.asarray(witness['local_surface_weight_bits'], np.uint32).view(np.float32)
        if (wd != dims or not np.isnan(weights).all()
                or not np.isnan(wf['tsk']).all() or not np.isnan(wf['tmn']).all()):
            raise ValueError("native poison run did not witness undefined surface weights")
        for name in ('tsk', 'tmn'):
            residue[name] = outputs.pop(name)
    return dict(capture=capture, outputs=outputs, native_undefined_residue=residue,
                undefined_witness=witness,
                words=sum(x['words'] for x in outputs.values()),
                different_words=sum(x['different_words'] for x in outputs.values()),
                kernel_manifest=kernel_manifest(),
                passed=all(x['different_words'] == 0 for x in outputs.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--undefined-witness', type=Path)
    args = parser.parse_args()
    result = run(args.capture, args.undefined_witness)
    args.receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({name: {k: v for k, v in row.items()
                            if k in ('words', 'different_words', 'max_ulp')}
                      for name, row in result['outputs'].items()}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
