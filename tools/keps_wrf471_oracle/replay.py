#!/usr/bin/env python3
"""Replay unmodified WRF 4.7.1 KEPS calls captured by instrument.py.

Select GPUWM_WRF_EXACT=1 before import. Every entry state comes from WRF;
there is no synthetic input generator or accumulation of GPU replay error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path
import struct

import numpy as np

MASS = ('u', 'v', 'th', 'qv', 'qc', 'rho', 'dz8w', 'pi', 'rthraten')
SURFACE = ('ust', 'hfx', 'qfx', 'br', 'znt', 'psim', 'psih')
STATE = ('tke_pbl', 'diss_pbl', 'tpe_pbl', 'pr_pbl', 'tke_adv', 'diss_adv', 'tpe_adv')
OUTPUT = ('rublten', 'rvblten', 'rthblten', 'rqvblten', 'rqcblten',
          'wu_tur', 'wv_tur', 'wt_tur', 'exch_h', 'exch_m')


def read_capture(path):
    raw = Path(path).read_bytes()
    if Path(path).suffix == '.xz':
        raw = lzma.decompress(raw)
    nz, ny, nx, step = struct.unpack_from('<4i', raw)
    dt, cp_air, gravity = struct.unpack_from('<3f', raw, 16)
    offset = 28

    def read(shape):
        nonlocal offset
        size = int(np.prod(shape))
        result = np.frombuffer(raw, dtype='<f4', count=size, offset=offset).reshape(shape).copy()
        offset += size * 4
        return result

    s3, s2 = (nz, ny, nx), (ny, nx)
    inputs = {name: read(s3) for name in MASS}
    inputs['p8w'] = read((nz + 1, ny, nx))
    inputs.update({name: read(s2) for name in SURFACE})
    state = {name: read(s3) for name in STATE}
    state['pblh'] = read(s2)
    expected = {name: read(s3) for name in OUTPUT + STATE}
    expected['pblh'] = read(s2)
    if offset != len(raw):
        raise ValueError(f'{path}: consumed {offset} of {len(raw)} bytes')
    return inputs, state, expected, dict(
        path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
        shape=list(s3), itimestep=step, dt=dt, cp_air=cp_air, g=gravity)


def replay(path):
    import cupy as cp
    from woof.core.keps import keps_step

    inputs, initial, expected, receipt = read_capture(path)
    inputs = {k: cp.asarray(v) for k, v in inputs.items()}
    state = {k: cp.asarray(v) for k, v in initial.items()}
    out = keps_step(inputs, state, dt=receipt['dt'], itimestep=receipt['itimestep'],
                    cp_air=receipt['cp_air'], g=receipt['g'])
    result = {**out, **state}
    receipt['fields'] = {}
    for name, want in expected.items():
        got = cp.asnumpy(result[name])
        bad = got.view(np.uint32) != want.view(np.uint32)
        finite = np.isfinite(want) & np.isfinite(got)
        sample = [dict(index=list(map(int, index)),
                       native_hex=f'{want.view(np.uint32)[tuple(index)]:08x}',
                       gpu_hex=f'{got.view(np.uint32)[tuple(index)]:08x}')
                  for index in np.argwhere(bad)[:4]]
        receipt['fields'][name] = dict(
            words=int(want.size), differing_words=int(bad.sum()),
            native_nonfinite=int((~np.isfinite(want)).sum()),
            gpu_nonfinite=int((~np.isfinite(got)).sum()),
            max_abs_finite=float(np.abs(got[finite].astype(np.float64) - want[finite]).max(initial=0)),
            first_differences=sample)
    receipt['words'] = sum(row['words'] for row in receipt['fields'].values())
    receipt['differing_words'] = sum(row['differing_words'] for row in receipt['fields'].values())
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path, nargs='+')
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    from woof.wrf_exact import ENABLED, compile_receipt
    if not ENABLED:
        parser.error('GPUWM_WRF_EXACT=1 must be set before import')
    rows = [replay(path) for path in args.captures]
    result = dict(schema='gpuwm-keps-wrf471-real-replay-v1', captures=rows,
                  words=sum(row['words'] for row in rows),
                  differing_words=sum(row['differing_words'] for row in rows),
                  compiler=compile_receipt())
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('captures', 'compiler')}))
    raise SystemExit(bool(result['differing_words']))
