"""Replay lane-instrumented real WRF MSKF_CPS calls, preserving INOUT words.

Capture format is documented by evidence/instrument_native.py in the lane
receipt. This verifier is outside the forecast loop. It refuses an unsupported
trigger or aerosol capture, rather than claiming those branches were compared.
Run with GPUWM_WRF_EXACT=1 for the bitwise proof.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def read_capture(path):
    data = path.read_bytes()
    offset = 0
    def read(dtype, count):
        nonlocal offset
        a = np.frombuffer(data, dtype=dtype, count=count, offset=offset).copy()
        offset += a.nbytes
        return a
    header = read('<i4', 12)
    nz, ncol, trigger, ktau, stepcu, warm, qr, qi, qs, aero, edrates, adapt = map(int, header)
    if nz < 8 or nz > 160 or ncol < 1 or trigger != 1 or aero != 0:
        raise ValueError(f'{path}: unsupported real-capture header {header.tolist()}')
    scalars = read('<f4', 17)
    initial, inputs, coords = [], [], []
    for _ in range(ncol):
        coords.append(read('<i4', 2))
        inputs.append(read('<f4', 10*nz + 5))
        initial.append(read('<f4', 26*nz + 8))
    expected = np.stack([read('<f4', 27*nz + 8) for _ in range(ncol)])
    if offset != len(data):
        raise ValueError(f'{path}: consumed {offset} of {len(data)} bytes')
    return header, scalars, np.stack(coords), np.stack(inputs), np.stack(initial), expected


def replay(path, *, graph=False):
    import cupy as cp
    from woof.core import mskf
    h, scalars, coords, values, initial, expected = read_capture(path)
    nz, ncol = map(int, h[:2])
    def dev3(x):
        return cp.asarray(np.ascontiguousarray(x.T.reshape(nz, 1, ncol)))
    inputs = {name: dev3(values[:, j*nz:(j+1)*nz])
              for j, name in enumerate(mskf.INPUT_3D)}
    w = cp.asarray(np.ascontiguousarray(values[:, 9*nz:10*nz+1].T.reshape(nz+1, 1, ncol)))
    for j, name in enumerate(mskf.INPUT_2D):
        inputs[name] = cp.asarray(np.ascontiguousarray(values[:, 10*nz+1+j].reshape(1, ncol)))
    outputs = {name: dev3(initial[:, j*nz:(j+1)*nz])
               for j, name in enumerate(mskf.OUTPUT_3D)}
    for j, name in enumerate(mskf.OUTPUT_2D):
        outputs[name] = cp.asarray(np.ascontiguousarray(initial[:, 26*nz+j].reshape(1, ncol)))
    outputs['cu_act'] = cp.asarray(initial[:, -1].astype(np.int32).reshape(1, ncol))
    dt, dx, cudt, aercu_fct = map(float, scalars[:4])
    def launch():
        mskf.update_w0avg(inputs['w0avg'], w, dt=dt, cudt_minutes=cudt,
                         stepcu=int(h[4]), adaptive=bool(h[11]))
        status = mskf.launch_mskf(inputs, outputs, dt=dt, dx=dx, cudt_minutes=cudt,
            aercu_fct=aercu_fct, kf_edrates=int(h[10]),
            phase=mskf.MskfPhase(*map(bool, h[5:9])))
        mskf.raise_on_status(status)
    if graph:
        from woof.core.health_ledger import HealthLedger, deferring
        baseline = {k:v.copy() for k,v in outputs.items()}
        w0 = inputs['w0avg'].copy()
        ledger = HealthLedger(label='MSKF real-input graph replay')
        with deferring(ledger):
            launch()
        ledger.drain()
        for k,v in baseline.items():
            cp.copyto(outputs[k],v)
        cp.copyto(inputs['w0avg'],w0)
        stream = cp.cuda.Stream(non_blocking=True)
        cp.cuda.get_current_stream().synchronize()
        with stream, deferring(ledger):
            stream.begin_capture()
            launch()
            executable = stream.end_capture()
            executable.launch(stream)
        stream.synchronize()
        ledger.drain()
    else:
        launch()
    chunks = [cp.asnumpy(outputs[name]).reshape(nz, ncol).T for name in mskf.OUTPUT_3D]
    chunks += [cp.asnumpy(outputs[name]).astype(np.float32).reshape(ncol, 1)
               for name in mskf.OUTPUT_2D + ('cu_act',)]
    chunks += [cp.asnumpy(inputs['w0avg']).reshape(nz, ncol).T]
    actual = np.ascontiguousarray(np.concatenate(chunks, axis=1))
    differing = actual.view(np.uint32) != expected.view(np.uint32)
    fields, start, native_words = {}, 0, 0
    inactive = set(mskf.AEROSOL_OUTPUTS)
    if not int(h[10]):
        inactive.update(mskf.EDRATE_OUTPUTS)
    for name, width in [(n, nz) for n in mskf.OUTPUT_3D] + [(n, 1) for n in mskf.OUTPUT_2D + ('cu_act',)] + [('w0avg', nz)]:
        fields[name] = int(differing[:, start:start+width].sum())
        if name not in inactive:
            native_words += ncol * width
        start += width
    first = []
    for col, word in np.argwhere(differing)[:12]:
        first.append({'grid_ij': coords[col].tolist(), 'word': int(word),
                      'expected_hex': f'{int(expected.view(np.uint32)[col,word]):08x}',
                      'actual_hex': f'{int(actual.view(np.uint32)[col,word]):08x}'})
    return {'file': path.name, 'capture_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'ktau': int(h[3]), 'nz': nz, 'columns': ncol, 'words': int(actual.size),
            'native_words': native_words, 'zero_extension_words': int(actual.size)-native_words,
            'inactive_native_fields': sorted(inactive),
            'differing_words': int(differing.sum()), 'per_field': fields,
            'expected_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
            'actual_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
            'precipitating_columns': int((expected[:, 26*nz] > 0).sum()),
            'constants_f32': scalars[4:].tolist(), 'first_differences': first}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('captures', nargs='+', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--graph', action='store_true', help='Capture the real column call as a CUDA graph')
    args = ap.parse_args()
    from woof.wrf_exact import ENABLED
    receipts = [replay(p, graph=args.graph) for p in args.captures]
    receipt = {'strict': ENABLED, 'cuda_graph': args.graph, 'calls': receipts,
               'words': sum(r['words'] for r in receipts),
               'native_words': sum(r['native_words'] for r in receipts),
               'differing_words': sum(r['differing_words'] for r in receipts)}
    args.out.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'calls'}))
    return 0 if receipt['differing_words'] == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
