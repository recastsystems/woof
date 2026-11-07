#!/usr/bin/env python3
"""Replay initial PW and held pressure from bounded real native captures.

This compares device words to captured native inputs. Reference profile
constants come from that run's own input, never a replacement atmosphere.
The capture's independently retained successful native-run receipt is required.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--wrfinput', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import cupy as cp
    from woof import netcdf_bridge as nc, wrf_exact
    from woof.core.kernels import get_kernel, module_source, module_options
    if not wrf_exact.ENABLED:
        parser.error('initial-PW native replay requires GPUWM_WRF_EXACT=1 before startup')
    with np.load(args.capture, allow_pickle=False) as archive:
        data = {k: np.ascontiguousarray(archive[k], dtype='f4')
                for k in ('p', 'pp', 'pb', 'pw', 'terrain')}
        provenance = json.loads(str(archive['provenance']))
    if not provenance.get('real_native_capture'):
        parser.error('capture lacks real native-run provenance')
    with nc.open_dataset(args.wrfinput) as src:
        constants = [np.float32(np.asarray(src.variables[k][:]).reshape(-1)[0])
                     for k in ('P00', 'T00', 'TLP')]
        c3f, c4f = [cp.asarray(np.asarray(src.variables[k][:]).reshape(-1), dtype='f4')
                     for k in ('C3F', 'C4F')]
        ptop = np.float32(np.asarray(src.variables['P_TOP'][:]).reshape(-1)[0])
    pressure = np.ascontiguousarray(data['pp'] + data['pb'])
    p, pb, ht = map(cp.asarray, (pressure, data['pb'], data['terrain']))
    nz, ny, nx = pressure.shape
    n = ny * nx
    out = cp.empty(data['pw'].shape, dtype='f4')
    bounds = cp.empty((2, ny, nx), dtype='f4')
    get_kernel('level_diagnostics', 'level_initial_pressure')(((n+127)//128,), (128,),
        (p, pb, ht, c3f, c4f, out, bounds, *constants, ptop,
         np.int32(nz), np.int32(n), np.int32(1)))
    initial = cp.asnumpy(out)
    held_p, held_pw = cp.empty_like(p), cp.empty_like(out)
    get_kernel('level_diagnostics', 'level_capture_pressure')(((n+127)//128,), (128,),
        (cp.asarray(data['p']), cp.asarray(data['pw']), held_p, held_pw,
         bounds, np.int32(nz), np.int32(n)))
    results = []
    for name, actual, expected in (('initial_pw', initial, data['pw']),
            ('held_pressure', cp.asnumpy(held_p), data['p']),
            ('held_pw', cp.asnumpy(held_pw), data['pw'])):
        results.append(dict(field=name, words=expected.size,
            differing_words=int(np.count_nonzero(actual.view('u4') != expected.view('u4'))),
            native_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
            device_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
    expected_bounds = np.stack((data['p'][:-1].max(axis=0), data['pw'][:-1].max(axis=0)))
    private_diff = int(np.count_nonzero(cp.asnumpy(bounds).view('u4') != expected_bounds.view('u4')))
    files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.capture, args.wrfinput)}
    receipt = dict(schema='wrf471-level-pressure-replay-v1', provenance=provenance,
        command=sys.argv, strict=True, dimensions=dict(nz=nz, ny=ny, nx=nx), files=files,
        input_pressure_vs_native_hydrostatic_max=float(np.max(np.abs(pressure-data['p']))),
        results=results, private_endpoint_bounds_differing_words=private_diff,
        private_endpoint_bounds_counted_as_native_words=False,
        module_source_sha256=hashlib.sha256(module_source('level_diagnostics').encode()).hexdigest(),
        compiler_options=list(wrf_exact.effective_options(module_options('level_diagnostics'))),
        gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
    receipt['status'] = 'PASS' if not private_diff and all(not x['differing_words'] for x in results) else 'FAIL'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'], results=results, private_bound_differences=private_diff)))
    return 0 if receipt['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
