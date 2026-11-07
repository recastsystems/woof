"""Strict CUDA replay of paired native forecast driver-boundary captures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .capture import layouts, read
from .diagnose import compare


def native_undefined_words(key, actual, expected, witness):
    """Count only storage independently witnessed undefined by stock WRF.

    Raw comparison is retained. Neither a floating-point difference nor
    an active-column diagnostic is exempted by this witness.
    """
    if key != 'cu7' or witness is None:
        return 0
    if (witness.get('kind') != 'native-undefined-storage-witness'
            or witness.get('passed') is not True
            or witness.get('source_commit') != 'f52c197ed39d12e087d02c50f412d90d418f6186'):
        raise ValueError('CU undefined storage needs the pinned native seeding witness')
    inactive = ((expected['lengath2d'] == 0) & (expected['maxg2d'] == 1)
                & (actual['lengath2d'] == expected['lengath2d'])
                & (actual['maxg2d'] == expected['maxg2d']))
    count = 0
    for name in ('ideep2d', 'jt2d'):
        bad = expected[name].view(np.uint32) != actual[name].view(np.uint32)
        count += int(np.count_nonzero(bad & inactive))
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--undefined-witness', type=Path)
    args = ap.parse_args()
    from woof import numerics, wrf_exact
    if not numerics.is_strict() or not wrf_exact.ENABLED:
        raise ValueError('native word replay requires GPUWM_WRF_EXACT=1 before import')
    import cupy as cp
    from woof.core.camzm import camzm_step
    from woof.core.uwshcu import uwshcu_step
    from woof.core.cammgmp import cammgmp_step, CAMMGMP_INOUT_3D, CAMMGMP_INOUT_2D
    schema = layouts()
    witness = (None if args.undefined_witness is None else
               json.loads(args.undefined_witness.read_text()))
    report = dict(kind='native-forecast-driver-capture', real_native_capture=True,
                  strict_numerics=True, numerics=numerics.receipt(),
                  cupy_version=cp.__version__, captures=[], words32=0,
                  differing_words=0, witnessed_native_undefined_words=0,
                  defined_differing_words=0,
                  undefined_witness=(None if args.undefined_witness is None else
                      dict(path=str(args.undefined_witness), sha256=hashlib.sha256(
                          args.undefined_witness.read_bytes()).hexdigest())))
    for p in sorted(args.directory.glob('*-in.bin')):
        key = p.name.split('-')[0]
        if key not in schema:
            continue
        refpath = p.with_name(p.name.replace('-in.bin', '-out.bin'))
        if not refpath.is_file():
            raise ValueError(f'no completed native output for {p}')
        meta, inputs = read(p, schema[key]['in'])
        refmeta, expected = read(refpath, schema[key]['out'])
        assert meta == refmeta
        invalid = {stage + '/' + n: int((~np.isfinite(v)).sum())
                   for stage, arrays in (('in', inputs), ('out', expected))
                   for n, v in arrays.items() if not np.isfinite(v).all()}
        if invalid:
            report['captures'].append(dict(path=str(p), native_nonfinite=invalid))
            args.out.write_text(json.dumps(report, indent=2) + '\n')
            continue
        dev = {n: cp.asarray(a) for n, a in inputs.items()}
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        if key == 'cu7':
            got = camzm_step(dev, dt=meta['dt'], dx=meta['dx'], cudt=meta['cudt'],
                             itimestep=meta['step'], bl_pbl_physics=meta['pbl'],
                             sf_sfclay_physics=meta['sfc'], diagnostics=True)
        elif key == 'shcu2':
            got = uwshcu_step(dev, {'cush': dev['cush']}, dt=meta['dt'],
                              has_qnc=meta['p_qnc'] >= meta['first_scalar'],
                              has_qni=meta['p_qni'] >= meta['first_scalar'])
            got['cush'] = dev['cush']
        else:
            got = {n: dev[n] for n in CAMMGMP_INOUT_3D + CAMMGMP_INOUT_2D}
            cammgmp_step(dev, got, dt=meta['dt'], itimestep=meta['step'],
                        cu_coupled=meta['cu'] == 7, shcu_coupled=meta['shcu'] == 2,
                        accum_mode=meta['accum'], aitken_mode=meta['aitken'],
                        coarse_mode=meta['coarse'])
        stop.record(); stop.synchronize()
        actual = {n: cp.asnumpy(a) for n, a in got.items()}
        grade = compare(actual, expected)
        undefined = native_undefined_words(key, actual, expected, witness)
        grade['witnessed_native_undefined_words'] = undefined
        grade['defined_differing_words'] = grade['differ'] - undefined
        for name, ref in expected.items():
            bad = ref.view(np.uint32) != actual[name].view(np.uint32)
            if bad.any():
                idx = tuple(np.argwhere(bad)[0])
                grade['fields'][name]['first'] = dict(index=list(map(int, idx)),
                    native=float(ref[idx]), cuda=float(actual[name][idx]))
        row = dict(input=str(p), output=str(refpath), meta=meta,
                   input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                   output_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),
                   launch_including_allocation_ms=cp.cuda.get_elapsed_time(start, stop),
                   grade=grade)
        report['captures'].append(row)
        report['words32'] += grade['words32']
        report['differing_words'] += grade['differ']
        report['witnessed_native_undefined_words'] += undefined
        report['defined_differing_words'] += grade['defined_differing_words']
        args.out.write_text(json.dumps(report, indent=2) + '\n')
        print(key, meta['step'], meta['its'], meta['jts'],
              grade['differ'], '/', grade['words32'], flush=True)
    if not report['captures']:
        raise ValueError('empty capture set')
    report['passed'] = (report['defined_differing_words'] == 0 and
                        all('native_nonfinite' not in c for c in report['captures']))
    report['compile_receipt'] = wrf_exact.compile_receipt()
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
