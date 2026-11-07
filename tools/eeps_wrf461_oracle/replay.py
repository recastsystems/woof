"""Replay recorded native EEPS calls and compare every output word.

Use GPUWM_WRF_EXACT=1. Each recorded call restores its own native carried
state, so real captures need no synthetic stand-in for scalar advection.
The sibling synthetic parity test separately checks multi-call GPU carry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def replay(path):
    import cupy as cp
    import numpy as np
    from woof.core import eeps_pbl
    from woof.verify.eeps_oracle import read_trace, ulp_distance

    trace = read_trace(path)
    def dev(array):
        array = cp.asarray(array)
        return cp.ascontiguousarray(array[:, None, :] if array.ndim == 2
                                    else array[None, :])
    report = {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'header':trace['header'], 'calls':[]}
    for step in trace['steps']:
        inputs = step['inputs']
        columns = {name:dev(inputs[name]) for name in eeps_pbl.COLUMN_INPUTS}
        surface = {name:dev(inputs[name]) for name in eeps_pbl.SURFACE_INPUTS}
        carried = {name:dev(inputs[key]) for name, key in (
            ('pek_pbl','pek'), ('pep_pbl','pep'),
            ('pek_adv','pek_adv'), ('pep_adv','pep_adv'))}
        outputs = eeps_pbl.eeps_step(columns, surface, carried,
            dt=trace['header']['dt'], first_step=step['itimestep'] == 1)
        outputs.update(pek=carried['pek_pbl'], pep=carried['pep_pbl'],
                       pek_adv=carried['pek_adv'], pep_adv=carried['pep_adv'])
        fields = {}
        for name, expected in step['outputs'].items():
            actual = cp.asnumpy(outputs[name]).reshape(expected.shape)
            mismatch = actual.view(np.uint32) != expected.view(np.uint32)
            item = {'words':int(expected.size), 'differing_words':int(mismatch.sum())}
            if expected.dtype.kind == 'f':
                item['max_ulp'] = int(ulp_distance(actual, expected).max())
                item['max_absolute_error'] = float(np.max(np.abs(actual - expected)))
            fields[name] = item
        report['calls'].append({'itimestep':step['itimestep'], 'fields':fields,
            'inputs':{name:{'minimum':float(array.min()), 'maximum':float(array.max()),
                            'nonzero_words':int(np.count_nonzero(array)),
                            'finite':bool(np.isfinite(array).all())}
                      for name,array in inputs.items()}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', nargs='+', type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get('GPUWM_WRF_EXACT') != '1':
        parser.error('the native word grade requires GPUWM_WRF_EXACT=1')
    reports = [replay(path) for path in args.captures]
    fields = [field for report in reports for call in report['calls']
              for field in call['fields'].values()]
    from woof.core import eeps_pbl
    from woof.wrf_exact import effective_options
    import cupy as cp
    receipt = {'schema':'eeps-native-module-replay-v1', 'strict':True,
               'gpu':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
               'cuda_runtime':cp.cuda.runtime.runtimeGetVersion(),
               'kernel_source_sha256':{str(report['header']['nz']):hashlib.sha256(
                   eeps_pbl.module_source(report['header']['nz']).encode()).hexdigest()
                   for report in reports},
               'compiler_options':list(effective_options(eeps_pbl.MODULE_OPTIONS)),
               'words':sum(x['words'] for x in fields),
               'differing_words':sum(x['differing_words'] for x in fields),
               'captures':reports}
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({key:value for key,value in receipt.items() if key != 'captures'}))
    return int(receipt['differing_words'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
