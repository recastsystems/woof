"""Module replay (measurement tool, not product code): the compute call split into several launches.

Usage: WOOF_HEX_NUMERICS=fast|strict python replay_slices.py CAPTURE_DIR COLUMNS OUT_JSON
Runs every captured compute-* call through this tree's module twice from the same outputs-before words: once in
one launch (the default budget) and once with the work budget cut so each launch takes COLUMNS solve columns (so
the c0 offset path runs), and compares all 41 output rows word for word (uint32 bits), and both against the
captured run's outputs-after (c2efafcf words).
"""
import json
import os
import sys

import numpy as np
import cupy as cp

CAPTURE, COLUMNS, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]

from woof.hex import numerics
from woof.hex.mpasdiag import convective_rrfs as new
COLUMN = {'zgrid', 'w', 'uReconstructZonal', 'uReconstructMeridional', 'theta_m', 'qv', 'exner',
          'pressure_base', 'pressure_p', 'refl10cm'}


def bits(a):
    return cp.asnumpy(a).view(np.uint32)


rows = []
for name in sorted(os.listdir(CAPTURE)):
    if not name.startswith('compute-'):
        continue
    folder = os.path.join(CAPTURE, name)
    meta = json.load(open(os.path.join(folder, 'meta.json')))
    arrays = {n: np.load(os.path.join(folder, f'in_{n}.npy')) for n in meta['arrays']}
    before = np.load(os.path.join(folder, 'outputs_before.npy'))
    after = np.load(os.path.join(folder, 'outputs_after.npy'))
    included = [f for i, f in enumerate(new.FIELDS) if meta['update_mask'] >> i & 1]
    ns, nl = meta['n_solve'], meta['n_levels']
    inputs = {n: cp.asarray(np.ascontiguousarray(a.T)) if n in COLUMN or n == 'vorticity' else cp.asarray(a)
              for n, a in arrays.items()}

    def run(budget):
        out_rows = [cp.asarray(before[i]).copy() for i in range(len(new.FIELDS))]
        d = new.ConvectiveDiagnostics(out_rows, ns, nl, included)
        if budget is not None:
            d.COMPUTE_WORK_BYTES = budget
        kw = {'rvord': meta['rvord']} if 'rvord' in meta else {}
        d.compute(inputs, meta['fields'], **kw)
        cp.cuda.Device().synchronize()
        rows_ = 17*nl + 3 + 2*d.CAPE_SUBSTEPS
        launches = -(-ns // min(ns, max(32, d.COMPUTE_WORK_BYTES // (4*rows_) // 32 * 32)))
        return cp.stack(out_rows), launches

    one, l1 = run(None)
    split, l2 = run(4 * (17*nl + 3 + 2*new.ConvectiveDiagnostics.CAPE_SUBSTEPS) * COLUMNS)
    row = dict(call=name, numerics=numerics.mode(), words=int(one.size), launches_default=l1, launches_split=l2,
               split_vs_one=int((bits(split) != bits(one)).sum()),
               one_vs_run=int((bits(one) != after.view(np.uint32)).sum()),
               split_vs_run=int((bits(split) != after.view(np.uint32)).sum()))
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump(rows, open(OUT, 'w'), indent=1)
