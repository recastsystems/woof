"""Module replay (measurement tool, not product code): the RRFS convective driver, c2efafcf vs this tree.

Usage: WOOF_HEX_NUMERICS=fast|strict python replay_conv.py OLD_TREE CAPTURE_DIR OUT_JSON
Runs every captured compute-*/update-* call through the old module (cell-major copies, one stacked output array)
and the new one (inputs in the pools' (level, cell) layout read in place, each output row its own allocation),
from the same captured outputs-before words, and compares all 41 output rows over every allocated cell word for
word (uint32 bits). Also checks the old replay reproduces the run's own outputs-after, and times each call.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import cupy as cp

OLD_TREE, CAPTURE, OUT = sys.argv[1:4]
REPEATS = 20


def load_old():
    path = os.path.join(OLD_TREE, 'src/hexcore/mpasdiag/convective_rrfs.py')
    spec = importlib.util.spec_from_file_location('hexcore.mpasdiag._convective_rrfs_c2efafcf', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


from woof.hex import numerics
from woof.hex.mpasdiag import convective_rrfs as new
old = load_old()
COLUMN = {'zgrid', 'w', 'uReconstructZonal', 'uReconstructMeridional', 'theta_m', 'qv', 'exner',
          'pressure_base', 'pressure_p', 'refl10cm'}


def timed(fn):
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    times = []
    for _ in range(REPEATS):
        start.record(); fn(); stop.record(); stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop))
    return float(np.median(times))


def bits(a):
    return cp.asnumpy(a).view(np.uint32)


rows = []
for name in sorted(os.listdir(CAPTURE)):
    folder = os.path.join(CAPTURE, name)
    meta = json.load(open(os.path.join(folder, 'meta.json')))
    arrays = {n: np.load(os.path.join(folder, f'in_{n}.npy')) for n in meta['arrays']}
    before = np.load(os.path.join(folder, 'outputs_before.npy'))
    after = np.load(os.path.join(folder, 'outputs_after.npy'))
    included = [f for i, f in enumerate(new.FIELDS) if meta['update_mask'] >> i & 1]
    ns, nl = meta['n_solve'], meta['n_levels']
    kind = name.split('-')[0]
    old_in = {n: cp.asarray(a) for n, a in arrays.items()}

    def pool_layout(packed):
        out = {}
        for n, a in arrays.items():
            if n in COLUMN:
                t = np.ascontiguousarray(a.T)            # (levels, allocated cells): the padded buffer layout
                out[n] = cp.asarray(np.ascontiguousarray(t[:, :ns]) if packed else t)  # packed: the solve extent
            elif n == 'vorticity':
                out[n] = cp.asarray(np.ascontiguousarray(a.T))
            else:
                out[n] = cp.asarray(a)
        return out

    def run_old():
        outputs = cp.asarray(before)
        d = old.ConvectiveDiagnostics(outputs, ns, nl, included)
        if kind == 'compute':
            call = lambda: d.compute(old_in, meta['fields'], rvord=meta['rvord']) if 'rvord' in meta else d.compute(old_in, meta['fields'])
        else:
            call = lambda: d.update(old_in, nsteps=meta['nsteps'], convection=meta['convection'], n_mesh_cells=meta['n_mesh_cells'])
        call(); cp.cuda.Device().synchronize()
        result = outputs.copy()
        def again():
            outputs[...] = cp.asarray(before); call()
        return result, timed(again)

    def run_new(packed):
        inputs = pool_layout(packed)
        out_rows = [cp.asarray(before[i]).copy() for i in range(len(new.FIELDS))]
        d = new.ConvectiveDiagnostics(out_rows, ns, nl, included)
        if kind == 'compute':
            call = lambda: d.compute(inputs, meta['fields'], rvord=meta['rvord']) if 'rvord' in meta else d.compute(inputs, meta['fields'])
        else:
            call = lambda: d.update(inputs, nsteps=meta['nsteps'], convection=meta['convection'], n_mesh_cells=meta['n_mesh_cells'])
        call(); cp.cuda.Device().synchronize()
        result = cp.stack(out_rows)
        base = [cp.asarray(before[i]) for i in range(len(new.FIELDS))]
        def again():
            for r, b in zip(out_rows, base): r[...] = b
            call()
        return result, timed(again)

    a, t_old = run_old()
    b, t_new = run_new(False)
    c, t_packed = run_new(True)
    a2, _ = run_old()
    row = dict(call=name, numerics=numerics.mode(), words=int(a.size),
               old_vs_run=int((bits(a) != after.view(np.uint32)).sum()),
               old_repeat=int((bits(a) != bits(a2)).sum()),
               new_vs_old=int((bits(b) != bits(a)).sum()),
               new_packed_vs_old=int((bits(c) != bits(a)).sum()),
               rows_changed_by_call=int(((bits(a) != before.view(np.uint32)).any(axis=1)).sum()),
               old_ms=round(t_old, 3), new_ms=round(t_new, 3), new_packed_ms=round(t_packed, 3))
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump(rows, open(OUT, 'w'), indent=1)
