"""Compare resident RRFS modules on captured words, including tiled independent columns.

Usage: python replay_current.py BASE_MODULE CAPTURE OUT_JSON [--tile 40] [--repeats 9]
Capture with capture_conv.py (layout=pool). All 41 rows and allocated words are checked.
CUDA-event times exclude restoration and copies, after one warm call per driver.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import cupy as cp
from woof.hex import numerics
from woof.hex.mpasdiag import convective_rrfs as new

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('base', type=Path)
p.add_argument('capture', type=Path)
p.add_argument('out', type=Path)
p.add_argument('--tile', type=int, default=40)
p.add_argument('--repeats', type=int, default=9)
a = p.parse_args()
spec = importlib.util.spec_from_file_location('hexcore.mpasdiag._convective_base', a.base)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)


def bits(rows):
    return cp.asnumpy(cp.stack(rows)).view(np.uint32)


def run(module, arrays, before, meta, kind, *, budget=None, cap=None):
    base = cp.asarray(before)
    rows = [base[i].copy() for i in range(len(new.FIELDS))]
    included = [f for i, f in enumerate(new.FIELDS) if meta['update_mask'] >> i & 1]
    d = module.ConvectiveDiagnostics(rows, meta['n_solve'], meta['n_levels'], included)
    if cap is not None:
        d.CAPE_SUBSTEPS = cap
    if budget is not None:
        d.COMPUTE_WORK_BYTES = budget
    if kind == 'compute':
        kw = {'rvord': meta['rvord']} if 'rvord' in meta else {}
        call = lambda: d.compute(arrays, meta['fields'], **kw)
    else:
        call = lambda: d.update(arrays, nsteps=meta['nsteps'], convection=meta['convection'],
                                n_mesh_cells=meta['n_mesh_cells'])
    call()
    result = bits(rows)
    times = []
    start, end = cp.cuda.Event(), cp.cuda.Event()
    for _ in range(a.repeats):
        for i, row in enumerate(rows):
            row[...] = base[i]
        start.record()
        call()
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))
    repeat_diff = int(np.count_nonzero(result != bits(rows)))
    return result, float(np.median(times)), repeat_diff


receipt = []
for folder in sorted(a.capture.iterdir()):
    if not folder.is_dir():
        continue
    meta = json.loads((folder/'meta.json').read_text())
    assert meta['layout'] == 'pool'
    arrays = {n: np.load(folder/f'in_{n}.npy') for n in meta['arrays']}
    before = np.load(folder/'outputs_before.npy')
    captured = np.load(folder/'outputs_after.npy').view(np.uint32)
    kind = folder.name.split('-')[0]
    inputs = {n: cp.asarray(v) for n, v in arrays.items()}
    baseline, t0, r0 = run(old, inputs, before, meta, kind)
    result, t1, r1 = run(new, inputs, before, meta, kind)
    row = dict(call=folder.name, numerics=numerics.mode(), columns=meta['n_solve'],
               words=int(result.size), different=int(np.count_nonzero(result != baseline)),
               base_vs_capture=int(np.count_nonzero(baseline != captured)),
               repeats_different=r0+r1, base_ms=t0, new_ms=t1)
    if kind == 'compute':
        budget = 4*(17*meta['n_levels']+3+2*new.ConvectiveDiagnostics.CAPE_SUBSTEPS)*600
        split, _, r2 = run(new, inputs, before, meta, kind, budget=budget)
        row['split_different'] = int(np.count_nonzero(split != result))
        row['split_repeats_different'] = r2
        row['launches_default'] = 1
        row['launches_split'] = -(-meta['n_solve'] // (600//32*32))
        # Exercise the fallback beyond the prepared path and its last valid read.
        short_base, _, _ = run(old, inputs, before, meta, kind, cap=8)
        short_new, _, _ = run(new, inputs, before, meta, kind, cap=8)
        row['short_path_different'] = int(np.count_nonzero(short_base != short_new))
        row['short_path_vs_default'] = int(np.count_nonzero(short_new != result))
    receipt.append(row)
    print(json.dumps(row), flush=True)
    if kind == 'compute' and a.tile > 1:
        ns = meta['n_solve']
        tiled_inputs = {n: cp.asarray(np.tile(v[:, :ns], (1, a.tile))) for n, v in arrays.items()}
        tiled_before = np.concatenate((np.tile(before[:, :ns], (1, a.tile)), before[:, ns:]), axis=1)
        tiled_meta = dict(meta, n_solve=ns*a.tile)
        expected = np.concatenate((np.tile(result[:, :ns], (1, a.tile)), result[:, ns:]), axis=1)
        tb, tt0, tr0 = run(old, tiled_inputs, tiled_before, tiled_meta, kind)
        tn, tt1, tr1 = run(new, tiled_inputs, tiled_before, tiled_meta, kind)
        row = dict(call=folder.name+'-tiled', numerics=numerics.mode(), columns=ns*a.tile,
                   words=int(tn.size), different=int(np.count_nonzero(tb != tn)),
                   tiled_vs_small=int(np.count_nonzero(tn != expected)), repeats_different=tr0+tr1,
                   base_ms=tt0, new_ms=tt1, base_ms_per_100k=tt0*100000/(ns*a.tile),
                   new_ms_per_100k=tt1*100000/(ns*a.tile))
        receipt.append(row)
        print(json.dumps(row), flush=True)
        del tiled_inputs
    del inputs
    cp.get_default_memory_pool().free_all_blocks()
a.out.write_text(json.dumps(receipt, indent=2)+'\n')
assert all(r['different'] == r['repeats_different'] == r.get('split_different', 0)
           == r.get('split_repeats_different', 0) == r.get('tiled_vs_small', 0)
           == r.get('short_path_different', 0) == r.get('short_path_vs_default', 0) == 0 for r in receipt)
