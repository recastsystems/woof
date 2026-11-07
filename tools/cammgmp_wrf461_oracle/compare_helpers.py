"""Small receipts for support-function differences against the WRF oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.cammgmp_wrf461_oracle.oracle_io import read_helpers

def main():
    import cupy as cp
    from woof.core.cammgmp import _module
    p = argparse.ArgumentParser()
    p.add_argument('oracle')
    p.add_argument('json')
    args = p.parse_args()
    h = read_helpers(Path(args.oracle) / 'helpers.bin')
    kernel = _module().get_function('cammgmp_helpers')
    err = cp.zeros(1, dtype=cp.int32)
    rows = {}
    for which, (name, arg) in enumerate((('estblf','t'), ('polysvp_liq','t'),
                                       ('polysvp_ice','t'), ('erf','erf_x'),
                                       ('gamma','gamma_x'))):
        x = cp.asarray(h[arg]); out = cp.empty_like(x)
        kernel(((x.size+127)//128,), (128,), (x,out,np.int32(x.size),np.int32(which),err))
        got = cp.asnumpy(out); ref = h[name]
        delta = np.abs(got-ref)
        bits = np.count_nonzero(got.view(np.uint64) != ref.view(np.uint64))
        row = dict(words=ref.size, differ=int(bits), max_abs=float(delta.max()),
                   max_rel=float(np.max(delta/np.maximum(np.abs(ref),1e-300))),
                   ref_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),
                   gpu_sha256=hashlib.sha256(got.tobytes()).hexdigest())
        rows[name] = row
        print(name, row)
    rows['status'] = int(err.get()[0])
    Path(args.json).write_text(json.dumps(rows,indent=2)+'\n')
    if rows['status']: raise SystemExit(1)

if __name__ == '__main__': main()
