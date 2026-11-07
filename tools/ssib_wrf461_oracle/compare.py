
"""Small per-field receipts for strict, default and Fortran evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from woof.core import ssib
from woof.core.fp32_ulp import monotone_fp32_key

def grade(out, ref):
    rows = {}
    for k, name in enumerate(ssib.STATE):
        a = np.ascontiguousarray(out[..., k])
        b = np.ascontiguousarray(ref[..., k])
        ia = monotone_fp32_key(a)
        ib = monotone_fp32_key(b)
        finite = np.isfinite(a) & np.isfinite(b)
        bits = a.view(np.uint32) != b.view(np.uint32)
        delta = np.abs(a.astype(np.float64)[finite] - b.astype(np.float64)[finite])
        rows[name] = dict(words=int(a.size), different_bits=int(bits.sum()),
            max_ulp=int(np.abs(ia - ib)[finite].max(initial=0)),
            max_abs=float(delta.max(initial=0)),
            nonfinite_actual=int((~np.isfinite(a)).sum()),
            nonfinite_reference=int((~np.isfinite(b)).sum()),
            nonfinite_mismatch=int((bits & ~finite).sum()),
            actual_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            reference_sha256=hashlib.sha256(b.tobytes()).hexdigest())
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--fixes', type=int)
    ap.add_argument('--evidence', type=Path)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    with np.load(root / 'tests/data/oracles/ssib/ssib-wrf461-columns.npz') as z:
        cols, reference = bytes(z['columns']), bytes(z['out_ref'])
        meta = json.loads(bytes(z['meta']).decode())
    shape = (meta['nstep'], meta['ncol'], len(ssib.STATE))
    ref = np.frombuffer(reference, '<f4').reshape(shape)
    args.work.mkdir(parents=True, exist_ok=True)
    columns = args.work / 'columns.bin'
    columns.write_bytes(cols)
    if args.evidence:
        actual = np.fromfile(args.evidence, '<f4').reshape(shape)
        errors, fixes, mode = 0, None, 'fortran-evidence'
    else:
        from woof import wrf_exact
        fixes = ssib.fixes_for_run() if args.fixes is None else args.fixes
        actual, errors = ssib.replay_columns(columns, fixes=fixes)
        mode = 'strict' if wrf_exact.ENABLED else 'default'
    rows = grade(actual, ref)
    receipt = dict(mode=mode, fixes=fixes, errors=errors, columns=meta['ncol'],
        steps=meta['nstep'], fields=rows,
        different_bits=sum(r['different_bits'] for r in rows.values()),
        reference='WRF 4.6.1 module_sf_ssib.F, -O0 -finit-local-zero',
        limitations=['The oracle driver transcribes the CASE arm, not the compiled surface driver.',
                    'Zero initialization defines reads that stock WRF leaves undefined.'])
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'fields'}))

if __name__ == '__main__':
    main()
