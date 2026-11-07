#!/usr/bin/env python3
"""Compare two NTU output files (oracle layout) field by field in FP32 ULP.

usage: compare.py BATCH_JSON BATCH_NAME A.bin B.bin [--json out.json] [--cols]
"""
import argparse, json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from woof.core.fp32_ulp import fp32_ulp_distance, MISMATCH

OUT = "TH QV QC QR QI QS QG QH NC NR NI NS NG NH QDCN QTCN QCCN QRCN QNIN FI FS VI VS VG AI AS AG AH I3M".split()
S2 = "RAINNC RAINNCV SNOWNC SNOWNCV GRAPNC GRAPNCV HAILNC HAILNCV SR".split()
INIT = "QDCN QTCN QCCN QRCN QNIN".split()

def read(path, m):
    nz = m["nz"]; nc = len(m["columns"]); st = m["steps"]
    a = np.fromfile(path, dtype="<f4"); off = 5 * nz * nc
    init = a[:off].reshape(5, nz, nc)
    steps = []
    for s in range(st):
        f3 = a[off:off + 29 * nz * nc].reshape(29, nz, nc); off += 29 * nz * nc
        f2 = a[off:off + 9 * nc].reshape(9, nc); off += 9 * nc
        steps.append((f3, f2))
    assert off == a.size, (path, off, a.size)
    return init, steps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meta"); ap.add_argument("batch"); ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--json"); ap.add_argument("--cols", action="store_true")
    args = ap.parse_args()
    m = json.load(open(args.meta))[args.batch]
    ia, sa = read(args.a, m); ib, sb = read(args.b, m)
    res = {"batch": args.batch, "columns": len(m["columns"]), "fields": {}}
    worst = 0
    bit_mismatches = 0
    nonfinite = 0
    def rec(name, x, y):
        nonlocal worst, bit_mismatches, nonfinite
        d = fp32_ulp_distance(x, y)
        mx = int(d.max()) if d.size else 0
        nd = int((d > 0).sum())
        bits = int(np.count_nonzero(x.view(np.uint32) != y.view(np.uint32)))
        nf = int(np.count_nonzero(~np.isfinite(y)))
        bit_mismatches += bits
        nonfinite += nf
        finite = np.isfinite(x) & np.isfinite(y)
        abs_error = float(np.max(np.abs(x[finite].astype(np.float64) - y[finite].astype(np.float64)))) if finite.any() else None
        sz = int(((x == y) & (np.signbit(x) != np.signbit(y))).sum())
        res["fields"][name] = {"max_ulp": mx, "ndiff": nd, "signed_zero_only": sz, "bit_mismatches": bits, "nonfinite": nf, "max_abs_error": abs_error}
        worst = max(worst, mx)
        if args.cols and nd:
            bad = np.unique(np.nonzero(d > 0)[-1])
            res["fields"][name]["columns"] = [m["columns"][c] for c in bad[:12]]
    for i, n in enumerate(INIT):
        rec(f"init.{n}", ia[i], ib[i])
    for s in range(len(sa)):
        for i, n in enumerate(OUT):
            rec(f"step{s+1}.{n}", sa[s][0][i], sb[s][0][i])
        for i, n in enumerate(S2):
            rec(f"step{s+1}.{n}", sa[s][1][i], sb[s][1][i])
    res["max_ulp"] = worst
    res["bit_mismatches"] = bit_mismatches
    res["nonfinite"] = nonfinite
    res["passed"] = bit_mismatches == 0 and nonfinite == 0
    bad = {k: v for k, v in res["fields"].items() if v["max_ulp"]}
    print(f"batch {args.batch}: {len(m['columns'])} columns, max ULP {worst if worst < MISMATCH else 'NaN-mismatch'}, "
          f"{len(bad)} of {len(res['fields'])} field-steps differ")
    for k, v in list(bad.items())[:40]:
        print("  ", k, v)
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)
    return 0 if res["passed"] else 1

if __name__ == "__main__":
    sys.exit(main())
