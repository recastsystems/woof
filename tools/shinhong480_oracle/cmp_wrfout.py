"""Word-for-word compare of two wrfout files: every common variable, every frame."""
import json, sys
import numpy as np
from netCDF4 import Dataset

a, b, out = sys.argv[1], sys.argv[2], sys.argv[3]
A, B = Dataset(a), Dataset(b)
for d in (A, B):
    d.set_auto_maskandscale(False)
va, vb = set(A.variables), set(B.variables)
rows, total, diff = {}, 0, 0
for name in sorted(va & vb):
    x, y = A.variables[name][:], B.variables[name][:]
    if x.shape != y.shape or x.dtype != y.dtype:
        rows[name] = dict(error=f"shape/dtype {x.shape}{x.dtype} vs {y.shape}{y.dtype}")
        continue
    if x.dtype.kind == "S":
        n = x.size; d = int((x != y).sum())
    else:
        xv = np.ascontiguousarray(x).view(np.uint8).reshape(-1, x.dtype.itemsize)
        yv = np.ascontiguousarray(y).view(np.uint8).reshape(-1, y.dtype.itemsize)
        n = xv.shape[0]; d = int((xv != yv).any(axis=1).sum())
    first = None
    if d and x.dtype.kind == "f" and x.ndim >= 1:
        bad = np.nonzero((x != y) & ~(np.isnan(x) & np.isnan(y)))
        if bad[0].size:
            first = int(bad[0][0])
    rows[name] = dict(words=n, different=d, first_frame=first)
    total += n; diff += d
res = dict(a=a, b=b, frames=int(A.dimensions["Time"].size), variables=len(rows),
           only_a=sorted(va - vb), only_b=sorted(vb - va), words=total, different=diff,
           differing={k: v for k, v in rows.items() if v.get("different") or v.get("error")})
open(out, "w").write(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: res[k] for k in ("frames", "variables", "words", "different", "only_a", "only_b")}))
print("differing:", json.dumps(res["differing"])[:3000])
