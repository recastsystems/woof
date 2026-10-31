"""cmpdirs.py A B out.json: every word of every NetCDF file the two runs wrote (bit patterns); file sets must match."""
import sys, json, numpy as np, netCDF4
from pathlib import Path
a, b = Path(sys.argv[1]), Path(sys.argv[2])
fa = sorted(p.relative_to(a).as_posix() for p in a.rglob("*.nc")); fb = sorted(p.relative_to(b).as_posix() for p in b.rglob("*.nc"))
report, tot, dif = {}, 0, 0
only = sorted(set(fa) ^ set(fb))
for f in sorted(set(fa) & set(fb)):
    da, db = netCDF4.Dataset(a / f), netCDF4.Dataset(b / f)
    rows = {}
    for name in sorted(set(da.variables) | set(db.variables)):
        if name not in da.variables or name not in db.variables: rows[name] = "absent in one"; dif += 1; continue
        va, vb = da.variables[name], db.variables[name]
        va.set_auto_maskandscale(False); vb.set_auto_maskandscale(False)
        x, y = np.asarray(va[:]), np.asarray(vb[:])
        if x.dtype.kind in "SUcO":
            if x.tobytes() != y.tobytes(): rows[name] = "text differs"; dif += 1
            continue
        if x.shape != y.shape or x.dtype != y.dtype: rows[name] = f"shape {x.shape} vs {y.shape}"; dif += 1; continue
        ne = (x.view(np.uint32 if x.itemsize == 4 else np.uint64 if x.itemsize == 8 else np.uint8) !=
              y.view(np.uint32 if y.itemsize == 4 else np.uint64 if y.itemsize == 8 else np.uint8))
        n = int(ne.sum()); tot += x.size; dif += n
        if n: rows[name] = f"{n}/{x.size}"
    report[f] = rows
res = dict(a=str(a), b=str(b), files=len(report), only_in_one=only, total_words=tot, differ=dif, report={k: v for k, v in report.items() if v})
json.dump(res, open(sys.argv[3], "w"), indent=1)
print(f"{a.name} vs {b.name}: {len(report)} files ({len(only)} only in one), {dif} of {tot} words differ")
for f, rows in list(res['report'].items())[:10]: print(" ", f, "->", "; ".join(f"{k}: {v}" for k, v in list(rows.items())[:10]))
