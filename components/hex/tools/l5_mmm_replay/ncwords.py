#!/usr/bin/env python3
"""Word-for-word comparison of every variable in matching history/diag files of two hex forecast dirs.

Usage: ncwords.py DIR_A DIR_B OUT.json [--files GLOB]   (no masking, no scaling: raw stored words)
"""
import glob
import json
import os
import sys

import numpy as np
import netCDF4

a_dir, b_dir, out = sys.argv[1:4]
pattern = sys.argv[sys.argv.index("--files") + 1] if "--files" in sys.argv else "*.nc"
names = sorted({os.path.basename(p) for p in glob.glob(os.path.join(a_dir, pattern))}
               & {os.path.basename(p) for p in glob.glob(os.path.join(b_dir, pattern))})
rows, words, different = [], 0, 0
for name in names:
    da, db = netCDF4.Dataset(os.path.join(a_dir, name)), netCDF4.Dataset(os.path.join(b_dir, name))
    da.set_auto_maskandscale(False)
    db.set_auto_maskandscale(False)
    fields = []
    for var in sorted(set(da.variables) | set(db.variables)):
        if var not in da.variables or var not in db.variables:
            fields.append(dict(name=var, missing=True))
            continue
        x, y = np.asarray(da.variables[var][...]), np.asarray(db.variables[var][...])
        if x.dtype.kind in "SUO" or x.dtype != y.dtype or x.shape != y.shape:
            same = x.shape == y.shape and bool(np.all(x == y))
            fields.append(dict(name=var, words=int(x.size), different=0 if same else int(x.size), kind="other"))
        else:
            vx = np.ascontiguousarray(x).view(np.dtype(f"u{x.dtype.itemsize}"))
            vy = np.ascontiguousarray(y).view(np.dtype(f"u{y.dtype.itemsize}"))
            fields.append(dict(name=var, words=int(x.size), different=int(np.count_nonzero(vx != vy))))
    w = sum(f.get("words", 0) for f in fields)
    d = sum(f.get("different", 0) for f in fields) + sum(1 for f in fields if f.get("missing"))
    rows.append(dict(file=name, words=w, different=d,
                     changed=[f["name"] for f in fields if f.get("different") or f.get("missing")]))
    words += w
    different += d
json.dump(dict(a=a_dir, b=b_dir, files=len(rows), words=words, different=different, rows=rows), open(out, "w"),
          indent=1)
print(f"{len(rows)} files, {words} words, {different} different")
for r in rows:
    if r["different"]:
        print(r["file"], r["different"], r["changed"][:12])
