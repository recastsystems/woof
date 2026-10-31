"""Word-for-word comparison of every netCDF variable two runs (or a run and native) both write.
Usage: compare_runs.py A_DIR B_DIR PATTERN... (e.g. 'history.*' 'diag.*'); prints per-file totals as JSON."""
import glob
import json
import os
import sys

import netCDF4
import numpy as np


def compare(path_a, path_b):
    a, b = netCDF4.Dataset(path_a), netCDF4.Dataset(path_b)
    for d in (a, b):
        d.set_auto_maskandscale(False)
    words = different = 0
    fields = []
    for name in a.variables:
        if name not in b.variables:
            continue
        x, y = a.variables[name][...], b.variables[name][...]
        if x.shape != y.shape:
            fields.append([name, "shape", list(x.shape), list(y.shape)])
            continue
        if x.dtype.kind in "fiu":
            wx = np.ascontiguousarray(x).view(f"u{x.dtype.itemsize}")
            wy = np.ascontiguousarray(y).view(f"u{y.dtype.itemsize}")
        else:
            wx, wy = x, y
        diff = int(np.count_nonzero(wx != wy))
        words += x.size
        different += diff
        if diff:
            fields.append([name, diff, int(x.size)])
    only = sorted(set(a.variables) ^ set(b.variables))
    return {"file": os.path.basename(path_a), "variables": len(set(a.variables) & set(b.variables)),
            "words": words, "different": different, "differing_fields": fields[:20], "not_in_both": only}


if __name__ == "__main__":
    da, db = sys.argv[1:3]
    for pattern in sys.argv[3:]:
        for path in sorted(glob.glob(os.path.join(da, pattern))):
            if path.endswith(".done") or not path.endswith(".nc"):
                continue
            other = os.path.join(db, os.path.basename(path))
            if os.path.exists(other):
                print(json.dumps(compare(path, other)), flush=True)
