"""Word-for-word comparison of two WRF history files (every variable, every frame).

    python3 tools/wrf480_capture/compare_history.py A.nc B.nc [--json OUT]

Used to prove the capture copy is inert: the instrumented wrf.exe's history must
equal the pristine referee's history in every stored word.  Values are compared as
raw bytes (so NaN payloads and signed zeros count), per variable and per frame.
"""
from __future__ import annotations

import json
import sys

import netCDF4
import numpy as np


def compare(a_path, b_path):
    a, b = netCDF4.Dataset(a_path), netCDF4.Dataset(b_path)
    for ds in (a, b):
        ds.set_auto_maskandscale(False)
    names_a, names_b = set(a.variables), set(b.variables)
    out = {"a": str(a_path), "b": str(b_path), "only_a": sorted(names_a - names_b),
           "only_b": sorted(names_b - names_a), "variables": 0, "words": 0,
           "different_words": 0, "differing": {}}
    for name in sorted(names_a & names_b):
        va, vb = a.variables[name], b.variables[name]
        if va.shape != vb.shape or va.dtype != vb.dtype:
            out["differing"][name] = {"reason": f"shape/dtype {va.shape} {va.dtype} vs {vb.shape} {vb.dtype}"}
            continue
        xa, xb = np.asarray(va[...]), np.asarray(vb[...])
        if xa.dtype.kind in "fiu":
            ua = xa.view(f"u{xa.dtype.itemsize}") if xa.dtype.itemsize in (1, 2, 4, 8) else xa
            ub = xb.view(f"u{xb.dtype.itemsize}") if xb.dtype.itemsize in (1, 2, 4, 8) else xb
            diff = int(np.count_nonzero(ua != ub))
        else:
            diff = int(np.count_nonzero(xa != xb))
        out["variables"] += 1
        out["words"] += int(xa.size)
        out["different_words"] += diff
        if diff:
            out["differing"][name] = {"words": diff, "of": int(xa.size)}
    out["frames"] = int(len(a.dimensions["Time"])) if "Time" in a.dimensions else None
    return out


def main(argv):
    res = compare(argv[1], argv[2])
    if "--json" in argv:
        with open(argv[argv.index("--json") + 1], "w") as fh:
            json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "differing"}),
          "differing:", list(res["differing"])[:20])
    return 0 if not res["differing"] and not res["only_a"] and not res["only_b"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
