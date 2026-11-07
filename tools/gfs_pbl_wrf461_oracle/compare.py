"""Grade a GFS PBL implementation against the WRF v4.6.1 oracle's words.

    python compare.py --impl ref|gpu [--receipt out.json] cases-*.out

Every recorded call of every oracle file is replayed from its own input
record; each output field (the six tendencies, UST, WSPD, PBLH, KPBL) is
compared word for word.  ``ref`` is the NumPy reference
(woof.verify.gfs_pbl_ref), ``gpu`` the CUDA launcher
(woof.core.gfs_pbl.gfs_pbl_step).  Exit status 1 on any differing word.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_io import OUT_2D, OUT_3D, read_oracle, ulp_distance  # noqa: E402

FIELDS = OUT_3D + OUT_2D + ("kpbl",)


def run_ref(inputs, dt, flqi):
    from woof.core.pbl_driver_constants import pbl_driver_constants_f32
    from woof.verify.gfs_pbl_ref import bl_gfs
    out = bl_gfs(inputs, dt=dt, flqi=flqi,
                 wrf_constants=pbl_driver_constants_f32())
    return out


def run_gpu(inputs, dt, flqi, module=None):
    import cupy as cp
    from woof.core.gfs_pbl import gfs_pbl_step
    nz, ncol = inputs["u"].shape
    names = {"u": "u", "v": "v", "t": "t", "qv": "qv", "qc": "qc",
             "qi": "qi", "p": "p", "exner": "pi", "dz": "dz8w", "z": "z"}
    columns = {k: cp.asarray(inputs[v].reshape(nz, 1, ncol))
               for k, v in names.items()}
    surface = {k: cp.asarray(inputs[k].reshape(1, ncol)) for k in
               ("psfc", "psim", "psih", "hfx", "qfx", "gz1oz0", "br", "wspd")}
    out = gfs_pbl_step(columns, surface, dt=dt, flqi=flqi, module=module)
    res = {name: cp.asnumpy(out[name]).reshape(nz, ncol) for name in OUT_3D}
    res["ust"] = cp.asnumpy(out["ust"]).reshape(ncol)
    res["wspd"] = cp.asnumpy(surface["wspd"]).reshape(ncol)
    res["pbl"] = cp.asnumpy(out["pblh"]).reshape(ncol)
    res["kpbl"] = cp.asnumpy(out["kpbl"]).reshape(ncol)
    return res


def grade(paths, impl, module=None):
    report = {"impl": impl, "files": {}, "fields": {}}
    totals = {f: {"words": 0, "differ": 0, "max_ulp": 0} for f in FIELDS}
    for path in paths:
        rec = read_oracle(path)
        fstat = {"calls": rec["nstep"], "columns": rec["ncol"],
                 "nz": rec["nz"], "ice": rec["flqi"], "dt": rec["dt"],
                 "first_difference": None,
                 "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for step, call in enumerate(rec["steps"]):
            want = call["outputs"]
            if impl == "ref":
                got = run_ref(call["inputs"], rec["dt"], rec["flqi"])
            else:
                got = run_gpu(call["inputs"], rec["dt"], rec["flqi"], module)
            for name in FIELDS:
                w, g = np.asarray(want[name]), np.asarray(got[name])
                if name == "kpbl":
                    bad = w != g
                    ulp = np.abs(w.astype(np.int64) - g.astype(np.int64))
                else:
                    # Word identity, NaN payloads and signed zeros included.
                    bad = w.view(np.uint32) != g.astype(np.float32).view(np.uint32)
                    ulp = ulp_distance(w, g)
                t = totals[name]
                t["words"] += int(w.size)
                t["differ"] += int(bad.sum())
                if bad.any():
                    t["max_ulp"] = max(t["max_ulp"], int(ulp[bad].max()))
                    if fstat["first_difference"] is None:
                        idx = np.argwhere(bad)[0].tolist()
                        fstat["first_difference"] = {
                            "call": step + 1, "field": name, "index": idx,
                            "wrf": repr(w[tuple(idx)]),
                            "port": repr(g[tuple(idx)])}
        report["files"][Path(path).name] = fstat
    report["fields"] = totals
    report["differing_words"] = sum(t["differ"] for t in totals.values())
    report["words"] = sum(t["words"] for t in totals.values())
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=("ref", "gpu"), required=True)
    ap.add_argument("--receipt")
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args(argv)
    report = grade(a.paths, a.impl)
    text = json.dumps(report, indent=1)
    if a.receipt:
        Path(a.receipt).write_text(text + "\n", encoding="utf-8")
    for name, t in report["fields"].items():
        print(f"{name:9s} words {t['words']:8d} differ {t['differ']:7d} "
              f"max_ulp {t['max_ulp']}")
    for fname, f in report["files"].items():
        if f["first_difference"]:
            print(fname, "first difference", f["first_difference"])
    print(f"TOTAL {report['words']} words, {report['differing_words']} differ")
    return 1 if report["differing_words"] else 0


if __name__ == "__main__":
    sys.exit(main())
