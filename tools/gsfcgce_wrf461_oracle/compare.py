#!/usr/bin/env python3
"""Bitwise comparison of two Goddard GCE oracle dump directories.

    python compare.py <fixdir> <ref_dir> <cand_dir> [--json out.json]
                      [--quiet] [--cases PREFIX,...]

Every frame after the input frame, every output field, every element.
"Match" means equal float32 bit patterns (NaN matches only NaN).  ULP
distance is the monotone int32 ordering exact across zero
(woof.core.fp32_ulp); a one-sided NaN is counted separately and never
folded into a ULP figure.

Fields split into the STATE set (theta, the six water species and the
seven surface accumulators/rates WRF returns) and the DIAGNOSTIC set
(REFL_10CM).  Equal state frames at every step mean equal per-step
tendencies, since both sides form the tendency from the same two frames.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from woof.core.fp32_ulp import monotone_fp32_key as ordered

OUT3D = ["th", "qv", "qc", "qr", "qi", "qs", "qg", "refl"]
OUT2D = ["rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
         "graupelncv", "sr"]
DIAGNOSTIC = {"refl"}


def read(path, ncol, nk, nsteps):
    buf = np.fromfile(path, dtype="<f4")
    per = len(OUT3D) * ncol * nk + len(OUT2D) * ncol
    if buf.size != per * (nsteps + 1):
        raise SystemExit(f"{path}: {buf.size} floats, expected "
                         f"{per * (nsteps + 1)}")
    frames = buf.reshape(nsteps + 1, per)
    out, o = {}, 0
    for f in OUT3D:
        out[f] = frames[:, o:o + ncol * nk].reshape(nsteps + 1, nk, ncol)
        o += ncol * nk
    for f in OUT2D:
        out[f] = frames[:, o:o + ncol].reshape(nsteps + 1, 1, ncol)
        o += ncol
    return out


def field_stats(r, c):
    rn, cn = np.isnan(r), np.isnan(c)
    same = r.view(np.uint32) == c.view(np.uint32)
    fin = ~(rn | cn)
    u = np.where(fin, np.abs(ordered(r) - ordered(c)), 0)
    a = np.where(fin, np.abs(r.astype(np.float64) - c.astype(np.float64)), 0)
    return same, int(np.count_nonzero(rn ^ cn)), u, a


def compare_case(case, ref_dir, cand_dir, labels):
    name, ncol, nk, ns = case["name"], case["ncol"], case["nk"], case["nsteps"]
    R = read(pathlib.Path(ref_dir) / f"{name}.ref", ncol, nk, ns)
    C = read(pathlib.Path(cand_dir) / f"{name}.ref", ncol, nk, ns)
    res = {"name": name, "fields": {}, "state_differing": 0,
           "state_elements": 0, "state_max_ulp": 0, "diag_differing": 0,
           "diag_max_ulp": 0, "diag_max_abs": 0.0, "nan_mismatch": 0,
           "first_step": None, "columns": []}
    bad_cols = np.zeros(ncol, dtype=bool)
    for f in OUT3D + OUT2D:
        r, c = R[f][1:], C[f][1:]
        same, nanmis, u, a = field_stats(r, c)
        diff = ~same
        nd = int(np.count_nonzero(diff))
        e = {"n_differing": nd, "max_ulp": int(u.max()),
             "max_abs": float(a.max()), "nan_mismatch": nanmis}
        if nd:
            steps = np.nonzero(diff.reshape(ns, -1).any(axis=1))[0]
            e["first_step"] = int(steps[0]) + 1
            s, k, i = np.unravel_index(int(np.argmax(np.where(diff, u, -1))),
                                       u.shape)
            e["worst_at"] = {"step": int(s) + 1, "k": int(k), "col": int(i),
                             "label": labels[i], "ref": float(r[s, k, i]),
                             "cand": float(c[s, k, i])}
            bad_cols |= diff.any(axis=(0, 1))
            fs = res["first_step"]
            res["first_step"] = e["first_step"] if fs is None else min(
                fs, e["first_step"])
        res["fields"][f] = e
        res["nan_mismatch"] += nanmis
        if f in DIAGNOSTIC:
            res["diag_differing"] += nd
            res["diag_max_ulp"] = max(res["diag_max_ulp"], e["max_ulp"])
            res["diag_max_abs"] = max(res["diag_max_abs"], e["max_abs"])
        else:
            res["state_differing"] += nd
            res["state_elements"] += int(r.size)
            res["state_max_ulp"] = max(res["state_max_ulp"], e["max_ulp"])
    res["columns"] = [labels[i] for i in np.nonzero(bad_cols)[0]]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixdir")
    ap.add_argument("ref_dir")
    ap.add_argument("cand_dir")
    ap.add_argument("--json")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="write measurements without accepting the candidate")
    ap.add_argument("--cases", default="")
    a = ap.parse_args()
    index = json.loads((pathlib.Path(a.fixdir) / "index.json").read_text())
    labels = index["columns"]
    want = [w for w in a.cases.split(",") if w]
    rows = []
    missing = []
    selected = []
    for case in index["cases"]:
        if want and not any(case["name"].startswith(w) for w in want):
            continue
        selected.append(case["name"])
        if not (pathlib.Path(a.cand_dir) / f"{case['name']}.ref").exists():
            print(f"{case['name']:24s} MISSING in candidate")
            missing.append(case["name"])
            continue
        if not (pathlib.Path(a.ref_dir) / f"{case['name']}.ref").exists():
            missing.append(case["name"] + ":reference")
            continue
        r = compare_case(case, a.ref_dir, a.cand_dir, labels)
        rows.append(r)
        print(f"{r['name']:24s} state_differing={r['state_differing']:7d}"
              f"/{r['state_elements']} state_max_ulp={r['state_max_ulp']}"
              f" refl_differing={r['diag_differing']}"
              f" refl_max_abs={r['diag_max_abs']:.3g}"
              f" nan={r['nan_mismatch']} first_step={r['first_step']}"
              f" columns={len(r['columns'])}")
        if not a.quiet:
            for f, e in r["fields"].items():
                if e["n_differing"]:
                    print(f"    {f:10s} n={e['n_differing']:6d} "
                          f"ulp={e['max_ulp']} abs={e['max_abs']:.3e} "
                          f"first={e.get('first_step')} "
                          f"worst={e.get('worst_at')}")
    summary = {
        "cases": len(rows),
        "selected_cases": len(selected),
        "missing_cases": missing,
        "state_bit_identical": bool(rows) and not missing and all(r["state_differing"] == 0 for r in rows),
        "state_elements": sum(r["state_elements"] for r in rows),
        "state_differing": sum(r["state_differing"] for r in rows),
        "state_max_ulp": max((r["state_max_ulp"] for r in rows), default=0),
        "refl_differing": sum(r["diag_differing"] for r in rows),
        "refl_max_abs_dbz": max((r["diag_max_abs"] for r in rows),
                                default=0.0),
        "nan_mismatch": sum(r["nan_mismatch"] for r in rows),
    }
    print("SUMMARY", json.dumps(summary))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"summary": summary, "cases": rows}, indent=1))
    if missing or not rows:
        return 2
    if a.report_only:
        return 0
    return 0 if (summary["state_bit_identical"] and
                 summary["nan_mismatch"] == 0 and
                 summary["refl_differing"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
