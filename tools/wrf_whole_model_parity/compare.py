"""Compare whole-model runs frame by frame: WOOF (or a second WRF build) against stock WRF.

usage: wm_compare.py REF_DIR CAND_DIR OUT.json [--label L]

REF_DIR holds WRF's wrfout_d01_* (any number of frames per file); CAND_DIR
holds wrfout_d01_* from WOOF or another WRF build.  Frames are matched by
their Times string.  For every variable present on both sides with the same
shape and a floating dtype, per frame: differing words, max ULP (binary32
ordered-integer distance), max |diff|, RMS diff over the whole array and RMS
over the interior (5-cell lateral rim excluded, the specified boundary
zone), plus the reference RMS of the field for scale.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from netCDF4 import Dataset

RIM = 5


def frames(directory):
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "wrfout_d01_*"))):
        ds = Dataset(path)
        ds.set_auto_maskandscale(False)
        times = ["".join(t.astype(str)).strip() for t in ds.variables["Times"][:]]
        for i, t in enumerate(times):
            out[t] = (path, i)
        ds.close()
    return out


def ordered(a):
    i = a.view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def interior(a):
    if a.ndim >= 2 and a.shape[-1] > 2 * RIM and a.shape[-2] > 2 * RIM:
        return a[..., RIM:-RIM, RIM:-RIM]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("cand"); ap.add_argument("out")
    ap.add_argument("--label", default="")
    ap.add_argument("--fields", default="")
    args = ap.parse_args()
    rf, cf = frames(args.ref), frames(args.cand)
    common_times = sorted(set(rf) & set(cf))
    wanted = set(filter(None, args.fields.split(",")))
    result = {"ref": args.ref, "cand": args.cand, "label": args.label,
              "ref_times": sorted(rf), "cand_times": sorted(cf),
              "frames": []}
    handles = {}

    def var(side, t):
        path, i = side[t]
        if path not in handles:
            ds = Dataset(path); ds.set_auto_maskandscale(False); handles[path] = ds
        return handles[path], i

    only_ref = only_cand = None
    for t in common_times:
        dr, ir = var(rf, t); dc, ic = var(cf, t)
        names_r = {n for n, v in dr.variables.items() if v.dtype.kind == "f"}
        names_c = {n for n, v in dc.variables.items() if v.dtype.kind == "f"}
        if only_ref is None:
            only_ref = sorted(names_r - names_c); only_cand = sorted(names_c - names_r)
        row = {"time": t, "fields": {}}
        for name in sorted(names_r & names_c):
            if wanted and name not in wanted:
                continue
            vr, vc = dr.variables[name], dc.variables[name]
            if vr.dimensions[0] == "Time":
                a = np.asarray(vr[ir]); b = np.asarray(vc[ic]) if vc.dimensions[0] == "Time" else np.asarray(vc[:])
            else:
                a = np.asarray(vr[:]); b = np.asarray(vc[:])
            if a.shape != b.shape:
                row["fields"][name] = {"shape_ref": list(a.shape), "shape_cand": list(b.shape)}
                continue
            a32 = a.astype(np.float32, copy=False); b32 = b.astype(np.float32, copy=False)
            diff_words = int(np.count_nonzero(a32.view(np.int32) != b32.view(np.int32)))
            entry = {"n": int(a.size), "differ": diff_words}
            if diff_words:
                fin = np.isfinite(a32) & np.isfinite(b32)
                d = (b32.astype(np.float64) - a32.astype(np.float64))
                d = np.where(fin, d, 0.0)
                ulp = np.abs(ordered(b32) - ordered(a32))
                ulp = np.where(fin, ulp, 0)
                di = interior(d)
                entry.update(max_ulp=int(ulp.max()), max_abs=float(np.abs(d).max()),
                             rms=float(np.sqrt(np.mean(d * d))),
                             rms_interior=float(np.sqrt(np.mean(di * di))) if di.size else None,
                             nonfinite_mismatch=int(np.count_nonzero(np.isfinite(a32) != np.isfinite(b32))))
            ar = a32.astype(np.float64)
            ar = np.where(np.isfinite(ar), ar, 0.0)
            entry["ref_rms"] = float(np.sqrt(np.mean(ar * ar)))
            row["fields"][name] = entry
        result["frames"].append(row)
    incomplete = []
    if not rf or not cf:
        incomplete.append("both arms must contain history frames")
    if set(rf) != set(cf):
        incomplete.append("history clocks differ between arms")
    if any(not row["fields"] for row in result["frames"]):
        incomplete.append("a matched frame has no compared fields")
    if wanted:
        absent = sorted({name for row in result["frames"] for name in wanted - row["fields"].keys()})
        if absent:
            incomplete.append("requested fields absent: " + ", ".join(absent))
    result["complete"] = not incomplete
    result["incomplete_reasons"] = incomplete
    result["only_ref"] = only_ref or []
    result["only_cand"] = only_cand or []
    for ds in handles.values():
        ds.close()
    with open(args.out, "w") as fh:
        json.dump(result, fh)
    # one-line summary per frame
    for row in result["frames"]:
        bad = {k: v for k, v in row["fields"].items() if v.get("differ")}
        lead = sorted(bad, key=lambda k: -bad[k].get("max_ulp", 0))[:3]
        print(row["time"], f"{len(bad)}/{len(row['fields'])} fields differ",
              " ".join(f"{k}:{bad[k]['max_ulp']}ulp" for k in lead))

    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
