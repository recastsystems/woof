#!/usr/bin/env python3
"""The CPU reference arm (``p3_backend = "reference"``) against a Fortran dump.

    python run_reference_arm.py <fixdir> <fortran_out_dir> <case>[,<case>...]
                                [--write <outdir>]

Drives ``woof.core.p3.mp_p3_wrapper_wrf`` (what ``apply_reference`` calls)
over the same fixture bytes the device runner reads, steps counted from 1,
and prints per step and field how many float32 words differ from the dump
and the largest ULP distance.  CPU only.  A fixture set whose index.json
says ``"predict_nc": true`` (make_columns_nc.py) runs the mp=51 call shape:
the fixture's droplet number is passed as ``nc_3d`` and compared as the
eighteenth field.  ``--write`` also saves the arm's own dumps (FORMAT.md
layout) so compare.py can put them beside the device arms.

The 2026-10-07 measurement (lane report, PHYSICS-VERIFY-2026-10-07/p3-mp50)
found this arm NOT bit-identical to WRF: its NumPy float32 libm calls were
different functions from the C library's.  Since 2026-10-08 it calls WOOF's
own float32 routines at every statement where the device kernels do.
"""
import argparse
import json
import pathlib
import time

import numpy as np

from woof.core.p3 import mp_p3_wrapper_wrf, p3_init

IN3D = ["th", "qv", "qc", "qr", "nr", "qi", "ni", "qir", "qib", "th_old",
        "qv_old", "pii", "p", "dz", "w"]
OUT3D = ["th", "qv", "qc", "qr", "nr", "qi", "ni", "qir", "qib", "th_old",
         "qv_old", "zdbz", "effc", "effi", "vmi", "di", "rhopo"]
OUT2D = ["rainnc", "rainncv", "sr", "snownc", "snowncv"]
DIAG = ["zdbz", "effc", "effi", "vmi", "di", "rhopo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixdir")
    ap.add_argument("refdir")
    ap.add_argument("cases")
    ap.add_argument("--write")
    a = ap.parse_args()
    rt = p3_init()
    index = json.load(open(f"{a.fixdir}/index.json"))
    predict_nc = bool(index.get("predict_nc", False))
    in3d = index.get("in3d", IN3D)
    out3d = index.get("out3d", OUT3D)
    idx = {c["name"]: c for c in index["cases"]}
    total_diff = 0
    for name in a.cases.split(","):
        c = idx[name]
        ncol, nk, ns, dt = c["ncol"], c["nk"], c["nsteps"], c["dt"]
        buf = np.fromfile(f"{a.fixdir}/{name}.bin", "<f4")
        h, o = {}, 0
        for n in in3d:
            h[n] = buf[o:o + ncol * nk].reshape((nk, ncol)).T.copy()
            o += ncol * nk
        s2 = [buf[o + i * ncol:o + (i + 1) * ncol].copy() for i in range(5)]
        d = {n: np.zeros((ncol, nk), np.float32) for n in DIAG}
        frames = []

        def snap():
            parts = [(h[n] if n in h else d[n]).T.ravel() for n in out3d]
            parts += list(s2)
            frames.append(np.concatenate(parts).astype("<f4"))

        snap()
        want = np.fromfile(f"{a.refdir}/{name}.ref", "<f4")
        per = want.size // (ns + 1)
        t0 = time.time()
        for it in range(1, ns + 1):
            mp_p3_wrapper_wrf(
                h["th"], h["qv"], h["qc"], h["qr"], h["nr"], h["qi"],
                h["qir"], h["ni"], h["qib"], h["th_old"], h["qv_old"],
                h["p"], h["dz"], float(dt), it, *s2, d["zdbz"], d["effc"],
                d["effi"], d["vmi"], d["di"], d["rhopo"],
                nc_3d=h["nc"] if predict_nc else None, runtime=rt)
            snap()
        allf = np.concatenate(frames)
        if a.write:
            out = pathlib.Path(a.write)
            out.mkdir(parents=True, exist_ok=True)
            allf.tofile(out / f"{name}.ref")
        got = allf[per:]
        w = want[per:]
        dif = got.view(np.uint32) != w.view(np.uint32)
        both = np.isfinite(got) & np.isfinite(w)
        ulp = np.abs(got.view(np.int32).astype(np.int64)
                     - w.view(np.int32).astype(np.int64))
        nf = len(out3d) * ncol * nk
        for st in range(ns):
            base = st * per
            for i, n in enumerate(out3d):
                sl = slice(base + i * ncol * nk, base + (i + 1) * ncol * nk)
                dd = int(dif[sl].sum())
                if dd:
                    print("  step", st + 1, n, dd, "max ulp",
                          int(ulp[sl][both[sl]].max()) if both[sl].any()
                          else -1)
            for i, n in enumerate(OUT2D):
                sl = slice(base + nf + i * ncol, base + nf + (i + 1) * ncol)
                dd = int(dif[sl].sum())
                if dd:
                    print("  step", st + 1, n, dd)
        total_diff += int(dif.sum())
        print(f"{name}: differing {int(dif.sum())} of {got.size}, max ulp "
              f"{int(ulp[both].max()) if both.any() else 0}, "
              f"{time.time() - t0:.0f}s", flush=True)
    print("TOTAL differing", total_diff)


if __name__ == "__main__":
    main()
