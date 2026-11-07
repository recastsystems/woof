#!/usr/bin/env python3
"""Run WOOF's Goddard GCE kernels over the oracle fixtures.

    python run_woof.py <fixdir> <outdir> [--cases PREFIX,...] [--plant]

Drives ``woof.core.mp_column_schemes.launch_column`` and ``launch_refl``
-- the two calls the forecast step's ``apply`` makes -- with the fixture
columns laid out as one j row (nz, 1, ncol), the step count, time step and
switches of each case, and writes the run_gsfcgce.F90 dump layout so
compare.py reads both sides the same way.

Strict arithmetic: run with GPUWM_WRF_EXACT=1; woof's import hook then
rewrites every NVRTC compile to the strict options, and the compile receipt
written beside the dumps proves which options compiled the kernels.

``--plant`` compiles the column kernel with ONE bit flipped in one process
coefficient (the lowest mantissa bit of rn18a, the rain-freezing exponent
of pgfr) -- the negative control: compare.py must report differences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

IN3D = ["th", "qv", "qc", "qr", "qi", "qs", "qg", "rho", "pii", "p", "z",
        "dz8w"]
IN2D = ["ht", "rainnc", "snownc", "graupelnc"]
OUT3D = ["th", "qv", "qc", "qr", "qi", "qs", "qg", "refl"]
OUT2D = ["rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
         "graupelncv", "sr"]
PLANT_OLD = "y2 = gfk_exp(C[S_RN18A] * (t0 - tair));"
PLANT_NEW = ("y2 = gfk_exp(__int_as_float(__float_as_int(C[S_RN18A]) ^ 1)"
             " * (t0 - tair));")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixdir")
    ap.add_argument("outdir")
    ap.add_argument("--cases", default="")
    ap.add_argument("--plant", action="store_true")
    a = ap.parse_args()

    import cupy as cp

    import woof
    from woof import wrf_exact
    from woof.core import kernels as K
    from woof.core import mp_column_schemes as M

    if a.plant:
        original = K._unit_text

        def planted(name, kernel_dir=K._KDIR):
            text = original(name, kernel_dir)
            if name == "gsfcgce":
                if text.count(PLANT_OLD) != 1:
                    raise SystemExit("plant anchor not found exactly once")
                text = text.replace(PLANT_OLD, PLANT_NEW)
            return text
        K._unit_text = planted

    row = M.MP_COLUMN_SCHEMES[97]
    fix = pathlib.Path(a.fixdir)
    out = pathlib.Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((fix / "index.json").read_text())
    want = [w for w in a.cases.split(",") if w]
    src = K.module_source("gsfcgce")
    receipt = {
        "gpuwm_file": woof.__file__,
        "GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", ""),
        "wrf_exact_enabled": wrf_exact.ENABLED,
        "planted": a.plant,
        "gsfcgce_cu_sha256": hashlib.sha256(
            (K._KDIR / "gsfcgce.cu").read_bytes()).hexdigest(),
        "module_source_sha256": hashlib.sha256(src.encode()).hexdigest(),
        "module_options": list(K.module_options("gsfcgce")),
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cupy": cp.__version__,
        "nvrtc": ".".join(str(x) for x in cp.cuda.nvrtc.getVersion()),
        "cases": {},
    }
    f32 = np.float32
    for case in index["cases"]:
        name = case["name"]
        if want and not any(name.startswith(w) for w in want):
            continue
        ncol, nk, ns, dt = case["ncol"], case["nk"], case["nsteps"], case["dt"]
        values = (int(case["ihail"]), int(case["ice2"]))
        buf = np.fromfile(fix / case["input"], dtype="<f4")
        assert buf.size == len(IN3D) * ncol * nk + len(IN2D) * ncol
        h, o = {}, 0
        for f in IN3D:
            h[f] = buf[o:o + ncol * nk].reshape(nk, 1, ncol).copy()
            o += ncol * nk
        for f in IN2D:
            h[f] = buf[o:o + ncol].reshape(1, ncol).copy()
            o += ncol
        d = {f: cp.asarray(v) for f, v in h.items()}
        for f in ("rainncv", "snowncv", "graupelncv", "sr"):
            d[f] = cp.zeros((1, ncol), dtype=cp.float32)
        d["refl"] = cp.zeros((nk, 1, ncol), dtype=cp.float32)
        frames = []

        def snap():
            parts = [cp.asnumpy(d[f]).ravel() for f in OUT3D]
            parts += [cp.asnumpy(d[f]).ravel() for f in OUT2D]
            frames.append(np.concatenate(parts).astype("<f4"))

        snap()
        t0 = time.perf_counter()
        for _ in range(ns):
            M.launch_column(row, d["th"], d, d["rho"], d["pii"], d["p"],
                            d["z"], d["dz8w"], d["ht"], d["rainnc"],
                            d["rainncv"], d["snownc"], d["snowncv"],
                            d["graupelnc"], d["graupelncv"], d["sr"],
                            f32(dt), values)
            M.launch_refl(row, d["th"], d["pii"], d["p"], d["qv"], d["qr"],
                          d["qs"], d["qg"], d["refl"], values)
            cp.cuda.Stream.null.synchronize()
            snap()
        path = out / f"{name}.ref"
        np.concatenate(frames).tofile(path)
        receipt["cases"][name] = {
            "seconds": time.perf_counter() - t0,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"{name}: {ns} steps", flush=True)
    receipt["frames"] = {
        "column": {"local_size_bytes": cp.cuda.driver.funcGetAttribute(3, K.get_kernel(row.unit, row.entry).ptr)},
        "reflectivity_256": {"local_size_bytes": cp.cuda.driver.funcGetAttribute(3, K.get_kernel(row.refl_unit, row.refl_entry).ptr)}}
    receipt["compile"] = wrf_exact.compile_receipt()
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1,
                                                 sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
