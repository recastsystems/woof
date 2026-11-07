#!/usr/bin/env python3
"""Run WOOF's P3 device kernels over the oracle fixtures, FORMAT.md dumps.

    python run_woof.py <fixdir> <outdir> [--arms unfused,fused,levels,sedlevels]

A fixture set whose index.json says ``"mp_physics": 52`` is the two-category
call shape (make_columns_2cat.py): the runner then passes the fixture's own
``nc`` as the PROGNOSTIC droplet number (log_predictNc, as the mp=52 driver
does by passing nc_3d), the category-2 fields, and launches the ``_c2``
kernels with the 2-category table set; its field lists come from index.json.

Calls ``woof.core.p3_device.run_p3_device`` -- the function the forecast
step (``woof.core.p3.apply``) calls -- with the wrapper's own locals
(nc and ssat zero, mp=50 specified Nc) and the tables ``device_tables()``
uploads in a run.  A fixture set whose index.json carries
``"predict_nc": true`` (make_columns_nc.py) is the mp=51 call shape: nc is
the fixture's prognostic droplet number, ``log_predictNc`` is on, and every
dump frame carries nc after the seventeen mp=50 fields.  Every arm in ``ARM_KERNELS`` can be driven; the forecast
uploads in a run.  A fixture set made with ``make_columns.py --variant 53``
(its index.json says so) runs the mp=53 call shape instead: the fixture's
own prognostic nc, the advected ice moment qzi, log_predictNc and
log_3momentIce on, and the three-moment tables.  Every arm in ``ARM_KERNELS`` can be driven; the forecast
selects ``CONFIG_ARM['cuda']``.

Strict build: run with GPUWM_WRF_EXACT=1 in the environment; woof's import
hook then rewrites every NVRTC compile to the strict options, and this
script writes the compile receipt next to the dumps so the run proves which
options actually compiled the kernel.
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

POISON = np.float32(-8.888e30)
IN3D = ["th", "qv", "qc", "qr", "nr", "qi", "ni", "qir", "qib",
        "th_old", "qv_old", "pii", "p", "dz", "w"]
IN2D = ["rainnc", "rainncv", "sr", "snownc", "snowncv"]
OUT3D = ["th", "qv", "qc", "qr", "nr", "qi", "ni", "qir", "qib",
         "th_old", "qv_old", "zdbz", "effc", "effi", "vmi", "di", "rhopo"]
OUT2D = ["rainnc", "rainncv", "sr", "snownc", "snowncv"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixdir")
    ap.add_argument("outdir")
    ap.add_argument("--arms", default="unfused,fused,levels,sedlevels")
    ap.add_argument("--cases", default="")
    ap.add_argument("--native-gnu239", action="store_true",
                    help="pinned WRF4.7.1 GNU13/glibc2.39 strict reference")
    ap.add_argument("--plant-intercategory-bit", action="store_true")
    ap.add_argument("--plant-mu-bit", action="store_true")
    ap.add_argument("--preport", action="store_true",
                    help="drive an engine WITHOUT the three-moment port over "
                         "an mp=53 fixture set (prognostic nc only, qzi left "
                         "as read): the gate's failing baseline")
    args = ap.parse_args()

    import cupy as cp

    import woof
    from woof.core import p3_device as PD
    from woof import wrf_exact

    fixdir = pathlib.Path(args.fixdir)
    outroot = pathlib.Path(args.outdir)
    index = json.loads((fixdir / "index.json").read_text())
    n_cat = int(index.get("n_cat", 1))
    variant = int(index.get("variant", index.get("mp_physics", 50)))
    in3d = list(index.get("in3d", IN3D))
    out3d = list(index.get("out3d", OUT3D))
    three_moment = variant == 53
    predict_nc = n_cat > 1 or bool(index.get("predict_nc", False)) or three_moment
    active_three = three_moment and not args.preport
    tables = PD.device_tables(n_cat=n_cat, three_moment=active_three,
                              native_gnu239=args.native_gnu239)
    src = PD.p3_source(native_gnu239=args.native_gnu239)
    if args.plant_intercategory_bit:
        if n_cat != 2:
            raise ValueError("the inter-category mutation requires n_cat=2")
        term = "qicol[cc][c] = fminf(qicol[cc][c], AT(qi2, k) * odt);"
        if src.count(term) != 1:
            raise ValueError("the planted-bit term must occur exactly once")
        src = src.replace(term, term + "\nqicol[cc][c] = __uint_as_float(__float_as_uint(qicol[cc][c]) ^ 1u);")
    if args.plant_mu_bit:
        if not active_three:
            raise ValueError("the mu coefficient mutation requires mp_physics=53")
        # Flip one stored coefficient bit in the closure, before the module
        # is compiled. Output buffers and the comparison gate stay unchanged.
        term = "3.3638e-3f"
        if src.count(term) != 1:
            raise ValueError("the planted mu coefficient must occur exactly once")
        value = np.asarray(np.float32(3.3638e-3)).view(np.uint32).item() ^ 1
        src = src.replace(term, f"__uint_as_float({value}u)")
    module = PD.p3_module(PD.DEFAULT_OPTIONS,
                          source=src if args.plant_intercategory_bit or args.plant_mu_bit else None,
                          native_gnu239=args.native_gnu239)
    want = [w for w in args.cases.split(",") if w]

    receipt = {
        "gpuwm_file": woof.__file__,
        "GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", ""),
        "wrf_exact_enabled": wrf_exact.ENABLED,
        "requested_options": list(PD.DEFAULT_OPTIONS),
        "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
        "p3_cu_sha256": hashlib.sha256(
            pathlib.Path(PD._P3_SOURCE).read_bytes()).hexdigest(),
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "compute_capability": "".join(
            str(x) for x in cp.cuda.Device().compute_capability),
        "cupy": cp.__version__,
        "nvrtc": ".".join(str(x) for x in cp.cuda.nvrtc.getVersion()),
        "arm_for_forecast": PD.CONFIG_ARM["cuda"],
        "log_predictNc": predict_nc,
        "n_cat": n_cat,
        "planted_intercategory_bit": args.plant_intercategory_bit,
        "planted_mu_bit": args.plant_mu_bit,
        "variant": variant,
        "native_gnu239": args.native_gnu239,
        "cases": {},
    }
    for arm in args.arms.split(","):
        outdir = outroot / arm
        outdir.mkdir(parents=True, exist_ok=True)
        for case in index["cases"]:
            name = case["name"]
            if want and not any(name.startswith(w) for w in want):
                continue
            ncol, nk = int(case["ncol"]), int(case["nk"])
            nsteps, dt = int(case["nsteps"]), float(case["dt"])
            buf = np.fromfile(fixdir / (name + ".bin"), dtype="<f4")
            assert buf.size == len(in3d) * ncol * nk + len(IN2D) * ncol
            h3, off = {}, 0
            for f in in3d:          # (ncol,nk) Fortran == (nk,ncol) C order
                h3[f] = buf[off:off + ncol * nk].reshape((nk, ncol)).copy()
                off += ncol * nk
            h2 = {}
            for f in IN2D:
                h2[f] = buf[off:off + ncol].copy()
                off += ncol
            if n_cat == 1:
                fields = {f: cp.asarray(h3[f]) for f in
                          ("qc", "qr", "nr", "qi", "qir", "ni", "qib", "th",
                           "qv", "th_old", "qv_old")}
                fields["nc"] = cp.zeros((nk, ncol), dtype=cp.float32)  # :847
            else:
                fields = {f: cp.asarray(h3[f]) for f in
                          ("qc", "qr", "nr", "nc", "th", "qv", "th_old",
                           "qv_old")}
                for c, sfx in ((1, ""), (2, "2")):
                    for f in ("qi", "qir", "ni", "qib"):
                        fields[f + sfx] = cp.asarray(h3[f + str(c)])
            fields["pres"] = cp.asarray(h3["p"])
            fields["dz"] = cp.asarray(h3["dz"])
            if predict_nc:                                           # :844
                fields["nc"] = cp.asarray(h3["nc"])
            else:                                                    # :847
                fields["nc"] = cp.zeros((nk, ncol), dtype=cp.float32)
            if three_moment:
                fields["qzi"] = cp.asarray(h3["qzi"])
            fields["ssat"] = cp.zeros((nk, ncol), dtype=cp.float32)  # :851
            diag = {d: cp.full((nk, ncol), POISON, dtype=cp.float32)
                    for d in PD.diag_slots(n_cat)}
            # the dump's names for the category fields
            alias = {"qi1": "qi", "qir1": "qir", "ni1": "ni", "qib1": "qib"}
            surf = {"prt_liq": cp.zeros(ncol, dtype=cp.float32),
                    "prt_sol": cp.zeros(ncol, dtype=cp.float32)}
            for s in OUT2D:
                surf[s] = cp.asarray(h2[s])
            ws = (PD.make_workspace(ncol, nk, three_moment=True)
                  if three_moment and not args.preport
                  else PD.make_workspace(ncol, nk))
            frames = []

            def snap():
                parts = []
                for f in out3d:
                    g = alias.get(f, f) if n_cat > 1 else f
                    parts.append(cp.asnumpy(fields[g] if g in fields
                                            else diag[g]).ravel())
                parts += [cp.asnumpy(surf[s]).ravel() for s in OUT2D]
                frames.append(np.concatenate(parts).astype("<f4"))

            snap()
            t0 = time.perf_counter()
            for it in range(1, nsteps + 1):
                PD.run_p3_device(fields, diag, surf, workspace=ws,
                                 tables=tables, dt=dt,
                                 it=int(case.get("native_itimestep", 1)) + it - 1, arm=arm,
                                 module=module, n_cat=n_cat,
                                 log_predictNc=predict_nc,
                                 log_3momentIce=active_three)
                cp.cuda.Stream.null.synchronize()
                snap()
            path = outdir / (name + ".ref")
            np.concatenate(frames).tofile(path)
            receipt["cases"][f"{arm}/{name}"] = {
                "seconds": time.perf_counter() - t0,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            print(f"{arm:9s} {name}: {nsteps} steps", flush=True)
    receipt["compile"] = wrf_exact.compile_receipt()
    (outroot / "receipt.json").write_text(json.dumps(receipt, indent=1,
                                                     sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
