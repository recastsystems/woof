"""Step WOOF's TEMF kernels over the oracle's columns, as run_temf.F90 does.

The replay keeps its own carried state across steps -- every *_temf field,
WM_TEMF, PBLH/KPBL, HFX/QFX, UST and the 10 m / 2 m diagnostics -- exactly
as the Fortran driver keeps WRF's, and advances the column with the same
float32 words (``x + dt*rx``, a negative qv or qc set to zero), so a single
wrong word in any step shows in every later one.  It returns the outputs in
:mod:`woof.verify.temf_oracle`'s layout, ready for word comparison.
"""
from __future__ import annotations

import numpy as np

from woof.verify.temf_oracle import (FIELDS_PBL2D, FIELDS_PBL3D,
                                      FIELDS_SFC2D, FIELDS_STATE, Cases)

F32 = np.float32
TEMIN = F32(1e-3)


def replay(cases: Cases, *, verify_stock_wrf: bool = False,
           nsteps: int | None = None) -> dict:
    """``{field: (nsteps, ...) array}`` of WOOF's words for every step."""
    import cupy as cp

    from woof.core.physics_inventory import (TEMF_PBL_STATE_2D,
                                              TEMF_PBL_STATE_3D)
    from woof.core.temf import launch_temf_pbl, launch_temf_sfclay

    ncol, nz = cases.ncol, cases.nz
    nsteps = cases.nsteps if nsteps is None else int(nsteps)
    f = cases.fields
    dt = F32(cases.dt)

    def dev(a):
        return cp.ascontiguousarray(cp.asarray(np.asarray(a, dtype=F32)))

    # The static column: WRF's p, pi, rho, z (mass levels) and P8W.
    p, pii, rho = dev(f["p"]), dev(f["pi"]), dev(f["rho"])
    z = dev(f["z"][:nz])
    p8w = dev(f["p8w"])
    znt, ht, mavail = dev(f["znt"]), dev(f["ht"]), dev(f["mavail"])
    fcor, psfc = dev(f["fcor"]), dev(f["psfc"])
    # The moving column, on the host so its update is numpy's float32.
    u, v = f["u"].copy(), f["v"].copy()
    th, qv, qc = f["th"].copy(), f["qv"].copy(), f["qc"].copy()
    pi_h = f["pi"]

    # Registry cold start; temfinit writes TEMIN over the nz mass levels.
    carried = {name: cp.zeros((nz, ncol), dtype=cp.float32)
               for name in TEMF_PBL_STATE_3D}
    carried["te_temf"][...] = TEMIN
    carried2d = {name: cp.zeros(ncol, dtype=cp.float32)
                 for name in TEMF_PBL_STATE_2D}
    zero2 = lambda: cp.zeros(ncol, dtype=cp.float32)  # noqa: E731
    sfc_state = {"hfx": dev(f["hfx0"]), "qfx": dev(f["qfx0"]),
                 "lh": zero2(), "ust": zero2(), "wm_temf": zero2(),
                 "qsfc": zero2()}
    sfc_out = {name: zero2() for name in (
        "flhc", "flqc", "exch_temf", "chs", "chs2", "cqs2", "u10", "v10",
        "th2", "t2", "q2", "qgh", "cpm")}
    pblh = dev(f["hpbl0"])
    kpbl = cp.zeros(ncol, dtype=cp.int32)
    tend = {name: cp.zeros((nz, ncol), dtype=cp.float32) for name in (
        "rublten", "rvblten", "rthblten", "rqvblten", "rqcblten")}

    rec = {name: [] for name in
           FIELDS_SFC2D + FIELDS_PBL3D + FIELDS_PBL2D + FIELDS_STATE}
    for n in range(nsteps):
        t = (th * pi_h).astype(F32)
        tsk = dev(cases.tsk[n])
        du, dv, dth, dqv, dqc = dev(u), dev(v), dev(th), dev(qv), dev(qc)
        dt_ = dev(t)
        surface = {"u1": du[0], "v1": dv[0], "th1": dth[0], "qv1": dqv[0],
                   "p1": p[0], "pi1": pii[0], "rho1": rho[0], "z1": z[0],
                   "ht": ht, "psfc": psfc, "znt": znt, "mavail": mavail,
                   "tsk": tsk, "fcor": fcor, "hd_temf": carried2d["hd_temf"]}
        state = dict(sfc_state, te1=carried["te_temf"][0])
        launch_temf_sfclay(surface, state, sfc_out,
                           verify_stock_wrf=verify_stock_wrf)
        sfc_rec = {"hfx": sfc_state["hfx"], "qfx": sfc_state["qfx"],
                   "lh": sfc_state["lh"], "flhc": sfc_out["flhc"],
                   "flqc": sfc_out["flqc"], "qgh": sfc_out["qgh"],
                   "qsfc": sfc_state["qsfc"], "ust_sfc": sfc_state["ust"],
                   "chs": sfc_out["chs"], "chs2": sfc_out["chs2"],
                   "cqs2": sfc_out["cqs2"], "cpm": sfc_out["cpm"],
                   "u10_sfc": sfc_out["u10"], "v10_sfc": sfc_out["v10"],
                   "th2": sfc_out["th2"], "t2_sfc": sfc_out["t2"],
                   "q2": sfc_out["q2"], "exch_temf": sfc_out["exch_temf"],
                   "wm_temf": sfc_state["wm_temf"]}
        for name in FIELDS_SFC2D:
            rec[name].append(cp.asnumpy(sfc_rec[name]).copy())
        columns = {"u": du, "v": dv, "th": dth, "t": dt_, "qv": dqv,
                   "qc": dqc, "p": p, "pi": pii, "rho": rho, "z": z}
        psurf = {"znt": znt, "ht": ht, "hfx": sfc_state["hfx"],
                 "qfx": sfc_state["qfx"], "tsk": tsk, "fcor": fcor}
        inout = {"ust": sfc_state["ust"], "pblh": pblh, "kpbl": kpbl,
                 "u10": sfc_out["u10"], "v10": sfc_out["v10"],
                 "t2": sfc_out["t2"]}
        launch_temf_pbl(columns, p8w, psurf, inout, carried, carried2d,
                        tend, dt=float(dt),
                        verify_stock_wrf=verify_stock_wrf)
        for name in ("rublten", "rvblten", "rthblten", "rqvblten",
                     "rqcblten"):
            rec[name].append(cp.asnumpy(tend[name]))
        for name in FIELDS_PBL3D[5:]:
            rec[name].append(cp.asnumpy(carried[name]))
        p2 = {"hpbl": pblh, "kpbl": kpbl, "u10": sfc_out["u10"],
              "v10": sfc_out["v10"], "t2": sfc_out["t2"],
              "ust": sfc_state["ust"]}
        p2.update(carried2d)
        for name in FIELDS_PBL2D:
            rec[name].append(cp.asnumpy(p2[name]).copy())
        r = {name: cp.asnumpy(tend[name]) for name in tend}
        u = (u + dt * r["rublten"]).astype(F32)
        v = (v + dt * r["rvblten"]).astype(F32)
        th = (th + dt * r["rthblten"]).astype(F32)
        qv = (qv + dt * r["rqvblten"]).astype(F32)
        qv = np.where(qv < F32(0.0), F32(0.0), qv).astype(F32)
        qc = (qc + dt * r["rqcblten"]).astype(F32)
        qc = np.where(qc < F32(0.0), F32(0.0), qc).astype(F32)
        for name, arr in zip(FIELDS_STATE, (u, v, th, qv, qc)):
            rec[name].append(arr.copy())
    return {name: np.stack(rows) for name, rows in rec.items()}


def write_outputs(path, out: dict) -> None:
    """Write ``out`` in run_temf.F90's byte layout."""
    nsteps = next(iter(out.values())).shape[0]
    with open(path, "wb") as fh:
        for n in range(nsteps):
            for name in FIELDS_SFC2D + FIELDS_PBL3D + FIELDS_PBL2D \
                    + FIELDS_STATE:
                arr = out[name][n]
                fh.write(np.ascontiguousarray(arr).astype(
                    "<i4" if arr.dtype == np.int32 else "<f4").tobytes())
