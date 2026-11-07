"""Readers for the GBM PBL (bl_pbl_physics=12) oracle files.

Written by tools/gbmpbl_wrf461_oracle (make_cases.py, run_gbmpbl.F90) and
published in woof/data/gbmpbl/oracle/.

The case file holds what the WRF driver was handed; the oracle file holds,
per set and per step, the state the scheme read and every word it wrote.
Arrays come back bottom-up ``(nz, ncol)`` (WRF level kme, never read, is
dropped) so they line up with the engine's ``(nz, ny, nx)`` columns.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from woof import data_assets

import numpy as np

CASE_FIELDS_3D = ("u", "v", "th", "qv", "qc", "qi", "p", "pi", "dz8w", "z",
                  "rthraten", "tke")
CASE_FIELDS_2D = ("psfc", "znt", "ust", "zol", "hol", "pbl", "regime", "psim",
                  "psih", "xland", "hfx", "qfx", "tsk", "gz1oz0", "wspd",
                  "br")
STATE_FIELDS = ("u", "v", "th", "t", "qv", "qc", "qi", "tke_in")
TENDENCY_FIELDS = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
                   "rqiblten")
COLUMN_OUTPUTS = ("exch_m", "exch_h", "exch_tke", "el_pbl", "tke_pbl")
SURFACE_OUTPUTS = ("pblh", "ust", "regime")

#: The published fixture directory.
ORACLE_DIR = data_assets.verify_oracle_path("gbmpbl/oracle")


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, dtype, count):
        arr = np.frombuffer(self.data, dtype=dtype, count=count,
                            offset=self.pos).copy()
        self.pos += arr.nbytes
        return arr


def read_cases(path) -> list[dict]:
    r = _Reader(Path(path).read_bytes())
    sets = []
    for _ in range(int(r.take("<i4", 1)[0])):
        ncol, nz, nstep = (int(x) for x in r.take("<i4", 3))
        s = {"ncol": ncol, "nz": nz, "nstep": nstep,
             "dt": np.float32(r.take("<f4", 1)[0])}
        for name in CASE_FIELDS_3D:
            s[name] = r.take("<f4", (nz + 1) * ncol).reshape(nz + 1, ncol)[:nz]
        for name in CASE_FIELDS_2D:
            s[name] = r.take("<f4", ncol)
        s["kpbl"] = r.take("<i4", ncol)
        sets.append(s)
    if r.pos != len(r.data):
        raise ValueError("trailing bytes in the GBM case file")
    return sets


def read_oracle(path) -> list[dict]:
    r = _Reader(Path(path).read_bytes())
    sets = []
    while r.pos < len(r.data):
        ncol, nz, nstep = (int(x) for x in r.take("<i4", 3))
        s = {"ncol": ncol, "nz": nz, "nstep": nstep,
             "dt": np.float32(r.take("<f4", 1)[0]), "steps": []}
        n3 = (nz + 1) * ncol
        for _ in range(nstep):
            step = {}
            for name in (*STATE_FIELDS, *TENDENCY_FIELDS, *COLUMN_OUTPUTS):
                step[name] = r.take("<f4", n3).reshape(nz + 1, ncol)[:nz]
            for name in SURFACE_OUTPUTS:
                step[name] = r.take("<f4", ncol)
            step["kpbl"] = r.take("<i4", ncol)
            s["steps"].append(step)
        sets.append(s)
    return sets


#: Fields compared per step: the state the scheme read (proves the chain),
#: then every word it published.
COMPARED = (*STATE_FIELDS, *TENDENCY_FIELDS, *COLUMN_OUTPUTS,
            *SURFACE_OUTPUTS, "kpbl")


def replay_set(case: dict, oracle: dict, *, launch=None) -> list[dict]:
    """Run the port over one oracle set on the current device.

    Starts from the case file's state and carries its OWN state from step
    to step with the harness's float32 update (run_gbmpbl.F90): u, v and
    theta advanced by dt times the tendency, T = theta * pi, the species
    advanced and clipped at zero, TKE_PBL carried as the scheme left it.
    Returns, per step, ``{field: max ULP}`` against the oracle (kpbl as an
    integer count of differing columns).
    """
    import cupy as cp

    from woof.core import gbmpbl
    from woof.core.fp32_ulp import fp32_ulp_distance

    launch = gbmpbl.launch_gbmpbl if launch is None else launch
    f32 = np.float32
    nz, ncol = case["nz"], case["ncol"]
    dt = f32(case["dt"])

    def dev3(a):
        return cp.asarray(np.ascontiguousarray(a, dtype=f32)[:, None, :])

    def dev2(a):
        return cp.asarray(np.ascontiguousarray(a, dtype=f32)[None, :])

    st = {k: np.array(case[k], dtype=f32) for k in
          ("u", "v", "th", "qv", "qc", "qi", "tke")}
    pi = np.array(case["pi"], dtype=f32)
    st["t"] = st["th"] * pi
    fixed = {"p": dev3(case["p"]), "exner": dev3(pi),
             "dz8w": dev3(case["dz8w"]), "rthraten": dev3(case["rthraten"])}
    surface = {k: dev2(case[k]) for k in
               ("psfc", "gz1oz0", "psim", "xland", "hfx", "qfx", "tsk", "regime")}
    results = []
    for step_index, want in enumerate(oracle["steps"]):
        got = {"tke_in": st["tke"].copy()}
        got.update({k: st[k].copy() for k in
                    ("u", "v", "th", "t", "qv", "qc", "qi")})
        columns = dict(fixed, **{k: dev3(st[k]) for k in
                                 ("u", "v", "t", "qv", "qc", "qi")})
        tke = dev3(st["tke"])
        outputs = {k: cp.empty((nz, 1, ncol), dtype=cp.float32)
                   for k in gbmpbl.GBMPBL_COLUMN_OUTPUTS}
        outputs.update(pblh=cp.empty((1, ncol), dtype=cp.float32),
                       ust=cp.empty((1, ncol), dtype=cp.float32),
                       kpbl=cp.empty((1, ncol), dtype=cp.int32))
        launch(columns, surface, tke, outputs, dt=float(dt))
        for k in (*TENDENCY_FIELDS, "exch_m", "exch_h", "exch_tke", "el_pbl"):
            got[k] = cp.asnumpy(outputs[k])[:, 0, :]
        got["tke_pbl"] = cp.asnumpy(tke)[:, 0, :]
        got["regime"] = cp.asnumpy(surface["regime"])[0]
        for k in ("pblh", "ust", "kpbl"):
            got[k] = cp.asnumpy(outputs[k])[0]
        row = {"step": step_index + 1}
        for k in COMPARED:
            if k == "kpbl":
                row[k] = int(np.count_nonzero(got[k] != want[k]))
            else:
                d = fp32_ulp_distance(got[k], want[k])
                row[k] = int(np.max(d)) if d.size else 0
        row["different_words"] = {
            k: int(np.count_nonzero(got[k].view(np.uint32) != want[k].view(np.uint32)))
            for k in COMPARED}
        row["field_hashes"] = {
            k: {"gpu": hashlib.sha256(got[k].tobytes()).hexdigest(),
                "wrf": hashlib.sha256(want[k].tobytes()).hexdigest()}
            for k in COMPARED}
        results.append(row)
        # The harness's update, word for word (run_gbmpbl.F90).
        st["u"] = st["u"] + dt * got["rublten"]
        st["v"] = st["v"] + dt * got["rvblten"]
        st["th"] = st["th"] + dt * got["rthblten"]
        st["t"] = st["th"] * pi
        st["qv"] = np.maximum(st["qv"] + dt * got["rqvblten"], f32(0.0))
        st["qc"] = np.maximum(st["qc"] + dt * got["rqcblten"], f32(0.0))
        st["qi"] = np.maximum(st["qi"] + dt * got["rqiblten"], f32(0.0))
        st["tke"] = got["tke_pbl"]
    return results
