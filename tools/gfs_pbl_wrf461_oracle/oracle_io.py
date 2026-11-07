"""Read the GFS PBL oracle's stream files (tools/gfs_pbl_wrf461_oracle).

``run_bl_gfs`` writes, per call, the call's complete input record and its
outputs (run_bl_gfs.F90's header).  Arrays come back in the engine's layout:
3-D fields ``(nz, ncol)`` with WRF's k = 1 first (the oracle's extra kme
level is dropped -- BL_GFS never reads or writes it), 2-D fields ``(ncol,)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIELDS_3D = ("u", "v", "th", "t", "qv", "qc", "qi", "p", "pi", "dz8w", "z")
FIELDS_2D = ("psfc", "ust", "pbl", "psim", "psih", "hfx", "qfx", "tsk",
             "gz1oz0", "wspd", "br")
OUT_3D = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
          "rqiblten")
OUT_2D = ("ust", "wspd", "pbl")


def read_oracle(path) -> dict:
    raw = Path(path).read_bytes()
    head = np.frombuffer(raw, "<i4", 4)
    ncol, nz, nstep, flqi = (int(x) for x in head)
    dt = float(np.frombuffer(raw, "<f4", 1, 16)[0])
    kme = nz + 1
    pos = 20
    n3 = ncol * kme
    steps = []

    def take(count, dtype):
        nonlocal pos
        arr = np.frombuffer(raw, dtype, count, pos)
        pos += 4 * count
        return arr

    for _ in range(nstep):
        inputs = {}
        for name in FIELDS_3D:
            inputs[name] = take(n3, "<f4").reshape(kme, ncol)[:nz].copy()
        for name in FIELDS_2D:
            inputs[name] = take(ncol, "<f4").copy()
        outputs = {}
        for name in OUT_3D:
            outputs[name] = take(n3, "<f4").reshape(kme, ncol)[:nz].copy()
        for name in OUT_2D:
            outputs[name] = take(ncol, "<f4").copy()
        outputs["kpbl"] = take(ncol, "<i4").copy()
        steps.append({"inputs": inputs, "outputs": outputs})
    if pos != len(raw):
        raise ValueError(f"{path}: {len(raw) - pos} trailing bytes")
    return {"ncol": ncol, "nz": nz, "nstep": nstep, "flqi": bool(flqi),
            "dt": dt, "steps": steps}


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|ULP| between float32 words (signed-magnitude ordered; -0 == +0)."""
    ia = a.astype("<f4").view("<i4").astype(np.int64)
    ib = b.astype("<f4").view("<i4").astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ib = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return np.abs(ia - ib)
