"""Reader for the EEPS (bl_pbl_physics=16) WRF v4.6.1 column-oracle trace.

``tools/eeps_wrf461_oracle/run_bl_eeps.F90`` writes one stream file per
column set: a header (ncol, nz, nsteps, first itimestep, qs/qg present, dt)
and then, for every EEPS call, the full input set the call saw followed by
the full output set it produced.  Every 3-D record is WRF bottom-up
``field(1:ncol, 1:nz, 1)`` in Fortran order, which is numpy ``(nz, ncol)``,
the engine's own ``(nz, ny, nx)`` layout with ``ny = 1``.  EXCH_H/EXCH_M
carry WRF's ``kte+1`` level and are ``(nz + 1, ncol)``.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
from woof.core.fp32_ulp import monotone_fp32_key

#: Column inputs, in record order.
INPUT_3D = ("u", "v", "t", "qv", "qc", "qi", "qr", "qs", "qg", "p", "pi",
            "rho", "rthraten", "dz8w")
#: Surface inputs, in record order.
INPUT_2D = ("psfc", "qsfc", "tsk", "ust", "rmol", "wspd", "xland", "hfx",
            "qfx")
#: Carried TKE/epsilon as the call received them.
CARRIED = ("pek", "pep", "pek_adv", "pep_adv")
#: Mass-level outputs, in record order.
OUTPUT_3D = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
             "rqiblten", "pek", "pep", "pek_adv", "pep_adv")
#: Full-level outputs (``nz + 1`` levels).
OUTPUT_FULL = ("exch_h", "exch_m")

REGIMES = ("convective_land", "stable_night", "neutral_windy_ocean",
           "cloud_topped", "very_shallow", "near_model_top", "mixed_phase",
           "moist_tropical")
PER_REGIME = 9


class _Cursor:
    def __init__(self, raw: bytes):
        self.raw = raw
        self.pos = 0

    def take(self, dtype, count):
        dtype = np.dtype(dtype)
        n = dtype.itemsize * count
        out = np.frombuffer(self.raw, dtype=dtype, count=count,
                            offset=self.pos).copy()
        self.pos += n
        return out


def read_trace(path) -> dict:
    """Parse one trace file into ``{"header": ..., "steps": [...]}``."""
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    cur = _Cursor(raw)
    ncol, nz, nsteps, first, has_qsqg = (int(x) for x in cur.take("<i4", 5))
    dt = float(cur.take("<f4", 1)[0])
    steps = []
    for _ in range(nsteps):
        inputs = {}
        for name in INPUT_3D:
            inputs[name] = cur.take("<f4", nz * ncol).reshape(nz, ncol)
        for name in INPUT_2D:
            inputs[name] = cur.take("<f4", ncol)
        for name in CARRIED:
            inputs[name] = cur.take("<f4", nz * ncol).reshape(nz, ncol)
        itimestep = int(cur.take("<i4", 1)[0])
        outputs = {}
        for name in OUTPUT_3D:
            outputs[name] = cur.take("<f4", nz * ncol).reshape(nz, ncol)
        for name in OUTPUT_FULL:
            outputs[name] = cur.take("<f4", (nz + 1) * ncol).reshape(
                nz + 1, ncol)
        outputs["pblh"] = cur.take("<f4", ncol)
        outputs["kpbl"] = cur.take("<i4", ncol)
        steps.append({"itimestep": itimestep, "inputs": inputs,
                      "outputs": outputs})
    if cur.pos != len(cur.raw):
        raise ValueError(f"{path}: {len(cur.raw) - cur.pos} trailing bytes; "
                         "the reader and run_bl_eeps.F90 disagree on layout")
    return {"header": {"ncol": ncol, "nz": nz, "nsteps": nsteps,
                       "first_itimestep": first, "has_qs_qg": bool(has_qsqg),
                       "dt": dt},
            "steps": steps}


def blend_carried(a: np.ndarray) -> np.ndarray:
    """The oracle's stand-in for scalar advection of PEK_ADV/PEP_ADV.

    ``run_bl_eeps.F90::blend``: interior levels become
    ``0.5*a(k) + 0.25*(a(k-1) + a(k+1))`` in float32, evaluated in that
    order (gfortran, -ffp-contract=off), the end levels are unchanged.
    """
    a = np.asarray(a, dtype=np.float32)
    out = a.copy()
    half, quarter = np.float32(0.5), np.float32(0.25)
    out[1:-1] = half * a[1:-1] + quarter * (a[:-2] + a[2:])
    return out


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Units in the last place between two float32 arrays (0 = identical)."""
    return np.abs(monotone_fp32_key(a) - monotone_fp32_key(b))
