"""Readers for the BMJ column-oracle files written by run_bmj.F90.

Every array is returned level-major, ``a[k, i]`` with k = 0 the lowest mass
level and i the column, which is the engine's ``a[k * ncol + i]`` layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from woof.core.fp32_ulp import monotone_fp32_key

ITB, JTB, ITBQ, JTBQ = 76, 134, 152, 440

LEVEL_INPUTS = ("t", "qv", "th", "pi", "pmid", "rho", "dz8w")
LEVEL_OUTPUTS = ("rthcuten", "rqvcuten", "ccldfra", "qcconv", "qiconv")
SURFACE_OUTPUTS = ("raincv", "pratec", "cutop", "cubot", "cldefi", "convcld")

#: The two oracle arms (run_bmj.F90): feedback flag, dt, stepcu.
ARMS = {1: (False, 60.0, 1), 2: (True, 30.0, 3)}
NSTEP = 4


def read_cases(path):
    raw = Path(path).read_bytes()
    ncol, nz = np.frombuffer(raw[:8], dtype="<i4")
    return int(ncol), int(nz)


def read_arm(path, ncol, nz, nstep=NSTEP):
    """List of per-step dicts of inputs (``in_*``) and outputs."""
    buf = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    off = 0

    def take(count, dtype):
        nonlocal off
        nbytes = count * 4
        out = buf[off:off + nbytes].view(dtype).copy()
        off += nbytes
        return out

    steps = []
    for _ in range(nstep):
        s = {}
        for name in LEVEL_INPUTS:
            s["in_" + name] = take(ncol * nz, "<f4").reshape(nz, ncol)
        s["in_pint"] = take(ncol * (nz + 1), "<f4").reshape(nz + 1, ncol)
        s["in_xland"] = take(ncol, "<f4")
        s["in_cldefi"] = take(ncol, "<f4")
        s["in_kpbl"] = take(ncol, "<i4")
        for name in LEVEL_OUTPUTS:
            s[name] = take(ncol * nz, "<f4").reshape(nz, ncol)
        for name in SURFACE_OUTPUTS:
            s[name] = take(ncol, "<f4")
        s["cu_act_flag"] = take(ncol, "<i4")
        steps.append(s)
    if off != buf.size:
        raise ValueError(f"{path}: {buf.size - off} trailing bytes; layout drift")
    return steps


def read_tables(path):
    """The BMJINIT tables, in Fortran (column-major) index order as numpy
    arrays shaped like the Fortran declaration (PTBL[ITB, JTB] etc.)."""
    a = np.frombuffer(Path(path).read_bytes(), dtype="<f4")
    off = 0

    def take(*shape):
        nonlocal off
        n = int(np.prod(shape))
        out = a[off:off + n].reshape(shape[::-1]).T.copy()
        off += n
        return out

    t = {}
    t["qs0"] = take(JTB)
    t["sqs"] = take(JTB)
    t["ptbl"] = take(ITB, JTB)
    t["the0"] = take(ITB)
    t["sthe"] = take(ITB)
    t["ttbl"] = take(JTB, ITB)
    t["the0q"] = take(ITBQ)
    t["stheq"] = take(ITBQ)
    t["ttblq"] = take(JTBQ, ITBQ)
    t["qs0_exp"] = take(JTB)
    t["sqs_exp"] = take(JTB)
    t["ptbl_exp"] = take(ITB, JTB)
    if off != a.size:
        raise ValueError(f"{path}: table layout drift")
    return t


def ulp_distance(a, b):
    """Elementwise float32 ULP distance (0 where bitwise equal)."""
    return np.abs(monotone_fp32_key(a) - monotone_fp32_key(b))
