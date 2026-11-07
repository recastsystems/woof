"""Readers for the ACM2 PBL oracle fixtures (tools/acm2_wrf461_oracle).

The oracle drives WRF v4.6.1's ``ACMPBL`` (phys/module_bl_acm.F, byte
pinned) over one tile row of independent columns for several consecutive
steps and records, per step, the state it was handed and everything it
returned.  These readers turn the stream files back into arrays laid out
the way the engine holds them: 3-D fields ``(nz, ncol)`` (level-major, one
column per lane), 2-D fields ``(ncol,)``.

Pure numpy; importable on a machine with no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Field order of the input file's 3-D block (run_acm2.F90 header).
INPUT_FIELDS_3D = ("u", "v", "p", "dz8w", "th", "t", "qv", "qc", "qi", "rho",
                   "pi")
#: Field order of the input file's 2-D block.
INPUT_FIELDS_2D = ("ust", "hfx", "qfx", "tsk", "psfc", "wspd", "psim",
                   "gz1oz0", "mut", "pblh", "regime", "rmol")
#: Per-step state handed to the scheme (3-D, then the carried 2-D).
STEP_STATE_3D = ("u", "v", "th", "t", "qv", "qc", "qi")
STEP_STATE_2D = ("pblh", "regime", "rmol")
#: Per-step outputs.
STEP_OUT_3D = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
               "rqiblten", "exch_h", "exch_m")
STEP_OUT_2D = ("pblh", "regime", "rmol")


@dataclass(frozen=True)
class AcmFixture:
    ncol: int
    nz: int
    nsteps: int
    xtime0: int
    constants: dict
    static3d: dict
    static2d: dict
    steps: list


class _Reader:
    def __init__(self, data: bytes):
        self.buf = memoryview(data)
        self.pos = 0

    def take(self, dtype: str, count: int) -> np.ndarray:
        size = np.dtype(dtype).itemsize * count
        out = np.frombuffer(self.buf[self.pos:self.pos + size], dtype=dtype,
                            count=count).copy()
        self.pos += size
        return out

    def done(self) -> bool:
        return self.pos == len(self.buf)


def _field3(reader: _Reader, ncol: int, nz: int) -> np.ndarray:
    # Written as Fortran tmp(nz, ncol): k fastest, so (ncol, nz) row-major.
    return reader.take("<f4", ncol * nz).reshape(ncol, nz).T.copy()


def load_inputs(stem: str | Path) -> AcmFixture:
    """Load ``<stem>.in.bin`` alone (``steps`` is empty)."""
    stem = Path(stem)
    rin = _Reader(Path(f"{stem}.in.bin").read_bytes())
    ncol, nz, nsteps, xtime0 = (int(x) for x in rin.take("<i4", 4))
    dtpbl, ep1, g, rovcp, rd, cpd = rin.take("<f4", 6)
    constants = {"dtpbl": dtpbl, "ep1": ep1, "g": g, "rovcp": rovcp,
                 "rd": rd, "cpd": cpd}
    static3d = {name: _field3(rin, ncol, nz) for name in INPUT_FIELDS_3D}
    static2d = {name: rin.take("<f4", ncol) for name in INPUT_FIELDS_2D}
    if not rin.done():
        raise ValueError(f"{stem}.in.bin has trailing bytes")
    return AcmFixture(ncol, nz, nsteps, xtime0, constants, static3d,
                      static2d, [])


def load_fixture(stem: str | Path) -> AcmFixture:
    """Load ``<stem>.in.bin`` and ``<stem>.out.bin``."""
    stem = Path(stem)
    fin = load_inputs(stem)
    ncol, nz, nsteps, xtime0 = fin.ncol, fin.nz, fin.nsteps, fin.xtime0
    constants, static3d, static2d = fin.constants, fin.static3d, fin.static2d
    rout = _Reader(Path(f"{stem}.out.bin").read_bytes())
    head = tuple(int(x) for x in rout.take("<i4", 4))
    if head != (ncol, nz, nsteps, xtime0):
        raise ValueError(f"{stem}: output header {head} does not match input")
    steps = []
    for _ in range(nsteps):
        xtime = int(rout.take("<i4", 1)[0])
        state3 = {name: _field3(rout, ncol, nz) for name in STEP_STATE_3D}
        state2 = {name: rout.take("<f4", ncol) for name in STEP_STATE_2D}
        out3 = {name: _field3(rout, ncol, nz) for name in STEP_OUT_3D}
        out2 = {name: rout.take("<f4", ncol) for name in STEP_OUT_2D}
        kpbl = rout.take("<i4", ncol)
        steps.append({"xtime": xtime, "state3d": state3, "state2d": state2,
                      "out3d": out3, "out2d": out2, "kpbl": kpbl})
    if not rout.done():
        raise ValueError(f"{stem}.out.bin has trailing bytes")
    return AcmFixture(ncol, nz, nsteps, xtime0, constants, static3d,
                      static2d, steps)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in float32 units in the last place, sign-magnitude ordered.

    +0 and -0 are 0 apart here; callers that need the sign of zero compare
    bit patterns directly.
    """
    ai = np.asarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    bi = np.asarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, -(ai & 0x7FFFFFFF), ai)
    bi = np.where(bi < 0, -(bi & 0x7FFFFFFF), bi)
    return np.abs(ai - bi)
