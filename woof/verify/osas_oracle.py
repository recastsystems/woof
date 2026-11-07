"""Readers for the WRF v4.6.1 CU_OSAS column oracle (cu_physics = 95).

``tools/osas_wrf461_oracle`` builds the byte-pinned ``module_cu_osas.F``
through WRF's own preprocessing and drives ``CU_OSAS`` over analytic
soundings; ``run_osas.F90`` documents the stream layouts read here.  Every
array is returned column-major in the engine's sense: a 3-D field is
``(kx, ncol)`` (level-major, column fastest, the layout the kernels read) and
a surface field is ``(ncol,)``, with column ``c = j*im + i`` (0-based) for
Fortran ``(i+1, j+1)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LEVEL_INPUTS = ("t", "qv", "qc", "qi", "u", "v", "p", "pi", "rho", "dz8w")
IFACE_INPUTS = ("w", "p8w")
LEVEL_OUTPUTS = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten")
SURFACE_OUTPUTS = ("raincv", "pratec", "htop", "hbot")

#: gfuncphys table extents (module_gfs_funcphys.F nxpvs .. nytlcl).
TABLE_SHAPES = (("tbpvs", (7501,)), ("tbtdp", (5001,)),
                ("tbthe", (241, 151)), ("tbtma", (151, 121)),
                ("tbqma", (151, 121)), ("tbtlcl", (151, 61)))
TABLE_COEFFICIENTS = ("c1xpvs", "c2xpvs", "c1xtdp", "c2xtdp", "c1xthe",
                      "c2xthe", "c1ythe", "c2ythe", "c1xma", "c2xma",
                      "c1yma", "c2yma", "c1xtlcl", "c2xtlcl", "c1ytlcl",
                      "c2ytlcl")


@dataclass
class OsasCall:
    dt: float
    stepcu: int
    itimestep: int
    chain: int
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)


@dataclass
class OsasRun:
    im: int
    jm: int
    kx: int
    calls: list

    @property
    def ncol(self) -> int:
        return self.im * self.jm


class _Stream:
    def __init__(self, data: bytes):
        self.data = data
        self.at = 0

    def take(self, dtype, count):
        dtype = np.dtype(dtype).newbyteorder("<")
        n = dtype.itemsize * count
        if self.at + n > len(self.data):
            raise ValueError("oracle stream ends early")
        out = np.frombuffer(self.data, dtype=dtype, count=count, offset=self.at)
        self.at += n
        return out.copy()

    def done(self) -> bool:
        return self.at == len(self.data)


def _levels(stream, im, jm, levels, dtype="f4"):
    """Fortran (im, levels, jm) -> (levels, jm*im)."""
    raw = stream.take(dtype, im * levels * jm).reshape(jm, levels, im)
    return np.ascontiguousarray(raw.transpose(1, 0, 2).reshape(levels, jm * im))


def _surface(stream, im, jm, dtype="f4"):
    return stream.take(dtype, im * jm).reshape(jm * im)


def read_run(path) -> OsasRun:
    """Read run_osas.F90's output: per call, the inputs used and the outputs."""
    s = _Stream(Path(path).read_bytes())
    im, jm, kx, ncall = (int(x) for x in s.take("i4", 4))
    calls = []
    for _ in range(ncall):
        dt = float(s.take("f4", 1)[0])
        stepcu, itimestep, chain = (int(x) for x in s.take("i4", 3))
        call = OsasCall(dt=dt, stepcu=stepcu, itimestep=itimestep, chain=chain)
        for name in LEVEL_INPUTS:
            call.inputs[name] = _levels(s, im, jm, kx)
        for name in IFACE_INPUTS:
            call.inputs[name] = _levels(s, im, jm, kx + 1)
        call.inputs["xland"] = _surface(s, im, jm)
        call.inputs["xkt2"] = _surface(s, im, jm, "f8")
        for name in LEVEL_OUTPUTS:
            call.outputs[name] = _levels(s, im, jm, kx)
        for name in SURFACE_OUTPUTS:
            call.outputs[name] = _surface(s, im, jm)
        calls.append(call)
    if not s.done():
        raise ValueError(f"{path}: trailing bytes after {ncall} calls")
    return OsasRun(im=im, jm=jm, kx=kx, calls=calls)


def read_stages(path, run: OsasRun) -> list[dict]:
    """The OSASCNV and SHALCV stage dumps, one dict per call.

    Each J row of each call wrote the deep record (T1, Q1, QL, RN, CLDWRK,
    KBOT, KTOP, KUO) and then the shallow record (T1, Q1); the rows are
    stitched back into whole-domain arrays here.
    """
    s = _Stream(Path(path).read_bytes())
    im, jm, kx = run.im, run.jm, run.kx
    out = []
    for _ in run.calls:
        rec = {name: [] for name in ("t1", "q1", "ql1", "ql2", "rn", "cldwrk",
                                     "kbot", "ktop", "kuo", "t1_shallow",
                                     "q1_shallow")}
        for _j in range(jm):
            rec["t1"].append(s.take("f8", im * kx).reshape(kx, im))
            rec["q1"].append(s.take("f8", im * kx).reshape(kx, im))
            ql = s.take("f8", im * kx * 2).reshape(2, kx, im)
            rec["ql1"].append(ql[0])
            rec["ql2"].append(ql[1])
            rec["rn"].append(s.take("f8", im))
            rec["cldwrk"].append(s.take("f8", im))
            rec["kbot"].append(s.take("i4", im))
            rec["ktop"].append(s.take("i4", im))
            rec["kuo"].append(s.take("i4", im))
            rec["t1_shallow"].append(s.take("f8", im * kx).reshape(kx, im))
            rec["q1_shallow"].append(s.take("f8", im * kx).reshape(kx, im))
        stage = {}
        for name, rows in rec.items():
            if rows[0].ndim == 2:
                stage[name] = np.ascontiguousarray(np.concatenate(rows, axis=1))
            else:
                stage[name] = np.concatenate(rows)
        out.append(stage)
    if not s.done():
        raise ValueError(f"{path}: trailing bytes after the stage records")
    return out


def read_tables(path) -> dict:
    """The gfuncphys coefficients and tables (Fortran column-major tables
    flattened in Fortran order, which is the layout the kernel builds)."""
    s = _Stream(Path(path).read_bytes())
    out = {name: float(v) for name, v in
           zip(TABLE_COEFFICIENTS, s.take("f8", len(TABLE_COEFFICIENTS)))}
    for name, shape in TABLE_SHAPES:
        out[name] = s.take("f8", int(np.prod(shape)))
    if not s.done():
        raise ValueError(f"{path}: trailing bytes after the tables")
    return out
