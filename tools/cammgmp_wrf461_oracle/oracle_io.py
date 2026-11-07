"""Readers for the CAMMGMP column-oracle deck, outputs and helper words.

The deck (make_cases.write_deck) and the outputs (run_cammgmp.F90) are
little-endian streams.  Arrays come back in the engine's layout: per-level
fields ``(nlev, ncol)`` (k = 1 at the surface first), surface fields
``(ncol,)``.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

try:  # package import (tests) or script directory (tools)
    from tools.cammgmp_wrf461_oracle.make_cases import (
        FIELDS_2D, FIELDS_3D, MAGIC, OUT_2D, OUT_3D, VERSION)
except ImportError:  # pragma: no cover
    from make_cases import FIELDS_2D, FIELDS_3D, MAGIC, OUT_2D, OUT_3D, VERSION


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def ints(self, n):
        out = struct.unpack_from(f"<{n}i", self.data, self.pos)
        self.pos += 4 * n
        return out

    def floats(self, n):
        out = struct.unpack_from(f"<{n}f", self.data, self.pos)
        self.pos += 4 * n
        return out

    def array(self, dtype, count):
        out = np.frombuffer(self.data, dtype=dtype, count=count, offset=self.pos)
        self.pos += out.nbytes
        return out.copy()


def _group_header(r):
    ncol, nz, nstep, it0, cu, shcu = r.ints(6)
    dt, accum, aitken, coarse = r.floats(4)
    return dict(ncol=ncol, nz=nz, nstep=nstep, it0=it0, cu=cu, shcu=shcu,
                dt=np.float32(dt), accum=np.float32(accum),
                aitken=np.float32(aitken), coarse=np.float32(coarse))


def read_deck(path) -> list[dict]:
    r = _Reader(Path(path).read_bytes())
    magic, version, ngroups = r.ints(3)
    if (magic, version) != (MAGIC, VERSION):
        raise ValueError(f"{path} is not a CAMMGMP deck (v{VERSION})")
    groups = []
    for _ in range(ngroups):
        g = _group_header(r)
        ncol, nlev = g["ncol"], g["nz"] + 1
        f3 = {}
        for name in FIELDS_3D:
            a = r.array("<f4", ncol * nlev).reshape(nlev, ncol)  # Fortran (i,k)
            f3[name] = np.ascontiguousarray(a)
        f2 = {name: r.array("<f4", ncol) for name in FIELDS_2D}
        g["fields3d"], g["fields2d"] = f3, f2
        groups.append(g)
    if r.pos != len(r.data):
        raise ValueError(f"{path}: {len(r.data) - r.pos} trailing bytes")
    return groups


def read_outputs(path) -> list[dict]:
    """Per group: header plus ``steps``, a list of {name: array} per step."""
    r = _Reader(Path(path).read_bytes())
    magic, version, ngroups = r.ints(3)
    if (magic, version) != (MAGIC, VERSION):
        raise ValueError(f"{path} is not a CAMMGMP oracle output")
    groups = []
    for _ in range(ngroups):
        g = _group_header(r)
        ncol, nz = g["ncol"], g["nz"]
        steps = []
        for _ in range(g["nstep"]):
            out = {}
            for name in OUT_3D:
                out[name] = np.ascontiguousarray(
                    r.array("<f4", ncol * nz).reshape(nz, ncol))
            for name in OUT_2D:
                out[name] = r.array("<f4", ncol)
            steps.append(out)
        g["steps"] = steps
        groups.append(g)
    if r.pos != len(r.data):
        raise ValueError(f"{path}: {len(r.data) - r.pos} trailing bytes")
    return groups


def read_helpers(path) -> dict:
    r = _Reader(Path(path).read_bytes())
    nt, ne, ng = r.ints(3)
    t = r.array("<f8", 4 * nt).reshape(nt, 4)
    e = r.array("<f8", 2 * ne).reshape(ne, 2)
    gm = r.array("<f8", 2 * ng).reshape(ng, 2)
    (nmode,) = r.ints(1)
    modal = {}
    for name in ("sigmag", "dgnum", "voltonumb", "voltonumblo", "voltonumbhi",
                 "alnsg"):
        modal[name] = r.array("<f8", nmode)
    modal["specdens"] = r.array("<f8", 3)
    modal["spechygro"] = r.array("<f8", 3)
    modal["numptr"] = r.array("<i4", nmode)
    modal["nspec"] = r.array("<i4", nmode)
    modal["lmassptr"] = r.array("<i4", 2 * nmode).reshape(nmode, 2)
    modal["lspectype"] = r.array("<i4", 2 * nmode).reshape(nmode, 2)
    if r.pos != len(r.data):
        raise ValueError(f"{path}: {len(r.data) - r.pos} trailing bytes")
    return dict(t=t[:, 0], estblf=t[:, 1], polysvp_liq=t[:, 2],
                polysvp_ice=t[:, 3], erf_x=e[:, 0], erf=e[:, 1],
                gamma_x=gm[:, 0], gamma=gm[:, 1], modal=modal)
