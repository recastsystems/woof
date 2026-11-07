"""Binary layout shared by run_wdm5.F90, run_woof.py and compare.py.

All numbers little-endian.  A (ncol, nz) field is stored with the column
index fastest (Fortran order of WRF's (i,k) arrays), which is numpy C order
of shape (nz, ncol) -- the layout WOOF's (nz, ny=1, nx=ncol) kernel arrays
already have.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

INPUT_LEVEL_FIELDS = ("th", "qv", "qc", "qr", "qi", "qs", "nn", "nc", "nr",
                      "den", "pii", "p", "delz")
INPUT_SURFACE_FIELDS = ("rain", "snow")
OUTPUT_LEVEL_FIELDS = ("th", "qv", "qc", "qr", "qi", "qs", "nn", "nc", "nr",
                       "re_cloud", "re_ice", "re_snow", "refl_10cm")
OUTPUT_SURFACE_FIELDS = ("rain", "rainncv", "snow", "snowncv", "sr")
_HEAD_INTS = 4          # ncol, nz, nsteps, itimestep0
_HEAD_BYTES = 4 * _HEAD_INTS + 4 * 2


@dataclass
class Case:
    ncol: int
    nz: int
    nsteps: int
    itimestep0: int
    delt: np.float32
    ccn0: np.float32
    levels: dict           # name -> float32 (nz, ncol)
    surface: dict          # name -> float32 (ncol,)


def write_input(path: Path, case: Case) -> None:
    with open(path, "wb") as f:
        np.asarray([case.ncol, case.nz, case.nsteps, case.itimestep0],
                   "<i4").tofile(f)
        np.asarray([case.delt, case.ccn0], "<f4").tofile(f)
        for name in INPUT_LEVEL_FIELDS:
            a = np.ascontiguousarray(case.levels[name], "<f4")
            assert a.shape == (case.nz, case.ncol), (name, a.shape)
            a.tofile(f)
        for name in INPUT_SURFACE_FIELDS:
            a = np.ascontiguousarray(case.surface[name], "<f4")
            assert a.shape == (case.ncol,), (name, a.shape)
            a.tofile(f)


def read_input(path: Path) -> Case:
    raw = Path(path).read_bytes()
    head = np.frombuffer(raw, "<i4", _HEAD_INTS)
    ncol, nz, nsteps, itimestep0 = (int(v) for v in head)
    delt, ccn0 = np.frombuffer(raw, "<f4", 2, 4 * _HEAD_INTS)
    off = _HEAD_BYTES
    levels, surface = {}, {}
    for name in INPUT_LEVEL_FIELDS:
        levels[name] = np.frombuffer(raw, "<f4", nz * ncol, off).reshape(
            nz, ncol).astype(np.float32)
        off += 4 * nz * ncol
    for name in INPUT_SURFACE_FIELDS:
        surface[name] = np.frombuffer(raw, "<f4", ncol, off).astype(np.float32)
        off += 4 * ncol
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return Case(ncol, nz, nsteps, itimestep0, np.float32(delt),
                np.float32(ccn0), levels, surface)


def write_output(path: Path, steps: list[tuple[dict, dict]]) -> None:
    with open(path, "wb") as f:
        for levels, surface in steps:
            for name in OUTPUT_LEVEL_FIELDS:
                np.ascontiguousarray(levels[name], "<f4").tofile(f)
            for name in OUTPUT_SURFACE_FIELDS:
                np.ascontiguousarray(surface[name], "<f4").tofile(f)


def read_output(path: Path, ncol: int, nz: int, nsteps: int):
    raw = np.fromfile(path, "<f4")
    per = len(OUTPUT_LEVEL_FIELDS) * nz * ncol + len(OUTPUT_SURFACE_FIELDS) * ncol
    if raw.size != per * nsteps:
        raise ValueError(f"{path}: {raw.size} words, expected {per * nsteps}")
    steps = []
    for s in range(nsteps):
        block = raw[s * per:(s + 1) * per]
        off = 0
        levels, surface = {}, {}
        for name in OUTPUT_LEVEL_FIELDS:
            levels[name] = block[off:off + nz * ncol].reshape(nz, ncol)
            off += nz * ncol
        for name in OUTPUT_SURFACE_FIELDS:
            surface[name] = block[off:off + ncol]
            off += ncol
        steps.append((levels, surface))
    return steps
