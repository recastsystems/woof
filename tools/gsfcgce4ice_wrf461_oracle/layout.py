"""Byte layout shared by the Goddard 4-ice oracle runner and its graders.

run_gsfcgce4ice.F90 reads and writes exactly this layout (raw little-endian
stream files, Fortran column-major arrays (ncol, nz) == C (nz, ncol)).  The
field order is the runner's argument order; changing one side without the
other is caught by ``read_output`` refusing a byte count that does not match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: INOUT and IN level fields of gsfcgce_4ice_nuwrf, in runner index order.
LEVEL_FIELDS = (
    "th", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
    "rho", "pii", "p", "z", "dz8w",
    "refl_10cm",
    "physc", "physe", "physd", "physs", "physm", "physf",
    "acphysc", "acphyse", "acphysd", "acphyss", "acphysm", "acphysf",
    "re_cloud_gsfc", "re_rain_gsfc", "re_ice_gsfc", "re_snow_gsfc",
    "re_graupel_gsfc", "re_hail_gsfc",
)
#: INTENT(IN) members of LEVEL_FIELDS: written back unchanged, not graded.
INPUT_ONLY = ("rho", "pii", "p", "z", "dz8w")
#: INTENT(OUT) sedimentation fluxes (kg m-2 s-1), output only.
PREC_FIELDS = ("preci3d", "precs3d", "precg3d", "prech3d", "precr3d")
SURFACE_FIELDS = ("rainnc", "rainncv", "snownc", "snowncv", "sr",
                  "graupelnc", "graupelncv", "hailnc", "hailncv")

GRADED_FIELDS = (tuple(f for f in LEVEL_FIELDS if f not in INPUT_ONLY)
                 + PREC_FIELDS + SURFACE_FIELDS)


@dataclass
class Case:
    """One oracle run: ``ncol`` independent columns advanced ``nstep`` calls."""
    dt: float
    dx: float
    itimestep: np.ndarray          # (nstep,) int32
    diagflag: np.ndarray           # (nstep,) int32, 0/1
    do_radar_ref: int
    ht: np.ndarray                 # (ncol,)
    xland: np.ndarray              # (ncol,)
    level: dict                    # name -> (nz, ncol) float32
    w: np.ndarray                  # (nz+1, ncol) float32
    surface: dict                  # name -> (ncol,) float32
    labels: list                   # (ncol,) column regime labels

    @property
    def ncol(self) -> int:
        return int(self.ht.shape[0])

    @property
    def nz(self) -> int:
        return int(self.level["th"].shape[0])

    @property
    def nstep(self) -> int:
        return int(self.itimestep.shape[0])


def _f32(a) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype != np.float32:
        raise TypeError(f"layout fields are float32, got {a.dtype}")
    return np.ascontiguousarray(a)


def write_input(case: Case, path: str | Path) -> None:
    ncol, nz = case.ncol, case.nz
    with open(path, "wb") as fh:
        fh.write(np.array([ncol, nz, case.nstep, case.do_radar_ref],
                          dtype="<i4").tobytes())
        fh.write(np.array([case.dt, case.dx], dtype="<f4").tobytes())
        fh.write(np.asarray(case.itimestep, dtype="<i4").tobytes())
        fh.write(np.asarray(case.diagflag, dtype="<i4").tobytes())
        fh.write(_f32(case.ht).astype("<f4").tobytes())
        fh.write(_f32(case.xland).astype("<f4").tobytes())
        for name in LEVEL_FIELDS:
            a = _f32(case.level[name])
            if a.shape != (nz, ncol):
                raise ValueError(f"{name} must be (nz, ncol), got {a.shape}")
            fh.write(a.astype("<f4").tobytes())
        if case.w.shape != (nz + 1, ncol):
            raise ValueError("w must be (nz+1, ncol)")
        fh.write(_f32(case.w).astype("<f4").tobytes())
        for name in SURFACE_FIELDS:
            fh.write(_f32(case.surface[name]).astype("<f4").tobytes())


def read_input(path: str | Path) -> Case:
    raw = Path(path).read_bytes()
    pos = 0

    def take(n, dtype):
        nonlocal pos
        size = np.dtype(dtype).itemsize * n
        out = np.frombuffer(raw, dtype=dtype, count=n, offset=pos).copy()
        pos += size
        return out

    ncol, nz, nstep, do_radar_ref = (int(v) for v in take(4, "<i4"))
    dt, dx = (float(v) for v in take(2, "<f4"))
    itimestep = take(nstep, "<i4")
    diagflag = take(nstep, "<i4")
    ht = take(ncol, "<f4").astype(np.float32)
    xland = take(ncol, "<f4").astype(np.float32)
    level = {n: take(nz * ncol, "<f4").reshape(nz, ncol).astype(np.float32)
             for n in LEVEL_FIELDS}
    w = take((nz + 1) * ncol, "<f4").reshape(nz + 1, ncol).astype(np.float32)
    surface = {n: take(ncol, "<f4").astype(np.float32)
               for n in SURFACE_FIELDS}
    if pos != len(raw):
        raise ValueError(f"{path}: {len(raw) - pos} trailing bytes")
    return Case(np.float32(dt), np.float32(dx), itimestep, diagflag,
                do_radar_ref, ht, xland, level, w, surface,
                [""] * ncol)


def read_output(path: str | Path, ncol: int, nz: int, nstep: int):
    """Return ``[step] -> {field: array}`` from a runner output file."""
    per_step = ((len(LEVEL_FIELDS) + len(PREC_FIELDS)) * nz * ncol
                + len(SURFACE_FIELDS) * ncol)
    raw = np.fromfile(path, dtype="<f4")
    if raw.size != per_step * nstep:
        raise ValueError(f"{path}: {raw.size} words, expected "
                         f"{per_step * nstep} for {nstep} steps")
    steps = []
    pos = 0
    for _ in range(nstep):
        out = {}
        for name in LEVEL_FIELDS + PREC_FIELDS:
            out[name] = raw[pos:pos + nz * ncol].reshape(nz, ncol).astype(
                np.float32)
            pos += nz * ncol
        for name in SURFACE_FIELDS:
            out[name] = raw[pos:pos + ncol].astype(np.float32)
            pos += ncol
        steps.append(out)
    return steps
