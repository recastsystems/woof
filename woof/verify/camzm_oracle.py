"""Reader for the WRF v4.6.1 CAM Zhang-McFarlane (cu_physics = 7) oracle.

The fixtures are written by ``tools/camzm_wrf461_oracle`` (see its README):
each ``<stem>.bin`` is raw little-endian words and ``<stem>.manifest`` names
them, one line per array, ``name dtype count byte_offset``.

A fixture (``zm-<group>``) carries ``const/...`` records (the scheme's
physical constants and its tile settings: dt, dx, cudt, the PBL arm) and,
per step ``s<n>/``, the exact float32 inputs camzm_driver received
(``s<n>/in/...``, with ``ran`` saying whether the driver computed or held)
and every output it left (``s<n>/out/...``).  Per-column arrays are stored
level-major with the column fastest, so :func:`step_arrays` hands them back
shaped ``(nlev, ncol)``, WRF's k = 1 (the surface) first.

``zm-<group>-stock`` is byte-unmodified WRF and is the default reference.
``zm-<group>`` is a historical altered-source diagnostic from
tools/camzm_wrf461_oracle/fixes.patch. Neither generated stream is a capture
from a real forecast. Raw streams are disposable; CAMZM_ORACLE_DIR can point
to a freshly generated fixture directory.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

_DTYPES = {"f8": "<f8", "f4": "<f4", "i4": "<i4"}

#: A generated fixture directory. Only manifests and receipts are packaged.
FIXTURE_DIR = Path(os.environ.get("CAMZM_ORACLE_DIR") or
                   Path(__file__).resolve().parents[2] / "tests/data/oracles/camzm")

#: The four tiles (case files) of the oracle: one dx, dt, cudt, PBL arm and
#: vertical grid each, as WRF fixes those per domain.
GROUPS = ("uw12km", "myj3km", "uw100km", "toptrop")

#: Mass-level float32 inputs, interface inputs and surface inputs, in the
#: order run_camzm.F90 writes them for every step.
MASS_INPUTS = ("th", "t", "p", "pi", "qv", "qc", "qi", "z", "dz8w", "u", "v",
               "cldfra")
FULL_INPUTS = ("p8w", "z_at_w", "tke_pbl")
SURFACE_INPUTS = ("tsk", "ust", "mavail", "pblh", "xland", "ht", "psfc",
                  "hfx", "qfx", "tpert")
INT_SURFACE_INPUTS = ("kpbl",)

#: Mass-level float32 outputs, in WRF's names (zmmu/zmmd are camzm_driver's
#: MU_OUT/MD_OUT, dlf its DLF_OUT, as module_cumulus_driver.F binds them).
MASS_OUTPUTS = (
    "rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rqcncuten", "rqincuten",
    "rucuten", "rvcuten", "zmmu", "zmmd", "zmdt", "zmdq", "dlf", "evaptzm",
    "fzsntzm", "evsntzm", "evapqzm", "zmflxprc", "zmflxsnw", "zmntprpd",
    "zmntsnpd", "zmeiheat", "zmmtu", "zmmtv", "zmupgu", "zmupgd", "zmvpgu",
    "zmvpgd", "zmicuu", "zmicud", "zmicvu", "zmicvd", "zmdice", "zmdliq",
    "evapcdp3d", "icwmrdp3d", "rprddp3d", "dp3d", "du3d", "ed3d", "eu3d",
    "md3d", "mu3d")
#: Interface (nk+1) outputs.
FULL_OUTPUTS = ("cmfmc", "cmfmcdzm")
#: Surface float32 outputs.
SURFACE_OUTPUTS = ("raincv", "pratec", "cape", "rliq", "pconvb", "pconvt",
                   "cubot", "cutop", "preccdzm", "precz", "dsubcld2d")
#: Surface int32 outputs (-1 where lengath2d = 0; see run_camzm.F90).
INT_SURFACE_OUTPUTS = ("ideep2d", "jt2d", "maxg2d", "lengath2d")


@dataclass(frozen=True)
class Fixture:
    """One oracle stream: every named array, read eagerly."""

    path: Path
    arrays: dict

    def __getitem__(self, name):
        return self.arrays[name]

    def __contains__(self, name):
        return name in self.arrays

    def names(self, prefix: str = ""):
        return [n for n in self.arrays if n.startswith(prefix)]


def load(stem) -> Fixture:
    """Read ``<stem>.bin`` through ``<stem>.manifest``."""
    stem = Path(stem)
    if stem.suffix in (".bin", ".manifest"):
        stem = stem.with_suffix("")
    raw = Path(f"{stem}.bin").read_bytes()
    arrays = {}
    for line in Path(f"{stem}.manifest").read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        name, dtype, count, offset = line.split()
        arrays[name] = np.frombuffer(raw, dtype=np.dtype(_DTYPES[dtype]),
                                     count=int(count), offset=int(offset)).copy()
    return Fixture(path=stem, arrays=arrays)


def fixture(group: str, *, stock: bool = True,
            directory: Path | None = None) -> Fixture:
    """The stock stream, or an explicitly requested altered-source stream."""
    d = FIXTURE_DIR if directory is None else Path(directory)
    return load(d / (f"zm-{group}-stock" if stock else f"zm-{group}"))


def dims(fx: Fixture) -> tuple[int, int, int]:
    """``(ncol, nk, nsteps)``."""
    return (int(fx["const/ncol"][0]), int(fx["const/nk"][0]),
            int(fx["const/nsteps"][0]))


def tile(fx: Fixture) -> dict:
    """The tile-wide settings WRF hands camzm_driver for this fixture."""
    return {
        "dt": float(fx["const/dt"][0]), "dx": float(fx["const/dx"][0]),
        "cudt": float(fx["const/cudt"][0]),
        "stepcu": int(fx["const/stepcu"][0]),
        "bl_pbl_physics": int(fx["const/bl_pbl_physics"][0]),
        "sf_sfclay_physics": int(fx["const/sf_sfclay_physics"][0]),
    }


def step_arrays(fx: Fixture, step: int) -> dict:
    """Every input and output of one step, per-level arrays ``(nlev, ncol)``."""
    ncol, nk, _ = dims(fx)
    a: dict = {"itimestep": int(fx[f"s{step}/in/itimestep"][0]),
               "ran": bool(fx[f"s{step}/in/ran"][0])}
    for name in MASS_INPUTS:
        a[name] = fx[f"s{step}/in/{name}"].reshape(nk, ncol)
    for name in FULL_INPUTS:
        a[name] = fx[f"s{step}/in/{name}"].reshape(nk + 1, ncol)
    for name in SURFACE_INPUTS + INT_SURFACE_INPUTS:
        a[name] = fx[f"s{step}/in/{name}"]
    out: dict = {}
    for name in MASS_OUTPUTS:
        out[name] = fx[f"s{step}/out/{name}"].reshape(nk, ncol)
    for name in FULL_OUTPUTS:
        out[name] = fx[f"s{step}/out/{name}"].reshape(nk + 1, ncol)
    for name in SURFACE_OUTPUTS + INT_SURFACE_OUTPUTS:
        out[name] = fx[f"s{step}/out/{name}"]
    a["out"] = out
    return a


def mismatches(got: dict, want: dict) -> dict:
    """``{field: differing word count}`` comparing BIT PATTERNS, not values.

    A float32 output is compared as its uint32 word, so -0 against +0 and a
    NaN payload both count; an int32 output is compared as itself.  The
    int32 jt2d/maxg2d/ideep2d are compared only where lengath2d is 1: the
    oracle writes -1 elsewhere because WRF leaves those words unassigned.
    """
    bad = {}
    convecting = np.asarray(want["lengath2d"]) != 0
    for name, w in want.items():
        g = np.asarray(got[name])
        w = np.asarray(w)
        if g.shape != w.shape:
            bad[name] = -1
            continue
        if w.dtype == np.float32:
            diff = g.astype(np.float32).view(np.uint32) != w.view(np.uint32)
        else:
            diff = g.astype(np.int32) != w
            if name in ("jt2d", "maxg2d", "ideep2d"):
                diff = diff & convecting
        n = int(np.count_nonzero(diff))
        if n:
            bad[name] = n
    return bad


def ulp_distance(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """float32 ULP distance (sign-magnitude ordered), NaN pairs counted 0."""
    def ordered(x):
        i = np.asarray(x, dtype=np.float32).view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)
    g, w = np.asarray(got, np.float32), np.asarray(want, np.float32)
    d = np.abs(ordered(g) - ordered(w))
    both_nan = np.isnan(g) & np.isnan(w)
    return np.where(both_nan, 0, d)
