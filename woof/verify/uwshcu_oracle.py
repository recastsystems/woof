"""Reader and grader for the WRF v4.6.1 UW shallow cumulus oracle fixtures.

The fixtures are written by ``tools/uwshcu_wrf461_oracle`` (see its
README): each ``<stem>.bin`` is raw little-endian words and
``<stem>.manifest`` names them, one line per array, ``name dtype count
byte_offset`` (the UW PBL oracle's writer, reused).

* ``cases-<grid>``: the full WRF driver, camuwshcu_driver, over several
  consecutive steps.  ``const/...`` holds the run's constants and gestbl's
  saturation table; ``s<n>/...`` every step's exact float32 inputs and every
  output the driver wrote.  Per-column arrays are stored column by column,
  so :func:`step_arrays` returns them shaped ``(ncol, nlev)``.
* ``core-<grid>``: compute_uwshcu_inv alone on binary64 inputs, ``in/...``
  and ``out/...`` in CAM orientation (level 1 = model top).

:func:`grade` compares exact bit patterns, including signed zeros. Shape
and dtype changes and nonfinite values fail the comparison. ULP distance
is diagnostic information only and never grants a passing grade.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DTYPES = {"f8": "<f8", "f4": "<f4", "i4": "<i4"}

#: Driver step inputs and outputs as run_uwshcu.F90 writes them.
STEP_MASS_INPUTS = ("u", "v", "t", "qv", "qc", "qi", "qnc", "qni", "p", "z",
                    "pi", "cldfra", "cldfra_old", "dlf_in")
STEP_FULL_INPUTS = ("p8w", "z_at_w", "tke_pbl", "cmfmc_in")
STEP_SURFACE_INPUTS = ("pblh", "ht", "cush_in", "cutop_in", "cubot_in",
                       "rliq_in")
STEP_MASS_OUTPUTS = ("rushten", "rvshten", "rthshten", "rqvshten",
                     "rqcshten", "rqishten", "rqcnshten", "rqinshten", "dlf",
                     "dlf2", "shfrc3d", "evapcsh", "icwmrsh", "rprdsh")
STEP_FULL_OUTPUTS = ("cmfmc", "cmfmc2", "cmfsl", "cmflq")
STEP_SURFACE_OUTPUTS = ("cush", "pratesh", "cutop", "cubot", "rliq", "rliq2")

#: compute_uwshcu_inv's outputs as run_uwshcu_core.F90 writes them.
CORE_FULL_OUTPUTS = ("umf", "slflx", "qtflx", "flxprc1", "flxsnow1")
CORE_MASS_OUTPUTS = ("qvten", "qlten", "qiten", "sten", "uten", "vten",
                     "qrten", "qsten", "evapc", "cufrc", "qcu", "qlu", "qiu",
                     "qc")
CORE_SURFACE_OUTPUTS = ("precip", "snow", "cbmf", "rliq", "cnt", "cnb",
                        "cush")


@dataclass(frozen=True)
class Fixture:
    """One oracle stream: every named array, read eagerly."""

    path: Path
    arrays: dict

    def __getitem__(self, name):
        return self.arrays[name]

    def __contains__(self, name):
        return name in self.arrays


def load(stem) -> Fixture:
    """Read ``<stem>.bin`` through ``<stem>.manifest``."""
    stem = Path(stem)
    if stem.suffix in (".bin", ".manifest"):
        stem = stem.with_suffix("")
    raw = Path(str(stem) + ".bin").read_bytes()
    arrays = {}
    for line in Path(str(stem) + ".manifest").read_text().splitlines():
        name, dtype, count, offset = line.split()
        arrays[name] = np.frombuffer(raw, _DTYPES[dtype], int(count),
                                     int(offset)).copy()
    return Fixture(stem, arrays)


def driver_dims(fx: Fixture) -> tuple[int, int, int]:
    """``(ncol, nk, nsteps)`` of a driver fixture."""
    return (int(fx["const/ncol"][0]), int(fx["const/nk"][0]),
            int(fx["const/nsteps"][0]))


def step_arrays(fx: Fixture, step: int) -> dict:
    """One step's inputs and outputs, per column: ``(ncol, nlev)``."""
    ncol, nk, _ = driver_dims(fx)
    pre = f"s{step}/"
    out = {}
    for name in STEP_MASS_INPUTS + STEP_MASS_OUTPUTS:
        out[name] = fx[pre + name].reshape(ncol, nk)
    for name in STEP_FULL_INPUTS + STEP_FULL_OUTPUTS:
        out[name] = fx[pre + name].reshape(ncol, nk + 1)
    for name in STEP_SURFACE_INPUTS + STEP_SURFACE_OUTPUTS:
        out[name] = fx[pre + name].reshape(ncol)
    out["dt"] = float(fx["const/dt"][0])
    out["itimestep"] = int(fx[pre + "itimestep"][0])
    out["has_qnc"] = bool(fx["const/has_qnc"][0])
    out["has_qni"] = bool(fx["const/has_qni"][0])
    return out


def core_arrays(fx: Fixture) -> dict:
    """compute_uwshcu_inv inputs and outputs: ``(nlev, ncol)``, row 0 the
    model top; tracers split into the NUMLIQ (``trl``) and NUMICE
    (``tri``) constituents, the ones whose tendencies leave the scheme."""
    ncol = int(fx["in/ncol"][0])
    nk = int(fx["in/nk"][0])
    out = {"ncol": ncol, "nk": nk, "dt": float(fx["in/dt"][0])}
    for name in ("p0", "z0", "dp0", "u0", "v0", "qv0", "ql0", "qi0", "t0"):
        out[name] = np.ascontiguousarray(fx["in/" + name].reshape(ncol, nk).T)
    for name in ("ps0", "zs0", "tke"):
        out[name] = np.ascontiguousarray(
            fx["in/" + name].reshape(ncol, nk + 1).T)
    tr0 = fx["in/tr0"].reshape(ncol, 5, nk)
    out["trl"] = np.ascontiguousarray(tr0[:, 3, :].T)
    out["tri"] = np.ascontiguousarray(tr0[:, 4, :].T)
    out["pblh"] = fx["in/pblh"].copy()
    out["cush"] = fx["in/cush_in"].copy()
    for name in CORE_FULL_OUTPUTS:
        out["out/" + name] = fx["out/" + name].reshape(ncol, nk + 1).T
    for name in CORE_MASS_OUTPUTS:
        out["out/" + name] = fx["out/" + name].reshape(ncol, nk).T
    trten = fx["out/trten"].reshape(ncol, 5, nk)
    out["out/trten123"] = trten[:, :3, :]
    out["out/trtl"] = trten[:, 3, :].T
    out["out/trti"] = trten[:, 4, :].T
    for name in CORE_SURFACE_OUTPUTS:
        out["out/" + name] = fx["out/" + name].copy()
    return out


def ulp_distance(actual, expected) -> np.ndarray:
    """Elementwise ordered-word distance, including the sign of zero."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("output shape or dtype differs")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite output")
    if expected.dtype == np.float64:
        word, sign = np.uint64, np.uint64(1 << 63)
    elif expected.dtype == np.float32:
        word, sign = np.uint32, np.uint32(1 << 31)
    else:
        if not np.issubdtype(expected.dtype, np.integer):
            raise ValueError("unsupported output dtype")
        return (actual != expected).astype(np.uint64)
    a, b = actual.view(word), expected.view(word)
    a = np.where(a & sign, ~a, a ^ sign).astype(np.uint64)
    b = np.where(b & sign, ~b, b ^ sign).astype(np.uint64)
    return np.maximum(a, b) - np.minimum(a, b)


def grade(actual, expected) -> tuple[int, int, int]:
    """``(mismatching words, max ULP, sign-of-zero-only words)``."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    d = ulp_distance(actual, expected)
    signed_zero = int(np.count_nonzero(
        (actual == 0) & (expected == 0)
        & (np.signbit(actual) != np.signbit(expected))))
    return int(np.count_nonzero(d)), int(d.max(initial=0)), signed_zero
