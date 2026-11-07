"""The TEMF column oracle's file layout, reader and word comparison.

``tools/temf_wrf461_oracle/run_temf.F90`` steps WRF v4.6.1's TEMF surface
layer (sf_sfclay_physics = 10) and PBL (bl_pbl_physics = 10) over the
columns ``make_columns.py`` writes, and records every output of every
step.  This module is the one description of both files, read by the
oracle's own arm comparison, by the WOOF replay
(``tools/temf_wrf461_oracle/run_woof.py``) and by the parity tests.  It
needs NumPy only.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = 1413827910           # b"FMET" little-endian: 'TEMF'
VERSION = 1

#: Surface-layer outputs, as the surface layer leaves them each step.
FIELDS_SFC2D = ("hfx", "qfx", "lh", "flhc", "flqc", "qgh", "qsfc", "ust_sfc",
                "chs", "chs2", "cqs2", "cpm", "u10_sfc", "v10_sfc", "th2",
                "t2_sfc", "q2", "exch_temf", "wm_temf")
#: PBL tendencies and the carried 3-D TEMF fields, after the PBL call.
FIELDS_PBL3D = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
                "te_temf", "kh_temf", "km_temf", "shf_temf", "qf_temf",
                "uw_temf", "vw_temf", "wupd_temf", "mf_temf", "thup_temf",
                "qtup_temf", "qlup_temf", "cf3d_temf")
#: PBL 2-D outputs; ``kpbl`` is an int32 word.
FIELDS_PBL2D = ("hpbl", "kpbl", "u10", "v10", "t2", "ust", "hd_temf",
                "lcl_temf", "hct_temf", "cfm_temf")
#: The column after the step's tendencies are applied.
FIELDS_STATE = ("u", "v", "th", "qv", "qc")

#: Static 2-D inputs, in file order.
INPUT_2D = ("ht", "xland", "mavail", "znt", "fcor", "psfc", "hfx0", "qfx0",
            "hpbl0")
#: Mass-level profiles, in file order.
INPUT_3D = ("u", "v", "th", "qv", "qc", "qi", "p", "pi", "rho")
#: (nz+1)-level profiles: the mass heights (top word unused) and P8W.
INPUT_FULL = ("z", "p8w")


@dataclass(frozen=True)
class Cases:
    ncol: int
    nz: int
    nsteps: int
    dt: np.float32
    fields: dict            # name -> float32 array, (ncol,), (nz, ncol) ...
    tsk: np.ndarray         # (nsteps, ncol)


def read_cases(path) -> Cases:
    raw = Path(path).read_bytes()
    hdr = struct.unpack_from("<8i", raw, 0)
    if hdr[0] != MAGIC or hdr[1] != VERSION:
        raise ValueError(f"{path} is not a TEMF case file (header {hdr[:2]})")
    ncol, nz, nsteps = hdr[2], hdr[3], hdr[4]
    dt = np.frombuffer(raw, "<f4", 1, 32)[0]
    words = np.frombuffer(raw, "<f4", offset=36).astype(np.float32)
    pos = 0
    fields = {}
    for name in INPUT_2D:
        fields[name] = words[pos:pos + ncol].copy(); pos += ncol
    tsk = words[pos:pos + ncol * nsteps].reshape(nsteps, ncol).copy()
    pos += ncol * nsteps
    for name in INPUT_3D:
        fields[name] = words[pos:pos + ncol * nz].reshape(nz, ncol).copy()
        pos += ncol * nz
    for name in INPUT_FULL:
        fields[name] = words[pos:pos + ncol * (nz + 1)].reshape(
            nz + 1, ncol).copy()
        pos += ncol * (nz + 1)
    if pos != words.size:
        raise ValueError(f"{path}: {words.size - pos} trailing words")
    return Cases(ncol, nz, nsteps, np.float32(dt), fields, tsk)


def step_words(ncol: int, nz: int) -> int:
    return ((len(FIELDS_SFC2D) + len(FIELDS_PBL2D)) * ncol
            + (len(FIELDS_PBL3D) + len(FIELDS_STATE)) * nz * ncol)


def read_outputs(path, cases: Cases) -> dict:
    """``{field: array (nsteps, ncol) or (nsteps, nz, ncol)}`` of raw words.

    Floating fields come back as float32, ``kpbl`` as int32; both are the
    file's words unchanged.
    """
    ncol, nz, nsteps = cases.ncol, cases.nz, cases.nsteps
    raw = np.fromfile(path, dtype="<u4")
    per = step_words(ncol, nz)
    if raw.size != per * nsteps:
        raise ValueError(f"{path}: {raw.size} words, expected {per * nsteps}")
    raw = raw.reshape(nsteps, per)
    out = {}
    pos = 0
    for name in FIELDS_SFC2D:
        out[name] = raw[:, pos:pos + ncol]; pos += ncol
    for name in FIELDS_PBL3D:
        out[name] = raw[:, pos:pos + nz * ncol].reshape(nsteps, nz, ncol)
        pos += nz * ncol
    for name in FIELDS_PBL2D:
        out[name] = raw[:, pos:pos + ncol]; pos += ncol
    for name in FIELDS_STATE:
        out[name] = raw[:, pos:pos + nz * ncol].reshape(nsteps, nz, ncol)
        pos += nz * ncol
    return {name: (value.astype(np.uint32).view(np.int32) if name == "kpbl"
                   else value.astype(np.uint32).view(np.float32))
            for name, value in out.items()}


def ulp_distance(a, b) -> np.ndarray:
    """Units in the last place between two float32 arrays, as int64.

    Equal words (and two NaNs) are 0; +0 and -0 are 0 apart; a NaN against
    a number is reported as 2**31.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    def ordered(x):
        i = x.view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)

    d = np.abs(ordered(a) - ordered(b))
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    d = np.where(nan_a & nan_b, 0, d)
    d = np.where(nan_a ^ nan_b, 2 ** 31, d)
    return d


def compare(ref: dict, got: dict, fields=None) -> dict:
    """Per field: mismatched words, total words, max ULP, first mismatch."""
    report = {}
    for name in (fields or ref):
        a, b = ref[name], got[name]
        if name == "kpbl":
            bad = a != b
            dist = np.where(bad, np.abs(a.astype(np.int64) - b), 0)
        else:
            dist = ulp_distance(a, b)
            bad = a.view(np.uint32) != b.view(np.uint32)
        first = None
        if bad.any():
            idx = tuple(int(i) for i in np.argwhere(bad)[0])
            first = {"index": idx, "ref": float(a[idx]), "got": float(b[idx])}
        report[name] = {"mismatched": int(bad.sum()), "words": int(bad.size),
                        "max_ulp": int(dist.max()) if dist.size else 0,
                        "first": first}
    return report


ALL_FIELDS = FIELDS_SFC2D + FIELDS_PBL3D + FIELDS_PBL2D + FIELDS_STATE


def step_digests(out: dict) -> dict:
    """``{field: [sha256 of step n's little-endian words, ...]}``.

    Two outputs agree word for word exactly when every digest agrees (NaN
    payloads included, which is stricter than :func:`compare`).
    """
    import hashlib

    digests = {}
    for name in ALL_FIELDS:
        arr = out[name]
        kind = "<i4" if arr.dtype == np.int32 else "<f4"
        digests[name] = [
            hashlib.sha256(np.ascontiguousarray(arr[n]).astype(kind)
                           .tobytes()).hexdigest()
            for n in range(arr.shape[0])]
    return digests
