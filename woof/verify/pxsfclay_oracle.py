"""Replay the PX surface-layer oracle's columns on the device and grade them.

The oracle (``tools/pxsfclay_wrf461_oracle``) writes two little-endian
files per configuration: the input (``<name>.in.bin``, built by
``make_cases.py``) and WRF's words (``<name>.out.bin``, from
``run_pxsfclay``, the unmodified Fortran).  :func:`replay` drives
:mod:`woof.core.pxsfclay` through exactly the step loop the Fortran
driver runs -- the same carried INOUT state, the same land-model overrides
between calls, the same zeroed OUT words before each call -- and
:func:`grade` compares every word.

The packaged fixtures of record live in ``woof/data/pxsfclay/oracle``
with their SHA-256 manifest; ``tests/test_pxsfclay_wrf461_parity.py``
replays them on the card.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from woof import data_assets

import numpy as np

from woof.core.physics_inventory import (PX_SFCLAY_INPUTS, PX_SFCLAY_SEA,
                                          PX_SFCLAY_STATE)

MAGIC_IN = 0x50585346
MAGIC_OUT = 0x5058534F
#: The Fortran driver's carried state, in its file order.
STATE_FIELDS = (*PX_SFCLAY_STATE[:-2], "sst")
FORCING_FIELDS = ("u1", "v1", "t1", "th1", "qv1", "p1", "dz8w", "psfc",
                  "tsk", "pblh", "mavail", "xland", "xice")
OVERRIDE_FIELDS = ("flag", "qsfc", "hfx", "qfx", "znt")
#: run_pxsfclay.F90's output order: 23 state words, U10, V10, 12 sea words.
OUT_FIELDS = (*STATE_FIELDS, "u10", "v10",
              *(f"{name}_sea" for name in PX_SFCLAY_SEA))
SEA_OUT = frozenset(f"{name}_sea" for name in PX_SFCLAY_SEA)
FIXTURE_DIR = data_assets.verify_oracle_path("pxsfclay/oracle")

assert len(STATE_FIELDS) == 23 and len(OUT_FIELDS) == 37
assert FORCING_FIELDS[:12] == PX_SFCLAY_INPUTS


@dataclass(frozen=True)
class Case:
    """One oracle configuration's input file, decoded."""

    name: str
    ncol: int
    nstep: int
    mode: int
    itstart: int
    tice2tsk: bool
    dx: float
    xice_threshold: float
    state0: np.ndarray            # (23, ncol)
    forcing: np.ndarray           # (nstep, 13, ncol)
    override: np.ndarray          # (nstep, 5, ncol)


def read_case(path: Path) -> Case:
    raw = Path(path).read_bytes()
    head = struct.unpack_from("<8i", raw, 0)
    if head[0] != MAGIC_IN or head[1] != 1:
        raise ValueError(f"{path} is not a PX oracle input file")
    _, _, ncol, nstep, mode, itstart, tice, _ = head
    dx, thr = struct.unpack_from("<2f", raw, 32)
    data = np.frombuffer(raw, dtype="<f4", offset=40)
    nstate = len(STATE_FIELDS) * ncol
    per_step = (len(FORCING_FIELDS) + len(OVERRIDE_FIELDS)) * ncol
    if data.size != nstate + nstep * per_step:
        raise ValueError(f"{path} has {data.size} words, expected "
                         f"{nstate + nstep * per_step}")
    state0 = data[:nstate].reshape(len(STATE_FIELDS), ncol)
    steps = data[nstate:].reshape(nstep, len(FORCING_FIELDS)
                                  + len(OVERRIDE_FIELDS), ncol)
    name = Path(path).name.split(".")[0]
    return Case(name, ncol, nstep, mode, itstart, bool(tice), float(dx),
                float(thr), state0.astype(np.float32),
                steps[:, :len(FORCING_FIELDS)].astype(np.float32),
                steps[:, len(FORCING_FIELDS):].astype(np.float32))


def read_words(path: Path, case: Case) -> np.ndarray:
    """WRF's words: (nstep, 37, ncol) float32."""
    raw = Path(path).read_bytes()
    magic, ncol, nstep, nout = struct.unpack_from("<4i", raw, 0)
    if (magic, ncol, nstep, nout) != (MAGIC_OUT, case.ncol, case.nstep,
                                      len(OUT_FIELDS)):
        raise ValueError(f"{path} does not belong to case {case.name}")
    return np.frombuffer(raw, dtype="<f4", offset=16).reshape(
        nstep, nout, ncol).astype(np.float32)


def ice_mask(case: Case, step: int) -> np.ndarray:
    """Cells the wrapper's open-water call covers on ``step`` (0-based)."""
    xice = case.forcing[step, FORCING_FIELDS.index("xice")]
    return (xice >= np.float32(case.xice_threshold)) & (xice <= np.float32(1.0))


def replay(case: Case, *, module=None) -> np.ndarray:
    """Run the case on the current device; returns (nstep, 37, ncol)."""
    import cupy as cp

    from woof.core.pxsfclay import launch_pxsfclay, launch_pxsfclay_seaice

    n = case.ncol
    state = {name: cp.asarray(case.state0[k].copy())
             for k, name in enumerate(STATE_FIELDS)}
    state["u10"] = cp.zeros(n, cp.float32)
    state["v10"] = cp.zeros(n, cp.float32)
    sst = state.pop("sst")
    out = np.zeros((case.nstep, len(OUT_FIELDS), n), np.float32)
    for step in range(case.nstep):
        itimestep = case.itstart + step
        forcing = {name: cp.asarray(case.forcing[step, k].copy())
                   for k, name in enumerate(FORCING_FIELDS)}
        if step > 0:
            ovr = case.override[step]
            on = cp.asarray(ovr[0] > np.float32(0.5))
            for k, name in enumerate(OVERRIDE_FIELDS[1:], start=1):
                state[name][...] = cp.where(on, cp.asarray(ovr[k]),
                                            state[name])
        state["u10"].fill(0.0)
        state["v10"].fill(0.0)
        sea = {name: cp.zeros(n, cp.float32) for name in PX_SFCLAY_SEA}
        inputs = {name: forcing[name] for name in PX_SFCLAY_INPUTS}
        if case.mode == 0:
            launch_pxsfclay(inputs, state, dx=case.dx, itimestep=itimestep,
                            module=module)
        else:
            launch_pxsfclay_seaice(
                inputs, state, sea, xice=forcing["xice"], sst=sst,
                dx=case.dx, itimestep=itimestep,
                xice_threshold=case.xice_threshold,
                tice2tsk_if2cold=case.tice2tsk, module=module)
        for k, name in enumerate(OUT_FIELDS):
            if name == "sst":
                array = sst
            elif name in SEA_OUT:
                array = sea[name[:-4]]
            else:
                array = state[name]
            out[step, k] = cp.asnumpy(array)
    return out


def _ordered(bits: np.ndarray) -> np.ndarray:
    """Map float32 bit patterns onto a monotone integer line."""
    b = bits.astype(np.int64)
    return np.where(b < 0x80000000, b + 0x80000000, 0xFFFFFFFF - b)


def grade(case: Case, device: np.ndarray, oracle: np.ndarray) -> dict:
    """Per-field differing words and max ULP over the graded words.

    Every word of every step is graded, except the open-water ``*_sea``
    words outside the wrapper's ice range (and all of them in mode 0):
    WRF's second call computes those on every cell, but no consumer reads
    them off the ice range, and the kernel does not write them there.
    """
    if device.shape != oracle.shape:
        raise ValueError("device and oracle shapes differ")
    report = {}
    for k, name in enumerate(OUT_FIELDS):
        if name in SEA_OUT:
            if case.mode == 0:
                continue
            mask = np.stack([ice_mask(case, s) for s in range(case.nstep)])
        else:
            mask = np.ones((case.nstep, case.ncol), bool)
        a = device[:, k][mask].view(np.uint32)
        b = oracle[:, k][mask].view(np.uint32)
        diff = a != b
        ulp = np.abs(_ordered(a) - _ordered(b))
        report[name] = {"words": int(mask.sum()),
                        "differing": int(diff.sum()),
                        "max_ulp": int(ulp.max()) if ulp.size else 0}
    return report


def fixture_manifest(directory: Path = FIXTURE_DIR) -> dict:
    """SHA-256 of every packaged fixture file, as MANIFEST.json records."""
    data_assets.require_verify_oracle(directory)
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    for name, digest in manifest["sha256"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"fixture {name} hashes {actual}, the manifest "
                             f"pins {digest}")
    return manifest


def fixture_cases(directory: Path = FIXTURE_DIR) -> list[tuple[Case, Path]]:
    manifest = fixture_manifest(directory)
    return [(read_case(directory / f"{name}.in.bin"),
             directory / f"{name}.out.bin")
            for name in manifest["configurations"]]


__all__ = ["Case", "FIXTURE_DIR", "OUT_FIELDS", "fixture_cases",
           "fixture_manifest", "grade", "ice_mask", "read_case",
           "read_words", "replay"]
