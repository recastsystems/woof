"""The packaged WRF v4.6.1 MRF column oracle (``bl_pbl_physics = 99``).

``tools/mrf_wrf461_oracle`` links its driver against the ``module_bl_mrf.o``
WRF's own build produced (gfortran 13.3.0, the scalar referee's FCOPTIM),
proves a rebuild of the pinned source at those flags writes the same bytes,
and packs every input and output word of every step into the two CSV files
read here.  Every float is its 8-hex-digit binary32 word, so signed zeros,
subnormals and NaN payloads survive; KPBL is decimal and is WRF's own
TOP-DOWN index (KPBL2D = KPBL0, module_bl_mrf.F:1284), which
:func:`wrf_kpbl_bottom_up` maps onto the bottom-up level the port
publishes.

This module imports nothing but numpy, so CPU gates can read the fixture on
a box with no GPU runtime.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from woof import data_assets

import numpy as np

MRF_ORACLE_DIR = data_assets.verify_oracle_path("mrf/oracle")

LEVEL_INPUTS = ("u", "v", "t", "qv", "qc", "qi", "p", "pi", "dz8w", "z")
LEVEL_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
                 "rqiblten")
SURFACE_INPUTS = ("dt", "xland", "psfc", "znt", "ust_in", "psim", "psih",
                  "hfx", "qfx", "tsk", "gz1oz0", "wspd_in", "br", "zol_in",
                  "regime_in", "hol_in", "pbl_in")
SURFACE_OUTPUTS = ("ust", "hol", "pbl", "zol", "regime", "wspd")

#: The pinned module and fixture words (oracle-sha256sums.txt), restated
#: so an edited fixture cannot pass by also editing its own receipt.
MODULE_SHA256 = ("a1306d0063043d001fa4a53b9219031b931999c7d7daea172fd97518"
                 "bc9b19eb")


def _words(column: list[str]) -> np.ndarray:
    return np.array([int(x, 16) for x in column],
                    dtype=np.uint32).view(np.float32)


@dataclass(frozen=True)
class MrfOracle:
    """The fixture as arrays.

    ``levels[name]`` is ``(nstep, nz, ncase)`` float32 with k bottom-up (the
    engine's order); ``surface[name]`` is ``(nstep, ncase)`` float32;
    ``kpbl`` is WRF's top-down word, ``(nstep, ncase)`` int32;
    ``flag_qi`` is ``(ncase,)`` bool (constant over steps).
    """

    nz: int
    ncase: int
    nstep: int
    cases: tuple[dict, ...]
    levels: dict
    surface: dict
    kpbl: np.ndarray
    flag_qi: np.ndarray


@lru_cache(maxsize=1)
def load_mrf_oracle(directory: Path = MRF_ORACLE_DIR) -> MrfOracle:
    data_assets.require_verify_oracle(directory)
    meta = json.loads((directory / "mrf-cases.json").read_text("utf-8"))
    nz, nstep = int(meta["nz"]), int(meta["nstep"])
    cases = tuple(meta["cases"])
    ncase = len(cases)
    with open(directory / "mrf-levels.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    if len(body) != nstep * ncase * nz:
        raise ValueError("mrf-levels.csv row count does not match the cases")
    cols = list(zip(*body))
    case = np.array(cols[0], dtype=np.int64)
    step = np.array(cols[1], dtype=np.int64)
    k = np.array(cols[2], dtype=np.int64)
    levels = {}
    for name in LEVEL_INPUTS + LEVEL_OUTPUTS:
        arr = np.empty((nstep, nz, ncase), dtype=np.float32)
        arr[step, k, case] = _words(list(cols[head.index(name)]))
        levels[name] = arr
    with open(directory / "mrf-surface.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    if len(body) != nstep * ncase:
        raise ValueError("mrf-surface.csv row count does not match the cases")
    cols = list(zip(*body))
    case = np.array(cols[0], dtype=np.int64)
    step = np.array(cols[1], dtype=np.int64)
    surface = {}
    for name in SURFACE_INPUTS + SURFACE_OUTPUTS:
        arr = np.empty((nstep, ncase), dtype=np.float32)
        arr[step, case] = _words(list(cols[head.index(name)]))
        surface[name] = arr
    kpbl = np.empty((nstep, ncase), dtype=np.int32)
    kpbl[step, case] = np.array(cols[head.index("kpbl")], dtype=np.int32)
    flag = np.empty((nstep, ncase), dtype=np.int32)
    flag[step, case] = np.array(cols[head.index("flag_qi")], dtype=np.int32)
    if not np.all(flag == flag[0]):
        raise ValueError("FLAG_QI changes between steps of one case")
    return MrfOracle(nz=nz, ncase=ncase, nstep=nstep, cases=cases,
                     levels=levels, surface=surface, kpbl=kpbl,
                     flag_qi=flag[0].astype(bool))


def wrf_kpbl_bottom_up(kpbl_wrf: np.ndarray, nz: int) -> np.ndarray:
    """WRF MRF's top-down KPBL2D word as the bottom-up level (kme - KPBL0)."""
    return (nz + 1 - np.asarray(kpbl_wrf)).astype(np.int32)


def fixture_sha256(directory: Path = MRF_ORACLE_DIR) -> dict[str, str]:
    """sha256 of every packaged fixture file, LF-normalised."""
    data_assets.require_verify_oracle(directory)
    out = {}
    for name in ("mrf-levels.csv", "mrf-surface.csv", "mrf-cases.json",
                 "build-receipt.txt"):
        data = (directory / name).read_bytes().replace(b"\r\n", b"\n")
        out[name] = hashlib.sha256(data).hexdigest()
    return out


def run_mrf_on_oracle(oracle: MrfOracle | None = None, *, kernel=None,
                      steps=None) -> dict[str, np.ndarray]:
    """Launch the CUDA kernel on every recorded step's INPUTS.

    Each step is fed the inputs WRF saw at that step (the fixture records
    them), so a default-arithmetic difference at one step never leaks into
    the next.  Cases are grouped by (dt, FLAG_QI) because both are scalar
    arguments of one launch; every group is a ``(nz, 1, n)`` domain.

    Returns host arrays shaped like the fixture: the six tendencies
    ``(nstep, nz, ncase)``, ``ust``/``hol``/``pbl`` ``(nstep, ncase)``
    float32 and ``kpbl`` ``(nstep, ncase)`` int32 (bottom-up, as published).
    """
    import cupy as cp

    from woof.core.mrf import launch_mrf

    o = oracle if oracle is not None else load_mrf_oracle()
    steps = range(o.nstep) if steps is None else steps
    out = {name: np.full((o.nstep, o.nz, o.ncase), np.nan, np.float32)
           for name in LEVEL_OUTPUTS}
    out.update({name: np.full((o.nstep, o.ncase), np.nan, np.float32)
                for name in ("ust", "hol", "pbl")})
    out["kpbl"] = np.zeros((o.nstep, o.ncase), np.int32)
    for s in steps:
        dts = o.surface["dt"][s]
        groups: dict[tuple[int, bool], list[int]] = {}
        for c in range(o.ncase):
            key = (int(dts[c].view(np.uint32)), bool(o.flag_qi[c]))
            groups.setdefault(key, []).append(c)
        for (dt_bits, flag), idx in sorted(groups.items()):
            sel = np.array(idx)

            def lev(name):
                return cp.asarray(np.ascontiguousarray(
                    o.levels[name][s][:, sel][:, None, :]))

            def sfc(name):
                return cp.asarray(np.ascontiguousarray(
                    o.surface[name][s][sel][None, :]))

            ust = sfc("ust_in")
            res = launch_mrf(
                *(lev(n) for n in LEVEL_INPUTS),
                psfc=sfc("psfc"), ust=ust, psim=sfc("psim"),
                psih=sfc("psih"), xland=sfc("xland"), hfx=sfc("hfx"),
                qfx=sfc("qfx"), gz1oz0=sfc("gz1oz0"), wspd=sfc("wspd_in"),
                br=sfc("br"),
                dt=float(np.uint32(dt_bits).view(np.float32)),
                flag_qi=flag, kernel=kernel)
            for name in LEVEL_OUTPUTS:
                out[name][s][:, sel] = cp.asnumpy(res[name])[:, 0, :]
            out["ust"][s][sel] = cp.asnumpy(ust)[0]
            out["hol"][s][sel] = cp.asnumpy(res["hol"])[0]
            out["pbl"][s][sel] = cp.asnumpy(res["pblh"])[0]
            out["kpbl"][s][sel] = cp.asnumpy(res["kpbl"])[0]
    return out
