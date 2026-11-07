"""Reader for the WRF v4.6.1 BouLac (bl_pbl_physics=8) column-oracle grids.

Written by ``tools/boulac_wrf461_oracle`` (see its README).  Each grid is
two files: ``<grid>.in.bin`` is make_cases.py's input (the static column,
the surface forcing and the BEP terms) and ``<grid>.rec.bin`` is
run_bl_boulac.F90's record of every step WRF's BOULAC ran on it: the step's
inputs (:data:`STEP_INPUTS`) and every output it wrote
(:data:`STEP_OUTPUTS`).  Level fields come back ``(nz, ncol)`` float32,
bottom-up -- the engine's ``(nz, ny, nx)`` layout with ``ny = 1`` -- and
surface fields ``(ncol,)``.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

#: The packaged fixtures of record.
FIXTURE_DIR = (Path(__file__).resolve().parents[2]
               / "tests" / "data" / "oracles" / "boulac")

MAGIC = 0xB0B1A8

#: make_cases.py's write order.
LEVEL_INPUTS = ("dz8w", "u", "v", "th", "qv", "qc", "rho", "tke")
SURFACE_INPUTS = ("hfx", "qfx", "ust", "frc_urb2d", "pblh")
BEP_LEVEL_INPUTS = ("a_u_bep", "a_v_bep", "a_t_bep", "a_q_bep", "a_e_bep",
                    "b_u_bep", "b_v_bep", "b_t_bep", "b_q_bep", "b_e_bep",
                    "dlg_bep", "dl_u_bep", "vl_bep")

#: run_bl_boulac.F90's per-step record order.
STEP_INPUTS = ("u", "v", "th", "qv", "qc", "tke", "dlg_bep")
STEP_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
                "tke", "el_pbl", "exch_h", "exch_m", "wu", "wv", "wt", "wq",
                "dlg_bep")


def _header(buf: bytes) -> dict:
    magic, nz, ncol, nsteps, flag_bep, idiff = struct.unpack_from(
        "<6i", buf, 0)
    if magic != MAGIC:
        raise ValueError("not a BouLac oracle file")
    dt, cp, g = struct.unpack_from("<3f", buf, 24)
    return dict(nz=nz, ncol=ncol, nsteps=nsteps, flag_bep=bool(flag_bep),
                idiff=idiff, dt=np.float32(dt), cp=np.float32(cp),
                g=np.float32(g))


class _Cursor:
    def __init__(self, buf: bytes, offset: int):
        self.buf = buf
        self.offset = offset

    def take(self, shape) -> np.ndarray:
        n = int(np.prod(shape))
        arr = np.frombuffer(self.buf, dtype="<f4", count=n,
                            offset=self.offset)
        self.offset += 4 * n
        return arr.reshape(shape).astype(np.float32)

    def done(self, path) -> None:
        if self.offset != len(self.buf):
            raise ValueError(f"{path}: {len(self.buf) - self.offset} "
                             "trailing bytes")


def read_inputs(path) -> dict:
    """The static grid: level inputs, surface inputs and the BEP terms."""
    buf = Path(path).read_bytes()
    out = _header(buf)
    nz, nc = out["nz"], out["ncol"]
    cur = _Cursor(buf, 36)
    for name in LEVEL_INPUTS:
        out[name] = cur.take((nz, nc))
    for name in SURFACE_INPUTS:
        out[name] = cur.take((nc,))
    for name in BEP_LEVEL_INPUTS:
        out[name] = cur.take((nz, nc))
    out["sf_bep"] = cur.take((nz + 1, nc))
    cur.done(path)
    return out


def read_record(path) -> dict:
    """Every step WRF ran: ``steps[i]["in"]`` and ``steps[i]["out"]``."""
    buf = Path(path).read_bytes()
    out = _header(buf)
    nz, nc = out["nz"], out["ncol"]
    cur = _Cursor(buf, 36)
    steps = []
    for _ in range(out["nsteps"]):
        step = {"in": {}, "out": {}}
        for name in STEP_INPUTS:
            step["in"][name] = cur.take((nz, nc))
        step["in"]["pblh"] = cur.take((nc,))
        for name in STEP_OUTPUTS:
            step["out"][name] = cur.take((nz, nc))
        step["out"]["pblh"] = cur.take((nc,))
        steps.append(step)
    cur.done(path)
    out["steps"] = steps
    return out


def grids(fixture_dir=FIXTURE_DIR) -> list[dict]:
    """The grid index make_cases.py wrote (name, shape, column labels)."""
    return json.loads((Path(fixture_dir) / "cases-index.json")
                      .read_text(encoding="utf-8"))


def load_grid(name: str, fixture_dir=FIXTURE_DIR) -> tuple[dict, dict]:
    """``(inputs, record)`` of one grid."""
    base = Path(fixture_dir)
    return (read_inputs(base / f"{name}.in.bin"),
            read_record(base / f"{name}.rec.bin"))
