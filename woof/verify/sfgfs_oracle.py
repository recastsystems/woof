"""The GFS surface-layer oracle fixture (tools/sfgfs_wrf461_oracle) as arrays.

``tests/data/oracles/sfgfs`` holds what ``build.sh`` wrote on the sprint box:
the case inputs (``cases.bin``), four output arms of the byte-unmodified WRF
v4.6.1 ``module_sf_gfs.F`` (``armA``..``armD.bin``, field order in
``fields.txt``), and the saturation table GFUNCPHYS built
(``fpvs_table.bin``) with an FPVS sweep (``fpvs_sweep.bin``).  This module
reads them and runs :mod:`woof.verify.sfgfs_ref` over the same steps, so a
test or a validation script can compare word for word.

Arms (run_sf_gfs.F90):

* ``A`` SF_GFS called directly (FRACTIONAL_SEAICE = 0), ISFFLX = 1
* ``B`` the FRACTIONAL_SEAICE wrapper as WRF v4.6.1 ships it
* ``C`` the same wrapper with the open-water call given XLAND_SEA -- the
  call WOOF makes (the declared divergence)
* ``D`` SF_GFS called directly with ISFFLX = 0
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from woof.verify import sfgfs_ref as ref
from woof.core.fp32_ulp import monotone_fp32_key

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data"
               / "oracles" / "sfgfs")
MAGIC = 0x47465331
STEP_FIELDS = ("u1", "v1", "t1", "qv1", "p1", "psfc", "tsk", "xland", "xice")
INITIAL_FIELDS = ("znt", "ust", "sst")
#: XICE_THRESHOLD of the wrapper arms (FRACTIONAL_SEAICE = 1).
WRAPPER_XICE_THRESHOLD = np.float32(0.02)

#: SHA-256 of fpvs_table.bin: tbpvs(1:7501), c1xpvs, c2xpvs as binary64,
#: little-endian, read out of module_gfs_funcphys's own storage after
#: GFUNCPHYS ran.  woof.core.sf_gfs.FPVS_TABLE_SHA256 is the same digest.
FPVS_TABLE_SHA256 = (
    "bef4a54b636c57f74951560d3f5f0a65e12337cbedeae1ac17be1376f66e2235")


@dataclass(frozen=True)
class SfgfsOracle:
    ncol: int
    nstep: int
    per_step: np.ndarray        # (nstep, len(STEP_FIELDS), ncol) float32
    initial: np.ndarray         # (len(INITIAL_FIELDS), ncol) float32
    direct_fields: tuple[str, ...]
    wrapper_fields: tuple[str, ...]
    arms: dict                  # name -> (nstep, nfield, ncol) float32
    fpvs_table: np.ndarray      # (7503,) float64
    sweep_t: np.ndarray         # float32
    sweep_fpvs: np.ndarray      # float64

    def fields(self, arm: str) -> tuple[str, ...]:
        return (self.wrapper_fields if arm in ("B", "C")
                else self.direct_fields)

    def field(self, arm: str, name: str) -> np.ndarray:
        """(nstep, ncol) words of one output of one arm."""
        return self.arms[arm][:, self.fields(arm).index(name), :]

    def column(self, step: int, i: int) -> dict:
        """Step ``step`` (0-based) inputs of column ``i``, float32."""
        return {name: np.float32(self.per_step[step, f, i])
                for f, name in enumerate(STEP_FIELDS)}

    def step_inputs(self, step: int) -> dict:
        return {name: np.ascontiguousarray(self.per_step[step, f, :])
                for f, name in enumerate(STEP_FIELDS)}


@lru_cache(maxsize=4)
def load_sfgfs_oracle(directory: Path | None = None) -> SfgfsOracle:
    root = FIXTURE_DIR if directory is None else Path(directory)
    raw = (root / "cases.bin").read_bytes()
    magic, ncol, nstep, nin, ninit = struct.unpack_from("<5i", raw, 0)
    if magic != MAGIC or nin != len(STEP_FIELDS) or ninit != len(
            INITIAL_FIELDS):
        raise ValueError(f"{root / 'cases.bin'} is not a GFS1 case file")
    body = np.frombuffer(raw, dtype="<f4", offset=20)
    per_step = body[:nstep * nin * ncol].reshape(nstep, nin, ncol)
    initial = body[nstep * nin * ncol:].reshape(ninit, ncol)
    lines = (root / "fields.txt").read_text(encoding="utf-8").split("\n")
    direct = tuple(lines[0].split()[1:])
    wrapper = tuple(lines[1].split()[1:])
    arms = {}
    for arm in "ABCD":
        names = wrapper if arm in "BC" else direct
        arms[arm] = np.fromfile(root / f"arm{arm}.bin", dtype="<f4").reshape(
            nstep, len(names), ncol)
    table = np.fromfile(root / "fpvs_table.bin", dtype="<f8")
    sweep = (root / "fpvs_sweep.bin").read_bytes()
    (n,) = struct.unpack_from("<i", sweep, 0)
    sweep_t = np.frombuffer(sweep, dtype="<f4", count=n, offset=4)
    sweep_p = np.frombuffer(sweep, dtype="<f8", count=n, offset=4 + 4 * n)
    return SfgfsOracle(ncol=ncol, nstep=nstep, per_step=per_step,
                       initial=initial, direct_fields=direct,
                       wrapper_fields=wrapper, arms=arms, fpvs_table=table,
                       sweep_t=sweep_t, sweep_fpvs=sweep_p)


def table_digest(table: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(table, dtype="<f8").tobytes()).hexdigest()


def reference_arm(fx: SfgfsOracle, arm: str,
                  table: np.ndarray | None = None) -> np.ndarray:
    """Run the CPU reference over every step of one arm, oracle layout."""
    table = ref.gpvs() if table is None else table
    k = ref.wrf_constants()
    names = fx.fields(arm)
    out = np.zeros((fx.nstep, len(names), fx.ncol), dtype=np.float32)
    for i in range(fx.ncol):
        state = {"znt": np.float32(fx.initial[0, i]),
                 "ust": np.float32(fx.initial[1, i]),
                 "sst": np.float32(fx.initial[2, i]),
                 "hfx": np.float32(0.0), "hfx_sea": np.float32(0.0)}
        for name in ref.SEA_CARRIERS:
            state[name] = np.float32(0.0)
        for step in range(fx.nstep):
            col = fx.column(step, i)
            if arm in ("A", "D"):
                got, znt, ust = ref.sf_gfs_column(
                    col, state["znt"], state["ust"], k, table,
                    isfflx=0 if arm == "D" else 1, hfx_in=state["hfx"])
                state["znt"], state["ust"] = znt, ust
                state["hfx"] = got["hfx"]
                row = dict(got, znt=znt, ust=ust)
            else:
                got, tsk_sea = ref.sf_gfs_seaice_column(
                    col, state, k, table, itimestep=step + 1,
                    xice_threshold=WRAPPER_XICE_THRESHOLD,
                    xland_sea_fix=(arm == "C"))
                row = dict(got, znt=state["znt"], ust=state["ust"],
                           sst=state["sst"], tsk_sea=tsk_sea)
                row.update({name: state[name] for name in ref.SEA_CARRIERS})
            for f, name in enumerate(names):
                out[step, f, i] = row[name]
    return out


def ulp_distance(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """float32 ULP distance, signed-zero aware (0 means identical words)."""
    return np.abs(monotone_fp32_key(got) - monotone_fp32_key(want))


#: Columns whose ARM B and ARM C words may differ: sea-ice cells (XICE at or
#: above the wrapper threshold) that real.exe marked XLAND = 1, where WRF's
#: open-water call runs as land.  Computed from the inputs, not listed.
def declared_divergence_columns(fx: SfgfsOracle) -> np.ndarray:
    xice = fx.per_step[0, STEP_FIELDS.index("xice"), :]
    xland = fx.per_step[0, STEP_FIELDS.index("xland"), :]
    return np.nonzero((xice >= WRAPPER_XICE_THRESHOLD)
                      & (xice <= np.float32(1.0))
                      & (xland < np.float32(1.5)))[0]


__all__ = ["FIXTURE_DIR", "FPVS_TABLE_SHA256", "INITIAL_FIELDS",
           "STEP_FIELDS", "SfgfsOracle", "WRAPPER_XICE_THRESHOLD",
           "declared_divergence_columns", "load_sfgfs_oracle",
           "reference_arm", "table_digest", "ulp_distance"]
