"""The K-epsilon PBL oracle fixtures (tools/keps_wrf461_oracle).

Each fixture SET is one ``keps-<set>.bin.xz`` (raw little-endian float32
words, xz-compressed) beside its ``keps-<set>.manifest`` (field name, level
count, column count, word offset), written by ``run_bl_keps.F90`` from the
gfortran build of WRF v4.6.1 ``module_bl_keps.F`` with the declared
divergences applied.  A set holds the static inputs, the initial evolving
state and every call's outputs; :func:`replay_inputs` rebuilds each call's
inputs with the oracle program's own float32 update ``x = x + xBLTEN*dt``,
so every input word is one the oracle fed WRF.
"""

from __future__ import annotations

import lzma
from dataclasses import dataclass
from pathlib import Path
from woof import data_assets

import numpy as np

FIXTURE_DIR = data_assets.verify_oracle_path("keps/oracle")
SETS = ("Acold", "Adev", "Bcold", "Bdev")

#: Per-call outputs the fixture records, WRF names.
STEP_OUTPUTS_3D = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
                   "tke_pbl", "diss_pbl", "tpe_pbl", "pr_pbl", "tke_adv",
                   "diss_adv", "tpe_adv", "wu_tur", "wv_tur", "wt_tur",
                   "wq_tur", "exch_h", "exch_m")
#: The oracle pre-fills these INTENT(OUT) arrays with this word before each
#: call, so a fixture word equal to it is one keps never wrote.
SENTINEL = np.float32(-999.0)


@dataclass(frozen=True)
class KepsSet:
    name: str
    ncol: int
    nz: int
    nstep: int
    itime0: int
    dt: np.float32
    fields: dict

    def step(self, name: str, istep: int) -> np.ndarray:
        return self.fields[f"{name}@{istep}"]


def _manifest(path: Path):
    fields, meta = {}, {}
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[1].isdigit():
            fields[parts[0]] = (int(parts[1]), int(parts[2]), int(parts[3]))
        elif parts:
            meta[parts[0]] = parts[1:]
    return fields, meta


def load_set(name: str, directory: Path = FIXTURE_DIR) -> KepsSet:
    data_assets.require_verify_oracle(directory)
    fields, meta = _manifest(directory / f"keps-{name}.manifest")
    raw = lzma.decompress((directory / f"keps-{name}.bin.xz").read_bytes())
    words = np.frombuffer(raw, dtype="<f4")
    if int(meta["end_words"][0]) != words.size:
        raise ValueError(f"keps-{name}: {words.size} words, manifest says "
                         f"{meta['end_words'][0]}")
    arrays = {}
    for field, (nlev, ncol, offset) in fields.items():
        block = words[offset:offset + max(nlev, 1) * ncol]
        arrays[field] = (block.reshape(nlev, ncol) if nlev else block).astype(
            np.float32, copy=True)
    ncol = int(meta["ncol"][0])
    nz = int(meta["ncol"][2])
    nstep = int(meta["ncol"][4])
    itime0 = int(meta["ncol"][6])
    dt = np.uint32(int(meta["dt"][0], 16)).view(np.float32)
    return KepsSet(name, ncol, nz, nstep, itime0, np.float32(dt), arrays)


def replay_inputs(fixture: KepsSet):
    """Yield ``(istep, itimestep, evolving, state_in)`` for every call.

    ``evolving`` holds u, v, th, qv, qc as the oracle passed them;
    ``state_in`` the scheme's carried state on entry (the previous call's
    outputs, or the set's initial state).
    """
    f = fixture.fields
    evolving = {k: f[f"{k}0"].copy() for k in ("u", "v", "th", "qv", "qc")}
    state = {"tke_pbl": f["tke0"].copy(), "diss_pbl": f["diss0"].copy(),
             "tpe_pbl": f["tpe0"].copy(), "pr_pbl": f["pr0"].copy(),
             "tke_adv": f["tke_adv0"].copy(), "diss_adv": f["diss_adv0"].copy(),
             "tpe_adv": f["tpe_adv0"].copy(), "pblh": f["pblh0"].copy()}
    tendency = {"u": "rublten", "v": "rvblten", "th": "rthblten",
                "qv": "rqvblten", "qc": "rqcblten"}
    dt = fixture.dt
    for istep in range(1, fixture.nstep + 1):
        yield istep, fixture.itime0 + istep - 1, evolving, state
        for name, ten in tendency.items():
            # run_bl_keps.F90: x = x + xBLTEN*dt, two float32 roundings.
            evolving[name] = (evolving[name]
                              + fixture.step(ten, istep) * dt).astype(np.float32)
        state = {name: fixture.step(name, istep).copy()
                 for name in ("tke_pbl", "diss_pbl", "tpe_pbl", "pr_pbl",
                              "tke_adv", "diss_adv", "tpe_adv")}
        state["pblh"] = fixture.step("pblh", istep).copy()


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in float32 units in the last place (bit patterns, so a NaN
    word only matches the same NaN word)."""
    ia = np.ascontiguousarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    ib = np.ascontiguousarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ib = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return np.abs(ia - ib)


__all__ = ["FIXTURE_DIR", "KepsSet", "SENTINEL", "SETS", "STEP_OUTPUTS_3D",
           "load_set", "replay_inputs", "ulp_distance"]
