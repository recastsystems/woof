"""Reader for the KF-CuP (cu_physics = 10) WRF v4.6.1 column capture.

The capture is written by tools/kfcup_wrf461_oracle (build.sh) from WRF's own
compiled ``phys/module_cu_kfcup.o`` and committed, gzipped, under
``tests/data/oracles/kfcup``.  Every float in it is its raw IEEE-754 word in
hex; this module turns the files into float32 arrays and packs them into the
argument layout of ``kfcup_column`` (woof/core/kernels/kfcup.cu), so the CPU
reference, the CUDA kernel and the WRF answer are compared on the same words.
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ORACLE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "kfcup"

#: kfcup_column's packed field orders (kfcup.cu KCI_/KCS_/KCH_/KCF_/KCO_).
from woof.core.kf import (
    KFCUP_LEVEL_INPUTS as LEVEL_INPUTS,
    KFCUP_SURFACE_INPUTS as SURFACE_INPUTS,
    KFCUP_HELD_SCALARS as HELD_SCALARS,
    KFCUP_HELD_INTS as HELD_INTS,
    KFCUP_LEVEL_OUTPUTS as LEVEL_OUTPUTS,
)

#: The words run_kfcup_cps.F90 seeds every INOUT/OUT array with before the
#: call, so a column the scheme does not write (the NCA hold, and every
#: branch WRF leaves a field alone on) carries its input word through.
SEED_LEVEL = {"rthcuten": 1.25e-5, "rqvcuten": -2.5e-9, "rqccuten": 3.5e-9,
              "rqrcuten": 4.5e-9, "rqicuten": 5.5e-9, "rqscuten": 6.5e-9,
              "cldfra_cup": 0.125, "wup_cup": 0.375, "qc_ic_cup": 1.5e-4,
              "qndrop_ic_cup": 2.5e6, "qc_iu_cup": 2.5e-4, "fcvt_qc_to_pr_cup": 0.25,
              "fcvt_qc_to_qi_cup": 0.5, "fcvt_qi_to_pr_cup": 0.75, "mfup_cup": 0.0175,
              "mfup_ent_cup": 0.0275, "mfdn_cup": -0.0125, "mfdn_ent_cup": 0.0225,
              "updfra_cup": 0.03125}
SEED_HELD = {"raincv": 7.75, "shall": 2.0, "cubot": -3.0, "cutop": -4.0,
             "slopesfc": -5.0, "slopeez": -6.0, "sigmasfc": -7.0, "sigmaez": -8.0,
             "taucloud": -9.0, "tactive": -10.0, "tstar": -11.0, "lnint": -12.0,
             "activefrac": -13.0, "wcloudbase": -14.0, "wact_cup": -15.0,
             "wulcl_cup": -16.0, "tcloud_cup": -17.0}
SEED_INTS = {"cupflag": 1, "cu_act_flag": 0}

FLAG_WARM_RAIN, FLAG_F_QR, FLAG_F_QI, FLAG_F_QS = 1, 2, 4, 8


def word(text: str) -> np.float32:
    return np.uint32(int(text, 16)).view(np.float32)


def _rows(name: str, root: Path | None = None):
    root = Path(root) if root is not None else ORACLE_DIR
    path = root / name
    if not path.exists():
        path = root / (name + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle)


@dataclass
class KfcupConfig:
    index: int
    name: str
    numbins: int
    thbinsize: np.float32
    rbinsize: np.float32
    mindeepfreq: np.float32
    minshallowfreq: np.float32
    dt: np.float32
    dx: np.float32
    stepcu: int
    ktau: int
    sf_sfclay_physics: int
    warm_rain: bool
    f_qr: bool
    f_qi: bool
    f_qs: bool

    @property
    def flags(self) -> int:
        return ((FLAG_WARM_RAIN if self.warm_rain else 0) | (FLAG_F_QR if self.f_qr else 0)
                | (FLAG_F_QI if self.f_qi else 0) | (FLAG_F_QS if self.f_qs else 0))


def configs(root=None) -> list[KfcupConfig]:
    out = []
    for r in _rows("kfcup-config.csv", root):
        out.append(KfcupConfig(
            index=int(r["cfg"]), name=r["name"], numbins=int(r["numbins"]),
            thbinsize=word(r["thbinsize"]), rbinsize=word(r["rbinsize"]),
            mindeepfreq=word(r["mindeepfreq"]), minshallowfreq=word(r["minshallowfreq"]),
            dt=word(r["dt"]), dx=word(r["dx"]), stepcu=int(r["stepcu"]), ktau=int(r["ktau"]),
            sf_sfclay_physics=int(r["sf_sfclay_physics"]), warm_rain=r["warm_rain"] == "T",
            f_qr=r["f_qr"] == "T", f_qi=r["f_qi"] == "T", f_qs=r["f_qs"] == "T"))
    return out


@dataclass
class KfcupCapture:
    """One configuration of the end-to-end capture, in (nz, ncol) layout."""

    config: KfcupConfig
    names: list[str]
    nz: int
    level_in: dict          # name -> (nz(+1), ncol) float32, the INITIAL words
    surface_in: dict        # name -> (ncol,)
    kpbl: np.ndarray        # (ncol,) int32
    level_out: dict         # name -> (nz, ncol), WRF's answer
    surface_out: dict       # name -> (ncol,)

    @property
    def ncol(self) -> int:
        return len(self.names)


def load_capture(cfg_index: int, root=None) -> KfcupCapture:
    cfg = next(c for c in configs(root) if c.index == cfg_index)
    lev_rows = [r for r in _rows("kfcup-in-levels.csv", root) if r["cfg"] == "1"]
    ncol = max(int(r["case"]) for r in lev_rows)
    nzp1 = max(int(r["k"]) for r in lev_rows)
    nz = nzp1 - 1
    fields = ("u", "v", "th", "t", "w", "dz8w", "p", "pi", "rho", "qv", "z", "z_at_w",
              "w0avg", "cldfra_cup", "rthcuten", "rqvcuten")
    level_in = {f: np.zeros((nzp1, ncol), np.float32) for f in fields}
    for r in lev_rows:
        c, k = int(r["case"]) - 1, int(r["k"]) - 1
        for f in fields:
            level_in[f][k, c] = word(r[f])
    surface_in = {}
    kpbl = np.zeros(ncol, np.int32)
    names = [""] * ncol
    sfields = ("psfc", "ht", "tsk", "hfx", "qfx", "xland", "mavail", "br", "regime", "pblh",
               "t2", "q2", "nca", "raincv", "shall")
    for f in sfields:
        surface_in[f] = np.zeros(ncol, np.float32)
    for r in _rows("kfcup-in-surface.csv", root):
        if int(r["cfg"]) != cfg_index:
            continue
        c = int(r["case"]) - 1
        names[c] = r["name"]
        kpbl[c] = int(r["kpbl"])
        for f in sfields:
            surface_in[f][c] = word(r[f])
    level_out = {}
    lfields = ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten", "rqscuten",
               "w0avg", "cldfra_cup", "cldfratend_cup", "wup_cup", "qc_ic_cup",
               "qndrop_ic_cup", "qc_iu_cup", "fcvt_qc_to_pr_cup", "fcvt_qc_to_qi_cup",
               "fcvt_qi_to_pr_cup", "mfup_cup", "mfup_ent_cup", "mfdn_cup", "mfdn_ent_cup",
               "updfra_cup")
    for f in lfields:
        level_out[f] = np.zeros((nzp1, ncol), np.float32)
    for r in _rows("kfcup-levels.csv", root):
        if int(r["cfg"]) != cfg_index:
            continue
        c, k = int(r["case"]) - 1, int(r["k"]) - 1
        for f in lfields:
            level_out[f][k, c] = word(r[f])
    level_out = {f: v[:nz].copy() for f, v in level_out.items()}
    surface_out = {}
    sofields = ("raincv", "nca", "cutop", "cubot", "shall", "slopesfc", "slopeez",
                "sigmasfc", "sigmaez", "taucloud", "tactive", "tstar", "lnint", "activefrac",
                "wcloudbase", "wact_cup", "wulcl_cup", "tcloud_cup")
    for f in sofields:
        surface_out[f] = np.zeros(ncol, np.float32)
    surface_out["cupflag"] = np.zeros(ncol, np.int32)
    surface_out["cu_act_flag"] = np.zeros(ncol, np.int32)
    for r in _rows("kfcup-surface.csv", root):
        if int(r["cfg"]) != cfg_index:
            continue
        c = int(r["case"]) - 1
        for f in sofields:
            surface_out[f][c] = word(r[f])
        surface_out["cupflag"][c] = int(r["cupflag"])
        surface_out["cu_act_flag"][c] = int(r["cu_act_flag"])
    return KfcupCapture(cfg, names, nz, level_in, surface_in, kpbl, level_out, surface_out)


def pack_column_inputs(cap: KfcupCapture) -> dict:
    """``kfcup_column``'s packed buffers for one capture, seeded as WRF was.

    ``w0avg`` is the scheme's UPDATED trigger velocity -- the capture's output
    word -- because the update itself runs in the driver before the column
    kernel (KainFritsch.update_trigger_history), and is graded separately.
    """
    nz, ncol = cap.nz, cap.ncol
    lev = np.zeros((len(LEVEL_INPUTS), nz, ncol), np.float32)
    for i, name in enumerate(LEVEL_INPUTS):
        if name == "w0avg":
            lev[i] = cap.level_out["w0avg"]
        else:
            lev[i] = cap.level_in[name][:nz]
    sca = np.zeros((len(SURFACE_INPUTS), ncol), np.float32)
    for i, name in enumerate(SURFACE_INPUTS):
        sca[i] = cap.surface_in[name]
    held = np.zeros((len(HELD_SCALARS), ncol), np.float32)
    for i, name in enumerate(HELD_SCALARS):
        held[i] = cap.surface_in["nca"] if name == "nca" else np.float32(SEED_HELD[name])
    iheld = np.zeros((len(HELD_INTS), ncol), np.int32)
    for i, name in enumerate(HELD_INTS):
        iheld[i] = SEED_INTS[name]
    out = np.zeros((len(LEVEL_OUTPUTS), nz, ncol), np.float32)
    for i, name in enumerate(LEVEL_OUTPUTS):
        out[i] = np.float32(SEED_LEVEL[name])
    return {"lev_in": lev, "sca_in": sca, "kpbl": cap.kpbl.copy(), "held": held,
            "iheld": iheld, "lev_out": out}


def expected_outputs(cap: KfcupCapture) -> dict:
    """WRF's words, keyed by kfcup_column's output names."""
    exp = {name: cap.level_out[name] for name in LEVEL_OUTPUTS}
    for name in HELD_SCALARS:
        exp[name] = cap.surface_out[name]
    exp["cupflag"] = cap.surface_out["cupflag"]
    exp["cu_act_flag"] = cap.surface_out["cu_act_flag"]
    return exp


def unpack_outputs(buffers: dict) -> dict:
    got = {name: buffers["lev_out"][i] for i, name in enumerate(LEVEL_OUTPUTS)}
    for i, name in enumerate(HELD_SCALARS):
        got[name] = buffers["held"][i]
    for i, name in enumerate(HELD_INTS):
        got[name] = buffers["iheld"][i]
    return got


def differing_words(got: dict, exp: dict) -> dict:
    """Per field, the number of words that differ bitwise (NaN payloads too)."""
    out = {}
    for name, e in exp.items():
        g = np.asarray(got[name])
        if g.dtype == np.float32:
            out[name] = int(np.count_nonzero(g.view(np.uint32) != e.view(np.uint32)))
        else:
            out[name] = int(np.count_nonzero(g != e))
    return out


__all__ = ["HELD_INTS", "HELD_SCALARS", "KfcupCapture", "KfcupConfig", "LEVEL_INPUTS",
           "LEVEL_OUTPUTS", "ORACLE_DIR", "SURFACE_INPUTS", "configs", "differing_words",
           "expected_outputs", "load_capture", "pack_column_inputs", "unpack_outputs", "word"]
