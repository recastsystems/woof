"""QNSE PBL (bl_pbl_physics=4) and its EDMF mass flux (mfshconv=1) against
WRF v4.6.1, word for word.

``tools/qnse_wrf461_oracle`` builds the byte-unmodified
``phys/module_bl_qnsepbl.F`` and ``phys/module_bl_mfshconvpbl.F`` with
gfortran 13 at WRF's own flags (-O2 -fno-fast-math -ffp-contract=off
-fno-tree-vectorize) and drives them as ``module_pbl_driver.F`` does
(:1515-1557) over 72 columns (convective, stable, stratocumulus land and
sea, shallow cumulus land and sea, sea, and an edge group: snow, sea ice,
sub-freezing, 1500/3000 m terrain, a jet, weak u*, southern and equatorial
Coriolis, the TKE floor, a carried CT) for four steps, in two arms:
``mf0`` (mfshconv = 0, dt = 30 s, stepbl = 1) and ``mf1`` (mfshconv = 1,
dt = 10 s, stepbl = 3).  The packed records are
``tests/data/oracles/qnse_pbl/qnse-<arm>.npz``.

Two grades:

* every step from the oracle's own recorded inputs (atmosphere AND carried
  state), so a defect is located to one step;
* the four steps CHAINED, the carried state (TKE_PBL, RC_MF, CT, THZ0,
  QZ0, QSFC, EXCH_H/M) taken from the kernel's own previous step, so the
  carry is graded too.

Under ``GPUWM_WRF_EXACT=1`` every word must be identical. Production uses
CUDA elementary functions and permits contraction. Its finite-output and
relative-error checks are separate from this strict word-grade proof.

The historical fixture calls native WRF with one-column tiles, selected
explicitly with native_tile=False. Production retains native WRF's tile
COUNT(GTEST) with a device reduction and a second updraft pass.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "oracles" / "qnse_pbl"
KERNEL = ROOT / "woof" / "core" / "kernels" / "qnsepbl.cu"
CONSTS = FIXTURE / "consts.txt"

STRICT = os.environ.get("GPUWM_WRF_EXACT", "0") == "1"

QNSE_LEVEL_OUT = {"rublten": "rublten", "rvblten": "rvblten",
                  "rthblten": "rthblten", "rqvblten": "rqvblten",
                  "rqcblten": "rqcblten", "el_myj": "el_pbl",
                  "exch_h": "exch_h", "exch_m": "exch_m", "tke": "tke"}
STATE_OUT = ("ct", "thz0", "qz0", "qsfc")
MF_OUT = ("massflux_edkf", "entr_edkf", "detr_edkf", "thl_up", "thv_up",
          "rt_up", "rv_up", "rc_up", "u_up", "v_up", "frac_up", "rc_mf",
          "wthv_mf", "lm_bl89")
ARMS = ("mf0", "mf1")



def load(arm: str) -> dict[str, np.ndarray]:
    with np.load(FIXTURE / f"qnse-{arm}.npz") as z:
        return {k: z[k] for k in z.files}


def ulp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ia = np.ascontiguousarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    ib = np.ascontiguousarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ib = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return np.abs(ia - ib)


def _state_from_oracle(fx, step):
    p = f"s{step}/"
    return {"tke": fx[p + "in_tke"], "rc_mf": fx[p + "in_rc_mf"],
            "exch_h": fx[p + "in_exch_h"], "exch_m": fx[p + "in_exch_m"],
            "ct": fx[p + "in_ct"], "thz0": fx[p + "in_thz0"],
            "qz0": fx[p + "in_qz0"], "qsfc": fx[p + "in_qsfc"]}


def _run(fx, step, carried):
    """One step on the device; returns every output as host arrays."""
    import cupy as cp

    from woof.core.myjpbl import (MFSHCONV_DIAGNOSTICS_3D,
                                   MFSHCONV_OUTPUTS_3D, qnse_pbl_step)

    nz, ncol = int(fx["meta_nz"]), int(fx["meta_ncol"])
    mfsh = bool(int(fx["meta_mfshconv"]))
    dtturbl = np.float32(fx["meta_dt"]) * np.float32(int(fx["meta_stepbl"]))
    p = f"s{step}/"

    def lev(a):
        return cp.asarray(np.ascontiguousarray(
            np.asarray(a, dtype=np.float32)[:nz].reshape(nz, 1, ncol)))

    def sfc(a):
        return cp.asarray(np.ascontiguousarray(
            np.asarray(a, dtype=np.float32).reshape(1, ncol)))

    columns = {"dz": lev(fx["dz"]), "u": lev(fx[p + "in_u"]),
               "v": lev(fx[p + "in_v"]), "t": lev(fx[p + "in_t"]),
               "th": lev(fx[p + "in_th"]), "exner": lev(fx["exner"]),
               "qv": lev(fx[p + "in_qv"]), "qc": lev(fx[p + "in_qc"]),
               "p": lev(fx["pmid"]), "rho": lev(fx["rho"])}
    surface = {"ht": sfc(fx["ht"]), "psfc": sfc(fx["pint"][0]),
               "ust": sfc(fx["ustar"]), "chklowq": sfc(fx["chklowq"]),
               "xland": sfc(fx["xland"]), "sice": sfc(fx["sice"]),
               "snow": sfc(fx["snow"]), "akhs": sfc(fx["akhs"]),
               "akms": sfc(fx["akms"]), "elflx": sfc(fx["lh"]),
               "hfx": sfc(fx["hfx"]), "qfx": sfc(fx["qfx"]),
               "uz0": sfc(fx["uz0"]), "vz0": sfc(fx["vz0"]),
               "corf": sfc(fx["corf"])}
    state = {n: sfc(carried[n]) for n in STATE_OUT}
    tke = lev(carried["tke"])
    mf_state = {"rc_mf": lev(carried["rc_mf"])}
    mf_outputs = {n: cp.zeros((nz, 1, ncol), dtype=cp.float32)
                  for n in MFSHCONV_OUTPUTS_3D}
    diagnostics = {n: cp.zeros((nz, 1, ncol), dtype=cp.float32)
                   for n in MFSHCONV_DIAGNOSTICS_3D}
    out = qnse_pbl_step(columns, surface, state, tke, dtturbl=float(dtturbl),
                        exch_h=lev(carried["exch_h"]),
                        exch_m=lev(carried["exch_m"]), mfshconv=mfsh,
                        mf_state=mf_state, mf_outputs=mf_outputs,
                        diagnostics=diagnostics, native_tile=False)
    got = {n: cp.asnumpy(out[n]).reshape(nz, ncol) for n in QNSE_LEVEL_OUT
           if n != "tke"}
    got["tke"] = cp.asnumpy(tke).reshape(nz, ncol)
    got["pblh"] = cp.asnumpy(out["pblh"]).reshape(ncol)
    got["kpbl"] = cp.asnumpy(out["kpbl"]).reshape(ncol)
    for n in STATE_OUT:
        got[n] = cp.asnumpy(state[n]).reshape(ncol)
    if mfsh:
        for n in MFSHCONV_OUTPUTS_3D:
            got[n] = cp.asnumpy(mf_outputs[n]).reshape(nz, ncol)
        got["rc_mf"] = cp.asnumpy(mf_state["rc_mf"]).reshape(nz, ncol)
        for n in MFSHCONV_DIAGNOSTICS_3D:
            got[n] = cp.asnumpy(diagnostics[n]).reshape(nz, ncol)
    return got


def _grade(fx, step, got):
    p = f"s{step}/"
    measured = {}
    for n, w in QNSE_LEVEL_OUT.items():
        measured[n] = int(ulp(got[n], fx[p + w]).max())
    measured["pblh"] = int(ulp(got["pblh"], fx[p + "pblh"]).max())
    measured["kpbl"] = int(np.abs(got["kpbl"] - fx[p + "kpbl"]).max())
    for n in STATE_OUT:
        measured[n] = int(ulp(got[n], fx[p + n]).max())
    if int(fx["meta_mfshconv"]):
        for n in MF_OUT:
            measured[n] = int(ulp(got[n], fx[p + n]).max())
    return measured


def _over(measured):
    assert STRICT, "word-grade comparison requires GPUWM_WRF_EXACT=1"
    return {key: value for key, value in measured.items() if value}


def _carry(got):
    return {"tke": got["tke"], "rc_mf": got.get("rc_mf"),
            "exch_h": got["exch_h"], "exch_m": got["exch_m"],
            **{n: got[n] for n in STATE_OUT}}


def test_fixture_reaches_the_arms():
    """The fixture is only evidence for the branches it reaches."""
    fx = load("mf1")
    nz = int(fx["meta_nz"])
    assert int(fx["meta_ncol"]) >= 64
    assert (fx["xland"] == 2).any() and (fx["xland"] == 1).any()
    assert (fx["snow"] > 0).any() and (fx["sice"] > 0.5).any()
    assert (fx["ht"] >= 1500).any() and (fx["corf"] < 0).any()
    assert (fx["corf"] == 0).any() and (fx["s1/in_ct"] != 0).any()
    assert (fx["hfx"] > 0).any() and (fx["hfx"] < 0).any()
    assert (fx["s1/in_qc"] > 0).any()                     # cloud-topped
    # Updrafts that stay dry, and updrafts that condense (LCL found).
    mf = fx["s1/massflux_edkf"]
    rc = fx["s1/rc_up"]
    active = mf.max(axis=0) > 0
    cloudy = rc.max(axis=0) > 0
    assert (active & ~cloudy).sum() >= 8 and (active & cloudy).sum() >= 8
    # A sub-freezing column whose updraft runs (the cold TH_R arm).
    t_in = fx["s1/in_t"][:nz]
    assert (active & (t_in[0] < 273.15)).any()
    # Stable layers (the QNSE stable length) and unstable ones.
    assert (fx["s1/rthblten"] < 0).any() and (fx["s1/rthblten"] > 0).any()
    # TKE carries: step 4 differs from step 1 in every column.
    assert (fx["s4/in_tke"] != fx["s1/in_tke"]).any(axis=0).all()
    # The carried RC_MF is nonzero by step 2.
    assert (fx["s2/in_rc_mf"] > 0).any()


def test_kernel_constants_are_the_folded_words():
    """Every pinned constant in kernels/qnsepbl.cu equals the word gfortran
    folded (the oracle's const_dump listing, consts.txt)."""
    words = {}
    for line in CONSTS.read_text(encoding="utf-8").splitlines():
        name, hexword, _ = line.split()
        words[name] = int(hexword, 16)
    pinned = re.findall(r"^#define\s+(\w+)\s+(-?0x[0-9a-fA-Fp.+-]+)f\s+//\s+(\S+)",
                        KERNEL.read_text(encoding="utf-8"), re.M)
    assert len(pinned) >= 80
    checked = 0
    for macro, literal, source in pinned:
        key = source if source in words else source.split("=")[0]
        if key not in words:
            continue
        value = np.float32(float.fromhex(literal))
        assert int(value.view(np.uint32)) == words[key], (macro, source)
        checked += 1
    assert checked >= 80, checked


@requires_gpu
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("step", [1, 2, 3, 4])
@pytest.mark.skipif(not STRICT, reason="word-grade proof selects GPUWM_WRF_EXACT=1")
def test_qnse_step_matches_wrf(arm, step):
    fx = load(arm)
    got = _run(fx, step, _state_from_oracle(fx, step))
    measured = _grade(fx, step, got)
    print("measured", arm, step, "strict" if STRICT else "default", measured)
    assert not _over(measured), _over(measured)


@requires_gpu
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.skipif(not STRICT, reason="word-grade proof selects GPUWM_WRF_EXACT=1")
def test_qnse_chained_steps_carry_like_wrf(arm):
    fx = load(arm)
    carried = _state_from_oracle(fx, 1)
    if carried["rc_mf"] is None:
        carried["rc_mf"] = np.zeros_like(carried["tke"])
    for step in (1, 2, 3, 4):
        got = _run(fx, step, carried)
        measured = _grade(fx, step, got)
        assert not _over(measured), (step, _over(measured))
        carried = _carry(got)
        if carried["rc_mf"] is None:
            carried["rc_mf"] = np.zeros_like(carried["tke"])


@requires_gpu
@pytest.mark.skipif(not STRICT, reason="word-grade proof selects GPUWM_WRF_EXACT=1")
def test_a_one_bit_qnse_constant_change_is_caught(monkeypatch):
    import cupy as cp
    from woof.core import kernels, myjpbl
    from woof.certify.kernel_manifest import record_module

    source = kernels.module_source("qnsepbl")
    match = re.search(r"(#define QN_G\s+)(0x[0-9a-fA-Fp.+-]+)f", source)
    assert match is not None
    original = np.float32(float.fromhex(match.group(2)))
    changed = np.array(int(original.view(np.uint32)) ^ 1, dtype=np.uint32).view(np.float32)
    planted = source[:match.start(2)] + float(changed).hex() + source[match.end(2):]
    assert planted != source
    options = kernels.module_options("qnsepbl")
    module = kernels._load_module_preserving_subnormals("qnsepbl-one-bit-control", planted, options)
    real_get = myjpbl.get_kernel
    monkeypatch.setattr(myjpbl, "get_kernel", lambda name, entry:
        module.get_function(entry) if name == "qnsepbl" else real_get(name, entry))
    fx = load("mf0")
    measured = _grade(fx, 1, _run(fx, 1, _state_from_oracle(fx, 1)))
    assert any(measured.values()), "a one-bit gravity change passed every field"


def _word_mismatches(fx, step, got):
    mapping=dict(QNSE_LEVEL_OUT)
    mapping.update({name:name for name in (*STATE_OUT,"pblh","kpbl")})
    if int(fx["meta_mfshconv"]):mapping.update({name:name for name in MF_OUT})
    bad={}
    for name,reference in mapping.items():
        a=np.ascontiguousarray(got[name]);b=np.ascontiguousarray(fx[f"s{step}/{reference}"])
        assert a.shape==b.shape and a.dtype==b.dtype,(name,a.shape,b.shape,a.dtype,b.dtype)
        count=int((a.view(np.uint32)!=b.view(np.uint32)).sum())
        if count:bad[name]=count
    return bad


@requires_gpu
@pytest.mark.parametrize("arm",ARMS)
@pytest.mark.skipif(not STRICT, reason="word-grade proof selects GPUWM_WRF_EXACT=1")
def test_every_qnse_output_word_including_zero_sign_is_identical(arm):
    fx=load(arm)
    carried=_state_from_oracle(fx,1)
    for step in (1,2,3,4):
        got=_run(fx,step,carried)
        bad=_word_mismatches(fx,step,got)
        print("word_mismatches",arm,step,bad)
        assert not bad,(step,bad)
        carried=_carry(got)
        if carried["rc_mf"] is None:carried["rc_mf"]=np.zeros_like(carried["tke"])
