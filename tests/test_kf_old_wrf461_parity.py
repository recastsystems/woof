"""cu_physics=99 (the previous Kain-Fritsch) against WRF v4.6.1, word for word.

The fixture is the output of ``tools/kf_old_wrf461_oracle/build.sh``: the
byte-unmodified ``phys/module_cu_kf.F`` driven through its own ``KFCPS``
entry point, one process per column, over 256 columns -- the curated
regimes (deep, cold-top, mid-latitude, high terrain, capped/shallow,
stable, dry, cold season), the two real HRRR columns the cu=1 tests carry,
and the wide-family columns ``select_cases.py`` kept because each reached
WRF source lines nothing else did.  Three oracle arms (the -O0 reference,
the same source with every uninitialised local poisoned, and WRF's own
production object) agreed word for word on every one of them before the
fixture was written.

Every output KFCPS produces is graded: the six tendencies, RAINCV, PRATEC,
NCA, the updated W0AVG, CU_ACT_FLAG and, where the case asks for
kf_edrates, UDR/DDR/UER/DER/TIMEC.  The pass condition is zero differing
words. It runs with ``GPUWM_WRF_EXACT=0`` and ``1`` in separate processes.
The pytest battery sets ``GPUWM_NUMERICS=strict`` in both processes; this
fixture alone does not grade the production fast compiler policy. The
lane's real forecast runs with the production fast default.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core.fp32_ulp import monotone_fp32_key

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (ROOT / "tests" / "data" / "kf_old_wrf461_oracle"
           / "kf-old-wrf461-oracle.npz")
FIXTURE_SHA256 = (
    "101e7708cecef84176885e32c62e758cec1382b4a97b3d2b6e90e37c6f823bbc")
KERNEL = ROOT / "woof" / "core" / "kernels" / "kf_old.cu"

FIELDS_3D = ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten",
             "rqscuten", "w0avg_out", "udr_kf", "ddr_kf", "uer_kf",
             "der_kf")
FIELDS_2D = ("raincv", "pratec", "nca_out", "timec_kf", "cu_act_flag",
             "fatal")


def _groups(data):
    return sorted({key.split("/")[0] for key in data.files})


def test_fixture_is_the_recorded_oracle_output():
    """The fixture bytes are the ones the oracle run recorded."""
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert digest == FIXTURE_SHA256


def test_fixture_reaches_the_branches_it_claims():
    """Enough columns trigger, hold, and take each species fold."""
    with np.load(FIXTURE) as data:
        total = triggered = held = 0
        folds = set()
        for g in _groups(data):
            nca_in = data[f"{g}/nca"]
            dt = data[f"{g}/dt"]
            pr = data[f"{g}/pratec"]
            total += nca_in.size
            hold = nca_in > np.float32(0.5) * dt
            held += int(hold.sum())
            fired = (~hold) & (data[f"{g}/nca_out"] != nca_in)
            triggered += int(fired.sum())
            for flags in zip(data[f"{g}/warm_rain"], data[f"{g}/f_qi"],
                             data[f"{g}/f_qs"], data[f"{g}/f_qr"],
                             fired):
                if flags[4]:
                    folds.add(tuple(int(x) for x in flags[:4]))
            assert not np.any(data[f"{g}/fatal"]), \
                "a fatal column is graded on its code alone; none expected"
    assert total >= 64
    assert triggered >= 64, triggered
    assert held >= 10, held
    # every KFCPS/KFPARA species fold reached by a convecting column
    assert folds >= {(0, 0, 0, 0), (1, 0, 0, 1), (0, 0, 0, 1),
                     (0, 1, 1, 1), (0, 1, 0, 1)}, folds


def test_workspace_slot_count_matches_the_kernel():
    from woof.core.kf import KF_OLD_WS_SLOTS

    src = KERNEL.read_text(encoding="utf-8")
    declared = int(re.search(r"#define KO_SLOTS (\d+)", src).group(1))
    used = {int(n) for n in re.findall(r"KO_SLOT\((\d+)\)", src)}
    assert declared == KF_OLD_WS_SLOTS
    assert used == set(range(KF_OLD_WS_SLOTS))


def _species(data, g, row):
    from woof.core.kf import KFOldSpecies

    return KFOldSpecies(bool(data[f"{g}/warm_rain"][row]),
                        bool(data[f"{g}/f_qi"][row]),
                        bool(data[f"{g}/f_qs"][row]),
                        bool(data[f"{g}/f_qr"][row]))


def _run_case(cp, data, g, row):
    from woof.core.kf import launch_kf_old

    def col(name):
        values = np.ascontiguousarray(data[f"{g}/{name}"][row],
                                      dtype=np.float32)
        return cp.asarray(values.reshape(values.size, 1, 1))

    w0avg = col("w0avg")
    edrates = bool(data[f"{g}/kf_edrates"][row])
    out = launch_kf_old(
        col("u"), col("v"), col("w"), col("t"), col("qv"), col("p"),
        col("rho"), col("dz8w"), col("pi"), w0avg,
        cp.full((1, 1), data[f"{g}/nca"][row], dtype=cp.float32),
        dx=float(data[f"{g}/dx"][row]), dt=float(data[f"{g}/dt"][row]),
        cudt_minutes=float(data[f"{g}/cudt"][row]),
        adaptive=bool(data[f"{g}/adapt"][row]),
        stepcu=int(data[f"{g}/stepcu"][row]),
        species=_species(data, g, row), kf_edrates=edrates)
    got = {name: cp.asnumpy(out[name]).ravel() for name in
           ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten",
            "rqscuten")}
    got["w0avg_out"] = cp.asnumpy(w0avg).ravel()
    nz = got["rthcuten"].size
    for name in ("udr_kf", "ddr_kf", "uer_kf", "der_kf"):
        got[name] = (cp.asnumpy(out[name]).ravel() if edrates
                     else np.zeros(nz, np.float32))
    got["timec_kf"] = (cp.asnumpy(out["timec_kf"]).ravel() if edrates
                       else np.zeros(1, np.float32))
    for name in ("raincv", "pratec", "nca_out", "cu_act_flag", "fatal"):
        got[name] = cp.asnumpy(out[name]).ravel()
    return got


def _ulp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in representable float32 steps (sign-magnitude ordered)."""
    return np.abs(monotone_fp32_key(a) - monotone_fp32_key(b))


def grade(cp) -> dict:
    """Per-field differing words and max ULP over the whole fixture."""
    table = {name: {"words": 0, "differing": 0, "max_ulp": 0}
             for name in FIELDS_3D + FIELDS_2D}
    cases = []
    with np.load(FIXTURE) as data:
        for g in _groups(data):
            for row in range(data[f"{g}/t"].shape[0]):
                got = _run_case(cp, data, g, row)
                bad = []
                for name in FIELDS_3D + FIELDS_2D:
                    want = np.ascontiguousarray(
                        data[f"{g}/{name}"][row]).ravel()
                    have = got[name].astype(want.dtype).ravel()
                    entry = table[name]
                    entry["words"] += want.size
                    if want.dtype == np.float32:
                        diff = want.view(np.uint32) != have.view(np.uint32)
                        if diff.any():
                            entry["max_ulp"] = max(
                                entry["max_ulp"],
                                int(_ulp(want, have).max()))
                    else:
                        diff = want != have
                    entry["differing"] += int(diff.sum())
                    if diff.any():
                        bad.append(name)
                if bad:
                    cases.append((g, row,
                                  int(data[f"{g}/case_index"][row]), bad))
    return {"fields": table, "failing_cases": cases}


@requires_gpu
def test_kf_old_matches_wrf461_word_for_word():
    import cupy as cp
    from woof import wrf_exact

    result = grade(cp)
    from woof.certify.kernel_manifest import kernel_manifest
    manifest = kernel_manifest()
    from woof.core.kernels import MODULE_KEY_ROOT
    compiled = manifest.get(f"{MODULE_KEY_ROOT}:kf_old")
    assert compiled is not None, sorted(manifest)
    assert any(option in ("--ftz=false", "-ftz=false")
               for option in compiled["options"]), compiled
    manifest_receipt = os.environ.get("WOOF_KF_OLD_MANIFEST_RECEIPT")
    if manifest_receipt:
        Path(manifest_receipt).write_text(json.dumps(manifest, indent=1) + "\n",
                                          encoding="utf-8")
    receipt = os.environ.get("WOOF_KF_OLD_PARITY_RECEIPT")
    if receipt:
        Path(receipt).write_text(json.dumps(
            {"strict": wrf_exact.ENABLED, **result}, indent=1) + "\n",
            encoding="utf-8")
    failing = {name: entry for name, entry in result["fields"].items()
               if entry["differing"]}
    assert not failing, (
        f"{len(result['failing_cases'])} cases differ; per field: "
        f"{failing}; first cases: {result['failing_cases'][:8]}")


def _planted_source() -> tuple[str, str]:
    """kf_old's assembled source with ONE bit of ONE constant flipped.

    3374.6525 is the liquid-water theta-e coefficient; the first use of it
    is in TPMIX's saturation branch, which every triggering column runs.
    The word is moved by its lowest mantissa bit and nothing else.
    """
    from woof.core.kernels import module_source

    src = module_source("kf_old")
    word = np.float32(3374.6525)
    flipped = (word.view(np.uint32) ^ np.uint32(1)).view(np.float32)
    literal = "3374.6525f"
    assert literal in src
    planted = float(flipped).hex() + "f"
    return src.replace(literal, planted, 1), planted


@requires_gpu
def test_a_planted_one_bit_change_is_caught(monkeypatch):
    """The grade detects a one-ULP constant edit in the kernel."""
    import cupy as cp
    import woof.core.kernels as kernels

    src, planted = _planted_source()
    module = kernels.compile_noftz_module(
        "kf_old", src, "kf_old_one_bit_mutation")
    planted_fn = module.get_function("kf_old_column")
    real = kernels.get_kernel

    def get_kernel(name, func):
        if (name, func) == ("kf_old", "kf_old_column"):
            return planted_fn
        return real(name, func)

    monkeypatch.setattr(kernels, "get_kernel", get_kernel)
    result = grade(cp)
    differing = sum(entry["differing"]
                    for entry in result["fields"].values())
    assert differing > 0, f"planted {planted} went undetected"
    assert len(result["failing_cases"]) >= 10, result["failing_cases"]


def test_old_kf_default_options_preserve_subnormals():
    from woof.core.kernels import module_options
    assert '--ftz=false' in module_options('kf_old')
    assert module_options('kf') == ('-std=c++17',)


@requires_gpu
def test_old_kf_shared_route_preserves_subnormal_words():
    import cupy as cp
    from woof.core.kernels import get_kernel
    x = np.array([1.e-30, 1.e-10, 1.e-39, 2.], dtype=np.float32)
    want = np.array([x[0]*x[1], x[2]/x[3]],dtype=np.float32)
    assert np.all(want != 0) and np.all(want < np.finfo(np.float32).tiny)
    out = cp.empty(2,dtype=cp.float32)
    get_kernel('kf_old','kf_old_float_probe')((1,),(32,),(cp.asarray(x),out))
    np.testing.assert_array_equal(cp.asnumpy(out).view(np.uint32),want.view(np.uint32))
