"""KIAPS SAS (cu_physics = 14) on the device against WRF v4.6.1 itself.

The oracle is the scalar WRF build's own ``module_cu_ksas.o`` driven by
``tools/ksas_wrf461_oracle/run_ksas.F90`` over 12 call groups x 64 columns
(``tests/fixtures/ksas_wrf461``): deep tropical, severe mid-latitude, capped
shallow, dry stable, strong shear, subsiding, moist neutral and random
soundings over land and water, at dx from 250 m to 30 km, dt 1.5 s to
150 s, stepcu 1, 3 and 5 and mp_physics 0, 1, 3, 6, 8 and 10 (ncloud 0, 1
and 2).  ``sas_ksas_column`` is graded on every word of all ten outputs:
RTHCUTEN, RQVCUTEN, RQCCUTEN, RQICUTEN (where the call carries QI), RUCUTEN,
RVCUTEN, RAINCV, PRATEC, HBOT and HTOP.

This constructed fixture grades strict arithmetic (``GPUWM_NUMERICS=strict``).
The default fast route uses CUDA math and contraction. This fixture is not
a replay captured from a native forecast, nor a forecast-skill receipt.

A planted one-bit change to one kernel constant must be caught: the same
source with KSAS_CXLAMU's last mantissa bit flipped is compiled beside the
real one and must disagree with WRF somewhere in the fixture.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = _ROOT / "tools" / "ksas_wrf461_oracle"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import gen_cases  # noqa: E402

from woof.core.fp32_ulp import fp32_ulp_distance  # noqa: E402
from woof.core.sas import SAS_VARIANTS, launch_column  # noqa: E402

FIXTURE = _ROOT / "tests" / "fixtures" / "ksas_wrf461"
#: tools/ksas_wrf461_oracle/build.sh writes these two digests; the fixture
#: is the bytes it hashed.
CASES_SHA256 = "f648ee440a90d97145233eab692fda93d06d5ce938e2767fc4705612ca2901c8"
ORACLE_SHA256 = "6ffb1c6d1246b6fc2e61e97ee3bf1da33914bf6c96f9636c65ef8556d280c9c2"

LEVEL_INPUTS = ("t3d", "qv3d", "qc3d", "qi3d", "p3d", "pi3d", "rho3d",
                "dz8w", "u3d", "v3d")


def _unpacked(name: str, digest: str, tmp_path_factory) -> Path:
    raw = gzip.decompress((FIXTURE / f"{name}.gz").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == digest, name
    path = tmp_path_factory.mktemp("ksas") / name
    path.write_bytes(raw)
    return path


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    cases = _unpacked("cases.bin", CASES_SHA256, tmp_path_factory)
    answers = _unpacked("ksas-oracle.bin", ORACLE_SHA256, tmp_path_factory)
    groups = gen_cases.read(cases)
    return groups, gen_cases.read_outputs(answers, groups)


def _run(group, function=None):
    a = group["arrays"]
    inputs = {name: cp.asarray(a[name]) for name in LEVEL_INPUTS}
    inputs["w"] = cp.asarray(a["w"])
    for name in ("xland", "hpbl", "hpbl_hold"):
        inputs[name] = cp.asarray(a[name])
    out = launch_column(
        SAS_VARIANTS[14], inputs=inputs, znu=cp.asarray(group["znu"]),
        dt=group["dt"], dx=group["dx"], stepcu=group["stepcu"],
        mp_physics=group["mp_physics"], f_qc=group["f_qc"],
        f_qi=group["f_qi"], function=function)
    return {k: cp.asnumpy(v) for k, v in out.items()}


def _graded_fields(group):
    names = ["rthcuten", "rqvcuten", "rucuten", "rvcuten", "raincv",
             "pratec", "hbot", "htop"]
    if group["f_qc"]:
        names.append("rqccuten")
    if group["f_qi"]:
        names.append("rqicuten")
    return names


def _ulp_ledger(groups, answers, function=None):
    ledger = {}
    for gi, (group, want) in enumerate(zip(groups, answers)):
        got = _run(group, function)
        for name in _graded_fields(group):
            d = fp32_ulp_distance(got[name], want[name])

            prev = ledger.get(name, (0, 0, 0))
            different = (got[name].view(np.uint32) != want[name].view(np.uint32))
            ledger[name] = (max(prev[0], int(d.max())),
                            prev[1] + int(different.sum()), prev[2] + d.size)
    return ledger


def test_fixture_reaches_every_outcome(oracle):
    """The fixture is only evidence if it rains, refuses and spans dx."""
    groups, answers = oracle
    raining = sum(int((o["raincv"] > 0).sum()) for o in answers)
    quiet = sum(int((o["raincv"] == 0).sum()) for o in answers)
    assert raining >= 64 and quiet >= 64, (raining, quiet)
    assert {g["stepcu"] for g in groups} >= {1, 3, 5}
    assert {g["mp_physics"] for g in groups} >= {0, 1, 3, 8}
    assert min(g["dx"] for g in groups) < 1000 < 5000 < max(g["dx"] for g in groups)
    assert sum(g["ncol"] for g in groups) >= 64


def test_ksas_zero_ulp_against_wrf(oracle):
    groups, answers = oracle
    ledger = _ulp_ledger(groups, answers)
    arm = "strict" if os.environ.get("GPUWM_WRF_EXACT") == "1" else "default"
    report = {name: {"max_ulp": v[0], "differing_words": v[1], "words": v[2]}
              for name, v in sorted(ledger.items())}
    print(f"\nKSAS {arm} arm vs WRF v4.6.1: {report}")
    assert all(v[1] == 0 for v in ledger.values()), report


def test_planted_bit_is_caught(oracle):
    """KSAS_CXLAMU with its last mantissa bit flipped must not pass."""
    from woof.core.kernels import module_options, module_source

    src = module_source("sas")
    anchor = "#define KSAS_CXLAMU 1.0e-3f"
    assert src.count(anchor) == 1
    planted_value = np.frombuffer(
        (np.float32(1.0e-3).view(np.uint32) ^ np.uint32(1)).tobytes(),
        dtype=np.float32)[0]
    planted = src.replace(
        anchor, f"#define KSAS_CXLAMU {float(planted_value)!r}f")
    assert planted != src
    from woof.core.kernels import _load_module_no_ftz
    module = _load_module_no_ftz("sas-planted", planted)
    function = module.get_function("sas_ksas_column")
    groups, answers = oracle
    ledger = _ulp_ledger(groups, answers, function=function)
    assert any(v[0] > 0 for v in ledger.values()), ledger


def test_ksas_float32_words_against_wrf(oracle):
    groups, answers = oracle
    for gi, (group, want) in enumerate(zip(groups, answers)):
        got = _run(group)
        for name in _graded_fields(group):
            assert np.array_equal(
                got[name].view(np.uint32), want[name].view(np.uint32)), (gi, name)
