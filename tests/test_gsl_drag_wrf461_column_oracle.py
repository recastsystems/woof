"""gwd_opt = 3 (the GSL drag suite) against WRF v4.6.1's own Fortran.

The fixture (``tests/data/oracles/gsl_drag``) is WRF v4.6.1's
byte-unmodified ``phys/module_bl_gwdo_gsl.F`` (sha256 pinned, identical to
v4.7.1's) compiled at -O0 with gfortran 13.3 on glibc 2.39 by
``tools/gsl_drag_wrf461_oracle/build.sh`` and called as WRF v4.6.1's PBL
driver calls it, on 256 physically consistent columns: an HRRR-style
50-level grid of 192 columns (40 hand-built edge cases, then convective,
stable-night, windy, blocking, ocean, high-terrain, snow/ice, tropical and
critical-level families) at twelve grid lengths that reach every taper
branch and both 0.01 thresholds, and a 40-level grid of 64 columns at four;
four of the cases run with spp_pbl = 1 and its pattern.

Every output is held word for word: RUBLTEN, RVBLTEN (with and without the
diagnostics, which WRF's gwd_diags must not change), RTHBLTEN, the eight
DTAUX3D/DTAUY3D components and the eight column stresses.  Measured on the
RTX PRO 6000 (sm_120) at NVRTC 12.9.86, under GPUWM_WRF_EXACT=1 and under
default arithmetic: 0 ULP on every lane of every case
(tools/gsl_drag_wrf461_oracle, REPORT in the lane's receipt).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_gpu

REPO = Path(__file__).resolve().parents[1]
ORACLE = REPO / "tests" / "data" / "oracles" / "gsl_drag"
TOOL = REPO / "tools" / "gsl_drag_wrf461_oracle"
SPP_CASES = ("A/dx3000_spp", "A/dx5000_spp", "A/dx13000_spp", "B/dx9000_spp")


def test_fixture_is_the_pinned_wrf_and_its_own_receipts():
    pins = [line.split() for line in
            (TOOL / "SOURCES.sha256").read_text().splitlines()
            if line and not line.startswith("#")]
    recorded = [line.split() for line in
                (ORACLE / "SOURCES.sha256").read_text().splitlines() if line]
    assert recorded == pins
    # The routine under test is the same bytes the WRF v4.7.1 oracle pins.
    v471 = dict(reversed(line.split()) for line in
                (REPO / "tools" / "terrain_drag_wrf471_oracle"
                 / "SOURCES.sha256").read_text().splitlines()
                if line and not line.startswith("#"))
    assert dict(reversed(p) for p in pins)["phys/module_bl_gwdo_gsl.F"] == (
        v471["phys/module_bl_gwdo_gsl.F"])
    digest, name = (ORACLE / "FIXTURES.sha256").read_text().split()
    assert hashlib.sha256((ORACLE / name).read_bytes()).hexdigest() == digest
    harness = dict(reversed(line.split()) for line in
                   (ORACLE / "HARNESS.sha256").read_text().splitlines())
    for rel, want in harness.items():
        assert hashlib.sha256((TOOL / rel).read_bytes()).hexdigest() == want, rel
    symbols = (ORACLE / "libm-symbols.txt").read_text()
    reference = symbols.split("# WRF-flags")[0]
    assert "_ZGV" not in reference and "atan2f" in reference


def test_fixture_reaches_every_component_and_threshold():
    from woof.verify.gsl_drag_oracle import load

    cases = load()
    assert len([k for k in cases if not k.endswith("/inputs")]) == 20
    a = cases["A/inputs"]
    assert a["u3d"].shape == (192, 50) and cases["B/inputs"]["u3d"].shape == (64, 40)
    active = lambda case, comp: int(  # noqa: E731
        (cases[case][f"dtaux3d_{comp}"] != 0).any(axis=1).sum())
    # 1009 m: small-scale taper 0.00973 (off); 1010 m: 0.0108 (on).
    assert active("A/dx1009", "fd") == 0 and active("A/dx1010", "fd") > 100
    # 3637 m: large-scale taper below 0.01 (off); 3638 m: 0.0100098 (on).
    assert active("A/dx3637", "ls") == 0 and active("A/dx3638", "ls") > 100
    for comp in ("ls", "bl", "ss", "fd"):
        assert active("A/dx13000", comp) > 0, comp
    for key, case in cases.items():
        if not key.endswith("/inputs"):
            assert int(case["diags_agree"]) == 1, key
    tiny = np.finfo(np.float32).tiny
    fd = cases["A/dx3000"]["dtaux3d_fd"]
    assert ((fd != 0) & (np.abs(fd) < tiny)).any(), "no subnormal lane"


@requires_gpu
@pytest.mark.parametrize("case", [
    "A/dx750", "A/dx1000", "A/dx1009", "A/dx1010", "A/dx3000", "A/dx3637",
    "A/dx3638", "A/dx5000", "A/dx9000", "A/dx12000", "A/dx13000",
    "A/dx25000", "B/dx3000", "B/dx9000", "B/dx15000", "B/dx30000",
    *SPP_CASES])
def test_gsl_drag_matches_wrf461_word_for_word(case):
    from woof.verify.gsl_drag_oracle import load, measure

    cases = load()
    grid = case.split("/")[0]
    result = measure({case: cases[case], f"{grid}/inputs": cases[f"{grid}/inputs"]})
    bad = {field: (s["differ"], s["max_ulp"], s["columns"][:8])
           for field, s in result[case].items() if s["differ"]}
    assert not bad, bad


@requires_gpu
def test_the_spp_cases_fail_without_the_pattern():
    """The defect the spp cases caught: launched with rstoch = 0 (the port
    before development branch), every spp case differs from WRF."""
    from woof.verify.gsl_drag_oracle import load, measure

    cases = load()
    for case in SPP_CASES:
        grid = case.split("/")[0]
        result = measure({case: cases[case],
                          f"{grid}/inputs": cases[f"{grid}/inputs"]},
                         pass_spp=False)
        assert result[case]["rublten"]["differ"] > 0, case
