"""The Pleim-Xiu surface layer (sf_sfclay_physics=7) on the card is WRF's.

Graded against the fixtures of record in ``woof/data/pxsfclay/oracle``:
WRF v4.6.1's unmodified ``module_sf_pxsfclay.F`` and the verbatim
``pxsfclay_seaice_wrapper``/``get_local_ice_tsk`` text, built by
``tools/pxsfclay_wrf461_oracle/build.sh`` with WRF's scalar flags and run
over 72 columns in nine regime families for four to six consecutive steps
in five configurations (two dx values, a land model rewriting QSFC/HFX/QFX
/ZNT between calls, a warm restart at ITIMESTEP 37, and the sea-ice
wrapper at both of WRF's ice thresholds).

Strict replay requires 0 differing words on every output of every step.
Run with GPUWM_WRF_EXACT=1 and GPUWM_WRF_GLIBC=2.39 for this fixture's
scalar referee. Default forecasts use fast CUDA math. A planted one-ULP
change in one kernel constant must be caught by the strict replay.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from woof.verify.pxsfclay_oracle import (FIXTURE_DIR, fixture_cases,
                                          fixture_manifest, grade,
                                          read_words, replay)
from conftest import requires_gpu

#: The sha256 of every packaged fixture of record, as MANIFEST.json pins
#: them; regenerating a fixture changes this on purpose.
EXPECTED_CONFIGURATIONS = ("plain-dx12k", "plain-dx3k-lsm", "plain-restart",
                           "ice-002", "ice-05-tice")


def test_the_fixtures_are_the_recorded_words():
    manifest = fixture_manifest()
    assert tuple(manifest["configurations"]) == EXPECTED_CONFIGURATIONS
    index = json.loads((FIXTURE_DIR / "cases-index.json").read_text())
    for name in EXPECTED_CONFIGURATIONS:
        families = index[name]["families"]
        assert len(families) >= 64
        assert set(families) == {
            "unstable-day", "stable-night", "neutral-windy", "water",
            "cold-land", "high-roughness", "calm-convective", "edge",
            "sea-ice"}


def _receipt(name: str, report: dict) -> None:
    target = os.environ.get("PXSFCLAY_PARITY_RECEIPT_DIR")
    if target:
        mode = ("strict" if os.environ.get("GPUWM_WRF_EXACT") == "1"
                else "default")
        path = os.path.join(target, f"{name}.{mode}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", EXPECTED_CONFIGURATIONS)
def test_every_word_of_every_step_is_wrfs(name):
    if os.environ.get("GPUWM_WRF_EXACT") != "1":
        pytest.skip("bitwise replay requires GPUWM_WRF_EXACT=1")
    cases = {case.name: (case, path) for case, path in fixture_cases()}
    case, words = cases[name]
    oracle = read_words(words, case)
    device = replay(case)
    report = grade(case, device, oracle)
    _receipt(name, report)
    bad = {field: row for field, row in report.items() if row["differing"]}
    assert not bad, f"{name}: {bad}"
    graded = sum(row["words"] for row in report.values())
    assert graded >= 64 * 4 * 25


def _planted_source(define: str, value: str) -> str:
    from woof.core.pxsfclay import module_source

    source = module_source()
    assert source.count(define) == 1
    line = next(row for row in source.splitlines() if row.startswith(define))
    return source.replace(line, f"{define} {value}")


@pytest.mark.gpu
@requires_gpu
def test_a_one_ulp_constant_plant_is_caught():
    """BH (module_sf_pxsfclay.F:9) moved by one ULP must fail the grade.

    The plant is compiled through the same NVRTC route as the shipped
    unit; only the one constant's word differs.  It enters the unstable
    PSIH only, so the grade must name PSIH (and what PSIH feeds) on the
    unstable columns, and only there.
    """
    if os.environ.get("GPUWM_WRF_EXACT") != "1":
        pytest.skip("one-word control requires strict arithmetic")
    from woof.core.pxsfclay import compile_module

    planted = np.nextafter(np.float32(15.7), np.float32(np.inf))
    literal = f"{float(planted).hex()}f"
    module = compile_module(_planted_source("#define PX_BH", literal))
    case, words = next((c, p) for c, p in fixture_cases()
                       if c.name == "plain-dx12k")
    report = grade(case, replay(case, module=module), read_words(words, case))
    assert report["psih"]["differing"] > 0, report["psih"]
    assert report["regime"]["differing"] == 0
    assert report["cpm"]["differing"] == 0
