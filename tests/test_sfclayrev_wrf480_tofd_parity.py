"""Revised MM5 surface layer against WRF v4.8.0, with and without KIM form drag.

WRF v4.8.0 (physics_mmm 550b5b4) adds ``kim_tofd`` to sf_sfclay_physics=1:
KIM turbulent orographic form drag on CD, CDA and UST (sf_sfclayrev.F90:
707-720, 779-783), default off.  ``tools/sfclayrev_wrf480_oracle/`` links a
column driver against the native 4.8.0 strict serial referee build's own
objects (sf_sfclayrev.o and module_sf_sfclayrev.o out of the build whose
wrf.exe is f6f47f86...; GNU Fortran 13.3.0, -O2 -fno-fast-math
-ffp-contract=off -fno-tree-vectorize, glibc 2.39) and runs it on the 4.6.1
oracle's 110 stress columns, each twice with a sub-grid orographic standard
deviation that straddles every TOFD branch (``make_inputs.py``): 220 columns,
six switch arms with kim_tofd off and the same six with it on, three chained
steps, 34 output words each.

The breakage this prevents: a form-drag port that moves CD/CDA/UST (and every
flux, 10 m wind and 2 m value downstream of UST) off WRF 4.8.0's words, or a
kim_tofd = .false. run that no longer matches 4.8.0's default.  Graded under
strict numerics, the arithmetic every bitwise proof runs.

One word per flux-off arm is not compared, as in the 4.6.1 test: with
isfflx = 0 WRF copies an unassigned local into LH.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "sfclayrev_wrf480_oracle"
DATA = REPO / "tests" / "data" / "oracles" / "sfclayrev480"
INPUTS = DATA / "sfclayrev480-inputs.hex"
OUTPUTS = DATA / "sfclayrev480-outputs.hex"


def _tool(name):
    spec = importlib.util.spec_from_file_location(f"sfclayrev480_{name}", TOOL / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _receipt():
    sums = {}
    for line in (DATA / "referee-receipt.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            digest, path = line.split()
            sums[Path(path).name] = digest
    return sums


def test_fixture_matches_its_referee_receipt():
    sums = _receipt()
    assert hashlib.sha256(INPUTS.read_bytes()).hexdigest() == sums["sfclayrev480-inputs.hex"]
    assert hashlib.sha256(OUTPUTS.read_bytes()).hexdigest() == sums["sfclayrev480-outputs.hex"]
    assert hashlib.sha256((TOOL / "run_sfclayrev480.F90").read_bytes()).hexdigest() \
        == sums["run_sfclayrev480.F90"], "the driver changed after the fixture was written"
    receipt = (DATA / "referee-receipt.txt").read_text()
    assert "13.3.0" in receipt and "f6f47f86de59" in receipt and "_ZGV" not in receipt
    # The referee's own scheme object (WRF v4.8.0 06d4240ae989 strict serial).
    assert sums["sf_sfclayrev.o"] == "9baef629e1b28209c34d525b3431e3dd7f30f9eefe6242dd505e4040c80f6950"


def test_make_inputs_reproduces_the_committed_columns(tmp_path):
    _tool("make_inputs").write(tmp_path)
    assert (tmp_path / "sfclayrev480-inputs.hex").read_bytes() == INPUTS.read_bytes()


def test_fixture_reaches_both_sides_of_every_tofd_branch():
    compare = _tool("compare")
    from woof.core.physics_inventory import SFCLAY_OUTPUTS
    _, inputs, want = compare.load(INPUTS, OUTPUTS, 3)
    varf = inputs["varf"]
    assert (varf == 5.0).any() and (varf > 5.0).any() and (varf < 5.0).any()
    cd = SFCLAY_OUTPUTS.index("cd")
    moved = (want[6:, :, :, cd] != want[:6, :, :, cd]).any(axis=(0, 1))
    assert moved.sum() > 100 and not moved[varf <= 5.0].any()
    # The za clamp of zf = min(varf*tofd_factor, za) acts on the 25 km column set.
    za = 0.5 * inputs["dz8w"]
    assert (varf * np.float32(0.003) > za).any()


@requires_gpu
def test_revised_mm5_matches_wrf480_with_and_without_form_drag():
    from woof import wrf_exact
    if not wrf_exact.ENABLED:
        pytest.skip("graded under strict numerics (GPUWM_WRF_EXACT=1)")
    compare = _tool("compare")
    cases, inputs, want = compare.load(INPUTS, OUTPUTS, 3)
    got = compare.port(inputs, want.shape, 3, 0.003)
    report = compare.grade(cases, want, got)
    assert report["compared_words"] == 267_960
    assert report["different_words"] == 0, report["first_differences"]
