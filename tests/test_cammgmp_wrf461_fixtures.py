"""The CAMMGMP (mp_physics = 11) oracle fixtures and constant tables, no device.

Companion of tests/test_cammgmp_wrf461_parity.py (the device grading).

The fixtures (``tests/data/oracles/cammgmp``, PROVENANCE.md there) were
written by ``tools/cammgmp_wrf461_oracle``: a driver linked against the
SCALAR WRF build's libwrflib.a (gfortran 13.3 -O2 -fno-fast-math
-ffp-contract=off -fno-tree-vectorize, glibc 2.39) calling CAM_INIT,
CAMMGMP_INIT and CAMMGMP unchanged, on 88 columns in five groups (deep
convective, stratiform, mixed-phase, warm rain, dry; coupled ZM/UW-shallow
inputs; high aerosol; a carried-state restart group at itimestep 37; edge
cases), three or four consecutive steps each.

The pristine output is the stock diagnostic. The corrected ndrop output
is an altered-source attribution instrument. All inputs are generated
soundings; they are not a capture from a real forecast.

CPU tests pin the fixture words and the constant tables.  GPU tests run
the recovered column launcher (woof.core.cammgmp.cammgmp_step) as a free-running
chain and require 0 differing words under GPUWM_WRF_EXACT=1; the default
arm is held to the same 0 (the port pins every binary64 rounding with
intrinsics, so the arms compile to the same arithmetic); a planted one-bit
change in a kernel constant must be caught.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ORACLE = Path(os.environ.get("CAMMGMP_ORACLE_DIR")
              or ROOT / "tests/data/oracles/cammgmp")

#: sha256 of the fixtures of record (tools/cammgmp_wrf461_oracle/build.sh on
#: the sprint port box, 2026-10-08).  A regenerated fixture changes these on
#: purpose.
FIXTURE_SHA256 = {
    "cases.bin":
        "572c7369c69fd31ec11fa969e65df9f5c44d54d701f92a3bae078a062867c434",
    "fixed-out.bin":
        "a41200c56325cb4ae5be81536e5010dc4f47a69f805c1bad57d49a119fc4283a",
    "pristine-out.bin":
        "e00d7f8d1ea5feb3fb80af31cfcd123ca04303d9255eb925b5fc58b8d0452c4c",
    "helpers.bin":
        "b988943137b07cc328e5a17f1f9899fc19872ed171f0c7b67db53cad44deba93",
    "words.txt":
        "76d6098fc6be4e44d805c7207511d82fcb115dbbd80d42813e24a2ed55f626e6",
}


def _need(name):
    path = ORACLE / name
    if not path.exists():
        pytest.skip(f"CAMMGMP oracle fixture {path} is absent")
    return path


def test_the_fixtures_are_the_recorded_words():
    if ORACLE != ROOT / "tests/data/oracles/cammgmp":
        pytest.skip("pins apply to the packaged fixtures")
    for name, digest in FIXTURE_SHA256.items():
        data = _need(name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name


def test_the_kernel_words_are_the_oracles():
    """cammgmp_words.cuh is gen_words.py of the committed words + helpers."""
    from tools.cammgmp_wrf461_oracle.gen_words import (
        header_text, modal_words, read_words)
    words = read_words(_need("words.txt"))
    modal = modal_words(_need("helpers.bin"))
    header = (ROOT / "woof/core/kernels/cammgmp_words.cuh").read_text(
        encoding="ascii")
    assert header == header_text(words, modal)


def test_the_saturation_table_is_the_oracles():
    from woof.core.uwpbl_constants import ESTBL
    from tools.cammgmp_wrf461_oracle.gen_words import read_words
    words = read_words(_need("words.txt"))
    table = np.array([words[f"wv_saturation.estbl({i})"] for i in range(1, 251)])
    assert np.array_equal(table.view(np.uint64),
                          np.asarray(ESTBL, np.float64).view(np.uint64))


def test_the_deck_is_the_generator():
    """cases.bin is make_cases.py's output, so the regimes stay reproducible."""
    recorded = _need("cases.bin").read_bytes()
    from tools.cammgmp_wrf461_oracle.make_cases import build_deck, write_deck
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "cases.bin"
        write_deck(out, build_deck())
        assert out.read_bytes() == recorded


def test_the_deck_covers_the_regimes():
    from tools.cammgmp_wrf461_oracle.oracle_io import read_deck
    groups = read_deck(_need("cases.bin"))
    assert sum(g["ncol"] for g in groups) >= 64
    assert any(g["cu"] == 7 and g["shcu"] == 2 for g in groups)
    assert any(g["it0"] > 1 for g in groups)          # carried state
    assert all(g["nstep"] >= 3 for g in groups)       # multi-step
    assert any(g["accum"] > 1.0e9 for g in groups)    # high aerosol
