"""WOOF's TEMF pair against WRF v4.6.1, word for word, on the card.

The fixture is tools/temf_wrf461_oracle's: 216 columns on three vertical
grids (twelve regime families), 30 steps each, run through WRF's own
temfsfclay and temfpbl entry points with all their carried state.  The
replay (woof.verify.temf_replay) carries WOOF's state the same way, so
every step of every field is graded:

* the native PBL kernel and historical stock surface ABI
  against WRF modules with the declared math shim: 0 ULP, which grades the
  transcription itself independently of the fixes;
* a one-bit change planted in one constant word is caught.

Agreement is by per-step SHA-256 of each field's words (the full words
stay on the oracle box; tools/temf_wrf461_oracle/compare_arms.py reads
them), so a single differing word in any step fails its field.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from woof.verify.temf_oracle import ALL_FIELDS, read_cases, step_digests

pytestmark = pytest.mark.gpu

FIXTURES = Path(__file__).parent / "fixtures" / "temf_wrf461"
GRIDS = ("g35", "g45", "g61")


@pytest.fixture(scope="module")
def digests():
    return json.loads((FIXTURES / "digests.json").read_text())


def _mismatches(want: dict, got: dict) -> dict:
    assert set(want) == set(got) == set(ALL_FIELDS)
    bad = {}
    for name in ALL_FIELDS:
        assert len(want[name]) == len(got[name]), name
        steps = [n for n, (a, b) in enumerate(zip(want[name], got[name]))
                 if a != b]
        if steps:
            bad[name] = steps[:5]
    return bad


def _replay(grid: str, **kw) -> dict:
    from woof.verify.temf_replay import replay
    return step_digests(replay(read_cases(FIXTURES / f"cases-{grid}.bin"),
                               **kw))


@pytest.mark.parametrize("grid", GRIDS)
def test_the_stock_arm_is_the_declared_shim_oracle_word_for_word(grid, digests):
    got = _replay(grid, verify_stock_wrf=True)
    assert _mismatches(digests["arms"]["stock"][grid], got) == {}


def test_a_planted_one_bit_constant_change_is_caught(digests, monkeypatch):
    """Flip the low bit of one TEMFC word (0.608, the virtual-temperature
    factor) and the replay must leave the oracle on the first step."""
    import woof.core.temf as temf
    from woof.core.temf_constants import TEMF_CONSTANTS

    word = dict((n, w) for n, _, w in TEMF_CONSTANTS)["K0P608"]
    original = temf.module_source

    def planted(*, verify_stock_wrf=False):
        src = original(verify_stock_wrf=verify_stock_wrf)
        old = f"0x{word:08X}u,  // ["
        assert src.count(old) == 1
        return src.replace(old, f"0x{word ^ 1:08X}u,  // [")

    temf._module.cache_clear()
    monkeypatch.setattr(temf, "module_source", planted)
    try:
        got = _replay("g45", nsteps=2, verify_stock_wrf=True)
    finally:
        monkeypatch.setattr(temf, "module_source", original)
        temf._module.cache_clear()
    want = {name: steps[:2] for name, steps in
            digests["arms"]["stock"]["g45"].items()}
    bad = _mismatches(want, got)
    assert bad, "a flipped constant word left every digest unchanged"
    assert any(0 in steps for steps in bad.values())
