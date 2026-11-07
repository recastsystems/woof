"""The production pair against real inputs from an instrumented native run."""
from pathlib import Path

import pytest

from woof.wrf_exact import ENABLED


@pytest.mark.gpu
@pytest.mark.skipif(not ENABLED, reason="bitwise native proof uses GPUWM_WRF_EXACT=1")
@pytest.mark.parametrize("routine", ["temfpbl", "temfsfclay"])
def test_production_matches_real_native_run_words(routine):
    from tools.temf_wrf461_oracle.replay_native import read_capture, replay

    fixture = Path(__file__).parent / "fixtures/temf_real_native"
    capture = read_capture(fixture, routine)
    result = replay(capture, routine, stock=False)
    assert result["words"] == {"temfpbl": 7136, "temfsfclay": 160}[routine]
    assert result["differing_words"] == 0, result
