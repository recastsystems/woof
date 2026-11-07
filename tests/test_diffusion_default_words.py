"""Default and strict diffusion must retain every sealed WRF output word."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]


_ARMS = [(scheme, strict)
         for scheme in ("km4", "km1", "km2", "km3", "diff1", "diff6", "advance_w", "w_damp")
         for strict in (False, True)]

#: DECLARED DIVERGENCE, (advance_w, default): the default build's advance_w
#: is not WRF's operation order.  Its default-arm WRF order was dropped at
#: integration (f66a15240, "Drop unqualified default acoustic arm": it failed
#: the real-column exactness gate), so a default acoustic launch keeps its
#: published arithmetic, woof.core.kernels.diffusion_kernel answers False
#: for the acoustic unit, and tests/assembled_legacy_proofs.py::
#: validate_published_acoustic pins that arithmetic's own words.  Against
#: WRF 4.6.1's advance_w on the upper-damping fixture it differs in this
#: many of its 153,600 words, measured per compute capability after the
#: damp_opt=3 coefficient took WRF's float32 product in the default build
#: (117,651 before it): sm_89 on the RTX 4090 (test host 1), sm_120 on the RTX
#: 5070 Ti (test host 4).  The count is the card's: NVRTC contracts the default
#: arithmetic per architecture.  The arm asserts the declared count rather
#: than dropping out, so a change in either direction means the default
#: acoustic arithmetic moved and must be declared again; the strict arm
#: holds 0 on every card.  A capability with no declared count still has
#: to show a partial divergence over the whole fixture, and prints its
#: count for declaring.
ADVANCE_W_DEFAULT_WORDS = 153_600
ADVANCE_W_DEFAULT_DECLARED = {"89": 111_509, "120": 111_728}


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("scheme,strict", _ARMS)
def test_diffusion_every_wrf_word(scheme, strict, tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GPUWM_WRF_EXACT")}
    if strict:
        env["GPUWM_WRF_EXACT"] = "1"
    receipt = tmp_path / "words.json"
    declared = scheme == "advance_w" and not strict
    process = subprocess.run([sys.executable, str(ROOT / "tools/diffusion_default_check.py"),
                              "--scheme", scheme, "--out", str(receipt)],
                             env=env, cwd=ROOT, check=not declared)
    result = json.loads(receipt.read_text())
    assert result["strict"] == strict
    assert result["words"] > 0
    if declared:
        import cupy
        capability = str(cupy.cuda.Device().compute_capability)
        print(f"advance_w default on sm_{capability}: "
              f"{result['different_words']} of {result['words']} words differ")
        assert process.returncode == 1 and result["pass"] is False
        assert result["words"] == ADVANCE_W_DEFAULT_WORDS
        if capability in ADVANCE_W_DEFAULT_DECLARED:
            assert result["different_words"] == ADVANCE_W_DEFAULT_DECLARED[capability]
        else:
            assert 0 < result["different_words"] < ADVANCE_W_DEFAULT_WORDS
        return
    assert result["pass"] and result["different_words"] == 0
