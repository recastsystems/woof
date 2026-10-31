"""Classic Thompson (mp=8) on the card against WRF v4.6.1, over regime
columns and thirty free-running steps, word for word.

THE BREAKAGE THIS PREVENTS
--------------------------
The real-column gates (tests/test_thompson_real_column_host_parity.py) hold
one call on 42 columns of one convective case, on the host build.  Regimes
they do not reach (a stable night with fog, a winter storm, cirrus, a
marine boundary layer, terrain at 3 to 5 km, a bright band, freezing rain,
and threshold columns at R1, subnormal, negative and signed-zero mass,
supercooled cloud below -38 C, 10 m layers) and the unit's own state fed
back into its next call were unguarded.  On them the unit differed from
unmodified WRF in words that the real columns never showed: WRF's
out-of-bounds read of the rain-collecting-graupel tables past the end of
their allocation (heap contents), and fresh graupel that WRF leaves
unflagged at the TAU+1 update.

The fixture (``tests/data/thompson_mp8_regime_columns_wrf461.npz``, cut by
``tools/thompson_mp8_column_oracle/oracle.py fixture``) holds 80 columns of
50 levels (make_columns.py, seed 20261007), dt 20 s, and the answers at
steps 1 and 30 of two WRF builds of the same pinned v4.6.1 module
(gfortran 13.3, glibc 2.39, -O2 -fno-tree-vectorize): the repaired control
(repair_wrf461.py) and unmodified WRF.  This test runs the production
adapter on the current device for thirty calls, handing each call the
state the last one left (mp_gt_driver's INTENT(INOUT) arrays), and holds:

* every output word at both steps equal to UNMODIFIED WRF's (the strict
  unit is the WRF transcription and keeps WRF's two classic-Thompson
  defects; 0 of 105,120 words differ, RTX 4090 sm_89, 2026-10-29), and
* the words where the unit differs from the repaired control equal, cell
  for cell, to the words where the repaired control differs from WRF.

The word-for-word claim uses ``GPUWM_WRF_EXACT=1``.  Run the adapter in a
fresh process so that selector takes effect before any kernel imports;
the default forecast adapter is a different, faster implementation.
"""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "thompson_mp8_regime_columns_wrf461.npz"
REPAIRED_SHA = "ccd74ab8a7296b1f58bf08b4c1c12ca44d7ea495860e210ee5ef9f901e8ab62d"
PINNED_SHA = "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802"


def test_the_regime_fixture_covers_every_regime_and_names_its_builds():
    z = np.load(FIXTURE)
    regimes = {str(label).split("/")[0] for label in z["labels"]}
    assert regimes == {"convective", "stable_night", "snow", "cirrus",
                       "ocean", "high_terrain", "bright_band",
                       "freezing_rain", "edge"}
    assert z["in_p"].shape == (80, 50)
    assert int(z["steps"]) == 30 and float(z["dt"]) == 20.0
    assert tuple(z["kept_steps"]) == (1, 30)
    receipt = str(z["receipt"])
    assert PINNED_SHA in receipt and REPAIRED_SHA in receipt
    assert "repaired libmvec_symbols = 0" in receipt
    # The two builds differ (the defects act on these columns), so the
    # footprint half of the GPU gate is not vacuous.
    moved = sum(int((z[f"repaired_{k}"].view(np.uint32)
                     != z[f"oracle_{k}"].view(np.uint32)).sum())
                for k in ("qr", "qg", "nr", "refl"))
    assert moved > 1000, moved


@requires_gpu
def test_classic_thompson_on_the_card_is_wrf461_word_for_word():
    from woof.physics_compat import thompson_table_root

    if not (Path(thompson_table_root()) / "freezeH2O.dat").exists():
        pytest.skip("the classic Thompson table set is not staged "
                    "(freezeH2O.dat is not packaged)")
    if __name__ != "__main__":
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())], cwd=ROOT,
            env=dict(os.environ, GPUWM_WRF_EXACT="1", GPUWM_NUMERICS="strict",
                     PYTHONPATH=os.pathsep.join(
                         [str(ROOT), os.environ.get("PYTHONPATH", "")])),
            capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr
        return

    import cupy as cp
    from woof.wrf_exact import ENABLED
    from tools.thompson_mp8_column_oracle import oracle

    assert ENABLED, "the WRF word proof requires the strict adapter"

    z = np.load(FIXTURE)
    have = oracle.run_woof(z, cp)
    keep = [s - 1 for s in z["kept_steps"]]
    have = {k: v[keep] for k, v in have.items()}
    repaired = {k: z[f"repaired_{k}"] for k in oracle.OUT3 + oracle.OUT2}
    pristine = {k: z[f"oracle_{k}"] for k in oracle.OUT3 + oracle.OUT2}

    unit = oracle._diff_mask(have, pristine)
    bad = {k: int(m.sum()) for k, m in unit.items() if m.any()}
    assert not bad, f"words differing from unmodified WRF v4.6.1: {bad}"

    unit_moves = oracle._diff_mask(have, repaired)
    repair_moves = oracle._diff_mask(oracle._microns(repaired), pristine)
    for name in unit_moves:
        assert np.array_equal(unit_moves[name], repair_moves[name]), name


if __name__ == "__main__":
    test_classic_thompson_on_the_card_is_wrf461_word_for_word()
