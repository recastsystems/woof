"""A calm level-1 wind through the UW PBL product launcher.

THE BREAKAGE THIS PREVENTS
--------------------------
camuwpbl divides by the level-1 wind speed (module_bl_camuwpbl_driver.F:
458-462, ``tauFac = rho*ust*ust/uMean``).  With u = v = 0 at level 1, or a
level-1 wind so small that u*u + v*v underflows in float32, the surface
stress is NaN.  On the card that NaN reached every tendency of the column
(rublten, rvblten and rthblten on all levels, tke_pbl, smaw3d, tpert2d,
qpert2d), and PhysicsDriver._run_uwpbl refused the step with
FloatingPointError: one exactly calm level-1 point ended the run.  WRF
v4.6.1 itself writes NaN TAURESX2D/TAURESY2D on that step and NaN
tendencies on the next (refute check of development branch, 2026-10-07,
tools/uwpbl_wrf461_verify/refute).  That is a WRF defect and is not
copied: a calm wind has no stress direction, so the kernel gives it zero
stress.  Every column whose level-1 speed is not zero keeps WRF's words,
which the second half of this test holds on the packaged fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.verify.uwpbl_oracle import (FULL_OUTPUTS, MASS_OUTPUTS,
                                       SURFACE_OUTPUTS, dims, step_arrays)
from test_uwpbl_launcher_wrf471_parity import _launch
from test_uwpbl_wrf471_parity import _fixture

OUTPUTS = MASS_OUTPUTS + FULL_OUTPUTS + SURFACE_OUTPUTS
CALM = 3      # columns 0..2 are made calm; the rest are the fixture's own


@pytest.mark.gpu
def test_a_calm_level1_wind_is_finite_and_no_other_column_moves():
    import cupy  # noqa: F401  (marks this test for -m "not gpu")

    fx = _fixture("g44")
    ncol, nk, _ = dims(fx)
    assert ncol > CALM
    a = step_arrays(fx, 1)
    calm = dict(a)
    for name in ("u", "v", "ust"):
        calm[name] = a[name].copy()
    calm["u"][0, 0] = 0.0                       # calm, ust > 0
    calm["v"][0, 0] = 0.0
    calm["u"][1, 0] = 0.0                       # calm and ust = 0
    calm["v"][1, 0] = 0.0
    calm["ust"][1] = 0.0
    calm["u"][2, 0] = np.float32(1e-30)         # u*u underflows in float32
    calm["v"][2, 0] = 0.0
    got = _launch(calm, nk, ncol)

    for name in OUTPUTS:
        value = np.asarray(got[name])
        if value.dtype == np.int32:
            continue
        bad = ~np.isfinite(value.reshape(ncol, -1)).all(axis=1)
        assert not bad.any(), (
            f"{name}: non-finite in columns {np.flatnonzero(bad).tolist()}")

    for name in OUTPUTS:
        want = np.asarray(a[name]).reshape(ncol, -1)[CALM:]
        have = np.asarray(got[name]).reshape(ncol, -1)[CALM:]
        if want.dtype != np.int32:
            want = want.view(np.uint32)
            have = have.astype(np.float32).view(np.uint32)
        assert np.array_equal(want, have), (
            f"{name}: a column with wind moved when another column was calm")
