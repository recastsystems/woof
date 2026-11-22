"""Scale-aware ground snow cover fraction (Noah-MP ``opt_scf = 2``).

Re-implemented from published descriptions, not bitwise to upstream:
Abolafia-Rosenzweig et al. 2025, JAMES 17(6), e2024MS004704, Section 2.3,
Eq. 4-5 (CC BY 4.0).  The hand values below were worked from the paper's
equations with the coefficients it prints; the 36 km values are the ones the
paper itself prints (0.0779, 1.9713), to its four digits.

The option-off gate is the last test: ``opt_scf = 1`` must give the same
words as a call that never names the option.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import requires_gpu

from woof.core.noahmp_scf import (DX_CAP_KM, ground_snow_cover_fraction,
                                   scale_aware_parameters)


# DX [km] -> (SCfactor, m), worked by hand from Eq. 4-5:
#   SCfactor = 0.0062 sinh(0.0555 DX) + 0.0555,  m = 0.9713 + tanh(0.7436 DX)
HAND = {
    1.0: (0.05584427667954617, 1.6026152718242246),
    3.0: (0.056537076228698295, 1.9484757240673232),
    13.0: (0.060371632943659395, 1.9712999919729828),
    25.0: (0.06714089274494181, 1.9712999999999998),
    36.0: (0.0779399282440653, 1.9713),
}


@pytest.mark.parametrize("dx_km", sorted(HAND))
def test_parameters_match_hand_values(dx_km):
    scf, m = scale_aware_parameters(np.array([dx_km * 1000.0]))
    assert scf.dtype == np.float32 and m.dtype == np.float32
    want_scf, want_m = HAND[dx_km]
    assert scf[0] == np.float32(want_scf)
    assert m[0] == np.float32(want_m)


def test_cap_is_the_papers_36_km_solution():
    # Section 2.3: "an upper constraint on SCfactor and m equivalent to the
    # solutions of Equations 4 and 5 when DX = 36-km ... 0.0779 and 1.9713".
    scf36, m36 = scale_aware_parameters(np.array([36_000.0]))
    assert round(float(scf36[0]), 4) == 0.0779
    assert round(float(m36[0]), 4) == 1.9713
    scf, m = scale_aware_parameters(np.array([36_001.0, 50_000.0, 1.0e6]))
    assert (scf == scf36[0]).all() and (m == m36[0]).all()
    assert DX_CAP_KM == 36.0


def test_parameters_increase_with_grid_size():
    dx = np.array([500.0, 1000.0, 3000.0, 9000.0, 13000.0, 25000.0, 36000.0])
    scf, m = scale_aware_parameters(dx)
    assert (np.diff(scf) > 0).all()
    assert (np.diff(m) >= 0).all()


@pytest.mark.parametrize("bad", [0.0, -3000.0, float("nan"), float("inf")])
def test_refuses_non_physical_spacing(bad):
    with pytest.raises(ValueError, match="positive finite grid spacing"):
        scale_aware_parameters(np.array([3000.0, bad]))


def test_hand_case_snow_cover_fraction():
    # 20 cm of snow holding 50 mm SWE: rho_snow = 250 kg m-3 (Eq. 3).
    h, swe = 0.20, 50.0
    for dx_km, want in ((1.0, 0.6776307663900459), (3.0, 0.5323097485000048),
                        (13.0, 0.4961411639716552), (25.0, 0.45366830668840535)):
        s, m = HAND[dx_km]
        assert math.isclose(math.tanh(h / (s * (swe / h / 100.0) ** m)), want,
                            rel_tol=1e-14)
        got = float(ground_snow_cover_fraction(h, swe, dx_km * 1000.0))
        assert math.isclose(got, want, rel_tol=1e-14)
    # No snow: no cover.
    assert float(ground_snow_cover_fraction(0.0, 0.0, 3000.0)) == 0.0


def _seg1_state(snowh, sneqv, scf_dx=None):
    import cupy as cp
    n = len(snowh)
    f = lambda v: cp.full(n, np.float32(v), dtype=cp.float32)
    state = {
        "ice": cp.zeros(n, dtype=cp.int32), "ist": cp.ones(n, dtype=cp.int32),
        "uu": f(3.0), "vv": f(-2.0), "elai": f(0.0), "esai": f(0.0),
        "snowh": cp.asarray(np.asarray(snowh, np.float32)),
        "sneqv": cp.asarray(np.asarray(sneqv, np.float32)),
        # MODIS grassland (category 10) MPTABLE row: MFSNO 2.00, SCFFAC 0.020.
        "mfsno": f(2.00), "scffac": f(0.020),
        "z0sno": f(0.002), "z0mvt": f(1.09), "hvt": f(20.0), "cwpvt": f(0.18),
        "urban_flag": cp.zeros(n, dtype=cp.bool_), "zref": f(10.0),
    }
    if scf_dx is not None:
        state["scf_dx"] = cp.asarray(np.asarray(scf_dx, np.float32))
    return state


@requires_gpu
def test_slab_scale_aware_fsno_matches_host_reference():
    import cupy as cp
    from woof.core.noahmp_energy_slab import energy_seg1

    snowh = [0.0, 0.01, 0.05, 0.20, 0.20, 0.20, 0.80, 2.0]
    sneqv = [0.0, 1.0, 10.0, 50.0, 50.0, 50.0, 280.0, 700.0]
    dx = [3000.0, 3000.0, 3000.0, 1000.0, 3000.0, 25000.0, 3000.0, 50000.0]
    g = energy_seg1(_seg1_state(snowh, sneqv, dx), opt_scf=2)
    got = cp.asnumpy(g["fsno"]).astype(np.float64)
    want = ground_snow_cover_fraction(snowh, sneqv, dx)
    assert got[0] == 0.0
    # float32 statements (glibc powf/tanhf transcriptions) against a float64
    # reference: agreement to a few binary32 ulps.
    np.testing.assert_allclose(got, want, rtol=4e-6, atol=1e-7)
    # The 20 cm / 250 kg m-3 hand case at 1, 3 and 25 km.
    np.testing.assert_allclose(got[3:6], [0.6776307663900459,
                                          0.5323097485000048,
                                          0.45366830668840535], rtol=4e-6)


@requires_gpu
def test_option_off_is_the_unchanged_statement_sequence():
    import cupy as cp
    from woof.core.noahmp_energy_slab import energy_seg1

    snowh = [0.0, 0.01, 0.05, 0.20, 0.80, 2.0]
    sneqv = [0.0, 1.0, 10.0, 50.0, 280.0, 700.0]
    unnamed = energy_seg1(_seg1_state(snowh, sneqv))
    off = energy_seg1(_seg1_state(snowh, sneqv, [3000.0] * 6), opt_scf=1)
    on = energy_seg1(_seg1_state(snowh, sneqv, [3000.0] * 6), opt_scf=2)
    for name in ("fsno", "z0mg", "z0m", "zpd", "zlvl"):
        a = cp.asnumpy(unnamed[name]).view(np.uint32)
        b = cp.asnumpy(off[name]).view(np.uint32)
        assert (a == b).all(), name
    # The option is live: the grassland table SCF for 20 cm at 250 kg m-3 is
    # tanh(0.2 / (0.020 * 2.5**2)) = 0.92; the scale-aware 3 km SCF is 0.53.
    assert float(cp.asnumpy(on["fsno"])[3]) < float(cp.asnumpy(off["fsno"])[3])


def test_refuses_unknown_option():
    pytest.importorskip("cupy")
    from woof.core.noahmp_energy_slab import energy_seg1
    with pytest.raises(ValueError, match="opt_scf"):
        energy_seg1({}, opt_scf=3)
