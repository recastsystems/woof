"""The ensemble headline kernel reads Q2 as a mixing ratio (CPU only).

WRF's Q2 is kg of vapour per kg of DRY air, so its vapour pressure is
``w p / (0.622 + w)``. The kernel used the specific-humidity form
``w p / (0.622 + 0.378 w)``, which read the 2 m dewpoint 0.1-0.25 K moist
(development branch). The CUDA source is checked as text and its
arithmetic re-evaluated in NumPy against the Rust import's reference values.
"""
from __future__ import annotations

import math

import numpy as np

from woof.ensemble.batch_product_output import _HEADLINE_SOURCE

MIXING_RATIO_FORM = ("__ddiv_rn(__dmul_rn(moisture, pressure), "
                     "__dadd_rn(0.622, moisture))")

# (specific humidity, pressure Pa, dewpoint K): the same worked values the
# Rust test mixing_ratio_humidity_tests pins (Bolton over water, e from the
# specific-humidity form, i.e. HRRR's SPFH -> DPT relation).
SPECIFIC_CASES = (
    (0.0150, 100_000.0, 293.5113271127926),
    (0.0040, 85_000.0, 271.58785978403705),
    (0.0200, 101_500.0, 298.449408723055),
)


def _kernel_dewpoint_k(q2, psfc):
    """The kernel's dewpoint arithmetic, in float64, on Q2 and PSFC."""
    moisture = np.float64(np.float32(q2))
    pressure = np.float64(np.float32(psfc))
    vapor = (moisture * pressure) / (0.622 + moisture)
    logarithm = math.log(max(vapor, 1.0) / 611.2)
    return float(np.float32(243.5 * logarithm / (17.67 - logarithm) + 273.15))


def test_kernel_uses_the_mixing_ratio_vapour_pressure():
    assert MIXING_RATIO_FORM in _HEADLINE_SOURCE
    assert "__dmul_rn(0.378" not in _HEADLINE_SOURCE


def test_kernel_dewpoint_matches_the_specific_humidity_reference():
    for q, p, td_ref in SPECIFIC_CASES:
        w = q / (1.0 - q)
        assert abs(_kernel_dewpoint_k(w, p) - td_ref) < 1e-3


# Real HRRR rows (hrrr.20260802 t18z wrfsfcf01): 2 m SPFH, surface PRES (Pa),
# 2 m DPT (K) as decoded. DPT is packed in 1/16 K steps, SPFH in 1e-5 steps,
# so one cell agrees to within one DPT step; the mean must sit within 0.02 K.
HRRR_ROWS = (
    (0.02012, 101_770.0, 298.63739013671875),
    (0.01919, 101_930.0, 297.82489013671875),
    (0.0046, 72_630.0, 271.32489013671875),
    (0.00869, 102_830.0, 285.44989013671875),
    (0.01471, 94_890.0, 292.38739013671875),
    (0.01902, 99_860.0, 297.38739013671875),
)


def _bolton_dewpoint_k(e_pa):
    ln = math.log(e_pa / 611.2)
    return 243.5 * ln / (17.67 - ln) + 273.15


def test_hrrr_spfh_reproduces_dpt_and_the_same_air_as_q2_agrees():
    errors = []
    for q, p, dpt in HRRR_ROWS:
        td_sh = _bolton_dewpoint_k(q * p / (0.622 + 0.378 * q))
        assert abs(td_sh - dpt) < 1.0 / 16.0
        errors.append(td_sh - dpt)
        assert abs(_kernel_dewpoint_k(q / (1.0 - q), p) - td_sh) < 1e-3
    assert abs(sum(errors) / len(errors)) < 0.02


def test_old_form_would_have_read_moist():
    w, p = 0.015, 100_000.0
    e_old = w * p / (0.622 + 0.378 * w)
    ln = math.log(e_old / 611.2)
    old = 243.5 * ln / (17.67 - ln) + 273.15
    assert old - _kernel_dewpoint_k(w, p) > 0.2
