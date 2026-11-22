"""TEMPO with RTE+RRTMGP: a species absent from a cell reaches radiation at the Thompson family's background radius.

THE BREAKAGE THIS PREVENTS (2026-11-09, TEMPO picked with RTE+RRTMGP on a hex zoom globe): TEMPO's effective-radius
routine returns zero for an absent species, the Thompson-family radiation branch gates effc/effi/effs against micron
bands that a zero is outside of, and the run stopped at its first radiation call after a microphysics step: "effc is
outside the physical-plausibility band [0.5, 100.0] microns".  CPU only.
"""
import numpy as np

from woof.core.rrtmgp import (
    EFFECTIVE_RADIUS_PLAUSIBLE_UM, THOMPSON_BACKGROUND_RADII_UM, absent_radii_at_background)
from woof.microphysics_schemes import TEMPO


def _radii():
    return {"effc": np.array([[0.0, 2.51, 50.0]], np.float32), "effi": np.array([[0.0, 2.51, 125.0]], np.float32),
            "effs": np.array([[0.0, 5.01, 999.0]], np.float32)}


def test_tempo_zeros_become_the_family_background_and_present_values_are_untouched():
    given = _radii()
    made = absent_radii_at_background(TEMPO.mp_id, given)
    for name, value in made.items():
        lower, upper = EFFECTIVE_RADIUS_PLAUSIBLE_UM[name]
        assert value.dtype == np.float32
        assert value[0, 0] == np.float32(THOMPSON_BACKGROUND_RADII_UM[name])
        np.testing.assert_array_equal(value[0, 1:], given[name][0, 1:])
        assert lower <= value.min() and value.max() <= upper
    # the state's own arrays are not written
    assert all(given[name][0, 0] == 0.0 for name in given)


def test_a_negative_or_non_finite_radius_still_reaches_the_gate():
    bad = {"effc": np.array([[-1.0, np.nan, np.inf, 0.0]], np.float32)}
    made = absent_radii_at_background(TEMPO.mp_id, bad)["effc"]
    assert made[0, 0] == -1.0 and np.isnan(made[0, 1]) and np.isinf(made[0, 2])
    assert made[0, 3] == np.float32(THOMPSON_BACKGROUND_RADII_UM["effc"])


def test_every_numbered_scheme_gets_its_radii_back_as_given():
    given = _radii()
    for mp_physics in (6, 8, 28, 18, 3):
        made = absent_radii_at_background(mp_physics, given)
        assert made is given
