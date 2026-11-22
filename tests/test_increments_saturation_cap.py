"""The increment writer caps vapour at saturation where the increment moved it.

CPU only.  THE BREAKAGE THIS PREVENTS: a storm-scale child's first radar
analysis left 2.8 to 6.9 Mt of vapour above liquid saturation in a box
whose backgrounds held under 0.01 Mt, because
``apply_increments`` added the filter's theta and vapour and checked
nothing but finiteness and the moment pairs.  Condensed at the first step,
that vapour warmed 1 to 3 km by 1.4 to 2.1 K over the whole box.  The cap
removes the excess (saturation over liquid water at the resulting
temperature, never below the background's own supersaturation ratio) at
the cells the increment moved, on the live state and on a checkpoint alike,
and says so in the receipt.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from woof.core import constants as c
from woof.ensemble import increments as increments_module
from woof.ensemble.increments import (apply_increments,
                                       apply_increments_to_checkpoint)

SHAPE = (3, 2, 2)
PRESSURE = np.array([95000.0, 85000.0, 70000.0])
TEMPERATURE = np.array([298.0, 290.0, 280.0])


def _saturation(temperature, pressure):
    """Bolton/Tetens saturation over liquid water, written out here so the
    cap is checked against the formula and not against its own helper."""
    es = 1000.0 * c.SVP1 * np.exp(c.SVP2 * (temperature - c.SVPT0)
                                  / (temperature - c.SVP3))
    return c.EP2 * es / np.maximum(pressure - es, 1.0)


def _background(relative_humidity=0.9):
    """A column at the given liquid relative humidity, in the state's own
    equation of state: ``alt`` is the inverse dry density that makes
    ``p * alt = Rd * T * (1 + (Rv/Rd) qv)`` hold at ``T``."""
    p = np.broadcast_to(PRESSURE[:, None, None], SHAPE).astype(np.float64)
    t = np.broadcast_to(TEMPERATURE[:, None, None], SHAPE).astype(np.float64)
    qv = relative_humidity * _saturation(t, p)
    alt = c.RD * t * (1.0 + c.RVOVRD * qv) / p
    exner = (p / c.P0) ** c.RCP
    thb = np.full(SHAPE, 300.0)
    return {"p": p.copy(), "alt": alt, "qv": qv, "thp": t / exner - thb,
            "thb": thb, "u": np.zeros(SHAPE)}


def _state(fields):
    return types.SimpleNamespace(**{name: np.array(value, copy=True)
                                    for name, value in fields.items()})


def test_a_supersaturating_vapour_increment_is_capped_at_saturation():
    fields = _background(0.9)
    state = _state(fields)
    saturation = _saturation(np.broadcast_to(TEMPERATURE[:, None, None], SHAPE),
                             fields["p"])
    increment = {"qv": 0.3 * saturation}          # to 120 percent
    increment["qv"][0, 0, 0] = 0.05 * saturation[0, 0, 0]  # to 95 percent
    seen = []
    receipt = apply_increments(state, increment,
                               saturation_observer=seen.append)
    assert np.allclose(state.qv[0, 0, 0], 0.95 * saturation[0, 0, 0],
                       rtol=1e-12)
    assert np.all(state.qv <= saturation * (1.0 + 1e-12))
    capped = np.ones(SHAPE, bool)
    capped[0, 0, 0] = False
    assert np.allclose(state.qv[capped], saturation[capped], rtol=1e-9)
    block = receipt["saturation"]
    assert block["schema"] == increments_module.SATURATION_CAP_SCHEMA
    assert block["evaluated"] is True
    assert block["cells_capped"] == int(capped.sum())
    expected = float((0.2 * saturation)[capped].sum())
    assert block["vapour_removed_kg_kg_sum"] == pytest.approx(expected,
                                                              rel=1e-9)
    (change,) = seen
    assert np.all(change["qv"] <= 0.0)
    assert change["qv"][~capped].max() == 0.0
    assert float(-change["qv"].sum()) == pytest.approx(expected, rel=1e-6)


def test_the_background_s_own_supersaturation_ratio_is_kept():
    """A background already at 104 percent is the scheme's own state: an
    increment that moves it is capped at 104 percent of the new saturation,
    not taken down to 100, and a cell the increment left alone keeps its
    vapour exactly."""
    fields = _background(1.04)
    state = _state(fields)
    saturation = _saturation(np.broadcast_to(TEMPERATURE[:, None, None], SHAPE),
                             fields["p"])
    increment = {"qv": np.zeros(SHAPE)}
    increment["qv"][1] = 0.1 * saturation[1]
    receipt = apply_increments(state, increment)
    assert np.allclose(state.qv[1], 1.04 * saturation[1], rtol=1e-6)
    assert np.array_equal(state.qv[[0, 2]], fields["qv"][[0, 2]])
    assert receipt["saturation"]["cells_capped"] == 4
    assert receipt["saturation"]["background_supersaturated_cells_kept"] == 4


def test_a_cooling_theta_increment_takes_the_vapour_it_cannot_hold():
    """Theta alone, cooled 3 K from 99 percent: the resulting air holds
    less, and the cap writes vapour the increment never named."""
    fields = _background(0.99)
    state = _state(fields)
    receipt = apply_increments(state, {"thp": np.full(SHAPE, -3.0)})
    exner = (fields["p"] / c.P0) ** c.RCP
    cooled = np.broadcast_to(TEMPERATURE[:, None, None], SHAPE) - 3.0 * exner
    assert np.allclose(state.qv, _saturation(cooled, fields["p"]), rtol=1e-6)
    assert receipt["saturation"]["qv_written_beyond_the_increment"] is True
    assert receipt["saturation"]["cells_capped"] == int(np.prod(SHAPE))


def test_an_increment_that_stays_below_saturation_writes_exactly_its_sum():
    fields = _background(0.5)
    state = _state(fields)
    increment = {"qv": np.full(SHAPE, 1.0e-4), "thp": np.full(SHAPE, 0.2)}
    receipt = apply_increments(state, increment)
    assert np.array_equal(state.qv, fields["qv"] + 1.0e-4)
    assert receipt["saturation"]["cells_capped"] == 0
    assert receipt["saturation"]["cells_moved"] == int(np.prod(SHAPE))


def test_a_state_without_its_equation_of_state_is_recorded_not_capped():
    fields = _background(0.9)
    fields.pop("alt")
    state = _state(fields)
    receipt = apply_increments(state, {"qv": np.full(SHAPE, 0.01)})
    assert receipt["saturation"]["evaluated"] is False
    assert "alt" in receipt["saturation"]["reason"]


def test_the_checkpoint_writer_takes_the_same_cap(tmp_path):
    fields = _background(0.9)
    background = tmp_path / "gpuwmrst_d01.npz"
    np.savez(background, **{f"state/{name}": value.astype(np.float32)
                            for name, value in fields.items()
                            if name != "thb"})
    saturation = _saturation(np.broadcast_to(TEMPERATURE[:, None, None], SHAPE),
                             fields["p"])
    receipt = apply_increments_to_checkpoint(
        background, {"qv": 0.3 * saturation}, tmp_path / "analysis.npz")
    with np.load(tmp_path / "analysis.npz") as data:
        qv = data["state/qv"].astype(np.float64)
    assert np.allclose(qv, saturation, rtol=2e-6)
    assert receipt["saturation"]["cells_capped"] == int(np.prod(SHAPE))
