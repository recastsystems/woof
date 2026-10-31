"""card_fit: a profile with one whole run in a regime is shaped below it by its plain suite's curve.

THE BREAKAGE (mesh-gen-fast, RTX 4090, 2026-10-29): the hrrr-like-aerosol default had one one-card run
(286,530 cells, 44,989 MiB on a 96 GB card) and below it the plan used device_admission's WOOF slope (~49 KB per
cell), which left a ~33 GiB intercept: 40,000 cells required 37,094 MiB, 60,000 required 38,300 and 124,855
required 41,960.5, so no custom mesh worth running was admitted on a 24 GB card.
"""
from __future__ import annotations

import pytest

from woof.hex import card_fit, mesh_point

AEROSOL = dict(suite="convection_permitting", profile="hrrr-like-aerosol")


def _row(row_id, cells, peak, *, card="96gib-188sm", profile=None, status="complete", **extra):
    row = {"id": row_id, "route": "mpas", "suite": "convection_permitting", "cells": cells, "levels": 55,
           "cards": 1, "card": card, "status": status, "dt_seconds": 18, "hours": 1, "numerics": "fast",
           "peak_process_vram_mib": peak, **extra}
    if profile is not None:
        row["profile"] = profile
    return row


def test_below_its_one_run_a_profile_follows_its_plain_suite_scaled_at_the_run():
    plain = [_row("p40", 40000, 10000.0), _row("p400", 400000, 46000.0)]
    anchor = _row("a300", 300000, 42000.0, profile="hrrr-like-aerosol")
    card = mesh_point.resolve_card("rtx-pro-6000")
    for cells in (20000, 60000, 150000, 299999):
        fit = card_fit.predict(cells, card=card, levels=55, dt_seconds=18, hours=1, observations=[*plain, anchor],
                               **AEROSOL)
        plain_at = lambda x: 10000.0 + (46000.0 - 10000.0) * (x - 40000) / (400000 - 40000)  # noqa: E731
        expected = plain_at(cells) * 42000.0 / plain_at(300000)
        assert fit["whole_run_floor"]["curve_mib"] == pytest.approx(expected, abs=1e-3), cells
        assert "plain suite's curve times the run's ratio" in fit["whole_run_floor"]["basis"]
    # At the run itself the price is the run, plus the margin; past it, cell-linear as before.
    at = card_fit.predict(300000, card=card, levels=55, dt_seconds=18, hours=1, observations=[*plain, anchor],
                          **AEROSOL)
    assert at["whole_run_floor"]["curve_mib"] == pytest.approx(42000.0, abs=1e-3)
    past = card_fit.predict(360000, card=card, levels=55, dt_seconds=18, hours=1, observations=[*plain, anchor],
                            **AEROSOL)
    assert past["whole_run_floor"]["curve_mib"] == pytest.approx(42000.0 * 360000 / 300000, abs=1e-3)


def test_the_ratio_is_taken_against_the_plain_suite_on_the_runs_own_card_and_memory():
    """A roomy card's run is set against the plain suite measured on that card, never a tight card's curve."""
    plain = [_row("p40", 40000, 10000.0, card="32gib-170sm"), _row("p400", 400000, 46000.0, card="32gib-170sm"),
             _row("p40-roomy", 40000, 11000.0), _row("p300-roomy", 300000, 40000.0)]
    anchor = _row("a300", 300000, 42000.0, profile="hrrr-like-aerosol")
    fit = card_fit.predict(100000, card=mesh_point.resolve_card("rtx-5090"), levels=55, dt_seconds=18, hours=1,
                           observations=[*plain, anchor], **AEROSOL)
    # The 5090's own plain curve (its two 32 GB rows) at 100 k cells, times 42,000 / 40,000 (the 96 GB plain run).
    tight = 10000.0 + 36000.0 * 60000 / 360000
    assert fit["whole_run_floor"]["curve_mib"] == pytest.approx(tight * 42000.0 / 40000.0, abs=1e-3)


def test_with_no_plain_run_the_old_slope_arm_still_prices_the_profile():
    anchor = _row("a300", 300000, 42000.0, profile="hrrr-like-aerosol")
    fit = card_fit.predict(100000, card=mesh_point.resolve_card("rtx-pro-6000"), levels=55, dt_seconds=18, hours=1,
                           observations=[anchor], **AEROSOL)
    assert "device_admission's slope through it" in fit["whole_run_floor"]["basis"]


@pytest.mark.parametrize("cells,most", [(40000, 20000.0), (60000, 22000.0)])
def test_a_24_gb_card_admits_the_aerosol_default_on_small_custom_meshes(cells, most):
    fit = card_fit.predict(cells, card=mesh_point.resolve_card("rtx-4090"), levels=55, dt_seconds=60, hours=1,
                           **AEROSOL)
    assert fit["physics"]["profile"] == "hrrr-like-aerosol"
    assert fit["fits"] and fit["required_free_mib"] < most, fit["required_free_mib"]


@pytest.mark.parametrize("key", ["rtx-4090", "rtx-5090", "rtx-pro-6000"])
def test_the_aerosol_default_is_never_priced_under_its_plain_suite_or_its_one_card_run(key):
    card = mesh_point.resolve_card(key)
    previous = 0.0
    for cells in range(20000, 286531, 13325):
        plain = card_fit.predict(cells, card=card, levels=55, dt_seconds=18, hours=1, suite="convection_permitting")
        aerosol = card_fit.predict(cells, card=card, levels=55, dt_seconds=18, hours=1, **AEROSOL)
        assert aerosol["predicted_peak_mib"] >= plain["predicted_peak_mib"], (key, cells)
        assert aerosol["predicted_peak_mib"] >= previous, (key, cells)
        previous = aerosol["predicted_peak_mib"]
    anchor = card_fit.predict(286530, card=mesh_point.resolve_card("rtx-pro-6000"), levels=55, dt_seconds=18,
                              hours=1, **AEROSOL)
    assert anchor["predicted_peak_mib"] == pytest.approx(44989.0 * (1 + card_fit.WHOLE_RUN_MARGIN), abs=.01)


HELD_24GB = "rtx5090-held-to-24564mib-r124859-55l-mpas-cp-aerosol-1card-fast-1h"


def test_a_24_gb_card_admits_the_measured_120k_custom_mesh_above_its_measured_peak():
    """The 124,859-cell custom mesh ran the aerosol default +1 h on a card held to a 4090's 24,564 MiB (18,916 MiB
    peak): a 4090 plan admits it and prices it above that measurement."""
    fit = card_fit.predict(124859, card=mesh_point.resolve_card("rtx-4090"), levels=55, dt_seconds=60, hours=1,
                           **AEROSOL)
    held = next(row for row in card_fit.shipped_observations() if row["id"] == HELD_24GB)
    assert fit["fits"], fit["required_free_mib"]
    assert fit["predicted_peak_mib"] >= held["peak_process_vram_mib"] * (1 + card_fit.WHOLE_RUN_MARGIN)


@pytest.mark.parametrize("key", ["rtx-5090", "rtx-pro-6000"])
def test_a_run_held_to_less_memory_never_prices_a_card_with_more(key):
    """A hex process holds more where the card has room, so the held-24 GB run is no price for a 32 or 96 GB card."""
    shipped = card_fit.shipped_observations()
    without = [row for row in shipped if row["id"] != HELD_24GB]
    assert len(without) == len(shipped) - 1
    card = mesh_point.resolve_card(key)
    for cells in (40000, 100021, 124859, 150000, 286530):
        with_row = card_fit.predict(cells, card=card, levels=55, dt_seconds=60, hours=1, **AEROSOL)
        alone = card_fit.predict(cells, card=card, levels=55, dt_seconds=60, hours=1, observations=without,
                                 **AEROSOL)
        assert with_row["predicted_peak_mib"] == alone["predicted_peak_mib"], (key, cells)
