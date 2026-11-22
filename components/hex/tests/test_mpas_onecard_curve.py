"""One member of a uniform MPAS-physics globe on one card is priced from one-card runs, not from a rank's share.

THE BREAKAGE THIS PREVENTS (uniform globes on one 96 GB card, 2026-11-15): the one-member curve carried the 10 km
eight-card rank share (737,281 local columns, a 102 GB failed-allocation floor) as if it were one member on one card,
so past 655,362 cells it rose 0.436 MiB a cell.  The forecast door refused before the forecast a 28 km globe as a
110,515.9 MiB footprint, 25 km as 202,288.9 MiB and 22 km as 334,067.4 MiB, on a card reporting 96,691.6 MiB free,
while one card held 59,318 MiB at 654,432 cells and 62,778 MiB at 700,362 cells.
"""
import pytest

from woof.hex.device_admission import (MIB, MPAS_ONE_CARD_PEAKS_20261115, MPAS_RANK_CURVE_PEAKS, MPAS_RANK_SHARE_PEAK,
                                      MPAS_ROUTE_RUN_PEAKS, ROUTE_RUN_MARGIN, CardProfile, model_for_card,
                                      mpas_rank_card_bytes, route_required_free_bytes, route_run_curve_mib)

PRO6000 = CardProfile(name="RTX PRO 6000", multiprocessors=188, max_threads_per_sm=1536)
#: What the card reported free at each refusal of 2026-11-15.
FREE_MIB = 96_691.6


def need_mib(cells):
    return route_required_free_bytes(cells, model_for_card(PRO6000, "global", levels=55), route="mpas") / MIB


def test_the_rank_share_prices_ranks_only():
    assert MPAS_RANK_SHARE_PEAK not in MPAS_ROUTE_RUN_PEAKS
    assert MPAS_RANK_SHARE_PEAK in MPAS_RANK_CURVE_PEAKS
    assert not set(MPAS_ONE_CARD_PEAKS_20261115) & set(MPAS_RANK_CURVE_PEAKS)


@pytest.mark.parametrize("peak", MPAS_ONE_CARD_PEAKS_20261115, ids=lambda p: f"{p.cells}")
def test_each_one_card_run_is_priced_just_above_what_it_held(peak):
    assert route_run_curve_mib(peak.cells, "mpas", 55) == pytest.approx(peak.peak_mib)
    assert need_mib(peak.cells) == pytest.approx(peak.peak_mib * (1.0 + ROUTE_RUN_MARGIN), rel=1e-9)


def test_the_globes_refused_that_day_fit_and_the_ones_that_cannot_still_refuse():
    assert need_mib(751_272) < FREE_MIB          # 28 km, was 113,719.1 MiB required
    assert need_mib(942_432) < FREE_MIB          # 25 km, was 205,492.1 MiB required
    assert need_mib(1_216_922) > 97_887          # 22 km asks more than the whole card
    # never below the slope the route holds between its spread-out runs (60 km to 29 km)
    slope = (62_778.0 - 18_136.0) / (700_362 - 163_842)
    assert route_run_curve_mib(942_432, "mpas", 55) >= 62_778.0 + slope * (942_432 - 700_362) - 1e-6


def test_a_rank_of_a_split_member_keeps_its_price():
    model = model_for_card(PRO6000, "global", levels=55)
    # the rank share is a floor for a rank of its own size, as before
    assert mpas_rank_card_bytes(737_281, 55, model) >= 102_000_000_000 * (1.0 + ROUTE_RUN_MARGIN)
    for local in (21_449, 84_060, 331_942, 655_361, 700_000, 737_281):
        assert route_run_curve_mib(local, "mpas", 55, ranks=True) >= route_run_curve_mib(local, "mpas", 55) - 1e-6


def test_the_plan_door_prices_one_card_as_the_forecast_door_does():
    # The mesh plan refused 25 km on one card as 98,749.9 MiB against the card's 97,887 (2026-11-15), a second gate
    # before the forecast door's: its whole-run curve had no MPAS run past 30 km.
    from woof.hex import card_fit, mesh_point
    akey, _generator, nameplate = next(v for v in mesh_point.CARD_ALIASES.values() if v[0] == "96gib-188sm")
    card = {"admission_card": akey, "nameplate_mib": nameplate}
    plan = lambda cells: card_fit.predict(cells, card=card, levels=55, dt_seconds=150.0, hours=6.0)
    for cells in (654_432, 700_362, 751_272, 942_432):
        assert plan(cells)["fits"], cells
        assert plan(cells)["predicted_peak_mib"] == pytest.approx(need_mib(cells), rel=1e-6)
    assert not plan(1_022_592)["fits"] and not plan(1_216_922)["fits"]
