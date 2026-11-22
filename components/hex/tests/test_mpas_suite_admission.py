"""The MPAS route is priced from the run's own suite where that suite has measured whole runs.

THE BREAKAGE THIS PREVENTS (hex 0.5.0, 2026-11-09): the 30 km RRFS hrrrv5 globe was priced from the
convection_permitting curve scaled by 59/55 levels.  One card: 72,596 MiB asked, 86,834 MiB held.  Two cards: 46,820
MiB asked per card, 65,810 held.  Four cards: 26,329 asked, 38,445 held.  A card with less free than the run needs
was admitted and the forecast would have run out of memory after its inputs were built.
"""
import pytest

from woof.hex.device_admission import (MIB, MPAS_RANK_CARD_ROWS, MPAS_ROUTE_RUN_PEAKS, CardProfile, model_for_card,
                                      mpas_rank_card_bytes, route_peak_rows, route_required_free_bytes,
                                      route_run_curve_mib)
from woof.hex.mpasphys.memory_admission import rank_requirements

PRO6000 = CardProfile(name="RTX PRO 6000", multiprocessors=188, max_threads_per_sm=1536)
CARDS = {device: PRO6000 for device in range(8)}
#: The three whole 30 km hrrrv5 runs of 2026-11-09: local columns per card, and each card's peak MiB.
MEASURED_30KM = {
    1: ((655_362,), (86_834,)),
    2: ((331_942, 331_934), (65_810, 65_491)),
    4: ((168_450, 167_789, 168_232, 168_012), (38_445, 37_229, 37_790, 37_537)),
}


@pytest.mark.parametrize("cards", sorted(MEASURED_30KM))
def test_the_30km_hrrrv5_globe_is_priced_above_what_each_card_held(cards):
    local, peaks = MEASURED_30KM[cards]
    need = rank_requirements(list(local), list(range(cards)), CARDS, 59, suite="hrrrv5")
    for device, peak in enumerate(peaks):
        assert peak * MIB < need[device] <= 1.12 * max(peaks) * MIB
    # the price this replaces, from the general curve
    old = rank_requirements(list(local), list(range(cards)), CARDS, 59)
    assert old[0] < max(peaks) * MIB < need[0]


def test_one_30km_hrrrv5_member_still_fits_an_empty_96_gb_card():
    need = rank_requirements([655_362], [0], CARDS, 59, suite="hrrrv5")[0]
    assert need == route_required_free_bytes(655_362, model_for_card(PRO6000, "global", levels=59), route="mpas",
                                             suite="hrrrv5")
    assert need < 96_785 * MIB        # the card's free memory before that run (101,486,362,624 bytes)


def test_a_suite_without_rows_of_its_own_keeps_the_general_curve():
    for levels in (55, 59, 41):
        model = model_for_card(PRO6000, "global", levels=levels)
        for cells in (2_562, 40_962, 163_842, 655_362, 737_281):
            general = route_required_free_bytes(cells, model, route="mpas")
            for suite in ("convection_permitting", "mesoscale_reference", "a-suite-nobody-measured"):
                assert route_required_free_bytes(cells, model, route="mpas", suite=suite) == general
                assert mpas_rank_card_bytes(cells, levels, model, suite=suite) == mpas_rank_card_bytes(cells, levels,
                                                                                                    model)
        assert route_required_free_bytes(40_962, model, route="arwen", suite="hrrrv5") == route_required_free_bytes(
            40_962, model, route="arwen")
    assert route_peak_rows("mpas", "convection_permitting") == route_peak_rows("mpas") == tuple(
        peak for peak in MPAS_ROUTE_RUN_PEAKS if peak.suite is None)


def test_the_hrrrv5_curve_passes_through_its_measured_runs_and_scales_with_levels():
    rows = route_peak_rows("mpas", "hrrrv5")
    assert rows and all(peak.suite == "hrrrv5" and peak.levels == 59 for peak in rows)
    for peak in rows:
        assert route_run_curve_mib(peak.cells, "mpas", 59, "hrrrv5") == pytest.approx(peak.peak_mib)
    # between the two: the 60 km globe ran in about 27 GB with strict numerics (a cross-check, not a row)
    assert 27_000 < route_run_curve_mib(163_842, "mpas", 59, "hrrrv5") < 29_000
    more_levels = route_required_free_bytes(40_962, model_for_card(PRO6000, "global", levels=65), route="mpas",
                                            suite="hrrrv5")
    assert more_levels == pytest.approx(13_228 * MIB * 1.10 * 65 / 59, rel=1e-6)
    fewer = route_required_free_bytes(40_962, model_for_card(PRO6000, "global", levels=55), route="mpas",
                                      suite="hrrrv5")
    assert fewer == pytest.approx(13_228 * MIB * 1.10, rel=1e-6)      # fewer levels are not priced below the run


def test_the_hrrrv5_rank_row_reads_its_measured_rank_sizes():
    row, = (r for r in MPAS_RANK_CARD_ROWS if r.suite == "hrrrv5")
    assert row.rank_mib(168_450) == pytest.approx(38_445) and row.rank_mib(331_942) == pytest.approx(65_810)
    assert row.rank_mib(655_362) == pytest.approx(86_834)
    assert row.rank_mib(84_000) < row.rank_mib(168_450) < row.rank_mib(250_000) < row.rank_mib(331_942)
    model = model_for_card(PRO6000, "global", levels=59)
    for local in (21_445, 84_060, 168_450, 331_942):
        rank = mpas_rank_card_bytes(local, 59, model, suite="hrrrv5")
        assert rank >= route_required_free_bytes(local, model, route="mpas", suite="hrrrv5")


def test_eight_cotenant_ranks_of_the_120km_hrrrv5_globe_keep_their_measured_price():
    # the claim shape: 8 ranks of x1.40962 on one 32 GB card (the co-tenant row was measured on hrrrv5 itself)
    local = [5_936, 5_937, 5_936, 5_937, 5_936, 5_937, 5_936, 5_936]
    card = CardProfile(name="RTX 5090", multiprocessors=170, max_threads_per_sm=1536)
    with_suite = rank_requirements(local, [0] * 8, {0: card}, 59, suite="hrrrv5")[0]
    assert with_suite == rank_requirements(local, [0] * 8, {0: card}, 59)[0]
    assert 27_958 * MIB < with_suite < 32_607 * MIB


def test_the_planners_read_the_suite_from_the_forecast_argv():
    from woof.hex.multi_member import argv_physics_route, argv_physics_suite
    assert argv_physics_suite(["forecast", "--mesh", "x1.655362", "--physics-suite", "hrrrv5"]) == "hrrrv5"
    default = ["forecast", "--mesh", "x1.163842"]
    assert argv_physics_route(default) == "mpas"
    assert not route_peak_rows("mpas", argv_physics_suite(default))[0].suite     # the general curve, as before
    assert argv_physics_suite(["forecast", "--mesh", "x1.163842", "--physics-suite", "arwen_mpas_column"]) is None


def test_the_ensemble_planner_asks_one_card_per_30km_hrrrv5_member(monkeypatch):
    from types import SimpleNamespace
    from woof.hex import global_ensemble
    card = {"name": "NVIDIA RTX PRO 6000 Blackwell", "free": 96_785 * MIB, "total": 97_887 * MIB, "sms": 188,
            "threads": 1536}
    monkeypatch.setattr(global_ensemble, "_probe_cards", lambda devices: [card, dict(card)])
    value = {"mesh": {"cells": 655_362}, "vertical": {"n_vert_levels": 59}, "members": [{}, {}],
             "commands": {"forecast": ["forecast", "--mesh", "x1.655362", "--physics-suite", "hrrrv5"]}}
    # one member holds 86.8 GB: two never share a 96 GB card
    assert global_ensemble.memory_group_limit(value, SimpleNamespace(devices=None)) == 1


def test_a_suite_row_never_lowers_a_price_and_small_meshes_keep_theirs():
    """46f6e59a extrapolated the hrrrv5 curve below its smallest measured run: a 4,186-cell hrrrv5 region went from
    7,642 to 9,704 MiB with no measurement behind it.  Below the smallest measured run the suite charges that run
    scaled by cells, so small meshes keep the general price; nowhere is the suite price below the general one."""
    for card in (PRO6000, CardProfile(name="RTX 5070 Ti", multiprocessors=70, max_threads_per_sm=1536)):
        for levels in (41, 55, 59, 65):
            model = model_for_card(card, "global", levels=levels)
            for cells in (2_562, 4_186, 10_242, 21_445, 30_000, 40_962, 84_060, 163_842, 330_000, 655_362, 2_621_442):
                general = route_required_free_bytes(cells, model, route="mpas")
                priced = route_required_free_bytes(cells, model, route="mpas", suite="hrrrv5")
                assert priced >= general, (cells, levels)
                assert mpas_rank_card_bytes(cells, levels, model, suite="hrrrv5") >= mpas_rank_card_bytes(cells, levels, model)
                if cells <= 21_445:
                    assert priced == general, (cells, levels)
    model = model_for_card(PRO6000, "global", levels=59)
    # a rank smaller than any measured hrrrv5 rank: the suite's one-member price raised by the measured rank ratio
    small = mpas_rank_card_bytes(21_445, 59, model, suite="hrrrv5")
    member = route_required_free_bytes(21_445, model, route="mpas", suite="hrrrv5")
    assert member <= small <= 1.45 * member


def test_the_footprint_model_floor_holds_on_an_unmeasured_vertical_grid():
    """46f6e59a reused the name of the footprint-model floor for a level count on vertical grids no run was measured
    on, so the floor was gone there."""
    from woof.hex.device_admission import required_free_bytes
    for levels in (41, 59, 65, 80):
        model = model_for_card(PRO6000, "global", levels=levels)
        for cells in (16, 2_562, 40_962, 655_362):
            floor = required_free_bytes(cells, model)
            for suite in (None, "hrrrv5", "convection_permitting"):
                assert route_required_free_bytes(cells, model, route="mpas", suite=suite) >= floor
