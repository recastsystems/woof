"""Same-host resident ranks are priced from same-host measurements.

Breakage: the graph-discounted single-card row admitted 15 km on four 96 GB
cards (664,586 local columns, 76,385 MiB predicted) and rank 0 died out of
memory at step 43 needing at least 98,088 MiB.
"""
from dataclasses import replace

from woof.hex.device_admission import (
    CURRENT_GLOBAL_ROW, KNOWN_CARDS, MIB, SAMEHOST_RANK_BYTES_PER_CELL,
    SAMEHOST_RANK_ENVELOPE_LOCAL_CELLS, SAMEHOST_RANK_ENVELOPE_PEAK_BYTES, _row_model,
    model_for_card, samehost_rank_model)

CAPACITY = 97887 * MIB
FREE_AT_15KM_ADMISSION = int(96_691.6 * MIB)


def _rank(cells):
    """A rank on the build the envelope was measured on (the 188 SM row of record)."""
    model = samehost_rank_model(_row_model("global/188sm"))
    return model, model.predict_bytes(cells) + model.margin_bytes()


def _rank_today(cells, card="96gib-188sm"):
    """A rank on today's build (CURRENT_GLOBAL_ROW, the hex memory census cuts)."""
    model = samehost_rank_model(model_for_card(KNOWN_CARDS[card], rollback_armed=False))
    return model, model.predict_bytes(cells) + model.margin_bytes()


def test_measured_15km_four_card_rank_is_refused_and_was_admitted_before():
    old = _row_model("global/188sm")
    graph_discounted = replace(old, bytes_per_cell=old.bytes_per_cell - 7017 * MIB / 655362)
    assert graph_discounted.predict_bytes(664586) / MIB < 76_400  # what admitted the run
    model, required = _rank(664586)
    assert required > 98_088 * MIB  # the measured out-of-memory lower bound
    assert required > FREE_AT_15KM_ADMISSION
    assert model.max_cells(FREE_AT_15KM_ADMISSION) < 664586


def test_envelope_passes_through_its_measurement_and_covers_the_same_size_runs():
    model, _ = _rank(1)
    assert abs(model.predict_bytes(SAMEHOST_RANK_ENVELOPE_LOCAL_CELLS)
               - SAMEHOST_RANK_ENVELOPE_PEAK_BYTES) < 1.0
    assert model.predict_bytes(327681) >= 61_345 * MIB  # 30 km on 2 cards, largest arm
    assert model.measured and model.rollback_bytes_per_cell == 0
    assert model.tiled_breakdown(822716) == _row_model("global/188sm").tiled_breakdown(822716)


def test_20km_on_four_cards_admits_and_on_two_refuses_on_the_enveloped_build():
    assert _rank(380000)[1] < FREE_AT_15KM_ADMISSION
    assert _rank(736266)[1] > CAPACITY


def test_today_a_rank_is_priced_at_the_measured_same_host_slope():
    """Two ranks on one RTX 5090 on today's build (hex/mem-census-cuts):
    x1.40962 peaked 6,054 MiB and x1.163842 14,818 MiB; the slope fitted from
    them prices every rank, above the single-card slope."""
    model, _ = _rank_today(1, "32gib-170sm")
    assert model.bytes_per_cell == SAMEHOST_RANK_BYTES_PER_CELL > CURRENT_GLOBAL_ROW["bytes_per_cell"]
    two_ranks = lambda a, b: model.predict_bytes(a) + model.predict_bytes(b)
    # Each rank charges its card's whole single-card core, so the one-card
    # pair (one process, one context) is over-priced, never under-priced.
    for (a, b), peak_mib in (((21553, 21557), 6054), ((83946, 83940), 14818)):
        assert two_ranks(a, b) >= peak_mib * MIB
        assert two_ranks(a, b) - peak_mib * MIB < 2 * model.core_bytes


def test_today_20km_fits_two_96gb_cards_and_30km_two_32gb_cards():
    # 20 km (1,472,532 cells) on two RTX PRO 6000: about 736,266 owned plus a
    # K=3 halo per rank.
    assert _rank_today(745_000)[1] < CAPACITY
    # 30 km (655,362 cells) on two RTX 5090: about 327,681 owned plus halo.
    assert _rank_today(332_000, "32gib-170sm")[1] < 31_500 * MIB


def test_unknown_and_supplied_rows_are_not_repriced():
    regional = model_for_card(KNOWN_CARDS["32gib-170sm"], "limited-area")
    assert samehost_rank_model(regional) is regional
    custom = replace(model_for_card(KNOWN_CARDS["96gib-188sm"]), bytes_per_cell=90000)
    assert samehost_rank_model(custom) is custom
    older = _row_model("global/170sm")
    assert samehost_rank_model(older) is older


def test_process_rank_row_covers_the_whole_24h_peak_and_failed_24gb_grant():
    model = samehost_rank_model(model_for_card(KNOWN_CARDS["32gib-170sm"]),
                                process_ranks=True)
    peak = sum(model.predict_bytes(n) for n in (21_445, 21_449))
    assert peak >= 22_524 * MIB
    assert peak + model.margin_bytes() > 24_000 * MIB
    assert peak + model.margin_bytes() < 31_000 * MIB
    assert model.margin_terms()["arena_placement"]["bytes"] == 3_347_054_592
    assert not model.measured and model.derived_from == "process-ranks-20261102/170sm"
    assert "derived equal split" in model.provenance
    assert model.predict_bytes(83_946) > model.predict_bytes(21_449)
    # An already priced row remains stable if a caller selects it twice.
    again = samehost_rank_model(model, process_ranks=True)
    assert again.predict_bytes(21_445) == model.predict_bytes(21_445)


def test_process_row_keeps_unmeasured_shapes_and_the_thread_row():
    for card, levels, configuration in ((KNOWN_CARDS["32gib-170sm"], 40, "global"),
                                         (KNOWN_CARDS["96gib-188sm"], 55, "global"),
                                         (KNOWN_CARDS["32gib-170sm"], 55, "limited-area")):
        model = model_for_card(card, configuration, levels=levels)
        before = samehost_rank_model(model)
        assert samehost_rank_model(model, process_ranks=True) == before
    model = model_for_card(KNOWN_CARDS["32gib-170sm"])
    assert samehost_rank_model(model, ranks=1, process_ranks=True) == samehost_rank_model(model, ranks=1)
