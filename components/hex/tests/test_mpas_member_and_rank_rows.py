"""MPAS co-resident members and one-rank-per-card ranks are priced from measured rows.

THE BREAKAGES THESE PREVENT (both-routes matrix, hex 0.4.14, 2026-11-06):
RED 2: two co-resident MPAS members of the 120 km globe asked 36.08 GB (two whole solo floors) and were refused on
any card with less free, while the two members peaked at 10.8 GB.
RED 4: the two-card MPAS route asked 15.68 GB free on card 0 for a rank of 21,445 local columns, while the whole
two-card run peaked 15.5 GB summed over both cards.
"""
import pytest

from woof.hex.device_admission import (MPAS_MEMBER_ROWS, MPAS_RANK_CARD_ROWS, CardProfile, admit_co_tenancy,
                                      co_tenancy_limits, model_for_card, mpas_member_increment_bytes,
                                      mpas_rank_card_bytes, route_required_free_bytes)

GB = 1_000_000_000
PRO6000 = CardProfile(name="RTX PRO 6000", multiprocessors=188, max_threads_per_sm=1536)
SPPT = 72_682_624   # the stochastic arrays one SPPT member of x1.40962 holds (the matrix's refusal line)


def _group(members, free, cells=40_962, levels=55):
    model = model_for_card(PRO6000, "global", levels=levels)
    solo = route_required_free_bytes(cells, model, route="mpas")
    increment = mpas_member_increment_bytes(cells, levels)
    return admit_co_tenancy(mesh_kind="global", cells=cells, levels=levels, members=members,
                            card_bytes=97 * GB, free_bytes=free, solo_required_bytes=solo,
                            member_additional_bytes=(SPPT,) * members,
                            member_increment_bytes=None if increment is None else min(increment, solo))


def test_two_members_of_the_120km_globe_fit_a_24_gb_free_card():
    record = _group(2, 24 * GB)
    assert record["required_free_bytes"] < 24 * GB
    assert record["required_free_bytes"] < 36_077_128_754          # the refused price
    row = MPAS_MEMBER_ROWS[0]
    measured = dict(row.peaks_mib)[2] * (1 << 20)
    assert record["required_free_bytes"] > measured                # never below what the card held
    assert "measured per-member increment" in record["basis"]


def test_every_measured_group_is_priced_above_its_peak():
    row = MPAS_MEMBER_ROWS[0]
    for members, peak in row.peaks_mib:
        assert _group(members, 97 * GB)["required_free_bytes"] > peak * (1 << 20)


def test_an_oversubscribed_group_still_refuses_with_the_real_number():
    with pytest.raises(MemoryError) as refused:
        _group(4, 24 * GB)
    text = str(refused.value)
    assert "co-tenancy refuses 4 members" in text and "measured increment" in text and "memory limit 3" in text


def test_larger_meshes_keep_the_full_solo_price_per_member():
    assert mpas_member_increment_bytes(163_842, 55) is not None    # the 60 km row
    assert mpas_member_increment_bytes(655_362, 55) is None        # no measured row at 30 km yet
    record = _group(2, 97 * GB, cells=163_842)
    assert "measured per-member increment" in record["basis"]


def test_planner_and_admission_count_the_same_members():
    model = model_for_card(PRO6000, "global", levels=55)
    solo = route_required_free_bytes(40_962, model, route="mpas")
    _, memory = co_tenancy_limits(mesh_kind="global", cells=40_962, levels=55, card_bytes=97 * GB,
                                  free_bytes=24 * GB, solo_required_bytes=solo, member_additional_bytes=(SPPT,) * 2,
                                  member_increment_bytes=mpas_member_increment_bytes(40_962, 55))
    assert memory == 3
    _group(2, 24 * GB)


def test_a_rank_alone_on_its_card_is_priced_from_the_measured_row():
    model = model_for_card(PRO6000, "global", levels=55)
    row = MPAS_RANK_CARD_ROWS[0]
    fixed, per_column = row.terms_mib()
    assert abs(fixed + per_column * row.cells - row.solo_peak_mib) < 1.0
    assert abs(fixed + per_column * row.rank_cells - row.rank_peak_mib) < 1.0
    member = route_required_free_bytes(21_445, model, route="mpas")
    new = mpas_rank_card_bytes(21_445, 55, model)
    assert row.rank_peak_mib * (1 << 20) * 1.09 < new < 15_679_367_879 and new >= member   # the old ask: 15.68e9
    # 60 km ranks: a rank holds more than one member of its columns, so the row raises the one-member price
    big = MPAS_RANK_CARD_ROWS[1]
    assert mpas_rank_card_bytes(big.rank_cells, 55, model) > big.rank_peak_mib * (1 << 20) * 1.09
    # past every row (30 km halves) and on more levels: the one-member price raised by the measured ratio
    for local, levels in ((330_000, 55), (21_445, 59)):
        one = route_required_free_bytes(local, model, route="mpas")
        assert one <= mpas_rank_card_bytes(local, levels, model) <= 1.3 * one


def test_two_card_admission_uses_the_rank_row_and_one_card_is_unchanged():
    from woof.hex.mpasphys.memory_admission import rank_requirements
    model = model_for_card(PRO6000, "global", levels=55)
    basis = {}
    need = rank_requirements([21_445, 21_449], [0, 1], {0: PRO6000, 1: PRO6000}, 55, basis=basis)
    assert all(value < 24 * GB for value in need.values()) and need[0] < 15_679_367_879
    assert basis[0]["method"].startswith("one rank per card")
    assert rank_requirements([40_962], [0], {0: PRO6000}, 55)[0] == route_required_free_bytes(40_962, model,
                                                                                              route="mpas")
    # an oversubscribed split still asks its real number: 2 ranks of the 30 km mesh need more than 24 GB each
    big = rank_requirements([330_000, 330_000], [0, 1], {0: PRO6000, 1: PRO6000}, 55)
    assert all(value > 24 * GB for value in big.values())
