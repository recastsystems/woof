"""Background-source selection for the cycling radar-DA nowcast.

Four things are under test and they are the four the switch can get
wrong: which cycle a source resolves to, whether the ensemble it feeds
is real, whether a domain the source cannot carry is refused before
anyone pays for a fetch, and whether selecting the default reproduces
the behaviour that was already shipped.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from woof.da import background  # noqa: E402
from woof.fetch import Area  # noqa: E402


WIND = [{"name": "u", "amplitude": 1.5, "length_scale_km": 150.0},
        {"name": "v", "amplitude": 1.5, "length_scale_km": 150.0}]


# ---------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------

def test_the_registry_is_the_roster_and_gfs_is_the_default():
    assert background.DEFAULT_BACKGROUND_SOURCE == "gfs"
    from woof.source_adapters import source_adapters
    assert set(background.BACKGROUND_SOURCES) == {row.source_id for row in source_adapters() if row.runnable}


def test_an_unknown_source_refuses_by_naming_the_roster():
    from woof.source_adapters import source_adapters
    with pytest.raises(background.BackgroundError, match="available sources") as caught:
        background.resolve_background_source("hrrre")
    assert str(caught.value).split('available sources: ', 1)[1].split(', ') == [row.source_id for row in source_adapters()]


def test_hrrr_carries_condensate_and_gfs_does_not():
    """The single largest first-guess difference, stated as data.

    Radial velocity cannot create condensate, so a background that
    starts with none is starting the filter from a cloud-free world.
    Recorded here because the receipt reports it and a comparison has to
    be able to attribute to it.
    """

    assert background.BACKGROUND_SOURCES["gfs"].initial_hydrometeors \
        == "explicit zero (WRF Vtable.GFS parity)"
    assert "QG" in background.BACKGROUND_SOURCES["hrrr"].initial_hydrometeors


# ---------------------------------------------------------------------
# the cycle each source resolves to
# ---------------------------------------------------------------------

@pytest.mark.parametrize("init_hour,expected_cycle_hour,expected_lead", [
    (4, 0, 4), (5, 0, 5), (6, 0, 6), (11, 6, 5),
])
def test_gfs_walks_back_in_six_hour_steps_past_the_four_hour_lag(
        init_hour, expected_cycle_hour, expected_lead):
    init = datetime(2026, 8, 5, init_hour)
    plan = background.plan_background_cycle(
        "gfs", init=init, now=init + timedelta(hours=3),
        run_seconds=3600.0)
    assert plan.cycle.hour == expected_cycle_hour
    assert plan.forecast_start_hour == expected_lead


def test_hrrr_hourly_cycling_puts_the_background_hours_fresher():
    """The reason to offer HRRR at all, measured against GFS on one init.

    A six-hourly cycle with a four-hour wait forces an init onto f004 or
    later; an hourly cycle with a ~one-hour wait can land on the init's
    OWN cycle.  That difference is the background age, and it is the
    number the switch buys.
    """

    init = datetime(2026, 8, 5, 4)
    now = init + timedelta(hours=2)
    gfs = background.plan_background_cycle(
        "gfs", init=init, now=now, run_seconds=3600.0)
    hrrr = background.plan_background_cycle(
        "hrrr", init=init, now=now, run_seconds=3600.0)
    assert gfs.forecast_start_hour == 4
    assert hrrr.forecast_start_hour == 0
    assert hrrr.cycle == init


def test_hrrr_waits_longer_for_a_window_that_needs_a_later_lead():
    """Publication is a stream, not an event, and the wait tracks the lead."""

    init = datetime(2026, 8, 5, 4)
    now = init + timedelta(minutes=70)
    short = background.plan_background_cycle(
        "hrrr", init=init, now=now, run_seconds=3600.0)
    long = background.plan_background_cycle(
        "hrrr", init=init, now=now, run_seconds=12 * 3600.0)
    assert long.publication_lag_seconds > short.publication_lag_seconds
    assert long.cycle <= short.cycle


@pytest.mark.parametrize("lead,hours,why", [
    # f001: every hour reaches it; the synoptic rule (0.83 h + 0.0203 h
    # per lead, 51.0 min) is slower than the off-synoptic one (0.8 h +
    # 0.0324 h per lead, 49.9 min), so it sets the wait.
    (1, 0.83 + 0.0203 * 1, "synoptic rule, every hour reaches f001"),
    # f018: the off-synoptic runs still reach it and their steeper slope
    # is slower (83.0 min against the synoptic 71.7 min).
    (18, 0.8 + 0.0324 * 18, "off-synoptic rule, the last lead they reach"),
    # f019: only the synoptic runs reach it, so their f019-f048 rule
    # (0.87 h + 0.0183 h per lead, 73.1 min) decides; the off-synoptic
    # line would have said 84.9 min for a lead those runs never post.
    (19, 0.87 + 0.0183 * 19, "synoptic f019+ rule, the only hours reaching"),
    (48, 0.87 + 0.0183 * 48, "synoptic f019+ rule, the only hours reaching"),
])
def test_the_hrrr_wait_is_its_row_s_expected_line(lead, hours, why):
    """The f001 and f018 waits a HRRR plan takes, and why (A136 L1 follow-ups).

    The wait is the route table's hrrr row on its EXPECTED line (the
    earliest posting seen, what woof.source_posting.expected_at gives),
    the slowest of the cycle hours whose horizon reaches the lead.  L1 had
    it on the latest posting seen, over all 24 hours: f001 107 min and
    f018 140 min, which the late 30 Sep 06Z and 07Z runs set, so every
    plan rode a background about an hour older than the one that was out.
    A cycle later than its expected line is waited for by the run's fetch
    until its row calls it late (tools/da_nowcast.py fetch_cmd), never
    skipped or failed.  The retired constant was 50 min + 3.06 min/lead.
    """

    lag = background.BACKGROUND_SOURCES["hrrr"].lag_seconds(lead)
    assert lag == pytest.approx(hours * 3600.0), why


def test_a_window_past_the_cycle_horizon_refuses_rather_than_truncates():
    """HRRR's 18 h (48 h at the synoptic hours) is the source's ceiling."""

    init = datetime(2026, 8, 5, 5)          # a 05Z cycle stops at f018
    selected = background.plan_background_cycle(
        "hrrr", init=init, now=init + timedelta(hours=2), run_seconds=30 * 3600.0)
    assert selected.cycle == datetime(2026, 8, 5, 0)
    assert selected.forecast_start_hour == 5 and selected.forecast_end_hour == 35
    with pytest.raises(background.BackgroundError, match="No complete"):
        background.plan_background_cycle(
            "hrrr", init=init, now=init + timedelta(hours=2),
            run_seconds=49 * 3600.0)


def test_the_extended_synoptic_hrrr_cycle_carries_a_longer_window():
    init = datetime(2026, 8, 5, 0)
    plan = background.plan_background_cycle(
        "hrrr", init=init, now=init + timedelta(hours=2),
        run_seconds=3600.0)
    assert plan.horizon_hours == 48


def test_an_init_off_the_hour_refuses():
    with pytest.raises(background.BackgroundError, match="whole hour"):
        background.plan_background_cycle(
            "hrrr", init=datetime(2026, 8, 5, 4, 15),
            now=datetime(2026, 8, 5, 7), run_seconds=3600.0)


# ---------------------------------------------------------------------
# how the ensemble is constructed, and when it refuses
# ---------------------------------------------------------------------

def test_every_member_carries_its_own_construction_record():
    plan = background.plan_member_backgrounds(
        control_name="control", members=3, seed=20260731,
        perturbed_fields=WIND, perturbed_species=[])
    assert [entry.trajectory for entry in plan] == \
        ["control", "0", "1", "2"]
    assert plan[0].perturbation_seed is None
    assert [entry.perturbation_seed for entry in plan[1:]] == \
        [20260731, 20260732, 20260733]
    assert all(entry.perturbed == ("u", "v") for entry in plan[1:])


def test_a_perturbation_that_touches_nothing_is_a_fabricated_ensemble():
    """The refusal the switch most needs, and it is source-independent.

    Every amplitude zero produces N bit-identical copies of the control.
    The member count in the receipt would be accurate and the ensemble
    would not be, so this refuses before a card is touched.
    """

    dead = [{"name": "u", "amplitude": 0.0, "length_scale_km": 150.0},
            {"name": "v", "amplitude": 0.0, "length_scale_km": 150.0}]
    with pytest.raises(background.BackgroundError,
                       match="fabricated ensemble"):
        background.plan_member_backgrounds(
            control_name="control", members=10, seed=1,
            perturbed_fields=dead, perturbed_species=[])


def test_hydrometeor_species_count_as_perturbed_fields():
    """A run whose only live amplitude is on species is a real ensemble."""

    dead_wind = [{"name": "u", "amplitude": 0.0, "length_scale_km": 150.0}]
    species = [{"mass_field": "qr", "amplitude": 0.7,
                "length_scale_km": 60.0}]
    plan = background.plan_member_backgrounds(
        control_name="control", members=2, seed=5,
        perturbed_fields=dead_wind, perturbed_species=species)
    assert plan[1].perturbed == ("qr",)


def test_zero_members_refuses():
    with pytest.raises(background.BackgroundError, match="positive integer"):
        background.plan_member_backgrounds(
            control_name="control", members=0, seed=1,
            perturbed_fields=WIND, perturbed_species=[])


def test_a_lagged_cycle_ensemble_refuses_and_says_why():
    """Named, priced and refused -- not silently replaced by the one mode.

    A lagged HRRR ensemble is the obvious thing to reach for once the
    cycle is hourly, and it is genuinely not reachable: each member
    would need its own prepared case, and the ensemble generation binds
    one prepared-content digest for all of them.
    """

    with pytest.raises(background.BackgroundError,
                       match="prepared_content_sha256"):
        background.plan_member_backgrounds(
            control_name="control", members=4, seed=1,
            perturbed_fields=WIND, perturbed_species=[],
            construction="lagged-cycle")


def test_an_unknown_construction_refuses_by_naming_the_roster():
    with pytest.raises(background.BackgroundError, match="unknown ensemble"):
        background.plan_member_backgrounds(
            control_name="control", members=4, seed=1,
            perturbed_fields=WIND, perturbed_species=[],
            construction="bred-vector")


# ---------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------

def test_a_global_source_accepts_every_box():
    background.refuse_uncovered_area(
        "gfs", Area(lat_south=-40.0, lon_west=140.0,
                    lat_north=-30.0, lon_east=150.0))


@pytest.mark.parametrize("area", [
    Area(lat_south=58.0, lon_west=-155.0,          # far north, off-grid
         lat_north=62.0, lon_east=-145.0),
    Area(lat_south=18.0, lon_west=-160.0,          # mid-Pacific
         lat_north=22.0, lon_east=-155.0),
])
def test_hrrr_refuses_a_box_its_grid_does_not_carry_and_names_the_way_out(
        area):
    with pytest.raises(background.BackgroundError) as excinfo:
        background.refuse_uncovered_area("hrrr", area)
    message = str(excinfo.value)
    assert "beyond HRRR coverage" in message
    assert "--source gfs" in message


def test_the_coverage_envelope_is_the_grid_definition_not_a_held_box():
    """One definition, shared with ``woof fetch``'s own --area gate."""

    from woof.fetch import source_coverage_envelope
    from woof.ingest.hrrr_target import hrrr_coverage_envelope

    assert source_coverage_envelope("hrrr") == hrrr_coverage_envelope()
    assert source_coverage_envelope("gfs") is None


# ---------------------------------------------------------------------
# the receipt
# ---------------------------------------------------------------------

def test_the_receipt_attributes_the_background_per_member():
    init = datetime(2026, 8, 5, 4)
    cycle = background.plan_background_cycle(
        "hrrr", init=init, now=init + timedelta(hours=2),
        run_seconds=7200.0)
    plan = background.plan_member_backgrounds(
        control_name="control", members=2, seed=7,
        perturbed_fields=WIND, perturbed_species=[])
    receipt = background.background_receipt(
        source="hrrr", cycle=cycle, members=plan,
        prepared_content_sha256="a" * 64)

    assert receipt["schema"] == background.BACKGROUND_RECEIPT_SCHEMA
    assert receipt["source"] == "hrrr"
    assert receipt["cycle"]["forecast_start_hour"] \
        == cycle.forecast_start_hour
    assert receipt["ensemble"]["member_count"] == 2
    assert receipt["ensemble"]["construction"] \
        == background.PERTURBED_DETERMINISTIC
    assert [entry["trajectory"]
            for entry in receipt["ensemble"]["members"]] == \
        ["control", "0", "1"]
    assert receipt["ensemble"]["members"][0]["perturbation_seed"] is None
    assert receipt["prepared_content_sha256"] == "a" * 64


def test_the_receipt_is_json_serializable():
    import json

    plan = background.plan_member_backgrounds(
        control_name="control", members=1, seed=1,
        perturbed_fields=WIND, perturbed_species=[])
    receipt = background.background_receipt(
        source="gfs", cycle=None, members=plan)
    assert json.loads(json.dumps(receipt)) == receipt
