"""Every source's posting is a table row, checked at load (A136 L1).

A run that starts on a source's first hours and waits for the rest needs,
per source, the posting shape, how a lead is asked about, how long past
its scheduled time a lead may be, and the poll ceiling
(the 2026-09-30 streaming design, 2.1 and 2.2).  They
sit beside A134's ``publication_lag`` rows: a ``posting`` block on every
route, and a ``legacy_posting`` row for the transports outside the table.

Nothing here reaches a network.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import math
from pathlib import Path

import pytest

from woof import fetch, fetch_routes, source_posting
from woof.source_cycles import cycle_grid_for


#: The packaged loader, held before any test patches it, so each test
#: starts from the packaged table rather than the last test's mutation.
_PACKAGED_TABLE = fetch_routes._load_table


def _document():
    return json.loads(json.dumps(_PACKAGED_TABLE()))


def _rebuild(monkeypatch, document):
    monkeypatch.setattr(fetch_routes, "_load_table", lambda: document)
    fetch_routes._build_routes()
    fetch_routes._build_legacy_posting()


# --------------------------------------------------------------------------
# every source has a row
# --------------------------------------------------------------------------

def test_every_route_and_every_legacy_source_declares_posting():
    for source in fetch_routes.route_ids():
        assert fetch_routes.route_for(source).posting is not None, source
    for source in fetch_routes.LEGACY_ROUTE_SOURCES:
        assert fetch_routes.legacy_posting(source) is not None, source
    assert set(fetch_routes.posting_sources()) == (
        set(fetch_routes.route_ids()) | set(fetch_routes.LEGACY_ROUTE_SOURCES))
    for source in fetch_routes.posting_sources():
        row = fetch_routes.posting_for(source)
        assert row.shape in fetch_routes.POSTING_SHAPES
        assert row.why.strip()


def test_only_a_rolling_row_streams():
    for source in fetch_routes.posting_sources():
        row = fetch_routes.posting_for(source)
        assert row.streams is (row.shape == "rolling")


def test_every_waited_row_has_measured_timing():
    """The rows' `measured` strings are the posting watch's evidence."""

    for source in fetch_routes.posting_sources():
        if not fetch_routes.posting_for(source).waited:
            continue
        rules = cycle_grid_for(source, posting=True).delays
        assert rules, source
        for rule in rules:
            assert rule.measured.strip(), (source, rule)


@pytest.mark.parametrize("source,floor,minutes", [
    ("hrrr", 60, 90), ("hrrr-prs", 60, 90), ("gfs", 60, 120),
    ("gefs", 60, 120), ("rap", 60, 95), ("icon-global", 60, 60),
    ("icon-eu", 60, 60), ("icon-d2", 60, 60), ("gem-gdps", 60, 70),
    ("ecmwf-open-data", 60, 60), ("aifs", 60, 60),
    ("rrfs", 90, 90), ("gdas", 90, 115), ("aigfs", 90, 110),
    ("aigefs", 90, 105),
])
def test_late_budgets_outlast_the_row_s_own_late_spread(source, floor, minutes):
    """A budget is the row's measured late spread plus the margin, never
    under the design's adopted budget (A136 L1 follow-ups).

    The design adopted 60 min for rolling NCEP, DWD, MSC and ECMWF
    sources and 90 for rrfs, gdas, aigfs and aigefs (DESIGN A136, open
    question 5), counted from each lead's earliest posting seen.  The
    watch then saw GFS and GEFS leads post 87 min and RAP leads 63 min
    after that line (the late 30 Sep NCEP runs), so an as-posted run of
    such an ordinary late cycle would have stopped with exit 75.  Those
    figures are floors now: each budget is its spread plus
    LATE_BUDGET_MARGIN_MINUTES, rounded up to 5 min, or the floor.
    """

    row = fetch_routes.posting_for(source)
    spread = fetch_routes.late_spread_minutes(
        cycle_grid_for(source, posting=True).delays)
    margin = fetch_routes.LATE_BUDGET_MARGIN_MINUTES
    assert margin == 30
    assert row.late_after_minutes == max(
        floor, 5 * math.ceil((spread + margin) / 5 - 1e-9)) == minutes
    assert row.late_after_minutes >= spread + margin
    assert row.poll_seconds == 30


def test_the_ncep_budgets_cover_the_late_runs_the_watch_saw():
    """The measured spreads the L1 check named, read from the rows."""

    spreads = {source: fetch_routes.late_spread_minutes(
        cycle_grid_for(source, posting=True).delays)
        for source in ("gfs", "gefs", "rap")}
    assert spreads == pytest.approx({"gfs": 87, "gefs": 87, "rap": 63})
    for source, spread in spreads.items():
        assert fetch_routes.posting_for(source).late_after_minutes > spread


def test_the_shapes_are_the_design_s():
    shapes = {source: fetch_routes.posting_for(source).shape
              for source in fetch_routes.posting_sources()}
    # AIFS posts a whole run within about 3 min (the AWS rung's object
    # times, nine runs); data.ecmwf.int's per-lead Last-Modified, which
    # read as a rolling posting, is its dissemination schedule.
    assert shapes["ecmwf-open-data"] == shapes["gdas"] == shapes["aifs"] == "whole_cycle"
    assert shapes["aigfs"] == shapes["aigefs"] == "donor_gated"
    assert shapes["era5"] == "brokered"
    assert fetch_routes.posting_for("era5", provider="arco").shape == "archive"
    assert {source for source, shape in shapes.items() if shape == "rolling"} == {
        "hrrr", "hrrr-prs", "gfs", "gefs", "rap", "rrfs", "icon-global",
        "icon-eu", "icon-d2", "gem-gdps", "ecmwf-ens", "rrfs-ens",
        # The native hybrid products post lead by lead like their
        # pressure-level siblings.
        "hrrr-native", "rap-native"}


def test_a_ready_check_that_asks_for_an_index_names_one_on_every_file():
    for source in fetch_routes.route_ids():
        route = fetch_routes.route_for(source)
        if route.posting.ready_check == "objects_and_index":
            assert all(row.idx_sidecar for row in route.files), source
    assert fetch_routes.legacy_posting("hrrr").idx_sidecar == ".idx"


def test_the_load_time_probeable_rule_is_the_fetch_s():
    """Load check 4 reads table facts (a route's per-lead files, a legacy
    source's endpoint ladder) because the loader cannot import the fetch;
    it must answer as `woof.fetch.cycle_is_probeable` does."""

    for source in fetch_routes.posting_sources():
        row = fetch_routes.posting_for(source)
        assert row.waited == fetch.cycle_is_probeable(source), source


# --------------------------------------------------------------------------
# each load check refuses its bad row with its sentence
# --------------------------------------------------------------------------

def test_check_1_a_route_without_posting_is_refused(monkeypatch):
    document = _document()
    del document["routes"]["gefs"]["posting"]
    with pytest.raises(ValueError, match="route gefs declares no posting block"):
        _rebuild(monkeypatch, document)


def test_check_1_a_legacy_source_without_a_row_is_refused(monkeypatch):
    document = _document()
    del document["legacy_posting"]["gdas"]
    with pytest.raises(ValueError, match="legacy source gdas declares no "
                                         "legacy_posting row"):
        _rebuild(monkeypatch, document)


def test_a_legacy_row_for_a_source_with_its_own_route_is_refused(monkeypatch):
    document = _document()
    document["legacy_posting"]["gefs"] = document["legacy_posting"]["gfs"]
    with pytest.raises(ValueError, match="no legacy transport"):
        _rebuild(monkeypatch, document)


def test_check_2_a_whole_cycle_with_a_per_lead_slope_is_refused(monkeypatch):
    document = _document()
    document["routes"]["ecmwf-open-data"]["publication_lag"][0]["per_lead_hours"] = 0.01
    with pytest.raises(ValueError, match="promise leads before the cycle posts"):
        _rebuild(monkeypatch, document)
    document = _document()
    document["legacy_posting"]["gdas"]["publication_lag"][0]["per_lead_hours"] = 0.01
    with pytest.raises(ValueError, match="legacy source gdas posts whole cycles"):
        _rebuild(monkeypatch, document)


def test_check_3_a_donor_gate_without_a_same_cycle_donor_is_refused(monkeypatch):
    document = _document()
    for donor in document["routes"]["aigfs"]["donors"]:
        donor["cycle"] = "previous"
    with pytest.raises(ValueError, match="claim a gate the start probe never asks"):
        _rebuild(monkeypatch, document)
    document = _document()
    document["routes"]["gefs"]["posting"]["shape"] = "donor_gated"
    with pytest.raises(ValueError, match="route gefs is gated by a donor"):
        _rebuild(monkeypatch, document)


def test_check_4_a_waited_row_with_nothing_to_probe_is_refused(monkeypatch):
    document = _document()
    document["legacy_posting"]["era5"]["posting"] = dict(
        document["legacy_posting"]["gfs"]["posting"])
    document["legacy_posting"]["era5"]["publication_lag"] = (
        document["legacy_posting"]["gfs"]["publication_lag"])
    with pytest.raises(ValueError, match="wait out its budget and fail"):
        _rebuild(monkeypatch, document)


def test_check_5_an_index_check_on_a_file_without_an_index_is_refused(monkeypatch):
    document = _document()
    document["routes"]["gefs"]["files"][1]["idx_sidecar"] = None
    with pytest.raises(ValueError, match="pgrb2b declare no idx_sidecar"):
        _rebuild(monkeypatch, document)
    document = _document()
    document["routes"]["icon-eu"]["posting"]["ready_check"] = "objects_and_index"
    with pytest.raises(ValueError, match="index URL the table never built"):
        _rebuild(monkeypatch, document)
    document = _document()
    del document["legacy_posting"]["hrrr"]["idx_sidecar"]
    with pytest.raises(ValueError, match="index URL the table never built"):
        _rebuild(monkeypatch, document)


@pytest.mark.parametrize("key,value,sentence", [
    ("late_after_minutes", 0, "fails every run whose lead is a second"),
    ("late_after_minutes", -5, "fails every run whose lead is a second"),
    ("poll_seconds", 1, "trips the NOMADS rate limiter"),
])
def test_check_6_a_zero_budget_or_a_fast_poll_is_refused(
        monkeypatch, key, value, sentence):
    document = _document()
    document["routes"]["rap"]["posting"][key] = value
    with pytest.raises(ValueError, match=sentence):
        _rebuild(monkeypatch, document)


@pytest.mark.parametrize("where,source", [
    ("routes", "rap"), ("routes", "gefs"), ("legacy_posting", "gfs"),
    ("legacy_posting", "hrrr"),
])
def test_check_7_a_budget_under_the_row_s_late_spread_is_refused(
        monkeypatch, where, source):
    """A budget shorter than the spread its own rows measured stops an
    as-posted run of an ordinary late cycle, one no later than a cycle
    already seen, with exit 75.  A budget equal to the spread loads."""

    spread = fetch_routes.late_spread_minutes(
        cycle_grid_for(source, posting=True).delays)
    document = _document()
    document[where][source]["posting"]["late_after_minutes"] = spread - 1
    with pytest.raises(ValueError, match=(
            "shorter than its own publication_lag rows' late spread")) as refusal:
        _rebuild(monkeypatch, document)
    assert "exit 75" in str(refusal.value)
    assert "ordinary late cycle" in str(refusal.value)
    assert f"at least {spread:g} min" in str(refusal.value)
    document = _document()
    document[where][source]["posting"]["late_after_minutes"] = spread
    _rebuild(monkeypatch, document)


def test_a_waited_row_without_a_budget_or_a_check_is_refused(monkeypatch):
    document = _document()
    del document["routes"]["rap"]["posting"]["late_after_minutes"]
    with pytest.raises(ValueError, match="would hold its run forever"):
        _rebuild(monkeypatch, document)
    document = _document()
    del document["routes"]["rap"]["posting"]["ready_check"]
    with pytest.raises(ValueError, match="tell a posted lead from a missing one"):
        _rebuild(monkeypatch, document)
    document = _document()
    document["routes"]["rap"]["posting"]["ready_check"] = "listing"
    with pytest.raises(ValueError, match="named for a later round"):
        _rebuild(monkeypatch, document)


def test_an_unknown_shape_or_a_missing_why_is_refused(monkeypatch):
    document = _document()
    document["routes"]["rap"]["posting"]["shape"] = "streaming"
    with pytest.raises(ValueError, match="posting shape 'streaming'"):
        _rebuild(monkeypatch, document)
    document = _document()
    document["routes"]["rap"]["posting"]["why"] = " "
    with pytest.raises(ValueError, match="does not say why"):
        _rebuild(monkeypatch, document)


def test_a_waited_legacy_row_without_timing_is_refused(monkeypatch):
    document = _document()
    del document["legacy_posting"]["hrrr"]["publication_lag"]
    with pytest.raises(ValueError, match="no scheduled time to wait from"):
        _rebuild(monkeypatch, document)


def test_the_packaged_table_loads_clean(monkeypatch):
    _rebuild(monkeypatch, _document())


# --------------------------------------------------------------------------
# expected_at is A134's arithmetic
# --------------------------------------------------------------------------

def _rule_cases():
    for source in fetch_routes.posting_sources():
        grid = cycle_grid_for(source, posting=True)
        if grid is None:
            continue
        for rule in grid.delays:
            hours = rule.cycle_hours or grid.hours
            for hour in hours:
                cycle = datetime(2026, 9, 29, hour)
                if next(r for r in grid.delays
                        if r.matches(hour, rule.from_lead)) is not rule:
                    continue
                yield source, cycle, rule
                break


_RULE_CASES = list(_rule_cases())


@pytest.mark.parametrize(
    "source,cycle,rule", _RULE_CASES,
    ids=[f"{source}-{cycle:%H}z-from{rule.from_lead}"
         for source, cycle, rule in _RULE_CASES])
def test_expected_at_is_the_rule_s_arithmetic(source, cycle, rule):
    for lead in (rule.from_lead, rule.from_lead + 6, rule.from_lead + 48):
        if next(r for r in cycle_grid_for(source, posting=True).delays
                if r.matches(cycle.hour, lead)) is not rule:
            continue
        hours = rule.hours + rule.per_lead_hours * lead
        expected = cycle + timedelta(hours=hours)
        assert source_posting.expected_at(source, cycle, lead) == expected
        assert source_posting.expected_at(source, cycle, lead) == (
            cycle + timedelta(hours=cycle_grid_for(source, posting=True).delay(cycle, lead)))
        budget = fetch_routes.posting_for(source).late_after_minutes
        assert source_posting.late_at(source, cycle, lead) == (
            expected + timedelta(minutes=budget))
        assert source_posting.publication_rule(source, cycle, lead) is rule


def test_a_legacy_grid_reads_its_table_rules():
    for source in ("gfs", "gdas", "hrrr"):
        rules = fetch_routes.legacy_publication_lag(source)
        assert rules
        grid = cycle_grid_for(source, posting=True)
        assert grid.delays == rules
        assert "legacy_posting" in grid.basis
        # The latest walk and the date page keep the probe's answer from
        # now for a legacy source until the as-posted latest replaces it.
        walk = cycle_grid_for(source)
        assert walk.delays == () and walk.delay_hours == 0
        assert walk.usual_delay == grid.usual_delay
    assert fetch_routes.legacy_publication_lag("era5") == ()
    # A route's grid is the same either way.
    assert cycle_grid_for("gefs", posting=True) == cycle_grid_for("gefs")


def test_a_whole_cycle_schedules_every_lead_at_once():
    cycle = datetime(2026, 9, 29, 12)
    times = {source_posting.expected_at("ecmwf-open-data", cycle, lead)
             for lead in (0, 3, 144, 360)}
    assert len(times) == 1
    times = {source_posting.expected_at("gdas", cycle, lead) for lead in range(10)}
    assert len(times) == 1


def test_a_config_budget_overrides_the_row_and_is_checked():
    cycle = datetime(2026, 9, 29, 12)
    due = source_posting.expected_at("gefs", cycle, 24)
    assert source_posting.late_at(
        "gefs", cycle, 24, late_after_minutes_override=15) == due + timedelta(minutes=15)
    for bad in (0, -1, float("nan"), True, "60"):
        with pytest.raises(ValueError, match="positive number of minutes"):
            source_posting.late_at("gefs", cycle, 24,
                                   late_after_minutes_override=bad)


def test_a_source_no_run_waits_on_has_no_late_time():
    cycle = datetime(2026, 9, 20, 6)
    assert source_posting.late_at("era5", cycle, 0) is None
    assert source_posting.late_at("era5", cycle, 0, provider="arco") is None


# --------------------------------------------------------------------------
# the schedule document
# --------------------------------------------------------------------------

def test_the_schedule_carries_every_key_the_go_relay_reads():
    """`woof.chain_events` relays `posting/schedule.json` into
    `posting_schedule` and `lead_posted` events; the keys it reads are the
    design's 3.5 names.  The start needs (and `expected_ready_at`) depend
    on the window's donors and delayed nests and are the fetch loop's."""

    cycle = datetime(2026, 9, 29, 12)
    document = source_posting.schedule("gefs", cycle, range(0, 49, 3),
                                       member="c00")
    assert document["schema"] == "gpuwm.posting-schedule.v1"
    for key in ("source", "member", "cycle", "as_posted", "shape", "streams",
                "why", "late_after_minutes", "expected_final_at",
                "table_sha256"):
        assert key in document, key
    assert document["cycle"] == "2026-09-29T12"
    assert document["table_sha256"] == fetch_routes.packaged_route_table_sha256()
    rows = document["leads"]
    assert [row["lead"] for row in rows] == list(range(0, 49, 3))
    for row in rows:
        for key in ("lead", "valid_time", "expected_at", "late_at",
                    "first_seen_at", "fetched_at", "endpoint", "state"):
            assert key in row, key
        assert row["state"] == "scheduled"
    assert rows[0]["valid_time"] == "2026-09-29T12:00:00Z"
    assert rows[-1]["valid_time"] == "2026-10-01T12:00:00Z"
    assert document["expected_final_at"] == rows[-1]["expected_at"]
    due = source_posting.expected_at("gefs", cycle, 48)
    assert rows[-1]["expected_at"] == due.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_the_schedule_module_names_no_source():
    """The arbitrary acceptance test: a model added as a row has a
    schedule with no edit to the module that computes it."""

    text = Path(source_posting.__file__).read_text(encoding="utf-8")
    for source in fetch_routes.posting_sources():
        assert f'"{source}"' not in text and f"'{source}'" not in text, source


# --------------------------------------------------------------------------
# the registry door carries the block
# --------------------------------------------------------------------------

@pytest.mark.parametrize("source", ["gefs", "hrrr", "ecmwf-open-data", "era5"])
def test_sources_json_row_carries_the_posting_block(capsys, source):
    from woof.sources_cli import sources_main

    assert sources_main(argparse.Namespace(source=source, json=True)) == 0
    document = json.loads(capsys.readouterr().out)
    (row,) = document["sources"]
    block = row["fetch"]["posting"]
    expected = fetch_routes.posting_for(source)
    assert block["shape"] == expected.shape
    assert block["streams"] == expected.streams
    assert block["late_after_minutes"] == expected.late_after_minutes
    assert block["poll_seconds"] == expected.poll_seconds
    assert block["why"] == expected.why
    assert block["table_sha256"] == fetch_routes.packaged_route_table_sha256()
    rules = cycle_grid_for(source, posting=True).delays
    assert [rule["hours"] for rule in block["publication_lag"]] == [
        rule.hours for rule in rules]
    assert all(rule["measured"] for rule in block["publication_lag"])
    if source == "era5":
        assert block["providers"]["arco"]["shape"] == "archive"


def test_the_listing_shows_each_source_s_posting_shape(capsys):
    from woof.sources_cli import sources_main

    assert sources_main(argparse.Namespace(source=None, json=False)) == 0
    text = capsys.readouterr().out
    assert "posting" in text.splitlines()[1]
    assert "whole_cycle" in text and "donor_gated" in text


# --------------------------------------------------------------------------
# HRRR's DA posting curve is its row
# --------------------------------------------------------------------------

def test_the_hrrr_da_wait_reads_its_row():
    from woof.da import background

    assert not hasattr(background, "HRRR_BASE_LAG_S")
    assert not hasattr(background, "HRRR_PER_LEAD_LAG_S")
    grid = cycle_grid_for("hrrr", posting=True)
    source = background.BACKGROUND_SOURCES["hrrr"]
    for lead in (0, 1, 6, 18, 19, 48):
        # The expected line (A136 L1 follow-ups: the latest posting seen
        # put every plan behind the cycle that was out), over the cycle
        # hours whose horizon reaches the lead.
        cycles = [datetime(2001, 1, 1, hour) for hour in grid.hours
                  if grid.horizon(datetime(2001, 1, 1, hour)) >= lead]
        expected = max(grid.delay(cycle, lead) for cycle in cycles) * 3600.0
        assert source.lag_seconds(lead) == pytest.approx(expected)
        assert source.lag_seconds(lead) == pytest.approx(max(
            (source_posting.expected_at("hrrr", cycle, lead) - cycle).total_seconds()
            for cycle in cycles))
    assert source.lag_seconds(18) > source.lag_seconds(0)


def test_hrrr_s_usual_delay_is_past_its_row_s_latest_posting():
    """The page takes a HRRR start this old as published while no check
    has answered; its adapter comment points at the row instead of
    quoting numbers, and this holds the value past the row's latest
    posting seen of each cycle's last lead (a synoptic f048, an
    off-synoptic f018)."""

    grid = cycle_grid_for("hrrr", posting=True)
    cycles = [datetime(2001, 1, 1, hour) for hour in grid.hours]
    latest = max(grid.delay(cycle, grid.horizon(cycle), settled=True)
                 for cycle in cycles)
    assert grid.usual_delay >= latest
    assert cycle_grid_for("hrrr").usual_delay == grid.usual_delay


@pytest.mark.parametrize("source,cycle,lead,posted_minutes", [
    # NOMADS listing times, 2026-10-23 (fix/go-posted-cycle-default).
    ("rap-native", datetime(2026, 10, 23, 5), 0, 47),
    ("rap-native", datetime(2026, 10, 23, 5), 18, 53),
    ("rap-native", datetime(2026, 10, 23, 0), 0, 82),
    ("rap-native", datetime(2026, 10, 22, 21), 51, 66),
    ("hrrr-native", datetime(2026, 10, 23, 5), 0, 50),
    ("hrrr-native", datetime(2026, 10, 23, 5), 18, 84),
    ("hrrr-native", datetime(2026, 10, 23, 6), 18, 72),
    ("hrrr-native", datetime(2026, 10, 23, 6), 48, 105),
])
def test_a_native_lead_is_asked_by_the_time_it_posted(
        source, cycle, lead, posted_minutes):
    """The breakage: the native rows were an unmeasured 1.5 h estimate, so a
    default `woof go` on a posted RAP cycle (f000 out at +47 min) did not
    ask until +85 min and sat ~40 min.  A lead is asked from its row's time
    minus ASK_AHEAD, so that line must not be after a posting seen."""

    from woof import source_readiness

    ask_from = (source_posting.expected_at(source, cycle, lead)
                - source_readiness.ASK_AHEAD)
    assert ask_from <= cycle + timedelta(minutes=posted_minutes)
