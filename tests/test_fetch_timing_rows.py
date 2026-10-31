"""The publisher timing facts a fetch runs on are table rows, and they hold.

Three defects measured on 2026-09-29/30 against the live publishers
(Downloads/WOOF-ALL-SOURCES-2026-09-29):

* ``--cycle latest`` started its walk at one publication lag per
  producer, too long for all of them: IFS 12 h against a measured
  7 h 34 min (00/12Z) and 6 h 27 min (06/18Z), AIFS 8 h against f000 at
  5 h 45 min, GEFS 8 h against 3 h 45 min, RAP 3 h against about an
  hour.  ERA5's keyless ARCO copy was resolved on the CDS's five days
  although it trails by six to seven.
* ECMWF's AWS mirror answers about half of all requests with
  ``503 SlowDown`` and the fetch gave up after 5 tries in 30 s.
* A GEFS window past f240 or an IFS window past f144 needed a hand
  typed ``--cadence 6``; the default refused it.

Nothing here reaches a network.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import io
from urllib.error import HTTPError

import pytest

from woof import domain_wizard, fetch, fetch_endpoints, fetch_routes
from woof.source_cycles import CycleGrid, route_cycle_grid


# --------------------------------------------------------------------------
# (1) publication delay per cycle hour, from the route table
# --------------------------------------------------------------------------

def test_ifs_publication_lag_is_a_row_per_cycle_hour():
    route = fetch_routes.route_for("ecmwf-open-data")
    assert route.publication_lag_hours(0) == route.publication_lag_hours(12) == 7.5
    assert route.publication_lag_hours(6) == route.publication_lag_hours(18) == 6.4
    # The run posts whole: no lead is later than f000.
    assert route.publication_lag_hours(12, 360) == 7.5


@pytest.mark.parametrize("source,lead,posted_hours", [
    # Earliest posting measured, cycle to Last-Modified / directory time.
    # AIFS: the AWS rung's own object times (S3 LastModified), which put a
    # whole run on the server at + 5 h 27 min to 5 h 40 min over nine runs
    # (A136 L1 posting watch, 2026-09-30).  The pins these replace (f000
    # + 5 h 45 min to f360 + 7 h 34 min) were data.ecmwf.int's
    # Last-Modified, its dissemination schedule: it served the 30 Sep 00Z
    # f360 at 07:04Z stamped 07:34Z.
    ("aifs", 0, 5 + 27 / 60), ("aifs", 360, 5 + 27 / 60),
    ("gefs", 0, 225 / 60), ("gefs", 384, 362 / 60),
    ("rap", 0, 48 / 60), ("rap", 21, 55 / 60), ("rap", 51, 68 / 60),
    ("rrfs", 0, 75 / 60), ("rrfs", 84, 185 / 60),
    ("icon-eu", 0, 2 + 44 / 60), ("icon-eu", 120, 3 + 39 / 60),
    ("gem-gdps", 0, 4 + 4 / 60), ("gem-gdps", 240, 5 + 3 / 60),
])
def test_declared_lags_never_start_the_walk_after_a_measured_posting(
        source, lead, posted_hours):
    """A declared lag later than a measured posting makes `latest` skip a
    cycle that is already out, which is the defect; earlier only costs a
    probe that answers no."""

    route = fetch_routes.route_for(source)
    for hour in route.cycle_hours:
        declared = route.publication_lag_hours(hour, lead)
        assert declared <= posted_hours + 1e-9, (source, hour, lead, declared)
        # And not so early that the walk asks a cycle hours before it exists.
        assert declared >= posted_hours - 0.75, (source, hour, lead, declared)


def test_every_route_declares_a_lag_for_every_cycle_hour_it_runs():
    for source in fetch_routes.route_ids():
        route = fetch_routes.route_for(source)
        for hour in route.cycle_hours:
            assert route.publication_lag_hours(hour) >= 0


def test_a_route_row_missing_a_cycle_hour_is_refused_at_load():
    with pytest.raises(ValueError, match="no publication lag for its 06, 18Z"):
        fetch_routes._lag_rules("x", {
            "cycle_hours": [0, 6, 12, 18],
            "publication_lag": [{"cycle_hours": [0, 12], "hours": 7.5}]})


@pytest.mark.parametrize("source,now,hours,newest", [
    # 23:56Z: IFS 12Z had been posted since 19:34Z; the 12 h lag gave 06Z.
    ("ecmwf-open-data", datetime(2026, 9, 29, 23, 56), 6, datetime(2026, 9, 29, 12)),
    # 00:06Z: AIFS 18Z f000 posted at 23:45Z; the 8 h lag gave 12Z.
    ("aifs", datetime(2026, 9, 30, 0, 6), 6, datetime(2026, 9, 29, 18)),
    # 00:15Z: GEFS 18Z complete on NOMADS by 00:04Z; the 8 h lag gave 12Z.
    ("gefs", datetime(2026, 9, 30, 0, 15), 6, datetime(2026, 9, 29, 18)),
    # 23:59Z: RAP 21Z and 22Z were complete; the 3 h lag gave 20Z.  The
    # walk now starts at 23Z, whose f006 can first be out at 23:50Z, and
    # the probe decides between it and 22Z.
    ("rap", datetime(2026, 9, 29, 23, 59), 6, datetime(2026, 9, 29, 23)),
])
def test_latest_starts_at_the_cycle_the_publisher_already_posted(
        source, now, hours, newest):
    grid = route_cycle_grid(source)
    assert grid.newest(now, hours) == newest
    walk = grid.candidates(now, hours)
    assert walk[0] == newest and walk[1] < newest


def test_a_long_window_waits_for_its_last_lead():
    # GEFS 18Z f000 is out at 21:45Z but f384 not until about 00:02Z.
    grid = route_cycle_grid("gefs")
    now = datetime(2026, 9, 29, 23, 0)
    assert grid.newest(now, 6) == datetime(2026, 9, 29, 18)
    assert grid.newest(now, 384) == datetime(2026, 9, 29, 12)


def test_a_grid_with_one_delay_answers_as_before():
    grid = CycleGrid(hours=(0, 6, 12, 18), delay_hours=5.0)
    for minute in range(0, 24 * 60, 17):
        now = datetime(2026, 9, 29, minute // 60, minute % 60)
        assert grid.newest(now) == grid.snap(now - timedelta(hours=5.0))
        assert grid.newest(now, 240) == grid.newest(now)


def test_latest_resolves_the_newest_published_ifs_cycle_through_the_probe():
    now = datetime(2026, 9, 29, 23, 56)
    asked: list[str] = []

    def probe(url: str) -> bool:
        asked.append(url)
        # The IFS objects, and the same cycle's GFS f000 sea-ice donor
        # (posted about 3.5 h after the cycle, before every IFS lead).
        return any(stamp in url for stamp in ("20260929120000", "20260929060000",
                                              "gfs.20260929/12/", "gfs.20260929/06/"))

    cycle = fetch.resolve_latest_cycle("ecmwf-open-data", 6, now=now, probe=probe)
    assert cycle == datetime(2026, 9, 29, 12)
    # The 18Z run is not due until 00:24Z, so it is not asked about at all.
    assert not any("20260929180000" in url for url in asked)


def test_latest_walks_back_when_the_newest_candidate_is_not_up_yet():
    # 19:31Z: the table allows 12Z (+7.5 h), the host has not posted it.
    now = datetime(2026, 9, 29, 19, 31)

    def probe(url: str) -> bool:
        # GFS 12Z (the sea-ice donor) is up at 19:31Z; IFS 12Z is not.
        return any(stamp in url for stamp in ("20260929060000", "gfs.20260929/12/", "gfs.20260929/06/"))

    assert fetch.resolve_latest_cycle(
        "ecmwf-open-data", 6, now=now, probe=probe) == datetime(2026, 9, 29, 6)


# ERA5: the keyless ARCO copy's own lag --------------------------------------

def test_era5_arco_latest_resolves_on_the_store_s_own_end(monkeypatch):
    from woof import source_availability

    monkeypatch.setattr(source_availability, "published_stop",
                        lambda window, **_: datetime(2026, 9, 23, 23))
    now = datetime(2026, 9, 30, 1)
    arco = fetch.resolve_latest_cycle("era5", 12, now=now, provider="arco")
    assert arco == datetime(2026, 9, 23, 6)
    assert arco + timedelta(hours=12) <= datetime(2026, 9, 23, 23)
    # The CDS's own five days are unchanged.
    assert fetch.resolve_latest_cycle("era5", 12, now=now) == datetime(2026, 9, 24, 12)


def test_era5_arco_latest_falls_back_to_its_declared_lag_row(monkeypatch):
    from woof import source_availability, source_adapters

    monkeypatch.setattr(source_availability, "published_stop", lambda window, **_: None)
    window = next(row for row in source_adapters.get_source_adapter("era5").archive_windows
                  if row.transport == "arco")
    assert window.publication_lag_hours == 168.0
    now = datetime(2026, 9, 30, 1)
    cycle = fetch.resolve_latest_cycle("era5", 12, now=now, provider="arco")
    assert cycle == datetime(2026, 9, 22, 12)
    grid, basis = fetch.provider_cycle_grid("era5", "arco")
    assert "168" in basis and grid.delay_hours == 168.0


def test_the_era5_latest_request_carries_its_provider():
    from woof.cli import parse_fetch_arguments

    args = parse_fetch_arguments(["--source", "era5", "--era5-provider", "arco",
                                  "--cycle", "latest", "--hours", "12",
                                  "--point", "35,-97", "--out", "x"])
    assert fetch.latest_cycle_request(args) == ("era5", 12, {"provider": "arco"})
    plain = parse_fetch_arguments(["--source", "era5", "--cycle", "latest",
                                   "--hours", "12", "--point", "35,-97", "--out", "x"])
    assert fetch.latest_cycle_request(plain) == ("era5", 12, {})


# --------------------------------------------------------------------------
# (2) a throttling mirror is waited out over the budget its row declares
# --------------------------------------------------------------------------

_MIRROR = "ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
_SLOWDOWN = (b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Error><Code>SlowDown"
             b"</Code><Message>Please reduce your request rate.</Message></Error>")


def _mirror():
    return fetch_endpoints.Endpoint(
        name="aws", base=f"https://{_MIRROR}", retention_hours=None,
        why="the mirror")


def _answer(code: int, body: bytes) -> HTTPError:
    return HTTPError(f"https://{_MIRROR}/x", code, "Slow Down", {}, io.BytesIO(body))


def test_the_ecmwf_mirror_declares_a_throttle_row_and_nothing_else_does():
    policy = fetch_endpoints.throttle_policy(_MIRROR)
    assert policy is not None
    assert policy.statuses == (503,) and policy.codes == ("SlowDown",)
    assert policy.budget_s >= 300 > fetch_endpoints.TRANSIENT_WAIT_LIMIT_S
    assert fetch_endpoints.throttle_policy("noaa-gefs-pds.s3.amazonaws.com") is None
    assert fetch_endpoints.throttle_policy("data.ecmwf.int") is None


def test_waits_double_with_jitter_up_to_the_row_s_ceiling():
    policy = fetch_endpoints.throttle_policy(_MIRROR)
    assert policy.wait(1, 0.0) == policy.first_wait_s / 2
    assert policy.wait(1, 1.0) == policy.first_wait_s
    assert policy.wait(3, 0.5) == pytest.approx(0.75 * 4 * policy.first_wait_s)
    assert policy.wait(40, 1.0) == policy.max_wait_s


def test_slowdown_is_waited_out_past_the_five_rounds_and_says_so():
    answers = iter([_answer(503, _SLOWDOWN) for _ in range(9)])
    calls, lines, pauses = [], [], []

    def transfer(endpoint):
        calls.append(endpoint.name)
        error = next(answers, None)
        if error is not None:
            raise error
        return {"bytes": 1}

    endpoint, entry = fetch_endpoints.ask_along_ladder(
        (_mirror(),), transfer, label="fetch ecmwf-open-data",
        name="20260920000000-0h-oper-fc.grib2", progress=lines.append,
        pause=pauses.append, jitter=lambda: 0.5)
    assert entry == {"bytes": 1} and len(calls) == 10
    assert len(pauses) == 9 > fetch_endpoints.TRANSIENT_ATTEMPTS
    # Doubling from 1.5 s (half of 2 plus half the jitter), capped at 45 s.
    assert pauses[:6] == [1.5, 3.0, 6.0, 12.0, 24.0, 45.0]
    waiting = [line for line in lines if "is throttling this client; waiting" in line]
    assert len(waiting) == 9
    assert _MIRROR in waiting[0] and "20260920000000-0h-oper-fc.grib2" in waiting[0]
    assert "600 s the route table allows this host" in waiting[-1]
    assert any("HTTP 503 SlowDown -- the host is throttling" in line for line in lines)


def test_a_throttle_that_outlasts_the_budget_is_refused_naming_it():
    lines, pauses = [], []

    def transfer(endpoint):
        raise _answer(503, _SLOWDOWN)

    with pytest.raises(fetch_endpoints.TransferRefusal) as error:
        fetch_endpoints.ask_along_ladder(
            (_mirror(),), transfer, label="fetch aifs", name="f.grib2",
            progress=lines.append, pause=pauses.append, jitter=lambda: 1.0)
    assert sum(pauses) == pytest.approx(600.0)
    assert "was still throttling when the 600 s the route table allows it were spent" \
        in str(error.value)
    assert "HTTP 503 SlowDown" in str(error.value)


def test_a_503_that_is_not_slowdown_keeps_the_ordinary_five_rounds():
    pauses = []

    def transfer(endpoint):
        raise _answer(503, b"<Error><Code>ServiceUnavailable</Code></Error>")

    with pytest.raises(fetch_endpoints.TransferRefusal):
        fetch_endpoints.ask_along_ladder(
            (_mirror(),), transfer, label="fetch aifs", name="f.grib2",
            progress=lambda _line: None, pause=pauses.append, jitter=lambda: 0.5)
    assert pauses == [2.0, 4.0, 8.0, 16.0]


# --------------------------------------------------------------------------
# (3) cadence by lead: the ladder rows, taken by the window itself
# --------------------------------------------------------------------------

def test_an_ifs_window_past_f144_takes_six_hourly_by_itself():
    plan = fetch_routes.resolve_request(
        "ecmwf-open-data", cycle=datetime(2026, 9, 29, 12), hours=240)
    assert plan.leads == tuple(range(0, 241, 6))


def test_a_gefs_window_past_f240_takes_six_hourly_by_itself():
    plan = fetch_routes.resolve_request(
        "gefs", cycle=datetime(2026, 9, 29, 12), hours=384)
    assert plan.leads == tuple(range(0, 385, 6))
    later = fetch_routes.resolve_request(
        "gefs", cycle=datetime(2026, 9, 29, 12), hours=6, start_hour=246)
    assert later.leads == (246, 252)


def test_a_window_the_default_serves_keeps_the_default():
    plan = fetch_routes.resolve_request(
        "ecmwf-open-data", cycle=datetime(2026, 9, 29, 12), hours=144)
    assert plan.leads == tuple(range(0, 145, 3))
    icon = fetch_routes.resolve_request(
        "icon-eu", cycle=datetime(2026, 9, 29, 12), hours=78)
    assert icon.leads == tuple(range(0, 79))


def test_an_icon_eu_window_past_f078_takes_three_hourly():
    plan = fetch_routes.resolve_request(
        "icon-eu", cycle=datetime(2026, 9, 29, 18), hours=120)
    assert plan.leads == tuple(range(0, 121, 3))


def test_a_named_cadence_is_honoured_and_refused_naming_the_one_that_serves():
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_request(
            "ecmwf-open-data", cycle=datetime(2026, 9, 29, 12), hours=240, cadence=3)
    message = str(error.value)
    assert "does not publish f147 at --cadence 3" in message
    # The remedy names the spacing that serves as both doors spell it.
    assert ("cadence 6 (--cadence 6, or cadence = 6 in [fetch]) is published "
            "over this whole window") in message
    plan = fetch_routes.resolve_request(
        "gefs", cycle=datetime(2026, 9, 29, 12), hours=24, cadence=6)
    assert plan.leads == (0, 6, 12, 18, 24)


def test_a_cycle_that_does_not_reach_the_window_still_refuses_by_horizon():
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_request(
            "ecmwf-open-data", cycle=datetime(2026, 9, 29, 6), hours=240)
    assert "forecasts through f144" in str(error.value)


def test_a_start_off_the_coarse_ladder_is_refused_naming_it():
    with pytest.raises(ValueError, match="does not publish f147"):
        fetch_routes.resolve_request(
            "ecmwf-open-data", cycle=datetime(2026, 9, 29, 12), hours=6,
            start_hour=147)


@pytest.mark.parametrize("source,hours,start,expected", [
    ("ecmwf-open-data", 240, 0, (6, 240)),
    ("ecmwf-open-data", 6, 150, (6, 6)),
    ("ecmwf-open-data", 144, 0, (3, 144)),
    ("ecmwf-open-data", 5, 0, (3, 6)),
    ("gefs", 300, 0, (6, 300)),
    ("gefs", 240, 0, (3, 240)),
    ("icon-eu", 120, 0, (3, 120)),
])
def test_the_domain_door_writes_the_spacing_the_window_s_ladder_publishes(
        source, hours, start, expected):
    assert domain_wizard.fetch_window(source, hours, start) == expected


def test_the_domain_door_honours_a_named_cadence():
    assert domain_wizard.fetch_window("ecmwf-open-data", 240, 0, 3) == (3, 240)


def test_latest_for_a_240_hour_ifs_window_picks_a_long_cycle():
    now = datetime(2026, 9, 30, 1, 0)

    def probe(url: str) -> bool:
        return True

    cycle = fetch.resolve_latest_cycle("ecmwf-open-data", 240, now=now, probe=probe)
    assert cycle.hour in (0, 12)


# --------------------------------------------------------------------------
# Second round: one spacing for the latest walk, the doors that name a
# cycle, and the latest posting seen for answers no probe checks
# --------------------------------------------------------------------------

@pytest.mark.parametrize("now,hours,expected", [
    # 06Z: ICON-EU's 03Z short run is newest, hourly only to f030.
    (datetime(2026, 9, 30, 6, 0), 36, datetime(2026, 9, 30, 0)),
    (datetime(2026, 9, 30, 6, 0), 42, datetime(2026, 9, 30, 0)),
    # 12Z: the 09Z short run is newest.
    (datetime(2026, 9, 30, 12, 0), 36, datetime(2026, 9, 30, 6)),
    (datetime(2026, 9, 30, 12, 0), 42, datetime(2026, 9, 30, 6)),
])
def test_latest_with_no_cadence_takes_one_spacing_for_the_whole_walk(now, hours, expected):
    """A short run that thins to 6-hourly past f030 is not the latest
    hourly window, although a probe finds everything it publishes."""

    cycle = fetch.resolve_latest_cycle("icon-eu", hours, now=now, probe=lambda url: True)
    assert cycle == expected
    plan = fetch_routes.resolve_request("icon-eu", cycle=cycle, hours=hours)
    assert plan.leads == tuple(range(0, hours + 1))


def test_a_saved_setup_start_takes_the_cycle_latest_hands_it():
    """companion_setups.start_setup resolves `latest` with no cadence and
    then checks the window at the setup's own hourly cadence."""

    route = fetch_routes.route_for("icon-eu")
    for now in (datetime(2026, 9, 30, 6, 0), datetime(2026, 9, 30, 12, 0)):
        for hours in (36, 42):
            cycle = fetch.resolve_latest_cycle("icon-eu", hours, now=now,
                                               probe=lambda url: True)
            assert fetch_routes.resolve_leads(route, cycle, hours, cadence=1) \
                == tuple(range(0, hours + 1))


def test_a_named_short_cycle_still_takes_its_own_ladder():
    plan = fetch_routes.resolve_request(
        "icon-eu", cycle=datetime(2026, 9, 30, 3), hours=36)
    assert plan.leads == (0, 6, 12, 18, 24, 30, 36)
    # The domain door asks the same cycle when one is named, and every
    # cycle hour when the cycle is `latest`.
    assert domain_wizard._fetch_cadence_h(
        "icon-eu", 0, 36, cycle=datetime(2026, 9, 30, 3)) == 6
    assert domain_wizard._fetch_cadence_h("icon-eu", 0, 36) == 1


def _cyclone_hints(source, moment, hours, point, start_hour=0):
    from woof.cyclone_sources import fetch_hints

    projection = domain_wizard._projection_entries(*point, "auto")
    return fetch_hints(source=source, moment=moment, hours=hours,
                       projection=projection, dims=(80, 80), dx_m=12000.0,
                       start_hour=start_hour)


@pytest.mark.parametrize("source,moment,hours,point,cadence,fetched", [
    ("ecmwf-open-data", datetime(2026, 9, 29, 0), 240, (15.0, -60.0), 6, 240),
    ("ecmwf-open-data", datetime(2026, 9, 29, 0), 241, (15.0, -60.0), 6, 246),
    ("ecmwf-open-data", datetime(2026, 9, 29, 0), 24, (15.0, -60.0), 3, 24),
    ("gefs", datetime(2026, 9, 29, 0), 300, (15.0, -60.0), 6, 300),
    ("icon-eu", datetime(2026, 9, 30, 3), 36, (50.0, 10.0), 6, 36),
])
def test_the_cyclone_door_takes_the_spacing_its_cycle_publishes(
        source, moment, hours, point, cadence, fetched):
    """It passed the source's usual spacing and refused these windows
    while `woof fetch` and `woof domain` took them."""

    hints = _cyclone_hints(source, moment, hours, point)
    assert (hints["cadence"], hints["hours"]) == (cadence, fetched)


def test_the_cyclone_door_prices_and_namelists_the_spacing_it_fetches():
    from woof.cyclone_setup import _forcing_interval

    assert _forcing_interval("ecmwf-open-data", cycle="2026092900",
                             start_hour=0, hours=240) == 6 * 3600.0
    assert _forcing_interval("ecmwf-open-data", cycle="2026092900",
                             start_hour=0, hours=24) == 3 * 3600.0
    assert _forcing_interval("ecmwf-open-data") == 3 * 3600.0


def test_a_catalog_case_opened_past_the_coarsening_lead_fetches_six_hourly(tmp_path):
    import json
    from pathlib import Path
    import tomllib
    from datetime import timezone
    from woof import case_catalog

    document = json.loads((Path(case_catalog.__file__).parent / "data" / "case-catalog"
                           / "example.json").read_text(encoding="utf-8"))
    case = next(row for row in document["cases"] if row["id"] == "synthetic-profile-example")
    option = case["source_options"][0]
    option.update(source="ecmwf-open-data", cycle_utc="2026-09-01T00:00:00Z")
    for tier in case["tiers"].values():
        tier["run_hours"] = 240
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    out = tmp_path / "opened.toml"
    case_catalog.create_case(path, "synthetic-profile-example", out=out, tier="lower",
                             now=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
                             geometry_only=True)
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert (raw["fetch"]["cadence"], raw["fetch"]["hours"]) == (6, 240)
    assert "interval_seconds = 21600" in out.with_suffix(".namelist.wps").read_text()


# The latest posting seen and the earliest, per cycle hour and lead:
# Last-Modified on each route's ladder head, read from a development machine at
# 2026-09-30 05:26Z (and the earlier listings the rows name).
_POSTINGS = [
    # (source, cycle hour, lead, earliest seen, latest seen), in hours
    ("ecmwf-open-data", 0, 360, 7.567, 7.567), ("ecmwf-open-data", 18, 144, 6.450, 6.450),
    # AIFS: the AWS rung's object times per cycle hour, 28 Sep 00Z to
    # 30 Sep 00Z (A136 L1; see the note on the earliest-line pins above).
    ("aifs", 0, 360, 5.505, 5.610), ("aifs", 6, 0, 5.448, 5.539),
    ("aifs", 12, 360, 5.540, 5.667), ("aifs", 18, 90, 5.498, 5.543),
    ("gefs", 0, 0, 3.753, 4.736), ("gefs", 6, 240, 5.187, 5.202),
    ("gefs", 12, 384, 6.034, 6.082), ("gefs", 0, 390, 24.09, 24.12),
    ("gefs", 0, 600, 25.04, 25.05), ("gefs", 0, 840, 26.12, 26.14),
    ("rap", 0, 0, 0.805, 1.399), ("rap", 0, 21, 0.926, 1.524), ("rap", 21, 51, 1.134, 1.200),
    ("rrfs", 18, 0, 1.229, 1.950), ("rrfs", 18, 84, 2.888, 3.500),
    ("icon-eu", 0, 0, 2.694, 2.741), ("icon-eu", 0, 78, 3.419, 3.503),
    ("icon-eu", 0, 120, 3.552, 3.659), ("icon-eu", 3, 48, 2.914, 2.970),
    ("gem-gdps", 0, 0, 3.838, 4.075), ("gem-gdps", 0, 120, 4.418, 4.792),
    ("gem-gdps", 12, 240, 4.832, 5.053),
]


@pytest.mark.parametrize("source,hour,lead,earliest,latest", _POSTINGS)
def test_each_row_lies_between_the_postings_it_was_measured_on(
        source, hour, lead, earliest, latest):
    """The walk starts at the earliest posting seen (later skips a cycle
    that is out) and an answer no probe checks waits for the latest one
    (earlier claims a lead that is not out)."""

    route = fetch_routes.route_for(source)
    assert route.publication_lag_hours(hour, lead) <= earliest + 1e-9
    assert route.publication_lag_hours(hour, lead, settled=True) >= latest - 1e-9


def test_the_gefs_extension_is_a_day_late_and_latest_knows_it():
    now = datetime(2026, 9, 29, 12, 0)
    # The 29 Sep 00Z f840 lands on 30 Sep at about 02:07Z.
    cycle = fetch.resolve_latest_cycle("gefs", 840, now=now, probe=lambda url: True)
    assert cycle == datetime(2026, 9, 28, 0)
    grid = route_cycle_grid("gefs")
    assert grid.delay(datetime(2026, 9, 29, 0), 384) < 7
    assert grid.delay(datetime(2026, 9, 29, 6), 384) < 7


def test_a_rule_an_earlier_row_shadows_is_refused_at_load():
    with pytest.raises(ValueError, match="shadowed by an earlier row"):
        fetch_routes._lag_rules("x", {
            "cycle_hours": [0, 6],
            "publication_lag": [{"cycle_hours": None, "hours": 3},
                                {"cycle_hours": [0], "from_lead": 390, "hours": 22}]})


def test_the_whole_run_delay_is_the_latest_posting_of_the_common_ladder():
    # GEFS's f384 of every cycle, not its f000 (3.7 h) and not the 00Z
    # extension a day later; IFS's whole runs by 7 h 34 min, AIFS's by
    # 5 h 40 min on the AWS rung (A136 L1: the 7 h 34 min it was pinned
    # at was data.ecmwf.int's dissemination schedule, not the posting).
    assert 6.082 <= route_cycle_grid("gefs").usual_delay < 8
    assert 5.667 <= route_cycle_grid("aifs").usual_delay < 6
    assert 7.567 <= route_cycle_grid("ecmwf-open-data").usual_delay < 8


def test_answers_no_probe_checks_wait_for_the_latest_posting():
    from woof.source_availability import availability
    from woof.da.background import _BackgroundSources

    # 21:45Z: GEFS 18Z f006 was first seen at 21:44Z, but a run has
    # posted up to an hour later, so the page's unchecked start is 12Z.
    document = availability("gefs", 6, now=datetime(2026, 9, 29, 21, 45))
    assert document["latest_candidate"] == "2026-09-29T18"
    assert document["due_start"] == "2026-09-29T12"
    # A DA background plan is not such an answer: it takes the expected
    # line and its run's fetch waits for a late cycle until the row calls
    # it late (A136 L1 follow-ups; it read the latest line until then and
    # rode a background about an hour older than the one that was out).
    lag = _BackgroundSources()["gefs"].publication_lag_seconds(384)
    assert lag == pytest.approx(max(
        route_cycle_grid("gefs").delay(datetime(2001, 1, 1, hour), 384)
        for hour in (0, 6, 12, 18)) * 3600)
    assert lag < route_cycle_grid("gefs").delay(
        datetime(2001, 1, 1, 0), 384, settled=True) * 3600
