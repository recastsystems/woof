"""A calendar must not offer a source period its own metadata rules out."""
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from woof import source_adapters
from woof.source_availability import (
    ArchiveWindow, availability, parse_cycle, resolve_latest, validate_cycle,
)
from woof.source_cycles import CycleGrid


NOW = datetime(2026, 9, 6, 15, 25)


@pytest.mark.parametrize("transport", [None, "aws", "ecmwf"])
@pytest.mark.parametrize("cycle,reason", [
    ("2024-01-15T00", "pressure"),
    ("2024-02-15T00", "pressure"),
    ("2024-03-01T00", "pressure"),
    ("2024-03-10T00", "soil"),
    ("2024-03-18T06", "soil"),
    ("2024-03-18T12", None),
    ("2024-03-20T00", None),
])
def test_publication_eras_own_calendar_and_fetch_refusals(transport, cycle, reason):
    from woof import fetch_routes
    from woof.source_availability import verdict

    document = availability("ifs", 3, now=NOW, transport=transport)
    assert document["earliest"] == "2024-03-18T12"
    if reason is None:
        assert validate_cycle(document, cycle)[0] == cycle
        assert verdict(document, cycle, now=NOW)["state"] == "yes"
        fetch_routes.resolve_request("ifs", cycle=parse_cycle(cycle), hours=3, host=transport)
    else:
        with pytest.raises(ValueError, match=reason) as fetch_error:
            fetch_routes.resolve_request("ifs", cycle=parse_cycle(cycle), hours=3, host=transport)
        with pytest.raises(ValueError) as calendar_error:
            validate_cycle(document, cycle)
        assert str(calendar_error.value) == str(fetch_error.value)
        answer = verdict(document, cycle, now=NOW)
        assert answer["state"] == "no"
        assert answer["why"] == str(fetch_error.value)


def test_current_transport_layout_bound_is_not_the_scientific_record_start():
    from woof import fetch_endpoints

    # GDAS reads only the current /atmos/ S3 layout.  GFS also reads the
    # dated archive layouts (32dea003f45, development branch), so its
    # bound is the oldest declared layout, still a layout bound and not the
    # model's scientific record start.
    oldest = min(row["record_start"] for row in fetch_endpoints.archive_rows("gfs"))
    assert oldest == "2005-02-15T00"
    for source, earliest, era in (("gfs", oldest, "pre-atmos"),
                                  ("gfs-0p25", oldest, "pre-atmos"),
                                  ("gdas", "2021-03-22T12", "/atmos/")):
        document = availability(source, 6, now=NOW)
        assert document["earliest"] == earliest
        assert document["cycle_hours"] == [0, 6, 12, 18]
        assert document["latest_candidate"] == "2026-09-06T12"
        archive = next(row for row in document["transports"] if row["transport"] == "s3")
        assert era in archive["note"]
        assert archive["documentation"]
    # A reader who deliberately selected the rolling endpoint is not handed
    # its archive's range or an automatic probe against a different host.
    rolling = availability("gfs", 6, now=NOW, transport="nomads")
    assert rolling["earliest"] is None
    assert rolling["transports"][0]["retention_hours"] == 240
    assert rolling["latest_supported"] is False


def test_utc_hour_choices_follow_the_requested_forecast_horizon():
    assert availability("hrrr", 18, now=NOW)["cycle_hours"] == list(range(24))
    extended = availability("hrrr", 24, now=NOW)
    assert extended["cycle_hours"] == [0, 6, 12, 18]
    assert extended["latest_candidate"] == "2026-09-06T12"
    assert extended["earliest"] == "2014-07-30T18"
    pressure = availability("hrrr-prs", 6, now=NOW)
    assert pressure["earliest"] == "2014-07-30T18"
    assert any(row["transport"] == "aws" and row["record_start"] for row in pressure["transports"])
    assert not availability("hrrr", 49, now=NOW)["latest_supported"]
    assert not availability("gfs", 385, now=NOW)["latest_supported"]


@pytest.mark.parametrize("source,cycles", [("era5", list(range(24))), ("era5-ml", list(range(24)))])
def test_analysis_latest_keeps_the_whole_period_behind_publication_lag(source, cycles):
    document = availability(source, 24, now=NOW)
    assert document["earliest"] == "1940-01-01T00"
    assert document["cycle_hours"] == cycles
    assert document["latest_label"] == "Latest expected"
    end = parse_cycle(document["latest_candidate"]) + timedelta(hours=24)
    assert end <= NOW - timedelta(hours=120)
    assert any("whole 24-hour" in note for note in document["notes"])
    assert "preliminary ERA5T" in document["transports"][0]["note"]


def test_explicit_hourly_analysis_start_matches_the_actual_native_request():
    """An off-synoptic start is retrieved as the exact times it asks for.

    This used to assert one request carrying
    ``["00:00", "01:00", "14:00" ... "23:00"]``: the CDS cross product of
    every date with every clock time, which for a window starting at 14Z
    retrieves 00Z and 01Z of the FIRST day and 14Z through 23Z of the
    second, neither of which was asked for.  The request is now written
    one day at a time, so the assertion here is the one that survives the
    change: the union over the requests is exactly the series.
    """

    from woof import fetch
    cycle = fetch.parse_cycle("2013-05-31T14", "era5")
    document = availability("era5", 11, now=NOW)
    assert validate_cycle(document, "2013-05-31T14:00:00Z")[0] == "2013-05-31T14"
    request = fetch.era5_request_template(
        cycle=cycle, hours=11, cadence=1, area=fetch.parse_area("30,-100,40,-90"))
    times = fetch._era5_times(cycle, 11, 1)
    assert times[0] == cycle and times[-1] == datetime(2013, 6, 1, 1)
    assert len(times) == 12
    for dataset in ("reanalysis-era5-pressure-levels",
                    "reanalysis-era5-single-levels"):
        retrieved = tuple(
            datetime.strptime(day + "T" + clock, "%Y-%m-%dT%H:%M")
            for row in request["requests"] if row["dataset"] == dataset
            for day in row["request"]["date"]
            for clock in row["request"]["time"])
        assert retrieved == times


def test_undeclared_retention_never_becomes_an_unlimited_archive(monkeypatch):
    # A row whose archive start is not written down: the real row with its
    # measured window removed, so the rule stays tested after every real
    # row declares one.
    undeclared = replace(source_adapters.get_source_adapter("aifs"), archive_windows=())
    monkeypatch.setattr(source_adapters, "get_source_adapter", lambda _source: undeclared)
    document = availability("aifs", 6, now=NOW)
    assert document["earliest"] is None
    assert all(row["record_start"] is None for row in document["transports"])
    assert any("not fully declared" in note for note in document["notes"])


def test_metadata_lookup_does_not_contact_provider(monkeypatch):
    from woof import fetch
    monkeypatch.setattr(fetch, "_head_ok", lambda _url: pytest.fail("metadata must stay offline"))
    assert availability("gfs", 6, now=NOW)["probeable"]


def test_latest_uses_the_real_resolver_and_checks_the_required_forecast_hour():
    checked = []

    def provider(url):
        checked.append(url)
        # A deterministic transport seam: the new 12Z cycle is incomplete.
        # The actual resolver must walk its registered candidates and URLs.
        return "gfs.t06z" in url and ".f006" in url

    # The whole-cycle rule asks the final forecast hour; as-posted is the
    # default (A136 L2) and is pinned below.
    result = resolve_latest("gfs", 6, now=NOW, probe=provider, as_posted=False)
    assert result["selected_cycle"] == "2026-09-06T06"
    assert result["resolution"]["basis"] == "provider_object_probe"
    assert len(checked) >= 2
    assert all("f006" in url for url in checked)
    assert result["resolution"]["objects"][-1]["available"]


def test_latest_as_posted_asks_the_start_needs_and_takes_the_newer_start():
    checked = []

    def provider(url):
        checked.append(url)
        # The 12Z cycle has its first leads out, not its last: as posted,
        # it is the start (the fetch takes f006 when it posts).
        return (("gfs.t06z" in url)
                or ("gfs.t12z" in url and (url.endswith(".f000")
                                           or url.endswith(".f003")
                                           or url.endswith(".f001"))))

    # Half an hour on, when the table says 12Z f003 is due.
    result = resolve_latest("gfs", 6, now=NOW + timedelta(minutes=30),
                            probe=provider)
    assert result["selected_cycle"] == "2026-09-06T12"
    assert not any("t12z" in url and "f006" in url for url in checked)


@pytest.mark.parametrize("hours", [1, 6, 24, 120, 240])
def test_the_analysis_window_back_off_is_applied_once(hours):
    """The date the calendar publishes and the date it resolves are one date.

    An analysis window is covered by successive analyses, so its newest
    start is the newest published analysis minus the window.  Both the
    calendar and the acquisition resolver used to subtract it, and the
    second subtraction was silent: a 240-hour era5 request selected a
    cycle ten days before the SAME document's ``latest_candidate``, with
    ``validate_cycle`` accepting it and the terminal's Latest button
    running it.  The two doors now call one function
    (``woof.fetch.analysis_window_reference``), so the only way to
    subtract twice is to write the subtraction twice again.
    """

    result = resolve_latest("era5", hours, now=NOW,
                            probe=lambda _url: pytest.fail("CDS exposes no object HEAD probe"))
    assert result["selected_cycle"] == result["latest_candidate"]
    end = parse_cycle(result["selected_cycle"]) + timedelta(hours=hours)
    assert end <= NOW - timedelta(hours=120)


def test_analysis_estimate_does_not_masquerade_as_a_provider_probe():
    result = resolve_latest("era5", 24, now=NOW,
                            probe=lambda _url: pytest.fail("CDS exposes no object HEAD probe"))
    assert result["selected_cycle"] == "2026-08-31T12"
    assert result["resolution"]["basis"] == "estimated_publication_schedule"
    assert result["resolution"]["objects"] == []


def test_new_source_and_closed_archive_are_table_work(monkeypatch):
    from woof import fetch_endpoints
    source = replace(source_adapters.get_source_adapter("era5"),
        source_id="new-analysis", aliases=(),
        cadence_mapping="uniform-declared-analysis-series-v1",
        cycle_grid=CycleGrid(hours=(3, 15), delay_hours=2,
                             record_end=datetime(2020, 2, 29, 15)),
        archive_windows=(ArchiveWindow("archive", "1900-01-01T03",
                         "Declared test record", ("https://example.invalid/provider",)),))
    monkeypatch.setattr(source_adapters, "get_source_adapter", lambda _source: source)
    monkeypatch.setattr(fetch_endpoints, "has_ladder", lambda _source: False)
    document = availability("new-analysis", 24, now=NOW)
    assert document["source_id"] == "new-analysis"
    assert document["cycle_hours"] == [3, 15]
    assert document["earliest"] == "1900-01-01T03"
    assert document["latest_candidate"] == "2020-02-28T15"
    result = resolve_latest("new-analysis", 24, now=NOW)
    assert result["selected_cycle"] == "2020-02-28T15"


@pytest.mark.parametrize("value", ["2026-09-05", "2026-09-05T06:30", "2026-09-05T06+02:00", "2026-02-29T06"])
def test_cycle_parser_does_not_silently_rewrite_the_intended_time(value):
    with pytest.raises(ValueError):
        parse_cycle(value)


def test_manual_date_has_the_same_known_bounds_and_horizon_rules():
    document = availability("hrrr", 24, now=NOW)
    for value in ("2014-07-29T00", "2026-09-05T07", "2026-09-07T00"):
        with pytest.raises(ValueError):
            validate_cycle(document, value)
    assert validate_cycle(document, "2026-09-05T06:00:00Z")[0] == "2026-09-05T06"


def test_saved_fetch_span_and_lead_are_not_shortened_to_the_experiment_window(tmp_path):
    from woof.source_availability import _context
    config = tmp_path / "forecast.toml"
    config.write_text('[experiment]\nrun_seconds=21600\n[fetch]\nsource="hrrr"\nhours=24\nforecast_start_hour=6\n', encoding="utf-8")
    source, hours, _ = _context(None, None, config, None)
    assert source == "hrrr" and hours == 30
    assert availability(source, hours, now=NOW)["cycle_hours"] == [0, 6, 12, 18]
    # Editing run duration does not silently reduce an existing fetch request.
    assert _context(None, 3, config, None)[1] == 30


# A GFS cycle six hours old at 06:04Z: the schedule allows it, but its
# objects are not on any host yet.
LIVE = datetime(2026, 9, 26, 6, 4)


def _nothing_from_06z(url):
    # Every host's grammar for the 06Z run: the NCEP tree (.t06z.) and the
    # dated archive layouts the GFS archive lane (32dea003f45) appended to
    # the ladder (GDEX gfs.0p25.YYYYMMDDHH, NCEI gfs_*_YYYYMMDD_HH00_).
    return not any(token in url for token in (".t06z.", "2026092606", "_0600_"))


def _stand_in(probe):
    from woof.source_availability import publication_refusal

    return lambda document, cycle, now=None: publication_refusal(
        document, cycle, now=now, probe=probe)


def test_a_start_the_fetch_would_refuse_is_not_offered_as_available():
    from woof.fetch import require_published_cycle
    from woof.source_availability import verdict

    document = availability("gfs", 6, now=LIVE)
    # The publication schedule alone calls it available; that was the defect.
    assert verdict(document, "2026-09-26T06", now=LIVE)["state"] == "yes"
    answer = verdict(document, "2026-09-26T06", now=LIVE,
                     confirm=_stand_in(_nothing_from_06z))
    assert answer["state"] == "no"
    # One source of truth: the sentence is the fetch's own refusal.
    with pytest.raises(RuntimeError) as refused:
        require_published_cycle("gfs", datetime(2026, 9, 26, 6), 6, now=LIVE,
                                probe=_nothing_from_06z)
    assert answer["why"] == str(refused.value)
    assert "2026-09-26T00Z" in answer["why"]
    assert verdict(document, "2026-09-26T00", now=LIVE,
                   confirm=_stand_in(_nothing_from_06z))["state"] == "yes"


def test_the_probe_asks_for_the_runs_last_hour():
    from woof.source_availability import verdict

    asked = []

    def probe(url):
        asked.append(url)
        return True

    verdict(availability("gfs", 48, now=LIVE), "2026-09-26T00", now=LIVE,
            confirm=_stand_in(probe))
    assert asked and all(".f048" in url for url in asked)


def test_an_old_start_is_judged_by_the_archive_bounds_without_a_probe():
    from woof.source_availability import verdict

    def probe(url):
        raise AssertionError(f"an old start must not be probed: {url}")

    document = availability("gfs", 6, now=LIVE)
    assert verdict(document, "2026-09-20T00", now=LIVE,
                   confirm=_stand_in(probe))["state"] == "yes"


def test_latest_is_the_newest_start_the_fetch_accepts():
    from woof.source_availability import confirmed_latest

    document = availability("gfs", 6, now=LIVE)
    assert document["latest_candidate"] == "2026-09-26T06"
    assert confirmed_latest(document, now=LIVE, probe=_nothing_from_06z) == "2026-09-26T00"
    assert confirmed_latest(document, now=LIVE, probe=lambda url: True) == "2026-09-26T06"
