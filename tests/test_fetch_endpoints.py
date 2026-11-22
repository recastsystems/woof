"""The endpoint ladder: which rung is asked, and which one moves bytes.

Every NCEP source publishes on two hosts that serve byte-identical
objects under byte-identical keys.  They differ in three ways, and all
three are table facts here: the operational server publishes hours
earlier, it keeps a bounded window, and it paces bulk transfers where
the archive does not.  So the router resolves the concrete cycle, takes
the endpoints whose retention covers that cycle's age -- and then, for
the transfer itself, prefers the throughput rung for any object the
mirror has already caught up with.

No network anywhere in this file -- the selection function, the
availability probe and the transfer's ladder walk are all driven
directly.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from urllib.error import HTTPError, URLError

import pytest

from woof import fetch, fetch_endpoints, fetch_pool, fetch_routes
from tools import download_gfs_native_subset as gfs_transport


#: The NCEP family the ladder covers, table routes and legacy alike.
NCEP_SOURCES = ("gfs", "gdas", "gefs", "hrrr", "hrrr-prs", "rap", "rrfs")

#: Sources outside the NCEP family, whose ladders are whatever single
#: publisher door they have always had.
OTHER_SOURCES = ("icon-eu", "gem-gdps", "ecmwf-open-data", "aifs")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, minute=0,
                                              second=0, microsecond=0)


# --------------------------------------------------------------------------
# Ladder selection by cycle age
# --------------------------------------------------------------------------

def test_a_latest_cycle_takes_the_operational_server_on_every_ncep_source():
    """The directive, as a table fact: NOMADS is the head of the ladder."""

    now = _utc_now()
    for source in NCEP_SOURCES:
        serving = fetch_endpoints.serving_ladder(
            source, cycle=now - timedelta(hours=2), now=now)
        assert serving, source
        assert serving[0].name == "nomads", source
        assert "nomads.ncep.noaa.gov" == serving[0].host, source


def test_an_old_cycle_skips_the_operational_server_it_cannot_serve():
    """A cycle past retention goes straight to the archive.

    The concrete breakage this prevents: a doomed request to a host
    that provably dropped the cycle days ago, paid for at 2.5 s of
    governed pacing per object before the 404 that was certain.
    """

    now = _utc_now()
    stale = now - timedelta(days=30)
    for source in ("gfs", "gdas", "gefs", "hrrr", "hrrr-prs", "rap", "rrfs"):
        serving = fetch_endpoints.serving_ladder(source, cycle=stale, now=now)
        assert serving[0].name in ("aws", "s3"), source
        assert all(endpoint.archive for endpoint in serving), source
        assert serving[0].retention_hours is None, source


def test_retention_is_an_optimisation_not_a_bar():
    """A source with no archive still tries its one endpoint.

    aigfs is NOMADS-only by ruling (the S3 objects under identical key
    names are a different product), so past its window there is nothing
    to skip to.  Dropping the only endpoint would turn a maybe into a
    refusal that never asked.
    """

    now = _utc_now()
    serving = fetch_endpoints.serving_ladder(
        "aigfs", cycle=now - timedelta(days=30), now=now)
    assert [endpoint.name for endpoint in serving] == ["nomads"]


def test_every_ncep_ladder_declares_a_measured_retention_and_an_archive():
    for source in NCEP_SOURCES:
        ladder = fetch_endpoints.ladder(source)
        assert len(ladder) >= 2, source
        assert ladder[0].retention_hours and ladder[0].retention_hours > 0
        assert ladder[0].why.strip(), source
        assert ladder[1].retention_hours is None, source


def test_a_pinned_transport_is_a_decision_and_does_not_fall_through():
    now = _utc_now()
    serving = fetch_endpoints.serving_ladder(
        "rap", cycle=now - timedelta(hours=2), now=now, pinned="aws")
    assert [endpoint.name for endpoint in serving] == ["aws"]


# --------------------------------------------------------------------------
# Non-NCEP regression: untouched
# --------------------------------------------------------------------------

def test_non_ncep_sources_head_their_ladder_with_the_publisher_door():
    """A fresh cycle asks the publisher's door; past its window the mirror, where there is one.

    The ECMWF door keeps 72 h and the AWS mirror behind it keeps every
    cycle, so a 400-day-old ECMWF cycle goes straight to the mirror; Google's
    mirror is the archive rung after it on every ECMWF route (it keeps every
    cycle back to 2025-01), for when the AWS mirror answers 503 SlowDown.
    The DWD and MSC doors have no archive behind them, so an old cycle
    still asks the one door (retention is an optimisation, not a bar).
    """

    now = _utc_now()
    fresh = {"icon-eu": ["dwd"], "gem-gdps": ["msc"],
             "ecmwf-open-data": ["ecmwf", "aws", "google"], "aifs": ["ecmwf", "aws", "google"]}
    old = {"icon-eu": ["dwd"], "gem-gdps": ["msc"],
           "ecmwf-open-data": ["aws", "google"], "aifs": ["aws", "google"]}
    for source in OTHER_SOURCES:
        for age, expected in ((timedelta(hours=2), fresh),
                              (timedelta(days=400), old)):
            serving = fetch_endpoints.serving_ladder(
                source, cycle=now - age, now=now)
            assert [rung.name for rung in serving] == expected[source], (source, age)


def test_a_non_ncep_route_plans_byte_identical_urls():
    plan = fetch_routes.resolve_request(
        "icon-eu", cycle=datetime(2026, 8, 17, 0), hours=1)
    assert plan.host.name == "dwd"
    assert all(obj.url.startswith(
        "https://opendata.dwd.de/weather/nwp/icon-eu/grib/")
        for obj in plan.objects)


# --------------------------------------------------------------------------
# Host caps come from the table
# --------------------------------------------------------------------------

def test_the_pool_reads_its_host_caps_from_the_table():
    assert (fetch_pool.HOST_FILE_WORKER_CAPS
            == dict(fetch_endpoints.host_caps()))
    assert "nomads.ncep.noaa.gov" in fetch_pool.HOST_FILE_WORKER_CAPS


def test_the_operational_server_gets_a_polite_cap_that_says_why():
    cap = fetch_pool.host_worker_cap("nomads.ncep.noaa.gov", 16)
    assert cap == 2
    assert 1 <= cap <= 3
    why = fetch_endpoints.host_cap_why("nomads.ncep.noaa.gov")
    assert "governor" in why and "2.5" in why


# --------------------------------------------------------------------------
# Fault classification and fall-through
# --------------------------------------------------------------------------

def _http_error(code, headers=None):
    return HTTPError("https://nomads.ncep.noaa.gov/x", code, "no",
                     headers or {}, None)


def test_the_faults_that_move_to_the_next_endpoint_are_named():
    assert "503" in fetch_endpoints.fault_reason(_http_error(503))
    assert "403" in fetch_endpoints.fault_reason(_http_error(403))
    assert "404" in fetch_endpoints.fault_reason(_http_error(404))
    refused = fetch_endpoints.fault_reason(
        URLError(ConnectionRefusedError(61, "refused")))
    assert "refused" in refused.lower()
    assert fetch_endpoints.fault_reason(KeyboardInterrupt()) is None


def test_a_retry_after_the_host_could_not_wait_out_is_carried_forward():
    reason = fetch_endpoints.fault_reason(
        _http_error(503, {"Retry-After": "900"}))
    assert "Retry-After" in reason and "900" in reason


# --------------------------------------------------------------------------
# The transfer walks the ladder
# --------------------------------------------------------------------------

def _grib(payload: bytes) -> bytes:
    return b"GRIB" + payload + b"7777"


def _ladder_downloader(*, failing: str, error, seen: list):
    """A transport that fails against one host and serves from the other."""

    def download(url, dest, *, magic, opener=None):
        seen.append(url)
        if failing in url:
            raise error
        dest.parent.mkdir(parents=True, exist_ok=True)
        body = _grib(url.encode())
        dest.write_bytes(body)
        return {"name": dest.name, "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(), "url": url}

    return download


def _fresh_plan(source="rap", hours=0):
    now = _utc_now()
    return fetch_routes.resolve_request(
        source, cycle=now - timedelta(hours=2), hours=hours, now=now)


#: The mirror has not caught up with this cycle yet -- publication lag,
#: the exact case the operational server exists for.  Every test below
#: that is about FALL-THROUGH rather than about host selection says so
#: explicitly, because "which host moves the bytes" is now a probed
#: fact and not a constant.
def _mirror_lagging(url):
    return False


def _mirror_caught_up(url):
    return True


def test_a_refused_connection_falls_through_too(tmp_path):
    plan = _fresh_plan()
    payload = fetch_routes.run_plan(
        plan, out=tmp_path, progress=lambda *_: None, probe=_mirror_lagging,
        downloader=_ladder_downloader(
            failing="nomads.ncep.noaa.gov",
            error=URLError(ConnectionRefusedError(61, "refused")), seen=[]))
    assert payload["files"][0]["endpoint"] == "aws"


def test_the_whole_ladder_failing_names_every_endpoint_and_why(tmp_path):
    plan = _fresh_plan()

    def always_fails(url, dest, *, magic, opener=None):
        if "nomads" in url:
            raise _http_error(503, {"Retry-After": "900"})
        raise HTTPError(url, 403, "denied", {}, None)

    with pytest.raises(ValueError) as error:
        fetch_routes.run_plan(plan, out=tmp_path, progress=lambda *_: None,
                              probe=_mirror_lagging, downloader=always_fails)
    message = str(error.value)
    assert "nomads.ncep.noaa.gov" in message
    assert "noaa-rap-pds" in message
    assert "503" in message and "403" in message
    assert "Retry-After" in message


def test_a_pinned_transport_refuses_without_trying_the_other_host(
        tmp_path, monkeypatch):
    from types import SimpleNamespace
    waits: list[float] = []
    monkeypatch.setattr(fetch_routes, "time",
                        SimpleNamespace(sleep=waits.append), raising=False)
    now = _utc_now()
    plan = fetch_routes.resolve_request(
        "rap", cycle=now - timedelta(hours=2), hours=0, now=now,
        host="nomads")
    seen: list[str] = []

    def always_fails(url, dest, *, magic, opener=None):
        seen.append(url)
        raise _http_error(503)

    def explode(url):
        raise AssertionError("a pinned transport is a decision, not a probe")

    with pytest.raises(ValueError):
        fetch_routes.run_plan(plan, out=tmp_path, progress=lambda *_: None,
                              probe=explode, downloader=always_fails)
    assert all("nomads" in url for url in seen)
    # The pinned host alone, asked on the shared schedule.
    assert len(seen) == fetch_endpoints.TRANSIENT_ATTEMPTS
    assert waits == [2, 4, 8, 16]


# --------------------------------------------------------------------------
# Availability-aware transfer host: the mirror when it HAS the object
# --------------------------------------------------------------------------

def test_the_table_declares_which_rung_moves_bulk_bytes_fastest():
    """Throughput preference is a rung attribute, not a code branch.

    The concrete breakage this prevents: at peak hours the operational
    server paced whole-file transfers at about 3 MB/s, so a 3.4 GB
    request took ~20 min where the archive had served the same volume
    in ~3 min.  Which rung is quicker is a property of the HOST, so it
    is declared beside the host.
    """

    for source in NCEP_SOURCES:
        rungs = fetch_endpoints.ladder(source)
        operational, archive = rungs[0], rungs[1]
        assert operational.name == "nomads", source
        assert archive.transfer_rank < operational.transfer_rank, source
        order = fetch_endpoints.transfer_order(rungs)
        assert order[0] is archive, source


def test_a_source_outside_the_ncep_family_keeps_its_table_order():
    """No rank declared means the ladder order IS the transfer order."""

    for source in OTHER_SOURCES:
        rungs = fetch_endpoints.ladder(source)
        assert fetch_endpoints.transfer_order(rungs) == rungs, source
        serving = fetch_endpoints.serving_ladder(
            source, cycle=_utc_now() - timedelta(hours=2))
        assert fetch_endpoints.transfer_probes(serving) == (), source


def test_the_mirror_is_probed_per_object_not_per_request(tmp_path):
    """One lead published on the mirror, the next not yet.

    The concrete breakage this prevents: a whole-request decision taken
    from the first object sends the entire remaining window to the slow
    host the moment one lead has not been mirrored yet.
    """

    plan = _fresh_plan(hours=1)
    assert len(plan.objects) == 2
    first = plan.objects[0].key

    probed = []
    seen = []

    def probe(url):
        probed.append(url)
        return url.endswith(first)

    payload = fetch_routes.run_plan(
        plan, out=tmp_path, progress=lambda *_: None,
        probe=probe,
        downloader=_ladder_downloader(failing="__never__", error=None,
                                      seen=seen))
    served = {entry["relpath"]: entry["endpoint"]
              for entry in payload["files"]}
    assert served[plan.objects[0].relpath] == "aws"
    assert served[plan.objects[1].relpath] == "nomads"
    assert len(probed) == len(plan.objects)
    assert all("noaa-rap-pds" in url for url in probed)
    assert any("noaa-rap-pds" in url for url in seen)
    assert any("nomads.ncep.noaa.gov" in url for url in seen)


def test_a_mirror_probe_that_errors_does_not_spend_the_transfer_ladder(
        tmp_path):
    """A 503 on the PROBE costs the transfer nothing.

    The concrete breakage this prevents: a throttled probe counted as
    an endpoint attempt would leave the object one failure away from a
    whole-ladder refusal, so a cheap HEAD could turn a fetch that had
    two good hosts into one that had none.
    """

    plan = _fresh_plan()

    def probe(url):
        raise _http_error(503, {"Retry-After": "900"})

    seen: list[str] = []
    payload = fetch_routes.run_plan(
        plan, out=tmp_path, progress=lambda *_: None, probe=probe,
        downloader=_ladder_downloader(
            failing="nomads.ncep.noaa.gov", error=_http_error(503, {"Retry-After": "900"}),
            seen=seen))

    # The operational server was still the transfer's head, and the
    # mirror was still behind it to catch the fall-through.
    assert any("nomads.ncep.noaa.gov" in url for url in seen)
    assert payload["files"][0]["endpoint"] == "aws"
    assert payload["endpoints"]["considered"] == ["nomads", "aws"]


def test_an_archive_era_cycle_is_never_probed_at_all(tmp_path):
    """One rung means nothing to choose between."""

    now = _utc_now()
    plan = fetch_routes.resolve_request(
        "rap", cycle=now - timedelta(days=30), hours=0, now=now)
    assert [rung.name for rung in plan.ladder] == ["aws"]

    def explode(url):
        raise AssertionError("a one-rung ladder has nothing to probe")

    payload = fetch_routes.run_plan(
        plan, out=tmp_path, progress=lambda *_: None, probe=explode,
        downloader=_ladder_downloader(failing="__never__", error=None,
                                      seen=[]))
    assert payload["files"][0]["endpoint"] == "aws"


def test_the_freshness_note_prints_only_when_the_mirror_lags(tmp_path):
    mirrored: list[str] = []
    mirrored_payload = fetch_routes.run_plan(
        _fresh_plan(), out=tmp_path / "a", progress=mirrored.append,
        probe=_mirror_caught_up,
        downloader=_ladder_downloader(failing="__never__", error=None,
                                      seen=[]))
    assert any("mirrored" in line and "throughput" in line
               for line in mirrored), mirrored
    assert not any("publishes before the mirrors" in line
                   for line in mirrored), mirrored

    declared = mirrored_payload["endpoints"]
    assert declared["considered"] == ["nomads", "aws"]
    assert declared["served"] == ["aws"]
    assert declared["transfer_preference"] == ["aws", "nomads"]
    assert declared["probe"] == {"endpoint": "aws", "objects": 1, "available": 1}
    assert all(entry["endpoint"] == "aws" for entry in mirrored_payload["files"])

    lagging: list[str] = []
    lagging_payload = fetch_routes.run_plan(
        _fresh_plan(), out=tmp_path / "b", progress=lagging.append,
        probe=_mirror_lagging,
        downloader=_ladder_downloader(failing="__never__", error=None,
                                      seen=[]))
    assert any("publishes before the mirrors" in line
               for line in lagging), lagging
    assert not any("mirrored" in line for line in lagging), lagging
    assert lagging_payload["endpoints"]["considered"] == ["nomads", "aws"]
    assert lagging_payload["endpoints"]["served"] == ["nomads"]
    assert all(entry["endpoint"] == "nomads" and
               entry["url"].startswith("https://nomads.ncep.noaa.gov/")
               for entry in lagging_payload["files"])


# --------------------------------------------------------------------------
# Receipts
# --------------------------------------------------------------------------

def test_a_resume_across_endpoints_is_the_same_request(tmp_path):
    """The ladder head may differ run to run; that is not a new request.

    The guard exists to stop two different CYCLES publishing one
    SHA256SUMS.  Recording the served endpoint as request identity
    would have made an ordinary fall-through look like that -- and the
    transfer host is now a probed fact, so it differs run to run by
    design.
    """

    plan = _fresh_plan()
    fetch_routes.run_plan(
        plan, out=tmp_path, progress=lambda *_: None, probe=_mirror_lagging,
        downloader=_ladder_downloader(failing="__never__", error=None,
                                      seen=[]))
    second = []
    payload = fetch_routes.run_plan(
        plan, out=tmp_path, progress=lambda *_: None, probe=_mirror_caught_up,
        downloader=_ladder_downloader(
            failing="nomads.ncep.noaa.gov",
            error=_http_error(503), seen=second))
    assert all(entry["reused"] for entry in payload["files"])
    assert second == []


# --------------------------------------------------------------------------
# The legacy transports walk the same ladder
# --------------------------------------------------------------------------

def _grib2_stream(messages: int) -> bytes:
    one = (b"GRIB" + b"\x00\x00" + b"\x00" + b"\x02"
           + (20).to_bytes(8, "big") + b"7777")
    return one * messages


def _recent_synoptic_cycle() -> datetime:
    """A GFS cycle inside the operational window, on a run hour."""

    now = _utc_now() - timedelta(hours=12)
    return now.replace(hour=(now.hour // 6) * 6)


def _gfs_env(monkeypatch, *, failing: str, error, mirrored: bool = False):
    """The whole-object route with no network.

    ``mirrored`` is what the availability probe answers: has the
    archive caught up with this cycle?  False is publication lag -- the
    case the operational server exists for -- and it is the default
    here so every test about FALL-THROUGH keeps the operational server
    as its transfer head.
    """

    urls: list[str] = []

    def download(url, destination, **kwargs):
        urls.append(url)
        if failing in url:
            raise error
        destination.write_bytes(_grib2_stream(3))

    monkeypatch.setattr(gfs_transport, "_download", download)
    monkeypatch.setattr(fetch, "gfs_live_index", lambda *a, **k: None)
    monkeypatch.setattr(fetch, "_gfs_index_record_count",
                        lambda url, **kwargs: 3)
    monkeypatch.setattr(fetch, "_head_ok", lambda url: mirrored)
    return urls


def test_the_whole_object_route_asks_the_operational_server_first(
        tmp_path, monkeypatch):
    """While the archive lags, freshness owns the transfer."""

    urls = _gfs_env(monkeypatch, failing="__never__", error=None)
    manifest = fetch.fetch_gfs_fullfile(
        cycle=_recent_synoptic_cycle(), hours=(0,), area=None,
        out=tmp_path / "gfs", progress=lambda line: None)
    assert all(url.startswith(fetch.GFS_NOMADS_BASE) for url in urls)
    document = json.loads(manifest.read_text())
    assert document["endpoints"]["considered"] == [
        rung.name for rung in fetch_endpoints.serving_ladder(
            "gfs", cycle=_recent_synoptic_cycle())]
    assert document["endpoints"]["served"] == ["nomads"]
    assert document["files"][0]["endpoint"] == "nomads"


def test_the_whole_object_route_takes_the_archive_once_it_has_the_cycle(
        tmp_path, monkeypatch):
    """The measured cost this refinement removes.

    Tonight's real run: the operational server paced whole-file
    transfers at about 3 MB/s per file at peak hours, so 3.4 GB took
    ~20 min where the archive had served the same volume in ~3 min.
    The archive is the same bytes under the same key.
    """

    urls = _gfs_env(monkeypatch, failing="__never__", error=None,
                    mirrored=True)
    manifest = fetch.fetch_gfs_fullfile(
        cycle=_recent_synoptic_cycle(), hours=(0,), area=None,
        out=tmp_path / "gfs", progress=lambda line: None)
    assert urls and all(url.startswith(fetch.GFS_S3_BASE) for url in urls)
    document = json.loads(manifest.read_text())
    assert document["endpoints"]["considered"] == [
        rung.name for rung in fetch_endpoints.serving_ladder(
            "gfs", cycle=_recent_synoptic_cycle())]
    assert document["endpoints"]["served"] == ["s3"]
    assert document["endpoints"]["transfer_preference"] == [
        rung.name for rung in fetch_endpoints.transfer_order(
            fetch_endpoints.serving_ladder("gfs", cycle=_recent_synoptic_cycle()))]
    assert document["transport"] == "s3"


def test_a_mirrored_whole_object_fetch_still_falls_back_to_freshness(
        tmp_path, monkeypatch):
    """Promotion REORDERS the ladder; it never shortens it."""

    urls = _gfs_env(monkeypatch, failing="noaa-gfs-bdp-pds",
                    error=_http_error(503), mirrored=True)
    manifest = fetch.fetch_gfs_fullfile(
        cycle=_recent_synoptic_cycle(), hours=(0,), area=None,
        out=tmp_path / "gfs", progress=lambda line: None)
    assert any(url.startswith(fetch.GFS_S3_BASE) for url in urls)
    assert any(url.startswith(fetch.GFS_NOMADS_BASE) for url in urls)
    document = json.loads(manifest.read_text())
    assert document["endpoints"]["served"] == ["nomads"]


def test_the_whole_object_route_falls_through_to_the_archive(
        tmp_path, monkeypatch):
    urls = _gfs_env(monkeypatch, failing="nomads.ncep.noaa.gov",
                    error=_http_error(503, {"Retry-After": "900"}))
    manifest = fetch.fetch_gfs_fullfile(
        cycle=_recent_synoptic_cycle(), hours=(0,), area=None,
        out=tmp_path / "gfs", progress=lambda line: None)
    assert any(url.startswith(fetch.GFS_NOMADS_BASE) for url in urls)
    assert any(url.startswith(fetch.GFS_S3_BASE) for url in urls)
    document = json.loads(manifest.read_text())
    assert document["endpoints"]["served"] == ["s3"]
    assert document["transport"] == "s3"


def test_the_whole_object_route_refuses_naming_every_endpoint(
        tmp_path, monkeypatch):
    _gfs_env(monkeypatch, failing="", error=_http_error(403))
    with pytest.raises(RuntimeError) as error:
        fetch.fetch_gfs_fullfile(
            cycle=_recent_synoptic_cycle(), hours=(0,), area=None,
            out=tmp_path / "gfs", progress=lambda line: None)
    message = str(error.value)
    assert "nomads.ncep.noaa.gov" in message
    assert "noaa-gfs-bdp-pds" in message
    assert message.count("403") >= 2


def test_an_archive_era_cycle_never_asks_the_operational_server(
        tmp_path, monkeypatch):
    """The doomed attempt this ladder exists to skip."""

    urls = _gfs_env(monkeypatch, failing="nomads.ncep.noaa.gov",
                    error=AssertionError("must not be asked"))
    fetch.fetch_gfs_fullfile(
        cycle=datetime(2026, 7, 28, 6), hours=(0,), area=None,
        out=tmp_path / "gfs", progress=lambda line: None)
    assert urls and all(url.startswith(fetch.GFS_S3_BASE) for url in urls)


def test_a_pinned_host_on_the_whole_object_route_does_not_fall_through(
        tmp_path, monkeypatch):
    urls = _gfs_env(monkeypatch, failing="nomads.ncep.noaa.gov",
                    error=_http_error(503))
    with pytest.raises(RuntimeError):
        fetch.fetch_gfs_fullfile(
            cycle=_recent_synoptic_cycle(), hours=(0,), area=None,
            out=tmp_path / "gfs", transport="nomads",
            progress=lambda line: None)
    assert all(url.startswith(fetch.GFS_NOMADS_BASE) for url in urls)


def test_archives_that_lag_do_not_serve_a_cycle_younger_than_their_delay():
    """WOOF production run_6452d3110cf0705f (2026-10-26): GFS 06Z asked for at 07:29Z, before it posted, was refused as
    NCAR GDEX's 404. An archive that takes a cycle some hours after it runs is not asked for a younger one."""
    from datetime import datetime
    from woof.fetch_endpoints import serving_ladder
    now = datetime(2026, 10, 26, 7, 29)
    young = [e.name for e in serving_ladder("gfs", cycle=datetime(2026, 10, 26, 6), now=now)]
    assert "gdex" not in young and "ncei-0p5" not in young and young[:2] == ["nomads", "s3"]
    day_old = [e.name for e in serving_ladder("gfs", cycle=datetime(2026, 10, 25, 0), now=now)]
    assert "gdex" not in day_old
    old = [e.name for e in serving_ladder("gfs", cycle=datetime(2026, 10, 20, 0), now=now)]
    assert "gdex" in old and "ncei-0p5" in old
