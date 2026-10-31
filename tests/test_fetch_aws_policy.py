"""Hosted AWS acquisition policy, with ordinary install defaults preserved."""

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from woof import (fetch, fetch_as_posted, fetch_endpoints, fetch_pool,
                   fetch_routes, go_cli, runplan, source_readiness, source_posting)
from woof.chain_events import HostedPostingRelay


@pytest.fixture
def aws_policy(monkeypatch):
    monkeypatch.setenv(fetch_endpoints.FETCH_POLICY_ENV, "aws")


def test_transport_names_come_from_the_declared_aws_endpoints():
    assert fetch_endpoints.FETCH_POLICY_CONTRACT == "aws-streaming-v1"
    for source in ("gfs", "gdas", "hrrr"):
        assert fetch_endpoints.aws_transport_for_source(source) == "s3"
    for source in ("hrrr-prs", "rap", "rrfs", "gefs", "aigefs", "ecmwf-open-data", "aifs"):
        assert fetch_endpoints.aws_transport_for_source(source) == "aws"
    for source in ("aigfs", "icon-global", "icon-eu", "gem-gdps", "era5"):
        assert fetch_endpoints.aws_transport_for_source(source) is None


def test_an_ordinary_install_still_walks_nomads_first(monkeypatch):
    monkeypatch.delenv(fetch_endpoints.FETCH_POLICY_ENV, raising=False)
    cycle = datetime(2026, 9, 30, 12)
    assert fetch_endpoints.serving_ladder("hrrr", cycle=cycle, now=cycle)[0].name == "nomads"
    assert go_cli.pinned_transport({"source": "hrrr"}) == (None, None)
    assert fetch_pool.host_worker_cap("nomads.ncep.noaa.gov", 6) == 2
    # Since the GFS archive lane (32dea003f45, lane/cap-gfs-archive) the S3
    # host declares dated layouts, and pinning a layout host makes the fetch
    # take whole objects itself (fetch._fetch_main), so the pin is admitted;
    # the crop host, which has no whole-object layout, is still refused.
    assert fetch.transport_refusal("gfs", "s3") is None
    assert "archive host" in fetch.transport_refusal("gfs", "nomads")


def test_aws_policy_pins_each_source_but_preserves_a_nomads_only_primary(aws_policy):
    cycle = datetime(2026, 9, 30, 12)
    for source, host in (("hrrr", "s3"), ("rap", "aws"), ("aifs", "aws")):
        assert [r.name for r in fetch_endpoints.serving_ladder(
            source, cycle=cycle, now=cycle, pinned="nomads")] == [host]
    assert [r.name for r in fetch_endpoints.serving_ladder("aigfs", cycle=cycle, now=cycle)] == ["nomads"]
    assert fetch.requested_as_posted(SimpleNamespace(source="aigfs", as_posted=True)) is True
    assert fetch_pool.host_worker_cap("nomads.ncep.noaa.gov", 6) == 1
    assert fetch_pool.host_worker_cap("noaa-hrrr-bdp-pds.s3.amazonaws.com", 6) == 6


def test_latest_hrrr_keeps_the_newest_start_when_later_soil_is_missing(aws_policy):
    now = datetime(2026, 9, 30, 23, 59)
    grid, _basis = fetch.provider_cycle_grid("hrrr", None, now=now)
    newest = grid.snap(now)
    missing = fetch.hrrr_object_url(newest, 6, "wrfprs", transport="s3")
    seen = []

    def probe(url):
        seen.append(url)
        return url != missing

    selected = fetch.resolve_latest_cycle("hrrr", 12, now=now, probe=probe, as_posted=True)
    assert selected == newest
    assert missing not in seen
    assert all("noaa-hrrr-bdp-pds.s3.amazonaws.com" in url for url in seen)
    accepted = {fetch.hrrr_object_url(selected, lead, product, transport="s3")
                for lead in (0, 1) for product in fetch.hrrr_lead_products()}
    assert accepted <= set(seen)


def test_latest_does_not_accept_a_nomads_only_cycle_when_aws_is_missing(aws_policy):
    seen = []

    def probe(url):
        seen.append(url)
        return "nomads.ncep.noaa.gov" in url

    with pytest.raises(RuntimeError, match="on s3"):
        fetch.resolve_latest_cycle("hrrr", 12, now=datetime(2026, 9, 30, 23, 59), probe=probe)
    assert seen and all("amazonaws.com" in url for url in seen)


def test_latest_aifs_includes_step_zero_surface_for_a_later_start(aws_policy):
    now = datetime(2026, 9, 30, 23, 59)
    grid, _basis = fetch.provider_cycle_grid("aifs", None, now=now)
    newest = grid.snap(now)
    latest_plan = fetch_routes.resolve_request("aifs", cycle=newest, start_hour=12, hours=6)
    missing = next(obj.url for obj in latest_plan.objects if obj.lead == 0)
    seen = []

    def probe(url):
        seen.append(url)
        return url != missing

    selected = fetch.resolve_latest_cycle("aifs", 18, start_hour=12, now=now, probe=probe)
    assert selected == newest - timedelta(hours=6)
    assert missing in seen
    # fix/ecmwf-seaice-aifs-ens (a2f2a6a4f1): the aifs route takes its sea ice
    # from the same cycle's GFS analysis, so the latest-cycle probe also asks
    # AWS for that donor's f000 and for nothing else outside ECMWF's bucket.
    donor = [url for url in seen
             if "ecmwf-forecasts.s3.eu-central-1.amazonaws.com" not in url]
    assert donor and all(
        url.startswith("https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.")
        and url.endswith(".pgrb2.0p25.f000") for url in donor)


@pytest.mark.parametrize("source", ["aigfs", "aigefs"])
def test_latest_hybrid_requires_its_own_aws_donor(source, aws_policy):
    now = datetime(2026, 9, 30, 23, 59)
    grid, _basis = fetch.provider_cycle_grid(source, None, now=now)
    newest = grid.candidates(now, 6)[0]
    missing = fetch.gfs_object_url(newest, 0, "gdas", transport="s3")
    seen = []

    def probe(url):
        seen.append(url)
        return url != missing

    selected = fetch.resolve_latest_cycle(source, 6, now=now, probe=probe)
    assert selected == newest - timedelta(hours=6)
    assert missing in seen
    donor_urls = [url for url in seen if "/gdas." in url]
    assert donor_urls and all("noaa-gfs-bdp-pds.s3.amazonaws.com" in url for url in donor_urls)


def test_go_and_run_plan_select_the_gfs_archive_records_path(tmp_path, aws_policy):
    config = Path(__file__).resolve().parents[1] / "configs/gfs_12km_quickstart.toml"
    plan = go_cli.plan_from_config(config, outdir=tmp_path / "run", as_posted=True)
    assert plan["transport"] == "s3"
    assert plan["transport_from"] == fetch_endpoints.FETCH_POLICY_ENV
    assert plan["as_posted"] is True
    command = go_cli.fetch_command(plan)
    assert command[command.index("--mode") + 1] == fetch.GFS_RECORDS_MODE == "idx-subset"
    assert command[command.index("--transport") + 1] == "s3"
    assert "--whole-cycle" not in command
    hints = runplan._pinned_fetch_hints(
        SimpleNamespace(run_options={}), {"source": "gfs", "as_posted": True, "late_after_minutes": 10})
    assert hints == {"source": "gfs", "transport": "s3", "as_posted": True, "late_after_minutes": 10}
    arguments = runplan._fetch_arguments_from_hints(hints, out=tmp_path / "data")
    assert arguments[arguments.index("--mode") + 1] == "idx-subset"
    assert fetch.transport_refusal("gfs", "s3") is None


def test_later_start_keeps_its_exact_window_in_latest_query(aws_policy):
    args = SimpleNamespace(source="gfs", hours=6, forecast_start_hour=12,
                           cadence=3, transport=None, as_posted=None)
    source, last, options = fetch.latest_cycle_request(args)
    assert (source, last, options["start_hour"], options["cadence"], options["transport"]) == ("gfs", 18, 12, 3, "s3")


def test_older_aws_cycle_preserves_the_fixed_valid_start(aws_policy):
    valid = datetime(2026, 9, 30, 15)
    newest = datetime(2026, 9, 30, 12)
    missing = fetch.gfs_object_url(newest, 6, "gfs", transport="s3")
    seen = []

    def probe(url):
        seen.append(url)
        return url != missing

    cycle, lead = fetch.resolve_cycle_for_valid_start(
        "gfs", valid, 12, 3, now=datetime(2026, 9, 30, 23), probe=probe)
    assert (cycle, lead) == (datetime(2026, 9, 30, 6), 9)
    assert cycle + timedelta(hours=lead) == valid
    assert all("noaa-gfs-bdp-pds.s3.amazonaws.com" in url for url in seen)
    assert fetch.gfs_object_url(cycle, 12, "gfs", transport="s3") in seen
    assert fetch.gfs_object_url(cycle, 21, "gfs", transport="s3") not in seen


def test_a_failed_aws_transfer_never_falls_through_to_nomads(tmp_path, aws_policy):
    cycle = datetime(2026, 9, 30, 12)
    plan = fetch_routes.resolve_request("rap", cycle=cycle, hours=0)
    assert plan.host.name == "aws"
    seen = []

    def download(url, dest, **kwargs):
        seen.append(url)
        raise HTTPError(url, 404, "missing", {}, None)

    with pytest.raises(Exception, match="404"):
        fetch_routes.run_plan(plan, out=tmp_path, progress=lambda *_: None,
                              downloader=download, probe=lambda *_: False)
    assert seen and all("noaa-rap-pds.s3.amazonaws.com" in url for url in seen)


def test_a_nomads_only_primary_passes_the_aws_pin_to_its_donor(tmp_path, monkeypatch, aws_policy):
    plan = fetch_routes.resolve_request("aigfs", cycle=datetime(2026, 9, 30, 12), hours=6)
    requested = []

    def donor(**kwargs):
        requested.append(kwargs)
        out = kwargs["out"]
        out.mkdir(parents=True)
        path = out / "fetch-manifest.json"
        path.write_text(json.dumps({"files": [{"name": "donor.grib2"}]}))
        return path

    monkeypatch.setattr(fetch, "fetch_gfs_fullfile", donor)
    fetch._fetch_route_donors(plan, SimpleNamespace(out=tmp_path, force_refetch=False, fetch_workers=None))
    assert plan.host.name == "nomads"
    assert requested[0]["source"] == "gdas" and requested[0]["transport"] == "s3"


class PublicationClock:
    def __init__(self, now):
        self.value = now
        self.waits = []
        self.on_sleep = None

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.value += timedelta(seconds=seconds)
        if self.on_sleep is not None:
            self.on_sleep()


def _pinned_fetch(tmp_path, source="hrrr-prs", hours=48, *extra):
    return runplan._parse_fetch_arguments([
        "--source", source, "--cycle", "2026-10-01T18", "--hours", str(hours),
        "--cadence", "6" if source == "aigefs" else "3" if source == "gfs" else "1", "--out", str(tmp_path),
        *(["--area", "25,-105,45,-85"] if source in ("gfs", "hrrr") else []),
        *extra])



def _fixture_downloader(downloads, clock):
    def download(url, dest, **kwargs):
        downloads.append((url, clock.now()))
        dest.parent.mkdir(parents=True, exist_ok=True)
        body = b"GRIB" + url.encode() + b"7777"
        dest.write_bytes(body)
        return {"name": dest.name, "bytes": len(body), "url": url,
                "sha256": hashlib.sha256(body).hexdigest()}
    return download


def _replay_route(monkeypatch, clock, probe, downloads):
    monkeypatch.setattr(fetch, "_head_answer", probe)
    monkeypatch.setattr(fetch_as_posted, "now_utc", clock.now)
    monkeypatch.setattr(fetch_as_posted, "pause", clock.sleep)
    real = fetch_routes.run_plan
    # The replay's fixture transport replaces whatever downloader the fetch
    # chose (the Rust stream door on a GRIB2 route).
    monkeypatch.setattr(fetch_routes, "run_plan", lambda plan, **kw: real(
        plan, **{**kw, "downloader": _fixture_downloader(downloads, clock), "probe": probe}))


@pytest.mark.parametrize("missing_lead", [24, 48])
def test_scheduled_aws_cycle_streams_start_before_later_inputs_post(
        tmp_path, monkeypatch, aws_policy, missing_lead, capsys):
    cycle = datetime(2026, 10, 1, 18)
    # First inputs exist before the prediction.  Positive HEADs must win.
    clock = PublicationClock(cycle + timedelta(minutes=5))
    started = clock.now()
    reveal = started + timedelta(seconds=45)
    queried, downloads, markers, events, waits = [], [], [], [], []

    def probe(url):
        queried.append(url)
        return not (f"t18z.wrfprsf{missing_lead:02d}." in url and clock.now() < reveal)

    document, code = source_readiness.readiness(
        "hrrr-prs", cycle, 48, cadence=1, now=clock.now(), probe=probe)
    assert code == 0 and document["as_posted"] is True
    # The start is f000, the analysis-time vegetation_surface wrfsfc
    # object the route fetches with it (6a69b356f), and f001.
    assert [(need["role"], need["lead"]) for need in document["start_needs"]] == [
        ("analysis", 0), ("supplement:vegetation_surface", 0), ("first_boundary", 1)]
    _replay_route(monkeypatch, clock, probe, downloads)
    relay = HostedPostingRelay(SimpleNamespace(emit=lambda tag, **fields: events.append((tag, fields))),
                               data_dir=tmp_path)
    def while_waiting():
        relay._relay.relay_posting()
        schedule = json.loads((tmp_path / "posting/schedule.json").read_text())
        waits.append(schedule["leads"][missing_lead]["state"])

    clock.on_sleep = while_waiting
    publish = fetch_as_posted.PostingLoop.publish

    def published(self, lead, objects, **kw):
        result = publish(self, lead, objects, **kw)
        markers.append((lead, clock.now()))
        relay._relay.relay_posting()
        return result

    monkeypatch.setattr(fetch_as_posted.PostingLoop, "publish", published)
    args = _pinned_fetch(tmp_path, "hrrr-prs", 48, "--fetch-workers", "1")
    assert fetch._fetch_main(args) == 0
    assert markers[:2] == [(0, started), (1, started)]
    assert [lead for lead, _ in markers] == list(range(49))
    assert markers[missing_lead][1] >= reveal
    assert clock.now() == started + timedelta(seconds=60)
    # 49 wrfprs leads plus the one step-0 wrfsfc supplement.
    assert len(downloads) == 50
    assert all(moment == started for _, moment in downloads[:3])
    assert sum("wrfsfcf00." in url for url, _ in downloads) == 1
    assert all("noaa-hrrr-bdp-pds.s3.amazonaws.com" in url for url, _ in downloads)
    assert all("noaa-hrrr-bdp-pds.s3.amazonaws.com" in url for url in queried)
    manifest = json.loads((tmp_path / "fetch-manifest.json").read_text())
    assert manifest["complete"] is True and len(manifest["files"]) == 50
    assert (tmp_path / "SHA256SUMS").is_file()
    schedule = json.loads((tmp_path / "posting/schedule.json").read_text())
    assert schedule["as_posted"] is True
    assert all(row["state"] == "ready" for row in schedule["leads"])
    assert {"posting_schedule", "lead_ready", "lead_posted"} <= {tag for tag, _ in events}
    assert waits and all(state == "waiting" for state in waits)
    assert clock.waits and all(seconds <= 30 for seconds in clock.waits)
    assert "not posted yet" in capsys.readouterr().out


_START_OBJECTS = ("wrfprsf00.", "wrfsfcf00.", "wrfprsf01.")


def test_streaming_timeout_keeps_the_verified_start_prefix(tmp_path, monkeypatch, aws_policy):
    clock = PublicationClock(datetime(2026, 10, 1, 18, 5))
    downloads = []
    probe = lambda url: any(name in url for name in _START_OBJECTS)
    _replay_route(monkeypatch, clock, probe, downloads)
    args = _pinned_fetch(tmp_path, "hrrr-prs", 48, "--fetch-workers", "1", "--wait-timeout-minutes", "1")
    assert fetch.fetch_main(args) == 75
    failure = json.loads((tmp_path / "posting/failed.json").read_text())
    assert failure["budget"] == "wait_timeout_minutes" and failure["lead"] == 2
    assert failure["leads_kept"] == [0, 1]
    assert len(downloads) == 3 and sum(clock.waits) == 60
    manifest = json.loads((tmp_path / "fetch-manifest.json").read_text())
    assert manifest["complete"] is False and len(manifest["files"]) == 3


def test_an_unposted_start_supplement_holds_the_start(tmp_path, monkeypatch, aws_policy):
    """The step-0 supplement is a start need of its own, asked by its own HEAD.

    Lead 0's posted answer covers only its primary object.  Skipping the
    supplement because lead 0 was already seen transferred the wrfsfc f000
    without one HEAD, so a supplement the mirror had not caught up with
    would fail as a download instead of being waited for.
    """
    clock = PublicationClock(datetime(2026, 10, 1, 18, 5))
    downloads, asked = [], []

    def probe(url):
        asked.append(url)
        return any(f"wrfprsf{lead:02d}." in url for lead in (0, 1))

    _replay_route(monkeypatch, clock, probe, downloads)
    args = _pinned_fetch(tmp_path, "hrrr-prs", 48, "--fetch-workers", "1", "--wait-timeout-minutes", "1")
    assert fetch.fetch_main(args) == 75
    failure = json.loads((tmp_path / "posting/failed.json").read_text())
    assert failure["budget"] == "wait_timeout_minutes" and failure["lead"] == 0
    assert failure["leads_kept"] == []
    assert any("wrfsfcf00." in url for url in asked)
    assert downloads == []


def test_streaming_start_wait_stops_when_its_run_is_cancelled(tmp_path, monkeypatch, aws_policy):
    clock = PublicationClock(datetime(2026, 10, 1, 18, 5))
    stop = fetch_as_posted.stop_event(tmp_path)
    clock.on_sleep = stop.set
    _replay_route(monkeypatch, clock, lambda url: False, [])
    try:
        with pytest.raises(fetch_as_posted.FetchStopped, match="stopped"):
            fetch._fetch_main(_pinned_fetch(tmp_path))
        assert len(clock.waits) == 1
    finally:
        fetch_as_posted.release_stop(tmp_path)


def test_first_aws_probe_and_later_probe_ignore_predictions(tmp_path, aws_policy):
    cycle = datetime(2026, 10, 1, 18)
    clock = PublicationClock(cycle + timedelta(minutes=5))
    window = source_readiness.resolve_window("hrrr-prs", cycle, 48, cadence=1, now=clock.now())
    loop = fetch_as_posted.PostingLoop(window, tmp_path, probe=lambda url: True,
                                     now=clock.now, sleep=lambda _: pytest.fail("available AWS waited"))
    try:
        loop.start()
        loop.wait_start()
        assert loop(48) == "aws"
        assert loop.posted_now(24)
    finally:
        loop.close()


def test_missing_aws_start_returns_poll_interval_before_prediction(aws_policy):
    document, code = source_readiness.readiness(
        "hrrr-prs", datetime(2026, 10, 1, 18), 48,
        now=datetime(2026, 10, 1, 18, 5), probe=lambda url: False)
    assert code == 75 and document["as_posted"] is True
    assert document["retry_after_seconds"] == 30


def test_latest_selects_early_aws_start_before_prediction_and_ignores_later_leads(aws_policy):
    cycle = datetime(2026, 10, 1, 18)
    seen = []

    def probe(url):
        seen.append(url)
        return "/hrrr.20261001/" in url and "t18z." in url and any(
            name in url for name in _START_OBJECTS)

    assert fetch.resolve_latest_cycle("hrrr-prs", 48, now=cycle + timedelta(minutes=5), probe=probe) == cycle
    assert seen and not any("wrfprsf24." in url or "wrfprsf48." in url for url in seen)


@pytest.mark.parametrize("missing_product,suffix", [
    ("wrfnat", ""), ("wrfprs", ""), ("wrfprs", ".idx"),
    ("wrfsfc", ""), ("wrfsfc", ".idx")])
def test_latest_backs_off_only_when_first_inputs_are_missing(aws_policy, missing_product, suffix):
    now = datetime(2026, 10, 1, 23, 59)
    cycle = datetime(2026, 10, 1, 23)
    missing = fetch.hrrr_object_url(cycle, 1, missing_product, transport="s3") + suffix
    seen = []

    def probe(url):
        seen.append(url)
        return url != missing

    assert fetch.resolve_latest_cycle("hrrr", 12, now=now, probe=probe) == cycle - timedelta(hours=1)
    assert missing in seen and all("amazonaws.com" in url for url in seen)


def test_fixed_valid_start_keeps_the_newest_first_needs_despite_missing_later_hours(aws_policy):
    valid = datetime(2026, 10, 1, 15)
    seen = []
    missing = fetch.gfs_object_url(datetime(2026, 10, 1, 12), 15, "gfs", transport="s3")
    cycle, lead = fetch.resolve_cycle_for_valid_start(
        "gfs", valid, 12, 3, now=datetime(2026, 10, 1, 23),
        probe=lambda url: seen.append(url) or url != missing)
    assert (cycle, lead) == (datetime(2026, 10, 1, 12), 3)
    assert cycle + timedelta(hours=lead) == valid and missing not in seen


def test_calendar_latest_accepts_positive_aws_start_before_its_prediction(aws_policy):
    from woof.source_availability import resolve_latest

    seen = []
    cycle = datetime(2026, 10, 1, 18)
    result = resolve_latest("hrrr-prs", 48, now=cycle + timedelta(minutes=5),
                            probe=lambda url: seen.append(url) or True)
    assert result["selected_cycle"] == "2026-10-01T18"
    assert not any("wrfprsf24." in url or "wrfprsf48." in url for url in seen)


def test_calendar_latest_respects_explicit_whole_cycle_under_aws(aws_policy):
    from woof.source_availability import resolve_latest

    seen = []
    resolve_latest("hrrr-prs", 48, now=datetime(2026, 10, 1, 23, 59),
                   as_posted=False, probe=lambda url: seen.append(url) or True)
    assert any("wrfprsf48." in url for url in seen)


def test_explicit_whole_cycle_stays_explicit_under_aws(aws_policy):
    args = SimpleNamespace(source="hrrr-prs", hours=48, forecast_start_hour=0,
                           cadence=1, transport=None, as_posted=False)
    _source, _last, options = fetch.latest_cycle_request(args)
    assert options["as_posted"] is False
    assert fetch.requested_as_posted(args) is False
    seen = []
    fetch.resolve_latest_cycle("hrrr-prs", 48, now=datetime(2026, 10, 1, 23, 59),
                               as_posted=False, probe=lambda url: seen.append(url) or True)
    assert any("wrfprsf48." in url for url in seen)


def test_ordinary_library_latest_default_and_explicit_streaming_are_unchanged(monkeypatch):
    monkeypatch.delenv(fetch_endpoints.FETCH_POLICY_ENV, raising=False)
    now = datetime(2026, 10, 1, 23, 59)
    seen = []
    fetch.resolve_latest_cycle("hrrr", 12, now=now, probe=lambda url: seen.append(url) or True)
    assert any("wrfnatf12." in url for url in seen)
    seen.clear()
    fetch.resolve_latest_cycle("hrrr", 12, now=now, as_posted=True,
                               probe=lambda url: seen.append(url) or True)
    assert any("wrfnatf01." in url for url in seen)
    assert not any("wrfnatf12." in url for url in seen)


def test_go_default_preserves_streaming_and_preparation_hints(tmp_path, aws_policy):
    config = Path(__file__).resolve().parents[1] / "configs/gfs_12km_quickstart.toml"
    plan = go_cli.plan_from_config(config, outdir=tmp_path / "run", late_after_minutes=2)
    assert plan["as_posted"] is not False and plan["late_after_minutes"] == 2
    command = go_cli.fetch_command(plan)
    assert "--whole-cycle" not in command
    assert command[command.index("--late-after-minutes") + 1] == "2"
    hints = runplan._pinned_fetch_hints(SimpleNamespace(run_options={}), {"source": "gfs"})
    assert hints["transport"] == "s3" and hints.get("as_posted") is not False


@pytest.mark.parametrize("source", ["aigfs", "aigefs"])
def test_named_hybrid_stream_waits_for_its_own_aws_donor(source, tmp_path, monkeypatch, aws_policy):
    cycle = datetime(2026, 10, 1, 18)
    # The NOMADS-only primary retains its original prediction gate; its
    # GDAS donor is AWS pinned and can open before the donor prediction.
    clock = PublicationClock(
        source_posting.expected_at(source, cycle, 6) + timedelta(seconds=1)
        if source == "aigfs" else cycle + timedelta(minutes=5))
    reveal = clock.now() + timedelta(seconds=45)
    queried, fetched = [], []

    def probe(url):
        queried.append(url)
        return clock.now() >= reveal if "/gdas.20261001/18/" in url else True

    monkeypatch.setattr(fetch, "_head_answer", probe)
    monkeypatch.setattr(fetch_as_posted, "now_utc", clock.now)
    monkeypatch.setattr(fetch_as_posted, "pause", clock.sleep)
    monkeypatch.setattr(fetch_routes, "run_plan", lambda plan, **kw: fetched.append((plan, clock.now())))
    monkeypatch.setattr(fetch, "_fetch_route_donors", lambda *args: {
        "physical_analysis_surface_data": tmp_path / "gdas.grib2"})
    args = _pinned_fetch(tmp_path, "aigefs", 6)
    args.source = source
    assert fetch._fetch_main(args) == 0
    assert fetched[0][1] >= reveal and sum(clock.waits) == 60
    donor_urls = [url for url in queried if "/gdas." in url]
    assert donor_urls and all("noaa-gfs-bdp-pds.s3.amazonaws.com" in url for url in donor_urls)
    assert fetched[0][0].host.name == ("nomads" if source == "aigfs" else "aws")


def test_hosted_publication_allowances_are_endpoint_and_lead_data(monkeypatch):
    cycle = datetime(2026, 10, 1, 18)
    sources = ("hrrr", "hrrr-prs", "gfs", "gdas", "rap", "aigfs")
    monkeypatch.delenv(fetch_endpoints.FETCH_POLICY_ENV, raising=False)
    baseline = {(source, lead): source_posting.expected_at(source, cycle, lead)
                for source in sources for lead in (0, 1, 12)}
    monkeypatch.setenv(fetch_endpoints.FETCH_POLICY_ENV, "aws")
    for source in sources:
        for lead in (0, 1, 12):
            minutes = (3 if lead <= 1 else 11) if source in ("hrrr", "hrrr-prs") else 16 if source == "gfs" else 0
            assert source_posting.expected_at(source, cycle, lead) == baseline[source, lead] + timedelta(minutes=minutes)
            assert source_posting.late_after_minutes(source) == source_posting.posting(source).late_after_minutes


@pytest.mark.parametrize("source", ["gfs", "hrrr"])
def test_native_aws_fetch_starts_its_prefix_before_the_final_hour(
        source, tmp_path, monkeypatch, aws_policy):
    cycle = datetime(2026, 10, 1, 18)
    clock = PublicationClock(cycle + timedelta(minutes=5))
    started = clock.now()
    reveal = started + timedelta(seconds=45)
    prefixes, queried = [], []

    def probe(url):
        queried.append(url)
        return clock.now() >= reveal if (url.endswith("f006") or "wrfprsf06." in url) else True

    def transfer(**kw):
        assert kw["transport"] == "s3"
        prefixes.append((kw["hours"], clock.now()))
        files = []
        for lead in kw["hours"]:
            path = tmp_path / f"f{lead:03d}.grib2"
            path.write_bytes(b"GRIBfixture7777")
            files.append({"forecast_hour": lead, "name": path.name, "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "transport": "s3"})
        manifest = tmp_path / "fetch-manifest.json"
        manifest.write_text(json.dumps({"files": files}))
        (tmp_path / "SHA256SUMS").write_text("fixture\n")
        if kw.get("on_hour_ready"):
            for lead in kw["hours"]:
                kw["on_hour_ready"](lead, manifest)
        return manifest

    monkeypatch.setattr(fetch, "_head_answer", probe)
    monkeypatch.setattr(fetch_as_posted, "now_utc", clock.now)
    monkeypatch.setattr(fetch_as_posted, "pause", clock.sleep)
    monkeypatch.setattr(fetch, "fetch_gfs_fullfile" if source == "gfs" else "fetch_hrrr", transfer)
    monkeypatch.setattr(fetch, "_window_posted", lambda *a, **k: pytest.fail("AWS start probed final cycle"))
    assert fetch._fetch_main(_pinned_fetch(tmp_path, source, 6)) == 0
    assert prefixes[0] == ((0, 3) if source == "gfs" else tuple(range(6)), started)
    assert prefixes[-1][0][-1] == 6 and prefixes[-1][1] >= reveal
    assert all("amazonaws.com" in url for url in queried)


def test_a_throttling_aws_pin_keeps_the_other_mirrors_behind_it(aws_policy):
    """WOOF run_19793057ff2979e3 (2026-10-29): ECMWF's AWS mirror answered
    HEAD 200 and GET 503 SlowDown while Google and data.ecmwf.int served the
    same keys; the AWS-only ladder ended every IFS 9 km run at fetch (75)."""
    cycle = datetime(2026, 10, 28, 18)
    now = cycle + timedelta(hours=14)
    for source, rungs in (("ecmwf-open-data-0p1", ["aws", "google", "ecmwf"]),
                          ("ecmwf-ens", ["aws", "google", "ecmwf"]),
                          ("ecmwf-open-data", ["aws", "ecmwf"]),
                          ("aifs", ["aws", "ecmwf"])):
        assert [r.name for r in fetch_endpoints.serving_ladder(
            source, cycle=cycle, now=now, throttle_fallthrough=True)] == rungs, source
        # Cycle and readiness questions stay on the pinned mirror.
        assert [r.name for r in fetch_endpoints.serving_ladder(
            source, cycle=cycle, now=now)] == ["aws"], source
    # NOAA's buckets declare no throttle row: still AWS alone, never NOMADS.
    for source, host in (("hrrr", "s3"), ("rap", "aws"), ("gefs", "aws")):
        assert [r.name for r in fetch_endpoints.serving_ladder(
            source, cycle=cycle, now=now, throttle_fallthrough=True)] == [host], source


def test_a_slowdown_on_the_aws_mirror_is_served_by_the_next_mirror(tmp_path, aws_policy):
    import io
    cycle = datetime(2026, 10, 28, 18)
    plan = fetch_routes.resolve_request(
        "ecmwf-open-data-0p1", cycle=cycle, hours=0,
        now=cycle + timedelta(hours=14))
    assert [r.name for r in plan.ladder] == ["aws", "google", "ecmwf"]
    seen = []

    def download(url, dest, **kwargs):
        seen.append(url)
        if "amazonaws.com" in url:
            raise HTTPError(url, 503, "Slow Down", {}, io.BytesIO(
                b"<Error><Code>SlowDown</Code><Message>Please reduce your "
                b"request rate.</Message></Error>"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"GRIB" + bytes(60) + b"7777")
        return {"name": dest.name, "bytes": dest.stat().st_size,
                "sha256": "x", "url": url}

    fetch_routes.run_plan(plan, out=tmp_path, progress=lambda *_: None,
                          downloader=download, probe=lambda *_: False)
    assert any("amazonaws.com" in url for url in seen)
    assert any("storage.googleapis.com" in url for url in seen)
    assert not any("data.ecmwf.int" in url for url in seen)
