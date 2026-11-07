"""Every fetch step uses the request the parser accepted.

The breakages these guard, each seen through the production code with
only the network replaced:

* a library fetch (``fetch_gfs``, ``fetch_gfs_fullfile``, ``fetch_hrrr``)
  into a folder holding another DAY's files kept the old bytes and
  published them under the new date, because file names carry the cycle
  hour but not the date and only the command line compared requests;
* a named HRRR cycle with ``--wait-for`` was refused by the publication
  check before its own wait loop could start, and a folder already
  holding the whole request was refused when the provider could not be
  asked;
* ``--cycle latest --transport s3`` picked a cycle only the operational
  server had, then downloaded from S3 alone;
* a run plan's ``fetch.args`` spelled ``--source=gfs`` or
  ``--cycle=latest`` resolved another source's cycle, or none at all;
* a plan's relative polygon was read from the folder the plan was
  launched from instead of the plan's own folder;
* ``--transport auto``, the HRRR default spelled out, was handed on as
  a host name and refused every HRRR fetch that typed it;
* an old cycle pinned to a host past its retention was told it was
  "not published yet" and pointed at today's cycle;
* a finished folder's re-run read the live index before using its own
  files, and a full-file re-run's receipt named the ladder head instead
  of the host that served the bytes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import functools
import json
from pathlib import Path
import re
from urllib.parse import unquote

import pytest

import woof.cli as cli
import woof.fetch as fetch
from woof import runplan, rustwx_fetch
from tools import download_gfs_native_subset as gfs_transport
from tools import download_hrrr_native_subset as hrrr_transport


AREA = "30,-100,40,-90"


class _Stop(Exception):
    """Raised by a stubbed transfer once the request it was handed is known."""


def _stream(messages: int, stamp: bytes) -> bytes:
    """Envelope-valid GRIB2 messages whose reserved octets carry ``stamp``."""
    one = (b"GRIB" + stamp + b"\x00\x02" + (20).to_bytes(8, "big")
           + b"7777")
    return one * messages


def _stamp(url: str) -> bytes:
    """The date an object URL asks for, as (month, day) octets."""
    found = re.search(r"(?:gfs|hrrr)\.(\d{8})", unquote(url))
    assert found, url
    day = datetime.strptime(found.group(1), "%Y%m%d")
    return bytes((day.month, day.day))


def _cycle_of(url: str) -> datetime:
    """The cycle a GFS or HRRR object URL names."""
    text = unquote(url)
    found = (re.search(r"hrrr\.(\d{8})/conus/hrrr\.t(\d{2})z", text)
             or re.search(r"gfs\.(\d{8})/(\d{2})/", text))
    assert found, url
    return datetime.strptime(found.group(1) + found.group(2), "%Y%m%d%H")


def _files(out: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes()
            for path in sorted(out.iterdir()) if path.is_file()}


def _manifest(out: Path) -> dict:
    return json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())


def _payload_stamps(out: Path, role: str) -> set[bytes]:
    return {(out / entry["name"]).read_bytes()[4:6]
            for entry in _manifest(out)["files"] if entry["role"] == role}


def _dated_gfs_download(downloads: list):
    def download(url, destination, **kwargs):
        downloads.append(url)
        destination.write_bytes(
            _stream(fetch.GFS_SUBSET_RECORD_COUNT, _stamp(url)))
    return download


def _dated_hrrr_product(downloads: list):
    def product(request, *, workers, retries, expected_count=-1):
        downloads.append(request.url)
        count = (hrrr_transport.SOIL_RECORD_COUNT if request.kind == "soil"
                 else hrrr_transport.ATMOSPHERE_RECORD_COUNT)
        request.destination.write_bytes(_stream(count, _stamp(request.url)))
        request.index_path.write_text("1:0:fixture\n", encoding="ascii")
        return {"kind": request.kind}
    return product


@pytest.fixture
def offline_index(monkeypatch):
    """The GFS live inventory cannot be read; the certified bars stand in."""
    monkeypatch.setattr(fetch, "gfs_live_index", lambda *a, **k: None)
    monkeypatch.setattr(fetch, "_gfs_index_record_count",
                        lambda url, **kwargs: None)


@pytest.fixture
def python_engine(monkeypatch):
    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: None)
    # Since 49f8a677f every HRRR soil file also carries the cycle's runtime
    # surface records, which only the Rust backbone cuts; with the
    # backbone hidden here that step is not this file's question (request
    # identity), so it appends nothing and records each table field as
    # appended.  The record matters: fetch_hrrr reuses a soil file only
    # when its receipt names every runtime surface field, so an empty
    # binding would fetch every soil file again.
    # tests/test_fetch.py and tests/test_runtime_surface_fallback.py own
    # the step itself.
    from woof import runtime_surface_fetch

    def recorded(path, *, adapter, **kwargs):
        return [{"field": name, "units": units, "selector": selector}
                for name, _product, selector, units, *_ in
                adapter.runtime_surface_fields]

    monkeypatch.setattr(runtime_surface_fetch,
                        "append_runtime_surface_records", recorded)


def _with_probe(monkeypatch, name: str, probe, **fixed):
    """Hand a fixture probe to a production function whose default is bound."""
    real = getattr(fetch, name)

    def wrapped(*args, **kwargs):
        kwargs.setdefault("probe", probe)
        for key, value in fixed.items():
            kwargs.setdefault(key, value)
        return real(*args, **kwargs)

    monkeypatch.setattr(fetch, name, wrapped)


# ---------------------------------------------------------------------------
# A library fetch compares requests inside its own output lock
# ---------------------------------------------------------------------------

def test_fetch_gfs_never_publishes_the_previous_days_files_under_a_new_date(
        tmp_path, monkeypatch, offline_index):
    downloads: list[str] = []
    monkeypatch.setattr(gfs_transport, "_download",
                        _dated_gfs_download(downloads))
    out = tmp_path / "gfs"
    common = dict(hours=(0, 3), area=fetch.parse_area(AREA), out=out,
                  progress=lambda line: None)
    fetch.fetch_gfs(cycle=datetime(2026, 1, 31, 6), **common)
    before = _files(out)
    downloads.clear()

    with pytest.raises(ValueError, match="different request") as refusal:
        fetch.fetch_gfs(cycle=datetime(2026, 2, 1, 6), **common)
    assert "2026-02-01T06" in str(refusal.value)
    assert downloads == []
    assert _files(out) == before
    assert _manifest(out)["cycle"] == "2026-01-31T06:00:00Z"

    fetch.fetch_gfs(cycle=datetime(2026, 2, 1, 6), force=True, **common)
    assert len(downloads) == 2
    assert _manifest(out)["cycle"] == "2026-02-01T06:00:00Z"
    assert _payload_stamps(out, "gfs-subset") == {bytes((2, 1))}


def test_fetch_gfs_fullfile_never_carries_new_years_eve_into_new_years_day(
        tmp_path, monkeypatch, offline_index):
    downloads: list[str] = []

    def download(url, destination, **kwargs):
        downloads.append(url)
        destination.write_bytes(_stream(12, _stamp(url)))

    monkeypatch.setattr(gfs_transport, "_download", download)
    monkeypatch.setattr(fetch, "_head_ok", lambda url: True)
    out = tmp_path / "gfs-full"
    common = dict(hours=(0, 3), area=None, out=out,
                  progress=lambda line: None)
    fetch.fetch_gfs_fullfile(cycle=datetime(2025, 12, 31, 6), **common)
    before = _files(out)
    downloads.clear()

    with pytest.raises(ValueError, match="different request"):
        fetch.fetch_gfs_fullfile(cycle=datetime(2026, 1, 1, 6), **common)
    assert downloads == []
    assert _files(out) == before

    fetch.fetch_gfs_fullfile(cycle=datetime(2026, 1, 1, 6), force=True,
                             **common)
    assert len(downloads) == 2
    assert _manifest(out)["cycle"] == "2026-01-01T06:00:00Z"
    assert _payload_stamps(out, "gfs-full-file") == {bytes((1, 1))}


def test_fetch_gfs_fullfile_refuses_a_folder_of_grib_filter_crops(
        tmp_path, monkeypatch, offline_index):
    """Same request, other transport: the two name and verify files apart."""
    downloads: list[str] = []
    monkeypatch.setattr(gfs_transport, "_download",
                        _dated_gfs_download(downloads))
    out = tmp_path / "gfs"
    fetch.fetch_gfs(cycle=datetime(2026, 1, 31, 6), hours=(0,),
                    area=fetch.parse_area(AREA), out=out,
                    progress=lambda line: None)
    with pytest.raises(ValueError, match="nomads-cgi-subset fetch"):
        fetch.fetch_gfs_fullfile(
            cycle=datetime(2026, 1, 31, 6), hours=(0,),
            area=fetch.parse_area(AREA), out=out,
            progress=lambda line: None)


def test_fetch_hrrr_never_publishes_the_previous_days_files_under_a_new_date(
        tmp_path, monkeypatch):
    downloads: list[str] = []
    monkeypatch.setattr(hrrr_transport, "_download_product",
                        _dated_hrrr_product(downloads))
    out = tmp_path / "hrrr"
    common = dict(hours=(0, 1), area=None, out=out,
                  progress=lambda line: None)
    fetch.fetch_hrrr(cycle=datetime(2026, 1, 31, 6), **common)
    before = _files(out)
    downloads.clear()

    with pytest.raises(ValueError, match="different request"):
        fetch.fetch_hrrr(cycle=datetime(2026, 2, 1, 6), **common)
    assert downloads == []
    assert _files(out) == before
    assert _manifest(out)["cycle"] == "2026-01-31T06:00:00Z"

    fetch.fetch_hrrr(cycle=datetime(2026, 2, 1, 6), force=True, **common)
    assert len(downloads) == 4
    assert _manifest(out)["cycle"] == "2026-02-01T06:00:00Z"
    stamps = {(out / entry["name"]).read_bytes()[4:6]
              for entry in _manifest(out)["files"]
              if entry["role"] in ("atmosphere", "soil")}
    assert stamps == {bytes((2, 1))}


# ---------------------------------------------------------------------------
# --wait-for reaches its wait loop; a complete folder needs no host
# ---------------------------------------------------------------------------

def test_a_named_hrrr_cycle_with_wait_for_waits_instead_of_refusing(
        tmp_path, monkeypatch, capsys, python_engine):
    downloads: list[str] = []
    monkeypatch.setattr(hrrr_transport, "_download_product",
                        _dated_hrrr_product(downloads))
    released = {"f02": False}

    def published(url):
        # f000 and f001 are up; f002 appears after the first poll.
        return released["f02"] or "f02." not in url.rsplit("/", 1)[-1]

    _with_probe(monkeypatch, "require_published_cycle", published)
    # --wait-for is the as-posted loop for every source now (A136 L2): the
    # loop asks each hour when it is due and waits by its own clock, and
    # the transport is called for each posted prefix (the HRRR-only wait
    # branch inside fetch_hrrr is no longer reached from the command).
    from woof import fetch_as_posted
    from woof import source_posting as rows

    monkeypatch.setattr(fetch, "_head_answer", published)
    clock = {"now": rows.expected_at("hrrr", datetime(2026, 1, 31, 6), 2)}
    start = clock["now"]

    def pause(seconds):
        clock["now"] += timedelta(seconds=seconds)
        released["f02"] = True

    monkeypatch.setattr(fetch_as_posted, "now_utc", lambda: clock["now"])
    monkeypatch.setattr(fetch_as_posted, "pause", pause)
    out = tmp_path / "hrrr"
    rc = cli.main(["fetch", "--source", "hrrr", "--cycle", "2026-01-31T06",
                   "--hours", "2", "--wait-for",
                   "--wait-timeout-minutes", "1", "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "f002 not posted yet" in captured.out
    assert clock["now"] > start
    assert _manifest(out)["forecast_hours"] == [0, 1, 2]
    assert sorted(path.name for path in (out / "posting").glob("f*.json")) == [
        "f000.json", "f001.json", "f002.json"]


def test_a_partial_folder_still_asks_whether_the_cycle_is_published(
        tmp_path, monkeypatch, capsys, python_engine):
    """Only a COMPLETE folder skips the question; a missing hour asks it."""
    downloads: list[str] = []
    monkeypatch.setattr(hrrr_transport, "_download_product",
                        _dated_hrrr_product(downloads))
    out = tmp_path / "hrrr"
    fetch.fetch_hrrr(cycle=datetime(2026, 1, 31, 6), hours=(0,),
                     area=None, out=out, progress=lambda line: None)
    downloads.clear()
    _with_probe(monkeypatch, "require_published_cycle", lambda url: False)
    rc = cli.main(["fetch", "--whole-cycle", "--source", "hrrr", "--cycle", "2026-01-31T06",
                   "--hours", "1", "--out", str(out)])
    assert rc != 0
    assert "not published" in capsys.readouterr().err
    assert downloads == []


# ---------------------------------------------------------------------------
# Latest is asked of the host the transfer is pinned to
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 2, 1, 7, 30)


def _hosts(nomads_newest: datetime, s3_newest: datetime):
    """The operational server ahead of the archive."""
    def probe(url):
        newest = nomads_newest if "nomads.ncep.noaa.gov" in url else s3_newest
        return _cycle_of(url) <= newest
    return probe


def _parse(tokens):
    return cli.build_parser().parse_args(["fetch", *tokens])


def test_latest_hrrr_pinned_to_s3_takes_the_newest_cycle_s3_holds(
        tmp_path, monkeypatch, python_engine):
    hosts = _hosts(datetime(2026, 2, 1, 6), datetime(2026, 2, 1, 5))
    _with_probe(monkeypatch, "resolve_latest_cycle", hosts, now=_NOW)
    # The same two hosts answer the named cycle's check and the as-posted
    # lead gate, so this test (not marked network) asks no real host.
    _with_probe(monkeypatch, "require_published_cycle", hosts, now=_NOW)
    monkeypatch.setattr(fetch, "_head_answer", hosts)
    handed = []

    def fetch_hrrr(**kwargs):
        handed.append((kwargs["cycle"], kwargs["transport"]))
        raise _Stop

    monkeypatch.setattr(fetch, "fetch_hrrr", fetch_hrrr)
    with pytest.raises(_Stop):
        fetch.fetch_main(_parse(["--source", "hrrr", "--cycle", "latest",
                                 "--hours", "1", "--transport", "s3",
                                 "--out", str(tmp_path / "hrrr")]))
    assert handed == [(datetime(2026, 2, 1, 5), "s3")]


def test_latest_gfs_fullfile_pinned_to_s3_takes_the_newest_cycle_s3_holds(
        tmp_path, monkeypatch, python_engine):
    _with_probe(monkeypatch, "resolve_latest_cycle",
                _hosts(datetime(2026, 2, 1, 6), datetime(2026, 2, 1, 0)),
                now=_NOW)
    handed = []

    def fetch_gfs_fullfile(**kwargs):
        handed.append((kwargs["cycle"], kwargs["transport"]))
        raise _Stop

    monkeypatch.setattr(fetch, "fetch_gfs_fullfile", fetch_gfs_fullfile)
    with pytest.raises(_Stop):
        fetch.fetch_main(_parse(["--source", "gfs", "--mode", "full-file",
                                 "--cycle", "latest", "--hours", "3",
                                 "--transport", "s3",
                                 "--out", str(tmp_path / "gfs")]))
    assert handed == [(datetime(2026, 2, 1, 0), "s3")]


def test_a_named_cycle_is_checked_on_the_pinned_host_only(
        tmp_path, monkeypatch, capsys, python_engine):
    """NOMADS having 06Z does not let an S3-pinned fetch start on it."""
    _with_probe(monkeypatch, "require_published_cycle",
                _hosts(datetime(2026, 2, 1, 6), datetime(2026, 2, 1, 5)),
                now=_NOW)
    _with_probe(monkeypatch, "resolve_latest_cycle",
                _hosts(datetime(2026, 2, 1, 6), datetime(2026, 2, 1, 5)),
                now=_NOW)
    monkeypatch.setattr(fetch, "fetch_hrrr",
                        lambda **kwargs: pytest.fail("transfer started"))
    rc = cli.main(["fetch", "--whole-cycle", "--source", "hrrr", "--cycle", "2026-02-01T06",
                   "--hours", "1", "--transport", "s3",
                   "--out", str(tmp_path / "hrrr")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "not published" in err and "2026-02-01T05Z" in err


# ---------------------------------------------------------------------------
# A run plan resolves latest from what the fetch parser read
# ---------------------------------------------------------------------------

@pytest.fixture
def recorded_latest(monkeypatch):
    asked = []

    def resolver(source, last_hour, **options):
        asked.append((source, last_hour, options.get("transport")))
        return datetime(2026, 2, 1, 0)

    monkeypatch.setattr(fetch, "resolve_latest_cycle", resolver)
    return asked


def test_equals_form_fetch_arguments_resolve_their_own_source_and_window(
        tmp_path, recorded_latest):
    arguments = ["--source=gfs", "--cycle", "latest", "--hours=24",
                 "--forecast-start-hour=3", "--area", AREA,
                 "--out", str(tmp_path / "gfs")]
    resolved, resolutions, _warnings = runplan.resolve_fetch_cycle(arguments)
    assert [row[:2] for row in recorded_latest] == [("gfs", 27)]
    assert resolved[resolved.index("--cycle") + 1] == "2026-02-01T00"
    assert resolutions[0]["value"] == "2026-02-01T00"


def test_an_equals_form_latest_cycle_is_resolved_and_written_back(
        tmp_path, recorded_latest):
    arguments = ["--source", "hrrr", "--cycle=latest", "--hours", "6",
                 "--transport=s3", "--out", str(tmp_path / "hrrr")]
    resolved, resolutions, _warnings = runplan.resolve_fetch_cycle(arguments)
    assert recorded_latest == [("hrrr", 6, "s3")]
    assert "--cycle=2026-02-01T00" in resolved
    assert not any(token.lower() in ("latest", "--cycle=latest")
                   for token in resolved)
    assert resolutions and resolutions[0]["basis"] == "resolved_latest"
    parsed = cli.build_parser().parse_args(["fetch", *resolved])
    assert parsed.cycle == "2026-02-01T00" and parsed.transport == "s3"


def test_a_repeated_cycle_option_leaves_no_latest_behind(
        tmp_path, recorded_latest):
    """argparse takes the last one; every spelling is rewritten."""
    arguments = ["--source", "gfs", "--cycle", "2026-01-01T00",
                 "--cycle=latest", "--hours", "6", "--area", AREA,
                 "--out", str(tmp_path / "gfs")]
    resolved, _resolutions, _warnings = runplan.resolve_fetch_cycle(arguments)
    assert recorded_latest[0][:2] == ("gfs", 6)
    parsed = cli.build_parser().parse_args(["fetch", *resolved])
    assert parsed.cycle == "2026-02-01T00"


def test_a_named_cycle_is_left_exactly_as_written(tmp_path, recorded_latest):
    arguments = ["--source=gfs", "--cycle=2026-01-31T06", "--hours=6",
                 "--area", AREA, "--out", str(tmp_path / "gfs")]
    resolved, resolutions, warnings = runplan.resolve_fetch_cycle(arguments)
    assert resolved == arguments
    assert (resolutions, warnings, recorded_latest) == ([], [], [])


# ---------------------------------------------------------------------------
# A relative file in a plan means a file beside the plan
# ---------------------------------------------------------------------------

def _polygon_plan(folder: Path) -> Path:
    from woof.gui.api import region_polygon

    folder.mkdir(parents=True)
    (folder / "region.geojson").write_text(
        json.dumps(region_polygon(39.0, -98.0, 300, 300)), encoding="utf-8")
    plan_path = folder / "plan.json"
    plan_path.write_text(json.dumps({
        "schema": runplan.PLAN_SCHEMA, "name": "relative-polygon",
        "route": "experiment",
        "config": {"intent": {"polygon": "region.geojson", "source": "era5",
                              "cycle": "2024-05-03T12", "hours": 1,
                              "vram_gib": 24}},
        "output_root": "run"}), encoding="utf-8")
    return plan_path


def test_run_plan_resolve_reads_a_relative_polygon_beside_the_plan(
        tmp_path, monkeypatch, capsys):
    plan_path = _polygon_plan(tmp_path / "portable plan")
    # A same-named file where the plan is launched from is not the plan's.
    (tmp_path / "region.geojson").write_text("not a region",
                                             encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rc = runplan.run_plan_main(cli.build_parser().parse_args(
        ["run-plan", "--resolve", str(plan_path)]))
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert json.loads(captured.out)["generated_config"]


def test_every_file_valued_intent_key_is_made_absolute_against_the_plan(
        tmp_path):
    plan = runplan.build_plan({
        "schema": runplan.PLAN_SCHEMA, "name": "paths", "route": "experiment",
        "config": {"intent": {
            "point": "39,-98", "source": "era5", "cycle": "2024-05-03T12",
            "hours": 1, "vram_gib": 24, "data_dir": "data",
            "forcing": ["forcing/*.grib", str(tmp_path / "abs.grib")],
            "vtable": "Vtable.custom", "geog_root": "geog"}},
    }, source="plan.json", base_dir=tmp_path, sha256="0" * 64)
    intent = plan.config_intent
    assert intent["data_dir"] == str(tmp_path / "data")
    assert intent["forcing"] == [str(tmp_path / "forcing" / "*.grib"),
                                 str(tmp_path / "abs.grib")]
    assert intent["vtable"] == str(tmp_path / "Vtable.custom")
    assert intent["geog_root"] == str(tmp_path / "geog")
    # Values that are not files are untouched.
    assert intent["point"] == "39,-98" and intent["cycle"] == "2024-05-03T12"


def test_the_path_keys_are_wizard_flags_that_take_a_path():
    import argparse

    commands = next(action for action in cli.build_parser()._actions
                    if isinstance(action, argparse._SubParsersAction))
    parser = commands.choices["domain"]
    by_flag = {flag: action for action in parser._actions
               for flag in action.option_strings}
    for key in runplan._INTENT_PATH_KEYS:
        action = by_flag[runplan._INTENT_FLAGS[key]]
        assert action.metavar in ("GEOJSON", "DIR", "GRIB") or \
            action.type is Path, key


# ---------------------------------------------------------------------------
# --transport auto is the unpinned default, on every HRRR route
# ---------------------------------------------------------------------------

def _every_host_answers(monkeypatch):
    """Every probe, on every host, says the object is there.

    That includes the as-posted loop's lead gate and the native prefix's
    host choice, which ask through ``_head_answer``: without it these
    tests (not marked network) asked the real hosts on every run.
    """
    monkeypatch.setattr(fetch, "_head_ok", lambda url: True)
    monkeypatch.setattr(fetch, "_head_answer", lambda url: True)
    for name in ("resolve_latest_cycle", "require_published_cycle"):
        _with_probe(monkeypatch, name, lambda url: True, now=_NOW)


def _handed_hrrr(monkeypatch) -> list:
    handed = []

    def fetch_hrrr(**kwargs):
        # The command no longer passes ``wait``: the as-posted loop waits
        # for each hour before the transport is called (A136 L2).
        handed.append((kwargs["cycle"], kwargs["transport"],
                       kwargs.get("wait", False)))
        raise _Stop

    monkeypatch.setattr(fetch, "fetch_hrrr", fetch_hrrr)
    return handed


def test_latest_hrrr_with_transport_auto_resolves_and_starts(
        tmp_path, monkeypatch, python_engine):
    _every_host_answers(monkeypatch)
    handed = _handed_hrrr(monkeypatch)
    with pytest.raises(_Stop):
        fetch.fetch_main(_parse(["--source", "hrrr", "--cycle", "latest",
                                 "--hours", "1", "--transport", "auto",
                                 "--out", str(tmp_path / "hrrr")]))
    assert len(handed) == 1
    cycle, transport, wait = handed[0]
    assert cycle.date() == _NOW.date() and not wait
    assert transport in ("nomads", "s3")


def test_a_named_hrrr_cycle_with_transport_auto_starts(
        tmp_path, monkeypatch, python_engine):
    _every_host_answers(monkeypatch)
    handed = _handed_hrrr(monkeypatch)
    with pytest.raises(_Stop):
        fetch.fetch_main(_parse(["--source", "hrrr",
                                 "--cycle", "2026-02-01T06", "--hours", "1",
                                 "--transport", "auto",
                                 "--out", str(tmp_path / "hrrr")]))
    assert [(cycle, transport in ("nomads", "s3"), wait)
            for cycle, transport, wait in handed] == [
        (datetime(2026, 2, 1, 6), True, False)]


@pytest.mark.parametrize("source", ["gfs", "gdas"])
def test_a_named_whole_file_cycle_with_transport_auto_starts_unpinned(
        tmp_path, monkeypatch, python_engine, source):
    """``auto`` reaches the whole-file transfer as no pin, not a host name.

    The publication checks already read it as unpinned; the transfer and
    the engine line named it as a host and refused the request only after
    the publication requests had gone out.
    """
    _every_host_answers(monkeypatch)
    handed = []

    def fetch_gfs_fullfile(**kwargs):
        handed.append((kwargs["cycle"], kwargs["transport"]))
        raise _Stop

    monkeypatch.setattr(fetch, "fetch_gfs_fullfile", fetch_gfs_fullfile)
    with pytest.raises(_Stop):
        fetch.fetch_main(_parse(["--source", source, "--mode", "full-file",
                                 "--cycle", "2026-02-01T06", "--hours", "3",
                                 "--transport", "auto",
                                 "--out", str(tmp_path / source)]))
    assert handed == [(datetime(2026, 2, 1, 6), None)]


def test_the_whole_file_library_door_reads_transport_auto_as_no_pin(
        tmp_path, monkeypatch):
    seen = []

    def locked(**kwargs):
        seen.append(kwargs["pinned_host"])
        raise _Stop

    monkeypatch.setattr(fetch, "_fetch_gfs_fullfile_locked", locked)
    for transport in ("auto", None, "s3"):
        with pytest.raises(_Stop):
            fetch.fetch_gfs_fullfile(cycle=datetime(2026, 2, 1, 6), hours=(0,),
                                     area=None, out=tmp_path / "gfs",
                                     transport=transport,
                                     progress=lambda line: None)
    assert seen == [None, None, "s3"]


def test_latest_hrrr_wait_for_with_transport_auto_waits_on_both_hosts(
        tmp_path, monkeypatch, python_engine):
    _every_host_answers(monkeypatch)
    handed = _handed_hrrr(monkeypatch)
    with pytest.raises(_Stop):
        fetch.fetch_main(_parse(["--source", "hrrr", "--cycle", "latest",
                                 "--hours", "2", "--wait-for",
                                 "--transport", "auto",
                                 "--out", str(tmp_path / "hrrr")]))
    # Every host answers, so the window is posted whole and moves in one
    # call on the resolved host; the waiting (when there is any) is the
    # as-posted loop's, over the whole serving ladder (A136 L2), not a
    # per-file wait inside the transport.
    assert len(handed) == 1
    assert handed[0][1] in ("nomads", "s3") and handed[0][2] is False


@pytest.mark.parametrize("pinned, option", [("s3", []), ("nomads", ["--transport", "auto"])], ids=["default", "auto"])
def test_a_complete_hrrr_folder_with_transport_auto_keeps_its_host(
        tmp_path, monkeypatch, capsys, python_engine, pinned, option):
    downloads: list[str] = []
    monkeypatch.setattr(hrrr_transport, "_download_product",
                        _dated_hrrr_product(downloads))
    out = tmp_path / "hrrr"
    fetch.fetch_hrrr(cycle=datetime(2026, 1, 31, 6), hours=(0, 1),
                     area=None, out=out, transport=pinned,
                     progress=lambda line: None)
    downloads.clear()
    monkeypatch.setattr(fetch, "_head_ok", lambda url: False)
    _with_probe(monkeypatch, "require_published_cycle", lambda url: False)
    rc = cli.main(["fetch", "--source", "hrrr", "--cycle", "2026-01-31T06",
                   "--hours", "1", *option,
                   "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert downloads == []
    assert {entry["transport"] for entry in _manifest(out)["files"]
            if entry["role"] in ("atmosphere", "soil")} == {pinned}
    assert "already in" in captured.out
    assert _manifest(out)["forecast_hours"] == [0, 1]


def test_run_plan_resolves_latest_with_transport_auto(tmp_path, monkeypatch):
    _with_probe(monkeypatch, "resolve_latest_cycle", lambda url: True,
                now=_NOW)
    arguments = ["--source", "hrrr", "--cycle", "latest", "--hours", "1",
                 "--transport", "auto", "--out", str(tmp_path / "hrrr")]
    resolved, resolutions, _warnings = runplan.resolve_fetch_cycle(arguments)
    parsed = cli.build_parser().parse_args(["fetch", *resolved])
    assert parsed.cycle == resolutions[0]["value"]
    assert parsed.cycle.startswith(f"{_NOW:%Y-%m-%d}")
    assert parsed.transport == "auto"


def test_run_plan_refuses_a_host_the_source_does_not_publish_on(tmp_path):
    """The resolver's refusal is the plan's to fix, and says so."""
    arguments = ["--source", "hrrr", "--cycle", "latest", "--hours", "1",
                 "--transport", "aws", "--out", str(tmp_path / "hrrr")]
    with pytest.raises(runplan.PlanError, match="publishes on nomads, s3"):
        runplan.resolve_fetch_cycle(arguments)


# ---------------------------------------------------------------------------
# A pinned host past its retention says so, and names the host that keeps it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", [(), ("--whole-cycle",)])
def test_an_old_cycle_pinned_to_nomads_names_retention_and_s3(
        tmp_path, monkeypatch, capsys, python_engine, rule):
    # As posted (the default) and whole-cycle alike: a pinned host is the
    # whole ladder, so a cycle past its retention can never start there.
    _with_probe(monkeypatch, "require_published_cycle",
                lambda url: "nomads.ncep.noaa.gov" not in url)
    monkeypatch.setattr(fetch, "fetch_hrrr",
                        lambda **kwargs: pytest.fail("transfer started"))
    rc = cli.main(["fetch", *rule, "--source", "hrrr", "--cycle", "2026-01-10T06",
                   "--hours", "1", "--transport", "nomads",
                   "--out", str(tmp_path / "hrrr")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "keeps only about the newest 48 h" in err
    assert "s3 still keeps it" in err and "--transport s3" in err
    assert "yet" not in err


def test_a_young_cycle_pinned_to_nomads_is_still_not_published_yet():
    """Inside the retention window a miss is publication, not retention."""
    refusal = fetch.cycle_publication_refusal(
        "hrrr", datetime(2026, 2, 1, 6), 1, now=_NOW,
        probe=lambda url: "nomads.ncep.noaa.gov" not in url,
        transport="nomads")
    assert "not published through f001 yet" in refusal
    assert "keeps only" not in refusal


# ---------------------------------------------------------------------------
# A finished folder reads no index, and its receipt keeps the serving host
# ---------------------------------------------------------------------------

def _recent_cycle() -> datetime:
    """A GFS cycle young enough that every host still keeps it."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    newest = now.replace(hour=now.hour - now.hour % 6, minute=0, second=0,
                         microsecond=0)
    return newest - timedelta(hours=12)


def _no_index(monkeypatch) -> None:
    def read(*args, **kwargs):
        raise AssertionError("a finished folder read the live index")

    monkeypatch.setattr(fetch, "gfs_live_index", read)
    monkeypatch.setattr(fetch, "_gfs_index_record_count", read)


def test_a_complete_recent_gfs_folder_is_reused_without_asking_anyone(
        tmp_path, monkeypatch, capsys, offline_index):
    """The recent cycle takes the publication check, not the archive switch."""
    cycle = _recent_cycle()
    assert not fetch.archive_only_cycle("gfs", cycle)
    downloads: list[str] = []
    monkeypatch.setattr(gfs_transport, "_download",
                        _dated_gfs_download(downloads))
    out = tmp_path / "gfs"
    fetch.fetch_gfs(cycle=cycle, hours=(0, 3), area=fetch.parse_area(AREA),
                    out=out, progress=lambda line: None)
    before = _files(out)

    def unreachable(*args, **kwargs):
        raise AssertionError("a complete folder asked the provider for bytes")

    monkeypatch.setattr(gfs_transport, "_download", unreachable)
    monkeypatch.setattr(fetch, "_head_ok", lambda url: False)
    _with_probe(monkeypatch, "require_published_cycle", lambda url: False)
    _no_index(monkeypatch)
    rc = cli.main(["fetch", "--source", "gfs",
                   "--cycle", f"{cycle:%Y-%m-%dT%H}", "--hours", "3",
                   "--area", AREA, "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "already in" in captured.out
    assert {name: body for name, body in _files(out).items()
            if name.endswith(".grib2")} == {
        name: body for name, body in before.items()
        if name.endswith(".grib2")}
    assert _manifest(out)["record_bars"][0]["expected"] == \
        fetch.GFS_SUBSET_RECORD_COUNT
    assert _manifest(out)["mode"] == "nomads-cgi-subset"


def test_a_fullfile_rerun_reads_no_index_and_keeps_the_serving_host(
        tmp_path, monkeypatch, capsys, offline_index):
    cycle = _recent_cycle()
    downloads: list[str] = []

    def download(url, destination, **kwargs):
        downloads.append(url)
        destination.write_bytes(_stream(12, _stamp(url)))

    monkeypatch.setattr(gfs_transport, "_download", download)
    # Every host answers, so the archive serves for throughput even
    # though the operational server heads the ladder for this cycle.
    monkeypatch.setattr(fetch, "_head_ok", lambda url: True)
    out = tmp_path / "gfs-full"
    common = dict(cycle=cycle, hours=(0, 3), area=None, out=out,
                  progress=lambda line: None)
    fetch.fetch_gfs_fullfile(**common)
    first = _manifest(out)
    assert first["transport"] == "s3"
    assert {entry["endpoint"] for entry in first["files"]
            if entry["role"] == "gfs-full-file"} == {"s3"}
    assert downloads and all("nomads" not in url for url in downloads)

    downloads.clear()

    def asked(url):
        raise AssertionError(f"a finished folder asked a host about {url}")

    monkeypatch.setattr(fetch, "_head_ok", asked)
    _no_index(monkeypatch)
    capsys.readouterr()
    fetch.fetch_gfs_fullfile(**common)
    again = _manifest(out)
    assert downloads == []
    # The progress lines name the host the receipt names, too.
    assert "nomads" not in capsys.readouterr().err
    payload = [entry for entry in again["files"]
               if entry["role"] == "gfs-full-file"]
    assert {entry["endpoint"] for entry in payload} == {"s3"}
    assert [entry["url"] for entry in payload] == [
        entry["url"] for entry in first["files"]
        if entry["role"] == "gfs-full-file"]
    assert not any(entry["downloaded"] for entry in payload)
    assert again["transport"] == "s3"
    assert again["endpoints"]["served"] == ["s3"]


def test_a_skipped_hrrr_file_keeps_the_host_that_served_it(
        tmp_path, monkeypatch, python_engine):
    downloads: list[str] = []
    monkeypatch.setattr(hrrr_transport, "_download_product",
                        _dated_hrrr_product(downloads))
    out = tmp_path / "hrrr"
    fetch.fetch_hrrr(cycle=datetime(2026, 1, 31, 6), hours=(0,), area=None,
                     out=out, transport="nomads", progress=lambda line: None)
    # The window grows by an hour, fetched from the other host.
    fetch.fetch_hrrr(cycle=datetime(2026, 1, 31, 6), hours=(0, 1),
                     area=None, out=out, transport="s3",
                     progress=lambda line: None)
    hosts = {(entry["forecast_hour"], entry["transport"])
             for entry in _manifest(out)["files"]
             if entry["role"] in ("atmosphere", "soil")}
    assert hosts == {(0, "nomads"), (1, "s3")}


# ---------------------------------------------------------------------------
# A finished folder is reused only while its bytes are the recorded ones
# ---------------------------------------------------------------------------

def _retiring_hrrr_hosts(downloads: list, retired: set):
    """HRRR hosts that serve until they are retired, then answer 404.

    The operational server lets a cycle go after its retention window;
    ``retired`` holds the host names that no longer keep the cycle.
    """
    from urllib.error import HTTPError

    serve = _dated_hrrr_product(downloads)

    def product(request, *, workers, retries, expected_count=-1):
        host = ("nomads" if "nomads.ncep.noaa.gov" in request.url
                else "s3")
        if host in retired:
            raise HTTPError(request.url + ".idx", 404, "Not Found", {}, None)
        return serve(request, workers=workers, retries=retries,
                     expected_count=expected_count)

    return product


def _damage_in_place(path: Path) -> None:
    """Change one stamp octet, keeping the size and every GRIB envelope."""
    body = bytearray(path.read_bytes())
    body[4] ^= 0xFF
    path.write_bytes(bytes(body))


def test_a_damaged_file_is_fetched_again_from_a_host_that_still_has_it(
        tmp_path, monkeypatch, capsys, python_engine):
    """The receipt's host is kept only while nothing needs to move.

    The folder came from the operational server; by the re-run that
    server has let the cycle go and the archive still keeps it.  One
    file damaged in place at its own size used to pass the folder as
    complete, skip the publication check and host choice, and fetch the
    file again from the operational server: HTTP 404, and on the Python
    transport a raw traceback.
    """
    downloads: list[str] = []
    retired: set[str] = set()
    monkeypatch.setattr(hrrr_transport, "_download_product",
                        _retiring_hrrr_hosts(downloads, retired))
    cycle = datetime(2026, 1, 31, 18)
    out = tmp_path / "hrrr"
    fetch.fetch_hrrr(cycle=cycle, hours=(0, 1), area=None, out=out,
                     transport="nomads", progress=lambda line: None)
    first = _manifest(out)
    damaged = out / "hrrr.t18z.wrfnatf01.grib2"
    _damage_in_place(damaged)
    assert damaged.stat().st_size == next(
        entry["bytes"] for entry in first["files"]
        if entry["name"] == damaged.name)
    downloads.clear()
    retired.add("nomads")

    def keeps(url):
        return "nomads.ncep.noaa.gov" not in url

    monkeypatch.setattr(fetch, "_head_ok", keeps)
    # The as-posted lead gate and the native prefix's host choice ask
    # through _head_answer; the same hosts answer it.
    monkeypatch.setattr(fetch, "_head_answer", keeps)
    _with_probe(monkeypatch, "require_published_cycle", keeps)
    rc = cli.main(["fetch", "--source", "hrrr", "--cycle", "2026-01-31T18",
                   "--hours", "1", "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "Traceback" not in captured.err
    assert "no longer holds the bytes its receipt recorded" in captured.out
    assert len(downloads) == 1
    assert "noaa-hrrr-bdp-pds" in downloads[0]
    assert "wrfnatf01" in downloads[0]
    again = {entry["name"]: entry for entry in _manifest(out)["files"]}
    assert again[damaged.name]["transport"] == "s3"
    assert again[damaged.name]["downloaded"] is True
    assert {entry["transport"] for name, entry in again.items()
            if entry["role"] in ("atmosphere", "soil")
            and name != damaged.name} == {"nomads"}
    assert damaged.read_bytes()[4:6] == bytes((1, 31))


def test_an_intact_folder_still_keeps_its_host_after_the_host_lets_go(
        tmp_path, monkeypatch, capsys, python_engine):
    """Nothing damaged, nothing moved: no host is asked at all."""
    downloads: list[str] = []
    retired: set[str] = set()
    monkeypatch.setattr(hrrr_transport, "_download_product",
                        _retiring_hrrr_hosts(downloads, retired))
    out = tmp_path / "hrrr"
    fetch.fetch_hrrr(cycle=datetime(2026, 1, 31, 18), hours=(0, 1),
                     area=None, out=out, transport="nomads",
                     progress=lambda line: None)
    downloads.clear()
    retired.update(("nomads", "s3"))

    def asked(url):
        raise AssertionError(f"an intact folder asked a host about {url}")

    monkeypatch.setattr(fetch, "_head_ok", asked)
    _with_probe(monkeypatch, "require_published_cycle", asked)
    rc = cli.main(["fetch", "--source", "hrrr", "--cycle", "2026-01-31T18",
                   "--hours", "1", "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert downloads == []
    assert {entry["transport"] for entry in _manifest(out)["files"]
            if entry["role"] in ("atmosphere", "soil")} == {"nomads"}


def test_an_hrrr_host_that_does_not_serve_a_product_is_refused_in_words(
        tmp_path, monkeypatch, capsys, python_engine):
    """A 404 from the Python transport is a sentence, not a traceback."""
    downloads: list[str] = []
    monkeypatch.setattr(hrrr_transport, "_download_product",
                        _retiring_hrrr_hosts(downloads, {"nomads", "s3"}))
    # Both hosts said the cycle was there a moment ago.
    _every_host_answers(monkeypatch)
    rc = cli.main(["fetch", "--whole-cycle", "--source", "hrrr", "--cycle", "2026-02-01T06",
                   "--hours", "1", "--transport", "s3",
                   "--out", str(tmp_path / "hrrr")])
    err = capsys.readouterr().err
    assert rc == 2, err
    assert "Traceback" not in err
    assert "s3 answered HTTP 404" in err
    # Both hours 404 on s3 here and the pool fetches f00 and f01 at once,
    # so the sentence names whichever failed first: f00 alone, f01 under
    # load (seen on test host 4's full CPU leg).  Either names this cycle.
    assert re.search(r"HRRR cycle 2026-02-01T06Z f0[01] ", err), err


def test_a_404_past_the_hosts_retention_names_the_host_that_keeps_it():
    from urllib.error import HTTPError

    cycle = datetime(2026, 1, 31, 18)
    error = HTTPError("https://nomads.ncep.noaa.gov/x", 404, "Not Found",
                      {}, None)
    said = fetch.hrrr_reach_refusal("nomads", cycle, 1, "atmosphere", error,
                                    now=cycle + timedelta(hours=96))
    assert "keeps only about the newest 48 h" in said
    assert "96 h old" in said
    assert "s3 still keeps it" in said and "--transport s3" in said
    young = fetch.hrrr_reach_refusal("nomads", cycle, 1, "atmosphere", error,
                                     now=cycle + timedelta(hours=3))
    assert "keeps only" not in young and "HTTP 404" in young


# ---------------------------------------------------------------------------
# A file that is only checked is not announced as a download from a host
# ---------------------------------------------------------------------------

def test_an_offline_gfs_rerun_names_no_host_for_files_it_only_checks(
        tmp_path, monkeypatch, capsys, offline_index):
    downloads: list[str] = []
    monkeypatch.setattr(gfs_transport, "_download",
                        _dated_gfs_download(downloads))
    out = tmp_path / "gfs"
    fetch.fetch_gfs(cycle=datetime(2026, 1, 31, 18), hours=(0, 3),
                    area=fetch.parse_area(AREA), out=out,
                    progress=lambda line: None)
    capsys.readouterr()
    monkeypatch.setattr(fetch, "_head_ok", lambda url: False)
    _with_probe(monkeypatch, "require_published_cycle", lambda url: False)
    rc = cli.main(["fetch", "--source", "gfs", "--cycle", "2026-01-31T18",
                   "--hours", "3", "--area", AREA, "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "without asking the provider" in captured.out
    assert "nomads" not in captured.err
    assert "starting (" not in captured.err
    assert captured.err.count("is already on disk; checking it here") == 2


def _never_asked(monkeypatch) -> None:
    """Every host question fails the test: nothing may reach a provider."""

    def asked(*args, **kwargs):
        raise AssertionError(f"a host was asked: {args!r}")

    monkeypatch.setattr(fetch, "_head_ok", asked)
    _with_probe(monkeypatch, "require_published_cycle", asked)
    monkeypatch.setattr(fetch, "gfs_live_index", asked)
    monkeypatch.setattr(fetch, "_gfs_index_record_count", asked)


@pytest.mark.parametrize("cycle", [datetime(2026, 1, 31, 6), None],
                         ids=["archive-only", "recent"])
def test_a_damaged_gfs_crop_is_refused_by_name_before_any_host_is_asked(
        tmp_path, monkeypatch, capsys, offline_index, cycle):
    """The GFS routes refuse a file whose bytes moved; they never refetch it.

    So a damaged crop is refused at the reuse check with the refusal the
    crop route gives it.  Sending the fetch on to the publication check
    instead could not change that answer: offline it was refused as
    "not published yet", and an old cycle was switched to the archive's
    whole objects and refused for the mode the folder recorded, neither
    refusal naming the damaged file.  A finished folder says it is
    checking its files once, not once per check.
    """
    cycle = cycle or _recent_cycle()
    downloads: list[str] = []
    monkeypatch.setattr(gfs_transport, "_download",
                        _dated_gfs_download(downloads))
    out = tmp_path / "gfs"
    fetch.fetch_gfs(cycle=cycle, hours=(0, 3), area=fetch.parse_area(AREA),
                    out=out, progress=lambda line: None)
    capsys.readouterr()
    downloads.clear()
    _never_asked(monkeypatch)
    argv = ["fetch", "--source", "gfs", "--cycle", f"{cycle:%Y-%m-%dT%H}",
            "--hours", "3", "--area", AREA, "--out", str(out)]
    checking = "checking them here without asking the provider"

    assert cli.main(argv) == 0
    captured = capsys.readouterr()
    assert captured.out.count(checking) == 1, captured.out

    damaged = out / next(entry["name"] for entry in _manifest(out)["files"]
                         if entry.get("forecast_hour") == 3)
    _damage_in_place(damaged)
    rc = cli.main(argv)
    captured = capsys.readouterr()
    assert rc == 2, captured.err
    assert "Traceback" not in captured.err
    assert (f"existing {damaged.name} does not match the sha256 recorded"
            in captured.err)
    assert "--force-refetch" in captured.err
    assert "not published" not in captured.err
    assert "--mode full-file" not in captured.out
    assert captured.out.count(checking) == 1, captured.out
    assert downloads == []


def test_a_library_gfs_fetch_refuses_a_damaged_file_before_reading_the_index(
        tmp_path, monkeypatch, offline_index):
    downloads: list[str] = []
    monkeypatch.setattr(gfs_transport, "_download",
                        _dated_gfs_download(downloads))
    out = tmp_path / "gfs"
    cycle = datetime(2026, 1, 31, 6)
    fetch.fetch_gfs(cycle=cycle, hours=(0, 3), area=fetch.parse_area(AREA),
                    out=out, progress=lambda line: None)
    damaged = out / next(entry["name"] for entry in _manifest(out)["files"]
                         if entry.get("forecast_hour") == 0)
    _damage_in_place(damaged)
    downloads.clear()
    _never_asked(monkeypatch)
    with pytest.raises(ValueError, match="does not match the sha256"):
        fetch.fetch_gfs(cycle=cycle, hours=(0, 3),
                        area=fetch.parse_area(AREA), out=out,
                        progress=lambda line: None)
    assert downloads == []


def _full_file_download(downloads: list):
    def download(url, destination, **kwargs):
        downloads.append(url)
        destination.write_bytes(_stream(12, _stamp(url)))
    return download


@pytest.mark.parametrize("route", ["subset", "full-file"])
def test_a_refused_damaged_file_stays_refused_on_the_next_run(
        tmp_path, monkeypatch, offline_index, route):
    """A refusal leaves the receipt that binds the damaged file as it was.

    Extending a finished window by one hour is not a finished request,
    so the damaged earlier hour met the route's own check.  The route
    had already published the receipt again with the verified hours
    before it, and without the damaged one, so the refusal erased the
    only record of the file's bytes: the next run took the damaged file
    as its own and published it.
    """
    downloads: list[str] = []
    if route == "subset":
        monkeypatch.setattr(gfs_transport, "_download",
                            _dated_gfs_download(downloads))
        door = functools.partial(fetch.fetch_gfs,
                                 area=fetch.parse_area(AREA))
    else:
        monkeypatch.setattr(gfs_transport, "_download",
                            _full_file_download(downloads))
        monkeypatch.setattr(fetch, "_head_ok", lambda url: True)
        door = functools.partial(fetch.fetch_gfs_fullfile, area=None)
    out = tmp_path / "gfs"
    cycle = datetime(2026, 1, 31, 6)
    door(cycle=cycle, hours=(0, 3), out=out, progress=lambda line: None)
    before = _manifest(out)
    damaged = out / next(entry["name"] for entry in before["files"]
                         if entry.get("forecast_hour") == 3)
    _damage_in_place(damaged)
    downloads.clear()
    for _attempt in range(2):
        with pytest.raises(ValueError, match=(
                f"existing {re.escape(damaged.name)} does not match the "
                "sha256")):
            door(cycle=cycle, hours=(0, 3, 6), out=out,
                 progress=lambda line: None)
        assert _manifest(out) == before
    assert downloads == []


def test_a_whole_object_that_is_no_longer_grib_is_refused_for_that(
        tmp_path, monkeypatch, offline_index):
    """A broken envelope is named as one, and the refusal stays a refusal."""
    downloads: list[str] = []
    monkeypatch.setattr(gfs_transport, "_download",
                        _full_file_download(downloads))
    monkeypatch.setattr(fetch, "_head_ok", lambda url: True)
    out = tmp_path / "gfs-full"
    common = dict(cycle=datetime(2026, 1, 31, 6), hours=(0, 3), area=None,
                  out=out, progress=lambda line: None)
    fetch.fetch_gfs_fullfile(**common)
    before = _manifest(out)
    damaged = out / next(entry["name"] for entry in before["files"]
                         if entry.get("forecast_hour") == 3)
    damaged.write_bytes(damaged.read_bytes() + b"junk")
    downloads.clear()
    for _attempt in range(2):
        with pytest.raises(ValueError, match="GRIB indicator") as refused:
            fetch.fetch_gfs_fullfile(**common)
        said = str(refused.value)
        assert said.startswith(f"existing {damaged.name} is no longer a "
                               "whole GRIB2 file")
        assert "--force-refetch" in said
        assert str(out) not in said
        assert _manifest(out) == before
    assert downloads == []


def test_an_offline_hrrr_rerun_names_no_host_for_files_it_only_checks(
        tmp_path, monkeypatch, capsys, python_engine):
    downloads: list[str] = []
    monkeypatch.setattr(hrrr_transport, "_download_product",
                        _dated_hrrr_product(downloads))
    out = tmp_path / "hrrr"
    fetch.fetch_hrrr(cycle=datetime(2026, 1, 31, 18), hours=(0, 1),
                     area=None, out=out, progress=lambda line: None)
    capsys.readouterr()
    monkeypatch.setattr(fetch, "_head_ok", lambda url: False)
    _with_probe(monkeypatch, "require_published_cycle", lambda url: False)
    rc = cli.main(["fetch", "--source", "hrrr", "--cycle", "2026-01-31T18",
                   "--hours", "1", "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "starting (" not in captured.err
    assert "amazonaws" not in captured.err
    assert captured.err.count("is already on disk; checking it here") == 4


# ---------------------------------------------------------------------------
# A finished folder's files are read once per re-run, never trusted stale
# ---------------------------------------------------------------------------

def test_an_existing_files_digest_is_read_once_and_never_outlives_its_bytes(
        tmp_path, monkeypatch):
    import os
    import time as clock

    reads: list[str] = []
    real_sha256 = fetch.sha256_file

    def counted(path):
        reads.append(Path(path).name)
        return real_sha256(path)

    monkeypatch.setattr(fetch, "sha256_file", counted)
    path = tmp_path / "hrrr.t18z.wrfnatf00.grib2"
    path.write_bytes(_stream(3, b"\x01\x1f"))
    first = fetch.existing_file_digest(path)
    assert fetch.existing_file_digest(path) == first
    # Just written: it can still change inside its timestamps' tick, so
    # it is read every time.
    assert reads == [path.name, path.name]

    later = clock.time_ns() + 3_000_000_000
    monkeypatch.setattr(fetch, "_digest_clock_ns", lambda: later)
    reads.clear()
    assert fetch.existing_file_digest(path) == first
    assert fetch.existing_file_digest(path) == first
    assert reads == [path.name]

    # The same size, other bytes, a later modification time: a new
    # question with a new answer.
    path.write_bytes(_stream(3, b"\x01\x1e"))
    stamp = path.stat()
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))
    changed = fetch.existing_file_digest(path)
    assert changed != first and changed == real_sha256(path)

    # A file replaced by another one is a new question too.
    other = tmp_path / "incoming"
    other.write_bytes(_stream(3, b"\x01\x1f"))
    os.replace(other, path)
    assert fetch.existing_file_digest(path) == first
