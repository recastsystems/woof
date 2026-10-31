"""``woof fetch`` gates: URL/cycle/area resolution, manifests, ERA5.

Everything here is CPU + mocked transport.  The two transport seams the
fetch front door reuses (``tools.download_gfs_native_subset._download``
and ``tools.download_hrrr_native_subset._download_product``) are
monkeypatched to publish synthetic-but-envelope-valid GRIB payloads, so
these tests pin URL construction, cycle resolution, record-count
verification, resumability, and manifest correctness without a network.
One optional live smoke (``WOOF_NETWORK_TESTS=1``) touches a single
small ``.idx`` object.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
from urllib.parse import parse_qsl, urlsplit

import pytest

import woof.cli as cli
import woof.fetch as fetch
from tools import download_gfs_native_subset as gfs_transport
from tools import download_hrrr_native_subset as hrrr_transport


#: The morning these fixtures' cycles were written against.  Every test
#: here names late-July 2026 cycles; the tests that care about a cycle's
#: age pass their own ``now``, and the rest assume the cycle is still on
#: the grib-filter host.
FIXTURE_NOW = datetime(2026, 7, 28, 8)


@pytest.fixture(autouse=True)
def _cycles_keep_their_age(monkeypatch):
    """A cycle's age is read against :data:`FIXTURE_NOW`, not the wall clock.

    Once those cycles aged past the grib-filter host's retention, a CLI
    fetch of one switched itself to whole objects from the archive and
    asked the real S3 index, so the request-identity, resume and
    front-door tests failed on the calendar instead of on the code.
    """

    from woof import fetch_endpoints

    real = fetch_endpoints.cycle_age_hours

    def age(cycle, now=None):
        return real(cycle, FIXTURE_NOW if now is None else now)

    monkeypatch.setattr(fetch_endpoints, "cycle_age_hours", age)


def _loopback(address) -> bool:
    import ipaddress

    host = address[0] if isinstance(address, tuple) else address
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _fixture_hosts_answer_offline(monkeypatch, request):
    """The fixtures' cycles are published on every host, and no real host is asked.

    The publication and host checks sent live HEAD requests for these
    late-July cycles, so a test such as a plain fetch into an empty
    ``--out`` passed with the network up and failed without it, on the
    network rather than on the code.  Every host now says it holds what
    these fixtures ask for, as the hosts did when the fixtures were
    written; a test that needs a host to say no hands in its own probe.
    Any other connection off this machine fails at once instead of
    waiting on a real host, so nothing here depends on one.  The opt-in
    live smoke (``@pytest.mark.network``) keeps the real network.
    """

    if request.node.get_closest_marker("network"):
        return
    import socket

    from woof import fetch_endpoints

    monkeypatch.setattr(fetch_endpoints, "object_available",
                        lambda url, **kwargs: True)
    # The publication check's own question, which tells a host not heard
    # from a no; unpatched, it asked the blocked network and waited out
    # its retries before calling each cycle unchecked.
    monkeypatch.setattr(fetch_endpoints, "object_answer",
                        lambda url, **kwargs: True)
    real_connect = socket.socket.connect

    def connect(self, address):
        if (self.family in (socket.AF_INET, socket.AF_INET6)
                and not _loopback(address)):
            raise OSError(f"a fetch test tried to reach {address[0]}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)


# ---------------------------------------------------------------------------
# Synthetic-but-envelope-valid GRIB payloads
# ---------------------------------------------------------------------------

def _grib2_stream(messages: int) -> bytes:
    """A stream of minimal, envelope-valid GRIB2 messages."""
    one = (b"GRIB" + b"\x00\x00" + b"\x00" + b"\x02"
           + (20).to_bytes(8, "big") + b"7777")
    return one * messages


def _millideg(value: float) -> bytes:
    """GRIB1's sign-magnitude 3-byte millidegree encoding."""
    scaled = int(round(abs(value) * 1000.0))
    if value < 0:
        scaled |= 0x800000
    return scaled.to_bytes(3, "big")


def _latlon_gds(*, south: float, west: float, north: float, east: float,
                step: float = 0.25) -> bytes:
    """A GRIB1 GDS for a regular lat/lon grid, scanning north to south."""
    ni = int(round((east - west) / step)) + 1
    nj = int(round((north - south) / step)) + 1
    gds = bytearray(32)
    gds[0:3] = (32).to_bytes(3, "big")
    gds[3] = 0            # NV
    gds[4] = 255          # PV/PL absent
    gds[5] = 0            # data representation type 0: lat/lon
    gds[6:8] = ni.to_bytes(2, "big")
    gds[8:10] = nj.to_bytes(2, "big")
    gds[10:13] = _millideg(north)      # La1: CDS scans from the north
    gds[13:16] = _millideg(west)       # Lo1
    gds[16] = 0x80 | 0x08              # increments given, spherical earth
    gds[17:20] = _millideg(south)      # La2
    gds[20:23] = _millideg(east)       # Lo2
    gds[23:25] = int(round(step * 1000)).to_bytes(2, "big")   # Di
    gds[25:27] = int(round(step * 1000)).to_bytes(2, "big")   # Dj
    gds[27] = 0           # scanning mode: +i, -j
    return bytes(gds)


def _grib1_message(parameter: int, level_type: int, level: int,
                   when: datetime, *, edition: int = 1,
                   gds: bytes | None = None) -> bytes:
    """One envelope-valid GRIB1 analysis message with a real PDS."""
    pds = bytearray(28)
    pds[0:3] = (28).to_bytes(3, "big")
    pds[3] = 1      # parameter table version
    pds[4] = 98     # ECMWF
    pds[7] = 0 if gds is None else 128   # bit 1: a GDS follows
    pds[8] = parameter
    pds[9] = level_type
    pds[10:12] = level.to_bytes(2, "big")
    year_of_century = when.year % 100 or 100
    century = (when.year - 1) // 100 + 1
    pds[12:17] = bytes((year_of_century, when.month, when.day,
                        when.hour, when.minute))
    pds[17] = 1     # hours
    pds[18] = 0     # P1 = 0: analysis
    pds[20] = 0     # time range indicator 0
    pds[24] = century
    body = bytes(pds) + (b"" if gds is None else gds)
    total = 8 + len(body) + 4
    return (b"GRIB" + total.to_bytes(3, "big") + bytes((edition,))
            + body + b"7777")


_TIMES = (datetime(1974, 4, 3, 0), datetime(1974, 4, 3, 6))
_LADDER = (100, 200, 300, 500, 700, 850, 925, 1000)


def _era5_set(path: Path, *, soil_level_type: int = 1,
              drop: tuple[int, ...] = (), times=_TIMES,
              orography: bool = True, gds: bytes | None = None,
              surface_gds: bytes | None = None) -> Path:
    payload = bytearray()
    surface_grid = gds if surface_gds is None else surface_gds
    for when in times:
        for parameter in fetch.ERA5_REQUIRED_PRESSURE:
            for level in _LADDER:
                payload += _grib1_message(parameter, 100, level, when,
                                          gds=gds)
        for parameter in fetch.ERA5_REQUIRED_SURFACE:
            if parameter in drop:
                continue
            level_type = (soil_level_type
                          if parameter in fetch.ERA5_SOIL_PARAMETERS else 1)
            payload += _grib1_message(parameter, level_type, 0, when,
                                      gds=surface_grid)
    if orography:
        payload += _grib1_message(
            fetch.ERA5_OROGRAPHY_PARAMETER, 1, 0, times[0], gds=surface_grid)
    path.write_bytes(bytes(payload))
    return path


# ---------------------------------------------------------------------------
# Area / cycle / hours parsing
# ---------------------------------------------------------------------------

def test_area_parses_corner_order_free_and_rejects_junk():
    area = fetch.parse_area("40,-90,30,-100")
    assert (area.lat_south, area.lon_west, area.lat_north,
            area.lon_east) == (30.0, -100.0, 40.0, -90.0)
    for bad in ("30,-100,40", "a,b,c,d", "30,-100,30,-90", "95,0,96,10"):
        with pytest.raises(ValueError):
            fetch.parse_area(bad)


def test_area_nomads_box_uses_0_360_and_widens_over_greenwich():
    box = fetch.parse_area("30,-100,40,-90").as_nomads()
    assert box == {"left_lon": 260.0, "right_lon": 270.0,
                   "bottom_lat": 30.0, "top_lat": 40.0}
    area = fetch.parse_area("40,-10,50,10")
    wrapped = area.as_nomads()
    assert (wrapped["left_lon"], wrapped["right_lon"]) == (0.0, 360.0)
    assert area.longitude_span_degrees == 20.0
    assert area.nomads_longitude_amplification == 18.0


def test_area_longitude_pair_wider_than_180_crosses_the_antimeridian():
    """Worldwide contract: 170,-170 is the 20-degree Pacific box over
    180E, never the silent 340-degree complement the old sorted()
    corners produced (that bug fetched the wrong data with no error)."""
    area = fetch.parse_area("30,170,50,-170")
    assert (area.lon_west, area.lon_east) == (170.0, -170.0)
    assert area.crosses_antimeridian
    # Order-free like every other corner pair.
    assert fetch.parse_area("50,-170,30,170").lon_west == 170.0
    # NOMADS serves the crossing box as a contiguous 0-360 subregion.
    box = area.as_nomads()
    assert (box["left_lon"], box["right_lon"]) == (170.0, 190.0)
    # CDS reads west > east as a crossing box, in the signed convention.
    assert area.as_cds() == [50.0, 170.0, 30.0, -170.0]
    # Non-crossing boxes are untouched.
    assert not fetch.parse_area("30,-100,40,-90").crosses_antimeridian
    # A full band (span 360) stays a full band.
    band = fetch.parse_area("-10,-180,10,180")
    assert not band.crosses_antimeridian
    assert band.as_nomads()["left_lon"] == 0.0
    assert band.as_nomads()["right_lon"] == 360.0
    assert band.nomads_longitude_amplification is None


def test_point_box_wraps_across_the_antimeridian():
    area = fetch.area_from_point("-17.8,178.5", 500.0)
    assert area.crosses_antimeridian
    assert area.lon_west > 0.0 > area.lon_east
    box = area.as_nomads()
    assert box["left_lon"] < 180.0 < box["right_lon"]


def test_point_radius_becomes_a_bounding_box():
    area = fetch.area_from_point("35,-97.5", 300.0)
    assert area.lat_south < 35.0 < area.lat_north
    assert area.lon_west < -97.5 < area.lon_east
    # longitude half-width exceeds latitude half-width off the equator
    assert (area.lon_east - area.lon_west) > (
        area.lat_north - area.lat_south)
    with pytest.raises(ValueError, match="radius-km"):
        fetch.area_from_point("35,-97.5", -1.0)
    with pytest.raises(ValueError, match="pole"):
        fetch.area_from_point("89.9,0", 300.0)


def test_cycle_parsing_enforces_source_cadence():
    assert fetch.parse_cycle("2026-07-28T06", "gfs") == datetime(
        2026, 7, 28, 6)
    assert fetch.parse_cycle("2026-07-28T05", "hrrr") == datetime(
        2026, 7, 28, 5)
    with pytest.raises(ValueError, match="00/06/12/18"):
        fetch.parse_cycle("2026-07-28T05", "gfs")
    with pytest.raises(ValueError, match="YYYY-MM-DDTHH"):
        fetch.parse_cycle("2026-07-28 05:00", "hrrr")


def test_forecast_hour_ladders():
    assert fetch.gfs_forecast_hours(6, 3) == (0, 3, 6)
    assert fetch.gfs_forecast_hours(2, 1) == (0, 1, 2)
    with pytest.raises(ValueError, match="multiple"):
        fetch.gfs_forecast_hours(4, 3)
    with pytest.raises(ValueError, match="horizon"):
        fetch.gfs_forecast_hours(387, 3)
    assert fetch.hrrr_forecast_hours(2, datetime(2026, 7, 28, 5)) == (
        0, 1, 2)
    with pytest.raises(ValueError, match="horizon f18"):
        fetch.hrrr_forecast_hours(24, datetime(2026, 7, 28, 5))
    assert fetch.hrrr_forecast_hours(
        24, datetime(2026, 7, 28, 6))[-1] == 24


# ---------------------------------------------------------------------------
# Latest-cycle resolution off the AWS S3 listing (HEAD probes)
# ---------------------------------------------------------------------------

def test_latest_gfs_probes_synoptic_cycles_newest_first():
    """Newest candidate first, and the operational server is asked first.

    The archive lags the operational server by minutes to hours, so
    resolving `latest` against the archive returned an older cycle than
    the one already published -- a run initialized an hour behind the
    best available state.
    """

    probed = []
    published = fetch.gfs_object_url(
        datetime(2026, 7, 27, 18), 6, transport="nomads")

    def probe(url: str) -> bool:
        probed.append(url)
        return url == published

    cycle = fetch.resolve_latest_cycle(
        "gfs", 6, now=datetime(2026, 7, 28, 5, 30), probe=probe)
    assert cycle == datetime(2026, 7, 27, 18)
    assert probed[0] == fetch.gfs_object_url(
        datetime(2026, 7, 28, 0), 6, transport="nomads")
    assert probed[-1].endswith("f006")


def test_latest_gfs_falls_through_to_the_archive():
    """The archive still answers when the operational server does not."""

    published = fetch.gfs_object_url(
        datetime(2026, 7, 27, 18), 6, transport="s3")
    cycle = fetch.resolve_latest_cycle(
        "gfs", 6, now=datetime(2026, 7, 28, 5, 30),
        probe=lambda url: url == published)
    assert cycle == datetime(2026, 7, 27, 18)


def test_latest_hrrr_skips_cycles_whose_horizon_is_too_short():
    def probe(url: str) -> bool:
        return True

    cycle = fetch.resolve_latest_cycle(
        "hrrr", 24, now=datetime(2026, 7, 28, 5, 30), probe=probe)
    # 05Z..01Z only reach f18; the newest extended cycle wins.
    assert cycle == datetime(2026, 7, 28, 0)


def test_latest_hrrr_requires_both_wrfnat_and_wrfprs():
    """A cycle whose final wrfprs (soil source) 404s must not win.

    During a live publication wrfnat can appear minutes before its
    wrfprs sibling; fetching needs both, so completeness probes both.
    """
    newest = datetime(2026, 7, 28, 5)
    probed = []

    def probe(url: str) -> bool:
        probed.append(url)
        # Newest cycle: wrfnat published, wrfprs not yet.
        if url == fetch.hrrr_object_url(newest, 2, "wrfprs",
                                        transport="nomads"):
            return False
        return True

    cycle = fetch.resolve_latest_cycle(
        "hrrr", 2, now=datetime(2026, 7, 28, 5, 30), probe=probe)
    assert cycle == datetime(2026, 7, 28, 4)
    for product in ("wrfnat", "wrfprs"):
        assert fetch.hrrr_object_url(
            newest, 2, product, transport="nomads") in probed
    # The winning cycle's soil object was probed too, not assumed.
    assert fetch.hrrr_object_url(
        cycle, 2, "wrfprs", transport="nomads") in probed


def test_latest_fails_closed_when_no_cycle_is_complete():
    with pytest.raises(RuntimeError, match="explicit --cycle"):
        fetch.resolve_latest_cycle(
            "gfs", 6, now=datetime(2026, 7, 28, 5, 30),
            probe=lambda url: False)


# ---------------------------------------------------------------------------
# `latest` for every registered source, from declared facts
# ---------------------------------------------------------------------------


def test_latest_resolves_a_reanalysis_to_its_newest_published_analysis():
    """ERA5 has a latest, and it is a time, not a refusal.

    The resolver used to answer this with "--cycle latest is only
    meaningful for gfs/gdas/hrrr; ERA5 is a reanalysis published with a
    delay of several days" -- which describes a delay, and a delay is a
    number, and a number resolves.  The CDS publishes no object to HEAD,
    so the declared delay IS the answer and no probe is spent.
    """

    probed = []
    cycle = fetch.resolve_latest_cycle(
        "era5", 6, now=datetime(2026, 7, 28, 5, 30),
        probe=lambda url: probed.append(url) or False)

    # The endpoint is five days back on the analysis grid; the start must
    # precede it by the requested six-hour window.
    assert cycle == datetime(2026, 7, 22, 18)
    # A job API has nothing to probe, so nothing was probed.
    assert probed == []


def test_latest_resolves_a_source_that_was_never_in_the_trio():
    """RAP is not gfs/gdas/hrrr and its `latest` is derived, not branched.

    The probe URLs come from the route table's own planner, so this
    holds for every table route without a line naming one.
    """

    now = datetime(2026, 7, 28, 5, 30)
    served = fetch.cycle_probe_urls("rap", datetime(2026, 7, 28, 2), 6)
    cycle = fetch.resolve_latest_cycle(
        "rap", 6, now=now, probe=lambda url: url in served)

    assert cycle == datetime(2026, 7, 28, 2)


def test_latest_refuses_an_undeclared_source_by_naming_the_absence():
    """Not "it is not on the list" -- what the row does not declare."""

    with pytest.raises(ValueError) as refusal:
        fetch.resolve_latest_cycle("nam", 6,
                                   now=datetime(2026, 7, 28, 5, 30))
    text = str(refusal.value)
    assert "nam" in text
    assert "declares when that source initializes" in text
    # And it must not name any other source as the reason.
    assert "gfs/gdas/hrrr" not in text
    assert "ERA5" not in text and "reanalysis" not in text


def test_every_registered_source_resolves_latest_or_says_what_it_lacks():
    """THE GATE THAT PREVENTS THE NEXT TRIO.

    Every row in the registry is asked the same question, and the answer
    is either a concrete cycle or a refusal DERIVED from that row.  A
    future capability decided by a hand-written list of model names
    fails here, because the refusal it produces will not carry the
    row-derived sentence.
    """

    from woof.source_adapters import source_adapters
    from woof.source_cycles import cycle_grid_for

    now = datetime(2026, 7, 28, 5, 30)
    resolved, refused = [], []
    for adapter in source_adapters():
        source = adapter.source_id
        try:
            cycle = fetch.resolve_latest_cycle(
                source, 0, now=now, probe=lambda url: True)
        except ValueError as refusal:
            assert cycle_grid_for(source) is None, (
                f"{source} declares a cycle grid but latest refused it")
            assert "declares when that source initializes" in str(refusal), (
                f"{source} was refused by something other than its row")
            refused.append(source)
            continue
        except RuntimeError:
            # A probe that answers yes cannot exhaust the ladder; this
            # would be a horizon miss, which is still a derived answer.
            refused.append(source)
            continue
        grid = cycle_grid_for(source)
        if grid is None:
            # Closed annual CF archives resolve their final analysis from
            # their packaged coverage, rather than a forecast cycle grid.
            from woof import cf_archive_fetch
            assert fetch.native_cf_fetch_contract(source) is not None
            assert adapter.time_axis == "analysis_times"
            metadata = cf_archive_fetch.row(source)
            assert cycle == datetime.fromisoformat(metadata["coverage_end"])
            assert cf_archive_fetch.validate_window(source, cycle, 0) == (cycle,)
        assert cycle <= now
        resolved.append(source)

    # The trio is dead: every table-route source resolves, and so does
    # the reanalysis that used to be the refusal's own example.
    for source in ("gfs", "gdas", "hrrr", "era5", "rap", "icon-eu",
                   "gem-gdps", "aifs"):
        assert source in resolved, f"{source} should resolve latest"
    assert refused, "the gate is vacuous if nothing is refused"


# ---------------------------------------------------------------------------
# GFS fetch: URL construction, record-count gate, series, manifest, resume
# ---------------------------------------------------------------------------

def _fake_gfs_download(url: str, destination: Path, **kwargs) -> None:
    destination.write_bytes(_grib2_stream(fetch.GFS_SUBSET_RECORD_COUNT))


def test_fetch_gfs_default_transport_overlaps_transfers(tmp_path,
                                                        monkeypatch):
    """Bounded concurrency is the DEFAULT, not a flag.

    Two NOMADS subset requests must genuinely be in flight together
    under the default pool (the NOMADS politeness cap is 2): each of
    the first two downloads waits until it has seen the other one
    start.  A serial transport deadlocks here and fails the timeout.
    """

    from woof import fetch_pool
    import threading
    urls = []
    started = {0: threading.Event(), 3: threading.Event()}

    def download(url, destination, **kwargs):
        urls.append(url)
        hour = int(url.split(".f0", 1)[1][:2])
        if hour in started:
            started[hour].set()
            other = started[3 if hour == 0 else 0]
            assert other.wait(timeout=30.0), \
                "the other transfer never started: the transport is serial"
        _fake_gfs_download(url, destination)

    monkeypatch.setattr(gfs_transport, "_download", download)
    published = []
    original_write = fetch.write_fetch_manifest

    def record_write(out, payload):
        published.append((payload["forecast_hours"],
                          "concurrency" in payload))
        return original_write(out, payload)

    monkeypatch.setattr(fetch, "write_fetch_manifest", record_write)
    out = tmp_path / "gfs"
    manifest_path = fetch.fetch_gfs(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3, 6),
        area=fetch.parse_area("30,-100,40,-90"), out=tmp_path / "gfs",
        progress=lambda line: None,
        derived_bar=lambda cycle, **kwargs: fetch.GFS_SUBSET_RECORD_COUNT)
    manifest = json.loads(manifest_path.read_text())
    receipt = manifest["concurrency"]
    assert receipt["schema"] == fetch_pool.POOL_RECEIPT_SCHEMA
    assert receipt["workers_requested"] == fetch_pool.DEFAULT_FILE_WORKERS
    assert receipt["workers_effective"] == 3
    assert receipt["host_caps"] == {"nomads.ncep.noaa.gov": 2}
    assert receipt["files"] == 3
    assert receipt["bytes"] == sum(
        item["bytes"] for item in manifest["files"]
        if item["role"] == "gfs-subset")
    assert receipt["wall_seconds"] > 0.0
    assert receipt["modeled_serial_seconds"] >= 0.0
    assert "effective_speedup" in receipt
    assert len(urls) == 3
    # Completion order is the pool's business; inspect the f000 request.
    urls.sort()
    query = dict(parse_qsl(urlsplit(urls[0]).query, keep_blank_values=True))
    assert urlsplit(urls[0]).path.endswith("filter_gfs_0p25.pl")
    assert query["file"] == "gfs.t06z.pgrb2.0p25.f000"
    assert query["dir"] == "/gfs.20260728/06/atmos"
    assert (query["leftlon"], query["rightlon"]) == ("260", "270")
    assert (query["bottomlat"], query["toplat"]) == ("30", "40")
    assert query["var_TSOIL"] == "on" and query["lev_500_mb"] == "on"

    series = (out / "gfs-series.tsv").read_text().splitlines()
    assert series == [
        "0\tgfs.t06z.pgrb2.0p25.f000.subset.grib2\t81",
        "3\tgfs.t06z.pgrb2.0p25.f003.subset.grib2\t96",
        "6\tgfs.t06z.pgrb2.0p25.f006.subset.grib2\t96",
    ]
    assert manifest["schema"] == fetch.FETCH_MANIFEST_SCHEMA
    assert manifest["source"] == "gfs"
    assert manifest["cycle"] == "2026-07-28T06:00:00Z"
    assert manifest["forecast_hours"] == [0, 3, 6]
    assert manifest["area"]["lon_west"] == -100.0
    roles = [item["role"] for item in manifest["files"]]
    # The payloads, the series, and the front-door receipts written
    # beside them: the manifest claims every canonical file in the
    # directory, exactly as the HRRR route's does.
    assert roles == ["gfs-subset"] * 3 + [
        "series", "checksums", "input-list", "prep-command"]
    for item in manifest["files"]:
        path = out / item["name"]
        assert item["bytes"] == path.stat().st_size
        assert item["sha256"] == hashlib.sha256(
            path.read_bytes()).hexdigest()
    assert manifest["payload_bytes"] == sum(
        item["bytes"] for item in manifest["files"])
    # One publication per verified hour, in hour order, plus the final
    # republication that attaches the completed run's concurrency
    # receipt (which cannot exist until every transfer has finished).
    assert [hours for hours, _ in published] == [
        [0], [0, 3], [0, 3, 6], [0, 3, 6]]
    assert [receipted for _, receipted in published] == [
        False, False, False, True]


def test_fetch_gfs_serial_transport_stays_reachable(tmp_path, monkeypatch):
    """--fetch-workers 1 is a knob, not a workaround: the caller's
    thread runs every transfer, in hour order, no pool involved."""

    import threading
    caller = threading.get_ident()
    threads = []
    order = []

    def download(url, destination, **kwargs):
        threads.append(threading.get_ident())
        order.append(url.split("file=")[1][:32])
        _fake_gfs_download(url, destination)

    monkeypatch.setattr(gfs_transport, "_download", download)
    manifest_path = fetch.fetch_gfs(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3, 6),
        area=fetch.parse_area("30,-100,40,-90"), out=tmp_path / "gfs",
        progress=lambda line: None, file_workers=1,
        derived_bar=lambda cycle, **kwargs: fetch.GFS_SUBSET_RECORD_COUNT)
    assert set(threads) == {caller}
    assert order == sorted(order)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["concurrency"]["workers_effective"] == 1


def test_fetch_gfs_concurrent_failure_names_the_hour_and_keeps_prefix(
        tmp_path, monkeypatch):
    """One bad file fails the request closed under concurrency too: the
    refusal still names the hour, and the manifest records exactly the
    verified contiguous prefix."""

    import threading
    f000_done = threading.Event()

    def download(url, destination, **kwargs):
        if ".f000" in url:
            _fake_gfs_download(url, destination)
            f000_done.set()
            return
        # Both later hours wait for f000 so the verified prefix is
        # deterministic, then f003 publishes a short (drifted) file.
        assert f000_done.wait(timeout=30.0)
        if ".f003" in url:
            destination.write_bytes(_grib2_stream(7))
        else:
            _fake_gfs_download(url, destination)

    monkeypatch.setattr(gfs_transport, "_download", download)
    out = tmp_path / "gfs"
    with pytest.raises(ValueError, match="f003, expected 124"):
        fetch.fetch_gfs(
            cycle=datetime(2026, 7, 28, 6), hours=(0, 3, 6),
            area=fetch.parse_area("30,-100,40,-90"), out=out,
            progress=lambda line: None,
            derived_bar=lambda cycle, **kwargs:
            fetch.GFS_SUBSET_RECORD_COUNT)
    manifest = json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())
    assert manifest["forecast_hours"] == [0]
    assert [item["role"] for item in manifest["files"]] == [
        "gfs-subset", "series", "checksums", "input-list", "prep-command"]


def test_fetch_gfs_discloses_prime_meridian_full_band_amplification(
        tmp_path, monkeypatch):
    urls = []

    def download(url, destination, **kwargs):
        urls.append(url)
        _fake_gfs_download(url, destination)

    monkeypatch.setattr(gfs_transport, "_download", download)
    lines = []
    out = tmp_path / "greenwich"
    manifest_path = fetch.fetch_gfs(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3),
        area=fetch.parse_area("40,-10,50,10"), out=out,
        progress=lines.append,
        derived_bar=lambda cycle, **kwargs: fetch.GFS_SUBSET_RECORD_COUNT)

    queries = [
        dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
        for url in urls]
    assert {(query["leftlon"], query["rightlon"])
            for query in queries} == {("0", "360")}
    notes = [line for line in lines if "NOTE" in line]
    assert len(notes) == 1
    assert "requested box lat 40..50, lon -10..10" in notes[0]
    assert "fetched band lat 40..50, lon 0..360" in notes[0]
    assert "18x longitude-span amplification" in notes[0]
    assert "compressed-byte amplification is data-dependent" in notes[0]
    # The widening is a handled condition, not a failure, and the line has
    # to say so: field reports read the bare disclosure as a warning that
    # something had gone wrong with the fetch.
    assert notes[0].endswith(
        "informational only -- the ingest interpolates the domain out of "
        "the wider band, so the only cost is download size and the run "
        "continues unchanged")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["nomads_area"]["left_lon"] == 0.0
    assert manifest["nomads_area"]["right_lon"] == 360.0
    assert manifest["longitude_span_amplification"] == 18.0
    assert "18x longitude-span amplification" in manifest["notes"]

    # An equal-width dateline crop is expressible as one narrow [0,360]
    # interval and must not inherit the Greenwich widening disclosure.
    urls.clear()
    lines.clear()
    dateline_manifest = fetch.fetch_gfs(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3),
        area=fetch.parse_area("40,170,50,-170"),
        out=tmp_path / "dateline", progress=lines.append,
        derived_bar=lambda cycle, **kwargs: fetch.GFS_SUBSET_RECORD_COUNT)
    query = dict(parse_qsl(urlsplit(urls[0]).query, keep_blank_values=True))
    assert (query["leftlon"], query["rightlon"]) == ("170", "190")
    assert not [line for line in lines if "NOTE" in line]
    manifest = json.loads(dateline_manifest.read_text())
    assert "longitude_span_amplification" not in manifest


# ---------------------------------------------------------------------------
# GFS full-file route: whole S3 objects, series, manifest, resume, refusals
# ---------------------------------------------------------------------------

#: One real f000 census (gfs.20260729/18z walked to 696 envelopes), but
#: the route pins the LIVE index against the walk, not this constant.
_FULLFILE_MESSAGES = 696


def _fullfile_env(monkeypatch, *, idx_records=_FULLFILE_MESSAGES,
                  payload_messages=_FULLFILE_MESSAGES):
    """Stub the two network seams of the full-file route.

    Returns the list the object URLs land in.  ``gfs_live_index`` is
    silenced so the ladder resolves to the certified constants without
    a network read.
    """

    urls: list[str] = []

    def download(url, destination, **kwargs):
        urls.append(url)
        destination.write_bytes(_grib2_stream(payload_messages))

    monkeypatch.setattr(gfs_transport, "_download", download)
    monkeypatch.setattr(fetch, "gfs_live_index",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(fetch, "_gfs_index_record_count",
                        lambda url, **kwargs: idx_records)
    return urls


def test_fetch_gfs_fullfile_takes_whole_objects_and_records_the_route(
        tmp_path, monkeypatch):
    urls = _fullfile_env(monkeypatch)
    out = tmp_path / "gfs-full"
    manifest_path = fetch.fetch_gfs_fullfile(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3), area=None,
        out=out, progress=lambda line: None)

    # The pool decides which transfer finishes first, so the ORDER of
    # these is not a contract and asserting it failed about one run in
    # five.  What is a contract: each whole object is requested exactly
    # once, and nothing else is.
    assert sorted(urls) == sorted(
        fetch.gfs_object_url(datetime(2026, 7, 28, 6), hour)
        for hour in (0, 3))
    series = (out / "gfs-series.tsv").read_text().splitlines()
    assert series == [
        "0\tgfs.t06z.pgrb2.0p25.f000\t81",
        "3\tgfs.t06z.pgrb2.0p25.f003\t96",
    ]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == fetch.FETCH_MANIFEST_SCHEMA
    assert manifest["mode"] == "full-file"
    assert manifest["transport"] == "s3"
    assert manifest["engine"] == "python"
    assert manifest["area"] is None
    assert manifest["forecast_hours"] == [0, 3]
    roles = [item["role"] for item in manifest["files"]]
    assert roles == ["gfs-full-file"] * 2 + [
        "series", "checksums", "input-list", "prep-command"]
    # The decode ladder is a request property the manifest must carry,
    # or the front door would let the bridge derive the mesosphere from
    # the whole-globe object.
    assert len(manifest["pressure_levels_hpa"]) == 21
    assert manifest["pressure_levels_hpa"][-1] == 1000.0
    assert manifest["source_top_pressure_pa"] == 10000.0
    for item in manifest["files"][:2]:
        assert item["grib2_messages"] == _FULLFILE_MESSAGES
        assert item["idx_records"] == _FULLFILE_MESSAGES
        assert item["url"].startswith(fetch.GFS_S3_BASE)
        path = out / item["name"]
        assert item["sha256"] == hashlib.sha256(
            path.read_bytes()).hexdigest()


def test_fullfile_without_index_binds_archived_ladder_before_publication(tmp_path, monkeypatch):
    _fullfile_env(monkeypatch, idx_records=None)
    calls = []
    available = (1., 2., 3., 5., 7., 10., 20., 30., 50., 70.,
                 *gfs_transport.PRESSURE_LEVELS_HPA)
    def inventory(path, source):
        calls.append(path)
        assert path.is_file() and source == 'gfs'
        return available, dict(method='native-file-inventory', file=path.name)
    monkeypatch.setattr(fetch, '_gfs_file_levels', inventory)
    out = tmp_path / 'archived'
    manifest = json.loads(fetch.fetch_gfs_fullfile(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3), area=None,
        out=out, top_pressure_pa=5000, progress=lambda line: None).read_text())
    assert len(calls) == 1
    assert manifest['pressure_levels_hpa'][:4] == [30., 50., 70., 100.]
    assert manifest['source_top_pressure_pa'] == 3000
    assert manifest['pressure_ladder_basis']['file_sha256'] == manifest['files'][0]['sha256']


def test_fetch_gfs_fullfile_pools_transfers_and_receipts(tmp_path,
                                                         monkeypatch):
    """The S3 whole-object route rides the same default pool: no NOMADS
    cap applies to the archive host, and the manifest receipts the run."""

    from woof import fetch_pool

    _fullfile_env(monkeypatch)
    out = tmp_path / "gfs-full"
    manifest_path = fetch.fetch_gfs_fullfile(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3, 6), area=None,
        out=out, progress=lambda line: None)
    manifest = json.loads(manifest_path.read_text())
    receipt = manifest["concurrency"]
    assert receipt["workers_requested"] == fetch_pool.DEFAULT_FILE_WORKERS
    assert receipt["workers_effective"] == 3
    assert receipt["host_caps"] == {}
    assert receipt["files"] == 3
    assert manifest["forecast_hours"] == [0, 3, 6]

    # And the serial transport stays reachable as a knob.
    serial_out = tmp_path / "gfs-full-serial"
    manifest = json.loads(fetch.fetch_gfs_fullfile(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3), area=None,
        out=serial_out, progress=lambda line: None,
        file_workers=1).read_text())
    assert manifest["concurrency"]["workers_effective"] == 1


def test_fetch_gfs_fullfile_stops_its_progress_ticker_on_every_exit(
        tmp_path, monkeypatch):
    """The whole-object route never closed its transfer monitor, so its
    ticker thread outlived the fetch and kept repeating the finished
    "2 of 2 files done" line into whatever the process wrote next: an
    in-process caller's later output, or the next test's captured stderr.
    Closed after a completed fetch and after a refused one."""

    import threading

    def tickers():
        return {thread for thread in threading.enumerate()
                if thread.name == "gpuwm-fetch-progress"}

    before = tickers()
    _fullfile_env(monkeypatch)
    fetch.fetch_gfs_fullfile(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3), area=None,
        out=tmp_path / "done", progress=lambda line: None)
    assert tickers() <= before

    _fullfile_env(monkeypatch, payload_messages=_FULLFILE_MESSAGES - 1)
    with pytest.raises(ValueError, match="695 GRIB2 messages"):
        fetch.fetch_gfs_fullfile(
            cycle=datetime(2026, 7, 28, 6), hours=(0, 3), area=None,
            out=tmp_path / "refused", progress=lambda line: None)
    assert tickers() <= before
    assert any(".rejected-" in path.name
               for path in (tmp_path / "refused").iterdir())


def test_fetch_gfs_fullfile_rust_engine_uses_the_backbone(
        tmp_path, monkeypatch):
    from woof import rustwx_fetch

    monkeypatch.setattr(fetch, "gfs_live_index",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(fetch, "_gfs_index_record_count",
                        lambda url, **kwargs: _FULLFILE_MESSAGES)
    calls: list[dict] = []

    def run_fetch(binary, **kwargs):
        calls.append(kwargs)
        name = (f"gfs.t06z.pgrb2.0p25."
                f"f{kwargs['hours'][0]:03d}")
        path = kwargs["out"] / name
        path.write_bytes(_grib2_stream(_FULLFILE_MESSAGES))
        return {"files": [{
            "name": name, "bytes": path.stat().st_size,
            "wall_seconds": 0.1, "source": "aws", "mode": "full-file",
            "mode_reason": "requested",
        }]}

    monkeypatch.setattr(rustwx_fetch, "run_fetch", run_fetch)
    manifest_path = fetch.fetch_gfs_fullfile(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3), area=None,
        out=tmp_path / "gfs-full", engine="rust",
        engine_bin=tmp_path / "rw_fetch", progress=lambda line: None)

    # The pool decides which transfer finishes first, so the ORDER these
    # leads were dispatched in belongs to the pool and not to this
    # contract; pinning it failed about one run in four.  What is a
    # contract: every requested lead is handed to the backbone exactly
    # once, and the manifest below carries the ordering the fetch does fix.
    assert sorted(call["hours"] for call in calls) == [(0,), (3,)]
    for call in calls:
        assert call["model"] == "gfs"
        assert call["product"] == "pgrb2.0p25"
        assert call["source"] == "aws"
        assert call["mode"] == "full-file"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["forecast_hours"] == [0, 3]
    assert manifest["engine"] == "rust"
    assert manifest["mode"] == "full-file"


def test_cli_fetch_gfs_mode_and_engine_contracts(tmp_path, capsys):
    def refused(needle: str, argv: list[str]) -> None:
        assert cli.main(argv) == 2
        err = capsys.readouterr().err
        assert needle in err and "Traceback" not in err

    base = ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
            "--hours", "3", "--out", str(tmp_path / "out")]
    refused("belong to '--mode full-file'",
            base + ["--area", "30,-100,40,-90", "--engine", "rust"])
    refused("--area is optional request identity", base)
    refused("--source hrrr or gfs/gdas only",
            ["fetch", "--source", "era5", "--cycle", "2026-07-28T06",
             "--hours", "6", "--area", "30,-100,40,-90",
             "--out", str(tmp_path / "era5"), "--mode", "full-file"])


def test_cli_fetch_gfs_fullfile_routes_through_the_new_transport(
        tmp_path, monkeypatch, capsys):
    from woof import rustwx_fetch

    _fullfile_env(monkeypatch)
    monkeypatch.setattr(fetch, "require_published_cycle",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: None)
    out = tmp_path / "gfs-full"
    rc = cli.main(["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
                   "--hours", "3", "--mode", "full-file",
                   "--out", str(out)])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "mode full-file" in printed
    manifest = json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())
    assert manifest["mode"] == "full-file"
    assert manifest["engine"] == "python"


def test_cli_fetch_gfs_mode_switch_refuses_on_a_subset_directory(
        tmp_path, monkeypatch, capsys):
    """The two transports name and verify files differently; one --out
    holds one of them."""

    from woof import rustwx_fetch

    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    monkeypatch.setattr(fetch, "gfs_live_index",
                        lambda *args, **kwargs: None)
    out = tmp_path / "gfs"
    fetch.fetch_gfs(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3),
        area=fetch.parse_area("30,-100,40,-90"), out=out,
        progress=lambda line: None,
        derived_bar=lambda cycle, **kwargs: fetch.GFS_SUBSET_RECORD_COUNT)

    monkeypatch.setattr(fetch, "require_published_cycle",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: None)
    rc = cli.main(["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
                   "--hours", "3", "--mode", "full-file",
                   "--area", "30,-100,40,-90", "--out", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "already holds a nomads-cgi-subset fetch" in err
    assert "Traceback" not in err


def test_author_front_door_manifest_binds_a_fullfile_directory(
        tmp_path, monkeypatch):
    """The front-door manifest serves both transports: a full-file
    directory authors the same hash-bound role inventory a subset one
    does, with the raw object names in the grib-fNNN roles."""

    _fullfile_env(monkeypatch)
    out = tmp_path / "gfs-full"
    fetch.fetch_gfs_fullfile(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3), area=None,
        out=out, progress=lambda line: None)

    bridge = tmp_path / "gfs_grib2_bridge.exe"
    bridge.write_bytes(b"not launched here; only hashed")
    wps = tmp_path / "namelist.wps"
    wps.write_text("&share\n/\n")
    config = tmp_path / "experiment.toml"
    config.write_text("[run]\n")
    path, digest = fetch.author_gfs_front_door_manifest(
        out=out, bridge=bridge, wps_namelist=wps,
        experiment_config=config, progress=lambda line: None)
    payload = json.loads(path.read_text())
    assert payload["schema"] == fetch.GFS_FRONT_DOOR_MANIFEST_SCHEMA
    assert payload["files"]["grib-f000"]["name"] == "gfs.t06z.pgrb2.0p25.f000"
    assert payload["files"]["grib-f003"]["name"] == "gfs.t06z.pgrb2.0p25.f003"
    assert payload["source"]["pressure_levels_hpa"][-1] == 1000.0
    assert payload["source"]["top_pressure_pa"] == 10000.0
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_fetch_gfs_first_hour_interrupt_records_empty_request(
        tmp_path, monkeypatch):
    def interrupt(url, destination, **kwargs):
        destination.with_suffix(destination.suffix + ".part").write_bytes(
            b"not a complete GRIB stream")
        raise KeyboardInterrupt

    monkeypatch.setattr(gfs_transport, "_download", interrupt)
    out = tmp_path / "gfs"
    area = fetch.parse_area("30,-100,40,-90")
    with pytest.raises(RuntimeError, match="resume exactly with") as caught:
        fetch.fetch_gfs(
            cycle=datetime(2026, 7, 28, 6), hours=(0, 3),
            area=area, out=out, progress=lambda line: None,
            derived_bar=lambda cycle, **kwargs:
            fetch.GFS_SUBSET_RECORD_COUNT)

    manifest = json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())
    assert manifest["forecast_hours"] == []
    assert [item["role"] for item in manifest["files"]] == [
        "series", "checksums", "input-list", "prep-command"]
    assert "f000.subset.grib2.part" in str(caught.value)
    # The empty prefix still records request identity, so the next run is
    # allowed to replace the .part instead of refusing a manifestless dir.
    fetch.check_prior_request(
        out, source="gfs", cycle=datetime(2026, 7, 28, 6), area=area)


def test_fetch_gfs_fails_closed_on_record_count_drift(tmp_path, monkeypatch):
    def short(url, destination, **kwargs):
        destination.write_bytes(_grib2_stream(7))

    monkeypatch.setattr(gfs_transport, "_download", short)
    with pytest.raises(ValueError, match="expected 124"):
        fetch.fetch_gfs(
            cycle=datetime(2026, 7, 28, 6), hours=(0, 3),
            area=fetch.parse_area("30,-100,40,-90"), out=tmp_path / "gfs",
            progress=lambda line: None)


def test_fetch_gfs_rejects_a_corrupt_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    kwargs = dict(cycle=datetime(2026, 7, 28, 6), hours=(0, 3),
                  area=fetch.parse_area("30,-100,40,-90"),
                  progress=lambda line: None)
    # A file no receipt vouches for is refused before it is even read:
    # the library door asks the same request question the command does.
    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "gfs.t06z.pgrb2.0p25.f000.subset.grib2").write_bytes(
        _grib2_stream(124) + b"junk")
    with pytest.raises(ValueError, match="carries no readable"):
        fetch.fetch_gfs(out=loose, **kwargs)
    # Under a matching receipt the file is read, and a corrupt one fails
    # its envelope walk rather than being re-blessed.
    out = tmp_path / "gfs"
    fetch.fetch_gfs(out=out, **kwargs)
    stale = out / "gfs.t06z.pgrb2.0p25.f000.subset.grib2"
    stale.write_bytes(_grib2_stream(124) + b"junk")
    with pytest.raises(ValueError, match="GRIB indicator"):
        fetch.fetch_gfs(out=out, **kwargs)


def test_grib2_message_count_walks_envelopes_exactly(tmp_path):
    path = tmp_path / "stream.grib2"
    path.write_bytes(_grib2_stream(5))
    assert fetch.count_grib2_messages(path) == 5
    path.write_bytes(_grib2_stream(5)[:-4] + b"xxxx")
    with pytest.raises(ValueError, match="7777"):
        fetch.count_grib2_messages(path)
    with pytest.raises(ValueError, match="truncated"):
        truncated = tmp_path / "short.grib2"
        truncated.write_bytes(_grib2_stream(5)[:-2])
        fetch.count_grib2_messages(truncated)
    path.write_bytes(b"GRIB" + b"\x00\x00\x00\x01"
                     + (20).to_bytes(8, "big") + b"7777")
    with pytest.raises(ValueError, match="edition"):
        fetch.count_grib2_messages(path)


# ---------------------------------------------------------------------------
# HRRR fetch: product set, manifest, checksums, resume, area gate
# ---------------------------------------------------------------------------

def _fake_hrrr_product(request, *, workers, retries,
                       expected_count=-1):
    # The real record cardinalities: an undersized fake would trip the
    # completeness bar fetch_hrrr now applies to fresh downloads too.
    request.destination.write_bytes(_grib2_stream(
        hrrr_transport.SOIL_RECORD_COUNT if request.kind == "soil"
        else hrrr_transport.ATMOSPHERE_RECORD_COUNT))
    request.index_path.write_text("1:0:fixture\n", encoding="ascii")
    return {"kind": request.kind}


def _install_runtime_surface_inputs(monkeypatch):
    """Exercise the real append/manifest path with synthetic transport bytes.

    Only Rust transport and extraction are substituted. Actual requests
    must match the source's product, pattern and numeric selectors, and
    the shipped code selects one envelope, appends it and binds its digest.
    This is a transport contract witness, not a native extraction claim.
    """
    from woof import runtime_surface_fetch, rustwx_fetch
    from woof.source_adapters import get_source_adapter
    adapter = get_source_adapter("hrrr")
    declared = {row[0]: row for row in adapter.runtime_surface_fields}
    calls = []

    def acquire(binary, **request):
        folder = Path(request["out"])
        name = folder.name
        assert name in declared
        _, product, selector, _units, numeric = declared[name]
        assert binary is not None
        assert request["model"] == adapter.upstream_model_id
        assert request["product"] == product
        assert request["mode"] == "auto" and request["keep_idx"] is True
        assert request["source"] in ("aws", "nomads")
        assert Path(request["pattern_file"]).read_text().splitlines() == [selector]
        assert len(request["hours"]) == 1
        lead = request["hours"][0]
        payload = _grib2_stream(2)
        filename = f"fixture-{product}-f{lead:02d}.grib2"
        (folder / filename).write_bytes(payload)
        calls.append({"field": name, "binary": binary,
                      "numeric_selector": dict(numeric), **request})
        return {"files": [{"name": filename,
            "grib_url": f"https://example.invalid/{request['source']}/{filename}",
            "sha256": hashlib.sha256(payload).hexdigest(), "mode": "idx-subset"}]}

    def extract(path, output, numeric_selector):
        name = Path(path).parent.name
        assert name in declared
        assert dict(numeric_selector) == dict(declared[name][4])
        assert Path(path).read_bytes() == _grib2_stream(2)
        Path(output).write_bytes(_grib2_stream(1))

    monkeypatch.setattr(rustwx_fetch, "run_fetch", acquire)
    monkeypatch.setattr(runtime_surface_fetch, "extract_runtime_record", extract)
    return calls


def _assert_runtime_surface_requests(calls, manifest, cycle):
    from woof.source_adapters import get_source_adapter
    rows = get_source_adapter("hrrr").runtime_surface_fields
    soil = [entry for entry in manifest["files"] if entry["role"] == "soil"]
    assert sorted((call["hours"][0], call["field"]) for call in calls) == sorted(
        (entry["forecast_hour"], row[0]) for entry in soil for row in rows)
    by_hour = {entry["forecast_hour"]: entry for entry in soil}
    for call in calls:
        assert call["date"] == f"{cycle:%Y%m%d}" and call["cycle"] == cycle.hour
        entry = by_hour[call["hours"][0]]
        assert call["source"] == fetch.RW_FETCH_SOURCES[entry["transport"]]
        assert entry["records"] == hrrr_transport.SOIL_RECORD_COUNT + len(rows)
        assert len(entry["runtime_surface"]) == len(rows)
        assert all(record["sha256"] == hashlib.sha256(_grib2_stream(1)).hexdigest()
                   for record in entry["runtime_surface"])


def test_fetch_hrrr_downloads_wrfnat_and_soil_products(tmp_path,
                                                       monkeypatch):
    seen = []

    def product(request, *, workers, retries, expected_count=-1):
        seen.append(request)
        return _fake_hrrr_product(request, workers=workers,
                                  retries=retries,
                                  expected_count=expected_count)

    monkeypatch.setattr(hrrr_transport, "_download_product", product)
    out = tmp_path / "hrrr"
    manifest_path = fetch.fetch_hrrr(
        cycle=datetime(2026, 7, 28, 5), hours=(0, 1), area=None, out=out,
        progress=lambda line: None)

    # Keyed by destination, not by call order: the pooled transport is
    # free to schedule the four products together.
    by_name = {request.destination.name: request for request in seen}
    assert sorted(request.kind for request in seen) == [
        "atmosphere", "atmosphere", "soil", "soil"]
    atmosphere0 = by_name["hrrr.t05z.wrfnatf00.grib2"]
    assert atmosphere0.url.startswith(
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20260728/conus/")
    assert atmosphere0.url.endswith("hrrr.t05z.wrfnatf00.grib2")
    assert atmosphere0.index_url.endswith(".idx")
    # Soil records come from the pressure file but publish as soilfNN.
    soil0 = by_name["hrrr.t05z.soilf00.grib2"]
    assert soil0.url.endswith("hrrr.t05z.wrfprsf00.grib2")

    manifest = json.loads(manifest_path.read_text())
    assert manifest["source"] == "hrrr"
    assert [item["role"] for item in manifest["files"]] == [
        "atmosphere", "soil", "atmosphere", "soil", "checksums"]
    sums = (out / "SHA256SUMS").read_text().splitlines()
    assert len(sums) == 4
    for line in sums:
        digest, name = line.split("  ")
        assert digest == hashlib.sha256(
            (out / name).read_bytes()).hexdigest()


def test_fetch_hrrr_wait_mode_stays_publication_ordered(tmp_path,
                                                        monkeypatch):
    """--wait-for follows the cycle as it publishes: each product is
    polled, then fetched, in order -- the polling is the pacing, so the
    transfers stay serial regardless of the pool default."""

    calls = []

    def product(request, *, workers, retries, expected_count=-1):
        calls.append(request.destination.name)
        return _fake_hrrr_product(request, workers=workers,
                                  retries=retries,
                                  expected_count=expected_count)

    monkeypatch.setattr(hrrr_transport, "_download_product", product)
    fetch.fetch_hrrr(
        cycle=datetime(2026, 7, 28, 5), hours=(0, 1), area=None,
        out=tmp_path / "hrrr", progress=lambda line: None,
        transport="auto", wait=True, wait_timeout_s=60.0,
        probe=lambda url: True, sleeper=lambda seconds: None)
    assert calls == [
        "hrrr.t05z.wrfnatf00.grib2", "hrrr.t05z.soilf00.grib2",
        "hrrr.t05z.wrfnatf01.grib2", "hrrr.t05z.soilf01.grib2",
    ]


def test_fetch_hrrr_redownloads_boundary_truncated_existing_file(
        tmp_path, monkeypatch):
    """A prefix truncated at a GRIB message boundary walks clean but
    fails the 561-record count: it must be re-downloaded, never blessed
    with a fresh digest."""
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    kwargs = dict(cycle=datetime(2026, 7, 28, 5), hours=(0,), area=None,
                  out=out)
    fetch.fetch_hrrr(**kwargs, progress=lambda line: None)

    victim = out / "hrrr.t05z.wrfnatf00.grib2"
    complete = victim.read_bytes()
    message = len(complete) // hrrr_transport.ATMOSPHERE_RECORD_COUNT
    truncated = complete[:100 * message]  # clean EOF, wrong count
    victim.write_bytes(truncated)
    truncated_sha = hashlib.sha256(truncated).hexdigest()

    downloaded = []

    def product(request, *, workers, retries, expected_count=-1):
        downloaded.append(request.destination.name)
        return _fake_hrrr_product(request, workers=workers,
                                  retries=retries,
                                  expected_count=expected_count)

    monkeypatch.setattr(hrrr_transport, "_download_product", product)
    lines: list[str] = []
    manifest_path = fetch.fetch_hrrr(**kwargs, progress=lines.append)

    assert downloaded == [victim.name]  # the soil file stays skipped
    assert any("expected "
               f"{hrrr_transport.ATMOSPHERE_RECORD_COUNT}" in line
               for line in lines)
    # The truncated bytes were quarantined, not deleted and not blessed.
    rejected = list(out.glob(f"{victim.name}.rejected-*"))
    assert len(rejected) == 1
    assert rejected[0].read_bytes() == truncated
    assert victim.read_bytes() == complete
    manifest = json.loads(manifest_path.read_text())
    shas = {item["name"]: item["sha256"] for item in manifest["files"]}
    assert shas[victim.name] == hashlib.sha256(complete).hexdigest()
    assert truncated_sha not in shas.values()
    assert truncated_sha not in (out / "SHA256SUMS").read_text()


def test_fetch_hrrr_redownloads_on_prior_manifest_digest_mismatch(
        tmp_path, monkeypatch):
    """Right count, wrong bytes: the prior manifest digest is the
    authority, so the file is re-downloaded rather than re-blessed."""
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    kwargs = dict(cycle=datetime(2026, 7, 28, 5), hours=(0,), area=None,
                  out=out)
    fetch.fetch_hrrr(**kwargs, progress=lambda line: None)

    victim = out / "hrrr.t05z.soilf00.grib2"
    complete = victim.read_bytes()
    tampered = bytearray(complete)
    tampered[6] ^= 0xFF  # discipline byte: envelope-valid, count intact
    victim.write_bytes(bytes(tampered))

    downloaded = []

    def product(request, *, workers, retries, expected_count=-1):
        downloaded.append(request.destination.name)
        return _fake_hrrr_product(request, workers=workers,
                                  retries=retries,
                                  expected_count=expected_count)

    monkeypatch.setattr(hrrr_transport, "_download_product", product)
    lines: list[str] = []
    fetch.fetch_hrrr(**kwargs, progress=lines.append)

    assert downloaded == [victim.name]
    assert any("prior fetch manifest" in line for line in lines)
    assert victim.read_bytes() == complete


def test_fetch_hrrr_rejects_areas_outside_conus():
    with pytest.raises(ValueError, match="CONUS"):
        fetch.fetch_hrrr(
            cycle=datetime(2026, 7, 28, 5), hours=(0, 1),
            area=fetch.parse_area("45,-10,55,10"), out=Path("unused"),
            progress=lambda line: None)


def test_hrrr_coverage_envelope_matches_the_native_grid_exactly():
    """The shared coverage definition IS the grid: the boundary-ring
    shortcut agrees with a full scan of all 1799 x 1059 mass points to
    the last bit (the projected pole lies outside the grid rectangle,
    so every lat/lon extreme is attained on the boundary)."""

    from woof.ingest.hrrr import hrrr_source_grid
    from woof.ingest.hrrr_target import hrrr_coverage_envelope

    latitude, longitude = hrrr_source_grid().latlon_mass()
    assert hrrr_coverage_envelope() == (
        float(latitude.min()), float(longitude.min()),
        float(latitude.max()), float(longitude.max()))


def test_hrrr_area_gate_derives_from_the_grid_not_a_hand_cap():
    """The retired hand-held box (lat 21.1..52.7) admitted latitudes the
    grid does not carry -- its own 52.70 cap sits north of the real
    52.6157 top.  The gate now sits exactly on the grid envelope: the
    inward-quantized envelope box passes, one hint-quantum past any
    edge fails, and an antimeridian-crossing box (which the old
    corner-order test waved through) is refused."""

    envelope = fetch.source_coverage_envelope("hrrr")
    assert envelope is not None
    south, west, north, east = fetch.area_bounds_inward(envelope)
    fetch.validate_fetch_area("hrrr", fetch.parse_area(
        f"{south:.2f},{west:.2f},{north:.2f},{east:.2f}"))
    for box in (
            f"{south - 0.01:.2f},{west:.2f},{north:.2f},{east:.2f}",
            f"{south:.2f},{west - 0.01:.2f},{north:.2f},{east:.2f}",
            f"{south:.2f},{west:.2f},{north + 0.01:.2f},{east:.2f}",
            f"{south:.2f},{west:.2f},{north:.2f},{east + 0.01:.2f}",
    ):
        with pytest.raises(ValueError, match="coverage"):
            fetch.validate_fetch_area("hrrr", fetch.parse_area(box))
    with pytest.raises(ValueError, match="coverage"):
        fetch.validate_fetch_area("hrrr", fetch.parse_area("30,170,45,-170"))
    # Global sources carry no coverage box; any parseable area passes.
    assert fetch.source_coverage_envelope("gfs") is None
    fetch.validate_fetch_area("gfs", fetch.parse_area("45,-10,55,10"))


# ---------------------------------------------------------------------------
# ERA5: template emission + user-file validation
# ---------------------------------------------------------------------------

def test_era5_template_binds_area_times_levels_and_variables(tmp_path):
    area = fetch.parse_area("30,-100,40,-90")
    template = fetch.era5_request_template(
        cycle=datetime(1974, 4, 3, 0), hours=12, area=area)
    pressure, single = (item["request"] for item in template["requests"])
    assert template["requests"][0]["dataset"] == (
        "reanalysis-era5-pressure-levels")
    assert pressure["area"] == [40.0, -100.0, 30.0, -90.0]
    assert pressure["date"] == ["1974-04-03"]
    assert pressure["time"] == ["00:00", "06:00", "12:00"]
    assert len(pressure["pressure_level"]) == 37
    assert "relative_humidity" in pressure["variable"]
    assert "geopotential" in single["variable"]  # invariant orography
    assert "snow_depth" in single["variable"]
    assert {"soil_temperature_level_4",
            "volumetric_soil_water_layer_4"} <= set(single["variable"])
    with pytest.raises(ValueError, match="multiple"):
        fetch.era5_request_template(
            cycle=datetime(1974, 4, 3, 0), hours=5, area=area)


def test_era5_validation_passes_a_complete_set(tmp_path):
    path = _era5_set(tmp_path / "era5.grib")
    report = fetch.validate_era5_files((path,), expected_times=_TIMES)
    assert report.ok, report.format()
    text = report.format()
    assert "8 levels" in text and "fully covered" in text


def test_era5_validation_accepts_native_cds_soil_level_type(tmp_path):
    path = _era5_set(tmp_path / "era5.grib", soil_level_type=112)
    assert fetch.validate_era5_files((path,)).ok


def test_era5_validation_names_missing_fields_and_times(tmp_path):
    dropped = _era5_set(tmp_path / "missing-soil.grib", drop=(41,))
    report = fetch.validate_era5_files((dropped,))
    assert not report.ok
    assert any("swvl3" in failure for failure in report.failures)

    bare = _era5_set(tmp_path / "no-orog.grib", orography=False)
    report = fetch.validate_era5_files((bare,))
    assert any("source-orography" in failure for failure in report.failures)

    short = _era5_set(tmp_path / "short.grib", times=_TIMES[:1])
    report = fetch.validate_era5_files((short,), expected_times=_TIMES)
    assert any("missing valid times" in failure
               for failure in report.failures)


def test_era5_validation_reports_the_delivered_geographic_extent(tmp_path):
    """A PASS that never says WHERE the bytes are cannot catch the most
    likely retrieval mistake: a file cropped to the wrong box."""

    path = _era5_set(
        tmp_path / "era5.grib",
        gds=_latlon_gds(south=34.5, west=-104.25, north=43.5, east=-91.75))
    report = fetch.validate_era5_files((path,))
    assert report.ok, report.format()
    text = report.format()
    assert "34.50" in text and "43.50" in text
    assert "-104.25" in text and "-91.75" in text
    assert "51x37" in text          # (east-west)/0.25 + 1 by (n-s)/0.25 + 1


def test_era5_validation_says_so_when_no_box_was_checked(tmp_path):
    path = _era5_set(
        tmp_path / "era5.grib",
        gds=_latlon_gds(south=34.5, west=-104.25, north=43.5, east=-91.75))
    text = fetch.validate_era5_files((path,)).format()
    assert "not checked against any requested box" in text
    assert "--area" in text


def test_era5_validation_fails_a_file_cropped_off_the_requested_box(tmp_path):
    """The wrong-box file is the expensive mistake: it validates, prepares,
    and only refuses at `woof check` or mid-run."""

    path = _era5_set(
        tmp_path / "era5.grib",
        gds=_latlon_gds(south=34.5, west=-104.25, north=43.5, east=-91.75))
    report = fetch.validate_era5_files(
        (path,), expected_area=fetch.parse_area("30,-110,45,-90"))
    assert not report.ok
    joined = " ".join(report.failures)
    assert "south" in joined and "west" in joined
    assert "34.50" in joined and "30.00" in joined


def test_era5_validation_tolerates_the_cds_grid_snap(tmp_path):
    """CDS snaps a requested area onto its native grid, and the snap can go
    inward by up to one cell.  One cell of shortfall is the provider's own
    rounding, not a mis-fetch."""

    path = _era5_set(
        tmp_path / "era5.grib",
        gds=_latlon_gds(south=34.5, west=-104.25, north=43.5, east=-91.75))
    report = fetch.validate_era5_files(
        (path,), expected_area=fetch.parse_area("34.30,-104.39,43.63,-91.61"))
    assert report.ok, report.format()
    assert "covers the requested box" in report.format()


def test_era5_validation_fails_when_the_two_retrievals_used_two_boxes(
        tmp_path):
    path = _era5_set(
        tmp_path / "era5.grib",
        gds=_latlon_gds(south=34.5, west=-104.25, north=43.5, east=-91.75),
        surface_gds=_latlon_gds(
            south=20.0, west=-100.0, north=30.0, east=-90.0))
    report = fetch.validate_era5_files((path,))
    assert not report.ok
    assert any("2 different grids" in failure for failure in report.failures)


def test_era5_validation_fails_closed_on_transport_defects(tmp_path):
    wrong_edition = tmp_path / "era5.grib"
    wrong_edition.write_bytes(
        _grib1_message(129, 100, 500, _TIMES[0], edition=2))
    report = fetch.validate_era5_files((wrong_edition,))
    assert any("edition" in failure for failure in report.failures)
    report = fetch.validate_era5_files((tmp_path / "absent.grib",))
    assert any("missing input file" in failure
               for failure in report.failures)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def test_cli_fetch_era5_template_flow(tmp_path, capsys):
    rc = cli.main([
        "fetch", "--source", "era5", "--cycle", "1974-04-03T00",
        "--hours", "12", "--area", "30,-100,40,-90",
        "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cdsapi" in out and "~/.cdsapirc" in out
    spec = json.loads((tmp_path / fetch.ERA5_REQUEST_NAME).read_text())
    assert spec["schema"] == "gpuwm-era5-cds-request-v1"
    assert len(spec["requests"]) == 2


def test_cli_fetch_era5_validate_flow(tmp_path, capsys):
    path = _era5_set(tmp_path / "era5.grib")
    rc = cli.main(["fetch", "--source", "era5", "--validate", str(path)])
    assert rc == 0
    assert "era5 validation: PASS" in capsys.readouterr().out
    bad = _era5_set(tmp_path / "bad.grib", orography=False)
    rc = cli.main(["fetch", "--source", "era5", "--validate", str(bad)])
    assert rc == 1


def test_cli_fetch_era5_validate_checks_the_box_when_area_is_given(
        tmp_path, capsys):
    path = _era5_set(
        tmp_path / "era5.grib",
        gds=_latlon_gds(south=34.5, west=-104.25, north=43.5, east=-91.75))
    rc = cli.main(["fetch", "--source", "era5", "--validate", str(path),
                   "--area", "34.30,-104.39,43.63,-91.61"])
    assert rc == 0
    assert "covers the requested box" in capsys.readouterr().out
    rc = cli.main(["fetch", "--source", "era5", "--validate", str(path),
                   "--area", "30,-110,45,-90"])
    assert rc == 1
    assert "era5 validation: FAIL" in capsys.readouterr().out


def test_era5_request_targets_land_in_out_not_the_users_cwd(tmp_path):
    out = tmp_path / "era5-raw"
    fetch.write_era5_request(
        cycle=datetime(1974, 4, 3, 0), hours=12,
        area=fetch.parse_area("30,-100,40,-90"), out=out,
        progress=lambda _line: None)
    spec = json.loads(
        (out / fetch.ERA5_REQUEST_NAME).read_text(encoding="utf-8"))
    for item in spec["requests"]:
        target = Path(item["target"])
        assert target.is_absolute(), item["target"]
        assert target.parent == out.resolve()
    assert Path(spec["combine_target"]).parent == out.resolve()


def test_era5_request_writes_a_runnable_retrieval_script(tmp_path, capsys):
    out = tmp_path / "era5-raw"
    rc = cli.main([
        "fetch", "--source", "era5", "--cycle", "1974-04-03T00",
        "--hours", "12", "--area", "30,-100,40,-90", "--out", str(out)])
    assert rc == 0
    script = out / fetch.ERA5_RETRIEVE_NAME
    assert script.is_file()
    body = script.read_text(encoding="utf-8")
    # The script must resolve its own directory: the retrieval commonly runs
    # from another interpreter (WSL) whose view of the path differs.
    assert "__file__" in body
    compile(body, str(script), "exec")
    printed = capsys.readouterr().out
    assert str(script) in printed
    # Printed commands must paste into one shell line -- a multi-line
    # `python -c` is not runnable in PowerShell.
    for line in printed.splitlines():
        stripped = line.strip()
        if stripped.startswith("python -c") or stripped.startswith("python3 -c"):
            assert stripped.count('"') % 2 == 0, stripped


def test_era5_instructions_do_not_send_windows_paths_into_wsl(tmp_path):
    """A retrieval snippet that embeds `C:\\...` cannot open the file from
    inside WSL, which is where the provider client usually runs on Windows."""

    out = tmp_path / "era5-raw"
    lines: list[str] = []
    fetch.write_era5_request(
        cycle=datetime(1974, 4, 3, 0), hours=12,
        area=fetch.parse_area("30,-100,40,-90"), out=out,
        progress=lines.append)
    text = "\n".join(lines)
    for line in text.splitlines():
        if "wsl " in line:
            assert re.search(r"[A-Za-z]:[\\/]", line) is None, line
    # And the key advisory must name the home the retrieval interpreter
    # reads, not only this box's home.
    assert "the interpreter that runs" in text


def test_era5_wsl_path_translation():
    """The three platform shapes, with the user segment substituted.

    The profile owner is written at run time rather than as a literal
    because the release snapshot builder scans this file: a spelled-out
    ``C:/Users/<name>`` here is a developer-absolute path that would
    ship, and it failed that gate twice on the 2.5.x line.  The
    substitution is the remedy the gate names, and it asserts the same
    thing -- a user profile is translated on Windows, whatever it is
    called, and a POSIX path comes back untouched.
    """

    user = "somebody"
    assert fetch.wsl_path(Path(f"C:/Users/{user}/era5/x.json")) == (
        f"/mnt/c/Users/{user}/era5/x.json")
    assert fetch.wsl_path(Path(rf"C:\Users\{user}\era5\x.json")) == (
        f"/mnt/c/Users/{user}/era5/x.json")
    assert fetch.wsl_path(Path(f"/home/{user}/x.json")) == f"/home/{user}/x.json"


@pytest.mark.parametrize("requires_retrieve", [False, True])
@pytest.mark.parametrize("retrieve", [False, True])
def test_retrieve_gate_and_hints_follow_the_registry(
        tmp_path, monkeypatch, requires_retrieve, retrieve):
    from dataclasses import replace
    from woof import source_adapters

    source = fetch.GFS_CONTAINER_SOURCES[0]
    original_row = source_adapters.get_source_adapter
    row = replace(original_row(source), fetch_requires_retrieve=requires_retrieve)
    monkeypatch.setattr(source_adapters, "get_source_adapter",
                        lambda name: row if name == source else original_row(name))
    args = cli.build_parser().parse_args([
        "fetch", "--source", source, "--cycle", "2026-07-28T06",
        "--hours", "6", "--area=35,-100,40,-95", "--out", str(tmp_path),
        *(["--retrieve"] if retrieve else [])])
    hints = {"source": source, "cycle": args.cycle, "hours": args.hours}
    if retrieve:
        hints["retrieve"] = True
    if retrieve and not requires_retrieve:
        for call in (lambda: fetch.fetch_main(args),
                     lambda: fetch.validate_fetch_hints(hints, source="request.toml")):
            with pytest.raises(ValueError, match="default fetch writes a request template"):
                call()
        return

    fetch.validate_fetch_hints(hints, source="request.toml")
    # A retrieval fact does not enable another transport's provider flags.
    with pytest.raises(ValueError, match="ERA5 only"):
        fetch.validate_fetch_hints({**hints, "era5_provider": "cds"}, source="request.toml")
    seen = {}
    validate = fetch.validate_fetch_hints

    def record(table, *, source):
        seen.update(table)
        return validate(table, source=source)

    class TransportReached(Exception):
        pass

    def contact(*args, **kwargs):
        raise TransportReached

    monkeypatch.setattr(fetch, "validate_fetch_hints", record)
    monkeypatch.setattr(fetch, "require_published_cycle", contact)
    with pytest.raises(TransportReached):
        fetch.fetch_main(args)
    assert ("retrieve" in seen) is requires_retrieve
    if requires_retrieve:
        assert seen["retrieve"] is retrieve


def test_cli_fetch_argument_contracts(tmp_path, capsys):
    with pytest.raises(SystemExit):  # --source is required with choices
        cli.main(["fetch", "--cycle", "latest"])

    def refused(needle: str, argv: list[str]) -> None:
        # Documented refusals surface as exit 2 + a stderr message
        # through the CLI dispatch boundary, never as a traceback.
        assert cli.main(argv) == 2
        err = capsys.readouterr().err
        assert needle in err and "Traceback" not in err

    refused("requires --area",
            ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
             "--hours", "6", "--out", str(tmp_path)])
    refused("mutually exclusive",
            ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
             "--hours", "6", "--out", str(tmp_path),
             "--area", "30,-100,40,-90", "--point", "35,-97"])
    refused("requires --radius-km",
            ["fetch", "--source", "hrrr", "--cycle", "2026-07-28T05",
             "--hours", "2", "--out", str(tmp_path),
             "--point", "35,-97"])
    refused("era5 only",
            ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
             "--hours", "6", "--out", str(tmp_path),
             "--validate", "x.grib"])
    refused("hourly",
            ["fetch", "--source", "hrrr", "--cycle", "2026-07-28T05",
             "--hours", "2", "--out", str(tmp_path), "--cadence", "3"])
    refused("requires --cycle",
            ["fetch", "--source", "hrrr", "--hours", "2",
             "--out", str(tmp_path)])
    # ERA5 latest resolves an analysis window and emits a request document;
    # it does not probe or retrieve anything from the keyed job API.
    assert cli.main(["fetch", "--source", "era5", "--cycle", "latest",
                     "--hours", "6", "--area", "30,-100,40,-90",
                     "--out", str(tmp_path / "era5-latest")]) == 0
    assert (tmp_path / "era5-latest" / "era5-cds-request.json").is_file()


def test_cli_fetch_gfs_end_to_end_with_mocked_transport(tmp_path,
                                                        monkeypatch,
                                                        capsys):
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    rc = cli.main(["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
                   "--hours", "6", "--area", "30,-100,40,-90",
                   "--out", str(out)])
    assert rc == 0
    printed = capsys.readouterr().out
    assert printed.count("fetch gfs f0") == 3
    assert fetch.FETCH_MANIFEST_NAME in printed
    assert (out / "gfs-series.tsv").is_file()


def test_cli_fetch_workers_knob_plumbs_refuses_and_documents(tmp_path,
                                                             monkeypatch,
                                                             capsys):
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    base = ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
            "--hours", "6", "--area", "30,-100,40,-90", "--out", str(out)]
    assert cli.main(base + ["--fetch-workers", "2"]) == 0
    capsys.readouterr()
    manifest = json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())
    assert manifest["concurrency"]["workers_requested"] == 2

    # Zero names no schedulable pool and refuses before any network.
    rc = cli.main(base + ["--fetch-workers", "0"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "serial transport" in err and "Traceback" not in err

    # ERA5 is a manual CDS retrieval: nothing to parallelize.
    rc = cli.main(["fetch", "--source", "era5", "--cycle", "2026-07-28T06",
                   "--hours", "6", "--area", "30,-100,40,-90",
                   "--out", str(tmp_path / "era5"), "--fetch-workers", "4"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--fetch-workers" in err and "era5" in err.lower()


def test_fetch_throughput_surfaces_the_concurrency_receipt(tmp_path,
                                                           monkeypatch):
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    fetch.fetch_gfs(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3),
        area=fetch.parse_area("30,-100,40,-90"), out=out,
        progress=lambda line: None,
        derived_bar=lambda cycle, **kwargs: fetch.GFS_SUBSET_RECORD_COUNT)
    throughput = fetch.fetch_throughput(out)
    receipt = throughput["concurrency"]
    assert receipt["workers_requested"] > 0
    assert receipt["wall_seconds"] > 0.0
    # A manifest written before the receipt existed says None, not zero.
    payload = json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())
    del payload["concurrency"]
    (out / fetch.FETCH_MANIFEST_NAME).write_text(
        json.dumps(payload), encoding="utf-8")
    assert fetch.fetch_throughput(out)["concurrency"] is None


# ---------------------------------------------------------------------------
# Area-blind resume trap: request-identity comparison + --force-refetch
# ---------------------------------------------------------------------------

def test_cli_refetch_with_different_area_refuses(tmp_path, monkeypatch,
                                                 capsys):
    """The acceptance repro: a larger --area into the same --out must
    refuse (per-file record counts are area-blind), naming the exact
    difference and the remedy."""
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    base = ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
            "--hours", "3", "--out", str(out)]
    assert cli.main(base + ["--area", "30,-100,40,-90"]) == 0
    capsys.readouterr()
    before = (out / "gfs.t06z.pgrb2.0p25.f000.subset.grib2").read_bytes()

    rc = cli.main(base + ["--area", "20,-110,50,-80"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "different request" in err
    assert "area" in err and "--force-refetch" in err
    assert "Traceback" not in err
    # The refusal changed nothing on disk.
    assert (out / "gfs.t06z.pgrb2.0p25.f000.subset.grib2"
            ).read_bytes() == before

    # --force-refetch moves the old files aside (never deletes) and
    # re-downloads; the manifest then records the new request.
    rc = cli.main(base + ["--area", "20,-110,50,-80", "--force-refetch"])
    assert rc == 0
    manifest = json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())
    assert manifest["area"]["lon_west"] == -110.0
    rejected = list(out.glob("*.rejected-*"))
    assert rejected, "old-area files must be quarantined, not deleted"


def test_cli_refetch_with_different_cycle_refuses(tmp_path, monkeypatch,
                                                  capsys):
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    tail = ["--hours", "3", "--area", "30,-100,40,-90", "--out", str(out)]
    assert cli.main(["fetch", "--source", "gfs",
                     "--cycle", "2026-07-28T06"] + tail) == 0
    capsys.readouterr()
    rc = cli.main(["fetch", "--source", "gfs",
                   "--cycle", "2026-07-28T12"] + tail)
    assert rc == 2
    err = capsys.readouterr().err
    assert "cycle" in err and "different request" in err


def test_refetch_hours_extension_still_resumes(tmp_path, monkeypatch,
                                               capsys):
    """Hours-only growth is byte-safe (same source/cycle/area produce
    identical per-hour files), so the window stays extendable."""
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    base = ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
            "--area", "30,-100,40,-90", "--out", str(out)]
    assert cli.main(base + ["--hours", "3"]) == 0
    capsys.readouterr()
    assert cli.main(base + ["--hours", "6"]) == 0
    printed = capsys.readouterr().out
    assert "verified -- skipped" in printed          # f000/f003 resumed
    assert "f006" in printed
    manifest = json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())
    assert manifest["forecast_hours"] == [0, 3, 6]


def test_gfs_interrupt_manifests_verified_prefix_and_resumes(
        tmp_path, monkeypatch, capsys):
    """Ctrl-C is an orderly, byte-accurate outcome: the completed prefix
    is digest-bound, an in-flight .part is named but not blessed, and the
    printed command resumes without re-downloading good bytes."""
    def interrupted_download(url, destination, **kwargs):
        # Keyed on the hour, not on call order: the pooled transport is
        # free to schedule hours together, and the interrupt contract
        # being pinned here is per-file, not per-schedule.
        if ".f000" in destination.name:
            _fake_gfs_download(url, destination)
            return
        partial = destination.with_suffix(destination.suffix + ".part")
        partial.write_bytes(b"GRIB interrupted, not a complete envelope")
        raise KeyboardInterrupt

    monkeypatch.setattr(gfs_transport, "_download", interrupted_download)
    monkeypatch.setattr(
        fetch, "require_published_cycle",
        lambda source, cycle, last_hour, **kwargs: None)
    original_fetch = fetch.fetch_gfs

    def no_live_inventory(**kwargs):
        return original_fetch(
            **kwargs,
            derived_bar=lambda cycle, **options:
            fetch.GFS_SUBSET_RECORD_COUNT)

    monkeypatch.setattr(fetch, "fetch_gfs", no_live_inventory)
    out = tmp_path / "gfs"
    base = ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
            "--hours", "6", "--cadence", "3",
            "--area", "30,-100,40,-90", "--out", str(out)]

    rc = cli.main(base)
    assert rc == 2
    err = capsys.readouterr().err
    first = "gfs.t06z.pgrb2.0p25.f000.subset.grib2"
    partial = "gfs.t06z.pgrb2.0p25.f003.subset.grib2.part"
    assert "Verified complete GRIB files on disk" in err
    assert first in err and "sha256" in err
    assert "Unverified partial/incomplete GRIB files" in err
    assert partial in err
    assert "resume exactly with: woof fetch --source gfs" in err
    assert "--cycle 2026-07-28T06 --hours 6 --cadence 3" in err
    assert str(out) in err
    assert "Traceback" not in err

    manifest = json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())
    assert manifest["forecast_hours"] == [0]
    assert [item["name"] for item in manifest["files"]] == [
        first, "gfs-series.tsv", "SHA256SUMS", "inputs.txt",
        "prep-command.txt"]
    assert (out / "gfs-series.tsv").read_text().splitlines() == [
        f"0\t{first}\t81"]
    assert hashlib.sha256((out / first).read_bytes()).hexdigest() == (
        manifest["files"][0]["sha256"])

    resumed = []

    def finish_download(url, destination, **kwargs):
        resumed.append(destination.name)
        destination.with_suffix(destination.suffix + ".part").unlink(
            missing_ok=True)
        _fake_gfs_download(url, destination)

    monkeypatch.setattr(gfs_transport, "_download", finish_download)
    assert cli.main(base) == 0
    printed = capsys.readouterr().out
    assert f"{first} exists" in printed and "verified -- skipped" in printed
    assert sorted(resumed) == [
        "gfs.t06z.pgrb2.0p25.f003.subset.grib2",
        "gfs.t06z.pgrb2.0p25.f006.subset.grib2",
    ]
    manifest = json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())
    assert manifest["forecast_hours"] == [0, 3, 6]
    assert not (out / partial).exists()


@pytest.mark.parametrize("damage", ["truncated-json", "foreign-schema"])
def test_resume_refuses_a_corrupt_or_foreign_manifest(tmp_path, monkeypatch,
                                                      capsys, damage):
    """A manifest that cannot be read as this schema proves nothing;
    the files beside it are unverified for the request and refuse."""
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    base = ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
            "--hours", "3", "--area", "30,-100,40,-90", "--out", str(out)]
    assert cli.main(base) == 0
    capsys.readouterr()
    manifest = out / fetch.FETCH_MANIFEST_NAME
    if damage == "truncated-json":
        manifest.write_bytes(manifest.read_bytes()[:40])
    else:
        manifest.write_text(json.dumps({"schema": "someone-elses-v9"}),
                            encoding="utf-8")

    rc = cli.main(base)
    assert rc == 2
    err = capsys.readouterr().err
    assert "UNVERIFIED" in err and "--force-refetch" in err


def test_fetch_into_missing_or_empty_out_still_proceeds(tmp_path,
                                                        monkeypatch,
                                                        capsys):
    """The no-manifest refusal must not over-fire: an absent or empty
    --out has nothing to mis-bless and fetches normally."""
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    empty = tmp_path / "pre-made"
    empty.mkdir()
    for out in (tmp_path / "absent", empty):
        assert cli.main(
            ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
             "--hours", "3", "--area", "30,-100,40,-90",
             "--out", str(out)]) == 0
    capsys.readouterr()


def test_gfs_resume_rejects_a_swapped_file_by_manifest_digest(
        tmp_path, monkeypatch, capsys):
    """A count-valid file with different bytes under a recorded name
    must not be re-blessed: the prior manifest's sha256 binds it."""
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    base = ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
            "--hours", "3", "--area", "30,-100,40,-90", "--out", str(out)]
    assert cli.main(base) == 0
    capsys.readouterr()
    target = out / "gfs.t06z.pgrb2.0p25.f000.subset.grib2"
    swapped = bytearray(target.read_bytes())
    swapped[4] ^= 0x01  # reserved byte: still 124 valid envelopes
    target.write_bytes(bytes(swapped))
    assert fetch.count_grib2_messages(target) == 124

    rc = cli.main(base)
    assert rc == 2
    err = capsys.readouterr().err
    assert "sha256" in err and "--force-refetch" in err


# ---------------------------------------------------------------------------
# fetch -> front door manifest bridge (GFS) + HRRR handoff line
# ---------------------------------------------------------------------------

def _front_door_inputs(tmp_path):
    bridge = tmp_path / "gfs_grib2_bridge.exe"
    bridge.write_bytes(b"MZ fake bridge payload")
    wps = tmp_path / "myarea.namelist.wps"
    wps.write_text("&share\n max_dom = 1,\n/\n", encoding="utf-8")
    config = tmp_path / "myarea.toml"
    config.write_text("[experiment]\nname = 'myarea'\n", encoding="utf-8")
    return bridge, wps, config


def test_front_door_manifest_feeds_the_gfs_verifier_unedited(
        tmp_path, monkeypatch, capsys):
    """The authored manifest must satisfy the front door's own verifier
    (woof.gfs_direct._verify_input_manifest) with zero hand edits --
    schema, source identity block, role inventory, and the bridge
    executable's own sha256."""
    from woof import gfs_direct

    assert (fetch.GFS_FRONT_DOOR_MANIFEST_SCHEMA
            == gfs_direct.INPUT_MANIFEST_SCHEMA)

    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    bridge, wps, config = _front_door_inputs(tmp_path)
    rc = cli.main([
        "fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
        "--hours", "3", "--area", "30,-100,40,-90", "--out", str(out),
        "--author-front-door-manifest", "--bridge", str(bridge),
        "--wps-namelist", str(wps), "--experiment-config", str(config)])
    assert rc == 0
    manifest_path = out / fetch.GFS_INPUT_MANIFEST_NAME
    digest = fetch.sha256_file(manifest_path)
    printed = capsys.readouterr().out
    assert f"--source-manifest-sha256 {digest}" in printed
    assert "rw-wps --source gfs" in printed
    assert "--cycle 2026-07-28_06:00:00" in printed

    payload = json.loads(manifest_path.read_text())
    # The identity block also carries the pressure ladder the fetch took
    # and the source top it implies: the front door validates the case's
    # p_top against the source top BEFORE the bridge runs, so it has to
    # read it from the receipt rather than from a constant.
    assert payload["source"] == {
        "model": "GFS", "product": "pgrb2.0p25",
        "cycle": "2026-07-28T06:00:00Z",
        "pressure_levels_hpa": [float(level) for level
                                in gfs_transport.PRESSURE_LEVELS_HPA],
        "top_pressure_pa": 10000.0,
    }
    assert payload["files"]["bridge"]["sha256"] == fetch.sha256_file(bridge)

    roles = {
        "series": out / "gfs-series.tsv",
        "bridge": bridge,
        "wps_namelist": wps,
        "experiment_config": config,
        "grib-f000": out / "gfs.t06z.pgrb2.0p25.f000.subset.grib2",
        "grib-f003": out / "gfs.t06z.pgrb2.0p25.f003.subset.grib2",
    }
    verified = gfs_direct._verify_input_manifest(
        manifest_path, digest, roles)
    assert verified["schema"] == gfs_direct.INPUT_MANIFEST_SCHEMA


@pytest.mark.gpu  # `woof domain` sizes against CuPy, like the ack test
def test_front_door_command_carries_the_profile_the_config_matches(
        tmp_path, monkeypatch, capsys):
    """The printed rw-wps line names --physics-profile when it can.

    rw-wps spells the flag optional, but absent it substitutes
    WSM6_PROFILE_ID and compares the experiment's physics against THAT,
    so a pasted command with no profile refuses every config except a
    wsm6 one -- including the Morrison worked example FIRST-LIGHT.md
    itself walks through.  Found by running the documented chain end to
    end; the profile is derived through the same authority the front
    door asks (identify_single_domain_profile), so the printed command
    can never name a profile the runner would then reject.
    """
    from woof.cli import main as cli_main
    from woof.experiment import load_experiment
    from woof.physics_compat import (MORRISON_PROFILE_ID,
                                      identify_single_domain_profile)

    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    assert cli.main(["fetch", "--source", "gfs", "--cycle",
                     "2026-07-28T06", "--hours", "3", "--area",
                     "30,-100,40,-90", "--out", str(out)]) == 0

    config = tmp_path / "profiled" / "case.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    assert cli_main(["domain", "--point=39.7,-96.6", "--card", "24gb",
                     "--ladder", "12", "--source", "gfs",
                     "--cycle", "2026-07-28T06",
                     "--physics-profile", MORRISON_PROFILE_ID,
                     "--out", str(config)]) == 0
    # Guard against a vacuous pass: the emitted config must genuinely
    # resolve to the shipped profile before the printed line is judged.
    matched = identify_single_domain_profile(
        load_experiment(config).root.run)
    assert matched == MORRISON_PROFILE_ID

    bridge = tmp_path / "gfs_grib2_bridge.exe"
    bridge.write_bytes(b"MZ fake bridge payload")
    capsys.readouterr()
    rc = cli.main([
        "fetch", "--source", "gfs", "--out", str(out),
        "--author-front-door-manifest", "--bridge", str(bridge),
        "--wps-namelist", str(config.with_suffix(".namelist.wps")),
        "--experiment-config", str(config)])
    assert rc == 0
    printed = capsys.readouterr().out
    assert f"--physics-profile {MORRISON_PROFILE_ID}" in printed


def test_front_door_author_only_mode_converts_an_existing_fetch(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    assert cli.main(["fetch", "--source", "gfs", "--cycle",
                     "2026-07-28T06", "--hours", "3", "--area",
                     "30,-100,40,-90", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    # A plain fetch names the authoring step, so nobody has to read
    # source to learn the front door needs a manifest.
    assert "--author-front-door-manifest" in printed

    bridge, wps, config = _front_door_inputs(tmp_path)
    rc = cli.main([
        "fetch", "--source", "gfs", "--out", str(out),
        "--author-front-door-manifest", "--bridge", str(bridge),
        "--wps-namelist", str(wps), "--experiment-config", str(config)])
    assert rc == 0
    assert (out / fetch.GFS_INPUT_MANIFEST_NAME).is_file()


def test_front_door_author_flag_contracts(tmp_path, capsys):
    def refused(needle: str, argv: list[str]) -> None:
        assert cli.main(argv) == 2
        err = capsys.readouterr().err
        assert needle in err and "Traceback" not in err

    refused("--source gfs/gdas only",
            ["fetch", "--source", "hrrr", "--cycle", "2026-07-28T05",
             "--hours", "2", "--out", str(tmp_path),
             "--author-front-door-manifest"])
    # --bridge is no longer in this list: it resolves itself through
    # woof.bridges when omitted (the documented long form printed a
    # checkout-only path that no wheel install has).
    refused("requires --out plus --wps-namelist, --experiment-config",
            ["fetch", "--source", "gfs", "--out", str(tmp_path),
             "--author-front-door-manifest"])
    refused("belong to --author-front-door-manifest",
            ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
             "--hours", "3", "--area", "30,-100,40,-90",
             "--out", str(tmp_path), "--bridge", "x.exe"])
    refused("not a completed",
            ["fetch", "--source", "gfs", "--out", str(tmp_path / "empty"),
             "--author-front-door-manifest", "--bridge", "b.exe",
             "--wps-namelist", "w", "--experiment-config", "c"])


def test_cli_fetch_hrrr_prints_the_front_door_handoff(tmp_path,
                                                      monkeypatch,
                                                      capsys):
    """HRRR's front door consumes SHA256SUMS directly; the missing link
    was its own sha256, so fetch prints the complete pair."""
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    rc = cli.main(["fetch", "--source", "hrrr", "--engine", "python", "--cycle",
                   "2026-07-28T05", "--hours", "1", "--transport", "s3",
                   "--out", str(out)])
    assert rc == 0
    printed = capsys.readouterr().out
    digest = fetch.sha256_file(out / "SHA256SUMS")
    assert f"--source-manifest-sha256 {digest}" in printed
    # The handoff used to end in a literal `...` on a command line, and
    # the consumer refuses it: `unrecognized arguments: ...`.  What it
    # prints now is a bound fragment plus comments, so nothing it
    # produces fails the moment it is pasted.
    assert "..." not in printed
    lines = printed.splitlines()
    bound = [line for line in lines
             if line.strip().startswith("woof prep ")]
    assert len(bound) == 1
    assert "--source hrrr" in bound[0]
    assert f"--source-root {out.resolve()}" in bound[0]
    assert f"--source-manifest {out.resolve() / 'SHA256SUMS'}" in bound[0]
    comments = " ".join(l for l in lines if l.strip().startswith("#"))
    for flag in ("--domain-spec", "--namelist-input", "--wps-namelist",
                 "--geog-root", "--experiment-config", "--valid-time",
                 "--output-root"):
        assert flag in comments, flag


def test_the_hrrr_handoff_runs_at_the_door_with_exactly_the_flags_it_names(
        tmp_path, monkeypatch, capsys):
    """The printed line, completed with the flags its comment names, runs.

    A world preparation pass built its HRRR argv from this handoff and
    the door refused every tile: ``invalid or missing run arguments:
    --namelist-input, --domain-spec (required with --geog-root)``.  The
    comment named four of the six flags the door needs.  So the check is
    the door itself: the bound line plus every flag the comment names,
    valued from the shipped HRRR demo set, through ``woof prep
    --dry-run``, which validates the whole argument contract and stops
    before any input is opened.
    """
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    assert cli.main(["fetch", "--source", "hrrr", "--engine", "python",
                     "--cycle", "2026-07-28T05", "--hours", "1",
                     "--transport", "s3", "--out", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    bound = [line.strip() for line in lines
             if line.strip().startswith("woof prep ")]
    assert len(bound) == 1
    argv = shlex.split(bound[0])[2:]
    comments = " ".join(line.strip().lstrip("#").strip() for line in lines
                        if line.strip().startswith("#"))
    named = re.search(r"cannot bind the run's own flags: (.*?) are yours",
                      comments)
    assert named is not None, comments
    flags = re.findall(r"--[a-z][a-z-]*", named.group(1))
    assert flags == list(fetch.HRRR_CALLER_SUPPLIES)
    demo = Path(__file__).resolve().parent.parent / "configs"
    values = {
        "--domain-spec": demo / "hrrr_native_quick_demo.d01-target.json",
        "--namelist-input": demo / "hrrr_native_quick_demo.namelist.input",
        "--wps-namelist": demo / "hrrr_native_quick_demo.namelist.wps",
        "--experiment-config": demo / "hrrr_native_quick_demo.toml",
        "--geog-root": tmp_path / "geog",
        "--output-root": tmp_path / "prepared",
    }
    completed = argv + [token for flag in flags
                        for token in (flag, str(values[flag]))]
    rc = cli.main(["prep", *completed, "--dry-run"])
    said = capsys.readouterr()
    assert "invalid or missing run arguments" not in said.err, said.err
    assert rc == 0, said.err
    # One flag fewer is refused by name: every named flag is one the
    # door needs, not decoration.
    for needed in ("--domain-spec", "--namelist-input"):
        cut = completed[:completed.index(needed)] + \
            completed[completed.index(needed) + 2:]
        assert cli.main(["prep", *cut, "--dry-run"]) != 0
        assert needed in capsys.readouterr().err


# ---------------------------------------------------------------------------
# HRRR NOMADS transport: URLs, auto resolution, manifest, resume, wait mode
# ---------------------------------------------------------------------------

def test_hrrr_object_url_speaks_both_transports():
    cycle = datetime(2026, 7, 28, 5)
    assert fetch.hrrr_object_url(cycle, 2, "wrfnat") == (
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20260728/conus/"
        "hrrr.t05z.wrfnatf02.grib2")
    assert fetch.hrrr_object_url(cycle, 2, "wrfprs",
                                 transport="nomads") == (
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/"
        "hrrr.20260728/conus/hrrr.t05z.wrfprsf02.grib2")
    with pytest.raises(ValueError, match="transport"):
        fetch.hrrr_object_url(cycle, 2, "wrfnat", transport="ftp")


def test_resolve_hrrr_transport_auto_takes_the_mirror_once_it_has_the_window():
    """Caught-up mirror, mirrored transfer.

    Both hosts serve identical bytes.  The operational server's whole
    advantage is having the cycle FIRST; once the archive has it too,
    that advantage is spent and what is left is throughput, which the
    archive wins -- measured at peak hours, ~3 MB/s per file against
    the archive serving the same 3.4 GB in about a sixth of the wall
    clock.
    """
    cycle = datetime(2026, 7, 28, 5)
    now = datetime(2026, 7, 28, 8)
    probed = []

    def probe(url):
        probed.append(url)
        return True

    lines = []
    got = fetch.resolve_hrrr_transport(
        cycle, "auto", last_hour=6, now=now, probe=probe,
        progress=lines.append)
    assert got == "s3"
    # The throughput rung is asked FIRST, and completeness mirrors
    # resolve_latest_cycle: EVERY final-hour object a lead's fetch reads.
    assert probed == [
        fetch.hrrr_object_url(cycle, 6, "wrfnat", transport="s3"),
        fetch.hrrr_object_url(cycle, 6, "wrfprs", transport="s3"),
        fetch.hrrr_object_url(cycle, 6, "wrfsfc", transport="s3")]
    assert any("mirrored" in line and "throughput" in line
               for line in lines), lines
    assert not any("publishes" in line for line in lines), lines


def test_resolve_hrrr_transport_auto_stays_on_nomads_while_the_mirror_lags():
    """Publication lag is the one thing the operational server is for.

    The freshness sentence is now EARNED: it prints only when the
    mirror was asked and did not have the window.
    """
    cycle = datetime(2026, 7, 28, 5)
    now = datetime(2026, 7, 28, 8)
    lines = []
    got = fetch.resolve_hrrr_transport(
        cycle, "auto", last_hour=6, now=now,
        probe=lambda url: url.startswith(fetch.HRRR_NOMADS_BASE),
        progress=lines.append)
    assert got == "nomads"
    assert any("using nomads" in line and "before the mirrors" in line
               for line in lines), lines
    assert not any("mirrored" in line for line in lines), lines


@pytest.mark.parametrize("index_only", [False, True])
def test_native_auto_transport_uses_one_host_for_every_prefix_object(index_only):
    cycle = datetime(2026, 7, 28, 5)
    now = datetime(2026, 7, 28, 8)
    leads = (15, 16, 17, 18)
    asked = []
    missing = fetch.hrrr_object_url(cycle, 17, "wrfprs", transport="s3")
    if index_only:
        missing += ".idx"

    def probe(url):
        asked.append(url)
        # S3's final pair is up, while an intermediate soil object or
        # its index still lags. NOMADS serves the entire requested set.
        return url != missing

    assert fetch.resolve_hrrr_transport(
        cycle, "auto", last_hour=18, now=now, probe=probe,
        required_leads=leads, progress=lambda *_: None) == "nomads"
    assert missing in asked
    expected = {fetch.hrrr_object_url(cycle, lead, product, transport="nomads")
                + suffix for lead in leads for product in fetch.hrrr_lead_products()
                for suffix in ("", ".idx")}
    assert expected <= set(asked)


def test_native_auto_transport_still_prefers_a_complete_s3_prefix():
    cycle = datetime(2026, 7, 28, 5)
    now = datetime(2026, 7, 28, 8)
    leads = (15, 16, 17, 18)
    asked = []
    assert fetch.resolve_hrrr_transport(
        cycle, "auto", last_hour=18, now=now,
        probe=lambda url: asked.append(url) or True,
        required_leads=leads, progress=lambda *_: None) == "s3"
    assert set(asked) == {
        fetch.hrrr_object_url(cycle, lead, product, transport="s3") + suffix
        for lead in leads for product in fetch.hrrr_lead_products()
        for suffix in ("", ".idx")}


@pytest.mark.parametrize("missing", ["wrfprs", "wrfprs.idx", "wrfsfc",
                                     "everything"])
def test_native_auto_transport_never_falls_back_to_an_unverified_prefix(missing):
    """Every host said an object of the prefix is missing: none is taken.

    ``wrfsfc``: NOAA posts it up to ~10 min after the lead's wrfnat and
    wrfprs, and the fetch reads VEGFRA from it; a probe that skipped it
    called the lead posted and the run died on the absent object.
    """

    cycle = datetime(2026, 7, 28, 5)
    product = "wrfsfc" if missing == "wrfsfc" else "wrfprs"
    lagging = {fetch.hrrr_object_url(cycle, 17, product, transport=name)
               + (".idx" if missing == "wrfprs.idx" else "")
               for name in ("s3", "nomads")}

    def probe(url):
        return False if missing == "everything" or url in lagging else True

    with pytest.raises(fetch._HrrrPrefixNotPosted, match="missing intermediate"):
        fetch.resolve_hrrr_transport(
            cycle, "auto", last_hour=18,
            now=datetime(2026, 7, 28, 8), required_leads=(15, 16, 17, 18),
            probe=probe, progress=lambda *_: None)


@pytest.mark.parametrize("unheard, missing", [
    ("nomads", "s3"), ("s3", "nomads"), ("s3", None)])
def test_native_auto_transport_lets_the_transfer_try_a_host_not_heard(
        unheard, missing):
    """GS-05 on the posted prefix: a probe with no answer is not a missing object.

    One host could not be heard about an intermediate soil index (a HEAD
    it refused and a GET it did not answer), the other said the index is
    missing or could not be heard either.  The unheard host is taken and
    its transfer's own index, inventory and digest checks decide; a host
    that said an object is missing is never taken.  Reading the unheard
    answer as missing made a host that refuses HEAD wait out the whole
    posting budget and exit 75 where the final-lead choice downloaded.
    """

    cycle = datetime(2026, 7, 28, 5)
    index = {name: fetch.hrrr_object_url(cycle, 17, "wrfprs", transport=name)
             + ".idx" for name in ("s3", "nomads")}

    def probe(url):
        if missing is not None and url == index[missing]:
            return False
        return None if url in (index[unheard], index["nomads"]
                               if missing is None else None) else True

    said = []
    assert fetch.resolve_hrrr_transport(
        cycle, "auto", last_hour=18, now=datetime(2026, 7, 28, 8),
        required_leads=(15, 16, 17, 18), probe=probe,
        progress=said.append) == unheard
    assert any(f"{unheard} could not be heard" in line
               and "own checks decide" in line for line in said), said


def test_native_auto_transport_waits_again_before_any_unverified_download(monkeypatch):
    cycle = datetime(2026, 7, 28, 5)
    leads = (15, 16, 17, 18)
    calls = []

    def resolve(got_cycle, requested, *, last_hour, required_leads):
        assert (got_cycle, requested, last_hour, required_leads) \
            == (cycle, "auto", 18, leads)
        calls.append("resolve")
        if calls.count("resolve") == 1:
            raise fetch._HrrrPrefixNotPosted("the host has not served the prefix")
        return "nomads"

    class Loop:
        def wait_again(self, waited, error):
            assert waited == (18,) and isinstance(error, fetch._HrrrPrefixNotPosted)
            calls.append("posting_wait")

        def __call__(self, lead):
            assert lead == 18
            calls.append("lead_gate")

    monkeypatch.setattr(fetch, "resolve_hrrr_transport", resolve)
    assert fetch._posted_hrrr_transport(Loop(), cycle, leads) == "nomads"
    assert calls == ["resolve", "posting_wait", "lead_gate", "resolve"]


def test_resolve_hrrr_transport_auto_falls_through_to_the_archive():
    """A window the operational server has not finished still fetches.

    The ladder is an order, not a pin: the last rung is the archive and
    it is reached without a probe, because probing it would only
    duplicate the refusal the transfer itself would give.
    """
    cycle = datetime(2026, 7, 28, 5)
    now = datetime(2026, 7, 28, 8)
    lines = []
    got = fetch.resolve_hrrr_transport(
        cycle, "auto", last_hour=6, now=now, probe=lambda url: False,
        progress=lines.append)
    assert got == "s3"
    assert any("does not serve" in line and "asking s3" in line
               for line in lines)


def test_resolve_hrrr_transport_skips_the_probe_for_archived_cycles():
    """Beyond NOMADS retention there is nothing to probe: auto goes
    straight to the S3 archive."""
    cycle = datetime(2026, 7, 20, 0)
    now = datetime(2026, 7, 28, 8)

    def explode(url):
        raise AssertionError("an archived cycle must not probe NOMADS")

    lines = []
    got = fetch.resolve_hrrr_transport(
        cycle, "auto", last_hour=6, now=now, probe=explode,
        progress=lines.append)
    assert got == "s3"
    assert any("retention" in line for line in lines)


def test_resolve_hrrr_transport_explicit_contracts():
    cycle = datetime(2026, 7, 20, 0)
    now = datetime(2026, 7, 28, 8)

    def explode(url):
        raise AssertionError("--transport s3 must not probe anything")

    assert fetch.resolve_hrrr_transport(
        cycle, "s3", last_hour=6, now=now, probe=explode) == "s3"
    with pytest.raises(ValueError, match="48 h"):
        fetch.resolve_hrrr_transport(
            cycle, "nomads", last_hour=6, now=now,
            probe=lambda url: False)
    with pytest.raises(ValueError, match="transport"):
        fetch.resolve_hrrr_transport(cycle, "ftp", last_hour=6, now=now)


def test_fetch_hrrr_nomads_transport_keeps_the_contracts(tmp_path,
                                                         monkeypatch):
    """Same 561/18 record bars, same manifest discipline -- only the
    host differs, and the manifest records it per file."""
    runtime_calls = _install_runtime_surface_inputs(monkeypatch)
    seen = []

    def product(request, *, workers, retries, expected_count=-1):
        seen.append(request)
        return _fake_hrrr_product(request, workers=workers,
                                  retries=retries,
                                  expected_count=expected_count)

    monkeypatch.setattr(hrrr_transport, "_download_product", product)
    out = tmp_path / "hrrr"
    manifest_path = fetch.fetch_hrrr(
        cycle=datetime(2026, 7, 28, 5), hours=(0, 1), area=None, out=out,
        transport="nomads", progress=lambda line: None)

    prefix = fetch.HRRR_NOMADS_BASE + "/hrrr.20260728/conus/"
    # Both products of both leads are requested, each exactly once.  The
    # order they are DISPATCHED in belongs to the download pool, not to
    # this contract, and pinning it failed about one run in five; the
    # manifest order below is the ordering this fetch does promise.
    assert sorted((request.kind, request.url) for request in seen) == sorted(
        (item["role"], item["url"])
        for item in json.loads(manifest_path.read_text())["files"]
        if item["role"] in ("atmosphere", "soil"))
    assert sorted(request.kind for request in seen) == [
        "atmosphere", "atmosphere", "soil", "soil"]
    assert all(request.url.startswith(prefix) for request in seen)
    assert all(request.index_url == request.url + ".idx"
               for request in seen)
    manifest = json.loads(manifest_path.read_text())
    _assert_runtime_surface_requests(runtime_calls, manifest, datetime(2026, 7, 28, 5))
    assert [item["role"] for item in manifest["files"]] == [
        "atmosphere", "soil", "atmosphere", "soil", "checksums"]
    assert [item.get("transport") for item in manifest["files"]] == [
        "nomads"] * 4 + [None]
    assert "byte-identical" in manifest["notes"]
    for item in manifest["files"][:-1]:
        assert item["sha256"] == hashlib.sha256(
            (out / item["name"]).read_bytes()).hexdigest()


def test_fetch_hrrr_resume_carries_across_transports(tmp_path,
                                                     monkeypatch):
    """A directory fetched over S3 resumes over NOMADS: the digest and
    record-count bars are host-independent because the files are
    byte-identical on both hosts."""
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    kwargs = dict(cycle=datetime(2026, 7, 28, 5), hours=(0,), area=None,
                  out=out)
    fetch.fetch_hrrr(**kwargs, transport="s3", progress=lambda line: None)

    def refuse(request, *, workers, retries):
        raise AssertionError("re-downloaded a file the digests verify")

    monkeypatch.setattr(hrrr_transport, "_download_product", refuse)
    lines: list[str] = []
    fetch.fetch_hrrr(**kwargs, transport="nomads", progress=lines.append)
    assert len(lines) == 2 and all("skipped" in line for line in lines)


def test_fetch_hrrr_refuses_unresolved_auto_outside_wait_mode(tmp_path):
    with pytest.raises(ValueError, match="resolve_hrrr_transport"):
        fetch.fetch_hrrr(
            cycle=datetime(2026, 7, 28, 5), hours=(0,), area=None,
            out=tmp_path / "hrrr", transport="auto",
            progress=lambda line: None)


class _LivePublication:
    """Fake clock/sleep/probe harness: a URL answers once the fake time
    reaches its scheduled publication instant."""

    def __init__(self, schedule):
        #: {(product, hour, transport): publish_time}; absent = never.
        self.schedule = schedule
        self.now = 0.0
        self.sleeps: list[float] = []
        self.probed: list[str] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        # The polling contract: never sleep past the 30 s cap.
        assert 0 < seconds <= fetch.HRRR_WAIT_POLL_SECONDS
        self.sleeps.append(seconds)
        self.now += seconds

    def probe(self, url: str) -> bool:
        import re

        self.probed.append(url)
        name = url[:-4] if url.endswith(".idx") else url
        transport = ("nomads" if name.startswith(fetch.HRRR_NOMADS_BASE)
                     else "s3")
        product = "wrfnat" if "wrfnat" in name else "wrfprs"
        hour = int(re.search(r"f(\d\d)\.grib2$", name).group(1))
        when = self.schedule.get((product, hour, transport))
        return when is not None and self.now >= when


def test_fetch_hrrr_wait_downloads_hours_as_they_publish(tmp_path,
                                                         monkeypatch):
    """The live-cycle mode: f00 is up immediately, f01 publishes 45
    fake-seconds in; both arrive from NOMADS without a single real
    sleep, and the finished manifest is the normal complete one."""
    runtime_calls = _install_runtime_surface_inputs(monkeypatch)
    live = _LivePublication({
        ("wrfnat", 0, "nomads"): 0.0, ("wrfprs", 0, "nomads"): 0.0,
        ("wrfnat", 1, "nomads"): 45.0, ("wrfprs", 1, "nomads"): 45.0,
    })
    downloaded = []

    def product(request, *, workers, retries, expected_count=-1):
        downloaded.append(request.url)
        return _fake_hrrr_product(request, workers=workers,
                                  retries=retries,
                                  expected_count=expected_count)

    monkeypatch.setattr(hrrr_transport, "_download_product", product)
    out = tmp_path / "hrrr"
    manifest_path = fetch.fetch_hrrr(
        cycle=datetime(2026, 7, 28, 5), hours=(0, 1), area=None, out=out,
        transport="auto", wait=True, wait_timeout_s=3600.0,
        probe=live.probe, sleeper=live.sleep, clock=live.clock,
        progress=lambda line: None)

    assert len(downloaded) == 4
    assert all(url.startswith(fetch.HRRR_NOMADS_BASE)
               for url in downloaded)
    assert live.sleeps, "f01 was not up at t=0, so the fetch had to poll"
    manifest = json.loads(manifest_path.read_text())
    _assert_runtime_surface_requests(runtime_calls, manifest, datetime(2026, 7, 28, 5))
    assert manifest["forecast_hours"] == [0, 1]
    assert [item.get("transport") for item in manifest["files"]] == [
        "nomads"] * 4 + [None]


def test_fetch_hrrr_wait_auto_falls_back_per_product(tmp_path,
                                                     monkeypatch):
    """Under auto, every poll round tries NOMADS first and S3 second,
    so a product NOMADS never serves still arrives -- from S3."""
    runtime_calls = _install_runtime_surface_inputs(monkeypatch)
    live = _LivePublication({
        ("wrfnat", 0, "nomads"): 0.0, ("wrfprs", 0, "nomads"): 0.0,
        ("wrfnat", 1, "nomads"): 30.0,
        ("wrfprs", 1, "s3"): 60.0,  # never on NOMADS
    })
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    manifest_path = fetch.fetch_hrrr(
        cycle=datetime(2026, 7, 28, 5), hours=(0, 1), area=None, out=out,
        transport="auto", wait=True, wait_timeout_s=3600.0,
        probe=live.probe, sleeper=live.sleep, clock=live.clock,
        progress=lambda line: None)
    manifest = json.loads(manifest_path.read_text())
    _assert_runtime_surface_requests(runtime_calls, manifest, datetime(2026, 7, 28, 5))
    by_name = {item["name"]: item.get("transport")
               for item in manifest["files"]}
    assert by_name["hrrr.t05z.wrfnatf01.grib2"] == "nomads"
    assert by_name["hrrr.t05z.soilf01.grib2"] == "s3"


def test_fetch_hrrr_wait_polls_nomads_and_transfers_from_the_mirror(
        tmp_path, monkeypatch):
    """The poll watches the operational server; the bytes come mirrored.

    Its head start is the point of polling it, so it keeps being asked
    every round.  But once the archive has the object too, the reason
    to transfer from the slower host is gone -- and at peak hours that
    reason cost ~20 min on a 3.4 GB request.
    """
    runtime_calls = _install_runtime_surface_inputs(monkeypatch)
    live = _LivePublication({
        ("wrfnat", 0, "nomads"): 0.0, ("wrfprs", 0, "nomads"): 0.0,
        ("wrfnat", 0, "s3"): 0.0, ("wrfprs", 0, "s3"): 0.0,
        # f01 reaches NOMADS first and the mirror catches up later.
        ("wrfnat", 1, "nomads"): 30.0, ("wrfprs", 1, "nomads"): 30.0,
        ("wrfnat", 1, "s3"): 60.0, ("wrfprs", 1, "s3"): 60.0,
    })
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    manifest_path = fetch.fetch_hrrr(
        cycle=datetime(2026, 7, 28, 5), hours=(0, 1), area=None, out=out,
        transport="auto", wait=True, wait_timeout_s=3600.0,
        probe=live.probe, sleeper=live.sleep, clock=live.clock,
        progress=lambda line: None)

    # The operational server was polled -- that is its head start.
    assert any(url.startswith(fetch.HRRR_NOMADS_BASE)
               for url in live.probed)
    manifest = json.loads(manifest_path.read_text())
    _assert_runtime_surface_requests(runtime_calls, manifest, datetime(2026, 7, 28, 5))
    by_name = {item["name"]: item.get("transport")
               for item in manifest["files"]}
    # f00 is on both, so it is mirrored; f01 is on NOMADS alone when it
    # first appears, so freshness wins that one.
    assert by_name["hrrr.t05z.wrfnatf00.grib2"] == "s3"
    assert by_name["hrrr.t05z.wrfnatf01.grib2"] == "nomads"


def test_fetch_hrrr_wait_times_out_accurately_and_resumes(tmp_path,
                                                        monkeypatch):
    """The accurate timeout: the complete f00 prefix is manifested (so a
    re-run RESUMES rather than refusing at the pre-manifest-interrupt
    gate), and the error says exactly what was and was not fetched."""
    _install_runtime_surface_inputs(monkeypatch)
    schedule = {
        ("wrfnat", 0, "nomads"): 0.0, ("wrfprs", 0, "nomads"): 0.0,
        # f01 never publishes anywhere within the window.
    }
    live = _LivePublication(schedule)
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    kwargs = dict(cycle=datetime(2026, 7, 28, 5), hours=(0, 1), area=None,
                  out=out, transport="auto", wait=True,
                  probe=live.probe, sleeper=live.sleep, clock=live.clock)
    with pytest.raises(RuntimeError) as caught:
        fetch.fetch_hrrr(**kwargs, wait_timeout_s=120.0,
                         progress=lambda line: None)
    message = str(caught.value)
    assert "timed out" in message
    assert "f00..f00" in message
    assert "resumes" in message

    manifest = json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())
    assert manifest["forecast_hours"] == [0]
    names = [item["name"] for item in manifest["files"]]
    assert names == ["hrrr.t05z.wrfnatf00.grib2",
                     "hrrr.t05z.soilf00.grib2", "SHA256SUMS"]
    assert len((out / "SHA256SUMS").read_text().splitlines()) == 2

    # The cycle finishes publishing; the same command completes,
    # resuming the verified f00 files instead of re-downloading them.
    schedule[("wrfnat", 1, "nomads")] = live.now
    schedule[("wrfprs", 1, "nomads")] = live.now
    downloaded = []

    def product(request, *, workers, retries, expected_count=-1):
        downloaded.append(request.destination.name)
        return _fake_hrrr_product(request, workers=workers,
                                  retries=retries,
                                  expected_count=expected_count)

    monkeypatch.setattr(hrrr_transport, "_download_product", product)
    lines: list[str] = []
    manifest_path = fetch.fetch_hrrr(**kwargs, wait_timeout_s=120.0,
                                     progress=lines.append)
    assert downloaded == ["hrrr.t05z.wrfnatf01.grib2",
                          "hrrr.t05z.soilf01.grib2"]
    assert sum("skipped" in line for line in lines) == 2
    manifest = json.loads(manifest_path.read_text())
    assert manifest["forecast_hours"] == [0, 1]


def test_cli_fetch_hrrr_transport_and_wait_contracts(tmp_path, capsys):
    def refused(needle: str, argv: list[str]) -> None:
        assert cli.main(argv) == 2
        err = capsys.readouterr().err
        assert needle in err and "Traceback" not in err

    # --transport names a rung of the endpoint ladder, and the GFS
    # container sources have one -- but only their whole-object
    # transport walks it.  The grib-filter crop is a CGI service on one
    # host, so there is no rung to name.
    refused("--mode full-file",
            ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
             "--hours", "6", "--area", "30,-100,40,-90",
             "--out", str(tmp_path), "--transport", "s3"])
    # --wait-for is the as-posted spelling for every source (A136 L2),
    # and --wait-timeout-minutes the as-posted window cap, so what is
    # refused is a wait budget beside --whole-cycle, which waits for nothing.
    refused("--whole-cycle waits for nothing",
            ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
             "--hours", "6", "--area", "30,-100,40,-90",
             "--out", str(tmp_path), "--whole-cycle",
             "--late-after-minutes", "30"])
    refused("--whole-cycle waits for nothing",
            ["fetch", "--source", "hrrr", "--cycle", "2026-07-28T05",
             "--hours", "2", "--out", str(tmp_path), "--whole-cycle",
             "--wait-timeout-minutes", "10"])
    refused("must be positive",
            ["fetch", "--source", "hrrr", "--cycle", "2026-07-28T05",
             "--hours", "2", "--out", str(tmp_path), "--wait-for",
             "--wait-timeout-minutes", "0"])


def test_cli_fetch_hrrr_auto_resolves_the_transport_once(tmp_path,
                                                         monkeypatch,
                                                         capsys):
    """The CLI's plain (non-wait) path resolves 'auto' to one concrete
    transport up front and hands fetch_hrrr the result."""
    calls = {}

    def resolve(cycle, requested, *, last_hour, **kwargs):
        calls["args"] = (cycle, requested, last_hour)
        return "s3"

    monkeypatch.setattr(fetch, "resolve_hrrr_transport", resolve)
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    rc = cli.main(["fetch", "--source", "hrrr", "--engine", "python", "--cycle",
                   "2026-07-28T05", "--hours", "1", "--out", str(out)])
    assert rc == 0
    assert calls["args"] == (datetime(2026, 7, 28, 5), "auto", 1)
    manifest = json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())
    assert {item.get("transport")
            for item in manifest["files"]} == {"s3", None}


def test_cli_fetch_hrrr_wait_timeout_is_an_orderly_exit(tmp_path,
                                                        monkeypatch,
                                                        capsys):
    """A --wait-for window that never fills is an operational outcome
    with a resume story -- exit 75 (the source fell behind, A136 L2) and
    the message naming the hour and the resume, never a traceback."""
    from woof import fetch_endpoints

    monkeypatch.setattr(fetch, "_head_ok", lambda url: False)
    monkeypatch.setattr(fetch, "_head_answer", lambda url: False)
    # The publication check's own question says no too (the fixture's
    # hosts say yes), so the window is not taken as posted whole.
    monkeypatch.setattr(fetch_endpoints, "object_answer",
                        lambda url, **kwargs: False)
    out = tmp_path / "hrrr"
    rc = cli.main(["fetch", "--source", "hrrr", "--engine", "python", "--cycle",
                   "2026-07-28T05", "--hours", "1", "--out", str(out),
                   "--wait-for", "--wait-timeout-minutes", "0.005"])
    assert rc == 75
    err = capsys.readouterr().err
    assert "f000 of the 2026-07-28T05 cycle" in err
    assert "resumes from the fetched prefix" in err
    assert "Traceback" not in err
    failed = json.loads((out / "posting" / "failed.json").read_text())
    assert failed["code"] == "source_behind" and failed["lead"] == 0


# ---------------------------------------------------------------------------
# Optional live smoke: one small .idx object (skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("WOOF_NETWORK_TESTS"),
                    reason="live network smoke; set WOOF_NETWORK_TESTS=1")
def test_live_latest_hrrr_index_is_a_canonical_noaa_index():
    from urllib.request import Request, urlopen

    cycle = fetch.resolve_latest_cycle("hrrr", 1)
    url = fetch.hrrr_object_url(cycle, 0, "wrfnat") + ".idx"
    request = Request(url, headers={"User-Agent": "gpuwm-fetch-test/1"})
    with urlopen(request, timeout=120) as response:
        first = response.read(4096).decode("ascii").splitlines()[0]
    sequence, offset, _rest = first.split(":", 2)
    assert (sequence, offset) == ("1", "0")


def test_a_named_incomplete_cycle_refuses_with_the_complete_one(monkeypatch):
    """A development machine #7: 20 lines of raw urllib.HTTPError for an early cycle.

    `--cycle 2026-07-30T00` an hour before that cycle exists dumped a
    traceback from inside the downloader, in a product that names the
    remedy for almost everything else.  The completeness probe already
    existed for `--cycle latest`; it just never ran for a named one.
    """
    from datetime import datetime
    import woof.fetch as fetch_module

    now = datetime(2026, 7, 30, 1, 30)
    published = datetime(2026, 7, 29, 18)
    complete = {
        url for url in fetch_module.cycle_probe_urls("gfs", published, 6)}

    def probe(url):
        # Only cycles at or before 18Z on the 29th exist.
        return any(url.startswith(u[:u.rindex("/")]) and u == url
                   for u in complete) or _older(url)

    def _older(url):
        for back in range(1, 9):
            older = published - timedelta(hours=6 * back)
            if url in fetch_module.cycle_probe_urls("gfs", older, 6):
                return True
        return False

    from datetime import timedelta
    # A published cycle passes silently.
    fetch_module.require_published_cycle(
        "gfs", published, 6, now=now, probe=probe)

    with pytest.raises(RuntimeError) as excinfo:
        fetch_module.require_published_cycle(
            "gfs", datetime(2026, 7, 30, 0), 6, now=now, probe=probe)
    message = str(excinfo.value)
    assert "2026-07-30T00Z is not published through f006 yet" in message
    assert "newest complete GFS cycle" in message
    assert "2026-07-29T18Z" in message
    assert "--cycle latest" in message


# ---------------------------------------------------------------------------
# The bridge the documented long form could not name on a wheel
# ---------------------------------------------------------------------------

def test_front_door_manifest_resolves_the_bridge_when_none_is_given(
        tmp_path, monkeypatch, capsys):
    """`front-door manifest inputs are missing: bridge:
    tools/grib1_bridge/target/release/gfs_grib2_bridge`

    FIRST-LIGHT step 3a printed that path.  It exists in a checkout and
    in no wheel install, so a person following the stage-by-stage route
    from pip met this after paying for the fetch -- while `woof go`,
    running the same stage, had resolved the bridge through
    woof.bridges all along.  Omitting --bridge now asks the same
    resolver.
    """

    from woof import bridges

    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    bridge, wps, config = _front_door_inputs(tmp_path)
    monkeypatch.setattr(
        bridges, "find_bridge",
        lambda name: bridge if name == "gfs_grib2_bridge" else None)

    assert cli.main([
        "fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
        "--hours", "3", "--area", "30,-100,40,-90", "--out", str(out),
        "--author-front-door-manifest",
        "--wps-namelist", str(wps), "--experiment-config", str(config)]) == 0
    payload = json.loads(
        (out / fetch.GFS_INPUT_MANIFEST_NAME).read_text(encoding="utf-8"))
    # The resolved executable is what the manifest BINDS, so the rw-wps
    # line printed next launches the binary this manifest was sealed
    # against -- not a second one the reader had to name correctly.
    assert payload["files"]["bridge"]["sha256"] == fetch.sha256_file(bridge)
    printed = capsys.readouterr().out
    assert "--bridge " in printed
    assert bridge.name in printed.split("--bridge ", 1)[1].split()[0]


def test_an_unresolvable_bridge_names_fetch_bridges_not_a_missing_path(
        tmp_path, monkeypatch, capsys):
    """Negative control: resolution failing is still one sentence.

    An install with no built decoder anywhere gets the command that
    stages one, rather than "inputs are missing: bridge: None".
    """

    from woof import bridges

    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    _bridge, wps, config = _front_door_inputs(tmp_path)
    assert cli.main(["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
                     "--hours", "3", "--area", "30,-100,40,-90",
                     "--out", str(out)]) == 0
    capsys.readouterr()
    monkeypatch.setattr(bridges, "find_bridge", lambda name: None)
    assert cli.main([
        "fetch", "--source", "gfs", "--out", str(out),
        "--author-front-door-manifest",
        "--wps-namelist", str(wps), "--experiment-config", str(config)]) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "woof fetch-bridges" in err
    assert "gfs_grib2_bridge" in err


def test_fetch_hrrr_pools_products_and_receipts_the_concurrency(
        tmp_path, monkeypatch):
    """The HRRR route rides the default pool in plain (non-wait) mode:
    the manifest receipts the run, and the per-complete-hour checkpoint
    publications keep their exact serial semantics."""

    from woof import fetch_pool

    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    published = []
    original_write = fetch.write_fetch_manifest

    def record_write(out, payload):
        published.append(payload["forecast_hours"])
        return original_write(out, payload)

    monkeypatch.setattr(fetch, "write_fetch_manifest", record_write)
    manifest_path = fetch.fetch_hrrr(
        cycle=datetime(2026, 7, 28, 5), hours=(0, 1), area=None,
        out=tmp_path / "hrrr", progress=lambda line: None)
    manifest = json.loads(manifest_path.read_text())
    receipt = manifest["concurrency"]
    assert receipt["workers_requested"] == fetch_pool.DEFAULT_FILE_WORKERS
    assert receipt["workers_effective"] == 4   # two hours x two products
    assert receipt["files"] == 4
    assert published == [[0], [0, 1], [0, 1]]
