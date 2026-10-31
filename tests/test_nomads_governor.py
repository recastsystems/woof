"""The Python half of the cross-process NOMADS rate governor.

The audited hole was that ``woof fetch --engine python`` -- which is
also what ``--engine auto`` falls back to on a machine where the Rust
backbone is not built -- went straight to :mod:`urllib` while believing
a governor was in force.  Two Python fetches had no shared pacing at
all, and the HRRR range transport could put eight concurrent requests
on the wire.

These gates prove three things: that the Python governor speaks the
*same* on-disk protocol as the Rust client (so the two pace each other
rather than each other's shadow), that its ambiguity cases resolve
towards waiting, and that the transports actually route through it.

No live request is issued anywhere in this file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from woof import nomads_governor as governor

NOMADS = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=x"
S3 = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260728/06/atmos/x"


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    state = tmp_path / "rustwx_nomads_rate_limit.state"
    monkeypatch.setenv(governor.STATE_PATH_ENV, str(state))
    monkeypatch.delenv(governor.REQUEST_LOG_ENV, raising=False)
    return state


class _Clock:
    """A sleeper that records instead of sleeping, and advances time."""

    def __init__(self, monkeypatch, start_ms: int = 1_000_000_000_000):
        self.now_ms = start_ms
        self.slept: list[float] = []
        monkeypatch.setattr(governor, "_now_ms", lambda: self.now_ms)

    def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now_ms += int(seconds * 1000)


# ---------------------------------------------------------------------------
# Protocol compatibility with the Rust client
# ---------------------------------------------------------------------------

def test_state_and_lock_paths_match_the_rust_client(tmp_path, monkeypatch):
    """Same file, or the two governors pace nothing in common."""

    monkeypatch.delenv(governor.STATE_PATH_ENV, raising=False)
    import tempfile
    assert governor.state_path() == (
        Path(tempfile.gettempdir()) / "rustwx_nomads_rate_limit.state")
    # `Path::with_extension("lock")` / `with_extension("tmp")` on the
    # Rust side; the same two names have to come out here.
    state = Path(tempfile.gettempdir()) / "rustwx_nomads_rate_limit.state"
    assert governor._lock_path(state).name == "rustwx_nomads_rate_limit.lock"
    assert governor._tmp_path(state).name == "rustwx_nomads_rate_limit.tmp"


def test_defaults_match_the_rust_constants(monkeypatch):
    monkeypatch.delenv(governor.MIN_INTERVAL_ENV, raising=False)
    monkeypatch.delenv(governor.COOLDOWN_ENV, raising=False)
    assert governor.min_interval_ms() == 2500
    assert governor.cooldown_ms() == 15 * 60 * 1000
    assert governor.LOCK_STALE_AFTER_MS == 120 * 1000


@pytest.mark.parametrize("interval,expected", [
    ("500", 500), ("9000", 9000), ("0", 2500), ("-1", 2500),
    ("junk", 2500),
])
def test_min_interval_override_is_read_the_rust_way(monkeypatch, interval,
                                                    expected):
    """Rust filters non-positive and unparseable values to the default."""

    monkeypatch.setenv(governor.MIN_INTERVAL_ENV, interval)
    assert governor.min_interval_ms() == expected


def test_state_written_here_is_readable_by_the_rust_parser(isolated_state):
    governor.write_state(isolated_state, 111, 222)
    text = isolated_state.read_text(encoding="utf-8")
    assert text == "last_request_ms=111\ncooldown_until_ms=222\n"
    assert governor.read_state(isolated_state) == (111, 222)


def test_state_written_by_rust_is_read_here(isolated_state):
    # Exactly what wx-core's write_nomads_state emits.
    isolated_state.write_text(
        "last_request_ms=1700000000000\ncooldown_until_ms=0\n",
        encoding="utf-8")
    assert governor.read_state(isolated_state) == (1700000000000, 0)


# ---------------------------------------------------------------------------
# Pacing
# ---------------------------------------------------------------------------

def test_non_nomads_urls_are_never_paced(monkeypatch, isolated_state):
    clock = _Clock(monkeypatch)
    governor.pace(S3, sleeper=clock)
    assert clock.slept == []
    assert not isolated_state.exists()


@pytest.mark.parametrize("gap_ms", [2500, 400])
def test_a_second_request_waits_out_the_recorded_gap(monkeypatch,
                                                     isolated_state, gap_ms):
    monkeypatch.setenv(governor.MIN_INTERVAL_ENV, str(gap_ms))
    clock = _Clock(monkeypatch)

    governor.pace(NOMADS, sleeper=clock)           # first: nothing recorded
    assert clock.slept == []
    last, _cooldown = governor.read_state(isolated_state)
    assert last == clock.now_ms

    clock.now_ms += 10                             # 10 ms later
    governor.pace(NOMADS, sleeper=clock)
    assert clock.slept and abs(
        sum(clock.slept) - (gap_ms - 10) / 1000.0) < 1e-6
    assert governor.read_state(isolated_state)[0] == clock.now_ms


def test_a_cooldown_outlasts_the_gap(monkeypatch, isolated_state):
    monkeypatch.setenv(governor.MIN_INTERVAL_ENV, "100")
    clock = _Clock(monkeypatch)
    governor.write_state(isolated_state, clock.now_ms,
                         clock.now_ms + 60_000)
    governor.pace(NOMADS, sleeper=clock)
    assert abs(sum(clock.slept) - 60.0) < 0.2


def test_mark_rate_limited_records_a_node_wide_cooldown(monkeypatch,
                                                        isolated_state):
    monkeypatch.setenv(governor.COOLDOWN_ENV, "30000")
    clock = _Clock(monkeypatch)
    governor.mark_rate_limited(NOMADS, "over_rate_limit")
    _last, cooldown = governor.read_state(isolated_state)
    assert cooldown == clock.now_ms + 30_000
    # A second signal inside an existing cooldown does not extend it.
    governor.mark_rate_limited(NOMADS, "again")
    assert governor.read_state(isolated_state)[1] == cooldown
    # And a non-NOMADS host never opens one.
    isolated_state.unlink()
    governor.mark_rate_limited(S3, "irrelevant")
    assert not isolated_state.exists()


# ---------------------------------------------------------------------------
# Ambiguity resolves towards waiting (stricter than the Rust reader)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("corrupt", [
    "", "garbage without an equals sign\n", "last_request_ms=not-a-number\n",
    "\x00\x01\x02", "cooldown_until_ms=5\n",
])
def test_corrupt_state_is_read_as_just_now_not_as_empty(monkeypatch,
                                                        isolated_state,
                                                        corrupt):
    """The lie: corruption parsed as zero is permission to send.

    wx-core maps an unreadable or malformed state to ``(0, 0)``, which
    makes ``last_request + gap`` lie in 1970 and waves the request
    through.  Here a *present* file that carries no usable timestamp is
    treated as though a request had just been made.
    """

    clock = _Clock(monkeypatch)
    isolated_state.write_bytes(corrupt.encode("utf-8", errors="ignore"))
    last, _cooldown = governor.read_state(isolated_state)
    assert last == clock.now_ms

    governor.pace(NOMADS, sleeper=clock)
    assert sum(clock.slept) == pytest.approx(2.5, abs=0.01)


def test_an_absent_state_is_a_genuine_empty_state(monkeypatch,
                                                  isolated_state):
    clock = _Clock(monkeypatch)
    assert not isolated_state.exists()
    assert governor.read_state(isolated_state) == (0, 0)
    governor.pace(NOMADS, sleeper=clock)
    assert clock.slept == []


def test_a_failed_state_write_still_spaces_the_request(monkeypatch,
                                                       isolated_state):
    """The lie: an unrecorded request lets the next process send now."""

    clock = _Clock(monkeypatch)
    monkeypatch.setattr(governor, "write_state",
                        lambda *args, **kwargs: False)
    governor.pace(NOMADS, sleeper=clock)
    assert sum(clock.slept) == pytest.approx(2.5, abs=0.01)


def test_an_unusable_lock_directory_still_spaces_the_request(monkeypatch):
    clock = _Clock(monkeypatch)
    monkeypatch.setattr(governor, "_acquire",
                        lambda lock, sleeper=None: None)
    governor.pace(NOMADS, sleeper=clock)
    assert sum(clock.slept) == pytest.approx(2.5, abs=0.01)


# ---------------------------------------------------------------------------
# Cross-process: two real processes share one pacer
# ---------------------------------------------------------------------------

_PACE_CHILD = """
    import json, sys, time
    from woof import nomads_governor as governor

    url = sys.argv[1]
    stamps = []
    for _ in range(int(sys.argv[2])):
        governor.pace(url)
        stamps.append(time.time())
    print(json.dumps(stamps))
"""


def test_two_processes_share_one_pacer(tmp_path, isolated_state):
    """No live request: pacing is proved by the wall-clock spacing."""

    env = dict(os.environ)
    env[governor.MIN_INTERVAL_ENV] = "300"
    children = [
        subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(_PACE_CHILD), NOMADS, "3"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=env, cwd=str(Path(__file__).resolve().parents[1]))
        for _ in range(2)]
    stamps: list[float] = []
    for child in children:
        out, err = child.communicate(timeout=180)
        assert child.returncode == 0, err
        stamps.extend(json.loads(out.strip().splitlines()[-1]))
    stamps.sort()
    gaps = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
    assert len(stamps) == 6
    # Six requests from two independent processes, every consecutive pair
    # at least the configured gap apart (a little slack for clock
    # granularity).  Without a shared pacer the two runs interleave and
    # several gaps land near zero.
    assert min(gaps) >= 0.28, gaps


# ---------------------------------------------------------------------------
# The transports actually route through it
# ---------------------------------------------------------------------------

class _Response:
    status = 200
    headers: dict = {}

    def read(self, *_args):
        return b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_paced_urlopen_paces_then_calls(monkeypatch, isolated_state):
    order: list[str] = []
    monkeypatch.setattr(governor, "pace",
                        lambda url, **kw: order.append(f"pace:{url}"))
    governor.paced_urlopen(
        Request(NOMADS), timeout=5,
        opener=lambda request, **kw: order.append("call") or _Response())
    assert order == [f"pace:{NOMADS}", "call"]


@pytest.mark.parametrize("error,expected_cooldown", [
    (HTTPError(NOMADS, 429, "Over Rate Limit", {}, None), True),
    (HTTPError(NOMADS, 503, "unavailable", {}, None), True),
    (HTTPError(NOMADS, 404, "absent", {}, None), False),
    (URLError("missing a location header"), True),
    (URLError("connection reset"), False),
])
def test_rate_limit_answers_open_a_cooldown(monkeypatch, isolated_state,
                                            error, expected_cooldown):
    clock = _Clock(monkeypatch)
    monkeypatch.setattr(governor, "pace", lambda url, **kw: None)

    def opener(request, **kwargs):
        raise error

    with pytest.raises(type(error)):
        governor.paced_urlopen(Request(NOMADS), opener=opener)
    if expected_cooldown:
        assert governor.read_state(isolated_state)[1] > clock.now_ms
    else:
        assert not isolated_state.exists()


def test_gfs_cgi_transport_is_governed(monkeypatch, tmp_path):
    from tools import download_gfs_native_subset as transport

    seen: list[str] = []
    monkeypatch.setattr(governor, "pace",
                        lambda url, **kw: seen.append(url))

    grib = b"GRIB" + b"\x00\x00\x00\x02" + (2048).to_bytes(8, "big") \
        + b"\x00" * 2036 + b"7777"

    class _Body(_Response):
        def __init__(self):
            self._left = [grib]

        def read(self, *_args):
            return self._left.pop() if self._left else b""

    monkeypatch.setattr(transport, "paced_urlopen",
                        lambda request, **kw: governor.paced_urlopen(
                            request, opener=lambda *a, **k: _Body(), **kw))
    transport._download(NOMADS, tmp_path / "subset.grib2")
    assert seen == [NOMADS]
    assert (tmp_path / "subset.grib2").read_bytes() == grib
    # No canonical `.part` survives, and no fixed one was ever created.
    assert [path.name for path in tmp_path.iterdir()] == ["subset.grib2"]


@pytest.mark.parametrize("url,requested,expected", [
    (NOMADS, 8, 1), (NOMADS, 1, 1), (S3, 8, 8), (S3, 2, 2),
])
def test_range_pool_width_is_governed_per_host(url, requested, expected):
    """Eight concurrent range GETs at NOMADS was the audited burst."""

    assert governor.governed_workers(url, requested) == expected


def test_hrrr_range_transport_asks_the_governor_for_its_pool_width():
    """The transport must ask, not hardcode -- bind the call site."""

    import inspect
    from tools import download_hrrr_native_subset as transport

    source = inspect.getsource(transport._download_subset)
    assert "governed_workers(url, workers)" in source
    assert "paced_urlopen" in inspect.getsource(transport._download_range)
    assert "paced_urlopen" in inspect.getsource(transport._head)
    assert "paced_urlopen" in inspect.getsource(transport._request_bytes)


# ---------------------------------------------------------------------------
# Retry-After: the server's own instruction is honored, never undercut
# ---------------------------------------------------------------------------

def test_retry_after_parses_seconds_and_http_dates():
    import email.utils

    error = HTTPError(NOMADS, 429, "rate", {"Retry-After": "120"}, None)
    assert governor.retry_after_seconds(error) == 120.0
    stamped = email.utils.formatdate(time.time() + 90, usegmt=True)
    dated = HTTPError(NOMADS, 503, "busy", {"Retry-After": stamped}, None)
    parsed = governor.retry_after_seconds(dated)
    assert parsed is not None and 60.0 <= parsed <= 91.0
    absent = HTTPError(NOMADS, 503, "busy", {}, None)
    assert governor.retry_after_seconds(absent) is None
    junk = HTTPError(NOMADS, 429, "rate", {"Retry-After": "soon"}, None)
    assert governor.retry_after_seconds(junk) is None
    past = HTTPError(NOMADS, 429, "rate", {"Retry-After": "-5"}, None)
    assert governor.retry_after_seconds(past) is None


def test_retry_after_extends_the_cooldown_but_never_shortens_it(
        monkeypatch, isolated_state):
    monkeypatch.setenv(governor.COOLDOWN_ENV, "30000")
    clock = _Clock(monkeypatch)
    # Longer than the configured cooldown: the server's word wins.
    governor.mark_rate_limited(NOMADS, "over_rate_limit",
                               retry_after_s=120.0)
    assert governor.read_state(isolated_state)[1] == clock.now_ms + 120_000
    # Shorter than the configured cooldown: the shared-state contract
    # with the Rust twin stays -- honoring a header must never make the
    # node-wide governor looser than its own configuration.
    isolated_state.unlink()
    governor.mark_rate_limited(NOMADS, "over_rate_limit", retry_after_s=1.0)
    assert governor.read_state(isolated_state)[1] == clock.now_ms + 30_000


def test_paced_urlopen_carries_retry_after_into_the_cooldown(
        monkeypatch, isolated_state):
    monkeypatch.setenv(governor.COOLDOWN_ENV, "30000")
    clock = _Clock(monkeypatch)
    monkeypatch.setattr(governor, "pace", lambda url, **kw: None)

    def opener(request, **kwargs):
        raise HTTPError(NOMADS, 429, "Over Rate Limit",
                        {"Retry-After": "3600"}, None)

    with pytest.raises(HTTPError):
        governor.paced_urlopen(Request(NOMADS), opener=opener)
    assert governor.read_state(isolated_state)[1] == (
        clock.now_ms + 3_600_000)


def test_gfs_transport_waits_out_retry_after_between_attempts(
        monkeypatch, tmp_path):
    from tools import download_gfs_native_subset as transport

    naps: list[float] = []
    monkeypatch.setattr(transport.time, "sleep", naps.append)
    grib = b"GRIB" + b"\x00\x00\x00\x02" + (2048).to_bytes(8, "big") \
        + b"\x00" * 2036 + b"7777"

    class _Body(_Response):
        def __init__(self):
            self._left = [grib]

        def read(self, *_args):
            return self._left.pop() if self._left else b""

    calls = {"n": 0}

    def flaky(request, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise HTTPError(S3, 503, "busy", {"Retry-After": "7"}, None)
        return _Body()

    monkeypatch.setattr(
        transport, "paced_urlopen",
        lambda request, **kw: flaky(request, **kw))
    transport._download(S3, tmp_path / "object.grib2")
    assert calls["n"] == 2
    # The wait between the attempts is at least the server's ask, not
    # only the transport's own backoff ladder.
    assert naps and max(naps) >= 7.0


def test_request_log_is_the_rust_json_line_shape(monkeypatch, tmp_path):
    log = tmp_path / "nomads.jsonl"
    monkeypatch.setenv(governor.REQUEST_LOG_ENV, str(log))
    governor.log_event(NOMADS, "get", "200", 42)
    record = json.loads(log.read_text().strip())
    assert set(record) == {"ts_ms", "pid", "kind", "status", "elapsed_ms",
                           "url"}
    assert record["url"] == NOMADS
    assert record["pid"] == os.getpid()
    assert record["elapsed_ms"] == 42


def test_a_slowdown_answer_keeps_its_code_after_the_governor_closes_it(
        monkeypatch, isolated_state):
    """The governor closes an HTTPError before re-raising it; S3's
    ``<Code>SlowDown</Code>`` must still match the host's throttle row
    (WOOF run_19793057ff2979e3, ECMWF's AWS mirror, 2026-10-29)."""
    import io

    from woof import fetch_endpoints

    url = ("https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/20261028/"
           "18z/ifs/0p1/oper/20261028180000-3h-oper-fc-sfc.grib2")
    error = HTTPError(url, 503, "Slow Down", {}, io.BytesIO(
        b"<?xml version=\"1.0\"?><Error><Code>SlowDown</Code>"
        b"<Message>Please reduce your request rate.</Message></Error>"))

    def opener(request, **kwargs):
        raise error

    with pytest.raises(HTTPError) as raised:
        governor.paced_urlopen(Request(url), opener=opener)
    assert fetch_endpoints.answer_code(raised.value) == "SlowDown"
    endpoint = fetch_endpoints.endpoint_named("ecmwf-open-data-0p1", "aws")
    assert fetch_endpoints.throttled(endpoint, raised.value) is not None
