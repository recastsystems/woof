"""No-network checks for box connection budgets and parallel range integrity."""
import hashlib
import io
import multiprocessing
import threading
import time
from pathlib import Path

import pytest

from woof import fetch_connections, fetch_pool, fetch_ranges, fetch_routes


@pytest.mark.parametrize("cap,workers,files", [
    (1, None, 1), (1, 150, 100), (2, 1, 4), (2, 4, 25),
    (7, None, 100), (7, 4, 4), (7, 24, 1),
    (30, None, 25), (30, 24, 100), (30, 150, 4),
    (61, 1, 100), (61, 150, 1),
])
def test_file_times_range_width_never_exceeds_box_cap(monkeypatch, cap, workers, files):
    monkeypatch.setenv("WOOF_FETCH_MAX_CONNECTIONS", str(cap))
    for row in fetch_pool._width_table()["rows"]:
        monkeypatch.setenv("WOOF_FETCH_ROW", row["id"])
        width = min(files, fetch_pool.resolve_file_workers(workers))
        streams = fetch_pool.chunk_streams_per_file(workers, files=files)
        assert 1 <= width * streams <= cap


@pytest.mark.parametrize("bad", [0, -1, True, "oops", 1.5])
def test_invalid_box_cap_names_the_connection_problem(bad):
    with pytest.raises(ValueError, match="cannot bound outbound sockets"):
        fetch_connections.max_connections(bad)


def _hold_slots(directory, ready):
    import os
    os.environ[fetch_connections.DIRECTORY_ENV] = directory
    os.environ[fetch_connections.MAX_CONNECTIONS_ENV] = "2"
    with fetch_connections.lease(2):
        ready.set()
        time.sleep(30)


def test_connection_lease_is_shared_across_processes_and_released_on_exit(tmp_path, monkeypatch):
    monkeypatch.setenv(fetch_connections.DIRECTORY_ENV, str(tmp_path))
    monkeypatch.setenv(fetch_connections.MAX_CONNECTIONS_ENV, "2")
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    child = ctx.Process(target=_hold_slots, args=(str(tmp_path), ready))
    child.start()
    acquired = threading.Event()
    def wait_for_slot():
        with fetch_connections.lease():
            acquired.set()
    waiter = threading.Thread(target=wait_for_slot)
    try:
        assert ready.wait(10)
        waiter.start()
        assert not acquired.wait(0.15)
        child.terminate()
        child.join(10)
        assert acquired.wait(5)
    finally:
        if child.is_alive():
            child.terminate()
        child.join(10)
        waiter.join(10)


class Response(io.BytesIO):
    def __init__(self, body, headers, status=200):
        super().__init__(body)
        self.headers, self.status = headers, status


def test_ranges_write_exact_disjoint_parts_and_pin_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_ranges, "MIN_PART", 20)
    payload = bytes(range(100))
    asked = []
    def open_object(request, **kwargs):
        span = request.get_header("Range")
        headers = {"Content-Length": "100", "Accept-Ranges": "bytes", "ETag": '"generation"'}
        if span is None:
            return Response(payload, headers)
        asked.append(span)
        assert request.get_header("If-match") == '"generation"'
        start, stop = map(int, span.removeprefix("bytes=").split("-"))
        headers["Content-Range"] = f"bytes {start}-{stop}/100"
        return Response(payload[start:stop+1], headers, 206)
    monkeypatch.setattr(fetch_ranges, "paced_urlopen", open_object)
    path = tmp_path / "input.part"
    size, digest = fetch_ranges.download("https://example.test/file", path,
        user_agent="test", streams=3)
    assert sorted(asked) == ["bytes=0-32", "bytes=33-65", "bytes=66-99"]
    assert path.read_bytes() == payload
    assert (size, digest) == (100, hashlib.sha256(payload).hexdigest())


@pytest.mark.parametrize("failure", ["ignored", "wrong-span", "changed", "truncated"])
def test_range_failure_never_publishes_a_partial_file(tmp_path, monkeypatch, failure):
    if failure == "truncated":
        from concurrent.futures import ThreadPoolExecutor
        # Queue the other ranges behind a failed first range to exercise cancellation.
        monkeypatch.setattr(fetch_ranges, "ThreadPoolExecutor",
                            lambda **kwargs: ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(fetch_ranges, "MIN_PART", 20)
    payload = bytes(range(100))
    def open_object(request, **kwargs):
        span = request.get_header("Range")
        headers = {"Content-Length": "100", "Accept-Ranges": "bytes", "ETag": '"one"'}
        if span is None:
            return Response(payload, headers)
        start, stop = map(int, span.removeprefix("bytes=").split("-"))
        headers["Content-Range"] = f"bytes {start}-{stop}/100"
        if failure == "wrong-span":
            headers["Content-Range"] = "bytes 0-0/100"
        if failure == "changed":
            headers["ETag"] = '"two"'
        body = payload[start:stop+1]
        return Response(body[:-1] if failure == "truncated" else body,
                        headers, 200 if failure == "ignored" else 206)
    monkeypatch.setattr(fetch_ranges, "paced_urlopen", open_object)
    dest = tmp_path / "input.grib2"
    if failure in ("ignored", "wrong-span"):
        # fix/ecmwf-open-data-range: a host that refuses ranges moves the
        # object on one plain stream instead of failing it.
        size, digest = fetch_ranges.download("https://example.test/file",
                                             tmp_path / "plain.part",
                                             user_agent="test", streams=3)
        assert (size, digest) == (100, hashlib.sha256(payload).hexdigest())
        assert (tmp_path / "plain.part").read_bytes() == payload
    with pytest.raises(ValueError):
        fetch_routes._download_object("https://example.test/file", dest,
                                       magic="GRIB", streams=3)
    assert not dest.exists() and not dest.with_suffix(".grib2.part").exists()


def test_config_and_cli_carry_box_cap():
    from woof import fetch, runplan
    from woof.cli import parse_fetch_arguments
    hints = {"source": "hrrr-prs", "cycle": "2026-10-18T12", "hours": 3,
             "fetch_max_connections": 7}
    fetch.validate_fetch_hints(hints, source="config.toml")
    args = parse_fetch_arguments(runplan._fetch_arguments_from_hints(hints, out=Path("inputs")))
    assert args.fetch_max_connections == 7


def test_pool_receipt_names_cap_and_actual_transport_limits(monkeypatch):
    monkeypatch.setenv(fetch_connections.MAX_CONNECTIONS_ENV, "7")
    def action():
        fetch_pool.record_streams(3, 6)
        return {"bytes": 10}
    _, receipt = fetch_pool.run_transfers([fetch_pool.TransferJob("input", None, action)], workers=2)
    assert receipt["max_connections"] == 7
    assert receipt["streams_used"] == {"input": 3}
    assert receipt["connection_slots_per_file"] == {"input": 6}


@pytest.mark.parametrize("idle_capable,idle", [(True, 0), (False, 10)])
def test_bridge_reserves_active_and_possible_idle_sockets(tmp_path, monkeypatch, idle_capable, idle):
    import contextlib
    from woof import rustwx_fetch
    monkeypatch.setenv(fetch_connections.MAX_CONNECTIONS_ENV, "30")
    monkeypatch.setattr(rustwx_fetch, "_idle_capable", lambda *args: idle_capable)
    reserved = []
    @contextlib.contextmanager
    def reserve(count, *, minimum=1):
        assert minimum == idle + 1
        reserved.append(count)
        yield count
        reserved.pop()
    monkeypatch.setattr(fetch_connections, "lease", reserve)
    def process(command, *, env, **kwargs):
        assert reserved == [30]
        assert int(env[rustwx_fetch.STREAMS_ENV]) == 30 - idle
        assert env["GPUWM_FETCH_NO_IDLE_CONNECTIONS"] == "1"
        return {"bytes": 1}
    monkeypatch.setattr(rustwx_fetch, "_run_process", process)
    assert rustwx_fetch._run([str(tmp_path / "bridge"), "object"], what="object",
                            env={rustwx_fetch.STREAMS_ENV: "150"}) == {"bytes": 1}
    assert not reserved


@pytest.mark.parametrize("host,configured,reported", [
    ("noaa-hrrr-bdp-pds.s3.amazonaws.com", 3, 3),
    ("noaa-hrrr-bdp-pds.s3.amazonaws.com", 3, 1),
    ("nomads.ncep.noaa.gov", 1, 1),
])
def test_promoted_mirror_sets_native_width_and_actual_receipt(
        tmp_path, monkeypatch, host, configured, reported):
    from datetime import datetime
    from types import SimpleNamespace
    from woof import fetch, rustwx_fetch
    monkeypatch.setenv(fetch_connections.MAX_CONNECTIONS_ENV, "30")
    monkeypatch.setenv(fetch_connections.DIRECTORY_ENV, str(tmp_path / "connections"))
    monkeypatch.setenv("WOOF_FETCH_ROW", "balanced")
    binary = tmp_path / "bridge"
    binary.write_text("test bridge")
    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: binary)
    monkeypatch.setattr(rustwx_fetch, "_idle_capable", lambda *a: True)
    monkeypatch.setattr(fetch, "_stream_capable", lambda *a: True)
    monkeypatch.setattr(fetch, "_remember_stream_facts", lambda *a: None)
    plan = fetch_routes.resolve_request("hrrr-prs", cycle=datetime(2026, 10, 18, 12), hours=24)
    assert fetch_pool.host_key(plan.objects[0].url) == "nomads.ncep.noaa.gov"
    download = fetch._route_stream_downloader(plan, SimpleNamespace(fetch_workers=None))
    def process(command, *, env, **kwargs):
        assert int(env[rustwx_fetch.STREAMS_ENV]) == configured
        return {"bytes": 1, "sha256": "0" * 64, "streams_used": reported}
    monkeypatch.setattr(rustwx_fetch, "_run_process", process)
    url = f"https://{host}/file.grib2"
    _, receipt = fetch_pool.run_transfers([fetch_pool.TransferJob("input", url,
        lambda: download(url, tmp_path / "input.grib2", magic="GRIB"))], workers=10)
    assert receipt["streams_used"] == {"input": reported}
    assert receipt["connection_slots_per_file"] == {"input": configured}


def test_response_holds_its_slot_for_the_body(tmp_path, monkeypatch):
    from woof.nomads_governor import paced_urlopen
    monkeypatch.setenv(fetch_connections.DIRECTORY_ENV, str(tmp_path))
    monkeypatch.setenv(fetch_connections.MAX_CONNECTIONS_ENV, "1")
    first = paced_urlopen("https://example.test/one", opener=lambda *a: Response(b"body", {}))
    opened = threading.Event()
    def second():
        with paced_urlopen("https://example.test/two", opener=lambda *a: Response(b"two", {})):
            opened.set()
    waiter = threading.Thread(target=second)
    waiter.start()
    try:
        assert not opened.wait(0.1)
        assert first.read() == b"body"
    finally:
        first.close()
        waiter.join(5)
    assert opened.is_set()


def test_box_wait_precedes_host_pacing(monkeypatch):
    import contextlib
    from woof import nomads_governor
    events = []
    @contextlib.contextmanager
    def reserve():
        events.append("reserve")
        yield
        events.append("release")
    monkeypatch.setattr(fetch_connections, "lease", reserve)
    monkeypatch.setattr(nomads_governor, "pace", lambda *a, **k: events.append("pace"))
    def opener(*args):
        events.append("send")
        return Response(b"body", {})
    with nomads_governor.paced_urlopen("https://example.test/file", opener=opener):
        assert events == ["reserve", "pace", "send"]
    assert events == ["reserve", "pace", "send", "release"]


@pytest.mark.parametrize("cap", [1, 7, 10])
def test_small_cap_uses_python_when_bridge_may_retain_idle_sockets(tmp_path, monkeypatch, cap):
    from datetime import datetime
    from types import SimpleNamespace
    from woof import fetch, rustwx_fetch
    monkeypatch.setenv(fetch_connections.MAX_CONNECTIONS_ENV, str(cap))
    binary = tmp_path / "old-bridge"
    binary.write_text("old bridge")
    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: binary)
    monkeypatch.setattr(rustwx_fetch, "_idle_capable", lambda *a: False)
    monkeypatch.setattr(fetch, "_stream_capable", lambda *a: True)
    plan = fetch_routes.resolve_request("hrrr-prs", cycle=datetime(2026, 10, 18, 12), hours=3, host="aws")
    assert fetch._route_stream_downloader(plan, SimpleNamespace(fetch_workers=None)) is None


def test_lease_grants_what_is_free_and_names_holders_while_waiting(tmp_path, monkeypatch, capfd):
    """test host 2026-10-20: another process's slots left setup waiting silently forever."""
    monkeypatch.setenv(fetch_connections.DIRECTORY_ENV, str(tmp_path))
    monkeypatch.setenv(fetch_connections.MAX_CONNECTIONS_ENV, "4")
    monkeypatch.setattr(fetch_connections, "WAIT_NOTICE_AFTER_S", 0.1)
    monkeypatch.setenv(fetch_connections.SLOT_WAIT_ENV, "0.5")
    with fetch_connections.lease(3) as held:
        assert held == 3
        assert fetch_connections.free_slots() == 1
        with fetch_connections.lease(4) as partial:
            assert partial == 1
        with fetch_connections.lease(1):
            with pytest.raises(RuntimeError, match="no fetch connection slot came free"):
                with fetch_connections.lease(2):
                    pass
    assert "waiting" in capfd.readouterr().err
