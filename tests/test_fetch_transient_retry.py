"""A source fetch waits out a host that is briefly unavailable.

The breakage: an IFS fetch pinned to the AWS mirror met HTTP 503 on 3 of
its 5 files, gave up after three tries 2 and 4 s apart, and the same
command run 20 s later completed.  Every table source moves its objects
through the one shared retry (:func:`woof.fetch_endpoints.ask_along_ladder`),
so these tests stand a flaky origin up on this computer and fetch two
different sources from it through the real transport.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import threading
from types import SimpleNamespace

import pytest

from woof import fetch_endpoints, fetch_routes, rustwx_fetch


#: The shared schedule, spelled out: four waits between five rounds.
SCHEDULE = [2.0, 4.0, 8.0, 16.0]


def _payload(path: str) -> bytes:
    return b"GRIB" + path.encode("ascii") + bytes(64) + b"7777"


class _FlakyOrigin:
    """A local host that fails the first ``failures`` asks for each object.

    ``fault`` is what a failing ask meets: ``"503"`` (the host is
    unavailable or throttling) or ``"reset"`` (the connection is closed
    before any answer).
    """

    def __init__(self, *, failures: int, fault: str = "503") -> None:
        self.failures = failures
        self.fault = fault
        self.asked: Counter[str] = Counter()
        self._lock = threading.Lock()
        origin = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):          # noqa: N802 - http.server API
                if self.path.endswith(".index"):
                    # A host that keeps no record index: the fetch takes
                    # the whole object, which is what these tests retry.
                    self.send_response(404, "Not Found")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                with origin._lock:
                    origin.asked[self.path] += 1
                    count = origin.asked[self.path]
                if count <= origin.failures:
                    if origin.fault == "reset":
                        self.close_connection = True
                        self.connection.shutdown(socket.SHUT_RDWR)
                        return
                    self.send_response(503, "Slow Down")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = _payload(self.path)
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)

    def __enter__(self) -> "_FlakyOrigin":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    def base(self, name: str) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/{name}"


@pytest.fixture(autouse=True)
def recorded_waits(monkeypatch):
    """The waits between rounds, recorded instead of slept through."""

    waits: list[float] = []
    monkeypatch.setattr(fetch_routes, "time",
                        SimpleNamespace(sleep=waits.append), raising=False)
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
                 "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    return waits


#: Two sources with different producers, key shapes and file rows,
#: each pinned to its AWS mirror the way the failing fetch was.
SOURCES = [("ecmwf-open-data", "aws"), ("rrfs", "aws")]


def _plan(source: str, host: str, origin: _FlakyOrigin, *, objects: int = 2):
    plan = fetch_routes.resolve_request(
        source, cycle=datetime(2026, 9, 22), hours=3, host=host)
    ladder = tuple(replace(endpoint, base=origin.base(endpoint.name))
                   for endpoint in plan.ladder)
    kept = plan.objects[:objects]
    return replace(plan, host=ladder[0], ladder=ladder, objects=kept,
                   primary_files=tuple(obj.relpath for obj in kept),
                   supplement_files=(), compose=())


def _run(plan, out, *, lines=None, workers=2):
    return fetch_routes.run_plan(
        plan, out=out, file_workers=workers, probe=lambda _url: False,
        progress=(lines.append if lines is not None else (lambda _l: None)))


@pytest.mark.parametrize("source,host", SOURCES)
def test_a_host_answering_503_for_four_tries_is_waited_out(
        tmp_path, recorded_waits, source, host):
    with _FlakyOrigin(failures=4) as origin:
        plan = _plan(source, host, origin)
        lines: list[str] = []
        payload = _run(plan, tmp_path, lines=lines)

    assert len(payload["files"]) == len(plan.objects)
    for obj in plan.objects:
        landed = (tmp_path / obj.relpath).read_bytes()
        path = "/" + host + "/" + obj.key
        assert landed == _payload(path)
        # Five asks: four 503s and the one that served.
        assert origin.asked[path] == fetch_endpoints.TRANSIENT_ATTEMPTS == 5
    # Each object waited 2, 4, 8 and 16 s between its rounds.
    assert sorted(recorded_waits) == sorted(SCHEDULE * len(plan.objects))
    assert (tmp_path / fetch_routes.MANIFEST_NAME).is_file()
    assert any("retrying" in line and "in 16 s (attempt 5/5)" in line
               for line in lines)
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize("source,host", SOURCES)
def test_a_dropped_connection_is_asked_again(tmp_path, recorded_waits,
                                             source, host):
    with _FlakyOrigin(failures=2, fault="reset") as origin:
        plan = _plan(source, host, origin, objects=1)
        payload = _run(plan, tmp_path, workers=1)

    obj = plan.objects[0]
    assert payload["files"][0]["relpath"] == obj.relpath
    assert origin.asked["/" + host + "/" + obj.key] == 3
    assert recorded_waits == SCHEDULE[:2]


@pytest.mark.parametrize("source,host", SOURCES)
def test_a_host_that_stays_down_is_refused_by_name(tmp_path, recorded_waits,
                                                  source, host):
    with _FlakyOrigin(failures=10 ** 6) as origin:
        plan = _plan(source, host, origin, objects=1)
        with pytest.raises(fetch_endpoints.TransferRefusal) as caught:
            _run(plan, tmp_path, workers=1)

    obj = plan.objects[0]
    refusal = caught.value
    text = str(refusal)
    # Which file, which host, why, how often and for how long.
    assert refusal.name == obj.relpath
    assert text.startswith(f"fetch {plan.source_id}: {obj.relpath}: ")
    assert f"{host} (127.0.0.1:" in text
    assert "HTTP 503 -- the host is unavailable or throttling" in text
    assert "(asked 5 times)" in text
    assert "in 5 rounds over 30 s" in text
    assert "start this forecast again to retry the remaining files" in text
    # One line per endpoint, not one per ask.
    assert text.count("HTTP 503") == 1
    # Bounded: five asks and no more, and no manifest for a request
    # that did not complete.
    assert origin.asked["/" + host + "/" + obj.key] == 5
    assert recorded_waits == SCHEDULE
    assert not (tmp_path / fetch_routes.MANIFEST_NAME).exists()


def test_a_missing_object_is_not_asked_again():
    endpoint = fetch_endpoints.Endpoint(name="mirror",
                                        base="https://mirror.invalid",
                                        retention_hours=None, why="")
    asked = []

    def transfer(where):
        asked.append(where.name)
        from urllib.error import HTTPError
        raise HTTPError(where.url("k"), 404, "not found", {}, None)

    with pytest.raises(fetch_endpoints.TransferRefusal) as caught:
        fetch_endpoints.ask_along_ladder(
            (endpoint,), transfer, label="fetch x", name="k",
            progress=lambda _l: None, pause=lambda _s: None)
    assert asked == ["mirror"]
    assert caught.value.rounds == 1


def test_the_rust_backbone_s_dropped_transfer_is_asked_again():
    """A transfer the Rust backbone's own short retries could not save.

    The GFS and HRRR routes move their bytes through ``rw_fetch``; a
    transfer it reports as cut off by the network (exit status
    ``EXIT_TRANSFER``) takes the same shared schedule as the table
    routes' 503.
    """

    dropped = rustwx_fetch.RwFetchError(
        "fetch", "connection reset by peer",
        returncode=rustwx_fetch.EXIT_TRANSFER)
    refused = rustwx_fetch.RwFetchError(
        "fetch", "HTTP 404", returncode=rustwx_fetch.EXIT_REFUSED)
    assert fetch_endpoints.retry_delay(dropped, 1, wait_limit_s=30.0) == 2.0
    assert fetch_endpoints.retry_delay(refused, 1, wait_limit_s=30.0) is None

    endpoint = fetch_endpoints.Endpoint(name="s3",
                                        base="https://s3.invalid",
                                        retention_hours=None, why="")
    waits: list[float] = []
    asked = []

    def transfer(where):
        asked.append(where.name)
        if len(asked) < 3:
            raise dropped
        return "landed"

    served, result = fetch_endpoints.ask_along_ladder(
        (endpoint,), transfer, label="fetch gfs f003", name="gfs.f003",
        progress=lambda _l: None, pause=waits.append)
    assert (served.name, result) == ("s3", "landed")
    assert waits == SCHEDULE[:2]
