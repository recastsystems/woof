"""Publication probes reuse their connections (woof.probe_connections).

Breakage this covers: every HEAD of a publication check opened its own TLS
connection, and ICON global's ~234 start objects on opendata.dwd.de took
52-58 s to ask from a US box (WOOF canary icon-offgrid-3-1-0.5, 2026-11-10).
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from woof import fetch_endpoints, nomads_governor, probe_connections


class _Host:
    def __init__(self, statuses=None, close_after=None):
        self.statuses = statuses or {}
        self.connections = 0
        self.requests = []
        self.close_after = close_after
        host = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                host.connections += 1
                self.served = 0

            def log_message(self, *args):
                pass

            def _answer(self, body: bytes):
                host.requests.append((self.command, self.path, self.headers.get("Range")))
                status = host.statuses.get(self.path, 200)
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.served += 1
                if host.close_after and self.served >= host.close_after:
                    self.send_header("Connection", "close")
                    self.close_connection = True
                self.end_headers()
                return status

            def do_HEAD(self):
                self._answer(b"x" * 10)

            def do_GET(self):
                body = b"<Code>SlowDown</Code>" if host.statuses.get(self.path) == 503 else b"G"
                self._answer(body)
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def host(monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY", "no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    probe_connections.close_idle()
    made = []

    def start(**kwargs):
        made.append(_Host(**kwargs))
        return made[-1]

    yield start
    probe_connections.close_idle()
    for one in made:
        one.stop()


def test_many_heads_share_one_connection(host):
    server = host()
    answers = [fetch_endpoints.object_answer(f"{server.base}/f{i}.grib2.bz2", timeout=5)
               for i in range(40)]
    assert answers == [True] * 40
    assert server.connections == 1
    assert [r[0] for r in server.requests] == ["HEAD"] * 40


def test_missing_object_is_no_and_the_connection_is_kept(host):
    server = host(statuses={"/gone": 404})
    assert fetch_endpoints.object_answer(f"{server.base}/gone", timeout=5) is False
    assert fetch_endpoints.object_answer(f"{server.base}/here", timeout=5) is True
    assert server.connections == 1


def test_refused_head_is_confirmed_by_a_one_byte_get(host):
    server = host(statuses={"/x": 405})
    # The GET is answered 405 too: no answer about the object, as before.
    assert fetch_endpoints.object_answer(f"{server.base}/x", timeout=5) is None
    assert server.requests == [("HEAD", "/x", None), ("GET", "/x", "bytes=0-0")]


def test_throttle_answer_body_is_readable(host):
    server = host(statuses={"/slow": 503})
    from urllib.request import Request
    with pytest.raises(Exception) as caught:
        nomads_governor.paced_urlopen(
            Request(f"{server.base}/slow", headers={"Range": "bytes=0-0"}), timeout=5)
    assert caught.value.code == 503
    assert fetch_endpoints.answer_code(caught.value) == "SlowDown"


def test_a_host_that_closes_between_probes_is_asked_on_a_new_connection(host):
    server = host(close_after=3)
    answers = [fetch_endpoints.object_answer(f"{server.base}/o{i}", timeout=5) for i in range(9)]
    assert answers == [True] * 9
    assert server.connections == 3


def test_downloads_are_not_kept_alive():
    from urllib.request import Request
    assert not probe_connections.kept_alive(Request("https://h.example/a.grib2"))
    assert not probe_connections.kept_alive(
        Request("https://h.example/a.grib2", headers={"Range": "bytes=0-99"}))
    assert probe_connections.kept_alive(Request("https://h.example/a", method="HEAD"))
    assert probe_connections.kept_alive(
        Request("https://h.example/a", headers={"Range": "bytes=0-0"}))


def test_a_proxied_host_goes_through_urlopen(monkeypatch):
    from urllib.request import Request
    monkeypatch.setenv("https_proxy", "http://proxy.example:3128")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    assert not probe_connections.kept_alive(Request("https://h.example/a", method="HEAD"))
