# SPDX-License-Identifier: Apache-2.0
"""An archive that answers "slow down" gets patience; any other failure keeps the usual count. Local server, no card.

THE BREAKAGE (front-door runs of the 0.6.0 candidate, 2026-11-15): the public ECMWF archive answered S3's SlowDown
(HTTP 503) to about three requests in four for an evening. Ten tries a request were not enough for a region that reads
an index and several byte ranges of three files, so every 2024 IFS start was refused on objects that were there.
"""
import http.server
import socketserver
import threading

import pytest

from woof.hex import init_sources as src

BODY = b"grib" * 4096


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _Origin:
    """One object; the next ``refuse`` requests are answered with HTTP ``status`` first."""

    def __init__(self):
        self.refuse, self.status, self.requests = 0, 503, 0
        origin = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                origin.requests += 1
                if origin.refuse > 0:
                    origin.refuse -= 1
                    text = b"<Error><Code>SlowDown</Code><Message>Please reduce your request rate.</Message></Error>"
                    self.send_response(origin.status)
                    self.send_header("Content-Length", str(len(text)))
                    self.end_headers()
                    self.wfile.write(text)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(BODY)))
                self.end_headers()
                self.wfile.write(BODY)

        self.server = _Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/20240519180000-0h-scda-fc.grib2"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def origin(monkeypatch):
    waits = []
    monkeypatch.setattr(src.time, "sleep", waits.append)
    monkeypatch.setattr(src, "_HEDGES", [])
    server = _Origin()
    server.waits = waits
    yield server
    server.close()


@pytest.mark.parametrize("status", [503, 429])
def test_a_slow_down_answer_outlasts_ten_tries(origin, status, capsys):
    origin.refuse, origin.status = 25, status   # the archive of 2026-11-15: far more than ten in a row
    assert src._http_get(origin.url) == BODY
    assert origin.requests == 26
    assert max(origin.waits) <= 30.0 and len(origin.waits) == 25
    said = capsys.readouterr().err
    assert said.count("keeps answering") == 1 and f"HTTP {status} (slow down)" in said and "up to 40 times" in said


def test_a_slow_down_that_never_ends_is_still_refused_by_address(origin):
    origin.refuse = 1000
    with pytest.raises(src.InitSourceRefusal, match=r"scda-fc\.grib2 could not be read after 40 attempts "
                                                    r"\(HTTP Error 503.*no partial product is kept"):
        src._http_get(origin.url)
    assert origin.requests == src.FETCH_THROTTLE_ATTEMPTS == 40


def test_another_server_error_keeps_the_usual_ten(origin):
    origin.refuse, origin.status = 1000, 500
    with pytest.raises(src.InitSourceRefusal, match="could not be read after 10 attempts"):
        src._http_get(origin.url)
    assert origin.requests == 10


def test_a_probe_with_fewer_tries_keeps_its_own_count(origin):
    origin.refuse = 1000
    with pytest.raises(src.InitSourceRefusal, match="could not be read after 3 attempts"):
        src._http_get(origin.url, attempts=3)
    assert origin.requests == 3
