"""A transient HTTP 504 from the release host is retried by the kernel-pack and physics-resource fetches.

THE BREAKAGE (hex 0.5.0 gate, 2026-11-09): a fresh install's first default global run was refused on one 504.
"""
import hashlib
import http.server
import threading

import urllib.error

import pytest

from woof.hex import assets
from woof.hex.mpasphys import resources_build as rb, resources_fetch


BODY = b"pinned bytes" * 1000


@pytest.fixture
def flaky_server(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            if len(calls) <= 2:
                self.send_error(504)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", calls
    server.shutdown()


def _row():
    return {"sha256": hashlib.sha256(BODY).hexdigest(), "bytes": len(BODY)}


def test_kernel_pack_fetch_retries_504(flaky_server, tmp_path, monkeypatch):
    url, calls = flaky_server
    monkeypatch.setattr(assets, "verify", lambda path, row, **kw: None)
    destination = tmp_path / "pack.json"
    assets._download("pack.json", _row(), None, {"url": url + "/pack.json"}, destination)
    assert destination.read_bytes() == BODY and len(calls) == 3


def test_physics_resource_fetch_retries_504(flaky_server, tmp_path):
    url, calls = flaky_server
    asset = {"name": "a.zip", "url": url + "/a.zip", **_row()}
    target = resources_fetch._download(asset, tmp_path, None, {"repository": "x/y", "tag": "t"})
    assert target.read_bytes() == BODY and len(calls) == 3


def test_permanent_404_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            self.send_error(404)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        asset = {"name": "a.zip", "url": f"http://127.0.0.1:{server.server_port}/a.zip", **_row()}
        # Either implementation of the retry may surface the permanent answer (wrapped or raw); it is never retried.
        with pytest.raises((rb.PhysicsResourceError, urllib.error.HTTPError)):
            resources_fetch._download(asset, tmp_path, None, {"repository": "x/y", "tag": "t"})
        assert len(calls) == 1
    finally:
        server.shutdown()
