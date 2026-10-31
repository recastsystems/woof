"""The RRFS table fetch places pinned files concurrently, large ones as resumable byte ranges (fetch_pinned).

THE BREAKAGE THESE PREVENT (rrfs/table-fetch-parallel, 2026-10-28): the first hrrrv5 run on a plain install fetched
the RRFS fork's 999 MB table tree one file at a time over one connection (7.9 min on an RTX 5090 host), and a fetch stopped part
way began the 675 MB TEMPO table again from byte 0.  A local HTTP server stands in for NOAA's bucket.
"""
from __future__ import annotations

import hashlib
import http.server
from pathlib import Path
import threading

import pytest

from woof.hex.mpasphys import resources_build as rb


def _payload(size: int, seed: bytes) -> bytes:
    blocks, counter = [], 0
    while sum(map(len, blocks)) < size:
        blocks.append(hashlib.sha256(seed + counter.to_bytes(8, "little")).digest())
        counter += 1
    return b"".join(blocks)[:size]


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.files: dict[str, bytes] = {}
        self.ranges = True            # honour Range requests
        self.refuse_from: int | None = None   # 404 every range starting at or after this byte
        self.seen: list[tuple[str, str | None]] = []
        self.lock = threading.Lock()


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        server = self.server
        body = server.files.get(self.path)
        wanted = self.headers.get("Range")
        with server.lock:
            server.seen.append((self.path, wanted))
        if body is None:
            self.send_error(404)
            return
        if wanted and server.ranges:
            start, end = (int(value) for value in wanted.removeprefix("bytes=").split("-"))
            if server.refuse_from is not None and start >= server.refuse_from:
                self.send_error(404)
                return
            part = body[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{start + len(part) - 1}/{len(body)}")
        else:
            part = body
            self.send_response(200)
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        self.wfile.write(part)


@pytest.fixture
def server(monkeypatch):
    serving = _Server()
    thread = threading.Thread(target=serving.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setitem(rb.URLS, "test-origin", f"http://127.0.0.1:{serving.server_address[1]}/{{path}}")
    monkeypatch.setattr(rb, "FETCH_CHUNK_BYTES", 64 << 10)
    monkeypatch.setattr(rb.time, "sleep", lambda seconds: None)
    yield serving
    serving.shutdown()
    serving.server_close()


def _row(tmp_path: Path, name: str, body: bytes) -> dict:
    sha = hashlib.sha256(body).hexdigest()
    return {"name": name, "origin": "test-origin", "path": f"fix/{name}", "sha256": sha, "bytes": len(body),
            "target": tmp_path / "cache" / sha / name}


def test_large_files_arrive_as_concurrent_ranges_with_the_pinned_bytes(tmp_path, server):
    big, small = _payload(1_000_003, b"tempo"), _payload(5_000, b"ccn")
    server.files.update({"/fix/qr_acr_qg": big, "/fix/ccn_activate.bin": small})
    rows = [_row(tmp_path, "ccn_activate.bin", small), _row(tmp_path, "qr_acr_qg", big)]
    placed = rb.fetch_pinned(rows, jobs=4, log=lambda message: None)
    assert [path.read_bytes() for path in placed] == [small, big]
    ranges = [wanted for path, wanted in server.seen if path == "/fix/qr_acr_qg"]
    assert len(ranges) == -(-len(big) // (64 << 10)) and all(ranges)
    assert not [path for path in (tmp_path / "cache").rglob(".partial-*")]


def test_a_stopped_fetch_resumes_with_only_the_missing_ranges(tmp_path, server):
    big = _payload(1_000_003, b"freeze")
    server.files["/fix/freeze_water"] = big
    row = _row(tmp_path, "freeze_water", big)
    server.refuse_from = 8 * (64 << 10)
    with pytest.raises(rb.PhysicsResourceError, match="could not fetch"):
        rb.fetch_pinned([row], jobs=2, log=lambda message: None)
    assert not row["target"].exists()
    first = {wanted for path, wanted in server.seen if wanted and int(wanted[6:].split("-")[0]) < server.refuse_from}
    server.refuse_from, server.seen = None, []
    messages = []
    placed = rb.fetch_pinned([row], jobs=2, log=messages.append)
    assert placed[0].read_bytes() == big
    again = {wanted for _, wanted in server.seen}
    assert first and not again & first, "a chunk the first fetch finished was fetched again"
    assert "resumed from an earlier fetch" in messages[-1]


def test_resumed_bytes_that_do_not_hash_to_the_pin_are_fetched_again(tmp_path, server):
    big = _payload(300_000, b"qs")
    server.files["/fix/qr_acr_qs"] = big
    row = _row(tmp_path, "qr_acr_qs", big)
    state = rb._RangedFile("qr_acr_qs", rb.URLS["test-origin"].format(path="fix/qr_acr_qs"), row["sha256"],
                           len(big), row["target"], 64 << 10)
    with open(state.partial, "r+b") as out:
        out.write(b"\xff" * (64 << 10))         # chunk 0 claimed finished but holding other bytes
    with open(state.journal, "ab") as journal:
        journal.write(b"0\n")
    placed = rb.fetch_pinned([row], jobs=2, log=lambda message: None)
    assert placed[0].read_bytes() == big


def test_a_server_without_ranges_gets_one_connection_and_the_same_bytes(tmp_path, server):
    big = _payload(400_000, b"cam")
    server.files["/fix/CAM_ABS_DATA.DBL"] = big
    server.ranges = False
    placed = rb.fetch_pinned([_row(tmp_path, "CAM_ABS_DATA.DBL", big)], jobs=3, log=lambda message: None)
    assert placed[0].read_bytes() == big


def test_bytes_that_differ_from_the_pin_are_never_published(tmp_path, server):
    big = _payload(300_000, b"pinned")
    server.files["/fix/qr_acr_qg"] = _payload(300_000, b"other")
    row = _row(tmp_path, "qr_acr_qg", big)
    with pytest.raises(rb.PhysicsResourceError, match="not the pinned"):
        rb.fetch_pinned([row], jobs=2, log=lambda message: None)
    assert not row["target"].exists() and not list(row["target"].parent.glob(".partial-*"))
