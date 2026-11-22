"""A host's ``transfer`` row: slow connections asked again, slow hosts widened.

Named breakage (WOOF canary row icon-offgrid-3-1-0.5, 2026-11-11 02:00-05:30Z):
DWD's open-data server served the ICON 00Z cycle at 3.9 s per 3.3 MB object
at the median and up to 54 s, and a 236-object start took 132-190 s instead
of 35 s.  These tests serve objects from a local server that trickles or
stalls on chosen connections, and check that the transport asks for the rest
on a fresh connection by byte range, starts over on a changed object, leaves
a host with no row exactly as before, and that the pool widens a slow host
and only a slow host.
"""

from __future__ import annotations

import hashlib
import http.server
import threading
import time

import pytest

from woof import fetch_endpoints, fetch_pool, fetch_ranges

SIZE = 600_000
BODY = bytes((i * 7 + 3) % 251 for i in range(SIZE))
ETAG = '"test-etag-1"'


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


def _serve(plan):
    """A server whose Nth GET behaves as ``plan(n, start)`` says.

    ``plan`` returns ("fast" | "trickle" | "stall" | "changed").  Every
    request's Range header is recorded on ``server.asked``.
    """

    lock = threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802
            with lock:
                n = len(self.server.asked)
                self.server.asked.append(self.headers.get("Range"))
            body = BODY
            start = 0
            ask = self.headers.get("Range")
            mode = plan(n, ask)
            if mode == "changed":
                body = bytes(reversed(BODY))
            if ask and mode != "changed" and self.headers.get("If-Range") == ETAG:
                start = int(ask.split("=")[1].split("-")[0])
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{SIZE - 1}/{SIZE}")
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(body) - start))
            self.send_header("ETag", ETAG if mode != "changed" else '"test-etag-2"')
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            data = body[start:]
            try:
                if mode == "trickle":
                    for i in range(0, len(data), 2_000):
                        self.wfile.write(data[i:i + 2_000])
                        self.wfile.flush()
                        time.sleep(0.05)          # ~40 kB/s
                elif mode == "stall":
                    self.wfile.write(data[:50_000])
                    self.wfile.flush()
                    time.sleep(5)
                else:
                    self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    server = _Server(("127.0.0.1", 0), Handler)
    server.asked = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def watched(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOF_FETCH_CONNECTION_DIR", str(tmp_path / "slots"))
    rows = {}

    def watch(host):
        return rows.get(host)

    monkeypatch.setattr(fetch_endpoints, "transfer_watch", watch)
    return rows


def _row(host, **override):
    values = dict(host=host, idle_s=1.0, slow_after_s=0.5, slow_below_bytes_per_s=150_000,
                  fresh_connections=2, wide_files_in_flight=6,
                  wide_below_bytes_per_s=1_000_000, sample=3, sample_min_bytes=0, why="test")
    values.update(override)
    return fetch_endpoints.TransferWatch(**values)


def _get(server, tmp_path, name="obj.grib2.bz2.part"):
    url = f"http://127.0.0.1:{server.server_address[1]}/{name[:-5]}"
    part = tmp_path / name
    written, digest = fetch_ranges.download(url, part, user_agent="test", timeout=30)
    return written, digest, part


def test_a_trickling_connection_is_asked_again_for_the_rest(watched, tmp_path):
    server = _serve(lambda n, ask: "trickle" if n == 0 else "fast")
    host = f"127.0.0.1:{server.server_address[1]}"
    watched[host] = _row(host)
    started = time.monotonic()
    written, digest, part = _get(server, tmp_path)
    took = time.monotonic() - started
    server.shutdown()
    assert written == SIZE and part.read_bytes() == BODY
    assert digest == hashlib.sha256(BODY).hexdigest()
    assert server.asked[0] is None
    # The second ask is a byte range from what the first connection gave.
    assert len(server.asked) == 2 and server.asked[1].startswith("bytes=")
    resumed_at = int(server.asked[1].split("=")[1].rstrip("-"))
    assert 0 < resumed_at < SIZE
    # The whole object at 40 kB/s would take 15 s.
    assert took < 5


def test_a_stalled_connection_is_asked_again_after_idle_s(watched, tmp_path):
    server = _serve(lambda n, ask: "stall" if n == 0 else "fast")
    host = f"127.0.0.1:{server.server_address[1]}"
    watched[host] = _row(host, slow_below_bytes_per_s=0)
    written, digest, part = _get(server, tmp_path)
    server.shutdown()
    assert part.read_bytes() == BODY
    assert server.asked[1] == "bytes=50000-"


def test_a_changed_object_starts_over_whole(watched, tmp_path):
    server = _serve(lambda n, ask: "trickle" if n == 0 else "changed")
    host = f"127.0.0.1:{server.server_address[1]}"
    watched[host] = _row(host)
    written, digest, part = _get(server, tmp_path)
    server.shutdown()
    assert part.read_bytes() == bytes(reversed(BODY))
    assert digest == hashlib.sha256(bytes(reversed(BODY))).hexdigest()


def test_the_last_connection_is_only_held_to_idle_s(watched, tmp_path):
    # Every connection trickles: after the allowed fresh connections the
    # file still arrives (a host slow everywhere delivers, never fails).
    server = _serve(lambda n, ask: "trickle")
    host = f"127.0.0.1:{server.server_address[1]}"
    watched[host] = _row(host, fresh_connections=1)
    written, digest, part = _get(server, tmp_path)
    server.shutdown()
    assert part.read_bytes() == BODY
    assert len(server.asked) == 2


def test_a_stall_on_the_last_connection_is_the_ladders_timeout(watched, tmp_path):
    server = _serve(lambda n, ask: "stall")
    host = f"127.0.0.1:{server.server_address[1]}"
    watched[host] = _row(host, fresh_connections=1, slow_below_bytes_per_s=0)
    with pytest.raises(TimeoutError):
        _get(server, tmp_path)
    server.shutdown()
    assert len(server.asked) == 2


def test_a_host_with_no_row_is_moved_as_before(watched, tmp_path):
    server = _serve(lambda n, ask: "trickle")
    written, digest, part = _get(server, tmp_path)
    server.shutdown()
    assert part.read_bytes() == BODY and server.asked == [None]


def _jobs(host, count, seconds, size=1_000_000):
    live = {"now": 0, "most": 0}
    lock = threading.Lock()

    def action():
        with lock:
            live["now"] += 1
            live["most"] = max(live["most"], live["now"])
        time.sleep(seconds)
        with lock:
            live["now"] -= 1
        return {"bytes": size, "downloaded": True}

    jobs = [fetch_pool.TransferJob(name=f"f{i}", url=f"https://{host}/f{i}", action=action)
            for i in range(count)]
    return jobs, live


def test_a_slow_host_is_widened_and_a_fast_one_is_not(watched, monkeypatch):
    monkeypatch.setattr(fetch_pool.fetch_connections, "process_share", lambda: 24)
    host = "slow.example"
    watched[host] = _row(host, wide_files_in_flight=6, wide_below_bytes_per_s=1_000_000, sample=3)
    # 1 MB in 0.2 s = 5 MB/s: never widened.
    jobs, live = _jobs(host, 12, 0.2)
    _entries, receipt = fetch_pool.run_transfers(jobs, workers=2)
    assert live["most"] == 2 and "host_widened" not in receipt
    # 1 MB in 2 s = 0.5 MB/s: widened to 6 after 3 files.
    jobs, live = _jobs(host, 12, 2.0)
    _entries, receipt = fetch_pool.run_transfers(jobs, workers=2)
    assert live["most"] == 6
    assert receipt["host_widened"][host]["files_in_flight"] == 6
    # An exact width asked for is kept.
    jobs, live = _jobs(host, 8, 2.0)
    _entries, receipt = fetch_pool.run_transfers(jobs, workers=2, widen=False)
    assert live["most"] == 2


def test_small_files_do_not_widen_a_fast_host(watched, monkeypatch):
    # Small files are mostly connection set-up: 0.2 MB in 0.5 s reads as
    # 0.4 MB/s however fast the host is.  With a size floor they are not
    # counted, and the large files (1 MB in 0.2 s) keep the pool's width.
    monkeypatch.setattr(fetch_pool.fetch_connections, "process_share", lambda: 24)
    host = "fast.example"
    watched[host] = _row(host, wide_files_in_flight=6, sample=3, sample_min_bytes=500_000)
    small, live_small = _jobs(host, 6, 0.5, size=200_000)
    large, live_large = _jobs(host, 6, 0.2)
    _entries, receipt = fetch_pool.run_transfers(small + large, workers=2)
    assert "host_widened" not in receipt
    assert max(live_small["most"], live_large["most"]) <= 2
