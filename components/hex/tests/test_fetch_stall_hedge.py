"""A stalled source connection is hedged on a new connection instead of holding the fetch for the socket timeout.

THE BREAKAGE (hex LAM vs zoom-globe measurement, reference system, 2026-10-27): one GFS f006 byte-range request stalled on
its S3 connection and the region's boundaries waited the full 300 s socket timeout before retrying it.  These tests
run the real fetch code against a local HTTP server that forces that stall (a connection that is accepted and never
answered, or one that stops part way through its body) and require the bytes to arrive, unchanged, a few stall
windows after the stall instead of a socket timeout later.
"""

from __future__ import annotations

import bz2
import http.server
import random
import socketserver
import threading
import time
from pathlib import Path

import pytest

from woof.hex import init_sources as src

STALL = 0.5
DEAD = 20.0
PAYLOAD = random.Random(20261027).randbytes(3 << 20)


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _Origin:
    """Objects by path, with byte ranges and HEAD.  ``stall[(path, range header)]`` holds how many of the next
    requests for that range stall, and how: ``"silent"`` (accepted, never answered) or ``"mid-body"`` (headers and
    half the body, then nothing)."""

    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.stall: dict[tuple[str, str | None], list[str]] = {}
        self.release = threading.Event()
        self.requests: list[tuple[str, str | None]] = []
        origin = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _serve(self, head: bool) -> None:
                key = (self.path, self.headers.get("Range"))
                origin.requests.append(key)
                plan = origin.stall.get(key) or []
                how = plan.pop(0) if plan else None
                body = origin.objects.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if how == "silent":
                    origin.release.wait(DEAD)
                    return
                status, first, last = 200, 0, len(body) - 1
                if key[1]:
                    spec = key[1].split("=", 1)[1]
                    a, b = spec.split("-")
                    first, last, status = int(a), (int(b) if b else len(body) - 1), 206
                piece = body[first:last + 1]
                self.send_response(status)
                self.send_header("Content-Length", str(len(piece)))
                self.send_header("Accept-Ranges", "bytes")
                if status == 206:
                    self.send_header("Content-Range", f"bytes {first}-{last}/{len(body)}")
                self.end_headers()
                if head:
                    return
                if how == "mid-body":
                    self.wfile.write(piece[: len(piece) // 2])
                    self.wfile.flush()
                    origin.release.wait(DEAD)
                    return
                self.wfile.write(piece)

            def do_GET(self):
                self._serve(head=False)

            def do_HEAD(self):
                self._serve(head=True)

        self.server = _Server(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def origin(monkeypatch):
    monkeypatch.setattr(src, "FETCH_STALL_SECONDS", STALL)
    monkeypatch.setattr(src, "FETCH_DEAD_SECONDS", DEAD)
    monkeypatch.setattr(src, "_HEDGES", [])
    server = _Origin({"/gfs.f006": PAYLOAD})
    yield server
    server.close()


@pytest.mark.parametrize("how", ["silent", "mid-body"])
def test_a_stalled_range_request_is_hedged_on_a_new_connection(origin, how):
    span = (1 << 20, (2 << 20) - 1)
    origin.stall[("/gfs.f006", f"bytes={span[0]}-{span[1]}")] = [how]
    started = time.monotonic()
    body = src._http_get(origin.base + "/gfs.f006", byte_range=span)
    took = time.monotonic() - started
    assert body == PAYLOAD[span[0]:span[1] + 1]
    assert took < 4 * STALL + 2.0, f"took {took:.1f} s: the stall was waited out, not hedged"
    # The stalled request is hedged on a new connection, and every hedge follows the rule: no byte from any live
    # copy for STALL.  How many hedges that takes is the host's: under load the hedge copy itself can send nothing
    # for STALL and is hedged once more, as the rule requires (two hedges in the hex 0.4.12 gate), so the count is
    # not fixed.  Only this one request was made, so every hedge is for its range, numbered from copy 2 on.
    events = list(src._HEDGES)
    assert events and all(event["range"] == f"bytes={span[0]}-{span[1]}" for event in events)
    assert events[0]["copy"] == 2
    assert all(event["no_byte_for_s"] >= STALL for event in events)


def test_the_selected_records_arrive_whole_and_in_order_when_one_piece_stalls(origin, tmp_path):
    """The exact path that stalled: an NCEP index, its records cut into range pieces fetched in parallel."""

    grib = b"GRIB" + random.Random(6).randbytes(12 << 20)[4:]
    origin.objects["/gfs.t18z.pgrb2.0p25.f006"] = grib
    offsets = [0, 3_000_000, 7_000_000, 10_500_000, 11_000_000]
    names = ["UGRD:10 m above ground", "VGRD:10 m above ground", "TMP:2 m above ground", "PRMSL:mean sea level",
             "HGT:surface"]
    idx = "".join(f"{n + 1}:{offset}:d=2025051918:{name}:6 hour fcst:\n"
                  for n, (offset, name) in enumerate(zip(offsets, names)))
    origin.objects["/gfs.t18z.pgrb2.0p25.f006.idx"] = idx.encode()
    # The middle piece stalls silently, the way the S3 connection did.
    pieces = src.coalesce_spans([(0, 10_499_999)])
    assert len(pieces) == 3
    stalled = f"bytes={pieces[1][0]}-{pieces[1][1]}"
    origin.stall[("/gfs.t18z.pgrb2.0p25.f006", stalled)] = ["silent"]
    plan = src.ProductPlan(role="atmos", transport="https", filename="gfs.f006.grib2",
                           url=origin.base + "/gfs.t18z.pgrb2.0p25.f006",
                           index_url=origin.base + "/gfs.t18z.pgrb2.0p25.f006.idx",
                           index_select={"records": [["UGRD|VGRD", "10 m above ground"], ["TMP", "2 m above ground"]]},
                           index_format="ncep-idx")
    started = time.monotonic()
    [receipt] = src.acquire([plan], tmp_path, cycle=None, jobs=4)
    took = time.monotonic() - started
    want = grib[:10_500_000]
    assert (tmp_path / "gfs.f006.grib2").read_bytes() == want
    assert took < 4 * STALL + 2.0, f"took {took:.1f} s"
    # The stalled piece is hedged; any other hedge is a piece of this plan that also sent nothing for STALL on a
    # loaded host (the rule, not a fixed count).
    hedged = [event["range"] for event in receipt["stalls_hedged"]]
    assert stalled in hedged
    assert set(hedged) <= {f"bytes={first}-{last}" for first, last in pieces}
    assert all(event["no_byte_for_s"] >= STALL for event in receipt["stalls_hedged"])
    assert receipt["requests"] == len(pieces) and receipt["records"] == 3
    assert not list(tmp_path.glob("*.part*"))


def test_a_whole_bz2_object_on_one_stream_is_hedged_and_the_loser_leaves_nothing(origin, tmp_path):
    """The single-stream path (ICON's bz2 objects): every copy streams to its own part file; the first complete one
    is the product and a stalled copy that wakes later drops its file."""

    text = PAYLOAD[: 1 << 20]
    packed = bz2.compress(text[: 1 << 19]) + bz2.compress(text[1 << 19:])  # two concatenated streams
    origin.objects["/icon_global_T_2M.grib2.bz2"] = packed
    origin.stall[("/icon_global_T_2M.grib2.bz2", None)] = ["mid-body"]
    plan = src.ProductPlan(role="t2m", transport="https", filename="T_2M.grib2", codec="bz2",
                           url=origin.base + "/icon_global_T_2M.grib2.bz2", index_url=None)
    target = tmp_path / plan.filename
    started = time.monotonic()
    how = src._fetch_whole_from(plan, target, plan.url)
    took = time.monotonic() - started
    assert target.read_bytes() == text
    assert took < 4 * STALL + 2.0
    # The stalled copy is hedged on a new connection, and every hedge follows the rule: no byte from any live copy
    # for STALL.  How many hedges that takes is the host's: at load ~55 (hex 0.5.0 sealing pass) the first hedge
    # copy itself sent nothing for STALL and was hedged once more, as the rule requires, so the count is not fixed.
    events = [event for event in src._HEDGES if event["url"] == plan.url]
    assert how["hedges"] >= 1 and how["hedges"] == len(events) and how["attempts"] == how["hedges"] + 1
    assert events[0]["copy"] == 2 and all(event["no_byte_for_s"] >= STALL for event in events)
    origin.release.set()  # the stalled copy wakes, finds the race decided and drops its part file
    deadline = time.monotonic() + 5.0
    while list(tmp_path.glob("*.part*")) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not list(tmp_path.glob("*.part*"))


def test_a_cut_body_is_retried_not_kept(origin, monkeypatch):
    """A connection that closes early returns a short body without an error; the declared length catches it."""

    monkeypatch.setattr(src, "FETCH_STALL_SECONDS", 30.0)
    calls = {"n": 0}
    real = src._MemorySink.finish

    def cut_first(self, headers):
        calls["n"] += 1
        if calls["n"] == 1:
            self.blocks = self.blocks[:1]
        return real(self, headers)

    monkeypatch.setattr(src._MemorySink, "finish", cut_first)
    monkeypatch.setattr(src.time, "sleep", lambda s: None)
    assert src._http_get(origin.base + "/gfs.f006") == PAYLOAD
    assert calls["n"] == 2


def test_a_missing_object_still_refuses_at_once_by_address(origin):
    started = time.monotonic()
    with pytest.raises(src.InitSourceRefusal, match="answered HTTP 404"):
        src._http_get(origin.base + "/not-posted.f006")
    assert time.monotonic() - started < 2.0


def test_the_object_size_head_is_hedged_too(origin):
    origin.stall[("/gfs.f006", None)] = ["silent"]
    started = time.monotonic()
    assert src._object_size(origin.base + "/gfs.f006") == len(PAYLOAD)
    assert time.monotonic() - started < 4 * STALL + 2.0
