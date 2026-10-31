"""The as-posted fetch against a local replay of a posting cycle (A136 L2).

A local HTTP server serves one cycle's objects of a real table route and
reveals lead ``k`` only once an injected clock passes its reveal time, so
the loop is driven exactly as a live cycle drives it: asked by the
polling rule, fetched the moment one host holds the whole lead, verified,
and published as a per-lead marker in lead order.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import http.server
import json
import threading
import time
from pathlib import Path
from urllib.request import urlopen

import pytest

from woof import fetch
from woof import fetch_as_posted
from woof import fetch_endpoints
from woof import fetch_pool
from woof import fetch_routes
from woof import source_posting as rows
from woof import source_readiness as readiness
from woof.fetch_endpoints import Endpoint


@pytest.fixture(autouse=True)
def _python_route_transport(monkeypatch):
    """These replays serve stand-in bytes, not GRIB2 messages.

    Where a built rw_fetch is present, a GRIB2 route streams through it and
    it refuses a payload without a GRIB2 indicator; the stream door has its
    own suite (tests/test_fetch_stream_tail.py). The posting schedule is what
    is under test here, so the replays keep the Python object transport.
    """
    monkeypatch.setattr(fetch, "_route_stream_downloader", lambda plan, args: None)

SOURCE = "gefs"
CYCLE = datetime(2026, 9, 29, 12)


class Clock:
    """Virtual naive-UTC time: every wait the loop takes moves it on."""

    def __init__(self, start: datetime) -> None:
        self._now = start
        self._lock = threading.Lock()

    def now(self) -> datetime:
        with self._lock:
            return self._now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self._now += timedelta(seconds=max(float(seconds), 0.001))


def payload(name: str) -> bytes:
    body = hashlib.sha256(name.encode()).digest() * 8
    return b"GRIB" + body + b"7777"


class Replay:
    """One cycle's objects on a local host, each revealed at its time."""

    def __init__(self, clock: Clock, plan, reveal) -> None:
        self.clock = clock
        self.objects: dict[str, tuple[bytes, datetime | None]] = {}
        self.unheard_until: dict[str, datetime] = {}
        self.truncate_once: set[str] = set()
        self.gets: list[str] = []
        #: The status every HEAD gets instead of its answer (a mirror or
        #: cache that serves only GET), or None for a host that answers it.
        self.head_status: int | None = None
        #: ``(path, Range)`` of every GET that asked a byte range.
        self.ranged: list[tuple[str, str]] = []
        for obj in plan.objects:
            at = reveal(obj.lead if obj.lead in plan.leads else plan.leads[0])
            self.objects["/" + obj.key] = (payload(obj.key), at)
            idx = [row.idx_sidecar for row in plan.files if row.role == obj.role][0]
            if idx:
                self.objects["/" + obj.key + idx] = (b"1:0:d=x\n", at)
        replay = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def _answer(self, body: bool) -> None:
                path = self.path.split("/", 2)[-1]
                path = "/" + path
                now = replay.clock.now()
                if not body and replay.head_status is not None:
                    self.send_response(replay.head_status)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                blocked = replay.unheard_until.get(path)
                if blocked is not None and now < blocked:
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                found = replay.objects.get(path)
                if found is None or found[1] is None or now < found[1]:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = found[0]
                requested = self.headers.get("Range") if body else None
                if requested is not None:
                    first, _, last = requested.split("=", 1)[1].partition("-")
                    first = int(first)
                    last = int(last) if last else len(data) - 1
                    part = data[first:last + 1]
                    replay.ranged.append((path, requested))
                    self.send_response(206)
                    self.send_header("Content-Range",
                                     f"bytes {first}-{last}/{len(data)}")
                    self.send_header("Content-Length", str(len(part)))
                    self.end_headers()
                    self.wfile.write(part)
                    return
                if body and path in replay.truncate_once:
                    replay.truncate_once.discard(path)
                    data = data[:-4] + b"XXXX"
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if body:
                    replay.gets.append(path)
                    self.wfile.write(data)

            def do_HEAD(self):
                self._answer(False)

            def do_GET(self):
                self._answer(True)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/replay"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _whole_object_plans(resolve):
    """``resolve_request`` whose plans move each object whole (no record selection by its index)."""
    import dataclasses

    def plain(row):
        return dataclasses.replace(row, record_index=None) if getattr(row, "record_index", None) else row

    def whole(*args, **kwargs):
        plan = resolve(*args, **kwargs)
        changes = {name: tuple(plain(row) for row in getattr(plan, name))
                   for name in ("files", "primary_files", "supplement_files", "objects")
                   if isinstance(getattr(plan, name, None), tuple)}
        return dataclasses.replace(plan, **changes)

    return whole


@pytest.fixture
def replay(monkeypatch):
    """A factory: ``replay(reveal)`` serves the gefs 12 h window locally."""

    made = []

    def build(reveal, *, hours: int = 12):
        start = rows.expected_at(SOURCE, CYCLE, 0) - timedelta(minutes=10)
        clock = Clock(start)
        plan = fetch_routes.resolve_request(SOURCE, cycle=CYCLE, hours=hours,
                                            cadence=3)
        server = Replay(clock, plan, lambda lead: reveal(lead, clock))
        made.append(server)
        rung = Endpoint(name="nomads", base=server.base, retention_hours=None,
                        why="the local replay of a posting cycle")
        monkeypatch.setattr(fetch_endpoints, "serving_ladder",
                            lambda source_id, **_kw: (rung,))
        monkeypatch.setattr(fetch_endpoints, "SETTLE_BACKOFF_S", ())
        # The posting mechanics under test move whole objects. Since fix/gefs-record-subset the GEFS rows
        # take their mapped records by the NCEP .idx beside each object (tests/test_record_index.py covers
        # that); this replay serves stand-in objects with no NCEP inventory, so its plans drop the selection.
        monkeypatch.setattr(fetch_routes, "resolve_request", _whole_object_plans(fetch_routes.resolve_request))
        monkeypatch.setattr(fetch_as_posted, "now_utc", clock.now)
        monkeypatch.setattr(fetch_as_posted, "pause", clock.sleep)
        # A transfer that does not verify is asked again by the loop, not
        # after the transport's own real-time backoff.
        monkeypatch.setattr(fetch_routes, "_TRANSFER_ATTEMPTS", 1)
        monkeypatch.setattr(fetch_routes, "_retry_delay",
                            lambda error, attempt: None)
        return server, clock

    yield build
    for server in made:
        server.close()


def on_schedule(lead: int, clock: Clock) -> datetime:
    """Lead ``lead`` posts 20 s after the table says it will."""

    return rows.expected_at(SOURCE, CYCLE, lead) + timedelta(seconds=20)


def fetch_args(out: Path, *extra: str, hours: int = 12) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    fetch.register_cli(parser.add_subparsers())
    return parser.parse_args([
        "fetch", "--source", SOURCE, "--cycle", CYCLE.strftime("%Y-%m-%dT%H"),
        "--hours", str(hours), "--cadence", "3", "--out", str(out),
        "--fetch-workers", "1", *extra])


def markers(out: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((out / "posting").glob("f[0-9][0-9][0-9].json"))]


@pytest.mark.parametrize("outcome", ["posts", "stays_absent", "crosses_deadline"])
@pytest.mark.parametrize("failed_lead", [0, 1])
def test_legacy_absent_object_waits_then_posts_or_exits_75(
        tmp_path, monkeypatch, outcome, failed_lead):
    """A positive HEAD cannot turn a missing runtime object into a fatal
    transfer, including when that transfer itself crosses late_at."""
    from woof.rustwx_fetch import RwFetchError

    window = readiness.resolve_window("hrrr", CYCLE, 1)
    late = rows.expected_at("hrrr", CYCLE, failed_lead) + timedelta(minutes=1)
    clock = Clock(late - timedelta(seconds=1))
    lines = []
    loop = fetch_as_posted.PostingLoop(
        window, tmp_path, late_after_minutes=1, probe=lambda url: True,
        now=clock.now, sleep=clock.sleep, progress=lines.append)
    # Keep the virtual clock and the real loop on one thread.
    monkeypatch.setattr(loop, "watch", lambda: None)
    attempts = []

    def transfer(leads):
        attempts.append(clock.now())
        if len(attempts) == 1 or outcome == "stays_absent":
            if outcome == "crosses_deadline":
                clock.sleep(2)
            raise RwFetchError(
                "fetch", f"f{failed_lead:03d}: no source served this object -- aws: "
                "the GRIB object is not present at this source", returncode=1)
        path = tmp_path / "fetch-manifest.json"
        path.write_text(json.dumps({"files": []}), encoding="utf-8")
        return path

    if outcome == "posts":
        fetch_as_posted.fetch_legacy_as_posted(loop, transfer)
        assert len(attempts) == 2
        assert [item["lead"] for item in markers(tmp_path)] == [0, 1]
    else:
        with pytest.raises(fetch_as_posted.SourceBehind) as caught:
            fetch_as_posted.fetch_legacy_as_posted(loop, transfer)
        assert caught.value.exit_code == 75
        assert caught.value.record["last_answer"] == readiness.NOT_POSTED
        assert caught.value.record["lead"] == failed_lead
        assert caught.value.record["late_at"] == rows.instant(late)
        assert "has not posted by" in str(caught.value)
        assert clock.now() >= late
        assert len(attempts) == (1 if outcome == "crosses_deadline" else 2)
        assert markers(tmp_path) == []
    if outcome != "crosses_deadline":
        assert any("is not fully posted yet" in line for line in lines)


def test_markers_land_in_lead_order_as_each_lead_posts(replay, tmp_path):
    server, clock = replay(on_schedule)
    out = tmp_path / "gefs"
    seen_order: list[int] = []
    real_publish = fetch_as_posted.PostingLoop.publish

    def publish(self, lead, objects, **named):
        seen_order.append(lead)
        # A lead's marker is written only once it is on disk and verified,
        # and never before the host held it.
        assert clock.now() >= on_schedule(lead, clock)
        return real_publish(self, lead, objects, **named)

    fetch_as_posted.PostingLoop.publish = publish
    try:
        assert fetch.fetch_main(fetch_args(out)) == 0
    finally:
        fetch_as_posted.PostingLoop.publish = real_publish
    assert seen_order == [0, 3, 6, 9, 12]
    written = markers(out)
    assert [marker["lead"] for marker in written] == [0, 3, 6, 9, 12]
    for marker in written:
        assert marker["schema"] == "gpuwm.posted-lead.v1"
        assert marker["source"] == SOURCE and marker["cycle"] == "2026-09-29T12"
        # The lead's two fetched objects, each with its own lead, and apart
        # from them the pair the route composes from them: an as-posted
        # mapped seal holds the composed primary's manifest row to this
        # marker (A136 L3 ii), and ``objects`` stays what was fetched.
        fetched = marker["objects"]
        composed = marker["composed"]
        assert len(fetched) == 2 and len(composed) == 1
        for item in fetched:
            assert set(item) == {"role", "name", "url", "endpoint", "bytes",
                                 "sha256", "lead"}
            assert item["lead"] == marker["lead"]
        assert set(composed[0]) == {"name", "bytes", "sha256", "lead",
                                    "parts"}
        assert composed[0]["lead"] == marker["lead"]
        assert sorted(composed[0]["parts"]) == sorted(
            item["name"] for item in fetched)
        for item in [*fetched, *composed]:
            data = (out / item["name"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
            assert len(data) == item["bytes"]
        assert marker["first_seen_at"] <= marker["fetched_at"]
    schedule = json.loads((out / "posting" / "schedule.json").read_text())
    for key in ("source", "member", "cycle", "as_posted", "shape", "streams",
                "why", "late_after_minutes", "start_needs",
                "expected_ready_at", "expected_final_at", "table_sha256"):
        assert key in schedule, key
    assert schedule["schema"] == "gpuwm.posting-schedule.v1"
    assert [need["role"] for need in schedule["start_needs"]] == [
        "analysis", "first_boundary"]
    for row in schedule["leads"]:
        assert set(("lead", "valid_time", "expected_at", "late_at",
                    "first_seen_at", "fetched_at", "endpoint",
                    "state")) <= set(row)
        assert row["state"] == "ready" and row["endpoint"] == "nomads"
    manifest = json.loads((out / "fetch-manifest.json").read_text())
    assert manifest["complete"] is True
    assert not (out / "posting" / "failed.json").exists()


def test_lead_ready_says_the_fetched_bytes_of_a_route_that_composes(
        replay, tmp_path):
    """DESIGN A136 3.5: ``lead_ready.bytes`` is what was fetched and verified.

    gefs composes each lead's two objects into one file, which is its
    parts' bytes again; with that file among the marker's objects, go's
    chain stream said every composed lead moved twice its bytes.  The
    markers here are the real loop's, relayed by the real chain stream.
    """

    from woof.chain_events import (CHAIN_EVENTS_FILENAME, GoChainEvents,
                                    read_chain_events)

    replay(on_schedule)
    out = tmp_path / "gefs"
    run = tmp_path / "run"
    run.mkdir()
    # The stream opens before the fetch, as `go` opens it: the relay carries only markers written since it
    # opened (POSTING_FRESH_SLACK_MS), and with the fetch first a loaded host dropped the earliest leads
    # (2.9.2 public-CI rehearsal routes B and C: leads 0 and 3, then 0 to 9, missing).
    chain = GoChainEvents()
    chain.open(tmp_path / CHAIN_EVENTS_FILENAME,
               plan={"data": out, "run": run, "render": None})
    assert fetch.fetch_main(fetch_args(out)) == 0
    fetched = {}
    for marker in markers(out):
        assert marker["composed"], "gefs composes each lead's parts"
        files = [out / item["name"] for item in marker["objects"]]
        fetched[marker["lead"]] = sum(path.stat().st_size for path in files)
        composed = sum(item["bytes"] for item in marker["composed"])
        assert composed == fetched[marker["lead"]]
    chain.finish(status="SUCCESS")
    ready = {record["lead"]: record["bytes"]
             for record in read_chain_events(tmp_path / CHAIN_EVENTS_FILENAME)
             if record["event"] == "lead_ready"}
    assert ready == fetched


def test_a_late_lead_stops_with_75_and_keeps_the_prefix(replay, tmp_path):
    def reveal(lead, clock):
        return None if lead == 6 else on_schedule(lead, clock)

    server, clock = replay(reveal)
    out = tmp_path / "gefs"
    assert fetch.fetch_main(fetch_args(out)) == 75
    failed = json.loads((out / "posting" / "failed.json").read_text())
    assert failed["code"] == "source_behind" and failed["lead"] == 6
    assert failed["heard"] is True and failed["last_answer"] == "not_posted"
    # The row's own budget, not a literal: the L1 follow-ups (d25ed8127)
    # raised gefs from 60 to 120 min, its measured late spread plus 30,
    # and a literal 60 here failed once both lanes met on integrate/2.8.
    budget = rows.late_after_minutes(SOURCE)
    assert budget is not None and budget >= 60
    assert failed["late_at"] == rows.instant(
        rows.expected_at(SOURCE, CYCLE, 6) + timedelta(minutes=budget))
    assert failed["late_after_minutes"] == budget
    assert "has not posted by" in failed["message"]
    # Not asked before its due time, and stopped only past its late time.
    assert clock.now() >= readiness.parse_instant(failed["late_at"])
    assert [marker["lead"] for marker in markers(out)] == [0, 3]
    manifest = json.loads((out / "fetch-manifest.json").read_text())
    assert manifest["complete"] is False
    assert sorted({entry["lead"] for entry in manifest["files"]}) == [0, 3]
    schedule = json.loads((out / "posting" / "schedule.json").read_text())
    states = {row["lead"]: row["state"] for row in schedule["leads"]}
    assert states[0] == "ready" and states[3] == "ready" and states[6] == "late"


def test_a_host_not_heard_is_never_counted_absent(replay, tmp_path):
    server, clock = replay(on_schedule)
    key = [path for path in server.objects if path.endswith(".f003")][0]
    # The host answers 503 for f003 for 40 minutes past its posting: the
    # loop keeps asking (not heard is not absent) and fetches it after.
    server.unheard_until[key] = on_schedule(3, clock) + timedelta(minutes=40)
    out = tmp_path / "gefs"
    assert fetch.fetch_main(fetch_args(out)) == 0
    assert [marker["lead"] for marker in markers(out)] == [0, 3, 6, 9, 12]
    seen = {marker["lead"]: marker["first_seen_at"] for marker in markers(out)}
    assert readiness.parse_instant(seen[3]) >= server.unheard_until[key]


def test_a_host_never_heard_is_said_as_not_heard(replay, tmp_path):
    server, clock = replay(on_schedule)
    for path in list(server.objects):
        if path.endswith(".f003"):
            server.unheard_until[path] = datetime(2100, 1, 1)
    out = tmp_path / "gefs"
    assert fetch.fetch_main(fetch_args(out)) == 75
    failed = json.loads((out / "posting" / "failed.json").read_text())
    assert failed["lead"] == 3 and failed["heard"] is False
    assert failed["last_answer"] == "not_heard"
    assert "could not be heard" in failed["message"]
    assert "has not posted" not in failed["message"]


@pytest.mark.parametrize("status", [405, 501])
def test_a_host_that_refuses_head_but_serves_get_is_heard(replay, tmp_path,
                                                          status):
    """A mirror or cache that serves GET but refuses HEAD.

    The watch asks each lead by HEAD; the refusal is asked again as a GET
    of the first byte, so each lead is heard absent until it posts and
    dated when it does.  Read as not heard, every lead went to the
    transfer unasked and its marker could not say when it posted.
    """

    server, clock = replay(on_schedule)
    server.head_status = status
    out = tmp_path / "gefs"
    assert fetch.fetch_main(fetch_args(out)) == 0
    written = markers(out)
    assert [marker["lead"] for marker in written] == [0, 3, 6, 9, 12]
    for marker in written:
        # Heard not posted before it posted, then heard posted.
        assert marker["posted_when_first_asked"] is False
        assert (readiness.parse_instant(marker["first_seen_at"])
                >= on_schedule(marker["lead"], clock))
    assert server.ranged
    assert {wanted for _path, wanted in server.ranged} == {
        fetch_endpoints.CONFIRM_RANGE}
    assert not (out / "posting" / "failed.json").exists()


def test_a_verification_failure_is_fetched_again(replay, tmp_path):
    server, clock = replay(on_schedule)
    key = [path for path in server.objects if path.endswith("pgrb2a.0p50.f006")][0]
    server.truncate_once.add(key)
    out = tmp_path / "gefs"
    assert fetch.fetch_main(fetch_args(out)) == 0
    # The truncated GET, then the one asked again on the next round.
    assert server.gets.count(key) == 2
    marker = [m for m in markers(out) if m["lead"] == 6][0]
    item = [i for i in marker["objects"] if i["name"].endswith("pgrb2a.0p50.f006")][0]
    assert item["sha256"] == hashlib.sha256(payload(key[1:])).hexdigest()


def test_a_late_launch_marks_every_lead_up_when_first_asked(replay, tmp_path):
    """Through the fetch door: a window whose final lead had posted before
    the fetch began moves with no per-lead ask, and every lead says it was
    already up (measured live on gefs 2026-10-01T06, six hours late: each
    lead read 375 to 377 min after its scheduled time, unflagged)."""

    server, clock = replay(lambda lead, clock: clock.now() - timedelta(hours=1))
    out = tmp_path / "gefs"
    args = fetch_args(out)
    args.fetch_workers = 4
    assert fetch.fetch_main(args) == 0
    assert [marker["posted_when_first_asked"] for marker in markers(out)] == [
        True] * 5
    schedule = json.loads((out / "posting" / "schedule.json").read_text())
    assert [row["posted_when_first_asked"] for row in schedule["leads"]] == [
        True] * 5
    assert [marker["lead"] for marker in markers(out)] == [0, 3, 6, 9, 12]


def comparable(manifest: dict) -> dict:
    """A manifest without its timing and posting fields (DESIGN A136 2.3).

    Where and when each object moved -- its URL and endpoint, seconds,
    whether this run moved or reused it, the pool receipt, the host
    receipt -- and the as-posted ``complete`` flag.
    """

    kept = {key: value for key, value in manifest.items()
            if key not in ("concurrency", "endpoints", "complete")}
    kept["files"] = [{key: value for key, value in entry.items()
                      if key not in ("url", "endpoint", "seconds", "reused")}
                     for entry in manifest["files"]]
    return kept


def test_as_posted_manifest_equals_the_one_shot_manifest(replay, tmp_path):
    server, clock = replay(on_schedule)
    posted = tmp_path / "posted"
    assert fetch.fetch_main(fetch_args(posted)) == 0
    whole = tmp_path / "whole"
    assert fetch.fetch_main(fetch_args(whole, "--whole-cycle")) == 0
    assert not (whole / "posting").exists()
    one = json.loads((posted / "fetch-manifest.json").read_text())
    two = json.loads((whole / "fetch-manifest.json").read_text())
    # Since 2.8.9 every route publishes verified lead prefixes while later
    # inputs download, so a whole-cycle manifest is written by the same
    # lead path and closes with complete=true; it never says false.
    assert two.get("complete", True) is True
    assert comparable(one) == comparable(two)
    assert (posted / "SHA256SUMS").read_bytes() == (whole / "SHA256SUMS").read_bytes()


def test_wait_for_is_as_posted_for_a_table_route(replay, tmp_path):
    server, clock = replay(on_schedule)
    out = tmp_path / "gefs"
    args = fetch_args(out, "--wait-for", "--wait-timeout-minutes", "600")
    assert args.as_posted is True
    assert fetch.fetch_main(args) == 0
    assert [marker["lead"] for marker in markers(out)] == [0, 3, 6, 9, 12]


def test_the_whole_window_cap_stops_with_75(replay, tmp_path):
    def reveal(lead, clock):
        return on_schedule(lead, clock) + timedelta(minutes=30 * (lead > 0))

    server, clock = replay(reveal)
    out = tmp_path / "gefs"
    assert fetch.fetch_main(fetch_args(
        out, "--wait-timeout-minutes", "20")) == 75
    failed = json.loads((out / "posting" / "failed.json").read_text())
    assert failed["budget"] == "wait_timeout_minutes"
    assert "--wait-timeout-minutes" in failed["message"]


@pytest.mark.parametrize("extra, said", [
    (("--whole-cycle", "--late-after-minutes", "30"), "--whole-cycle waits for nothing"),
    (("--late-after-minutes", "0"), "positive and finite"),
    (("--no-probe",), "--no-probe belongs to --readiness"),
])
def test_posting_flags_refuse_what_they_cannot_mean(tmp_path, extra, said):
    with pytest.raises(ValueError, match=said):
        fetch.fetch_main(fetch_args(tmp_path / "x", *extra))


@pytest.mark.parametrize("whole", [True, False])
def test_a_forced_legacy_refetch_leaves_no_marker_of_a_moved_file(
        tmp_path, monkeypatch, whole):
    """A forced refetch sets the old payloads aside; a lead marker left
    beside them would tell a reader (L3's PostedLeads, the tree consumer)
    that a lead is ready whose file is gone, until it is fetched again."""

    cycle = datetime(2026, 9, 29, 12)
    window = readiness.resolve_window("gfs", cycle, 3, cadence=3)
    out = tmp_path / "gfs"
    folder = out / "posting"
    folder.mkdir(parents=True)
    for lead in window.leads:
        (folder / fetch_as_posted.marker_name(lead)).write_text(json.dumps({
            "schema": fetch_as_posted.MARKER_SCHEMA, "source": "gfs",
            "member": None, "cycle": "2026-09-29T12", "lead": lead,
            "objects": [{"name": "old"}]}), encoding="utf-8")
    (out / "old").write_bytes(b"old")
    clock = Clock(rows.expected_at("gfs", cycle, window.leads[-1])
                  + timedelta(minutes=5))
    monkeypatch.setattr(fetch_as_posted, "pause", clock.sleep)
    loop = fetch_as_posted.PostingLoop(
        window, out, probe=lambda url: True, now=clock.now,
        sleep=clock.sleep, progress=lambda *_: None)
    seen = []

    def transfer(hours, force):
        seen.append(sorted(path.name for path in
                           folder.glob("f[0-9][0-9][0-9].json")))
        manifest = out / "fetch-manifest.json"
        manifest.write_text(json.dumps({"files": [
            {"forecast_hour": lead, "name": f"new.f{lead:03d}", "role": "atmos",
             "url": "u", "endpoint": "nomads", "bytes": 3, "sha256": "x"}
            for lead in hours]}), encoding="utf-8")
        return manifest

    fetch._legacy_transfer(loop, transfer, list(window.leads), force=True,
                           whole=whole)
    assert seen[0] == []
    for lead in window.leads:
        marker = json.loads((folder / fetch_as_posted.marker_name(lead)).read_text())
        assert marker["objects"][0]["name"] == f"new.f{lead:03d}"


@pytest.mark.parametrize("whole", [True, False])
def test_a_native_batch_publishes_each_verified_hour_before_transfer_returns(
        tmp_path, monkeypatch, whole):
    """Native completion survives the prefix wrapper and starts preparation.

    The prefix arm publishes early through the incremental callback, then
    the wrapper's final publication must preserve those marker bytes and
    close its posting watch. Dropping that callback at the wrapper either
    raises on an undefined publisher or rewrites the markers at the end.
    """

    from tools import download_hrrr_native_subset as native_transport
    from test_fetch_request_identity import _dated_hrrr_product

    window = readiness.resolve_window("hrrr", CYCLE, 2, start_hour=1)
    out = tmp_path / "hrrr"
    clock = Clock(rows.expected_at("hrrr", CYCLE, window.leads[-1])
                  + timedelta(minutes=5))
    monkeypatch.setattr(fetch_as_posted, "pause", clock.sleep)
    monkeypatch.setattr(native_transport, "_download_product",
                        _dated_hrrr_product([]))
    # 6a69b356f061, lane/hrrr-statics, adds analyzed VEGFRA to the soil
    # stream. Keep this existing synthetic batch offline at the native
    # transport and numeric extractor seams, retaining the actual append,
    # digest, per-hour publication and repeated-marker checks below.
    from woof import runtime_surface_fetch, rustwx_fetch

    vegetation_selector = (("discipline", 2), ("category", 0), ("parameter", 4),
                           ("level_type", 1), ("level_value", 0), ("pdt", 0))

    def runtime_fetch(binary, **kwargs):
        assert binary == tmp_path / "runtime-fetch-fixture"
        assert kwargs["model"] == "hrrr" and kwargs["source"] == "aws"
        assert kwargs["date"] == f"{CYCLE:%Y%m%d}" and kwargs["cycle"] == CYCLE.hour
        assert kwargs["product"] == "sfc"
        assert kwargs["pattern_file"].read_text().splitlines() == ["VEG:surface"]
        lead, = kwargs["hours"]
        name = f"hrrr.t{CYCLE:%H}z.wrfsfcf{lead:02d}.grib2"
        data = HrrrIndexHeadRefused.VEGETATION_RECORD
        (kwargs["out"] / name).write_bytes(data)
        return {"files": [{"name": name, "grib_url": f"https://fixture.invalid/{name}",
                           "mode": "idx-subset",
                           "sha256": hashlib.sha256(data).hexdigest()}]}

    def extract_replay_record(source, output, numeric_selector):
        assert tuple(numeric_selector) == vegetation_selector
        assert source.read_bytes() == HrrrIndexHeadRefused.VEGETATION_RECORD
        output.write_bytes(source.read_bytes())

    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin",
                        lambda: tmp_path / "runtime-fetch-fixture")
    monkeypatch.setattr(rustwx_fetch, "probe_fetch_bin", lambda _: (True, "fixture"))
    monkeypatch.setattr(rustwx_fetch, "run_fetch", runtime_fetch)
    monkeypatch.setattr(runtime_surface_fetch, "extract_runtime_record",
                        extract_replay_record)
    last_posted = [whole]
    loop = fetch_as_posted.PostingLoop(
        window, out,
        probe=lambda url: (last_posted[0] or
                           f"f{window.leads[-1]:02d}.grib2" not in url),
        now=clock.now,
        sleep=clock.sleep, progress=lambda *_: None)
    written = []
    first_bytes = {}
    real_publish = loop.publish

    def publish(lead, objects):
        written.append(lead)
        result = real_publish(lead, objects)
        first_bytes[lead] = result.read_bytes()
        return result

    monkeypatch.setattr(loop, "publish", publish)
    transfer_returned = []

    def transfer(hours, force, ready=None):
        returned = [False]

        def hourly(lead, manifest):
            assert not returned[0]
            ready(lead, manifest)
            assert [row["lead"] for row in markers(out)] == written
            document = json.loads(manifest.read_text())
            soils = [item for item in document["files"] if item.get("role") == "soil"]
            assert soils
            assert all(item["records"] == native_transport.SOIL_RECORD_COUNT + 1
                       and item["runtime_surface"][0]["field"] == "VEGFRA"
                       for item in soils)
            for item in fetch_as_posted.legacy_objects(document, lead):
                path = out / item["name"]
                assert item["bytes"] == path.stat().st_size
                assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            # A repeated completion or the final batch publication must
            # keep the marker the preparation has already bound.
            clock.sleep(30)
            ready(lead, manifest)
            assert (loop.folder / fetch_as_posted.marker_name(lead)
                    ).read_bytes() == first_bytes[lead]

        result = fetch.fetch_hrrr(
            cycle=CYCLE, hours=tuple(hours), area=None, out=out,
            force=force, transport="s3", engine="python", file_workers=1,
            progress=lambda *_: None,
            on_hour_ready=hourly if ready is not None else None)
        returned[0] = True
        transfer_returned.append(True)
        last_posted[0] = True
        return result

    manifest = fetch._legacy_transfer(
        loop, transfer, window.leads, force=False, whole=whole,
        incremental_transfer=transfer)
    assert transfer_returned == [True] * (1 if whole else 2)
    assert written == list(window.leads)
    assert json.loads(manifest.read_text())["forecast_hours"] == list(window.leads)
    for lead, data in first_bytes.items():
        assert (loop.folder / fetch_as_posted.marker_name(lead)).read_bytes() == data
    if not whole:
        assert loop._watch_stop.is_set()
        assert loop._watcher is None or not loop._watcher.is_alive()


@pytest.mark.parametrize("changed", [False, True])
def test_an_early_native_completion_cannot_rebind_or_skip_a_posted_lead(
        tmp_path, changed):
    window = readiness.resolve_window("hrrr", CYCLE, 1, start_hour=1)
    out = tmp_path / "hrrr"
    clock = Clock(rows.expected_at("hrrr", CYCLE, window.leads[-1])
                  + timedelta(minutes=5))
    loop = fetch_as_posted.PostingLoop(
        window, out, probe=lambda _url: True, now=clock.now,
        sleep=clock.sleep, progress=lambda *_: None)

    def incremental(_hours, _force, ready):
        manifest = out / "fetch-manifest.json"
        lead = window.leads[0] if changed else window.leads[1]
        files = [{"forecast_hour": lead, "name": "native.grib2",
                  "role": "atmosphere", "bytes": 1, "sha256": "a" * 64,
                  "url": "u", "transport": "s3"}]
        manifest.write_text(json.dumps({"files": files}))
        ready(lead, manifest)
        files[0]["sha256"] = "b" * 64
        manifest.write_text(json.dumps({"files": files}))
        ready(lead, manifest)
        return manifest

    with pytest.raises(ValueError, match=("changed after its posted marker"
                                         if changed else "ordered prefix")):
        fetch._legacy_transfer(loop, lambda *_: None, window.leads,
                               force=False, whole=True,
                               incremental_transfer=incremental)


def test_a_pinned_native_prefix_waits_for_an_intermediate_s3_soil_object(
        tmp_path):
    """A later lead or NOMADS cannot authorize a missing pinned S3 object."""

    window = readiness.resolve_window(
        "hrrr", CYCLE, 3, start_hour=15, transport="s3")
    out = tmp_path / "hrrr"
    clock = Clock(rows.expected_at("hrrr", CYCLE, window.leads[-1])
                  + timedelta(minutes=5))
    soil_posted = [False]
    asked = []

    def probe(url):
        asked.append(url)
        # The operational mirror already serves every lead. Only the
        # chosen archive's pressure object for the intermediate lead lags.
        if "nomads.ncep.noaa.gov" in url:
            return True
        return soil_posted[0] or "wrfprsf17.grib2" not in url

    loop = fetch_as_posted.PostingLoop(
        window, out, probe=probe, now=clock.now, sleep=clock.sleep,
        progress=lambda *_: None)
    transfers = []
    first_markers = {}

    def incremental(hours, force, ready):
        assert not force
        if not soil_posted[0]:
            assert tuple(hours) == (15, 16)
        transfers.append(tuple(hours))
        manifest = out / "fetch-manifest.json"
        files = []
        for lead in hours:
            for kind in ("atmosphere", "soil"):
                name = f"{kind}-f{lead:03d}.grib2"
                data = payload(name)
                (out / name).write_bytes(data)
                files.append({"forecast_hour": lead, "name": name,
                              "role": kind, "bytes": len(data),
                              "sha256": hashlib.sha256(data).hexdigest(),
                              "url": "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/" + name,
                              "transport": "s3"})
            manifest.write_text(json.dumps({"files": files}))
            ready(lead, manifest)
            marker = loop.folder / fetch_as_posted.marker_name(lead)
            if lead in first_markers:
                assert marker.read_bytes() == first_markers[lead]
            else:
                first_markers[lead] = marker.read_bytes()
        soil_posted[0] = True
        clock.sleep(30)
        return manifest

    # Even a final-lead proxy claiming the window whole must not bypass
    # the native per-lead gate. S3 f18 is available before its f17 soil.
    fetch._legacy_transfer(loop, lambda *_: None, window.leads,
                           force=False, whole=True,
                           incremental_transfer=incremental)
    assert transfers == [(15, 16), (15, 16, 17, 18)]
    assert asked and all("noaa-hrrr-bdp-pds.s3.amazonaws.com" in url
                         for url in asked)
    assert [marker["lead"] for marker in markers(out)] == [15, 16, 17, 18]
    for lead, data in first_markers.items():
        assert (loop.folder / fetch_as_posted.marker_name(lead)).read_bytes() == data


class HrrrIndexHeadRefused:
    """A native HRRR host whose objects answer HEAD and whose indexes do not.

    Each lead's wrfnat, wrfprs and wrfsfc, with their ``.idx``, are served from
    ``reveal(lead)`` on the replay clock.  A GRIB object answers HEAD and
    byte-range GETs; an index answers GET only, and 405 to HEAD, as a
    mirror or cache that refuses HEAD does.  The payloads are minimal GRIB2
    envelopes under complete certified inventories, so the transfer's own
    index, inventory, envelope and digest checks run unchanged.
    """

    RECORD = b"GRIB" + b"\x0a\x01" + b"\x00\x02" + (20).to_bytes(8, "big") + b"7777"
    VEGETATION_RECORD = (b"GRIB" + b"\x0a\x01" + b"\x02\x02"
                         + (20).to_bytes(8, "big") + b"7777")

    def __init__(self, cycle: datetime, clock: Clock, reveal) -> None:
        from tools import download_hrrr_native_subset as transport

        stamp = f"d={cycle:%Y%m%d%H}"

        def product(fields):
            index = "".join(
                f"{number}:{(number - 1) * len(self.RECORD)}:{stamp}:"
                f"{variable}:{level}:anl\n"
                for number, (variable, level) in enumerate(fields, 1))
            return self.RECORD * len(fields), index.encode("ascii")

        products = {
            "wrfnat": product(
                [(variable, f"{level} hybrid level")
                 for variable in transport.HYBRID_FIELDS
                 for level in range(1, 51)] + list(transport.SURFACE_FIELDS)),
            "wrfprs": product([(variable, level)
                               for variable in ("TSOIL", "SOILW")
                               for level in transport.SOIL_LEVELS]),
            "wrfsfc": (self.VEGETATION_RECORD,
                       f"1:0:{stamp}:VEG:surface:anl\n".encode("ascii")),
        }
        self.requests: list[tuple[str, str, str | None]] = []
        host = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def _empty(self, status: int) -> None:
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _object(self) -> bytes | None:
                name = self.path.rsplit("/", 1)[-1]
                product, _, lead = name.split(".")[2].rpartition("f")
                if clock.now() < reveal(int(lead)):
                    return None
                return products[product][int(name.endswith(".idx"))]

            def do_HEAD(self):
                host.requests.append(("HEAD", self.path, None))
                if self.path.endswith(".idx"):
                    self._empty(405)
                    return
                data = self._object()
                if data is None:
                    self._empty(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

            def do_GET(self):
                wanted = self.headers.get("Range")
                host.requests.append(("GET", self.path, wanted))
                data = self._object()
                if data is None:
                    self._empty(404)
                    return
                if wanted:
                    first, _, last = wanted.split("=", 1)[1].partition("-")
                    first = int(first)
                    last = int(last) if last else len(data) - 1
                    whole, data = len(data), data[first:last + 1]
                    self.send_response(206)
                    self.send_header("Content-Range",
                                     f"bytes {first}-{last}/{whole}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self.wfile.write(data)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_a_native_hrrr_host_that_refuses_index_head_still_fetches_as_posted(
        tmp_path, monkeypatch):
    """A136 L7c(b): an index HEAD the host refuses is not a missing index.

    Reproduced against the posted-prefix host choice: with every index
    HEAD answered 405 it made no GET and exited 75 after the posting
    budget, where the one-host final-lead choice before it downloaded and
    published both leads.  Each refused HEAD is now asked again as a GET
    of the index's first byte, and the leads download and publish their
    markers as they post.
    """

    from tools import download_hrrr_native_subset as transport
    from woof import runtime_surface_fetch, rustwx_fetch
    from woof.source_adapters import get_source_adapter

    clock = Clock(rows.expected_at("hrrr", CYCLE, 0) - timedelta(minutes=5))

    def reveal(lead: int) -> datetime:
        return rows.expected_at("hrrr", CYCLE, lead) + timedelta(seconds=20)

    host = HrrrIndexHeadRefused(CYCLE, clock, reveal)
    # 6a69b356f061, lane/hrrr-statics, appends one analyzed VEGFRA record
    # from wrfsfc to each native soil stream. These minimal envelopes test
    # posting/transfer, not native numeric decoding: route only that fetch
    # and extractor seam to this same replay instead of a live NOAA object.
    runtime_calls = []
    extraction_calls = []
    runtime_row, = get_source_adapter("hrrr").runtime_surface_fields
    vegetation_selector = (("discipline", 2), ("category", 0), ("parameter", 4),
                           ("level_type", 1), ("level_value", 0), ("pdt", 0))
    assert runtime_row == ("VEGFRA", "sfc", "VEG:surface", "percent",
                           vegetation_selector)

    def runtime_fetch(binary, **kwargs):
        assert binary == tmp_path / "runtime-fetch-fixture"
        assert kwargs["model"] == "hrrr"
        assert kwargs["date"] == f"{CYCLE:%Y%m%d}"
        assert kwargs["cycle"] == CYCLE.hour
        assert kwargs["product"] == runtime_row[1] == "sfc"
        assert kwargs["mode"] == "auto" and kwargs["keep_idx"]
        assert kwargs["pattern_file"].read_text().splitlines() == [runtime_row[2]]
        lead, = kwargs["hours"]
        prefix = {"aws": "s3", "nomads": "nomads"}[kwargs["source"]]
        name = f"hrrr.t{CYCLE:%H}z.wrfsfcf{lead:02d}.grib2"
        url = f"{host.base}/{prefix}/hrrr.{CYCLE:%Y%m%d}/conus/{name}"
        with urlopen(url) as response:
            data = response.read()
        assert data == host.VEGETATION_RECORD
        (kwargs["out"] / name).write_bytes(data)
        runtime_calls.append((lead, url))
        return {"files": [{"name": name, "grib_url": url,
                           "mode": "full-file",
                           "sha256": hashlib.sha256(data).hexdigest()}]}

    def extract_replay_record(source, output, numeric_selector):
        assert tuple(numeric_selector) == vegetation_selector
        assert source.read_bytes() == host.VEGETATION_RECORD
        extraction_calls.append(tuple(numeric_selector))
        output.write_bytes(source.read_bytes())

    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin",
                        lambda: tmp_path / "runtime-fetch-fixture")
    monkeypatch.setattr(rustwx_fetch, "probe_fetch_bin", lambda _: (True, "fixture"))
    monkeypatch.setattr(rustwx_fetch, "run_fetch", runtime_fetch)
    monkeypatch.setattr(runtime_surface_fetch, "extract_runtime_record",
                        extract_replay_record)
    monkeypatch.setattr(fetch, "HRRR_S3_BASE", host.base + "/s3")
    monkeypatch.setattr(fetch, "HRRR_NOMADS_BASE", host.base + "/nomads")
    monkeypatch.setattr(fetch_endpoints, "SETTLE_BACKOFF_S", ())
    monkeypatch.setattr(fetch_as_posted, "now_utc", clock.now)
    monkeypatch.setattr(fetch_as_posted, "pause", clock.sleep)
    parser = argparse.ArgumentParser()
    fetch.register_cli(parser.add_subparsers())
    out = tmp_path / "hrrr"
    args = parser.parse_args([
        "fetch", "--source", "hrrr", "--cycle", CYCLE.strftime("%Y-%m-%dT%H"),
        "--hours", "1", "--out", str(out), "--engine", "python",
        "--fetch-workers", "1", "--wait-timeout-minutes", "60"])
    try:
        assert fetch.fetch_main(args) == 0
    finally:
        host.close()
    assert not (out / "posting" / "failed.json").exists()
    written = markers(out)
    assert [marker["lead"] for marker in written] == [0, 1]
    for marker in written:
        assert (readiness.parse_instant(marker["first_seen_at"])
                >= reveal(marker["lead"]))
    manifest = json.loads((out / "fetch-manifest.json").read_text())
    assert manifest["forecast_hours"] == [0, 1]
    # 6a69b356f061, lane/hrrr-statics: 18 soil messages plus the separate
    # analyzed VEGFRA message. This is a message count, not a request count.
    assert [entry["records"] for entry in manifest["files"]
            if "records" in entry] == [transport.ATMOSPHERE_RECORD_COUNT,
                                       transport.SOIL_RECORD_COUNT + 1] * 2
    assert [lead for lead, _ in runtime_calls] == [0, 1]
    assert extraction_calls == [vegetation_selector] * 2
    assert [path for method, path, _ in host.requests
            if method == "GET" and ".wrfsfcf" in path
            and path.endswith(".grib2")] == [
        url.removeprefix(host.base) for _, url in runtime_calls]
    soils = [entry for entry in manifest["files"] if entry.get("role") == "soil"]
    assert len(soils) == 2
    for entry, (_, url) in zip(soils, runtime_calls):
        binding, = entry["runtime_surface"]
        assert binding["field"] == "VEGFRA" and binding["units"] == "percent"
        assert binding["url"] == url
        assert binding["numeric_selector"] == dict(vegetation_selector)
        assert binding["sha256"] == hashlib.sha256(host.VEGETATION_RECORD).hexdigest()
        assert (out / entry["name"]).read_bytes().endswith(host.VEGETATION_RECORD)
    refused = {path for method, path, _ in host.requests
               if method == "HEAD" and path.endswith(".idx")}
    confirmed = {path for method, path, wanted in host.requests
                 if method == "GET" and wanted == fetch_endpoints.CONFIRM_RANGE}
    assert refused and refused <= confirmed
    # One host moved every byte the transfer read: a fetch never mixes hosts.
    moved = {path.split("/")[1] for method, path, wanted in host.requests
             if method == "GET" and wanted != fetch_endpoints.CONFIRM_RANGE}
    assert len(moved) == 1


class _Answered:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _wire(head, get):
    """A fake network: HEAD gets ``head``, GET gets ``get`` (a status or "timeout")."""

    from urllib.error import HTTPError, URLError

    asked: list[tuple[str, str | None]] = []

    def opener(request, timeout=None):
        asked.append((request.get_method(), request.get_header("Range")))
        step = head if request.get_method() == "HEAD" else get
        if step == "timeout":
            raise URLError(TimeoutError(110, "timed out"))
        if not 200 <= step < 300:
            raise HTTPError(request.full_url, step, "status", {}, None)
        return _Answered(step)

    return opener, asked


@pytest.mark.parametrize("head, get, expected", [
    (405, 206, True), (501, 200, True), (403, 206, True), (502, 206, True),
    (405, 404, False), (405, 410, False), (405, 405, None),
    (405, "timeout", None)])
def test_a_refused_head_is_asked_again_as_a_one_byte_get(head, get, expected):
    opener, asked = _wire(head, get)
    assert fetch_endpoints.object_answer(
        "https://mirror.example/hrrr.t12z.wrfprsf01.grib2.idx", timeout=1,
        opener=opener) is expected
    assert asked == [("HEAD", None), ("GET", fetch_endpoints.CONFIRM_RANGE)]


@pytest.mark.parametrize("url, head, expected", [
    # The host said the object is missing.
    ("https://mirror.example/x", 404, False),
    ("https://mirror.example/x", 410, False),
    # The host asked this client to slow down: no second request at once.
    ("https://mirror.example/x", 429, None),
    ("https://mirror.example/x", 503, None),
    # NCEP's missing-path row already reads its 403.
    ("https://nomads.ncep.noaa.gov/pub/x", 403, None),
    # No answer at all: left to the transfer, whose own checks decide.
    ("https://mirror.example/x", "timeout", None)])
def test_an_answered_or_unheard_head_is_not_asked_again(url, head, expected):
    opener, asked = _wire(head, 200)
    assert fetch_endpoints.object_answer(
        url, timeout=1, opener=opener) is expected
    assert asked == [("HEAD", None)]


def test_the_posting_file_names_are_the_ones_the_go_relay_reads():
    """The writer's names live in a module the RW-WPS wheel stages; the
    relay's module is not in that wheel and carries its own copy.  A
    drift between the two would have go's stream watch a folder the fetch
    never writes."""

    from woof import chain_events

    for name in ("POSTING_DIRNAME", "POSTING_SCHEDULE_NAME",
                 "POSTING_FAILED_NAME"):
        assert getattr(rows, name) == getattr(chain_events, name), name


def test_fetch_table_rows_carry_as_posted_and_the_budget():
    assert fetch.FETCH_HINT_ROWS["as_posted"].default is True
    fetch.validate_fetch_hints({"source": "gefs", "as_posted": False},
                               source="t.toml")
    fetch.validate_fetch_hints({"source": "gefs", "late_after_minutes": 90},
                               source="t.toml")
    with pytest.raises(ValueError, match="positive number of minutes"):
        fetch.validate_fetch_hints({"source": "gefs", "late_after_minutes": 0},
                                   source="t.toml")
    with pytest.raises(ValueError, match="waits for nothing"):
        fetch.validate_fetch_hints({"source": "gefs", "as_posted": False,
                                    "late_after_minutes": 30}, source="t.toml")


def _watched_loop(tmp_path, reveal, *, hours: int = 6):
    """A gefs window's loop on the virtual clock, its probe answering by
    ``reveal(lead)`` and counting every ask."""

    import re

    plan = fetch_routes.resolve_request(SOURCE, cycle=CYCLE, hours=hours,
                                        cadence=3)
    window = readiness.Window(source=SOURCE, cycle=CYCLE,
                              leads=tuple(plan.leads), cadence=3, member=None,
                              transport=None, plan=plan, hours=hours)
    clock = Clock(rows.expected_at(SOURCE, CYCLE, 0) + timedelta(seconds=25))
    asked: list[int] = []

    def probe(url):
        lead = int(re.search(r"\.f(\d{3})", url.rsplit("/", 1)[-1]).group(1))
        asked.append(lead)
        at = reveal(lead)
        return at is not None and clock.now() >= at

    loop = fetch_as_posted.PostingLoop(window, tmp_path / SOURCE, probe=probe,
                                       now=clock.now, sleep=clock.sleep,
                                       progress=lambda *_: None)
    return loop, clock, asked


def test_a_lead_is_dated_when_it_posts_not_when_the_transfer_reaches_it(
        tmp_path, monkeypatch):
    """DESIGN A136 2.2 and 3.5: ``first_seen_at`` (and ``lead_posted``) is
    when one host first held the lead.  Measured live 2026-10-01: a source
    of about 125 objects per lead posted faster than the fetch moved them,
    and because a lead was asked only when the transfer reached it, each
    lead was said posted later and later after its scheduled time while
    the host had it on time.  The posting watch asks a due lead every
    ``poll_seconds`` whatever the transfer reaches; its rounds are driven
    by hand here on the virtual clock, and the transfer reaches f006 ten
    minutes after it posted."""

    monkeypatch.setattr(fetch_as_posted.PostingLoop, "watch", lambda self: None)
    loop, clock, asked = _watched_loop(tmp_path, lambda lead: on_schedule(lead, None))
    loop.start()
    loop.wait_start()
    posted = on_schedule(6, None)
    poll = loop.poll
    while clock.now() < posted + timedelta(seconds=3 * poll):
        loop._watch_round()
        clock.sleep(fetch_as_posted.WATCH_WAKE_SECONDS)
    clock.sleep(600)
    before = len(asked)
    assert loop(6)
    assert len(asked) == before, "the gate asked a lead the watch had seen"
    row = [row for row in loop.document["leads"] if row["lead"] == 6][0]
    seen = readiness.parse_instant(row["first_seen_at"])
    assert row["state"] == "posted"
    assert posted - timedelta(seconds=1) <= seen <= posted + timedelta(
        seconds=poll + fetch_as_posted.WATCH_WAKE_SECONDS)


def test_the_posting_watch_keeps_dating_leads_after_one_whose_host_was_not_heard(
        tmp_path, monkeypatch):
    """Measured live on aigfs 2026-10-01T00: its analysis lead was let
    through with its host not heard, the watch stopped at that lead, and
    every later lead was dated by its transfer again.  A lead the transfer
    is trying unheard is left to it; the leads after it are still watched."""

    monkeypatch.setattr(fetch_as_posted.PostingLoop, "watch", lambda self: None)
    loop, clock, asked = _watched_loop(tmp_path, lambda lead: on_schedule(lead, None))
    loop.start()
    loop.wait_start()
    with loop._lock:
        # The analysis lead as the start wait lets it through unheard.
        loop._seen.pop(0, None)
        loop._tried[0] = readiness.NOT_HEARD
    posted = on_schedule(6, None)
    while clock.now() < posted + timedelta(seconds=3 * loop.poll):
        loop._watch_round()
        clock.sleep(fetch_as_posted.WATCH_WAKE_SECONDS)
    row = [row for row in loop.document["leads"] if row["lead"] == 6][0]
    assert row["state"] == "posted"
    seen = readiness.parse_instant(row["first_seen_at"])
    assert posted - timedelta(seconds=1) <= seen <= posted + timedelta(
        seconds=loop.poll + fetch_as_posted.WATCH_WAKE_SECONDS)


def test_a_lead_already_up_at_its_first_ask_says_its_time_is_only_a_bound(
        tmp_path, monkeypatch):
    """DESIGN A136 3.5: ``lead_posted`` is the signal to re-fit a row.
    Measured live on aigfs 2026-10-01T00, fetched three hours after its
    leads posted: every lead was first asked, and dated, three hours late,
    so a re-fit from those times alone would have moved the row by the
    launch, not by the publisher.  A lead a host had already posted at its
    first ask says so (``posted_when_first_asked``), on its schedule row
    and its marker; a lead seen not posted and then posted does not."""

    monkeypatch.setattr(fetch_as_posted.PostingLoop, "watch", lambda self: None)
    # On schedule: the clock starts 5 s after f000 posted, before f003.
    loop, clock, _asked = _watched_loop(
        tmp_path / "live", lambda lead: on_schedule(lead, None))
    loop.start()
    loop.wait_start()
    posted = on_schedule(6, None)
    while clock.now() < posted + timedelta(seconds=3 * loop.poll):
        loop._watch_round()
        clock.sleep(fetch_as_posted.WATCH_WAKE_SECONDS)
    live = {row["lead"]: row for row in loop.document["leads"]}
    assert live[0]["posted_when_first_asked"] is True
    assert live[3]["posted_when_first_asked"] is False
    assert live[6]["posted_when_first_asked"] is False
    marker = json.loads(loop.publish(6, []).read_text(encoding="utf-8"))
    assert marker["posted_when_first_asked"] is False

    # Launched after the whole window posted: every lead was already up.
    late, clock, _asked = _watched_loop(tmp_path / "late", lambda lead: CYCLE)
    late.start()
    late.wait_start()
    while not late._watch_round():
        clock.sleep(fetch_as_posted.WATCH_WAKE_SECONDS)
    assert all(row["posted_when_first_asked"] is True
               for row in late.document["leads"])
    marker = json.loads(late.publish(3, []).read_text(encoding="utf-8"))
    assert marker["posted_when_first_asked"] is True

    # A window already out when the fetch began moves with no per-lead
    # ask (the fetch's whole-window path); its leads were up before any
    # ask too.  A lead let through with its host not heard says nothing.
    whole, clock, asked = _watched_loop(tmp_path / "whole", lambda lead: CYCLE)
    whole.start()
    with whole._lock:
        whole._tried[3] = readiness.NOT_HEARD
    first = json.loads(whole.publish(0, []).read_text(encoding="utf-8"))
    unheard = json.loads(whole.publish(3, []).read_text(encoding="utf-8"))
    assert asked == []
    assert first["posted_when_first_asked"] is True
    assert first["first_seen_at"] == first["fetched_at"]
    assert unheard["posted_when_first_asked"] is None
    rows_by_lead = {row["lead"]: row for row in whole.document["leads"]}
    assert rows_by_lead[0]["posted_when_first_asked"] is True
    assert "posted_when_first_asked" not in rows_by_lead[3]


def test_the_posting_watch_starts_after_the_start_needs_and_stops_on_close(
        tmp_path):
    loop, clock, asked = _watched_loop(
        tmp_path, lambda lead: None if lead == 6 else on_schedule(lead, None))
    loop.start()
    assert loop._watcher is None
    loop.wait_start()
    try:
        assert loop._watcher is not None and loop._watcher.is_alive()
    finally:
        loop.close()
    assert not loop._watcher.is_alive()
    # A watch stopped by close never starts again for that loop.
    loop.watch()
    assert not loop._watcher.is_alive()


def _donor_gated_fetch(tmp_path, monkeypatch, start_after_expected,
                       reveal_after=timedelta(minutes=3)):
    """aigfs 2026-09-29T00 whose own leads are all out; its GDAS donor posts
    ``reveal_after`` its scheduled time (None: never).  Returns (rc or the
    refusal, clock, the transfers asked for, the donor's posting time)."""

    import types

    source, cycle = "aigfs", datetime(2026, 9, 29, 0)
    donor = fetch_routes.route_for(source).donors[0]
    expected = rows.expected_at(donor.source, cycle, 0)
    clock = Clock(expected + start_after_expected)
    reveal = None if reveal_after is None else expected + reveal_after

    def out_now():
        return reveal is not None and clock.now() >= reveal

    def probe(url):
        if f"{donor.source}." in url.rsplit("/", 1)[-1]:
            return out_now()
        return True

    def published(src, cyc, lead, **_kw):
        if src == donor.source and not out_now():
            raise RuntimeError(f"{src.upper()} cycle {cyc:%Y-%m-%dT%H}Z is "
                               "not published through f000 yet")

    calls: list[dict] = []
    monkeypatch.setattr(fetch, "require_published_cycle", published)
    monkeypatch.setattr(fetch, "_fetch_route_donors", lambda plan, args: {})
    monkeypatch.setattr(fetch_routes, "run_plan",
                        lambda plan, **kw: calls.append(kw) or {})
    monkeypatch.setattr(fetch_routes, "write_handoff", lambda *a, **k: None)
    monkeypatch.setattr(fetch_routes, "handoff_lines", lambda *a, **k: [])
    monkeypatch.setattr(fetch_as_posted, "pause", clock.sleep)
    plan = fetch_routes.resolve_request(source, cycle=cycle, hours=6)
    window = readiness.Window(source=source, cycle=cycle, leads=tuple(plan.leads),
                              cadence=None, member=None, transport=None,
                              plan=plan, hours=6)
    out = tmp_path / source
    loop = fetch_as_posted.PostingLoop(window, out, probe=probe, now=clock.now,
                                       sleep=clock.sleep,
                                       progress=lambda *_: None)
    args = types.SimpleNamespace(force_refetch=False, out=out, fetch_workers=1)
    try:
        result = fetch._route_fetch_as_posted(plan, args, loop,
                                              last=max(plan.leads), options={})
    except RuntimeError as error:
        result = error
    return result, clock, calls, reveal


def test_a_donor_gated_window_waits_for_its_donor_instead_of_refusing(
        tmp_path, monkeypatch):
    """DESIGN A136 1b and 3.3: a donor-gated source starts when its donor
    posts.  Measured live 2026-10-01: aigfs 00Z, every AI lead posted, its
    readiness answer 75 (waiting on the GDAS f000 donor, due 06:48Z), and
    the fetch launched at 06:41Z refused with exit 2 instead of waiting, so
    a site launching by the readiness rule skipped the cycle."""

    result, clock, calls, reveal = _donor_gated_fetch(
        tmp_path, monkeypatch, timedelta(minutes=-2))
    assert result == 0, result
    assert clock.now() >= reveal, "the fetch did not wait for its donor"
    assert len(calls) == 1 and calls[0]["lead_gate"] is None


def test_a_donor_past_its_late_time_is_still_refused_in_words(
        tmp_path, monkeypatch):
    """Past its late time a donor that is not out will not appear by
    waiting: the whole-cycle refusal, naming the donor, stands."""

    donor = fetch_routes.route_for("aigfs").donors[0]
    cycle = datetime(2026, 9, 29, 0)
    # The donor need's own budget (its source's row), as the loop reads it.
    past = (rows.late_at(donor.source, cycle, 0)
            - rows.expected_at(donor.source, cycle, 0) + timedelta(minutes=1))
    result, clock, calls, reveal = _donor_gated_fetch(
        tmp_path, monkeypatch, past, reveal_after=None)
    assert isinstance(result, RuntimeError), result
    assert "takes part of its start from the GDAS analysis" in str(result)
    assert not calls


def test_a_start_wait_marks_the_analysis_and_first_boundary_rows_waiting(
        replay, tmp_path):
    """While the loop waits for its start needs, the rows of the needs on
    the window's own source and lead say ``waiting`` (DESIGN A136 3.5), as
    a lead's own wait does; a page drawing schedule.json otherwise showed
    the two leads the run was blocked on as merely scheduled."""

    server, clock = replay(on_schedule)
    out = tmp_path / "gefs"
    seen: dict[int, set] = {0: set(), 3: set()}
    real_write = fetch_as_posted.PostingLoop.write_schedule

    def write_schedule(self):
        for row in self.document["leads"]:
            if row["lead"] in seen:
                seen[row["lead"]].add(row["state"])
        return real_write(self)

    fetch_as_posted.PostingLoop.write_schedule = write_schedule
    try:
        assert fetch.fetch_main(fetch_args(out)) == 0
    finally:
        fetch_as_posted.PostingLoop.write_schedule = real_write
    assert "waiting" in seen[0] and "waiting" in seen[3]


def test_the_hrrr_operational_rung_says_the_as_posted_wait():
    """The rung's reason is printed by every default HRRR fetch that takes
    the operational server, so it names the as-posted wait, not a flag the
    user did not pass."""

    rung = [endpoint for endpoint in fetch_endpoints.ladder("hrrr")
            if endpoint.name == "nomads"][0]
    assert "--wait-for" not in rung.why
    assert "as-posted" in rung.why


def _schedule_snapshots(monkeypatch) -> list[dict]:
    """Every schedule the loop publishes, as a reader of the file sees it."""

    seen: list[dict] = []
    real_write = fetch_as_posted.PostingLoop.write_schedule

    def write_schedule(self):
        seen.append(json.loads(json.dumps(self.document)))
        return real_write(self)

    monkeypatch.setattr(fetch_as_posted.PostingLoop, "write_schedule",
                        write_schedule)
    return seen


def test_each_lead_row_records_the_host_s_last_answer(replay, tmp_path,
                                                      monkeypatch):
    """A lead asked before it posts says ``not_posted`` on its row, then
    ``posted``: the schedule is the record a wait on the lead reads (A136
    L10 item 3)."""

    server, clock = replay(on_schedule)
    out = tmp_path / "gefs"
    seen = _schedule_snapshots(monkeypatch)
    assert fetch.fetch_main(fetch_args(out)) == 0
    answers = {lead: [] for lead in (0, 3, 6, 9, 12)}
    for schedule in seen:
        for row in schedule["leads"]:
            said = row["last_answer"]
            if said is not None and (not answers[row["lead"]]
                                     or answers[row["lead"]][-1] != said):
                answers[row["lead"]].append(said)
    assert answers[6] == ["not_posted", "posted"]
    final = json.loads((out / "posting" / "schedule.json").read_text())
    assert {row["last_answer"] for row in final["leads"]} == {"posted"}


def test_a_start_need_whose_host_cannot_be_heard_says_so(replay, tmp_path,
                                                         monkeypatch):
    """The run's start wait on a lead whose host does not answer says the
    host cannot be heard, never "not posted yet" or "not fetched yet":
    something the engine could not see is not the publisher being late
    (DESIGN A136 3.6).  The row carried no answer, so both were said."""

    from woof.chain_events import start_wait_row
    from woof.ingest.boundary_stream import source_wait_reason

    server, clock = replay(on_schedule)
    for path in list(server.objects):
        if path.endswith(".f000") or ".f000." in path:
            server.unheard_until[path] = on_schedule(0, clock) + timedelta(
                minutes=40)
    out = tmp_path / "gefs"
    seen = _schedule_snapshots(monkeypatch)
    assert fetch.fetch_main(fetch_args(out)) == 0
    waits = [start_wait_row(schedule) for schedule in seen]
    unheard = [row for row in waits
               if row is not None and row["lead"] == 0
               and row["last_answer"] == "not_heard"]
    assert unheard, "the start wait on f000 never said its host was not heard"
    reason = source_wait_reason({**unheard[-1], "source": SOURCE})
    assert reason.startswith("gefs f000 is not fetched: its host cannot be "
                             "heard")
    assert "not posted yet" not in reason
    assert "not fetched yet" not in reason
    assert [marker["lead"] for marker in markers(out)] == [0, 3, 6, 9, 12]


class _FileFailed(RuntimeError):
    """One file's own refusal: the failure the request ends with."""


@pytest.mark.parametrize("stopped_by", ("another_file", "the_chain"))
def test_a_lead_waiting_to_post_under_a_chain_stop_ends_when_either_stop_fires(
        tmp_path, stopped_by):
    """The staged chain registers its stop (``stop_event``) for every
    as-posted fetch it runs, and the gate's posting wait runs inside the
    transfer pool's jobs.  That wait blocked on the chain's stop alone, so
    when another file failed the request after its retries, a worker
    waiting for a later lead slept its wait out, and the pool, which waits
    for its jobs to wind down before it raises, raised only then: no
    ``failed.json``, the preparation still waiting on the failed lead, and
    the other workers still downloading.  Reproduced on a development machine: the
    request raised after the 20 s posting wait instead of at 0.5 s.  The
    wait now ends on either stop: the request's (``TransferCancelled``,
    e4224502f) and the chain's (``FetchStopped``)."""

    assert fetch_as_posted.pause is time.sleep, "the real wait is under test"
    out = tmp_path / SOURCE
    chain_stop = fetch_as_posted.stop_event(out)
    try:
        plan = fetch_routes.resolve_request(SOURCE, cycle=CYCLE, hours=6,
                                            cadence=3)
        window = readiness.Window(source=SOURCE, cycle=CYCLE,
                                  leads=tuple(plan.leads), cadence=3,
                                  member=None, transport=None, plan=plan,
                                  hours=6)
        # An hour before f006 is due, on a clock that moves with real
        # time; the whole window's cap bounds the wait to 20 s, so a wait
        # that does not end on the stop fails here instead of hanging.
        begun = time.monotonic()
        base = rows.expected_at(SOURCE, CYCLE, 6) - timedelta(hours=1)
        asked: list[str] = []
        loop = fetch_as_posted.PostingLoop(
            window, out, wait_timeout_minutes=20 / 60,
            probe=lambda url: asked.append(url) or False,
            now=lambda: base + timedelta(seconds=time.monotonic() - begun),
            progress=lambda *_: None)
        loop.start()
        waits: list[float] = []
        waiting = threading.Event()
        real_pause = loop._pause

        def pause(seconds):
            waits.append(seconds)
            waiting.set()
            real_pause(seconds)

        loop._pause = pause
        ended: dict[str, BaseException] = {}

        def later_lead():
            try:
                return {"name": "f006", "bytes": 1, "endpoint": loop(6)}
            except BaseException as error:
                ended["f006"] = error
                raise

        def other_file():
            assert waiting.wait(10), "f006 never waited for its posting"
            if stopped_by == "the_chain":
                chain_stop.set()
                return {"name": "f000", "bytes": 1}
            raise _FileFailed("f000: the transfer failed after its retries")

        expected = (_FileFailed if stopped_by == "another_file"
                    else fetch_as_posted.FetchStopped)
        started = time.monotonic()
        with pytest.raises(expected):
            fetch_pool.run_transfers(
                [fetch_pool.TransferJob(name="f006", url=None,
                                        action=later_lead),
                 fetch_pool.TransferJob(name="f000", url=None,
                                        action=other_file)],
                workers=2)
        elapsed = time.monotonic() - started
    finally:
        fetch_as_posted.release_stop(out)
    assert waits and waits[0] >= 15.0, "the posting wait under test is short"
    assert elapsed < 5.0, "the posting wait was served"
    assert asked == [], "a lead not due was asked"
    if stopped_by == "another_file":
        assert isinstance(ended["f006"], fetch_pool.TransferCancelled)
        assert not chain_stop.is_set()
    else:
        assert isinstance(ended["f006"], fetch_as_posted.FetchStopped)
    # Ended by a stop, not by its budget: nothing says the lead was late.
    assert not (out / "posting" / "failed.json").exists()
