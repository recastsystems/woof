"""New forecast never waits: the source rows arrive on their own, and forecasts queue for the card.

Two defects this file holds shut:

- The When step sat on "Checking which data holds this start..." for
  minutes, and on a second visit drew no calendar at all.  Every source
  was put to the fetch's object probe inside the request, one HEAD after
  another with a 60 s timeout, NOMADS HEADs queued behind a running
  forecast's downloads in the node's governor, and ICON alone asked about
  300 objects in series (205 s for a recent start and 138 s for a 2024
  one, measured on a node running a forecast).  ``/api/sources`` asked the same probe before
  the page could draw anything, and so did every fit.
- A second forecast could not be started while one ran: the card lock
  refused it, and there was nowhere to leave it.
- A 1996 date, and a fit for a 2021 event, still asked every host about
  today's runs (91 s, 22 s), and the start check and the newest-start
  check each asked the same GEM cycle's 174 objects.  The fetch's own
  check before a run downloads asked those objects one at a time.

No server is asked here: every HEAD is a stand-in.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
import sys
import threading
import time

import pytest

from woof.gui import queue as queue_module
from woof.gui import runs
from woof.gui.server import build_server, serve_in_thread
from woof.source_availability import (ProbeSession, availability, opening_start, page_latest,
                                       page_start_check, publication_refusal)
from test_gui_server import FakeRunner, request

LIVE = datetime(2026, 9, 26, 6, 4)


# ---------------------------------------------------------------- the probe a page asks

def test_a_session_asks_each_object_once_and_stops_at_its_deadline():
    asked = []

    def slow(url):
        asked.append(url)
        time.sleep(0.3)
        return True

    session = ProbeSession(deadline_s=0.5, head=slow)
    started = time.monotonic()
    session.prefetch([f"https://example.invalid/{i}" for i in range(8)])
    # Eight objects asked at once take one object's time, not eight.
    assert time.monotonic() - started < 1.5
    assert session("https://example.invalid/0") is True and len(asked) == 8
    time.sleep(0.4)
    # Past the deadline nothing more is sent, and the silence is counted as silence.
    assert session.answer("https://example.invalid/late") is None
    assert session.misses == 1 and len(asked) == 8


def test_nomads_objects_are_never_asked_in_bulk():
    asked = []
    session = ProbeSession(head=lambda url: asked.append(url) or True)
    session.prefetch(["https://nomads.ncep.noaa.gov/a", "https://noaa-gfs-bdp-pds.s3.amazonaws.com/b"])
    assert asked == ["https://noaa-gfs-bdp-pds.s3.amazonaws.com/b"]


@pytest.mark.parametrize("published", [
    lambda url: True,                        # every rung holds it
    lambda url: ".t06z." not in url,         # nothing of the 06Z run yet
    lambda url: "nomads" not in url,         # only the archive holds it
    lambda url: "nomads" in url,             # only the operational server holds it
])
def test_a_pages_start_check_answers_what_the_fetch_answers(published):
    document = availability("gfs", 6, now=LIVE)
    fetch_says = publication_refusal(document, "2026-09-26T06", now=LIVE, probe=published)
    page_says = page_start_check(document, "2026-09-26T06", now=LIVE, session=ProbeSession(head=published))
    assert page_says["state"] == ("published" if fetch_says is None else "not-published")


def test_a_host_that_is_not_heard_names_no_newest_start_and_says_so():
    # The schedule declares no publication delay for GFS (the probe decides), so its estimate is the cycle still
    # being made: returned as the newest start, a page opened on it and Start accepted it.
    document = availability("gfs", 6, now=LIVE)
    silent = ProbeSession(head=lambda url: None)
    assert page_start_check(document, "2026-09-26T00", now=LIVE, session=silent)["state"] == "unchecked"
    newest = page_latest(document, now=LIVE, session=ProbeSession(head=lambda url: None))
    assert newest == {"latest": None, "basis": "unchecked", "downloads": False, "missing": []}
    # Every host answered and none holds a whole start: still nothing is named, and each start asked is missing.
    nothing = page_latest(document, now=LIVE, session=ProbeSession(head=lambda url: False))
    assert nothing["latest"] is None and nothing["basis"] == "checked" and nothing["downloads"] is False
    assert nothing["missing"][:2] == ["2026-09-26T06", "2026-09-26T00"]


def test_a_3_hour_window_on_6_hourly_files_is_checked_as_the_download_the_run_makes():
    # A 3-hour window on a source whose files come every 6 hours: integrate/2.8 asked the fetch's lead resolver
    # about the 3-hour window, which it refuses, inside the request, and the whole source list answered 500.  The
    # run downloads hours 0 and 6, so that is what the check asks about.
    from woof.gui.availability import Board

    asked = []

    def present(url):
        asked.append(url)
        return True

    document = availability("aigfs", 3, now=LIVE)
    assert document["last_hour"] == 6
    assert page_start_check(document, "2026-09-26T00", now=LIVE, session=ProbeSession(head=present)) == {
        "state": "published"}
    assert asked and all(".f006." in url for url in asked), asked
    board = Board(session_factory=lambda: ProbeSession(head=present))
    try:
        board.row("aigfs", "AIGFS", "2026-09-26T00", 3, now=LIVE)
        assert board.wait(10)
        row = board.row("aigfs", "AIGFS", "2026-09-26T00", 3, now=LIVE)
    finally:
        board.close()
    assert row["state"] == "yes" and row["starts"] == "now" and row["basis"] == "checked" and not row["checking"]


def test_a_start_the_fetchs_own_resolver_refuses_is_that_rows_no_in_its_words(monkeypatch):
    # Any other refusal the fetch's URL builder raises for one start is that row's no, never an error for the list
    # and never "may have it": the run's own download would refuse the same start.
    from woof.gui.availability import Board

    def refuse(*args, **kwargs):
        raise ValueError("--source gfs: the 00Z cycle does not publish f006")

    monkeypatch.setattr("woof.source_availability._rung_urls", refuse)
    board = Board(session_factory=lambda: ProbeSession(head=lambda url: True))
    try:
        board.row("gfs", "GFS", "2026-09-26T00", 6, now=LIVE)
        assert board.wait(10)
        row = board.row("gfs", "GFS", "2026-09-26T00", 6, now=LIVE)
    finally:
        board.close()
    assert row["state"] == "no" and row["starts"] == "no" and not row["checking"] and row["basis"] == "checked"
    assert row["why"] == "The download refuses this start and length: --source gfs: the 00Z cycle does not publish f006"


def test_a_newest_start_found_past_an_unheard_host_is_not_called_checked():
    # The 06Z objects go unheard; 00Z is answered present.  00Z certainly downloads, but 06Z may too, so it is not
    # said to be missing.
    document = availability("gfs", 6, now=LIVE)
    unheard_06z = ProbeSession(head=lambda url: None if ".t06z." in url else True)
    assert page_latest(document, now=LIVE, session=unheard_06z) == {
        "latest": "2026-09-26T00", "basis": "unchecked", "downloads": True, "missing": []}
    every_answer = ProbeSession(head=lambda url: ".t06z." not in url)
    assert page_latest(document, now=LIVE, session=every_answer) == {
        "latest": "2026-09-26T00", "basis": "checked", "downloads": True, "missing": ["2026-09-26T06"]}


def test_a_late_nomads_turn_does_not_pull_the_newest_start_back():
    # With another forecast downloading from NOMADS, the newer starts' NOMADS HEADs got no turn in the page's
    # budget and a later one did, so the resolver's walk of the operational server stopped a day back, although the
    # archive had already answered the newer start complete.
    document = availability("gfs", 6, now=LIVE)

    def late_turn(archive_06z):
        def head(url):
            if "nomads" in url:
                return None if ".t06z." in url else True
            return archive_06z if ".t06z." in url else True
        return ProbeSession(head=head)

    assert page_latest(document, now=LIVE, session=late_turn(True)) == {
        "latest": "2026-09-26T06", "basis": "checked", "downloads": True, "missing": []}
    # The archive lags the operational server: its "not yet" leaves 06Z open, so it is not said to be missing.
    assert page_latest(document, now=LIVE, session=late_turn(False)) == {
        "latest": "2026-09-26T00", "basis": "unchecked", "downloads": True, "missing": []}


# ---------------------------------------------------------------- the one rule

def test_each_source_whose_probe_decides_publication_declares_its_usual_delay():
    """With no publication delay declared (the probe decides there), a page whose check had not answered opened on
    the start still being made."""

    for source, hours in (("gfs", 6.0), ("gdas", 8.0), ("hrrr", 3.0)):
        document = availability(source, 6, now=LIVE)
        assert document["usual_delay_hours"] == hours
        assert document["due_start"] <= document["latest_candidate"]
    # LIVE less each delay, on each source's hours.
    assert availability("gfs", 6, now=LIVE)["due_start"] == "2026-09-26T00"
    assert availability("hrrr", 6, now=LIVE)["due_start"] == "2026-09-26T03"
    assert availability("gdas", 6, now=LIVE)["due_start"] == "2026-09-25T18"
    # A source whose schedule already starts at its measured lag keeps it: the due start is its schedule's own.
    icon = availability("icon-eu", 6, now=LIVE)
    assert icon["due_start"] == icon["latest_candidate"]


def test_the_page_opens_on_the_newest_start_start_takes():
    document = availability("gfs", 6, now=LIVE)
    assert opening_start(document, None) == "2026-09-26T00"
    # Confirmed whole and newer than the due start: opened on.
    assert opening_start(document, "2026-09-26T06") == "2026-09-26T06"
    # Confirmed whole but older: the due start is opened on, since newer starts may be published.
    assert opening_start(document, "2026-09-25T12") == "2026-09-26T00"
    # A due start a check found missing is not taken: the newest one before it that no check found missing is.
    assert opening_start(document, "2026-09-25T18", ["2026-09-26T00"]) == "2026-09-25T18"
    assert opening_start(document, "2026-09-25T12", ["2026-09-26T00"]) == "2026-09-25T18"
    assert opening_start(document, None, ["2026-09-26T00", "2026-09-25T18"]) == "2026-09-25T12"


# ---------------------------------------------------------------- a busy NOMADS and an archive mirror that is silent

#: Half an hour after a GFS start: that run publishes over the next five hours, and the schedule, which declares
#: no publication delay for GFS (the probe decides there), calls it the newest start.
MIDNIGHT = datetime(2026, 9, 27, 0, 29)
#: The newest GFS start past its usual publication delay at MIDNIGHT, and the start still being made.
DUE, FRONTIER = "2026-09-26T18", "2026-09-27T00"
#: A piece of ``answers`` that lets NOMADS HEADs matching the rest of it have their turn.
NOMADS_TURN = "nomads:"


class _Present:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _unheard_server(tmp_path, monkeypatch, clock, runner=None):
    """A page server on a node whose NOMADS turns a running forecast holds, with the archive mirror silent.

    Every NOMADS HEAD goes through the real governor under an hour-long cooldown, so each is refused its turn
    within the page's budget and nothing is sent; every other host times out.  A test makes the mirror answer by
    putting a piece of a URL in ``answers`` (True present, False a 404), and lets one NOMADS HEAD have its turn
    with a piece that starts ``nomads:``.
    """

    from urllib.error import HTTPError, URLError

    from woof import nomads_governor

    state = tmp_path / "nomads.state"
    monkeypatch.setenv(nomads_governor.STATE_PATH_ENV, str(state))
    stamp = nomads_governor._now_ms()
    nomads_governor.write_state(state, stamp, stamp + 3_600_000)
    real = nomads_governor.paced_urlopen
    answers: dict[str, bool] = {}

    def paced(request, *, timeout=None, max_wait_s=None, **_):
        url = request.full_url
        if nomads_governor.is_nomads_url(url):
            found = next((value for piece, value in answers.items()
                          if piece.startswith(NOMADS_TURN) and piece[len(NOMADS_TURN):] in url), None)
            if found is None:
                return real(request, timeout=timeout, max_wait_s=max_wait_s,
                            opener=lambda *a, **k: pytest.fail(f"NOMADS was sent {url} past its turn"))
        else:
            found = next((value for piece, value in answers.items()
                          if not piece.startswith(NOMADS_TURN) and piece in url), None)
        if found is None:
            raise URLError("timed out")
        if not found:
            raise HTTPError(url, 404, "Not Found", {}, None)
        return _Present()

    monkeypatch.setattr(nomads_governor, "paced_urlopen", paced)
    monkeypatch.setattr("woof.source_availability.published_stop", lambda window, **_: None)
    monkeypatch.setattr("woof.fetch._head_ok", lambda url: pytest.fail(f"a request probed {url} itself"))
    if clock is not None:
        monkeypatch.setattr("gpuwm.gui.api._utcnow", lambda: clock)
    runner = runner or FakeRunner()
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    serve_in_thread(server)
    return server, runner, answers


@pytest.fixture()
def unheard(tmp_path, monkeypatch):
    server, runner, answers = _unheard_server(tmp_path, monkeypatch, MIDNIGHT)
    yield server, runner, answers
    server.shutdown()
    server.server_close()


def _gfs_row(server, start):
    request(server, "GET", f"/api/sources/availability?time={start}&hours=6")
    assert server.api.board.wait(30)
    _, check = request(server, "GET", f"/api/sources/availability?time={start}&hours=6")
    return {row["id"]: row for row in check["sources"]}["gfs"], check


def _age_out(board):
    """Every answer the board holds is old, so the next read of each asks again.

    The per-object answers every check shares (``board.heard``) age with them: time that makes a row's answer old
    makes the answers it was built from old too.
    """

    with board._lock:
        for key, (stamp, answer) in list(board._done.items()):
            board._done[key] = (stamp - 10_000, answer)
    with board.heard._lock:
        for url, (stamp, found) in list(board.heard._answers.items()):
            board.heard._answers[url] = (stamp - 10_000, found)


def test_a_start_no_check_confirmed_is_never_offered_as_newest_and_only_queue_it_takes_it(unheard):
    """With a forecast downloading from NOMADS and the GFS mirror not answering, New forecast opened by itself on
    the start still being made, offered it as Newest run, its row said "has it" and Start accepted it; its download
    then refused it."""

    server, runner, _ = unheard
    assert availability("gfs", 6, now=MIDNIGHT)["latest_candidate"] == FRONTIER

    # Until a check confirms a start, the page opens on the newest start past GFS's usual publication delay and
    # names no newest run: while the check is out, and once it could hear no host.
    _, body = request(server, "GET", "/api/sources")
    assert body["default_cycle"] == DUE and body["default_newest_run"] is None and body["default_checking"]
    assert server.api.board.wait(30)
    _, body = request(server, "GET", "/api/sources")
    assert body["default_cycle"] == DUE and body["default_newest_run"] is None

    # The start still being made is neither confirmed nor past the delay: only Queue it takes it.
    row, check = _gfs_row(server, FRONTIER)
    assert row["state"] == "unknown" and row["starts"] == "queue" and row["basis"] == "unchecked"
    assert "did not answer in time" in row["why"] and row["confirmed"] is None and row["newest_run"] is None
    assert row["opening"] == DUE and check["best"] is None
    draft = {**DRAFT, "cycle": FRONTIER, "hours": 6}
    for extra in ({"name": "shown", "dry_run": True}, {"name": "now"}):
        response, answer = request(server, "POST", "/api/create/start", body={**draft, **extra})
        assert response.status == 422, answer
        assert "not confirmed" in answer["message"] and answer["fix"].startswith("Queue it instead")
        assert "2026-09-26 18:00 UTC" in answer["fix"]
    assert runner.launched == [] and not (server.root / "now").exists()
    # A fit is never held or refused on it.
    response, _ = request(server, "POST", "/api/create/fit", body=draft)
    assert response.status == 200

    # The due start: nothing confirmed it either, but it is past the usual delay, so Start takes it and says why.
    row, _ = _gfs_row(server, DUE)
    assert row["state"] == "yes" and row["starts"] == "now" and row["basis"] == "unchecked"
    assert "usually takes to publish a whole run (about 6 hours)" in row["note"]
    response, answer = request(server, "POST", "/api/create/start",
                               body={**DRAFT, "cycle": DUE, "hours": 6, "name": "due"})
    assert response.status == 200, answer
    assert len(runner.launched) == 1


@pytest.mark.parametrize("older", ["2026-09-26T06", "2026-09-26T00"])
def test_an_older_start_found_while_newer_starts_go_unheard_is_never_opened_on_or_called_newest(unheard, older):
    """With NOMADS busy and the GFS mirror not answering, the newer starts' NOMADS HEADs got no turn and an older
    start's did: New forecast pulled its draft back onto that older start and offered it as Newest run, beside a
    Yesterday pick newer than it, while newer starts were published."""

    server, runner, answers = unheard
    day, hour = older[:10].replace("-", ""), older[11:]
    answers[f"{NOMADS_TURN}gfs.{day}/{hour}/"] = True
    request(server, "GET", "/api/sources")
    assert server.api.board.wait(30)
    _, body = request(server, "GET", "/api/sources")
    # The opening start stays the due one, and no newest run is named.
    assert body["default_cycle"] == DUE and body["default_newest_run"] is None

    row, _ = _gfs_row(server, DUE)
    assert row["confirmed"] == older and row["opening"] == DUE and row["newest_run"] is None
    assert row["state"] == "yes" and row["starts"] == "now"
    # Start takes the older start itself: it was found whole, and its row says so without inferring.
    mine, _ = _gfs_row(server, older)
    assert mine["state"] == "yes" and mine["starts"] == "now" and "inferred_from" not in mine
    for extra in ({"name": "shown", "dry_run": True}, {"name": "confirmed"}):
        response, answer = request(server, "POST", "/api/create/start",
                                   body={**DRAFT, "cycle": older, "hours": 6, **extra})
        assert response.status == 200, answer
    assert len(runner.launched) == 1

    # The mirror confirms the due start whole while the start after it still goes unheard: opened on, not the
    # newest run.
    answers["gfs.20260926/18/"] = True
    _age_out(server.api.board)
    request(server, "GET", "/api/sources")
    assert server.api.board.wait(30)
    _, body = request(server, "GET", "/api/sources")
    assert body["default_cycle"] == DUE and body["default_newest_run"] is None
    # Once every server answers that the start after it is not whole yet, it is the newest run.
    answers.update({"gfs.20260927/00/": False, f"{NOMADS_TURN}gfs.20260927/00/": False})
    _age_out(server.api.board)
    request(server, "GET", "/api/sources")
    assert server.api.board.wait(30)
    _, body = request(server, "GET", "/api/sources")
    assert body["default_cycle"] == DUE and body["default_newest_run"] == DUE


def test_a_start_found_whole_on_the_mirror_is_opened_on_and_the_starts_before_it_are_published(unheard):
    server, runner, answers = unheard
    # The mirror holds 18Z whole and not yet 00Z; the 12Z objects go unheard; NOMADS still gets no turn.
    answers.update({"gfs.20260927/00/": False, "gfs.20260926/18/": True})
    request(server, "GET", "/api/sources")
    assert server.api.board.wait(30)
    _, body = request(server, "GET", "/api/sources")
    # The mirror's "not yet" for 00Z leaves it open, since the mirror publishes after NOMADS: 18Z is opened on and is
    # not called the newest run.
    assert body["default_cycle"] == DUE and body["default_newest_run"] is None

    newest, _ = _gfs_row(server, FRONTIER)
    assert newest["starts"] == "queue" and newest["basis"] == "unchecked" and newest["confirmed"] == DUE
    assert newest["newest_run"] is None and newest["newest_basis"] == "unchecked"
    # 12Z went unheard itself, but 18Z was found whole and a source publishes its starts in order.
    earlier, _ = _gfs_row(server, "2026-09-26T12")
    assert earlier["state"] == "yes" and earlier["basis"] == "checked" and earlier["starts"] == "now"
    assert earlier["inferred_from"] == DUE and "was found whole" in earlier["note"]
    response, answer = request(server, "POST", "/api/create/start",
                               body={**DRAFT, "cycle": "2026-09-26T12", "hours": 6, "name": "earlier"})
    assert response.status == 200, answer
    assert len(runner.launched) == 1

    # NOMADS answers 00Z not whole as well: 18Z is the newest run.
    answers[f"{NOMADS_TURN}gfs.20260927/00/"] = False
    _age_out(server.api.board)
    newest, _ = _gfs_row(server, FRONTIER)
    assert newest["confirmed"] == DUE and newest["newest_run"] == DUE and newest["newest_basis"] == "checked"

    # A later check that hears no host at all never moves the confirmed start back, or drops it; whether 00Z has
    # been published since is not known, so nothing is the newest run.
    answers.clear()
    _age_out(server.api.board)
    newest, _ = _gfs_row(server, FRONTIER)
    assert newest["confirmed"] == DUE and newest["opening"] == DUE and newest["starts"] == "queue"
    assert newest["newest_run"] is None


def test_a_late_start_every_server_answers_missing_is_not_opened_on_or_started_but_can_be_queued(unheard):
    """A start past its usual publication delay that its servers still answer missing (a late run): the page opened
    on it and Start took it, and its download refused it."""

    server, runner, answers = unheard
    whole = "2026-09-26T12"
    answers.update({"gfs.20260926/18/": False, f"{NOMADS_TURN}gfs.20260926/18/": False, "gfs.20260926/12/": True})
    request(server, "GET", "/api/sources")
    assert server.api.board.wait(30)
    row, _ = _gfs_row(server, DUE)
    assert row["state"] == "unknown" and row["starts"] == "queue" and row["basis"] == "checked"
    assert row["why"].startswith("Not published through hour 6 yet") and DUE in row["missing"]
    # 00Z went unheard: 12Z is opened on, and is the newest run only once 00Z is answered not whole too.
    assert row["confirmed"] == whole and row["opening"] == whole and row["newest_run"] is None
    answers.update({"gfs.20260927/00/": False, f"{NOMADS_TURN}gfs.20260927/00/": False})
    _age_out(server.api.board)
    row, _ = _gfs_row(server, DUE)
    assert row["confirmed"] == whole and row["opening"] == whole and row["newest_run"] == whole
    _, body = request(server, "GET", "/api/sources")
    assert body["default_cycle"] == whole and body["default_newest_run"] == whole
    response, answer = request(server, "POST", "/api/create/start",
                               body={**DRAFT, "cycle": DUE, "hours": 6, "name": "late"})
    assert response.status == 422 and "2026-09-26 12:00 UTC" in answer["fix"], answer
    response, answer = request(server, "POST", "/api/create/start",
                               body={**DRAFT, "cycle": DUE, "hours": 6, "name": "late", "queue": True})
    assert response.status == 200 and "has not published" in answer["waits_for_data"], answer
    assert runner.launched == []


def test_a_start_every_server_answers_missing_before_a_whole_later_start_is_refused_not_queued(unheard):
    """A start its servers answer missing while a later start is whole was skipped by the publisher: the row said
    "not published yet" and Queue it held a run for data that would never come."""

    server, runner, answers = unheard
    skipped = "2026-09-26T12"
    answers.update({"gfs.20260926/18/": True, "gfs.20260926/12/": False, f"{NOMADS_TURN}gfs.20260926/12/": False})
    request(server, "GET", "/api/sources")
    assert server.api.board.wait(30)
    row, _ = _gfs_row(server, skipped)
    assert row["confirmed"] == DUE
    assert row["state"] == "no" and row["starts"] == "no" and row["basis"] == "checked", row
    assert "waiting will not bring it" in row["why"] and "2026-09-26 18:00 UTC" in row["fix"]
    response, answer = request(server, "POST", "/api/create/start",
                               body={**DRAFT, "cycle": skipped, "hours": 6, "name": "skipped", "queue": True})
    assert response.status == 422 and "waiting will not bring it" in answer["message"], answer
    assert runner.launched == []
    # The start it is refused in favour of still starts.
    row, _ = _gfs_row(server, DUE)
    assert row["state"] == "yes" and row["starts"] == "now"


def test_the_start_found_whole_reads_published_without_inferring_from_itself(unheard):
    """The row of the newest start found whole said the data server did not answer about it, but that it was found
    whole, so it was published too: both about the same start."""

    server, _, answers = unheard
    answers[f"{NOMADS_TURN}gfs.20260926/12/"] = True
    newest, _ = _gfs_row(server, "2026-09-26T12")
    assert newest["confirmed"] == "2026-09-26T12" and newest["basis"] == "checked"
    # Nothing is heard now: its own check goes unheard, and the start found whole before is kept.
    answers.clear()
    _age_out(server.api.board)
    same, _ = _gfs_row(server, "2026-09-26T12")
    assert same["state"] == "yes" and same["basis"] == "checked" and same["starts"] == "now"
    assert same["confirmed"] == "2026-09-26T12"
    assert "inferred_from" not in same and "note" not in same
    earlier, _ = _gfs_row(server, "2026-09-26T06")
    assert earlier["state"] == "yes" and earlier["inferred_from"] == "2026-09-26T12"
    assert "2026-09-26 12:00 UTC was found whole" in earlier["note"]


#: Four and a half hours after a GFS start: that run's files for a 6-hour forecast are usually on the servers by now,
#: but the start is inside GFS's usual publication delay, so the newest start past the delay is still the one before.
DAWN = datetime(2026, 9, 27, 4, 30)


def test_a_start_found_whole_while_the_start_after_it_goes_unheard_is_not_the_newest_run(tmp_path, monkeypatch):
    """With NOMADS busy and the GFS mirror not answering, only the newest due start's NOMADS HEADs got a turn: New
    forecast called that start the Newest run, while the start after it, whose files for a 6-hour forecast were on the
    servers, read "may have it"."""

    assert availability("gfs", 6, now=DAWN)["due_start"] == DUE
    server, runner, answers = _unheard_server(tmp_path, monkeypatch, DAWN)
    try:
        answers[f"{NOMADS_TURN}gfs.20260926/18/"] = True
        request(server, "GET", "/api/sources")
        assert server.api.board.wait(30)
        _, body = request(server, "GET", "/api/sources")
        # Opened on the start found whole, which is the due start too; it is not called the newest run.
        assert body["default_cycle"] == DUE and body["default_newest_run"] is None and not body["default_checking"]

        newer, _ = _gfs_row(server, FRONTIER)
        assert newer["state"] == "unknown" and newer["starts"] == "queue" and newer["basis"] == "unchecked"
        assert newer["confirmed"] == DUE and newer["opening"] == DUE
        assert newer["newest_run"] is None and newer["newest_basis"] == "unchecked"
        found, _ = _gfs_row(server, DUE)
        assert found["state"] == "yes" and found["starts"] == "now" and found["basis"] == "checked"
        assert found["newest_run"] is None and "inferred_from" not in found and "note" not in found
        # Start takes the start found whole; the start after it is Queue it's alone.
        response, answer = request(server, "POST", "/api/create/start",
                                   body={**DRAFT, "cycle": DUE, "hours": 6, "name": "found"})
        assert response.status == 200, answer
        response, answer = request(server, "POST", "/api/create/start",
                                   body={**DRAFT, "cycle": FRONTIER, "hours": 6, "name": "newer"})
        assert response.status == 422 and answer["fix"].startswith("Queue it instead"), answer
        assert len(runner.launched) == 1

        # Every server answers the start after it not whole yet: now it is the newest run.
        answers.update({"gfs.20260927/00/": False, f"{NOMADS_TURN}gfs.20260927/00/": False})
        _age_out(server.api.board)
        newer, _ = _gfs_row(server, FRONTIER)
        assert newer["newest_run"] == DUE and newer["newest_basis"] == "checked"
        _, body = request(server, "GET", "/api/sources")
        assert body["default_cycle"] == DUE and body["default_newest_run"] == DUE

        # The start after it is published: it is opened on and is the newest run, being the schedule's newest start.
        answers[f"{NOMADS_TURN}gfs.20260927/00/"] = True
        _age_out(server.api.board)
        newer, _ = _gfs_row(server, FRONTIER)
        assert newer["state"] == "yes" and newer["starts"] == "now"
        assert newer["confirmed"] == newer["opening"] == newer["newest_run"] == FRONTIER
        _, body = request(server, "GET", "/api/sources")
        assert body["default_cycle"] == FRONTIER and body["default_newest_run"] == FRONTIER
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("found", ["2026-09-27T03", "2026-09-27T05"])
def test_an_hourly_start_found_whole_while_later_hours_go_unheard_is_not_the_newest_run(found):
    """HRRR publishes a start every hour and usually takes 3 hours, so two starts after the newest due start can be
    published: the start found whole was called the newest run while a later one went unheard."""

    from woof.gui.availability import Board

    moment = datetime(2026, 9, 27, 6, 30)
    heard: dict[str, bool | None] = {}

    def head(url):
        hour = next((stamp for stamp in ("t03z", "t04z", "t05z", "t06z") if f".{stamp}." in url), None)
        return heard.get(hour, True if hour is None else None)

    board = Board(session_factory=lambda: ProbeSession(head=head))
    try:
        heard.update({"t03z": True, f"t{found[11:]}z": True})
        board.newest("hrrr", 6, now=moment)
        assert board.wait(30)
        newest = board.newest("hrrr", 6, now=moment)
        assert newest["due"] == "2026-09-27T03" and newest["confirmed"] == found and newest["opening"] == found
        assert newest["newest_run"] is None and newest["newest_basis"] == "unchecked"
        # Each later hour answered not whole: the start found whole is the newest run.
        heard.update({f"t{hour:02d}z": False for hour in range(int(found[11:]) + 1, 7)})
        _age_out(board)
        board.newest("hrrr", 6, now=moment)
        assert board.wait(30)
        newest = board.newest("hrrr", 6, now=moment)
        assert newest["confirmed"] == found and newest["newest_run"] == found and newest["newest_basis"] == "checked"
    finally:
        board.close()


def test_queue_it_holds_an_unconfirmed_start_until_a_check_confirms_it_then_starts_it_in_turn(tmp_path, monkeypatch):
    """A forecast queued on a start its servers had not published yet was started by the queue and refused at its
    download."""

    server, runner, answers = _unheard_server(tmp_path, monkeypatch, MIDNIGHT, runner=Gated())
    try:
        queue = server.api.queue
        queue._cards = lambda: {"devices": [{"name": "Test card", "memory_total_mib": 16 * 1024,
                                             "memory_used_mib": 512}], "processes": []}
        queue._disk = lambda: 200.0
        runner.holder = held_by("job-x")
        _gfs_row(server, FRONTIER)
        early = queue_one(server, "early", cycle=FRONTIER, hours=6)
        assert "not confirmed" in early["waits_for_data"]
        later = queue_one(server, "later", cycle=DUE, hours=6)
        assert later["waits_for_data"] is None
        # Said at once, before the scheduler looks.
        _, listing = request(server, "GET", "/api/queue")
        assert [item["run"] for item in listing["items"]] == ["early", "later"]
        held = listing["items"][0]["held"]
        assert held.startswith("Held until a check confirms the GFS 2026-09-27 00:00 UTC start is published")
        # Waiting for its start to be published ends by itself, so the page shows it as a wait, not a fault.
        assert listing["items"][0]["awaits_data"] is True and listing["items"][1]["awaits_data"] is False
        # The card frees: the one behind it goes, since Start takes its start; the held one keeps its place.
        runner.holder = None
        assert queue.tick() == ["later"]
        assert queue.order() == ["early"]
        finish(server, "later")
        # A page server restart keeps it in line, still held.
        again = build_server(server.root, port=0, runner=Gated(), token="u" * 43)
        try:
            assert again.api.queue.order() == ["early"]
            marker = json.loads((server.root / "early" / runs.QUEUED).read_text(encoding="utf-8"))
            assert marker["held"] == held and marker["data"]["cycle"] == FRONTIER
        finally:
            again.server_close()
        # The mirror now holds the start whole: once a check confirms it, the scheduler starts it.
        answers["gfs.20260927/00/"] = True
        _age_out(server.api.board)
        assert queue.tick() == []           # this look asks again; the answer is not in yet
        assert server.api.board.wait(30)
        assert queue.tick() == ["early"]
        assert queue.order() == [] and len(runner.launched) == 2
    finally:
        server.shutdown()
        server.server_close()


def test_the_newest_start_is_found_asking_one_starts_objects_when_that_start_is_whole():
    # The three newest starts' objects were asked at once: about 400 HEADs for one GEM row where 174 answer it.
    from woof import fetch

    document = availability("gem-gdps", 6, now=LIVE)
    asked = []
    newest = page_latest(document, now=LIVE, session=ProbeSession(head=lambda url: asked.append(url) or True))
    assert newest == {"latest": document["latest_candidate"], "basis": "checked", "downloads": True,
                      "missing": []}
    whole = fetch.cycle_probe_urls("gem-gdps", datetime.strptime(newest["latest"], "%Y-%m-%dT%H"), 6)
    assert sorted(asked) == sorted(whole)


def test_an_old_start_asks_no_server():
    document = availability("gfs", 6, now=LIVE)
    never = ProbeSession(head=lambda url: pytest.fail(f"an old start probed {url}"))
    assert page_start_check(document, "2024-05-06T18", now=LIVE, session=never)["state"] == "skipped"


def test_the_governor_budget_sends_nothing_rather_than_queue(tmp_path, monkeypatch):
    from woof import fetch_endpoints, nomads_governor

    state = tmp_path / "nomads.state"
    monkeypatch.setenv(nomads_governor.STATE_PATH_ENV, str(state))
    now = nomads_governor._now_ms()
    nomads_governor.write_state(state, now, now + 60_000)   # a cooldown a minute long
    before = state.read_text(encoding="utf-8")
    opened = []
    answer = fetch_endpoints.object_answer("https://nomads.ncep.noaa.gov/x", timeout=1, max_wait_s=0.2,
                                           opener=lambda *a, **k: opened.append(a))
    assert answer is None and opened == []
    assert state.read_text(encoding="utf-8") == before
    with pytest.raises(nomads_governor.PaceBudgetExceeded):
        nomads_governor.pace("https://nomads.ncep.noaa.gov/x", max_wait_s=0.2)


def test_a_404_is_an_answer_and_a_timeout_is_not():
    from urllib.error import HTTPError, URLError

    from woof import fetch_endpoints

    def missing(request, timeout=None):
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    def slow(request, timeout=None):
        raise URLError("timed out")

    assert fetch_endpoints.object_answer("https://s3.example/x", timeout=1, opener=missing) is False
    assert fetch_endpoints.object_answer("https://s3.example/x", timeout=1, opener=slow) is None


@pytest.mark.parametrize("sent_answer", [None, True])
def test_a_check_that_waited_on_a_head_is_handed_only_a_fresh_answer(monkeypatch, sent_answer):
    # Seen on test host 4 with GEM: the host did not answer the newest start's HEAD, and the newest-start check, which
    # had waited on the start check's HEAD of the same object, was handed the "missing" kept from 200 s before.
    # The row then named a start 12 h older as the checked newest, and kept it for 10 minutes.
    from woof.gui.availability import FRESH_NO_S, Heard

    now = [0.0]
    heard = Heard(clock=lambda: now[0])
    url = "https://dd.weather.gc.ca/model_gem_global/x.grib2"
    monkeypatch.setattr("woof.source_availability.quick_head", lambda asked: False)
    assert heard.head(url) is False
    now[0] = FRESH_NO_S + 110.0                 # the missing answer is past its time

    sent, release, waiting = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def host(asked):
        calls.append(asked)
        sent.set()
        release.wait(10)
        return sent_answer

    monkeypatch.setattr("woof.source_availability.quick_head", host)
    answers = {}
    sender = threading.Thread(target=lambda: answers.__setitem__("sender", heard.head(url)))
    sender.start()
    assert sent.wait(5)

    class Watched:
        """The in-flight HEAD's event, saying when the second check has begun to wait on it."""

        def __init__(self, inner):
            self.inner = inner

        def wait(self, timeout=None):
            waiting.set()
            return self.inner.wait(timeout)

    with heard._lock:
        heard._flight[url] = Watched(heard._flight[url])
    waiter = threading.Thread(target=lambda: answers.__setitem__("waiter", heard.head(url)))
    waiter.start()
    assert waiting.wait(5)
    release.set()
    sender.join(10)
    waiter.join(10)
    assert answers == {"sender": sent_answer, "waiter": sent_answer}
    assert len(calls) == 1, "the check that waited sent its own HEAD"


# ---------------------------------------------------------------- the endpoints answer at once

def held_by(job_id: str) -> dict:
    """What :meth:`Runner.card_holder` answers while ``job_id`` holds the card (a job writing no run folder here)."""

    return {"job_id": job_id, "started_utc": "2026-09-27T09:30:00+00:00", "kind": "gui:run-plan", "folder": None}


class Gated(FakeRunner):
    """A runner whose engine calls can be held, like an engine starting while a forecast loads the machine."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.gate.set()
        self.holder = None

    def query(self, argv, **kwargs):
        self.gate.wait(30)
        return super().query(argv, **kwargs)

    def card_holder(self):
        return self.holder


@pytest.fixture()
def slow_hosts(tmp_path, monkeypatch):
    """A page server whose data hosts each take two seconds per HEAD."""

    def slow(url):
        time.sleep(2.0)
        return True

    monkeypatch.setattr("woof.source_availability.quick_head", slow)
    monkeypatch.setattr("woof.source_availability.published_stop", lambda window, **_: None)
    monkeypatch.setattr("woof.fetch._head_ok", lambda url: pytest.fail(f"a request probed {url} itself"))
    monkeypatch.setattr("gpuwm.gui.api._utcnow", lambda: LIVE)
    runner = Gated()
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    serve_in_thread(server)
    yield server, runner
    server.shutdown()
    server.server_close()


def timed(server, method, path, body=None):
    started = time.monotonic()
    response, answer = request(server, method, path, body=body, timeout=30)
    return time.monotonic() - started, response, answer


def test_the_source_rows_answer_at_once_and_fill_in_as_the_hosts_answer(slow_hosts):
    server, _ = slow_hosts
    took, response, body = timed(server, "GET", "/api/sources/availability?time=2026-09-26T00&hours=6")
    assert response.status == 200 and took < 1.5
    row = {r["id"]: r for r in body["sources"]}["gfs"]
    assert row["checking"] and row["state"] == "yes" and body["pending"] >= 1
    assert server.api.board.wait(20)
    took, _, body = timed(server, "GET", "/api/sources/availability?time=2026-09-26T00&hours=6")
    row = {r["id"]: r for r in body["sources"]}["gfs"]
    assert took < 1.5 and not row["checking"] and row["basis"] == "checked" and body["pending"] == 0
    assert row["checked_age_s"] is not None


def test_the_source_list_and_a_fit_never_wait_for_a_probe(slow_hosts):
    server, _ = slow_hosts
    took, response, body = timed(server, "GET", "/api/sources")
    assert response.status == 200 and took < 1.5 and body["default_checking"]
    draft = {"source": "gfs", "cycle": "2026-09-26T00", "lat": 35.5, "lon": -97.0, "width_km": 600,
             "height_km": 600, "hours": 6, "card": "8gb", "dx_km": 12}
    took, response, body = timed(server, "POST", "/api/create/fit", draft)
    assert response.status == 200, body
    assert took < 1.5


def test_an_engine_answer_past_its_time_is_served_while_the_next_is_fetched(slow_hosts):
    server, runner = slow_hosts
    request(server, "GET", "/api/system")
    request(server, "GET", "/api/sources")
    with server.api._cache_lock:
        for key, (stamp, document) in list(server.api._cache.items()):
            server.api._cache[key] = (stamp - 10_000, document)
    runner.gate.clear()      # the engine is slow to start now
    try:
        for path in ("/api/system", "/api/sources"):
            took, response, _ = timed(server, "GET", path)
            assert response.status == 200 and took < 1.5, path
    finally:
        runner.gate.set()


def test_a_restarted_page_server_answers_the_source_list_from_the_engines_last_answer(slow_hosts):
    """The first page request after a page server started waited 2.7 to 4.4 s on a loaded machine for the engine's
    first source list, although the same engine had answered it before the restart."""

    server, runner = slow_hosts
    _, before = request(server, "GET", "/api/sources")
    again = build_server(server.root, port=0, runner=Gated(), token="u" * 43)
    serve_in_thread(again)
    try:
        again.api.runner.gate.clear()        # the engine is slow to start after the restart
        took, response, body = timed(again, "GET", "/api/sources")
        assert response.status == 200 and took < 1.5, body
        assert [row["id"] for row in body["sources"]] == [row["id"] for row in before["sources"]]
        # It is asked again behind the answer, and its new answer is the one kept.
        again.api.runner.gate.set()
        deadline = time.monotonic() + 10
        while not any("--sources" in argv for argv in again.api.runner.queries) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert any("--sources" in argv for argv in again.api.runner.queries)
    finally:
        again.api.runner.gate.set()
        again.shutdown()
        again.server_close()
    # An answer another engine gave is not served: a changed command asks this engine and waits for it.
    kept = server.root / ".arwen-gui" / "engine" / "sources.json"
    saved = json.loads(kept.read_text(encoding="utf-8"))
    kept.write_text(json.dumps({**saved, "argv": [*saved["argv"], "--other"]}), encoding="utf-8")
    fresh = build_server(server.root, port=0, runner=Gated(), token="v" * 43)
    try:
        fresh.api.runner.gate.clear()
        asker = threading.Thread(target=lambda: fresh.api.sources())
        asker.start()
        asker.join(0.5)
        assert asker.is_alive(), "an answer kept for another command was served"
        fresh.api.runner.gate.set()
        asker.join(10)
    finally:
        fresh.api.runner.gate.set()
        fresh.server_close()


def test_the_machines_list_never_waits_on_ssh(tmp_path):
    from woof.gui.machines import Registry

    class Slow(Registry):
        def rows(self):
            return [{"name": "box", "host": "me@box"}]

        def probe(self, name, *, fresh=False):
            time.sleep(2.0)
            return {"name": name, "state": "idle"}

    registry = Slow(tmp_path / "machines.toml")
    started = time.monotonic()
    first = registry.listing()
    assert time.monotonic() - started < 0.5 and first[0]["state"] == "checking"
    time.sleep(2.5)
    assert registry.listing()[0]["state"] == "idle"


# ---------------------------------------------------------------- no page waits on a host, and no host is asked for nothing

class Hosts:
    """Every HEAD a probe sends, each taking one second, counted where the fetch's probe and a page's meet."""

    def __init__(self, seconds: float = 1.0) -> None:
        self.seconds = seconds
        self.lock = threading.Lock()
        self.sent: list[str] = []
        self.answered = 0
        self.reply_gate: threading.Event | None = None

    def _one(self, url: str) -> None:
        with self.lock:
            self.sent.append(url)
        if self.reply_gate is None:
            time.sleep(self.seconds)
        else:
            self.reply_gate.wait()
        with self.lock:
            self.answered += 1

    def paced_urlopen(self, request, **_):
        # A page's HEAD (fetch_endpoints.object_answer) and the fetch's own (object_available) both open here.
        self._one(request if isinstance(request, str) else request.full_url)
        return _Present()

    def head_ok(self, url: str) -> bool:
        # The fetch's probe, stood in for by name too, so a check already running keeps this stand-in.
        self._one(url)
        return True

    def heads(self) -> list[str]:
        with self.lock:
            return list(self.sent)


class Offers(FakeRunner):
    """The engine offering several sources, as New forecast lists them."""

    def __init__(self, sources) -> None:
        super().__init__()
        self.sources = tuple(sources)

    def query(self, argv, **kwargs):
        if "--sources" in argv:
            self.queries.append(list(argv))
            return {"sources": [{"source_id": source, "display_name": source.upper(),
                                 "max_forecast_hour": 0 if source == "era5" else 48,
                                 "run_plan": {"intent_supported": True, "intent_routes": ["prepared"],
                                              "requires_source_root": False}} for source in self.sources]}
        if "--physics-profiles" in argv:
            self.queries.append(list(argv))
            return {"sources": [{"source_id": source, "default_profile_id": "p1",
                                 "profiles": [{"profile_id": "p1", "admissible": True, "is_default": True}]}
                                for source in self.sources],
                    "profiles": [{"profile_id": "p1", "summary": "p1: words"}]}
        return super().query(argv, **kwargs)


def settle(server, timeout: float = 60.0) -> None:
    """Every background check the page server started has finished."""

    board = getattr(server.api, "board", None)
    assert board is None or board.wait(timeout)


@pytest.fixture()
def counted_hosts(tmp_path, monkeypatch):
    hosts = Hosts()
    monkeypatch.setattr("woof.nomads_governor.paced_urlopen", hosts.paced_urlopen)
    monkeypatch.setattr("woof.fetch._head_ok", hosts.head_ok)
    monkeypatch.setattr("woof.source_availability.published_stop", lambda window, **_: None)
    servers = []

    def serve(*sources):
        server = build_server(tmp_path / f"runs-{len(servers)}", port=0, runner=Offers(sources), token="t" * 43)
        serve_in_thread(server)
        servers.append(server)
        return server

    yield hosts, serve
    for server in servers:
        settle(server)
        server.shutdown()
        server.server_close()


def answered_in(server, hosts, method, path, body=None):
    try:
        return timed(server, method, path, body)
    except TimeoutError:
        pytest.fail(f"{method} {path} had no answer in 30 s; {len(hosts.heads())} HEADs were sent meanwhile")


def test_a_1996_start_is_answered_at_once_and_no_host_is_asked(counted_hosts):
    # Measured: 91 to 94 s per ask of the date picker, every source asked about today's runs for a 1996 date.
    hosts, serve = counted_hosts
    server = serve("gfs", "hrrr", "gem-gdps", "icon-eu", "era5")
    path = "/api/sources/availability?time=1996-05-28T00&hours=6"
    took, response, body = answered_in(server, hosts, "GET", path)
    assert response.status == 200 and took < 2.0, took
    settle(server)
    assert hosts.heads() == []
    rows = {row["id"]: row for row in body["sources"]}
    assert rows["era5"]["state"] == "yes"
    assert all(rows[source]["state"] == "no" for source in ("gfs", "hrrr", "gem-gdps", "icon-eu"))
    took, _, again = answered_in(server, hosts, "GET", path)
    assert took < 2.0 and again.get("pending", 0) == 0
    settle(server)
    assert hosts.heads() == []


def test_a_recent_start_is_answered_before_any_host_and_each_object_is_asked_once(counted_hosts, monkeypatch):
    # Measured: 127 s and 348 HEADs for GEM alone, its newest cycle's final lead asked twice, and the same again
    # on every fit and on Start.
    from woof.source_availability import availability

    hosts, serve = counted_hosts
    # A pending start is taken from the schedule only after its settled publication time.
    # The live clock can fall between the earliest and settled posting times.
    monkeypatch.setattr("gpuwm.gui.api._utcnow", lambda: LIVE)
    hosts.reply_gate = threading.Event()
    server = serve("gem-gdps", "era5")
    start = availability("gem-gdps", 6, now=LIVE)["latest_candidate"]
    path = f"/api/sources/availability?time={start}&hours=6"
    try:
        took, response, body = answered_in(server, hosts, "GET", path)
        assert response.status == 200 and hosts.answered == 0, (took, hosts.answered)
        row = {row["id"]: row for row in body["sources"]}["gem-gdps"]
        assert row["state"] == "yes" and row["checking"]
        draft = {"source": "gem-gdps", "cycle": start, "lat": 50.0, "lon": -100.0, "width_km": 600,
                 "height_km": 600, "hours": 6, "card": "8gb", "dx_km": 12}
        # The fit never waits for a check.  Start waits, at most SETTLE_S, for the one check of the start it writes, the
        # check the list already started (Board.settle): it sends nothing of its own, as the count below shows.
        from woof.gui.availability import SETTLE_S

        for step, bound in (("/api/create/fit", 1.0), ("/api/create/start", SETTLE_S + 2.0)):
            if step == "/api/create/start":
                hosts.reply_gate.set()
            took, response, answer = answered_in(server, hosts, "POST", step,
                                                {**draft, "name": "gem-now", "dry_run": True})
            assert response.status == 200 and took < bound, (step, took, answer)
            if step == "/api/create/fit":
                assert hosts.answered == 0
    finally:
        hosts.reply_gate.set()
    settle(server)
    asked = hosts.heads()
    assert asked, "the start was never put to the probe"
    assert len(asked) == len(set(asked)), "an object was asked twice"
    took, response, again = answered_in(server, hosts, "GET", path)
    assert response.status == 200 and took < 1.0
    settle(server)
    assert hosts.heads() == asked, "the same request asked the hosts again"


def test_a_fit_and_an_event_start_for_2021_ask_no_host_about_todays_runs(counted_hosts):
    # Measured: 22 s of HEADs against today's HRRR runs on NOMADS before the 2021 EF4 event's run page opened.
    hosts, serve = counted_hosts
    server = serve("hrrr")
    draft = {"source": "hrrr", "cycle": "2021-12-10T18", "lat": 36.5, "lon": -89.0, "width_km": 600,
             "height_km": 600, "hours": 6, "card": "8gb", "dx_km": 3}
    took, response, body = answered_in(server, hosts, "POST", "/api/create/fit", draft)
    assert response.status == 200, body
    settle(server)
    assert hosts.heads() == []
    # The event button asks the page server the same question of its start before launching.
    assert server.api.availability_of("hrrr", "2021-12-10T18", 6)["state"] == "yes"
    settle(server)
    assert hosts.heads() == []


def test_a_closed_page_server_sends_no_more_heads(counted_hosts):
    # Measured: 126 HEADs after server_close, sent through whichever HEAD was in place by then: a later test's
    # counting stand-in (an object "asked twice") or the real hosts.
    from woof.gui.availability import head_bound_s
    from woof.source_availability import availability

    hosts, serve = counted_hosts
    server = serve("gem-gdps", "icon-eu")
    for source in ("gem-gdps", "icon-eu"):
        start = availability(source, 6)["latest_candidate"]
        response, _ = request(server, "GET", f"/api/sources/availability?time={start}&hours=6", timeout=30)
        assert response.status == 200
    deadline = time.monotonic() + 30
    while not hosts.heads() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert hosts.heads(), "no check was started"
    server.shutdown()
    started = time.monotonic()
    server.server_close()
    assert time.monotonic() - started < head_bound_s() + 2.0
    sent = hosts.heads()
    assert hosts.answered == len(sent), "a HEAD was still out when the server said it was closed"
    time.sleep(2.5)                             # two and a half HEADs' time
    assert hosts.heads() == sent, f"{len(hosts.heads()) - len(sent)} HEADs went out after server_close"


def test_the_fetchs_own_publication_check_asks_one_hosts_objects_side_by_side():
    # Asked one after another, GEM's final lead (about 174 objects) held a run's start for minutes.
    from woof import fetch, fetch_pool
    from woof.source_availability import availability

    start = availability("gem-gdps", 6, now=LIVE)["latest_candidate"]
    cycle = datetime.strptime(start, "%Y-%m-%dT%H")
    urls = fetch.cycle_probe_urls("gem-gdps", cycle, 6)
    assert len(urls) > 100
    lock = threading.Lock()
    asked, flying, peak = [], [0], [0]

    def present(url):
        with lock:
            asked.append(url)
            flying[0] += 1
            peak[0] = max(peak[0], flying[0])
        time.sleep(0.02)
        with lock:
            flying[0] -= 1
        return True

    started = time.monotonic()
    assert fetch.cycle_publication_refusal("gem-gdps", cycle, 6, now=LIVE, probe=present) is None
    took = time.monotonic() - started
    assert sorted(asked) == sorted(set(urls))
    assert peak[0] == fetch_pool.DEFAULT_FILE_WORKERS
    assert took < 0.02 * len(urls) / 2, took


def test_the_fetchs_check_stops_at_a_missing_object_and_keeps_a_hosts_cap():
    from woof import fetch, fetch_pool

    urls = [f"https://dd.example.invalid/object-{i}" for i in range(40)]
    asked = []

    def first_missing(url):
        asked.append(url)
        if url == urls[0]:
            return False
        time.sleep(0.05)
        return True

    assert fetch.objects_published(urls, first_missing) is False
    assert len(asked) <= fetch_pool.DEFAULT_FILE_WORKERS
    assert fetch.objects_published([], first_missing) is False

    lock = threading.Lock()
    flying, peak = [0], [0]

    def nomads(url):
        with lock:
            flying[0] += 1
            peak[0] = max(peak[0], flying[0])
        time.sleep(0.05)
        with lock:
            flying[0] -= 1
        return True

    assert fetch.objects_published([f"https://nomads.ncep.noaa.gov/x{i}" for i in range(6)], nomads)
    assert peak[0] == fetch_pool.HOST_FILE_WORKER_CAPS["nomads.ncep.noaa.gov"]


# ---------------------------------------------------------------- the queue

DRAFT = {"source": "gfs", "cycle": "2026-09-24T00", "lat": 35.5, "lon": -97.0, "width_km": 300,
         "height_km": 300, "hours": 1, "card": "16gb", "dx_km": 12}


@pytest.fixture()
def queued(tmp_path, monkeypatch):
    monkeypatch.setattr("woof.source_availability.quick_head", lambda url: True)
    monkeypatch.setattr("woof.fetch._head_ok", lambda url: True)
    runner = Gated()
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    card = {"devices": [{"name": "Test card", "memory_total_mib": 16 * 1024, "memory_used_mib": 512}],
            "processes": []}
    disk = {"free": 200.0}
    server.api.queue._cards = lambda: card
    server.api.queue._disk = lambda: disk["free"]
    serve_in_thread(server)
    yield server, runner, card, disk
    server.shutdown()
    server.server_close()


def queue_one(server, name, **extra):
    response, body = request(server, "POST", "/api/create/start",
                             body={**DRAFT, "name": name, "queue": True, "need_gib": 5.3, **extra})
    assert response.status == 200, body
    assert body["queued"] and body["run"] == name
    return body


def finish(server, name):
    """The forecast the fake runner started ends: its manifest names a dead pid and it records completion."""

    run = server.root / name
    (run / "run-manifest.json").write_text(json.dumps({"pid": 999_999_9}), encoding="utf-8")
    (run / "gui-job.json").write_text(json.dumps({"wrapper_pid": 999_999_9}), encoding="utf-8")
    (run / "events.jsonl").write_text(json.dumps({"sequence": 0, "event": "completed"}) + "\n", encoding="utf-8")


def test_a_busy_card_queues_forecasts_in_order_and_starts_them_one_at_a_time(queued):
    server, runner, _, _ = queued
    runner.holder = held_by("job-x")
    response, body = request(server, "GET", "/api/queue?need_gib=5.3")
    assert body["expect"]["busy"] and not body["expect"]["start_now"]
    queue_one(server, "first")
    queue_one(server, "second")
    _, listing = request(server, "GET", "/api/queue")
    assert [item["run"] for item in listing["items"]] == ["first", "second"]
    rows = {row["id"]: row for row in request(server, "GET", "/api/runs")[1]["runs"]}
    assert rows["first"]["status"]["state"] == "queued"
    assert (server.root / "first" / "plan.json").is_file()

    # Busy: nothing starts, and each says why it waits.
    assert server.api.queue.tick() == []
    assert runner.launched == []
    marker = json.loads((server.root / "first" / runs.QUEUED).read_text(encoding="utf-8"))
    assert "running another forecast" in marker["waiting"]

    # Free: the first starts, the second waits for it.
    runner.holder = None
    assert server.api.queue.tick() == ["first"]
    assert len(runner.launched) == 1 and "first" in runner.launched[0][-1]
    runner.holder = held_by("job-first")
    assert server.api.queue.tick() == []
    finish(server, "first")
    runner.holder = None
    assert server.api.queue.tick() == ["second"]
    assert server.api.queue.order() == []


def test_the_queue_survives_a_page_server_restart(queued, tmp_path):
    server, runner, _, _ = queued
    runner.holder = held_by("job-busy")
    queue_one(server, "a")
    queue_one(server, "b")
    queue_one(server, "c")
    request(server, "POST", "/api/queue/c/up")
    request(server, "POST", "/api/queue/a/down")
    again = build_server(server.root, port=0, runner=FakeRunner(), token="u" * 43)
    try:
        assert again.api.queue.order() == ["c", "a", "b"]
    finally:
        again.server_close()
    response, body = request(server, "POST", "/api/queue/a/remove")
    assert response.status == 200 and not (server.root / "a").exists()
    assert server.api.queue.order() == ["c", "b"]
    response, _ = request(server, "POST", "/api/queue/a/remove")
    assert response.status == 404


def test_moving_a_forecast_says_at_once_why_each_one_waits(queued):
    """Right after Move up the two rows' reasons were swapped until the next scheduler look, up to 5 s later: the
    one now first said it waited for the forecasts ahead of it, and the one behind named the running forecast."""

    server, runner, _, _ = queued
    runner.holder = held_by("job-x")
    queue_one(server, "one")
    queue_one(server, "two")
    queue_one(server, "big", need_gib=40.0)
    assert server.api.queue.tick() == []
    ahead = "Waiting for the forecasts ahead of it."

    def reasons():
        _, listing = request(server, "GET", "/api/queue")
        return [(item["run"], item["waiting"], bool(item["held"])) for item in listing["items"]]

    first = reasons()
    assert [run for run, _, _ in first] == ["one", "two", "big"]
    card = first[0][1]
    assert "running another forecast" in card and first[1][1] == ahead and first[2][2]
    server.api.queue._wake.clear()
    response, _ = request(server, "POST", "/api/queue/two/up")
    assert response.status == 200
    assert reasons()[:2] == [("two", card, False), ("one", ahead, False)]
    # The scheduler is asked to look again at once, and its look agrees.
    assert server.api.queue._wake.is_set()
    assert server.api.queue.tick() == []
    assert reasons()[:2] == [("two", card, False), ("one", ahead, False)]
    # A held forecast is passed over by the scheduler, so moving one past it changes no one's words.
    request(server, "POST", "/api/queue/big/up")
    after = reasons()
    assert [run for run, _, _ in after] == ["two", "big", "one"]
    assert after[0][1] == card and after[2][1] == ahead and after[1][2]


def test_a_forecast_just_queued_says_why_it_waits_before_the_scheduler_looks(queued):
    """A forecast queued behind one held for its data said nothing of why it waited until the scheduler's next look
    had read the card, which took most of a second on a loaded machine."""

    server, runner, _, _ = queued
    runner.holder = held_by("job-x")
    server.api.queue.tick = lambda: []      # the scheduler does not look during this test

    def reasons():
        _, listing = request(server, "GET", "/api/queue")
        return [(item["run"], item["waiting"], bool(item["held"])) for item in listing["items"]]

    queue_one(server, "first")
    card = reasons()[0][1]
    assert "running another forecast" in card, card
    queue_one(server, "second")
    assert reasons()[1] == ("second", "Waiting for the forecasts ahead of it.", False)
    # Behind only a forecast held for its size, a new one is first in line for the card.
    queue_one(server, "big", need_gib=40.0)
    server.api.queue._note(server.root / "big", json.loads((server.root / "big" / runs.QUEUED).read_text()),
                           held="Held: too big.", waiting=None)
    request(server, "POST", "/api/queue/first/remove")
    request(server, "POST", "/api/queue/second/remove")
    queue_one(server, "third")
    assert reasons()[-1] == ("third", card, False)


def test_a_forecast_that_no_longer_fits_is_held_with_the_reason_and_the_next_one_goes(queued):
    server, runner, card, disk = queued
    runner.holder = held_by("job-busy")
    queue_one(server, "huge", need_gib=40.0)
    queue_one(server, "small")
    runner.holder = None
    assert server.api.queue.tick() == ["small"]
    marker = json.loads((server.root / "huge" / runs.QUEUED).read_text(encoding="utf-8"))
    assert marker["held"].startswith("Held:") and "40.0 GiB" in marker["held"]
    _, listing = request(server, "GET", "/api/queue")
    assert listing["items"][0]["held"] == marker["held"]
    # Held for want of card memory needs the person, so the page shows it as a fault, unlike a wait for data.
    assert listing["items"][0]["awaits_data"] is False

    # A full disk holds a forecast too, and it starts by itself once there is room.
    finish(server, "small")
    runner.holder = held_by("job-busy")
    queue_one(server, "later")
    runner.holder = None
    disk["free"] = 1.0
    assert server.api.queue.tick() == []
    assert "disk" in json.loads((server.root / "later" / runs.QUEUED).read_text(encoding="utf-8"))["held"]
    disk["free"] = 200.0
    assert server.api.queue.tick() == ["later"]


def test_a_forecast_too_big_for_the_card_is_held_at_once_even_behind_others(queued):
    """A forecast bigger than the card, queued third, said "Waiting for the forecasts ahead of it" until its turn."""

    server, runner, _, _ = queued
    runner.holder = held_by("job-busy")
    queue_one(server, "first")
    queue_one(server, "second")
    queue_one(server, "huge", need_gib=40.0)
    assert server.api.queue.tick() == []
    held = {name: json.loads((server.root / name / runs.QUEUED).read_text(encoding="utf-8"))
            for name in ("first", "second", "huge")}
    assert held["first"]["held"] is None and held["first"]["waiting"]
    assert held["second"]["waiting"] == "Waiting for the forecasts ahead of it."
    assert held["huge"]["held"].startswith("Held:") and "40.0 GiB" in held["huge"]["held"]
    assert held["huge"]["waiting"] is None
    _, body = request(server, "GET", "/api/queue?need_gib=5.3")
    assert body["expect"]["ahead"] == 2          # a held forecast does not stand in front of a new one


def test_another_owners_line_holds_the_card_and_ours_is_taken_and_handed_over(queued, tmp_path):
    server, runner, card, _ = queued
    owner = tmp_path / "gpu-mutex" / "OWNER"
    owner.parent.mkdir()
    (owner.parent / "OWNER.lock").write_text("", encoding="utf-8")
    owner.write_text(f"desk-gui 2026-09-26T00:00:00Z pid {os.getpid()} bounded 60 min\n", encoding="utf-8")
    server.api.queue.owner_file = str(owner)
    server.api.queue.owner_tag = "queue-test"
    runner.holder = None
    queue_one(server, "waits")
    assert server.api.queue.tick() == []
    marker = json.loads((server.root / "waits" / runs.QUEUED).read_text(encoding="utf-8"))
    assert marker["waiting"] == "The card is held by desk-gui."
    _, body = request(server, "GET", "/api/queue?need_gib=5.3")
    assert body["expect"]["busy"] and body["expect"]["ahead"] == 1

    owner.write_text("", encoding="utf-8")
    server.api.queue.pid = 424_242        # this server's own line is handed to the run it starts
    assert server.api.queue.tick() == ["waits"]
    lines = owner.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and lines[0].startswith("queue-test ") and f" pid {os.getpid()} " in lines[0]
    # The run's pid is the fake runner's (this test process); once it is gone the line goes, and only it.
    owner.write_text(owner.read_text(encoding="utf-8").replace(f" pid {os.getpid()} ", " pid 999999999 ")
                     + "someone 2026-09-26T00:00:00Z pid 1 bounded 5 min\n", encoding="utf-8")
    document = server.api.queue._document()
    document["owned"] = [{"run": "waits", "pid": 999_999_999}]
    server.api.queue._save(document)
    server.api.queue.tick()
    assert owner.read_text(encoding="utf-8") == "someone 2026-09-26T00:00:00Z pid 1 bounded 5 min\n"


def _wrapper_runner(jobs_root):
    """The real job manager and wrapper, launching a stand-in for the engine (a sleep) that needs no CuPy."""

    from woof.gui.jobs import Runner

    class WrapperRunner(Runner):
        def runtime_gap(self):
            return None

    return WrapperRunner(jobs_root=jobs_root)


def _owner_queue(tmp_path, runner):
    owner = tmp_path / "gpu-mutex" / "OWNER"
    owner.parent.mkdir()
    owner.write_text("", encoding="utf-8")
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    queue = server.api.queue
    queue.owner_file = str(owner)
    queue.owner_tag = "gui-test"
    queue._cards = lambda: {"devices": [{"name": "Test card", "memory_total_mib": 16 * 1024,
                                          "memory_used_mib": 512}], "processes": []}
    return server, queue, owner


def _started(server, queue, owner, name, seconds):
    rundir = server.root / name
    rundir.mkdir(parents=True)
    argv = [sys.executable, "-c", f"import time; time.sleep({seconds})"]
    reply = queue.launch_holding_card(
        name, lambda: server.api._launch_now(name, rundir, argv, owner_file=str(owner)), 5.0)
    return int(reply.body["job"]["wrapper_pid"])


def _gone(pid, within=60.0):
    from woof.machine_agent import pid_alive

    deadline = time.monotonic() + within
    while pid_alive(pid):
        assert time.monotonic() < deadline, f"pid {pid} still running"
        time.sleep(0.2)


def test_a_runs_owner_line_goes_when_the_run_ends_with_no_page_server_left(tmp_path):
    """Closing ``woof gui`` mid-run left the run's OWNER line behind after the run ended (a real wrapper here)."""

    server, queue, owner = _owner_queue(tmp_path, _wrapper_runner(tmp_path / "jobs"))
    wrapper = _started(server, queue, owner, "ends-alone", 3)
    lines = owner.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and lines[0].startswith("gui-test ") and f" pid {wrapper} " in lines[0]
    server.server_close()          # the page server is gone, and its scheduler never ran
    _gone(wrapper)
    assert owner.read_text(encoding="utf-8") == ""


def test_a_run_that_ends_before_its_line_reaches_it_leaves_no_line(tmp_path, monkeypatch):
    from woof.machine_agent import retag_card

    server, queue, owner = _owner_queue(tmp_path, _wrapper_runner(tmp_path / "jobs"))
    real_retag = retag_card

    def late(owner_file, old, new, **kwargs):
        _gone(new)                 # the run and its wrapper end before the hand-over
        return real_retag(owner_file, old, new, **kwargs)

    monkeypatch.setattr("gpuwm.machine_agent.retag_card", late)
    _started(server, queue, owner, "ends-early", 0)
    assert owner.read_text(encoding="utf-8") == ""
    server.server_close()


def test_start_now_is_offered_only_when_the_card_can_take_a_second_forecast(queued):
    server, runner, card, _ = queued
    card["processes"] = [{"pid": 1, "used_mib": 2048}]
    _, body = request(server, "GET", "/api/queue?need_gib=5.3")
    # Another program uses 2.5 GB of 16; this forecast fits beside it, so it starts now.
    assert not body["expect"]["busy"] and body["expect"]["start_now"]
    card["devices"][0]["memory_used_mib"] = 14 * 1024
    server.api.queue._local = None
    _, body = request(server, "GET", "/api/queue?need_gib=5.3")
    assert body["expect"]["busy"] and not body["expect"]["start_now"]
    assert "Another program is using the card" in body["expect"]["why"]


def test_a_dry_run_queue_writes_nothing(queued):
    server, runner, _, _ = queued
    response, body = request(server, "POST", "/api/create/start",
                             body={**DRAFT, "name": "dry", "queue": True, "dry_run": True})
    assert response.status == 200 and body["dry_run"] and body["queued"]
    assert not (server.root / "dry").exists() and server.api.queue.order() == []


def test_a_page_is_answered_at_once_while_the_queue_starts_a_forecast(queued):
    """The scheduler's start (the launch, nvidia-smi, SSH for a Machines node) never holds a page's request."""

    server, runner, card, _ = queued
    runner.holder = held_by("job-busy")
    queue_one(server, "first")
    launching = threading.Event()
    release = threading.Event()
    launch = runner.launch

    def slow_launch(rundir, argv):
        launching.set()
        release.wait(30)
        return launch(rundir, argv)

    runner.launch = slow_launch
    runner.holder = None
    ticked = []
    worker = threading.Thread(target=lambda: ticked.extend(server.api.queue.tick()))
    worker.start()
    try:
        assert launching.wait(10)
        elapsed, response, body = timed(server, "GET", "/api/queue?need_gib=5.3")
        assert response.status == 200 and elapsed < 1.0, elapsed
        # The one starting has left the line a page sees; one queued meanwhile goes in behind it.
        assert body["items"] == []
        elapsed, response, _ = timed(server, "POST", "/api/create/start",
                                     body={**DRAFT, "name": "second", "queue": True, "need_gib": 5.3})
        assert response.status == 200 and elapsed < 1.0, elapsed
        # A Remove of the one starting is refused rather than deleting a folder a run is starting in.
        response, _ = request(server, "POST", "/api/queue/first/remove")
        assert response.status == 404 and (server.root / "first").is_dir()
    finally:
        release.set()
        worker.join(30)
    assert ticked == ["first"]
    assert server.api.queue.order() == ["second"]


def test_a_page_read_of_the_queue_never_walks_the_forecasts_folder(queued, monkeypatch):
    """With 1,500 run folders each GET /api/queue walked the whole folder twice (1 to 2 s), every 3 s from Review."""

    server, runner, _, _ = queued
    runner.holder = held_by("job-busy")
    queue_one(server, "a")
    walks = []
    walk = runs.iter_runs
    monkeypatch.setattr(runs, "iter_runs", lambda root: walks.append(root) or walk(root))
    for _ in range(3):
        response, body = request(server, "GET", "/api/queue?need_gib=5.3")
        assert response.status == 200 and [item["run"] for item in body["items"]] == ["a"]
        assert body["expect"]["ahead"] == 1
    request(server, "GET", "/api/runs")
    walks.clear()
    request(server, "GET", "/api/queue")
    assert walks == []

    # A queued folder the order file does not list (a crash between its two writes) joins the line when the
    # scheduler looks through the folder: when it starts, and then once a minute, never on every tick.
    stray = server.root / "stray"
    stray.mkdir()
    (stray / "plan.json").write_text("{}", encoding="utf-8")
    (stray / runs.QUEUED).write_text(json.dumps({"queued_utc": "2026-09-26T00:00:00Z", "machine": "this-computer"}),
                                     encoding="utf-8")
    server.api.queue.tick()
    assert len(walks) == 1 and server.api.queue.order() == ["a", "stray"]
    server.api.queue.tick()
    assert len(walks) == 1
    server.api.queue._scanned -= queue_module.SCAN_EVERY_S
    server.api.queue.tick()
    assert len(walks) == 2


def test_a_page_read_of_the_card_never_waits_on_nvidia_smi(queued):
    server, runner, card, _ = queued
    server.api.queue.local_card(fresh=True)
    server.api.queue.tick_s = 0.01            # every reading is already old

    def slow():
        time.sleep(2.0)
        return card

    server.api.queue._cards = slow
    time.sleep(0.05)
    elapsed, response, body = timed(server, "GET", "/api/queue?need_gib=5.3")
    assert response.status == 200 and elapsed < 1.0, elapsed
    assert body["expect"]["card_name"] == "Test card"


def test_a_start_that_does_not_happen_keeps_its_place_in_line(queued):
    from woof.gui.jobs import Refused

    server, runner, _, _ = queued
    runner.holder = held_by("job-busy")
    for name in ("a", "b", "c"):
        queue_one(server, name)
    runner.holder = None
    launch = runner.launch

    def refuse(rundir, argv):
        raise Refused("the GPU is held by running job job-other")

    runner.launch = refuse
    assert server.api.queue.tick() == []
    assert server.api.queue.order() == ["a", "b", "c"]
    marker = json.loads((server.root / "a" / runs.QUEUED).read_text(encoding="utf-8"))
    assert "job-other" in marker["waiting"]
    assert not (server.root / "a" / queue_module.STARTING).exists()
    runner.launch = launch
    assert server.api.queue.tick() == ["a"]
    assert server.api.queue.order() == ["b", "c"]


def test_a_start_a_crash_cut_short_goes_back_in_its_place(queued):
    server, runner, _, _ = queued
    runner.holder = held_by("job-busy")
    for name in ("cut", "began", "next"):
        queue_one(server, name)
    # The page server stopped between taking each off the line and hearing back from its start: "cut" never
    # started; "began" did (its job file is there).
    for name in ("cut", "began"):
        os.replace(server.root / name / runs.QUEUED, server.root / name / queue_module.STARTING)
    (server.root / "began" / runs.JOB).write_text(json.dumps({"wrapper_pid": 999_999_9}), encoding="utf-8")
    again = build_server(server.root, port=0, runner=Gated(), token="u" * 43)
    try:
        again.api.queue._cards = server.api.queue._cards
        again.api.queue._disk = server.api.queue._disk
        again.api.runner.holder = held_by("job-busy")
        assert again.api.queue.order() == ["next"]
        assert again.api.queue.tick() == []
        assert again.api.queue.order() == ["cut", "next"]
        assert not (server.root / "began" / queue_module.STARTING).exists()
        assert not (server.root / "began" / runs.QUEUED).exists()
    finally:
        again.server_close()


def test_start_waits_for_the_check_of_a_recent_start_it_writes_and_a_fit_does_not(tmp_path, monkeypatch):
    """A start picked while its source's check was still out passed Start and was refused only at its download."""

    def unpublished(url):
        time.sleep(1.0)
        return False

    monkeypatch.setattr("woof.source_availability.quick_head", unpublished)
    monkeypatch.setattr("woof.source_availability.published_stop", lambda window, **_: None)
    monkeypatch.setattr("woof.fetch._head_ok", lambda url: True)
    runner = FakeRunner()
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    serve_in_thread(server)
    try:
        draft = {**DRAFT, "cycle": _frontier(), "hours": 6}
        took, response, _ = timed(server, "POST", "/api/create/fit", draft)
        assert response.status == 200 and took < 1.0
        # Show command answers what Start would, so it waits for the same check and refuses the same start.
        took, response, body = timed(server, "POST", "/api/create/start", {**draft, "name": "fresh", "dry_run": True})
        assert response.status == 422, body
        assert "not published" in body["message"] and took < 25
        took, response, body = timed(server, "POST", "/api/create/start", {**draft, "name": "fresh"})
        assert response.status == 422, body
        assert "not published" in body["message"] and took < 25
        assert not runner.launched and not (server.root / "fresh").exists()
    finally:
        server.shutdown()
        server.server_close()


def test_an_events_button_waits_for_the_check_of_a_recent_start_and_sends_a_queue_only_start_to_customise(
        tmp_path, monkeypatch):
    """An event's button refused a start from the last day while its check was still out, a start Start waits for and
    takes, and told a start only Queue it takes to pick another card size."""

    from woof.gui import api as api_module

    published = {"now": True}

    def slow(url):
        time.sleep(1.0)
        return published["now"]

    monkeypatch.setattr("woof.source_availability.quick_head", slow)
    monkeypatch.setattr("woof.source_availability.published_stop", lambda window, **_: None)
    monkeypatch.setattr("woof.fetch._head_ok", lambda url: pytest.fail(f"a request probed {url} itself"))
    monkeypatch.setattr(api_module, "disk_free_gib", lambda path: 500.0)
    runner = FakeRunner()
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    serve_in_thread(server)
    start = _frontier()
    layout = {"card_gb": 16, "fits": True, "source": "gfs", "start": f"{start}:00Z", "length_h": 6, "disk_gib": 1.0,
              "box": {"lat": 35.5, "lon": -97.0, "width_km": 300, "height_km": 300},
              "intent": {"source": "gfs", "cycle": start, "hours": 6, "card": "16gb", "root_dx_km": 12}}
    store = {"events": {"recent-storm": {"title": "Recent storm"}},
             "recipes": {"recent-storm": {"title": "Recent storm", "start": f"{start}:00Z", "cards": [layout]}}}
    monkeypatch.setattr(server.api.wiki.store, "data", lambda: store)
    body = {"event": "recent-storm", "card_gb": 16}
    try:
        # The check is out when the button is pressed: it waits for it, and the start is taken.
        took, response, answer = timed(server, "POST", "/api/wiki/simulate", {**body, "dry_run": True})
        assert response.status == 200, answer
        assert took >= 1.0 and took < 25
        # Its servers have not published it (asked afresh, as a new page server would): the button says so and
        # points to Customise, where Queue it takes it.
        published["now"] = False
        with server.api.board._lock:
            server.api.board._done.clear()
            server.api.board._known.clear()
        with server.api.board.heard._lock:
            server.api.board.heard._answers.clear()
        took, response, answer = timed(server, "POST", "/api/wiki/simulate", body)
        assert response.status == 422, answer
        assert "not published" in answer["message"] and took < 25
        assert "Customise" in answer["fix"] and "Queue it" in answer["fix"] and "card size" not in answer["fix"]
        assert not runner.launched
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------- the When step in a browser

def _browser():
    """Headless Chrome through Playwright, or a skip where this machine has neither."""

    sync_api = pytest.importorskip("playwright.sync_api")
    playwright = sync_api.sync_playwright().start()
    for options in ({}, {"channel": "chrome"}, {"channel": "msedge"}):
        try:
            return playwright, playwright.chromium.launch(headless=True, **options)
        except Exception:  # noqa: BLE001 - try the next browser this machine may have
            continue
    playwright.stop()
    pytest.skip("no headless Chromium, Chrome or Edge on this machine")


@pytest.fixture()
def held_hosts(tmp_path, monkeypatch):
    """A page server whose data hosts answer only when the test lets them, so the rows stay checking."""

    gate = threading.Event()

    def held(url):
        gate.wait(20)
        return True

    monkeypatch.setattr("woof.source_availability.quick_head", held)
    monkeypatch.setattr("woof.source_availability.published_stop", lambda window, **_: None)
    monkeypatch.setattr("woof.fetch._head_ok", lambda url: True)
    server = build_server(tmp_path / "runs", port=0, runner=FakeRunner(), token="t" * 43)
    serve_in_thread(server)
    try:
        yield server, gate
    finally:
        gate.set()
        server.shutdown()
        server.server_close()


def _frontier() -> str:
    """The newest six-hourly start by this clock: the one still being made, which only its check can confirm."""

    moment = datetime.now(timezone.utc)
    hour = (moment.hour // 6) * 6
    return f"{moment:%Y-%m-%d}T{hour:02d}"


def _open_when(server, browser, asked):
    page = browser.new_page(viewport={"width": 1366, "height": 900})
    page.on("request", lambda r: asked.append(r.url) if "/api/sources/availability" in r.url else None)
    page.goto(f"{server.url}#/create/35.5,-97,600x600/when/at/{_frontier()}")
    page.locator("input[placeholder='YYYY-MM-DD']").wait_for()
    page.locator(".option.src").first.wait_for()
    return page


def test_a_date_typed_while_the_rows_check_keeps_the_keyboard(held_hosts):
    server, gate = held_hosts
    playwright, browser = _browser()
    try:
        asked = []
        page = _open_when(server, browser, asked)
        box = page.locator("input[placeholder='YYYY-MM-DD']")
        page.evaluate("""() => {
          document.querySelector("input[placeholder='YYYY-MM-DD']").dataset.mark = "kept";
          document.querySelector(".cal select").dataset.mark = "kept";
        }""")
        assert "checking" in page.locator(".option.src").first.inner_text()
        before = len(asked)
        box.click()
        box.press("Control+A")
        box.press_sequentially("2024-05-06", delay=300)
        # The rows were asked again while the keys went in, and neither the date box nor the calendar was redrawn.
        assert len(asked) - before >= 2
        mine = """() => {
          const box = document.querySelector("input[placeholder='YYYY-MM-DD']");
          return {focus: document.activeElement === box, value: box.value, box: box.dataset.mark,
                  month: document.querySelector(".cal select").dataset.mark};
        }"""
        assert page.evaluate(mine) == {"focus": True, "value": "2024-05-06", "box": "kept", "month": "kept"}
        # The hosts answer; the rows fill in beside the box without taking the keyboard.
        gate.set()
        page.wait_for_function("""() => ![...document.querySelectorAll(".option.src")].some((b) => /checking/.test(b.innerText))""",
                               timeout=20000)
        assert page.evaluate(mine) == {"focus": True, "value": "2024-05-06", "box": "kept", "month": "kept"}
        box.press("Enter")
        page.wait_for_function("""() => location.hash.includes("/at/2024-05-06T")""", timeout=5000)
    finally:
        browser.close()
        playwright.stop()


def test_next_goes_on_while_the_chosen_sources_check_is_out(held_hosts):
    """The When step held Next until the chosen source's check answered: a check still out blocked choosing."""

    server, gate = held_hosts
    playwright, browser = _browser()
    try:
        page = _open_when(server, browser, [])
        state = page.evaluate("""() => {
          const next = [...document.querySelectorAll(".nav2 button")].pop();
          const row = document.querySelector(".option.src.on");
          return {next: next.disabled, row: row ? row.innerText : ""};
        }""")
        assert not state["next"] and "checking" in state["row"], state
        gate.set()
        page.wait_for_function("""() => { const on = document.querySelector(".option.src.on");
                                          return on && !/checking/.test(on.innerText); }""", timeout=20000)
        assert not page.evaluate("""() => [...document.querySelectorAll(".nav2 button")].pop().disabled""")
    finally:
        browser.close()
        playwright.stop()


def _due_by_clock(hours=6):
    """The newest six-hourly start at least ``hours`` old by this clock: GFS's start past its usual delay."""

    moment = datetime.now(timezone.utc) - timedelta(hours=hours)
    return moment.replace(hour=moment.hour // 6 * 6, minute=0, second=0, microsecond=0)


def test_when_opens_on_the_start_past_the_usual_delay_and_names_no_newest_run_while_no_server_is_heard(
        tmp_path, monkeypatch):
    """The page opened by itself on the cycle still being made and offered it as Newest run."""

    server, _, _ = _unheard_server(tmp_path, monkeypatch, None)
    playwright, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": 1366, "height": 900})
        page.goto(f"{server.url}#/create/35.5,-97,600x600/when")
        page.locator(".option.src").first.wait_for()
        page.wait_for_function("""() => { const on = document.querySelector(".option.src.on");
                                          return on && !/checking/.test(on.innerText); }""", timeout=40000)
        opened = _due_by_clock()
        assert page.locator("input[placeholder='YYYY-MM-DD']").input_value() == f"{opened:%Y-%m-%d}"
        assert page.locator(".datetime select").input_value() == str(opened.hour)
        picks = page.locator(".choices").first.inner_text()
        assert "Newest run" not in picks and "Yesterday" in picks
        # Nothing confirmed it, but it is past GFS's usual publication time: GFS has it, the row says how it knows,
        # and Next goes on.
        row = page.locator(".option.src.on").inner_text()
        assert "has it" in row and "usually takes to publish a whole run" in row
        assert not page.evaluate("""() => [...document.querySelectorAll(".nav2 button")].pop().disabled""")
    finally:
        browser.close()
        playwright.stop()
        server.shutdown()
        server.server_close()


def test_when_does_not_pull_back_onto_an_older_start_or_call_it_the_newest_run(tmp_path, monkeypatch):
    """With NOMADS busy and the mirror silent, an older start's NOMADS HEADs got a turn: the page opened on it with
    "Newest run" picked, beside a newer "Yesterday" pick."""

    server, _, answers = _unheard_server(tmp_path, monkeypatch, None)
    opened = _due_by_clock()
    older = opened - timedelta(hours=12)
    answers[f"{NOMADS_TURN}gfs.{older:%Y%m%d}/{older:%H}/"] = True
    playwright, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": 1366, "height": 900})
        page.goto(f"{server.url}#/create/35.5,-97,600x600/when")
        page.locator(".option.src").first.wait_for()
        # The older start is found whole: the server's own answer says so.
        deadline = time.monotonic() + 40
        while True:
            assert server.api.board.wait(30)
            _, check = request(server, "GET", f"/api/sources/availability?time={opened:%Y-%m-%dT%H}&hours=6")
            row = {r["id"]: r for r in check["sources"]}["gfs"]
            if row["confirmed"] == f"{older:%Y-%m-%dT%H}" or time.monotonic() > deadline:
                break
            time.sleep(0.5)
        assert row["confirmed"] == f"{older:%Y-%m-%dT%H}" and row["newest_run"] is None
        page.wait_for_function("""() => { const on = document.querySelector(".option.src.on");
                                          return on && !/checking/.test(on.innerText); }""", timeout=40000)
        page.wait_for_timeout(1500)
        # The page stays where it opened and offers no Newest run.
        assert page.locator("input[placeholder='YYYY-MM-DD']").input_value() == f"{opened:%Y-%m-%d}"
        assert page.locator(".datetime select").input_value() == str(opened.hour)
        picks = page.locator(".choices").first.inner_text()
        assert "Newest run" not in picks and "Yesterday" in picks
    finally:
        browser.close()
        playwright.stop()
        server.shutdown()
        server.server_close()


def test_when_offers_no_newest_run_pick_while_the_start_after_the_one_found_whole_goes_unheard(tmp_path, monkeypatch):
    """With NOMADS busy and the mirror silent, the newest due start's NOMADS HEADs got a turn and the start after it
    went unheard: the page offered the start found whole as Newest run and shaded the days after it."""

    server, _, answers = _unheard_server(tmp_path, monkeypatch, None)
    opened = _due_by_clock()
    after = opened + timedelta(hours=6)
    answers[f"{NOMADS_TURN}gfs.{opened:%Y%m%d}/{opened:%H}/"] = True
    playwright, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": 1366, "height": 900})
        page.goto(f"{server.url}#/create/35.5,-97,600x600/when")
        page.locator(".option.src").first.wait_for()
        deadline = time.monotonic() + 40
        while True:
            assert server.api.board.wait(30)
            _, check = request(server, "GET", f"/api/sources/availability?time={opened:%Y-%m-%dT%H}&hours=6")
            row = {r["id"]: r for r in check["sources"]}["gfs"]
            if row["confirmed"] == f"{opened:%Y-%m-%dT%H}" or time.monotonic() > deadline:
                break
            time.sleep(0.5)
        assert row["confirmed"] == row["opening"] == f"{opened:%Y-%m-%dT%H}" and row["newest_run"] is None
        page.wait_for_function("""() => { const on = document.querySelector(".option.src.on");
                                          return on && /has it/.test(on.innerText); }""", timeout=40000)
        page.wait_for_timeout(1500)
        assert page.locator("input[placeholder='YYYY-MM-DD']").input_value() == f"{opened:%Y-%m-%d}"
        assert page.locator(".datetime select").input_value() == str(opened.hour)
        assert "Newest run" not in page.locator(".choices").first.inner_text()

        # Every server answers the start after it not whole yet: the page's next ask names it the newest run.
        piece = f"gfs.{after:%Y%m%d}/{after:%H}/"
        answers.update({piece: False, f"{NOMADS_TURN}{piece}": False})
        _age_out(server.api.board)
        page.wait_for_function("""() => /Newest run/.test(document.querySelector(".choices").innerText)""",
                               timeout=45000)
        assert page.locator(".choices .on").first.inner_text().startswith("Newest run")
    finally:
        browser.close()
        playwright.stop()
        server.shutdown()
        server.server_close()


def test_a_start_only_queue_it_takes_goes_through_to_review_and_waits_in_line(tmp_path, monkeypatch):
    """A start no data server had confirmed could not be picked at all, and nothing could wait for it."""

    runner = Gated()
    server, _, _ = _unheard_server(tmp_path, monkeypatch, None, runner=runner)
    server.api.queue._cards = lambda: {"devices": [{"name": "Test card", "memory_total_mib": 16 * 1024,
                                                    "memory_used_mib": 512}], "processes": []}
    server.api.queue._disk = lambda: 200.0
    playwright, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": 1366, "height": 900})
        page.goto(f"{server.url}#/create/35.5,-97,300x300/when/at/{_frontier()}/src/gfs")
        page.wait_for_function("""() => { const on = document.querySelector(".option.src.on");
                                          return on && !/checking/.test(on.innerText); }""", timeout=40000)
        row = page.locator(".option.src.on").inner_text()
        assert "may have it" in row and "did not answer in time" in row
        note = page.locator(".nav2 > .note")
        assert note.is_visible() and "Queue it holds the forecast" in note.inner_text()
        assert not page.evaluate("""() => [...document.querySelectorAll(".nav2 button")].pop().disabled""")

        # On to Review by its steps, as a person goes: a link straight to Review in a tab that never drew it has
        # no draft behind it, and New forecast starts that one again at Where.
        for title in ("How fine", "Physics", "Review"):
            page.wait_for_function("""() => { const b = [...document.querySelectorAll('.steppanel .nav2 button')]
                                               .find((x) => x.textContent === 'Next'); return b && !b.disabled; }""",
                                   timeout=20000)
            page.locator(".steppanel .nav2 button.primary", has_text="Next").click()
            page.wait_for_function("(n) => { const b = document.querySelector('nav.steps button.on');"
                                   " return b && b.textContent.includes(n); }", arg=title, timeout=20000)
        page.wait_for_function("""() => { const b = [...document.querySelectorAll('.nav2 button')].find((x) => /Queue it/.test(x.textContent));
                                          return b && !b.disabled; }""", timeout=40000)
        state = page.evaluate("""() => {
          const buttons = [...document.querySelectorAll('.nav2 button')];
          const now = buttons.find((x) => /Start now/.test(x.textContent));
          const note = document.querySelector('.nav2 > .note');
          const rows = [...document.querySelectorAll('table.review tr')].map((tr) => tr.innerText);
          return {now: now ? now.disabled : null, note: note && !note.hidden ? note.textContent : "",
                  starts: rows.find((r) => /^Starts/.test(r)) || ""};
        }""")
        assert state["now"] is True and "not confirmed yet" in state["note"], state
        # One line in the button bar.
        box = page.locator(".nav2 > .note").bounding_box()
        line = page.evaluate("() => parseFloat(getComputedStyle(document.querySelector('.nav2 > .note')).lineHeight)")
        assert box["height"] <= line * 1.5
        assert "Once a check confirms the start is published" in state["starts"], state
        page.locator(".nav2 button", has_text="Queue it").click()
        page.wait_for_function("() => location.hash.startsWith('#/runs')", timeout=20000)
        _, listing = request(server, "GET", "/api/queue")
        assert len(listing["items"]) == 1 and listing["items"][0]["held"].startswith("Held until a check confirms")
        assert runner.launched == []
    finally:
        browser.close()
        playwright.stop()
        server.shutdown()
        server.server_close()


def test_a_reanalysis_opens_on_its_own_newest_start_not_the_clock_less_its_delay(tmp_path, monkeypatch):
    """A page working out its opening start from its own clock and a source's usual publication delay alone opened a
    reanalysis window whose end is not published yet: the whole window must be behind the delay, not its start."""

    from test_gui_any_date import TwoSources

    monkeypatch.setattr("woof.source_availability.quick_head", lambda url: True)
    monkeypatch.setattr("woof.source_availability.published_stop", lambda window, **_: None)
    monkeypatch.setattr("woof.fetch._head_ok", lambda url: True)
    server = build_server(tmp_path / "runs", port=0, runner=TwoSources(), token="t" * 43)
    serve_in_thread(server)
    playwright, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": 1366, "height": 900})
        page.goto(f"{server.url}#/create/35.5,-97,600x600/when/src/era5")
        page.wait_for_function("""() => { const on = document.querySelector(".option.src.on");
                                          return on && /ERA5/.test(on.innerText) && !/checking/.test(on.innerText); }""",
                               timeout=40000)
        page.wait_for_timeout(1500)
        cycle = page.evaluate("() => location.hash.split('/at/')[1].split('/')[0]")
        _, check = request(server, "GET", f"/api/sources/availability?time={cycle}&hours=6")
        row = {r["id"]: r for r in check["sources"]}["era5"]
        assert cycle == row["opening"] == row["confirmed"] == row["newest_run"], (cycle, row)
        assert row["state"] == "yes" and row["starts"] == "now"
        assert "Newest run" in page.locator(".choices").first.inner_text()
    finally:
        browser.close()
        playwright.stop()
        server.shutdown()
        server.server_close()


def test_a_links_own_start_survives_the_page_being_drawn_again_while_it_loads(tmp_path, monkeypatch):
    """A New forecast link opened while the app was still loading lost its own start: the page rewrote the link
    without it until the source list answered, and a second draw from the rewritten link opened on the server's
    opening start instead."""

    monkeypatch.setattr("woof.source_availability.quick_head", lambda url: True)
    monkeypatch.setattr("woof.source_availability.published_stop", lambda window, **_: None)
    monkeypatch.setattr("woof.fetch._head_ok", lambda url: True)
    server = build_server(tmp_path / "runs", port=0, runner=FakeRunner(), token="t" * 43)
    serve_in_thread(server)
    playwright, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": 1366, "height": 900})

        def slow(route):
            time.sleep(2.0)
            route.continue_()

        page.route("**/api/sources", slow)
        page.goto(f"{server.url}#/create/35.5,-97,600x600/when/at/2026-09-24T00/src/gfs")
        page.locator("input[placeholder='YYYY-MM-DD']").wait_for()
        assert "/at/2026-09-24T00" in page.evaluate("() => location.hash")
        page.evaluate("() => window.dispatchEvent(new HashChangeEvent('hashchange'))")
        page.locator("input[placeholder='YYYY-MM-DD']").wait_for()
        page.locator(".option.src").first.wait_for(timeout=20000)
        # Once the source list is in (its "Reading the list" line goes), the link's start is still the draft's.
        page.wait_for_function("""() => !/Reading the list of data sources/.test(document.body.innerText)""", timeout=20000)
        assert page.locator("input[placeholder='YYYY-MM-DD']").input_value() == "2026-09-24"
        assert "/at/2026-09-24T00" in page.evaluate("() => location.hash")
    finally:
        browser.close()
        playwright.stop()
        server.shutdown()
        server.server_close()


def test_rows_that_answer_before_the_source_list_do_not_pick_the_source(tmp_path, monkeypatch):
    """The rows came back before the source list, the page took whichever source had answered first
    (HRRR, while GFS was still checking), and then opened on that source's newest start instead of the list's own
    first choice."""

    from test_gui_any_date import TwoSources

    class ThreeSources(TwoSources):
        def query(self, argv, **kwargs):
            answer = super().query(argv, **kwargs)
            if "--sources" in argv:
                answer = {"sources": [*answer["sources"], {
                    "source_id": "hrrr", "display_name": "HRRR", "max_forecast_hour": 48,
                    "maturity": {"status": "certified_stock_wrf"}, "coverage": None,
                    "run_plan": {"intent_supported": True, "intent_routes": ["prepared"],
                                 "requires_source_root": False}}]}
            return answer

    gate = threading.Event()

    def head(url):
        if "hrrr" in url:
            return True
        gate.wait(20)
        return True

    monkeypatch.setattr("woof.source_availability.quick_head", head)
    monkeypatch.setattr("woof.source_availability.published_stop", lambda window, **_: None)
    monkeypatch.setattr("woof.fetch._head_ok", lambda url: True)
    server = build_server(tmp_path / "runs", port=0, runner=ThreeSources(), token="t" * 43)
    serve_in_thread(server)
    # The start-up warm call has the engine's answers in hand, as it does on a running page server.
    _, listed = request(server, "GET", "/api/sources")
    assert listed["preferred"][0] == "gfs"
    playwright, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": 1366, "height": 900})
        sources_back = []

        def slow(route):
            time.sleep(2.0)
            route.continue_()

        page.route("**/api/sources", slow)
        page.on("requestfinished", lambda r: sources_back.append(time.monotonic()) if r.url.endswith("/api/sources") else None)
        page.goto(f"{server.url}#/create/35.5,-97,600x600/when/at/{_frontier()}")
        # The rows answer first: GFS is still checking, and HRRR already has it.
        page.wait_for_function("""() => [...document.querySelectorAll(".option.src")].some((b) => /HRRR/.test(b.innerText) && /has it/.test(b.innerText))""",
                               timeout=20000)
        page.wait_for_function("() => !!document.querySelector('.option.src.on')", timeout=20000)
        assert sources_back, "the page picked a source before the source list came back"
        assert page.locator(".option.src.on b").inner_text() == "GFS"
        gate.set()
        page.wait_for_function("""() => { const on = document.querySelector('.option.src.on');
                                          return on && !/checking/.test(on.innerText); }""", timeout=20000)
        assert page.locator(".option.src.on b").inner_text() == "GFS"
    finally:
        gate.set()
        browser.close()
        playwright.stop()
        server.shutdown()
        server.server_close()


def test_reviews_buttons_are_not_redrawn_by_each_queue_answer(tmp_path, monkeypatch):
    """Review asks the queue every 3 s; a Queue it button replaced on each answer lost a click that spanned it."""

    monkeypatch.setattr("woof.source_availability.quick_head", lambda url: True)
    monkeypatch.setattr("woof.fetch._head_ok", lambda url: True)
    runner = Gated()
    runner.holder = held_by("job-x")
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    server.api.queue._cards = lambda: {"devices": [{"name": "Test card", "memory_total_mib": 16 * 1024,
                                                    "memory_used_mib": 512}], "processes": []}
    server.api.queue._disk = lambda: 200.0
    serve_in_thread(server)
    playwright, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": 1366, "height": 900})
        asked = []
        page.on("request", lambda r: asked.append(r.url) if "/api/queue" in r.url else None)
        # Review reached as a person reaches it: a link straight to Review in a tab that never drew it has no
        # draft behind it, and New forecast starts that one again at Where.
        page.goto(f"{server.url}#/create/35.5,-97,300x300/when/at/2026-09-24T00/src/gfs")
        for title in ("How fine", "Physics", "Review"):
            page.wait_for_function("""() => { const b = [...document.querySelectorAll('.steppanel .nav2 button')]
                                               .find((x) => x.textContent === 'Next'); return b && !b.disabled; }""",
                                   timeout=20000)
            page.locator(".steppanel .nav2 button.primary", has_text="Next").click()
            page.wait_for_function("(n) => { const b = document.querySelector('nav.steps button.on');"
                                   " return b && b.textContent.includes(n); }", arg=title, timeout=20000)
        queue_it = page.locator(".nav2 button", has_text="Queue it")
        queue_it.wait_for()
        page.wait_for_function("""() => { const b = [...document.querySelectorAll('.nav2 button')].find((x) => /Queue it/.test(x.textContent));
                                          return b && !b.disabled; }""", timeout=20000)
        page.evaluate("""() => { [...document.querySelectorAll('.nav2 button')].find((x) => /Queue it/.test(x.textContent)).dataset.mark = "kept"; }""")
        before = len(asked)
        deadline = time.monotonic() + 15
        while len(asked) - before < 3 and time.monotonic() < deadline:
            page.wait_for_timeout(250)
        assert len(asked) - before >= 3
        assert queue_it.get_attribute("data-mark") == "kept"
        # the line about Start now sits in the button bar, in view without scrolling, on one line
        box = page.locator(".nav2 > .note").bounding_box()
        assert box is not None and box["y"] + box["height"] <= 900
        line = page.evaluate("() => parseFloat(getComputedStyle(document.querySelector('.nav2 > .note')).lineHeight)")
        assert box["height"] <= line * 1.5
        # ... and the bar never covers the Name box (at 1366 by 900 it hid all but the label)
        name = page.locator(".steppanel .fields.sub input").first.bounding_box()
        bar = page.locator(".steppanel .nav2").bounding_box()
        assert name is not None and name["y"] + name["height"] <= bar["y"]
    finally:
        browser.close()
        playwright.stop()
        server.shutdown()
        server.server_close()
