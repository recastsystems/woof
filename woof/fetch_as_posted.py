"""The as-posted fetch: every source's window, lead by lead as it posts.

``woof fetch --as-posted`` (the default; ``--whole-cycle`` is the old
rule) is one loop for every source (DESIGN A136 2.3).  It writes the
posting schedule first, waits for the window's start needs, then moves
each lead the moment one host holds all of it, verifies it, and
publishes a per-lead marker, in lead order, while the window's manifest
grows.  The bytes move through the transfer code that exists today --
the route engine (:func:`woof.fetch_routes.run_plan`, gated per lead)
or a legacy transport called for the verified prefix -- so no source
gets a path of its own.

Files, all under ``<out>/posting/`` (the names :mod:`woof.source_posting`
declares and :mod:`woof.chain_events` relays):

``schedule.json``  ``gpuwm.posting-schedule.v1``: every lead's
                   ``valid_time``, ``expected_at``, ``late_at``,
                   ``first_seen_at`` (with ``posted_when_first_asked``
                   once posted: true when the lead was already up at its
                   first ask, so the time only bounds its posting),
                   ``fetched_at``, ``endpoint`` and
                   ``state`` (``scheduled``, ``waiting``, ``posted``,
                   ``ready``, ``late``), once asked its ``last_answer``
                   (``posted``, ``not_posted``, ``not_heard``,
                   ``failed_verification``), the start needs and
                   ``expected_ready_at``.
``fNNN.json``      ``gpuwm.posted-lead.v1``, one per verified lead: its
                   fetched ``objects`` and, for a route that composes a
                   lead's parts into one file, those files under
                   ``composed``.
``failed.json``    ``code: source_behind`` when a lead passed its late
                   time (or the window its ``--wait-timeout-minutes``);
                   the fetch then exits 75 and a rerun resumes from the
                   verified prefix.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from woof import fetch_endpoints
from woof import fetch_pool
from woof import source_posting as rows
from woof import source_readiness as readiness
from woof.source_posting import (POSTING_DIRNAME, POSTING_FAILED_NAME,
                                  POSTING_SCHEDULE_NAME)

MARKER_SCHEMA = "gpuwm.posted-lead.v1"

#: The exit code of a fetch whose source fell behind its budget
#: (DESIGN A136 3.6); run-plan and go use the same code.
SOURCE_BEHIND_EXIT = 75


#: rw_fetch's refusal of an object its host does not hold (rw-fetch
#: plan.rs ``decide``): the object is not posted there, not a fault.
OBJECT_ABSENT_REASON = "the GRIB object is not present at this source"


def object_not_posted(error: BaseException) -> bool:
    """Whether a transfer failed only because an object is not up yet.

    The breakage this catches: a lead whose posted check did not ask one
    of the objects its transfer reads (HRRR's ``wrfsfc``, posted up to
    ~10 min after ``wrfnat``/``wrfprs``) failed the whole run at once on
    rw_fetch's absent-object refusal instead of being waited for.
    """

    from woof.rustwx_fetch import RwFetchError

    return (isinstance(error, RwFetchError)
            and OBJECT_ABSENT_REASON in str(error.reason))


class SourceBehind(RuntimeError):
    """A lead passed its late time, or the window its whole-window cap."""

    exit_code = SOURCE_BEHIND_EXIT

    def __init__(self, message: str, record: Mapping[str, object]):
        super().__init__(message)
        self.record = dict(record)


class FetchStopped(RuntimeError):
    """The run this fetch fetched for ended, so the fetch ended too."""


#: A stop per fetch folder, registered by a chain that runs this fetch in
#: its own process beside the preparation (run-plan's staged chain).  The
#: fetch's waits are minutes long and run on its transfer workers' threads
#: as well as its own, so the stop is keyed by the folder, not carried by
#: the calling context.
_STOPS: dict[str, threading.Event] = {}
_STOPS_LOCK = threading.Lock()


def _stop_key(out) -> str:
    return str(Path(out).resolve())


def stop_event(out) -> threading.Event:
    """The event that ends the as-posted fetch into ``out`` at its next wait.

    The breakage it prevents: a chain that failed beside its fetch (the
    preparation or the forecast) returned while the fetch, in the same
    process, went on waiting for later leads for as long as their budget
    allows, downloading for a run that had ended.
    """

    with _STOPS_LOCK:
        return _STOPS.setdefault(_stop_key(out), threading.Event())


def release_stop(out) -> None:
    """Forget ``out``'s stop once its fetch has ended."""

    with _STOPS_LOCK:
        _STOPS.pop(_stop_key(out), None)


def now_utc() -> datetime:
    """The loop's clock (naive UTC); a replay test puts its own here."""

    return datetime.now(timezone.utc).replace(tzinfo=None)


#: The loop's wait between asks.  The real :func:`time.sleep` is the
#: stop-aware wait inside the transfer pool (a sibling's failure ends it
#: at once); a replay test puts its own clock here.
pause = time.sleep

#: How often (real seconds) the posting watch wakes to see whether a lead
#: has come due.  A lead is still asked at most once per ``poll_seconds``
#: of the loop's clock; this only bounds how late after its due time the
#: first ask can be.
WATCH_WAKE_SECONDS = 5.0
#: The longest :meth:`PostingLoop.close` waits for an ask in flight: one
#: HEAD's timeout (``fetch_endpoints.settled_object_answer``).  An ask of
#: a host that cannot be heard, asked again after it, ends on its own and
#: writes nothing (the watch is stopped).
WATCH_CLOSE_SECONDS = 60.0
#: How often (real seconds) a transfer worker's posting wait under a
#: chain's stop looks whether another file failed the request
#: (:meth:`PostingLoop._pause`): the two stops are separate events, and a
#: wait can block on only one of them.
STOP_WAKE_SECONDS = 0.25


def _clock(moment: datetime | None) -> str:
    return "?" if moment is None else f"{moment:%H:%M}Z"


def marker_name(lead: int) -> str:
    return f"f{int(lead):03d}.json"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    from woof.fetch_routes import _write_json as write

    write(path, payload)


class PostingLoop:
    """The schedule, the per-lead wait and the markers of one window.

    Called with a lead (``loop(lead)``) it is the ``lead_gate`` the
    route engine asks before an object of that lead moves: it blocks
    until one host holds the whole lead and returns that host, or
    raises :class:`SourceBehind` once the lead is late.  It is safe to
    call from the transfer pool's workers: one caller per lead asks the
    host, the others wait for its answer.
    """

    def __init__(self, window: readiness.Window, out: Path, *,
                 late_after_minutes: float | None = None,
                 wait_timeout_minutes: float | None = None,
                 nest_start_leads: Mapping[str, int] | None = None,
                 probe=None, now: Callable[[], datetime] | None = None,
                 sleep: Callable[[float], None] | None = None,
                 progress=print) -> None:
        self.window = window
        self.out = Path(out)
        self.folder = self.out / POSTING_DIRNAME
        self.probe = probe
        self.now = now if now is not None else (lambda: now_utc())
        self.sleep = sleep
        self.progress = progress
        self.row = rows.posting(window.source, window.provider)
        self.poll = float(self.row.poll_seconds or 30.0)
        self.budget = rows.late_after_minutes(
            window.source, late_after_minutes, provider=window.provider)
        started = self.now()
        self.deadline = (None if wait_timeout_minutes is None
                         else started + timedelta(minutes=float(wait_timeout_minutes)))
        self.wait_timeout_minutes = wait_timeout_minutes
        self.needs = readiness.start_needs(
            window, late_after_minutes=late_after_minutes,
            nest_start_leads=nest_start_leads)
        self.document = rows.schedule(
            window.source, window.cycle, window.leads, member=window.member,
            as_posted=True, late_after_minutes_override=late_after_minutes,
            provider=window.provider)
        stamps = [need.expected_at for need in self.needs
                  if need.expected_at is not None]
        self.document["start_needs"] = [need.row() for need in self.needs]
        self.document["expected_ready_at"] = rows.instant(
            max(stamps) if stamps else None)
        self.document["schedule_path"] = str(self.folder / POSTING_SCHEDULE_NAME)
        self._rows = {int(row["lead"]): row for row in self.document["leads"]}
        self._lock = threading.RLock()
        self._lead_locks: dict[int, threading.Lock] = {}
        self._seen: dict[int, str | None] = {}
        #: Leads whose host could not be heard when they were due: the
        #: transfer is let try them, as the one-shot fetch does (GS-05),
        #: and the download's own checks decide.  A lead here is never
        #: counted posted, and a transfer of it that fails is asked again
        #: each round until its late time.
        self._tried: dict[int, str] = {}
        #: Leads a host has answered ``not_posted`` at least once.  A lead
        #: posted when it was first asked (a late launch, a past cycle) was
        #: already up then: its ``first_seen_at`` is the ask, which only
        #: bounds when it posted, so its row says
        #: ``posted_when_first_asked`` and a row re-fit reads it as a bound.
        self._absent: set[int] = set()
        #: The posting watch (:meth:`watch`): when it last asked each lead.
        self._watch_asked: dict[int, datetime] = {}
        self._watch_stop = threading.Event()
        self._watcher: threading.Thread | None = None
        self.label = f"fetch {window.source} {window.cycle:%Y-%m-%dT%H}"
        if window.member:
            self.label += f" {window.member}"
        with _STOPS_LOCK:
            #: The stop a chain registered for this folder (:func:`stop_event`).
            self._stop_signal = _STOPS.get(_stop_key(self.out))

    # ------------------------------------------------------------------
    # files

    def start(self, *, fresh: bool = False) -> None:
        """Clear a previous run's failure and publish the schedule.

        ``fresh`` (a forced refetch) drops every lead marker as well:
        the payloads they describe were just moved aside, and a marker
        left behind would tell a reader a lead is ready whose file is
        gone.
        """

        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / POSTING_FAILED_NAME).unlink(missing_ok=True)
        if fresh:
            for marker in self.folder.glob("f[0-9][0-9][0-9].json"):
                marker.unlink(missing_ok=True)
        for lead, row in self._rows.items():
            marker = self.folder / marker_name(lead)
            if marker.is_file():
                # A rerun resumes from the verified prefix: a lead whose
                # marker is still here was fetched and verified before.
                try:
                    known = json.loads(marker.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    marker.unlink(missing_ok=True)
                    continue
                if (known.get("cycle") != self.document["cycle"]
                        or known.get("source") != self.window.source):
                    marker.unlink(missing_ok=True)
                    continue
        self.write_schedule()

    def write_schedule(self) -> None:
        with self._lock:
            _write_json(self.folder / POSTING_SCHEDULE_NAME, self.document)

    def _mark(self, lead: int, **fields) -> None:
        with self._lock:
            row = self._rows.get(int(lead))
            if row is None:
                return
            row.update(fields)
            self.write_schedule()

    # ------------------------------------------------------------------
    # waiting

    def _stop(self, lead: int | None, answer: str | None, *,
              cap: bool = False, need: readiness.Need | None = None,
              error: BaseException | None = None) -> None:
        lead_value = None if lead is None else int(lead)
        row = self._rows.get(lead_value, {}) if lead_value is not None else {}
        expected = (need.expected_at if need is not None
                    else readiness.parse_instant(row.get("expected_at")))
        late = (need.late_at if need is not None
                else readiness.parse_instant(row.get("late_at")))
        source = need.source if need is not None else self.window.source
        cycle = f"{self.window.cycle:%Y-%m-%dT%H}"
        heard = answer != readiness.NOT_HEARD
        named = f"{source} f{lead_value:03d}" if lead_value is not None else source
        if cap:
            said = (f"{named} of the {cycle} cycle is not in after the whole "
                    f"window's --wait-timeout-minutes "
                    f"{float(self.wait_timeout_minutes):g}")
        elif not heard:
            said = (f"{named} of the {cycle} cycle could not be heard by "
                    f"{_clock(late)}, {self.budget:g} min after its scheduled "
                    f"time ({_clock(expected)}); the host did not answer "
                    "whether it holds it, which is not the publisher being "
                    "late")
            if error is not None:
                said += f", and its transfer did not succeed either ({error})"
        elif answer == "failed_verification":
            said = (f"{named} of the {cycle} cycle did not verify by "
                    f"{_clock(late)}, {self.budget:g} min after its scheduled "
                    f"time ({_clock(expected)}; the source table's "
                    f"late_after_minutes is {self.budget:g})")
        else:
            said = (f"{named} of the {cycle} cycle has not posted by "
                    f"{_clock(late)}, {self.budget:g} min after its scheduled "
                    f"time ({_clock(expected)}; the source table's "
                    f"late_after_minutes is {self.budget:g})")
        kept = sorted(lead for lead, row in self._rows.items()
                      if row.get("state") == "ready")
        said += (f".  The fetched leads through f{kept[-1]:03d} are kept"
                 if kept else ".  No lead was fetched")
        said += ("\n  what to do: launch the same command again once "
                 f"{'it' if lead_value is None else f'f{lead_value:03d}'} "
                 "posts (it resumes from the fetched prefix), or raise "
                 "[fetch] late_after_minutes for this source.")
        record = {
            "code": "source_behind", "source": source,
            "member": self.window.member, "cycle": cycle, "lead": lead_value,
            "valid_time": (None if lead_value is None else rows.instant(
                self.window.cycle + timedelta(hours=lead_value))),
            "expected_at": rows.instant(expected),
            "late_at": rows.instant(late),
            "late_after_minutes": self.budget, "heard": heard,
            "last_answer": answer,
            "budget": "wait_timeout_minutes" if cap else "late_after_minutes",
            "transfer_error": None if error is None else str(error),
            "leads_kept": kept, "message": said,
        }
        self._watch_stop.set()
        self.folder.mkdir(parents=True, exist_ok=True)
        _write_json(self.folder / POSTING_FAILED_NAME, record)
        if lead_value is not None:
            self._mark(lead_value, state="late")
        raise SourceBehind(said, record)

    def check_stop(self) -> None:
        """Raise :class:`FetchStopped` once the run fetched for has ended."""

        if self._stop_signal is not None and self._stop_signal.is_set():
            raise FetchStopped(
                f"{self.label}: stopped, because the run it fetched for "
                "ended")

    def _pause(self, seconds: float) -> None:
        """Wait ``seconds`` between asks, or less once either stop fires.

        Two stops end the wait.  The chain's (:func:`stop_event`, c3b4f47a2):
        the run this fetch fetched for has ended, so :class:`FetchStopped`.
        And, on a transfer worker (the gate :meth:`__call__` and
        :meth:`refetch` run inside the pool's jobs), the request's: another
        file failed it, so :class:`woof.fetch_pool.TransferCancelled`
        (:func:`woof.fetch_pool.sleep_unless_stopped`, e4224502f).

        The breakage it prevents: under a chain's stop the wait blocked on
        that stop alone, so a worker waiting for a later lead to post slept
        its wait out after another file had failed the request.  The pool
        waits for its jobs to wind down before it raises, so the fetch's
        failure, its ``failed.json`` and the preparation waiting on the
        failed lead all waited for leads not yet posted, while the other
        workers went on downloading.
        """

        self.check_stop()
        seconds = max(0.0, seconds)
        if (self._stop_signal is not None and self.sleep is None
                and pause is time.sleep):
            job = fetch_pool.current_job()
            if job is None:
                # The fetch's own thread (its start wait): only the
                # chain's stop can end it.
                self._stop_signal.wait(seconds)
            else:
                # The two stops are separate events: wait on the chain's
                # in short slices, looking at the request's between them.
                end = time.monotonic() + seconds
                while True:
                    job.raise_if_stopped()
                    left = end - time.monotonic()
                    if left <= 0 or self._stop_signal.wait(
                            min(left, STOP_WAKE_SECONDS)):
                        break
                job.raise_if_stopped()
        else:
            fetch_pool.sleep_unless_stopped(
                seconds, sleep=self.sleep if self.sleep is not None else pause)
        self.check_stop()

    def _await(self, lead: int | None, ask, *, expected: datetime | None,
               late: datetime | None, need: readiness.Need | None = None
               ) -> dict[str, object]:
        """Ask by the polling rule until posted; stop when late."""

        announced = False
        last = None
        while True:
            now = self.now()
            if self.deadline is not None and now >= self.deadline:
                self._stop(lead, last, cap=True, need=need)
            # An AWS mirror can finish before its planning allowance.  Its
            # positive answer, rather than the predicted time, opens the gate.
            aws = fetch_endpoints.policy_uses_aws(
                self.window.source if need is None else need.source)
            ask_from = (None if expected is None or aws
                        else expected - readiness.ASK_AHEAD)
            if ask_from is not None and now < ask_from:
                if not announced:
                    self._announce(lead, expected, late, need)
                    announced = True
                self._pause(self._bounded(ask_from - now, now))
                continue
            answer = ask()
            self._answered(lead, need, answer["answer"])
            self._heard_absent(lead, need, answer["answer"])
            if answer["answer"] in (readiness.POSTED, readiness.NOT_HEARD):
                # Posted, or the host could not be heard: a host not heard
                # is never counted absent, so the transfer is let try the
                # lead and its own checks decide (GS-05).  A transfer that
                # then fails is asked again each round until the late time.
                return answer
            last = answer["answer"]
            if late is not None and now >= late:
                self._stop(lead, last, need=need)
            if not announced:
                self._announce(lead, expected, late, need)
                announced = True
            wait = timedelta(seconds=self.poll)
            if late is not None:
                wait = min(wait, max(late - now, timedelta(seconds=1)))
            self._pause(self._bounded(wait, now))

    def _own(self, lead: int | None, need: readiness.Need | None) -> bool:
        """Whether a wait on ``need`` is the wait of this window's ``lead``.

        A start need on the window's own source and lead (the analysis, the
        first boundary, a whole cycle's final lead) is that lead's own
        wait; a donor or an invariant object is not a lead of this window
        and leaves the rows alone.
        """

        return lead is not None and (
            need is None or (need.source == self.window.source
                             and need.lead is not None
                             and int(need.lead) == int(lead)))

    def _answered(self, lead: int | None, need: readiness.Need | None,
                  answer: str | None) -> None:
        """Record the host's last answer about ``lead`` on its schedule row.

        The breakage it prevents: a start-need lead whose host could not be
        heard left its row ``scheduled`` (the fetch lets the transfer try
        such a lead, GS-05), so the run's start wait read "not fetched yet"
        or "not posted yet" for a lead nothing could see, and something the
        engine could not see is never reported as the publisher being late
        (DESIGN A136 3.6).  A start need not heard is the run's wait, so its
        row is ``waiting`` with that answer.  Written only when the answer
        changes, so a poll does not rewrite the schedule.
        """

        if answer is None or not self._own(lead, need):
            return
        lead = int(lead)
        with self._lock:
            row = self._rows.get(lead)
            if row is None:
                return
            fields = {}
            if row.get("last_answer") != answer:
                fields["last_answer"] = answer
            if (answer == readiness.NOT_HEARD and need is not None
                    and row.get("state") in (None, "scheduled")):
                fields["state"] = "waiting"
            if fields:
                self._mark(lead, **fields)

    def _bounded(self, wait: timedelta, now: datetime) -> float:
        if self.deadline is not None:
            wait = min(wait, max(self.deadline - now, timedelta(seconds=1)))
        return wait.total_seconds()

    def _heard_absent(self, lead: int | None, need: readiness.Need | None,
                      answer: str | None) -> None:
        """Note a host saying this window's ``lead`` is not posted yet."""

        if answer != readiness.NOT_POSTED or lead is None:
            return
        if need is not None and not (need.source == self.window.source
                                     and need.lead is not None
                                     and int(need.lead) == int(lead)):
            # A donor or an invariant object, not this lead.
            return
        with self._lock:
            self._absent.add(int(lead))

    def _announce(self, lead, expected, late, need) -> None:
        what = (f"f{int(lead):03d}" if need is None
                else f"{need.role} {need.source} f{int(need.lead):03d}"
                if need.lead is not None else f"{need.role} objects")
        self.progress(f"{self.label}: {what} not posted yet; the schedule "
                      f"says from about {_clock(expected)}"
                      + (f"; waiting until {_clock(late)}" if late else ""))
        # A start need on the window's own source and lead is that lead's
        # own wait, so its row says so too (:meth:`_own`).
        if self._own(lead, need):
            self._mark(lead, state="waiting")

    def wait_start(self) -> None:
        """Wait for every start need (the run's ``phase: start`` wait)."""

        for need in self.needs:
            lead = self.window.start_lead if need.lead is None else need.lead
            # A lead already seen covers the needs that ARE that lead.  A
            # supplement or invariant names its own objects, which the lead's
            # posted answer never asked: the breakage this prevents is the
            # HRRR-PRS vegetation_surface wrfsfc f000 (6a69b356f), skipped
            # because the analysis need had just marked lead 0 seen, then
            # transferred without one HEAD while it might not be posted.
            covered = not (need.role == "invariant"
                           or need.role.startswith("supplement:"))
            if (covered and need.source == self.window.source
                    and int(lead) in self._seen):
                continue

            def ask(need=need):
                return readiness.need_answer(self.window, need,
                                             probe=self.probe, now=self.now())

            answer = self._await(lead if need.source == self.window.source
                                 else None, ask, expected=need.expected_at,
                                 late=need.late_at, need=need)
            if need.source == self.window.source and need.role in (
                    "analysis", "first_boundary", "whole_cycle"):
                if answer["answer"] == readiness.POSTED:
                    self._posted(int(lead), answer)
                else:
                    self._unheard(int(lead), answer["answer"])
        self.watch()

    # ------------------------------------------------------------------
    # the posting watch

    def watch(self) -> None:
        """Ask each lead on its own schedule, whatever the transfers reach.

        The polling rule (DESIGN A136 2.2) asks a lead once it is due and
        every ``poll_seconds`` after, so its ``first_seen_at`` (and the
        ``lead_posted`` event read from it) is when a host first held it.
        The transfer's gate (:meth:`__call__`) asks a lead only when the
        transfer pool reaches it, so on its own it dated a lead by the
        transfer's progress whenever the transfer ran behind the posting.
        The breakage this prevents, measured live on 2026-10-01 (a rolling
        source of about 125 objects per lead, posting faster than the
        fetch moved them): each lead was said posted later and later after
        its scheduled time while the host's own listing had it on time, so
        a row re-fitted from those events would have been moved late by
        the transfer's lag, not the publisher's.

        The watch only observes: it never waits a budget out, never says a
        lead late (the gate does, by its own clock), and asks one lead at
        a time in lead order, stopping at the first that is not posted, so
        a lead not yet posted costs one ask per ``poll_seconds``, as the
        gate's own wait does.  A lead the gate is asking itself is left to
        it.  Started once the start needs are in, stopped by
        :meth:`close` or when every lead has been seen.
        """

        if self._watcher is not None or self._watch_stop.is_set():
            return
        self._watcher = threading.Thread(
            target=self._watch_run, daemon=True,
            name=f"posting-watch-{self.window.source}")
        self._watcher.start()

    def close(self) -> None:
        """Stop the posting watch (the fetch ended, whatever its outcome).

        Waits for an ask already in flight (one lead's HEADs, each under
        its own timeout), so no watch outlives the fetch that started it
        and asks a host on its behalf afterwards; a watch that has seen
        every lead has already ended.
        """

        self._watch_stop.set()
        watcher = self._watcher
        if watcher is not None and watcher is not threading.current_thread():
            watcher.join(timeout=WATCH_CLOSE_SECONDS)

    def _watch_run(self) -> None:
        while not self._watch_stop.is_set():
            try:
                finished = self._watch_round()
            except Exception as error:  # noqa: BLE001 - an observer only
                # The gate still asks every lead it reaches, so a failed
                # watch costs the posting time's precision, never the run.
                self.progress(f"{self.label}: the posting watch stopped "
                              f"({error}); each lead is dated when the "
                              "transfer reaches it")
                return
            if finished:
                return
            self._watch_stop.wait(WATCH_WAKE_SECONDS)

    def _watch_round(self) -> bool:
        """One pass in lead order; True once no lead is left to watch."""

        for lead in sorted(self._rows):
            if self._watch_stop.is_set():
                return True
            with self._lock:
                if lead in self._seen:
                    continue
                if lead in self._tried:
                    # Let through unheard: its transfer decides, and the
                    # leads after it are still watched.
                    continue
                gate = self._lead_locks.setdefault(lead, threading.Lock())
            row = self._rows[lead]
            now = self.now()
            expected = readiness.parse_instant(row.get("expected_at"))
            late = readiness.parse_instant(row.get("late_at"))
            if (not fetch_endpoints.policy_uses_aws(self.window.source)
                    and expected is not None
                    and now < expected - readiness.ASK_AHEAD):
                return False
            if late is not None and now >= late:
                # Past its late time it is the gate's to say late.
                return False
            asked = self._watch_asked.get(lead)
            if asked is not None and (now - asked).total_seconds() < self.poll:
                return False
            if not gate.acquire(blocking=False):
                # The transfer's gate is asking this lead itself.
                return False
            try:
                with self._lock:
                    if lead in self._seen:
                        continue
                self._watch_asked[lead] = now
                answer = readiness.lead_answer(self.window, lead,
                                               probe=self.probe, now=now)
                if self._watch_stop.is_set():
                    return True
                self._answered(lead, None, answer["answer"])
                self._heard_absent(lead, None, answer["answer"])
                if (answer["answer"] != readiness.POSTED
                        or self._watch_stop.is_set()):
                    # Not posted, or the fetch ended while it was asked:
                    # nothing is written into a schedule the fetch closed.
                    return False
                self._posted(lead, answer)
            finally:
                gate.release()
        return True

    def _unheard(self, lead: int, answer: str) -> None:
        with self._lock:
            if lead not in self._seen:
                self._tried[lead] = answer

    def _posted(self, lead: int, answer: Mapping[str, object]) -> None:
        with self._lock:
            if lead in self._seen:
                return
            self._seen[lead] = answer.get("endpoint")
            self._mark(lead, state="posted",
                       first_seen_at=rows.instant(self.now()),
                       posted_when_first_asked=lead not in self._absent,
                       endpoint=answer.get("endpoint"))

    def __call__(self, lead: int) -> str | None:
        lead = int(lead)
        # No later lead starts moving for a run that has ended.
        self.check_stop()
        with self._lock:
            if lead in self._seen:
                return self._seen[lead]
            if lead in self._tried:
                return None
            gate = self._lead_locks.setdefault(lead, threading.Lock())
        with gate:
            with self._lock:
                if lead in self._seen:
                    return self._seen[lead]
                if lead in self._tried:
                    return None
            row = self._rows[lead]
            answer = self._await(
                lead, lambda: readiness.lead_answer(
                    self.window, lead, probe=self.probe, now=self.now()),
                expected=readiness.parse_instant(row.get("expected_at")),
                late=readiness.parse_instant(row.get("late_at")))
            if answer["answer"] != readiness.POSTED:
                self._unheard(lead, answer["answer"])
                return None
            self._posted(lead, answer)
            return self._seen[lead]

    def tried_unheard(self, leads) -> bool:
        """Whether any of ``leads`` is being tried with its host not heard."""

        with self._lock:
            return any(int(lead) in self._tried for lead in leads)

    def posted_now(self, lead: int) -> bool:
        """Whether ``lead`` is posted, asked once, only if it is due."""

        row = self._rows[int(lead)]
        expected = readiness.parse_instant(row.get("expected_at"))
        if int(lead) in self._seen:
            return True
        if (not fetch_endpoints.policy_uses_aws(self.window.source)
                and expected is not None
                and self.now() < expected - readiness.ASK_AHEAD):
            return False
        answer = readiness.lead_answer(self.window, int(lead),
                                       probe=self.probe, now=self.now())
        self._answered(int(lead), None, answer["answer"])
        self._heard_absent(int(lead), None, answer["answer"])
        if answer["answer"] == readiness.POSTED:
            self._posted(int(lead), answer)
            return True
        return False

    def refetch(self, lead: int, error: BaseException) -> str | None:
        """A transfer of ``lead`` failed: wait a round, ask again.

        Either it did not verify (a HEAD can answer before an operational
        server has finished writing), or its host could not be heard and
        the transfer that was let try it failed too.  Past the lead's late
        time (or the whole window's cap) the run stops with exit 75, and
        the sentence says which of the two it was; a lead its host said is
        posted whose transfer every endpoint refused ends the fetch with
        that refusal instead.
        """

        lead = int(lead)
        self.wait_again([lead], error)
        return self(lead)

    def wait_again(self, leads, error: BaseException) -> None:
        """One round's wait after a failed transfer of ``leads``, or the stop."""

        leads = [int(lead) for lead in leads]
        now = self.now()
        absent = object_not_posted(error)
        with self._lock:
            unheard = next((lead for lead in leads if lead in self._tried), None)
            answer = (self._tried[unheard] if unheard is not None
                      else readiness.NOT_POSTED if absent
                      else "failed_verification")
        named = unheard if unheard is not None else leads[0]
        # rw_fetch names the failed forecast hour. A prefix can contain
        # later hours with later budgets; they must not extend this one's.
        if absent:
            match = re.search(r"\bf(\d+):", str(error.reason))
            if match is not None and int(match[1]) in leads:
                named = int(match[1])
        for lead in leads:
            with self._lock:
                said = self._tried.get(
                    lead, readiness.NOT_POSTED if absent
                    else "failed_verification")
            self._answered(lead, None, said)
        if self.deadline is not None and now >= self.deadline:
            self._stop(named, answer, cap=True, error=error)
        lates = [readiness.parse_instant(self._rows[lead].get("late_at"))
                 for lead in ([named] if absent else leads)]
        if all(late is not None and now >= late for late in lates):
            if (unheard is None and not absent
                    and isinstance(error, fetch_endpoints.TransferRefusal)):
                # The host said the lead is posted and every endpoint then
                # refused its transfer once its own retries and throttle
                # waits were spent: the source is not behind, the transfer
                # failed, so the refusal (each host and why) ends the fetch,
                # never SourceBehind's exit 75.  Breakage it prevents (WOOF
                # 2026-10-29, run_19793057ff2979e3): ECMWF's AWS mirror
                # throttled one file of a cycle posted hours before and the
                # run failed "did not verify by 03:30Z, 60 min after its
                # scheduled time".
                raise error
            self._stop(named, answer, error=error)
        said = ("could not be heard, and its transfer did not succeed"
                if unheard is not None
                else "is not fully posted yet" if absent
                else "did not verify")
        self.progress(f"{self.label}: f{named:03d} {said} ({error}); "
                      f"asking again in {self.poll:g} s")
        self._pause(self._bounded(timedelta(seconds=self.poll), now))
        with self._lock:
            for lead in leads:
                self._seen.pop(lead, None)
                self._tried.pop(lead, None)

    # ------------------------------------------------------------------
    # markers

    def publish(self, lead: int, objects: Sequence[Mapping[str, object]], *,
                composed: Sequence[Mapping[str, object]] = ()) -> Path:
        """Write ``fNNN.json`` for a verified lead and move its row to ``ready``.

        ``objects`` are the lead's fetched and verified objects; ``composed``
        (:func:`composed_objects`) the files a route built from them, kept
        under the marker's ``composed`` key, which only a route that
        composes writes.
        """

        lead = int(lead)
        row = self._rows[lead]
        fetched = self.now()
        with self._lock:
            unheard = self._tried.pop(lead, None) is not None
        if row.get("first_seen_at") is None:
            row["first_seen_at"] = rows.instant(fetched)
            if not unheard:
                # Never asked on its own: the window was already out when
                # the fetch began (its final lead posted, a late launch or
                # a past cycle), or a later lead was found posted first.
                # Either way it was up before it was asked, and the time
                # is the fetch's.  A lead let through with its host not
                # heard says nothing.  Measured live on gefs 2026-10-01T06
                # fetched as posted six hours late: each lead read 375 to
                # 377 min after its scheduled time and no flag said why.
                row["posted_when_first_asked"] = True
        marker = {
            "schema": MARKER_SCHEMA,
            "source": self.window.source,
            "member": self.window.member,
            "cycle": self.document["cycle"],
            "lead": lead,
            "valid_time": row.get("valid_time"),
            "objects": [dict(item) for item in objects],
            "expected_at": row.get("expected_at"),
            "late_at": row.get("late_at"),
            "first_seen_at": row.get("first_seen_at"),
            "posted_when_first_asked": row.get("posted_when_first_asked"),
            "fetched_at": rows.instant(fetched),
        }
        if composed:
            marker["composed"] = [dict(item) for item in composed]
        from woof.ingest.stream_resume import preserve_posted_marker
        marker = preserve_posted_marker(marker)
        path = self.folder / marker_name(lead)
        _write_json(path, marker)
        endpoint = next((item.get("endpoint") for item in objects
                         if item.get("endpoint")), row.get("endpoint"))
        self._mark(lead, state="ready", fetched_at=marker["fetched_at"],
                   endpoint=endpoint)
        size = sum(int(item.get("bytes") or 0) for item in objects)
        expected = readiness.parse_instant(row.get("expected_at"))
        seen = readiness.parse_instant(row.get("first_seen_at"))
        after = ("" if expected is None or seen is None else
                 f" ({(seen - expected).total_seconds() / 60.0:.1f} min after "
                 "its scheduled time"
                 + ("; it was already up when first asked"
                    if row.get("posted_when_first_asked") else "") + ")")
        self.progress(f"{self.label}: f{lead:03d} posted {row['first_seen_at']}"
                      f"{after}; {size / 1e6:.0f} MB verified")
        return path


def route_objects(entries: Sequence[Mapping[str, object]]) -> list[dict]:
    """A table route's fetched lead objects as its marker names them.

    Each object keeps its own ``lead`` from the plan: an object the route
    fetches with the window's first lead group but that belongs to no lead
    of the window (a cycle-invariant field, the step-0 statics of a window
    starting later) says so, so an as-posted preparation decodes it with
    every lead batch.
    """

    return [{"role": entry.get("role"),
             "name": entry.get("relpath", entry.get("name")),
             "url": entry.get("url"), "endpoint": entry.get("endpoint"),
             "bytes": entry.get("bytes"), "sha256": entry.get("sha256"),
             "lead": entry.get("lead")}
            for entry in entries]


def composed_objects(composed: Sequence[Mapping[str, object]]) -> list[dict]:
    """The files a route built from a lead's objects, as its marker names them.

    One GRIB per lead from its parts, which a preparation reads instead of
    the parts; each is named with its digest, so the seal holds the
    manifest row of a composed primary to its lead's marker.  They sit
    under the marker's own ``composed`` key, never among its ``objects``:
    ``objects`` are what was fetched and verified (DESIGN A136 2.3), which
    the chain stream's ``lead_ready`` bytes and the fetch's "MB verified"
    line sum, and a composed file is its parts' bytes again.
    """

    return [{"name": item.get("name"), "bytes": item.get("bytes"),
             "sha256": item.get("sha256"), "lead": item.get("lead"),
             "parts": list(item.get("parts") or ())}
            for item in composed]


def marker_files(marker: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Every file a lead marker names: its fetched objects, then its composed files."""

    return [item for key in ("objects", "composed")
            for item in (marker.get(key) or ()) if isinstance(item, Mapping)]


def legacy_objects(manifest: Mapping[str, object], lead: int) -> list[dict]:
    return [{"role": entry.get("role"), "name": entry.get("name"),
             "url": entry.get("url"),
             "endpoint": entry.get("endpoint", entry.get("transport")),
             "bytes": entry.get("bytes"), "sha256": entry.get("sha256")}
            for entry in manifest.get("files", [])
            if entry.get("forecast_hour") == lead]


def fetch_route_as_posted(loop: PostingLoop, run: Callable[..., dict]) -> dict:
    """A table route's window as it posts: gated transfers, markers in lead order."""

    loop.start()
    try:
        loop.wait_start()
        return run(lead_gate=loop,
                   on_lead=lambda lead, entries, composed: loop.publish(
                       lead, route_objects(entries),
                       composed=composed_objects(composed)))
    finally:
        loop.close()


#: What a legacy transport raises when a file did not arrive or verify:
#: a refusal in its own words (ValueError, RuntimeError) or the network's
#: (OSError).  Under the loop it is asked again only for a lead whose host
#: could not be heard or whose transfer reports an absent object; other
#: failures keep the transport's own retries and refusal.
LEGACY_TRANSFER_FAULTS = (ValueError, RuntimeError, OSError)


def fetch_legacy_as_posted(loop: PostingLoop,
                           fetch_prefix: Callable[[tuple[int, ...]], Path], *,
                           fresh: bool = False,
                           publish_lead=None) -> Path:
    """A legacy transport's window as it posts, one verified prefix at a time.

    The transport is called for the leads posted so far and reuses what
    it already verified, so its manifest grows with the prefix.  Leads
    already posted when the loop reaches them move in one call, side by
    side under the host caps; the markers still follow lead order.  A
    lead whose host could not be heard is let through to the transport,
    whose own checks decide; if that transfer fails it is asked again
    each round until the lead's late time, then the run stops with 75.
    ``fresh`` is a forced refetch: see :meth:`PostingLoop.start`.
    ``publish_lead`` lets an incremental transfer keep a marker already
    published before its batch returned, without rewriting its timestamps.
    """

    loop.start(fresh=fresh)
    try:
        return _legacy_prefixes(loop, fetch_prefix, publish_lead=publish_lead)
    finally:
        loop.close()


def _legacy_prefixes(loop: PostingLoop,
                     fetch_prefix: Callable[[tuple[int, ...]], Path], *,
                     publish_lead=None) -> Path:
    loop.wait_start()
    publish = loop.publish if publish_lead is None else publish_lead
    leads = loop.window.leads
    index = 0
    manifest_path = None
    while index < len(leads):
        loop(leads[index])
        through = index
        # Objects can reach a mirror out of lead order. Its final object
        # is no authority to download a missing intermediate lead.
        while through + 1 < len(leads) and loop.posted_now(
                leads[through + 1]):
            through += 1
        batch = leads[index:through + 1]
        try:
            manifest_path = fetch_prefix(tuple(leads[:through + 1]))
        except LEGACY_TRANSFER_FAULTS as error:
            # A host that refused an object as absent has not posted it
            # yet: wait for it until the lead's late time (then exit 75),
            # as for a lead whose host could not be heard.  Let wait_again
            # enforce the deadline even if the transfer crossed it.
            if not (loop.tried_unheard(batch)
                    or object_not_posted(error)):
                raise
            loop.wait_again(batch, error)
            continue
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        for lead in leads[index:through + 1]:
            publish(lead, legacy_objects(manifest, lead))
        index = through + 1
    return Path(manifest_path)


__all__ = ["LEGACY_TRANSFER_FAULTS", "MARKER_SCHEMA", "OBJECT_ABSENT_REASON",
           "PostingLoop", "object_not_posted",
           "SOURCE_BEHIND_EXIT",
           "SourceBehind", "composed_objects", "fetch_legacy_as_posted",
           "fetch_route_as_posted", "legacy_objects", "marker_files",
           "marker_name", "route_objects"]
