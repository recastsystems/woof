"""Bounded-concurrency transfer pool for multi-file acquisitions.

The scaling study (ARWEN-PRESIM-SCALING-2026-08-16) measured the cold
fetch stage at 0.47 s + 3.14 s **per file**, serial: file count dominates
and bytes barely matter, because each request spends most of its wall
clock waiting on the service, not on the pipe.  Sources that publish one
object per field per lead (ICON-EU: 251 objects per state; GEM: 351)
turn that per-file constant into the whole pre-sim budget.  The remedy
is to keep several requests in flight, bounded, without weakening a
single integrity property of the serial transport.

This module is the one shared implementation of that remedy.  It moves
no bytes itself: a :class:`TransferJob` carries an ``action`` callable
that downloads AND verifies one file exactly as the serial loop did
(envelope walk, record bar, sha256 -- whatever that route's bars are),
returning the manifest entry for it.  What the pool owns is scheduling
and bookkeeping:

* **Bounded workers.**  ``workers`` file transfers at most are in
  flight; :data:`DEFAULT_FILE_WORKERS` is the engine default and 1 is
  the serial transport, byte-for-byte the old loop (no threads are
  created at all).
* **Per-host politeness.**  A same-host cap bounds how many of those
  workers may target one host.  NOMADS is capped at
  :data:`NOMADS_FILE_WORKER_CAP` regardless of the pool size: the
  service is fragile, server-limited, and every request to it is
  *additionally* paced by the node-wide 2.5 s governor
  (:mod:`woof.nomads_governor`), which this pool never bypasses --
  concurrency there only overlaps in-flight service time, it never
  raises the request rate.  A host's files take its slots in
  submission order (:class:`_HostTurns`), so a capped host moves a
  window's leads first to last.
* **In-order admission.**  ``on_admitted`` fires on the caller's
  thread, in submission order, as the verified prefix grows -- so a
  route's per-file manifest publication keeps its exact serial
  semantics (an interrupted fetch still records a contiguous verified
  prefix, and receipts are never written from two threads).
* **Fail-closed per file, and at once.**  The first job to fail, in
  whatever order the jobs finish, fails the whole request: jobs not
  yet started are cancelled, jobs in flight are told to stop (a
  backbone subprocess is terminated, a Python transport gives up at
  its next chunk through :func:`raise_if_stopped` and in the middle of
  a retry or governor wait through :func:`sleep_unless_stopped`), and
  once they have wound down the original refusal -- whose message
  names the file -- is re-raised unchanged.  Files that landed stay on disk,
  unclaimed by any receipt past the admitted prefix, and the next run
  re-verifies them under the ordinary bars.
* **An accurate receipt.**  Files, bytes, workers requested and
  effective, the per-host caps that actually bound this run, wall
  seconds, the serial model (the sum of per-file seconds), and the
  effective speedup against it.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import contextlib
import functools
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
import sys
import threading
import time
from typing import Callable
from urllib.parse import urlsplit

from woof import fetch_endpoints, fetch_connections


@functools.lru_cache(maxsize=1)
def _width_table():
    table = json.loads(Path(__file__).with_name("fetch-width-rows.json").read_text(encoding="utf-8"))
    if table.get("schema") != "gpuwm-fetch-width-rows-v1":
        raise ValueError("fetch width table has an unknown schema; an unbounded transfer pool would exhaust connections")
    ids = set()
    for row in table["rows"]:
        if row["id"] in ids or any(type(row[key]) is not int or row[key] < 1
                                   for key in ("workers", "stream_budget")):
            raise ValueError("fetch width table has duplicate ids or nonpositive budgets; the transfer pool cannot be scheduled")
        ids.add(row["id"])
    if table["default"] not in ids:
        raise ValueError("fetch width table has no default row; the transfer pool cannot be scheduled")
    return table


def worker_profile() -> dict:
    """Select a packaged row using optional deployment/network hints.

    WOOF_FETCH_ROW pins a row. WOOF_FETCH_PROVIDER and WOOF_FETCH_BOX
    match rows, and WOOF_FETCH_BANDWIDTH_MBPS is megabytes per second.
    No metadata request or provider-specific adapter is involved.
    """
    table = _width_table()
    pin = os.environ.get("WOOF_FETCH_ROW")
    if pin:
        matches = [row for row in table["rows"] if row["id"] == pin]
        if not matches:
            raise ValueError(f"WOOF_FETCH_ROW {pin!r} is not in fetch-width-rows.json; the requested transfer budget is unknown")
        return dict(matches[0])
    provider = os.environ.get("WOOF_FETCH_PROVIDER", "").lower()
    box = os.environ.get("WOOF_FETCH_BOX", "").lower()
    raw = os.environ.get("WOOF_FETCH_BANDWIDTH_MBPS")
    bandwidth = None if raw is None else float(raw)
    if bandwidth is not None and (not math.isfinite(bandwidth) or bandwidth < 0):
        raise ValueError("WOOF_FETCH_BANDWIDTH_MBPS must be finite and nonnegative; an invalid link budget cannot size transfers")
    matches = []
    for row in table["rows"]:
        if row["provider"] not in ("*", provider) or row["box"] not in ("*", box):
            continue
        bounded = "bandwidth_min_mbps" in row
        if bounded and (bandwidth is None or bandwidth < row["bandwidth_min_mbps"]
                        or (row.get("bandwidth_max_mbps") is not None
                            and bandwidth >= row["bandwidth_max_mbps"])):
            continue
        # A measured link hint wins over an unmeasured provider fallback.
        score = (int(row["box"] != "*"), int(bounded), int(row["provider"] != "*"))
        matches.append((score, row))
    row = max(matches, key=lambda item: item[0])[1] if matches else next(
        row for row in table["rows"] if row["id"] == table["default"])
    return dict(row)

#: The engine default: how many file transfers are in flight at once.
#: Chosen against the measured fetch shape -- per-file service latency
#: dominates, so a handful of overlapped requests recovers most of the
#: idle pipe without presenting as a burst to any provider.
DEFAULT_FILE_WORKERS = next(row["workers"] for row in _width_table()["rows"]
                            if row["id"] == _width_table()["default"])

#: Hosts with a politeness cap tighter than the pool size, and the cap
#: each one gets.  TABLE DATA, read from ``host_policy`` in the packaged
#: acquisition authority: a provider that starts throttling is a row
#: there, not an edit here, and the same table already says which hosts
#: a source is asked for.  The cap on the operational NCEP server rides
#: on top of the node-wide request-spacing governor
#: (:mod:`woof.nomads_governor`), which this pool never bypasses --
#: concurrency there only overlaps in-flight service time, it never
#: raises the request rate.
HOST_FILE_WORKER_CAPS = dict(fetch_endpoints.host_caps())

#: The operational NCEP server's cap, named for the callers that print
#: it in help text.  Its reasoning lives with the row, in the table.
NOMADS_FILE_WORKER_CAP = HOST_FILE_WORKER_CAPS["nomads.ncep.noaa.gov"]

#: Schema of the receipt :func:`run_transfers` returns.
POOL_RECEIPT_SCHEMA = "gpuwm-fetch-pool-receipt-v1"

_COMPLETION_LOCAL = threading.local()


@contextlib.contextmanager
def completion_hook(callback):
    """Caller-scoped handoff hook, including routes with an ordered callback.

    This observes verified completions without changing their ordered
    admission contract. Nested calls restore their caller's hook.
    """
    prior = getattr(_COMPLETION_LOCAL, "callback", None)
    _COMPLETION_LOCAL.callback = callback
    try:
        yield
    finally:
        _COMPLETION_LOCAL.callback = prior

#: Streams are table-driven, then clipped to the box connection cap.
#: File workers are clipped too, including explicit --fetch-workers values.
FETCH_STREAM_BUDGET = next(row["stream_budget"] for row in _width_table()["rows"]
                            if row["id"] == _width_table()["default"])


def chunk_streams_per_file(workers: int | None, *, files: int,
                           host: str | None = None) -> int:
    """Chunk streams each in-flight file gets from :data:`FETCH_STREAM_BUDGET`.

    ``workers`` is the pool size the route asked for, ``files`` how many
    it has to move and ``host`` the host they come from, whose own cap
    may keep fewer in flight. File workers are clipped to this process's
    share of the box cap (:func:`woof.fetch_connections.process_share`),
    so other woof processes on the box still find slots; every admitted
    file gets at least one stream.
    """

    pool = resolve_file_workers(workers)
    in_flight = min(pool, max(1, files))
    if host:
        in_flight = min(in_flight, host_worker_cap(host, pool))
    budget = min(worker_profile()["stream_budget"], fetch_connections.process_share())
    if host and host.lower() == "nomads.ncep.noaa.gov":
        return 1
    return max(1, budget // max(1, in_flight))


@dataclass(frozen=True)
class TransferJob:
    """One file's transfer-and-verify unit of work.

    ``name`` is what a refusal names.  ``url`` exists for host
    politeness only (None means no host, so no cap); the ``action``
    already knows where its bytes come from.  ``action`` returns the
    route's manifest entry for the file and raises the route's own
    refusal on any failure -- the pool adds no bars and removes none.

    The last three are what a :class:`woof.progress.TransferMonitor`
    needs to SAY something about this file while it is moving, and they
    are all optional because none of them is a transport property:
    ``token`` is what to call this file to a person (``f01 atmosphere``),
    ``expected_bytes`` is its size where a route already knows one, and
    ``path`` is where it lands -- which lets the in-flight byte count be
    read off the growing file when the action's transport reports
    nothing back.

    ``on_disk`` says the file is already at ``path`` and the action
    checks it before anything moves.  The start line then says the file
    is being checked and names no host: announcing a host for a file
    that is only read here told a user whose fetch had just said it
    would ask no provider that it was asking one.  A route that fetches
    a file again when it fails the check keeps ``url``, so the job stays
    under that host's cap; one that refuses instead passes None.
    """

    name: str
    url: str | None
    action: Callable[[], dict]
    token: str | None = None
    expected_bytes: int | None = None
    path: object | None = None
    on_disk: bool = False


def host_key(url: str | None) -> str:
    """The politeness key for ``url``: its lower-cased netloc."""

    if not url:
        return ""
    return urlsplit(url).netloc.lower()


def host_worker_cap(host: str, workers: int) -> int:
    """How many of ``workers`` may target ``host`` at once."""

    if fetch_endpoints.aws_fetch_policy() and host.lower() == "nomads.ncep.noaa.gov":
        return min(1, workers)
    cap = HOST_FILE_WORKER_CAPS.get(host.lower())
    if cap is None:
        return workers
    return min(cap, workers)


def resolve_file_workers(requested: int | None) -> int:
    """The file-worker count a route runs with.

    ``None`` is the engine default.  1 is the serial transport -- a
    first-class knob, not a workaround.  Zero or a negative count names
    no schedulable pool and refuses.
    """

    if requested is None:
        return min(worker_profile()["workers"], fetch_connections.process_share())
    workers = int(requested)
    if workers < 1:
        raise ValueError(
            f"--fetch-workers {requested} is not a schedulable pool; pass "
            "a positive count (1 is the serial transport)")
    return min(workers, fetch_connections.process_share())


class TransferCancelled(RuntimeError):
    """This file's transfer was stopped because another file failed.

    Not a fault of this file or of its host, so nothing retries it and
    no endpoint ladder walks past it
    (:func:`woof.fetch_endpoints.fault_reason` has no reason for it),
    and the pool never raises it as the request's failure: the refusal
    that stopped it is the one raised.
    """


class _Stop:
    """One request's stop signal, and the subprocesses it ends.

    WHY THE POOL OWNS THIS.  A late failure used to surface only after
    every earlier file had landed: the pool read results in submission
    order and let in-flight transfers run on.  A 44-file HRRR fetch
    lost its last soil object 228 s in and then kept downloading about
    3 GB for another 190 s before it said so.  The pool is the one
    place that knows the request is already lost, so it is the one that
    stops the rest.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._processes: set = set()
        self._conditions: set[threading.Condition] = set()
        self.streams_used: dict[str, int] = {}
        self.connection_slots: dict[str, int] = {}
        #: Submission index of the job whose failure fired the stop, or
        #: None (not fired, or fired by an interrupt on the caller's
        #: thread).  The first to fire wins.
        self.origin: int | None = None

    @property
    def fired(self) -> bool:
        return self._event.is_set()

    def wait(self, seconds: float) -> bool:
        """Block ``seconds`` or until the stop fires; True once it has."""

        return self._event.wait(max(0.0, seconds))

    def fire(self, origin: int | None = None) -> None:
        """Stop every job: set the flag, then end each live subprocess."""

        with self._lock:
            if not self._event.is_set():
                self.origin = origin
            self._event.set()
            processes = list(self._processes)
            conditions = list(self._conditions)
        for condition in conditions:
            with condition:
                condition.notify_all()
        for process in processes:
            _terminate(process)

    def notify_when_stopped(self, condition: threading.Condition) -> None:
        """Wake a host-slot wait when this request fails or is interrupted."""

        with self._lock:
            self._conditions.add(condition)
            fired = self._event.is_set()
        if fired:
            with condition:
                condition.notify_all()

    @contextlib.contextmanager
    def adopt(self, process):
        """End ``process`` if the request is stopped while it runs."""

        with self._lock:
            self._processes.add(process)
            fired = self._event.is_set()
        try:
            if fired:
                # Stopped between the job's last check and its launch.
                _terminate(process)
            yield process
        finally:
            with self._lock:
                self._processes.discard(process)


def _terminate(process) -> None:
    try:
        process.terminate()
    except OSError:              # already gone
        pass


class _HostTurns:
    """One host's transfer slots, handed out in submission order.

    The breakage this prevents, measured live on 2026-10-01 (a GEFS window
    fetched as it posted: two objects per lead on NOMADS, cap 2, six
    workers): with a plain semaphore, the worker that had just freed a
    slot took the next queued file and got the slot back before a waiting
    worker woke, so files 2 to 5 (f003 and f006) moved only after f024.
    The forecast needs f000 and f003 to start, so it started once the
    whole window was in, and under the as-posted gate each barging file
    also held its slot while its lead had not posted yet.  Here a file
    takes a slot only when every earlier file of its host has taken one.

    Every earlier file has already been taken by a worker (the executor's
    queue is first in, first out), so its turn always comes, unless the
    request was stopped and it was cancelled unstarted; a waiter therefore
    gives up once the stop fires, and the pool cancels it.
    """

    def __init__(self, cap: int, indices) -> None:
        self._cond = threading.Condition()
        self._cap = self._free = int(cap)
        self._turns = list(indices)
        self._next = 0

    def acquire(self, index: int, stop: _Stop) -> bool:
        """Wait for ``index``'s turn and a free slot; False once stopped."""

        stop.notify_when_stopped(self._cond)
        with self._cond:
            while True:
                if stop.fired:
                    return False
                if (self._free > 0 and self._next < len(self._turns)
                        and self._turns[self._next] == index):
                    break
                self._cond.wait()
            self._next += 1
            self._free -= 1
            # The next file's turn may already have a free slot.
            self._cond.notify_all()
            return True

    def release(self) -> None:
        with self._cond:
            if self._free >= self._cap:
                raise ValueError("host transfer slot released more than acquired")
            self._free += 1
            self._cond.notify_all()

    def widen(self, cap: int) -> bool:
        """Raise this host's cap to ``cap`` (never lower it); True when it rose."""

        with self._cond:
            if cap <= self._cap:
                return False
            self._free += cap - self._cap
            self._cap = cap
            self._cond.notify_all()
            return True


class _Widening:
    """One host's ``transfer`` row applied to a request: when to widen it.

    Fed each landed file's bytes and seconds; once ``sample`` downloaded
    files of at least the row's ``sample_min_bytes`` are in and the last
    ``sample`` of them averaged below the row's
    ``wide_below_bytes_per_s``, the host's turns widen to ``cap`` once,
    for the rest of the request (see
    :class:`woof.fetch_endpoints.TransferWatch`).
    """

    def __init__(self, watch, turns: _HostTurns, cap: int) -> None:
        self.watch, self.turns, self.cap = watch, turns, cap
        self._lock = threading.Lock()
        self._landed: list[tuple[int, float]] = []
        self.widened: dict | None = None

    def landed(self, entry: dict, seconds: float) -> None:
        size = entry.get("bytes") if isinstance(entry, dict) else None
        if (self.widened is not None or not isinstance(size, int) or isinstance(size, bool)
                or size <= 0 or size < self.watch.sample_min_bytes or seconds <= 0
                or entry.get("reused") or entry.get("adopted")
                or entry.get("downloaded") is False):
            return
        with self._lock:
            if self.widened is not None:
                return
            self._landed.append((size, seconds))
            last = self._landed[-self.watch.sample:]
            if len(last) < self.watch.sample:
                return
            rate = sum(b for b, _ in last) / sum(s for _, s in last)
            if rate >= self.watch.wide_below_bytes_per_s or not self.turns.widen(self.cap):
                return
            self.widened = {"after_files": len(self._landed), "files_in_flight": self.cap,
                            "bytes_per_s": round(rate)}
        print(f"woof fetch: {self.watch.host} is serving {rate / 1e6:.2f} MB/s per file "
              f"over the last {len(last)} files (below {self.watch.wide_below_bytes_per_s / 1e6:.2f}); "
              f"moving up to {self.cap} of its files at once", file=sys.stderr, flush=True)


class TransferContext:
    """What a transport can reach of the pool job it is running under.

    Set on the worker thread around exactly one job and found with
    :func:`current_job`.  A transport that never looks keeps working
    unchanged; one that looks can hand the pool a subprocess to end, or
    give up at its next chunk, once another file has failed the request.
    """

    __slots__ = ("name", "_stop", "_shared", "byte_relay")

    def __init__(self, name: str, stop: _Stop, *, shared: bool = True) -> None:
        self.name = name
        self._stop = stop
        #: Other jobs of the same request can be running beside this one,
        #: so another file's failure can stop it.  False on the serial
        #: transport, where a failure ends the request by raising.
        self._shared = shared
        self.byte_relay = None

    @property
    def stopped(self) -> bool:
        return self._stop.fired

    def raise_if_stopped(self) -> None:
        if self._stop.fired:
            raise TransferCancelled(
                f"{self.name}: stopped because another file failed the "
                "request")

    def sleep(self, seconds: float,
              sleep: Callable[[float], None] | None = None) -> None:
        """Wait ``seconds``, cut short once another file fails the request.

        ``sleep`` is a clock the caller was given in place of the real
        one (a test's); it is kept, with the stop checked on both sides
        of it.  Without one, a job beside others waits on the stop
        signal itself, and a job alone (the serial transport, which
        nothing else can stop) sleeps.
        """

        self.raise_if_stopped()
        if sleep is not None:
            sleep(seconds)
        elif self._shared:
            self._stop.wait(seconds)
        else:
            time.sleep(seconds)
        self.raise_if_stopped()

    def adopt(self, process):
        """Terminate ``process`` if the request is stopped while it runs."""

        return self._stop.adopt(process)


_CURRENT = threading.local()

#: The real clock, told apart from a clock a test put in its place.
_REAL_SLEEP = time.sleep


def current_job() -> TransferContext | None:
    """The pool job this thread is running, or None outside the pool."""

    return getattr(_CURRENT, "job", None)


@contextlib.contextmanager
def working_for(job: TransferContext | None):
    """Carry the pool job ``job`` onto this thread for the block.

    For a transport that hands its work to threads of its own, which do
    not carry the pool job: take :func:`current_job` on the pool's
    thread and run each piece of work inside this.
    """

    previous = getattr(_CURRENT, "job", None)
    _CURRENT.job = job
    try:
        yield job
    finally:
        _CURRENT.job = previous


def raise_if_stopped(job: TransferContext | None = None) -> None:
    """Give up now if the request this work belongs to has already failed.

    For a transport's chunk loop and retry loop; ``job`` defaults to the
    one this thread works for (:func:`current_job`).  A no-op outside
    the pool and while the request is healthy.
    """

    job = current_job() if job is None else job
    if job is not None:
        job.raise_if_stopped()


def sleep_unless_stopped(seconds: float, *,
                         sleep: Callable[[float], None] | None = None,
                         job: TransferContext | None = None) -> None:
    """Sleep ``seconds``, or give up as soon as the request has failed.

    For a transport's wait before its next attempt, the NOMADS
    governor's wait before a request and the wait for an object to be
    published.  :func:`raise_if_stopped` only looks once a wait is over,
    so a stopped job used to sleep out its backoff, a ``Retry-After``, a
    node-wide cooldown or a 30 s publication poll first (and the poll
    never looked at all), and the refusal waited for it.  Here a job of
    a pool with other files in flight waits on the request's stop
    signal, and raises :class:`TransferCancelled` the moment it fires.

    ``sleep`` is the caller's clock.  The real :func:`time.sleep` (what
    a transport passes in production) is the stop-aware wait inside the
    pool and itself outside it; any other clock, a test's, is kept as it
    is, with the stop checked on both sides of it.  ``job`` defaults to
    :func:`current_job`.
    """

    job = current_job() if job is None else job
    injected = None if sleep is None or sleep is _REAL_SLEEP else sleep
    if job is None:
        (injected if injected is not None else time.sleep)(seconds)
        return
    job.sleep(seconds, injected)


def record_streams(streams: int, slots: int | None = None) -> None:
    """Record the transport limits actually selected for this file."""
    job = current_job()
    if job is not None:
        with job._stop._lock:
            job._stop.streams_used[job.name] = max(streams, job._stop.streams_used.get(job.name, 1))
            job._stop.connection_slots[job.name] = max(slots or streams, job._stop.connection_slots.get(job.name, 1))


def _receipt(*, entries: list[dict], workers_requested: int,
             workers_effective: int, host_caps: dict[str, int],
             wall_seconds: float, serial_seconds: float,
             streams_used: dict, connection_slots: dict) -> dict:
    total_bytes = sum(
        entry["bytes"] for entry in entries
        if isinstance(entry.get("bytes"), int)
        and not isinstance(entry.get("bytes"), bool))
    speedup = (round(serial_seconds / wall_seconds, 2)
               if wall_seconds > 0.0 else None)
    return {
        "schema": POOL_RECEIPT_SCHEMA,
        "files": len(entries),
        "bytes": total_bytes,
        "workers_requested": workers_requested,
        "workers_effective": workers_effective,
        "worker_row": worker_profile()["id"],
        "stream_budget": min(worker_profile()["stream_budget"], fetch_connections.max_connections()),
        "max_connections": fetch_connections.max_connections(),
        "streams_used": dict(streams_used),
        "connection_slots_per_file": dict(connection_slots),
        "host_caps": dict(host_caps),
        "wall_seconds": round(wall_seconds, 6),
        "modeled_serial_seconds": round(serial_seconds, 6),
        "effective_speedup": speedup,
    }


def _watch(monitor, job: TransferJob, stop: _Stop, index: int,
           on_start: Callable[[], None] | None = None, *,
           shared: bool = False):
    """Wrap ``job.action`` so the monitor hears when it starts and ends.

    AROUND the action, never inside it: what a route's transport does is
    the route's business, and the pool's promise that it "adds no bars
    and removes none" has to survive this.  The wrapper reports two
    facts the pool already owns -- this file began, this file ended --
    and re-raises whatever the action raised, unchanged.  It also makes
    the job reachable from the transport (:func:`current_job`), which is
    how a transport learns that the request has already failed.
    """

    def watched() -> dict:
        context = TransferContext(job.name, stop, shared=shared)
        context.byte_relay = (None if monitor is None else
                              lambda: monitor.relay(job.name))
        # Stopped while this job waited for a worker or a host slot: it
        # never started, so it is not announced either.
        context.raise_if_stopped()
        if on_start is not None:
            on_start()
        if monitor is not None:
            monitor.start(job.name, token=job.token,
                          host=None if job.on_disk else host_key(job.url) or None,
                          expected_bytes=job.expected_bytes, path=job.path,
                          on_disk=job.on_disk)
        started = time.perf_counter()
        _CURRENT.job = context
        try:
            entry = job.action()
        except BaseException as error:
            cancelled = isinstance(error, TransferCancelled)
            if not cancelled:
                # HERE, on the failing worker, before it goes back for
                # the next queued job: fired from the caller's thread a
                # moment later, the stop would reach that job only after
                # it had started.
                stop.fire(index)
            # A FAILED file is finished too.  Left open, it would be
            # counted as in flight for the rest of the request and the
            # consolidated line would claim work that stopped.
            if monitor is not None:
                monitor.finish(job.name,
                               seconds=time.perf_counter() - started,
                               failed=True, cancelled=cancelled)
            raise
        finally:
            _CURRENT.job = None
        if monitor is not None:
            size = entry.get("bytes") if isinstance(entry, dict) else None
            monitor.finish(job.name,
                           size=size if isinstance(size, int)
                           and not isinstance(size, bool) else None,
                           seconds=time.perf_counter() - started)
        return entry

    return watched


def _failure(future) -> BaseException | None:
    if not future.done() or future.cancelled():
        return None
    return future.exception()


def _raise_first_failure(futures, origin: int | None) -> None:
    """Re-raise the failure that ended the request, unchanged.

    The job that fired the stop, when it is known; otherwise the
    earliest submitted real failure, so a job that was only stopped
    because of it is never what the caller is told.
    """

    if origin is not None and _failure(futures[origin]) is not None:
        futures[origin].result()
    failures = [future for future in futures if _failure(future) is not None]
    for future in failures:
        if not isinstance(_failure(future), TransferCancelled):
            future.result()
    for future in failures:
        future.result()


def run_transfers(jobs, *, workers: int,
                  on_admitted: Callable[[int, dict], None] | None = None,
                  on_completed: Callable[[int, dict], None] | None = None,
                  monitor=None, widen: bool = True) -> tuple[list[dict], dict]:
    """Run every job; return ``(entries, receipt)`` in submission order.

    ``on_admitted(index, entry)`` fires on the caller's thread, in
    submission order, as each verified file joins the contiguous
    admitted prefix -- publish receipts there exactly as the serial
    loop did after each file.

    ``on_completed`` fires on the caller's thread as verified jobs finish,
    before ordered admission. It may bind supplements while earlier files
    still land; it must not publish a complete forecast or an ordered prefix.

    ``monitor`` is a :class:`woof.progress.TransferMonitor` that hears
    the whole request up front and then the start and the end of every
    job, so a request with six files in flight says which ones, and out
    of how many, as it goes instead of only when each one lands.  The
    transfer call sites always pass one; the ladder PROBE pass
    deliberately does not, because a HEAD is not a download and
    announcing eight of them would bury the eight lines that matter.

    Failure semantics: the FIRST job to fail, in completion order, ends
    the request.  Jobs not yet started are cancelled and never start,
    jobs in flight are told to stop (see :class:`_Stop`), and once they
    have wound down the contiguous verified prefix is admitted and the
    original exception is re-raised unchanged.  A ``KeyboardInterrupt``
    -- whether it lands here or inside an action -- stops the rest the
    same way and propagates, so route-level interrupt handling keeps
    working unchanged.

    ``widen`` lets a host whose ``host_policy`` row has a ``transfer``
    object move more of its files at once while it serves each one slowly
    (:class:`woof.fetch_endpoints.TransferWatch`); a caller that asked
    for an exact width passes False.  The receipt's ``host_widened`` says
    when that happened.
    """

    jobs = list(jobs)
    workers = resolve_file_workers(workers)
    workers_effective = min(workers, len(jobs)) if jobs else 0
    seen_hosts = {host_key(job.url) for job in jobs}
    host_caps = {
        host: host_worker_cap(host, workers)
        for host in sorted(seen_hosts)
        if host and host_worker_cap(host, workers) < workers_effective}
    entries: list[dict] = []
    serial_seconds = 0.0
    stop = _Stop()
    if monitor is not None and jobs:
        # Before any worker exists.  See TransferMonitor.begin: paying
        # the ticker's thread creation inside the first worker makes that
        # worker late relative to its siblings, which reorders the
        # transfers.  And the WHOLE request, so the first line already
        # says out of how many: a monitor that learned each file only as
        # it started printed "0 of 6", then "6 of 12", then "16 of 22".
        monitor.begin(files=[(job.name, job.expected_bytes)
                             for job in jobs])
    started = time.perf_counter()
    completion = getattr(_COMPLETION_LOCAL, "callback", None)

    def completed(index, entry):
        if completion is not None:
            completion(index, entry)
        if on_completed is not None:
            on_completed(index, entry)

    def admit(index: int, entry: dict) -> None:
        entries.append(entry)
        if on_admitted is not None:
            on_admitted(index, entry)

    if workers_effective <= 1:
        # The serial transport: the caller's thread, in order, no
        # thread machinery at all -- byte-for-byte the old loop.
        # A failure here is already immediate: nothing else is in flight.
        for index, job in enumerate(jobs):
            job_started = time.perf_counter()
            try:
                entry = _watch(monitor, job, stop, index)()
            except Exception:
                if monitor is not None:
                    monitor.stopped(job.name, in_flight=0,
                                    queued=len(jobs) - index - 1)
                raise
            serial_seconds += time.perf_counter() - job_started
            completed(index, entry)
            admit(index, entry)
        return entries, _receipt(
            entries=entries, workers_requested=workers,
            workers_effective=workers_effective, host_caps=host_caps,
            wall_seconds=time.perf_counter() - started,
            serial_seconds=serial_seconds, streams_used=stop.streams_used,
            connection_slots=stop.connection_slots)

    turns = {
        host: _HostTurns(host_worker_cap(host, workers),
                         [index for index, job in enumerate(jobs)
                          if host_key(job.url) == host])
        for host in seen_hosts if host}
    # A host whose table row widens it while it serves slowly gets the
    # threads for that width now; its turns stay at the pool's width
    # until the row's evidence is in (_Widening).
    widenings: dict[str, _Widening] = {}
    if widen:
        share = None
        for host in sorted(turns):
            watch = fetch_endpoints.transfer_watch(host)
            if watch is None:
                continue
            if share is None:
                share = fetch_connections.process_share()
            mine = sum(1 for job in jobs if host_key(job.url) == host)
            cap = min(watch.wide_files_in_flight, share, mine)
            if cap > host_worker_cap(host, workers):
                widenings[host] = _Widening(watch, turns[host], cap)
    threads = max([workers_effective] + [w.cap for w in widenings.values()])
    timing_lock = threading.Lock()

    announced: set[int] = set()

    def timed(index: int, job: TransferJob) -> dict:
        nonlocal serial_seconds
        gate = turns.get(host_key(job.url))
        if gate is not None and not gate.acquire(index, stop):
            # Stopped while it waited for its turn: it never started, so
            # it is cancelled unannounced, as a file the stop reaches
            # after it got a slot is (TransferContext.raise_if_stopped).
            raise TransferCancelled(
                f"{job.name}: stopped because another file failed the "
                "request")
        try:
            job_started = time.perf_counter()
            entry = _watch(monitor, job, stop, index,
                           on_start=lambda: announced.add(index),
                           shared=True)()
            elapsed = time.perf_counter() - job_started
        finally:
            if gate is not None:
                gate.release()
        with timing_lock:
            serial_seconds += elapsed
        widening = widenings.get(host_key(job.url))
        if widening is not None:
            widening.landed(entry, elapsed)
        return entry

    executor = ThreadPoolExecutor(
        max_workers=threads,
        thread_name_prefix="gpuwm-fetch")
    futures: list = []
    admitted = 0

    def admit_prefix() -> None:
        # In submission order: the admitted prefix is contiguous by
        # construction, whatever order the files finish in.
        nonlocal admitted
        while admitted < len(futures):
            future = futures[admitted]
            if (not future.done() or future.cancelled()
                    or _failure(future) is not None):
                return
            admit(admitted, future.result())
            admitted += 1

    def cut_short(index: int) -> bool:
        # Still moving, or already wound down by the stop itself: the
        # failing worker ends the backbones before its own result is
        # settled, so some of them can be done by the time this looks.
        future = futures[index]
        if not future.done():
            return True
        return (not future.cancelled()
                and isinstance(future.exception(), TransferCancelled))

    def halt(origin: int | None) -> tuple[int, int]:
        """Stop the request: (files cut short in flight, never started)."""

        stop.fire(origin)
        for future in futures:
            future.cancel()
        begun = set(announced)
        in_flight = sum(1 for index in begun
                        if index != origin and cut_short(index))
        return in_flight, len(jobs) - len(begun)

    try:
        futures = [executor.submit(timed, index, job)
                   for index, job in enumerate(jobs)]
        pending = set(futures)
        while pending and not stop.fired:
            _finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            if on_completed is not None or completion is not None:
                for future in _finished:
                    if not future.cancelled() and _failure(future) is None:
                        completed(futures.index(future), future.result())
            admit_prefix()
        if stop.fired:
            # The failing worker fired the stop before its own future
            # was settled, so its index is known here even when another
            # job's cancellation settled first.
            origin = stop.origin
            in_flight, queued = halt(origin)
            if monitor is not None and origin is not None:
                monitor.stopped(jobs[origin].name,
                                in_flight=in_flight, queued=queued)
            # The files still moving have been told to stop, not waited
            # out.  What is waited for is only their winding down, so a
            # file a worker is still writing never races the caller's
            # own disk scan (the interrupt path reads the directory).
            wait(futures)
            admit_prefix()
            _raise_first_failure(futures, origin)
    except BaseException:
        halt(None)
        raise
    finally:
        executor.shutdown(wait=True)
    receipt = _receipt(
        entries=entries, workers_requested=workers,
        workers_effective=workers_effective, host_caps=host_caps,
        wall_seconds=time.perf_counter() - started,
        serial_seconds=serial_seconds, streams_used=stop.streams_used,
        connection_slots=stop.connection_slots)
    widened = {host: w.widened for host, w in widenings.items() if w.widened}
    if widened:
        receipt["host_widened"] = widened
    return entries, receipt


__all__ = [
    "DEFAULT_FILE_WORKERS", "FETCH_STREAM_BUDGET", "HOST_FILE_WORKER_CAPS",
    "NOMADS_FILE_WORKER_CAP", "POOL_RECEIPT_SCHEMA", "TransferCancelled",
    "TransferContext", "TransferJob", "chunk_streams_per_file",
    "current_job", "host_key", "host_worker_cap", "raise_if_stopped",
    "resolve_file_workers", "run_transfers", "sleep_unless_stopped",
    "working_for",
]
