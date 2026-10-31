"""The cross-process NOMADS rate governor, spoken by the Python transport.

NOMADS is a shared public service behind Akamai, and it answers a caller
who asks too fast with an over-rate-limit page and, eventually, an IP
block.  The vendored Rust fetch backbone has always paced itself against
a *node-wide* state file, so any number of concurrent ``rw_fetch``
processes on one machine still space their NOMADS requests.

The Python transports did not.  ``woof fetch --engine python`` (and
``--engine auto`` on a machine where the backbone is not built) went
straight to :mod:`urllib`, and the HRRR range downloader ran eight
worker threads doing it.  A user scripting a handful of parallel fetches
could therefore be rate-limited or banned while believing a governor was
in force -- and a Rust process and a Python process fetching side by
side paced only half the traffic.

This module is the same governor, in Python, over the *same files*:

* the same state path (``RUSTWX_NOMADS_RATE_STATE``, else
  ``<temp>/rustwx_nomads_rate_limit.state``);
* the same ``key=value`` body (``last_request_ms``, ``cooldown_until_ms``);
* the same sentinel lock beside it, the same 120 s staleness rule, the
  same dead-PID recovery where the platform can prove death;
* the same defaults -- 2.5 s between requests, a 15 minute cooldown once
  an over-rate-limit answer is seen -- and the same
  ``RUSTWX_NOMADS_MIN_INTERVAL_MS`` / ``RUSTWX_NOMADS_COOLDOWN_MS``
  overrides.

So a Rust fetch and a Python fetch on one node now share one pacer
rather than running two independent ones.

Two deliberate differences, both **stricter** than the Rust side, never
looser -- the governor's job is to protect a public service, so every
ambiguity here resolves towards waiting:

1. A state file that exists but does not parse is treated as "a request
   just happened", not as an empty state.  Corruption must not become
   permission to send.
2. When the shared state cannot be written, the request still waits out
   the minimum gap locally before proceeding.  Failing to record a
   request must not let the next process send immediately either.

Only ``nomads.ncep.noaa.gov`` is paced.  The S3 mirrors WOOF prefers
for HRRR are not rate limited in this way and are left alone, exactly as
the Rust client leaves them alone.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

#: The host the governor exists for.  Same test as the Rust client's
#: ``is_nomads_url``: a substring match on the production hostname.
NOMADS_HOST = "nomads.ncep.noaa.gov"

STATE_PATH_ENV = "RUSTWX_NOMADS_RATE_STATE"
MIN_INTERVAL_ENV = "RUSTWX_NOMADS_MIN_INTERVAL_MS"
COOLDOWN_ENV = "RUSTWX_NOMADS_COOLDOWN_MS"
REQUEST_LOG_ENV = "RUSTWX_NOMADS_REQUEST_LOG"

#: Defaults, byte-for-byte the Rust client's constants.
DEFAULT_MIN_INTERVAL_MS = 2500
DEFAULT_COOLDOWN_MS = 15 * 60 * 1000
LOCK_STALE_AFTER_MS = 120 * 1000

_LOCK_POLL_S = 0.05

#: HTTP statuses that mean "you are being rate limited" rather than
#: "that object is not there".
_RATE_LIMIT_STATUSES = frozenset({429, 503})


def is_nomads_url(url: str) -> bool:
    return NOMADS_HOST in url


def state_path() -> Path:
    override = os.environ.get(STATE_PATH_ENV)
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "rustwx_nomads_rate_limit.state"


def _lock_path(state: Path) -> Path:
    return state.with_suffix(".lock")


def _tmp_path(state: Path) -> Path:
    return state.with_suffix(".tmp")


def _env_ms(name: str, fallback: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return fallback
    try:
        value = int(raw)
    except ValueError:
        return fallback
    return value if value > 0 else fallback


def min_interval_ms() -> int:
    return _env_ms(MIN_INTERVAL_ENV, DEFAULT_MIN_INTERVAL_MS)


def cooldown_ms() -> int:
    return _env_ms(COOLDOWN_ENV, DEFAULT_COOLDOWN_MS)


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _read_state_detail(path: Path) -> tuple[int, int, bool]:
    """``(last_request_ms, cooldown_until_ms, sound)``.

    ``sound`` is False when the file exists but carries no usable
    ``last_request_ms`` -- the case wx-core maps to zero timestamps and
    therefore to permission to send immediately.
    """

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return 0, 0, True          # nobody has fetched yet: a real zero
    except OSError:
        return _now_ms(), 0, False
    last_request_ms: int | None = None
    cooldown_until_ms = 0
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        try:
            parsed = int(value.strip())
        except ValueError:
            continue
        if parsed < 0:
            continue
        if key.strip() == "last_request_ms":
            last_request_ms = parsed
        elif key.strip() == "cooldown_until_ms":
            cooldown_until_ms = parsed
    if last_request_ms is None:
        return _now_ms(), cooldown_until_ms, False
    return last_request_ms, cooldown_until_ms, True


def read_state(path: Path) -> tuple[int, int]:
    """``(last_request_ms, cooldown_until_ms)`` from the shared state.

    An absent file is a genuine empty state -- nobody has fetched yet --
    and yields zeros.  A file that exists but carries no parseable
    ``last_request_ms`` is corruption, and corruption is answered by
    pretending a request just happened, so the caller waits a full gap
    instead of being waved through.
    """

    last_request_ms, cooldown_until_ms, _sound = _read_state_detail(path)
    return last_request_ms, cooldown_until_ms


def write_state(path: Path, last_request_ms: int,
                cooldown_until_ms: int) -> bool:
    """Publish the shared state; False when the bytes did not land."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = _tmp_path(path)
        tmp.write_text(
            f"last_request_ms={last_request_ms}\n"
            f"cooldown_until_ms={cooldown_until_ms}\n",
            encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    except OSError:
        return False
    return True


def _lock_is_stale(lock: Path) -> bool:
    try:
        age_ms = _now_ms() - int(lock.stat().st_mtime * 1000)
    except OSError:
        return False
    if age_ms > LOCK_STALE_AFTER_MS:
        return True
    if os.name != "nt":
        try:
            recorded = lock.read_text(encoding="utf-8").split()
        except OSError:
            return False
        if recorded and recorded[0].isdigit():
            if not Path("/proc").joinpath(recorded[0]).exists():
                return True
    return False


class _Sentinel:
    """The cooperative lock the Rust client uses, with its semantics."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._held = False

    def __enter__(self) -> "_Sentinel":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        if self._held:
            self._held = False
            try:
                self.path.unlink()
            except OSError:
                pass


def _acquire(lock: Path, *, sleeper=time.sleep) -> _Sentinel | None:
    sentinel = _Sentinel(lock)
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    while True:
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _lock_is_stale(lock):
                try:
                    lock.unlink()
                except OSError:
                    pass
                continue
            sleeper(_LOCK_POLL_S)
            continue
        except OSError:
            return None
        try:
            os.write(handle, f"{os.getpid()} {_now_ms()}\n".encode("ascii"))
        finally:
            os.close(handle)
        sentinel._held = True
        return sentinel


def log_event(url: str, kind: str, status: str,
              elapsed_ms: int | None = None) -> None:
    """Append one JSON line to the shared request log, when enabled.

    Same shape and same env var as the Rust client's log, so a mixed
    Rust/Python run produces one interleaved record of what this node
    actually sent.
    """

    path = os.environ.get(REQUEST_LOG_ENV)
    if not path:
        return
    escaped = url.replace("\\", "\\\\").replace('"', '\\"')
    line = (
        '{"ts_ms":%d,"pid":%d,"kind":"%s","status":"%s","elapsed_ms":%s,'
        '"url":"%s"}\n' % (
            _now_ms(), os.getpid(), kind, status.replace('"', "'"),
            "null" if elapsed_ms is None else str(elapsed_ms), escaped))
    try:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(line)
    except OSError:
        pass


class PaceBudgetExceeded(Exception):
    """The request was NOT sent: its turn under the governor is further away than the caller will wait.

    Nothing was recorded, so the node-wide spacing is exactly as it was.
    A page asking whether a file exists would rather say "not checked"
    than queue behind a download for minutes; a transfer never passes a
    budget and waits as long as the governor says.
    """


class _Budget:
    def __init__(self, sleeper, max_wait_s: float) -> None:
        self.sleeper = sleeper
        self.left = max_wait_s

    def __call__(self, seconds: float) -> None:
        if seconds > self.left:
            raise PaceBudgetExceeded(f"the NOMADS governor asks for {seconds:.1f} s more")
        self.left -= seconds
        self.sleeper(seconds)


def _wait(seconds: float) -> None:
    """The governor's own wait: :func:`time.sleep`, unless a pooled
    download this request belongs to has already failed.

    Then it gives up at once (:class:`woof.fetch_pool.TransferCancelled`)
    instead of serving out a gap or a 15 minute cooldown for a request
    whose answer nobody will use.  Giving up sends nothing, so this is
    never looser than the plain wait.
    """

    from woof.fetch_pool import sleep_unless_stopped  # local: import order
    sleep_unless_stopped(seconds)


def pace(url: str, *, sleeper=None, max_wait_s: float | None = None) -> None:
    """Block until this process may send ``url``.

    A no-op for every host but NOMADS.  For NOMADS it takes the shared
    sentinel, reads the node-wide state, waits out whichever is later --
    the cooldown or the minimum gap since the last request anyone made
    -- and then records this request before returning.  ``sleeper``
    defaults to :func:`_wait`.

    ``max_wait_s`` bounds the total wait: when the governor would make
    this caller wait longer, :class:`PaceBudgetExceeded` is raised
    before anything is sent or recorded.  The governor is never looser
    for it: a request that goes out still went through the same gap.
    """

    if not is_nomads_url(url):
        return
    if sleeper is None:
        sleeper = _wait
    if max_wait_s is not None:
        sleeper = _Budget(sleeper, max_wait_s)
    gap_ms = min_interval_ms()
    state = state_path()
    lock = _lock_path(state)
    # A corrupt state buys exactly one full gap, then this process
    # republishes a sound one.  Charging it every round would spin
    # forever against a file nobody can parse -- refusing to send is not
    # the same as refusing to finish.
    corruption_paid = False
    while True:
        sentinel = _acquire(lock, sleeper=sleeper)
        if sentinel is None:
            # The lock is unusable (permissions, read-only temp).  Fail
            # closed: space the request locally rather than send now.
            sleeper(gap_ms / 1000.0)
            return
        try:
            last_request_ms, cooldown_until_ms, sound = \
                _read_state_detail(state)
            if not sound:
                if not corruption_paid:
                    sentinel.release()
                    sleeper(gap_ms / 1000.0)
                    corruption_paid = True
                    continue
                last_request_ms = 0   # the gap has been served in full
            now = _now_ms()
            sleep_until = max(cooldown_until_ms, last_request_ms + gap_ms)
            if sleep_until > now:
                sentinel.release()
                sleeper((sleep_until - now) / 1000.0)
                continue
            if not write_state(state, now, cooldown_until_ms):
                # The record did not land, so the next process will not
                # see this request.  Absorb the gap here instead.
                sentinel.release()
                sleeper(gap_ms / 1000.0)
            return
        finally:
            sentinel.release()


def retry_after_seconds(error: BaseException) -> float | None:
    """The server's own ``Retry-After`` ask, in seconds, or ``None``.

    Read off an :class:`HTTPError` (whose ``headers`` carry the
    answer's header block).  Both spellings the header allows are
    parsed: delay-seconds and an HTTP-date.  Junk, absence, and asks
    that are already in the past all yield ``None`` -- an instruction
    that cannot be read is not an instruction.
    """

    headers = getattr(error, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except AttributeError:
        return None
    if raw is None:
        return None
    raw = str(raw).strip()
    try:
        seconds = float(raw)
    except ValueError:
        import email.utils
        try:
            when = email.utils.parsedate_to_datetime(raw) if raw else None
        except ValueError:
            return None
        if when is None:
            return None
        seconds = when.timestamp() - time.time()
    return seconds if seconds > 0.0 else None


def mark_rate_limited(url: str, reason: str,
                      retry_after_s: float | None = None) -> None:
    """Record a node-wide cooldown after an over-rate-limit answer.

    ``retry_after_s`` is the server's own ``Retry-After`` instruction,
    when it sent one.  It can only EXTEND the cooldown beyond this
    node's configured one, never shorten it: the state file is shared
    with the Rust client, and honoring a header must not make the
    node-wide governor looser than either twin's own configuration.
    """

    if not is_nomads_url(url):
        return
    cooldown = cooldown_ms()
    if retry_after_s is not None and retry_after_s > 0.0:
        cooldown = max(cooldown, int(retry_after_s * 1000.0))
    state = state_path()
    lock = _lock_path(state)
    sentinel = _acquire(lock)
    try:
        last_request_ms, existing = read_state(state)
        now = _now_ms()
        if existing > now:
            log_event(url, "cooldown_existing", reason)
            return
        write_state(state, last_request_ms, now + cooldown)
        log_event(url, "cooldown", reason)
    finally:
        if sentinel is not None:
            sentinel.release()


def _is_rate_limit(error: BaseException) -> bool:
    if isinstance(error, HTTPError):
        return error.code in _RATE_LIMIT_STATUSES
    if isinstance(error, URLError):
        # Akamai's over-rate-limit answer on NOMADS is a redirect with no
        # Location header; urllib surfaces that as a URLError naming it.
        return "location" in str(error.reason).lower()
    return False


def paced_urlopen(request: Request | str, *, timeout: float | None = None,
                  opener=urlopen, sleeper=None, max_wait_s: float | None = None):
    """:func:`urllib.request.urlopen` under the governor.

    Paces the request when it is bound for NOMADS, logs it when the
    shared request log is enabled, and marks the node-wide cooldown when
    the answer is an over-rate-limit one.  Non-NOMADS URLs pass straight
    through, unpaced and unlogged, exactly as in the Rust client.
    ``max_wait_s`` is :func:`pace`'s: past it, nothing is sent and
    :class:`PaceBudgetExceeded` is raised.
    """

    url = request if isinstance(request, str) else request.full_url
    started = _now_ms()
    kind = "get"
    if isinstance(request, Request) and request.get_method() == "HEAD":
        kind = "head"
    elif isinstance(request, Request) and request.has_header("Range"):
        kind = "get_range"
    from woof.fetch_connections import lease, LeasedResponse
    from woof.fetch_pool import record_streams
    record_streams(1)
    reservation = lease()
    reservation.__enter__()
    try:
        # Wait for a box slot BEFORE taking a host's pacing turn. A wait
        # afterward could bunch several spaced requests at slot release.
        pace(url, sleeper=sleeper, max_wait_s=max_wait_s)
        started = _now_ms()
        response = (opener(request) if timeout is None
                    else opener(request, timeout=timeout))
    except BaseException as error:
        # HTTPError owns a response socket too. Close it before another
        # worker can reuse this slot, including failed range requests.
        try:
            if isinstance(error, HTTPError):
                if error.code in (429, 503):
                    # Read the answer's error code (S3's <Code>SlowDown</Code>)
                    # BEFORE closing: it is cached on the error, and a host's
                    # throttle row is matched on it.  Named breakage: closing
                    # first left the body unreadable, so ECMWF's AWS mirror's
                    # 503 SlowDown never matched its 600 s throttle row and an
                    # IFS 0.1-degree fetch gave up after 5 rounds in 30 s
                    # (WOOF run_19793057ff2979e3, exit 75, 2026-10-29).
                    from woof.fetch_endpoints import answer_code
                    answer_code(error)
                error.close()
        finally:
            reservation.__exit__(None, None, None)
        if _is_rate_limit(error):
            mark_rate_limited(url, type(error).__name__,
                              retry_after_s=retry_after_seconds(error))
        log_event(url, kind, f"error:{type(error).__name__}",
                  _now_ms() - started)
        raise
    log_event(url, kind, str(getattr(response, "status", "ok")),
              _now_ms() - started)
    return LeasedResponse(response, reservation)


def governed_workers(url: str, requested: int) -> int:
    """How many transfer threads may target ``url``.

    A NOMADS transfer is serialized by the governor anyway -- every
    worker would queue on the same sentinel for the same 2.5 s gap -- so
    running eight of them buys nothing and makes the burst shape depend
    on thread scheduling.  One worker states the governed behaviour
    plainly.  Other hosts keep the pool they asked for.
    """

    return 1 if is_nomads_url(url) else requested


__all__ = [
    "COOLDOWN_ENV",
    "DEFAULT_COOLDOWN_MS",
    "DEFAULT_MIN_INTERVAL_MS",
    "LOCK_STALE_AFTER_MS",
    "MIN_INTERVAL_ENV",
    "NOMADS_HOST",
    "PaceBudgetExceeded",
    "REQUEST_LOG_ENV",
    "STATE_PATH_ENV",
    "cooldown_ms",
    "governed_workers",
    "is_nomads_url",
    "log_event",
    "mark_rate_limited",
    "min_interval_ms",
    "pace",
    "paced_urlopen",
    "retry_after_seconds",
    "read_state",
    "state_path",
    "write_state",
]
