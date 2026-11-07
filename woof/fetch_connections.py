"""Connection leases shared by engine fetch processes on one box.

Locks last for the response body (or the entire Rust subprocess), not
just the request headers. Kernel locks are released on process death.
Use the same cap and state directory for all engine processes on a box.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
import tempfile
import time

DEFAULT_MAX_CONNECTIONS = 30
MAX_CONNECTIONS_ENV = "WOOF_FETCH_MAX_CONNECTIONS"
DIRECTORY_ENV = "WOOF_FETCH_CONNECTION_DIR"


def max_connections(value=None) -> int:
    raw = os.environ.get(MAX_CONNECTIONS_ENV, DEFAULT_MAX_CONNECTIONS) if value is None else value
    try:
        count = int(raw)
    except (TypeError, ValueError):
        count = 0
    if isinstance(raw, bool) or str(count) != str(raw) or count < 1:
        raise ValueError(f"fetch max connections {raw!r} must be a positive integer; an invalid cap cannot bound outbound sockets")
    return count


def _try_lock(handle) -> bool:
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
    else:
        import fcntl
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True



def _open_slot(path):
    """Open one box slot so every user of the machine can open it after us (mode 0666)."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o666)
    try:
        os.chmod(path, 0o666)
    except OSError:
        pass  # not ours: whoever created it already opened it to everyone
    return os.fdopen(fd, "a+b")


def _usable_slot_directory(directory):
    """The box-wide slot directory, shared by every user; per-user when another user's copy is closed to us.

    The slots live under /tmp so concurrent jobs contend for one box cap.  The first process to create the
    directory opens it to everyone (sticky, world-writable); a directory another user left closed (an older
    install, a stricter umask) refused every fetch on a shared machine with "[Errno 13] Permission denied:
    .../slot-0" (2.9.3 release battery on a host where root had created it), so this process then keeps its
    own per-user slots instead of failing.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(directory, 0o1777)
        except OSError:
            pass
        probe = directory / "slot-0"
        _open_slot(probe).close()
        return directory
    except OSError:
        uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
        own = directory.with_name(f"{directory.name}-{uid}")
        own.mkdir(parents=True, exist_ok=True)
        return own

#: Seconds a waiter stays quiet before it says what it is waiting on, then how often it repeats.
WAIT_NOTICE_AFTER_S = 5.0
WAIT_NOTICE_EVERY_S = 60.0
#: How long a waiter may find no free slot at all before it refuses (and names the holders).
SLOT_WAIT_ENV = "WOOF_FETCH_SLOT_WAIT_S"
DEFAULT_SLOT_WAIT_S = 1800.0
#: One process sizing its own pool leaves cap // RESERVE_FRACTION slots for the other woof processes on
#: the box, so a newcomer finds slots even while this process's slowest transfer runs.
RESERVE_FRACTION = 5


def _slot_directory() -> Path:
    # TMPDIR often differs between concurrent jobs. Unix processes must
    # still contend for the same box slots.
    shared_temp = Path(tempfile.gettempdir()) if os.name == "nt" else Path("/tmp")
    directory = Path(os.environ.get(DIRECTORY_ENV) or
                     (shared_temp / "gpuwm-fetch-connections"))
    return _usable_slot_directory(directory)


def _slot_wait_limit() -> float:
    raw = os.environ.get(SLOT_WAIT_ENV)
    if raw is None:
        return DEFAULT_SLOT_WAIT_S
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not value > 0:
        raise ValueError(f"{SLOT_WAIT_ENV}={raw!r} must be a positive number of seconds; "
                         "an invalid limit cannot bound the wait for a fetch connection slot")
    return value


def free_slots() -> int:
    """How many of the box's slots no process holds right now (a probe; each lock is let go at once)."""
    directory = _slot_directory()
    free = 0
    for index in range(max_connections()):
        handle = _open_slot(directory / f"slot-{index}")
        try:
            if _try_lock(handle):
                free += 1
        finally:
            handle.close()
    return free


def process_share() -> int:
    """The slots one process sizes its file pool and chunk streams from.

    Sized from the slots free now, less a fifth of the box cap left for the other woof processes on the
    box.  Sizing from the whole cap let one ``woof fetch`` take all 30 slots for the life of its slowest
    transfer (test host, 2026-10-20: three 10-socket AWS files at ~35 KB/s held them 19-28 min) while every
    other woof process on the box waited.  At least 1.
    """
    cap = max_connections()
    return max(1, min(free_slots(), cap) - cap // RESERVE_FRACTION)


def slot_holders(directory: Path | None = None) -> list[str]:
    """``pid holds N (cmdline)`` for every process holding a slot, from /proc/locks (empty where there is none)."""
    directory = _slot_directory() if directory is None else directory
    keys = set()
    try:
        for index in range(max_connections()):
            info = (directory / f"slot-{index}").stat()
            keys.add((os.major(info.st_dev), os.minor(info.st_dev), info.st_ino))
        lines = Path("/proc/locks").read_text().splitlines()
    except (OSError, AttributeError, ValueError):
        return []
    counts: dict[str, int] = {}
    for line in lines:
        fields = line.split()
        if len(fields) < 6 or fields[1] == "->":
            continue
        try:
            major, minor, inode = fields[5].split(":")
            key = (int(major, 16), int(minor, 16), int(inode))
        except ValueError:
            continue
        if key in keys:
            counts[fields[4]] = counts.get(fields[4], 0) + 1
    holders = []
    for pid, held in sorted(counts.items(), key=lambda item: -item[1]):
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            command = "?"
        if len(command) > 200:
            command = command[:197] + "..."
        holders.append(f"pid {pid} holds {held} ({command})")
    return holders


def _say(text: str) -> None:
    import sys
    print(f"woof fetch: {text}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def lease(count=1, *, minimum=1):
    """Hold between ``minimum`` and ``count`` box slots; yields how many were granted.

    The grant is whatever is free, up to ``count`` and at least ``minimum``, so a waiter no longer sits
    behind a bundle sized against the whole box cap while slots are free.  A waiter that finds fewer than
    ``minimum`` free says so on stderr after a few seconds, naming the processes that hold the slots, and
    refuses after ``WOOF_FETCH_SLOT_WAIT_S`` (default 1800 s) instead of hanging silently (test host
    2026-10-20: ``woof setup --with-geog`` sat with no output for over 10 minutes behind another woof
    process's slots).
    """
    from woof import fetch_pool
    cap = max_connections()
    if not 1 <= count <= cap:
        raise ValueError(f"{count} connections exceed the box fetch cap {cap}")
    minimum = max(1, min(int(minimum), count))
    directory = _slot_directory()
    limit = _slot_wait_limit()
    handles = []
    began = time.monotonic()
    next_notice = began + WAIT_NOTICE_AFTER_S
    try:
        while True:
            fetch_pool.raise_if_stopped()
            for index in range(cap):
                handle = _open_slot(directory / f"slot-{index}")
                if _try_lock(handle):
                    handles.append(handle)
                    if len(handles) == count:
                        break
                else:
                    handle.close()
            if len(handles) >= minimum:
                break
            for handle in handles:
                handle.close()
            handles.clear()
            now = time.monotonic()
            if now - began >= limit:
                holders = slot_holders(directory) or ["holders unknown (no /proc/locks)"]
                raise RuntimeError(
                    f"REFUSED: no fetch connection slot came free in {limit:.0f} s ({minimum} of {cap} "
                    f"needed in {directory}); held by: {'; '.join(holders)}. Wait for those transfers, stop "
                    f"them, or raise {SLOT_WAIT_ENV} / {MAX_CONNECTIONS_ENV}")
            if now >= next_notice:
                holders = slot_holders(directory) or ["holders unknown (no /proc/locks)"]
                _say(f"waiting {now - began:.0f} s for {minimum} of {cap} box fetch connection slots "
                     f"({directory}); held by: {'; '.join(holders)}; refusing after {limit:.0f} s")
                next_notice = now + WAIT_NOTICE_EVERY_S
            fetch_pool.sleep_unless_stopped(0.05)
        waited = time.monotonic() - began
        if waited >= WAIT_NOTICE_AFTER_S:
            _say(f"got {len(handles)} fetch connection slot(s) after {waited:.0f} s")
        yield len(handles)
    finally:
        for handle in handles:
            handle.close()


class LeasedResponse:
    """A urllib response whose socket slot is held until it closes."""
    def __init__(self, response, reservation):
        self._response = response
        self._reservation = reservation

    def __getattr__(self, name):
        return getattr(self._response, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            close = getattr(self._response, "close", None)
            if close is not None:
                close()
        finally:
            if self._reservation is not None:
                reservation, self._reservation = self._reservation, None
                reservation.__exit__(None, None, None)

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()
