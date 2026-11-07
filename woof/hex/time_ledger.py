"""Best-effort C1 events, using the shared emitter without creating a run early."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar

_member = ContextVar("hex_ledger_member", default=None)

#: How old the list of this process tree's pids may be before a stage boundary walks /proc for it again.
TREE_REFRESH_SECONDS = 2.0
_tree_lock = threading.Lock()
_tree = {"pids": None, "at": 0.0, "root": None}
try:
    _TICK = float(os.sysconf("SC_CLK_TCK"))
except (AttributeError, ValueError, OSError):
    _TICK = 100.0


def _own_and_reaped_ticks(pid):
    """utime + stime + cutime + cstime of one pid (``woof.timeline.parallelism._stat``'s sum), or None when gone."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    fields = raw[raw.rfind(b")") + 2:].split()
    return int(fields[11]) + int(fields[12]) + int(fields[13]) + int(fields[14])


def tree_cpu_seconds(*, clock=time.monotonic):
    """This process tree's CPU seconds, counted as ``woof.timeline.parallelism.tree_cpu_seconds`` counts them
    (every pid of the tree: its own ticks plus its reaped children's), at a cost that does not grow with threads.

    THE BREAKAGE THIS PREVENTS (hex host-bound, reference system 2026-10-27): ``tree_cpu_seconds`` finds the live
    descendants by reading ``/proc/<pid>/task/<tid>/children`` for every thread, at every stage boundary.  A hex
    forecast holds ~170 threads (the 128-thread kernel I/O pool among them) and the MPAS-physics step loop opens
    and closes three timed stages a step, so every step read ~1,000 /proc files on the thread that issues the
    card's work: py-spy put 34 % of the LTS class threads' step-loop CPU there (96/3 km Milton mesh, 4 classes),
    with the card idle meanwhile.  Here the pid list is rediscovered (that same walk) at most every
    ``TREE_REFRESH_SECONDS``; between walks a boundary reads one ``/proc/<pid>/stat`` per known pid (usually one or
    two).  A child that is reaped between walks is counted in its parent's reaped ticks, as before; one born and
    still alive since the last walk is counted from the next walk on, at most 2 s late.  The ledger's cpu_s is a
    diagnostic (stage cores = CPU / wall); no forecast word depends on it.
    """
    from woof.timeline.parallelism import tree_pids
    now = clock()
    root = os.getpid()
    with _tree_lock:
        pids = _tree["pids"]
        if pids is None or _tree["root"] != root or now - _tree["at"] >= TREE_REFRESH_SECONDS:
            pids = _tree["pids"] = tuple(tree_pids(root))
            _tree["at"], _tree["root"] = now, root
    total = 0
    for pid in pids:
        ticks = _own_and_reaped_ticks(pid)
        if ticks is not None:
            total += ticks
    return round(total / _TICK, 3)


def _cpu_attrs(attrs, phase, t_mono):
    """Every live begin/end carries attrs.cpu_s, its process tree's CPU seconds at that moment, so each stage's
    cores = CPU / wall (owner ruling 2026-10-16: no stage on one CPU thread while cards or cores wait). Read here, not at
    append time, because hex holds early events until its run folder exists and replays them with their own
    clocks. ``tree_cpu_seconds`` reads /proc (live descendants and reaped children); off Linux, or with an engine
    without woof.timeline.parallelism, it falls back to os.times(). No device sync, no profiler."""
    if phase not in ("begin", "end") or t_mono is not None or (attrs and "cpu_s" in attrs):
        return attrs
    try:
        if not sys.platform.startswith("linux"):
            raise OSError("no /proc")
        cpu = tree_cpu_seconds()
    except Exception:
        t = os.times()
        cpu = round(t.user + t.system + t.children_user + t.children_system, 3)
    return attrs if cpu is None else {**(attrs or {}), "cpu_s": cpu}


@contextmanager
def member(run_id, *, card, root, open_stages=()):
    """``open_stages`` carries the stages a member opened on another thread (its device phase) into
    this context, so the stage it ends here (``post.forecast``) closes the interval it began there."""
    state = {"run_id": run_id, "card": card, "root": Path(root),
             "pending": [], "open": set(open_stages)}
    token = _member.set(state)
    try:
        yield
    finally:
        _member.reset(token)

_root = None
_pending = []
_warned = set()
_open = set()


def configure(root):
    global _root
    _root = None if root is None else Path(root)
    _pending.clear()
    _open.clear()


def card_index():
    # CUDA's visible ordinal zero is not necessarily the rental's card zero.
    value = (os.environ.get("WOOF_CARD") or os.environ.get("GPU_MUTEX_CARDS", "").split(",")[0]
             or os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0])
    return int(value) if value.isdecimal() else None


def emit(stage, phase="mark", *, attrs=None, root=None, t_mono=None, t_unix=None):
    """Retain early clocks until the driver's exclusive destination exists."""
    global _root
    attrs = _cpu_attrs(attrs, phase, t_mono)
    state = _member.get()
    if state is not None:
        destination = Path(root) if root is not None else state["root"]
        path = os.environ.get("WOOF_TIME_LEDGER") or str(destination / "timeline.jsonl")
        try:
            if phase == "begin":
                state["open"].add(stage)
            elif phase == "end":
                state["open"].discard(stage)
            event = (stage, phase, attrs or {}, time.monotonic() if t_mono is None else t_mono,
                     time.time() if t_unix is None else t_unix)
            if not os.environ.get("WOOF_TIME_LEDGER") and not destination.is_dir():
                state["pending"].append(event)
                return
            try:
                from woof.timeline.ledger import Ledger
            except ModuleNotFoundError:
                # An engine without woof.timeline (gpuwm 2.8.7 and earlier)
                # has no ledger to append to; the forecast runs unrecorded.
                state["pending"].clear()
                return
            ledger = Ledger(path, product="hex", card=state["card"], run_id=state["run_id"])
            for name, boundary, detail, mono, wall in [*state["pending"], event]:
                ledger.emit(name, boundary, attrs=detail, t_mono=mono, t_unix=wall)
            state["pending"].clear()
        except Exception as error:
            if path not in _warned:
                _warned.add(path)
                print(f"time ledger: cannot append {path}: {type(error).__name__}: {error}", file=sys.stderr)
        return
    destination = Path(root) if root is not None else _root
    if _root is None and destination is not None:
        _root = destination
    path = os.environ.get("WOOF_TIME_LEDGER") or (
        str(destination / "timeline.jsonl") if destination is not None else None)
    if path is None:
        return
    try:
        if phase == "begin":
            _open.add(stage)
        elif phase == "end":
            _open.discard(stage)
        try:
            from woof.timeline.ledger import Ledger, emit as append
        except ModuleNotFoundError:
            # No woof.timeline in this engine (gpuwm 2.8.7 and earlier):
            # nothing to append to, and nothing to warn about every run.
            _pending.clear()
            return
        event = (stage, phase, attrs or {}, time.monotonic() if t_mono is None else t_mono,
                 time.time() if t_unix is None else t_unix)
        if not os.environ.get("WOOF_TIME_LEDGER") and not destination.is_dir():
            _pending.append(event)
            return
        if not _pending and t_mono is None and t_unix is None:
            append(stage, phase, path=path, attrs=attrs, product="hex", card=card_index())
            return
        ledger = Ledger(path, product="hex", card=card_index())
        for name, boundary, detail, mono, wall in [*_pending, event]:
            ledger.emit(name, boundary, attrs=detail, t_mono=mono, t_unix=wall)
        _pending.clear()
    except BaseException as error:
        try:
            if path not in _warned:
                _warned.add(path)
                print(f"time ledger: cannot append {path}: {type(error).__name__}: {error}", file=sys.stderr)
        except BaseException:
            pass


def open_stages():
    state = _member.get()
    return set(state["open"] if state is not None else _open)


def end_if_open(stage):
    state = _member.get()
    if stage in (state["open"] if state is not None else _open):
        emit(stage, "end")


def begin_if_closed(stage, *, root=None):
    state = _member.get()
    if stage not in (state["open"] if state is not None else _open):
        emit(stage, "begin", root=root)


@contextmanager
def stage(name):
    """Pair a preparation interval, including a command's original refusal."""
    emit(name, "begin")
    try:
        yield
    finally:
        emit(name, "end")
