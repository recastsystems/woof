"""What a run says on its way out when something else decides to stop it.

A three-domain ERA5 forecast was reported as printing this, in full::

    $ woof run configs/run60.toml --outdir out/run60
    woof run: gpuwm 2.6.0 -- installed wheel at .../site-packages/gpuwm
    Terminated

One word, from the shell, not from woof.  ``Terminated`` is bash reporting
that its foreground job died on SIGTERM, and the package installed no
handler anywhere (``grep -rn "signal.signal(" gpuwm/`` was empty), so the
signal hit both woof processes at their default disposition: instant
death, no ``finally``, nothing printed.  The user got one word because
woof had nothing to say -- and could not tell whether it was their
config, their card, their data, or a woof crash.

This module is the sentence.  It says four things, because those are the
four a reader cannot recover afterwards:

* WHICH PHASE the run had reached, read from the run's own
  ``run-progress.json`` -- ``preparing:build-domain-tree`` is the forcing
  decode, ``integrating`` is the forecast, and they point at different
  causes;
* HOW MUCH MEMORY was held at that instant, host first, because the
  common killer is host RAM and no woof preflight prices any;
* THAT THE SIGNAL CAME FROM OUTSIDE, with the usual senders named, so
  the reader looks at ``journalctl`` rather than at their config;
* WHERE THE REAL OUTPUT IS -- a supervised run is two processes and the
  worker's stderr is a file in ``--outdir``, which the terminal never
  sees.

Three properties this deliberately keeps.

**The exit status does not change.**  The handler reports, restores the
disposition it replaced, and re-delivers the signal to itself.  A run
killed by SIGTERM still dies of SIGTERM (shell status 143); a Ctrl-C
still reaches the ``KeyboardInterrupt`` contract in
:mod:`woof.cli` and still returns 130.  A handler that swallowed the
signal would be worse than no handler at all.

Delivery is CPython's, not this module's: the C handler sets a flag and
the Python handler runs at the next bytecode boundary, so a worker deep
in a long C call reports when that call returns -- the same latency a
``KeyboardInterrupt`` has always had there.  The supervisor's own
``_terminate_fresh_worker`` still escalates to SIGKILL after ten
seconds, so nothing waits longer for this than it did before.

**A masked signal stays masked.**  A shell that starts a job in the
background sets SIGINT to ``SIG_IGN`` for it (POSIX job control), and
:func:`woof.cli._warn_if_interrupt_is_ignored` already declines to
fight that.  So does this: an ignored disposition is left exactly as the
shell set it and nothing is installed over it.

**Nothing here allocates that it can avoid.**  The killer is usually
memory, so the report is pre-rendered at install time and the handler
writes bytes with :func:`os.write` on file descriptor 2 -- not through
``print``, whose buffered ``TextIOWrapper`` the interrupted frame may
have been inside of.  It reads one small file, takes one ``pread`` on a
descriptor opened in advance, and calls no CUDA entry point at all.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
from pathlib import Path


#: The two signals a run dies of with no exception to carry the news.
#: SIGKILL is not here and cannot be: it is not deliverable to a handler.
WATCHED_SIGNAL_NAMES = ("SIGTERM", "SIGINT")

#: Enough of ``run-progress.json`` to hold the whole record.  The file is a
#: single compact JSON object of eleven scalar fields; 4 KiB is roughly ten
#: times its size, and one read that cannot be short is cheaper here than a
#: loop.
_HEARTBEAT_READ_BYTES = 4096

#: The heartbeat's phase field, as the bytes ``atomic_write_json`` emits for
#: it (``json.dumps(..., separators=(",", ":"))``).  Phases are normalized
#: to ``[A-Za-z0-9_.-]`` before publication, so the value never carries a
#: JSON escape and the scan below is exact.
_PHASE_KEY = b'"status":"'

_GIB = float(1 << 30)


def _bytes_text(count: int) -> str:
    """GiB to two places, or MiB below one, matching the run reports."""

    if count < _GIB:
        return f"{count / (1 << 20):.1f} MiB"
    return f"{count / _GIB:.2f} GiB"


def _open_rss_counter() -> int | None:
    """A read-only descriptor on this process's own RSS, opened early.

    Opened at install time so the handler spends one ``pread`` instead of
    an ``open`` on a machine that may be out of memory, out of descriptors,
    or both.  Absent off Linux, where the line reports that it was not read
    rather than inventing a number.
    """

    try:
        return os.open("/proc/self/statm", os.O_RDONLY)
    except OSError:
        return None


def _rss_bytes(counter_fd: int | None, page_size: int) -> int | None:
    """Resident set size now, from ``/proc/self/statm`` field two."""

    if counter_fd is None:
        return None
    try:
        return int(os.pread(counter_fd, 128, 0).split()[1]) * page_size
    except (OSError, IndexError, ValueError):
        return None


def _device_bytes() -> int | None:
    """Device bytes THIS process holds, from CuPy's own bookkeeping.

    ``memGetInfo`` is not called here and must not be: it stands up a CUDA
    primary context on a process that has none (``woof/runplan.py``,
    ``woof/go_cli.py``, and :mod:`woof.local_gpu` all say so), which is
    the last thing to do inside a signal handler on a box that just ran out
    of memory.  The default pool's counters are in-process bookkeeping over
    allocations already made; a process that never allocated reads zero
    without touching the device.

    ``None`` means CuPy was never imported here -- the supervisor parent,
    always.  The scope of the number, when there is one, is the same one
    :mod:`woof.core.gpu_mem_watch` labels ``cupy_pool_total``: bytes held
    by this process, not bytes used on the card.
    """

    module = sys.modules.get("cupy")
    if module is None:
        return None
    try:
        return int(module.get_default_memory_pool().total_bytes())
    except Exception:       # a half-imported CuPy has nothing to say
        return None


def _phase(heartbeat: str | None) -> str | None:
    """The run's own last published phase, without a JSON parse.

    The heartbeat is read from disk rather than from an in-memory copy for
    one reason: in the supervisor parent there is no in-memory copy of the
    WORKER's phase, and the worker's phase is the one that answers "where
    did it die".  Both processes publish to and read the same file.
    """

    if heartbeat is None:
        return None
    try:
        heartbeat_fd = os.open(heartbeat, os.O_RDONLY)
    except OSError:
        return None
    try:
        raw = os.read(heartbeat_fd, _HEARTBEAT_READ_BYTES)
    except OSError:
        return None
    finally:
        os.close(heartbeat_fd)
    start = raw.find(_PHASE_KEY)
    if start < 0:
        return None
    start += len(_PHASE_KEY)
    end = raw.find(b'"', start)
    if end < 0:
        return None
    return raw[start:end].decode("ascii", errors="replace")


#: Who sends SIGTERM to a forecast on a Linux box.  Every entry is a real
#: sender rather than a hedge, and the two commands are the ones that
#: settle which it was.
_SIGTERM_SENDERS = (
    b"  Nothing in woof sends a signal to this process, so it arrived\n"
    b"  from the host.  The usual senders are the kernel OOM killer\n"
    b"  escalating through systemd (OOMPolicy=stop, which SIGTERMs the\n"
    b"  whole session scope), systemd-oomd, earlyoom or nohang, a job\n"
    b"  scheduler's time limit, a `timeout` wrapper, and a kill or pkill\n"
    b"  on the process group.  These name it:\n"
    b"    journalctl -b | grep -i 'killed process'\n"
    b"    free -g\n")

#: The same paragraph for the worker, which woof's own supervisor DOES
#: signal: ``_terminate_fresh_worker`` sends SIGTERM here when a heartbeat
#: watchdog fires.  It then prints the reason on the parent's terminal, so
#: a silent parent is the tell that this came from the host instead.
_SIGTERM_SENDERS_WORKER = (
    b"  woof's supervisor sends SIGTERM to this worker when one of its\n"
    b"  heartbeat watchdogs fires, and then prints the reason on the\n"
    b"  terminal.  If the terminal said nothing, the signal came from the\n"
    b"  host instead: the kernel OOM killer escalating through systemd\n"
    b"  (OOMPolicy=stop), systemd-oomd, earlyoom or nohang, a scheduler's\n"
    b"  time limit, or a kill or pkill on the process group.  These name\n"
    b"  it:\n"
    b"    journalctl -b | grep -i 'killed process'\n"
    b"    free -g\n")

#: SIGINT is a Ctrl-C at the terminal, and the OOM paragraph above would
#: be a lie about it.  What is worth saying is what the reader cannot get
#: any other way -- the phase and the memory above this line, at the
#: instant they gave up -- so this says only who sent it.  The bounded
#: Ctrl-C contract itself (one sentence, exit 130, nothing signalled) is
#: woof/cli.py's and is printed by it, unchanged.
_SIGINT_SENDER = (
    b"  SIGINT is a Ctrl-C at this terminal, or a `kill -INT`.  woof\n"
    b"  did not send it.\n")

#: Which verb the header uses, and which paragraph closes the report, for
#: each signal this module handles.
_VERBS = {"SIGTERM": "stopped", "SIGINT": "interrupted"}

#: A door whose runs ARE stopped with these two signals (``stoppable``):
#: ``woof downscale``.  Its own refusal tells a reader to send SIGTERM,
#: the page and the terminal app stop a run with a SIGINT to its process
#: group, and the page follows that with SIGTERM when the run is still
#: going a minute later.  "Nothing in woof sends a signal" would be
#: false there, so the header says the stop is being recorded and the
#: tail names those senders before the host's.
_STOPPABLE_TAILS = {
    "SIGTERM": (
        b"  This is how a stop is asked for: `kill -TERM <pid>` at a\n"
        b"  terminal, or woof's page, which follows its Ctrl+C with SIGTERM\n"
        b"  when a run is still going a minute later.  If nobody asked for a\n"
        b"  stop, the host sent it: the kernel OOM killer escalating through\n"
        b"  systemd (OOMPolicy=stop), systemd-oomd, earlyoom or nohang, a\n"
        b"  scheduler's time limit, or a kill or pkill on the process group.\n"
        b"  These name it:\n"
        b"    journalctl -b | grep -i 'killed process'\n"
        b"    free -g\n"),
    "SIGINT": (
        b"  SIGINT is a Ctrl-C at this terminal, a `kill -INT`, or the Stop\n"
        b"  button of woof's page or terminal app, which send it to the\n"
        b"  run's whole process group.\n"),
}


@contextlib.contextmanager
def report_on_signal(command: str, *, heartbeat: str | Path | None = None,
                     logs: tuple[str | Path, ...] = (),
                     worker: bool = False, stoppable: bool = False):
    """Say what was happening here, for as long as this door owns the run.

    ``command`` is the front door as the reader typed it (``woof run``).
    ``heartbeat`` is this run's ``run-progress.json``; ``logs`` are files
    holding output the reader's terminal never saw.  Both are passed in
    rather than derived: :mod:`woof.supervisor` owns those names, and a
    second spelling of them here is a drift waiting to happen.

    Restores the dispositions it replaced on the way out, including on an
    exception, because ``woof.cli.main`` is called repeatedly in one
    interpreter by the test suite and by embedders -- the same reason
    ``woof.explain.explain_scope`` exists.

    ``stoppable`` is a door whose runs are stopped with exactly these
    signals and which records the stop itself (``woof downscale``):
    the header says the stop is being recorded rather than that it came
    from outside woof, and the tail names the page, the terminal app
    and ``kill -TERM`` before the host.  Delivery is unchanged: the
    disposition this replaced still receives the signal.
    """

    heartbeat_path = None if heartbeat is None else str(heartbeat)
    page_size = 4096
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        pass
    counter_fd = _open_rss_counter()
    pid = os.getpid()
    # Everything that does not depend on the moment of death, rendered
    # once, here, where allocation is free.
    pointers = b""
    for path in logs:
        pointers += f"  Output the terminal never saw: {path}\n".encode(
            "utf-8", errors="replace")
    headers = {}
    tails = {}
    previous = {}
    # A door with no heartbeat (``woof downscale``) keeps its stage in
    # its event stream instead, and "not published yet" would read as a
    # run that had not started.
    unpublished = ("not published yet" if heartbeat_path is not None
                   else "this door keeps no heartbeat; its events.jsonl "
                        "names the stage")

    def _report(number: int) -> None:
        # Header first and tail last, each one pre-rendered: the reader
        # gets the sentence that names the killer even if the middle
        # block -- the only part that reads anything -- cannot be built.
        _write(headers[number])
        try:
            rss = _rss_bytes(counter_fd, page_size)
            device = _device_bytes()
            _write(b"".join((
                b"  phase:  ",
                (_phase(heartbeat_path) or unpublished).encode(
                    "ascii", errors="replace"),
                b"\n  host:   ",
                (b"RSS is not readable on this platform" if rss is None
                 else _bytes_text(rss).encode("ascii")
                 + b" resident (RSS) in this process"),
                b"\n  device: ",
                (b"CuPy was never imported in this process" if device is None
                 else b"no device allocation in this process" if not device
                 else _bytes_text(device).encode("ascii")
                 + b" held by this process's CuPy pool"),
                b"\n")))
        except Exception:       # a report is not worth an exception here
            pass
        _write(tails[number])

    def _handler(number, frame):
        _report(number)
        # Re-delivered, never swallowed.  Restoring the disposition this
        # replaced and handing the signal back is what keeps the exit
        # status identical: SIG_DFL kills the process on SIGTERM exactly
        # as it did before, and CPython's default_int_handler raises the
        # KeyboardInterrupt that woof/cli.py turns into exit 130.
        earlier = previous[number]
        try:
            signal.signal(number, earlier)
        except (ValueError, OSError):       # pragma: no cover
            pass
        if callable(earlier):
            earlier(number, frame)
            return
        try:
            os.kill(pid, number)
        except OSError:                     # pragma: no cover
            os._exit(128 + number)

    installed = []
    try:
        for name in WATCHED_SIGNAL_NAMES:
            number = getattr(signal, name, None)
            if number is None:
                continue
            try:
                earlier = signal.getsignal(number)
            except (ValueError, OSError):   # pragma: no cover
                continue
            # A shell that backgrounded this job masked the signal on
            # purpose.  woof/cli.py names that rather than fighting it,
            # and installing over it here would defeat the same intent.
            if earlier is signal.SIG_IGN:
                continue
            # ``getsignal`` answers None when the disposition was NOT
            # installed through the signal module -- a C extension, an
            # embedding host, ``faulthandler.register``.  That value is
            # not accepted back by ``signal.signal``, which raises
            # TypeError, and TypeError is in neither restore path's
            # caught tuple: installing here would break both promises
            # this module makes.  The handler would die of an unhandled
            # TypeError instead of re-delivering the signal (exit 1, not
            # 143), and the context manager's ``finally`` would raise on
            # the way out of a run that SUCCEEDED.  Substituting SIG_DFL
            # would fix the crash and silently discard a disposition
            # somebody else installed on purpose, which is the same
            # intent the SIG_IGN branch above declines to fight, so this
            # declines too: a run that cannot install one is exactly as
            # it was before this module existed.
            if earlier is None:
                continue
            headers[number] = (
                f"\n{command}: {name} (signal {int(number)}) at pid {pid} "
                + ("-- stopping; the run records the stop and ends.\n"
                   if stoppable else
                   f"-- {_VERBS[name]} from outside woof.\n")).encode(
                    "utf-8", errors="replace")
            if stoppable:
                tails[number] = _STOPPABLE_TAILS[name] + pointers
            elif name == "SIGINT":
                tails[number] = _SIGINT_SENDER + pointers
            else:
                tails[number] = (_SIGTERM_SENDERS_WORKER if worker
                                 else _SIGTERM_SENDERS) + pointers
            previous[number] = earlier
            try:
                signal.signal(number, _handler)
            except (ValueError, OSError):
                # Not the main thread of the main interpreter, or no
                # signal support here.  A run that cannot install one is
                # exactly as it was before this module existed.
                continue
            installed.append((number, earlier))
        yield
    finally:
        for number, earlier in installed:
            try:
                signal.signal(number, earlier)
            except (ValueError, OSError):   # pragma: no cover
                pass
        if counter_fd is not None:
            try:
                os.close(counter_fd)
            except OSError:                 # pragma: no cover
                pass


def _write(text: bytes) -> None:
    """One write, on the descriptor, never through a buffered stream.

    ``print`` would go through ``sys.stderr``, a buffered
    ``TextIOWrapper``: a Python signal handler runs in the main thread
    between bytecodes, so a signal that lands while the interrupted frame
    is inside ``sys.stderr.write`` would re-enter a writer that is not
    re-entrant.  Descriptor 2 is also what :func:`subprocess.Popen`
    redirected into ``worker-NN.stderr.log``, so a worker's notice lands
    in the file the reader is being pointed at.
    """

    try:
        os.write(2, text)
    except OSError:                         # pragma: no cover
        pass


__all__ = ["WATCHED_SIGNAL_NAMES", "report_on_signal"]
