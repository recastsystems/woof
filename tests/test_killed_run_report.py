"""A killed run says what it was doing, and still dies of the same signal.

The report that produced this file was one word long::

    $ woof run configs/run60.toml --outdir out/run60
    woof run: gpuwm 2.6.0 -- installed wheel at .../site-packages/gpuwm
    Terminated

``Terminated`` is the shell, not woof: bash saying its foreground job
died on SIGTERM.  The package installed no handler anywhere -- the only
two signal references in it were a *read* of the SIGINT disposition in
``woof/cli.py`` and the MCP job canceller's outbound ``killpg`` -- so
the signal hit both woof processes at ``SIG_DFL`` and they died with
nothing printed.  The reader could not tell whether it was their config,
their card, their data, or a woof crash, and the one file that would
have said (``out/run60/worker-01.stderr.log``) they did not know existed.

What this file pins:

* both front doors of a supervised run install the report -- the parent
  the shell waits on AND the worker that holds the forcing and the card;
* the report names the phase from the run's own ``run-progress.json``,
  the host RSS at that instant, and the worker log;
* the signal is re-delivered, never swallowed: a run killed by SIGTERM
  still dies of SIGTERM, and a Ctrl-C still reaches the
  ``KeyboardInterrupt`` contract ``woof/cli.py`` turns into exit 130;
* an ignored disposition is left ignored, because a shell that
  backgrounded the job masked it on purpose (the same reasoning as
  ``tests/test_interrupt_contract.py``);
* the device leg reads CuPy's own pool counters and never ``memGetInfo``,
  which would stand up a CUDA primary context inside a signal handler on
  a box that just ran out of memory.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import types
from pathlib import Path

import pytest

import woof.supervisor as supervisor
from woof.signal_report import _phase, report_on_signal
from woof.supervisor import (HEARTBEAT_NAME, HEARTBEAT_SCHEMA,
                              WORKER_STDERR_NAME, Heartbeat, SupervisorResult,
                              write_heartbeat)

REPO = Path(__file__).resolve().parents[1]


def _heartbeat(status):
    return Heartbeat(
        HEARTBEAT_SCHEMA, "run-1", "a" * 64, os.getpid(),
        "2026-09-03T00:00:00Z", "2026-09-03T00:00:01Z", status,
        0.0, 0, None, None)


def _run_args(tmp_path):
    """The namespace ``woof run`` hands the supervised front door."""

    return argparse.Namespace(
        command="run", config=tmp_path / "experiment.toml",
        outdir=tmp_path / "out", restart=None, gpu_uuid=None,
        supervisor_max_restarts=3, prep_timeout=None, allow_shared_gpu=False,
        health_debug=False, directory_input_hash=None)


# --------------------------------------------------------------------
# Both doors install it
# --------------------------------------------------------------------

def test_the_parent_the_shell_waits_on_installs_the_report(monkeypatch,
                                                           tmp_path, capsys):
    """`supervise_from_cli` is the process that printed nothing at all."""

    before = signal.getsignal(signal.SIGTERM)
    seen = []

    def supervised(*_args, **_kwargs):
        seen.append(signal.getsignal(signal.SIGTERM))
        return SupervisorResult("run-1", 1, _heartbeat("complete"), (), ())

    monkeypatch.setattr(supervisor, "supervise_experiment", supervised)
    assert supervisor.supervise_from_cli(_run_args(tmp_path)) == 0
    capsys.readouterr()

    assert seen and seen[0] is not before, (
        "the supervised run ran with SIGTERM at the disposition it "
        "inherited; a SIGTERM there prints nothing at all")
    assert callable(seen[0])
    assert signal.getsignal(signal.SIGTERM) is before, (
        "the door left its handler behind after returning")


def test_the_worker_that_holds_the_forcing_installs_it_too(monkeypatch,
                                                           tmp_path):
    """Two processes, two doors.  The worker is where the forcing decode
    and the card are, so it is the one an OOM kill lands on first."""

    before = signal.getsignal(signal.SIGTERM)
    seen = []

    def worker(_args):
        seen.append(signal.getsignal(signal.SIGTERM))
        return 0

    monkeypatch.setattr(supervisor, "_worker_main", worker)
    assert supervisor.main([
        "worker", "--config", str(tmp_path / "experiment.toml"),
        "--config-payload", str(tmp_path / "captured.toml"),
        "--outdir", str(tmp_path / "out")]) == 0

    assert seen and seen[0] is not before
    assert callable(seen[0])
    assert signal.getsignal(signal.SIGTERM) is before


# --------------------------------------------------------------------
# What it says, and that the death is unchanged
# --------------------------------------------------------------------

_CHILD = """
import os, signal, sys
sys.path.insert(0, {repo!r})
from woof.signal_report import report_on_signal
from woof.supervisor import HEARTBEAT_NAME, WORKER_STDERR_NAME
outdir = {outdir!r}
with report_on_signal(
        "woof run", heartbeat=os.path.join(outdir, HEARTBEAT_NAME),
        logs=(os.path.join(outdir, WORKER_STDERR_NAME.format(attempt=1)),)):
    os.kill(os.getpid(), signal.SIGTERM)
    sys.stderr.write("THE HANDLER SWALLOWED THE SIGNAL\\n")
"""


@pytest.mark.skipif(os.name == 'nt', reason="Windows os.kill(SIGTERM) uses abrupt TerminateProcess; this test requires POSIX signal delivery and wait status")
def test_a_real_sigterm_is_reported_and_then_still_kills_the_process(
        tmp_path):
    """End to end, in a child, because the signal really is delivered.

    A handler that reported and then swallowed the signal would be worse
    than no handler: the run would carry on unkillable and the shell's
    exit status would lie.  Asserting the child's wait status is
    ``-SIGTERM`` is what holds the re-delivery down.
    """

    outdir = tmp_path / "out"
    outdir.mkdir()
    write_heartbeat(outdir / HEARTBEAT_NAME,
                    _heartbeat("preparing:build-domain-tree"))

    finished = subprocess.run(
        [sys.executable, "-c", _CHILD.format(repo=str(REPO),
                                             outdir=str(outdir))],
        capture_output=True, text=True, timeout=120)

    assert finished.returncode == -signal.SIGTERM, (
        "the process no longer dies of the signal it was sent")
    assert "THE HANDLER SWALLOWED THE SIGNAL" not in finished.stderr
    report = finished.stderr
    assert "SIGTERM (signal 15)" in report
    assert "preparing:build-domain-tree" in report      # the phase
    assert "resident (RSS)" in report                   # host memory
    assert "OOM killer" in report and "journalctl" in report
    assert str(outdir / WORKER_STDERR_NAME.format(attempt=1)) in report


def test_a_ctrl_c_still_reaches_the_exit_130_contract(capfd):
    """SIGINT is chained to the disposition it replaced, not consumed.

    ``woof/cli.py`` catches ``KeyboardInterrupt`` and returns 130 with
    one sentence.  That contract is not this module's to change, so the
    handler restores CPython's ``default_int_handler`` and calls it --
    which raises, exactly as an unhandled SIGINT did.
    """

    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with report_on_signal("woof run"):
            handler = signal.getsignal(signal.SIGINT)
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGINT, None)
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, previous)

    report = capfd.readouterr().err
    assert "SIGINT (signal 2)" in report
    assert "Ctrl-C" in report
    # The OOM paragraph belongs to SIGTERM.  A Ctrl-C was not sent by
    # earlyoom and saying so would be a lie in the one place a reader is
    # most inclined to believe it.
    assert "OOM killer" not in report


def test_a_masked_signal_is_left_exactly_as_the_shell_set_it():
    """A backgrounded job's SIGINT is ``SIG_IGN`` by POSIX job control.

    ``woof/cli.py`` declines to install over that and names it instead;
    installing here would defeat the same intent, and would make Ctrl-C
    stop a job the shell had deliberately detached from it.
    """

    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with report_on_signal("woof run"):
            assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, previous)


def test_a_disposition_this_module_cannot_restore_is_left_alone(monkeypatch,
                                                                capfd):
    """``getsignal`` answers ``None`` for a handler installed from outside
    the signal module -- a C extension, an embedding host,
    ``faulthandler.register``.  ``signal.signal`` will not take that value
    back: it raises ``TypeError``, which neither restore path catches.  A
    door that installed over it would therefore break both promises this
    module makes -- the handler would die of an unhandled ``TypeError``
    (exit 1) instead of re-delivering the signal (143), and the context
    manager's ``finally`` would raise on the way out of a run that
    SUCCEEDED.  So it is left alone, for the same reason a masked signal
    is, and the scope is one signal: SIGINT is still installed here.
    """

    real_getsignal = signal.getsignal
    before = real_getsignal(signal.SIGTERM)
    interrupt_before = real_getsignal(signal.SIGINT)

    def foreign(number):
        return None if number == signal.SIGTERM else real_getsignal(number)

    monkeypatch.setattr(signal, "getsignal", foreign)
    try:
        with report_on_signal("woof run"):
            assert real_getsignal(signal.SIGTERM) is before, (
                "installed over a disposition it cannot put back")
            assert real_getsignal(signal.SIGINT) is not interrupt_before
    finally:
        signal.signal(signal.SIGTERM, before)

    assert real_getsignal(signal.SIGTERM) is before
    assert real_getsignal(signal.SIGINT) is interrupt_before
    assert capfd.readouterr().err == ""


# --------------------------------------------------------------------
# The two readings, and what they are allowed to touch
# --------------------------------------------------------------------

def test_the_phase_is_read_from_what_the_run_itself_publishes(tmp_path):
    """Non-vacuity for the byte scan: the writer is the product's own.

    The scan looks for the literal bytes ``"status":"`` because
    ``atomic_write_json`` publishes compact sorted JSON and phases are
    normalized to ``[A-Za-z0-9_.-]`` before publication.  Both halves are
    somebody else's to change, so the reader is exercised against the
    real writer rather than against a hand-typed string.
    """

    path = tmp_path / HEARTBEAT_NAME
    write_heartbeat(path, _heartbeat("preparing:build-domain-tree"))
    assert _phase(str(path)) == "preparing:build-domain-tree"

    write_heartbeat(path, _heartbeat("integrating"))
    assert _phase(str(path)) == "integrating"

    # A run that died before its first beat has no file, and says so
    # rather than inventing a phase.
    assert _phase(str(tmp_path / "absent.json")) is None


def test_the_device_leg_reads_the_pool_and_never_memgetinfo(monkeypatch,
                                                            capfd):
    """``memGetInfo`` stands up a CUDA primary context on a process that
    has none -- ``woof/runplan.py``, ``woof/go_cli.py`` and
    :mod:`woof.local_gpu` all say so.  Doing that inside a signal
    handler, on a box that has just run out of memory, is the last thing
    to do.  The default pool's counters are in-process bookkeeping and
    need no device at all.
    """

    def forbidden():
        raise AssertionError(
            "the death notice called memGetInfo and stood up a context")

    fake = types.ModuleType("cupy")
    fake.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(memGetInfo=forbidden))
    fake.get_default_memory_pool = lambda: types.SimpleNamespace(
        total_bytes=lambda: 11 * (1 << 30))
    monkeypatch.setitem(sys.modules, "cupy", fake)

    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_a: None)   # a chainable stand-in
    try:
        with report_on_signal("woof run worker", worker=True):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, previous)

    report = capfd.readouterr().err
    assert "11.00 GiB held by this process's CuPy pool" in report
