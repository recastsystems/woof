"""The prepared door's background checkpoint writer and its two orderings.

Breakage this guards (1 km CONUS, 8 x RTX 5090, 2026-10-04): every hourly
checkpoint stopped the model for 31 to 37 s -- about 12 s waiting for the
hour's history frame to land, then 18 to 21 s writing a 25 GB archive.  The
checkpoint is now snapshotted between two steps and written on a thread of
its own.  What must not change with it:

* a checkpoint is published only after every history frame up to its valid
  time is durable (``PerDomainWrfoutWriters.durability_barrier``);
* the published archive is the synchronous writer's, byte for byte;
* the snapshot owns what it writes, so later steps cannot change it;
* a failed write is raised on the stepping thread, never lost;
* a frame's landing latency is timed from the frame, not from the log's
  creation or from whatever step the model has since reached.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pytest

from tests.test_restart import (_LIFECYCLE_FREE_CHILD_DIGEST,
                                _LIFECYCLE_FREE_ROOT_DIGEST,
                                _PINNED_UNDER_MEASURED_CLOCK,
                                _canonical_member_digest,
                                _sealed_tree_fixture)
from tests.test_wrfout import _manual_async_writer, _queue_cpu_ticket


# -- the publisher ------------------------------------------------------------

def test_a_failed_background_write_is_raised_on_the_stepping_thread():
    from woof.prepared_single_domain_forecast import _CheckpointPublisher

    publisher = _CheckpointPublisher()

    def fails():
        raise OSError("disk full")

    publisher.submit(fails)
    with pytest.raises(RuntimeError, match="checkpoint writer failed: OSError: disk full"):
        publisher.wait()
    publisher.wait()  # reported once, then clear


def test_at_most_one_checkpoint_is_in_flight():
    from woof.prepared_single_domain_forecast import _CheckpointPublisher

    publisher = _CheckpointPublisher()
    release = threading.Event()
    order = []

    def first():
        release.wait()
        order.append("first")

    publisher.submit(first)
    threading.Timer(0.2, release.set).start()
    publisher.submit(lambda: order.append("second"))
    publisher.wait()
    assert order == ["first", "second"]


@pytest.mark.parametrize("failure_phase", ["capture", "history", "submit"])
def test_single_checkpoint_releases_borrowed_store_before_publication_failure(
        monkeypatch, tmp_path, failure_phase):
    """History/submit failures must release a snapshot that never publishes.

    Bind the production nested hook without initializing a forecast; use the
    actual pending checkpoint and publisher, with faults at the I/O boundary.
    A retained store guard otherwise blocks the next ranked frame download.
    """
    import ast
    import contextlib
    from pathlib import Path
    from types import SimpleNamespace

    from woof import prepared_single_domain_forecast as single
    from woof.io import restart

    guards = []
    model = SimpleNamespace(
        _last_checkpoint=None, schedule=SimpleNamespace(
            clock=SimpleNamespace(tick_den=1)))
    pending = restart.PendingTreeCheckpoint(
        directory=tmp_path, model=model, root_id=1,
        members=[(1, lambda: pytest.fail("failed history must not publish"))],
        lenders=[SimpleNamespace(add_store_guard=guards.append)])
    monkeypatch.setattr(restart, "write_tree_restart", lambda *a, **k: pending)

    def fails():
        raise OSError("injected " + failure_phase + " failure")

    publisher = single._CheckpointPublisher()
    if failure_phase == "submit":
        monkeypatch.setattr(publisher, "submit", lambda job, **_: fails())
    writers = SimpleNamespace(
        durability_barrier=(fails if failure_phase == "capture" else lambda: fails),
        drain=fails)
    source_path = Path(single.__file__)
    source = ast.parse(source_path.read_text())
    hook = next(
        node for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef) and node.name == "restart_handler"
        and any(isinstance(child, ast.Name) and child.id == "checkpoint_publisher"
                for child in ast.walk(node)))
    env = dict(
        checkpoint_publisher=publisher, writers=writers,
        exp=SimpleNamespace(start_time=datetime(2026, 10, 18),
                            root=SimpleNamespace(grid_id=1), auto_epssm=()),
        timedelta=timedelta, time=time, observer=None, outdir=tmp_path,
        runtime=SimpleNamespace(
            _writing_progress=lambda *a, **k: contextlib.nullcontext(),
            _checkpoint_work_bytes=lambda tree: 1),
        checkpoints=[], checkpoint_ticks={}, step_log=None, Path=Path)
    exec(compile(ast.Module(body=[hook], type_ignores=[]), str(source_path), "exec"), env)
    try:
        with pytest.raises((OSError, RuntimeError), match="injected .* failure"):
            env["restart_handler"](model, 900)
            publisher.wait()
        assert guards[0].__self__.is_set(), "borrowed store guard remains locked"
        assert pending._members == [], "failed checkpoint retains borrowed views"
        assert model._last_checkpoint is None
    finally:
        pending.abandon()
        publisher.wait()


def test_the_writer_never_retires_the_set_the_heartbeat_is_reading(
        monkeypatch, tmp_path):
    """Hourly checkpoints with ``--keep-checkpoints 1`` rotate cleanly.

    Breakage (ml-probs T1, 2026-10-27, gpuwm 2.9.8): every 60 h GFS 12 km
    parent with ``restart_interval_s = 3600`` died 3-7 h in with
    FileNotFoundError on its own checkpoint (run dir held the 08Z set, the
    run asked for the 07Z one).  The 08Z restart handler joins the 07Z
    write, which hands ``model._last_checkpoint`` = 07Z to the period
    commit; the supervisor heartbeat reads that set back in full on the
    stepping thread while the writer thread publishes 08Z -- and the writer
    retired 07Z under it.  Driven here through the door's own restart
    handler, the real tree writer, the real publisher and the real
    heartbeat, with the 08Z write held until the commit has named 07Z.
    """
    import ast
    import contextlib
    from pathlib import Path
    from types import SimpleNamespace

    from woof import prepared_single_domain_forecast as single
    from woof.resume import KEEP_CHECKPOINTS_ENV, discover_checkpoint_sets
    from woof.supervisor import RuntimeHeartbeat

    monkeypatch.setenv(KEEP_CHECKPOINTS_ENV, "1")
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3 * 3600.0, payload_seed=31,
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)
    gates = {}

    def barrier():
        gate = gates.setdefault(len(gates) + 1, threading.Event())
        return gate.wait

    publisher = single._CheckpointPublisher()
    source_path = Path(single.__file__)
    tree = ast.parse(source_path.read_text())
    hook = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "restart_handler"
        and any(isinstance(child, ast.Name) and child.id == "checkpoint_publisher"
                for child in ast.walk(node)))
    env = dict(
        checkpoint_publisher=publisher,
        writers=SimpleNamespace(durability_barrier=barrier, drain=lambda: None),
        exp=SimpleNamespace(start_time=start, root=SimpleNamespace(grid_id=1),
                            auto_epssm=()),
        timedelta=timedelta, time=time, observer=None, outdir=tmp_path,
        runtime=SimpleNamespace(
            _writing_progress=lambda *a, **k: contextlib.nullcontext(),
            _checkpoint_work_bytes=lambda tree: 1),
        checkpoints=[], checkpoint_ticks={},
        step_log=SimpleNamespace(restart_written=lambda **_: None), Path=Path)
    exec(compile(ast.Module(body=[hook], type_ignores=[]), str(source_path),
                 "exec"), env)
    handler = env["restart_handler"]
    heartbeat = RuntimeHeartbeat(
        tmp_path / "heartbeat.json", run_id="rotation", config_sha256="0" * 64,
        started_at_utc="2026-10-27T00:00:00Z")

    def hours_on_disk():
        return sorted(int(item.valid_time.hour)
                      for item in discover_checkpoint_sets(tmp_path))

    def join_writer():
        thread = publisher._thread
        if thread is not None:
            thread.join(timeout=60)
            assert not thread.is_alive()

    try:
        handler(source, 3600)
        gates[1].set()
        # The hour-2 handler joins the hour-1 write and submits hour 2,
        # whose write waits on its history barrier.
        handler(source, 7200)
        named = source._last_checkpoint
        assert named is not None and "_01_00_00__" in Path(named).name
        # The hour-2 write lands while the period commit still names hour 1.
        gates[2].set()
        join_writer()
        heartbeat(model_elapsed_seconds=7200.0, outer_step=2,
                  last_durable_wrfout=None, last_checkpoint=named)
        assert heartbeat.last_checkpoint == str(Path(named).resolve())
        assert hours_on_disk() == [1, 2]
        # Retention still holds: the next handler retires hour 1 on this
        # thread, and the end-of-run wait leaves only the newest set.
        handler(source, 10800)
        assert hours_on_disk() == [2]
        gates[3].set()
        publisher.wait()
        assert hours_on_disk() == [3]
    finally:
        for gate in gates.values():
            gate.set()
        join_writer()


# -- the snapshot -------------------------------------------------------------

def test_a_deferred_checkpoint_is_the_synchronous_one_byte_for_byte(
        monkeypatch, tmp_path):
    from woof.io import restart

    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31,
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)
    pending = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600),
        defer_publish=True)
    assert isinstance(pending, restart.PendingTreeCheckpoint)
    assert pending.deferrable
    assert not list(tmp_path.glob("gpuwmrst_*")), "nothing lands before publish"
    assert source._last_checkpoint is None
    root_path = pending.publish()
    child_path = next(p for p in tmp_path.glob("gpuwmrst_d02_*.npz"))
    assert source._last_checkpoint == root_path
    assert _canonical_member_digest(root_path) == _LIFECYCLE_FREE_ROOT_DIGEST
    assert _canonical_member_digest(child_path) == _LIFECYCLE_FREE_CHILD_DIGEST


def test_a_deferred_checkpoint_owns_what_it_writes(monkeypatch, tmp_path):
    """Steps taken between the snapshot and the write change nothing in it."""
    from woof.io import restart

    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31,
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)
    pending = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600),
        defer_publish=True)
    touched = 0
    for node in source.walk_parent_first():
        for value in restart.state_manifest(node.state).values():
            if isinstance(value, np.ndarray) and value.size \
                    and value.dtype.kind == "f":
                value[...] += 1.0
                touched += 1
    assert touched
    root_path = pending.publish()
    assert _canonical_member_digest(root_path) == _LIFECYCLE_FREE_ROOT_DIGEST


def test_frames_in_flight_refuse_a_synchronous_checkpoint_but_not_a_deferred_one(
        monkeypatch, tmp_path):
    """The D2H count stops a checkpoint landing ahead of its frames.  A
    deferred one is published only after them, so it is not refused; found
    by the first real run, whose CPU fixtures carry no io manager."""
    from types import SimpleNamespace
    from woof.io import restart

    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31,
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)
    source._io_manager = SimpleNamespace(pending=1)
    with pytest.raises(restart.RestartMismatchError, match="'D2H': 1"):
        restart.write_tree_restart(
            tmp_path, source, start + timedelta(seconds=3600))
    pending = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600),
        defer_publish=True)
    root_path = pending.publish()
    assert _canonical_member_digest(root_path) == _LIFECYCLE_FREE_ROOT_DIGEST


def test_a_failed_publish_removes_the_generation_and_releases_its_store(
        monkeypatch, tmp_path):
    from woof.io import restart

    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31)

    class Lender:
        guards = []

        def add_store_guard(self, guard):
            self.guards.append(guard)

    pending = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600),
        defer_publish=True)
    lender = Lender()
    # Rebuild with a lender and a root member that fails after the child.
    members = list(pending._members)
    child_gid, child_member = members[0]

    def root_fails():
        raise OSError("injected root write failure")

    failing = restart.PendingTreeCheckpoint(
        directory=tmp_path, model=source, root_id=1,
        members=[(child_gid, child_member), (1, root_fails)],
        lenders=[lender])
    assert len(lender.guards) == 1
    with pytest.raises(OSError, match="injected root write failure"):
        failing.publish()
    assert not list(tmp_path.glob("gpuwmrst_*")), \
        "a half-written generation must not stay on disk"
    lender.guards[0]()  # released: returns at once instead of blocking
    assert source._last_checkpoint is None


# -- the history ordering -----------------------------------------------------

class _BlockingWriter:
    started = None
    finish = None

    def __init__(self, _path, **_kwargs):
        pass

    def __enter__(self):
        return self

    def write_frame(self, _time_str, _fields):
        type(self).started.set()
        type(self).finish.wait()

    def __exit__(self, *_args):
        return False


def test_the_barrier_waits_for_frames_queued_before_it_and_no_later(
        monkeypatch, tmp_path):
    import woof.io.wrfout as wrfout

    _BlockingWriter.started = threading.Event()
    _BlockingWriter.finish = threading.Event()
    monkeypatch.setattr(wrfout, "WrfoutWriter", _BlockingWriter)
    writer = _manual_async_writer(threading.Event())
    try:
        _queue_cpu_ticket(writer, tmp_path / "f00")
        assert _BlockingWriter.started.wait(timeout=30)
        target = writer.durability_target()
        assert target == 1
        landed = threading.Event()
        waiter = threading.Thread(
            target=lambda: (writer.wait_landed(target), landed.set()))
        waiter.start()
        assert not landed.wait(timeout=0.3), "returned before the frame landed"
        _BlockingWriter.finish.set()
        assert landed.wait(timeout=30)
        waiter.join()
        # A frame queued after the barrier is not waited for.
        _BlockingWriter.started.clear()
        _BlockingWriter.finish.clear()
        _queue_cpu_ticket(writer, tmp_path / "f01")
        assert _BlockingWriter.started.wait(timeout=30)
        writer.wait_landed(target)  # returns at once
        _BlockingWriter.finish.set()
        writer.drain()
        assert writer.durability_target() == 2
    finally:
        _BlockingWriter.finish.set()
        writer.close()


def test_a_frame_that_never_lands_fails_the_barrier_by_name(
        monkeypatch, tmp_path):
    import woof.io.wrfout as wrfout

    class FailingWriter(_BlockingWriter):
        def write_frame(self, _time_str, _fields):
            raise OSError("injected history failure")

    monkeypatch.setattr(wrfout, "WrfoutWriter", FailingWriter)
    writer = _manual_async_writer(threading.Event())
    try:
        _queue_cpu_ticket(writer, tmp_path / "f00")
        target = writer.durability_target()
        with pytest.raises(RuntimeError, match="injected history failure"):
            writer.wait_landed(target)
    finally:
        with pytest.raises(RuntimeError):
            writer.close()


# -- the landing latency ------------------------------------------------------

def test_a_frame_is_timed_from_its_own_hand_off(tmp_path):
    from woof.progress_log import StepLog

    log = StepLog(start_time=datetime(2026, 10, 3, 21), run_seconds=7200.0,
                  text_stream=StringIO(), jsonl_path=tmp_path / "p.jsonl")
    valid0 = datetime(2026, 10, 3, 21)
    valid1 = valid0 + timedelta(hours=1)
    frame0 = tmp_path / "wrfout_d01_0"
    frame1 = tmp_path / "wrfout_d01_1"
    frame0.write_bytes(b"x")
    frame1.write_bytes(b"x")
    time.sleep(0.5)                       # preflight, cache restore, ...
    log.frame_submitted(domain=1, valid_time=valid0)
    log.output_committed(domain=1, valid_time=valid0, path=frame0)
    log.domain_step(grid_id=1, step_count=1, model_seconds=3600.0,
                    step_wall_seconds=0.1)
    log.frame_submitted(domain=1, valid_time=valid1)
    time.sleep(0.3)                       # the write
    log.domain_step(grid_id=1, step_count=2, model_seconds=3606.0,
                    step_wall_seconds=0.1)  # the model moved on meanwhile
    log.output_committed(domain=1, valid_time=valid1, path=frame1)
    log.close()
    import json
    rows = [json.loads(line) for line in
            (tmp_path / "p.jsonl").read_text().splitlines()]
    latencies = [r["durable_after_seconds"] for r in rows
                 if r["event"] == "output_written"]
    assert latencies[0] < 0.4, "frame 0 was timed from the log's creation"
    assert latencies[1] >= 0.3, "frame 1 was timed from a later step"
