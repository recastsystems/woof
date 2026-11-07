"""History frames written by the background process are the bytes a direct call writes."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from woof.hex import history_writer
from woof.hex.drivers import run_cuda_v841_full_physics_x4 as proof


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(rng, cells, edges, levels, soil):
    arrays = {name: rng.standard_normal((levels, cells)).astype(np.float32) for name in proof.SCALAR_NAMES}
    arrays.update(
        theta=rng.uniform(250, 320, (levels, cells)).astype(np.float32),
        normal_u=rng.standard_normal((levels, edges)).astype(np.float32),
        w=rng.standard_normal((levels + 1, cells)).astype(np.float32),
        smois=rng.uniform(0, 0.4, (soil, cells)).astype(np.float32),
        t2=rng.uniform(250, 320, cells).astype(np.float32),
        ignored=np.zeros((3, 3), dtype=np.float32),
    )
    static = {
        "indexToCellID": np.arange(1, cells + 1, dtype=np.int32),
        "latCell": rng.uniform(-1, 1, cells).astype(np.float32),
        "lonCell": rng.uniform(0, 6, cells).astype(np.float32),
        "ter": rng.uniform(0, 3000, cells).astype(np.float32),
        "indexToEdgeID": np.arange(1, edges + 1, dtype=np.int32),
        "latEdge": rng.uniform(-1, 1, edges).astype(np.float32),
        "lonEdge": rng.uniform(0, 6, edges).astype(np.float32),
    }
    return {"arrays": arrays}, static


@pytest.fixture
def small_mesh(monkeypatch):
    # The mesh binding rebinds these at run time; the writer process must
    # receive the forecast's values, not the module defaults.
    for name, value in (("N_CELLS", 97), ("N_EDGES", 291), ("N_LEVELS", 7),
                        ("N_INTERFACES", 8), ("N_SOIL_LEVELS", 4)):
        monkeypatch.setattr(proof, name, value)
    return 97, 291, 7, 4


@pytest.mark.parametrize("with_clock", [False, True])
def test_background_frames_are_the_direct_bytes(tmp_path, small_mesh, monkeypatch, with_clock):
    monkeypatch.delenv(history_writer.ENVIRONMENT, raising=False)
    monkeypatch.delenv("WOOF_HEX_HISTORY_SPOOL", raising=False)
    rng = np.random.default_rng(3)
    frames = [_frame(rng, *small_mesh) for _ in range(3)]
    if with_clock:
        for i, (snapshot, _) in enumerate(frames):
            snapshot["valid_time"] = f"2026-10-06_{i:02d}:00:00"
    (tmp_path / "direct").mkdir()
    (tmp_path / "background").mkdir()
    direct = [proof.write_snapshot_netcdf(tmp_path / "direct" / f"h{i}.nc", *frame)
              for i, frame in enumerate(frames)]
    writer = history_writer.open_writer(proof.write_snapshot_netcdf)
    assert writer is not None
    assert writer.ready["prepared"] is True
    assert writer.max_pending == 2
    futures = [writer.submit(tmp_path / "background" / f"h{i}.nc", *frame) for i, frame in enumerate(frames)]
    results = [writer.result(future) for future in futures]
    writer.close()
    for i, (a, b) in enumerate(zip(direct, results)):
        assert _sha(tmp_path / "direct" / f"h{i}.nc") == _sha(tmp_path / "background" / f"h{i}.nc")
        assert a["sha256"] == b["sha256"] and a["bytes"] == b["bytes"]
        assert history_writer.durable_marker(tmp_path / "background" / f"h{i}.nc").is_file()
        assert b["path"] == str(tmp_path / "background" / f"h{i}.nc")
        from netCDF4 import Dataset
        with Dataset(b["path"]) as data:
            if with_clock:
                clock = data.variables["xtime"]
                assert clock.dimensions == ("Time", "StrLen")
                assert clock[:].tobytes().decode("ascii").strip() == frames[i][0]["valid_time"]
            else:
                assert "xtime" not in data.variables
    receipt = writer.receipt()
    assert receipt["broken"] is None
    assert [row["how"] for row in receipt["frames"]] == ["background"] * 3
    assert all(row["transport"]["mode"] == "mapped-protocol5" for row in receipt["frames"])
    assert not list((tmp_path / "background").glob("hex-history-*"))
    assert writer.ready['io_priority'] == 'idle'
    import importlib.util
    if importlib.util.find_spec("woof") is None or importlib.util.find_spec("woof.timeline") is None:
        return  # no engine ledger (gpuwm 2.8.7 and earlier): the frames above are the test
    events = [json.loads(line) for line in (tmp_path / "background" / "timeline.jsonl").read_text().splitlines()]
    assert len([e for e in events if e['stage'] == 'forecast.output.history.durable']) == 3
    events = [e for e in events if e['stage'] == 'forecast.output.history']
    assert len(events) == 6
    assert all(event["stage"] == "forecast.output.history" and event["product"] == "hex" for event in events)
    assert [event["phase"] for event in events] == ["begin", "end"] * 3
    assert all(event["attrs"]["span_id"] == event["attrs"]["file"] for event in events)


def test_overlapping_fallback_writes_have_distinct_span_ids(tmp_path, monkeypatch):
    pytest.importorskip("woof.timeline.ledger", reason="the engine ships no woof.timeline (woof 2.8.7 and earlier)")
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    path = tmp_path / "concurrent-writer.jsonl"
    monkeypatch.setenv("WOOF_TIME_LEDGER", str(path))
    entered_a, entered_b, release_b = threading.Event(), threading.Event(), threading.Event()

    def write(destination):
        if destination.name.startswith(".a.nc."):
            entered_a.set()
            assert entered_b.wait(60)
        else:
            entered_b.set()
            assert release_b.wait(60)
        destination.write_bytes(b'complete')
        return destination.name

    writer = SimpleNamespace(function=write, receipt_rows=[])
    # Five-second joins measured filesystem/scheduler load rather than span
    # identity in the shared G0 run. Keep the forced overlap and exact event
    # ordering, and wait for each write's Future so failures cannot be hidden.
    with ThreadPoolExecutor(max_workers=2) as workers:
        a = workers.submit(history_writer.BackgroundWriter._fallback,
                           writer, history_writer._Frame(tmp_path / "a.nc", ()))
        try:
            assert entered_a.wait(60)
            b = workers.submit(history_writer.BackgroundWriter._fallback,
                               writer, history_writer._Frame(tmp_path / "b.nc", ()))
            a.result(timeout=60).result()
        finally:
            release_b.set()
        b.result(timeout=60).result()
    assert (tmp_path / "a.nc").read_bytes() == b"complete"
    assert (tmp_path / "b.nc").read_bytes() == b"complete"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events = [e for e in events if e['stage'] == 'forecast.output.history']
    assert [(event["attrs"]["span_id"], event["phase" ]) for event in events] == [
        ("a.nc", "begin"), ("b.nc", "begin"), ("a.nc", "end"), ("b.nc", "end")]


def test_a_writer_refusal_is_raised_in_the_forecast(tmp_path, small_mesh):
    frame = _frame(np.random.default_rng(4), *small_mesh)
    target = tmp_path / "exists.nc"
    target.write_bytes(b"")
    writer = history_writer.BackgroundWriter(proof.write_snapshot_netcdf)
    future = writer.submit(target, *frame)
    with pytest.raises(FileExistsError):
        writer.result(future)
    writer.close()


def test_product_ready_sees_the_closed_durable_native_file(tmp_path, small_mesh):
    from pathlib import Path
    from woof.hex.history_ready import HistoryReady

    seen = []
    def consume(event):
        path = Path(event["path"])
        assert history_writer.durable_marker(path).is_file()
        assert path.stat().st_size == event["bytes"]
        assert _sha(path) == event["sha256"]
        seen.append(event)

    ready = HistoryReady(consume)
    writer = history_writer.BackgroundWriter(proof.write_snapshot_netcdf)
    frame = _frame(np.random.default_rng(6), *small_mesh)
    future = writer.submit(tmp_path / "closed.nc", *frame)
    context = {"step": 30, "label": "2026-10-18_03.00.00",
               "valid_time": "2026-10-18_03:00:00"}
    ready.watch(future, **context)
    try:
        ready.submit(writer.result(future), **context)
    finally:
        writer.close()
        ready.close()
    assert len(seen) == 1 and seen[0]["step"] == 30


def test_a_broken_writer_falls_back_to_the_direct_write(tmp_path, small_mesh):
    frame = _frame(np.random.default_rng(5), *small_mesh)
    proof.write_snapshot_netcdf(tmp_path / "direct.nc", *frame)
    writer = history_writer.BackgroundWriter(proof.write_snapshot_netcdf)
    writer.module_file = "/nowhere/else.py"  # the child refuses a different module file
    first = writer.result(writer.submit(tmp_path / "a.nc", *frame))
    second = writer.result(writer.submit(tmp_path / "b.nc", *frame))
    writer.close()
    assert first["sha256"] == second["sha256"] == _sha(tmp_path / "direct.nc")
    receipt = writer.receipt()
    assert "not /nowhere/else.py" in receipt["broken"]
    assert [row["how"] for row in receipt["frames"]] == ["in-process fallback"] * 2


def test_switched_off_writes_in_process(monkeypatch):
    monkeypatch.setenv(history_writer.ENVIRONMENT, "off")
    assert history_writer.open_writer(proof.write_snapshot_netcdf) is None


def test_capture_worker_defers_broken_writer_recovery_to_forecast(tmp_path, small_mesh, monkeypatch):
    import threading
    from concurrent.futures import Future, ThreadPoolExecutor

    frame = _frame(np.random.default_rng(7), *small_mesh)
    direct = proof.write_snapshot_netcdf(tmp_path / "direct.nc", *frame)
    writer = history_writer.BackgroundWriter(proof.write_snapshot_netcdf)
    calls = []
    fallback = writer._fallback
    def tracked(frame):
        calls.append(threading.get_ident())
        return fallback(frame)
    monkeypatch.setattr(writer, "_fallback", tracked)
    writer.module_file = "/nowhere/capture.py"
    try:
        # Fail the child, retaining its first frame until forecast-side recovery.
        first = writer.submit(tmp_path / "first.nc", *frame, defer_fallback=True)
        first.result(timeout=5)
        assert writer.receipt()["broken"] is not None
        # A later submission sees an already broken writer on a capture worker.
        with ThreadPoolExecutor(max_workers=1) as capture:
            second = capture.submit(writer.submit, tmp_path / "second.nc", *frame,
                                    defer_fallback=True).result(timeout=5)
        assert calls == []
        assert not (tmp_path / "first.nc").exists()
        assert not (tmp_path / "second.nc").exists()
        # A healthy pending write must never block the next capture boundary.
        assert writer.recover_if_needed(Future()) is None
        for future in (first, second):
            recovered = writer.recover_if_needed(future)
            assert recovered["sha256"] == direct["sha256"]
            assert writer.result(future) == recovered  # no repeated fallback/hash
        assert calls == [threading.get_ident()] * 2
        assert [r["how"] for r in writer.receipt()["frames"]] == ["in-process fallback"] * 2
    finally:
        writer.close()


def test_a_transparent_tracer_is_looked_through(tmp_path, small_mesh):
    import functools

    @functools.wraps(proof.write_snapshot_netcdf)
    def traced(*args, **kwargs):
        return proof.write_snapshot_netcdf(*args, **kwargs)

    traced.__prep_cache_transparent__ = True
    writer = history_writer.BackgroundWriter(traced)
    assert writer.function is proof.write_snapshot_netcdf
    writer.close()


def test_only_arrays_the_frame_solely_owns_are_handed_over_uncopied():
    held_elsewhere = np.ones(8, dtype=np.float32)
    keeper = {"buffer": held_elsewhere}  # e.g. a backend's live host buffer
    base = np.arange(16, dtype=np.float32)
    arrays = {
        "fresh": np.zeros(8, dtype=np.float32),
        "shared": held_elsewhere,
        "view": base[:8],
    }
    fresh = arrays["fresh"]
    del fresh  # the frame's own download: only the mapping holds it
    handed = history_writer.private_arrays(arrays)
    assert handed["fresh"] is arrays["fresh"]
    assert handed["shared"] is not keeper["buffer"] and np.array_equal(handed["shared"], keeper["buffer"])
    assert handed["view"].base is None and np.array_equal(handed["view"], base[:8])
    keeper["buffer"][:] = 7.0  # the next step writes into its buffer
    assert np.all(handed["shared"] == 1.0)


def test_a_tracer_around_the_hand_off_does_not_force_copies():
    import functools

    @functools.wraps(history_writer.private_arrays)
    def traced(*args, **kwargs):  # what the start-up tracer wraps every hexcore function in
        return history_writer.private_arrays(*args, **kwargs)

    arrays = {"a": np.zeros(8, dtype=np.float32), "b": np.ones(8, dtype=np.float32)}
    handed = traced(arrays)
    assert all(handed[name] is arrays[name] for name in arrays)
    assert history_writer.handoffs[-1]["copied"] == []


def test_partial_write_never_publishes_history(tmp_path):
    target = tmp_path / "cuda-history.2026-10-07_12.00.00.nc"
    def fail(path):
        path.write_bytes(b"half a frame")
        assert not target.exists()
        raise RuntimeError("interrupted frame")
    with pytest.raises(RuntimeError, match="interrupted frame"):
        history_writer.atomic_write(fail, target)
    assert not target.exists()
    assert not history_writer.durable_marker(target).exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert (tmp_path / ".history-durable-v1").is_file()


def test_sync_failure_after_rename_cannot_announce_frame(tmp_path, monkeypatch):
    target = tmp_path / "frame.nc"
    def fail_directory(path):
        if target.exists():
            raise OSError("directory sync interrupted")
    monkeypatch.setattr(history_writer, "_sync_directory", fail_directory)
    with pytest.raises(OSError, match="sync interrupted"):
        history_writer.atomic_write(lambda p: p.write_bytes(b"complete"), target)
    assert target.is_file()
    assert not history_writer.durable_marker(target).exists()


def test_digest_and_file_sync_overlap_before_publication(tmp_path, monkeypatch):
    import threading

    target = tmp_path / "frame.nc"
    digest_entered, sync_entered = threading.Event(), threading.Event()
    original_sync = history_writer._sync_file

    def write(path, *, _hash_file=True):
        assert not _hash_file
        path.write_bytes(b"complete")
        return {"path": str(path), "bytes": 8}

    def digest(path):
        digest_entered.set()
        assert sync_entered.wait(10)
        assert not target.exists()
        assert not history_writer.durable_marker(target).exists()
        return _sha(path)

    def sync(path):
        sync_entered.set()
        assert digest_entered.wait(10)
        assert not target.exists()
        original_sync(path)

    write.__history_digest__ = digest
    monkeypatch.setattr(history_writer, "_sync_file", sync)
    record = history_writer.atomic_write(write, target)
    assert record["sha256"] == _sha(target)
    assert history_writer.durable_marker(target).is_file()


@pytest.mark.parametrize("failure", ["digest", "sync"])
def test_digest_or_file_sync_failure_never_publishes(tmp_path, monkeypatch, failure):
    import threading

    target = tmp_path / "frame.nc"
    digest_entered, sync_finished, digest_finished = (
        threading.Event(), threading.Event(), threading.Event())

    def write(path, *, _hash_file=True):
        path.write_bytes(b"complete")
        return {"path": str(path), "bytes": 8}

    def digest(path):
        digest_entered.set()
        try:
            assert sync_finished.wait(10)
            assert path.read_bytes() == b"complete"
            if failure == "digest":
                raise OSError("digest interrupted")
            return _sha(path)
        finally:
            digest_finished.set()

    def sync(path):
        assert digest_entered.wait(10)
        sync_finished.set()
        if failure == "sync":
            raise OSError("sync interrupted")

    write.__history_digest__ = digest
    monkeypatch.setattr(history_writer, "_sync_file", sync)
    with pytest.raises(OSError, match=f"{failure} interrupted"):
        history_writer.atomic_write(write, target)
    assert digest_finished.is_set()
    assert not target.exists()
    assert not history_writer.durable_marker(target).exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_destination_is_preserved_even_in_fallback(tmp_path, small_mesh):
    target = tmp_path / "exists.nc"
    target.write_bytes(b"keep this")
    writer = history_writer.BackgroundWriter(proof.write_snapshot_netcdf, max_pending=90)
    assert writer.max_pending == 2
    writer._broken = "forced transport fault"
    with pytest.raises(FileExistsError):
        writer.result(writer.submit(target, *_frame(np.random.default_rng(1), *small_mesh)))
    writer.close()
    writer.close()
    assert target.read_bytes() == b"keep this"
    with pytest.raises(RuntimeError, match="closed"):
        writer.submit(tmp_path / "after.nc")


def _interrupted_writer(path, signal):
    import time
    path.write_bytes(b"incomplete native history")
    signal.write_text(str(path))
    time.sleep(30)


def test_killed_child_leaves_only_a_hidden_unannounced_temporary_frame(tmp_path):
    import time
    path = tmp_path / "cuda-history.2026-10-07_12.00.00.nc"
    signal = tmp_path / "writing"
    writer = history_writer.BackgroundWriter(_interrupted_writer)
    try:
        writer.submit(path, signal)
        deadline = time.monotonic() + 10
        while not signal.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert signal.exists(), writer.receipt()
        writer._child.kill()  # requested crash injection, only this writer child
        writer._child.wait(timeout=5)
        writer.close()
        assert not path.exists()
        assert not history_writer.durable_marker(path).exists()
        assert list(tmp_path.glob(".*.tmp"))
        assert (tmp_path / ".history-durable-v1").is_file()
    finally:
        if writer._child.poll() is None:
            writer._child.kill()
        writer.close()


def test_aliases_within_the_frame_are_copied_before_the_next_step():
    live = np.ones(8, dtype=np.float32)
    arrays = {"first": live, "second": live}
    handed = history_writer.private_arrays(arrays)
    live[:] = 7.0
    assert all(np.all(value == 1.0) for value in handed.values())
    assert history_writer.handoffs[-1]["copied"] == ["first", "second"]


def test_spool_failure_cleans_up_and_preserves_fallback_arrays(tmp_path, small_mesh, monkeypatch):
    spool = tmp_path / "spool"
    spool.mkdir()
    monkeypatch.setenv("WOOF_HEX_HISTORY_SPOOL", str(spool))
    frame = _frame(np.random.default_rng(15), *small_mesh)
    proof.write_snapshot_netcdf(tmp_path / "direct.nc", *frame)

    def interrupted(payload, path):
        path.with_suffix(".buffers").write_bytes(b"incomplete handoff")
        raise OSError("spool interrupted")

    monkeypatch.setattr(history_writer.history_spool, "dump_payload", interrupted)
    writer = history_writer.BackgroundWriter(proof.write_snapshot_netcdf)
    try:
        result = writer.result(writer.submit(tmp_path / "fallback.nc", *frame))
    finally:
        writer.close()
    assert result["sha256"] == _sha(tmp_path / "direct.nc")
    assert "spool interrupted" in writer.receipt()["broken"]
    assert not list(spool.iterdir())
