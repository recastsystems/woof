"""Chained preparation: a streamed boundary source is the precomputed one.

The prepared tree is written as a head (everything the start time makes),
one segment per boundary interval with its ready marker written last, and a
seal that writes the same ``header.json`` and ``proof.json`` as the one-shot
writer.  These tests pin the three claims that make that safe:

* the bytes: every streamed interval, the header's ``content_sha256`` and
  ``setup_fingerprint`` equal the one-shot cache for the same inputs;
* the wait: a consumer blocks on a missing marker with a named reason,
  never reads an array whose marker is absent, and is released when the
  marker lands;
* the ends: a producer failure, a silent producer and a stop each end the
  wait by name, and an unfinished tree is rebuilt by the next preparation.

Every test is CPU-only and needs no device.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core.grid import BaseState, make_vertical_coord
from woof.ingest import boundary_stream
from woof.ingest.boundary_stream import (
    BoundaryProducerFailed, BoundaryProducerSilent, BoundaryStreamError,
    BoundaryStreamStopped, PreparedTreeWriter, StreamedIntervals,
    chained_enabled, read_head, remove_unfinished_tree,
    segment_marker_path, streamed_boundaries, unfinished_tree_reason,
    verify_seal,
)
from woof.ingest.lateral_bc import (
    RationalTimeLaw, StateBoundaryFrames, attach_streaming_lateral_boundaries,
    build_lateral_boundaries, interval_index, _resident_interval,
)
from woof.ingest.prepared_cache import (
    PreparedCacheReader, boundary_schedule, write_prepared_cache,
)
from woof.io.restart import STATE_SETUP_ARRAYS, STATE_SETUP_SCALARS
from woof.state_serialization_contract import (
    lateral_boundary_prefix_identity, setup_fingerprint,
)


FIELDS = ("u", "v", "theta", "phi", "mu")


def _snapshots(count=4, seed=20260928, nz=3, ny=12, nx=14):
    rng = np.random.default_rng(seed)
    return [
        {name: rng.standard_normal((nz, ny, nx)) for name in FIELDS}
        for _ in range(count)
    ]


def _times(count):
    start = datetime(2026, 9, 28, 0, 0, 0)
    return [start + timedelta(hours=n) for n in range(count)]


def _state(boundaries=None):
    state = SimpleNamespace(u=np.arange(12, dtype=np.float32).reshape(3, 2, 2))
    for index, name in enumerate(STATE_SETUP_ARRAYS):
        setattr(state, name, np.array([index], dtype=np.float32))
    for name, value in {
            "mub": None, "p_top": 10_000.0, "cf1": 1.0, "cf2": 2.0,
            "cf3": 3.0, "cfn": 4.0, "cfn1": 5.0, "has_msf": True,
            "rotational": True}.items():
        setattr(state, name, value)
    assert {"mub", "p_top", "cf1", "cf2", "cf3", "cfn", "cfn1", "has_msf",
            "rotational"} == set(STATE_SETUP_SCALARS)
    state.lateral_boundaries = boundaries
    return state


def _initial(boundaries=None):
    coord = make_vertical_coord(2, hybrid_opt=0)
    base = BaseState(
        mub=np.full((2, 2), 90_000.0), p_top=10_000.0,
        pb=np.full((2, 2, 2), 50_000.0), alb=np.full((2, 2, 2), 0.8),
        thb=np.full((2, 2, 2), 290.0), phb=np.zeros((3, 2, 2)),
        terrain_z=np.zeros((2, 2)))
    return SimpleNamespace(
        state=_state(boundaries), coord=coord, base=base,
        surface_pressure=np.full((2, 2), 99_000.0),
        surface_qv=np.full((2, 2), 0.01))


def _met():
    surface = np.ones((2, 2), dtype=np.float32)
    return SimpleNamespace(fields={
        "LANDSEA": surface, "SKINTEMP": 280.0 * surface,
        "SOILT": np.ones((9, 2, 2), dtype=np.float32),
        "SOILW": np.full((9, 2, 2), 0.2, dtype=np.float32),
        "T2": 279.0 * surface,
        "U10": np.ones((2, 3), dtype=np.float32),
        "V10": np.ones((3, 2), dtype=np.float32),
    })


def _frames(snapshots):
    frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    for index, snapshot in enumerate(snapshots):
        frames.add_snapshot(snapshot, index=index)
    return frames


def _assert_same_interval(got, want, label=""):
    assert got.start_seconds == want.start_seconds, label
    assert got.end_seconds == want.end_seconds, label
    assert sorted(got.fields) == sorted(want.fields), label
    for name in want.fields:
        for side in ("west", "east", "south", "north"):
            a = getattr(got.fields[name], side)
            b = getattr(want.fields[name], side)
            assert a.value.dtype == b.value.dtype
            assert a.value.tobytes() == b.value.tobytes(), (label, name, side)
            assert a.tendency.tobytes() == b.tendency.tobytes(), (
                label, name, side)
            assert (a.time_law is None) == (b.time_law is None)
            if a.time_law is not None:
                for coefficient in ("quadratic", "denominator_rate"):
                    assert getattr(a.time_law, coefficient).tobytes() == \
                        getattr(b.time_law, coefficient).tobytes()


PROOF_HEAD = {"schema": "test-proof", "forcing_hours": [0, 1, 2, 3]}


def _chained_tree(tmp_path, snapshots, *, chained=True, name="chained",
                  sealed_forcing_extension=False, stop_after=None,
                  proof_head=PROOF_HEAD):
    """Produce a tree START FIRST: head, then each interval as it exists."""

    times = _times(len(snapshots))
    staging = tmp_path / f".tmp-{name}"
    staging.mkdir()
    output = tmp_path / name
    identity = {"source": "boundary-stream-test"}
    writer = PreparedTreeWriter(
        staging=staging, output_root=output, identity=identity,
        chained=chained, sealed_forcing_extension=sealed_forcing_extension)
    frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(snapshots[0], index=0)
    seconds = [(t - times[0]).total_seconds() for t in times]
    writer.write_head(
        initial_result=_initial(), met=_met(), lbc={
            "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
            "schedule": [[seconds[k], seconds[k + 1]]
                         for k in range(len(times) - 1)],
            "fields": frames.inventory},
        proof_head=proof_head, input_manifest_sha256="0" * 64,
        forcing=frames)
    for index in range(1, len(snapshots)):
        frames.add_snapshot(snapshots[index], index=index)
        writer.write_segment(index - 1, frames.interval(index - 1, times))
        frames.release(index - 1)
        if stop_after is not None and index - 1 == stop_after:
            return writer, output
    receipt = writer.seal_cache()
    writer.publish({**proof_head,
                    "prepared_cache": {"content_sha256":
                                       receipt["content_sha256"]},
                    "boundary_stream": writer.boundary_stream_proof()})
    return writer, output


@pytest.mark.parametrize("chained", [True, False])
def test_source_head_publication_is_observed_once_after_atomic_publish(tmp_path, monkeypatch, chained):
    from woof.mapped_composition import shared_composition_source
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("WOOF_TIME_LEDGER", str(ledger))
    with shared_composition_source("a" * 64, None):
        writer, output = _chained_tree(tmp_path, _snapshots(), chained=chained)
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    heads = [event for event in events if event["stage"] == "prep.head.publish"]
    assert len(heads) == 1
    assert heads[0]["attrs"]["source_key"] == "a" * 64
    assert heads[0]["attrs"]["head_sha256"] == read_head(output)["head_sha256"]
    assert heads[0]["product"] == "ensemble"


def _one_shot(tmp_path, snapshots, *, sealed_forcing_extension=False):
    times = _times(len(snapshots))
    boundaries = _frames(snapshots).build(times)
    initial = _initial(boundaries)
    path = tmp_path / "one-shot"
    receipt = write_prepared_cache(
        path, identity={"source": "boundary-stream-test"},
        initial_result=initial, met=_met(), boundaries=boundaries,
        sealed_forcing_extension=sealed_forcing_extension)
    return path, receipt, boundaries, initial


# ---------------------------------------------------------------------------
# 1. The bytes
# ---------------------------------------------------------------------------


def test_each_interval_equals_the_whole_set_build_bit_for_bit():
    snapshots = _snapshots(5)
    times = _times(5)
    whole = build_lateral_boundaries(
        snapshots, times, spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(snapshots[0], index=0)
    streamed = []
    for index in range(1, len(snapshots)):
        frames.add_snapshot(snapshots[index], index=index)
        streamed.append(frames.interval(index - 1, times))
        frames.release(index - 1)
        # Only two frames are ever held: the start-first order's residency.
        assert len(frames) == 1
    for index, (got, want) in enumerate(zip(streamed, whole.intervals)):
        _assert_same_interval(got, want, index)


def test_a_released_frame_is_named_when_asked_for_again():
    snapshots = _snapshots(3)
    times = _times(3)
    frames = _frames(snapshots)
    frames.interval(0, times)
    frames.release(0)
    with pytest.raises(ValueError, match="released"):
        frames.interval(0, times)
    with pytest.raises(ValueError, match="added twice"):
        frames.add_snapshot(snapshots[0], index=0)


@pytest.mark.parametrize("sealed_forcing_extension", [False, True])
def test_the_chained_header_is_the_one_shot_header(tmp_path,
                                                   sealed_forcing_extension):
    snapshots = _snapshots(4)
    path, receipt, boundaries, initial = _one_shot(
        tmp_path, snapshots,
        sealed_forcing_extension=sealed_forcing_extension)
    writer, output = _chained_tree(
        tmp_path, snapshots,
        sealed_forcing_extension=sealed_forcing_extension)
    one = json.loads((path / "header.json").read_text(encoding="utf-8"))
    two = json.loads((output / "prepared-cache" / "header.json").read_text(
        encoding="utf-8"))
    assert two["content_sha256"] == one["content_sha256"] \
        == receipt["content_sha256"]
    assert two["metadata"]["setup_fingerprint"] \
        == one["metadata"]["setup_fingerprint"] \
        == setup_fingerprint(initial.state)
    assert two["arrays"] == one["arrays"]
    if sealed_forcing_extension:
        assert two["metadata"]["lateral_boundary_prefix"] \
            == lateral_boundary_prefix_identity(initial.state)
    for spec in one["arrays"].values():
        assert (output / "prepared-cache" / spec["file"]).read_bytes() \
            == (path / spec["file"]).read_bytes()
    # The sealed chained cache is an ordinary sealed cache.
    reader = PreparedCacheReader(
        output / "prepared-cache",
        expected_identity={"source": "boundary-stream-test"})
    assert reader.verify_all()["content_sha256"] == one["content_sha256"]


def test_a_rational_time_law_streams_like_the_one_shot_cache(tmp_path):
    snapshots = _snapshots(3)
    times = _times(3)
    whole = _frames(snapshots).build(times)

    def lawful(interval):
        fields = {}
        for name, field in interval.fields.items():
            def with_law(side):
                return replace(side, time_law=RationalTimeLaw(
                    np.full(side.value.shape, 1e-4),
                    np.full(side.value.shape, 1e-5)))
            fields[name] = type(field)(
                *(with_law(getattr(field, s))
                  for s in ("west", "east", "south", "north")))
        return replace(interval, fields=fields)

    boundaries = replace(whole, intervals=tuple(
        lawful(interval) for interval in whole.intervals))
    initial = _initial(boundaries)
    one = write_prepared_cache(
        tmp_path / "one", identity={"source": "law"}, initial_result=initial,
        met=_met(), boundaries=boundaries, sealed_forcing_extension=True)
    staging = tmp_path / ".tmp-law"
    staging.mkdir()
    writer = PreparedTreeWriter(
        staging=staging, output_root=tmp_path / "law",
        identity={"source": "law"}, chained=True,
        sealed_forcing_extension=True)
    writer.write_head(initial_result=_initial(), met=_met(),
                      lbc=boundary_schedule(boundaries), proof_head={},
                      forcing=_frames(snapshots))
    for index, interval in enumerate(boundaries.intervals):
        writer.write_segment(index, interval)
    assert writer.seal_cache()["content_sha256"] == one["content_sha256"]
    head = read_head(tmp_path / "law")
    stream = StreamedIntervals(tmp_path / "law", head=head)
    for index, interval in enumerate(boundaries.intervals):
        _assert_same_interval(stream[index], interval, index)


def test_the_streamed_reader_returns_the_written_intervals(tmp_path):
    snapshots = _snapshots(4)
    whole = _frames(snapshots).build(_times(4))
    writer, output = _chained_tree(tmp_path, snapshots)
    head = read_head(output, expected_sha256=writer.head_sha256)
    stream = StreamedIntervals(output, head=head)
    assert len(stream) == 3
    assert stream.bounds == ((0.0, 3600.0), (3600.0, 7200.0),
                             (7200.0, 10800.0))
    for index, interval in enumerate(whole.intervals):
        _assert_same_interval(stream[index], interval, index)
        # One object per interval: the device slot reloads by identity.
        assert stream[index] is stream[index]
    assert verify_seal(output, head=head,
                       consumed=stream.consumed_markers())["head_sha256"] \
        == writer.head_sha256


def test_the_interval_search_reads_only_the_schedule(tmp_path):
    snapshots = _snapshots(4)
    writer, output = _chained_tree(tmp_path, snapshots, stop_after=0)
    stream = StreamedIntervals(output, head=read_head(output))
    # Interval 2 is not prepared; finding it must not wait for it.
    assert interval_index(stream, 9000.0) == 2
    assert interval_index(stream, 10800.0) == 2
    assert stream.ready_prefix() == 1


def test_the_proof_is_the_head_proof_plus_the_seal_keys(tmp_path):
    snapshots = _snapshots(3)
    writer, output = _chained_tree(tmp_path, snapshots, stop_after=1)
    writer.seal_cache()
    with pytest.raises(RuntimeError, match="forcing_hours"):
        writer.publish({"schema": "test-proof", "forcing_hours": [0, 1]})
    with pytest.raises(ValueError, match="seal-only"):
        PreparedTreeWriter(
            staging=tmp_path / ".x", output_root=tmp_path / "x",
            identity={}, chained=False).write_head(
                initial_result=_initial(), met=_met(), lbc=None,
                proof_head={"prepared_cache": {}})


def test_an_unchained_writer_publishes_the_same_tree_at_the_seal(tmp_path):
    snapshots = _snapshots(4)
    chained_writer, chained = _chained_tree(tmp_path, snapshots, name="a")
    seen = []
    staging = tmp_path / ".tmp-b"
    staging.mkdir()
    times = _times(4)
    writer = PreparedTreeWriter(
        staging=staging, output_root=tmp_path / "b",
        identity={"source": "boundary-stream-test"}, chained=False)
    frames = _frames(snapshots)
    writer.write_head(initial_result=_initial(), met=_met(),
                      lbc=boundary_schedule(frames.build(times)),
                      proof_head=PROOF_HEAD,
                      input_manifest_sha256="0" * 64)
    for index in range(3):
        seen.append((tmp_path / "b").exists())
        writer.write_segment(index, frames.interval(index, times))
    receipt = writer.seal_cache()
    assert not (tmp_path / "b").exists()
    writer.publish({**PROOF_HEAD, "prepared_cache": {
        "content_sha256": receipt["content_sha256"]},
        "boundary_stream": writer.boundary_stream_proof()})
    assert seen == [False, False, False]
    # Same head digest either way: the chaining decision is not identity.
    assert writer.head_sha256 == chained_writer.head_sha256

    def inventory(root):
        return sorted(str(path.relative_to(root)).replace("\\", "/")
                      for path in root.rglob("*") if path.is_file()
                      and path.name != "producer.json")

    assert inventory(tmp_path / "b") == inventory(chained)
    for name in inventory(chained):
        if name.endswith(".npy") or name.startswith("boundary-stream/segments"):
            assert (tmp_path / "b" / name).read_bytes() \
                == (chained / name).read_bytes(), name


def test_the_switch_reads_the_environment(monkeypatch):
    monkeypatch.delenv("WOOF_CHAINED_PREP", raising=False)
    assert chained_enabled() is boundary_stream.CHAINED_DEFAULT
    # On since the GPU proof runs matched; the variable is the off switch.
    assert boundary_stream.CHAINED_DEFAULT is True
    for value in ("0", "off", "false", "NO"):
        monkeypatch.setenv("WOOF_CHAINED_PREP", value)
        assert not chained_enabled()
    monkeypatch.setenv("WOOF_CHAINED_PREP", "1")
    assert chained_enabled()


# ---------------------------------------------------------------------------
# 2. The wait
# ---------------------------------------------------------------------------


def test_a_consumer_waits_for_its_marker_with_the_named_reason(tmp_path):
    snapshots = _snapshots(4)
    times = _times(4)
    writer, output = _chained_tree(tmp_path, snapshots, stop_after=0)
    waits = []
    stream = StreamedIntervals(output, head=read_head(output),
                               on_wait=waits.append, poll_seconds=0.01)
    assert stream[0] is not None
    got = {}

    def consume():
        got["interval"] = stream[1]

    reader = threading.Thread(target=consume)
    reader.start()
    deadline = time.monotonic() + 5
    while not waits and time.monotonic() < deadline:
        time.sleep(0.01)
    assert waits and waits[0]["reason"] == (
        "boundary interval 1 (3600 s to 7200 s) is not prepared yet")
    assert reader.is_alive()
    frames = _frames(snapshots)
    writer.write_segment(1, frames.interval(1, times))
    reader.join(5)
    assert not reader.is_alive()
    _assert_same_interval(got["interval"], frames.interval(1, times))
    assert waits[-1] is None
    assert [index for index, _ in stream.waits] == [1]


def test_streamed_attach_reloads_each_interval_like_the_eager_series(
        tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        float32=np.float32,
        asarray=lambda value, dtype=None: np.asarray(value, dtype=dtype)))
    snapshots = _snapshots(4, ny=12, nx=14)
    whole = _frames(snapshots).build(_times(4))
    writer, output = _chained_tree(tmp_path, snapshots)
    boundaries = streamed_boundaries(output, head=read_head(output))
    buffers = {}

    def scratch(shape, slot):
        buffer = buffers.get(slot)
        if buffer is None:
            buffer = buffers[slot] = np.zeros(tuple(shape), np.float32)
        return buffer

    state = SimpleNamespace(_scratch=buffers, scratch=scratch)
    attach_streaming_lateral_boundaries(state, boundaries)
    device = state._lateral_boundary_device.intervals[0]
    for elapsed, index in ((0.0, 0), (3600.0, 1), (9000.0, 2)):
        assert _resident_interval(state, boundaries.interval_at(elapsed)) \
            is device
        np.testing.assert_array_equal(
            device.fields["theta"].north.tendency,
            np.asarray(whole.intervals[index].fields["theta"].north.tendency,
                       np.float32))


# ---------------------------------------------------------------------------
# 3. Producer failure and silence
# ---------------------------------------------------------------------------


def test_a_silent_producer_ends_the_wait_by_name(tmp_path):
    snapshots = _snapshots(4)
    writer, output = _chained_tree(tmp_path, snapshots, stop_after=0)
    writer._stop_heartbeat(final="producing")
    beat_path = output / "boundary-stream" / "producer.json"
    beat = json.loads(beat_path.read_text(encoding="utf-8"))
    # Another process's producer (a heartbeat from this process's own
    # thread proves nothing about liveness, and is not judged).
    beat.update(updated_epoch=time.time() - 3600.0, times_built=2,
                pid=beat["pid"] + 1)
    beat_path.write_text(json.dumps(beat), encoding="utf-8")
    stream = StreamedIntervals(output, head=read_head(output),
                               poll_seconds=0.01)
    with pytest.raises(BoundaryProducerSilent,
                       match=r"silent for \d+ s after building time 2"):
        stream[1]


# ---------------------------------------------------------------------------
# 4. Stop, and the unfinished tree
# ---------------------------------------------------------------------------


def test_a_stop_makes_the_producer_exit_unsealed(tmp_path):
    snapshots = _snapshots(4)
    times = _times(4)
    writer, output = _chained_tree(tmp_path, snapshots, stop_after=0)
    StreamedIntervals(output, head=read_head(output)).stop(
        "the forecast was interrupted")
    frames = _frames(snapshots)
    with pytest.raises(BoundaryStreamStopped,
                       match="the forecast was interrupted"):
        writer.write_segment(1, frames.interval(1, times))
    writer.fail(BoundaryStreamStopped("stopped"))
    assert not (output / "proof.json").exists()
    assert unfinished_tree_reason(output) is not None
    lines = []
    assert remove_unfinished_tree(output, log=lines.append)
    assert not output.exists()
    assert "building it again" in lines[0]


def test_the_existing_folder_refusal_leaves_an_unfinished_tree_to_the_rebuild(
        tmp_path):
    """``woof prep`` checks an existing --output-root before it writes
    anything; an unfinished chained tree is the preparer's to remove and
    rebuild, so that check must not refuse it, while a live or complete
    tree is still refused."""

    from woof.ingest.source_coverage import existing_output_root_refusal

    snapshots = _snapshots(4)
    times = _times(4)
    writer, output = _chained_tree(tmp_path, snapshots, stop_after=0)
    StreamedIntervals(output, head=read_head(output)).stop("interrupted")
    with pytest.raises(BoundaryStreamStopped):
        writer.write_segment(1, _frames(snapshots).interval(1, times))
    writer.fail(BoundaryStreamStopped("stopped"))
    assert unfinished_tree_reason(output) is not None
    assert existing_output_root_refusal(output) is None
    writer2, live = _chained_tree(tmp_path, snapshots, stop_after=0,
                                  name="live")
    assert existing_output_root_refusal(live) is not None
    writer2._stop_heartbeat(final="producing")
    _, complete = _chained_tree(tmp_path, snapshots, name="done")
    assert existing_output_root_refusal(complete) is not None


def test_a_chained_producer_outlives_the_reader_of_its_console(
        tmp_path, monkeypatch):
    """The process reading a producer's console (``woof go``, which also
    hosts the forecast) can die before the seal.  The producer's next line
    must not fail the preparation: a retry reuses it and a checkpoint
    written before the seal resumes only on it."""

    import sys

    class _ReaderGone:
        def write(self, text):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(sys, "stdout", _ReaderGone())
    monkeypatch.setattr(sys, "stderr", _ReaderGone())
    writer, output = _chained_tree(tmp_path, _snapshots(3))
    print("a line nobody reads", flush=True)
    assert sys.stdout.reader_gone
    assert (output / "proof.json").exists()
    assert not (output / "boundary-stream" / "failed.json").exists()


def test_a_changed_segment_after_consumption_fails_the_seal_check(tmp_path):
    snapshots = _snapshots(3)
    writer, output = _chained_tree(tmp_path, snapshots)
    head = read_head(output)
    stream = StreamedIntervals(output, head=head)
    list(stream)
    consumed = stream.consumed_markers()
    consumed[1] = {**consumed[1], "payload_bytes": 0}
    with pytest.raises(BoundaryStreamError, match="changed after"):
        verify_seal(output, head=head, consumed=consumed)
    with pytest.raises(BoundaryStreamError, match="pinned head"):
        read_head(output, expected_sha256="f" * 64)


# ---------------------------------------------------------------------------
# 5. The orchestration every chain uses
# ---------------------------------------------------------------------------


def test_the_forecast_starts_at_the_head_while_preparation_continues(
        tmp_path):
    snapshots = _snapshots(4)
    times = _times(4)
    released = threading.Event()
    seen = {}

    def prepare():
        staging = tmp_path / ".tmp-run"
        staging.mkdir()
        writer = PreparedTreeWriter(
            staging=staging, output_root=tmp_path / "run",
            identity={"source": "boundary-stream-test"}, chained=True)
        frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1,
                                     relax_zone=4)
        frames.add_snapshot(snapshots[0], index=0)
        writer.write_head(initial_result=_initial(), met=_met(),
                          lbc=boundary_schedule(_frames(snapshots).build(times)),
                          proof_head=PROOF_HEAD, forcing=frames)
        assert released.wait(10), "the forecast never started at the head"
        for index in range(1, 4):
            frames.add_snapshot(snapshots[index], index=index)
            writer.write_segment(index - 1, frames.interval(index - 1, times))
        writer.seal_cache()
        writer.publish({**PROOF_HEAD,
                        "boundary_stream": writer.boundary_stream_proof()})
        return "sealed"

    def forecast(head):
        seen["head"] = head
        seen["sealed_at_start"] = (tmp_path / "run" / "proof.json").exists()
        released.set()
        stream = StreamedIntervals(tmp_path / "run",
                                   head=read_head(tmp_path / "run"),
                                   poll_seconds=0.01)
        return len(list(stream))

    heads = []
    result = boundary_stream.run_chained(
        prepared_root=tmp_path / "run", prepare=prepare, forecast=forecast,
        on_head=heads.append, poll_seconds=0.01)
    assert result == ("sealed", 3)
    assert seen["head"] == heads[0] and seen["sealed_at_start"] is False


def test_a_tree_published_sealed_is_bound_by_its_proof(tmp_path):
    calls = []
    result = boundary_stream.run_chained(
        prepared_root=tmp_path / "run",
        prepare=lambda: _chained_tree(tmp_path, _snapshots(3), chained=False,
                                      name="run")[1],
        forecast=lambda head: calls.append(head) or "ran",
        poll_seconds=0.01)
    assert calls == [None] and result[1] == "ran"


def test_a_preparation_that_fails_before_its_head_runs_no_forecast(tmp_path):
    calls = []

    def prepare():
        raise ValueError("the source could not be decoded")

    with pytest.raises(ValueError, match="could not be decoded"):
        boundary_stream.run_chained(
            prepared_root=tmp_path / "run", prepare=prepare,
            forecast=calls.append, poll_seconds=0.01)
    assert calls == []


def _producer(tmp_path, snapshots, *, gate=None, name="run"):
    """A chained producer for run_chained: ``gate`` holds it after its head."""

    times = _times(len(snapshots))
    outcome = {}

    def prepare():
        staging = tmp_path / f".tmp-{name}"
        staging.mkdir()
        writer = PreparedTreeWriter(
            staging=staging, output_root=tmp_path / name,
            identity={"source": "boundary-stream-test"}, chained=True)
        frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1,
                                     relax_zone=4)
        frames.add_snapshot(snapshots[0], index=0)
        try:
            writer.write_head(
                initial_result=_initial(), met=_met(),
                lbc=boundary_schedule(_frames(snapshots).build(times)),
                proof_head=PROOF_HEAD, forcing=frames)
            outcome["head"] = writer.head_sha256
            if gate is not None:
                assert gate.wait(10), "the forecast never released the gate"
            for index in range(1, len(snapshots)):
                frames.add_snapshot(snapshots[index], index=index)
                writer.write_segment(index - 1,
                                     frames.interval(index - 1, times))
            writer.seal_cache()
            writer.publish({**PROOF_HEAD,
                            "boundary_stream": writer.boundary_stream_proof()})
            outcome["sealed"] = True
            return "sealed"
        except BaseException as error:
            outcome["error"] = error
            writer.fail(error)
            raise

    return prepare, outcome


def test_a_failed_forecast_is_said_at_once_and_stops_its_producer(
        tmp_path, capsys):
    # A forecast that failed 25 s after its preflight used to leave its
    # producer building every later forcing time (19-24 min on test host/test host,
    # 2026-10-21) behind waiting:preparation before the error was raised.
    gate = threading.Event()
    prepare, outcome = _producer(tmp_path, _snapshots(4), gate=gate)

    def forecast(head):
        threading.Timer(0.2, gate.set).start()
        raise FloatingPointError("the model went unstable")

    with pytest.raises(FloatingPointError, match="unstable"):
        boundary_stream.run_chained(
            prepared_root=tmp_path / "run", prepare=prepare,
            forecast=forecast, poll_seconds=0.01)
    stop_file = tmp_path / "run" / "boundary-stream" / "stop.json"
    assert json.loads(stop_file.read_text(encoding="utf-8"))["reason"]         .startswith("the forecast failed: FloatingPointError")
    assert isinstance(outcome["error"], BoundaryStreamStopped)
    assert not outcome.get("sealed")
    assert not (tmp_path / "run" / "proof.json").exists()
    said = capsys.readouterr().err
    assert ("the forecast failed (FloatingPointError: the model went "
            "unstable); stopping the preparation") in said


@pytest.mark.parametrize("stop", ["go", "stage"])
def test_a_stop_raised_as_an_exception_stops_its_producer(tmp_path, capsys,
                                                          stop):
    # Ctrl-C reaches a woof go chain as GoInterrupted and a run-plan
    # chain as a stage that answered SIGINT with 130: both are ordinary
    # exceptions, and both are the user's stop, not a failed forecast.
    from woof.go_cli import GoInterrupted
    from woof.runplan import StageExitError

    error = (GoInterrupted("forecast", 4321) if stop == "go"
             else StageExitError("forecast", 130))
    gate = threading.Event()
    prepare, outcome = _producer(tmp_path, _snapshots(4), gate=gate)

    def forecast(head):
        threading.Timer(0.2, gate.set).start()
        raise error

    with pytest.raises(type(error)):
        boundary_stream.run_chained(
            prepared_root=tmp_path / "run", prepare=prepare,
            forecast=forecast, poll_seconds=0.01)
    stop_file = tmp_path / "run" / "boundary-stream" / "stop.json"
    assert json.loads(stop_file.read_text(encoding="utf-8"))["reason"] \
        .startswith(f"the forecast ended: {type(error).__name__}")
    assert isinstance(outcome["error"], BoundaryStreamStopped)
    assert not outcome.get("sealed")
    assert not (tmp_path / "run" / "proof.json").exists()
    assert "the forecast failed" not in capsys.readouterr().err


def test_a_retry_never_binds_the_previous_attempts_head(tmp_path):
    # Attempt 1 left a failed head in the output root; the retry's
    # preparation moves it aside and publishes its own.
    snapshots = _snapshots(4)
    old_writer, output = _chained_tree(tmp_path, snapshots, stop_after=0,
                                       name="run")
    old_writer.fail(ValueError("attempt 1 failed"))
    old_head = old_writer.head_sha256
    # The retry's head differs from attempt 1's only in time; the digest
    # covers the basis, so the test tells them apart by the tree bound.
    moved = threading.Event()

    def retry():
        time.sleep(0.3)
        output.rename(tmp_path / "run.superseded")
        moved.set()
        writer, _ = _chained_tree(tmp_path, snapshots, name="run")
        return writer.head_sha256

    seen = []

    def forecast(head):
        seen.append((head, moved.is_set()))
        return "ran"

    prepared, ran = boundary_stream.run_chained(
        prepared_root=output, prepare=retry, forecast=forecast,
        poll_seconds=0.01)
    assert ran == "ran" and len(seen) == 1
    head, after_move = seen[0]
    assert after_move, "the forecast was bound before the retry's head"
    assert head is None or head == prepared
    assert old_head  # attempt 1's head existed and was never bound


def test_a_same_process_producer_that_dies_silently_ends_the_wait(tmp_path):
    # The disk filled: neither a segment nor failed.json could be written.
    gate = threading.Event()
    snapshots = _snapshots(4)
    times = _times(4)
    ended = {}

    def prepare():
        staging = tmp_path / ".tmp-run"
        staging.mkdir()
        writer = PreparedTreeWriter(
            staging=staging, output_root=tmp_path / "run",
            identity={"source": "boundary-stream-test"}, chained=True)
        frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1,
                                     relax_zone=4)
        frames.add_snapshot(snapshots[0], index=0)
        writer.write_head(initial_result=_initial(), met=_met(),
                          lbc=boundary_schedule(_frames(snapshots).build(times)),
                          proof_head=PROOF_HEAD, forcing=frames)
        gate.wait(10)
        writer._stop_heartbeat(final="producing")
        raise OSError(28, "No space left on device")

    def forecast(head):
        stream = StreamedIntervals(tmp_path / "run",
                                   head=read_head(tmp_path / "run"),
                                   poll_seconds=0.01)
        gate.set()
        try:
            return stream[0]
        except BoundaryProducerFailed as error:
            ended["reason"] = str(error)
            raise

    # run_chained also writes failed.json for the dead producer, so the
    # wait ends by either road; the deadline proves it ended at all.
    started = time.monotonic()
    with pytest.raises(BoundaryProducerFailed):
        boundary_stream.run_chained(
            prepared_root=tmp_path / "run", prepare=prepare,
            forecast=forecast, poll_seconds=0.01)
    assert time.monotonic() - started < 30
    assert "interval 0" in ended["reason"] or "No space" in ended["reason"]


def test_the_dead_producer_verdict_names_the_interval(tmp_path):
    writer, output = _chained_tree(tmp_path, _snapshots(4), stop_after=0)
    writer._stop_heartbeat(final="producing")
    stream = StreamedIntervals(output, head=read_head(output),
                               poll_seconds=0.01)
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    key = boundary_stream._root_key(output)
    boundary_stream._LOCAL_PRODUCERS[key] = dead
    try:
        with pytest.raises(BoundaryProducerFailed,
                           match="ended without preparing interval 2"):
            stream.require(2)
        assert stream.require(0)["index"] == 0  # a ready marker still reads
    finally:
        del boundary_stream._LOCAL_PRODUCERS[key]


def test_a_failed_heartbeat_state_ends_the_wait(tmp_path):
    writer, output = _chained_tree(tmp_path, _snapshots(4), stop_after=0)
    writer._stop_heartbeat(final="failed")
    stream = StreamedIntervals(output, head=read_head(output),
                               poll_seconds=0.01)
    with pytest.raises(BoundaryProducerFailed, match="failed after building"):
        stream.require(2)


def test_a_failed_tree_is_never_bound(tmp_path):
    writer, output = _chained_tree(tmp_path, _snapshots(4), stop_after=0)
    assert boundary_stream.bind_head(output, writer.head_sha256)
    writer.fail(ValueError("decode failed"))
    with pytest.raises(BoundaryStreamError, match="will never be sealed"):
        boundary_stream.bind_head(output, writer.head_sha256)


@pytest.mark.parametrize("workers", [1, 3])
def test_the_streamed_loop_writes_what_the_route_loop_wrote(tmp_path, request,
                                                          workers):
    def stop_writer(writer):
        heartbeat = writer._heartbeat
        writer._stop_heartbeat(final="producing")
        if heartbeat is not None:
            heartbeat.join(timeout=5)
            assert not heartbeat.is_alive()

    snapshots = _snapshots(4)
    times = _times(4)
    manual_writer, _ = _chained_tree(tmp_path, snapshots, stop_after=2,
                                     name="manual")
    request.addfinalizer(lambda: stop_writer(manual_writer))
    manual = manual_writer.seal_cache()
    staging = tmp_path / ".tmp-loop"
    staging.mkdir()
    writer = PreparedTreeWriter(
        staging=staging, output_root=tmp_path / "loop",
        identity={"source": "boundary-stream-test"}, chained=True)
    request.addfinalizer(lambda: stop_writer(writer))
    frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(snapshots[0], index=0)
    seconds = [(t - times[0]).total_seconds() for t in times]
    writer.write_head(
        initial_result=_initial(), met=_met(), lbc={
            "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
            "schedule": [[seconds[k], seconds[k + 1]] for k in range(3)],
            "fields": frames.inventory},
        proof_head=PROOF_HEAD, input_manifest_sha256="0" * 64,
        forcing=frames)
    released = []

    class _Frames:
        def __getattr__(self, name):
            return getattr(frames, name)

        def add_state(self, state, index):
            frames.add_snapshot(state, index=index)

    # All concurrent builds must start before the first can finish.  The
    # writer must still publish snapshots and segments in index order.
    from threading import Barrier
    started = Barrier(workers)

    def build(k):
        if workers > 1:
            started.wait(timeout=5)
        return None, SimpleNamespace(state=snapshots[k])

    writer.stream_forcing_times(
        count=4, forcing=_Frames(), times=times,
        build_forcing_time=build, workers=workers,
        release=lambda: released.append(True))
    receipt = writer.seal_cache()
    assert len(released) == 3 and writer.times_built == 3
    assert receipt["content_sha256"] == manual["content_sha256"]


def test_a_preparation_published_sealed_says_why(monkeypatch, capsys):
    for kind, reason in boundary_stream.SEALED_REASONS.items():
        monkeypatch.delenv(boundary_stream.CHAINED_ENV, raising=False)
        boundary_stream.say_prepared_sealed(kind)
        assert capsys.readouterr().err == (
            f"prepare: {reason}; the forecast starts after preparation\n")
        # Turned off for diagnosis, nothing was expected to start early.
        monkeypatch.setenv(boundary_stream.CHAINED_ENV, "0")
        boundary_stream.say_prepared_sealed(kind)
        assert capsys.readouterr().err == ""
    with pytest.raises(KeyError):
        boundary_stream.say_prepared_sealed("no-such-route")


def test_every_route_that_prepares_sealed_says_so_where_it_builds_forcing():
    # Source-level: these routes need a decoded chain and a device to
    # reach for real.  Each row is said on every route that builds its
    # forcing that way, so a tree or met_em run that starts after its
    # preparation says why, as a declined chained route does.
    root = Path(__file__).resolve().parents[1]
    sayers = {
        # No tree row: a mapped tree chains on either backend (its seal
        # re-reads the start states from the head, so the CUDA row is
        # retired) and its forecast starts on its head (A136 L6 retired the
        # domain_tree_forecast row), a GFS-series tree (go) chains the same
        # way (A136 L7b retired the gfs_domain_tree row), and a native HRRR
        # tree chains on its root preparation's head (A136 L7c retired the
        # domain_tree row).
        "met_em": ("woof/metem_forecast.py",),
        "experiment_run": ("woof/runtime.py",),
        # A single native HRRR domain chains (A136 L7c retired the
        # native_hrrr row); with a water overlay it declines by the
        # overlay's row, as the ERA5 route does.
        "water_overlay": ("tools/hrrr_single_domain_benchmark.py",),
    }
    assert set(sayers) == set(boundary_stream.SEALED_REASONS)
    assert "native_hrrr" not in boundary_stream.SEALED_REASONS
    for kind, paths in sayers.items():
        for path in paths:
            source = (root / path).read_text(encoding="utf-8")
            assert f'say_prepared_sealed("{kind}")' in source, (
                f"{path} builds its forcing sealed without saying why")
    era5 = (root / "woof/era5_direct.py").read_text(encoding="utf-8")
    assert 'writer.decline_chaining(SEALED_REASONS["water_overlay"])' in era5


def test_host_ram_is_priced_for_every_producer(monkeypatch, tmp_path):
    gib = 1 << 30
    fits = boundary_stream.host_admission(
        forecast_bytes=4 * gib, producer_bytes=4 * gib,
        available_bytes=10 * gib)
    assert fits["admitted"]
    refused = boundary_stream.host_admission(
        forecast_bytes=6 * gib, producer_bytes=4 * gib,
        available_bytes=10 * gib)
    assert refused["reason"] == (
        "chained preparation not admitted: forecast 6.00 GiB + "
        "preparation 4.00 GiB > 9.00 GiB of available host RAM")
    monkeypatch.setattr(boundary_stream, "_host_available", lambda: None)
    unmeasured = boundary_stream.host_admission(forecast_bytes=1)
    assert not unmeasured["admitted"]
    assert "could not be measured" in unmeasured["reason"]
    # A head that does not fit is published at its seal.
    monkeypatch.setattr(boundary_stream, "_host_available", lambda: 1)
    writer, output = _chained_tree(tmp_path, _snapshots(3), stop_after=0,
                                   name="tight")
    assert writer.chained is False and not output.exists()
    assert writer.head["decision"]["host"]["admitted"] is False


def _host_priced(monkeypatch, *, available):
    """A machine with ``available`` bytes of RAM and a producer that
    needs nothing on top of what it already holds."""

    monkeypatch.setattr(boundary_stream, "_host_available", lambda: available)
    monkeypatch.setattr(boundary_stream, "process_memory_bytes",
                        lambda: (1 << 30, 1 << 30))


def _large_snapshots():
    # Frames large enough that the boundary series is megabytes and the
    # head's arrays are a few kilobytes.
    return _snapshots(4, nz=20, ny=60, nx=70)


def test_host_ram_prices_the_head_the_boundary_series_and_the_process(
        monkeypatch, tmp_path):
    _host_priced(monkeypatch, available=64 << 30)
    writer, output = _chained_tree(tmp_path, _large_snapshots())
    host = writer.head["decision"]["host"]
    parts = host["forecast_host_parts"]
    assert host["admitted"] and writer.chained
    assert parts["head_payload_bytes"] == \
        read_head(output)["basis"]["cache"]["payload_bytes"]
    # The series is what the segments wrote and what the forecast's
    # reader keeps: every interval it loads stays loaded for the run.
    written = sum(
        json.loads(segment_marker_path(output, k).read_text(
            encoding="utf-8"))["payload_bytes"] for k in range(3))
    stream = StreamedIntervals(output, head=read_head(output))
    held = sum(
        getattr(getattr(interval.fields[name], side), part).nbytes
        for interval in stream for name in interval.fields
        for side in ("west", "east", "south", "north")
        for part in ("value", "tendency"))
    assert parts["boundary_series_bytes"] == written == held > 0
    assert parts["process_floor_bytes"] \
        == boundary_stream.FORECAST_HOST_FLOOR_BYTES
    assert host["forecast_host_bytes"] == parts["total_bytes"] == (
        parts["head_payload_bytes"] + parts["boundary_series_bytes"]
        + parts["process_floor_bytes"])


def test_a_host_that_holds_the_head_but_not_the_forecast_declines(
        monkeypatch, tmp_path):
    # A machine with 1 GiB available holds this head's arrays many times
    # over, and not the forecast process that reads them.
    _host_priced(monkeypatch, available=1 << 30)
    writer, output = _chained_tree(tmp_path, _large_snapshots(),
                                   stop_after=0, name="small")
    host = writer.head["decision"]["host"]
    assert host["forecast_host_parts"]["head_payload_bytes"] \
        < host["host_budget_bytes"]
    assert not host["admitted"] and writer.chained is False
    assert not output.exists()
    assert host["reason"].startswith(
        "chained preparation not admitted: forecast 2.")
    # Room for the head and the process, not for the boundary series.
    parts = host["forecast_host_parts"]
    tight = int((parts["head_payload_bytes"] + parts["process_floor_bytes"]
                 + parts["boundary_series_bytes"] // 2) / 0.9) + 1
    _host_priced(monkeypatch, available=tight)
    writer, output = _chained_tree(tmp_path, _large_snapshots(),
                                   stop_after=0, name="tight")
    assert not writer.head["decision"]["host"]["admitted"]
    assert writer.chained is False and not output.exists()
    _host_priced(monkeypatch, available=tight
                 + parts["boundary_series_bytes"])
    writer, output = _chained_tree(tmp_path, _large_snapshots(),
                                   stop_after=0, name="fits")
    assert writer.head["decision"]["host"]["admitted"] and output.exists()
    writer._stop_heartbeat(final="producing")


def test_a_chained_head_without_its_forcing_is_not_admitted(
        monkeypatch, tmp_path):
    _host_priced(monkeypatch, available=64 << 30)
    snapshots = _snapshots(3)
    staging = tmp_path / ".tmp-unpriced"
    staging.mkdir()
    writer = PreparedTreeWriter(
        staging=staging, output_root=tmp_path / "unpriced",
        identity={"source": "boundary-stream-test"}, chained=True)
    writer.write_head(initial_result=_initial(), met=_met(),
                      lbc=boundary_schedule(_frames(snapshots).build(
                          _times(3))),
                      proof_head=PROOF_HEAD)
    host = writer.head["decision"]["host"]
    assert not host["admitted"] and writer.chained is False
    assert "could not be measured" in host["reason"]
    assert host["forecast_host_parts"]["boundary_series_bytes"] is None


# ---------------------------------------------------------------------------
# 6. Admission: the forecast and its producer on one card
# ---------------------------------------------------------------------------


@pytest.fixture
def _forecast_of(monkeypatch):
    import woof.core.preflight as preflight

    def set_bytes(value):
        monkeypatch.setattr(
            preflight, "admission_estimate",
            lambda experiment, machine=None, source=None: SimpleNamespace(
                peak_envelope_bytes=value))
    return set_bytes


def test_a_gpu_producer_is_admitted_only_when_both_fit(_forecast_of,
                                                       tmp_path):
    gib = 1 << 30
    _forecast_of(10 * gib)
    fits = boundary_stream.chained_admission(
        experiment=object(), backend="cuda", device_bytes=4 * gib,
        card=(12 * gib, 4 * gib))
    assert fits["admitted"]
    refused = boundary_stream.chained_admission(
        experiment=object(), backend="cuda", device_bytes=8 * gib,
        card=(12 * gib, 4 * gib))
    assert not refused["admitted"]
    # The card less the other-process margin every memory door keeps
    # (A163: the 10% share stood in for non-pool residency the forecast's
    # envelope now carries itself): 12 + 4 - 0.5.
    assert refused["reason"] == (
        "chained preparation not admitted: forecast 10.00 GiB + "
        "preparation 8.00 GiB > 15.50 GiB on the GPU")
    staging = tmp_path / ".tmp"
    staging.mkdir()
    writer = PreparedTreeWriter(staging=staging, output_root=tmp_path / "t",
                                identity={}, chained=True)
    writer.admit(experiment=object(), backend="cuda", device_bytes=8 * gib,
                 card=(12 * gib, 4 * gib))
    assert writer.chained is False


def test_an_unmeasured_gpu_producer_is_not_admitted(_forecast_of):
    _forecast_of(1)
    decision = boundary_stream.chained_admission(
        experiment=object(), backend="cuda", device_bytes=None,
        card=(1 << 40, 0))
    assert not decision["admitted"]
    assert "could not be measured" in decision["reason"]


def test_a_preparation_only_install_publishes_at_its_seal(monkeypatch,
                                                          tmp_path, capsys):
    """No forecast installed: no head for one, and no forecast to price.

    The RW-WPS preparation package ships this writer with the era5, gfs
    and mapped routes but neither the forecast's admission estimate nor a
    forecast.  There a CUDA preparation died at its admission on the
    missing woof.core.preflight.  Nothing in that install can bind a
    head, so the writer publishes at its seal, even when chaining was
    asked for, and prices nothing.
    """

    assert boundary_stream.forecast_installed()  # this checkout has both
    monkeypatch.setattr(boundary_stream, "forecast_installed", lambda: False)

    def unreachable(**_):
        raise AssertionError("priced a forecast this install cannot run")

    monkeypatch.setattr(boundary_stream, "chained_admission", unreachable)
    monkeypatch.setattr(boundary_stream, "host_admission", unreachable)
    monkeypatch.setenv("WOOF_CHAINED_PREP", "1")
    expected = {"chained": False,
                "reason": boundary_stream.PREPARATION_ONLY_REASON}
    staging = tmp_path / ".tmp"
    staging.mkdir()
    writer = PreparedTreeWriter(staging=staging, output_root=tmp_path / "t",
                                identity={})
    assert writer.chained is False
    for backend in ("cuda", "cpu"):
        assert writer.admit(experiment=object(), backend=backend,
                            device_bytes=1) == expected
    writer, output = _chained_tree(tmp_path, _snapshots(3), stop_after=0,
                                   name="unsealed")
    assert writer.chained is False and not output.exists()
    assert writer.head["decision"] == expected
    writer, output = _chained_tree(tmp_path, _snapshots(3), name="sealed")
    assert writer.published and (output / "proof.json").is_file()
    assert "forecast" not in capsys.readouterr().err


def test_a_tile_series_over_a_streamed_domain_is_lazy_too(tmp_path):
    from woof.core.streaming import _WindowedIntervals

    writer, output = _chained_tree(tmp_path, _snapshots(4), stop_after=0)
    domain = streamed_boundaries(output, head=read_head(output))
    tile = _WindowedIntervals(domain, None, seam="zeros", snapshot=None)
    assert tile.bounds == domain.intervals.bounds
    assert interval_index(tile.domain_intervals, 9000.0) == 2
    eager = _WindowedIntervals(_frames(_snapshots(3)).build(_times(3)), None,
                               seam="zeros", snapshot=None)
    assert not hasattr(eager, "bounds")


def test_sim_binds_the_head_of_a_preparation_still_running(tmp_path):
    from woof import stage_cli

    proof_head = {"schema": "gpuwm-gfs-direct-wrf-proof-v3",
                  "forcing_hours": [0, 1, 2, 3]}
    writer, output = _chained_tree(tmp_path, _snapshots(4), stop_after=0,
                                   proof_head=proof_head)
    bundle = stage_cli.unsealed_head_bundle(output)
    assert bundle["head_sha256"] == writer.head_sha256
    assert bundle["source"] == "gfs" and bundle["layout"] == "single"
    command = stage_cli.sim_command(
        bundle, experiment_config=tmp_path / "e.toml",
        wps_namelist=tmp_path / "n.wps", outdir=tmp_path / "run")
    assert command[command.index("--prepared-head-sha256") + 1] \
        == writer.head_sha256
    assert command[command.index("--prepared-root") + 1] == str(output)
    assert "--proof-sha256" not in command
    # A preparation that failed is never started on.
    writer.fail(ValueError("decode failed"))
    assert stage_cli.unsealed_head_bundle(output) is None
