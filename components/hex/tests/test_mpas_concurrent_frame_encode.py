"""MPAS-route history frames encode side by side, in frame order where order is observable (no GPU, no netCDF).

THE COST THIS REMOVES (30 km globe over 4 cards, V5090 box, 2026-10-27): one rw_mpas_stream process encoded each
history frame on one core, 24-36 s against ~23 s of stepping per forecast hour, so the run waited 40-63 s after its
last step.  Frames now encode concurrently; file ids are drawn in frame order, an appending frame waits for every
earlier one, and frames are published (history-ready.jsonl) in frame order.  Synthetic frames exercise ordering only.
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from woof.hex.mpasphys.streams_output_v842 import PhysicsStreams


class _Ids:
    def __init__(self):
        self.drawn = 0

    def next(self):
        self.drawn += 1
        return self.drawn


def _writer(tmp_path, monkeypatch):
    import woof.hex.history_writer as history_writer
    monkeypatch.setattr(history_writer, "mark_closed_frame_durable", lambda target: None)
    writer = PhysicsStreams.__new__(PhysicsStreams)
    writer._executor = ThreadPoolExecutor(max_workers=PhysicsStreams.ENCODE_WORKERS)
    writer._last_done = threading.Event()
    writer._last_done.set()
    writer._publish_after = {}
    writer._file_attributes = {}
    writer.output_validator = None   # PhysicsStreams.__init__ default (357d2a3f: deferred checks on the writer)
    writer.attrs = {"title": "frame"}
    writer.file_ids = _Ids()
    writer.output = tmp_path
    return writer


def _jobs(tmp_path, step, record=0):
    stream = SimpleNamespace(name="output", options={"output_interval": "1:00:00"})
    return [(stream, tmp_path / f"history.{step:03d}.nc", record)]


def test_frames_encode_together_and_publish_in_frame_order(tmp_path, monkeypatch):
    assert PhysicsStreams.ENCODE_WORKERS >= 2
    writer = _writer(tmp_path, monkeypatch)
    finished = []

    def encode(jobs, step, hold):
        time.sleep(hold)
        finished.append(step)
        writer._publish_closed(jobs, step)
        return [step]

    futures = []
    for step, hold in ((0, 0.6), (10, 0.05)):
        jobs = _jobs(tmp_path, step)
        futures.append(writer._submit_encode(jobs, step, encode, jobs, step, hold))
    assert [f.result(timeout=10) for f in futures] == [[0], [10]]
    assert finished == [10, 0], "the later frame did not encode beside the earlier one"
    published = [json.loads(line)["step"] for line in (tmp_path / "history-ready.jsonl").read_text().splitlines()]
    assert published == [0, 10]
    ids = [writer._file_attributes[tmp_path / f"history.{step:03d}.nc"]["file_id"] for step in (0, 10)]
    assert ids == [1, 2]
    assert writer._publish_after == {}
    writer._executor.shutdown()


def test_an_appending_frame_waits_for_every_earlier_frame(tmp_path, monkeypatch):
    writer = _writer(tmp_path, monkeypatch)
    order = []

    def encode(jobs, step, hold):
        time.sleep(hold)
        order.append(step)
        return []

    first = _jobs(tmp_path, 0)
    append = _jobs(tmp_path, 0, record=1)
    futures = [writer._submit_encode(first, 0, encode, first, 0, 0.4),
               writer._submit_encode(append, 10, encode, append, 10, 0.0)]
    for future in futures:
        future.result(timeout=10)
    assert order == [0, 10]
    writer._executor.shutdown()


def test_a_failed_frame_releases_the_next_one(tmp_path, monkeypatch):
    writer = _writer(tmp_path, monkeypatch)

    def fails(jobs, step):
        raise RuntimeError("encode failed")

    def publishes(jobs, step):
        writer._publish_closed(jobs, step)
        return [step]

    first, second = _jobs(tmp_path, 0), _jobs(tmp_path, 10)
    bad = writer._submit_encode(first, 0, fails, first, 0)
    good = writer._submit_encode(second, 10, publishes, second, 10)
    with pytest.raises(RuntimeError):
        bad.result(timeout=10)
    assert good.result(timeout=10) == [10]
    writer._executor.shutdown()
