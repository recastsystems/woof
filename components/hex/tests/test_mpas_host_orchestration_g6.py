"""G6 host orchestration on the MPAS-physics route: moving publication cadence and the step record writer.

CPU tier; the whole-run evidence (storm-following and zoom step times, bitwise streams) is in the release row.
"""
import pytest
import json
from types import SimpleNamespace

from woof.hex.mpasphys.runtime_v842 import (StepRecordWriter, moving_columns_due, moving_publication_due,
                                           native_publication_required)


def route(**kw):
    values = dict(control=object(), frame_seconds=900.0, offset_seconds=0.0, shock=False)
    values.update(kw)
    return SimpleNamespace(**values)


def test_moving_route_downloads_pools_only_at_frames_and_ends():
    moving = route()
    dt, steps = 18.75, 192
    due = [s for s in range(steps + 1) if moving_publication_due(moving, step=s, steps=steps, dt=dt)]
    # 900 s frames at 18.75 s: every 48th step, plus the start and the last step.
    assert due == [0, 48, 96, 144, 192]
    assert not moving_publication_due(None, step=48, steps=steps, dt=dt)


def test_moving_frame_cadence_follows_a_resumed_carrier_offset():
    moving = route(offset_seconds=3600.0, frame_seconds=1800.0)
    due = [s for s in range(1, 200) if moving_publication_due(moving, step=s, steps=500, dt=18.75)]
    assert due == [96, 192]


def test_a_checkpoint_route_downloads_pools_at_every_checkpoint_step():
    """The checkpoint carrier seals the host pool views; a mid-run checkpoint step must download them, or a
    restart from it starts from stale pools (restart != straight)."""
    checkpoints = route(control=None, checkpoint_due=lambda step, dt: step * dt == 3600.0)
    due = [s for s in range(13) if moving_publication_due(checkpoints, step=s, steps=12, dt=600.0)]
    assert due == [0, 6, 12]


def test_moving_without_controller_needs_only_the_last_step_and_shock_needs_columns():
    quiet = route(control=None)
    assert [s for s in range(11) if moving_publication_due(quiet, step=s, steps=10, dt=60.0)] == [0, 10]
    shock = route(control=None, shock=True)
    assert all(moving_columns_due(shock, step=s, steps=10, dt=60.0) for s in range(11))
    assert not moving_publication_due(shock, step=5, steps=10, dt=60.0)


def test_publication_policy_no_longer_forces_every_moving_step():
    streams = []
    native = SimpleNamespace(alarm_ringing=lambda name: False)
    common = dict(steps=10, lazy=True)
    assert not native_publication_required(native, streams, step=5, moving=None, **common)
    assert native_publication_required(native, streams, step=5, moving=route(), **common)
    assert native_publication_required(native, streams, step=0, moving=None, **common)


def test_step_record_writer_keeps_call_log_bytes_and_order(tmp_path):
    clock = iter(range(0, 1000, 5))
    writer = StepRecordWriter(every_seconds=30.0, clock=lambda: next(clock))
    calls = tmp_path / 'physics-calls.jsonl'
    receipt_path = tmp_path / 'native-physics-receipt.json'
    expected = ''
    for step in range(1, 21):
        line = json.dumps(dict(step=step, calls=[dict(slot='pbl', step=step)])) + '\n'
        expected += line
        writer.append(calls, line)
        writer.receipt(receipt_path, dict(steps_completed=step), force=step <= 1)
    writer.close()
    assert calls.read_text() == expected
    # Forced at step 1, then at most one serialisation per 30 s of the (fake) clock.
    assert 1 < writer.receipt_writes < 20
    assert json.loads(receipt_path.read_text())['steps_completed'] >= 14
    assert not (tmp_path / 'native-physics-receipt.json.partial').exists()


def test_step_record_writer_every_step_when_interval_is_zero(tmp_path):
    writer = StepRecordWriter(every_seconds=0.0)
    path = tmp_path / 'r.json'
    for step in range(5):
        assert writer.receipt(path, dict(step=step))
    writer.drain()
    assert json.loads(path.read_text()) == dict(step=4)
    writer.close()


@pytest.fixture(autouse=True)
def _speed_word_off(monkeypatch):
    """These tests state the law with WOOF_HEX_PUBLISH_LEAN off (it is on by default since 19 Nov; speed_words)."""
    monkeypatch.setenv("WOOF_HEX_PUBLISH_LEAN", "0")
