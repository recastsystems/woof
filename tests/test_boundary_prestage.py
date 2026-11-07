"""Small scheduling smoke; the landing gate is the real-size 3 h pair."""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.boundary_prestage import prestage_boundaries, current_boundary_prestage
from woof.ingest.lateral_bc import StateBoundaryFrames, build_lateral_boundaries


def test_prestage_waits_for_analysis_reuses_tail_and_preserves_every_word(monkeypatch, tmp_path):
    rng = np.random.default_rng(54)
    snapshots = [{"theta": rng.normal(size=(3, 12, 14))} for _ in range(4)]
    analysis = {"theta": rng.normal(size=(3, 12, 14))}
    calls = []
    times = [0., 3600., 7200., 10800.]
    monkeypatch.setattr("woof.initial_source.read_initial_inputs", lambda p: {"path": p})

    class Frames(StateBoundaryFrames):
        def add_state(self, state, *, index):
            self.add_snapshot(state, index=index)

    def prepare():
        handle = current_boundary_prestage()
        frames = Frames()
        frames.add_snapshot(snapshots[0], index=0)
        def build(index):
            calls.append(index)
            return None, SimpleNamespace(state=snapshots[index])
        request = handle.prepare_independent(forcing=frames, times=times,
            build=build, release=lambda: None, workers=2, output_root=tmp_path)
        assert request["path"].name == "initial.json"
        frames.replace_snapshot(analysis, index=0)
        handle.complete(frames, times)
        handle.cache_content_sha256 = "sealed"
        return 0

    with prestage_boundaries(prepare, pin=False) as handle:
        assert handle.wait_ready(5)
        assert sorted(calls) == [1, 2, 3]
        assert handle.boundaries is None
        tail = handle._tail
        assert handle.finish_initial(tmp_path / "initial.json", timeout=5) == 0
        expected = build_lateral_boundaries([analysis, *snapshots[1:]], times)
        actual = handle.for_forecast(SimpleNamespace(prepared_root=tmp_path,
            cache_reader=SimpleNamespace(content_sha256="sealed")))
        assert actual.intervals[1] is tail[0]
        for left, right in zip(actual.intervals, expected.intervals):
            assert left.end_frame_sha256 == right.end_frame_sha256
            for side in ("west", "east", "south", "north"):
                for name in ("value", "tendency"):
                    a = getattr(getattr(left.fields["theta"], side), name)
                    b = getattr(getattr(right.fields["theta"], side), name)
                    np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
        assert sorted(calls) == [1, 2, 3]  # no replay after analysis arrival


def test_producer_failure_reaches_waiter():
    def fail():
        raise ValueError("source decode failed")
    with prestage_boundaries(fail, pin=False) as handle:
        with pytest.raises(ValueError, match="source decode failed"):
            handle.wait_ready(5)


@pytest.mark.parametrize("hip", [False, True])
def test_resident_table_registration_uses_native_hip_binding(monkeypatch, hip):
    import sys
    from woof import boundary_prestage, hip_runtime
    calls = []
    def stub(*args):
        raise NotImplementedError("CuPy HIP host-registration stub")
    runtime = SimpleNamespace(is_hip=hip,
        hostRegister=stub if hip else lambda *a: calls.append(("cuda-register", a)),
        hostUnregister=stub if hip else lambda *a: calls.append(("cuda-unregister", a)))
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)))
    monkeypatch.setattr(hip_runtime, "host_register", lambda *a: calls.append(("hip-register", a)))
    monkeypatch.setattr(hip_runtime, "host_unregister", lambda *a: calls.append(("hip-unregister", a)))
    array = np.arange(128, dtype=np.float32)
    monkeypatch.setattr(boundary_prestage, "_arrays", lambda intervals: iter((array, array)))
    pins = boundary_prestage._PinnedTables((None,))
    pins.add((None,))  # shared pages stay registered once
    assert len(pins.regions) == 1
    address, end = pins.regions[0]
    pins.close()
    prefix = "hip" if hip else "cuda"
    assert calls == [(prefix + "-register", (address, end-address, 1)),
                     (prefix + "-unregister", (address,))]
    assert not pins.regions
