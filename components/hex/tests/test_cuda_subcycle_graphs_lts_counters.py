"""Host-only: a replayed subcycle keeps local time stepping's launch counts.

With ``--local-timestep`` the subcycle's rate-class launches are captured with it
(woof.hex.cuda_subcycle_graphs), so a replay runs no Python and the attachment's
``launch_counts`` would stop at the capture.  The graph controller treats them like
its other counters: the captured subcycle's delta is added on every replay, and the
warm-up passes leave them as they were.  Brought from fork jacobbieker/woof PR 473
(Apache-2.0).
"""

from __future__ import annotations

from types import SimpleNamespace

from woof.hex.cuda_subcycle_graphs import SubcycleGraphs


def _driver(lts):
    return SimpleNamespace(cp=None, regional_v841=None, mixing_calls_v841=5, local_timestep=lts)


def test_counters_carry_the_lts_launch_counts():
    lts = SimpleNamespace(launch_counts={"reflux_settles": 3, "acoustic_substep_class_launches": 14})
    graphs = SubcycleGraphs(_driver(lts))
    before = graphs._counters()
    assert before == (5, 0, 0, 14, 3)  # sorted by name: class launches, then settles
    lts.launch_counts["acoustic_substep_class_launches"] += 28
    lts.launch_counts["reflux_settles"] += 6
    graphs.driver.mixing_calls_v841 += 1
    delta = tuple(a - b for a, b in zip(graphs._counters(), before))
    graphs._set_counters(before)  # a warm pass is rolled back
    assert lts.launch_counts == {"reflux_settles": 3, "acoustic_substep_class_launches": 14}
    graphs._set_counters(tuple(a + b for a, b in zip(before, delta)))  # one replay
    assert lts.launch_counts == {"reflux_settles": 9, "acoustic_substep_class_launches": 42}
    assert graphs.driver.mixing_calls_v841 == 6


def test_counters_without_lts_are_the_three_dycore_counters():
    graphs = SubcycleGraphs(_driver(None))
    assert graphs._counters() == (5, 0, 0)
    graphs._set_counters((7, 0, 0))
    assert graphs.driver.mixing_calls_v841 == 7


def test_an_interrupted_pass_restores_the_lts_stage_tracker():
    from woof.hex.cuda_driver_lts import _StageTracker

    tracker = _StageTracker(expected=(1, 3, 6))
    for stage in (1, 3, 6):  # one whole subcycle: the tracker ends on the last stage
        for small_step in range(1, stage + 1):
            tracker.note(small_step)
    lts = SimpleNamespace(launch_counts={}, tracker=tracker)
    graphs = SubcycleGraphs(_driver(lts))
    saved = graphs._tracker_state()
    assert saved == (2, 6)
    tracker.note(1)
    tracker.note(1)  # a warm pass stopped by an OOM inside the second stage
    graphs._restore_tracker(saved)
    for stage in (1, 3, 6):  # the eager fallback restarts at RK1 without a refusal
        for small_step in range(1, stage + 1):
            tracker.note(small_step)
    assert graphs._tracker_state() == (2, 6)
    assert SubcycleGraphs(_driver(None))._tracker_state() is None


def test_the_forecast_driver_no_longer_skips_graphs_for_local_time_stepping():
    """The driver attached no graph controller under --local-timestep through hex 0.6.1."""
    from pathlib import Path

    import woof.hex.drivers.run_cuda_v841_forecast as driver

    text = Path(driver.__file__).read_text()
    assert "graph_requested and lts_attachment is None" not in text
    assert 'os.environ.get("WOOF_HEX_CUDA_GRAPHS_LTS", "1") != "0"' in text
