"""Hex stages carry their measured CPU seconds, also when hex holds the event until its run folder exists.

Breakage it prevents: 2026-10-16, a hex init stage (rw_mpas_geometry) ran 8+ min on one core while the card
idled and no hex receipt said how many cores any stage used.
"""
import json
import os

from woof.hex import time_ledger


def test_buffered_begin_end_keep_the_cpu_clock_of_their_moment(tmp_path, monkeypatch):
    monkeypatch.delenv("WOOF_TIME_LEDGER", raising=False)
    run = tmp_path / "run"
    time_ledger.configure(run)                    # the folder does not exist yet: events are held
    time_ledger.emit("prep.host", "begin")
    x = 0
    for i in range(2_000_000):
        x += i
    time_ledger.emit("prep.host", "end")
    assert all("cpu_s" in e[2] for e in time_ledger._pending)
    before, after = (e[2]["cpu_s"] for e in time_ledger._pending)
    assert after >= before
    run.mkdir()
    time_ledger.emit("prep.destination", "mark")
    rows = [json.loads(x) for x in (run / "timeline.jsonl").read_text().splitlines()] if (
        run / "timeline.jsonl").exists() else []
    if rows:                                      # an engine with woof.timeline appends them
        cpu = [r["attrs"].get("cpu_s") for r in rows if r["phase"] in ("begin", "end")]
        assert cpu == [before, after]
    time_ledger.configure(None)


def test_marks_and_relayed_boundaries_carry_no_cpu():
    assert time_ledger._cpu_attrs({"a": 1}, "mark", None) == {"a": 1}
    assert time_ledger._cpu_attrs(None, "begin", 12.5) is None
    assert "cpu_s" in time_ledger._cpu_attrs(None, "end", None)


def test_tree_clock_walks_the_thread_children_at_most_every_refresh(monkeypatch):
    """Host-bound reference system 2026-10-27: woof's tree walk read /proc/<pid>/task/<tid>/children for ~170 threads at
    every stage boundary (34 % of the MPAS-physics LTS class threads' step CPU).  The pid list is reused for
    TREE_REFRESH_SECONDS; each boundary still reads the tree's own and reaped ticks, so the clock moves."""
    import sys
    import types
    import pytest
    if not sys.platform.startswith("linux"):
        pytest.skip("reads /proc")
    walks = []
    module = types.ModuleType("woof.timeline.parallelism")
    module.tree_pids = lambda root=None: (walks.append(root), [os.getpid()])[1]
    monkeypatch.setitem(sys.modules, "woof.timeline.parallelism", module)
    monkeypatch.setattr(time_ledger, "_tree", {"pids": None, "at": 0.0, "root": None})
    now = [50.0]
    first = time_ledger.tree_cpu_seconds(clock=lambda: now[0])
    x = 0
    for i in range(3_000_000):
        x += i
    second = time_ledger.tree_cpu_seconds(clock=lambda: now[0])
    assert len(walks) == 1 and second > first
    now[0] += time_ledger.TREE_REFRESH_SECONDS
    time_ledger.tree_cpu_seconds(clock=lambda: now[0])
    assert len(walks) == 2
    t = os.times()
    assert abs(second - (t.user + t.system + t.children_user + t.children_system)) < 0.5
