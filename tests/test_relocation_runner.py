"""CPU contracts for the relocation runner: schedule, receipts, restart.

The runner is exercised against the same CPU fake tree the staging tests
prove the primitive on, so a scheduled move here executes the REAL
relocate_child -- plan, admissibility, donor alignment, transplant --
with only the device replaced by numpy.  What is asserted:

- config governance for the leg-2 keys (cadence_seconds,
  [[relocation.move]], their interaction with [relocation.follow]);
- cadence gating and manual-itinerary consumption at cycle boundaries,
  through the storm tracker's exact provider signature;
- receipts: time, offsets (requested AND executed), overlap fraction,
  fields transplanted, fill counts and provenance, staging block;
- the restart posture: every executed move chains the live fingerprint,
  and a checkpoint header carrying the relocation block refuses with the
  promises-nothing posture BY NAME.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core.nest_relocation import (RESTART_ACROSS_MOVE_POSTURE,
                                        RelocationRefusal,
                                        mark_fingerprint_across_move)
from woof.core.relocation_runner import (ManualMoveProvider,
                                          RelocationRunner)
from woof.experiment import (RelocationConfig, ScheduledRelocationMove,
                              _build_relocation)

from test_nest_relocation_staging import (_cpu_tree, _footprint_statics,
                                          _initializer, _ramp_state, _grandchild)


# ---------------------------------------------------------------------------
# Config governance
# ---------------------------------------------------------------------------

def _domains(history_interval_s=900.0):
    """Stand-ins carrying exactly what _build_relocation reads.

    parent_id and history_interval_s joined the list for the refl-stash
    cadence contract (issue #111): the tracker watches grid_id 2's
    PARENT, so the root's history interval is the one that gates the
    follow cadence.
    """
    return [
        SimpleNamespace(grid_id=1, parent_id=None, time_step=60,
                        time_step_fract_num=0, time_step_fract_den=1,
                        history_interval_s=history_interval_s),
        SimpleNamespace(grid_id=2, parent_id=1, time_step=None,
                        history_interval_s=history_interval_s),
    ]


def _build(raw, run_seconds=3600.0):
    return _build_relocation(raw, "test.toml", _domains(), run_seconds)


def _raw(**over):
    table = {"enabled": True, "grid_id": 2, "max_move_parent_cells": 4}
    table.update(over)
    return {"relocation": table}


_MOVE = {"at_seconds": 600.0, "di_parent_cells": 2}
_FOLLOW = dict(field="uh", threshold=25.0, fallback_threshold=40.0,
               search_margin_cells=10, min_shift_cells=2,
               max_shift_cells=4, cooldown_seconds=900.0)


def test_manual_itinerary_loads_and_echoes():
    cfg = _build(_raw(cadence_seconds=600.0, move=[
        _MOVE, {"at_seconds": 1200.0, "dj_parent_cells": -1}]))
    assert cfg.cadence_seconds == 600.0
    assert [m.to_json() for m in cfg.moves] == [
        {"at_seconds": 600.0, "di_parent_cells": 2, "dj_parent_cells": 0},
        {"at_seconds": 1200.0, "di_parent_cells": 0, "dj_parent_cells": -1},
    ]
    echo = cfg.receipt()
    assert echo["cadence_seconds"] == 600.0
    assert len(echo["moves"]) == 2 and echo["follow"] is None


def test_move_row_unknown_key_refuses_by_name():
    with pytest.raises(ValueError, match="di_parent_cell"):
        _build(_raw(move=[{"at_seconds": 600.0, "di_parent_cell": 2}]))


def test_move_row_requires_at_seconds():
    with pytest.raises(ValueError, match="at_seconds"):
        _build(_raw(move=[{"di_parent_cells": 2}]))


def test_move_rows_must_increase_strictly():
    with pytest.raises(ValueError, match="strictly"):
        _build(_raw(move=[_MOVE, dict(_MOVE)]))


def test_move_must_land_on_a_root_step():
    with pytest.raises(ValueError, match="whole number of root steps"):
        _build(_raw(move=[{"at_seconds": 90.0, "di_parent_cells": 1}]))


def test_move_past_the_run_end_refuses():
    with pytest.raises(ValueError, match="past the end"):
        _build(_raw(move=[_MOVE]), run_seconds=600.0)


def test_move_off_the_cadence_refuses():
    with pytest.raises(ValueError, match="cadence"):
        _build(_raw(cadence_seconds=480.0, move=[_MOVE]))


def test_cadence_must_land_on_a_root_step():
    with pytest.raises(ValueError, match="whole number of root steps"):
        _build(_raw(cadence_seconds=90.0, move=[
            {"at_seconds": 180.0, "di_parent_cells": 1}]))


def test_cadence_without_a_follow_source_refuses():
    with pytest.raises(ValueError, match="no follow source"):
        _build(_raw(cadence_seconds=600.0))


def test_two_follow_sources_refuse():
    with pytest.raises(ValueError, match="two"):
        _build(_raw(move=[_MOVE], follow=dict(_FOLLOW)))


def test_moves_on_a_disabled_relocation_refuse():
    with pytest.raises(ValueError, match="enabled"):
        _build({"relocation": {"enabled": False, "move": [_MOVE]}})


def test_fractional_cells_refuse():
    with pytest.raises(ValueError, match="whole number of parent cells"):
        ScheduledRelocationMove(at_seconds=600.0, di_parent_cells=1.5)


def test_follow_table_with_cadence_loads():
    cfg = _build(_raw(cadence_seconds=900.0, follow=dict(_FOLLOW)))
    assert cfg.cadence_seconds == 900.0
    assert cfg.follow is not None and not cfg.moves


# --- The refl stash has to be able to serve the cadence (issue #111) -----
#
# refl_10cm is stashed by the microphysics on the HISTORY cadence, so a
# tracker consulted off that cadence asks for a plane that does not
# exist -- and finds out at the first evaluation where UH is under
# threshold and the echo fallback is consulted, which is hours into a
# run and is precisely when a storm-following nest should be working.
# The contract was already written in prose above cadence_seconds in
# configs/moving_nest_20110427_follow_2km.toml; nothing enforced it.


def _follow_build(cadence, history_interval_s=900.0, run_seconds=43200.0):
    return _build_relocation(
        _raw(cadence_seconds=cadence, follow=dict(_FOLLOW)), "test.toml",
        _domains(history_interval_s), run_seconds)


@pytest.mark.parametrize("cadence", [900.0, 1800.0, 2700.0, 3600.0])
def test_a_cadence_the_stash_can_serve_loads(cadence):
    """Whole multiples of the watched domain's history interval pass."""
    cfg = _follow_build(cadence)
    assert cfg.cadence_seconds == cadence
    assert cfg.follow is not None


@pytest.mark.parametrize("cadence", [300.0, 600.0, 1200.0, 2400.0])
def test_a_cadence_the_stash_cannot_serve_refuses_at_load(cadence):
    """And it refuses AT LOAD, naming both knobs and the fix."""
    with pytest.raises(ValueError) as caught:
        _follow_build(cadence)
    message = str(caught.value)
    assert "cadence_seconds" in message           # knob one
    assert "history_interval_s" in message        # knob two
    assert "whole multiple" in message            # the fix
    assert "refl_10cm" in message                 # why


def test_a_follow_block_with_no_cadence_refuses():
    """Absent means EVERY cycle boundary, which the stash cannot serve."""
    with pytest.raises(ValueError, match="no cadence_seconds"):
        _follow_build(None)


def test_the_refusal_names_the_watched_parent_not_the_moving_child():
    """grid_id names the child that MOVES; the tracker reads its parent.

    Naming the wrong domain would send a reader to edit a
    history_interval_s that has nothing to do with the failure.
    """
    with pytest.raises(ValueError) as caught:
        _follow_build(600.0)
    message = str(caught.value)
    assert "grid_id = 1" in message and "parent of the relocating" in message


@pytest.mark.parametrize("domains", [
    # A child whose parent_id names no domain, and a domain list with no
    # history_interval_s at all.  Both are OTHER validators' refusals;
    # this check must not preempt them with a cadence message that sends
    # the reader to the wrong knob.
    [SimpleNamespace(grid_id=1, parent_id=None, time_step=60,
                     time_step_fract_num=0, time_step_fract_den=1,
                     history_interval_s=900.0),
     SimpleNamespace(grid_id=2, parent_id=7, time_step=None,
                     history_interval_s=900.0)],
    [SimpleNamespace(grid_id=1, parent_id=None, time_step=60,
                     time_step_fract_num=0, time_step_fract_den=1),
     SimpleNamespace(grid_id=2, parent_id=1, time_step=None)],
])
def test_an_unresolvable_tree_is_left_to_its_own_validator(domains):
    cfg = _build_relocation(
        _raw(cadence_seconds=600.0, follow=dict(_FOLLOW)), "test.toml",
        domains, 43200.0)
    assert cfg.follow is not None


def test_an_itinerary_without_a_tracker_is_not_bound_by_the_stash():
    """[[relocation.move]] reads no fields, so no plane has to exist."""
    cfg = _build(_raw(cadence_seconds=600.0, move=[_MOVE]))
    assert cfg.cadence_seconds == 600.0 and cfg.follow is None


def test_the_shipped_follow_config_satisfies_its_own_contract():
    """The config whose comment states this rule must obey it.

    It is the only committed [relocation.follow] worked example, so a
    contract it fails is a contract that ships broken.
    """
    import tomllib

    root = Path(__file__).parents[1]
    for name in ("moving_nest_20110427_follow_2km.toml",
                 "moving_nest_20110427_spawn_2km.toml"):
        path = root / "configs" / name
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
        if "follow" not in raw.get("relocation", {}):
            continue
        cadence = raw["relocation"]["cadence_seconds"]
        child = next(d for d in raw["domain"]
                     if d["grid_id"] == raw["relocation"]["grid_id"])
        parent = next(d for d in raw["domain"]
                      if d["grid_id"] == child["parent_id"])
        multiples = cadence / parent["history_interval_s"]
        assert multiples == int(multiples) and multiples >= 1, name


# ---------------------------------------------------------------------------
# The manual provider speaks the tracker's exact contract
# ---------------------------------------------------------------------------

def test_manual_provider_fires_each_row_exactly_once():
    provider = ManualMoveProvider((
        ScheduledRelocationMove(60.0, 1, 0),
        ScheduledRelocationMove(120.0, 0, -1)))
    # (parent_state, nest_footprint, t) -- the storm tracker's signature.
    assert provider(None, None, 30.0) is None
    assert provider(None, None, 60.0) == (1, 0)
    assert provider(None, None, 60.0) is None
    assert provider(None, None, 120.0) == (0, -1)
    assert provider(None, None, 180.0) is None
    assert provider.unconsumed() == ()


def test_manual_provider_retires_a_row_behind_the_clock_and_says_so():
    """A row the clock is already past is INHERITED, not pending.

    It used to stay at the head of the queue for ever, where -- because
    the head is matched exactly -- it could never fire again and blocked
    every later row with it.  Retiring it is the reading that was always
    correct on the only route that reaches this state: a row behind a
    checkpoint executed BEFORE that checkpoint was written, and its
    result is already in the restored state.  The receipt still carries
    it, because silently dropping a scripted move is the defect one layer
    over from silently never making it.
    """
    provider = ManualMoveProvider((ScheduledRelocationMove(60.0, 1, 0),))
    assert provider(None, None, 120.0) is None
    assert provider.unconsumed() == ()
    assert len(provider.behind_the_clock()) == 1


def test_a_resumed_leg_still_fires_the_row_AHEAD_of_the_clock():
    """The defect itself.

    MEASURED on a 3 / 1 km tree with moves at 1200 s and 2400 s, resuming
    from the 1800 s checkpoint: NEITHER move executed, the nest stayed at
    its 1200 s footprint, and the two frames after the missed 2400 s move
    differed from the uninterrupted run -- while the restore banner said
    the resume reproduces that run bit for bit.
    """
    provider = ManualMoveProvider((
        ScheduledRelocationMove(1200.0, 1, 0),
        ScheduledRelocationMove(2400.0, 0, -1)))
    assert provider(None, None, 1800.0) is None
    assert provider(None, None, 2400.0) == (0, -1)
    assert len(provider.behind_the_clock()) == 1
    assert provider.unconsumed() == ()


def test_a_run_that_starts_at_zero_retires_nothing():
    """The control: retiring must not become a way to skip a live row."""
    provider = ManualMoveProvider((
        ScheduledRelocationMove(60.0, 1, 0),
        ScheduledRelocationMove(120.0, 0, -1)))
    assert provider(None, None, 60.0) == (1, 0)
    assert provider.behind_the_clock() == ()


# ---------------------------------------------------------------------------
# The runner on the CPU fake tree
# ---------------------------------------------------------------------------

def _clock(ticks):
    # ``spec`` mirrors the real clock surface the activation-epoch publish
    # reads (start_ticks/step_ticks); these fixtures model domains active
    # from t=0 on a 60 s step, matching the schedule stand-ins below.
    return SimpleNamespace(ticks=int(ticks), tick_den=1,
                           elapsed_seconds=float(ticks),
                           spec=SimpleNamespace(start_ticks=0, step_ticks=60))


def _model(parent, child):
    nodes = {int(parent.cfg.grid_id): parent, int(child.cfg.grid_id): child}
    model = SimpleNamespace(
        root=parent,
        schedule=SimpleNamespace(
            period_ticks=60, clock=SimpleNamespace(tick_den=1)),
        experiment_fingerprint="f" * 64)
    model.node = lambda gid: nodes[gid]
    return model


def _runner(parent_plane, config, **kwargs):
    kwargs.setdefault("staging", "host")
    kwargs.setdefault("initializer", _initializer(parent_plane))
    kwargs.setdefault(
        "static_provenance",
        "footprint-parametric synthetic statics (test)")
    kwargs.setdefault("on_child_built", lambda *args: None)
    return RelocationRunner(
        config=config,
        schedule=SimpleNamespace(
            period_ticks=60, clock=SimpleNamespace(tick_den=1)),
        **kwargs)


def _advance(model, parent, child, ticks):
    parent.clock = _clock(ticks)
    child.clock = _clock(ticks)


def _manual_config(moves, cadence=None):
    return RelocationConfig(
        enabled=True, grid_id=2, max_move_parent_cells=4,
        min_overlap_fraction=0.25, cadence_seconds=cadence,
        moves=tuple(moves))


def test_checkpoint_placement_rebuilds_a_disjoint_descendant():
    """A saved footprint may be beyond the entire original fine domain."""
    parent_plane, parent, child = _cpu_tree()
    grand = _grandchild(child, ratio=3, nx=6, ny=6)
    model = _model(parent, child)
    _advance(model, parent, child, 0)
    rebuilt = []

    def reground(*, node, plan, delta_parent_cells):
        assert node is grand
        assert plan.disjoint
        rebuilt.append(delta_parent_cells)
        return {"statics": "rebuilt for checkpoint"}

    runner = _runner(parent_plane, _manual_config(()),
                     provider=lambda *_a, **_k: None,
                     reground_descendant=reground)
    want = int(child.cfg.i_parent_start) + 4
    receipt = runner.adopt_placement(model, child, i_parent_start=want,
                                    j_parent_start=int(child.cfg.j_parent_start))
    assert receipt['event'] == 'placement-restored'
    assert child.cfg.i_parent_start == want
    assert rebuilt == [(4 * child.cfg.parent_grid_ratio, 0)]
    assert runner.moves_executed == 0


def test_scheduled_moves_execute_at_their_boundaries(tmp_path):
    parent_plane, parent, child = _cpu_tree()
    start_i = int(child.cfg.i_parent_start)
    start_j = int(child.cfg.j_parent_start)
    model = _model(parent, child)
    receipts_path = tmp_path / "relocation.json"
    runner = _runner(parent_plane, _manual_config((
        ScheduledRelocationMove(60.0, 2, 0),
        ScheduledRelocationMove(120.0, 0, 1))),
        receipts_path=receipts_path)
    base_fingerprint = model.experiment_fingerprint

    outcomes = []
    for ticks in (0, 60, 120, 180):
        _advance(model, parent, child, ticks)
        outcomes.append(runner.on_period_begin(
            model, {1: parent.clock}, period=ticks // 60))
    summary = runner.close_receipt(model)

    assert outcomes[0] is None                      # t=0 is placement
    assert outcomes[1]["event"] == "relocated"
    assert outcomes[2]["event"] == "relocated"
    assert outcomes[3]["event"] == "held"           # itinerary exhausted
    assert child.cfg.i_parent_start == start_i + 2
    assert child.cfg.j_parent_start == start_j + 1
    assert runner.moves_executed == 2
    assert summary["moves_executed"] == 2
    assert summary["unconsumed_moves"] == []

    # The move receipt's required content: time, offsets, overlap
    # fraction, fields transplanted, fill counts, provenance, staging.
    move = outcomes[1]
    assert move["elapsed_seconds"] == 60.0
    assert move["requested_shift_parent_cells"] == [2, 0]
    assert move["executed_shift_parent_cells"] == [2, 0]
    assert move["clamped_by"] == []
    assert move["placement_from"]["i_parent_start"] == start_i
    assert move["placement_to"]["i_parent_start"] == start_i + 2
    assert 0.0 < move["overlap_fraction"] < 1.0
    fill = move["fill"]
    assert fill["fields_transplanted"] > 0
    assert fill["overlap_cells"] + fill["spin_up_cells"] == (
        int(child.cfg.run.nx) * int(child.cfg.run.ny))
    assert "parent" in fill["strip_fill_source"]
    assert "footprint-parametric" in fill["static_fields"]
    assert move["staging"]["mode"] == "host"
    assert move["donor_alignment_pass"] is True
    assert move["generation"] == 1
    assert outcomes[2]["generation"] == 2

    # Fingerprint marking: chained, deterministic, and reconstructible
    # from the receipts alone.
    assert model.experiment_fingerprint != base_fingerprint
    expected = base_fingerprint
    for outcome in outcomes[1:3]:
        assert outcome["experiment_fingerprint"]["before"] == expected
        expected = mark_fingerprint_across_move(
            expected, outcome["record_sha256"])
        assert outcome["experiment_fingerprint"]["after"] == expected
    assert model.experiment_fingerprint == expected

    # The run receipts: on the model, and durable at receipts_path.
    assert model._relocation_receipts is runner.receipts
    document = json.loads(receipts_path.read_text(encoding="utf-8"))
    assert document["config"]["enabled"] is True
    events = [row["event"] for row in document["receipts"]]
    assert events == ["relocated", "relocated", "held", "summary"]


def test_the_run_end_summary_appears_exactly_once(tmp_path):
    """One run, one summary row -- however many times the executor closes.

    ``execute_experiment`` closes the runner every time it returns, and on
    a leg-walking route (``woof.runtime.walk_spawn_legs``) it returns once
    per leg.  A live 12-move run shipped ``relocation_receipts.json`` with
    two byte-identical summary rows in a 25-row list, so a consumer
    counting summaries double-counted.  The surviving row must also be the
    LAST state, not the first: freezing it would report a stale
    ``moves_executed``.
    """

    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    receipts_path = tmp_path / "relocation.json"
    runner = _runner(parent_plane, _manual_config((
        ScheduledRelocationMove(60.0, 1, 0),
        ScheduledRelocationMove(120.0, 1, 0))),
        receipts_path=receipts_path)

    # Leg one: one move, then the executor returns and closes.
    for ticks in (0, 60):
        _advance(model, parent, child, ticks)
        runner.on_period_begin(model, {1: parent.clock})
    first = runner.close_receipt(model)
    assert first["moves_executed"] == 1

    # Leg two: the second move, and a second close.
    _advance(model, parent, child, 120)
    runner.on_period_begin(model, {1: parent.clock})
    second = runner.close_receipt(model)

    summaries = [row for row in runner.receipts if row["event"] == "summary"]
    assert len(summaries) == 1
    assert summaries[0] is second
    assert summaries[0]["moves_executed"] == 2
    assert summaries[0]["unconsumed_moves"] == []

    document = json.loads(receipts_path.read_text(encoding="utf-8"))
    events = [row["event"] for row in document["receipts"]]
    assert events.count("summary") == 1
    # And it stays the run's final row rather than being stranded mid-list
    # by the moves that came after the first close.
    assert events == ["relocated", "relocated", "summary"]


def test_cadence_gates_the_opportunities():
    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(120.0, 1, 0),), cadence=120.0))
    assert runner.cadence_periods == 2
    consulted = []
    original = runner.provider

    def spy(parent_state, footprint, t):
        consulted.append(t)
        return original(parent_state, footprint, t)

    runner.provider = spy
    for ticks in (0, 60, 120, 180, 240):
        _advance(model, parent, child, ticks)
        runner.on_period_begin(model, {1: parent.clock})
    # Only the cadence boundaries were consulted -- never t=0, never the
    # off-cadence periods.
    assert consulted == [120.0, 240.0]
    assert runner.moves_executed == 1


def test_requested_shift_clamps_to_the_bounds_and_says_so():
    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(60.0, 9, 0),)))  # over max_move=4
    _advance(model, parent, child, 60)
    outcome = runner.on_period_begin(model, {1: parent.clock})
    assert outcome["event"] == "relocated"
    assert outcome["requested_shift_parent_cells"] == [9, 0]
    assert outcome["executed_shift_parent_cells"] == [4, 0]
    assert "max_move_parent_cells" in outcome["clamped_by"]


def test_unstarted_child_holds_with_a_reason():
    parent_plane, parent, child = _cpu_tree()
    child._started = False
    model = _model(parent, child)
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(60.0, 1, 0),)))
    _advance(model, parent, child, 60)
    outcome = runner.on_period_begin(model, {1: parent.clock})
    assert outcome["event"] == "held"
    assert "not started" in outcome["reason"]


def test_executed_moves_notify_a_provider_that_asks():
    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(60.0, 9, 0),)))
    notified = []
    runner.provider.notify_move_executed = (
        lambda t, shift: notified.append((t, shift)))
    _advance(model, parent, child, 60)
    runner.on_period_begin(model, {1: parent.clock})
    # Accepted-move semantics: the notification carries the EXECUTED
    # (clamped) shift, not the request.
    assert notified == [(60.0, (4, 0))]


def test_before_rebuild_receives_the_grid_id_before_the_release():
    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(60.0, 1, 0),)))
    _advance(model, parent, child, 60)
    dropped = []
    runner.on_period_begin(model, {1: parent.clock},
                           before_rebuild=lambda gid: dropped.append(gid))
    assert dropped == [2]


@pytest.mark.parametrize("collection", [False, True])
def test_complete_move_timer_includes_rebuilt_child_validation(
        monkeypatch, tmp_path, collection):
    from woof.core import relocation_runner as rr

    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    receipt_path = tmp_path / "moves.json"
    model.nodes_by_grid_id = {1: parent, 2: child}
    original_i = child.cfg.i_parent_start
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(60.0, 1, 0),)), receipts_path=receipt_path)
    mover = rr.RelocationRunnerCollection((runner,)) if collection else runner
    clock = [100.0]
    monkeypatch.setattr(rr.time, "perf_counter", lambda: clock[0])
    checked = []

    def validate(gid):
        # The new child is live before the executor checks its state.
        assert child.cfg.i_parent_start == original_i + 1
        checked.append(gid)
        clock[0] += 3.5

    _advance(model, parent, child, 60)
    mover.on_period_begin(model, {1: parent.clock}, after_rebuild=validate)
    row = runner.receipts[-1]
    assert checked == [2]
    assert row["move_wall_seconds"] == 3.5
    assert row["remap_wall_seconds"] == 0.0
    persisted = json.loads(receipt_path.read_text())["receipts"][-1]
    assert persisted["move_wall_seconds"] == 3.5

    # An exhausted itinerary holds without rebuilding or checking a child.
    _advance(model, parent, child, 120)
    mover.on_period_begin(model, {1: parent.clock}, after_rebuild=validate)
    assert checked == [2]


def test_rebuilt_child_validation_failure_propagates():
    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(60.0, 1, 0),)))
    _advance(model, parent, child, 60)

    def unhealthy(gid):
        raise RuntimeError(f"unhealthy rebuilt child {gid}")

    with pytest.raises(RuntimeError, match="unhealthy rebuilt child 2"):
        runner.on_period_begin(model, {1: parent.clock}, after_rebuild=unhealthy)
    assert not any(row["event"] == "relocated" for row in runner.receipts)


@pytest.mark.parametrize("collection", [False, True])
@pytest.mark.parametrize("staging", ["host", "device"])
def test_mid_tree_move_rebinds_descendant_health(collection, staging):
    from woof.core.relocation_runner import RelocationRunnerCollection
    from test_nest_relocation_staging import _grandchild

    parent_plane, parent, child = _cpu_tree()
    grand = _grandchild(child, ratio=3, nx=60, ny=60)
    model = _model(parent, child)
    model.nodes_by_grid_id = {1: parent, 2: child, 3: grand}
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(60.0, 1, 0),)))
    runner.staging = staging
    validators = {2: child.state, 3: grand.state}
    checked = []

    def reground(*, node, plan, delta_parent_cells):
        # The production regrounder releases these arrays before rebuilding.
        assert node.cfg.grid_id not in validators
        node.state = _ramp_state(60, 60)
        return {"donor_alignment": {"pass": True}, "statics": "rebuilt"}

    def validate(gid):
        validators[gid] = {2: child, 3: grand}[gid].state
        checked.append(gid)

    runner.reground_descendant = reground
    mover = RelocationRunnerCollection((runner,)) if collection else runner
    _advance(model, parent, child, 60)
    mover.on_period_begin(model, {1: parent.clock},
                          before_rebuild=lambda gid: validators.pop(gid, None),
                          after_rebuild=validate)
    assert checked == [2, 3]
    assert validators[3] is grand.state
    assert runner.receipts[-1]["descendants"][0]["grid_id"] == 3


def test_descendant_health_failure_stops_before_move_success():
    from test_nest_relocation_staging import _grandchild

    parent_plane, parent, child = _cpu_tree()
    grand = _grandchild(child, ratio=3, nx=60, ny=60)
    model = _model(parent, child)
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(60.0, 1, 0),)))
    runner.reground_descendant = lambda **kwargs: {}

    def unhealthy(gid):
        if gid == grand.cfg.grid_id:
            raise RuntimeError("unhealthy rebuilt descendant")

    _advance(model, parent, child, 60)
    with pytest.raises(RuntimeError, match="unhealthy rebuilt descendant"):
        runner.on_period_begin(model, {1: parent.clock}, after_rebuild=unhealthy)
    assert not any(row["event"] == "relocated" for row in runner.receipts)


def test_runner_construction_refusals():
    schedule = SimpleNamespace(period_ticks=60,
                               clock=SimpleNamespace(tick_den=1))
    with pytest.raises(RelocationRefusal, match="enabled"):
        RelocationRunner(config=RelocationConfig(), schedule=schedule,
                         on_child_built=lambda *a: None)
    bounds_only = RelocationConfig(enabled=True, grid_id=2)
    with pytest.raises(RelocationRefusal, match="no follow source"):
        RelocationRunner(config=bounds_only, schedule=schedule,
                         on_child_built=lambda *a: None)
    manual = _manual_config((ScheduledRelocationMove(60.0, 1, 0),))
    with pytest.raises(RelocationRefusal, match="on_child_built"):
        RelocationRunner(config=manual, schedule=schedule,
                         on_child_built=None)


def test_executor_refuses_a_follow_source_with_no_runner():
    from woof.core.model import ModelRuntimeStatus, execute_experiment

    exp = SimpleNamespace(relocation=_manual_config(
        (ScheduledRelocationMove(60.0, 1, 0),)))
    model = SimpleNamespace(
        _runtime_status=ModelRuntimeStatus(),
        _activation_context={"experiment": exp})
    with pytest.raises(RuntimeError, match="relocation.move"):
        execute_experiment(model, validate_state=False)


def test_run_experiment_front_door_lift_and_residual_refusals(
        tmp_path, monkeypatch):
    """Leg-3 governance: the case-data route no longer refuses a follow
    source by name -- it wires the real-data runner instead -- while a
    follow source with no child to move still refuses at the door."""
    from woof import runtime

    from woof.core.streaming import OFF as _STREAMING_OFF
    from woof.simulated_radar_config import OFF as RADAR_OFF

    manual = _manual_config((ScheduledRelocationMove(60.0, 1, 0),))
    # ``streaming`` is not decoration on this double: since preflight-ledger
    # and feat-route-wire, run_experiment refuses a [tiles] block at
    # ADMISSION -- before the ingest -- because none of its arms consult one.
    # A real Experiment always carries the attribute (build_experiment
    # defaults it to OFF); a double without it raises AttributeError from
    # inside the front door and never reaches the refusal under test.
    single = SimpleNamespace(relocation=manual, domains=(object(),),
                             feedback=0, tiles=_STREAMING_OFF, simulated_radar=RADAR_OFF)
    # The gate lives in run_experiment's body: the wrapper opens the step
    # log off exp.start_time first (01d3e4ec2f), which this double omits.
    with pytest.raises(ValueError, match="no nest to move"):
        runtime._run_experiment(single, None, tmp_path / "out")

    # Multi-domain: the old front-door ValueError is GONE.  The config
    # passes the gate and reaches ordinary preparation (witnessed by a
    # sentinel planted on the first preparation step).
    class _ReachedPreparation(Exception):
        pass

    import woof.io.wrfout as wrfout_module

    def _sentinel(_outdir):
        raise _ReachedPreparation

    monkeypatch.setattr(
        wrfout_module, "quarantine_orphan_wrfouts", _sentinel)
    tree = SimpleNamespace(relocation=manual,
                           domains=(object(), object()), feedback=0,
                           tiles=_STREAMING_OFF, simulated_radar=RADAR_OFF)
    with pytest.raises(_ReachedPreparation):
        runtime._run_experiment(tree, None, tmp_path / "out2")


def test_prepared_tree_route_still_refuses_a_corridor_less_follow_source():
    """A prepared tree runs without the case's GEOG source, so the
    refusal survives for CORRIDOR-LESS bundles -- and now names the
    remedy: re-prepare with --statics-corridor (the sealed
    child-resolution statics the runner crops per move)."""
    import inspect

    from woof import prepared_domain_tree_forecast as route

    source = inspect.getsource(route)
    assert "cannot rebuild a relocated child's statics" in source
    assert "--statics-corridor" in source
    assert "woof run" in source


def test_run_route_wires_initializer_preparer_and_provenance(tmp_path):
    """build_real_relocation_runner assembles the leg-3 machinery: the
    footprint-rebuilt-statics initializer (with its alignment frame,
    strip-fill provenance and post-transplant hook) and the land-surface
    preparer, bound into a RelocationRunner."""
    from woof.runtime import (RealRelocationChildPreparer,
                               build_real_relocation_runner)
    from woof.core.relocation_runner import RelocationRunner
    from woof.ingest.relocation_init import (
        REAL_DATA_FOOTPRINT_REBUILT_STATICS, REAL_DATA_STRIP_FILL_SOURCE)
    from woof.static.lambert import LambertGrid
    from woof.static.sampling_contract import current_sampling_contract
    from test_nest_relocation_staging import _scaffold

    exp = _scaffold()
    child_dc = exp.domains[1]
    manual = _manual_config((ScheduledRelocationMove(60.0, 1, 0),))
    exp = SimpleNamespace(
        relocation=manual, domains=exp.domains, vertical=object(),
        start_time=None)
    grid = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.0, dx=1000.0, dy=1000.0, e_we=13, e_sn=13)
    node = SimpleNamespace(cfg=child_dc, grid=grid)
    model = SimpleNamespace(
        node=lambda gid: node, _input_catalog=object(),
        _prepared_by_grid_id={2: SimpleNamespace(static_sampling_contract=
            current_sampling_contract())},
        schedule=SimpleNamespace(
            period_ticks=60, clock=SimpleNamespace(tick_den=1)))
    runner = build_real_relocation_runner(exp, None, model, tmp_path)
    assert isinstance(runner, RelocationRunner)
    assert isinstance(runner.on_child_built, RealRelocationChildPreparer)
    assert runner.static_provenance == REAL_DATA_FOOTPRINT_REBUILT_STATICS
    initializer = runner.initializer
    assert initializer.strip_fill_source == REAL_DATA_STRIP_FILL_SOURCE
    assert initializer.donor_alignment_frame_width == (
        int(child_dc.run.spec_bdy_width)
        + int(getattr(child_dc, "blend_width", 5)))
    assert callable(initializer.post_transplant)
    # Bounds-only [relocation] stays the manual/API mechanism: no runner.
    bounds_only = SimpleNamespace(relocation=RelocationConfig(
        enabled=True, grid_id=2), domains=exp.domains)
    assert build_real_relocation_runner(
        bounds_only, None, model, tmp_path) is None


# ---------------------------------------------------------------------------
# Restart across a move: marked, and loud
# ---------------------------------------------------------------------------

def test_fingerprint_marking_is_deterministic_and_order_sensitive():
    marked_a = mark_fingerprint_across_move("base", "record1")
    assert marked_a == mark_fingerprint_across_move("base", "record1")
    assert marked_a != "base"
    forward = mark_fingerprint_across_move(marked_a, "record2")
    backward = mark_fingerprint_across_move(
        mark_fingerprint_across_move("base", "record2"), "record1")
    assert forward != backward


def test_checkpoint_relocation_block_refuses_with_the_posture():
    from woof.io.restart import tree_fingerprint_mismatch_reason

    header = {
        "experiment_fingerprint": "marked",
        "relocation": {"moves": 2, "segment_id": "seg123",
                       "posture": RESTART_ACROSS_MOVE_POSTURE},
    }
    reason = tree_fingerprint_mismatch_reason(
        2, header, SimpleNamespace())
    assert "2 nest relocation(s)" in reason
    assert "seg123" in reason
    # The header still SAYS what it is on any mismatch -- what changed is
    # that the saying is a description, not a refusal to try.
    assert "resumes only into the run that wrote it" in reason


def test_moved_receipts_supply_the_checkpoint_header_block():
    """write_tree_restart reads model._relocation_receipts; the rows the
    runner produces must carry exactly the fields it stamps."""
    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(60.0, 1, 0),)))
    _advance(model, parent, child, 60)
    runner.on_period_begin(model, {1: parent.clock})
    moved = [row for row in model._relocation_receipts
             if row.get("event") == "relocated"]
    assert len(moved) == 1
    assert set(moved[0]) >= {"record_sha256", "segment_id", "grid_id"}


# ---------------------------------------------------------------------------
# The follower's restart state: the segment, the counter and the timers
# ---------------------------------------------------------------------------

def _two_moves(parent_plane, parent, child, model, **kwargs):
    runner = _runner(parent_plane, _manual_config((
        ScheduledRelocationMove(60.0, 1, 0),
        ScheduledRelocationMove(120.0, 0, 1))), **kwargs)
    for ticks in (60, 120):
        _advance(model, parent, child, ticks)
        runner.on_period_begin(model, {1: parent.clock})
    return runner


def test_the_follower_block_carries_the_segment_and_the_counter():
    import json

    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    runner = _two_moves(parent_plane, parent, child, model)

    block = json.loads(json.dumps(runner.state_json(), allow_nan=False))
    assert sorted(block) == ["last_move_t", "last_proposal_t",
                             "moves_executed", "segment"]
    assert block["moves_executed"] == 2
    assert block["segment"]["generation"] == 2
    assert block["segment"]["segment_id"] == runner._segment.segment_id
    # A manual itinerary is not a tracker and holds no hysteresis.
    assert block["last_proposal_t"] is None
    assert block["last_move_t"] is None


def test_a_restored_generation_continues_rather_than_resetting():
    """The generation counts moves off ONE base preparation.  A follower
    that resumes at generation 0 writes a receipt claiming a nest that has
    moved twice is where it was prepared, and the next move's record chains
    off the base instead of off its predecessor -- so the digest the
    checkpoint's fingerprint was folded from can never be reproduced."""
    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    runner = _two_moves(parent_plane, parent, child, model)
    block = runner.state_json()
    straight_through = runner._segment

    resumed_plane, resumed_parent, resumed_child = _cpu_tree()
    resumed_model = _model(resumed_parent, resumed_child)
    resumed = _runner(resumed_plane, _manual_config(
        (ScheduledRelocationMove(180.0, 1, 0),)))
    resumed.restore_state(block)
    assert resumed.moves_executed == 2
    assert resumed._segment.generation == 2
    assert resumed._segment.segment_id == straight_through.segment_id

    # The third move continues the chain: generation 3, and its record
    # names the SECOND move's digest as its predecessor.
    _advance(resumed_model, resumed_parent, resumed_child, 180)
    outcome = resumed.on_period_begin(resumed_model,
                                      {1: resumed_parent.clock})
    assert outcome["event"] == "relocated"
    assert resumed._segment.generation == 3
    assert resumed.moves_executed == 3
    assert (resumed._segment.records[-1].predecessor_sha256
            == straight_through.records[-1].sha256)

    # The control: a follower that forgot the segment restarts at 1.
    cold_plane, cold_parent, cold_child = _cpu_tree()
    cold_model = _model(cold_parent, cold_child)
    cold = _runner(cold_plane, _manual_config(
        (ScheduledRelocationMove(180.0, 1, 0),)))
    _advance(cold_model, cold_parent, cold_child, 180)
    cold.on_period_begin(cold_model, {1: cold_parent.clock})
    assert cold._segment.generation == 1
    assert cold.moves_executed == 1


def _tracker_config():
    return RelocationConfig(
        enabled=True, grid_id=2, max_move_parent_cells=4,
        min_overlap_fraction=0.25, cadence_seconds=None, moves=())


def test_a_tracker_follower_round_trips_its_cooldown_through_the_runner():
    """The follower entry carries the tracker's hysteresis, so the runner
    is the one seam a resume has to reach to restore both."""
    from woof.core.storm_tracking import FollowConfig, StormTracker

    parent_plane, _parent, _child = _cpu_tree()
    follow = FollowConfig(
        field="uh", threshold=50.0, fallback_threshold=40.0,
        search_margin_cells=10, min_shift_cells=1, max_shift_cells=4,
        cooldown_seconds=1800.0)
    tracker = StormTracker(follow)
    tracker._last_proposal_t = 60.0
    tracker._last_move_t = 60.0
    runner = _runner(parent_plane, _tracker_config(), provider=tracker)

    block = runner.state_json()
    assert block["last_proposal_t"] == 60.0 and block["last_move_t"] == 60.0

    resumed_tracker = StormTracker(follow)
    resumed = _runner(parent_plane, _tracker_config(),
                      provider=resumed_tracker)
    resumed.restore_state(block)
    assert resumed_tracker.state_json() == {"last_proposal_t": 60.0,
                                            "last_move_t": 60.0}
    assert resumed.state_json() == block


def test_a_follower_block_with_hysteresis_a_manual_provider_cannot_hold():
    """Refused rather than dropped: a tracker's cooldown silently thrown
    away lets the resumed nest move at the first cadence boundary."""
    parent_plane, _parent, _child = _cpu_tree()
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(60.0, 1, 0),)))
    with pytest.raises(RelocationRefusal, match="cooldown"):
        runner.restore_state({"segment": None, "moves_executed": 0,
                              "last_proposal_t": 60.0, "last_move_t": None})


def test_the_follower_block_tolerates_the_writers_own_keys():
    """kind/current_placement/declared_placement are the checkpoint
    writer's half of the same entry; the runner reads none of them, and an
    entry carrying them must not have to be stripped first."""
    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    runner = _two_moves(parent_plane, parent, child, model)
    entry = dict(runner.state_json(), kind="legacy",
                 current_placement=[9, 9], declared_placement=[7, 7])

    resumed_plane, _p, _c = _cpu_tree()
    resumed = _runner(resumed_plane, _manual_config(
        (ScheduledRelocationMove(180.0, 1, 0),)))
    resumed.restore_state(entry)
    assert resumed._segment.segment_id == runner._segment.segment_id

    with pytest.raises(RelocationRefusal, match="lineage"):
        resumed.restore_state(dict(entry, lineage={}))


def test_prepared_tree_route_wires_the_corridor_runner(tmp_path):
    """The corridor lift: build_prepared_tree_relocation_runner assembles
    the SAME RelocationRunner shape the case-data route does -- the
    shared initializer with the corridor-crop statics seam, the
    prepared-route preparer, corridor provenance -- and keeps the
    bounds-only and corridor-less postures."""

    from woof.core.relocation_runner import RelocationRunner
    from woof.runtime import (PreparedTreeRelocationChildPreparer,
                               build_prepared_tree_relocation_runner)
    from woof.static.corridor import (CORRIDOR_REBUILT_STATICS,
                                       CORRIDOR_STRIP_FILL_SOURCE,
                                       ChildStaticsCorridor)
    from woof.static.lambert import LambertGrid
    from test_nest_relocation_staging import _scaffold

    scaffold = _scaffold()
    child_dc = scaffold.domains[1]
    manual = _manual_config((ScheduledRelocationMove(60.0, 1, 0),))
    exp = SimpleNamespace(
        relocation=manual, domains=scaffold.domains, vertical=object(),
        start_time=None)
    exp.domain = {int(dc.grid_id): dc for dc in exp.domains}.__getitem__
    grid = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.0, dx=1000.0, dy=1000.0, e_we=13, e_sn=13)
    node = SimpleNamespace(cfg=child_dc, grid=grid)
    model = SimpleNamespace(
        node=lambda gid: node,
        schedule=SimpleNamespace(
            period_ticks=60, clock=SimpleNamespace(tick_den=1)))
    corridor = ChildStaticsCorridor(
        geometry={"grid_id": 2, "parent_grid_ratio": 3, "child_nx": 12,
                  "child_ny": 12, "corridor_nx": 36, "corridor_ny": 36},
        fields={}, cache_sha256="f" * 64)
    workspace = object()

    runner = build_prepared_tree_relocation_runner(
        exp, statics_corridor=corridor, model=model, outdir=tmp_path,
        radiation_workspace=workspace)
    assert isinstance(runner, RelocationRunner)
    assert isinstance(runner.on_child_built,
                      PreparedTreeRelocationChildPreparer)
    assert runner.on_child_built.data is None
    assert runner.on_child_built._radiation_workspace is workspace
    assert runner.static_provenance == CORRIDOR_REBUILT_STATICS
    initializer = runner.initializer
    assert initializer.static_provenance == CORRIDOR_REBUILT_STATICS
    assert initializer.strip_fill_source == CORRIDOR_STRIP_FILL_SOURCE
    assert callable(initializer.post_transplant)
    assert initializer.donor_alignment_frame_width == (
        int(child_dc.run.spec_bdy_width)
        + int(getattr(child_dc, "blend_width", 5)))

    # Bounds-only [relocation] stays runnerless, exactly as elsewhere.
    bounds_only = SimpleNamespace(relocation=RelocationConfig(
        enabled=True, grid_id=2), domains=scaffold.domains)
    assert build_prepared_tree_relocation_runner(
        bounds_only, statics_corridor=corridor, model=model,
        outdir=tmp_path) is None

    # A follow source with no corridor may not sneak past the preflight
    # refusal through this constructor.
    with pytest.raises(ValueError, match="verified statics corridor"):
        build_prepared_tree_relocation_runner(
            exp, statics_corridor=None, model=model, outdir=tmp_path)


def test_receipts_write_survives_a_transient_windows_reader(monkeypatch,
                                                            tmp_path):
    """A reader holding the receipts open must not kill the run.

    Windows refuses the atomic rename while any reader holds the
    destination; receipts exist to be read mid-run, and a tail killed a
    6 h forecast with WinError 5 here (measured).  Two transient
    refusals then success must land the payload; a PERMANENT reader
    downgrades to a warning with the .tmp preserved.
    """
    import os

    import woof.core.relocation_runner as rr

    target = tmp_path / "relocation_receipts.json"
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(5, "Access is denied", str(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(rr.os, "replace", flaky)
    monkeypatch.setattr(rr.time, "sleep", lambda _s: None)
    rr._atomic_json(target, {"ok": 1})
    assert json.loads(target.read_text()) == {"ok": 1}

    def always(src, dst):
        raise PermissionError(5, "Access is denied", str(dst))

    monkeypatch.setattr(rr.os, "replace", always)
    with pytest.warns(UserWarning, match="stayed stale"):
        rr._atomic_json(target, {"ok": 2})
    assert json.loads(target.read_text()) == {"ok": 1}
    assert json.loads(
        (tmp_path / "relocation_receipts.json.tmp").read_text()) == {"ok": 2}


# ---------------------------------------------------------------------------
# [relocation.containment]: the mover's parent slides, the mover holds still
# ---------------------------------------------------------------------------

def _domains3(history_interval_s=900.0):
    """The three-level stand-in: d03 tracked, d02 its parent."""
    return [
        SimpleNamespace(grid_id=1, parent_id=None, time_step=60,
                        time_step_fract_num=0, time_step_fract_den=1,
                        history_interval_s=history_interval_s),
        SimpleNamespace(grid_id=2, parent_id=1, time_step=None,
                        history_interval_s=history_interval_s),
        SimpleNamespace(grid_id=3, parent_id=2, time_step=None,
                        history_interval_s=history_interval_s),
    ]


def _build3(raw, run_seconds=3600.0):
    from woof.experiment import _build_relocation

    return _build_relocation(raw, "test.toml", _domains3(), run_seconds)


_PRESSURE_FOLLOW = dict(field="pressure", threshold=30.0,
                        level_hpa=850.0, search_margin_cells=10,
                        min_shift_cells=1, max_shift_cells=4,
                        cooldown_seconds=0.0)


def test_containment_parses_and_echoes():
    cfg = _build3({"relocation": {
        "enabled": True, "grid_id": 3, "max_move_parent_cells": 4,
        "cadence_seconds": 600.0, "follow": dict(_PRESSURE_FOLLOW),
        "containment": {"grid_id": 2, "deadband_cells": 8,
                        "max_move_parent_cells": 1,
                        "cadence_seconds": 1200.0}}})
    assert cfg.containment.grid_id == 2
    assert cfg.containment.deadband_cells == 8
    assert cfg.receipt()["containment"] == {
        "grid_id": 2, "deadband_cells": 8, "max_move_parent_cells": 1,
        "cadence_seconds": 1200.0}


def test_containment_must_name_the_movers_parent():
    # d02 tracked, containment naming d03 (a child, not the parent).
    with pytest.raises(ValueError, match="not the tracked mover's parent"):
        _build3({"relocation": {
            "enabled": True, "grid_id": 2,
            "follow": dict(_PRESSURE_FOLLOW),
            "containment": {"grid_id": 3}}})
    # The root can never slide, named before ancestry is even asked.
    with pytest.raises(ValueError, match="root or an invalid id"):
        _build3({"relocation": {
            "enabled": True, "grid_id": 3,
            "follow": dict(_PRESSURE_FOLLOW),
            "containment": {"grid_id": 1}}})


def test_containment_without_a_follow_source_refuses():
    with pytest.raises(ValueError, match="tracks nothing"):
        _build3({"relocation": {
            "enabled": True, "grid_id": 3,
            "containment": {"grid_id": 2}}})


def test_containment_cadence_must_land_on_root_steps():
    with pytest.raises(ValueError, match="containment.*whole number"):
        _build3({"relocation": {
            "enabled": True, "grid_id": 3,
            "follow": dict(_PRESSURE_FOLLOW),
            "containment": {"grid_id": 2, "cadence_seconds": 90.0}}})


def test_containment_unknown_key_refuses_by_name():
    with pytest.raises(ValueError, match="deadband_cell"):
        _build3({"relocation": {
            "enabled": True, "grid_id": 3,
            "follow": dict(_PRESSURE_FOLLOW),
            "containment": {"grid_id": 2, "deadband_cell": 8}}})


def _tree3(d03_start=(70, 56)):
    """d01 -> d02 (real CPU state) -> d03 (stub, earth-fixed under slides).

    ``d03_start`` places the mover inside d02: centred is (56, 56), so
    the deviation the containment leg answers is the offset from there.
    """
    from dataclasses import replace

    parent_plane, parent, child = _cpu_tree()
    d03_run = replace(child.cfg.run, nx=30, ny=30,
                      dx=child.cfg.run.dx / 3, dy=child.cfg.run.dy / 3,
                      dt=child.cfg.run.dt / 3)
    d03_cfg = replace(child.cfg, grid_id=3, parent_id=2,
                      i_parent_start=int(d03_start[0]),
                      j_parent_start=int(d03_start[1]),
                      parent_grid_ratio=3, run=d03_run)
    d03 = SimpleNamespace(
        cfg=d03_cfg, state=SimpleNamespace(), grid="d03-grid",
        parent=child,
        coupler=SimpleNamespace(relocate=lambda: {
            "rolling_tables": "INVALID"}),
        clock=SimpleNamespace(ticks=0), children=[], _started=True)
    child.children = [d03]
    return parent_plane, parent, child, d03


def test_earth_fixed_descendant_holds_still_while_its_parent_slides():
    """relocate_child with earth_fixed_descendants: the descendant's
    placement compensates exactly, its state is CARRIED (the object), and
    no reground handler is demanded for it."""
    from woof.core.nest_relocation import relocate_child

    parent_plane, _parent, child, d03 = _tree3()
    d03_state = d03.state
    receipt = relocate_child(
        child,
        i_parent_start=int(child.cfg.i_parent_start) + 2,
        j_parent_start=int(child.cfg.j_parent_start),
        initializer=_initializer(parent_plane),
        static_provenance="footprint-parametric synthetic statics (test)",
        state_digest=lambda _s: "digest",
        staging="device",
        earth_fixed_descendants=frozenset({3}))
    assert d03.cfg.i_parent_start == 70 - 2 * 3
    assert d03.cfg.j_parent_start == 56
    assert d03.state is d03_state
    row = receipt["descendants"][0]
    assert row["earth_fixed"] is True
    assert row["state_carried_bitwise"] is True
    assert row["placement_from"] == [70, 56]
    assert row["placement_to"] == [64, 56]
    assert row["coupler"] == {"rolling_tables": "INVALID"}


def _containment_runner(parent, child, d03, parent_plane, *,
                        deadband=8, receipts_path=None, mover_cap=4,
                        contain_cap=2, min_overlap_fraction=None):
    relocation = {
        "enabled": True, "grid_id": 3, "max_move_parent_cells": mover_cap,
        "cadence_seconds": 600.0, "follow": dict(_PRESSURE_FOLLOW),
        "containment": {"grid_id": 2, "deadband_cells": deadband,
                        "max_move_parent_cells": contain_cap}}
    if min_overlap_fraction is not None:
        relocation["min_overlap_fraction"] = min_overlap_fraction
    cfg = _build3({"relocation": relocation})
    schedule = SimpleNamespace(
        clock=SimpleNamespace(tick_den=1), period_ticks=60)
    runner = RelocationRunner(
        config=cfg, schedule=schedule,
        on_child_built=lambda *a, **k: None,
        provider=lambda parent_state, nest, t, **_kw: None,
        staging="device", receipts_path=receipts_path)
    runner.wire_containment(
        initializer=_initializer(parent_plane),
        on_child_built=lambda *a, **k: None,
        static_provenance="footprint-parametric synthetic statics (test)")
    parent.clock = _clock(600)
    child.clock = _clock(600)
    d03.clock = _clock(600)
    nodes = {1: parent, 2: child, 3: d03}
    model = SimpleNamespace(
        root=parent, node=lambda gid: nodes[int(gid)],
        nodes_by_grid_id=nodes,
        schedule=schedule,
        experiment_fingerprint="base-fingerprint",
        walk_parent_first=lambda: [parent, child, d03],
        _scratch_arena=None, _dycore_state_workspace=None)
    clocks = {1: _clock(600)}
    return runner, model, clocks


def test_containment_slides_the_parent_and_compensates_the_mover():
    parent_plane, parent, child, d03 = _tree3()
    d03_state = d03.state
    runner, model, clocks = _containment_runner(
        parent, child, d03, parent_plane)
    runner.on_period_begin(model, clocks)
    # The mover leg held (provider returns None); the containment row is
    # in the receipts before it.
    events = [r["event"] for r in runner.receipts]
    assert events == ["contained", "held"]
    contained = runner.receipts[0]
    # d03 span 10 in d02 cells; centered start (120-10)//2+1 = 56, so a
    # placement of 70 deviates +14; the slide ASKS for round(14/3) = 5
    # and its own cap of 2 cuts it, which the row now names rather than
    # rewriting the request in place and reporting them equal.
    assert contained["mover_deviation_cells"] == [14, 0]
    assert contained["requested_shift_parent_cells"] == [5, 0]
    assert contained["executed_shift_parent_cells"] == [2, 0]
    assert contained["clamped_by"] == ["containment.max_move_parent_cells"]
    assert child.cfg.i_parent_start == 85 + 2
    assert d03.cfg.i_parent_start == 70 - 6
    assert d03.state is d03_state
    assert contained["descendants"][0]["earth_fixed"] is True
    assert runner.containment_moves_executed == 1
    assert model.experiment_fingerprint != "base-fingerprint"


def test_containment_inside_the_deadband_is_silent():
    parent_plane, parent, child, d03 = _tree3()
    from dataclasses import replace

    d03.cfg = replace(d03.cfg, i_parent_start=58)   # dev +2 < 8
    runner, model, clocks = _containment_runner(
        parent, child, d03, parent_plane)
    runner.on_period_begin(model, clocks)
    events = [r["event"] for r in runner.receipts]
    assert events == ["held"]
    assert child.cfg.i_parent_start == 85
    assert runner.containment_moves_executed == 0


@pytest.mark.parametrize("collection", [False, True])
@pytest.mark.parametrize("containment_only", [False, True])
def test_containment_rebinds_health_before_recording_a_slide(
        tmp_path, collection, containment_only):
    from woof.core.relocation_runner import RelocationRunnerCollection

    parent_plane, parent, child, d03 = _tree3()
    receipt_path = tmp_path / "containment.json"
    runner, model, clocks = _containment_runner(
        parent, child, d03, parent_plane, receipts_path=receipt_path)
    runner.staging = "host"
    if containment_only:
        runner.cadence_periods *= 2
    mover = RelocationRunnerCollection((runner,)) if collection else runner
    old_state = child.state
    validators = {2: old_state, 3: d03.state}
    checked = []

    def validate(gid):
        assert gid == 2 and gid not in validators
        assert child.state is not old_state
        assert child.cfg.i_parent_start == 87
        assert not runner.receipts  # validation precedes a success receipt
        validators[gid] = child.state
        checked.append(gid)

    mover.on_period_begin(
        model, clocks, before_rebuild=lambda gid: validators.pop(gid),
        after_rebuild=validate)
    assert checked == [2]
    assert validators == {2: child.state, 3: d03.state}
    events = [r["event"] for r in runner.receipts]
    assert events == (["contained"] if containment_only else ["contained", "held"])
    assert json.loads(receipt_path.read_text())["receipts"][0]["event"] == "contained"


def test_containment_health_failure_stops_before_a_success_receipt(tmp_path):
    parent_plane, parent, child, d03 = _tree3()
    receipt_path = tmp_path / "containment.json"
    runner, model, clocks = _containment_runner(
        parent, child, d03, parent_plane, receipts_path=receipt_path)

    def unhealthy(gid):
        raise RuntimeError(f"unhealthy rebuilt ancestor {gid}")

    with pytest.raises(RuntimeError, match="unhealthy rebuilt ancestor 2"):
        runner.on_period_begin(model, clocks, after_rebuild=unhealthy)
    assert not runner.receipts
    assert not receipt_path.exists()


def test_containment_rebinds_riding_sibling_but_preserves_earth_fixed_health():
    from dataclasses import replace
    from test_nest_relocation_staging import _grandchild

    parent_plane, parent, child, tracked = _tree3()
    sibling = _grandchild(child, ratio=3, nx=60, ny=60)
    sibling.cfg = replace(sibling.cfg, grid_id=4)
    child.children = [tracked, sibling]
    runner, model, clocks = _containment_runner(
        parent, child, tracked, parent_plane)
    model.nodes_by_grid_id[4] = sibling
    runner.staging = "host"
    validators = {2: child.state, 3: tracked.state, 4: sibling.state}
    tracked_state = tracked.state
    checked = []

    def reground(*, node, plan, delta_parent_cells):
        assert node.cfg.grid_id == 4 and 4 not in validators
        node.state = _ramp_state(60, 60)
        return {"donor_alignment": {"pass": True}}

    def validate(gid):
        validators[gid] = model.node(gid).state
        checked.append(gid)

    runner.reground_descendant = reground
    runner.on_period_begin(
        model, clocks, before_rebuild=lambda gid: validators.pop(gid),
        after_rebuild=validate)
    assert checked == [2, 4]
    assert validators[4] is sibling.state
    assert validators[3] is tracked_state is tracked.state
    assert runner.receipts[0]["event"] == "contained"


def test_containment_without_route_wiring_refuses_on_the_ledger():
    parent_plane, parent, child, d03 = _tree3()
    runner, model, clocks = _containment_runner(
        parent, child, d03, parent_plane)
    runner.containment_initializer = None
    runner.containment_preparer = None
    runner.on_period_begin(model, clocks)
    events = [r["event"] for r in runner.receipts]
    assert events == ["containment_refused", "held"]
    assert child.cfg.i_parent_start == 85


def test_root_frame_shift_composes_the_slide_into_the_mover_translation():
    """The exact live failure, as arithmetic: after d02 slid (-1, -2) and
    d03 proposed (94, 92), the mover's true translation from its t0
    reference is (-21, -56) of its OWN cells -- the placement-difference
    formula said (0, -14) and the drift gate refused by exactly the
    missing (21, 42) cells (0.25 deg).  origin_in_frame_cells is the
    arithmetic the fix stands on."""
    from woof.static.corridor import origin_in_frame_cells

    def cfgs(d02_place, d03_place):
        return {
            1: SimpleNamespace(grid_id=1, parent_id=0,
                               i_parent_start=1, j_parent_start=1,
                               parent_grid_ratio=1),
            2: SimpleNamespace(grid_id=2, parent_id=1,
                               i_parent_start=d02_place[0],
                               j_parent_start=d02_place[1],
                               parent_grid_ratio=3),
            3: SimpleNamespace(grid_id=3, parent_id=2,
                               i_parent_start=d03_place[0],
                               j_parent_start=d03_place[1],
                               parent_grid_ratio=7),
        }

    ref = origin_in_frame_cells(cfgs((74, 46), (94, 94)), 3, 1)
    live = origin_in_frame_cells(cfgs((73, 44), (94, 92)), 3, 1)
    shift = (live[0] - ref[0], live[1] - ref[1])
    assert shift == (-21, -56)
    legacy = ((94 - 94) * 7, (92 - 94) * 7)
    assert legacy == (0, -14)
    missing = (shift[0] - legacy[0], shift[1] - legacy[1])
    # ... which is exactly the d02 slide expressed in d03 cells.
    assert missing == (-1 * 21, -2 * 21)


def test_the_tracker_is_consulted_in_the_ALREADY_SLID_frame():
    """The ordering claim of 6802bb4d, asserted substantively.

    At a shared boundary the slide runs FIRST, so one opportunity
    re-centres AND follows instead of following and immediately
    un-centring what it just did.  The existing test above notices a
    swap through the receipt ORDER; this one notices it through the
    thing that actually matters -- what the tracker was handed.

    If the two legs were swapped, the provider would be given d03's
    PRE-slide placement (70) and its parent's pre-slide state, and would
    compute a shift relative to a frame that is about to move under it:
    a correct answer applied at the wrong place, which is precisely the
    class of defect commits 1f46814c and a9a05cc5 were.
    """
    parent_plane, parent, child, d03 = _tree3()
    seen = []

    runner, model, clocks = _containment_runner(
        parent, child, d03, parent_plane)
    parent_state_before = child.state

    def spy(parent_state, nest_footprint, t, **_kw):
        seen.append({
            "i_parent_start": int(nest_footprint.i_parent_start),
            "j_parent_start": int(nest_footprint.j_parent_start),
            "parent_state": parent_state,
        })
        return None

    runner.provider = spy
    runner.on_period_begin(model, clocks)

    assert len(seen) == 1
    # The slide was +2 d01 cells, so the earth-fixed compensation moved
    # d03 by -2 x 3 = -6.  The tracker must see 64, never 70.
    assert seen[0]["i_parent_start"] == 70 - 6
    assert seen[0]["i_parent_start"] == int(d03.cfg.i_parent_start)
    # ... and the parent state it reduces is the one the slide rebuilt.
    assert seen[0]["parent_state"] is child.state
    assert seen[0]["parent_state"] is not parent_state_before


def test_swapping_the_two_legs_would_be_caught():
    """The guard above is only a guard if the wrong order fails it.

    Rather than trust that, run the legs in the WRONG order explicitly
    and assert the tracker then sees the stale placement -- so the test
    above is pinned to a real difference and not to an invariant that
    holds either way.
    """
    parent_plane, parent, child, d03 = _tree3()
    runner, model, clocks = _containment_runner(
        parent, child, d03, parent_plane)
    seen = []
    runner.provider = lambda ps, fp, t, **_kw: (
        seen.append(int(fp.i_parent_start)) or None)

    # The mover leg first, by hand ...
    elapsed = 600.0
    node = model.node(3)
    runner.provider(node.parent.state, node.cfg, elapsed)
    # ... then the slide.
    runner._containment_opportunity(model, clocks, elapsed, None, None)

    assert seen == [70]                       # the STALE placement
    assert int(d03.cfg.i_parent_start) == 64  # slid afterwards


# ---------------------------------------------------------------------------
# The live stream: a move a map can draw
# ---------------------------------------------------------------------------
#
# The receipts have always carried every move.  What they cannot do is
# say anything WHILE THE RUN IS ALIVE: relocation_receipts.json is
# rewritten whole on every row, for a post-mortem to read afterwards, so
# a live map had no per-step stream to tail and the nest rectangles
# could not move.  These cells hold the runner to emitting into
# progress.jsonl as well, at the same instants the receipts record.


def _logged(model, tmp_path):
    """Attach a real step log to the fake model, the way a route does."""
    from datetime import datetime

    from woof.progress_log import StepLog, publish_step_log

    log = StepLog(start_time=datetime(2026, 8, 15, 0, 0, 0),
                  run_seconds=3600.0,
                  jsonl_path=tmp_path / "progress.jsonl")
    publish_step_log(model, log)
    return log


def _stream(tmp_path, log):
    from woof.progress_log import read_step_log

    log.close(status="SUCCESS")
    return read_step_log(tmp_path / "progress.jsonl")


def test_an_executed_move_reaches_the_per_step_stream(tmp_path):
    parent_plane, parent, child = _cpu_tree()
    start = (int(child.cfg.i_parent_start), int(child.cfg.j_parent_start))
    model = _model(parent, child)
    log = _logged(model, tmp_path)
    runner = _runner(parent_plane, _manual_config((
        ScheduledRelocationMove(60.0, 2, 0),)))

    for ticks in (0, 60, 120):
        _advance(model, parent, child, ticks)
        runner.on_period_begin(model, {1: parent.clock})

    moves = [r for r in _stream(tmp_path, log) if r["event"] == "nest_moved"]
    assert len(moves) == 1, "one executed move, one event"
    move, = moves
    assert move["domain"] == 2
    assert move["model_seconds"] == 60.0
    assert move["valid_time"] == "2026-08-15_00:01:00"
    assert move["placement_from"] == {"i_parent_start": start[0],
                                      "j_parent_start": start[1]}
    assert move["placement_to"] == {"i_parent_start": start[0] + 2,
                                    "j_parent_start": start[1]}
    assert move["executed_shift_parent_cells"] == [2, 0]
    assert move["requested_shift_parent_cells"] == [2, 0]


def test_a_held_cadence_boundary_puts_nothing_on_the_stream(tmp_path):
    """A hold redraws nothing, so it is not an event.

    The receipts carry every hold with its reason; putting one on the
    per-step stream would emit a record at every cadence boundary a
    storm sits still and give a consumer nothing to do with it.
    """
    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    log = _logged(model, tmp_path)
    runner = _runner(parent_plane, _manual_config(()),
                     provider=lambda *_a, **_k: None)

    for ticks in (0, 60, 120):
        _advance(model, parent, child, ticks)
        runner.on_period_begin(model, {1: parent.clock})

    assert [r["event"] for r in runner.receipts] == ["held", "held"]
    assert not [r for r in _stream(tmp_path, log)
                if r["event"].startswith("nest_")]


def test_a_containment_slide_reaches_the_stream_naming_both_domains(tmp_path):
    parent_plane, parent, child, d03 = _tree3()
    runner, model, clocks = _containment_runner(
        parent, child, d03, parent_plane)
    log = _logged(model, tmp_path)
    runner.on_period_begin(model, clocks)

    slide, = [r for r in _stream(tmp_path, log)
              if r["event"] == "containment_moved"]
    # The domain that MOVED is d02; d03 is the mover it moved for, and
    # it stayed earth-fixed under the slide.
    assert slide["domain"] == 2 and slide["mover"] == 3
    assert slide["placement_from"]["i_parent_start"] == 85
    assert slide["placement_to"]["i_parent_start"] == 87
    assert slide["mover_deviation_cells"] == [14, 0]
    assert slide["executed_shift_parent_cells"] == [2, 0]


def test_publishing_a_log_changes_no_decision_the_runner_makes(tmp_path):
    """The emitters are telemetry, and telemetry never steers.

    The same property the track writer has to have: if publishing a log
    changed one decision, the stream would be steering the model it
    claims to observe.
    """
    def run(with_log):
        parent_plane, parent, child = _cpu_tree()
        model = _model(parent, child)
        if with_log:
            _logged(model, tmp_path)
        runner = _runner(parent_plane, _manual_config((
            ScheduledRelocationMove(60.0, 2, 0),
            ScheduledRelocationMove(120.0, 0, 1))))
        for ticks in (0, 60, 120, 180):
            _advance(model, parent, child, ticks)
            runner.on_period_begin(model, {1: parent.clock})
        runner.close_receipt(model)
        return ([r["event"] for r in runner.receipts],
                (int(child.cfg.i_parent_start),
                 int(child.cfg.j_parent_start)),
                model.experiment_fingerprint)

    assert run(False) == run(True)
