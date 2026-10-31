"""The acceptance test for restart with a declared nest lifecycle.

The contract this file adjudicates is the whole point of the lifecycle
restart work: a run SPLIT at a checkpoint and resumed produces the same
frames and the same final state as the unbroken run, for any number of
segments, at every interesting instant in a lifecycle -- before the slot
fires, in the middle of an episode, exactly on the leg lattice, inside
the re-arm cooldown after a retirement, and in the middle of the second
episode.

What is real here: the experiment, the ``SpawnRunner`` and its policy
tables, ``walk_spawn_legs``, the executor, the clocks and schedules, the
tree checkpoint writer and reader, the lifecycle block, the resume
reconstruction, and the resume boundary pass.  What is stood in is
``woof.core.dycore.step`` -- NumPy states are not valid CUDA integrator
inputs, the same instrument rule ``test_spawn_runner`` states -- and the
history writer, whose file format is not what a restart contract is
about.  The stand-in integrator is a PURE FUNCTION OF STATE, with no
counter of its own, so a split run and an unbroken run see identical
inputs and any difference between them belongs to the restart path.

A "frame payload" below is the sha256 of the state arrays at the history
instant, taken under the pathname the writer set would have chosen for
that domain and episode.  That is the content a wrfout frame is derived
from; hashing the file instead would measure the NetCDF encoder.

Receipts are deliberately OUTSIDE the identity: each segment owns its own
``spawn_receipts.json`` and relocation ledger.  The trajectory and the
frames are the contract.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from woof.core import uh_diag
from woof.core.nest_lifecycle import RearmConfig, RetireConfig
from woof.core.nest_spawn import SpawnConfig
from woof.io import restart as restart_io

from test_nest_spawn_init import _live_parent
from test_spawn_runner import (_CountingCoupler, _CountingPreparer, _Prepared,
                               _leg_walk_case)

#: The leg cadence every case below stops on: the root's history interval,
#: which is what :func:`woof.runtime._spawn_leg_seconds` reads when no
#: ``[relocation]`` cadence is configured.
LEG = 120.0
START = datetime(1974, 4, 3, 12, 0, 0)


class _RestorableCoupler(_CountingCoupler):
    """The executor's coupler surface plus the two RESTORE seams.

    ``restore_tree_restart`` invalidates every child coupler after
    applying arrays -- the interpolation tables it holds describe the
    pre-restore parent -- and ``relocate_child`` rebuilds them.  A CPU
    tier has no device tables, so both are bookkeeping here.

    ``feedback_commit`` is NOT bookkeeping.  It moves a value the child
    computed into the parent, deterministically, because that is the
    transfer the retirement-feedback proof is about: a fixture whose
    feedback changes nothing cannot tell a parent that received its
    retiring child's last increment from one that did not.
    """

    #: Set False by the feedback-disabled control arm.
    feeds_back = True

    def invalidate(self):
        self.valid = False

    def relocate(self):
        self.valid = False
        return {"rebuilt": True}

    def feedback_commit(self, node, *, start=False):
        if not self.feeds_back or node.parent is None:
            return
        child = _float_fields(node.state)
        parent_fields = _float_fields(node.parent.state)
        shared = sorted(set(child) & set(parent_fields))
        if not shared:
            return
        name = shared[0]
        increment = np.float32(
            float(np.float64(child[name].sum()) % 1024.0))
        parent_fields[name] += increment
        _normalize_ring(node.parent.state, node.parent.cfg.run)


class _StarvedCoupler(_RestorableCoupler):
    """The control: every other seam identical, the transfer removed."""

    feeds_back = False


# ---------------------------------------------------------------------------
# The stand-in integrator: deterministic, and a pure function of state
# ---------------------------------------------------------------------------

def _normalize_ring(state, cfg) -> None:
    """Hold the spec-zone ring invariant the real producers hold.

    A restore ends by zeroing the ring of the microphysics latent-heating
    field (``normalize_spec_zone_ring_after_restore``): no WRF-valid
    trajectory can carry values there, because the real microphysics
    never writes the ring.  A stand-in that DOES write it produces a
    state the restore legitimately declines to reproduce, and the
    resulting "difference" would measure the fixture rather than the
    restart path.
    """
    from woof.core.microphysics import normalize_spec_zone_ring_after_restore

    normalize_spec_zone_ring_after_restore(state, cfg)


def _float_fields(state):
    """Every float32 state array the checkpoint serializes, by name."""
    out = {}
    for name in sorted(restart_io.STATE_SERIALIZED_ATTRS):
        value = getattr(state, name, None)
        if isinstance(value, np.ndarray) and value.dtype == np.float32:
            out[name] = value
    return out


def _det_step(state, cfg, **_kwargs):
    """One deterministic integrator step, derived from the state alone.

    No step counter, no clock read, no module-level accumulator: the
    output is a function of the input arrays and the grid id.  A segment
    that resumes from restored arrays therefore MUST reproduce the
    unbroken run's next state, and a difference can only come from a
    field the restore did not carry.

    The spec-zone ring normalization is not decoration.  A restore ends
    by zeroing the ring of the microphysics latent-heating field
    (``normalize_spec_zone_ring_after_restore``) because no WRF-valid
    trajectory can carry values there -- the real microphysics never
    writes the ring.  An integrator stand-in that DOES write it produces
    a state the restore legitimately declines to reproduce, and the
    resulting "difference" measures the fixture rather than the restart
    path.  Holding the invariant the real producer holds is what makes
    the comparison below mean what it says.
    """
    gid = np.float32(int(cfg.grid_id))
    fields = _float_fields(state)
    for array in fields.values():
        array *= np.float32(1.0009765625)
        array += gid
    _normalize_ring(state, cfg)

    # The consumer window the spawn/retire watches read.  Folding it here
    # puts it INSIDE the trajectory: a resume that restores an empty
    # window under-reads the next decision, which is exactly the defect
    # the persisted windows exist to prevent.
    window = state.existing_scratch(uh_diag.UH_SPAWN_WINDOW_SLOT)
    if window is not None:
        source = next(
            (array for array in fields.values()
             if array.ndim >= 2 and array.shape[-2:] == window.shape), None)
        if source is not None:
            plane = np.abs(source.reshape(-1, *window.shape)[0])
            np.maximum(window, plane, out=window)


# ---------------------------------------------------------------------------
# The case: a two-domain tree whose d02 is born, retires and comes back
# ---------------------------------------------------------------------------

def _case(*, run_seconds=720.0, restart_interval_s=60.0, spawn_at=LEG,
          retire=None, rearm=None, child_history_s=60.0):
    """A micro two-domain experiment on a 120 s leg lattice.

    The deterministic ``time`` forms throughout: a field trigger's instant
    moves with the physics, and what lands on disk then cannot be checked
    against a timetable, which is what makes it right in production and
    useless as evidence.
    """
    exp = _leg_walk_case()
    root = replace(
        exp.domains[0], history_interval_s=LEG,
        run=replace(exp.domains[0].run, run_seconds=run_seconds,
                    output_interval_s=LEG, specified=False,
                    restart_interval_s=restart_interval_s))
    child = replace(
        exp.domains[1], history_interval_s=child_history_s,
        spawn=SpawnConfig(trigger="time", at_s=spawn_at),
        retire=retire, rearm=rearm,
        run=replace(exp.domains[1].run, run_seconds=run_seconds,
                    output_interval_s=child_history_s,
                    restart_interval_s=restart_interval_s))
    # The restart alarm is a WHOLE-TREE cadence evaluated on the d01
    # clock, so it lives on the experiment and not on a domain's run.
    return replace(exp, run_seconds=run_seconds, domains=(root, child),
                   restart_interval_s=restart_interval_s)


#: Born at the first leg boundary, retired 120 s of episode age later,
#: re-armed after a 120 s cooldown, and allowed exactly two episodes.
EPISODIC = {
    "retire": RetireConfig(trigger="time", at_s=LEG, min_lifetime_s=0.0,
                           sustained_s=0.0),
    "rearm": RearmConfig(max_firings=2, cooldown_s=LEG),
}


def _build(dexp, coupler=_RestorableCoupler):
    """A fresh pre-spawn tree, its runner, and the route's collaborators."""
    from woof.core.spawn_runner import SpawnRunner
    from woof.verify.cases.nest_ideal_common import assemble_idealized_tree

    parent, grids = _live_parent(dexp)
    preparer = _CountingPreparer()
    runner = SpawnRunner.from_experiment(
        dexp, on_child_built=preparer, array_module=np)
    model = assemble_idealized_tree(
        runner.active, parent.state, grids=(grids[0],),
        coupler_factory=coupler,
        domain_preparer=lambda *_a, **_k: None)
    model._prepared_by_grid_id = {1: _Prepared()}
    model._activation_context = {"experiment": dexp}
    # The consumer window the folding step and the spawn watch share.
    model.root.state.scratch(
        (int(model.root.cfg.run.ny), int(model.root.cfg.run.nx)),
        uh_diag.UH_SPAWN_WINDOW_SLOT)
    return model, runner, preparer


# ---------------------------------------------------------------------------
# Running a segment, and hashing what it produced
# ---------------------------------------------------------------------------

class _StopAtCheckpoint(Exception):
    """A kill at the instant a checkpoint became durable."""


def _state_digest(state) -> str:
    digest = hashlib.sha256()
    for name, array in _float_fields(state).items():
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _frame_key(model, node, ticks, episodes) -> str:
    """The pathname the writer set would choose for this frame."""
    from woof.core.nest_lifecycle import output_episode
    from woof.io.wrfout import wrfout_filename

    gid = int(node.cfg.grid_id)
    valid = START + timedelta(seconds=ticks / node.clock.tick_den)
    declared = next((dc for dc in model._activation_context["experiment"]
                     .domains if int(dc.grid_id) == gid), None)
    episode = output_episode(declared, int(episodes.get(gid, 0)))
    name = wrfout_filename(valid, gid)
    return name if episode == 0 else f"d{gid:02d}/episode-{episode:03d}/{name}"


def _run(dexp, outdir, *, model=None, runner=None, stop_ticks=None,
         coupler=_RestorableCoupler):
    """Integrate one segment; return its frames, checkpoints and state.

    ``stop_ticks`` models the kill: the checkpoint at that tick is written
    and then the process stops, which is where a real split happens.  The
    checkpoint is taken at PERIOD_BEGIN, so the state at the raise is
    exactly the state the file holds.
    """
    from woof.runtime import walk_spawn_legs

    if model is None:
        model, runner, _ = _build(dexp, coupler)
    frames: dict[str, str] = {}
    checkpoints: dict[int, object] = {}

    def history_handler(tree, node, ticks):
        key = _frame_key(tree, node, ticks, runner.episodes)
        frames[key] = _state_digest(node.state)

    def restart_handler(tree, ticks):
        valid = START + timedelta(seconds=ticks / tree.schedule.clock.tick_den)
        checkpoints[int(ticks)] = restart_io.write_tree_restart(
            outdir, tree, valid)
        if stop_ticks is not None and int(ticks) == int(stop_ticks):
            raise _StopAtCheckpoint(int(ticks))

    try:
        walk_spawn_legs(
            model, dexp, None, spawn_runner=runner, writers=None,
            lbc_interval_s=None, coupler_factory=coupler,
            validate_state=False, history_handler=history_handler,
            restart_handler=restart_handler)
    except _StopAtCheckpoint:
        pass
    return SimpleResult(model=model, runner=runner, frames=frames,
                        checkpoints=checkpoints)


class SimpleResult:
    def __init__(self, *, model, runner, frames, checkpoints):
        self.model = model
        self.runner = runner
        self.frames = frames
        self.checkpoints = checkpoints

    @property
    def final_states(self) -> dict[int, str]:
        return {int(gid): _state_digest(node.state)
                for gid, node in self.model.nodes_by_grid_id.items()}


def _resume(dexp, path, outdir, *, stop_ticks=None,
            coupler=_RestorableCoupler):
    """The route's own restore chain, then the rest of the run."""
    from woof.runtime import (publish_lifecycle_runners,
                               remark_relocation_fingerprint,
                               restore_nest_followers, restore_nest_lifecycle)

    model, runner, _ = _build(dexp, coupler)
    publish_lifecycle_runners(model, spawn_runner=runner,
                              leg_seconds=LEG)
    peek = restart_io.read_tree_lifecycle_header(path, model)
    if peek.block is not None:
        restore_nest_lifecycle(
            model, dexp, peek, spawn_runner=runner, lbc_interval_s=None,
            coupler_factory=coupler)
        restore_nest_followers(model, peek, spawn_runner=runner)
    remark_relocation_fingerprint(model, peek)
    restart_io.restore_tree_restart(peek.root_path, model)
    return _run(dexp, outdir, model=model, runner=runner,
                stop_ticks=stop_ticks, coupler=coupler)


def _checkpoint_arrays(path):
    """Every ``state/`` and ``scratch/`` array of one checkpoint set.

    Headers are excluded on purpose: ``created``, ``producer`` and
    ``checkpoint_set_id`` legitimately differ between a run and its
    resumed continuation, and requiring them equal would be requiring the
    two runs to be the same process.
    """
    from pathlib import Path

    path = Path(path)
    instant = restart_io._TREE_RESTART_NAME.fullmatch(path.name).group(
        "instant")
    out = {}
    for member in sorted(path.parent.glob(f"gpuwmrst_d*_{instant}*.npz")):
        gid = int(member.name[10:12])
        with np.load(member, allow_pickle=False) as archive:
            for name in sorted(archive.files):
                if name.startswith(("state/", "scratch/")):
                    out[(gid, name)] = hashlib.sha256(
                        np.ascontiguousarray(archive[name]).tobytes()
                    ).hexdigest()
    return out


# ---------------------------------------------------------------------------
# The identity matrix
# ---------------------------------------------------------------------------
#
# Each point is one instant a checkpoint can land on in a lifecycle, and
# each is a different half of the restore path:
#
#   t =  60  pre-fire            -- the watch is still pending
#   t = 120  exactly-on-lattice  -- written BEFORE the boundary evaluated
#   t = 180  mid-episode-1       -- a live child mid-leg
#   t = 300  post-retire-cooldown-- d02 detached, the re-arm timer running
#   t = 420  mid-episode-2       -- a SECOND episode, numbered
#
# The mid-follow-between-moves point needs a per-domain follower and is
# covered separately by the follower identity test below.

MATRIX = [
    ("pre-fire", 60),
    ("exactly-on-lattice", 120),
    ("mid-episode-1", 180),
    ("post-retire-in-cooldown", 300),
    ("mid-episode-2", 420),
]


@pytest.fixture()
def integrator(monkeypatch):
    monkeypatch.setattr("woof.core.dycore.step", _det_step)
    return _det_step


@pytest.mark.parametrize("label,split_ticks", MATRIX,
                         ids=[label for label, _ in MATRIX])
def test_a_run_split_at_a_lifecycle_instant_is_bit_identical(
        integrator, tmp_path, label, split_ticks):
    """The acceptance bar, one matrix point at a time.

    Bit-identical or this fails.  Relaxing the comparison, or excluding a
    field to make it pass, would leave exactly the defect the whole
    lifecycle-restart series exists to prevent: a resumed run that looks
    like a continuation and is not one.
    """
    dexp = _case(retire=EPISODIC["retire"], rearm=EPISODIC["rearm"])

    straight = _run(dexp, tmp_path / "straight")
    first = _run(dexp, tmp_path / f"seg1-{split_ticks}",
                 stop_ticks=split_ticks)
    assert split_ticks in first.checkpoints, (
        f"no checkpoint landed at t = {split_ticks} s, so the "
        f"{label} matrix point was never exercised")
    second = _resume(dexp, first.checkpoints[split_ticks],
                     tmp_path / f"seg2-{split_ticks}")

    segmented = dict(first.frames)
    segmented.update(second.frames)
    assert segmented == straight.frames, (
        f"{label}: the split run's frames differ from the unbroken run's")
    assert second.final_states == straight.final_states, (
        f"{label}: the resumed run's final state differs")
    assert sorted(second.model.nodes_by_grid_id) == sorted(
        straight.model.nodes_by_grid_id)
    assert second.runner.episodes == straight.runner.episodes
    assert second.runner.spawns_executed == straight.runner.spawns_executed


def test_the_final_checkpoints_arrays_are_identical_across_a_split(
        integrator, tmp_path):
    """State AND scratch, member by member, at the run's last checkpoint.

    The frames prove the trajectory; this proves what the NEXT segment
    would resume from, which is the part a frame comparison cannot see --
    a rebuilt scratch slot that differs only shows up one segment later.
    """
    dexp = _case(retire=EPISODIC["retire"], rearm=EPISODIC["rearm"])
    straight = _run(dexp, tmp_path / "straight")
    first = _run(dexp, tmp_path / "seg1", stop_ticks=180)
    second = _resume(dexp, first.checkpoints[180], tmp_path / "seg2")

    last = max(straight.checkpoints)
    assert max(second.checkpoints) == last
    assert _checkpoint_arrays(second.checkpoints[last]) == \
        _checkpoint_arrays(straight.checkpoints[last])


def test_three_segments_chain_as_one_run(integrator, tmp_path):
    """A resumed run's OWN checkpoint has to be resumable in turn.

    Two splits, not one: the fingerprint a segment writes is chained to
    the move history it inherited, so a segment that starts its chain
    fresh writes checkpoints the third segment cannot address.  N here is
    3 because 3 is the smallest N that can tell the difference.
    """
    dexp = _case(retire=EPISODIC["retire"], rearm=EPISODIC["rearm"])
    straight = _run(dexp, tmp_path / "straight")

    first = _run(dexp, tmp_path / "seg1", stop_ticks=120)
    second = _resume(dexp, first.checkpoints[120], tmp_path / "seg2",
                     stop_ticks=420)
    third = _resume(dexp, second.checkpoints[420], tmp_path / "seg3")

    segmented = dict(first.frames)
    segmented.update(second.frames)
    segmented.update(third.frames)
    assert segmented == straight.frames
    assert third.final_states == straight.final_states
    assert third.runner.episodes == straight.runner.episodes


def test_an_unsplit_run_actually_exercises_the_whole_lifecycle(
        integrator, tmp_path):
    """Non-vacuity: the matrix is only evidence if the run it splits has
    the events the matrix names.  Two episodes, one retirement, and
    frames under both episode directories."""
    dexp = _case(retire=EPISODIC["retire"], rearm=EPISODIC["rearm"])
    straight = _run(dexp, tmp_path / "straight")

    assert straight.runner.spawns_executed == 2
    assert straight.runner.episodes[2] == 2
    assert 2 in straight.runner.retired_times
    assert any("episode-001" in key for key in straight.frames)
    assert any("episode-002" in key for key in straight.frames)
    assert len(straight.checkpoints) == 12


# ---------------------------------------------------------------------------
# The uh-trigger arm: the same identity where the DECISION reads a window
# ---------------------------------------------------------------------------

def test_a_uh_triggered_retirement_survives_a_split_identically(
        integrator, tmp_path):
    """A field trigger reads the consumer window, and the window is a
    fold since the watch last looked.  Split mid-fold, the resumed run
    must read the same maximum the unbroken run read -- which is the
    whole reason the windows ride the checkpoint rather than restarting
    empty."""
    retire = RetireConfig(trigger="uh", threshold=1.0e9, sustained_s=0.0,
                          min_lifetime_s=0.0)
    dexp = _case(retire=retire, rearm=RearmConfig(max_firings=2,
                                                  cooldown_s=LEG))
    straight = _run(dexp, tmp_path / "straight")
    first = _run(dexp, tmp_path / "seg1", stop_ticks=180)
    second = _resume(dexp, first.checkpoints[180], tmp_path / "seg2")

    segmented = dict(first.frames)
    segmented.update(second.frames)
    assert segmented == straight.frames
    assert second.final_states == straight.final_states
    assert second.runner.episodes == straight.runner.episodes


def test_the_window_a_resume_reads_is_the_window_the_run_folded(
        integrator, tmp_path):
    """The mechanism the test above depends on, named on its own: an
    empty window at resume is a different decision input, and a decision
    input that differs silently is the defect."""
    dexp = _case(retire=EPISODIC["retire"], rearm=EPISODIC["rearm"])
    first = _run(dexp, tmp_path / "seg1", stop_ticks=60)
    live = first.model.root.state.existing_scratch(
        uh_diag.UH_SPAWN_WINDOW_SLOT)
    assert live.any(), "the fixture never folded anything to carry"

    from woof.runtime import publish_lifecycle_runners

    resumed, runner, _ = _build(dexp)
    publish_lifecycle_runners(resumed, spawn_runner=runner, leg_seconds=LEG)
    peek = restart_io.read_tree_lifecycle_header(
        first.checkpoints[60], resumed)
    restart_io.restore_tree_restart(peek.root_path, resumed)
    carried = resumed.root.state.existing_scratch(
        uh_diag.UH_SPAWN_WINDOW_SLOT)
    assert carried.tobytes() == live.tobytes()


# ---------------------------------------------------------------------------
# Guard interaction: the duplicate-valid-time refusal across a split
# ---------------------------------------------------------------------------

def test_a_resumed_segment_publishes_the_boundary_frame_once(
        integrator, tmp_path):
    """The instant a checkpoint lands on belongs to both segments, and
    the resumed run must not publish it a second time: the atomic replace
    would destroy the first segment's frame with nothing saying so.  The
    suppression is the restore's own committed-history set."""
    dexp = _case(retire=EPISODIC["retire"], rearm=EPISODIC["rearm"])
    first = _run(dexp, tmp_path / "seg1", stop_ticks=120)
    second = _resume(dexp, first.checkpoints[120], tmp_path / "seg2")

    overlap = sorted(set(first.frames) & set(second.frames))
    assert not overlap, (
        f"the resumed segment re-published {overlap}, which the unbroken "
        "run wrote exactly once")


def test_a_real_same_run_duplicate_still_refuses(tmp_path):
    """The other half: the guard is not weakened by admitting a resume."""
    import datetime as _dt
    from types import SimpleNamespace

    from woof.io.wrfout import PerDomainWrfoutWriters

    writers = object.__new__(PerDomainWrfoutWriters)
    writers.output_dir = tmp_path
    writers.start_time = _dt.datetime(1974, 4, 3, 12, 0, 0)
    writers._metadata_by_grid_id = {2: {}}
    writers._archived_paths = []
    writers._episode_by_grid_id = {2: 1}
    writers._published_paths = set()
    writers._writers = {2: SimpleNamespace(submit=lambda *_a, **_k: None)}
    node = SimpleNamespace(cfg=SimpleNamespace(grid_id=2),
                           clock=SimpleNamespace(tick_den=1),
                           state=SimpleNamespace())

    writers.submit(node, 0)
    with pytest.raises(RuntimeError, match="THIS run"):
        writers.submit(node, 0)


def test_two_episodes_at_the_same_valid_time_land_in_disjoint_places(
        integrator, tmp_path):
    """A re-armed slot's frames cannot alias its previous episode's, so a
    cross-episode collision is not a duplicate and is not refused."""
    dexp = _case(retire=EPISODIC["retire"], rearm=EPISODIC["rearm"])
    straight = _run(dexp, tmp_path / "straight")

    by_episode = {}
    for key in straight.frames:
        if "/episode-" in key:
            episode = key.split("/episode-")[1][:3]
            by_episode.setdefault(episode, set()).add(key.rsplit("/", 1)[1])
    assert sorted(by_episode) == ["001", "002"]
    assert all("/episode-001/" not in key or "/episode-002/" not in key
               for key in straight.frames)
    assert len(set(straight.frames)) == len(straight.frames)


# ---------------------------------------------------------------------------
# Retirement feedback: proved, not built (plan Decision F)
# ---------------------------------------------------------------------------

def test_a_legs_final_period_feeds_every_active_child_back_once(
        integrator, tmp_path):
    """Schedule level.  A leg ends where retirement is decided, so the
    last thing a retiring episode's schedule must do is hand its parent
    the increment it computed.  The final period suppresses LOOP-POSITION
    feedback and the last-io tail re-issues exactly one FEEDBACK op per
    active child; without that op the retired nest's last leg of work is
    dropped on the floor and the parent integrates on without it."""
    dexp = _case(retire=EPISODIC["retire"], rearm=EPISODIC["rearm"])
    first = _run(dexp, tmp_path / "seg1", stop_ticks=180)
    schedule = first.model.schedule

    active_children = [int(node.cfg.grid_id)
                       for node in first.model.walk_parent_first()
                       if node.parent is not None]
    assert active_children == [2], "no live child, so nothing to prove"
    tail = [op for op in schedule.final_period if op.kind == "FEEDBACK"]
    assert [int(op.grid_id) for op in tail] == active_children
    assert len(tail) == len(active_children)


#: The parent's frame at the retirement instant: d02 is born at t = 120,
#: retires 120 s of episode age later, and the root's history cadence is
#: the leg, so t = 240 is both the retirement boundary and a d01 frame.
_RETIREMENT_FRAME = "wrfout_d01_1974-04-03_12_04_00"


def _feedback_arms(tmp_path, *, run_seconds=360.0):
    """Three arms that differ in exactly one thing each."""
    retiring = _case(retire=EPISODIC["retire"], rearm=EPISODIC["rearm"],
                     run_seconds=run_seconds)
    never = _case(run_seconds=run_seconds)
    return (_run(retiring, tmp_path / "retire"),
            _run(never, tmp_path / "no-retire"),
            _run(retiring, tmp_path / "starved", coupler=_StarvedCoupler))


def test_the_parent_state_at_retirement_is_the_state_without_retirement(
        integrator, tmp_path):
    """Runtime level, and the exact bar Decision F names.

    Retirement is decided at the boundary that ENDS a leg, and the leg's
    final period has already re-issued the child's feedback by then.  So
    the parent state at the retirement instant must equal the same run
    with retirement disabled: the retiring episode's last increment is
    committed before the detach.  Divergence AFTER that instant is the
    feature, not the defect, and is asserted here too -- an equality that
    also held afterwards would mean retirement did nothing.
    """
    retire_arm, never_arm, _starved = _feedback_arms(tmp_path)

    assert retire_arm.runner.retired_times.get(2) == 240.0
    assert 2 not in never_arm.runner.retired_times
    assert retire_arm.frames[_RETIREMENT_FRAME] == \
        never_arm.frames[_RETIREMENT_FRAME], (
            "the retiring episode's last feedback did not reach the parent "
            "before its subtree was detached")
    assert retire_arm.final_states[1] != never_arm.final_states[1], (
        "the two arms never diverged, so retirement changed nothing and "
        "the equality above is about a run with no retirement in it")


def test_the_feedback_equality_is_not_vacuous(integrator, tmp_path):
    """Non-vacuity, as the control the equality needs.

    With the child-to-parent transfer removed and everything else held
    identical, the parent's state at the retirement instant DIFFERS.  So
    the equality above is measuring the feedback that arrived, and is not
    measuring a fixture in which nothing ever moved.
    """
    retire_arm, _never, starved = _feedback_arms(tmp_path)

    assert starved.runner.retired_times.get(2) == 240.0, (
        "the control must retire on the same timetable, or it is a "
        "different experiment and not a control"
    )
    assert starved.frames[_RETIREMENT_FRAME] != \
        retire_arm.frames[_RETIREMENT_FRAME], (
            "removing the feedback transfer changed nothing, so this "
            "fixture cannot tell a fed-back parent from a starved one")
