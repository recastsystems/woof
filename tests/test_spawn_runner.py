"""Seam A: SpawnRunner activating a dormant nest inside a REAL runner walk.

The lane left the pieces proven in isolation -- the controller's trigger
arithmetic (tests/test_nest_spawn.py), the materializer's terrain
calibration (tests/test_nest_spawn_init.py) -- plus one hand-written
end-to-end that performed the leg surgery inline.  This file covers the
seam that replaces that inline sequence: the runner that a route mounts
once and consults at every leg boundary.

Everything below walks REAL clocks, a REAL schedule and the REAL
executor; only ``woof.core.dycore.step`` is mocked, because NumPy
states are not valid CUDA integrator inputs -- the same instrument rule
the spawn lane's own end-to-end states.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core.nest_spawn import SpawnConfig
from woof.core.spawn_runner import (SPAWN_RUNNER_CONTRACT, SpawnRunner,
                                     SpawnRunnerRefusal)
from woof.static.lambert import grids_from_projection_config
from test_nest_spawn_init import _experiment, _host, _live_parent

LEG = 120.0          # 2 parent steps at dt = 60 s; 6 child steps at ratio 3


@pytest.fixture()
def case():
    """A fresh two-domain micro case whose d02 is DORMANT on a time trigger."""
    exp = _experiment()
    parent, grids = _live_parent(exp)
    dormant = replace(exp.domains[1],
                      spawn=SpawnConfig(trigger="time", at_s=LEG))
    return {"exp": replace(exp, domains=(exp.domains[0], dormant)),
            "parent": parent, "grids": grids}


class _CountingCoupler:
    """The one-way coupler surface the executor drives, counted."""

    def __init__(self, child, feedback=0):
        self.child = child
        self.valid = False
        self.forces = 0

    def force(self, node):
        assert node is self.child
        self.valid = True
        self.forces += 1

    def feedback_prepare(self, node, out):
        out.payload = None

    def feedback_commit(self, node, *, start=False):
        pass

    def feedback_finalize(self, node, *, start=False):
        pass


def _leg(exp, seconds=LEG):
    """Trim an experiment view to one leg's length."""
    return replace(exp, run_seconds=seconds, domains=tuple(
        replace(dc, history_interval_s=seconds,
                run=replace(dc.run, run_seconds=seconds,
                            output_interval_s=seconds))
        for dc in exp.domains))


def _walk(exp, root_state, *, grids=None, child_initializer=None,
          coupler_factory=_CountingCoupler, resume_seconds=0.0):
    """Assemble and integrate one leg; return the model and its report.

    ``resume_seconds`` carries the assembled clocks to the tick the tree
    is actually at before integrating, which is what
    ``woof.runtime._retarget_tree_schedule`` does at every leg boundary.
    A leg is a WINDOW of one run's absolute clock, not a fresh run: the
    experiment view spans t = 0 to the leg's end and the executor derives
    its resume period from the root clock.  Modelling leg B as its own
    0-based run only ever worked because a spawned nest recorded no
    activation epoch; once it records one, an epoch of 120 s in a leg
    replayed from 0 would say the newborn starts at the END of it.
    """
    from woof.core.model import execute_experiment
    from woof.verify.cases.nest_ideal_common import assemble_idealized_tree

    kwargs = {}
    if child_initializer is not None:
        kwargs["child_initializer"] = child_initializer
    model = assemble_idealized_tree(
        exp, root_state,
        grids=(grids_from_projection_config(exp) if grids is None else grids),
        coupler_factory=coupler_factory,
        domain_preparer=lambda *_a, **_k: None, **kwargs)
    if resume_seconds:
        for node in model.walk_parent_first():
            clock = node.clock
            ticks = int(round(float(resume_seconds) * clock.tick_den))
            clock.ticks = ticks
            clock.step_count = max(
                0, (ticks - clock.spec.start_ticks) // clock.spec.step_ticks)
    return model, execute_experiment(model, validate_state=False)


# ---------------------------------------------------------------------------
# The headline: a dormant nest is born mid-walk and integrates in leg B
# ---------------------------------------------------------------------------

def test_spawn_fires_inside_a_real_two_leg_walk(case, monkeypatch):
    """Leg A integrates the root alone; the trigger fires at the boundary;
    leg B integrates the newborn, adopted through the runner's own
    initializer, with its node, clock and coupler attached by the
    ordinary leg-boundary rebuild."""
    stepped: list[int] = []
    monkeypatch.setattr("woof.core.dycore.step",
                        lambda state, cfg, **_k: stepped.append(cfg.grid_id))

    built: list[tuple] = []

    def on_child_built(initialized, child_dc, parent_node):
        built.append((int(child_dc.grid_id), initialized, parent_node))

    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=on_child_built, array_module=np)
    assert runner is not None and runner.pending == (2,)

    # ---- leg A: the dormant nest costs zero compute ---------------------
    pre = runner.active
    assert [dc.grid_id for dc in pre.domains] == [1]
    model_a, report_a = _walk(
        _leg(pre), case["parent"].state, grids=(case["grids"][0],))
    assert report_a.steps == 2 and stepped == [1, 1]

    # ---- the leg boundary: one call, the whole activation ---------------
    record = runner.on_leg_boundary(model_a)
    assert record is not None
    assert record["contract"] == SPAWN_RUNNER_CONTRACT
    assert record["event"] == "spawned"
    assert record["grid_ids"] == [2]
    assert record["active_grid_ids"] == [1, 2]
    assert record["pending_after"] == []
    assert runner.spawns_executed == 1
    assert runner.spawned == {2: (20, 18)}

    # The physics/land seam fired, with the newborn's own config.
    assert [gid for gid, _i, _p in built] == [2]
    assert built[0][2] is model_a.root

    # The birth certificate rode along, and the parent was not written.
    born = record["born"][0]
    assert born["placement"] == [20, 18]
    assert born["spawn_receipt"]["parent_bitwise_unchanged"] is True
    assert born["spawn_receipt"]["trigger"]["decision"] == "fired"
    assert born["spawn_receipt"]["atmosphere_source"]["kind"] == "parent-sint"
    assert "child_result" not in born["spawn_receipt"]

    # ---- leg B: the activated tree --------------------------------------
    act = record["experiment"]
    assert [dc.grid_id for dc in act.domains] == [1, 2]
    stepped.clear()
    model_b, report_b = _walk(
        _leg(act, 2 * LEG), model_a.root.state,
        child_initializer=record["child_initializer"],
        resume_seconds=LEG)

    child = model_b.node(2)
    assert child.parent is model_b.root
    assert child.state is built[0][1].state      # the newborn, not a rebuild
    assert child.clock is not None               # clock attached by rebuild
    assert child.coupler is not None             # coupler attached too
    assert report_b.steps == 2 + 6
    assert stepped.count(1) == 2 and stepped.count(2) == 6
    assert child.coupler.forces == 2             # one FORCE per parent step
    for name in ("thp", "mup", "u", "v", "w"):
        assert np.isfinite(_host(getattr(child.state, name))).all()


def test_the_birth_reaches_the_per_step_stream(case, monkeypatch, tmp_path):
    """A nest being born is an event a live view has to be told about.

    Before this, the only record of a birth was ``spawn_receipts.json``,
    which is written for a post-mortem: anything watching the run saw a
    new domain start stepping and had to infer what had happened, with
    no placement, no episode number and no position.
    """
    from datetime import datetime

    from woof.progress_log import (NEST_EVENTS, StepLog, publish_step_log,
                                    read_step_log)

    monkeypatch.setattr("woof.core.dycore.step", lambda *_a, **_k: None)
    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=lambda *_a: None, array_module=np)
    model, _ = _walk(_leg(runner.active), case["parent"].state,
                     grids=(case["grids"][0],))

    log = StepLog(start_time=datetime(2026, 8, 15, 0, 0, 0),
                  run_seconds=600.0, jsonl_path=tmp_path / "progress.jsonl")
    publish_step_log(model, log)
    assert runner.on_leg_boundary(model) is not None
    log.close(status="SUCCESS")

    records = [r for r in read_step_log(tmp_path / "progress.jsonl")
               if r["event"] in NEST_EVENTS]
    born, = records
    assert born["event"] == "nest_spawned"
    assert born["domain"] == 2
    assert born["parent"] == 1
    assert born["episode"] == 1
    assert born["placement"] == {"i_parent_start": 20, "j_parent_start": 18}
    assert born["model_seconds"] == pytest.approx(LEG)
    # A newborn has taken no step of its own and says so.
    assert born["step"] == 0
    # The tree is projected, so the birth has a place on a map -- read
    # off the newborn's OWN grid, which is what a rectangle is drawn at.
    assert born["lat"] is not None and born["lon"] is not None
    assert -90.0 <= born["lat"] <= 90.0


def test_a_run_that_never_fires_puts_nothing_on_the_stream(case, monkeypatch,
                                                           tmp_path):
    """A watch that has not fired is not an event.

    The leg BEFORE the trigger evaluates the same watch and decides
    nothing happened; the stream has to stay silent through it, or every
    leg boundary of every spawn run emits a record saying so.
    """
    from datetime import datetime

    from woof.progress_log import (NEST_EVENTS, StepLog, publish_step_log,
                                    read_step_log)

    monkeypatch.setattr("woof.core.dycore.step", lambda *_a, **_k: None)
    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=lambda *_a: None, array_module=np)
    model, _ = _walk(_leg(runner.active, LEG / 2), case["parent"].state,
                     grids=(case["grids"][0],))

    log = StepLog(start_time=datetime(2026, 8, 15, 0, 0, 0),
                  run_seconds=600.0, jsonl_path=tmp_path / "progress.jsonl")
    publish_step_log(model, log)
    assert runner.on_leg_boundary(model, t=LEG / 2) is None
    log.close(status="SUCCESS")

    assert not [r for r in read_step_log(tmp_path / "progress.jsonl")
                if r["event"] in NEST_EVENTS]


def test_a_second_boundary_after_the_fire_is_a_no_op(case, monkeypatch):
    """Once every watch has fired the runner stops doing work, and the
    active view keeps the newborn at its fired placement."""
    monkeypatch.setattr("woof.core.dycore.step",
                        lambda *_a, **_k: None)
    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=lambda *_a: None, array_module=np)
    model_a, _ = _walk(_leg(runner.active), case["parent"].state,
                       grids=(case["grids"][0],))
    assert runner.on_leg_boundary(model_a) is not None

    act = runner.active
    model_b, _ = _walk(_leg(act, 2 * LEG), model_a.root.state,
                       child_initializer=runner.child_initializer(),
                       resume_seconds=LEG)
    assert runner.pending == ()
    assert runner.on_leg_boundary(model_b) is None
    assert runner.spawns_executed == 1
    assert [dc.grid_id for dc in runner.active.domains] == [1, 2]


# ---------------------------------------------------------------------------
# Holding, and the closing account
# ---------------------------------------------------------------------------

def test_a_boundary_before_the_window_holds_and_says_so(case, monkeypatch):
    monkeypatch.setattr("woof.core.dycore.step",
                        lambda *_a, **_k: None)
    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=lambda *_a: None, array_module=np)
    # Half a leg: the manual trigger's instant has not arrived.
    model, _ = _walk(_leg(runner.active, 60.0), case["parent"].state,
                     grids=(case["grids"][0],))
    assert runner.on_leg_boundary(model) is None
    assert runner.spawns_executed == 0
    held = runner.receipts[-1]
    assert held["event"] == "held" and held["pending"] == [2]
    assert held["elapsed_seconds"] == 60.0
    # Nothing joined the tree.
    assert [dc.grid_id for dc in runner.active.domains] == [1]


def test_close_receipt_names_the_watch_that_never_fired(case):
    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=lambda *_a: None, array_module=np)
    entry = runner.close_receipt()
    assert entry["event"] == "closed"
    assert entry["spawns_executed"] == 0
    assert [row["grid_id"] for row in entry["never_fired"]] == [2]
    assert "reservation was held" in entry["never_fired"][0]["note"] or \
        "still open" in entry["never_fired"][0]["note"]


def test_receipts_are_written_when_a_path_is_given(case, tmp_path,
                                                   monkeypatch):
    import json

    monkeypatch.setattr("woof.core.dycore.step", lambda *_a, **_k: None)
    path = tmp_path / "spawn-receipts.jsonl"
    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=lambda *_a: None, array_module=np,
        receipts_path=path)
    model, _ = _walk(_leg(runner.active), case["parent"].state,
                     grids=(case["grids"][0],))
    runner.on_leg_boundary(model)
    runner.close_receipt()
    # One complete JSON object per line, appended as each boundary is
    # decided -- see woof.core.spawn_runner.RECEIPTS_SUFFIX for why this
    # is not one document rewritten every time.
    rows = [json.loads(line) for line
            in path.read_text(encoding="utf-8").splitlines()]
    assert {row["contract"] for row in rows} == {SPAWN_RUNNER_CONTRACT}
    assert [row["event"] for row in rows] == ["spawned", "closed"]


# ---------------------------------------------------------------------------
# The wiring itself: what the runner hands the controller
# ---------------------------------------------------------------------------

class _Node:
    def __init__(self, cfg, state, parent=None):
        self.cfg = cfg
        self.state = state
        self.grid = None
        self.parent = parent
        self._started = True


class _Model:
    """The three surfaces the runner reads off a model."""

    def __init__(self, nodes):
        self._nodes = {int(n.cfg.grid_id): n for n in nodes}
        self.root = nodes[0]
        #: The retirement pass reads membership off this map, exactly as
        #: woof.core.model.Model publishes it.
        self.nodes_by_grid_id = self._nodes

    def walk_parent_first(self):
        return [self._nodes[gid] for gid in sorted(self._nodes)]

    def node(self, gid):
        return self._nodes[int(gid)]


def test_live_child_footprints_are_handed_to_the_controller(case):
    """The exclusion rule only works if the runner actually reports the
    LIVE children; this pins the argument, which is the seam's job (the
    exclusion arithmetic itself is the controller's, proven upstream)."""
    exp = case["exp"]
    live_child = replace(exp.domains[1], spawn=None, grid_id=3,
                         run=replace(exp.domains[1].run, grid_id=3))
    with_live = replace(exp, domains=(*exp.domains, live_child))

    runner = SpawnRunner.from_experiment(
        with_live, on_child_built=lambda *_a: None, array_module=np)
    seen = {}

    def spy(parent_states, t, *, active_footprints=()):
        seen["parents"] = dict(parent_states)
        seen["footprints"] = tuple(active_footprints)
        seen["t"] = t
        return ()

    runner.controller.evaluate_all = spy
    root_node = _Node(exp.domains[0], case["parent"].state)
    child_node = _Node(live_child, object(), parent=root_node)
    assert runner.on_leg_boundary(_Model([root_node, child_node]),
                                  t=300.0) is None

    assert seen["t"] == 300.0
    assert sorted(seen["parents"]) == [1, 3]
    assert [fp.grid_id for fp in seen["footprints"]] == [3]
    fp = seen["footprints"][0]
    assert (fp.i_parent_start, fp.j_parent_start) == (
        live_child.i_parent_start, live_child.j_parent_start)


def test_the_materialized_config_is_the_adjudicated_one(case, monkeypatch):
    """A field trigger's placement is nothing like the declared
    placeholder, so the runner must materialize the config
    ``active_experiment`` produced -- not ``exp.domain(gid)``."""
    monkeypatch.setattr("woof.core.dycore.step", lambda *_a, **_k: None)
    seen = {}

    real = None

    def spy(child_dc, parent_node, **kwargs):
        seen["placement"] = (int(child_dc.i_parent_start),
                             int(child_dc.j_parent_start))
        return real(child_dc, parent_node, **kwargs)

    import woof.ingest.nest_spawn_init as init_mod
    real = init_mod.spawn_child_from_parent
    monkeypatch.setattr(init_mod, "spawn_child_from_parent", spy)

    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=lambda *_a: None, array_module=np)
    model, _ = _walk(_leg(runner.active), case["parent"].state,
                     grids=(case["grids"][0],))
    record = runner.on_leg_boundary(model)

    fired = tuple(runner.spawned[2])
    assert seen["placement"] == fired
    activated = record["experiment"].domain(2)
    assert (activated.i_parent_start, activated.j_parent_start) == fired


# ---------------------------------------------------------------------------
# Refusals: the two things the mechanism must not invent
# ---------------------------------------------------------------------------

def test_no_dormant_nest_means_no_runner():
    exp = _experiment()
    assert SpawnRunner.from_experiment(
        exp, on_child_built=lambda *_a: None) is None
    with pytest.raises(SpawnRunnerRefusal, match="declares no dormant"):
        SpawnRunner(experiment=exp, on_child_built=lambda *_a: None)


def test_a_runner_without_the_physics_preparer_refuses(case):
    with pytest.raises(SpawnRunnerRefusal, match="on_child_built"):
        SpawnRunner(experiment=case["exp"], on_child_built=None)
    with pytest.raises(SpawnRunnerRefusal, match="on_child_built"):
        SpawnRunner.from_experiment(case["exp"], on_child_built=None)


# ---------------------------------------------------------------------------
# The run route's leg walk: the production consumer of on_leg_boundary
# ---------------------------------------------------------------------------

class _Prepared:
    """The prepared-case surface the attach step reads."""

    def __init__(self):
        self.static_fields = None
        self.geog_selection = None
        self.initial_result = None


class _CountingPreparer:
    """A route preparer: attaches nothing, records everything."""

    def __init__(self):
        self.prepared_by_grid_id = {}
        self.calls = []

    def __call__(self, initialized, child_dc, parent_node):
        gid = int(child_dc.grid_id)
        self.calls.append((gid, int(parent_node.cfg.grid_id)))
        self.prepared_by_grid_id[gid] = _Prepared()


def _leg_walk_case():
    """A micro case whose leg cadence (120 s) straddles the trigger."""
    exp = _experiment()
    root = replace(exp.domains[0], history_interval_s=120.0,
                   run=replace(exp.domains[0].run, run_seconds=240.0,
                               output_interval_s=120.0))
    child = replace(exp.domains[1], history_interval_s=120.0,
                    spawn=SpawnConfig(trigger="time", at_s=120.0),
                    run=replace(exp.domains[1].run, run_seconds=240.0,
                                output_interval_s=120.0))
    return replace(exp, run_seconds=240.0, domains=(root, child))


def test_walk_spawn_legs_births_the_nest_inside_one_run(monkeypatch):
    """The whole product story on the real executor: a dormant nest costs
    zero compute across the first leg, is born at the 120 s boundary from
    the LIVE parent, and integrates its own substeps for the rest of the
    run -- one `walk_spawn_legs` call, no caller-side leg bookkeeping."""
    from woof.runtime import walk_spawn_legs
    from woof.verify.cases.nest_ideal_common import assemble_idealized_tree

    dexp = _leg_walk_case()
    parent, grids = _live_parent(dexp)

    stepped: list[int] = []
    monkeypatch.setattr("woof.core.dycore.step",
                        lambda state, cfg, **_k: stepped.append(cfg.grid_id))

    preparer = _CountingPreparer()
    runner = SpawnRunner.from_experiment(
        dexp, on_child_built=preparer, array_module=np)
    assert runner.pending == (2,)

    # The tree the run starts with: the dormant nest is absent.
    pre = runner.active
    assert [dc.grid_id for dc in pre.domains] == [1]
    model = assemble_idealized_tree(
        pre, parent.state, grids=(grids[0],),
        coupler_factory=_CountingCoupler,
        domain_preparer=lambda *_a, **_k: None)
    model._prepared_by_grid_id = {1: _Prepared()}

    walk_spawn_legs(model, dexp, None, spawn_runner=runner,
                    writers=None, lbc_interval_s=None,
                    coupler_factory=_CountingCoupler,
                    validate_state=False)

    # The nest was born, once, from the live parent.
    assert runner.spawns_executed == 1
    assert preparer.calls == [(2, 1)]
    assert sorted(model.nodes_by_grid_id) == [1, 2]

    child = model.node(2)
    assert child.parent is model.root
    assert child in model.root.children
    assert child.coupler is not None and child._started

    # Zero compute while dormant, full cadence afterwards: the parent ran
    # 4 steps (2 per leg); the child ran 6 -- 3 per parent step, but only
    # across the SECOND leg.
    assert stepped.count(1) == 4
    assert stepped.count(2) == 6
    assert child.coupler.forces == 2

    # Both clocks land on the run end together.
    assert float(model.root.clock.elapsed_seconds) == 240.0
    assert float(child.clock.elapsed_seconds) == 240.0
    for name in ("thp", "mup", "u", "v", "w"):
        assert np.isfinite(_host(getattr(child.state, name))).all()

    # The closing account is on the model.
    assert model._spawn_receipts[-1]["event"] == "closed"
    assert model._spawn_receipts[-1]["spawns_executed"] == 1


def test_walk_spawn_legs_runs_straight_through_when_nothing_fires(
        monkeypatch):
    """A window that never opens costs its reservation and nothing else:
    the walk still reaches the end, the tree stays root-only, and the
    closing receipt names the watch that never fired."""
    from woof.runtime import walk_spawn_legs
    from woof.verify.cases.nest_ideal_common import assemble_idealized_tree

    exp = _leg_walk_case()
    # A trigger instant beyond the run: the watch can never fire.
    dexp = replace(exp, domains=(
        exp.domains[0],
        replace(exp.domains[1], spawn=SpawnConfig(trigger="time",
                                                  at_s=100000.0))))
    parent, grids = _live_parent(dexp)
    stepped: list[int] = []
    monkeypatch.setattr("woof.core.dycore.step",
                        lambda state, cfg, **_k: stepped.append(cfg.grid_id))

    preparer = _CountingPreparer()
    runner = SpawnRunner.from_experiment(
        dexp, on_child_built=preparer, array_module=np)
    model = assemble_idealized_tree(
        runner.active, parent.state, grids=(grids[0],),
        coupler_factory=_CountingCoupler,
        domain_preparer=lambda *_a, **_k: None)
    model._prepared_by_grid_id = {1: _Prepared()}

    walk_spawn_legs(model, dexp, None, spawn_runner=runner,
                    writers=None, lbc_interval_s=None,
                    coupler_factory=_CountingCoupler,
                    validate_state=False)

    assert runner.spawns_executed == 0
    assert preparer.calls == []
    assert sorted(model.nodes_by_grid_id) == [1]
    assert stepped.count(1) == 4 and stepped.count(2) == 0
    assert float(model.root.clock.elapsed_seconds) == 240.0
    closed = model._spawn_receipts[-1]
    assert closed["event"] == "closed"
    assert [row["grid_id"] for row in closed["never_fired"]] == [2]


def test_the_recorded_receipts_stay_json_serialisable(case, monkeypatch,
                                                      tmp_path):
    """self.receipts is durable evidence; the live objects the driver
    needs must ride on the RETURNED record only.  Caught by the
    idealized demo, whose receipt writer has no default=str."""
    import json

    monkeypatch.setattr("woof.core.dycore.step", lambda *_a, **_k: None)
    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=lambda *_a: None, array_module=np)
    model, _ = _walk(_leg(runner.active), case["parent"].state,
                     grids=(case["grids"][0],))
    record = runner.on_leg_boundary(model)
    assert record is not None
    # The driver's half: live objects present.
    assert "experiment" in record and "child_results" in record
    assert callable(record["child_initializer"])
    # The durable half: no live objects, and strictly serialisable.
    for row in runner.receipts:
        assert "experiment" not in row
        assert "child_initializer" not in row
        assert "child_results" not in row
    json.dumps(runner.receipts)      # raises if anything live leaked in


# ---------------------------------------------------------------------------
# The spawn block of the lifecycle restart header
# ---------------------------------------------------------------------------
#
# Every field below is asserted the same way: restore the block FAITHFULLY
# and restore it with that one field carrying the value a build which never
# persisted it would have, then take the SAME next boundary on the SAME
# model and show the two decisions differ.  A round-trip test that only
# checks the JSON comes back equal passes just as happily when nothing in
# the runner reads it, which is the instrument trap this lane exists inside.

def _dormant(exp, **over):
    """Re-declare d02 with the lifecycle tables one test needs."""
    from dataclasses import replace as _replace

    return _replace(exp, domains=(exp.domains[0],
                                  _replace(exp.domains[1], **over)))


def _stub_materializer(monkeypatch):
    """Keep the boundary DECISION real and the materialization out of it.

    The decision under test is the controller's plus the runner's
    lifecycle arithmetic; building a real newborn on every restore
    permutation would multiply the cost without touching what is asserted.
    """
    import woof.ingest.nest_spawn_init as init_mod

    monkeypatch.setattr(
        init_mod, "spawn_child_from_parent",
        lambda child_dc, parent_node, **kwargs: {
            "child_result": SimpleNamespace(
                state=None, grid=None, coord=None, domain=child_dc),
            "trigger": {"decision": "fired"}})


def _fresh(exp):
    return SpawnRunner.from_experiment(
        exp, on_child_built=lambda *_a: None, array_module=np)


def _restored(exp, block):
    runner = _fresh(exp)
    runner.restore_state(block)
    runner.controller.drain_receipts()      # the construction rows
    return runner


def _edit(block, path, value):
    """The block a build that did not persist ``path`` would have written."""
    import copy

    out = copy.deepcopy(block)
    cursor = out
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    return out


def _fired_runner(case, monkeypatch, **over):
    """Fire d02 at t = 120 s; return (runner, model, exp, root_node)."""
    _stub_materializer(monkeypatch)
    exp = _dormant(case["exp"], **over)
    runner = _fresh(exp)
    root = _Node(exp.domains[0], case["parent"].state)
    model = _Model([root])
    assert runner.on_leg_boundary(model, t=120.0)["grid_ids"] == [2]
    return runner, model, exp, root


def _with_live_child(runner, model, root):
    """Attach the fired child to the fake model, as the leg rebuild does."""
    child = _Node(runner.active.domain(2), object(), parent=root)
    model._nodes[2] = child
    return child


def _quiet_window(parent_state):
    """The spawn consumer's own UH window, allocated and quiet."""
    from woof.core.uh_diag import UH_SPAWN_WINDOW_SLOT

    return parent_state.scratch(np.asarray(parent_state.mup).shape,
                                UH_SPAWN_WINDOW_SLOT)


def test_the_spawn_block_is_exactly_the_headers_six_keys(case, monkeypatch):
    """The shape is the checkpoint schema's, so the writer can slot it in
    unchanged; a seventh key here is a header this build cannot read back."""
    import json

    runner, _model, exp, _root = _fired_runner(case, monkeypatch)
    block = runner.state_json()
    assert sorted(block) == ["episodes", "quiet_since", "retired",
                             "spawned", "spawns_executed", "watches"]
    assert sorted(block["spawned"]["2"]) == [
        "born_t", "current", "episode", "fired"]
    assert sorted(block["watches"]["2"]) == ["closed", "fired"]
    assert block["episodes"] == {"2": 1}
    assert block["retired"] == {}
    assert block["spawns_executed"] == 1
    # allow_nan=False is the header writer's posture, not a suggestion.
    json.dumps(block, allow_nan=False, sort_keys=True)


def test_spawn_block_floats_round_trip_bit_for_bit(case, monkeypatch):
    import json

    runner, _model, exp, _root = _fired_runner(case, monkeypatch)
    exact = 1234.5678901234567
    runner.birth_times[2] = exact
    block = runner.state_json()
    reloaded = json.loads(json.dumps(block, allow_nan=False))
    assert reloaded["spawned"]["2"]["born_t"] == exact
    assert reloaded == block
    assert _restored(exp, reloaded).state_json() == block


def test_a_nonfinite_lifecycle_timer_refuses_at_the_block(case, monkeypatch):
    """Refused here, where the message names the timer, rather than inside
    json.dumps(allow_nan=False), which aborts the whole checkpoint write."""
    runner, _model, _exp, _root = _fired_runner(case, monkeypatch)
    runner.birth_times[2] = float("nan")
    with pytest.raises(SpawnRunnerRefusal, match="born_t"):
        runner.state_json()


def test_the_block_separates_the_fired_placement_from_the_current_one(
        case, monkeypatch):
    """validate_spawn_placement must see the FIRED placement on the way
    back in; a follower's post-move position lives only on node.cfg."""
    from dataclasses import replace as _replace

    runner, model, exp, root = _fired_runner(case, monkeypatch)
    child = _with_live_child(runner, model, root)
    child.cfg = _replace(child.cfg, i_parent_start=24, j_parent_start=21)

    block = runner.state_json(model)
    assert block["spawned"]["2"]["fired"] == [20, 18]
    assert block["spawned"]["2"]["current"] == [24, 21]

    restored = _fresh(exp)
    current = restored.restore_state(block)
    assert current == {2: (24, 21)}
    assert restored.spawned == {2: (20, 18)}


def test_restored_fired_flag_stops_a_second_episode(case, monkeypatch):
    """fired -> re-fire: a watch restored as un-fired spawns the slot
    again at the very next boundary, silently creating episode 2."""
    runner, model, exp, _root = _fired_runner(case, monkeypatch)
    block = runner.state_json()
    assert block["watches"]["2"]["fired"] is True

    faithful = _restored(exp, block)
    assert faithful.pending == ()
    assert faithful.on_leg_boundary(model, t=240.0) is None
    assert faithful.spawns_executed == 1 and faithful.episodes[2] == 1

    naive = _restored(exp, _edit(block, ("watches", "2", "fired"), False))
    assert naive.pending == (2,)
    record = naive.on_leg_boundary(model, t=240.0)
    assert record is not None and record["grid_ids"] == [2]
    assert naive.spawns_executed == 2 and naive.episodes[2] == 2


def test_restored_closed_window_does_not_reopen(case, monkeypatch):
    """closed -> window reopen: a watch restored as open re-enters
    ``pending``, so ``needs_boundaries`` keeps the walk performing leg
    surgery for a slot that can never fire, and the run closes the same
    window twice."""
    exp = _dormant(case["exp"], spawn=SpawnConfig(
        trigger="uh", threshold=100.0, earliest_s=0.0, latest_s=100.0))
    runner = _fresh(exp)
    model = _Model([_Node(exp.domains[0], case["parent"].state)])
    assert runner.on_leg_boundary(model, t=200.0) is None
    block = runner.state_json()
    assert block["watches"]["2"]["closed"] is True
    assert runner.pending == () and runner.needs_boundaries is False

    faithful = _restored(exp, block)
    assert faithful.pending == () and faithful.needs_boundaries is False
    assert faithful.on_leg_boundary(model, t=320.0) is None
    assert faithful.receipts[-1]["pending"] == []
    assert not [row for row in faithful.receipts[-1].get("watch_receipts", ())
                if row["decision"] == "window-closed"]

    naive = _restored(exp, _edit(block, ("watches", "2", "closed"), False))
    assert naive.pending == (2,) and naive.needs_boundaries is True
    assert naive.on_leg_boundary(model, t=320.0) is None
    # The reopened watch was consulted again and shut the SAME window a
    # second time, 220 s after latest_s had already passed.
    assert [row for row in naive.receipts[-1].get("watch_receipts", ())
            if row["decision"] == "window-closed"]


def test_restored_quiet_timer_retires_at_the_instant_it_would_have(
        case, monkeypatch):
    """quiet_since -> hold instead of retire: the sustained-decay timer is
    episode-local and unrecoverable from the field, so a dropped one
    restarts the whole ``sustained_s`` window at the resume instant."""
    from woof.core.nest_lifecycle import RetireConfig

    runner, model, exp, root = _fired_runner(case, monkeypatch, retire=(
        RetireConfig(trigger="uh", threshold=60.0, sustained_s=900.0,
                     min_lifetime_s=0.0)))
    _with_live_child(runner, model, root)
    _quiet_window(case["parent"].state)

    assert runner.on_leg_boundary(model, t=240.0) is None   # quiet starts
    block = runner.state_json(model)
    assert block["quiet_since"] == {"2": 240.0}

    faithful = _restored(exp, block)
    record = faithful.on_leg_boundary(model, t=1140.0)
    assert record is not None and record["retired_grid_ids"] == [2]

    naive = _restored(exp, _edit(block, ("quiet_since", "2"), None))
    assert naive.on_leg_boundary(model, t=1140.0) is None
    assert naive.retired == set()


def test_restored_birth_time_keeps_the_episode_age(case, monkeypatch):
    """born_t -> a time-triggered retirement measures episode age, so a
    birth time re-stamped at resume postpones the retirement forever."""
    from woof.core.nest_lifecycle import RetireConfig

    runner, model, exp, root = _fired_runner(
        case, monkeypatch, retire=RetireConfig(trigger="time", at_s=600.0))
    _with_live_child(runner, model, root)
    block = runner.state_json(model)
    assert block["spawned"]["2"]["born_t"] == 120.0

    faithful = _restored(exp, block)
    record = faithful.on_leg_boundary(model, t=720.0)
    assert record is not None and record["retired_grid_ids"] == [2]

    naive = _restored(exp, _edit(block, ("spawned", "2", "born_t"), 600.0))
    assert naive.on_leg_boundary(model, t=720.0) is None


def test_restored_retirement_time_does_not_restart_the_cooldown(
        case, monkeypatch):
    """retired_t -> cooldown restart: re-stamping it at resume holds the
    slot dormant for a second full cooldown it already served."""
    from woof.core.nest_lifecycle import RearmConfig, RetireConfig

    runner, model, exp, root = _fired_runner(
        case, monkeypatch, retire=RetireConfig(trigger="time", at_s=0.0),
        rearm=RearmConfig(max_firings=2, cooldown_s=1800.0))
    _with_live_child(runner, model, root)
    assert runner.on_leg_boundary(model, t=240.0)["retired_grid_ids"] == [2]

    block = runner.state_json(model)
    assert block["retired"] == {"2": {"retired_t": 240.0, "episode": 1}}
    assert block["spawned"] == {}

    faithful = _restored(exp, block)
    record = faithful.on_leg_boundary(model, t=2040.0)
    assert record is not None and record["rearmed_grid_ids"] == [2]
    assert faithful.episodes[2] == 2

    naive = _restored(
        exp, _edit(block, ("retired", "2", "retired_t"), 2040.0))
    record = naive.on_leg_boundary(model, t=2040.0)
    assert record is None or record["rearmed_grid_ids"] == []
    assert naive.episodes[2] == 1


def test_restored_episode_count_bounds_the_re_arm(case, monkeypatch):
    """episode -> max_firings miscount: a slot declared for ONE firing
    fires a second time when the count restarts at zero."""
    from woof.core.nest_lifecycle import RearmConfig, RetireConfig

    runner, model, exp, root = _fired_runner(
        case, monkeypatch, retire=RetireConfig(trigger="time", at_s=0.0),
        rearm=RearmConfig(max_firings=1, cooldown_s=0.0))
    _with_live_child(runner, model, root)
    assert runner.on_leg_boundary(model, t=240.0)["retired_grid_ids"] == [2]
    block = runner.state_json(model)
    assert block["episodes"] == {"2": 1}

    faithful = _restored(exp, block)
    record = faithful.on_leg_boundary(model, t=360.0)
    assert record is None or record["rearmed_grid_ids"] == []
    assert faithful.episodes[2] == 1

    # The count lives in two places on purpose, so a lone edit is caught
    # by the block's own consistency gate; a build that never persisted
    # it writes zero in both.
    zeroed = _edit(_edit(block, ("episodes", "2"), 0),
                   ("retired", "2", "episode"), 0)
    naive = _restored(exp, zeroed)
    record = naive.on_leg_boundary(model, t=360.0)
    assert record is not None and record["rearmed_grid_ids"] == [2]
    assert naive.episodes[2] == 1     # the re-armed slot fired again


def test_never_fired_slots_carry_a_zero_episode(case, monkeypatch):
    """The map covers every declared slot, so a restored runner cannot
    tell "never fired" from "absent from the block"."""
    exp = _dormant(case["exp"])
    block = _fresh(exp).state_json()
    assert block["episodes"] == {"2": 0}
    assert block["watches"] == {"2": {"fired": False, "closed": False}}
    assert block["spawned"] == {} and block["retired"] == {}
    assert _restored(exp, block).pending == (2,)


# -- the block's own refusals ----------------------------------------------

def test_a_partial_spawn_block_refuses_by_name(case, monkeypatch):
    runner, _model, exp, _root = _fired_runner(case, monkeypatch)
    block = runner.state_json()
    for key in sorted(block):
        partial = {name: value for name, value in block.items()
                   if name != key}
        with pytest.raises(SpawnRunnerRefusal, match=key):
            _fresh(exp).restore_state(partial)


def test_an_unknown_spawn_block_key_refuses(case, monkeypatch):
    runner, _model, exp, _root = _fired_runner(case, monkeypatch)
    block = dict(runner.state_json(), lineage={})
    with pytest.raises(SpawnRunnerRefusal, match="lineage"):
        _fresh(exp).restore_state(block)


def test_a_slot_this_experiment_does_not_declare_refuses(case, monkeypatch):
    runner, _model, exp, _root = _fired_runner(case, monkeypatch)
    block = runner.state_json()
    block["watches"]["7"] = {"fired": False, "closed": False}
    with pytest.raises(SpawnRunnerRefusal, match="d07"):
        _fresh(exp).restore_state(block)


def test_an_episode_count_that_disagrees_with_itself_refuses(
        case, monkeypatch):
    runner, _model, exp, _root = _fired_runner(case, monkeypatch)
    block = _edit(runner.state_json(), ("spawned", "2", "episode"), 4)
    with pytest.raises(SpawnRunnerRefusal, match="episode"):
        _fresh(exp).restore_state(block)


def test_a_refused_block_leaves_the_runner_untouched(case, monkeypatch):
    """The candidate posture on_leg_boundary already keeps: a refusal
    mid-restore must not leave a half-seeded runner integrating."""
    runner, _model, exp, _root = _fired_runner(case, monkeypatch)
    block = runner.state_json()
    block["watches"]["7"] = {"fired": False, "closed": False}
    target = _fresh(exp)
    before = target.state_json()
    with pytest.raises(SpawnRunnerRefusal):
        target.restore_state(block)
    assert target.state_json() == before
