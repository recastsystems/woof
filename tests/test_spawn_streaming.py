"""Spawn-at-trigger under ``[tiles]``: the three places it went wrong.

``tilestream/test_spawn_stream.py`` is the GPU half of this -- it measures
what a streamed domain's trigger plane actually contains.  This file is the
wiring half, and it runs on NumPy states with ``dycore.step`` mocked, the same
instrument rule ``tests/test_spawn_runner.py`` states: the objects under test
here are the executor, the leg walk and the route, and none of them inspects a
value.

The three defects, each with the test that reproduces it:

1.  THE ROUTE NEVER LOOKED AT ``[tiles]`` AT ALL.
    ``woof.runtime.run_experiment``'s domain-tree branch is the only route
    that drives a spawn (``walk_spawn_legs``); the only two routes that built
    a stepper mapping are the PREPARED forecast routes, and both of those
    refuse a dormant nest by name (``experiment.refuse_unrouted_spawn``).
    So spawning and streaming were on disjoint routes, and a spawn
    experiment configured ``mode = "on"`` ran fully resident with nothing
    saying so -- the block was parsed, validated and echoed into the
    resolved-config report, and then dropped.

2.  A NEWBORN WAS NEVER ADJUDICATED.
    ``streaming.steppers_for_tree`` walks the tree ONCE, before the run.
    ``walk_spawn_legs`` calls ``execute_experiment`` once per LEG with that
    same mapping, and a domain born at a leg boundary is not in it, so
    ``model.py``'s ``steppers.get(grid_id, step)`` resolved it to
    ``dycore.step``.  The comment that fallback carries is about a
    DELAYED-START child -- a domain that was in the tree when the mapping was
    built and simply had not started -- and it was written before streaming
    existed.

3.  THE TRIGGER READ A DEAD PLANE.
    ``SpawnWatch.evaluate`` takes its plane off ``parent_state``; a streamed
    domain's arrays live in its store and its ``DomainState`` stops changing
    at attach.  The GPU half measures the plane; here we pin that the leg
    walk performs the store->state publish before evaluating and the
    state->store adopt after, so the runner's own window reset reaches the
    domain.

WHY SIX OF THESE NINE CARRY ``@pytest.mark.gpu``

The instrument is NumPy end to end and nothing here asserts a value, but
``execute_experiment`` releases the CuPy default pool's UNUSED blocks on
every step and every period commit (``model._trim_default_pool``, and
``pool_trim_per_period`` defaults to True), so the six tests that drive a
real leg walk open a CUDA context anyway.  Unmarked, they were green on NO
machine and had been since v2.0.0: they hard-failed
``cudaErrorNoDevice`` under the release battery's mandated CPU-only stage-1
leg (``GPUWM_NO_LOCAL_GPU=1``, whose conftest backstop sets
``CUDA_VISIBLE_DEVICES=-1``), and an unmarked test is DESELECTED by the
rented card's ``-m gpu`` shard, so the shard never covered them either.

The marker is per test rather than a module-wide ``pytestmark`` on purpose.
The other three never call ``execute_experiment``, they are the only CPU
coverage this file has, and marking the module wholesale would trade one
silent gap for another -- ``tests/test_gpu_marker_discipline.py`` states the
rule this follows: over-marking is coverage loss wearing a safety costume.

The device contact is at CALL time, not import or collection time: this file
collects cleanly on a device-free box (9 collected, no CUDA context), so the
markers are the whole fix here.  Teaching ``_trim_default_pool`` to no-op
without a device would let all nine run CPU-only, but that is a change to a
default-on product path in the memory-pressure code, and it belongs to its
own ruling rather than to a marker repair.
"""

from __future__ import annotations

import ast
import inspect
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from woof.core import streaming
from woof.core.nest_spawn import SpawnConfig
from woof.core.spawn_runner import SpawnRunner
from test_nest_spawn_init import _experiment, _live_parent
from test_spawn_runner import _CountingCoupler, _CountingPreparer, _Prepared

LEG = 120.0


def _case():
    """The micro case of ``tests/test_spawn_runner.py``: d02 dormant at 120 s."""
    exp = _experiment()
    root = replace(exp.domains[0], history_interval_s=LEG,
                   run=replace(exp.domains[0].run, run_seconds=240.0,
                               output_interval_s=LEG))
    child = replace(exp.domains[1], history_interval_s=LEG,
                    spawn=SpawnConfig(trigger="time", at_s=LEG),
                    run=replace(exp.domains[1].run, run_seconds=240.0,
                                output_interval_s=LEG))
    return replace(exp, run_seconds=240.0, domains=(root, child))


def _model_for(dexp, runner):
    from woof.verify.cases.nest_ideal_common import assemble_idealized_tree

    parent, grids = _live_parent(dexp)
    model = assemble_idealized_tree(
        runner.active, parent.state, grids=(grids[0],),
        coupler_factory=_CountingCoupler,
        domain_preparer=lambda *_a, **_k: None)
    model._prepared_by_grid_id = {1: _Prepared()}
    return model


class _FakeStreamed(streaming.StreamedDomain):
    """A ``StreamedDomain`` that counts calls instead of sweeping tiles.

    Subclassed rather than duck-typed on purpose: ``streaming.is_streaming``
    is an ``isinstance`` check and the leg walk's publish/adopt seam keys off
    it, so a lookalike would exercise a branch the product never takes.
    """

    def __init__(self, grid_id, planes=None):
        self.grid_id = int(grid_id)
        self.calls = 0
        self.exchanges = []
        self._planes = dict(planes or {})
        self.decision = None
        self._run = None
        self._state = None
        # A REAL streamed domain always carries a clock: since
        # feat-moving-nest-stream, execute_experiment calls
        # StreamedDomain.impose_clock before every step and that method
        # REFUSES a domain with scalars=None (attach(scalars=None) is the
        # gate's CARRY NOTHING control).  A double that carried None was
        # only ever passing because the model loop had not yet learned to
        # drive the clock.
        self.scalars = {"elapsed_seconds": 0.0}
        self.host_store = True
        self.steps = 0
        self.report = {}

    def __call__(self, state, cfg, **kw):
        self.calls += 1

    def sync_to_state(self, *_a, **_k):
        """Scoped out, deliberately.

        feat-moving-nest-stream taught the executor to project a streamed
        parent's WHOLE store onto its state on the relocation cadence, and
        that projection has its own seam and its own gate
        (tilestream/test_moving_nest.py).  This double stands in for the
        SPAWN seam -- publish/adopt of one named plane -- and a double that
        also had to be a working whole-domain projection would be testing
        the other branch's code, not this one's.  Overridden rather than
        given a ``_state``, so a real projection appearing here would be a
        deliberate change and not an accident.
        """
        return 0

    @property
    def store(self):
        return self._planes

    def publish(self, names):
        self.exchanges.append(("publish", tuple(names)))
        return tuple(names)

    def adopt(self, names):
        self.exchanges.append(("adopt", tuple(names)))
        return tuple(names)


# ---------------------------------------------------------------------------
# 1. the route
# ---------------------------------------------------------------------------

def _same_expression(node, expression):
    return ast.dump(node) == ast.dump(ast.parse(expression, mode="eval").body)


def _only(nodes, description):
    assert len(nodes) == 1, description
    return nodes[0]


def _body_call(body, name):
    # A direct statement: the bare call, or the call bound to one name (the
    # spawn walk keeps each leg's executor report).
    return _only([
        statement.value for statement in body
        if isinstance(statement, (ast.Expr, ast.Assign))
        and isinstance(statement.value, ast.Call)
        and _same_expression(statement.value.func, name)
    ], f"expected one direct {name} call")


def _assert_spawn_streaming_ownership(tree):
    """Follow the tree dispatch, plan construction and both real consumers.

    Inspect direct statements in the owning scopes: a comment, unused
    nested function or call hidden in another branch cannot satisfy an edge.
    """
    functions = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
    route = functions["_run_experiment"]
    scheduled = functions["_run_built_experiment"]
    walk = functions["walk_spawn_legs"]

    # The final return is the multi-domain branch; the adaptive single-domain
    # call is nested earlier and supplies its already prepared stepper map.
    assert isinstance(route.body[-1], ast.Return)
    dispatch = route.body[-1].value
    assert isinstance(dispatch, ast.Call)
    assert _same_expression(dispatch.func, "_run_built_experiment")
    assert [_same_expression(arg, expected) for arg, expected in
            zip(dispatch.args, ("exp", "data", "outdir", "model"))] == [True] * 4
    assert len(dispatch.args) == 4
    assert not any(kw.arg in (None, "prepared_steppers")
                   for kw in dispatch.keywords), "tree must enter plan construction"
    defaults = dict(zip((arg.arg for arg in scheduled.args.kwonlyargs),
                        scheduled.args.kw_defaults))
    assert _same_expression(defaults["prepared_steppers"], "None")
    model = _only([statement for statement in route.body[:-1]
                   if isinstance(statement, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "model"
                           for target in statement.targets)], "tree model owner")
    assert _same_expression(model.value, "build_experiment(exp, data)")
    assert any(isinstance(statement, ast.ImportFrom)
               and statement.module == "woof.core.model"
               and any(alias.name == "build_experiment" and alias.asname is None
                       for alias in statement.names) for statement in route.body)

    writers = _only([statement for statement in scheduled.body
                    if isinstance(statement, ast.With)
                    and any(isinstance(item.context_expr, ast.Call)
                            and _same_expression(item.context_expr.func,
                                                 "PerDomainWrfoutWriters")
                            for item in statement.items)], "scheduled writer scope")
    plan = _only([statement for statement in writers.body
                  if isinstance(statement, ast.If)
                  and _same_expression(statement.test, "prepared_steppers is None")],
                 "tree planning branch")
    assignment = _only([statement for statement in plan.body
                        if isinstance(statement, ast.Assign)
                        and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name)
                        and statement.targets[0].id == "steppers"], "stepper plan owner")
    planner = assignment.value
    assert isinstance(planner, ast.Call)
    assert _same_expression(planner.func, "_streaming.steppers_for_tree")
    assert len(planner.args) == 2
    assert all(_same_expression(arg, expected) for arg, expected in
               zip(planner.args, ("model", "exp.tiles")))
    builders = _only([kw.value for kw in planner.keywords if kw.arg == "builders"],
                     "streaming requires real builders")
    assert _same_expression(builders, "_streaming.builders_for_tree(model, exp.tiles)")
    assert any(isinstance(statement, ast.ImportFrom)
               and statement.module == "woof.core"
               and any(alias.name == "streaming" and alias.asname == "_streaming"
                       for alias in statement.names) for statement in writers.body)

    execution = _only([statement for statement in writers.body
                       if isinstance(statement, ast.If)
                       and _same_expression(statement.test, "already_complete")],
                      "scheduled execution branches")
    ordinary = _only([statement for statement in execution.orelse
                      if isinstance(statement, ast.If)
                      and _same_expression(statement.test, "spawn_runner is None")],
                     "ordinary versus spawn dispatch")
    execute = _body_call(ordinary.body, "execute_experiment")
    spawn = _body_call(ordinary.orelse, "walk_spawn_legs")
    assert len(execute.args) == 1 and _same_expression(execute.args[0], "model")
    assert len(spawn.args) == 3
    assert all(_same_expression(arg, expected) for arg, expected in
               zip(spawn.args, ("model", "exp", "data")))
    for consumer in (execute, spawn):
        mapping = _only([kw.value for kw in consumer.keywords if kw.arg == "steppers"],
                        "both consumers require the planned mapping")
        assert _same_expression(mapping, "steppers")
    plan_index = writers.body.index(plan)
    execute_index = writers.body.index(execution)
    assert plan_index < execute_index
    assert not any(isinstance(node, ast.Name) and node.id == "steppers"
                   and isinstance(node.ctx, ast.Store)
                   for statement in writers.body[plan_index + 1:execute_index + 1]
                   for node in ast.walk(statement)), "planned mapping was replaced"

    assert walk.args.kwarg.arg == "execute_kwargs"
    assert "steppers" not in {arg.arg for arg in (*walk.args.args,
                                                 *walk.args.kwonlyargs)}
    loop = _only([statement for statement in walk.body
                  if isinstance(statement, ast.While)], "spawn leg loop")
    leg_execute = _body_call(loop.body, "execute_experiment")
    # Each leg passes the incoming keywords on (leg_kwargs is execute_kwargs,
    # less a schedule stop that belongs to a later leg).
    assert any(kw.arg is None and (_same_expression(kw.value, "execute_kwargs")
                                   or _same_expression(kw.value, "leg_kwargs"))
               for kw in leg_execute.keywords), "spawn walk dropped incoming steppers"
    for owner in (scheduled, walk):
        assert any(isinstance(statement, ast.ImportFrom)
                   and statement.module == "woof.core.model"
                   and any(alias.name == "execute_experiment" and alias.asname is None
                           for alias in statement.names) for statement in owner.body)
    return {"dispatch": dispatch, "planner": planner, "execute": execute,
            "spawn": spawn, "leg_execute": leg_execute}


def test_the_spawn_route_now_consults_the_streaming_block():
    """The public tree route carries its built mapping through both executors."""
    from woof import runtime

    tree = ast.parse(inspect.getsource(runtime))
    _assert_spawn_streaming_ownership(tree)
    # Keep these controls in the same CPU case: each removes one meaningful
    # edge without running a forecast or changing the source on disk.
    for broken_edge in ("dispatch", "plan-bypass", "builders", "execute",
                        "spawn", "leg_execute"):
        mutant = deepcopy(tree)
        edges = _assert_spawn_streaming_ownership(mutant)
        if broken_edge == "dispatch":
            edges["dispatch"].func.id = "unrelated_runner"
        elif broken_edge == "plan-bypass":
            edges["dispatch"].keywords.append(
                ast.keyword(arg="prepared_steppers", value=ast.Dict(keys=[], values=[])))
        elif broken_edge == "builders":
            edges["planner"].keywords = [kw for kw in edges["planner"].keywords
                                         if kw.arg != "builders"]
        elif broken_edge == "leg_execute":
            edges["leg_execute"].keywords = [kw for kw in edges["leg_execute"].keywords
                                             if kw.arg is not None]
        else:
            keyword = next(kw for kw in edges[broken_edge].keywords if kw.arg == "steppers")
            keyword.value = ast.Dict(keys=[], values=[])
        with pytest.raises(AssertionError):
            _assert_spawn_streaming_ownership(mutant)


def test_both_spawn_capable_routes_either_stream_or_refuse_by_name():
    """No route may take a spawn config and quietly ignore [tiles].

    The prepared routes refuse a dormant nest outright
    (``refuse_unrouted_spawn``); the real-data tree route now adjudicates.
    Those are the only two dispositions allowed.
    """
    from woof import prepared_domain_tree_forecast as tree
    from woof import prepared_single_domain_forecast as single

    src = inspect.getsource(single)
    assert "refuse_unrouted_spawn" in src
    assert "steppers_for_tree" in src
    # The prepared tree route spawns (docs/storm-nests.md) and adjudicates
    # each newborn against the cold plan that priced it.
    src = inspect.getsource(tree)
    assert "steppers_for_tree" in src
    assert "spawned_stepper_factory=_newborn_stepper" in src


# ---------------------------------------------------------------------------
# 2. the newborn's stepper
# ---------------------------------------------------------------------------

@pytest.mark.gpu  # execute_experiment's default-on pool trim opens a context
def test_a_newborn_inside_a_streamed_run_is_refused_not_silently_resident(
        monkeypatch):
    """THE DEFECT, reproduced and then refused.

    Without the adjudication the run completes happily and d02 integrates
    through ``dycore.step`` while d01 streams -- which is silently right when
    the newborn is small and an out-of-memory death when it is the big one,
    and in both cases the run certifies a spawn path that never once ran
    under the mode its config named.
    """
    from woof.runtime import walk_spawn_legs

    dexp = _case()
    monkeypatch.setattr("woof.core.dycore.step",
                        lambda state, cfg, **_k: None)
    runner = SpawnRunner.from_experiment(
        dexp, on_child_built=_CountingPreparer(), array_module=np)
    model = _model_for(dexp, runner)

    with pytest.raises(RuntimeError, match=r"born at a spawn boundary"):
        walk_spawn_legs(model, dexp, None, spawn_runner=runner,
                        writers=None, lbc_interval_s=None,
                        coupler_factory=_CountingCoupler,
                        validate_state=False,
                        steppers={1: _FakeStreamed(1)})


@pytest.mark.gpu  # execute_experiment's default-on pool trim opens a context
def test_the_defect_itself_with_the_adjudication_removed(monkeypatch):
    """THE CONTROL for the refusal above: what the shipped walk actually did.

    The adjudication is neutered back to the passthrough it replaced, and the
    run completes -- happily, healthily, with d02 taking every one of its six
    substeps through ``woof.core.dycore.step`` while d01 streams.  Nothing
    raised, nothing was logged, and the receipt the run writes says a nest
    was born.  This is the outcome the refusal exists to forbid, and it ships
    here so that "the refusal fires" is a statement about a real defect
    rather than about a branch nobody could reach.
    """
    from woof import runtime
    from woof.runtime import walk_spawn_legs

    dexp = _case()
    dycore_calls: list[int] = []
    monkeypatch.setattr(
        "woof.core.dycore.step",
        lambda state, cfg, **_k: dycore_calls.append(int(cfg.grid_id)))
    monkeypatch.setattr(
        runtime, "_adjudicate_newborn_steppers",
        lambda steppers, model, attached, factory: steppers)
    runner = SpawnRunner.from_experiment(
        dexp, on_child_built=_CountingPreparer(), array_module=np)
    model = _model_for(dexp, runner)

    parent = _FakeStreamed(1)
    walk_spawn_legs(model, dexp, None, spawn_runner=runner,
                    writers=None, lbc_interval_s=None,
                    coupler_factory=_CountingCoupler,
                    validate_state=False, steppers={1: parent})

    assert runner.spawns_executed == 1
    assert model._spawn_receipts[-1]["event"] == "closed"
    assert parent.calls == 4                 # d01 streamed, as configured
    assert dycore_calls == [2] * 6           # d02 did NOT, and said nothing


@pytest.mark.gpu  # execute_experiment's default-on pool trim opens a context
def test_the_factory_binds_the_newborn_and_the_parent_keeps_its_stepper(
        monkeypatch):
    """With a factory the newborn is adjudicated; the parent is untouched.

    The parent's identity matters as much as the newborn's binding: a
    ``StreamedDomain`` owns the domain's arrays in its store, so re-attaching
    one at a leg boundary would copy the ATTACH-TIME state over the store and
    discard every step the run had taken.  The walk must extend the mapping,
    never rebuild it.
    """
    from woof.runtime import walk_spawn_legs

    dexp = _case()
    dycore_calls: list[int] = []
    monkeypatch.setattr(
        "woof.core.dycore.step",
        lambda state, cfg, **_k: dycore_calls.append(int(cfg.grid_id)))
    runner = SpawnRunner.from_experiment(
        dexp, on_child_built=_CountingPreparer(), array_module=np)
    model = _model_for(dexp, runner)

    parent = _FakeStreamed(1)
    born: list[int] = []

    def factory(grid_id, node):
        born.append(int(grid_id))
        assert node is model.node(int(grid_id))
        return _FakeStreamed(grid_id)

    walk_spawn_legs(model, dexp, None, spawn_runner=runner,
                    writers=None, lbc_interval_s=None,
                    coupler_factory=_CountingCoupler,
                    spawned_stepper_factory=factory,
                    validate_state=False, steppers={1: parent})

    assert born == [2]
    # d01 stepped through its ORIGINAL streamed object across both legs
    # (2 steps per leg at dt=60 s over a 120 s leg).
    assert parent.calls == 4
    # d02 stepped through the newborn's streamed object, never the dycore:
    # 3 substeps per parent step across the second leg only.
    assert dycore_calls == []


@pytest.mark.gpu  # execute_experiment's default-on pool trim opens a context
def test_a_resident_run_never_consults_the_factory(monkeypatch):
    """``[tiles]`` absent must cost the spawn path exactly nothing.

    ``steppers_for_tree`` returns ``{}`` when the block is absent AND when
    ``auto`` finds every domain fits, so an empty mapping is the only signal
    that nothing streams -- and with it the walk must not refuse, must not
    call a factory it was not given, and must bind the dycore's own step.
    """
    from woof.runtime import walk_spawn_legs

    dexp = _case()
    stepped: list[int] = []
    monkeypatch.setattr(
        "woof.core.dycore.step",
        lambda state, cfg, **_k: stepped.append(int(cfg.grid_id)))
    runner = SpawnRunner.from_experiment(
        dexp, on_child_built=_CountingPreparer(), array_module=np)
    model = _model_for(dexp, runner)

    walk_spawn_legs(model, dexp, None, spawn_runner=runner,
                    writers=None, lbc_interval_s=None,
                    coupler_factory=_CountingCoupler,
                    validate_state=False, steppers={})
    assert stepped.count(1) == 4 and stepped.count(2) == 6


# ---------------------------------------------------------------------------
# 3. the trigger's plane
# ---------------------------------------------------------------------------

@pytest.mark.gpu  # execute_experiment's default-on pool trim opens a context
def test_the_leg_walk_publishes_the_consumer_plane_before_the_trigger_looks(
        monkeypatch):
    """store -> state before ``on_leg_boundary``, state -> store after.

    Order is the whole content of this test.  Publishing after the trigger
    would hand it the attach-time plane; adopting before the reset would put
    the pre-reset window back into the domain.
    """
    from woof.runtime import walk_spawn_legs

    dexp = _case()
    monkeypatch.setattr("woof.core.dycore.step",
                        lambda state, cfg, **_k: None)
    runner = SpawnRunner.from_experiment(
        dexp, on_child_built=_CountingPreparer(), array_module=np)
    model = _model_for(dexp, runner)
    root = model.root
    # nwp_diagnostics = 1's eagerly allocated windows.  Without them there is
    # no plane for any consumer to read and the walk must move nothing.
    ny, nx = int(root.cfg.run.ny), int(root.cfg.run.nx)
    for slot in ("uh_follow_window", "uh_spawn_window"):
        root.state.scratch((ny, nx), slot)

    parent = _FakeStreamed(1)
    order: list[object] = []
    real_boundary = runner.on_leg_boundary

    def spy(model_, *, t=None):
        order.append("evaluate")
        return real_boundary(model_, t=t)

    runner.on_leg_boundary = spy
    monkeypatch.setattr(_FakeStreamed, "publish",
                        lambda self, names: order.append(("publish", names)))
    monkeypatch.setattr(_FakeStreamed, "adopt",
                        lambda self, names: order.append(("adopt", names)))

    walk_spawn_legs(model, dexp, None, spawn_runner=runner,
                    writers=None, lbc_interval_s=None,
                    coupler_factory=_CountingCoupler,
                    spawned_stepper_factory=lambda gid, node: None,
                    validate_state=False, steppers={1: parent})

    assert [row[0] if isinstance(row, tuple) else row
            for row in order][:3] == ["publish", "evaluate", "adopt"], order
    # THIS consumer's slot and no one else's.  The follow window belongs to
    # the relocation runner, which resets it on its own cadence from inside
    # execute_experiment; publishing it here could undo a reset the tracker
    # had already made.
    assert order[0][1] == ("scratch/uh_spawn_window",)


@pytest.mark.gpu  # execute_experiment's default-on pool trim opens a context
def test_nothing_moves_when_the_diagnostic_is_off(monkeypatch):
    """``nwp_diagnostics = 0`` allocates no window; the seam is a no-op.

    The negative control for the publish seam: a state with no window slot
    must not reach ``publish`` at all, because ``publish`` REFUSES a name the
    store does not hold and a refusal here would break every streamed run
    that never asked for the diagnostic.
    """
    from woof.runtime import walk_spawn_legs

    dexp = _case()
    monkeypatch.setattr("woof.core.dycore.step",
                        lambda state, cfg, **_k: None)
    runner = SpawnRunner.from_experiment(
        dexp, on_child_built=_CountingPreparer(), array_module=np)
    model = _model_for(dexp, runner)
    assert model.root.state.existing_scratch("uh_spawn_window") is None

    parent = _FakeStreamed(1)
    walk_spawn_legs(model, dexp, None, spawn_runner=runner,
                    writers=None, lbc_interval_s=None,
                    coupler_factory=_CountingCoupler,
                    spawned_stepper_factory=lambda gid, node: None,
                    validate_state=False, steppers={1: parent})
    assert parent.exchanges == []


def test_the_streaming_inventory_carries_what_restart_deliberately_does_not():
    """Streaming adds tracker windows and carried strict acoustic mass buffers.

    The restart classification is right and stays right: a window means "max
    since that consumer last looked" and a checkpoint cannot know when the
    consumer will next look.  Streaming asks a different question -- does
    this survive the gap between two STEPS -- and the answer is yes, so the
    two inventories are allowed to differ and this pins exactly how.
    """
    from woof.core import uh_diag
    from woof.io import restart
    from tilestream import physics_inventory as physinv

    added = set(physinv.streaming_only_members())
    # Strict acoustic mass carriers survive the gap between steps too.
    # Exact restart now retains these too; tracker windows remain stream-only.
    acoustic_slots = {"wrf_exact_muts", "wrf_exact_mu_halo"}
    # The physics-tendency output's pre-microphysics vapor (3.0) is carried
    # through the stream and rewritten before its first use after a restart
    # (woof.core.physics_tendency_output; preflight lifetime 'carrying').
    assert added == ({f"scratch/{s}" for s in uh_diag.TRACKER_WINDOW_SLOTS}
                     | {"scratch/tendency_qv_before"})
    for slot in acoustic_slots:
        assert restart.classify_scratch_slot(slot) == "carry"
        assert slot not in restart.SERIALIZED_SCRATCH_SLOTS
        assert slot in restart.CHECKPOINTED_CARRIED_SCRATCH_SLOTS
    for slot in uh_diag.TRACKER_WINDOW_SLOTS:
        # "carry", not "rebuild", since feat-uh-accum: restart.py grew a
        # THIRD scratch class for exactly this pair -- genuine cross-step
        # state that a checkpoint deliberately does not save -- because
        # with only serialize/rebuild to say it in they were filed
        # "rebuild" and a streamed run then never carried them at all.
        # The claim this test makes is unchanged and is now checked
        # against the classifier that can express it: NOT serialized.
        assert restart.classify_scratch_slot(slot) == "carry"
    # up_heli_max is the same operator with a different resetter and it was
    # already carried, which is what makes the omission a classification
    # accident rather than a physics decision.
    assert restart.classify_scratch_slot("up_heli_max") == "serialize"
