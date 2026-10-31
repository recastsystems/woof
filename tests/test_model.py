"""CPU gates for the Task-14 model loop and tree restart contract."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

import importlib.util

import numpy as np
import pytest

from woof.core.clock import (Op, STEP, Schedule, build_schedule,
                              execute_schedule, resolve_clock)
from woof.core.model import (PERIOD_BEGIN, DomainNode, ExperimentState,
                              ModelRuntimeStatus, execute_experiment,
                              experiment_fingerprint)
from woof.io import restart
from woof.static.lambert import grids_from_projection_config
from woof.verify.cases.nest_ideal_r1_moist import load_scaffold


class _Coupler:
    """Stands in for ``woof.core.nest.NestCoupler``.

    The keyword-only pair mirrors that constructor and must keep
    mirroring it.  A double whose signature is narrower than its
    collaborator's turns a widened production call into a TypeError
    raised from inside the double, which reads as a defect in the code
    under test rather than as the stale fixture it is.  Both values are
    held so a caller can be asserted to have passed them on.
    """

    def __init__(self, child, *, feedback: int = 0, smooth_option: int = 0):
        self.child = child
        self.feedback = int(feedback)
        self.smooth_option = int(smooth_option)
        self.valid = False
        self.calls = []

    def force(self, node):
        lead = node.parent.clock.ticks - node.clock.ticks
        interval = node.parent.clock.spec.step_ticks
        if lead != interval:
            raise RuntimeError("parent must lead child by one parent interval")
        self.valid = True
        self.calls.append("force")

    def invalidate(self):
        self.valid = False
        self.calls.append("invalidate")

    def feedback_prepare(self, node, out):
        self.calls.append("prepare")

    def feedback_commit(self, node, *, start=False):
        self.calls.append("commit")

    def feedback_finalize(self, node, *, start=False):
        self.calls.append("finalize")


class _HistoryPhysics:
    def __init__(self):
        self.mp_physics = 10
        self.refl_10cm = None


class _HistoryState:
    """Minimal concrete state for runtime's real REFL history handoff."""

    def __init__(self):
        self.elapsed_seconds = 0.0
        self.qv = np.ones((1,), dtype=np.float32)
        self.physics = _HistoryPhysics()


class _RecordingWriters:
    def __init__(self):
        self.frames = []
        self.drains = 0
        self.last_durable_wrfout = None

    @property
    def pending(self):
        return 0

    def submit(self, node, ticks, *, refl_field=None):
        if ticks:
            assert refl_field is not None
        self.frames.append((node.cfg.grid_id, ticks, refl_field))

    def drain(self):
        self.drains += 1


def _model_state():
    return SimpleNamespace(elapsed_seconds=0.0)


def _tree_model(exp, clock, *, fingerprint="fixture-fingerprint",
                state_factory=_model_state):
    schedule = build_schedule(exp, clock)
    clocks = clock.clocks()
    grids = grids_from_projection_config(exp)
    root = DomainNode(
        exp.domains[0], grids[0], state_factory(),
        clocks[1], None, [], None)
    child = DomainNode(
        exp.domains[1], grids[1], state_factory(),
        clocks[2], root, [], None)
    child.coupler = _Coupler(child)
    root.children.append(child)
    model = ExperimentState(
        root, {1: root, 2: child}, schedule, None, fingerprint)
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._scratch_arena = None
    model._dycore_state_workspace = None
    model._io_manager = None
    model._last_checkpoint = None
    return model


def _model():
    exp = load_scaffold()
    return exp, _tree_model(exp, resolve_clock(exp))


def _delayed_model():
    base = load_scaffold()
    domains = tuple(
        replace(
            dc, run=replace(dc.run, run_seconds=300.0),
            start_time=(
                base.start_time + timedelta(seconds=120)
                if dc.grid_id == 2 else dc.start_time))
        for dc in base.domains)
    exp = replace(base, run_seconds=300.0, domains=domains)
    model = _tree_model(
        exp, resolve_clock(exp, lbc_interval_s=60),
        fingerprint="delayed-fixture")
    model.node(2)._started = False
    return exp, model


def _history_model(run_seconds: float):
    exp = load_scaffold(variant="n2b")
    domains = tuple(replace(
        dc, history_interval_s=900.0,
        run=replace(dc.run, run_seconds=run_seconds,
                    output_interval_s=900.0,
                    restart_interval_s=1800.0))
        for dc in exp.domains)
    exp = replace(exp, run_seconds=run_seconds,
                  restart_interval_s=1800.0, domains=domains)
    return _tree_model(
        exp, resolve_clock(exp), fingerprint="history-fixture",
        state_factory=_HistoryState)


def test_experiment_fingerprint_allows_only_schedule_extension_fields():
    """Tree identity separates trajectory setup from resumable cadence."""
    exp = load_scaffold()
    catalog = SimpleNamespace(run_provenance={"forcing": "same"})
    extended_domains = tuple(replace(
        dc, history_interval_s=1200.0,
        run=replace(dc.run, run_seconds=7200.0,
                    output_interval_s=1200.0,
                    restart_interval_s=2400.0))
        for dc in exp.domains)
    extended = replace(
        exp, run_seconds=7200.0, restart_interval_s=2400.0,
        domains=extended_domains)

    assert experiment_fingerprint(exp, catalog) == \
        experiment_fingerprint(extended, catalog)
    assert experiment_fingerprint(exp, catalog) == experiment_fingerprint(
        replace(exp, acknowledgements=("expert-tuple-v1", "site-v2")),
        catalog)
    assert experiment_fingerprint(exp, catalog) != experiment_fingerprint(
        replace(extended, blend_width=extended.blend_width + 1), catalog)


def test_mixed_transition_implementation_is_restart_fingerprint_bound(
        monkeypatch):
    import woof.core.microphysics_transition as transition

    exp = load_scaffold(variant="n2b")
    root, child = exp.domains
    mixed = replace(exp, domains=(
        replace(root, run=replace(
            root.run, mp_physics=8, moist=True, moist_cq=True)),
        replace(child, run=replace(
            child.run, mp_physics=18, moist=True, moist_cq=True,
            nest_microphysics_transition=transition.MP8_TO_MP18_POLICY)),
    ))
    catalog = SimpleNamespace(run_provenance={"forcing": "same"})
    before = experiment_fingerprint(mixed, catalog)
    monkeypatch.setattr(
        transition, "transition_implementation_identity",
        lambda: {
            "driver_sha256": "1" * 64,
            "kernel_sha256": "2" * 64,
            "contract_sha256": "3" * 64,
        })
    assert experiment_fingerprint(mixed, catalog) != before


def test_ratio1_wk82_scaffold_is_cpu_buildable_and_morrison_complete():
    dry = load_scaffold(variant="n2a")
    moist = load_scaffold(variant="n2b")
    assert len(dry.domains) == len(moist.domains) == 2
    assert all(not dc.run.moist and dc.run.mp_physics == 0
               for dc in dry.domains)
    root, child = moist.domains
    assert (child.parent_grid_ratio,
            child.parent_time_step_ratio) == (1, 1)
    assert root.run.mp_physics == child.run.mp_physics == 10
    assert root.run.moist and child.run.moist
    assert (root.run.nx, root.run.ny, root.run.nz, root.run.dt) == (
        child.run.nx, child.run.ny, child.run.nz, child.run.dt)
    schedule = build_schedule(moist, resolve_clock(moist))
    assert Schedule.op_counts(schedule.interior_period) == {
        "STEP": 2, "FORCE": 1, "FEEDBACK": 1}


def test_disabled_restart_does_not_poll_per_period(monkeypatch):
    exp, model = _model()
    assert exp.restart_interval_s == 0.0

    def unexpected_poll(_clock):
        raise AssertionError("disabled restart alarm was polled")

    monkeypatch.setattr(
        "woof.core.clock.DomainClock.restart_due", unexpected_poll)
    monkeypatch.setattr("woof.core.dycore.step",
                        lambda *_args, **_kwargs: None)
    report = execute_experiment(
        model, validate_state=False, pool_trim_per_period=False)
    assert report.restarts == 0


def test_executor_wires_step_force_feedback_and_before_solve_alarms(
        monkeypatch):
    _exp, model = _model()
    step_calls = []
    history = []

    def fake_step(state, cfg, *, refl_10cm_due=False, **kwargs):
        step_calls.append((cfg.grid_id, state.elapsed_seconds,
                           bool(refl_10cm_due)))

    monkeypatch.setattr("woof.core.dycore.step", fake_step)
    report = execute_experiment(
        model, validate_state=False,
        history_handler=lambda tree, node, ticks: history.append(
            (node.cfg.grid_id, ticks)))

    assert report.steps == 20
    assert report.forces == report.feedback_calls == 10
    assert history == [(1, 0), (2, 0), (1, 30), (2, 30),
                       (1, 60), (2, 60)]
    assert [call for call in step_calls if call[2]] == [
        (1, 24.0, True), (2, 24.0, True),
        (1, 54.0, True), (2, 54.0, True)]
    assert {node.clock.ticks for node in model.walk_parent_first()} == {60}
    assert model._runtime_status.schedule_cursor == PERIOD_BEGIN


def test_health_debug_checks_each_domain_step_and_force(monkeypatch):
    """Attribution mode catches defects before the next physics consumer."""
    _exp, model = _model()
    phases = []

    class _RecordingValidator:
        def __init__(self, state):
            self.state = state

        def require_healthy(self, *, phase=None):
            phases.append((self.state, phase))

    monkeypatch.setattr(
        "woof.core.health.StateHealthValidator", _RecordingValidator)
    # This test records phase scheduling on scalar-only state doubles. The
    # actual GPU stability report is exercised by test_model_stability_gpu.
    monkeypatch.setattr("woof.core.streaming.step_health",
                        lambda *args, **kwargs: {"nan": False, "cfl": 0.0})
    monkeypatch.setattr("woof.core.dycore.step",
                        lambda *_args, **_kwargs: None)

    execute_experiment(
        model, health_debug=True, pool_trim_per_period=False)

    root = model.node(1).state
    child = model.node(2).state
    assert sum(state is root and phase == "pre-step.d01"
               for state, phase in phases) == 10
    assert sum(state is root and phase == "post-step.d01"
               for state, phase in phases) == 10
    assert sum(state is child and phase == "pre-step.d02"
               for state, phase in phases) == 10
    assert sum(state is child and phase == "post-step.d02"
               for state, phase in phases) == 10
    assert sum(state is child and phase == "post-force.d02-from-d01"
               for state, phase in phases) == 10


def test_executor_owns_shared_dycore_workspace_for_step_and_force(
        monkeypatch):
    """The real dispatcher guards every state-observing domain turn."""
    _exp, model = _model()

    class _TurnWorkspace:
        def __init__(self):
            self.owner = None
            self.acquisitions = []

        @contextmanager
        def acquire(self, owner):
            assert self.owner is None
            self.owner = owner
            self.acquisitions.append(owner)
            try:
                yield self
            finally:
                self.owner = None

    workspace = _TurnWorkspace()
    model._dycore_state_workspace = workspace

    def fake_step(state, cfg, **kwargs):
        assert workspace.owner == ("STEP", cfg.grid_id)

    child = model.node(2)
    original_force = child.coupler.force

    def checked_force(node):
        assert workspace.owner == ("FORCE", 2, 1)
        original_force(node)

    child.coupler.force = checked_force
    monkeypatch.setattr("woof.core.dycore.step", fake_step)
    execute_experiment(
        model, validate_state=False, pool_trim_per_period=False)

    assert workspace.owner is None
    assert workspace.acquisitions.count(("STEP", 1)) == 10
    assert workspace.acquisitions.count(("STEP", 2)) == 10
    assert workspace.acquisitions.count(("FORCE", 2, 1)) == 10


def test_first_step_failure_is_stamped_stepping_not_the_writer_phase(
        monkeypatch, tmp_path):
    """The tree runner publishes ``preparing:initialize-domain-writers`` and
    hands the model to this executor; before the fix nothing updated the
    phase again until the first period commit, so a step-0 physics failure
    was stamped as a writer-preparation failure and sent a real diagnosis
    down the wrong path.  Stepping now announces itself, on the same
    heartbeat run-progress.json already tracks.
    """
    import json
    import os

    import woof.supervisor as supervisor

    progress = supervisor.RuntimeHeartbeat(
        tmp_path / "run-progress.json", run_id="step0-run",
        config_sha256="a" * 64, started_at_utc="2026-08-05T00:00:00Z")
    progress.preparing("initialize-domain-writers")
    assert progress.last_phase == "preparing:initialize-domain-writers"

    _exp, model = _model()

    def failing_step(*_args, **_kwargs):
        raise RuntimeError("injected first-step physics failure")

    monkeypatch.setattr("woof.core.dycore.step", failing_step)
    with pytest.raises(RuntimeError, match="injected first-step"):
        execute_experiment(
            model, validate_state=False, progress_callback=progress,
            pool_trim_per_period=False)

    assert progress.last_phase == "stepping:outer-1"
    assert progress.last_step == 0
    # run-progress.json agrees: stepping began, zero outer steps completed.
    heartbeat = supervisor.read_heartbeat(tmp_path / "run-progress.json")
    assert heartbeat.status == "integrating"
    assert heartbeat.outer_step == 0

    # The capsule a worker would write from this failure tells the truth.
    monkeypatch.setattr(supervisor, "git_commit", lambda: "test-commit")
    capsule_path = supervisor.write_failure_capsule(
        tmp_path / "failure-capsule.json", run_id="step0-run",
        config_path=tmp_path / "case.toml", config_sha256="a" * 64,
        input_hashes={}, gpu=supervisor.GPUIdentity(
            "GPU-test", "610.74", "RTX 5090"),
        last_phase=progress.last_phase, last_step=progress.last_step,
        exception_type="RuntimeError",
        exception_message="injected first-step physics failure",
        exception_traceback="trace", last_durable_wrfout=None,
        last_checkpoint=None, worker_pid=os.getpid())
    capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
    assert capsule["last_phase"] == "stepping:outer-1"
    assert capsule["last_phase"] != "preparing:initialize-domain-writers"
    assert capsule["last_step"] == 0


def test_resumed_stepping_phase_carries_the_restored_step_number(monkeypatch):
    """On resume the transition names the next outer step, not outer-1."""
    _exp, model = _model()
    for node in model.walk_parent_first():
        node.clock.ticks = 30
        node.clock.step_count = 30 // node.clock.spec.step_ticks
        node.state.elapsed_seconds = 30.0
    model._resumed = True
    model._resume_committed_history_grid_ids = frozenset({1, 2})
    events = []

    def recording_callback(**event):
        events.append(event)

    monkeypatch.setattr("woof.core.dycore.step",
                        lambda *_args, **_kwargs: None)
    execute_experiment(
        model, validate_state=False, progress_callback=recording_callback,
        pool_trim_per_period=False)

    resumed_step = 30 // model.root.clock.spec.step_ticks
    assert events[0]["phase"] == f"stepping:outer-{resumed_step + 1}"
    assert events[0]["outer_step"] == resumed_step
    assert events[1]["phase"] == "post-d01-sync"
    assert events[1]["outer_step"] == resumed_step + 1


def test_history_alarm_t0_and_fifteen_minute_coincidence():
    exp = load_scaffold()
    domains = tuple(replace(
        dc, history_interval_s=900.0,
        run=replace(dc.run, run_seconds=900.0, output_interval_s=900.0))
        for dc in exp.domains)
    exp = replace(exp, run_seconds=900.0, domains=domains)
    schedule = build_schedule(exp, resolve_clock(exp))
    alarms = []
    execute_schedule(
        schedule,
        on_history=lambda gid, ticks: alarms.append((gid, ticks)))
    assert alarms == [(1, 0), (2, 0), (1, 900), (2, 900)]


def test_resume_suppresses_every_committed_domain_alarm():
    exp = load_scaffold()
    schedule = build_schedule(exp, resolve_clock(exp))
    clocks = schedule.clock.clocks()
    for clock in clocks.values():
        clock.ticks = 30
        clock.step_count = 5
    alarms = []
    execute_schedule(
        schedule, clocks=clocks, start_period=5,
        committed_initial_history_grid_ids={1, 2},
        on_history=lambda gid, ticks: alarms.append((gid, ticks)))
    assert alarms == [(1, 60), (2, 60)]


#: The checkpoint-question tests below stub ``woof.core.dycore.step`` and
#: hold a ``PhysicsDriver`` shell, and both modules import cupy when they
#: are imported.  No device is opened (the driver is never built), so the
#: tests run on any box whose install carries cupy and are declared skipped,
#: with the reason, where it is not installed at all.
needs_cupy_import = pytest.mark.skipif(
    importlib.util.find_spec("cupy") is None,
    reason=("cupy is not installed, and woof.core.dycore / "
            "woof.core.physics import it at module scope"))

@needs_cupy_import
def test_the_checkpoint_question_is_asked_before_step_0_of_every_domain(
        monkeypatch):
    """R-046: a run that will checkpoint names its physics setup FIRST.

    The defect: nothing asked whether a checkpoint could name a domain's
    physics until the writer did, at the first restart interval, with the
    forecast to that point already spent.  The gate is in
    ``execute_experiment`` and this pins WHERE it fires -- before any step,
    once per domain, parent first -- rather than only that it fires.
    """
    from woof.core.physics import PhysicsDriver
    from woof.io import restart as restart_mod

    order = []

    def fake_step(state, cfg, **_kwargs):
        order.append(("step", cfg.grid_id))

    monkeypatch.setattr("woof.core.dycore.step", fake_step)
    monkeypatch.setattr(
        restart_mod, "physics_setup_identity",
        lambda state, cfg: order.append(("identity", int(cfg.grid_id))) or {})

    model = _history_model(1800.0)
    for node in model.walk_parent_first():
        # A driver-shaped object without a driver's cost: the gate asks the
        # question of a PhysicsDriver, and building a real one needs a card.
        node.state.physics = object.__new__(PhysicsDriver)
    execute_experiment(model, validate_state=False,
                       restart_handler=lambda _tree, _ticks: None)

    assert order[:2] == [("identity", 1), ("identity", 2)], order[:4]
    assert order[2][0] == "step", order[:4]


@needs_cupy_import
def test_the_checkpoint_question_is_not_asked_of_a_foreign_physics_object(
        monkeypatch):
    """The identity is defined over a ``PhysicsDriver`` and over nothing else.

    ``woof/core/physics.py`` is the one production site that attaches a
    driver and it attaches a ``PhysicsDriver``, so a state carrying another
    object is a route this identity is not defined over: it would fail on an
    ATTRIBUTE, naming the gate rather than whatever attached the object, and
    it meets the restart writer at the writer's own door unchanged.
    """
    from woof.io import restart as restart_mod

    asked = []
    monkeypatch.setattr("woof.core.dycore.step",
                        lambda state, cfg, **_kwargs: None)
    monkeypatch.setattr(restart_mod, "physics_setup_identity",
                        lambda state, cfg: asked.append(cfg) or {})

    model = _history_model(1800.0)
    execute_experiment(model, validate_state=False,
                       restart_handler=lambda _tree, _ticks: None)
    assert asked == []


@needs_cupy_import
def test_the_checkpoint_question_is_one_function_both_doors_call(monkeypatch):
    """The rule about WHO is asked lives in one place (audit R-046).

    ``execute_experiment`` skips a foreign physics object and asks a
    driverless domain; ``woof.runtime.integrate_prepared_case`` used to
    ask ``physics_setup_identity`` itself, so the same foreign object raised
    there on an attribute -- a difference visible only to whichever door a
    user happened to run.  Both call this function; here is what it does.
    """
    from types import SimpleNamespace

    from woof.core.physics import PhysicsDriver
    from woof.io import restart as restart_mod

    asked = []
    monkeypatch.setattr(restart_mod, "physics_setup_identity",
                        lambda state, cfg: asked.append(state) or {})

    cfg = SimpleNamespace()
    foreign = SimpleNamespace(physics=SimpleNamespace(step=lambda: None))
    restart_mod.ask_checkpoint_physics_identity(foreign, cfg)
    assert asked == [], "a route this identity is not defined over"

    driverless = SimpleNamespace(physics=None)
    driver_state = SimpleNamespace(physics=object.__new__(PhysicsDriver))
    restart_mod.ask_checkpoint_physics_identity(driverless, cfg)
    restart_mod.ask_checkpoint_physics_identity(driver_state, cfg)
    assert asked == [driverless, driver_state]


@needs_cupy_import
def test_the_checkpoint_question_is_asked_once_per_domain(monkeypatch):
    """A delayed-start child is asked before step 0, and not again at it.

    The second ask was justified by a comment saying a delayed child "has
    no state and no physics driver" before step 0.  It has both:
    ``build_experiment`` builds every configured child's state and prepares
    its driver, withholding only ``_started``, and ``walk_parent_first``
    yields the whole tree -- so the pre-step-0 walk already asks it, hours
    before its activation, which is the earlier and better placement.  No
    second ask stands at activation: on_domain_start is reached only for
    configured delayed children, which that walk already covered.
    """
    from types import SimpleNamespace

    from woof.core.physics import PhysicsDriver
    from woof.io import restart as restart_mod
    from test_delayed_nest_activation import (DELAY_SECONDS, _stashing_step,
                                              _tree)

    exp, model = _tree(delay_s=DELAY_SECONDS)
    assert model.node(2)._started is False, "the fixture is not delayed"
    monkeypatch.setattr("woof.core.nest.NestCoupler", _Coupler)
    monkeypatch.setattr("woof.core.dycore.step", _stashing_step)

    asked = []
    monkeypatch.setattr(
        restart_mod, "physics_setup_identity",
        lambda state, cfg: asked.append(int(cfg.grid_id)) or {})
    for node in model.walk_parent_first():
        node.state.physics = object.__new__(PhysicsDriver)
    analysis = _HistoryState()
    analysis.physics = object.__new__(PhysicsDriver)
    case = SimpleNamespace(initial_result=SimpleNamespace(state=analysis))

    activations = []

    def initialize(node, clock):
        activations.append(int(node.cfg.grid_id))
        return SimpleNamespace(grid=node.grid, state=analysis), case

    execute_experiment(
        model, validate_state=False, delayed_child_initializer=initialize,
        skip_feedback_path=True,
        restart_handler=lambda _tree, _ticks: None)

    assert activations == [2], activations
    assert asked == [1, 2], (
        "the checkpoint question is asked once per domain, before step 0; "
        f"got {asked}")


@needs_cupy_import
def test_a_run_that_writes_no_checkpoints_is_not_asked_at_all(monkeypatch):
    """No ``restart_handler`` is exactly "this run writes no checkpoints"."""
    from woof.core.physics import PhysicsDriver
    from woof.io import restart as restart_mod

    asked = []
    monkeypatch.setattr("woof.core.dycore.step",
                        lambda state, cfg, **_kwargs: None)
    monkeypatch.setattr(restart_mod, "physics_setup_identity",
                        lambda state, cfg: asked.append(cfg) or {})

    model = _history_model(900.0)
    for node in model.walk_parent_first():
        node.state.physics = object.__new__(PhysicsDriver)
    execute_experiment(model, validate_state=False)
    assert asked == []


def test_real_history_handler_30_minute_restart_split_has_exact_frames(
        monkeypatch):
    """Converged F1: the production consume_refl_10cm path must survive
    the exact 30-minute split with no serialized child stash.  Pre-fix, the
    resumed d02 t=1800 callback consumes a missing stash and raises; post-fix,
    both committed boundary frames are suppressed and t=2700 is produced once.
    """
    from woof.core.refl import stash_refl_10cm
    from woof.runtime import _submit_tree_history_frame

    def fake_step(state, cfg, *, refl_10cm_due=False, **_kwargs):
        if refl_10cm_due:
            endpoint = int(round(state.elapsed_seconds + cfg.dt))
            stash_refl_10cm(
                state, np.asarray([cfg.grid_id, endpoint], dtype=np.int64))

    monkeypatch.setattr("woof.core.dycore.step", fake_step)

    straight = _history_model(2700.0)
    straight_writers = _RecordingWriters()
    straight._io_manager = straight_writers
    execute_experiment(
        straight, validate_state=False,
        history_handler=lambda _tree, node, ticks: (
            _submit_tree_history_frame(straight_writers, node, ticks)))

    first = _history_model(1800.0)
    first_writers = _RecordingWriters()
    first._io_manager = first_writers
    checkpoints = []
    execute_experiment(
        first, validate_state=False,
        history_handler=lambda _tree, node, ticks: (
            _submit_tree_history_frame(first_writers, node, ticks)),
        restart_handler=lambda _tree, ticks: checkpoints.append(
            (ticks, tuple((gid, tick)
                          for gid, tick, _refl in first_writers.frames))))
    assert checkpoints == [(1800, ((1, 0), (2, 0), (1, 900), (2, 900),
                                    (1, 1800), (2, 1800)))]

    resumed = _history_model(2700.0)
    for node in resumed.walk_parent_first():
        node.clock.ticks = 1800
        node.clock.step_count = 1800 // node.clock.spec.step_ticks
        node.state.elapsed_seconds = 1800.0
        assert node.state.physics.refl_10cm is None
    resumed._resumed = True
    resumed._resume_committed_history_grid_ids = frozenset({1, 2})
    resumed_writers = _RecordingWriters()
    resumed._io_manager = resumed_writers
    execute_experiment(
        resumed, validate_state=False,
        history_handler=lambda _tree, node, ticks: (
            _submit_tree_history_frame(resumed_writers, node, ticks)))

    straight_by_domain = {
        gid: [ticks for frame_gid, ticks, _refl in straight_writers.frames
              if frame_gid == gid]
        for gid in (1, 2)}
    split_frames = first_writers.frames + resumed_writers.frames
    split_by_domain = {
        gid: [ticks for frame_gid, ticks, _refl in split_frames
              if frame_gid == gid]
        for gid in (1, 2)}
    assert straight_by_domain == split_by_domain == {
        1: [0, 900, 1800, 2700], 2: [0, 900, 1800, 2700]}


def test_fifteen_minute_history_restart_and_lbc_seam_ordering():
    exp = load_scaffold()
    domains = tuple(replace(
        dc, history_interval_s=900.0,
        run=replace(dc.run, run_seconds=1800.0,
                    output_interval_s=900.0,
                    restart_interval_s=900.0))
        for dc in exp.domains)
    exp = replace(exp, run_seconds=1800.0,
                  restart_interval_s=900.0, domains=domains)
    schedule = build_schedule(
        exp, resolve_clock(exp, lbc_interval_s=900.0))
    events = []
    execute_schedule(
        schedule,
        on_history=lambda gid, ticks: events.append(
            ("history", gid, ticks)),
        on_restart=lambda ticks: events.append(("restart", 1, ticks)),
        on_lbc_reset=lambda ticks: events.append(("lbc_reset", 1, ticks)))
    assert [event for event in events if event[2] == 900] == [
        ("history", 1, 900), ("history", 2, 900),
        ("restart", 1, 900), ("lbc_reset", 1, 900)]


def test_child_step_asserts_while_restored_tables_are_invalid(monkeypatch):
    _exp, model = _model()
    model.schedule = Schedule(
        clock=model.schedule.clock,
        interior_period=(Op(STEP, 2, 0),),
        final_period=(Op(STEP, 2, 0),))
    monkeypatch.setattr("woof.core.dycore.step", lambda *args, **kwargs: None)
    with pytest.raises(AssertionError, match="INVALID nest tables"):
        execute_experiment(model, validate_state=False)


def test_tree_checkpoint_header_and_pending_work_refusal(monkeypatch,
                                                         tmp_path):
    _exp, model = _model()
    captured = []

    def fake_write(path, state, cfg, **kwargs):
        path.write_bytes(b"fixture")
        captured.append((cfg.grid_id, kwargs["tree_header"]))
        return path

    monkeypatch.setattr(restart, "write_restart", fake_write)
    root_path = restart.write_tree_restart(
        tmp_path, model, datetime(1982, 5, 20))
    assert root_path.name.startswith("gpuwmrst_d01_")
    assert [gid for gid, _header in captured] == [2, 1]
    assert len({header["checkpoint_set_id"]
                for _gid, header in captured}) == 1
    for gid, header in captured:
        assert header["grid_id"] == gid
        assert header["experiment_fingerprint"] == "fixture-fingerprint"
        assert header["phase"] == PERIOD_BEGIN
        assert header["nest_tables"] == "REBUILT"
        assert (header["elapsed_ticks"], header["tick_den"]) == (0, 1)

    model._runtime_status.pending_d2h = 1
    with pytest.raises(restart.RestartMismatchError, match="pending work"):
        restart.write_tree_restart(
            tmp_path, model, datetime(1982, 5, 20, 0, 1))
    model._runtime_status.pending_d2h = 0
    model._runtime_status.schedule_cursor = "EXECUTING"
    with pytest.raises(restart.RestartMismatchError,
                       match="explicit PERIOD_BEGIN"):
        restart.write_tree_restart(
            tmp_path, model, datetime(1982, 5, 20, 0, 2))


def test_tree_restart_preserves_not_started_child_then_activates_at_seam(
        monkeypatch, tmp_path):
    exp, source = _delayed_model()
    checkpoint_ticks = 60
    for node in source.walk_parent_first():
        node.clock.ticks = checkpoint_ticks
    source.root.clock.step_count = (
        checkpoint_ticks // source.root.clock.spec.step_ticks)
    captured = {}

    def fake_write(path, state, cfg, **kwargs):
        path.write_bytes(b"fixture")
        captured[cfg.grid_id] = {
            **kwargs["tree_header"],
            "format_version": restart.RESTART_FORMAT_VERSION,
            "elapsed_seconds": state.elapsed_seconds,
        }
        return path

    monkeypatch.setattr(restart, "write_restart", fake_write)
    root_path = restart.write_tree_restart(
        tmp_path, source, exp.start_time + timedelta(seconds=60))

    assert captured[1]["domain_lifecycle"] == "STARTED"
    assert captured[2]["domain_lifecycle"] == "NOT_STARTED"
    assert captured[2]["nest_tables"] == "NOT_STARTED"
    assert captured[2]["domain_start_ticks"] == 120
    assert captured[2]["domain_start_time"] == (
        exp.start_time + timedelta(seconds=120)).isoformat()

    _fresh_exp, resumed = _delayed_model()
    monkeypatch.setattr(
        restart, "_validate_restart",
        lambda path, *_args: SimpleNamespace(
            header=captured[int(path.name.split("_d")[1][:2])]))
    monkeypatch.setattr(
        restart, "_apply_validated_restart", lambda *args, **kwargs: None)
    restart.restore_tree_restart(root_path, resumed)

    child = resumed.node(2)
    assert child._started is False
    assert child.clock.ticks == 60
    starts = []
    report = execute_schedule(
        resumed.schedule,
        clocks={node.cfg.grid_id: node.clock
                for node in resumed.walk_parent_first()},
        start_period=(
            checkpoint_ticks // resumed.schedule.period_ticks),
        started_grid_ids=(1,),
        committed_initial_history_grid_ids=(1,),
        on_domain_start=lambda gid, clock:
            starts.append((gid, clock.ticks)))
    assert starts == [(2, 120)]
    assert report.clocks[2].step_count > 0


def test_failed_same_time_tree_rewrite_preserves_last_complete_generation(
        monkeypatch, tmp_path):
    """A failed rewrite must not replace members of the committed set.

    The child is published before the root commit marker.  Reusing fixed
    per-instant member names therefore used to replace the valid child's
    archive before a root-write failure, leaving the older checkpoint
    unrecoverably mixed across two set IDs.
    """
    _exp, model = _model()
    fail_root = {"value": False}

    def fake_write(path, state, cfg, **kwargs):
        if fail_root["value"] and cfg.grid_id == 1:
            raise OSError("injected root publication failure")
        header = kwargs["tree_header"]
        path.write_bytes(
            f"{header['checkpoint_set_id']}:{cfg.grid_id}".encode("ascii"))
        return path

    monkeypatch.setattr(restart, "write_restart", fake_write)
    instant = datetime(1982, 5, 20)
    committed_root = restart.write_tree_restart(tmp_path, model, instant)
    committed = restart._tree_restart_paths(committed_root, {1, 2})
    committed_bytes = {gid: path.read_bytes()
                       for gid, path in committed.items()}

    fail_root["value"] = True
    with pytest.raises(OSError, match="root publication failure"):
        restart.write_tree_restart(tmp_path, model, instant)

    assert committed_root.exists()
    assert restart._tree_restart_paths(committed_root, {1, 2}) == committed
    assert {gid: path.read_bytes() for gid, path in committed.items()} == \
        committed_bytes
    # Failed, uncommitted generation members are best-effort cleaned up.
    assert set(tmp_path.iterdir()) == set(committed.values())


def _tree_headers(model, *, phase=PERIOD_BEGIN, child_ticks=0):
    headers = {}
    for node in model.walk_parent_first():
        ticks = child_ticks if node.cfg.grid_id == 2 else 0
        headers[node.cfg.grid_id] = {
            "format_version": restart.RESTART_FORMAT_VERSION,
            "experiment_fingerprint": model.experiment_fingerprint,
            "checkpoint_set_id": "fixture-set",
            "grid_id": node.cfg.grid_id,
            "parent_id": node.cfg.parent_id,
            "domain_ids": [1, 2],
            "elapsed_ticks": ticks,
            "tick_den": 1,
            "elapsed_seconds": float(ticks),
            "phase": phase,
            "nest_tables": "REBUILT",
            "dtbc_fp32_bits": int(np.float32(0).view(np.uint32)),
        }
    return headers


def _tree_files(tmp_path):
    paths = {}
    for gid in (1, 2):
        path = tmp_path / f"gpuwmrst_d{gid:02d}_1982-05-20_00_00_00.npz"
        path.write_bytes(b"fixture")
        paths[gid] = path
    return paths


def test_tree_restore_refuses_partial_phase_and_mismatched_ticks(
        monkeypatch, tmp_path):
    _exp, model = _model()
    paths = _tree_files(tmp_path)
    headers = _tree_headers(model)
    monkeypatch.setattr(
        restart, "_validate_restart",
        lambda path, *_args: SimpleNamespace(
            header=headers[int(path.name.split("_d")[1][:2])]))
    monkeypatch.setattr(restart, "_apply_validated_restart",
                        lambda *args, **kwargs: None)

    paths[2].unlink()
    with pytest.raises(restart.RestartMismatchError, match="partial"):
        restart.restore_tree_restart(paths[1], model)
    paths[2].write_bytes(b"fixture")

    headers[2]["phase"] = "EXECUTING"
    with pytest.raises(restart.RestartMismatchError, match="phase"):
        restart.restore_tree_restart(paths[1], model)
    headers[2]["phase"] = PERIOD_BEGIN
    headers[2]["elapsed_ticks"] = headers[2]["elapsed_seconds"] = 6
    with pytest.raises(restart.RestartMismatchError,
                       match="elapsed ticks mismatch"):
        restart.restore_tree_restart(paths[1], model)


def test_tree_restore_invalidates_children_and_forbids_eager_force(
        monkeypatch, tmp_path):
    _exp, model = _model()
    paths = _tree_files(tmp_path)
    headers = _tree_headers(model)
    monkeypatch.setattr(
        restart, "_validate_restart",
        lambda path, *_args: SimpleNamespace(
            header=headers[int(path.name.split("_d")[1][:2])]))
    monkeypatch.setattr(restart, "_apply_validated_restart",
                        lambda *args, **kwargs: None)

    info = restart.restore_tree_restart(paths[1], model)
    child = model.node(2)
    assert info.phase == PERIOD_BEGIN
    assert child.coupler.valid is False
    from woof.core.nest import NestCoupler
    # Bypass only geometry registration (the ratio-1 identity scaffold fills
    # the parent and deliberately lacks a +-2 interpolation halo).  Calling
    # the real method still executes its identity and restored-clock guard.
    production_coupler = object.__new__(NestCoupler)
    production_coupler.child_node = child
    with pytest.raises(RuntimeError, match="parent must lead"):
        production_coupler.force(child)  # real parent(t) guard, lines 181-186
