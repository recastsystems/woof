"""Scoped ensemble handoffs inside the unchanged ordinary forecast runners.

Capture is synchronous: streamed store views may be reused by the next sweep.
No member clock, physics selector or history/reset schedule is owned here.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemberOutputCapture:
    callback: object
    member_id: int
    keep_member_files: bool = False
    counter_calendars: dict = field(default_factory=dict, compare=False, repr=False)
    initialize_callback: object = field(default=None, compare=False, repr=False)
    resuming: bool = field(default=False, kw_only=True)
    restart_pending: bool = field(default=False, kw_only=True)
    provenance: dict | None = field(default=None, compare=False)

    def submit_hydro(self, *, valid_time, reach_ids, discharge, stream=None,
                     channel_depth_m=None):
        """Borrow this member's resident reach output at a root deadline.

        The history collector owns the call; ordinary routing steps never
        enter this method. Preserve native precision and producer ordering.
        """
        owner = getattr(self.callback, "__self__", self.callback)
        submit = getattr(owner, "submit_hydro", None)
        if not callable(submit):
            raise RuntimeError("member capture has no hydro product consumer")
        return submit(member_id=self.member_id, valid_time=valid_time,
            reach_ids=reach_ids, discharge=discharge, stream=stream,
            channel_depth_m=channel_depth_m)

    def work_bytes(self, metadata):
        owner = getattr(self.callback, "__self__", self.callback)
        estimate = getattr(owner, "capturework_bytes", None)
        return None if estimate is None else int(estimate(metadata))

    def committed_step(self, *, node, model):
        owner = getattr(self.callback, "__self__", self.callback)
        observe = getattr(owner, "committed_step", None)
        if callable(observe):
            observe(node=node, model=model)

    def history_committed(self, proof, *, grid_id, episode=0, valid_time):
        """Register a durable raw identity before a consumer can retire it.

        The owner decides whether the history is its own to ledger: this
        capture's flag only says the writer must write it (simulated radar
        needs histories the request does not retain).
        """
        owner = getattr(self.callback, "__self__", self.callback)
        register = getattr(owner, "history_committed", None)
        if self.keep_member_files and callable(register):
            return register(proof, member_id=self.member_id, grid_id=grid_id,
                episode=episode, valid_time=valid_time)

    def history_committer(self, *, grid_id, episode=0):
        return lambda proof, valid_time: self.history_committed(proof,
            grid_id=grid_id, episode=episode, valid_time=valid_time)

    def submit(self, *, state, streamed, metadata, refl_field, valid_time,
               grid_id, episode=0, clock=None):
        result = self.callback(state=state, streamed=streamed, metadata=metadata,
                             refl_field=refl_field, valid_time=valid_time,
                             grid_id=int(grid_id), episode=int(episode),
                             member_id=self.member_id)
        if clock is not None:
            calendar = self.counter_calendars.get((int(grid_id), id(clock)))
            owner = getattr(self.callback, "__self__", self.callback)
            has_counter = getattr(owner, "has_rain_counter", None)
            if calendar is not None and callable(has_counter) and has_counter(
                    valid_time=valid_time, grid_id=grid_id, episode=episode,
                    member_id=self.member_id, latitude=metadata["XLAT"],
                    longitude=metadata["XLONG"]):
                calendar.mark_captured(clock.ticks, member_id=self.member_id, episode=episode)
        return result

    def observe_model_counters(self, model, *, start_time):
        """Observe committed clocks without adding output or model alarms."""
        owner = getattr(self.callback, "__self__", self.callback)
        if not callable(getattr(owner, "capture_rain_counters", None)):
            return
        from woof.ensemble.output_counters import CounterDeadlineCalendar, capture_due_counter
        from woof.runtime import _metadata_frame
        for node in model.walk_parent_first():
            if not getattr(node, "_started", True):
                continue
            manager = getattr(model, "_io_manager", None)
            episode = int(getattr(manager, "_episode_by_grid_id", {}).get(node.cfg.grid_id, 0))
            key = (int(node.cfg.grid_id), id(node.clock))
            calendar = self.counter_calendars.get(key)
            if calendar is None:
                calendar = CounterDeadlineCalendar.from_node(node)
                self.counter_calendars[key] = calendar
            # Metadata construction and counter transfers are confined to
            # exact deadlines. Ordinary steps only update missed-endpoint
            # accounting, and cannot be shortened by this observer.
            if node.clock.ticks not in calendar.deadlines:
                calendar.observe(node.clock.ticks, member_id=self.member_id, episode=episode)
                continue
            case = model._prepared_by_grid_id[node.cfg.grid_id]
            metadata = _metadata_frame(node.grid, case.static_fields)
            file_grid = getattr(case, "wrf_file_grid", None)
            if file_grid:
                metadata = {**metadata, **file_grid}
            capture_due_counter(node, owner, self.member_id, calendar=calendar,
                start_time=start_time, metadata=metadata, episode=episode,
                streamed=getattr(node.state, "_streamed_domain", None))


_CAPTURE = ContextVar("gpuwm_ensemble_member_output", default=None)
_SESSION = ContextVar("gpuwm_ensemble_session", default=None)
_DISPATCHING = ContextVar("gpuwm_ensemble_dispatching", default=False)


def current_capture():
    return _CAPTURE.get()


def current_session():
    return None if _DISPATCHING.get() else _SESSION.get()


def observe_current_counters(model, *, start_time):
    capture = current_capture()
    if capture is not None:
        capture.observe_model_counters(model, start_time=start_time)


def observe_initial_diagnostics(model):
    """Hand the restored, perturbed state to diagnostics once before stepping."""
    capture = current_capture()
    if capture is not None:
        owner = getattr(capture.callback, "__self__", capture.callback)
        observe = getattr(owner, "observe_initial_state", None)
        if callable(observe):
            observe(model=model)


def bind_current_member_model(model):
    """Attach member pattern owners before original restart validation."""
    capture = current_capture()
    if capture is not None and capture.initialize_callback is not None:
        capture.initialize_callback(model=model)


def bind_current_member_state(*, prepared_case, state, cfg, grid, clock=None):
    """Bind an actual single-domain state before restart or stepping."""
    capture = current_capture()
    if capture is not None and capture.initialize_callback is not None:
        capture.initialize_callback(prepared_case=prepared_case, state=state,
                                    cfg=cfg, grid=grid, clock=clock)


def current_member_reconstruction_owner(state):
    """Retain spectral continuation across an original placement rebuild."""
    capture = current_capture()
    if capture is None or capture.initialize_callback is None:
        return None
    owner = getattr(state, "_ensemble_stochastic", None)
    if owner is None or not owner.enabled:
        return None
    if owner.member_id != capture.member_id:
        raise ValueError("reconstructed state carries another ensemble member's spectral owner")
    if owner.hook.pending_step is not None:
        raise ValueError("stochastic state cannot be reconstructed during an unfinished timestep")
    return owner


def bind_reconstructed_member_state(*, state, cfg, grid, clock, prepared_case=None,
                                    previous_owner=None):
    """Births bind their seed; moves retain spectra and the original clock."""
    capture = current_capture()
    if capture is None or capture.initialize_callback is None:
        return
    if previous_owner is not None:
        existing = getattr(state, "_ensemble_stochastic", None)
        if existing is not None and existing is not previous_owner:
            raise ValueError("reconstruction created two spectral owners for one member domain")
        state._ensemble_stochastic = previous_owner
    bind_current_member_state(prepared_case=prepared_case, state=state, cfg=cfg,
                              grid=grid, clock=clock)


def bind_reconstructed_member_node(node, *, previous_owner=None, prepared_case=None):
    """The inactive path asks nothing of an ordinary node's attributes."""
    capture = current_capture()
    if capture is None or capture.initialize_callback is None:
        return
    bind_reconstructed_member_state(state=node.state, cfg=node.cfg.run,
        grid=node.grid, clock=node.clock, prepared_case=prepared_case,
        previous_owner=previous_owner)


def initialized_bootstrap_handoff(callback, *, inputs, model, node,
                                  output_directory, observer, step_log):
    """An explicit native executor may consume an ordinary initialized root.

    None means the callback declined before mutation and the original runner
    continues. A failed native execution propagates; it is never rerun through
    another clock or physics path. The initialization log closes on every
    early terminal return.
    """
    if callback is None:
        return None
    try:
        report = callback(inputs=inputs, model=model, node=node,
                          output_directory=output_directory, observer=observer,
                          step_observer=(step_log.step_observer if step_log.enabled else None))
        if report is not None and report.get("status") != "PASS":
            raise RuntimeError("native ensemble returned a failing forecast receipt")
    except BaseException as error:
        step_log.close(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    if report is not None:
        _name_native_packs(step_log, report)
        step_log.close(status="SUCCESS")
    return report


def _name_native_packs(step_log, report):
    """Say in the run's step log what the native packs advanced.

    The log belongs to the first member's ordinary runner, which hands its
    initialized root to the native packs and takes no step of its own. Its
    step lines follow the pack that holds the first member, so its closing
    count is that clock's steps; this line adds the roster the packs carried.
    Breakage this prevents: a native run that advanced every member closed
    its log with "SUCCESS COMPLETE SIMULATION, 0 steps".
    """
    packs = [row["result"] for row in report.get("member_results", ())
             if isinstance(row, dict) and isinstance(row.get("result"), dict)
             and isinstance(row["result"].get("executor"), dict)]
    if not packs:
        return
    steps = max(int(pack["executor"].get("steps", 0)) for pack in packs)
    member_steps = sum(int(pack["executor"].get("member_steps", 0)) for pack in packs)
    members = len(report.get("members_completed", ()))
    step_log.phase(f"native ensemble forecast, {members} members, {steps} steps each",
                   report.get("wall_seconds"), members=members, steps_per_member=steps,
                   member_steps=member_steps, packs=len(packs))


@contextmanager
def member_output_scope(capture):
    token = _CAPTURE.set(capture)
    dispatch_token = _DISPATCHING.set(True)
    try:
        yield capture
    finally:
        _DISPATCHING.reset(dispatch_token)
        _CAPTURE.reset(token)


@contextmanager
def ensemble_scope(session):
    token = _SESSION.set(session)
    try:
        yield session
    finally:
        _SESSION.reset(token)
