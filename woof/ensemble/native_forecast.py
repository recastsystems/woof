"""Forecast a qualified initialized root through the ordinary output clock.

The caller supplies one ordinary initialized node. Native factories own the
member state and physics banks. Output adapters borrow views without changing
values, and the roster collector owns diagnostic arithmetic and Rust output.
"""
from __future__ import annotations

from dataclasses import asdict, fields as dataclass_fields, replace
from datetime import timedelta
import math
import time
from types import SimpleNamespace
import numpy as np

from woof.ensemble.prepared_batch import prepare_native_member_batch, native_prepared_eligibility

NATIVE_FORECAST_CONTRACT = "gpuwm-ensemble-initialized-native-forecast-v1"


class NativeLaunchRefused(ValueError):
    """Launch preparation refused a native pack before any member advanced.

    The native path rewrites CUDA text and the Python source of the stock
    radiation call under counted source audits, and those audits run while
    the pack's launches are prepared. A refusal there has moved no member
    state, clock, counter or product, so the caller may decline to the
    ordinary runner with ``reason`` as its named fallback reason.
    """
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = str(reason)


def member_column_view(array, *, member, members, ny, nx):
    """Borrow a member stripe or a shared surface without field arithmetic."""
    if array is None:
        return None
    shape = getattr(array, "shape", ())
    if len(shape) in (2, 3) and shape[-1] in (nx, nx + 1) and members > 1:
        rows = shape[-2]
        if rows in (members * ny, members * (ny + 1)):
            height = rows // members
            return (array[:, member * height:(member + 1) * height] if len(shape) == 3
                    else array[member * height:(member + 1) * height])
    return array


def borrow_physics_scratch(view, driver, *, member, members, ny, nx):
    """Borrow persistent native diagnostics from the physics owner.

    Correct mode's Thompson books CLOUDNC/CLOUDNCV (finding WB1,
    woof/core/thompson_cloud_booking.py) into the scratch of the state its
    physics driver runs on: the packed physics bank of a native ensemble
    (PackedThompsonState), never the batch's own scratch. History, validation
    and checkpoint frames read ``state._scratch``, so a member view built from
    the batch alone wrote CLOUDNC as zeros (N > 1, where the registry priced
    an unwritten batch slot of that name) or not at all (N = 1), while the
    same member run ordinarily wrote its booked total. ISHMAEL likewise
    owns its precipitation accumulators and three-habit diagnostic bank
    in the packed physics workspace; those views follow the same route.
    """
    from woof.core.streaming import domain_scratch
    from woof.ensemble.batch_physics import THOMPSON_CLOUD_BOOKING_SLOTS
    owner = getattr(driver, "state", None)
    if owner is None:
        return view
    # Through the sanctioned reader rather than binding ``existing_scratch``
    # here: this module names the up_heli_max slot, and
    # tests/test_uh_lifecycle.py forbids an unsanctioned module that does so
    # from acquiring the scratch API indirectly.  domain_scratch answers
    # None for a state without the slot, as the direct lookup did.
    booked = {}
    slots = THOMPSON_CLOUD_BOOKING_SLOTS
    if getattr(driver, "mp_physics", None) == 55:
        from woof.physics_registry import consumer_row_for_selector
        native = consumer_row_for_selector("microphysics", "native_column", 55)
        slots += tuple(native["surfaces"].values()) + tuple(native["diagnostics"].values())

    if getattr(getattr(driver, "cumulus_callable", None), "scheme", None) == 99:
        # CU99's held rates and expiry clocks are written in the physics
        # bank. Observers must borrow those words, not the batch's cold slots.
        slots = (*slots, *(name for name in owner._scratch if name.startswith("cu_")))
    for slot in slots:
        value = domain_scratch(owner, slot)
        if value is not None:
            booked[slot] = member_column_view(value, member=member, members=members, ny=ny, nx=nx)
    if booked:
        view._scratch = {**view._scratch, **booked}
    return view


def member_output_view(owners, member, *, output_fields=None):
    """Expose the original physics writer's fields as borrowed member views."""
    from woof.ensemble.batch_dycore import member_domain_view
    batch, driver, cfg = owners.batch, owners.physics.driver, owners.batch.cfg
    if output_fields is None:
        output_fields = driver.output_fields()
    borrow = lambda value: member_column_view(value, member=member, members=batch.members,
                                               ny=cfg.ny, nx=cfg.nx)
    fields = {name: borrow(value) for name, value in output_fields.items()}
    view = member_domain_view(batch, member)
    borrow_physics_scratch(view, driver, member=member, members=batch.members, ny=cfg.ny, nx=cfg.nx)
    microphysics = getattr(driver, "microphysics", None)
    if microphysics is not None:
        microphysics = replace(microphysics, **{
            field.name: borrow(getattr(microphysics, field.name))
            for field in dataclass_fields(microphysics) if field.init})
    view.physics = SimpleNamespace(fields={name: borrow(value) for name, value in driver.fields.items()},
        surface_enabled=driver.surface_enabled, output_fields=lambda: fields,
        scheme_dispatch=getattr(driver, "scheme_dispatch", None),
        noah_params=getattr(driver, "noah_params", None), microphysics=microphysics)
    return view


def _clock_snapshot(clock):
    return {name: getattr(clock, name) for name in
            ("ticks", "step_ticks", "tick_den", "run_ticks", "step_count", "dt_fp32", "dtbc_fp32")}


def _receipt_value(value):
    """Keep metadata JSON-ready without converting any weather-field array."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {name: _receipt_value(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_receipt_value(item) for item in value]
    return value


def _member_failure(member, text):
    """A step failure that carries its member for the wave and the manifest."""
    error = RuntimeError(text)
    error.ensemble_member_ids = (int(member),)
    return error


def _require_step_health(reports, *, member_ids, step):
    """The ordinary per-step NaN and live-thickness predicate."""
    if len(reports) != len(member_ids):
        raise RuntimeError("native stability reduction returned an incomplete member roster")
    for member, report in zip(member_ids, reports):
        if report["nan"]:
            raise _member_failure(member, f"member {member} integration produced a non-finite state at model step {step}")
        cfl = report["cfl"]
        if cfl is not None and not math.isfinite(float(cfl)):
            raise _member_failure(member, f"member {member} integration produced a non-finite vertical Courant number at model step {step}")


def run_initialized_native_ensemble(inputs, node, *, members, member_ids=None, member_seeds=None,
                                    collector, available_bytes, initializer=None, progress_callback=None,
                                    validation_callback=None, output_metadata=None,
                                    bootstrap_pool_live_increment_bytes=None, array_module=None,
                                    step_observer=None, launch_prepared=None):
    """Advance all members, or return None for the ordinary fallback route.

    ``validation_callback`` is an optional word-capture seam receiving the
    live banks at initialized, history and final positions. The executor does
    no default full-state download. The caller must release ordinary model
    construction handles after packing; they are excluded from retained
    native owners but can still affect the observed pool while held outside.

    A source-audit or adapter refusal while the pack and its launches are
    prepared raises ``NativeLaunchRefused``: nothing has advanced, so the
    caller can decline to the ordinary runner. ``launch_prepared`` is called
    once preparation is complete and before the first health gate, frame or
    step; it may raise to stop this pack there. ``step_observer`` receives
    each completed pack step in the ordinary step log's own signature.
    """
    if not native_prepared_eligibility(inputs, node, members=members,
        keep_member_files=bool(getattr(collector, "keep_member_files", False))).eligible:
        return None
    from woof.certify.kernel_manifest import kernel_manifest
    if array_module is None:
        import cupy as array_module
    started = time.perf_counter()
    modules_before = kernel_manifest()
    pool_before = array_module.get_default_memory_pool().used_bytes()
    from woof.core.refl import consume_refl_10cm, refl_10cm_stash_is_due
    from woof.core.uh_diag import reset_up_heli_max
    from woof.ensemble.batch_health import PreparedBatchStability, PreparedBatchStateHealth
    # LAUNCH PREPARATION. The pack is built and every launch audited inside
    # this stretch; the ordinary initialized node is only read. The native
    # adapters refuse with ValueError (their two unsupported classes derive
    # from it), so that one class is the audit refusal and nothing else is
    # reinterpreted: a memory, device or runtime failure still fails the run.
    try:
        owners = prepare_native_member_batch(inputs, node, members=members, member_ids=member_ids,
            member_seeds=member_seeds, member_initializer=initializer, available_bytes=available_bytes,
            bootstrap_pool_live_increment_bytes=bootstrap_pool_live_increment_bytes,
            array_module=array_module, keep_member_files=bool(getattr(collector, "keep_member_files", False)))
        if owners is None:
            return None
        cfg, batch, physics, clock = owners.batch.cfg, owners.batch, owners.physics, owners.clock
        if clock is not node.clock:
            raise RuntimeError("native forecast changed the ordinary initialized node clock")
        if output_metadata is None:
            from woof.runtime import _metadata_frame
            bundle = next(bundle for bundle in inputs.domains if int(bundle.grid_id) == int(node.cfg.grid_id))
            # A WRF-file door domain writes the file's own grid words, as the
            # ordinary writer does (woof.io.wrfout, wrf_file_grid), with the
            # run's configuration so the rows WRF initializes rather than reads
            # (AREA2D, DX2D, the seed arrays, NEST_POS ...) are written too, as
            # prepared_domain_tree_forecast passes domain.run (here the batch RunConfig).
            from woof.io.history_layout import wrf_file_grid_history_fields
            output_metadata = {**_metadata_frame(node.grid, bundle.static_fields),
                               **wrf_file_grid_history_fields(bundle, cfg)}
        # Each allocation receives a current remaining budget. An integer caller
        # budget is decremented by all native plans already constructed above.
        remaining = None if callable(available_bytes) else int(available_bytes)
        if remaining is not None:
            remaining -= max(0, array_module.get_default_memory_pool().used_bytes() - pool_before)
        def budget():
            return int(available_bytes()) if callable(available_bytes) else remaining
        stability = PreparedBatchStability(batch, cfg, boundary_width=cfg.spec_bdy_width,
                                           available_bytes=budget(), array_module=array_module)
        if remaining is not None:
            remaining -= stability.plan.required_bytes(members)
        validator = PreparedBatchStateHealth(batch, physics_driver=physics.driver, tables=owners.tables,
                                              available_bytes=budget(), array_module=array_module,
                                              member_ids=owners.member_ids)
    except ValueError as refusal:
        raise NativeLaunchRefused(
            "native launch preparation refused before any member advanced: "
            f"{type(refusal).__name__}: {refusal}") from refusal
    if launch_prepared is not None:
        launch_prepared()
    initial_health = validator.require_healthy(phase="initialized-or-restored")
    initialized = time.perf_counter()
    history, validation_records = [], []
    counter_calendar = None
    if callable(getattr(collector, "capture_rain_counters", None)):
        from woof.ensemble.output_counters import CounterDeadlineCalendar, capture_due_counter
        counter_calendar = CounterDeadlineCalendar.from_node(node)
    def capture_counters():
        if counter_calendar is None:
            return
        # The original driver owns these two accumulators. RAINSH is the
        # original always-zero output declaration, so no zero field or full
        # output_fields dictionary is constructed for this observer.
        actual = {}
        if cfg.mp_physics:
            actual["RAINNC"] = physics.driver.microphysics.rainnc
        if getattr(physics.driver, "rainc", None) is not None:
            actual["RAINC"] = physics.driver.rainc
        counter_node = SimpleNamespace(cfg=node.cfg, clock=clock, state=None, _started=True)
        for member, member_id in enumerate(owners.member_ids):
            fields = {name: member_column_view(value, member=member, members=members, ny=cfg.ny, nx=cfg.nx)
                      for name, value in actual.items()}
            capture_due_counter(counter_node, collector, member_id, calendar=counter_calendar,
                start_time=inputs.experiment.start_time, metadata=output_metadata, counter_fields=fields)
    last_history_ticks = None
    steps, lbc_resets = 0, 0
    def validation(phase, views=None):
        if validation_callback is not None:
            record = validation_callback(phase=phase, owners=owners, member_ids=owners.member_ids,
                member_seeds=owners.member_seeds, clock=clock, member_views=views)
            validation_records.append({"phase": phase, "ticks": int(clock.ticks), "record": record})
    validation("initialized")
    capture_counters()
    def frame_if_due():
        nonlocal last_history_ticks
        if not clock.history_due() or last_history_ticks == clock.ticks:
            return
        validator.require_healthy(phase=f"pre-history.d{int(node.cfg.grid_id):02d}")
        reports = stability()
        reflected = (consume_refl_10cm(physics.state)
            if refl_10cm_stash_is_due(clock.ticks, domain_start_ticks=clock.spec.start_ticks) else None)
        from woof.core.physics_inventory import REFL_10CM_MICROPHYSICS
        if (reflected is None and clock.ticks == 0
                and getattr(cfg, "mp_physics", 0) in REFL_10CM_MICROPHYSICS):
            from woof.core.refl import analysis_refl_10cm
            with physics.state.analysis_diagnostic_view():
                reflected = analysis_refl_10cm(physics.state)
        fields = physics.driver.output_fields()
        views = tuple(member_output_view(owners, member, output_fields=fields) for member in range(members))
        validation("history", views)
        valid = inputs.experiment.start_time + timedelta(seconds=clock.elapsed_seconds)
        for member, view in enumerate(views):
            refl = member_column_view(reflected, member=member, members=members, ny=cfg.ny, nx=cfg.nx)
            collector.submit(state=view, streamed=None, metadata=output_metadata, refl_field=refl,
                valid_time=valid, grid_id=int(node.cfg.grid_id), episode=0, member_id=owners.member_ids[member])
            # The ordinary history path resets after the synchronous consumer
            # has borrowed the completed frame. Absent UH remains a no-op.
            reset_up_heli_max(view)
        last_history_ticks = int(clock.ticks)
        history.extend({"member_id": member, "grid_id": int(node.cfg.grid_id),
            "ticks": int(clock.ticks), "elapsed_seconds": float(clock.elapsed_seconds), **report}
            for member, report in zip(owners.member_ids, reports))
    frame_if_due()
    while not clock.at_stop_time:
        frame_if_due()
        if clock.lbc_reset_due():
            clock.mark_force()
            lbc_resets += 1
        clock.prepare_step()
        batch.elapsed_seconds = float(clock.elapsed_seconds_fp32)
        batch.clock["dtbc_fp32"] = clock.dtbc_fp32
        before = _clock_snapshot(clock)
        step_started = time.perf_counter()
        owners.advance(refl_10cm_due=clock.history_rings_within_step())
        reports = stability()
        _require_step_health(reports, member_ids=owners.member_ids, step=clock.step_count + 1)
        clock.advance()
        if (batch.clock["ticks"] != clock.ticks or batch.clock["step_count"] != clock.step_count
                or clock.ticks != before["ticks"] + before["step_ticks"]):
            raise RuntimeError("native private bookkeeping and ordinary clock advanced different steps")
        batch.elapsed_seconds = clock.elapsed_seconds
        batch.clock.update(_clock_snapshot(clock))
        capture_counters()
        steps += 1
        if step_observer is not None:
            # One pack step is one step of every member on the shared
            # clock. Telemetry never fails a run, as in the ordinary executor.
            try:
                step_observer(grid_id=int(node.cfg.grid_id), step_count=int(clock.step_count),
                    model_seconds=clock.ticks / clock.tick_den,
                    step_wall_seconds=time.perf_counter() - step_started,
                    dt=before["step_ticks"] / before["tick_den"])
            except Exception:  # noqa: BLE001 - telemetry never fails a run
                pass
        if clock.at_stop_time or clock.history_due() or clock.step_count % 4 == 0:
            validator.require_healthy(phase=f"post-d01-sync.d{int(node.cfg.grid_id):02d}")
        frame_if_due()
        if progress_callback is not None:
            progress_callback(status="RUNNING", model_elapsed_seconds=float(clock.elapsed_seconds),
                outer_step=int(clock.step_count), requested_run_seconds=float(inputs.experiment.run_seconds),
                forecast_wall_seconds=time.perf_counter() - initialized, members=members,
                member_ids=owners.member_ids, backend="native_member_batched")
    final_health = validator.require_healthy(phase=f"final.d{int(node.cfg.grid_id):02d}")
    final_stability = stability()
    validation("final")
    array_module.cuda.get_current_stream().synchronize()
    modules = kernel_manifest()
    final_pool = array_module.get_default_memory_pool()
    complete = (set(owners.member_ids) == set(getattr(collector, "member_order",
        tuple(range(getattr(collector, "members", members))))))
    products = collector.require_complete() if complete else collector.receipt()
    elapsed = time.perf_counter() - started
    return _receipt_value({"schema": NATIVE_FORECAST_CONTRACT, "status": "PASS", "source": inputs.source,
        "readiness": "IMPLEMENTED_UNVERIFIED", "execution_plan": inputs.execution_plan,
        "backend": "native_member_batched", "completed_seconds": float(clock.elapsed_seconds),
        "wall_seconds": elapsed, "timing_seconds": {"initialization": initialized - started,
            "forecast_execution": time.perf_counter() - initialized, "total": elapsed},
        "executor": {"steps": steps, "member_steps": steps * members, "forces": 0,
            "feedback_calls": 0, "lbc_resets": lbc_resets, "clock": _clock_snapshot(clock)},
        "health": {"initial": [asdict(report) for report in initial_health],
            "final": [asdict(report) for report in final_health], "final_stability": list(final_stability),
            "history": history, "stability_backend": stability.receipt, "full_state_backend": validator.receipt},
        "native_allocations": owners.receipt(), "products": products, "validation": validation_records,
        "precipitation_counter_calendar": (None if counter_calendar is None else
            [counter_calendar.receipt(member_id=member, through_ticks=clock.ticks) for member in owners.member_ids]),
        "member_history_count": 0, "registered_compilation_modules": {"before": len(modules_before),
            "after": len(modules), "added": len(set(modules) - set(modules_before))},
        "pool_live_bytes": final_pool.used_bytes(), "pool_reserved_bytes": final_pool.total_bytes(),
        "timing_scope": "initialized ordinary root through native forecast and finished eligible roster products"})


__all__ = ["NATIVE_FORECAST_CONTRACT", "NativeLaunchRefused", "member_column_view", "member_output_view",
           "run_initialized_native_ensemble"]
