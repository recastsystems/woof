"""Automatic execution from the ordinary initialized-root handoff.

One card retains an immutable, ordinary initialized source until its last
wave ends. Only its audited state/driver containers are borrowed. Mutable
member arrays are built afresh for each wave and the original DomainClock
constructor supplies every wave's clock. Different source trajectories keep
their original initialized member path until their native bank is qualified.
"""
from __future__ import annotations

from copy import copy
from contextlib import nullcontext
from dataclasses import dataclass, asdict, is_dataclass, replace
from fractions import Fraction
from operator import index
import sys
import threading
import time
import gc
from pathlib import Path

import numpy as np

from woof.ensemble.admission import AllocatorMargin, MemoryComponent
from woof.ensemble.packing import CardBudget, pack_members
from woof.ensemble.prepared_batch import native_prepared_eligibility, native_memory_model_from_node
from woof.ensemble.request import EnsembleRequest
from woof.ensemble.seeds import member_seed
from woof.ensemble.batch_state import BatchStateUnsupported

CONTRACT = "gpuwm-ensemble-prepared-execution.v1"


def member_device_ids_for_request(inputs, request, *, visible_count, current_device=0):
    """Use visible member cards, preserving an existing spatial device graph."""
    request = EnsembleRequest.from_mapping(request)
    if isinstance(visible_count, (bool, np.bool_)) or index(visible_count) < 1:
        raise ValueError("member execution needs a positive visible card count")
    visible_count = index(visible_count)
    spatial = getattr(inputs.experiment, "devices", None)
    if int(getattr(spatial, "count", 1)) > 1:
        from woof.core.devices import validate_device_count
        validate_device_count(spatial, visible_count)
        from woof.free_threading import refuse_multi_card_under_gil
        refuse_multi_card_under_gil(spatial.count, "woof ensemble")
        # One member owns the entire original graph. Another member may not
        # start on a card that is already serving its sibling spatial rank.
        graph = spatial.device_ids()
        if request.member_device_ids not in (None, (graph[0],)):
            raise ValueError("member cards cannot overlap the selected spatial device graph")
        return (graph[0],)
    ids = request.member_device_ids or tuple(range(visible_count))
    if any(device >= visible_count for device in ids):
        raise ValueError("member device selection names a non-visible physical card")
    return ids


def ordinary_member_execution_inputs(inputs, *, batch, external_components,
                                     allocator_margin, force_streaming=None):
    """Bind the ordinary tile door to a card's budget and output reservations.

    Physics, clock, source/head authority and prepared arrays are untouched.
    This execution-only overlay has the same checkpoint identity. Existing
    on/pinned tile choices stay selected; an off road gains auto admission
    when the resident member did not fit. The original tile planner decides
    its actual legal tile, host budget, radiation and scratch sizes.
    """
    stream = batch.execution_mode == "ordinary_streamed_member" if force_streaming is None else bool(force_streaming)
    if not stream:
        return inputs, {"changed": False, "device_id": batch.device_id,
                        "execution_mode": batch.execution_mode}
    external_components = tuple(external_components)
    if (not external_components or any(not isinstance(component, MemoryComponent) or not component.evidence
        for component in external_components) or not isinstance(allocator_margin, AllocatorMargin)
        or not allocator_margin.evidence):
        raise ValueError("streamed ensemble admission needs named output/health/stochastic reservations and allocator evidence")
    rows = tuple(component.inventory(1) for component in external_components)
    withheld = sum(row["required_bytes"] for row in rows) + allocator_margin.required_bytes(rows)
    budget = int(batch.available_bytes) - withheld
    if budget <= 0:
        raise MemoryError("card has no positive ordinary tile budget after its named ensemble reservations")
    if not is_dataclass(inputs) or not is_dataclass(inputs.experiment):
        raise TypeError("execution-only tile overlay needs typed prepared inputs and ExperimentConfig")
    from woof.core.streaming import StreamingOptions
    from woof.core.model import restart_identity_payload
    exp = inputs.experiment
    before = restart_identity_payload(exp)
    def options(value):
        if value is None:
            return None
        if not isinstance(value, StreamingOptions):
            raise TypeError("ordinary tile overlay needs the original StreamingOptions")
        admitted = min(budget, value.vram_budget_bytes) if value.vram_budget_bytes is not None else budget
        return replace(value, mode="auto" if value.mode == "off" else value.mode,
                       vram_budget_bytes=admitted)
    domains = tuple(replace(domain, tiles=options(domain.tiles)) for domain in exp.domains)
    updated = replace(exp, tiles=options(exp.tiles), domains=domains)
    if restart_identity_payload(updated) != before:
        raise RuntimeError("execution-only tile overlay changed the original restart identity")
    result = replace(inputs, experiment=updated)
    return result, {"changed": True, "device_id": batch.device_id,
        "execution_mode": "ordinary_streamed_member", "sampled_free_bytes": batch.available_bytes,
        "ensemble_withheld_bytes": withheld, "ordinary_tile_budget_bytes": budget,
        "external_components": rows, "allocator_margin_evidence": allocator_margin.evidence,
        "physics_selection": "unchanged", "clock_selection": "unchanged",
        "stream_head": "unchanged", "checkpoint_identity": "unchanged",
        "tile_selection": "existing auto/pinned planner with card-specific remaining budget"}


@dataclass(frozen=True)
class InitializedCardEvidence:
    """A live after-bootstrap card sample with explicit future reservations."""
    card: CardBudget
    runtime_reservation: MemoryComponent
    allocator_margin: AllocatorMargin
    bootstrap_live_bytes: int
    resident_threads: int = 0
    evidence: str = ""

    def __post_init__(self):
        if not isinstance(self.card, CardBudget):
            raise TypeError("initialized card evidence needs a physical card budget")
        if (not isinstance(self.runtime_reservation, MemoryComponent)
                or not self.runtime_reservation.evidence
                or not isinstance(self.allocator_margin, AllocatorMargin)
                or not self.allocator_margin.evidence or not self.evidence):
            raise ValueError("initialized native admission needs runtime, allocator and sampling evidence")
        for name in ("bootstrap_live_bytes", "resident_threads"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or index(value) < 0:
                raise ValueError("initialized evidence byte/thread values must be nonnegative integers")
            object.__setattr__(self, name, index(value))

    def receipt(self):
        return {"device_id": self.card.device_id, "sampled_free_bytes": self.card.available_bytes,
                "total_bytes": self.card.total_bytes, "free_sample_timing": "after_bootstrap",
                "bootstrap_live_bytes": self.bootstrap_live_bytes,
                "runtime_reservation": self.runtime_reservation.inventory(1),
                "allocator_margin_evidence": self.allocator_margin.evidence,
                "sampling_evidence": self.evidence}


def sample_initialized_card_evidence(inputs, *, array_module=None):
    """Read the current card; no import or device access occurs at module load."""
    if array_module is None:
        import cupy as array_module
    from woof.core.preflight import (
        FORECAST_POOL_HEADROOM, EXTERNAL_MARGIN_BYTES, local_memory_profile_from_device,
        non_pool_device_bytes, release_unreachable_device_memory,
    )
    device = int(array_module.cuda.runtime.getDevice())
    array_module.cuda.get_current_stream().synchronize()
    # The pack is sized from this reading, so garbage and cached blocks
    # left by earlier work in the process must not count as used: the
    # native decline gate's four-member pack was admitted as two packs
    # after other ensembles had run in the same process.
    release_unreachable_device_memory(array_module)
    free, total = array_module.cuda.runtime.memGetInfo()
    profile = local_memory_profile_from_device(array_module, device_id=device)
    props = array_module.cuda.runtime.getDeviceProperties(device)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    pool = array_module.get_default_memory_pool()
    # Future native modules and column local-memory backing have not all
    # launched at this seam. Reserve the original calibrated bound in full;
    # the receipt calls this an envelope, not a measured new allocation.
    runtime = MemoryComponent("native_runtime_envelope", "runtime",
        fixed_bytes=non_pool_device_bytes(inputs.experiment, profile=profile), basis="envelope",
        evidence=f"original non_pool_device_bytes on live profile {profile.name}; no unloaded-module exemption")
    margin_fraction = Fraction(str(FORECAST_POOL_HEADROOM)) - 1
    margin = AllocatorMargin(margin_fraction.numerator, margin_fraction.denominator,
        minimum_bytes=int(EXTERNAL_MARGIN_BYTES),
        evidence="original measured FORECAST_POOL_HEADROOM and EXTERNAL_MARGIN_BYTES")
    card = CardBudget(device, int(free), int(total), str(name), str(profile.name))
    return InitializedCardEvidence(card, runtime, margin, int(pool.used_bytes()),
        resident_threads=int(profile.resident_thread_capacity),
        evidence="cudaMemGetInfo after ordinary initialization and synchronization; warm bootstrap retained through all waves")


@dataclass(frozen=True)
class InitializedCardBootstrap:
    inputs: object
    node: object
    evidence: InitializedCardEvidence


def _source_selection(request, roster):
    if roster is None:
        return tuple(range(request.members)), tuple(member_seed(request.base_seed, member)
                                                    for member in range(request.members)), None
    from woof.ensemble.member_preparation import PreparedMemberRoster
    if not isinstance(roster, PreparedMemberRoster):
        raise TypeError("member-specific execution needs a verified PreparedMemberRoster")
    if len(roster.members) != request.members:
        raise ValueError("prepared source roster size differs from the requested ensemble")
    ids = tuple(member.member_id for member in roster.members)
    selected = roster.select(ids)
    return ids, tuple(member.seed for member in selected), roster.receipt()


def _member_specific(request, roster, input_provider):
    """Members take their own inputs: a prepared roster, an input provider or request sources."""
    return roster is not None or input_provider is not None or bool(request.sources)


def _fallback_reasons(inputs, node, request, *, roster, input_provider, stochastic_enabled,
                      require_member_sources=True):
    """Name every reason this root cannot pack the requested roster.

    Members with their own inputs (time-lagged cycles, other models, a
    prepared roster) pack when every member's own ordinary bootstrap is
    bound to the root as member sources and each is compatible with it. A
    root without them is refused by name: packing would advance one source
    under several member names and publish zero spread as an ensemble. The
    automatic executor checks with ``require_member_sources=False`` first,
    bootstraps and binds the members, then plans with the default.
    """
    reasons = list(native_prepared_eligibility(inputs, node, members=request.members,
        keep_member_files=request.keep_member_files).reasons)
    from woof.ensemble.surface_recipe import is_surface_recipe
    if request.perturbation is not None and not is_surface_recipe(request.perturbation):
        reasons.append("member perturbation preparation remains on its original bound initializer path")
    binding = getattr(node.state, "_ensemble_stochastic", None)
    if stochastic_enabled or bool(getattr(binding, "enabled", False)):
        reasons.append("active stochastic physics retains the original member timestep binding; the native packed coupling is not qualified")
    if require_member_sources and _member_specific(request, roster, input_provider):
        from woof.ensemble.prepared_batch import member_sources_of
        sources = member_sources_of(node)
        ids = _source_selection(request, roster)[0]
        if sources is None or not sources.covers(ids):
            reasons.append("member-specific inputs are not all bootstrapped as member sources on this root; "
                           "packing would advance one source under several member names")
        elif not reasons:
            reasons.extend(sources.compatibility_reasons(node))
    return tuple(dict.fromkeys(reasons))


def _bootstrap_member_sources(runner, *, node, ids, first_id, member_inputs_for, out, options,
                              array_module=None, initialize_callback_factory=None, member_seeds=None, provenance=()):
    """Bootstrap every non-root member through the ordinary runner and snapshot it.

    Each member's own runner initializes on this card exactly as it would
    for its own forecast, stops at the bootstrap seam (zero forecast steps),
    and is snapshotted to host before its model is released. The root
    member stays live. Returns the bound roster and one receipt per member
    with the pool bytes before the bootstrap and after its release.
    """
    from woof.ensemble.runtime_context import member_output_scope, MemberOutputCapture
    from woof.ensemble.prepared_batch import NativeMemberSources, snapshot_member_source
    if array_module is None:
        import cupy as array_module
    pool = array_module.get_default_memory_pool()
    snapshots, receipts = [], []
    for member_id in ids:
        if member_id == first_id:
            continue
        member_inputs = member_inputs_for(member_id)
        path = Path(out) / ".ensemble-initialization" / f"member-{member_id:04d}"
        path.mkdir(parents=True, exist_ok=False)
        holder = {}

        def capture(*, node, inputs, **unused):
            receipt = {"source": getattr(inputs, "source", None),
                       "prepared_root": str(getattr(inputs, "prepared_root", "") or ""),
                       "authority_sha256": dict(getattr(inputs, "authority_sha256", None)
                                                or getattr(inputs, "file_sha256", None) or {})}
            # The snapshot reads the module off the state itself (CUDA in
            # production, NumPy under a host fixture); the executor's module
            # only meters the pool around the bootstrap.
            holder["source"] = snapshot_member_source(node, member_id=member_id, receipt=receipt)
            return {"status": "PASS", "ensemble_bootstrap_only": True,
                    "completed_seconds": 0.0, "forecast_steps": 0}

        started = time.perf_counter()
        live_before = int(pool.used_bytes())
        initialize = (None if initialize_callback_factory is None else initialize_callback_factory(
            member_id=member_id, seed=(None if member_seeds is None else member_seeds[member_id]),
            prepared_member=None))
        output = (None if initialize is None and not provenance else MemberOutputCapture(
            lambda **unused: None, member_id, initialize_callback=initialize,
            provenance=(provenance[member_id] if provenance else None)))
        with member_output_scope(output):
            report = runner(member_inputs, output_directory=path, ensemble_bootstrap=capture, **options)
        report = asdict(report) if is_dataclass(report) else report
        if ("source" not in holder or not isinstance(report, dict) or report.get("status") != "PASS"
                or report.get("forecast_steps") != 0):
            raise RuntimeError(f"member {member_id} ordinary bootstrap did not stop before its first forecast step")
        del report, member_inputs
        gc.collect()
        array_module.cuda.get_current_stream().synchronize()
        receipts.append({"member_id": member_id, "path": str(path.relative_to(out)), "forecast_steps": 0,
                         "bootstrap_seconds": time.perf_counter() - started,
                         "pool_live_before_bytes": live_before,
                         "pool_live_after_release_bytes": int(pool.used_bytes()),
                         "host_snapshot_bytes": holder["source"].nbytes})
        snapshots.append(holder.pop("source"))
    sources = NativeMemberSources(first_id, snapshots, receipt={"bootstraps": receipts})
    return sources, receipts


def native_output_memory_components(node, collector):
    """Actual diagnostic, counter, replay and health declarations, CPU only."""
    from woof.core.device_inventory import state_array_shapes
    from woof.ensemble.batch_health import stability_memory_plan, state_health_memory_plan
    from woof.ensemble.batch_product_output import replay_memory_plan_for_shape
    cfg = node.cfg.run
    shape = (cfg.ny, cfg.nx)
    shapes = state_array_shapes(cfg)
    diagnostic = collector.memory_plan(shape, refl_levels=cfg.nz, host_inputs=True)
    replay = replay_memory_plan_for_shape(collector.requests, shape,
        members=collector.members, tile_rows=collector.tile_rows)
    stability, _ = stability_memory_plan(members=1, u_shape=shapes["u"],
        w_shape=shapes["w"], theta_shape=shapes["thp"])
    # Replay covers the complete roster even when a resident wave is smaller.
    # Its plan uses shared bytes so it is not multiplied by the wave count.
    components = (
        MemoryComponent("native_output_diagnostics", "products", plan=diagnostic,
            evidence="collector.memory_plan, including initial/earlier and exact 1/3/6 h counter surfaces and host uploads"),
        MemoryComponent("native_roster_replay", "products", fixed_bytes=replay.required_bytes(collector.members),
            evidence="replay_memory_plan_for_shape for the full requested roster and bounded tile rows"),
        MemoryComponent("native_step_health", "health", plan=stability,
            evidence="PreparedBatchStability allocation declarations"),
        MemoryComponent("native_full_state_health", "health", plan=state_health_memory_plan(),
            evidence="PreparedBatchStateHealth allocation declarations"),
    )
    from woof.ensemble.output_counters import CounterDeadlineCalendar
    calendar = CounterDeadlineCalendar.from_node(node)
    six_hours = 6 * 3600 * node.clock.tick_den
    ticks = sorted(calendar.deadlines)
    left, max_window = 0, 0
    for right, tick in enumerate(ticks):
        while ticks[left] < tick - six_hours:
            left += 1
        # The collector always keeps a separate exact baseline copy.
        max_window = max(max_window, right - left + 2)
    host = {"counter_endpoint_bank_bytes": cfg.ny * cfg.nx * 4 * collector.members * max_window,
            "counter_endpoint_storage": "host float32 exact observations; device surfaces included in diagnostic plan",
            "maximum_retained_endpoints_per_member": max_window,
            "forecast_volume_output_bytes": 0, "replay_members": collector.members}
    return components, host


@dataclass(frozen=True)
class PreparedExecutionPlan:
    global_member_ids: tuple[int, ...]
    seeds: tuple[int, ...]
    packing: object | None
    models: dict
    admissions: tuple[dict, ...]
    source_receipt: object
    fallback_reasons: tuple[str, ...] = ()
    bootstraps: tuple = ()

    @property
    def native(self):
        return self.packing is not None and not self.fallback_reasons

    def receipt(self):
        mapping = [{"product_slot": slot, "member_id": member, "seed": seed}
                   for slot, (member, seed) in enumerate(zip(self.global_member_ids, self.seeds))]
        return {"contract": CONTRACT, "native_admitted": self.native,
                "fallback": "ordinary_member_runner" if not self.native else None,
                "fallback_reasons": list(self.fallback_reasons), "member_slots": mapping,
                "source_roster": self.source_receipt, "admissions": list(self.admissions),
                "packing": None if self.packing is None else self.packing.receipt(),
                "warm_source_lifetime": "ordinary initialized source retained unchanged on each card until all waves end"}


def plan_initialized_member_execution(inputs, node, request, collector, *, evidence=None,
                                      card_bootstraps=(), roster=None, input_provider=None,
                                      stochastic_enabled=False, model_factory=None):
    """Make a complete packing decision before allocating a native batch."""
    request = EnsembleRequest.from_mapping(request)
    ids, seeds, source_receipt = _source_selection(request, roster)
    reasons = _fallback_reasons(inputs, node, request, roster=roster,
        input_provider=input_provider, stochastic_enabled=stochastic_enabled)
    if reasons:
        return PreparedExecutionPlan(ids, seeds, None, {}, (), source_receipt, reasons)
    evidence = evidence or sample_initialized_card_evidence(inputs)
    bootstraps = (InitializedCardBootstrap(inputs, node, evidence),) + tuple(card_bootstraps)
    if len({source.evidence.card.device_id for source in bootstraps}) != len(bootstraps):
        raise ValueError("initialized source leases must name distinct physical cards")
    if request.member_device_ids is not None and tuple(source.evidence.card.device_id for source in bootstraps) != request.member_device_ids:
        return PreparedExecutionPlan(ids, seeds, None, {}, (), source_receipt,
            ("requested member cards need one original initialized bootstrap per card; cross-device driver cloning is not supported",))
    models, admissions, cards = {}, [], []
    model_factory = native_memory_model_from_node if model_factory is None else model_factory
    for source in bootstraps:
        decision = native_prepared_eligibility(source.inputs, source.node, members=request.members,
            keep_member_files=request.keep_member_files)
        if not decision.eligible:
            return PreparedExecutionPlan(ids, seeds, None, {}, (), source_receipt, decision.reasons)
        external, host = native_output_memory_components(source.node, collector)
        sample = source.evidence
        try:
            result = model_factory(source.inputs, source.node, runtime_reservation=sample.runtime_reservation,
                allocator_margin=sample.allocator_margin, external_components=external,
                bootstrap_live_bytes=sample.bootstrap_live_bytes, free_sample_timing="after_bootstrap",
                sampled_free_bytes=sample.card.available_bytes, resident_threads=sample.resident_threads)
        except BatchStateUnsupported as error:
            return PreparedExecutionPlan(ids, seeds, None, {}, (), source_receipt,
                (f"ordinary initialized ownership has no qualified native inventory: {error}",))
        if result is None:
            return PreparedExecutionPlan(ids, seeds, None, {}, (), source_receipt,
                ("this original bootstrap has no qualified native allocation model",))
        model, sampling = result
        models[sample.card.device_id] = model
        admissions.append(dict(sample.receipt(), native_sampling=sampling, host_output_bound=host))
        cards.append(sample.card)
    packing = pack_members(request.members, tuple(cards), models, batched=True,
        max_ordinary_members_per_device=request.max_ordinary_members_per_device,
        reason="qualified common-input fixed-clock native graph; all allocations and warm source priced")
    # A zero-fit card selects the ordinary streamed door. It never receives
    # a native allocation, and its selected tiling/physics are not changed.
    return PreparedExecutionPlan(ids, seeds, packing, models, tuple(admissions), source_receipt,
                                 bootstraps=bootstraps)


class WarmInitializedRoot:
    """Audited container-only lease for one qualified ordinary bootstrap.

    Numerical arrays and immutable table owners are borrowed only as source
    inputs to native word copies. The native initializer may detach its shell
    cycle. It cannot detach the retained ordinary source state/driver cycle.
    """
    def __init__(self, inputs, node):
        from woof.core.clock import DomainClock
        from woof.core.physics import PhysicsDriver
        from woof.core.state import DomainState
        if not isinstance(node.state, DomainState) or not isinstance(node.state.physics, PhysicsDriver) or not isinstance(node.clock, DomainClock):
            raise TypeError("warm native lease supports the audited DomainState/PhysicsDriver/DomainClock graph")
        if not native_prepared_eligibility(inputs, node, members=2).eligible:
            raise ValueError("warm native lease needs a qualified ordinary initialized source")
        self.inputs, self.node = inputs, node
        self.state, self.driver, self.clock = node.state, node.state.physics, node.clock
        self._clock_key = _clock_key(self.clock)

    def wave_node(self):
        from woof.core.clock import DomainClock
        from woof.core.physics import PhysicsDriver
        from woof.core.state import DomainState
        self.require_unchanged()
        state = DomainState.__new__(DomainState)
        state.__dict__.update(self.state.__dict__)
        state._scratch = dict(self.state._scratch)
        driver = PhysicsDriver.__new__(PhysicsDriver)
        driver.__dict__.update(self.driver.__dict__)
        state.physics, driver.state = driver, state
        clock = DomainClock(self.clock.spec, self.clock.tick_den, self.clock.run_ticks)
        if _clock_key(clock) != self._clock_key:
            raise ValueError("ordinary initialized clock differs from its original fresh constructor")
        node = copy(self.node)
        node.state, node.clock = state, clock
        return node

    def require_unchanged(self):
        if (self.node.state is not self.state or self.state.physics is not self.driver
                or self.driver.state is not self.state or _clock_key(self.clock) != self._clock_key):
            raise RuntimeError("retained ordinary initialization or clock was mutated between waves")


def _clock_key(clock):
    return (clock.spec, clock.tick_den, clock.run_ticks, clock.ticks, clock.step_count,
            clock.step_ticks, clock.dt_fp32.tobytes(), clock.dtbc_fp32.tobytes(), clock.adaptive_state)


class _NativeDeclined(Exception):
    """A native pack declined during launch preparation; nothing advanced."""
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = str(reason)


class _WaveArrayModule:
    """Report allocations from this wave's explicitly owned CUDA pool."""
    def __init__(self, original, pool):
        self.original, self.pool = original, pool

    def get_default_memory_pool(self):
        return self.pool

    def __getattr__(self, name):
        return getattr(self.original, name)


def execute_initialized_member_execution(inputs, node, request, collector, *, evidence=None,
                                         card_bootstraps=(), roster=None, input_provider=None,
                                         ordinary_executor=None, initializer=None,
                                         stochastic_enabled=False, array_module=None,
                                         device_scope=None, native_runner=None, lease_factory=None,
                                         progress_callback=None, validation_callback=None,
                                         output_metadata=None, model_factory=None,
                                         step_observer=None, on_decline=None):
    """Run admitted waves, or decline before mutation for the ordinary path.

    ``ordinary_executor`` receives a single original prepared member, global
    ID, seed and selected source binding. Its usual runner owns admission,
    tile streaming, clock, stochastic binding and output. If no such callback
    is supplied for a required singleton/streamed tail, this whole handoff
    declines before advancing or allocating any native member.

    A source-audit refusal while the first wave prepares its launches is a
    decline too, with the refusal as its named reason: nothing has advanced,
    the wave's own allocations are returned and the retained ordinary source
    is verified unchanged. ``on_decline`` receives the reasons of every
    decline. ``step_observer`` follows the pack that holds the first member.
    """
    request = EnsembleRequest.from_mapping(request)
    def decline(*reasons):
        if on_decline is not None:
            on_decline(tuple(reasons))
        return None
    plan = plan_initialized_member_execution(inputs, node, request, collector,
        evidence=evidence, card_bootstraps=card_bootstraps, roster=roster,
        input_provider=input_provider, stochastic_enabled=stochastic_enabled, model_factory=model_factory)
    if not plan.native:
        return decline(*plan.fallback_reasons)
    if any(0 in batch.member_indices and batch.execution_mode != "member_batched"
           for batch in plan.packing.batches):
        # The callback already owns member zero's original initialized run.
        # Its continuation belongs to that runner, not a recursive duplicate
        # that would claim the same output directory or retain two drivers.
        return decline("the first member does not fit a native pack on its card; "
                       "its original initialized run continues")
    if ordinary_executor is None and any(batch.execution_mode != "member_batched" for batch in plan.packing.batches):
        return decline("a member outside the native packs has no ordinary executor on this door")
    if array_module is None:
        import cupy as array_module
    if device_scope is None:
        device_scope = lambda device: array_module.cuda.Device(device)
    if native_runner is None:
        from woof.ensemble.native_forecast import run_initialized_native_ensemble
        native_runner = run_initialized_native_ensemble
    lease_factory = WarmInitializedRoot if lease_factory is None else lease_factory
    sources = {source.evidence.card.device_id: source for source in plan.bootstraps}
    leases = {device: lease_factory(source.inputs, source.node) for device, source in sources.items()}
    started = time.perf_counter()
    from woof.ensemble.execution import execute_member_packing, name_failing_members
    from woof.ensemble.native_forecast import NativeLaunchRefused
    # Whether any native pack has passed launch preparation. A refusal can
    # decline only while none has: after that, frames and steps exist.
    launch_lock = threading.Lock()
    launch = {"began": False, "declined": None}
    def launch_prepared():
        with launch_lock:
            if launch["declined"] is not None:
                raise _NativeDeclined(launch["declined"])
            launch["began"] = True
    def execute(batch):
        ids = tuple(plan.global_member_ids[slot] for slot in batch.member_indices)
        try:
            return execute_admitted(batch, ids)
        except _NativeDeclined:
            raise
        except Exception as error:
            # Named by the original member IDs, not by pack slots.
            name_failing_members(error, ids, device_id=batch.device_id, wave=batch.wave,
                                 execution_mode=batch.execution_mode)
            raise
    def execute_admitted(batch, ids):
        source = sources[batch.device_id]
        seeds = tuple(plan.seeds[slot] for slot in batch.member_indices)
        with device_scope(batch.device_id):
            if batch.execution_mode != "member_batched":
                selected = None if roster is None else roster.select(ids)[0]
                member_inputs = (selected.inputs if selected is not None else source.inputs if input_provider is None
                    else input_provider(shared_inputs=inputs, member_id=ids[0], request=request))
                try:
                    return ordinary_executor(inputs=member_inputs, member_id=ids[0], seed=seeds[0],
                        prepared_member=selected, batch=batch, collector=collector)
                finally:
                    # A finished ordinary tail member's state and driver are
                    # cyclic garbage; free them before the next wave starts.
                    gc.collect()
            lease = leases[batch.device_id]
            wave_node = lease.wave_node()
            pool_type = getattr(array_module.cuda, "MemoryPool", None)
            pool = None if pool_type is None else pool_type()
            allocation_scope = nullcontext() if pool is None else array_module.cuda.using_allocator(pool.malloc)
            wave_module = array_module if pool is None else _WaveArrayModule(array_module, pool)
            # The run's step log follows the pack that holds the first member.
            observed = ({} if step_observer is None or 0 not in batch.member_indices
                        else {"step_observer": step_observer})
            try:
                with allocation_scope:
                    result = native_runner(source.inputs, wave_node, members=len(ids), member_ids=ids,
                        member_seeds=seeds, collector=collector, available_bytes=batch.available_bytes,
                        initializer=initializer, progress_callback=progress_callback,
                        validation_callback=validation_callback, output_metadata=output_metadata,
                        bootstrap_pool_live_increment_bytes=source.evidence.bootstrap_live_bytes,
                        array_module=wave_module, launch_prepared=launch_prepared, **observed)
            except NativeLaunchRefused as refusal:
                with launch_lock:
                    if launch["began"]:
                        # Another pack already holds frames or steps.
                        raise RuntimeError(
                            "a native pack's launch preparation refused after another pack had begun; "
                            f"no retry is permitted: {refusal.reason}") from refusal
                    if launch["declined"] is None:
                        launch["declined"] = refusal.reason
                    reason = launch["declined"]
                raise _NativeDeclined(reason) from refusal
            finally:
                # Dead native driver cycles are teardown, not reset or retry.
                # Only cached blocks in this wave's own pool are retired.
                gc.collect()
                array_module.cuda.get_current_stream().synchronize()
                if pool is not None:
                    pool.free_all_blocks()
            if result is None or result.get("status") != "PASS":
                raise RuntimeError("admitted native execution declined or failed after binding; no retry is permitted")
            array_module.cuda.get_current_stream().synchronize()
            lease.require_unchanged()
            return result
    try:
        results = execute_member_packing(plan.packing, execute)
    except _NativeDeclined as declined:
        # No pack began: every allocation of the wave was returned by its
        # own teardown. The retained ordinary source must be as it was.
        for lease in leases.values():
            lease.require_unchanged()
        print("ensemble: the native member pack declined and every member runs through "
              f"the ordinary runner: {declined.reason}", file=sys.stderr, flush=True)
        return decline(declined.reason)
    records = [{"batch": batch.receipt(), "global_member_ids": [plan.global_member_ids[slot] for slot in batch.member_indices],
                "result": result} for batch, result in results]
    if any(not isinstance(row["result"], dict) or row["result"].get("status") != "PASS" for row in records):
        raise RuntimeError("one original member returned a failing forecast receipt")
    return {"schema": CONTRACT, "status": "PASS", "source": inputs.source,
            "execution_plan": inputs.execution_plan, "completed_seconds": float(inputs.experiment.run_seconds),
            "backend": "native_member_batched_with_original_fallback", "wall_seconds": time.perf_counter() - started,
            "admission": plan.receipt(), "members_completed": list(plan.global_member_ids),
            "member_results": records, "member_history_count": 0,
            "qualification": "CPU strategy tested; whole-wave identity requires the release GPU gate"}


@dataclass(frozen=True)
class AutomaticPreparedResult:
    """Native completion or one completed ordinary member to cache unchanged."""
    mode: str
    report: object
    first_member_id: int
    first_seed: int
    first_output_directory: Path
    admission: object

    @property
    def native_complete(self):
        return self.mode == "native_complete"


def make_automatic_prepared_executor(runner, *, request, collector, output_directory,
                                     member_roster=None, input_provider=None, device_scope=None,
                                     card_bootstrap_factory=None, evidence_provider=None,
                                     initialized_member_binding=None, initializer=None,
                                     initialize_callback_factory=None,
                                     stochastic_enabled=False, array_module=None,
                                     model_factory=None, lease_factory=None,
                                     native_runner=None):
    """Create the Session's first prepared-run attempt, without a cold N plan.

    The returned callable is ``run(inputs, *, runner_options=None)``. It runs
    the actual ordinary initializer and supplies its initialized-root seam.
    ``native_complete`` covers every requested member. ``ordinary_first``
    contains the completed first member receipt for Session's existing loop
    to reuse; the original first member is never initialized or advanced twice.

    A source roster is consumed through ``select`` in original ID order.
    Additional cards require their own original initialized source factory.
    ``initialized_member_binding`` may attach an admitted resident stochastic
    binding at the original seam, but cannot change initialized SPP selectors.
    """
    request = EnsembleRequest.from_mapping(request)
    ids, seeds, _ = _source_selection(request, member_roster)
    out = Path(output_directory)
    def selected_inputs(shared_inputs, member_id):
        selected = None if member_roster is None else member_roster.select((member_id,))[0]
        member_inputs = (selected.inputs if selected is not None else shared_inputs if input_provider is None
            else input_provider(shared_inputs=shared_inputs, member_id=member_id, request=request))
        return member_inputs, selected
    def run(shared_inputs, *, runner_options=None):
        from woof.ensemble.runtime_context import MemberOutputCapture, member_output_scope
        from woof.ensemble.radar_output import member_history_required, finish_member_radar
        options = dict(runner_options or {})
        options.pop("ensemble_bootstrap", None)
        options["first_products"] = None
        completed_native = None
        admission = None
        ordinary_admissions = {}
        first_id, first_seed = ids[0], seeds[0]
        first_inputs, first_source = selected_inputs(shared_inputs, first_id)
        first_out = out / "members" / f"member-{first_id:04d}"
        def counter_receipts(record, capture, member_id):
            if not capture.counter_calendars:
                return record
            record = dict(record)
            record["ensemble_counter_observations"] = [calendar.receipt(member_id=member_id, episode=episode)
                for calendar in capture.counter_calendars.values()
                for observed_member, episode in calendar._progress if observed_member == member_id]
            return record
        def ordinary(inputs, member_id, seed, prepared_member, batch, collector):
            initialize = (None if initialize_callback_factory is None else initialize_callback_factory(
                member_id=member_id, seed=seed, prepared_member=prepared_member))
            capture = MemberOutputCapture(collector.submit, member_id,
                                          member_history_required(inputs, request.keep_member_files),
                                          initialize_callback=initialize,
                                          provenance=(request.provenance[member_id] if request.provenance else None))
            overlay = None
            if batch.execution_mode == "ordinary_streamed_member":
                external, margin = ordinary_admissions[batch.device_id]
                inputs, overlay = ordinary_member_execution_inputs(inputs, batch=batch,
                    external_components=external, allocator_margin=margin)
            def bind(**kwargs):
                if initialized_member_binding is not None:
                    initialized_member_binding(inputs=inputs, model=kwargs["model"], node=kwargs["node"],
                        member_id=member_id, seed=seed, prepared_member=prepared_member)
                return None
            member_out = out / "members" / f"member-{member_id:04d}"
            member_out.mkdir(parents=True, exist_ok=False)
            with member_output_scope(capture):
                result = runner(inputs, output_directory=member_out,
                    ensemble_bootstrap=bind, **options)
            record = asdict(result) if is_dataclass(result) else result
            radar = finish_member_radar(inputs, member_out, result, keep_member_files=request.keep_member_files)
            if radar is not None:
                record = dict(record, ensemble_radar_products=radar)
            if overlay is not None:
                record = dict(record, ensemble_execution_overlay=overlay)
            return counter_receipts(record, capture, member_id)
        def handoff(*, inputs, model, node, output_directory, observer, step_observer):
            nonlocal completed_native, admission
            if initialized_member_binding is not None:
                initialized_member_binding(inputs=inputs, model=model, node=node,
                    member_id=first_id, seed=first_seed, prepared_member=first_source)
            reasons = _fallback_reasons(inputs, node, request, roster=member_roster,
                input_provider=input_provider, stochastic_enabled=stochastic_enabled,
                require_member_sources=False)
            if reasons:
                admission = {"contract": CONTRACT, "native_admitted": False,
                    "fallback_reasons": list(reasons), "first_member_id": first_id,
                    "first_member_seed": first_seed}
                return None
            # Members with their own inputs: bootstrap each one through its
            # own ordinary initializer on this card, snapshot it, release it,
            # and bind the roster to this root before anything is priced.
            bootstrap_receipts = []
            sources = None
            if _member_specific(request, member_roster, input_provider) and any(member != first_id for member in ids):
                from woof.ensemble.prepared_batch import bind_member_sources
                sources, bootstrap_receipts = _bootstrap_member_sources(runner, node=node, ids=ids,
                    first_id=first_id, member_inputs_for=lambda member: selected_inputs(shared_inputs, member)[0],
                    out=out, options=options, array_module=array_module,
                    initialize_callback_factory=initialize_callback_factory,
                    member_seeds=dict(zip(ids, seeds)), provenance=request.provenance)
                reasons = sources.compatibility_reasons(node)
                if reasons:
                    admission = {"contract": CONTRACT, "native_admitted": False,
                        "fallback_reasons": list(reasons), "first_member_id": first_id,
                        "first_member_seed": first_seed, "member_bootstraps": bootstrap_receipts}
                    return None
                bind_member_sources(node, sources)
            sample = (sample_initialized_card_evidence(inputs, array_module=array_module)
                if evidence_provider is None else evidence_provider(inputs=inputs, model=model, node=node))
            additional = ()
            requested_cards = request.member_device_ids or (sample.card.device_id,)
            if requested_cards != (sample.card.device_id,):
                if card_bootstrap_factory is None:
                    admission = {"contract": CONTRACT, "native_admitted": False,
                        "fallback_reasons": ["multiple member cards need original per-card bootstrap leases"],
                        "first_member_id": first_id}
                    return None
                if sources is not None and first_inputs is not shared_inputs:
                    admission = {"contract": CONTRACT, "native_admitted": False,
                        "fallback_reasons": [f"additional card roots are initialized from the shared inputs, "
                                             f"not from member {first_id}'s own trajectory; one card packs this roster"],
                        "first_member_id": first_id, "member_bootstraps": bootstrap_receipts}
                    return None
                additional = tuple(card_bootstrap_factory(device_id=device, shared_inputs=shared_inputs,
                    request=request) for device in requested_cards if device != sample.card.device_id)
                if sources is not None:
                    for extra in additional:
                        bind_member_sources(extra.node, sources)
            for source in (InitializedCardBootstrap(inputs, node, sample),) + additional:
                external, _ = native_output_memory_components(source.node, collector)
                ordinary_admissions[source.evidence.card.device_id] = (external, source.evidence.allocator_margin)
            output_metadata = None
            if getattr(inputs, "domains", None) is None:
                # Single-domain prepared inputs carry their static fields
                # directly; the tree road reads them off its domain bundle.
                from woof.runtime import _metadata_frame
                output_metadata = _metadata_frame(node.grid, inputs.static)
            def declined(reasons):
                nonlocal admission
                admission = {"contract": CONTRACT, "native_admitted": False,
                    "fallback": "ordinary_member_runner", "fallback_reasons": list(reasons),
                    "first_member_id": first_id, "first_member_seed": first_seed,
                    "member_bootstraps": bootstrap_receipts}
            completed_native = execute_initialized_member_execution(inputs, node, request, collector,
                evidence=sample, card_bootstraps=additional, roster=member_roster,
                input_provider=input_provider, ordinary_executor=ordinary, initializer=initializer,
                stochastic_enabled=stochastic_enabled, array_module=array_module,
                device_scope=device_scope, model_factory=model_factory, lease_factory=lease_factory,
                native_runner=native_runner, progress_callback=observer, output_metadata=output_metadata,
                step_observer=step_observer, on_decline=declined)
            if completed_native is not None:
                admission = dict(completed_native["admission"], member_bootstraps=bootstrap_receipts)
                completed_native["admission"] = admission
            return completed_native
        initialize = (None if initialize_callback_factory is None else initialize_callback_factory(
            member_id=first_id, seed=first_seed, prepared_member=first_source))
        capture = MemberOutputCapture(collector.submit, first_id,
                                      member_history_required(first_inputs, request.keep_member_files),
                                      initialize_callback=initialize,
                                      provenance=(request.provenance[first_id] if request.provenance else None))
        initial_device = None if request.member_device_ids is None else request.member_device_ids[0]
        scope = nullcontext() if initial_device is None or device_scope is None else device_scope(initial_device)
        # The ensemble session claims every member directory before execution (production.py, the directory
        # holding only member-provenance.json); the first member runs in that claim. Any other content is
        # another run's output and is still refused (member-roster ensembles failed here with FileExistsError).
        if not (first_out.is_dir() and {q.name for q in first_out.iterdir()} <= {"member-provenance.json"}):
            first_out.mkdir(parents=True, exist_ok=False)
        with scope, member_output_scope(capture):
            result = runner(first_inputs, output_directory=first_out, ensemble_bootstrap=handoff, **options)
        report = asdict(result) if is_dataclass(result) else result
        if not isinstance(report, dict) or report.get("status") != "PASS":
            raise RuntimeError("automatic prepared attempt returned a failing original forecast receipt")
        if completed_native is None:
            radar = finish_member_radar(first_inputs, first_out, result, keep_member_files=request.keep_member_files)
            if radar is not None:
                report = dict(report, ensemble_radar_products=radar)
            report = counter_receipts(report, capture, first_id)
        return AutomaticPreparedResult("native_complete" if completed_native is not None else "ordinary_first",
            report, first_id, first_seed, first_out, admission)
    return run


__all__ = ["CONTRACT", "InitializedCardEvidence", "InitializedCardBootstrap", "PreparedExecutionPlan",
           "WarmInitializedRoot", "sample_initialized_card_evidence", "native_output_memory_components",
           "plan_initialized_member_execution", "execute_initialized_member_execution",
           "AutomaticPreparedResult", "make_automatic_prepared_executor",
           "member_device_ids_for_request", "ordinary_member_execution_inputs"]
