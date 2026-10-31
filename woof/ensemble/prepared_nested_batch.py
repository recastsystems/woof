"""Hybrid member nesting around genuine prepared trees and stock callbacks.

The ordinary trees retain their state, initialized physics and independent
integer clocks. Explicitly priced resident banks supply admitted edge
operations; ordinary STEP remains the default. This is a component execution
seam, not admission of the complete native forecast graph.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import copy_context
from copy import deepcopy
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from operator import index
from types import MappingProxyType, SimpleNamespace

import numpy as np

from woof.core.cfl_member import member_cfl_scope
from woof.core.device_inventory import state_array_shapes
from woof.ensemble.batch_state import (
    BatchedDomainState, BatchStateUnsupported, SHARED_STATE_CANDIDATES,
    _REQUIRED_SCALARS, _exact_key, state_array_specs,
)
from woof.ensemble.batch_storage import BatchMemoryPlan, BatchStorage
from woof.ensemble.packed_schedule import (
    MemberScheduleBinding, ScheduleBatchAdmission, execute_packed_member_schedules,
)

CONTRACT = "gpuwm-prepared-hybrid-nested-members-v1"
_STATE_NAMES = {"t": "thp", "ph": "php", "mu": "mup"}


def _require_identity_transition(contract, source_cfg, target_cfg, fields, *, optional=False):
    """Accept the stock same-scheme contract only when it transforms nothing."""
    from woof.core.microphysics_transition import (
        MicrophysicsTransitionContract, transition_handles_field, transition_target_fields,
    )
    if optional and contract is None:
        return
    if (not isinstance(contract, MicrophysicsTransitionContract) or contract.mixed
            or contract.source_mp_physics != int(source_cfg.mp_physics)
            or contract.target_mp_physics != int(target_cfg.mp_physics)
            or int(source_cfg.mp_physics) != int(target_cfg.mp_physics)
            or transition_target_fields(contract)
            or any(transition_handles_field(contract, field) for field in fields)):
        raise BatchStateUnsupported("mixed microphysics FORCE/feedback requires its original diagnosed-species transition")


def _bytes(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an integer byte count")
    value = index(value)
    if value < 0:
        raise ValueError(f"{label} must not be negative")
    return value


def capture_original_member_schedule(member_id, model, *, operation_authority,
                                     execution_options=None, execution_callable=None):
    """Capture the actual executor closures without advancing any member.

    All setup and controller construction belongs to ``execute_experiment``.
    Its STEP closure already owns ``before_step`` and retains health, clock,
    output and runtime-status behavior. No numerical callback is copied here.
    """
    if execution_callable is None:
        from woof.core.model import execute_experiment
        execution_callable = execute_experiment
    options = dict(execution_options or {})
    if "schedule_dispatch" in options:
        raise ValueError("schedule capture owns its executor dispatch seam")
    captures = []
    with member_cfl_scope() as owner:
        def capture(*, model, schedule, callbacks, adaptive_driver, clocks, step_bindings,
                    skip_feedback_path, start_period, started_grid_ids,
                    committed_initial_history_grid_ids):
            if schedule is not model.schedule or any(
                    clocks[int(node.cfg.grid_id)] is not node.clock
                    for node in model.walk_parent_first()):
                raise ValueError("captured schedule or domain clocks differ from their original owners")
            binding = MemberScheduleBinding(member_id, model, callbacks,
                operation_authority=operation_authority, adaptive_driver=adaptive_driver,
                start_period=start_period, started_grid_ids=tuple(started_grid_ids),
                committed_initial_history_grid_ids=tuple(committed_initial_history_grid_ids),
                skip_feedback_path=skip_feedback_path, step_owns_before_step=True,
                step_bindings=step_bindings)
            # The executor enabled its controller inside this outer owner.
            # Preserve that same context for every callback and CFL read.
            binding.context, binding.cfl_owner = copy_context(), owner
            captures.append(binding)
            return binding
        result = execution_callable(model, schedule_dispatch=capture, **options)
    if len(captures) != 1 or result is not captures[0]:
        raise RuntimeError("ordinary executor did not return exactly its captured original schedule")
    binding = captures[0]
    # Capture exits the ordinary executor's finally block before forecasting.
    # This owner is private, and no CFL reduction has been launched yet.
    binding.cfl_owner.enabled = binding.adaptive_driver is not None
    return binding


@dataclass(frozen=True)
class PreparedTreeMember:
    member_id: int
    inputs: object
    model: object
    receipt: dict


def bootstrap_prepared_tree_members(member_ids, *, inputs_for, runner, output_for,
                                    runner_options=None, member_initializer=None):
    """Stop each ordinary prepared runner at its existing initialization seam.

    ``inputs_for`` owns reuse of a shared immutable preparation. Every roster
    member is still initialized from its own selected inputs and physics.
    The returned model handles deliberately retain their genuine GPU owners.
    """
    ids = tuple(member_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("prepared nested bootstrap needs unique member IDs")
    options = dict(runner_options or {})
    if "ensemble_bootstrap" in options:
        raise ValueError("member tree bootstrap owns the ordinary handoff seam")
    rows = []
    for member_id in ids:
        inputs, captured = inputs_for(member_id), []
        def capture(*, inputs, model, node, **unused):
            if node is not model.root:
                raise ValueError("prepared tree bootstrap handed off a non-root node")
            if member_initializer is not None:
                member_initializer(member_id=member_id, inputs=inputs, model=model)
            nodes = tuple(model.walk_parent_first())
            if any(node.clock.step_count != 0 or node.clock.ticks != node.clock.spec.start_ticks
                   for node in nodes):
                raise ValueError("prepared member bootstrap occurs after forecast stepping")
            receipt = {"member_id": member_id, "domain_ids": [int(node.cfg.grid_id) for node in nodes],
                "ordinary_initialization": True, "forecast_steps": 0,
                "physics_selection": "unchanged", "clock_selection": "unchanged"}
            captured.append(PreparedTreeMember(member_id, inputs, model, receipt))
            return {"status": "PASS", "ensemble_bootstrap_only": True}
        result = runner(inputs, output_directory=output_for(member_id), ensemble_bootstrap=capture, **options)
        if (len(captured) != 1 or not isinstance(result, dict)
                or result.get("ensemble_bootstrap_only") is not True):
            raise RuntimeError(f"member {member_id} ordinary preparation did not stop at its initialized tree")
        rows.append(captured[0])
    return tuple(rows)


def _same_words(xp, left, right):
    return bool(xp.array_equal(left.view(np.uint8), right.view(np.uint8)))


def _domain_bank_plan(nodes, *, shared_fields, array_module, field_names=None):
    nodes = tuple(nodes)
    cfg, xp = nodes[0].cfg.run, array_module
    if any(node.cfg.run.chem_sets for node in nodes):
        raise BatchStateUnsupported("active chemistry keeps its original member species arrays, source-hour caches and mass ledger; component banks lack their typed state ownership and producer handoff")
    specs = state_array_specs(cfg, shared_fields=shared_fields)
    if field_names is not None:
        names = frozenset(field_names)
        missing = names - {spec.name for spec in specs}
        if missing:
            raise BatchStateUnsupported(f"edge kernel dependencies {sorted(missing)} lack inventoried original state allocations")
        if not names:
            raise BatchStateUnsupported("a grid with no nest edge has no audited edge-only state bank")
        specs = tuple(spec for spec in specs if spec.name in names)
    config_key = _exact_key(cfg)
    for slot, node in enumerate(nodes):
        if _exact_key(node.cfg.run) != config_key:
            raise BatchStateUnsupported(f"member slot {slot} has a different domain configuration; one edge ABI cannot substitute it")
        if getattr(node.state, "_streamed_domain", None) is not None:
            raise BatchStateUnsupported("resident edge banks cannot read a streamed domain's attach-time state")
        if any(not hasattr(node.state, name) for name in _REQUIRED_SCALARS):
            raise BatchStateUnsupported("resident edge bank omits original diagnostic scalar carriers")
        for spec in specs:
            value = getattr(node.state, spec.name)
            if (not isinstance(value, xp.ndarray) or tuple(value.shape) != spec.shape
                    or value.dtype != np.dtype(spec.dtype) or not value.flags.c_contiguous):
                raise BatchStateUnsupported(f"member slot {slot} field {spec.name} lacks its original resident shape/dtype/layout")
            if spec.ownership == "shared" and slot and not _same_words(xp, value, getattr(nodes[0].state, spec.name)):
                raise BatchStateUnsupported(f"shared field {spec.name} has different member words; sharing would broadcast the wrong state")
    return BatchMemoryPlan(specs, reserved_bytes=0)


class ScheduledDomainBanks(BatchedDomainState):
    """Copies for edge operations; live member clocks remain external owners.

    This explicitly prices retained ordinary states alongside these copies.
    It does not alter ``from_prepared``'s fixed-clock contract, erase adaptive
    state, change a configuration or rebind initialized physics pointers.
    """
    def __init__(self, nodes, plan, *, array_module, available_bytes):
        self.nodes, self.xp = tuple(nodes), array_module
        self.members, self.plan = len(self.nodes), plan
        self.storage = BatchStorage(plan, self.members, array_module=array_module, available_bytes=available_bytes)
        self.cfg = deepcopy(self.nodes[0].cfg.run)
        self.scalars = MappingProxyType({name: getattr(self.nodes[0].state, name) for name in _REQUIRED_SCALARS})
        self._scratch, self.phb_host_members = {}, ()
        self.clock, self.physics, self.lateral_boundaries = None, None, None
        self._host_setup_state = array_module is np
        self.copy_in_bytes = self.copy_out_bytes = 0
        self.synchronize_from_originals()

    def validate_originals(self):
        config = _exact_key(self.nodes[0].cfg.run)
        scalars = {name: getattr(self.nodes[0].state, name) for name in _REQUIRED_SCALARS}
        scalar_key = _exact_key(scalars)
        for node in self.nodes:
            if getattr(node.state, "_streamed_domain", None) is not None:
                raise BatchStateUnsupported("live edge endpoint became streamed; its resident fields are no longer the canonical source")
            if _exact_key(node.cfg.run) != config or _exact_key(
                    {name: getattr(node.state, name) for name in _REQUIRED_SCALARS}) != scalar_key:
                raise BatchStateUnsupported("live member diagnostic configuration or scalar words differ; regroup before a packed edge")
            for spec in self.plan.arrays:
                value = getattr(node.state, spec.name, None)
                if (not isinstance(value, self.xp.ndarray) or tuple(value.shape) != spec.shape
                        or value.dtype != np.dtype(spec.dtype) or not value.flags.c_contiguous):
                    raise BatchStateUnsupported(f"live field {spec.name} changed its original resident shape/dtype/layout; rebind before a packed edge")
                original_device, bank_device = getattr(value, "device", None), getattr(self.storage.arrays[spec.name], "device", None)
                if original_device != bank_device:
                    raise BatchStateUnsupported(f"live field {spec.name} moved to another device; no admitted edge transfer exists")
        for spec in self.plan.arrays:
            if spec.ownership == "shared":
                source = getattr(self.nodes[0].state, spec.name)
                if any(not _same_words(self.xp, source, getattr(node.state, spec.name)) for node in self.nodes[1:]):
                    raise BatchStateUnsupported(f"live shared field {spec.name} differs; regroup before a packed edge")

    def synchronize_from_originals(self):
        self.validate_originals()
        scalars = {name: getattr(self.nodes[0].state, name) for name in _REQUIRED_SCALARS}
        self.cfg, self.scalars = deepcopy(self.nodes[0].cfg.run), MappingProxyType(deepcopy(scalars))
        for spec in self.plan.arrays:
            target = self.storage.arrays[spec.name]
            if spec.ownership == "shared":
                source = getattr(self.nodes[0].state, spec.name)
                self.xp.copyto(target, source, casting="no")
                self.copy_in_bytes += int(source.nbytes)
            else:
                for slot, node in enumerate(self.nodes):
                    source = getattr(node.state, spec.name)
                    self.xp.copyto(target[slot], source, casting="no")
                    self.copy_in_bytes += int(source.nbytes)

    def copy_mutations_to_originals(self, names):
        for name in names:
            if self.storage.specs[name].ownership != "member":
                raise ValueError(f"packed edge cannot write shared field {name}")
            for slot, node in enumerate(self.nodes):
                source = self.member_view(name, slot)
                self.xp.copyto(getattr(node.state, name), source, casting="no")
                self.copy_out_bytes += int(source.nbytes)


class PreparedHybridNestedBatch:
    """Run captured original schedules with an explicitly injected pack factory.

    A factory declares all retained edge/physics plans before allocation and
    provides qualified group admission plus numerical callbacks. The default
    has no packed callbacks. Failures after mutation are never retried.
    """
    def __init__(self, bindings, *, ordinary_forecast_bytes, available_bytes,
                 ordinary_memory_evidence, pack_factory=None, array_module=None,
                 shared_fields=tuple(sorted(SHARED_STATE_CANDIDATES)), completion_wait):
        self.bindings = tuple(bindings)
        if (not self.bindings or any(not isinstance(binding, MemberScheduleBinding) for binding in self.bindings)
                or len({binding.member_id for binding in self.bindings}) != len(self.bindings)):
            raise ValueError("hybrid nested execution needs unique original schedule bindings")
        if not ordinary_memory_evidence or not callable(completion_wait):
            raise ValueError("hybrid admission needs full ordinary forecast evidence and callback queue completion")
        if array_module is None:
            import cupy as array_module
        self.xp, self.completion_wait = array_module, completion_wait
        self._closed, self._retired_copies = False, None
        self.available_bytes = _bytes(available_bytes, "available_bytes")
        self.ordinary_memory_evidence = str(ordinary_memory_evidence)
        ids = tuple(binding.member_id for binding in self.bindings)
        if set(ordinary_forecast_bytes) != set(ids):
            raise ValueError("hybrid admission needs the full ordinary forecast envelope of every retained member")
        for binding in self.bindings:
            resident_fields = sum(int(getattr(node.state, name).nbytes)
                for node in binding.model.walk_parent_first()
                for name in state_array_shapes(node.cfg.run)
                if getattr(node.state, name, None) is not None)
            if _bytes(ordinary_forecast_bytes[binding.member_id], "ordinary forecast bytes") < resident_fields:
                raise ValueError(f"member {binding.member_id} forecast reservation is smaller than its retained resident state fields")
        self.ordinary_bytes = sum(_bytes(ordinary_forecast_bytes[member], "ordinary forecast bytes") for member in ids)
        self.banks, self.plans, self.packed = {}, {}, None
        self.fallback_reasons = []
        topology = tuple(self.bindings[0].clocks)
        if any(tuple(binding.clocks) != topology for binding in self.bindings):
            self.fallback_reasons.append("member domain trees differ; no common resident edge bank layout")
            pack_factory = None
        if pack_factory is not None:
            try:
                for gid in topology:
                    nodes = tuple(binding.model.node(gid) for binding in self.bindings)
                    selected = (pack_factory.state_fields_for_grid(self.bindings, gid)
                        if hasattr(pack_factory, "state_fields_for_grid") else None)
                    self.plans[gid] = _domain_bank_plan(nodes, shared_fields=shared_fields,
                        array_module=array_module, field_names=selected)
                descriptors = {gid: SimpleNamespace(cfg=self.bindings[0].model.node(gid).cfg.run,
                    storage=SimpleNamespace(specs={spec.name: spec for spec in plan.arrays}))
                    for gid, plan in self.plans.items()}
                extra = dict(pack_factory.memory_plans(self.bindings, descriptors))
                if any(not isinstance(plan, BatchMemoryPlan) for plan in extra.values()):
                    raise TypeError("qualified pack factory must price every allocation with BatchMemoryPlan")
            except BatchStateUnsupported as error:
                self.fallback_reasons.append(str(error))
                self.plans.clear()
                pack_factory = None
        extra = {} if pack_factory is None else extra
        self.additional_plans = extra
        members = len(self.bindings)
        self.required_bytes = self.ordinary_bytes + sum(plan.required_bytes(members) for plan in
                                                       tuple(self.plans.values()) + tuple(extra.values()))
        if self.required_bytes > self.available_bytes:
            raise MemoryError(f"hybrid members retain {self.ordinary_bytes} ordinary forecast bytes and need {self.required_bytes} total bytes; {self.available_bytes} bytes are available")
        remaining = self.available_bytes - self.ordinary_bytes
        if pack_factory is not None:
            try:
                for gid, plan in self.plans.items():
                    self.banks[gid] = ScheduledDomainBanks(tuple(binding.model.node(gid) for binding in self.bindings),
                        plan, array_module=array_module, available_bytes=remaining)
                    remaining -= plan.required_bytes(members)
                self.packed = pack_factory.prepare(self.bindings, self.banks, available_bytes=remaining, array_module=array_module)
            except BaseException as error:
                for cleanup in (self.completion_wait, getattr(pack_factory, "close", None)):
                    if cleanup is not None:
                        try:
                            cleanup()
                        except BaseException as cleanup_error:
                            error.add_note(f"hybrid preparation cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")
                self.banks.clear()
                raise
        for reason in self.fallback_reasons:
            print(f"hybrid members {ids}: original callbacks: {reason}", flush=True)

    def execute(self, *, original_group_dispatch=None):
        if self._closed or self._retired_copies is not None:
            raise RuntimeError("hybrid wave retirement started; its original bindings must be initialized again")
        for binding in self.bindings:
            if binding.adaptive_driver is not None:
                binding.cfl_owner.enabled = True
        fallback_records, recorded = [], set()
        def record(group, row):
            if self.packed is None or row["packed"] or group.kind not in self.packed.callbacks:
                return
            reason = row["admission_reason"] or (
                "one ready member cannot use this all-roster bank binding; remapping its state would substitute another member")
            key = (group.kind, group.member_ids, reason)
            if key not in recorded:
                recorded.add(key)
                fallback_records.append({"operation": group.kind, "member_ids": list(group.member_ids), "reason": reason})
                print(f"hybrid {group.kind} members {group.member_ids}: original callbacks: {reason}", flush=True)
        def complete():
            self._complete_queues()
        result = execute_packed_member_schedules(self.bindings,
            admitted_callbacks={} if self.packed is None else self.packed.callbacks,
            batch_admission=None if self.packed is None else self.packed.admit,
            completion_wait=complete, record_callback=record,
            original_group_dispatch=original_group_dispatch)
        result.update(contract=CONTRACT, execution="hybrid_original_members_with_admitted_components",
            complete_native_forecast=False, ordinary_step_default=True,
            fallback_reasons=list(self.fallback_reasons), runtime_fallbacks=fallback_records, memory={
                "ordinary_forecasts_retained_bytes": self.ordinary_bytes,
                "ordinary_memory_evidence": self.ordinary_memory_evidence,
                "bank_bytes": sum(plan.required_bytes(len(self.bindings)) for plan in self.plans.values()),
                "component_bytes": sum(plan.required_bytes(len(self.bindings)) for plan in self.additional_plans.values()),
                "total_required_bytes": self.required_bytes, "available_bytes": self.available_bytes},
            copies={"into_banks_bytes": sum(bank.copy_in_bytes for bank in self.banks.values()),
                    "to_originals_bytes": sum(bank.copy_out_bytes for bank in self.banks.values())},
            original_configuration="unchanged", original_adaptive_controllers="retained")
        if self.packed is not None:
            result["packed_components"] = self.packed.receipt()
        return result

    def _complete_queues(self):
        error = None
        callbacks = ((self.packed.completion_wait,) if self.packed is not None
                     and hasattr(self.packed, "completion_wait") else ()) + (self.completion_wait,)
        for callback in callbacks:
            try:
                callback()
            except BaseException as caught:
                if error is None:
                    error = caught
                elif caught is not error:
                    error.add_note(f"another hybrid queue completion also failed: {type(caught).__name__}: {caught}")
        if error is not None:
            raise error

    def close(self):
        """Retire this wave after queues drain; external models remain usable."""
        if self._closed:
            return
        if self._retired_copies is None:
            self._retired_copies = {"into_banks_bytes": sum(bank.copy_in_bytes for bank in self.banks.values()),
                                   "to_originals_bytes": sum(bank.copy_out_bytes for bank in self.banks.values())}
        error = None
        try:
            self._complete_queues()
        except BaseException as caught:
            error = caught
        try:
            if self.packed is not None and hasattr(self.packed, "close"):
                self.packed.close()
        except BaseException as caught:
            if error is None:
                error = caught
            elif caught is not error:
                error.add_note(f"hybrid component retirement also failed: {type(caught).__name__}: {caught}")
        if error is not None:
            raise error
        self.banks.clear()
        self.bindings = ()
        self._closed = True


class ResidentNestedEdgeFactory:
    """An explicitly supplied qualified resident-edge component factory.

    Only a complete roster with its bound original step can use this initial
    edge binding. Divergent frontiers and changed adaptive intervals keep
    their original callbacks. Ozone routing remains the genuine per-member
    producer. A full forecast admission is deliberately not granted here.
    """
    def __init__(self, *, qualification_receipt, edge_state_only=False):
        if not isinstance(qualification_receipt, dict) or not qualification_receipt:
            raise ValueError("resident edge factory needs its component qualification receipt")
        self.qualification_receipt = dict(qualification_receipt)
        if type(edge_state_only) is not bool:
            raise TypeError("edge-only bank inventory needs an explicit boolean ownership declaration")
        self.edge_state_only = edge_state_only
        self.bindings, self.member_ids = (), ()
        self.nodes, self.fields, self.banks, self.edges, self.owner_keys, self.callbacks = {}, {}, {}, {}, {}, {}
        self._edge_streams, self._retirement_receipt, self._closed, self._retiring = {}, None, False, False

    def state_fields_for_grid(self, bindings, grid_id):
        if not self.edge_state_only:
            return None
        from woof.ensemble.batch_nesting import prepared_tree_edge_state_fields
        domains = tuple(node.cfg for node in bindings[0].model.walk_parent_first())
        return prepared_tree_edge_state_fields(domains)[int(grid_id)]

    @staticmethod
    def _owner_key(node):
        return (id(node), id(node.state), id(node.parent), id(node.parent.state), id(node.coupler),
                int(node.coupler.placement_generation), id(node.clock), id(node.parent.clock))

    def memory_plans(self, bindings, descriptors):
        from woof.core.nest import NestCoupler
        from woof.core.preflight import nest_field_kinds
        from woof.ensemble.batch_nesting import member_nest_memory_plan
        self.bindings = tuple(bindings)
        self.member_ids = tuple(binding.member_id for binding in bindings)
        self.nodes, self.fields, result = {}, {}, {}
        for node in bindings[0].model.walk_parent_first():
            if node.parent is None:
                continue
            gid, parent_id = int(node.cfg.grid_id), int(node.parent.cfg.grid_id)
            nodes = tuple(binding.model.node(gid) for binding in bindings)
            fields = tuple(nest_field_kinds(node.cfg.run))
            for candidate in nodes:
                coupler = candidate.coupler
                if not isinstance(coupler, NestCoupler):
                    raise BatchStateUnsupported("resident packed edge needs the actual ordinary NestCoupler metadata owner")
                if tuple(nest_field_kinds(candidate.parent.cfg.run)) != fields:
                    raise BatchStateUnsupported("mixed microphysics FORCE/feedback requires its original diagnosed-species transition")
                _require_identity_transition(coupler.microphysics_transition,
                    candidate.parent.cfg.run, candidate.cfg.run, fields)
                _require_identity_transition(coupler.microphysics_reverse_transition,
                    candidate.cfg.run, candidate.parent.cfg.run, fields, optional=coupler.feedback == 0)
                if coupler.inflow_perturbation is not None:
                    raise BatchStateUnsupported("active nest inflow keeps its original post-interpolation perturbation binding")
                if coupler.placement_generation != node.coupler.placement_generation:
                    raise BatchStateUnsupported("member nest placement generations differ; shared edge geometry would be stale")
                for stagger, reg in node.coupler.registrations.items():
                    other = coupler.registrations[stagger]
                    if any(_exact_key(getattr(reg, name)) != _exact_key(getattr(other, name))
                           for name in ("nri", "nrj", "i_parent_start", "j_parent_start", "nxc", "nyc", "nxp", "nyp", "xstag", "ystag", "wrapper")):
                        raise BatchStateUnsupported("member edge registration differs; shared donor geometry would force another nest")
                    if any(getattr(reg, name).tobytes() != getattr(other, name).tobytes()
                           for name in ("ci", "ip", "cj", "jp", "xig", "xjg")):
                        raise BatchStateUnsupported("member edge donor words differ; shared interpolation would force another nest")
            self.nodes[gid], self.fields[gid] = nodes, fields
            result[f"edge:{gid}"] = member_nest_memory_plan(descriptors[parent_id], descriptors[gid],
                                                            node.coupler.registrations, fields)
        if not result:
            raise BatchStateUnsupported("resident nested edge factory has no actual parent/child edge")
        return result

    def prepare(self, bindings, banks, *, available_bytes, array_module):
        from woof.ensemble.batch_nesting import PreparedMemberNestEdge
        self.banks, self.edges, self.xp = banks, {}, array_module
        self._note_edge_stream()
        self.owner_keys = {}
        remaining = available_bytes
        for gid, nodes in self.nodes.items():
            node = nodes[0]
            edge = PreparedMemberNestEdge(banks[int(node.parent.cfg.grid_id)], banks[gid],
                registrations=node.coupler.registrations, fields=self.fields[gid],
                parent_dt_fp32=node.parent.clock.dt_fp32,
                parent_interval_ticks=node.parent.clock.step_ticks,
                parent_tick_den=node.parent.clock.tick_den, available_bytes=remaining,
                feedback=node.coupler.feedback, smooth_option=node.coupler.smooth_option,
                array_module=array_module, member_ids=self.member_ids,
                parent_clocks=tuple(candidate.parent.clock for candidate in nodes),
                child_clocks=tuple(candidate.clock for candidate in nodes))
            remaining -= edge.plan.required_bytes(len(bindings))
            self.edges[gid] = edge
            self.owner_keys[gid] = tuple(self._owner_key(candidate) for candidate in nodes)
        self.callbacks = {kind: self.execute for kind in ("force", "feedback_commit", "feedback_finalize")}
        return self

    def admit(self, group):
        if self._closed or self._retiring:
            return ScheduleBatchAdmission(False, "packed edge wave retirement started; its former component owners cannot resume")
        if group.member_ids != self.member_ids:
            return ScheduleBatchAdmission(False, "divergent member frontier keeps its original edge; this bank binding owns the complete roster")
        gid = group.requests[0].grid_id
        edge = self.edges.get(gid)
        if edge is None:
            return ScheduleBatchAdmission(False, "no qualified resident packed binding for this edge")
        try:
            self.banks[gid].validate_originals()
            self.banks[int(self.nodes[gid][0].parent.cfg.grid_id)].validate_originals()
        except BatchStateUnsupported as error:
            return ScheduleBatchAdmission(False, str(error))
        for slot, request in enumerate(group.requests):
            node = request.binding.model.node(gid)
            if (self._owner_key(node) != self.owner_keys[gid][slot] or node.coupler.inflow_perturbation is not None):
                return ScheduleBatchAdmission(False, "edge placement or inflow changed; original coupling owns reconstruction")
            if group.kind == "force":
                try:
                    for stagger, reg in edge.registrations.items():
                        other = node.coupler.registrations[stagger]
                        if (any(_exact_key(getattr(reg, name)) != _exact_key(getattr(other, name))
                                for name in ("nri", "nrj", "i_parent_start", "j_parent_start", "nxc", "nyc", "nxp", "nyp", "xstag", "ystag", "wrapper"))
                                or any(getattr(reg, name).tobytes() != getattr(other, name).tobytes()
                                       for name in ("ci", "ip", "cj", "jp", "xig", "xjg"))):
                            raise BatchStateUnsupported("original member donor geometry changed; rebuilding only scalar FORCE args would retain stale donors")
                    _require_identity_transition(node.coupler.microphysics_transition,
                        node.parent.cfg.run, node.cfg.run, self.fields[gid])
                except (BatchStateUnsupported, KeyError) as error:
                    return ScheduleBatchAdmission(False, str(error))
        if group.kind == "force":
            try:
                edge.validate_parent_interval(parent_clocks=tuple(node.parent.clock for node in self.nodes[gid]),
                                              child_clocks=tuple(node.clock for node in self.nodes[gid]))
            except (BatchStateUnsupported, ValueError) as error:
                return ScheduleBatchAdmission(False, str(error))
        if group.kind == "feedback_finalize" and edge._prepared_feedback is None:
            return ScheduleBatchAdmission(False, "feedback commit ran ordinarily; its original finalize owns the same transaction")
        return ScheduleBatchAdmission(True, "qualified complete-roster resident edge with original member clocks")

    def execute(self, group):
        if self._closed or self._retiring:
            raise RuntimeError("packed edge wave retirement started; no admitted coupling owners remain")
        self._note_edge_stream()
        gid, kind = group.requests[0].grid_id, group.kind
        nodes, edge = self.nodes[gid], self.edges[gid]
        parent = self.banks[int(nodes[0].parent.cfg.grid_id)]
        child = self.banks[gid]
        # Validate every endpoint before ozone, coupling or feedback writes.
        parent.validate_originals()
        child.validate_originals()
        if kind == "force":
            admission = self.admit(group)
            if not admission.eligible:
                raise BatchStateUnsupported("packed FORCE authority changed after admission: " + admission.reason)
            edge.rebind_parent_interval(parent_clocks=tuple(node.parent.clock for node in nodes),
                                        child_clocks=tuple(node.clock for node in nodes))
            from woof.core.cam_ozone import transfer_parent_ozone
            for request, node in zip(group.requests, nodes, strict=True):
                transferred = request.binding.context.run(transfer_parent_ozone, node, node.coupler.registrations["m"])
                node.coupler.force_sync_bytes += transferred
            parent.synchronize_from_originals()
            child.synchronize_from_originals()
            edge.force(parent_clocks=[node.parent.clock for node in nodes], child_clocks=[node.clock for node in nodes])
        elif kind == "feedback_commit":
            if edge.feedback:
                for node in nodes:
                    payload = node.coupler._prepared_feedback
                    if (payload is None or tuple(payload["kinds"]) != self.fields[gid]
                            or payload["dropped_kinds"]):
                        raise BatchStateUnsupported("ordinary feedback plan differs from the packed field inventory; no diagnosed species may be substituted")
                parent.synchronize_from_originals()
                child.synchronize_from_originals()
                edge.feedback_prepare(parent_clocks=[node.parent.clock for node in nodes], child_clocks=[node.clock for node in nodes])
                edge.feedback_commit()
                parent.copy_mutations_to_originals(tuple(_STATE_NAMES.get(name, name) for name in self.fields[gid]))
        elif kind == "feedback_finalize":
            edge.feedback_finalize()
            if edge.feedback:
                parent.copy_mutations_to_originals(("p", "al", "alt"))
        else:
            raise ValueError("resident edge factory received a non-edge operation")
        results = []
        for slot, request in enumerate(group.requests):
            node = nodes[slot]
            with _replay_edge_metadata(node, edge, kind, slot):
                results.append(request.binding.context.run(request.binding.callbacks[kind], *request.args))
        return tuple(results)

    def receipt(self):
        if self._retirement_receipt is not None:
            return deepcopy(self._retirement_receipt)
        return {"qualification": self.qualification_receipt,
            "edges": {gid: edge.receipt() for gid, edge in self.edges.items()},
            "ozone_force": "original_member_transfer_parent_ozone",
            "step": "original_execute_experiment_callback",
            "adaptive_interval_policy": "rebind_common_validated_interval_original_fallback_for_divergence",
            "member_subset_policy": "original_fallback_for_divergent_frontiers",
            "metadata": "original_callbacks_with_completed_edge_numerics"}

    def _note_edge_stream(self):
        if hasattr(self, "xp"):
            stream = self.xp.cuda.get_current_stream()
            key = getattr(stream, "device_id", None), getattr(stream, "ptr", id(stream))
            self._edge_streams[key] = stream

    def completion_wait(self):
        if self._closed:
            return
        self._note_edge_stream()
        error = None
        for stream in self._edge_streams.values():
            try:
                stream.synchronize()
            except BaseException as caught:
                if error is None:
                    error = caught
                elif caught is not error:
                    error.add_note(f"another edge queue completion also failed: {type(caught).__name__}: {caught}")
        if error is not None:
            raise error

    def _retire_handles(self):
        self.callbacks.clear()
        self.edges.clear()
        # The supplied bank dictionary may still belong to the caller.
        self.banks = {}
        self.bindings, self.member_ids = (), ()
        self.nodes.clear()
        self.fields.clear()
        self.owner_keys.clear()
        self._edge_streams.clear()
        self._closed = True

    def close(self):
        if self._closed:
            return
        self._retiring = True
        self.completion_wait()
        if self._retirement_receipt is None:
            self._retirement_receipt = deepcopy(self.receipt())
        self._retire_handles()


class ProductionNestedPackFactory(ResidentNestedEdgeFactory):
    """Join genuine atmospheric STEP callbacks at qualified physics leaves.

    Dycore, staggered geometry, boundary application and health remain the
    original member operations. Native physics joins at its actual entry;
    FORCE and feedback use the separately admitted resident component.
    This hybrid driver does not qualify the complete native RK graph.
    """
    def __init__(self, *, qualification_receipt, reuse_original_workspaces=False, edge_state_only=False):
        super().__init__(qualification_receipt=qualification_receipt, edge_state_only=edge_state_only)
        if type(reuse_original_workspaces) is not bool:
            raise TypeError("original workspace reuse needs an explicit boolean ownership declaration")
        self.reuse_original_workspaces = reuse_original_workspaces
        self.physics, self._wrapped_steps = {}, []
        self._step_pool, self._streams, self._closed = None, (), False
        self._step_groups, self.physics_plans = 0, {}

    def memory_plans(self, bindings, descriptors):
        from woof.ensemble.scheduled_production_physics import production_physics_memory_plan
        from woof.ensemble.packed_production_physics import validate_production_physics_group
        plans = super().memory_plans(bindings, descriptors)
        self.physics_plans = {}
        for gid in descriptors:
            if any(not isinstance(binding.step_bindings, dict) for binding in bindings):
                raise BatchStateUnsupported("original STEP capture did not expose its actual stepper owners; replacing a closure would drop its metadata")
            nodes = tuple(binding.model.node(gid) for binding in bindings)
            drivers = tuple(node.state.physics for node in nodes)
            configs = tuple(node.cfg.run for node in nodes)
            try:
                validate_production_physics_group(drivers, configs)
                plan = production_physics_memory_plan(drivers, configs,
                    reuse_original_workspaces=self.reuse_original_workspaces)
            except (ValueError, TypeError, AttributeError) as error:
                raise BatchStateUnsupported(f"grid {gid} has no qualified production physics entry binding: {error}") from error
            plans[f"physics:{gid}"] = self.physics_plans[gid] = plan
        return plans

    def prepare(self, bindings, banks, *, available_bytes, array_module):
        from woof.core.dycore import step
        from woof.ensemble.scheduled_production_physics import ScheduledProductionPhysics
        try:
            super().prepare(bindings, banks, available_bytes=available_bytes, array_module=array_module)
            remaining = available_bytes - sum(edge.plan.required_bytes(len(bindings)) for edge in self.edges.values())
            self._device = int(array_module.cuda.runtime.getDevice())
            self._streams = tuple(array_module.cuda.Stream(non_blocking=True) for _ in bindings)
            self._step_pool = ThreadPoolExecutor(max_workers=len(bindings), thread_name_prefix="ensemble-original-step")
            for gid, plan in self.physics_plans.items():
                nodes = tuple(binding.model.node(gid) for binding in bindings)
                group = ScheduledProductionPhysics(tuple(node.state.physics for node in nodes),
                    tuple(node.cfg.run for node in nodes), available_bytes=remaining, array_module=array_module,
                    reuse_original_workspaces=self.reuse_original_workspaces)
                remaining -= plan.required_bytes(len(bindings))
                self.physics[gid] = group
                for slot, binding in enumerate(bindings):
                    owners = binding.step_bindings
                    present, original = gid in owners, owners.get(gid, step)
                    # Streamed/spatial steppers retain their original stores
                    # and ranked queues; this is a resident stock-dycore seam.
                    if original is not step:
                        raise BatchStateUnsupported("production packed STEP needs the original resident dycore function; a streamed/ranked stepper keeps its own queue")
                    joined = group.wrap_step(slot, original)
                    def dispatch(state, cfg, *args, group=group, joined=joined, original=original, **kwargs):
                        return joined(state, cfg, *args, **kwargs) if group._armed else original(state, cfg, *args, **kwargs)
                    owners[gid] = dispatch
                    self._wrapped_steps.append((owners, gid, present, original, dispatch))
            self._edge_callbacks = dict(self.callbacks)
            self.callbacks["step"] = self.execute_step
            self._step_groups = 0
            return self
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(f"production factory preparation cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")
            raise

    def admit(self, group):
        if self._closed or self._retiring:
            return ScheduleBatchAdmission(False, "production wave retirement started; its former steppers and physics owners cannot resume")
        if group.kind != "step":
            return super().admit(group)
        if group.member_ids != self.member_ids:
            return ScheduleBatchAdmission(False, "divergent original STEP frontiers keep original per-member physics; this entry barrier owns the complete roster")
        gid = group.requests[0].grid_id
        owner = self.physics.get(gid)
        if owner is None:
            return ScheduleBatchAdmission(False, "this grid has no qualified original atmospheric STEP/physics binding")
        for slot, request in enumerate(group.requests):
            state = request.binding.model.node(gid).state
            if state is not owner.owner.drivers[slot].state or state.physics is not owner.owner.drivers[slot]:
                return ScheduleBatchAdmission(False, "original member state or physics owner was reconstructed; regroup before replacing its entry")
            expected = next(wrapper for owners, grid, _, _, wrapper in self._wrapped_steps
                if owners is request.binding.step_bindings and grid == gid)
            if request.binding.step_bindings.get(gid) is not expected:
                return ScheduleBatchAdmission(False, "original stepper owner changed; retain its new callback rather than a former packed binding")
        return ScheduleBatchAdmission(True, "qualified real member STEP callbacks and physics-entry handshake", "member_owned_rings")

    def execute_step(self, group):
        if self._closed or self._retiring:
            raise RuntimeError("production wave retirement started; STEP cannot use restored or released physics owners")
        owner = self.physics[group.requests[0].grid_id]
        owner.arm()
        def work(slot, request):
            failure, result = None, None
            try:
                with self.xp.cuda.Device(self._device), self._streams[slot]:
                    self._streams[slot].wait_event(ready)
                    if owner._error is not None:
                        raise owner._error
                    result = request.binding.context.run(request.binding.callbacks["step"], *request.args)
            except BaseException as error:
                failure = error
                owner.abort(error)
            finally:
                # Device/stream entry can fail before the ordinary callback.
                # It still owns a roster slot which must retire at the barrier.
                def synchronize():
                    with self.xp.cuda.Device(self._device):
                        self._streams[slot].synchronize()
                for cleanup in (lambda: owner.callback_finished(slot), synchronize):
                    try:
                        cleanup()
                    except BaseException as cleanup_error:
                        owner.abort(cleanup_error)
                        if failure is None:
                            failure = cleanup_error
                        elif cleanup_error is not failure:
                            failure.add_note(f"member STEP cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")
            if failure is not None:
                raise failure
            return result
        futures, partial_launch = {}, False
        results, failure = [None] * len(group.requests), None

        def failed(error, phase):
            nonlocal failure
            try:
                owner.abort(error)
            except BaseException as abort_error:
                error.add_note(f"physics abort also failed: {type(abort_error).__name__}: {abort_error}")
            if failure is None:
                # A member may already have failed while submit was running.
                # The rendezvous records the original failure under its lock.
                failure = owner._error if owner._error is not None else error
            if error is not failure:
                failure.add_note(f"{phase} also failed: {type(error).__name__}: {error}")

        try:
            ready = self.xp.cuda.Event()
            ready.record(self.xp.cuda.get_current_stream())
            for slot, request in enumerate(group.requests):
                futures[self._step_pool.submit(work, slot, request)] = slot
        except BaseException as error:
            partial_launch = True
            failed(error, "member STEP launch")
            # An executor can enqueue a task before a thread-start failure
            # prevents submit from returning its Future. Drain the entire
            # owned pool, not just the Futures which reached this mapping.
            # No forecast continues after a partial launch.
            try:
                self._step_pool.shutdown(wait=True, cancel_futures=True)
            except BaseException as cleanup_error:
                failed(cleanup_error, "partial STEP pool completion")
            for slot in range(len(group.requests)):
                if slot not in owner._callbacks_finished:
                    try:
                        owner.callback_finished(slot)
                    except BaseException as cleanup_error:
                        failed(cleanup_error, "unscheduled member completion")
        # shutdown(cancel_futures=True) can leave queued Futures CANCELLED
        # without a worker's CANCELLED_AND_NOTIFIED transition. as_completed
        # would wait forever for those. All owned workers have joined on the
        # partial-launch path, so result() reads each final/cancelled outcome.
        completed = futures if partial_launch else as_completed(futures)
        for future in completed:
            try:
                results[futures[future]] = future.result()
            except BaseException as error:
                failed(error, "member STEP callback")
        try:
            owner.finish()
        except BaseException as error:
            if failure is None:
                raise
            if error is not failure:
                failure.add_note(f"joined physics completion also failed: {type(error).__name__}: {error}")
        if failure is not None:
            raise failure
        self._step_groups += 1
        return tuple(results)

    def completion_wait(self):
        if self._closed:
            return
        error = None
        self._note_edge_stream()
        queues = self._streams + tuple(self._edge_streams.values())
        for stream in queues:
            try:
                stream.synchronize()
            except BaseException as caught:
                if error is None:
                    error = caught
                elif caught is not error:
                    error.add_note(f"another member queue completion also failed: {type(caught).__name__}: {caught}")
        if error is not None:
            raise error

    def close(self):
        if self._closed:
            return
        self._retiring = True
        error = None
        if self._step_pool is not None:
            try:
                self._step_pool.shutdown(wait=True, cancel_futures=True)
            except BaseException as caught:
                error = caught
        try:
            self.completion_wait()
        except BaseException as caught:
            if error is None:
                error = caught
            elif caught is not error:
                error.add_note(f"member stream completion also failed: {type(caught).__name__}: {caught}")
        if self._retirement_receipt is None:
            try:
                self._retirement_receipt = deepcopy(self.receipt())
            except BaseException as caught:
                if error is None:
                    error = caught
                elif caught is not error:
                    error.add_note(f"component receipt capture also failed: {type(caught).__name__}: {caught}")
        for owners, gid, present, original, wrapper in reversed(self._wrapped_steps):
            if owners.get(gid) is wrapper:
                if present:
                    owners[gid] = original
                else:
                    del owners[gid]
        self._wrapped_steps.clear()
        for group in self.physics.values():
            try:
                group.close()
            except BaseException as caught:
                if error is None:
                    error = caught
                elif caught is not error:
                    error.add_note(f"another physics owner cleanup also failed: {type(caught).__name__}: {caught}")
        if error is not None:
            raise error
        self.physics.clear()
        self.physics_plans.clear()
        self._streams, self._step_pool = (), None
        self._retire_handles()

    def receipt(self):
        if self._retirement_receipt is not None:
            return deepcopy(self._retirement_receipt)
        result = super().receipt()
        result.update(step="original_complete_callbacks_with_native_physics_entry", step_groups=self._step_groups,
            step_streams="persistent_private_member_streams_and_original_cfl_contexts",
            physics={gid: {"entry": group.receipt, "leaves": group.owner.receipt,
                           "plan_bytes": group.plan.required_bytes(len(self.bindings))}
                     for gid, group in self.physics.items()},
            complete_native_rk_graph=False)
        return result


@contextmanager
def _replay_edge_metadata(node, edge, kind, slot):
    """Original callback wrappers observe the already completed component."""
    original = node.coupler
    class CompletedEdge:
        def __getattr__(self, name):
            return getattr(original, name)

        def force(self, candidate):
            if candidate is not node or kind != "force":
                raise ValueError("packed FORCE metadata replay differs from the original child")
            from woof.ingest.lateral_bc import attach_nest_boundaries
            fields = {name: {side: tuple(value[slot] for value in pair) for side, pair in sides.items()}
                      for name, sides in edge.tables.items()}
            run = node.cfg.run
            attach_nest_boundaries(node.state, fields, clock=node.clock,
                spec_bdy_width=run.spec_bdy_width, spec_zone=run.spec_zone, relax_zone=run.relax_zone)
            if original.first_parent_ticks is None:
                original.first_parent_ticks = int(node.parent.clock.ticks)
                original.first_parent_step = int(node.parent.clock.step_count)
            original.last_parent_ticks, original.last_parent_step = int(node.parent.clock.ticks), int(node.parent.clock.step_count)
            original.force_count += 1
            original._last_tables, original._valid = MappingProxyType(fields), True

        def feedback_commit(self, candidate, *, start=False):
            # start: the nest-start transaction (model.build_experiment and the delayed activation call it with
            # start=True since c49be92105); the replay counts it as it counted the plain start call before.
            if candidate is not node or kind != "feedback_commit":
                raise ValueError("packed feedback metadata replay differs from the original child")
            if original.feedback:
                original.feedback_count += 1
                original.last_feedback_ticks = int(node.clock.ticks)

        def feedback_finalize(self, candidate, *, start=False):
            if candidate is not node or kind != "feedback_finalize":
                raise ValueError("packed feedback finalize replay differs from the original child")
            if original.feedback:
                original._prepared_feedback = None
    node.coupler = CompletedEdge()
    try:
        yield
    finally:
        node.coupler = original


__all__ = ["CONTRACT", "PreparedTreeMember", "bootstrap_prepared_tree_members",
           "capture_original_member_schedule", "ScheduledDomainBanks", "PreparedHybridNestedBatch",
           "ResidentNestedEdgeFactory", "ProductionNestedPackFactory"]
