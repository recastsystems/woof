"""Native packing from the ordinary prepared-tree initialization seam.

The prepared runner owns source restore, terrain adaptation, configuration,
physics initialization and the integer clock. This module consumes one such
initialized root. Features without a complete native graph keep the ordinary
runner, including active streaming and independent adaptive clocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np

from woof.core.device_inventory import state_array_shapes
from woof.ensemble.batch_state import (
    BatchedDomainState, BatchStateUnsupported, PreparedHostMember,
    SHARED_STATE_CANDIDATES, _exact_key,
)
from woof.ensemble.suite_capabilities import plan_suite

PREPARED_BATCH_CONTRACT = "gpuwm-ensemble-prepared-native-v1"
_CLOCK_FIELDS = ("ticks", "step_ticks", "tick_den", "run_ticks", "step_count", "dt_fp32", "dtbc_fp32")
_REBUILT_OWNERS = frozenset({
    "physics", "lateral_boundaries", "_scratch", "_scratch_arena", "_host_setup_state", "_phb_host",
    "_lateral_boundary_device", "_dycore_state_workspace",
    "_ensemble_surface_state",
})
_REBUILT_BOUNDARY_SCRATCH = frozenset({"lbc_forcing_tables", "lbc_evaluated_tables"})


def _native_bootstrap_scratch(state):
    """Exclude only registered boundary mirrors rebuilt by member tables."""
    scratch = dict(state._scratch)
    rebuilt = _REBUILT_BOUNDARY_SCRATCH & scratch.keys()
    if rebuilt:
        owner = getattr(state, "_lateral_boundary_device", None)
        registered = set(getattr(owner, "scratch_slots", ()))
        if (owner is None or not rebuilt <= registered
                or getattr(state, "lateral_boundaries", None) is None):
            raise BatchStateUnsupported("boundary mirror scratch has no registered original forcing owner")
        for name in rebuilt:
            del scratch[name]
    return scratch


@dataclass(frozen=True)
class NativePreparedEligibility:
    eligible: bool
    reasons: tuple[str, ...]
    suite: object

    def receipt(self):
        return {"contract": PREPARED_BATCH_CONTRACT, "eligible": self.eligible,
                "fallback_reasons": list(self.reasons), "fallback": "ordinary_member_runner",
                "suite": self.suite.receipt(), "physics_selection": "unchanged",
                "streaming_selection": "unchanged", "clock_selection": "unchanged"}


def _tiles_mode(exp, node):
    domain = exp.root if node is None else node.cfg
    override = getattr(domain, "tiles", None)
    options = override if override is not None else getattr(exp, "tiles", None)
    return getattr(options, "mode", "off")


def _domain_count(inputs):
    """Prepared-tree inputs carry their domain bundles; single-domain inputs carry one domain."""
    domains = getattr(inputs, "domains", None)
    return 1 if domains is None else len(domains)


def native_prepared_eligibility(inputs, node=None, *, members=2, restart=None, keep_member_files=False):
    """Decide native eligibility without changing the ordinary run choices.

    Auto tiling remains configured. After ordinary initialization a resident
    root may batch; an active tile store uses the original tiled runner.
    Before initialization an auto decision has not yet been established.
    """
    exp = inputs.experiment
    cfg = exp.root.run if node is None else node.cfg.run
    suite = plan_suite(cfg, members=members)
    reasons = list(suite.native_fallback_reasons)
    if getattr(cfg, 'dfi_opt', 0):
        reasons.append("DFI initializes each perturbed member through its ordinary coupled startup before packing")
    nudging = getattr(exp, "nudging", None)
    if nudging is not None and any(nudging.for_domain(d.grid_id) is not None for d in exp.domains):
        reasons.append("member-indexed native batch stages do not launch the resident FDDA graphs; each member keeps its ordinary GPU graph and independent forcing clock")
    if any(d.run.sf_surface_physics == 7 and d.run.pxlsm_soil_nudge for d in exp.domains):
        reasons.append("PX soil analyses retain each member's resident CUDA record clock and physics graph")
    if cfg.tracer_opt == 2:
        reasons.append("passive tracers keep the ordinary member runner with independent advection controls and zero-inflow boundaries")
    if cfg.traj_opt == 1:
        reasons.append("native trajectories retain each member's independent particles and point-output clock in the ordinary runner")
    from woof.ensemble.optional_routing import enabled_for
    if enabled_for(inputs):
        reasons.append("NextGen keeps each member's original resident hydro driver")
    if members == 1:
        reasons.append("N=1 keeps the ordinary runner and output writer")
    if len(exp.domains) != 1 or _domain_count(inputs) != 1 or exp.root.parent_id != 0:
        reasons.append("the nested native forecast graph lacks qualified member-indexed parent FORCE, child boundary clocks and feedback transactions")
    if node is not None and (node.parent is not None or node.children or node.coupler is not None):
        reasons.append("this initialized node has parent or child coupling")
    if getattr(inputs, "stream_head", None) is not None:
        reasons.append("head-bound input keeps its original as-posted boundary stream")
    mode = _tiles_mode(exp, node)
    active_store = node is not None and getattr(node.state, "_streamed_domain", None) is not None
    if mode == "on" or active_store or (mode == "auto" and node is None):
        reasons.append("active or unresolved tile streaming keeps its ordinary store and tile stepper")
    if int(getattr(getattr(exp, "devices", None), "count", 1)) > 1:
        reasons.append("a spatially split domain keeps its original ranked device graph")
    if restart is not None:
        reasons.append("restart restoration keeps the ordinary retained physics and clock contract")
    if keep_member_files:
        reasons.append("member histories use the original writer until the full native member writer is qualified")
    from woof.ensemble.radar_output import radar_enabled
    if radar_enabled(inputs):
        reasons.append("simulated radar keeps the original member volume writer and live native radar consumer")
    relocation = getattr(exp, "relocation", None)
    if relocation is not None and getattr(relocation, "enabled", False):
        reasons.append("relocating domains keep their original reconstruction and forcing transactions")
    if cfg.km_opt not in (1, 4) or cfg.khdif > 0 or cfg.kvdif > 0:
        reasons.append("the selected mixing graph has no complete native RK binding")
    if cfg.diff_6th_opt == 1:
        reasons.append("the original moist graph has no non-monotonic sixth-order diffusion implementation")
    if cfg.open_x or cfg.open_y or cfg.nested:
        reasons.append("open or nested boundary graphs keep their original stage producers")
    if cfg.zadvect_implicit or cfg.nwp_diagnostics or cfg.tke_budget:
        reasons.append("implicit transport or optional diagnostics keep their original complete stage graph")
    if getattr(cfg, "hmix_k_diag", False):
        reasons.append("XKMH/XKHH output reads original dycore mixing scratch; its native diagnostic binding is not qualified")
    if cfg.time_step_sound < 2 or cfg.time_step_sound % 2:
        reasons.append("the native moist acoustic graph requires the original even sound-step schedule")
    if cfg.nz < 3 or min(cfg.ny, cfg.nx) < 2 or isinstance(cfg.dx, np.generic) or isinstance(cfg.dy, np.generic):
        reasons.append("this surface-w shape or scalar type uses the ordinary eager arithmetic")
    if cfg.h_sca_adv_order not in (2, 5) or (cfg.h_sca_adv_order == 5 and min(cfg.ny, cfg.nx) < 7):
        reasons.append("the selected horizontal geopotential stencil has no valid native shape")
    from woof.wrf_exact import ENABLED
    if ENABLED:
        reasons.append("strict arithmetic keeps the ordinary complete graph")
    if node is not None:
        state, clock = node.state, node.clock
        if clock.adaptive_state is not None:
            reasons.append("the initialized adaptive controller retains its independent member clocks")
        if clock.step_count != 0 or clock.ticks != clock.spec.start_ticks:
            reasons.append("the native bootstrap seam precedes forecast stepping")
        if clock.spec.start_ticks != 0:
            reasons.append("a delayed activation keeps its original domain epoch")
        if clock.step_ticks / clock.tick_den != float(cfg.dt) or np.float32(cfg.dt).tobytes() != clock.dt_fp32.tobytes():
            reasons.append("the original integer clock and configured native step differ")
        if any(getattr(state, name, None) is not None for name in ("rthften", "rqvften")):
            reasons.append("advective cumulus forcing retains its original member exports")
        if any(not isinstance(getattr(state, name, None), np.float32) for name in ("cf1", "cf2", "cf3")):
            reasons.append("surface-w coefficients use the ordinary eager scalar arithmetic")
        if state.physics is None or getattr(state.physics, "state", None) is not state:
            reasons.append("native packing needs the ordinary initialized driver attached to this state")
        if getattr(state.physics, "hmix_k_diag", None) is not None and not getattr(cfg, "hmix_k_diag", False):
            reasons.append("the initialized mixing diagnostic owner needs its original scratch-to-output handoff")
        radiation = getattr(state.physics, "radiation_callable", None)
        from woof.core.radiation_composition import legacy_radiation_adapter
        legacy = legacy_radiation_adapter(radiation)
        if legacy is None:
            reasons.append("this ordinary driver has no legacy RRTMG owner for native sharing")
        elif getattr(legacy, "_ozone_provider", None) is not None:
            reasons.append("parent-routed ozone keeps its original parent column provider")
        if cfg.specified and getattr(state, "lateral_boundaries", None) is None:
            reasons.append("specified native stages need the ordinary prepared boundary tables")
    # Preserve order while avoiding duplicate reasons from adjacent contracts.
    reasons = tuple(dict.fromkeys(reasons))
    return NativePreparedEligibility(not reasons and suite.uses_native_batch, reasons, suite)


def _host_words(value, name):
    if isinstance(value, np.ndarray):
        result = value.copy(order="C")
    else:
        getter = getattr(value, "get", None)
        if getter is None:
            raise BatchStateUnsupported(f"ordinary bootstrap array {name!r} has no explicit host word transfer")
        result = getter()
    if not isinstance(result, np.ndarray) or not result.flags.c_contiguous:
        raise BatchStateUnsupported(f"ordinary bootstrap array {name!r} did not produce a contiguous host snapshot")
    return result


def prepared_host_from_node(node, *, extra_specs=(), scratch_slots=None):
    """Snapshot one ordinary initialized state and its exact integer clock.

    Reconstructed GPU boundary mirrors and arenas do not become metadata.
    New numerical scalars are retained and typed byte-checked. An unknown
    owner fails native admission instead of being silently discarded.
    """
    cfg, state = node.cfg.run, node.state
    names = state_array_shapes(cfg)
    arrays = {name: _host_words(getattr(state, name), name) for name in names}
    for spec in extra_specs:
        if spec.name in arrays:
            raise ValueError("native RK workspace overlaps the ordinary state inventory")
        arrays[spec.name] = np.zeros(spec.shape, spec.dtype)
    scalars = {name: value for name, value in vars(state).items() if name not in names and name not in _REBUILT_OWNERS}
    # These are generated by a tiled/ranked runner, not transferable state.
    if any(name in scalars for name in ("_streamed_domain", "_streamed_store", "_streamed_scratch")):
        raise BatchStateUnsupported("an active streamed state keeps its original store owner")
    _exact_key(scalars)
    snapshot = {name: getattr(node.clock, name) for name in _CLOCK_FIELDS}
    snapshot["adaptive_state"] = node.clock.adaptive_state
    scratch = {name: _host_words(value, "scratch:" + name) for name, value in _native_bootstrap_scratch(state).items()}
    phb_host = getattr(state, "_phb_host", None)
    return PreparedHostMember(cfg, arrays, scalars, snapshot, scratch=scratch, phb_host=phb_host)


@dataclass
class PreparedNativeMemberBatch:
    batch: object
    physics: object
    tables: object | None
    clock: object
    advance: object
    eligibility: NativePreparedEligibility
    member_ids: tuple[int, ...]
    member_seeds: tuple[int, ...]
    initializer_receipt: object
    snapshot_bytes: int
    member_sources: object = None

    def receipt(self):
        sources = self.member_sources
        return {"contract": PREPARED_BATCH_CONTRACT, "execution": "native_member_batched",
                "eligibility": self.eligibility.receipt(), "member_ids": list(self.member_ids),
                "member_seeds": list(self.member_seeds), "member_initializer": self.initializer_receipt,
                "ordinary_initialized_bootstraps": 1 if sources is None else len(self.member_ids),
                "ordinary_member_forecasts_advanced": 0,
                "member_sources": (None if sources is None else
                    [sources.describe(member) for member in self.member_ids]),
                "bootstrap_host_snapshot_bytes": self.snapshot_bytes,
                "state_plan_bytes": self.batch.plan.required_bytes(self.batch.members),
                "boundary_plan_bytes": 0 if self.tables is None else self.tables.plan.required_bytes(self.batch.members),
                "physics": self.physics.receipt, "clock": {name: getattr(self.clock, name) for name in _CLOCK_FIELDS},
                "bootstrap_handle_policy": "caller releases original model and construction handles"}


@dataclass(frozen=True)
class NativeMemberSource:
    """One roster member's own ordinary bootstrap, snapshotted to host.

    ``prepared`` is its dycore state and scratch (:class:`PreparedHostMember`),
    ``boundaries`` its complete prepared lateral forcing tables, ``physics``
    its initialized driver words (:class:`PhysicsBootstrapSnapshot`). The
    GPU model that produced it has been released; nothing here aliases a
    device array.
    """
    member_id: int
    prepared: PreparedHostMember
    boundaries: object
    physics: object
    receipt: dict

    @property
    def nbytes(self):
        return (sum(int(value.nbytes) for value in self.prepared.arrays.values())
                + sum(int(value.nbytes) for value in (self.prepared.scratch or {}).values())
                + int(self.physics.nbytes))


_SOURCES_ATTRIBUTE = "_ensemble_member_sources"


def _array_module_of(state):
    """NumPy for a host-prepared state (CPU tests), otherwise the CUDA module."""
    if isinstance(getattr(state, "p", None), np.ndarray):
        return np
    import cupy
    return cupy


def _state_scalars(state, names):
    return {name: value for name, value in vars(state).items() if name not in names and name not in _REBUILT_OWNERS}


def _boundary_signature(boundaries):
    """Zones, interval clock, field inventory and time-law topology of one forcing set."""
    if boundaries is None:
        return None
    intervals = tuple(boundaries.intervals)
    fields = tuple(sorted(intervals[0].fields)) if intervals else ()
    sides = ("west", "east", "south", "north")
    return {"zones": (int(boundaries.spec_bdy_width), int(boundaries.spec_zone), int(boundaries.relax_zone),
                      tuple(boundaries.seam_sides)),
            "times": tuple((interval.start_seconds, interval.end_seconds) for interval in intervals),
            "fields": tuple(tuple(sorted(interval.fields)) for interval in intervals),
            "topology": tuple(tuple(getattr(interval.fields[name], side).time_law is not None
                                    for name in fields for side in sides) for interval in intervals)}


class NativeMemberSources:
    """Every roster member's own bootstrap, keyed by global member id.

    The root member stays live on its card as the ordinary initialized
    root; every other member is a host snapshot taken from that member's
    own ordinary bootstrap on the same card. A pack built from this roster
    gives each slot its own state, scratch, forcing tables, land and
    physics words. Shared state (base state, metrics, coordinates), shared
    land surface fields and the owners taken from the root (radiation
    tables and geometry, land tables) must be byte-identical to the root's;
    a member that differs there is named in :meth:`compatibility_reasons`
    and the roster declines to pack before any allocation.
    """

    def __init__(self, root_member_id, snapshots, *, receipt=None):
        self.root_member_id = index(root_member_id)
        snapshots = tuple(snapshots)
        if any(not isinstance(item, NativeMemberSource) for item in snapshots):
            raise TypeError("member sources need NativeMemberSource snapshots")
        self.snapshots = {item.member_id: item for item in snapshots}
        if len(self.snapshots) != len(snapshots) or self.root_member_id in self.snapshots:
            raise ValueError("member sources need one snapshot per non-root member and none for the root")
        self.receipt = dict(receipt or {})
        self._member_owned_surface = None

    def member_owned_surface_fields(self, node):
        """Static surface fields the members disagree on, packed per member."""
        if self._member_owned_surface is None:
            self.compatibility_reasons(node)
        return self._member_owned_surface

    @property
    def member_ids(self):
        return (self.root_member_id,) + tuple(sorted(self.snapshots))

    def covers(self, member_ids):
        return set(member_ids) <= set(self.member_ids)

    def describe(self, member_id):
        if member_id == self.root_member_id:
            return {"member_id": member_id, "source": "live ordinary bootstrap (root)"}
        item = self.snapshots[member_id]
        return {"member_id": member_id, "source": "own ordinary bootstrap, host snapshot",
                "host_bytes": item.nbytes, "bootstrap": dict(item.receipt)}

    def compatibility_reasons(self, node, *, array_module=None):
        """Name every way a snapshot cannot share one packed launch with the root."""
        from woof.ensemble.batch_physics_init import DEMOTABLE_SURFACE_FIELDS, bootstrap_physics_structure
        from woof.ensemble.batch_state import _clock_snapshot
        if array_module is None:
            array_module = _array_module_of(node.state)
        cfg, state = node.cfg.run, node.state
        reasons = []
        member_owned = set()
        names = state_array_shapes(cfg)
        root_cfg = _exact_key(cfg)
        root_scalars = _exact_key(_state_scalars(state, names))
        root_clock = _exact_key(_clock_snapshot({name: getattr(node.clock, name) for name in _CLOCK_FIELDS}
                                                | {"adaptive_state": node.clock.adaptive_state}))
        root_boundaries = _boundary_signature(getattr(state, "lateral_boundaries", None))
        shared_state = tuple(sorted(SHARED_STATE_CANDIDATES & names.keys()))
        root_shared = {name: _host_words(getattr(state, name), name).tobytes() for name in shared_state}
        root_structure, root_shared_physics, root_digests = bootstrap_physics_structure(
            cfg, state.physics, array_module=array_module)
        root_shared_physics = {path: _host_words(value, path).tobytes() for path, value in root_shared_physics.items()}
        for member_id in sorted(self.snapshots):
            item = self.snapshots[member_id]
            label = f"member {member_id}"
            try:
                if _exact_key(item.prepared.cfg) != root_cfg:
                    reasons.append(f"{label} configuration or grid differs from the root; one batch ABI cannot serve both")
                    continue
                clock = _exact_key(_clock_snapshot(item.prepared.clock))
                scalars = _exact_key(item.prepared.scalars)
            except (ValueError, TypeError, BatchStateUnsupported) as error:
                reasons.append(f"{label} configuration, clock or scalar metadata cannot be compared with the root: {error}")
                continue
            if clock != root_clock:
                reasons.append(f"{label} integer clock (step, interval or forcing cadence) differs from the root; "
                               "a common step would change its schedule")
            if scalars != root_scalars:
                reasons.append(f"{label} scalar state metadata differs from the root in value, type or bytes")
            differing = [name for name in shared_state
                         if item.prepared.arrays[name].tobytes(order="C") != root_shared[name]]
            if differing:
                reasons.append(f"{label} shared base-state or metric fields differ from the root: {differing}")
            signature = _boundary_signature(item.boundaries)
            if signature != root_boundaries:
                if signature is None or root_boundaries is None:
                    reasons.append(f"{label} lateral forcing is attached on one of root and member only")
                else:
                    for part, meaning in (("zones", "zones or seam sides"), ("times", "interval clock"),
                                          ("fields", "field inventory"), ("topology", "time-law topology")):
                        if part == "topology" and signature["fields"] != root_boundaries["fields"]:
                            continue  # the topology is per field; a different inventory already says it
                        if signature[part] != root_boundaries[part]:
                            reasons.append(f"{label} lateral forcing {meaning} differ from the root: "
                                           f"member {signature[part]!r}, root {root_boundaries[part]!r}")
            if item.physics.layout != tuple((path, shape, dtype) for path, shape, dtype, _owner in root_structure):
                reasons.append(f"{label} physics driver inventory differs from the root; one packed driver cannot bind both")
            else:
                shared_differs = [path for path, words in root_shared_physics.items()
                                  if item.physics.arrays[path].tobytes() != words]
                # A static surface field another preparation chain sets
                # differently (deep soil temperature, say) is packed per
                # member; the fields the radiation and surface-layer launches
                # bind as one bank cannot be.
                demoted = [path for path in shared_differs if path.rsplit("/", 1)[-1] in DEMOTABLE_SURFACE_FIELDS]
                refused = [path for path in shared_differs if path not in demoted]
                member_owned.update(path.rsplit("/", 1)[-1] for path in demoted)
                if refused:
                    reasons.append(f"{label} shared land surface fields differ from the root and cannot be "
                                   f"member-owned: {refused}")
            owner_differs = [owner for owner, digest in root_digests.items()
                             if item.physics.owner_digests.get(owner) != digest]
            if owner_differs:
                reasons.append(f"{label} bootstrap built different shared physics owners than the root: {owner_differs}")
        self._member_owned_surface = tuple(sorted(member_owned))
        return tuple(reasons)


def snapshot_member_source(node, *, member_id, receipt=None, array_module=None):
    """Snapshot one ordinary initialized node to host as a member source.

    Taken at the same seam the root is packed from: after initialization,
    before any forecast step. Nothing on the node is changed.
    """
    from woof.ensemble.batch_moist_dycore import workspace_specs, required_scratch_slots
    from woof.ensemble.batch_physics_init import bootstrap_physics_snapshot
    if array_module is None:
        array_module = _array_module_of(node.state)
    cfg = node.cfg.run
    if node.clock.step_count != 0 or node.clock.ticks != node.clock.spec.start_ticks:
        raise ValueError("a member source is snapshotted before its first forecast step")
    prepared = prepared_host_from_node(node, extra_specs=workspace_specs(cfg),
                                       scratch_slots=required_scratch_slots(cfg))
    physics = bootstrap_physics_snapshot(cfg, node.state.physics, array_module=array_module)
    return NativeMemberSource(index(member_id), prepared, getattr(node.state, "lateral_boundaries", None),
                              physics, dict(receipt or {}))


def bind_member_sources(node, sources):
    """Attach the roster's member sources to an ordinary initialized root."""
    if not isinstance(sources, NativeMemberSources):
        raise TypeError("a root binds NativeMemberSources")
    setattr(node, _SOURCES_ATTRIBUTE, sources)
    return sources


def member_sources_of(node):
    return getattr(node, _SOURCES_ATTRIBUTE, None)


def prepare_native_member_batch(inputs, node, *, members, member_ids=None, member_seeds=None,
                                member_initializer=None, member_boundaries=None, available_bytes,
                                bootstrap_pool_live_increment_bytes=None, array_module=None,
                                restart=None, keep_member_files=False, member_sources=None):
    """Pack an eligible ordinary root, or return None for its original runner.

    Call ``native_prepared_eligibility`` for the precise fallback receipt.
    ``available_bytes`` is the caller's admitted free-byte budget after the
    ordinary bootstrap, or a callable returning its remaining free budget.
    No member state, source preparation or clock is synthesized here.

    With member sources (``member_sources`` or the roster bound to the root
    through :func:`bind_member_sources`) every slot is built from its own
    member's bootstrap; the root member's slot is packed from the live
    node. Without them every slot is a copy of the root.
    """
    decision = native_prepared_eligibility(inputs, node, members=members, restart=restart,
                                           keep_member_files=keep_member_files)
    if not decision.eligible:
        return None
    if array_module is None:
        import cupy as array_module
    ids = tuple(range(members)) if member_ids is None else tuple(member_ids)
    if len(ids) != members or len(set(ids)) != members or any(isinstance(value, (bool, np.bool_)) for value in ids):
        raise ValueError("native member ids must be a unique roster of the requested size")
    ids = tuple(index(value) for value in ids)
    if any(value < 0 for value in ids):
        raise ValueError("native member ids cannot be negative")
    seeds = () if member_seeds is None else tuple(member_seeds)
    if member_initializer is not None and len(seeds) != members:
        raise ValueError("a native initializer needs one explicit seed per member")
    if seeds and (len(seeds) != members or any(isinstance(value, (bool, np.bool_)) for value in seeds)):
        raise ValueError("native seeds must match the member roster")
    seeds = tuple(index(value) for value in seeds)
    if any(not 0 <= value < 2**64 for value in seeds):
        raise ValueError("native member seeds must fit uint64")
    sources = member_sources_of(node) if member_sources is None else member_sources
    if sources is not None:
        if not isinstance(sources, NativeMemberSources):
            raise TypeError("member sources must be NativeMemberSources")
        if member_boundaries is not None:
            raise ValueError("member sources carry their own boundary tables")
        if not sources.covers(ids):
            raise ValueError(f"native member ids {ids} are not all bootstrapped member sources {sources.member_ids}")
        reasons = sources.compatibility_reasons(node, array_module=array_module)
        if reasons:
            raise BatchStateUnsupported("; ".join(reasons))
    from woof.ensemble.batch_moist_dycore import workspace_specs, required_scratch_slots, prepare_moist_step
    from woof.ensemble.batch_physics_init import initialize_member_physics_from_bootstrap
    cfg = node.cfg.run
    extras, scratch_slots = workspace_specs(cfg), required_scratch_slots(cfg)
    prepared = prepared_host_from_node(node, extra_specs=extras, scratch_slots=scratch_slots)
    if sources is None:
        prepared_members = (prepared,) * members
        member_bootstraps = None
        initialization_source = "one ordinary prepared-tree bootstrap before member perturbations"
    else:
        prepared_members = tuple(prepared if member == sources.root_member_id else sources.snapshots[member].prepared
                                 for member in ids)
        member_bootstraps = tuple(None if member == sources.root_member_id else sources.snapshots[member].physics
                                  for member in ids)
        initialization_source = "each member's own ordinary bootstrap; the root member live, the others from host snapshots"
    snapshot_bytes = sum(sum(value.nbytes for value in item.arrays.values())
                         + sum(value.nbytes for value in (item.scratch or {}).values())
                         for item in {id(item): item for item in prepared_members}.values())
    remaining = None if callable(available_bytes) else index(available_bytes)
    def budget():
        return index(available_bytes()) if callable(available_bytes) else remaining
    batch = BatchedDomainState.from_prepared(prepared_members, array_module=array_module,
        available_bytes=budget(), shared_fields=tuple(sorted(SHARED_STATE_CANDIDATES & state_array_shapes(cfg).keys())),
        extra_specs=extras, scratch_slots=scratch_slots)
    if remaining is not None:
        remaining -= batch.plan.required_bytes(members)
    del prepared, prepared_members
    tables = None
    if cfg.specified:
        from woof.ensemble.batch_boundaries import MemberBoundaryTables
        if sources is not None:
            boundaries = tuple(node.state.lateral_boundaries if member == sources.root_member_id
                               else sources.snapshots[member].boundaries for member in ids)
        else:
            boundaries = ((node.state.lateral_boundaries,) * members if member_boundaries is None else tuple(member_boundaries))
        if len(boundaries) != members:
            raise ValueError("native boundary roster differs from the requested members")
        tables = MemberBoundaryTables.from_prepared(boundaries, cfg, array_module=array_module, available_bytes=budget())
        if remaining is not None:
            remaining -= tables.plan.required_bytes(members)
    physics = initialize_member_physics_from_bootstrap(batch, node.state, node.state.physics,
        available_bytes=budget(), bootstrap_pool_live_increment_bytes=bootstrap_pool_live_increment_bytes,
        initialization_source=initialization_source, array_module=array_module,
        member_bootstraps=member_bootstraps,
        member_owned_surface=() if sources is None else sources.member_owned_surface_fields(node))
    initializer_receipt = None
    if member_initializer is not None:
        initializer_receipt = member_initializer(state=batch, cfg=batch.cfg, member_indices=ids,
            seeds=seeds, phase="after_physics_before_step")
    advance = prepare_moist_step(batch, physics_adapter=physics, tables=tables, boundary_clock=node.clock)
    return PreparedNativeMemberBatch(batch, physics, tables, node.clock, advance, decision,
                                      ids, seeds, initializer_receipt, snapshot_bytes, sources)


def native_memory_model_from_node(inputs, node, *, runtime_reservation, allocator_margin,
                                  external_components, bootstrap_live_bytes,
                                  free_sample_timing="before_bootstrap", sampled_free_bytes=None,
                                  resident_threads=0, boundary_plan=None):
    """Price the native graph before allocating a batch from its actual input.

    State, scratch, physics and forcing use their construction declarations.
    The existing legacy engine estimator prices call peaks with member-aligned
    chunks. External components supply products, health, counter banks and
    bounded output replay. Runtime/context/local-memory reservation and pool
    margin are explicit caller evidence, rather than hidden fitted factors.

    ``before_bootstrap`` budgets charge the measured bootstrap once. For an
    ``after_bootstrap`` free sample those live bytes already reduced the
    budget and are reported without another charge. No plan retains scalar
    state or driver array references after it is built.
    """
    from woof.ensemble.admission import EnsembleMemoryModel, MemoryComponent, AllocatorMargin
    from woof.ensemble.batch_storage import BatchMemoryPlan, BatchArraySpec
    from woof.ensemble.batch_state import state_array_specs
    from woof.core.preflight import scratch_slot_registry
    from woof.ensemble.batch_moist_dycore import workspace_specs, required_scratch_slots
    from woof.ensemble.batch_physics_init import native_bootstrap_allocation_plans
    from woof.ensemble.batch_physics import (
        column_memory_plan, packed_physics_atmosphere_fields, physics_coupling_memory_plan,
        default_column_extra_plan, ysu_output_plan, ring_memory_plan,
    )
    decision = native_prepared_eligibility(inputs, node, members=2)
    if not decision.eligible:
        return None
    if not isinstance(runtime_reservation, MemoryComponent) or not runtime_reservation.evidence:
        raise ValueError("native runtime reservation needs a named component and calibration evidence")
    if not isinstance(allocator_margin, AllocatorMargin) or not allocator_margin.evidence:
        raise ValueError("native allocator margin needs explicit calibration evidence")
    external_components = tuple(external_components)
    if not external_components or any(not isinstance(component, MemoryComponent) for component in external_components):
        raise ValueError("native memory model needs the product, diagnostic and bounded output components")
    if free_sample_timing not in ("before_bootstrap", "after_bootstrap"):
        raise ValueError("native free sample must name its bootstrap timing point")
    bootstrap_live_bytes = index(bootstrap_live_bytes)
    if bootstrap_live_bytes < 0:
        raise ValueError("bootstrap live bytes cannot be negative")
    cfg, state = node.cfg.run, node.state
    shared = tuple(sorted(SHARED_STATE_CANDIDATES & state_array_shapes(cfg).keys()))
    specs = state_array_specs(cfg, shared_fields=shared) + tuple(workspace_specs(cfg))
    scratch = dict(required_scratch_slots(cfg))
    scratch.update({name: value.dtype for name, value in _native_bootstrap_scratch(state).items()})
    registry = scratch_slot_registry(cfg)
    if set(scratch) - registry.keys():
        raise BatchStateUnsupported("ordinary bootstrap scratch has an unclassified native allocation")
    specs += tuple(BatchArraySpec("scratch:" + name, registry[name], "member", dtype)
                   for name, dtype in scratch.items())
    state_plan = BatchMemoryPlan(specs, reserved_bytes=0)
    sources = member_sources_of(node)
    bootstrap = native_bootstrap_allocation_plans(cfg, state.physics, state_specs=specs,
        member_owned_surface=() if sources is None else sources.member_owned_surface_fields(node))
    sizes = dict(nz=cfg.nz, ny=cfg.ny, nx=cfg.nx)
    components = [MemoryComponent("native_state", "state", plan=state_plan,
                                   evidence="state_array_specs and original scratch registry"),
        MemoryComponent("native_physics_bank", "physics", plan=bootstrap.bank_plan,
                        evidence="native bootstrap array aliases and column declarations"),
        MemoryComponent("native_surface_shared", "shared", plan=bootstrap.shared_plan,
                        evidence="audited immutable bootstrap surface fields"),
        MemoryComponent("native_validation_status", "physics", plan=bootstrap.status_plan,
                        evidence="original one-word microphysics validation control"),
        MemoryComponent("native_atmosphere", "physics", plan=column_memory_plan(packed_physics_atmosphere_fields(**sizes)),
                        evidence="PreparedPackedPhysicsAtmosphere construction fields"),
        MemoryComponent("native_pbl_coupling", "physics", plan=physics_coupling_memory_plan(**sizes),
                        evidence="PreparedPhysicsCoupling construction"),
        MemoryComponent("native_radiation_coupling", "physics", plan=physics_coupling_memory_plan(**sizes),
                        evidence="separate retained radiation coupling owner"),
        MemoryComponent("native_driver_extra", "physics", plan=default_column_extra_plan(**sizes),
                        evidence="default native column binding construction"),
        MemoryComponent("native_ysu", "physics", plan_for_members=lambda members: ysu_output_plan(**sizes, members=members),
                        evidence="YSU retained output and native full launch workspace"),
        MemoryComponent("native_radiation_coszen", "radiation", plan=BatchMemoryPlan((
            BatchArraySpec("radiation:coszen", (cfg.ny, cfg.nx), "member"),), reserved_bytes=0),
                        evidence="prepare_legacy_member_radiation output bank"),
        runtime_reservation,
        MemoryComponent("ordinary_bootstrap_live", "bootstrap",
                        fixed_bytes=bootstrap_live_bytes if free_sample_timing == "before_bootstrap" else 0,
                        basis="measured", evidence=f"{free_sample_timing} free sample; bootstrap live {bootstrap_live_bytes} bytes"),
    ]
    if cfg.cu_physics == 99:
        from woof.ensemble.batch_kf import kf_old_driver_reservation
        components += [
            MemoryComponent("native_cumulus_coupling", "physics", plan=physics_coupling_memory_plan(**sizes),
                            evidence="CU99 separate retained member-column coupling owner"),
            MemoryComponent("native_cu99_call_peak", "physics",
                plan_for_members=lambda members: BatchMemoryPlan((BatchArraySpec(
                    "cumulus:due_call_envelope", (kf_old_driver_reservation(**sizes,
                        members=members, edrates=cfg.kf_edrates),), "shared", "uint8"),), reserved_bytes=0),
                basis="envelope", evidence="native CU99 outputs, bounded workspace and stock hold/health temporary upper bound"),
        ]
    if cfg.specified:
        if boundary_plan is None:
            from woof.ensemble.batch_boundaries import MemberBoundaryTables
            # A bound roster prices every member's own tables; one root
            # prices a roster of copies. Both declare per-member ownership.
            boundaries = (state.lateral_boundaries,) + (() if sources is None else
                tuple(sources.snapshots[member].boundaries for member in sorted(sources.snapshots)))
            boundary_plan = MemberBoundaryTables.memory_plan(boundaries, cfg)
        components.append(MemoryComponent("native_boundaries", "boundary", plan=boundary_plan,
                                          evidence="MemberBoundaryTables allocation declarations"))
    if (cfg.specified or cfg.nested) and cfg.spec_zone:
        from woof.core.physics_inventory import ring_guard_row
        row = ring_guard_row(cfg.mp_physics)
        fields = {name: spec for name, spec in ((spec.name, spec) for spec in specs)}
        levels = {name: fields[name].shape[0] for name in row["state_fields"] if name in bootstrap.model_names}
        slots = {field.name: field.shape for field in bootstrap.scratch_fields}
        levels.update({name: 1 for name in row["surface_slots"] if name in slots})
        if "refl_10cm" in slots:
            levels["refl_10cm"] = 1
        from woof.ensemble.batch_physics import THOMPSON_CLOUD_BOOKING_SLOTS
        levels.update({name: 1 for name in THOMPSON_CLOUD_BOOKING_SLOTS if name in slots})
        levels["h_diabatic"] = cfg.nz
        components.append(MemoryComponent("native_registered_ring" if cfg.mp_physics == 55 else "native_thompson_ring", "physics",
            plan=ring_memory_plan(levels, ny=cfg.ny, nx=cfg.nx, width=cfg.spec_zone, zero_fields=("h_diabatic",)),
            evidence="Configured microphysics ring state and surface slot declarations"))
    from woof.core.radiation_composition import legacy_radiation_adapter
    legacy = legacy_radiation_adapter(state.physics.radiation_callable)
    from woof.core import rrtmg_legacy as radiation
    coefficients, p_top, column_chunk = legacy._C, state.p_top, legacy.column_chunk
    cells, nz, o3input = cfg.ny * cfg.nx, cfg.nz, int(legacy.o3input)
    lw_chunk = radiation._lw.batch_column_chunk(radiation._lw.NGPTLW, radiation._lw.LW_BATCH_COLUMN_CHUNK_CEILING,
                                               resident_threads=resident_threads)
    sw_chunk = radiation._sw.sw_batch_column_chunk(nz + 1, resident_threads=resident_threads)
    def radiation_envelope(members):
        lw = max(members, int(column_chunk or lw_chunk) // members * members)
        sw = max(members, int(column_chunk or sw_chunk) // members * members)
        common = dict(ncol=members * cells, nz=nz, p_top=p_top, lw_coefficients=coefficients,
                      resident_threads=resident_threads, o3input=0)
        peak = max(radiation.legacy_radiation_vram_bytes(**common, column_chunk=lw, longwave=True, shortwave=False),
                   radiation.legacy_radiation_vram_bytes(**common, column_chunk=sw, longwave=False, shortwave=True))
        return BatchMemoryPlan((BatchArraySpec("radiation:transient_call_envelope", (peak,), "shared", "uint8"),), reserved_bytes=0)
    components.append(MemoryComponent("native_radiation_call_peak", "radiation", plan_for_members=radiation_envelope,
        basis="envelope", evidence="original legacy_radiation_vram_bytes with member-aligned LW/SW chunks and all-day upper bound"))
    if o3input == 2 and getattr(legacy, "_o33d_grid", None) is None:
        components.append(MemoryComponent("native_radiation_shared_ozone", "shared", plan=BatchMemoryPlan((
            BatchArraySpec("radiation:shared_ozone", (nz, cfg.ny, cfg.nx), "shared"),), reserved_bytes=0),
            evidence="member adapter retains one geometry-sized ozone grid; preexisting grids belong to bootstrap live bytes"))
    components.extend(external_components)
    model = EnsembleMemoryModel(tuple(components), allocator_margin=allocator_margin,
        inventory_id=PREPARED_BATCH_CONTRACT + ":" + free_sample_timing)
    # Sampling metadata does not influence allocation arithmetic.
    model_sampling = {"free_sample_timing": free_sample_timing, "sampled_free_bytes": sampled_free_bytes,
                      "bootstrap_live_bytes": bootstrap_live_bytes,
                      "runtime_reservation_evidence": runtime_reservation.evidence}
    return model, model_sampling


__all__ = ["PREPARED_BATCH_CONTRACT", "NativePreparedEligibility", "PreparedNativeMemberBatch",
           "NativeMemberSource", "NativeMemberSources", "snapshot_member_source",
           "bind_member_sources", "member_sources_of",
           "native_prepared_eligibility", "prepared_host_from_node", "prepare_native_member_batch",
           "native_memory_model_from_node"]
