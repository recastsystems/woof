"""Domain-tree construction and the Phase-5 flat-schedule executor."""

from __future__ import annotations

from woof.core.physics_inventory import REFL_10CM_MICROPHYSICS

from collections.abc import Iterator, Mapping
from contextlib import nullcontext
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import math
import numpy as np
from pathlib import Path
import sys
import time
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from woof.core.clock import DomainClock, Schedule
from woof.core.restart_handler_traits import durable_waiter, orders_history_itself
from woof.config import (SURFACE_RADIATION_POLICY_REQUIRED,
                          radiation_scheme_ids)
from woof.core.state import DomainState
from woof.experiment import DomainConfig
from woof.static.lambert import LambertGrid


RestartPhase = Literal["PERIOD_BEGIN"]
PERIOD_BEGIN: RestartPhase = "PERIOD_BEGIN"
"""The only legal multi-domain checkpoint phase.

Task 14 serializes this marker explicitly; it is never inferred from elapsed
time.  Restore invalidates child boundary tables, and the ordinary parent STEP
then FORCE must rebuild them before any child STEP.
"""


@dataclass
class FeedbackScratch:
    """Placeholder for child-owned restricted feedback data.

    Phase 5 allocates no feedback payload because ``feedback=0`` is enforced
    at experiment load.  Phase 5b replaces ``payload`` with the typed scratch
    views produced by ``feedback_prepare`` without changing the structural
    coupler protocol or schedule table.
    """

    payload: object | None = None


@runtime_checkable
class NestCoupler(Protocol):
    """Node-facing parent/child coupling surface.

    Concrete couplers satisfy this protocol structurally; they do not need to
    import or inherit it.  The parent-read-only rule applies to ``force``
    ONLY.  Phase 5b's feedback transaction makes its parent-mutation boundary
    explicit; all three feedback phases remain dormant while Phase 5 enforces
    one-way coupling at experiment load.

    WRF initialization is part of that activation contract, not an optional
    post-hoc sweep: ``med_nest_initial`` saves ``parent%ht_coarse`` and calls
    ``med_nest_feedback`` (share/mediation_integrate.F:774-777); when
    ``feedback != 0`` the mediation layer mutates the parent
    (:1035-1037), after which initialization re-runs ``start_domain`` on the
    parent (:787-838).  A Phase-5b builder must reproduce that ordered
    initialization transaction.  A bottom-up feedback sweep after all
    domains have been initialized is not equivalent.
    """

    def force(self, node: DomainNode) -> None:
        """Read the parent without mutation and refresh/mutate the child."""
        ...

    def feedback_prepare(self, node: DomainNode,
                         out: FeedbackScratch) -> None:
        """Read the child only and write restricted data into scratch."""
        ...

    def feedback_commit(self, node: DomainNode, *, start: bool = False) -> None:
        """PARENT MUTATED: masked merge plus ``ht_coarse`` bookkeeping.

        ``start`` marks the nest-start transaction (med_nest_initial)."""
        ...

    def feedback_finalize(self, node: DomainNode, *, start: bool = False) -> None:
        """PARENT MUTATED: restore parent halo/diagnostic validity."""
        ...


@dataclass
class DomainNode:
    """One configured domain and its parent-first tree links."""

    cfg: DomainConfig
    grid: LambertGrid
    state: DomainState
    clock: DomainClock
    parent: DomainNode | None
    children: list[DomainNode]
    coupler: NestCoupler | None

    def __post_init__(self) -> None:
        # Lifecycle state is deliberately not a dataclass field: the public
        # structural contract remains the configured tree, while delayed
        # activation and restart restore own this runtime-only marker.
        self._started = True
        cfg_id = self.cfg.grid_id
        if self.cfg.run.grid_id != cfg_id:
            raise ValueError(
                f"DomainNode grid_id mismatch: cfg.grid_id={cfg_id}, "
                f"cfg.run.grid_id={self.cfg.run.grid_id}")
        if self.clock.spec.grid_id != cfg_id:
            raise ValueError(
                f"DomainNode grid_id mismatch: cfg.grid_id={cfg_id}, "
                f"clock.spec.grid_id={self.clock.spec.grid_id}")

        if self.cfg.parent_id == 0:
            if self.parent is not None:
                raise ValueError(
                    f"root DomainNode grid_id={cfg_id} must have parent=None")
            if self.coupler is not None:
                raise ValueError(
                    f"root DomainNode grid_id={cfg_id} must have coupler=None")
            return

        if self.parent is None:
            raise ValueError(
                f"child DomainNode grid_id={cfg_id} must receive its "
                "already-constructed parent (parent-before-child)")
        if self.parent.cfg.grid_id != self.cfg.parent_id:
            raise ValueError(
                f"DomainNode parent_id mismatch for grid_id={cfg_id}: "
                f"cfg.parent_id={self.cfg.parent_id}, "
                f"parent.cfg.grid_id={self.parent.cfg.grid_id}")


@dataclass
class ExperimentState:
    """Loop-free container populated by the full Task-14 builder.

    The future executor may checkpoint this tree only at ``PERIOD_BEGIN``
    with equal domain ticks, no pending FORCE/FEEDBACK/D2H/mutation, and all
    prior feedback committed.  A restored tree starts with invalid child
    boundary tables; eager rebuild from parent(t) is forbidden, and a child
    STEP before the normal parent STEP + FORCE rebuild is an assertion error.
    """

    root: DomainNode
    nodes_by_grid_id: Mapping[int, DomainNode]
    schedule: Schedule
    memory_ledger: object | None
    experiment_fingerprint: str

    #: The route's build-time context (experiment, case data, forcing
    #: times, radiation workspace), set by ``build_experiment``.  Left
    #: DELIBERATELY unannotated so it stays a plain class attribute and
    #: not a dataclass field: hand-assembled trees (the idealized and
    #: verify paths) construct this positionally and never set it, and a
    #: real default is what lets the executor read it as an ordinary
    #: attribute -- the clock-module audit
    #: (tests/test_clock.py::test_no_float_elapsed_accumulation_audit)
    #: bans getattr reflection here.
    _activation_context = None

    #: The experiment this tree was CONFIGURED from, published by
    #: whichever route built the tree (:func:`publish_declared_experiment`).
    #: Deliberately not a key of ``_activation_context`` above: that
    #: bundle is one route's ingest context and a tree assembled from a
    #: prepared cache has none of it, so a checkpoint that reached for
    #: the declaration there could only be written by the route that
    #: happened to carry the ingest.  Same unannotated class attribute
    #: for the same reason.
    _declared_experiment = None

    #: Resolved at executor entry, then carried by run receipts.
    _pool_trim_policy = None

    def node(self, grid_id: int) -> DomainNode:
        """Return one domain node by its configured grid identifier."""
        return self.nodes_by_grid_id[grid_id]

    def walk_parent_first(self) -> Iterator[DomainNode]:
        """Walk the tree depth-first with every parent before its children."""
        stack = [self.root]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(reversed(current.children))


@dataclass
class SharedRRTMGPChunkWorkspace:
    """One real byte allocation shared by all sequential RRTMGP adapters.

    Each solver phase lays its audited live set over the same backing.  Common
    optics entries keep identical offsets into the following RTE phase, while
    dead mask storage is reused by sources/fluxes.  ``phase`` returns only the
    active column prefix; kernels or explicit fills overwrite every returned
    element before its first read (``RRTMGP_WORKSPACE_LIFETIME_AUDIT``).

    The hidden injection fields keep this allocation CPU-testable with NumPy;
    production passes neither and therefore performs exactly one CuPy byte
    allocation of the preflight ledger's phase maximum.
    """

    nz: int
    column_chunk: int
    p_top: float = 5000.0
    _array_module: object | None = field(
        default=None, repr=False, compare=False)
    _phase_layouts_input: object | None = field(
        default=None, repr=False, compare=False)
    _storage: object = field(init=False, repr=False, compare=False)
    _phase_layouts: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.nz < 1 or self.column_chunk < 1:
            raise ValueError("RRTMGP workspace dimensions must be positive")
        if not math.isfinite(self.p_top) or self.p_top < 0.0:
            raise ValueError(
                "RRTMGP workspace p_top must be finite and nonnegative")
        xp = self._array_module
        if xp is None:
            import cupy as xp
        layouts = self._phase_layouts_input
        if layouts is None:
            from woof.core.preflight import rrtmgp_workspace_phases
            layouts = rrtmgp_workspace_phases(
                self.nz, self.column_chunk, self.p_top)
        layouts = {str(phase): dict(items)
                   for phase, items in dict(layouts).items()}
        if not layouts:
            raise ValueError("RRTMGP workspace has no solver phases")
        totals = {}
        for phase, items in layouts.items():
            offset = 0
            for name, (shape, itemsize) in items.items():
                shape = tuple(int(extent) for extent in shape)
                itemsize = int(itemsize)
                if (not shape or shape[0] != self.column_chunk
                        or any(extent < 1 for extent in shape)
                        or itemsize not in (1, 4)):
                    raise ValueError(
                        f"invalid RRTMGP workspace slot {phase}/{name}: "
                        f"{shape}, itemsize={itemsize}")
                if offset % itemsize:
                    raise ValueError(
                        f"unaligned RRTMGP workspace slot {phase}/{name}")
                offset = offset + math.prod(shape) * itemsize
            totals[phase] = offset
        storage = xp.empty((max(totals.values()),), dtype=xp.uint8)
        self._storage = storage
        self._phase_layouts = MappingProxyType(layouts)

    @property
    def nbytes(self) -> int:
        return int(self._storage.nbytes)

    @property
    def storage(self):
        """The single backing allocation (identity/debug inspection only)."""
        return self._storage

    def phase(self, name: str, ncol: int) -> Mapping[str, object]:
        """Return active-prefix views for one audited solver live set."""
        if ncol < 1 or ncol > self.column_chunk:
            raise ValueError(
                f"RRTMGP chunk columns {ncol} outside 1..{self.column_chunk}")
        try:
            layout = self._phase_layouts[name]
        except KeyError as exc:
            raise KeyError(f"unknown RRTMGP workspace phase {name!r}") from exc
        views = {}
        offset = 0
        for slot, (raw_shape, itemsize) in layout.items():
            shape = tuple(int(extent) for extent in raw_shape)
            count = math.prod(shape)
            size = count * int(itemsize)
            dtype = np.bool_ if int(itemsize) == 1 else np.float32
            full = self._storage[offset:offset + size].view(dtype).reshape(shape)
            views[slot] = full[:ncol]
            offset = offset + size
        return MappingProxyType(views)


@dataclass(frozen=True)
class ModelMemoryLedger:
    """Builder record tying the runtime arena/workspace to the estimator."""

    estimate: object
    shared_scratch_arena_bytes: int
    shared_dycore_state_workspace_bytes: int
    radiation_workspace: SharedRRTMGPChunkWorkspace | None


@dataclass
class ModelRuntimeStatus:
    """Transaction state used by the tree checkpoint phase contract."""

    schedule_cursor: str = PERIOD_BEGIN
    pending_force: int = 0
    pending_feedback: int = 0
    pending_d2h: int = 0
    mutation_in_progress: bool = False
    prior_feedback_committed: bool = True


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, Path)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


#: The experiment fields a restart is permitted to differ in.  This IS
#: the published ``woof run --restart`` contract ("only the forecast
#: length / output and restart cadence may differ"), held in one place so
#: every route that builds a restart identity excludes the same set.
RESTART_TOLERATED_EXPERIMENT_FIELDS = (
    "run_seconds", "restart_interval_s", "acknowledgements",
    # Relocation ADMISSIBILITY BOUNDS, not a trajectory input.  Nothing in
    # RelocationConfig reaches a computed value: it says which child may
    # move and how far, and a move is an explicit event recorded in its
    # own segment chain, never something these bounds derive.  Excluding
    # it is also the conservative reading of the restart contract rather
    # than a relaxation of it -- the field is new, so binding it would
    # move the fingerprint of every experiment that never mentions
    # relocation and refuse every checkpoint written before it existed.
    # Held byte-inert by tests/test_nest_relocation.py.
    "relocation",
    # [tiles] is an EXECUTION choice and the only thing it claims is
    # that it changes nothing.  A domain integrated resident and the same
    # domain streamed tile-by-tile from pinned host RAM produce the same
    # bytes -- proven carrier by carrier at every physics rung, and proven
    # again across a checkpoint in four legs (streamed -> file -> streamed,
    # streamed -> file -> MONOLITHIC through the unmodified restore path,
    # monolithic -> file -> streamed, all bit-exact).  Binding the mode
    # would therefore refuse exactly the operation it exists for: a
    # forecast that outgrew its card resuming on the card it outgrew.
    # See woof.core.streaming.identity_payload_entry.
    "tiles", "devices",
    # [output] selects which variables reach the HISTORY tape
    # (woof.io.history_selection).  It changes no number the model
    # computes and touches no checkpoint: checkpoints are separate files
    # written from model state by woof.io.restart, never from the
    # history frame.  Binding it would refuse the operation the surface
    # exists for -- a run that filled its disk resuming with a trimmed
    # tape -- so a trimmed run resumes a full run's checkpoints and back
    # again.  Same law as "tiles" above.
    "output", "simulated_radar")
#: The history window (history_begin_s / history_end_s) is output-only
#: like the cadence beside it: it decides which instants reach the history
#: tape and changes no number the model integrates, so a resume may move
#: it -- and every fingerprint written before the fields existed keeps its
#: value because they leave the payload unconditionally.
RESTART_TOLERATED_DOMAIN_FIELDS = (
    "history_interval_s", "history_begin_s", "history_end_s")
RESTART_TOLERATED_RUN_FIELDS = (
    "run_seconds", "output_interval_s", "history_interval2_s", "history_interval_change_s", "restart_interval_s",
    "surface_energy_diag",
    # Correct mode WB1 (woof/correct_mode.py OUTPUT_ONLY_KEYS): books the
    # settled-cloud total CLOUDNC and changes no prognostic word
    # (tests/test_correct_mode_water_budget.py, every word bit-identical).
    "thompson_cloud_surface_booking")

#: ``mp_physics`` -> the RunConfig fields ONLY that scheme reads.  A domain
#: running some other scheme drops them from its identity entirely rather
#: than carrying them at their defaults -- the absent-stays-absent
#: convention ``perturbation`` and the per-domain ``spawn`` declaration
#: already use, applied to scheme-scoped knobs.
#:
#: This is not a relaxation of the restart contract, and the test that
#: matters is whether any code path can read the field: under a different
#: ``mp_physics`` no launcher, adapter, allocator or diagnostic touches
#: them (``woof/core/wdm6.py`` is the only reader of ``wdm6_hail_opt``,
#: ``woof/core/state.py``'s CCN fill the only reader of
#: ``wdm6_ccn_conc``, ``woof/core/nssl2_contract.py`` the only reader of
#: the five NSSL selectors, and all of them sit behind their scheme's
#: dispatch row).  ``validate_run_config`` REFUSES a non-default value on
#: any other scheme for every field listed here, so off-scheme these
#: fields can hold only their defaults and dropping them discards no
#: information.  A field no path reads cannot move a trajectory, so
#: binding it would buy no safety and would cost the thing that IS
#: essential: every experiment written before these schemes existed
#: keeps its exact fingerprint, and every checkpoint those runs wrote
#: stays resumable.  Under their own scheme the fields bind, value for
#: value, because there they ARE the trajectory.
#:
#: The NSSL row was added at the 1.9 assembly.  Its five selectors landed
#: unscoped and moved the pre-NSSL fingerprint anchor
#: (``tests/test_water_overlay.py``), which is the exact regression this
#: table exists to prevent; the anchor is back at its lane-base value with
#: the row in place.  Every scheme-scoped knob goes here.
#: The mp=28 row was added at the 2.5.8 integration.  Its two aerosol-source
#: selectors (lane/wif-default) landed unscoped, exactly as the NSSL five
#: had, and moved BOTH frozen anchors in ``tests/test_water_overlay.py`` --
#: measured by payload diff against the pristine base f27fc897a: exactly
#: four added keys, these two per domain, no other change.  Unscoped they
#: would have made every 2.5.7 checkpoint refuse to resume under 2.5.8 for
#: fields only the aerosol-aware Thompson path reads (every reader in
#: woof/ingest/real.py sits behind ``cfg.mp_physics == 28``).
SCHEME_SCOPED_RUN_FIELDS: dict[int, tuple[str, ...]] = {
    16: ("wdm6_hail_opt", "wdm6_ccn_conc"),
    18: ("nssl_2moment_on", "nssl_hail_on", "nssl_ccn_on",
         "nssl_density_on", "nssl_3moment"),
    28: ("wif_climatology_path", "mp28_aerosol_source"),
    # The mp=50 row was added with the P3 CUDA port.  p3_backend selects
    # WHICH implementation integrates the column, and the three arms are
    # byte-identical on every p3-fortref fixture where they have been
    # compared, so it is not even a trajectory choice -- but it is still a
    # RunConfig field, and an unscoped RunConfig field moves EVERY
    # experiment fingerprint whether or not it changes a number.  It is
    # scoped here, and woof/config.py refuses a non-default value under
    # any other scheme, which is what makes the scoping lossless.
    50: ("p3_backend",),
}

#: Cumulus-scoped knobs: each RunConfig field and the ``cu_physics`` values
#: that read it.  Same convention as :data:`SCHEME_SCOPED_RUN_FIELDS`, keyed
#: by the cumulus selector instead: a domain whose cumulus scheme does not
#: read the field drops it from its identity (woof.config refuses a
#: non-default value there, which is what makes the drop lossless), so no
#: fingerprint written before the field existed moves.
def _cumulus_scoped_run_fields() -> dict[str, frozenset]:
    from woof.config import NSAS_DX_FACTOR_SCHEMES, SAS_PGCON_SCHEMES
    return {"nsas_dx_factor": NSAS_DX_FACTOR_SCHEMES,
            "sas_pgcon": SAS_PGCON_SCHEMES}

#: The adaptive-timestep surface, scoped OFF the restart identity whenever
#: ``use_adaptive_time_step`` is false -- see ``restart_identity_payload``.
#: The flag itself is in the tuple because "off" is the state every
#: checkpoint written before this block existed describes, so it must drop
#: out entirely rather than appear as an added false.
ADAPTIVE_TIMESTEP_RUN_FIELDS: tuple[str, ...] = (
    "use_adaptive_time_step", "step_to_output_time", "adaptation_domain",
    "target_cfl", "target_hcfl", "max_step_increase_pct",
    "starting_time_step", "starting_time_step_den",
    "max_time_step", "max_time_step_den",
    "min_time_step", "min_time_step_den",
    # Not a WRF key: the floor the steep-terrain rules put under the
    # substep count the clock derives from its step (RunConfig's own
    # note says why).  Read by nothing under a fixed clock, so it drops
    # out with the block there; a clamp like min_time_step under an
    # adaptive one, so a resume may change it and says so.
    "min_time_step_sound",
)

#: The same surface MINUS the feature flag: controller POLICY.
#:
#: These govern how the clock behaves from the resume point onward; they
#: are not model state, and the checkpoint does not depend on them.  A
#: restart may therefore change them -- see ``woof.io.restart``, which
#: allows it and reports the change rather than refusing, because a
#: checkpoint whose controller setting cannot be altered is useless for
#: recovering the run that setting killed.
#:
#: ``use_adaptive_time_step`` is excluded deliberately.  Flipping it
#: leaves the carried controller state describing a clock that no longer
#: runs, so that stays a refusal.
ADAPTIVE_POLICY_RUN_FIELDS: frozenset[str] = frozenset(
    ADAPTIVE_TIMESTEP_RUN_FIELDS) - {"use_adaptive_time_step"}


def restart_identity_payload(exp) -> dict:
    """The experiment, minus everything a restart may legally change.

    Extracted so the prepared domain-tree runner can bind the same
    identity this function defines instead of hashing the experiment
    FILE.  A file digest moves when any byte of the TOML moves, which
    made the tree route refuse every one of the three changes the
    ``--restart`` contract publishes as permitted -- including extending
    the run, which is the reason people configure checkpoints at all.
    """

    from woof.experiment import experiment_config_document
    from woof.checkpoint_identity import (
        drop_default_diffusion_selectors, drop_default_spp_selectors)
    experiment = _jsonable(experiment_config_document(exp))
    # The new attribute follower is an explicitly selected trajectory policy.
    # Bind its source, reduction, direction and movement controls while keeping
    # the pre-existing relocation exemption byte-identical for legacy signals.
    attribute_policy = experiment.get("relocation", {})
    if (attribute_policy.get("follow") or {}).get("field") == "attribute":
        attribute_policy = {key: value for key, value in attribute_policy.items()
                            if key != "track"
                            # The reach bound's DEFAULT drops out, on the
                            # absent-stays-absent convention below: every
                            # fingerprint written before the key existed
                            # keeps its value, and a declared bound binds.
                            and not (key == "reach_speed_m_s"
                                     and value is None)}
    else:
        attribute_policy = None
    for name in RESTART_TOLERATED_EXPERIMENT_FIELDS:
        experiment.pop(name, None)
    if attribute_policy is not None:
        experiment["relocation"] = attribute_policy
    # ABSENT [perturbation] stays absent from the identity payload, not
    # present-as-null: every experiment written before the field existed
    # keeps its exact fingerprint and restart identity (the mixed_edges
    # convention in experiment_fingerprint, applied at the source).  A
    # configured block binds, value for value.
    if experiment.get("perturbation") is None:
        experiment.pop("perturbation", None)
    # Same convention for [spectral_numerics]: ABSENT stays absent, so
    # every pre-feature fingerprint and checkpoint is preserved.  A
    # PRESENT block binds value for value -- the Level-2 contract makes
    # the resolved operator config part of what the run IS, because a
    # resume across a changed operator (an applied filter appearing,
    # disappearing, or retuned mid-trajectory) is not the same
    # experiment and must refuse.
    if experiment.get("spectral_numerics") is None:
        experiment.pop("spectral_numerics", None)
    if experiment.get("nudging") is None:
        experiment.pop("nudging", None)
    # Same convention for [physics_params]: ABSENT stays absent, so every
    # pre-feature fingerprint and checkpoint is preserved.  A PRESENT set
    # binds value for value: resuming a checkpoint under other constants is
    # another trajectory and must refuse.
    if experiment.get("physics_params") is None:
        experiment.pop("physics_params", None)
    # [radar_heating]: absent stays absent (experiment_config_document drops
    # an OFF table, so every pre-feature fingerprint is unchanged).  Present
    # it binds, so a checkpoint written heated cannot resume under other
    # heating or none -- except the windows ROOT, which says where the files
    # are and not what they hold; the windows are held to the run's grid
    # identity and their own digests when they are read.
    heating = experiment.get("radar_heating")
    if isinstance(heating, dict):
        heating.pop("windows", None)
    # Same convention for the declared constant downward longwave: ABSENT
    # stays absent, so every experiment written before the field existed
    # keeps its exact fingerprint and its checkpoints keep resuming.  A
    # DECLARED number binds, because it is the flux the land surface
    # integrates for the whole forecast -- resuming a 300 W m-2
    # trajectory under 410 is a different run.
    if experiment.get("constant_glw_wm2") is None:
        experiment.pop("constant_glw_wm2", None)
    # The mixing-length provenance label leaves the identity
    # UNCONDITIONALLY, on the [tiles] convention: it records WHO chose
    # each domain's mix_isotropic, while the chosen value itself sits on
    # run.mix_isotropic and binds there.  An auto-selected 1 and a
    # written 1 integrate the same bytes, so each must resume the
    # other's checkpoints -- and every pre-feature fingerprint stays
    # byte-identical because the key is simply absent, as it always was.
    experiment.pop("auto_mix_isotropic", None)
    # The off-centering provenance label leaves for the same reason: the
    # epssm that runs (raised by the measured floor or not) binds on
    # run.epssm, and an "auto" 0.1 resumes a written 0.1's checkpoints.
    experiment.pop("auto_epssm", None)
    # WRF's smooth_cg_topo, absent-stays-absent: off it is how every root
    # terrain was built before the key existed; on, it binds, because the
    # blended root terrain is what the whole run integrates.
    if not experiment.get("smooth_cg_topo", False):
        experiment.pop("smooth_cg_topo", None)
    for domain in experiment.get("domains", ()):
        for name in RESTART_TOLERATED_DOMAIN_FIELDS:
            domain.pop(name, None)
        # Same convention for the per-domain spawn declaration: absent
        # stays absent (pre-feature fingerprints byte-identical), and a
        # declared spawn BINDS, value for value.  It decides trajectory --
        # when a slot fires, where the newborn lands, how many episodes it
        # may serve -- and a resume rebuilds the checkpoint's live tree
        # from the runner's state, so restoring one under a DIFFERENT
        # declaration would re-materialize episodes against a policy that
        # never produced them.  Binding it makes that a refusal.
        if domain.get("spawn") is None:
            domain.pop("spawn", None)
        # The three per-domain lifecycle tables take the SAME convention as
        # the spawn declaration beside them, and for the same two reasons in
        # both directions.
        #
        # Absent stays absent because these fields landed on DomainConfig
        # defaulting to None, and the payload is built by
        # ``dataclasses.asdict``: an unpopped None serializes as
        # ``retire: null, rearm: null, follow: null`` on EVERY domain of
        # EVERY experiment, including every experiment written before the
        # fields existed.  That is pure identity churn -- it moves each of
        # those fingerprints without changing one number any of them
        # integrates, and a moved fingerprint is a checkpoint that refuses
        # to restore.  It cost the pre-lifecycle anchor
        # (tests/test_water_overlay.py) exactly the way the NSSL row above
        # did, which is what that anchor is for.
        #
        # A DECLARED table binds, value for value, because each of the three
        # decides trajectory: [retire] says when a child stops integrating,
        # [rearm] how many episodes the slot may serve, [follow] where the
        # child sits.  A resume across a change to any of them must refuse.
        for name in ("retire", "rearm", "follow"):
            if domain.get(name) is None:
                domain.pop(name, None)
        # A follower's reach bound takes the same convention one level
        # down: its default (None) drops out so every follower fingerprint
        # written before the key existed is unchanged, and a DECLARED
        # bound binds, because it decides where the nest may go.
        follow = domain.get("follow")
        if isinstance(follow, dict) and follow.get("reach_speed_m_s") is None:
            follow.pop("reach_speed_m_s", None)
        # The per-domain [tiles] road leaves the identity UNCONDITIONALLY,
        # declared or not -- the one place this file's absent-stays-absent
        # convention is not enough.  A declared spawn binds because it
        # decides the trajectory a resume rebuilds; a declared tiling binds
        # NOTHING because its entire claim is that it changes no bytes, and
        # binding it would refuse the operation it exists for: a domain
        # that streamed resuming resident, or a domain that outgrew its
        # card resuming streamed.  Same law as the tree-wide table above
        # (RESTART_TOLERATED_EXPERIMENT_FIELDS "tiles").
        domain.pop("tiles", None)
        # The per-domain [output] selection leaves UNCONDITIONALLY too,
        # and for the same reason: it decides which variables reach the
        # HISTORY tape and binds no number the model computes.  A domain
        # that wrote the full inventory must resume trimmed, and one that
        # trimmed must resume full.
        domain.pop("output", None)
        # The quick-look stream is output-only on the same terms.
        domain.pop("full_history_interval_s", None)
        domain.pop("quicklook_output", None)
        run = domain.get("run", {})
        drop_default_diffusion_selectors(run)
        drop_default_spp_selectors(run)
        # ghg_input, absent-stays-absent: 0 is the fixed-gas radiation
        # every fingerprint written before the key ran; 1 binds.
        if not run.get("ghg_input", 0):
            run.pop("ghg_input", None)
        # The historical scalar operator was PD. Its unchanged default
        # leaves old tree checkpoint identities intact; TKE activation binds.
        if run.get("scalar_adv_opt", 1) == 1:
            run.pop("scalar_adv_opt", None)
        from woof.lightning_contract import drop_default_lightning_selectors
        drop_default_lightning_selectors(run)
        # fractional_seaice, absent-stays-absent: 0 is the 0.5 sea-ice
        # threshold every RUC fingerprint written before the field ran.
        if not run.get("fractional_seaice", 0):
            run.pop("fractional_seaice", None)
        # These radiation-driver fields were absent from older fingerprints.
        # Zero retains the prior trajectory; every enabled selector binds.
        if run.get("aer_opt", 0) != 2:
            run.pop("aer_type", None)
            run.pop("aer_aod550_opt", None)
            run.pop("aer_angexp_opt", None)
            run.pop("aer_ssa_opt", None)
            run.pop("aer_asy_opt", None)
            run.pop("aer_aod550_val", None)
            run.pop("aer_angexp_val", None)
            run.pop("aer_ssa_val", None)
            run.pop("aer_asy_val", None)
            run.pop("aer_auxinput15", None)
            run.pop("aer_auxinput15_interval", None)
        for name in ("swint_opt", "aer_opt", "alb_sol"):
            if not run.get(name, 0):
                run.pop(name, None)
        for name in RESTART_TOLERATED_RUN_FIELDS:
            run.pop(name, None)
        # ``eta_levels`` on the absent-stays-absent convention of the
        # blocks above: its default ``None`` means "inherit the ladder the
        # source carries", which is exactly and only what every experiment
        # written before the field existed did, so at its default it
        # carries no trajectory information and binding it was pure
        # identity churn -- the prepared-cache table already tolerated it
        # and this payload did not (ENG-010).  A DECLARED ladder binds,
        # value for value: a child on its own eta grid is a different run.
        if run.get("eta_levels") is None:
            run.pop("eta_levels", None)
        # The downscaled child's two lateral-boundary keys, on the same
        # convention: at their defaults (0.0, False) they are WRF's own
        # relaxation and zero-gradient w, which is what every experiment
        # written before they existed ran, so they drop out and every
        # pre-field fingerprint and checkpoint is unchanged.  A set value
        # binds: a zone relaxed on another time scale, or with w, is a
        # different run.
        if not run.get("relax_timescale_s"):
            run.pop("relax_timescale_s", None)
        if not run.get("relax_w"):
            run.pop("relax_w", None)
        # New canopy controls bind selected nondefault physics while their
        # defaults preserve the older numerical-generation fingerprint.
        if run.get("urban_ucm_generation", "wrf_471") == "wrf_471":
            run.pop("urban_ucm_generation", None)
        for name in ("urban_treeoption", "urban_ttscheme", "urban_fvg"):
            if run.get(name, -1) == -1:
                run.pop(name, None)
        if run.get("urban_table_dir", "") == "":
            run.pop("urban_table_dir", None)
        # The urban canopy keys, on the same convention: with
        # sf_urban_physics = 0 (WRF's default and every experiment written
        # before the urban models existed) all three drop out, so no
        # pre-urban fingerprint or checkpoint moves; a selected urban model
        # binds all three, because each decides what the urban columns
        # integrate.  woof/io/restart.py's configuration digest drops the
        # same three on the same condition (_URBAN_DIGEST_FIELDS).
        if not run.get("sf_urban_physics"):
            for name in ("sf_urban_physics", "use_wudapt_lcz",
                         "num_urban_hi", "urban_water_balance"):
                run.pop(name, None)
        elif int(run.get("sf_urban_physics", 0)) != 1:
            run.pop("urban_water_balance", None)
        elif run.get("urban_water_balance", "wrf_471") == "wrf_471":
            run.pop("urban_water_balance", None)
        # Scheme-scoped knobs leave the identity of every domain that does
        # not select their scheme (:data:`SCHEME_SCOPED_RUN_FIELDS`).
        selected = run.get("mp_physics")
        for scheme, names in SCHEME_SCOPED_RUN_FIELDS.items():
            if selected == scheme:
                continue
            for name in names:
                run.pop(name, None)
        for name, readers in _cumulus_scoped_run_fields().items():
            if run.get("cu_physics") not in readers:
                run.pop(name, None)
        # The correct-mode keys (woof/correct_mode.py), by the checkpoint
        # echo's rule (woof/io/restart.py _drop_default_off_run_keys): a key
        # binds only where it is on AND its scheme reads it.  Off is the
        # matched arithmetic every fingerprint before the keys ran, and a
        # key no selected scheme reads reaches nothing, so every experiment
        # without WSM6 or New Tiedtke keeps its fingerprint under the 2.9.0
        # correct-mode default, and an exact-mode WSM6 run keeps its own.
        # Breakage this prevents: the keys joined RunConfig (False, then
        # default-on) and moved every recorded fingerprint and tree restart
        # identity, schemes or not (tests/test_water_overlay.py,
        # tests/test_devices_door.py).
        from woof.correct_mode import (CORRECTION_KEYS, OUTPUT_ONLY_KEYS,
                                        key_in_scope)
        for name in CORRECTION_KEYS:
            if (name in OUTPUT_ONLY_KEYS or not run.get(name, False)
                    or not key_in_scope(name, run)):
                run.pop(name, None)
        # THE SURFACE-RADIATION CARRIER POLICY, on the same convention the
        # perturbation and spawn blocks above use: the DEFAULT drops out of
        # the identity payload, so every experiment written before the
        # policy existed keeps its exact fingerprint and restart identity,
        # and the DECLARED ESCAPE binds.  That is the whole requirement --
        # "wrf_compat_zero" has to be part of what a run IS, because a leg
        # that consumed a sky nobody computed is not the same experiment as
        # one that did not, and a resume across the change must refuse.  A
        # label that changed every existing fingerprint while changing no
        # trajectory would be identity churn, which is the opposite of what
        # identity is for.
        if run.get("surface_radiation_policy") == (
                SURFACE_RADIATION_POLICY_REQUIRED):
            run.pop("surface_radiation_policy", None)
        # THE ADAPTIVE-TIMESTEP BLOCK, on that same convention and for the
        # same reason.  Twelve fields joined RunConfig at once, and an
        # unscoped RunConfig field moves EVERY experiment fingerprint
        # whether or not it changes a number -- which would have refused
        # every checkpoint already written, for a controller that is off.
        #
        # OFF is the whole argument, and it is stronger here than
        # "the default value": with use_adaptive_time_step false the
        # controller never runs, so the other eleven are read by nothing
        # at all and cannot have influenced a single byte of the
        # trajectory.  Dropping them discards no information.
        #
        # ON, the flag and the CONFIGURED step still bind value for value.
        #
        # The POLICY does not, and this paragraph used to say it did: "a
        # resume that changed target_cfl, a clamp, or the growth bound
        # would be integrating on a different clock from the leg that
        # wrote the checkpoint, and that is not the same experiment."
        # True, and the conclusion drawn from it was wrong -- integrating
        # on a different clock from here on is precisely what a recovery
        # resume is for, and binding it made restart_interval_s useless
        # for the one case that most wants it: a run that died because of
        # the setting you now need to change.
        #
        # `woof.io.restart` allows such a resume and REPORTS the change,
        # naming each field and its old -> new value.  This fingerprint
        # has to agree with that, or it refuses what the walk beside it
        # just permitted -- which is what happened, and it surfaced as a
        # relocation-lineage refusal because a moved tree picks that
        # message for ANY fingerprint mismatch.
        #
        # `dt` stays bound: the identity carries the CONFIGURED step, not
        # the adapted one, so it is stable across a run and changing it
        # really is a different experiment.
        if not run.get("use_adaptive_time_step", False):
            for name in ADAPTIVE_TIMESTEP_RUN_FIELDS:
                run.pop(name, None)
        else:
            for name in ADAPTIVE_POLICY_RUN_FIELDS:
                run.pop(name, None)
        # An absent key and the default describe the original clock.
        if not run.get("adaptive_nest_lattice", False):
            run.pop("adaptive_nest_lattice", None)
        if not run.get("use_rap_aero_icbc", False):
            # The absent selector read no analyzed donor and no operational
            # surface source. Keep older default restart identities intact.
            run.pop("use_rap_aero_icbc", None)
        # WRF's slope_rad / topo_shading / shadlen, absent-stays-absent:
        # off, none of the three is read, so every fingerprint written
        # before they existed keeps its value; a domain that turns the
        # slope flux on binds all three, because together they are the
        # shortwave its land surface integrates.
        if not run.get("slope_rad", 0):
            for name in ("slope_rad", "topo_shading", "shadlen"):
                run.pop(name, None)
        elif not run.get("topo_shading", 0):
            run.pop("topo_shading", None)
            run.pop("shadlen", None)
        # ... and the original explicit vertical advection (A158).
        if not run.get("zadvect_implicit", 0):
            run.pop("zadvect_implicit", None)
        if run.get("sf_surface_physics", 0) != 4:
            run.pop("noahmp_snow_age", None)
        elif run.get("noahmp_snow_age") == "wrf_461":
            run.pop("noahmp_snow_age", None)
        # The solar generation: wrf_461 is what every earlier fingerprint
        # ran, so it is omitted; the WRF 4.8 default is bound by name.
        if run.get("solar_generation", "wrf_461") == "wrf_461":
            run.pop("solar_generation", None)
        if run.get("zadvect_implicit_variant", "wrf_471") == "wrf_471":
            run.pop("zadvect_implicit_variant", None)
        # ... and the RUC SOILPROP lineage, omitted at its wrf_45 default so
        # a non-RUC run's fingerprint does not move; the wrf_461 name is
        # bound because its soil water differs.
        if run.get("ruc_soilprop", "wrf_45") == "wrf_45":
            run.pop("ruc_soilprop", None)
        # ... and the Thompson generation, omitted at its wrf_461 default;
        # the fork's name is bound because its microphysics differs.
        if run.get("thompson_version", "wrf_461") == "wrf_461":
            run.pop("thompson_version", None)
        if run.get("thompson_fork_snow_fall", "blend") == "blend":
            run.pop("thompson_fork_snow_fall", None)
        # ... and the MYNN generation, omitted at wrf_461 (every earlier
        # fingerprint), bound by name off it; its gsd_41 TKE option is
        # omitted at its default for the same reason.
        if run.get("wind_farm_scheme", "off") == "off":
            run.pop("wind_farm_scheme", None)
            run.pop("wind_farm_table", None)
            run.pop("wind_farm_max_dx_m", None)
            run.pop("wind_farms", None)
            run.pop("windfarm_tke_factor", None)
        else:
            from woof.wind_table import read_table
            _, run["wind_farm_table_sha256"] = read_table(
                run["wind_farm_table"], run["wind_farm_scheme"])
        if run.get("bl_mynn_version", "wrf_461") == "wrf_461":
            run.pop("bl_mynn_version", None)
        # Headers before this selector ran the legacy YSU kernel.
        if (run.get("bl_pbl_physics", 0) != 1
                or run.get("ysu_version", "wrf_461") == "wrf_461"):
            run.pop("ysu_version", None)
        if run.get("bl_mynn_gsd41_unsquared_qtke", False) is False:
            run.pop("bl_mynn_gsd41_unsquared_qtke", None)
        if run.get("bl_mynn_cloud_tendency_form", "wrf_461") == "wrf_461":
            run.pop("bl_mynn_cloud_tendency_form", None)
        # WRF's cycling flag, omitted at its .false. default.
        if run.get("cycling", False) is False:
            run.pop("cycling", None)
        # ... and the MYNN surface-layer generation, omitted at its wrf_461
        # default (what every earlier build ran) so no fingerprint moves.
        if run.get("mynn_sfclay_variant", "wrf_461") == "wrf_461":
            run.pop("mynn_sfclay_variant", None)
        # ... and the WRF v4.6.1 sixth-order filter form with no second
        # factor, and no mp_zero_out pass: omitted at those defaults so
        # every earlier fingerprint keeps its value; the fork form, its
        # factor and an active zero-out bind, because their moisture
        # differs.
        if run.get("diff_6th_form", "wrf_461") == "wrf_461":
            run.pop("diff_6th_form", None)
        if run.get("diff_6th_factor2", None) is None:
            run.pop("diff_6th_factor2", None)
        if run.get("upper_wind_limiter_form", "wrf_461") == "wrf_461":
            run.pop("upper_wind_limiter_form", None)
        if not run.get("mp_zero_out", 0):
            run.pop("mp_zero_out", None)
            run.pop("mp_zero_out_thresh", None)
            run.pop("mp_zero_out_all", None)
        # ... and the public RUC irrigation form (wrf_461),
        # omitted at its default so a non-RUC run's fingerprint does not
        # move; the wrf_45 name is bound because its soil water differs.
        if run.get("ruc_irrigation", "wrf_461") == "wrf_461":
            run.pop("ruc_irrigation", None)
        # ... and the RUC QVG cold start, omitted at its default.
        if run.get("ruc_qvg_cold_start", "wrf") == "wrf":
            run.pop("ruc_qvg_cold_start", None)
        # ... and the RUC 2 m diagnostic, omitted at its flux default.
        if run.get("ruc_2m_diagnostic", "flux") == "flux":
            run.pop("ruc_2m_diagnostic", None)
        # ... and the RUC snow lineage, omitted at its wrf_461 default so a
        # non-RUC run's fingerprint does not move.
        if run.get("ruc_snow", "wrf_461") == "wrf_461":
            run.pop("ruc_snow", None)
        # ... and WRF's vert_order 3 ladder with the flux5 momentum
        # stencil, which every echo before the order fields described.
        for name, default in (("v_sca_adv_order", 3), ("v_mom_adv_order", 3),
                              ("h_mom_adv_order", 5)):
            if int(run.get(name, default)) == default:
                run.pop(name, None)
        # ... and w_damp measured from Courant 1, WRF's default (A165).
        if float(run.get("w_crit_cfl", 1.0)) == 1.0:
            run.pop("w_crit_cfl", None)
        from woof.surface_ocean import drop_inert_ocean
        drop_inert_ocean(run)
        if not run.get("sf_lake_physics", 0):
            run.pop("sf_lake_physics", None)
            run.pop("use_lakedepth", None)
            run.pop("lakedepth_default", None)
            run.pop("lake_min_elev", None)
        # Noah mosaic, absent-stays-absent: off (WRF's default), no tile is
        # integrated and none of the three keys is read, so every
        # fingerprint written before mosaic existed keeps its value.  On,
        # the switch and tile count bind; the urban canopy rule binds when
        # it is not WRF's ("dominant"), because the town rule gives towns in
        # rural cells an urban fraction WRF's rule leaves at zero.
        # woof/io/restart.py's header and digest drop them on the same
        # conditions.
        if not run.get("sf_surface_mosaic", 0):
            for name in ("sf_surface_mosaic", "mosaic_cat",
                         "mosaic_urban_canopy"):
                run.pop(name, None)
        elif run.get("mosaic_urban_canopy") == "dominant":
            run.pop("mosaic_urban_canopy", None)
        # ... and no sub-grid terrain drag, WRF's default (topo_wind and
        # gwd_opt 0; woof.core.terrain_drag): absent-stays-absent, so every
        # fingerprint written before the two fields existed keeps its value.
        for name in ("topo_wind", "gwd_opt"):
            if not run.get(name, 0):
                run.pop(name, None)
        # Off, post-PBL scalar diffusion preserves pre-option fingerprints.
        if not run.get("scalar_pblmix", 0):
            run.pop("scalar_pblmix", None)
        # The chem block, on the adaptive block's OFF argument: with
        # chem_sets empty no chem key is read by anything (woof.config.
        # validate_chem_config), so dropping the block keeps every chem-off
        # fingerprint and checkpoint identity what it was before the block
        # existed.  A chem run binds the whole block value for value.
        if not run.get("chem_sets"):
            from woof.config import CHEM_RUN_FIELDS
            for name in CHEM_RUN_FIELDS:
                run.pop(name, None)
        # The fire block on the same rule (woof.config.inert_fire_fields).
        from woof.config import inert_fire_fields
        for name in inert_fire_fields(run):
            run.pop(name, None)
        # The terrain-clock mode: "measured" is the derivation every
        # fingerprint written before the field ran under, so it drops
        # out; "pinned" binds, because the same configured clock then
        # integrates where the measured rule might have divided it.
        # "local_face", the default from 2.8.8, binds too: it can run a
        # longer step than "measured" on the same grid, so a run under it
        # is not the trajectory an absent key names.
        if run.get("terrain_clock", "measured") == "measured":
            run.pop("terrain_clock", None)
        # Earlier runs held shortwave between calls, used zero aerosol,
        # and ran the WRF v4.6.1 cloud wrapper. Only moved forms bind.
        if run.get("aer_opt", 0) != 2:
            run.pop("aer_type", None)
            run.pop("aer_aod550_opt", None)
            run.pop("aer_angexp_opt", None)
            run.pop("aer_ssa_opt", None)
            run.pop("aer_asy_opt", None)
            run.pop("aer_aod550_val", None)
            run.pop("aer_angexp_val", None)
            run.pop("aer_ssa_val", None)
            run.pop("aer_asy_val", None)
            run.pop("aer_auxinput15", None)
            run.pop("aer_auxinput15_interval", None)
        for key in ("swint_opt", "aer_opt"):
            if not run.get(key, 0):
                run.pop(key, None)
        if run.get("rrtmg_cloud_optics_form", "wrf_461") == "wrf_461":
            run.pop("rrtmg_cloud_optics_form", None)
        # Earlier runs took rural aerosol type 1 everywhere; only the
        # land/water type binds.
        if run.get("rrtmg_aer3_type_source", "rural") == "rural":
            run.pop("rrtmg_aer3_type_source", None)
        # Source members, not a path alone, identify prescribed radiation.
        from woof.core.rrtmg_smoke_identity import bind_smoke_source_identity
        if run.get("rrtmg_smoke_manifest", ""):
            bind_smoke_source_identity(run,
                start_time=domain.get("start_time") or exp.start_time,
                nz=run["nz"])
        else:
            bind_smoke_source_identity(run)
    return experiment


def experiment_fingerprint(exp, catalog) -> str:
    """Hash immutable trajectory setup plus the InputCatalog inventory.

    Forecast duration and output/restart cadence are deliberately excluded:
    they can change on resume without changing any integration before or
    after the checkpoint.  Domain geometry, timestep, physics, nesting,
    forcing provenance, and every other experiment field remain bound.
    """
    experiment = restart_identity_payload(exp)
    payload = {
        "experiment": experiment,
        "input_catalog": _jsonable(catalog.run_provenance),
    }
    # A mixed-scheme restart is trajectory-compatible only with the exact
    # translator implementation, field map, and policy metadata.  Keep the
    # legacy/same-scheme payload byte-inert, but bind every mixed directed
    # edge to its complete executable identity.
    from woof.core.microphysics_transition import (
        resolve_microphysics_transition,
    )
    by_id = {domain.grid_id: domain for domain in exp.domains}
    mixed_edges = []
    for domain in exp.domains:
        if domain.parent_id == 0:
            continue
        contract = resolve_microphysics_transition(
            by_id[domain.parent_id].run, domain.run)
        if contract.mixed:
            mixed_edges.append({
                "source_domain": int(domain.parent_id),
                "target_domain": int(domain.grid_id),
                "contract": _jsonable(contract.receipt()),
            })
    if mixed_edges:
        payload["mixed_microphysics_transitions"] = mixed_edges
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def uses_modern_rrtmgp_workspace(exp) -> bool:
    """Whether the tree build must allocate the shared RRTMGP workspace.

    Variant-aware, mirroring ``estimate_experiment``: the persistent
    :class:`SharedRRTMGPChunkWorkspace` exists only for the modern
    RTE+RRTMGP adapter.  The legacy-RRTMG variant runs one domain at a
    time under its engine-default ``column_chunk=None`` contract with
    LW/SW allocations sequenced and freed between calls, so its VRAM
    price is the estimator's transient call-peak envelope
    (``workspace_bytes`` under ``uses_legacy``), never a held workspace
    -- constructing (and cross-checking) the modern workspace for it
    both wastes the allocation and trips the memory-ledger drift guard.
    Mixed variants retain this modern workspace while legacy engines run.
    """
    from woof.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant

    variants_44 = {rrtmg_variant(dc.run) for dc in exp.domains
                   if 4 in radiation_scheme_ids(dc.run)}
    return bool(variants_44) and variants_44 != {RRTMG_VARIANT_LEGACY}


def _forcing_cadence_seconds(catalog) -> float:
    """Resolve T9's LBC cadence from decoded InputCatalog records."""
    records = tuple(catalog.lbc_records)
    if not records:
        raise ValueError("the forcing catalog has no LBC interval")
    deltas = {float(record.delta_seconds) for record in records}
    if len(deltas) != 1:
        raise ValueError(
            "the integer DomainClock requires one uniform forcing cadence; "
            f"InputCatalog deltas are {sorted(deltas)}")
    return deltas.pop()


def publish_declared_experiment(model, exp) -> None:
    """Publish, on the tree, the experiment the tree was configured from.

    THE ONE CARRIER, for every route.  A checkpoint that persists a live
    follower has to record where each domain was DECLARED alongside where
    it now sits, or a resume cannot tell a nest the follower moved from a
    nest somebody reconfigured -- and the declaration is a fact about the
    configuration, not about whichever ingest built the tree.  Any route
    that assembles an :class:`ExperimentState` calls this, so restart
    stays a capability of the tree rather than of the route.
    """
    if exp is None:
        raise ValueError(
            "a tree is published with the experiment it was configured "
            "from; publishing None leaves the checkpoint writer unable to "
            "say whether a follower's recorded placement is the declared "
            "one or one the follower moved to")
    model._declared_experiment = exp


def _adapt_experiment_vertical_for_case(exp, case_data, catalog):
    """The coordinate this case's own ground can order, applied.

    Returns the experiment the whole run then uses.  The survey reads the
    root's statics through :func:`woof.runtime.case_static_fields` -- the
    same cached call the root preparation makes a few lines below -- and
    each other declared domain's terrain at its own resolution, so a nest
    carrying higher peaks than its parent is priced before the parent's
    coordinate is built rather than after.
    """

    from woof import runtime
    from woof.explain import warn
    from woof.static.build import GeogSelection
    from woof.static.lambert import grids_from_projection_config
    from woof.vertical_adaptation import (adapt_experiment_for_statics,
                                           static_catalog_for_survey)

    if not exp.vertical.eta_levels:
        return exp
    grids = tuple(grids_from_projection_config(exp))
    # A declared field of CaseDataConfig, default None (the identity
    # path), read as one: this module is a clock module under an AST
    # audit that bans getattr
    # (tests/test_clock.py::test_no_float_elapsed_accumulation_audit),
    # and every route that reaches this line hands the validated case
    # data, whose geog_root and wps_namelist are read the same way.
    highres = case_data.static_highres
    root = exp.root
    terrain = runtime.case_static_fields(
        grids[0], case_data.geog_root,
        selection=GeogSelection.from_case_data(
            case_data, domain_id=root.grid_id),
        static_highres=highres, domain_id=root.grid_id,
        case_date=exp.start_time.date())["HGT_M"]

    def announce(sentence: str) -> None:
        warn(sentence,
             "WRF v4.6.1 dyn_em/nest_init_utils.F:1158-1182 calls a column "
             "the coordinate cannot order fatal and names reducing etac as "
             "the remedy, and a column it only just orders keeps one layer "
             "too thin to integrate, which a lower etac thickens; the etac is "
             "derived here from the terrain this run can "
             "actually touch and applied, so every domain of this run "
             "integrates the same coordinate.  p_top is untouched.")

    adapted, _adaptation = adapt_experiment_for_statics(
        exp, grids, root_terrain=terrain,
        static_catalog=(static_catalog_for_survey(catalog)
                        if len(exp.domains) > 1 else None),
        static_highres=highres, announce=announce)
    return adapted


def build_experiment(exp, case_data) -> ExperimentState:
    """Build the all-resident domain tree parent before child.

    The real root follows the existing Task-2 preparation path.  Each child
    then follows Task 12's WRF-order initialization and receives its own
    physics driver and Task-13 coupler.  Every state is constructed against
    the one shared transient arena before any scratch slot is materialized.
    """
    from woof.experiment import _bind_physics_params
    exp = _bind_physics_params(exp, "woof.core.model.build_experiment")
    from woof import runtime
    from woof.core.clock import build_schedule, resolve_clock
    from woof.core.preflight import estimate_experiment
    from woof.core.state import (build_shared_dycore_state_workspace,
                                  build_shared_scratch_arena)
    from woof.ingest.grib import clear_forcing_caches
    from woof.ingest.lateral_bc import bind_lateral_boundary_clock
    from woof.ingest.nest_init import initialize_child
    from woof.core.nest import NestCoupler as ConcreteNestCoupler

    from woof.progress import prep_stage

    # Each step below says when it starts and ends, and the start state
    # names its grid, so a run page can show where a nested preparation is:
    # this builder was one silent stretch of minutes.  Said to this
    # process's listener only (stderr=False): the builder runs inside the
    # command that holds the run, whose stderr is a person's terminal.
    with prep_stage("source_decode", label="Read the starting data",
                    stderr=False):
        catalog = runtime._runtime_input_catalog(case_data)
        # THE COORDINATE, BEFORE ANY OTHER READ OF THE EXPERIMENT.  This route
        # holds the experiment for the whole run -- the root below, every
        # child, and every spawn and relocation that reads exp.vertical later
        # -- so the derivation belongs at the top of the builder and not
        # beside the root preparation, which would leave the children on the
        # configured value.  The line's POSITION is the contract: the
        # startup tree (`active`, below) is taken from `exp` by value, and
        # each child's DomainConfig is stored on its node, where
        # core/streaming.domain_vertical_coord rebuilds a streamed tile
        # buffer's coordinate from cfg.etac.  Derived after that copy, a
        # streamed nest would rebuild its buffer on a coordinate its own
        # domain is not on, which is the disagreement this lane exists to
        # make impossible.  The root's statics are built through the same
        # cached function the preparation makes a few lines below, so asking
        # for them here costs nothing there.
        exp = _adapt_experiment_vertical_for_case(exp, case_data, catalog)
        snapshots = runtime.forcing_snapshots(case_data, catalog)
    forcing_times = runtime.forcing_schedule(exp, case_data, snapshots)
    lbc_interval_s = _forcing_cadence_seconds(catalog)
    # Dormant (spawn-declared) nests are RESERVED but not INTEGRATED:
    # the memory plan, the shared arenas and the fingerprint below price
    # and bind the FULL experiment (a declared-but-never-triggered nest
    # costs its reserved VRAM and zero compute -- that is the contract),
    # while the clock, the schedule and the child-init loop see only the
    # active tree.  Mid-run activation is the spawn runner's leg
    # boundary (woof.experiment.active_experiment), not this builder's.
    from woof.experiment import pre_spawn_experiment
    from woof.vertical_adaptation import (
        refuse_tree_off_the_experiment_coordinate)
    active = pre_spawn_experiment(exp)
    # The startup tree is a fresh copy of the experiment's DomainConfigs
    # and every child node below stores one, so it must already carry the
    # coordinate the derivation above chose.  The refusal names what a
    # split tree breaks.
    refuse_tree_off_the_experiment_coordinate(
        exp, active.domains, route="experiment-tree")
    # The startup tree's tick is the DECLARED tree's (dormant nests
    # included), so a nest spawned later joins without re-meaning a tick.
    tick_clock = resolve_clock(
        active, lbc_interval_s=lbc_interval_s,
        tick_den_floor=(resolve_clock(exp, lbc_interval_s=lbc_interval_s).tick_den
                        if active is not exp and len(active.domains) != len(exp.domains)
                        else None))
    schedule = build_schedule(active, tick_clock)
    clocks = tick_clock.clocks()
    estimate = estimate_experiment(
        exp, forcing_interval_seconds=lbc_interval_s)
    dycore_state_workspace = (
        build_shared_dycore_state_workspace(exp.domains)
        if len(exp.domains) > 1 else None)
    if (dycore_state_workspace is not None
            and dycore_state_workspace.nbytes
            != estimate.dycore_state_workspace_bytes):
        raise RuntimeError(
            "runtime shared dycore-state workspace drifted from the memory "
            f"ledger: {dycore_state_workspace.nbytes} != "
            f"{estimate.dycore_state_workspace_bytes} bytes")
    arena = (build_shared_scratch_arena(exp.domains)
             if len(exp.domains) > 1 else None)
    if arena is not None and arena.nbytes != estimate.scratch_arena_bytes:
        raise RuntimeError(
            "runtime scratch arena drifted from the memory ledger: "
            f"{arena.nbytes} != {estimate.scratch_arena_bytes} bytes")
    # Variant-aware (F' contract): the shared chunk workspace is a modern
    # RTE+RRTMGP object.  Under ra_rrtmg_variant='rrtmg_legacy' the
    # estimator's workspace_bytes is the legacy transient call-peak
    # envelope, not a persistent allocation, and the legacy adapter keeps
    # its engine-default column_chunk=None contract -- so no workspace is
    # built (and no chunking attributes are stamped onto the adapter).
    radiation_workspace = (
        SharedRRTMGPChunkWorkspace(
            nz=exp.root.run.nz, column_chunk=exp.column_chunk,
            p_top=exp.vertical.p_top)
        if uses_modern_rrtmgp_workspace(exp) else None)
    if (radiation_workspace is not None
            and radiation_workspace.nbytes != estimate.workspace_bytes):
        raise RuntimeError(
            "runtime RRTMGP workspace drifted from the memory ledger: "
            f"{radiation_workspace.nbytes} != {estimate.workspace_bytes} bytes")
    ledger = ModelMemoryLedger(
        estimate=estimate,
        shared_scratch_arena_bytes=(0 if arena is None else arena.nbytes),
        shared_dycore_state_workspace_bytes=(
            0 if dycore_state_workspace is None
            else dycore_state_workspace.nbytes),
        radiation_workspace=radiation_workspace)

    # Numbered among the grids that start with the run, the grid id said
    # beside it: a dormant (spawn-declared) nest ahead of an active one
    # made the id read as "grid 3 of 2".
    grid_count = len(active.domains)
    grid_place = {int(dc.grid_id): place
                  for place, dc in enumerate(active.domains, start=1)}

    def grid_step(grid_id: int):
        return prep_stage("domain_initialize",
                          label="Start state and boundaries",
                          index=grid_place[int(grid_id)], count=grid_count,
                          grid_id=int(grid_id), stderr=False)

    with grid_step(exp.root.grid_id):
        prepared_root = runtime.prepare_root_experiment_case(
            exp, case_data, input_catalog=catalog,
            forcing_by_time=snapshots, scratch_arena=arena,
            dycore_state_workspace=dycore_state_workspace)
    # The raw decode is spent: the root's initial state and every
    # boundary frame are built from it above, and nothing below reads it.
    # The children about to be initialized -- and the mid-run delayed
    # start, spawn and relocation initializers that reach the same door
    # later -- take their source snapshot from the CATALOG
    # (woof.ingest.nest_init._initial_snapshot), which holds its own
    # frozen copies and is untouched here.  Held to the end of the run
    # this made the ERA5 decode, not the forecast, the host-memory
    # binding phase: the catalog's copies, a SECOND frozen set under the
    # runtime's own cache key, and the partials both were copied from.
    # The mapping is dropped as well as the caches -- a cleared cache
    # frees nothing while a local still names the arrays.  Byte-inert:
    # these caches memoize a pure function of immutable input bytes, so
    # the only thing a later decode of the same key loses is time.
    del snapshots
    clear_forcing_caches()
    root_dc = exp.root
    root = DomainNode(
        cfg=root_dc, grid=prepared_root.grid,
        state=prepared_root.initial_result.state,
        clock=clocks[root_dc.grid_id], parent=None, children=[], coupler=None)
    root._started = True
    # Davies clock bind (seam closure 2026-07-28, retires the F20
    # adjudication): the root's external Davies launches consume WRF's
    # post-increment dtbc.  WRF resets dtbc on every boundary read
    # (share/mediation_integrate.F:1515-1522) and adds one model step at
    # solve entry BEFORE any relax/spec consumer (dyn_em/solve_em.F:
    # 371-372); the executor's on_lbc_reset + prepare_step already runs
    # that exact recurrence on the root DomainClock, so binding here --
    # after attachment, before any solve or restart restoration -- makes
    # every root boundary launch take dtbc_launch_fp32 (dt..T_bdy) in
    # place of the retired one-step-lagged elapsed-based value (0..T-dt).
    # The former frozen-Phase-4 anchor bytes encode the retired
    # semantics; the N-series ratchets regenerate against the seam-
    # closure anchor epoch (PROVENANCE.md "Root external-boundary dtbc").
    bind_lateral_boundary_clock(root.state, root.clock)
    nodes: dict[int, DomainNode] = {root_dc.grid_id: root}
    prepared_by_id = {root_dc.grid_id: prepared_root}

    from woof.core.cam_ozone import configure_cam_ozone
    configure_cam_ozone(root.state, root_dc.run, exp=exp, dc=root_dc, grid=root.grid)
    root_radiation = (root.state.physics.radiation_callable
                      if root.state.physics is not None else None)
    from woof.core.radiation_composition import attach_modern_workspace
    attach_modern_workspace(root_radiation, radiation_workspace)

    initial_perturbation_receipts = []
    if exp.perturbation is not None:
        initial_perturbation_receipts.append(
            dict(prepared_root.initial_result.initial_perturbation))

    for dc in active.domains[1:]:
        parent = nodes[dc.parent_id]
        # A child that starts WITH the experiment applies the configured
        # initial-state bubbles on its own grid (real-data nest init
        # re-ingests the source per domain, so nothing arrives from the
        # parent's theta).  A delayed-start child initializes from the
        # analysis at its activation time -- the parent has evolved the
        # bubble by then -- so it takes no fresh analytic bubble.
        with grid_step(dc.grid_id):
            child_starts_at_t0 = clocks[dc.grid_id].spec.start_ticks == 0
            initialized = initialize_child(
                dc, parent, catalog, exp.vertical,
                source_orography=case_data.source_orography,
                scratch_arena=arena,
                dycore_state_workspace=dycore_state_workspace,
                sfcp_to_sfcp=case_data.sfcp_to_sfcp,
                initial_perturbation=(
                    exp.perturbation if child_starts_at_t0 else None))
            if exp.perturbation is not None:
                initial_perturbation_receipts.append(
                    dict(initialized.real.initial_perturbation)
                    if child_starts_at_t0 else {
                        "grid_id": int(dc.grid_id),
                        "applied": False,
                        "reason": "delayed start: this domain initializes "
                                  "from the analysis at its activation time, "
                                  "after the perturbation instant",
                    })
            prepared = runtime.prepare_child_case(
                initialized, dc, exp=exp, data=case_data,
                forcing_times=forcing_times,
                radiation_workspace=radiation_workspace,
                # Legacy-RRTMG ozone routing (WRF computes o33d on the root
                # domain only; children receive parent-interpolated fields):
                # the child's adapter takes the parent's radiation callable
                # as its ozone provider.  Inert for every other variant.
                radiation_parent=parent.state.physics.radiation_callable)
        node = DomainNode(
            cfg=dc, grid=initialized.grid, state=initialized.state,
            clock=clocks[dc.grid_id], parent=parent,
            children=[], coupler=None)
        node.coupler = ConcreteNestCoupler(
            node, feedback=exp.feedback,
            smooth_option=exp.smooth_option)
        node._started = clocks[dc.grid_id].spec.start_ticks == 0
        parent.children.append(node)
        # The marker is present both before and after FORCE.  It makes the
        # restart setup fingerprint independent of rolling table contents.
        node.state._nest_restart_classification = "REBUILT"
        nodes[dc.grid_id] = node
        prepared_by_id[dc.grid_id] = prepared
        if exp.feedback == 1 and node._started:
            initial = FeedbackScratch()
            node.coupler.feedback_prepare(node, initial)
            node.coupler.feedback_commit(node, start=True)
            node.coupler.feedback_finalize(node, start=True)

    built = ExperimentState(
        root=root, nodes_by_grid_id=MappingProxyType(nodes),
        schedule=schedule, memory_ledger=ledger,
        experiment_fingerprint=experiment_fingerprint(exp, catalog))
    # Keep the wave-1 dataclass contract stable: Task-14 runtime machinery is
    # attached as non-field infrastructure, so T12/T13 consumers remain inert.
    built._scratch_arena = arena
    built._dycore_state_workspace = dycore_state_workspace
    built._prepared_by_grid_id = prepared_by_id
    built._initial_perturbation_receipts = tuple(
        initial_perturbation_receipts)
    built._input_catalog = catalog
    publish_declared_experiment(built, exp)
    built._activation_context = {
        "experiment": exp,
        "case_data": case_data,
        "forcing_times": forcing_times,
        "radiation_workspace": radiation_workspace,
    }
    built._runtime_status = ModelRuntimeStatus()
    built._feedback_provenance = runtime.feedback_provenance(exp)
    built._resumed = False
    built._resume_committed_history_grid_ids = frozenset()
    built._io_manager = None
    built._last_checkpoint = None
    return built


_POOL_TRIM_DEFAULTS = {
    "win32": (True, "Release unused blocks to avoid measured WDDM page demotion."),
}
_POOL_TRIM_FALLBACK = (
    False, "Retain cached blocks so other processes cannot claim them between steps; "
    "no trimming speed-up is established on this platform.")


def _resolve_pool_trim_policy(requested: bool | None) -> dict[str, object]:
    """Select the measured platform default or an allocator diagnostic override."""
    default, reason = _POOL_TRIM_DEFAULTS.get(sys.platform, _POOL_TRIM_FALLBACK)
    return {
        "schema": "gpuwm-pool-trim-policy-v1",
        "platform": sys.platform,
        "release_unused_blocks": default if requested is None else bool(requested),
        "cadence": "after each step and period",
        "selection": "platform-default" if requested is None else "explicit-override",
        "reason": reason if requested is None else "Explicit allocator diagnostic override. " + reason,
        "windows_measurement": (
            "2026-07-17, four domains, five simulated minutes: 32% less wall time; "
            "pool held at 21.5-23.4 GiB. See docs/da-ensemble-parallel.md."),
    }


def _trim_default_pool() -> None:
    """Release UNUSED cached CuPy pool blocks back to the driver.

    Live allocations are untouched, so this cannot change any computed
    value (see execute_experiment's pool_trim_per_period).

    THE BREAKAGE THE GUARD PREVENTS.  A trim is an optimisation, never a
    correctness step, but it runs at every step boundary -- so when the
    pool itself cannot be reached, raising here aborts a run that never
    needed a device.  That is reachable with CuPy INSTALLED and every
    device masked (``CUDA_VISIBLE_DEVICES=-1``, which tests/conftest.py
    plants for the whole session): the first pool call answers
    ``cudaErrorNoDevice`` and killed host-only nest runs at their first
    step boundary.  An unreachable pool leaves the cache where it is,
    which is the same state a trim that freed nothing would leave.
    """
    import cupy as cp
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except cp.cuda.runtime.CUDARuntimeError:
        return


def _ask_the_checkpoints_question(node) -> None:
    """Ask this domain's checkpoint physics identity before its first step.

    The rule, the reason and the ``PhysicsDriver``-only restriction live in
    :func:`woof.io.restart.ask_checkpoint_physics_identity`, which the
    other forecast door (``woof.runtime.integrate_prepared_case``) calls
    too -- one function, so the two doors cannot answer the same question
    differently (audit R-046).
    """
    from woof.io.restart import ask_checkpoint_physics_identity

    ask_checkpoint_physics_identity(node.state, node.cfg.run)


#: The activation price's terms, in the words a refusal prints.
_ACTIVATION_TERM_NAMES = (
    ("model_state", "model state"), ("physics", "physics"),
    ("forcing_analysis", "forcing analysis on the nest's grid"),
    ("vertical_setup", "vertical setup"),
    ("setup_residual", "setup temporaries"),
    ("pool_headroom", "allocator headroom"))


def _admit_nest_activation(node, catalog, exp, *, streamed_owner=None):
    """``woof run``'s activation door: a delayed nest's rebuild, priced first.

    On this route a delayed nest is initialized again at its start from the
    catalog's analysis at that time (:func:`woof.ingest.nest_init
    .initialize_child`): a new state and physics driver, the analysis
    interpolated onto the nest's grid, and the vertical setup beside them.
    The run's admission prices the steady tree, and no admission priced
    this rebuild, so a tree that fitted its steady state could stop in a
    CUDA out-of-memory at the nest's start, hours into the forecast.  It is
    priced here from the DECODED analysis the rebuild reads
    (:func:`woof.ingest.preparation_price.price_nest_activation`) against
    what the card has free once the startup build is released, and refused
    by name before the rebuild allocates.  The figure is an upper bound, so
    ``--no-memory-gate`` skips it.  MEASURED on an RTX 5070 Ti, a 3 km 168
    x 132 x 49 nest starting 6 h into a 9 km ERA5 root: priced 0.48 GiB,
    the rebuild itself peaked 0.23 GiB over the released startup build.

    ``streamed_owner`` is the stepper a streamed nest is re-attached with
    after the rebuild.  Its replacement tile buffers are allocated inside
    this same activation, from bytes the free figure counts because the
    close handed the old buffers back, so their claim is priced here too
    (:func:`woof.core.streaming.reattach_claim_terms`): on that nest the
    re-attachment took the peak to 0.85 GiB, over the 0.48 GiB the rebuild
    alone was priced at.

    The shared admission estimate carries no such term: the prepared route
    restores its delayed nests from their prepared caches instead.  A
    catalog with no analysis at the nest's start prices nothing here, and
    ``initialize_child`` refuses it by name.
    """
    import math

    from woof.core.ozone_contract import cam_ozone_domain_ids
    from woof.core.preflight import physics_array_shapes
    from woof.core.resident_admission import admit
    from woof.ingest.nest_init import _initial_snapshot
    from woof.ingest.preparation_price import price_nest_activation

    dc = node.cfg
    try:
        snapshot = _initial_snapshot(catalog, dc.start_time)
    except ValueError:
        return None
    physics = sum(4 * math.prod(shape) for shape in physics_array_shapes(
        dc.run, cam_ozone=int(dc.grid_id) in cam_ozone_domain_ids(exp)
    ).values())
    price = price_nest_activation(dc.run, snapshot, physics_bytes=physics)
    terms = {text: int(price.terms.get(key, 0))
             for key, text in _ACTIVATION_TERM_NAMES}
    if streamed_owner is not None:
        from woof.core.streaming import reattach_claim_terms
        terms.update(reattach_claim_terms(streamed_owner, node))
    return admit(
        f"nest d{int(dc.grid_id):02d} starting at "
        f"{exp.domain_start_time(dc.grid_id):%Y-%m-%d %H:%M}, rebuilt from "
        "the analysis at its start",
        terms, stage=price.stage, envelope=True,
        remedy=("start the nest with the forecast, so it is built with the "
                "tree the run was admitted on; free card memory (close "
                "other programs using the GPU); or use a smaller nest"))


def _release_startup_build(model, node, validators, *,
                           streamed_owner=None) -> None:
    """Drop every owner of a delayed child's startup build before its rebuild.

    ``build_experiment`` gives a delayed child a complete state and a
    prepared physics driver at t = 0 (the checkpoint question and the
    health validators ask them there), and activation builds a SECOND
    complete child from the analysis at its start time.  While the node's
    state, the prepared-case map and the domain's health validator still
    named the first build, both were on the card together for the whole
    rebuild, so a tree that fit its steady state ran out of device memory
    at the child's activation: one extra child state and physics driver
    over everything admission priced.  MEASURED on an RTX PRO 4500, a 240
    x 216 3 km HRRR tree with a 180 x 180 1 km nest starting an hour in
    (`woof go`): the card peaked at 4,899 MiB with both builds held and
    at 4,337 MiB with this release, every history file byte-identical.
    The 288 MiB activation still adds is the nest's first-step transients,
    which the forecast admission prices (282 MiB), so on that route the
    transition now sits inside what the steady tree is admitted on.

    The release is the one relocation performs on an outgoing child
    (:func:`woof.core.nest_relocation.release_state_arrays`): every array
    reference the state holds is dropped, and the cumulus adapter breaks
    its reference cycle with the driver, so the allocator hands the same
    bytes back to the rebuild.  The emptied state object stays on the node
    until the rebuild replaces it, so nothing that walks the tree meets a
    missing state.  A streamed startup build that a route restores into a
    new host store is left to that route's initializer, which closes and
    rebinds its own stepper and lets its outgoing store go (the prepared
    domain-tree forecast).  That startup state is a
    :class:`woof.core.streamed_state.CanonicalStoreState`, a view of the
    store that refuses every resident read, its radiation included, so
    nothing here reads its physics: that route builds a nest's startup
    radiation from its prepared cache with no
    :class:`woof.core.rrtmg_legacy.ParentOzoneProvider`, so no waiting
    child reads it either.
    ``streamed_owner`` is the stepper of a streamed child the DEFAULT road
    rebuilds resident (``woof run``): that road attached the stepper to a
    resident startup build, so its tile buffers are closed here and the
    startup state's arrays released like a resident child's, and
    :func:`woof.core.streaming.reattach_rebuilt_domain` binds it to the
    rebuilt state once that exists.
    The collector runs once here, for this activation only, so a cycle no
    release contract names cannot keep the first build alive either.
    """
    import gc

    from woof.core.nest_relocation import release_state_arrays
    from woof.core.streamed_state import CanonicalStoreState

    grid_id = int(node.cfg.grid_id)
    validators.pop(grid_id, None)
    model._prepared_by_grid_id.pop(grid_id, None)
    # Decided before any read of the startup state: a canonical host view
    # refuses a read of its physics with CanonicalStateRefused, and its
    # route restores it (see the docstring).
    restored_by_route = isinstance(node.state, CanonicalStoreState)
    if restored_by_route:
        streamed = True
    else:
        try:
            streamed = node.state._streamed_domain is not None
        except AttributeError:
            streamed = False
        try:
            outgoing = node.state.physics.radiation_callable
        except AttributeError:
            outgoing = None
        if outgoing is not None:
            # A legacy-RRTMG child still waiting for its start reads its
            # ozone through this outgoing radiation, and that link alone
            # would keep it alive until the child starts.
            from woof.core.rrtmg_legacy import release_waiting_ozone_links
            release_waiting_ozone_links(
                outgoing,
                [child for child in node.children if not child._started])
    if streamed_owner is not None:
        from woof.core.streaming import release_outgoing_store
        streamed_owner.tiled_run.close()
        release_outgoing_store(streamed_owner)
        streamed = False
    if not streamed:
        node.coupler = None
        release_state_arrays(node.state)
        # Scratch slots sit in a dict, which the array walk above does not
        # enter; a slot outside the shared arena owns its bytes.
        try:
            node.state._scratch.clear()
        except AttributeError:
            pass
    gc.collect()


def _bind_execution_physics_params(model, experiment):
    """Require a built tree to retain the constants it was built under.

    Construction may bind a set; execution cannot first select one because
    the existing state or parameter tables may already contain defaults.
    The published declaration also prevents re-labelling that state with
    another set even when no affected kernel has compiled yet.
    """
    from woof.experiment import _bind_physics_params
    from woof.physics_params import active, document, environment_set

    try:
        declared = model._declared_experiment
    except AttributeError:
        declared = None
    if declared is None:
        try:
            context = model._activation_context
        except AttributeError:
            context = None
        if context is not None:
            declared = context.get("experiment")
    authority = experiment if experiment is not None else declared
    from_env = environment_set()
    before = active()
    if authority is None:
        if before is not None or from_env is not None:
            raise ValueError(
                "execute_experiment received a built tree without its "
                "physics parameter declaration while a set is selected; "
                "its existing state cannot be labelled with those constants. "
                "Pass the original experiment used to construct the tree.")
        return None
    try:
        requested = authority.physics_params
    except AttributeError:
        requested = None
    if requested is None:
        requested = from_env
    if requested is not None and (
            before is None or requested.sha256() != before.sha256()):
        raise ValueError(
            "execute_experiment cannot first apply or switch a physics "
            "parameter set on an already-built tree: its state, kernels "
            "or tables may contain other constants. Select the set before "
            "constructing the tree in its own process.")
    if declared is not None:
        try:
            original = declared.physics_params
        except AttributeError:
            original = None
        if document(original) != document(requested):
            raise ValueError(
                "execute_experiment's physics parameter set differs from "
                "the built tree's declaration; re-labelling its existing "
                "state would bind checkpoints to constants it did not run.")
    bound = _bind_physics_params(authority, "woof.core.model.execute_experiment")
    if requested is not None:
        publish_declared_experiment(model, bound)
        return bound
    # Preserve the existing explicit-argument/context fallback for every
    # other execution feature when parameter binding is absent.
    return experiment


def execute_experiment(
        model: ExperimentState, *, history_handler=None,
        restart_handler=None, progress_callback=None,
        validate_state: bool = True, health_debug: bool = False,
        arena_nan_poison: bool = False,
        skip_feedback_path: bool = False,
        pool_trim_per_period: bool | None = None,
        relocation_runner=None, steppers=None, step_observer=None,
        experiment=None, delayed_child_initializer=None, schedule_dispatch=None):
    """Wire one :class:`ExperimentState` into ``execute_schedule``.

    STEP calls the domain's existing dycore, FORCE calls its node-facing
    coupler, and FEEDBACK walks the dormant three-phase transaction.  The
    clock module remains the only op-table walker and therefore the only
    source of runtime scheduling comparisons.

    ``relocation_runner`` is a
    :class:`woof.core.relocation_runner.RelocationRunner` constructed by
    the route (the route owns the child-physics preparer and, for
    ``follow = "provider"``, the live tracker).  It is consulted at every
    PERIOD_BEGIN -- the complete cycle boundary where all clocks are
    synchronized, so every relocation lands on a parent-step boundary --
    and a moved child gets a fresh health validator armed immediately.  A
    configuration that schedules relocation on a route that did not wire a
    runner refuses here, at start, rather than integrating a nest that
    silently never follows anything.

    ``pool_trim_per_period=None`` selects the platform default: release
    UNUSED cached blocks after each step and period on Windows, retain
    them elsewhere. On a shared Linux card, returning those blocks lets
    competing processes claim memory the next radiation step needs.
    Measured on Windows with four domains (2026-07-17 5-sim-min A/B): churn re-inflates
    the pool from ~21 to ~30 GiB held against ~20 GiB live, driving WDDM
    page demotion; per-period trimming held the pool at 21.5-23.4 GiB,
    kept the device below the demotion band, and cut wall time 32%.
    Byte-inert by construction -- free_all_blocks releases only unused
    cached blocks, never live allocations, so no computed value can
    change; allocator behavior is not part of any ratified comparator.
    Explicit booleans remain available for allocator forensics. The resolved
    rule and its reason are carried by the model into the run receipt.

    ``steppers`` is ``{grid_id: callable}``, the callable a STEP op steps
    that domain with.  ``None`` -- and any grid absent from a supplied
    mapping -- binds ``woof.core.dycore.step`` ITSELF, so an experiment
    that configures nothing runs the identical function through the
    identical call, not a wrapper that forwards to it.  A domain whose
    ``[tiles]`` fired binds a
    :class:`woof.core.streaming.StreamedDomain` instead, which has the
    same signature and advances the same domain by the same one model
    step -- out of a pinned host store, one tile of it on the card at a
    time.  Everything around the call is unchanged BECAUSE the loop is
    unchanged: the same clock refresh either side, the same health
    validators, the same ``refl_10cm_due`` handshake, the same history
    and restart ops on the same cadences.

    ``delayed_child_initializer`` optionally supplies a route's dated
    child analysis at activation. It receives the child node and exact
    clock and returns ``(initialized, prepared)`` with the same grid/state
    and prepared-case attributes as the ordinary case-data initializer.
    The executor still owns the clock refresh, coupler, initial feedback,
    health validation and activation marker. Omission uses the input catalog.

    ``step_observer`` is called once per DOMAIN per model time step,
    immediately after that domain's STEP op commits, with
    ``grid_id``/``step_count``/``model_seconds``/``step_wall_seconds``.
    It is what lets a front door print WRF's ``Timing for main:`` line
    per step (:class:`woof.progress_log.StepLog`); ``progress_callback``
    below cannot, because it fires once per ROOT step and a nest taking
    36 substeps inside one of those is invisible to it.  ``None`` -- the
    default -- adds one ``is not None`` test per STEP op and changes
    nothing else.

    ``schedule_dispatch`` is an internal capture/dispatch seam for a
    qualified member scheduler. It receives this executor's actual
    callbacks and adaptive controller after setup. Omission calls the
    original integer executor directly.
    """
    experiment = _bind_execution_physics_params(model, experiment)
    # The host thread WAITS for the card (blocking sync) instead of
    # spinning a core at 100 % for the whole step loop (woof/cuda_wait.py).
    # Host-side only: no kernel, order or value changes.
    from woof.cuda_wait import apply as _apply_cuda_wait
    _apply_cuda_wait()
    # A [radar_heating] table is honored by the route that attached it or
    # refused here; a route that never attaches it would integrate an
    # unheated forecast under the name of a heated one.
    from woof.da.forecast_heating import require_routed
    require_routed(model, experiment)
    from woof.core.restart_request import RestartRequest
    restart_request = RestartRequest()
    requested_restart = [False]
    from woof.core.clock import execute_schedule
    from woof.core.dycore import step
    from woof.core.state import refresh_model_time

    model._pool_trim_policy = _resolve_pool_trim_policy(pool_trim_per_period)
    pool_trim_per_period = model._pool_trim_policy["release_unused_blocks"]
    status = model._runtime_status
    context = model._activation_context
    if relocation_runner is None and context is not None:
        exp = context.get("experiment")
        relocation = None if exp is None else exp.relocation
        # A follow target that is still DORMANT is not a void arm: the
        # nest does not exist yet, so there is nothing to move and no
        # grid to anchor a runner on.  The leg walk
        # (woof.runtime.walk_spawn_legs) wires the runner at the birth
        # boundary, which is the first instant that nest could legally
        # move -- so the refusal is DEFERRED here, not waived.  A target
        # that is present in the tree still refuses exactly as before.
        target = None if relocation is None else relocation.grid_id
        # Attribute access in a try, never reflection: this module is
        # AST-audited against getattr/setattr
        # (tests/test_clock.py::test_no_float_elapsed_accumulation_audit),
        # and the only callers that lack these attributes are the stub
        # models in the refusal tests, for which no carve-out applies.
        try:
            dormant_target = (
                target is not None
                and int(target) not in model.nodes_by_grid_id
                and exp is not None
                and any(dc.grid_id == int(target) and dc.spawn is not None
                        for dc in exp.domains))
        except AttributeError:
            dormant_target = False
        if (relocation is not None and relocation.enabled
                and not dormant_target
                and (relocation.follow is not None or relocation.moves)):
            source = ("[relocation.follow]" if relocation.follow is not None
                      else "[[relocation.move]]")
            raise RuntimeError(
                f"[relocation] configures a follow source ({source}) but "
                "this route wired no RelocationRunner; a nest that "
                "silently never moves is a void experiment arm, so this "
                "refuses instead.  The case-data route wires the real-data "
                "runner (woof.runtime.build_real_relocation_runner: "
                "footprint-rebuilt statics + the land-surface preparer); "
                "any other route constructs "
                "woof.core.relocation_runner.RelocationRunner and passes "
                "it here, or drives relocate_child directly")
    # Resolved to a plain dict here, keyed by grid, so a delayed-start child
    # that joins the tree later falls back to the dycore's own step rather
    # than to whatever the last domain happened to bind.
    steppers = dict(steppers or {})
    #: Grid ids the checkpoint's question has already been asked of, so a
    #: domain is asked once and no later than it can be.
    asked_the_checkpoints_question: set[int] = set()
    if restart_handler is not None:
        # THE CHECKPOINT'S QUESTION, ASKED BEFORE STEP 0, for every domain
        # of the tree.  ``restart_handler is not None`` is exactly "this
        # run will write checkpoints", and until audit R-046 the first
        # thing that asked whether a checkpoint could NAME this physics
        # setup was the writer, at the first restart interval, with the
        # forecast to that point already spent.  Plan review answers the
        # config half (woof.physics_registry.require_consumer_rows, from
        # validate_run_config); the driver half -- a physics callable
        # whose class a stock-class row cannot bind, a land-surface
        # parameter bundle that cannot be digested -- needs the
        # constructed driver, which exists here and nowhere earlier.  The
        # identity is discarded: what this buys is the refusal's
        # placement, not the value.
        #
        # EVERY CONFIGURED DOMAIN IS IN THIS WALK, delayed-start children
        # included: build_experiment gives each configured child a state
        # and a prepared driver and only withholds ``_started``, and
        # walk_parent_first yields the whole tree.  So this is where a
        # delayed child's refusal happens too -- at t=0 rather than hours
        # later at its activation, which is the earlier of the two and
        # the one worth having.
        for node in model.walk_parent_first():
            _ask_the_checkpoints_question(node)
            asked_the_checkpoints_question.add(int(node.cfg.grid_id))
    # THE LEVEL-2 SPECTRAL SEAM (woof.spectral_seam).  ``None`` -- every
    # experiment without an active [spectral_numerics] -- costs the STEP
    # op one ``is not None`` test and nothing else.  Active, it refuses a
    # streamed domain and a false periodic declaration HERE, at start,
    # and the STEP op below calls it once per domain immediately after
    # the slow RK state commit.  Attached to the model so leg walks
    # (spawn, restart) that re-enter this function keep one receipt
    # ledger for the whole run.
    # ``experiment`` is the AUTHORITY here and the activation context is
    # only the fallback, because the two prepared routes -- the ones the
    # Level-2 contract names as honoring the config -- build their own
    # ``ExperimentState`` and therefore have no activation context at
    # all.  Reading the context alone made an active [spectral_numerics]
    # invisible on exactly those routes: a shadow tree run on real HRRR
    # bytes wrote zero receipts and still emitted a clean PASS capsule
    # (2026-08-18), which is the silently-absent operator the
    # honored-or-refused governance exists to forbid.
    from woof.spectral_seam import attach_seam
    seam_experiment = experiment
    if seam_experiment is None and context is not None:
        seam_experiment = context.get("experiment")
    spectral_seam = attach_seam(model, seam_experiment, steppers)
    nudging = getattr(seam_experiment, "nudging", None)
    if nudging is not None:
        for node in model.walk_parent_first():
            if (nudging.for_domain(node.cfg.grid_id) is not None
                    and getattr(node.state, "_spectral_nudging_owner", None) is None):
                raise ValueError("NUDGING_ANALYSES_UNBOUND: this domain has no prepared analysis forcing; use the prepared forecast route")
    feedback_scratch = {
        node.cfg.grid_id: FeedbackScratch()
        for node in model.walk_parent_first() if node.parent is not None}
    arena = model._scratch_arena
    dycore_state_workspace = model._dycore_state_workspace
    io_manager = model._io_manager
    restart_enabled = model.root.clock.spec.restart_ticks is not None

    validators = {}
    if validate_state:
        # A streamed domain lives in its canonical store, even when preparation
        # began resident. Bind the existing health rules to that store;
        # unstreamed domains keep the existing resident validator.
        from woof.core.health import health_validator_for_domain
        validators = {node.cfg.grid_id: health_validator_for_domain(model, node)
                      for node in model.walk_parent_first()}
        # Indexed, not bound to a loop name: a loop name outlives the loop
        # for the whole run, and the last validator it named is a delayed
        # child's startup build when that child comes last, which kept the
        # build alive past its activation (see _release_startup_build).
        for gid in validators:
            validators[gid].require_healthy(phase="initialized-or-restored")

    def poison() -> None:
        if arena_nan_poison and arena is not None:
            arena.poison()

    def domain_turn(owner):
        if dycore_state_workspace is None:
            return nullcontext()
        return dycore_state_workspace.acquire(owner)

    def _streamed(grid_id):
        """The streamed stepper for one grid, or ``None`` if it is resident.

        ``steppers`` holds ``dycore.step`` for nothing -- a resident grid is
        simply absent -- so this is the whole test.
        """
        from woof.core.streaming import is_streaming

        stepper = steppers.get(int(grid_id))
        return stepper if is_streaming(stepper) else None

    def allocation_scope(grid_id):
        stream = _streamed(grid_id)
        return nullcontext() if stream is None else stream.allocation_scope()

    def on_period_begin(period, clocks) -> None:
        status.schedule_cursor = PERIOD_BEGIN
        status.prior_feedback_committed = True
        if relocation_runner is None:
            return
        # A STREAMED parent keeps its arrays in a store, not on its state,
        # and the whole of relocation reads the state: the tracker reduces
        # the parent's UH plane, the runner zeroes that plane, and the
        # rebuild takes a full SINT of the live parent.  With a host store
        # every one of those reads t=0 unless the store is projected onto
        # the state first -- and the zeroing has to be projected BACK, or
        # the window silently stops meaning "since I last looked".
        #
        # Only on a cadence boundary, and only for the target's parent.  The
        # planes are (ny, nx); the SINT source is the whole state, which is
        # why it is done here (once per cadence) and not per step.
        from woof.core.streaming import TRACKER_PLANE_CARRIERS

        parent_streams = {}
        # Bare runner stubs in the refusal controls predate collections.
        try:
            collection = bool(relocation_runner.is_collection)
        except AttributeError:
            collection = False
        due_targets = ()
        if collection:
            due_targets = tuple(
                gid for gid in relocation_runner.target_grid_ids
                if gid in model.nodes_by_grid_id
                and relocation_runner.runners[gid].is_due(model, clocks))
        elif relocation_runner.is_due(model, clocks):
            due_targets = (int(relocation_runner.config.grid_id),)
        for gid in due_targets:
            target = model.node(gid)
            runner = relocation_runner.runners[gid] if collection else relocation_runner
            try:
                reconstruction_factory = runner.streamed_reconstruction_factory
            except AttributeError:
                reconstruction_factory = None
            store_rebuild = (_streamed(gid) is not None
                             and callable(reconstruction_factory))
            if _streamed(gid) is not None and not store_rebuild:
                raise RuntimeError(
                    f"[relocation] targets grid {gid}, which is STREAMED. "
                    "Relocation must rebuild its store, tile plan, geography "
                    "gathers and packed nest-table windows before replacing "
                    "the live stepper; that reconstruction is not implemented. "
                    "Run the moving child resident (the parent may stream).")
            if target.parent is not None and not store_rebuild:
                parent_gid = int(target.parent.cfg.grid_id)
                parent_stream = _streamed(parent_gid)
                if parent_stream is not None and parent_gid not in parent_streams:
                    # The tracker reads its canonical plane directly; an
                    # accepted move also needs the existing full-state donor.
                    parent_stream.sync_to_state()
                    parent_streams[parent_gid] = parent_stream

        def validate_rebuilt_child(gid):
            if validate_state:
                from woof.core.health import health_validator_for_domain
                validators[gid] = health_validator_for_domain(model, model.node(gid))
                validators[gid].require_healthy(
                    phase=f"post-relocation.d{gid:02d}")

        relocation_runner.on_period_begin(
            model, clocks, period=period,
            # The executor's validator holds live references into the
            # outgoing child; dropping it here is what lets the
            # host-staged release actually free the device bytes.
            before_rebuild=lambda gid: validators.pop(gid, None),
            after_rebuild=validate_rebuilt_child)
        for parent_stream in parent_streams.values():
            # reset_tracker_window already zeroes every live copy, including
            # generated follower carriers. Preserve the legacy fixed-plane
            # synchronization for providers that write those planes directly.
            parent_stream.sync_from_state(TRACKER_PLANE_CARRIERS)

    def on_domain_start(grid_id, clock) -> None:
        """Run the ordinary WRF-order child init at its delayed boundary."""
        from woof.core.nest import NestCoupler as ConcreteNestCoupler
        from woof.ingest.nest_init import initialize_child
        from woof.runtime import prepare_child_case

        node = model.node(grid_id)
        if node.parent is None:
            raise RuntimeError("the root domain cannot have a delayed start")
        if bool(node._started):
            return
        context = model._activation_context
        exp = context["experiment"]
        # Deliberately no initial_perturbation here: a delayed-start
        # child initializes from the analysis at its ACTIVATION time,
        # after the perturbation instant, and the receipts already say
        # so (build_experiment's delayed-start row).
        #
        # A STREAMED child on the default road (`woof run`) was attached
        # from a resident startup build, and the rebuild below replaces
        # that state: its stepper is closed here, before the rebuild
        # allocates, and re-attached to the rebuilt state after it.  A
        # route with its own initializer rebinds its own owner.
        rebound = (_streamed(grid_id) if delayed_child_initializer is None
                   else None)
        _release_startup_build(model, node, validators,
                               streamed_owner=rebound)
        if delayed_child_initializer is None:
            data = context["case_data"]
            # After the release above, so the card's free figure includes
            # the startup build this rebuild replaces.
            _admit_nest_activation(node, model._input_catalog, exp,
                                   streamed_owner=rebound)
            initialized = initialize_child(
                node.cfg, node.parent, model._input_catalog, exp.vertical,
                source_orography=data.source_orography,
                scratch_arena=arena,
                dycore_state_workspace=dycore_state_workspace,
                sfcp_to_sfcp=data.sfcp_to_sfcp)
            prepared = prepare_child_case(
                initialized, node.cfg, exp=exp, data=data,
                forcing_times=context["forcing_times"],
                radiation_workspace=context["radiation_workspace"],
                radiation_parent=(
                    node.parent.state.physics.radiation_callable))
        else:
            initialized, prepared = delayed_child_initializer(node, clock)
        node.grid = initialized.grid
        node.state = initialized.state
        node.coupler = ConcreteNestCoupler(
            node, feedback=exp.feedback,
            smooth_option=exp.smooth_option)
        refresh_model_time(node.state, clock)
        node.state._nest_restart_classification = "REBUILT"
        node._started = True
        # No checkpoint question here.  The pre-step-0 walk above asked it
        # of every configured domain, delayed-start children included, and
        # this function is reached only from the clock's delayed-start
        # boundary (woof/core/clock.py) for exactly those children; a node
        # constructed anywhere else (a restart restore, a relocation
        # rebuild) is marked started by its own constructor and returns at
        # the top of this function.  A guarded second ask stood here for a
        # domain "joining the tree after the start", and no path delivers
        # one: the guard was never false and the ask never ran.
        model._prepared_by_grid_id[grid_id] = prepared
        if exp.feedback == 1:
            initial = FeedbackScratch()
            node.coupler.feedback_prepare(node, initial)
            node.coupler.feedback_commit(node, start=True)
            node.coupler.feedback_finalize(node, start=True)
        if rebound is not None:
            # After the feedback seed, as build_experiment's t = 0 children
            # are seeded before the route attaches their steppers.
            from woof.core.streaming import reattach_rebuilt_domain
            reattach_rebuilt_domain(rebound, node)
        from woof.ensemble.runtime_context import bind_reconstructed_member_node
        bind_reconstructed_member_node(node, prepared_case=prepared)
        if validate_state:
            from woof.core.health import health_validator_for_domain
            validators[grid_id] = health_validator_for_domain(model, node)
            validators[grid_id].require_healthy(
                phase=f"delayed-start.d{grid_id:02d}")

    #: Host wall each domain's STEP ops have taken since the run began,
    #: summed per grid.  Forwarded on every period commit so a reader can
    #: say which grid of a nested run sets its pace; the same host-side
    #: pair `step_observer` reports, never a device synchronise.
    step_wall_by_domain: dict[int, float] = {}

    # Scheme health status words (microphysics, YSU, KF, RRTMGP, ...) are
    # recorded on the device during the step and read ONCE when the step
    # returns, through woof.core.health_ledger, instead of one blocking
    # device-to-host read per scheme in the middle of the step.  Each of those
    # reads drained the launch queue: the host waited for the card, then the
    # card waited for the host's next launches.  Measured on a 6 h Boston
    # 250 m run (2.8.5, one RTX PRO 6000): the microphysics read alone held
    # the main thread 9.4 ms per domain step with the card about 50% busy.
    # Healthy runs change in nothing but wall time (the reads are error-only;
    # see health_ledger).  A sick step still raises the same exception, from
    # the same step, before step_health, output, feedback or a checkpoint can
    # observe its state.  A ledger lives on one device, so a run that sees
    # more than one keeps the immediate reads, as does --health-debug, whose
    # point is the immediate report.
    step_ledger = None
    if not health_debug:
        try:
            import cupy as _cp
            if int(_cp.cuda.runtime.getDeviceCount()) == 1:
                from woof.core.health_ledger import HealthLedger
                step_ledger = HealthLedger(label="forecast step")
        except Exception:  # noqa: BLE001 - no CUDA: immediate reads, as before
            step_ledger = None

    def on_step(grid_id, clock) -> None:
        # WRF prints one `Timing for main:` line per model time step per
        # domain, and THIS is the only place that number exists: the
        # period-commit callback below fires once per ROOT step, so a
        # nest taking 36 substeps inside one of them was invisible to
        # every progress consumer this package had.  One perf_counter
        # pair, host-side, deliberately WITHOUT a device synchronise --
        # syncing per step to get a "true" GPU time would serialise the
        # pipeline the timing exists to observe, and WRF's own number is
        # host wall time across the step's launches too.
        started_wall = time.perf_counter()
        with domain_turn(("STEP", grid_id)):
            node = model.node(grid_id)
            if adaptive_driver is not None:
                # After any relocation rebuild, before the solve.
                adaptive_driver.before_step(grid_id)
            if node.clock is not clock:
                raise RuntimeError(
                    "DomainNode clock identity drifted from executor")
            if node.parent is not None:
                if node.coupler is None or not bool(node.coupler.valid):
                    raise AssertionError(
                        f"child d{grid_id:02d} STEP attempted with INVALID "
                        "nest tables; ordinary parent STEP -> FORCE is "
                        "required")
            if validators and health_debug:
                validators[grid_id].require_healthy(
                    phase=f"pre-step.d{grid_id:02d}")
            status.schedule_cursor = "EXECUTING"
            # Kernel-facing curr_secs mirrors WRF REAL; after the solve the
            # public state clock is refreshed from exact integer ticks.
            refresh_model_time(node.state, clock, kernel_launch=True)
            # The clock is the CALENDAR AUTHORITY, and a streamed domain
            # does not read the state it was just written onto: its tiles
            # take elapsed_seconds from the sweep's scalar carriers.  Impose
            # it here, from the same value the resident kernels get, or the
            # domain runs a second free-running clock and every physics
            # cadence -- and dtbc -- is evaluated against it.
            streamed_here = _streamed(grid_id)
            if streamed_here is not None:
                streamed_here.impose_clock(node.state.elapsed_seconds)
            from woof.core import health_ledger as _health_ledger
            with _health_ledger.deferring(step_ledger):
                steppers.get(grid_id, step)(
                    node.state, node.cfg.run,
                    # REFL_10CM is a one-frame producer/consumer handoff. A
                    # headless forecast still advances history alarms in the
                    # clock report, but has no output consumer, so it must not
                    # stage a field that can never be consumed.
                    refl_10cm_due=(
                        history_handler is not None
                        and node.cfg.run.mp_physics in REFL_10CM_MICROPHYSICS
                        and clock.history_rings_within_step()),
                    **({"fire_history_due": clock.history_rings_within_step()}
                       if node.cfg.run.ifire == 2 else {}))
            if step_ledger is not None:
                # One read for every scheme site of this step; raises the
                # first site's own refusal.  Before anything downstream.
                step_ledger.drain()
            from woof.core.nest_round_trip import th_phy_m_t0_after_step
            th_phy_m_t0_after_step(node)
            refresh_model_time(node.state, clock, after_step=True)
            # Scientific accumulators see every committed domain step. This
            # is not the best-effort telemetry seam below: losing a wind
            # maximum must fail the product instead of silently undersampling.
            from woof.ensemble.runtime_context import current_capture
            capture = current_capture()
            if capture is not None:
                capture.committed_step(node=node, model=model)
            if spectral_seam is not None:
                # THE Level-2 slow-large-step hook seam.  The RK slow-mode
                # state above is committed and the acoustic substeps are
                # over; output, nest feedback and the next large step have
                # not happened.  Fires once per domain per model time
                # step.  The step is numbered the way step_observer below
                # numbers it -- the clock still holds pre-step ticks, so
                # the committed step is step_count + 1 -- and shadow/apply
                # receipts land in the seam's ledger for the run capsule.
                spectral_seam.after_step(
                    node.state, grid_id, node.cfg.run,
                    step_count=clock.step_count + 1,
                    model_seconds=((clock.ticks + clock.step_ticks)
                                   / clock.tick_den),
                    streamed=streamed_here is not None)
            if validate_state:
                # Adaptive single-domain runs use this executor too. Keep the
                # existing per-step NaN and live-layer CFL gate before output,
                # feedback or checkpoint consumers can observe this state.
                # Streamed reports come from the canonical sweep, not its
                # unchanged resident initialization snapshot.
                from woof.core.streaming import step_health

                report = step_health(steppers.get(grid_id), node.state,
                                     node.cfg.run,
                                     boundary_width=node.cfg.run.spec_bdy_width)
                if report["nan"]:
                    raise RuntimeError(
                        f"d{grid_id:02d} integration produced a non-finite state "
                        f"at model step {clock.step_count + 1}")
                cfl = report["cfl"]
                if cfl is not None and not math.isfinite(float(cfl)):
                    raise RuntimeError(
                        f"d{grid_id:02d} integration produced a non-finite "
                        f"vertical Courant number at model step {clock.step_count + 1}: "
                        "a model layer's live thickness is non-positive or non-finite")
            poison()
            if streamed_here is not None:
                # The same authority AFTER the step as before it.  The tiles
                # advanced their own carrier clock by dt from the REAL
                # kernel-facing image imposed above, so until the next
                # step's impose the domain's scalars held that free-running
                # sum while the state held the exact next tick -- and the
                # store-direct final digest reads the scalars.  MEASURED on
                # a 3 h HRRR-grid forecast with a fractional adaptive dt:
                # elapsed_seconds 10800.0 resident, 10800.0002734375 on the
                # streamed road, every wrfout byte identical, so the two
                # canonical digests differed for one forecast.
                streamed_here.impose_clock(node.state.elapsed_seconds)
            if validators and health_debug:
                validators[grid_id].require_healthy(
                    phase=f"post-step.d{grid_id:02d}")
            if pool_trim_per_period:
                # Step-cadence trim: one root period holds up to 36 d04
                # substeps of transients; trimming only at period commit let
                # the pool refill past the WDDM ceiling on the 4-domain
                # shape (measured 32.2 GiB peak, 50 s above 31.5 GiB in a
                # 15-sim-min run). Same byte-inert release, per STEP op.
                _trim_default_pool()
        if step_observer is not None:
            # Outside the turn, after the step committed: a step that
            # raised is not a step that happened.  Telemetry never fails
            # a run, so a sink that raises is dropped for that one line.
            #
            # POST-STEP values, and they have to be computed rather than
            # read: `execute_schedule` calls `dom.advance()` AFTER this
            # hook returns (woof/core/clock.py, the STEP op), so the
            # clock still holds the state the step started from.
            # Reporting it raw would number the first step 0 and stamp
            # it with the previous step's valid time -- off by one, on
            # every line, in the direction a reader cannot detect.  Same
            # integer-tick derivation `refresh_model_time(after_step=True)`
            # uses: one FP64 division of exact integers, never an
            # accumulation.
            try:
                step_observer(
                    grid_id=grid_id, step_count=clock.step_count + 1,
                    model_seconds=((clock.ticks + clock.step_ticks)
                                   / clock.tick_den),
                    step_wall_seconds=time.perf_counter() - started_wall,
                    # The step JUST TAKEN, from the same integer-tick
                    # lattice as model_seconds above -- not the step the
                    # clock is about to adopt.  Under an adaptive dt the
                    # two differ on most steps, and reporting the next
                    # one would label every line with a step that has
                    # not happened yet.  Whether this is emitted at all
                    # is the log's decision, not this seam's.
                    dt=clock.step_ticks / clock.tick_den)
            except Exception:  # noqa: BLE001 - telemetry never fails a run
                pass
        step_wall_by_domain[grid_id] = (step_wall_by_domain.get(grid_id, 0)
                                        + (time.perf_counter() - started_wall))

    def on_force(child_id, parent_id, child_clock, parent_clock) -> None:
        with domain_turn(("FORCE", child_id, parent_id)), allocation_scope(child_id):
            node = model.node(child_id)
            if node.parent is None or node.parent.cfg.grid_id != parent_id:
                raise RuntimeError(
                    "schedule FORCE edge differs from domain tree")
            status.pending_force = status.pending_force + 1
            status.prior_feedback_committed = False
            # NO store projection here, any more.  The coupler owns its own
            # reads: ``NestCoupler._coupled_parent_field`` pulls exactly the
            # two windows FORCE needs through the store seam
            # (``streaming.refresh_from_store`` with
            # ``nest.parent_footprint_window``), so the executor projecting
            # a window of EVERY carrier on the coupler's behalf was the same
            # bytes moved twice -- and a second copy of the footprint
            # geometry that a relocation could leave behind.  The window
            # arithmetic and its halo bound moved to ``woof.core.nest``
            # with the reads.
            try:
                node.coupler.force(node)
                if validators and health_debug:
                    validators[child_id].require_healthy(
                        phase=(f"post-force.d{child_id:02d}"
                               f"-from-d{parent_id:02d}"))
            finally:
                status.pending_force = status.pending_force - 1
            poison()

    def on_feedback_prepare(child_id, parent_id, child_clock,
                            parent_clock) -> None:
        node = model.node(child_id)
        status.pending_feedback = status.pending_feedback + 1
        node.coupler.feedback_prepare(node, feedback_scratch[child_id])

    def on_feedback_commit(child_id, parent_id, child_clock,
                           parent_clock) -> None:
        node = model.node(child_id)
        # Two-way feedback is the one nest op that WRITES the parent, and a
        # streamed parent's writes have to land in its store or they are
        # thrown away at the next sweep.  The COUPLER owns that transaction:
        # ``NestCoupler.feedback_commit`` starts from the store and ends in
        # it (``_sync_in``/``_sync_out`` -> ``streaming.refresh_from_store``
        # / ``commit_to_store``, both of which drain the in-flight sweep
        # tail first), and ``feedback_finalize`` carries the whole-parent
        # diagnostics back the same way.  MEASURED bit-identical to the
        # all-resident run, with both feedback controls firing:
        # ``tilestream/test_nest.py`` leg 3 (coupler-driven) and
        # ``tilestream/test_nest_executor.py`` (this dispatch, through
        # execute_experiment).
        #
        # A refusal used to stand here, asserting "nothing projects a write
        # back".  It arrived in the same three-way merge (2358b3c06) as the
        # coupler arm it contradicted -- two lanes, one merged truth
        # missing -- and no gate drove this dispatch with a streamed parent,
        # so the contradiction sat unexercised.  The dispatch is now the
        # same for both modes, and so is the coupler: either or both ends of
        # a coupling edge may stream, and a cross-scheme edge off a streamed
        # parent pulls the transition's declared planes through the child's
        # footprint window like every other FORCE read.
        status.mutation_in_progress = True
        try:
            with allocation_scope(child_id):
                node.coupler.feedback_commit(node)
        finally:
            status.mutation_in_progress = False

    def on_feedback_finalize(child_id, parent_id, child_clock,
                             parent_clock) -> None:
        node = model.node(child_id)
        status.mutation_in_progress = True
        try:
            with allocation_scope(child_id):
                node.coupler.feedback_finalize(node)
        finally:
            status.mutation_in_progress = False
            status.pending_feedback = status.pending_feedback - 1
        poison()

    def on_history(grid_id, ticks) -> None:
        if history_handler is not None:
            if validators:
                validators[grid_id].require_healthy(
                    phase=f"pre-history.d{grid_id:02d}")
            node = model.node(grid_id)
            nudging_owner = getattr(node.state, '_spectral_nudging_owner', None)
            if nudging_owner is not None:
                nudging_owner.check_at_seam()
            from woof.wrf_exact import ENABLED as wrf_exact
            if wrf_exact and node.parent is not None:
                # WRF writes a nest's frame at an instant it shares with its
                # parent after the parent's next med_nest_force, whose
                # couple/uncouple round trip the frame then carries
                # (woof.core.nest_round_trip.round_tripped_frame_fields).
                node.state._wrf_frame_round_trip_due = (
                    int(node.clock.ticks) == int(node.parent.clock.ticks)
                    and int(node.clock.ticks) < int(node.clock.run_ticks))
                node.state._wrf_round_trip_run = node.cfg.run
            history_handler(model, node, ticks)
        if io_manager is not None:
            status.pending_d2h = int(io_manager.pending)

    def on_restart(ticks) -> None:
        # A handler that orders its checkpoint after the history frames
        # itself (the prepared single-domain door's background checkpoint
        # writer) is not made to wait here for frames still being written.
        if io_manager is not None and not orders_history_itself(
                restart_handler):
            io_manager.drain()
            status.pending_d2h = int(io_manager.pending)
        if restart_handler is not None:
            restart_handler(model, ticks)
            if requested_restart[0]:
                # A worker's request is acknowledged only once the
                # checkpoint it asked for is durable.
                wait = durable_waiter(restart_handler)
                if wait is not None:
                    wait()
                restart_request.acknowledge()
                requested_restart[0] = False

    def on_period_end(period, clocks) -> None:
        status.schedule_cursor = PERIOD_BEGIN
        status.prior_feedback_committed = status.pending_feedback == 0
        root_clock = clocks[model.root.cfg.grid_id]
        requested_restart[0] = (restart_handler is not None
                                and restart_request.pending())
        mandatory = (
            root_clock.at_stop_time
            or any(clock.history_due() for clock in clocks.values())
            or (restart_enabled and root_clock.restart_due())
            or requested_restart[0])
        if validators and (health_debug or mandatory
                           or root_clock.step_count % 4 == 0):
            for gid, validator in validators.items():
                validator.require_healthy(
                    phase=f"post-d01-sync.d{gid:02d}")

    #: Wall clock of the outer step that is committing, measured the
    #: same way the single-domain loop measures its own
    #: (runtime.integrate_prepared_case).  It used to be published as a
    #: literal 0.0 from here, which meant the tree route -- the one that
    #: runs the deep nests, where the number matters most -- reported no
    #: cadence at all: every consumer of step_wall_seconds (the
    #: supervisor's stale-step threshold, any progress display) was
    #: reading a constant.  One perf_counter pair per outer step.
    period_wall = [time.perf_counter()]

    def on_period_commit(period, clocks) -> None:
        root_clock = clocks[model.root.cfg.grid_id]
        # Scheduled restart operations have already run. A still-pending
        # request therefore needs one write at this fully synchronized
        # boundary, after history resets and feedback have committed.
        if requested_restart[0]:
            on_restart(root_clock.ticks)
        if pool_trim_per_period:
            _trim_default_pool()
        now = time.perf_counter()
        # WALL time, not model time.  The local is deliberately NOT named
        # step_wall_seconds even though the keyword it feeds is: the
        # clock audit (tests/test_clock.py) bans any assignment in this
        # module whose target name carries "elapsed"/"second", because
        # MODEL elapsed seconds must be tick-derived and never
        # float-accumulated.  A perf_counter difference is a different
        # quantity that happens to share the word, and the audit keeps
        # its full force here: an actual elapsed-seconds assignment
        # still trips it.
        step_wall = now - period_wall[0]
        period_wall[0] = now
        if progress_callback is not None:
            progress_callback(
                model_elapsed_seconds=root_clock.elapsed_seconds,
                outer_step=root_clock.step_count,
                last_durable_wrfout=(None if io_manager is None else
                                     io_manager.last_durable_wrfout),
                last_checkpoint=model._last_checkpoint,
                phase="post-d01-sync",
                step_wall_seconds=step_wall,
                # The whole tree's clocks, not only the root's.  This
                # dict was already in scope and deliberately unforwarded,
                # which left a nested run with no per-domain advancement
                # anywhere on the callback: a consumer that wanted it had
                # to parse child wrfout filenames, the re-derivation the
                # output hook exists to avoid.
                #
                # Additive and safe.  Every consumer of this callback
                # takes **kwargs -- the supervisor heartbeat, both
                # prepared runners, run-plan's observer, the verify
                # cases -- so an extra key is a TypeError for none of
                # them, and one that ignores it behaves exactly as
                # before.
                #
                # Read, never accumulated: each value is that clock's
                # own tick-derived figure, which is what the clock audit
                # in tests/test_clock.py is protecting.
                domain_clocks={
                    grid_id: float(clock.elapsed_seconds)
                    for grid_id, clock in clocks.items()},
                domain_step_wall=dict(step_wall_by_domain))

    clocks = {node.cfg.grid_id: node.clock
              for node in model.walk_parent_first()}
    root_ticks = model.root.clock.ticks
    if root_ticks % model.schedule.period_ticks != 0:
        raise ValueError("model does not start at a PERIOD_BEGIN tick")
    start_period = root_ticks // model.schedule.period_ticks
    if progress_callback is not None:
        # Stepping begins here.  Without this transition the last published
        # phase is whatever preparation label the runner left behind
        # (initialize-domain-writers on the tree path), so a first-step
        # physics failure was stamped as a preparation failure and sent a
        # real diagnosis down the wrong path.  ``outer_step`` stays the
        # completed-step count run-progress.json already tracks; the phase
        # names the 1-based outer step being attempted, matching the
        # single-domain loop's ``outer-N.substep-M`` convention.
        root_clock = model.root.clock
        progress_callback(
            model_elapsed_seconds=root_clock.elapsed_seconds,
            outer_step=root_clock.step_count,
            last_durable_wrfout=(None if io_manager is None else
                                 io_manager.last_durable_wrfout),
            last_checkpoint=model._last_checkpoint,
            phase=f"stepping:outer-{root_clock.step_count + 1}",
            step_wall_seconds=0.0)
    # Start the cadence clock at the last instant before stepping, so
    # the first outer step's wall time is the step's and not the
    # transition heartbeat's.
    period_wall[0] = time.perf_counter()
    # ADAPTIVE TIME STEP.  Off, this is None and execute_schedule takes
    # its precomputed periodic walk exactly as before -- the whole feature
    # is one argument that stays None.  On, the driver owns every float:
    # clock.py's walk is AST-audited integer-only, so what reaches it is a
    # step_ticks and nothing else.
    adaptive_driver = None
    if bool(model.root.cfg.run.use_adaptive_time_step):
        from woof.core.adaptive_clock import AdaptiveClockDriver
        from woof.core.dycore import (enable_wrf_cfl_recording,
                                       take_wrf_cfl)
        # The controller reads the same reduction the probe does rather
        # than carrying a second copy of the kernel.
        enable_wrf_cfl_recording()
        try:
            moving_deadline = relocation_runner.next_deadline_ticks
        except AttributeError:
            moving_deadline = None
        adaptive_driver = AdaptiveClockDriver(
            model, cfl_source=take_wrf_cfl,
            tick_den=model.schedule.clock.tick_den,
            carrier_source=lambda gid: (
                None if _streamed(gid) is None
                else _streamed(gid).carrier_provenance()),
            map_factor_source=lambda gid: (
                None if _streamed(gid) is None
                else _streamed(gid).maximum_map_factor()),
            deadline_source=moving_deadline)

    try:
        if schedule_dispatch is not None:
            execution = schedule_dispatch(
                model=model, schedule=model.schedule,
                step_bindings=steppers,
                callbacks={"step": on_step, "force": on_force,
                    "feedback_prepare": on_feedback_prepare,
                    "feedback_commit": on_feedback_commit,
                    "feedback_finalize": on_feedback_finalize,
                    "history": on_history, "restart": on_restart,
                    "domain_start": on_domain_start,
                    "period_begin": on_period_begin,
                    "period_end": on_period_end,
                    "period_commit": on_period_commit},
                adaptive_driver=adaptive_driver, clocks=clocks,
                skip_feedback_path=skip_feedback_path,
                start_period=start_period,
                started_grid_ids=tuple(
                    node.cfg.grid_id for node in model.walk_parent_first()
                    if bool(node._started)),
                committed_initial_history_grid_ids=(
                    model._resume_committed_history_grid_ids
                    if bool(model._resumed) else ()))
        else:
            execution = execute_schedule(
                model.schedule, on_step=on_step, on_force=on_force,
                on_period_steps=adaptive_driver,
                on_feedback_prepare=on_feedback_prepare,
                on_feedback_commit=on_feedback_commit,
                on_feedback_finalize=on_feedback_finalize,
                on_history=on_history, on_restart=on_restart,
                on_domain_start=on_domain_start,
                on_period_begin=on_period_begin, on_period_end=on_period_end,
                on_period_commit=on_period_commit,
                skip_feedback_path=skip_feedback_path, clocks=clocks,
                start_period=start_period,
                started_grid_ids=(
                    node.cfg.grid_id for node in model.walk_parent_first()
                    if bool(node._started)),
                committed_initial_history_grid_ids=(
                    model._resume_committed_history_grid_ids
                    if bool(model._resumed) else ()))
    finally:
        if adaptive_driver is not None:
            # THE FOLD IS A MODULE GLOBAL, so it outlives this run.  A
            # second experiment in the same process -- a chained `woof
            # go`, a test module -- would otherwise keep launching
            # w_cfl_stat every RK stage for a run that never asked for
            # it, hold 4.72 MB per domain, and read this run's row for a
            # grid_id it reuses.  In a finally so a refused run releases
            # it too.
            from woof.core.dycore import reset_wrf_cfl_recording
            reset_wrf_cfl_recording()
    if relocation_runner is not None:
        relocation_runner.close_receipt(model)
    return execution


__all__ = [
    "PERIOD_BEGIN", "RestartPhase", "DomainNode", "FeedbackScratch",
    "NestCoupler", "ExperimentState", "ModelMemoryLedger",
    "ModelRuntimeStatus", "SharedRRTMGPChunkWorkspace", "build_experiment",
    "execute_experiment", "experiment_fingerprint",
]
