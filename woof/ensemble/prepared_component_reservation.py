"""Pre-thread allocation metadata for original prepared component waves.

Descriptors contain shapes and policy identities only. They are never usable
as forecast states or coefficient tables. Actual route owners must still pass
the live allocation and numerical admission checks before a packed operation.
"""
from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from datetime import date, datetime
from fractions import Fraction
from types import SimpleNamespace
from collections.abc import Mapping
import hashlib
import json
import math

import numpy as np

from woof.ensemble.batch_state import _exact_key, state_array_specs
from woof.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan
from woof.ensemble.prepared_route_wave import RouteWaveReservation
from woof.ensemble.batch_storage import _positive_count as _positive


def _identity(value):
    if isinstance(value, (date, datetime)):
        return (type(value).__name__, value.isoformat())
    if isinstance(value, Fraction):
        return ("fraction", value.numerator, value.denominator)
    if is_dataclass(value):
        return tuple((field.name, _identity(getattr(value, field.name)))
                     for field in dataclass_fields(value) if field.init and not field.name.startswith("_"))
    if isinstance(value, (tuple, list)):
        return tuple(_identity(item) for item in value)
    return _exact_key(value)


@dataclass(frozen=True)
class PreparedDomainAllocationMetadata:
    """Authorities the experiment alone does not describe.

    ``landuse_categories=None`` explicitly declares absent source fractions.
    Missing metadata is not interpreted as that declaration. Coefficient rows
    name ``table_owner:array`` (for example ``lw_tables:kmajor``) and describe
    every potentially cold device upload. Cache reuse may reduce the live plan.
    """
    p_top: float
    radiation_column_chunk: int
    native_workspace: bool
    soil_layers: int
    landuse_categories: int | None
    coefficient_uploads: tuple[BatchArraySpec, ...]
    radiation_table_identity: str
    resident: bool
    evidence: str
    mynn_column_chunk: int | None = None

    def __post_init__(self):
        if isinstance(self.p_top, (bool, np.bool_)) or not math.isfinite(self.p_top) or self.p_top <= 0:
            raise ValueError("prepared scalar p_top must be positive and finite")
        _positive(self.radiation_column_chunk, "resolved radiation column chunk")
        _positive(self.soil_layers, "prepared soil layers")
        if self.mynn_column_chunk is not None:
            _positive(self.mynn_column_chunk, "original MYNN column chunk")
        if self.landuse_categories is not None:
            _positive(self.landuse_categories, "prepared landuse category count")
        if type(self.native_workspace) is not bool or type(self.resident) is not bool:
            raise TypeError("workspace and resident declarations must be boolean")
        rows = tuple(self.coefficient_uploads)
        if not rows or any(not isinstance(row, BatchArraySpec) or row.ownership != "shared" for row in rows):
            raise ValueError("declare every cold immutable coefficient backing as a shared allocation")
        if len({row.name for row in rows}) != len(rows):
            raise ValueError("duplicate coefficient backing names hide uploads")
        for text in (self.radiation_table_identity, self.evidence):
            if not isinstance(text, str) or not text.strip():
                raise ValueError("prepared allocation metadata needs table identity and source evidence")
        object.__setattr__(self, "coefficient_uploads", rows)


@dataclass(frozen=True)
class ComponentReservationDecision:
    reservation: RouteWaveReservation | None
    ordinary_reason: str | None
    missing_metadata: tuple[str, ...] = ()

    @property
    def eligible(self):
        return self.reservation is not None and self.ordinary_reason is None


@dataclass(frozen=True)
class _ArrayDescriptor:
    shape: tuple[int, ...]
    dtype: object
    device: object = None


def _packaged_coefficient_tables():
    from woof.core.rrtmgp import load_gas_tables, load_cloud_tables
    return (("lw_tables", load_gas_tables("lw")), ("sw_tables", load_gas_tables("sw")),
            ("lw_cloud_tables", load_cloud_tables("lw")), ("sw_cloud_tables", load_cloud_tables("sw")))


def packaged_coefficient_allocation_specs():
    """Use the original CPU coefficient loaders and upload dtype rule."""
    rows = []
    for name, table in _packaged_coefficient_tables():
        for field, array in vars(table).items():
            if isinstance(array, np.ndarray) and array.size:
                dtype = "uint8" if array.dtype == np.dtype("bool") else "float32"
                rows.append(BatchArraySpec(f"{name}:{field}", array.shape, "shared", dtype))
    return tuple(rows)


def packaged_coefficient_identity():
    """Hash the original restart identities of the resolved host tables."""
    from woof.io.restart import _resolved_object_setup_identity
    identity = {name: _resolved_object_setup_identity(table, name)
                for name, table in _packaged_coefficient_tables()}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def prepared_component_allocation_metadata(member_inputs, *, resident_decisions=None):
    """Read shapes and policy from already verified prepared tree owners.

    No forecast field is loaded or transformed here. Scalar top and radiation
    chunk follow the genuine tree constructor, including its shared modern
    workspace predicate. Source category and canonical soil shapes come from
    the preflight's static fields and checked cache manifest. For a configured
    tile road, the caller supplies its original cold streaming decisions;
    absence cannot be interpreted as a resident allocation.
    """
    from woof.prepared_domain_tree_forecast import PreparedTreeInputs
    from woof.core.model import uses_modern_rrtmgp_workspace
    from woof.core.preflight import soil_layer_count
    from woof.core import streaming
    from woof.core.mynn_pbl_scratch import resolve_mynn_column_chunk
    if not isinstance(member_inputs, Mapping) or not member_inputs:
        raise ValueError("allocation metadata requires identified prepared members")
    if any(not isinstance(inputs, PreparedTreeInputs) for inputs in member_inputs.values()):
        raise TypeError("allocation metadata requires verified PreparedTreeInputs owners")
    decisions = {} if resident_decisions is None else resident_decisions
    if not isinstance(decisions, Mapping):
        raise TypeError("original cold streaming decisions must be a mapping")
    uploads = packaged_coefficient_allocation_specs()
    table_identity = packaged_coefficient_identity()
    result = {}
    for member, inputs in member_inputs.items():
        exp = inputs.experiment
        top, chunk = exp.vertical.p_top, exp.column_chunk
        native_workspace = uses_modern_rrtmgp_workspace(exp)
        bundles = {int(bundle.grid_id): bundle for bundle in inputs.domains}
        if len(bundles) != len(inputs.domains) or set(bundles) != {int(dc.grid_id) for dc in exp.domains}:
            raise ValueError(f"member {member}: prepared domain inventory differs from its experiment")
        for domain in exp.domains:
            gid, cfg = int(domain.grid_id), domain.run
            bundle = bundles[gid]
            if int(bundle.parent_id) != int(domain.parent_id):
                raise ValueError(f"member {member} domain {gid}: prepared parent authority differs")
            reader = bundle.cache_reader
            base_top = reader.metadata["base_scalars"]["p_top"]
            if _identity(base_top) != _identity(top):
                raise ValueError(f"member {member} domain {gid}: prepared scalar top differs from the experiment")
            soil_layers = soil_layer_count(cfg)
            for name in ("TSLB", "SMOIS", "SH2O"):
                spec = reader.arrays.get("surface/" + name)
                if spec is not None and tuple(spec["shape"]) != (soil_layers, cfg.ny, cfg.nx):
                    raise ValueError(f"member {member} domain {gid}: prepared {name} shape differs from the original soil inventory")
            fractions = bundle.static_fields.get("LANDUSEF")
            if fractions is None:
                raise ValueError(f"member {member} domain {gid}: verified native static lacks its LANDUSEF authority")
            shape = tuple(fractions.shape)
            if len(shape) != 3 or shape[1:] != (cfg.ny, cfg.nx):
                raise ValueError(f"member {member} domain {gid}: native LANDUSEF shape differs from the original grid")
            options = streaming.options_for_domain(domain, getattr(exp, "tiles", None))
            if options is None or options.mode == "off":
                resident = True
            else:
                decision = decisions.get((member, gid))
                if decision is None or type(getattr(decision, "stream", None)) is not bool:
                    raise ValueError(f"member {member} domain {gid}: configured tile road requires its original cold streaming decision before component pricing")
                resident = not decision.stream
            pin = bundle.authority_sha256
            result[(member, gid)] = PreparedDomainAllocationMetadata(top, chunk,
                native_workspace, soil_layers, shape[0], uploads, table_identity, resident,
                "verified prepared domain authorities " + json.dumps(dict(pin), sort_keys=True)
                + "; original experiment vertical top and radiation chunk; original cold road",
                mynn_column_chunk=resolve_mynn_column_chunk(cfg.nz))
    return result


def _physics_plan(cfg, metadata, *, members, device_id, reuse_original_workspaces=False):
    """Feed the qualified shape inventory, never a cloned numerical driver."""
    from woof.core.preflight import physics_array_shapes
    from woof.ensemble.scheduled_production_physics import production_physics_memory_plan
    shapes = {name.removeprefix("fields/"): shape for name, shape in physics_array_shapes(cfg).items()
              if name.startswith("fields/")}
    if metadata.landuse_categories is not None:
        shapes["landusef"] = (metadata.landuse_categories, cfg.ny, cfg.nx)
    integers = {"ivgtyp", "isltyp", "kpbl", "ktop_plume"}
    fields = {name: _ArrayDescriptor(tuple(shape), np.dtype("int32" if name in integers else "float32"))
              for name, shape in shapes.items()}
    state = SimpleNamespace(p_top=metadata.p_top, _mynn_rank_column_chunk=metadata.mynn_column_chunk)
    for spec in state_array_specs(cfg):
        setattr(state, spec.name, _ArrayDescriptor(spec.shape, np.dtype(spec.dtype)))
    # The helper reads table upload arrays only from real ndarray objects.
    # Descriptors deliberately expose none. The complete cold upload catalogue
    # is added explicitly below rather than pretending coefficients are live.
    tables = {name: SimpleNamespace(_device={device_id: None}) for name in
              ("lw_tables", "sw_tables", "lw_cloud_tables", "sw_cloud_tables")}
    adapter = SimpleNamespace(column_chunk=metadata.radiation_column_chunk,
        chunk_workspace=object() if metadata.native_workspace else None,
        latitude_deg=_ArrayDescriptor((cfg.ny, cfg.nx), np.dtype("float32"), SimpleNamespace(id=device_id)),
        **tables)
    driver = SimpleNamespace(fields=fields, state=state, radiation_callable=adapter,
        mynn_sfclay_sea_result=object())
    plan = production_physics_memory_plan((driver,) * members, (cfg,) * members,
        reuse_original_workspaces=reuse_original_workspaces)
    prefix = "radiation:('radiation', 0):new_table:"
    catalogue = {row.name: row for row in packaged_coefficient_allocation_specs()}
    declared = {row.name: row for row in metadata.coefficient_uploads}
    if catalogue != declared:
        raise ValueError("coefficient upload catalogue differs from current packaged tables; original table owners retain their private layout")
    uploads = tuple(BatchArraySpec(prefix + row.name, row.shape, row.ownership, row.dtype)
                    for row in metadata.coefficient_uploads)
    return BatchMemoryPlan(plan.arrays + uploads, reserved_bytes=plan.reserved_bytes)


def _plans(domains, metadata_by_grid, *, members, device_id, shared_fields,
           reuse_original_workspaces=False, edge_state_only=False):
    from woof.core.preflight import nest_slot_shapes, nest_slot_dtypes, nest_field_kinds
    from woof.ensemble.batch_nesting import member_nest_memory_plan
    plans, descriptors = {}, {}
    if type(edge_state_only) is not bool:
        raise TypeError("edge-only bank inventory needs an explicit boolean ownership declaration")
    if edge_state_only:
        from woof.ensemble.batch_nesting import prepared_tree_edge_state_fields
        selected = prepared_tree_edge_state_fields(domains)
    for domain in domains:
        gid, cfg = int(domain.grid_id), domain.run
        specs = state_array_specs(cfg, shared_fields=shared_fields)
        if edge_state_only:
            names = selected[gid]
            missing = names - {row.name for row in specs}
            if missing or not names:
                raise ValueError(f"domain {gid}: edge dependency inventory is missing original state allocations {sorted(missing)}")
            specs = tuple(row for row in specs if row.name in names)
        plan = BatchMemoryPlan(specs, reserved_bytes=0)
        plans[f"bank:{gid}"] = plan
        descriptors[gid] = SimpleNamespace(cfg=cfg, storage=SimpleNamespace(specs={row.name: row for row in plan.arrays}))
        plans[f"physics:{gid}"] = _physics_plan(cfg, metadata_by_grid[gid], members=members, device_id=device_id,
            reuse_original_workspaces=reuse_original_workspaces)
    by_grid = {int(domain.grid_id): domain for domain in domains}
    for child in domains:
        if not child.parent_id:
            continue
        parent = by_grid[int(child.parent_id)]
        if child.run.mp_physics != parent.run.mp_physics or child.run.nz != parent.run.nz:
            raise ValueError("nested component edge needs identical species and vertical inventories; original transitions retain mixed schemes")
        shapes = nest_slot_shapes(child, child.run.spec_bdy_width, parent)
        dtypes = nest_slot_dtypes(child, child.run.spec_bdy_width, parent)
        registrations = {}
        for stagger in ("m", "x", "y"):
            registrations[stagger] = SimpleNamespace(**{name: _ArrayDescriptor(
                tuple(shapes[f"nest_sint_{name}_{stagger}"]), np.dtype(dtypes[f"nest_sint_{name}_{stagger}"]))
                for name in ("ci", "ip", "cj", "jp", "xig", "xjg")})
        gid = int(child.grid_id)
        plans[f"edge:{gid}"] = member_nest_memory_plan(descriptors[int(child.parent_id)],
            descriptors[gid], registrations, nest_field_kinds(child.run))
    return plans


def plan_prepared_component_reservation(member_inputs, *, domain_metadata=None,
        ordinary_forecast_bytes, fixed_bytes, evidence, available_bytes,
        device_id=0, shared_fields=(), radar=False, resident_decisions=None,
        reuse_original_workspaces=False, edge_state_only=False):
    """Return a cold admission decision without allocating any forecast array.

    The caller already owns the original complete forecast and output estimates.
    This function only adds official component inventories. Unsupported or
    missing authorities return a concrete ordinary reason; ordinary wave sizing
    and the original streamed door remain the caller's responsibility.
    """
    def decline(reason, missing=()):
        return ComponentReservationDecision(None, reason, tuple(missing))
    if not isinstance(member_inputs, Mapping) or not member_inputs:
        raise ValueError("component reservation needs globally identified prepared members")
    if len(member_inputs) < 2:
        return decline("a singleton retains its original runner; there is no member cohort to join")
    if radar:
        return decline("simulated radar retains original member volume writers and native consumers")
    if set(member_inputs) != set(ordinary_forecast_bytes):
        raise ValueError("reserve each original wave member exactly once")
    experiments = [value.experiment for value in member_inputs.values()]
    first = experiments[0]
    domains = tuple(first.domains)
    if len(domains) < 2:
        return decline("prepared component edge factory requires an actual parent and nest")
    if any(int(getattr(getattr(exp, "devices", None), "count", 1)) > 1 for exp in experiments):
        return decline("ranked members retain their original device communicators and CFL owners")
    for exp in experiments:
        if (_identity(exp.start_time) != _identity(first.start_time)
                or _identity(exp.domains) != _identity(domains)
                or exp.feedback != first.feedback
                or getattr(exp, "smooth_option", 0) != getattr(first, "smooth_option", 0)):
            return decline("member configurations, calendar or nest layout differ; a common bank cannot substitute their original authorities")
        if getattr(getattr(exp, "relocation", None), "enabled", False):
            return decline("moving nests retain original relocation and rebuilt geometry owners")
    from woof.config import (RunConfig, mynn_mixscalars_driver_value,
                              radiation_scheme_ids)
    from woof.physics_compat import rrtmg_variant, RRTMG_VARIANT_RTE_RRTMGP
    from woof.core.preflight import soil_layer_count
    from woof.ensemble.batch_mynn import _pbl_options
    for domain in domains:
        cfg, gid = domain.run, int(domain.grid_id)
        if cfg.chem_sets:
            return decline(f"domain {gid}: active chemistry keeps its original member species arrays, source-hour caches and mass ledger; component banks lack their typed state ownership and producer handoff")
        if (cfg.sf_surface_physics, cfg.sf_sfclay_physics, cfg.bl_pbl_physics) != (3, 5, 5):
            return decline(f"domain {gid}: packed production binds RUC/MYNN surface and PBL leaves; selected land or turbulence scheme retains its original driver")
        if cfg.bl_mynn_version != "wrf_461":
            return decline(f"domain {gid}: GSD cloud output needs legacy RRTMG coupling, and the original numerical MYNN driver requires icloud_bl=1")
        if radiation_scheme_ids(cfg) != (4, 4) or rrtmg_variant(cfg) != RRTMG_VARIANT_RTE_RRTMGP:
            return decline(f"domain {gid}: packed production has no original radiation leaf binding for this selected pair")
        policy_names = ("alb_sol", "swint_opt", "aer_opt")
        default_policy = tuple(RunConfig.__dataclass_fields__[name].default for name in policy_names)
        if tuple(getattr(cfg, name) for name in policy_names) != default_policy:
            return decline(f"domain {gid}: selected solar/aerosol policy needs its current packed carriers and ABI qualification")
        if getattr(cfg, "inflow_perturbation", 0):
            return decline(f"domain {gid}: active nest inflow retains its original post-interpolation perturbation binding")
        if cfg.cu_physics not in (0, 99):
            return decline(f"domain {gid}: packed production cumulus binds cu_physics=99; the selected scheme needs its own original column leaf")
        if cfg.mosaic_lu or cfg.mosaic_soil or any(getattr(cfg, f"spp_{name}", 0) for name in ("conv", "pbl", "lsm")):
            return decline(f"domain {gid}: mosaic or stochastic leaf patterns lack their private packed bindings")
        try:
            options = {name: getattr(cfg, name) for name in ("bl_mynn_cloudpdf", "bl_mynn_mixlength",
                "bl_mynn_edmf", "bl_mynn_edmf_mom", "bl_mynn_edmf_tke", "bl_mynn_mixscalars",
                "bl_mynn_cloudmix", "bl_mynn_mixqt", "bl_mynn_output", "bl_mynn_tkeadvect")}
            _pbl_options({**options, "closure": cfg.bl_mynn_closure, "bl_mynn_version": cfg.bl_mynn_version,
                "bl_mynn_gsd41_unsquared_qtke": cfg.bl_mynn_gsd41_unsquared_qtke,
                "bl_mynn_cloud_tendency_form": cfg.bl_mynn_cloud_tendency_form,
                "icloud_bl": cfg.icloud_bl,
                # The value the driver is handed: 0 when the key is inert
                # (no qn family in the microphysics), as in the single run.
                "bl_mynn_mixscalars": mynn_mixscalars_driver_value(cfg)})
        except (ValueError, TypeError) as error:
            return decline(f"domain {gid}: {error}")
        for name, allowed in (("ruc_irrigation", ("wrf_45", "wrf_461")),
                              ("ruc_qvg_cold_start", ("air", "wrf")),
                              ("ruc_2m_diagnostic", ("flux", "log_profile")),
                              ("ruc_snow", ("wrf_45", "wrf_461"))):
            if type(getattr(cfg, name)) is not str or getattr(cfg, name) not in allowed:
                return decline(f"domain {gid}: {name} does not name an original RUC numerical branch")
        if soil_layer_count(cfg) != 6:
            return decline(f"domain {gid}: production cohort currently requires the original six-layer RUC allocation inventory")
    if domain_metadata is None:
        try:
            domain_metadata = prepared_component_allocation_metadata(member_inputs,
                resident_decisions=resident_decisions)
        except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
            return decline(f"prepared component metadata cannot bind its original source authorities: {error}")
    missing = [f"member {member} domain {int(domain.grid_id)} allocation metadata" for member in member_inputs
               for domain in domains if (member, int(domain.grid_id)) not in domain_metadata]
    if missing:
        return decline("prepared scalar top, resolved radiation workspace/chunk, soil/category layout and coefficient authorities are required before threads", missing)
    first_member = next(iter(member_inputs))
    selected = {}
    for domain in domains:
        gid = int(domain.grid_id)
        rows = [domain_metadata[(member, gid)] for member in member_inputs]
        if any(not isinstance(row, PreparedDomainAllocationMetadata) for row in rows):
            return decline(f"domain {gid}: allocation metadata is not a typed prepared authority")
        if any(not row.resident for row in rows):
            return decline(f"domain {gid}: streamed members retain original canonical stores and attach-time layouts")
        def layout_identity(row):
            return tuple((field.name, _identity(getattr(row, field.name))) for field in dataclass_fields(row)
                         if field.name != "evidence")
        if any(layout_identity(row) != layout_identity(rows[0]) for row in rows[1:]):
            return decline(f"domain {gid}: member scalar top, radiation policy, category layout or coefficient identity differs")
        if rows[0].soil_layers != soil_layer_count(domain.run):
            return decline(f"domain {gid}: prepared soil layout differs from the original configuration inventory")
        selected[gid] = domain_metadata[(first_member, gid)]
    try:
        plans = _plans(domains, selected, members=len(member_inputs), device_id=device_id,
                       shared_fields=shared_fields, reuse_original_workspaces=reuse_original_workspaces,
                       edge_state_only=edge_state_only)
        reservation = RouteWaveReservation(ordinary_forecast_bytes, plans, fixed_bytes,
            {**evidence, "native": "official state/edge/physics shapes plus explicit cold coefficient upload metadata before runner threads"}, available_bytes)
        reason = reservation.admit(tuple(member_inputs))
    except (ValueError, TypeError, KeyError, AttributeError, MemoryError) as error:
        return decline(f"configuration-only component inventory is unavailable: {error}")
    return ComponentReservationDecision(reservation, reason)


__all__ = ["PreparedDomainAllocationMetadata", "ComponentReservationDecision",
           "packaged_coefficient_allocation_specs", "packaged_coefficient_identity",
           "prepared_component_allocation_metadata", "plan_prepared_component_reservation"]
