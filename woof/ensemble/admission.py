"""CPU memory inventory for ensemble packing without changing a member run.

The constructor receives the allocation plans used by the execution adapters.
It does not infer that a physics field is immutable, estimate a member by a
fitted multiplier, or open a device. Unknown components stay in the ordinary
single-member path until their allocation inventory is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from operator import index
from typing import Literal

import numpy as np

from woof.ensemble.batch_storage import BatchMemoryPlan

MemoryBasis = Literal["allocation", "measured", "envelope"]


def _integer(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a boolean")
    value = index(value)
    if value < (1 if positive else 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


@dataclass(frozen=True)
class MemoryComponent:
    """One named allocation plan or explicit fixed/member reservation.

    ``independent_members`` prices separate per-member allocations, rounding
    each member's buffer before multiplying. A contiguous member-batched
    backing rounds after multiplying, exactly as ``BatchMemoryPlan`` does.
    A plan's own reservation is held once unless independent allocations are
    requested. Aliased buffers must occur in only one component.
    """

    name: str
    category: str
    plan: BatchMemoryPlan | None = None
    fixed_bytes: int = 0
    per_member_bytes: int = 0
    independent_members: bool = False
    basis: MemoryBasis = "allocation"
    evidence: str = ""
    plan_for_members: object = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("memory component needs a name")
        if not isinstance(self.category, str) or not self.category:
            raise ValueError("memory component needs a category")
        if self.plan is not None and not isinstance(self.plan, BatchMemoryPlan):
            raise TypeError("memory component plan must be BatchMemoryPlan")
        if self.plan_for_members is not None:
            if self.plan is not None or not callable(self.plan_for_members):
                raise TypeError("member-dependent plan needs a callable and no fixed plan")
        if self.basis not in ("allocation", "measured", "envelope"):
            raise ValueError("memory component basis must name its allocation or evidence")
        if not isinstance(self.independent_members, bool):
            raise TypeError("independent_members must be a boolean")
        for name in ("fixed_bytes", "per_member_bytes"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))

    def inventory(self, members):
        members = _integer(members, "members", positive=True)
        plan = (self.plan if self.plan_for_members is None
                else self.plan_for_members(1 if self.independent_members else members))
        if plan is not None and not isinstance(plan, BatchMemoryPlan):
            raise TypeError("member-dependent inventory did not return BatchMemoryPlan")
        if plan is None:
            rows = ()
            plan_reserve = 0
        elif not self.independent_members:
            rows = plan.inventory(members)
            plan_reserve = plan.reserved_bytes
        else:
            # Shared arrays remain one allocation even when member-private
            # fields are separate native single-member buffers.
            rows = tuple(dict(row,
                shape=((members,) + tuple(row["shape"]) if row["ownership"] == "member"
                       else row["shape"]),
                payload_bytes=row["payload_bytes"] * (members if row["ownership"] == "member" else 1),
                allocated_bytes=row["allocated_bytes"] * (members if row["ownership"] == "member" else 1),
                allocation_count=(members if row["ownership"] == "member" else 1))
                for row in plan.inventory(1))
            plan_reserve = plan.reserved_bytes * members
        reserved = self.fixed_bytes + self.per_member_bytes * members + plan_reserve
        allocated = sum(row["allocated_bytes"] for row in rows)
        return {"name": self.name, "category": self.category, "basis": self.basis,
                "evidence": self.evidence, "independent_members": self.independent_members,
                "arrays": rows, "array_bytes": allocated, "reservation_bytes": reserved,
                "required_bytes": allocated + reserved}


@dataclass(frozen=True)
class AllocatorMargin:
    """Explicit pool retention/fragmentation allowance, with rational sizing."""

    numerator: int = 0
    denominator: int = 1
    minimum_bytes: int = 0
    categories: tuple[str, ...] = ()
    evidence: str = ""

    def __post_init__(self):
        object.__setattr__(self, "numerator", _integer(self.numerator, "margin numerator"))
        object.__setattr__(self, "denominator", _integer(self.denominator, "margin denominator", positive=True))
        object.__setattr__(self, "minimum_bytes", _integer(self.minimum_bytes, "margin minimum"))
        categories = tuple(self.categories)
        if any(not isinstance(c, str) or not c for c in categories) or len(set(categories)) != len(categories):
            raise ValueError("allocator margin categories need distinct nonempty names")
        object.__setattr__(self, "categories", categories)

    @classmethod
    def from_fraction(cls, fraction, **kwargs):
        value = Fraction(str(fraction))
        if value < 0:
            raise ValueError("allocator margin fraction must not be negative")
        return cls(value.numerator, value.denominator, **kwargs)

    def required_bytes(self, rows):
        base = sum(row["required_bytes"] for row in rows
                   if not self.categories or row["category"] in self.categories)
        return max(self.minimum_bytes,
                   (base * self.numerator + self.denominator - 1) // self.denominator)


@dataclass(frozen=True)
class EnsembleMemoryModel:
    """Complete execution-adapter inventory and its separately named margin."""

    components: tuple[MemoryComponent, ...]
    allocator_margin: AllocatorMargin = AllocatorMargin()
    inventory_id: str = ""

    def __post_init__(self):
        components = tuple(self.components)
        if not components or any(not isinstance(c, MemoryComponent) for c in components):
            raise ValueError("ensemble memory model needs named components")
        if len({c.name for c in components}) != len(components):
            raise ValueError("duplicate component names would hide device memory")
        if not isinstance(self.allocator_margin, AllocatorMargin):
            raise TypeError("ensemble memory model needs an AllocatorMargin")
        object.__setattr__(self, "components", components)

    def inventory(self, members):
        members = _integer(members, "members", positive=True)
        rows = tuple(component.inventory(members) for component in self.components)
        margin = self.allocator_margin.required_bytes(rows)
        categories = {}
        for row in rows:
            categories[row["category"]] = categories.get(row["category"], 0) + row["required_bytes"]
        categories["allocator_margin"] = margin
        return {"inventory_id": self.inventory_id, "members": members, "components": rows,
                "category_bytes": categories, "allocator_margin_bytes": margin,
                "allocator_margin_evidence": self.allocator_margin.evidence,
                "required_bytes": sum(row["required_bytes"] for row in rows) + margin}

    def required_bytes(self, members):
        return self.inventory(members)["required_bytes"]

    def largest_that_fits(self, available_bytes, *, max_members):
        available = _integer(available_bytes, "available_bytes")
        maximum = _integer(max_members, "max_members", positive=True)
        low, high = 0, maximum
        while low < high:
            middle = (low + high + 1) // 2
            if self.required_bytes(middle) <= available:
                low = middle
            else:
                high = middle - 1
        return low

    @classmethod
    def from_plans(cls, plans, *, reservations=(), allocator_margin=None, inventory_id=""):
        """Bind ``{component_name: (category, BatchMemoryPlan)}`` without inference."""
        components = tuple(MemoryComponent(name, category, plan)
                           for name, (category, plan) in dict(plans).items()) + tuple(reservations)
        return cls(components, AllocatorMargin() if allocator_margin is None else allocator_margin,
                   inventory_id)


@dataclass(frozen=True)
class MemoryCalibration:
    """One measured peak bound to a concrete inventory and device fingerprint.

    Device residency includes context/kernel backing that is outside the pool.
    These are separate observations. A measurement never automatically grants
    a sharing exemption or relabels an envelope as an exact allocation.
    """

    inventory_id: str
    members: int
    device_fingerprint: str
    planned_array_bytes: int
    pool_live_peak_bytes: int
    pool_reserved_peak_bytes: int
    device_increment_peak_bytes: int
    evidence: str

    def __post_init__(self):
        for name in ("inventory_id", "device_fingerprint", "evidence"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"calibration needs {name}")
        object.__setattr__(self, "members", _integer(self.members, "calibration members", positive=True))
        for name in ("planned_array_bytes", "pool_live_peak_bytes", "pool_reserved_peak_bytes",
                     "device_increment_peak_bytes"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.pool_reserved_peak_bytes < self.pool_live_peak_bytes:
            raise ValueError("pool reservation cannot be below its live peak")

    @property
    def unpriced_live_bytes(self):
        return max(0, self.pool_live_peak_bytes - self.planned_array_bytes)

    @property
    def allocator_retention_bytes(self):
        return self.pool_reserved_peak_bytes - self.pool_live_peak_bytes

    @property
    def non_pool_bytes(self):
        return max(0, self.device_increment_peak_bytes - self.pool_reserved_peak_bytes)

    def receipt(self, model):
        if model.inventory_id != self.inventory_id:
            raise ValueError("calibration inventory differs from the execution adapter")
        inventory = model.inventory(self.members)
        actual_arrays = sum(row["array_bytes"] for row in inventory["components"])
        if actual_arrays != self.planned_array_bytes:
            raise ValueError("calibration array bytes differ from the allocation inventory")
        return {"inventory_id": self.inventory_id, "members": self.members,
                "device_fingerprint": self.device_fingerprint, "evidence": self.evidence,
                "planned_array_bytes": actual_arrays, "planned_required_bytes": inventory["required_bytes"],
                "pool_live_peak_bytes": self.pool_live_peak_bytes,
                "pool_reserved_peak_bytes": self.pool_reserved_peak_bytes,
                "device_increment_peak_bytes": self.device_increment_peak_bytes,
                "unpriced_live_bytes": self.unpriced_live_bytes,
                "allocator_retention_bytes": self.allocator_retention_bytes,
                "non_pool_bytes": self.non_pool_bytes,
                "peak_within_plan": self.device_increment_peak_bytes <= inventory["required_bytes"]}


def ordinary_memory_model_from_estimate(estimate, *, inventory_id="ordinary-preflight"):
    """Retain the ordinary preflight envelope for one live member at a time.

    Existing preflight owns every suite, nest scratch arena, radiation chunk,
    context and launch-time local-memory term. These are *envelope* components,
    not member-batched allocation claims. The ordinary executor reuses them
    between members; no N-fold resident physics bank is required.
    """
    domains = tuple(estimate.domains)
    state = sum(domain.category_bytes("state") for domain in domains)
    scratch = sum(domain.category_bytes("scratch") + domain.category_bytes("lbc")
                  + domain.category_bytes("nest") for domain in domains)
    state -= int(estimate.dycore_state_saved_bytes)
    scratch -= int(estimate.scratch_arena_saved_bytes)
    totals = {"state": state, "scratch_boundary_nest": scratch}
    categories = {item.category for domain in domains for item in domain.items} | {"physics", "diagnostic"}
    for category in sorted(categories - {"state", "scratch", "lbc", "nest", "transient", "sase"}):
        totals[category] = sum(domain.category_bytes(category) for domain in domains)
    # Shared coefficient tables (for example Thompson's) are priced once per
    # process by preflight, like the K tables, and reused between members.
    totals.update(shared_tables=int(estimate.k_tables_bytes),
                  shared_physics_tables=int(getattr(estimate, "physics_tables_bytes", 0)),
                  column_workspace=int(estimate.workspace_bytes),
                  step_transients=int(estimate.transient_peak_bytes),
                  # Coupled hydro's resident state (default-on hydro):
                  # preflight adds it to resident_bytes, so it is in the subtotal; its
                  # transient already rides transient_peak_bytes.  Without this row every
                  # hydro-covered ensemble was refused as "unclassified" (2.9.4 rolling
                  # test, WoFS from HRRR over Oklahoma, fold fe99a1680d).
                  hydro_resident=int((getattr(estimate, "hydro_memory", None) or {})
                                     .get("resident_bytes", 0)))
    subtotal = sum(totals.values())
    if subtotal != int(estimate.subtotal_bytes):
        raise ValueError("ordinary preflight inventory has an unclassified memory category")
    peak = int(estimate.peak_envelope_bytes)
    if peak < subtotal:
        raise ValueError("ordinary peak envelope is below its inventoried subtotal")
    totals["context_kernel_allocator_envelope"] = peak - subtotal
    evidence = str(estimate.envelope_basis)
    components = tuple(MemoryComponent(name, name, fixed_bytes=value, basis="envelope", evidence=evidence)
                       for name, value in totals.items())
    return EnsembleMemoryModel(components, inventory_id=inventory_id)


__all__ = ["AllocatorMargin", "EnsembleMemoryModel", "MemoryCalibration", "MemoryComponent",
           "ordinary_memory_model_from_estimate"]
