"""Move driver-held physics continuation state across a nest relocation.

TWO INVENTORIES LIVE HERE, and they are separate because their fill
rules differ.  The per-column CONTINUATION slots below cold-start on
fresh ground.  The surface-radiation CARRIERS at the end of this module
cannot: a carrier is consumed on every surface step but produced only on
the radiation cadence, so fresh ground has to arrive already carrying a
flux, and it takes the same nearest-same-landmask-class donor fill the
land-surface continuation fields take.

WHY THIS EXISTS (user report, 2026-08-16, moving nests with Kain-
Fritsch: "really weird artifacts").  A discrete relocation carries the
restart layer's serialised STATE (:func:`woof.core.nest_relocation.
relocatable_attrs`) and -- through the route preparers -- the land-
surface continuation fields.  Everything else the physics driver holds
per column was re-initialised from cold on the WHOLE child at every
accepted move: the KF NCA hold timers, the held cumulus rates,
PRATEC/RAINCV, the RAINC/RAINNC precipitation accumulators, the W0AVG
running trigger history.  Measured consequence on a moving KF nest:
convection cut off domain-wide at each move (held heating zeroed
mid-NCA), every column became simultaneously re-eligible (NCA back to
-100), the trigger memory restarted from zero (one 1/TST sample instead
of a converged mean), and the accumulated-precipitation products reset
to zero mid-run.  None of it is KF-only: RAINNC and UP_HELI_MAX reset
the same way.

THE RULE, so the inventory cannot rot: the shift set is DERIVED from
the restart registry, never hand-listed here.  What makes a slot
restart-serialised (``woof.io.restart.SERIALIZED_SCRATCH_SLOTS``) is
exactly the property -- cross-step per-column memory that nothing
recomputes -- that makes it relocation-carried, and the cumulus
adapter's array inventory (``CUMULUS_CALLABLE_ARRAYS``) rides the same
contract. Raw PBL forcing uses DRIVER_HELD_FORCING_ATTRS from that same
registry: GF/New Tiedtke may consume it before the next PBL call after a
move. The overlap retains its rates and fresh ground starts at zero.
Positive-cadence PBL rates and radiation heating ride the same inventory.
After the transplant, PhysicsDriver.recouple_after_relocation applies the
ordinary mass coupling and face interpolation on the rebuilt grid to PBL,
radiation and cumulus rates. It runs no scheme and changes no cadence.

GEOMETRY.  The overlap shifts in index space with the same
:class:`~woof.core.nest_relocation.RelocationPlan` window the
serialised-state transplant uses.  The freshly exposed strip takes each
slot's documented COLD value -- new ground has no convection memory --
which is 0 for every slot except ``cu_nca``, whose cold value is the
-100 eligibility sentinel (woof/core/physics.py driver init;
module_cu_kfeta.F:3152-3156).

ACCUMULATED PRECIPITATION belongs to the nest that integrated it. The
route preparers use core.nest_accumulation to archive the outgoing interior
in a fixed native frame and restore previously visited cells. Unvisited
strips start at zero. Products must report their sampled exposure, since the
nest cannot estimate precipitation while absent. Parent SINT still fills
prognostic fields, including hydrometeors; it does not seed rain totals.
The parent_strip_accumulations helper below retains the historical policy
only as an explicit reference for reproduction of pre-fix runs.
"""

from __future__ import annotations

import numpy as np

#: Cold-start value per registry slot for freshly exposed ground.  Every
#: slot not named here cold-starts at exactly 0.0 (the driver's own
#: zero-fill); ``cu_nca`` is the one non-zero initialisation the driver
#: performs (physics.py: ``self.cu_nca[...] = -100`` so every column is
#: eligible on the first call, WRF module_cu_kfeta.F:3152-3156).
PHYSICS_CONTINUATION_COLD_VALUES: dict[str, float] = {
    "cu_nca": -100.0,
}

#: The key the cumulus adapter's trigger history travels under.  Not a
#: scratch slot -- it lives on the callable (restart serialises it as
#: ``cumulus/w0avg``) -- so it is namespaced the same way here.
W0AVG_KEY = "cumulus/w0avg"


def continuation_slots() -> tuple[str, ...]:
    """The relocation-carried scratch inventory: the restart registry.

    One list, owned by :mod:`woof.io.restart`, answers both "what does
    a checkpoint carry" and "what does a relocation carry" for
    driver-held per-column continuation state -- the same single-list
    principle :func:`woof.core.nest_relocation.relocatable_attrs`
    applies to the serialised state.
    """
    from woof.io.restart import SERIALIZED_SCRATCH_SLOTS

    return tuple(sorted(SERIALIZED_SCRATCH_SLOTS))


def _host(value) -> np.ndarray:
    get = getattr(value, "get", None)
    if callable(get) and hasattr(value, "__cuda_array_interface__"):
        return np.ascontiguousarray(get())
    return np.ascontiguousarray(np.asarray(value))


def _array_module(value):
    if hasattr(value, "__cuda_array_interface__"):
        import cupy as cp

        return cp
    return np


def _capture(value):
    """Own the bytes across arena release without a device-to-host round trip."""
    xp = _array_module(value)
    return xp.array(value, copy=True, order="C")


def capture_continuation(state, driver, *, store=None) -> dict[str, np.ndarray]:
    """Copies of every registry slot on the outgoing child's device.

    Called while the outgoing child is whole (the runner's
    ``capture_outgoing`` seam -- host staging releases the device arrays
    right after).  Absent slots are simply not captured: a scheme that
    never allocated ``cu_*`` has no cumulus memory to move, and the
    restore side invents nothing.
    """
    def capture(name, value):
        # Shared PBL/GF rates occur in both manifests under the same key.
        if name in captured:
            return captured[name]
        if store is None:
            return _capture(value)
        # The template supplies inventory only. Copy canonical bytes directly;
        # copying its last slab first would allocate device buffers to discard.
        key = "scratch/" + name if "/" not in name else name
        if key not in store:
            raise ValueError(f"streamed continuation is missing canonical carrier {key}")
        return _host(store[key]).copy()

    captured: dict[str, np.ndarray] = {}
    existing = getattr(state, "existing_scratch", None)
    if callable(existing):
        for slot in continuation_slots():
            value = existing(slot)
            if value is not None:
                captured[slot] = capture(slot, value)
    from woof.io.restart import (CUMULUS_CALLABLE_ARRAYS,
                                  DRIVER_HELD_FORCING_ATTRS)

    for name in sorted(DRIVER_HELD_FORCING_ATTRS):
        value = getattr(driver, name, None)
        if value is not None:
            captured[f"held/{name}"] = capture(f"held/{name}", value)
    from woof.io.restart import pbl_raw_manifest, pbl_diagnostic_manifest

    for key, value in {**pbl_raw_manifest(driver),
                       **pbl_diagnostic_manifest(driver)}.items():
        captured[key] = capture(key, value)
    # Radiation's physical heating rates already have canonical storage.
    # Carry them independently of surface fluxes: both feed future physics.
    for name in ("rthratenlw", "rthratensw"):
        value = getattr(driver, name, None)
        if value is not None:
            captured[f"driver/{name}"] = capture(f"driver/{name}", value)
    # ETARA accumulates cloud state between radiation calls. It belongs to
    # the ground column and must survive the same overlap transplant.
    from woof.gfdleta_contract import GFDLETA_FIELDS
    for name in GFDLETA_FIELDS:
        value = getattr(driver, 'fields', {}).get(name)
        if value is not None:
            captured[f"fields/{name}"] = capture(f"fields/{name}", value)
    adapter = getattr(driver, "cumulus_callable", None)
    for name in sorted(CUMULUS_CALLABLE_ARRAYS):
        value = getattr(adapter, name, None)
        if value is not None:
            captured[f"cumulus/{name}"] = capture(f"cumulus/{name}", value)
    for name in ("nday", "nyear"):
        value = getattr(driver, "fields", {}).get(name)
        if value is not None:
            captured[f"fields/{name}"] = capture(f"fields/{name}", value)
    return captured


#: Integrated precipitation counters carried on the fixed nest lattice.
#: Fresh unvisited ground starts at zero. The legacy OFF control alone
#: uses parent totals. RAINSH is written as zero and has no scratch slot.
STRIP_ACCUMULATORS = ("mp_rainnc", "mp_snownc", "mp_graupelnc", "mp_hailnc",
                      "cu_rainc", "mp_cloudnc")

#: What the receipt says the rule is.
STRIP_ACCUMULATION_RULE = (
    "accumulated precipitation on ground a nest arrives on mid-run, by a "
    "move or at a spawned nest's birth, starts from the parent's there "
    "(WRF SINT, the Registry 'd' interpolation med_interp_domain applies to "
    "exposed cells), not zero, so every cell of the nest holds the rain "
    "since the run began; a nest with no cumulus accumulator carries the "
    "parent's RAINC in RAINNC so the total is kept")


def parent_strip_accumulations(parent_node, child_dc,
                               slots) -> tuple[dict[str, np.ndarray], dict]:
    """The parent's accumulated precipitation on the child's new footprint.

    Returns ``(strip, receipt)``: ``strip`` maps each child slot in
    :data:`STRIP_ACCUMULATORS` (of those ``slots`` names) to a host array of
    the child's full extent, WRF's SINT of the parent's same accumulator at
    the new placement; :func:`shift_continuation` takes its strip cells.
    The parent's RAINC goes into the child's RAINNC when the child keeps no
    RAINC (a nest that runs no cumulus scheme writes RAINC as zero), so the
    child's total is the parent's either way.  A parent that holds none of
    them (an idealized tree, a test double) seeds nothing, and the receipt
    says why.
    """
    slots = [slot for slot in STRIP_ACCUMULATORS if slot in set(slots)]
    receipt: dict[str, object] = {"rule": STRIP_ACCUMULATION_RULE,
                                  "seeded": [], "convective_in_rainnc": False}
    state = getattr(parent_node, "state", None)
    existing = getattr(state, "existing_scratch", None)
    if not slots or not callable(existing):
        receipt["reason"] = ("the child carries no accumulator slot"
                             if not slots else
                             "the parent holds no scratch accumulators")
        return {}, receipt
    # Both host-importable: this runs, and is tested, where cupy is absent
    # (the NumPy SINT is the device kernel's host twin).
    from woof.core.nest_interp import register_nest, sint

    reg = register_nest(
        nri=child_dc.parent_grid_ratio, nrj=child_dc.parent_grid_ratio,
        i_parent_start=child_dc.i_parent_start,
        j_parent_start=child_dc.j_parent_start,
        child_nx=child_dc.run.nx, child_ny=child_dc.run.ny,
        parent_nx=parent_node.cfg.run.nx, parent_ny=parent_node.cfg.run.ny,
        stagger="", wrapper="interp")

    def parent_value(slot):
        # A parent array that is not the parent's whole mass grid (a
        # streamed parent's slab template) is not the parent's rain: it
        # seeds nothing rather than interpolating a slab as if it were the
        # domain, and the receipt names it.
        value = existing(slot)
        if value is None:
            return None
        if tuple(value.shape[-2:]) != (int(reg.nyp), int(reg.nxp)):
            receipt.setdefault("not_parent_extent", {})[slot] = list(value.shape)
            return None
        return value

    strip: dict[str, np.ndarray] = {}
    for slot in slots:
        value = parent_value(slot)
        if value is None:
            continue
        strip[slot] = _host(sint(value, reg)).astype(np.float32, copy=False)
    if "cu_rainc" not in slots and "mp_rainnc" in slots:
        convective = parent_value("cu_rainc")
        if convective is not None:
            folded = _host(sint(convective, reg)).astype(np.float32)
            strip["mp_rainnc"] = (strip["mp_rainnc"] + folded
                                  if "mp_rainnc" in strip else folded)
            receipt["convective_in_rainnc"] = True
    receipt["seeded"] = sorted(strip)
    return strip, receipt


def seed_birth_accumulations(state, parent_node, child_dc) -> dict:
    """Start nest-only counters at birth, without importing parent history."""
    existing = getattr(state, "existing_scratch", None)
    slots = ([slot for slot in STRIP_ACCUMULATORS if existing(slot) is not None]
             if callable(existing) else [])
    if not getattr(getattr(child_dc, 'run', None), 'nest_accumulation_archive', True):
        seeds, receipt = parent_strip_accumulations(parent_node, child_dc, slots)
        for slot, value in seeds.items():
            existing(slot)[...] = _array_module(existing(slot)).asarray(value)
        receipt['event'] = 'birth'
        return receipt
    for slot in slots:
        existing(slot)[...] = 0
    return dict(event="birth", rule="nest-only totals since birth", seeded=slots,
                convective_in_rainnc=False, parent_precipitation=False)


def shift_continuation(captured: dict[str, np.ndarray],
                       plan, strip=None, ring: int = 0) -> dict[str, np.ndarray]:
    """Index-space shift of every captured array onto the new footprint.

    The overlap is a pure copy through ``plan.window`` -- the same
    arithmetic, and therefore the same cells, as the serialised-state
    transplant -- and the strip is the slot's cold value, or, for a slot
    ``strip`` names, that array's values there (a moving nest's fresh
    ground takes its archived nest-only total, or zero if unvisited).  For those slots the cells of the
    old footprint's ``ring``-wide specified ring that land inside the new
    footprint take ``strip`` too: the nest never accumulated there.  A
    disjoint plan (nothing shared) yields all-cold arrays, which is
    exactly what a brand-new domain would carry.
    """
    from woof.avgflx_contract import PERSISTENT as AVGFLX_PERSISTENT

    strip = strip or {}
    shifted: dict[str, np.ndarray] = {}
    for name, value in captured.items():
        xp = _array_module(value)
        # WRF mediation_nest_move calls start_domain with restart=false;
        # start_em clears the complete avgflx interval, including its count.
        if name in AVGFLX_PERSISTENT:
            shifted[name] = xp.zeros_like(value)
            continue
        if name in ("fields/nday", "fields/nyear"):
            # Domain clocks are independent of the placement and cannot be
            # shifted or donor-filled as horizontal fields.
            shifted[name] = xp.array(value, copy=True)
            continue
        cold = np.float32(PHYSICS_CONTINUATION_COLD_VALUES.get(name, 0.0))
        if name == 'fields/hbot':
            # WRF ETARA's empty-convection sentinel is KTE+1, not zero.
            cold = np.float32(captured['driver/rthratenlw'].shape[0] + 1)
        staged = xp.full_like(value, cold)
        seed = strip.get(name)
        if seed is not None and tuple(seed.shape) != tuple(value.shape):
            seed = None
        if seed is not None:
            seed = xp.asarray(seed)
            staged[...] = seed
        window = plan.window(value.shape)
        if window is not None:
            (dst_j, src_j), (dst_i, src_i) = window
            staged[..., dst_j, dst_i] = value[..., src_j, src_i]
            if seed is not None and ring > 0:
                stale = _old_ring_inside(value.shape[-2:], window, ring)
                stale = xp.asarray(stale)
                staged[..., stale] = seed[..., stale]
        shifted[name] = staged
    return shifted


def _old_ring_inside(shape, window, ring: int) -> np.ndarray:
    """The new footprint's cells that were the old footprint's specified
    ring, the cells the nest's microphysics never touched."""
    from woof.core.physics_inventory import spec_zone_ring_slices

    ny, nx = (int(n) for n in shape)
    old = np.zeros((ny, nx), dtype=bool)
    for section in spec_zone_ring_slices(ny, nx, int(ring)):
        old[section] = True
    (dst_j, src_j), (dst_i, src_i) = window
    stale = np.zeros((ny, nx), dtype=bool)
    stale[dst_j, dst_i] = old[src_j, src_i]
    return stale


def accumulation_ring(run) -> int:
    """The specified ring width the nest's microphysics skips: WRF's
    ``spec_zone`` on a specified or nested domain, else 0
    (woof/core/microphysics.py ``_ring_guard_slices``)."""
    if not (getattr(run, "specified", False) or getattr(run, "nested", False)):
        return 0
    return max(int(getattr(run, "spec_zone", 0) or 0), 0)


def restore_continuation(state, driver,
                         shifted: dict[str, np.ndarray]) -> dict[str, object]:
    """Write the shifted continuation onto the rebuilt child, in place.

    The driver's ``cu_*``/``mp_*`` members alias the state's canonical
    scratch slots (woof/io/restart.py driver-alias classification), so
    writing ``state.scratch(slot)[...]`` updates the driver's own arrays
    with no second copy.  W0AVG is re-bound the way the restart restore
    binds it: the array attaches to the adapter and ``_history_state``
    points at the NEW state, so the adapter's identity check does not
    re-zero the history on its next due call.
    """
    moved: list[str] = []
    cold_only: list[str] = []
    cold_flags = []
    for slot in continuation_slots():
        staged = shifted.get(slot)
        if staged is None:
            continue
        target = state.scratch(staged.shape, slot, dtype=staged.dtype)
        if hasattr(target, "__cuda_array_interface__"):
            import cupy as cp

            target[...] = cp.asarray(staged)
        else:
            target[...] = staged
        moved.append(slot)
        cold_flags.append((slot, staged.any()))
    from woof.io.restart import DRIVER_HELD_FORCING_ATTRS

    for name in sorted(DRIVER_HELD_FORCING_ATTRS):
        key = f"held/{name}"
        staged = shifted.get(key)
        if staged is None:
            continue
        target = getattr(driver, name, None)
        if target is None or target.shape != staged.shape:
            raise ValueError(f"relocated PBL carrier {key} has no matching target")
        if hasattr(target, "__cuda_array_interface__"):
            import cupy as cp

            target[...] = cp.asarray(staged)
        else:
            target[...] = staged
        moved.append(key)
        cold_flags.append((key, staged.any()))
    from woof.io.restart import pbl_raw_manifest, pbl_diagnostic_manifest

    raw_targets = {**pbl_raw_manifest(driver), **pbl_diagnostic_manifest(driver)}
    from woof.gfdleta_contract import GFDLETA_FIELDS
    for name in (*GFDLETA_FIELDS, "nday", "nyear"):
        value = getattr(driver, "fields", {}).get(name)
        if value is not None:
            raw_targets[f"fields/{name}"] = value
    for name in ("rthratenlw", "rthratensw"):
        value = getattr(driver, name, None)
        if value is not None:
            raw_targets[f"driver/{name}"] = value
    for key, target in raw_targets.items():
        staged = shifted.get(key)
        if staged is None or key in moved:
            continue
        if tuple(target.shape) != tuple(staged.shape):
            raise ValueError(f"relocated physics carrier {key} has no matching target")
        if hasattr(target, "__cuda_array_interface__"):
            import cupy as cp

            target[...] = cp.asarray(staged)
        else:
            target[...] = staged
        moved.append(key)
        cold_flags.append((key, staged.any()))
    w0avg_moved = False
    from woof.io.restart import CUMULUS_CALLABLE_ARRAYS
    adapter = getattr(driver, "cumulus_callable", None)
    for name in sorted(CUMULUS_CALLABLE_ARRAYS):
        key = f"cumulus/{name}"
        staged = shifted.get(key)
        if staged is None or adapter is None:
            continue
        current = getattr(adapter, name, None)
        if (current is not None and tuple(current.shape) == tuple(staged.shape)
                and current.dtype == staged.dtype):
            _array_module(current).copyto(current, _array_module(current).asarray(staged))
        else:
            sample = getattr(state, "w", staged)
            setattr(adapter, name, _array_module(sample).array(staged, copy=True))
        adapter._history_state = state
        # The rebuilt clock has passed the outgoing sample. The next due
        # call adds a new sample and preserves every retained cloud carrier.
        adapter._history_time = None
        if name == "w0avg":
            w0avg_moved = True
        else:
            moved.append(key)
            cold_flags.append((key, staged.any()))
    if cold_flags:
        # One small diagnostic transfer, rather than a synchronization for
        # every carried physics array.
        xp = next((_array_module(flag) for _, flag in cold_flags
                   if hasattr(flag, "__cuda_array_interface__")), np)
        flags = _host(xp.stack([flag for _, flag in cold_flags]))
        cold_only = [name for (name, _), flag in zip(cold_flags, flags) if not flag]
    return {
        "registry": "woof.io.restart.SERIALIZED_SCRATCH_SLOTS"
                    " + CUMULUS_CALLABLE_ARRAYS + DRIVER_HELD_FORCING_ATTRS",
        "slots_moved": moved,
        "slots_all_cold": cold_only,
        "w0avg_moved": w0avg_moved,
    }


# ---------------------------------------------------------------------------
# The surface-radiation carriers
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS (test host B GPU campaign, 2026-08-24).  A relocation
# rebuilds the moved child's physics driver from cold
# (`woof.runtime.rebuild_child_driver_from_land_state` ->
# `initialize_physics`), which allocates a fresh buffer for every
# radiative carrier and seeds a fresh CarrierContract in which glw and
# swdown are `unwritten`.  Neither the buffers nor the ledger were in
# any carry set, so after every accepted move the moved domain's
# radiation provenance read "nothing has ever written this buffer" and
# the first surface call before the next due radiation call refused:
#
#   CarrierContractError: GLW (downward longwave at the surface, W m-2)
#   has no producer, and Noah (sf_surface_physics=2) is about to consume
#   it at model second 360.
#
# MEASURED, one forecast hour per arm on a real GFS case: relocation off
# with radt = 12 min passed; relocation on with radt = 12 min refused at
# the first move; relocation on with radt = 6 min -- aligned so a
# radiation call fell due on the very step the 360 s move landed on --
# passed with five executed moves.  Under the wizard's shipped defaults
# ANY relocation cadence shorter than the radiation interval refused at
# the first move.  The producer guard is correct; the transplant was
# short by exactly this inventory.
#
# THE RULE, so this inventory cannot rot either: the carried set is
# DERIVED from `radiation_carriers.CONSUMER_CARRIERS`, the same matrix
# the refusal reads to decide what a land-surface scheme eats.  A
# carrier added to that matrix tomorrow moves across relocations the day
# it is added.
#
# WHAT MOVES, in two halves that must both move or neither is worth
# anything.  The BUFFERS shift in index space through the same plan
# window as the serialised state, and the freshly exposed strip takes
# the nearest same-landmask-class donor inside the overlap -- the
# identical `DonorFillPlan` the land-surface continuation fields take,
# because a carrier is exactly that kind of field: consumed every
# surface step, produced only on the radiation cadence.  A zero strip
# would be a fabricated flux and a cold-start strip would be the
# allocation fill the contract exists to refuse.  The LEDGER moves
# VERBATIM -- source and last-write model time both -- because the
# second half of the same guard is a staleness test against
# (radiation cadence + one step), and re-stamping the move time would
# blind it: the transplanted flux really is as old as the outgoing
# child's, and saying so is what keeps "a producer that stopped"
# detectable across a move.


def relocatable_carriers() -> tuple[str, ...]:
    """The surface-radiation carriers a relocation carries.

    The consumer matrix, and nothing hand-listed beside it: whatever a
    land-surface scheme is checked for before it consumes is exactly
    what a relocation has to keep, which is the same single-list
    principle :func:`continuation_slots` and
    :func:`woof.core.nest_relocation.relocatable_attrs` apply.
    """
    from woof.core.radiation_carriers import CONSUMER_CARRIERS

    names: set[str] = set()
    for carriers in CONSUMER_CARRIERS.values():
        names.update(carriers)
    return tuple(sorted(names))


def capture_carriers(driver, *, store=None, scalars=None) -> dict[str, object]:
    """Device copies of the outgoing child's carriers, plus its ledger.

    Called from the preparer's ``capture_outgoing`` seam while the
    outgoing child is still whole.  A carrier the configuration never
    allocated is simply not captured -- a Noah run has no GSW to move --
    and a driver assembled without a contract yields ``None``, which the
    restore side reports rather than papers over.
    """
    fields: dict[str, np.ndarray] = {}
    contract = None
    if driver is not None:
        driver_fields = getattr(driver, "fields", None) or {}
        for name in relocatable_carriers():
            value = driver_fields.get(name)
            if value is not None:
                if store is None:
                    fields[name] = _capture(value)
                else:
                    key = "fields/" + name
                    if key not in store:
                        raise ValueError(f"streamed radiation is missing canonical carrier {key}")
                    fields[name] = _host(store[key]).copy()
        carriers = getattr(driver, "carriers", None)
        if carriers is not None and store is None:
            contract = carriers.state()
    if store is not None:
        if scalars is None:
            raise ValueError("streamed radiation capture requires canonical scalar carriers")
        from copy import deepcopy
        contract = deepcopy(scalars.get("carriers"))
    return {"fields": fields, "contract": contract}


def shift_carriers(captured: dict[str, np.ndarray], plan,
                   fill) -> dict[str, np.ndarray]:
    """Index-space shift of every captured carrier, strip donor-filled.

    ``fill`` is the move's own :class:`~woof.ingest.relocation_init.
    DonorFillPlan` -- the one the land-surface continuation fields
    already ride -- so a carrier and the skin temperature it drives
    arrive on the fresh strip from the SAME donor column.  It is a
    required argument, not an optional one: there is no admissible
    degraded fill for a flux, and a strip left at zero is the fabricated
    measurement the carrier contract exists to refuse.
    """
    if fill is None:
        raise ValueError(
            "shift_carriers requires the move's donor fill plan; a "
            "radiation carrier consumed on the freshly exposed strip has "
            "no cold value -- zero is a fabricated flux and the "
            "allocation fill is what the carrier contract refuses -- so "
            "the strip must come from a real donor column")
    shifted: dict[str, np.ndarray] = {}
    for name, value in captured.items():
        staged = _array_module(value).zeros_like(value)
        window = plan.window(value.shape)
        if window is None:
            continue
        (dst_j, src_j), (dst_i, src_i) = window
        staged[..., dst_j, dst_i] = value[..., src_j, src_i]
        shifted[name] = fill.apply(staged)
    return shifted


def restore_carriers(driver, shifted: dict[str, np.ndarray],
                     contract) -> dict[str, object]:
    """Write the shifted carriers and their provenance onto the rebuilt child.

    The buffers land in the driver's own ``fields`` mapping, which is
    the same storage the radiation seam writes and the consumption check
    reads, so there is no second copy to disagree with.  The ledger is
    re-established through :meth:`~woof.core.radiation_carriers.
    CarrierContract.restore`, the restart layer's own entry point, with
    the outgoing child's model times intact.
    """
    moved: list[str] = []
    absent: list[str] = []
    fields = getattr(driver, "fields", None)
    fields = {} if fields is None else fields
    for name in relocatable_carriers():
        staged = shifted.get(name)
        if staged is None:
            continue
        target = fields.get(name)
        if target is None:
            absent.append(name)
            continue
        if tuple(target.shape) != tuple(staged.shape):
            from woof.core.nest_relocation import RelocationRefusal

            raise RelocationRefusal(
                f"carrier {name!r} is {tuple(staged.shape)} on the "
                f"outgoing child and {tuple(target.shape)} on the "
                "incoming one; a relocation changes position, never "
                "extent")
        if hasattr(target, "__cuda_array_interface__"):
            import cupy as cp

            target[...] = cp.asarray(staged, dtype=target.dtype)
        else:
            target[...] = staged.astype(target.dtype, copy=False)
        moved.append(name)
    carriers = getattr(driver, "carriers", None)
    restored = False
    if carriers is not None and contract is not None:
        carriers.restore(contract)
        restored = True
    return {
        "matrix": "woof.core.radiation_carriers.CONSUMER_CARRIERS",
        "restored": True,
        "carriers_moved": moved,
        "carriers_absent": absent,
        "ledger_restored": restored,
        "ledger": {} if contract is None else {
            name: dict(row) for name, row in sorted(contract.items())},
    }


__all__ = [
    "PHYSICS_CONTINUATION_COLD_VALUES", "STRIP_ACCUMULATION_RULE",
    "STRIP_ACCUMULATORS", "W0AVG_KEY",
    "accumulation_ring", "capture_carriers", "capture_continuation",
    "continuation_slots", "parent_strip_accumulations",
    "relocatable_carriers", "restore_carriers", "restore_continuation",
    "seed_birth_accumulations", "shift_carriers", "shift_continuation",
]
