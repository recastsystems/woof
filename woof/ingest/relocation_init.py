"""Real-data child rebuild at a new placement (statics-on-move, leg 3).

The 2026-08-06 requirement (project ruling, via the WRF moving-nest discussion): a
relocated nest's STATIC fields -- terrain, landuse, soil categories --
must be REBUILT for the new footprint from the nest's own static source
at nest resolution, never inherited parent-interpolated, because
over-land storm-following at 2 km lives on resolved terrain.  This
module is the initializer that satisfies it on the real-data routes, and
it is deliberately assembled from the SAME machinery the t=0 nest cold
start uses (:mod:`woof.ingest.nest_init`) rather than a re-derivation:

1. **Footprint-rebuilt statics.**  :func:`woof.static.build.
   build_static_for_domain` is footprint-parametric and is invoked for
   the new footprint (plus the ``[static.highres]`` overlay when the
   case enables it, whose tile cache makes repeat moves cheap).  The
   footprint's grid comes from :meth:`ProjectedGrid.translated` -- the
   reference grid moved by an exact whole number of its own cells -- so
   a cell two placements share evaluates to bitwise-identical
   coordinates and therefore bitwise-identical statics (identical
   source + identical cells = identical bytes).  That equality is what
   lets the bitwise overlap transplant survive a statics rebuild, and
   the route's preparer ASSERTS it on every move rather than assuming
   it.

2. **Fresh-strip atmosphere on the new terrain.**  The atmosphere at
   move time exists only in the live parent, so the child is filled by
   the ordinary full-parent SINT (:func:`~woof.ingest.nest_init.
   parent_only_init`) and then adjusted to the rebuilt fine terrain by
   the SAME sequence the t=0 real child runs after its own fine-terrain
   analysis: the analytic fine base (``_make_real_base``), WRF's
   three-operand terrain blend (``blend_terrain`` on ht/mub/phb against
   the parent SINT captures), ``adjust_tempqv`` for the base-column-mass
   change, the ``start_domain`` base/EOS re-derivation, and the
   real-nest ``press_adj`` MU correction -- one call,
   ``_adjust_and_rederive``, reused verbatim.  On overlap cells outside
   both placements' blend frames this reproduces the t=0 base bitwise
   (the analytic base is a per-column pure function of the terrain
   bytes), which is what keeps the donor-alignment instrument armed.

3. **Blend frames resplit against the rebuilt base, and must.**  Inside
   the two placements' blend frames the child base states genuinely
   differ -- ``blend_terrain`` makes a cell's EFFECTIVE terrain a
   function of its ROW, so a move that changes a column's row changes
   the ground under it.  The perturbations are carried BITWISE anyway
   (WRF's ``mediation_nest_move.F`` does exactly that), which lets the
   column totals change by the base-state slab between the two effective
   terrains -- the mass and heat of the air below the old numerical
   floor.  :func:`rederive_after_transplant` (installed as the
   initializer's ``post_transplant`` hook) counts the base-changed cells
   and re-derives the EOS.  Preserving the totals there instead REFUSES
   that slab and manufactures a hydrostatic hole; it was built, measured
   on Melissa's Andes-frame move, and retired.  See that function.

4. **Land state moves by donor fill.**  Soil, snow and surface fields
   live on the physics driver, outside the serialised-state transplant;
   leg 1's contract moves them with a second transplant against the
   same plan, and the newly exposed strip is filled from its nearest
   same-landmask-class donor inside the overlap
   (:func:`donor_fill_plan`), with the counts recorded on the move
   receipt.  The route-side driver rebuild lives in
   :class:`woof.runtime.RealRelocationChildPreparer`.

Nothing here decides when or where to move.  A restart across a move is
no longer refused and no longer promises nothing: it reproduces the run
that wrote the checkpoint bit for bit, which puts this module's fill on
the resume path as well as the live one.  See
:data:`woof.core.nest_relocation.RESTART_ACROSS_MOVE_POSTURE`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from woof.core.nest_relocation import RelocationRefusal

#: The receipt-stated provenance for statics rebuilt by this initializer.
REAL_DATA_FOOTPRINT_REBUILT_STATICS = (
    "footprint-rebuilt own-source statics: build_static_for_domain on the "
    "placement-translated child grid (30s baseline, [static.highres] "
    "overlay when the case enables it); terrain adjustment via the t=0 "
    "nest cold-start sequence (blend_terrain + adjust_tempqv + "
    "start_domain re-derivation + press_adj)")

#: What fills the strip a move exposes under this initializer.
REAL_DATA_STRIP_FILL_SOURCE = (
    "full-parent SINT adjusted to footprint-rebuilt fine terrain "
    "(t=0 nest cold-start lineage); overlap then stamped bitwise from "
    "the outgoing child, blend-frame perturbations carried bitwise and "
    "resplit against the rebuilt base; land-surface state donor-filled "
    "per leg-1's contract")

#: Driver-held land-surface continuation state a move carries: the Noah
#: prognostics, snow state, and the seeded surface diagnostics.  Fields a
#: configuration does not allocate are skipped by name, exactly like the
#: serialized-state transplant.  Accumulators (RAINC/RAINNC lineage) are
#: deliberately NOT here: they are re-initialised at the new placement --
#: leg 1's documented contract -- and the receipt says so.
from woof.ocean.mixed_layer import STATE_FIELDS as PWP_CONTINUATION_FIELDS

LAND_SURFACE_CONTINUATION_FIELDS = (
    "tsk", "tslb", "smois", "sh2o", "smcrel",
    "canwat", "snow", "snowh", "snowc", "snotime",
    "xice", "qsfc", "ust",
    "psfc", "t2", "q2", "th2", "u10", "v10",
    *PWP_CONTINUATION_FIELDS,
)


def surface_continuation_fields(cfg, *, interpolate=False):
    """Carry active surface state; only Registry ``d`` fields interpolate.

    DTW and TSK_SAVE have restart flags only. They survive a relocation's
    overlap transplant, but a newborn column retains its own initialization.
    Domain counters NDAY/NYEAR travel separately, without interpolation.
    """
    names = list(LAND_SURFACE_CONTINUATION_FIELDS)
    if getattr(cfg, "sst_update", 0):
        names += ["sst", "xicem", "tmn", "xland", "ivgtyp", "isltyp",
                  "vegfra", "albedo", "albbck", "emiss", "embck"] if not interpolate else ["xicem", "tmn", "vegfra"]
        if not interpolate:
            names += ["xice_save", "tsk_save"]
    if getattr(cfg, "sst_skin", 0):
        names += ["sstsk"]
        if not interpolate:
            names += ["dtw"]
    if getattr(cfg, "tmn_update", 0):
        names += ["tmn", "tyr", "tyra", "tdly", "tlag"]
    return tuple(dict.fromkeys(names))

#: Statics the overlap-equality assertion refuses on.  Everything the
#: model or driver consumes; compared bitwise on shared ground.
OVERLAP_STATIC_EQUALITY_FIELDS = (
    "HGT_M", "LANDMASK", "LU_INDEX", "LANDUSEF",
    "SOILCTOP", "SCT_DOM", "SOILCBOT", "SCB_DOM",
    "GREENFRAC", "LAI12M", "ALBEDO12M", "SNOALB", "SOILTEMP", "TMN",
)

#: (perturbation, base) pairs whose split changes when the base changes.
#: Counted and reported per move; the perturbations themselves are
#: CARRIED BITWISE, WRF's own move behaviour -- see
#: :func:`rederive_after_transplant` for why the totals-preserving
#: alternative was built, measured, and retired.
_SPLIT_PAIRS = (("thp", "thb"), ("php", "phb"), ("mup", "mub2d"))


def _host(value, dtype=None) -> np.ndarray:
    if hasattr(value, "get") and hasattr(value, "__cuda_array_interface__"):
        value = value.get()
    array = np.asarray(value)
    return array if dtype is None else np.asarray(array, dtype=dtype)


# ---------------------------------------------------------------------------
# Overlap statics equality: asserted, never assumed
# ---------------------------------------------------------------------------

def overlap_statics_mismatches(old_fields, new_fields, plan,
                               names=OVERLAP_STATIC_EQUALITY_FIELDS
                               ) -> dict[str, object]:
    """Count per-field mismatches on the shared ground.

    ``old_fields`` are the outgoing footprint's statics, ``new_fields``
    the rebuilt ones; ``plan`` supplies the index-space windows
    (``new[dst] <-> old[src]``).  Identical source + identical cells
    must give identical VALUES; the caller refuses on any count in
    ``mismatched_fields``.

    ONE ULP IS TOLERATED ON FLOAT FIELDS, MEASURED, NOT ASSUMED.  The
    climatology resamplers accumulate source pixels in an order that
    shifts with the source crop, and floating-point addition is not
    associative, so the same pixels over the same cell can land one ULP
    apart between two footprints.  Measured on the 2011-04-27 ERA5
    moving-nest case (work/complaint2/probe_greenfrac2.py): a 2-parent-
    cell move rebuilt GREENFRAC bitwise-identical on the whole overlap
    except THREE month-values of one west-edge cell, each exactly one
    float64 ULP off (5.6e-17 on 0.335) -- and the refusal killed a
    6-hour forecast at its first relocation over it.  Adjacent floats
    are physics-identical ground; a category flip, a different pixel
    set, or any drift beyond adjacency is still a statics-build defect
    and still refuses.  Within-tolerance counts are REPORTED per field
    (``within_one_ulp``), never silent, mirroring the sixteen_pt
    clamp's bounded-margin-with-counted-advisory posture.
    """
    fields: dict[str, int] = {}
    within: dict[str, int] = {}
    compared_cells = 0
    for name in names:
        old = old_fields.get(name)
        new = new_fields.get(name)
        if old is None or new is None:
            continue
        old = np.asarray(old)
        new = np.asarray(new)
        if old.shape != new.shape:
            fields[name] = max(int(old.size), int(new.size), 1)
            continue
        window = plan.window(old.shape)
        if window is None:
            continue
        (dst_j, src_j), (dst_i, src_i) = window
        actual = np.ascontiguousarray(new[..., dst_j, dst_i])
        expected = np.ascontiguousarray(old[..., src_j, src_i])
        if actual.dtype.kind == "f":
            unequal = actual != expected
            # One-ULP adjacency, exactly: stepping the smaller value
            # toward the larger reaches it.  NaN fails every compare and
            # therefore lands in the refusing count, deliberately.
            adjacent = unequal & (np.nextafter(actual, expected)
                                  == expected)
            beyond = int(np.count_nonzero(unequal & ~adjacent))
            fields[name] = beyond
            ulp_count = int(np.count_nonzero(adjacent))
            if ulp_count:
                within[name] = ulp_count
        else:
            fields[name] = int(np.count_nonzero(actual != expected))
        compared_cells += int(actual.size)
    return {
        "fields": fields,
        "compared_cells": int(compared_cells),
        "mismatched_fields": {name: count for name, count in fields.items()
                              if count},
        "within_one_ulp": within,
        "pass": bool(compared_cells) and not any(fields.values()),
    }


# ---------------------------------------------------------------------------
# Donor fill: land-surface state for the strip
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DonorFillPlan:
    """Nearest-donor assignment for every strip cell, with provenance.

    ``donor_flat`` maps each cell of the new footprint to the flat index
    of the overlap cell that donates its land-surface state; overlap
    cells map to themselves.  Donors are exact-Euclidean nearest within
    the overlap, matched on the NEW footprint's landmask class (the
    donor's class equals the outgoing child's on shared ground by the
    overlap-statics equality this move already asserted); a strip cell
    whose class has no donor anywhere in the overlap falls back to the
    nearest overlap cell of any class, and that fallback is COUNTED,
    never silent.
    """

    donor_flat: np.ndarray
    counts: dict[str, int]

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Fill one field's strip cells from their donors (trailing y, x)."""
        from woof.core.physics_continuation import _array_module

        xp = _array_module(values)
        values = xp.asarray(values)
        ny, nx = self.donor_flat.shape
        if values.shape[-2:] != (ny, nx):
            raise ValueError(
                f"donor fill built for (ny, nx) = {(ny, nx)}, got a field "
                f"with trailing shape {values.shape[-2:]}")
        flat = values.reshape(values.shape[:-2] + (ny * nx,))
        donors = xp.asarray(self.donor_flat.ravel())
        return flat[..., donors].reshape(values.shape)


def _nearest_donor(target_yx: np.ndarray, donor_yx: np.ndarray, *,
                   device: bool = False) -> np.ndarray:
    """Exact nearest donor per target (first index breaks ties)."""
    if device and len(target_yx):
        import cupy as cp
        from woof.core.kernels import get_kernel

        kernel = get_kernel("nest", "relocation_nearest_donor")
        picked = cp.empty(len(target_yx), dtype=cp.int64)
        kernel((len(target_yx),), (128,), (
            cp.asarray(target_yx, dtype=cp.int64),
            cp.asarray(donor_yx, dtype=cp.int64), picked,
            np.int32(len(donor_yx))))
        return picked.get()
    out = np.empty(target_yx.shape[0], dtype=np.int64)
    chunk = max(1, int(4e6) // max(1, donor_yx.shape[0]))
    for start in range(0, target_yx.shape[0], chunk):
        stop = min(start + chunk, target_yx.shape[0])
        block = target_yx[start:stop]
        d2 = ((block[:, None, 0] - donor_yx[None, :, 0]) ** 2
              + (block[:, None, 1] - donor_yx[None, :, 1]) ** 2)
        out[start:stop] = np.argmin(d2, axis=1)
    return out


def donor_fill_plan(*, overlap_mask: np.ndarray,
                    landmask: np.ndarray, device: bool = False) -> DonorFillPlan:
    """Build the strip's donor assignment from the overlap and landmask."""
    overlap_mask = np.asarray(overlap_mask, dtype=bool)
    land = np.asarray(landmask) >= 0.5
    if overlap_mask.shape != land.shape:
        raise ValueError(
            f"overlap mask {overlap_mask.shape} and landmask "
            f"{land.shape} disagree")
    ny, nx = overlap_mask.shape
    if not overlap_mask.any():
        raise RelocationRefusal(
            "donor fill requires a non-empty overlap; a disjoint move is "
            "refused upstream and cannot reach here")
    donor_flat = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)
    counts = {
        "strip_cells": int(np.count_nonzero(~overlap_mask)),
        "overlap_cells": int(np.count_nonzero(overlap_mask)),
        "land_donor_filled": 0,
        "water_donor_filled": 0,
        "class_fallback_filled": 0,
    }
    yy, xx = np.mgrid[0:ny, 0:nx]
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float64)
    flat_overlap = overlap_mask.ravel()
    flat_land = land.ravel()
    unresolved = np.zeros(ny * nx, dtype=bool)
    for is_land, label in ((True, "land_donor_filled"),
                           (False, "water_donor_filled")):
        targets = ~flat_overlap & (flat_land == is_land)
        if not targets.any():
            continue
        donors = np.flatnonzero(flat_overlap & (flat_land == is_land))
        if donors.size == 0:
            unresolved |= targets
            continue
        picked = donors[_nearest_donor(coords[targets], coords[donors], device=device)]
        donor_flat.ravel()[np.flatnonzero(targets)] = picked
        counts[label] = int(np.count_nonzero(targets))
    if unresolved.any():
        donors = np.flatnonzero(flat_overlap)
        picked = donors[_nearest_donor(coords[unresolved], coords[donors], device=device)]
        donor_flat.ravel()[np.flatnonzero(unresolved)] = picked
        counts["class_fallback_filled"] = int(np.count_nonzero(unresolved))
    return DonorFillPlan(donor_flat=donor_flat, counts=counts)


def overlap_mask_for_plan(plan, shape) -> np.ndarray:
    """Boolean (ny, nx) mask of the new footprint's overlap cells."""
    mask = np.zeros(tuple(int(n) for n in shape), dtype=bool)
    window = plan.window(mask.shape)
    if window is not None:
        (dst_j, _sj), (dst_i, _si) = window
        mask[dst_j, dst_i] = True
    return mask


# ---------------------------------------------------------------------------
# After the transplant: carry perturbations bitwise, re-derive the EOS
# ---------------------------------------------------------------------------

def relocation_base_changed_cells(source_state, target_state, plan):
    """Count changed base bytes independently of the diagnostic backend."""
    base_changed: dict[str, int] = {}
    for pert_name, base_name in _SPLIT_PAIRS:
        pert_target = getattr(target_state, pert_name, None)
        base_target = getattr(target_state, base_name, None)
        base_source = getattr(source_state, base_name, None)
        if any(value is None for value in (pert_target, base_target,
                                           base_source)):
            continue
        window = plan.window(np.shape(pert_target))
        if window is None:
            continue
        (dst_j, src_j), (dst_i, src_i) = window
        base_out = _host(base_source, np.float32)[..., src_j, src_i]
        base_in = _host(base_target, np.float32)[..., dst_j, dst_i]
        differs = base_out.view(np.uint32) != base_in.view(np.uint32)
        base_changed[pert_name] = int(np.count_nonzero(differs))
    return base_changed


def rederive_after_transplant(*, source_state, target_state,
                              plan, cfg) -> dict[str, object]:
    """Re-derive the EOS on the rebuilt child; perturbations stay bitwise.

    Inside either placement's blend frame the base states differ -- the
    blend makes a cell's EFFECTIVE terrain a function of its ROW, so a
    move that changes a column's row changes the ground under it, by
    hundreds of metres where the fine and parent terrain disagree.  The
    stamped perturbations are deliberately left alone there: against the
    rebuilt base the split re-derives WRF's own way (``mediation_nest_move.F``
    carries the perturbation arrays and recomputes nothing), and the
    column totals CHANGE by exactly the base-state slab between the two
    effective terrains, which is the mass, heat and geopotential of the
    air that was below the old numerical floor.

    A totals-preserving alternative was built here first, and it is
    retired because it was MEASURED to manufacture the imbalance it
    meant to prevent.  Preserving column totals across an effective-
    terrain DROP refuses the slab: on Melissa's eighth relocation
    (2025-10-22 02:00, d02's southern blend frame on the Colombian
    Andes, fine terrain 2271 m against parent ~1100 m) it left ``MU``
    at -4590 Pa -- a 4.6 kPa dry-mass hole against the column's own
    terrain -- and the model surface 100 m below the ground
    (``z(k=0)`` 1001.8 m under ``HGT`` 1104.5 m), and the column went
    non-finite six d02 steps later.  Carrying the perturbations bitwise
    survived the same move on the same bytes.  Over water the two
    behaviours are identical -- fine and parent terrain agree, the base
    bytes match, there is nothing to split differently -- which is why
    every ocean-track case was blind to the difference.

    The base-changed cells are still counted per pair, per move: the
    counts are the receipt's statement of how much ground changed role,
    and a nonzero count over water would again be the wiring defect the
    old instrument was hunting.  The EOS diagnostics are then re-derived
    so ``p``/``al``/``alt`` agree with the carried perturbations against
    the rebuilt base.
    """
    from woof.core.diagnostics import update_diagnostics

    base_changed = relocation_base_changed_cells(source_state, target_state, plan)
    update_diagnostics(target_state, cfg.hypsometric_opt)
    return {
        "pairs": {pert: base for pert, base in _SPLIT_PAIRS},
        "base_changed_cells": base_changed,
        "perturbation_carry": (
            "bitwise (WRF mediation_nest_move lineage); totals resplit "
            "against the rebuilt base"),
        "diagnostics_rederived": True,
    }


def clamp_interpolation_undershoot(state) -> dict[str, int]:
    """Zero-floor the interpolation-filled nonnegative fields, counted.

    The fresh strip is filled by SINT of the live parent, and SINT's
    stencil rings: near a sharp hydrometeor edge it can undershoot a
    nonnegative field by an epsilon-scale negative (first seen on the
    first real-data move with active microphysics -- one Morrison rain-
    number cell at -1.8e-15 -- which the idealized proofs could never
    expose because their hydrometeors were identically zero at move
    time).  A rebuilt child must satisfy the state's own health
    invariants BEFORE its first step, and WRF's own microphysics drivers
    clamp these fields internally anyway, so the defined behaviour is an
    exact zero floor on exactly the fields whose shared health rule says
    ``>= 0`` (non-strict): the authority is
    :func:`woof.core.health.rule_for_field`, never a private list, so
    a field added to the model reaches both the gate and this floor at
    once.  Strict-lower fields (coupled mass, total specific volume) are
    deliberately untouched -- zero is illegal for them and a violation
    there must FAIL the gate, not be papered over.  Counts are returned
    for the receipt, never silent.
    """
    from woof.core.health import rule_for_field
    from woof.core.nest_relocation import relocatable_attrs

    clamped: dict[str, int] = {}
    for name in relocatable_attrs():
        value = getattr(state, name, None)
        if value is None:
            continue
        rule = rule_for_field(name)
        if rule.lower != 0.0 or rule.strict_lower:
            continue
        negative = value < 0
        count = int(negative.sum())
        if count:
            value[negative] = 0
            clamped[name] = count
    return clamped


# ---------------------------------------------------------------------------
# The initializer factory
# ---------------------------------------------------------------------------

def _build_footprint_statics(grid, catalog, child_dc):
    """The t=0 static build, verbatim inputs, on the supplied grid."""
    from woof.ingest.nest_init import _static_catalog
    from woof.static.build import (build_static_for_domain,
                                    geog_selection_from_catalog)

    static_catalog = _static_catalog(catalog)
    fields = build_static_for_domain(
        grid, static_catalog, child_dc.grid_id,
        **({"cfg": child_dc.run}
           if getattr(child_dc.run, "sf_lake_physics", 0) else {}))
    selection = geog_selection_from_catalog(static_catalog, child_dc.grid_id)
    landuse_attrs = selection.landuse_global_attrs()
    highres = getattr(catalog, "static_highres", None)
    highres_applied = False
    if highres is not None and getattr(highres, "enabled", False):
        from woof.static.highres_production import apply_highres_statics

        # Same case_date input as the t=0 build: statics are a
        # time-invariant property of the domain, and a drifting date here
        # would break the overlap equality against the t=0 footprint.
        fields, overlay_receipt = apply_highres_statics(
            fields, grid, config=highres, domain_id=child_dc.grid_id,
            case_date=child_dc.start_time.date(),
            landuse_attrs=landuse_attrs)
        # A block scoped by grid spacing leaves a coarser domain alone.
        highres_applied = overlay_receipt is not None
    return fields, selection, landuse_attrs, highres_applied


def real_relocation_initializer(*, catalog=None, vertical, child_config,
                                reference_grid, reference_i_parent_start,
                                reference_j_parent_start,
                                drift_tolerance_deg: float = 1.0e-8,
                                statics_builder=None,
                                root_frame_shift=None):
    """Build the per-footprint rebuild seam for one real-data child.

    ``reference_grid`` is the child's grid as originally placed
    (``reference_i/j_parent_start``); every relocated footprint is that
    grid translated by an exact whole number of child cells, which is
    the construction that makes footprint-rebuilt statics bitwise-equal
    on shared ground (see the module docstring).  The returned
    ``initialize(new_dc, parent_node, ...)`` satisfies
    :func:`woof.core.nest_relocation.relocate_child`'s initializer
    contract, states its provenance, scopes the donor-alignment
    instrument to the region where its invariant is defined
    (``donor_alignment_frame_width``), and installs the post-transplant
    EOS re-derivation as its ``post_transplant`` hook.

    WHERE THE STATICS COME FROM is the one route-owned seam.  The
    case-data route holds its input ``catalog`` (the GEOG source) and
    rebuilds per footprint through ``build_static_for_domain``; the
    prepared routes hold no ingest inputs and pass ``statics_builder``
    instead -- a callable ``(grid, new_dc) -> fields`` (the sealed
    statics-corridor crop, :func:`woof.static.corridor
    .corridor_footprint_statics_builder`) carrying ``static_provenance``
    and ``source_label`` attributes for the receipt.  Everything after
    the statics -- the SINT fill, the t=0 terrain adjustment, the
    re-derivation hook -- is one implementation, never forked per route.
    """
    if not hasattr(child_config, "run") or not hasattr(child_config,
                                                       "grid_id"):
        raise TypeError(
            "child_config must be the child's DomainConfig (an object "
            "with grid_id, parent_grid_ratio and a run config)")
    if (catalog is None) == (statics_builder is None):
        raise TypeError(
            "real_relocation_initializer takes exactly one statics "
            "source: the case-data route's input catalog OR a "
            "statics_builder (the prepared routes' sealed-corridor "
            "crop); a rebuild with neither has nothing to build "
            "footprint statics from, and with both one would silently "
            "shadow the other")
    cfg = child_config.run
    ratio = int(child_config.parent_grid_ratio)
    ref_i = int(reference_i_parent_start)
    ref_j = int(reference_j_parent_start)
    spec_width = int(cfg.spec_bdy_width)
    blend_width = int(getattr(child_config, "blend_width", 5))

    def prepare_footprint(new_dc, parent_node):
        from types import SimpleNamespace
        from woof.ingest.nest_init import _child_grid
        if int(new_dc.grid_id) != int(child_config.grid_id):
            raise RelocationRefusal(
                f"this initializer serves grid_id {child_config.grid_id}, "
                f"asked to rebuild grid_id {new_dc.grid_id}")
        if root_frame_shift is not None:
            # A mover whose ANCESTOR also moves ([relocation.containment])
            # cannot derive its earth displacement from its own placement
            # change: the placement is an offset inside a frame that
            # slides.  The route's resolver composes the live ancestor
            # chain (origin_in_frame_cells over the live tree patched
            # with new_dc, minus the same origin at reference time), in
            # this child's own cells -- every term an integer.  MEASURED
            # without it: the first d03 move after a d02 slide translated
            # from the stale frame and the drift gate below refused at
            # 2.5e-01 deg, which is exactly the slide (-1, -2) d01 cells
            # in child cells.
            shift_i, shift_j = (int(v) for v in
                                root_frame_shift(new_dc, parent_node))
        else:
            shift_i = (int(new_dc.i_parent_start) - ref_i) * ratio
            shift_j = (int(new_dc.j_parent_start) - ref_j) * ratio
        grid = reference_grid.translated(shift_i, shift_j)
        # The translated frame and the parent's nest arithmetic must
        # describe the same physical corner; drift means the reference
        # this factory holds is not this tree's child, which is a wiring
        # defect and refuses before any state is touched.
        resolved = _child_grid(new_dc, parent_node)
        lat_t, lon_t = grid.ij_to_latlon(1.0, 1.0)
        lat_r, lon_r = resolved.ij_to_latlon(1.0, 1.0)
        drift = max(abs(float(lat_t) - float(lat_r)),
                    abs(float(lon_t) - float(lon_r)))
        if drift > drift_tolerance_deg:
            raise RelocationRefusal(
                f"placement-translated grid drifts {drift:.3e} deg from the "
                f"parent-resolved placement at ({new_dc.i_parent_start}, "
                f"{new_dc.j_parent_start}); the initializer's reference "
                "grid does not belong to this tree")

        started = time.perf_counter()
        if statics_builder is None:
            static_fields, selection, landuse_attrs, highres_applied = \
                _build_footprint_statics(grid, catalog, new_dc)
            static_source = str(selection.root)
        else:
            static_fields = statics_builder(grid, new_dc)
            static_source = str(getattr(
                statics_builder, "source_label", "statics_builder"))
            highres_applied = bool(getattr(
                statics_builder, "highres_applied", False))
        statics_seconds = time.perf_counter() - started

        return SimpleNamespace(domain=new_dc, grid=grid, static_fields=static_fields,
            static_source=static_source, highres_applied=highres_applied,
            statics_seconds=statics_seconds, shift_i=shift_i, shift_j=shift_j, drift=drift)

    def initialize(new_dc, parent_node, *, scratch_arena=None,
                   dycore_state_workspace=None, window=None, footprint=None):
        from woof.ingest.nest_init import (_adjust_and_rederive, _as_like,
                                            _blend_terrain_triple,
                                            _capture_parent_blend_fields,
                                            _child_grid,
                                            _shared_vertical_coord,
                                            parent_only_init,
                                            seed_rk_time_t_copies)
        from woof.ingest.real import _make_real_base

        footprint = prepare_footprint(new_dc, parent_node) if footprint is None else footprint
        if footprint.domain != new_dc:
            raise RelocationRefusal("prepared footprint belongs to another domain placement")
        grid, static_fields = footprint.grid, footprint.static_fields
        static_source, highres_applied = footprint.static_source, footprint.highres_applied
        statics_seconds = footprint.statics_seconds
        shift_i, shift_j, drift = footprint.shift_i, footprint.shift_j, footprint.drift

        if window is not None:
            from woof.core.nest_interp import window_registration
            from woof.ingest.nest_init import _mass_registration
            window_registration(_mass_registration(new_dc, parent_node), window)
            sy, sx = window
            cropped_static = {}
            for name, value in static_fields.items():
                array = np.asarray(value)
                if array.ndim < 2:
                    cropped_static[name] = value
                    continue
                nyf, nxf = array.shape[-2:]
                if nyf not in (cfg.ny, cfg.ny+1) or nxf not in (cfg.nx, cfg.nx+1):
                    raise RelocationRefusal(
                        f"static {name} has unsupported reconstruction extent {array.shape}")
                cropped_static[name] = array[..., sy.start:sy.stop+(nyf-cfg.ny),
                                              sx.start:sx.stop+(nxf-cfg.nx)]
            static_fields = cropped_static

        extra = {}
        if scratch_arena is not None:
            extra["scratch_arena"] = scratch_arena
        if dycore_state_workspace is not None:
            extra["dycore_state_workspace"] = dycore_state_workspace
        sint_started = time.perf_counter()
        initialized = parent_only_init(new_dc, parent_node, grid=grid,
                                       window=window,
                                       clamp_undershoot=window is None,
                                       **extra)
        sint_seconds = time.perf_counter() - sint_started

        # The t=0 terrain-adjustment sequence, on the parent-interpolated
        # state.  parent_only_init already ran update_diagnostics, so
        # state.p is the pre-change pressure adjust_tempqv requires --
        # the same ordering finalize_prepared_child establishes at t=0.
        adjust_started = time.perf_counter()
        state = initialized.state
        coord = _shared_vertical_coord(vertical, cfg.nz)
        save_mub = state.mub2d.copy()
        fine = _make_real_base(
            coord, np.asarray(static_fields["HGT_M"], dtype=np.float64),
            float(vertical.p_top), float(cfg.base_temp),
            int(cfg.hypsometric_opt))
        state.ht[...] = _as_like(fine.terrain_z, state.ht)
        state.mub2d[...] = _as_like(fine.mub, state.mub2d)
        # Setter, not assignment: same reason as nest_spawn_init's
        # own-grid arm -- the relocated child's EOS must read the float64
        # base thickness of the geopotential it just moved onto.
        state.set_base_geopotential(fine.phb)
        ht_int, mub_int, phb_int = _capture_parent_blend_fields(
            new_dc, parent_node, window=window,
            device_windows=not isinstance(state.thp, np.ndarray))
        ht_int = _as_like(ht_int, state.ht)
        mub_int = _as_like(mub_int, state.mub2d)
        phb_int = _as_like(phb_int, state.phb)
        blend_args = dict(spec_bdy_width=spec_width, blend_width=blend_width)
        if window is not None:
            blend_args.update(domain_shape=(cfg.ny, cfg.nx),
                              origin=(window[0].start, window[1].start))
        # In float64 on the analytic fine base (_blend_terrain_triple), so
        # the relocated child keeps the EOS correction its FP32 store drops
        # exactly as the t=0 child does; outside both blend frames the
        # triple is that fine base, the placement invariant the
        # donor-alignment instrument reads.
        blended = _blend_terrain_triple(
            state, (ht_int, mub_int, phb_int), fine, **blend_args)
        # A MOVE IS NOT AN INITIALIZATION.  WRF blends the same terrain
        # triple here that `med_nest_initial` does, and then stops:
        # `adjust_tempqv` is absent from `share/mediation_nest_move.F`
        # entirely, and `press_adj` is set .FALSE. for both parent (:242)
        # and nest (:261) before `start_domain`.  Only the t = 0 path
        # (`mediation_integrate.F` :763 and :809) runs them.
        #
        # The reason is physical: at t = 0 the child's columns came from
        # its parent and have never been consistent with its own terrain,
        # so theta and qv must be corrected for the base-column-mass
        # change.  A moving child's columns are its own and already
        # consistent; correcting them against a blend that only perturbs
        # `mub` in the frame injects an anomaly into fields that were
        # right.
        _adjust_and_rederive(state, cfg, coord, save_mub,
                             static_fields["HGT_M"], blended=blended,
                             column_mass_correction=False)
        undershoot = clamp_interpolation_undershoot(state)
        seed_rk_time_t_copies(state, preserve_thermal=True)
        del ht_int, mub_int, phb_int, save_mub, blended
        adjust_seconds = time.perf_counter() - adjust_started

        receipt = {
            "initializer": "real_relocation_initializer",
            "static_source": static_source,
            "static_provenance": initialize.static_provenance,
            "highres_applied": bool(highres_applied),
            "placement_translation_child_cells": [shift_i, shift_j],
            "translation_drift_deg": float(drift),
            "interpolation_undershoot_clamped": undershoot,
            "timings_seconds": {
                "static_rebuild": statics_seconds,
                "atmosphere_sint": sint_seconds,
                "terrain_adjustment": adjust_seconds,
            },
            "terrain_adjustment": (
                "blend_terrain(ht/mub/phb) + adjust_tempqv + start_domain "
                "base/EOS re-derivation + press_adj (t=0 lineage, "
                f"spec_bdy_width={spec_width}, blend_width={blend_width})"),
        }
        from dataclasses import replace as _replace

        return _replace(initialized, grid=initialized.grid,
                        static_fields=static_fields,
                        preprocess_receipt=receipt)

    initialize.prepare_footprint = prepare_footprint

    def post_transplant(*, source_state, target_state, plan):
        return rederive_after_transplant(
            source_state=source_state, target_state=target_state,
            plan=plan, cfg=cfg)

    if statics_builder is None:
        initialize.static_provenance = REAL_DATA_FOOTPRINT_REBUILT_STATICS
        initialize.strip_fill_source = REAL_DATA_STRIP_FILL_SOURCE
    else:
        initialize.static_provenance = str(getattr(
            statics_builder, "static_provenance",
            REAL_DATA_FOOTPRINT_REBUILT_STATICS))
        from woof.static.corridor import CORRIDOR_STRIP_FILL_SOURCE
        initialize.strip_fill_source = CORRIDOR_STRIP_FILL_SOURCE
    # The donor-alignment instrument compares SINT/analytic base fields
    # that are placement-invariant only OUTSIDE both placements' blend
    # frames; inside them the t=0 machinery blends toward the parent at
    # each footprint's own edge and the invariant is genuinely undefined.
    initialize.donor_alignment_frame_width = spec_width + blend_width
    initialize.post_transplant = post_transplant
    return initialize


__all__ = [
    "DonorFillPlan", "LAND_SURFACE_CONTINUATION_FIELDS",
    "OVERLAP_STATIC_EQUALITY_FIELDS",
    "REAL_DATA_FOOTPRINT_REBUILT_STATICS", "REAL_DATA_STRIP_FILL_SOURCE",
    "clamp_interpolation_undershoot", "donor_fill_plan",
    "overlap_mask_for_plan",
    "overlap_statics_mismatches", "real_relocation_initializer",
    "rederive_after_transplant",
]
