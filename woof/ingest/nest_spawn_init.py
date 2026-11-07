"""Materialize a spawn-triggered nest: own-grid statics, parent atmosphere.

The spawn half of the storm-following program's birth event.  A dormant
nest (:mod:`woof.core.nest_spawn`) fires mid-run; THIS module builds the
child that fires into existence, out of exactly two ingredients:

1. OWN-GRID STATICS for the trigger-chosen footprint, through the same
   footprint-parametric static path every declared child uses
   (:func:`woof.static.build.build_static_for_domain` on the child grid
   nested at the fired placement, with the ``[static.highres]`` overlay
   when the case enables it -- the tile cache is what makes a
   trigger-time build fast); and
2. THE CURRENT PARENT STATE for the atmosphere, through the existing
   nest cold-start machinery: :func:`woof.ingest.nest_init
   .parent_only_init`'s full-parent SINT fill, followed by the SAME
   terrain-adjustment sequence the real-data child path runs at t = 0 --
   ``blend_terrain`` on the ht/mub/phb operand triple, ``adjust_tempqv``
   for the base-column-mass change, the ``start_domain`` base/EOS
   re-derivation, and the real-nest ``press_adj`` MU correction
   (``_adjust_and_rederive``) -- invoked mid-run.

WHY THE PARENT AND NOT THE ANALYSIS.  The delayed-start path
(:func:`woof.core.model.execute_experiment`'s ``on_domain_start``)
initializes a late child from the EXTERNAL analysis at its activation
time.  A spawn trigger fires precisely because the parent has developed
a storm the analysis does not contain -- initializing from the analysis
would materialize a nest without the storm it exists to follow.  So the
atmosphere is the live parent's, and the only external input is the
static ground.

THE TERRAIN ADOPTION, MAPPED ONTO THE REAL PATH.  On the real-data path
the child's fine-frame base state comes from the analytic hydrostatic
base on its own terrain (``module_initialize_real``'s construction;
:func:`woof.ingest.real._make_real_base`) and its atmosphere from the
analysis; the parent contributes only the SINT-captured ht/mub/phb
blend operands.  Here the roles map exactly: the parent-frame child
(the ``parent_only_init`` product) IS the pre-adjustment state whose
``save_mub`` frame the atmosphere was built in, the parent-frame
ht/mub/phb ARE the SINT captures (they were produced by the same
operator), and the fine-frame operands come from ``_make_real_base`` on
the own-grid ``HGT_M``.  The blend then takes the parent near the
boundary and the fine terrain in the interior, and ``adjust_tempqv`` /
``press_adj`` correct theta/qv/MU for the column-mass change --
byte-for-byte the sequence :func:`woof.ingest.nest_init
.finalize_prepared_child` runs, with the analysis replaced by the
parent.  The calibration point: with fine terrain identical to the
parent-frame terrain (a flat case) every operand of the adoption is the
identity, so the adopted child is BITWISE the plain ``parent_only_init``
child -- pinned by tests/test_nest_spawn_init.py.

WHAT THIS MODULE DOES NOT DO.  It decides nothing (the trigger did) and
attaches nothing to the live tree: node construction, clocks, coupler
and physics-driver attachment are the runner's, through the same
``on_child_built`` seam :func:`woof.core.nest_relocation.relocate_child`
publishes, and the schedule surgery is
:func:`woof.experiment.active_experiment` at a leg boundary.  Land
DRIVER state (Noah/NoahMP continuation) follows the same rule it does at
every leg boundary and relocation: re-initialised by ``on_child_built``,
never invented here.

A restart across a spawn resumes, and this module is on that path too:
:func:`woof.runtime.restore_nest_lifecycle` re-materializes every live
episode through the same entry point a leg boundary uses, at the
placement the slot fired on.  What the spawn still does is move the tree
restart fingerprint by construction -- the activated placement differs
from the declared one -- so a checkpoint written after a spawn is refused
by a FRESH build and resumes only into the run that wrote it.  Nothing
here relaxes that.
"""

from __future__ import annotations

import logging

import numpy as np

from woof.ingest.nest_init import (ChildInitResult, _adjust_and_rederive,
                                    _as_like, _blend_terrain_triple,
                                    _child_grid_from_parent,
                                    _static_catalog, parent_only_init,
                                    seed_rk_time_t_copies)

#: Versioned label carried by every spawn-materialization receipt.
SPAWN_INIT_CONTRACT = "gpuwm-nest-spawn-init.v1"

_LOG = logging.getLogger("gpuwm.nest_spawn_init")


class SpawnInitRefusal(ValueError):
    """A materialization this module will not perform quietly."""


def _host(value) -> np.ndarray:
    get = getattr(value, "get", None)
    if callable(get) and hasattr(value, "__cuda_array_interface__"):
        return np.asarray(get())
    return np.asarray(value)


def prepare_spawn_statics(child_dc, parent_node, catalog, *,
                          valid_date=None) -> dict[str, object]:
    """Own-grid statics for the fired footprint (the trigger-time build).

    ``child_dc`` already carries the FIRED placement.  Returns
    ``{"static_fields", "landuse_attrs", "receipt"}``; the receipt names
    the static source so the spawn receipt can carry provenance.  The
    ``[static.highres]`` overlay applies exactly as it does on the
    declared-child path (:func:`woof.ingest.nest_init
    ._prepare_child_input_on_grid`); ``valid_date`` selects its dated
    tiles and should be the SPAWN instant's date, since that is the date
    the nest will integrate.
    """
    from woof.static.build import (build_static_for_domain,
                                    geog_selection_from_catalog)

    grid = _child_grid_from_parent(
        child_dc, parent_node.cfg.grid_id, parent_node.grid)
    static_catalog = _static_catalog(catalog)
    static_fields = build_static_for_domain(
        grid, static_catalog, child_dc.grid_id,
        **({"cfg": child_dc.run}
           if getattr(child_dc.run, "sf_lake_physics", 0) else {}))
    landuse_attrs = geog_selection_from_catalog(
        static_catalog, child_dc.grid_id).landuse_global_attrs()
    receipt: dict[str, object] = {
        "static_source": "footprint-parametric WPS_GEOG (30s baseline)",
        "highres_overlay": False,
        "placement": [int(child_dc.i_parent_start),
                      int(child_dc.j_parent_start)],
    }
    highres = getattr(catalog, "static_highres", None)
    if highres is not None and getattr(highres, "enabled", False):
        from woof.static.highres_production import apply_highres_statics
        static_fields, overlay_receipt = apply_highres_statics(
            static_fields, grid, config=highres,
            domain_id=child_dc.grid_id,
            case_date=valid_date,
            landuse_attrs=landuse_attrs)
        # A block scoped by grid spacing leaves a coarser domain alone.
        if overlay_receipt is not None:
            receipt["highres_overlay"] = True
            receipt["highres"] = overlay_receipt
    return {"static_fields": static_fields,
            "landuse_attrs": dict(landuse_attrs), "receipt": receipt}


#: Land-use categories whose masked interpolation keys on ISICE rather
#: than ISWATER.  The Registry spells the flag per field: XICE/XICEM/SNOWSI
#: carry ``interp_mask_field:lu_index,isice``, everything else in the
#: land-surface family carries ``...,iswater`` (Registry.EM_COMMON:842-846
#: against :790/:839-841/:868-872/:1417).
SEA_ICE_MASKED_FIELDS = frozenset({"xice", "xicem", "snowsi"})

#: The receipt-stated provenance of a newborn nest's land-surface state.
SPAWN_LAND_SOURCE = (
    "parent-interpolated land-surface state: WRF's masked nest "
    "interpolator (interp_fcn.F:4075-4275, the Registry's "
    "interp_mask_field:lu_index,iswater / ,isice) applied to the LIVE "
    "parent driver's soil, snow, skin and seeded surface diagnostics, "
    "against the newborn's OWN-GRID land-use categories")


def spawn_land_state_from_parent(child_dc, parent_node, *, static_fields,
                                 parent_static_fields, landuse_attrs,
                                 names=None) -> dict[str, object]:
    """Interpolate the live parent's land-surface state onto the newborn.

    THE ESTABLISHED PRACTICE, and the one WRF has no alternative to here.
    A nest that starts later than its parent has two supported routes
    (Users' Guide chapter 5, "Nesting"):

    * ``fine_input_stream = 2`` -- 3-D meteorology interpolated from the
      parent, but static AND MASKED SURFACE fields (soil temperature and
      moisture among them) read from the nest's own ``wrfinput``.  That
      file is a ``real.exe`` product at the nest's own footprint and start
      time, so it presupposes knowing both in advance; and
    * ``input_from_file = .false.`` -- "the model interpolates all
      variables required in the nest from the coarse domain fields",
      which is what ``med_nest_initial``'s unconditional
      ``med_interp_domain(parent, nest)`` does before any input file is
      consulted (share/mediation_integrate.F:670, :758-766).

    A trigger-spawned nest picks its footprint mid-run, from a storm the
    analysis does not contain, so the first route cannot exist for it:
    there is no ``real.exe`` product at a footprint nobody knew about.
    The second is therefore the route, and it is not a degradation of
    WRF practice but the same operator WRF runs at every nest birth and,
    via the identical ``med_interp_domain`` call after each
    ``shift_domain_em``, at every moving-nest leading edge
    (share/mediation_nest_move.F:186).  HAFS's storm-following nest
    states the rule in words: interpolation at the leading edge "taking
    into account the land/sea/ice mask to only consider values from the
    same surface type".

    WOOF keeps the half of ``fine_input_stream = 2`` it CAN have -- the
    own-grid statics (:func:`prepare_spawn_statics`) -- which is strictly
    more than ``input_from_file = .false.`` gives, and is exactly why the
    masked interpolator is essential rather than decorative: the child's
    land-use categories are its OWN, resolved at the child's dx, so they
    disagree with the parent's wherever finer terrain resolves a coast,
    lake or island the parent smoothed away.  Those disagreements are
    counted, per field family, never silent.

    ``names`` defaults to :data:`~woof.ingest.relocation_init
    .LAND_SURFACE_CONTINUATION_FIELDS`, the same inventory a relocation
    carries; a field the parent's configuration never allocated is skipped
    by name and named in the receipt, exactly as the transplant does.

    Returns ``{"fields": {name: host ndarray}, "receipt": {...}}``.
    """
    from woof.core.nest_interp import interp_mask_field
    from woof.ingest.relocation_init import surface_continuation_fields

    if names is None:
        names = surface_continuation_fields(child_dc.run, interpolate=True)
    driver = getattr(parent_node.state, "physics", None)
    if driver is None or not getattr(driver, "fields", None):
        raise SpawnInitRefusal(
            "the parent carries no physics driver, so it has no "
            "land-surface state to interpolate onto the newborn; a "
            "spawned nest cannot invent one")
    for source, label in ((static_fields, "the newborn's own-grid statics"),
                          (parent_static_fields, "the parent's statics")):
        if source is None or "LU_INDEX" not in source:
            raise SpawnInitRefusal(
                f"{label} carry no LU_INDEX; WRF's masked surface "
                "interpolator keys on the land-use category "
                "(interp_mask_field:lu_index,...) and has no defined "
                "behaviour without it")
    child_lu = _host(static_fields["LU_INDEX"])
    parent_lu = _host(parent_static_fields["LU_INDEX"])
    iswater = int(landuse_attrs["ISWATER"])
    isice = int(landuse_attrs["ISICE"])
    nri = int(child_dc.parent_grid_ratio)
    geometry = {
        "nri": nri, "nrj": nri,
        "i_parent_start": int(child_dc.i_parent_start),
        "j_parent_start": int(child_dc.j_parent_start),
        "child_landuse": child_lu, "parent_landuse": parent_lu,
    }

    fields: dict[str, np.ndarray] = {}
    absent: list[str] = []
    shape_skipped: dict[str, object] = {}
    by_flag: dict[str, dict[str, object]] = {}
    for name in names:
        value = driver.fields.get(name)
        if value is None:
            absent.append(name)
            continue
        coarse = _host(value)
        if coarse.shape[-2:] != parent_lu.shape:
            # A field that does not live on the parent's mass grid is not
            # something this operator is defined for.  Named, not guessed.
            shape_skipped[name] = list(coarse.shape)
            continue
        sea_ice = name in SEA_ICE_MASKED_FIELDS
        flag = isice if sea_ice else iswater
        label = "isice" if sea_ice else "iswater"
        interpolated, counts = interp_mask_field(
            coarse, flag_category=flag, **geometry)
        fields[name] = interpolated
        entry = by_flag.setdefault(label, {"flag_category": int(flag),
                                           "fields": [], "counts": counts})
        entry["fields"].append(name)
        if entry["counts"] != counts:      # pragma: no cover - invariant
            raise SpawnInitRefusal(
                "the masked interpolator returned different class "
                f"accounting for {name} than for its family; the mask "
                "decision is per column and cannot depend on the field")

    conflicts = sum(int(entry["counts"]["opposite_class_bilinear"])
                    for entry in by_flag.values())
    receipt: dict[str, object] = {
        "source": SPAWN_LAND_SOURCE,
        "operator": "interp_mask_field (interp_fcn.F:4075-4275)",
        "parent_grid_id": int(parent_node.cfg.grid_id),
        "placement": [int(child_dc.i_parent_start),
                      int(child_dc.j_parent_start)],
        "parent_grid_ratio": nri,
        "landuse": {"MMINLU": str(landuse_attrs.get("MMINLU", "")),
                    "ISWATER": iswater, "ISICE": isice},
        "by_mask_flag": by_flag,
        "fields_interpolated": sorted(fields),
        "fields_absent": sorted(absent),
        "fields_shape_skipped": shape_skipped,
        "land_class_conflict_cells": int(conflicts),
        "accumulators_reinitialized": True,
    }
    _LOG.info("nest-spawn-land-init %s", receipt)
    return {"fields": fields, "receipt": receipt}


def _adopt_own_terrain(initialized: ChildInitResult, child_dc,
                       static_fields, *, blend_width: int,
                       vertical=None) -> dict[str, object]:
    """Swap the parent-SINT ground for the own-grid statics, WRF-order.

    Runs the real-path adjustment sequence on the parent_only_init
    product (module docstring: the role mapping).  Mutates
    ``initialized.state`` in place and returns the adoption receipt.
    """
    from woof.ingest.real import _make_real_base

    state = initialized.state
    cfg = child_dc.run
    # The experiment's ONE shared coordinate when the route hands it over,
    # exactly the coordinate a relocation re-derives this nest's base state
    # on (woof.ingest.relocation_init: _shared_vertical_coord).  The SINT
    # product's own coord is the parent's copy; on a prepared tree it carries
    # the preparation's arithmetic, and a spawned nest born on it could not
    # follow its storm: the first move's donor alignment refused pb/alb/thb/
    # phb on the overlap (a rented box, 2026-10-21 18:16Z).
    if vertical is not None and not getattr(vertical, "eta_levels", None):
        vertical = None          # no shared ladder declared: the SINT coord stands
    if vertical is not None:
        from woof.ingest.nest_init import _shared_vertical_coord
        coord = _shared_vertical_coord(vertical, int(child_dc.run.nz))
    else:
        coord = initialized.coord
    if not cfg.terrain_opt:
        raise SpawnInitRefusal(
            "own-grid statics adoption requires terrain_opt != 0: a "
            "terrain-free child has no ht/mub/phb frame to adopt into")
    if not cfg.moist:
        raise SpawnInitRefusal(
            "own-grid statics adoption requires moist=True, exactly as "
            "the real-data child path does: adjust_tempqv corrects "
            "theta AND qv for the column-mass change, and a dry state "
            "has no qv to correct")
    if state.p_top is None:
        raise SpawnInitRefusal(
            "own-grid statics adoption requires a hybrid-coordinate "
            "parent (state.p_top is set); the idealized ztop scaffold "
            "has no analytic real base to build fine-frame operands from")
    if "HGT_M" not in static_fields:
        raise SpawnInitRefusal(
            "spawn statics carry no HGT_M; the footprint-parametric "
            "static build did not produce terrain, so there is nothing "
            "to adopt")
    fine_terrain = np.asarray(_host(static_fields["HGT_M"]),
                              dtype=np.float64)
    if tuple(fine_terrain.shape) != tuple(state.ht.shape):
        raise SpawnInitRefusal(
            f"own-grid HGT_M shape {fine_terrain.shape} differs from the "
            f"spawned child's {tuple(state.ht.shape)}; the statics were "
            "built for a different footprint")

    # The parent-frame operand triple IS the SINT capture (it was
    # produced by the same operator inside parent_only_init), and the
    # parent-frame mub is the save_mub frame the atmosphere was built in.
    ht_capture = state.ht.copy()
    mub_capture = state.mub2d.copy()
    phb_capture = state.phb.copy()
    save_mub = state.mub2d.copy()

    # Fine-frame operands: the analytic hydrostatic real base on the
    # child's own terrain -- exactly what real.exe supplies on the
    # declared-child path.
    fine_base = _make_real_base(
        coord, fine_terrain,
        float(state.p_top if vertical is None else vertical.p_top),
        float(cfg.base_temp), int(cfg.hypsometric_opt))
    state.ht[...] = _as_like(fine_terrain, state.ht)
    state.mub2d[...] = _as_like(fine_base.mub, state.mub2d)
    # Through the setter, so the child's EOS reads the float64 base layer
    # thickness of its OWN geopotential rather than the parent's.  The
    # blend below then rewrites phb in place over the boundary rim; the
    # correction and the host height cache keep describing the pre-blend
    # profile there, which the residual spelling bounds to <= one ulp of
    # phb in the diagnostic (woof/core/state.py::set_base_geopotential).
    state.set_base_geopotential(fine_base.phb)

    # WRF blends all three fields (mediation_integrate.F:733-741); never
    # blend ht alone and derive, because base construction is nonlinear.
    # In float64 on the fine base, as the t=0 child does
    # (_blend_terrain_triple), so the re-derived base keeps its EOS
    # correction over the rim too.
    spec_width = int(cfg.spec_bdy_width)
    blended = _blend_terrain_triple(
        state, (ht_capture, mub_capture, phb_capture), fine_base,
        spec_bdy_width=spec_width, blend_width=int(blend_width))

    # theta/qv adjustment for the column-mass change, start_domain
    # base/EOS re-derivation, then the real-nest press_adj MU correction
    # -- the exact finalize_prepared_child tail.
    _adjust_and_rederive(state, cfg, coord, save_mub, fine_terrain,
                         blended=blended)
    mub_shift = _host(state.mub2d).astype(np.float64) - _host(
        save_mub).astype(np.float64)
    ht_shift = _host(state.ht).astype(np.float64) - _host(
        ht_capture).astype(np.float64)
    return {
        "static_source": "own-grid",
        "spec_bdy_width": spec_width,
        "blend_width": int(blend_width),
        "terrain_max_abs_shift_m": float(np.max(np.abs(ht_shift))),
        "mub_max_abs_shift_pa": float(np.max(np.abs(mub_shift))),
        "adjustment": ("blend_terrain(ht,mub,phb) + adjust_tempqv + "
                       "start_domain base/EOS re-derivation + press_adj"),
    }


def spawned_child_device_bytes(child_dc, parent_dc, *, scratch_arena=None,
                               dycore_state_workspace=None,
                               urban_columns=None) -> dict[str, int]:
    """What a spawned child newly allocates on the card, by part.

    Priced from the same per-domain inventory the forecast admission sums
    (:func:`woof.core.preflight.estimate_domain`): the child's state, the
    physics driver its preparer attaches, and the nest interpolation
    tables its coupler fills.  A slot the tree's shared scratch arena
    already backs, and a state symbol the shared dycore workspace already
    backs, cost nothing new: both were sized over every declared domain,
    dormant ones included, at startup.  ``urban_columns`` is the child's
    urban column count read off its spawn statics
    (:func:`spawn_urban_columns`), which prices BEP+BEM's column workspace
    at the plan the child builds (A176).
    """
    from woof.config import radiation_scheme_ids
    from woof.core.preflight import (estimate_domain,
                                      shared_dycore_state_symbols)
    from woof.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant

    run = child_dc.run
    cam_ozone = bool(4 in radiation_scheme_ids(run)
                     and rrtmg_variant(run) == RRTMG_VARIANT_LEGACY
                     and int(run.o3input) == 2)
    estimate = estimate_domain(
        child_dc, spec_bdy_width=int(run.spec_bdy_width),
        cam_ozone=cam_ozone, parent=parent_dc, urban_columns=urban_columns)
    shared_state = (shared_dycore_state_symbols(run)
                    if dycore_state_workspace is not None else frozenset())

    def arena_backed(slot: str) -> bool:
        return scratch_arena is not None and bool(scratch_arena.has_slot(slot))

    return {
        "state": sum(item.nbytes for item in estimate.items
                     if item.category == "state"
                     and item.name not in shared_state),
        "physics": estimate.category_bytes("physics"),
        "interpolation": sum(item.nbytes for item in estimate.items
                             if item.category == "nest"
                             and not arena_backed(item.name)),
    }


def card_free_bytes() -> int:
    """Bytes a new allocation can take on this process's card right now.

    Free on the device (``cudaMemGetInfo`` capped by the device-wide NVML
    figure, :func:`woof.core.preflight.device_free_and_total_bytes`) plus
    the blocks this process's CuPy pool holds unused, which the pool hands
    out again before it asks the driver.
    """
    import cupy as cp

    from woof.core.preflight import device_free_and_total_bytes

    free, _total = device_free_and_total_bytes()
    return int(free) + int(cp.get_default_memory_pool().free_bytes())


def size_text(nbytes: int) -> str:
    """A byte count in GiB from 1 GiB up, in MiB below it."""
    nbytes = int(nbytes)
    if nbytes >= 1024 ** 3:
        return f"{nbytes / 1024 ** 3:.2f} GiB"
    return f"{nbytes / 1024 ** 2:.1f} MiB"


def spawn_urban_columns(child_dc, static_fields) -> int | None:
    """The spawned child's urban columns, read off the statics it is about
    to adopt (:func:`prepare_spawn_statics`'s product: its LU_INDEX on the
    land-use dataset its ``landuse_attrs`` name).  ``None`` (priced at
    every column urban) when the child runs no BEP+BEM, follows a storm
    (its ground changes after the birth), or the statics name no dataset,
    as on the parent-SINT branch."""
    if getattr(child_dc, "follow", None) is not None:
        return None
    if not (isinstance(static_fields, dict) and "static_fields" in static_fields):
        return None
    attrs = static_fields.get("landuse_attrs") or {}
    fields = static_fields.get("static_fields")
    if fields is None or "MMINLU" not in attrs:
        return None
    from woof.core.urban_state import prepared_urban_columns

    return prepared_urban_columns(
        child_dc.run, fields.get("LU_INDEX"),
        landuse_dataset=str(attrs["MMINLU"]))


def admit_spawned_child(child_dc, parent_node, *, free_bytes: int,
                        scratch_arena=None, dycore_state_workspace=None,
                        urban_columns=None) -> dict[str, object]:
    """Refuse a spawn the card cannot hold, before any of it is allocated.

    THE BREAKAGE THIS PREVENTS.  A trigger fires mid-run, beside a tree
    that is already on the card.  With no fit check the child's state went
    straight to the allocator, so a card that could not hold it ended the
    run in a raw CUDA out-of-memory error part-way through building a
    child, hours in.  The startup fit check reserves a declared spawn's
    bytes, but nothing held the card to that reservation afterwards:
    another program on the card, or growth since the start, spends it.
    """
    from woof.core.preflight import (forecast_pool_estimate_bytes,
                                      forecast_pool_headroom,
                                      urban_held_bytes)

    headroom = forecast_pool_headroom((child_dc.run, parent_node.cfg))
    parts = spawned_child_device_bytes(
        child_dc, parent_node.cfg, scratch_arena=scratch_arena,
        dycore_state_workspace=dycore_state_workspace,
        urban_columns=urban_columns)
    itemized = int(sum(parts.values()))
    held = urban_held_bytes(child_dc.run, urban_columns=urban_columns)
    # The child's arrays go through the same pool as the tree's, so they
    # carry the forecast's one measured pool margin (A163): the startup
    # envelope priced this nest at that margin, and the check at the
    # spawn reads the same number instead of the bare itemized sum.  The
    # arrays held at their allocated size (urban) are priced there, as
    # the startup envelope prices them.
    need = forecast_pool_estimate_bytes(
        itemized, held_exact_bytes=held, headroom=headroom)
    receipt: dict[str, object] = {
        "need_bytes": need, "itemized_bytes": itemized,
        "held_exact_bytes": held, "urban_columns": urban_columns,
        "pool_headroom": headroom, "parts_bytes": dict(parts),
        "card_free_bytes": int(free_bytes), "fits": need <= int(free_bytes),
    }
    if not receipt["fits"]:
        raise SpawnInitRefusal(
            f"spawning d{int(child_dc.grid_id):02d} needs {size_text(need)} "
            f"on the card (state {size_text(parts['state'])}, physics "
            f"{size_text(parts['physics'])}, nest interpolation tables "
            f"{size_text(parts['interpolation'])}, x "
            f"{headroom:.2f} measured pool margin) and the card can hand "
            f"out {size_text(free_bytes)} now; building it beside the "
            "live tree would end the run in a CUDA out-of-memory error "
            "part-way through the child.  The startup fit check reserved "
            "this nest, so something has taken that memory since: another "
            "program on the card, or allocations the run made after its "
            "start.  Free the card, or declare a smaller spawn nest.")
    return receipt


def spawn_child_from_parent(child_dc, parent_node, *,
                            static_fields=None, blend_width: int = 5,
                            scratch_arena=None, dycore_state_workspace=None,
                            array_module=None, on_child_built=None,
                            state_digest=None,
                            trigger_receipt=None,
                            device_free_bytes=None,
                            vertical=None) -> dict[str, object]:
    """Materialize one fired nest from the LIVE parent, and account for it.

    ``child_dc`` carries the FIRED placement
    (:func:`woof.experiment.active_experiment` produced it).
    ``static_fields`` is the :func:`prepare_spawn_statics` product, or
    ``None`` for the statics-free branch (idealized cases, and any case
    whose ground the parent's SINT already describes) -- the child then
    keeps the parent-SINT ground, exactly as every idealized cold start
    does.  ``trigger_receipt`` is the fired :class:`woof.core.nest_spawn
    .SpawnEvent`'s receipt, carried verbatim so the birth certificate
    names its own evidence.

    The parent is read and never written; that is measured (state digest
    before and after) rather than asserted, the relocate_child idiom.
    ``on_child_built`` is the caller's per-domain seam (map policy,
    physics/land driver) and fires before the receipts are cut, so a
    driver initialises from the parent-interpolated fields.

    Before anything is allocated, a child built on the card is admitted
    against what the card can hand out (:func:`admit_spawned_child`).
    ``device_free_bytes`` is the probe for that figure, a callable
    returning bytes; it defaults to :func:`card_free_bytes` for a card
    build and is not asked for a NumPy build, which allocates nothing on
    a card.

    Returns the receipt dict; the live product rides under
    ``"child_result"`` (a :class:`ChildInitResult`), the ``segment_state``
    precedent for carrying a live object beside its JSON.
    """
    if state_digest is None:
        from woof.ensemble.state_sha import live_state_sha256
        state_digest = live_state_sha256

    device_admission = None
    if device_free_bytes is None and array_module is None:
        device_free_bytes = card_free_bytes
    if device_free_bytes is not None:
        device_admission = admit_spawned_child(
            child_dc, parent_node, free_bytes=int(device_free_bytes()),
            scratch_arena=scratch_arena,
            dycore_state_workspace=dycore_state_workspace,
            urban_columns=spawn_urban_columns(child_dc, static_fields))

    parent_sha_before = state_digest(parent_node.state)

    extra = {}
    if scratch_arena is not None:
        extra["scratch_arena"] = scratch_arena
    if dycore_state_workspace is not None:
        extra["dycore_state_workspace"] = dycore_state_workspace
    if array_module is not None:
        extra["array_module"] = array_module
    # register_nest refuses a placement whose donors fall outside the
    # parent's +-2 SINT stencil, so an off-grid fired placement fails
    # HERE, before anything exists to clean up.
    initialized = parent_only_init(child_dc, parent_node, **extra)

    fields = (static_fields.get("static_fields")
              if isinstance(static_fields, dict)
              and "static_fields" in static_fields else static_fields)
    statics_receipt = (static_fields.get("receipt")
                       if isinstance(static_fields, dict) else None)
    if fields is not None:
        adoption = _adopt_own_terrain(
            initialized, child_dc, fields, blend_width=blend_width,
            vertical=vertical)
        # The RK time-t copies were seeded from the parent-frame state
        # inside parent_only_init; the adoption changed thp/mup, so the
        # seeds are taken again (WRF start_domain's post-condition, the
        # same reason relocate_child reseeds after its transplant).
        initialized = ChildInitResult(
            state=initialized.state, grid=initialized.grid,
            coord=initialized.coord, real=None,
            static_fields=fields, horizontal=None, soil=None,
            domain=child_dc,
            # Carried, not dropped.  The terrain adoption rewrites thp
            # and mup and touches no number concentration, so the
            # positive-definite account taken at interpolation time is
            # still the true one -- and a rebuild that silently lost it
            # made the birth certificate report an empty fix-up on a
            # birth that had needed one.
            positive_definite_clamp=initialized.positive_definite_clamp)
    else:
        adoption = {"static_source": "parent-sint",
                    "note": "no static catalog at spawn; the child keeps "
                            "the parent-SINT ground, as every idealized "
                            "cold start does"}

    if on_child_built is not None:
        on_child_built(initialized, child_dc, parent_node)
    # The preparer owns the land-surface half of the birth (it is the
    # seam that holds the case data and the parent's statics), and
    # publishes its accounting the way the relocation preparer does --
    # duck-typed, so a route without one simply carries None.
    land_receipt = getattr(on_child_built, "last_receipt", None)
    seeded = seed_rk_time_t_copies(initialized.state)

    parent_sha_after = state_digest(parent_node.state)
    if parent_sha_after != parent_sha_before:
        raise SpawnInitRefusal(
            "the parent state changed across a spawn; materialization "
            "reads the parent and must never write it "
            f"({parent_sha_before} -> {parent_sha_after})")
    child_sha = state_digest(initialized.state)

    receipt: dict[str, object] = {
        "contract": SPAWN_INIT_CONTRACT,
        "grid_id": int(child_dc.grid_id),
        "parent_grid_id": int(parent_node.cfg.grid_id),
        "placement": [int(child_dc.i_parent_start),
                      int(child_dc.j_parent_start)],
        "trigger": trigger_receipt,
        "device_admission": device_admission,
        "statics": statics_receipt,
        "terrain": adoption,
        "land_surface": land_receipt,
        "atmosphere_source": {
            "kind": "parent-sint",
            "note": ("full-parent SINT of the LIVE parent state through "
                     "parent_only_init: the spawned nest is born inside "
                     "the storm its trigger saw, not inside a stale "
                     "analysis"),
            "parent_state_sha256": parent_sha_before,
            # What the interpolation had to be fixed up for, by field and
            # by cell count, with the tolerance each field was judged
            # against.  An empty table means it landed clean; a birth
            # that needed no fix-up says so with a number rather than by
            # omitting the subject.
            "positive_definite_clamp": dict(
                getattr(initialized, "positive_definite_clamp", None) or {}),
        },
        "rk_seeds_refreshed": list(seeded),
        "parent_state_sha256_before": parent_sha_before,
        "parent_state_sha256_after": parent_sha_after,
        "parent_bitwise_unchanged": parent_sha_before == parent_sha_after,
        "child_state_sha256": child_sha,
        "child_result": initialized,
    }
    _LOG.info("nest-spawn-init %s", {
        key: value for key, value in receipt.items()
        if key != "child_result"})
    return receipt


__all__ = [
    "SEA_ICE_MASKED_FIELDS", "SPAWN_INIT_CONTRACT", "SPAWN_LAND_SOURCE",
    "SpawnInitRefusal", "admit_spawned_child", "card_free_bytes",
    "prepare_spawn_statics", "spawn_child_from_parent",
    "spawn_land_state_from_parent", "spawned_child_device_bytes",
]
