"""A downscaled child that starts warm from an earlier child of itself.

``woof downscale --warm-from CHECKPOINT --warm-shift DI,DJ`` builds the
child at its placement from the parent exactly as a cold start does, then
carries what an earlier child at a nearby placement had integrated over the
ground both share.  It is the offline counterpart of a moving nest's move
(:mod:`woof.core.nest_relocation`), and it uses the same pieces:

* the restart layer's serialised state (:func:`woof.core.nest_relocation.
  relocatable_attrs`) is stamped bitwise onto the overlap by
  :func:`woof.core.nest_relocation.transplant_overlap`, with the same
  index-space window (``dst[d] <- src[d + shift]``);
* the land-surface continuation fields (:data:`woof.ingest.relocation_init.
  LAND_SURFACE_CONTINUATION_FIELDS`) and the accumulated-precipitation slots
  (:data:`woof.core.physics_continuation.STRIP_ACCUMULATORS`) follow the
  same window;
* the EOS diagnostics are re-derived against the child's own base state,
  as :func:`woof.ingest.relocation_init.rederive_after_transplant` does
  after a move, and the RK time-t copies are re-seeded.

The freshly covered strip keeps the cold start from the parent.  Its
accumulated precipitation is the parent's since the first segment began
(the moving runner writes it, :func:`woof.moving_downscale.
strip_accumulations`), so a rain total has no band of too little rain along
the leading edge, the rule WRF applies to a moving nest's exposed cells; the
earlier child's specified ring, where it lands inside the new one, takes the
parent's too (its microphysics never ran there).

``DI,DJ`` is the move in CHILD cells, new placement minus old: a child that
moved one parent cell east at ratio 3 is ``3,0``.  The earlier checkpoint
must have this child's shape; anything else is refused by name.  A move
larger than the child shares no ground with it, so nothing is carried: the
child keeps its cold start from the parent, and every cell takes the
parent's rain since the child began.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: The parent history variable each accumulated-precipitation slot is
#: seeded from on fresh ground (WRF Registry names).
STRIP_ACCUMULATOR_HISTORY = {
    "mp_rainnc": "RAINNC",
    "cu_rainc": "RAINC",
    "mp_snownc": "SNOWNC",
    "mp_graupelnc": "GRAUPELNC",
    "mp_hailnc": "HAILNC",
}

WARM_START_CONTRACT = "gpuwm.offline-child-warm-start.v1"


class WarmStartRefusal(ValueError):
    """A warm start that cannot be applied, with the reason."""


def parse_warm_shift(raw) -> tuple[int, int]:
    parts = str(raw).split(",")
    if len(parts) != 2:
        raise ValueError(f"--warm-shift must be DI,DJ in child cells, got {raw!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(
            f"--warm-shift must be two whole numbers of child cells, got {raw!r}"
        ) from None


def _load(path: Path):
    from woof.io.restart import _HEADER_KEY, _decode_header

    with np.load(path, allow_pickle=False) as data:
        header = _decode_header(data, path)
        stored = {key: data[key] for key in data.files if key != _HEADER_KEY}
    return header, stored


def _stamp(target, source, plan) -> int:
    """Overlap copy of one (…, ny, nx) field; returns the cells stamped."""
    window = plan.window(source.shape)
    if window is None:
        return 0
    (dst_j, src_j), (dst_i, src_i) = window
    chunk = np.ascontiguousarray(source[..., src_j, src_i])
    if hasattr(target, "__cuda_array_interface__"):
        import cupy as cp

        chunk = cp.asarray(chunk)
    target[..., dst_j, dst_i] = chunk
    return int((dst_j.stop - dst_j.start) * (dst_i.stop - dst_i.start))


def _strip_mask(plan, shape, ring: int = 0) -> np.ndarray:
    """True on the cells of a (ny, nx) field that take the parent's rain.

    The cells the overlap did NOT cover, and -- WRF's rule for a moving
    nest, :mod:`woof.core.physics_continuation` -- the earlier child's
    specified ring (``ring`` cells wide) where it lands inside the new
    footprint: the earlier child's microphysics never ran there, so its rain
    total is not a total, and kept it drew a line of too little rain inside
    the new child, one per move.
    """
    ny, nx = shape
    mask = np.ones(shape, dtype=bool)
    window = plan.window(shape)
    if window is not None:
        (dst_j, _), (dst_i, _) = window
        mask[dst_j, dst_i] = False
    if ring > 0:
        jj, ii = np.meshgrid(np.arange(ny) + plan.shift_j,
                             np.arange(nx) + plan.shift_i, indexing="ij")
        inside = (jj >= 0) & (jj < ny) & (ii >= 0) & (ii < nx)
        on_ring = inside & ((jj < ring) | (jj >= ny - ring)
                            | (ii < ring) | (ii >= nx - ring))
        mask |= on_ring
    return mask


def _start_cold(child, cfg, *, checkpoint, header, plan, shift, strip) -> dict:
    """The receipt of a move that shares no ground: nothing is carried, and
    the strip -- every cell -- takes the parent's rain since the child began.
    """
    strip_values = {}
    if strip is not None:
        with np.load(strip, allow_pickle=False) as data:
            strip_values = {key: data[key] for key in data.files}
    existing = getattr(child, "existing_scratch", None)
    accumulators = {}
    for slot in STRIP_ACCUMULATOR_HISTORY:
        target = existing(slot) if callable(existing) else None
        if target is None:
            continue
        entry = {"overlap_cells": 0, "strip_cells": 0}
        seeded = strip_values.get(slot)
        if seeded is not None:
            if tuple(seeded.shape) != tuple(target.shape[-2:]):
                raise WarmStartRefusal(
                    f"strip accumulation {slot} is {tuple(seeded.shape)}, the "
                    f"child's is {tuple(target.shape)}")
            mask = _strip_mask(plan, tuple(target.shape[-2:]),
                               ring=int(getattr(cfg, "spec_zone", 1) or 0))
            host = (np.asarray(target.get()) if hasattr(target, "get")
                    else np.array(target, copy=True))
            host[..., mask] = seeded.astype(host.dtype)[mask]
            if hasattr(target, "__cuda_array_interface__"):
                import cupy as cp

                target[...] = cp.asarray(host)
            else:
                target[...] = host
            entry["strip_cells"] = int(mask.sum())
        accumulators[slot] = entry
    return {
        "contract": WARM_START_CONTRACT,
        "checkpoint": str(checkpoint),
        "checkpoint_valid_time": header.get("valid_time"),
        "shift_child_cells": [int(shift[0]), int(shift[1])],
        "start": "cold",
        "cold_reason": (
            f"a move of ({int(shift[0])}, {int(shift[1])}) child cells is "
            f"larger than this {cfg.nx}x{cfg.ny} child, so it shares no "
            "ground with the earlier one; it starts from the parent here"),
        "overlap_fraction": 0.0,
        "overlap_cells": 0,
        "strip_cells": int(plan.child_cells),
        "state_fields_stamped": 0,
        "state_fields_skipped": {},
        "land_fields_stamped": [],
        "accumulators": accumulators,
        "strip_accumulations": None if strip is None else str(strip),
        "rk_seeds_refreshed": [],
    }


def apply_warm_start(child, cfg, driver, *, checkpoint, shift,
                     strip=None) -> dict:
    """Carry ``checkpoint``'s state onto ``child`` where the two overlap.

    Called by the offline runner after the cold-start child and its physics
    driver are built and before the stepper is made, so a streamed child's
    store is filled from the warmed state.
    """
    from woof.core.diagnostics import update_diagnostics
    from woof.core.nest_relocation import (
        HostStateSnapshot, Placement, plan_relocation, relocatable_attrs,
        transplant_overlap)
    from woof.ingest.nest_init import seed_rk_time_t_copies
    from woof.ingest.relocation_init import LAND_SURFACE_CONTINUATION_FIELDS

    checkpoint = Path(checkpoint)
    shift_i, shift_j = (int(shift[0]), int(shift[1]))
    header, stored = _load(checkpoint)
    config = header.get("config") or {}
    for key, live in (("nx", cfg.nx), ("ny", cfg.ny), ("nz", cfg.nz)):
        if key in config and int(config[key]) != int(live):
            raise WarmStartRefusal(
                f"{checkpoint} is a {config.get('nx')}x{config.get('ny')}x"
                f"{config.get('nz')} child and this one is {cfg.nx}x{cfg.ny}x"
                f"{cfg.nz}; a warm start carries state between placements of "
                "one child, never between two shapes")
    for key, live in (("dx", cfg.dx), ("dy", cfg.dy)):
        if key in config and not np.isclose(float(config[key]), float(live),
                                            rtol=1e-6):
            raise WarmStartRefusal(
                f"{checkpoint} was written at {key}={config[key]} and this "
                f"child runs {key}={live}; the overlap would not be the same "
                "ground")
    # The shift is in child cells, so the plan is taken at ratio 1 between
    # two placements that differ by exactly it (both kept 1-based).
    base = 1 + max(abs(shift_i), abs(shift_j))
    plan = plan_relocation(
        placement_from=Placement(grid_id=int(cfg.grid_id),
                                 i_parent_start=base, j_parent_start=base),
        placement_to=Placement(grid_id=int(cfg.grid_id),
                               i_parent_start=base + shift_i,
                               j_parent_start=base + shift_j),
        parent_grid_ratio=1, child_nx=int(cfg.nx), child_ny=int(cfg.ny))
    if plan.disjoint:
        # A move larger than the grid: no ground is shared, so there is no
        # state to carry, and the child keeps its cold start from the parent
        # at the new place -- the start its first segment had.  The whole
        # grid is fresh ground, so every cell takes the parent's rain since
        # the child began (the strip is the whole grid, as
        # moving_downscale.reseeded_after already counts it).  This used to
        # refuse, which ended a storm-following run at the first move longer
        # than its inner grid: a 1 km grid can jump 60 km at one move, and a
        # 96-cell 333 m grid inside it is 32 km across.
        return _start_cold(child, cfg, checkpoint=checkpoint, header=header,
                           plan=plan, shift=(shift_i, shift_j), strip=strip)

    names = [name for name in relocatable_attrs(child)
             if f"state/{name}" in stored]
    snapshot = HostStateSnapshot(
        {name: stored[f"state/{name}"] for name in names}, pinned=False)
    transplant = transplant_overlap(source_state=snapshot, target_state=child,
                                    plan=plan, attrs=names)

    land = {}
    fields = getattr(driver, "fields", None) or {}
    for name in LAND_SURFACE_CONTINUATION_FIELDS:
        source = stored.get(f"fields/{name}")
        target = fields.get(name) if hasattr(fields, "get") else None
        if source is None or target is None:
            continue
        if tuple(source.shape) != tuple(target.shape):
            raise WarmStartRefusal(
                f"land field {name} is {tuple(source.shape)} in {checkpoint} "
                f"and {tuple(target.shape)} on this child")
        land[name] = _stamp(target, source, plan)

    accumulators = {}
    strip_values = {}
    if strip is not None:
        with np.load(strip, allow_pickle=False) as data:
            strip_values = {key: data[key] for key in data.files}
    existing = getattr(child, "existing_scratch", None)
    for slot in STRIP_ACCUMULATOR_HISTORY:
        target = existing(slot) if callable(existing) else None
        source = stored.get(f"scratch/{slot}")
        if target is None:
            continue
        entry = {"overlap_cells": 0, "strip_cells": 0}
        if source is not None and tuple(source.shape) == tuple(target.shape):
            entry["overlap_cells"] = _stamp(target, source, plan)
        seeded = strip_values.get(slot)
        if seeded is not None:
            if tuple(seeded.shape) != tuple(target.shape[-2:]):
                raise WarmStartRefusal(
                    f"strip accumulation {slot} is {tuple(seeded.shape)}, the "
                    f"child's is {tuple(target.shape)}")
            mask = _strip_mask(plan, tuple(target.shape[-2:]),
                               ring=int(getattr(cfg, "spec_zone", 1) or 0))
            host = (np.asarray(target.get()) if hasattr(target, "get")
                    else np.array(target, copy=True))
            host[..., mask] = seeded.astype(host.dtype)[mask]
            if hasattr(target, "__cuda_array_interface__"):
                import cupy as cp

                target[...] = cp.asarray(host)
            else:
                target[...] = host
            entry["strip_cells"] = int(mask.sum())
        accumulators[slot] = entry

    update_diagnostics(child, int(getattr(cfg, "hypsometric_opt", 1)))
    seeded = seed_rk_time_t_copies(child)
    return {
        "contract": WARM_START_CONTRACT,
        "checkpoint": str(checkpoint),
        "checkpoint_valid_time": header.get("valid_time"),
        "shift_child_cells": [shift_i, shift_j],
        "overlap_fraction": float(plan.overlap_fraction),
        "overlap_cells": int(plan.overlap_cells),
        "strip_cells": int(plan.child_cells - plan.overlap_cells),
        "state_fields_stamped": int(transplant["stamped_field_count"]),
        "state_fields_skipped": dict(transplant["skipped"]),
        "land_fields_stamped": sorted(land),
        "accumulators": accumulators,
        "strip_accumulations": None if strip is None else str(strip),
        "rk_seeds_refreshed": list(seeded),
    }


__all__ = ["STRIP_ACCUMULATOR_HISTORY", "WARM_START_CONTRACT",
           "WarmStartRefusal", "apply_warm_start", "parse_warm_shift"]
