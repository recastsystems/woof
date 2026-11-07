"""Global continuation decisions shared by resident and slab rebuilds."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RelocationContinuation:
    captured: dict
    plan: object
    statics_verdict: dict
    fill: object
    land: dict


def stage_relocation_continuation(captured, new_dc, static, *, plan=None, parent_node=None):
    """Shift on the full domain and choose global same-class donor columns.

    The result may be sliced for bounded device construction afterwards.
    Choosing donors separately within slabs would change scientific fill at
    slab boundaries, so no window enters this operation.
    """
    from woof.core.nest_relocation import Placement, RelocationRefusal, plan_relocation
    from woof.ingest.relocation_init import (
        donor_fill_plan, overlap_mask_for_plan, overlap_statics_mismatches)

    if captured is None or captured["grid_id"] != int(new_dc.grid_id):
        raise RelocationRefusal(
            "RealRelocationChildPreparer.__call__ without a matching "
            "capture_outgoing: the runner drives both seams, and a "
            "rebuild that never saw the outgoing child has no land "
            "state to move")
    if static is None:
        raise RelocationRefusal(
            "the relocation initializer produced no static fields; "
            "the real-data route requires footprint-rebuilt statics")
    cfg = new_dc.run
    # Descendants can move in ground coordinates without changing their
    # placement relative to their parent. The route's override is authority.
    if plan is None:
        plan = plan_relocation(
            placement_from=Placement(
                grid_id=captured["grid_id"],
                i_parent_start=captured["i_parent_start"],
                j_parent_start=captured["j_parent_start"]),
            placement_to=Placement(
                grid_id=int(new_dc.grid_id),
                i_parent_start=int(new_dc.i_parent_start),
                j_parent_start=int(new_dc.j_parent_start), generation=1),
            parent_grid_ratio=int(new_dc.parent_grid_ratio),
            child_nx=int(cfg.nx), child_ny=int(cfg.ny))
    if captured["static_fields"] is None:
        raise RelocationRefusal(
            "the outgoing child's statics are not on record, so the "
            "overlap-statics equality cannot be asserted; a move "
            "whose essential claim cannot be checked is refused")
    verdict = overlap_statics_mismatches(captured["static_fields"], static, plan)
    if not verdict["pass"]:
        raise RelocationRefusal(
            "footprint-rebuilt statics differ from the outgoing "
            "child's on shared ground beyond one-ulp rounding; "
            "transplanting would combine the carried "
            "state with different land-surface statics: "
            f"{verdict['mismatched_fields'] or verdict}")
    # Resident forecasts already own a CUDA context. Keep nearest-donor
    # search there; the old pairwise CPU search stalled every ocean move.
    from woof.core.physics_continuation import _array_module
    device = captured.get("device", False) or any(
        hasattr(value, "__cuda_array_interface__")
        for value in captured.get("continuation", {}).values())
    fill = donor_fill_plan(overlap_mask=overlap_mask_for_plan(plan, (cfg.ny, cfg.nx)),
                           landmask=np.asarray(static["LANDMASK"]), device=device)
    # Ocean profiles on newly exposed water come from the live parent.
    # The overlap retains the outgoing child bit for bit, including its wake.
    from woof.ingest.relocation_init import PWP_CONTINUATION_FIELDS
    ocean = set(captured["fields"]) & set(PWP_CONTINUATION_FIELDS)
    parent_ocean = {}
    if ocean:
        if ocean != set(PWP_CONTINUATION_FIELDS):
            raise RelocationRefusal("moving PWP requires the outgoing child's complete ocean state")
        from woof.ocean.mixed_layer.nest import parent_columns
        driver = getattr(getattr(parent_node, "state", None), "physics", None)
        if driver is None or not ocean <= set(driver.fields):
            raise RelocationRefusal("moving PWP needs the live parent's complete ocean state")
        parent_ocean = parent_columns(driver.fields, new_dc, np.asarray(static["LANDMASK"]))
    moved = {}
    for name, old in captured["fields"].items():
        window = plan.window(old.shape)
        if window is None:
            continue
        (dst_j, src_j), (dst_i, src_i) = window
        xp = _array_module(old)
        if name in parent_ocean:
            staged = xp.asarray(parent_ocean[name], dtype=old.dtype).copy()
            if staged.shape != old.shape:
                raise RelocationRefusal("parent and child PWP must use the same ocean layers")
            staged[..., dst_j, dst_i] = old[..., src_j, src_i]
            moved[name] = staged
        else:
            staged = xp.zeros_like(old)
            staged[..., dst_j, dst_i] = old[..., src_j, src_i]
            moved[name] = fill.apply(staged)
    return RelocationContinuation(captured, plan, verdict, fill, moved)
