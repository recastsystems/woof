"""Executable proof of discrete nest relocation on the idealized r3 tree.

The case borrows the N2c ratio-3 WK82 geometry
(:mod:`woof.verify.cases.nest_ideal_r3`) and shortens it: what is under
test is the re-grid at a cycle boundary, not the two-hour storm.  It runs
one short leg to give the child some fine-scale structure of its own, then
performs two relocations on the resulting tree and scores five components.

WHY BOTH DIRECTIONS.  A move test that only checks "the overlap matches"
cannot tell a correct shift from no shift at all, so every component here
is paired with something that must fail if the mechanism is wrong:

1. ``null_move_bit_identity`` -- moving to the SAME placement must leave
   the child bitwise identical.  This is the calibration point: it is the
   one relocation whose answer is known in advance, and it exercises the
   whole path (cold start, donor check, transplant, RK reseed, coupler
   rebuild) rather than short-circuiting.  The state digest alone would
   be partly circular, since at zero shift the stamp covers exactly the
   fields the digest covers, so the component ALSO compares the state the
   transplant does not write: the SINT base state and the map factors,
   which a null move recomputes from scratch and must reproduce exactly,
   and the RK time-t copies, which are checked against WRF
   ``start_domain``'s post-condition. The moist-theta previous thermal
   time level is carried separately and checked by the overlap witness;
   the remaining copies are workspaces refreshed before their next use.
2. ``parent_bitwise_unchanged`` -- the parent's serialised state digest,
   taken around each relocation.  A re-grid reads the parent; if it ever
   writes one, the registered non-mutating-coupling property is gone.
3. ``overlap_bit_identity`` -- on the ground the child still covers, the
   moved child must equal the outgoing child shifted in index space,
   bitwise, on every field of the restart contract.  Measured against a
   HOST copy taken before the move, so the comparison cannot be satisfied
   by the same device buffer being read twice.
4. ``donor_alignment`` -- the child's SINT-derived base state must agree
   bitwise on the overlap.  This is computed by the SINT operator and not
   by the shift arithmetic, so it is an independent witness that the two
   placements really do share donor cells.
5. ``post_move_integration`` -- the relocated tree integrates a second leg
   with the health validator armed, finite everywhere, and no boundary
   blow-up. A re-grid that produced a plausible-looking but unbalanced
   child shows up here rather than in a norm.

The strip left behind by the move is NOT expected to match anything: it is
freshly interpolated from the parent and is the spin-up cost the placement
policy exists to keep small.  It is reported, never asserted away.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from woof.core.model import execute_experiment
from woof.core.nest_relocation import (Placement, base_segment,
                                        plan_relocation, relocatable_attrs,
                                        relocate_child)
from woof.verify.cases.nest_ideal_common import (
    _preserve_parent_map_policy, assemble_idealized_tree,
    consume_history_reflectivity, prepare_idealized_domain, write_json)
from woof.verify.cases.nest_ideal_r3 import _build_root, load_scaffold

#: Sim seconds per leg.  Two legs of ten parent steps is enough for the
#: child to develop structure the parent does not have (which is what
#: makes the overlap transplant a real claim) and short enough to share a
#: card with other work.
LEG_SECONDS = 60.0

#: Default move, in PARENT cells.  Four cells of the 1 km parent is twelve
#: child cells of the 333 m child: a 90 % overlap on a 120-cell child, in
#: the band the architecture's "maximum move per cycle" guard describes.
DEFAULT_MOVE_PARENT_CELLS = 4

COMPONENTS = (
    "null_move_bit_identity",
    "parent_bitwise_unchanged",
    "overlap_bit_identity",
    "donor_alignment",
    "transplant_treatment_proof",
    "post_move_integration",
)

#: The transplant must change a real share of the overlap, or components 3
#: and 4 are satisfied by the cold-start fill alone and prove nothing.  A
#: tenth of the overlap cells is a deliberately low bar: the point is to
#: separate "the treatment fired" from "the arms are identical", not to
#: assert a magnitude.
MIN_TREATED_OVERLAP_FRACTION = 0.10


def short_scaffold(*, run_seconds: float = LEG_SECONDS,
                   i_parent_start: int | None = None,
                   j_parent_start: int | None = None,
                   use_theta_m: int = 0):
    """The N2c geometry over one short leg, optionally re-placed."""
    exp = load_scaffold()
    root, child = exp.domains
    history = run_seconds
    root = replace(
        root, history_interval_s=history,
        run=replace(root.run, run_seconds=run_seconds,
                    output_interval_s=history, use_theta_m=use_theta_m))
    child = replace(
        child, history_interval_s=history,
        run=replace(child.run, run_seconds=run_seconds,
                    output_interval_s=history, use_theta_m=use_theta_m))
    if i_parent_start is not None:
        child = replace(child, i_parent_start=int(i_parent_start),
                        j_parent_start=int(j_parent_start))
    return replace(exp, run_seconds=run_seconds, domains=(root, child))


def _host(value) -> np.ndarray:
    get = getattr(value, "get", None)
    if callable(get) and hasattr(value, "__cuda_array_interface__"):
        return np.ascontiguousarray(get())
    return np.ascontiguousarray(np.asarray(value))


def _bit_mismatches(actual: np.ndarray, expected: np.ndarray) -> int:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return max(int(actual.size), int(expected.size), 1)
    if actual.dtype == np.float32:
        return int(np.count_nonzero(
            actual.view(np.uint32) != expected.view(np.uint32)))
    return int(np.count_nonzero(actual != expected))


def snapshot_child(state) -> dict[str, np.ndarray]:
    """Host copies of everything a relocation is required to carry."""
    out = {}
    for name in relocatable_attrs(state):
        value = getattr(state, name, None)
        if value is not None:
            out[name] = _host(value)
    return out


#: Child state that a relocation REBUILDS rather than copies, and that a
#: null move must therefore reproduce exactly: the SINT-derived base state
#: and the child's own map factors.  Comparing these across a null move is
#: not circular the way comparing the stamped fields to themselves would
#: be -- they are recomputed from the parent and from the child's grid, so
#: any drift in the cold start or the map-field recompute lands here.
REBUILT_NOT_COPIED = ("pb", "alb", "thb", "phb", "mub2d",
                      "msft", "msfu", "msfv")


def snapshot_derived(state) -> dict[str, np.ndarray]:
    """Host copies of :data:`REBUILT_NOT_COPIED`."""
    return {name: _host(getattr(state, name))
            for name in REBUILT_NOT_COPIED
            if getattr(state, name, None) is not None}


def rk_seed_consistency(state) -> dict[str, object]:
    """Check the refreshed RK copies after a move.

    Most time-t arrays are workspaces rewritten before their next use,
    so WRF start_domain seeds them from the current state. With moist
    theta, thp0 instead supplies the previous thermal time level to the
    physics tendency conversion. It belongs to the carried inventory,
    checked by the independent overlap comparison below.
    """
    from woof.ingest.nest_init import RK_TIME_T_SEED_PAIRS

    fields = {}
    for current, seed in RK_TIME_T_SEED_PAIRS:
        if seed == "thp0" and getattr(state, "use_theta_m", 0):
            # Moist physics reads the previous thermal time level before
            # small_step_prep; overlap_component checks its carried words.
            continue
        value = getattr(state, current, None)
        copy = getattr(state, seed, None)
        if value is None or copy is None:
            continue
        fields[seed] = {
            "count": int(_host(copy).size),
            "bit_mismatches": _bit_mismatches(_host(copy), _host(value)),
        }
    return {
        "note": ("RK time-t copies equal their current field, per WRF "
                 "start_domain (selected moist thp0 is carried separately); "
                 "the other pre-move values are per-step scratch "
                 "and are refreshed before their next use"),
        "fields": fields,
        "compared_fields": len(fields),
        "pass": bool(fields and all(item["bit_mismatches"] == 0
                                    for item in fields.values())),
    }


def overlap_component(before: dict[str, np.ndarray], state,
                      plan) -> dict[str, object]:
    """Score the transplant field by field against the pre-move host copy."""
    fields: dict[str, object] = {}
    for name, source in before.items():
        target = getattr(state, name, None)
        if target is None:
            fields[name] = {"pass": False, "reason": "absent after the move"}
            continue
        moved = _host(target)
        window = plan.window(source.shape)
        if window is None:
            fields[name] = {"pass": False, "reason": "disjoint footprints"}
            continue
        (dst_j, src_j), (dst_i, src_i) = window
        actual = np.ascontiguousarray(moved[..., dst_j, dst_i])
        expected = np.ascontiguousarray(source[..., src_j, src_i])
        mismatches = _bit_mismatches(actual, expected)
        strip_finite = bool(np.isfinite(moved).all())
        fields[name] = {
            "overlap_cells": int(actual.size),
            "overlap_bit_mismatches": mismatches,
            "whole_field_finite": strip_finite,
            "whole_field_max_abs": float(np.max(np.abs(
                moved.astype(np.float64)))) if moved.size else 0.0,
            "pass": bool(mismatches == 0 and strip_finite and actual.size > 0),
        }
    return {
        "compared_fields": sorted(fields),
        "fields": fields,
        "pass": bool(fields and all(item["pass"] for item in fields.values())),
    }


def treatment_component(cold: dict[str, np.ndarray], state,
                        plan) -> dict[str, object]:
    """Prove the transplant did something the cold start had not already done.

    ``cold`` is the child EXACTLY as the parent-interpolated cold start left
    it, captured through the ``on_child_built`` seam before the stamp.  If
    the stamped overlap equalled that, then "the overlap matches the shifted
    old child" would be a statement about SINT reproducibility and not about
    the transplant at all -- the two arms would be the same arm.  This
    component counts the cells where they differ.
    """
    fields: dict[str, object] = {}
    treated = compared = 0
    for name, cold_value in cold.items():
        target = getattr(state, name, None)
        if target is None:
            continue
        window = plan.window(cold_value.shape)
        if window is None:
            continue
        (dst_j, _), (dst_i, _) = window
        final = np.ascontiguousarray(_host(target)[..., dst_j, dst_i])
        before = np.ascontiguousarray(cold_value[..., dst_j, dst_i])
        changed = _bit_mismatches(final, before)
        treated += changed
        compared += int(final.size)
        fields[name] = {
            "overlap_cells": int(final.size),
            "cells_the_transplant_changed": changed,
            "fraction_changed": (0.0 if final.size == 0
                                 else changed / final.size),
        }
    fraction = 0.0 if compared == 0 else treated / compared
    return {
        "note": ("cells where the stamped value differs from the "
                 "parent-interpolated cold start; zero would mean the "
                 "transplant is a no-op and the overlap components are "
                 "vacuous"),
        "fields": fields,
        "overlap_cells_compared": compared,
        "overlap_cells_changed": treated,
        "fraction_changed": fraction,
        "minimum_fraction": MIN_TREATED_OVERLAP_FRACTION,
        "pass": bool(compared > 0
                     and fraction >= MIN_TREATED_OVERLAP_FRACTION),
    }


def _child_preparer(start_time, capture=None):
    """The caller seam: idealized map policy, then a fresh physics driver.

    The cold-start path builds no physics driver, and the restart-serialised
    inventory a relocation carries deliberately excludes physics
    continuation state.  Re-initialising it here is the same thing every
    leg boundary on the DA route already does, and it is named rather than
    implied: what the child keeps across a move is its serialised state,
    not its accumulators.
    """
    def prepare(initialized, new_dc, parent_node) -> None:
        _preserve_parent_map_policy(parent_node.state, initialized.state)
        prepare_idealized_domain(
            initialized.state, new_dc, initialized.grid, start_time)
        if capture is not None:
            # The cold start is complete and the stamp has not happened
            # yet; this is the only instant the untreated arm exists.
            capture.update(snapshot_child(initialized.state))
    return prepare


def _adopting_initializer(state, grid):
    """Carry an already-relocated child into the next leg's assembly."""
    def initialize(child_dc, parent_node, **_kwargs):
        from woof.ingest.nest_init import ChildInitResult

        return ChildInitResult(
            state=state, grid=grid, coord=None, real=None,
            static_fields=None, horizontal=None, soil=None,
            domain=child_dc)
    return initialize


def run(outdir: str | Path, *,
        move_parent_cells: int = DEFAULT_MOVE_PARENT_CELLS,
        leg_seconds: float = LEG_SECONDS,
        use_theta_m: int = 0) -> dict[str, object]:
    """Integrate, relocate twice, integrate again, and score five components."""
    from woof.core.dycore import stability_report
    from woof.static.lambert import grids_from_projection_config
    from woof.verify.metrics import boundary_zone_blowup
    from woof.verify.nest_gates import gate

    # Imported, not restated: the post-move boundary check is the SAME
    # frozen predicate and threshold the static ratio-3 case is held to,
    # so a relocated child is not graded on an easier curve.
    boundary_gate = gate("N2", "wk_r3_boundary_zone_blowup")
    if boundary_gate.kind != "max" or boundary_gate.threshold is None:
        raise RuntimeError("the N2c boundary gate kind/threshold changed")

    outdir = Path(outdir)
    exp = short_scaffold(run_seconds=float(leg_seconds),
                         use_theta_m=use_theta_m)
    root_dc, child_dc = exp.domains
    start_placement = Placement(
        grid_id=child_dc.grid_id,
        i_parent_start=child_dc.i_parent_start,
        j_parent_start=child_dc.j_parent_start)

    # ---- leg A: give the child structure of its own ----------------------
    grids = grids_from_projection_config(exp)
    model = assemble_idealized_tree(exp, _build_root(exp), grids=grids)
    child = model.node(child_dc.grid_id)
    leg_a = execute_experiment(model)
    clock_lead = int(model.root.clock.ticks) - int(child.clock.ticks)

    segment = base_segment(child_dc)
    prepare_child = _child_preparer(exp.start_time)

    # ---- component 1: the null move --------------------------------------
    # The stamped fields are compared through the state digest, and the
    # NON-stamped ones (RK seeds, base state, map factors) through their
    # own host copies.  The second half is the part that is not circular:
    # a null move rebuilds all of it from scratch, so any drift in the
    # cold start, the reseed or the map-field recompute lands here.
    derived_before = snapshot_derived(child.state)
    null_receipt = relocate_child(
        child, i_parent_start=start_placement.i_parent_start,
        j_parent_start=start_placement.j_parent_start,
        segment=segment, on_child_built=prepare_child)
    derived_after = snapshot_derived(child.state)
    derived = {}
    for name, expected in derived_before.items():
        actual = derived_after.get(name)
        derived[name] = {
            "count": int(expected.size),
            "bit_mismatches": (None if actual is None
                               else _bit_mismatches(actual, expected)),
        }
    derived_pass = bool(derived and all(
        item["bit_mismatches"] == 0 for item in derived.values()))
    seeds = rk_seed_consistency(child.state)
    segment_json = null_receipt["segment"]
    null_component = {
        "placement": list(start_placement.position),
        "child_state_sha256_before": null_receipt["child_state_sha256_before"],
        "child_state_sha256_after": null_receipt["child_state_sha256_after"],
        "child_bitwise_identical": bool(null_receipt["child_state_unchanged"]),
        "parent_bitwise_unchanged": bool(
            null_receipt["parent_bitwise_unchanged"]),
        "overlap_fraction": null_receipt["plan"]["overlap_fraction"],
        "transplanted_fields": null_receipt["transplant"]["stamped_field_count"],
        "rebuilt_not_copied": {
            "note": ("SINT base state and map factors -- recomputed by the "
                     "move, so a null move must reproduce them exactly"),
            "fields": derived,
            "compared_fields": len(derived),
            "pass": derived_pass,
        },
        "rk_seed_consistency": seeds,
        "pass": bool(null_receipt["child_state_unchanged"]
                     and null_receipt["parent_bitwise_unchanged"]
                     and null_receipt["plan"]["overlap_fraction"] == 1.0
                     and derived_pass and seeds["pass"]),
    }

    # ---- component 2-4: the real move ------------------------------------
    before = snapshot_child(child.state)
    moved_to = Placement(
        grid_id=child_dc.grid_id,
        i_parent_start=start_placement.i_parent_start
        + int(move_parent_cells),
        j_parent_start=start_placement.j_parent_start,
        generation=1)
    plan = plan_relocation(
        placement_from=start_placement, placement_to=moved_to,
        parent_grid_ratio=child_dc.parent_grid_ratio,
        child_nx=child_dc.run.nx, child_ny=child_dc.run.ny)
    cold: dict[str, np.ndarray] = {}
    move_receipt = relocate_child(
        child, i_parent_start=moved_to.i_parent_start,
        j_parent_start=moved_to.j_parent_start,
        segment=null_receipt["segment_state"],
        on_child_built=_child_preparer(exp.start_time, capture=cold))

    overlap = overlap_component(before, child.state, plan)
    treatment = treatment_component(cold, child.state, plan)
    parent_component = {
        "null_move": {
            "before": null_receipt["parent_state_sha256_before"],
            "after": null_receipt["parent_state_sha256_after"],
            "unchanged": bool(null_receipt["parent_bitwise_unchanged"]),
        },
        "real_move": {
            "before": move_receipt["parent_state_sha256_before"],
            "after": move_receipt["parent_state_sha256_after"],
            "unchanged": bool(move_receipt["parent_bitwise_unchanged"]),
        },
        "pass": bool(null_receipt["parent_bitwise_unchanged"]
                     and move_receipt["parent_bitwise_unchanged"]),
    }
    donor_component = {
        "null_move": null_receipt["donor_alignment"],
        "real_move": move_receipt["donor_alignment"],
        "pass": bool(null_receipt["donor_alignment"]["pass"]
                     and move_receipt["donor_alignment"]["pass"]),
    }

    # ---- component 5: integrate the relocated tree -----------------------
    exp_b = short_scaffold(
        run_seconds=float(leg_seconds),
        i_parent_start=moved_to.i_parent_start,
        j_parent_start=moved_to.j_parent_start,
        use_theta_m=use_theta_m)
    model_b = assemble_idealized_tree(
        exp_b, model.root.state,
        grids=grids_from_projection_config(exp_b),
        child_initializer=_adopting_initializer(child.state, child.grid),
        domain_preparer=lambda *_args, **_kwargs: None)
    child_b = model_b.node(child_dc.grid_id)
    gate_width = child_b.cfg.run.spec_bdy_width
    series: list[dict[str, object]] = []

    def snapshot(_model, node, ticks):
        consume_history_reflectivity(node, ticks)
        if node is not child_b:
            return
        health = stability_report(
            node.state, node.cfg.run, boundary_width=gate_width)
        blowup = float(boundary_zone_blowup(
            health["boundary_w_max"], health["interior_w_max"]))
        finite = all(
            bool(np.isfinite(_host(getattr(node.state, name))).all())
            for name in relocatable_attrs()
            if getattr(node.state, name, None) is not None)
        series.append({
            "ticks": int(ticks),
            "elapsed_seconds": float(node.clock.elapsed_seconds),
            "boundary_max_abs_w_ms": float(health["boundary_w_max"]),
            "interior_max_abs_w_ms": float(health["interior_w_max"]),
            "boundary_zone_blowup": blowup,
            "state_finite": finite,
            "pass": bool(finite and np.isfinite(blowup)
                         and blowup <= boundary_gate.threshold),
        })

    leg_b = execute_experiment(model_b, history_handler=snapshot,
                               validate_state=True)
    integration_component = {
        "leg_seconds": float(leg_seconds),
        "steps": int(leg_b.steps),
        "forces": int(leg_b.forces),
        "health_validator_armed": True,
        "boundary_gate": {
            "metric": boundary_gate.metric,
            "threshold": float(boundary_gate.threshold),
        },
        "series": series,
        "pass": bool(series and all(item["pass"] for item in series)
                     and int(leg_b.steps) > 0),
    }

    components = {
        "null_move_bit_identity": null_component,
        "parent_bitwise_unchanged": parent_component,
        "overlap_bit_identity": overlap,
        "donor_alignment": donor_component,
        "transplant_treatment_proof": treatment,
        "post_move_integration": integration_component,
    }
    report = {
        "case": "nest_relocate_r3",
        "use_theta_m": int(use_theta_m),
        "geometry": {
            "parent": [child_dc.parent_grid_ratio,
                       int(root_dc.run.nx), int(root_dc.run.ny),
                       int(root_dc.run.nz)],
            "child": [int(child_dc.run.nx), int(child_dc.run.ny),
                      int(child_dc.run.nz)],
            "placement_from": list(start_placement.position),
            "placement_to": list(moved_to.position),
            "move_parent_cells": int(move_parent_cells),
        },
        "plan": plan.to_json(),
        "leg_a": {"steps": int(leg_a.steps), "forces": int(leg_a.forces),
                  "clock_lead_at_relocation_ticks": clock_lead},
        "null_move_segment": segment_json,
        "real_move_segment": move_receipt["segment"],
        "coupler": move_receipt["coupler"],
        "transplant": move_receipt["transplant"],
        "components": components,
        "verdict_rule": "pass iff every component passes",
        "pass": bool(all(components[name]["pass"] for name in COMPONENTS)),
    }
    write_json(outdir / "report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m woof.verify.cases.nest_relocate")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--move-parent-cells", type=int,
                        default=DEFAULT_MOVE_PARENT_CELLS)
    parser.add_argument("--leg-seconds", type=float, default=LEG_SECONDS)
    parser.add_argument("--use-theta-m", type=int, choices=(0, 1), default=0)
    args = parser.parse_args(argv)
    report = run(args.outdir, move_parent_cells=args.move_parent_cells,
                 leg_seconds=args.leg_seconds, use_theta_m=args.use_theta_m)
    print(json.dumps(
        {name: report["components"][name]["pass"] for name in COMPONENTS},
        indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


__all__ = [
    "COMPONENTS", "DEFAULT_MOVE_PARENT_CELLS", "LEG_SECONDS", "main",
    "overlap_component", "run", "short_scaffold", "snapshot_child",
]


if __name__ == "__main__":  # pragma: no cover - controller entry point
    raise SystemExit(main())
