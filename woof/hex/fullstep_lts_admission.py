"""Which rate each cell may take, and which clocks a class map can keep.

CPU only.  Two decisions the active full-step route needs before any device
object exists:

* ``admit_mesh_rates`` -- the dyadic rate of every cell from the mesh spacing,
  capped by the initial advective, acoustic and (with transport) donor bounds
  at the class step, buffered and graded.  This is the admission the dry and
  transport stages measured (``tools/run_fullstep_lts_dry.py``); it lives here
  so the forecast route and the instrument share one rule.
* ``require_class_clock`` -- every global event (radiation, history, the end
  of the run) must fall on a step boundary of the coarsest class.

Classes are static: the bounds are taken from the initial state with margin
and nothing re-admits a class during a run.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Mapping

import numpy as np

EARTH_RADIUS_M = 6371229.0
# The class of a limited-area mesh's lateral-boundary zone.  THE BREAKAGE A
# COARSER ZONE CLASS CAUSES: the relaxation zone's coefficients are per-step
# quantities -- the Rayleigh term (mask-1)/nRelaxZone/(50*dt*scaling) and the
# Laplacian filter's 10*dt clock, both on the outer step
# (``regional_v841.adjust_dynamics_relaxzone_tend``) -- so a zone stepped at
# rate r relaxes toward the boundary data r times more weakly per unit time
# than the same run at the fixed step.  At rate 1 the zone takes the fixed
# run's own step and is forced exactly as the fixed run forces it.
BOUNDARY_RATE = 1
# Interior rings added to the zone so a relaxation-zone cell's stencil
# (its neighbours, and theirs for the Laplacian of the u deviation) never
# reads a cell of another class.
BOUNDARY_BUFFER_RINGS = 2


#: Every switch of a moving variable-resolution region run starts with this
#: (woof.hex.moving_control); the moving controller sets several on each segment.
MOVING_ENV_PREFIX = "WOOF_HEX_MOVING_"

#: Convection-permitting spacing held at the base rate (moving runs, and since H7 every LTS run by default).
MOVING_CONVECTIVE_SPACING_M = 8000.0


def moving_region_requested(environ: Mapping[str, str] | None = None) -> bool:
    """True when any non-empty ``WOOF_HEX_MOVING_*`` switch is set."""

    environ = os.environ if environ is None else environ
    return any(key.startswith(MOVING_ENV_PREFIX) and value for key, value in environ.items())


#: The switches ``woof hex moving`` sets on the forecasts it launches: a segment's carrier, control and
#: remap (``woof.hex.moving_run.RUN_STATE_VARIABLES``) and the post-move shock series a fixed control
#: records.  Knobs such as ``WOOF_HEX_MOVING_BALANCE`` are not among them.
MOVING_LAUNCH_VARIABLES = (
    "WOOF_HEX_MOVING_CHECKPOINT_IN", "WOOF_HEX_MOVING_CHECKPOINT_OUT", "WOOF_HEX_MOVING_CHECKPOINT_WAIT",
    "WOOF_HEX_MOVING_CONTROL", "WOOF_HEX_MOVING_OPERATOR", "WOOF_HEX_MOVING_SOURCE_GRID",
    "WOOF_HEX_MOVING_ZONE", "WOOF_HEX_MOVING_SHOCK_ZONE", "WOOF_HEX_MOVING_SHOCK",
    "GPUWM_HEX_MOVING_ANALYSIS", "GPUWM_HEX_MOVING_BACKGROUND",
)

#: Every moving-route refusal starts with this sentence, so ``woof hex moving`` stops on it at once
#: instead of retrying a command that refuses the same way every time.
MOVING_ROUTE_REFUSAL = ("a moving region runs only on the active local-time-stepping WOOF route "
                        "(one card, or several cards of one host) or on the MPAS suite route (one card)")

# Retired 2026-10-15 for the multi-card class blocks: the move hooks (stop request, carrier export,
# remapped resume per block, shared clock offset, shock series) now exist in the multi-card executor
# (woof.hex.drivers.fullstep_lts_forecast_v841._execute_multi).  "same-host" names what still has none:
# the same-host ranks of the fixed step, which also run a mesh that classes entirely at rate one.
# The MPAS suite route has its move hooks (woof.hex.mpasphys.moving_v842) and is no longer refused here;
# its one narrower gap (zone carry) is refused there, with this same sentence first.
_MOVING_ROUTES = {
    "same-host": ("the same-host ranks of the fixed step (which also run a mesh that classes entirely at rate one)",
                  "Several cards move a region with --fullstep-local-timestep --fullstep-lts-mode active and a "
                  "maximum rate above one"),
    "fixed-step": ("the fixed-step executor", "Moving segments run with --fullstep-local-timestep"),
    # Completeness row #42 (hex 0.4.0 recut): on one card a mesh that classes entirely at rate one (a uniform
    # mesh, or --fullstep-lts-max-rate 1 in a case) is handed to the native single-class step, which reads none
    # of the controller's switches, so a moving segment ran its mesh to the end and never moved.
    "single-class": ("the native single-class step, which runs a mesh that classes entirely at rate one on one card",
                     "Move a region on a mesh whose classes reach above rate one: a variable-resolution mesh with "
                     "--fullstep-lts-max-rate above one (woof hex moving uses 8). A same-mesh resume with no "
                     "controller (WOOF_HEX_MOVING_CHECKPOINT_IN alone) still runs on this step"),
}

#: The launch switches that carry the moving controller's intent to move: its control channel (stop requests,
#: verdicts, recovery seals), the carrier the next member resumes from, and the post-move shock series.  Their
#: hooks exist only in the class executors.  The resume switches (WOOF_HEX_MOVING_CHECKPOINT_IN with its WAIT,
#: ZONE, OPERATOR and SOURCE_GRID) are read by execute_fullstep_lts before dispatch, on every route, so a run that
#: sets only those is a resume with no intent to move: the same-mesh resume identity control
#: (tests/test_moving_inactive_noop.py) runs on the native single-class step unchanged.
MOVING_INTENT_VARIABLES = ("WOOF_HEX_MOVING_CONTROL", "WOOF_HEX_MOVING_CHECKPOINT_OUT",
                           "WOOF_HEX_MOVING_SHOCK", "WOOF_HEX_MOVING_SHOCK_ZONE")


def forecast_moving_route(*, fullstep_local_timestep: bool, fullstep_lts_mode: str | None,
                          fullstep_lts_max_rate: int | None, multi_card: bool) -> str | None:
    """The forecast door's route as :func:`moving_route_refusal` names it, or None for a route with the hooks.

    Active LTS has the move hooks on one card and on several (the multi-card class blocks); the
    fixed-step executor and the same-host ranks of the fixed step have none, nor has a multi-card
    run whose maximum rate is one (it runs those ranks).
    """

    if not multi_card:
        return None if fullstep_local_timestep else "fixed-step"
    if fullstep_local_timestep and fullstep_lts_mode == "active" and fullstep_lts_max_rate != 1:
        return None
    return "same-host"


def moving_launch_switches(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """The ``woof hex moving`` launch switches set (non-empty) in ``environ``."""

    environ = os.environ if environ is None else environ
    return tuple(name for name in MOVING_LAUNCH_VARIABLES if environ.get(name))


def moving_route_refusal(route: str, environ: Mapping[str, str] | None = None) -> str | None:
    """The refusal for a moving launch on a route without the move hooks, or None when nothing is moving.

    THE BREAKAGE THIS PREVENTS: a moving region's stop request, carrier export, remapped resume, clock
    offset and post-move shock series live in the class executors of active local time stepping
    (woof.hex.drivers.fullstep_lts_forecast_v841, one card or several) and on the MPAS suite route
    (woof.hex.mpasphys.moving_v842).  The same-host ranks of the fixed step and the fixed-step executor
    read none of those switches, so a moving segment on either ran to its end on one mesh and never
    moved, and a fixed control recorded no shock series.
    """

    where, remedy = _MOVING_ROUTES[route]
    switches = moving_launch_switches(environ)
    if route == "single-class":
        # Only the controller's intent refuses here: a resume alone runs (see MOVING_INTENT_VARIABLES).
        switches = tuple(name for name in switches if name in MOVING_INTENT_VARIABLES)
    if not switches:
        return None
    return (f"{MOVING_ROUTE_REFUSAL}. None of the move hooks (stop request, carrier export, remapped resume, "
            f"post-move shock series) exist on {where}, so this run ({', '.join(switches)} set) would never "
            f"move and would record no shock series. {remedy}")

# The class of a limited-area mesh's lateral-boundary zone.  THE BREAKAGE A
# COARSER ZONE CLASS CAUSES: the relaxation zone's coefficients are per-step
# quantities -- the Rayleigh term (mask-1)/nRelaxZone/(50*dt*scaling) and the
# Laplacian filter's 10*dt clock, both on the outer step
# (``regional_v841.adjust_dynamics_relaxzone_tend``) -- so a zone stepped at
# rate r relaxes toward the boundary data r times more weakly per unit time
# than the same run at the fixed step.  At rate 1 the zone takes the fixed
# run's own step and is forced exactly as the fixed run forces it.
BOUNDARY_RATE = 1
# Interior rings added to the zone so a relaxation-zone cell's stencil
# (its neighbours, and theirs for the Laplacian of the u deviation) never
# reads a cell of another class.
BOUNDARY_BUFFER_RINGS = 2


def boundary_zone(mesh: Any, cells_on_edge: np.ndarray, n_cells: int) -> np.ndarray:
    """Cells of the specified and relaxation zones plus the buffer rings, as a mask."""

    from .fullstep_lts_classes import inner_edges

    if "bdyMaskCell" not in mesh.arrays:
        raise ValueError(
            "a limited-area mesh (an edge names an absent cell) carries no bdyMaskCell: its "
            "lateral-boundary zone cannot be placed in one rate class")
    mask = np.asarray(mesh.arrays["bdyMaskCell"]).reshape(-1)[:n_cells] > 0
    edges = inner_edges(cells_on_edge)
    for _ in range(BOUNDARY_BUFFER_RINGS):
        touched = mask[edges[:, 0]] | mask[edges[:, 1]]
        grown = mask.copy()
        grown[edges[touched, 0]] = True
        grown[edges[touched, 1]] = True
        mask = grown
    return mask


def metric_lengths(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    """dcEdge in metres and cellsOnEdge; a unit-sphere grid file is scaled by the model radius."""

    dc = np.asarray(mesh.arrays["dcEdge"], dtype=np.float64)
    if dc.max() < 10.0:
        dc = dc * EARTH_RADIUS_M
    return dc, np.asarray(mesh.arrays["cellsOnEdge"], dtype=np.int64)


def initial_donor_fraction(mesh: Any, state: Any, vertical: Any, dt: float) -> np.ndarray:
    """Per column, the largest fraction of a level's dry mass the initial fluxes carry out in ``dt``."""

    rho = np.asarray(state.rho, np.float64)
    ru = np.asarray(state.rho_u, np.float64)
    rw = np.asarray(state.rho_w, np.float64)
    n_levels, n_cells = rho.shape
    _, coe = metric_lengths(mesh)
    dv = np.asarray(mesh.arrays["dvEdge"], np.float64)
    if np.asarray(mesh.arrays["dcEdge"]).max() < 10.0:
        dv = dv * EARTH_RADIUS_M
    area = np.asarray(mesh.area_cell, np.float64)
    rdzw = 1.0 / np.asarray(vertical.dzw, np.float64)
    worst = np.zeros(n_cells)
    # an outer edge of a limited-area mesh names the absent cell (-1): its flux
    # leaves the one real cell, and the absent side's bin is dropped
    coe = np.where(coe < 0, n_cells, coe)
    for k in range(n_levels):
        flux = ru[k] * dv
        out = (np.bincount(coe[:, 0], weights=np.maximum(flux, 0.0), minlength=n_cells + 1)[:n_cells]
               + np.bincount(coe[:, 1], weights=np.maximum(-flux, 0.0), minlength=n_cells + 1)[:n_cells]) / area
        out += rdzw[k] * (np.maximum(rw[k + 1], 0.0) - np.minimum(rw[k], 0.0))
        np.maximum(worst, dt * out / rho[k], out=worst)
    return worst


def admit_mesh_rates(
    mesh: Any,
    state: Any,
    saved: Any,
    vertical: Any,
    *,
    dt: float,
    max_rate: int,
    target_advection: float = 0.80,
    target_acoustic: float = 0.70,
    target_transport: float | None = 0.90,
    demote: Iterable[tuple[np.ndarray, int]] = (),
    acoustic_substeps: int,
    boundary_rate: int = BOUNDARY_RATE,
    convective_spacing_m: float = 8000.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Dyadic rate per cell (int32) and the receipt of how it was decided.

    ``target_transport=None`` leaves the donor bound out (scalar transport off).
    ``demote`` holds masks of cells to at most a rate (instrument maps only).
    ``convective_spacing_m``: cells finer than this stay at rate 1 (see below).
    ``acoustic_substeps`` is the run's N: the acoustic bound is the Courant
    number of one acoustic sub-step, ``dt / (3 N)`` with three dynamics splits.
    A mesh whose geometry gives one class returns all ones before any bound is
    evaluated: the caller dispatches to the native step, and no bound of a
    class map applies to a run that has none.
    """

    from .fullstep_lts import buffer_rates, geometric_rates
    from .fullstep_lts_classes import DYADIC, grade_rates, planes_per_step, cell_spacing, inner_edges

    acoustic_dt = float(dt) / planes_per_step(acoustic_substeps)

    dc, coe = metric_lengths(mesh)
    n_cells = int(np.asarray(state.rho).shape[1])
    info: dict[str, Any] = {"source": "cell minimum dcEdge over the mesh minimum, dyadic"}
    regional = bool(np.any(coe < 0))
    zone = boundary_zone(mesh, coe, n_cells) if regional else None
    if regional:
        # geometric_rates refuses the absent-cell sentinel of a limited-area
        # mesh's outer edges; this is its ladder on the incident-edge minimum
        spacing_geo = cell_spacing(dc, coe, n_cells)
        rates = np.ones(n_cells, dtype=np.int32)
        for rate in DYADIC[1:]:
            if rate <= int(max_rate):
                rates[spacing_geo >= spacing_geo.min() * rate] = rate
    else:
        rates = geometric_rates(dc, coe, n_cells, int(max_rate))
    info["geometric_cells"] = {str(r): int(np.count_nonzero(rates == r)) for r in DYADIC}
    # Convection-permitting cells keep the base step.  Classes are admitted once,
    # from the initial state, and every bound below scales with the cell's
    # horizontal spacing -- but the vertical Courant number of a resolved
    # updraft, w dt / dz, does not: a 6 km ring promoted to 2 dt meets the same
    # 25-35 m/s upper-tropospheric updrafts as the 3 km cap once deep convection
    # grows there, at 37.5 s against ~700-800 m layers (Courant 1.2-1.5).
    # Breakage prevented: the 48 h moving typhoon runs of 2026-10-12 (start
    # member and its fallback both refused the outer step 4.7 h and 7.1 h in, w
    # 23 -> 31 m/s in the last macro) and the earlier long-run blow-ups at the
    # 3 -> 6 km transition (k 32-38).  WOOF_HEX_LTS_CONVECTIVE_SPACING_M
    # overrides the spacing (0 turns the hold off, for the instrument only).
    convective_spacing = float(os.environ.get("WOOF_HEX_LTS_CONVECTIVE_SPACING_M", convective_spacing_m))
    if convective_spacing > 0.0:
        spacing_convective = cell_spacing(dc, coe, n_cells)
        held = (spacing_convective < convective_spacing) & (rates > 1)
        rates = np.where(held, 1, rates).astype(np.int32)
        info["convective_spacing_m"] = convective_spacing
        info["convective_cells_held_at_base_rate"] = int(np.count_nonzero(held))
    if int(rates.max()) == 1:
        info["cells"] = dict(info["geometric_cells"])
        return rates.astype(np.int32), info
    speed = np.abs(np.asarray(saved.normal_velocity, dtype=np.float64)).max(axis=0) / dc
    present = np.where(coe < 0, n_cells, coe)
    advective = np.zeros(n_cells + 1)
    np.maximum.at(advective, present[:, 0], speed)
    np.maximum.at(advective, present[:, 1], speed)
    advective = advective[:n_cells]
    temperature = np.asarray(saved.theta_m, np.float64) * np.asarray(saved.exner, np.float64)
    sound = np.sqrt(1.4 * 287.0 * temperature).max(axis=0)
    spacing = np.full(n_cells + 1, np.inf)
    np.minimum.at(spacing, present[:, 0], dc)
    np.minimum.at(spacing, present[:, 1], dc)
    acoustic = sound / spacing[:n_cells]
    # fraction of the target used at the base step; a class of rate r uses r times it
    used = np.maximum(dt * advective / target_advection, acoustic_dt * acoustic / target_acoustic)
    promote = used
    if target_transport is not None:
        # The donor bound only withholds a promotion: a column at rate 1 takes
        # the same step as the fixed route, whatever its donor fraction.
        donor = initial_donor_fraction(mesh, state, vertical, dt)
        promote = np.maximum(used, donor / target_transport)
        info["largest_base_step_donor_fraction"] = float(donor.max())
    allowed = np.ones(n_cells, dtype=np.int32)
    for rate in DYADIC[1:]:
        allowed[rate * promote <= 1.0] = rate
    rates = np.minimum(rates, allowed).astype(np.int32)
    info["largest_base_step_fraction_of_target"] = float(used.max())
    if np.any(used > 1.0):
        cell = int(np.argmax(used))
        raise ValueError(
            f"cell {cell} exceeds the advective/acoustic bound at the BASE step "
            f"(advective {dt * advective[cell]:.3f}, acoustic {acoustic_dt * acoustic[cell]:.3f}); "
            "a class map cannot repair a base step that is itself too long")
    info["admitted_cells"] = {str(r): int(np.count_nonzero(rates == r)) for r in DYADIC}
    info["targets"] = {"advection": float(target_advection), "acoustic": float(target_acoustic),
                       "acoustic_substeps": int(acoustic_substeps)}
    if target_transport is not None:
        info["targets"]["transport_donor_initial_state"] = float(target_transport)
    rates = buffer_rates(rates, inner_edges(coe))
    for mask, rate in demote:
        rates = np.where(np.asarray(mask, dtype=bool), np.minimum(rates, int(rate)), rates).astype(np.int32)
    if zone is not None:
        rates = np.where(zone, np.minimum(rates, int(boundary_rate)), rates).astype(np.int32)
    rates = grade_rates(rates, coe)
    if zone is not None:
        zone_rates = np.unique(rates[zone])
        if zone_rates.size != 1:
            raise AssertionError("the lateral-boundary zone did not end in one rate class")
        info["boundary_zone"] = {
            "policy": ("every cell of the specified and relaxation zones (bdyMaskCell >= 1) and "
                       f"{BOUNDARY_BUFFER_RINGS} interior rings is held at rate {int(boundary_rate)}, "
                       "so the zone is one class on one clock and its forcing runs on that class's step"),
            "rate": int(zone_rates[0]), "cells": int(np.count_nonzero(zone)),
            "buffer_rings": BOUNDARY_BUFFER_RINGS}
    info["cells"] = {str(r): int(np.count_nonzero(rates == r)) for r in DYADIC}
    return rates.astype(np.int32), info


def _whole_multiple(seconds: float, unit: float) -> bool:
    ratio = float(seconds) / float(unit)
    return ratio >= 1.0 - 1.0e-9 and abs(ratio - round(ratio)) <= 1.0e-9 * max(ratio, 1.0)


def suggest_base_steps(base_dt: float, top_rate: int, events: Mapping[str, float]) -> list[float]:
    """Base steps on the eighth-second grid, nearest ``base_dt``, whose top class keeps every event."""

    found: list[float] = []
    eighths = int(round(base_dt * 8))
    for offset in sorted(range(-eighths // 3, eighths // 3 + 1), key=abs):
        candidate = (eighths + offset) / 8.0
        if candidate > 0 and all(_whole_multiple(v, candidate * top_rate) for v in events.values()):
            found.append(candidate)
        if len(found) == 3:
            break
    return found


def require_class_clock(base_dt: float, rates: Iterable[int], events: Mapping[str, float]) -> dict[str, Any]:
    """Every global event must be a step boundary of every class, i.e. of the coarsest.

    THE BREAKAGE THIS PREVENTS: a class's physics is one resident column
    backend sealed at the class's own step (``rate * base_dt``): radiation
    fires on its own step count and every scheme integrates that duration.
    An event that is not a multiple of the coarsest class step would need an
    event tail, a shortened class step, and the sealed backend refuses any
    step but its own; rounding the event instead would fire radiation or
    write history at different model times in different classes.
    """

    top = max(int(rate) for rate in rates)
    unit = float(base_dt) * top
    offending = {name: float(value) for name, value in events.items() if not _whole_multiple(value, unit)}
    if offending:
        choices = suggest_base_steps(float(base_dt), top, events)
        raise ValueError(
            f"full-step local time stepping: {offending} s is not a whole number of the coarsest "
            f"class step ({top} x {base_dt} s = {unit} s). Each class's physics is sealed at its own "
            "step, so an event inside a class step would fire radiation or write history at "
            "different model times in different classes. Choose a base step whose coarsest class "
            f"step divides every event (nearest: {choices or 'none within a third of this one'} s) "
            "or a lower --fullstep-lts-max-rate")
    return {"coarsest_class_step_seconds": unit,
            "events_in_class_steps": {name: int(round(float(value) / unit)) for name, value in events.items()}}
