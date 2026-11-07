"""Whether active full-step LTS costs less than the fixed step on this mesh, on the run's physics route.

CPU only: the mesh's class census (woof.hex.fullstep_lts_census) priced with the per-step cost measured on the run's
route (``data/fullstep-lts-step-cost.v1.json``, one row per route: table work).  :func:`route_choice` decides;
:func:`mpas_route_choice` is the MPAS suite route's row.

The rows were fit on hex 67a20a51, before the host-bound step fix (fix/hex-host-bound) that cuts the per-step part:
a tree that lowers the per-step host floor re-measures its rows; the rule does not change.  Re-measured on 67a20a51
with the corrected class clock (reference system, one RTX 5090, 3 h each, warm hours 2-3, box load 60-200): MPAS route active
LTS at 15 s 116.1 s per forecast hour, fixed 18 s 85.75; a 1 h run with LTS requested and this choice taking the
fixed step matched the plain fixed run in 0 of 726,897,938 history and diag words.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

#: Switch for the choice: set to 1 to run active LTS even where the measured rule predicts it loses.
FORCE_ENV = "WOOF_HEX_FULLSTEP_LTS_FORCE"

#: The measured cost of one step on one card, per physics route: a fixed part per (class) step plus a part per cell.
#: Each row is a two-point fit to two whole runs on the same mesh, case and card; for the MPAS suite route
#: (HEX-LAM-VS-ZOOM-2026-10-27, RTX 5090, hex 0.4.9 tree 67a20a51, 96 -> 3 km zoom globe of 149,322 cells, warm hours
#: 2-3):
#:   fixed step 18 s:                    85.7 s per forecast hour = 200 steps of (a + 149,322 b)
#:   active LTS, 15 s base, classes 1/2/4/8: 105.9 s per forecast hour = 450 class steps of a + 240 x 76,733 b
#: (76,733 cell steps per base step from ``fullstep_lts_census.mesh_class_census`` of that mesh: 64,832 / 955 /
#: 7,852 / 75,683 cells at rates 1 / 2 / 4 / 8; the run's own plan had 64,832 / 999 / 7,849 / 75,642).
#: Checks against runs not used in the fit: LTS at a 12.5 s base on that mesh, predicted 127.1 s, measured 135.1 s;
#: the 59,405-cell LAM at 18 s (which also updates its boundary every step), predicted 53.7 s, measured 58.2 s.
#: The ArWen route measured the same shape on the same mesh (HEX-LTS-REGRESSION-2026-10-27: 79.3 fixed, 98.4 LTS,
#: 0.81x, the MPAS route's ratio): the fixed part is the per-step host floor of both step loops (eager launches; a
#: class step attaches no subcycle graph).
STEP_COST_TABLE = Path(__file__).with_name("data") / "fullstep-lts-step-cost.v1.json"


def load_step_costs(path: str | os.PathLike | None = None) -> dict[str, dict[str, Any]]:
    """``{route: {"seconds_per_step", "seconds_per_cell_step", "source"}}`` from the step-cost table."""

    import json
    table = json.loads(Path(path or STEP_COST_TABLE).read_text())
    routes = table.get("routes") or {}
    for route, row in routes.items():
        if not (float(row["seconds_per_step"]) >= 0.0 and float(row["seconds_per_cell_step"]) > 0.0):
            raise ValueError(f"step-cost row {route!r} needs a non-negative seconds_per_step and a positive "
                             f"seconds_per_cell_step")
    return {str(route): dict(row) for route, row in routes.items()}


_MPAS_ROW = load_step_costs()["mpas-suite"]
MPAS_STEP_SECONDS = float(_MPAS_ROW["seconds_per_step"])
MPAS_CELL_STEP_SECONDS = float(_MPAS_ROW["seconds_per_cell_step"])
MPAS_COST_SOURCE = str(_MPAS_ROW["source"])


def route_cost(census: Mapping[int, int], *, dt_seconds: float, row: Mapping[str, Any]) -> float:
    """Predicted card seconds per forecast hour for a class census at base step ``dt_seconds`` (one class = fixed)."""

    steps_per_hour = 3600.0 / float(dt_seconds)
    per_step, per_cell = float(row["seconds_per_step"]), float(row["seconds_per_cell_step"])
    return sum(steps_per_hour / int(rate) * (per_step + per_cell * int(cells))
               for rate, cells in census.items() if int(cells) > 0)


def route_choice(census: Mapping[int, int], *, route: str, fixed_dt_seconds: float, lts_dt_seconds: float,
                 costs: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Active LTS or the fixed step on ``route`` ("mpas-suite" or "arwen"), by that route's measured per-step cost.

    The fixed step steps every cell at ``fixed_dt_seconds``; LTS steps class ``r`` at ``r x lts_dt_seconds`` and
    pays the per-step part once per class step.  The cheaper prediction wins; a tie keeps the fixed step (the same
    answer for less machinery).  A route with no row is refused by name (the choice would be a guess).
    """

    costs = load_step_costs() if costs is None else costs
    if route not in costs:
        raise ValueError(f"no measured step cost for the {route!r} route in {STEP_COST_TABLE.name} "
                         f"(rows: {', '.join(sorted(costs))}); measure a fixed and an LTS run and add the row")
    row = costs[route]
    cells = sum(int(count) for count in census.values())
    fixed = route_cost({1: cells}, dt_seconds=fixed_dt_seconds, row=row)
    lts = route_cost(census, dt_seconds=lts_dt_seconds, row=row)
    return {"choice": "lts" if lts < fixed else "fixed", "route": route,
            "predicted_seconds_per_forecast_hour": {"fixed": round(fixed, 1), "lts": round(lts, 1)},
            "fixed_dt_seconds": float(fixed_dt_seconds), "lts_dt_seconds": float(lts_dt_seconds),
            "census_cells_per_rate": {str(rate): int(count) for rate, count in sorted(census.items())},
            "cost_model": {"seconds_per_step": float(row["seconds_per_step"]),
                           "seconds_per_cell_step": float(row["seconds_per_cell_step"]),
                           "source": str(row.get("source", ""))}}


def clock_fits(step_seconds: float, rate: int, events: Mapping[str, float]) -> bool:
    """Whether ``rate x step_seconds`` divides every event (the class clock; mpas_defaults.lts_class_clock_dt)."""

    unit = float(step_seconds) * int(rate)
    values = [float(v) for v in events.values() if v and float(v) > 0.0]
    return all(abs(v / unit - round(v / unit)) <= 1.0e-9 * max(v / unit, 1.0) and v / unit >= 1.0 - 1.0e-9
               for v in values)


def plan_choice(n_cells: int, *, route: str, fixed_dt_seconds: float,
                candidates: list[tuple[int, float, Mapping[int, int]]],
                costs: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The cheapest of the fixed step and active LTS at each candidate ``(top rate, base step, census)``.

    THE BREAKAGE A SINGLE CANDIDATE CAUSED (HEX-LTS-REGRESSION-2026-10-27): the class clock keeps the mesh's top
    class and moves the step down until that class's step divides every event, so on the 96 -> 3 km zoom globe
    (radiation every 1800 s, 8 x 18 s = 144 s does not divide it) LTS ran at 15 s, 20 % more steps than the fixed
    step's 18 s.  A top class of 4 keeps 18 s (4 x 18 s = 72 s divides 1800 s) and steps fewer classes per base
    step; which one is cheaper is the route's measured cost model's answer, not the clock's.  A tie keeps the
    higher top class (the plan the executors take by default).
    """

    costs = load_step_costs() if costs is None else costs
    if route not in costs:
        raise ValueError(f"no measured step cost for the {route!r} route in {STEP_COST_TABLE.name} "
                         f"(rows: {', '.join(sorted(costs))}); measure a fixed and an LTS run and add the row")
    row = costs[route]
    fixed = route_cost({1: int(n_cells)}, dt_seconds=fixed_dt_seconds, row=row)
    rows = [{"top_rate": int(rate), "dt_seconds": float(dt),
             "predicted_seconds_per_forecast_hour": round(route_cost(census, dt_seconds=dt, row=row), 1),
             "census_cells_per_rate": {str(r): int(c) for r, c in sorted(census.items())}}
            for rate, dt, census in candidates]
    best = None
    for candidate in rows:   # candidates come top rate first, so a tie keeps the higher top class
        if best is None or candidate["predicted_seconds_per_forecast_hour"] < best["predicted_seconds_per_forecast_hour"]:
            best = candidate
    lts = None if best is None else best["predicted_seconds_per_forecast_hour"]
    return {"choice": "lts" if lts is not None and lts < round(fixed, 1) else "fixed", "route": route,
            "predicted_seconds_per_forecast_hour": {"fixed": round(fixed, 1), "lts": lts},
            "fixed_dt_seconds": float(fixed_dt_seconds),
            "lts_dt_seconds": None if best is None else best["dt_seconds"],
            "lts_top_rate": None if best is None else best["top_rate"],
            "candidates": rows,
            "census_cells_per_rate": None if best is None else best["census_cells_per_rate"],
            "cost_model": {"seconds_per_step": float(row["seconds_per_step"]),
                           "seconds_per_cell_step": float(row["seconds_per_cell_step"]),
                           "source": str(row.get("source", ""))}}


def mpas_route_cost(census: Mapping[int, int], *, dt_seconds: float) -> float:
    """:func:`route_cost` on the MPAS suite route's row."""

    return route_cost(census, dt_seconds=dt_seconds, row=_MPAS_ROW)


def mpas_route_choice(census: Mapping[int, int], *, fixed_dt_seconds: float, lts_dt_seconds: float) -> dict[str, Any]:
    """:func:`route_choice` on the MPAS suite route."""

    return route_choice(census, route="mpas-suite", fixed_dt_seconds=fixed_dt_seconds,
                        lts_dt_seconds=lts_dt_seconds, costs={"mpas-suite": _MPAS_ROW})
