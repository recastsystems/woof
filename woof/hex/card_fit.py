"""Global mesh predictions and process-scoped whole-run measurements.

Memory comes from device_admission. Runtime extrapolates, by cell-steps, a
whole run measured on the same physics route, card and card count, and is
explicitly not a measurement of the new geometry.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence

from .errors import MpasPortError


#: Headroom over the measured whole-run curve. The curve passes through each
#: run's sampled peak, which a 1 s sampler can miss by a subsecond spike, and
#: runs between two measured sizes are interpolated. 10 % puts every measured
#: run of today's build 10-12 % under its prediction (tests/test_global_vr_mesh_plan.py
#: test_shipped_curve_lands_10_to_15_percent_over_every_measured_run).
WHOLE_RUN_MARGIN = 0.10


def whole_run_curve(rows: Sequence[Mapping[str, Any]], count: int, *,
                    base_mib: Any, below_shape_mib: Any = None) -> dict[str, Any] | None:
    """Peak MiB from this card's measured whole runs, plus WHOLE_RUN_MARGIN.

    THE BREAKAGE IT PREVENTS (WOOF run-controls, 2026-10-24): the nearest run
    at or below a size was held CONSTANT below the smallest run, so an
    80 -> 5 km zoom globe of 128,942 cells was priced at 25,272 MiB (the
    216,189-cell run) and peaked at 13,036 MiB; a 52,618-cell globe was priced
    the same and peaked at 7,472 MiB. Quotes, refusals and card leases were
    sized 2-3.4x too big. Before that, the largest floor over every run priced
    the 581,895-cell globe at 68.0 GB against 44,450 MiB (feasibility defect 3).

    Two or more runs: straight between neighbouring measured runs (peaks made
    non-decreasing in cells), the first segment's slope below the smallest and
    the last segment's slope past the largest. One run: below its size, the
    ``below_shape_mib(cells, run)`` curve (a profile's plain suite, route_floor;
    ``run`` None prices the plan's card) scaled by the run's ratio to it at the
    run's own size, card and card memory, else device_admission's own slope
    through it; cell-linear above
    (the old anchor's conservative side). Derived between and past the
    measured sizes, not measured there.

    THE BREAKAGE THE SHAPE ARM PREVENTS (mesh-gen-fast, RTX 4090,
    2026-10-29): the hrrr-like-aerosol default has one one-card run (286,530
    cells, 44,989 MiB) and device_admission's WOOF slope through it is ~49 KB
    per cell, so it left a ~33 GiB intercept: 124,855 cells were priced at
    41,960.5 MiB required, 60,000 at 38,300 and 40,000 at 37,094, and no
    custom mesh worth running was admitted on a 24 GB card. The plain suite's
    own curve falls ~84 KB per cell toward its measured small runs (40,962
    cells: 12,664 MiB), and a profile only adds per-cell fields to it.
    """
    if not rows:
        return None
    by_cells: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        cells = int(row["cells"])
        if cells not in by_cells or row["peak_process_vram_mib"] > by_cells[cells]["peak_process_vram_mib"]:
            by_cells[cells] = row
    points = sorted(by_cells.values(), key=lambda row: row["cells"])
    xs = [float(row["cells"]) for row in points]
    ys: list[float] = []
    for row in points:
        ys.append(max([float(row["peak_process_vram_mib"]), *ys[-1:]]))
    x = float(count)
    if len(points) == 1:
        anchor_shape = here_shape = None
        if x <= xs[0] and below_shape_mib is not None:
            # The ratio like for like: the plain suite on the run's own card and card memory; the shape on the plan's.
            anchor_shape = below_shape_mib(int(xs[0]), points[0])
            here_shape = below_shape_mib(count, None)
        if x <= xs[0] and anchor_shape and here_shape and anchor_shape > 0 and here_shape > 0:
            ratio = ys[0] / anchor_shape
            curve = here_shape * ratio
            basis = (f"one measured whole run: below its size the plain suite's curve times the run's ratio to it "
                     f"on its own card and card memory at its size ({ys[0]:,.0f} / {anchor_shape:,.0f} MiB = {ratio:.4f})")
        elif x <= xs[0]:
            curve = ys[0] + float(base_mib(count)) - float(base_mib(int(xs[0])))
            basis = "one measured whole run: device_admission's slope through it below its size"
        else:
            curve = ys[0] * x / xs[0]
            basis = "one measured whole run: cell-linear past its size"
    else:
        segment = next((i for i in range(1, len(xs)) if x <= xs[i]), len(xs) - 1)
        x0, x1, y0, y1 = xs[segment - 1], xs[segment], ys[segment - 1], ys[segment]
        curve = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        basis = ("straight between the measured whole runs" if xs[0] <= x <= xs[-1] else
                 "the first measured segment's slope below the smallest run" if x < xs[0] else
                 "the last measured segment's slope past the largest run")
    nearest = min(points, key=lambda row: abs(math.log(x / float(row["cells"]))))
    floor = max(0.0, curve) * (1.0 + WHOLE_RUN_MARGIN)
    return {"observation": dict(nearest), "observations": [row["id"] if "id" in row else row.get("receipt_sha256")
                                                           for row in points],
            "curve_mib": round(curve, 3), "margin_fraction": WHOLE_RUN_MARGIN,
            "floor_mib": round(floor, 3),
            "basis": f"{basis}, plus {WHOLE_RUN_MARGIN:.0%}; derived at other sizes, not measured there"}


#: THE BREAKAGE IT PREVENTS (box VRFS, 8 x RTX 5090, woof hex 0.4.9, 2026-10-28): the plan priced every globe from
#: the whole runs in card-fit-observations.v1.json whatever physics it would run, and every run there was an ArWen-
#: route run (hex 0.4.3 and 0.4.4, before MPAS physics became the default in 0.4.6).  The 119,691-cell 96 -> 3 km zoom
#: globe on the default MPAS physics was priced at a 13,821 MiB peak and peaked at 19,187 MiB; the 286,530-cell one was
#: priced at 31,865 MiB and ran one card out of memory at its first microphysics step needing at least 33,580 MiB.  The
#: plan also had no card count: priced against two cards' memory (--vram-gib 64), the 417,810-cell globe "fit" and
#: ran both cards out of memory at the first physics step.  So a plan is priced from the whole runs of the physics it
#: runs (route and suite) and, per card, at the card count it runs on (``cards``).
#:
#: Statuses a whole-run row may carry: a completed run (its sampled per-card peak) and a run that stopped out of
#: memory (``at_least_mib``: what the card held plus the request that failed, a floor on the real need).
CURVE_STATUSES = ("complete", "out_of_memory")

#: THE BREAKAGE IT PREVENTS (hex 0.4.10 review, 2026-10-29): 0.4.10's regional GFS cold-start default is
#: convection_permitting with the hrrr-like-aerosol profile (physics-default.v1.json recipe_rules: aerosol Thompson,
#: three more transported scalars nc/nifa/nwfa, the aerosol workspace), but the plan priced plain
#: convection_permitting, so on the r700 zoom globe it printed "convection_permitting ... 28,715 MiB per card" for a
#: forecast that ran the aerosol profile.  A profile is priced from its own whole runs (rows carry ``profile``), and
#: never below its plain suite's price (a profile only adds fields).  Rows with no ``profile`` are the plain suite.


def shipped_observations() -> list[dict[str, Any]]:
    return list(json.loads((Path(__file__).with_name("data") /
                            "card-fit-observations.v1.json").read_text())["observations"])


#: THE BREAKAGE CARD CAPACITY PREVENTS (hex 0.4.10 review, V8X6000 2026-10-29): a run's peak is the memory its
#: processes held, and a hex process holds more on a card with room to spare: device_memory_release keeps idle pool
#: sets while 15 % of the card is free, and several schemes size their tiles from the free memory (RRTMG, MYNN, GF).
#: The 2-card aerosol r700 row (30,813 MiB, measured on 96 GB cards) priced 2 x RTX 5090 at 33,894 MiB per card and
#: refused it, where the same run held to a 5090's 32,607 MiB ran (card-fit-observations: the card_capacity_mib row).
#: So at each size a plan takes the runs measured with the card memory nearest its own card's from above (else the
#: most below); a row's capacity is ``card_capacity_mib`` when the run was held to less than its card, else its card's.


def row_capacity_mib(row: Mapping[str, Any]) -> float:
    """The device memory a measured run had on each card: ``card_capacity_mib``, else its card's nameplate."""
    if row.get("card_capacity_mib") is not None:
        return float(row["card_capacity_mib"])
    from .mesh_point import CARD_ALIASES

    for key, _generator, nameplate in CARD_ALIASES.values():
        if key == str(row.get("card")):
            return float(nameplate)
    return math.inf


#: How far below a plan's card memory a held run still counts as measured on that card: a card's nameplate and the
#: total its driver reports differ (an RTX 4090: 24,576 vs 24,564 MiB).
HELD_CAPACITY_TOLERANCE = 0.01


def _by_capacity(points: Sequence[dict[str, Any]], capacity_mib: float) -> tuple[list[dict[str, Any]], list[float]]:
    """At each size, the completed runs measured with the smallest card memory at or above ``capacity_mib`` (else
    the largest below it); out-of-memory floors are kept whatever card they ran on (a floor on the need anywhere).
    A run held below its card (``card_capacity_mib``) never prices a card with more memory than it had: it is a
    measurement of that memory only, and a hex process holds more where there is room (THE BREAKAGE IT PREVENTS,
    mesh-gen-fast 2026-10-29: the 124,859-cell aerosol run held to a 4090's 24,564 MiB peaked at 18,916 MiB, which
    a 5090 or a 96 GB card would have taken as its price at that size).
    Returns the kept points and the capacities the completed ones were measured with."""
    groups: dict[int, list[dict[str, Any]]] = {}
    kept = [row for row in points if row["status"] != "complete"]
    for row in points:
        if row["status"] == "complete":
            held = row.get("card_capacity_mib")
            if held is not None and float(held) < float(capacity_mib) * (1.0 - HELD_CAPACITY_TOLERANCE):
                continue
            groups.setdefault(int(row["cells"]), []).append(row)
    used: set[float] = set()
    for rows in groups.values():
        capacities = {row_capacity_mib(row) for row in rows}
        above = sorted(value for value in capacities if value >= capacity_mib)
        pick = above[0] if above else max(capacities)
        used.add(pick)
        kept += [row for row in rows if row_capacity_mib(row) == pick]
    return kept, sorted(used)


def run_physics(suite: str | None = None, route: str | None = None, *, profile: str | None = None,
                context: Mapping[str, Any] | None = None) -> tuple[str, str, str | None]:
    """The (route, suite, profile) a plan prices: as named, else the hex default the forecast will take
    (MPAS physics, its suite by mesh; with a door's recipe ``context`` the first matching recipe_rules row, so a
    regional GFS cold start is priced on its suite and profile, physics_default.ensure)."""
    from . import physics_default
    from .physics_suites import mpas_suite_names

    if suite is None:
        if route not in (None, "mpas"):
            return str(route), physics_default.arwen_suite(), None
        rule = physics_default.default_rule(None, context=context)
        return "mpas", str(rule["suite"]), (profile if profile is not None else rule.get("profile"))
    if route is None:
        route = "mpas" if suite in mpas_suite_names() else "arwen"
    return str(route), str(suite), profile


def _rank_rows(rows: Sequence[Mapping[str, Any]], levels: int) -> list[dict[str, Any]]:
    """Whole-run rows as curve points: columns per card (at ``levels``-level equivalence) against the per-card peak
    (an out-of-memory row's floor)."""
    out = []
    for row in rows:
        cards = max(1, int(row.get("cards", 1)))
        peak = row["peak_process_vram_mib"] if row["status"] == "complete" else row["at_least_mib"]
        columns = float(row["cells"]) / cards * float(row["levels"]) / float(levels)
        out.append({**row, "cells": int(math.ceil(columns)), "peak_process_vram_mib": float(peak),
                    "global_cells": int(row["cells"]), "cards": cards})
    return out


def route_floor(observations: Sequence[Mapping[str, Any]], columns: int, *, card: str, levels: int, route: str,
                suite: str, cards: int, base_mib: Any, other_suites: bool = False,
                profile: str | None = None, capacity_mib: float | None = None) -> dict[str, Any] | None:
    """The whole-run floor of one card's peak, from the runs of this physics in this card regime, or None.

    The WOOF route keeps its rule: this card's completed runs at this level count (its device_admission row is a
    per-card WOOF measurement).  Every other route is priced only from its own runs, since device_admission's rows
    are WOOF rows: this suite's (with ``other_suites``, the route's other suites' when this one has none), at this
    level count (else scaled by levels), on this card when it has two or more, else on every card; one card
    (``cards`` 1) and same-host ranks (``cards`` > 1, a rank's peak against its columns ``cells / cards``) are
    separate regimes, since a rank holds more than one card's loop at the same columns.  The curve is the completed
    runs' curve, never below the curve through the completed runs and the out-of-memory floors.  A ``profile`` (an
    experiment profile over the suite, e.g. hrrr-like-aerosol) takes only the runs that carried it; ``None`` takes
    only the plain suite's runs.  With ``capacity_mib`` (the card memory the plan prices), each size takes the runs
    measured with the card memory nearest it from above, else the most below (``_by_capacity``)."""
    multi = int(cards) > 1
    rows = [row for row in observations
            if str(row.get("route", "arwen")) == route and row["status"] in CURVE_STATUSES
            and (int(row.get("cards", 1)) > 1) == multi]
    notes = []
    if route == "arwen":
        chosen = [row for row in rows if row["card"] == card and row["levels"] == levels
                  and row["status"] == "complete"]
    else:
        chosen = [row for row in rows if row.get("suite") == suite and row.get("profile") == profile]
        if not chosen and other_suites:
            chosen = rows
            if rows:
                notes.append(f"no whole run of {suite}: the {route} route's other suites")
        exact = [row for row in chosen if row["levels"] == levels]
        if not exact and chosen:
            notes.append(f"no whole run at {levels} levels: runs at other level counts scaled by levels")
        chosen = exact or chosen
        here = [row for row in chosen if row["card"] == card]
        if len(here) < 2 and len(here) < len(chosen):
            notes.append("fewer than two runs on this card: runs on every card")
        else:
            chosen = here
    if not chosen:
        return None
    points = _rank_rows(chosen, levels)
    capacities = sorted({row_capacity_mib(row) for row in points if row["status"] == "complete"})
    if capacity_mib is not None:
        before = len(points)
        points, capacities = _by_capacity(points, float(capacity_mib))
        if len(points) < before:
            notes.append(f"at each size the runs measured with the card memory nearest {float(capacity_mib):,.0f} MiB "
                         f"({', '.join(f'{value:,.0f}' for value in capacities)} MiB)")
    finished = [row for row in points if row["status"] == "complete"]
    below_shape = None
    if profile is not None and route != "arwen":
        # A profile with one run in this regime is shaped below it by its plain suite's curve (whole_run_curve).
        def below_shape(at_columns: int, run: Mapping[str, Any] | None) -> float | None:
            # ``run``: the profile's own run (its card and card memory); None: the plan's card and memory.
            plain = route_floor(observations, int(math.ceil(at_columns)), levels=levels, route=route, suite=suite,
                                cards=cards, base_mib=base_mib,
                                card=card if run is None else str(run["card"]),
                                capacity_mib=capacity_mib if run is None else row_capacity_mib(run))
            return None if plain is None else float(plain["curve_mib"])
    curves = [whole_run_curve(rows, columns, base_mib=base_mib, below_shape_mib=below_shape)
              for rows in (finished, points) if rows]
    curve = max(curves, key=lambda value: value["curve_mib"])
    curve["capacities_mib"] = capacities
    curve["regime"] = f"same-host ranks, {columns:,} columns per card" if multi else "one card"
    curve["physics"] = {"route": route, "suite": suite, **({"profile": profile} if profile is not None else {})}
    curve["out_of_memory_floors"] = [row.get("id") for row in points if row["status"] == "out_of_memory"]
    if notes:
        curve["basis"] += "; " + "; ".join(notes)
    return curve


def rank_excess(observations: Sequence[Mapping[str, Any]], *, route: str) -> dict[str, Any] | None:
    """The largest measured ratio of a same-host rank's peak to its own physics' one-card curve at its columns.

    For a suite no multi-card run has measured: its one-card curve at a rank's columns, times this ratio, so a rank
    is never priced as one card's loop (measured 2026-10-28: a 143,265-column rank peaked 1.23x the one-card curve)."""
    from . import device_admission as admission

    worst = None
    for row in observations:
        if (str(row.get("route", "arwen")) != route or int(row.get("cards", 1)) <= 1
                or row["status"] != "complete"):
            continue
        columns = int(math.ceil(float(row["cells"]) / int(row["cards"])))
        model = admission.model_for_card(admission.KNOWN_CARDS[str(row["card"])], "global", levels=int(row["levels"]))
        single = route_floor(observations, columns, card=row["card"], levels=int(row["levels"]), route=route,
                             suite=str(row.get("suite")), cards=1, profile=row.get("profile"),
                             capacity_mib=row_capacity_mib(row),
                             base_mib=lambda cells, model=model: model.predict_bytes(cells) / admission.MIB)
        if single is None or single["curve_mib"] <= 0:
            continue
        ratio = float(row["peak_process_vram_mib"]) / single["curve_mib"]
        if worst is None or ratio > worst["ratio"]:
            worst = {"ratio": round(ratio, 4), "observation": row.get("id"), "columns": columns,
                     "one_card_curve_mib": single["curve_mib"]}
    return worst


def profile_excess(observations: Sequence[Mapping[str, Any]], *, route: str, suite: str,
                   profile: str) -> dict[str, Any] | None:
    """The largest measured ratio of a run of ``profile`` to its plain suite's curve at the same columns and card
    regime: what a profile with no run in some regime is priced at over its plain suite there.

    Only like for like where it can be: a profile run is set against plain runs measured with the same card memory
    (``row_capacity_mib``), since a roomy card's peak carries kept idle pool sets and free-sized tiles a tight card's
    does not (V8X6000 2026-10-29: 96 GB aerosol rows against 32 GB plain rows read 1.18x).  With no such pair, the
    ratios across card sizes, labelled."""
    from . import device_admission as admission

    worst = like = None
    for row in observations:
        if (str(row.get("route", "arwen")) != route or row.get("suite") != suite or row.get("profile") != profile
                or row["status"] != "complete"):
            continue
        cards = max(1, int(row.get("cards", 1)))
        columns = int(math.ceil(float(row["cells"]) / cards))
        capacity = row_capacity_mib(row)
        model = admission.model_for_card(admission.KNOWN_CARDS[str(row["card"])], "global", levels=int(row["levels"]))
        plain = route_floor(observations, columns, card=row["card"], levels=int(row["levels"]), route=route,
                            suite=suite, cards=cards, capacity_mib=capacity,
                            base_mib=lambda cells, model=model: model.predict_bytes(cells) / admission.MIB)
        if plain is None or plain["curve_mib"] <= 0:
            continue
        ratio = float(row["peak_process_vram_mib"]) / plain["curve_mib"]
        entry = {"ratio": round(ratio, 4), "observation": row.get("id"), "columns": columns, "cards": cards,
                 "plain_curve_mib": plain["curve_mib"], "card_capacity_mib": capacity,
                 "plain_capacities_mib": plain["capacities_mib"]}
        if plain["capacities_mib"] == [capacity] and (like is None or ratio > like["ratio"]):
            like = entry
        if worst is None or ratio > worst["ratio"]:
            worst = entry
    if like is not None:
        return like
    if worst is not None:
        worst["basis"] = "no profile run on the card memory of a plain run: the ratio across card sizes"
    return worst


#: THE BREAKAGE THE RUNTIME KEYS PREVENT (hex 0.4.10 review, V8X6000 2026-10-29): the runtime took only the one-card
#: rows of global-runs' measured_seconds_per_forecast_hour, all ArWen-route x1 globes at the YSU arms' 300 s step, for
#: every route and card count.  The r700 96 -> 3 km zoom globe on 2 RTX PRO 6000s (MPAS route, the 0.4.10 default) was
#: priced 626.66 s for 1 h against 173.4 s measured (+261 %), and 2 and 4 cards both printed 7,517 s for 12 h.  A rate
#: now comes only from a whole run of the same route on the same card at the same card count (that suite and profile's
#: own runs first); with none, the plan says so instead of borrowing another route's or card's rate.


def runtime_rows(observations: Sequence[Mapping[str, Any]], table: Mapping[str, Any], *, card: str,
                 card_name: str) -> list[dict[str, Any]]:
    """Every measured whole-run forecast rate, keyed by route, suite, profile, card and card count.

    From the whole-run observations (fast numerics, complete) that carry a forecast-stage wall: ``runtime`` (the
    steady per-hour rate and the whole run's wall, so startup is measured apart) or ``forecast_wall_seconds`` over
    ``hours`` (startup not separated from the hourly rate).  And from global-runs' measured_seconds_per_forecast_hour
    rows (all-in rates per hour of the x1 globes) on this card, the one their ``card_match`` names."""
    from . import device_admission as admission
    from .global_door import row_step

    rows: list[dict[str, Any]] = []
    for row in observations:
        if row.get("status") != "complete" or str(row.get("numerics", "fast")) != "fast":
            continue
        measured = row.get("runtime")
        if measured is not None:
            wall, hours = float(measured["whole_run_seconds"]), float(measured["hours"])
            steady = measured.get("seconds_per_forecast_hour")
            evidence = measured.get("evidence", row.get("evidence"))
        elif row.get("forecast_wall_seconds") is not None:
            wall, hours, steady = float(row["forecast_wall_seconds"]), float(row["hours"]), None
            evidence = row.get("evidence")
        else:
            continue
        rate = wall / hours if steady is None else float(steady)
        rows.append({"id": row.get("id"), "route": str(row.get("route", "arwen")), "suite": row.get("suite"),
                     "profile": row.get("profile"), "card": str(row["card"]), "cards": int(row.get("cards", 1)),
                     "cells": float(row["cells"]), "levels": int(row["levels"]),
                     "dt_seconds": float(row["dt_seconds"]), "hours": hours, "wall_seconds": wall,
                     "seconds_per_forecast_hour": rate,
                     "startup_seconds": 0.0 if steady is None else max(0.0, wall - hours * rate),
                     "startup_separated": steady is not None, "evidence": evidence})
    for row in table.get("measured_seconds_per_forecast_hour", []):
        if str(row.get("card_match", row["card"])).lower() not in card_name.lower():
            continue
        mesh = table["meshes"][row["mesh"]]
        physics = table["physics"][row.get("physics", "arwen")]
        # The table's rates were measured at the physics arm's capped step: keep that clock even when a newer engine
        # retires the cap for new forecasts.
        rows.append({"id": f"global-runs {row['mesh']} {row['cards']}-card", "route": str(row.get("route", "arwen")),
                     "suite": row.get("suite"), "profile": None, "card": card, "cards": int(row["cards"]),
                     "cells": float(mesh["cells"]), "levels": admission.MODEL_VERTICAL_LEVELS,
                     "dt_seconds": float(row_step(mesh, physics, engine=frozenset())["dt_seconds"]), "hours": None,
                     "wall_seconds": None, "seconds_per_forecast_hour": float(row["seconds"]), "startup_seconds": 0.0,
                     "startup_separated": False, "evidence": row.get("basis")})
    return rows


def runtime_estimate(rows: Sequence[Mapping[str, Any]], *, cells: float, levels: int, dt_seconds: float,
                     hours: float, route: str, suite: str, profile: str | None, card: str,
                     cards: int) -> dict[str, Any]:
    """The forecast stage's wall for ``hours``, from the nearest-sized measured run of this route on this card at this
    card count: this suite and profile's runs, else the suite's runs with another profile, else the route's other
    suites (named in the basis).  Startup plus that run's hourly rate scaled by cell-steps per card (cells, levels,
    steps).  None, saying so, when no run of this route on this card at this count was measured.

    Startup is the run's own when it was measured apart from the steady hour.  A run measured only as one whole
    stage (a 1 h run is mostly startup) takes the startup of this route's nearest card count that measured it apart:
    host work (imports, kernel pack, init read, partition, the cold first step), measured 77-78 s at 2, 4 and 8 cards
    alike on VRFS.  Without one its whole stage is spread over its hours, which prices a longer run high."""
    cards = int(cards)
    here = [row for row in rows if row["route"] == route and row["card"] == card and row["cards"] == cards]
    named = f"{route} route ({suite}{' + ' + profile if profile else ''}) on {cards} x {card}"
    if not here:
        return {"seconds": None, "basis": f"no measured rate for this route/card: no whole run of the {named}"}
    tiers = (([row for row in here if row["suite"] == suite and row["profile"] == profile], None),
             ([row for row in here if row["suite"] == suite],
              f"no run of {suite}{' + ' + profile if profile else ''} here: that suite's runs with another profile"),
             (here, f"no run of {suite} here: the {route} route's other suites"))
    chosen, note = next(tier for tier in tiers if tier[0])
    work = float(cells) * int(levels)
    reference = dict(min(chosen, key=lambda row: abs(math.log(work / (row["cells"] * row["levels"])))))
    basis = (f"EXTRAPOLATED cell-steps per card from a measured whole run of the {named} ({reference['id']}); "
             "the forecast stage only (excludes mesh, static, init and products); geometry, physics cadence, cache "
             "and contention can change runtime")
    startup, startup_from = reference["startup_seconds"], None
    if reference["startup_separated"]:
        basis += f"; startup {startup:.1f} s measured apart from the steady hourly rate"
    elif reference["hours"] is not None:
        measured = [row for row in rows if row["route"] == route and row["startup_separated"]
                    and row["startup_seconds"] < reference["wall_seconds"]]
        if measured:
            donor = min(measured, key=lambda row: (abs(row["cards"] - cards), row["suite"] != suite,
                                                   row["card"] != card))
            startup, startup_from = donor["startup_seconds"], donor["id"]
            reference["seconds_per_forecast_hour"] = (reference["wall_seconds"] - startup) / reference["hours"]
            basis += (f"; that run's {reference['hours']:g} h stage less a {startup:.1f} s startup measured apart on "
                      f"{donor['cards']} x {donor['card']} ({donor['id']})")
        else:
            basis += (f"; startup not separated: the rate is that run's whole forecast stage over its "
                      f"{reference['hours']:g} h, so a longer run is priced high")
    else:
        basis += "; the rate is all-in per forecast hour"
    if note:
        basis += "; " + note
    factor = (float(cells) / reference["cells"]) * (int(levels) / reference["levels"]) * (
        reference["dt_seconds"] / float(dt_seconds))
    stepping = reference["seconds_per_forecast_hour"] * float(hours) * factor
    return {"seconds": round(startup + stepping, 3), "basis": basis, "reference": reference,
            "cell_step_factor": factor, "startup_seconds": round(startup, 3), "startup_from": startup_from,
            "stepping_seconds": round(stepping, 3)}


def predict(cells: float, *, card: Mapping[str, Any], levels: int,
            dt_seconds: float, hours: float, budget_mib: float | None = None,
            table: Mapping[str, Any] | None = None,
            observations: Sequence[Mapping[str, Any]] | None = None,
            suite: str | None = None, route: str | None = None, cards: int = 1,
            profile: str | None = None, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One card's peak and verdict for a globe of ``cells`` run on ``cards`` same-host cards of this kind, on the
    physics named (``suite``/``route``/``profile``; none named is the hex default, MPAS physics, with the
    recipe_rules row a door's ``context`` matches)."""
    from . import device_admission as admission
    from .global_door import load_table

    if not all(math.isfinite(float(v)) and float(v) > 0
               for v in (cells, levels, dt_seconds, hours)):
        raise MpasPortError("card-fit requires finite positive cells, levels, timestep and hours")
    budget = float(card["nameplate_mib"] if budget_mib is None else budget_mib)
    if not math.isfinite(budget) or budget <= 0:
        raise MpasPortError("card-fit memory budget must be finite and positive")
    if int(cards) != cards or int(cards) < 1:
        raise MpasPortError(f"card-fit prices one or more cards, got {cards!r}")
    cards = int(cards)
    route, suite, profile = run_physics(suite, route, profile=profile, context=context)
    count = math.ceil(cells)
    columns = math.ceil(count / cards)
    card_profile = admission.KNOWN_CARDS[str(card["admission_card"])]
    model = admission.model_for_card(card_profile, configuration="global", levels=int(levels))
    if cards > 1:
        model = admission.samehost_rank_model(model, ranks=cards)
    base_peak = model.predict_bytes(columns) / admission.MIB
    required = model.required_bytes(columns) / admission.MIB
    margin = required - base_peak
    if observations is None:
        observations = shipped_observations()
    calibration = None
    predicted_peak = base_peak
    base_mib = lambda cells: model.predict_bytes(cells) / admission.MIB  # noqa: E731
    where = dict(card=str(card["admission_card"]), levels=int(levels), route=route, suite=suite, base_mib=base_mib,
                 capacity_mib=budget)

    def physics_curve(chosen_profile: str | None) -> dict[str, Any] | None:
        # This suite (and profile) in this regime; then (ranks) its one-card curve times the largest measured rank
        # excess; then, for a plain suite only, the route's other suites.  A suite is never priced from another
        # suite while it has runs of its own.
        found = route_floor(observations, columns, cards=cards, profile=chosen_profile, **where)
        if found is None and cards > 1 and route != "arwen":
            excess = rank_excess(observations, route=route)
            single = route_floor(observations, columns, cards=1, profile=chosen_profile, **where)
            if excess is not None and single is not None:
                scaled = single["curve_mib"] * max(1.0, excess["ratio"])
                found = dict(single, curve_mib=round(scaled, 3),
                             floor_mib=round(scaled * (1.0 + WHOLE_RUN_MARGIN), 3),
                             regime=f"same-host ranks, {columns:,} columns per card", rank_excess=excess,
                             basis=(single["basis"] + f"; no multi-card run of this physics: its one-card curve at "
                                    f"a rank's columns times the largest measured rank excess ({excess['ratio']:.3f})"))
        if found is None and route != "arwen" and chosen_profile is None:
            found = route_floor(observations, columns, cards=cards, other_suites=True, **where)
        return found

    curve = physics_curve(None)
    if profile is not None and route != "arwen":
        # A profile only adds fields to its suite (aerosol Thompson: nc/nifa/nwfa and its workspace), so it is never
        # priced under the plain suite; with runs of its own it is priced from them where they are higher.
        own = physics_curve(profile)
        if own is None:
            excess = profile_excess(observations, route=route, suite=suite, profile=profile)
            if curve is not None and excess is not None:
                scaled = curve["curve_mib"] * max(1.0, excess["ratio"])
                curve = dict(curve, curve_mib=round(scaled, 3), floor_mib=round(scaled * (1.0 + WHOLE_RUN_MARGIN), 3),
                             physics={**curve["physics"], "profile": profile}, profile_excess=excess,
                             basis=(curve["basis"] + f"; no run of profile {profile} in this regime: the plain "
                                    f"suite's curve times its largest measured excess ({excess['ratio']:.3f})"))
            elif curve is not None:
                curve = dict(curve, physics={**curve["physics"], "profile": profile},
                             basis=curve["basis"] + f"; profile {profile} has no whole run: the plain suite's runs")
        elif curve is None or own["floor_mib"] >= curve["floor_mib"]:
            curve = own
        else:
            curve = dict(curve, physics={**curve["physics"], "profile": profile},
                         basis=curve["basis"] + f"; the plain suite's runs, above profile {profile}'s own here")
    # Never lower device_admission: the curve is a floor on its peak.
    if curve is not None and curve["floor_mib"] > predicted_peak:
        predicted_peak = curve["floor_mib"]
        calibration = curve
    required = predicted_peak + margin
    table = load_table() if table is None else table
    runtime = runtime_estimate(
        runtime_rows(observations, table, card=str(card["admission_card"]), card_name=card_profile.name),
        cells=count, levels=int(levels), dt_seconds=dt_seconds, hours=hours, route=route, suite=suite,
        profile=profile, card=str(card["admission_card"]), cards=cards)
    return {
        "schema": "gpuwm-hex.card-fit/v1", "cells": count, "levels": levels,
        "dt_seconds": dt_seconds, "hours": hours, "cards": cards, "columns_per_card": columns,
        "physics": {"route": route, "suite": suite, **({"profile": profile} if profile is not None else {})},
        "predicted_peak_mib": round(predicted_peak, 3),
        "base_predicted_peak_mib": round(base_peak, 3),
        "whole_run_floor": calibration,
        "required_free_mib": round(required, 3), "budget_mib": budget,
        "fits": required <= budget, "model": model.as_dict(), "runtime": runtime,
        "scope": ("per card, at the card count given, from the whole runs of the physics named; memory terms "
                  "outside the measured levels and sizes are derived; launch rechecks actual free memory"),
    }


#: The card counts a plan tries when one card does not hold the run (one host's same-host ranks).
CARD_COUNTS = (1, 2, 3, 4, 5, 6, 7, 8)


def by_card_count(cells: float, **kwargs: Any) -> dict[str, Any]:
    """The per-card price at every count in :data:`CARD_COUNTS`, and the fewest cards that hold the run."""
    rows = []
    for count in CARD_COUNTS:
        fit = predict(cells, cards=count, **kwargs)
        rows.append({"cards": count, "columns_per_card": fit["columns_per_card"],
                     "predicted_peak_mib": fit["predicted_peak_mib"],
                     "required_free_mib": fit["required_free_mib"], "fits": fit["fits"]})
    fewest = next((row["cards"] for row in rows if row["fits"]), None)
    return {"by_cards": rows, "fewest_cards_that_fit": fewest}


def comparison(prediction: Mapping[str, Any], measured: Mapping[str, Any]) -> dict[str, Any]:
    """A per-card prediction against the stage's largest card.

    THE BREAKAGE IT PREVENTS (hex 0.4.10 review, V8X6000 2026-10-29): the stage's peak is summed over every card its
    processes hold, so a 2-card run's receipt compared a 28,722 MiB per-card prediction with 60,722 MiB (both cards)
    and said the plan was 52.7 % under.  The error is taken against the largest card (``peak_card_vram_mib``), the
    stage total kept beside it."""
    total = measured.get("peak_process_vram_mib")
    peak = measured.get("peak_card_vram_mib", total)
    wall = measured["wall_seconds"]
    expected = prediction["runtime"]["seconds"]
    return {
        "predicted_peak_mib": prediction["predicted_peak_mib"],
        "measured_peak_process_vram_mib": total,
        "measured_peak_card_vram_mib": peak,
        "memory_error_percent": None if peak is None or peak <= 0 else
            round(100 * (prediction["predicted_peak_mib"] / peak - 1), 3),
        "predicted_forecast_seconds": expected, "measured_forecast_seconds": wall,
        "runtime_error_percent": None if expected is None else round(100 * (expected / wall - 1), 3),
        "instrument": "1 s compute-process samples of the stage process group, per card (the largest card) and summed; subsecond peaks can be missed; wall includes process startup and JIT",
    }


def _process_group(group: int) -> tuple[set[int], float, int]:
    pids: set[int] = set()
    ticks = 0.0
    threads = 0
    hz = os.sysconf("SC_CLK_TCK")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group:
                pids.add(int(entry.name))
                ticks += (int(fields[11]) + int(fields[12])) / hz
                threads += int(fields[17])
        except (OSError, ValueError, IndexError):
            continue
    return pids, ticks, threads


def _gpu_memory_by_card(pids: set[int]) -> dict[str, float] | None:
    """MiB the processes ``pids`` hold on each card (by GPU UUID), or None when they hold none or nothing answers."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False)
        if result.returncode:
            return None
        cards: dict[str, float] = {}
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 3 and int(parts[1]) in pids:
                cards[parts[0]] = cards.get(parts[0], 0.0) + float(parts[2])
        return cards or None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None



class BackgroundStage:
    """A door stage started beside the foreground ones (the zoom globe's source fetch and decode beside its mesh
    and static, U-H6).

    No per-second sampling and no signal handlers: those belong to the foreground :func:`run_stage`. The child is
    reaped only in :meth:`finish`, so a foreground stage's RUSAGE_CHILDREN delta never counts this one's CPU; its own
    CPU comes from wait4, and its wall ends when it exits (waitid WNOWAIT), not when the door gets round to it."""

    def __init__(self, argv: Sequence[str], *, name: str, directory: Path,
                 env: Mapping[str, str] | None = None):
        import datetime
        import threading

        directory.mkdir(parents=True, exist_ok=True)
        self.name, self.argv = name, [str(a) for a in argv]
        self.log = directory / f"{name}.log"
        self._output = self.log.open("w")
        self._output.write("argv: " + repr(self.argv) + "\n")
        self._output.flush()
        self.started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._started = time.monotonic()
        self._ended: float | None = None
        self.process = subprocess.Popen(self.argv, stdout=self._output, stderr=subprocess.STDOUT,
                                        env=None if env is None else dict(env), start_new_session=True)
        self._watcher = threading.Thread(target=self._watch, name=f"stage-{name}", daemon=True)
        self._watcher.start()
        self._result: dict[str, Any] | None = None

    def _watch(self) -> None:
        try:
            os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOWAIT)
        except ChildProcessError:
            pass
        self._ended = time.monotonic()

    def finish(self) -> dict[str, Any]:
        """Wait for the stage, reap it and return its row (the shape :func:`run_stage` returns)."""
        if self._result is not None:
            return self._result
        self._watcher.join()
        _pid, status, usage = os.wait4(self.process.pid, 0)
        code = os.waitstatus_to_exitcode(status)
        self.process.returncode = code
        self._output.close()
        wall = (self._ended or time.monotonic()) - self._started
        cpu = usage.ru_utime + usage.ru_stime
        self._result = {
            "stage": self.name, "argv": self.argv, "returncode": code, "started_utc": self.started_utc,
            "wall_seconds": round(wall, 3), "cpu_seconds": round(cpu, 3),
            "average_cpu_cores": round(cpu / max(wall, .001), 3), "peak_process_vram_mib": None,
            "log": str(self.log), "background": True}
        return self._result

    def cancel(self) -> None:
        """Stop the stage (the door failed before it needed it) and reap it."""
        if self._result is not None:
            return
        if self._ended is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.finish()


def start_stage(argv: Sequence[str], *, name: str, directory: Path,
                env: Mapping[str, str] | None = None) -> BackgroundStage:
    """Start a door stage beside the foreground ones; :meth:`BackgroundStage.finish` joins it."""
    return BackgroundStage(argv, name=name, directory=directory, env=env)


def run_stage(argv: Sequence[str], *, name: str, directory: Path,
              env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run an existing door with logs; never substitute whole-card usage."""
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / f"{name}.log"
    samples = directory / f"{name}.samples.csv"
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak = None
    max_threads = 0
    last_ticks = 0.0
    last_time = started
    single_seconds = 0.0
    longest_single = 0.0
    card_peaks: dict[str, float] = {}
    with log.open("w") as output, samples.open("w", newline="") as stream:
        output.write("argv: " + repr(list(argv)) + "\n")
        output.flush()
        writer = csv.writer(stream)
        writer.writerow(["elapsed_seconds", "cpu_cores", "threads", "process_vram_mib", "pids"])
        process = subprocess.Popen(list(argv), stdout=output, stderr=subprocess.STDOUT,
                                   env=None if env is None else dict(env), start_new_session=True)
        def interrupted(signum, _frame):
            raise InterruptedError(f"stage {name} interrupted by signal {signum}")

        previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            while True:
                now = time.monotonic()
                pids, ticks, threads = _process_group(process.pid)
                by_card = _gpu_memory_by_card(pids)
                gpu = sum(by_card.values()) if by_card else None
                if gpu is not None:
                    peak = max(peak or 0, gpu)
                    for uuid, used in by_card.items():
                        card_peaks[uuid] = max(card_peaks.get(uuid, 0.0), used)
                cores = max(0, ticks - last_ticks) / max(now - last_time, .001)
                if .7 <= cores <= 1.3 and gpu is None:
                    single_seconds += now - last_time
                else:
                    longest_single = max(longest_single, single_seconds)
                    single_seconds = 0
                max_threads = max(max_threads, threads)
                writer.writerow([round(now - started, 3), round(cores, 3), threads, gpu, " ".join(map(str, sorted(pids)))])
                stream.flush()
                last_ticks, last_time = ticks, now
                if process.poll() is not None:
                    break
                time.sleep(1)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall = time.monotonic() - started
    cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    return {
        "stage": name, "argv": list(argv), "returncode": process.returncode,
        "wall_seconds": round(wall, 3), "cpu_seconds": round(cpu, 3),
        "average_cpu_cores": round(cpu / wall, 3), "maximum_sampled_threads": max_threads,
        "peak_process_vram_mib": peak, "peak_card_vram_mib": max(card_peaks.values()) if card_peaks else None,
        "peak_vram_mib_by_card": card_peaks, "log": str(log), "samples": str(samples),
        "longest_single_core_cpu_only_seconds": round(max(longest_single, single_seconds), 3),
        "parallelism_environment": {key: (env or os.environ).get(key) for key in
                                    ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
