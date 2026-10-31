"""``woof hex mesh-plan`` -- ask a resolution spec every question before building.

WHY THIS DOOR EXISTS.  ``mesh-check`` validates a mesh that already exists.
Until this leg there was nothing to ask about a mesh that does NOT exist yet,
so the only way to find out whether a spec was buildable was to build it --
and ``rw_mpas_mesh --dry-run``, the thing that looks like the answer, sizes
without applying the build's gates.  Measured 2026-08-28 on this desktop:
one design sized clean and was refused 87 ms later by the build's pre-run
arithmetic; a second sized clean, cleared that gate, relaxed 217,621 cells
over **1,251 s** and was then refused on the 200 m shortest-dual-edge floor
at edge 562175 (36.8 m over an 886 m dcEdge), with no grid written.  The
report that opened this lane measured the same pair at 711 s on another box.

``--point LAT,LON`` (:mod:`woof.hex.mesh_point`) writes the spec itself --
the registered 937.5 m ladder moved to the point -- prices its CULL against
the named card on the one admission surface, and with ``--generate``
builds, admits, registers (a runtime row beside the mesh) and culls it.

This leg costs about a fifth of a second and answers three things:

* what the spec costs -- cells, and the device footprint when a card is named;
* every gate the build applies that a SPEC decides, applied here, so a spec
  the build would refuse is refused here first, with the remedy in numbers;
* every gate the build applies that a spec does NOT decide, named, with the
  quantities that decide it and the earliest point at which it can be known.

The third list is not an apology.  A sizing path that silently omits a gate
is how a user spends twenty minutes to learn something; a sizing path that
says which gate it cannot evaluate, and why, spends nothing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import MpasPortError


def _load(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise MpasPortError(
            f"--spec {path} cannot be read: {error}. A resolution spec is the "
            "one input this leg has; there is no default that would name a "
            "file that exists"
        ) from error
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as error:
        raise MpasPortError(f"--spec {path} is not valid JSON: {error}") from error
    if not isinstance(spec, dict) or "background_km" not in spec:
        raise MpasPortError(
            f"--spec {path} carries no background_km, so it is not a "
            "resolution spec. The shape is "
            '{"background_km": KM, "regions": [{"shape": {...}, '
            '"spacing_km": KM, "transition_cells": N}]}'
        )
    return spec


def _band(edge: Mapping[str, Any]) -> str:
    pair = edge.get("predicted_shortest_dual_edge_m")
    if not pair:
        return "not predicted"
    low, high = pair
    return f"{low:.1f} to {high:.1f} m"


def _nearest(edge: Mapping[str, Any], count: int = 2) -> list[str]:
    """The measured rows closest in finest spacing, so a verdict has evidence."""

    finest = edge.get("finest_requested_spacing_km") or 0.0
    samples = edge.get("measured_samples") or []
    if finest <= 0.0 or not samples:
        return []
    ordered = sorted(
        samples, key=lambda row: abs(math.log(row["finest_km"] / finest))
    )
    return [
        f"      measured          {row['label']}: {row['finest_km']:g} km finest, "
        f"{row['min_dv_edge_m']:.1f} m shortest dual edge, "
        f"{'emitted' if row['emitted'] else 'REFUSED'}"
        for row in ordered[:count]
    ]


def _report(spec: Mapping[str, Any], receipt: Mapping[str, Any]) -> list[str]:
    gates = receipt.get("gates_applied_by_hexcore") or {}
    band = gates.get("transition_band") or {}
    edge = gates.get("short_dual_edge_floor") or {}
    name = spec.get("name") or "(unnamed spec)"
    lines = [
        f"mesh-plan: {name}",
        f"  cells            {receipt.get('predicted_cells', float('nan')):,.0f} "
        "(the generator's own sizing integral)",
    ]
    if receipt.get("card"):
        lines.append(
            f"  footprint        {receipt.get('footprint_mib'):,.1f} MiB on "
            f"{receipt['card']}"
        )
    else:
        lines.append(
            "  footprint        not sized: no --card, and the fixed term is a "
            "property of the part"
        )
    lines.extend([
        "",
        "  GATES THIS SPEC DECIDES, applied here:",
        f"    transition band   PASS  {band.get('steepest_gradient_percent_per_cell', float('nan')):.4f} "
        f"%/cell -> {band.get('band_cells', float('nan')):.2f} cells "
        f"(floor {band.get('band_cells_floor', float('nan')):.0f}, ceiling "
        f"{band.get('gradient_percent_per_cell_ceiling', float('nan')):.4f} %/cell)",
        "",
        "  GATES ONLY A BUILD DECIDES, named rather than skipped:",
        f"    shortest dual edge  {str(edge.get('verdict', 'unknown')).upper()}",
        f"      limit             {edge.get('limit_m', float('nan')):.0f} m "
        f"({edge.get('gate', 'unknown gate')})",
        f"      predicted         {_band(edge)} at a "
        f"{edge.get('finest_requested_spacing_km', float('nan')):g} km finest "
        f"spacing, over {edge.get('prediction_samples', 0)} measured meshes",
        f"      why not decidable {edge.get('why_not', '')}",
        f"      basis             {edge.get('prediction_basis', '')}",
    ])
    lines.extend(_nearest(edge))
    lines.extend([
        f"      earliest known    {edge.get('earliest_this_can_be_known', '')}",
        f"      to clear it       {edge.get('if_it_refuses_change', '')}",
    ])
    return lines


def run_mesh_plan(arguments: argparse.Namespace) -> int:
    from .swath.sizing import dry_run

    if getattr(arguments, "global_mesh", False):
        return run_global_mesh(arguments)
    if getattr(arguments, "run", False):
        raise MpasPortError("--run requires --global-mesh --generate")

    drawn = (getattr(arguments, "shape", None) is not None
             or getattr(arguments, "ellipse", None) is not None)
    if (getattr(arguments, "point", None) is not None or drawn) and arguments.spec is not None:
        raise MpasPortError(
            "--spec and --point (or --shape/--ellipse) were both given; a plan prices one "
            "request. --spec hands the generator an authored spec unchanged, --point "
            "and a drawn core write the ladder spec for you"
        )
    if getattr(arguments, "point", None) is not None or drawn:
        from .mesh_point import run_point

        return run_point(arguments)
    if arguments.spec is None:
        raise MpasPortError(
            "neither --spec nor --point was given; a plan needs a request. "
            "Pass --spec SPEC.json for an authored resolution spec, or "
            "--point LAT,LON [--fine-dx-m M --radius-km KM --card KEY] for a "
            "fine core at a point"
        )
    spec = _load(arguments.spec)
    receipt = dry_run(
        spec,
        engine=arguments.mesh_exe,
        card=arguments.card,
        vram_gib=arguments.vram_gib,
    )
    if arguments.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print("\n".join(_report(spec, receipt)))
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return 0


def add_mesh_plan_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "mesh-plan",
        help="price a resolution spec and apply the gates its build applies",
    )
    parser.add_argument(
        "--spec", type=Path, default=None,
        help="the resolution spec JSON rw_mpas_mesh would be given "
             "(or --point, below)",
    )
    parser.add_argument(
        "--card", default=None,
        help="a measured card key, so the receipt carries a device footprint",
    )
    parser.add_argument(
        "--vram-gib", type=float, default=None,
        help="a device budget instead of the named card's own memory",
    )
    parser.add_argument("--mesh-exe", type=Path, default=None)
    parser.add_argument(
        "--out", type=Path, default=None, help="write the receipt here as well"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="print the whole receipt instead of the report",
    )
    from .mesh_point import add_point_arguments

    add_point_arguments(parser)
    from woof.output_formats import add_arguments
    add_arguments(parser)
    global_group = parser.add_argument_group("generated global variable-resolution forecast")
    from .products_door import add_label_arguments

    add_label_arguments(global_group)
    global_group.add_argument("--global-mesh", action="store_true",
                              help="price/build the whole global mesh, without a limited-area cull")
    global_group.add_argument("--template-only", action="store_true",
                              help="with --global-mesh --generate: build (or fetch) only the point's cap-family "
                                   "rotation template, staged into WOOF_HEX_PREP_STORE_OUTBOX for an image bake; "
                                   "no static, no run")
    global_group.add_argument("--run", action="store_true",
                              help="with --global-mesh --generate: init-source, forecast and Rust products")
    global_group.add_argument("--cards", type=int, default=None, metavar="N",
                              help="same-host cards the forecast runs on, one rank per card (default: the "
                                   "--forecast-arg=--samehost-devices count, else 1); the plan prices one card's "
                                   "share and lists every count up to 8; with --run it passes --samehost-devices "
                                   "0..N-1 when --forecast-arg names none")
    global_group.add_argument("--source", default=None, help="an init-source table row")
    global_group.add_argument("--cycle", default=None, help="YYYY-MM-DDTHH")
    global_group.add_argument("--lead-hours", type=int, default=0, metavar="H",
                              help="with --run: start from this forecast hour of --cycle (a forecast source's own "
                                   "cadence, e.g. GFS 2026-10-23T18 --lead-hours 348); the run's start is cycle + H")
    global_group.add_argument("--from-last-hour", action="store_true",
                              help="with --run: start from the source's LAST forecast hour for --cycle and run --hours "
                                   "on (a free run of our own model past the source; the receipt says so)")
    global_group.add_argument("--hours", type=float, default=12.0)
    global_group.add_argument("--history-every-minutes", type=int, default=180)
    global_group.add_argument("--staged", type=Path, action="append", default=[])
    global_group.add_argument("--source-table", type=Path, default=None)
    global_group.add_argument("--member", type=int, default=None)
    global_group.add_argument("--jobs", type=int, default=8)
    global_group.add_argument("--region", action="append", default=[],
                              help="products view NAME=region:CLAT,CLON,W_KM,H_KM")
    global_group.add_argument("--forecast-arg", action="append", default=[], metavar="ARG",
                              help="with --run: a flag for the forecast door, e.g. "
                                   "--forecast-arg=--physics-suite=convection_permitting; repeat for several")
    parser.set_defaults(handler=run_mesh_plan)


__all__ = ["add_mesh_plan_parser", "run_mesh_plan"]


#: THE BREAKAGE THIS PREVENTS (U-H6, production run_fde6921eaf18b793, 2026-10-26; owner ruling 2026-10-23: "zoom
#: globes use premade circular VR meshes rotated to the user's point, never fresh mesh generation on the start
#: path"): a zoom globe (120 -> 15 km, 2,000 km core at 40N 97W) on a new box generated its mesh with rw_mpas_mesh
#: (16.6 s) because the library holds exact points only. A concentric cap family is the same mesh wherever its
#: centre is, so the door rotates the family's template (centred at the moving door's TEMPLATE_CENTER and keyed in
#: the prep store by its spec alone, so the bake's library runs stage it) to the point, as storm-following start
#: members already do. An exact-point library entry still wins; any other spec (several centres, a non-cap shape,
#: a uniform globe) and a point near the template's antipode generate as before.
ROTATION_TEMPLATE_NAME = "zoom-globe rotation template"


def concentric_cap_center(spec: Mapping[str, Any]) -> tuple[float, float] | None:
    """The one centre every refinement region of SPEC is a cap about, else None (rotation would not preserve it)."""
    regions = spec.get("regions") or []
    centres = set()
    for region in regions:
        shape = region.get("shape") or {}
        if shape.get("kind") != "cap" or set(shape) - {"kind", "center_deg", "radius_km"}:
            return None
        lat, lon = (float(value) for value in shape["center_deg"])
        centres.add((lat, lon))
    return next(iter(centres)) if len(centres) == 1 else None


def rotation_template_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """SPEC's cap family about the template centre, named for the family alone (no point): the prep-store key."""
    from .moving_members import TEMPLATE_CENTER

    template = json.loads(json.dumps(spec))
    template["name"] = ROTATION_TEMPLATE_NAME
    for region in template["regions"]:
        region["shape"]["center_deg"] = [float(TEMPLATE_CENTER[0]), float(TEMPLATE_CENTER[1])]
    return template


def samehost_cards(flags: list[str]) -> int | None:
    """The card count a forecast-flag list's ``--samehost-devices D0,D1,...`` names (the last one wins), or None."""
    count = None
    for index, item in enumerate(flags):
        name, equals, value = str(item).partition("=")
        if name == "--samehost-devices":
            if not equals:
                value = str(flags[index + 1]) if index + 1 < len(flags) else ""
            count = len([device for device in value.split(",") if device.strip()])
    return count


def plan_cards(arguments: argparse.Namespace, flags: list[str]) -> int:
    """The cards a globe plan prices and runs on: ``--cards``, else the forecast's ``--samehost-devices``, else 1."""
    asked = getattr(arguments, "cards", None)
    named = samehost_cards(flags)
    if asked is not None and int(asked) < 1:
        raise MpasPortError(f"--cards is one or more cards, got {asked}")
    if asked is not None and named is not None and int(asked) != named:
        raise MpasPortError(f"--cards {asked} and --forecast-arg=--samehost-devices ({named} cards) disagree; "
                            "the plan prices the cards the forecast runs on, so give one")
    return int(asked if asked is not None else named or 1)


def _recipe_context(arguments: argparse.Namespace, regions: Sequence[Any]) -> dict[str, Any]:
    """The context physics_default.ensure picks the forecast's default recipe on (geometry, source row id, cold
    start), shared by the plan's price and the run's forecast flags so both name the same physics."""
    source = getattr(arguments, "source", None)
    if source:
        from .init_sources import InitSourceRefusal, load_table as source_rows, resolve_source as source_row

        try:
            source = source_row(source_rows(getattr(arguments, "source_table", None)), source)["id"]
        except InitSourceRefusal:
            pass  # --run refuses it by name; a plan alone keeps the name as given (no recipe row matches it)
    return {"geometry": "zoom-globe" if regions else "global", "source": source, "start": "cold"}


def _producer_suite(forecast_flags: Sequence[str]) -> str | None:
    """The suite the init names when it initializes through its own producer (hrrrv5, rrfs_*), else None."""
    from .physics_default import init_flags, init_producer
    flags = init_flags(forecast_flags)
    if "--physics-suite" not in flags:
        return None
    suite = flags[flags.index("--physics-suite") + 1]
    return suite if init_producer(suite) else None


def producer_init_geog(forecast_flags: Sequence[str], geog: Any) -> list[str]:
    """``--geog`` for init-source when the planned suite builds its own static (init_sources.producer_geog_flags).

    THE BREAKAGE THIS PREVENTS (zoom globe + hrrrv5 from IFS, 2026-10-31): the door's init-source passed no
    geography, so the RRFS producer refused at the init ("mesh.static.nc lacks landusef, soilf, lai12m ... Pass
    --grid and --geog"). Every other suite's init argv is unchanged.
    """
    return ["--geog", str(geog)] if _producer_suite(forecast_flags) is not None and geog is not None else []


def _producer_makes_drag_file(forecast_flags: Sequence[str], geog: Any) -> bool:
    """Whether the suite's init producer writes ugwp_oro_data.nc itself (its GSL generator, which needs GSL's
    topo_ugwp_30s and topo_ugwp_2.5m in the geography; cold_start_rrfs._gsl_orography)."""
    suite = _producer_suite(forecast_flags)
    if suite is None or geog is None:
        return False
    from .mpasphys.rrfs.cold_start_rrfs import producer_for_suite
    found = producer_for_suite(suite)
    if found is None:
        return False
    gwd = found[1].get("gwd") or {}
    return gwd.get("default") == "gsl-oro" and all((Path(geog) / name).exists() for name in gwd.get("geog", ()))


def orographic_statistics_job(forecast_flags: Sequence[str], mesh: str, grid: Path, geog: Any, root: Path):
    """Start hex's own drag statistics (woof.hex.ugwp_oro, Rust on every core) beside the init, or None.

    THE BREAKAGE THIS PREVENTS (zoom globe + hrrrv5, 2026-10-31): the zoom-globe door had no drag-statistics step,
    which the global, country and regional doors have; hrrrv5's GSL drag (bl_ugwp_gwdo) reads the twenty per-cell
    statistics, so the forecast door built them on the card's clock. None when the run's physics reads none, when
    the caller names its own --orographic-statistics, or when the producer's GSL generator makes the file (it wins,
    as on the regional door).
    """
    from .ugwp_oro import forecast_args_read_ugwp
    flags = [str(item) for item in forecast_flags]
    if (not forecast_args_read_ugwp(flags)
            or any(item.split("=", 1)[0] == "--orographic-statistics" for item in flags)
            or _producer_makes_drag_file(flags, geog)):
        return None
    from concurrent.futures import ThreadPoolExecutor
    from .ugwp_oro import prepare_for_forecast
    log = root / "logs" / "orographic-statistics.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    def note(message: Any) -> None:
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"{message}\n")

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ugwp-oro")
    future = pool.submit(prepare_for_forecast, ["--mesh", mesh, *flags], grid, geog, log=note)
    pool.shutdown(wait=False)
    return future


def finish_orographic_statistics(job, forecast_flags: Sequence[str], root: Path) -> dict[str, Any] | None:
    """The drag statistics file the forecast reads: the producer's (init/ugwp_oro_data.nc), else hex's own."""
    from .ugwp_oro import UgwpOroError, forecast_args_read_ugwp
    if job is not None:
        try:
            record = job.result()
        except UgwpOroError as error:
            raise MpasPortError(f"the GSL drag's orographic statistics could not be built: {error}") from None
        return None if record is None else {"path": str(record["out"]), "method": "hex-ugwp-oro",
                                            **{k: record[k] for k in ("sha256", "source", "seconds") if k in record}}
    flags = [str(item) for item in forecast_flags]
    made = root / "init" / "ugwp_oro_data.nc"
    if (forecast_args_read_ugwp(flags) and made.is_file()
            and not any(item.split("=", 1)[0] == "--orographic-statistics" for item in flags)):
        return {"path": str(made), "method": "gsl-oro"}
    return None


def run_global_mesh(arguments: argparse.Namespace) -> int:
    """A generated spec to a bound global forecast through existing doors.

    No per-event registry edits, source adapters, boundary inventing or
    replacement data processing. The Rust generator and static builder make
    the mesh; init-source supplies meteorology; forecast rechecks admissions.
    """
    import datetime
    import os
    import sys
    import time
    from . import card_fit, mesh_point, mesh_rows
    from .engines import MESH, STATIC, resolve
    from .swath.sizing import dry_run
    from .vertical_spec import VerticalSpec

    if getattr(arguments, "shape", None) is not None or getattr(arguments, "ellipse", None) is not None:
        raise MpasPortError(
            "--global-mesh takes a drawn core as a --spec (any shape row, any number of "
            "regions); --shape and --ellipse build the limited-area door's ladder")
    if arguments.point is not None and arguments.spec is not None:
        raise MpasPortError("--spec and --point are mutually exclusive")
    if arguments.spec is not None:
        spec = dict(_load(arguments.spec))
    elif arguments.point is not None:
        spec = mesh_point.ladder_spec(
            mesh_point.parse_point(arguments.point), fine_dx_m=arguments.fine_dx_m,
            radius_km=arguments.radius_km, background_km=arguments.background_km,
            transition_factor=arguments.transition_factor, ring_factor=arguments.ring_factor,
            name=arguments.name)
    else:
        raise MpasPortError("--global-mesh requires --spec or --point")
    regions = spec.get("regions") or []
    finest = min([float(spec["background_km"])] + [float(r["spacing_km"]) for r in regions]) * 1000
    if not math.isfinite(finest) or finest <= 0:
        raise MpasPortError("the finest mesh spacing must be finite and positive")
    card = mesh_point.resolve_card(arguments.card)
    if card is None:
        raise MpasPortError("--global-mesh requires --card to price the whole forecast")
    vertical = (arguments.vertical_spec or
                Path(__file__).with_name("data") / "countries" / "tc55-v1.json").resolve()
    levels = VerticalSpec.from_file(vertical).n_vert_levels
    from .mpas_defaults import default_dt_seconds
    dt = float(default_dt_seconds(finest)["dt_seconds"])
    # The physics and the card count the forecast will run, priced before anything is built (a plan priced on
    # another route's runs, with no card count, said a 417,810-cell globe fit two 32 GB cards: card_fit).
    from .global_door import forecast_arg_physics as planned_physics
    from .multi_member import argv_physics_route
    from .physics_default import ensure as with_default_physics
    # No physics named: the hex default the forecast will take (woof.hex.physics_default), on the same recipe context
    # the run passes it, so a GFS cold-start zoom globe is priced on convection_permitting + hrrr-like-aerosol (the
    # recipe_rules row), not on plain convection_permitting (hex 0.4.10 review: the plan said 28,715 MiB of plain
    # convection_permitting per card for a forecast that ran the aerosol profile).
    recipe_context = _recipe_context(arguments, regions)
    plan_flags = with_default_physics([str(item) for item in getattr(arguments, "forecast_arg", None) or ()], finest,
                                      context=recipe_context)
    planned = planned_physics(plan_flags)
    physics = {"route": argv_physics_route(plan_flags), "suite": planned.get("suite"),
               "profile": planned.get("profile")}
    cards = plan_cards(arguments, plan_flags)
    # The forecast runs on the cards priced: --cards N with no --samehost-devices of the caller's passes 0..N-1.
    samehost = ([] if cards == 1 or samehost_cards(plan_flags) is not None else
                ["--samehost-devices=" + ",".join(str(index) for index in range(cards))])
    if getattr(arguments, "template_only", False):
        if arguments.run or not arguments.generate or arguments.out_dir is None:
            raise MpasPortError("--template-only takes --generate --out-dir and no --run")
        if concentric_cap_center(spec) is None:
            raise MpasPortError("--template-only needs a spec whose regions are caps about one centre (a --point "
                                "ladder): no other spec is rotated, so its template would never be used")
    if arguments.run:
        if not arguments.generate or arguments.out_dir is None:
            raise MpasPortError("--run requires --generate --out-dir")
        if not arguments.source or not arguments.cycle:
            raise MpasPortError("--run requires --source and --cycle")
        from .init_sources import Cycle, load_table, resolve_source, resolve_writer, resolve_vtable
        issued = Cycle.parse(arguments.cycle)
        lead = int(getattr(arguments, "lead_hours", 0) or 0)
        if lead < 0:
            raise MpasPortError(f"--lead-hours {lead} is negative; it is the forecast hour of --cycle the run starts from")
        if getattr(arguments, "from_last_hour", False):
            from .source_horizon import SourceHorizonError, resolve_lead
            try:
                lead = resolve_lead(resolve_source(load_table(arguments.source_table), arguments.source),
                                    issued.when, lead, True)
            except SourceHorizonError as error:
                raise MpasPortError(str(error)) from None
            arguments.lead_hours = lead
        # The run starts at that forecast hour: the init is the hour's forecast (init-source --lead-hours, which
        # checks it against the source row's cadence and horizon), and a globe needs no boundaries.
        cycle = Cycle(issued.when + datetime.timedelta(hours=lead))
        issued_label = f"{arguments.cycle} f{lead:03d}" if lead else arguments.cycle
        source_table = load_table(arguments.source_table)
        source = resolve_source(source_table, arguments.source)
        from .init_sources import admit_cycle

        # Refused before minutes of mesh and static: a cycle the source never posted or posted without what
        # the init needs (the row's cycles block).
        admit_cycle(source, issued)
        resolve_writer(None)
        resolve_vtable(source)
        from .source_horizon import beyond_source, label as horizon_label
        extension = beyond_source(source, issued.when, lead, arguments.hours)
        if extension and not getattr(arguments, "brand", None):
            # Every sounding of a run past its source says so, top right (the host's own --brand wins).
            from .source_horizon import brand as horizon_brand
            arguments.brand = horizon_brand(f"{arguments.source} {issued_label}", extension)
        issued_label = horizon_label(issued_label, extension)
        if lead:
            from .init_sources import template_context

            # Refused here, before minutes of mesh and static: a lead off the row's cadence or past its horizon.
            template_context(issued, source, arguments.member, lead)
        seconds = arguments.hours * 3600
        cadence = arguments.history_every_minutes * 60
        if (not math.isfinite(seconds) or seconds <= 0 or cadence <= 0 or
                abs(seconds / cadence - round(seconds / cadence)) > 1e-8 or
                abs(cadence / dt - round(cadence / dt)) > 1e-8):
            raise MpasPortError("history cadence must divide the positive run length in whole timesteps")
    engine = resolve(MESH, arguments.mesh_exe)
    sizing = dry_run(spec, engine=engine)
    pricing = dict(card=card, levels=levels, dt_seconds=dt, hours=arguments.hours,
                   budget_mib=None if arguments.vram_gib is None else arguments.vram_gib * 1024,
                   suite=physics["suite"], route=physics["route"], profile=physics["profile"])
    fit = card_fit.predict(sizing["predicted_cells"], cards=cards, **pricing)
    fit.update(card_fit.by_card_count(sizing["predicted_cells"], **pricing))
    receipt: dict[str, Any] = {
        "schema": "gpuwm-hex.global-vr/v1", "status": "planned", "spec": spec,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sizing": sizing, "prediction_before_generation": fit,
        "vertical_spec": str(vertical), "stages": [],
    }
    if arguments.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        named_profile = fit["physics"].get("profile")
        runtime_seconds = fit["runtime"]["seconds"]
        print(f"global mesh: {fit['cells']:,} predicted cells, {fit['physics']['suite']}"
              f"{' + ' + named_profile if named_profile else ''} "
              f"({fit['physics']['route']} route) on {cards} card{'s' if cards > 1 else ''}: "
              f"{fit['predicted_peak_mib']:,.1f} MiB peak per card, "
              f"{fit['required_free_mib']:,.1f} MiB required, fits={fit['fits']}; "
              f"fewest cards that fit: {fit['fewest_cards_that_fit'] or 'more than ' + str(card_fit.CARD_COUNTS[-1])}; "
              + (f"forecast runtime {runtime_seconds:,.1f} s (extrapolated from a measured run of this route, card "
                 f"and card count)" if runtime_seconds is not None else
                 "forecast runtime: no measured rate for this route/card"), flush=True)
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if not arguments.generate:
        return 0
    if not fit["fits"]:
        fewest = fit["fewest_cards_that_fit"]
        raise MpasPortError(
            f"whole global mesh requires {fit['required_free_mib']:,.1f} MiB per card on {cards} card(s); "
            f"budget {fit['budget_mib']:,.1f} MiB. "
            + (f"It fits {fewest} cards: pass --cards {fewest}. " if fewest else "")
            + "Or coarsen or shrink the refinement region")
    if arguments.out_dir is None:
        raise MpasPortError("--global-mesh --generate requires --out-dir")
    root = arguments.out_dir.expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "global-vr-receipt.json"
    if receipt_path.exists():
        raise MpasPortError(f"{receipt_path} exists; use a fresh --out-dir to preserve its evidence")
    spec_path, grid, static = [root / name for name in ("resolution-spec.json", "mesh.grid.nc", "mesh.static.nc")]
    rows_file = root / mesh_rows.MESH_ROWS_FILENAME
    for path in (spec_path, grid, static, rows_file):
        if path.exists():
            raise MpasPortError(f"{path} exists; use a fresh --out-dir")
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    receipt["spec_sha256"] = mesh_rows.sha256_file(spec_path)
    receipt["vertical_spec_sha256"] = mesh_rows.sha256_file(vertical)
    receipt["rows_file"] = str(rows_file)
    env = dict(os.environ)
    receipt["numerics_environment"] = {k: env.get(k) for k in ("WOOF_HEX_NUMERICS", "GPUWM_WRF_EXACT", "WOOF_HEX_CUDA_GRAPHS")}
    if arguments.run:
        # The hours past the source's last forecast hour, named (None when the source covers the run).
        receipt["lead_hours"] = lead
        receipt["beyond_source"] = extension
    started = time.monotonic()

    def save() -> None:
        receipt["wall_seconds"] = round(time.monotonic() - started, 3)
        pending = receipt_path.with_suffix(".json.pending")
        pending.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
        pending.replace(receipt_path)

    def stage(name: str, argv: list[Any]) -> dict[str, Any]:
        receipt["status"] = "running"
        receipt["active_stage"] = name
        # The stage's argv is on the receipt BEFORE the stage runs: a door that
        # joins the live run (woof hex add-nest at +3 h) reads the forecast's
        # folder, mesh, init, start and hours from it.  "stages" gains the row
        # only when the stage ends, so reading that alone refused every
        # mid-run add (shapes 2-3 checker, 2026-10-23).
        receipt.setdefault("started_stages", {})[name] = {
            "argv": [str(a) for a in argv],
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        save()
        print(f"STAGE {name}", flush=True)
        result = card_fit.run_stage([str(a) for a in argv], name=name, directory=root / "logs", env=env)
        receipt["stages"].append(result)
        save()
        if result["returncode"]:
            raise MpasPortError(f"{name} exited {result['returncode']}; log {result['log']}")
        return result

    from .physics_default import ensure, init_flags
    extra = ensure([str(item) for item in getattr(arguments, "forecast_arg", None) or ()], finest,
                   context=recipe_context)
    prefix = [sys.executable, "-m", "woof.hex.cli"]
    prefetch = None
    live_export = None
    try:
        static_engine = resolve(STATIC, arguments.static_exe)
        geog = mesh_point.resolve_geog(arguments.geog)
        generator_receipt = root / "mesh-generator.json"
        static_receipt = root / "mesh-static.json"
        receipt["engines"] = {
            "mesh": {"path": str(engine), "sha256": mesh_rows.sha256_file(engine)},
            "static": {"path": str(static_engine), "sha256": mesh_rows.sha256_file(static_engine)}}
        # The mesh and its static depend on the spec and the geography alone, never the cycle: a premade library
        # entry (woof.hex.prep_store, WOOF_HEX_PREP_STORE: a directory or an S3 ticket) is fetched instead of built,
        # byte-identical by its SHA-256. Measured on a g7e box 2026-10-23: the 60->15 km globe's mesh and static
        # took minutes to build and a library fetch takes seconds.
        from . import prep_store

        def stored_stage(name: str, argv: list[Any]) -> None:
            got = prep_store.run_or_fetch(argv, runner=lambda a: stage(name, a))
            if got.get("source") == "store":
                receipt["stages"].append({"name": name, "source": "store", "key": got.get("key"),
                                          **{k: v for k, v in got.items() if k in ("bytes", "seconds", "transfer_seconds", "store")}})
                save()
                print(f"STAGE {name} from the library ({got.get('key', '')[:12]})", flush=True)

        if arguments.run:
            init_common = ["init-source", "--source", arguments.source, "--cycle", arguments.cycle,
                           *(["--lead-hours", lead] if lead else []), "--out-dir", root / "init",
                           "--jobs", arguments.jobs, *init_flags(extra)]
            for path in arguments.staged:
                init_common += ["--staged", path]
            if arguments.source_table:
                init_common += ["--table", arguments.source_table]
            if arguments.member is not None:
                init_common += ["--member", arguments.member]
            # A suite whose init has its own producer (hrrrv5, rrfs_*) builds its own static from the geography
            # (init_sources.producer_geog_flags): the zoom globe hands the init the geography its static was built from.
            init_common += producer_init_geog(extra, geog)
            # The source fetch and decode need no mesh: they run beside the mesh and static (U-H6: 13.4 s that
            # started only after a 57.9 s static on the production zoom globe), and init-source --from-decoded
            # builds the init from them once the static exists.
            prefetch = card_fit.start_stage([*prefix, *init_common, "--stop-after", "decode"],
                                            name="init-source-decode", directory=root / "logs", env=env)
            receipt.setdefault("started_stages", {})["init-source-decode"] = {
                "argv": prefetch.argv, "started_utc": prefetch.started_utc, "beside": ["mesh", "static"]}
            save()
        mesh_argv = [engine, "--spec", spec_path, "--out", grid, "--receipt", generator_receipt]
        from .moving_members import TEMPLATE_CENTER, _near_antipode

        centre = concentric_cap_center(spec)
        template_only = bool(getattr(arguments, "template_only", False))
        exact = None if centre is None or template_only else prep_store.fetch_only(mesh_argv)
        if exact is not None:
            receipt["stages"].append({"name": "mesh", "source": "store", "key": exact.get("key"),
                                      **{k: v for k, v in exact.items()
                                         if k in ("bytes", "seconds", "transfer_seconds", "store")}})
            save()
            print(f"STAGE mesh from the library ({exact.get('key', '')[:12]})", flush=True)
        elif centre is None or (not template_only and _near_antipode(centre, TEMPLATE_CENTER)):
            stored_stage("mesh", mesh_argv)
        else:
            import hashlib

            from .moving_mesh import rotate_grid

            began = time.monotonic()
            folder = root / "template"
            folder.mkdir()
            template = rotation_template_spec(spec)
            template_spec = folder / "rotation-template.spec.json"
            template_spec.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n")
            template_grid = folder / "rotation-template.grid.nc"
            template_receipt = folder / "rotation-template-mesh.json"
            stored_stage("mesh-template", [engine, "--spec", template_spec, "--out", template_grid,
                                           "--receipt", template_receipt])
            template_sha = mesh_rows.sha256_file(template_grid)
            if template_only:
                receipt["template"] = {"spec": template, "grid": str(template_grid), "sha256": template_sha}
                receipt["status"] = "template"
                receipt.pop("active_stage", None)
                save()
                print(f"RECEIPT {receipt_path}", flush=True)
                return 0
            # The same template and point give the same bytes, so the static's prep-store key repeats.
            file_id = hashlib.sha256(f"{template_sha} {centre[0]!r} {centre[1]!r}".encode()).hexdigest()[:32]
            rotation = rotate_grid(template_grid, grid, start=TEMPLATE_CENTER, end=centre, file_id=file_id)
            rotation["source_sha256"] = template_sha
            generator_receipt.write_text(json.dumps({
                "schema": "gpuwm-hex.zoom-globe-rotation/v1",
                "emitted": {"path": str(grid), "sha256": rotation["sha256"]},
                "rotation": rotation,
                "template": {"spec": template, "spec_sha256": mesh_rows.sha256_file(template_spec),
                             "generator_receipt": (json.loads(template_receipt.read_text())
                                                   if template_receipt.is_file() else None)},
            }, indent=2, sort_keys=True, default=str) + "\n")
            template_grid.unlink()
            receipt["stages"].append({"name": "mesh", "source": "rotated-template",
                                      "wall_seconds": round(time.monotonic() - began, 3),
                                      "rotation_seconds": round(rotation["wall_seconds"], 3),
                                      "template_sha256": template_sha, "end_deg": list(centre),
                                      "parallelism": {"device": "CPU", "threads": 1,
                                                      "scope": "vectorised array rotation"}})
            save()
            print(f"STAGE mesh rotated from the cap-family template in {rotation['wall_seconds']:.1f} s", flush=True)
        stored_stage("static", [static_engine, "--grid", grid, "--out", static, "--geog", geog,
                                "--nominal-dx-m", finest, "--native-init", "--native-gwd-static",
                                "--receipt", static_receipt])
        errors = mesh_point.static_coordinate_errors(grid, static)
        if errors:
            raise MpasPortError(errors[0])
        admission = mesh_point.admit_pair(grid, static, fine_dx_m=finest)
        dt = admission["timestep_choice"]["dt_seconds"]
        # The point fields remain descriptive legacy metadata. The complete
        # authored spec, rather than an invented cap, is the row's authority.
        shape = min(regions, key=lambda region: float(region["spacing_km"]))["shape"] if regions else {}
        point = tuple(shape.get("center_deg", [0.0, 0.0]))
        row = mesh_rows.describe_resolution_spec(
            spec=spec, name=f"hex-vr-{receipt['spec_sha256'][:16]}-{admission['n_cells']}",
            grid=grid, static=static, spec_path=spec_path,
            generator_receipt=generator_receipt, static_receipt=static_receipt,
            point_deg=point, fine_dx_m=finest, core_radius_km=float(shape.get("radius_km", 0)),
            background_km=float(spec["background_km"]), dt_seconds=dt,
            n_levels=levels, admission=admission)
        mesh_rows.write_rows(rows_file, [row])
        receipt["mesh_row"] = row.as_dict()
        measured_fit = card_fit.predict(row.n_cells, cards=cards, **dict(pricing, dt_seconds=dt))
        receipt["prediction_on_generated_mesh"] = measured_fit
        save()
        if not measured_fit["fits"]:
            raise MpasPortError("the emitted global mesh exceeds its card budget; see prediction_on_generated_mesh")
        if arguments.run:
            cadence = arguments.history_every_minutes * 60
            if abs(cadence / dt - round(cadence / dt)) > 1e-8:
                raise MpasPortError(f"history cadence does not divide the measured mesh timestep {dt} s")
            env[mesh_rows.MESH_ROWS_ENVIRONMENT] = os.pathsep.join(
                [str(rows_file), *[str(p) for p in mesh_rows.row_files()]])
            decoded = prefetch.finish()
            receipt["stages"].append(decoded)
            save()
            if decoded["returncode"]:
                raise MpasPortError(f"init-source-decode exited {decoded['returncode']}; log {decoded['log']}")
            # The GSL drag's orographic statistics (hrrrv5's bl_ugwp_gwdo), built beside the init on all cores so
            # the forecast never waits on them (orographic_statistics_job).
            oro_job = orographic_statistics_job(extra, row.name, grid, geog, root)
            stage("init-source", [*prefix, *init_common, "--from-decoded",
                                  "--grid", grid, "--static", static, "--vertical-spec", vertical])
            oro = finish_orographic_statistics(oro_job, extra, root)
            if oro is not None:
                receipt["orographic_statistics"] = oro
                extra += ["--orographic-statistics", oro["path"]]
                save()
            init_receipt = root / "init" / "init-source-receipt.json"
            receipt["source_binding"] = json.loads(init_receipt.read_text())
            run = root / "forecast"
            # The f000 maps the moment the first frame is written, not after the whole forecast (first_look.py);
            # both passes name the same case, so the full pass replaces the first look's pictures in place.
            from .first_look import FirstLook
            from .products_door import label_argv
            case = f"{row.name}-{cycle.start_time[:13].replace('-', '').replace('_', '')}"
            look = FirstLook(run, grid=grid, init=root / "init" / "init.nc", start_time=cycle.start_time,
                             out=root / "products", case=case,
                             extra=[*(a for region in arguments.region for a in ("--region", region)),
                                    *label_argv(arguments)],
                             log=root / "logs" / "first-look.log")
            look.start()
            # The run's export beside the forecast, each frame as it is closed, when the host names one
            # (WOOF_HEX_EXPORT_LIVE, export_door.live_export_command): the export step after the door then takes
            # those frames instead of converting the whole run on the CPU while the cards sit idle.
            from .export_door import LiveExport
            frames = int(round(float(arguments.hours) * 60 / float(arguments.history_every_minutes))) + 1
            live_export = LiveExport(run, grid=grid, init=root / "init" / "init.nc", start_time=cycle.start_time,
                                     frames=frames, log=root / "logs" / "live-export.log").start()
            # A named suite (--forecast-arg=--physics-suite=NAME) takes the forecast door's own step for it,
            # as the global door does; the caller's flags come last and win.
            from .global_door import forecast_arg_physics
            named = forecast_arg_physics(extra)
            # The receipt names the physics the forecast ran, as the global door's does (the zoom-globe receipt
            # named none: both-routes gate, rg1).  A run that names no mode runs the forecast door's default mode
            # (correct, on both routes), so that is what the receipt says.  THE BREAKAGE THIS PREVENTS (hex-meshes-
            # sources review, 2026-10-29): '--forecast-arg=--physics-suite=convection_permitting' ran correct mode
            # but the receipt said "exact", and the one-card shapes table was read as exact-mode numbers.
            from .correct_mode import DEFAULT_PHYSICS_MODE
            receipt["physics"] = {"suite": named.get("suite") or "arwen",
                                  "mode": named.get("mode") or DEFAULT_PHYSICS_MODE,
                                  **({"profile": named["profile"]} if named.get("profile") else {}),
                                  **({"namelist": named["namelist"]} if named.get("namelist") else {})}
            # A suite run with the selector changes of one of its named variants is labelled as that variant
            # ("hrrrv5 + storm scheme"), with what differs from the suite, never as the suite alone.
            from .mpasphys.route_table import forecast_arg_overrides, named_variant
            variant = named_variant(named.get("suite"), forecast_arg_overrides(extra))
            if variant is not None:
                receipt["physics"]["variant"] = variant[0]
                receipt["physics"]["differences_from_suite"] = list(variant[1].get("differences_from_suite", ()))
                print(f"physics: {variant[0]} ({named.get('suite')} with "
                      + ", ".join(f"{k}={v}" for k, v in variant[1]["selectors"].items()) + ")", flush=True)
            from .physics_default import door_step
            # Full-step LTS: the forecast door picks the step, so its default moves onto the class clock
            # (mpas_defaults.lts_class_clock_dt); an explicit --dt here would be refused by that clock instead.
            lts = any(str(item).split("=", 1)[0] == "--fullstep-local-timestep" for item in extra)
            # A named suite takes the forecast door's step, unless this mesh's measured admission step is shorter.
            step = [] if lts else door_step(named, dt, finest)
            from woof.output_formats import argument_values
            extra += argument_values(arguments)
            measured = stage("forecast", [*prefix, "forecast", "--mesh", row.name,
                "--grid", grid, "--static", static, "--init", root / "init" / "init.nc",
                "--init-source", f"{arguments.source} {issued_label}; spec {receipt['spec_sha256']}",
                "--start-time", cycle.start_time, "--hours", arguments.hours,
                *step, "--history-every-minutes", arguments.history_every_minutes,
                "--out", run, "--scratch", root / "scratch", "--no-products", *extra, *samehost])
            receipt["first_look"] = look.finish()
            # The forecast was the run's last GPU work; the products are CPU work in Rust binaries, so a card
            # queue gets its card back now instead of after the maps (woof.hex.card_lease).
            from .card_lease import gpu_work_done
            receipt["card_given_back_before_products"] = gpu_work_done(env)
            receipt["predicted_vs_measured"] = card_fit.comparison(measured_fit, measured)
            products = [*prefix, "products", run, "--out", root / "products", "--case", case, "--jobs", arguments.jobs,
                        *label_argv(arguments)]
            for region in arguments.region:
                products += ["--region", region]
            if look.receipt() is not None:  # the hours the first look rendered while the forecast ran keep their maps
                products += ["--live-frames", look.receipt()]
            stage("products", products)
            receipt["products_manifest"] = str(root / "products")
            receipt["live_export"] = live_export.finish(forecast_ok=True)
            live_export = None
        receipt["status"] = "complete" if arguments.run else "generated"
        receipt.pop("active_stage", None)
        save()
    except BaseException as error:
        if prefetch is not None:
            prefetch.cancel()  # the door failed before the init needed it
        if live_export is not None:
            receipt["live_export"] = live_export.finish(forecast_ok=False)
        if not isinstance(error, Exception):
            raise
        receipt["status"] = "failed"
        receipt["error"] = str(error)
        save()
        raise
    print(f"RECEIPT {receipt_path}", flush=True)
    return 0
