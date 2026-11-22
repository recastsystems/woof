"""``woof hex physics``: the named physics suites, on the host, with no card.

Three verbs, all read from the one suite table (woof.hex.physics_suites):

* ``list``    -- every named suite, MPAS's own suites, and every option the
  column seams carry or refuse (with the breakage and the follow-up);
* ``resolve`` -- what a ``--physics-suite``/``--physics`` request runs: the
  seam, the backend row, the convection request, the differences.  A suite
  the forecast door sends down an MPAS suite route (the ``suite_routes`` of
  the mpasphys_v842 and mpasphys_rrfs rows) reports that route, resolved by
  the door's own ``woof.hex.mpasphys.forecast_v842.native_route``;
* ``import``  -- a ``namelist.atmosphere``, option by option.

The forecast door takes the same flags, so a request checked here is the
request the door resolves.  Exit status: 0 resolved, 3 refused.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Mapping

from .errors import ConfigurationRefusal

REFUSED_EXIT = 3


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=False, default=str))


def _mesh_spacing(mesh: str | None) -> float | None:
    """The registered row's declared spacing, so ``auto`` cumulus resolves
    here as the forecast door resolves it; ``None`` for no or an unknown mesh."""

    if mesh is None:
        return None
    from .forecast_door import load_registry

    row = load_registry().get(mesh)
    return None if row is None else getattr(row, "nominal_dx_m", None)


def _route_providers(table: Any) -> list[tuple[str, Any]]:
    """The table rows that carry an MPAS suite route, chosen as forecast_v842.native_route chooses them."""
    return [(key, row) for key, row in table.items()
            if isinstance(row, Mapping) and (row.get("backend") == "mpasphys" or row.get("backend_row") == key)
            and "suite_routes" in row]


def _mpas_route(table: Any, name: str | None, overrides: Any = None, mode: str | None = None) -> Any:
    """What the forecast door runs for this request on an MPAS suite route, or None when it has none.

    The door (forecast_door._run_forecast) calls forecast_v842.native_route first, and a name in a
    provider's ``suite_routes`` runs that provider's driver port (hex-cuda by default) without reaching
    physics_suites.resolve_physics.  Resolving it here with the same function keeps these verbs true to
    the door: before this they judged every MPAS suite on the WOOF-ArWen engine columns, refused
    convection_permitting on the engine-MYNN cloud pairing and named engine or column rows the forecast
    never runs.
    """
    if name is None:
        return None
    matches = [(key, row) for key, row in _route_providers(table) if name in row["suite_routes"]]
    if not matches:
        return None
    from .forecast_door import PHYSICS_MODE_DEFAULT
    from .mpasphys.forecast_v842 import native_route

    request = argparse.Namespace(physics_namelist=None, physics_suite=name, physics=list(overrides or ()) or None,
                                 physics_mode=mode, physics_correct_skip=None, physics_default_backend=None)
    config, _ = native_route(request, default_mode=PHYSICS_MODE_DEFAULT)
    if config is None:
        return None
    key, row = matches[0]
    return {
        "route": "mpas_suite",
        "suite": name,
        "native_suite": getattr(config, "native_suite", config.suite),
        "backend_row": key,
        "driver_module": row.get("driver_module", "woof.hex.mpasphys.driver_v842"),
        "default_backend": row["default_backend"],
        "physics_mode": config.mode,
        "corrections_applied": sorted(config.corrections),
        "gaps": [gap.as_dict() for gap in config.gaps],
        "selectors": dict(config.selectors),
        "backends": {slot: str(backend) for slot, backend in config.backends.items()},
        "cpu_slots": [f"{item.slot}={item.scheme}: {item.reason}" for item in config.cpu_slots],
    }


def _list(arguments: argparse.Namespace) -> int:
    from .physics_suites import load_table, mpas_suite_names, resolve_physics, suite_names

    table = load_table()
    names = suite_names() + mpas_suite_names()
    names += tuple(dict.fromkeys(alias for _, row in _route_providers(table) for alias in row["suite_routes"]
                                 if alias not in names))
    suites = []
    from .physics_default import default_suite
    door_default = default_suite(_mesh_spacing(arguments.mesh))
    for name in names:
        route = None
        try:
            route = _mpas_route(table, name)
            if route is not None:
                verdict = {"runs": True, "seam": "mpas_suite", "backend_row": route["backend_row"],
                           "driver_module": route["driver_module"],
                           "default_backend": route["default_backend"], "physics_mode": route["physics_mode"]}
            else:
                resolved = resolve_physics(
                    name, mesh=arguments.mesh, nominal_dx_m=_mesh_spacing(arguments.mesh))
                verdict = {"runs": True, "seam": resolved.seam, "backend_row": resolved.backend_row,
                           "convection_request": resolved.convection}
        except ConfigurationRefusal as error:
            verdict = {"runs": False, "refusals": [
                item.line() for item in getattr(error, "refusals", ())] or [str(error)]}
        except ValueError as error:
            verdict = {"runs": False, "refusals": [str(error)]}
        row = next((item for item in table["suites"] if item["name"] == name), None)
        mpas_row = table["mpas_suites"]["suites"].get(name)
        suites.append({
            "name": name,
            "kind": "hex" if row is not None else "MPAS",
            # The hex default (woof.hex.physics_default: MPAS-A's suite for this mesh); arwen_route_default marks
            # the suite the ArWen route takes when only an override names it.
            "default": name == door_default,
            "arwen_route_default": bool(row is not None and row.get("default")),
            "describes": None if row is None else row["describes"],
            "selectors": dict(row["selectors"]) if row is not None
            else dict(mpas_row) if mpas_row is not None
            else dict(route["selectors"]) if route is not None else {},
            **verdict,
        })
    options = {
        key: {value: (f"refused: {item['refused']}" if "refused" in item
                      else "hex rule (decided at the mesh bind)" if not item.get("engine")
                      else "WOOF " + ", ".join(f"{k}={v!r}" for k, v in item["engine"].items()))
              for value, item in rows.items()}
        for key, rows in table["options"].items()
    }
    # What exists is the engine's to say: every implemented option of WOOF's
    # registry, selectable by id or selector value under the same key.
    from . import engine_registry

    registry: dict[str, Any] = {}
    try:
        for key, route in table["engine_selectors"]["selectors"].items():
            if "component" in route:
                registry[key] = {
                    ident: {route["selector"]: option["selectors"][route["selector"]],
                            "label": option.get("label"), "maturity": option.get("maturity")}
                    for ident, option in engine_registry.implemented_options(
                        route["component"]).items()
                    if route["selector"] in option["selectors"]
                }
            else:
                row = engine_registry.parameter_row(route["parameter"]) or {}
                registry[key] = {route["parameter"]: row.get("enum")}
        registry_source = str(engine_registry.engine_root())
        seam_problem = engine_registry.generic_seam_problem()
    except engine_registry.EngineRegistryUnavailable as error:
        registry_source, seam_problem = None, str(error)
    _print({"mesh": arguments.mesh, "suites": suites, "options": options,
            "woof_registry": {"engine": registry_source, "options": registry,
                              "generic_seam_problem": seam_problem},
            "generation_selectors": {
                key: list(row["values"]) for key, row in table["generation_selectors"].items()}})
    return 0


def _resolve(arguments: argparse.Namespace) -> int:
    from .physics_suites import parse_override, resolve_physics

    from .physics_suites import load_table
    from .physics_default import default_suite

    if arguments.physics_suite is None and not arguments.physics:
        # Nothing named: what the forecast door runs, the hex default MPAS suite for the mesh.
        arguments.physics_suite = default_suite(_mesh_spacing(arguments.mesh))
    try:
        route = _mpas_route(load_table(), arguments.physics_suite, arguments.physics, arguments.physics_mode)
    except ValueError as error:
        _print({"resolved": False, "refusals": [str(error)]})
        return REFUSED_EXIT
    if route is not None:
        _print({"resolved": True, **route})
        return 0
    try:
        overrides = dict(parse_override(text) for text in arguments.physics or ())
        resolved = resolve_physics(
            arguments.physics_suite, overrides, mesh=arguments.mesh,
            nominal_dx_m=_mesh_spacing(arguments.mesh))
    except ConfigurationRefusal as error:
        _print({"resolved": False, "refusals": [
            {"option": item.knob, "value": item.value, "reason": item.reason,
             "follow_up": item.follow_up}
            for item in getattr(error, "refusals", ())] or [str(error)]})
        return REFUSED_EXIT
    _print({"resolved": True, **resolved.receipt()})
    return 0


def _import(arguments: argparse.Namespace) -> int:
    from .physics_namelist import read_physics_namelist

    result = read_physics_namelist(arguments.namelist, mesh=arguments.mesh)
    _print(result.report())
    return 0 if result.admitted else REFUSED_EXIT


def add_physics_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "physics",
        help="list, resolve and import named physics suites (no GPU)",
        description=(
            "The physics a forecast runs is a named suite plus overrides, read "
            "from one table (data/profiles/mpas-physics-suites.v1.json). These "
            "verbs answer, on the host, what `woof hex forecast` would run for "
            "the same flags."
        ),
    )
    verbs = parser.add_subparsers(dest="physics_verb", required=True)
    listing = verbs.add_parser("list", help="every suite and every option")
    listing.add_argument("--mesh", default=None,
                         help="a registered mesh row, for suites with per-mesh selectors")
    listing.set_defaults(handler=_list)
    resolve = verbs.add_parser("resolve", help="what one request runs")
    resolve.add_argument("--physics-suite", default=None, metavar="NAME")
    resolve.add_argument("--physics", action="append", default=None, metavar="KEY=VALUE")
    resolve.add_argument("--mesh", default=None)
    resolve.add_argument("--physics-mode", choices=("correct", "exact"), default=None,
                         help="as on the forecast door: an MPAS suite route runs correct unless exact is named")
    resolve.set_defaults(handler=_resolve)
    importer = verbs.add_parser("import", help="a namelist.atmosphere, option by option")
    importer.add_argument("namelist", metavar="NAMELIST")
    importer.add_argument("--mesh", default=None)
    importer.set_defaults(handler=_import)
