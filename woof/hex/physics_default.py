# Modified 2026-10-28: experimental HRRR-like profile and aerosol route support.
"""The native MPAS default and observation-qualified regional cold-start recipes.

MPAS physics is the default, on the command line and on the site (since 2026-10-25).
Fixed means default, so the default is flipped here, in code, for every door: a run that names no physics
option runs the MPAS route (``woof.hex.mpasphys``) with the suite the table
``data/profiles/physics-default.v1.json`` picks for the mesh's finest cell spacing (MPAS-A practice:
convection_permitting wherever cells are under 10 km; on coarser quasi-uniform meshes too in this release, the
reasons in the table).

ArWen physics stays selectable and unchanged: ``--physics-suite arwen_mpas_column`` takes the road a bare run took
before 0.4.6, byte for byte (``forecast_door.resolve_door_physics``).  Any other hex suite, a ``--physics
KEY=VALUE`` override, an explicit ``--convection`` scheme or an explicit ``--physics-backend`` row names physics
on its own and is never overridden by this default.

Regional cold-start doors also supply a context to ``ensure``. First-match recipe rows
select their suite, mode and profile before initialization. Explicit physics and proof
modes retain the native request. Global and warm-cycle defaults are unchanged.

Two entry points:

* :func:`apply` for a parsed forecast-door namespace (``woof hex forecast``, every member and segment process);
* :func:`ensure` for the forecast-flag lists the other doors pass through (``--forecast-arg``, a moving case's
  ``physics_args``), so their plans, step rules, card prices and rosters see the suite the forecast will run.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
import json
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "gpuwm-hex.physics-default/v1"
TABLE_PATH = "data/profiles/physics-default.v1.json"

#: Forecast-door flags that name the run's physics.  ``--physics`` is matched exactly (not ``--physics-mode``).
_NAMING_FLAGS = ("--physics-profile", "--physics-suite", "--physics-namelist", "--physics", "--physics-backend")
#: ``--convection`` names physics unless it is the door default.
_CONVECTION_DEFAULTS = ("auto", "")


@lru_cache(maxsize=1)
def load_table() -> Mapping[str, Any]:
    text = resources.files("woof.hex").joinpath(TABLE_PATH).read_text(encoding="utf-8")
    table = json.loads(text)
    if table.get("schema") != SCHEMA:
        raise ValueError(f"{TABLE_PATH}: schema {table.get('schema')!r} is not {SCHEMA}")
    rules = table.get("rules") or []
    if not rules or rules[-1].get("finest_dx_below_m") is not None:
        raise ValueError(f"{TABLE_PATH}: the last rule must match every mesh (finest_dx_below_m null)")
    return table


def arwen_suite() -> str:
    """The suite name that selects the WOOF-ArWen route as a bare run took it before 0.4.6."""
    return str(load_table()["arwen_suite"])


def default_suite(finest_dx_m: float | None = None) -> str:
    """The MPAS suite a run with no physics named takes on a mesh whose finest spacing is ``finest_dx_m``.

    ``None`` (spacing unknown at the caller) takes the suite every rule agrees on, else the last rule's."""
    return str(default_rule(finest_dx_m)["suite"])


def default_rule(finest_dx_m: float | None = None, *, context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    for rule in load_table().get("recipe_rules", ()):
        if context and all(context.get(key) in (value if isinstance(value, list) else [value])
                           for key, value in rule["match"].items()):
            return rule
    rules = load_table()["rules"]
    if finest_dx_m is None:
        suites = {rule["suite"] for rule in rules}
        return rules[0] if len(suites) == 1 else rules[-1]
    for rule in rules:
        bound = rule.get("finest_dx_below_m")
        if bound is None or float(finest_dx_m) < float(bound):
            return rule
    return rules[-1]


def record(finest_dx_m: float | None = None, *, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What a receipt or plan says about a defaulted run."""
    rule = default_rule(finest_dx_m, context=context)
    return {**{k: rule[k] for k in ("id", "mode", "profile") if k in rule},"route": load_table()["route"], "suite": rule["suite"], "finest_dx_m": finest_dx_m,
            "source": "the hex default (MPAS physics, suite by mesh: data/profiles/physics-default.v1.json)",
            "why": rule["why"]}


def _flag_name(item: str) -> str:
    return str(item).split("=", 1)[0]


def _unwrap(items: Iterable[Any]) -> list[str]:
    """``--forecast-arg=X`` spellings reduced to ``X`` (a door's own argv may carry them)."""
    out = []
    for item in items or ():
        text = str(item)
        out.append(text.split("=", 1)[1] if text.startswith("--forecast-arg=") else text)
    return out


#: ``--physics KEY=VALUE`` keys that pick a scheme GENERATION on the default suite (latest upstream or the pin),
#: not the run's physics: ``--physics gwdo_version=v842`` alone keeps the default MPAS suite (gwdo_version.py).
VERSION_KEYS = ("gwdo_version",)


def _version_only(value: Any) -> bool:
    return str(value).split("=", 1)[0].strip().casefold() in VERSION_KEYS


def names_physics(flags: Sequence[Any] | None) -> bool:
    """Whether a forecast-flag list names the run's physics (so no default applies)."""
    items = _unwrap(flags or ())
    for index, item in enumerate(items):
        name = _flag_name(item)
        if name == "--physics":
            value = item.split("=", 1)[1] if "=" in item else (items[index + 1] if index + 1 < len(items) else "")
            if _version_only(value):
                continue
        if name in _NAMING_FLAGS:
            return True
        if name == "--convection":
            value = item.split("=", 1)[1] if "=" in item else (items[index + 1] if index + 1 < len(items) else "")
            if str(value).strip().lower() not in _CONVECTION_DEFAULTS:
                return True
    return False


def ensure(flags: Sequence[Any] | None, finest_dx_m: float | None = None, *,
           context: Mapping[str, Any] | None = None) -> list[str]:
    """Append the matching table recipe when no physics is named; preserve explicit requests."""
    items = [str(item) for item in flags or ()]
    if names_physics(items):
        return items
    from .global_door import forecast_arg_physics
    mode = forecast_arg_physics(_unwrap(items)).get("mode")
    # An explicit proof mode keeps the pre-promotion request, including its native suite.
    rule = default_rule(finest_dx_m, context=context if mode in (None, "correct") else None)
    return [*items, f"--physics-suite={rule['suite']}",
            *([f"--physics-mode={rule['mode']}"] if "mode" in rule and mode is None else []),
            *([f"--physics-profile={rule['profile']}"] if "profile" in rule else [])]


def physics_overrides(flags: Sequence[Any] | None) -> list[str]:
    """Every ``--physics KEY=VALUE`` a forecast-flag list carries, in order, as ``KEY=VALUE`` (either spelling:
    ``--physics=KEY=VALUE`` or two items; ``--physics-mode`` and the other ``--physics-*`` flags are not overrides)."""
    items = _unwrap(flags or ())
    found: list[str] = []
    index = 0
    while index < len(items):
        flag, equals, value = items[index].partition("=")
        if flag == "--physics":
            if not equals:
                if index + 1 >= len(items):
                    break  # the forecast door refuses a flag with no value by name
                index += 1
                value = items[index]
            found.append(value)
        index += 1
    return found


def init_flags(flags: Sequence[Any] | None) -> list[str]:
    """Forward the resolved recipe to cold initialization, preserving native case-7 init.

    An ArWen-route suite is forwarded WITH the run's single scheme choices (``--physics KEY=VALUE``), and choices
    named with no suite are forwarded under the suite the forecast door gives them (the WOOF-ArWen suite).  THE BREAKAGE
    THIS PREVENTS (2026-11-09, a zoom globe from ERA5 1957-12-18T12 on the WOOF-ArWen suite with NSSL two-moment
    microphysics picked): only the suite was forwarded, so the init was built for the suite's own microphysics, had
    no variable for the picked scheme's species (NSSL's ten, P3's qib and qir, TEMPO's nwfa and nifa) and the
    forecast refused at its physics start.  A request that names a suite and no choice forwards exactly what it did.
    """
    from .global_door import forecast_arg_physics
    from .physics_suites import DEFAULT_SUITE, mpas_suite_names, suite_names
    named = forecast_arg_physics(_unwrap(flags or ()))
    if named.get("profile"):
        return ["--physics-profile", named["profile"]]
    suite = named.get("suite")
    picks = [item for override in physics_overrides(flags) for item in ("--physics", override)]
    if suite is None and picks and not named.get("namelist"):
        suite = DEFAULT_SUITE
    if suite and suite in suite_names() and suite not in mpas_suite_names():
        return ["--physics-suite", suite, *picks]
    if suite and init_producer(suite):
        # A suite whose model initializes with its own producer (RRFS-MPAS: hrrrv5 and the rrfs_* suites, RUC on
        # nine soil levels).  THE BREAKAGE IT PREVENTS: `woof hex regional --forecast-arg=--physics-suite=hrrrv5`
        # built MPAS-A's Noah four-layer case-7 init for a RUC suite, because hrrrv5 is listed with MPAS's own suites.
        return ["--physics-suite", suite]
    return []


def init_producer(suite: str) -> bool:
    """Whether ``suite`` initializes through its own producer row (data/profiles/init-producers.v1.json)."""
    from .mpasphys.rrfs.cold_start_rrfs import producer_for_suite
    try:
        return producer_for_suite(suite) is not None
    except Exception:
        return False


def namespace_names_physics(arguments: Any) -> bool:
    """Whether a parsed forecast-door namespace names the run's physics."""
    from .physics_backend_admission import DEFAULT_BACKEND

    if getattr(arguments, "physics_profile", None) not in (None, ""):
        return True
    if getattr(arguments, "physics_suite", None) not in (None, ""):
        return True
    if getattr(arguments, "physics_namelist", None) is not None:
        return True
    if any(not _version_only(value) for value in (getattr(arguments, "physics", None) or ())):
        return True
    if getattr(arguments, "physics_backend", None) not in (None, DEFAULT_BACKEND):
        return True
    return str(getattr(arguments, "convection", "auto") or "auto").strip().lower() not in _CONVECTION_DEFAULTS


def apply(arguments: Any, finest_dx_m: float | None = None) -> dict[str, Any] | None:
    """Give a forecast-door namespace that names no physics the default MPAS suite; returns the record, or None
    when the run named its own physics (then nothing changes)."""
    if namespace_names_physics(arguments):
        return None
    block = record(finest_dx_m)
    arguments.physics_suite = block["suite"]
    arguments.physics_default = block
    return block


def suite_takes_door_step(named: Mapping[str, Any]) -> bool:
    """Whether a door's named physics (``global_door.forecast_arg_physics``) takes the forecast door's own step for
    that suite, so the door passes no mesh ``--dt`` of its own.  The WOOF-ArWen suite named alone keeps the door's mesh
    step, exactly as a bare run did before 0.4.6."""
    if named.get("namelist"):
        return True
    suite = named.get("suite")
    return bool(suite) and suite != arwen_suite()


def suite_deck_step(named: Mapping[str, Any], finest_dx_m: Any) -> dict[str, Any] | None:
    """The step the forecast door takes from a named suite's own per-mesh deck, or None when it has no row here.

    hrrrv5 carries NOAA's rrfs-workflow ``config.meshdep`` rows (``workflow_meshdep`` in the suites table): on a mesh
    whose nominal spacing is within ``within_factor`` of a row's, a forecast with no ``--dt`` runs that row's step
    (20 s near 3 km), not the 6 s per km rule (``mpasphys.forecast_v842._workflow_meshdep``).  This reads the same
    rows by the same nearest-within-factor choice, so a door can say and price the step that will run.  A namelist
    decides its own step and has no deck here.
    """
    suite = named.get("suite")
    if not suite or named.get("namelist") or finest_dx_m is None:
        return None
    try:
        dx = float(finest_dx_m)
    except (TypeError, ValueError):
        return None
    if not dx > 0.0:
        return None
    import math

    from .physics_suites import load_table as suites_table

    for key, route in suites_table().items():
        spec = route.get("workflow_meshdep") if isinstance(route, Mapping) else None
        if not spec or str((spec.get("when") or {}).get("config_physics_suite")) != str(suite):
            continue
        pick = min(spec["meshes"], key=lambda m: abs(math.log(dx / float(m["nominal_dx_m"]))))
        if abs(math.log(dx / float(pick["nominal_dx_m"]))) > math.log(float(spec.get("within_factor", 1.2))):
            return None
        return {"dt_seconds": float(pick["dt"]), "substeps": int(pick["substeps"]), "mesh": str(pick["mesh"]),
                "source": f"{suite} deck row {pick['mesh']} ({key})"}
    return None


def forecast_step(named: Mapping[str, Any], dt_seconds: Any, finest_dx_m: Any,
                  courant_limit_seconds: Any = None) -> dict[str, Any]:
    """The step a door's forecast will run on its mesh, and why: what :func:`door_step` passes, read back.

    ``dt_seconds`` is the mesh's admitted step, ``courant_limit_seconds`` the largest step the mesh's own shortest
    edge admits (``mesh_point.choose_timestep``).  The result names the step, its source and its margin under the
    limit, so the plan, the ADMIT line, the receipt and the card price all state the step the run takes.
    """
    passed = door_step(named, dt_seconds, finest_dx_m, courant_limit_seconds)
    deck = suite_deck_step(named, finest_dx_m) if suite_takes_door_step(named) else None
    if passed:
        step, source = float(passed[1]), "the mesh's admitted step"
        if deck is not None and courant_limit_seconds is not None and deck["dt_seconds"] > float(courant_limit_seconds):
            source = (f"the mesh's admitted step: {deck['source']} asks {deck['dt_seconds']:g} s, over this mesh's "
                      f"limit of {float(courant_limit_seconds):.3f} s")
    elif deck is not None:
        step, source = deck["dt_seconds"], deck["source"]
    elif named.get("namelist"):
        return {"dt_seconds": None, "source": "the namelist's config_dt"}
    elif dt_seconds is not None and finest_dx_m is not None:
        from .mpas_defaults import default_dt_seconds

        step, source = float(default_dt_seconds(float(finest_dx_m))["dt_seconds"]), "6 s per km of the fine spacing"
    else:
        return {"dt_seconds": None if dt_seconds is None else float(dt_seconds), "source": "the mesh row's step"}
    out: dict[str, Any] = {"dt_seconds": step, "source": source}
    if courant_limit_seconds is not None:
        out["courant_limit_seconds"] = float(courant_limit_seconds)
        out["margin"] = float(courant_limit_seconds) / step
    return out


def door_step(named: Mapping[str, Any], dt_seconds: Any, finest_dx_m: Any,
              courant_limit_seconds: Any = None) -> list[str]:
    """The ``--dt`` a door passes the forecast door beside its physics.

    A named MPAS suite takes the forecast door's own step for it (``mpas_defaults.default_dt_seconds``: 6 s per km
    of the finest nominal spacing), so the door passes no step of its own -- unless the door's mesh step is SHORTER
    than that rule, and then the mesh step is passed.  THE BREAKAGE THIS PREVENTS (shapes run zc1200, 2026-10-29,
    eight-card system): the 96 -> 3 km zoom globe with a 1,200 km core was admitted at 12 s (shortest dual edge 1,342 m,
    acoustic Courant limit 12.45 s), but the default MPAS suite stepped it at the rule's 18 s, 1.45x past the
    mesh's own measured limit, because the zoom door passed no ``--dt`` beside the suite.  A namelist (its
    ``config_dt`` decides) and the WOOF-ArWen suite keep their roads unchanged."""
    if not suite_takes_door_step(named):
        return ["--dt", str(dt_seconds)]
    if named.get("namelist") or dt_seconds is None or finest_dx_m is None:
        return []
    from .mpas_defaults import default_dt_seconds

    rule = float(default_dt_seconds(float(finest_dx_m))["dt_seconds"])
    if float(dt_seconds) < rule:
        return ["--dt", str(dt_seconds)]
    # THE BREAKAGE THIS PREVENTS (18 Nov 2026, zoom-globe twins on a 16 GB card): the comparison above is
    # against the 6 s per km rule, but hrrrv5 runs its own deck step (NOAA's 20 s near 3 km).  A 3 km mesh admitted
    # at 18 s whose shortest edge puts its limit under 20 s (a 24 km-background globe: 19.6 s; a 12 km-rim region:
    # 19.9 s) got no --dt, so the forecast stepped it past the mesh's own limit.  With the limit given, such a mesh
    # runs its admitted step; a mesh the deck step fits keeps NOAA's deck untouched.
    if courant_limit_seconds is not None:
        deck = suite_deck_step(named, finest_dx_m)
        if deck is not None and deck["dt_seconds"] > float(courant_limit_seconds):
            return ["--dt", str(dt_seconds)]
    return []


def is_arwen_default(suite: str | None, *, overrides: Any = None, namelist: Any = None,
                     convection: Any = "auto") -> bool:
    """True when a request names the WOOF-ArWen suite and nothing else: the road a bare run took before 0.4.6."""
    return (suite == arwen_suite() and not overrides and namelist is None
            and str(convection or "auto").strip().lower() in _CONVECTION_DEFAULTS)
