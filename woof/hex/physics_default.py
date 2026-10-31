# Modified 2026-10-28: experimental HRRR-like profile and aerosol route support.
"""The native MPAS default and observation-qualified regional cold-start recipes.

Owner ruling 2026-10-25: "now that mpas physics are good those should be the default in general and on site".
Fixed means default, so the default is flipped here, in code, for every door: a run that names no physics
option runs the MPAS route (``woof.hex.mpasphys``) with the suite the table
``data/profiles/physics-default.v1.json`` picks for the mesh's finest cell spacing (MPAS-A practice:
convection_permitting wherever cells are under 10 km; on coarser quasi-uniform meshes too in this release, the
reasons in the table).

WOOF physics stays selectable and unchanged: ``--physics-suite arwen_mpas_column`` takes the road a bare run took
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
    """The suite name that selects the WOOF route as a bare run took it before 0.4.6."""
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


def names_physics(flags: Sequence[Any] | None) -> bool:
    """Whether a forecast-flag list names the run's physics (so no default applies)."""
    items = _unwrap(flags or ())
    for index, item in enumerate(items):
        name = _flag_name(item)
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


def init_flags(flags: Sequence[Any] | None) -> list[str]:
    """Forward the resolved recipe to cold initialization, preserving native case-7 init."""
    from .global_door import forecast_arg_physics
    from .physics_suites import mpas_suite_names, suite_names
    named = forecast_arg_physics(_unwrap(flags or ()))
    if named.get("profile"):
        return ["--physics-profile", named["profile"]]
    suite = named.get("suite")
    if suite and suite in suite_names() and suite not in mpas_suite_names():
        return ["--physics-suite", suite]
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
    if getattr(arguments, "physics", None):
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
    that suite, so the door passes no mesh ``--dt`` of its own.  The WOOF suite named alone keeps the door's mesh
    step, exactly as a bare run did before 0.4.6."""
    if named.get("namelist"):
        return True
    suite = named.get("suite")
    return bool(suite) and suite != arwen_suite()


def door_step(named: Mapping[str, Any], dt_seconds: Any, finest_dx_m: Any) -> list[str]:
    """The ``--dt`` a door passes the forecast door beside its physics.

    A named MPAS suite takes the forecast door's own step for it (``mpas_defaults.default_dt_seconds``: 6 s per km
    of the finest nominal spacing), so the door passes no step of its own -- unless the door's mesh step is SHORTER
    than that rule, and then the mesh step is passed.  THE BREAKAGE THIS PREVENTS (shapes run zc1200, 2026-10-29,
    V8X6000): the 96 -> 3 km zoom globe with a 1,200 km core was admitted at 12 s (shortest dual edge 1,342 m,
    acoustic Courant limit 12.45 s), but the default MPAS suite stepped it at the rule's 18 s, 1.45x past the
    mesh's own measured limit, because the zoom door passed no ``--dt`` beside the suite.  A namelist (its
    ``config_dt`` decides) and the WOOF suite keep their roads unchanged."""
    if not suite_takes_door_step(named):
        return ["--dt", str(dt_seconds)]
    if named.get("namelist") or dt_seconds is None or finest_dx_m is None:
        return []
    from .mpas_defaults import default_dt_seconds

    rule = float(default_dt_seconds(float(finest_dx_m))["dt_seconds"])
    return ["--dt", str(dt_seconds)] if float(dt_seconds) < rule else []


def is_arwen_default(suite: str | None, *, overrides: Any = None, namelist: Any = None,
                     convection: Any = "auto") -> bool:
    """True when a request names the WOOF suite and nothing else: the road a bare run took before 0.4.6."""
    return (suite == arwen_suite() and not overrides and namelist is None
            and str(convection or "auto").strip().lower() in _CONVECTION_DEFAULTS)
