"""WOOF's own physics registry, read at resolve time.

Hex runs WOOF's physics through a column seam.  Which schemes exist is not
hex's to list: the engine publishes them in ``woof/physics_registry_v2.json``
(components, their implemented options and selector values, and every knob
under ``parameters``).  This module reads that document from the engine tree
the run names and answers, for one hex selector key and one value, which
``RunConfig`` assignments the value makes -- so a scheme WOOF adds is
selectable on hex the day the engine registers it, with no hex edit.

WHAT IS READ, AND HOW.  The registry JSON is read as a file from the engine
tree (``--gpuwm-checkout``, else the importable ``woof`` package's own
directory); nothing here imports the engine.  The two questions that need
engine CODE -- "does WOOF accept this combination" and "which species, soil
column, statics and source files does it run" -- are asked of the engine's
generic column seam (``woof.core.mpas_column_suite.column_physics_plan``)
through :func:`describe_selection`.

REFUSALS.  A value the registry does not declare, an option it declares
unimplemented, a parameter value outside its enum: each is refused with
what the registry says.  An engine tree with no registry, or with no generic
seam, is named as the missing dependency it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "GENERIC_SEAM_ENTRYPOINT",
    "GENERIC_SEAM_FUNCTION",
    "GENERIC_SEAM_MODULE",
    "EngineOption",
    "EngineRegistryUnavailable",
    "EngineValueRefusal",
    "component_values",
    "describe_selection",
    "engine_root",
    "generic_seam_problem",
    "implemented_options",
    "load_registry",
    "parameter_row",
    "plan_mapping",
    "parameter_value",
    "resolve_selector_value",
    "selection_labels",
    "checkout_with_registry",
    "setting_value",
]

REGISTRY_RELATIVE = "woof/physics_registry_v2.json"
GENERIC_SEAM_MODULE = "woof.core.mpas_column_suite"
GENERIC_SEAM_FUNCTION = "column_physics_plan"
GENERIC_SEAM_ENTRYPOINT = "run_mpas_column_physics"
GENERIC_SEAM_RELATIVE = "woof/core/mpas_column_suite.py"


class EngineRegistryUnavailable(RuntimeError):
    """No engine tree with a physics registry could be found."""


class EngineValueRefusal(ValueError):
    """The registry does not carry this value; ``remedy`` says what it has."""

    def __init__(self, reason: str, remedy: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.remedy = remedy


@dataclass(frozen=True)
class EngineOption:
    """One registry option (or parameter value), as RunConfig assignments."""

    option_id: str
    label: str
    maturity: str | None
    assignments: Mapping[str, Any]


def checkout_with_registry(checkout: str | Path | None) -> Path | None:
    """``checkout`` when it is an engine tree carrying a registry, else
    ``None`` (the importable ``woof`` then answers).  The doors pass the
    result to the resolver; nothing is remembered between calls."""

    if checkout is None:
        return None
    root = Path(checkout).expanduser().resolve()
    return root if (root / REGISTRY_RELATIVE).is_file() else None


def engine_root(checkout: str | Path | None = None) -> Path:
    """The engine tree whose registry answers: the named checkout, else the
    tree ``woof`` is (or would be) imported from."""

    if checkout is not None:
        root = Path(checkout).expanduser().resolve()
        if not (root / REGISTRY_RELATIVE).is_file():
            raise EngineRegistryUnavailable(
                f"the engine tree at {root} carries no {REGISTRY_RELATIVE}, so "
                "nothing says which physics options WOOF runs")
        return root
    loaded = sys.modules.get("woof")
    origin = getattr(loaded, "__file__", None) if loaded is not None else None
    if origin is None:
        try:
            spec = importlib.util.find_spec("woof")
        except (ImportError, ValueError):
            spec = None
        origin = None if spec is None else spec.origin
    if origin is None:
        raise EngineRegistryUnavailable(
            "no woof engine is importable and no --gpuwm-checkout was named, "
            "so WOOF's physics registry cannot be read")
    root = Path(origin).resolve().parent.parent
    if not (root / REGISTRY_RELATIVE).is_file():
        raise EngineRegistryUnavailable(
            f"the woof at {root} carries no {REGISTRY_RELATIVE}")
    return root


@lru_cache(maxsize=8)
def _load(path: str, mtime_ns: int, size: int) -> Mapping[str, Any]:
    del mtime_ns, size  # cache key only: an edited registry is re-read
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document.get("components"), dict):
        raise EngineRegistryUnavailable(f"{path} carries no components block")
    return document


def load_registry(checkout: str | Path | None = None) -> Mapping[str, Any]:
    """The engine's registry document.  Read-only by convention."""

    path = engine_root(checkout) / REGISTRY_RELATIVE
    stat = path.stat()
    return _load(str(path), stat.st_mtime_ns, stat.st_size)


def implemented_options(component: str, checkout: str | Path | None = None) -> dict[str, Mapping[str, Any]]:
    """``{option id: option}`` over one component's implemented options."""

    components = load_registry(checkout)["components"]
    if component not in components:
        raise EngineRegistryUnavailable(
            f"WOOF's physics registry has no component {component!r}")
    return {
        ident: option
        for ident, option in components[component].get("options", {}).items()
        if isinstance(option, dict) and option.get("implemented") is True
        and isinstance(option.get("selectors"), dict) and option["selectors"]
    }


def component_values(component: str, selector: str, checkout: str | Path | None = None) -> dict[int, tuple[str, ...]]:
    """``{selector value: option ids}`` over the implemented options."""

    values: dict[int, list[str]] = {}
    for ident, option in implemented_options(component, checkout).items():
        if selector in option["selectors"]:
            values.setdefault(int(option["selectors"][selector]), []).append(ident)
    return {value: tuple(ids) for value, ids in sorted(values.items())}


def _as_int(text: Any) -> int | None:
    if isinstance(text, bool):
        return None
    if isinstance(text, int):
        return text
    try:
        return int(str(text).strip())
    except ValueError:
        return None


def parameter_row(name: str, checkout: str | Path | None = None) -> Mapping[str, Any] | None:
    """The registry's ``parameters`` row for one knob, or ``None``."""

    row = load_registry(checkout).get("parameters", {}).get(name)
    return row if isinstance(row, dict) else None


def setting_reader(name: str, checkout: str | Path | None = None) -> tuple[str | None, str | None]:
    """Who reads one engine setting: ``(component, consuming_read)``.

    ``component`` is the registry component the name is a selector key of
    (``km_opt`` -> ``turbulence``), ``consuming_read`` the module its
    ``parameters`` row says reads it.  Either is ``None`` when the registry
    does not say.
    """

    registry = load_registry(checkout)
    component = next(
        (key for key, block in registry["components"].items()
         if name in block.get("selector_keys", ())), None)
    row = parameter_row(name, checkout)
    return component, (None if row is None else row.get("consuming_read"))


def parameter_value(name: str, value: Any, checkout: str | Path | None = None) -> Any:
    """``value`` typed by the registry's own parameter row, or a refusal."""

    row = parameter_row(name, checkout)
    if row is None:
        raise EngineValueRefusal(
            f"WOOF's physics registry declares no parameter {name!r}, so it "
            "would be accepted and never read",
            "a parameter of woof/physics_registry_v2.json")
    if row.get("implemented") is False:
        raise EngineValueRefusal(
            "WOOF itself does not run it: "
            + str(row.get("unimplemented_reason", "the registry marks it unimplemented")),
            "a parameter the engine implements")
    kind = row.get("type")
    typed: Any = value
    text = str(value).strip().strip("'\"")
    try:
        if kind == "integer" and not isinstance(value, bool):
            typed = int(text)
        elif kind == "number" and not isinstance(value, bool):
            typed = float(text)
        elif kind == "boolean":
            if isinstance(value, bool):
                typed = value
            elif text.lower().strip(".") in ("true", "t", "1"):
                typed = True
            elif text.lower().strip(".") in ("false", "f", "0"):
                typed = False
            else:
                raise ValueError(text)
        elif kind == "string":
            typed = text
    except ValueError:
        raise EngineValueRefusal(
            f"the registry types {name} as {kind}", f"a {kind} value") from None
    enum = row.get("enum")
    if isinstance(enum, list) and typed not in enum:
        raise EngineValueRefusal(
            f"WOOF admits {name} in {enum}", f"{name} in {enum}")
    return typed


def setting_value(name: str, value: Any, checkout: str | Path | None = None) -> Any:
    """One engine setting outside hex's nine selector keys, typed.

    A selector key of another registry component (``sf_urban_physics``,
    ``km_opt``) must be a value one of its implemented options carries; a
    row of the registry's ``parameters`` is typed and bounded by that row;
    anything else is passed as the scalar it spells, and the engine's plan
    refuses it by name if it is not a RunConfig field.
    """

    registry = load_registry(checkout)
    for component, block in registry["components"].items():
        if name in block.get("selector_keys", ()):
            found = resolve_selector_value(
                {"component": component, "selector": name}, value, checkout)
            return found.assignments[name]
    if parameter_row(name, checkout) is not None:
        return parameter_value(name, value, checkout)
    if isinstance(value, str):
        text = value.strip().strip("'\"")
        lowered = text.lower().strip(".")
        if lowered in ("true", "false"):
            return lowered == "true"
        for cast in (int, float):
            try:
                return cast(text)
            except ValueError:
                continue
        return text
    return value


def resolve_selector_value(
    route: Mapping[str, Any], value: Any, checkout: str | Path | None = None
) -> EngineOption:
    """One hex selector value, resolved against the registry.

    ``route`` is the selector's row of the suite table's ``engine_selectors``
    block: ``{"component": id, "selector": key}`` for a registry component
    (the value is an option id or the selector's own number), or
    ``{"parameter": name}`` for a knob of the registry's ``parameters``.
    """

    if "parameter" in route:
        name = str(route["parameter"])
        typed = parameter_value(name, value, checkout)
        return EngineOption(f"{name}={typed}", f"{name}={typed}", None,
                            MappingProxyType({name: typed}))
    component, selector = str(route["component"]), str(route["selector"])
    options = implemented_options(component, checkout)
    text = str(value).strip()
    chosen: list[str] = []
    if text in options and selector in options[text]["selectors"]:
        chosen = [text]
    else:
        number = _as_int(text)
        if number is not None:
            chosen = list(component_values(component, selector, checkout).get(number, ()))
    if not chosen:
        every = load_registry(checkout)["components"][component].get("options", {})
        if text in every:
            raise EngineValueRefusal(
                f"WOOF's physics registry declares {component} option {text!r} "
                f"at maturity {every[text].get('maturity')!r} and does not "
                "implement it, so no engine code would run under its name",
                f"an implemented {component} option: {sorted(options)}")
        known = {v: ids for v, ids in component_values(component, selector, checkout).items()}
        raise EngineValueRefusal(
            f"WOOF's physics registry declares no {component} option {text!r} "
            f"(neither an option id nor a {selector} value), so the run could "
            "not say which scheme it is",
            f"{selector} in {sorted(known)} or an option id in {sorted(options)}")
    option = options[chosen[0]]
    number = int(option["selectors"][selector])
    return EngineOption(
        option_id=chosen[0] if len(chosen) == 1 else "|".join(chosen),
        label=str(option.get("label", chosen[0])),
        maturity=option.get("maturity"),
        assignments=MappingProxyType({selector: number}),
    )


def selection_labels(
    physics: Mapping[str, Any], checkout: str | Path | None = None,
    *, skip: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """The registry's own labels for one RunConfig physics mapping.

    Each implemented option whose selector assignments all appear in
    ``physics`` contributes its ``label`` (``Thompson / MP8``); an assignment
    no option's selectors account for is spelled ``field=value``.  Fields in
    ``skip`` are left out (the cumulus, which hex decides at the mesh bind
    and labels through convection admission).  Order is the mapping's own,
    so one selection always reads the same.  No scheme is named here: a
    scheme WOOF registers is labelled the day it is registered.
    """

    remaining = {str(k): v for k, v in physics.items() if str(k) not in skip}
    labelled: dict[str, str] = {}
    for component in load_registry(checkout)["components"]:
        for ident, option in implemented_options(component, checkout).items():
            selectors = option["selectors"]
            if all(name in remaining and _same(remaining[name], value)
                   for name, value in selectors.items()):
                first = next(name for name in remaining if name in selectors)
                labelled.setdefault(first, str(option.get("label", ident)))
                for name in selectors:
                    if name != first:
                        labelled.setdefault(name, "")
    labels = []
    for name, value in remaining.items():
        label = labelled.get(name)
        if label is None:
            labels.append(f"{name}={value}")
        elif label:
            labels.append(label)
    return tuple(labels)


def _same(left: Any, right: Any) -> bool:
    a, b = _as_int(left), _as_int(right)
    return (a == b) if a is not None and b is not None else str(left) == str(right)


def generic_seam_problem(checkout: str | Path | None = None) -> str | None:
    """``None`` when the engine tree carries the generic column seam.

    THE BREAKAGE THIS NAMES: a selection resolved from the registry is run
    by handing the engine one physics mapping.  An engine whose column seam
    takes only its five slot names cannot receive it, and the run would stop
    with a TypeError after the card is reserved.  Read from the seam's
    source text, so the door asks without importing the engine.
    """

    try:
        root = engine_root(checkout)
    except EngineRegistryUnavailable as error:
        return str(error)
    source = root / GENERIC_SEAM_RELATIVE
    if not source.is_file():
        return (
            f"the engine tree at {root} carries no {GENERIC_SEAM_RELATIVE}; a "
            "registry-resolved physics selection has no seam to be handed to.  "
            "REMEDY: an engine carrying the generic column seam")
    if f"def {GENERIC_SEAM_FUNCTION}" not in source.read_text(encoding="utf-8", errors="replace"):
        return (
            f"the engine tree at {root} carries a column seam with no "
            f"{GENERIC_SEAM_FUNCTION}: it takes five slot names, not a physics "
            "mapping, so a registry-resolved selection could not reach its "
            "RunConfig.  REMEDY: an engine carrying the generic column seam "
            "(development branch or the release that lands it)")
    return None


def describe_selection(
    physics: Mapping[str, Any], checkout: str | Path | None = None
) -> Mapping[str, Any] | None:
    """The engine's own account of one physics mapping, or ``None``.

    Calls ``woof.core.mpas_column_suite.column_physics_plan(physics)``: the
    engine builds the RunConfig it would run, judges it with its own
    validators (``validate_run_config``, ``validate_physics_capabilities``)
    and returns the resolved selection, the transported species, the soil
    column and the statics it reads.  A combination WOOF refuses raises
    ``ValueError`` with WOOF's own words, which the caller reports as the
    refusal.  ``None`` when the engine tree has no generic seam
    (:func:`generic_seam_problem` says so in words) or cannot be imported in
    this process.

    The door resolves before the mesh and the timestep are bound, so the
    plan is asked without them; an engine whose plan requires them is handed
    placeholders, which the species, soil and statics do not depend on.
    """

    if generic_seam_problem(checkout) is not None:
        return None
    root = engine_root(checkout)
    loaded = sys.modules.get("woof")
    if loaded is not None:
        origin = getattr(loaded, "__file__", None)
        if origin is not None and Path(origin).resolve().parent.parent != root:
            return None
    elif not any(isinstance(entry, str) and Path(entry or ".").resolve() == root.resolve()
                     for entry in sys.path):
        sys.path.insert(0, str(root))
    try:
        import importlib

        module = importlib.import_module(GENERIC_SEAM_MODULE)
    except ImportError:
        return None
    planner = getattr(module, GENERIC_SEAM_FUNCTION)
    try:
        plan = planner(dict(physics))
    except TypeError:
        plan = planner(
            dict(physics), n_levels=10, dt=60.0, radiation_seconds=600.0,
            surface_pbl_seconds=60.0, cumulus_seconds=None)
    return plan_mapping(plan)


_PLAN_FIELDS = (
    "physics", "species", "mass_species", "number_species", "state_fields",
    "radii", "n_soil_levels", "static_fields", "surface_accumulators", "identity",
)


def plan_mapping(plan: Any) -> Mapping[str, Any]:
    """An engine ``ColumnPhysicsPlan`` (or a mapping) as a read-only mapping."""

    if isinstance(plan, Mapping):
        return MappingProxyType(dict(plan))
    return MappingProxyType({
        name: getattr(plan, name) for name in _PLAN_FIELDS if hasattr(plan, name)
    })
