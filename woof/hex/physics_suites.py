"""Named physics suites and per-scheme overrides, as table work.

A hex forecast runs WOOF's physics through one of two column seams: the frozen
``woof.core.mpas_column_batch`` (YSU, revised MM5, Noah-MP, legacy RRTMG; a
microphysics per backend row) or the selectable ``woof.core.mpas_column_suite``
(the boundary layer, surface layer, land surface and drag as selectors).  Until
this module the only way to choose between them was ``--physics-backend ROW``,
and ``config_physics_suite`` had one admitted value.

WHAT THIS MODULE IS.  The reader of ``data/profiles/mpas-physics-suites.v1.json``
and the one resolver every front door calls: the configuration
(``config_physics_suite``), the CLI (``--physics-suite`` and ``--physics
KEY=VALUE``) and the MPAS ``namelist.atmosphere`` importer
(:mod:`woof.hex.physics_namelist`).  The resolver turns a suite name plus
overrides spelled with MPAS's own namelist keys (``config_pbl_scheme`` and the
eight others) and WOOF's generation selectors (``bl_mynn_version`` ...) into a
:class:`ResolvedPhysics`: which seam runs, which backend row, which convection
request the mesh bind decides from, and every known difference from the
configuration a suite is shaped after.

THE ARBITRARY ACCEPTANCE TEST.  No code path keys on a suite name, and the
table is not the list of schemes hex can run.  WHAT EXISTS is WOOF's own
physics registry (``woof/physics_registry_v2.json``, read from the run's
engine tree by :mod:`woof.hex.engine_registry` at resolve time): any option it
implements is selectable by its registry id (``kessler-mp1``) or its selector
value (``1``), under the MPAS key or WOOF's own (``mp_physics=1``), and any
other engine setting by its registry parameter name.  The table carries
NAMES: an MPAS spelling is an alias row onto the engine's selectors, a suite
is a row of spellings.  A scheme WOOF adds needs no hex edit.

THREE SEAMS, ONE DECISION.  The default selection (and the Thompson twin)
runs on the frozen column batch exactly as before; the selections the
anchored column-suite profiles name run behind ``run_mpas_column_suite``
exactly as before; EVERY other selection is handed to the engine's generic
column seam (``run_mpas_column_physics``) as one mapping of RunConfig
assignments, and WOOF's own validators judge the combination.  The species
row, the backend row, the soil column and the drag statistics of such a
selection are derived from the engine's plan
(``column_physics_plan``), never from a list here.

THE DEFAULT DOES NOT MOVE.  ``arwen_mpas_column`` resolves to the frozen
``wsm6_column`` row with the ``auto`` convection request: exactly what a bare
run selected before this module existed.  :meth:`ResolvedPhysics.is_default`
is what the doors test before changing anything about a run.

REFUSALS NAME THE BREAKAGE.  What is still refused: an option WOOF itself
cannot run (no registry option, or one the registry marks unimplemented), a
combination WOOF's validators refuse (reported in WOOF's words), a static
input no hex mesh stages, and an engine tree without the generic seam.  "No
seam runs it" is no longer a reason.  All refusals of
one request are reported together (:class:`PhysicsSelectionRefusal`), so a
namelist with three unsupported options names all three at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from importlib import resources
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .errors import ConfigurationRefusal

__all__ = [
    "DEFAULT_SUITE",
    "SCHEMA",
    "OptionRefusal",
    "PhysicsSelectionRefusal",
    "ResolvedPhysics",
    "engine_selector_aliases",
    "generation_keys",
    "load_table",
    "mpas_suite_names",
    "parse_override",
    "recordable_suite_names",
    "resolve_physics",
    "selector_keys",
    "suite_names",
    "suite_row",
    "validate_table",
]

SCHEMA = "gpuwm-hex.mpas-physics-suites/v1"
TABLE_PATH = "data/profiles/mpas-physics-suites.v1.json"

#: The suite a configuration selects when nothing names one.  A constant, not
#: "the row marked default" alone: the table's ``default`` flag is held equal
#: to this by :func:`load_table`, so the two cannot disagree.
DEFAULT_SUITE = "arwen_mpas_column"

_SEAMS = ("frozen", "column_suite", "engine")
_CONVECTION_REQUESTS = ("auto", "off", "gf", "ntiedtke")
_ROW_KINDS = ("carried", "engine", "refused")


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
class PhysicsTableError(ValueError):
    """The suite table violates its own schema."""


def _check_selectors(where: str, selectors: Mapping[str, Any], keys: tuple[str, ...],
                     options: Mapping[str, Mapping[str, Any]], *, complete: bool) -> None:
    unknown = sorted(set(selectors) - set(keys))
    if unknown:
        raise PhysicsTableError(f"{where} names selectors {unknown} outside {list(keys)}")
    if complete and set(selectors) != set(keys):
        raise PhysicsTableError(
            f"{where} must name every selector; missing {sorted(set(keys) - set(selectors))}"
        )
    for key, value in selectors.items():
        if value not in options[key]:
            raise PhysicsTableError(
                f"{where}: {key}={value!r} has no row in options.{key}"
            )


def route_default_for(row: Mapping[str, Any], suite: str | None) -> bool:
    """Whether a route row is the default for ``suite`` when no pin set is named.

    ``route_default`` is absent or true (default for every alias it routes), false (never), or a list of the
    aliases it is the default for. A list lets one pin set become the default for the suites proven on it
    while the other aliases stay on the previous pin until they are proven too (RRFS 44c0849: hrrrv5 only).
    """
    value = row.get("route_default", True)
    if isinstance(value, bool):
        return value
    return suite in value


@lru_cache(maxsize=1)
def load_table() -> Mapping[str, Any]:
    """The suite table, validated.  A malformed table refuses by name."""

    text = resources.files("woof.hex").joinpath(TABLE_PATH).read_text(encoding="utf-8")
    return validate_table(json.loads(text))


def validate_table(table: Mapping[str, Any]) -> Mapping[str, Any]:
    """Hold one parsed table to the schema; return it frozen."""

    if table.get("schema") != SCHEMA:
        raise PhysicsTableError(f"{TABLE_PATH} schema {table.get('schema')!r} is not {SCHEMA!r}")
    keys = tuple(table["selector_keys"])
    options = table["options"]
    if set(options) != set(keys):
        raise PhysicsTableError("options must carry exactly one family per selector key")
    for key, rows in options.items():
        for value, row in rows.items():
            if ("refused" in row) == ("carried" in row or "engine" in row):
                raise PhysicsTableError(
                    f"options.{key}.{value} must be refused, or an alias onto the "
                    "engine's selectors (engine, with carried when a legacy seam "
                    "also names it), exactly one"
                )
            if "refused" not in row and value != "auto" and not row.get("engine"):
                raise PhysicsTableError(
                    f"options.{key}.{value} names no engine assignment; an MPAS "
                    "spelling is an alias onto WOOF's selectors or it is nothing"
                )
            if "refused" in row and not str(row.get("follow_up", "")).strip():
                raise PhysicsTableError(
                    f"options.{key}.{value} is refused and names no follow-up; a "
                    "refusal that does not say what would lift it is a dead end"
                )
    for pairing in table["pairings"]:
        for side in ("if", "then"):
            for key, values in pairing[side].items():
                for value in values:
                    if value not in options.get(key, {}):
                        raise PhysicsTableError(f"pairing names {key}={value!r}, which has no row")
    routes = table["engine_selectors"]["selectors"]
    if set(routes) != set(keys):
        raise PhysicsTableError("engine_selectors must route exactly the selector keys")
    for key, route in routes.items():
        if not (("parameter" in route) ^ ("component" in route and "selector" in route)):
            raise PhysicsTableError(
                f"engine_selectors.{key} must name a registry component and its "
                "selector, or a registry parameter")
    generation = table["generation_selectors"]
    for name, row in generation.items():
        if not row["values"] or row["slot"] not in (
            "microphysics", "pbl_scheme", "surface_layer_scheme", "radiation_scheme"
        ):
            raise PhysicsTableError(f"generation selector {name} is malformed")
    names: set[str] = set()
    defaults = []
    for row in table["suites"]:
        name = row["name"]
        if name in names:
            raise PhysicsTableError(f"suite {name!r} is declared twice")
        if name != name.strip().lower() or not name:
            raise PhysicsTableError(f"suite name {name!r} is not a lowercase name")
        names.add(name)
        if row.get("default"):
            defaults.append(name)
        _check_selectors(f"suite {name}", row["selectors"], keys, options, complete=True)
        for mesh, part in row.get("selectors_by_mesh", {}).items():
            _check_selectors(f"suite {name} mesh {mesh}", part, keys, options, complete=False)
        for gen in row.get("generation", {}):
            if gen not in generation:
                raise PhysicsTableError(f"suite {name} names generation selector {gen!r} with no row")
        if row.get("shaped_after") and not row.get("differences_from_shaped_after"):
            raise PhysicsTableError(
                f"suite {name} is shaped after {row['shaped_after']!r} and lists no "
                "difference from it; every known difference is part of the row"
            )
    if defaults != [DEFAULT_SUITE]:
        raise PhysicsTableError(
            f"exactly one suite, {DEFAULT_SUITE!r}, is the default; the table marks {defaults}"
        )
    mpas = table["mpas_suites"]["suites"]
    collide = sorted(names & set(mpas))
    if collide:
        raise PhysicsTableError(
            f"hex suite names {collide} collide with MPAS's own config_physics_suite "
            "values; a namelist's value would then mean two different tables"
        )
    namelist = table.get("namelist", {})
    defaults = namelist.get("registry_defaults")
    if defaults is not None:
        registries = defaults["registries"]
        for name in [defaults["default_registry"], *defaults["by_suite"].values()]:
            if name not in registries:
                raise PhysicsTableError(f"registry_defaults names registry {name!r} with no row")
        for suite in defaults["by_suite"]:
            if suite not in mpas:
                raise PhysicsTableError(
                    f"registry_defaults.by_suite names {suite!r}, which is not an MPAS suite")
        for name, registry in registries.items():
            for option in registry["values"]:
                if option not in namelist["options"]:
                    raise PhysicsTableError(
                        f"registry {name} defaults {option!r}, which no namelist.options "
                        "row reads; the default would be written and never dispositioned")
    for name, selectors in mpas.items():
        if set(selectors) != set(keys):
            raise PhysicsTableError(f"MPAS suite {name} must name every selector")
        for key, value in selectors.items():
            if value not in options[key]:
                raise PhysicsTableError(
                    f"MPAS suite {name}: {key}={value!r} has no row in options.{key}; "
                    "every value MPAS can resolve must be carried or refused by name"
                )
    return _freeze(table)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def selector_keys() -> tuple[str, ...]:
    return tuple(load_table()["selector_keys"])


def generation_keys() -> tuple[str, ...]:
    return tuple(load_table()["generation_selectors"])


def suite_names() -> tuple[str, ...]:
    """Hex's named suites, in table order."""

    return tuple(row["name"] for row in load_table()["suites"])


def mpas_suite_names() -> tuple[str, ...]:
    """MPAS's own ``config_physics_suite`` values the table resolves."""

    return tuple(load_table()["mpas_suites"]["suites"])


def recordable_suite_names() -> tuple[str, ...]:
    """Every value a configuration's ``config_physics_suite`` may record."""

    return suite_names() + mpas_suite_names()


def suite_row(name: str) -> Mapping[str, Any]:
    for row in load_table()["suites"]:
        if row["name"] == name:
            return row
    raise KeyError(name)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OptionRefusal:
    """One option a request cannot have, with the breakage and the remedy."""

    knob: str
    value: Any
    reason: str
    follow_up: str

    def line(self) -> str:
        return f"{self.knob}={self.value!r}: {self.reason} (follow-up: {self.follow_up})"


class PhysicsSelectionRefusal(ConfigurationRefusal):
    """Every option of one physics request that is refused, named together."""

    def __init__(self, refusals: Iterable[OptionRefusal], *, context: str) -> None:
        self.refusals = tuple(refusals)
        first = self.refusals[0]
        lines = "; ".join(item.line() for item in self.refusals)
        super().__init__(
            "physics" if len(self.refusals) > 1 else first.knob,
            context if len(self.refusals) > 1 else first.value,
            f"{len(self.refusals)} physics option(s) refused -- {lines}",
            "a selection every option of which the column seam carries "
            "(woof.hex.physics_suites; woof hex physics list)",
        )


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------
def parse_override(text: str) -> tuple[str, Any]:
    """``KEY=VALUE`` from the CLI, typed by the key's own table row."""

    if "=" not in text:
        raise ConfigurationRefusal(
            "--physics", text, "an override is KEY=VALUE", "--physics config_pbl_scheme=bl_mynn"
        )
    key, raw = (part.strip() for part in text.split("=", 1))
    raw = raw.strip("'\"")
    table = load_table()
    if key in table["generation_selectors"]:
        values = table["generation_selectors"][key]["values"]
        if all(isinstance(value, int) for value in values):
            try:
                return key, int(raw)
            except ValueError:
                return key, raw
        return key, raw
    if key == "config_frac_seaice" or table["namelist"]["options"].get(key, {}).get("kind") == "surface_update":
        lowered = raw.lower().strip(".")
        if lowered in ("true", "t"):
            return key, True
        if lowered in ("false", "f"):
            return key, False
        return key, raw
    return key, raw


def engine_selector_aliases() -> Mapping[str, str]:
    """WOOF's own selector names -> the MPAS key each one selects.

    ``mp_physics`` is ``config_microp_scheme`` and so on, read from the
    table's ``engine_selectors`` block, so a request may be spelled either
    way and means one thing.
    """

    aliases: dict[str, str] = {}
    for key, route in load_table()["engine_selectors"]["selectors"].items():
        aliases[str(route.get("selector", route.get("parameter")))] = key
    return MappingProxyType(aliases)


# ---------------------------------------------------------------------------
# the resolution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedPhysics:
    """One physics request, resolved against the table."""

    suite: str
    suite_kind: str
    mesh: str | None
    selectors: Mapping[str, str]
    origins: Mapping[str, str]
    overrides: Mapping[str, Any]
    generation: Mapping[str, Any]
    frac_seaice: bool
    convection: str
    seam: str
    backend_row: str
    column_profile: Mapping[str, Any] | None
    differences: tuple[str, ...]
    notes: tuple[str, ...] = field(default=())
    #: The RunConfig assignments handed to the engine's generic column seam
    #: (``None`` on the frozen and column-suite seams, which take slot names).
    engine_physics: Mapping[str, Any] | None = None
    #: Engine settings the request named beyond the nine selectors.
    settings: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    #: WOOF's RunConfig assignment each selector's row names, by selector key
    #: (every seam; the cumulus is left to the convection decision).  What a
    #: receipt labels a selection from when no mapping was handed over.
    engine_assignments: Mapping[str, Mapping[str, Any]] = field(
        default_factory=lambda: MappingProxyType({}))

    def is_default(self) -> bool:
        """True exactly when the run is what a bare run selected before
        named suites: the default suite, no override, the frozen row."""

        return (
            self.suite == DEFAULT_SUITE
            and not self.overrides
            and self.backend_row == "wsm6_column"
            and self.convection == "auto"
            and not self.generation
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "suite": self.suite,
            "suite_kind": self.suite_kind,
            "mesh": self.mesh,
            "selectors": dict(self.selectors),
            "origins": dict(self.origins),
            "overrides": dict(self.overrides),
            "generation": dict(self.generation),
            "frac_seaice": self.frac_seaice,
            "convection_request": self.convection,
            "seam": self.seam,
            "backend_row": self.backend_row,
            "column_profile": None if self.column_profile is None else self.column_profile["id"],
            "differences_from_shaped_after": list(self.differences),
            "notes": list(self.notes),
            "engine_physics": None if self.engine_physics is None else dict(self.engine_physics),
            "settings": dict(self.settings),
        }


def _starting_point(suite: str | None, mesh: str | None) -> tuple[str, str, dict, dict, Mapping | None]:
    table = load_table()
    name = DEFAULT_SUITE if suite in (None, "") else str(suite).strip()
    for row in table["suites"]:
        if row["name"] == name:
            selectors = dict(row["selectors"])
            origins = {key: "suite" for key in selectors}
            if mesh is not None and mesh in row.get("selectors_by_mesh", {}):
                for key, value in row["selectors_by_mesh"][mesh].items():
                    selectors[key] = value
                    origins[key] = f"suite, mesh {mesh}"
            return name, "hex", selectors, origins, row
    mpas = table["mpas_suites"]["suites"]
    if name in mpas:
        selectors = dict(mpas[name])
        return name, "mpas", selectors, {key: f"MPAS suite {name}" for key in selectors}, None
    raise PhysicsSelectionRefusal(
        [OptionRefusal(
            "config_physics_suite", name,
            "no suite carries this name, so nothing says which schemes the run "
            f"would execute; hex suites {list(suite_names())}, MPAS suites "
            f"{list(mpas)}",
            "a row in data/profiles/mpas-physics-suites.v1.json",
        )],
        context=name,
    )


def resolve_physics(
    suite: str | None = None,
    overrides: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
    *,
    mesh: str | None = None,
    nominal_dx_m: float | None = None,
    engine_checkout: Any = None,
) -> ResolvedPhysics:
    """Resolve a suite name plus overrides to the seam that runs it.

    ``overrides`` keys are MPAS selector keys (``config_pbl_scheme`` ...),
    WOOF's own selector names (``bl_pbl_physics`` ...), WOOF generation
    selectors (``bl_mynn_version`` ...), ``config_frac_seaice`` and any other
    engine setting.  A selector value is an MPAS spelling from the table, a
    registry option id or the selector's number.  A selector override valued
    ``'suite'`` keeps the suite's value, as MPAS does.  ``mesh`` is the
    registered mesh row name a suite's per-mesh selectors key on.
    ``nominal_dx_m``, the mesh row's declared spacing when the caller knows
    it, lets the ``auto`` cumulus request be decided here.  ``engine_checkout``
    names the engine tree whose registry answers (default: the importable
    ``woof``).  Every refusal is collected and raised together.
    """

    from . import engine_registry

    table = load_table()
    options = table["options"]
    keys = tuple(table["selector_keys"])
    routes = table["engine_selectors"]["selectors"]
    aliases = engine_selector_aliases()
    owned = tuple(table["engine_selectors"]["seam_owned"])
    name, kind, selectors, origins, row = _starting_point(suite, mesh)
    given = dict(overrides.items() if isinstance(overrides, Mapping) else (overrides or ()))
    refusals: list[OptionRefusal] = []
    generation: dict[str, Any] = dict(row.get("generation", {})) if row is not None else {}
    frac_seaice = bool(row.get("frac_seaice", False)) if row is not None else False
    recorded: dict[str, Any] = {}
    settings: dict[str, Any] = {}

    for key, value in given.items():
        if key in keys or key in aliases:
            target = key if key in keys else aliases[key]
            text = str(value).strip()
            if text == "suite":
                recorded[key] = "suite"
                continue
            selectors[target] = text
            origins[target] = "override"
            recorded[key] = text
        elif key in table["generation_selectors"]:
            generation[key] = value
            recorded[key] = value
        elif key == "config_frac_seaice":
            if not isinstance(value, bool):
                refusals.append(OptionRefusal(
                    key, value, "the MPAS Registry type is logical",
                    "config_frac_seaice=true or false"))
                continue
            frac_seaice = value
            recorded[key] = value
        elif table["namelist"]["options"].get(key, {}).get("kind") == "surface_update":
            if not isinstance(value, bool):
                refusals.append(OptionRefusal(key, value, "the MPAS Registry type is logical",
                                              f"{key}=true or false"))
            else:
                recorded[key] = value
        elif key in owned:
            refusals.append(OptionRefusal(
                key, value, str(table["engine_selectors"]["seam_owned_reason"]),
                "the run's own mesh, timestep and cadence flags"))
        else:
            # An engine setting: typed and bounded by the registry's own
            # parameter row when it has one; otherwise handed to the engine,
            # whose plan refuses a key that is not a RunConfig field by name.
            try:
                # THE BREAKAGE THIS REFUSES: an ARW dycore setting (km_opt,
                # diff_6th_opt, epssm) resolved, moved the run onto the
                # generic seam and was recorded in its receipt; hex runs the
                # MPAS-A dycore, so nothing ever read it (2026-10-11).
                dycore = table["engine_selectors"]["dycore_owned"]
                component, reader = engine_registry.setting_reader(key, engine_checkout)
                if component in dycore["components"] or reader in dycore["consuming_reads"]:
                    refusals.append(OptionRefusal(
                        key, value,
                        f"{dycore['reason']} (WOOF reads it in "
                        f"{reader or 'its ' + str(component) + ' component'})",
                        str(dycore["follow_up"])))
                    continue
                typed = engine_registry.setting_value(key, value, engine_checkout)
            except engine_registry.EngineValueRefusal as error:
                refusals.append(OptionRefusal(key, value, error.reason, error.remedy))
                continue
            except engine_registry.EngineRegistryUnavailable as error:
                refusals.append(OptionRefusal(
                    key, value,
                    "it is not one of hex's selector keys, and WOOF's physics "
                    f"registry could not be read to say what it is ({error})",
                    f"one of {list(keys) + list(table['generation_selectors'])}, "
                    "or an engine setting with --gpuwm-checkout naming the engine"))
                continue
            settings[key] = typed
            recorded[key] = typed

    # Each selector: a table spelling, else an option of WOOF's registry.
    rows: dict[str, Mapping[str, Any]] = {}
    notes: list[str] = []
    for key in keys:
        value = selectors[key]
        option = options[key].get(value)
        if option is None:
            try:
                found = engine_registry.resolve_selector_value(
                    routes[key], value, engine_checkout)
            except engine_registry.EngineValueRefusal as error:
                refusals.append(OptionRefusal(
                    key, value,
                    f"{error.reason}; MPAS spellings hex knows: {sorted(options[key])}",
                    error.remedy))
                continue
            except engine_registry.EngineRegistryUnavailable as error:
                refusals.append(OptionRefusal(
                    key, value,
                    f"it is not an MPAS spelling hex knows ({sorted(options[key])}) "
                    f"and WOOF's physics registry could not be read ({error})",
                    "name the engine tree (--gpuwm-checkout) or use a known spelling"))
                continue
            # A registry value that an MPAS spelling already names is that
            # spelling (the first in table order), so one scheme has one
            # name in the receipt however it was asked for.
            spelled = next(
                (text for text, item in options[key].items()
                 if "refused" not in item and all(
                     item.get("engine", {}).get(name) == number
                     for name, number in found.assignments.items())),
                None)
            if spelled is not None:
                selectors[key] = spelled
                option = options[key][spelled]
            else:
                if "component" in routes[key] and "|" not in found.option_id:
                    selectors[key] = found.option_id
                option = MappingProxyType({
                    "engine": dict(found.assignments),
                    "registry_option": found.option_id,
                    "note": f"WOOF registry option {found.option_id}: {found.label}"
                            + (f" (maturity {found.maturity})" if found.maturity else ""),
                })
        if "refused" in option:
            refusals.append(OptionRefusal(key, selectors[key], option["refused"], option["follow_up"]))
            continue
        rows[key] = option
        if "note" in option:
            notes.append(f"{key}={selectors[key]!r}: {option['note']}")

    for component in rows.values():
        for key, value in component.get("carried", {}).get("generation_defaults", {}).items():
            generation.setdefault(key, value)

    #: Why a selection leaves the two slot-named seams: not a refusal, the
    #: reason it is handed to the generic seam instead.
    beyond: list[str] = []
    if not refusals:
        for pairing in table["pairings"]:
            if all(selectors[key] in values for key, values in pairing["if"].items()):
                for key, values in pairing["then"].items():
                    if selectors[key] not in values and pairing.get("judge") == "engine":
                        # A pairing the slot-named seams hold, which is
                        # WOOF's own validators' to judge on the generic one.
                        beyond.append(
                            f"{key}={selectors[key]!r} with "
                            + ", ".join(f"{k}={selectors[k]!r}" for k in pairing["if"]))
                    elif selectors[key] not in values:
                        refusals.append(OptionRefusal(
                            key, selectors[key], pairing["reason"],
                            f"{key} in {list(values)} with "
                            + ", ".join(f"{k}={selectors[k]!r}" for k in pairing["if"]),
                        ))
        # A generation selector belongs to the scheme it is a generation
        # OF, on every seam: on another scheme it would be recorded and
        # never read.
        gen_rows = table["generation_selectors"]
        slot_keys = {
            "microphysics": ("config_microp_scheme", "microphysics"),
            "pbl_scheme": ("config_pbl_scheme", "pbl_scheme"),
            "surface_layer_scheme": ("config_sfclayer_scheme", "surface_layer_scheme"),
            "radiation_scheme": ("config_radt_lw_scheme", "radiation_scheme"),
        }
        for gen, value in generation.items():
            gen_row = gen_rows[gen]
            key, slot = slot_keys[gen_row["slot"]]
            runs = (rows[key].get("carried") or {}).get(slot, selectors[key])
            if value not in gen_row["values"]:
                refusals.append(OptionRefusal(
                    gen, value, f"the engine admits only {list(gen_row['values'])}",
                    f"{gen} in {list(gen_row['values'])}"))
            elif runs not in gen_row["schemes"]:
                refusals.append(OptionRefusal(
                    gen, value,
                    f"it selects a generation of {list(gen_row['schemes'])} and this "
                    f"run's {gen_row['slot']} is {runs!r}, so it "
                    "would be recorded and never read",
                    f"select a {gen_row['slot']} it is a generation of"))

    seam = backend = None
    profile = None
    convection = None
    engine_physics = None
    if not refusals:
        carried = {key: item.get("carried") for key, item in rows.items()}
        if settings:
            beyond.append(f"engine settings {sorted(settings)}")
        beyond.extend(
            f"{key}={selectors[key]!r}" for key in keys if carried[key] is None)
        if not beyond:
            slots = {
                "radiation_scheme": carried["config_radt_lw_scheme"]["radiation_scheme"],
                "surface_layer_scheme": carried["config_sfclayer_scheme"]["surface_layer_scheme"],
                "land_surface_scheme": carried["config_lsm_scheme"]["land_surface_scheme"],
                "pbl_scheme": carried["config_pbl_scheme"]["pbl_scheme"],
            }
            micro = carried["config_microp_scheme"]
            drag = carried["config_gwdo_scheme"]["drag"]
            cloud = carried["config_radt_cld_scheme"]["cloud"]
            convection = carried["config_convection_scheme"]["convection"]
            frozen = table["frozen_seam"]
            if slots["land_surface_scheme"] == "noahmp" and frac_seaice:
                refusals.append(OptionRefusal(
                    "config_frac_seaice", True,
                    "Noah-MP classifies sea ice by its xice_threshold; fractional "
                    "sea ice would be recorded and never read",
                    "config_frac_seaice=false with sf_noahmp"))
            if carried["config_radt_sw_scheme"]["radiation_scheme"] != slots["radiation_scheme"]:
                beyond.append("a longwave and a shortwave of different families")
            elif dict(frozen["selection"]) == slots and cloud == frozen["cloud"]:
                seam = "frozen"
                backend = micro.get("frozen_row")
                if backend is None:
                    beyond.append(
                        f"{selectors['config_microp_scheme']} with the frozen batch's "
                        "surface and boundary-layer schemes")
                if drag != frozen["drag"]:
                    beyond.append(
                        f"config_gwdo_scheme={selectors['config_gwdo_scheme']!r} with "
                        "the frozen batch's schemes")
                if generation:
                    beyond.append(f"generation selectors {sorted(generation)} on the frozen batch's schemes")
            else:
                seam = "column_suite"
                spec = table["column_suite_seam"]
                if drag not in spec["drag"]:
                    beyond.append(
                        f"config_gwdo_scheme={selectors['config_gwdo_scheme']!r} beside "
                        "a selectable boundary layer")
                ruled = convection
                if convection == "auto" and nominal_dx_m is not None:
                    ruled = _auto_convection(float(nominal_dx_m), options)
                if ruled != "auto" and ruled not in spec["convection"]:
                    beyond.append(f"cumulus {ruled!r} behind the column-suite seam")
                if not refusals and not beyond:
                    profile = _column_profile(
                        selectors=selectors, slots=slots, micro=micro, drag=spec["drag"][drag],
                        lsm=carried["config_lsm_scheme"], frac_seaice=frac_seaice,
                        generation=generation, suite=name, row=row,
                    )
                    backend = profile["backend_row"]
        if beyond and not refusals:
            seam = "engine"
            profile, convection, engine_physics, engine_notes = _engine_profile(
                table=table, selectors=selectors, rows=rows, settings=settings,
                generation=generation, frac_seaice=frac_seaice, suite=name, row=row,
                nominal_dx_m=nominal_dx_m, engine_checkout=engine_checkout,
                beyond=beyond, refusals=refusals)
            if profile is not None:
                backend = profile["backend_row"]
                notes.extend(engine_notes)

    if refusals:
        raise PhysicsSelectionRefusal(refusals, context=name)

    differences = tuple(row.get("differences_from_shaped_after", ())) if row is not None else ()
    return ResolvedPhysics(
        suite=name,
        suite_kind=kind,
        mesh=mesh,
        selectors=MappingProxyType(dict(selectors)),
        origins=MappingProxyType(dict(origins)),
        overrides=MappingProxyType(recorded),
        generation=MappingProxyType(dict(generation)),
        frac_seaice=frac_seaice,
        convection=str(convection),
        seam=str(seam),
        backend_row=str(backend),
        column_profile=profile,
        differences=differences,
        notes=tuple(notes),
        engine_physics=None if engine_physics is None else MappingProxyType(dict(engine_physics)),
        settings=MappingProxyType(dict(settings)),
        engine_assignments=MappingProxyType({
            key: MappingProxyType(dict(rows[key].get("engine", {})))
            for key in keys if key in rows and key != "config_convection_scheme"
        }),
    )


# ---------------------------------------------------------------------------
# the generic seam: a selection as WOOF's own RunConfig assignments
# ---------------------------------------------------------------------------
def _engine_profile(*, table, selectors, rows, settings, generation, frac_seaice,
                    suite, row, nominal_dx_m, engine_checkout, beyond, refusals):
    """The profile of a selection the generic column seam runs.

    Returns ``(profile, convection request, physics mapping, notes)``;
    appends to ``refusals`` and returns a ``None`` profile when the selection
    cannot run.  Everything about the selection that is not a NAME comes from
    the engine: the assignments from the registry, the verdict on the
    combination, the species, the soil column and the statics from the
    engine's own plan.
    """

    from . import convection_admission, engine_registry
    from . import cuda_suite_physics_v841 as adapter
    from .species_row import (
        engine_species_row, register_engine_row, registered_species_rows,
    )

    keys = tuple(table["selector_keys"])
    routes = table["engine_selectors"]["selectors"]
    why = "; ".join(beyond)
    physics: dict[str, Any] = {}
    owner: dict[str, str] = {}
    for key in keys:
        if key == "config_convection_scheme":
            continue
        for field_name, value in rows[key]["engine"].items():
            if field_name in physics and physics[field_name] != value:
                refusals.append(OptionRefusal(
                    key, selectors[key],
                    f"it assigns WOOF's {field_name}={value!r} and "
                    f"{owner[field_name]}={selectors[owner[field_name]]!r} assigns "
                    f"{physics[field_name]!r}; one run has one {field_name}",
                    f"selections that agree on {field_name}"))
                continue
            physics[field_name] = value
            owner[field_name] = key
    for field_name, value in {**generation, **settings}.items():
        if field_name in physics and physics[field_name] != value:
            refusals.append(OptionRefusal(
                field_name, value,
                f"{owner[field_name]}={selectors[owner[field_name]]!r} already assigns "
                f"WOOF's {field_name}={physics[field_name]!r}",
                f"one value for {field_name}"))
            continue
        physics[field_name] = value
    land = rows["config_lsm_scheme"]
    if frac_seaice:
        if (land.get("carried") or {}).get("land_surface_scheme") == "noahmp":
            refusals.append(OptionRefusal(
                "config_frac_seaice", True,
                "Noah-MP classifies sea ice by its xice_threshold; fractional "
                "sea ice would be recorded and never read",
                "config_frac_seaice=false with sf_noahmp"))
        else:
            physics["fractional_seaice"] = 1

    # The cumulus: the hex resolution rule decides ``auto`` at the mesh bind;
    # an explicit scheme is WOOF's cu_physics, registered so the config name,
    # the constructor name and the receipt label exist for it.
    cumulus = rows["config_convection_scheme"]
    carried_cu = (cumulus.get("carried") or {}).get("convection")
    plan_physics = dict(physics)
    if carried_cu is not None:
        convection = carried_cu
        decided = carried_cu
        if carried_cu == "auto" and nominal_dx_m is not None:
            decided = _auto_convection(float(nominal_dx_m), table["options"])
        if decided != "auto":
            plan_physics["cu_physics"] = convection_admission.cu_physics_for_constructor_scheme(
                None if decided == "off" else decided)
    else:
        number = int(cumulus["engine"]["cu_physics"])
        label = selectors["config_convection_scheme"]
        option_id = cumulus.get("registry_option")
        try:
            ids = engine_registry.component_values(
                "cumulus", "cu_physics", engine_checkout).get(number, ())
            if ids:
                option_id = ids[0]
                label = str(engine_registry.implemented_options(
                    "cumulus", engine_checkout)[ids[0]].get("label", ids[0]))
        except engine_registry.EngineRegistryUnavailable:
            pass
        convection = str(option_id or f"cu_physics_{number}")
        convection_admission.register_engine_scheme(
            selectors["config_convection_scheme"], constructor_name=convection,
            cu_physics=number, label=label)
        plan_physics["cu_physics"] = number
    if refusals:
        return None, None, None, ()

    problem = engine_registry.generic_seam_problem(engine_checkout)
    if problem is not None:
        refusals.append(OptionRefusal(
            "physics", why,
            f"this selection ({why}) runs on the engine's generic column seam, "
            f"and {problem}",
            "an engine tree carrying woof.core.mpas_column_suite."
            f"{engine_registry.GENERIC_SEAM_ENTRYPOINT} (--gpuwm-checkout)"))
        return None, None, None, ()
    try:
        plan = engine_registry.describe_selection(plan_physics, engine_checkout)
    except (ValueError, TypeError, KeyError) as error:
        # A refusal that names one setting the request gave is that
        # setting's; otherwise it is the combination's.
        named = next((name for name in settings if repr(name) in str(error)), None)
        refusals.append(OptionRefusal(
            named or "physics",
            settings[named] if named else ", ".join(
                f"{key}={plan_physics[key]!r}" for key in sorted(plan_physics)),
            f"WOOF itself refuses this combination: {error}",
            "a combination woof's validate_run_config and "
            "validate_physics_capabilities accept"))
        return None, None, None, ()
    if plan is None:
        refusals.append(OptionRefusal(
            "physics", why,
            "the engine's generic column seam could not be imported in this "
            "process to plan the selection, so its species and soil column are "
            "unknown and the scalar block could not be sized",
            "run with the engine tree importable (PYTHONPATH or --gpuwm-checkout)"))
        return None, None, None, ()

    species = tuple(str(item) for item in plan["species"])
    masses = tuple(str(item) for item in plan["mass_species"])
    mp_number = int(dict(plan.get("physics", {})).get("mp_physics", physics.get("mp_physics", 0)))
    scheme = selectors["config_microp_scheme"]
    declared = registered_species_rows().get(scheme)
    if declared is not None and set(declared.names()) != set(species):
        scheme = f"{scheme}@{rows['config_microp_scheme'].get('registry_option', mp_number)}"
        declared = None
    if declared is None:
        state_fields = tuple(str(item) for item in plan.get("state_fields", ()))
        declared = register_engine_row(engine_species_row(
            scheme_name=scheme, mp_physics=mp_number, species=species,
            mass_species=masses,
            surface_accumulators=plan.get("surface_accumulators"),
            radius_names=tuple(
                str(n) for n in plan.get(
                    "radii", tuple(n for n in ("effc", "effi", "effs") if n in state_fields))),
            history_names=table["engine_selectors"]["species_history_names"],
        ))

    statics = tuple(str(item) for item in plan.get("static_fields", ()))
    oro = None
    if statics:
        oro = _statics_spec(statics, table)
        if oro is None:
            refusals.append(OptionRefusal(
                "physics", why,
                f"missing static input: the selection reads the per-cell statics "
                f"{list(statics)}, and no hex mesh file stages that set (the "
                "static file carries the YSU drag's ten, ugwp_oro_data the GSL "
                "drag's twenty)",
                "a statics source row for them in engine_selectors.statics"))
            return None, None, None, ()

    n_soil = int(plan["n_soil_levels"])
    depths = land.get("soil_level_depths_m", (land.get("carried") or {}).get("soil_level_depths_m"))
    if depths is not None and len(depths) != n_soil:
        depths = None
    slot_name = {
        key: ((rows[key].get("carried") or {}).get(slot) or selectors[key])
        for key, slot in (
            ("config_sfclayer_scheme", "surface_layer_scheme"),
            ("config_lsm_scheme", "land_surface_scheme"),
            ("config_pbl_scheme", "pbl_scheme"),
        )
    }
    lw = (rows["config_radt_lw_scheme"].get("carried") or {}).get("radiation_scheme")
    sw = (rows["config_radt_sw_scheme"].get("carried") or {}).get("radiation_scheme")
    radiation = lw if lw is not None and lw == sw else (
        f"{selectors['config_radt_lw_scheme']}+{selectors['config_radt_sw_scheme']}")
    drag = {0: "off", 1: "ysu_gwdo", 3: "gsl"}.get(
        int(physics.get("gwd_opt", 0)), f"gwd_opt={physics.get('gwd_opt')}")
    selection = {
        "radiation_scheme": radiation,
        "surface_layer_scheme": slot_name["config_sfclayer_scheme"],
        "land_surface_scheme": slot_name["config_lsm_scheme"],
        "pbl_scheme": slot_name["config_pbl_scheme"],
        "drag_scheme": drag,
    }
    differences = list(row.get("differences_from_shaped_after", ())) if row is not None else []
    key = json.dumps(
        {"physics": {k: physics[k] for k in sorted(physics)}, "scheme": scheme,
         "soil": n_soil, "suite": suite, "differences": differences},
        sort_keys=True, default=str)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    profile = {
        "id": f"engine-{digest}",
        "backend_row": f"engine_{digest}",
        "describes": (
            f"resolved for suite {suite!r} from WOOF's physics registry: "
            + ", ".join(f"{k}={physics[k]!r}" for k in sorted(physics))),
        "engine_batch": {
            "module": engine_registry.GENERIC_SEAM_MODULE,
            "entrypoint": engine_registry.GENERIC_SEAM_ENTRYPOINT,
        },
        "selection": selection,
        "physics": {k: physics[k] for k in sorted(physics)},
        "microphysics": declared.scheme_name,
        "species": list(declared.names()),
        "n_soil_levels": n_soil,
        "soil_level_depths_m": None if depths is None else list(depths),
        "fractional_seaice": physics.get("fractional_seaice"),
        "mpas_selectors": {
            k: selectors[k] for k in keys if k != "config_convection_scheme"},
        "generation": dict(generation),
        "orographic_statistics": oro,
        "differences_from_shaped_after": differences,
        "synthesized": True,
        "generic": True,
    }
    adapter.register_profile(profile)
    _register_backend_row(profile)
    notes = (
        f"generic column seam ({engine_registry.GENERIC_SEAM_ENTRYPOINT}): {why}",
        "physics on hex is graded against observations (MRMS, ASOS); this "
        "selection holds no anchor until it has been scored",
    )
    return MappingProxyType(profile), convection, physics, notes


def _statics_spec(fields: tuple[str, ...], table: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The mesh file layout that carries exactly these engine statics."""

    wanted = set(fields)
    sources = [_thaw(item) for item in table["engine_selectors"]["statics"]]
    try:
        gsl = dict(_gsl_statistics_spec())
        gsl.setdefault("file", "ugwp_oro_data")
        sources.append(gsl)
    except PhysicsTableError:
        pass
    for source in sources:
        if wanted == set(source["variables"]):
            return source
    return None


def _auto_convection(nominal_dx_m: float, options: Mapping[str, Any]) -> str:
    """The seam's convection value the ``auto`` rule selects at this spacing.

    The rule is :func:`woof.hex.convection_admission.convection_for_spacing`
    (an MPAS spelling); the table's option row turns it into the value the
    seam specs are keyed by.
    """

    from .convection_admission import convection_for_spacing

    spelled = convection_for_spacing(nominal_dx_m)
    return str(options["config_convection_scheme"][spelled]["carried"]["convection"])


# ---------------------------------------------------------------------------
# column-suite profiles: a profile row when one exists, else synthesized
# ---------------------------------------------------------------------------
def _column_profile(*, selectors, slots, micro, drag, lsm, frac_seaice, generation, suite, row):
    from . import cuda_suite_physics_v841 as adapter

    selection = dict(slots, drag_scheme=drag)
    n_soil = int(lsm["n_soil_levels"])
    seaice = None if slots["land_surface_scheme"] == "noahmp" else int(bool(frac_seaice))
    mpas = {key: selectors[key] for key in selector_keys() if key != "config_convection_scheme"}
    for existing in adapter.load_physics_suites().values():
        if (
            dict(existing["selection"]) == selection
            and existing["microphysics"] == next(
                name for name, value in _microphysics_spellings().items()
                if value == micro["microphysics"])
            and int(existing["n_soil_levels"]) == n_soil
            and existing["fractional_seaice"] == seaice
            and dict(existing.get("generation", {})) == dict(generation)
        ):
            return existing
    depths = lsm.get("soil_level_depths_m")
    if depths is None:
        raise PhysicsSelectionRefusal(
            [OptionRefusal(
                "config_lsm_scheme", selectors["config_lsm_scheme"],
                "the column-suite seam checks the init's soil depths against the "
                "land surface's own, and this land surface's row names none, so a "
                "start interpolated to other depths would run without error",
                "soil_level_depths_m in the land surface's option row")],
            context=suite)
    oro = None
    if drag == "gsl":
        oro = _gsl_statistics_spec()
    key = json.dumps(
        {"selection": selection, "micro": micro["microphysics"], "soil": n_soil,
         "seaice": seaice, "generation": dict(sorted(generation.items())),
         "suite": suite, "differences": list(row.get("differences_from_shaped_after", ()))
         if row is not None else []},
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    profile = {
        "id": f"column-{digest}",
        "backend_row": f"column_suite_{digest}",
        "describes": (
            f"synthesized for suite {suite!r} from the option rows: "
            + ", ".join(f"{k}={v}" for k, v in sorted(selection.items()))
            + f", microphysics {micro['microphysics']}"
        ),
        "engine_batch": {
            "module": "woof.core.mpas_column_suite",
            "entrypoint": "run_mpas_column_suite",
        },
        "selection": selection,
        "microphysics": next(
            name for name, value in _microphysics_spellings().items()
            if value == micro["microphysics"]),
        "n_soil_levels": n_soil,
        "soil_level_depths_m": list(depths),
        "fractional_seaice": seaice,
        "mpas_selectors": mpas,
        "generation": dict(generation),
        "orographic_statistics": oro,
        "differences_from_shaped_after": list(
            row.get("differences_from_shaped_after", ()) if row is not None else ()),
        "synthesized": True,
    }
    adapter.register_profile(profile)
    _register_backend_row(profile)
    return MappingProxyType(profile)


def _microphysics_spellings() -> Mapping[str, str]:
    """MPAS microphysics spelling -> engine scheme, from the carried rows."""

    rows = load_table()["options"]["config_microp_scheme"]
    return {name: row["carried"]["microphysics"] for name, row in rows.items() if "carried" in row}


def _gsl_statistics_spec() -> Mapping[str, Any]:
    """The GSL drag's statistics file layout, from the profile table that
    already carries it (the ``mynn-ruc-gsl`` row); one copy, read here."""

    from . import cuda_suite_physics_v841 as adapter

    for existing in adapter.load_physics_suites().values():
        if existing["selection"]["drag_scheme"] == "gsl":
            return _thaw(existing["orographic_statistics"])
    raise PhysicsTableError("no physics-suites.v1.json profile carries the GSL statistics layout")


def _register_backend_row(profile: Mapping[str, Any]) -> None:
    from .physics_backend_admission import PhysicsBackendRow, register_synthesized_row
    from .species_row import species_row_for_scheme

    species = species_row_for_scheme(profile["microphysics"])
    names = tuple(species.names())
    # The driven block is the leading run of the six masses and then the hydrometeor numbers a boundary stream may
    # carry (the in-tree Thompson rows' split, woof.hex.physics_backend_admission): a stream drives the prefix of it
    # it supplies, so masses entering at the rings bring their numbers when the stream has them.
    drivable = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc")
    lead = 0
    while lead < len(names) and names[lead] in drivable:
        lead += 1
    driven, appended = names[:lead], names[lead:]
    register_synthesized_row(PhysicsBackendRow(
        name=profile["backend_row"],
        adapter_module="woof.hex.cuda_suite_physics_v841",
        batch_module=str(profile["engine_batch"]["module"]),
        batch_entrypoint=str(profile["engine_batch"]["entrypoint"]),
        driven_scalar_names=driven,
        appended_scalar_names=appended,
        anchor_configuration_class=f"v841-column-suite-{profile['id']}",
        anchored=False,
        unanchored_remedy=(
            "Earn it with this selection's own 1, 2 and 4 card byte identity on a "
            "registered mesh and a scored forecast against MRMS and ASOS."
        ),
    ))
