"""What physics WOOF can run, what each choice means, and whether a mix runs.

One engine-owned answer for every front end (``woof physics-catalog``,
``GET /api/physics`` and ``POST /api/physics/check`` in ``woof gui``, and
an assistant that composes a forecast for someone):

* :func:`catalog` lists every family and every scheme the registry has,
  with a plain one-line description, its cost against the default suite,
  where it is valid (grid spacing, vertical levels, card size), the named
  suites that carry it, and the presets by intention.
* :func:`check` takes any combination and answers valid, or the refusal
  of the engine door that stopped it, word for word, with the nearest
  combinations the same doors admit.

Nothing here is a second rule table.  The schemes, couplings, maturity
and suites are read from ``physics_registry_v2.json``; a combination is
judged by building a :class:`woof.config.RunConfig` and passing it
through the doors a real run passes through
(:func:`woof.config.validate_run_config`,
:func:`woof.physics_compat.validate_physics_capabilities`, the
radiation-off land-surface refusal and the source route's emission gate);
cost comes from the measured step rates in :mod:`woof.core.pace` and says
"unmeasured" where no measurement exists; grid-spacing advice is the
wizard's own gray-zone sentences sampled on a ladder of spacings.

The only words written here live in ``physics_catalog_v1.json``: one
description per scheme and the presets.  A new scheme is one description
row; a new preset is one row naming a registered suite.
"""
from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
import itertools
import json
import re
from pathlib import Path
from typing import Any, Mapping

TABLE_PATH = Path(__file__).with_name("physics_catalog_v1.json")
TABLE_SCHEMA = "gpuwm.physics-catalog-table.v1"
SCHEMA = "gpuwm.physics-catalog.v1"
CHECK_SCHEMA = "gpuwm.physics-check.v1"

#: Grid spacings the gray-zone advice is sampled at.  Sample points, not
#: thresholds: the thresholds are the wizard's own constants, and a
#: spacing here only decides where the catalog asks them.
DX_LADDER_KM = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 12.0, 25.0)

#: The probe grid a combination is checked on when the caller names no
#: size: small enough to be instant, the reference level count of the
#: measured step rates.
PROBE_COLUMNS_SIDE = 100
PROBE_NZ = 49
PROBE_DX_KM = 3.0

#: Families checked together when searching for the nearest valid
#: combination, beyond one-family swaps: the registry's coupling laws
#: join exactly these (``requires_components`` and ``refused_when`` rows
#: name a sibling in one of these groups).  A coupling outside them is
#: still found by the one- and two-family sweep.
COUPLED_GROUPS = (
    ("pbl", "surface_layer"),
    ("pbl", "turbulence"),
    ("pbl", "surface_layer", "turbulence"),
    ("microphysics", "radiation"),
    ("cumulus", "pbl"),
    ("land_surface", "radiation"),
    ("urban", "land_surface", "pbl"),
)

NEIGHBOUR_LIMIT = 5


class CatalogError(ValueError):
    """A malformed request, said in plain words."""


# ------------------------------------------------------------------ inputs

@lru_cache(maxsize=1)
def table() -> dict[str, Any]:
    document = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    if document.get("schema") != TABLE_SCHEMA:
        raise ValueError(f"{TABLE_PATH.name}: expected schema {TABLE_SCHEMA!r}")
    # Curated prose remains authoritative. A new scheme already declares its
    # name and scope in the registry; staging it does not require a duplicate
    # catalog option row. This fallback makes no cost or qualification claim.
    descriptions = document["descriptions"]
    for family, component in _registry()["components"].items():
        rows = descriptions.setdefault(family, {})
        for option_id, option in component["options"].items():
            if option_id not in rows:
                words = option.get("description") or option.get("label")
                if not isinstance(words, str) or not words.strip():
                    raise ValueError(
                        f"{family}.{option_id}: registry needs a description or label")
                rows[option_id] = words
    return document


@lru_cache(maxsize=1)
def _registry() -> dict[str, Any]:
    """The registry, read once: this module never writes to it."""

    from woof.physics_registry import physics_registry

    return physics_registry()


@lru_cache(maxsize=1)
def _menu() -> dict[str, dict[str, dict[str, Any]]]:
    """family -> choice id -> the engine's own settings row for it.

    Implemented options come from
    :func:`woof.companion_domains.physics_components`, the payload the
    configuration editor saves through, so a choice applied here is the
    same edit a saved file carries (radiation fans out into its engine
    variants there).  An option the registry declares but does not
    implement gets its selectors only, so the capability door refuses it
    with the registry's own blocker instead of this module deciding.
    """

    from woof.companion_domains import physics_components

    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for component in physics_components():
        rows[component["id"]] = {option["id"]: option for option in component["options"]}
    for family, component in _registry()["components"].items():
        for option_id, option in component["options"].items():
            family_rows = rows.setdefault(family, {})
            if any(row["registry_option_id"] == option_id for row in family_rows.values()):
                continue
            family_rows[option_id] = {
                "id": option_id, "registry_option_id": option_id,
                "label": option.get("label", option_id),
                "selectors": dict(option.get("selectors") or {}),
                "settings": dict(option.get("selectors") or {}),
            }
    return rows


def _run_fields() -> set[str]:
    from woof.config import RunConfig

    return {item.name for item in fields(RunConfig)}


def _run_default(key: str) -> Any:
    """What a run takes for ``key`` when no table of the file states it."""

    from woof.config import RunConfig

    return next(item.default for item in fields(RunConfig) if item.name == key)


#: The urban canopy keys (woof/config.py), which a mix leaves out of a file
#: that never stated them while urban physics stays at its default, off.
_URBAN_RUN_KEYS = ("sf_urban_physics", "use_wudapt_lcz", "num_urban_hi")


#: The switches a nest running no cumulus keeps when a mix is written
#: into its file: the mix's cumulus is the root's.
_NEST_CUMULUS = ("cu_physics", "cudt_minutes")


def default_source() -> str:
    from woof.domain_wizard import DEFAULT_WIZARD_SOURCE

    return DEFAULT_WIZARD_SOURCE


def known_source(source: str | None) -> str | None:
    """``source`` when a registered source has that id; refused otherwise.

    An unknown id used to fall back to the default suite and skip the
    route check, so the answer was about some other source.
    """

    if not source:
        return None
    from woof.physics_menu import registered_sources

    names = registered_sources()
    if source not in names:
        raise CatalogError(f"No data source called {source!r}. Sources: {', '.join(names)}.")
    return source


def default_suite(source: str | None = None, finest_dx_m: float | None = None,
                  domains: int = 1) -> str:
    """The suite a run of ``source`` gets with none named, at its finest grid.

    The call `woof domain` binds (:func:`woof.physics_menu.default_profile_for`),
    so a check that names no suite describes the suite the run carries.  With no
    spacing it is the source's own default, as the catalog's table shows it.
    """

    from woof.physics_menu import default_profile_for

    return default_profile_for(source or default_source(), finest_dx_m, domains)


def _request_grid(request: Mapping[str, Any]) -> tuple[float, int]:
    """The finest grid spacing in metres and the domain count a request asks about.

    ``finest_dx_km`` and ``domains`` describe a run with nests: the finest grid
    and how many grids there are.  Without them the request is one domain at
    ``dx_km`` (the probe spacing when that is absent too).  The default suite
    is read at this grid, which is what `woof domain` binds: on 2.8.1 before
    this the check named the source's own default on a 750 m grid that runs the
    sub-km suite, so New forecast's Physics step described physics the run did
    not have.
    """

    try:
        dx_km = float(request.get("dx_km") or PROBE_DX_KM)
        finest_km = float(request.get("finest_dx_km") or dx_km)
        domains = request.get("domains") or 1
        if isinstance(domains, bool) or int(domains) != float(domains):
            raise ValueError(domains)
        domains = int(domains)
    except (TypeError, ValueError) as error:
        raise CatalogError("dx_km and finest_dx_km must be numbers and domains a whole number.") from error
    if not 0.01 <= finest_km <= 200.0:
        raise CatalogError("finest_dx_km must be between 0.01 and 200.")
    if domains < 1:
        raise CatalogError("domains must be 1 or more.")
    return finest_km * 1000.0, domains


def request_default_suite(request: Mapping[str, Any]) -> str:
    """The suite a check request runs when it names none: the default at its grid."""

    finest_m, domains = _request_grid(request)
    return default_suite(request.get("source") or None, finest_m, domains)


def experiment_grid(text: str) -> dict[str, Any]:
    """An experiment file's grid as check request keys: ``dx_km``, ``finest_dx_km`` and ``domains``.

    The root's spacing is its ``[[domain]]`` table's ``dx``; a nest states its
    own or divides its parent's by ``parent_grid_ratio``, as `woof domain`
    writes them.  Empty for a file with no domain tables or no root spacing.
    """

    import tomllib

    tables = tomllib.loads(text).get("domain") or []
    if not tables or not tables[0].get("dx"):
        return {}
    spacing: dict[int, float] = {}
    for index, table in enumerate(tables):
        grid_id = int(table.get("grid_id") or index + 1)
        if table.get("dx"):
            spacing[grid_id] = float(table["dx"])
            continue
        parent = spacing.get(int(table.get("parent_id") or index))
        ratio = int(table.get("parent_grid_ratio") or 0)
        if parent is None or ratio < 1:
            return {}
        spacing[grid_id] = parent / ratio
    return {"dx_km": float(tables[0]["dx"]) / 1000.0,
            "finest_dx_km": min(spacing.values()) / 1000.0, "domains": len(tables)}


def _suite_switches(suite: str) -> dict[str, Any]:
    from woof.physics_compat import single_domain_runtime_switches

    return dict(single_domain_runtime_switches(suite))


def _template_components(suite: str) -> dict[str, str] | None:
    template = _registry()["templates"].get(suite)
    return dict(template["components"]) if template else None


# ------------------------------------------------------------------ probing

def _probe(settings: Mapping[str, Any], *, dx_km: float, nz: int,
           columns_side: int = PROBE_COLUMNS_SIDE):
    """A RunConfig carrying ``settings`` on a small grid at ``dx_km``."""

    from woof.config import RunConfig

    known = _run_fields()
    dx = float(dx_km) * 1000.0
    return RunConfig(nx=columns_side, ny=columns_side, nz=int(nz), dx=dx, dy=dx,
                     ztop=20000.0, dt=max(0.5, 6.0 * float(dx_km)),
                     run_seconds=3600.0,
                     **{key: value for key, value in settings.items() if key in known})


def _first_refusal(cfg, *, source: str | None, dx_km: float,
                   window: Mapping[str, Any] | None = None,
                   domains: int = 1) -> dict[str, Any] | None:
    """The first engine door that refuses ``cfg``, in the order a run meets them.

    ``window`` (start time, run seconds, reference latitude and longitude)
    adds the load-time nocturnal-radiation door, which only a place and a
    clock can ask.  ``domains`` is how many grids the run has: the source's
    route is asked for that many, as `woof domain` asks it.
    """

    from woof.config import validate_run_config
    from woof.physics_compat import (
        PhysicsCapabilityError, identify_single_domain_profile,
        profile_declared_acknowledgements, radiation_off_land_surface_refusal,
        validate_physics_capabilities)
    from woof.physics_menu import switch_route_blocker

    doors = (
        ("configuration", "woof.config.validate_run_config",
         lambda: validate_run_config(cfg)),
        ("capability", "woof.physics_compat.validate_physics_capabilities",
         lambda: validate_physics_capabilities(cfg)),
    )
    for door, owner, call in doors:
        try:
            call()
        except (PhysicsCapabilityError, ValueError, NotImplementedError, KeyError) as error:
            return {"door": door, "owner": owner, "message": str(error).strip("'\"")}
    suite = identify_single_domain_profile(cfg)
    acknowledgements = profile_declared_acknowledgements(suite) if suite else ()
    refusal = radiation_off_land_surface_refusal(
        [cfg], acknowledgements=tuple(acknowledgements),
        declared_selectors=("ra_lw_physics", "ra_sw_physics"))
    if refusal:
        return {"door": "radiation-off-land-surface",
                "owner": "woof.physics_compat.radiation_off_land_surface_refusal",
                "message": refusal}
    if window is not None:
        from woof.physics_compat import nocturnal_radiation_refusal

        night = nocturnal_radiation_refusal(
            [cfg], start_time=window["start_time"], run_seconds=window["run_seconds"],
            ref_lat=window["ref_lat"], ref_lon=window["ref_lon"],
            acknowledgements=tuple(acknowledgements))
        if night:
            return {"door": "nocturnal-radiation",
                    "owner": "woof.physics_compat.nocturnal_radiation_refusal",
                    "message": night}
    if source:
        blocker = switch_route_blocker(vars(cfg), source, domains=domains)
        if blocker:
            return {"door": "source-route", "owner": "woof.physics_menu.switch_route_blocker",
                    "message": blocker}
    return None


def _headline(sentence: str) -> str:
    """The first sentence of an engine sentence, without the per-grid counts.

    The wizard's advisories name how many domains and the finest spacing;
    for one probe grid both are noise, and dropping them is what lets the
    same advice at 2 km and at 3 km read as one line.
    """

    from woof.explain import split

    head = split(sentence)[0].strip()
    lines = [line.strip(" -") for line in head.splitlines() if line.strip()]
    if len(lines) > 1:
        head = lines[0].rstrip(":") + ": " + lines[1]
    head = re.sub(r"\s*\(finest [^)]*\)", "", head)
    head = re.sub(r"\b\d+ domain\(s\) ", "", head)
    for mark in (" -- ", "; ", ". "):
        if mark in head:
            head = head.split(mark)[0]
    while head.count("(") > head.count(")"):
        # The cut fell inside a parenthesis; the aside goes, the clause stays.
        head = head[:head.rfind("(")]
    head = head.rstrip(" .,")
    first = head.split(" ", 1)[0]
    if first.isalpha() and first.islower():
        head = head[:1].upper() + head[1:]
    return head + "."


def _advisories(settings: Mapping[str, Any], dx_km: float) -> list[str]:
    from woof.domain_wizard import cumulus_gray_zone_advisory, gray_zone_advisory

    lines = list(gray_zone_advisory([dx_km], dict(settings)))
    lines += cumulus_gray_zone_advisory([dx_km], [int(settings.get("cu_physics", 0) or 0)])
    return lines


def _scheme_named(family: str, switch: str, settings: Mapping[str, Any]) -> str:
    """The plain name of the scheme ``settings`` select for ``family``."""

    value = settings.get(switch)
    for row in _menu().get(family, {}).values():
        if row["settings"].get(switch) == value:
            return str(row.get("label") or value)
    return f"{switch} = {value}"


def _plain_advisory(sentence: str, settings: Mapping[str, Any], dx_km: float) -> str:
    """One engine advisory as a plain sentence a person reads on a page.

    The engine's sentences open with a tag (``CUMULUS:``, ``GRAY ZONE:``)
    and name switches by number, which is right in a generated file and
    wrong on a page.  The tag says which advice it is; the sentence here
    says the same thing with the schemes named.  An advisory with a tag
    this table does not hold keeps the engine's first clause, untagged.
    """

    tag, _, _ = sentence.partition(":")
    at = f"At {dx_km:g} km"
    if tag == "CUMULUS GRAY ZONE":
        return (f"{at} the grid is in the gray zone for storms: they are neither fully resolved nor small enough "
                f"for {_scheme_named('cumulus', 'cu_physics', settings)} to handle alone. Many forecast models run "
                "this pairing on purpose.")
    if tag == "CUMULUS" and "counted twice" in sentence:
        return (f"{at} the grid resolves storms itself, and {_scheme_named('cumulus', 'cu_physics', settings)} "
                "also runs, so storm heating and rain can be counted twice.")
    if tag == "CUMULUS":
        return (f"{at} the grid resolves storms itself. {_scheme_named('cumulus', 'cu_physics', settings)} scales "
                "its own share down as the grid resolves them, so they are not counted twice.")
    if tag == "GRAY ZONE":
        return (f"{at} the grid partly resolves the eddies in the boundary layer, while "
                f"{_scheme_named('pbl', 'bl_pbl_physics', settings)} treats them as too small to see. A 3-D "
                "turbulence scheme with the boundary layer scheme off suits this spacing.")
    head = _headline(sentence)
    return head.split(": ", 1)[1] if head.split(":", 1)[0].isupper() and ": " in head else head


# ------------------------------------------------------------------ cost and card

def _cost(cfg, reference_cfg) -> dict[str, Any]:
    """This config's step rate against the reference's, measured or not."""

    from woof.core.pace import step_rate
    from tilestream.autoplan import rung_of

    rung, reference = rung_of(cfg), rung_of(reference_cfg)
    rate, base = step_rate(rung), step_rate(reference)
    row: dict[str, Any] = {"rung": rung, "reference_rung": reference}
    if rate is None or base is None:
        row.update(relative=None, measured=False, words="Unmeasured.")
        return row
    measured = bool(rate.measured and base.measured)
    if rung == reference:
        row.update(relative=1.0, measured=False,
                   words="Unmeasured for this scheme. Planned at the default's rate.")
        return row
    relative = round(rate.low / base.low, 2)
    row["relative"] = relative
    row["measured"] = measured
    row["words"] = (f"{relative:g} times the default's step cost, measured on the {rung} rung, "
                    "which the planner prices every such combination at." if measured else
                    f"Unmeasured. Planned at up to {relative:g} times the default's step cost.")
    row["basis"] = rate.basis
    return row


@lru_cache(maxsize=16)
def _machine(card: str):
    from woof.domain_wizard import resolve_sizing_budget
    from tilestream.autoplan import Machine

    sizing = resolve_sizing_budget(card, None)
    return Machine(vram_bytes=int(sizing.free_bytes), host_bytes=64 << 30, name=card)


def cards() -> tuple[str, ...]:
    from woof.domain_wizard import CARD_VRAM_GIB

    return tuple(sorted(CARD_VRAM_GIB, key=lambda name: CARD_VRAM_GIB[name]))


def _card_fit(cfg) -> dict[str, Any]:
    """Largest grid each card holds on the resident road at this level count."""

    from woof.core.pace import resident_column_limit

    fit = {}
    for card in cards():
        try:
            columns = resident_column_limit(cfg, _machine(card))
        except Exception:  # noqa: BLE001 - an unpriced card is reported as unpriced
            columns = None
        side = int(columns ** 0.5) if columns else None
        fit[card] = {"columns": columns, "square_side": side,
                     "words": (f"about {side} by {side} points at {cfg.nz} levels before tiling"
                               if side else "not priced")}
    return fit


# ------------------------------------------------------------------ catalog

def _scheme_base(family: str, option_id: str, menu_id: str) -> tuple[str | None, dict[str, Any]]:
    """The settings a scheme is priced and placed in.

    The default suite with only this scheme swapped in, when the engine
    admits that, so the cost and card answers are about the scheme and
    not about the rest of some suite.  A scheme that cannot sit in the
    default (MYJ needs its own surface layer) is described in the first
    Create-page suite that carries it, and the reply names that suite.
    """

    from woof.physics_menu import shipped_profiles

    settings = _suite_switches(default_suite())
    settings.update(_menu()[family][menu_id]["settings"])
    cfg = _probe(settings, dx_km=PROBE_DX_KM, nz=PROBE_NZ)
    if _first_refusal(cfg, source=None, dx_km=PROBE_DX_KM) is None:
        return default_suite(), settings
    for suite in shipped_profiles():
        components = _template_components(suite) or {}
        if components.get(family) == option_id:
            return suite, _suite_switches(suite)
    return None, settings


def _resolution(settings: Mapping[str, Any], switch_keys) -> dict[str, Any]:
    """Where the wizard's own grid-spacing advice speaks about this family.

    An advisory belongs to a family when it names one of the family's
    switches (``bl_pbl_physics``, ``cu_physics``): the sentences are the
    engine's, and so is the switch they name.
    """

    quiet, advised = [], {}
    for dx in DX_LADDER_KM:
        lines = [_headline(line) for line in _advisories(settings, dx)]
        lines = [line for line in lines if any(key in line for key in switch_keys)]
        if not lines:
            quiet.append(dx)
            continue
        advised.setdefault(lines[0], []).append(dx)
    return {"quiet_at_dx_km": quiet,
            "advisories": [{"dx_km": spacings, "headline": text} for text, spacings in advised.items()],
            "basis": "woof.domain_wizard gray_zone_advisory and cumulus_gray_zone_advisory, "
                     f"asked at {', '.join(f'{dx:g}' for dx in DX_LADDER_KM)} km"}


def _levels(option: Mapping[str, Any]) -> dict[str, Any] | None:
    bounds = (option.get("consumers") or {}).get("vertical_level_bounds")
    if not isinstance(bounds, Mapping):
        return None
    return {"minimum": bounds.get("minimum"), "maximum": bounds.get("maximum")}


def _streams(selectors: Mapping[str, Any]) -> dict[str, Any] | None:
    if "ra_lw_physics" not in selectors:
        return None
    return {"longwave": selectors.get("ra_lw_physics"), "shortwave": selectors.get("ra_sw_physics")}


def catalog(*, source: str | None = None) -> dict[str, Any]:
    """Every family, every scheme, every suite, every preset."""

    from woof.physics_menu import (profile_facts, shipped_profiles,
                                    spacing_default_menu_rows)
    from woof.physics_registry import registry_sha256

    source = known_source(source)
    words = table()
    registry = _registry()
    reference_suite = default_suite(source)
    reference_cfg = _probe(_suite_switches(reference_suite), dx_km=PROBE_DX_KM, nz=PROBE_NZ)
    reference_components = _template_components(reference_suite) or {}
    shipped = set(shipped_profiles())
    families = []
    order = list(words["family_order"]) + [f for f in registry["components"] if f not in words["family_order"]]
    for family in order:
        component = registry["components"][family]
        family_words = words["families"].get(family, {})
        schemes = []
        for option_id, option in component["options"].items():
            menu_rows = [row for row in _menu()[family].values() if row["registry_option_id"] == option_id]
            variants = []
            for row in menu_rows:
                variant = row["id"].split(":", 1)[1] if ":" in row["id"] else None
                if variant is None:
                    continue
                variants.append({"choice": row["id"], "label": row["label"],
                                 "description": words.get("variant_descriptions", {}).get(family, {}).get(row["id"], "")})
            first = menu_rows[0]["id"] if menu_rows else option_id
            implemented = option.get("implemented") is True
            reachability = option.get("reachability") or {}
            scheme: dict[str, Any] = {
                "id": option_id,
                "choice": first,
                "label": option.get("label", option_id),
                "description": words["descriptions"].get(family, {}).get(option_id, ""),
                "is_default": reference_components.get(family) == option_id,
                "implemented": implemented,
                "reachability": reachability.get("state"),
                "blocker": reachability.get("blocker"),
                "maturity": option.get("maturity"),
                "scientific_evidence": option.get("scientific_evidence"),
                "switches": dict(option.get("selectors") or {}),
                "streams": _streams(option.get("selectors") or {}),
                "variants": variants,
                "requires": dict((option.get("constraints") or {}).get("requires_components") or {}),
                "requires_reasons": dict((option.get("constraints") or {}).get("requires_components_reasons") or {}),
                "notes": list(option.get("warnings") or []),
                "suites": sorted(t for t, row in registry["templates"].items()
                                 if row.get("components", {}).get(family) == option_id),
                "vertical_levels": _levels(option),
            }
            if implemented:
                described_in, settings = _scheme_base(family, option_id, first)
                cfg = _probe(settings, dx_km=PROBE_DX_KM, nz=PROBE_NZ)
                scheme["described_in"] = ("the default suite with this scheme swapped in"
                                          if described_in == reference_suite else
                                          described_in or "the default suite with this scheme swapped in, which the engine refuses; no Create-page suite carries it")
                scheme["cost"] = (_cost(cfg, reference_cfg) if not scheme["is_default"] else
                                  {"rung": None, "relative": 1.0, "measured": True,
                                   "words": "The default: every cost here is measured against it."})
                scheme["resolution"] = _resolution(settings, component.get("selector_keys") or ())
                scheme["cards"] = _card_fit(cfg)
            else:
                scheme["cost"] = {"relative": None, "measured": False, "words": "Not runnable."}
            schemes.append(scheme)
        families.append({"id": family, "name": family_words.get("name", family),
                         "what": family_words.get("what", ""),
                         "switch_keys": list(component.get("selector_keys") or []),
                         "streams": ["longwave", "shortwave"] if "ra_lw_physics" in (component.get("selector_keys") or []) else None,
                         "schemes": schemes})
    suites = []
    for suite_id, row in registry["templates"].items():
        entry = {"id": suite_id, "label": row.get("label", suite_id), "maturity": row.get("maturity"),
                 "components": dict(row.get("components") or {}),
                 "on_create_page": suite_id in shipped, "notes": list(row.get("warnings") or [])}
        if suite_id in shipped:
            facts = profile_facts(suite_id)
            entry.update(day_only=facts["day_only"], day_only_reason=facts["day_only_reason"],
                         verification=facts["maturity"].get("verification_status"),
                         vertical_levels={key: facts["vertical_levels"].get(key) for key in ("minimum", "maximum")},
                         summary=facts["summary"],
                         cost=_cost(_probe(facts["switches"], dx_km=PROBE_DX_KM, nz=PROBE_NZ), reference_cfg))
        suites.append(entry)
    return {
        "schema": SCHEMA,
        "physics_registry_sha256": registry_sha256(),
        "default_suite": reference_suite,
        # The default by grid spacing, ahead of default_suite when the
        # finest grid is finer than a row's bound: the row `woof domain`
        # and New forecast bind (woof.physics_menu.SPACING_DEFAULTS),
        # answered for one domain and for a tree.
        "spacing_defaults": spacing_default_menu_rows(
            source or default_source()),
        "source": source or default_source(),
        "cost_reference": {"suite": reference_suite, "words": "Cost is per model step against the default suite, "
                           "from the measured step rates in woof.core.pace. A scheme the rates do not separate "
                           "from the default says unmeasured."},
        "cards": list(cards()),
        "families": families,
        "scheme_count": sum(len(f["schemes"]) for f in families),
        "suites": suites,
        "presets": presets(),
    }


def presets() -> list[dict[str, Any]]:
    return [dict(row) for row in table()["presets"]]


def preset(preset_id: str) -> dict[str, Any]:
    for row in presets():
        if row["id"] == preset_id:
            return row
    names = ", ".join(row["id"] for row in presets())
    raise CatalogError(f"No preset called {preset_id!r}. Presets: {names}.")


# ------------------------------------------------------------------ check

def _choice_row(family: str, choice: str) -> dict[str, Any]:
    rows = _menu().get(family)
    if rows is None:
        raise CatalogError(f"No physics family called {family!r}. Families: {', '.join(sorted(_menu()))}.")
    if choice in rows:
        return rows[choice]
    matches = [row for row in rows.values() if row["registry_option_id"] == choice]
    if matches:
        return matches[0]
    raise CatalogError(f"{family} has no scheme called {choice!r}. Schemes: {', '.join(sorted(rows))}.")


def _radiation_streams(value: Mapping[str, Any]) -> dict[str, Any]:
    """Radiation by stream, composed the way the capability door composes it.

    Each stream's settings come from the first implemented registry
    option whose selector carries that stream's switch, which is the
    order :func:`woof.physics_compat._resolve_physics_component_options`
    uses for an independent pair; the switches themselves come last.
    """

    try:
        wanted = {"ra_lw_physics": int(value["longwave"]), "ra_sw_physics": int(value["shortwave"])}
    except (KeyError, TypeError, ValueError) as error:
        raise CatalogError("radiation by stream needs whole-number longwave and shortwave switches.") from error
    settings: dict[str, Any] = {}
    for key, switch in wanted.items():
        rows = [row for row in _menu()["radiation"].values()
                if row.get("selectors", {}).get(key) == switch and "settings" in row]
        if not rows:
            raise CatalogError(f"No radiation scheme has {key} = {switch}.")
        settings.update(rows[0]["settings"])
    settings.update(wanted, ra_physics=0)
    return settings


def _window(request: Mapping[str, Any]) -> dict[str, Any] | None:
    """The forecast window a night check needs, or None when none is given.

    Spelled as Create spells it: ``cycle`` (``2026-09-24T12``), ``hours``
    (6 when absent, Create's own default), ``lat`` and ``lon`` of the box
    centre.
    """

    from datetime import datetime

    given = [key for key in ("cycle", "lat", "lon") if request.get(key) not in (None, "")]
    if not given:
        return None
    if len(given) < 3:
        raise CatalogError("A night check needs cycle, lat and lon together; "
                           f"this request has only {', '.join(given)}.")
    text = str(request["cycle"]).strip().replace("Z", "")
    try:
        start = datetime.strptime(text, "%Y-%m-%dT%H") if len(text) == 13 else datetime.fromisoformat(text)
        hours = float(request.get("hours") or 6)
        lat, lon = float(request["lat"]), float(request["lon"])
    except (TypeError, ValueError) as error:
        raise CatalogError("cycle must look like 2026-09-24T12, and hours, lat and lon must be numbers.") from error
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 360.0 and 0 < hours <= 384):
        raise CatalogError("lat must be -90 to 90, lon -180 to 360 and hours 1 to 384.")
    return {"start_time": start.replace(tzinfo=None), "run_seconds": hours * 3600.0,
            "ref_lat": lat, "ref_lon": lon, "cycle": text, "hours": hours}


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(str(value))


def experiment_toml(settings: Mapping[str, Any]) -> str:
    """The ``[shared]`` physics lines an experiment file runs this mix with.

    The door for a mix no named suite matches: ``woof domain`` writes an
    experiment for the nearest suite, these lines replace its physics
    switches, and ``woof check`` then ``woof run`` take the file.
    """

    known = _run_fields()
    lines = ["[shared]"]
    lines += [f"{key} = {_toml_value(value)}" for key, value in sorted(settings.items())
              if key in known and value is not None]
    return "\n".join(lines) + "\n"


def _with_preset(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``request`` with a preset's suite and spacing filled into its blanks.

    A suite named by an old profile ID is read as its current ID here, the
    one step every request passes, so the base suite, its components and
    the set the check names all agree for either spelling.
    """

    request = dict(request)
    if isinstance(request.get("suite"), str):
        from woof.physics_registry import canonical_template_id

        request["suite"] = canonical_template_id(request["suite"])
    if not request.get("preset"):
        return request, None
    row = preset(str(request["preset"]))
    request.setdefault("suite", row["suite"])
    request.setdefault("dx_km", row.get("dx_km"))
    return request, row


def _changed(base: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    """The switches this mix sets differently from its base suite."""

    known = _run_fields()
    reference = _suite_switches(base)
    return {key: value for key, value in settings.items()
            if key in known and value is not None and reference.get(key) != value}


def apply_to_experiment(text: str, request: Mapping[str, Any], *, load: bool = True) -> str:
    """An experiment file's text with this mix's physics in it.

    The door for a mix no named suite matches.  ``[shared]`` gets every
    physics switch of the mix; a ``[[domain]]`` table keeps its own
    resolution settings (``radt``, ``cudt_minutes`` and the like, which
    ``woof domain`` sized for that grid) except a switch the mix changes
    from its base suite, which is written there too, because a domain's
    own value outranks ``[shared]``.  The result is loaded by the real
    experiment loader before it is returned, so every door a run meets
    at load has already answered.  ``load=False`` leaves that to the
    caller: ``woof domain`` loads every layout its fit tries itself, and
    judges one too big for the source or the pole as a size, not as the
    mix's refusal.
    """

    import tempfile

    from woof.experiment import load_experiment
    from woof.toml_document import iter_toml_statements

    import tomllib

    request = dict(request)
    # The root this file runs decides whether its cumulus is retired, not
    # the probe spacing, and its finest grid and domain count decide which
    # default a mix naming no suite changes, as they decided the suite
    # `woof domain` wrote into it.  A key the request states wins.
    request = {**experiment_grid(text),
               **{key: value for key, value in request.items() if value not in (None, "")}}
    verdict = check(request)
    if not verdict["valid"]:
        raise CatalogError(verdict["words"])
    request, _ = _with_preset(request)
    base, settings, _, _ = _settings(request)
    # A file that already runs cumulus at its root asked for it: the
    # emitter writes an active scheme on a convection-permitting root only
    # when the user named a suite or chose cumulus.  A mix that does not
    # touch cumulus keeps what the file runs; only a file already running
    # it off is written off.
    requested = _cumulus_requested(request) or bool(file_cumulus(text)[0])
    settings, _ = _as_emitted(settings, base, dx_km=float(verdict["dx_km"]), requested=requested)
    known = _run_fields()
    full = {key: value for key, value in settings.items() if key in known and value is not None}
    changed = _changed(base, settings)
    document = tomllib.loads(text)
    shared = document.get("shared") or {}
    tables = document.get("domain") or []
    # A mix that moves the file onto RUC land writes the RUC cold start
    # every authoring door writes (physics_source_defaults.
    # AUTHORED_RUC_DEFAULTS): without it a RUC suite put on a Noah file
    # (a sensitivity '-ruc-' member, the marine-fog preset) cold-started
    # the ground vapour the public-WRF way and dropped the land skin
    # 3.5-4 K on its first step.  A file already on RUC keeps what it runs
    # (an imported WRF namelist is the parity path), and a value the file
    # or the request states is kept.
    from woof.physics_source_defaults import authored_ruc_defaults

    def file_land() -> Any:
        root = tables[0] if tables else {}
        return root.get("sf_surface_physics",
                        shared.get("sf_surface_physics", _run_default("sf_surface_physics")))

    if not authored_ruc_defaults(file_land()):
        for key, value in authored_ruc_defaults(full.get("sf_surface_physics", file_land())).items():
            if key not in full and key not in shared and not any(key in table for table in tables):
                full[key] = value
    # A nest the file runs without cumulus keeps it off: the mix's cumulus
    # is the root's, and ``woof domain`` pins every nest to cu_physics 0.
    # Writing a changed scheme into every table turned cumulus on at a
    # 1.33 km nest when the mix picked one for its 12 km root.  A nest
    # that states no cu_physics runs [shared]'s, or the engine's 0.
    quiet_nests = {index for index, table in enumerate(tables)
                   if index and not int(table.get("cu_physics", shared.get("cu_physics", _run_default("cu_physics")))
                                        or 0)}
    nest_changed = {key: value for key, value in changed.items() if key not in _NEST_CUMULUS}

    def reads_shared(index: int, table: Mapping[str, Any], key: str) -> bool:
        return key not in table and not (index in quiet_nests and key in _NEST_CUMULUS)

    # [shared] carries every switch of the mix it already states, and
    # each other one some grid runs from there.  A key every grid states
    # for itself stays out: the domains' values are the ones that run.
    # A key only some grids state still goes in, because the others run
    # the [shared] value: leaving it out gave them the engine default
    # instead of the mix (radt 0, radiation on every model step).
    in_shared = {key for key in full if key in shared or not tables
                 or any(reads_shared(index, table, key) for index, table in enumerate(tables))}
    # A urban key the file never states, at the engine's own urban-off
    # default, stays out: every grid already runs that value, and writing
    # sf_urban_physics = 0 into a file that never named urban physics made
    # every mix touch a component it did not choose.
    in_shared -= {key for key in _URBAN_RUN_KEYS
                  if key not in shared and not any(key in table for table in tables)
                  and full.get(key) == _run_default(key)}
    # A quiet nest that states no cu_physics would run the scheme [shared]
    # now carries, so it is written 0 in its own table.
    pins = {index: {"cu_physics": 0} for index in quiet_nests
            if "cu_physics" not in tables[index] and "cu_physics" in in_shared
            and int(full.get("cu_physics") or 0)}
    section, shared_end, seen_shared, domain_index = None, None, set(), -1
    out: list[str] = []

    def end_of_table() -> int:
        at = len(out)
        while at > 0 and not out[at - 1].strip():
            at -= 1
        return at

    def close_section() -> None:
        nonlocal shared_end
        if section == "shared":
            shared_end = end_of_table()
        elif section == "domain" and domain_index in pins:
            at = end_of_table()
            out[at:at] = [f"{key} = {_toml_value(value)}" for key, value in pins[domain_index].items()]

    # Whole TOML statements, named by TOML itself: ["shared"] is the
    # [shared] table and "mp_physics" the mp_physics key, as the file
    # ``woof domain-fit`` writes spells them, and a value written over
    # several lines is one statement.  A statement already holding the
    # value the mix runs is kept as the file wrote it.
    for kind, path, lines in iter_toml_statements(text):
        if kind in ("table", "array"):
            close_section()
            section = ("shared" if (kind, path) == ("table", ("shared",)) else
                       "domain" if (kind, path) == ("array", ("domain",)) else "other")
            if section == "domain":
                domain_index += 1
            out.extend(lines)
            continue
        key = path[0] if kind == "assignment" and len(path) == 1 else None
        wanted = (full if section == "shared" else
                  (nest_changed if domain_index in quiet_nests else changed) if section == "domain" else {})
        if key in wanted:
            if section == "shared":
                seen_shared.add(key)
            stated = tomllib.loads("\n".join(lines) + "\n")[key]
            if type(stated) is not type(wanted[key]) or stated != wanted[key]:
                out.append(f"{key} = {_toml_value(wanted[key])}")
                continue
        out.extend(lines)
    close_section()
    if shared_end is None:
        raise CatalogError("The experiment file has no [shared] table, so there is nowhere to put the physics.")
    missing = [f"{key} = {_toml_value(full[key])}" for key in sorted(in_shared) if key not in seen_shared]
    out[shared_end:shared_end] = missing
    result = "\n".join(out) + "\n"
    # Both catalog --into and domain --physics-choices write this file.
    # The root's final switches, including its cumulus choice, own the header.
    from woof.domain_wizard import mix_physics_summary

    result = re.sub(r"^# PHYSICS: .*$",
                    lambda _: f"# PHYSICS: {mix_physics_summary(result, request)}.",
                    result, flags=re.MULTILINE)
    if not load:
        return result
    with tempfile.TemporaryDirectory() as folder:
        trial = Path(folder) / "experiment.toml"
        trial.write_text(result, encoding="utf-8")
        try:
            load_experiment(trial)
        except (ValueError, KeyError, TypeError, NotImplementedError) as error:
            raise CatalogError(f"The experiment loader refuses this file with the mix in it: {error}") from error
    return result


def file_cumulus(text: str) -> tuple[int, float | None]:
    """The cumulus scheme and step an experiment file's root runs.

    A ``[[domain]]`` table's own value outranks ``[shared]``, as the
    experiment loader reads it.  A file that sets neither answers 0.
    """

    import tomllib

    document = tomllib.loads(text)
    shared = document.get("shared") or {}
    tables = document.get("domain") or []
    root = tables[0] if tables else {}
    scheme = root.get("cu_physics", shared.get("cu_physics", 0))
    step = root.get("cudt_minutes", shared.get("cudt_minutes"))
    return int(scheme or 0), (None if step is None else float(step))


def cumulus_change(before: str, after: str) -> str | None:
    """One sentence when a mix changed the root's cumulus, else None."""

    old, new = file_cumulus(before)[0], file_cumulus(after)[0]
    if old == new:
        return None
    if not new:
        return f"The mix turns the root's cumulus off (cu_physics {old} -> 0)."
    if not old:
        return f"The mix turns the root's cumulus on (cu_physics 0 -> {new})."
    return f"The mix changes the root's cumulus (cu_physics {old} -> {new})."


@lru_cache(maxsize=1)
def _physics_keys() -> frozenset[str]:
    """Every switch some shipped suite sets: what counts as physics in a file."""

    from woof.physics_menu import shipped_profiles

    keys: set[str] = set()
    for suite in shipped_profiles():
        keys.update(_suite_switches(suite))
    return frozenset(keys & _run_fields())


def _inert(switches: Mapping[str, Any], components: Mapping[str, str]) -> set[str]:
    """Scheme settings in ``switches`` that no resolved component reads.

    A key only one scheme's settings carry (``wsm6_hail_opt``,
    ``morr_rimed_ice``) does nothing when that scheme is not the one
    running.  A key several schemes carry (``moist_cq``, ``icloud``) or
    none does (``radt``, ``epssm``) is kept as live: the registry does
    not say which of those a scheme ignores.  With nothing resolved,
    nothing is called inert.
    """

    if not components:
        return set()
    owners: dict[str, set[str]] = {}
    live: set[str] = set()
    for family, rows in _menu().items():
        running = components.get(family)
        for row in rows.values():
            for key in row["settings"]:
                owners.setdefault(key, set()).add(f"{family}/{row['registry_option_id']}")
            if running is not None and running in (row["id"], row["registry_option_id"]):
                live.update(row["settings"])
    return {key for key in switches if len(owners.get(key, ())) == 1 and key not in live}


def experiment_physics(text: str) -> dict[str, Any]:
    """The physics an experiment file runs, as a run manifest records it.

    ``switches`` are the ``[shared]`` physics switches the resolved
    schemes read, and ``inert`` the ones a scheme that is not running
    would read (kept, so the file can be rebuilt); ``domains`` lists
    each ``[[domain]]`` table's own physics values (they outrank
    ``[shared]``); ``suite`` and ``components`` are what the first domain
    resolves to, ``suite`` null when no named suite matches.
    """

    import tomllib

    raw = tomllib.loads(text)
    keys = _physics_keys()
    shared = {key: value for key, value in (raw.get("shared") or {}).items() if key in keys}
    tables = raw.get("domain") or []
    domains = [{"grid_id": table.get("grid_id"),
                **{key: value for key, value in table.items() if key in keys}} for table in tables]
    first = dict(shared)
    if tables:
        first.update({key: value for key, value in tables[0].items() if key in keys})
    dx_m = float((tables[0].get("dx") if tables else None) or PROBE_DX_KM * 1000.0)
    cfg = _probe(first, dx_km=dx_m / 1000.0, nz=int((raw.get("shared") or {}).get("nz") or PROBE_NZ))
    components = _resolved(cfg)
    inert = _inert(shared, components)
    return {"suite": _named(cfg), "components": components,
            "switches": {key: value for key, value in shared.items() if key not in inert},
            "inert": {key: shared[key] for key in sorted(inert)}, "domains": domains}


def write_experiment(into: Path, out: Path, request: Mapping[str, Any]) -> list[Path]:
    """Write ``into`` with this mix's physics to ``out``, with its companion files.

    ``woof domain`` writes the experiment's route inputs beside it under
    the same stem (``<stem>.namelist.wps`` and, on the HRRR route, the
    namelist pair), and the route reads them by that stem.  A mix printed
    under a new name without them is refused at the first stage
    ("base WPS namelist is missing"), so they are copied under the new
    stem.
    """

    import shutil

    from woof.hrrr_route_inputs import route_input_paths

    into, out = Path(into), Path(out)
    if into.resolve() == out.resolve():
        raise CatalogError("--out must name a new file; the experiment the mix goes into is kept as it is.")
    companions = route_input_paths(into)
    if companions["namelist_input"].exists():
        # The HRRR route runs from a namelist.input that states physics
        # too; rewriting only the TOML would leave the pair disagreeing
        # and the run is refused at its identity check.
        raise CatalogError(f"{companions['namelist_input'].name} beside this experiment states its physics as "
                           "well, and this command rewrites only the TOML, so the run would be refused at the "
                           "route's identity check. Write the experiment with the suite you want instead.")
    text = apply_to_experiment(into.read_text(encoding="utf-8"), request)
    targets = route_input_paths(out)
    pairs = [(companions[key], targets[key]) for key in companions if companions[key].exists()]
    taken = [path for path in [out, *(target for _, target in pairs)] if path.exists()]
    if taken:
        # Checked before anything is written, so a refusal leaves no half
        # of a new experiment beside someone's existing files.
        raise CatalogError(f"{', '.join(path.name for path in taken)} already "
                           f"{'exists' if len(taken) == 1 else 'exist'} in {out.parent}; name a new file.")
    with open(out, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    written = [out]
    for source, target in pairs:
        shutil.copyfile(source, target)
        written.append(target)
    return written


def _lineage_parameters(suite_settings: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, str]:
    """The suite's source-generation settings: parameter name -> its family.

    A generation setting is a family-bound registry parameter whose values
    are named code generations (a string enum, e.g. ``thompson_version``,
    ``bl_mynn_version``, ``mynn_sfclay_variant``) and which the suite sets
    away from the registry default.  Together they are the code lineage
    the suite was assembled from (the HRRR suite: the NOAA WRF 3.9 fork
    Thompson, GSD 4.1 MYNN and the GSL surface layer).  Integer tuning
    knobs (``scalar_pblmix``, ``bl_mynn_mixlength``) are not generations.
    """

    lineage: dict[str, str] = {}
    for name, spec in registry["parameters"].items():
        if (not isinstance(spec, Mapping) or not spec.get("component_id") or "default" not in spec
                or name not in suite_settings):
            continue
        values = spec.get("enum") or []
        if not values or not all(isinstance(value, str) for value in values):
            continue
        if suite_settings[name] != spec["default"]:
            lineage[name] = str(spec["component_id"])
    return lineage


def _follow_choices(settings: dict[str, Any], chosen: dict[str, str],
                    suite_settings: Mapping[str, Any], *, explicit: set[str],
                    template: Mapping[str, str] | None = None) -> None:
    """The companion edits a scheme choice carries into its suite, in place.

    A choice is one family's scheme; the rest of the suite stays.  Four
    things in the rest belong to the scheme that was swapped out, and
    keeping them made the engine refuse the mix (U-E4: 46 suite/choice
    pairs refused at every source and spacing, customer
    run_5f83d3dcf5f1cec3) or run a scheme nobody asked for:

    * a sibling the chosen scheme declares it needs
      (``requires_components``: MYJ needs the Eta surface layer, YSU and
      Shin-Hong an MM5 one) comes with it, the first the registry lists,
      unless the request chose that family itself;
    * the suite's source-generation settings (:func:`_lineage_parameters`)
      are one lineage: they hold while every scheme that carries one is
      kept.  Once a choice replaces one of those schemes the mix is no
      longer that lineage, so the rest go back to the registry's default
      generation.  Before this, the HRRR suite with its PBL swapped to YSU
      (and Noah, KF) still ran the NOAA WRF 3.9 fork Thompson under a label
      that reads only "Thompson aerosol-aware": a user asking for WRF's
      mp28 got the fork, which on a measured 3 km case gave about a
      quarter less rain than WRF 4.6.1's Thompson from the same inputs.
      A request that wants the fork in such a mix sets ``thompson_version``
      in ``settings``, which applies after this;
    * a parameter the suite read and the mix no longer reads (the HRRR
      suite's ``thompson_version = 'wrf_39_noaa'`` once mp_physics is not
      28, the GSD MYNN generation once the PBL is not MYNN) goes back to
      its registry default, as an unread knob must
      (:func:`woof.physics_registry.registry_knob_is_read`);
    * the WRF RRTMG mapping token follows the RRTMG port that runs: a
      choice whose declared remedy moves 4/4 radiation to the legacy port
      (Lin) leaves the suite's RTE+RRTMGP substitution token naming a
      solver the run no longer has, so it takes the legacy token.
    """

    from woof.config import (RRTMG_VARIANT_RTE_RRTMGP, WRF_RRTMG_LEGACY,
                              WRF_RRTMG_SUBSTITUTION_TOKENS)
    from woof.physics_registry import registry_knob_is_read

    registry = _registry()
    menu = _menu()
    for _ in range(len(menu)):
        moved = False
        for family in sorted(explicit | {f for f in chosen if f not in explicit}):
            option = registry["components"].get(family, {}).get("options", {}).get(_registry_id(chosen.get(family)))
            needs = ((option or {}).get("constraints") or {}).get("requires_components") or {}
            for sibling, allowed in needs.items():
                if sibling in explicit or _registry_id(chosen.get(sibling)) in allowed:
                    continue
                rows = [row for wanted in allowed for row in menu.get(sibling, {}).values()
                        if row["registry_option_id"] == wanted
                        and registry["components"][sibling]["options"][wanted].get("implemented") is True]
                if not rows:
                    continue
                settings.update(rows[0]["settings"])
                chosen[sibling] = rows[0]["id"]
                moved = True
        if not moved:
            break
    parameters = registry["parameters"]
    if template:
        lineage = _lineage_parameters(suite_settings, registry)
        replaced = {family for family, scheme in template.items()
                    if _registry_id(chosen.get(family)) != _registry_id(scheme)}
        if replaced & set(lineage.values()):
            for name in lineage:
                settings[name] = parameters[name]["default"]
    for _ in range(len(parameters)):
        reverted = False
        for name, spec in parameters.items():
            if (name not in settings or not isinstance(spec, Mapping) or "default" not in spec
                    or settings[name] == spec["default"]):
                continue
            if registry_knob_is_read(name, suite_settings, registry) and not registry_knob_is_read(
                    name, settings, registry):
                settings[name] = spec["default"]
                reverted = True
        if not reverted:
            break
    if (settings.get("wrf_rrtmg_compatibility") in WRF_RRTMG_SUBSTITUTION_TOKENS
            and settings.get("ra_rrtmg_variant", RRTMG_VARIANT_RTE_RRTMGP) != RRTMG_VARIANT_RTE_RRTMGP):
        settings["wrf_rrtmg_compatibility"] = WRF_RRTMG_LEGACY


def _settings(request: Mapping[str, Any]) -> tuple[str, dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    suite = request.get("suite")
    # No suite named: the default at the request's grid, the one the run binds.
    base = str(suite) if suite else request_default_suite(request)
    try:
        settings = _suite_switches(base)
    except (KeyError, ValueError) as error:
        from woof.physics_menu import shipped_profiles

        raise CatalogError(f"No suite called {base!r} on the Create page. Suites: "
                           f"{', '.join(shipped_profiles())}.") from error
    template = _template_components(base)
    chosen = dict(template or {})
    suite_settings = dict(settings)
    blocked: list[dict[str, Any]] = []
    for family, choice in (request.get("choices") or {}).items():
        if family == "radiation" and isinstance(choice, Mapping):
            settings.update(_radiation_streams(choice))
            chosen[family] = f"longwave {choice.get('longwave')}, shortwave {choice.get('shortwave')}"
            continue
        row = _choice_row(str(family), str(choice))
        settings.update(row["settings"])
        chosen[str(family)] = row["id"]
        option = _registry()["components"][str(family)]["options"][row["registry_option_id"]]
        if option.get("implemented") is not True:
            reachability = option.get("reachability") or {}
            blocked.append({
                "door": "registry",
                "owner": f"woof/physics_registry_v2.json#/components/{family}/options/{row['registry_option_id']}",
                "message": f"{option.get('label', row['id'])}: "
                           f"{reachability.get('blocker') or 'declared unimplemented'}"})
    _follow_choices(settings, chosen, suite_settings,
                    explicit={str(family) for family in (request.get("choices") or {})},
                    template=template)
    overrides = {str(key): value for key, value in (request.get("settings") or {}).items()}
    # A name RunConfig has no field for sets nothing: the probe and the
    # written file both drop it, so a misspelled mp_physcs was answered
    # valid and ran the suite's microphysics.
    unknown = sorted(set(overrides) - _run_fields())
    if unknown:
        import difflib

        words = []
        for key in unknown:
            near = difflib.get_close_matches(key, sorted(_run_fields()), n=3, cutoff=0.75)
            words.append(f"No setting called {key!r}" + (f"; did you mean {' or '.join(near)}?" if near else "."))
        raise CatalogError(" ".join(words) + " Settings take the switch names an experiment file's [shared] "
                                              "table uses, such as mp_physics.")
    settings.update(overrides)
    return base, settings, chosen, blocked


def _cumulus_requested(request: Mapping[str, Any]) -> bool:
    """Whether the caller asked for the root's cumulus, as the wizard reads it.

    Naming a suite (``suite``, or a preset standing for one) is what
    ``--physics-profile`` is on ``woof domain`` and ``physics_profile``
    is on Create: the suite is emitted verbatim, cumulus included.  A
    cumulus choice or a ``cu_physics`` or ``cudt_minutes`` setting asks
    for it directly.  Anything else runs the default suite as the wizard
    emits it.
    """

    return bool(request.get("suite") or "cumulus" in (request.get("choices") or {})
                or {"cu_physics", "cudt_minutes"} & set(request.get("settings") or {}))


def _as_emitted(settings: Mapping[str, Any], base: str, *, dx_km: float,
                requested: bool) -> tuple[dict[str, Any], list[str]]:
    """``settings`` as the emitter writes them for a root at ``dx_km``.

    The switch comes from :func:`woof.domain_wizard.root_cumulus` and
    the sentence from :func:`woof.domain_wizard.cumulus_retired_headline`,
    the two calls ``woof domain`` and Create make, so the check cannot
    describe a cumulus scheme the run will not have.
    """

    from woof.domain_wizard import cumulus_retired_headline, root_cumulus

    emitted = root_cumulus(settings, dx_km, cumulus_requested=requested)
    if emitted == dict(settings):
        return emitted, []
    return emitted, cumulus_retired_headline(base, dx_km, cumulus_requested=requested)


def _verdict(settings: Mapping[str, Any], *, dx_km: float, nz: int,
             source: str | None, window: Mapping[str, Any] | None = None,
             domains: int = 1) -> tuple[Any, dict[str, Any] | None]:
    try:
        cfg = _probe(settings, dx_km=dx_km, nz=nz)
    except (TypeError, ValueError) as error:
        return None, {"door": "configuration", "owner": "woof.config.RunConfig", "message": str(error)}
    return cfg, _first_refusal(cfg, source=source, dx_km=dx_km, window=window, domains=domains)


def _named(cfg) -> str | None:
    from woof.physics_compat import identify_single_domain_profile

    try:
        return identify_single_domain_profile(cfg)
    except Exception:  # noqa: BLE001 - an unidentifiable mix is simply unnamed
        return None


def _suite_label(suite: str) -> str:
    """The registry's plain name for a suite, the id when it has none."""

    return str((_registry()["templates"].get(suite) or {}).get("label") or suite)


def _without_cumulus(label: str, suite: str) -> str:
    """A suite's plain name with its cumulus scheme shown as off.

    The registry spells a suite's parts joined by `` + `` and names its
    cumulus scheme by the option's label or that label's initials
    (Kain-Fritsch as KF); a suite whose cumulus the grid retired is
    shown with that part as ``cumulus off``, the registry's own words
    for a suite with none.
    """

    component = (_registry()["templates"].get(suite) or {}).get("components", {}).get("cumulus")
    option = str((_registry()["components"].get("cumulus", {}).get("options", {}).get(component) or {})
                 .get("label") or "")
    if component in (None, "off"):
        # The suite runs no cumulus scheme of its own: its name already says so.
        return label
    initials = "".join(word[0] for word in re.split(r"[-\s]+", option) if word).upper()
    parts = label.split(" + ")
    for index, part in enumerate(parts):
        if option and part in (option, initials):
            parts[index] = "cumulus off"
            return " + ".join(parts)
    return f"{label}, cumulus off"


def _resolved(cfg) -> dict[str, str]:
    from woof.physics_compat import validate_physics_capabilities

    try:
        return dict(validate_physics_capabilities(cfg))
    except Exception:  # noqa: BLE001 - only asked of admitted configs
        return {}


def _neighbours(settings: Mapping[str, Any], chosen: Mapping[str, str], *, dx_km: float,
                nz: int, source: str | None, cfg, kept=(),
                window: Mapping[str, Any] | None = None, default: str | None = None,
                domains: int = 1) -> list[dict[str, Any]]:
    """The nearest combinations the same doors admit.

    Ranked: first those that keep every family the caller chose
    explicitly (``kept``), then fewest families changed, then the
    registry's own declared remedy, then a combination that is a named
    suite, then fewest families away from ``default`` (the suite the run
    gets with none named, the source's own when not given).
    """

    from woof.physics_compat import conditional_refusals_for

    menu = _menu()
    defaults = _template_components(default or default_suite(source)) or {}
    candidates: list[tuple[str, dict[str, Any], dict[str, str]]] = []
    if cfg is not None:
        for rule in conditional_refusals_for(cfg):
            if rule.get("remedy_settings"):
                candidates.append((str(rule.get("remedy_label") or rule["reason"]),
                                   dict(rule["remedy_settings"]), {}))
    groups = [(family,) for family in menu] + [g for g in COUPLED_GROUPS if all(f in menu for f in g)]
    for group in groups:
        for picks in itertools.product(*(menu[f].values() for f in group)):
            if any(chosen.get(f) == row["id"] for f, row in zip(group, picks)):
                continue
            edit: dict[str, Any] = {}
            changes = {}
            for family, row in zip(group, picks):
                edit.update(row["settings"])
                changes[family] = row["id"]
            candidates.append(("", edit, changes))
    found, seen = [], set()
    for label, edit, changes in candidates:
        trial = dict(settings)
        trial.update(edit)
        trial_cfg, refusal = _verdict(trial, dx_km=dx_km, nz=nz, source=source, window=window,
                                      domains=domains)
        if refusal is not None:
            continue
        resolved = _resolved(trial_cfg)
        identity = json.dumps(resolved, sort_keys=True)
        if identity in seen:
            continue
        seen.add(identity)
        changed = {f: {"from": chosen.get(f), "to": resolved.get(f, changes.get(f))}
                   for f in resolved if resolved.get(f) != _registry_id(chosen.get(f))}
        if not changed and changes:
            changed = {f: {"from": chosen.get(f), "to": v} for f, v in changes.items()}
        named = _named(trial_cfg)
        words = label or "; ".join(
            f"{_family_name(f)}: {_label(f, c['to'])}" for f, c in changed.items())
        found.append({"words": words, "choices": {f: c["to"] for f, c in changed.items()},
                      "settings": edit, "changes": changed, "named_suite": named,
                      "advisories": [_headline(line) for line in _advisories(trial, dx_km)],
                      "_rank": (sum(1 for f in changed if f in kept), len(changed),
                                0 if label else 1, 0 if named else 1,
                                sum(1 for f, c in changed.items() if c["to"] != defaults.get(f)))})
    found.sort(key=lambda row: row["_rank"])
    for row in found:
        row.pop("_rank")
    return found[:NEIGHBOUR_LIMIT]


def _family_name(family: str) -> str:
    return table()["families"].get(family, {}).get("name", family.replace("_", " "))


def _label(family: str, choice: str | None) -> str:
    row = _menu().get(family, {}).get(str(choice))
    if row is None:
        option = _registry()["components"].get(family, {}).get("options", {}).get(str(choice)) or {}
        return option.get("label", str(choice))
    return row.get("label", str(choice))


def _registry_id(choice: str | None) -> str | None:
    return choice.split(":", 1)[0] if isinstance(choice, str) else choice


def check(request: Mapping[str, Any]) -> dict[str, Any]:
    """Valid, or the engine's own refusal with the nearest valid combinations.

    ``request``: ``suite`` (a registered suite id; when absent, the
    default `woof domain` binds for this source at the request's finest
    grid), ``preset`` (a preset id, standing for its suite and grid
    spacing), ``choices`` (family -> scheme id, or for radiation a
    ``{"longwave": n, "shortwave": n}`` pair), ``settings`` (raw switch
    overrides), ``dx_km`` (the root's spacing), ``finest_dx_km`` and
    ``domains`` (a run with nests: its finest spacing and how many grids
    it has; one grid at ``dx_km`` when absent), ``nz``, ``source`` and
    ``card``.
    """

    if not isinstance(request, Mapping):
        raise CatalogError("The check takes a JSON object.")
    request = dict(request)
    request, chosen_preset = _with_preset(request)
    try:
        dx_km = float(request.get("dx_km") or PROBE_DX_KM)
        nz = int(request.get("nz") or PROBE_NZ)
    except (TypeError, ValueError) as error:
        raise CatalogError("dx_km and nz must be numbers.") from error
    if not 0.01 <= dx_km <= 200.0:
        raise CatalogError("dx_km must be between 0.01 and 200.")
    source = known_source(request.get("source") or None)
    window = _window(request)
    card = request.get("card") or None
    if card is not None and card not in cards():
        raise CatalogError(f"card must be one of {', '.join(cards())}.")
    finest_m, domains = _request_grid(request)
    # The suite this grid runs with none named, from the row `woof domain`
    # binds: the base of a request naming no suite and the reference its
    # cost, its neighbours and its plan are measured from.
    default = default_suite(source, finest_m, domains)
    base, settings, chosen, blocked = _settings(request)
    as_written = dict(settings)
    settings, retired = _as_emitted(settings, base, dx_km=dx_km, requested=_cumulus_requested(request))
    if retired:
        chosen["cumulus"] = next(key for key, row in _menu()["cumulus"].items()
                                 if row["settings"].get("cu_physics") == settings["cu_physics"])
    cfg, refusal = _verdict(settings, dx_km=dx_km, nz=nz, source=source, window=window, domains=domains)
    if blocked:
        refusal = blocked[0]
    result: dict[str, Any] = {
        "schema": CHECK_SCHEMA, "valid": refusal is None, "base_suite": base,
        "preset": chosen_preset["id"] if chosen_preset else None,
        "dx_km": dx_km, "finest_dx_km": finest_m / 1000.0, "domains": domains, "default_suite": default,
        "nz": nz, "source": source, "card": card, "choices": chosen,
        "cumulus_retired": retired[0] if retired else None,
        "window": None if window is None else {"cycle": window["cycle"], "hours": window["hours"],
                                               "lat": window["ref_lat"], "lon": window["ref_lon"]},
    }
    if refusal is not None:
        result["refusal"] = refusal
        result["words"] = _headline(refusal["message"])
        result["neighbours"] = _neighbours(settings, chosen, dx_km=dx_km, nz=nz, source=source, cfg=cfg,
                                           kept=tuple((request.get("choices") or {}).keys()), window=window,
                                           default=default, domains=domains)
        return result
    from woof.physics_menu import switches_day_only_reason

    # A retired cumulus is the suite as the emitter writes it at this
    # spacing, so a mix that matches no suite once retired is still named
    # by the suite it was before retirement.
    named = _named(cfg) or (_named(_probe(as_written, dx_km=dx_km, nz=nz)) if retired else None)
    from woof.domain_wizard import root_cumulus

    reference_cfg = _probe(root_cumulus(_suite_switches(default), dx_km,
                                        cumulus_requested=False), dx_km=dx_km, nz=nz)
    advisories = _advisories(settings, dx_km)
    result.update(
        resolved=_resolved(cfg),
        named_suite=named,
        day_only_reason=switches_day_only_reason(settings),
        advisories=[{"words": _plain_advisory(line, settings, dx_km), "headline": _headline(line),
                     "sentence": line} for line in advisories],
        cost=(_cost(cfg, reference_cfg) if _resolved(cfg) != _resolved(reference_cfg) else
              {"rung": None, "relative": 1.0, "measured": True, "words": "The default's cost."}),
    )
    if card:
        result["card_fit"] = _card_fit(cfg)[card]
    result["experiment_toml"] = experiment_toml(settings)
    result["changed_from_suite"] = _changed(base, settings)
    result["named_suite_label"] = (None if not named else
                                   _without_cumulus(_suite_label(named), named) if retired else
                                   _suite_label(named))
    # What a run plan carries to run exactly what this check describes.
    # A named suite: that suite, and whether the root's cumulus is the
    # suite's or the grid's.  ``woof domain --cumulus grid`` retires it
    # with the same root_cumulus call this check made, so the two cannot
    # differ.  Any other set: the choices themselves, which
    # ``woof domain --physics-choices`` writes over the base suite with
    # this module's own --into write, checked at the file's own root
    # spacing, so the plan runs the cumulus this check decided there.
    # Raw ``settings`` overrides have no intent key; a set that needs
    # them goes in through an experiment file (--into) instead.
    if named:
        plan_intent = {"physics_profile": named, **({} if _cumulus_requested(request) else {"cumulus": "grid"})}
    elif request.get("settings") or not request.get("choices"):
        plan_intent = None
    else:
        plan_intent = {"physics_choices": dict(request["choices"]),
                       **({} if base == default else {"physics_profile": base})}
    result["plan_intent"] = plan_intent
    result["on_create_page"] = plan_intent is not None
    if named:
        result["words"] = "Runs." if named == base else f"Runs. This is the set {result['named_suite_label']}."
    else:
        result["words"] = "Runs. No named set matches these choices."
    if retired:
        # ``cumulus_retired`` keeps the wizard's own sentence; the words
        # a person reads say the same thing without the switch names.
        result["words"] = result["words"].replace(
            "Runs.", f"Runs. Cumulus is off at {dx_km:g} km, where the grid resolves storms itself.", 1)
    if result["day_only_reason"]:
        night = ("This window is all daylight." if window is not None else
                 "A window that includes local night is refused.")
        rest = result["words"].removeprefix("Runs.").strip()
        result["words"] = " ".join(part for part in (
            "Runs by day only: shortwave with no longwave.", night, rest) if part)
    # The Thompson generation is said whenever it is not WRF 4.6.1's, so a
    # mix that keeps HRRR's WRFV3.9 Thompson (the HRRR set, that set with
    # its cumulus or radiation changed, or a request naming it) never runs
    # it under a label that reads only "Thompson aerosol-aware".
    if settings.get("mp_physics") == 28 and settings.get("thompson_version") == "wrf_39_noaa":
        result["words"] += (" Its Thompson is HRRR's WRFV3.9 version "
                            "(thompson_version = \"wrf_39_noaa\"), not WRF 4.6.1's.")
    return result


# ------------------------------------------------------------------ CLI

def _check_request(argument: str, stdin) -> Any:
    """The --check request: JSON text, a file holding it, or ``-`` for stdin.

    The text is read as JSON before it is taken for a file name.  The GUI
    sends its request inline, and a request naming several families is
    longer than a file name may be, so asking the file system about it
    first failed on Linux with "File name too long" before the check ran.
    """

    if argument == "-":
        return json.loads(stdin.read())
    try:
        return json.loads(argument)
    except json.JSONDecodeError as error:
        path = Path(argument)
        try:
            named = path.is_file()
        except (OSError, ValueError):
            named = False
        if not named:
            raise ValueError(f"--check takes a JSON request, a file holding one, or - for stdin; "
                             f"this is not JSON ({error}) and names no file.") from error
        return json.loads(path.read_text(encoding="utf-8"))


def main(args) -> int:
    import sys

    try:
        if args.check is not None:
            request = _check_request(args.check, sys.stdin)
            if getattr(args, "into", None) and getattr(args, "out", None):
                for path in write_experiment(Path(args.into), Path(args.out), request):
                    print(f"wrote {path}")
                change = cumulus_change(Path(args.into).read_text(encoding="utf-8"),
                                        Path(args.out).read_text(encoding="utf-8"))
                if change:
                    print(change)
                return 0
            if getattr(args, "into", None):
                before = Path(args.into).read_text(encoding="utf-8")
                after = apply_to_experiment(before, request)
                print(after, end="")
                change = cumulus_change(before, after)
                if change:
                    # stdout is the file; the sentence goes beside it.
                    print(change, file=sys.stderr)
                return 0
            document = check(request)
            if getattr(args, "emit", False):
                if not document["valid"]:
                    print(f"physics-catalog: {document['words']}", file=sys.stderr)
                    return 2
                print(document["experiment_toml"], end="")
                return 0
        elif args.preset:
            document = preset(args.preset)
        else:
            document = catalog(source=args.source)
    except (CatalogError, ValueError, OSError) as error:
        # OSError: a request file (or an --into / --out file) that cannot be
        # read or written answers with the same error document as a bad
        # request, not a traceback the page cannot show.
        if args.json:
            print(json.dumps({"schema": CHECK_SCHEMA, "error": str(error)}))
        else:
            print(f"physics-catalog: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(document, indent=1, default=str))
        return 0
    _print_human(document)
    return 0


def _print_human(document: Mapping[str, Any]) -> None:
    if document.get("schema") == CHECK_SCHEMA:
        print(document["words"])
        if not document["valid"]:
            print(f"  {document['refusal']['message']}")
            for row in document.get("neighbours") or []:
                print(f"  try: {row['words']}")
        for row in document.get("advisories") or []:
            print(f"  note: {row.get('words') or row['headline']}")
        return
    if "families" not in document:
        print(json.dumps(document, indent=1))
        return
    print(f"Default suite: {document['default_suite']}")
    for family in document["families"]:
        print(f"\n{family['name']}: {family['what']}")
        for scheme in family["schemes"]:
            mark = "*" if scheme["is_default"] else " "
            cost = scheme["cost"]
            short = ("default" if scheme["is_default"] else
                     "not runnable" if not scheme["implemented"] else
                     f"{cost['relative']:g}x, measured" if cost.get("measured") and cost.get("relative") not in (None, 1.0)
                     else "cost unmeasured")
            print(f" {mark} {scheme['id']:<24} {scheme['description']}  [{short}]")
    print("\nPresets:")
    for row in document["presets"]:
        print(f"  {row['id']:<28} {row['intention']}: {row['why']}")


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "physics-catalog",
        help="list every physics scheme and suite, or check a combination",
        description="Every physics family and scheme with a plain description, its cost against "
                    "the default, where it is valid and its suites; --check JSON answers whether a "
                    "combination runs, and if not, the refusal and the nearest combinations that do.")
    parser.add_argument("--json", action="store_true", help="print JSON for the GUI or scripts")
    parser.add_argument("--source", help="price and default against this data source's default suite")
    parser.add_argument("--check", metavar="JSON", help="a check request as JSON text, a file, or - for stdin")
    parser.add_argument("--preset", help="print one preset row")
    parser.add_argument("--emit", action="store_true",
                        help="with --check: print the experiment-file physics lines for a mix that runs")
    parser.add_argument("--into", metavar="EXPERIMENT",
                        help="with --check: print this experiment file with the mix's physics in it, "
                             "after the experiment loader has accepted it")
    parser.add_argument("--out", metavar="NEW.toml",
                        help="with --into: write the result here, with the experiment's companion files "
                             "(its namelist.wps) copied under the new name, instead of printing it")
    parser.set_defaults(func=main)


__all__ = ["CHECK_SCHEMA", "SCHEMA", "CatalogError", "catalog", "check", "default_suite",
           "apply_to_experiment", "cumulus_change", "experiment_grid", "experiment_physics", "file_cumulus",
           "experiment_toml", "known_source", "request_default_suite", "write_experiment",
           "preset", "presets", "register_cli", "table"]
