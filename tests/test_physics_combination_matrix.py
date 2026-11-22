"""Every physics combination a plan can name is launchable, or refused by name.

THE QUESTION.  The physics registry (``woof/physics_registry_v2.json``) is
the one inventory of what WOOF-ArWen implements.  A user composes seven
components from it -- cumulus, land surface, microphysics, PBL, radiation,
surface layer, turbulence -- on one of the registered runner routes, for
one of the registered sources.  For every such composition exactly one of
two things must be true at the two doors a launch passes through
(:func:`woof.physics_registry.validate_physics_plan`, the plan door every
front end shows before anything is prepared, and
:func:`woof.config.validate_run_config`, the door every RunConfig passes
at run start):

* the composition is LAUNCHABLE at both doors; or
* it is REFUSED, and every door that refuses says, in its own text, WHICH
  scheme it turns away, WHAT breaks if it did not, and the WAY OUT.

Anything else is a defect: a refusal that names none of the three ("runner
route does not allow this component option to vary per domain" named
nothing), or a plan the first door passes and the second refuses (a
radiation override on an RRTMG template used to keep the template's
receipt token beside a Dudhia pair and die at run start).

THE INVENTORY IS THE REGISTRY, never a list typed here.  Options are read
off ``components.<c>.options`` where ``implemented`` is true; routes and
sources off ``runner_routes``; the base template a plan starts from is
chosen from the templates the route reaches for that source.  A scheme
added to the registry is in the matrix the day it lands.

BOUNDED, STATED.  The full product is 144,000 tuples per route and source
and would say nothing the pairs do not.  The matrix is: every PAIR of
components exhaustively (the other five held at the shipped default
template's choice), a deterministic SAMPLE of full tuples (seeded, so a
failure reproduces), and the USER-REPORT tuples always -- the pinned-tile
forecast that was priced at 2.4x its resident run and the Milbrandt-Yau,
New Tiedtke, RUC forecast that died at its first checkpoint.  Each tuple is
crossed with every implemented route and every source that route registers.

THE PRICING INVARIANT.  For every launchable tuple that prices both ways,
a streamed forecast of the user-report width whose two buffers hold a
fraction of the domain never prices above the resident one.  The user
report was a streamed envelope 2.4x the resident because buffers were
itemized as an N x 1 rectangle; the invariant holds on both pricing roads
(the itemized prepared model for RTE+RRTMGP host-store runs, the measured
rung model for everything else).

THE LAW, AS A MEASUREMENT.  "Names a scheme, a breakage and a remedy" is
three marker sets over the refusal text: a registry option id, label word
or selector key; a consequence clause (because / so / would / cannot /
reads / publishes / zero / wrong ...); an imperative or alternative
(select / set / use / or ... / requires x=y).  Markers are a heuristic and
are calibrated below in both directions, on real refusals from both doors,
so the instrument cannot pass everything or fail everything.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import random
import re
import sys
import tomllib
import warnings
from pathlib import Path

import pytest

from woof.config import (
    KM_OPT_ZERO_ACK,
    SASE_PBL_SCHEME,
    RunConfig,
    validate_run_config,
)
from woof.physics_registry import (
    expert_template_ids_for_source,
    DEFAULT_TEMPLATE_ID,
    PLAN_SCHEMA,
    physics_registry,
    registry_sha256,
    validate_physics_plan,
)

REGISTRY = physics_registry()
REGISTRY_SHA256 = registry_sha256(REGISTRY)
COMPONENTS: tuple[str, ...] = tuple(sorted(REGISTRY["components"]))

#: Deterministic sample: a failure names its tuple and reproduces.
SEED = 20260910
SAMPLE_SIZE = 400

#: The two user reports this matrix exists for, as registry option ids.
#: The first is the pinned-tile forecast priced at 2.4x its resident run;
#: the second is the forecast that integrated 59 minutes and died at its
#: first checkpoint.  A ``variant`` hint asks for a base template whose
#: radiation engine is that one, so both RRTMG arms of the second report
#: are walked.
USER_REPORT_TUPLES: tuple[tuple[dict[str, str], str | None], ...] = (
    (dict(cumulus="off", land_surface="noah", microphysics="thompson-mp8",
          pbl="mynn", radiation="rte-rrtmgp", surface_layer="mynn",
          turbulence="smagorinsky-2d", urban="none"), None),
    (dict(cumulus="new-tiedtke", land_surface="ruc-lsm",
          microphysics="milbrandt2mom-mp9", pbl="ysu",
          radiation="rte-rrtmgp", surface_layer="classic-mm5",
          turbulence="smagorinsky-2d", urban="none"), None),
    (dict(cumulus="new-tiedtke", land_surface="noah",
          microphysics="milbrandt2mom-mp9", pbl="ysu",
          radiation="rte-rrtmgp", surface_layer="classic-mm5",
          turbulence="smagorinsky-2d", urban="none"), "rrtmg_legacy"),
)

#: Small and legal for every invariant the run door checks; nz sits inside
#: every shipped vertical bound (Grell-Freitas declares a 12-level floor).
_GEOMETRY = {"nx": 16, "ny": 16, "nz": 40, "dx": 3000.0, "dy": 3000.0,
             "ztop": 20000.0, "dt": 6.0, "run_seconds": 60.0}
_RUN_CONFIG_FIELDS = frozenset(
    field.name for field in dataclasses.fields(RunConfig))


# ---------------------------------------------------------------------------
# The inventory, read off the registry.
# ---------------------------------------------------------------------------

def implemented_options() -> dict[str, tuple[str, ...]]:
    return {
        component: tuple(sorted(
            option_id for option_id, option
            in REGISTRY["components"][component]["options"].items()
            if option.get("implemented") is True))
        for component in COMPONENTS}


def anchor_components() -> dict[str, str]:
    """The shipped default template's choices hold the components a pair
    does not vary."""
    return dict(REGISTRY["templates"][DEFAULT_TEMPLATE_ID]["components"])


def matrix_tuples() -> dict[tuple[str, ...], str]:
    """Every tuple the matrix walks, keyed in COMPONENTS order -> why."""
    options = implemented_options()
    anchor = anchor_components()
    seen: dict[tuple[str, ...], str] = {}

    def add(choice: dict[str, str], why: str) -> None:
        seen.setdefault(tuple(choice[c] for c in COMPONENTS), why)

    for left, right in itertools.combinations(COMPONENTS, 2):
        for a in options[left]:
            for b in options[right]:
                choice = dict(anchor)
                choice[left], choice[right] = a, b
                add(choice, "pair")
    rng = random.Random(SEED)
    for _ in range(SAMPLE_SIZE):
        add({c: rng.choice(options[c]) for c in COMPONENTS}, "sample")
    for choice, _variant in USER_REPORT_TUPLES:
        add(choice, "user-report")
    return seen


def implemented_routes() -> list[tuple[str, dict]]:
    return sorted(
        (route_id, route)
        for route_id, route in REGISTRY["runner_routes"].items()
        if route.get("implemented") is True)


def route_cells() -> list[tuple[str, str]]:
    """(runner_id, source_id) for every implemented route and source."""
    return [(route_id, source_id)
            for route_id, route in implemented_routes()
            for source_id in sorted(route.get("source_ids", []))]


# ---------------------------------------------------------------------------
# One plan per tuple, route and source.
# ---------------------------------------------------------------------------

def reachable_templates(route: dict, source_id: str) -> tuple[list[str], set[str]]:
    """Templates the route declares for the source (normal + expert), or
    every registered template when the route declares none for it -- the
    validator's own rule: an undeclared source reaches a template with a
    route-evidence warning, not a refusal."""
    normal = list(route.get("source_template_ids", {}).get(source_id) or [])
    expert = expert_template_ids_for_source(route, source_id)
    ids = normal + expert
    if not ids:
        ids = sorted(REGISTRY["templates"])
    return ids, set(expert)


def base_template(choice: dict[str, str], candidates: list[str],
                  variant: str | None = None) -> str:
    """The registered template closest to ``choice``.

    Land surface weighs most: no route lets it vary per domain, so a
    template that carries the tuple's land surface is the only way to it.
    A ``variant`` hint prefers a template whose radiation engine parameter
    is that variant.  Ties break on sorted id, so the choice is stable.
    """
    best: tuple[int, str] | None = None
    for template_id in sorted(candidates):
        template = REGISTRY["templates"][template_id]
        components = template["components"]
        score = 10 * (components["land_surface"] == choice["land_surface"])
        score += sum(components[c] == choice[c] for c in COMPONENTS)
        if variant is not None:
            score += 20 * (template.get("parameters", {}).get(
                "ra_rrtmg_variant", "rte-rrtmgp") == variant)
        if best is None or score > best[0]:
            best = (score, template_id)
    assert best is not None
    return best[1]


def plan_for(route_id: str, route: dict, source_id: str,
             choice: dict[str, str], variant: str | None = None) -> dict:
    candidates, expert = reachable_templates(route, source_id)
    template_id = base_template(choice, candidates, variant)
    template_components = REGISTRY["templates"][template_id]["components"]
    overrides = {c: v for c, v in choice.items() if template_components[c] != v}
    nested = "one-way-nested-v1" in route["topology_ids"]
    parameters: dict[str, object] = {}
    if (route.get("mode") == "experiment-per-domain"
            and choice["microphysics"] == "off"
            and "moist" in route.get("allowed_parameter_keys", [])):
        # The remedy the registry names for microphysics off on a
        # real-source route; supplied so the tuple is measured rather
        # than refused for a declaration it could have carried.
        parameters["moist"] = True

    def domain(domain_id: str, root: bool) -> dict:
        entry: dict = {"domain_id": domain_id, "template_id": template_id}
        if overrides:
            entry["components"] = dict(overrides)
        merged = dict(parameters)
        if nested and not root:
            merged["spec_exp"] = 0.0
        if merged:
            entry["parameters"] = merged
        return entry

    plan: dict = {
        "schema": PLAN_SCHEMA,
        "plan_id": "physics-combination-matrix-v1",
        "registry_sha256": REGISTRY_SHA256,
        "context": {"source_id": source_id, "runner_id": route_id,
                    "topology_id": route["topology_ids"][0]},
        "domains": [domain("d01", True)] + ([domain("d02", False)] if nested else []),
        "edges": ([{"parent_domain_id": "d01", "child_domain_id": "d02"}]
                  if nested else []),
    }
    acknowledgement = route.get("expert_acknowledgement_id")
    if template_id in expert and acknowledgement:
        plan["acknowledgements"] = [acknowledgement]
    return plan


def run_door(settings: dict, *, nested: bool) -> str | None:
    """validate_run_config on the settings the plan door resolved."""
    keywords = {name: value for name, value in settings.items()
                if name in _RUN_CONFIG_FIELDS}
    keywords.update(_GEOMETRY)
    keywords["nested"] = nested
    try:
        validate_run_config(RunConfig(**keywords))
    except (ValueError, NotImplementedError, TypeError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


# ---------------------------------------------------------------------------
# The gate law as an instrument.
# ---------------------------------------------------------------------------

_LABEL_NOISE = frozenset(
    "off and the two one model layer surface warm rain land through legacy "
    "selector longwave shortwave clear sky radiation double moment "
    "prognostic order operator mixing constant closure supplied category "
    "aerosol aware no physics microphysics cumulus four moisture ice "
    # Fragments of a selector KEY quoted inside a label ("no km_opt
    # operator"): the word splitter breaks on the underscore, and "opt"
    # is not a scheme.
    "opt "
    # "(WRF option 91)" in the MM5 surface-layer labels: "option" is the
    # word every refusal uses for any scheme, so as a marker it made the
    # bare control "runner route does not allow this component option to
    # vary per domain" read as naming one.
    "option".split())


def _scheme_markers() -> list[str]:
    markers: set[str] = set()
    for component in COMPONENTS:
        spec = REGISTRY["components"][component]
        markers.update(spec.get("selector_keys", []))
        for option_id, option in spec["options"].items():
            markers.add(option_id)
            for word in re.split(r"[^A-Za-z0-9+-]+", option.get("label", "")):
                if len(word) >= 3 and word.lower() not in _LABEL_NOISE:
                    markers.add(word)
    return sorted((re.escape(m) for m in markers), key=len, reverse=True)


#: A scheme: a registry option id, a word of its label, or a selector key.
SCHEME = re.compile("|".join(_scheme_markers()))
#: A breakage: a clause of consequence.
BREAKAGE = re.compile(
    r"because|so that|so with|so a |so the |so only|so every|so it |so this|"
    r"which |would |cannot|can not|does not|do not|is not|are not|has no|"
    r"have no|no (row|table|closure|oracle|silent|substitution|shipped|"
    r"explicit|other|template|column|boundary|KPBL)|fatal|refus|unwritten|"
    r"zero|uninitiali|undefined|degenerat|wrong|silently|"
    r"not (yet )?(implemented|ported|executable|admitted|a |an |defined|"
    r"derived|declared|the )|missing|lacks|leaves|stays|"
    r"only (with|the|when|where|its|an |a |from|through)|allocated|reads|"
    r"publishes|divides|writes none|never|double-count|dies|died|"
    r"nothing (writes|computes|reads)|has not declared", re.IGNORECASE)
#: A remedy: an imperative, an alternative, or the setting that clears it.
REMEDY = re.compile(
    r"select|choose|\bset |\buse |\brun |\badd |remove|\bpass |declare|"
    r"acknowledg|switch|\bpair |instead|\bor (a |an |the |select|set|choose|"
    r"use|pair|km_opt|bl_pbl|sf_|mp_|cu_|ra_|no |another|leave|start|"
    r"register|widen)|remedy|way out|--[a-z]|admits|template|which admit|"
    r"\bgive |\bname |supply|provide|install|widen|start from|"
    # "requires surface_layer in ['eta-similarity']" names the setting
    # and the values it admits, which is the way out.
    r"requires? [a-z_]+ ?= ?\S|requires? [a-z_]+ in \[", re.IGNORECASE)


def law_verdict(text: str) -> tuple[bool, bool, bool]:
    return (bool(SCHEME.search(text)), bool(BREAKAGE.search(text)),
            bool(REMEDY.search(text)))


def obeys_the_law(text: str) -> bool:
    return all(law_verdict(text))


# ---------------------------------------------------------------------------
# The walk, and its receipt.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Cell:
    runner_id: str
    source_id: str
    tuples: int = 0
    launchable: int = 0
    refused: int = 0
    plan_ok_run_refused: int = 0
    plan_refused_run_ok: int = 0
    violations: list = dataclasses.field(default_factory=list)
    launchable_settings: dict = dataclasses.field(default_factory=dict)


def walk_cell(runner_id: str, source_id: str) -> Cell:
    route = REGISTRY["runner_routes"][runner_id]
    cell = Cell(runner_id, source_id)
    tuples = matrix_tuples()
    variants: list[tuple[dict[str, str], str | None]] = [
        (dict(zip(COMPONENTS, key)), None) for key in tuples]
    variants += [(choice, variant) for choice, variant in USER_REPORT_TUPLES
                 if variant is not None]
    for choice, variant in variants:
        cell.tuples += 1
        plan = plan_for(runner_id, route, source_id, choice, variant)
        report = validate_physics_plan(plan)
        plan_text = "\n".join(error["message"] for error in report["errors"])
        run_refusals = []
        for index, resolved in enumerate(report["resolved_domains"]):
            refusal = run_door(resolved["settings"], nested=index > 0)
            if refusal is not None:
                run_refusals.append(refusal)
        label = json.dumps(choice, sort_keys=True) + (
            f" variant={variant}" if variant else "")
        if report["launchable"] and not run_refusals:
            cell.launchable += 1
            key = tuple(choice[c] for c in COMPONENTS) + (variant,)
            cell.launchable_settings.setdefault(
                key, report["resolved_domains"][0]["settings"])
            continue
        cell.refused += 1
        if report["launchable"] and run_refusals:
            cell.plan_ok_run_refused += 1
            cell.violations.append(
                f"{label}: plan door LAUNCHABLE, run door REFUSES -- "
                f"{run_refusals[0][:300]}")
        if not report["launchable"] and not run_refusals:
            cell.plan_refused_run_ok += 1
        if report["errors"] and not obeys_the_law(plan_text):
            cell.violations.append(
                f"{label}: plan door refusal {law_verdict(plan_text)} "
                f"(scheme, breakage, remedy) -- {plan_text[:300]}")
        for refusal in run_refusals:
            if not obeys_the_law(refusal):
                cell.violations.append(
                    f"{label}: run door refusal {law_verdict(refusal)} "
                    f"(scheme, breakage, remedy) -- {refusal[:300]}")
    return cell


def matrix_report() -> dict:
    """Counts over every route and source, for the release record."""
    cells = [walk_cell(runner_id, source_id)
             for runner_id, source_id in route_cells()]
    return {
        "tuples_per_cell": cells[0].tuples if cells else 0,
        "cells": len(cells),
        "launchable": sum(c.launchable for c in cells),
        "refused": sum(c.refused for c in cells),
        "plan_ok_run_refused": sum(c.plan_ok_run_refused for c in cells),
        "plan_refused_run_ok": sum(c.plan_refused_run_ok for c in cells),
        "violations": sum(len(c.violations) for c in cells),
        "per_route": {
            runner_id: {
                "launchable": sum(c.launchable for c in cells if c.runner_id == runner_id),
                "refused": sum(c.refused for c in cells if c.runner_id == runner_id),
                "sources": sum(1 for c in cells if c.runner_id == runner_id)}
            for runner_id, _route in implemented_routes()},
    }


# ---------------------------------------------------------------------------
# Controls: the instrument, both directions.
# ---------------------------------------------------------------------------

def test_the_inventory_is_read_off_the_registry_and_is_not_small():
    options = implemented_options()
    assert set(options) == set(COMPONENTS)
    assert all(len(values) >= 2 for values in options.values()), options
    assert "sase" not in options["microphysics"], (
        "an implemented=false option must not enter the matrix")
    tuples = matrix_tuples()
    assert len(tuples) > 500
    assert set(tuples.values()) == {"pair", "sample", "user-report"} or \
        set(tuples.values()) <= {"pair", "sample", "user-report"}
    for choice, _variant in USER_REPORT_TUPLES:
        assert tuple(choice[c] for c in COMPONENTS) in tuples


def test_the_sample_is_deterministic():
    assert list(matrix_tuples()) == list(matrix_tuples())


@pytest.mark.parametrize("text", [
    # The run door, on a pairing it refuses.
    "km_opt=3 (3-D Smagorinsky) is admitted with bl_pbl_physics=0 only. "
    "Its vertical exchange pair (kmv/khv) is applied by vertical_diffusion_2, "
    "which is PBL-off-gated, so with a PBL scheme on only the horizontal half "
    "of the closure would run. Select bl_pbl_physics=0, or km_opt=4.",
    # The plan door, a dependency row with its reason.
    "option 'mynn' (surface_layer) requires pbl in ['off', 'mynn', 'sase'], "
    "got 'ysu': the MYNN surface layer publishes psim/psih/gz1oz0 and does "
    "NOT publish fm/fh. Select the revised MM5 surface layer for it.",
])
def test_the_law_instrument_accepts_a_reasoned_refusal(text):
    assert law_verdict(text) == (True, True, True)


@pytest.mark.parametrize("text, expected", [
    ("runner route does not allow this component option to vary per domain",
     (False, True, False)),
    ("option 'myj' requires surface_layer in ['eta-similarity']",
     (True, False, True)),
    ("something went wrong", (False, True, False)),
    ("", (False, False, False)),
])
def test_the_law_instrument_rejects_a_bare_refusal(text, expected):
    assert law_verdict(text) == expected
    assert not obeys_the_law(text)


def test_a_mutated_plan_is_still_refused():
    """The walk can still produce a refusal: an unimplemented option."""
    route_id = "tools.prepared_domain_tree_forecast"
    route = REGISTRY["runner_routes"][route_id]
    choice = anchor_components()
    plan = plan_for(route_id, route, sorted(route["source_ids"])[0], choice)
    plan["domains"][0]["components"] = {"microphysics": "sase"}
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert "unimplemented-option" in {e["code"] for e in report["errors"]}


# ---------------------------------------------------------------------------
# The matrix.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("runner_id, source_id", route_cells())
def test_every_tuple_is_launchable_or_refused_by_name(runner_id, source_id):
    cell = walk_cell(runner_id, source_id)
    assert cell.tuples > 500
    assert cell.launchable + cell.refused == cell.tuples
    assert cell.violations == [], (
        f"{runner_id} / {source_id}: {len(cell.violations)} of {cell.tuples} "
        f"tuples violate the gate law or split the two doors "
        f"(launchable {cell.launchable}, refused {cell.refused}, "
        f"plan-ok-run-refused {cell.plan_ok_run_refused}):\n  "
        + "\n  ".join(cell.violations[:40]))


def test_the_user_report_tuples_are_launchable_where_their_route_reaches_them():
    """The two reports name compositions the registry implements; on the
    experiment route they must launch (the checkpoint death and the
    pricing were run-time defects, not pairing refusals)."""
    route_id = "tools.prepared_domain_tree_forecast"
    route = REGISTRY["runner_routes"][route_id]
    for choice, variant in USER_REPORT_TUPLES:
        plan = plan_for(route_id, route, "era5", choice, variant)
        report = validate_physics_plan(plan)
        assert report["launchable"] is True, (choice, variant, report["errors"])
        for index, resolved in enumerate(report["resolved_domains"]):
            assert run_door(resolved["settings"], nested=index > 0) is None


def test_a_radiation_override_on_an_rrtmg_template_resolves_the_receipt_token():
    """The two-door split this matrix found: a plan that overrides
    radiation away from the RRTMG 4/4 pair on a template that carries the
    RRTMG receipt token passed plan review and was refused at run start.
    The non-RRTMG radiation options now resolve the token to 'none'."""
    route_id = "tools.prepared_domain_tree_forecast"
    route = REGISTRY["runner_routes"][route_id]
    choice = anchor_components()
    overridden = []
    for radiation in implemented_options()["radiation"]:
        if radiation == choice["radiation"]:
            continue
        probe = dict(choice, radiation=radiation)
        plan = plan_for(route_id, route, "era5", probe)
        # Exercise an override on the original template even when a named
        # sibling already matches this tuple and plan_for selects it.
        for domain in plan["domains"]:
            domain["template_id"] = DEFAULT_TEMPLATE_ID
            domain["components"] = dict(probe)
        assert plan["domains"][0]["template_id"] == DEFAULT_TEMPLATE_ID
        report = validate_physics_plan(plan)
        settings = report["resolved_domains"][0]["settings"]
        # Since ca920cb60 a template that carries this radiation itself (the
        # Dudhia daytime suite) is reachable on this route and plan_for may
        # pick it; the plan is pinned back to the RRTMG template above, so
        # every option here is a real override of it.
        overridden.append(radiation)
        if REGISTRY["components"]["radiation"]["options"][radiation]["selectors"] != {
                "ra_lw_physics": 4, "ra_sw_physics": 4}:
            assert settings.get("wrf_rrtmg_compatibility") == "none", radiation
        if report["launchable"]:
            assert run_door(settings, nested=False) is None, radiation
    assert overridden, "no radiation option overrode the RRTMG template"


def _both_doors(choice: dict[str, str]) -> tuple[dict, str, list[str]]:
    route_id = "tools.prepared_domain_tree_forecast"
    route = REGISTRY["runner_routes"][route_id]
    report = validate_physics_plan(plan_for(route_id, route, "era5", choice))
    plan_text = "\n".join(error["message"] for error in report["errors"])
    run_refusals = [
        refusal for index, resolved in enumerate(report["resolved_domains"])
        if (refusal := run_door(resolved["settings"], nested=index > 0))]
    return report, plan_text, run_refusals


def test_the_two_doors_agree_on_the_tuples_3_0_1_split():
    """The 3.0.1 split, one tuple per family, fast.

    The matrix above found 85 to 101 tuples per cell that plan review
    called LAUNCHABLE and validate_run_config refused, in three families:
    RCON (mp=29) on anything, refused only at run start by its speed guard
    RCON-SPEED-01; and UDM (27) and MAD-WRF (96) on RTE+RRTMGP, whose
    refused_when rule tools/build_registry.py wrote before the passes that
    rewrite those two options whole, so the registry shipped without it.

    Now RCON launches at both doors (its guard named a speed gap, not a
    breakage), and UDM and MAD-WRF are refused at both, each door leading
    with the RTE+RRTMGP adapter's own recorded reason.
    """
    from woof.core.rrtmgp import _NO_CLOUD_OPTICS_COUPLING

    anchor = anchor_components()
    assert anchor["radiation"] == "rte-rrtmgp", anchor

    report, plan_text, run_refusals = _both_doors(
        dict(anchor, microphysics="rcon-mp29"))
    assert report["launchable"] is True, plan_text
    assert run_refusals == []

    for option_id, selector in (("udm-mp27", 27), ("madwrf-mp96", 96)):
        report, plan_text, run_refusals = _both_doors(
            dict(anchor, microphysics=option_id))
        assert report["launchable"] is False, (option_id, run_refusals)
        assert run_refusals, (option_id, plan_text)
        reason = _NO_CLOUD_OPTICS_COUPLING[selector]
        assert reason in plan_text, (option_id, plan_text)
        assert all(reason in refusal for refusal in run_refusals), (
            option_id, run_refusals)
        assert "rrtmg_legacy" in plan_text and obeys_the_law(plan_text)
        # The remedy the plan door names clears both doors.
        legacy = validate_physics_plan(plan_for(
            "tools.prepared_domain_tree_forecast",
            REGISTRY["runner_routes"]["tools.prepared_domain_tree_forecast"],
            "era5", dict(anchor, microphysics=option_id), "rrtmg_legacy"))
        assert legacy["launchable"] is True, (option_id, legacy["errors"])
        for index, resolved in enumerate(legacy["resolved_domains"]):
            assert run_door(resolved["settings"], nested=index > 0) is None


def test_pleim_xiu_land_needs_a_surface_layer_at_both_doors():
    """The fourth 3.0.1 family: Pleim-Xiu land with the surface layer off.

    validate_run_config refuses every active land surface with
    sf_sfclay_physics=0 (the exchange fields it reads would stay zero);
    plan review carried that rule for Noah, Noah-MP, RUC and the slab but
    not for Pleim-Xiu land, whose option the builder writes whole with no
    constraints, so a sampled PX tuple with PBL and surface layer off was
    LAUNCHABLE at the plan and refused at run start.
    """
    anchor = anchor_components()
    report, plan_text, run_refusals = _both_doors(dict(
        anchor, land_surface="pleim-xiu", surface_layer="off", pbl="off"))
    assert run_refusals and "requires a surface layer" in run_refusals[0]
    assert report["launchable"] is False, run_refusals
    assert "exchange fields" in plan_text and obeys_the_law(plan_text)
    # With its own surface layer and PBL the pairing launches at both.
    report, plan_text, run_refusals = _both_doors(dict(
        anchor, land_surface="pleim-xiu", surface_layer="pleim-xiu",
        pbl="acm2"))
    assert report["launchable"] is True, plan_text
    assert run_refusals == []


def test_plan_review_names_the_rcon_table_stream_the_run_refuses_without(
        tmp_path, monkeypatch):
    """RCON's other door: the table stream the run cannot start without.

    With its speed guard gone the run admits mp_physics=29 and then needs
    one 381 MB stream woof neither ships nor builds.  Absent, the run
    refuses by name before any step (woof.tables); the option declared no
    asset row, so plan review said nothing at all about it.  It now reports
    the same file, and every root it searched, in ``install_state`` (which
    by the registry's rule does not change ``launchable``).
    """
    from woof.core.rcon import TABLE_BYTES, TABLE_FILENAME
    from woof.table_assets import MissingTableAssets
    from woof.tables import _ensure_for_load

    monkeypatch.setenv("WOOF_RCON_TABLE_ROOT", str(tmp_path))
    choice = dict(anchor_components(), microphysics="rcon-mp29")

    def unresolved(report: dict) -> list[dict]:
        return [note for note in report["install_state"]
                if note["code"] == "asset-unresolved"
                and "rcon" in note["path"]]

    report, plan_text, run_refusals = _both_doors(choice)
    assert report["launchable"] is True, plan_text
    assert run_refusals == []
    notes = unresolved(report)
    assert len(notes) == len(report["resolved_domains"]), report["install_state"]
    for note in notes:
        assert TABLE_FILENAME in note["message"]
        assert str(tmp_path) in note["message"]
        assert "fetch-tables" in note["message"]
    # What the run does with the same settings on the same machine.
    for resolved in report["resolved_domains"]:
        with pytest.raises(MissingTableAssets) as refusal:
            _ensure_for_load(resolved["settings"], fetch=False)
        assert str(tmp_path / TABLE_FILENAME) in str(refusal.value)
        assert "fetch-tables --schemes rcon" in str(refusal.value)

    # A file of the pinned size answers plan review (it checks existence
    # and size; the loader verifies SHA-256 before any table is used).
    with open(tmp_path / TABLE_FILENAME, "wb") as stream:
        stream.truncate(TABLE_BYTES)
    report, plan_text, run_refusals = _both_doors(choice)
    assert report["launchable"] is True, plan_text
    assert unresolved(report) == []


def test_plan_review_gives_spp_conv_on_grell_freitas_the_run_s_verdict():
    """The one spp_conv pairing with a consumer gets a verdict, not a crash.

    Plan review calls validate_spp_config on a stand-in that carried no
    ``cu_gf_version``.  The check reads it only once cu_physics is 3, so
    spp_conv=1 with Grell-Freitas -- the only pairing the run can accept --
    ended plan review in an AttributeError.  Both doors now give the same
    answer: refused by name on the default generation (GFL carries no SPP
    pattern), launchable where the run itself selects wrf_461 GFDRV.
    """
    route_id = "tools.prepared_domain_tree_forecast"
    route = REGISTRY["runner_routes"][route_id]
    assert {"spp_conv", "lightning_option"} <= set(
        route["allowed_parameter_keys"])
    choice = dict(anchor_components(), cumulus="grell-freitas")

    def both(parameters: dict) -> tuple[dict, list[str]]:
        plan = plan_for(route_id, route, "era5", choice)
        for domain in plan["domains"]:
            domain.setdefault("parameters", {}).update(parameters)
        report = validate_physics_plan(plan)
        return report, [
            refusal for index, resolved in enumerate(report["resolved_domains"])
            if (refusal := run_door(resolved["settings"], nested=index > 0))]

    report, run_refusals = both({"spp_conv": 1})
    assert report["launchable"] is False
    plan_text = "\n".join(error["message"] for error in report["errors"])
    assert {error["code"] for error in report["errors"]} == {"spp-consumer"}
    assert run_refusals and all(
        "spp_conv=1 requires cu_gf_version=wrf_461" in refusal
        for refusal in run_refusals), run_refusals
    assert "spp_conv=1 requires cu_gf_version=wrf_461" in plan_text

    # PR92 lightning from the cumulus cloud top makes RunConfig select the
    # wrf_461 GFDRV, which reads the pattern: launchable at both doors.
    report, run_refusals = both({"spp_conv": 1, "lightning_option": 11})
    assert report["launchable"] is True, report["errors"]
    assert run_refusals == []


# ---------------------------------------------------------------------------
# The pricing invariant, on the user-report width.
# ---------------------------------------------------------------------------

#: The user report's geometry class: a 572x524x49 single domain with
#: pinned tiles.  The user's own 250x250 tiling (two buffers of a 286x286
#: window, 55% of the columns each) and a fractional one whose buffers
#: hold well under a fifth of the domain each.
_PRICING_NX, _PRICING_NY, _PRICING_NZ = 572, 524, 49
_PRICING_TILINGS = (250, 100)
_GIB = 1024 ** 3

_PRICING_TOML = """\
[experiment]
name = "physics-combination-matrix-pricing"
start_time = {start_time}
run_seconds = {run_seconds}
restart_interval_s = 0.0
{acknowledgements}
[projection]
map_proj = "lambert"
ref_lat = {ref_lat}
ref_lon = {ref_lon}
truelat1 = 30.0
truelat2 = 45.0
stand_lon = {ref_lon}

[shared]
nz = {nz}
ztop = 20000.0
map_proj = 1
time_step_sound = 4
{shared}

[tiles]
mode = "on"
tile_nx = {tile}
tile_ny = {tile}
nbuffers = 2

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = {nx}
ny = {ny}
time_step = 12
dx = 2400.0
specified = true
nested = false
history_interval_s = 900.0
"""

#: The resolved settings the pricing TOML carries: the selectors and the
#: companions the user report's own config carried.
_PRICING_KEYS = (
    "mp_physics", "cu_physics", "bl_pbl_physics", "sf_sfclay_physics",
    "sf_surface_physics", "ra_lw_physics", "ra_sw_physics", "km_opt",
    "moist", "moist_cq", "num_soil_layers", "cudt_minutes", "ra_physics",
    "ra_rrtmg_variant", "wrf_rrtmg_compatibility", "icloud", "radt")


def _toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)


def _pricing_experiment(settings: dict, tile: int):
    from tools import report_physics_composition_walk as walk
    from woof.experiment import build_experiment

    shared = {key: settings[key] for key in _PRICING_KEYS if key in settings}
    acknowledgements = list(walk.radiation_acknowledgements(shared))
    if (shared.get("km_opt") == 0
            and shared.get("bl_pbl_physics") != SASE_PBL_SCHEME):
        shared["km_opt_zero_acknowledgement"] = KM_OPT_ZERO_ACK
    declared = (
        "acknowledgements = ["
        + ", ".join(f'"{value}"' for value in acknowledgements) + "]\n"
        if acknowledgements else "")
    body = "\n".join(f"{key} = {_toml_scalar(value)}"
                     for key, value in sorted(shared.items()))
    text = _PRICING_TOML.format(
        start_time=walk.START_TIME, run_seconds=walk.RUN_SECONDS,
        ref_lat=walk.REF_LAT, ref_lon=walk.REF_LON, nz=_PRICING_NZ,
        shared=body, acknowledgements=declared, tile=tile,
        nx=_PRICING_NX, ny=_PRICING_NY)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_experiment(tomllib.loads(text),
                                source="physics-combination-matrix.toml")


def _pricing_machine():
    from woof.core import preflight as pf
    from tilestream import autoplan as ap

    profile = pf.DeviceLocalMemoryProfile(
        "NVIDIA GeForce RTX 3080", 68, 1536, bare_context_bytes=182452224)
    return profile, ap.Machine(int(10 * _GIB), int(64 * _GIB),
                               device_profile=profile)


def test_a_streamed_forecast_never_prices_above_the_resident_one():
    """Every launchable tuple of one experiment cell, priced both ways at
    the user-report width on two tilings.  A tuple the pricing road
    refuses by name (Noah-MP on a card whose compile platform was not
    read) or the experiment door refuses by name is recorded, held to the
    law, and not counted as priced.

    The invariant is asserted where streaming HOLDS LESS THAN HALF THE
    DOMAIN: ``nbuffers`` windows together covering at most half the
    columns.  Two 282x282 buffers of a 572x524 domain cover 53 % of it,
    and each buffer carries its own radiation storage, which the tile
    memory model measured as a real per-buffer allocation
    (woof/core/prepared_tile_memory.py: the same domain and tiling
    peaked at 11.11 GiB on the card against a 11.8 GiB resident price),
    so at that tiling the streamed price legitimately meets the resident
    one and the comparison says nothing about double counting.  Those
    cases are recorded and must all be the large tiling."""
    from woof.core import preflight as pf

    cell = walk_cell("tools.prepared_domain_tree_forecast", "era5")
    assert cell.launchable_settings, "no launchable tuple to price"
    profile, machine = _pricing_machine()
    priced = 0
    not_priced: list[str] = []
    over_half: list[str] = []
    violations: list[str] = []
    for key, settings in sorted(cell.launchable_settings.items(),
                                key=lambda item: json.dumps(item[0])):
        label = json.dumps(key)
        for tile in _PRICING_TILINGS:
            try:
                experiment = _pricing_experiment(settings, tile)
                phases = pf.estimate_phases(
                    experiment, source="icon-eu", profile=profile,
                    machine=machine, forcing_intervals=6,
                    forcing_interval_seconds=3600.0)
            except ModuleNotFoundError as exc:
                # An optional device dependency this box does not carry
                # (cupy on a CPU node) is not a refusal of the composition;
                # the tuple is recorded as unpriced here.
                not_priced.append(f"{label} tile {tile}: {exc}")
                break
            except Exception as exc:  # noqa: BLE001 -- the refusal is the record
                text = f"{type(exc).__name__}: {exc}"
                if not obeys_the_law(text):
                    violations.append(f"{label} tile {tile}: pricing refusal "
                                      f"{law_verdict(text)} -- {text[:300]}")
                not_priced.append(f"{label} tile {tile}: {text[:160]}")
                break
            streamed = phases.streamed
            if streamed is None:
                not_priced.append(f"{label} tile {tile}: did not stream")
                continue
            priced += 1
            held = (int(streamed.nbuffers) * int(streamed.window_nx)
                    * int(streamed.window_ny))
            if 2 * held > _PRICING_NX * _PRICING_NY:
                over_half.append(f"{label} tile {tile}")
                continue
            resident = phases.resident_forecast_envelope_bytes
            if streamed.peak_vram_bytes > resident:
                violations.append(
                    f"{label} tile {tile}: streamed "
                    f"{streamed.peak_vram_bytes / _GIB:.2f} GiB > resident "
                    f"{resident / _GIB:.2f} GiB (window "
                    f"{streamed.window_nx}x{streamed.window_ny}, rung "
                    f"{streamed.rung})")
    assert priced > 0, not_priced[:10]
    assert priced > len(over_half), "every priced case held over half the domain"
    assert all(case.endswith(f"tile {max(_PRICING_TILINGS)}")
               for case in over_half), over_half[:10]
    assert violations == [], (
        f"{len(violations)} of {priced} priced cases break the invariant; "
        f"{len(not_priced)} not priced:\n  " + "\n  ".join(violations[:30]))


if __name__ == "__main__":
    report = matrix_report()
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()
