"""``implemented`` and reachable are independent, so both are declared.

Three schemes reached bitwise parity and had ``implemented: true`` written on
them, and a user could select none of them.  Every template pinned the same
triple (``pbl: ysu``, ``surface_layer: classic-mm5``, ``land_surface: noah``),
the tree route allowed only ``microphysics`` and ``cumulus`` to vary, and the
benchmark route allowed nothing -- so ``implemented: true`` was necessary and
nowhere near sufficient, and nothing said so.

The naive gate would be the biconditional ``implemented <=> reachable``.  It is
the wrong gate, because reachability has meaningful levels.  ``pbl: off``,
``surface_layer: revised-mm5`` and ``radiation: off`` are implemented and
normally reachable as option-scoped component overrides; Noah-MP is implemented
but expert-template-only; and the analytic clear-sky proxy is implemented but
deliberately outside registry reachability because it is not a WRF scheme.

So the intended relationship is DECLARED, per option, in
``reachability.state``, and this file recomputes every state from the templates
and the routes and fails on a difference.  What that buys over the naive gate:
an option that becomes reachable by accident -- a template edit, a route
widening, a source gaining a template list -- fails here with the state it
actually has, and an option that quietly stops being reachable fails the same
way.  The half of the biconditional that is not negotiable is still enforced:
an option that is not implemented must be declared unreachable, and an
implemented one declared unreachable must name its blocker.

The blockers are not prose.  ``test_the_ruc_blocker_is_still_true`` re-runs the
ingest refusal the RUC row cites, so the declaration cannot outlive the reason
for it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from woof.physics_registry import (expert_template_ids_for_source,
                                    physics_registry)

#: Easiest first.  An option reachable more than one way is declared by its
#: easiest path, because that is the one a user will find.
_STATE_ORDER = ("template", "component-override", "expert-template",
               "unreachable")

_ROOT = Path(__file__).resolve().parents[1]


def test_mynn_ruc_and_noahmp_templates_follow_existing_lsm_routes():
    """Expose the pairings without broadening either LSM's source contract."""
    registry = physics_registry()
    ruc_id = "wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1"
    mynn_ruc_id = (
        "wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1")
    noahmp_id = (
        "wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1")
    mynn_noahmp_id = (
        "wsm6-mynn-mynn-noahmp-no-radiation-expert-only-v1")

    for template_id, lsm in (
            (mynn_ruc_id, "ruc-lsm"), (mynn_noahmp_id, "noah-mp")):
        assert registry["templates"][template_id]["components"] | {
            "microphysics": "wsm6-mp6",
            "pbl": "mynn",
            "surface_layer": "mynn",
            "land_surface": lsm,
        } == registry["templates"][template_id]["components"]
    assert not any(
        "differs from wsm6-ysu-mm5-noah-no-radiation-v1 in exactly ONE"
        in warning
        for warning in registry["templates"][mynn_ruc_id]["warnings"]
    )

    for lsm in ("ruc-lsm", "noah-mp"):
        required = registry["components"]["land_surface"]["options"][lsm][
            "constraints"]["requires_components"]["surface_layer"]
        assert "mynn" in required

    for route in registry["runner_routes"].values():
        for declared in route.get("source_template_ids", {}).values():
            assert (ruc_id in declared) == (mynn_ruc_id in declared)
        for declared in route.get("expert_template_ids", {}).values():
            assert (noahmp_id in declared) == (mynn_noahmp_id in declared)


def _computed_states(registry: dict) -> dict[tuple[str, str], str]:
    """Recompute every option's reachability from templates and routes.

    Mirrors :func:`woof.physics_registry.validate_physics_plan`'s own rules,
    including the one that used to make this question unanswerable: a route
    that declares any template list must be exhaustive, and a source it
    registers with no list reaches no template.
    """

    components = registry["components"]
    templates = registry["templates"]
    routes = registry["runner_routes"]

    reached: dict[tuple[str, str], set[str]] = {
        (component_id, option_id): set()
        for component_id, component in components.items()
        for option_id in component["options"]
    }

    for route in routes.values():
        if route.get("implemented") is not True:
            continue
        normal = route.get("source_template_ids", {}) or {}
        expert = route.get("expert_template_ids", {}) or {}
        declares = bool(normal) or bool(expert)
        per_domain = route.get("mode") == "experiment-per-domain"
        overridable = set(route.get("allowed_component_overrides", []) or [])
        option_overrides = route.get("allowed_component_options", {}) or {}
        expert_selector_keys = set(
            route.get("allowed_expert_selector_keys", []) or [])

        for source_id in route.get("source_ids", []) or []:
            if declares:
                normal_ids = list(normal.get(source_id, []) or [])
                expert_ids = expert_template_ids_for_source(route, source_id)
            else:
                # No declaration at all: every registered template is legal.
                normal_ids, expert_ids = list(templates), []
            for template_id in normal_ids:
                for component_id, option_id in templates[
                        template_id]["components"].items():
                    reached[(component_id, option_id)].add("template")
            for template_id in expert_ids:
                for component_id, option_id in templates[
                        template_id]["components"].items():
                    reached[(component_id, option_id)].add("expert-template")
            if not (normal_ids or expert_ids):
                # An override still needs a base template to override.
                continue
            for component_id, component in components.items():
                selector_keys = set(component.get("selector_keys", []) or [])
                by_override = per_domain and component_id in overridable
                by_selector = bool(selector_keys) and selector_keys <= (
                    expert_selector_keys)
                admitted_options = set(
                    option_overrides.get(component_id, []) or [])
                if not (by_override or by_selector or admitted_options):
                    continue
                for option_id in component["options"]:
                    if (
                        not (by_override or by_selector)
                        and option_id not in admitted_options
                    ):
                        continue
                    reached[(component_id, option_id)].add(
                        "component-override")

    states: dict[tuple[str, str], str] = {}
    for (component_id, option_id), ways in reached.items():
        option = components[component_id]["options"][option_id]
        if option.get("implemented") is not True:
            # Nameable is not reachable.  ``microphysics: sase`` sits inside the
            # tree route's allowed_component_overrides, so a plan CAN name it --
            # and woof/physics_registry.py refuses it with
            # ``unimplemented-option`` on every route, from every template, for
            # every source.  Reachable here means "a plan naming it can be
            # launchable", so an unimplemented option is unreachable by
            # construction, and
            # test_an_unimplemented_option_is_refused_by_the_resolver is what
            # holds that construction to the resolver's actual behaviour rather
            # than to this comment.
            states[(component_id, option_id)] = "unreachable"
            continue
        states[(component_id, option_id)] = next(
            (state for state in _STATE_ORDER if state in ways), "unreachable")
    return states


def _declared_states(registry: dict) -> dict[tuple[str, str], dict]:
    return {
        (component_id, option_id): option.get("reachability")
        for component_id, component in registry["components"].items()
        for option_id, option in component["options"].items()
    }


def test_every_option_declares_its_reachability() -> None:
    registry = physics_registry()
    missing = sorted(
        f"{component_id}.{option_id}"
        for (component_id, option_id), declaration
        in _declared_states(registry).items()
        if not isinstance(declaration, dict)
        or declaration.get("state") not in _STATE_ORDER
    )
    assert missing == [], (
        "these options declare no usable reachability.state; a new option must "
        f"say how a user reaches it, or that nobody can: {missing}")


def test_declared_reachability_matches_the_templates_and_routes() -> None:
    registry = physics_registry()
    computed = _computed_states(registry)
    declared = _declared_states(registry)
    wrong = sorted(
        f"{component_id}.{option_id}: declared "
        f"{declared[(component_id, option_id)]['state']!r}, actually "
        f"{state!r}"
        for (component_id, option_id), state in computed.items()
    if declared[(component_id, option_id)]["state"] != state)
    assert wrong == [], (
        "the registry's reachability declarations disagree with what its own "
        "templates and routes make selectable:\n  " + "\n  ".join(wrong))


def test_every_implemented_option_is_selected_by_a_registered_suite() -> None:
    """AUDIT R-067: implemented, and offered by a NAMED suite.

    ``reachability`` already says an option is selectable; this says
    something narrower and, for a user, more useful -- that a registered
    TEMPLATE selects it, so the option can be asked for by name instead of
    only by hand-composing a tuple.  Eleven implemented schemes failed
    that on 2026-09-10: Milbrandt-Yau, WDM6, aerosol-aware Thompson, New
    Tiedtke, Grell-Freitas, MYJ, the Eta surface layer, SASE, the revised
    MM5 surface layer and three of the five turbulence closures were each
    ``implemented: true`` with no suite anywhere.  That is the
    ship-only-what-users-can-reach rule failing quietly, and nothing said
    so.

    An option that deliberately has no suite names its reason here, so
    retiring one is a grep rather than an argument.  The degradations do:
    a run with no land surface, no surface layer, no microphysics or no
    radiation is a thing a user asks for on purpose, and shipping a
    PRESET for it would be recommending it.
    """

    registry = physics_registry()
    selected = {
        (component_id, option_id)
        for template in registry["templates"].values()
        for component_id, option_id in template["components"].items()
    }
    cited = {
        ("cumulus", "grell-3d"):
            "column and diagnostic conformance is measured, but no composed "
            "forecast or observation score qualifies a recommended suite; "
            "a config or imported WRF namelist selects cu_physics=5 directly",
        ("pbl", "temf"): "the retained upper-level mass-flux recurrence produces unreasonable diagnostics; no named suite recommends this nonoperational pair",
        ("surface_layer", "temf"): "paired TEMF upper-level mass flux is nonoperational; no named suite recommends it",
        ("pbl", "qnse"):
            "column conformance and a short CLI run are not observation "
            "verification; the QNSE pair remains available by its public "
            "WRF namelist selectors, without a recommended preset",
        ("surface_layer", "qnse"):
            "the QNSE surface is selected with its PBL through public "
            "WRF namelist metadata; no preset recommends the pair before "
            "observation verification",
        ("land_surface", "off"): "a degradation, not a suite to recommend",
        ("surface_layer", "off"): "a degradation, not a suite to recommend",
        ("microphysics", "off"): "a degradation, not a suite to recommend",
        ("radiation", "off"): "a degradation, not a suite to recommend",
        ("radiation", "analytic-clear-sky"):
            "not a forecast radiation scheme: no cloud, aerosol or gas "
            "optics, so it stays behind the expert acknowledgement",
        ("radiation", "gsfc-shortwave"):
            "WRF old GSFC shortwave has matched column outputs but no "
            "whole-forecast or observation score, so a component override "
            "or an imported ra_sw_physics=2 namelist asks for it directly",
        **{("urban", option_id):
           "an urban canopy is added to a suite in its run's [shared] "
           "table for a city domain; no preset recommends one before its "
           "observation verification is recorded"
           for option_id in ("slucm", "bep", "bep-bem")},
        # Port lanes expose these through the stock namelist and component
        # overrides. Their native replay and individual real cases do not
        # establish an observation-qualified recommended suite.
        **{key: f"{selector} is available through a stock namelist or component override; its port receipt does not establish a recommended suite"
           for key, selector in (
               (("cumulus", "cam-deep"), "cu_physics=7"),
               (("land_surface", "pleim-xiu"), "sf_surface_physics=7"),
               (("microphysics", "cam-mg-mp11"), "mp_physics=11"),
               (("microphysics", "ntu-mp56"), "mp_physics=56"),
               (("pbl", "acm2"), "bl_pbl_physics=7"),
               (("radiation", "relaxation-31"), "ra_lw_physics=31/ra_sw_physics=31"),
               (("surface_layer", "pleim-xiu"), "sf_sfclay_physics=7"),
               (("cumulus", "gfs-sas"), "cu_physics=94"),
               (("cumulus", "kf-cup"), "cu_physics=10"),
               (("cumulus", "kiaps-sas"), "cu_physics=14"),
               (("microphysics", "gsfcgce-mp97"), "mp_physics=97"),
               (("microphysics", "p3-mp51"), "mp_physics=51"),
               (("microphysics", "p3-mp52"), "mp_physics=52"),
               (("microphysics", "p3-mp53"), "mp_physics=53"),
               (("pbl", "gbm"), "bl_pbl_physics=12"),
               (("pbl", "mrf"), "bl_pbl_physics=99"),
               (("radiation", "wrf-gfdleta"), "ra_lw_physics=99/ra_sw_physics=99"),
               (("turbulence", "sms-3dtke"), "km_opt=5"))},
        ("microphysics", "ishmael-mp55"):
            "Native adaptive-habit ice is numerically checked as a column but has no observation-qualified suite; imported namelists and per-domain component overrides select it directly",
        ("cumulus", "betts-miller-janjic"):
            "BMJ is admitted by cu_physics=2 in configurations and imported "
            "WRF namelists. Its column conformance and restart checks do "
            "not establish forecast skill, so no preset recommends it "
            "before observation verification is recorded",

        ("cumulus", "kain-fritsch-cu99"):
            "the new cu=99 port has column parity but no matched forecast "
            "or observation qualification; public namelists and per-domain "
            "overrides select it without recommending a preset",
        ("pbl", "uw"):
            "WRF v4.7.1's UW moist-turbulence PBL is graded word for word "
            "against WRF's own columns but has no forecast or observation "
            "score yet, so no preset recommends it; a config asks for it "
            "directly (bl_pbl_physics = 9, per domain too) and an imported "
            "WRF namelist carries it",
        ("microphysics", "tempo"):
            "TEMPO is asked for by name through the rrfs-v2-hex-tempo model "
            "preset (woof preset run), an RRFS configuration on the hex "
            "dycore with declared divergences, not a WRF-side template; a "
            "config asks for it directly (mp_physics = \"tempo\")",
        ("cumulus", "tiedtke"):
            "WRF's original Tiedtke (cu_physics = 6) is checked against an "
            "independent column oracle but has no forecast or observation "
            "score yet, so no preset recommends it; a config asks for it "
            "directly (cu_physics = 6)",
        **{("microphysics", option_id):
           f"WRF's {label} is graded word for word against WRF's own "
           "Fortran column but has no forecast or observation score yet, so "
           "no preset recommends it; a config asks for it directly "
           f"(mp_physics = {mp}) and an imported WRF namelist carries it"
           for option_id, label, mp in (
               ("wsm3-mp3", "WSM3 simple-ice scheme", 3),
               ("wsm5-mp4", "WSM5 five-class scheme", 4),
               ("ferrier-aligo-mp5", "Ferrier-Aligo scheme", 5),
               ("wsm6r-mp106", "revised WSM6 (WSM6R)", 106),
               ("wsm7-mp24", "WSM7 seven-class scheme with hail", 24),
                   ("sbu-ylin-mp13", "SBU-YLin five-class scheme", 13))},
        ("radiation", "new-goddard"):
            "WRF's 2017 Goddard radiation (ra_lw/ra_sw_physics = 5) is graded "
            "against WRF's own columns but has no forecast or observation "
            "score yet, so no preset recommends it; a config or an imported "
            "WRF namelist selects it directly",
        ("radiation", "fu-liou-gu"):
            "WRF's 7/7 pair has real-column replay and one three-hour "
            "forecast with station comparisons, which does not qualify a "
            "general recommended suite; a config or a stock WRF namelist "
            "selects ra_lw_physics=7 and ra_sw_physics=7 directly",
        **{("pbl", option):
           f"WRF's {label} has native column and lifecycle checks but no "
           "general observation score for a recommended suite; config and "
           f"stock namelist import select bl_pbl_physics={selector} directly"
           for option, label, selector in (
               ("boulac", "BouLac", 8), ("gfs", "GFS", 3),
               ("keps", "K-epsilon", 17))},
        ("surface_layer", "gfs"):
            "WRF's GFS surface layer has native column and lifecycle checks "
            "but no general observation score for a recommended suite; "
            "config and stock namelist import select sf_sfclay_physics=3",
        ("microphysics", "fast-sbm-mp30"):
            "WRF's FAST_SBM bin scheme (mp_physics = 30) is graded bit for "
            "bit against WRF's own column calls but has no forecast or "
            "observation score yet, so no preset recommends it; `woof "
            "domain --physics-choices` swaps it into any legacy-RRTMG suite "
            "and an imported WRF namelist carries it",
    }

    missing = sorted(
        f"{component_id}.{option_id}"
        for component_id, component in registry["components"].items()
        for option_id, option in component["options"].items()
        if option.get("implemented") is True
        and (component_id, option_id) not in selected
        and (component_id, option_id) not in cited
        and not (option.get("reachability", {}).get("state") == "component-override"
                 and option.get("extensions", {}).get("preset_scope_reason"))
    )
    assert missing == [], (
        "these options are implemented and no registered template selects "
        "them, so a user can only reach them by composing a tuple by hand: "
        f"{missing}. Register a suite, or cite the reason it has none.")

    stale = sorted(
        f"{component_id}.{option_id}"
        for (component_id, option_id) in cited
        if (component_id, option_id) in selected
    )
    assert stale == [], (
        "these citations claim no suite selects the option and one now "
        f"does; retire the citation: {stale}")


def test_an_unimplemented_option_is_refused_by_the_resolver() -> None:
    """The half of the biconditional that is not negotiable.

    ``_computed_states`` calls an unimplemented option unreachable.  That is a
    claim about ``validate_physics_plan``, so it is checked against
    ``validate_physics_plan``: name each one on a plan built from the route and
    template that would otherwise reach it, and require the refusal.
    """

    from woof.physics_registry import (PLAN_SCHEMA, registry_sha256,
                                        validate_physics_plan)

    registry = physics_registry()
    unimplemented = [
        (component_id, option_id)
        for component_id, component in sorted(registry["components"].items())
        for option_id, option in sorted(component["options"].items())
        if option.get("implemented") is not True
    ]
    assert unimplemented, (
        "the registry publishes no unimplemented option, so this gate "
        "measured nothing")

    route_id = "tools.prepared_domain_tree_forecast"
    route = registry["runner_routes"][route_id]
    template_id = route["source_template_ids"]["hrrr"][0]
    for component_id, option_id in unimplemented:
        plan = {
            "schema": PLAN_SCHEMA,
            "plan_id": "unimplemented-option-probe-v1",
            "registry_sha256": registry_sha256(),
            "context": {"source_id": "hrrr", "runner_id": route_id,
                        "topology_id": "one-way-nested-v1"},
            "domains": [
                {"domain_id": "d01", "template_id": template_id,
                 "components": {component_id: option_id}},
                {"domain_id": "d02", "template_id": template_id,
                 "components": {component_id: option_id},
                 "parameters": {"spec_exp": 0.0}},
            ],
            "edges": [{"parent_domain_id": "d01", "child_domain_id": "d02"}],
        }
        report = validate_physics_plan(plan)
        assert report["launchable"] is False, (
            f"{component_id}.{option_id} is not implemented and a plan naming "
            "it is launchable")
        codes = {error["code"] for error in report["errors"]}
        assert codes & {"unimplemented-option", "component-override-route"}, (
            f"{component_id}.{option_id} is refused, but for none of the "
            f"reasons that make it unreachable: {sorted(codes)}")


def test_an_unreachable_implemented_option_names_its_blocker() -> None:
    """The other half, which a biconditional would have got wrong.

    An implemented option nobody can select is not automatically a defect --
    but it is always a claim that owes a reason, and the reason has to be
    written down where the reader of ``implemented: true`` will see it.
    """

    registry = physics_registry()
    declared = _declared_states(registry)
    missing, spurious = [], []
    for (component_id, option_id), declaration in sorted(declared.items()):
        blocker = declaration.get("blocker")
        if declaration["state"] == "unreachable":
            if not isinstance(blocker, str) or len(blocker.strip()) < 40:
                missing.append(f"{component_id}.{option_id}")
        elif blocker is not None:
            spurious.append(f"{component_id}.{option_id}")
    assert missing == [], (
        f"unreachable options with no stated blocker: {missing}")
    assert spurious == [], (
        f"reachable options carrying a blocker: {spurious}")


def test_expert_templates_keep_their_advisory_metadata() -> None:
    """An expert preset retains its warning even when ordinary overrides exist."""
    registry = physics_registry()
    assert any(route.get("expert_template_ids") for route in registry["runner_routes"].values())
    templates = registry["templates"]
    for route_id, route in registry["runner_routes"].items():
        expert = route.get("expert_template_ids", {}) or {}
        if not expert:
            continue
        acknowledgement = route.get("expert_acknowledgement_id")
        assert isinstance(acknowledgement, str) and acknowledgement.strip(), (
            f"{route_id} offers expert templates with no "
            "expert_acknowledgement_id, so the advisory cannot be identified")
        warnings = route.get("expert_warnings") or []
        assert warnings and all(isinstance(text, str) and text.strip()
                                for text in warnings), (
            f"{route_id} offers expert templates with no expert_warnings; an "
            "expert path that does not say what is wrong is just a hidden one")
        for template_ids in expert.values():
            for template_id in template_ids:
                assert template_id in templates, (
                    f"{route_id} offers unregistered expert template "
                    f"{template_id!r}")


def _perturbed(mutate) -> list[str]:
    registry = copy.deepcopy(physics_registry())
    mutate(registry)
    computed = _computed_states(registry)
    declared = _declared_states(registry)
    return sorted(
        f"{component_id}.{option_id}"
        for (component_id, option_id), state in computed.items()
        if declared[(component_id, option_id)]["state"] != state)


def test_the_gate_fails_when_a_template_makes_something_reachable() -> None:
    """A gate nobody has seen fail is not evidence, part one.

    WRF's classic 1/1 RRTM+Dudhia pair is a component override that no
    template selects.  Swap it into a registered template and its easiest
    reachability becomes ``template``; the recomputation must notice that
    undeclared promotion.

    This probe used to be the revised MM5 surface layer, which the audit
    R-067 templates now select -- an option that is already ``template``
    cannot be promoted, so the control measured nothing and said so by
    failing.  Retired here rather than deleted: the gate still needs an
    option nobody has given a suite.
    """

    def mutate(registry: dict) -> None:
        registry["templates"]["wsm6-ysu-mm5-noah-no-radiation-v1"][
            "components"]["pbl"] = "uw"

    wrong = _perturbed(mutate)
    assert "pbl.uw" in wrong, wrong
    # dudhia-shortwave does NOT flip: other templates still select it, which
    # is the asymmetry a per-option declaration captures and a per-template
    # one would not.
    assert "radiation.dudhia-shortwave" not in wrong, wrong


def test_the_gate_fails_when_a_route_widens() -> None:
    """A future opaque option cannot acquire an undeclared selection path."""
    def mutate(registry: dict) -> None:
        options = registry["components"]["pbl"]["options"]
        options["opaque-option"] = copy.deepcopy(options["ysu"])
        options["opaque-option"]["reachability"] = {
            "state": "unreachable", "blocker": "No selection path exists before the route declaration changes."}
        route = registry["runner_routes"]["tools.prepared_domain_tree_forecast"]
        route["allowed_component_options"]["pbl"].append("opaque-option")
    wrong = _perturbed(mutate)
    assert "pbl.opaque-option" in wrong, wrong
    assert "land_surface.noah-mp" not in wrong, wrong


def test_the_gate_fails_when_an_expert_template_loses_its_acknowledgement():
    """Part three: an expert path that stops being one.

    Drop the route's acknowledgement id and the resolver must refuse the expert
    template outright -- ``expert-route-policy`` -- rather than quietly promote
    it to a peer of the base templates.  That is the failure mode that matters:
    a warned expert path decaying into an unwarned normal one.
    """

    from woof.physics_registry import (PLAN_SCHEMA, registry_sha256,
                                        validate_physics_plan)

    registry = copy.deepcopy(physics_registry())
    route = registry["runner_routes"]["tools.prepared_domain_tree_forecast"]
    template_id = route["expert_template_ids"]["hrrr"][0]
    route.pop("expert_acknowledgement_id")

    plan = {
        "schema": PLAN_SCHEMA,
        "plan_id": "expert-policy-probe-v1",
        "registry_sha256": registry_sha256(registry),
        "context": {"source_id": "hrrr",
                    "runner_id": "tools.prepared_domain_tree_forecast",
                    "topology_id": "one-way-nested-v1"},
        "domains": [
            {"domain_id": "d01", "template_id": template_id},
            {"domain_id": "d02", "template_id": template_id,
             "parameters": {"spec_exp": 0.0}},
        ],
        "edges": [{"parent_domain_id": "d01", "child_domain_id": "d02"}],
    }
    report = validate_physics_plan(plan, registry=registry)
    assert report["launchable"] is False
    assert "expert-route-policy" in {
        error["code"] for error in report["errors"]}

    # ...and the shipped route does publish one, so the same plan only needs the
    # consent rather than being structurally broken.
    shipped = validate_physics_plan({
        **plan,
        "registry_sha256": registry_sha256(),
        "acknowledgements": [
            physics_registry()["runner_routes"][
                "tools.prepared_domain_tree_forecast"][
                    "expert_acknowledgement_id"]],
    })
    assert shipped["launchable"] is True, shipped["errors"]


def test_the_ruc_soil_contract_refusal_is_still_true() -> None:
    """Re-run the refusal the RUC row cites, wherever it is now recorded.

    This began as ``test_the_ruc_blocker_is_still_true``: ``ruc-lsm`` was
    declared ``unreachable`` and cited ``woof/ingest/soil_contract.py`` as
    the blocker.  The RUC lane has since registered a RUC template on the
    era5 and gfs single-domain routes, so the option IS reachable and the
    recomputation above is the authority on that -- but the declarative soil
    contract still admits exactly one target, Noah's four layers, so the
    refusal itself did not go away.  It moved from a ``blocker`` to the
    option's warnings, and the point of this test is unchanged: the claim
    must not outlive the code it rests on.  If a soil lane lands a RUC
    target, this fails and the warning has to be revisited.
    """

    from woof.ingest.soil_contract import validate_soil_layer_contract

    composition = json.loads(
        (_ROOT / "configs" / "rw-wps-era5-netcdf-terrain.composition.json"
         ).read_text(encoding="utf-8"))
    contract = copy.deepcopy(composition["soil_layers"])
    validate_soil_layer_contract(contract)          # the Noah target is fine

    # RUC's nine-level grid, as layer bounds rather than WRF's level depths.
    ruc = [0.0, 0.01, 0.04, 0.1, 0.3, 0.6, 1.0, 1.6, 2.2, 3.0]
    contract["target_layers"] = [
        {"top": top, "bottom": bottom}
        for top, bottom in zip(ruc, ruc[1:])
    ]
    assert len(contract["target_layers"]) == 9
    with pytest.raises(ValueError, match="target_layers differ"):
        validate_soil_layer_contract(contract)

    option = physics_registry()["components"]["land_surface"][
        "options"]["ruc-lsm"]
    assert option["reachability"]["state"] == "template"
    assert "blocker" not in option["reachability"], (
        "ruc-lsm is reachable through a registered template, so a blocker "
        "here would be a declaration contradicting the routes")
    assert any("soil_contract.py" in warning
               for warning in option.get("warnings", [])), (
        "the soil-contract refusal above still fires, so the option must "
        "still say so somewhere a user reads")


def test_the_mynn_suite_is_reachable_as_a_pair() -> None:
    """The lane's own deliverable, asserted rather than assumed."""

    registry = physics_registry()
    computed = _computed_states(registry)
    assert computed[("pbl", "mynn")] == "template"
    assert computed[("surface_layer", "mynn")] == "template"

    # ...and only ever together, because half a suite is a different model.
    for template_id, template in registry["templates"].items():
        pair = (template["components"]["pbl"] == "mynn",
                template["components"]["surface_layer"] == "mynn")
        assert pair[0] == pair[1], (
            f"template {template_id!r} selects half the MYNN suite; "
            "woof/physics_compat.py refuses that at runtime, so a template "
            "offering it would be a launchable-looking dead end")


def test_no_source_a_route_serves_is_published_as_reaching_nothing() -> None:
    """A served source declares a suite list, or says why it names none.

    ``source_template_ids`` is the reachability declaration a front end
    reads, and an EMPTY list there is not a silence: a route that declares
    any list at all is exhaustive, so the source reaches no named suite,
    and ``expert_template_ids_for_source`` withholds the route-wide expert
    list from it as well.  Completing a served-but-undeclared source with
    an empty list for want of a measurement therefore takes every front
    door away from a source the runner runs: it took the six composition
    suites and the three Noah-MP expert suites from the single-domain
    route's aigfs and era5-l137, which reach that runner through the same
    generic mapped route as the siblings that declare all nine.

    The one source that may name nothing is the caller-supplied
    composition, whose physics the caller states.

    Asked of FIXED-TEMPLATE routes, where a registered template is the
    only door: an experiment-per-domain route composes from its own
    component declaration instead, so a source with no template list there
    still reaches the options the route lists per domain, which is a
    different shape rather than an empty one.
    """

    registry = physics_registry()
    names_no_suite = {"mapped"}
    for runner_id, route in registry["runner_routes"].items():
        if route.get("mode") != "fixed-template":
            continue
        declared = route.get("source_template_ids") or {}
        if not declared:
            # A route that declares no list anywhere is not exhaustive:
            # every registered template is legal on it, so there is no
            # emptiness to read.
            continue
        for source_id in route.get("source_ids", []) or []:
            if source_id in names_no_suite:
                assert not declared.get(source_id), (
                    f"{runner_id} declares suites for {source_id}, whose "
                    "composition is the caller's")
                continue
            offered = list(declared.get(source_id) or [])
            offered += expert_template_ids_for_source(route, source_id)
            assert offered, (
                f"runner route {runner_id} serves {source_id} and publishes "
                "no named suite for it at all, so every front door that "
                "reads the declaration offers that source nothing")


def test_the_generic_basis_carries_no_source_specific_evidence() -> None:
    """What an unmeasured source is priced from names no other source.

    The basis is the suites EVERY declared source of the route names, so a
    row one source's own verification minted cannot survive it.  Asserted
    against the two sources that are priced from it, because the first,
    route-wide version of that completion handed a source another source's
    evidence-scoped registrations.
    """

    registry = physics_registry()
    route = registry["runner_routes"]["tools.prepared_single_domain_forecast"]
    declared = route["source_template_ids"]
    priced = ("aigfs", "era5-l137")
    measured = [source_id for source_id in route["source_ids"]
                if source_id not in priced and declared.get(source_id)]
    assert len(measured) >= 2, measured
    compared = 0
    for source_id in priced:
        # A priced source with an EMPTY list is the defect this pass
        # exists to close, and it also made every assertion below
        # iterate nothing: the guard passed on the tree where the
        # emptiness lived.  A silent pass is not a pass.
        assert declared[source_id], source_id
        for template_id in declared[source_id]:
            compared += 1
            assert all(template_id in declared[other] for other in measured), (
                f"{template_id} is priced onto {source_id} and is not a suite "
                "every measured source of this route names, so it carries "
                "another source's evidence")
    assert compared, (priced, {s: declared[s] for s in priced})
