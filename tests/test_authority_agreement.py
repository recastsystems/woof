"""The two gatekeepers must not disagree about any configuration.

GPUWM has two independent authorities over "may this physics suite run", and
until this file existed nothing compared them:

* the registry's ``constraints`` machinery -- ``required_settings``,
  ``forbidden_setting_values`` and ``requires_components`` -- evaluated by
  :func:`woof.physics_registry.validate_physics_plan`, which is what a
  launcher shows a user *before* anything is prepared; and
* :func:`woof.config.validate_run_config`, which every RunConfig passes
  through on both the legacy and the experiment path, and which is therefore
  what actually decides whether a run starts.

They disagreed.  For the MYNN 5/5 suite with Noah the registry raised
``component-dependency: option 'noah' requires surface_layer in
[revised-mm5, classic-mm5]`` while ``validate_run_config`` returned OK -- and
that pair is the exact configuration the MYNN PBL option's own warning cites
as its runtime evidence (``tests/test_mynn_pbl_runtime.py:49``).  WRF was the
tiebreaker and the registry was wrong: ``share/module_check_a_mundo.F`` has no
``sf_sfclay_physics`` constraint for ``sf_surface_physics=2``, and
``phys/module_surface_driver.F:2386-2390`` hands ``SFCLAY_mynn`` the same
``chs/chs2/cqs2/cpm/flhc/flqc/qgh/qsfc`` exchange coefficients
``CASE (LSMSCHEME)`` consumes.  The constraint is gone; this file is what stops
the next one.

Two layers, because they answer different questions:

**Exhaustive component cross-product.**  All 4,032 combinations of the six
components' registered options, run through the shipped constraint evaluator
and through ``validate_run_config`` on the settings that evaluator resolved.
This is exhaustion, not sampling: the property is proven over the whole space
rather than over examples.  It needs a permissive runner route, because the
shipped routes deliberately do not let a user vary the PBL or the land-surface
model per domain -- so the route is synthesised here, and ONLY the route.  The
constraints, the options, the parameter specs and both authorities' code are
the shipped ones.  This is the layer the MYNN/Noah bug lived in, and note that
a reachable-only enumeration would have MISSED it: MYNN was unreachable, so
the disagreement was invisible from the selectable surface.

**Reachable surface.**  Every plan a user can actually build from the shipped
templates, routes, allowed component overrides and allowed per-domain
parameters.  Here the direction that matters is absolute: a plan the registry
calls launchable must survive the runtime's own battery, or the GUI is
offering something that will not start.

Scope, stated so it cannot drift: the two authorities do not have the same
field of view.  A route, a topology, a template registration or a tree-edge
transition is a question a single per-domain RunConfig cannot answer, and
``_REGISTRY_ONLY_CODES`` names those.  ``_SHARED_CODES`` names the ones both
authorities own, and a refusal carrying any of those must be a refusal on both
sides.  ``_INSTALL_STATE_CODES`` names a third kind, added after the mp=28
dataset refusal was tried in both of the other two: a question about the
MACHINE rather than about the configuration.  Those do not travel in
``errors`` at all -- ``validate_physics_plan`` reports them in
``install_state`` and leaves ``launchable`` to the plan -- and one that
turns up in ``errors`` anyway fails this gate, as does an error code in
neither of the other two sets, so a new code has to be classified by
whoever adds it.
"""

from __future__ import annotations

import dataclasses
import itertools
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import RunConfig, validate_run_config
from woof.physics_compat import NOAHMP_MEASURED_COLUMN_CEILING
from woof.physics_registry import (
    expert_template_ids_for_source,
    INSTALL_STATE_CODES,
    PLAN_SCHEMA,
    physics_registry,
    registry_sha256,
    validate_physics_plan,
)

#: Questions a per-domain RunConfig cannot answer, so the registry may refuse
#: them alone: they are about routing, topology, plan shape or tree edges.
_REGISTRY_ONLY_CODES = frozenset({
    # A requirement row that names no files at all.  Not an install
    # question -- it reads the same on every machine -- but a registry
    # that cannot be resolved anywhere, and a RunConfig has no field for
    # it.  tools/build_registry.py refuses to emit one; this is the
    # reader's half, so a registry from anywhere cannot buy a vacuous
    # pass (audit R-045).
    "asset-undeclared",
    "component-override-route",
    # AUDIT R-005.  A conditional refusal scoped by ``sources``: native HRRR
    # supplies analyzed condensate that microphysics-off cannot retain.  A
    # RunConfig carries no source identity, so this is the one refusal in
    # the constraint battery the per-domain authority cannot mirror -- and
    # it has its own code so excusing it does not excuse the rest of that
    # battery.
    "component-source-refusal",
    "expert-route-policy",
    "graph-setting-constraint",
    # AUDIT R-058.  ``nonuniform-base-template`` compared template LABELS and
    # is retired; this is its replacement, and it is still registry-only for
    # the reason the old code was: a per-domain RunConfig is ONE domain, so
    # nothing built from it can see that a sibling resolved a different value
    # for a setting the tree loader carries once.
    "nonuniform-shared-setting",
    "parameter-route",
    # AUDIT R-005.  Both halves are registry-only for one reason: a
    # RunConfig carries the resolved boolean and no route, so it cannot
    # distinguish a stated moist=false from the microphysics-off option's
    # own default, and it cannot know that every source on this route
    # enters woof.ingest.real.  The refusal therefore has to be the
    # registry's, and it has to fire at review -- resolving the value here
    # instead would be a resolution nothing downstream performs.
    "real-source-mp-off-requires-explicit-moist",
    "real-source-mp-off-requires-moist",
    "transition-required-setting",
    # A published cross edge whose endpoint has no ported closure: the
    # nest-edge resolver refuses it at tree load, and only a plan with edges
    # can see it.  Generated from the resolver's own ported set.
    "transition-unported-endpoint",
    # An unimplemented option is refused BEFORE its selectors are projected:
    # woof/physics_registry.py stops at ``unimplemented-option`` and
    # ``continue``s without the ``settings.update(selectors)`` an implemented
    # option gets, so the resolved settings never carry the choice and no
    # RunConfig built from them can see it.  The runtime authority is not
    # excused from the question, it is handed a different one --
    # ``test_an_unimplemented_option_is_refused_when_its_selectors_are_forced``
    # asks it directly, and finding that it answered WRONG for the 1/1
    # radiation pair is what put the ra_lw_physics=1 refusal into
    # woof/config.py.
    "unimplemented-option",
    "unknown-expert-acknowledgement",
    "unsupported-component-transition",
})

#: The physics-suite question both authorities own.  A registry refusal
#: carrying any of these must also be a ``validate_run_config`` refusal.
_SHARED_CODES = frozenset({
    "parameter-reader",
    "irrigation-pairing",
    # The conditional kind (constraints.refused_when), added at 1.9 because
    # the mp=9 / RTE+RRTMGP cloud-optics coupling is a CONJUNCTION -- a
    # radiation option AND an ra_rrtmg_variant -- and none of the other
    # three kinds can state one.  It is shared, not registry-only: the
    # coupling is a property of a single per-domain RunConfig, and
    # woof/core/rrtmgp.py refuses exactly the same conjunction at run
    # time.  This gate is what measured the disagreement while the rule
    # lived as unevaluated prose in the option's ``extensions``: 320 of
    # 118,800 component combinations, every one of them mp=9 with
    # RTE+RRTMGP.
    "component-conditional-refusal",
    "component-dependency",
    # A selected option lacking a row some consumer will read during the
    # run (checkpoint identity, vertical bound, ring guard, ...).  Shared:
    # validate_run_config asks the same question through the same function
    # (woof.physics_registry.consumer_row_gaps), so a registry that refuses
    # here must be refused by the runtime authority too.
    "consumer-row-missing",
    "component-forbidden-setting",
    "component-required-setting",
    # The multi-valued twin of required-setting, for the one setting a
    # scheme can define more than one value of: RUC's soil geometry.  It
    # is SHARED for the reason it exists -- the single-valued kind made
    # plan review narrower than the loader on exactly that setting, and
    # this battery is what holds the two doors equal.
    "component-admitted-setting",
    # The Noah mosaic coupling law (fa2e5efd8, c517106a8).  Shared by
    # construction: plan review calls woof.config's
    # validate_noah_mosaic_config, the function validate_run_config calls,
    # on the resolved settings.  Unclassified, every mosaic_urban_canopy
    # override was reported as a disagreement it is not.
    "noah-mosaic-pairing",
    # Consumer requirements are also checked by validate_run_config.
    "parameter-consumer",
    "parameter-value",
    # The SPP consumer law, shared the same way: plan review calls
    # woof.config's validate_spp_config, the function validate_run_config
    # calls, on the resolved settings.  Before it did, 340 reachable plans
    # with spp_conv=1 or spp_pbl=1 and no consuming scheme were offered as
    # launchable and refused at the run door.
    "spp-consumer",
    # The scalar_pblmix consumer pair and the gsd_41/spp_pbl exclusion,
    # shared by the same construction (woof.config's
    # validate_scalar_pblmix_consumer and validate_mynn_generation_spp).
    "pbl-coupling",
    # The lightning package's producer law, shared by the same construction:
    # plan review calls woof.lightning_contract.validate_lightning, the
    # function validate_run_config calls.  Before it did, 602 reachable plans
    # on the 2.9.0 round-2 tree offered lightning_option 1, 2 or 11 with no
    # reflectivity or cumulus cloud top as launchable.
    "lightning-dependency",
    "madwrf-dependency",
    "unimplemented-selector",
    "unknown-selector-combination",
})

#: Questions about the MACHINE, not about the configuration: is a dataset
#: or a table set INSTALLED here.  The registry answers them at plan
#: review and ``validate_run_config`` deliberately does not, so they are
#: neither shared nor a scope difference in the registry-only sense.
#:
#: THEY ARE REPORTED, NOT REFUSED.  ``validate_physics_plan`` puts them in
#: ``install_state`` and leaves ``launchable`` to the plan, because a plan
#: is portable and an install is not: two of classic Thompson's four
#: tables are excluded from the wheel, so appending them to ``errors``
#: made the DEFAULT template unlaunchable on every fresh install and
#: turned six shipped tests in ``tests/test_physics_registry.py`` red in
#: exactly the state the clean-venv release replay runs in.
#:
#: WHY THIS SET EXISTS AT ALL.  ``lateral-forcing-dataset`` was classified
#: registry-only, then shared, then registry-only again, and each flip was
#: an argument about topology that missed what actually separates the two
#: authorities here.  ``validate_run_config`` is also the namelist
#: importer's battery, and an import emits a TOML and reads no dataset, so
#: a battery that answered "is the 225 MB file installed" refused a format
#: conversion for a missing download and offered a way out
#: (``mp28_aerosol_source``) that only exists in the TOML it refused to
#: write.  Making the runtime authority answer it turned this file's own
#: shipped evidence gate from PASS to FAIL.
#:
#: THE RUNTIME AUTHORITY IS NOT EXCUSED, it is handed a different door.
#: Each row names the door that raises it, and
#: ``test_every_install_state_code_is_raised_by_a_run_door`` walks them:
#:
#: * ``lateral-forcing-dataset`` -> ``woof.config``'s
#:   ``validate_experiment_preparation``, called by ``woof go``'s stage
#:   composer BEFORE the fetch stage is spawned and by the experiment run
#:   dispatch, with the per-domain ``validate_run_preparation`` kept as a
#:   floor in ``woof.ingest.real.initialize_real``.
#: * ``asset-unresolved`` -> the scheme's own staging preflight and loader,
#:   which name the member that is not there when a forecast selects the
#:   scheme (``woof.prepared_domain_tree_forecast._verify_thompson_assets``
#:   -> ``woof.table_assets.require_thompson_tables`` ->
#:   ``woof.core.thompson_contract.validate_table_assets``;
#:   ``woof.core.rrtmgp._table``).
#: NOT A SECOND LIST.  The module under test owns the inventory
#: (:data:`woof.physics_registry.INSTALL_STATE_CODES`) because it is the
#: thing that decides which bucket an issue lands in; a copy here would
#: let the two drift and this file would still be green.
_INSTALL_STATE_CODES = INSTALL_STATE_CODES

_PERMISSIVE_RUNNER = "test.exhaustive-component-cross-product"

#: Small enough that the Noah-MP column rail is not what is being measured
#: here (``tests/test_noahmp_column_budget.py`` owns that), and legal for every
#: other invariant in the battery.
_COLUMNS_NX = 16
_COLUMNS_NY = 16
_GEOMETRY = {
    "nx": _COLUMNS_NX, "ny": _COLUMNS_NY, "nz": 20,
    "dx": 3000.0, "dy": 3000.0, "ztop": 20000.0,
    "dt": 6.0, "run_seconds": 60.0,
}

_RUN_CONFIG_FIELDS = frozenset(
    field.name for field in dataclasses.fields(RunConfig))


def test_the_cross_product_geometry_stays_inside_the_noahmp_rail() -> None:
    """Keep this file measuring coupling, not grid width."""
    assert _COLUMNS_NX * _COLUMNS_NY <= NOAHMP_MEASURED_COLUMN_CEILING


def _base_template_id(registry: dict) -> str:
    """The template every synthesised plan starts from.

    Any template works -- every component is overridden explicitly -- so the
    choice is the one whose parameters are the plainest: no RRTMGP
    compatibility token, no per-domain override columns.
    """

    return "wsm6-ysu-mm5-noah-no-radiation-v1"


def _permissive_registry() -> dict:
    """The shipped registry plus one route that permits every override.

    Nothing else is touched.  In particular no option, constraint, parameter
    spec or warning is modified, so the constraint evaluation exercised below
    is the shipped one.
    """

    registry = physics_registry()
    registry["runner_routes"][_PERMISSIVE_RUNNER] = {
        "allowed_component_overrides": sorted(registry["components"]),
        "allowed_expert_selector_keys": [],
        "allowed_expert_setting_keys": [],
        "allowed_parameter_keys": [],
        "implemented": True,
        "mode": "experiment-per-domain",
        "require_explicit_components": True,
        "source_ids": ["*"],
        "source_template_ids": {},
        "topology_ids": ["single-domain-v1"],
    }
    return registry


def _single_domain_plan(registry: dict, runner_id: str, source_id: str,
                        template_id: str, components: dict | None = None,
                        parameters: dict | None = None,
                        acknowledgements: list | None = None) -> dict:
    domain: dict = {"domain_id": "d01", "template_id": template_id}
    if components:
        domain["components"] = dict(components)
    if parameters:
        domain["parameters"] = dict(parameters)
    plan = {
        "schema": PLAN_SCHEMA,
        "plan_id": "authority-agreement-probe-v1",
        "registry_sha256": registry_sha256(registry),
        "context": {"source_id": source_id, "runner_id": runner_id,
                    "topology_id": "single-domain-v1"},
        "domains": [domain],
        "edges": [],
    }
    if acknowledgements:
        plan["acknowledgements"] = list(acknowledgements)
    return plan


def _tree_plan(registry: dict, runner_id: str, source_id: str,
               template_id: str, components: dict | None = None,
               parameters: dict | None = None,
               acknowledgements: list | None = None) -> dict:
    def domain(domain_id: str, root: bool) -> dict:
        entry: dict = {"domain_id": domain_id, "template_id": template_id}
        if components:
            entry["components"] = dict(components)
        merged = dict(parameters or {})
        if not root:
            merged["spec_exp"] = 0.0
        if merged:
            entry["parameters"] = merged
        return entry

    plan = {
        "schema": PLAN_SCHEMA,
        "plan_id": "authority-agreement-tree-probe-v1",
        "registry_sha256": registry_sha256(registry),
        "context": {"source_id": source_id, "runner_id": runner_id,
                    "topology_id": "one-way-nested-v1"},
        "domains": [domain("d01", True), domain("d02", False)],
        "edges": [{"parent_domain_id": "d01", "child_domain_id": "d02"}],
    }
    if acknowledgements:
        plan["acknowledgements"] = list(acknowledgements)
    return plan


def _run_config(settings: dict, *, nested: bool) -> RunConfig:
    """Build the RunConfig the resolved settings describe.

    Only registry settings that name a real RunConfig field are carried;
    experiment-scope knobs (``p_top``, ``blend_width``, ``co2_vmr``) belong to
    the experiment schema and unimplemented knobs never reach ``settings`` at
    all (``tests/test_physics_registry_declarations.py`` owns both).
    """

    keywords = {name: value for name, value in settings.items()
                if name in _RUN_CONFIG_FIELDS}
    keywords.update(_GEOMETRY)
    keywords["nested"] = nested
    return RunConfig(**keywords)


def _config_refusal(settings: dict, *, nested: bool) -> str | None:
    try:
        validate_run_config(_run_config(settings, nested=nested))
    except (ValueError, NotImplementedError, TypeError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _declared_environment(report: dict) -> dict[str, str]:
    """Environment the plan's OWN asset requirements say a launch needs.

    The registry may publish an external table set as an asset requirement
    with ``enable_environment`` and ``root_environment`` rather than as an
    error, the same way it publishes bundled package data: a prerequisite a
    launcher checks, not a malformed choice.  So the accurate comparison is
    "given the prerequisites the plan itself declares, do the two authorities
    agree".  Since the mp8 promotion (packaged Thompson tables, product/v1
    packaging lane 2026-07-28) no shipped requirement declares either key --
    Thompson's entry is a packaged-table-set with only an optional
    root-override env -- so this helper currently returns {} for every plan;
    the mechanism stays because it is the declared contract for any future
    external table set.
    """

    environment: dict[str, str] = {}
    for entry in report.get("asset_requirements", []):
        requirement = entry.get("requirement", {})
        enable = requirement.get("enable_environment", {})
        if isinstance(enable, dict):
            for name, value in enable.items():
                environment[str(name)] = str(value)
        root = requirement.get("root_environment")
        if isinstance(root, str) and root:
            # Any non-empty path satisfies the readiness authority; the BYTES
            # are validated at launch, which is not this gate's question.
            environment[root] = str(Path(__file__).resolve().parent)
    return environment


def _domain_index_of(path: str) -> int | None:
    """Which domain a registry issue is ABOUT, from its path, or None.

    Every per-domain issue this module raises is anchored at
    ``domains[<index>]`` (woof/physics_registry.py builds that prefix
    once, as ``base_path``); anything else -- a plan-level or context-level
    issue -- is about the whole plan.
    """

    import re

    matched = re.match(r"domains\[(\d+)\]", str(path))
    return int(matched.group(1)) if matched else None


def _shared_codes_for_domain(report: dict, index: int) -> list[str]:
    """Shared-code refusals THIS domain must also earn from the runtime.

    This used to be the plan's whole error list, compared against every
    domain in turn.  That was survivable only while every shared code
    happened to be uniform across a plan's domains, and it stopped being
    survivable the moment a code could refuse one domain and not another:
    a two-domain tree whose CHILD is refused would demand that the ROOT's
    RunConfig be refused too, which is a disagreement the two authorities
    do not actually have.  An issue names its domain; this reads it.
    """

    return sorted({
        error["code"] for error in report["errors"]
        if error["code"] in _SHARED_CODES
        and _domain_index_of(error["path"]) in (None, index)
    })


def _compare(label: str, report: dict, disagreements: list[str],
             registry: dict | None = None,
             monkeypatch=None) -> None:
    """Record any way in which the two authorities differ about ``report``."""

    registry = physics_registry() if registry is None else registry
    codes = {error["code"] for error in report["errors"]}
    unclassified = sorted(codes - _REGISTRY_ONLY_CODES - _SHARED_CODES)
    if unclassified:
        disagreements.append(
            f"{label}: registry raised unclassified error code(s) "
            f"{unclassified}; classify them in _REGISTRY_ONLY_CODES or "
            "_SHARED_CODES so this gate keeps its meaning")
        return
    misfiled = sorted(codes & _INSTALL_STATE_CODES)
    if misfiled:
        disagreements.append(
            f"{label}: install-state code(s) {misfiled} reached `errors`, "
            "where they decide `launchable`; a machine question belongs in "
            "`install_state`")
        return
    reported = {row["code"] for row in report["install_state"]}
    stray = sorted(reported - _INSTALL_STATE_CODES)
    if stray:
        disagreements.append(
            f"{label}: `install_state` carries {stray}, which "
            "INSTALL_STATE_CODES does not name")
        return

    environment = _declared_environment(report)
    for index, resolved in enumerate(report["resolved_domains"]):
        nested = index > 0
        shared = _shared_codes_for_domain(report, index)
        if environment and monkeypatch is not None:
            for name, value in environment.items():
                monkeypatch.setenv(name, value)
            try:
                refusal = _config_refusal(resolved["settings"], nested=nested)
            finally:
                for name in environment:
                    monkeypatch.delenv(name, raising=False)
        else:
            refusal = _config_refusal(resolved["settings"], nested=nested)
        if report["launchable"]:
            if refusal is not None:
                disagreements.append(
                    f"{label} [{resolved['domain_id']}]: registry says "
                    f"LAUNCHABLE, validate_run_config REFUSES -- {refusal}")
        elif shared:
            if refusal is None:
                disagreements.append(
                    f"{label} [{resolved['domain_id']}]: registry REFUSES on "
                    f"{shared}, validate_run_config says OK")


_CORE_WALK = None


def _core_walk_slice(bounds):
    return _CORE_WALK(*bounds)


def _walk_across_cores(walk, total: int) -> tuple[int, list[str]]:
    """Run ``walk(start, stop)`` over ``[0, total)`` on every core, in order."""
    global _CORE_WALK
    import multiprocessing
    import os

    workers = min(os.cpu_count() or 1, 32)
    if os.environ.get("PYTEST_XDIST_WORKER"):
        # Under pytest -n N this worker is already one of N: a 24-process fork pool here
        # multiplied one -n 2 run into 26 processes on a shared 24-core box (slim campaign, 2026-10-20).
        workers = min(workers, 2)
    if total < 2 * workers or "fork" not in multiprocessing.get_all_start_methods():
        return walk(0, total)
    slices = max(1, workers * 4)
    size = -(-total // slices)
    bounds = [(start, min(start + size, total))
              for start in range(0, total, size)]
    _CORE_WALK = walk
    try:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_core_walk_slice, bounds, chunksize=1)
    finally:
        _CORE_WALK = None
    found: list[str] = []
    walked = 0
    for count, part in parts:
        walked += count
        found.extend(part)
    return walked, found


def _component_option_ids(registry: dict) -> dict[str, list[str]]:
    return {
        component_id: sorted(component["options"])
        for component_id, component in sorted(registry["components"].items())
    }


def test_exhaustive_component_cross_product_agrees_on_every_combination(
        monkeypatch):
    """Every combination of every registered option, both authorities."""

    registry = _permissive_registry()
    template_id = _base_template_id(registry)
    options = _component_option_ids(registry)
    component_ids = sorted(options)

    # The fixture is fixed throughout the exhaustive walk. Hash its exact
    # bytes once, retaining every option comparison and checking immutability
    # afterward instead of serializing the same document for every tuple.
    from woof import physics_registry as registry_module
    original_hash = registry_module.registry_sha256
    fixed_hash = original_hash(registry)

    def fixture_hash(value=None):
        return fixed_hash if value is registry else original_hash(value)

    # The validator normally copies caller-owned registries. This fixture
    # is immutable for the whole walk, and the hash assertion below checks
    # that claim independently. Reuse ONLY this object so its cached hash
    # also reaches the validator; all other deepcopy calls stay unchanged.
    original_copy = registry_module.deepcopy

    def fixture_copy(value, memo=None):
        return registry if value is registry else original_copy(value, memo)

    monkeypatch.setattr(registry_module, "deepcopy", fixture_copy)
    monkeypatch.setattr(registry_module, "registry_sha256", fixture_hash)
    monkeypatch.setitem(globals(), "registry_sha256", fixture_hash)

    option_lists = [options[c] for c in component_ids]
    expected = 1
    for option_list in option_lists:
        expected *= len(option_list)

    def walk(start: int, stop: int) -> tuple[int, list[str]]:
        found: list[str] = []
        walked = 0
        for values in itertools.islice(
                itertools.product(*option_lists), start, stop):
            components = dict(zip(component_ids, values))
            walked += 1
            plan = _single_domain_plan(
                registry, _PERMISSIVE_RUNNER, "any-source", template_id,
                components=components)
            report = validate_physics_plan(plan, registry=registry)
            _compare(repr(components), report, found, registry=registry,
                     monkeypatch=monkeypatch)
        return walked, found

    # The product is tens of thousands of plans, each a few milliseconds
    # of pure Python: on one core it outran the suite's timeout (a 1,200 s
    # serial run did not finish), leaving one xdist worker busy while the
    # rest idled.  Forked workers share the fixture and the monkeypatches
    # above; the slices are walked in product order and joined in order,
    # so the record is the one serial walk would write.
    combinations, disagreements = _walk_across_cores(walk, expected)
    assert combinations == expected, (
        f"the cross-product enumerated {combinations} of {expected} "
        "combinations")
    assert combinations >= 4032, (
        "the registry should carry at least the component options this gate "
        f"was written against; enumerated {combinations}")
    assert original_hash(registry) == fixed_hash, "the shared fixture changed during the walk"
    assert disagreements == [], (
        f"{len(disagreements)} of {combinations} component combinations are "
        "decided differently by the registry and by validate_run_config:\n  "
        + "\n  ".join(disagreements[:40]))


@pytest.mark.parametrize("spelling", ["split", "aggregate"])
@pytest.mark.parametrize("microphysics,variant,refuses", [
    # mp=9 against RTE+RRTMGP used to be the one True row here: the
    # registry's refused_when and validate_run_config both refused it for a
    # missing cloud-optics row.  The row landed (woof.core.rrtmgp
    # ``9: "milbrandt2"``), the registry constraint regenerated empty and
    # the config-door raise retired with it, so BOTH authorities now admit
    # the pairing on both selector spellings -- and this test keeps
    # measuring that they agree, in the accepting direction.
    ("milbrandt2mom-mp9", "rte-rrtmgp", False),
    ("milbrandt2mom-mp9", "rrtmg_legacy", False),
    ("wsm6-mp6", "rte-rrtmgp", False),
])
def test_mp9_cloud_optics_gate_covers_both_selector_spellings(
        spelling, microphysics, variant, refuses):
    registry = _permissive_registry()
    registry["runner_routes"][_PERMISSIVE_RUNNER]["allowed_parameter_keys"] = ["ra_rrtmg_variant"]
    # Find WSM6 through its selector so this control uses the registry's name.
    if microphysics == "wsm6-mp6":
        microphysics = next(key for key, row in registry["components"]["microphysics"]["options"].items()
                            if row.get("selectors") == {"mp_physics": 6})
    report = validate_physics_plan(_single_domain_plan(
        registry, _PERMISSIVE_RUNNER, "any-source", _base_template_id(registry),
        components={"microphysics": microphysics, "radiation": "rte-rrtmgp"},
        parameters={"ra_rrtmg_variant": variant}), registry=registry)
    assert report["launchable"] is not refuses, report["errors"]
    conditional = [row for row in report["errors"] if row["code"] == "component-conditional-refusal"]
    assert bool(conditional) is refuses
    settings = dict(report["resolved_domains"][0]["settings"])
    if spelling == "aggregate":
        # The spelling a coupled pair reaches the run door in when it
        # came from a WRF namelist.  It used to be a SECOND registry
        # option, keyed on the (-1, -1) sentinel, and the parametrize
        # above named the two option ids; it is one spelling of the one
        # 4/4 option now, so the rewrite happens here and the gate is
        # still measured on both.
        pair = (settings["ra_lw_physics"], settings["ra_sw_physics"])
        assert pair[0] == pair[1], pair
        settings.update(ra_physics=pair[0], ra_lw_physics=-1,
                        ra_sw_physics=-1)
    runtime = _config_refusal(settings, nested=False)
    assert (runtime is not None) is refuses
    if refuses:
        assert "cloud-optics" in runtime


def _mp28_option_id(registry: dict) -> str:
    """Aerosol-aware Thompson, found by its selector rather than its name."""

    return next(
        key for key, row in registry["components"]["microphysics"][
            "options"].items()
        if row.get("selectors") == {"mp_physics": 28})


def _without_the_wif_dataset(monkeypatch, tmp_path):
    """The state a machine that has not staged the 225 MB dataset is in.

    Every rung of the resolver is pointed somewhere empty, including the
    working directory (WRF's own ``constants_name`` rule is the last
    rung), so this measures a decision of the test rather than of the host
    -- the reference node has the dataset staged.
    """

    from woof.ingest import wif_climatology, wif_dataset

    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_PATH_ENV,
                       raising=False)
    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_ROOT_ENV,
                       raising=False)
    empty = tmp_path / "no-staged-wif"
    empty.mkdir(exist_ok=True)
    monkeypatch.setenv(wif_dataset.WIF_DATASET_ROOT_ENV, str(empty))
    monkeypatch.chdir(empty)
    return empty


def _run_door_refusal(settings: dict, *, nested: bool) -> str | None:
    """What the RUN door says about these settings, or None.

    The install-state authority on the runtime side.  It is not
    ``validate_run_config``: that battery decides the CONFIGURATION and is
    shared with the doors that only translate one.
    """

    from woof.config import validate_run_preparation

    try:
        validate_run_preparation(_run_config(settings, nested=nested))
    except (ValueError, NotImplementedError, TypeError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


@pytest.mark.parametrize("laterally_forced", [True, False])
def test_a_specified_root_is_not_a_hole_in_the_lateral_forcing_gate(
        laterally_forced, tmp_path, monkeypatch):
    """The half of "laterally forced" that lives in a per-domain setting.

    THE SPLIT THIS CLOSES.  The registry decided lateral forcing from the
    plan's EDGES alone -- "does this domain have a parent" -- and called a
    real-data ROOT launchable.  A real-data root sets ``specified``
    (woof/domain_wizard.py, woof/hrrr_hierarchy_direct.py,
    woof/downscale.py all do), so on the single most common plan there is
    -- one specified root running aerosol-aware Thompson -- plan review
    said LAUNCHABLE about a run that cannot be prepared.  Both authorities
    read ``specified`` now, and the RUNTIME side of the comparison is the
    run door, because that is where an install-state question is answered
    at run time.
    """

    registry = _permissive_registry()
    registry["runner_routes"][_PERMISSIVE_RUNNER][
        "allowed_parameter_keys"] = ["specified"]
    _without_the_wif_dataset(monkeypatch, tmp_path)
    report = validate_physics_plan(_single_domain_plan(
        registry, _PERMISSIVE_RUNNER, "any-source",
        _base_template_id(registry),
        components={"microphysics": _mp28_option_id(registry)},
        parameters={"specified": laterally_forced}), registry=registry)
    settings = report["resolved_domains"][0]["settings"]
    assert settings["specified"] is laterally_forced
    codes = [row["code"] for row in report["install_state"]]
    refusal = _run_door_refusal(settings, nested=False)

    assert ("lateral-forcing-dataset" in codes) is laterally_forced, codes
    assert (refusal is not None) is laterally_forced, refusal
    # Reported, never refused: an absent download does not make the plan
    # wrong, and the same document is launchable on the node that has it.
    assert report["launchable"] is True, report["errors"]
    # The CONFIGURATION battery says nothing either way: a TOML naming a
    # dataset this machine does not have is still a valid configuration,
    # and the doors that only translate one must not be refused by it.
    assert _config_refusal(settings, nested=False) is None
    if laterally_forced:
        # Same tuple, same sentence, same two ways out, both authorities.
        registry_said = " ".join(
            row["message"] for row in report["install_state"]
            if row["code"] == "lateral-forcing-dataset")
        for clause in ("QNWFA_QNIFA_SIGMA_MONTHLY.dat",
                       "mp28_aerosol_source='synthetic'"):
            assert clause in registry_said, registry_said
            assert clause in refusal, refusal


def test_a_nested_domain_is_not_refused_for_a_defect_it_cannot_have(
        tmp_path, monkeypatch):
    """The other half of the condition, deleted rather than kept.

    ``nested`` used to refuse beside ``specified``.  It named the same
    breakage -- zero-inflow aerosol boundaries and the walk to WRF's
    aerosol floor -- on a path that does not have it: nwfa and nifa are
    members of ``woof.ingest.lateral_bc.COUPLED_SCALAR_STATE_FIELDS``, so
    a nest edge carries them from the parent whatever the parent's aerosol
    source was, and the depletion is carried by the specified-BC inventory
    (``_external_scalar_boundary_fields``).  An idealized tree -- periodic
    root, no ``specified`` anywhere -- was refused for a defect it cannot
    have, which is the half of the gate law about what a refusal PREVENTS.
    """

    from woof.ingest.lateral_bc import COUPLED_SCALAR_STATE_FIELDS

    assert {"nwfa", "nifa"} <= COUPLED_SCALAR_STATE_FIELDS

    registry = _permissive_registry()
    _without_the_wif_dataset(monkeypatch, tmp_path)
    settings = validate_physics_plan(_single_domain_plan(
        registry, _PERMISSIVE_RUNNER, "any-source",
        _base_template_id(registry),
        components={"microphysics": _mp28_option_id(registry)}),
        registry=registry)["resolved_domains"][0]["settings"]
    assert _run_door_refusal(settings, nested=True) is None
    assert _run_door_refusal(dict(settings, specified=True),
                             nested=False) is not None


def _wheel_shaped_companion(tmp_path, monkeypatch, requirement):
    """Point the packaged rung at a companion laid out the way a WHEEL is.

    The two externalized classic-Thompson tables are excluded from the
    recast-woof-data wheel by size, but a source checkout tracks them and an
    editable install of the companion carries them, so on a developer
    machine the packaged rung resolves the whole set and the fresh-install
    state these tests reproduce never appears.  The mirror built here
    holds exactly the members a wheel carries, at their declared sizes
    (sparse files; the resolver checks existence and size, never bytes),
    so the state is the test's decision on any machine.
    """

    from woof import data_assets
    from woof.table_assets import EXTERNALIZED_TABLE_FILENAMES

    mirror = tmp_path / "wheel-companion"
    tables = mirror.joinpath(*str(
        requirement["resolution"]["data_relative"]).split("/"))
    tables.mkdir(parents=True)
    for asset in requirement["assets"]:
        if asset["filename"] in EXTERNALIZED_TABLE_FILENAMES:
            continue
        with open(tables / asset["filename"], "wb") as handle:
            handle.truncate(int(asset["bytes"]))
    monkeypatch.setattr(data_assets, "companion_root", lambda: mirror)
    return mirror


def test_the_asset_code_is_emitted_classified_and_names_what_is_missing(
        tmp_path, monkeypatch):
    """``asset-unresolved`` on a machine that is short of a table set.

    THE STATE THIS REPRODUCES is a default install: two of classic
    Thompson's four tables (``freezeH2O.dat``, ``qr_acr_qg_V4.dat``) are
    excluded from the wheel and arrive only through ``woof
    fetch-tables``, so the requirement cannot resolve until an operator
    stages them.  The staged root is redirected and the root override is
    pointed at an empty directory, which makes this a decision of the
    test on any machine rather than of whatever the host has staged.

    Three things are measured, because the code was shipped with only the
    resolving direction exercised: it is EMITTED, it is CLASSIFIED (an
    unclassified code fails every other test in this file, on a state a
    fresh install is in), and the files it names are the files that are
    actually missing.  That last one is not bookkeeping: the resolver
    reported the LAST root walked rather than the closest, so it named
    two files that were installed in the packaged root all along and sent
    an operator after 50 MB they already had.
    """

    from woof.table_assets import EXTERNALIZED_TABLE_FILENAMES

    registry = _permissive_registry()
    mp8 = next(key for key, row in registry["components"]["microphysics"][
        "options"].items() if row.get("selectors") == {"mp_physics": 8})
    monkeypatch.setenv("HOME", str(tmp_path / "no-staged-tables"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "no-staged-tables"))
    empty = tmp_path / "no-table-root"
    empty.mkdir()
    monkeypatch.setenv("WOOF_THOMPSON_TABLE_ROOT", str(empty))
    _wheel_shaped_companion(
        tmp_path, monkeypatch,
        registry["components"]["microphysics"]["options"][mp8][
            "asset_requirements"][0])

    report = validate_physics_plan(_single_domain_plan(
        registry, _PERMISSIVE_RUNNER, "any-source",
        _base_template_id(registry), components={"microphysics": mp8}),
        registry=registry)
    codes = {row["code"] for row in report["install_state"]}
    assert "asset-unresolved" in codes, report["install_state"]
    assert "asset-unresolved" in _INSTALL_STATE_CODES
    assert not (codes - _INSTALL_STATE_CODES), codes
    # AND THE PLAN IS STILL LAUNCHABLE.  This is the state a fresh wheel
    # is in for the DEFAULT template, so a verdict that reads the machine
    # here makes `woof source --validate-physics-plan` exit nonzero on a
    # correct plan and turns the shipped suites red on the clean-venv
    # replay.  The run door is what refuses, by name, below.
    assert report["launchable"] is True, report["errors"]
    assert not report["errors"], report["errors"]

    entry = next(row for row in report["asset_requirements"]
                 if "thompson-mp8" in str(row["requirement"]["id"]))
    assert entry["resolution"]["resolved"] is False
    # Exactly the members a wheel does not carry, and nothing that is
    # sitting in the packaged root the ladder already walked.
    assert set(entry["resolution"]["missing"]) == set(
        EXTERNALIZED_TABLE_FILENAMES), entry["resolution"]

    # And the configuration battery is silent about it, which is what
    # makes this an install-state code rather than a shared one.
    assert _config_refusal(report["resolved_domains"][0]["settings"],
                           nested=False) is None


def test_every_install_state_row_names_the_domain_it_is_about(
        tmp_path, monkeypatch):
    """An install-state row is anchored at ``domains[<index>]``, per domain.

    Every per-domain issue this module raises is anchored at the
    ``domains[<index>]`` prefix built once as ``base_path``, and a reader
    -- this file's own ``_domain_index_of``, and any panel that shows a
    row beside the domain it belongs to -- takes the domain from that
    anchor.  The two asset rows were written as
    ``plan.domains[<grid id>,<grid id>].asset_requirements.<id>``: a shape
    that matches no anchor, carries grid ids where every other row carries
    an index, and folds every domain needing the set into one row.  So the
    one code that says a table set is missing on THIS machine was the one
    code whose domain could not be read.

    A two-domain tree is used because the single-domain case cannot tell a
    per-domain row from a folded one.
    """

    registry = _permissive_registry()
    mp8 = next(key for key, row in registry["components"]["microphysics"][
        "options"].items() if row.get("selectors") == {"mp_physics": 8})
    monkeypatch.setenv("HOME", str(tmp_path / "no-staged-tables"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "no-staged-tables"))
    empty = tmp_path / "no-table-root"
    empty.mkdir()
    monkeypatch.setenv("WOOF_THOMPSON_TABLE_ROOT", str(empty))
    _wheel_shaped_companion(
        tmp_path, monkeypatch,
        registry["components"]["microphysics"]["options"][mp8][
            "asset_requirements"][0])

    report = validate_physics_plan(_tree_plan(
        registry, _PERMISSIVE_RUNNER, "any-source",
        _base_template_id(registry), components={"microphysics": mp8}),
        registry=registry)

    rows = report["install_state"]
    assert rows, report
    unreadable = [row["path"] for row in rows
                  if _domain_index_of(row["path"]) is None]
    assert not unreadable, unreadable

    asset_rows = [row for row in rows if row["code"] == "asset-unresolved"]
    assert sorted(_domain_index_of(row["path"]) for row in asset_rows) == [
        0, 1], [row["path"] for row in asset_rows]
    requirement = registry["components"]["microphysics"]["options"][
        mp8]["asset_requirements"][0]["id"]
    for row in asset_rows:
        assert row["path"].endswith(
            f".asset_requirements.{requirement}"), row["path"]
    # The aggregate block still answers "which domains", by grid id.
    entry = next(row for row in report["asset_requirements"]
                 if "thompson-mp8" in str(row["requirement"]["id"]))
    assert entry["domain_ids"] == ["d01", "d02"], entry


def test_every_install_state_code_is_raised_by_a_run_door(
        tmp_path, monkeypatch):
    """A code the runtime battery does not own still has to be answered.

    ``_INSTALL_STATE_CODES`` excuses ``validate_run_config``; it does not
    excuse the run.  Each row is walked here against the door named in
    that set's comment, so "the registry alone asks this" can never become
    "nothing else asks this".
    """

    assert _INSTALL_STATE_CODES == {"asset-unresolved",
                                    "lateral-forcing-dataset"}

    # lateral-forcing-dataset: the run door raises the registry's sentence.
    registry = _permissive_registry()
    registry["runner_routes"][_PERMISSIVE_RUNNER][
        "allowed_parameter_keys"] = ["specified"]
    _without_the_wif_dataset(monkeypatch, tmp_path)
    report = validate_physics_plan(_single_domain_plan(
        registry, _PERMISSIVE_RUNNER, "any-source",
        _base_template_id(registry),
        components={"microphysics": _mp28_option_id(registry)},
        parameters={"specified": True}), registry=registry)
    assert "lateral-forcing-dataset" in {
        row["code"] for row in report["install_state"]}
    assert _run_door_refusal(report["resolved_domains"][0]["settings"],
                             nested=False) is not None

    # asset-unresolved: THE PATH, not the leaf.  Calling
    # thompson_contract.validate_table_assets here would have proved the
    # function refuses an empty directory and stayed green if every caller
    # were deleted -- which is the claim this test exists to make.  So the
    # forecast stage's own entry point is called, with the table root
    # pointed at an empty directory, and the refusal that comes back is
    # the one a run meets.
    from woof import physics_compat
    from woof.core import thompson_contract
    from woof import prepared_domain_tree_forecast

    empty = tmp_path / "no-tables"
    empty.mkdir()
    monkeypatch.setenv(physics_compat.THOMPSON_TABLE_ROOT_ENV, str(empty))
    mp8_tree = SimpleNamespace(domains=(
        SimpleNamespace(grid_id=1, run=SimpleNamespace(
            mp_physics=8, thompson_version="wrf_461",
            thompson_table_dir=None)),))
    with pytest.raises(FileNotFoundError) as missing:
        prepared_domain_tree_forecast._verify_thompson_assets(mp8_tree)
    said = str(missing.value)
    # It names the files and the command that stages them, not a path
    # inside site-packages.
    assert "woof fetch-tables" in said, said
    assert any(asset.filename in said
               for asset in thompson_contract.CLASSIC_TABLE_ASSETS), said
    # And the leaf the path ends at is the one the ledger above names.
    with pytest.raises(FileNotFoundError) as leaf:
        thompson_contract.validate_table_assets(empty)
    assert "Thompson table asset" in str(leaf.value)


def test_the_named_way_out_is_taken_the_same_way_by_both_authorities(
        tmp_path, monkeypatch):
    """A refusal stands only while its way out works -- on both sides."""

    registry = _permissive_registry()
    registry["runner_routes"][_PERMISSIVE_RUNNER]["allowed_parameter_keys"] = [
        "specified", "mp28_aerosol_source"]
    _without_the_wif_dataset(monkeypatch, tmp_path)
    report = validate_physics_plan(_single_domain_plan(
        registry, _PERMISSIVE_RUNNER, "any-source",
        _base_template_id(registry),
        components={"microphysics": _mp28_option_id(registry)},
        parameters={"specified": True, "mp28_aerosol_source": "synthetic"}),
        registry=registry)
    codes = [row["code"] for row in report["install_state"]]
    assert "lateral-forcing-dataset" not in codes, codes
    assert _run_door_refusal(report["resolved_domains"][0]["settings"],
                             nested=False) is None

    # And the OTHER way out -- staging the dataset -- is answered by the
    # same resolver on both sides, including the rung a generic asset
    # ladder does not have: the working directory, which is WRF's own
    # constants_name rule.  A structurally valid file is the fixture
    # because presence alone is not what either authority asks any more.
    from woof.ingest import wif_climatology
    from wif_intermediate_stub import write_minimal_wif_intermediate

    staged = tmp_path / "cwd-rung"
    staged.mkdir()
    write_minimal_wif_intermediate(
        staged / wif_climatology.WIF_CLIMATOLOGY_FILE)
    monkeypatch.chdir(staged)
    report = validate_physics_plan(_single_domain_plan(
        registry, _PERMISSIVE_RUNNER, "any-source",
        _base_template_id(registry),
        components={"microphysics": _mp28_option_id(registry)},
        parameters={"specified": True}), registry=registry)
    codes = [row["code"] for row in report["install_state"]]
    assert "lateral-forcing-dataset" not in codes, codes
    assert _run_door_refusal(report["resolved_domains"][0]["settings"],
                             nested=False) is None


def _selectable_parameter_values(registry: dict, name: str) -> list:
    """Values of a route-allowed per-domain parameter the registry declares.

    Enumerable specs are enumerated exhaustively; a continuous one contributes
    only its declared default, because a numeric range is not a selection and
    ``_parameter_error`` already gates its bounds.
    """

    spec = registry["parameters"].get(name)
    if not isinstance(spec, dict):
        return []
    if spec.get("type") == "boolean":
        return [False, True]
    enum = spec.get("enum")
    if isinstance(enum, list):
        return list(enum)
    return [spec["default"]] if "default" in spec else []


def _reachable_plans(registry: dict):
    """Every plan a user can build from the shipped templates and routes."""

    routes = registry["runner_routes"]
    for runner_id, route in sorted(routes.items()):
        if route.get("implemented") is not True:
            continue
        acknowledgement = route.get("expert_acknowledgement_id")
        expert = route.get("expert_template_ids", {}) or {}
        normal = route.get("source_template_ids", {}) or {}
        builder = (_tree_plan if "one-way-nested-v1" in route["topology_ids"]
                   else _single_domain_plan)
        overridable = (
            sorted(route.get("allowed_component_overrides", []))
            if route.get("mode") == "experiment-per-domain" else [])
        parameter_axes = {
            name: _selectable_parameter_values(registry, name)
            for name in sorted(route.get("allowed_parameter_keys", []))
        }
        parameter_axes = {name: values
                          for name, values in parameter_axes.items() if values}

        for source_id in sorted(route.get("source_ids", [])):
            template_ids = list(normal.get(source_id, []))
            expert_ids = expert_template_ids_for_source(route, source_id)
            for template_id in template_ids + expert_ids:
                acknowledgements = (
                    [acknowledgement]
                    if template_id in expert_ids and acknowledgement else None)
                # The template alone, then one axis at a time: a full Cartesian
                # product over overrides AND parameters would multiply into
                # tens of thousands of plans without exercising a coupling the
                # exhaustive layer above has not already decided.
                yield (runner_id, source_id, template_id,
                       builder(registry, runner_id, source_id, template_id,
                               acknowledgements=acknowledgements))
                for component_id in overridable:
                    for option_id in sorted(
                            registry["components"][component_id]["options"]):
                        yield (runner_id, source_id, template_id,
                               builder(registry, runner_id, source_id,
                                       template_id,
                                       components={component_id: option_id},
                                       acknowledgements=acknowledgements))
                for name, values in parameter_axes.items():
                    for value in values:
                        yield (runner_id, source_id, template_id,
                               builder(registry, runner_id, source_id,
                                       template_id,
                                       parameters={name: value},
                                       acknowledgements=acknowledgements))


def test_every_reachable_plan_the_registry_calls_launchable_actually_validates(
        monkeypatch):
    """The direction a user feels: offered means startable."""

    registry = physics_registry()
    disagreements: list[str] = []
    plans = 0
    launchable = 0
    for runner_id, source_id, template_id, plan in _reachable_plans(registry):
        plans += 1
        report = validate_physics_plan(plan)
        if report["launchable"]:
            launchable += 1
        _compare(f"{runner_id} {source_id} {template_id} "
                 f"{plan['domains'][0].get('components', {})}"
                 f"{plan['domains'][0].get('parameters', {})}",
                 report, disagreements, registry=registry,
                 monkeypatch=monkeypatch)

    assert plans > 100, f"the reachable surface enumerated only {plans} plans"
    assert launchable > 0, "no reachable plan is launchable at all"
    assert disagreements == [], (
        f"{len(disagreements)} of {plans} reachable plans are decided "
        "differently by the registry and by validate_run_config:\n  "
        + "\n  ".join(disagreements[:40]))


def test_the_gate_fails_when_an_authority_is_perturbed():
    """A gate nobody has seen fail is not evidence.

    Reintroduce the exact constraint that was wrong -- Noah requiring an MM5
    surface layer -- and the comparison must report the MYNN 5/5 pair.  The
    perturbation is applied to a deep copy, so the shipped registry is not
    touched.
    """

    registry = _permissive_registry()
    noah = registry["components"]["land_surface"]["options"]["noah"]
    noah["constraints"]["requires_components"] = {
        "surface_layer": ["revised-mm5", "classic-mm5"]}

    components = {
        "cumulus": "off", "land_surface": "noah", "microphysics": "wsm6-mp6",
        "pbl": "mynn", "radiation": "dudhia-shortwave",
        "surface_layer": "mynn",
    }
    plan = _single_domain_plan(
        registry, _PERMISSIVE_RUNNER, "any-source",
        _base_template_id(registry), components=components)
    report = validate_physics_plan(plan, registry=registry)
    assert report["launchable"] is False
    assert "component-dependency" in {
        error["code"] for error in report["errors"]}

    disagreements: list[str] = []
    _compare("perturbed-noah", report, disagreements)
    assert len(disagreements) == 1, disagreements
    assert "registry REFUSES" in disagreements[0]
    assert "component-dependency" in disagreements[0]

    # ...and the shipped registry still agrees about that same pair.
    shipped = _permissive_registry()
    unperturbed = validate_physics_plan(
        _single_domain_plan(shipped, _PERMISSIVE_RUNNER, "any-source",
                            _base_template_id(shipped),
                            components=components),
        registry=shipped)
    assert unperturbed["launchable"] is True, unperturbed["errors"]
    agreed: list[str] = []
    _compare("shipped-noah", unperturbed, agreed)
    assert agreed == []


def test_the_gate_fails_when_the_runtime_authority_is_the_wrong_one():
    """The other direction: launchable but unstartable is also a failure."""

    registry = _permissive_registry()
    components = {
        "cumulus": "off", "land_surface": "noah", "microphysics": "wsm6-mp6",
        "pbl": "ysu", "radiation": "dudhia-shortwave",
        "surface_layer": "classic-mm5",
    }
    report = validate_physics_plan(
        _single_domain_plan(registry, _PERMISSIVE_RUNNER, "any-source",
                            _base_template_id(registry),
                            components=components),
        registry=registry)
    assert report["launchable"] is True, report["errors"]

    settings = dict(report["resolved_domains"][0]["settings"])
    # The template's km_opt=4 under diff_opt=2 takes a nonzero khdif as
    # written (WRF ignores it there; WRF's own fire namelists set it).  Under
    # diff_opt=1 the same khdif is exactly what validate_run_config refuses
    # (WRF's diff_opt=1 vertical diffusion would read the constants), and the
    # registry spec for khdif is a bare non-negative number, so it cannot see
    # it.
    assert settings["km_opt"] == 4
    settings["diff_opt"] = 1
    settings["khdif"] = 100.0
    settings["diff_opt"] = 1  # WRF ignores these constants under diff_opt=2.
    assert _config_refusal(settings, nested=False) is not None
    forged = deepcopy(report)
    forged["resolved_domains"][0]["settings"] = settings
    disagreements: list[str] = []
    _compare("forged-khdif", forged, disagreements)
    assert len(disagreements) == 1, disagreements
    assert "registry says LAUNCHABLE" in disagreements[0]


def test_an_unimplemented_option_is_refused_when_its_selectors_are_forced():
    """The question the cross-product cannot ask, asked directly.

    ``validate_physics_plan`` refuses an unimplemented option without ever
    projecting its selectors into the resolved settings, so the cross-product
    above can only see the registry's half of that refusal.  A namelist import
    or a hand-written TOML has no such courtesy: it puts the selector value
    straight into a RunConfig.  So force each unimplemented option's selectors
    onto an otherwise valid configuration and require the runtime authority to
    refuse them too.

    This found a real hole.  ``ra_lw_physics=1`` -- WRF's RRTM longwave, the
    ``wrf-rrtm-dudhia`` pair -- was inside ``validate_run_config``'s accepted
    set ``(0, 1, 4, 90)`` and no readiness blocker covered it, so a config
    naming it validated cleanly and then raised NotImplementedError from
    ``woof/core/physics.py`` initialize_physics at driver construction.
    """

    registry = physics_registry()
    report = validate_physics_plan(
        _single_domain_plan(registry, "tools.prepared_single_domain_forecast",
                            "gfs", _base_template_id(registry)))
    assert report["launchable"] is True, report["errors"]
    baseline = dict(report["resolved_domains"][0]["settings"])
    assert _config_refusal(baseline, nested=False) is None

    exercised = []
    for component_id, component in sorted(registry["components"].items()):
        for option_id, option in sorted(component["options"].items()):
            if option.get("implemented") is True:
                continue
            selectors = option.get("selectors") or {}
            if not selectors:
                # No projection exists at all; the reachability declaration is
                # what carries this option's accuracy
                # (tests/test_registry_reachability.py).
                continue
            forced = dict(baseline)
            forced.update(selectors)
            refusal = _config_refusal(forced, nested=False)
            exercised.append((f"{component_id}.{option_id}", refusal))
            assert refusal is not None, (
                f"{component_id}.{option_id} is registered as NOT implemented "
                f"and its selectors {selectors} are accepted by "
                "validate_run_config; a namelist or TOML naming them would "
                "validate and then fail at driver construction")
    if exercised:
        return
    # Every unimplemented option currently declares NO selectors, so the
    # loop above measured nothing.  That is genuinely true today --
    # radiation.wrf-rrtm-dudhia was the last one that did, and its port
    # landed (woof/core/rrtm_lw.py) -- and it is exactly the state in
    # which a gate quietly stops gating.  Rather than delete the gate and
    # lose the mechanism the day a new port registers its selectors, keep
    # it and prove two things instead: that the emptiness has the reason
    # claimed, and that the detector still detects.
    unimplemented = {
        f"{component_id}.{option_id}": option.get("selectors") or {}
        for component_id, component in registry["components"].items()
        for option_id, option in component["options"].items()
        if option.get("implemented") is not True}
    assert unimplemented, (
        "no option is registered unimplemented at all; this gate has no "
        "subject left and should be deleted rather than left standing")
    assert all(not selectors for selectors in unimplemented.values()),         unimplemented
    # Positive control: plant a selector value the runtime authority must
    # refuse and require _config_refusal to see it.  If this ever passes
    # silently the loop above was never going to catch anything either.
    planted = dict(baseline)
    planted["mp_physics"] = 999
    assert _config_refusal(planted, nested=False) is not None, (
        "the refusal detector this gate depends on no longer refuses an "
        "impossible selector, so its empty result above means nothing")


@pytest.mark.parametrize("code", sorted(_SHARED_CODES | _REGISTRY_ONLY_CODES))
def test_every_classified_code_is_a_code_the_validator_can_emit(code: str):
    """Classification cannot drift into naming codes that no longer exist."""

    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "woof"
              / "physics_registry.py").read_text(encoding="utf-8")
    assert f'"{code}"' in source, (
        f"{code!r} is classified here but woof/physics_registry.py never "
        "emits it")


# ===========================================================================
# THE CONSUMER TABLES AGREE WITH THE REGISTRY.
#
# The 2026-09-10 trigger: an accepted, dispatched, registry-implemented
# scheme (Milbrandt-Yau) had no row in ONE of the tree's hand-kept scheme
# lists, and the list was read for the first time inside the run.  The
# registry now publishes a ``consumers`` block per implemented option
# (tools/build_registry.py), every consumer derives its table from that
# block or asserts equality with it at import, and the assertions below are
# the same questions asked as tests -- one per consumer -- so a scheme that
# lands without its rows fails the suite, not a forecast.
#
# Tables that cannot be derived yet are held to the registry with their
# deliberate omissions CITED by defect id (``cited_absences``): a citation
# that stops being true fails here, which is the retirement sweep as a test.
# ===========================================================================

from woof.physics_registry import (
    CONSUMER_ROWS_KEY,
    CONSUMER_ROW_CONTRACT,
    component_options,
    consumer_row_gaps,
    consumer_rows_by_selector,
    implemented_selector_values,
    require_consumer_rows_agreement,
    require_registry_agreement,
    require_template_menu_agreement,
)

_ROOT = Path(__file__).resolve().parents[1]


def _observed_and_absences(consumer: str):
    """The consumer's own key set and its cited absences, by name."""

    if consumer == "checkpoint_identity.MICROPHYSICS_ALGORITHM_IDENTITIES":
        from woof import checkpoint_identity as ci
        return "microphysics", ci.MICROPHYSICS_ALGORITHM_IDENTITIES, {}
    if consumer == "checkpoint_identity.PBL_ALGORITHM_IDENTITIES":
        from woof import checkpoint_identity as ci
        return "pbl", ci.PBL_ALGORITHM_IDENTITIES, {}
    if consumer == "checkpoint_identity.SURFACE_LAYER_ALGORITHM_IDENTITIES":
        from woof import checkpoint_identity as ci
        return "surface_layer", ci.SURFACE_LAYER_ALGORITHM_IDENTITIES, {}
    if consumer == "checkpoint_identity.LAND_SURFACE_ALGORITHM_IDENTITIES":
        from woof import checkpoint_identity as ci
        return "land_surface", ci.LAND_SURFACE_ALGORITHM_IDENTITIES, {}
    if consumer == "checkpoint_identity.CUMULUS_ALGORITHM_IDENTITIES":
        from woof import checkpoint_identity as ci
        return "cumulus", ci.CUMULUS_ALGORITHM_IDENTITIES, {}
    if consumer == "checkpoint_identity.LAND_SURFACE_PARAMETER_SOURCES":
        from woof import checkpoint_identity as ci
        return "land_surface", ci.LAND_SURFACE_PARAMETER_SOURCES, {
            0: "no land surface, no parameter bundle"}
    if consumer == "microphysics_transition.PORTED_MP_PHYSICS":
        from woof.core import microphysics_transition as mt
        absences = dict(mt.MIXED_EDGE_ABSENCES)
        absences.update({mp: ": ".join(mt._UNVALIDATED_MIXED_EDGE_REASONS[mp])
                         for mp in mt.UNVALIDATED_MIXED_EDGE_SELECTORS})
        # Column-kernel schemes (MP7 Goddard four-ice) carry their refusal
        # on the registry option itself; the module's own registry check
        # cites the same rows (_require_agreement_with_the_registry).
        absences.update({
            mp: option["extensions"]["column_kernel"]["mixed_edge_refusal"]
            for mp, option in mt._column_unvalidated_options().items()})
        return "microphysics", mt.PORTED_MP_PHYSICS, absences
    if consumer == "obsop.CLEAR_AIR_FLOOR_DBZ":
        from woof.da import obsop
        from woof.microphysics_schemes import NAMED_MP_IDS
        return "microphysics", obsop.CLEAR_AIR_FLOOR_DBZ, {
            0: "no microphysics, no operator",
            11: ("CAMMGMP never writes REFL_10CM; its retained zero field "
                 "does not define a radar operator or a clear-air floor"),
            30: ("FAST_SBM writes no REFL_10CM, so radar DA has no H(x) and "
                 "no clear-air floor to read on mp_physics=30 (the cited "
                 "absence in woof.da.obsop._require_agreement_with_the_registry)"),
            **{mp: "named scheme: its Z is fused with the step "
               "(obsop.NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP), radar DA "
               "refuses it, no floor is read" for mp in NAMED_MP_IDS if mp != 11},
            # The ported schemes the operator refuses by name with their WRF
            # lines (MP55 ISHMAEL, MP7, the Ferrier pair), which therefore
            # read no clear-air floor either.
            **{mp: reason for mp, reason
               in obsop.NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP.items()
               if mp not in obsop.CLEAR_AIR_FLOOR_DBZ}}
    if consumer == "refl.REFL_10CM_INPUT_SPECIES + SCHEME_NATIVE_REFL_10CM":
        from woof.core import refl
        # The module's own cited absences (CAM MG never writes REFL_10CM).
        return "microphysics", (set(refl.REFL_10CM_INPUT_SPECIES)
                                | set(refl.SCHEME_NATIVE_REFL_10CM)), dict(
            refl.SCHEME_ABSENT_REFL_10CM)
    if consumer == "moments._STATIC_SCHEMES":
        from woof.da import moments
        return "microphysics", moments._STATIC_SCHEMES, {
            0: "no microphysics, no moments",
            18: "resolved from namelist switches by woof.core.nssl2_contract"}
    if consumer == "offline_child.OFFLINE_CHILD_MP_PHYSICS":
        from woof import offline_child
        absences = {
            mp: row["refusal"] for mp, row in
            consumer_rows_by_selector("microphysics", "offline_child").items()
            if row.get("same_scheme") is not True}
        return "microphysics", offline_child.OFFLINE_CHILD_MP_PHYSICS, absences
    if consumer == "wrf_physics_inventory._INVENTORIES":
        from woof.wrf_physics_inventory import supported_stock_wrf_mp_physics
        from woof.microphysics_schemes import NAMED_MP_IDS
        return "microphysics", supported_stock_wrf_mp_physics(), {
            mp: "export-only scope (audit R-014)" for mp, row in
            consumer_rows_by_selector("microphysics", "stock_wrf_export").items()
            if row.get("inventoried") is not True}
    if consumer == "physics_compat._GLW_CONSUMING_SURFACE_SCHEMES":
        from woof.physics_compat import _GLW_CONSUMING_SURFACE_SCHEMES
        return "land_surface", _GLW_CONSUMING_SURFACE_SCHEMES, {
            0: "no land surface reads GLW"}
    if consumer == "physics_inventory ring guard":
        from woof.core.physics_inventory import ring_guard_row
        rows = {mp: True for mp in implemented_selector_values("microphysics")
                if mp and ring_guard_row(mp)}
        return "microphysics", rows, {0: "no microphysics, no ring guard"}
    if consumer == "rrtmgp._MP_CLOUD_OPTICS_SCHEME + _NO_CLOUD_OPTICS_COUPLING":
        from woof.core import rrtmgp
        return "microphysics", (set(rrtmgp._MP_CLOUD_OPTICS_SCHEME)
                                | set(rrtmgp._NO_CLOUD_OPTICS_COUPLING)), {}
    if consumer == "rrtmg_legacy._MP_DECLARES_RADII":
        from woof.core import rrtmg_legacy
        return "microphysics", rrtmg_legacy._MP_DECLARES_RADII, {}
    # The WRF v4.6.1 transcription stays hand-written by design (it
    # transcribes WRF's own fatals); its gaps are cited -- and cited ONCE,
    # in the module that owns the axes, not a second time here.  These
    # four dicts used to be that second copy, which is the drift family
    # this whole gate exists for: the cu=16 citation outlived its defect
    # by exactly as long as it took someone to notice this file.  Reading
    # AXIS_EXCLUSIONS means adding a scheme to an axis retires its
    # citation in one edit, and a citation with no reason is refused by
    # the axis-coverage test above.
    from woof import wrf461_compatibility as w

    def axis_absences(axis: str) -> dict:
        return {value: reason
                for (name, value), reason in w.AXIS_EXCLUSIONS.items()
                if name == axis}

    if consumer == "wrf461_compatibility.MP_OPTIONS":
        return "microphysics", w.MP_OPTIONS, axis_absences("mp_physics")
    if consumer == "wrf461_compatibility.PBL_OPTIONS":
        return "pbl", w.PBL_OPTIONS, axis_absences("bl_pbl_physics")
    if consumer == "wrf461_compatibility.SURFACE_LAYER_OPTIONS":
        return ("surface_layer", w.SURFACE_LAYER_OPTIONS,
                axis_absences("sf_sfclay_physics"))
    if consumer == "wrf461_compatibility.CUMULUS_OPTIONS":
        return "cumulus", w.CUMULUS_OPTIONS, axis_absences("cu_physics")
    if consumer == "wrf461_compatibility.LAND_SURFACE_OPTIONS":
        return ("land_surface", w.LAND_SURFACE_OPTIONS,
                axis_absences("sf_surface_physics"))
    raise KeyError(consumer)


_CONSUMERS = (
    "checkpoint_identity.MICROPHYSICS_ALGORITHM_IDENTITIES",
    "checkpoint_identity.PBL_ALGORITHM_IDENTITIES",
    "checkpoint_identity.SURFACE_LAYER_ALGORITHM_IDENTITIES",
    "checkpoint_identity.LAND_SURFACE_ALGORITHM_IDENTITIES",
    "checkpoint_identity.CUMULUS_ALGORITHM_IDENTITIES",
    "checkpoint_identity.LAND_SURFACE_PARAMETER_SOURCES",
    "microphysics_transition.PORTED_MP_PHYSICS",
    "obsop.CLEAR_AIR_FLOOR_DBZ",
    "refl.REFL_10CM_INPUT_SPECIES + SCHEME_NATIVE_REFL_10CM",
    "moments._STATIC_SCHEMES",
    "offline_child.OFFLINE_CHILD_MP_PHYSICS",
    "wrf_physics_inventory._INVENTORIES",
    "physics_compat._GLW_CONSUMING_SURFACE_SCHEMES",
    "physics_inventory ring guard",
    "rrtmgp._MP_CLOUD_OPTICS_SCHEME + _NO_CLOUD_OPTICS_COUPLING",
    "rrtmg_legacy._MP_DECLARES_RADII",
    "wrf461_compatibility.MP_OPTIONS",
    "wrf461_compatibility.PBL_OPTIONS",
    "wrf461_compatibility.SURFACE_LAYER_OPTIONS",
    "wrf461_compatibility.CUMULUS_OPTIONS",
    "wrf461_compatibility.LAND_SURFACE_OPTIONS",
)


@pytest.mark.parametrize("consumer", _CONSUMERS)
def test_every_consumer_table_agrees_with_the_registry(consumer):
    """implemented selectors == consumer keys, every gap cited."""

    component_id, observed, absences = _observed_and_absences(consumer)
    require_registry_agreement(consumer, component_id, set(observed),
                               cited_absences=absences)


def test_the_agreement_helper_fails_on_an_uncited_gap():
    """A gate nobody has seen fail is not evidence."""

    from woof import checkpoint_identity as ci

    table = dict(ci.MICROPHYSICS_ALGORITHM_IDENTITIES)
    table.pop(9)
    with pytest.raises(RuntimeError, match=r"selector 9 .* has no row"):
        require_registry_agreement("probe", "microphysics", table)
    # ...and a citation that has stopped being true fails too.
    with pytest.raises(RuntimeError, match="retire the citation"):
        require_registry_agreement(
            "probe", "microphysics", ci.MICROPHYSICS_ALGORITHM_IDENTITIES,
            cited_absences={9: "stale"})


def test_every_implemented_option_carries_every_consumer_row():
    """The contract the builder enforces, held on the shipped document."""

    registry = physics_registry()
    for component_id, contract in CONSUMER_ROW_CONTRACT.items():
        for option_id, option in registry["components"][component_id]["options"].items():
            if option.get("implemented") is not True:
                assert CONSUMER_ROWS_KEY not in option, (
                    f"{component_id}.{option_id} is not implemented and "
                    "still publishes consumer rows")
                continue
            rows = option.get(CONSUMER_ROWS_KEY)
            assert isinstance(rows, dict), f"{component_id}.{option_id}"
            assert set(contract) <= set(rows), (
                f"{component_id}.{option_id} lacks "
                f"{sorted(set(contract) - set(rows))}")


def test_the_consumer_row_gate_is_silent_for_every_implemented_tuple():
    """After the registry carries every row the gate has nothing to say."""

    registry = physics_registry()
    options = _component_option_ids(registry)
    for values in itertools.product(*(options[c] for c in sorted(options))):
        settings = {}
        for component_id, option_id in zip(sorted(options), values):
            option = registry["components"][component_id]["options"][option_id]
            if option.get("implemented") is not True:
                break
            settings.update(option.get("selectors", {}))
        else:
            assert consumer_row_gaps(settings) == [], settings


def test_the_consumer_row_gate_refuses_a_missing_row_before_step_0(monkeypatch):
    """Positive control: take one row away and both authorities refuse."""

    import woof.physics_registry as pr

    perturbed = deepcopy(pr.registry_view())
    option = perturbed["components"]["microphysics"]["options"]["milbrandt2mom-mp9"]
    del option[CONSUMER_ROWS_KEY]["ring_guard"]
    monkeypatch.setattr(pr, "_REGISTRY", perturbed)

    cfg = RunConfig(**_GEOMETRY, moist=True, mp_physics=9,
                    ra_lw_physics=4, ra_sw_physics=4,
                    ra_rrtmg_variant="rrtmg_legacy", cu_physics=0,
                    bl_pbl_physics=1, sf_sfclay_physics=91,
                    sf_surface_physics=2, num_soil_layers=4)
    with pytest.raises(ValueError) as caught:
        validate_run_config(cfg)
    message = str(caught.value)
    # The refusal names the scheme, the consumer and the way out.
    assert "milbrandt2mom-mp9" in message
    assert "ring_guard" in message
    assert "ring guard" in message
    assert "tools/build_registry.py" in message

    # The registry authority says the same thing under the shared code.
    plan_registry = deepcopy(perturbed)
    plan_registry["runner_routes"][_PERMISSIVE_RUNNER] = (
        _permissive_registry()["runner_routes"][_PERMISSIVE_RUNNER])
    plan_registry["runner_routes"][_PERMISSIVE_RUNNER][
        "allowed_parameter_keys"] = ["ra_rrtmg_variant"]
    report = validate_physics_plan(_single_domain_plan(
        plan_registry, _PERMISSIVE_RUNNER, "any-source",
        _base_template_id(plan_registry),
        components={"microphysics": "milbrandt2mom-mp9",
                    "radiation": "rte-rrtmgp", "cumulus": "off"},
        parameters={"ra_rrtmg_variant": "rrtmg_legacy"}),
        registry=plan_registry)
    assert report["launchable"] is False
    codes = {error["code"] for error in report["errors"]}
    assert "consumer-row-missing" in codes
    disagreements: list[str] = []
    _compare("perturbed-ring-guard", report, disagreements,
             registry=plan_registry)
    assert disagreements == []


def test_every_implemented_scheme_couples_to_rte_rrtmgp_at_plan_review():
    """No cloud_optics row publishes a null coupling except the one judged
    exclusions (FAST_SBM and Purdue Lin, refused at the run door), and every other
    implemented scheme passes the 4/4 RTE+RRTMGP pairing on the DEFAULT
    variant at plan review.

    This test used to assert the opposite half -- that the one scheme
    publishing a null coupling (mp=9) was refused by validate_run_config
    with the legacy variant named as the way out -- and carried its own
    retirement note for the day no scheme published a null.  That day is
    here: the Milbrandt-Yau row landed (woof.core.rrtmgp ``9:
    "milbrandt2"``), so the test now pins that the null state does not
    come back and that the refusal it described no longer fires on any
    implemented scheme.  The generic gate is unchanged: an UNJUDGED
    selector still fails closed in woof.core.rrtmgp.cloud_optics_scheme.
    """

    rows = consumer_rows_by_selector("microphysics", "cloud_optics")
    assert rows, "no cloud_optics rows published"
    uncoupled = sorted(mp for mp, row in rows.items()
                       if row["rte_rrtmgp_coupling"] is None)
    # FAST_SBM (mp=30) is the one JUDGED exclusion: its bin effective-radius
    # integrals are not qualified, so woof.core.rrtmgp records it in
    # _NO_CLOUD_OPTICS_COUPLING and both authorities refuse the RTE+RRTMGP
    # pairing (measured below) with the legacy-RRTMG remedy.  Any other
    # null is a gap, as it was when mp=9 was retired by writing its row.
    # SBU-YLin (13) and NTU (56) landed with the same judged exclusion as
    # Lin: neither hands radiation radii (NTU is absent from WRF's use_mp_re
    # disjunction), and each row names the legacy-RRTMG remedy.
    assert uncoupled == [2, 13, 30, 56], (
        "a scheme publishes a null RTE+RRTMGP coupling other than the "
        f"judged FAST_SBM, Lin, SBU-YLin and NTU exclusions: {uncoupled}")
    for mp in (13, 56):
        assert "rrtmg_legacy" in rows[mp]["rte_rrtmgp_refusal"]
    assert rows[30]["rte_rrtmgp_refusal"] is not None
    # Lin (mp=2) has no effective-radius output at all (Registry.EM_COMMON:3016).
    assert "Lin supplies no effective radii" in rows[2]["rte_rrtmgp_refusal"]
    assert rows[9]["rte_rrtmgp_coupling"] == "milbrandt2"
    assert rows[9]["rte_rrtmgp_refusal"] is None
    assert rows[9]["legacy_declares_radii"] is False
    for mp in sorted(rows):
        if mp == 0:
            continue
        kwargs = dict(_GEOMETRY, moist=True, mp_physics=mp,
                      ra_lw_physics=4, ra_sw_physics=4, cu_physics=0,
                      bl_pbl_physics=1, sf_sfclay_physics=91,
                      sf_surface_physics=2, num_soil_layers=4)
        refusal = _config_refusal(dict(kwargs), nested=False)
        if rows[mp]["rte_rrtmgp_coupling"] is None:
            # The run door refuses before step 0, naming the remedy.
            assert refusal is not None and "cloud-optics" in refusal, (mp, refusal)
            assert "rrtmg_legacy" in refusal, (mp, refusal)
        else:
            assert refusal is None, (mp, refusal)
        legacy = _config_refusal(
            dict(kwargs, ra_rrtmg_variant="rrtmg_legacy"), nested=False)
        assert legacy is None, (mp, legacy)


def _registry_publishing_an_unported_edge():
    """A registry doctored to publish one cross edge with an unported end.

    The SHIPPED registry publishes no such edge any more: it published ten
    of them when mp=9 was in neither of the resolver's tuples, and the
    build now resolves every published row through the resolver itself, so
    an over-claim fails the build (audit R-003).  Nor is any selector
    unported any longer -- 2.7.3 ratified the last two entry closures.  The
    gate still has to be exercised, and a positive control that can only
    run while a defect exists is not a control, so the UNPORTED END is
    synthesised here: the registry's own wsm6 -> wdm6 row is kept and
    wdm6-mp16's consumer row is doctored to say its mixed edges are not
    ported, which is what a hand-edited registry would look like.
    """

    registry = physics_registry()
    rules = registry["transitions"]["microphysics-one-way-v1"]["cross_options"]
    rule = next(rule for rule in rules
                if rule["parent_option_id"] == "wsm6-mp6"
                and rule["child_option_id"] == "wdm6-mp16")
    option = registry["components"]["microphysics"]["options"]["wdm6-mp16"]
    option[CONSUMER_ROWS_KEY] = deepcopy(dict(option[CONSUMER_ROWS_KEY]))
    option[CONSUMER_ROWS_KEY]["nest_transition"] = {"mixed_edge_ported": False}
    return registry, "wsm6-mp6", "wdm6-mp16", rule


def test_the_unported_endpoint_refusal_names_the_way_out():
    registry, parent, child, _rule = _registry_publishing_an_unported_edge()
    plan = _tree_plan(registry, "tools.prepared_domain_tree_forecast", "era5",
                      "wsm6-ysu-mm5-noah-no-radiation-v1")
    plan["domains"][0]["components"] = {"microphysics": parent}
    plan["domains"][1]["components"] = {"microphysics": child}
    report = validate_physics_plan(plan, registry=registry)
    messages = [error["message"] for error in report["errors"]
                if error["code"] == "transition-unported-endpoint"]
    assert messages, report["errors"]
    for message in messages:
        assert "same-scheme edge" in message
        assert "mixed_edge_ported=true" in message


def test_every_published_cross_edge_has_two_ported_endpoints():
    """The shipped registry publishes no over-claimed edge at all.

    This is the statement the gate above exists to enforce, asserted on the
    shipped table: before R-003 the registry admitted ten mp=9 edges that
    woof.core.microphysics_transition refused at nest construction.
    """

    unported = {
        option_id for option_id, option in component_options("microphysics").items()
        if option[CONSUMER_ROWS_KEY]["nest_transition"]["mixed_edge_ported"] is not True}
    published = {
        (rule["parent_option_id"], rule["child_option_id"])
        for rule in physics_registry()[
            "transitions"]["microphysics-one-way-v1"]["cross_options"]}
    touching = sorted(edge for edge in published if set(edge) & unported)
    assert not touching, (
        "the registry publishes cross edges whose endpoints the nest-edge "
        f"resolver does not port: {touching}")


def test_the_menu_citations_are_named_through_the_registry():
    """The R-068 omissions are keyed by the templates' composition and by
    the door's own menu, never by a source-named id literal, and the
    resolved keys are exactly the templates without a runtime-switch
    row."""

    from woof.physics_compat import (
        _SINGLE_DOMAIN_RUNTIME_SWITCHES, _TEMPLATES_OUTSIDE_THE_SINGLE_DOMAIN_MENU)
    from woof.physics_menu import (
        _TEMPLATES_OUTSIDE_THE_WIZARD_MENU, WIZARD_PHYSICS_PROFILES)
    from woof.physics_registry import (
        DEFAULT_TEMPLATE_ID, template_ids_with_components)

    templates = set(physics_registry()["templates"])
    without_switch_row = templates - set(_SINGLE_DOMAIN_RUNTIME_SWITCHES)
    assert set(_TEMPLATES_OUTSIDE_THE_SINGLE_DOMAIN_MENU) == without_switch_row
    assert without_switch_row <= set(_TEMPLATES_OUTSIDE_THE_WIZARD_MENU)
    assert DEFAULT_TEMPLATE_ID in without_switch_row
    # RETIRED, with the citation it guarded: this asserted that the one
    # WSM6 + KF template on the aggregate RTE+RRTMGP option had no
    # runtime-switch row.  It has one now -- the menu is derived from
    # every fixed-template route's own declaration, that template is
    # declared on one of them, and the derivation reproduced the
    # twenty-three-value row the single-domain door used to hand-type for
    # it exactly.  The assertion is inverted rather than deleted, so a
    # regression that drops the template back out of the menu, or
    # re-cites it as an omission, fails here.
    aggregate_kf = template_ids_with_components(
        microphysics="wsm6-mp6", cumulus="kain-fritsch",
        radiation="rte-rrtmgp")
    assert len(aggregate_kf) == 1
    assert aggregate_kf[0] not in without_switch_row
    assert _SINGLE_DOMAIN_RUNTIME_SWITCHES[aggregate_kf[0]]["mp_physics"] == 6
    # The SAME retirement, on the wizard's side of the pair.  It was the
    # half that did not get swept: woof/physics_menu.py went on excluding
    # this template from the wizard menu with the R-068 reason after the
    # door had resolved a row for it, so a user picking a suite was never
    # offered a composition the route beneath the wizard declares.  Both
    # legs are asserted here so the next sweep is one test, not two greps.
    assert aggregate_kf[0] not in _TEMPLATES_OUTSIDE_THE_WIZARD_MENU
    assert aggregate_kf[0] in WIZARD_PHYSICS_PROFILES
    assert template_ids_with_components(microphysics="no-such-option") == ()
    source = (_ROOT / "woof" / "physics_compat.py").read_text(encoding="utf-8")
    for template_id in without_switch_row:
        assert template_id not in source, template_id


def test_a_cross_edge_touching_an_unported_endpoint_is_refused_at_plan_review():
    """Plan review refuses a published cross edge the resolver would refuse
    at tree load, with the resolver's own reason."""

    registry, parent, child, rule = _registry_publishing_an_unported_edge()
    route = registry["runner_routes"]["tools.prepared_domain_tree_forecast"]
    template_id = "wsm6-ysu-mm5-noah-no-radiation-v1"
    plan = _tree_plan(registry, "tools.prepared_domain_tree_forecast", "era5",
                      template_id)
    plan["domains"][0]["components"] = {"microphysics": parent}
    plan["domains"][1]["components"] = {"microphysics": child}
    plan["domains"][1]["parameters"].update(rule.get("required_child_settings", {}))
    plan["domains"][0].setdefault("parameters", {}).update(
        rule.get("required_parent_settings", {}))
    assert "microphysics" in route["allowed_component_overrides"]
    report = validate_physics_plan(plan, registry=registry)
    codes = {error["code"] for error in report["errors"]}
    assert "transition-unported-endpoint" in codes, report["errors"]


def test_the_two_profile_menus_agree_with_the_templates():
    from woof.physics_compat import (
        SINGLE_DOMAIN_PHYSICS_PROFILES, _SINGLE_DOMAIN_RUNTIME_SWITCHES,
        _TEMPLATES_OUTSIDE_THE_SINGLE_DOMAIN_MENU)
    from woof.physics_menu import (
        WIZARD_PHYSICS_PROFILES, _TEMPLATES_OUTSIDE_THE_WIZARD_MENU)

    require_template_menu_agreement(
        "SINGLE_DOMAIN_PHYSICS_PROFILES", SINGLE_DOMAIN_PHYSICS_PROFILES,
        cited_absences=_TEMPLATES_OUTSIDE_THE_SINGLE_DOMAIN_MENU)
    require_template_menu_agreement(
        "_SINGLE_DOMAIN_RUNTIME_SWITCHES", tuple(_SINGLE_DOMAIN_RUNTIME_SWITCHES),
        cited_absences=_TEMPLATES_OUTSIDE_THE_SINGLE_DOMAIN_MENU)
    require_template_menu_agreement(
        "WIZARD_PHYSICS_PROFILES", WIZARD_PHYSICS_PROFILES,
        cited_absences=_TEMPLATES_OUTSIDE_THE_WIZARD_MENU)
    assert set(WIZARD_PHYSICS_PROFILES) <= set(SINGLE_DOMAIN_PHYSICS_PROFILES)
    # Every runtime-switch row's selectors are the template's own.
    registry = physics_registry()
    for template_id, switches in _SINGLE_DOMAIN_RUNTIME_SWITCHES.items():
        template = registry["templates"][template_id]
        for component_id, option_id in template["components"].items():
            option = registry["components"][component_id]["options"][option_id]
            for key, value in option.get("selectors", {}).items():
                if value == -1:
                    continue  # the legacy aggregate spelling
                assert switches.get(key) == value, (
                    template_id, component_id, key, switches.get(key), value)


def test_the_streamed_composite_reads_the_operator_not_a_copy_of_morrison():
    """The streamed composite derives its species (audit R-052, CLOSED).

    The guard this replaces pinned ``tilestream/realcase.py``'s literal
    ``REFL_SPECIES`` tuple to Morrison's registry row so the copy could not
    drift into a third spelling.  The copy is gone -- both streamed lanes
    call ``woof.core.refl.refl_10cm_input_species`` -- so what is pinned
    now is that neither lane has grown a new literal, and that the derived
    answer is right for every scheme the operator dispatches, not just for
    the one whose list used to be typed there.
    """

    from woof.core.refl import REFL_10CM_INPUT_SPECIES
    from tilestream.realcase import _refl_slab_carriers

    for path in ("realcase.py", "bigdomain.py"):
        source = (_ROOT / "tilestream" / path).read_text(encoding="utf-8")
        assert "REFL_SPECIES" not in source, (
            f"tilestream/{path} has grown a reflectivity species literal "
            "again; the operator's REFL_10CM_INPUT_SPECIES is the one table")
        assert '"nr", "qs", "ns"' not in source, (
            f"tilestream/{path} spells Morrison's six-moment set inline")

    for mp, species in REFL_10CM_INPUT_SPECIES.items():
        assert _refl_slab_carriers(mp) == tuple(species) + ("p", "thp"), mp


def test_the_streamed_composite_refuses_a_native_z_scheme_by_name():
    """A native-Z scheme's slab route says where the field IS.

    mp=9, 18 and 50 read no operator inputs, so asking for their slab
    carriers is not a missing-field condition and must not be reported as
    one: the refusal carries the scheme's own reason out of
    ``SCHEME_NATIVE_REFL_10CM``.
    """

    from woof.core.refl import (NativeReflectivityScheme,
                                 SCHEME_NATIVE_REFL_10CM)
    from tilestream.realcase import _refl_slab_carriers

    for mp in SCHEME_NATIVE_REFL_10CM:
        with pytest.raises(NativeReflectivityScheme, match="refl_10cm"):
            _refl_slab_carriers(mp)


def _refl_probe_state():
    """A buffer-shaped stand-in: the two scratch methods priming reads."""

    class _Buffer:
        def __init__(self):
            self.slots = {}

        def existing_scratch(self, name):
            return self.slots.get(name)

        def scratch(self, shape, name):
            self.slots[name] = np.zeros(shape, dtype=np.float32)
            return self.slots[name]

    return _Buffer()


def test_a_reflectivity_carrying_sweep_is_configured_end_to_end():
    """Audit R-052, the half the store-side inventory did not reach.

    A store that lists ``scratch/refl_10cm`` publishes the model's own dBZ
    only if the sweep WRITES it, and that takes four agreeing pieces:
    the inventory rule the transport moves by must name the slot, every
    tile buffer must carry it before its inventory is taken, every tile
    step must be told a frame is due, and each tile's one-frame stash must
    be cleared or the second tile a buffer serves refuses the sweep.  The
    lane that shipped the store side alone left every tile scattering an
    unwritten window, which is a plausible 0 dBZ product.
    """

    from tilestream import driver, physics_inventory as physinv

    built = []

    def factory(tile_cfg):
        built.append(tile_cfg)
        return _refl_probe_state()

    store = {"state/qv": np.zeros((2, 2, 2), dtype=np.float32),
             physinv.REFL_KEY: np.zeros((2, 2, 2), dtype=np.float32)}
    out = driver.reflectivity_run_kwargs(
        {"tile_state_factory": factory, "nz": 2}, store)

    assert out["inventory_fn"] is physinv.carrier_inventory_with_refl
    assert out["step_kwargs"]["refl_10cm_due"] is True
    assert callable(out["post_step_hook"])
    tile = out["tile_state_factory"](SimpleNamespace(nz=2, ny=2, nx=2))
    assert tile.existing_scratch("refl_10cm") is not None, (
        "a buffer whose slot is not primed differs from the store by "
        "exactly that key and TiledRun refuses the sweep")
    # The hook clears a tile's handoff and tolerates a tile with none.
    out["post_step_hook"](_HandoffState(), None, 0, None)


class _HandoffState:
    """A tile state whose driver holds an unconsumed REFL handoff."""

    def __init__(self):
        self.physics = SimpleNamespace(refl_10cm=np.zeros((1,)))


def test_a_store_without_the_reflectivity_slot_is_left_alone():
    """A lane that asked for no reflectivity pays for none of it."""

    from tilestream import driver

    kwargs = {"tile_state_factory": lambda cfg: None, "nz": 2}
    assert driver.reflectivity_run_kwargs(kwargs, {"state/qv": 1}) == kwargs


def test_run_tiled_forwards_step_kwargs_to_the_sweep(monkeypatch):
    """``refl_10cm_due`` reaches a sweep through the whole-run entry point.

    It was reachable only by holding a ``TiledRun``, and both streamed
    product lanes call ``run_tiled``, so no tile in either ever ran
    calc_refl10cm.
    """

    from tilestream import driver

    seen = {}

    class _FakeRun:
        def __init__(self, *args, **kwargs):
            pass

        def sweep(self, nsteps, *, step_kwargs=None, report=None,
                  progress=None):
            seen["nsteps"] = nsteps
            seen["step_kwargs"] = step_kwargs

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(driver, "TiledRun", _FakeRun)
    driver.run_tiled({}, None, 4, 4, nsteps=3,
                     step_kwargs={"refl_10cm_due": True})
    assert seen["step_kwargs"] == {"refl_10cm_due": True}
    assert seen["nsteps"] == 3 and seen["closed"]


def _reflectivity_wired_sweeps(module_name: str) -> list[str]:
    """Functions in a product lane whose ``run_tiled`` sweep WRITES the slot.

    Structural rather than textual: the lane's own module is parsed, and a
    function counts only when the keyword dict a ``run_tiled`` call expands
    is a name that a ``reflectivity_run_kwargs`` call in that same function
    rebound, AND the function marks its store swept afterwards
    (``CaseStores.reflectivity_swept`` / ``refl_stash[...] = "computed"``).
    A substring search cannot tell any of that from a comment mentioning the
    function beside a call that was deleted -- which both of these lanes
    carry, in prose, right next to the call.
    """
    import ast

    tree = ast.parse((_ROOT / "tilestream" / module_name).read_text(
        encoding="utf-8"))
    wired = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        rebound = set()
        for node in ast.walk(func):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and _called_name(node.value) == "reflectivity_run_kwargs"):
                rebound.update(t.id for t in node.targets
                               if isinstance(t, ast.Name))
        if not rebound:
            continue
        swept = any(
            (isinstance(node, ast.Call)
             and _called_name(node) == "reflectivity_swept")
            or (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Name)
                        and "refl_stash" in t.value.id for t in node.targets))
            for node in ast.walk(func))
        for node in ast.walk(func):
            if (isinstance(node, ast.Call)
                    and _called_name(node) == "run_tiled"
                    and any(kw.arg is None and isinstance(kw.value, ast.Name)
                            and kw.value.id in rebound for kw in node.keywords)
                    and swept):
                wired.append(func.name)
                break
    return wired


def _called_name(call) -> str:
    import ast

    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def test_the_streamed_product_lanes_ask_their_sweeps_for_reflectivity():
    """Both lanes audit R-052 names configure the sweep, not only the store.

    Structural rather than executed: the sweep itself is a card's worth of
    work.  What is pinned is that the lane whose product IS reflectivity
    cannot go back to building a slot-carrying store and sweeping it without
    the keywords that fill the slot -- the half of R-052 that the store-side
    inventory left undone, and the half a prose mention of the function
    beside a deleted call would have satisfied.
    """

    realcase = _reflectivity_wired_sweeps("run_realcase.py")
    # The forecast lane and the overlap A/B pair, which has to write the
    # slot on BOTH arms or the arms differ for a reason that is not the
    # transport under test.
    assert len(realcase) >= 2, realcase
    assert _reflectivity_wired_sweeps("run_bigdomain.py"), "bigdomain"


def test_the_reflectivity_stash_key_has_exactly_one_spelling_in_shipped_code():
    """R-052's single-spelling rule, checked outside the package that made it.

    ``woof.core.streaming`` owns the ``refl_10cm`` handoff for every
    streamed route -- the key, the priming, the inventory rule and the
    post-step clear -- and tilestream re-exports its name.  The claim beside
    those re-exports is that ONE module spells the key; it was true inside
    tilestream and false in the tree, because the restart reader typed the
    string twice while calling ``woof/core/streaming.py``'s constant the
    owner.  A second spelling is how a store built by one module and swept
    by another came to disagree in the first place, so the rule is checked
    here rather than asserted in a comment.

    Tests are exempt on purpose: a test that spells the key out is what
    notices the constant changing under it.
    """
    import ast

    from woof.core.streaming import REFL_STORE_KEY

    owner = _ROOT / "woof" / "core" / "streaming.py"
    offenders = []
    for path in sorted(_ROOT.glob("woof/**/*.py")) + sorted(
            _ROOT.glob("tilestream/*.py")) + sorted(_ROOT.glob("tools/*.py")):
        if path == owner or path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Docstrings and comments may name the key they explain; a STRING
        # the module evaluates is the spelling this rule is about.
        prose = {id(node.body[0].value) for node in ast.walk(tree)
                 if isinstance(node, (ast.Module, ast.ClassDef,
                                      ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.body and isinstance(node.body[0], ast.Expr)
                 and isinstance(node.body[0].value, ast.Constant)
                 and isinstance(node.body[0].value.value, str)}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant)
                    and node.value == REFL_STORE_KEY
                    and id(node) not in prose):
                offenders.append(f"{path.relative_to(_ROOT)}:{node.lineno}")
    assert not offenders, (
        "the reflectivity stash key is spelled outside "
        f"woof/core/streaming.py at {offenders}; import REFL_STORE_KEY")


def test_both_forecast_doors_route_the_checkpoint_question_through_one_function():
    """R-046's placement, in the door that does not have a tree.

    ``execute_experiment`` skips a ``state.physics`` that is not a
    ``PhysicsDriver`` (the identity is not defined over that route and would
    fail on an attribute, naming the gate rather than whatever attached the
    object); ``integrate_prepared_case`` asked ``physics_setup_identity``
    directly and so carried none of that.  One function owns the rule now,
    and the pin is that neither door reaches past it.
    """
    import ast

    doors = {("woof", "runtime.py"): "integrate_prepared_case",
             ("woof", "core", "model.py"): "_ask_the_checkpoints_question"}
    for parts, name in doors.items():
        tree = ast.parse((_ROOT.joinpath(*parts)).read_text(encoding="utf-8"))
        func = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == name), None)
        assert func is not None, name
        called = {_called_name(n) for n in ast.walk(func)
                  if isinstance(n, ast.Call)}
        assert "ask_checkpoint_physics_identity" in called, name
        assert "physics_setup_identity" not in called, name


def test_the_composite_reads_the_slot_only_when_a_call_has_written_it():
    """A PRIMED slot is zeros, and zeros are a plausible weak-echo product.

    Both halves matter: a computed slot is returned (the only route there
    is for a scheme that publishes its dBZ through no operator), and a
    primed one is not -- a native-Z scheme whose store has never been swept
    is refused, by name, with the way out.
    """

    from tilestream import physics_inventory as physinv
    from tilestream.realcase import (CaseStores, RealCaseError,
                                     composite_reflectivity)

    field = np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2)

    def case(stash, mp):
        return CaseStores(
            cfg=SimpleNamespace(nz=2, ny=2, nx=2, mp_physics=mp),
            grid=None, store={physinv.REFL_KEY: field.copy()},
            geo_store={}, scalars={}, boundaries=None, start_time=None,
            forcing_times=(), lat=None, lon=None, terrain=None, coord=None,
            store_bytes=0, geo_bytes=0, refl_stash=stash)

    computed = case("computed", 9)
    out = composite_reflectivity(computed, slab_rows=2)
    assert np.array_equal(out, field.max(axis=0))
    # A COPY: the store's array is a pinned buffer a later sweep scatters
    # into, and a product already handed to a caller must not change under
    # it.
    volume = composite_reflectivity(computed, slab_rows=2, column_max=False)
    volume[...] = -99.0
    assert float(computed.store[physinv.REFL_KEY].max()) == float(field.max())

    with pytest.raises(RealCaseError, match="reflectivity_run_kwargs"):
        composite_reflectivity(case("primed", 9), slab_rows=2)

    swept = case("primed", 9)
    swept.reflectivity_swept()
    assert swept.refl_stash == "computed"
    assert np.array_equal(composite_reflectivity(swept, slab_rows=2),
                          field.max(axis=0))


def test_the_preflight_reflectivity_rail_agrees_with_the_operator():
    """Pricing and dispatch are one decision (audit R-052).

    Positive control for the import-time hold: perturb either set and the
    module's own check refuses, so the gate is evidence rather than a
    comment.
    """

    from woof.core import preflight
    from woof.core.refl import (REFL_10CM_INPUT_SPECIES,
                                 SCHEME_NATIVE_REFL_10CM)

    from woof.core.column_schemes import COLUMN_SCHEMES
    assert set(preflight._REFLECTIVITY_MICROPHYSICS) == set(
        REFL_10CM_INPUT_SPECIES) - set(COLUMN_SCHEMES)
    assert set(preflight._SELF_REFLECTIVITY_MICROPHYSICS) == set(
        SCHEME_NATIVE_REFL_10CM) | set(COLUMN_SCHEMES)
    # mp=18 in particular: it computes its own dBZ, so it must never be
    # charged refl.cu's frame.
    assert 18 not in preflight._REFLECTIVITY_MICROPHYSICS
    assert 18 in preflight._SELF_REFLECTIVITY_MICROPHYSICS

    saved = preflight._REFLECTIVITY_MICROPHYSICS
    try:
        preflight._REFLECTIVITY_MICROPHYSICS = frozenset(saved | {18})
        with pytest.raises(RuntimeError, match="reflectivity-rail sets"):
            preflight._hold_reflectivity_rail_equal_to_the_operator()
    finally:
        preflight._REFLECTIVITY_MICROPHYSICS = saved
    preflight._hold_reflectivity_rail_equal_to_the_operator()


def test_the_consumer_export_for_the_rust_crates_matches_the_registry():
    """The JSON the renderers read equals what the registry says."""

    import json

    from woof.io.wrf_output_schema import PRECIPITATION_OUTPUT_FIELDS

    export = json.loads((_ROOT / "woof" / "physics_consumer_export_v1.json")
                        .read_text(encoding="utf-8"))
    assert export["schema"] == "gpuwm-physics-consumer-export-v1"
    assert export["precipitation_output_fields"] == list(PRECIPITATION_OUTPUT_FIELDS)
    implemented = implemented_selector_values("microphysics")
    assert {int(key) for key in export["microphysics"]} == set(implemented)
    for mp, option_id in implemented.items():
        row = export["microphysics"][str(mp)]
        assert row["option_id"] == option_id
        inventory = consumer_rows_by_selector("microphysics", "stock_wrf_export")[mp]
        assert row["stock_wrf_export_inventoried"] is bool(inventory["inventoried"])
    # HAILNC is filled by the admitted hail-bearing schemes: Milbrandt-Yau,
    # NSSL and WSM7, then Goddard four-ice, WDM7 and NTU from their ports.
    assert export["scheme_bound_precipitation_fields"]["HAILNC"] == [
        7, 9, 18, 24, 26, 56]


def test_every_transcribed_axis_covers_the_selectable_set_or_declares_why():
    """The drift the cumulus axis had, closed for every axis at once.

    ``woof.wrf461_compatibility`` transcribes WRF v4.6.1's own verdicts,
    and its axes are the ported set it represents.  cu_physics=16 was
    ported, selectable from ``woof.config.CU_SCHEMES``, and published by
    the registry this module generates as an implemented option -- while
    the matrix dimension that registry ALSO publishes said ``[0, 1, 3]``.
    One artifact contradicting itself, and nothing compared the two,
    because nothing on a run path reads the cumulus axis: the drift was
    latent rather than a live refusal, which is exactly how the mp=9
    checkpoint table stayed wrong until a run died on it.

    So the property is not set equality -- three selectors are outside
    these axes on purpose -- but coverage: every value a user can select
    is either represented here with its citation, or carries a row in
    ``AXIS_EXCLUSIONS`` saying why it cannot be.  A newly ported scheme
    fails this test until it does one or the other, and a blank reason is
    not a row.
    """

    from woof import config as config_module
    from woof import wrf461_compatibility as matrix

    axes = {
        "mp_physics": (matrix.MP_OPTIONS, config_module.MP_PHYSICS_ACCEPTED),
        "bl_pbl_physics": (matrix.PBL_OPTIONS, config_module.PBL_SCHEMES),
        "sf_sfclay_physics": (
            matrix.SURFACE_LAYER_OPTIONS, config_module.SURFACE_LAYER_SCHEMES),
        "sf_surface_physics": (
            matrix.LAND_SURFACE_OPTIONS, config_module.LAND_SURFACE_SCHEMES),
        "cu_physics": (matrix.CUMULUS_OPTIONS, config_module.CU_SCHEMES),
    }
    for axis, (represented, selectable) in axes.items():
        declared = {value for (name, value) in matrix.AXIS_EXCLUSIONS
                    if name == axis}
        assert set(represented) <= set(selectable), (
            f"{axis}: the matrix represents values woof cannot select")
        assert set(selectable) - set(represented) == declared, (
            f"{axis}: every selectable value must be represented in the "
            "WRF v4.6.1 transcription or declared in AXIS_EXCLUSIONS with "
            "its reason")
    for (axis, value), reason in matrix.AXIS_EXCLUSIONS.items():
        assert axis in axes, f"AXIS_EXCLUSIONS names an unknown axis {axis!r}"
        assert isinstance(reason, str) and reason.strip(), (
            f"AXIS_EXCLUSIONS[{axis!r}, {value!r}] carries no reason")
