from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import woof
import pytest

from woof.physics_registry import (
    expert_template_ids_for_source,
    INSTALL_STATE_CODES,
    MORRISON_TEMPLATE_ID,
    PLAN_SCHEMA,
    REGISTRY_SCHEMA,
    THOMPSON_TEMPLATE_ID,
    VALIDATION_SCHEMA,
    WSM6_TEMPLATE_ID,
    canonical_json,
    canonical_sha256,
    physics_registry,
    registry_sha256,
    validate_physics_plan,
)
from woof.physics_compat import (
    COMPOSITION_SUITE_PROFILE_IDS,
    KESSLER_PROFILE_ID,
    MYNN_NOAHMP_PROFILE_ID,
    MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID,
    NSSL2_LEGACY_RRTMG_PROFILE_ID,
    NSSL2_PROFILE_ID,
    NOAHMP_PROFILE_ID,
    P3_LEGACY_RRTMG_PROFILE_ID,
    SINGLE_DOMAIN_PHYSICS_PROFILES,
    THOMPSON_LEGACY_RRTMG_PROFILE_ID,
    THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
    identify_single_domain_profile,
    single_domain_runtime_switches,
)
from woof.source_cli import EXIT_CONFIG, main
from tools.hrrr_single_domain_benchmark import (
    runner_capabilities as hrrr_runner_capabilities,
)
from tools.prepared_domain_tree_forecast import (
    runner_capabilities as tree_runner_capabilities,
)
from tools.prepared_single_domain_forecast import (
    runner_capabilities as prepared_single_runner_capabilities,
)


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "woof" / "physics_registry_v2.json"


@pytest.mark.parametrize("option", ["off", "kessler-mp1", "wsm6-mp6"])
def test_moisture_pressure_correction_is_enabled_for_every_mass_package(option):
    registry = physics_registry()
    assert registry["parameters"]["moist_cq"]["default"] is True
    assert registry["components"]["microphysics"]["options"][option][
        "parameters"]["moist_cq"] is True


def _mixed_plan() -> dict[str, object]:
    return {
        "schema": PLAN_SCHEMA,
        "plan_id": "mixed-thompson-nssl-proof-v1",
        "registry_sha256": registry_sha256(),
        "context": {
            "source_id": "hrrr",
            "runner_id": "tools.prepared_domain_tree_forecast",
            "topology_id": "one-way-nested-v1",
        },
        "domains": [
            {"domain_id": "d01", "template_id": THOMPSON_TEMPLATE_ID},
            {
                "domain_id": "d02",
                "template_id": THOMPSON_TEMPLATE_ID,
                "components": {"microphysics": "nssl2-mp18"},
                "parameters": {
                    "epssm": 0.1,
                    "nest_microphysics_transition": (
                        "mp8-to-mp18-mass-diagnosed-v1"
                    ),
                },
            },
        ],
        "edges": [{"parent_domain_id": "d01", "child_domain_id": "d02"}],
    }


def _source_offering(template_id: str) -> str:
    """A source whose declared template list reaches ``template_id``.

    Sources and templates are not interchangeable: v1.1.1 withdrew RUC
    from the GFS route (a GFS-initialised RUC forecast cannot complete
    its first step) while leaving it on ERA5, which was never exercised.
    A helper that hardcoded one source would report that withdrawal as a
    failure of every claim it happens to be checking.
    """

    route = physics_registry()["runner_routes"][
        "tools.prepared_single_domain_forecast"]
    declared = route["source_template_ids"]
    for source_id in ("gfs", *sorted(declared)):
        if template_id in declared.get(source_id, ()):
            return source_id
    for source_id in ("gfs", *sorted(route.get("source_ids", []))):
        if template_id in expert_template_ids_for_source(route, source_id):
            return source_id
    raise AssertionError(
        f"no source offers {template_id}; it is unreachable on this route")


def _single_plan(template_id: str = WSM6_TEMPLATE_ID) -> dict[str, object]:
    plan = {
        "schema": PLAN_SCHEMA,
        "plan_id": "single-domain-proof-v1",
        "registry_sha256": registry_sha256(),
        "context": {
            "source_id": _source_offering(template_id),
            "runner_id": "tools.prepared_single_domain_forecast",
            "topology_id": "single-domain-v1",
        },
        "domains": [{"domain_id": "d01", "template_id": template_id}],
        "edges": [],
    }
    if template_id in (NOAHMP_PROFILE_ID, MYNN_NOAHMP_PROFILE_ID,
                       MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID):
        plan["acknowledgements"] = [
            "noahmp-host-column-throughput-v1"]
    return plan


@pytest.mark.parametrize("surface", ["revised-mm5", "classic-mm5", "pleim-xiu"])
def test_acm2_surface_pairs_are_launchable_physics_plans(surface):
    plan = _single_plan()
    plan["domains"][0]["components"] = {"pbl": "acm2", "surface_layer": surface}
    report = validate_physics_plan(plan)
    assert report["launchable"], report["errors"]


def _uniform_tree(template_id: str = WSM6_TEMPLATE_ID) -> dict[str, object]:
    return {
        "schema": PLAN_SCHEMA,
        "plan_id": "uniform-tree-proof-v1",
        "registry_sha256": registry_sha256(),
        "context": {
            "source_id": "hrrr",
            "runner_id": "tools.prepared_domain_tree_forecast",
            "topology_id": "one-way-nested-v1",
        },
        "domains": [
            {"domain_id": "d01", "template_id": template_id},
            {"domain_id": "d02", "template_id": template_id},
        ],
        "edges": [{"parent_domain_id": "d01", "child_domain_id": "d02"}],
    }


@pytest.mark.parametrize("profile", [MORRISON_TEMPLATE_ID, NSSL2_PROFILE_ID])
def test_historical_maturity_aliases_keep_plan_warnings_and_physics(profile):
    registry = physics_registry()
    plan = _single_plan(profile)
    original = validate_physics_plan(plan, registry=registry)
    aliases = registry["maturity_ladder"]["aliases"]
    historical = {current: old for old, current in aliases.items()}
    for component in registry["components"].values():
        for option in component["options"].values():
            option["maturity"] = historical.get(
                option.get("maturity"), option.get("maturity"))
    for template in registry["templates"].values():
        template["maturity"] = historical.get(
            template.get("maturity"), template.get("maturity"))
    plan["registry_sha256"] = registry_sha256(registry)
    compatible = validate_physics_plan(plan, registry=registry)
    assert compatible["launchable"] == original["launchable"]
    assert compatible["warnings"] == original["warnings"]
    assert compatible["errors"] == original["errors"]
    assert compatible["resolved_domains"] == original["resolved_domains"]


def test_historical_maturity_keeps_the_reported_wrf_evidence(monkeypatch):
    from types import SimpleNamespace
    import importlib
    from woof.physics_compat import single_domain_verification_status

    module = importlib.import_module("woof.physics_registry")
    config = SimpleNamespace(**single_domain_runtime_switches(MORRISON_TEMPLATE_ID))
    original = single_domain_verification_status(config)
    registry = physics_registry()
    registry["templates"][MORRISON_TEMPLATE_ID]["maturity"] = "model-validated"
    monkeypatch.setattr(module, "physics_registry", lambda: registry)
    assert single_domain_verification_status(config) == original
    assert original["matched_profile_maturity"] == "wrf-matched-run"


@pytest.mark.parametrize("old,new", [
    ("thompson-mp8-ysu-mm5-noah-validation-v1",
     "thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1"),
    ("nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-validation-candidate-v1",
     "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-wrf-comparison-candidate-v1"),
    ("nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-validation-candidate-v1",
     "nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-wrf-comparison-candidate-v1"),
])
def test_every_historical_profile_resolves_through_the_public_doors(old, new):
    from woof.cli import build_parser
    from woof.physics_registry import canonical_template_id, registry_physics_parts
    from woof.physics_compat import (
        physics_selection_differences, validate_single_domain_physics_profile)
    from tools.hrrr_single_domain_benchmark import (
        _native_hrrr_profile_contract, _native_hrrr_runtime_switches,
        _initialization_contract_profile)

    assert canonical_template_id(old) == new
    assert old not in SINGLE_DOMAIN_PHYSICS_PROFILES
    assert new in SINGLE_DOMAIN_PHYSICS_PROFILES
    settings = single_domain_runtime_switches(old)
    assert settings == single_domain_runtime_switches(new)
    assert _native_hrrr_profile_contract(old) == _native_hrrr_profile_contract(new)
    assert _native_hrrr_runtime_switches(old) == _native_hrrr_runtime_switches(new)
    assert _initialization_contract_profile(old) == _initialization_contract_profile(new)
    current = validate_single_domain_physics_profile(new, config=settings)
    assert validate_single_domain_physics_profile(old, config=settings) == current
    historical = {**current, "profile": old}
    assert physics_selection_differences(historical, current, settings=settings) == []
    assert registry_physics_parts(profile=old) == registry_physics_parts(profile=new)
    assert f"templates.{old}" in registry_physics_parts(profile=new)
    for command in ("domain", "prep"):
        argv = [command, "--physics-profile", old]
        if command == "domain":
            argv += ["--point", "35,-97", "--cycle", "2026-01-01T00", "--out", "alias.toml"]
        args = build_parser().parse_args(argv)
        assert args.physics_profile == new
    modern_plan = _single_plan(new)
    old_plan = {**modern_plan, "domains": [{"domain_id": "d01", "template_id": old}]}
    modern = validate_physics_plan(modern_plan)
    compatible = validate_physics_plan(old_plan)
    for field in ("launchable", "errors", "warnings", "resolved_domains"):
        assert compatible[field] == modern[field], field
    # A supplied old registry still owns its old identifiers and digest.
    # Compatibility never silently rebinds an exact-digest plan.
    legacy_registry = json.loads(canonical_json(physics_registry()).replace(new, old))
    legacy_registry.pop("template_aliases")
    old_plan["registry_sha256"] = registry_sha256(legacy_registry)
    legacy = validate_physics_plan(old_plan, registry=legacy_registry)
    assert legacy["launchable"] == modern["launchable"]
    assert legacy["errors"] == modern["errors"]
    assert [row["settings"] for row in legacy["resolved_domains"]] == [
        row["settings"] for row in modern["resolved_domains"]]
    current_with_old_pin = validate_physics_plan(old_plan)
    assert "stale-registry-binding" in {row["code"] for row in current_with_old_pin["errors"]}


def test_composition_exemptions_do_not_report_current_exact_suite_verification():
    from types import SimpleNamespace
    from woof.physics_compat import (VERIFICATION_WRF_VERIFIED, single_domain_verification_status)
    from woof.physics_registry import THOMPSON_KF_TEMPLATE_ID
    from woof.physics_menu import maturity

    for profile in (THOMPSON_KF_TEMPLATE_ID, MORRISON_TEMPLATE_ID):
        report = maturity(profile)
        assert report["verification_scope"] == "composition-exemption"
        assert report["verification_status"] != VERIFICATION_WRF_VERIFIED
    status = single_domain_verification_status(SimpleNamespace(
        **single_domain_runtime_switches(MORRISON_TEMPLATE_ID)))
    assert status["matched_profile_verification_scope"] == "composition-exemption"
    assert status["status"] != VERIFICATION_WRF_VERIFIED
    assert "carries WRF-verification evidence" not in status["sentence"]


def test_tracked_registry_is_the_exact_canonical_gpuwm_authority():
    registry = physics_registry()
    raw = REGISTRY_PATH.read_bytes()
    assert registry["schema"] == REGISTRY_SCHEMA
    assert registry["plan_schema"] == PLAN_SCHEMA
    assert registry["validation_schema"] == VALIDATION_SCHEMA
    assert raw == canonical_json(registry).encode("utf-8") + b"\n"
    assert registry_sha256() == hashlib.sha256(raw[:-1]).hexdigest()


def test_every_repo_local_registry_citation_still_says_what_the_claim_says():
    """The registry's evidence is line numbers, and line numbers rot.

    ``tools/check_registry_citations.py`` re-resolves every ``file:line`` the
    registry publishes: WRF paths must be DECLARED external (so a typo in a
    repo path cannot be silently downgraded to "must be somebody else's
    file"), and every citation into this worktree must carry an ANCHOR -- a
    substring the CLAIM is about -- that still appears in the cited lines.
    Existence is not enough; a drifted citation resolves to a line that
    exists and says something else.

    WHY THIS TEST EXISTS.  The checker was written, never wired to anything,
    and rotted: run for the first time on 2026-08-01 it reported 90 failures,
    ~30 of them ``RESOLVED`` rows for citations the registry had not carried
    for several releases, and its ``EXTERNAL`` table was missing
    ``phys/module_mp_thompson.F``, ``module_mp_thompson.F``,
    ``dyn_em/module_initialize_real.F`` and ``phys/module_physics_init.F``,
    so every one of the ten mp=28 citations failed. A checker nobody runs
    catches nothing, and its own tables become a second thing to be wrong.

    IT FOUND REAL DEFECTS, AND THEY WERE PUBLISHED RATHER THAN SUPPRESSED.
    Two YSU warnings cited ``kernels/ysu.cu`` lines that did not say what
    the warning says, and sat in the checker's ``DRIFTED`` table as OPEN
    rows, each carrying the anchor its claim is about so ``check`` would
    fail the moment the cited range contained it.  Both are fixed in the
    text: the ``kpbl < nz`` guard is cited at the line that carries it and
    is anchored in ``RESOLVED``, and the bare ``:1315`` the scanner had
    attributed to the kernel is spelled ``bl_ysu.F90:1315``, the WRF line
    it always meant.  ``DRIFTED`` is empty, and the set is pinned below so
    a new open defect cannot be added without its evidence and an owner.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import check_registry_citations as checker
    finally:
        sys.path.pop(0)

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    found = checker.citations(registry)
    assert len(found) >= 60, (
        "the registry publishes almost no file:line evidence any more; that "
        f"is either a regression or a scanner defect ({len(found)} found)")

    failures = checker.check(registry)
    assert failures == [], "\n  ".join([""] + failures)

    # The open defects are REPORTED, not hidden: every row names its owner so
    # a reader of this test knows where the fix lands.
    for line in checker.drifted_report():
        assert "OPEN (owner " in line, line
    assert set(checker.DRIFTED) == set(), (
        "the enumerated citation-defect list changed; a new entry needs the "
        "measured evidence and an owner, and a removed one needs its claim "
        f"re-verified: {sorted(checker.DRIFTED)}")

    # The two citations that were open defects are held to the lines their
    # claims are about: the guard by its anchor, the WRF line as a declared
    # external citation.
    guard = [citation for citation, row in checker.RESOLVED.items()
             if row == ("woof/core/kernels/ysu_480.cu", "kpbl < nz")]
    assert len(guard) == 1 and guard[0] in found, guard
    # bl_ysu.F90:1315 (the ctopo-absent drag arm) left the registry with the
    # warning that cited it: development branch put every YSU entry point on
    # WRF's default ctopo arm, :1308.
    assert "bl_ysu.F90:1315" not in found
    assert "bl_ysu.F90" in checker.EXTERNAL
    assert not [citation for citation in found
                if citation in ("kernels/ysu.cu:252", "kernels/ysu.cu:1315")]

    # And the mp=28 row's own citations are all covered, which is the thing
    # this wave rewrote.
    mp28 = [citation for citation, where in found.items()
            if any(MP28_OPTION_ID in path for path in where)]
    assert len(mp28) >= 10, sorted(mp28)
    for citation in mp28:
        path = citation.rpartition(":")[0]
        assert path in checker.EXTERNAL or citation in checker.RESOLVED, (
            f"the mp=28 option cites {citation}, which nothing resolves")


def test_a_carrier_of_a_named_routine_may_not_cite_that_file_by_line_too():
    """A line number kept beside the name is checked by nothing.

    ``NAMED_ROUTINES`` asks only that a declared carrier cite the routine
    SOMEWHERE, so a page carrying both ``path::routine`` and
    ``path:2293`` passes its carrier check while the number rots
    unnoticed -- which is the defect the name was adopted to remove,
    surviving on the page that adopted it.  Measured, not supposed:
    ``docs/da-nowcast-demo.md`` carried ``woof/core/dycore.py:2293``
    one paragraph above ``woof/core/dycore.py::apply_w_damping``,
    correct on the day and checked by nothing, and the next thing to
    move above that routine would have rotted it in silence exactly as
    :2178 rotted before it.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import check_registry_citations as checker
    finally:
        sys.path.pop(0)

    # The instrument first: a name alone is clean, and the rule finds
    # every number, so a test that reports nothing cannot pass by
    # matching nothing.
    assert checker.line_citations(
        "cite woof/core/dycore.py::apply_w_damping and nothing else",
        "woof/core/dycore.py") == ()
    assert checker.line_citations(
        "(`woof/core/dycore.py:2293`), also woof/core/dycore.py:7 here",
        "woof/core/dycore.py") == (2293, 7)

    assert checker.NAMED_ROUTINES, (
        "the named-routine table is empty, so this rule is checking nothing")
    failures = checker.named_routine_failures()
    assert failures == [], "; ".join(failures)


def test_mixed_thompson_outer_and_nssl_inner_plan_is_launchable():
    report = validate_physics_plan(_mixed_plan())
    assert report["schema"] == VALIDATION_SCHEMA
    assert report["launchable"] is True
    assert report["errors"] == []
    assert report["plan_id"] == "mixed-thompson-nssl-proof-v1"
    assert report["context"] == {
        "source_id": "hrrr",
        "runner_id": "tools.prepared_domain_tree_forecast",
        "topology_id": "one-way-nested-v1",
        "edges": [{"parent_domain_id": "d01", "child_domain_id": "d02"}],
    }
    assert [
        domain["settings"]["mp_physics"] for domain in report["resolved_domains"]
    ] == [8, 18]
    assert report["resolved_domains"][1]["settings"]["epssm"] == 0.1
    assert {item["requirement"]["id"] for item in report["asset_requirements"]} >= {
        "wrf-v4.6.1-classic-thompson-mp8-gfortran13-v1",
    }
    warning_codes = {warning["code"] for warning in report["warnings"]}
    assert "maturity" in warning_codes
    assert "component-warning" in warning_codes


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


def test_a_machine_short_of_a_table_set_still_gets_a_launchable_plan(
        tmp_path, monkeypatch):
    """The install question is REPORTED at plan review, never the verdict.

    THE STATE THIS REPRODUCES is a fresh install: ``freezeH2O.dat`` and
    ``qr_acr_qg_V4.dat`` are excluded from the wheel and arrive only
    through ``woof fetch-tables``, so classic Thompson's requirement
    cannot resolve until an operator stages them -- and classic Thompson
    is the DEFAULT template.  A verdict that read the machine therefore
    called the default plan unlaunchable on every fresh install, exited
    ``woof source --validate-physics-plan`` nonzero on a correct plan,
    and turned six tests in this file red in exactly the state the
    clean-venv release replay runs in.

    A plan is portable; an install is not.  ``launchable`` answers the
    plan, ``install_state`` answers the machine -- naming what is
    missing, where it was looked for and how to stage it -- and the door
    about to load the scheme is what refuses (walked by
    ``tests/test_authority_agreement.py::
    test_every_install_state_code_is_raised_by_a_run_door``).
    """

    empty = tmp_path / "no-tables"
    empty.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "no-staged-tables"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "no-staged-tables"))
    monkeypatch.setenv("WOOF_THOMPSON_TABLE_ROOT", str(empty))
    _wheel_shaped_companion(
        tmp_path, monkeypatch,
        physics_registry()["components"]["microphysics"]["options"][
            "thompson-mp8"][
            "asset_requirements"][0])

    report = validate_physics_plan(_single_plan(THOMPSON_TEMPLATE_ID))
    assert report["launchable"] is True, report["errors"]
    assert report["errors"] == []

    reported = [row for row in report["install_state"]
                if row["code"] == "asset-unresolved"]
    assert reported, report["install_state"]
    said = " ".join(row["message"] for row in reported)
    assert "woof fetch-tables" in said, said
    for row in report["install_state"]:
        assert row["code"] in INSTALL_STATE_CODES, row
        assert set(row) == {"code", "path", "message"}, row

    # And the CLI agrees: a correct plan on a machine short of a table
    # set exits 0 and carries the same report.
    plan_path = tmp_path / "thompson.json"
    plan_path.write_text(json.dumps(_single_plan(THOMPSON_TEMPLATE_ID)),
                         encoding="utf-8")
    assert main(["--validate-physics-plan", str(plan_path)]) == 0


def test_same_microphysics_nested_edge_is_launchable():
    report = validate_physics_plan(_uniform_tree())
    assert report["launchable"] is True
    assert report["errors"] == []


@pytest.mark.parametrize(
    ("mutator", "expected_code"),
    [
        (lambda plan: plan.update(edges=[]), "tree-edge-count"),
        (
            lambda plan: plan.update(
                edges=[
                    {"parent_domain_id": "d01", "child_domain_id": "d02"},
                    {"parent_domain_id": "d01", "child_domain_id": "d02"},
                ]
            ),
            "duplicate-edge",
        ),
    ],
)
def test_nested_topology_rejects_missing_and_duplicate_edges(mutator, expected_code):
    plan = _uniform_tree()
    mutator(plan)
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert expected_code in {error["code"] for error in report["errors"]}


def test_nested_topology_rejects_cycles_and_multiple_parents():
    plan = _uniform_tree()
    plan["domains"].append({"domain_id": "d03", "template_id": WSM6_TEMPLATE_ID})
    plan["edges"] = [
        {"parent_domain_id": "d01", "child_domain_id": "d02"},
        {"parent_domain_id": "d02", "child_domain_id": "d03"},
        {"parent_domain_id": "d03", "child_domain_id": "d01"},
    ]
    cyclic = validate_physics_plan(plan)
    assert cyclic["launchable"] is False
    assert "tree-cycle" in {error["code"] for error in cyclic["errors"]}

    plan["edges"] = [
        {"parent_domain_id": "d01", "child_domain_id": "d03"},
        {"parent_domain_id": "d02", "child_domain_id": "d03"},
    ]
    multi_parent = validate_physics_plan(plan)
    assert multi_parent["launchable"] is False
    assert "multiple-parents" in {
        error["code"] for error in multi_parent["errors"]
    }


def test_single_domain_topology_rejects_any_edge():
    plan = _single_plan()
    plan["edges"] = [{"parent_domain_id": "d01", "child_domain_id": "d01"}]
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert "single-domain-edges" in {error["code"] for error in report["errors"]}


@pytest.mark.parametrize("domain_id", ["", " ", "\t"])
def test_domain_ids_must_be_nonempty_after_trimming(domain_id):
    plan = _single_plan()
    plan["domains"][0]["domain_id"] = domain_id
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert "domain-id" in {error["code"] for error in report["errors"]}


def test_runner_routes_match_live_source_and_topology_contracts():
    # Nesting is a property of the topology, not of the source: the tree runner
    # reads every source's hierarchy document, so every source it declares is
    # routable here. Routable is not validated -- GFS in particular has an
    # open root/child humidity-convention question -- but that is a physics
    # judgement, not a routing one, and does not belong in this gate.
    tree = _uniform_tree()
    for source_id in ("hrrr", "era5", "gfs", "20crv3"):
        tree["context"]["source_id"] = source_id
        assert validate_physics_plan(tree)["launchable"] is True, source_id

    tree = _uniform_tree()
    tree["context"]["source_id"] = "not-a-registered-source"
    wrong_tree_source = validate_physics_plan(tree)
    assert wrong_tree_source["launchable"] is False
    assert "unsupported-source-route" in {
        error["code"] for error in wrong_tree_source["errors"]
    }

    tree = _uniform_tree()
    tree["context"]["topology_id"] = "offline-child-v1"
    offline = validate_physics_plan(tree)
    assert offline["launchable"] is False
    assert "unsupported-topology-route" in {
        error["code"] for error in offline["errors"]
    }

    hrrr_single = _single_plan()
    hrrr_single["context"].update(
        source_id="hrrr", runner_id="tools.hrrr_single_domain_benchmark"
    )
    assert validate_physics_plan(hrrr_single)["launchable"] is True
    hrrr_single["context"]["source_id"] = "era5"
    wrong_single_source = validate_physics_plan(hrrr_single)
    assert wrong_single_source["launchable"] is False
    assert "unsupported-source-route" in {
        error["code"] for error in wrong_single_source["errors"]
    }


@pytest.mark.parametrize("source_id", ("hrrr", "era5", "gfs", "20crv3"))
def test_real_source_mp_off_requires_explicit_moist_carrier(source_id):
    plan = _uniform_tree()
    plan["context"]["source_id"] = source_id
    for domain in plan["domains"]:
        domain["components"] = {"microphysics": "off"}

    # AUDIT R-005.  This was briefly a WARNING that said moist=true "was
    # resolved for you".  Nothing performs that resolution -- this
    # function reports, and the RunConfig a runner builds takes moist
    # from the microphysics-off option's own row -- so review called
    # launchable a plan woof/ingest/real.py refuses before step 0.  It
    # is an error again, at review, and it names the route's own
    # declaration of where the value lives instead of a door two of the
    # three routes do not have.
    unstated = validate_physics_plan(plan)
    assert unstated["launchable"] is False, unstated["warnings"]
    refusals = [
        error for error in unstated["errors"]
        if error["code"] == "real-source-mp-off-requires-explicit-moist"
    ]
    assert [error["path"] for error in refusals] == [
        "domains[0].parameters.moist",
        "domains[1].parameters.moist",
    ]
    site = physics_registry()["runner_routes"][
        "tools.prepared_domain_tree_forecast"]["moist_declaration_site"]
    assert all(site in error["message"] for error in refusals)
    # The named breakage is the loader's own sentence, not a paraphrase:
    # if that refusal is ever retired, this assertion goes with it.
    loader = (Path(woof.__file__).parent
              / "ingest" / "real.py").read_text(encoding="utf-8")
    assert "real initialization requires cfg.moist=True" in loader
    assert all("real initialization requires cfg.moist=True"
               in error["message"] for error in refusals)

    if source_id == "hrrr":
        # The one REAL per-source incompatibility the old error carried in
        # prose is a registry row now, refused at review by name rather
        # than inside woof/ingest/real.py an initialization later.  It
        # carries its OWN code, because it is the one refusal in the
        # constraint battery that a per-domain RunConfig cannot mirror --
        # a RunConfig has no source identity.
        native_errors = [
            error for error in unstated["errors"]
            if error["code"] == "component-source-refusal"
        ]
        assert native_errors, unstated["errors"]
        assert all("analyzed QC/QR/QI/QS/QG" in error["message"]
                   for error in native_errors)

    # An EXPLICIT dry column is refused with its own sentence, because a
    # user who wrote moist=false made a different mistake from one who
    # wrote nothing, and the remedy differs.
    for domain in plan["domains"]:
        domain["parameters"] = {"moist": False}
    dry = validate_physics_plan(plan)
    assert dry["launchable"] is False
    dry_errors = [error for error in dry["errors"]
                  if error["code"] == "real-source-mp-off-requires-moist"]
    assert dry_errors, dry["errors"]
    assert all(site in error["message"] for error in dry_errors)

    for domain in plan["domains"]:
        domain["parameters"] = {"moist": True}
    admitted = validate_physics_plan(plan)
    native = source_id == "hrrr"
    assert admitted["launchable"] is not native, admitted["errors"]
    assert all(domain["settings"]["moist"] is True
               for domain in admitted["resolved_domains"])


def test_a_route_refusal_is_the_route_s_own_reason_not_one_sentence_for_all():
    """A refusal may not name a breakage the route it fires on does not have.

    ``tools/build_registry.py`` writes the reason per route and per
    component, and nothing read either: both fixed-template routes
    printed "runs its registered templates unchanged".  That is the benchmark route's reason and only
    its reason.  The prepared single-domain route declares the tree
    route's component overrides (the 2026-07-31 ruling removed that
    runner's profile whitelist), so it was telling a user the route runs
    its templates unchanged while the same validator admitted a cumulus,
    microphysics, turbulence or PBL override on the same plan, and
    pointing them away from the reason the exclusion actually has.

    One refusal is delivered about ONE component, so the clause is the
    refused component's own: reading a route-wide reason instead told a
    user who named land_surface what is wrong with an analytic radiation
    scheme, in two sentences about a breakage that did not fire, and
    restated the admitted lists the sentence before it already names.
    """

    # What the route ADMITS as an override.  Asked of the route question
    # alone: a suite can still be refused for a pairing its own options
    # state, which is a different refusal with a different sentence.
    for component, option in (("cumulus", "grell-freitas"),
                              ("microphysics", "thompson-mp8"),
                              ("pbl", "mynn"),
                              ("turbulence", "smagorinsky-3d")):
        admits = _single_plan()
        admits["domains"][0]["components"] = {component: option}
        report = validate_physics_plan(admits)
        assert "component-override-route" not in {
            error["code"] for error in report["errors"]}, report["errors"]

    admitted = _single_plan()
    admitted["domains"][0]["components"] = {"land_surface": "noah-mp"}
    report = validate_physics_plan(admitted)
    assert not [error for error in report["errors"]
                if error["code"] == "component-override-route"], report["errors"]

    # A genuine option-specific radiation exclusion retains its explanation.
    other = _single_plan()
    other["domains"][0]["components"] = {"radiation": "analytic-clear-sky"}
    other_errors = [
        error for error in validate_physics_plan(other)["errors"]
        if error["code"] == "component-override-route"]
    assert other_errors, other
    for error in other_errors:
        message = error["message"]
        assert "carries no cloud, aerosol or gas optics" in message, message
        assert "expert_template_ids gating" not in message, message

    # The one route the immutability sentence describes still says it,
    # and says it as its OWN declared reason.
    benchmark = _single_plan()
    benchmark["context"].update(
        source_id="hrrr", runner_id="tools.hrrr_single_domain_benchmark")
    benchmark["domains"][0]["template_id"] = physics_registry()[
        "runner_routes"]["tools.hrrr_single_domain_benchmark"][
            "source_template_ids"]["hrrr"][0]
    benchmark["domains"][0]["components"] = {"microphysics": "thompson-mp8"}
    sealed = validate_physics_plan(benchmark)
    sealed_errors = [error for error in sealed["errors"]
                     if error["code"] == "component-override-route"]
    assert sealed_errors, sealed["errors"]
    assert all("only comparable against the immutable template"
               in error["message"] for error in sealed_errors)
    assert all("retire the comparison the route exists to publish"
               in error["message"] for error in sealed_errors)
    # A declared reason does not restate the subject the message just
    # named: "on runner 'X': benchmark route: its published ..." read as
    # two colons and two labels for one route.
    assert all("': benchmark route:" not in error["message"]
               for error in sealed_errors), sealed_errors




@pytest.mark.parametrize("runner_id,mode", [
    ("tools.prepared_single_domain_forecast", "single"),
    ("tools.prepared_domain_tree_forecast", "tree"),
])
@pytest.mark.parametrize("component,option", [
    ("pbl", "ysuu"), ("surface_layer", "typo-mm5"),
    ("land_surface", "typo-noah"), ("radiation", "typo-radiation"),
    ("not-a-component", "not-an-option"),
])
def test_unrecognized_selection_does_not_claim_a_route_restriction(
        runner_id, mode, component, option):
    plan = _single_plan() if mode == "single" else _uniform_tree()
    plan["context"]["runner_id"] = runner_id
    plan["domains"][0]["components"] = {component: option}
    report = validate_physics_plan(plan)
    path = "domains[0].components." + component
    issues = [error for error in report["errors"] if error["path"] == path]
    expected = "unknown-component" if component == "not-a-component" else "unknown-option"
    assert any(error["code"] == expected for error in issues), issues
    assert all(error["code"] != "component-override-route" for error in issues), issues
    assert report["launchable"] is False


def test_declared_route_fallback_is_generated_from_its_own_metadata():
    routes = physics_registry()["runner_routes"]
    checked = 0
    for route in routes.values():
        if not route.get("component_override_refusal_reasons"):
            continue
        general = route.get("component_override_refusal_reason", "")
        assert "whole-component override axes" in general
        for component in route["allowed_component_overrides"]:
            assert component in general
        checked += 1
    assert checked > 0, "no partially restricted route was tested"


@pytest.mark.parametrize("seed", range(12))
def test_future_route_and_component_rows_do_not_inherit_benchmark_text(seed):
    from woof.physics_registry import _fixed_template_override_refusal
    component = f"fixture_component_{seed}"
    option = f"fixture_option_{seed}"
    runner = f"fixture.runner_{seed}"
    route = {
        "mode": "fixed-template", "implemented": True,
        "allowed_component_overrides": [f"fixture_free_{seed}"],
        "allowed_component_options": {component: ["base"]},
    }
    message = _fixed_template_override_refusal(
        runner, {runner: route}, {}, (), "fixture", {component: "base"},
        component, option)
    assert "immutable template" not in message
    assert "registered templates unchanged" not in message
    assert f"fixture_free_{seed}" in message
    assert component in message


def test_a_deferred_per_domain_parameter_names_its_component_and_its_value():
    """The omission the route publishes is the sentence the user gets.

    ``runner_routes.<route>.deferred_parameter_keys`` maps every knob the
    per-domain loader accepts but the route leaves out of
    ``allowed_parameter_keys`` to the component it belongs to and the way
    to the value.  It was written and read by nothing, so a plan naming
    ``ra_physics`` per domain -- a knob ``woof.experiment``'s
    ``_DOMAIN_RUN_OVERRIDES`` accepts -- was refused with "runner route
    does not accept this per-domain setting" and nothing else, while the
    release surface said the route publishes the omission.
    """

    route = physics_registry()["runner_routes"][
        "tools.prepared_domain_tree_forecast"]
    deferred = route["deferred_parameter_keys"]
    assert "ra_physics" in deferred and deferred["ra_physics"]
    assert "ra_physics" not in route["allowed_parameter_keys"]

    plan = _uniform_tree()
    plan["domains"][1]["parameters"] = {"ra_physics": 0}
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    errors = [error for error in report["errors"]
              if error["code"] == "parameter-route"
              and error["path"].endswith(".parameters.ra_physics")]
    assert errors, report["errors"]
    for error in errors:
        assert deferred["ra_physics"] in error["message"]
        assert "every radiation option states ra_physics" in error["message"]
        assert "Select the radiation option" in error["message"]

    # A knob no route defers is still refused, and the sentence names the
    # declaration that lists what this route DOES take per domain.
    routes = physics_registry()["runner_routes"]
    single = routes["tools.prepared_single_domain_forecast"]
    assert "epssm" not in single.get("deferred_parameter_keys", {})
    assert single["allowed_parameter_keys"]
    plain = _single_plan()
    plain["domains"][0]["parameters"] = {"epssm": 0.1}
    other = [error for error in validate_physics_plan(plain)["errors"]
             if error["code"] == "parameter-route"]
    assert other
    assert all(
        "runner_routes.tools.prepared_single_domain_forecast"
        ".allowed_parameter_keys" in error["message"] for error in other)

    # The fixed benchmark now accepts four native controls. An
    # unrelated parameter must still name its allowed vocabulary.
    sealed_id = "tools.hrrr_single_domain_benchmark"
    assert routes[sealed_id]["allowed_parameter_keys"] == [
        "accum_mode", "aitken_mode", "coarse_mode", "shcu_physics"]
    sealed = _single_plan()
    sealed["context"].update(source_id="hrrr", runner_id=sealed_id)
    sealed["domains"][0]["template_id"] = routes[sealed_id][
        "source_template_ids"]["hrrr"][0]
    sealed["domains"][0]["parameters"] = {"epssm": 0.1}
    closed = [error for error in validate_physics_plan(sealed)["errors"]
              if error["code"] == "parameter-route"]
    assert closed
    assert all(f"runner_routes.{sealed_id}.allowed_parameter_keys"
               in error["message"] for error in closed)

    # And the fourth case, which a review read as an unreachable branch:
    # a plan that names NO runner route at all reaches this refusal, with
    # route_parameter_keys empty because the lookup found no route.  The
    # sentence it delivers is asserted here so the branch is not retired
    # as dead a second time.
    nameless = _single_plan()
    nameless["context"].pop("runner_id")
    nameless["domains"][0]["parameters"] = {"epssm": 0.1}
    routeless = [error for error in validate_physics_plan(nameless)["errors"]
                 if error["code"] == "parameter-route"]
    assert routeless, nameless
    assert all("this plan names no runner route" in error["message"]
               for error in routeless), routeless


def test_the_per_domain_routes_name_what_their_option_lists_exclude():
    # An actual optics limitation is distinct from throughput advice.
    for runner_id in ("tools.prepared_domain_tree_forecast",
                      "tools.prepared_single_domain_forecast"):
        plan = _uniform_tree() if "tree" in runner_id else _single_plan()
        plan["context"]["runner_id"] = runner_id
        plan["context"]["source_id"] = "gfs"
        for domain in plan["domains"]:
            domain["components"] = {"radiation": "analytic-clear-sky"}
        refused = validate_physics_plan(plan)
        assert refused["launchable"] is False
        errors = [error for error in refused["errors"]
                  if error["code"] == "component-override-route"]
        assert errors, refused["errors"]
        assert all("carries no cloud, aerosol or gas optics" in error["message"]
                   for error in errors), errors


def test_unnamed_tree_tuple_governance_uses_registry_reachability_only(
        capsys):
    from woof.physics_compat import (
        PhysicsCapabilityError,
        multi_domain_physics_selection,
    )

    normal = single_domain_runtime_switches(WSM6_TEMPLATE_ID)
    receipt = multi_domain_physics_selection({1: normal, 2: normal})
    assert {
        domain["governance"]["state"]
        for domain in receipt["domains"].values()
    } == {"registry-reachable"}
    assert receipt["acknowledgements"] == []
    assert receipt["acknowledgement_provenance"] == {}

    pbl_off = {**normal, "bl_pbl_physics": 0}
    pbl_off_receipt = multi_domain_physics_selection(
        {1: pbl_off, 2: pbl_off})
    assert {
        domain["governance"]["state"]
        for domain in pbl_off_receipt["domains"].values()
    } == {"registry-reachable"}
    assert pbl_off_receipt["acknowledgements"] == []

    outside = {
        **normal,
        "ra_lw_physics": 90,
        "ra_sw_physics": 90,
    }
    # Warn-not-block owns the SEVERITY of this site: a tuple outside the
    # registry's declared reachability is still individually implemented,
    # so it runs and says so -- one line per domain, naming the tuple, the
    # state and both published ways to acknowledge it.  What this test
    # guards is unchanged: that the governance is computed from registry
    # reachability alone, that it is REPORTED, and that the
    # acknowledgement is what flips it.
    unacked = multi_domain_physics_selection({1: outside, 2: outside})
    lines = [line for line in capsys.readouterr().err.splitlines()
             if line.startswith("warning:")]
    assert len(lines) == 2, lines
    message = "\n".join(lines)
    assert "d01 physics tuple" in message
    assert "d02 physics tuple" in message
    assert "ra_lw_physics=90" in message
    assert "ra_sw_physics=90" in message
    assert "outside-registry-declared-reachability" in message
    assert (
        '--ack expert-tuple-v1 or acknowledgements = ["expert-tuple-v1"]'
        in message
    )
    # Warned, and recorded as unacknowledged -- not silently blessed.
    assert {
        domain["governance"]["state"]
        for domain in unacked["domains"].values()
    } == {"outside-registry-declared-reachability"}
    assert not any(domain["governance"]["acknowledged"]
                   for domain in unacked["domains"].values())

    acknowledged = multi_domain_physics_selection(
        {1: outside, 2: outside},
        expert_acknowledgements=("expert-tuple-v1",),
    )
    assert {
        domain["governance"]["state"]
        for domain in acknowledged["domains"].values()
    } == {"outside-registry-declared-reachability"}
    assert all(domain["governance"]["acknowledged"]
               for domain in acknowledged["domains"].values())


def test_unnamed_tree_noah_tuple_needs_no_throughput_acknowledgement(capsys):
    from woof.physics_compat import multi_domain_physics_selection
    expert = single_domain_runtime_switches(NOAHMP_PROFILE_ID)
    receipt = multi_domain_physics_selection({1: expert, 2: expert})
    assert len(receipt["domains"]) == 2
    assert "noahmp-host-column-throughput-v1" not in capsys.readouterr().err


def test_registry_routes_drift_check_against_live_runner_capabilities():
    registry = physics_registry()
    routes = registry["runner_routes"]

    hrrr_capabilities = hrrr_runner_capabilities()
    hrrr_route = routes[hrrr_capabilities["runner"]]
    assert hrrr_route["source_ids"] == hrrr_capabilities["supported_sources"]
    assert (
        hrrr_route["source_template_ids"]["hrrr"]
        + hrrr_route["expert_template_ids"]["hrrr"]
    ) == hrrr_capabilities["physics_profile_ids"]

    single_capabilities = prepared_single_runner_capabilities()
    single_route = routes[single_capabilities["runner"]]
    assert single_route["source_ids"] == single_capabilities["supported_sources"]
    for source_id, source in single_capabilities["source_profiles"].items():
        routed = list(single_route["source_template_ids"][source_id])
        routed += expert_template_ids_for_source(single_route, source_id)
        assert routed == source["physics_profile_ids"]

    tree_capabilities = tree_runner_capabilities()
    tree_route = routes[tree_capabilities["runner"]]
    assert tree_route["source_ids"] == tree_capabilities["supported_sources"]
    assert tree_route["topology_ids"] == ["one-way-nested-v1"]


def test_every_composition_suite_reaches_a_door_that_can_resolve_it():
    """R-067's other half: DECLARED is not offered, and offered is not run.

    The pass that minted these six suites declared each of them on all
    three routes and proved them through ``validate_physics_plan``, which
    reads the registry -- so the proof said the registry agrees with
    itself.  The native benchmark RUNNER refused every one of them at its
    own door with ``unsupported native HRRR physics profile``, because
    that runner's replay tables are keyed by the profiles IT declares and
    a composition with no native WRF run behind it has no namelist
    contract to be replayed against.

    So this asks the DOORS, both ways: the prepared single-domain runner
    offers each suite and resolves it by name switch for switch, and the
    native benchmark runner neither offers it nor pretends it could.
    """

    import tools.hrrr_single_domain_benchmark as benchmark
    import tools.prepared_single_domain_forecast as single

    offered = single.runner_capabilities()
    benchmark_profiles = benchmark.runner_capabilities()["physics_profile_ids"]
    assert COMPOSITION_SUITE_PROFILE_IDS

    for profile in COMPOSITION_SUITE_PROFILE_IDS:
        sources = sorted(
            source_id
            for source_id, row in offered["source_profiles"].items()
            if profile in row["physics_profile_ids"])
        assert sources, f"{profile} is on no source of the prepared route"

        switches = single._profile_runtime_switches(sources[0], profile)
        assert switches["mp_physics"] is not None
        # Every switch the composition declares, resolved -- the same
        # inventory the door writes into the experiment it materializes.
        assert switches == single_domain_runtime_switches(profile)

        # OSAS included: 2a23840a9db took it off the native route, whose
        # runner has no namelist contract to replay it against.
        assert profile not in benchmark_profiles, (
            f"{profile} is offered by a runner that refuses it")
        with pytest.raises(ValueError, match="unsupported native HRRR"):
            benchmark._native_hrrr_runtime_switches(profile)


def test_a_route_refusal_is_reachable_and_fires_at_plan_review():
    """The refusals exist where a user meets them, and they refuse.

    The sentences that keep a template off a route were written in
    tools/build_registry.py and stayed there: a grep of the shipped
    registry for their text returned nothing, plan review returned
    launchable with a warning saying "the resolved runtime settings still
    apply", and the runner then refused at its own door with a bare
    ``unsupported native HRRR physics profile``.  So the breakage was
    never named and the way out was never offered, on a route the user
    had already paid to prepare.

    Both legs are checked here: the registry PUBLISHES each refusal, and
    validate_physics_plan refuses on it with the published sentence.
    """

    registry = physics_registry()
    routes = registry["runner_routes"]
    published = {
        route_id: route.get("refused_template_ids", {}) or {}
        for route_id, route in routes.items()}
    assert any(published.values()), (
        "no route publishes a refusal; the reasons are back in the builder")

    checked = 0
    for route_id, refusals in published.items():
        route = routes[route_id]
        if route.get("mode") != "fixed-template":
            continue
        source_id = route["source_ids"][0]
        for template_id, reason in refusals.items():
            assert template_id in registry["templates"], template_id
            # The way out, named: another door, or an acknowledgement.
            assert ("tools.prepared" in reason
                    or "route" in reason), (template_id, reason)
            plan = _single_plan()
            plan["context"].update(source_id=source_id, runner_id=route_id)
            plan["domains"][0]["template_id"] = template_id
            report = validate_physics_plan(plan)
            assert report["launchable"] is False, (route_id, template_id)
            refused = [error for error in report["errors"]
                       if error["code"] == "template-refused-on-route"]
            assert refused, report["errors"]
            assert all(reason in error["message"] for error in refused)
            # And NOT the evidence warning, which says the opposite.
            assert "template-route-evidence" not in {
                warning["code"] for warning in report["warnings"]}
            checked += 1
    assert checked >= 6, checked


def test_a_template_the_route_merely_does_not_declare_still_runs():
    """The reverse leg of the refusal above, so it cannot widen.

    A template with no evidence entry for a source is not a refusal: the
    resolved settings still apply and the run is launchable with a
    warning.  That distinction is the whole reason the refusal is keyed
    on a published row rather than on absence from the declaration.
    """

    registry = physics_registry()
    route = registry["runner_routes"]["tools.prepared_single_domain_forecast"]
    refused = set(route.get("refused_template_ids", {}) or {})
    source_id, declared = next(
        (source_id, ids)
        for source_id, ids in route["source_template_ids"].items()
        if ids)
    expert: set[str] = set()
    for ids in (route.get("expert_template_ids", {}) or {}).values():
        expert |= set(ids)
    undeclared = next(
        template_id for template_id in sorted(registry["templates"])
        if template_id not in declared and template_id not in refused
        and template_id not in expert)
    plan = _single_plan()
    plan["context"].update(source_id=source_id)
    plan["domains"][0]["template_id"] = undeclared
    report = validate_physics_plan(plan)
    assert "template-refused-on-route" not in {
        error["code"] for error in report["errors"]}
    assert "template-route-evidence" in {
        warning["code"] for warning in report["warnings"]}


_PREPARED_ROUTES = ("tools.prepared_domain_tree_forecast",
                    "tools.prepared_single_domain_forecast")


def test_both_prepared_routes_list_every_implemented_cumulus_scheme():
    """The per-domain cumulus list names every scheme the engine runs.

    New Tiedtke (cu_physics 16) is implemented, and both prepared routes
    published a cumulus list of off, Kain-Fritsch and Grell-Freitas only:
    the builder's list was typed by hand and was not updated when the
    scheme landed, so every reader of the declaration saw it as reachable
    through the Milbrandt-Yau suite alone.  The list is derived from the
    implemented options now, and naming the scheme still carries its own
    required settings.
    """

    registry = physics_registry()
    implemented = {
        option_id
        for option_id, option in registry["components"]["cumulus"][
            "options"].items()
        if option.get("implemented") is True}
    assert "new-tiedtke" in implemented
    for route_id in _PREPARED_ROUTES:
        listed = registry["runner_routes"][route_id][
            "allowed_component_options"]["cumulus"]
        assert "new-tiedtke" in listed, route_id
        assert set(listed) == implemented, route_id
        assert len(listed) == len(set(listed)), route_id

    plan = _single_plan(THOMPSON_TEMPLATE_ID)
    plan["domains"][0]["components"] = {"cumulus": "new-tiedtke"}
    report = validate_physics_plan(plan)
    assert report["launchable"] is True, report["errors"]
    settings = report["resolved_domains"][0]["settings"]
    assert settings["cu_physics"] == 16
    assert settings["cudt_minutes"] == 0.0


def test_the_aerosol_aware_suite_is_offered_on_the_prepared_single_domain_route():
    """The mp_physics=28 suite reaches the single-domain door it can run on.

    The prepared single-domain route refused it at plan review with
    "source_absent_microphysics has no arm for mp_physics=28".  That arm
    exists (nc, nr and ni start at exact zero), the real-data ingest
    seeds nwfa and nifa from the WIF monthly climatology, and a missing
    dataset is refused by name before the fetch with 'synthetic' offered
    as the way out.  So the refusal outlived its fix, and the benchmark
    route repeated the same false sentence where its real reason is that
    no native WRF run of this composition exists to replay.
    """

    from woof.physics_menu import WIZARD_PHYSICS_PROFILES
    from woof.physics_registry import template_ids_with_components

    registry = physics_registry()
    routes = registry["runner_routes"]
    # Selected by its whole qualified MYJ/ETA/Noah composition: a second
    # mp28 suite (the GSD MYNN 4.1 one) now exists beside it.
    (suite,) = template_ids_with_components(
        microphysics="thompson-aerosol-mp28", pbl="myj",
        surface_layer="eta-similarity", land_surface="noah")
    suites = template_ids_with_components(microphysics="thompson-aerosol-mp28")
    assert sorted(suites) == sorted([
        suite, "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"]), suites
    single = routes["tools.prepared_single_domain_forecast"]
    assert suite not in (single.get("refused_template_ids") or {})
    assert suite in single["source_template_ids"]["gfs"]

    plan = _single_plan(suite)
    plan["domains"][0]["parameters"] = {"mp28_aerosol_source": "synthetic"}
    report = validate_physics_plan(plan)
    assert report["launchable"] is True, report["errors"]
    assert report["resolved_domains"][0]["settings"]["mp_physics"] == 28

    # The single-domain door resolves a runtime product for it, so the
    # menus built from that door offer it.
    assert suite in SINGLE_DOMAIN_PHYSICS_PROFILES
    assert suite in WIZARD_PHYSICS_PROFILES
    assert single_domain_runtime_switches(suite)["mp_physics"] == 28

    # The benchmark keeps it off, for its own reason.
    reason = routes["tools.hrrr_single_domain_benchmark"][
        "refused_template_ids"][suite]
    assert "no native run of this composition exists" in reason
    for route in routes.values():
        for published in (route.get("refused_template_ids") or {}).values():
            assert "microphysics_cold_start" not in published


def test_fixed_template_runners_reject_all_overrides():
    # AUDIT R-021.  The blanket "fixed-template runner accepts only its
    # immutable template_id" is retired: it described no runner -- this
    # route's own runner had its profile whitelist removed by the
    # 2026-07-31 ruling and runs any engine-valid suite -- and the route
    # declaration now mirrors the tree route's.  What a route declares is
    # admitted, on every route, through the one question.
    plan = _single_plan()
    plan["domains"][0]["components"] = {"microphysics": "wsm6-mp6"}
    components = validate_physics_plan(plan)
    assert components["launchable"] is True, components["errors"]

    # The BENCHMARK route stays closed, and its refusal names what the
    # closure protects rather than restating itself.
    benchmark = _single_plan()
    benchmark["context"].update(
        source_id="hrrr", runner_id="tools.hrrr_single_domain_benchmark")
    benchmark["domains"][0]["template_id"] = physics_registry()[
        "runner_routes"]["tools.hrrr_single_domain_benchmark"][
            "source_template_ids"]["hrrr"][0]
    # A component the benchmark template does NOT already select, so the
    # request is a real override rather than a restatement.
    benchmark["domains"][0]["components"] = {"microphysics": "thompson-mp8"}
    refused = validate_physics_plan(benchmark)
    assert refused["launchable"] is False
    override_errors = [error for error in refused["errors"]
                       if error["code"] == "component-override-route"]
    assert override_errors, refused["errors"]
    assert all("only comparable against the immutable template"
               in error["message"] for error in override_errors)

    plan = _single_plan()
    plan["domains"][0]["parameters"] = {"epssm": 0.1}
    parameters = validate_physics_plan(plan)
    assert parameters["launchable"] is False
    assert "parameter-route" in {error["code"] for error in parameters["errors"]}

    # AUDIT R-059: the expert refusal names what the emptiness protects
    # and the way out, rather than restating that the route says no.  The
    # two routes get two sentences, because they stopped refusing the
    # same things when R-021 widened one of them: a shared "invalidates
    # the seal" line no longer separated what is refused from what is
    # allowed, since this route now ACCEPTS a component override.
    plan = _single_plan()
    plan["domains"][0]["expert_overrides"] = {"settings": {"epssm": 0.1}}
    expert = validate_physics_plan(plan)
    assert expert["launchable"] is False
    expert_errors = [error for error in expert["errors"]
                     if error["code"] == "expert-setting-route"]
    assert expert_errors, expert["errors"]
    assert all("no per-domain override table" in error["message"]
               and "COMPONENT choice is a different question"
               in error["message"] for error in expert_errors)

    benchmark = _single_plan()
    benchmark["context"].update(
        source_id="hrrr", runner_id="tools.hrrr_single_domain_benchmark")
    benchmark["domains"][0]["template_id"] = physics_registry()[
        "runner_routes"]["tools.hrrr_single_domain_benchmark"][
            "source_template_ids"]["hrrr"][0]
    benchmark["domains"][0]["expert_overrides"] = {"settings": {"epssm": 0.1}}
    benchmark_expert = validate_physics_plan(benchmark)
    assert benchmark_expert["launchable"] is False
    benchmark_errors = [error for error in benchmark_expert["errors"]
                        if error["code"] == "expert-setting-route"]
    assert benchmark_errors, benchmark_expert["errors"]
    assert all("varies NOTHING" in error["message"]
               for error in benchmark_errors)


@pytest.mark.parametrize(
    ("component", "option"),
    # What the tree route still refuses per domain, and why neither is the
    # contentless refusal audit R-022/R-023 retired. The analytic proxy
    # carries no cloud, aerosol or gas optics and is
    # not a forecast product.
    [("radiation", "analytic-clear-sky")],
)
def test_tree_route_rejects_currently_unsupported_component_variation(
        component, option):
    plan = _uniform_tree()
    plan["domains"][1]["components"] = {component: option}
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert "component-override-route" in {
        error["code"] for error in report["errors"]
    }


@pytest.mark.parametrize(
    ("component", "option"),
    [("pbl", "off"), ("surface_layer", "revised-mm5"),
     ("radiation", "off")],
)
def test_tree_route_admits_wrf_legal_harmless_component_variation(
        component, option):
    plan = _uniform_tree()
    plan["domains"][1]["components"] = {component: option}
    report = validate_physics_plan(plan)
    assert report["launchable"] is True, report["errors"]


def test_tree_route_admits_morrison_to_nssl_only_with_matrix_policy():
    plan = _uniform_tree(MORRISON_TEMPLATE_ID)
    plan["domains"][1]["components"] = {"microphysics": "nssl2-mp18"}
    plan["domains"][1]["parameters"] = {
        "nest_microphysics_transition": "mp8-to-mp18-mass-diagnosed-v1"
    }
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert "transition-required-setting" in {
        error["code"] for error in report["errors"]
    }

    plan["domains"][1]["parameters"] = {
        "nest_microphysics_transition": "mp-edge-mass-diagnosed-v1"
    }
    admitted = validate_physics_plan(plan)
    assert admitted["launchable"] is True
    assert "transition-maturity" in {
        warning["code"] for warning in admitted["warnings"]
    }


def test_registry_advertises_every_mixed_edge_accurately():
    """Every ordered pair of transported-moment schemes, one row each.

    The count is n*(n-1) over the nine schemes whose mixed edges the
    runtime resolver ports: mp 1/6/8/9/10/16/18/28/50 give 72.  The last
    four joined by ratification -- mp=50 with its rime-pair closure, mp=9
    with the scheme's own consistency-block closure, and mp=16 and mp=28
    with their entry closures -- and each joined with every partner at
    once, because the build resolves every published row through the
    resolver itself, so an over-claim fails the build instead of being
    contained by a skip.  Exactly one row is ratified.  Written out
    rather than derived from the rules under test, which would make the
    assertion vacuous.
    """
    rules = physics_registry()["transitions"][
        "microphysics-one-way-v1"]["cross_options"]
    pairs = {
        (rule["parent_option_id"], rule["child_option_id"])
        for rule in rules
    }
    assert len(rules) == len(pairs) == 132
    others = ("kessler-mp1", "wsm6-mp6", "thompson-mp8", "morrison-mp10",
              "nssl2-mp18", "milbrandt2mom-mp9", "wdm6-mp16",
              "thompson-aerosol-mp28", "lin-mp2", "wsm7-mp24", "sbu-ylin-mp13")
    p3_pairs = {pair for pair in pairs if "p3-mp50" in pair}
    assert p3_pairs == {
        *((other, "p3-mp50") for other in others),
        *(("p3-mp50", other) for other in others),
    }
    my2_pairs = {pair for pair in pairs if "milbrandt2mom-mp9" in pair}
    assert len(my2_pairs) == 22
    for option_id in ("wdm6-mp16", "thompson-aerosol-mp28", "lin-mp2", "wsm7-mp24", "sbu-ylin-mp13"):
        assert len({pair for pair in pairs if option_id in pair}) == 22
    ratified = [rule for rule in rules if rule["status"] == "ratified"]
    assert [(rule["parent_option_id"], rule["child_option_id"])
            for rule in ratified] == [("thompson-mp8", "nssl2-mp18")]
    experimental = [
        rule for rule in rules if rule["status"] == "experimental"
    ]
    assert len(experimental) == 131
    assert {
        rule["maturity"] for rule in experimental
    } == {"experimental-runtime"}


def test_data_driven_component_constraints_reject_engine_impossible_settings():
    plan = _uniform_tree()
    plan["domains"][1]["parameters"] = {"moist": False}
    dry_mp6 = validate_physics_plan(plan)
    assert dry_mp6["launchable"] is False
    assert "component-required-setting" in {
        error["code"] for error in dry_mp6["errors"]
    }

    # Noah at nine layers.  The code is component-admitted-setting, not
    # component-required-setting: soil geometry is the one setting a
    # scheme can admit MORE THAN ONE value for (RUC defines six and nine),
    # so it is stated as a set read from the schemes' own modules rather
    # than as a single pinned value.  The single-value kind could only say
    # one of RUC's two, which is how plan review came to refuse a
    # six-level column the loader admits and a forecast has run on.
    plan = _uniform_tree(MORRISON_TEMPLATE_ID)
    plan["domains"][1]["parameters"] = {"num_soil_layers": 9}
    wrong_soil = validate_physics_plan(plan)
    assert wrong_soil["launchable"] is False
    assert {error["code"] for error in wrong_soil["errors"]} >= {
        "parameter-route",
        "component-admitted-setting",
    }
    admitted = [error for error in wrong_soil["errors"]
                if error["code"] == "component-admitted-setting"]
    assert admitted and "LEVEL DEPTHS" in admitted[0]["message"]


def test_ruc_admits_both_soil_geometries_its_own_module_defines():
    """Plan review is not narrower than the run door on soil geometry.

    ``woof.config`` admits a six-level RUC column, ``ruc.cu`` sizes
    itself for it and a completed forecast has written a wrfout on it.
    Plan review refused it, twice -- a parameter enum of [4, 9] and a
    ``required_settings`` pin of 9 -- with a message naming no breakage.
    Both are now read from
    ``woof.core.ruc_contract.WRF_SUPPORTED_NUM_SOIL_LAYERS``.
    """

    from woof.config import LAND_SURFACE_SOIL_LAYERS
    from woof.physics_registry import physics_registry

    registry = physics_registry()
    # The global declaration covers every land-surface provider, while
    # the per-option constraints below continue to restrict each scheme.
    declared = sorted({int(count) for selector in LAND_SURFACE_SOIL_LAYERS
                       for count in LAND_SURFACE_SOIL_LAYERS[selector]})
    assert registry["parameters"]["num_soil_layers"]["enum"] == declared
    for option_id, selector in (("noah", 2), ("ruc-lsm", 3), ("noah-mp", 4)):
        constraints = (registry["components"]["land_surface"]["options"]
                       [option_id]["constraints"])
        assert "num_soil_layers" not in constraints.get(
            "required_settings", {})
        assert (constraints["admitted_setting_values"]["num_soil_layers"]
                == [int(count)
                    for count in LAND_SURFACE_SOIL_LAYERS[selector]])
        assert constraints["admitted_setting_values_reasons"][
            "num_soil_layers"]


def test_graph_policy_rejects_nonzero_spec_exp_on_nested_child():
    plan = _uniform_tree()
    plan["domains"][1]["parameters"] = {"spec_exp": 0.33}
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert "graph-setting-constraint" in {
        error["code"] for error in report["errors"]
    }


def test_huge_numeric_parameter_returns_validation_error_instead_of_crashing(
    tmp_path: Path, capsys
):
    plan = _uniform_tree()
    plan["domains"][1]["parameters"] = {"spec_exp": 10**10_000}
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert "parameter-value" in {error["code"] for error in report["errors"]}

    plan_path = tmp_path / "huge-number-plan.json"
    cli_plan = _uniform_tree()
    cli_plan["domains"][1]["parameters"] = {"spec_exp": "__HUGE_NUMBER__"}
    rendered = json.dumps(cli_plan).replace('"__HUGE_NUMBER__"', "1e9999")
    plan_path.write_text(rendered, encoding="utf-8")
    assert main(["--validate-physics-plan", str(plan_path)]) == EXIT_CONFIG
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["launchable"] is False
    assert cli_report["errors"]


def _profiles_off_the_prepared_single_domain_route() -> tuple[str, ...]:
    """Profiles no source declares on the prepared single-domain route.

    Some compositions are registered on the HRRR routes only -- the
    Kessler rule: no other source inherits evidence from an HRRR-bound
    run -- and the audit R-067 suites are registered on the routes their
    composition is valid for rather than on all of them.  Both are
    unreachable on the route the parametrized test below walks, and both
    carry the hrrr-context test instead.

    DERIVED, not listed (audit R-067).  A hand tuple here is a fourth
    scheme table: it went stale the moment a template was registered on a
    different route, and it reported that as a failure of the claim it
    happened to be checking.
    """

    route = physics_registry()["runner_routes"][
        "tools.prepared_single_domain_forecast"]
    offered = {
        template_id
        for key in ("source_template_ids", "expert_template_ids")
        for declared in (route.get(key, {}) or {}).values()
        for template_id in declared
    }
    return tuple(profile for profile in SINGLE_DOMAIN_PHYSICS_PROFILES
                 if profile not in offered)


_HRRR_ONLY_PROFILE_IDS = _profiles_off_the_prepared_single_domain_route()


@pytest.mark.parametrize(
    "template_id",
    tuple(
        profile for profile in SINGLE_DOMAIN_PHYSICS_PROFILES
        if profile not in _HRRR_ONLY_PROFILE_IDS),
)
def test_v2_templates_preserve_every_existing_v1_runtime_switch(template_id):
    report = validate_physics_plan(_single_plan(template_id))
    assert report["launchable"] is True
    resolved = report["resolved_domains"][0]["settings"]
    for key, value in single_domain_runtime_switches(template_id).items():
        assert resolved[key] == value


@pytest.mark.parametrize("template_id", _HRRR_ONLY_PROFILE_IDS)
def test_hrrr_only_templates_preserve_their_runtime_switches(template_id):
    plan = _single_plan(WSM6_TEMPLATE_ID)
    plan["context"].update(
        source_id="hrrr",
        runner_id="tools.hrrr_single_domain_benchmark",
    )
    plan["domains"][0]["template_id"] = template_id
    report = validate_physics_plan(plan)
    assert report["launchable"] is True, report["errors"]
    resolved = report["resolved_domains"][0]["settings"]
    for key, value in single_domain_runtime_switches(template_id).items():
        assert resolved[key] == value


@pytest.mark.parametrize(
    "template_id",
    (NSSL2_PROFILE_ID, NSSL2_LEGACY_RRTMG_PROFILE_ID),
)
def test_nssl2_radiation_profiles_have_distinct_exact_identities(template_id):
    from types import SimpleNamespace

    switches = single_domain_runtime_switches(template_id)
    assert identify_single_domain_profile(SimpleNamespace(**switches)) == (
        template_id)
    assert switches["ra_rrtmg_variant"] in {
        "rte-rrtmgp", "rrtmg_legacy"}


def test_implemented_unverified_maturity_warns_but_never_blocks():
    registry = physics_registry()
    option = registry["components"]["microphysics"]["options"]["wsm6-mp6"]
    option["maturity"] = "implemented-unverified"
    plan = _single_plan()
    plan["registry_sha256"] = registry_sha256(registry)
    report = validate_physics_plan(plan, registry=registry)
    assert report["launchable"] is True
    assert report["errors"] == []
    assert any(
        warning["code"] == "maturity"
        and "implemented-unverified" in warning["message"]
        for warning in report["warnings"]
    )


def test_opaque_future_maturity_warns_conservatively_without_blocking():
    registry = physics_registry()
    option = registry["components"]["microphysics"]["options"]["wsm6-mp6"]
    option["maturity"] = "candidate-from-upstream"
    plan = _single_plan()
    plan["registry_sha256"] = registry_sha256(registry)
    report = validate_physics_plan(plan, registry=registry)
    assert report["launchable"] is True
    assert any(
        warning["code"] == "maturity"
        and "candidate-from-upstream" in warning["message"]
        for warning in report["warnings"]
    )


def test_template_maturity_and_warning_are_preserved_for_20cr_profile():
    template_id = (
        "20crv3-wsm6-ysu-mm5-noah-kf-rte-rrtmgp-implemented-unverified-v1"
    )
    plan = _single_plan(template_id)
    plan["context"]["source_id"] = "20crv3"
    report = validate_physics_plan(plan)
    assert report["launchable"] is True
    assert {warning["code"] for warning in report["warnings"]} >= {
        "template-maturity",
        "template-warning",
    }


def test_mynn_component_dependencies_are_the_wrf_v461_cells():
    """WRF's 16-cell matrix, PLUS the one PBL that is outside it.

    The MYNN surface layer's dependency row transcribes WRF's isfc class
    check, and that check has no cell for SASE at all: bl_pbl_physics=900
    is a WOOF closure, outside the transcription's PBL axis, and asking
    it for a verdict raises rather than answering.  Excluding SASE was
    therefore an intersection of two tables, not a physical statement --
    and the physical statement runs the other way: SASE reads
    ust/hfx/qfx/wspd, MYNN_SURFACE_OUTPUTS publishes all four, and the
    MYNN surface result is allocated on sf_sfclay_physics=5 alone,
    independent of the PBL selector.  The pairing is unmeasured, which is
    maturity, and maturity warns here rather than blocking.
    """

    components = physics_registry()["components"]
    # "uw" on the same terms as SASE: the UW moist-turbulence PBL reads
    # only ust/hfx/qfx from its surface layer, WRF v4.7.1 places no isfc
    # law on CAMUWPBLSCHEME (module_physics_init.F:3825-3832), and the
    # scheme is outside the v4.6.1 transcription's PBL axis
    # (woof.wrf461_compatibility.AXIS_EXCLUSIONS).
    assert components["surface_layer"]["options"]["mynn"]["constraints"][
        "requires_components"
    ] == {"pbl": ["off", "mynn", "sase", "uw", "eeps", "keps"]}
    assert components["pbl"]["options"]["mynn"]["constraints"][
        "requires_components"
    ] == {
        "surface_layer": ["revised-mm5", "classic-mm5", "mynn"]}
    sase = components["pbl"]["options"]["sase"]["constraints"]
    assert sase["requires_components"]["surface_layer"] == [
        "revised-mm5", "classic-mm5", "mynn", "pleim-xiu"]
    # The cadence pin went with it: the driver runs SASE at bldt_seconds
    # and holds/recouples its tendencies across skipped calls, and
    # tests/test_sase_cadence*.py exercise 0.1 s and 5.0 s.
    assert "bldt" not in sase["required_settings"]


def test_mynn_is_implemented_and_warns_rather_than_blocking():
    """Both halves run, and both say what has not been verified.

    ``implemented: true`` is what puts a scheme in a user's picker, so the
    warnings carry what a user has to know before choosing it: that no
    woof/WRF trajectory comparison exists, and -- since development branch --
    that the CUDA scheme is bitwise WRF v4.6.1 on every oracle, with the
    test that holds it named; phim/phih run on the device.
    """
    components = physics_registry()["components"]
    for component in ("pbl", "surface_layer"):
        option = components[component]["options"]["mynn"]
        assert option["implemented"] is True, component
        assert option["maturity"] == "implemented-unverified", component
        assert option["warnings"], component
        assert any("UNVERIFIED against a WRF forecast" in warning
                   for warning in option["warnings"]), component
    pbl_warnings = " ".join(components["pbl"]["options"]["mynn"]["warnings"])
    assert "bitwise WRF v4.6.1 on every oracle" in pbl_warnings
    assert "tests/test_mynn_wrf461_exact_gpu.py" in pbl_warnings
    assert "evaluated on the device" in pbl_warnings


def test_noahmp_is_implemented_and_warns_rather_than_blocking():
    """It runs, and the warnings say exactly what has not been earned.

    ``implemented: true`` is what puts a scheme in a user's picker, so the
    warnings carry the measurements a user needs before choosing it: that no
    woof/WRF trajectory comparison exists, that the whole column runs on
    the DEVICE through the slab orchestration at a measured 0.202-0.227 s
    per 360,000-column call (2026-07-27, twice, one RTX 5090 -- the figure
    that retired the host-era "6.4 ms per land column" scaling blocker),
    that glacier columns run the dedicated NOAHMP_GLACIER port, that sea
    ice has no energy balance under the configurable xice_threshold, and
    that the WRF six-rate and carried-COSZEN seams are active.

    The pinned figures are the ones
    ``woof/core/noahmp_runtime.py`` ``NOAHMP_RUNTIME_RESTRICTIONS``
    ["column_solver_location"] records and
    ``docs/noahmp_device_column_report.md`` publishes; the previous
    revision of this docstring quoted the host-era figure long after the
    measurement moved, which is exactly how a quoted figure becomes false.
    ``tests/test_noahmp_runtime.py::test_the_column_cost_is_what_the_
    registry_says`` stays the always-on order-of-magnitude gate on the
    small-grid per-column cost.
    """
    option = physics_registry()["components"]["land_surface"]["options"][
        "noah-mp"]
    assert option["implemented"] is True
    assert option["maturity"] == "implemented-unverified"
    text = " ".join(option["warnings"])
    assert "UNVERIFIED against a WRF forecast" in text
    assert "runs on the DEVICE" in text
    # Pin the figures themselves, so the warning cannot drift away from
    # what the slab timing runs measured.
    assert "0.202-0.227 s" in text
    assert "7.3-8.2 wall seconds per simulated minute" in text
    assert "GLACIER columns run the dedicated NOAHMP_GLACIER port" in text
    assert "never to NOAHMP_SFLX" in text
    assert "no sea-ice surface energy balance" in text
    assert "xice_threshold" in text
    assert "SIX-RATE precipitation seam is active" in text
    assert "COSZEN is a radiation-driver carrier" in text
    assert "MEASURED INERT" in text
    # Every knob the option pins must be an implemented knob: an implemented
    # option may not pin one nothing reads.
    from woof.physics_registry import parameter_is_implemented

    declared = physics_registry()["parameters"]
    for name in option["parameters"]:
        assert parameter_is_implemented(declared[name]), name


def test_noahmp_is_admitted_with_mm5_or_the_coupled_mynn_suite():
    """The registry and runtime authorities expose the same pairings.

    "eta-similarity" joined this list with the MYJ port.  The list is a
    STRUCTURAL statement -- which surface layers write the exchange fields
    this LSM seam reads -- and the Eta layer writes all of them
    (UST/CHS/CHS2/CQS2/FLHC/FLQC plus the driver's BR); no Noah-MP runtime
    read is MM5-specific. PX supplies the same required exchange fields.
    Its evidence tier lives in the surface-layer
    option's own maturity, which is implemented-unverified, not here.
    """
    option = physics_registry()["components"]["land_surface"]["options"][
        "noah-mp"]
    assert option["constraints"]["requires_components"] == {
        "surface_layer": ["revised-mm5", "classic-mm5", "mynn",
                          "eta-similarity", "temf", "gfs", "qnse", "pleim-xiu"]}


def test_registry_refuses_the_same_mynn_surface_ysu_cell_as_wrf():
    """MYNN surface with YSU is the fatal half of WRF's mixed-pair law."""

    plan = _single_plan()
    plan["domains"][0]["components"] = {"surface_layer": "mynn"}
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert "component-dependency" in {error["code"]
                                      for error in report["errors"]}


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("plan_id", None, "plan-id"),
        ("plan_id", "", "plan-id"),
        ("registry_sha256", None, "registry-binding"),
        ("registry_sha256", "0" * 64, "stale-registry-binding"),
    ],
)
def test_missing_or_stale_plan_identity_binding_blocks(field, value, error_code):
    plan = _single_plan()
    if value is None:
        plan.pop(field)
    else:
        plan[field] = value
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert error_code in {error["code"] for error in report["errors"]}
    assert all(set(issue) == {"code", "path", "message"} for issue in report["errors"])


@pytest.mark.parametrize(
    ("component", "option"),
    [
        ("microphysics", "not-a-scheme"),
        # ("land_surface", "ruc-lsm") stood here until RUC was admitted, and
        # ("radiation", "wrf-rrtm-dudhia") until the WRF RRTM longwave port
        # landed.  "sase" is now the only registered-but-unimplemented
        # option left in the whole registry, so this path has exactly one
        # case to make it with; the assertion below is guarded so the case
        # cannot go on passing vacuously if that option is admitted too.
        ("microphysics", "sase"),
    ],
)
def test_unknown_and_registered_but_unimplemented_options_block(component, option):
    if option != "not-a-scheme":
        registry = physics_registry()
        assert registry["components"][component]["options"][option][
            "implemented"] is False, (
            f"{component}.{option} is implemented now; point this case at "
            "another registered-but-unimplemented option or the "
            "unimplemented-option error path loses its only witness")
    plan = _single_plan()
    plan["domains"][0]["components"] = {component: option}
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    expected = "unknown-option" if option == "not-a-scheme" else "unimplemented-option"
    assert expected in {error["code"] for error in report["errors"]}


def test_expert_settings_are_generic_but_selectors_cannot_claim_implementation():
    registry = physics_registry()
    route = registry["runner_routes"]["tools.prepared_single_domain_forecast"]
    route["mode"] = "experiment-per-domain"
    route["allowed_expert_setting_keys"] = ["future_tuning_coefficient"]
    plan = _single_plan()
    plan["domains"][0]["expert_overrides"] = {
        "settings": {"future_tuning_coefficient": 0.25}
    }
    plan["registry_sha256"] = registry_sha256(registry)
    report = validate_physics_plan(plan, registry=registry)
    assert report["launchable"] is True
    assert report["resolved_domains"][0]["settings"][
        "future_tuning_coefficient"
    ] == 0.25
    assert any(
        warning["code"] == "untyped-expert-setting"
        for warning in report["warnings"]
    )

    plan["domains"][0]["expert_overrides"] = {
        "selectors": {"future_physics_selector": 77}
    }
    blocked = validate_physics_plan(plan, registry=registry)
    assert blocked["launchable"] is False
    assert "unknown-expert-selector" in {
        error["code"] for error in blocked["errors"]
    }

    # MP7 now has a real implementation; use a future unknown value.
    plan["domains"][0]["expert_overrides"] = {"selectors": {"mp_physics": 777}}
    blocked_value = validate_physics_plan(plan, registry=registry)
    assert blocked_value["launchable"] is False
    assert "unknown-selector-combination" in {
        error["code"] for error in blocked_value["errors"]
    }


def test_registry_and_plan_hashes_are_deterministic_over_mapping_order():
    original = _mixed_plan()
    reordered = {
        "edges": original["edges"],
        "domains": original["domains"],
        "registry_sha256": original["registry_sha256"],
        "context": {
            "topology_id": "one-way-nested-v1",
            "runner_id": "tools.prepared_domain_tree_forecast",
            "source_id": "hrrr",
        },
        "plan_id": original["plan_id"],
        "schema": PLAN_SCHEMA,
    }
    first = validate_physics_plan(original)
    second = validate_physics_plan(reordered)
    assert first["plan_sha256"] == second["plan_sha256"]
    assert first["plan_sha256"] == canonical_sha256(original)
    assert first["registry_sha256"] == second["registry_sha256"]
    assert first["registry_sha256"] == registry_sha256()


def test_source_cli_registry_and_validator_are_compact_and_mutually_exclusive(
    tmp_path: Path, capsys
):
    assert main(["--show-physics-registry"]) == 0
    rendered = capsys.readouterr().out
    assert rendered == canonical_json(physics_registry()) + "\n"

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_mixed_plan()), encoding="utf-8")
    assert main(["--validate-physics-plan", str(plan_path)]) == 0
    report_text = capsys.readouterr().out
    assert "\n" not in report_text.rstrip("\n")
    assert json.loads(report_text)["launchable"] is True

    blocked = _single_plan()
    blocked["domains"][0]["components"] = {"microphysics": "sase"}
    plan_path.write_text(json.dumps(blocked), encoding="utf-8")
    assert main(["--validate-physics-plan", str(plan_path)]) == EXIT_CONFIG
    assert json.loads(capsys.readouterr().out)["launchable"] is False

    plan_path.write_text("{", encoding="utf-8")
    assert main(["--validate-physics-plan", str(plan_path)]) == EXIT_CONFIG
    unreadable = json.loads(capsys.readouterr().out)
    assert unreadable["plan_sha256"] is None
    assert unreadable["plan_id"] is None
    assert unreadable["context"] is None
    assert set(unreadable["errors"][0]) == {"code", "path", "message"}

    with pytest.raises(SystemExit) as raised:
        main(["--show-physics-registry", "--list-sources"])
    assert raised.value.code == 2


def test_source_cli_creates_exact_gpuwm_canonical_plan_without_replacement(
    tmp_path: Path, capsys
):
    plan = _mixed_plan()
    plan["domains"][1]["parameters"]["epssm"] = 1e-7
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    canonical_path = tmp_path / "canonical.json"

    assert main([
        "--validate-physics-plan",
        str(plan_path),
        "--canonical-physics-plan-output",
        str(canonical_path),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["launchable"] is True
    canonical_bytes = canonical_path.read_bytes()
    assert canonical_bytes == canonical_json(plan).encode("utf-8")
    assert b"1e-07" in canonical_bytes
    assert not canonical_bytes.endswith((b"\n", b"\r"))

    canonical_path.write_bytes(b"sentinel")
    assert main([
        "--validate-physics-plan",
        str(plan_path),
        "--canonical-physics-plan-output",
        str(canonical_path),
    ]) == EXIT_CONFIG
    refused = json.loads(capsys.readouterr().out)
    assert refused["launchable"] is False
    assert "canonical-plan-write" in {
        error["code"] for error in refused["errors"]
    }
    assert canonical_path.read_bytes() == b"sentinel"


def test_source_cli_canonical_output_follows_validity_and_flag_contract(
    tmp_path: Path, capsys
):
    plan_path = tmp_path / "plan.json"
    canonical_path = tmp_path / "canonical.json"
    blocked = _single_plan()
    blocked["domains"][0]["components"] = {"microphysics": "sase"}
    plan_path.write_text(json.dumps(blocked), encoding="utf-8")

    assert main([
        "--validate-physics-plan",
        str(plan_path),
        "--canonical-physics-plan-output",
        str(canonical_path),
    ]) == EXIT_CONFIG
    assert json.loads(capsys.readouterr().out)["launchable"] is False
    assert canonical_path.read_bytes() == canonical_json(blocked).encode("utf-8")

    canonical_path.unlink()
    plan_path.write_text("{", encoding="utf-8")
    assert main([
        "--validate-physics-plan",
        str(plan_path),
        "--canonical-physics-plan-output",
        str(canonical_path),
    ]) == EXIT_CONFIG
    assert json.loads(capsys.readouterr().out)["launchable"] is False
    assert not canonical_path.exists()

    with pytest.raises(SystemExit) as raised:
        main(["--canonical-physics-plan-output", str(canonical_path)])
    assert raised.value.code == 2


def test_physics_registry_module_is_stdlib_only_and_cli_does_not_import_cupy():
    script = r'''
from importlib.abc import MetaPathFinder
import sys

class RejectNumerics(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"numpy", "cupy"} or fullname.startswith(("numpy.", "cupy.")):
            raise AssertionError(f"unexpected numerical runtime import: {fullname}")
        return None

sys.meta_path.insert(0, RejectNumerics())
import woof.physics_registry
assert "numpy" not in sys.modules
assert "cupy" not in sys.modules
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    cli_script = r'''
from importlib.abc import MetaPathFinder
import runpy
import sys

class RejectCupy(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cupy" or fullname.startswith("cupy."):
            raise AssertionError(f"unexpected GPU runtime import: {fullname}")
        return None

sys.meta_path.insert(0, RejectCupy())
sys.argv = ["woof-wrf-init", "--show-physics-registry"]
runpy.run_module("woof.source_cli", run_name="__main__")
'''
    completed = subprocess.run(
        [sys.executable, "-c", cli_script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["schema"] == REGISTRY_SCHEMA


def test_real_windows_subprocess_emits_exact_lf_registry_and_validation_bytes(
    tmp_path: Path,
):
    registry_query = subprocess.run(
        [sys.executable, "-m", "woof.source_cli", "--show-physics-registry"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert registry_query.returncode == 0, registry_query.stderr.decode()
    assert registry_query.stdout == REGISTRY_PATH.read_bytes()
    assert registry_query.stdout.endswith(b"\n")
    assert not registry_query.stdout.endswith(b"\r\n")

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_single_plan()), encoding="utf-8")
    validation = subprocess.run(
        [
            sys.executable,
            "-m",
            "woof.source_cli",
            "--validate-physics-plan",
            str(plan_path),
            "--canonical-physics-plan-output",
            str(tmp_path / "canonical-plan.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert validation.returncode == 0, validation.stderr.decode()
    assert validation.stdout.endswith(b"\n")
    assert not validation.stdout.endswith(b"\r\n")
    assert json.loads(validation.stdout)["launchable"] is True
    assert (tmp_path / "canonical-plan.json").read_bytes() == canonical_json(
        _single_plan()
    ).encode("utf-8")


def test_registry_is_pinned_to_lf_for_source_archives(tmp_path):
    # A source archive carries the attributes, but intentionally has no .git.
    (tmp_path / ".gitattributes").write_bytes((ROOT / ".gitattributes").read_bytes())
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True,
                   capture_output=True, text=True)
    attributes = subprocess.run(
        ["git", "check-attr", "eol", "--", "woof/physics_registry_v2.json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert attributes.stdout.strip().endswith(": eol: lf")


# ==========================================================================
# Aerosol-aware Thompson (mp_physics=28)
#
# The registry entry for mp=28 is a claim about evidence, and the four things
# a reader most needs from it are the four things prose is worst at holding:
# what the label means, what the measurements actually were, what a user can
# and cannot select, and which WRF behaviours woof deliberately does not
# reproduce.  Each is pinned below against the shipped registry, and where a
# claim has a runtime counterpart the runtime is asked directly rather than
# quoted.
# ==========================================================================

MP28_OPTION_ID = "thompson-aerosol-mp28"

#: Every committed WRF v4.6.1 aerosol column fixture, derived from the files
#: on disk rather than typed out, so the registry's published evidence cannot
#: claim a fixture that does not exist or omit one that does.
#:
#: THE GLOB WIDENED ON 2026-08-01, and that is the point.  It used to be
#: ``aero-*-column.csv`` -- the nineteen scenarios MP28_PORT_SPEC.md names --
#: while ``tests/test_thompson_aerosol_adapter.py::_FIXTURES`` globbed
#: ``*-column.csv`` and drove TWENTY-TWO.  The three extra columns
#: (``wp08-freeze``, ``wp08-melt``, ``wp08-nusweep``, oracle ids 120-122 from
#: the same ``build_aero.sh`` run) were therefore outside the registry's
#: published partition entirely, and two of them MISS the gate.  Matching the
#: gate's own glob is what stops a fixture from failing in a place no
#: published number covers.
def _committed_aerosol_fixture_ids() -> set[str]:
    root = ROOT / "woof" / "data" / "thompson" / "oracle-aero"
    return {path.name[: -len("-column.csv")]
            for path in root.glob("*-column.csv")}


def _spec_aerosol_fixture_ids() -> set[str]:
    """The nineteen ``aero-*`` scenarios MP28_PORT_SPEC.md specifies."""
    return {name for name in _committed_aerosol_fixture_ids()
            if name.startswith("aero-")}


def _mp28_option() -> dict:
    return physics_registry()["components"]["microphysics"]["options"][
        MP28_OPTION_ID]


def _mp28_tree_plan() -> dict:
    """Both domains on mp=28, with the aerosol source said out loud.

    An mp=28 domain with EXTERNAL lateral boundaries needs WRF's monthly
    WIF climatology or a deliberate ``mp28_aerosol_source``.  A test plan
    cannot depend on a 225 MB dataset being staged on whatever machine
    runs it, so it takes the other way out, which is the way out the
    refusal names and the one a user without the dataset takes.  (A child
    would not need it -- nwfa/nifa cross a nest edge as coupled scalars --
    but this tree's root is where the plan carries ``specified``.)
    """
    plan = _uniform_tree()
    for domain in plan["domains"]:
        domain["components"] = {"microphysics": MP28_OPTION_ID}
        domain.setdefault("parameters", {})[
            "mp28_aerosol_source"] = "synthetic"
    return plan


def test_mp28_is_registered_at_the_maturity_its_evidence_earns():
    """implemented, warned, and no higher than implemented-unverified.

    ``implemented: true`` is what puts a scheme in a user's picker, so the
    label beside it has to be the one the evidence supports and not one step
    more.  mp=28 has 19 committed WRF column fixtures, per-kernel Fortran
    oracles and bit-exact device-helper probes -- and no forecast has ever
    been run with it, so ``validation-candidate`` (which requires a ratified
    reference comparison) and ``model-validated`` (which requires a matched
    multi-hour run with published decay tables) are both unavailable.  The
    vocabulary is this tree's own, published in
    ``docs/public/PHYSICS.md``; a rename lane rebasing it must move this pin
    with the rest.
    """
    option = _mp28_option()
    assert option["implemented"] is True
    assert option["maturity"] == "implemented-unverified"
    assert option["selectors"] == {"mp_physics": 28}
    assert option["label"] == "Thompson aerosol-aware / MP28"

    registry = physics_registry()
    policy = registry["warning_policy"]
    assert option["maturity"] in policy["warn_maturities"], (
        "the label must be one the warning policy warns on; a scheme with no "
        "forecast evidence must not resolve silently")
    assert option["maturity"] not in policy["nonwarning_maturities"]

    # Same shape as the model-validated sibling it is a port of: moist state
    # required, cq loading on, and nothing else pinned.
    assert option["parameters"] == {"moist": True, "moist_cq": True}
    assert option["constraints"]["required_settings"] == {"moist": True}

    # An implemented option may not pin a knob nothing honours.
    from woof.physics_registry import parameter_is_implemented

    declared = registry["parameters"]
    for name in option["parameters"]:
        assert parameter_is_implemented(declared[name]), name


def test_mp28_publishes_its_measured_column_evidence_not_a_forecast_claim():
    """The column gate is green; the registry says so in numbers, and no more.

    ``implemented-unverified`` says "column-oracle-measured".  Since the 2.8.6
    accumulator rework the column deck is also CLEAN but for the gate's
    declared divergence -- 21 of 22 at the flat gate with no allowance since
    2.8.8 moved rain collecting graupel onto the one slab WRF's tables hold,
    the 22nd (``aero-cold-overlap``) missing by that divergence alone -- and
    the label still does not move, because a clean single-call deck is not a
    forecast comparison.  The measurement
    lives on the option where a user reads it rather than only in a test
    file.  The published partition must cover every committed fixture
    exactly once: a fixture that quietly leaves the residual list without
    joining the clean list would otherwise vanish.

    This test was ``..._not_a_clean_claim`` and asserted a residual existed,
    which was the accurate claim until the rework closed the last four
    fixtures; it now asserts the opposite, with the same partition checks.
    The one residual it admits is the gate's own declared divergence
    (``_G3_DECLARED_DIVERGENCE``), read back from the gate rather than
    retyped, so a residual that is not declared there still fails here.
    """
    import importlib.util

    gate_spec = importlib.util.spec_from_file_location(
        "_mp28_adapter_partition",
        ROOT / "tests" / "test_thompson_aerosol_adapter.py")
    gate = importlib.util.module_from_spec(gate_spec)
    sys.modules.setdefault("_mp28_adapter_partition", gate)
    gate_spec.loader.exec_module(gate)
    declared = set(gate._G3_DECLARED_DIVERGENCE)

    evidence = _mp28_option()["extensions"]["column_oracle_evidence"]
    committed = _committed_aerosol_fixture_ids()
    spec = _spec_aerosol_fixture_ids()
    assert len(committed) == 22, sorted(committed)
    assert len(spec) == 19, sorted(spec)
    assert sorted(committed - spec) == [
        "wp08-freeze", "wp08-melt", "wp08-nusweep"], sorted(committed - spec)
    # BOTH counts, and the registry must publish both: the number the spec
    # names and the number the gate actually drives.  Publishing only the
    # first is how wp08-freeze and wp08-nusweep sat above the gate in no
    # published class for four waves.
    assert evidence["fixtures"] == len(committed)
    assert evidence["spec_fixtures"] == len(spec)
    assert evidence["gate_relative"] == 2.0e-6
    # A forecast comparison now exists -- docs/public/wrf-comparison/
    # mp28-matched-trajectory.md, an idealized doubly-periodic single-domain
    # run against unmodified WRF v4.6.1.  The anti-overclaim rule this
    # assertion has always enforced is unchanged, only sharpened: the entry
    # may exist, but it must carry its own FAILED gate and its own list of
    # what it does not establish.  A comparison published without those two
    # is exactly the overclaim the original None guarded against.
    forecast = evidence["forecast_trajectory_comparison"]
    assert isinstance(forecast, dict) and forecast, (
        "forecast_trajectory_comparison must either stay None or be a "
        "populated record; an empty one is a claim with no evidence")
    assert forecast["document"] == (
        "docs/public/wrf-comparison/mp28-matched-trajectory.md")
    assert forecast["declared_verdict"] == "HOLD", (
        "the pre-declared gate FAILED (V3) and the registry must say so; "
        "changing this to a pass without re-running the comparison is the "
        "overclaim this row exists to prevent")
    for key in ("not_nested_not_real_data", "control_result",
                "what_it_does_not_establish"):
        assert forecast[key].strip(), key
    # The comparison is idealized.  It must never be read as a real-data or
    # nested validation, and the record has to say that in its own words.
    assert "nested" in forecast["what_it_does_not_establish"]
    assert forecast["kind"].startswith("idealized")

    clean = set(evidence["clean_fixtures"])
    residual = set(evidence["residual_fixtures"])
    carved = set(evidence["carved_out_bound"])
    # The near-cancellation bound is the THIRD publication class and has to
    # be counted as one.  It was redundant while aero-reduces-to-classic was
    # also in carved_out_bound; the 1.4.1 merge retired that entry, and
    # without this term the union drops the one fixture the port documents
    # most heavily and reports it as unpublished.
    near = set(evidence["near_cancellation_bound"]["fixtures"])
    assert clean | residual | carved | near == committed, (
        "the published evidence and the committed fixture deck disagree: "
        f"{sorted(committed ^ (clean | residual | carved | near))}")
    assert not (clean & residual) and not (clean & carved) \
        and not (residual & carved)

    # Every residual is published as a number rather than as an adjective.
    # Since the 2.8.6 accumulator rework the only one is the gate's declared
    # divergence, and the clean list is the rest of the deck.
    assert residual == declared, evidence["residual_fixtures"]
    assert clean == committed - declared
    for name, fields in evidence["residual_fixtures"].items():
        assert fields, name
        for field, value in fields.items():
            assert isinstance(value, float) and value > evidence[
                "gate_relative"], (name, field, value)

    # ...and the allowanced fixture is disclosed as such rather than folded
    # in with the clean set, because it is a departure somebody chose.  It is
    # now disclosed under near_cancellation_bound alone: the 1.4.1 merge
    # retired the relative carve-out it also used to sit under, so carved is
    # EMPTY and the assertion moves to `near` rather than being deleted.
    assert carved == set()
    # Retired by the 2.8.6 accumulator rework: level 6 of
    # aero-reduces-to-classic is bit-exact against WRF in qr and nr.
    assert near == set()
    # ONE FIELD, not two.  This literal used to read ``{"qr", "nr_per_kg"}``
    # and stayed green for a wave after WP-13a's level-wise sedimentation
    # density took that fixture's ``qr`` to 1.788e-07 -- inside the FLAT
    # 2.0e-06 gate -- and deleted it from ``_END_TO_END_BOUNDS``.  A registry
    # that publishes a carve-out on a field the gate no longer carves out
    # overstates the port's own relaxation by a whole quantity, which is the
    # opposite of the failure this row exists to prevent, so the set is
    # asserted here as a second opinion and read back from the gate itself by
    # ``test_mp28_evidence_matches_the_bound_the_adapter_gate_actually_
    # applies``.
    # ONE LEVEL, not two.  Same second-opinion role the field-set assertion
    # above it used to play for the retired relative bound.
    assert evidence["near_cancellation_bound"]["fixtures"] == {}

    # THE TWO COUNTS, PUBLISHED SEPARATELY.  ``clean_fixtures`` is the
    # UNEXCEPTIONED list -- nothing held out, no bounds dict -- and the gated
    # count is that plus exactly the carved-out fixtures.  Conflating them is
    # how a port claims a clean number it did not earn, so both are numbers
    # on the option and both are derived from the same partition here.
    assert evidence["clean_unexceptioned"] == len(clean)
    assert evidence["clean_as_gated"] == len(clean) + len(carved) + len(near)
    assert evidence["clean_unexceptioned"] == (
        evidence["fixtures"] - len(declared)), (
        "the gate clears the whole deck but its declared divergence with "
        "nothing held out, so the published unexceptioned count must be the "
        "deck size less that divergence")
    # ...and the clean deck did NOT raise the label, which is the claim this
    # row must never overstate.
    assert _mp28_option()["maturity"] == "implemented-unverified"

    # EVERY departure from the flat gate is named, and each one says which
    # direction it moved.  An unpublished allowance is indistinguishable
    # from a hidden one.
    allowances = evidence["allowances"]
    for allowance in allowances:
        # ``carved | near`` rather than ``carved`` alone: an allowance may be
        # a METRIC change published under near_cancellation_bound as well as
        # a relative bound published under carved_out_bound, and after the
        # 1.4.1 merge the port's only surviving allowance is the former.
        assert set(allowance["fixtures"]) <= (carved | near), (
            "an allowance that applies to a fixture the registry does not "
            f"publish as carved out: {allowance}")
        assert allowance["direction"], allowance
    # TWO, not three.  ``_REFL_DB_BOUNDS`` was RETIRED by WP-13a, not
    # widened: the residual it covered went 5.283e-04 dB -> 3.242e-05 dB,
    # inside the flat 2.0e-4 dB gate, and the gate's dict is now empty.  A
    # retired allowance may not keep being published as a live one -- that
    # reads as a relaxation the port no longer takes -- so it moves to
    # ``retired_allowances``, which must still name its gate constant and
    # say what happened to it.
    # The literals below are a SECOND OPINION on the gate's own
    # _G3_ALLOWANCES, which test_mp28_evidence_publishes_the_allowances_the_
    # gate_actually_has reads back directly.  Both moved at the 1.4.1 merge,
    # in the retiring direction only: _END_TO_END_BOUNDS went live -> retired
    # when the inherited mp=8 sedimentation reconciliations took the residual
    # it covered from 5.700e-06 to 4.146e-07.
    # NONE live: the 2.8.6 accumulator rework retired the last one,
    # _NEAR_CANCELLATION_LEVELS, when level 6 went bit-exact.
    assert allowances == [], allowances
    retired = evidence["retired_allowances"]
    assert {a["gate_constant"] for a in retired} == {
        "_END_TO_END_BOUNDS", "_NEAR_CANCELLATION_LEVELS",
        "_REFL_DB_BOUNDS"}, retired
    for allowance in retired:
        assert allowance["is"] is None and allowance["direction"], allowance
    assert not ({a["gate_constant"] for a in retired}
                & {a["gate_constant"] for a in allowances}), (
        "an allowance cannot be both live and retired")

    # The gate compares 23 quantities, not the 16 the residual table is
    # published in, and the registry must say so or a reader will read the
    # smaller number as the whole comparison.
    assert evidence["compared_quantities"] == 23
    assert sum(evidence["compared_quantities_breakdown"].values()) == 23
    assert evidence["compared_quantities"] > evidence["compared_fields"]
    assert evidence["gate_reflectivity_db"] == 2.0e-4


def test_mp28_evidence_matches_the_bound_the_adapter_gate_actually_applies():
    """Bind the published numbers to the gate that produced them.

    The registry quotes a 2e-6 relative gate and one carved-out fixture.  Both
    are decisions made in ``tests/test_thompson_aerosol_adapter.py``, so they
    are read back from it: if the port ever widens that default bound or adds
    a second carve-out, the registry's published evidence stops describing the
    measurement and this fails instead of drifting.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_mp28_adapter_gate",
        ROOT / "tests" / "test_thompson_aerosol_adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_mp28_adapter_gate", module)
    spec.loader.exec_module(module)

    evidence = _mp28_option()["extensions"]["column_oracle_evidence"]
    assert evidence["gate_relative"] == module._END_TO_END_DEFAULT_BOUND
    assert evidence["compared_fields"] == len(module._END_TO_END_FIELDS) + 1, (
        "the registry publishes a field count the gate does not compare; the "
        "+1 is rainnc_mm, which the gate carries beside the column fields")
    assert set(evidence["carved_out_bound"]) == set(
        module._END_TO_END_BOUNDS), (
        "the registry publishes a different set of carved-out fixtures than "
        "the gate applies")
    for name, fields in module._END_TO_END_BOUNDS.items():
        published = evidence["carved_out_bound"][name]
        assert set(published) == set(fields), name
        for field, bound in fields.items():
            assert published[field] <= bound, (
                f"{name}.{field}: the registry publishes {published[field]!r} "
                f"as the measured residual, which cannot exceed the gate's "
                f"own carved-out bound {bound!r}")


def test_mp28_evidence_publishes_the_allowances_the_gate_actually_has():
    """The published allowance LIST is the gate's own, name for name.

    ``test_mp28_publishes_its_measured_column_residuals_not_a_clean_claim``
    asserts the set of gate constants as a literal, which is a second opinion
    and is meant to be one.  This is the first opinion: the list is read back
    out of ``tests/test_thompson_aerosol_adapter.py::_G3_ALLOWANCES``, the
    object that enumerates every departure from the flat gate, so adding an
    allowance to the gate without publishing it -- or leaving a RETIRED one
    published as live -- fails here rather than drifting for a wave.

    BEFORE THIS TEST: ``_REFL_DB_BOUNDS`` was retired by WP-13a (the residual
    it covered went 5.283e-04 dB -> 3.242e-05 dB, inside the flat 2.0e-4 dB
    gate, and the dict was emptied) and the registry kept publishing it as
    one of "the port's three named allowances" -- a relaxation the port does
    not take, advertised as though it did.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_mp28_adapter_allowances",
        ROOT / "tests" / "test_thompson_aerosol_adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_mp28_adapter_allowances", module)
    spec.loader.exec_module(module)

    evidence = _mp28_option()["extensions"]["column_oracle_evidence"]
    live = {a["gate_constant"] for a in evidence["allowances"]}
    retired = {a["gate_constant"] for a in evidence["retired_allowances"]}

    assert live == {constant for _name, constant, _fixtures, _why
                    in module._G3_ALLOWANCES}, (
        "the registry publishes a different set of live allowances than the "
        f"gate's _G3_ALLOWANCES applies: registry {sorted(live)}, gate "
        f"{sorted({c for _n, c, _f, _w in module._G3_ALLOWANCES})}")

    # A constant published as RETIRED must actually be inert in the gate.
    # ``_REFL_DB_BOUNDS`` is kept as an empty dict rather than deleted so
    # ``_g3_bound`` keeps one code path; empty is the proof it buys nothing.
    for constant in retired:
        assert getattr(module, constant) == {}, (
            f"{constant} is published as retired but the gate still applies "
            f"it: {getattr(module, constant)!r}")

    # ...and the fixtures each live allowance names are the gate's own.
    by_constant = {constant: set(fixtures)
                   for _name, constant, fixtures, _why
                   in module._G3_ALLOWANCES}
    for allowance in evidence["allowances"]:
        assert set(allowance["fixtures"]) == by_constant[
            allowance["gate_constant"]], allowance


def test_mp28_published_clean_set_is_the_gates_unexceptioned_clean_set():
    """``clean_fixtures`` is the gate's flat-gate clean list, not a summary.

    The registry's clean list is the strongest claim on the row -- it says
    these columns agree with unmodified WRF on all 23 compared quantities
    with NOTHING held out -- so it is bound to the gate's own pinned
    ``_G3_UNEXCEPTIONED_CLEAN``, which
    ``tests/test_thompson_aerosol_adapter.py`` re-measures on the device.
    The two counts the option publishes are recomputed from it here as well,
    so a stale count and a stale list cannot cover for each other.

    BEFORE THIS TEST: the registry published 15 clean fixtures and a "15 of
    22 / 16 of 22" pair of counts while the gate measured 17 and 18, i.e. the
    row UNDERSTATED the port by two whole columns; the same stale set listed
    ``aero-drop-evap`` and ``aero-ice-demott-idxin`` as residual fixtures
    carrying rainnc 5.165e-04 and 1.279e-04 that now measure exactly 0.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_mp28_adapter_clean",
        ROOT / "tests" / "test_thompson_aerosol_adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_mp28_adapter_clean", module)
    spec.loader.exec_module(module)

    evidence = _mp28_option()["extensions"]["column_oracle_evidence"]
    published = set(evidence["clean_fixtures"])
    measured = set(module._G3_UNEXCEPTIONED_CLEAN)
    assert published == measured, (
        "the registry's clean_fixtures is not the gate's "
        f"_G3_UNEXCEPTIONED_CLEAN: only in registry "
        f"{sorted(published - measured)}, only in gate "
        f"{sorted(measured - published)}")

    assert set(evidence["residual_fixtures"]) == set(module._G3_RESIDUALS), (
        "the registry's residual_fixtures is not the gate's _G3_RESIDUALS: "
        f"registry {sorted(evidence['residual_fixtures'])}, gate "
        f"{sorted(module._G3_RESIDUALS)}")
    for name, fields in module._G3_RESIDUALS.items():
        assert set(evidence["residual_fixtures"][name]) == set(fields), name
        for field, value in fields.items():
            assert (f"{evidence['residual_fixtures'][name][field]:.3e}"
                    == f"{value:.3e}"), (name, field)

    assert evidence["clean_unexceptioned"] == len(measured)
    assert evidence["clean_as_gated"] == len(module._G3_GATED_CLEAN)
    # The counts are quoted in prose on the same row; the prose must not be
    # able to say something the numbers do not.
    note = evidence["clean_counts_note"]
    assert f"{len(measured)} of {len(module._FIXTURES)}" in note, note
    assert (f"{len(module._G3_GATED_CLEAN)} of {len(module._FIXTURES)}"
            in note), note


def test_every_gate_the_mp28_row_cites_exists_and_is_a_test():
    """A cited gate that does not exist is a fabricated receipt.

    The option's evidence is now a set of pointers -- the G3 column gate, the
    G4 self-forecast gate, the aerosol-initialisation cost measurement and
    its pinned gap -- and a reader is expected to be able to run each of
    them.  Every ``path::name`` the row publishes is resolved here against
    the file on disk, so a rename that leaves the registry behind fails
    instead of turning the evidence into a claim about a test nobody can
    find.
    """
    option = _mp28_option()
    evidence = option["extensions"]["column_oracle_evidence"]
    initialisation = option["extensions"]["aerosol_initialisation"]

    cited = [
        evidence["test"],
        evidence["self_forecast_gate"]["test"],
        evidence["self_forecast_gate"]["longest_integration"]["test"],
        initialisation["measured_forecast_sensitivity"]["test"],
        initialisation["call_site_pin"],
        initialisation["installed_state_pin"],
        initialisation["gpuwm_implementation_evidence"].split(" --")[0],
    ]
    missing = []
    for citation in cited:
        path, _, name = citation.partition("::")
        source = ROOT / path
        if not source.is_file():
            missing.append(f"{path} does not exist")
            continue
        if f"def {name}(" not in source.read_text(encoding="utf-8"):
            missing.append(f"{path} has no test named {name}")
    assert missing == [], missing


def test_mp28_publishes_the_near_cancellation_relaxation_too():
    """The gate has TWO relaxations; the registry must publish both.

    ``carved_out_bound`` was the only one on the option, but
    ``tests/test_thompson_aerosol_adapter.py`` also replaces the relative
    metric with an absolute one -- 32 ulps of the ENTRY value -- at one level
    of one fixture, where a 10 s step evaporates 99.958 % of the rain and the
    survivor is the difference of two nearly equal float32 numbers.  An
    unpublished relaxation is indistinguishable from a hidden one, so it is
    read back from the gate's own constants here: widen the ulp allowance or
    hold out another level and the published evidence stops describing the
    measurement.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_mp28_adapter_gate_ulp",
        ROOT / "tests" / "test_thompson_aerosol_adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_mp28_adapter_gate_ulp", module)
    spec.loader.exec_module(module)

    evidence = _mp28_option()["extensions"]["column_oracle_evidence"]
    published = evidence["near_cancellation_bound"]

    assert published["ulps_of_entry_value"] == module._NEAR_CANCELLATION_ULPS
    assert {name: tuple(levels)
            for name, levels in published["fixtures"].items()} == {
        name: tuple(levels)
        for name, levels in module._NEAR_CANCELLATION_LEVELS.items()}, (
        "the registry publishes a different set of near-cancellation levels "
        "than the gate holds out")
    # Every held-out fixture must be one the registry already classifies, and
    # the measured ulp figures must sit under the allowance they are quoted
    # against.
    classified = (set(evidence["clean_fixtures"])
                  | set(evidence["residual_fixtures"])
                  | set(evidence["carved_out_bound"])
                  | set(published["fixtures"]))
    for name, measured in published["measured"].items():
        assert name in classified, name
        for field, value in measured.items():
            assert value < published["ulps_of_entry_value"], (name, field)


def test_mp28_has_its_own_suite_and_is_still_no_default():
    """Selectable as a named suite, and in nobody's default.

    AUDIT R-067.  The scheme used to be reachable only as a hand-written
    per-domain override -- implemented, and offered by no named suite --
    which is the ship-only-what-users-can-reach rule failing quietly.  It
    has a template now, so what this guard holds is the half a
    reachability recomputation cannot express as an intention: that
    the generic suite and explicit source-version suite select it, that
    their routes can initialize them, and that the shipped default is
    unchanged.
    """
    from woof.physics_registry import (DEFAULT_TEMPLATE_ID,
                                        THOMPSON_KF_TEMPLATE_ID)

    registry = physics_registry()
    assert _mp28_option()["reachability"] == {"state": "template"}
    assert "blocker" not in _mp28_option()["reachability"], (
        "a reachable option carrying a blocker is a self-contradiction")

    selecting = [
        template_id for template_id, template in registry["templates"].items()
        if template["components"]["microphysics"] == MP28_OPTION_ID
    ]
    generic_suite = "thompson-aerosol-mp28-myj-eta-noah-rte-rrtmgp-v1"
    source_suite = "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"
    # These two compositions select mp28, and neither is the default.
    assert sorted(selecting) == [generic_suite, source_suite], selecting
    assert DEFAULT_TEMPLATE_ID not in selecting
    assert DEFAULT_TEMPLATE_ID == THOMPSON_KF_TEMPLATE_ID
    assert registry["templates"][DEFAULT_TEMPLATE_ID]["components"][
        "microphysics"] == "thompson-mp8"

    # The suite is declared on every route whose runner builds its own
    # initialization, which since the mp_physics=28 cold-start arm landed
    # (audit R-044) is both prepared routes.  The native benchmark keeps
    # it off for its own named reason: it replays a native WRF run and no
    # native run of this composition exists.
    for route_id, route in registry["runner_routes"].items():
        declared = {
            template_id
            for key in ("source_template_ids", "expert_template_ids")
            for ids in (route.get(key, {}) or {}).values()
            for template_id in ids
        }
        assert (generic_suite in declared) == (
            route_id != "tools.hrrr_single_domain_benchmark"), route_id
    source_route = registry["runner_routes"][
        "tools.prepared_single_domain_forecast"]
    assert any(source_suite in ids for ids in
               source_route["source_template_ids"].values())
    assert source_suite not in source_route["source_template_ids"]["gfs"]

    # It really is selectable, both as its own suite and as the override.
    report = validate_physics_plan(_mp28_tree_plan())
    assert report["launchable"] is True, report["errors"]
    assert [domain["settings"]["mp_physics"]
            for domain in report["resolved_domains"]] == [28, 28]


def test_mp28_plan_warns_at_every_deviation_rather_than_blocking():
    """A user who selects mp=28 is told what has not been earned.

    ``maturity_never_blocks`` is registry policy, so the only protection a
    user has is that the warnings actually say the things.  Each phrase below
    is a distinct deviation the port committed to publishing, and a warning
    list that loses one silently would otherwise still pass every structural
    gate.
    """
    report = validate_physics_plan(_mp28_tree_plan())
    assert report["launchable"] is True, report["errors"]
    codes = {warning["code"] for warning in report["warnings"]}
    assert {"maturity", "component-warning"} <= codes

    emitted = {
        warning["message"] for warning in report["warnings"]
        if warning["code"] == "component-warning"
        and warning["path"].endswith(".components.microphysics")
    }
    # Every option warning must reach plan review, not merely one phrase
    # accidentally supplied by another component's warning.
    assert set(_mp28_option()["warnings"]) <= emitted
    text = " ".join(emitted)
    for phrase in (
        "No matched REAL-DATA or NESTED WRF trajectory",
        "THE COLUMN EVIDENCE IS CLEAN BUT FOR ONE DECLARED DIVERGENCE; "
        "THE FORECAST EVIDENCE IS NOT",
        "AEROSOL INPUT LIMITS",
        "woof/core/physics.py::initialize_physics",
        "HISTORICAL SYNTHETIC-PROFILE SENSITIVITY",
        "the aerosol-free run rains 63.8% MORE",
        "5.4x fewer droplets",
        "wif_input_opt=0 but mp_physics=28",
        "dyn_em/module_initialize_real.F:2734-2736",
        "current WIF-climatology route",
        "MYNN aerosol-number mixing is optional",
        "bl_mynn_mixscalars=0",
        "bldt=0",
        "MIXED NESTING uses the registered transition policy",
        "DELIBERATE THERMODYNAMIC DIVERGENCE FROM mp_physics=8",
        "CCN_ACTIVATE.BIN",
    ):
        assert phrase in text, f"the mp=28 warnings no longer say: {phrase}"
    for retired_claim in (
        # Retired by the 2.8.6 accumulator rework: every column but the
        # declared divergence clears the flat gate, so the warning may no
        # longer say they do not.
        "THE COLUMN EVIDENCE IS NOT CLEAN",
        "carries NO aerosol inflow",
        "flag_qnc/flag_qnwfa/flag_qnifa to MYNN as literal False",
        "MIXED NESTING IS REFUSED BY NAME",
        "byte-frozen at its",
    ):
        assert retired_claim not in text


def test_mp28_mixed_nest_edge_is_published_by_registry_and_runtime_alike():
    """One decision, asserted on both authorities.

    The registry publishes the edge as a cross-scheme rule and
    ``woof/core/microphysics_transition.py`` resolves it with a receipt
    that names each seeded value.  Two independent mechanisms for one
    decision is exactly the shape that drifts, so both are exercised here
    against the same pair.

    This test used to require the opposite, and audit R-004 turned it over
    with the refusal it guarded: what the old refusal called an unmeasured
    closure was WRF's own non-aerosol-aware fallback set, the values WRF
    installs whenever ``is_aerosol_aware`` is false.
    """
    from types import SimpleNamespace

    from woof.core.microphysics_transition import (
        SAME_SCHEME_POLICY,
        resolve_microphysics_transition,
    )

    def cfg(mp_physics, policy=None):
        return SimpleNamespace(
            mp_physics=mp_physics, moist=True, moist_cq=True,
            morr_rimed_ice=1, wsm6_hail_opt=0, wdm6_hail_opt=0,
            wdm6_ccn_conc=1.0e8,
            nest_microphysics_transition=(
                SAME_SCHEME_POLICY if policy is None else policy))

    rules = physics_registry()["transitions"]["microphysics-one-way-v1"]
    published = {
        (rule["parent_option_id"], rule["child_option_id"])
        for rule in rules["cross_options"]
    }
    assert ("thompson-mp8", MP28_OPTION_ID) in published
    assert (MP28_OPTION_ID, "thompson-mp8") in published

    mixed = _uniform_tree()
    mixed["domains"][0]["components"] = {"microphysics": "thompson-mp8"}
    mixed["domains"][1]["components"] = {"microphysics": MP28_OPTION_ID}
    child_parameters = mixed["domains"][1].setdefault("parameters", {})
    child_parameters["nest_microphysics_transition"] = (
        "mp-edge-mass-diagnosed-v1")
    child_parameters["mp28_aerosol_source"] = "synthetic"
    report = validate_physics_plan(mixed)
    assert "unsupported-component-transition" not in {
        error["code"] for error in report["errors"]}, report["errors"]

    for parent_mp, child_mp in ((8, 28), (28, 8)):
        contract = resolve_microphysics_transition(
            cfg(parent_mp), cfg(child_mp, "mp-edge-mass-diagnosed-v1"))
        assert contract.mixed is True
        seeded = {row["target_field"]: row.get("seeded_value")
                  for row in contract.species_actions()
                  if row["action"] == "diagnosed"}
        if child_mp == 28:
            assert seeded["nc"] == 100.0e6
            assert seeded["nwfa"] == 11.1e6
            assert seeded["nifa"] == 5.0e3
        else:
            # Leaving mp=28 needs no closure: the aerosol numbers are
            # dropped, and rain and ice numbers are carried (both schemes
            # are inverse-exponential in rain and cloud ice).
            assert set(seeded) == set()
            carried = {row["target_field"] for row in contract.species_actions()
                       if row["action"] == "mapped"
                       and row["reason"] == "shared_exponential_number_moment"}
            assert carried == {"nr", "ni"}

    # The same-scheme edge, which was always admitted, on both authorities.
    uniform = validate_physics_plan(_mp28_tree_plan())
    assert uniform["launchable"] is True, uniform["errors"]
    contract = resolve_microphysics_transition(cfg(28), cfg(28))
    assert contract.mixed is False
    assert (contract.source_mp_physics, contract.target_mp_physics) == (28, 28)


def test_mp28_activation_table_is_declared_as_a_shipped_but_separate_asset():
    """The asset row must say the wheel satisfies it, without joining mp=8.

    ``CCN_ACTIVATE.BIN`` is third-party parcel-model output that WRF
    redistributes; since 2026-08-01 this repository redistributes the same
    bytes, so ``redistributed_by_gpuwm`` must say so -- a row still claiming
    the operator has to supply it would send every user to a WRF ``run/``
    directory for a file already installed beside the code.

    ``kind`` is ``packaged-table-set`` (audit R-045).  It read
    ``operator-supplied-table-set`` from before the file shipped, and the
    row's own note and its ``redistributed_by_gpuwm`` field said the
    opposite in the same object; the stale word reached the refusal text a
    user saw.  ``kind`` names how the bytes ARRIVE.  What must never join
    the CLASSIC mp=8 set is this ASSET, and the thing that keeps it out is
    the ``id`` -- a separate set with its own root and its own environment
    overrides -- which is asserted below along with the classic contract
    being unchanged.  The pins are read back from
    ``woof/core/thompson_aerosol_contract.py`` so the registry and the
    loader cannot disagree about which bytes are meant, and the classic mp=8
    contract is asserted UNCHANGED so no existing launch inherits the
    dependency.
    """
    from woof.core.thompson_aerosol_contract import (
        AEROSOL_ASSET_REDISTRIBUTED,
        AEROSOL_TABLE_ASSETS,
        AEROSOL_TABLE_SET_ID,
    )
    from woof.core.thompson_contract import CLASSIC_TABLE_ASSETS

    requirements = _mp28_option()["asset_requirements"]
    assert len(requirements) == 1
    requirement = requirements[0]
    assert requirement["id"] == AEROSOL_TABLE_SET_ID
    assert requirement["kind"] == "packaged-table-set"
    assert requirement["redistributed_by_gpuwm"] is AEROSOL_ASSET_REDISTRIBUTED
    assert AEROSOL_ASSET_REDISTRIBUTED is True
    assert requirement["regenerable"] is False
    # Shipped is not regenerable: nothing in woof or WRF recomputes these
    # numbers, so a lost copy is fetched from a WRF release, not rebuilt.
    # Inside the recast-woof-data companion distribution since 2.5.0, at the same
    # relative path it always had.
    assert requirement["installed_root"] == "woof_data/data/thompson/tables"
    # And it declares the ladder plan review resolves it down, so a
    # missing table refuses at review rather than inside the run.
    assert requirement["resolution"]["data_relative"] == "thompson/tables"

    pinned = AEROSOL_TABLE_ASSETS[0]
    assert requirement["assets"] == [{
        "filename": pinned.filename,
        "bytes": pinned.bytes,
        "sha256": pinned.sha256,
    }]
    # An operator with a WRF tree must be able to act on the row alone.
    assert requirement["path_environment_override"] == (
        "WOOF_THOMPSON_CCN_ACTIVATE")
    assert requirement["root_environment_override"] == (
        "WOOF_THOMPSON_TABLE_ROOT")

    # The classic set is untouched: no mp=8 launch acquires this file.
    classic = {asset.filename for asset in CLASSIC_TABLE_ASSETS}
    assert pinned.filename not in classic
    mp8 = physics_registry()["components"]["microphysics"]["options"][
        "thompson-mp8"]["asset_requirements"]
    assert len(mp8) == 1
    assert {asset["filename"] for asset in mp8[0]["assets"]} == classic

    # A resolved mp=28 plan really does carry the requirement to the launcher.
    report = validate_physics_plan(_mp28_tree_plan())
    assert AEROSOL_TABLE_SET_ID in {
        item["requirement"]["id"] for item in report["asset_requirements"]}


def test_the_aerosol_roadmap_knobs_are_published_and_stay_unsettable():
    """WRF's aerosol-ingest knobs: roadmap rows, and the two that landed.

    mp=28 exists, so the right place for the family that is still absent
    is the published roadmap: typed, reasoned, and refused.

    AUDIT R-060 moved two of them off that roadmap, and the reason they
    were on it is the defect the audit found: the gate that kept these
    rows true proved only the POSITIVE claim, so a lane that landed the
    read was under no obligation to move the row and was mechanically
    forbidden from citing it.  ``woof/ingest/wif_climatology.py`` IS the
    WIF ingest the ``wif_input_opt`` row said woof did not have, and
    ``woof/ingest/real.py`` branches on the ``(aer_init_opt,
    wif_input_opt) == (1, 1)`` pair that ``woof/config.py``'s own error
    text tells a user to select -- a pair the registry made unspellable,
    one row refusing it and the other not existing at all.

    ``aer_fire_emit_opt`` stays absent, and now for the reason that
    survives: woof has no biomass-burning emission subsystem, so nothing
    carries the value.  WRF declaring it ``derived`` is a statement about
    WRF's namelist, and under the own-way ruling it does not decide what
    woof publishes -- what decides that is whether woof reads it.
    """
    registry = physics_registry()
    parameters = registry["parameters"]

    for name in (
        "num_wif_levels", "use_aero_icbc",
        "qna_update",
        "dust_emis", "wif_fire_emit", "wif_fire_inj",
        "progn", "naer",
    ):
        spec = parameters[name]
        assert spec["implemented"] is False, name
        assert spec["unimplemented_reason"].strip(), name
        assert "default" not in spec, name

    for name in ("wif_input_opt", "aer_init_opt"):
        spec = parameters[name]
        assert spec.get("implemented") is not False, name
        assert spec["consuming_read"], (
            f"{name} is published as implemented and cites no consuming "
            "read; tools/check_parameter_claims.py proves the claim, and a "
            "row with nothing to prove is the drift R-060 closed")

    assert "aer_fire_emit_opt" not in parameters, (
        "woof has no biomass-burning emission subsystem, so nothing "
        "carries aer_fire_emit_opt and publishing it would invent a control")

    # The three rows that pre-date the port now point at the option that
    # exists, instead of describing the scheme as unported.
    for name in ("progn", "naer", "use_aero_icbc"):
        assert MP28_OPTION_ID in parameters[name]["unimplemented_reason"], name

    # Published is not settable -- for the rows that are still roadmap.
    plan = _mp28_tree_plan()
    plan["domains"][1]["parameters"] = {"num_wif_levels": 0}
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert {"parameter-value", "parameter-route"} & {
        error["code"] for error in report["errors"]}


def test_mp28_does_not_disturb_the_frozen_mp8_registry_entry():
    """The whole port's premise, asserted at the registry layer too.

    mp=8 is the model-validated scheme every shipped template selects.  A
    sibling entry that quietly changed its maturity, its pins or its asset set
    would move a validated trajectory through configuration rather than
    through code, which no numerical gate in this tree is watching for.
    """
    mp8 = physics_registry()["components"]["microphysics"]["options"][
        "thompson-mp8"]
    # RE-PINNED at the 1.4.1 merge, and this is the one pin in the file that
    # is allowed to move for a reason outside mp=28: the mp=8 lane renamed
    # its own maturity tier on the release line.  The assertion still says
    # exactly what it said -- mp=8's registry entry is whatever the release
    # line makes it and mp=28 does not touch it -- so a change originating
    # in THIS branch still fails here.
    assert mp8["maturity"] == "wrf-matched-run"
    assert mp8["implemented"] is True
    assert mp8["reachability"] == {"state": "template"}
    assert mp8["selectors"] == {"mp_physics": 8}
    assert mp8["parameters"] == {"moist": True, "moist_cq": True}
    assert mp8["asset_requirements"][0]["id"] == (
        "wrf-v4.6.1-classic-thompson-mp8-gfortran13-v1")
    assert len(mp8["asset_requirements"][0]["assets"]) == 4


def test_the_published_mp28_evidence_agrees_between_registry_and_docs():
    """Nothing mechanically checks prose, so this does, for the numbers.

    ``docs/public/PHYSICS.md`` republishes the mp=28 column residuals for a
    reader who will never open the registry.  Two copies of a measurement is
    how a measurement goes stale, so every number the registry publishes must
    appear in the page, spelled the same way, and the page's maturity label
    for the row must be the registry's own.  This is deliberately one-
    directional: the page may say MORE than the registry (it carries the
    mechanism for each residual), but it may not disagree.
    """
    page = (ROOT / "docs" / "public" / "PHYSICS.md").read_text(
        encoding="utf-8")
    option = _mp28_option()
    evidence = option["extensions"]["column_oracle_evidence"]

    assert "| Thompson aerosol-aware | 28 |" in page, (
        "the microphysics table has no mp=28 row")
    assert f"| {option['maturity']} |" in page, (
        "the page does not use the maturity vocabulary the registry uses; a "
        "vocabulary rebase must move both")

    missing = []
    for name, fields in evidence["residual_fixtures"].items():
        if f"`{name}`" not in page:
            missing.append(name)
        for field, value in fields.items():
            if f"{value:.3e}" not in page:
                missing.append(f"{name}.{field}={value:.3e}")
    for name, fields in evidence["carved_out_bound"].items():
        if f"`{name}`" not in page:
            missing.append(name)
        for field, value in fields.items():
            if f"{value:.3e}" not in page:
                missing.append(f"{name}.{field}={value:.3e}")
    for name in evidence["clean_fixtures"]:
        if f"`{name}`" not in page:
            missing.append(name)
    assert missing == [], (
        "docs/public/PHYSICS.md no longer publishes what the registry "
        f"measured: {missing}")

    # What the page must and must not claim about forecast evidence.
    #
    # REWRITTEN ON 2026-08-01, in the commit that landed the matched
    # trajectory.  Two things were wrong with what stood here.
    #
    # 1.  It was VACUOUS.  It required "no matched" and "trajectory"
    #     anywhere in the whole page, and what satisfied it was three map
    #     PROJECTION rows -- "no matched WRF run" at PHYSICS.md:628-630 --
    #     which have nothing to do with microphysics.  Deleting every word
    #     about mp=28's forecast evidence would not have failed it.  It is
    #     now sliced to the mp=28 section, the same slice
    #     tests/test_physics_md_aerosol_claims.py uses.
    #
    # 2.  It required a claim that had become FALSE.  "No forecast has ever
    #     been verified against WRF" was true until a single-domain doubly
    #     periodic idealized forecast was run against WRF v4.6.1 and
    #     published, with its own failed declared condition, in
    #     docs/public/wrf-comparison/mp28-matched-trajectory.md.  A gate that
    #     forces the page to keep publishing a superseded sentence is a gate
    #     that manufactures a false statement.
    #
    # What is still true, and is what this now requires, is the NARROWER
    # claim: no REAL-DATA or NESTED matched comparison is recorded here.
    # The WIF boundary carrier now exists; absence of comparison evidence
    # must not be described as absence of a current runtime capability.
    # Tokens, not sentences, so a rewrite of the prose does not break it
    # and cannot quietly drop the qualification either.
    section = page[page.index("### Thompson aerosol-aware (`mp_physics = 28`)")
                   :page.index("## Planetary boundary layer")].lower()

    assert "real-data" in section and "nested" in section, (
        "the mp=28 section must qualify its forecast evidence: the matched "
        "comparison that exists is idealized, and no real-data or nested "
        "matched WRF comparison is recorded here")
    assert re.search(
        r"no real-data or nested\b[^.]*comparison against matched wrf"
        r"[^.]*is recorded here", section), (
        "the mp=28 section must explicitly state the missing real-data "
        "and nested matched-WRF comparison evidence")
    assert "wrf-comparison/mp28-matched-trajectory.md" in section, (
        "the mp=28 section must point at the one matched forecast "
        "comparison, so a reader reaches its limits and its failed "
        "declared condition without being told they exist")
    assert "implemented-unverified" in section, (
        "the matched idealized comparison did not raise the maturity label, "
        "and the section is where a reader learns that")
    for overclaim in ("model-validated | 28", "validation-candidate | 28",
                      "wrf-matched-run-candidate | 28",
                      "wrf-matched-run | 28"):
        assert overclaim not in page, overclaim

    # PROVENANCE.md is the repo-level deviation register and republishes the
    # same measurement in its D9 entry; it drifts the same way.
    register = (ROOT / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "### D9." in register, (
        "the deviation register has no mp=28 entry")
    drifted = []
    for group in ("residual_fixtures", "carved_out_bound"):
        for name, fields in evidence[group].items():
            for field, value in fields.items():
                if f"{value:.3e}" not in register:
                    drifted.append(f"{name}.{field}={value:.3e}")
    assert drifted == [], (
        f"PROVENANCE.md D9k no longer matches the registry: {drifted}")

    # CONFIGURATION.md is where a user looks for a knob, so every aerosol
    # knob the registry refuses has to be findable there under its own name.
    knobs = (ROOT / "docs" / "public" / "CONFIGURATION.md").read_text(
        encoding="utf-8")
    for name in (
        "use_aero_icbc", "use_rap_aero_icbc", "wif_input_opt",
        "num_wif_levels", "qna_update", "grav_settling",
        "dust_emis", "wif_fire_emit", "wif_fire_inj",
    ):
        assert f"`{name}`" in knobs, (
            f"CONFIGURATION.md does not document the refused knob {name}")
    assert "aer_init_opt" in knobs and "derived" in knobs, (
        "the page must say why aer_init_opt is not a WOOF knob")


def test_mp28_published_residuals_still_equal_a_live_adapter_measurement():
    """The published evidence is RE-MEASURED here, not just cross-checked.

    Every other gate on this row compares one written-down number against
    another written-down number: registry against docs, registry against the
    gate's declared bound, registry against the fixture deck on disk.  All of
    those stay green while the whole set drifts away from what the port
    actually computes, because a transcription is only as current as the last
    person who retyped it -- and this row's entire accuracy rests on the
    numbers being the real ones.  So this runs the nineteen fixtures through
    the shipped adapter on the device and rebuilds the published partition
    from the result.

    Three things are asserted, and the first two are the ones a stale
    transcription breaks:

    * a fixture the registry calls CLEAN must clear the gate on every field
      today -- otherwise the option is publishing a clean claim it no longer
      earns, which is the exact overclaim ``implemented-unverified`` exists to
      prevent;
    * a fixture the registry lists as a residual must still miss, on exactly
      the fields published, at exactly the published values;
    * the values are compared at the precision they are published to
      (``%.3e``), because that is the precision at which they are quoted in
      ``docs/public/PHYSICS.md``, ``PROVENANCE.md`` and the registry alike.

    Exact comparison is legitimate rather than flaky: the measurement was
    repeated three times end to end on this device and all 19x16 values were
    BIT-identical across repeats, so there is no run-to-run jitter for a
    tolerance to absorb.  If that ever stops being true the right response is
    to record the spread, not to widen this.
    """
    import importlib.util

    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:      # pragma: no cover
            pytest.skip("no CUDA device")
    except Exception:                                 # pragma: no cover
        pytest.skip("no CUDA device")

    spec = importlib.util.spec_from_file_location(
        "_mp28_adapter_gate_live",
        ROOT / "tests" / "test_thompson_aerosol_adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_mp28_adapter_gate_live", module)
    spec.loader.exec_module(module)
    # The adapter gate skips itself when the four classic tables or
    # CCN_ACTIVATE.BIN are not staged; this must skip for the same reason
    # rather than fail, because an unstaged table is an environment fact.
    module._tables_or_skip()

    evidence = _mp28_option()["extensions"]["column_oracle_evidence"]
    gate = evidence["gate_relative"]
    published_residual = evidence["residual_fixtures"]
    published_carved = evidence["carved_out_bound"]
    clean = set(evidence["clean_fixtures"])

    drift: list[str] = []
    for scenario in module._FIXTURES:
        measured, _ = module._run_g3(cp, scenario)
        carved = module._END_TO_END_BOUNDS.get(scenario, {})
        missing = {field: value for field, value in measured.items()
                   if not value <= carved.get(field, gate)}

        if scenario in clean:
            if missing:
                drift.append(
                    f"{scenario} is published CLEAN but misses "
                    + ", ".join(f"{field}={value:.3e}"
                                for field, value in sorted(missing.items())))
            continue

        expected = published_residual.get(scenario) or published_carved.get(
            scenario)
        if expected is None:
            # THE THIRD CLASS, and it became essential at the 1.4.1 merge.
            # A fixture can also be published as clean-only-under-the-
            # near-cancellation-bound, which is a METRIC change at one level
            # rather than a relative carve-out and so has never lived in
            # published_carved.  aero-reduces-to-classic was in BOTH classes
            # until the merge retired its relative bound; it is now in this
            # one alone, and reading only the first two would have reported
            # the port's best-documented fixture as unpublished.
            if scenario in evidence["near_cancellation_bound"]["fixtures"]:
                if missing:
                    drift.append(
                        f"{scenario} is published clean under the "
                        "near-cancellation bound but misses "
                        + ", ".join(f"{field}={value:.3e}"
                                    for field, value in sorted(
                                        missing.items())))
                continue
            drift.append(f"{scenario} is in no published class")
            continue

        if scenario in published_carved:
            # A carved-out fixture clears the gate only because of the bound,
            # so what is published is the measured value under it.
            for field, value in expected.items():
                got = measured[field]
                if f"{got:.3e}" != f"{value:.3e}":
                    drift.append(
                        f"{scenario}.{field}: published {value:.3e}, "
                        f"measured {got:.3e}")
            continue

        if set(missing) != set(expected):
            drift.append(
                f"{scenario}: published misses {sorted(expected)}, "
                f"measured misses {sorted(missing)}")
        for field, value in expected.items():
            got = measured.get(field)
            if got is None or f"{got:.3e}" != f"{value:.3e}":
                drift.append(
                    f"{scenario}.{field}: published {value:.3e}, measured "
                    + ("absent" if got is None else f"{got:.3e}"))

    assert drift == [], (
        "the registry publishes mp=28 column evidence that the adapter no "
        "longer produces. Re-measure with "
        "tests/test_thompson_aerosol_adapter.py::"
        "test_g3_end_to_end_against_all_nineteen_oracle_fixtures and move the "
        "numbers in tools/build_registry.py (MP28_G3_CLEAN / "
        "MP28_G3_RESIDUALS / MP28_G3_CARVED_OUT), then regenerate the "
        "registry and update docs/public/PHYSICS.md and PROVENANCE.md D9k. "
        "Never round a residual toward the gate.\n  " + "\n  ".join(drift))


# ==========================================================================
# What the registry says an mp=28 run DOES, versus what the shipped tree
# does.  Both gates below exist because a machine-readable claim that is
# false about the model that ships is worse than no claim at all: a front end
# renders it, a user plans around it, and nothing in the tree contradicts it.
# ==========================================================================

def _microphysics_init_production_callers() -> list[str]:
    """Every non-test call site of ``microphysics.microphysics_init``.

    Deliberately the SAME scan as ``tests/test_mp28_forecast_smoke.py::
    test_gap_microphysics_init_has_no_production_call_site``, so the pinned
    gap and the published claim cannot disagree about whether WRF's synthetic
    CCN/IN profile is installed on a real run.  ``microphysics.py`` itself is
    inspected above its own definition only, so the definition is not counted
    as a call.
    """
    root = ROOT / "woof"
    callers: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if path.name == "microphysics.py" and "def microphysics_init" in text:
            head = text.split("def microphysics_init", 1)[0]
            if "microphysics_init(" in head:
                callers.append(path.relative_to(ROOT).as_posix())
            continue
        if re.search(r"\bmicrophysics_init\s*\(", text):
            callers.append(path.relative_to(ROOT).as_posix())
    return callers


def test_mp28_publishes_the_aerosol_initialisation_the_tree_actually_performs():
    """The registry may not promise a CCN profile the tree never installs.

    WRF's ``thompson_init`` fills a synthetic CCN/IN profile at domain
    construction whenever the aerosol fields arrive unset
    (phys/module_mp_thompson.F:493-558) and derives ``nwfa2d`` from it at
    :509-510.  woof implements that fill in
    ``woof/core/microphysics.py::microphysics_init`` and proves it against
    WRF -- but until something in ``gpuwm/`` CALLS it, every mp=28 forecast
    starts from ``nwfa = nifa = 0`` and the terminal apply's clamps
    (:3972-4021) hold the aerosol at its floors for the whole run.  The
    registry used to publish the opposite ("Every mp_physics=28 run takes
    thompson_init's SYNTHETIC CCN/IN profile"), which a front end renders as
    a capability.

    So the published claim is derived from the tree here, not transcribed:
    the call sites are scanned, and the registry must name exactly what the
    scan finds.  When the hook is finally wired this test does not go stale --
    it demands that the registry stop warning, and names the caller.
    """
    option = _mp28_option()
    assert "aerosol_initialisation" in option["extensions"], (
        "the registry does not publish which aerosol initialisation an "
        "mp_physics=28 run actually performs; a reader cannot tell whether "
        "WRF's synthetic CCN/IN profile is installed")
    block = option["extensions"]["aerosol_initialisation"]

    from woof.core import microphysics

    assert callable(microphysics.microphysics_init)
    assert block["gpuwm_implementation"] == (
        "woof/core/microphysics.py::microphysics_init")
    assert block["wrf_source"].startswith("phys/module_mp_thompson.F:493-558")

    callers = _microphysics_init_production_callers()
    published = block["production_call_site"]

    warnings = " ".join(option["warnings"])
    # The exact sentence the port shipped as false.  Named as a literal so a
    # rewrite that reintroduces it fails here rather than being rediscovered.
    false_claim = "takes thompson_init's SYNTHETIC CCN/IN profile"

    if not callers:                                   # pragma: no cover
        # The state this row was written for, kept live so that a refactor
        # which drops the call reopens the warning instead of leaving the
        # registry promising a profile nothing installs.
        assert published is None, (
            f"the registry publishes production_call_site={published!r} "
            "while nothing in gpuwm/ calls microphysics_init")
        assert false_claim not in warnings, (
            "the registry still claims mp=28 runs take WRF's synthetic CCN/IN "
            "profile, and nothing in gpuwm/ calls microphysics_init")
        assert "NO AEROSOL INITIALISATION" in warnings, (
            "the warning list does not name the gap at all; a user selecting "
            "the scheme is told nothing about running at the CCN floor")
        assert block["shipped_profile"].startswith("none:"), block
        assert "11.1e6" in block["shipped_consequence"], (
            "the consequence must name the floor the run is pinned at")
        assert block["operator_workaround"], (
            "a documented gap must tell the operator what to do instead")
        return

    # THE CALL IS WIRED (2026-08-01).  What the registry must now do is name
    # the caller exactly, in a form that is repo-relative rather than a path
    # on whoever's machine generated it, and stop warning about a gap that
    # is closed -- a stale warning is as false as a stale capability claim.
    assert isinstance(published, str) and "::" in published, (
        "production_call_site must name module::function, not just a module; "
        f"got {published!r}")
    module_path, _, function = published.partition("::")
    assert module_path == callers[0], (
        f"the registry publishes production_call_site={published!r} while "
        f"the tree has {callers!r}")
    assert not module_path.startswith("/") and ".." not in module_path, (
        f"the published call site is not repo-relative: {module_path!r}")
    source = (ROOT / module_path).read_text(encoding="utf-8")
    assert f"def {function}(" in source, (
        f"{module_path} has no function named {function}")
    assert len(callers) == 1, (
        "WRF calls thompson_init from mp_init and nowhere else; a second "
        f"woof caller is how once-per-domain becomes twice: {callers}")

    assert false_claim not in warnings, false_claim
    assert "NO AEROSOL INITIALISATION" not in warnings, (
        f"{published} now installs the profile; that warning is stale")
    assert "NOTHING in gpuwm/ CALLS it" not in warnings, (
        "the registry still warns that nothing calls the hook, and "
        f"{published} does")
    assert "AEROSOL INPUT LIMITS" in warnings
    assert "analyzed QNWFA/QNIFA pair" in warnings
    assert "monthly WIF climatology reader" in warnings
    assert "black-carbon (nbca) species" in warnings
    assert "NO AEROSOL INGEST" not in warnings
    assert published in warnings, (
        "the human-readable warning must name the caller too; a front end "
        "renders warnings and not extensions")

    assert block["shipped_profile"].startswith("WRF's synthetic"), block
    # The installed profile must be described by its OWN values, and they
    # must be the fixture's, not a round number: WRF's :508 exponent carries
    # the first layer's thickness, so the lowest level is nowhere near the
    # naCCN1 + naCCN0 = 350e6 ceiling.
    for value in ("1.478987e+08", "5.000000e+07"):
        assert value in block["shipped_profile"], value
    assert "350e6" not in block["shipped_profile"]
    assert "strictly ABOVE" in block["shipped_consequence"], (
        "the consequence of installing the profile is that a domain starts "
        "above WRF's clamps rather than pinned at them; that is what makes "
        "it a physics difference rather than a cosmetic one")
    assert block["operator_workaround"].startswith("none needed"), (
        "the workaround must stop telling operators to call the hook "
        "themselves now that initialize_physics does")

    cost = block["measured_forecast_sensitivity"]
    for key in ("test", "steps", "timestep_s", "domain", "reading",
                "initial_mean_nwfa_per_kg_with_profile",
                "initial_mean_nwfa_per_kg_without_profile",
                "peak_nc_per_kg_with_profile",
                "peak_nc_per_kg_without_profile",
                "domain_total_rainnc_mm_with_profile",
                "domain_total_rainnc_mm_without_profile",
                "domain_total_rainnc_relative_excess"):
        assert key in cost, key
    # Internally consistent: the published excess must be the ratio of the
    # two published accumulations, not an independently typed number.
    excess = (cost["domain_total_rainnc_mm_without_profile"]
              / cost["domain_total_rainnc_mm_with_profile"] - 1.0)
    assert abs(excess - cost["domain_total_rainnc_relative_excess"]) < 1e-3
    assert cost["domain_total_rainnc_relative_excess"] > 0.10, (
        "the number that made this the port's largest measured error is the "
        "number that now says how much the profile is worth; it does not "
        "disappear when the call lands")
    assert (cost["peak_nc_per_kg_without_profile"]
            < cost["peak_nc_per_kg_with_profile"]), (
        "the published mechanism is fewer droplets without CCN")
    assert cost["initial_mean_nwfa_per_kg_without_profile"] == 0.0
    ratio = (cost["peak_nc_per_kg_with_profile"]
             / cost["peak_nc_per_kg_without_profile"])
    assert abs(ratio - cost["droplet_ratio_with_over_without"]) < 0.05, (
        "the published droplet ratio is not the ratio of the two published "
        f"peaks ({ratio:.3f})")
    assert cost["test"].startswith("tests/test_mp28_forecast_smoke.py::")
    # The numbers must also be in the human-readable warning, because a
    # front end renders warnings and not extensions.
    for value in ("1.5980e+08", "2.9451e+07", "1.957357", "3.207102"):
        assert value in warnings, value


def test_the_published_aerosol_initialisation_cost_is_still_the_measured_one():
    """RE-MEASURE the published defect size rather than trusting the digits.

    Runs the same two forecasts the evidence comes from -- identical in every
    respect except that one calls ``microphysics_init`` -- and checks the
    claim the registry actually makes.

    WHAT IS GATED, AND WHY NOT THE DIGITS.  ``domain_total_rainnc_*`` is
    published as a SNAPSHOT of how large this gap is, not as a physics pin:
    any legitimate mp=28 numerics change (a closed column residual, a
    corrected rate) moves the sixth decimal of both accumulations, and a
    digit-exact gate here would go red on an improvement.  What must not
    change silently is the claim: the aerosol-free run rains MORE, by a lot,
    and it produces FEWER droplets.  So the sign and the >10 % magnitude are
    gated exactly as published, and the published excess is required to stay
    within a factor of two of the live one -- tight enough to catch the gap
    being closed, fixed, or growing an order of magnitude, and loose enough
    not to fail on a legitimate physics correction elsewhere in the port.
    """
    import importlib.util

    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:      # pragma: no cover
            pytest.skip("no CUDA device")
    except Exception:                                 # pragma: no cover
        pytest.skip("no CUDA device")

    spec = importlib.util.spec_from_file_location(
        "_mp28_forecast_smoke_live",
        ROOT / "tests" / "test_mp28_forecast_smoke.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_mp28_forecast_smoke_live", module)
    spec.loader.exec_module(module)
    module._tables_or_skip()

    cost = _mp28_option()["extensions"]["aerosol_initialisation"][
        "measured_forecast_sensitivity"]
    assert cost["steps"] == module.FORECAST_STEPS
    assert cost["timestep_s"] == module.FORECAST_DT

    _cfg, filled = module._forecast(cp, bubble=True, initialise=True)
    _cfg2, unfilled = module._forecast(cp, bubble=True, initialise=False)
    assert filled["init_receipt"], "the control run installed no profile"
    assert unfilled["init_receipt"] == {}
    assert float(unfilled["nwfa_initial"].max()) == 0.0

    rain_filled = float(filled["rain_sum"][-1])
    rain_unfilled = float(unfilled["rain_sum"][-1])
    assert rain_filled > 0.0 and rain_unfilled > 0.0
    live = rain_unfilled / rain_filled - 1.0
    published = cost["domain_total_rainnc_relative_excess"]

    assert live > 0.10, (
        "the published defect size no longer holds: removing WRF's CCN "
        f"profile changed domain-total rain by {live:+.2%}")
    assert max(unfilled["nc_max"]) < max(filled["nc_max"]), (
        "the published mechanism (fewer, larger droplets without CCN) is not "
        "the one acting")
    assert 0.5 <= (1.0 + live) / (1.0 + published) <= 2.0, (
        f"the registry publishes a {published:+.1%} rain excess and the tree "
        f"now measures {live:+.1%}; re-measure with {cost['test']} and move "
        "the numbers in tools/build_registry.py")


def test_the_registry_flag_qs_contract_is_the_one_the_shipped_runtime_applies():
    """The published MYNN snow classification must be the runtime's own.

    WRF's Registry.EM_COMMON:3036 gives package ``thompsonaero`` (mp_physics
    == 28) ``moist:qv,qc,qr,qi,qs,qg``, so ``F_QS`` is TRUE for it;
    phys/module_pbl_driver.F:877 forwards that flag and
    phys/module_bl_mynn.F:734/:876 substitute ``sqs = 0`` when it is false.
    The registry has published 28 as a flag_qs-true selector since the scheme
    was registered -- while ``woof/core/mynn_pbl_runtime.py`` passed
    ``flag_qs=False`` for it, so MYNN never saw snow under mp=28 and the
    published claim was false about the model that ships.

    The two are bound together here rather than compared by eye: the runtime
    set is the authority for what ships, WRF's Registry is the authority for
    what is right, and the registry may not disagree with either.
    """
    from woof.core.mynn_pbl_runtime import (
        MYNN_SNOW_MICROPHYSICS, mynn_flag_qs)

    registry = physics_registry()
    species = registry["components"]["pbl"]["options"]["mynn"][
        "extensions"]["supplied_moisture_species"]
    published_true = species["flag_qs_true_microphysics_selectors"]
    published_false = species["flag_qs_false_microphysics_selectors"]

    assert 28 in published_true, (
        "WRF Registry.EM_COMMON:3036 declares qs for the thompsonaero "
        "package, so F_QS is true for mp_physics=28")
    assert sorted(MYNN_SNOW_MICROPHYSICS) == published_true, (
        "the registry publishes a MYNN snow classification the shipped "
        f"runtime does not apply: registry {published_true}, runtime "
        f"{sorted(MYNN_SNOW_MICROPHYSICS)}")
    for selector in published_true:
        assert mynn_flag_qs(selector) is True, selector
    for selector in published_false:
        assert mynn_flag_qs(selector) is False, selector
    assert species["gpuwm_runtime_source"].startswith(
        "woof/core/mynn_pbl_runtime.py::MYNN_SNOW_MICROPHYSICS"), (
        "the published classification must name the shipped set it is "
        "checked against")

    # Every implemented microphysics option has to be on exactly one side.
    live = {int(option["selectors"]["mp_physics"])
            for option in registry["components"]["microphysics"][
                "options"].values()
            if option.get("implemented") is True}
    assert live == set(published_true) | set(published_false)
    assert not set(published_true) & set(published_false)


def test_mp28_mynn_is_handed_snow_rather_than_a_published_promise():
    """The classification is only worth what the driver does with it.

    A selector list is a claim about behaviour, so the behaviour is measured:
    the committed WRF v4.6.1 MYNN driver oracle's snow-bearing column is run
    through woof's own driver with the flag the mp=28 runtime supplies, once
    with the oracle's snow and once with it zeroed.  If mp=28 really is a
    flag_qs-true selector the two must differ; when the runtime passed
    ``flag_qs=False`` they were bitwise identical, which is exactly what
    "MYNN never sees snow under mp=28" means.

    MEASURED on the ``snow_anvil`` column (max sqs 4.08e-05): with the flag
    on, ``qi_bl`` peaks at 5.4863e-07 and with it off it is exactly 0, and
    every other MYNN output moves with it.
    """
    import importlib.util

    import numpy as np

    from woof.core.mynn_pbl import mynn_bl_driver
    from woof.core.mynn_pbl_runtime import mynn_flag_qs

    spec = importlib.util.spec_from_file_location(
        "_mynn_driver_oracle", ROOT / "tests" / "test_mynn_pbl.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_mynn_driver_oracle", module)
    spec.loader.exec_module(module)

    _blocks, values, initflag, delt = module._driver_step(2)
    index = module.DRIVER_CASES.index("snow_anvil")
    assert float(values["sqs"][index].max()) > 1.0e-5, (
        "the oracle column carries no snow, so this measures nothing")

    supplied = mynn_bl_driver(
        values, initflag=initflag, delt=delt, flag_qs=mynn_flag_qs(28))
    zeroed = {name: array.copy() for name, array in values.items()}
    zeroed["sqs"][...] = 0.0
    withheld = mynn_bl_driver(
        zeroed, initflag=initflag, delt=delt, flag_qs=mynn_flag_qs(28))

    moved = [name for name in ("qi_bl", "qc_bl", "cldfra_bl", "rqvblten",
                               "rthblten", "exch_h")
             if not np.array_equal(np.asarray(supplied[name])[index],
                                   np.asarray(withheld[name])[index])]
    assert "qi_bl" in moved, (
        "MYNN produced the same cloud ice with and without the column's "
        "snow, so mp_physics=28 is not being handed snow at all")
    assert float(np.max(np.asarray(withheld["qi_bl"])[index])) == 0.0
    assert float(np.max(np.asarray(supplied["qi_bl"])[index])) > 0.0
    assert len(moved) >= 5, moved


def test_an_unnamed_mp28_tree_is_registry_reachable_without_an_acknowledgement():
    """The front door a user actually reaches, asked directly.

    ``validate_physics_plan`` is the declarative authority; the tuple-capability
    check in :func:`woof.physics_compat.multi_domain_physics_selection` is the
    one a launch goes through, and it consults the SAME registry reachability.
    A component-override option that resolved as
    ``outside-registry-declared-reachability`` would demand
    ``--ack expert-tuple-v1`` for a scheme the registry says is normally
    selectable, which is the shape of a front door that disagrees with its own
    catalogue.
    """
    from woof.physics_compat import (
        multi_domain_physics_selection,
        single_domain_runtime_switches,
    )

    aerosol = {**single_domain_runtime_switches(WSM6_TEMPLATE_ID),
               "mp_physics": 28}
    receipt = multi_domain_physics_selection({1: aerosol, 2: aerosol})
    assert {domain["governance"]["state"]
            for domain in receipt["domains"].values()} == {
        "registry-reachable"}
    assert receipt["acknowledgements"] == []


@pytest.mark.parametrize("profile", [NOAHMP_PROFILE_ID, MYNN_NOAHMP_PROFILE_ID,
                                     MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID])
def test_named_profile_advisory_does_not_change_executable_selectors(
        profile, capsys):
    from woof.physics_compat import (single_domain_physics_selection,
                                     validate_single_domain_physics_profile)

    settings = single_domain_runtime_switches(profile)
    named = validate_single_domain_physics_profile(profile, config=settings)
    assert "unacknowledged" in capsys.readouterr().err
    assert named["governance"]["acknowledged"] is False
    assert named["acknowledgements"] == []
    unnamed = single_domain_physics_selection(config=settings)
    assert named["selectors"] == unnamed["domains"]["1"]["selectors"]
    capsys.readouterr()
    acknowledged = validate_single_domain_physics_profile(
        profile, config=settings,
        expert_acknowledgements=("noahmp-host-column-throughput-v1",))
    assert acknowledged["governance"]["acknowledged"] is True
    assert not capsys.readouterr().err
    assert acknowledged["selectors"] == named["selectors"]

    # An explicit profile is still an assertion of its actual settings.
    with pytest.raises(ValueError, match="differs from profile"):
        validate_single_domain_physics_profile(
            profile, config=dict(settings, epssm=float(settings["epssm"]) + 0.01))


@pytest.mark.parametrize("profile", [NOAHMP_PROFILE_ID, MYNN_NOAHMP_PROFILE_ID,
                                     MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID])
def test_registry_plan_can_run_without_expert_advisory_acknowledgement(profile):
    plan = _single_plan(profile)
    plan.pop("acknowledgements")
    report = validate_physics_plan(plan)
    assert report["launchable"] is True, report["errors"]
    assert not report["errors"]
    assert "expert-acknowledgement-advisory" in {
        warning["code"] for warning in report["warnings"]}
