"""Fail-closed gates for the Thompson + MYNN + RUC port lane."""

from __future__ import annotations

from dataclasses import replace

import pytest

from woof.config import RunConfig, validate_run_config
from woof.physics_compat import (
    IMPLICIT_RUNTIME_SWITCHES,
    PhysicsCapabilityError,
    SINGLE_DOMAIN_PHYSICS_PROFILES,
    UnsupportedPhysicsSuiteError,
    WSM6_PROFILE_ID,
    WRF_RRTMG_TO_RTE_RRTMGP,
    implicit_runtime_switches,
    pending_wrf_physics_components,
    single_domain_runtime_switches,
    validate_physics_capabilities,
)


def _cfg(**overrides) -> RunConfig:
    values = dict(nx=4, ny=3, nz=4, dx=1000.0, dy=1000.0,
                  ztop=10000.0, dt=5.0, run_seconds=10.0)
    values.update(overrides)
    return RunConfig(**values)



@pytest.fixture
def captured_warnings():
    """Every :func:`woof.explain.warn` record raised inside the block.

    ``woof.explain.warn`` prints to stderr and feeds registered observers;
    it does NOT go through Python's ``warnings`` module, so ``pytest.warns``
    cannot see it.  The observer hook is the tree's own machine-facing
    surface for exactly this, and it gives the two halves as fields rather
    than a line to pattern-match.
    """
    from woof.explain import add_warning_observer, remove_warning_observer

    records: list[dict] = []
    add_warning_observer(records.append)
    try:
        yield records
    finally:
        remove_warning_observer(records.append)

def test_full_target_mynn_ruc_pair_has_no_remaining_blocker(captured_warnings):
    """The target suite's receipt after the ownership port.

    MYNN no longer appears because 5/5 runs, "RUC land-surface model" no
    longer appears because RUC runs, and "Thompson microphysics" no longer
    appears because mp8 is first-class now that the canonical classic
    tables ship as package data (product decision, product/v1 packaging
    lane 2026-07-28 -- the byte validation moved to table load, it did not
    disappear).  The MYNN/RUC pair is now admitted because WRF's ordered
    RUC write-back and SFCDIAGS_RUCLSM ownership are explicit.
    """
    blockers = pending_wrf_physics_components(
        mp_physics=8, sf_sfclay_physics=5, bl_pbl_physics=5,
        sf_surface_physics=3, num_soil_layers=9)
    assert blockers == ()

    # SIX LEVELS IS NO LONGER BLOCKED.  The 'RUC forecast column soil
    # geometry' blocker was retired when the column stopped being pinned to
    # nine: ruc.cu sizes every soil scratch from RUC_NZS, and the host and
    # device lanes resolve the count from the profile.  What replaced it is a
    # WARNING, because what six levels lacks is a WRF forecast oracle, not a
    # code path -- the warn-not-block ruling this file already applies to
    # Noah-MP's column width.
    six = pending_wrf_physics_components(
        mp_physics=6, sf_sfclay_physics=91, bl_pbl_physics=1,
        sf_surface_physics=3, num_soil_layers=6)
    assert six == ()
    assert any("no WRF forecast oracle" in record["action"]
               for record in captured_warnings), captured_warnings

    # WRF's OWN refusal survives untouched: a count init_soil_depth_3 does
    # not tabulate has no zs at all, and that is a blocker at 4 and at 5.
    for count in (4, 5):
        refused = pending_wrf_physics_components(
            mp_physics=6, sf_sfclay_physics=91, bl_pbl_physics=1,
            sf_surface_physics=3, num_soil_layers=count)
        assert [item.component for item in refused] == ["RUC soil geometry"]
        assert refused[0].selectors == (
            ("sf_surface_physics", 3), ("num_soil_layers", count))

    # And the admitted combination reports nothing at all, which is what makes
    # every refusal above a statement rather than a blanket.
    assert pending_wrf_physics_components(
        mp_physics=6, sf_sfclay_physics=91, bl_pbl_physics=1,
        sf_surface_physics=3, num_soil_layers=9) == ()


def test_every_shipped_profile_answers_its_own_implicit_switches():
    """The drift guard for the one authority on moist_cq/top_lid.

    Three places used to decide these: the shipped profiles (read by the
    HRRR root preparer AND the domain wizard) and the WRF namelist
    importer, which invented ``moist_cq = mp_physics > 0`` and WRF's
    open-top Registry default.  For every WSM6-family suite the two
    answers were opposite, which is why a public root prepared from a
    profile could not bind a public hierarchy imported from the same
    namelist.  If a new profile ever disagrees with what this lookup
    returns for its own selectors, that separation is back, and this
    fails.
    """

    for profile in SINGLE_DOMAIN_PHYSICS_PROFILES:
        switches = single_domain_runtime_switches(profile)
        assert switches["moist_cq"] is True, profile
        resolved = implicit_runtime_switches(**switches)
        assert profile in resolved["profiles"], profile
        for name in IMPLICIT_RUNTIME_SWITCHES:
            assert resolved[name] == switches[name], (profile, name)
        assert profile in resolved["source"]

    # An unmatched moist suite still uses WRF's moisture pressure
    # correction; unrelated implicit switches retain their defaults.
    defaults = _cfg()
    unknown = implicit_runtime_switches(
        mp_physics=6, sf_sfclay_physics=1, sf_surface_physics=0,
        bl_pbl_physics=0, cu_physics=0, num_soil_layers=4,
        ra_lw_physics=1, ra_sw_physics=1)
    assert unknown["profiles"] == ()
    assert unknown["moist_cq"] is True
    assert unknown["top_lid"] == defaults.top_lid
    assert "not one of the shipped" in unknown["source"]

    # An empty selection is not a match for everything.
    empty = implicit_runtime_switches()
    assert empty["profiles"] == ()
    assert empty["moist_cq"] is True


@pytest.mark.parametrize("mp,sfclay,expected_cq,matched", [
    (8, 1, True, False),
    (8, 91, True, True),
    (0, 1, True, False),
])
def test_surface_scheme_selection_keeps_the_moist_pressure_correction(
        mp, sfclay, expected_cq, matched):
    resolved = implicit_runtime_switches(
        mp_physics=mp, sf_sfclay_physics=sfclay, sf_surface_physics=2,
        bl_pbl_physics=1, cu_physics=0, num_soil_layers=4,
        ra_lw_physics=4, ra_sw_physics=4)
    assert resolved["moist_cq"] is expected_cq
    assert bool(resolved["profiles"]) is matched
    if not matched:
        assert "WRF moisture pressure correction" in resolved["source"]


def test_no_shipped_selector_reachable_option_is_unimplemented():
    """Why the refusal below is driven against a PERTURBED registry.

    This case used to run the 1/1 WRF RRTM + Dudhia pair through the front
    door and expect a refusal, because ``radiation/wrf-rrtm-dudhia`` was
    registered-but-unimplemented.  814a68ffc ("wire(rrtm): the 1/1 pair
    becomes selectable, and the ladder loses a rung") WIRED that pair --
    woof/config.py stopped raising for ra_lw_physics=1,
    woof/core/physics.py bound (1,1) to RRTMDudhiaRadiation, and the
    registry row flipped to implemented=true.  That lane repointed the
    sibling case in tests/test_physics_registry.py and missed this one,
    so it went on asserting a refusal that had correctly stopped firing.

    THE GUARD DID NOT BREAK.  The branch is still in
    woof/physics_compat.py::_resolve_physics_component_options and still
    refuses; what it lost is a witness.  The one registered-but-
    unimplemented option left in the whole registry is
    ``microphysics/sase``, and it carries NO selectors, so no
    configuration can name it and the selector resolver can never make it
    a candidate.  This test states that fact -- and fails the day it stops
    being true, which is the day a real witness exists and the perturbed
    case below should be replaced by it.
    """

    from woof.physics_registry import physics_registry

    registry = physics_registry()
    unwitnessed = [
        (component_id, option_id)
        for component_id, component in registry["components"].items()
        if component.get("selector_keys")
        for option_id, option in component["options"].items()
        if option.get("implemented") is not True and option.get("selectors")
    ]
    assert unwitnessed == [], (
        "these options are selector-reachable AND unimplemented, so the "
        "front-door refusal has a real witness again: drive "
        "test_front_door_capability_refusal_cites_unimplemented_registry_"
        f"option from {unwitnessed} instead of from a perturbed registry")


def test_front_door_capability_refusal_cites_unimplemented_registry_option(
        monkeypatch):
    """The refusal names the pointer, the blocker and the selectors.

    Driven against a deep copy with ``radiation/wrf-rrtm-dudhia`` flipped
    back to unimplemented -- the exact shape the shipped registry carried
    until 814a68ffc -- because the shipped registry no longer contains a
    selector-reachable unimplemented option (see the test above).  The
    perturbation is on a copy; the shipped registry is not touched.
    """

    from copy import deepcopy

    import woof.physics_registry as registry_module

    blocker = "Not implemented: the 1/1 WRF RRTM+Dudhia pair"
    perturbed = deepcopy(registry_module.physics_registry())
    option = perturbed["components"]["radiation"]["options"][
        "wrf-rrtm-dudhia"]
    assert option["selectors"] == {"ra_lw_physics": 1, "ra_sw_physics": 1}
    option["implemented"] = False
    option["reachability"] = {"state": "unreachable", "blocker": blocker}
    monkeypatch.setattr(registry_module, "physics_registry",
                        lambda *a, **k: perturbed)

    selected = single_domain_runtime_switches(WSM6_PROFILE_ID)
    selected.update(ra_lw_physics=1, ra_sw_physics=1)

    with pytest.raises(PhysicsCapabilityError) as caught:
        validate_physics_capabilities(selected)

    message = str(caught.value)
    assert (
        "woof/physics_registry_v2.json#/components/radiation/options/"
        "wrf-rrtm-dudhia"
    ) in message
    assert blocker in message
    assert "selectors {'ra_lw_physics': 1, 'ra_sw_physics': 1}" in message


def test_the_1_1_pair_that_814a68ffc_wired_is_admitted():
    """The other half of the widening: the pair now RUNS.

    A refusal that stopped firing is only correct if the thing it refused
    works.  This is the positive statement 814a68ffc earned, and it is
    what stops the case above from being restored by reflex.
    """

    selected = single_domain_runtime_switches(WSM6_PROFILE_ID)
    selected.update(ra_lw_physics=1, ra_sw_physics=1)
    resolved = validate_physics_capabilities(selected)
    assert resolved["radiation"] == "wrf-rrtm-dudhia"


def test_mixed_mynn_pairings_follow_wrf_v461_instead_of_a_half_suite_gate():
    """WRF admits three mixed pairings and fatals the other three."""

    for surface, pbl in (
            (5, 5), (1, 5), (91, 5), (5, 0), (1, 0), (91, 0)):
        assert pending_wrf_physics_components(
            mp_physics=6, sf_sfclay_physics=surface,
            bl_pbl_physics=pbl, sf_surface_physics=2,
            num_soil_layers=4) == ()

    for surface, pbl in ((5, 1), (0, 1), (0, 5)):
        blockers = pending_wrf_physics_components(
            mp_physics=6, sf_sfclay_physics=surface,
            bl_pbl_physics=pbl, sf_surface_physics=2,
            num_soil_layers=4)
        assert [item.component for item in blockers] == [
            "WRF v4.6.1 PBL/surface-layer compatibility"]
        assert blockers[0].selectors == (
            ("sf_sfclay_physics", surface), ("bl_pbl_physics", pbl))
        assert "phys/module_physics_init.F:" in blockers[0].missing[1]


def test_target_runconfig_is_admitted_without_substitution(captured_warnings):
    """The target MYNN/RUC suite validates directly.

    RUC at nine layers is admitted, mp8 is first-class (packaged tables;
    the promotion is the same product decision documented in
    ``test_full_target_mynn_ruc_pair_has_no_remaining_blocker``), and the
    surface-driver ownership port makes the MYNN/RUC pair first-class.
    """
    cfg = _cfg(nz=5, moist=True, mp_physics=8, sf_sfclay_physics=5,
               bl_pbl_physics=5, sf_surface_physics=3,
               num_soil_layers=9)
    assert validate_run_config(cfg) is cfg

    # SIX IS ADMITTED NOW, and admitted is not the same as validated: it
    # passes validation and warns, naming exactly what has not been measured.
    six = _cfg(moist=True, mp_physics=6, sf_sfclay_physics=91,
               bl_pbl_physics=1, sf_surface_physics=3, num_soil_layers=6)
    assert validate_run_config(six) is six
    assert any("no WRF forecast oracle" in record["action"]
               for record in captured_warnings), captured_warnings

    # A geometry WRF itself does not tabulate is still refused outright.  On
    # THIS path the refusal comes from woof.config's soil-layer resolver,
    # which runs before the physics-suite check ever sees the config, so it
    # is a ValueError rather than an UnsupportedPhysicsSuiteError.  The
    # physics-suite half of the same refusal is asserted directly against
    # pending_wrf_physics_components in
    # test_full_target_mynn_ruc_pair_has_no_remaining_blocker.
    for count in (4, 5):
        bad = _cfg(moist=True, mp_physics=6, sf_sfclay_physics=91,
                   bl_pbl_physics=1, sf_surface_physics=3,
                   num_soil_layers=count)
        with pytest.raises(ValueError, match="must be 6 or 9"):
            validate_run_config(bad)

    # And the admitted combination validates, which is what makes the two
    # refusals above meaningful rather than a blanket "RUC is unsupported".
    validate_run_config(_cfg(moist=True, mp_physics=6, sf_sfclay_physics=91,
                             bl_pbl_physics=1, sf_surface_physics=3,
                             num_soil_layers=9))


def test_thompson_runconfig_is_admitted_without_environment(monkeypatch):
    """mp8 validates with NO Thompson environment set (first-class contract).

    Table bytes are not this gate's question: the packaged root (or an
    override) is byte-validated at load, where a wrong root fails closed.
    """
    monkeypatch.delenv("WOOF_EXPERIMENTAL_THOMPSON_MP8", raising=False)
    monkeypatch.delenv("WOOF_THOMPSON_TABLE_ROOT", raising=False)
    validate_run_config(_cfg(moist=True, mp_physics=8))


def test_rrtmg_compatibility_token_is_explicit_and_pair_bound():
    native = validate_run_config(_cfg(ra_physics=4))
    assert native.wrf_rrtmg_compatibility == "none"

    imported = validate_run_config(replace(
        native, wrf_rrtmg_compatibility=WRF_RRTMG_TO_RTE_RRTMGP))
    assert imported.wrf_rrtmg_compatibility == WRF_RRTMG_TO_RTE_RRTMGP

    with pytest.raises(ValueError,
                       match="the token is the receipt of the RRTMG 4/4"):
        validate_run_config(_cfg(
            wrf_rrtmg_compatibility=WRF_RRTMG_TO_RTE_RRTMGP))
    with pytest.raises(ValueError, match="wrf_rrtmg_compatibility must"):
        validate_run_config(_cfg(wrf_rrtmg_compatibility="approximate"))


def test_rrtmg_adapter_rejects_cloud_coupling_off():
    """The refusal stands, and it names an EVIDENCE gap, not a physics one.

    The old assertion matched "always on", the phrase that made this an
    option-envelope policy: icloud is WRF's clear-sky switch, woof
    honours it on the 1/1 pair, and the legacy preparation carries both
    of its arms.  What the 4/4 pair lacks is a recorded oracle case at
    icloud=0, so the message has to say which kind of gap it is and where
    the switch does work -- otherwise the next reader reads "not
    implemented" and deletes a refusal that is protecting a receipt.
    """

    with pytest.raises(ValueError) as caught:
        validate_run_config(_cfg(ra_physics=4, icloud=0))
    message = str(caught.value)
    assert "evidence gap" in message
    assert "oracle case at icloud=0" in message
    assert "ra_lw_physics=1" in message and "ra_sw_physics=1" in message


def test_current_noah_contract_rejects_non_four_layer_state():
    with pytest.raises(ValueError, match="must be 4"):
        validate_run_config(_cfg(sf_surface_physics=2, num_soil_layers=9))


def test_a_half_named_component_is_told_which_keys_it_left_out():
    """The message stops printing a value nobody wrote.

    A component is selected by its whole selector tuple, and a mapping
    that names SOME of a component's keys used to be refused with
    ``{'ra_lw_physics': 4, 'ra_sw_physics': None}`` -- the None is this
    resolver's own filler, read back to the user as if they had typed it.
    Selector keys carry no registry default on purpose (filling one in
    would substitute a scheme nobody asked for), so the fix is a distinct
    message naming the keys that carry no value, not a substitution.
    """

    from woof.physics_compat import _resolve_physics_component_options

    with pytest.raises(PhysicsCapabilityError) as caught:
        _resolve_physics_component_options({"ra_lw_physics": 4})
    message = str(caught.value)
    assert "'ra_sw_physics'" in message
    assert "TOGETHER" in message
    assert "None" not in message


def test_an_unimplemented_selector_value_is_told_what_is_implemented():
    """A refusal that lists the values that DO resolve.

    The tuple is refused because no implemented option carries it; the
    reader's next question is what does, and the registry knows.
    """

    from woof.physics_compat import _resolve_physics_component_options

    with pytest.raises(PhysicsCapabilityError) as caught:
        _resolve_physics_component_options(
            # 7/7 (Fu-Liou-Gu) is ported now; 24/24 (fast RRTMG) is not.
            {"ra_lw_physics": 24, "ra_sw_physics": 24})
    message = str(caught.value)
    assert "Implemented selector tuples for this component" in message
    assert "'ra_lw_physics': 4" in message
