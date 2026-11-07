"""Selector-VALUE dispatch: no scheme may stand in for another.

The physics driver used to branch on the *truthiness* of
``sf_surface_physics`` and ``bl_pbl_physics``.  Unlocking a selector without
adding a runner would therefore have run Noah under a RUC or Noah-MP request
and YSU under a MYNN request -- plausible numbers, wrong scheme, no error.
These are the negative controls for that, plus the admission seams that must
stay fail-closed until each port lands.

The tests are pure Python and never touch a device: ``woof.core.physics``
imports cupy, but ``_cpu_driver`` rebinds the module's ``cp`` to numpy and
constructs the driver with ``object.__new__``, so no array is ever allocated
on a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import (LAND_SURFACE_SCHEMES, LAND_SURFACE_SOIL_LAYERS,
                          PBL_SCHEMES, RunConfig, SURFACE_LAYER_SCHEMES,
                          soil_layer_count, validate_run_config)
from woof.physics_compat import UnsupportedPhysicsSuiteError


def _cfg(**overrides) -> RunConfig:
    values = dict(nx=4, ny=3, nz=4, dx=1000.0, dy=1000.0,
                  ztop=10000.0, dt=5.0, run_seconds=10.0)
    values.update(overrides)
    return RunConfig(**values)


# ---------------------------------------------------------------------------
# (a) The dispatch table itself
# ---------------------------------------------------------------------------

def test_dispatch_table_routes_only_values_with_a_real_runner():
    from woof.core.physics import PHYSICS_SLOT_DISPATCH

    assert PHYSICS_SLOT_DISPATCH == {
        # 2 is the Eta similarity surface layer, routed to its OWN runner
        # by the MYJ port rather than to a fourth _run_sfclay arm: it
        # publishes AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 and none of the MM5
        # layers' MOL/ZOL/PSIM/PSIH, which is why it is admitted only
        # beside bl_pbl_physics=2.
        "sf_sfclay_physics": {0: None, 1: "_run_sfclay",
                              2: "_run_myj_sfclay", 3: "_run_gfs_sfclay",
                              4: "_run_myj_sfclay", 5: "_run_sfclay",
                              7: "_run_px_sfclay",
                              10: "_run_temf_sfclay",
                              # 89 is WRF 4.7.1's prescribed SCM surface
                              # layer (af5ca0c3d59), its own runner.
                              89: "_run_ideal_scm_sfclay",
                              91: "_run_sfclay"},
        "sf_surface_physics": {0: None, 1: "_run_slab", 2: "_run_noah", 3: "_run_ruc",
                               4: "_run_noahmp", 7: "_run_pxlsm", 8: "_run_column_land_surface"},
        # 2 is MYJ, routed by the MYJ port (implemented-unverified: a
        # transcription of the byte-frozen module_bl_myjpbl.F with no
        # oracle comparison run yet); 11 is Shin-Hong, routed to its own
        # runner by the Phase-D wiring (max ULP 0 conformance against the
        # byte-frozen WRF v4.6.1 module); 900 is SASE, the one ArWen-only
        # scheme: it takes a value outside WRF's namespace precisely so
        # this table can route it without ever shadowing a WRF selector.
        # 9 is the UW moist-turbulence PBL (WRF v4.7.1 CAMUWPBL), routed
        # to its own runner and graded against tools/uwpbl_wrf471_oracle.
        # 3 is the GFS nonlocal PBL (WRF v4.6.1 module_bl_gfs.F), routed
        # to its own runner and graded against tools/gfs_pbl_wrf461_oracle.
        "bl_pbl_physics": {0: None, 1: "_run_ysu", 2: "_run_myj_pbl",
                           3: "_run_gfs_pbl", 4: "_run_qnse_pbl", 8: "_run_boulac",
                           5: "_run_mynn_pbl", 7: "_run_acm2", 9: "_run_uwpbl",
                           10: "_run_temf_pbl", 11: "_run_shinhong",
                           12: "_run_gbmpbl", 16: "_run_eeps",
                           99: "_run_mrf", 900: "_run_sase",
                           17: "_run_keps",
                           901: "_run_satmedmf"},
        # 1 is WRF's one-dimensional ocean mixed layer (OML1D), graded
        # against WRF's own column module.
        # 2 is WRF's 3-D PWP ocean (development branch, aa6bdeed308), run by
        # the same surface-ocean owner as the Pollard mixed layer.
        "sf_ocean_physics": {0: None, 1: "_run_surface_ocean",
                             2: "_run_surface_ocean"},
    }


@pytest.mark.parametrize("selector,value", [
    # RUC's 3 used to be the only row here and is now routed to _run_ruc, so
    # the gate moves to WRF selector values woof still has no runner for
    # rather than being deleted with the scheme it was written for.  These are
    # all real WRF v4.6.1 options: 7 is PX LSM, 8 is SSiB and 6 is MYNN3.
    # The five-layer slab left this list with its native CUDA port.
    # bl_pbl_physics=2 (MYJ) left
    # this list with the MYJ port: it now has a runner, so keeping it here
    # would pin the absence of a scheme that is present.
    ("sf_surface_physics", 5),
    ("bl_pbl_physics", 6),
    ("sf_sfclay_physics", 6),
])
def test_unrouted_selector_never_resolves_to_another_scheme(selector, value):
    from woof.core.physics import (UnroutedPhysicsSelectorError,
                                    resolve_physics_slot)

    with pytest.raises(UnroutedPhysicsSelectorError) as caught:
        resolve_physics_slot(selector, value)
    message = str(caught.value)
    assert "PHYSICS_SLOT_DISPATCH" in message
    # The refusal must not read as "unsupported, so we used the default".
    assert "refusing to substitute" in message
    assert caught.value.selector == selector
    assert caught.value.value == value


def test_routed_values_resolve_to_their_own_runner():
    from woof.core.physics import resolve_physics_slot

    assert resolve_physics_slot("sf_sfclay_physics", 0) is None
    assert resolve_physics_slot("sf_sfclay_physics", 1) == "_run_sfclay"
    assert resolve_physics_slot("sf_sfclay_physics", 5) == "_run_sfclay"
    assert resolve_physics_slot("sf_sfclay_physics", 91) == "_run_sfclay"
    assert resolve_physics_slot("sf_surface_physics", 1) == "_run_slab"
    assert resolve_physics_slot("sf_surface_physics", 2) == "_run_noah"
    assert resolve_physics_slot("sf_surface_physics", 3) == "_run_ruc"
    assert resolve_physics_slot("sf_surface_physics", 4) == "_run_noahmp"
    assert resolve_physics_slot("bl_pbl_physics", 1) == "_run_ysu"
    assert resolve_physics_slot("bl_pbl_physics", 5) == "_run_mynn_pbl"
    assert resolve_physics_slot("bl_pbl_physics", 11) == "_run_shinhong"
    assert resolve_physics_slot("bl_pbl_physics", 901) == "_run_satmedmf"


# ---------------------------------------------------------------------------
# (b) The driver actually dispatching -- the negative controls
# ---------------------------------------------------------------------------

def _cpu_driver(monkeypatch, *, sf_sfclay_physics, sf_surface_physics,
                bl_pbl_physics):
    """Minimal CPU-only ``PhysicsDriver.compute`` harness (no device)."""
    import woof.core.physics as physics

    monkeypatch.setattr(physics, "cp", np)
    # The lowest-level vapour and the land mask feed WRF's final land Q2
    # cap (module_surface_driver.F:4443-4457), which compute() applies
    # after every surface layer since development branch (f6eb30981);
    # production's _prepare_atmosphere and fields always carry both.
    atmosphere = {
        "pressure": np.full((1, 1, 2), 95000.0, np.float32),
        "p_interface": np.array(
            [[[100000.0, 90000.0]], [[95000.0, 85000.0]]], np.float32),
        "qv": np.full((1, 1, 2), 0.01, np.float32),
    }
    monkeypatch.setattr(
        physics, "_prepare_atmosphere", lambda state: atmosphere)
    state = SimpleNamespace(elapsed_seconds=0.0, c1h=np.ones(1, np.float32),
                            c2h=np.zeros(1, np.float32),
                            total_mu=lambda: np.ones((1, 2), np.float32))
    cfg = RunConfig(nx=2, ny=1, nz=1, dx=1000., dy=1000.,
                    ztop=10000., run_seconds=0.,
        dt=60.0, bldt=0.0, ra_physics=0, ra_rrtmg_variant="rte-rrtmgp",
        sst_update=0, sst_skin=0, tmn_update=0,
        sf_sfclay_physics=sf_sfclay_physics,
        sf_surface_physics=sf_surface_physics,
        bl_pbl_physics=bl_pbl_physics, cu_physics=0,
        # RunConfig's default; compute() reads it for the RUC Q2 cap
        # since 614632b77.
        ruc_2m_diagnostic="flux")
    driver = object.__new__(physics.PhysicsDriver)
    driver.farms = None
    # The TC high-wind surface exchange (TC fold): off unless a TC run asks.
    driver.tc_surface = None
    driver.fog_settling = None
    driver.ruc_params = None
    driver.pxlsm_params = None
    driver._wrf_rthraten = None
    driver._wrf_rthraten_coupled = True
    driver.swint = None
    driver.state = state
    driver.fields = {
        name: np.full((1, 2), value, np.float32)
        for name, value in {
            "p_hyd": 95000.0, "psfc": -123.0, "tsk": 290.0, "hfx": 0.0, "qfx": 0.0,
            "qsfc": 0.0, "chs2": 0.0, "cqs2": 0.0,
            "t2": -999.0, "q2": -999.0, "th2": -999.0, "xland": 1.0,
        }.items()
    }
    for name in physics.PBL_HISTORY_RATES:
        driver.fields[name] = np.zeros((1, 1, 2), np.float32)
    for name, source in physics.SURFACE_ENERGY_ACCUMULATORS.items():
        for field in (name, source):
            driver.fields.setdefault(field, np.zeros((1, 2), np.float32))
    driver.surface_enabled = True
    # WRF swint_opt = 1's carrier (89aba028e): the real constructor leaves
    # this slot None when swint_opt is 0, and compute() reads it every step.
    driver.swint = None
    driver.stepbl = 1
    driver.radt_minutes = 12.0
    driver.radt_seconds = 720.0
    driver.cudt_minutes = 5.0
    driver.call_counts = {
        "radiation": 0, "sfclay": 0, "noah": 0, "ysu": 0,
        "cumulus": 0, "cumulus_history": 0,
    }
    driver.tendencies = object()
    driver._compose_tendencies = lambda cfg_arg: None
    # THE CARRIER CONTRACT is a slot compute() reads before it dispatches
    # to a land-surface runner, so a hand-built driver declares one like
    # every other slot above.  Seeded as a run with live radiation is
    # seeded -- every carrier this scheme reads, produced now -- because
    # this file is about WHICH RUNNER a selector reaches and not about
    # whether the sky is real.  The refusal itself, on a driver assembled
    # WITHOUT a contract, is pinned in
    # tests/test_surface_radiation_carrier_contract.py; if it were pinned
    # here instead, every dispatch assertion below would stop running.
    from woof.core import radiation_carriers
    driver.carriers = radiation_carriers.CarrierContract()
    for carrier in radiation_carriers.consumer_carriers(sf_surface_physics):
        driver.carriers.declare(
            carrier,
            source=radiation_carriers.CARRIER_SOURCE_RADIATION_SCHEME,
            model_time=0.0)
    return physics, state, cfg, driver


def test_selecting_ruc_does_not_run_noah(monkeypatch):
    """sf_surface_physics=3 now runs, and what it runs must be RUC.

    The refusal this test used to assert has been replaced by a routed
    runner, exactly as happened for Noah-MP and MYNN below, so the negative
    control moves with it rather than being deleted: the selector must reach
    ``_run_ruc`` and Noah must stay untouched.  This is the original sin the
    file was written for -- ``compute`` once branched on the TRUTHINESS of
    ``sf_surface_physics``, so a RUC request ran Noah and produced a
    complete, plausible, finite forecast with no error.

    ``tests/test_ruc_runtime.py`` carries the same guarantee against the real
    driver, with ``launch_noah`` itself booby-trapped rather than the
    driver's bound method replaced; this one is the cheap CPU-only half that
    runs without a GPU.
    """
    physics, state, cfg, driver = _cpu_driver(
        monkeypatch, sf_sfclay_physics=91, sf_surface_physics=3,
        bl_pbl_physics=1)
    seen = {}
    driver._run_noah = lambda *_: pytest.fail(
        "RUC (sf_surface_physics=3) must never run Noah")
    driver._run_noahmp = lambda *_: pytest.fail(
        "RUC (sf_surface_physics=3) must never run Noah-MP")
    driver._run_sfclay = lambda *_: None
    driver._run_ysu = lambda *_: None
    driver._run_ruc = lambda atmosphere, cfg_arg, itimestep: seen.update(
        option=cfg_arg.sf_surface_physics)

    driver.compute(state, cfg)

    assert seen == {"option": 3}
    assert driver.call_counts["noah"] == 1   # the shared LSM-call counter


def test_selecting_noahmp_does_not_run_noah(monkeypatch):
    """sf_surface_physics=4 now runs, and what it runs must be Noah-MP.

    The refusal this test used to assert has been replaced by a routed
    runner, exactly as happened for MYNN below, so the negative control moves
    with it: the selector must reach ``_run_noahmp`` and Noah must stay
    untouched.  A dispatch that fell through to Noah would still produce
    plausible fluxes and a plausible soil column, which is the failure mode
    this file exists to prevent.
    """
    physics, state, cfg, driver = _cpu_driver(
        monkeypatch, sf_sfclay_physics=91, sf_surface_physics=4,
        bl_pbl_physics=1)
    seen = {}
    driver._run_noah = lambda *_: pytest.fail(
        "Noah-MP (sf_surface_physics=4) must never run Noah")
    driver._run_sfclay = lambda *_: None
    driver._run_ysu = lambda *_: None
    driver._run_noahmp = lambda atmosphere, cfg_arg, itimestep: seen.update(
        option=cfg_arg.sf_surface_physics)

    driver.compute(state, cfg)

    assert seen == {"option": 4}
    assert driver.call_counts["noah"] == 1   # the shared LSM-call counter


def test_selecting_mynn_pbl_does_not_run_ysu(monkeypatch):
    """bl_pbl_physics=5 now runs, and what it runs must be MYNN.

    The refusal this test used to assert has been replaced by a routed
    runner, so the negative control has to move with it: the selector must
    reach ``_run_mynn_pbl`` and YSU must stay untouched.  A dispatch that
    fell through to YSU would still produce plausible tendencies, which is
    exactly the failure mode this file exists to prevent.
    """
    physics, state, cfg, driver = _cpu_driver(
        monkeypatch, sf_sfclay_physics=5, sf_surface_physics=0,
        bl_pbl_physics=5)
    seen = {}
    driver._run_sfclay = lambda *_: None
    driver._run_noah = lambda *_: pytest.fail("no LSM is selected")
    driver._run_ysu = lambda *_: pytest.fail(
        "MYNN (bl_pbl_physics=5) must never run YSU")
    driver._run_mynn_pbl = lambda atmosphere, cfg_arg: seen.update(
        option=cfg_arg.bl_pbl_physics)

    driver.compute(state, cfg)

    assert seen == {"option": 5}
    assert driver.call_counts["ysu"] == 1   # the shared PBL-call counter


def test_mynn_surface_layer_value_routes_to_the_mynn_branch(monkeypatch):
    """sf_sfclay_physics=5 reaches _run_sfclay, and _run_sfclay honours 5."""
    physics, state, cfg, driver = _cpu_driver(
        monkeypatch, sf_sfclay_physics=5, sf_surface_physics=0,
        bl_pbl_physics=0)
    seen = {}
    driver._run_sfclay = lambda atmosphere, cfg_arg: seen.update(
        option=cfg_arg.sf_sfclay_physics)
    driver._run_noah = lambda *_: pytest.fail("no LSM is selected")
    driver.pbl_tendencies = None
    monkeypatch.setattr(
        physics.PhysicsTendencies, "zeros",
        staticmethod(lambda *args, **kwargs: object()))

    driver.compute(state, cfg)

    assert seen == {"option": 5}
    assert driver.call_counts["sfclay"] == 1
    assert driver.call_counts["noah"] == 0
    assert driver.call_counts["ysu"] == 0


def test_run_sfclay_refuses_an_unknown_surface_layer_value(monkeypatch):
    """The MM5 launcher must not be the fallthrough for any new value."""
    physics, _state, cfg, driver = _cpu_driver(
        monkeypatch, sf_sfclay_physics=2, sf_surface_physics=0,
        bl_pbl_physics=0)
    driver.mynn_sfclay_result = None
    monkeypatch.setattr(physics, "launch_sfclay", lambda *a, **k: pytest.fail(
        "an unrouted surface-layer value must not reach the MM5 launcher"))
    with pytest.raises(physics.UnroutedPhysicsSelectorError):
        driver._run_sfclay({}, cfg)


# Gate successor: test_noah_sfcdiags_is_scheme_specific; the assertions cover the admitted behavior.
def test_noah_and_slab_sfcdiags_are_scheme_specific():
    from woof.core.physics import LAND_SURFACE_SFCDIAGS_SCHEMES

    # WRF calls SFCDIAGS after Noah and slab. MYNN's surface layer diagnoses
    # T2/Q2/TH2 itself, so a new LSM must opt in rather than inherit this.
    assert LAND_SURFACE_SFCDIAGS_SCHEMES == frozenset({1, 2})


# ---------------------------------------------------------------------------
# (c) Admission seams stay fail-closed, with a receipt that names the gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("overrides,component", [
    # THE ROWS MOVED OFF SOIL GEOMETRY ENTIRELY, in two steps on this branch.
    #
    # First the {6, 9} land-surface geometry lift renamed what these rows
    # asserted: the blocker they named, "RUC soil geometry", stopped being
    # true of six levels once woof.core.ruc.ruc_soil_geometry became
    # parametric and oracle-matched at both counts, so the rows were
    # repointed at the narrower "RUC forecast column soil geometry" -- the
    # CUDA column's own nine-level pin -- and went green there.
    #
    # THE RUC_NZS LIFT THEN RETIRED THAT SECOND BLOCKER TOO, so there is no
    # longer any soil-geometry refusal for these two configs to receive.  The
    # rows cannot simply be repointed a third time either: measured while
    # retiring it, woof.config's soil-layer resolver refuses EVERY geometry
    # mismatch before validate_run_config reaches the physics-suite check, so
    # on THIS path a bad num_soil_layers is a ValueError from the resolver
    # and the soil-geometry blockers are not reachable at all.  The one that
    # survives -- WRF's own refusal outside {6, 9} -- is asserted directly
    # against pending_wrf_physics_components in
    # test_a_geometry_wrf_does_not_tabulate_is_still_blocked below, which is
    # the surface where it does fire.
    #
    # What remains here is a coherent WRF request woof refuses through the
    # suite check, which is what this test is actually for: the RECEIPT.
    (dict(sf_sfclay_physics=2, bl_pbl_physics=1),
     "WRF v4.6.1 PBL/surface-layer compatibility"),
    (dict(sf_sfclay_physics=5, bl_pbl_physics=2),
     "WRF v4.6.1 PBL/surface-layer compatibility"),
])
def test_in_port_schemes_fail_closed_with_a_port_receipt(overrides, component):
    from woof.config import ConfigurationRefused
    with pytest.raises(ConfigurationRefused) as caught:
        validate_run_config(_cfg(**overrides))
    text = str(caught.value)
    assert component == "WRF v4.6.1 PBL/surface-layer compatibility"
    assert "sf_sfclay_physics=" in text and "bl_pbl_physics=" in text
    assert "requires" in text or "is admitted with" in text
    # The receipt naming the DRIVER gate rather than only the science used to
    # be asserted here, because RUC's own blocker text cited
    # PHYSICS_SLOT_DISPATCH as one of its missing pieces.  It has that row
    # now, so claiming otherwise would be false.  The property has not been
    # dropped: it moves to the unrouted-value receipt, which is where the
    # driver gate is the actual reason, and is asserted here so the coverage
    # stays in one place instead of quietly leaving the file.
    from woof.core.physics import (UnroutedPhysicsSelectorError,
                                    resolve_physics_slot)
    with pytest.raises(UnroutedPhysicsSelectorError) as unrouted:
        resolve_physics_slot("sf_surface_physics", 5)
    assert "PHYSICS_SLOT_DISPATCH" in str(unrouted.value)
    assert "refusing to substitute" in str(unrouted.value)


@pytest.mark.parametrize("selectors,component", [
    # WRF's OWN refusal: init_soil_depth_3 tabulates zs for 6 and 9 and
    # leaves it uninitialised otherwise, and is fatal at 4 and 5.
    (dict(sf_surface_physics=3, num_soil_layers=4), "RUC soil geometry"),
    (dict(sf_surface_physics=3, num_soil_layers=5), "RUC soil geometry"),
    # Noah-MP's geometry is a woof fixture statement, and unchanged here.
    (dict(sf_surface_physics=4, num_soil_layers=9), "Noah-MP soil geometry"),
    (dict(sf_surface_physics=4, num_soil_layers=6), "Noah-MP soil geometry"),
])
def test_a_geometry_wrf_does_not_tabulate_is_still_blocked(selectors, component):
    """The soil-geometry blockers, asserted where they actually fire.

    The RUC_NZS lift retired ONE of the two RUC soil blockers -- the forecast
    column's nine-level pin -- and deliberately left the other untouched.
    This is the half that survives, plus Noah-MP's, checked directly against
    pending_wrf_physics_components because validate_run_config's own
    soil-layer resolver refuses a bad geometry before the suite check runs.

    Six levels is NOT in this list any more, and that is the point: it is an
    admitted geometry that warns rather than blocks.
    """
    from woof.physics_compat import pending_wrf_physics_components

    blockers = pending_wrf_physics_components(
        mp_physics=8, sf_sfclay_physics=91, bl_pbl_physics=1, **selectors)
    assert component in [item.component for item in blockers]


def test_six_soil_levels_is_no_longer_a_blocker():
    """The retirement itself, stated as a test rather than an absence.

    A guard that is deleted and not replaced leaves nothing that fails if it
    comes back by accident.
    """
    from woof.physics_compat import pending_wrf_physics_components

    blockers = pending_wrf_physics_components(
        mp_physics=8, sf_sfclay_physics=91, bl_pbl_physics=1,
        sf_surface_physics=3, num_soil_layers=6)
    assert blockers == ()
    assert soil_layer_count(_cfg(sf_surface_physics=3,
                                 num_soil_layers=6)) == 6


@pytest.mark.parametrize("overrides", [
    dict(sf_sfclay_physics=5, bl_pbl_physics=1),
    dict(sf_sfclay_physics=0, bl_pbl_physics=5),
])
def test_wrf_fatal_pbl_surface_layer_cells_are_refused(overrides):
    with pytest.raises(UnsupportedPhysicsSuiteError) as caught:
        validate_run_config(_cfg(**overrides))
    assert [item.component for item in caught.value.blockers] == [
        "WRF v4.6.1 PBL/surface-layer compatibility"]
    assert "phys/module_physics_init.F:" in str(caught.value)
    assert "no substitutions were applied" in str(caught.value)


@pytest.mark.parametrize("surface,pbl", [
    (5, 5),
    (1, 5),
    (91, 5),
    (5, 0),
])
def test_wrf_legal_mynn_pairings_are_admitted(surface, pbl):
    validate_run_config(_cfg(
        sf_sfclay_physics=surface, bl_pbl_physics=pbl, nz=5))


def test_noahmp_is_admitted_at_four_soil_layers_and_only_there():
    """The receipt that replaced the blanket Noah-MP refusal.

    sf_surface_physics=4 no longer raises: it has a dispatch row, a driver, a
    cold start, restart identity and output.  What is still refused is a soil
    geometry no Noah-MP fixture covers, and the refusal has to name that
    rather than the scheme.
    """
    from woof.physics_compat import pending_wrf_physics_components

    # The surface layer is carried explicitly: a land-surface model without
    # one is refused by validate_run_config and by woof/core/physics.py
    # initialize_physics alike (Noah reads CHS/CHS2/CQS2/QGH/RIB from it and
    # Noah-MP reads the same seam), so a bare sf_surface_physics=4 config was
    # never a runnable one and is not what this test is about.
    validate_run_config(_cfg(sf_surface_physics=4, sf_sfclay_physics=91))
    # Through validate_run_config the schema table refuses it first, because
    # a nine-layer Noah-MP is not a coherent WRF request in the first place.
    with pytest.raises(ValueError, match="num_soil_layers must be 4"):
        validate_run_config(_cfg(sf_surface_physics=4, sf_sfclay_physics=91,
                                 num_soil_layers=9))
    # The readiness authority still carries its own row, because
    # woof/namelist_import.py consults it on a RAW namelist request that has
    # not been through the schema table.
    blockers = pending_wrf_physics_components(
        mp_physics=6, sf_sfclay_physics=91, bl_pbl_physics=1,
        sf_surface_physics=4, num_soil_layers=9)
    assert [item.component for item in blockers] == ["Noah-MP soil geometry"]
    assert pending_wrf_physics_components(
        mp_physics=6, sf_sfclay_physics=91, bl_pbl_physics=1,
        sf_surface_physics=4, num_soil_layers=4) == ()


def test_schema_tables_and_soil_geometry():
    # 2 (Eta similarity) joined with the MYJ port, together with
    # bl_pbl_physics=2 below: the two are admitted as a PAIR and
    # validate_myj_pairing refuses either half alone.
    # 89 joined with the ideal single-column surface layer (idealscmsfclay).
    assert SURFACE_LAYER_SCHEMES == (0, 1, 2, 3, 4, 5, 7, 10, 89, 91)
    assert LAND_SURFACE_SCHEMES == (0, 1, 2, 3, 4, 7, 8)
    # Pin moved (0, 1, 5, 900) -> (0, 1, 5, 11, 900) by the Phase-D
    # Shin-Hong runtime wiring: 11 is WRF's own bl_pbl_physics number for
    # module_bl_shinhong.F, admitted with a routed runner, restart
    # identity and preflight rows in the same change -- exactly the one
    # widening the registry admission (a5e101c5) enumerated.  900 is
    # SASE: ArWen-only, deliberately outside WRF's namespace (which runs
    # to 99) so it can never collide with a scheme WRF adds later.
    # 9 joined with the UW moist-turbulence port (WRF v4.7.1
    # module_bl_camuwpbl_driver.F): a routed runner, a restart identity,
    # a registry option and a namelist-import row in the same change.
    # 901 is the non-WRF CCPP closure, with its own runner and asset identity.
    assert PBL_SCHEMES == (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 16, 17, 99, 900, 901)
    assert LAND_SURFACE_SOIL_LAYERS[2] == (4,)
    assert LAND_SURFACE_SOIL_LAYERS[3] == (6, 9)
    assert LAND_SURFACE_SOIL_LAYERS[4] == (4,)
    assert soil_layer_count(_cfg()) == 4

    # Noah keeps its exact four-layer refusal message.
    with pytest.raises(ValueError, match="must be 4"):
        validate_run_config(_cfg(sf_surface_physics=2, num_soil_layers=9))
    # RUC's geometry is checked against RUC's own table, and the port
    # receipt still refuses the run.
    with pytest.raises(ValueError, match="must be 6 or 9"):
        validate_run_config(_cfg(sf_surface_physics=3, num_soil_layers=4))
    # A value in no scheme's schema is still a hard error.
    with pytest.raises(ValueError, match="sf_surface_physics must be"):
        validate_run_config(_cfg(sf_surface_physics=5))
    with pytest.raises(ValueError, match="bl_pbl_physics must be"):
        validate_run_config(_cfg(bl_pbl_physics=6))
    with pytest.raises(ValueError, match="sf_sfclay_physics must be"):
        validate_run_config(_cfg(sf_sfclay_physics=6))
    # 2/2 is now IN the schema, so what refuses a half-suite is the
    # pairing law rather than the schema tables.  Pinned here so the two
    # refusals cannot be confused for each other.
    with pytest.raises(ValueError, match="requires sf_sfclay_physics=2"):
        validate_run_config(_cfg(bl_pbl_physics=2))
    with pytest.raises(ValueError,
                       match="admitted with bl_pbl_physics=2"):
        validate_run_config(_cfg(sf_sfclay_physics=2))


def test_noah_only_options_require_the_noah_selector():
    # usemonalb/rdlai2d are read by Noah AND RUC; with no LSM the refusal
    # names both schemes that do read them.
    with pytest.raises(ValueError, match=r"Noah LSM \(sf_surface_physics=2\) "
                                         r"and the RUC LSM \(sf_surface_physics=3\)"):
        validate_run_config(_cfg(sf_surface_physics=0, usemonalb=True))
    with pytest.raises(ValueError, match=r"Noah LSM \(sf_surface_physics=2\) "
                                         r"and the RUC LSM \(sf_surface_physics=3\)"):
        validate_run_config(_cfg(sf_surface_physics=0, rdlai2d=True))
    # opt_thcnd stays Noah only.
    with pytest.raises(ValueError, match="opt_thcnd is a Noah LSM option"):
        validate_run_config(_cfg(sf_surface_physics=0, opt_thcnd=2))
    with pytest.raises(ValueError, match="require sf_sfclay_physics=1 or 91"):
        validate_run_config(_cfg(sf_sfclay_physics=0, isftcflx=1))


# ---------------------------------------------------------------------------
# (d) Restart identity + VRAM inventory reach the in-port schemes
# ---------------------------------------------------------------------------

def test_restart_identities_exist_for_every_schema_selector():
    from woof.io import restart

    assert set(restart.SURFACE_LAYER_ALGORITHM_IDENTITIES) == \
        set(SURFACE_LAYER_SCHEMES)
    assert set(restart.LAND_SURFACE_ALGORITHM_IDENTITIES) == \
        set(LAND_SURFACE_SCHEMES)
    assert set(restart.PBL_ALGORITHM_IDENTITIES) == set(PBL_SCHEMES)
    # An active LSM without a parameter-bundle row cannot checkpoint: the
    # file would otherwise claim Noah's tables for a run that never used them.
    assert set(restart.LAND_SURFACE_PARAMETER_SOURCES) == {1, 2, 3, 4, 7, 8}
    assert restart.LAND_SURFACE_PARAMETER_SOURCES[2][0] == "noah_params"
    assert restart.LAND_SURFACE_PARAMETER_SOURCES[3][0] == "ruc_params"
    assert restart.LAND_SURFACE_PARAMETER_SOURCES[4][0] == "noahmp_params"
    # RUC reads the RUC SECTIONS of the same three files Noah reads, so the
    # asset roles overlap while the bundle object does not.  LANDUSE.TBL is
    # deliberately absent: woof.core.ruc never opens it, because RUC's
    # roughness, albedo and emissivity come from its own VEGPARM rows.
    assert restart.LAND_SURFACE_PARAMETER_SOURCES[3][1] == (
        "noah_vegparm", "noah_soilparm", "noah_genparm")
    assert "scheme_dispatch" in restart.DRIVER_REBUILT_ATTRS


def test_land_surface_parameters_never_fall_through_to_noahs_bundle():
    """A checkpoint must not claim Noah's tables for a non-Noah run."""
    import dataclasses

    from woof.io import restart

    @dataclasses.dataclass
    class _Params:
        a: float = 1.5
        b: int = 3

    driver = SimpleNamespace(noah_params=_Params())
    noah = restart._land_surface_parameters_identity(
        SimpleNamespace(ua_phys=False, sf_surface_physics=2), driver)
    assert noah["values"] == {"a": 1.5, "b": 3}
    assert set(noah) == {"class", "arrays", "values", "sha256"}

    # A scheme with NO row at all still cannot checkpoint.  7 is PX LSM, a
    # real WRF option woof has no runner and no parameter row for; 3 used to
    # stand here and now has both, so the gate moves rather than relaxes.
    with pytest.raises(restart.RestartManifestError,
                       match="LAND_SURFACE_PARAMETER_SOURCES"):
        restart._land_surface_parameters_identity(
            SimpleNamespace(ua_phys=False, sf_surface_physics=5), driver)

    # And RUC, which HAS a row, still refuses when the driver carries no RUC
    # bundle -- a checkpoint must not fall through to the Noah params that
    # happen to be attached to this stub driver.
    with pytest.raises(restart.RestartManifestError,
                       match="PhysicsDriver.ruc_params"):
        restart._land_surface_parameters_identity(
            SimpleNamespace(ua_phys=False, sf_surface_physics=3, ruc_params=None), driver)


def test_asset_identity_binds_only_the_active_lsm_tables():
    from woof.io import restart

    def _cfg_for(scheme):
        return SimpleNamespace(
            sf_surface_physics=scheme, cu_physics=0, ra_physics=0,
            ra_lw_physics=-1, ra_sw_physics=-1)

    assert sorted(restart._active_asset_identity(_cfg_for(2), None)) == [
        "noah_genparm", "noah_landuse", "noah_soilparm", "noah_vegparm"]
    assert restart._active_asset_identity(_cfg_for(0), None) == {}
    # RUC binds three roles, not four: the RUC sections of VEGPARM and
    # SOILPARM plus GENPARM.  LANDUSE.TBL is Noah-only.
    assert sorted(restart._active_asset_identity(_cfg_for(3), None)) == [
        "noah_genparm", "noah_soilparm", "noah_vegparm"]
    # CLM (5) has no admitted parameter-source row.
    with pytest.raises(restart.RestartManifestError,
                       match="LAND_SURFACE_PARAMETER_SOURCES"):
        restart._active_asset_identity(_cfg_for(5), None)


def test_vram_inventory_counts_the_mynn_surface_allocations():
    from woof.core import preflight as pf
    from woof.core.mynn_sfclay import MYNN_SURFACE_OUTPUTS

    base = set(pf.physics_field_names_2d())
    mynn = set(pf.physics_field_names_2d(_cfg(sf_sfclay_physics=5)))
    extra = set(MYNN_SURFACE_OUTPUTS) - base
    assert extra, "MYNN must add at least one persistent surface field"
    assert extra <= mynn
    # ... and only when MYNN is selected: carrying them in every MM5 run
    # would change the restart inventory of unrelated configurations.
    assert not (extra & base)


def test_ccpp_asset_identity_binds_only_the_active_pbl_table():
    from woof.io import restart

    inactive = _cfg(sf_surface_physics=0, cu_physics=0, ra_physics=0,
                    bl_pbl_physics=0)
    active = _cfg(sf_surface_physics=0, cu_physics=0, ra_physics=0,
                  bl_pbl_physics=901)
    assert restart._active_asset_identity(inactive, None) == {}
    assert set(restart._active_asset_identity(active, None)) == {"satmedmf_manifest"}
