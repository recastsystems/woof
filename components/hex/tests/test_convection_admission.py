"""Convection admission: the scheme is OFF below 3 km, by default.

It was ruled on 2026-08-26 that convection is switched off below 3 km,
answering directly whether a fine swath runs the cumulus scheme at all.
This is that ruling, made executable.

**Why it is a gate and not a preference.**  THE BREAKAGE THIS PREVENTS,
MEASURED (2026-08-26, `evidence/dt-anchors-20260826/RECEIPT.md`): WRF pins
``cudt = 0`` for ``cu_physics = 3``, so Grell-Freitas recomputes every
model step and no configuration relaxes it.  A 5 s timestep therefore calls
the closure 720 times an hour against the 30 the lane was proven at, and the
5 s anchor measured a |w| mean climbing 12.0 / 49.6 / 73.5 / 87.5 m/s over
four half-hour windows against a 120 s control's 1.15 / 1.17 / 1.20 / 1.48,
with |w| max 102.67 against 1.680.  A 3 km swath declares 20 s and a 750 m
swath declares 5 s, so every mesh the ruling covers runs the closure at
between six and twenty-four times its proven call rate.

**Fixed means default.**  A bare run on a sub-3-km mesh must stop calling
the scheme with nobody passing a flag; the explicit ``--convection`` switch
exists to run A/B arms (that is how the attribution above was measured),
never as the remedy.

**Convection-off is a NEW admitted configuration, not a mutation of the
proven one.**  The frozen v8.4.1 column-physics lane's proven configuration
stays exactly reachable and exactly refusing: how often a scheme is called
is part of what an anchor certifies, so a convection-off run earns its own
anchors and cannot borrow the GF rows'.

Everything here runs on a CPU-only box.
"""

from __future__ import annotations

import dataclasses

import pytest

from woof.hex import convection_admission, dt_admission
from woof.hex.convection_admission import (
    CONVECTION_OFF_BELOW_M,
    ConvectionAdmissionError,
    NTIEDTKE_AT_OR_ABOVE_M,
    SCHEME_GRELL_FREITAS,
    SCHEME_NTIEDTKE,
    SCHEME_OFF,
)
from woof.hex.config_v841 import V841MpasColumnPhysicsConfig
from woof.hex.errors import ConfigurationRefusal
from _layout import PACKAGE_DIR


# ---------------------------------------------------------------------------
# the ruling itself
# ---------------------------------------------------------------------------
def test_the_ruling_switches_convection_off_below_three_kilometres():
    assert CONVECTION_OFF_BELOW_M == 3_000.0
    assert convection_admission.convection_for_spacing(2_999.0) == SCHEME_OFF
    assert convection_admission.convection_for_spacing(750.0) == SCHEME_OFF
    assert convection_admission.convection_for_spacing(1_000.0) == SCHEME_OFF
    # From the threshold up to 20 km the proven configuration is unchanged.
    assert (
        convection_admission.convection_for_spacing(3_000.0) == SCHEME_GRELL_FREITAS
    )
    assert (
        convection_admission.convection_for_spacing(19_999.0) == SCHEME_GRELL_FREITAS
    )


def test_the_2026_10_05_ruling_selects_new_tiedtke_at_twenty_kilometres_and_coarser():
    """Obs-scored on two 72 h x1.40962 cases; see the RECEIPT it names."""

    assert NTIEDTKE_AT_OR_ABOVE_M == 20_000.0
    assert convection_admission.convection_for_spacing(20_000.0) == SCHEME_NTIEDTKE
    assert convection_admission.convection_for_spacing(120_000.0) == SCHEME_NTIEDTKE
    decision = convection_admission.convection_decision(nominal_dx_m=120_000.0)
    assert decision["source"] == "resolution"
    assert decision["ntiedtke_threshold_m"] == NTIEDTKE_AT_OR_ABOVE_M
    assert "2026-10-05" in decision["ntiedtke_ruling"]
    assert "20000 m New Tiedtke threshold" in decision["note"]
    assert "RECEIPT.md" in convection_admission.NTIEDTKE_MEASUREMENT


def test_the_finest_spacing_anywhere_on_the_mesh_decides():
    """A swath is fine somewhere and coarse elsewhere; the fine part rules.

    the question put was about a fine SWATH inside a coarser mesh, so the
    decision is taken on the finest spacing the mesh carries, not on its
    average or its declared nominal alone.
    """

    decision = convection_admission.convection_decision(
        nominal_dx_m=15_000.0, minimum_dc_edge_m=2_800.0
    )
    assert decision["scheme"] == SCHEME_OFF
    assert decision["finest_spacing_m"] == 2_800.0
    assert decision["below_threshold"] is True

    coarse = convection_admission.convection_decision(
        nominal_dx_m=120_000.0, minimum_dc_edge_m=97_076.0
    )
    assert coarse["scheme"] == SCHEME_NTIEDTKE
    assert coarse["finest_spacing_m"] == 97_076.0
    assert coarse["below_threshold"] is False


def test_the_decision_rides_into_a_receipt_naming_the_ruling_and_the_measurement():
    decision = convection_admission.convection_decision(nominal_dx_m=750.0)
    assert decision["threshold_m"] == CONVECTION_OFF_BELOW_M
    assert "2026-08-26" in decision["ruling"]
    # A gate names the concrete breakage it prevents, with the measurement.
    assert "102.67" in decision["breakage"]
    assert "720" in decision["breakage"]
    assert decision["source"] == "resolution"


def test_a_spacing_that_is_not_a_positive_length_is_refused_by_name():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ConvectionAdmissionError) as error:
            convection_admission.convection_for_spacing(bad)
        assert "finite and positive" in str(error.value)


# ---------------------------------------------------------------------------
# the two vocabularies, and the one translation between them
# ---------------------------------------------------------------------------
def test_the_config_name_and_the_sealed_constructor_name_agree():
    assert convection_admission.constructor_scheme(SCHEME_GRELL_FREITAS) == "gf"
    assert convection_admission.constructor_scheme(SCHEME_OFF) is None
    assert convection_admission.constructor_scheme(SCHEME_NTIEDTKE) == "ntiedtke"
    # A name no physics selection resolved has no constructor name; a
    # cumulus option of WOOF's registry gets one when a run selects it
    # (tests/test_arbitrary_physics.py).
    with pytest.raises(ConvectionAdmissionError, match="not a cumulus selection this run resolved"):
        convection_admission.constructor_scheme("cu_never_resolved")


def test_new_tiedtke_below_twenty_kilometres_is_reached_only_by_asking_for_it():
    """The RRFS v2 12 km scheme: 'auto' picks it only at 20 km and coarser.

    Until 2026-10-05 'auto' never chose it; the ruling of that date made it
    the default at 20 km and coarser, so this keeps checking the band where
    it is still only an explicit selection.
    """

    assert SCHEME_NTIEDTKE in convection_admission.ADMITTED_CONVECTION_SCHEMES
    assert convection_admission.label(SCHEME_NTIEDTKE) == "New Tiedtke"
    assert convection_admission.gf_ishallow(SCHEME_NTIEDTKE) == 0
    assert "ntiedtke" in convection_admission.EVERY_STEP_SCHEMES
    for dx in (750.0, 3_000.0, 12_000.0, 19_999.0):
        assert convection_admission.convection_decision(nominal_dx_m=dx)[
            "scheme"
        ] != SCHEME_NTIEDTKE
    chosen = convection_admission.convection_decision(
        nominal_dx_m=12_000.0, minimum_dc_edge_m=10_000.0, requested="ntiedtke"
    )
    assert chosen["scheme"] == SCHEME_NTIEDTKE
    assert chosen["constructor_scheme"] == "ntiedtke"
    assert chosen["source"] == "explicit"
    assert "RRFS v2" in chosen["note"]
    fine = convection_admission.convection_decision(
        nominal_dx_m=2_000.0, requested="ntiedtke"
    )
    assert "OVERRIDES the ruling" in fine["note"]


def test_gf_shallow_is_off_when_there_is_no_gf_to_be_shallow_in():
    """``gf_ishallow=1`` with no GF is refused by the sealed constructor."""

    assert convection_admission.gf_ishallow(SCHEME_GRELL_FREITAS) == 1
    assert convection_admission.gf_ishallow(SCHEME_OFF) == 0


# ---------------------------------------------------------------------------
# fixed means default
# ---------------------------------------------------------------------------
def test_a_bare_run_below_three_kilometres_selects_no_cumulus_scheme():
    """No flag, no argument: the scheme is simply not selected."""

    assert convection_admission.default_convection_scheme(nominal_dx_m=750.0) == (
        SCHEME_OFF
    )
    assert convection_admission.default_convection_scheme(nominal_dx_m=2_500.0) == (
        SCHEME_OFF
    )
    assert convection_admission.default_convection_scheme(
        nominal_dx_m=15_000.0
    ) == SCHEME_GRELL_FREITAS
    assert convection_admission.default_convection_scheme(
        nominal_dx_m=120_000.0
    ) == SCHEME_NTIEDTKE


def test_an_explicit_switch_is_an_ab_arm_and_says_so():
    forced_off = convection_admission.convection_decision(
        nominal_dx_m=120_000.0, requested="off"
    )
    assert forced_off["scheme"] == SCHEME_OFF
    assert forced_off["source"] == "explicit"
    assert "A/B" in forced_off["note"]

    forced_on = convection_admission.convection_decision(
        nominal_dx_m=750.0, requested="gf"
    )
    assert forced_on["scheme"] == SCHEME_GRELL_FREITAS
    assert forced_on["source"] == "explicit"
    # Overriding the ruling on a mesh the ruling covers must say what it is.
    assert "ruling" in forced_on["note"]

    auto = convection_admission.convection_decision(
        nominal_dx_m=120_000.0, requested="auto"
    )
    assert auto["source"] == "resolution"

    with pytest.raises(ConvectionAdmissionError):
        convection_admission.convection_decision(
            nominal_dx_m=120_000.0, requested="sometimes"
        )


# ---------------------------------------------------------------------------
# the frozen configuration: still reachable, still exact
# ---------------------------------------------------------------------------
def test_the_proven_grell_freitas_configuration_is_unchanged():
    config = V841MpasColumnPhysicsConfig()
    config.validate()
    assert config.config_convection_scheme == SCHEME_GRELL_FREITAS
    assert config.config_cudt_seconds == 120.0


def test_convection_off_is_admitted_as_a_configuration_of_its_own():
    """No anchor exists for convection-off at 120 s; since 2026-10-10 that
    is recorded, not refused, and the configuration builds."""

    config = V841MpasColumnPhysicsConfig(
        config_dt=120.0,
        config_bldt_seconds=120.0,
        config_convection_scheme=SCHEME_OFF,
        config_cudt_seconds=None,
    )
    config.validate()
    record = dt_admission.anchor_record(
        120.0, radiation_seconds=600.0, surface_pbl_seconds=120.0,
        cumulus_seconds=None, cumulus_scheme=None,
    )
    assert record["matched"] is False


def test_an_unknown_convection_scheme_is_still_refused():
    config = V841MpasColumnPhysicsConfig(config_convection_scheme="cu_never_resolved")
    with pytest.raises(ConfigurationRefusal) as error:
        config.validate()
    assert "config_convection_scheme" in str(error.value)
    assert "never called" in str(error.value)


def test_new_tiedtke_is_anchored_at_120_s_and_nowhere_else():
    """Its own 2026-10-05 anchor admits the RK3 120 s lane; every other
    timestep is still refused, because a GF anchor measured a different
    forcing and New Tiedtke has earned none there."""

    config = V841MpasColumnPhysicsConfig(config_convection_scheme=SCHEME_NTIEDTKE)
    config.validate()
    anchor = dt_admission.admitted_timestep(120.0, "ntiedtke", 120.0)
    assert anchor is not None and anchor.admitted_on == "2026-10-05"
    for dt in (100.0, 75.0, 20.0, 5.0):
        assert dt_admission.admitted_timestep(dt, "ntiedtke", dt) is None, dt


def test_the_cumulus_cadence_and_the_scheme_must_agree():
    """``off`` with a cumulus cadence is a configuration that means nothing."""

    with pytest.raises(ConfigurationRefusal) as error:
        V841MpasColumnPhysicsConfig(
            config_convection_scheme=SCHEME_OFF, config_cudt_seconds=120.0
        ).validate()
    assert "config_cudt_seconds" in str(error.value)

    with pytest.raises(ConfigurationRefusal) as error:
        V841MpasColumnPhysicsConfig(
            config_convection_scheme=SCHEME_GRELL_FREITAS, config_cudt_seconds=None
        ).validate()
    assert "config_cudt_seconds" in str(error.value)


# ---------------------------------------------------------------------------
# an anchor certifies a CONFIGURATION at a timestep, not a timestep alone
# ---------------------------------------------------------------------------
def test_the_registry_is_keyed_by_configuration_not_by_timestep_alone():
    gf_key = dt_admission.dt_key(20.0, "gf")
    off_key = dt_admission.dt_key(20.0, None)
    assert gf_key != off_key
    assert "gf" in gf_key
    assert "off" in off_key


def test_a_convection_off_run_cannot_borrow_the_grell_freitas_anchor():
    """The record never reads the GF row for a convection-off run: the
    nearest row it reports is the convection-off configuration's own, and a
    match is never claimed from the GF anchor."""

    assert dt_admission.admitted_timestep(120.0, "gf") is not None
    record = dt_admission.anchor_record(
        120.0,
        radiation_seconds=600.0,
        surface_pbl_seconds=120.0,
        cumulus_seconds=None,
        cumulus_scheme=None,
    )
    assert record["matched"] is False
    assert record["nearest_row"] is None
    assert record["configuration"]["cumulus_scheme"] is None


def test_every_registered_anchor_keys_on_its_own_configuration():
    for key, anchor in dt_admission.ADMITTED_TIMESTEPS.items():
        assert key == dt_admission.dt_key(anchor.dt_seconds, anchor.cumulus_scheme)


def test_a_candidate_mint_separates_the_two_configurations():
    """20 s holds a GF anchor and a convection-off anchor: each is its own
    row, and a candidate mint at an anchored configuration keeps that row."""

    gf = dt_admission.admitted_timestep(20.0, "gf")
    off = dt_admission.admitted_timestep(20.0, None)
    assert gf is not None and off is not None and gf is not off
    with dt_admission.candidate_mint(
        20.0,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="test, no card",
        cumulus_scheme="gf",
    ) as row:
        assert row is gf


def test_the_unanchored_refusal_names_the_configuration():
    """The record names the configuration where the refusal used to."""

    record = dt_admission.anchor_record(
        30.0, radiation_seconds=1800.0, surface_pbl_seconds=30.0,
        cumulus_seconds=None, cumulus_scheme=None,
    )
    assert record["matched"] is False
    assert record["configuration"]["cumulus_scheme"] is None
    assert record["configuration"]["dt_seconds"] == 30.0
    assert "convection off" in record["registered"]


def test_the_admitted_summary_distinguishes_the_configurations():
    roster = dt_admission.admitted_summary()
    assert "120 s" in roster
    for anchor in dt_admission.ADMITTED_TIMESTEPS.values():
        label = "convection off" if anchor.cumulus_scheme is None else "GF"
        assert label in roster


# ---------------------------------------------------------------------------
# the schedule receipt already knows how to describe convection-off
# ---------------------------------------------------------------------------
def test_a_convection_off_schedule_receipt_reports_no_cumulus_calls():
    receipt = dt_admission.schedule_receipt(5.0, cumulus_scheme=None)
    cadences = receipt["cadences"]
    assert cadences["cumulus_scheme"] is None
    assert cadences["cumulus_seconds"] is None
    assert cadences["stepcu"] == 0
    assert cadences["cumulus_calls_per_hour"] is None


def test_an_anchor_row_is_immutable_and_serialises_with_its_scheme():
    anchor = dt_admission.admitted_timestep(120.0, "gf")
    assert anchor is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        anchor.cumulus_scheme = None  # type: ignore[misc]
    assert anchor.as_dict()["cumulus_scheme"] == "gf"


# ---------------------------------------------------------------------------
# guards that were keyed on the timestep alone, and are not any more
# ---------------------------------------------------------------------------
def test_the_row_override_admits_a_convection_off_mint_at_an_anchored_timestep():
    """RETIRED GUARD, MEASURED (2026-08-26, the proving RTX 5070 Ti).

    ``mesh_row_candidate`` refuses to override a registry row's timestep when
    that timestep already holds a real anchor -- correctly, because a proven
    timestep needs a registered row.  It asked the question of the TIMESTEP
    though, with the Grell-Freitas default, so once the registry was keyed by
    configuration it refused every convection-off mint at 20 s and 5 s with
    "20 s holds a real anchor" when the configuration being earned held none.
    It killed the first convection-off arm in 0.1 s.
    """

    from woof.hex import mesh_row_candidate

    # DERIVED, never hardcoded.  This control named 20.0 while 20 s held a
    # Grell-Freitas anchor and no convection-off one -- and then the
    # convection-off anchors were earned and the control broke on its own
    # success, which is exactly the trap the dt-anchors mutation control fell
    # into on the same day.  So it searches: a timestep that IS anchored with
    # the closure on and is NOT anchored with it off.
    probe = next(
        (
            anchor.dt_seconds
            for anchor in sorted(
                dt_admission.ADMITTED_TIMESTEPS.values(),
                key=lambda a: a.dt_seconds,
            )
            if anchor.cumulus_scheme == "gf"
            and dt_admission.admitted_timestep(anchor.dt_seconds, None) is None
        ),
        None,
    )
    assert probe is not None, (
        "every Grell-Freitas timestep now also holds a convection-off anchor; "
        "that is a finding, not a test failure -- re-read this control"
    )
    assert dt_admission.admitted_timestep(probe, "gf") is not None

    with dt_admission.candidate_mint(
        probe,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="test, no card",
        cumulus_scheme=None,
        cumulus_seconds=None,
    ):
        with mesh_row_candidate.candidate_mesh_dt(
            "x1.40962",
            probe,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
            cumulus_scheme=None,
        ) as override:
            assert override.dt_seconds == probe
            assert "x1.40962" in mesh_row_candidate.active_overrides()
    assert "x1.40962" not in mesh_row_candidate.active_overrides()


def test_the_row_override_still_refuses_an_anchored_configuration():
    """RETIRED 2026-10-10: an anchored configuration may be overridden."""

    from woof.hex import mesh_row_candidate

    with mesh_row_candidate.candidate_mesh_dt(
        "x1.40962",
        20.0,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        cumulus_scheme="gf",
    ):
        assert dict(mesh_row_candidate.active_overrides()) == {"x1.40962": 20.0}


def test_the_physics_cadence_table_reports_no_convection_rather_than_a_zero():
    """RETIRED GUARD, MEASURED (2026-08-26, the proving RTX 5070 Ti).

    ``_v841_physics_cadences`` called ``float(config.config_cudt_seconds)``
    unconditionally, so the first convection-off arm bound clean, admitted
    its timestep, built its sealed constructor and died inside composite
    step 0 with ``float() argument must be a string or a real number, not
    'NoneType'`` -- surfacing as "composite step at 0.0 s was aborted
    without publication", which names neither the field nor the reason.

    ``None`` and not ``0.0``: every other entry in that table is a call
    interval, and a zero there reads as "called every instant".
    """

    from woof.hex.cuda_driver import _v841_physics_cadences

    off = V841MpasColumnPhysicsConfig(
        config_convection_scheme=SCHEME_OFF, config_cudt_seconds=None
    )
    cadences = _v841_physics_cadences(off)
    assert cadences["convection"] is None
    assert cadences["microphysics"] == 120.0
    # MPAS-A's radiation default, config_radtlw/sw_interval = 00:30:00.
    assert cadences["radiation_lw"] == 1800.0


def test_the_run_refuses_two_sources_for_one_convection_decision():
    """The bind's request and the driver's must be the same string.

    The 2026-08-26 clock defect was two independent sources for one
    timestep, discovered 285 s and 18,820 MiB into a run.  The convection
    selection travels the same road and gets the same check.
    """

    import importlib.util
    import sys
    from pathlib import Path

    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    source = (PACKAGE_DIR / "drivers" / "run_cuda_v841_forecast.py").read_text(encoding="utf-8")
    assert 'decision.get("requested") != convection' in source
    assert "One decision, one " in source
    # And the door hands the SAME string to the bind and to the driver.
    door = (PACKAGE_DIR / "forecast_door.py").read_text(encoding="utf-8")
    assert '"--convection", request.convection,' in door
    assert "convection=request.convection," in door


# ---------------------------------------------------------------------------
# the door's row-alone answer, which must be the run's answer
# ---------------------------------------------------------------------------
def test_the_door_answers_the_convection_question_from_the_row_alone():
    """No card, no file: preflight and the run must not disagree."""

    from woof.hex import forecast_door

    admitted = forecast_door.admit_timestep(
        "x1.40962", 120.0, nominal_dx_m=120_000.0, convection="auto",
        radiation_seconds=600.0,
    )
    assert admitted["convection"]["constructor_scheme"] == "ntiedtke"
    assert admitted["convection"]["source"] == "resolution"
    assert admitted["convection"]["scheme"] == SCHEME_NTIEDTKE
    # The anchor's verdict rides along as evidence, so a caller sees what
    # the row's physics did before spending a card on it.
    assert admitted["anchor"]["row"]["physics_health"].startswith("TRACKS THE CONTROL")
    # The Grell-Freitas reference stays reachable as an explicit arm.
    gf = forecast_door.admit_timestep(
        "x1.40962", 120.0, nominal_dx_m=120_000.0, convection="gf",
        radiation_seconds=600.0,
    )
    assert gf["convection"]["constructor_scheme"] == "gf"
    assert gf["anchor"]["row"]["physics_health"].startswith("REFERENCE")


def test_every_registered_row_gets_the_scheme_its_own_spacing_selects():
    """The ruling decides each row from its own spacing, and nothing else.

    THIS TEST WAS RE-READ ON 2026-08-29, NOT RELAXED, and its own former
    message is why.  It used to assert that the FINEST registered mesh is at
    or above the 3,000 m threshold -- true when it was written (finest
    15,000 m) and still true at 4,000 m -- and it printed the instruction for
    the day that stopped holding: "a row finer than the threshold is now
    registered; the ruling fires on it, and this test should be re-read
    rather than relaxed".

    That day is here: ``v0.9.120.110533`` declares 937.5 m.  So the claim the
    test protects is restated as what it always meant -- every row gets the
    scheme ITS OWN spacing selects -- and it now checks BOTH sides, which is
    strictly more than the single-sided version could:

    * every row at or above the threshold still selects Grell-Freitas, so
      nothing that was registered before this ruling moved; and
    * every row below it selects no cumulus scheme at all, so the ruling
      actually fires rather than being a threshold nobody crosses.

    Keyed on the threshold rather than on a list of names, so the next
    sub-kilometre row is absorbed without editing this test again.
    """

    import importlib.util
    import sys
    from pathlib import Path

    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "_test_mesh_binding_convection", PACKAGE_DIR / "drivers" / "mpas_mesh_binding.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_test_mesh_binding_convection"] = module
    spec.loader.exec_module(module)

    coarse, fine, tiedtke = [], [], []
    for name, row in module.MESH_BINDINGS.items():
        dx = float(row.nominal_dx_m)
        decision = convection_admission.convection_decision(nominal_dx_m=dx)
        if dx >= NTIEDTKE_AT_OR_ABOVE_M:
            assert decision["scheme"] == SCHEME_NTIEDTKE, name
            tiedtke.append(name)
        elif dx >= CONVECTION_OFF_BELOW_M:
            assert decision["scheme"] == SCHEME_GRELL_FREITAS, name
            coarse.append(name)
        else:
            assert decision["scheme"] == SCHEME_OFF, name
            fine.append(name)
        # Either way the decision is the RULING's and not a flag's.
        assert decision["source"] == "resolution", name
        # THE BREAKAGE THE 20 km THRESHOLD PREVENTS: a row moved to New
        # Tiedtke whose own timestep holds no New Tiedtke anchor would
        # refuse a bare run that ran before.  Every row the 2026-10-05
        # ruling moves must be admitted at its dt.
        if decision["scheme"] == SCHEME_NTIEDTKE:
            assert dt_admission.admitted_timestep(
                float(row.dt_seconds), "ntiedtke", float(row.dt_seconds)
            ) is not None, (name, row.dt_seconds)
    assert tiedtke, "no registered row is at or above 20 km"
    # Both arms have to be exercised, or this test is single-sided again
    # without saying so.
    assert coarse, "no registered row is at or above the threshold"
    assert fine, (
        "no registered row is below the threshold, so the convection-off "
        "half of the ruling is untested here; if that is genuinely the "
        "state of the registry, say so rather than deleting the assertion"
    )
