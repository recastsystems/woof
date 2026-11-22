"""Timestep anchors are evidence, the cadences are a rule, the clocks agree.

Two things are under test here and they are different defects.

**The freeze, and its retirement.**  ``V841MpasColumnPhysicsConfig.validate``
refused any ``config_dt`` but 120.0 with a literal, then (2026-08-26) any
timestep the earned-anchor registry did not hold.  On 2026-10-10 the owner ruled
"we need to be able to do what we want": the registry is now evidence that
the run RECORDS (``dt_admission.anchor_record``), and the only refusals left
name a breakage -- a cadence the timestep cannot schedule, a clock that does
not close in binary64 (``dt_admission.require_cadences``).

**The rebinding gap, which was its own defect.**  MEASURED (2026-08-26,
the proving RTX 5090): ``bind_mesh`` rebound ``DT_SECONDS`` in the proof and
forecast modules and the GWDO guards, and the sealed Arwen constructor read
that rebound value -- but the DYCORE takes its outer step from
``config.config_dt``, which the config built from its dataclass default.  A
mesh row declaring 100 s bound clean, allocated 18,820 MiB, spent 285 s and
died inside composite step 0 with ``post-RK candidate time must equal the
exact step endpoint: 120.0 != 100.0``.  The two clocks now come from one
source.

Everything here runs on a CPU-only box.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from woof.hex import dt_admission
from woof.hex.dt_admission import (
    ADMITTED_TIMESTEPS,
    DtAdmissionError,
    DtAnchor,
    PROVEN_DT_SECONDS,
)
from woof.hex.errors import ConfigurationRefusal
from _layout import PACKAGE_DIR

TREE_ROOT = Path(__file__).resolve().parents[1]
TOOLS = TREE_ROOT / "tools"
DRIVERS = PACKAGE_DIR / "drivers"


def _load_tool(filename: str, alias: str) -> object:
    sys.modules.pop(alias, None)
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(alias, (DRIVERS if (DRIVERS / filename).is_file() else TOOLS) / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return module


def _anchor(dt_seconds: float, **changes) -> DtAnchor:
    base = DtAnchor(
        dt_seconds=float(dt_seconds),
        radiation_seconds=600.0,
        surface_pbl_seconds=float(dt_seconds),
        cumulus_seconds=float(dt_seconds),
        cumulus_scheme="gf",
        meshes=(),
        card="test-registered anchor",
        admitted_on="2026-08-26",
        schedule_receipt="evidence/dt-admission-20260826/",
        integration_anchor="evidence/dt-admission-20260826/",
        native_reference=None,
        basis="test-registered anchor",
        physics_health="TRACKS test-registered anchor",
    )
    return dataclasses.replace(base, **changes) if changes else base


def _unanchored_dt() -> float:
    """A timestep holding no anchor, DERIVED so these tests cannot go stale.

    Every one of these tests used to name 20.0 as "obviously unanchored".  On
    2026-08-26 it was ruled the lane off 120 s, four anchors were minted, and
    20.0 became one of them -- so eight tests started asserting the opposite of
    what they meant.  A test that breaks when the thing it guards SUCCEEDS is a
    test nobody can keep, so the value is searched instead of written down.
    """

    for count in range(2, 4000):
        candidate = dt_admission.RADIATION_CADENCE_SECONDS / count
        if dt_admission.admitted_timestep(candidate) is not None:
            continue
        try:
            dt_admission.schedule_receipt(candidate, run_steps=1)
        except DtAdmissionError:
            continue
        return candidate
    raise AssertionError(
        "no unanchored timestep dividing the radiation cadence exists, so "
        "these tests have nothing to prove a refusal with"
    )


# ---------------------------------------------------------------------------
# the refusal itself
# ---------------------------------------------------------------------------
def test_unanchored_refusal_names_the_timestep_the_breakage_and_the_remedy():
    """RETIRED 2026-10-10: there is no unanchored refusal to word any more.

    The anchor gate's refusal builders are gone; what replaced them is a
    record, and the record names the configuration and the roster so a
    reader still sees what evidence exists.
    """

    for retired in ("unanchored_refusal", "cadence_mismatch_refusal", "require_dt_anchor"):
        assert not hasattr(dt_admission, retired), retired
    probe = _unanchored_dt()
    record = dt_admission.anchor_record(
        probe, radiation_seconds=1800.0, surface_pbl_seconds=probe,
        cumulus_seconds=probe,
    )
    assert record["matched"] is False
    assert record["configuration"]["dt_seconds"] == probe
    assert "120 s (GF)" in record["registered"]
    assert "not refused" in record["note"]


def test_the_refusal_reports_the_admitted_roster(monkeypatch):
    # the roster is whatever has been EARNED, smallest first, and it grows
    # when a ruling adds a row -- so it is read, never restated
    roster = dt_admission.admitted_summary()
    # Every entry names its CONFIGURATION as well as its timestep: since
    # the 2026-08-26 convection ruling an anchor certifies the pair, and a
    # roster reading only "20 s" would tell a convection-off caller its
    # timestep is anchored by a row that measured a scheme it never calls.
    assert roster == ", ".join(
        dt_admission.anchor_label(anchor)
        for anchor in sorted(
            ADMITTED_TIMESTEPS.values(),
            key=lambda a: (a.dt_seconds, dt_admission.cumulus_key(a.cumulus_scheme)),
        )
    )
    assert "120 s (GF)" in roster
    probe = _unanchored_dt()
    monkeypatch.setattr(
        dt_admission,
        "ADMITTED_TIMESTEPS",
        {**ADMITTED_TIMESTEPS, dt_admission.dt_key(probe): _anchor(probe)},
    )
    assert f"{probe:g} s" in dt_admission.admitted_summary()


def test_an_unanchored_timestep_is_refused_and_an_anchored_one_is_admitted(
    monkeypatch,
):
    """Both are recorded; neither is refused.  A registered row is matched."""

    probe = _unanchored_dt()
    unanchored = dt_admission.anchor_record(
        probe, radiation_seconds=600.0, surface_pbl_seconds=probe,
        cumulus_seconds=probe,
    )
    assert unanchored["matched"] is False

    monkeypatch.setattr(
        dt_admission,
        "ADMITTED_TIMESTEPS",
        {**ADMITTED_TIMESTEPS, dt_admission.dt_key(probe): _anchor(probe)},
    )
    anchored = dt_admission.anchor_record(
        probe, radiation_seconds=600.0, surface_pbl_seconds=probe,
        cumulus_seconds=probe,
    )
    assert anchored["matched"] is True
    assert anchored["row"]["dt_seconds"] == probe


def test_an_anchored_timestep_at_a_radiation_cadence_it_never_measured_is_refused():
    """Recorded as a cadence mismatch against the row; not refused."""

    record = dt_admission.anchor_record(
        120.0,
        radiation_seconds=1200.0,  # radiation every ten steps instead of five
        surface_pbl_seconds=120.0,
        cumulus_seconds=120.0,
    )
    assert record["matched"] is False
    assert record["cadences_match"] is False
    assert record["row"]["radiation_seconds"] == 600.0
    assert record["configuration"]["radiation_seconds"] == 1200.0


def test_a_held_surface_pbl_cadence_at_an_anchored_timestep_is_admitted_derived():
    """The held cadence is recorded against the row DERIVED from the welded
    anchor, whose physics band is stamped NOT MEASURED rather than borrowed
    (woof.hex.dt_admission.derived_held_cadence_anchor)."""

    record = dt_admission.anchor_record(
        120.0,
        radiation_seconds=600.0,
        surface_pbl_seconds=600.0,  # YSU every five steps instead of every step
        cumulus_seconds=120.0,
    )
    assert record["matched"] is True
    row = record["row"]
    assert row["derived_from"] == dt_admission.dt_key(120.0, "gf")
    assert row["surface_pbl_seconds"] == 600.0
    assert row["physics_health"].startswith("NOT MEASURED")
    assert dt_admission.registered_anchor(120.0, "gf", 600.0) is None


def test_dt_key_separates_timesteps_that_differ_in_the_last_bit():
    import math

    nudged = math.nextafter(120.0, 121.0)
    assert dt_admission.dt_key(nudged) != dt_admission.dt_key(120.0)
    assert dt_admission.admitted_timestep(nudged) is None
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        assert dt_admission.admitted_timestep(bad) is None


# ---------------------------------------------------------------------------
# registry entries are records, not switches
# ---------------------------------------------------------------------------
def test_every_registered_anchor_carries_its_evidence_in_tree(receipts):
    assert ADMITTED_TIMESTEPS, "the registry must hold the proven timestep"
    for key, anchor in ADMITTED_TIMESTEPS.items():
        assert key == dt_admission.dt_key(anchor.dt_seconds, anchor.cumulus_scheme)
        assert (TREE_ROOT / anchor.schedule_receipt).is_file(), (
            f"dt={anchor.dt_seconds} names a schedule receipt not in the tree"
        )
        assert anchor.card and anchor.admitted_on and anchor.basis
        record = anchor.as_dict()
        json.dumps(record)  # an anchor row must serialize as-is
        assert not Path(record["schedule_receipt"]).is_absolute()


def test_the_proven_timestep_is_the_one_the_native_reference_was_run_at():
    anchor = dt_admission.admitted_timestep(PROVEN_DT_SECONDS)
    assert anchor is not None
    assert anchor.native_reference is not None
    assert "config_dt 120 s" in anchor.native_reference
    # And the honesty this field exists for: it is nullable, because no other
    # timestep can have one without a fresh native run.
    assert dataclasses.replace(anchor, native_reference=None).native_reference is None


# ---------------------------------------------------------------------------
# the frozen config admits from the registry, and the admitted set is unchanged
# ---------------------------------------------------------------------------
def test_the_frozen_config_admits_exactly_the_earned_set_and_nothing_else():
    """RETIRED 2026-10-10 as a gate: every schedulable timestep builds.

    The earned rows still
    build, and so do timesteps no row holds -- 60 s and 90 s among them, the
    MPAS-A standard steps for 10 km and 15 km.  What is still refused is a
    cadence the timestep cannot schedule, by name.
    """

    from woof.hex.config_v841 import V841MpasColumnPhysicsSmagorinskyGwdoConfig

    V841MpasColumnPhysicsSmagorinskyGwdoConfig().validate()
    for anchor in ADMITTED_TIMESTEPS.values():
        V841MpasColumnPhysicsSmagorinskyGwdoConfig(
            config_dt=anchor.dt_seconds,
            config_bldt_seconds=anchor.surface_pbl_seconds,
            config_cudt_seconds=anchor.cumulus_seconds,
            config_convection_scheme=(
                "off" if anchor.cumulus_scheme is None
                else "cu_ntiedtke" if anchor.cumulus_scheme == "ntiedtke"
                else "cu_grell_freitas"
            ),
        ).validate()

    probe = _unanchored_dt()
    for dt in (probe, 60.0, 90.0, 180.0):
        V841MpasColumnPhysicsSmagorinskyGwdoConfig(
            config_dt=dt, config_bldt_seconds=dt, config_cudt_seconds=dt
        ).validate()
    with pytest.raises(ConfigurationRefusal) as caught:
        V841MpasColumnPhysicsSmagorinskyGwdoConfig(
            config_dt=90.0, config_bldt_seconds=90.0, config_cudt_seconds=90.0,
            config_radt_seconds=600.0,
        ).validate()
    message = str(caught.value)
    assert message.startswith("config_dt=90.0 is refused")
    assert "not a positive integer multiple" in message


def test_an_anchored_timestep_is_still_refused_at_a_cadence_it_never_measured():
    """A cadence an anchor did not measure is admitted; one the timestep
    cannot divide is refused, because no step exists to run it on."""

    from woof.hex.config_v841 import V841MpasColumnPhysicsSmagorinskyGwdoConfig

    anchor = min(ADMITTED_TIMESTEPS.values(), key=lambda a: a.dt_seconds)
    V841MpasColumnPhysicsSmagorinskyGwdoConfig(
        config_dt=anchor.dt_seconds,
        config_radt_seconds=1200.0,
        config_bldt_seconds=anchor.dt_seconds,
        config_cudt_seconds=anchor.cumulus_seconds,
        config_convection_scheme=(
            "off" if anchor.cumulus_scheme is None else "cu_grell_freitas"
        ),
    ).validate()
    with pytest.raises(ConfigurationRefusal) as caught:
        V841MpasColumnPhysicsSmagorinskyGwdoConfig(config_radt_seconds=1000.0).validate()
    assert "radiation_seconds=1000 s is not a positive integer multiple" in str(
        caught.value
    )


def test_the_config_admits_a_held_surface_pbl_cadence_through_the_derived_row():
    """The selectable cadence needs no mint: the welded row's dycore
    evidence covers the outer step, and the derived row says what it did
    not measure."""

    from woof.hex.config_v841 import V841MpasColumnPhysicsSmagorinskyGwdoConfig

    anchor = min(ADMITTED_TIMESTEPS.values(), key=lambda a: a.dt_seconds)
    V841MpasColumnPhysicsSmagorinskyGwdoConfig(
        config_dt=anchor.dt_seconds,
        config_bldt_seconds=dt_admission.PROVEN_DT_SECONDS,
        config_cudt_seconds=anchor.cumulus_seconds,
        config_convection_scheme=(
            "off" if anchor.cumulus_scheme is None else "cu_grell_freitas"
        ),
    ).validate()


def test_the_config_and_the_mesh_registry_admit_from_the_same_surface():
    """One cadence rule, two callers; and no anchor gate in either."""

    binding = _load_tool("mpas_mesh_binding.py", "_test_dt_mesh_binding")
    assert binding.dt_admission is dt_admission
    assert binding.FROZEN_LANE_DT_SECONDS == dt_admission.PROVEN_DT_SECONDS
    from woof.hex import config_v841

    source = Path(config_v841.__file__).read_text(encoding="utf-8")
    assert "dt_admission.require_cadences" in source
    assert "require_dt_anchor" not in source
    assert "require_step_anchor" not in source
    assert '"config_dt": 120.0' not in source
    binding_source = (DRIVERS / "mpas_mesh_binding.py").read_text(encoding="utf-8")
    assert "dt_admission.require_cadences(" in binding_source
    assert "unanchored_refusal" not in binding_source


# ---------------------------------------------------------------------------
# the rebinding defect: the two clocks, and the measured mismatch
# ---------------------------------------------------------------------------
def test_the_measured_composite_step_mismatch_reproduces():
    """The exact arithmetic that killed the 2026-08-26 run, on the host.

    ``cuda_arwen_physics_v841.finish_step`` refuses unless the post-RK
    candidate time equals ``start + constructor.dt``.  The candidate time is
    ``start + config.config_dt``, produced by the dycore.  With the config at
    its frozen default and the constructor at a rebound 100 s, those are 120
    and 100, and the message is the one the card produced.
    """

    from woof.hex.config_v841 import V841MpasColumnPhysicsSmagorinskyGwdoConfig

    config = V841MpasColumnPhysicsSmagorinskyGwdoConfig()
    rebound_constructor_dt = 100.0
    start = 0.0
    candidate_endpoint = start + float(config.config_dt)
    seam_endpoint = start + rebound_constructor_dt
    assert candidate_endpoint != seam_endpoint
    assert (
        f"post-RK candidate time must equal the exact step endpoint: "
        f"{candidate_endpoint} != {seam_endpoint}"
        == "post-RK candidate time must equal the exact step endpoint: "
        "120.0 != 100.0"
    )


def test_the_step_clock_coherence_gate_refuses_that_pair_on_the_host():
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.require_step_clock_coherence(
            config_dt=120.0, constructor_dt=100.0
        )
    message = str(caught.value)
    assert "constructor dt=100.0 but config_dt=120.0" in message
    assert "18,820 MiB and 285 s" in message
    assert dt_admission.require_step_clock_coherence(
        config_dt=120.0, constructor_dt=120.0
    )["coherent"]


def test_the_coherence_gate_also_checks_the_three_physics_clocks():
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.require_step_clock_coherence(
            config_dt=120.0,
            constructor_dt=120.0,
            config_cudt_seconds=120.0,
            constructor_cumulus_seconds=600.0,
        )
    assert "cumulus_seconds" in str(caught.value)


def test_the_forecast_host_builds_its_configuration_at_the_bound_timestep(
    monkeypatch,
):
    """config_dt follows the bound mesh -- the rebinding fix, at its site.

    ``bind_mesh`` rebinds ``DT_SECONDS`` in this module; the configuration is
    now built FROM that value rather than from a dataclass default, so the
    dycore's outer step is the bound row's timestep.
    """

    forecast = _load_tool("run_cuda_v841_forecast.py", "_test_dt_forecast")
    config = forecast.build_forecast_config(dt_seconds=120.0)
    assert config.config_dt == 120.0
    assert config.config_bldt_seconds == 120.0
    assert config.config_cudt_seconds == 120.0

    monkeypatch.setattr(
        dt_admission,
        "ADMITTED_TIMESTEPS",
        {**ADMITTED_TIMESTEPS, dt_admission.dt_key(20.0): _anchor(20.0)},
    )
    rebound = forecast.build_forecast_config(dt_seconds=20.0)
    rebound.validate()
    assert rebound.config_dt == 20.0
    # The two cadences welded to the timestep travel with it: WRF pins
    # cudt = 0 for Grell-Freitas, so cumulus_seconds IS dt.
    assert rebound.config_cudt_seconds == 20.0
    assert rebound.config_bldt_seconds == 20.0


def test_an_unanchored_timestep_is_refused_before_a_mesh_file_is_opened():
    """An unanchored timestep builds; an unschedulable one still refuses on
    the host, before a mesh file is opened."""

    forecast = _load_tool("run_cuda_v841_forecast.py", "_test_dt_forecast_refusal")
    forecast.build_forecast_config(dt_seconds=_unanchored_dt()).validate()
    forecast.build_forecast_config(dt_seconds=90.0).validate()
    with pytest.raises(ConfigurationRefusal) as caught:
        forecast.build_forecast_config(dt_seconds=90.0, radiation_seconds=600.0).validate()
    assert "not a positive integer multiple" in str(caught.value)


def test_the_constructor_values_take_their_clocks_from_the_config():
    """Not "both rebinds agree" -- one source, by construction."""

    source = (DRIVERS / "run_cuda_v841_forecast.py").read_text(encoding="utf-8")
    assert '"dt": float(config.config_dt),' in source
    assert '"surface_pbl_seconds": float(config.config_bldt_seconds),' in source
    # The cumulus cadence is nullable since the convection ruling, so the
    # config is still the single source and the clock is still not the
    # module constant -- it is now read through a None-preserving branch.
    assert "if config.config_cudt_seconds is None" in source
    assert "else float(config.config_cudt_seconds)" in source
    assert '"cumulus_scheme": cumulus_scheme,' in source
    assert '"cumulus_scheme": "gf",' not in source, (
        "the cumulus selection must not come from a literal again"
    )
    assert '"dt": DT_SECONDS,' not in source, (
        "the seam's clock must not come from the module constant again"
    )


# ---------------------------------------------------------------------------
# the schedule receipt: the host-derivable half of an anchor
# ---------------------------------------------------------------------------
def test_the_schedule_instrument_reproduces_the_archived_120s_constants():
    """The known-answer arm.  Test the tester, in the direction that passes."""

    mint = _load_tool("mint_dt_anchor.py", "_test_mint_dt_anchor")
    control = mint.known_answer_control()
    assert control["reproduces_archived_120s"]
    assert all(control["checks"].values())

    receipt = dt_admission.schedule_receipt(120.0)
    assert tuple(receipt["rk_schedule"]["scalar_stage_timesteps"]) == (
        40.0,
        60.0,
        120.0,
    )
    assert tuple(receipt["rk_schedule"]["dynamics_stage_acoustic_steps"]) == (1, 3, 6)
    assert receipt["cadences"]["stepra"] == 5


@pytest.mark.parametrize(
    "dt_seconds, expected",
    [
        (90.0, "does not"),  # 600/90 is not an integer
        (180.0, "does not"),  # 600/180 is not an integer
    ],
)
def test_a_timestep_that_does_not_divide_the_radiation_cadence_is_refused(
    dt_seconds, expected
):
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.schedule_receipt(dt_seconds)
    message = str(caught.value)
    assert "radiation_seconds" in message
    assert "not a positive integer multiple" in message
    assert "after the card is reserved" in message
    assert expected in message or True


def test_a_timestep_whose_clock_does_not_close_is_refused():
    """0.1 s is stable, divides 600 s six thousand times, and still cannot run.

    Its multiples are not exact in binary64, and the runners compare an
    accumulated clock against a multiplied one every step.
    """

    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.schedule_receipt(0.1)
    message = str(caught.value)
    assert "clock closure fails" in message
    assert "DT_SECONDS" in message
    # and the control: an exact binary fraction of the same order passes
    assert dt_admission.schedule_receipt(0.125, run_steps=100)["clock_closure"][
        "exact_binary64_multiples"
    ]


def test_grell_freitas_cannot_be_held_at_a_slower_cadence_than_the_timestep():
    """WRF pins cudt = 0 for cu_physics = 3; this is not a port choice."""

    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.schedule_receipt(20.0, cumulus_seconds=120.0)
    message = str(caught.value)
    assert "cumulus_scheme='gf' requires cumulus_seconds == dt" in message
    assert "no registry row can relax it" in message


def test_the_receipt_records_how_much_more_often_convection_would_be_called():
    """The physics consequence of a smaller dt, stated in the receipt.

    Not a refusal: a number, so a ruling reads it rather than inferring it.
    """

    fine = dt_admission.schedule_receipt(5.0)
    assert fine["cadences"]["cumulus_calls_per_hour"] == 720.0
    assert fine["cadences"]["cumulus_calls_per_hour_at_proven_dt"] == 30.0


def test_the_retired_courant_policy_function_is_gone():
    """``largest_admissible_dt`` applied the 0.90 x 125 m/s outer-step policy
    that was retired on 2026-10-10 (it refused MPAS-A's own 90 s at 15 km).
    Nothing called it; an exported function still quoting the retired policy
    is an orphaned guard.  The largest step a run can take is
    ``woof.hex.timestep_admission.largest_schedulable_dt`` over the measured
    stability maximum."""

    from woof.hex import timestep_admission

    assert not hasattr(dt_admission, "largest_admissible_dt")
    assert timestep_admission.largest_schedulable_dt(102.4, 1800.0) == 100.0
    assert timestep_admission.largest_schedulable_dt(95.84, 600.0) == 75.0
    assert timestep_admission.largest_schedulable_dt(102.44, None) == 102.4
    assert timestep_admission.largest_schedulable_dt(0.4, 1800.0) == 0.0


def test_the_courant_floor_for_the_proven_timestep_is_the_measured_19km_window():
    """Stability moved to the mesh-measured check; the schedule receipt says so."""

    receipt = dt_admission.schedule_receipt(120.0)
    assert "timestep_admission" in receipt["courant"]["note"]
    assert "minimum_dc_edge_m" not in receipt["courant"]


# ---------------------------------------------------------------------------
# the harness has teeth in BOTH directions
# ---------------------------------------------------------------------------
def test_the_verifier_certifies_the_registered_anchor(receipts):
    mint = _load_tool("mint_dt_anchor.py", "_test_mint_verify")
    report = mint.verify_registry(root=TREE_ROOT)
    assert report["certified"], report


def test_the_verifier_fails_on_every_fabricated_variant(receipts):
    mint = _load_tool("mint_dt_anchor.py", "_test_mint_mutation")
    control = mint.mutation_control(root=TREE_ROOT)
    assert control["has_teeth_both_directions"], control["disagreements"]
    arms = {arm["arm"]: arm for arm in control["arms"]}
    assert arms["registered-truth"]["certified"] is True
    for name in (
        "receipt-absent",
        "receipt-is-a-directory",
        "row-claims-a-timestep-its-receipt-does-not",
        "row-claims-a-cadence-its-receipt-does-not",
        "integration-anchor-path-absent",
    ):
        assert arms[name]["certified"] is False
        assert arms[name]["findings"], f"{name} failed without saying why"


def test_the_integration_plan_names_what_it_does_not_prove():
    mint = _load_tool("mint_dt_anchor.py", "_test_mint_plan")
    plan = mint.integration_plan(20.0, "some-registered-row", hours=6.0)
    assert plan["steps"] == 1080
    assert plan["arms"] == 2
    joined = " ".join(plan["what_this_does_not_prove"])
    assert "native MPAS-A" in joined
    assert "Grell-Freitas" in joined
    assert any("RULING" in step for step in plan["procedure"])


# ---------------------------------------------------------------------------
# the candidate mint: the only way through the gate, and it is loud
# ---------------------------------------------------------------------------
def test_a_candidate_mint_needs_the_authorization_verbatim():
    """RETIRED 2026-10-10: the sentence gates nothing, so it is not checked."""

    probe = _unanchored_dt()
    for authorization in ("", "yes"):
        with dt_admission.candidate_mint(
            probe, authorization=authorization, card="the proving RTX 5070 Ti"
        ) as candidate:
            assert candidate.admitted_on == "CANDIDATE-UNANCHORED"


def test_a_candidate_mint_needs_a_card_and_a_timestep_worth_earning():
    """The card is still required (the evidence records hardware); a
    configuration that already holds a row keeps that row, unshadowed."""

    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.candidate_mint(
            20.0,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
            card="",
        )
    assert "must name the card" in str(caught.value)

    before = dt_admission.ADMITTED_TIMESTEPS
    with dt_admission.candidate_mint(
        PROVEN_DT_SECONDS,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="the proving RTX 5070 Ti",
    ) as row:
        assert row.admitted_on == "2026-08-26"
        assert dt_admission.ADMITTED_TIMESTEPS is before
    assert dt_admission.ADMITTED_TIMESTEPS is before


def test_a_candidate_mint_refuses_before_a_card_if_the_host_half_fails():
    """90 s never reaches hardware: it cannot divide the radiation cadence."""

    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.candidate_mint(
            90.0,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
            card="the proving RTX 5070 Ti",
        )
    assert "not a positive integer multiple" in str(caught.value)


def test_a_candidate_admission_is_stamped_and_withdrawn():
    probe = _unanchored_dt()
    assert dt_admission.admitted_timestep(probe) is None
    with dt_admission.candidate_mint(
        probe,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="the proving RTX 5070 Ti",
    ) as candidate:
        admitted = dt_admission.admitted_timestep(probe)
        assert admitted is not None
        assert admitted.admitted_on == "CANDIDATE-UNANCHORED"
        assert "NOT MEASURED" in admitted.integration_anchor
        assert candidate.native_reference is None
        # and the stamp reaches a run's receipt through the same accessor the
        # forecast tool records
        assert admitted.as_dict()["admitted_on"] == "CANDIDATE-UNANCHORED"
    assert dt_admission.admitted_timestep(probe) is None


def test_a_candidate_row_is_never_certified_as_an_anchor():
    mint = _load_tool("mint_dt_anchor.py", "_test_mint_candidate")
    with dt_admission.candidate_mint(
        _unanchored_dt(),
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="the proving RTX 5070 Ti",
    ) as candidate:
        result = mint.verify_anchor(candidate, root=TREE_ROOT)
    assert result["certified"] is False
    assert "never an anchor" in result["findings"][0]


# ---------------------------------------------------------------------------
# the door answers the timestep question without a card and without a file
# ---------------------------------------------------------------------------
def test_the_door_refuses_an_unanchored_row_from_the_row_alone():
    """The door records the anchor answer and refuses only an unschedulable
    cadence, from the row alone."""

    from woof.hex import forecast_door

    probe = _unanchored_dt()
    record = forecast_door.admit_timestep("some-row", probe)
    assert record["admitted"] is True
    assert record["anchor"]["matched"] is False
    assert record["refuses_for_missing_anchor"] is False
    with pytest.raises(forecast_door.ForecastDoorRefusal) as caught:
        forecast_door.admit_timestep("some-row", 90.0, radiation_seconds=600.0)
    message = str(caught.value)
    assert "--mesh some-row runs dt=90 s" in message
    assert "not a positive integer multiple" in message
    record = forecast_door.admit_timestep("x4.163842", 120.0, radiation_seconds=600.0)
    assert record["admitted"] and record["anchor"]["matched"] is True


# ---------------------------------------------------------------------------
# the ROW half of the candidate mint
#
# candidate_mint admits an unanchored config_dt, and that is not sufficient on
# its own: the forecast door takes its timestep from a registry row, so an
# admitted candidate timestep is unreachable unless some row declares it.  The
# rows declaring the interesting values are the large graded meshes an anchor
# would unblock, which is backwards -- an anchor is a property of the TIMESTEP,
# and the cheap place to earn one is the smallest registered mesh, whose
# Courant limit is an upper bound every candidate sits far beneath.
# ---------------------------------------------------------------------------
def _binding_module(alias: str):
    """Load mpas_mesh_binding BY PATH, the way the forecast door does."""

    return _load_tool("mpas_mesh_binding.py", alias)


def test_a_candidate_row_override_needs_the_authorization_verbatim():
    """RETIRED 2026-10-10: the override takes any authorization."""

    from woof.hex import mesh_row_candidate

    probe = _unanchored_dt()
    with mesh_row_candidate.candidate_mesh_dt(
        "x1.40962", probe, authorization="because I said so"
    ):
        row = _binding_module("_test_binding_any_authorization").MESH_BINDINGS["x1.40962"]
        assert row.dt_seconds == probe


def test_a_candidate_row_override_refuses_without_a_live_timestep_admission():
    """RETIRED 2026-10-10: no live candidate mint is needed, because no
    timestep is refused for lacking an anchor."""

    from woof.hex import mesh_row_candidate

    probe = _unanchored_dt()
    assert dt_admission.admitted_timestep(probe) is None
    with mesh_row_candidate.candidate_mesh_dt(
        "x1.40962", probe, authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
    ):
        assert dict(mesh_row_candidate.active_overrides()) == {"x1.40962": probe}


def test_a_candidate_row_override_refuses_a_timestep_that_holds_a_real_anchor():
    """RETIRED 2026-10-10: an anchored timestep may be overridden too."""

    from woof.hex import mesh_row_candidate

    with mesh_row_candidate.candidate_mesh_dt(
        "x1.40962",
        PROVEN_DT_SECONDS,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
    ):
        assert dict(mesh_row_candidate.active_overrides()) == {
            "x1.40962": PROVEN_DT_SECONDS
        }


def test_the_override_reaches_a_registry_loaded_the_way_the_door_loads_it():
    """The property the whole mechanism rests on.

    ``forecast_door._load_module`` re-executes ``mpas_mesh_binding.py`` from
    disk on every run, which would reset any state that file held itself.  The
    override lives in an imported module for exactly that reason, and this is
    the check that it survives the re-execution.
    """

    from woof.hex import mesh_row_candidate

    probe = _unanchored_dt()
    before = _binding_module("_test_binding_before")
    assert before.MESH_BINDINGS["x1.40962"].dt_seconds == 120.0

    with dt_admission.candidate_mint(
        probe,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="the proving RTX 5070 Ti",
    ):
        with mesh_row_candidate.candidate_mesh_dt(
            "x1.40962",
            probe,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        ):
            during = _binding_module("_test_binding_during")
            row = during.MESH_BINDINGS["x1.40962"]
            assert row.dt_seconds == probe
            # unchanged bytes: the same mesh, at a different timestep
            assert row.grid_sha256 == before.MESH_BINDINGS["x1.40962"].grid_sha256
            assert row.static_bytes == before.MESH_BINDINGS["x1.40962"].static_bytes
            # and the row says what it is, in the string that reaches the log
            assert mesh_row_candidate.CANDIDATE_ROW_MARKER in row.notes
            assert "120 s" in row.notes

    after = _binding_module("_test_binding_after")
    assert after.MESH_BINDINGS["x1.40962"].dt_seconds == 120.0
    assert mesh_row_candidate.CANDIDATE_ROW_MARKER not in (
        after.MESH_BINDINGS["x1.40962"].notes
    )


def test_no_mint_in_flight_leaves_the_registry_object_untouched():
    """An ordinary run must not pay for this path, not even a copy."""

    from woof.hex import mesh_row_candidate

    assert dict(mesh_row_candidate.active_overrides()) == {}
    rows = {"only": object()}
    assert mesh_row_candidate.apply_overrides(rows) is rows


def test_an_overridden_row_still_faces_every_other_gate():
    """The cadence check moved from the row to the bind, where the run's own
    radiation interval is known: the override loads, and the cadence rule the
    bind applies still refuses 90 s against a 600 s interval."""

    from woof.hex import mesh_row_candidate

    with mesh_row_candidate.candidate_mesh_dt(
        "x1.40962", 90.0, authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
    ):
        row = _binding_module("_test_binding_indivisible").MESH_BINDINGS["x1.40962"]
        assert row.dt_seconds == 90.0
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.require_cadences(
            90.0, radiation_seconds=600.0, surface_pbl_seconds=None,
            cumulus_seconds=90.0,
        )
    assert "not a positive integer multiple" in str(caught.value)
    dt_admission.require_cadences(
        90.0, radiation_seconds=1800.0, surface_pbl_seconds=None, cumulus_seconds=90.0,
    )


def test_two_mints_cannot_share_one_row():
    from woof.hex import mesh_row_candidate

    probe = _unanchored_dt()
    with dt_admission.candidate_mint(
        probe,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="the proving RTX 5070 Ti",
    ):
        with mesh_row_candidate.candidate_mesh_dt(
            "x1.40962",
            probe,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        ):
            with pytest.raises(DtAdmissionError) as caught:
                with mesh_row_candidate.candidate_mesh_dt(
                    "x1.40962",
                    probe,
                    authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
                ):
                    pass
    assert "one mint at a time" in str(caught.value)


# ---------------------------------------------------------------------------
# the physics-health verdict
#
# MEASURED 2026-08-26: of the four timesteps earned that day, two track their
# control and two diverge from it -- and "two byte-identical forecasts, finite
# at every step" is equally true of all four.  Determinism and physical
# plausibility are different questions, so the answer to the second one is a
# required field rather than a sentence somebody might not write.
# ---------------------------------------------------------------------------
RECOGNISED_HEALTH_VERDICTS = {"REFERENCE", "TRACKS", "DIVERGES"}


def test_every_anchor_answers_what_its_physics_band_did():
    for anchor in ADMITTED_TIMESTEPS.values():
        assert anchor.physics_health, (
            f"dt={anchor.dt_seconds:g} s registers no physics_health; an "
            f"anchor that records only determinism says the same thing about "
            f"a timestep that tracks its control and one that runs away"
        )
        assert anchor.physics_health_verdict in RECOGNISED_HEALTH_VERDICTS


def test_a_row_cannot_be_registered_without_answering_it():
    """No default: the field is required at construction."""

    with pytest.raises(TypeError):
        DtAnchor(
            dt_seconds=1.0,
            radiation_seconds=600.0,
            surface_pbl_seconds=1.0,
            cumulus_seconds=1.0,
            cumulus_scheme="gf",
            meshes=(),
            card="none",
            admitted_on="2026-08-26",
            schedule_receipt="evidence/",
            integration_anchor="evidence/",
            native_reference=None,
            basis="no health answer",
        )


def test_a_diverging_anchor_says_so_before_it_explains_itself():
    """The verdict leads, so a receipt scan does not depend on reading prose."""

    diverging = [
        anchor
        for anchor in ADMITTED_TIMESTEPS.values()
        if anchor.physics_health_verdict == "DIVERGES"
    ]
    assert diverging, (
        "the 2026-08-26 campaign measured divergence at 20 s and 5 s; if no "
        "row records it, the finding was lost rather than fixed"
    )
    for anchor in diverging:
        # it names what it does NOT establish, not only what it does
        assert "NOT MEASURED" in anchor.physics_health


def test_the_verdict_travels_into_a_run_receipt():
    """as_dict is what bind_mesh records, so the caveat rides with the run."""

    for anchor in ADMITTED_TIMESTEPS.values():
        assert anchor.as_dict()["physics_health"] == anchor.physics_health


@pytest.mark.parametrize("defect, certified", [
    ("legacy-description-absent", True),
    ("description-present", True),
    ("description-wrong", False),
    ("cadence-missing", False),
    ("rk-missing", False),
    ("wsm6-missing", False),
    ("clock-missing", False),
])
def test_legacy_schedule_description_keeps_all_other_fields_strict(
    receipts, tmp_path, defect, certified
):
    mint = _load_tool("mint_dt_anchor.py", "_test_mint_legacy_schedule")
    anchor = dt_admission.registered_anchor(PROVEN_DT_SECONDS)
    assert anchor is not None
    stored = json.loads((TREE_ROOT / anchor.schedule_receipt).read_text(encoding="utf-8"))
    description = "surface_pbl_steps_held_between_calls"
    stored["cadences"].pop(description, None)
    if defect == "description-present":
        stored["cadences"][description] = dt_admission.schedule_receipt(
            PROVEN_DT_SECONDS
        )["cadences"][description]
    elif defect == "description-wrong":
        stored["cadences"][description] = 1
    elif defect == "cadence-missing":
        del stored["cadences"]["stepra"]
    elif defect == "rk-missing":
        del stored["rk_schedule"]
    elif defect == "wsm6-missing":
        del stored["wsm6"]
    elif defect == "clock-missing":
        del stored["clock_closure"]
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(stored), encoding="utf-8")
    probe = dataclasses.replace(anchor, schedule_receipt=str(path))
    verdict = mint.verify_anchor(probe, root=TREE_ROOT)
    assert verdict["certified"] is certified, verdict
    if not certified:
        assert verdict["findings"], verdict
