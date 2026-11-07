"""The surface-radiation carrier contract, checked in both directions.

Every test here is in one of two classes, and the class is stated so a
reader can tell what a green run proves:

RECEIPT-IMPLIES-CONSUMPTION -- the contract's record and what the LSM
actually eats are the same object.  A test that only read the record
would pass on a tree where the check was never wired into the consumer
seam, which is the failure mode that let three of these holes ship.

RED-ON-REVERT -- each test names the exact pre-contract behaviour it
would pass under, so a revert of the fix turns it red rather than
leaving it green on a different code path.  The three named holes are
RUC's GSW in daylight, Noah's SWDOWN under radiation-off, and Noah-MP's
COSZEN frozen at its first minute.

These are CPU-side contract tests: they exercise the record, the matrix,
the policy and the refusal text, none of which touch a GPU.  The seam
that calls the check from PhysicsDriver.compute is covered by the driver
suites that run a step.
"""
from __future__ import annotations

import pytest
from woof.config import RunConfig

from woof.core.radiation_carriers import (
    CARRIER_SOURCE_ANALYTIC_GEOMETRY,
    CARRIER_SOURCE_DECLARED_CONSTANT,
    CARRIER_SOURCE_EXTERNAL_ARRAY,
    CARRIER_SOURCE_RADIATION_SCHEME,
    CARRIER_SOURCE_UNWRITTEN,
    CARRIER_SOURCE_WRF_COMPAT_ZERO,
    CARRIER_SOURCES,
    CONSUMER_CARRIERS,
    SURFACE_RADIATION_POLICY_REQUIRED,
    SURFACE_RADIATION_POLICY_WRF_COMPAT_ZERO,
    CarrierContract,
    CarrierContractError,
    consumer_carriers,
    validate_surface_radiation_policy,
)

import numpy as np


def _fields(**values):
    """Host arrays standing in for the driver's device carriers."""
    return {name: np.full((2, 2), float(value), dtype=np.float32)
            for name, value in values.items()}


def _check(contract, scheme, fields, *, model_time=0.0, radt=900.0, dt=20.0):
    contract.check_before_consumption(
        sf_surface_physics=scheme, fields=fields, model_time=model_time,
        radiation_interval_seconds=radt, timestep_seconds=dt)


# --------------------------------------------------------------------
# The matrix itself
# --------------------------------------------------------------------

def test_the_consumer_matrix_covers_all_five_schemes():
    """Pin the matrix.  A row that moves without a thought is a new hole."""

    assert CONSUMER_CARRIERS == {
        0: (),
        1: ("glw", "gsw"),
        2: ("glw", "swdown"),
        3: ("glw", "gsw"),
        4: ("glw", "swdown", "coszen"),
        7: ("glw", "gsw"),
        8: ("glw", "swdown", "coszen"),
    }


def test_a_new_lsm_scheme_refuses_rather_than_consuming_defaults():
    """RED-ON-REVERT: a permissive default is the defect, not the fallback.

    A land-surface scheme that is not in the matrix has never declared
    what sky it eats.  Returning ``()`` for it -- "requires nothing" --
    would let the next ported LSM integrate whatever the buffers held,
    which is precisely how this class of defect reproduces itself.
    """

    with pytest.raises(CarrierContractError) as caught:
        consumer_carriers(5)
    message = str(caught.value)
    assert "sf_surface_physics=5" in message
    assert "consumer matrix" in message
    assert "CONSUMER_CARRIERS" in message


def test_the_check_refuses_an_unmatrixed_scheme_at_the_consumer_seam():
    """The same refusal, reached through the consumption check itself."""

    contract = CarrierContract()
    with pytest.raises(CarrierContractError):
        _check(contract, 5, _fields(glw=350.0))


def test_no_lsm_requires_no_carrier():
    """An atmosphere-only run is not this contract's business."""

    contract = CarrierContract()
    _check(contract, 0, {})       # no raise, no records needed


# --------------------------------------------------------------------
# The three named holes
# --------------------------------------------------------------------

def test_ruc_gsw_in_daylight_refuses_when_nothing_wrote_it():
    """RED-ON-REVERT: RUC's GSW, frozen at the allocation zeros.

    Pre-contract, ``initialize_physics`` allocated ``f["gsw"]`` as zeros
    and a radiation-free run never wrote it again, so RUC integrated a
    net shortwave of exactly zero for the whole forecast -- at local noon
    as much as at midnight.  Zero is plausible at night, which is why no
    value test could find this and why the check is on the SOURCE.
    """

    contract = CarrierContract()
    contract.declare("glw", source=CARRIER_SOURCE_DECLARED_CONSTANT)
    contract.declare("gsw", source=CARRIER_SOURCE_UNWRITTEN)
    with pytest.raises(CarrierContractError) as caught:
        _check(contract, 3, _fields(glw=300.0, gsw=0.0), model_time=43200.0)
    message = str(caught.value)
    assert "GSW" in message                     # names the carrier
    assert "RUC" in message                     # names the consumer
    assert "sf_surface_physics=3" in message
    assert "no producer" in message
    assert "ra_lw_physics=4" in message         # names the fix


def test_noah_swdown_in_daylight_refuses_when_nothing_wrote_it():
    """RED-ON-REVERT: Noah's SWDOWN under a radiation-off configuration."""

    contract = CarrierContract()
    contract.declare("glw", source=CARRIER_SOURCE_DECLARED_CONSTANT)
    contract.declare("swdown", source=CARRIER_SOURCE_UNWRITTEN)
    with pytest.raises(CarrierContractError) as caught:
        _check(contract, 2, _fields(glw=300.0, swdown=0.0),
               model_time=43200.0)
    message = str(caught.value)
    assert "SWDOWN" in message
    assert "Noah" in message
    assert "allocation fill" in message


def test_stale_coszen_refuses_even_though_its_value_is_finite():
    """RED-ON-REVERT: Noah-MP's COSZEN, written once and never again.

    Pre-contract, a radiation-free Noah-MP run seeded COSZEN at the
    half-interval hour angle of its first radiation slot and then left it
    alone, so a twelve-hour forecast computed its canopy radiative
    transfer against the sun angle of its first minute.  The value stayed
    finite and in range the whole time; only its AGE says anything is
    wrong, which is why the contract carries one.
    """

    contract = CarrierContract()
    contract.declare("glw", source=CARRIER_SOURCE_RADIATION_SCHEME,
                     model_time=43200.0)
    contract.declare("swdown", source=CARRIER_SOURCE_RADIATION_SCHEME,
                     model_time=43200.0)
    contract.declare("coszen", source=CARRIER_SOURCE_ANALYTIC_GEOMETRY,
                     model_time=0.0)
    with pytest.raises(CarrierContractError) as caught:
        _check(contract, 4, _fields(glw=350.0, swdown=800.0, coszen=0.31),
               model_time=43200.0, radt=900.0, dt=20.0)
    message = str(caught.value)
    assert "COSZEN" in message
    assert "stale" in message
    assert "Noah-MP" in message
    assert "43200" in message                   # the consuming instant
    assert "920" in message                     # radt 900 + one step 20


def test_a_fresh_analytic_coszen_passes():
    """The negative control: the same carrier, refreshed on cadence."""

    contract = CarrierContract()
    for name in ("glw", "swdown"):
        contract.declare(name, source=CARRIER_SOURCE_RADIATION_SCHEME,
                         model_time=43200.0)
    contract.declare("coszen", source=CARRIER_SOURCE_ANALYTIC_GEOMETRY,
                     model_time=43200.0)
    _check(contract, 4, _fields(glw=350.0, swdown=800.0, coszen=0.9),
           model_time=43200.0)


# --------------------------------------------------------------------
# Night, and why zero passes at night and fails at noon
# --------------------------------------------------------------------

def test_zero_shortwave_at_night_passes_on_source_and_age_not_plausibility():
    """A scheme that computed zero is a scheme that ran.

    The distinction this test exists to pin: the SAME numbers pass here
    and fail in the daylight tests above, and the difference is entirely
    in the provenance record.  A contract that reasoned about whether a
    number looked right would have admitted the 300 W m-2 that started
    all of this, because 300 looks right everywhere.
    """

    contract = CarrierContract()
    contract.declare("glw", source=CARRIER_SOURCE_RADIATION_SCHEME,
                     model_time=3600.0)
    contract.declare("swdown", source=CARRIER_SOURCE_RADIATION_SCHEME,
                     model_time=3600.0)
    _check(contract, 2, _fields(glw=310.0, swdown=0.0), model_time=3600.0)


def test_a_nonfinite_carrier_refuses_before_the_lsm_call():
    contract = CarrierContract()
    contract.declare("glw", source=CARRIER_SOURCE_RADIATION_SCHEME,
                     model_time=0.0)
    contract.declare("swdown", source=CARRIER_SOURCE_RADIATION_SCHEME,
                     model_time=0.0)
    fields = _fields(glw=300.0, swdown=0.0)
    fields["glw"][1, 1] = np.nan
    with pytest.raises(CarrierContractError) as caught:
        _check(contract, 2, fields)
    assert "not finite" in str(caught.value)


# --------------------------------------------------------------------
# The escape
# --------------------------------------------------------------------

def test_the_escape_labels_every_carrier_it_touches():
    """``wrf_compat_zero`` consumes, but never silently.

    The old behaviour is reproduced ONLY under the declared escape, and
    the reproduction is labelled: every carrier the escape admitted
    carries ``wrf_compat_zero`` in the record, so the receipt and the
    checkpoint both say what happened.  An escape that consumed without
    labelling would be the original defect with a config key on it.
    """

    contract = CarrierContract(SURFACE_RADIATION_POLICY_WRF_COMPAT_ZERO)
    _check(contract, 3, _fields(glw=0.0, gsw=0.0), model_time=43200.0)
    assert contract.source("glw") == CARRIER_SOURCE_WRF_COMPAT_ZERO
    assert contract.source("gsw") == CARRIER_SOURCE_WRF_COMPAT_ZERO
    rows = contract.report(_fields(glw=0.0, gsw=0.0))
    assert rows["gsw"]["source"] == CARRIER_SOURCE_WRF_COMPAT_ZERO
    assert rows["gsw"]["value"] == 0.0


def test_the_escape_is_never_selected_automatically():
    """The default is ``required`` and nothing in the contract changes it."""

    assert CarrierContract().policy == SURFACE_RADIATION_POLICY_REQUIRED
    contract = CarrierContract()
    with pytest.raises(CarrierContractError):
        _check(contract, 2, _fields(glw=0.0, swdown=0.0))
    assert contract.policy == SURFACE_RADIATION_POLICY_REQUIRED


def test_a_misspelled_policy_refuses_and_names_both_choices():
    with pytest.raises(ValueError) as caught:
        validate_surface_radiation_policy("wrf-compat-zero")
    message = str(caught.value)
    assert "wrf-compat-zero" in message
    assert "required" in message
    assert "wrf_compat_zero" in message
    assert "experimental forcing" in message


def test_a_declared_constant_is_documented_as_forcing_not_availability():
    """A typed GLW is a producer for the check and a forcing in the record.

    Both halves matter.  It passes -- a caller who typed a number owns the
    consequence and an idealised column may legitimately want a fixed sky
    -- and it is stored under its own source, so nothing downstream can
    read it as "radiation available".
    """

    contract = CarrierContract()
    contract.declare("glw", source=CARRIER_SOURCE_DECLARED_CONSTANT)
    contract.declare("swdown", source=CARRIER_SOURCE_EXTERNAL_ARRAY)
    _check(contract, 2, _fields(glw=300.0, swdown=500.0))
    assert contract.source("glw") == CARRIER_SOURCE_DECLARED_CONSTANT
    assert contract.source("glw") != CARRIER_SOURCE_RADIATION_SCHEME


# --------------------------------------------------------------------
# Provenance vocabulary and serialization
# --------------------------------------------------------------------

def test_an_unknown_source_is_refused_at_declaration():
    """A misspelled provenance must never reach a checkpoint."""

    contract = CarrierContract()
    with pytest.raises(CarrierContractError):
        contract.declare("glw", source="radiation-scheme", model_time=0.0)


def test_a_producing_source_must_say_when_it_ran():
    contract = CarrierContract()
    with pytest.raises(CarrierContractError):
        contract.declare("glw", source=CARRIER_SOURCE_RADIATION_SCHEME)


def test_the_record_round_trips_through_a_checkpoint():
    """RESTART: provenance and age survive, because they cannot be rebuilt."""

    contract = CarrierContract()
    contract.declare("glw", source=CARRIER_SOURCE_RADIATION_SCHEME,
                     model_time=3600.0)
    contract.declare("swdown", source=CARRIER_SOURCE_DECLARED_CONSTANT)
    state = contract.state()

    resumed = CarrierContract()
    resumed.restore(state)
    assert resumed.source("glw") == CARRIER_SOURCE_RADIATION_SCHEME
    assert resumed.record("glw").last_update_model_time == 3600.0
    assert resumed.record("swdown").last_update_model_time is None
    # The property that matters: the resumed contract answers the
    # consumption question the same way the uninterrupted one does, on a
    # step where no radiation call is due.
    fields = _fields(glw=350.0, swdown=0.0)
    _check(contract, 2, fields, model_time=3620.0)
    _check(resumed, 2, fields, model_time=3620.0)


def test_a_checkpoint_source_outside_this_builds_vocabulary_refuses():
    resumed = CarrierContract()
    with pytest.raises(CarrierContractError):
        resumed.restore({"glw": {"source": "sky_god",
                                 "last_update_model_time": None}})


def test_the_source_vocabulary_includes_the_shared_cam_producer():
    assert CARRIER_SOURCES == {
        "radiation_scheme", "declared_constant", "external_array",
        "analytic_geometry", "wrf_compat_zero", "unwritten", "cam_ozone",
    }


# --------------------------------------------------------------------
# DA: the published 2 m fields after an analysis
# --------------------------------------------------------------------

def test_the_da_refresh_seam_reruns_the_final_two_metre_provider():
    """RECEIPT-IMPLIES-CONSUMPTION, on the DA side.

    ``woof.ensemble.member.refresh_diagnostics`` is the post-condition
    every increment applier already calls.  T2 and Q2 are diagnosed, not
    prognostic, so an analysis that moved the lowest model level leaves
    the PUBLISHED pair describing the pre-analysis state -- and the next
    output frame carries them out as though they described the analysis.
    The rerun rides this seam rather than a cycle driver, so a driver
    cannot forget it and a new applier inherits it.

    The receipt field is three-valued on purpose and the test pins all
    three meanings: None when no driver is attached, False when the
    scheme's 2 m provider lives inside its LSM step and cannot be rerun
    without rerunning the LSM, True when it reran.
    """
    from types import SimpleNamespace

    from woof.ensemble import member

    calls = []

    class _Driver:
        def __init__(self, answer):
            self.answer = answer

        def refresh_surface_diagnostics_after_analysis(self, state):
            calls.append(state)
            return self.answer

    state = SimpleNamespace(physics=None)
    monkey = {}

    def _update(_state, _hypsometric_opt):
        monkey["ran"] = True

    def _sha(_state):
        return "sha"

    import woof.core.diagnostics as diagnostics
    original_update = diagnostics.update_diagnostics
    original_sha = member.diagnostics_sha256
    diagnostics.update_diagnostics = _update
    member.diagnostics_sha256 = _sha
    try:
        assert member.refresh_diagnostics(
            state, hypsometric_opt=2)["surface_diagnostics_reran"] is None
        assert not calls

        state.physics = _Driver(False)
        assert member.refresh_diagnostics(
            state, hypsometric_opt=2)["surface_diagnostics_reran"] is False
        assert len(calls) == 1

        state.physics = _Driver(True)
        assert member.refresh_diagnostics(
            state, hypsometric_opt=2)["surface_diagnostics_reran"] is True
        assert len(calls) == 2
        assert calls[-1] is state
    finally:
        diagnostics.update_diagnostics = original_update
        member.diagnostics_sha256 = original_sha


def test_the_provider_rerun_dispatches_on_the_resolved_runner():
    """Only the scheme whose 2 m provider is OUTSIDE its LSM step reruns.

    Noah's SFCDIAGS is a separate call the driver makes after the LSM, so
    it can be rerun against an analysed column.  RUC and Noah-MP compute
    their 2 m fields inside their own LSM steps, and rerunning those would
    be advancing physics rather than republishing a diagnostic.  The test
    reads the resolved runner name, which is what the method dispatches
    on, so a scheme renumbering cannot silently redirect it.
    """
    from woof.core.physics import PhysicsDriver

    driver = object.__new__(PhysicsDriver)
    driver.surface_enabled = True
    driver.scheme_dispatch = {"sf_surface_physics": "_run_ruc"}
    assert driver.refresh_surface_diagnostics_after_analysis(None) is False

    driver.scheme_dispatch = {"sf_surface_physics": "_run_noahmp"}
    assert driver.refresh_surface_diagnostics_after_analysis(None) is False

    driver.surface_enabled = False
    driver.scheme_dispatch = {"sf_surface_physics": "_run_noah"}
    assert driver.refresh_surface_diagnostics_after_analysis(None) is False


# --------------------------------------------------------------------
# The config key: spelled wrong, and valued wrong
# --------------------------------------------------------------------

def test_the_policy_key_is_a_run_field_so_a_misspelling_refuses():
    """The misspelling refusal is INHERITED, and that is stated here.

    ``woof.experiment`` builds ``[shared]``'s known-key set from
    ``fields(RunConfig)`` and calls ``_reject_unknown_keys`` on it, so a
    key that is a RunConfig field gets misspelling refusal by
    construction rather than by a list someone remembered to extend.
    That is why this lane adds no entry to any optional-key pin: there is
    no pin for this table, and adding one would be a second, drifting
    answer to a question the field set already answers.

    Pinned from both sides -- the field exists with the documented
    default, and a neighbouring name is NOT a field, so the refusal has
    something to refuse.
    """
    from dataclasses import fields as dataclass_fields

    from woof.config import RunConfig

    names = {f.name: f for f in dataclass_fields(RunConfig)}
    assert "surface_radiation_policy" in names
    assert names["surface_radiation_policy"].default == (
        SURFACE_RADIATION_POLICY_REQUIRED)
    assert "surface_radiation_policies" not in names
    assert "surface_radiaton_policy" not in names


def test_validate_run_config_refuses_a_bad_policy_value():
    """The value check is at config load, not only at the consumer seam."""

    from woof.config import RunConfig, validate_run_config

    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000.0, dy=1000.0, dt=1.0,
                    ztop=12000.0, run_seconds=60.0,
                    surface_radiation_policy="wrf_compat")
    with pytest.raises(ValueError) as caught:
        validate_run_config(cfg)
    assert "surface_radiation_policy" in str(caught.value)


def test_the_policy_vocabulary_is_importable_without_the_core():
    """The standalone RW-WPS wheel carries woof.config and not woof.core.

    A config-load refusal that reached into the core would hand a
    standalone preprocessing user an ImportError instead of the sentence
    written for them.  That defect withdrew 1.8.8, so the direction of
    this import is pinned rather than left to be rediscovered.
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).parents[1] / "woof" / "config.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    core_imports = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module and node.module.startswith("woof.core")}
    # config.py legitimately reaches ONE core module today
    # (woof.core.sase_limits), which the wheel therefore carries.  The
    # assertion is not "no core imports" -- that would be a rule this
    # tree does not hold -- but that the carrier contract is not among
    # them, because it is the one this lane would otherwise have added
    # and woof/core/radiation_carriers.py imports FROM here.
    assert not any(module.startswith("woof.core.radiation_carriers")
                   for module in core_imports), sorted(core_imports)
    # And the other direction, so the arrow is pinned rather than
    # assumed: the contract takes its vocabulary from the config module.
    contract = (Path(__file__).parents[1] / "woof" / "core"
                / "radiation_carriers.py").read_text(encoding="utf-8")
    assert "from woof.config import" in contract


# --------------------------------------------------------------------
# Consumer-cell awareness: refusal protects consumption, cell by cell
# --------------------------------------------------------------------

def _footprint(xland_value, *, land_cells=0, ice_cells=0):
    """A 2x2 xland/xice pair with the named number of consumer cells."""
    xland = np.full((2, 2), float(xland_value), dtype=np.float32)
    xice = np.zeros((2, 2), dtype=np.float32)
    flat_land, flat_ice = xland.reshape(-1), xice.reshape(-1)
    for k in range(land_cells):
        flat_land[k] = 1.0
    for k in range(ice_cells):
        flat_ice[-1 - k] = 0.6
    return {"xland": xland, "xice": xice}


def test_an_all_water_domain_consumes_nothing_and_is_not_refused():
    """RECEIPT-IMPLIES-CONSUMPTION, inverted: zero consumers, zero checks.

    Noah, RUC and Noah-MP all skip open-water columns, so a domain that
    is entirely open water feeds its land-surface scheme exactly zero
    cells and an unwritten carrier corrupts nothing.  This is the
    open-water invariant test's configuration
    (tests/test_open_water_longwave_invariant.py): all water, radiation
    off, Noah selected, SWDOWN never written -- a legitimate idealised
    marine setup the contract must let run.
    """
    contract = CarrierContract()          # nothing declared at all
    fields = _footprint(2.0)              # xland=2 everywhere, no ice
    _check(contract, 2, fields)           # does not raise


def test_one_land_cell_restores_the_full_refusal():
    """RED-ON-REVERT for the footprint itself: consumption means checking."""
    contract = CarrierContract()
    fields = _footprint(2.0, land_cells=1)
    with pytest.raises(CarrierContractError, match="has no producer"):
        _check(contract, 2, fields)


def test_sea_ice_on_open_water_counts_as_a_consumer():
    """Ice is over-approximated: any xice > 0 keeps the check armed.

    The schemes threshold sea ice at 0.5 before treating a column as
    land-like; the footprint counts any nonzero ice instead, so it can
    only over-state the consumer set and the refusal can only fire more
    often than strictly needed, never less.
    """
    contract = CarrierContract()
    fields = _footprint(2.0, ice_cells=1)
    with pytest.raises(CarrierContractError, match="has no producer"):
        _check(contract, 2, fields)


def test_an_unknown_footprint_is_treated_as_consuming():
    """FAIL-CLOSED: no xland to read means the check runs in full."""
    contract = CarrierContract()
    with pytest.raises(CarrierContractError, match="has no producer"):
        _check(contract, 2, _fields(glw=300.0))   # carriers, no footprint
    with pytest.raises(CarrierContractError, match="has no producer"):
        _check(contract, 2, None)


# --------------------------------------------------------------------
# The check under a deferred-health region (CUDA graph capture)
# --------------------------------------------------------------------
# Blocking device-to-host reads are refused outright inside CUDA graph
# capture, and tilestream wraps captured steps in
# ``health_ledger.deferring`` -- both graph-replay rungs of the 2.0
# tiles gate died on the footprint read below.  Under an active ledger
# the check therefore (a) skips the footprint read and runs in full,
# which is the documented fail-closed direction, and (b) records the
# finiteness verdict into the ledger instead of reading it, so the
# refusal arrives at the sweep drain with its type intact.

def _declared_pair(contract):
    contract.declare("glw", source=CARRIER_SOURCE_DECLARED_CONSTANT)
    contract.declare("swdown", source=CARRIER_SOURCE_DECLARED_CONSTANT)


def test_a_host_carrier_keeps_the_immediate_refusal_under_a_ledger():
    """Host arrays have no capture problem, so nothing is deferred."""
    from woof.core import health_ledger

    class _NeverReached:
        def record(self, *args, **kwargs):     # pragma: no cover
            raise AssertionError("a host carrier must be read immediately")

    contract = CarrierContract()
    _declared_pair(contract)
    fields = _fields(glw=300.0, swdown=0.0)
    fields["glw"][1, 1] = np.nan
    with health_ledger.deferring(_NeverReached()):
        with pytest.raises(CarrierContractError, match="not finite"):
            _check(contract, 2, fields)


def test_an_active_ledger_over_enforces_the_footprint_never_under():
    """All-water passes eagerly; under a ledger the check runs in full.

    The footprint read is the D2H the ledger exists to remove, so a
    deferred check treats every cell as consuming -- the same fail-closed
    default an unreadable footprint already takes.  The refusal an
    all-water captured domain gains is an over-enforcement, and this test
    states it on purpose so a change that flips it to UNDER-enforcement
    (skipping the check when it cannot read) goes red.
    """
    from woof.core import health_ledger

    class _InertLedger:
        def record(self, *args, **kwargs):     # pragma: no cover
            raise AssertionError("no device carrier is present to record")

    contract = CarrierContract()               # nothing declared
    fields = _footprint(2.0)                   # all water: eager passes
    _check(contract, 2, fields)
    with health_ledger.deferring(_InertLedger()):
        with pytest.raises(CarrierContractError, match="has no producer"):
            _check(contract, 2, fields)


def test_a_device_nonfinite_verdict_is_recorded_and_raised_at_the_drain():
    """The deferred half, on a card: no raise inside, THIS type at drain."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")
    from woof.core import health_ledger

    contract = CarrierContract()
    _declared_pair(contract)
    fields = {name: cp.asarray(value)
              for name, value in _fields(glw=300.0, swdown=0.0).items()}
    fields["glw"][1, 1] = cp.float32(np.nan)
    ledger = health_ledger.HealthLedger()
    with health_ledger.deferring(ledger):
        _check(contract, 2, fields)            # records, must not raise
    with pytest.raises(CarrierContractError) as caught:
        ledger.drain()
    message = str(caught.value)
    assert "not finite" in message
    assert "deferred health drain" in message
    # Drained is cleared: a healthy next sweep does not re-raise history.
    ledger.drain()


def test_the_check_is_capturable_inside_a_cuda_graph():
    """The seam the tiles gate graph rungs exercise, as a unit test.

    Under an active ledger the whole consumption check must complete
    inside CUDA stream capture -- no D2H, no GraphCaptureError -- and the
    captured finiteness reduction must keep working on REPLAY: the same
    graph, replayed after the carrier goes non-finite, flags the ledger.
    """
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")
    from woof.core import health_ledger

    contract = CarrierContract()
    _declared_pair(contract)
    fields = {name: cp.asarray(value)
              for name, value in _fields(glw=300.0, swdown=0.0).items()}
    ledger = health_ledger.HealthLedger()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        with health_ledger.deferring(ledger):
            # Warm the reduction kernels eagerly-on-device first, so the
            # capture below records only settled launches.
            _check(contract, 2, fields)
            stream.synchronize()
            stream.begin_capture()
            try:
                _check(contract, 2, fields)
            finally:
                graph = stream.end_capture()
        graph.launch(stream)
        stream.synchronize()
    ledger.drain()                             # healthy: nothing flagged
    fields["glw"][0, 0] = cp.float32(np.nan)
    with stream:
        graph.launch(stream)
        stream.synchronize()
    with pytest.raises(CarrierContractError, match="not finite"):
        ledger.drain()


# --------------------------------------------------------------------
# The consumer seam itself: the refusal fires through compute()
# --------------------------------------------------------------------

def _hand_built_compute_driver(monkeypatch, *, carriers):
    """A hand-built PhysicsDriver aimed at the LSM branch of compute().

    Mirrors the driver suite's CPU fixture: object.__new__, stub schemes,
    NumPy in place of CuPy.  What it does NOT do is pre-seed the carrier
    records -- the point is that the refusal must come out of
    ``PhysicsDriver.compute`` itself, through the wired
    ``check_before_consumption`` call, not out of a test calling the
    check directly.
    """
    from types import SimpleNamespace
    import woof.core.physics as physics
    from woof.config import RRTMG_VARIANT_LEGACY

    monkeypatch.setattr(physics, "cp", np)
    atmosphere = {
        "p_interface": np.array(
            [[[100000.0, 90000.0]], [[95000.0, 85000.0]]], np.float32),
        "qv": np.array([[[9.0e-3, 7.0e-3]]], np.float32),
    }
    monkeypatch.setattr(
        physics, "_prepare_atmosphere", lambda state: atmosphere)
    state = SimpleNamespace(elapsed_seconds=0.0)
    cfg = RunConfig(nx=2, ny=1, nz=1, dx=1000., dy=1000.,
        ztop=10000., run_seconds=0.,
        dt=60.0, bldt=0.0, ra_physics=0, sf_sfclay_physics=91,
        sf_surface_physics=2, bl_pbl_physics=1, cu_physics=0,
        # The 2.9.1 radiation boundary-flux accumulator reads the variant
        # (radiation off: no fluxes either way).
        ra_rrtmg_variant=RRTMG_VARIANT_LEGACY,
        sst_update=0, tmn_update=0)
    driver = object.__new__(physics.PhysicsDriver)
    driver.farms = None
    driver.ruc_params = None
    driver.pxlsm_params = None
    driver.swint = None
    driver.state = state
    driver.fields = {
        name: np.full((1, 2), value, np.float32)
        for name, value in {
            "psfc": -123.0, "tsk": 290.0, "hfx": 0.0, "qfx": 0.0,
            "qsfc": 0.0, "chs2": 0.0, "cqs2": 0.0,
            "t2": -999.0, "q2": -999.0, "th2": -999.0, "xland": 1.0,
        }.items()
    }
    driver.surface_enabled = True
    driver.carriers = carriers
    # The normal constructor carries this inactive slot on radiation-off
    # runs; the synthetic compute driver must reach the carrier contract.
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
    driver._compose_tendencies = lambda cfg: None
    driver._run_sfclay = lambda *_: None
    driver._run_ysu = lambda *_: None
    reached = []
    driver._run_noah = lambda *_: reached.append("noah")
    return physics, state, cfg, driver, reached


def test_the_refusal_fires_through_compute_with_no_preseeded_contract(
        monkeypatch):
    """THE SEAM (RED-ON-REVERT for physics.py's wiring itself).

    The contract object exists but carries no records, exactly what a
    driver whose producers never ran holds.  ``compute`` must refuse
    before ``_run_noah`` -- delete the ``check_before_consumption`` call
    from ``PhysicsDriver.compute`` and this test goes red by running the
    LSM stub and raising nothing.  (Verified red in that state when this
    test was written.)
    """
    physics, state, cfg, driver, reached = _hand_built_compute_driver(
        monkeypatch, carriers=CarrierContract())
    with pytest.raises(CarrierContractError, match="has no producer"):
        driver.compute(state, cfg)
    assert reached == [], "Noah ran before the contract refused"


def test_a_leg_that_begins_off_the_radiation_cadence_runs_its_producer(
        monkeypatch):
    """THE CYCLE LEG.  A driver's first step is not always itimestep 1.

    WRF's mandatory itimestep == 1 radiation call is what seeds GLW
    before the land surface consumes it, and ``itimestep`` inside
    ``compute`` is reconstructed from the domain's own elapsed model
    time.  A DA cycle leg is built fresh on an analysis state that is
    already some way into the run, so its first step lands at whatever
    phase of the radiation cadence that time falls on -- here model
    second 900 against a 12 minute cadence and a 60 s step, which is
    step 16 of a 12-step interval and not due.  The carriers are seeded
    unwritten because radiation is their producer, so the contract
    refused at the leg's first surface call and every cycle under a
    both-streams profile died there.

    The producer runs instead.  The refusal is untouched for the
    configurations that have no producer: the test above still meets it
    with radiation off.
    """
    physics, state, cfg, driver, reached = _hand_built_compute_driver(
        monkeypatch, carriers=CarrierContract())
    state.elapsed_seconds = 900.0
    from dataclasses import replace
    cfg = replace(cfg, ra_lw_physics=4, ra_sw_physics=4)
    driver.carriers.declare("glw", source=CARRIER_SOURCE_UNWRITTEN)
    driver.carriers.declare("swdown", source=CARRIER_SOURCE_UNWRITTEN)
    driver._refresh_surface_diagnostics = lambda atmosphere, **kwargs: None

    def _radiation(atmosphere, state, cfg):
        reached.append("radiation")
        for name in ("glw", "swdown"):
            driver.carriers.declare(
                name, source=CARRIER_SOURCE_RADIATION_SCHEME,
                model_time=float(state.elapsed_seconds))

    driver._run_radiation = _radiation
    driver.compute(state, cfg)
    assert reached == ["radiation", "noah"], reached
    assert driver.carriers.record("glw").source == (
        CARRIER_SOURCE_RADIATION_SCHEME)
    assert driver.carriers_need_producer_refresh is False
    # And it is once: the second step of the leg is on cadence and the
    # carriers now have a producer, so nothing forces another call.
    reached.clear()
    state.elapsed_seconds = 960.0
    driver.compute(state, cfg)
    assert reached == ["noah"], reached


def test_the_same_leg_meets_the_refusal_when_nothing_asks_for_a_producer(
        monkeypatch):
    """RED ON REVERT for the branch above, in one file with it.

    The only thing standing between this leg and the refusal it used to
    die on is the driver asking the contract which consumed carriers
    have no source.  Stub that question empty -- the state of the tree
    before the fix, where nobody asked -- and the same leg, the same
    profile and the same model time meet ``has no producer`` again.
    """
    physics, state, cfg, driver, reached = _hand_built_compute_driver(
        monkeypatch, carriers=CarrierContract())
    state.elapsed_seconds = 900.0
    from dataclasses import replace
    cfg = replace(cfg, ra_lw_physics=4, ra_sw_physics=4)
    driver._refresh_surface_diagnostics = lambda atmosphere, **kwargs: None
    driver._run_radiation = lambda *_: reached.append("radiation")
    monkeypatch.setattr(CarrierContract, "unsourced_consumed",
                        lambda self, sf_surface_physics: ())
    with pytest.raises(CarrierContractError, match="has no producer"):
        driver.compute(state, cfg)
    assert reached == [], reached


def test_the_contract_reports_which_consumed_carriers_have_no_source():
    """The question the driver asks ahead of the consumption check."""
    contract = CarrierContract()
    # A contract with no records at all is every carrier unwritten, the
    # same answer an explicitly unwritten record gives: nothing has
    # written them either way.
    assert set(contract.unsourced_consumed(2)) == {"glw", "swdown"}
    contract.declare("glw", source=CARRIER_SOURCE_UNWRITTEN)
    contract.declare("swdown", source=CARRIER_SOURCE_UNWRITTEN)
    assert set(contract.unsourced_consumed(2)) == {"glw", "swdown"}
    contract.declare("glw", source=CARRIER_SOURCE_RADIATION_SCHEME,
                     model_time=0.0)
    assert contract.unsourced_consumed(2) == ("swdown",)
    # No land-surface scheme consumes nothing, and asks for nothing.
    assert contract.unsourced_consumed(0) == ()


def test_compute_refuses_a_driver_assembled_without_any_contract(
        monkeypatch):
    """The None arm of the same seam: skipping the contract is refused."""
    physics, state, cfg, driver, reached = _hand_built_compute_driver(
        monkeypatch, carriers=None)
    with pytest.raises(CarrierContractError,
                       match="assembled without a"):
        driver.compute(state, cfg)
    assert reached == []


def test_set_forcing_without_a_contract_refuses_rather_than_crashing(
        monkeypatch):
    """The forcing door's None guard, same sentence family as its siblings."""
    physics, state, cfg, driver, reached = _hand_built_compute_driver(
        monkeypatch, carriers=None)
    driver.fields["glw"] = np.zeros((1, 2), np.float32)
    with pytest.raises(CarrierContractError,
                       match="without a surface-radiation "
                             "carrier contract to record"):
        driver.set_forcing(glw=310.0)
    # rainbl alone takes no provenance and needs no contract.
    driver.fields["rainbl"] = np.zeros((1, 2), np.float32)
    driver.set_forcing(rainbl=np.zeros((1, 2), np.float32))


# --------------------------------------------------------------------
# The promised receipt surfaces: report() reaches wrfout metadata
# --------------------------------------------------------------------

def test_carrier_provenance_attrs_carry_policy_source_and_age():
    from types import SimpleNamespace

    from woof.io.wrfout import carrier_provenance_attrs

    contract = CarrierContract()
    contract.declare("glw", source=CARRIER_SOURCE_RADIATION_SCHEME,
                     model_time=720.0)
    contract.declare("swdown", source=CARRIER_SOURCE_DECLARED_CONSTANT)
    attrs = carrier_provenance_attrs(SimpleNamespace(carriers=contract))

    assert attrs["GPUWM_SURFACE_RADIATION_POLICY"] == \
        SURFACE_RADIATION_POLICY_REQUIRED
    assert attrs["GPUWM_CARRIER_GLW_SOURCE"] == \
        CARRIER_SOURCE_RADIATION_SCHEME
    assert float(attrs["GPUWM_CARRIER_GLW_LAST_UPDATE"]) == 720.0
    assert attrs["GPUWM_CARRIER_SWDOWN_SOURCE"] == \
        CARRIER_SOURCE_DECLARED_CONSTANT
    # -1.0 is the documented "constant by declaration, has no age".
    assert float(attrs["GPUWM_CARRIER_SWDOWN_LAST_UPDATE"]) == -1.0
    # A state with no driver or no contract stamps nothing rather than
    # a fabricated row.
    assert carrier_provenance_attrs(None) == {}
    assert carrier_provenance_attrs(
        SimpleNamespace(carriers=None)) == {}


def test_the_per_domain_submit_stamps_carrier_provenance_per_frame():
    """The wiring, on the real ``PerDomainWrfoutWriters.submit`` method."""
    from datetime import datetime
    from pathlib import Path
    from types import SimpleNamespace

    from woof.io.wrfout import PerDomainWrfoutWriters

    class _FakeWriter:
        global_attrs = {"TITLE": "held"}

        def __init__(self):
            self.calls = []
            self.completed_observers = []
            self.history_selections = []

        def submit(self, path, valid_time, state, *, extra_fields=None,
                   refl_field=None, global_attrs=None,
                   completed_observer=None, history_selection=None):
            self.calls.append(global_attrs)
            self.completed_observers.append(completed_observer)
            self.history_selections.append(history_selection)

    contract = CarrierContract()
    contract.declare("glw", source=CARRIER_SOURCE_RADIATION_SCHEME,
                     model_time=0.0)
    fake = _FakeWriter()
    writers = object.__new__(PerDomainWrfoutWriters)
    writers.output_dir = Path(".")
    writers.start_time = datetime(2000, 1, 1)
    writers._writers = {1: fake}
    writers._metadata_by_grid_id = {1: {}}
    # The nest-lifecycle landing (6e4c28654, retire/rearm episodes) gave
    # submit an episode read; empty means episode 0, the un-respawned
    # domain every prior frame described.
    writers._episode_by_grid_id = {}
    # The duplicate-frame guard (0da88f7c2) tracks what THIS run already
    # published; empty is the state at first submit.
    writers._published_paths = set()
    node = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1, run=RunConfig(
            nx=2, ny=1, nz=1, dx=1000., dy=1000., ztop=10000.,
            dt=60., run_seconds=0.)),
        clock=SimpleNamespace(tick_den=1),
        state=SimpleNamespace(physics=SimpleNamespace(carriers=contract)))

    writers.submit(node, 0)
    (attrs,) = fake.calls
    assert attrs["TITLE"] == "held"            # base attrs kept
    assert attrs["GPUWM_CARRIER_GLW_SOURCE"] == \
        CARRIER_SOURCE_RADIATION_SCHEME
    assert fake.completed_observers == [None]
    assert fake.history_selections == [None]
    assert writers._writers[1] is fake

    # A state without a driver stamps nothing: the ticket rides with the
    # writer's standing attribute set.
    bare = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1, run=RunConfig(
            nx=2, ny=1, nz=1, dx=1000., dy=1000., ztop=10000.,
            dt=60., run_seconds=0.)),
        clock=SimpleNamespace(tick_den=1),
        state=SimpleNamespace())
    writers.submit(bare, 1)
    assert fake.calls[-1] is None
    assert fake.completed_observers == [None, None]
    assert fake.history_selections == [None, None]
    assert writers._writers[1] is fake


# --------------------------------------------------------------------
# The real checkpoint path: provenance and age survive a restart
# --------------------------------------------------------------------

def test_carrier_provenance_survives_the_real_checkpoint_path(
        monkeypatch, tmp_path):
    """Serialize mid-interval on a non-radiation-due step; restore; compare.

    Through ``woof.io.restart.write_restart`` and ``restore_restart``
    themselves, not a hand round-trip of ``state()``.  The declared
    instant (720 s) is one radiation cadence before the checkpoint, so
    the resumed contract holds an age a rebuilt one could not know: a
    fresh contract would say "unwritten" (and refuse a healthy run) or
    "just written" (and admit a producer that stopped).
    """
    from woof.io import restart
    from test_restart import _cfg, _fill_setup, _shim_driver_state

    cfg = _cfg(moist=True, mp_physics=10, bl_pbl_physics=1, bldt=0.0)
    source, source_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    source_driver.carriers.declare(
        "glw", source=CARRIER_SOURCE_RADIATION_SCHEME, model_time=720.0)
    source_driver.carriers.declare(
        "swdown", source=CARRIER_SOURCE_RADIATION_SCHEME, model_time=720.0)
    uninterrupted = source_driver.carriers.state()

    path = restart.write_restart(tmp_path / "carriers.npz", source, cfg)

    resumed, resumed_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(resumed)
    restart.restore_restart(path, resumed, cfg)

    # Bit-equal: state() carries the float unrounded, and == on finite
    # floats is bit comparison.
    assert resumed_driver.carriers.state() == uninterrupted
    record = resumed_driver.carriers.record("glw")
    assert record.source == CARRIER_SOURCE_RADIATION_SCHEME
    assert record.last_update_model_time == 720.0
    # A carried contract never forces the legacy producer refresh.
    assert resumed_driver.carriers_need_producer_refresh is False


def test_a_pre_contract_checkpoint_forces_one_producer_refresh(
        monkeypatch, tmp_path):
    """The legacy arm: no header mapping means refresh, never guess."""
    import json

    from woof.io import restart
    from test_restart import _cfg, _fill_setup, _shim_driver_state

    cfg = _cfg(moist=True, mp_physics=10, bl_pbl_physics=1, bldt=0.0)
    source, source_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    path = restart.write_restart(tmp_path / "legacy.npz", source, cfg)

    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    header = json.loads(bytes(bytearray(
        payload[restart._HEADER_KEY])).decode("utf-8"))
    header["driver"].pop("carriers", None)
    header["driver"].pop("surface_radiation_policy", None)
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    legacy = tmp_path / "pre-contract.npz"
    with legacy.open("wb") as stream:
        np.savez(stream, **payload)

    resumed, resumed_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(resumed)
    restart.restore_restart(legacy, resumed, cfg)
    assert resumed_driver.carriers_need_producer_refresh is True


def test_a_resume_that_changed_the_policy_is_refused(
        monkeypatch, tmp_path):
    import json

    from woof.io import restart
    from test_restart import _cfg, _fill_setup, _shim_driver_state

    cfg = _cfg(moist=True, mp_physics=10, bl_pbl_physics=1, bldt=0.0)
    source, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    path = restart.write_restart(tmp_path / "policy.npz", source, cfg)

    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    header = json.loads(bytes(bytearray(
        payload[restart._HEADER_KEY])).decode("utf-8"))
    header["driver"]["surface_radiation_policy"] = \
        SURFACE_RADIATION_POLICY_WRF_COMPAT_ZERO
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    swapped = tmp_path / "policy-swapped.npz"
    with swapped.open("wb") as stream:
        np.savez(stream, **payload)

    resumed, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(resumed)
    with pytest.raises(restart.RestartMismatchError,
                       match="surface_radiation_policy"):
        restart.restore_restart(swapped, resumed, cfg)
