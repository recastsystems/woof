"""The carrier contract's produced-at ledger survives the streamed sweep.

THE DEFECT THIS PINS (2.0 blocker, measured on the streamed real-case
twin): the freshness law (woof/core/radiation_carriers.py,
``CarrierContract.check_before_consumption``) reads WHO wrote each
radiative carrier and WHEN, off the consuming driver's own contract.  A
resident run has one driver for the whole forecast, so the radiation
seam's restamp (physics.py, ``_run_radiation``) accumulates naturally.
A STREAMED run rebuilds tile buffers -- each with its own driver, each
stamped at the buffer's local t=0 -- and restored only the clock and the
call counters onto them (``tilestream.physics_inventory
.set_carrier_scalars``).  The produced-at ledger never rode along, so a
fresh ``TiledRun`` at domain hour N consumed hour-N fields under stamps
from buffer birth: the twin died at model second 3620 with "GLW ...
last wrote it at model second 0 ... tolerance 260 s", first surface step
of its second wrfout frame, while the resident arm ran the full six
hours and every frame both arms produced was bit-identical.  Radiation
WAS running; only the ledger was silent.

The fix threads the contract through the same round trip the clock
already takes: ``carrier_scalars`` carries ``CarrierContract.state()``
(plus the policy), ``set_carrier_scalars`` restores it through
``merge_carrier_records``, the sweep's clock epilogue publishes it, and
the streamed checkpoint header carries it exactly as restart.py's does.
Every case here is CPU-hermetic: a real ``CarrierContract`` on a stub
driver, no device, no cupy.

RED-ON-REVERT: ``roundtrip_heals_the_hour_boundary_refusal`` is the
defect in one function -- revert the ``carriers`` entries out of
``carrier_scalars``/``set_carrier_scalars`` and it raises the exact
CarrierContractError the twin log shows.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from woof.core.radiation_carriers import (
    CARRIER_SOURCE_DECLARED_CONSTANT, CARRIER_SOURCE_RADIATION_SCHEME,
    CarrierContract, CarrierContractError)
from tilestream import physics_inventory as physinv


def _driver(*, glw_at=None, swdown_at=None, policy="required"):
    """A stub state+driver pair carrying a REAL CarrierContract.

    The stamps mimic a tile buffer: ``glw_at``/``swdown_at`` are the model
    seconds the buffer's own (build-local) radiation last wrote them.
    """
    contract = CarrierContract(policy)
    if glw_at is not None:
        contract.declare("glw", source=CARRIER_SOURCE_RADIATION_SCHEME,
                         model_time=glw_at)
    if swdown_at is not None:
        contract.declare("swdown", source=CARRIER_SOURCE_RADIATION_SCHEME,
                         model_time=swdown_at)
    driver = SimpleNamespace(
        call_counts={"radiation": 1, "noah": 1},
        ysu_nan_guard_fires=0,
        microphysics_updates=3,
        carriers=contract,
        radiation_callable=None,
        carriers_need_producer_refresh=False)
    state = SimpleNamespace(elapsed_seconds=0.0, physics=driver)
    return state, driver


def _consume(contract, model_time, *, fields=None):
    """One Noah consumption under the twin's cadence (radt 240 s, dt 20 s)."""
    contract.check_before_consumption(
        sf_surface_physics=2, fields=fields, model_time=model_time,
        radiation_interval_seconds=240.0, timestep_seconds=20.0)


def scalars_carry_the_contract():
    """carrier_scalars rides the records and the policy with the clock."""
    state, driver = _driver(glw_at=3600.0, swdown_at=3600.0)
    state.elapsed_seconds = 3610.0
    out = physinv.carrier_scalars(state)
    assert out["elapsed_seconds"] == 3610.0
    assert out["surface_radiation_policy"] == "required"
    assert out["carriers"]["glw"] == {
        "source": CARRIER_SOURCE_RADIATION_SCHEME,
        "last_update_model_time": 3600.0}
    assert out["carriers"]["swdown"]["last_update_model_time"] == 3600.0
    # The records are a snapshot, not a live view: stamping the contract
    # after the read must not mutate a published clock.
    driver.carriers.declare("glw", source=CARRIER_SOURCE_RADIATION_SCHEME,
                            model_time=3840.0)
    assert out["carriers"]["glw"]["last_update_model_time"] == 3600.0


def roundtrip_heals_the_hour_boundary_refusal():
    """THE DEFECT: a fresh buffer at domain hour 1 must not read stamp 0.

    Domain side stamped at 3600 (the frame boundary's own radiation call);
    fresh tile buffer stamped at its build-local 0.  Without the ledger in
    the round trip the first consumption at 3620 computes age 3620 against
    tolerance 260 and raises -- the twin_a.log crash verbatim.  With it,
    the age is 20 s and Noah consumes.
    """
    domain_state, _ = _driver(glw_at=3600.0, swdown_at=3600.0)
    domain_state.elapsed_seconds = 3600.0
    clock = physinv.carrier_scalars(domain_state)

    buffer_state, buffer_driver = _driver(glw_at=0.0, swdown_at=0.0)
    with pytest.raises(CarrierContractError, match="stale"):
        _consume(buffer_driver.carriers, 3620.0)   # the pre-fix behaviour

    physinv.set_carrier_scalars(buffer_state, clock)
    assert buffer_state.elapsed_seconds == 3600.0
    record = buffer_driver.carriers.record("glw")
    assert record.last_update_model_time == 3600.0
    _consume(buffer_driver.carriers, 3620.0)       # no refusal


def restore_does_not_rewind_a_captured_restamp():
    """Same source, buffer stamp newer, within the clock: the stamp survives.

    This is the CUDA-graph seam: a captured step's Python restamped the
    buffer at the current model second; the domain dict was published a
    cadence earlier.  Re-imposing the older stamp would erase a producer
    run that happened.
    """
    domain_state, _ = _driver(glw_at=3360.0, swdown_at=3360.0)
    domain_state.elapsed_seconds = 3600.0
    clock = physinv.carrier_scalars(domain_state)

    buffer_state, buffer_driver = _driver(glw_at=3600.0, swdown_at=3360.0)
    physinv.set_carrier_scalars(buffer_state, clock)
    assert (buffer_driver.carriers.record("glw").last_update_model_time
            == 3600.0)
    assert (buffer_driver.carriers.record("swdown").last_update_model_time
            == 3360.0)


def restore_rewinds_a_stamp_from_the_discarded_future():
    """A restart rewind must not resume with stamps ahead of its clock.

    The freshness law refuses negative ages, so a stamp that survived from
    the pre-restore future would refuse the resumed run outright.
    """
    domain_state, _ = _driver(glw_at=3600.0, swdown_at=3600.0)
    domain_state.elapsed_seconds = 3600.0
    clock = physinv.carrier_scalars(domain_state)

    buffer_state, buffer_driver = _driver(glw_at=7200.0, swdown_at=7200.0)
    buffer_state.elapsed_seconds = 7220.0
    physinv.set_carrier_scalars(buffer_state, clock)
    assert (buffer_driver.carriers.record("glw").last_update_model_time
            == 3600.0)
    _consume(buffer_driver.carriers, 3620.0)


def restore_replaces_provenance_wholesale():
    """The domain's SOURCE always wins: a gathered buffer has no history."""
    domain_state, domain_driver = _driver()
    domain_driver.carriers.declare(
        "glw", source=CARRIER_SOURCE_DECLARED_CONSTANT)
    domain_driver.carriers.declare(
        "swdown", source=CARRIER_SOURCE_RADIATION_SCHEME, model_time=3600.0)
    domain_state.elapsed_seconds = 3600.0
    clock = physinv.carrier_scalars(domain_state)

    buffer_state, buffer_driver = _driver(glw_at=3600.0, swdown_at=0.0)
    physinv.set_carrier_scalars(buffer_state, clock)
    record = buffer_driver.carriers.record("glw")
    assert record.source == CARRIER_SOURCE_DECLARED_CONSTANT
    assert record.last_update_model_time is None


def buffer_only_records_survive_a_partial_dict():
    """A carrier the clock does not mention keeps its buffer record.

    Dropping it would relabel a declared carrier ``unwritten`` and turn the
    next consumption into a refusal about dict completeness.
    """
    buffer_state, buffer_driver = _driver(glw_at=100.0)
    buffer_driver.carriers.declare(
        "gsw", source=CARRIER_SOURCE_DECLARED_CONSTANT)
    clock = {"elapsed_seconds": 120.0,
             "call_counts": {"radiation": 1},
             "ysu_nan_guard_fires": 0,
             "microphysics_updates": 0,
             "carriers": {"glw": {
                 "source": CARRIER_SOURCE_RADIATION_SCHEME,
                 "last_update_model_time": 120.0}},
             "surface_radiation_policy": "required"}
    physinv.set_carrier_scalars(buffer_state, clock)
    assert (buffer_driver.carriers.record("glw").last_update_model_time
            == 120.0)
    assert (buffer_driver.carriers.record("gsw").source
            == CARRIER_SOURCE_DECLARED_CONSTANT)


def policy_mismatch_refuses():
    """A clock published under one policy cannot restore under another."""
    buffer_state, _ = _driver(glw_at=0.0)
    clock = {"elapsed_seconds": 20.0,
             "call_counts": {},
             "ysu_nan_guard_fires": 0,
             "microphysics_updates": 0,
             "carriers": {},
             "surface_radiation_policy": "wrf_compat_zero"}
    with pytest.raises(CarrierContractError, match="policy"):
        physinv.set_carrier_scalars(buffer_state, clock)


def legacy_dict_without_records_marks_the_refresh():
    """A pre-contract checkpoint's scalars ask for the producer refresh.

    ``tilestream.checkpoint.restart_scalars`` sets the marker for headers
    that predate the contract; restoring such a dict must arm
    ``carriers_need_producer_refresh`` (restart.py's own legacy remedy)
    rather than leave the buffer's build-time stamps to refuse the resume.
    """
    buffer_state, buffer_driver = _driver(glw_at=0.0)
    clock = {"elapsed_seconds": 3600.0,
             "call_counts": {"radiation": 15},
             "ysu_nan_guard_fires": 0,
             "microphysics_updates": 0,
             "carriers_need_producer_refresh": True}
    physinv.set_carrier_scalars(buffer_state, clock)
    assert buffer_driver.carriers_need_producer_refresh is True
    # And a dict that carries records does NOT arm it.
    buffer_driver.carriers_need_producer_refresh = False
    clock2 = dict(clock)
    del clock2["carriers_need_producer_refresh"]
    clock2["carriers"] = {"glw": {
        "source": CARRIER_SOURCE_RADIATION_SCHEME,
        "last_update_model_time": 3600.0}}
    physinv.set_carrier_scalars(buffer_state, clock2)
    assert buffer_driver.carriers_need_producer_refresh is False


def checkpoint_header_roundtrips_the_records():
    """restart_scalars forwards the contract; a legacy header sets the marker.

    The header dict here is shaped exactly as restart.write_restart and
    tilestream.checkpoint.store_restart_header compose it (driver key with
    carriers + surface_radiation_policy since the carrier contract lane).
    """
    from tilestream import checkpoint as tsck

    header = {
        "elapsed_seconds": 3600.0,
        "driver": {
            "call_counts": {"radiation": 16, "noah": 181},
            "ysu_nan_guard_fires": 0,
            "microphysics_updates": 181,
            "carriers": {
                "glw": {"source": CARRIER_SOURCE_RADIATION_SCHEME,
                        "last_update_model_time": 3600.0},
                "swdown": {"source": CARRIER_SOURCE_RADIATION_SCHEME,
                           "last_update_model_time": 3600.0}},
            "surface_radiation_policy": "required"}}
    scalars = tsck.restart_scalars(header)
    assert scalars["carriers"]["glw"]["last_update_model_time"] == 3600.0
    assert scalars["surface_radiation_policy"] == "required"
    assert "carriers_need_producer_refresh" not in scalars
    # The whole of a resume's restore is one set_carrier_scalars call.
    buffer_state, buffer_driver = _driver(glw_at=0.0, swdown_at=0.0)
    physinv.set_carrier_scalars(buffer_state, scalars)
    _consume(buffer_driver.carriers, 3620.0)

    legacy = {"elapsed_seconds": 3600.0,
              "driver": {"call_counts": {"radiation": 16},
                         "ysu_nan_guard_fires": 0,
                         "microphysics_updates": 181}}
    got = tsck.restart_scalars(legacy)
    assert got.get("carriers_need_producer_refresh") is True
    assert "carriers" not in got


def graph_delta_carries_the_records_as_an_absolute():
    """scalar_delta/apply_scalar_delta: counters stay increments, records
    stay tables.

    A produced-at stamp has no meaningful increment -- the captured step's
    producer wrote it AT a model second -- so the graph stepper re-applies
    the post-step value whole while the counters keep their arithmetic.
    """
    from tilestream.graphcap import apply_scalar_delta, scalar_delta

    records = {"glw": {"source": CARRIER_SOURCE_RADIATION_SCHEME,
                       "last_update_model_time": 240.0}}
    before = {"elapsed_seconds": 220.0,
              "call_counts": {"radiation": 1, "noah": 11},
              "ysu_nan_guard_fires": 0, "microphysics_updates": 11,
              "carriers": {"glw": {
                  "source": CARRIER_SOURCE_RADIATION_SCHEME,
                  "last_update_model_time": 0.0}},
              "surface_radiation_policy": "required"}
    after = {"elapsed_seconds": 240.0,
             "call_counts": {"radiation": 2, "noah": 12},
             "ysu_nan_guard_fires": 0, "microphysics_updates": 12,
             "carriers": records,
             "surface_radiation_policy": "required"}
    delta = scalar_delta(before, after)
    assert delta["call_counts"] == {"radiation": 1, "noah": 1}
    assert delta["carriers"] == records
    replayed = apply_scalar_delta(before, delta)
    assert replayed["elapsed_seconds"] == 240.0
    assert replayed["call_counts"] == after["call_counts"]
    assert replayed["carriers"] == records
    assert replayed["surface_radiation_policy"] == "required"


def sweep_epilogue_publishes_the_records():
    """_advance_clock's published clock carries the buffers' fresh stamps.

    This is the second half of the round trip: without it, the fix would
    heal frame 2 and re-arm the bomb for frame 3.
    """
    from tilestream.driver import _advance_clock

    clock = {"elapsed_seconds": 3600.0,
             "call_counts": {"radiation": 15, "noah": 180},
             "ysu_nan_guard_fires": 0,
             "microphysics_updates": 180,
             "carriers": {"glw": {
                 "source": CARRIER_SOURCE_RADIATION_SCHEME,
                 "last_update_model_time": 3360.0}},
             "surface_radiation_policy": "required"}
    tiles = []
    for _ in range(2):
        state, driver = _driver(glw_at=3600.0)
        state.elapsed_seconds = 3620.0
        driver.call_counts = {"radiation": 16, "noah": 181}
        driver.microphysics_updates = 181
        tiles.append(state)
    out = _advance_clock(clock, tiles, 1, physinv)
    assert out["elapsed_seconds"] == 3620.0
    assert out["carriers"]["glw"]["last_update_model_time"] == 3600.0
    assert out["surface_radiation_policy"] == "required"


TESTS = (
    scalars_carry_the_contract,
    roundtrip_heals_the_hour_boundary_refusal,
    restore_does_not_rewind_a_captured_restamp,
    restore_rewinds_a_stamp_from_the_discarded_future,
    restore_replaces_provenance_wholesale,
    buffer_only_records_survive_a_partial_dict,
    policy_mismatch_refuses,
    legacy_dict_without_records_marks_the_refresh,
    checkpoint_header_roundtrips_the_records,
    graph_delta_carries_the_records_as_an_absolute,
    sweep_epilogue_publishes_the_records,
)


@pytest.mark.parametrize("case", TESTS, ids=lambda f: f.__name__)
def test_streamed_carrier_freshness(case):
    case()


@pytest.mark.parametrize("explicit_target", [False, True])
def test_device_store_refresh_restores_clock_and_provenance(explicit_target):
    from woof.core.streaming import StreamedDomain

    state, driver = _driver(glw_at=0., swdown_at=0.)
    current, current_driver = _driver(glw_at=3600., swdown_at=3600.)
    current.elapsed_seconds = 3620.
    current_driver.call_counts.update(radiation=16, noah=181)
    current_driver.ysu_nan_guard_fires = 2
    current_driver.microphysics_updates = 181
    canonical = physinv.carrier_scalars(current)
    drained = []
    stream = object.__new__(StreamedDomain)
    stream.host_store = False
    stream._state = state if not explicit_target else None
    stream.scalars = {}

    def drain():
        drained.append(True)
        stream.scalars.update(canonical)

    stream._run = SimpleNamespace(drain=drain)
    # The arrays already alias the store. A refresh must still wait for
    # the sweep and restore all non-array state before external readers.
    with pytest.raises(CarrierContractError, match="stale"):
        _consume(driver.carriers, 3620.)
    assert stream.refresh_state(state if explicit_target else None) == 0
    assert drained == [True]
    assert physinv.carrier_scalars(state) == canonical
    _consume(driver.carriers, 3620.)


@pytest.mark.parametrize("has_state,has_scalars", [(False, True), (True, False)])
def test_device_store_refresh_without_scalar_destination_still_drains(has_state, has_scalars):
    from woof.core.streaming import StreamedDomain

    state, _ = _driver(glw_at=0., swdown_at=0.)
    stream = object.__new__(StreamedDomain)
    stream.host_store = False
    stream._state = state if has_state else None
    stream.scalars = {"elapsed_seconds": 12.} if has_scalars else None
    drained = []
    stream._run = SimpleNamespace(drain=lambda: drained.append(True))
    assert stream.refresh_state() == 0
    assert drained == [True]
    assert state.elapsed_seconds == 0.


def test_device_store_refresh_preserves_carrier_policy_validation():
    from woof.core.streaming import StreamedDomain

    state, _ = _driver(glw_at=0., policy="required")
    current, _ = _driver(glw_at=3600., policy="wrf_compat_zero")
    stream = object.__new__(StreamedDomain)
    stream.host_store = False
    stream._state = state
    stream.scalars = physinv.carrier_scalars(current)
    stream._run = SimpleNamespace(drain=lambda: None)
    with pytest.raises(CarrierContractError, match="policy"):
        stream.refresh_state()
