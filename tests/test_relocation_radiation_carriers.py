"""Surface-radiation carriers across a discrete nest relocation.

THE DEFECT THIS PINS (test host 2 GPU campaign, 2026-08-24).  A relocation
rebuilds the moved child's physics driver from cold, which allocates a
fresh carrier buffer for every radiative carrier and seeds a fresh
:class:`~woof.core.radiation_carriers.CarrierContract` in which
``glw``/``swdown`` are ``unwritten``.  The preparer carried the 19
land-surface continuation fields and the registry-derived per-column
driver state, but neither the carrier BUFFERS nor the carrier LEDGER
were in either carry set.  So the first surface call after a move met a
contract that said nothing had ever written GLW, and refused::

    CarrierContractError: GLW (downward longwave at the surface, W m-2)
    has no producer, and Noah (sf_surface_physics=2) is about to consume
    it at model second 360.

Measured three-arm isolation, wizard defaults, one forecast hour:
relocation off with ``radt = 12`` passed; relocation on with
``radt = 12`` refused at the first move (t = 360 s); relocation on with
``radt = 6``, aligned so a radiation call fell due on the very step the
move landed on, passed with five moves.  So the trigger is the move and
the escape was a cadence coincidence -- under the wizard's shipped
defaults ANY relocation cadence shorter than the radiation interval
refused at the first move.  The producer guard is correct; the
transplant was short.

The contract under test, in three parts:

* the carried inventory is DERIVED from the consumer matrix
  (:data:`~woof.core.radiation_carriers.CONSUMER_CARRIERS`) that the
  refusal itself reads, so a carrier added to the matrix tomorrow moves
  across relocations the day it is added;
* the buffers move the way every other consumed-between-cadence surface
  field moves -- index-space transplant through the plan window, strip
  from the nearest same-landmask-class donor;
* the LEDGER moves verbatim, source and last-write model time both.
  Re-stamping the move time would blind the staleness instrument, which
  is the second half of the same guard: the transplanted flux really is
  as old as the outgoing child's, and the tolerance (radiation cadence
  plus one step) is what decides whether that is still admissible.

Instrument rule: every carry assertion is paired with a control that
fails without the carry.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from woof.core.nest_relocation import Placement, plan_relocation
from woof.core.radiation_carriers import (CARRIER_SOURCE_RADIATION_SCHEME,
                                           CARRIER_SOURCE_UNWRITTEN,
                                           CarrierContract,
                                           CarrierContractError)
from woof.ingest.relocation_init import donor_fill_plan, overlap_mask_for_plan

F32 = np.float32
RATIO = 3
NX = NY = 12

#: The campaign's own numbers: the wizard default radiation interval, the
#: relocation cadence that refused against it, and the root step.
RADT_SECONDS = 720.0
MOVE_SECOND = 360.0
DT_SECONDS = 22.5


def _plan(di=2, dj=1):
    return plan_relocation(
        placement_from=Placement(grid_id=2, i_parent_start=10,
                                 j_parent_start=10),
        placement_to=Placement(grid_id=2, i_parent_start=10 + di,
                               j_parent_start=10 + dj, generation=1),
        parent_grid_ratio=RATIO, child_nx=NX, child_ny=NY)


def _distinct(shape, seed, *, offset=0.0):
    rng = np.random.default_rng(seed)
    return (offset + rng.standard_normal(shape)).astype(F32)


def _fill_for(plan, shape=(NY, NX)):
    """The preparer's own donor fill: overlap mask + landmask classes."""
    landmask = np.zeros(shape)
    landmask[:, : shape[1] // 2] = 1.0
    return donor_fill_plan(overlap_mask=overlap_mask_for_plan(plan, shape),
                           landmask=landmask)


def _outgoing_driver(*, last_radiation_second=0.0):
    """A moved child mid-run: carriers written by a radiation call."""
    contract = CarrierContract()
    fields = {
        "glw": _distinct((NY, NX), 21, offset=330.0),
        "swdown": _distinct((NY, NX), 22, offset=610.0),
        "xland": np.ones((NY, NX), dtype=F32),
        "tsk": _distinct((NY, NX), 23, offset=290.0),
    }
    for name in ("glw", "swdown"):
        contract.declare(name, source=CARRIER_SOURCE_RADIATION_SCHEME,
                         model_time=last_radiation_second)
    return SimpleNamespace(fields=fields, carriers=contract)


def _rebuilt_driver():
    """What ``initialize_physics`` hands back after a cold rebuild."""
    contract = CarrierContract()
    contract.declare("glw", source=CARRIER_SOURCE_UNWRITTEN)
    contract.declare("swdown", source=CARRIER_SOURCE_UNWRITTEN)
    return SimpleNamespace(
        fields={"glw": np.full((NY, NX), 300.0, dtype=F32),
                "swdown": np.zeros((NY, NX), dtype=F32),
                "xland": np.ones((NY, NX), dtype=F32),
                "tsk": np.full((NY, NX), 300.0, dtype=F32)},
        carriers=contract)


# ---------------------------------------------------------------------------
# The inventory is the consumer matrix, not a hand list
# ---------------------------------------------------------------------------

def test_carried_carriers_are_the_consumer_matrix():
    from woof.core.physics_continuation import relocatable_carriers
    from woof.core.radiation_carriers import CONSUMER_CARRIERS

    named = set()
    for carriers in CONSUMER_CARRIERS.values():
        named.update(carriers)
    assert set(relocatable_carriers()) == named
    # The four the ledger actually names, so a silent shrink is visible.
    assert set(relocatable_carriers()) == {"glw", "swdown", "gsw", "coszen"}


# ---------------------------------------------------------------------------
# THE CLASS REPRODUCER: the guard must not fire on a relocated child
# ---------------------------------------------------------------------------

def test_a_relocated_child_still_has_a_producer_for_its_next_surface_read():
    """The measured refusal, CPU-side: move at 360 s, radt 720 s, Noah."""
    plan = _plan()
    outgoing = _outgoing_driver(last_radiation_second=0.0)
    rebuilt = _rebuilt_driver()

    # The control: without the carry the rebuilt child refuses exactly as
    # the campaign's arm A did.
    with pytest.raises(CarrierContractError) as refused:
        rebuilt.carriers.check_before_consumption(
            sf_surface_physics=2, fields=rebuilt.fields,
            model_time=MOVE_SECOND,
            radiation_interval_seconds=RADT_SECONDS,
            timestep_seconds=DT_SECONDS)
    assert "has no producer" in str(refused.value)

    from woof.core.physics_continuation import (capture_carriers,
                                                 restore_carriers,
                                                 shift_carriers)

    captured = capture_carriers(outgoing)
    shifted = shift_carriers(captured["fields"], plan, _fill_for(plan))
    restore_carriers(rebuilt, shifted, captured["contract"])

    # And with it, the same call at the same second passes.
    rebuilt.carriers.check_before_consumption(
        sf_surface_physics=2, fields=rebuilt.fields,
        model_time=MOVE_SECOND, radiation_interval_seconds=RADT_SECONDS,
        timestep_seconds=DT_SECONDS)


def test_the_carried_ledger_keeps_the_staleness_instrument_armed():
    """A producer that stopped is still caught after a move."""
    from woof.core.physics_continuation import (capture_carriers,
                                                 restore_carriers,
                                                 shift_carriers)

    plan = _plan()
    # Radiation last ran more than a cadence plus a step before the move.
    outgoing = _outgoing_driver(last_radiation_second=-2000.0)
    rebuilt = _rebuilt_driver()
    captured = capture_carriers(outgoing)
    restore_carriers(
        rebuilt, shift_carriers(captured["fields"], plan, _fill_for(plan)),
        captured["contract"])
    with pytest.raises(CarrierContractError) as stale:
        rebuilt.carriers.check_before_consumption(
            sf_surface_physics=2, fields=rebuilt.fields,
            model_time=MOVE_SECOND,
            radiation_interval_seconds=RADT_SECONDS,
            timestep_seconds=DT_SECONDS)
    assert "is stale" in str(stale.value)


# ---------------------------------------------------------------------------
# Capture -> shift -> restore, both directions
# ---------------------------------------------------------------------------

def test_the_overlap_moves_bitwise_and_the_strip_is_donor_filled():
    from woof.core.physics_continuation import (capture_carriers,
                                                 restore_carriers,
                                                 shift_carriers)

    plan = _plan()
    outgoing = _outgoing_driver()
    rebuilt = _rebuilt_driver()
    captured = capture_carriers(outgoing)
    receipt = restore_carriers(
        rebuilt, shift_carriers(captured["fields"], plan, _fill_for(plan)),
        captured["contract"])

    (dst_j, src_j), (dst_i, src_i) = plan.window((NY, NX))
    for name in ("glw", "swdown"):
        old = outgoing.fields[name]
        new = rebuilt.fields[name]
        assert np.array_equal(new[dst_j, dst_i], old[src_j, src_i]), name
        # Control: an unshifted copy does not satisfy the same check.
        assert not np.array_equal(new, old), name
        # The strip carries a real donor value, never the allocation fill
        # and never a zero flux.
        strip = np.ones((NY, NX), dtype=bool)
        strip[dst_j, dst_i] = False
        assert strip.any()
        assert np.all(np.isin(new[strip], old)), name
    assert sorted(receipt["carriers_moved"]) == ["glw", "swdown"]
    assert receipt["ledger_restored"] is True

    # Non-carrier driver fields are none of this mechanism's business.
    assert np.array_equal(rebuilt.fields["tsk"],
                          np.full((NY, NX), 300.0, dtype=F32))


def test_a_null_move_is_the_identity():
    from woof.core.physics_continuation import (capture_carriers,
                                                 restore_carriers,
                                                 shift_carriers)

    plan = _plan(di=0, dj=0)
    outgoing = _outgoing_driver()
    rebuilt = _rebuilt_driver()
    captured = capture_carriers(outgoing)
    restore_carriers(
        rebuilt, shift_carriers(captured["fields"], plan, _fill_for(plan)),
        captured["contract"])
    for name in ("glw", "swdown"):
        assert np.array_equal(rebuilt.fields[name], outgoing.fields[name])
    assert rebuilt.carriers.state() == outgoing.carriers.state()


def test_a_carrier_the_configuration_never_allocated_is_not_invented():
    """No GSW/COSZEN on a Noah run means none on the rebuilt child."""
    from woof.core.physics_continuation import (capture_carriers,
                                                 restore_carriers,
                                                 shift_carriers)

    plan = _plan()
    outgoing = _outgoing_driver()
    rebuilt = _rebuilt_driver()
    captured = capture_carriers(outgoing)
    assert set(captured["fields"]) == {"glw", "swdown"}
    receipt = restore_carriers(
        rebuilt, shift_carriers(captured["fields"], plan, _fill_for(plan)),
        captured["contract"])
    assert "gsw" not in rebuilt.fields
    assert "coszen" not in rebuilt.fields
    assert receipt["carriers_absent"] == []


def test_a_driver_without_a_contract_carries_nothing_and_says_so():
    from woof.core.physics_continuation import capture_carriers

    captured = capture_carriers(SimpleNamespace(fields={}, carriers=None))
    assert captured["fields"] == {}
    assert captured["contract"] is None


# ---------------------------------------------------------------------------
# The preparer wires the mechanism (the front-door routes get it)
# ---------------------------------------------------------------------------

class _ScratchState:
    """A DomainState stand-in exposing the scratch-slot surface."""

    def __init__(self, arrays=None):
        self._scratch = dict(arrays or {})

    def existing_scratch(self, slot):
        return self._scratch.get(slot)

    def scratch(self, shape, slot, dtype=None):
        shape = tuple(shape)
        buf = self._scratch.get(slot)
        if buf is None:
            buf = np.zeros(shape, dtype=np.dtype(dtype or np.float32))
            self._scratch[slot] = buf
        assert buf.shape == shape
        return buf


def _preparer_fixture(monkeypatch):
    from woof.runtime import RealRelocationChildPreparer

    ny = nx = 6
    child_dc = SimpleNamespace(
        grid_id=2, i_parent_start=4, j_parent_start=4,
        parent_grid_ratio=1, run=SimpleNamespace(nx=nx, ny=ny))
    ground = np.arange(20 * 20, dtype=np.float64).reshape(20, 20)

    def statics_for(dc):
        i0, j0 = int(dc.i_parent_start), int(dc.j_parent_start)
        return {"HGT_M": ground[j0:j0 + ny, i0:i0 + nx].copy(),
                "LANDMASK": np.ones((ny, nx))}

    model = SimpleNamespace(
        _prepared_by_grid_id={2: SimpleNamespace(
            static_fields=statics_for(child_dc))},
        _activation_context={})
    preparer = RealRelocationChildPreparer(
        exp=SimpleNamespace(), data=SimpleNamespace(), model=model)
    monkeypatch.setattr(preparer, "_rebuild_driver", lambda *args: 0.125)

    out_contract = CarrierContract()
    out_contract.declare("glw", source=CARRIER_SOURCE_RADIATION_SCHEME,
                         model_time=0.0)
    out_contract.declare("swdown", source=CARRIER_SOURCE_RADIATION_SCHEME,
                         model_time=0.0)
    out_state = _ScratchState()
    out_state.physics = SimpleNamespace(
        fields={"tsk": np.arange(ny * nx, dtype=F32).reshape(ny, nx),
                "glw": _distinct((ny, nx), 31, offset=330.0),
                "swdown": _distinct((ny, nx), 32, offset=610.0),
                "xland": np.ones((ny, nx), dtype=F32)},
        cumulus_callable=None, radiation_callable=None, carriers=out_contract, call_counts={"radiation": 3},
        ysu_nan_guard_fires=0, microphysics_updates=9)
    out_state.elapsed_seconds = 60.
    node = SimpleNamespace(cfg=child_dc, state=out_state)

    new_dc = SimpleNamespace(**{**vars(child_dc), "i_parent_start": 6,
                                "j_parent_start": 4})
    new_state = _ScratchState()
    new_contract = CarrierContract()
    new_contract.declare("glw", source=CARRIER_SOURCE_UNWRITTEN)
    new_contract.declare("swdown", source=CARRIER_SOURCE_UNWRITTEN)
    new_state.physics = SimpleNamespace(
        fields={"glw": np.full((ny, nx), 300.0, dtype=F32),
                "swdown": np.zeros((ny, nx), dtype=F32),
                "xland": np.ones((ny, nx), dtype=F32)},
        cumulus_callable=None, radiation_callable=None, carriers=new_contract, call_counts={},
        ysu_nan_guard_fires=0, microphysics_updates=0)
    initialized = SimpleNamespace(static_fields=statics_for(new_dc),
                                  grid="new-grid", state=new_state)
    return preparer, node, new_dc, initialized, out_state, new_state


def test_the_preparer_carries_the_carriers_and_reports_them(monkeypatch):
    preparer, node, new_dc, initialized, out_state, new_state = (
        _preparer_fixture(monkeypatch))
    preparer.capture_outgoing(node)
    preparer(initialized, new_dc, SimpleNamespace())

    receipt = preparer.last_receipt["radiation_carriers"]
    assert sorted(receipt["carriers_moved"]) == ["glw", "swdown"]
    assert receipt["ledger_restored"] is True

    # di = +2 parent cells at ratio 1: the incoming child's overlap is the
    # outgoing child's shifted in index space.
    for name in ("glw", "swdown"):
        old = out_state.physics.fields[name]
        new = new_state.physics.fields[name]
        assert np.array_equal(new[:, :4], old[:, 2:]), name

    # And the ledger the refusal reads now names a producer at the move.
    new_state.physics.carriers.check_before_consumption(
        sf_surface_physics=2, fields=new_state.physics.fields,
        model_time=MOVE_SECOND, radiation_interval_seconds=RADT_SECONDS,
        timestep_seconds=DT_SECONDS)


def test_the_preparer_is_accurate_when_the_rebuilt_child_has_no_driver(
        monkeypatch):
    preparer, node, new_dc, initialized, _out, new_state = (
        _preparer_fixture(monkeypatch))
    del new_state.physics
    preparer.capture_outgoing(node)
    preparer(initialized, new_dc, SimpleNamespace())
    receipt = preparer.last_receipt["radiation_carriers"]
    assert receipt["restored"] is False
