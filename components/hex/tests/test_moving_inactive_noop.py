"""With no WOOF_HEX_MOVING_* switch set, the moving-region hooks are not taken.

Breakage prevented: the moving variable-resolution region lands inside the
full-step LTS driver, the forecast door and the LTS class admission, which the
default WOOF-ArWen route runs.  These checks pin that an unset environment leaves
that route as it was: no moving module is imported, the host and the history
schedule reach the executor as the same objects (no carrier, no clock offset,
even when the prepared state's clock is not zero), and the class map is the
pre-landing one.
"""
from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from woof.hex.drivers import fullstep_lts_forecast_v841 as lts
from woof.hex.fullstep_lts_admission import admit_mesh_rates, moving_region_requested
from woof.hex.moving_control import MovingControl

MOVING_MODULES = ("woof.hex.moving_checkpoint", "woof.hex.moving_control", "woof.hex.moving_zone_state",
                  "woof.hex.moving_physics", "woof.hex.moving_state", "woof.hex.moving_zone")


def _clear_moving(monkeypatch):
    for key in list(os.environ):
        if key.startswith("WOOF_HEX_MOVING_"):
            monkeypatch.delenv(key)
    monkeypatch.delenv("WOOF_HEX_LTS_CONVECTIVE_SPACING_M", raising=False)
    monkeypatch.delenv("WOOF_HEX_ASYNC_HISTORY", raising=False)


def _one_class_host(time_seconds):
    """A uniform 3 km line of cells: the geometry gives one class, so the driver
    hands the run to the native executor right after admission."""
    n, levels = 8, 4
    coe = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)
    dc = np.full(n - 1, 3000.0)
    mesh = SimpleNamespace(arrays={"dcEdge": dc, "cellsOnEdge": coe, "dvEdge": dc * 0.6},
                           area_cell=np.full(n, 9.0e6))
    state = SimpleNamespace(rho=np.ones((levels, n)), rho_u=np.zeros((levels, n - 1)),
                            rho_w=np.zeros((levels + 1, n)), time_seconds=time_seconds)
    prepared = SimpleNamespace(mesh=mesh, state=state, saved_diagnostics=None,
                               vertical=SimpleNamespace(dzw=np.full(levels, 500.0)))
    config = SimpleNamespace(config_dt=18.0, config_scalar_advection=False, config_number_of_sub_steps=6)
    return {"config": config, "prepared": prepared}


def _schedule():
    return {"steps": 200, "capture_steps": [0, 200],
            "labels": {0: "2025-11-03_00.00.00", 200: "2025-11-03_01.00.00"},
            "valid_times": {0: "2025-11-03_00:00:00", 200: "2025-11-03_01:00:00"}}


def _execute(tmp_path, host, schedule):
    seen = {}

    def native_execute(**kwargs):
        seen.update(kwargs)
        return {"native": True}

    result = lts.execute_fullstep_lts(
        SimpleNamespace(proof=None), native_execute=native_execute, policy={}, host=host, schedule=schedule,
        cache_root=tmp_path, output_root=tmp_path, arwen_checkout=None, source_receipt=None,
        authority_receipt=None, fingerprint_every=0)
    return result, seen


def test_switch_predicate_reads_only_non_empty_moving_switches():
    assert not moving_region_requested({})
    assert not moving_region_requested({"WOOF_HEX_MOVING_SHOCK": "", "WOOF_HEX_ASYNC_HISTORY": "1",
                                        "WOOF_HEX_LTS_CONVECTIVE_SPACING_M": "8000"})
    assert moving_region_requested({"WOOF_HEX_MOVING_CONTROL": "/x"})
    assert MovingControl.from_environ({}) is None


@pytest.mark.parametrize("time_seconds", [0.0, 3600.0])
def test_unset_switches_hand_the_same_host_and_schedule_to_the_executor(tmp_path, monkeypatch, time_seconds):
    _clear_moving(monkeypatch)
    for name in MOVING_MODULES:                       # any moving import would raise ImportError
        monkeypatch.setitem(sys.modules, name, None)
    host, schedule = _one_class_host(time_seconds), _schedule()
    labels = dict(schedule["labels"])
    result, seen = _execute(tmp_path, host, schedule)
    assert seen["host"] is host
    assert seen["schedule"] is schedule and schedule["labels"] == labels
    assert result == {"native": True, "fullstep_lts": {
        "dispatch": "native single-class step", "rate_assignment": result["fullstep_lts"]["rate_assignment"]}}
    info = result["fullstep_lts"]["rate_assignment"]
    # Since H7 (hex 0.4.0) the convective hold is every LTS run's default (8 km), so the receipt names it with
    # or without the moving switches; what the switches must not change is its value.
    assert info.get("convective_spacing_m") == 8000.0
    assert list(tmp_path.iterdir()) == []             # no carrier, receipt or remap file written


def test_class_map_without_switches_is_the_default_lts_map(monkeypatch):
    _clear_moving(monkeypatch)
    spacings = [3000.0] * 12 + [6000.0] * 24 + [12000.0] * 24
    n = len(spacings)
    coe = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)
    dc = np.array([max(spacings[i], spacings[i + 1]) for i in range(n - 1)])
    mesh = SimpleNamespace(arrays={"dcEdge": dc, "cellsOnEdge": coe, "dvEdge": dc * 0.6},
                           area_cell=np.array(spacings) ** 2)
    state = SimpleNamespace(rho=np.ones((4, n)), rho_u=np.zeros((4, n - 1)), rho_w=np.zeros((5, n)))
    saved = SimpleNamespace(normal_velocity=np.full((4, n - 1), 1.0), theta_m=np.full((4, n), 300.0),
                            exner=np.full((4, n), 1.0))
    vertical = SimpleNamespace(dzw=np.full(4, 500.0))
    args = (mesh, state, saved, vertical)
    kwargs = dict(dt=18.75, max_rate=8, acoustic_substeps=6, target_transport=None)
    # H7 (hex 0.4.0): the 8 km convective hold is every LTS run's default (the 3->6 km transition blow-ups);
    # the moving switches must leave the default class map exactly as it is, and turning the hold off
    # (instrument only) must still give the purely geometric map.
    default_rates, default_info = admit_mesh_rates(*args, **kwargs)
    held_rates, held_info = admit_mesh_rates(*args, convective_spacing_m=8000.0, **kwargs)
    np.testing.assert_array_equal(default_rates, held_rates)
    assert default_info == held_info and default_info["convective_cells_held_at_base_rate"] > 0
    off_rates, _ = admit_mesh_rates(*args, convective_spacing_m=0.0, **kwargs)
    assert not np.array_equal(off_rates, default_rates)
    monkeypatch.setenv("WOOF_HEX_MOVING_SHOCK", "1")
    moving_rates, moving_info = admit_mesh_rates(*args, **kwargs)
    np.testing.assert_array_equal(moving_rates, default_rates)
    assert moving_info == default_info


def test_the_carrier_switch_is_what_engages_the_resume_hook(tmp_path, monkeypatch):
    _clear_moving(monkeypatch)
    calls = []
    fake = ModuleType("woof.hex.moving_checkpoint")
    fake.load_boundary = lambda path: calls.append(("load", path)) or {"rates": np.ones(8, np.int32)}
    fake.resume_same_mesh_host = lambda host, payload: calls.append(("resume",)) or host
    fake.carry_physics_top_pressure = lambda host, payload: calls.append(("top",)) or host
    monkeypatch.setitem(sys.modules, "woof.hex.moving_checkpoint", fake)
    _execute(tmp_path, _one_class_host(0.0), _schedule())
    assert calls == []
    monkeypatch.setenv("WOOF_HEX_MOVING_CHECKPOINT_IN", str(tmp_path / "boundary.pkl"))
    _execute(tmp_path, _one_class_host(0.0), _schedule())
    assert calls == [("load", str(tmp_path / "boundary.pkl")), ("resume",), ("top",)]


@pytest.mark.parametrize("intent", ["WOOF_HEX_MOVING_CONTROL", "WOOF_HEX_MOVING_CHECKPOINT_OUT"])
def test_a_moving_segment_on_a_one_class_mesh_is_refused_by_name(tmp_path, monkeypatch, intent):
    """Completeness row #42: one card, a mesh that classes entirely at rate one, and the controller's intent to
    move.  The native single-class step reads none of the move switches, so the segment would run to its end and
    never move; it is refused by name before the executor runs.  The same-mesh resume above (the carrier alone,
    no controller) still runs on that step."""
    from woof.hex.fullstep_lts_admission import MOVING_ROUTE_REFUSAL

    _clear_moving(monkeypatch)
    monkeypatch.setenv(intent, str(tmp_path / "control"))
    with pytest.raises(ValueError) as refused:
        _execute(tmp_path, _one_class_host(0.0), _schedule())
    text = str(refused.value)
    assert text.startswith(MOVING_ROUTE_REFUSAL) and intent in text and "never move" in text
    assert "native single-class step" in text and "WOOF_HEX_MOVING_CHECKPOINT_IN alone" in text
