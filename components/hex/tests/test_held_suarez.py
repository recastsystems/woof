"""The named suite and mesh door must select the same column physics."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_named_suite_has_thermal_and_rk_momentum():
    woof = pytest.importorskip('woof')
    pytest.importorskip('cupy')  # woof.core.mpas_column_suite imports the CuPy microphysics module
    from woof.hex.physics_suites import resolve_physics
    from woof.core.mpas_column_suite import column_physics_plan
    selected = resolve_physics('held_suarez', {}, mesh='x1.40962',
        engine_checkout=Path(woof.__file__).resolve().parent.parent)
    assert selected.convection == 'off'
    plan = column_physics_plan(selected.engine_physics)
    assert plan.config.ra_lw_physics == 31
    assert plan.config.ra_sw_physics == 0
    assert plan.config.cu_physics == 0
    assert plan.identity['rk_momentum'] == 'held-suarez-edge-v1'


@pytest.mark.parametrize('argument,expected', [([], 'auto'),
    (['--convection', 'off'], 'off'), (['--convection=off'], 'off')])
def test_mesh_binding_receives_the_forecasts_convection(tmp_path, monkeypatch,
                                                        argument, expected):
    path = Path(__file__).parents[1] / 'tools/run_cuda_v841_forecast_mesh.py'
    spec = importlib.util.spec_from_file_location('held_suarez_mesh_door_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}
    def bind(*args, **kwargs):
        captured.update(kwargs)
        return {}
    binding = SimpleNamespace(MESH_BINDINGS={}, bind_mesh=bind)
    forecast = SimpleNamespace(proof=object())
    monkeypatch.setattr(module, '_load',
        lambda name, path: binding if name == 'mpas_mesh_binding' else forecast)
    assert module.main(['--repo', str(tmp_path), '--mesh', 'x1.40962',
        '--dt', '300', '--verify-only', '--', '--grid', 'grid.nc',
        '--static', 'static.nc', *argument]) == 0
    assert captured['convection'] == expected


@pytest.mark.parametrize("stray", [None, "sfclay", "ysu", "noah"])
def test_forcing_only_cadence_checks_disabled_stack_counters(stray):
    from woof.hex import pbl_cadence
    from woof.hex.drivers.run_cuda_v841_forecast import _physics_cadence_proof
    counts = dict(sfclay=0, ysu=0, noah=0, radiation=3)
    if stray is not None:
        counts[stray] = 1
    receipts = [{"backend": {"cadence": {
        "surface_pbl_enabled": False, "surface_pbl_ran": False,
        "radiation_ran": True, "call_counts": counts}}} for _ in range(3)]
    decision = pbl_cadence.pbl_cadence_decision(dt_seconds=300., requested="auto")
    result = _physics_cadence_proof(decision, receipts, 3)
    assert result["surface_pbl_calls_expected"] == 0
    assert result["surface_pbl_steps_held"] == 0
    assert result["surface_pbl_steps_disabled"] == 3
    assert result["consistent"] == (stray is None)
