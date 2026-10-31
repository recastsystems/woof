"""The adapted native door preserves its namelist and emits evidence on failure."""
from argparse import Namespace
import json

import pytest

from woof.hex import analysis_door, forecast_door, kernel_prewarm, licence_gate
from woof.hex.mpasphys import forecast_v842


@pytest.mark.parametrize('fail', [False, True])
def test_native_prewarm_precedes_run_and_receipt_survives_failure(tmp_path, monkeypatch, fail):
    events = []
    config = object()
    text = '&nhyd_model\n config_dt = 0.9375\n/\n'
    arguments = Namespace(out=tmp_path, dt=0.9375)
    monkeypatch.setattr(analysis_door, 'resolve_analysis', lambda *a, **k: None)
    monkeypatch.setattr(forecast_v842, 'native_route', lambda *a, **k: (config, text))
    monkeypatch.setattr(forecast_door, 'mpas_suite_lts_refusal', lambda a: None)
    # The shared grid/static/init file checks (forecast_door.require_forecast_inputs, tested in
    # test_preflight_missing_inputs) run before the native step; this door test fakes the forecast
    # and passes no files, so they are not its subject.
    monkeypatch.setattr(forecast_door, 'require_forecast_inputs', lambda *a, **k: (None, None, None))
    # These door tests fake the forecast itself; the ported-route suite refusal
    # (resident_v842.ported_route_refusal, tested in test_mpasphys_resident_v842) is not their subject.
    monkeypatch.setattr("woof.hex.mpasphys.resident_v842.ported_route_refusal", lambda config: None)
    monkeypatch.setattr(licence_gate, 'admit', lambda c, a: c)
    monkeypatch.setattr(forecast_door, 'announce_physics_mode', lambda *a: None)
    monkeypatch.setattr(forecast_door, 'announce_cpu_slots', lambda *a: None)
    monkeypatch.setattr(kernel_prewarm, 'install_recorder', lambda: events.append('record'))
    monkeypatch.setattr(kernel_prewarm, 'start_replay_async', lambda: events.append('replay'))
    monkeypatch.setattr(kernel_prewarm, 'receipt', lambda: {'test_events': list(events)})

    def run(a, c, native_text):
        assert a is arguments and c is config and native_text == text
        assert events == ['record', 'replay']
        events.append('run')
        if fail:
            raise RuntimeError('controlled native failure')
        return 7

    monkeypatch.setattr(forecast_v842, 'run_forecast', run)
    if fail:
        with pytest.raises(Exception, match='controlled native failure'):
            forecast_door._run_forecast(arguments)
    else:
        assert forecast_door._run_forecast(arguments) == 7
    receipt = json.loads((tmp_path / 'kernel-plan-receipt.json').read_text())
    assert receipt == {'test_events': ['record', 'replay', 'run']}
