"""Forecast runtime failures stay named; host preparation stays reachable."""
from types import SimpleNamespace
import importlib

import numpy as np
import pytest

from woof import forecast_gpu


@pytest.fixture(autouse=True)
def clean_runtime(monkeypatch):
    monkeypatch.setattr(forecast_gpu, '_working_runtime', None)
    monkeypatch.delenv('GPUWM_NO_LOCAL_GPU', raising=False)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)


@pytest.mark.parametrize('kind', ['missing', 'unloadable', 'driver', 'zero'])
def test_runtime_gap_is_named_and_keeps_its_evidence(monkeypatch, kind):
    def import_module(name):
        if kind == 'missing':
            raise ModuleNotFoundError("No module named 'cupy'")
        if kind == 'unloadable':
            raise OSError('missing runtime library')
        def count():
            if kind == 'driver':
                raise RuntimeError('insufficient driver')
            return 0
        return SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=count)))

    monkeypatch.setattr(forecast_gpu.importlib, 'import_module', import_module)
    from woof import capabilities
    monkeypatch.setattr(capabilities, 'installed_cupy_version', lambda: (14, 0))
    with pytest.raises(forecast_gpu.ForecastGPURequired) as caught:
        forecast_gpu.require_forecast_gpu()
    expected = dict(missing='ModuleNotFoundError', unloadable='missing runtime library',
                    driver='insufficient driver', zero='0 NVIDIA devices')[kind]
    assert expected in caught.value.detected
    assert forecast_gpu.FORECAST_GPU_REQUIREMENT in str(caught.value)


@pytest.mark.parametrize('variable,value', [('CUDA_VISIBLE_DEVICES', ''),
                                          ('CUDA_VISIBLE_DEVICES', '-1'),
                                          ('GPUWM_NO_LOCAL_GPU', '1')])
def test_opt_out_never_imports_runtime(monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    monkeypatch.setattr(forecast_gpu.importlib, 'import_module',
                        lambda _: pytest.fail('opened disabled CUDA runtime'))
    with pytest.raises(forecast_gpu.ForecastGPURequired):
        forecast_gpu.require_forecast_gpu()


def test_success_is_silent_and_does_not_reprobe_per_step(monkeypatch):
    calls = []
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(
        getDeviceCount=lambda: calls.append('probe') or 1)))
    monkeypatch.setattr(forecast_gpu.importlib, 'import_module', lambda _: cp)
    from woof import capabilities
    monkeypatch.setattr(capabilities, 'installed_cupy_version', lambda: (14, 0))
    for _ in range(3):
        forecast_gpu.require_forecast_gpu()
    assert calls == ['probe']
    # The opt-out still wins after a successful check.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(forecast_gpu.ForecastGPURequired):
        forecast_gpu.require_forecast_gpu()


def test_numpy_setup_allocation_survives_but_forecast_allocation_refuses(monkeypatch):
    from woof.config import RunConfig
    from woof.core.state import DomainState
    cfg = RunConfig(nx=12, ny=12, nz=12, dx=1000, dy=1000,
                    ztop=10000, dt=1, run_seconds=1)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    state = DomainState(cfg, array_module=np)
    assert isinstance(state.u, np.ndarray)
    with pytest.raises(forecast_gpu.ForecastGPURequired):
        DomainState(cfg)
    # Exercise the guarded import itself: the collection-time dependency
    # scanner otherwise follows its CUDA branch and skips this CPU test.
    step = importlib.import_module('woof.core.dycore').step
    with pytest.raises(forecast_gpu.ForecastGPURequired, match='NumPy setup state'):
        step(state, cfg)


def test_direct_step_import_survives_without_cupy_and_refuses_before_state_access(monkeypatch):
    step = importlib.import_module('woof.core.dycore').step
    from woof.config import RunConfig
    cfg = RunConfig(nx=12, ny=12, nz=12, dx=1000, dy=1000,
                    ztop=10000, dt=1, run_seconds=1)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(forecast_gpu.ForecastGPURequired):
        step(None, cfg)


@pytest.mark.parametrize('command', ['run', 'go', 'sim'])
def test_cli_no_device_has_no_traceback_or_dispatch(monkeypatch, capsys, command):
    from woof import cli, capabilities
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(capabilities, 'is_installed', lambda _: True)
    monkeypatch.setattr(cli, '_dispatch', lambda _: pytest.fail('dispatched without device'))
    arguments = [command, 'config.toml'] if command != 'sim' else [command, 'prepared', '--experiment-config', 'config.toml', '--outdir', 'out']
    assert cli.main(arguments) == 2
    text = capsys.readouterr().err
    assert 'ForecastGPURequired' in text and 'hides all NVIDIA devices' in text
    assert 'Traceback' not in text


def test_doctor_requirement_precedes_inventory_without_changing_json(monkeypatch, capsys):
    import json
    from woof import doctor
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(doctor, 'collect_checks', lambda *a, **kw: [])
    assert doctor.doctor_main(SimpleNamespace(json=True)) == 0
    out = capsys.readouterr()
    assert out.err.startswith('woof doctor: ForecastGPURequired:')
    assert json.loads(out.out) == []
