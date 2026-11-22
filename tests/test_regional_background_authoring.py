"""Regional authoring uses exact source windows and preparation owners."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import replace
import tomllib

import pytest

from conftest import requires_cupy


@requires_cupy
def test_staged_prepare_only_retains_output_and_never_calls_forecast(tmp_path, monkeypatch):
    from woof import runplan
    from test_runplan import _executed_staged_chain
    original = runplan._staged_chain
    returned = []
    def preparation(*args, **kwargs):
        target = kwargs['run_dir'] / 'chain' / 'run'
        target.mkdir(parents=True, exist_ok=True)
        frame = target / 'preserved.nc'
        frame.write_bytes(b'committed input generation')
        value = original(*args, **kwargs, prepare_only=True)
        assert frame.read_bytes() == b'committed input generation'
        returned.append(value)
        return value
    monkeypatch.setattr(runplan, '_staged_chain', preparation)
    staged, _, _, _ = _executed_staged_chain(tmp_path, monkeypatch)
    assert [row[0] for row in staged] == ['fetch', 'prepare']
    assert returned[0]['forecast_started'] is False
    assert Path(returned[0]['experiment_config']).is_file()


def test_native_hourly_prepare_only_returns_published_authorities(tmp_path, monkeypatch):
    from woof import runplan
    from test_runplan_tiles import _drive
    from woof import stage_cli
    # The existing process double writes a digest-bearing empty proof. This
    # control observes its published authority relay, not proof validation.
    monkeypatch.setattr(stage_cli, 'resolve_bundle', lambda root: dict(
        document=root / 'proof.json', source='hrrr', layout='single', domains=1))
    original = runplan._hrrr_chain
    returned = []
    def preparation(*args, **kwargs):
        value = original(*args, **kwargs, prepare_only=True)
        returned.append(value)
        return value
    monkeypatch.setattr(runplan, '_hrrr_chain', preparation)
    captured, _, stages = _drive(tmp_path, monkeypatch)
    assert not captured
    assert [row[0] for row in stages] == ['fetch', 'prepare']
    assert returned[0]['forecast_started'] is False
    assert Path(returned[0]['experiment_config']).parent == Path(returned[0]['prepared_root'])


def _automatic_sources():
    from woof.background_contract import catalog
    return [row['source'] for row in catalog()['sources'] if 'automatic' in row.get('initialization_modes', [])]

