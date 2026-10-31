"""The public demo owns fresh run paths without escaping its work directory."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def example():
    path = Path(__file__).resolve().parents[1] / "examples/prepare_simulated_radar_bands.py"
    spec = importlib.util.spec_from_file_location("radar_band_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("label", ["", ".", "..", "../outside", "/absolute", "nested/run"])
def test_invalid_run_labels_fail_before_creating_or_starting_work(example, monkeypatch, tmp_path, label):
    work = tmp_path / "uncreated"
    monkeypatch.setattr("sys.argv", ["example", "on", "--work", str(work), "--run-label", label])
    with pytest.raises(SystemExit) as refused:
        example.main()
    assert refused.value.code == 2
    assert not work.exists()


def test_fresh_run_label_reaches_forecast_without_changing_science(example, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(example, "forecast", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr("sys.argv", ["example", "on", "--work", str(tmp_path), "--run-label", "fresh-on"])
    example.main()
    assert calls == [((tmp_path.resolve(), "on"), {"smoke": False, "label": "fresh-on"})]
