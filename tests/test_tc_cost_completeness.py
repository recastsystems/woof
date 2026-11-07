from types import SimpleNamespace
import pytest
from woof.tc.receipt import event_cost


@pytest.mark.parametrize('calls,sampled,unsampled,steps', [
    (300, 0, 44, 300), (300, 0, 300, 300), (300, 299, 0, 300), (1, 1, 0, 0)])
def test_incomplete_cost_cannot_report_zero(calls, sampled, unsampled, steps):
    meter = SimpleNamespace(calls=calls, sampled=sampled, unsampled=unsampled,
                            before_ms=0., advance_ms=0., total_ms=0.)
    row = event_cost(meter, steps=steps)
    assert not row['cost_measurement_valid']
    assert row['added_surface_gpu_ms_per_step'] is None
    assert row['exchange_and_waves_gpu_ms'] is None


def test_complete_sample_extrapolates_only_missing_samples():
    meter = SimpleNamespace(calls=300, sampled=250, unsampled=50,
                            before_ms=10., advance_ms=5., total_ms=15.)
    row = event_cost(meter, steps=600)
    assert row['cost_measurement_valid']
    assert row['exchange_and_waves_gpu_ms'] == 12.
    assert row['added_surface_gpu_ms_per_step'] == .03
