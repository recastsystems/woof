"""Forecast acquisition uses each source cycle's declared coverage."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from woof.chem_table import catalog, _freeze, _thaw
from woof.data_store_fetch import resolve_acquisition, DataStoreHorizonExceeded


def plan(row, start, hours, **kwargs):
    return resolve_acquisition(row, area="38,-80,44,-71", start=start,
        end=start + timedelta(hours=hours), **kwargs)


@pytest.mark.parametrize("hour", [0, 6, 12, 18])
def test_extended_cycles_accept_a_whole_day(hour):
    start = datetime(2023, 6, 7, hour, tzinfo=timezone.utc)
    result = plan(catalog().sources["hrrr-native-smoke"], start, 24)
    assert {request.cycle for request in result.requests} == {start.isoformat()}


@pytest.mark.parametrize("hour", [1, 5, 7, 23])
def test_beyond_all_source_cycles_still_refuses(hour):
    with pytest.raises(DataStoreHorizonExceeded, match="exceeds horizon"):
        plan(catalog().sources["hrrr-native-smoke"], datetime(2023, 6, 7, hour), 60)


def test_long_boundary_forecast_uses_previous_extended_cycle():
    result = plan(catalog().sources["rap-native-smoke"], datetime(2023, 6, 7), 24)
    assert {request.cycle for request in result.requests} == {"2023-06-06T21:00:00+00:00"}
    assert all(max(map(int, request.request["leadtime_hour"])) == 27
               for request in result.requests)


@pytest.mark.parametrize("source", ["rap-native-smoke", "hrrr-native-smoke"])
def test_existing_short_request_keeps_exact_cycle_and_request(source):
    row = catalog().sources[source]
    acquisition = _thaw(row.acquisition)
    acquisition.pop("cycle_horizon_h")
    legacy = replace(row, acquisition=_freeze(acquisition))
    start = datetime(2023, 6, 7, 5)
    before, after = plan(legacy, start, 3), plan(row, start, 3)
    assert [(r.cycle, r.request, r.sha256) for r in before.requests] == [
        (r.cycle, r.request, r.sha256) for r in after.requests]


def test_extended_cycle_respects_latency_and_never_uses_future_analysis():
    row = catalog().sources["hrrr-native-smoke"]
    start = datetime(2023, 6, 7, 6, tzinfo=timezone.utc)
    result = plan(row, start, 24, now=start)
    assert {request.cycle for request in result.requests} == {"2023-06-07T00:00:00+00:00"}


@pytest.mark.parametrize("horizons", [{"24": 48}, {"0": -1}, {"0": True}, [], {"0": 1.5}])
def test_invalid_extended_cycle_metadata_refuses(horizons):
    row = catalog().sources["hrrr-native-smoke"]
    acquisition = {**_thaw(row.acquisition), "cycle_horizon_h": horizons}
    with pytest.raises(ValueError, match="invalid cycle_horizon_h"):
        plan(replace(row, acquisition=_freeze(acquisition)), datetime(2023, 6, 7), 3)
