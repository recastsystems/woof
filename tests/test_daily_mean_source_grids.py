"""Daily lake-air means must not mix distinct source grid locations."""
from datetime import datetime, timedelta
from types import SimpleNamespace
import numpy as np
from woof.ingest.horiz import DailyMeanAirTemperature


def snapshot(hour, temperature, latitude, longitude):
    shape = (len(latitude), len(longitude))
    return SimpleNamespace(valid_time=datetime(2024, 1, 1) + timedelta(hours=hour),
        latitude=np.asarray(latitude), longitude=np.asarray(longitude), projection=None,
        fields={'SKINTEMP':np.full(shape, 290.), 'T2':np.full(shape, temperature)})


def test_different_shapes_and_equal_shapes_at_different_locations():
    frames = [snapshot(0, 280., [0, 1, 2], [0, 1, 2]),
              snapshot(3, 290., [0, 2], [0, 2]),
              snapshot(6, 300., [0, 2], [0, 2]),
              snapshot(9, 320., [1, 3], [0, 2])]
    mean = DailyMeanAirTemperature(frames)
    for frame, value, used in zip(frames, [280., 295., 295., 320.], [1, 2, 2, 1]):
        field, receipt = mean(frame)
        np.testing.assert_array_equal(field, np.full_like(frame.fields['T2'], value))
        assert len(receipt['times']) == used


def test_uniform_grid_keeps_forcing_order_arithmetic():
    values = [280.1234567, 291.2345678, 301.3456789]
    frames = [snapshot(i * 3, value, [0, 1], [0, 1]) for i, value in enumerate(values)]
    expected = frames[0].fields['T2'].copy()
    for frame in frames[1:]:
        expected = expected + frame.fields['T2']
    expected /= len(frames)
    field, _ = DailyMeanAirTemperature(frames)(frames[1])
    assert field.tobytes() == expected.tobytes()


def test_projected_grid_accepts_immutable_projection_metadata():
    from types import MappingProxyType
    first = snapshot(0, 280., [0, 1], [0, 1])
    second = snapshot(3, 300., [0, 1], [0, 1])
    first.projection = MappingProxyType({'family':'lambert_conformal',
        'parameters':MappingProxyType({'latin1':30.})})
    second.projection = {'family':'lambert_conformal', 'parameters':{'latin1':30.}}
    field, receipt = DailyMeanAirTemperature([first, second])(first)
    np.testing.assert_array_equal(field, np.full((2, 2), 290.))
    assert len(receipt['times']) == 2
