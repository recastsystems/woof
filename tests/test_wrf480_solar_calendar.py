"""The wrf_480 solar generation reads WRF's frame calendar words.

WRF forms JULIAN as REAL(dayOfYear_r8) - 1.0 (frame/module_domain.F
domain_clock_get).  The earlier statements rounded the day fraction once;
on binade-edge days that word differs, and the native WRF 4.8.0 strict
replay (tools/wrf480_solar_replay) showed DECLIN, COSZEN and HRANG follow
it.  wrf_461 keeps the earlier words byte for byte.
"""
from datetime import datetime, timedelta

import numpy as np
import pytest

from woof.core.solar import radiation_calendar
from woof.core.swint import calendar_scalars
from woof.core.wrf_radiation_clock import wrf_calendar_scalars


def _old(start, elapsed):
    valid = start + timedelta(seconds=float(elapsed))
    hour = valid.hour + valid.minute / 60.0 + valid.second / 3600.0 + valid.microsecond / 3.6e9
    julian = np.float32((valid.timetuple().tm_yday - 1) + hour / 24.0)
    gmt = np.float32(start.hour + start.minute / 60.0 + start.second / 3600.0
                     + start.microsecond / 3.6e9)
    return julian, np.float32(float(elapsed) / 60.0), gmt


def _bits(values):
    return tuple(int(np.float32(v).view(np.uint32)) for v in values)


# Day 127 (2023-05-08) at 00Z + 7 min steps: the replay's edge-d127 calls.
EDGE = [(datetime(2023, 5, 8), 420.0 * k) for k in range(0, 206, 5)]
PLAIN = [(datetime(2024, 5, 25, 18), 180.0 * k) for k in range(21)]


@pytest.mark.parametrize("start,elapsed", EDGE + PLAIN)
def test_wrf_480_is_the_frame_calendar(start, elapsed):
    assert _bits(radiation_calendar(start, elapsed)) == _bits(wrf_calendar_scalars(start, elapsed))
    assert _bits(calendar_scalars(start, elapsed)) == _bits(wrf_calendar_scalars(start, elapsed))


@pytest.mark.parametrize("start,elapsed", EDGE + PLAIN)
def test_wrf_461_keeps_the_earlier_words(start, elapsed):
    assert _bits(radiation_calendar(start, elapsed, generation="wrf_461")) == _bits(_old(start, elapsed))
    assert _bits(calendar_scalars(start, elapsed, generation="wrf_461")) == _bits(_old(start, elapsed))


def test_the_edge_day_differs_and_a071_does_not():
    edge = sum(_bits(radiation_calendar(s, e))[0] != _bits(_old(s, e))[0] for s, e in EDGE)
    plain = sum(_bits(radiation_calendar(s, e)) != _bits(_old(s, e)) for s, e in PLAIN)
    assert edge > 0
    assert plain == 0


def test_unknown_generation_refuses():
    with pytest.raises(ValueError, match="COSZEN"):
        radiation_calendar(datetime(2024, 1, 1), 0.0, generation="wrf_471")
