from datetime import datetime, timezone
import json
import numpy as np
from netCDF4 import Dataset
import pytest

from woof.hex.init_clock import valid_time


@pytest.mark.parametrize("clock", ["", "2025-05-19_06:00:00"])
def test_current_clock_wins_over_start_attribute(tmp_path, clock):
    path = tmp_path / "init.nc"
    with Dataset(path, "w") as state:
        state.createDimension("Time", 1)
        state.createDimension("StrLen", 64)
        state.config_start_time = "2025-05-19_03:00:00"
        state.createVariable("xtime", "S1", ("Time", "StrLen"))[:] = np.frombuffer(
            clock.ljust(64).encode(), dtype="S1")
    expected = 6 if clock else 3
    assert valid_time(path) == datetime(2025, 5, 19, expected, tzinfo=timezone.utc)


def test_multiple_records_refuse(tmp_path):
    path = tmp_path / "history.nc"
    with Dataset(path, "w") as state:
        state.createDimension("Time", 2)
        state.createDimension("StrLen", 19)
        state.createVariable("xtime", "S1", ("Time", "StrLen"))[:] = np.frombuffer(
            b"2025-05-19_03:00:00", dtype="S1")
    with pytest.raises(ValueError, match="multiple clocks"):
        valid_time(path)


def test_manifest_requires_known_schema_and_normalizes_utc(tmp_path):
    path = tmp_path / "init.json"
    row = dict(schema="gpuwm-hex.mapped-init/v1", valid="2025-05-19T07:00:00+01:00")
    path.write_text(json.dumps(row))
    assert valid_time(path) == datetime(2025, 5, 19, 6, tzinfo=timezone.utc)
    path.write_text(json.dumps(dict(row, schema="unknown")))
    with pytest.raises(ValueError, match="unrecognized"):
        valid_time(path)
