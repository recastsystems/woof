from datetime import datetime
from pathlib import Path

from tools.tmn_surface_forecast_receipt import history_paths, valid_time


def test_native_history_scan_excludes_json_authority_sidecars(tmp_path):
    history = tmp_path / "wrfout"
    history.mkdir()
    index = tmp_path / "index"
    index.mkdir()
    names = ["wrfout_d01_2019-09-05_00_00_00", "wrfout_d01_2019-09-04_12_00_00"]
    for name in names:
        (history / name).write_bytes(b"native dataset placeholder")
        (index / (name + ".json")).write_text("{}")
    assert [path.name for path in history_paths(tmp_path)] == sorted(names)


def test_native_history_clock_accepts_both_writer_spellings(tmp_path):
    for name in ["wrfout_d01_2019-09-06_00:00:00", "wrfout_d01_2019-09-06_00_00_00"]:
        assert valid_time(Path(name)) == datetime(2019, 9, 6)
