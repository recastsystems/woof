"""A region from a global source at a forecast lead: start = cycle + lead, boundaries from the same cycle's later leads."""
import os

import pytest

from woof.hex import regional_door as door
from woof.hex.cli import build_parser


def plan(tmp_path, *extra):
    args = build_parser().parse_args([
        "regional", "--point", "33.5,-101.0", "--work-dir", str(tmp_path / "run"), *extra])
    return args, door.plan(args)


def test_gfs_at_lead_348_starts_at_the_valid_time(tmp_path):
    _, value = plan(tmp_path, "--source", "gfs", "--cycle", "2026-10-23T18", "--lead-hours", "348",
                    "--hours", "24", "--fine-dx-m", "3000", "--radius-km", "550", "--background-km", "48")
    assert value["cycle"] == "2026-10-23T18"
    assert value["start_time"] == "2026-11-07_06:00:00"
    assert value["stop_time"] == "2026-11-08_06:00:00"
    assert value["source"]["intermediate_row"] is None
    # The finest step GFS posts over 348-372 h (3 h past 120 h), not the row's 6 h interval.
    assert value["boundary_interval_hours"] == 3
    assert value["source"]["first_guess"]["nfglevels"] == 34


def test_lead_off_the_source_cadence_or_past_its_range_is_refused(tmp_path):
    with pytest.raises(door.RegionalRefusal, match="writes no lead"):
        plan(tmp_path, "--source", "gfs", "--cycle", "2026-10-23T18", "--lead-hours", "349", "--hours", "24")
    with pytest.raises(door.RegionalRefusal, match="writes no lead"):
        plan(tmp_path, "--source", "gfs", "--cycle", "2026-10-23T18", "--lead-hours", "372", "--hours", "24")


def test_limited_area_source_keeps_its_own_door_and_lead_zero(tmp_path):
    _, value = plan(tmp_path, "--source", "hrrr", "--cycle", "2026-10-23T18", "--hours", "3")
    assert value["start_time"] == "2026-10-23_18:00:00" and value["lead_hours"] == 0
    assert value["source"]["intermediate_row"] == "hrrr"
    with pytest.raises(door.RegionalRefusal, match="lead"):
        plan(tmp_path, "--source", "hrrr", "--cycle", "2026-10-23T18", "--hours", "3", "--lead-hours", "6")


@pytest.mark.skipif(os.name == "nt", reason="WPS intermediate names carry ':', which NTFS reads as a stream")
def test_table_intermediates_run_one_decode_per_boundary_lead(tmp_path, monkeypatch):
    args, value = plan(tmp_path, "--source", "gfs", "--cycle", "2026-10-23T18", "--lead-hours", "348",
                       "--hours", "6")
    calls = []

    class Done:
        returncode = 0

    def run(argv, stdout=None, stderr=None):
        lead = int(argv[argv.index("--lead-hours") + 1])
        out = door.Path(argv[argv.index("--out-dir") + 1])
        calls.append(lead)
        (out / f"MET:2026-11-07_{(lead - 342):02d}").write_bytes(b"x")
        return Done()

    monkeypatch.setattr(door.subprocess, "run", run)

    class Status:
        def note(self, text):
            pass

    result = door._stage_intermediate(args, value, Status())
    # GFS posts every 3 h past 120 h, so the region reads every one of those leads, not the row's 6 h interval.
    assert sorted(calls) == [348, 351, 354]
    assert result["files"] == 3
    assert door._met_file(value).name == "MET:2026-11-07_06"


def test_hex_suite_shapes_the_init_and_mpas_suites_do_not():
    assert door._init_suite_argv({"forecast_args": ["--physics-suite=rrfs_v2_dev"]}) == [
        "--physics-suite", "rrfs_v2_dev"]
    assert door._init_suite_argv({"forecast_args": ["--physics-suite", "convection_permitting"]}) == []
    assert door._init_suite_argv({"forecast_args": []}) == []
