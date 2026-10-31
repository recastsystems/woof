"""A hex run that starts from a later forecast hour of its cycle (--lead-hours).

The breakage this covers: every hex door started at its cycle's first hour, so a run over a pattern a source shows at
day 14 (GFS 2026-10-23 18Z f348) had no front door; WOOF had to refuse it ("A hex run starts at its data's first hour").
With --lead-hours the init is that hour's forecast, the start time is cycle + lead, and a region's boundaries read the
same cycle further on.
"""

import pytest

from woof.hex import global_door
from woof.hex.cli import build_parser
from woof.hex.country_door import plan as country_plan
from woof.hex.errors import MpasPortError
from woof.hex.init_sources import InitSourceRefusal
from woof.hex.lbc_producer import source_endpoint_commands


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_boundary_endpoints_read_the_same_cycle_from_the_start_lead(tmp_path):
    commands = source_endpoint_commands(source="gfs", cycle="2026-10-23T18", hours=12, out_dir=tmp_path,
                                        lead_hours=348)
    # GFS posts every 3 h past 120 h: the endpoints are every one of those leads.
    assert [item["valid_time"] for item in commands] == [
        "2026-11-07_06:00:00", "2026-11-07_09:00:00", "2026-11-07_12:00:00", "2026-11-07_15:00:00",
        "2026-11-07_18:00:00"]
    assert {_flag(item["argv"], "--cycle") for item in commands} == {"2026-10-23T18"}
    assert [_flag(item["argv"], "--lead-hours") for item in commands] == ["348", "351", "354", "357", "360"]


def test_without_a_lead_the_endpoints_are_the_cycle_s_own_hours(tmp_path):
    commands = source_endpoint_commands(source="gfs", cycle="2026-10-23T18", hours=6, out_dir=tmp_path)
    # GFS posts hourly to 120 h: a 6 h run reads every hour, not only its two ends.
    assert [_flag(item["argv"], "--lead-hours") for item in commands] == [str(h) for h in range(7)]
    assert commands[0]["valid_time"] == "2026-10-23_18:00:00"


def test_a_lead_past_the_source_horizon_or_off_its_cadence_is_refused(tmp_path):
    with pytest.raises(InitSourceRefusal, match="outside the row's forecast cadence/range"):
        source_endpoint_commands(source="gfs", cycle="2026-10-23T18", hours=48, out_dir=tmp_path, lead_hours=348)
    with pytest.raises(InitSourceRefusal, match="outside the row's forecast cadence/range"):
        source_endpoint_commands(source="gfs", cycle="2026-10-23T18", hours=6, out_dir=tmp_path, lead_hours=347)


def test_an_analysis_source_has_no_lead(tmp_path):
    with pytest.raises(MpasPortError, match="analysis source"):
        source_endpoint_commands(source="era5", cycle="2024-05-06T12", hours=6, out_dir=tmp_path, lead_hours=6)


def test_a_point_region_starts_at_the_lead_and_its_init_is_that_hour(tmp_path):
    args = build_parser().parse_args([
        "country", "--point", "37,-98", "--radius-km", "600", "--fine-dx-m", "3000", "--background-km", "96",
        "--source", "gfs", "--cycle", "2026-10-23T18", "--lead-hours", "348", "--hours", "6",
        "--work-dir", str(tmp_path / "run"), "--forecast-arg=--physics-suite=rrfs_v2_dev"])
    value = country_plan(args)
    assert value["start_time"] == "2026-11-07_06:00:00"
    init = value["commands"]["init"]
    assert _flag(init, "--cycle") == "2026-10-23T18" and _flag(init, "--lead-hours") == "348"
    forecast = value["commands"]["forecast"]
    assert _flag(forecast, "--start-time") == "2026-11-07_06:00:00"
    assert "gfs 2026-10-23T18 f348" in _flag(forecast, "--init-source")
    assert "--physics-suite=rrfs_v2_dev" in forecast
    # GFS posts every 3 h past 120 h: every one of those leads drives the boundary.
    assert [_flag(e["argv"], "--lead-hours") for e in value["boundary_endpoints"]] == ["348", "351", "354"]
    assert _flag(value["commands"]["lbc"], "--start-time") == "2026-11-07_06:00:00"
    build_parser().parse_args(forecast)


def test_a_globe_starts_at_the_lead(tmp_path, monkeypatch):
    monkeypatch.setattr(global_door, "_visible_cards", lambda: [])
    args = build_parser().parse_args([
        "global", "--row", "global-60km", "--source", "gfs", "--cycle", "2026-10-23T18", "--lead-hours", "348",
        "--hours", "12", "--work-dir", str(tmp_path / "run"), "--cache-dir", str(tmp_path / "cache"), "--stage", "plan"])
    value = global_door.plan(args)
    assert value["start_time"] == "2026-11-07_06:00:00" and value["lead_hours"] == 348
    assert _flag(value["commands"]["init"], "--lead-hours") == "348"
    assert _flag(value["commands"]["forecast"], "--start-time") == "2026-11-07_06:00:00"


def test_a_globe_lead_off_the_cadence_is_refused_before_any_download(tmp_path):
    args = build_parser().parse_args([
        "global", "--row", "global-60km", "--source", "gfs", "--cycle", "2026-10-23T18", "--lead-hours", "350",
        "--work-dir", str(tmp_path / "run"), "--cache-dir", str(tmp_path / "cache"), "--stage", "plan"])
    with pytest.raises(InitSourceRefusal, match="cadence/range"):
        global_door.plan(args)


def test_mesh_plan_takes_the_lead():
    args = build_parser().parse_args([
        "mesh-plan", "--global-mesh", "--point", "37,-98", "--fine-dx-m", "3000", "--background-km", "96",
        "--radius-km", "800", "--run", "--source", "gfs", "--cycle", "2026-10-23T18", "--lead-hours", "348"])
    assert args.lead_hours == 348
