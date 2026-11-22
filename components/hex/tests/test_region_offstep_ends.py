"""A hex region whose length is not a whole number of its source's posting steps runs: its boundary series reads the
next posted hour past the run's end, and the forecast's boundary pool interpolates to the end in time, as it does
between any two boundary times.

THE BREAKAGE THESE GUARD (2026-11-13 walk of the hex region orders the site accepts): both region doors planned a
region's edges only on a step that divides the run's length, so the site quoted, and the box then refused at plan,
"<source> boundary cycle ... writes no lead at every boundary time from 0 h to 118 h ...": ICON 100 h from 00Z, 118 h
from 18Z and 10 h from f090; IFS, AIFS and GEFS for 25 h; GFS for 125 h. Every test here but the controls fails on the
combined fixes before this change. A run past the cycle's last posted hour stays refused, and the refusal names the
last hour and the longest run that has one.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex import init_sources as src
from woof.hex import regional_door as door
from woof.hex.cli import build_parser

TABLE = src.load_table(None)


def row(name):
    return src.resolve_source(TABLE, name)


def at(name, cycle, lead, hours):
    """boundary_plan's (step, span) for a region of ``hours`` from ``lead`` of ``cycle``."""
    import math

    _, _, offset, step = door.boundary_plan(row(name), datetime.strptime(cycle, "%Y-%m-%dT%H"), lead, hours)
    return step, math.ceil(hours / step) * step


@pytest.mark.parametrize("name, cycle, lead, hours, step, span", [
    ("icon", "2026-11-13T00", 0, 100, 3, 102),     # hourly to f078, then 3-hourly: f099 and f102 bracket the end
    ("icon", "2026-11-12T18", 0, 118, 3, 120),     # the 18 UTC run's last hour, f120, is past the end
    ("icon", "2026-11-13T00", 90, 10, 3, 12),      # from f090 to f100: reads f099 and f102
    ("ifs", "2026-11-12T12", 0, 25, 3, 27),
    ("ifs", "2026-11-12T12", 0, 100, 3, 102),
    ("ifs-0p1", "2026-11-12T12", 0, 25, 3, 27),
    ("aifs", "2026-11-12T12", 0, 25, 6, 30),
    ("gefs", "2026-11-12T12", 0, 25, 3, 27),
    ("gfs", "2026-11-12T12", 0, 125, 3, 126),      # hourly to f120, then 3-hourly
    ("gfs", "2026-11-12T12", 120, 4, 3, 6),        # past f120 only: f123 and f126 bracket the end
    ("gfs", "2026-11-12T12", 0, 383, 3, 384),      # one hour short of the cycle's last
])
def test_a_region_ending_between_posted_hours_reads_the_next_one(name, cycle, lead, hours, step, span):
    assert at(name, cycle, lead, hours) == (step, span)


@pytest.mark.parametrize("name, cycle, lead, hours, step", [
    ("gfs", "2026-11-12T12", 0, 24, 1),
    ("icon", "2026-11-13T00", 0, 24, 1),
    ("ifs", "2026-11-12T12", 0, 12, 3),
    ("gefs", "2026-11-12T12", 0, 240, 3),
    ("aifs", "2026-11-12T12", 0, 360, 6),
])
def test_a_length_on_the_step_is_planned_as_before(name, cycle, lead, hours, step):
    # Control: a whole number of steps keeps its step and ends on its last boundary time.
    assert at(name, cycle, lead, hours) == (step, hours)


@pytest.mark.parametrize("name, cycle, lead, hours, last, most", [
    ("icon", "2026-11-12T18", 0, 121, 120, 120),
    ("icon", "2026-11-13T00", 0, 181, 180, 180),
    ("icon", "2026-11-13T00", 177, 4, 180, 3),
    ("ifs", "2026-11-12T18", 0, 145, 144, 144),
    ("gfs", "2026-11-12T12", 381, 5, 384, 3),
    ("ifs", "2024-05-14T12", 0, 241, 240, 240),    # a dated ladder: 2024 00/12 UTC runs stopped at f240
    ("ifs", "2024-05-14T18", 0, 91, 90, 90),       # and 06/18 UTC runs at f090
])
def test_a_region_past_the_cycles_last_hour_is_refused_naming_the_longest_run(name, cycle, lead, hours, last, most):
    with pytest.raises(door.RegionalRefusal) as refused:
        at(name, cycle, lead, hours)
    text = str(refused.value)
    assert "writes no lead" in text
    assert f"its last posted hour is f{last:03d}" in text
    assert f"the longest run from f{lead:03d} with a posted boundary hour at or past its end is {most} h" in text
    # The longest run it names does plan.
    assert at(name, cycle, lead, most)[0]


def test_an_unposted_start_is_refused_with_no_run_to_name():
    with pytest.raises(door.RegionalRefusal, match="no run from f382 has a posted boundary step"):
        at("gfs", "2026-11-12T12", 382, 2)


def _regional(tmp_path, source, cycle, lead, hours, *extra):
    return door.plan(build_parser().parse_args([
        "regional", "--point", "50.0,10.0", "--cycle", cycle, "--source", source, "--lead-hours", str(lead),
        "--hours", str(hours), "--fine-dx-m", "30000", "--radius-km", "600", "--background-km", "120",
        "--work-dir", str(tmp_path / "run"), *extra]))


def test_the_regional_plan_carries_the_series_past_the_end_and_the_forecast_stops_at_the_end(tmp_path):
    value = _regional(tmp_path, "icon", "2026-11-13T00", 90, 10)
    assert value["start_time"] == "2026-11-16_18:00:00"
    assert value["stop_time"] == "2026-11-17_04:00:00"
    assert value["boundary_interval_hours"] == 3 and value["boundary_hours"] == 12
    assert value["boundary_stop_time"] == "2026-11-17_06:00:00"


def test_the_regional_table_stage_reads_every_lead_through_the_one_past_the_end(tmp_path, monkeypatch):
    seen = []

    def run(argv, stdout=None, stderr=None):
        lead = int(argv[argv.index("--lead-hours") + 1])
        seen.append(lead)
        out = Path(argv[argv.index("--out-dir") + 1])
        (out / f"MET:2026-11-{13 + lead // 24:02d}_{lead % 24:02d}").write_text("")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(door.subprocess, "run", run)
    value = {"paths": {"met": str(tmp_path / "met")}, "lead_hours": 90, "hours": 10, "boundary_interval_hours": 3,
             "source": {"id": "icon"}, "cycle": "2026-11-13T00"}
    door._stage_intermediate_from_table(SimpleNamespace(workers=1), value, SimpleNamespace(note=lambda text: None))
    assert sorted(seen) == [90, 93, 96, 99, 102]
    assert door._table_decode_workers(SimpleNamespace(workers=8), value) == 5


def test_the_regional_boundary_stage_writes_the_series_through_its_last_boundary_time(tmp_path, monkeypatch):
    value = _regional(tmp_path, "ifs", "2026-11-12T12", 0, 25)
    called = []
    monkeypatch.setattr(door, "_generated", lambda value: {"cull": {"lbc_dir": str(tmp_path / "lbc")}})
    monkeypatch.setattr(door, "_first_guess", lambda value, boundary=False: {"nfglevels": 14, "use_spechumd": "no"})
    monkeypatch.setattr(door, "_cli", lambda argv, status: called.append(argv) or 0)
    door._stage_lbc(SimpleNamespace(boundary_recipe=None), value, SimpleNamespace(note=lambda text: None))
    argv = called[0]
    assert argv[argv.index("--start-time") + 1] == "2026-11-12_12:00:00"
    assert argv[argv.index("--stop-time") + 1] == "2026-11-13_15:00:00"   # f027, past the 25 h end
    assert value["stop_time"] == "2026-11-13_13:00:00"


def test_the_country_door_plans_endpoints_and_boundaries_through_the_one_past_the_end(tmp_path):
    from woof.hex.country_door import plan

    args = build_parser().parse_args([
        "country", "--point", "50.000,10.000", "--radius-km", "1200", "--source", "icon", "--cycle", "2026-11-13T00",
        "--lead-hours", "90", "--hours", "10", "--history-every-minutes", "60", "--work-dir", str(tmp_path / "run"),
        "--skip", "score", "--forecast-arg=--physics-suite=convection_permitting"])
    value = plan(args)
    endpoints = value["boundary_endpoints"]
    leads = [int(e["argv"][e["argv"].index("--lead-hours") + 1]) for e in endpoints]
    assert leads == [90, 93, 96, 99, 102]
    assert endpoints[-1]["valid_time"] == "2026-11-17_06:00:00"
    lbc = value["commands"]["lbc"]
    assert lbc[lbc.index("--stop-time") + 1] == "2026-11-17_06:00:00"


def test_a_requested_era5_step_that_does_not_divide_the_run_reads_the_analysis_past_the_end():
    assert door.boundary_plan(row("era5"), datetime(2011, 4, 26, 6), 0, 48, requested=5)[3] == 5
    assert door.boundary_span(48, 5) == 50 and door.boundary_span(48, None) == 48
    with pytest.raises(door.RegionalRefusal, match="whole multiple of 1 h"):
        door.boundary_plan(row("era5"), datetime(2011, 4, 26, 6), 0, 48, requested=0)
