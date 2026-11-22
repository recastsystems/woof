"""Hex regions from ICON and from GEFS, on both region doors (the class, not one order).

THE BREAKAGES THESE GUARD (2026-11-13 class walk of the site's hex region orders):
- every ICON region was refused at plan, "icon declares no forecast cadence for its boundaries": the icon row read
  step 0 only, and even with forecast hours gdt101_remap was handed every record as lead 0 at the init's valid time,
  so no ICON forecast hour could be read;
- every GEFS region on the regional door failed after its mesh, "source 'gefs' is an ensemble; pass --member": that
  door took no --member.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from woof.hex import init_sources as src
from woof.hex import regional_door as door
from woof.hex.cli import build_parser
from woof.hex.lbc_producer import source_endpoint_commands
from woof.hex.source_horizon import last_posted_lead

TABLE = src.load_table(None)
ICON = src.resolve_source(TABLE, "icon")


def _regional(tmp_path, source, cycle, lead, hours, *extra):
    return door.plan(build_parser().parse_args([
        "regional", "--point", "50.0,10.0", "--cycle", cycle, "--source", source, "--lead-hours", str(lead),
        "--hours", str(hours), "--fine-dx-m", "30000", "--radius-km", "600", "--background-km", "120",
        "--work-dir", str(tmp_path / "run"), *extra]))


def test_icon_posts_dwds_forecast_hours():
    """DWD's listing (2026-11-13): 00/12 UTC hourly to f078 then 3-hourly to f180; 06/18 UTC the same to f120."""
    forecast = ICON["forecast"]
    posted = lambda hour, lead: src._lead_posted(forecast, hour, lead)  # noqa: E731
    assert all(posted(0, lead) for lead in range(0, 79)) and all(posted(12, lead) for lead in range(81, 181, 3))
    assert not posted(0, 79) and not posted(0, 183) and posted(0, 180)
    assert posted(6, 120) and not posted(6, 123) and posted(18, 78) and not posted(18, 80)
    assert last_posted_lead(ICON, 0) == 180 and last_posted_lead(ICON, 6) == 120


def test_an_icon_lead_reads_that_hours_objects_and_the_cycles_time_invariant_ones():
    plans = src.plan_products(ICON, src.Cycle.parse("2026-11-13T00"), None, 7)
    names = [plan.filename for plan in plans]
    assert "icon_global_icosahedral_pressure-level_2026111300_007_850_T.grib2" in names
    assert "icon_global_icosahedral_soil-level_2026111300_007_54_T_SO.grib2" in names
    assert "icon_global_icosahedral_time-invariant_2026111300_HSURF.grib2" in names
    assert not any("_000_" in name for name in names)
    by_name = {plan.filename: plan for plan in plans}
    assert by_name["icon_global_icosahedral_pressure-level_2026111300_007_850_T.grib2"].remap["lead"] == "7"
    assert "lead" not in by_name["icon_global_icosahedral_time-invariant_2026111300_HSURF.grib2"].remap
    # Step 0 is the file it always was.
    zero = [plan.filename for plan in src.plan_products(ICON, src.Cycle.parse("2026-11-13T00"), None, 0)]
    assert "icon_global_icosahedral_pressure-level_2026111300_000_850_T.grib2" in zero
    with pytest.raises(src.InitSourceRefusal, match="lead 123"):
        src.plan_products(ICON, src.Cycle.parse("2026-11-13T06"), None, 123)


def test_the_remap_job_names_the_source_cycle_and_each_records_lead(tmp_path, monkeypatch):
    """gdt101_remap refuses a record whose reference time or lead differs from its job's ('forecast lead mismatch')."""
    calls = []

    class Done:
        returncode = 0
        stderr = ""

    def run(argv, **_kw):
        calls.append(list(argv))
        if argv[1] == "plan":
            Path(argv[-4]).write_bytes(b"plan")
        return Done()

    monkeypatch.setattr(src.subprocess, "run", run)
    plans = src.plan_products(ICON, src.Cycle.parse("2026-11-13T00"), None, 7)
    products = [{"file": str(tmp_path / plan.filename), "id": plan.product_id, "remap": plan.remap} for plan in plans]
    tool = tmp_path / "gdt101_remap"
    tool.write_bytes(b"tool")
    src.remap(ICON, products, src.Cycle.parse("2026-11-13T00"), tmp_path, tool)
    assert calls[0][1] == "plan" and "2026111300" in calls[0]
    jobs = [line.split("\t") for line in (tmp_path / "remap" / "jobs.tsv").read_text().splitlines()]
    lead = {Path(job[0]).name: (job[-2], job[-1]) for job in jobs}
    assert lead["icon_global_icosahedral_pressure-level_2026111300_007_850_T.grib2"] == ("2026111300", str(7 * 3600))
    assert lead["icon_global_icosahedral_time-invariant_2026111300_HSURF.grib2"] == ("2026111300", "0")


def test_init_source_remaps_with_the_source_cycle_not_the_valid_time():
    import inspect

    text = inspect.getsource(src.run_init_source_command)
    assert 'remap(row, receipt["products"], source_cycle, out_dir, remapper)' in text


def test_a_remap_lead_other_than_the_template_is_refused():
    row = json.loads(json.dumps(ICON))
    row["products"][4]["remap"]["lead"] = "3"
    with pytest.raises(src.InitSourceRefusal, match="remap lead"):
        src.validate_row(row)


def test_single_stream_objects_come_several_at_once(tmp_path, monkeypatch):
    """ICON's ~120 bz2 objects a lead came one after another (992 s for one lead on a test host)."""
    import threading
    import time

    live, most = [0], [0]
    lock = threading.Lock()

    def whole(plan, target, jobs=1):
        with lock:
            live[0] += 1
            most[0] = max(most[0], live[0])
        time.sleep(0.05)
        target.write_bytes(b"x")
        with lock:
            live[0] -= 1
        return {"url": plan.url}

    monkeypatch.setattr(src, "_fetch_whole", whole)
    plans = src.plan_products(ICON, src.Cycle.parse("2026-11-13T00"), None, 3)[:20]
    out = src.acquire(plans, tmp_path, src.Cycle.parse("2026-11-13T03"))
    assert [Path(item["file"]).name for item in out] == [plan.filename for plan in plans]
    assert 1 < most[0] <= src.SINGLE_STREAM_OBJECTS_AT_ONCE


def test_an_icon_region_has_boundaries_on_both_doors(tmp_path):
    """A 24 h country region from ICON reads 25 hourly leads of its one cycle; the regional door plans the same step."""
    commands = source_endpoint_commands(source="icon", cycle="2026-11-13T00", hours=24.0, out_dir=tmp_path / "bs")
    leads = [int(item["argv"][item["argv"].index("--lead-hours") + 1]) for item in commands]
    cycles = {item["argv"][item["argv"].index("--cycle") + 1] for item in commands}
    assert leads == list(range(0, 25)) and cycles == {"2026-11-13T00"}
    value = _regional(tmp_path, "icon", "2026-11-13T00", 0, 24)
    assert value["boundary_interval_hours"] == 1 and value["boundary_lead_hours"] == 0
    # A start hour past f078 takes ICON's 3-hourly steps; a run past a 06 UTC run's f120 is refused before the mesh.
    assert _regional(tmp_path, "icon", "2026-11-13T00", 90, 12)["boundary_interval_hours"] == 3
    with pytest.raises(door.RegionalRefusal, match="writes no lead"):
        _regional(tmp_path, "icon", "2026-11-12T06", 108, 24)


def test_icons_upper_air_donor_reaches_every_icon_hour():
    from woof.ingest.deep_soil_donor import donor_candidates
    from woof.ingest.upper_air_donor import validate_upper_air_donor

    donor = validate_upper_air_donor(ICON["upper_air_donor"])
    for lead in (0, 78, 120, 180):
        valid = datetime(2026, 11, 13, 0) + __import__("datetime").timedelta(hours=lead)
        assert donor_candidates(donor, valid, datetime(2026, 11, 13, 0)), lead


def test_a_gefs_region_on_the_regional_door_names_its_member(tmp_path):
    value = _regional(tmp_path, "gefs", "2026-11-11T12", 0, 6, "--member", "0")
    assert value["member"] == 0 and value["member_rows"] == ["gefs"]
    with pytest.raises(door.RegionalRefusal, match="is an ensemble; pass --member"):
        _regional(tmp_path, "gefs", "2026-11-11T12", 0, 6)
    with pytest.raises(door.RegionalRefusal, match="outside"):
        _regional(tmp_path, "gefs", "2026-11-11T12", 0, 6, "--member", "31")
    with pytest.raises(door.RegionalRefusal, match="declare no members"):
        _regional(tmp_path, "gfs", "2026-11-11T12", 0, 6, "--member", "0")


def test_every_table_hour_of_a_gefs_region_reads_the_member(tmp_path, monkeypatch):
    value = _regional(tmp_path, "gefs", "2026-11-11T12", 0, 6, "--member", "0")
    seen = []

    class Done:
        returncode = 0

    def run(argv, **_kw):
        seen.append(list(argv))
        out = Path(argv[argv.index("--out-dir") + 1])
        (out / f"MET:{len(seen)}").write_bytes(b"")
        return Done()

    monkeypatch.setattr(door.subprocess, "run", run)
    door._stage_intermediate_from_table(type("A", (), {"workers": 1})(), value, type("S", (), {"note": print}))
    assert seen and all(argv[argv.index("--member") + 1] == "0" for argv in seen)
    gfs = _regional(tmp_path / "gfs", "gfs", "2026-11-11T12", 0, 6)
    seen.clear()
    door._stage_intermediate_from_table(type("A", (), {"workers": 1})(), gfs, type("S", (), {"note": print}))
    assert seen and not any("--member" in argv for argv in seen)


def test_a_country_regions_boundaries_read_the_levels_each_endpoint_wrote(tmp_path):
    """ICON's 19 levels plus its GFS upper-air donor's 8 made 27-level boundary files; rw_mpas_lbc, told 19, refused."""
    from woof.hex.country_door import _measured_first_guess

    met = tmp_path / "boundary-met"
    met.mkdir()
    items = []
    for lead, levels in ((0, 27), (1, 27), (2, 26)):
        receipt = tmp_path / f"endpoint-{lead}" / "init-source-receipt.json"
        receipt.parent.mkdir()
        receipt.write_text(json.dumps({"effective_first_guess": {"nfglevels": levels, "nfgsoillevels": 8}}))
        items.append({"receipt": str(receipt), "met": str(met / f"MET:{lead}")})
    declared = dict(ICON["first_guess"])
    assert _measured_first_guess(met, declared)["nfglevels"] == declared["nfglevels"] == 19
    (met / "source-endpoints.json").write_text(json.dumps({"endpoints": items}))
    assert _measured_first_guess(met, declared)["nfglevels"] == 27
