"""A hex region from an analysis series (ERA5) takes its lateral boundaries from the source's own analyses.

THE BREAKAGE THESE PREVENT (2026-11-13): every hex region from ERA5, for any date and length, was refused by the
country door before its mesh: "era5 declares no forecast cadence for its boundaries". The regional door (drawn and
hrrrv5 regions) refused it the same way. ERA5 has an analysis every hour, so every boundary time is an analysis that
exists. Every test below but the step-0 one fails on woof hex 0.5.0.
"""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex.cli import build_parser
from woof.hex.init_sources import load_table, resolve_source
from woof.hex.regional_door import RegionalRefusal, boundary_plan, resolve_regional_source

CYCLE = datetime(2011, 4, 26, 6)


def row(name):
    return resolve_source(load_table(), name)


@pytest.mark.parametrize("name", ["era5", "era5-cds"])
def test_both_era5_rows_declare_an_hourly_analysis_series(name):
    from woof.hex.lbc_producer import analysis_every_hours

    assert analysis_every_hours(row(name)) == 1


def test_an_era5_region_plans_hourly_boundaries_from_its_own_analyses():
    boundary, issued, offset, step = boundary_plan(row("era5"), CYCLE, 0, 48)
    assert (boundary["id"], issued, offset, step) == ("era5", CYCLE, 0, 1)


def test_a_coarser_step_is_a_whole_multiple_of_the_analysis_step():
    # A step that does not divide the run reads the analysis past its end (test_region_offstep_ends).
    assert boundary_plan(row("era5"), CYCLE, 0, 48, requested=3)[3] == 3
    with pytest.raises(RegionalRefusal, match="whole multiple of 1 h"):
        boundary_plan(row("era5"), CYCLE, 0, 48, requested=0)


def test_a_start_hour_on_a_reanalysis_is_refused_by_name():
    with pytest.raises(RegionalRefusal, match="reanalysis with an analysis every 1 h and no forecast hours"):
        boundary_plan(row("era5"), CYCLE, 6, 12)


def test_a_row_with_no_forecast_and_no_analysis_series_stays_refused():
    # A global row that reads only its step 0 (ICON did, until hex-icon-gefs-regions gave it DWD's forecast hours):
    # its successive analyses are not a boundary series, so it is refused at plan until it declares one.
    import copy

    table = copy.deepcopy(load_table())
    step0 = copy.deepcopy(next(r for r in table["sources"] if r["id"] == "icon"))
    step0.pop("forecast")
    step0.update(id="icon-step0", aliases=[])
    table["sources"].append(step0)
    with pytest.raises(RegionalRefusal, match="icon-step0 declares no forecast cadence"):
        boundary_plan(step0, datetime(2026, 11, 12, 0), 0, 6, source_table=table)


def test_the_country_door_plans_a_48_h_era5_region_with_49_hourly_analyses(tmp_path):
    from woof.hex.country_door import plan

    args = build_parser().parse_args([
        "country", "--point", "38.000,-97.000", "--radius-km", "2500", "--source", "era5", "--cycle", "2011-04-26T06",
        "--hours", "48", "--history-every-minutes", "60", "--work-dir", str(tmp_path / "run"), "--skip", "score",
        "--forecast-arg=--physics-suite=convection_permitting"])
    endpoints = plan(args)["boundary_endpoints"]
    clocks = [(e["argv"][e["argv"].index("--cycle") + 1], e["argv"][e["argv"].index("--lead-hours") + 1])
              for e in endpoints]
    assert len(endpoints) == 49
    assert clocks[0] == ("2011-04-26T06", "0") and clocks[1] == ("2011-04-26T07", "0")
    assert clocks[-1] == ("2011-04-28T06", "0")
    assert endpoints[-1]["valid_time"] == "2011-04-28_06:00:00"


def test_the_regional_door_decodes_each_boundary_hour_at_its_own_valid_time(tmp_path, monkeypatch):
    from woof.hex import regional_door as door

    seen = []

    def run(argv, stdout=None, stderr=None):
        seen.append((argv[argv.index("--cycle") + 1], argv[argv.index("--lead-hours") + 1]))
        out = Path(argv[argv.index("--out-dir") + 1])
        when = datetime.strptime(argv[argv.index("--cycle") + 1], "%Y-%m-%dT%H")
        (out / f"MET:{when:%Y-%m-%d_%H}").write_text("")
        (out / "products").mkdir()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(door.subprocess, "run", run)
    value = {"paths": {"met": str(tmp_path / "met")}, "lead_hours": 0, "hours": 3, "boundary_interval_hours": 1,
             "source": {"id": "era5"}, "cycle": "2011-04-26T06"}
    door._stage_intermediate_from_table(SimpleNamespace(workers=1), value, SimpleNamespace(note=lambda text: None))
    assert sorted(seen) == [("2011-04-26T06", "0"), ("2011-04-26T07", "0"), ("2011-04-26T08", "0"),
                            ("2011-04-26T09", "0")]
    leads = tmp_path / "met" / "leads"
    assert [(leads / f"f{lead:03d}" / "products").is_dir() for lead in range(4)] == [True, False, False, False]


def test_a_boundary_endpoint_keeps_only_its_intermediate_and_receipt(tmp_path):
    # Each hourly ERA5 endpoint kept its whole-globe download (2.2 GB) and composed frames (1.9 GB) beside its 1.0 GB
    # intermediate, so a 48 h region wrote about 245 GB. The start endpoint keeps its inputs (the init reads beside them).
    import hashlib
    import json

    from woof.hex.lbc_producer import produce_source_endpoints, source_endpoint_commands

    commands = source_endpoint_commands(source="era5", cycle="2011-04-26T06", hours=2, out_dir=tmp_path / "src")
    assert [c["boundary_only"] for c in commands] == [False, True, True]

    def runner(argv):
        out = Path(argv[argv.index("--out-dir") + 1])
        when = datetime.strptime(argv[argv.index("--cycle") + 1], "%Y-%m-%dT%H")
        for spent in ("products", "mapped"):
            (out / spent).mkdir(parents=True)
            (out / spent / "big").write_bytes(b"x")
        met = out / f"MET:{when:%Y-%m-%d_%H}"
        met.write_bytes(b"met")
        (out / "init-source-receipt.json").write_text(json.dumps({
            "valid_time": f"{when:%Y-%m-%d_%H}:00:00",
            "decode": {"met": str(met), "met_sha256": hashlib.sha256(b"met").hexdigest()}}))
        return 0

    produce_source_endpoints(commands, tmp_path / "met", workers=1, runner=runner)
    assert (tmp_path / "src/endpoint-0000/products").is_dir() and (tmp_path / "src/endpoint-0000/mapped").is_dir()
    for lead in (1, 2):
        kept = sorted(p.name for p in (tmp_path / f"src/endpoint-{lead:04d}").iterdir())
        assert kept == [f"MET:2011-04-26_{6 + lead:02d}", "init-source-receipt.json"]
        assert (tmp_path / "met" / f"MET:2011-04-26_{6 + lead:02d}").read_bytes() == b"met"


def test_a_forecast_row_still_reads_leads_of_its_one_cycle():
    from woof.hex.lbc_producer import endpoint_clock

    assert endpoint_clock(row("gfs"), CYCLE, 9) == (CYCLE, 9)
    assert endpoint_clock(resolve_regional_source("era5")[1], CYCLE, 9) == (datetime(2011, 4, 26, 15), 0)
