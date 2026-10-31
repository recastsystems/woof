"""Regional orchestration receipts must survive unsuccessful stages."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex import regional_door as door
from woof.hex.cli import build_parser
from woof.hex.errors import MpasPortError


def arguments(tmp_path):
    return build_parser().parse_args([
        "regional", "--point", "36.5,-97.5", "--cycle", "2024-05-06T18",
        "--hours", "3", "--work-dir", str(tmp_path / "run")])


@pytest.mark.parametrize("failure", [None, "returncode", "refusal", "exception"])
def test_receipt_records_stage_outcomes_and_restores_environment(tmp_path, monkeypatch, failure):
    args = arguments(tmp_path)
    monkeypatch.setenv("WOOF_HEX_MESH_ROWS", "previous-rows")
    monkeypatch.setenv("WOOF_HEX_REGIONAL_CONTRACT_DIR", "previous-contracts")
    mesh = {"row": "generated-cull", "grid": "grid.nc", "static": "static.nc"}
    monkeypatch.setattr(door, "_generated", lambda value: {"cull": mesh})
    seen = []

    def runner(name):
        def run(args, value, status):
            live = json.loads((args.work_dir / "regional-receipt.json").read_text())
            assert live["stages"][name]["outcome"] == "running"
            assert os.environ["WOOF_HEX_MESH_ROWS"].endswith(os.pathsep + "previous-rows")
            seen.append(name)
            if name == "products":
                if failure == "refusal":
                    raise MpasPortError("renderer missing")
                if failure == "exception":
                    raise RuntimeError("renderer crashed")
                return {"returncode": 1 if failure else 0,
                        "counts": {"rendered": 4, "skipped": 2, "failed": 1 if failure else 0}}
            return {"returncode": 0}
        return run

    monkeypatch.setattr(door, "_RUNNERS", {name: runner(name) for name in door.STAGES})
    if failure in ("refusal", "exception"):
        with pytest.raises((MpasPortError, RuntimeError)):
            door.run_regional(args)
    else:
        assert door.run_regional(args) == (1 if failure else 0)
    receipt = json.loads((args.work_dir / "regional-receipt.json").read_text())
    assert receipt["mesh"] == mesh
    assert receipt["outcome"] == ("refused" if failure == "refusal" else "failed" if failure else "done")
    assert receipt["seconds"] >= 0 and receipt["finished_utc"] >= receipt["started_utc"]
    assert seen == list(door.STAGES)
    for name, stage in receipt["stages"].items():
        assert stage["outcome"] == ("failed" if failure and name == "products" else "done")
        assert stage["seconds"] >= 0 and stage["finished_utc"] >= stage["started_utc"]
    if failure not in ("refusal", "exception"):
        assert receipt["products"]["counts"]["rendered"] == 4
    assert os.environ["WOOF_HEX_MESH_ROWS"] == "previous-rows"
    assert os.environ["WOOF_HEX_REGIONAL_CONTRACT_DIR"] == "previous-contracts"


def test_hrrr_is_limited_area_and_cannot_initialize_global_mesh():
    from woof.hex.init_sources import load_table, resolve_source

    table = load_table()
    assert resolve_source(table, "hrrr-conus", coverage="limited_area")["intermediate"]["row"] == "hrrr"
    with pytest.raises(MpasPortError, match="covers a limited area"):
        resolve_source(table, "hrrr")
    with pytest.raises(MpasPortError, match="is a global source"):
        resolve_source(table, "gfs", coverage="limited_area")


def _gfs_region(tmp_path, cycle, lead, hours, *extra, source="gfs"):
    return door.plan(build_parser().parse_args([
        "regional", "--point=-5.5,18.5", "--cycle", cycle, "--source", source, "--lead-hours", str(lead),
        "--hours", str(hours), "--fine-dx-m", "3000", "--radius-km", "600", "--background-km", "48",
        "--work-dir", str(tmp_path / "run"), *extra]))


def test_gfs_region_starts_at_any_lead_gfs_writes(tmp_path):
    """GFS writes every hour to 120 h (its row's ladder). A region from GFS 18Z +3 h was refused because the plan
    read only the row's 6 h interval."""
    value = _gfs_region(tmp_path, "2026-10-27T18", 3, 12)
    assert value["start_time"] == "2026-10-27_21:00:00"
    assert value["boundary_interval_hours"] == 1
    # past 120 h GFS writes every 3 h: 123 h, 126 h and 129 h are leads, 121 h is not
    assert _gfs_region(tmp_path, "2026-10-27T18", 123, 6)["boundary_interval_hours"] == 3
    with pytest.raises(door.RegionalRefusal, match="writes no lead"):
        _gfs_region(tmp_path, "2026-10-27T18", 121, 6)
    with pytest.raises(door.RegionalRefusal, match="writes no lead"):
        _gfs_region(tmp_path, "2026-10-27T18", 380, 6)


def test_table_regions_take_the_finest_step_the_source_posts(tmp_path):
    """The HRDPS hex sheets (2026-10-29) put "hex from GFS 0.25 deg (6-hourly boundaries)" beside "HRDPS (hourly
    boundaries)": the gfs, ifs, ifs-0p1, ifs-ens and gefs rows all declare interval_hours 6, and the door took that as
    the step although GFS posts hourly to 120 h and IFS and GEFS every 3 h. A 6 h region from GFS got only its start
    and end boundaries."""
    assert _gfs_region(tmp_path, "2026-10-07T18", 0, 6)["boundary_interval_hours"] == 1
    assert _gfs_region(tmp_path, "2026-10-07T18", 0, 7)["boundary_interval_hours"] == 1
    for source in ("ifs", "gefs"):
        assert _gfs_region(tmp_path, "2026-10-07T18", 0, 6, source=source)["boundary_interval_hours"] == 3, source
    # IFS 0.1 degree starts at its first preparable cycle (2026-10-20 12Z, fix/hex-ifs-0p1-first-cycle).
    assert _gfs_region(tmp_path, "2026-10-21T00", 0, 6, source="ifs-0p1")["boundary_interval_hours"] == 3
    # GFS past 120 h and IFS past 144 h (00/12 UTC) post more coarsely; a window across the change takes the
    # finest step posted through all of it.
    assert _gfs_region(tmp_path, "2026-10-07T18", 114, 12)["boundary_interval_hours"] == 3
    assert _gfs_region(tmp_path, "2026-10-07T12", 138, 12, source="ifs")["boundary_interval_hours"] == 6


def test_boundary_interval_knob_picks_a_posted_coarser_step_or_is_refused_by_name(tmp_path):
    assert _gfs_region(tmp_path, "2026-10-07T18", 0, 12, "--boundary-interval-hours", "6")[
        "boundary_interval_hours"] == 6
    assert _gfs_region(tmp_path, "2026-10-07T18", 3, 12, "--boundary-interval-hours", "3")[
        "boundary_interval_hours"] == 3
    with pytest.raises(door.RegionalRefusal, match="--boundary-interval-hours 4"):
        _gfs_region(tmp_path, "2026-10-07T18", 0, 6, "--boundary-interval-hours", "4")
    with pytest.raises(door.RegionalRefusal, match="--boundary-interval-hours 1"):
        _gfs_region(tmp_path, "2026-10-07T18", 0, 6, "--boundary-interval-hours", "1", source="ifs")
    with pytest.raises(door.RegionalRefusal, match="would be ignored"):
        door.plan(build_parser().parse_args([
            "regional", "--point", "33.5,-101.0", "--source", "hrrr", "--cycle", "2026-10-23T18", "--hours", "3",
            "--boundary-interval-hours", "1", "--work-dir", str(tmp_path / "run")]))


def _hrrrv5_plan(tmp_path, source="gfs"):
    args = build_parser().parse_args([
        "regional", "--point", "45.75,-75.75", "--cycle", "2026-10-07T18", "--hours", "6",
        "--source", source, "--work-dir", str(tmp_path / "run"), "--forecast-arg=--physics-suite=hrrrv5"])
    return args, door.plan(args)


def test_a_producer_suite_takes_its_producers_boundaries_and_the_measured_levels(tmp_path, monkeypatch):
    """hrrrv5 through `woof hex regional`: the boundaries were rw_mpas_lbc's v8.4.2 series on the row's level
    count, for an init on the RRFS fork's L60 grid; the counts the intermediate door measured now win."""

    args, value = _hrrrv5_plan(tmp_path)
    met = tmp_path / "run" / "met"
    met.mkdir(parents=True)
    (met / "intermediate-receipt.json").write_text(json.dumps(
        {"files": [], "levels": {"count": 39, "init_switches": {"--nfgsoillevels": 4}}}))
    lbc_dir = tmp_path / "run" / "mesh" / "lbc"
    monkeypatch.setattr(door, "_generated", lambda value: {"cull": {"lbc_dir": str(lbc_dir)}})
    seen = []
    monkeypatch.setattr(door, "_cli", lambda argv, status: seen.append(argv) or 0)
    assert door._first_guess(value)["nfglevels"] == 39
    assert door._producer_suite(value) == "hrrrv5"
    door._stage_lbc(args, value, status=None)
    argv = seen[0]
    assert argv[argv.index("--nfglevels") + 1] == "39"
    assert argv[argv.index("--physics-suite") + 1] == "hrrrv5"


def test_an_mpas_a_suite_keeps_the_v842_boundaries(tmp_path, monkeypatch):
    args = build_parser().parse_args([
        "regional", "--point", "45.75,-75.75", "--cycle", "2026-10-07T18", "--hours", "6",
        "--source", "gfs", "--work-dir", str(tmp_path / "run")])
    value = door.plan(args)
    monkeypatch.setattr(door, "_generated", lambda value: {"cull": {"lbc_dir": str(tmp_path / "lbc")}})
    seen = []
    monkeypatch.setattr(door, "_cli", lambda argv, status: seen.append(argv) or 0)
    assert door._producer_suite(value) is None
    door._stage_lbc(args, value, status=None)
    assert "--physics-suite" not in seen[0]
    # No intermediate receipt yet: the row's own count.
    assert seen[0][seen[0].index("--nfglevels") + 1] == "34"


def test_the_lbc_door_hands_a_producer_suite_to_its_case_9(tmp_path, monkeypatch):
    from datetime import datetime

    from woof.hex import hrrr_intermediate as hi
    from woof.hex.mpasphys.rrfs import cold_start_rrfs as cs

    init = tmp_path / "init.nc"
    init.write_bytes(b"\0")
    mets = [tmp_path / f"MET_{hour}" for hour in (18, 19)]
    for path in mets:
        path.write_bytes(b"\0")
    monkeypatch.setattr(hi, "intermediate_valid_times", lambda met_dir: [
        (datetime(2026, 10, 7, 18), mets[0]), (datetime(2026, 10, 7, 19), mets[1])])
    called = {}

    def fake(producer, **kwargs):
        called.update(kwargs, producer=producer[0])
        out = kwargs["out_dir"]
        out.mkdir(parents=True, exist_ok=True)
        files = []
        for stamp in ("2026-10-07_18.00.00", "2026-10-07_19.00.00"):
            (out / f"lbc.{stamp}.nc").write_bytes(b"\0")
            files.append(str(out / f"lbc.{stamp}.nc"))
        return {"producer": {"key": producer[0]}, "seconds": 0.0, "files": files, "stages": []}

    monkeypatch.setattr(cs, "run_rrfs_lbc", fake)
    args = build_parser().parse_args([
        "lbc", "--grid", str(init), "--met-dir", str(tmp_path), "--out-dir", str(tmp_path / "lbc"),
        "--start-time", "2026-10-07_18:00:00", "--stop-time", "2026-10-07_19:00:00", "--nfglevels", "39",
        "--extrap-airtemp", "lapse-rate", "--use-spechumd", "yes", "--physics-suite", "hrrrv5"])
    assert hi.run_lbc(args) == 0
    assert called["producer"].startswith("rrfs-mpas")
    assert called["nfglevels"] == 39 and called["init"] == init
    assert [when for when, _ in called["intervals"]] == ["2026-10-07_18:00:00", "2026-10-07_19:00:00"]
    receipt = json.loads((tmp_path / "lbc" / "lbc-door-receipt.json").read_text())
    assert receipt["route"] == "init-producer case 9" and len(receipt["files"]) == 2


def test_a_gsl_drag_suite_gets_its_orographic_statistics_before_the_card(tmp_path, monkeypatch):
    """hrrrv5 from HRDPS (Ottawa 2026-10-07 18Z, 2026-10-29): the producer makes ugwp_oro_data.nc only with GSL's
    topo_ugwp geography; without it the door passed nothing and the forecast died after the contract.  The
    producer's file wins; otherwise hex's own method builds it beside the cull's grid at the init stage."""
    import woof.hex.ugwp_oro as ugwp_oro

    args, value = _hrrrv5_plan(tmp_path)
    grid = tmp_path / "run" / "mesh" / "cull.grid.nc"
    monkeypatch.setattr(door, "_generated", lambda value: {"cull": {"row": "cull-row", "grid": str(grid)}})
    built = []
    monkeypatch.setattr(ugwp_oro, "ensure", lambda g, out, geog=None, log=None: built.append((g, out)) or
                        {"source": "built", "out": str(out)})
    status = SimpleNamespace(note=lambda text: None)
    assert door._orographic_statistics(args, value, status) == ugwp_oro.default_path(grid, "cull-row")
    assert built == [(str(grid), ugwp_oro.default_path(grid, "cull-row"))]
    made = Path(value["paths"]["init"]).with_name("ugwp_oro_data.nc")
    made.parent.mkdir(parents=True, exist_ok=True)
    made.write_bytes(b"")
    assert door._orographic_statistics(args, value, status) == made and len(built) == 1


def test_a_suite_without_the_gsl_drag_names_no_orographic_statistics(tmp_path, monkeypatch):
    args = build_parser().parse_args([
        "regional", "--point", "45.75,-75.75", "--cycle", "2026-10-07T18", "--hours", "6",
        "--source", "gfs", "--work-dir", str(tmp_path / "run"),
        "--forecast-arg=--physics-suite=convection_permitting"])
    value = door.plan(args)
    monkeypatch.setattr(door, "_generated", lambda value: pytest.fail("no statistics are needed"))
    assert door._orographic_statistics(args, value, status=None) is None


def test_the_region_is_drawn_while_its_forecast_runs_and_the_full_pass_keeps_those_hours(tmp_path, monkeypatch):
    """The independent check of the map-speed fix (2026-10-30): `woof hex regional` rendered its maps only in its
    products stage after the forecast, so a region run showed no map until its whole forecast had run. A first look
    now follows the forecast (one frame per draw, several at once) into the case the full pass names, and the full
    pass takes its receipt."""
    args = arguments(tmp_path)
    mesh = {"row": "generated-cull", "grid": "grid.nc", "static": "static.nc"}
    monkeypatch.setattr(door, "_generated", lambda value: {"cull": mesh})
    looks = []

    class Look:
        def __init__(self, forecast, **kwargs):
            self.forecast, self.kwargs, self.started, self.finished = forecast, kwargs, False, False
            looks.append(self)

        def start(self):
            self.started = True

        def finish(self):
            self.finished = True
            return {"status": "drawn", "live": {"drawn": 3}}

        def receipt(self):
            return Path(self.forecast).parent / "live-frames.json"

    import woof.hex.first_look as first_look
    monkeypatch.setattr(first_look, "FirstLook", Look)
    seen = {}

    def cli(argv, status):
        seen["products"] = [str(a) for a in argv]
        case = Path(argv[argv.index("--out") + 1]) / argv[argv.index("--case") + 1]
        case.mkdir(parents=True)
        (case / "products-manifest.json").write_text(json.dumps({"views": []}))
        return 0

    def runner(name):
        def run(args, value, status):
            if name == "forecast":
                assert looks and looks[0].started and not looks[0].finished   # drawing beside the forecast
            return {"returncode": 0}
        return run

    monkeypatch.setattr(door, "_cli", cli)
    monkeypatch.setattr(door, "_RUNNERS", {**{name: runner(name) for name in door.STAGES},
                                           "products": door._stage_products})
    assert door.run_regional(args) == 0
    (look,) = looks
    assert look.finished and look.kwargs["case"] == "generated-cull-2024050618"
    assert Path(look.kwargs["out"]) == Path(look.forecast) / "products"
    products = seen["products"]
    assert products[products.index("--case") + 1] == "generated-cull-2024050618"   # the case the live draws used
    assert products[products.index("--live-frames") + 1] == str(Path(look.forecast).parent / "live-frames.json")
    receipt = json.loads((args.work_dir / "regional-receipt.json").read_text())
    assert receipt["stages"]["forecast"]["first_look"]["status"] == "drawn"
