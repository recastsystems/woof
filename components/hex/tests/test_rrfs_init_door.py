"""The init door makes RRFS-MPAS initial conditions from its producer row.

RETIRES (2026-10-28): "hex does not make RRFS initial conditions in this release". The
numbers are the Rust producer's, proven against the fork's strict native init_atmosphere
on real inputs (tests/rrfs_init_door_replay.py, docs/rrfs-init.md); these tests hold the
door's wiring: suite -> producer row -> the exact hrrrv5 job recipe -> the worker.
"""
from __future__ import annotations

import json
import stat
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from woof.hex.mpasphys.rrfs import cold_start_rrfs as cs


def _row():
    key, row = cs.producer_for_suite("hrrrv5")
    return row


def test_every_rrfs_suite_route_has_the_fork_producer():
    from woof.hex.physics_suites import load_table

    rrfs = load_table()["mpasphys_rrfs"]
    for suite in rrfs["suite_routes"]:
        key, row = cs.producer_for_suite(suite, pin_set="rrfs")
        assert row["source_pin"] == rrfs["model_commit"]
        assert row["physics_table_key"] == "mpasphys_rrfs"
    # No pin named: each alias takes the producer of its default route (44c0849 for hrrrv5 only).
    new = load_table()["mpasphys_rrfs_44c0849"]
    for suite in rrfs["suite_routes"]:
        key, row = cs.producer_for_suite(suite)
        expected = new if suite == "hrrrv5" else rrfs
        assert row["source_pin"] == expected["model_commit"], suite
    assert cs.producer_for_suite("convection_permitting") is None
    assert cs.producer_for_suite(None) is None


def test_init_selection_routes_an_rrfs_suite_to_its_producer():
    from woof.hex.init_physics import init_selection

    selection = init_selection("hrrrv5", None)
    assert selection["land_surface"] == "ruc" and selection["scalar_names"] == ()
    assert selection["producer"][1]["source_pin"].startswith("44c0849")
    assert selection["physics_initialization"]["suite"] == "hrrrv5"


def test_the_l60_levels_text_is_the_workflow_file():
    text = cs.specified_levels_text(_row()).splitlines()
    assert len(text) == 60 and text[0] == "0" and text[1] == "17.9503453" and text[-1] == "25878.712"


def test_case7_jobs_carry_the_hrrrv5_namelist(tmp_path):
    row = _row()
    jobs = cs.case7_jobs(row, static=tmp_path / "static.nc", met=tmp_path / "GFS:2023-07-08_12",
                         start_time="2023-07-08_12:00:00", nfglevels=34, nfgsoillevels=4,
                         artifact=tmp_path / "a.nc", out=tmp_path / "init.nc", levels_file=tmp_path / "Z",
                         aerosols=tmp_path / "Q.dat", version="t")
    prepare, case7 = (j["config"] for j in jobs)
    assert [j["stage"] for j in jobs] == ["prepare", "case7"]
    vertical = prepare["vertical"]
    assert (vertical["levels"], vertical["top"], vertical["scheme"]) == (59, 25878.712, "tc")
    assert (vertical["terrain_passes"], vertical["surface_passes"], vertical["minimum_fraction"]) == (1, 30, 0.3)
    assert vertical["regional_halo"] is False and prepare["specified_levels_path"] == str(tmp_path / "Z")
    # 44c0849 init Registry: defc_a and defc_b retired, tke added (157 on 1d6e370e).
    assert prepare["soil_levels"] == 9 and len(prepare["variables"]) == 156
    times = {v["name"]: v["text"] for v in prepare["variables"] if v["name"] in ("initial_time", "xtime")}
    assert times == {"initial_time": "2023-07-08_12:00:00", "xtime": "2023-07-08_12:00:00"}
    assert case7["n_soil_layers"] == 9 and case7["nvegopt"] == 2 and case7["n_fg_soil_levels"] == 4
    assert case7["landuse_table"] == "MODIFIED_IGBP_MODIS_NOAH_15s" and case7["tempo_rap"] is False
    assert case7["tsk_seaice_threshold"] == 271.4 and case7["frac_seaice"] is True
    assert case7["capsule_path"] == case7["reference_path"] == str(tmp_path / "a.nc")


def test_a_limited_area_mesh_takes_the_levels_inline_with_the_halo(tmp_path):
    jobs = cs.case7_jobs(_row(), static=tmp_path / "s", met=tmp_path / "m", start_time="2023-07-08_12:00:00",
                         nfglevels=34, nfgsoillevels=4, artifact=tmp_path / "a", out=tmp_path / "o",
                         levels_file=tmp_path / "Z", aerosols=tmp_path / "q", regional=True)
    prepare = jobs[0]["config"]
    assert prepare["vertical"]["scheme"] == "specified" and prepare["vertical"]["regional_halo"] is True
    assert len(prepare["vertical"]["specified"]) == 60 and "specified_levels_path" not in prepare


def test_case9_jobs_make_the_carrier_from_the_init(tmp_path):
    jobs = cs.case9_jobs(_row(), init=tmp_path / "init.nc", start_time="2023-07-08_12:00:00",
                         intervals=[("2023-07-08_13:00:00", tmp_path / "m13"), ("2023-07-08_14:00:00", tmp_path / "m14")],
                         nfglevels=34, out_dir=tmp_path / "lbc", carrier=tmp_path / "c.nc", aerosols=tmp_path / "q")
    artifact, case9 = (j["config"] for j in jobs)
    assert artifact["template"] == str(tmp_path / "init.nc") and artifact["dimensions"]["nSoilLevels"] == 9
    assert case9["hydrometeors_rrfs"] is True and case9["hydrometeors_gfs"] is False
    assert (case9["start_time"], case9["stop_time"]) == ("2023-07-08_13:00:00", "2023-07-08_14:00:00")


FAKE_WORKER = """#!{python}
import json, sys, pathlib
ABI = {abi!r}
if sys.argv[1:] == ["--abi"]:
    print(ABI); sys.exit(0)
job = json.load(sys.stdin)
log = pathlib.Path({log!r})
with log.open("a") as f:
    f.write(json.dumps(job) + "\\n")
cfg = job["config"]
for key in ("output_path", "out_path"):
    if key in cfg:
        pathlib.Path(cfg[key]).write_bytes(b"x")
print(json.dumps({{"stage": job["stage"]}}))
"""


def _door(tmp_path, monkeypatch, *extra):
    from woof.hex import init_door
    import argparse

    worker = tmp_path / "rw_mpas_init_v831"
    worker.write_text(FAKE_WORKER.format(python=sys.executable, abi=_row()["worker"]["abi"],
                                         log=str(tmp_path / "jobs.jsonl")))
    worker.chmod(worker.stat().st_mode | stat.S_IXUSR)
    met = tmp_path / "GFS:2023-07-08_12"
    met.write_bytes(b"\0" * 8)
    (tmp_path / "static.nc").write_bytes(b"s")
    monkeypatch.setattr(cs, "static_missing_fields", lambda static, row: [])
    monkeypatch.setattr(cs, "mesh_is_regional", lambda path: False)
    monkeypatch.setattr(cs, "resolve_aerosol_table", lambda explicit: tmp_path / "Q.dat")
    (tmp_path / "Q.dat").write_bytes(b"q")
    monkeypatch.setattr(init_door, "resolve_met_source", lambda path: Path(path))
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()
    init_door.add_init_parser(commands)
    arguments = parser.parse_args([
        "init", "--physics-suite", "hrrrv5", "--met", str(met), "--static", str(tmp_path / "static.nc"),
        "--out", str(tmp_path / "init.nc"), "--start-time", "2023-07-08_12:00:00", "--nfglevels", "34",
        "--nfgsoillevels", "4", "--rrfs-worker", str(worker), *extra])
    return arguments


def test_the_door_runs_the_producer_and_writes_its_receipt(tmp_path, monkeypatch):
    from woof.hex.init_door import run_init_command

    arguments = _door(tmp_path, monkeypatch)
    assert run_init_command(arguments) == 0
    jobs = [json.loads(line) for line in (tmp_path / "jobs.jsonl").read_text().splitlines()]
    assert [j["stage"] for j in jobs] == ["prepare", "case7"]
    assert jobs[1]["config"]["met_path"].endswith("GFS:2023-07-08_12")
    receipt = json.loads((tmp_path / "init.nc.provenance.json").read_text())
    assert receipt["producer"]["source_pin"].startswith("44c0849")
    assert [s["stage"] for s in receipt["stages"]] == ["prepare", "case7"]
    assert receipt["ugwp_oro"]["made_here"] is False


def test_the_door_writes_boundaries_when_asked(tmp_path, monkeypatch):
    from woof.hex.init_door import run_init_command

    m13 = tmp_path / "GFS:2023-07-08_13"
    m13.write_bytes(b"\0")
    arguments = _door(tmp_path, monkeypatch, "--lbc-met", f"2023-07-08_13:00:00={m13}")
    assert run_init_command(arguments) == 0
    stages = [json.loads(line)["stage"] for line in (tmp_path / "jobs.jsonl").read_text().splitlines()]
    assert stages == ["prepare", "case7", "artifact", "case9"]


def test_a_static_without_the_fork_fields_and_no_geography_is_refused_by_name(tmp_path, monkeypatch):
    from woof.hex.init_door import InitDoorRefusal, run_init_command

    arguments = _door(tmp_path, monkeypatch)
    monkeypatch.setattr(cs, "static_missing_fields", lambda static, row: ["landusef", "soilf", "lai12m"])
    with pytest.raises(InitDoorRefusal, match="landusef, soilf, lai12m.*--geog"):
        run_init_command(arguments)


def test_a_worker_with_another_abi_is_refused(tmp_path, monkeypatch):
    from woof.hex.init_door import InitDoorRefusal, run_init_command

    arguments = _door(tmp_path, monkeypatch)
    text = Path(arguments.rrfs_worker).read_text().replace("hex-init-job/v1", "hex-init-job/v0", 1)
    Path(arguments.rrfs_worker).write_text(text)
    with pytest.raises(InitDoorRefusal, match="another contract"):
        run_init_command(arguments)


def test_the_cold_start_reads_exactly_one_initial_stream():
    assert cs.select_initial_state_stream({"config_do_restart": False, "config_do_DAcycling": False}) == "input"
    assert cs.select_initial_state_stream({"config_do_restart": False, "config_do_DAcycling": True}) == "da_state"
    assert cs.select_initial_state_stream({"config_do_restart": True, "config_do_DAcycling": True}) == "restart"


def test_an_init_whose_land_soil_the_producer_left_at_zero_is_refused_and_removed(tmp_path):
    """hrrrv5 from ICON (2026-10-29): ICON's eight soil levels take neither branch of the fork's RUC soil recipe
    (Noah's four, or RUC's own nine nodes), and every land cell started at 0 K, rc 0."""
    import numpy as np
    from netCDF4 import Dataset

    out = tmp_path / "init.nc"
    with Dataset(out, "w") as d:
        d.createDimension("Time", None); d.createDimension("nCells", 3); d.createDimension("nSoilLevels", 9)
        d.createVariable("tslb", "f4", ("Time", "nCells", "nSoilLevels"))[0] = np.array(
            [[0.0] * 9, [285.0] * 9, [0.0] * 9])
        d.createVariable("landmask", "i4", ("nCells",))[:] = [1, 1, 0]
    with pytest.raises(cs.RrfsInitError, match="1 of 2 land cells with a soil temperature of 0 K.*8 soil levels"):
        cs._refuse_cold_land_soil(out, _row(), 8)
    assert not out.exists()
    with Dataset(tmp_path / "ok.nc", "w") as d:
        d.createDimension("Time", None); d.createDimension("nCells", 2); d.createDimension("nSoilLevels", 9)
        d.createVariable("tslb", "f4", ("Time", "nCells", "nSoilLevels"))[0] = np.full((2, 9), 280.0)
        d.createVariable("landmask", "i4", ("nCells",))[:] = [1, 0]
    cs._refuse_cold_land_soil(tmp_path / "ok.nc", _row(), 4)


def test_the_mpas_route_finds_or_builds_the_gsl_drag_statistics_beside_the_grid(tmp_path, monkeypatch):
    """hrrrv5 on the MPAS route with no --orographic-statistics reached the fork's init on Registry zeros and died
    after the contract (2026-10-29).  The route now takes <mesh>.ugwp_oro_data.nc beside --grid, built there when
    absent, as the column route does; a deck's streams file or a drag that reads no statistics is unchanged."""
    from types import SimpleNamespace

    import woof.hex.ugwp_oro as ugwp_oro
    from woof.hex.forecast_door import mpas_route_orographic_statistics

    grid = tmp_path / "mesh" / "cull.grid.nc"
    built = []
    monkeypatch.setattr(ugwp_oro, "ensure", lambda g, out, geog=None, log=None: built.append(out) or {})
    ugwp = SimpleNamespace(execution_options={"config_gwdo_scheme": "bl_ugwp_gwdo"})
    run = SimpleNamespace(grid=grid, mesh="cull-row", streams=None, physics_namelist=None)
    assert mpas_route_orographic_statistics(run, ugwp) == grid.parent / "cull-row.ugwp_oro_data.nc"
    assert built == [grid.parent / "cull-row.ugwp_oro_data.nc"]
    ysu = SimpleNamespace(execution_options={"config_gwdo_scheme": "bl_ysu_gwdo"})
    assert mpas_route_orographic_statistics(run, ysu) is None
    deck = SimpleNamespace(grid=grid, mesh="cull-row", streams=tmp_path / "streams.atmosphere", physics_namelist=None)
    assert mpas_route_orographic_statistics(deck, ugwp) is None and len(built) == 1
