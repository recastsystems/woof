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
    assert case7["landuse_table"] == "MODIFIED_IGBP_MODIS_NOAH_15s" and case7["tempo_rap"] is True
    assert case7["tsk_seaice_threshold"] == 271.4 and case7["frac_seaice"] is True
    assert case7["capsule_path"] == case7["reference_path"] == str(tmp_path / "a.nc")
    assert case7["absent_numbers"] == "from-mass"


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
    # Only this switch writes the number fields, and from-mass fills them: NOAA's GFS switch (gfs true, rrfs
    # false) leaves the model zero rain, cloud and ice number, an echo ring on every edge (docs/rrfs-init.md).
    assert case9["absent_numbers"] == "from-mass"
    assert case9["tempo_rap"] is True
    assert (case9["start_time"], case9["stop_time"]) == ("2023-07-08_13:00:00", "2023-07-08_14:00:00")


def test_source_first_aerosols_are_enabled_on_every_producer():
    for row in cs.load_producers()["producers"].values():
        if row["aerosols"]["rule"] == "source-first":
            assert row["case7"]["tempo_rap"] is True


def test_aerosol_receipt_distinguishes_native_climatology_and_unknown():
    job = {"stage": "case7", "config": {"tempo_rap": True, "met_path": "MET:time"}}
    native = {"aerosols": {"source": "", "source_present": False, "records_read": 0},
              "emit": {"computed": ["nwfa", "nifa"]}}
    assert cs.aerosol_selection(job, native)["selected"] == "native"
    assert cs.aerosol_selection(job, native)["source"] == "MET:time"
    monthly = {"aerosols": {"source": "Q.dat", "source_present": True, "records_read": 1080}}
    assert cs.aerosol_selection(job, monthly)["selected"] == "climatology"
    assert cs.aerosol_selection(job, monthly)["source"] == "Q.dat"
    assert cs.aerosol_selection(job, {"aerosols": {}})["selected"] == "unknown"
    assert cs.aerosol_selection(job, {}) is None
    assert cs.aerosol_selection({"stage": "prepare"}, native) is None


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
    # The door now inspects soil records instead of trusting a stale count.
    import numpy as np
    from test_soil_handover import write_met
    write_met(met, np.full((4, 2, 2), 280., np.float32),
              np.full((4, 2, 2), .2, np.float32), nodes=False, source="NCEP GFS")
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
        "--nfgsoillevels", "4", "--rrfs-worker", str(worker),
        # The fake producer writes a one-byte placeholder, not a state file.
        "--condensate-balance", "no", *extra])
    return arguments


def test_the_door_runs_the_producer_and_writes_its_receipt(tmp_path, monkeypatch):
    from woof.hex.init_door import run_init_command

    arguments = _door(tmp_path, monkeypatch)
    assert run_init_command(arguments) == 0
    jobs = [json.loads(line) for line in (tmp_path / "jobs.jsonl").read_text().splitlines()]
    assert [j["stage"] for j in jobs] == ["prepare", "case7"]
    # GFS carries four Noah layers and no RUC nodes: the fork's own interpolation builds the nodes.
    assert jobs[1]["config"]["met_path"].endswith("GFS:2023-07-08_12")
    assert jobs[1]["config"]["n_fg_soil_levels"] == 4
    receipt = json.loads((tmp_path / "init.nc.provenance.json").read_text())
    assert receipt["producer"]["source_pin"].startswith("44c0849")
    assert receipt["soil_handover"]["path"] == "fork-interpolation"
    assert receipt["soil_handover"]["applied"] is False
    assert receipt["soil_handover"]["requested_nfgsoillevels"] == 4
    assert [s["stage"] for s in receipt["stages"]] == ["prepare", "case7"]
    assert receipt["ugwp_oro"]["made_here"] is False


def test_the_rrfs_door_completes_loaded_balance_before_final_digest(tmp_path, monkeypatch):
    from woof.hex.init_door import run_init_command
    from woof.hex import init_balance

    arguments = _door(tmp_path, monkeypatch, "--condensate-balance", "yes")
    calls = []

    def complete(path, *, enabled):
        calls.append((path, enabled))
        path.write_bytes(b"completed loaded state")
        return {"enabled": enabled, "columns": 2}

    monkeypatch.setattr(init_balance, "complete_init", complete)
    assert run_init_command(arguments) == 0
    assert calls == [(tmp_path / "init.nc", True)]
    receipt = json.loads((tmp_path / "init.nc.provenance.json").read_text())
    assert receipt["condensate_balance"] == {"enabled": True, "columns": 2, "chosen_by": "--condensate-balance"}
    assert receipt["output"]["sha256"] == cs._sha256(tmp_path / "init.nc")


def test_the_door_writes_boundaries_when_asked(tmp_path, monkeypatch):
    from woof.hex.init_door import run_init_command

    m13 = tmp_path / "GFS:2023-07-08_13"
    m13.write_bytes(b"\0")
    arguments = _door(tmp_path, monkeypatch, "--lbc-met", f"2023-07-08_13:00:00={m13}")
    assert run_init_command(arguments) == 0
    stages = [json.loads(line)["stage"] for line in (tmp_path / "jobs.jsonl").read_text().splitlines()]
    assert stages == ["prepare", "case7", "artifact", "case9"]


def test_a_recipe_switch_the_producer_does_not_use_is_named_and_recorded(tmp_path, monkeypatch, capsys):
    # model audit D22: Oklahoma's 6 km start passed these three and the producer ran its row's values, rc 0, no word
    from woof.hex.init_door import run_init_command

    arguments = _door(tmp_path, monkeypatch, "--extrap-airtemp", "constant", "--tsk-seaice-threshold", "100.0",
                      "--landuse-table", "MODIFIED_IGBP_MODIS_NOAH")
    assert run_init_command(arguments) == 0
    text = capsys.readouterr().err
    for word in ("these switches are NOT used",
                 "--extrap-airtemp constant (the recipe runs 'lapse-rate')",
                 "--tsk-seaice-threshold 100.0 (the recipe runs 271.4)",
                 "--landuse-table MODIFIED_IGBP_MODIS_NOAH (the recipe runs 'MODIFIED_IGBP_MODIS_NOAH_15s')"):
        assert word in text
    assert text.count("these switches are NOT used") == 1
    receipt = json.loads((tmp_path / "init.nc.provenance.json").read_text())
    assert receipt["ignored_switches"] == {
        "--extrap-airtemp": {"given": "constant", "recipe": "lapse-rate"},
        "--landuse-table": {"given": "MODIFIED_IGBP_MODIS_NOAH", "recipe": "MODIFIED_IGBP_MODIS_NOAH_15s"},
        "--tsk-seaice-threshold": {"given": 100.0, "recipe": 271.4}}
    case7 = json.loads((tmp_path / "jobs.jsonl").read_text().splitlines()[1])["config"]
    assert (case7["extrap_airtemp"], case7["tsk_seaice_threshold"]) == ("lapse-rate", 271.4)


def test_recipe_switches_equal_to_the_row_run_and_are_recorded(tmp_path, monkeypatch):
    from woof.hex.init_door import run_init_command

    arguments = _door(tmp_path, monkeypatch, "--extrap-airtemp", "lapse-rate", "--tsk-seaice-threshold", "271.4",
                      "--use-spechumd", "yes", "--frac-seaice", "yes", "--theta-adv-order", "3",
                      "--coef-3rd-order", "0.25", "--deep-soil-moisture", "reproduce-fortran")
    assert run_init_command(arguments) == 0
    receipt = json.loads((tmp_path / "init.nc.provenance.json").read_text())
    assert receipt["given_switches"] == {"--extrap-airtemp": "lapse-rate", "--use-spechumd": "yes",
                                         "--theta-adv-order": 3, "--coef-3rd-order": 0.25,
                                         "--deep-soil-moisture": "reproduce-fortran", "--frac-seaice": "yes",
                                         "--tsk-seaice-threshold": 271.4}
    assert receipt["ignored_switches"] == {}
    plain = tmp_path / "plain"
    plain.mkdir()
    assert run_init_command(_door(plain, monkeypatch)) == 0
    assert json.loads((plain / "init.nc.provenance.json").read_text())["given_switches"] == {}


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


def _wps_v5(path, fields, map_source="DWD Offenbach"):
    """A big-endian WPS v5 intermediate on a 3 x 1 lat-lon box: ``fields`` is [(name, units, values)]."""
    import struct

    def record(payload):
        return struct.pack(">i", len(payload)) + payload + struct.pack(">i", len(payload))

    with open(path, "wb") as handle:
        for name, units, values in fields:
            meta = ("2026-11-11_00:00:00".ljust(24).encode() + struct.pack(">f", 0.0)
                    + map_source.ljust(32).encode() + name.ljust(9).encode() + units.ljust(25).encode()
                    + "a test field".ljust(46).encode() + struct.pack(">fiii", 200100.0, 3, 1, 0))
            projection = b"SWCORNER".ljust(8) + struct.pack(">fffff", 50.0, 9.0, 0.25, 0.25, 6371.229)
            for payload in (struct.pack(">i", 5), meta, projection, struct.pack(">i", 0),
                            struct.pack(">3f", *values)):
                handle.write(record(payload))
    return path


#: ICON's eight TERRA soil levels as its Vtable names them (nodes 0.5 cm .. 14.58 m).
_ICON_SOIL = ("001", "002", "006", "018", "054", "162", "486", "999")


def test_a_first_guess_soil_the_fork_reads_is_passed_untouched(tmp_path):
    """GFS and IFS hand the fork four layers and HRRR nine nodes: their first guess is the case-7 input as it was."""
    met = _wps_v5(tmp_path / "MET:2026-11-11_00", [("ST000010", "K", (280.0, 281.0, 282.0))])
    for count in (4, 9):
        assert cs.fork_soil_first_guess(_row(), met, count, tmp_path / "work") == (met, count, None)
    assert not (tmp_path / "work").exists()


def test_icons_eight_soil_levels_are_moved_onto_noahs_four_layers(tmp_path):
    """hrrrv5 from ICON (2026-10-29): ICON's eight levels take neither branch of the fork's RUC soil recipe (Noah's
    four, or RUC's own nine nodes) and their names are not RUC's, so case 7 read no soil and every land cell
    started at 0 K, rc 0. The copy case 7 now reads carries Noah's four layers interpolated from ICON's column at
    each layer's mid-depth; the sea point ICON leaves at 0 keeps it (case 7's land-sea mask fills it), and every
    other field is the source's own bytes."""
    import numpy as np

    depths = np.array([1.0, 2.0, 6.0, 18.0, 54.0, 162.0, 486.0, 999.0])
    temperature = {name: (270.0 + depth / 10.0, 280.0 + depth / 10.0, 0.0) for name, depth in zip(_ICON_SOIL, depths)}
    moisture = {name: (0.2 + depth / 1000.0, 0.0, 0.0) for name, depth in zip(_ICON_SOIL, depths)}
    fields = [("TT", "K", (290.0, 291.0, 292.0)),
              *[(f"SOILT{name}", "K", temperature[name]) for name in _ICON_SOIL],
              *[(f"SOILM{name}", "fraction", moisture[name]) for name in _ICON_SOIL],
              ("LANDSEA", "proprtn", (1.0, 1.0, 0.0))]
    met = _wps_v5(tmp_path / "MET:2026-11-11_00", fields)
    out, count, note = cs.fork_soil_first_guess(_row(), met, 8, tmp_path / "work")
    assert count == 4 and out == tmp_path / "work" / met.name and note["moved_to"] == "noah-four-layers"
    with out.open("rb") as handle:
        got = {name: np.frombuffer(parts[-1][4:-4], dtype=">f4") for name, _, _, parts in cs._wps_groups(handle)}
    assert list(got) == ["TT", "ST000010", "ST010040", "ST040100", "ST100200",
                         "SM000010", "SM010040", "SM040100", "SM100200", "LANDSEA"]
    for name, mid in (("ST000010", 5.0), ("ST010040", 25.0), ("ST040100", 70.0), ("ST100200", 150.0)):
        np.testing.assert_allclose(got[name][:2], [270.0 + mid / 10.0, 280.0 + mid / 10.0], rtol=1e-6)
        assert got[name][2] == 0.0
    np.testing.assert_allclose(got["SM040100"][0], 0.2 + 70.0 / 1000.0, rtol=1e-6)
    assert got["SM040100"][1] == 0.0  # a dry (glacier) land point stays dry; the fork floors it
    with met.open("rb") as before, out.open("rb") as after:
        kept = {name: parts for name, _, _, parts in cs._wps_groups(before) if name in ("TT", "LANDSEA")}
        assert kept == {name: parts for name, _, _, parts in cs._wps_groups(after) if name in ("TT", "LANDSEA")}


def test_a_soil_column_that_stops_short_of_a_metre_is_refused_before_the_producer(tmp_path):
    """A one-slab source (HRDPS's 0-10 cm before its deep-soil donor) cannot be moved onto Noah's four."""
    met = _wps_v5(tmp_path / "MET:2026-11-11_00", [("ST000010", "K", (280.0, 281.0, 0.0)),
                                                    ("SM000010", "m3 m-3", (0.3, 0.3, 0.0))])
    with pytest.raises(cs.RrfsInitError, match="without levels reaching 1 m.*deep-soil donor"):
        cs.fork_soil_first_guess(_row(), met, 1, tmp_path / "work")


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


def test_the_default_door_recovers_native_nodes_despite_legacy_four(tmp_path, monkeypatch):
    import numpy as np
    from test_soil_handover import write_met
    from woof.hex.init_door import run_init_command

    arguments = _door(tmp_path, monkeypatch)
    write_met(arguments.met, np.full((9, 2, 2), 280., np.float32),
              np.full((9, 2, 2), .2, np.float32), extra_layers=True)
    assert arguments.nfgsoillevels == 4
    assert run_init_command(arguments) == 0
    jobs = [json.loads(line) for line in (tmp_path / "jobs.jsonl").read_text().splitlines()]
    assert jobs[1]["config"]["n_fg_soil_levels"] == 9
    receipt = json.loads((tmp_path / "init.nc.provenance.json").read_text())
    assert receipt["soil_handover"]["path"] == "native-nodes"
    assert receipt["soil_handover"]["requested_nfgsoillevels"] == 4
    assert not (tmp_path / "init.nc.rrfs-work/soil-normalized.met").exists()


def test_the_default_door_infers_count_when_omitted(tmp_path, monkeypatch):
    import numpy as np
    from test_soil_handover import write_met
    from woof.hex.init_door import run_init_command
    arguments = _door(tmp_path, monkeypatch)
    write_met(arguments.met, np.full((9, 2, 2), 280., np.float32),
              np.full((9, 2, 2), .2, np.float32), extra_layers=True)
    arguments.nfgsoillevels = None
    assert run_init_command(arguments) == 0
    receipt = json.loads((tmp_path / "init.nc.provenance.json").read_text())
    assert receipt["soil_handover"]["n_fg_soil_levels"] == 9


def test_a_layer_only_first_guess_still_needs_its_soil_count(tmp_path, monkeypatch):
    import pytest
    from woof.hex.init_door import run_init_command
    arguments = _door(tmp_path, monkeypatch)
    arguments.nfgsoillevels = None
    with pytest.raises(Exception, match="nfgsoillevels"):
        run_init_command(arguments)


def test_a_refused_balance_removes_the_rrfs_init(tmp_path, monkeypatch):
    """The balance writes block by block; a refusal part way must not leave a half-balanced init."""
    from woof.hex.init_door import InitDoorRefusal, run_init_command
    from woof.hex import init_balance

    arguments = _door(tmp_path, monkeypatch, "--condensate-balance", "yes")

    def refuse(path, *, enabled):
        path.write_bytes(b"half balanced")
        raise ValueError("loaded hydrostatic balance produced invalid pressure")

    monkeypatch.setattr(init_balance, "complete_init", refuse)
    with pytest.raises(InitDoorRefusal, match="incomplete file was removed"):
        run_init_command(arguments)
    assert not (tmp_path / "init.nc").exists()
