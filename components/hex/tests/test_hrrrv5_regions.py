"""hrrrv5 (and the other producer suites) on the region doors, any shape, any card count.

THE BREAKAGES THESE TESTS HOLD SHUT (hrrrv5 everywhere, 2026-10-31):
* ``country --forecast-arg=--physics-suite=hrrrv5`` built hex's own v8.4.2 start state and boundaries, and the
  forecast died at its start (KeyError 'nscat' in pool_v842) from GFS and from IFS;
* the country door took no ellipse and no GeoJSON outline, which the site offers;
* ``regional`` had no card flags, so a region over one card's memory could not run;
* a cleaned physics-resource cache whose RRFS tables were gone was trusted and the forecast died on a missing table;
* the fork static read NOAA's licence-unclear soiltemp_1deg; ERA5's deep-soil temperature is the default now.
"""
import json
from pathlib import Path

import pytest

from woof.hex.cli import build_parser
from woof.hex.country_door import plan


def country(tmp_path, *extra):
    return build_parser().parse_args(["country", "--cycle", "2026-10-30T18", "--source", "gfs",
                                      "--work-dir", str(tmp_path / "run"), "--fine-dx-m", "3000",
                                      "--background-km", "96", *extra])


HRRRV5 = "--forecast-arg=--physics-suite=hrrrv5"


def test_country_hrrrv5_region_starts_from_the_producer_on_the_cut(tmp_path):
    value = plan(country(tmp_path, "--point", "38.5,-15", "--radius-km", "300", HRRRV5))
    commands = value["commands"]
    init = commands["producer_init"]
    root = tmp_path / "run"
    # The producer's case 7 on the cut, from the start time's decoded intermediate (boundary endpoint 0).
    assert init[0] == "init" and init[init.index("--physics-suite") + 1] == "hrrrv5"
    assert init[init.index("--met") + 1] == str(root / "boundary-met" / "MET:2026-10-30_18")
    assert init[init.index("--grid") + 1] == str(root / "mesh.grid.nc")
    assert init[init.index("--static") + 1] == str(root / "mesh.static.nc")
    rrfs_init = str(root / "rrfs-init" / "init.nc")
    assert init[init.index("--out") + 1] == rrfs_init == value["paths"]["init"]
    # The cut carries grid and static only; the boundaries are the producer's case 9 on its init.
    assert "--parent-init" not in commands["cull"]
    lbc = commands["lbc"]
    assert lbc[lbc.index("--grid") + 1] == rrfs_init and lbc[lbc.index("--physics-suite") + 1] == "hrrrv5"
    forecast = commands["forecast"]
    assert forecast[forecast.index("--init") + 1] == rrfs_init
    for name, command in commands.items():
        if name != "init":
            build_parser().parse_args(command)


def test_country_mpas_suite_region_is_unchanged(tmp_path):
    value = plan(country(tmp_path, "--point", "38.5,-15", "--radius-km", "300",
                         "--forecast-arg=--physics-suite=convection_permitting"))
    commands = value["commands"]
    assert "producer_init" not in commands
    assert "--parent-init" in commands["cull"] and "--physics-suite" not in commands["lbc"]
    assert commands["forecast"][commands["forecast"].index("--init") + 1].endswith("mesh.init.nc")


def test_country_prepare_skips_the_parent_init_and_runs_the_producer(tmp_path, monkeypatch):
    """The prepare stage order for a producer suite: no parent v8.4.2 init, the cull, the endpoints, then the
    producer's init with the geography, then its boundaries."""
    from woof.hex import country_door, cli, mesh_point, prep_store, region_static, ugwp_oro, lbc_producer, mesh_rows
    value = plan(country(tmp_path, "--ellipse", "35,135,400,250,60", HRRRV5))
    ran = []
    monkeypatch.setattr(cli, "main", lambda argv: ran.append(list(argv)) or 0)
    monkeypatch.setattr(prep_store, "run_or_fetch", lambda argv, **kw: 0)
    monkeypatch.setattr(region_static, "build_parent_static", lambda **kw: {"built": True})
    monkeypatch.setattr(mesh_point, "admit_pair", lambda *a, **k: {})
    monkeypatch.setattr(mesh_point, "resolve_geog", lambda explicit: tmp_path / "geog")
    monkeypatch.setattr(ugwp_oro, "prepare_for_forecast", lambda *a, **k: None)
    monkeypatch.setattr(mesh_rows, "describe_generated", lambda **kw: {})
    monkeypatch.setattr(mesh_rows, "describe_cull", lambda **kw: {})
    monkeypatch.setattr(mesh_rows, "write_rows", lambda *a, **k: None)
    monkeypatch.setattr(lbc_producer, "produce_source_endpoints", lambda *a, **k: [])
    endpoint = Path(value["boundary_endpoints"][0]["argv"][value["boundary_endpoints"][0]["argv"].index("--out-dir") + 1])
    endpoint.mkdir(parents=True)
    (endpoint / "init-source-receipt.json").write_text(json.dumps({"products": [], "surface_input": None}))
    (tmp_path / "run").mkdir(exist_ok=True)
    args = country(tmp_path, "--ellipse", "35,135,400,250,60", HRRRV5)
    status = type("S", (), {"note": lambda self, text: None})()
    assert country_door._prepare_region(args, value, status) == 0
    names = [argv[0] for argv in ran]
    assert names == ["cull", "init", "lbc"], names
    init = ran[1]
    assert init[init.index("--geog") + 1] == str(tmp_path / "geog")


def test_country_takes_an_ellipse_and_a_geojson_outline(tmp_path):
    ellipse = plan(country(tmp_path, "--ellipse", "35,135,400,250,60", HRRRV5))
    assert ellipse["row"]["geometry"]["kind"] == "polygon" and len(ellipse["row"]["geometry"]["vertices_deg"]) > 16
    lat, lon = ellipse["row"]["mesh"]["point_deg"]
    assert abs(lat - 35) < 1.5 and abs(lon - 135) < 1.5
    shape = tmp_path / "europe.geojson"
    shape.write_text(json.dumps({"type": "Polygon", "coordinates": [[[2, 44], [16, 44], [16, 50], [8, 52], [2, 49],
                                                                      [2, 44]]]}))
    drawn = plan(country(tmp_path, "--shape", str(shape), HRRRV5))
    assert drawn["row"]["geometry"]["kind"] == "polygon"
    assert [44.0, 2.0] in drawn["row"]["geometry"]["vertices_deg"]
    assert drawn["row"]["id"] != ellipse["row"]["id"]


def test_a_drawn_circle_is_the_point_cap(tmp_path):
    shape = tmp_path / "cap.json"
    shape.write_text(json.dumps({"kind": "cap", "center_deg": [38.5, -15], "radius_km": 300}))
    drawn = plan(country(tmp_path, "--shape", str(shape), HRRRV5))
    point = plan(country(tmp_path, "--point", "38.5,-15", "--radius-km", "300", HRRRV5))
    assert drawn["row"]["id"] == point["row"]["id"] and drawn["row"]["geometry"]["kind"] == "cap"


def test_country_forwards_the_card_flags(tmp_path):
    value = plan(country(tmp_path, "--point", "38.5,-15", "--radius-km", "300", HRRRV5,
                         "--samehost-devices", "0,1,2,3", "--samehost-ranks", "processes"))
    forecast = value["commands"]["forecast"]
    assert forecast[forecast.index("--samehost-devices") + 1] == "0,1,2,3"
    assert forecast[forecast.index("--samehost-ranks") + 1] == "processes"


def test_regional_forwards_the_card_flags_to_the_forecast(tmp_path):
    from woof.hex.regional_door import samehost_argv
    args = build_parser().parse_args(["regional", "--point", "38.5,-15", "--work-dir", str(tmp_path / "r"),
                                      "--cycle", "2026-10-30T18", "--source", "gfs",
                                      "--samehost-devices", "0,1,2,3", "--deep-soil-temperature", "era5"])
    assert samehost_argv(args) == ["--samehost-devices", "0,1,2,3"]
    args = build_parser().parse_args(["regional", "--point", "38.5,-15", "--work-dir", str(tmp_path / "r"),
                                      "--cycle", "2026-10-30T18", "--source", "gfs"])
    assert samehost_argv(args) == []


def test_the_fork_static_reads_era5_deep_soil_by_default(tmp_path):
    from woof.hex.mpasphys.rrfs import cold_start_rrfs as cs
    _, row = cs.producer_for_suite("hrrrv5")
    geog = tmp_path / "geog"
    for name in {*row["static"]["geog"].values(), "soiltemp_1deg"}:
        (geog / name).mkdir(parents=True, exist_ok=True)
    job = cs.static_job(row, grid=tmp_path / "g.nc", geog=geog, out=tmp_path / "s.nc", valid_time="t")
    assert job["config"]["geog"]["soil_temperature"] == str(geog / "soiltemp_era5_1deg")
    job = cs.static_job(row, grid=tmp_path / "g.nc", geog=geog, out=tmp_path / "s.nc", valid_time="t",
                        deep_soil="islscp-i")
    assert job["config"]["geog"]["soil_temperature"] == str(geog / "soiltemp_1deg")
    choice, chosen = cs.deep_soil_choice(row)
    assert choice == "era5" and "warmer" in chosen["about"] and "licence" in chosen
    assert "licence_hold" in cs.deep_soil_choice(row, "islscp-i")[1]
    (geog / "soiltemp_era5_1deg").rmdir()
    with pytest.raises(cs.RrfsInitError, match="soiltemp_era5_1deg"):
        cs.static_job(row, grid=tmp_path / "g.nc", geog=geog, out=tmp_path / "s.nc", valid_time="t")


def test_a_cache_whose_rrfs_tables_are_gone_fetches_them_again(tmp_path, monkeypatch):
    from woof.hex.mpasphys import resources_build as rb
    from woof.hex.mpasphys.native_library_build import source_manifest
    pin_set = "rrfs-44c0849"
    manifest = source_manifest(pin_set)
    tree = tmp_path / "rrfs-source" / "tree"
    files = tree / rb._PHYSICS_FILES
    files.mkdir(parents=True)
    names = list(manifest["tables"])
    def place(name):  # sparse: the check reads sizes, never contents
        with (files / name).open("wb") as handle:
            handle.truncate(int(manifest["tables"][name].get("bytes") or 1))

    for name in names:
        place(name)
    assert rb.source_tables_missing(tree, pin_set) == []
    (files / names[0]).unlink()
    assert rb.source_tables_missing(tree, pin_set) == [names[0]]
    key = "mpas-source-" + pin_set
    monkeypatch.setattr(rb, "resolve_resource", lambda k, cache=None: tree)
    assembled = []

    def assemble(root, **kw):
        assembled.append(root)
        place(names[0])
        return {"path": str(tree), "sha256": "0"}

    monkeypatch.setattr(rb, "assemble_rrfs_source", assemble)
    monkeypatch.setattr(rb, "record_installed", lambda root, rows: root / "installed.json")
    out = rb.ensure_route_resources((key,), cache=tmp_path, pin_set=pin_set, log=lambda text: None)
    assert out == {key: tree} and len(assembled) == 1
