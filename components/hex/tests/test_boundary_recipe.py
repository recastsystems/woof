"""Parent selection uses atmospheric metadata and stays independent of suite and pin."""
import struct

import pytest

from woof.hex.mpasphys.rrfs import cold_start_rrfs as cs


def met(path, source, field="TT"):
    def record(payload):
        return struct.pack(">i", len(payload)) + payload + struct.pack(">i", len(payload))
    def text(value, size):
        return value.encode().ljust(size)
    metadata = (text("2023-03-31_18:00:00", 24) + struct.pack(">f", 0) + text(source, 32)
                + text(field, 9) + text("K", 25) + text("temperature", 46)
                + struct.pack(">fiii", 85000, 1, 1, 0))
    path.write_bytes(b"".join(record(p) for p in (
        struct.pack(">i", 5), metadata, text("SWCORNER", 8) + struct.pack(">fffff", 40, -90, 1, 1, 6371),
        struct.pack(">i", 0), struct.pack(">f", 280))))
    return path


@pytest.mark.parametrize("source,expected", [
    ("NCEP GFS Model", (False, True)), ("ncep-gfs", (False, True)),
    ("RAP", (True, False)), ("NCEP HRRR", (True, False)),
    ("RRFS", (True, False)), ("ECMWF", (False, False)),
    ("AI-GFS", (False, False)),
])
def test_noaa_parent_settings_on_every_producer_route(tmp_path, source, expected):
    path = met(tmp_path / "misleading-filename", source)
    for producer in cs.load_producers()["producers"].values():
        for suite in producer["suite_routes"]:
            from woof.hex.physics_suites import load_table
            pin = load_table()[producer["physics_table_key"]]["pin_set"]
            _, row = cs.producer_for_suite(suite, pin_set=pin)
            chosen, receipt = cs.boundary_recipe(row, "noaa", [("2023-03-31_18:00:00", path)])
            assert (chosen["lbc"]["hydrometeors_rrfs"], chosen["lbc"]["hydrometeors_gfs"]) == expected
            assert chosen["lbc"]["absent_numbers"] == "zero"
            assert chosen["lbc"]["aerosol_climatology"] is False
            assert chosen["case7"] == row["case7"]
            assert row["lbc"]["absent_numbers"] == "from-mass"
            assert receipt["name"] == "noaa" and receipt["parents"][0]["map_source"] == source


def test_default_never_scans_or_changes_the_recipe():
    _, row = cs.producer_for_suite("hrrrv5")
    chosen, receipt = cs.boundary_recipe(row, None, [("time", "/absent")])
    assert chosen is row and receipt["name"] == "default"


def test_new_recipe_is_only_a_table_row(tmp_path, monkeypatch):
    table = dict(cs.load_producers())
    table["boundary_recipes"] = {"mass": {"source": "test row", "parents": {},
        "otherwise": {"config": {"hydrometeors_rrfs": False, "hydrometeors_gfs": True,
                                  "absent_numbers": "zero"}}}}
    _, row = cs.producer_for_suite("hrrrv5")
    monkeypatch.setattr(cs, "load_producers", lambda: table)
    selected, receipt = cs.boundary_recipe(row, "mass", [("time", met(tmp_path / "met", "ECMWF"))])
    assert selected["lbc"]["hydrometeors_gfs"] and receipt["name"] == "mass"


def test_surface_donor_does_not_select_the_parent(tmp_path):
    donor = met(tmp_path / "donor", "NCEP GFS Model", "LANDSEA")
    parent = met(tmp_path / "parent", "ECMWF")
    combined = tmp_path / "combined"
    combined.write_bytes(donor.read_bytes() + parent.read_bytes())
    _, row = cs.producer_for_suite("hrrrv5")
    _, receipt = cs.boundary_recipe(row, "noaa", [("time", combined)])
    assert receipt["parents"][0]["parent"] == "other"
    with pytest.raises(cs.RrfsInitError, match="different hydrometeor settings"):
        cs.boundary_recipe(row, "noaa", [("t1", parent), ("t2", met(tmp_path / "gfs", "GFS"))])
    with pytest.raises(cs.RrfsInitError, match="no TT"):
        cs.boundary_recipe(row, "noaa", [("time", donor)])


@pytest.mark.parametrize("suite", [None, "convection_permitting", "mesoscale_reference",
                                   "arwen_mpas_column", "arwen_thompson_column"])
def test_routes_without_a_producer_name_the_fields_they_would_drop(suite):
    from woof.hex.init_physics import init_selection
    selection = init_selection(suite, None)
    with pytest.raises(cs.RrfsInitError, match="silently drop those parent fields"):
        cs.require_boundary_producer("noaa", selection.get("producer"))
    cs.require_boundary_producer(None, selection.get("producer"))


@pytest.mark.parametrize("suite", sorted(cs.producer_suites()))
@pytest.mark.parametrize("pin", ["rrfs-44c0849", "rrfs-1d6e370e"])
def test_country_forwards_the_option_for_every_producer_suite(tmp_path, suite, pin):
    from woof.hex.cli import build_parser
    from woof.hex.country_door import plan
    args = build_parser().parse_args([
        "country", "--point", "40,-90", "--radius-km", "300", "--cycle", "2023-03-31T18",
        "--hours", "6", "--work-dir", str(tmp_path), "--boundary-recipe", "noaa",
        f"--forecast-arg=--physics-suite={suite}", f"--forecast-arg=--physics-pin-set={pin}"])
    value = plan(args)
    lbc = build_parser().parse_args(value["commands"]["lbc"])
    assert lbc.boundary_recipe == "noaa" and lbc.physics_suite == suite
    assert lbc.physics_pin_set == pin
    assert build_parser().parse_args(value["commands"]["producer_init"]).physics_pin_set == pin


@pytest.mark.parametrize("suite", sorted(cs.producer_suites()))
@pytest.mark.parametrize("pin", ["rrfs-44c0849", "rrfs-1d6e370e"])
def test_regional_forwards_the_option_for_every_producer_suite(tmp_path, monkeypatch, suite, pin):
    from woof.hex.cli import build_parser
    from woof.hex import regional_door as door
    args = build_parser().parse_args([
        "regional", "--point", "40,-90", "--cycle", "2023-03-31T18", "--hours", "6",
        "--work-dir", str(tmp_path), "--boundary-recipe", "noaa",
        f"--forecast-arg=--physics-suite={suite}", f"--forecast-arg=--physics-pin-set={pin}"])
    value = door.plan(args)
    monkeypatch.setattr(door, "_generated", lambda value: {"cull": {"lbc_dir": str(tmp_path / "lbc")}})
    commands = []
    monkeypatch.setattr(door, "_cli", lambda argv, status: commands.append(argv) or 0)
    door._stage_lbc(args, value, None)
    lbc = build_parser().parse_args(commands[0])
    assert lbc.boundary_recipe == "noaa" and lbc.physics_suite == suite
    assert lbc.physics_pin_set == pin
    assert door._init_suite_argv(value)[-2:] == ["--physics-pin-set", pin]


def test_country_without_boundaries_records_unapplied_option(tmp_path):
    from woof.hex.cli import build_parser
    from woof.hex.country_door import plan
    args = build_parser().parse_args([
        "country", "--row", "philippines", "--cycle", "2023-03-31T18",
        "--work-dir", str(tmp_path), "--boundary-recipe", "noaa"])
    assert plan(args)["boundary_recipe"]["applied"] is False


@pytest.mark.parametrize("noaa", [False, True])
def test_case9_keeps_aerosol_fields_but_passes_the_climatology_policy(tmp_path, noaa):
    _, row = cs.producer_for_suite("hrrrv5")
    intervals = [("2023-03-31_18:00:00", met(tmp_path / "met", "GFS"))]
    chosen, _ = cs.boundary_recipe(row, "noaa" if noaa else None, intervals)
    config = cs.case9_jobs(chosen, init=tmp_path / "init.nc",
        start_time=intervals[0][0], intervals=intervals, nfglevels=58,
        out_dir=tmp_path / "lbc", carrier=tmp_path / "carrier.nc",
        aerosols=tmp_path / "aerosols.dat")[1]["config"]
    assert config["aerosolaware"] is True
    # The default plan omits the worker's true default; NOAA must carry its false override.
    if noaa:
        assert config["aerosol_climatology"] is False
    else:
        assert "aerosol_climatology" not in config
    assert config["absent_numbers"] == ("zero" if noaa else "from-mass")
