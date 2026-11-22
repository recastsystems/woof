# SPDX-License-Identifier: Apache-2.0
"""A region on a producer suite reads its geography before it fetches a source hour. No card, no network.

THE BREAKAGE (a front-door run of a 0.6.0 build): ``woof hex country --source icon
--forecast-arg=--physics-suite=hrrrv5`` on a machine whose geography held the door's own datasets but not the fork
static's built the mesh and fetched and decoded seven ICON hours (186 s, 5.1 GB) before the init named the missing
datasets (modis_landuse_20class_15s).
"""
import argparse

import pytest

from woof.hex import country_door as door
from woof.hex.mpasphys.rrfs import cold_start_rrfs as cold

SUITE = "hrrrv5"
INIT = ["init", "--met", "MET:2026-11-15_12", "--static", "mesh.static.nc", "--grid", "mesh.grid.nc", "--out",
        "init.nc", "--start-time", "2026-11-15_12:00:00", "--nfglevels", "19", "--nfgsoillevels", "8",
        "--physics-suite", SUITE]


def _row():
    return cold.producer_for_suite(SUITE)[1]


def _geog(tmp_path, leave_out=()):
    root = tmp_path / "geog"
    root.mkdir()
    for name in cold.static_geog_datasets(_row()).values():
        if name not in leave_out:
            (root / name).mkdir(parents=True, exist_ok=True)
    return root


def _args(geog, **given):
    return argparse.Namespace(geog=geog, static=None, **given)


def test_the_fork_static_names_its_datasets():
    names = cold.static_geog_datasets(_row())
    assert "modis_landuse_20class_15s" in names.values() and names["soil_temperature"] == "soiltemp_era5_1deg"
    assert cold.static_geog_datasets(_row(), "islscp-i")["soil_temperature"] != names["soil_temperature"]


def test_a_whole_geography_passes(tmp_path):
    door.refuse_producer_geog_missing(_args(_geog(tmp_path)), {"commands": {"producer_init": list(INIT)}})


def test_a_missing_dataset_is_named_with_what_to_do(tmp_path):
    geog = _geog(tmp_path, leave_out=("modis_landuse_20class_15s",))
    with pytest.raises(cold.RrfsInitError, match=r"lacks modis_landuse_20class_15s, which the fork static reads "
                                                 r".*woof fetch-geog.*or point --geog at a root that has them"):
        door.refuse_producer_geog_missing(_args(geog), {"commands": {"producer_init": list(INIT)}})


def test_the_missing_datasets_come_with_the_exact_command(tmp_path):
    # 0.6.0 said "stage them with woof fetch-geog": its default set leaves these two out, so a user who ran it
    # met the same refusal again (seen with ArWen 3.0.3). The command must name them.
    geog = _geog(tmp_path, leave_out=("modis_landuse_20class_15s", "lai_modis_30s"))
    with pytest.raises(cold.RrfsInitError) as caught:
        door.refuse_producer_geog_missing(_args(geog), {"commands": {"producer_init": list(INIT)}})
    text = str(caught.value)
    names = text.split(" lacks ", 1)[1].split(", which", 1)[0].split(", ")
    assert set(names) >= {"modis_landuse_20class_15s"}
    assert f"woof fetch-geog --root {geog} --datasets {','.join(names)} " in text, text


def test_the_chosen_deep_soil_dataset_is_the_one_read(tmp_path):
    geog = _geog(tmp_path)   # holds the default (ERA5) deep-soil temperature only
    chosen = cold.static_geog_datasets(_row(), "islscp-i")["soil_temperature"]
    with pytest.raises(cold.RrfsInitError, match=f"lacks {chosen}"):
        door.refuse_producer_geog_missing(
            _args(geog), {"commands": {"producer_init": INIT + ["--deep-soil-temperature", "islscp-i"]}})


def test_other_suites_and_a_given_static_are_left_alone(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    door.refuse_producer_geog_missing(_args(empty), {"commands": {}})   # the door's own init: no producer
    given = argparse.Namespace(geog=empty, static=tmp_path / "fork.static.nc")
    door.refuse_producer_geog_missing(given, {"commands": {"producer_init": list(INIT)}})   # the init reads that file


def test_the_region_stops_before_its_first_source_hour(tmp_path, monkeypatch):
    def fetched(*a, **k):
        raise AssertionError("a source hour was fetched before the geography was read")

    monkeypatch.setattr(door, "_SourceBesideMesh", fetched)
    geog = _geog(tmp_path, leave_out=("modis_landuse_20class_15s",))
    args = _args(geog, work_dir=tmp_path / "work", endpoint_workers=None)
    value = {"commands": {"producer_init": list(INIT)}, "boundary_endpoints": []}
    with pytest.raises(cold.RrfsInitError, match="lacks modis_landuse_20class_15s"):
        door._prepare_region(args, value, status=None)
    assert not (tmp_path / "work").exists()


def test_the_init_still_says_it(tmp_path):
    geog = _geog(tmp_path, leave_out=("modis_landuse_20class_15s",))
    with pytest.raises(cold.RrfsInitError, match="lacks modis_landuse_20class_15s"):
        cold.static_job(_row(), grid=tmp_path / "g.nc", geog=geog, out=tmp_path / "s.nc",
                        valid_time="2026-11-15_12:00:00")
