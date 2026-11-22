"""The columns at 180 degrees of a global category dataset are repaired.

WPS_GEOG's 30 arc-second MODIS land use carries water, lake and ice in its
last column and its first two where the ground either side is land, which
put a strip of water and ice down Taveuni in a 1 km domain's land mask
(woof.static.geog.repair_seam_columns).  These tests build a small global
dataset with the same defect and check the copy a build reads.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from woof.static import geog
from woof.static.geog import (GeogDataset, repair_seam_columns,
                               seam_repaired_dataset)

WATER, LAKE, ICE, FOREST, CROP = 17, 21, 15, 2, 12
WET = (WATER, LAKE, ICE)


def _strips(ny, land=FOREST):
    last = np.full((ny, 2), land, dtype=np.int64)
    first = np.full((ny, 3), land, dtype=np.int64)
    return last, first


def test_strip_of_ice_and_water_on_land_is_read_as_land():
    last, first = _strips(9)
    last[2:7, 1] = WATER          # column nx
    first[3:6, 0] = ICE           # column 1
    first[4, 1] = LAKE            # column 2
    new_last, new_first, changes = repair_seam_columns(last, first, WET, water=WATER)
    assert (new_last == FOREST).all() and (new_first == FOREST).all()
    assert len(changes) == 5 + 3 + 1
    assert {(was, now) for _, _, was, now in changes} == {
        (WATER, FOREST), (ICE, FOREST), (LAKE, FOREST)}


def test_takes_the_inland_pixel_on_its_own_side():
    last, first = _strips(3)
    last[:, 0] = CROP
    last[1, 1] = WATER
    first[1, 0] = ICE
    new_last, new_first, _ = repair_seam_columns(last, first, WET, water=WATER)
    assert new_last[1, 1] == CROP      # west of 180: column nx-1's class
    assert new_first[1, 0] == FOREST   # east of 180: column 3's class


def test_a_coast_along_180_keeps_its_water():
    # Land west of the line, sea east of it: half the clean neighbours are
    # water, so the edge water is a coast and stays.
    last, first = _strips(5)
    first[:, :] = WATER
    last[:, 1] = WATER
    new_last, new_first, changes = repair_seam_columns(last, first, WET, water=WATER)
    assert changes == []
    assert (new_last == last).all() and (new_first == first).all()


def test_ice_on_a_coast_takes_the_pixel_inland_of_it():
    # Sea east of the line, land west of it: rule 1 leaves the edge alone,
    # but ice with no ice beside it is not a glacier on a tropical shore.
    last, first = _strips(5)
    first[:, :] = WATER
    first[2, 0] = ICE          # column 1: inland (column 3) is sea
    last[2, 1] = ICE           # column nx: inland (column nx-1) is forest
    new_last, new_first, changes = repair_seam_columns(
        last, first, WET, water=WATER)
    assert new_first[2, 0] == WATER and new_last[2, 1] == FOREST
    assert len(changes) == 2


def test_ice_beside_ice_stays():
    last, first = _strips(5, land=ICE)
    first[:, 2] = WATER        # an ice shelf front at 180
    new_last, new_first, changes = repair_seam_columns(
        last, first, WET, water=WATER)
    assert changes == []


def test_rows_with_no_staged_tile_are_not_neighbours():
    last, first = _strips(4)
    last[2:, :] = -1
    first[2:, :] = -1
    first[1, 0] = ICE
    _, new_first, changes = repair_seam_columns(last, first, WET, water=WATER)
    assert new_first[1, 0] == FOREST and len(changes) == 1
    assert (new_first[2:] == -1).all()


def _write_dataset(root, grid, tile=6):
    """A global 30-degree categorical dataset with ``grid`` (ny, nx), row 1
    south, as WPS_GEOG stores it."""
    ny, nx = grid.shape
    root.mkdir(parents=True)
    (root / "index").write_text(
        "type=categorical\ncategory_min=1\ncategory_max=21\n"
        "projection=regular_ll\n"
        f"dx={360.0 / nx}\ndy={180.0 / ny}\nknown_x=1.0\nknown_y=1.0\n"
        f"known_lat={-90.0 + 90.0 / ny}\nknown_lon={-180.0 + 180.0 / nx}\n"
        f"wordsize=1\ntile_x={tile}\ntile_y={tile}\ntile_z=1\n"
        "iswater=17\nislake=21\nisice=15\nisurban=13\n", encoding="utf-8")
    for y0 in range(0, ny, tile):
        for x0 in range(0, nx, tile):
            name = f"{x0 + 1:05d}-{x0 + tile:05d}.{y0 + 1:05d}-{y0 + tile:05d}"
            grid[y0:y0 + tile, x0:x0 + tile].astype(np.uint8).tofile(
                root / name)
    return root


def _globe():
    grid = np.full((12, 24), WATER, dtype=np.uint8)
    grid[3:9, :4] = FOREST     # an island across 180: columns 1..4 ...
    grid[3:9, -4:] = FOREST    # ... and nx-3..nx
    return grid


def test_repaired_copy_is_what_a_build_reads(tmp_path):
    grid = _globe()
    grid[4:8, -1] = LAKE       # the defect: column nx
    grid[4:8, 0] = ICE         # and column 1
    source = _write_dataset(tmp_path / "geog" / "modis_landuse_20class_30s_with_lakes", grid)
    cache = tmp_path / "cache"
    geog._SEAM_REPAIRED.clear()
    copy = seam_repaired_dataset(source, cache_root=cache)
    assert copy != source and copy.name == source.name
    ds = GeogDataset(copy)
    window = ds.read_window(1, 24, 1, 12).raw[0]
    expected = _globe()
    assert (window == expected).all()
    # Tiles the repair leaves alone are links to the published ones.
    untouched = copy / "00007-00012.00001-00006"
    assert untouched.samefile(source / untouched.name)
    receipt = json.loads((copy.parent / "seam-repair.json").read_text())
    assert receipt["changed_pixels"] == 8
    assert sorted(receipt["tiles_rewritten"]) == [
        "00001-00006.00001-00006", "00001-00006.00007-00012",
        "00019-00024.00001-00006", "00019-00024.00007-00012"]
    # A second process finds the same copy without scanning again.
    geog._SEAM_REPAIRED.clear()
    assert seam_repaired_dataset(source, cache_root=cache) == copy


def test_a_clean_dataset_is_read_where_it_is_and_nothing_is_written(tmp_path):
    source = _write_dataset(tmp_path / "geog" / "clean", _globe())
    cache = tmp_path / "cache"
    geog._SEAM_REPAIRED.clear()
    assert seam_repaired_dataset(source, cache_root=cache) == source
    assert not cache.exists()


def test_switch_reads_the_tiles_as_published(tmp_path, monkeypatch):
    grid = _globe()
    grid[4:8, 0] = ICE
    source = _write_dataset(tmp_path / "geog" / "defect", grid)
    geog._SEAM_REPAIRED.clear()
    monkeypatch.setenv(geog.SEAM_REPAIR_ENV, "off")
    assert seam_repaired_dataset(source, cache_root=tmp_path / "c") == source


def test_static_selection_reads_the_repaired_land_use(tmp_path, monkeypatch):
    from woof.static.build import GeogSelection, _seam_repaired_selection

    root = tmp_path / "WPS_GEOG"
    grid = _globe()
    grid[4:8, 0] = ICE
    _write_dataset(root / "modis_landuse_20class_30s_with_lakes", grid)
    monkeypatch.setenv(geog.SEAM_CACHE_ENV, str(tmp_path / "cache"))
    geog._SEAM_REPAIRED.clear()
    selection = _seam_repaired_selection(None, root)
    assert selection is not None
    path = selection.path("landuse")
    assert path.name == "modis_landuse_20class_30s_with_lakes"
    assert path.parent.parent == tmp_path / "cache"
    assert selection.root == root
    assert selection.terrain == GeogSelection.fallback(root).terrain
    # The other fields still resolve under the GEOG root.
    assert selection.path("terrain") == root / "topo_gmted2010_30s"


def test_unstaged_land_use_leaves_the_selection_alone(tmp_path):
    from woof.static.build import _seam_repaired_selection
    assert _seam_repaired_selection(None, tmp_path) is None


@pytest.mark.parametrize("bad", ["type=continuous\n", ""])
def test_non_category_index_is_left_alone(tmp_path, bad):
    root = tmp_path / "d"
    root.mkdir()
    (root / "index").write_text(bad)
    geog._SEAM_REPAIRED.clear()
    assert seam_repaired_dataset(root, cache_root=tmp_path / "c") == root


def _defect_source(tmp_path):
    grid = _globe()
    grid[4:8, 0] = ICE
    return _write_dataset(tmp_path / "geog" / "modis_landuse_20class_30s", grid)


def test_the_repair_line_is_one_gpuwm_go_relays(tmp_path, capsys):
    # `woof go` and `woof prep` relay only "note:" and "warning:" lines
    # from a passing preparation stage (go_cli.LaunchOutput, prep_output).
    source = _defect_source(tmp_path)
    geog._SEAM_REPAIRED.clear()
    seam_repaired_dataset(source, cache_root=tmp_path / "cache")
    lines = capsys.readouterr().err.splitlines()
    assert any(line.startswith("note: geography: modis_landuse_20class_30s: "
                               "4 pixel(s)") for line in lines), lines


def test_an_unwritable_cache_writes_the_copy_under_the_temp_folder(
        tmp_path, monkeypatch):
    import tempfile
    source = _defect_source(tmp_path)
    blocked = tmp_path / "not-a-folder"
    blocked.write_text("")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    geog._SEAM_REPAIRED.clear()
    copy = seam_repaired_dataset(source, cache_root=blocked)
    assert copy != source
    assert (tmp_path / "tmp") in copy.parents
    window = GeogDataset(copy).read_window(1, 24, 1, 12).raw[0]
    assert (window == _globe()).all()


def test_no_writable_folder_stops_the_build_rather_than_read_the_strip(
        tmp_path, monkeypatch):
    # Before: the tiles were read as published with one stderr line, so a
    # box with a read-only home got the strip back with nothing to show it.
    import tempfile
    source = _defect_source(tmp_path)
    blocked = tmp_path / "not-a-folder"
    blocked.write_text("")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(blocked))
    geog._SEAM_REPAIRED.clear()
    with pytest.raises(OSError, match="WOOF_GEOG_SEAM_CACHE"):
        seam_repaired_dataset(source, cache_root=blocked)


def test_an_archive_whose_column_1_is_not_at_180_is_read_as_stored(tmp_path):
    # Its columns nx and 1 are not the ones on the line, so the rule is not
    # theirs: repairing them would rewrite real coast at Greenwich.
    grid = _globe()
    grid[4:8, 0] = ICE
    source = _write_dataset(tmp_path / "geog" / "greenwich", grid)
    index = (source / "index").read_text()
    (source / "index").write_text(
        index.replace("known_lon=-172.5", "known_lon=7.5"))
    geog._SEAM_REPAIRED.clear()
    assert seam_repaired_dataset(source, cache_root=tmp_path / "c") == source


# -- the SoilGrids void beside 180 degrees --------------------------------------

LOAM, SOILWATER = 6, 14


def _soil_strips(ny=200, reach=120, void=30, land_rows=slice(50, 150)):
    """(west, east) as repair_soil_void takes them: loam on ``land_rows``,
    water elsewhere, and ``void`` columns west of the line all water."""
    west = np.full((ny, reach + 1), SOILWATER, dtype=np.int64)
    east = np.full((ny, reach + 1), SOILWATER, dtype=np.int64)
    west[land_rows] = LOAM
    east[land_rows] = LOAM
    if void:
        west[:, reach + 1 - void:] = SOILWATER
    return west, east


def test_soil_void_beside_180_is_read_as_the_soil_beside_it():
    # Before: the land under the void took water soil (silty clay loam in
    # the WRF route, ocean in the hex one).
    west, east = _soil_strips()
    new_west, new_east, info = geog.repair_soil_void(west, east)
    assert (info["west"], info["east"], info["filled"]) == (30, 0, 100 * 30)
    assert (new_west[50:150] == LOAM).all()
    assert (new_west[:50] == SOILWATER).all() and (new_west[150:] == SOILWATER).all()
    assert (new_east == east).all()


def test_soil_void_takes_the_nearer_land_and_skips_sea_beside_it():
    # The column just outside the void is sea on this row; the land beyond
    # it is used, and the nearer side wins.
    west, east = _soil_strips()
    west[60, 71:91] = SOILWATER        # sea just west of the void ...
    west[60, 70] = 3                   # ... land 20 columns farther out
    east[60, 0] = 9                    # land at the line, east side
    new_west, _, _ = geog.repair_soil_void(west, east)
    assert new_west[60, 120] == 9      # beside the line: east is nearer
    assert new_west[60, 91] == 3       # the void's west end: west is nearer


def test_soil_without_a_void_is_read_as_stored():
    west, east = _soil_strips(void=0)
    new_west, new_east, info = geog.repair_soil_void(west, east)
    assert info is None and (new_west == west).all() and (new_east == east).all()


def _write_soil_dataset(root, grid, tile=240):
    ny, nx = grid.shape
    root.mkdir(parents=True)
    (root / "index").write_text(
        "type=categorical\ncategory_min=1\ncategory_max=16\n"
        "projection=regular_ll\n"
        f"dx={360.0 / nx}\ndy={180.0 / ny}\nknown_x=1.0\nknown_y=1.0\n"
        f"known_lat={-90.0 + 90.0 / ny}\nknown_lon={-180.0 + 180.0 / nx}\n"
        f"wordsize=1\ntile_x={tile}\ntile_y={ny}\ntile_z=1\n", encoding="utf-8")
    for x0 in range(0, nx, tile):
        name = f"{x0 + 1:05d}-{x0 + tile:05d}.00001-{ny:05d}"
        grid[:, x0:x0 + tile].astype(np.uint8).tofile(root / name)
    return root


def test_static_selection_reads_the_filled_soil(tmp_path, monkeypatch):
    from woof.static.build import _seam_repaired_selection

    grid = np.full((180, 720), SOILWATER, dtype=np.uint8)
    grid[60:120] = LOAM
    grid[:, -30:] = SOILWATER
    root = tmp_path / "WPS_GEOG"
    _write_soil_dataset(root / "soilgrids" / "texture_top", grid)
    monkeypatch.setenv(geog.SEAM_CACHE_ENV, str(tmp_path / "cache"))
    geog._SEAM_REPAIRED.clear()
    selection = _seam_repaired_selection(None, root)
    path = selection.path("soil_top")
    assert path.parent.parent == tmp_path / "cache"
    filled = GeogDataset(path).read_window(1, 720, 1, 180).raw[0]
    assert (filled[60:120] == LOAM).all() and (filled[:60] == SOILWATER).all()
    # The bottom set is not staged, so it keeps its own path.
    assert selection.path("soil_bottom") == root / "soilgrids" / "texture_bot"


class _EdgeGrid:
    """A grid whose edge longitudes are given: all the repair's note rule reads."""

    def __init__(self, lon_west, lon_east, e_we=20, e_sn=10):
        self.e_we, self.e_sn = e_we, e_sn
        self._west, self._east = lon_west, lon_east

    def ij_to_latlon(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        lon = self._west + (x - 1.0) / (self.e_we - 1.0) * (self._east - self._west)
        return np.zeros_like(lon), ((lon + 180.0) % 360.0) - 180.0


@pytest.mark.parametrize("west,east,said", [
    (-96.0, -87.0, False),    # Iowa: nowhere near 180
    (179.2, 180.7, True),     # Taveuni: across the line
    (170.0, 178.5, True),     # ends 1.5 degrees short of it: inside the reach
    (150.0, 177.0, False),    # ends 3 degrees short
])
def test_the_repair_lines_are_for_a_domain_that_reaches_180(
        tmp_path, monkeypatch, capsys, west, east, said):
    """THE BREAKAGE THIS PREVENTS (3.0.3 candidate, 2026-11-14): a 12 km and
    3 km forecast over Iowa printed three lines about pixels "at 180
    degrees" on every preparation.  Every domain still reads the repaired
    copy; only a domain that reaches the line is told what it changed."""
    from woof.static.build import _grid_reaches_180, _seam_repaired_selection

    grid = np.full((180, 720), SOILWATER, dtype=np.uint8)
    grid[60:120] = LOAM
    grid[:, -30:] = SOILWATER
    root = tmp_path / "WPS_GEOG"
    _write_soil_dataset(root / "soilgrids" / "texture_top", grid)
    monkeypatch.setenv(geog.SEAM_CACHE_ENV, str(tmp_path / "cache"))
    geog._SEAM_REPAIRED.clear()
    domain = _EdgeGrid(west, east)
    assert _grid_reaches_180(domain) is said
    selection = _seam_repaired_selection(None, root, grid=domain)
    assert selection.path("soil_top").parent.parent == tmp_path / "cache"
    lines = [line for line in capsys.readouterr().err.splitlines()
             if line.startswith("note: geography:")]
    assert bool(lines) is said, lines


def test_a_grid_the_rule_cannot_read_keeps_the_lines():
    from woof.static.build import _grid_reaches_180

    assert _grid_reaches_180(object()) is True
