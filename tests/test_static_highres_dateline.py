"""High-resolution terrain for a domain whose footprint crosses 180 degrees.

THE BREAKAGE THIS PREVENTS (WOOF gate, 2026-11-12, tag nam-021d): the
cyclone run on the Central Pacific storm at 9.8 N 173.4 W refused in
prepare with "[static.highres] refused (dateline-window-unbuilt)" on the
staged and the live release alike.  Its 1 km nest's statics corridor covers
the whole 2,400 km parent, 176.7 E .. 163.0 W, and the window writers and
the warp read only the cut -180..180 frame.  Such a footprint is now built
turned half a turn (tiles and grid together), which changes no cell.

The proof here is a twin: two real Copernicus GLO-30 clips either side of
8 E, moved to either side of 180 degrees, must give a grid on the line
exactly the terrain the same grid gets at 8 E from the clips as published.
"""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from woof.static import rust_bridge
from woof.static.highres import sha256_file
from woof.static.highres_fetch import (FetchedFile, FootprintBBox,
                                        _geotiff_west,
                                        _shift_geotiff_longitude,
                                        dateline_rotation_deg,
                                        domain_footprint, rotate_footprint)
from woof.static.highres_production import (HighresRefusal,
                                             HighresStaticConfig, _apply,
                                             apply_highres_statics)
from woof.static.lambert import LambertGrid
from woof.static.projection import MercatorGrid

HIGHRES_FIXTURES = (Path(__file__).resolve().parent.parent
                    / "tools" / "rustwx" / "crates" / "static-fields"
                    / "tests" / "fixtures" / "highres")
MODIS21_ATTRS = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
                 "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}
#: The clips meet at 8 E; these moves put that seam on 180 degrees.
WEST_MOVE, EAST_MOVE = 172.0, -188.0


def _fixture(name: str) -> Path:
    path = HIGHRES_FIXTURES / name
    if not path.is_file():
        pytest.skip(f"{name} is not in this checkout")
    return path


def _fetched(path: Path) -> FetchedFile:
    return FetchedFile(path=path, url=f"fixture:{path.name}",
                       sha256=sha256_file(path), bytes=path.stat().st_size,
                       fetched_utc="", cache_hit=True)


def _moved(name: str, delta: float, folder: Path) -> FetchedFile:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    shutil.copyfile(_fixture(name), target)
    _shift_geotiff_longitude(target, delta)
    return _fetched(target)


def _grid(ref_lon: float) -> LambertGrid:
    """An 11 x 11 cell, 100 m grid over the clips' seam."""
    return LambertGrid(ref_lat=46.459, ref_lon=ref_lon, truelat1=30.0,
                       truelat2=60.0, stand_lon=ref_lon, dx=100.0, dy=100.0,
                       e_we=12, e_sn=12)


def _baseline(n: int = 11) -> dict:
    return {"HGT_M": np.full((n, n), 500.0),
            "LU_INDEX": np.full((n, n), 10.0),
            "LANDMASK": np.ones((n, n)),
            "SOILTEMP": np.full((n, n), 285.0)}


def _terrain(grid, tiles, cache_root: Path, monkeypatch) -> np.ndarray:
    monkeypatch.setattr(
        "woof.static.highres_production.fetch_copernicus_dem_tiles",
        lambda bbox, cache_root, urlopen=None: (tuple(tiles), ()))
    config = HighresStaticConfig(enabled=True, cache_root=cache_root,
                                 fields="terrain",
                                 terrain_source="copernicus-dem-glo30")
    # The overlay itself (production's _apply), without the terrain slope
    # bound apply_highres_statics runs after it: these are 100 m Alpine
    # clips, steeper than any dynamics this grid would be run with.
    fields, detail = _apply(
        _baseline(), grid, config=config, case_date=date(2026, 11, 12),
        landuse_attrs=MODIS21_ATTRS)
    rotation = detail["fetch"].get("terrain_window_rotation")
    return np.asarray(fields["HGT_M"], dtype=np.float64), rotation


def test_moving_a_geotiff_moves_its_georeferencing_and_nothing_else(
        tmp_path):
    source = _fixture("mosaic_west.tif")
    target = tmp_path / source.name
    shutil.copyfile(source, target)
    before = target.read_bytes()
    west = _geotiff_west(target)

    assert _shift_geotiff_longitude(target, WEST_MOVE) >= 1
    after = target.read_bytes()
    assert _geotiff_west(target) == pytest.approx(west + WEST_MOVE, abs=1e-9)
    assert len(after) == len(before)
    changed = sum(a != b for a, b in zip(before, after))
    assert 0 < changed <= 16, f"{changed} bytes changed"


def test_a_dateline_footprint_is_turned_into_the_cut_frame():
    on_line = FootprintBBox(2.12, 18.35, 176.68, 196.95)
    assert dateline_rotation_deg(on_line) == 180.0
    turned = rotate_footprint(on_line, 180.0)
    assert -180.0 < turned.lon_min < turned.lon_max < 180.0
    assert turned.lon_max - turned.lon_min == pytest.approx(20.27)
    # Off the line, and the whole band, are not turned.
    assert dateline_rotation_deg(FootprintBBox(30, 40, -100, -90)) == 0.0
    assert dateline_rotation_deg(FootprintBBox(-90, -60, -183, 183)) == 0.0


def test_the_storm_box_footprint_is_continued_and_turnable():
    """The gate's own box: the 12 km Mercator parent on 9.8 N 173.4 W."""
    parent = MercatorGrid(ref_lat=9.8, ref_lon=-173.4, truelat1=9.8,
                          truelat2=9.8, stand_lon=-173.4, dx=12000.0,
                          dy=12000.0, e_we=201, e_sn=161)
    bbox = domain_footprint(parent, 3)
    assert bbox.lon_min < 180.0 < bbox.lon_max
    assert dateline_rotation_deg(bbox) == 180.0


def test_a_grid_on_the_line_gets_the_terrain_its_twin_gets_off_it(
        tmp_path, monkeypatch):
    if rust_bridge.unavailable_reason() is not None:
        pytest.fail("the Rust static-fields bridge is the default terrain "
                    f"door and must load: {rust_bridge.unavailable_reason()}")
    published = [_fetched(_fixture("mosaic_west.tif")),
                 _fetched(_fixture("mosaic_east.tif"))]
    on_line = [_moved("mosaic_west.tif", WEST_MOVE, tmp_path / "w"),
               _moved("mosaic_east.tif", EAST_MOVE, tmp_path / "e")]

    twin_grid = _grid(8.002)
    line_grid = _grid(8.002 + WEST_MOVE - 360.0)   # -179.998
    footprint = domain_footprint(line_grid, 3)
    assert footprint.lon_min < 180.0 < footprint.lon_max

    twin, twin_turn = _terrain(twin_grid, published, tmp_path / "twin",
                               monkeypatch)
    line, line_turn = _terrain(line_grid, on_line, tmp_path / "line",
                               monkeypatch)

    assert twin_turn is None
    assert line_turn["longitude_turn_deg"] == 180.0
    assert np.isfinite(line).all()
    assert float(twin.max() - twin.min()) > 10.0, "real relief"
    assert not np.any(line == 500.0), "no cell kept the baseline"
    np.testing.assert_allclose(line, twin, rtol=0.0, atol=1e-6)


def test_fields_all_across_the_line_names_land_cover_not_terrain(tmp_path):
    grid = _grid(-179.998)

    def no_network(*args, **kwargs):
        raise AssertionError("no byte may be requested")

    config = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                 fields="all")
    with pytest.raises(HighresRefusal) as failure:
        apply_highres_statics(
            _baseline(), grid, config=config, domain_id=1,
            case_date=date(2026, 11, 12), landuse_attrs=MODIS21_ATTRS,
            urlopen=no_network)
    assert failure.value.reason == "dateline-window-unbuilt"
    assert "land cover and soil" in failure.value.detail
    assert "fields = \"auto\"" in failure.value.detail
