"""GMTED EAST OF 180 (woof.geog_assets.repair_gmted_east_of_180): the shipped GLO-30 patch is written into the
GMTED tiles once per geography root, interior and the neighbouring tiles' borders alike, only where the tiles store
0 m, and never under GPUWM_GEOG_SEAM_REPAIR=off or on another layout."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from woof import geog_assets

INDEX = """type = continuous
signed = yes
projection = regular_ll
dx = 0.00833333
dy = 0.00833333
known_x = 1.0
known_y = 1.0
known_lat = -89.99583
known_lon = 0.004166667
wordsize = 2
tile_x = 1200
tile_y = 1200
tile_z = 1
tile_bdr=3
units="meters MSL"
"""
SIDE = 1206


def _tile(root: Path, tx: int, ty: int, fill: int = 0) -> Path:
    path = root / geog_assets.GMTED_DATASET / (f"{tx * 1200 + 1:05d}-{tx * 1200 + 1200:05d}."
                                               f"{ty * 1200 + 1:05d}-{ty * 1200 + 1200:05d}")
    np.full((SIDE, SIDE), fill, ">i2").tofile(path)
    return path


def _read(path: Path) -> np.ndarray:
    return np.fromfile(path, ">i2").reshape(SIDE, SIDE)


@pytest.fixture
def geog(tmp_path, monkeypatch):
    d = tmp_path / geog_assets.GMTED_DATASET
    d.mkdir()
    (d / "index").write_text(INDEX)
    # Taveuni's tile (x 21601-22800, y 8401-9600) and its west neighbour, whose border holds x 21601-21603
    east, west = _tile(tmp_path, 18, 7), _tile(tmp_path, 17, 7, fill=5)
    patch = tmp_path / "patch.npz"
    # x 21601 (first column east of 180, in the west tile's border too), x 21700 (interior), one already non-zero
    np.savez_compressed(patch, x=np.array([21601, 21700, 21650], np.int32), y=np.array([8784, 8790, 8800], np.int32),
                        h=np.array([812, 1051, 300], np.int16))
    a = _read(east)
    a[8800 - 8401 + 3, 21650 - 21601 + 3] = 7          # the source has a height here: kept
    a.tofile(east)
    monkeypatch.setattr(geog_assets, "GMTED_E180_PATCH", patch)
    monkeypatch.delenv("GPUWM_GEOG_SEAM_REPAIR", raising=False)
    return tmp_path, east, west


def test_the_patch_lands_in_the_tile_and_the_neighbour_border_once(geog):
    root, east, west = geog
    sizes = (east.stat().st_size, west.stat().st_size)
    assert geog_assets.repair_gmted_east_of_180(root, progress=lambda s: None) == 2   # the west border word is 5
    a, w = _read(east), _read(west)
    assert a[8784 - 8401 + 3, 0 + 3] == 812
    assert a[8790 - 8401 + 3, 99 + 3] == 1051
    assert a[8800 - 8401 + 3, 49 + 3] == 7
    assert w[8784 - 8401 + 3, 1203] == 5             # x 21601 in the west tile's border: not 0 there, kept
    assert (east.stat().st_size, west.stat().st_size) == sizes                     # the install receipt still holds
    assert (root / geog_assets.GMTED_E180_MARK).is_file()
    assert geog_assets.repair_gmted_east_of_180(root, progress=lambda s: None) == 0  # once per root


def test_a_zero_border_word_in_the_neighbour_is_repaired_too(geog):
    root, east, west = geog
    _tile(root, 17, 7, fill=0)
    geog_assets.repair_gmted_east_of_180(root, progress=lambda s: None)
    assert _read(west)[8784 - 8401 + 3, 1203] == 812


def test_off_switch_and_other_layouts_read_as_stored(geog, monkeypatch):
    root, east, _ = geog
    before = _read(east).copy()
    monkeypatch.setenv("GPUWM_GEOG_SEAM_REPAIR", "off")
    assert geog_assets.repair_gmted_east_of_180(root, progress=lambda s: None) == 0
    monkeypatch.delenv("GPUWM_GEOG_SEAM_REPAIR")
    (root / geog_assets.GMTED_DATASET / "index").write_text(INDEX.replace("known_lon = 0.004166667",
                                                                          "known_lon = -179.99583"))
    assert geog_assets.repair_gmted_east_of_180(root, progress=lambda s: None) == 0
    assert (_read(east) == before).all()
    assert not (root / geog_assets.GMTED_E180_MARK).exists()


def test_the_shipped_patch_carries_taveuni():
    p = np.load(Path(geog_assets.__file__).resolve().parent / "data" / "geog_repairs" / "gmted2010_e180_glo30.npz")
    x, y, h = p["x"], p["y"], p["h"]
    assert ((x >= 21601) & (x <= 22800)).all() and (h > 0).all()
    lon = (x - 1) / 120.0 + 0.004166667 - 360.0
    lat = (y - 1) / 120.0 - 89.99583
    taveuni = (lat > -17.1) & (lat < -16.6) & (lon < -179.8)
    assert taveuni.sum() > 300 and h[taveuni].max() > 1000
