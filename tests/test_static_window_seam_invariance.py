"""Statics may not depend on the SOURCE WINDOW a build happened to read.

THE BREAKAGE THIS PREVENTS, measured on test host 1 2026-08-20: a prepared
moving-nest run refused every relocation with

    footprint-rebuilt statics differ from the outgoing child's on shared
    ground ... {'HGT_M': 10773, 'TMN': 65}

and produced ZERO wrfout frames on all four relocation arms.  The
outgoing child's statics were built on its own 198x198 footprint; the
rebuild cropped the parent-extent statics corridor.  Both cover the same
cells of the same geography, so both must give the same bytes -- but the
corridor's source window straddles the terrain dataset's x-wrap seam
(that WPS_GEOG terrain tree starts at longitude 0.0042, and the corridor
reaches west of Greenwich) while the footprint's window does not.

The mechanism: ``cell_coords`` used to shift a point into the window's
frame (``xi + nx_global`` when ``xi < win.x0 - 0.5``) and the tile
interpolator shifted it straight back (``xx - nx_global``).  In float64
that round trip is LOSSY -- 567.121676837268 + 43200 - 43200 comes back
567.1216768372697 -- so the same cell of the same geography sampled the
source at coordinates that differed in their last bits, and the
interpolated terrain differed by up to 2.8e-11 m in 11581 of 39204
shared cells.  Whether the shift fires at all depends on the window,
which depends on the extent, which is why two builds of the same ground
disagreed.

These tests build the same footprint twice -- once directly, once as a
crop of a wider grid on the same lattice whose window crosses the seam
-- and demand byte equality, on the Rust default route and on the
``WOOF_STATIC_PYTHON=1`` numpy route alike.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from woof.static.build import _DEFAULT_GEOG_DIRS
import pytest

from woof.static.build import GeogSelection, build_static
from woof.static.lambert import LambertGrid


# ---------------------------------------------------------------------------
# A synthetic WPS_GEOG whose datasets wrap in x with the seam INSIDE the
# domain: global regular_ll geometry, tiles staged only where the domain
# reads (the shape every real 30s tree has around longitude zero).
# ---------------------------------------------------------------------------

def _write_index(dirpath: Path, kv: dict) -> dict:
    dirpath.mkdir(parents=True, exist_ok=True)
    lines = [f"{key} = {value}" for key, value in kv.items()
             if value is not None]
    (dirpath / "index").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return kv


def _dtype(kv: dict) -> np.dtype:
    ws = int(kv["wordsize"])
    base = {1: "i1", 2: "i2", 4: "i4"}[ws]
    if str(kv.get("signed", "no")).lower() not in ("yes", "true", ".true."):
        base = "u" + base[1:]
    return np.dtype(">" + base) if ws > 1 else np.dtype(base)


def _stage(dirpath: Path, kv: dict, origins, values) -> None:
    """Write the staged tiles; ``values(x, y, z)`` takes GLOBAL indices."""
    tx, ty = int(kv["tile_x"]), int(kv["tile_y"])
    nz = int(kv.get("tile_z", 1))
    dt = _dtype(kv)
    for xs, ys in origins:
        z, y, x = np.meshgrid(np.arange(nz),
                              np.arange(ys, ys + ty),
                              np.arange(xs, xs + tx), indexing="ij")
        name = (f"{xs:05d}-{xs + tx - 1:05d}."
                f"{ys:05d}-{ys + ty - 1:05d}")
        values(x, y, z).astype(dt).tofile(dirpath / name)


#: 0.02 deg sources: 18000x9000 global, x = 1 at longitude 0.01, tiles
#: staged either side of the seam.  The domain sits just east of it.
_FINE = dict(projection="regular_ll", dx=0.02, dy=0.02,
             known_x=1.0, known_y=1.0, known_lat=-89.99, known_lon=0.01,
             tile_x=200, tile_y=200)
_FINE_ORIGINS = ((1, 5801), (1, 6001), (17801, 5801), (17801, 6001))

#: 0.1 deg sources: 3600x1800 global, same seam story, coarser.
_COARSE = dict(projection="regular_ll", dx=0.1, dy=0.1,
               known_x=1.0, known_y=1.0, known_lat=-89.95, known_lon=0.05,
               tile_x=60, tile_y=60)
_COARSE_ORIGINS = ((1, 1141), (1, 1201), (3541, 1141), (3541, 1201))


def _seam_wps_geog(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    # terrain: fine, continuous, signed metres, strongly varying so a
    # last-bit coordinate difference moves the interpolated value.
    kv = _write_index(root / "topo_gmted2010_30s",
                      dict(_FINE, type="continuous", signed="yes",
                           wordsize=2, tile_z=1, units='"meters MSL"'))
    _stage(root / "topo_gmted2010_30s", kv, _FINE_ORIGINS,
           lambda x, y, z: (37 * x + 91 * y) % 1500 + 3 * ((x * y) % 7))

    # landuse: fine, categorical, with water blocks (category 17).
    kv = _write_index(root / "modis_landuse_20class_30s_with_lakes",
                      dict(_FINE, type="categorical", category_min=1,
                           category_max=21, wordsize=1, tile_z=1,
                           mminlu='"MODIFIED_IGBP_MODIS_NOAH"',
                           iswater=17, islake=21, isice=15, isurban=13))
    _stage(root / "modis_landuse_20class_30s_with_lakes", kv, _FINE_ORIGINS,
           lambda x, y, z: np.where(((x // 37) + (y // 41)) % 5 == 0, 17,
                                    1 + (3 * x + 5 * y) % 21))

    for name in (_DEFAULT_GEOG_DIRS["soil_top"], _DEFAULT_GEOG_DIRS["soil_bottom"]):
        kv = _write_index(root / name,
                          dict(_COARSE, type="categorical", category_min=1,
                               category_max=16, wordsize=1, tile_z=1))
        _stage(root / name, kv, _COARSE_ORIGINS,
               lambda x, y, z: 1 + (2 * x + y) % 16)

    for name, nz, scale, span in (("greenfrac_fpar_modis", 12, 0.01, 100),
                                  ("lai_modis_10m", 12, 0.1, 60),
                                  ("albedo_modis", 12, 1.0, 30),
                                  ("maxsnowalb_modis", 1, 1.0, 80),
                                  (_DEFAULT_GEOG_DIRS["soil_temperature"], 1, 1.0, 300)):
        kv = _write_index(root / name,
                          dict(_COARSE, type="continuous", signed="yes",
                               wordsize=2, tile_z=nz, scale_factor=scale))
        _stage(root / name, kv, _COARSE_ORIGINS,
               lambda x, y, z, span=span: (x + 2 * y + 5 * z) % span + 1)
    return root


@pytest.fixture(scope="module")
def seam_geog(tmp_path_factory) -> Path:
    return _seam_wps_geog(tmp_path_factory.mktemp("wps-geog-seam"))


#: The footprint: 40x40 cells of 2 km centred at 0.75E, entirely east of
#: the terrain tree's x = 1 seam, so ITS window never wraps.
_NX = 40
_WEST_CELLS = 60


def _footprint_grid() -> LambertGrid:
    return LambertGrid(ref_lat=30.0, ref_lon=0.75, truelat1=30.0,
                       truelat2=30.0, stand_lon=0.75,
                       dx=2000.0, dy=2000.0, e_we=_NX + 1, e_sn=_NX + 1)


def _wide_grid() -> LambertGrid:
    """The same lattice, reaching 120 km further west -- across the seam."""
    return _footprint_grid().translated(
        -_WEST_CELLS, 0, e_we=_NX + _WEST_CELLS + 1, e_sn=_NX + 1)


def _crop(field: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(field)[..., :, _WEST_CELLS:_WEST_CELLS + _NX])


def _assert_windows_straddle(seam_geog: Path) -> None:
    """The fixture must actually pose the question (validate the instrument).

    A footprint window that also wrapped, or a wide window that did not,
    would make the equality below pass for the wrong reason.
    """
    from woof.static.build import GeogDataset, _DomainSampler

    topo = GeogDataset(GeogSelection.fallback(seam_geog).path("terrain"))
    assert topo.wraps_x, "the terrain fixture must be a wrapping global source"
    narrow = _DomainSampler(_footprint_grid(), 3).window(topo)
    wide = _DomainSampler(_wide_grid(), 3).window(topo)
    assert narrow.x0 >= 1 and narrow.x1 < topo.nx_global, (
        f"the footprint window {narrow.x0}..{narrow.x1} was expected to sit "
        "east of the seam")
    assert wide.x1 > topo.nx_global > wide.x0, (
        f"the wide window {wide.x0}..{wide.x1} was expected to cross the "
        f"x = {topo.nx_global} seam")


@pytest.mark.parametrize("route", ("rust-default", "python-fallback"))
def test_same_ground_same_bytes_across_a_seam_crossing_window(
        seam_geog, monkeypatch, route):
    """Identical source + identical cells = identical bytes, whatever
    window the build read them through."""
    from woof.static import rust_bridge

    _assert_windows_straddle(seam_geog)
    if route == "python-fallback":
        monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
    else:
        monkeypatch.delenv(rust_bridge.STATIC_PYTHON_ENV, raising=False)
        reason = rust_bridge.unavailable_reason()
        if reason is not None:
            pytest.fail(
                "the Rust static-fields bridge is not loadable, so the "
                f"DEFAULT statics route cannot be gated here: {reason}")

    selection = GeogSelection.fallback(seam_geog)
    direct = build_static(_footprint_grid(), seam_geog, selection=selection)
    wide = build_static(_wide_grid(), seam_geog, selection=selection)

    assert sorted(direct) == sorted(wide)
    for name in sorted(direct):
        expected = np.asarray(direct[name])
        actual = _crop(wide[name])
        assert actual.shape == expected.shape, name
        unequal = int(np.count_nonzero(actual != expected))
        assert unequal == 0, (
            f"{name} differs in {unequal} of {expected.size} cells between "
            f"two builds of the same ground (max |delta| "
            f"{float(np.abs(actual - expected).max()):.3e}); the source "
            "window is not part of the answer")


# ---------------------------------------------------------------------------
# The same claim against a source shaped like a REAL WPS_GEOG tree: its
# declared dx is a TRUNCATED decimal, so nx_global * dx is NOT 360 and an
# unfolded source column names a different PLACE, not a rounded copy of the
# same one.  The fixture above uses dx = 0.02 with 18000 columns, whose
# product is 360 to within a float64 rounding, so it cannot pose this
# question at all, which is why the defect below survived it.
#
# THE BREAKAGE THIS PREVENTS, measured 2026-09-15 on the Bering Sea 12/3 km
# cyclone run (polar stereographic, parent spanning the dateline, 3 km
# follower on the sealed statics corridor): the first relocation at model
# second 900 refused with
#
#     footprint-rebuilt statics differ from the outgoing child's on shared
#     ground ... {'LU_INDEX': 2, 'LANDUSEF': 542, 'SOILCTOP': 72,
#                 'SCT_DOM': 2, 'SOILCBOT': 72, 'SCB_DOM': 2}
#
# because the corridor's source window crosses the antimeridian and the
# child footprint's does not.  read_window resolves that wrap and hands
# back the CANONICAL column's bytes, but the pixel binner kept the
# unwrapped index, so it asked for the longitude of column 43861 while
# holding column 661's value: 1.44e-4 deg apart on a 30-arcsec tree, about
# 7.4 m, enough to bin a boundary pixel into the neighbouring destination
# cell.
# ---------------------------------------------------------------------------

#: 0.0333333 deg: 10800x5400 global, column 1 ON the dateline.  The
#: truncation leaves 360 - 10800*dx = 3.6e-4 deg unaccounted for.
_TRUNC = dict(projection="regular_ll", dx=0.0333333, dy=0.0333333,
              known_x=1.0, known_y=1.0,
              known_lat=-89.98333, known_lon=-179.98333,
              tile_x=100, tile_y=100)

#: 0.1 deg: 3600x1800 global, same seam, for the monthly climatologies.
_TRUNC_COARSE = dict(projection="regular_ll", dx=0.1, dy=0.1,
                     known_x=1.0, known_y=1.0,
                     known_lat=-89.95, known_lon=-179.95,
                     tile_x=60, tile_y=60)

_D_RATIO = 4
_D_PARENT_NX, _D_PARENT_NY = 30, 10
_D_CHILD_N = 30
_D_REF_I, _D_REF_J = 22, 3          # reference placement, east of the seam
_D_ALT_I, _D_ALT_J = 19, 3          # overlapping placement, west of it


def _dateline_parent_grid() -> LambertGrid:
    return LambertGrid(ref_lat=20.0, ref_lon=-178.0, truelat1=20.0,
                       truelat2=20.0, stand_lon=-178.0,
                       dx=80000.0, dy=80000.0,
                       e_we=_D_PARENT_NX + 1, e_sn=_D_PARENT_NY + 1)


def _dateline_reference_grid() -> LambertGrid:
    return _dateline_parent_grid().nest(
        _D_REF_I, _D_REF_J, _D_RATIO, _D_CHILD_N + 1, _D_CHILD_N + 1)


def _child_dc(i_parent_start=_D_REF_I, j_parent_start=_D_REF_J):
    return SimpleNamespace(
        grid_id=2, parent_id=1, parent_grid_ratio=_D_RATIO,
        i_parent_start=i_parent_start, j_parent_start=j_parent_start,
        run=SimpleNamespace(nx=_D_CHILD_N, ny=_D_CHILD_N))


def _parent_run():
    return SimpleNamespace(nx=_D_PARENT_NX, ny=_D_PARENT_NY)


def _source_span(kv: dict, grid) -> tuple[range, range]:
    """Source columns/rows the grid needs, wrap resolved, margin included."""
    nx_global = round(360.0 / kv["dx"])
    ny_global = round(180.0 / abs(kv["dy"]))
    x = np.linspace(-4.0, grid.e_we + 4.0, 60)
    y = np.linspace(-4.0, grid.e_sn + 4.0, 60)
    lat, lon = grid.ij_to_latlon(*np.meshgrid(x, y))
    xs = 1.0 + np.mod(np.asarray(lon, dtype=np.float64) - kv["known_lon"],
                      360.0) / kv["dx"]
    ys = 1.0 + (np.asarray(lat, dtype=np.float64)
                - kv["known_lat"]) / kv["dy"]
    if float(xs.max() - xs.min()) > nx_global / 2.0:
        xs = np.where(xs < nx_global / 2.0, xs + nx_global, xs)
    margin = 8
    return (range(int(np.floor(xs.min())) - margin,
                  int(np.ceil(xs.max())) + margin + 1),
            range(max(1, int(np.floor(ys.min())) - margin),
                  min(ny_global, int(np.ceil(ys.max())) + margin) + 1))


def _stage_span(dirpath: Path, kv: dict, grids, values) -> None:
    """Write every tile the grids read; values(x, y, z) takes GLOBAL
    (canonical) indices."""
    nx_global = round(360.0 / kv["dx"])
    tx, ty = int(kv["tile_x"]), int(kv["tile_y"])
    nz = int(kv.get("tile_z", 1))
    dt = _dtype(kv)
    origins = set()
    for grid in grids:
        cols, rows = _source_span(kv, grid)
        for column in cols:
            canonical = (column - 1) % nx_global + 1
            for row in rows:
                origins.add(((canonical - 1) // tx * tx + 1,
                             (row - 1) // ty * ty + 1))
    for xs, ys in sorted(origins):
        z, y, x = np.meshgrid(np.arange(nz),
                              np.arange(ys, ys + ty),
                              np.arange(xs, xs + tx), indexing="ij")
        name = (f"{xs:05d}-{xs + tx - 1:05d}."
                f"{ys:05d}-{ys + ty - 1:05d}")
        values(x, y, z).astype(dt).tofile(dirpath / name)


def _dateline_wps_geog(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    corridor = _dateline_reference_grid().translated(
        -(_D_REF_I - 1) * _D_RATIO, -(_D_REF_J - 1) * _D_RATIO,
        e_we=_D_PARENT_NX * _D_RATIO + 1, e_sn=_D_PARENT_NY * _D_RATIO + 1)
    grids = [corridor, _dateline_reference_grid(),
             _dateline_reference_grid().translated(
                 (_D_ALT_I - _D_REF_I) * _D_RATIO,
                 (_D_ALT_J - _D_REF_J) * _D_RATIO)]

    kv = _write_index(root / "topo_gmted2010_30s",
                      dict(_TRUNC, type="continuous", signed="yes",
                           wordsize=2, tile_z=1, units='"meters MSL"'))
    _stage_span(root / "topo_gmted2010_30s", kv, grids,
                lambda x, y, z: (29 * x + 71 * y) % 2100 + 5 * ((x * y) % 11))

    kv = _write_index(root / "modis_landuse_20class_30s_with_lakes",
                      dict(_TRUNC, type="categorical", category_min=1,
                           category_max=21, wordsize=1, tile_z=1,
                           mminlu='"MODIFIED_IGBP_MODIS_NOAH"',
                           iswater=17, islake=21, isice=15, isurban=13))
    _stage_span(root / "modis_landuse_20class_30s_with_lakes", kv, grids,
                lambda x, y, z: np.where(((x // 23) + (y // 29)) % 4 == 0, 17,
                                         1 + (7 * x + 3 * y) % 21))

    for name in (_DEFAULT_GEOG_DIRS["soil_top"], _DEFAULT_GEOG_DIRS["soil_bottom"]):
        kv = _write_index(root / name,
                          dict(_TRUNC, type="categorical", category_min=1,
                               category_max=16, wordsize=1, tile_z=1))
        _stage_span(root / name, kv, grids,
                    lambda x, y, z: 1 + (5 * x + 3 * y) % 16)

    for name, nz, scale, span in (("greenfrac_fpar_modis", 12, 0.01, 100),
                                  ("lai_modis_10m", 12, 0.1, 60),
                                  ("albedo_modis", 12, 1.0, 30),
                                  ("maxsnowalb_modis", 1, 1.0, 80),
                                  (_DEFAULT_GEOG_DIRS["soil_temperature"], 1, 1.0, 300)):
        kv = _write_index(root / name,
                          dict(_TRUNC_COARSE, type="continuous", signed="yes",
                               wordsize=2, tile_z=nz, scale_factor=scale))
        _stage_span(root / name, kv, grids,
                    lambda x, y, z, span=span: (x + 2 * y + 5 * z) % span + 1)
    return root


@pytest.fixture(scope="module")
def dateline_geog(tmp_path_factory) -> Path:
    return _dateline_wps_geog(tmp_path_factory.mktemp("wps-geog-dateline"))


def _dateline_corridor(geog_root: Path):
    """The sealed corridor for this child, built exactly as preparation
    builds it (parent-extent grid on the child's own lattice)."""
    from woof.static.corridor import (ChildStaticsCorridor, corridor_geometry,
                                       corridor_grid)

    geometry = corridor_geometry(_child_dc(), _parent_run())
    grid = corridor_grid(_dateline_reference_grid(), geometry)
    fields = build_static(grid, geog_root,
                          selection=GeogSelection.fallback(geog_root))
    return ChildStaticsCorridor(geometry=geometry, fields=fields,
                                cache_sha256="0" * 64)


def _assert_dateline_instrument(geog_root: Path) -> None:
    """The fixture must actually pose the question.

    Four facts, each of which the defect needs and any of which, if the
    fixture lost it, would make the equality below pass for free: the
    landuse source wraps and its declared dx does NOT tile 360 degrees, the
    corridor's window crosses the seam while staying narrower than the
    globe, the reference footprint's window does not cross it, and a source
    tile edge falls inside that footprint.
    """
    from woof.static.build import GeogDataset, _DomainSampler
    from woof.static.corridor import corridor_geometry, corridor_grid

    lu = GeogDataset(GeogSelection.fallback(geog_root).path("landuse"))
    assert lu.wraps_x, "the landuse fixture must be a wrapping global source"
    deficit = abs(lu.index.dx * lu.nx_global - 360.0)
    assert 1e-5 < deficit < 1e-3, (
        f"the fixture's declared dx tiles 360 deg to within {deficit:.3e}; a "
        "source whose dx divides 360 cannot show an unfolded column as a "
        "different place, which is how the real defect hid")

    footprint = _DomainSampler(_dateline_reference_grid(), 3).window(lu)
    geometry = corridor_geometry(_child_dc(), _parent_run())
    corridor = _DomainSampler(
        corridor_grid(_dateline_reference_grid(), geometry), 3).window(lu)
    assert footprint.x0 >= 1 and footprint.x1 <= lu.nx_global, (
        f"the footprint window {footprint.x0}..{footprint.x1} was expected "
        f"to sit east of the x = {lu.nx_global} seam")
    assert corridor.x1 > lu.nx_global >= corridor.x0, (
        f"the corridor window {corridor.x0}..{corridor.x1} was expected to "
        f"cross the x = {lu.nx_global} seam")
    assert corridor.raw.shape[2] < lu.nx_global, (
        "the corridor window must be NARROWER than the global axis; a wider "
        "one takes the already-folded branch and asks nothing")

    tile_x = int(lu.index.tile_x)
    edges = [edge for edge in range(1, lu.nx_global + 1, tile_x)
             if footprint.x0 < edge < footprint.x1]
    assert edges, (
        f"no source tile edge falls inside the footprint window "
        f"{footprint.x0}..{footprint.x1}; the crop would not be tested at a "
        "tile boundary")


@pytest.mark.parametrize("route", ("rust-default", "python-fallback"))
def test_corridor_crop_equals_direct_build_across_the_dateline(
        dateline_geog, monkeypatch, route):
    """Two overlapping footprints of one sealed corridor, each equal to the
    build made directly for that placement, and equal to each other on the
    ground they share.

    This is the equality woof.ingest.relocation_continuation asserts at
    every move, posed on the geometry that broke it: the outgoing child's
    statics come from its own footprint build and the incoming child's from
    a crop of the parent-extent corridor.
    """
    from woof.static import rust_bridge

    _assert_dateline_instrument(dateline_geog)
    if route == "python-fallback":
        monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
    else:
        monkeypatch.delenv(rust_bridge.STATIC_PYTHON_ENV, raising=False)
        reason = rust_bridge.unavailable_reason()
        if reason is not None:
            pytest.fail(
                "the Rust static-fields bridge is not loadable, so the "
                f"DEFAULT statics route cannot be gated here: {reason}")

    selection = GeogSelection.fallback(dateline_geog)
    corridor = _dateline_corridor(dateline_geog)
    reference = _dateline_reference_grid()

    crops = {}
    for ip, jp in ((_D_REF_I, _D_REF_J), (_D_ALT_I, _D_ALT_J)):
        direct = build_static(
            reference.translated((ip - _D_REF_I) * _D_RATIO,
                                 (jp - _D_REF_J) * _D_RATIO),
            dateline_geog, selection=selection)
        crop = crops.setdefault((ip, jp), corridor.crop(ip, jp))
        assert sorted(crop) == sorted(direct)
        for name in sorted(direct):
            expected = np.asarray(direct[name])
            actual = np.asarray(crop[name])
            assert actual.shape == expected.shape, name
            unequal = int(np.count_nonzero(actual != expected))
            assert unequal == 0, (
                f"{name} differs in {unequal} of {expected.size} cells "
                f"between the corridor crop at ({ip}, {jp}) and the build "
                f"made directly for that placement (max |delta| "
                f"{float(np.abs(actual - expected).max()):.3e}); identical "
                "source + identical cells must give identical bytes")

    # ... and therefore on the ground the two placements share.
    shift = (_D_REF_I - _D_ALT_I) * _D_RATIO
    assert 0 < shift < _D_CHILD_N, "the two placements must overlap"
    first = crops[(_D_REF_I, _D_REF_J)]
    second = crops[(_D_ALT_I, _D_ALT_J)]
    for name in sorted(first):
        a = np.asarray(first[name])[..., :, :_D_CHILD_N - shift]
        b = np.asarray(second[name])[..., :, shift:]
        assert a.tobytes() == b.tobytes(), (
            f"{name} differs between two overlapping footprints on their "
            "shared ground")
