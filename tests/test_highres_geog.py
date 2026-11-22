"""Focused gates for provenance-bound high-resolution geography."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from woof.static.highres import (
    BoundRaster,
    _raster_geometry,
    merge_highres_overrides,
    resample_continuous,
    resample_mapped_categories,
    sha256_file,
    usda_texture_category,
)
from woof.static.lambert import LambertGrid


def _grid() -> LambertGrid:
    return LambertGrid(
        ref_lat=39.66, ref_lon=-84.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-84.0, dx=500.0, dy=500.0, e_we=9, e_sn=9,
    )


def _write_raster(path: Path, values: np.ndarray) -> None:
    rasterio = pytest.importorskip("rasterio")
    transform = rasterio.transform.from_bounds(
        -84.1, 39.57, -83.9, 39.75, values.shape[1], values.shape[0])
    with rasterio.open(
            path, "w", driver="GTiff", height=values.shape[0],
            width=values.shape[1], count=1, dtype=values.dtype,
            crs="EPSG:4326", transform=transform) as dataset:
        dataset.write(values, 1)


def _bound(path: Path, *, role: str = "test") -> BoundRaster:
    return BoundRaster(
        path=path, sha256=sha256_file(path), source_id="synthetic",
        role=role, source_url="https://example.invalid/source",
        license_id="test-only", license_url="https://example.invalid/license",
        nominal_resolution="test",
    )


@pytest.mark.parametrize("mismatch", ["hash", "size"])
def test_bound_raster_rejects_mismatch_before_decode(tmp_path, mismatch):
    path = tmp_path / "source.bin"
    path.write_bytes(b"payload")
    source = BoundRaster(
        path=path, sha256="0" * 64 if mismatch == "hash" else sha256_file(path),
        source_id="synthetic", role="terrain",
        source_url="https://example.invalid/source", license_id="test-only",
        license_url="https://example.invalid/license", nominal_resolution="test",
        expected_bytes=999 if mismatch == "size" else None,
    )
    with pytest.raises(ValueError, match=f"{mismatch} mismatch"):
        source.verify()


def test_raster_geometry_matches_wps_lambert_mass_points():
    pyproj = pytest.importorskip("pyproj")
    grid = _grid()
    crs, transform, (ny, nx) = _raster_geometry(grid)
    columns, rows = np.meshgrid(
        np.arange(nx, dtype=np.float64) + 0.5,
        np.arange(ny, dtype=np.float64) + 0.5,
    )
    x, y = transform * (columns, rows)
    longitude, latitude = pyproj.Transformer.from_crs(
        crs, "EPSG:4326", always_xy=True).transform(x, y)
    expected_latitude, expected_longitude = grid.latlon_mass()
    np.testing.assert_allclose(latitude[::-1], expected_latitude, atol=2.0e-12)
    np.testing.assert_allclose(longitude[::-1], expected_longitude, atol=2.0e-12)


def test_lambert_resampling_preserves_constant_and_category_fractions(tmp_path):
    terrain_path = tmp_path / "terrain.tif"
    _write_raster(terrain_path, np.full((180, 200), 247.5, dtype=np.float32))
    terrain = resample_continuous(_bound(terrain_path), _grid())
    np.testing.assert_allclose(terrain, 247.5, rtol=0.0, atol=1.0e-9)

    categories = np.full((180, 200), 82, dtype=np.uint8)
    categories[:, :100] = 11
    category_path = tmp_path / "landcover.tif"
    _write_raster(category_path, categories)
    fractions = resample_mapped_categories(
        _bound(category_path), _grid(), {11: 21, 82: 12},
        category_count=21,
    )
    assert fractions.shape == (21, 8, 8)
    np.testing.assert_allclose(fractions.sum(axis=0), 1.0, atol=1.0e-6)
    assert np.any(fractions[20] > 0.99)
    assert np.any(fractions[11] > 0.99)


def test_usda_texture_rules_cover_all_wrf_mineral_categories():
    samples = np.asarray([
        [92, 5, 3], [82, 12, 6], [65, 25, 10], [20, 65, 15],
        [5, 90, 5], [40, 40, 20], [55, 15, 30], [10, 55, 35],
        [35, 30, 35], [52, 8, 40], [10, 47, 43], [30, 25, 45],
    ], dtype=np.float64)
    actual = usda_texture_category(
        samples[:, 0][None, :], samples[:, 1][None, :],
        samples[:, 2][None, :],
    )[0]
    np.testing.assert_array_equal(actual, np.arange(1, 13))


def test_merge_highres_overrides_keeps_complete_noah_state():
    shape = (4, 4)
    old_land = np.ones(shape, dtype=np.float64)
    old_land[0, 0] = 0.0
    baseline = {
        "HGT_M": np.full(shape, 100.0),
        "LANDMASK": old_land,
        "LU_INDEX": np.where(old_land > 0.5, 12.0, 21.0),
        "LANDUSEF": np.zeros((21, *shape)),
        "SOILCTOP": np.zeros((16, *shape)),
        "SCT_DOM": np.full(shape, 6.0),
        "SOILCBOT": np.zeros((16, *shape)),
        "SCB_DOM": np.full(shape, 6.0),
        "GREENFRAC": np.full((12, *shape), 0.5),
        "LAI12M": np.full((12, *shape), 2.0),
        "ALBEDO12M": np.full((12, *shape), 15.0),
        "SNOALB": np.full(shape, 0.4),
        "SOILTEMP": np.full(shape, 285.0),
        "TMN": np.full(shape, 284.35),
    }
    for name in ("LANDUSEF", "SOILCTOP", "SOILCBOT"):
        baseline[name][0] = 1.0
    new_land = old_land.copy()
    new_land[0, 0] = 1.0
    new_land[3, 3] = 0.0
    overrides = {
        "HGT_M": np.full(shape, 120.0),
        "LANDMASK": new_land,
        "LU_INDEX": np.where(new_land > 0.5, 12.0, 21.0),
        "LANDUSEF": baseline["LANDUSEF"],
        "SOILCTOP": baseline["SOILCTOP"],
        "SCT_DOM": baseline["SCT_DOM"],
        "SOILCBOT": baseline["SOILCBOT"],
        "SCB_DOM": baseline["SCB_DOM"],
    }
    merged, audit = merge_highres_overrides(baseline, overrides)
    assert audit["newly_land_nearest_climatology_fallback_cells"] == 1
    assert audit["newly_water_masked_cells"] == 1
    assert merged["SOILTEMP"][0, 0] == 285.0
    assert merged["SOILTEMP"][3, 3] == 0.0
    assert merged["ALBEDO12M"][0, 3, 3] == 8.0
    assert merged["TMN"][0, 0] == pytest.approx(285.0 - 0.0065 * 120.0)
    assert all(np.isfinite(value).all() for value in merged.values())


def _merge_inputs(shape=(4, 4), *, soiltemp=288.0):
    """Baseline/override pair with one newly-land and one newly-water cell."""
    old_land = np.ones(shape)
    old_land[0, 0] = 0.0
    baseline = {
        "HGT_M": np.full(shape, 100.0),
        "LANDMASK": old_land,
        "LU_INDEX": np.where(old_land > 0.5, 12.0, 21.0),
        "LANDUSEF": np.zeros((21, *shape)),
        "SOILCTOP": np.zeros((16, *shape)),
        "SCT_DOM": np.full(shape, 6.0),
        "SOILCBOT": np.zeros((16, *shape)),
        "SCB_DOM": np.full(shape, 6.0),
        "GREENFRAC": np.full((12, *shape), 0.5),
        "LAI12M": np.full((12, *shape), 2.0),
        "ALBEDO12M": np.full((12, *shape), 15.0),
        "SNOALB": np.full(shape, 0.4),
        "SOILTEMP": np.full(shape, soiltemp),
        "TMN": np.full(shape, soiltemp - 0.65),
    }
    for name in ("LANDUSEF", "SOILCTOP", "SOILCBOT"):
        baseline[name][0] = 1.0
    new_land = old_land.copy()
    new_land[0, 0] = 1.0
    new_land[3, 3] = 0.0
    overrides = {name: baseline[name] for name in
                 ("LANDUSEF", "SOILCTOP", "SCT_DOM", "SOILCBOT", "SCB_DOM")}
    overrides["HGT_M"] = np.full(shape, 100.0)
    overrides["LANDMASK"] = new_land
    overrides["LU_INDEX"] = np.where(new_land > 0.5, 12.0, 21.0)
    return baseline, overrides, new_land


def _sea_inputs(shape=(6, 6)):
    """A fine grid over open sea: the baseline has no land cell.

    Water as geogrid writes it: LU_INDEX 17 (MODIS water), the masked
    climatologies at their water fills (ALBEDO12M 8, SOILTEMP 0 K).
    """
    water = np.zeros(shape)
    luf = np.zeros((21, *shape))
    luf[16] = 1.0
    soil = np.zeros((16, *shape))
    soil[13] = 1.0
    baseline = {
        "HGT_M": np.zeros(shape),
        "LANDMASK": water,
        "LU_INDEX": np.full(shape, 17.0),
        "LANDUSEF": luf,
        "SOILCTOP": soil,
        "SCT_DOM": np.full(shape, 14.0),
        "SOILCBOT": soil.copy(),
        "SCB_DOM": np.full(shape, 14.0),
        "GREENFRAC": np.zeros((12, *shape)),
        "LAI12M": np.zeros((12, *shape)),
        "ALBEDO12M": np.full((12, *shape), 8.0),
        "SNOALB": np.zeros(shape),
        "SOILTEMP": np.zeros(shape),
        "TMN": np.zeros(shape),
    }
    overrides = {name: np.array(baseline[name], copy=True) for name in
                 ("LANDUSEF", "LANDMASK", "LU_INDEX", "SOILCTOP", "SCT_DOM",
                  "SOILCBOT", "SCB_DOM")}
    overrides["HGT_M"] = np.full(shape, -0.25)  # sea-level terrain source
    return baseline, overrides


def _with_islet(baseline, overrides, cells):
    """``overrides`` in which the land cover finds land at ``cells``."""
    islet = {name: np.array(value, copy=True)
             for name, value in overrides.items()}
    for j, i in cells:
        islet["LANDMASK"][j, i] = 1.0
        islet["LU_INDEX"][j, i] = 16.0  # barren
        islet["LANDUSEF"][:, j, i] = 0.0
        islet["LANDUSEF"][15, j, i] = 1.0
        for name, dom in (("SOILCTOP", "SCT_DOM"), ("SOILCBOT", "SCB_DOM")):
            islet[name][:, j, i] = 0.0
            islet[name][0, j, i] = 1.0  # sand
            islet[dom][j, i] = 1.0
        islet["HGT_M"][j, i] = 2.0
    return islet


@pytest.mark.parametrize("engine", ["default", "python"])
def test_a_grid_with_no_land_merges_without_a_donor_search(monkeypatch,
                                                            engine):
    """A fine grid over open sea has nothing to donate and needs no donor.

    The defect: every merge searched for the nearest baseline land cell,
    so a 333 m grid over the sea off Florida (all water at 30 arc seconds)
    stopped with "nearest-donor mask must be 2-D with a valid cell".
    """
    from woof.static import rust_bridge

    if engine == "python":
        monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
    baseline, overrides = _sea_inputs()
    merged, audit = merge_highres_overrides(baseline, overrides)
    np.testing.assert_array_equal(merged["LANDMASK"], 0.0)
    np.testing.assert_array_equal(merged["LU_INDEX"], 17.0)
    np.testing.assert_array_equal(merged["HGT_M"], -0.25)
    np.testing.assert_array_equal(merged["ALBEDO12M"], 8.0)
    np.testing.assert_array_equal(merged["SOILTEMP"], 0.0)
    np.testing.assert_array_equal(merged["TMN"], 0.0)
    assert audit["newly_land_nearest_climatology_fallback_cells"] == 0
    assert audit["newly_land_without_donor_kept_water_cells"] == 0
    assert audit["unchanged_land_water_cells"] == 36
    assert all(np.isfinite(value).all() for value in merged.values())


@pytest.mark.parametrize("engine", ["default", "python"])
def test_new_land_with_no_land_to_donate_keeps_the_baseline_water(
        monkeypatch, engine):
    """An islet the land cover finds on a grid whose baseline is all sea.

    No baseline land cell can give it vegetation, albedo or deep-soil
    temperature, so it would reach the model as land at 0 K.  It keeps
    the baseline's water in every overlay field but the terrain, and the
    receipt counts it.
    """
    from woof.static import rust_bridge

    if engine == "python":
        monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
    baseline, overrides = _sea_inputs()
    islet = _with_islet(baseline, overrides, [(2, 3), (3, 3)])
    merged, audit = merge_highres_overrides(baseline, islet)
    for name in ("LANDMASK", "LU_INDEX", "LANDUSEF", "SOILCTOP", "SCT_DOM",
                 "SOILCBOT", "SCB_DOM", "ALBEDO12M", "SOILTEMP", "TMN"):
        np.testing.assert_array_equal(merged[name], baseline[name],
                                      err_msg=name)
    assert merged["HGT_M"][2, 3] == 2.0
    assert audit["newly_land_without_donor_kept_water_cells"] == 2
    assert audit["newly_land_nearest_climatology_fallback_cells"] == 0
    # The caller's overrides are not changed under it.
    assert islet["LANDMASK"][2, 3] == 1.0


@pytest.mark.parametrize("engine", ["default", "python"])
def test_a_grid_with_no_water_merges_and_its_new_water_is_masked(
        monkeypatch, engine):
    """The mirror: an all-land grid (no water cell anywhere) merges, and a
    lake the land cover finds in it takes the water fills, with no donor
    search for water (none is ever made)."""
    from woof.static import rust_bridge

    if engine == "python":
        monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
    baseline, overrides, _ = _merge_inputs()
    baseline["LANDMASK"] = np.ones((4, 4))
    baseline["LU_INDEX"] = np.full((4, 4), 12.0)
    overrides = {name: np.array(value, copy=True)
                 for name, value in overrides.items()}
    overrides["LANDMASK"] = np.ones((4, 4))
    overrides["LU_INDEX"] = np.full((4, 4), 12.0)
    merged, audit = merge_highres_overrides(baseline, overrides)
    assert audit["newly_water_masked_cells"] == 0
    np.testing.assert_array_equal(merged["SOILTEMP"], 288.0)
    assert np.all(merged["TMN"] > 170.0)

    overrides["LANDMASK"][1, 1] = 0.0
    overrides["LU_INDEX"][1, 1] = 21.0
    merged, audit = merge_highres_overrides(baseline, overrides)
    assert audit["newly_water_masked_cells"] == 1
    assert merged["SOILTEMP"][1, 1] == 0.0
    assert merged["ALBEDO12M"][0, 1, 1] == 8.0


@pytest.mark.parametrize("engine", ["default", "python"])
def test_a_moving_grid_merges_on_every_placement_land_or_none(
        monkeypatch, engine):
    """A storm-following grid builds its statics at every placement, and
    its land mask changes as it moves: open sea, then a coast, then sea
    with an islet.  Every placement merges; each keeps its own answer."""
    from woof.static import rust_bridge

    if engine == "python":
        monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
    # One coastline the grid slides across: land east of column 14.
    world_sea, _ = _sea_inputs(shape=(6, 24))
    world = {name: np.array(value, copy=True)
             for name, value in world_sea.items()}
    world["LANDMASK"][:, 14:] = 1.0
    world["LU_INDEX"][:, 14:] = 12.0
    world["LANDUSEF"][:, :, 14:] = 0.0
    world["LANDUSEF"][11, :, 14:] = 1.0
    for name, dom in (("SOILCTOP", "SCT_DOM"), ("SOILCBOT", "SCB_DOM")):
        world[name][:, :, 14:] = 0.0
        world[name][5, :, 14:] = 1.0
        world[dom][:, 14:] = 6.0
    world["GREENFRAC"][:, :, 14:] = 0.6
    world["LAI12M"][:, :, 14:] = 2.5
    world["ALBEDO12M"][:, :, 14:] = 17.0
    world["SNOALB"][:, 14:] = 0.5
    world["SOILTEMP"][:, 14:] = 296.0
    world["TMN"][:, 14:] = 296.0

    def placement(i0):
        cut = {name: np.array(value[..., i0:i0 + 6], copy=True)
               for name, value in world.items()}
        overrides = {name: np.array(cut[name], copy=True) for name in
                     ("LANDUSEF", "LANDMASK", "LU_INDEX", "SOILCTOP",
                      "SCT_DOM", "SOILCBOT", "SCB_DOM")}
        overrides["HGT_M"] = np.where(cut["LANDMASK"] > 0.5, 3.0, 0.0)
        return cut, overrides

    # 1: open sea.  2: the coast, where the land cover finds the coast one
    # cell farther out to sea.  3: back over the sea with an islet.
    sea, sea_over = placement(0)
    coast, coast_over = placement(10)
    coast_over["LANDMASK"][:, 3] = 1.0
    coast_over["LU_INDEX"][:, 3] = 12.0
    islet_base, islet_over = placement(2)
    islet_over = _with_islet(islet_base, islet_over, [(4, 1)])

    merged, audit = merge_highres_overrides(sea, sea_over)
    np.testing.assert_array_equal(merged["LANDMASK"], 0.0)

    merged, audit = merge_highres_overrides(coast, coast_over)
    assert audit["newly_land_nearest_climatology_fallback_cells"] == 6
    # The new coast takes the nearest old land's climatology.
    np.testing.assert_array_equal(merged["SOILTEMP"][:, 3], 296.0)
    np.testing.assert_array_equal(merged["GREENFRAC"][:, :, 3], 0.6)
    np.testing.assert_array_equal(merged["SOILTEMP"][:, :3], 0.0)

    merged, audit = merge_highres_overrides(islet_base, islet_over)
    assert audit["newly_land_without_donor_kept_water_cells"] == 1
    np.testing.assert_array_equal(merged["LANDMASK"], 0.0)


def test_soil_over_open_sea_has_no_range_and_does_not_stop_the_grid(
        monkeypatch):
    """A footprint over open sea holds no SoilGrids pixel.  The library
    reports the range of none as null, and the step read it with float(),
    so a 333 m grid at 28.3 N 79.9 W stopped with "float() argument must
    be a string or a real number, not 'NoneType'" before it reached the
    merge.  No pixel is no range, recorded as such."""
    from woof.static import highres

    class SeaLibrary:
        @staticmethod
        def highres_resample(request):
            assert request["kind"] == "soil-categories"
            return ({"FRACTIONS": np.zeros((16, 8, 8))},
                    {"raw_component_total_percent_min": None,
                     "raw_component_total_percent_max": None,
                     "valid_source_pixels": 0})

    monkeypatch.setattr(highres, "_static_rust", lambda operation: SeaLibrary)
    fractions, audit = highres.soilgrids_category_fractions(
        {}, {"0-5cm": 1.0}, _grid())
    assert fractions.shape == (16, 8, 8)
    assert audit == {"raw_component_total_percent_min": None,
                     "raw_component_total_percent_max": None,
                     "valid_source_pixels": 0}


def test_merged_deep_soil_is_gated_on_land_and_masked_on_water(capsys):
    """A 0 K deep soil temperature on land refuses; on water it is a mask.

    The water fill used to be 0.0, a finite number that satisfied the
    finiteness gate and made a manufactured 0 K indistinguishable from a
    real one.  A whole-domain deep-soil decode failure -- SOILTEMP 0 K
    everywhere -- therefore produced TMN -0.65 K over land, an empty
    stderr, and an audit that counted three other things.  Downstream
    that becomes TMN = TSK for the entire domain, which removes the
    seasonal thermal reservoir under the soil column for the whole
    forecast.

    Both directions:  the healthy merge stays silent and clean with the
    water cells masked, and the whole-domain failure refuses by name with
    the land count.
    """
    baseline, overrides, new_land = _merge_inputs()
    merged, audit = merge_highres_overrides(baseline, overrides)
    # The legitimate water-cell path: silent, clean, and NAMED in the
    # receipt so a reader can tell the 0.0 from a temperature.
    assert capsys.readouterr().err == ""
    assert audit["deep_soil_water_masked_cells"] == 1
    water = new_land < 0.5
    np.testing.assert_array_equal(merged["SOILTEMP"][water], [0.0])
    np.testing.assert_array_equal(merged["TMN"][water], [0.0])
    land = ~water
    assert np.all(merged["TMN"][land] > 170.0)
    assert all(np.isfinite(value).all() for value in merged.values())

    # A whole-domain deep-soil decode failure.
    dead, overrides, _ = _merge_inputs(soiltemp=0.0)
    dead["SOILTEMP"] = np.zeros((4, 4))
    with pytest.raises(ValueError) as excinfo:
        merge_highres_overrides(dead, overrides)
    message = str(excinfo.value)
    assert "SOILTEMP is not a temperature on 15 land cell(s) of 15" in message
    assert "170..400" in message
    assert "geog soil_temperature" in message

    # One bad land cell is equally visible, and counted as one.
    holed, overrides, _ = _merge_inputs()
    holed["SOILTEMP"] = np.array(holed["SOILTEMP"], copy=True)
    holed["SOILTEMP"][2, 2] = 0.0
    with pytest.raises(ValueError) as excinfo:
        merge_highres_overrides(holed, overrides)
    assert "not a temperature on 1 land cell(s) of 15" in str(excinfo.value)
