"""Gates for the international (terrain-only) high-resolution lane.

Covers the per-source coverage model, the near-global tile enumerators,
the absent-tile contract, the terrain-only science and the config surface
-- all without touching the network or any real raster.
The end-to-end agreement against 3DEP on a real US footprint is a
separate, network-bound validation recorded in the evidence gallery.
"""
from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from woof.static.highres import (build_terrain_override,
                                  merge_terrain_override)
from woof.static.highres_fetch import (
    COPERNICUS_DEM_TILE_URL,
    CoverageError,
    FootprintBBox,
    SRTM_GL1_TILE_URL,
    SourceAbsent,
    TERRAIN_SOURCES,
    copernicus_dem_tile_ids,
    domain_footprint,
    fetch_copernicus_dem_tiles,
    fetch_srtm_gl1_tiles,
    one_degree_tile_bbox,
    srtm_tile_ids,
    terrain_source_coverage,
)
from woof.static.highres_production import (
    HighresRefusal,
    HighresStaticConfig,
    apply_highres_statics,
    parse_static_table,
)
from woof.static.lambert import LambertGrid

MODIS21_ATTRS = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
                 "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}


def _european_grid(dx: float = 3000.0, n: int = 41) -> LambertGrid:
    """A domain over central Europe -- outside every US collection."""
    return LambertGrid(
        ref_lat=48.2, ref_lon=16.4, truelat1=30.0, truelat2=60.0,
        stand_lon=16.4, dx=dx, dy=dx, e_we=n, e_sn=n)


def _us_grid(dx: float = 3000.0, n: int = 41) -> LambertGrid:
    return LambertGrid(
        ref_lat=38.68, ref_lon=-98.15, truelat1=30.0, truelat2=60.0,
        stand_lon=-98.15, dx=dx, dy=dx, e_we=n, e_sn=n)


def _baseline(ny: int, nx: int, *, land: bool = True) -> dict:
    return {
        "HGT_M": np.full((ny, nx), 500.0),
        "LU_INDEX": np.full((ny, nx), 10.0),
        "LANDMASK": np.full((ny, nx), 1.0 if land else 0.0),
        "SOILTEMP": np.full((ny, nx), 285.0),
    }


# ---------------------------------------------------------------------------
# Per-source coverage model
# ---------------------------------------------------------------------------

def test_every_terrain_source_declares_licence_and_attribution():
    for source_id, coverage in TERRAIN_SOURCES.items():
        assert coverage.source_id == source_id
        assert coverage.role == "terrain"
        assert coverage.license_id and coverage.license_url
        assert coverage.attribution, f"{source_id} carries no attribution"
        assert coverage.source_url.startswith("https://")


def test_coverage_check_names_source_footprint_and_overshoot():
    coverage = terrain_source_coverage("usgs-3dep-13as")
    bbox = FootprintBBox(lat_min=47.0, lat_max=52.0,
                         lon_min=5.0, lon_max=10.0)
    with pytest.raises(CoverageError) as failure:
        coverage.check(bbox)
    message = str(failure.value)
    assert "usgs-3dep-13as" in message
    assert "lat_max" in message           # the footprint is quoted
    assert "east_by_deg" in message       # the observed overshoot is quoted
    assert coverage.outside(bbox)["north_by_deg"] == pytest.approx(2.5)


def test_copernicus_covers_europe_but_srtm_stops_at_sixty_north():
    scandinavia = FootprintBBox(lat_min=61.0, lat_max=62.0,
                                lon_min=9.0, lon_max=10.0)
    terrain_source_coverage("copernicus-dem-glo30").check(scandinavia)
    with pytest.raises(CoverageError, match="srtm-gl1"):
        terrain_source_coverage("srtm-gl1").check(scandinavia)


def test_unknown_terrain_source_names_the_known_ones():
    with pytest.raises(CoverageError) as failure:
        terrain_source_coverage("aster-gdem")
    assert "copernicus-dem-glo30" in str(failure.value)


# ---------------------------------------------------------------------------
# Near-global tile enumeration
# ---------------------------------------------------------------------------

def test_copernicus_tiles_in_all_four_quadrants():
    assert copernicus_dem_tile_ids(FootprintBBox(
        39.2, 39.8, -104.8, -104.2)) == ("N39_00_W105_00",)
    assert copernicus_dem_tile_ids(FootprintBBox(
        -33.9, -33.2, 18.2, 18.8)) == ("S34_00_E018_00",)
    assert set(copernicus_dem_tile_ids(FootprintBBox(
        47.8, 48.3, 16.2, 16.7))) == {"N47_00_E016_00", "N48_00_E016_00"}


def test_copernicus_tile_bbox_round_trips_both_hemispheres():
    box = one_degree_tile_bbox("S34_00_E018_00")
    assert (box.lat_min, box.lat_max) == (-34.0, -33.0)
    assert (box.lon_min, box.lon_max) == (18.0, 19.0)
    box = one_degree_tile_bbox("N39W105")
    assert (box.lat_min, box.lat_max) == (39.0, 40.0)
    assert (box.lon_min, box.lon_max) == (-105.0, -104.0)


def test_srtm_tile_ids_use_the_compact_naming():
    assert srtm_tile_ids(FootprintBBox(39.2, 39.8, -104.8, -104.2)) \
        == ("N39W105",)


def test_a_continued_longitude_range_enumerates_across_the_line():
    """RETIRES test_antimeridian_span_refuses_at_enumeration.

    That gate refused any bbox wider than 180 degrees of longitude as "an
    antimeridian wrap, not a domain".  A dateline footprint no longer
    arrives wrapped: :func:`domain_footprint` reports it as a CONTINUED
    range, and the integer-degree loop names each tile through
    ``((lon_sw + 180) % 360) - 180``, so the two sides of the line are
    adjacent tiles and nothing special happens.
    """
    tiles = copernicus_dem_tile_ids(FootprintBBox(30.0, 31.0, 179.4, 180.6))
    assert tiles == ("N30_00_E179_00", "N30_00_W180_00")
    for tile in tiles:
        box = one_degree_tile_bbox(tile)
        assert box.lat_min == 30.0 and box.lat_max == 31.0


# ---------------------------------------------------------------------------
# Absent tiles: outside the source's coverage, handed back by id
# ---------------------------------------------------------------------------

class _FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_absent_copernicus_tile_is_returned_not_swallowed(tmp_path):
    def urlopen(url, offset):
        if "N48_00_E016_00" in url:
            raise SourceAbsent(f"{url} -> HTTP 404")
        return _FakeResponse(b"elevation")

    tiles, absent = fetch_copernicus_dem_tiles(
        FootprintBBox(47.8, 48.3, 16.2, 16.7), tmp_path, urlopen=urlopen)
    assert len(tiles) == 1
    assert absent == ("N48_00_E016_00",)


def test_all_tiles_absent_is_handed_back_not_refused(tmp_path):
    """RETIRES test_all_tiles_absent_refuses_as_open_water.

    A footprint no published tile reaches is outside the source's
    coverage everywhere; the production shell takes the baseline terrain
    on every cell (test_static_highres_coverage.py), so the fetch layer
    reports the absence instead of deciding it is fatal.
    """
    def urlopen(url, offset):
        raise SourceAbsent(f"{url} -> HTTP 404")

    tiles, absent = fetch_copernicus_dem_tiles(
        FootprintBBox(29.2, 29.8, -40.8, -40.2), tmp_path, urlopen=urlopen)
    assert tiles == ()
    assert absent == ("N29_00_W041_00",)


def test_srtm_fetch_uses_the_anonymous_mirror(tmp_path):
    seen = []

    def urlopen(url, offset):
        seen.append(url)
        return _FakeResponse(b"elevation")

    fetch_srtm_gl1_tiles(FootprintBBox(39.2, 39.8, -104.8, -104.2),
                         tmp_path, urlopen=urlopen)
    assert seen == [SRTM_GL1_TILE_URL.format(tile="N39W105")]
    assert "opentopography" in seen[0]
    # No credential, token or signature is ever appended.
    assert "?" not in seen[0] and "X-Amz" not in seen[0]


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------

def test_parse_accepts_terrain_source_and_fields(tmp_path):
    config = parse_static_table(
        {"highres": {"enabled": True, "cache_root": str(tmp_path),
                     "terrain_source": "copernicus-dem-glo30",
                     "fields": "terrain"}},
        source="case.toml", base_dir=tmp_path)
    assert config.terrain_source == "copernicus-dem-glo30"
    assert config.fields == "terrain"
    assert config.echo()["fields"] == "terrain"


def test_parse_defaults_both_new_keys_to_auto(tmp_path):
    config = parse_static_table(
        {"highres": {"enabled": True, "cache_root": str(tmp_path)}},
        source="case.toml", base_dir=tmp_path)
    assert (config.terrain_source, config.fields) == ("auto", "auto")


def test_parse_refuses_unknown_terrain_source_naming_choices(tmp_path):
    with pytest.raises(ValueError, match="copernicus-dem-glo30"):
        parse_static_table(
            {"highres": {"enabled": True, "cache_root": str(tmp_path),
                         "terrain_source": "aster"}},
            source="case.toml", base_dir=tmp_path)


def test_parse_refuses_unknown_fields_choice(tmp_path):
    with pytest.raises(ValueError, match="'terrain'"):
        parse_static_table(
            {"highres": {"enabled": True, "cache_root": str(tmp_path),
                         "fields": "landuse"}},
            source="case.toml", base_dir=tmp_path)


# ---------------------------------------------------------------------------
# Plan resolution and refusals
# ---------------------------------------------------------------------------

def _plan(config, grid):
    from woof.static.highres_fetch import domain_footprint
    from woof.static.build import HALO
    from woof.static.highres_production import _resolve_plan
    return _resolve_plan(config, domain_footprint(grid, HALO))


def test_auto_picks_us_stack_inside_the_us(tmp_path):
    mode, coverage = _plan(
        HighresStaticConfig(enabled=True, cache_root=tmp_path), _us_grid())
    assert (mode, coverage.source_id) == ("all", "usgs-3dep-13as")


def test_auto_runs_the_full_overlay_abroad_on_copernicus(tmp_path):
    """RETIRES test_auto_picks_terrain_only_copernicus_abroad: the default
    land cover is global, so abroad takes land use and soil too."""
    mode, coverage = _plan(
        HighresStaticConfig(enabled=True, cache_root=tmp_path),
        _european_grid())
    assert (mode, coverage.source_id) == ("all", "copernicus-dem-glo30")


def test_auto_with_nlcd_pinned_abroad_is_terrain_only(tmp_path):
    mode, coverage = _plan(
        HighresStaticConfig(enabled=True, cache_root=tmp_path,
                            landcover_source="annual-nlcd"),
        _european_grid())
    assert (mode, coverage.source_id) == ("terrain", "copernicus-dem-glo30")


def test_fields_all_abroad_with_nlcd_refuses_naming_the_source(tmp_path):
    with pytest.raises(HighresRefusal) as failure:
        _plan(HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                  fields="all",
                                  landcover_source="annual-nlcd"),
              _european_grid())
    assert failure.value.reason == "landcover-source-missing"
    assert "annual-nlcd" in failure.value.detail
    assert "fields = \"terrain\"" in failure.value.detail


def test_pinned_us_source_abroad_refuses_naming_the_source(tmp_path):
    with pytest.raises(HighresRefusal) as failure:
        _plan(HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                  fields="terrain",
                                  terrain_source="usgs-3dep-13as"),
              _european_grid())
    assert failure.value.reason == "outside-source-coverage"
    assert "usgs-3dep-13as" in failure.value.detail


def test_terrain_only_is_allowed_inside_the_us_for_cross_validation(tmp_path):
    mode, coverage = _plan(
        HighresStaticConfig(enabled=True, cache_root=tmp_path,
                            fields="terrain",
                            terrain_source="copernicus-dem-glo30"),
        _us_grid())
    assert (mode, coverage.source_id) == ("terrain", "copernicus-dem-glo30")


# ---------------------------------------------------------------------------
# Terrain-only science
# ---------------------------------------------------------------------------

def test_merge_terrain_override_recomputes_tmn_and_holds_the_mask():
    baseline = _baseline(6, 5)
    baseline["LANDMASK"][0, 0] = 0.0
    baseline["TMN"] = baseline["SOILTEMP"] - 0.0065 * baseline["HGT_M"]
    new_hgt = np.full((6, 5), 1500.0)
    merged, audit = merge_terrain_override(baseline, {"HGT_M": new_hgt})
    assert np.array_equal(merged["LANDMASK"], baseline["LANDMASK"])
    assert np.array_equal(merged["LU_INDEX"], baseline["LU_INDEX"])
    # Land cells lapse with the new height; the water cell keeps SOILTEMP.
    assert merged["TMN"][1, 1] == pytest.approx(285.0 - 0.0065 * 1500.0)
    assert merged["TMN"][0, 0] == pytest.approx(285.0)
    assert audit["terrain_cells_changed"] == 30
    assert audit["newly_land_nearest_climatology_fallback_cells"] == 0


def test_merge_terrain_override_refuses_non_terrain_overrides():
    baseline = _baseline(4, 4)
    with pytest.raises(KeyError, match="LANDMASK"):
        merge_terrain_override(
            baseline, {"HGT_M": np.zeros((4, 4)),
                       "LANDMASK": np.ones((4, 4))})


def test_merge_terrain_override_refuses_shape_mismatch():
    baseline = _baseline(4, 4)
    with pytest.raises(ValueError, match="shape"):
        merge_terrain_override(baseline, {"HGT_M": np.zeros((3, 3))})


def test_build_terrain_override_returns_terrain_alone(monkeypatch):
    grid = _us_grid(n=11)
    calls = {}

    def fake_resample(source, target_grid, *, method):
        calls["method"] = method
        ny, nx = target_grid.e_sn - 1, target_grid.e_we - 1
        return np.linspace(0.0, 1000.0, ny * nx).reshape(ny, nx)

    class _Terrain:
        def receipt(self):
            return {"source_id": "copernicus-dem-glo30"}

    monkeypatch.setattr("woof.static.highres.resample_continuous",
                        fake_resample)
    fields, audit = build_terrain_override(grid, terrain=_Terrain(), halo=3)
    assert set(fields) == {"HGT_M"}
    assert fields["HGT_M"].shape == (grid.e_sn - 1, grid.e_we - 1)
    assert calls["method"] == "average"
    assert "terrain only" in audit["method"]


# ---------------------------------------------------------------------------
# Antimeridian
# ---------------------------------------------------------------------------

def _dateline_grid(dx: float = 3000.0, n: int = 41) -> LambertGrid:
    """A small domain sitting on 180 degrees."""
    return LambertGrid(
        ref_lat=30.0, ref_lon=180.0, truelat1=30.0, truelat2=60.0,
        stand_lon=180.0, dx=dx, dy=dx, e_we=n, e_sn=n)


def test_a_dateline_domain_enumerates_its_own_tiles():
    """RETIRES test_antimeridian_domain_refuses_before_any_fetch.

    A domain on 180 degrees is a domain.  Its footprint must describe the
    degree and a half it occupies, not the whole planet, and the
    one-degree enumerator must return the handful of tiles either side of
    the line.
    """
    grid = _dateline_grid()
    bbox = domain_footprint(grid, 3)

    assert bbox.lat_max - bbox.lat_min == pytest.approx(1.30, abs=0.2)
    # The wrap is gone: a continued range, not a 360-degree bounding box.
    assert bbox.lon_max - bbox.lon_min < 2.0
    assert bbox.lon_max > 180.0 > bbox.lon_min

    tiles = copernicus_dem_tile_ids(bbox)
    assert len(tiles) < 20
    assert "N29_00_E179_00" in tiles
    assert "N29_00_W180_00" in tiles
    for tile in tiles:
        box = one_degree_tile_bbox(tile)
        assert box.lat_max - box.lat_min == 1.0
        assert box.lon_max - box.lon_min == 1.0


def test_copernicus_coverage_accepts_a_continued_longitude_range():
    """A source published for every longitude cannot be left east or west.

    Without the ``global_lon`` marker a continued footprint overshoots the
    -180..180 envelope by a fraction of a degree and the coverage gate
    reports an east overshoot, which is an artefact of the frame rather
    than a fact about the product.
    """
    coverage = terrain_source_coverage("copernicus-dem-glo30")
    assert coverage.global_lon is True
    continued = FootprintBBox(lat_min=29.3, lat_max=30.7,
                              lon_min=179.2, lon_max=180.8)
    assert coverage.outside(continued) == {}
    coverage.check(continued)
    # The latitude legs are untouched: GLO-30 still stops at 84 N.
    assert "north_by_deg" in coverage.outside(
        FootprintBBox(83.0, 86.0, 179.2, 180.8))


def test_no_shape_gate_refuses_a_dateline_domain(tmp_path):
    """RETIRES test_no_gate_refuses_a_dateline_domain_before_coverage.

    Being on 180 degrees is not itself an objection any more: there is no
    surviving gate that judges the SHAPE of a footprint or the projection
    it came from.  What can still stop a dateline domain is one named
    capability gap -- the mosaic window is written in the cut -180..180
    frame -- or a per-source coverage fact.  Both name a way out.  The
    earlier name promised the stop came after coverage, which is no longer
    true and was never the point.
    """
    config = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                 fields="all")
    with pytest.raises(HighresRefusal) as failure:
        apply_highres_statics(
            _baseline(40, 40), _dateline_grid(), config=config, domain_id=1,
            case_date=date(2021, 5, 15), landuse_attrs=MODIS21_ATTRS)
    assert failure.value.reason not in {"antimeridian-footprint",
                                        "unsupported-projection",
                                        "coastal-footprint"}
    assert failure.value.reason in {"outside-source-coverage",
                                    "landcover-source-missing",
                                    "dateline-window-unbuilt"}


# ---------------------------------------------------------------------------
# Every projection this tree builds reaches the overlay
# ---------------------------------------------------------------------------

def test_mercator_and_polar_domains_reach_the_overlay(tmp_path):
    """The overlay's gate is a grid TYPE, not a projection allow-list.

    ``projection_class`` already refuses every map_proj this tree cannot
    build, at grid construction, i.e. at configuration load.  What reaches
    static production is therefore always lambert, mercator or polar, and
    the overlay resamples all three through the grid's own PROJ CRS.  So
    whatever comes back here must be a coverage fact about the source that
    was selected, never "unsupported-projection".
    """
    from woof.static.projection import MercatorGrid, PolarStereoGrid

    mercator = MercatorGrid(
        ref_lat=40.0, ref_lon=-100.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-100.0, dx=3000.0, dy=3000.0, e_we=41, e_sn=41)
    polar = PolarStereoGrid(
        ref_lat=70.0, ref_lon=-40.0, truelat1=60.0, truelat2=60.0,
        stand_lon=-40.0, dx=3000.0, dy=3000.0, e_we=41, e_sn=41)

    # 3DEP over the Mercator domain and Copernicus over the polar one are
    # both reported absent, so each run refuses on a COVERAGE fact about
    # the source it chose and no byte is requested from the network.
    def absent(request, timeout=None):
        raise SourceAbsent(getattr(request, "full_url", str(request)))

    for grid in (mercator, polar):
        config = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                     fields="all")
        with pytest.raises(HighresRefusal) as failure:
            apply_highres_statics(
                _baseline(40, 40), grid, config=config, domain_id=1,
                case_date=date(2021, 5, 15), landuse_attrs=MODIS21_ATTRS,
                urlopen=absent)
        assert failure.value.reason != "unsupported-projection"
        assert failure.value.reason in {
            "outside-source-coverage", "landcover-source-missing",
            "missing-source-coverage"}


def test_require_projected_grid_accepts_every_built_projection():
    """The surviving refusal is a type fact and nothing more."""
    from woof.static.highres_production import (HighresRefusal as _Refusal,
                                                 _require_projected_grid)
    from woof.static.projection import MercatorGrid, PolarStereoGrid

    _require_projected_grid(_us_grid())
    _require_projected_grid(MercatorGrid(
        ref_lat=40.0, ref_lon=-100.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-100.0, dx=3000.0, dy=3000.0, e_we=41, e_sn=41))
    _require_projected_grid(PolarStereoGrid(
        ref_lat=70.0, ref_lon=-40.0, truelat1=60.0, truelat2=60.0,
        stand_lon=-40.0, dx=3000.0, dy=3000.0, e_we=41, e_sn=41))
    with pytest.raises(_Refusal) as failure:
        _require_projected_grid(object())
    assert failure.value.reason == "unsupported-projection"


def test_grid_crs_is_general_across_projections():
    """The layer under the gate really is parameterised by map_proj."""
    pytest.importorskip("pyproj")
    from woof.static.highres import _grid_crs
    from woof.static.projection import MercatorGrid, PolarStereoGrid

    mercator = MercatorGrid(
        ref_lat=40.0, ref_lon=-100.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-100.0, dx=3000.0, dy=3000.0, e_we=41, e_sn=41)
    polar = PolarStereoGrid(
        ref_lat=70.0, ref_lon=-40.0, truelat1=60.0, truelat2=60.0,
        stand_lon=-40.0, dx=3000.0, dy=3000.0, e_we=41, e_sn=41)
    assert "merc" in _grid_crs(mercator).to_proj4()
    assert "stere" in _grid_crs(polar).to_proj4()


# ---------------------------------------------------------------------------
# One domain, two sources: the receipts must not overwrite each other
# ---------------------------------------------------------------------------

def test_two_terrain_sources_on_one_domain_write_two_receipts(tmp_path):
    """Cross-validating a domain through two DEMs must keep both receipts.

    The documented way to find out what changing ``terrain_source`` does to
    your terrain is to build the same domain twice and compare.  The
    receipt is the artifact that records which source ran, so if both runs
    write to the same path the first run's provenance is destroyed by the
    second -- and the comparison it exists to support becomes unauditable.
    Two runs that differ only in the source they asked for are two runs.
    """
    import json

    # 70 N: outside 3DEP (a US collection) and outside SRTM's 60 N ceiling,
    # so both refuse on coverage without touching the network.
    grid = LambertGrid(ref_lat=70.0, ref_lon=25.0, truelat1=30.0,
                       truelat2=60.0, stand_lon=25.0, dx=3000.0, dy=3000.0,
                       e_we=41, e_sn=41)
    baseline = _baseline(40, 40)

    written = {}
    for source in ("usgs-3dep-13as", "srtm-gl1"):
        config = HighresStaticConfig(
            enabled=True, cache_root=tmp_path, on_refuse="error",
            terrain_source=source, fields="terrain")
        with pytest.raises(HighresRefusal):
            apply_highres_statics(
                baseline, grid, config=config, domain_id=1,
                case_date=date(2024, 6, 1), landuse_attrs=MODIS21_ATTRS)
        receipt_path = next((tmp_path/"receipts").glob(f"*{source}*.json"))
        receipt = json.loads(receipt_path.read_text())
        receipt['receipt_path'] = str(receipt_path)
        assert receipt["status"] == "REFUSED"
        written[source] = Path(receipt["receipt_path"])

    assert written["usgs-3dep-13as"] != written["srtm-gl1"], (
        "both terrain sources wrote the same receipt file "
        f"{written['usgs-3dep-13as']}; the first run's provenance was "
        "overwritten by the second")
    for source, path in written.items():
        assert path.is_file(), path
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["config"]["terrain_source"] == source


# ---------------------------------------------------------------------------
# The continued-frame refusal fires at plan review, before any fetch
# ---------------------------------------------------------------------------

def _count_requests():
    """A urlopen that records every request and never returns a byte."""
    seen = []

    def urlopen(url, offset):
        seen.append(url)
        raise SourceAbsent(f"{url} -> HTTP 404")

    return seen, urlopen


def test_a_dateline_terrain_domain_refuses_before_any_network_request(
        tmp_path):
    """The frame refusal is plan review, not a late discovery.

    ``domain_footprint`` continues a dateline domain past 180 degrees and
    the tile enumerators read it in that frame, but the mosaic window is
    still written in the cut -180..180 frame.  The run must therefore stop
    on the footprint it just computed, with the network untouched: a
    silently enormous fetch that ends in "no" is exactly the failure this
    program refuses.
    """
    seen, urlopen = _count_requests()
    config = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                 fields="terrain",
                                 terrain_source="copernicus-dem-glo30")
    grid = _dateline_grid()
    # The footprint really is continued, and really does enumerate tiles:
    # without the refusal this run has somewhere to go.
    bbox = domain_footprint(grid, 3)
    assert bbox.lon_max > 180.0
    assert len(copernicus_dem_tile_ids(bbox)) > 0

    with pytest.raises(HighresRefusal) as failure:
        apply_highres_statics(
            _baseline(40, 40), grid, config=config, domain_id=1,
            case_date=date(2021, 5, 15), landuse_attrs=MODIS21_ATTRS,
            urlopen=urlopen)

    assert failure.value.reason == "dateline-window-unbuilt"
    assert seen == [], (
        f"{len(seen)} network request(s) were made before the refusal; "
        "the frame check must run on the footprint, not after the fetch")
    # Names the breakage and both ways out.
    detail = failure.value.detail
    assert "shifted" in detail
    assert "Move the domain off 180 degrees" in detail
    assert "30-arc-second baseline" in detail


def test_a_pole_enclosing_domain_refuses_before_enumerating_its_tiles(
        tmp_path):
    """The worst case is the one that must never reach the fetch loop.

    A polar-stereographic domain that encloses the south pole occupies
    every longitude, so its footprint is the full band and its one-degree
    enumeration is thousands of tiles -- all of them inside Copernicus
    GLO-30's published envelope, so coverage says yes.  The only thing
    between that configuration and a multi-thousand-tile download is this
    refusal, and it has to fire first.
    """
    from woof.static.projection import PolarStereoGrid

    grid = PolarStereoGrid(
        ref_lat=-89.5, ref_lon=0.0, truelat1=-60.0, truelat2=-60.0,
        stand_lon=0.0, dx=40000.0, dy=40000.0, e_we=41, e_sn=41)
    bbox = domain_footprint(grid, 3)
    assert len(copernicus_dem_tile_ids(bbox)) > 1000
    # This footprint really is the whole band, not a crossing of a line.
    assert bbox.lon_max - bbox.lon_min >= 360.0

    seen, urlopen = _count_requests()
    config = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                 fields="terrain",
                                 terrain_source="copernicus-dem-glo30")
    with pytest.raises(HighresRefusal) as failure:
        apply_highres_statics(
            _baseline(40, 40), grid, config=config, domain_id=1,
            case_date=date(2021, 5, 15), landuse_attrs=MODIS21_ATTRS,
            urlopen=urlopen)

    assert failure.value.reason == "dateline-window-unbuilt"
    assert seen == [], (
        f"{len(seen)} network request(s) were made before the refusal")
    # And coverage really would have said yes: GLO-30 reaches 90 S and is
    # published for every longitude, so nothing downstream objects to this
    # footprint.  The refusal above is the only thing between it and a
    # multi-thousand-tile download.
    terrain_source_coverage("copernicus-dem-glo30").check(bbox)


def test_the_pole_enclosing_refusal_names_the_pole_not_the_dateline(
        tmp_path):
    """A refusal that misnames the fact sends the reader nowhere.

    A domain wrapped around a projection pole is not sitting on 180
    degrees: its footprint occupies every longitude, there is no line
    with tiles either side of it, and "move the domain off 180 degrees"
    is a remedy that cannot be taken -- move it where you like and the
    footprint is still the whole band.  The refusal must state what is
    true of THIS footprint and offer a way out that builds, which is a
    smaller domain or one further from the projection pole.  The second
    way out, the 30-arc-second baseline, is valid for both footprints
    and stays.
    """
    from woof.static.projection import PolarStereoGrid

    polar = PolarStereoGrid(
        ref_lat=-89.5, ref_lon=0.0, truelat1=-60.0, truelat2=-60.0,
        stand_lon=0.0, dx=40000.0, dy=40000.0, e_we=41, e_sn=41)
    # An ordinary large Lambert whose lat/lon envelope wraps the north
    # pole reaches the same branch, with stand_lon nowhere near 180.
    lambert = LambertGrid(
        ref_lat=45.0, ref_lon=-100.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-100.0, dx=90000.0, dy=90000.0, e_we=121, e_sn=101)

    for grid in (polar, lambert):
        bbox = domain_footprint(grid, 3)
        assert bbox.lon_max - bbox.lon_min >= 360.0
        seen, urlopen = _count_requests()
        config = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                     fields="terrain",
                                     terrain_source="copernicus-dem-glo30")
        with pytest.raises(HighresRefusal) as failure:
            apply_highres_statics(
                _baseline(40, 40), grid, config=config, domain_id=1,
                case_date=date(2021, 5, 15), landuse_attrs=MODIS21_ATTRS,
                urlopen=urlopen)
        detail = failure.value.detail
        assert failure.value.reason == "dateline-window-unbuilt"
        assert seen == []
        # The fact this footprint actually has.
        assert "occupies every longitude" in detail
        assert "corners span 180 degrees or more" in detail
        assert "projection pole" in detail
        # Not the dateline story, and not the remedy that cannot be taken.
        assert "Move the domain off 180 degrees" not in detail
        assert "either side of the line" not in detail
        # The way out that builds, and the one that is valid for both.
        assert "Shrink the domain" in detail
        assert "span less than 180 degrees of longitude" in detail
        assert "30-arc-second baseline" in detail


def test_a_domain_on_the_line_still_gets_the_dateline_wording(tmp_path):
    """The branch did not cost the on-the-line domain its own statement.

    A domain straddling 180 degrees has a narrow continued range, really
    does have tiles either side of a line, and really can be moved off
    it.  That footprint must keep the dateline wording and must not be
    told to shrink away from a pole it is nowhere near.
    """
    seen, urlopen = _count_requests()
    config = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                 fields="terrain",
                                 terrain_source="copernicus-dem-glo30")
    grid = _dateline_grid()
    bbox = domain_footprint(grid, 3)
    assert bbox.lon_max - bbox.lon_min < 180.0

    with pytest.raises(HighresRefusal) as failure:
        apply_highres_statics(
            _baseline(40, 40), grid, config=config, domain_id=1,
            case_date=date(2021, 5, 15), landuse_attrs=MODIS21_ATTRS,
            urlopen=urlopen)
    detail = failure.value.detail
    assert failure.value.reason == "dateline-window-unbuilt"
    assert seen == []
    assert "either side of the line" in detail
    assert "Move the domain off 180 degrees" in detail
    assert "occupies every longitude" not in detail
    assert "Shrink the domain" not in detail
    assert "30-arc-second baseline" in detail


def test_the_full_mode_takes_the_same_frame_refusal(tmp_path):
    """One function, both modes.

    ``fields = "all"`` computes the same footprint and must stop on the
    same fact, ahead of the land-cover coverage refusal that would
    otherwise be reported first.
    """
    seen, urlopen = _count_requests()
    config = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                 fields="all")
    with pytest.raises(HighresRefusal) as failure:
        apply_highres_statics(
            _baseline(40, 40), _dateline_grid(), config=config, domain_id=1,
            case_date=date(2021, 5, 15), landuse_attrs=MODIS21_ATTRS,
            urlopen=urlopen)
    assert failure.value.reason == "dateline-window-unbuilt"
    assert seen == []


def test_both_window_writers_keep_the_frame_refusal_as_a_backstop(tmp_path):
    """Plan review is the door; the two writers are still the backstop.

    A caller that reaches a window writer directly, without going through
    production's plan review, must get the same named refusal from the
    same function rather than a silently shifted mosaic.
    """
    from woof.static.highres_fetch import (FetchedFile,
                                            derive_global_terrain_window,
                                            derive_terrain_window)

    continued = FootprintBBox(lat_min=29.3, lat_max=30.7,
                              lon_min=179.2, lon_max=180.8)
    tile = FetchedFile(path=tmp_path / "absent.tif", url="derived:test",
                       sha256="0" * 64, bytes=1, fetched_utc="",
                       cache_hit=False)

    for writer in (derive_terrain_window, derive_global_terrain_window):
        with pytest.raises(HighresRefusal) as failure:
            writer([tile], continued, tmp_path)
        assert failure.value.reason == "dateline-window-unbuilt"
        assert "cut -180..180 frame" in failure.value.detail
