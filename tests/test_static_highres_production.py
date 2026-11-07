"""Focused gates for the production high-resolution static geography lane.

Covers the config surface (unknown keys refuse; absence is the identity),
the footprint-parametric tile enumeration, the cache-hit path, and the
US-interior/coast gates -- all without touching the network or any real
raster.  Coverage gaps (cells a source does not reach) take the baseline
and are covered in ``test_static_highres_coverage.py``.
"""
from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from woof.static.highres_fetch import (
    CoverageError,
    FootprintBBox,
    domain_footprint,
    fetch_file,
    nlcd_year_for,
    three_dep_tile_ids,
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


def _us_interior_grid(dx: float = 3000.0, n: int = 41) -> LambertGrid:
    return LambertGrid(
        ref_lat=38.68, ref_lon=-98.15, truelat1=30.0, truelat2=60.0,
        stand_lon=-98.15, dx=dx, dy=dx, e_we=n, e_sn=n)


def _baseline(ny: int, nx: int) -> dict[str, np.ndarray]:
    return {
        "HGT_M": np.full((ny, nx), 500.0),
        "LU_INDEX": np.full((ny, nx), 10.0),
        "LANDMASK": np.ones((ny, nx)),
    }


# ---------------------------------------------------------------------------
# Footprint and tile enumeration
# ---------------------------------------------------------------------------

def test_domain_footprint_covers_halo_extended_corners():
    grid = _us_interior_grid()
    inner = domain_footprint(grid, halo=0, margin_deg=0.0)
    outer = domain_footprint(grid, halo=3)
    assert outer.lat_min < inner.lat_min
    assert outer.lat_max > inner.lat_max
    assert outer.lon_min < inner.lon_min
    assert outer.lon_max > inner.lon_max
    lat, lon = grid.ij_to_latlon(
        np.array([0.5, grid.e_we - 0.5]), np.array([0.5, grid.e_sn - 0.5]))
    assert outer.lat_min < float(np.min(lat)) <= float(np.max(lat)) \
        < outer.lat_max
    assert outer.lon_min < float(np.min(lon)) <= float(np.max(lon)) \
        < outer.lon_max


def test_three_dep_tile_enumeration_from_bbox():
    bbox = FootprintBBox(lat_min=38.05, lat_max=39.31,
                         lon_min=-98.975, lon_max=-97.325)
    assert set(three_dep_tile_ids(bbox)) == {
        "n39w099", "n39w098", "n40w099", "n40w098"}


def test_three_dep_tile_enumeration_single_tile_interior():
    bbox = FootprintBBox(lat_min=37.2, lat_max=37.8,
                         lon_min=-98.9, lon_max=-98.2)
    assert three_dep_tile_ids(bbox) == ("n38w099",)


def test_three_dep_tile_enumeration_refuses_other_quadrants():
    with pytest.raises(CoverageError, match="quadrant"):
        three_dep_tile_ids(FootprintBBox(lat_min=-2.0, lat_max=-1.0,
                                         lon_min=-70.0, lon_max=-69.0))


def test_nlcd_year_nearest_and_anachronism():
    assert nlcd_year_for(date(1974, 4, 3)) == (1985, 11)
    assert nlcd_year_for(date(2021, 5, 15)) == (2021, 0)
    assert nlcd_year_for(date(2030, 1, 1)) == (2024, 6)


# ---------------------------------------------------------------------------
# Fetch: cache hit and synthetic coverage gap
# ---------------------------------------------------------------------------

class _FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_fetch_file_records_sha_and_hits_cache(tmp_path):
    calls = []

    def urlopen(url, offset):
        calls.append(url)
        return _FakeResponse(b"tile-payload")

    target = tmp_path / "cache" / "artifact.bin"
    first = fetch_file("https://example.invalid/a", target, urlopen=urlopen)
    assert first.cache_hit is False
    assert first.bytes == len(b"tile-payload")
    sidecar = json.loads(
        (target.parent / (target.name + ".sha256.json")).read_text())
    assert sidecar["sha256"] == first.sha256
    assert calls == ["https://example.invalid/a"]

    second = fetch_file("https://example.invalid/a", target, urlopen=urlopen)
    assert second.cache_hit is True
    assert second.sha256 == first.sha256
    assert calls == ["https://example.invalid/a"]  # no second network touch


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------

def test_parse_static_table_absent_is_none(tmp_path):
    assert parse_static_table(None, source="case.toml",
                              base_dir=tmp_path) is None


def test_parse_static_table_accepts_and_resolves(tmp_path):
    config = parse_static_table(
        {"highres": {"enabled": True, "cache_root": "hr-cache",
                     "on_refuse": "error"}},
        source="case.toml", base_dir=tmp_path)
    assert config == HighresStaticConfig(
        enabled=True, cache_root=tmp_path / "hr-cache",
        on_refuse="error")
    assert config.echo() == {
        "enabled": "true", "cache_root": str(tmp_path / "hr-cache"),
        "on_refuse": "error", "terrain_source": "auto",
        "fields": "auto", "landcover_source": "auto"}


@pytest.mark.parametrize("table,match", [
    ({"highres": {"enabled": True, "cache_roots": "x"}}, "cache_roots"),
    ({"hires": {}}, "hires"),
    ({}, "declares nothing"),
    ({"highres": {"enabled": True, "cache_root": "x", "on_refuse": "warn"}}, "on_refuse"),
    ({"highres": {"enabled": True}}, "cache_root"),
])
def test_parse_static_table_refuses_invalid_config(tmp_path, table, match):
    with pytest.raises(ValueError, match=match):
        parse_static_table(table, source="case.toml", base_dir=tmp_path)


def test_case_data_loader_carries_static_block(tmp_path):
    from woof.case_data import build_case_data

    for name in ("forcing.grib", "Vtable", "namelist.wps"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "WPS_GEOG").mkdir()
    data = build_case_data(
        {"forcing": "forcing.grib", "vtable": "Vtable",
         "wps_namelist": "namelist.wps", "geog_root": "WPS_GEOG",
         "sfcp_to_sfcp": True, "output_title": "t"},
        source="case.toml", base_dir=tmp_path)
    # Absence of the block is the identity: the field defaults to None and
    # every seam guards on it, so current behavior is bit-identical.
    assert data.static_highres is None


# ---------------------------------------------------------------------------
# Application identity and refusal gates (no network, no rasters)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("disabled", [False, True], ids=["absent", "disabled"])
def test_apply_inactive_config_is_identity_object(tmp_path, disabled):
    baseline = _baseline(4, 4)
    config = HighresStaticConfig(enabled=False, cache_root=tmp_path) if disabled else None
    fields, receipt = apply_highres_statics(
        baseline, _us_interior_grid(), config=config, domain_id=1,
        case_date=date(2021, 5, 15), landuse_attrs=MODIS21_ATTRS)
    assert fields is baseline
    assert receipt is None


def test_apply_refuses_outside_us_coverage(tmp_path):
    grid = LambertGrid(
        ref_lat=52.0, ref_lon=-98.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-98.0, dx=3000.0, dy=3000.0, e_we=41, e_sn=41)
    config = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                 terrain_source="usgs-3dep-13as")
    with pytest.raises(HighresRefusal) as failure:
        apply_highres_statics(
            _baseline(40, 40), grid, config=config, domain_id=1,
            case_date=date(2021, 5, 15), landuse_attrs=MODIS21_ATTRS)
    assert failure.value.reason == "outside-source-coverage"
    assert "usgs-3dep-13as" in failure.value.detail


# ---------------------------------------------------------------------------
# The coast-safe water rule (RETIRES test_apply_refuses_coastal_footprint_
# naming_method, which asserted reason == "coastal-footprint")
# ---------------------------------------------------------------------------

#: Model-grid cell the baseline calls WRF ocean, and one it calls land.
_OCEAN_CELL = (3, 7)
_INLAND_WATER_CELL = (20, 25)


def _full_baseline(ny: int, nx: int) -> dict[str, np.ndarray]:
    """A complete Noah static state, the shape merge_highres_overrides wants."""
    luf = np.zeros((21, ny, nx))
    luf[9] = 1.0                                    # MODIS category 10
    soil = np.zeros((16, ny, nx))
    soil[2] = 1.0                                   # WRF soil category 3
    base = _baseline(ny, nx)
    base.update({
        "LANDUSEF": luf,
        "SOILCTOP": soil.copy(),
        "SOILCBOT": soil.copy(),
        "SCT_DOM": np.full((ny, nx), 3.0),
        "SCB_DOM": np.full((ny, nx), 3.0),
        "GREENFRAC": np.full((12, ny, nx), 0.5),
        "LAI12M": np.full((12, ny, nx), 1.0),
        "ALBEDO12M": np.full((12, ny, nx), 20.0),
        "SNOALB": np.full((ny, nx), 60.0),
        "SOILTEMP": np.full((ny, nx), 285.0),
    })
    return base


class _StubRaster:
    def __init__(self, source_id: str, role: str):
        self._receipt = {"source_id": source_id, "role": role}

    def receipt(self):
        return dict(self._receipt)


def _stub_the_overlay(monkeypatch, *, water_cells, halo):
    """Stub the fetch and the three resamples; keep the science real.

    Everything the network and the raster decoders would produce is
    supplied here, so what actually runs is the part under test: the
    crosswalked open-water fraction, the ocean/lake split against the
    baseline water field, and the landmask/LU_INDEX rules built on top.
    """
    from woof.static import highres as highres_module

    def fake_fetch_and_bind(bbox, cache_root, case_date, *, coverage, grid,
                            baseline, urlopen=None, landcover_source=None):
        assert landcover_source.source_id == "annual-nlcd"
        return (_StubRaster("usgs-3dep-13as", "terrain"),
                _StubRaster("annual-nlcd", "landcover"),
                {("sand", "0-5cm"): _StubRaster("soilgrids-v2", "soil")},
                {"landcover_year": 2021, "landcover_anachronism_years": 0,
                 "bytes_fetched": 0})

    def fake_resample_continuous(source, grid, *, method):
        ny, nx = grid.e_sn - 1, grid.e_we - 1
        return np.linspace(300.0, 900.0, ny * nx).reshape(ny, nx)

    def fake_resample_mapped_categories(source, grid, mapping, *,
                                        category_count):
        # The crosswalk sends NLCD 11 (open water) to MODIS 21; this stub
        # stands in for the warp that delivers that fraction, and puts it
        # on the two named cells only.
        assert mapping[11] == 21
        ny, nx = grid.e_sn - 1, grid.e_we - 1
        fractions = np.zeros((category_count, ny, nx))
        fractions[9] = 1.0                          # land everywhere else
        for (row, col) in water_cells:
            fractions[:, row + halo, col + halo] = 0.0
            fractions[20, row + halo, col + halo] = 1.0
        return fractions

    def fake_soilgrids_category_fractions(sources, weights, grid, *,
                                          category_count):
        ny, nx = grid.e_sn - 1, grid.e_we - 1
        fractions = np.zeros((category_count, ny, nx))
        fractions[5] = 1.0
        return fractions, {"raw_component_total_percent_min": 100.0,
                           "raw_component_total_percent_max": 100.0,
                           "valid_source_pixels": ny * nx}

    monkeypatch.setattr(
        "woof.static.highres_production._fetch_and_bind", fake_fetch_and_bind)
    monkeypatch.setattr(highres_module, "resample_continuous",
                        fake_resample_continuous)
    monkeypatch.setattr(highres_module, "resample_mapped_categories",
                        fake_resample_mapped_categories)
    monkeypatch.setattr(highres_module, "soilgrids_category_fractions",
                        fake_soilgrids_category_fractions)


def test_a_coastal_domain_keeps_ocean_and_takes_high_resolution_land_use(
        tmp_path, monkeypatch):
    """A coast is not a refusal: the sea stays ocean and the lake stays lake.

    The NLCD crosswalk has one open water class, so it cannot tell a
    lake from the sea.  The discriminator is the domain's own 30-arc-second
    baseline water field, which is already on the model grid.  Open water on
    a cell the baseline calls WRF ocean category 17 becomes ocean; anywhere
    else it stays lake category 21.
    """
    from woof.static.highres import HALO

    grid = _us_interior_grid()
    baseline = _full_baseline(40, 40)
    baseline["LU_INDEX"][_OCEAN_CELL] = 17.0   # WRF ocean in the baseline
    _stub_the_overlay(monkeypatch,
                      water_cells=(_OCEAN_CELL, _INLAND_WATER_CELL),
                      halo=HALO)

    config = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                 landcover_source="annual-nlcd")
    fields, receipt = apply_highres_statics(
        baseline, grid, config=config, domain_id=1,
        case_date=date(2021, 5, 15), landuse_attrs=MODIS21_ATTRS)

    assert receipt["status"] == "APPLIED"
    assert fields["LU_INDEX"][_OCEAN_CELL] == 17.0
    assert fields["LU_INDEX"][_INLAND_WATER_CELL] == 21.0
    assert np.allclose(fields["LANDUSEF"].sum(axis=0), 1.0)
    assert fields["LANDMASK"][_OCEAN_CELL] == 0.0
    assert fields["LANDMASK"][_INLAND_WATER_CELL] == 0.0

    split = receipt["override_audit"]["water_split"]
    assert split["ocean_cells_from_baseline_water"] == 1
    assert split["lake_cells"] == 1
    assert "baseline LU_INDEX water field" in split["method"]
    assert "17" in split["method"] and "21" in split["method"]


def test_build_highres_overrides_splits_ocean_from_lake_on_a_hand_mask():
    """The split alone, with no fetch path anywhere near it."""
    from woof.static.highres import _split_ocean_from_lake

    luf = np.zeros((21, 4, 4))
    luf[9] = 1.0
    luf[:, 1, 1] = 0.0
    luf[20, 1, 1] = 1.0                     # crosswalked open water: the sea
    luf[:, 2, 3] = 0.0
    luf[20, 2, 3] = 1.0                     # crosswalked open water: a lake
    ocean = np.zeros((4, 4), dtype=bool)
    ocean[1, 1] = True

    audit = _split_ocean_from_lake(luf, ocean, iswater=17, islake=21)

    assert luf[16, 1, 1] == 1.0 and luf[20, 1, 1] == 0.0
    assert luf[20, 2, 3] == 1.0 and luf[16, 2, 3] == 0.0
    assert np.allclose(luf.sum(axis=0), 1.0)
    assert audit["ocean_cells_from_baseline_water"] == 1
    assert audit["lake_cells"] == 1
    assert "baseline LU_INDEX water field" in audit["method"]


def test_apply_refuses_non_modis21_landuse(tmp_path):
    config = HighresStaticConfig(enabled=True, cache_root=tmp_path)
    attrs = dict(MODIS21_ATTRS, ISWATER=16, ISLAKE="")
    with pytest.raises(HighresRefusal) as failure:
        apply_highres_statics(
            _baseline(40, 40), _us_interior_grid(), config=config,
            domain_id=1, case_date=date(2021, 5, 15), landuse_attrs=attrs)
    assert failure.value.reason == "landuse-inventory-mismatch"


def test_retired_fallback_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="retired"):
        parse_static_table({"highres": {"enabled": True, "cache_root": str(tmp_path),
                           "on_refuse": "fallback-30s"}}, source="case.toml", base_dir=tmp_path)


# ---------------------------------------------------------------------------
# An enabled block on a lane that cannot honor it refuses, naming the lane
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The coast-safe split is default-on, and both doors take it
# ---------------------------------------------------------------------------

def test_build_highres_overrides_requires_the_baseline_ocean_mask():
    """Fixed means default-on: the discriminator has no default.

    An optional ``baseline_ocean`` would make the coast-safe rule opt-in,
    and a caller that omitted it would silently take the crosswalk's
    inland reading and put the sea in WRF lake 21.  The parameter is
    required, so that branch cannot be reached by forgetting it.
    """
    import inspect

    from woof.static.highres import build_highres_overrides

    parameters = inspect.signature(build_highres_overrides).parameters
    assert "baseline_ocean" in parameters, (
        "build_highres_overrides takes no baseline_ocean at all, so the "
        "crosswalk cannot tell the sea from a lake")
    assert parameters["baseline_ocean"].default is inspect.Parameter.empty, (
        "baseline_ocean carries a default, which makes the coast-safe "
        "split opt-in")


def test_the_split_refuses_a_missing_mask_instead_of_skipping_it():
    """Explicit ``None`` is refused by name, not quietly ignored."""
    from woof.static.highres import _split_ocean_from_lake

    luf = np.zeros((21, 2, 2))
    luf[20] = 1.0
    with pytest.raises(ValueError) as failure:
        _split_ocean_from_lake(luf, None, iswater=17, islake=21)
    message = str(failure.value)
    assert "30-arc-second" in message
    assert "baseline_ocean_mask" in message      # the way out
    assert "all-False" in message                # and the other way out


def test_baseline_ocean_mask_reads_the_baseline_water_field():
    """The one discriminator, derived once, from the domain's own field."""
    from woof.static.highres import baseline_ocean_mask

    baseline = {"LU_INDEX": np.array([[17.0, 21.0], [10.0, 17.0]])}
    mask = baseline_ocean_mask(baseline)
    assert mask.dtype == bool
    np.testing.assert_array_equal(
        mask, np.array([[True, False], [False, True]]))
    # The inland lake category is NOT ocean.
    assert mask[0, 1] == False          # noqa: E712 -- the point is the value

    with pytest.raises(ValueError) as failure:
        baseline_ocean_mask({"HGT_M": np.zeros((2, 2))})
    assert "LU_INDEX" in str(failure.value)


def test_the_pilot_door_takes_the_same_split_as_the_production_door():
    """Two doors never disagree about one configuration.

    ``tools/run_highres_geog_pilot.py`` builds the same overrides on the
    same baseline as :func:`apply_highres_statics`.  If it called
    ``build_highres_overrides`` without the ocean mask, the pilot's
    comparison plots and metrics would report a coastline the production
    run does not produce.
    """
    import ast

    tool = (Path(__file__).resolve().parents[1]
            / "tools" / "run_highres_geog_pilot.py")
    tree = ast.parse(tool.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "build_highres_overrides"]
    assert calls, "the pilot no longer builds high-resolution overrides"
    for call in calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert "baseline_ocean" in keywords, (
            f"{tool.name}:{call.lineno} builds overrides without "
            "baseline_ocean, so the pilot door takes the crosswalk's "
            "lake-everywhere reading while the production door splits "
            "the sea out")


def test_the_module_docstring_describes_no_retired_coast_gate():
    """A retired guard's description is retired with it.

    The first thing a reader of this module sees is its docstring.  While
    the coast refusal existed the docstring stated it as current
    behaviour; the refusal is gone, replaced by the ocean/lake split, and
    a docstring still promising a refusal that cannot fire is a false
    statement about the program in the file that defines it.
    """
    import woof.static.highres_production as production

    assert not hasattr(production, "_require_coast_free"), (
        "the coast gate is back; this test guards its description, not "
        "its absence")

    text = production.__doc__
    assert "not coast-safe" not in text
    assert "coast gate" not in text
    # What replaced it is described instead, by the name a reader can grep.
    assert "_split_ocean_from_lake" in text
    # And the surviving footprint refusal says when it fires.
    assert "before anything is fetched" in text
