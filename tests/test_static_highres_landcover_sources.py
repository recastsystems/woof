"""Land cover is a table of sources, and the default one reaches every land.

The breakage these gates prevent, named: the only high-resolution land
cover was a United States collection, so everywhere else a high-resolution
run kept the 30-arc-second MODIS land use (1 km squares that show up in
evening 2 m temperature on a 500 m grid), and ``fields = "auto"`` ran
terrain alone there.  CGLC-MODIS-LCZ, the 100 m global land cover WRF and
WPS ship from version 4.5, is now a row of the source table and the default
everywhere from 60 S to 78 N; Annual NLCD stays selectable by name inside
the United States.  Its Local Climate Zone classes arrive as WRF's urban
category, because no urban canopy scheme runs.  Separately, the land-cover
window added its 2 km margin in the raster's own units, which on a
geographic raster is 2000 degrees: the window was the whole global raster.

The tests in the last two sections write a small geographic 8-bit GeoTIFF
in the same layout as the global raster and send it through the real Rust
window and warp; they fail on the tree before this change.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import re
import struct
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from woof.static import highres as highres_module
from woof.static.build import HALO
from woof.static.highres import (BoundRaster, CGLC_MODIS_LCZ_TO_MODIS21,
                                  MODIS21_CATEGORY_COUNT, MODIS21_ISICE,
                                  MODIS21_ISLAKE, MODIS21_ISURBAN,
                                  MODIS21_ISWATER, NLCD_TO_MODIS21_INLAND,
                                  WATER_FROM_SOURCE, WATER_RULES,
                                  WATER_SPLIT_BY_BASELINE,
                                  baseline_ocean_mask,
                                  build_highres_overrides, sha256_file)
from woof.static.highres_fetch import (DEFAULT_LANDCOVER_SOURCE,
                                        LANDCOVER_FETCH_KINDS,
                                        LANDCOVER_SOURCES, FootprintBBox,
                                        derive_landcover_window,
                                        domain_footprint, fetch_landcover,
                                        landcover_window_audit,
                                        margin_degrees,
                                        record_local_artifact)
from woof.static.highres_production import (HighresRefusal,
                                             HighresStaticConfig,
                                             _resolve_plan, _select_landcover,
                                             apply_highres_statics,
                                             parse_static_table)
from woof.static.lambert import LambertGrid

MODIS21_ATTRS = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
                 "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}

NOAH_TABLES = (Path(__file__).resolve().parent.parent / "woof" / "data"
               / "noah_tables")


def _grid(lat: float, lon: float, *, dx: float = 3000.0, n: int = 41
          ) -> LambertGrid:
    return LambertGrid(ref_lat=lat, ref_lon=lon, truelat1=30.0,
                       truelat2=60.0, stand_lon=lon, dx=dx, dy=dx,
                       e_we=n, e_sn=n)


#: Five footprints by what the sources say about them, not by place.
INSIDE_US = (38.68, -98.15)
OUTSIDE_US = (48.2, 16.4)
INSIDE_US_LATITUDES_ABROAD = (45.0, 12.0)
TROPICAL = (1.35, 103.8)
NORTH_OF_78 = (80.0, 20.0)


def _config(tmp_path, **kwargs) -> HighresStaticConfig:
    return HighresStaticConfig(enabled=True, cache_root=tmp_path, **kwargs)


def _plan(config, grid):
    return _resolve_plan(config, domain_footprint(grid, HALO))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_landcover_source_defaults_to_auto_and_is_echoed(tmp_path):
    config = parse_static_table(
        {"highres": {"enabled": True, "cache_root": "c"}},
        source="case.toml", base_dir=tmp_path)
    assert config.landcover_source == "auto"
    assert config.echo()["landcover_source"] == "auto"
    assert _select_landcover(config).source_id == "cglc-modis-lcz"
    assert DEFAULT_LANDCOVER_SOURCE == "cglc-modis-lcz"


def test_landcover_source_takes_each_row_and_refuses_an_unknown_one(
        tmp_path):
    for source_id in LANDCOVER_SOURCES:
        config = parse_static_table(
            {"highres": {"enabled": True, "cache_root": "c",
                         "landcover_source": source_id}},
            source="case.toml", base_dir=tmp_path)
        assert config.echo()["landcover_source"] == source_id
        assert _select_landcover(config).source_id == source_id
    with pytest.raises(ValueError) as failure:
        parse_static_table(
            {"highres": {"enabled": True, "cache_root": "c",
                         "landcover_source": "worldcover"}},
            source="case.toml", base_dir=tmp_path)
    message = str(failure.value)
    assert "cglc-modis-lcz" in message and "annual-nlcd" in message
    with pytest.raises(ValueError, match="landcover_sources"):
        parse_static_table(
            {"highres": {"enabled": True, "cache_root": "c",
                         "landcover_sources": "auto"}},
            source="case.toml", base_dir=tmp_path)


def test_two_landcover_sources_on_one_domain_write_two_receipts(tmp_path):
    """Changing the land-cover source is a different request; the second
    receipt must not overwrite the first."""
    grid = _grid(*NORTH_OF_78)
    written = {}
    for source_id in LANDCOVER_SOURCES:
        config = _config(tmp_path, fields="all", on_refuse="fallback-30s",
                         landcover_source=source_id)
        _, receipt = apply_highres_statics(
            {"HGT_M": np.zeros((40, 40))}, grid, config=config, domain_id=1,
            case_date=date(2024, 6, 1), landuse_attrs=MODIS21_ATTRS)
        assert receipt["status"] == "REFUSED"
        assert receipt["refusal"]["reason"] == "landcover-source-missing"
        written[source_id] = Path(receipt["receipt_path"])
    assert len(set(written.values())) == len(LANDCOVER_SOURCES)
    for source_id, path in written.items():
        assert f"lc-{source_id}" in path.name
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["config"]["landcover_source"] == source_id


# ---------------------------------------------------------------------------
# The plan: CGLC-MODIS-LCZ by default at home and abroad
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("where, terrain", [
    (INSIDE_US, "usgs-3dep-13as"),
    (OUTSIDE_US, "copernicus-dem-glo30"),
    (TROPICAL, "copernicus-dem-glo30"),
])
def test_auto_runs_the_full_overlay_wherever_the_default_reaches(
        tmp_path, where, terrain):
    config = _config(tmp_path)
    mode, coverage = _plan(config, _grid(*where))
    assert (mode, coverage.source_id) == ("all", terrain)
    assert _select_landcover(config).source_id == "cglc-modis-lcz"


def test_nlcd_pinned_inside_the_us_is_the_full_overlay(tmp_path):
    config = _config(tmp_path, landcover_source="annual-nlcd")
    mode, coverage = _plan(config, _grid(*INSIDE_US))
    assert (mode, coverage.source_id) == ("all", "usgs-3dep-13as")
    assert _select_landcover(config).source_id == "annual-nlcd"


def test_nlcd_pinned_wholly_outside_the_us_is_refused_or_terrain_only(
        tmp_path):
    auto = _config(tmp_path, landcover_source="annual-nlcd")
    mode, _ = _plan(auto, _grid(*OUTSIDE_US))
    assert mode == "terrain"
    with pytest.raises(HighresRefusal) as failure:
        _plan(_config(tmp_path, fields="all",
                      landcover_source="annual-nlcd"), _grid(*OUTSIDE_US))
    detail = failure.value.detail
    assert failure.value.reason == "landcover-source-missing"
    assert "'annual-nlcd'" in detail
    # The way out is named: the global collection, or terrain alone.
    assert "cglc-modis-lcz" in detail
    assert "fields = \"terrain\"" in detail


def test_north_of_78_the_default_hands_land_use_to_the_baseline(tmp_path):
    mode, _ = _plan(_config(tmp_path), _grid(*NORTH_OF_78))
    assert mode == "terrain"
    with pytest.raises(HighresRefusal) as failure:
        _plan(_config(tmp_path, fields="all"), _grid(*NORTH_OF_78))
    assert "'cglc-modis-lcz'" in failure.value.detail


@pytest.mark.parametrize("lat", [78.0, -60.0])
def test_a_domain_across_a_published_edge_runs_the_full_overlay(
        tmp_path, lat):
    grid = _grid(lat, 20.0)
    bbox = domain_footprint(grid, HALO)
    assert bbox.lat_min < lat < bbox.lat_max
    mode, _ = _plan(_config(tmp_path), grid)
    assert mode == "all"


def test_the_terrain_only_console_line_says_why(tmp_path, capsys):
    """The retired line claimed no global land cover was wired."""
    from woof.static.highres_fetch import SourceAbsent

    def absent(url, offset):
        raise SourceAbsent(f"{url} -> HTTP 404")

    for config, reason in (
            (_config(tmp_path, fields="terrain"), 'fields = "terrain"'),
            (_config(tmp_path), "cglc-modis-lcz is published over "
                                "latitudes -60..78")):
        _, receipt = apply_highres_statics(
            {"HGT_M": np.zeros((40, 40)), "LANDMASK": np.ones((40, 40)),
             "SOILTEMP": np.full((40, 40), 280.0)},
            _grid(*NORTH_OF_78), config=config, domain_id=1,
            case_date=date(2024, 6, 1), landuse_attrs=MODIS21_ATTRS,
            urlopen=absent)
        printed = capsys.readouterr().out
        assert "no global land-cover source is wired" not in printed
        assert reason in printed
        assert reason in receipt["terrain_only_reason"]
        assert reason in receipt["scope_statement"]


def test_a_pinned_source_abroad_is_placed_on_both_axes(tmp_path, capsys):
    """A domain inside Annual NLCD's latitudes but east of its longitudes.
    The reason used to name the latitudes alone, so the console and the
    receipt said a 45 N domain lay outside a source published over
    24..49.5 N.  It names both axes and the domain's own footprint."""
    from woof.static.highres_fetch import SourceAbsent

    def absent(url, offset):
        raise SourceAbsent(f"{url} -> HTTP 404")

    _, receipt = apply_highres_statics(
        {"HGT_M": np.zeros((40, 40)), "LANDMASK": np.ones((40, 40)),
         "SOILTEMP": np.full((40, 40), 280.0)},
        _grid(*INSIDE_US_LATITUDES_ABROAD),
        config=_config(tmp_path, landcover_source="annual-nlcd"),
        domain_id=1, case_date=date(2024, 6, 1),
        landuse_attrs=MODIS21_ATTRS, urlopen=absent)
    printed = capsys.readouterr().out
    reason = receipt["terrain_only_reason"]
    assert ("annual-nlcd is published over latitudes 24..49.5 and "
            "longitudes -125..-66.5") in reason
    footprint = re.search(
        r"spans latitudes (-?\d+\.\d+)\.\.(-?\d+\.\d+) and longitudes "
        r"(-?\d+\.\d+)\.\.(-?\d+\.\d+) with its halo", reason)
    assert footprint is not None, reason
    lat_min, lat_max, lon_min, lon_max = map(float, footprint.groups())
    lat, lon = INSIDE_US_LATITUDES_ABROAD
    assert 24.0 < lat_min < lat < lat_max < 49.5
    assert -66.5 < lon_min < lon < lon_max
    assert reason in printed
    assert reason in receipt["scope_statement"]


# ---------------------------------------------------------------------------
# The crosswalk and the engine's land-use inventory agree
# ---------------------------------------------------------------------------

def _vegparm_modis_block() -> tuple[int, dict[str, int]]:
    """(row count, scalar keys) of VEGPARM.TBL's MODIS block."""
    lines = [line.rstrip() for line in (NOAH_TABLES / "VEGPARM.TBL")
             .read_text(encoding="utf-8").splitlines()]
    start = lines.index("MODIFIED_IGBP_MODIS_NOAH")
    rows = int(lines[start + 1].split(",")[0])
    keys: dict[str, int] = {}
    for index in range(start + 2 + rows, len(lines) - 1):
        name = lines[index].strip()
        if name.startswith("Vegetation Parameters"):
            break
        if re.fullmatch(r"[A-Z_0-9]+", name):
            value = lines[index + 1].strip()
            if re.fullmatch(r"-?\d+", value):
                keys[name] = int(value)
    return rows, keys


def test_cglc_crosswalk_keeps_the_modis_classes_and_makes_lcz_urban():
    mapping = CGLC_MODIS_LCZ_TO_MODIS21
    for category in range(1, 22):
        assert mapping[category] == category
    for lcz in range(51, 62):
        assert mapping[lcz] == MODIS21_ISURBAN
    assert set(mapping) == set(range(1, 22)) | set(range(51, 62))
    # The unclassified value is no data, never a class.
    assert 0 not in mapping
    assert LANDCOVER_SOURCES["cglc-modis-lcz"].nodata == 0.0


def test_every_landcover_row_targets_the_engine_inventory():
    from woof.io.wrfout import _DEFAULT_LANDUSE_ATTRS as written

    assert MODIS21_CATEGORY_COUNT == written["NUM_LAND_CAT"]
    assert MODIS21_ISWATER == written["ISWATER"]
    assert MODIS21_ISLAKE == written["ISLAKE"]
    assert MODIS21_ISURBAN == written["ISURBAN"]
    assert MODIS21_ISICE == written["ISICE"]
    rows, keys = _vegparm_modis_block()
    lcz_classes = {keys[f"LCZ_{n}"] for n in range(1, 12)}
    assert lcz_classes == set(range(51, 62))
    water = {MODIS21_ISWATER, MODIS21_ISLAKE}

    for source_id, row in LANDCOVER_SOURCES.items():
        targets = set(row.crosswalk.values())
        assert targets <= set(range(1, MODIS21_CATEGORY_COUNT + 1)), source_id
        # Every land class has a VEGPARM row for Noah to read.
        assert all(t <= rows for t in targets - water), source_id
        assert row.water in WATER_RULES
        assert row.fetch in LANDCOVER_FETCH_KINDS
        assert row.coverage.role == "landcover"
        assert row.coverage.license_id and row.coverage.license_url
        assert row.coverage.attribution, source_id
        assert row.coverage.source_url.startswith("https://")
        assert row.first_year <= row.last_year
        if row.fetch == "whole-geotiff":
            assert row.pinned_bytes and row.pinned_bytes > 0
            assert re.fullmatch(r"[0-9a-f]{32}", row.pinned_md5 or "")
            assert re.fullmatch(r"[0-9a-f]{64}", row.pinned_sha256 or "")
        # A water rule that splits needs a lake target to split from.
        if row.water == WATER_SPLIT_BY_BASELINE:
            assert MODIS21_ISLAKE in targets
        else:
            assert {MODIS21_ISWATER, MODIS21_ISLAKE} <= targets

    cglc = LANDCOVER_SOURCES["cglc-modis-lcz"]
    for raw in lcz_classes:
        assert cglc.crosswalk[raw] == MODIS21_ISURBAN
    assert cglc.coverage.license_id == "CC-BY-4.0"
    assert "10.5281/zenodo.7670653" in cglc.coverage.attribution
    assert NLCD_TO_MODIS21_INLAND is LANDCOVER_SOURCES[
        "annual-nlcd"].crosswalk


def test_nlcd_developed_open_space_is_grassland_without_an_urban_canopy():
    # NLCD 21 is under 20 % impervious.  As ISURBAN it gave whole airport and
    # suburban cells Noah's bulk-urban soil, and the SF template's ASOS cells
    # stayed +5.4 F too warm all night (CITY-NIGHT-TEMP-2026-10-28.md).
    from woof.static.highres import landcover_legend

    plain, count = landcover_legend("annual-nlcd", sf_urban_physics=0)
    assert count == MODIS21_CATEGORY_COUNT
    assert plain[21] == 10
    assert {plain[raw] for raw in (22, 23, 24)} == {MODIS21_ISURBAN}
    # An urban canopy still takes 21 as low-density urban (its area
    # fraction comes from the impervious-class midpoint).
    canopy, _ = landcover_legend("annual-nlcd", sf_urban_physics=1)
    assert canopy[21] == canopy[22] != 10


def test_the_notice_and_data_page_carry_the_land_cover_licence():
    root = Path(__file__).resolve().parent.parent
    notice = (root / "NOTICE").read_text(encoding="utf-8")
    assert "CGLC-MODIS-LCZ" in notice
    assert "10.5281/zenodo.7670653" in notice
    assert "CC BY 4.0" in notice
    data = (root / "docs" / "public" / "DATA.md").read_text(encoding="utf-8")
    assert "CGLC-MODIS-LCZ" in data


# ---------------------------------------------------------------------------
# Water, per source
# ---------------------------------------------------------------------------

class _Stub:
    def __init__(self, source_id: str):
        self._source_id = source_id

    def receipt(self):
        return {"source_id": self._source_id}


N = 12


def _noah_baseline(n: int = N) -> dict[str, np.ndarray]:
    luf = np.zeros((21, n, n))
    luf[11] = 1.0
    soil = np.zeros((16, n, n))
    soil[2] = 1.0
    landmask = np.ones((n, n))
    lu_index = np.full((n, n), 12.0)
    # The baseline calls the east column ocean.
    luf[:, :, -1] = 0.0
    luf[16, :, -1] = 1.0
    landmask[:, -1] = 0.0
    lu_index[:, -1] = 17.0
    return {
        "HGT_M": np.full((n, n), 50.0), "LANDUSEF": luf,
        "LANDMASK": landmask, "LU_INDEX": lu_index,
        "SOILCTOP": soil.copy(), "SOILCBOT": soil.copy(),
        "SCT_DOM": np.full((n, n), 3.0), "SCB_DOM": np.full((n, n), 3.0),
        "GREENFRAC": np.full((12, n, n), 0.5),
        "LAI12M": np.full((12, n, n), 1.0),
        "ALBEDO12M": np.full((12, n, n), 18.0),
        "SNOALB": np.full((n, n), 60.0),
        "SOILTEMP": np.full((n, n), 285.0),
    }


#: (row, col) of a cell the source calls sea where the baseline says land,
#: and one the source calls lake where the baseline says ocean.
SOURCE_SEA = (4, 3)
SOURCE_LAKE_ON_BASELINE_OCEAN = (6, N - 1)


def _uniform_soil_fractions(sources, weights, grid, *, category_count):
    ny, nx = grid.e_sn - 1, grid.e_we - 1
    fractions = np.zeros((category_count, ny, nx))
    fractions[5] = 1.0
    return fractions, {"raw_component_total_percent_min": 100.0,
                       "raw_component_total_percent_max": 100.0,
                       "valid_source_pixels": ny * nx}


def _stub_categories(monkeypatch):
    def fake_categories(source, grid, mapping, *, category_count):
        ny, nx = grid.e_sn - 1, grid.e_we - 1
        halo = (ny - N) // 2
        fractions = np.zeros((category_count, ny, nx))
        fractions[11] = 1.0
        for (row, col), category in ((SOURCE_SEA, 17),
                                     (SOURCE_LAKE_ON_BASELINE_OCEAN, 21)):
            fractions[:, row + halo, col + halo] = 0.0
            fractions[category - 1, row + halo, col + halo] = 1.0
        return fractions

    monkeypatch.setattr(highres_module, "resample_mapped_categories",
                        fake_categories)
    monkeypatch.setattr(highres_module, "soilgrids_category_fractions",
                        _uniform_soil_fractions)


@pytest.mark.parametrize("rule, sea, lake", [
    (WATER_FROM_SOURCE, 17.0, 21.0),
    # One open-water class: the baseline decides, so the source's lake on
    # a baseline-ocean cell becomes ocean.
    (WATER_SPLIT_BY_BASELINE, 17.0, 17.0),
])
def test_each_water_rule_does_what_its_row_says(monkeypatch, rule, sea,
                                               lake):
    _stub_categories(monkeypatch)
    grid = _grid(40.0, -74.0, dx=1000.0, n=N + 1)
    baseline = _noah_baseline()
    fields, audit = build_highres_overrides(
        grid, terrain=None, landcover=_Stub("cglc-modis-lcz-2018"),
        soil_sources={("sand", "0-5cm"): _Stub("soilgrids-v2")},
        baseline_ocean=baseline_ocean_mask(baseline), halo=HALO,
        baseline=baseline, landcover_mapping=CGLC_MODIS_LCZ_TO_MODIS21,
        landcover_water=rule)
    assert fields["LU_INDEX"][SOURCE_SEA] == sea
    assert fields["LU_INDEX"][SOURCE_LAKE_ON_BASELINE_OCEAN] == lake
    method = audit["water_split"]["method"]
    if rule == WATER_FROM_SOURCE:
        assert "separates the sea" in method
        assert audit["water_split"]["ocean_cells_from_source"] >= 1
    else:
        assert "baseline LU_INDEX water field" in method


def test_an_unknown_water_rule_is_refused(monkeypatch):
    _stub_categories(monkeypatch)
    baseline = _noah_baseline()
    with pytest.raises(ValueError, match="landcover_water"):
        build_highres_overrides(
            _grid(40.0, -74.0, dx=1000.0, n=N + 1), terrain=None,
            landcover=_Stub("x"), soil_sources={},
            baseline_ocean=baseline_ocean_mask(baseline), halo=HALO,
            baseline=baseline, landcover_water="guess")


# ---------------------------------------------------------------------------
# The production door: which row ran, what each cell took
# ---------------------------------------------------------------------------

def _stub_fetch(monkeypatch, seen: dict, *, pixels=None):
    def fake_fetch_and_bind(bbox, cache_root, case_date, *, coverage, grid,
                            baseline, urlopen=None, landcover_source=None):
        seen["landcover_source"] = landcover_source
        year, anachronism = landcover_source.year_for(case_date)
        return (None, _Stub(landcover_source.bound_id(year)),
                {("sand", "0-5cm"): _Stub("soilgrids-v2")},
                {"landcover_source": landcover_source.source_id,
                 "landcover_year": year,
                 "landcover_anachronism_years": anachronism,
                 "landcover_window_audit": (
                     None if pixels is None
                     else {"category_pixels": pixels}),
                 "bytes_fetched": 0, "terrain_bytes_fetched": 0})

    monkeypatch.setattr(
        "woof.static.highres_production._fetch_and_bind",
        fake_fetch_and_bind)


def test_the_default_row_reaches_the_overlay_with_its_own_rules(
        tmp_path, monkeypatch, capsys):
    _stub_categories(monkeypatch)
    seen: dict = {}
    _stub_fetch(monkeypatch, seen,
                pixels={"0": 40, "12": 900, "17": 30, "51": 7, "56": 5})
    grid = _grid(40.0, -74.0, dx=1000.0, n=N + 1)
    baseline = _noah_baseline()
    fields, receipt = apply_highres_statics(
        baseline, grid, config=_config(tmp_path), domain_id=1,
        case_date=date(2024, 6, 1), landuse_attrs=MODIS21_ATTRS)

    assert seen["landcover_source"].source_id == "cglc-modis-lcz"
    assert receipt["status"] == "APPLIED"
    # The source's own lake stands on a baseline-ocean cell.
    assert fields["LU_INDEX"][SOURCE_LAKE_ON_BASELINE_OCEAN] == 21.0
    landcover = receipt["landcover"]
    assert landcover["source_id"] == "cglc-modis-lcz-2018"
    assert landcover["reference_year"] == 2018
    assert landcover["anachronism_years"] == 6
    assert "represents 2018" in landcover["anachronism"]
    assert landcover["urban_collapse"]["pixels"] == 12
    assert landcover["urban_collapse"]["source_classes"] == list(
        range(51, 62))
    assert landcover["unclassified_pixels"] == 40
    assert receipt["landcover_source"]["source_id"] == "cglc-modis-lcz"
    assert receipt["config"]["landcover_source"] == "auto"
    assert "10.5281/zenodo.7670653" in receipt["attributions"]["land_cover"]
    # Every cell of every field is accounted for by one group.
    for key, record in receipt["coverage"]["fields"].items():
        groups = record["cell_groups"]
        assert sum(g["cells"] for g in groups.values()) \
            == record["cell_count"], key
    land_use = receipt["coverage"]["fields"]["land_use"]["cell_groups"]
    assert land_use["source"]["takes"] == "cglc-modis-lcz-2018"
    assert land_use["source"]["cells"] == N * N
    printed = capsys.readouterr().out
    assert "land use cglc-modis-lcz-2018" in printed


def test_nlcd_pinned_runs_its_own_row(tmp_path, monkeypatch):
    _stub_categories(monkeypatch)
    seen: dict = {}
    _stub_fetch(monkeypatch, seen)
    grid = _grid(40.0, -74.0, dx=1000.0, n=N + 1)
    fields, receipt = apply_highres_statics(
        _noah_baseline(), grid,
        config=_config(tmp_path, landcover_source="annual-nlcd"),
        domain_id=1, case_date=date(2021, 5, 15),
        landuse_attrs=MODIS21_ATTRS)
    assert seen["landcover_source"].source_id == "annual-nlcd"
    assert receipt["landcover"]["source_id"] == "annual-nlcd-2021"
    assert "anachronism" not in receipt
    # One open-water class: the baseline's ocean wins.
    assert fields["LU_INDEX"][SOURCE_LAKE_ON_BASELINE_OCEAN] == 17.0


# ---------------------------------------------------------------------------
# The whole-GeoTIFF fetch holds its pins
# ---------------------------------------------------------------------------

class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _pinned_row(payload: bytes, **overrides):
    row = LANDCOVER_SOURCES["cglc-modis-lcz"]
    pins = {"url": "https://example.invalid/lc.tif",
            "pinned_bytes": len(payload),
            "pinned_md5": hashlib.md5(payload).hexdigest(),
            "pinned_sha256": hashlib.sha256(payload).hexdigest()}
    pins.update(overrides)
    return dataclasses.replace(row, **pins)


def test_a_whole_geotiff_is_fetched_once_and_held_to_its_pins(tmp_path):
    payload = b"II*\x00" + bytes(range(256)) * 8
    calls = []

    def urlopen(url, offset):
        calls.append((url, offset))
        return _Response(payload)

    row = _pinned_row(payload)
    downloaded, raster = fetch_landcover(row, 2018, tmp_path,
                                         urlopen=urlopen)
    assert downloaded == (raster,)
    assert raster.path == tmp_path / "cglc_modis_lcz" / "CGLC_MODIS_LCZ.tif"
    assert raster.sha256 == hashlib.sha256(payload).hexdigest()
    assert raster.cache_hit is False
    _, again = fetch_landcover(row, 2018, tmp_path, urlopen=urlopen)
    assert again.cache_hit is True
    assert len(calls) == 1

    # A changed pin causes a new fetch and rejects its unpublished bytes.
    # The prior verified payload survives a replacement that fails integrity.
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_landcover(_pinned_row(payload, pinned_sha256="0" * 64), 2018,
                        tmp_path, urlopen=urlopen)
    assert raster.path.read_bytes() == payload
    assert len(calls) == 2


@pytest.mark.parametrize("override, words", [
    ({"pinned_md5": "0" * 32}, "MD5"),
    ({"pinned_bytes": 5}, "bytes"),
])
def test_a_fresh_payload_that_misses_a_pin_is_refused_and_removed(
        tmp_path, override, words):
    payload = b"not the published raster" * 40

    def urlopen(url, offset):
        return _Response(payload)

    row = _pinned_row(payload, pinned_sha256=None, **override)
    with pytest.raises(ValueError, match=words) as failure:
        fetch_landcover(row, 2018, tmp_path, urlopen=urlopen)
    assert "fetches it again" in str(failure.value)
    assert not (tmp_path / "cglc_modis_lcz" / "CGLC_MODIS_LCZ.tif").exists()


# ---------------------------------------------------------------------------
# A small synthetic tile through the real Rust window and warp
# ---------------------------------------------------------------------------

STEP = 0.001                 # degrees per pixel
WEST, NORTH = 10.0, 45.3     # the tile's north-west corner
NX, NY = 1000, 600           # 1.0 x 0.6 degrees


def _bridge_or_fail():
    from woof.static import rust_bridge

    reason = rust_bridge.unavailable_reason()
    if reason is not None:
        pytest.fail(f"the Rust static-fields bridge is not loadable: {reason}")


def _geographic_byte_tiff(path: Path, values: np.ndarray) -> None:
    """One uncompressed strip, EPSG:4326, PixelIsArea: the global raster's
    geokeys and sample type in the smallest file that carries them."""
    values = np.ascontiguousarray(values, dtype=np.uint8)
    ny, nx = values.shape
    data = values.tobytes()
    formats = {3: "H", 4: "I", 12: "d"}
    tags = {
        256: (4, [nx]), 257: (4, [ny]), 258: (3, [8]), 259: (3, [1]),
        262: (3, [1]), 273: (4, [0]), 277: (3, [1]), 278: (4, [ny]),
        279: (4, [len(data)]), 339: (3, [1]),
        33550: (12, [STEP, STEP, 0.0]),
        33922: (12, [0.0, 0.0, 0.0, WEST, NORTH, 0.0]),
        34735: (3, [1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1,
                    2048, 0, 1, 4326]),
    }
    packed = {tag: struct.pack("<" + formats[kind] * len(v), *v)
              for tag, (kind, v) in tags.items()}
    ifd_size = 2 + 12 * len(tags) + 4
    cursor = 8 + ifd_size
    data_offset = cursor + sum(len(p) for p in packed.values() if len(p) > 4)
    packed[273] = struct.pack("<I", data_offset)
    ifd, extra = struct.pack("<H", len(tags)), b""
    for tag in sorted(tags):
        kind, v = tags[tag]
        payload = packed[tag]
        if len(payload) <= 4:
            ifd += struct.pack("<HHI", tag, kind, len(v))
            ifd += payload.ljust(4, b"\0")
        else:
            ifd += struct.pack("<HHII", tag, kind, len(v), cursor)
            extra += payload
            cursor += len(payload)
    ifd += struct.pack("<I", 0)
    path.write_bytes(b"II" + struct.pack("<HI", 42, 8) + ifd + extra + data)


def _synthetic_tile(tmp_path: Path):
    """West third Local Climate Zones, then cropland and forest, then
    lake, then sea; the southern tenth unclassified (0)."""
    values = np.zeros((NY, NX), dtype=np.uint8)
    cols = np.arange(NX)
    values[:, cols < 330] = (51 + (cols[cols < 330] // 30) % 11)
    values[:, (cols >= 330) & (cols < 600)] = 12
    values[::2, (cols >= 330) & (cols < 600)] = 4
    values[:, (cols >= 600) & (cols < 800)] = 21
    values[:, cols >= 800] = 17
    values[int(NY * 0.9):, :] = 0
    path = tmp_path / "tile" / "synthetic_lc.tif"
    path.parent.mkdir(parents=True)
    _geographic_byte_tiff(path, values)
    return record_local_artifact(path, url="file:synthetic_lc.tif"), values


def _bound(window) -> BoundRaster:
    row = LANDCOVER_SOURCES["cglc-modis-lcz"]
    return BoundRaster(
        path=window.path, sha256=window.sha256,
        source_id="synthetic-2018", role="landcover",
        source_url="https://example.invalid/source", license_id="test-only",
        license_url="https://example.invalid/licence",
        nominal_resolution="0.001 degree", nodata_override=row.nodata)


def test_the_window_of_a_geographic_tile_is_the_footprint_not_the_tile(
        tmp_path):
    """Read in the raster's own units the 2 km margin was 2000 degrees,
    and the window was the whole raster (the whole planet, for the global
    one)."""
    _bridge_or_fail()
    raster, values = _synthetic_tile(tmp_path)
    bbox = FootprintBBox(lat_min=44.95, lat_max=45.15,
                         lon_min=10.40, lon_max=10.70)
    window = derive_landcover_window(raster, bbox, tmp_path / "cache")
    audit = landcover_window_audit(window)
    assert audit is not None
    dlon, dlat = margin_degrees(bbox.lat_min, bbox.lat_max, 2000.0)
    expect_nx = (bbox.lon_max - bbox.lon_min + 2 * dlon) / STEP
    expect_ny = (bbox.lat_max - bbox.lat_min + 2 * dlat) / STEP
    ny, nx = audit["output_shape"]
    assert abs(nx - expect_nx) <= 2 and abs(ny - expect_ny) <= 2
    assert (ny, nx) != (NY, NX)
    assert audit["clipped_to_raster"] is False
    # Every pixel of the window is counted, by raw class.
    counts = {int(k): v for k, v in audit["category_pixels"].items()}
    assert sum(counts.values()) == ny * nx
    assert set(counts) <= {0, 4, 12, 21, 17} | set(range(51, 62))


def test_the_synthetic_tile_reaches_the_grid_with_lcz_as_urban(tmp_path):
    _bridge_or_fail()
    raster, _ = _synthetic_tile(tmp_path)
    grid = _grid(45.02, 10.55, dx=1000.0, n=51)
    bbox = domain_footprint(grid, HALO)
    window = derive_landcover_window(raster, bbox, tmp_path / "cache")
    fractions = highres_module.resample_mapped_categories(
        _bound(window), grid, CGLC_MODIS_LCZ_TO_MODIS21,
        category_count=MODIS21_CATEGORY_COUNT)
    assert fractions.shape[0] == MODIS21_CATEGORY_COUNT
    covered = np.all(np.isfinite(fractions), axis=0)
    assert covered.all()
    np.testing.assert_allclose(fractions.sum(axis=0), 1.0, rtol=1e-12)
    lat, lon = grid.latlon_mass()
    west = lon < 10.30
    lakes = (lon > 10.62) & (lon < 10.78)
    sea = lon > 10.82
    assert west.any() and lakes.any() and sea.any()
    assert np.all(fractions[MODIS21_ISURBAN - 1][west] == 1.0)
    assert np.all(fractions[MODIS21_ISLAKE - 1][lakes] == 1.0)
    assert np.all(fractions[MODIS21_ISWATER - 1][sea] == 1.0)


def test_cells_past_the_tiles_edges_take_the_baseline(tmp_path,
                                                     monkeypatch):
    """The coverage fallback at a published edge, on the real warp: the
    tile stops at 45.3 N (as the global raster stops at 78 N) and its
    southern strip is unclassified (as the open sea past the coastal zone
    is); cells there take the 30-arc-second baseline."""
    _bridge_or_fail()
    raster, _ = _synthetic_tile(tmp_path)
    n = 81
    grid = _grid(45.03, 10.45, dx=1000.0, n=n)
    bbox = domain_footprint(grid, HALO)
    assert bbox.lat_max > NORTH
    window = derive_landcover_window(raster, bbox, tmp_path / "cache")
    assert landcover_window_audit(window)["clipped_to_raster"] is True

    monkeypatch.setattr(highres_module, "soilgrids_category_fractions",
                        _uniform_soil_fractions)
    baseline = _noah_baseline(n - 1)
    baseline["LU_INDEX"][:] = 12.0
    baseline["LANDMASK"][:] = 1.0
    baseline["LANDUSEF"][:] = 0.0
    baseline["LANDUSEF"][11] = 1.0
    fields, audit = build_highres_overrides(
        grid, terrain=None, landcover=_bound(window),
        soil_sources={("sand", "0-5cm"): _Stub("soilgrids-v2")},
        baseline_ocean=baseline_ocean_mask(baseline), halo=HALO,
        baseline=baseline, landcover_mapping=CGLC_MODIS_LCZ_TO_MODIS21,
        landcover_water=WATER_FROM_SOURCE)
    lat, lon = grid.latlon_mass()
    north = lat > NORTH + 0.01
    unclassified = lat < 44.745
    assert north.any() and unclassified.any()
    for outside in (north, unclassified):
        np.testing.assert_array_equal(fields["LU_INDEX"][outside], 12.0)
        np.testing.assert_array_equal(
            fields["LANDUSEF"][:, outside],
            baseline["LANDUSEF"][:, outside])
    record = audit["coverage"]["fields"]["land_use"]
    assert record["cells_outside_coverage"] >= int(
        north.sum() + unclassified.sum())
    assert (record["cell_groups"]["baseline"]["cells"]
            == record["cells_outside_coverage"])
    assert record["cell_groups"]["blended"]["cells"] > 0
    # Inside the tile its own classes are in: the zones are urban.
    inside_zones = (lon > 10.10) & (lon < 10.30) & (lat > 44.85) & (
        lat < 45.2)
    assert inside_zones.any()
    np.testing.assert_array_equal(fields["LU_INDEX"][inside_zones],
                                  float(MODIS21_ISURBAN))


# ---------------------------------------------------------------------------
# The memory preflight of `woof go` reads a config carrying the overlay
# ---------------------------------------------------------------------------

def test_the_go_memory_preflight_loads_a_config_carrying_the_overlay(
        tmp_path, capsys):
    """`woof go` sizes the run through the memory preflight's loader,
    which split off [fetch] but not [static], so every wizard config that
    turned [static.highres] on stopped with "carries [static] ...
    unsplit" before a byte was fetched.  A bad key in the block is still
    refused by name."""
    from woof.cli import main as cli_main
    from woof.core.preflight import _load_experiment_any

    out = tmp_path / "area.toml"
    assert cli_main(["domain", "--point=35.3,-97.5", "--source", "gfs",
                     "--cycle", "2026-07-29T18", "--hours", "6",
                     "--card", "12gb", "--out", str(out)]) == 0
    capsys.readouterr()
    text = out.read_text(encoding="utf-8")
    block = ('\n[static.highres]\nenabled = true\n'
             'cache_root = "highres-cache"\nfields = "auto"\n'
             'landcover_source = "auto"\n')
    out.write_text(text + block, encoding="utf-8")
    exp = _load_experiment_any(out)
    assert exp.domains

    out.write_text(text + block.replace("landcover_source",
                                        "landcover_sources"),
                   encoding="utf-8")
    with pytest.raises(ValueError, match="landcover_sources"):
        _load_experiment_any(out)


# ---------------------------------------------------------------------------
# New land on a coast gets a land soil
# ---------------------------------------------------------------------------

def test_new_land_the_soil_source_misses_takes_the_nearest_land_soil(
        monkeypatch):
    """A cell the 100 m land cover calls land and the 30-arc-second maps
    call sea: the soil source masks it and the baseline's soil there is
    water.  Handed water soil on land, initialization gives the cell
    WRF's silty clay loam (ISLTYP 8) whatever soil surrounds it.  It takes
    the nearest land cell's soil instead."""
    new_land = (5, N - 1)

    def fake_categories(source, grid, mapping, *, category_count):
        ny, nx = grid.e_sn - 1, grid.e_we - 1
        fractions = np.zeros((category_count, ny, nx))
        fractions[12] = 1.0                       # urban everywhere
        return fractions

    def fake_soil(sources, weights, grid, *, category_count):
        ny, nx = grid.e_sn - 1, grid.e_we - 1
        halo = (ny - N) // 2
        fractions = np.zeros((category_count, ny, nx))
        fractions[5] = 1.0                        # soil category 6
        fractions[:, new_land[0] + halo, new_land[1] + halo] = np.nan
        return fractions, {"raw_component_total_percent_min": 100.0,
                           "raw_component_total_percent_max": 100.0,
                           "valid_source_pixels": ny * nx}

    monkeypatch.setattr(highres_module, "resample_mapped_categories",
                        fake_categories)
    monkeypatch.setattr(highres_module, "soilgrids_category_fractions",
                        fake_soil)
    baseline = _noah_baseline()
    for name in ("SOILCTOP", "SOILCBOT"):
        baseline[name][:, :, -1] = 0.0
        baseline[name][13, :, -1] = 1.0           # the baseline's sea soil
    fields, audit = build_highres_overrides(
        _grid(40.0, -74.0, dx=1000.0, n=N + 1), terrain=None,
        landcover=_Stub("cglc-modis-lcz-2018"),
        soil_sources={("sand", "0-5cm"): _Stub("soilgrids-v2")},
        baseline_ocean=baseline_ocean_mask(baseline), halo=HALO,
        baseline=baseline, landcover_mapping=CGLC_MODIS_LCZ_TO_MODIS21,
        landcover_water=WATER_FROM_SOURCE)
    assert fields["LANDMASK"][new_land] == 1.0
    assert fields["LU_INDEX"][new_land] == 13.0
    for name, dom in (("SOILCTOP", "SCT_DOM"), ("SOILCBOT", "SCB_DOM")):
        assert fields[dom][new_land] == 6.0
        np.testing.assert_allclose(fields[name][:, new_land[0], new_land[1]],
                                   fields[name][:, new_land[0], 0])
        assert not np.any((fields[dom] == 14.0) & (fields["LANDMASK"] > 0))
    assert audit["soil"]["top_0_30cm"][
        "water_soil_land_cells_from_nearest_land"] == 1
