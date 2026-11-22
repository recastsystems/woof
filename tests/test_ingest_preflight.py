"""Phase-5 Task-3 gates: complete CPU input/static/table preflight."""
from __future__ import annotations

import os

import argparse
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from woof.case_data import load_experiment_case
from woof.ingest.grib import (
    Era5DecodeResult, Era5Snapshot, _merge_catalog_partials,
    _merge_partials, _PartialSnapshot,
)
from woof.ingest.preflight import (CatalogBuildError, build_input_catalog,
                                    output_records, preflight_report,
                                    register_cli)
from woof.ingest.soil import (
    preprocess_noah_soil as _preprocess_noah_soil)


# These fixtures hand the soil router the raw SST/SKINTEMP pair on purpose:
# what they pin is WRF's OWN per-cell water-skin fallback
# (module_initialize_real.F:2844-2866), which is exactly what
# `water_temperature_policy = "wrf_compat"` names.  The router refuses that
# pair when nobody declares a decision, so the declaration is made once here
# instead of at every call site below, and the tests keep asserting the
# historical numbers they were written for.
def preprocess_noah_soil(fields, **kwargs):
    kwargs.setdefault("water_temperature_policy", "wrf_compat")
    return _preprocess_noah_soil(fields, **kwargs)

from woof.static.geog import GeogDataset
from woof.static.lambert import LambertGrid

from test_case_data import make_case_toml

REPO = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get("WOOF_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
MAY99 = Path(os.environ.get("WOOF_TEST_MAY99_DATA",
                    "gpuwm-fixture-unset/may99-data"))
MAY_FILES = (MAY99 / "era5_may1999_pl.grib",
             MAY99 / "era5_may1999_sl.grib")
MAY_HASHES = {
    "era5_may1999_pl.grib":
        "c695442394b0154cb219951f81194f0842bf68000d8c99b35f7aec6ba8f1d1f7",
    "era5_may1999_sl.grib":
        "2f4af209c8daeea05c4a1bcad516dda3a2652f01be7aa692b75d24e7b6e64f4b",
}
requires_may99 = pytest.mark.skipif(
    not all(path.is_file() for path in MAY_FILES)
    or not (BUNDLE / "era5_grib/Vtable.ERA5_CDO").is_file(),
    reason="staged May-1999 ERA5 or reference Vtable is absent",
)
requires_real74 = pytest.mark.skipif(
    not (BUNDLE / "era5_grib/era5_19740403.grb").is_file()
    or not (BUNDLE / "era5_grib/Vtable.ERA5_CDO").is_file(),
    reason="read-only 1974 reference bundle is absent",
)


def _valid_grib(path: Path) -> None:
    path.write_bytes(b"GRIB" + (12).to_bytes(3, "big") + b"\x01" + b"7777")


def _vtable(path: Path) -> None:
    path.write_text(
        "GRIB1| Level| From | To | metgrid | metgrid | Description |\n"
        "Param| Type |Level1|Level2| Name   | Units   |             |\n"
        " 130 | 100  |   *  |      | TT     | K       | Temperature |\n",
        encoding="utf-8",
    )


def _write_geog_dataset(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "index").write_text(
        "type=continuous\nprojection=regular_ll\ndx=90\ndy=30\n"
        "known_x=1\nknown_y=1\nknown_lat=0\nknown_lon=0\n"
        "wordsize=1\ntile_x=2\ntile_y=2\ntile_z=1\n",
        encoding="utf-8",
    )
    for xs in (1, 3):
        for ys in (1, 3):
            (path / f"{xs:05d}-{xs + 1:05d}.{ys:05d}-{ys + 1:05d}").write_bytes(
                bytes([1, 2, 3, 4])
            )


def _orography(path: Path, shape=(1, 20, 24)) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", shape[0])
        dataset.createDimension("south_north", shape[1])
        dataset.createDimension("west_east", shape[2])
        variable = dataset.createVariable(
            "SOILHGT", "f4", ("Time", "south_north", "west_east")
        )
        variable[:] = 100.0


def _fields(levels: np.ndarray, *, bad=None,
            shape=(2, 2)) -> dict[str, np.ndarray]:
    pressure_shape = (levels.size, *shape)
    fields = {
        "Z": np.full(pressure_shape, 5000.0, dtype=np.float64),
        "T": np.full(pressure_shape, 275.0, dtype=np.float64),
        "U": np.full(pressure_shape, 10.0, dtype=np.float64),
        "V": np.full(pressure_shape, 5.0, dtype=np.float64),
        "RH": np.full(pressure_shape, 50.0, dtype=np.float64),
        "U10": np.full(shape, 8.0, dtype=np.float64),
        "V10": np.full(shape, 3.0, dtype=np.float64),
        "T2": np.full(shape, 285.0, dtype=np.float64),
        "D2": np.full(shape, 280.0, dtype=np.float64),
        "LANDSEA": np.ones(shape, dtype=np.float64),
        "PSFC": np.full(shape, 100000.0, dtype=np.float64),
        "SKINTEMP": np.full(shape, 286.0, dtype=np.float64),
        "SST": np.full(shape, 290.0, dtype=np.float64),
        "SNOW_EC": np.zeros(shape, dtype=np.float64),
        "ST000007": np.full(shape, 284.0, dtype=np.float64),
        "ST007028": np.full(shape, 283.0, dtype=np.float64),
        "ST028100": np.full(shape, 282.0, dtype=np.float64),
        "ST100289": np.full(shape, 281.0, dtype=np.float64),
        "SM000007": np.full(shape, 0.25, dtype=np.float64),
        "SM007028": np.full(shape, 0.26, dtype=np.float64),
        "SM028100": np.full(shape, 0.27, dtype=np.float64),
        "SM100289": np.full(shape, 0.28, dtype=np.float64),
    }
    if bad == "nan":
        fields["T"][0, 1, 1] = np.nan
    elif bad == "sst-nan":
        fields["SST"][1, 1] = np.nan
    elif bad == "range":
        fields["U"][1, 0, 1] = 999.0
    elif bad == "both":
        fields["T"][0, 1, 1] = np.nan
        fields["U"][1, 0, 1] = 999.0
    return fields


class _Grid:
    e_we = 25
    e_sn = 21
    dx = 12000.0
    dy = 12000.0

    def latlon_mass(self):
        return (np.zeros((20, 24), dtype=np.float64),
                np.zeros((20, 24), dtype=np.float64))


@pytest.fixture
def synthetic_case(tmp_path, monkeypatch):
    config = make_case_toml(tmp_path)
    exp, data = load_experiment_case(config)
    forcing = tmp_path / "forcing" / "era5.grb"
    _valid_grib(forcing)
    _vtable(tmp_path / "Vtable.ERA5")
    _orography(tmp_path / "invariant.nc")
    from woof.geog_assets import WRF_DEFAULT_GEOG_DIRS
    for name in WRF_DEFAULT_GEOG_DIRS:
        _write_geog_dataset(tmp_path / "GEOG" / name)
    data = replace(data, forcing=(forcing,), forcing_interval_s=5400.0)
    monkeypatch.setattr("woof.ingest.preflight._grid_for",
                        lambda exp, data: _Grid())

    def install(times, *, levels=(100.0, 1000.0), bad=None,
                sst_bitmap_holes=None, landsea=None, seaice=None, snow=None,
                soilgeo=None, rh_warnings=None, sst_bitmap_provenance=True):
        times = tuple(times)
        level_array = np.asarray(levels, dtype=np.float64)
        shape = ((2, 2) if landsea is None else np.asarray(landsea).shape)
        holes = ((None,) * len(times) if sst_bitmap_holes is None
                 else tuple(sst_bitmap_holes))
        if len(holes) != len(times):
            raise ValueError("sst_bitmap_holes must align with times")
        snapshots = []
        bitmap_missing = {}
        for value, hole in zip(times, holes):
            fields = _fields(level_array, bad=bad, shape=shape)
            if landsea is not None:
                fields["LANDSEA"] = np.asarray(landsea, dtype=np.float64)
            if seaice is not None:
                fields["SEAICE"] = np.asarray(seaice, dtype=np.float64)
            if snow is not None:
                fields["SNOW_EC"] = np.asarray(snow, dtype=np.float64)
            if soilgeo is not None:
                fields["SOILGEO"] = np.asarray(soilgeo, dtype=np.float64)
            if hole is not None:
                mask = np.zeros(fields["SST"].shape, dtype=bool)
                mask[hole] = True
                fields["SST"][mask] = np.nan
                if sst_bitmap_provenance:
                    bitmap_missing[(value, "SST")] = mask
            if rh_warnings and value in rh_warnings:
                fields["RH"][0, 0, 0] = rh_warnings[value]
            snapshots.append(Era5Snapshot(
                valid_time=value,
                levels_hpa=level_array,
                latitude=np.linspace(-1.0, 1.0, shape[0], dtype=np.float64),
                longitude=np.linspace(-1.0, 1.0, shape[1], dtype=np.float64),
                fields=fields,
            ))
        snapshots = tuple(snapshots)
        sources = {
            (snapshot.valid_time, name): (forcing.resolve(),)
            for snapshot in snapshots for name in snapshot.fields
        }
        decoded = Era5DecodeResult(snapshots, sources, bitmap_missing)
        monkeypatch.setattr("woof.ingest.preflight.cached_era5_forcing",
                            lambda *args, **kwargs: decoded)
        return decoded

    return SimpleNamespace(exp=exp, data=data, install=install,
                           forcing=forcing, root=tmp_path)


def _retime(exp, run_seconds, history_interval_s=None):
    dc = exp.domains[0]
    cadence = (dc.history_interval_s if history_interval_s is None
               else history_interval_s)
    run = replace(dc.run, run_seconds=float(run_seconds),
                  output_interval_s=float(cadence))
    dc = replace(dc, history_interval_s=float(cadence), run=run)
    return replace(exp, run_seconds=float(run_seconds), domains=(dc,))


def test_all_synthetic_defects_are_named_in_one_complete_report(synthetic_case):
    case = synthetic_case
    start = case.exp.start_time
    exp = _retime(case.exp, 3 * 3600.0)
    case.install((start, start + timedelta(hours=3)),
                 levels=(200.0, 1000.0), bad="both")
    # Missing GEOG tile and wrong target-orography shape are independent.
    (case.root / "GEOG/topo_gmted2010_30s/00001-00002.00001-00002").unlink()
    (case.root / "invariant.nc").unlink()
    _orography(case.root / "invariant.nc", shape=(1, 19, 24))

    report = preflight_report(exp, case.data)
    text = report.format()
    assert not report.ok
    for defect in ("missing-time", "level-coverage", "nonfinite",
                   "meteorological-bounds", "missing-geog-tile",
                   "orography-shape"):
        assert f"[{defect}]" in text
    assert "file=" in text and "variable=T" in text and "index=(0, 1, 1)" in text
    assert "all independent checks ran" in text


@pytest.mark.parametrize("bad,code,variable,index", [
    ("nan", "nonfinite", "T", "(0, 1, 1)"),
    ("sst-nan", "nonfinite", "SST", "(1, 1)"),
    ("range", "meteorological-bounds", "U", "(1, 0, 1)"),
])
def test_field_defects_name_file_variable_and_index(
        synthetic_case, bad, code, variable, index):
    case = synthetic_case
    start = case.exp.start_time
    case.install((start, start + timedelta(minutes=90)), bad=bad)
    report = preflight_report(case.exp, case.data)
    matching = [issue for issue in report.failures if issue.code == code]
    assert matching
    rendered = matching[0].format()
    assert str(case.forcing.resolve()) in rendered
    assert f"variable={variable}" in rendered and f"index={index}" in rendered


def test_time_varying_native_sst_bitmap_hole_is_rejected(synthetic_case):
    case = synthetic_case
    start = case.exp.start_time
    case.install(
        (start, start + timedelta(minutes=90)),
        sst_bitmap_holes=((0, 0), (0, 1)),
    )

    report = preflight_report(case.exp, case.data)
    failures = [
        issue for issue in report.failures
        if issue.code == "nonfinite" and issue.variable == "SST"
    ]
    assert failures
    assert {issue.index for issue in failures} == {(0, 0), (0, 1)}
    assert all("bitmap changes between valid times" in issue.message
               for issue in failures)


def test_stable_native_sst_bitmap_hole_far_from_coast_is_rejected(
        synthetic_case):
    case = synthetic_case
    start = case.exp.start_time
    landsea = np.zeros((20, 20), dtype=np.float64)
    landsea[0, 0] = 1.0
    case.install(
        (start, start + timedelta(minutes=90)),
        sst_bitmap_holes=((15, 15), (15, 15)),
        landsea=landsea,
    )

    report = preflight_report(case.exp, case.data)
    failures = [
        issue for issue in report.failures
        if issue.code == "nonfinite" and issue.variable == "SST"
    ]
    assert failures
    assert {issue.index for issue in failures} == {(15, 15)}
    assert all("outside land/coastal support" in issue.message
               for issue in failures)


def test_stable_native_sst_hole_on_a_small_island_is_coastal(synthetic_case):
    # ERA5 leaves SST missing on cells of land fraction ~0.2 and up; an
    # island cell at 0.38 with open water all round is coastal, not open
    # water, and must not be refused (reproduced on typhoon and cyclone
    # boxes from the ARCO store).
    case = synthetic_case
    start = case.exp.start_time
    landsea = np.zeros((20, 20), dtype=np.float64)
    landsea[0, 0] = 1.0
    landsea[15, 15] = 0.38
    case.install(
        (start, start + timedelta(minutes=90)),
        sst_bitmap_holes=((15, 15), (15, 15)),
        landsea=landsea,
    )

    report = preflight_report(case.exp, case.data)
    assert not [issue for issue in report.failures
                if issue.code == "nonfinite" and issue.variable == "SST"]


@pytest.mark.parametrize("support", ["land", "coastal", "unproven"])
def test_repeated_era5_sst_holes_require_source_mask_support(
        synthetic_case, support):
    """Twelve repeated SST holes and two RH warnings match the reported pattern.

    The user's source file is not this synthetic fixture: these controls
    distinguish the accepted native land/coast case from an unproven NaN.
    """
    case = synthetic_case
    start = datetime(2016, 6, 22, 3)
    times = tuple(start + timedelta(hours=6 * index) for index in range(12))
    landsea = np.zeros((20, 20), dtype=np.float64)
    hole = (8, 8)
    landsea[8, 8 if support != "coastal" else 7] = 1.0
    case.install(times, sst_bitmap_holes=(hole,) * len(times), landsea=landsea,
                 rh_warnings={times[4]: 151.319, times[9]: 150.401},
                 sst_bitmap_provenance=support != "unproven")
    exp = replace(_retime(case.exp, 66 * 3600), start_time=start)
    data = replace(case.data, forcing_interval_s=6 * 3600.0)
    report = preflight_report(exp, data)
    assert report.catalog.valid_times == times
    assert report.run_ceiling_seconds == 237600
    warnings = [issue for issue in report.warnings
                if issue.code == "meteorological-bounds" and issue.variable == "RH"]
    assert len(warnings) == 2
    if support == "unproven":
        assert len(report.errors) == 12
        assert all(issue.variable == "SST" and issue.index == hole
                   and "no native GRIB missing-value bitmap" in issue.message
                   for issue in report.errors)
    else:
        assert report.ok, report.format()


def test_truncated_grib_fails_before_decoder_and_names_file(synthetic_case,
                                                            monkeypatch):
    case = synthetic_case
    case.forcing.write_bytes(b"GRIB\x00\x01")
    monkeypatch.setattr(
        "woof.ingest.preflight.cached_era5_forcing",
        lambda *args, **kwargs: pytest.fail("decoder must not run for truncated GRIB"),
    )
    with pytest.raises(CatalogBuildError, match="truncated GRIB1") as caught:
        build_input_catalog(case.data)
    assert str(case.forcing.resolve()) in str(caught.value)
    report = preflight_report(case.exp, case.data)
    assert "[grib-encoding]" in report.format()


def test_mixed_disjoint_time_grids_name_both_files_and_times(
        synthetic_case, monkeypatch):
    case = synthetic_case
    second = case.forcing.with_name("era5_day2.grb")
    _valid_grib(second)
    start = case.exp.start_time

    def partial(source, valid_time, shape):
        return _PartialSnapshot(
            source=source.resolve(), valid_time=valid_time,
            levels_hpa=(1000,),
            latitude=np.linspace(55.0, 18.0, shape[0]),
            longitude=np.linspace(-115.0, -52.0, shape[1]),
            fields={"T": np.zeros((1, *shape), dtype=np.float64)},
        )

    # Reviewer construction: a 0.25-degree day and a differently subsetted
    # day have disjoint valid times, identical inventories, and valid levels.
    partials = (
        partial(case.forcing, start, (149, 253)),
        partial(second, start + timedelta(minutes=90), (75, 127)),
    )
    monkeypatch.setattr(
        "woof.ingest.preflight.cached_era5_forcing",
        lambda *args, **kwargs: _merge_partials(partials),
    )
    data = replace(case.data, forcing=(case.forcing, second))
    report = preflight_report(case.exp, data)
    text = report.format()
    assert "[grib-decode]" in text
    assert "grids differ across valid times" in text
    for partial_snapshot in partials:
        assert str(partial_snapshot.source) in text
        assert partial_snapshot.valid_time.isoformat() in text


def test_catalog_discovery_ignores_unpaired_auxiliary_only_time(tmp_path):
    shape = (2, 3)
    start = datetime(1974, 4, 3, 12)
    latitude = np.linspace(40.0, 39.0, shape[0])
    longitude = np.linspace(-85.0, -83.0, shape[1])
    pressure = _PartialSnapshot(
        source=tmp_path / "pressure.grb", valid_time=start,
        levels_hpa=(1000,), latitude=latitude, longitude=longitude,
        fields={"T": np.full((1, *shape), 275.0)},
    )
    surface = _PartialSnapshot(
        source=tmp_path / "surface.grb", valid_time=start,
        levels_hpa=(), latitude=latitude, longitude=longitude,
        fields={"PMSL": np.full(shape, 101325.0)},
    )
    invariant = _PartialSnapshot(
        source=tmp_path / "supplement.grb",
        valid_time=start - timedelta(hours=12),
        levels_hpa=(), latitude=latitude, longitude=longitude,
        fields={"PMSL": np.full(shape, 101300.0)},
    )

    decoded = _merge_catalog_partials((invariant, pressure, surface))
    assert tuple(item.valid_time for item in decoded.snapshots) == (start,)
    assert set(decoded.snapshots[0].fields) == {"T", "PMSL"}
    assert decoded.field_sources[(start, "T")] == (pressure.source,)
    assert decoded.field_sources[(start, "PMSL")] == (surface.source,)

    with pytest.raises(ValueError, match="no pressure-level fields"):
        _merge_catalog_partials(
            (invariant, pressure, surface),
            valid_times=(invariant.valid_time,))


def test_geog_coverage_mask_rejects_unexplained_fill_unless_sparse(
        synthetic_case):
    path = synthetic_case.root / "GEOG/topo_gmted2010_30s"
    (path / "00001-00002.00001-00002").unlink()
    dataset = GeogDataset(path)
    mask = dataset.tile_coverage_mask(1, 4, 1, 4)
    assert mask.shape == (4, 4) and not mask[:2, :2].any()
    with pytest.raises(FileNotFoundError, match="unexplained fill.*source index"):
        dataset.read_window(1, 4, 1, 4)
    sparse = GeogDataset(path, sparse=True).read_window(1, 4, 1, 4)
    assert sparse.coverage is not None and not sparse.coverage[:2, :2].any()


def test_geog_out_of_extent_fill_is_not_a_missing_tile(synthetic_case):
    path = synthetic_case.root / "GEOG/topo_gmted2010_30s"
    dataset = GeogDataset(path)
    window = dataset.read_window(1, 4, 0, 4)
    assert window.coverage is not None
    assert not window.coverage[0].any()
    assert window.coverage[1:].all()
    assert dataset.missing_tiles(1, 4, 0, 4) == ()


def test_geog_preflight_window_matches_static_extended_halo(tmp_path):
    from woof.ingest.preflight import _geog_window
    from woof.static.build import _DomainSampler

    path = tmp_path / "fine_geog"
    path.mkdir()
    (path / "index").write_text(
        "type=continuous\nprojection=regular_ll\ndx=0.25\ndy=0.25\n"
        "known_x=721\nknown_y=361\nknown_lat=0\nknown_lon=0\n"
        "wordsize=1\ntile_x=1440\ntile_y=720\ntile_z=1\n",
        encoding="utf-8",
    )
    (path / "00001-01440.00001-00720").write_bytes(b"")
    dataset = GeogDataset(path)
    grid = LambertGrid(
        ref_lat=0.0, ref_lon=0.0, truelat1=30.0, truelat2=60.0,
        stand_lon=0.0, dx=12000.0, dy=12000.0, e_we=25, e_sn=21,
    )
    actual = _geog_window(dataset, grid, SimpleNamespace(), "fine")
    sampler = _DomainSampler(grid)
    x, y = dataset.latlon_to_xy(sampler.lat_c, sampler.lon_c)
    expected = (
        int(np.floor(x.min())) - 3, int(np.ceil(x.max())) + 3,
        max(1, int(np.floor(y.min())) - 3),
        min(dataset.ny_global, int(np.ceil(y.max())) + 3),
    )
    assert actual == expected

    mass_lat, mass_lon = grid.latlon_mass()
    mass_x, mass_y = dataset.latlon_to_xy(mass_lat, mass_lon)
    old_mass_only = (
        int(np.floor(mass_x.min())) - 4,
        int(np.ceil(mass_x.max())) + 4,
        max(1, int(np.floor(mass_y.min())) - 4),
        min(dataset.ny_global, int(np.ceil(mass_y.max())) + 4),
    )
    assert (actual[0] < old_mass_only[0] or actual[1] > old_mass_only[1]
            or actual[2] < old_mass_only[2] or actual[3] > old_mass_only[3])


def test_non_even_hour_forcing_actual_lbc_deltas_and_subhour_outputs(
        synthetic_case):
    case = synthetic_case
    start = case.exp.start_time
    exp = _retime(case.exp, 3 * 3600.0, history_interval_s=1800.0)
    case.install((start, start + timedelta(minutes=90),
                  start + timedelta(minutes=180)))
    report = preflight_report(exp, case.data)
    assert report.ok, report.format()
    assert [record.delta_seconds for record in report.catalog.lbc_records] == [
        5400.0, 5400.0]
    assert report.run_ceiling_seconds == 10800.0
    records = output_records(exp, 1)
    assert records[1][0] == 30                 # 1800 s / 60 s
    assert records[1][1].minute == 30
    assert records[1][2].endswith("1999-05-03_12_30_00")
    assert all(":" not in record[2] for record in records)


def test_synthetic_preflight_sea_ice_snow_reaches_ice_surface_branch(
        synthetic_case):
    case = synthetic_case
    start = case.exp.start_time
    landsea = np.array([[1.0, 0.0], [1.0, 0.0]])
    seaice = np.array([[0.7, 0.8], [0.0, 0.2]])
    snow = np.array([[0.01, 0.02], [0.03, 0.04]])
    case.install(
        (start, start + timedelta(minutes=90)), landsea=landsea,
        seaice=seaice, snow=snow)
    report = preflight_report(case.exp, case.data)
    assert report.ok, report.format()
    assert "SEAICE" in report.catalog.inventory
    assert report.catalog.masks[(start, "SEAICE")].count == 2

    fields = report.catalog.snapshots[0].fields
    soil = preprocess_noah_soil(
        fields, soil_type=np.where(landsea >= 0.5, 6, 14),
        deep_soil_temperature=np.full(landsea.shape, 280.0))
    # Source land masks away the spurious land-cell fraction.  The real
    # sea-ice cell is XLAND=1 and XICE=1, the exact Noah ICE=1 predicate.
    # Snow reconciliation is surface-independent, matching WRF real init.
    assert soil.xice[0, 0] == 0.0
    assert soil.xland[0, 1] == 1.0 and soil.xice[0, 1] == 1.0
    assert soil.xland[1, 1] == 2.0 and soil.xice[1, 1] == 0.0
    assert soil.snow_water[0, 1] == 20.0
    assert soil.snow_water[1, 1] == 40.0
    np.testing.assert_array_equal(soil.soil_moisture[:, 0, 1], 1.0)
    np.testing.assert_array_equal(soil.liquid_moisture[:, 0, 1], 0.0)
    assert soil.deep_soil_temperature[0, 1] == 271.4
    midpoints = (np.arange(4) + 0.5) * 0.75
    expected_tslb = ((3.0 - midpoints) * soil.tsk[0, 1]
                     + midpoints * 271.4) / 3.0
    np.testing.assert_allclose(
        soil.soil_temperature[:, 0, 1], expected_tslb, rtol=2.5e-7, atol=0)  # REAL arithmetic, as real.exe


def test_preflight_accepts_wrf_seaice_flag_values_for_init_repair(
        synthetic_case):
    case = synthetic_case
    start = case.exp.start_time
    seaice = np.array([[255.0, 0.8], [0.0, 0.0]])
    case.install((start, start + timedelta(minutes=90)), seaice=seaice)
    report = preflight_report(case.exp, case.data)
    assert report.ok, report.format()
    assert report.catalog.masks[(start, "SEAICE")].count == 1


def test_preflight_names_declared_and_catalog_orography_conflict(
        synthetic_case):
    case = synthetic_case
    start = case.exp.start_time
    case.install(
        (start, start + timedelta(minutes=90)),
        soilgeo=np.full((2, 2), 981.0))
    report = preflight_report(case.exp, case.data)
    assert not report.ok
    text = report.format()
    assert "declared source_orography" in text
    assert "SOILGEO via era5_z_invariant" in text


def test_catalog_hashes_are_run_provenance_and_ceiling_is_coverage_derived(
        synthetic_case):
    case = synthetic_case
    start = case.exp.start_time
    case.install((start, start + timedelta(minutes=90)))
    catalog = build_input_catalog(case.data)
    assert catalog.run_ceiling_seconds == 5400.0
    assert catalog.run_provenance["input_catalog_sha256"] == catalog.fingerprint
    assert catalog.run_provenance["files"]
    forcing = next(item for item in catalog.files if item.role == "forcing")
    assert forcing.sha256 == catalog.file_hashes[str(case.forcing.resolve())]


def test_table_validation_reports_checksum_and_shape(synthetic_case, tmp_path):
    case = synthetic_case
    start = case.exp.start_time
    case.install((start, start + timedelta(minutes=90)))
    root = tmp_path / "broken_tables"
    (root / "kf_lutab").mkdir(parents=True)
    np.savez(root / "kf_lutab/kf_lutab.npz",
             temperature=np.zeros((2, 2), dtype=np.float64))
    data = SimpleNamespace(**case.data.__dict__, table_root=root)
    report = preflight_report(case.exp, data)
    text = report.format()
    assert "[table-checksum]" in text
    assert "[table-shape]" in text or "[table-inventory]" in text
    assert "kf_lutab.npz" in text


def test_cli_registrar_returns_failure_without_importing_cupy(monkeypatch,
                                                               capsys,
                                                               tmp_path):
    from woof.ingest import preflight
    sentinel_exp = object()
    sentinel_data = object()
    monkeypatch.setattr("woof.case_data.load_experiment_case",
                        lambda path: (sentinel_exp, sentinel_data))
    report = preflight.PreflightReport(
        preflight._empty_catalog("ERA5"),
        (preflight.PreflightIssue("named-defect", "broken input"),), ())
    monkeypatch.setattr(preflight, "preflight_report",
                        lambda exp, data: report)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    register_cli(subparsers)
    # The path must EXIST and must carry the shape `_check_command`
    # routes on, even though the loader below is mocked: a4417efb
    # refuses a path that is not a readable regular file before anything
    # opens it, and `_check_command` returns 0 early for a legacy-shaped
    # config or one with no [case_data] table.  The BODY is never read
    # -- `load_experiment_case` is replaced -- so what is written here
    # is the shape of the route, not a case fixture.
    config = tmp_path / "case.toml"
    config.write_text("[experiment]\n[[domain]]\n[case_data]\n",
                      encoding="utf-8")
    args = parser.parse_args(["check", str(config)])
    assert args.ingest_preflight_handler(args) == 1
    assert "named-defect" in capsys.readouterr().out


@pytest.mark.parametrize("order", ("input-first", "memory-first"))
def test_input_and_memory_cli_registrars_compose_in_both_orders(
        monkeypatch, order):
    from woof.core import preflight as memory_preflight
    from woof.ingest import preflight as input_preflight

    calls = []
    input_result = [0]

    def input_handler(args):
        calls.append("input")
        return input_result[0]

    def memory_handler(args):
        calls.append("memory")
        return 0

    monkeypatch.setattr(input_preflight, "_check_command", input_handler)
    monkeypatch.setattr(memory_preflight, "check_main", memory_handler)
    parser = argparse.ArgumentParser(prog="woof")
    subparsers = parser.add_subparsers(dest="command", required=True)
    registrars = {
        "input": input_preflight.register_cli,
        "memory": memory_preflight.register_cli,
    }
    sequence = (("input", "memory") if order == "input-first"
                else ("memory", "input"))
    for name in sequence:
        registrars[name](subparsers)

    check = subparsers.choices["check"]
    assert check.get_default("func") is memory_handler
    # Controller handoff: cheap CPU input validation first; estimator/--alloc
    # runs only after it passes.
    check.set_defaults(
        func=lambda args: args.ingest_preflight_handler(args)
        or memory_preflight.check_main(args)
    )
    args = parser.parse_args(["check", "case.toml", "--alloc"])
    assert args.alloc is True
    assert args.func(args) == 0
    assert calls == ["input", "memory"]

    calls.clear()
    input_result[0] = 1
    args = parser.parse_args(["check", "case.toml", "--alloc"])
    assert args.func(args) == 1
    assert calls == ["input"]


def test_composed_check_accepts_a_runconfig_shaped_config(capsys):
    """`woof check` on a legacy RunConfig TOML reaches the estimator.

    The composed CLI used to hand every config to
    ``load_experiment_case``, which rejects the RunConfig shape for
    lacking ``[case_data]`` -- while the estimator registrar wraps the
    same file as a one-domain experiment.  The input preflight now skips
    the shape it has nothing to say about, so both paths agree.
    """
    import cupy  # noqa: F401  (marks this test for -m "not gpu")

    import woof.cli as cli

    rc = cli.main(["check", str(REPO / "configs/real74_d01.toml"),
                   "--budget-gib", "100"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "declares no [case_data] inputs" in out
    assert "memory preflight" in out


def test_composed_check_json_keeps_stdout_machine_readable(capsys):
    """RunConfig-shape skip note goes to stderr under --json."""
    import json

    import cupy  # noqa: F401  (marks this test for -m "not gpu")

    import woof.cli as cli

    rc = cli.main(["check", str(REPO / "configs/real74_d01.toml"),
                   "--budget-gib", "100", "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "declares no [case_data] inputs" in captured.err
    payload = json.loads(captured.out)
    assert payload["gates"]["alloc_estimate_le_wddm_budget"] is True


def test_composed_check_still_routes_experiment_shape_to_input_preflight(
        tmp_path, monkeypatch, capsys):
    """An [experiment]-shaped config still runs the input preflight
    first, and its failure short-circuits the estimator."""
    import woof.cli as cli
    from woof.ingest import preflight as input_preflight

    config = tmp_path / "case.toml"
    # A [case_data] table must be present: the composed check now fails
    # the no-[case_data] experiment shape with its own actionable message
    # BEFORE the loader runs (tests/test_domain_wizard.py pins that path).
    config.write_text("[experiment]\n[case_data]\n", encoding="utf-8")
    sentinel = (object(), object())
    monkeypatch.setattr("woof.case_data.load_experiment_case",
                        lambda path: sentinel)
    report = input_preflight.PreflightReport(
        input_preflight._empty_catalog("ERA5"),
        (input_preflight.PreflightIssue("named-defect", "broken input"),),
        ())
    monkeypatch.setattr(input_preflight, "preflight_report",
                        lambda exp, data: report)
    rc = cli.main(["check", str(config), "--budget-gib", "100"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "named-defect" in out
    assert "memory preflight" not in out


def test_bridge_dependency_is_portable_vendored_and_pinned():
    manifest = (REPO / "tools/grib1_bridge/Cargo.toml").read_text()
    vendor = (REPO / "tools/grib1_bridge/vendor/VENDOR.md").read_text()
    lockfile = (REPO / "tools/grib1_bridge/Cargo.lock").read_text()
    parser = (REPO / "tools/grib1_bridge/vendor/grib-core/src/grib1/"
              "parser.rs").read_text()
    assert 'path = "vendor/grib-core"' in manifest
    assert "Documents/Codex" not in manifest
    assert "fe9797f86c8958b5a625a4a7682c6b6aeff6b309" in vendor
    assert "7c036c2b18c0a9bb014eddd55cdacf84a244fed0" in vendor
    # The product VENDOR.md spells the toolchain-compatibility rationale
    # "Rust 1.75" (GFS-bridge rewrite); the certified branch's stronger
    # assertion set is kept, re-anchored on the resolved document.
    assert "Rust 1.75" in vendor
    assert "version = 3" in lockfile and "version = 4" not in lockfile
    assert "is_none_or" not in parser
    assert parser.count("map_or(true, |end| end > data.len())") == 6
    assert (REPO / "tools/grib1_bridge/.cargo/config.toml").is_file()


@requires_real74
def test_real74_case_data_preflight_passes_end_to_end():
    exp, data = load_experiment_case(REPO / "configs/real74_d01_exp.toml")
    report = preflight_report(exp, data)
    assert report.ok, report.format()
    assert "SEAICE" not in report.catalog.inventory
    masks = [
        report.catalog.masks[(valid_time, "SST_MISSING")].mask
        for valid_time in report.catalog.valid_times
    ]
    assert all(np.array_equal(mask, masks[0]) for mask in masks[1:])


@requires_may99
def test_real_may1999_native_grib1_catalog_smoke():
    base_exp, base_data = load_experiment_case(
        REPO / "configs/real74_d01_exp.toml"
    )
    exp = replace(base_exp, start_time=datetime(1999, 5, 3, 12))
    data = replace(
        base_data, forcing=MAY_FILES, source_orography=None, co2_vmr=None)
    report = preflight_report(exp, data)
    assert report.ok, report.format()
    catalog = report.catalog
    assert catalog.valid_times == (
        datetime(1999, 5, 3, 12), datetime(1999, 5, 3, 18),
        datetime(1999, 5, 4, 0))
    assert len(catalog.raw_valid_times) == 6  # CDS date x time cross-product
    assert len(catalog.levels_hpa) == 37
    assert catalog.levels_hpa[0] == 1.0 and catalog.levels_hpa[-1] == 1000.0
    assert set(catalog.inventory) == {
        "Z", "T", "U", "V", "RH", "U10", "V10", "T2", "D2",
        "LANDSEA", "PSFC", "PMSL", "SKINTEMP", "SST", "SEAICE",
        "SNOW_EC", "SOILGEO", "ST000007", "ST007028", "ST028100",
        "ST100289", "SM000007", "SM007028", "SM028100", "SM100289",
    }
    forcing_records = {item.path.name: item for item in catalog.files
                       if item.role == "forcing"}
    assert {name: item.sha256 for name, item in forcing_records.items()} == MAY_HASHES
    assert catalog.provenance["encoding"].startswith("native GRIB1")
    for valid_time in catalog.valid_times:
        sst = catalog.masks[(valid_time, "SST_MISSING")].mask
        seaice = catalog.masks[(valid_time, "SEAICE_MISSING")].mask
        np.testing.assert_array_equal(sst, seaice)


def test_both_doors_admit_a_descending_pressure_ladder_identically(
        synthetic_case):
    """One admission for the level ordering, read by ``run`` and ``check``.

    ERA5 publishes ``pressure_level`` DESCENDING, 1000 hPa first.  The run
    door's vertical interpolation takes a source pressure column strictly
    monotonic in either direction and reverses an ascending one
    (woof/ingest/cpu_backend.py), so a keyless ARCO window of exactly
    that shape initialized and ran end to end; the check door refused the
    same file with "pressure levels must be finite/strictly increasing"
    and never reached its memory section.  Two more verdicts rode on the
    same premise: read in file order, a descending ladder makes
    ``levels[0]`` the deepest level and ``levels[-1]`` the highest, which
    inverts both the p_top coverage test and the 1000 hPa terrain-column
    test.

    The doors are pinned EQUAL here: the same forcing published in either
    order gives one catalog ladder and one report.
    """

    case = synthetic_case
    exp = _retime(case.exp, 5400.0)
    times = (case.exp.start_time,
             case.exp.start_time + timedelta(seconds=5400))
    published = (1000.0, 500.0, 200.0, 50.0)

    case.install(times, levels=published)
    descending_run = build_input_catalog(case.data)      # the run door's gate
    descending_check = preflight_report(exp, case.data)  # `woof check`

    case.install(times, levels=tuple(reversed(published)))
    ascending_run = build_input_catalog(case.data)
    ascending_check = preflight_report(exp, case.data)

    # The check door refuses nothing about the ordering, and reports
    # exactly what it reports for the ascending publication.
    assert not [issue for issue in descending_check.failures
                if issue.code in ("levels", "level-coverage",
                                  "below-surface-support")],         descending_check.format()
    assert ([(issue.code, issue.message, issue.severity)
             for issue in descending_check.failures]
            == [(issue.code, issue.message, issue.severity)
                for issue in ascending_check.failures])
    assert descending_check.ok, descending_check.format()

    # The run door admits both, and the catalog BOTH doors build through
    # carries one ascending ladder.
    assert descending_run.levels_hpa == (50.0, 200.0, 500.0, 1000.0)
    assert descending_run.levels_hpa == ascending_run.levels_hpa

    # A ladder neither door can order is still named, by one function,
    # and it names the refusal the run would otherwise hit downstream.
    from woof.ingest.preflight import admitted_pressure_ladder
    assert admitted_pressure_ladder((1000.0, 500.0, 500.0, 50.0)) is None
    assert admitted_pressure_ladder((1000.0, 50.0, 500.0)) is None
    assert admitted_pressure_ladder((1000.0, np.nan)) is None
    case.install(times, levels=(1000.0, 500.0, 500.0, 50.0))
    repeated = preflight_report(exp, case.data)
    refusals = [issue for issue in repeated.failures if issue.code == "levels"]
    assert len(refusals) == 1
    assert "strictly monotonic in every column" in refusals[0].message
    assert "one record per pressure level" in refusals[0].message


def test_run_start_refuses_a_forcing_product_missing_required_variables():
    """``woof run`` must name the absent variables, not trip over them.

    The run path calls ``build_input_catalog`` and NEVER
    ``preflight_report`` -- that lives behind ``woof check``.  So a
    pressure-level-only ERA5 download (the two CDS products are separate
    downloads, and taking only one is an easy mistake) used to run the
    full hashed decode, then die far downstream on whichever consumer
    indexed a surface field first: a KeyError naming soil fields on the
    single-domain path, and a different KeyError about D2/RH2 on the
    multi-domain one.  Neither said "your forcing has no surface data".

    Only ABSENCE is gated here; value judgements stay advisory in
    ``woof check``.
    """

    from woof.ingest.preflight import (
        CatalogBuildError, PreflightIssue, _REQUIRED_SURFACE,
        _missing_required_inventory)

    class _Catalog:
        inventory = ("Z", "T", "U", "V", "RH")
        snapshots = ()

    missing = _missing_required_inventory(_Catalog())
    assert set(missing) == _REQUIRED_SURFACE
    assert "SKINTEMP" in missing and "SM000007" in missing
    # Pressure-level variables the product DOES carry are not reported.
    assert not {"Z", "T", "U", "V", "RH"} & set(missing)

    issue = PreflightIssue("inventory", f"forcing inventory is missing {missing}")
    error = CatalogBuildError([issue])
    assert "SKINTEMP" in str(error)


def test_a_complete_forcing_inventory_is_not_gated():
    """The gate must be silent on every product that can actually run."""

    from woof.ingest.preflight import (
        _REQUIRED_PRESSURE, _REQUIRED_SURFACE, _missing_required_inventory)

    class _Catalog:
        inventory = tuple(sorted(
            _REQUIRED_PRESSURE | _REQUIRED_SURFACE | {"PMSL", "SOILGEO"}))
        snapshots = ()

    assert _missing_required_inventory(_Catalog()) == []


def test_child_catalog_retains_selected_lake_geography_with_shared_baseline(
        synthetic_case):
    from woof.static.build import geog_selection_from_catalog
    case = synthetic_case
    start = case.exp.start_time
    case.install((start, start + timedelta(minutes=90)))
    selected = case.root / "selected-geog"
    selected.mkdir()
    for dataset in case.data.geog_root.iterdir():
        (selected / dataset.name).symlink_to(dataset, target_is_directory=True)
    lake = selected / "lake_depth"
    lake.mkdir()
    (lake / "index").write_text("selected supplemental lake dataset\n")
    catalog = build_input_catalog(replace(case.data, geog_root=selected))
    # Baseline hashes point into the donor; optional lake physics must retain
    # the separately selected supplemental dataset in the declared root.
    assert catalog.geog_root == selected
    selection = geog_selection_from_catalog(catalog, 1)
    assert selection.root == selected
    assert (selection.root / "lake_depth" / "index").is_file()
    assert not (case.data.geog_root / "lake_depth").exists()
