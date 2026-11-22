"""landmask_water on NetCDF input (door item 9).

A mapping's missing policy belongs to the mapping, not to the byte format.
ERA5 publishes soil temperature and soil moisture OVER WATER (about 271 K and
slightly negative moisture, with no missing mask), so an ERA5-shaped NetCDF
mapping declares ``landmask_water`` on its soil fields exactly as the DWD GRIB
mappings do.  Before this fix only the GRIB assembly applied the mask; the
NetCDF decoders (Python and the Rust engine) returned the water "soil"
unmasked, and every coastal land column then interpolated ocean values in.

The committed fixture ``landmask-water.nc`` beside the Rust crate's tests is
the file :func:`write_landmask_source` writes, so the cargo test
``landmask_water_masks_soil_over_water_on_the_netcdf_route`` and these tests
judge the same case.
"""

from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from woof.mapped_source import _decode_netcdf, decode_mapped_source

from test_mapped_source import _mapping, _write_mapping, _write_source


FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "tools" / "rw_wps" / "crates" / "mapped-engine" / "tests" / "fixtures"
)
FIXTURE_MAPPING = FIXTURES / "landmask-water.mapping.json"
FIXTURE_SOURCE = FIXTURES / "landmask-water.nc"

# Land fraction on a 3 x 4 grid.  0.5 is land (the threshold is "< 0.5 is
# water"), 0.49 and 0.3 are water: six water cells per plane.
LSM = np.asarray([
    [1.0, 0.0, 0.5, 0.49],
    [0.0, 0.8, 0.0, 1.0],
    [0.3, 1.0, 1.0, 0.0],
])
WATER = LSM < 0.5
WATER_SOIL_TEMPERATURE = 271.46
WATER_SOIL_MOISTURE = -0.013


def write_landmask_source(path: Path) -> None:
    """The ERA5-shaped fixture: two times, two soil layers, soil over water."""

    with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as dataset:
        for name, size in (("time", 2), ("soil", 2), ("latitude", 3), ("longitude", 4)):
            dataset.createDimension(name, size)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "hours since 2026-07-20 00:00:00"
        time.calendar = "standard"
        time[:] = [0.0, 6.0]
        latitude = dataset.createVariable("latitude", "f8", ("latitude",))
        latitude.units = "degrees_north"
        latitude[:] = [40.0, 40.25, 40.5]
        longitude = dataset.createVariable("longitude", "f8", ("longitude",))
        longitude.units = "degrees_east"
        longitude[:] = [10.0, 10.25, 10.5, 10.75]
        lsm = dataset.createVariable("lsm", "f8", ("time", "latitude", "longitude"))
        lsm.units = "1"
        lsm[:] = np.stack([LSM, LSM])
        stl = np.empty((2, 2, 3, 4))
        swvl = np.empty((2, 2, 3, 4))
        for time_index in range(2):
            for layer in range(2):
                stl[time_index, layer] = np.where(
                    WATER, WATER_SOIL_TEMPERATURE, 285.0 + layer + time_index)
                swvl[time_index, layer] = np.where(
                    WATER, WATER_SOIL_MOISTURE, 0.30 - 0.05 * layer)
        variable = dataset.createVariable(
            "stl", "f8", ("time", "soil", "latitude", "longitude"))
        variable.units = "K"
        variable[:] = stl
        variable = dataset.createVariable(
            "swvl", "f8", ("time", "soil", "latitude", "longitude"))
        variable.units = "m3 m-3"
        variable[:] = swvl


def _assert_soil_masked_on_water(values: np.ndarray, label: str) -> None:
    assert values.shape == (2, 3, 4), label
    for layer in values:
        assert np.isnan(layer[WATER]).all(), f"{label}: water soil not masked: {layer[WATER]}"
        assert np.isfinite(layer[~WATER]).all(), f"{label}: land soil lost"
        assert not np.isin(
            layer[~WATER], [WATER_SOIL_TEMPERATURE, WATER_SOIL_MOISTURE]).any()


def test_the_committed_fixture_is_the_generated_case(tmp_path):
    fresh = tmp_path / "fresh.nc"
    write_landmask_source(fresh)
    with netCDF4.Dataset(FIXTURE_SOURCE) as committed, netCDF4.Dataset(fresh) as written:
        assert set(committed.variables) == set(written.variables)
        for name in written.variables:
            np.testing.assert_array_equal(
                np.ma.filled(committed[name][:], np.nan),
                np.ma.filled(written[name][:], np.nan),
                err_msg=name,
            )


def test_netcdf_decoder_masks_landmask_water_soil_on_water_cells(tmp_path):
    source = tmp_path / "source.nc"
    write_landmask_source(source)
    mapping = json.loads(FIXTURE_MAPPING.read_text(encoding="utf-8"))
    collection = _decode_netcdf(mapping, [source])
    soil_keys = [key for key in collection.direct if key[2] != "land_fraction"]
    assert len(soil_keys) == 4  # two soil fields at two valid times
    for key in soil_keys:
        value = collection.direct[key]
        _assert_soil_masked_on_water(value.values, f"{key[2]} at {key[0]}")
        assert value.missing_count == 12  # six water cells on each of two layers
    for key in collection.direct:
        if key[2] == "land_fraction":
            assert collection.direct[key].missing_count == 0


def test_netcdf_decoder_without_the_policy_keeps_the_bytes(tmp_path):
    """The mask is the policy's doing: a reject mapping is byte-unchanged."""

    source = tmp_path / "source.nc"
    write_landmask_source(source)
    mapping = json.loads(FIXTURE_MAPPING.read_text(encoding="utf-8"))
    for name in ("soil_temperature", "volumetric_soil_moisture"):
        mapping["fields"][name]["missing"] = {"kind": "reject"}
    collection = _decode_netcdf(mapping, [source])
    for key, value in collection.direct.items():
        assert value.missing_count == 0, key
    soil = collection.direct[next(
        key for key in collection.direct if key[2] == "soil_temperature")]
    assert (soil.values[:, WATER] == WATER_SOIL_TEMPERATURE).all()


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_a_plain_netcdf_decode_masks_water_soil_on_either_engine(
        tmp_path, monkeypatch, engine):
    """The front door: a whole canonical frame from a CF NetCDF mapping."""

    monkeypatch.setenv("GPUWM_MAPPED_ENGINE", engine)
    mapping = _mapping()
    for name in ("soil_temperature", "volumetric_soil_moisture"):
        mapping["fields"][name]["missing"] = {"kind": "landmask_water"}
    mapping_path = tmp_path / "mapping.json"
    _write_mapping(mapping_path, mapping)
    source = tmp_path / "source.nc"
    _write_source(source)
    water = np.zeros((5, 6), dtype=bool)
    water[:, :2] = True
    with netCDF4.Dataset(source, "a") as dataset:
        land = np.where(water, 0.0, 0.75)
        dataset["land_fraction"][:] = np.stack([land, land])
        temperature = dataset["soil_temperature"][:]
        temperature[..., water] = WATER_SOIL_TEMPERATURE
        dataset["soil_temperature"][:] = temperature
    frames = decode_mapped_source(mapping_path, [source])
    assert len(frames) == 2
    for frame in frames:
        for name in ("soil_temperature", "volumetric_soil_moisture"):
            values = frame.fields[name].values
            assert values.shape == (4, 5, 6)
            assert np.isnan(values[:, water]).all(), f"{name} at {frame.valid_time}"
            assert np.isfinite(values[:, ~water]).all(), f"{name} at {frame.valid_time}"
