"""Whole-variable reads of classic netCDF files on threads return netCDF4-python's arrays (no GPU).

THE COST THIS REMOVES (30 km globe, V5090 box, 2026-10-27): Mesh.from_netcdf read every grid/init variable through
netCDF4 one after another on the forecast's main thread (a 30 km init: 6.3 s for 135 variables, 0.7 s on threads).
"""
import numpy as np
import pytest
from netCDF4 import Dataset

from woof.hex import classic_netcdf
from woof.hex.mesh import _read_netcdf


def _write(path, data_model):
    with Dataset(path, "w", format=data_model) as ds:
        ds.createDimension("Time", None)
        ds.createDimension("nCells", 7)
        ds.createDimension("nVertLevels", 3)
        ds.createDimension("StrLen", 4)
        ds.title = "classic"
        rng = np.random.default_rng(3)
        for name, kind, dims in (("areaCell", "f8", ("nCells",)), ("zgrid", "f4", ("nCells", "nVertLevels")),
                                 ("cellsOnCell", "i4", ("nCells", "nVertLevels")), ("mask", "i1", ("nCells",)),
                                 ("short", "i2", ("nCells",)), ("theta", "f4", ("Time", "nCells", "nVertLevels")),
                                 ("u", "f4", ("Time", "nCells"))):
            var = ds.createVariable(name, kind, dims)
            var.units = "x"
            shape = tuple(1 if d == "Time" else len(ds.dimensions[d]) for d in dims)
            var[...] = (rng.standard_normal(shape) * 100).astype(kind)
        ds.createVariable("xtime", "S1", ("Time", "StrLen"))[0, :] = np.array(list("2025"), dtype="S1")
        scaled = ds.createVariable("scaled", "i2", ("nCells",))
        scaled.scale_factor = 0.5
        scaled[...] = np.arange(7) * 0.5
        ds.createVariable("scalar", "f4", ())[...] = 3.5


@pytest.mark.parametrize("data_model", ["NETCDF3_CLASSIC", "NETCDF3_64BIT_OFFSET", "NETCDF3_64BIT_DATA"])
def test_threaded_reads_are_netcdf4s_arrays(tmp_path, monkeypatch, data_model):
    path = tmp_path / "file.nc"
    _write(path, data_model)
    table, records = classic_netcdf.extents(path)
    assert records == 1
    readable = {name for name in table if classic_netcdf.readable(table[name], records, ())}
    assert {"xtime", "scalar"}.isdisjoint(readable) and {"zgrid", "theta", "u", "areaCell"} <= readable
    monkeypatch.setenv("WOOF_HEX_CLASSIC_READ", "off")
    reference = _read_netcdf(path)
    monkeypatch.setenv("WOOF_HEX_CLASSIC_READ", "on")
    threaded = _read_netcdf(path)
    assert reference[0] == threaded[0] and reference[3] == threaded[3]
    assert list(reference[2]) == list(threaded[2])
    assert str(reference[1]) == str(threaded[1]) and str(reference[4]) == str(threaded[4])
    for name, value in reference[2].items():
        assert value.dtype == threaded[2][name].dtype and value.shape == threaded[2][name].shape, name
        assert value.tobytes() == threaded[2][name].tobytes(), name


def test_a_netcdf4_file_is_not_classic(tmp_path):
    path = tmp_path / "hdf5.nc"
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("x", 2)
        ds.createVariable("a", "f4", ("x",))[...] = 1
    assert classic_netcdf.extents(path) is None


@pytest.mark.parametrize("data_model", ["NETCDF3_CLASSIC", "NETCDF3_64BIT_DATA"])
def test_whole_reads_take_netcdf4s_arrays_in_blocks(tmp_path, monkeypatch, data_model):
    """THE COST THIS REMOVES: the dycore loaders read ~1 GB of each member's init on one thread (woof.hex.driver)."""
    path = tmp_path / "file.nc"
    _write(path, data_model)
    monkeypatch.setattr(classic_netcdf, "_BLOCK", 8)  # several blocks per variable, swapped in place each
    names = ["zgrid", "theta", "u", "areaCell", "cellsOnCell", "mask", "short", "xtime", "scalar", "scaled", "absent"]
    with Dataset(path) as ds, classic_netcdf.WholeReads(path, names, workers=4) as reads:
        for name in ("zgrid", "theta", "u", "areaCell", "cellsOnCell", "mask", "short"):
            variable = ds.variables[name]
            got = reads.take(name, variable.shape, variable.dtype)
            masked = np.asarray(variable[...])
            variable.set_auto_mask(False)
            plain = variable[...]
            for reference in (masked, plain):
                assert got.dtype == reference.dtype and got.shape == reference.shape, name
                assert got.tobytes() == reference.tobytes(), name
            assert got.flags.owndata and got.flags.c_contiguous, name
            assert reads.take(name, variable.shape, variable.dtype) is None  # taken once
        for name in ("xtime", "scalar", "scaled", "absent"):  # netCDF4 keeps these
            assert reads.take(name, (7,), np.float32) is None
    with classic_netcdf.WholeReads(path, ["zgrid"], workers=4) as reads:
        assert reads.take("zgrid", (7, 4), np.float32) is None  # metadata disagrees: netCDF4 reads it
    monkeypatch.setenv("WOOF_HEX_CLASSIC_READ", "off")
    with classic_netcdf.WholeReads(path, ["zgrid"], workers=4) as reads:
        assert reads.take("zgrid", (7, 3), np.float32) is None


def test_driver_logical_fields_match_netcdf4(tmp_path, monkeypatch):
    from woof.hex import driver

    path = tmp_path / "file.nc"
    _write(path, "NETCDF3_64BIT_DATA")
    monkeypatch.setattr(classic_netcdf, "_BLOCK", 8)
    with Dataset(path) as plain, Dataset(path) as opened, driver._whole_reads(opened, path, ["theta", "zgrid"]) as ds:
        for name in ("theta", "zgrid"):
            assert isinstance(ds.variables[name], driver._WholeVariable)
            expected = driver._logical_netcdf_field(plain.variables[name], level_dimension="nVertLevels",
                                                    entity_dimension="nCells", time_index=0)
            got = driver._logical_netcdf_field(ds.variables[name], level_dimension="nVertLevels",
                                               entity_dimension="nCells", time_index=0)
            assert got.dtype == expected.dtype and got.shape == expected.shape
            assert got.tobytes() == expected.tobytes(), name
        assert "u" in ds.variables and ds.variables["u"] is opened.variables["u"]
