"""The gridded observation products are WRITTEN on the project's Rust by default.

F3 of the 2026-08-18 boundary audit.  ``gpuwm-obs.radar-grid.v1`` and its
satellite twin ``gpuwm-obs.goes-grid.v1`` -- the two files the DA lanes read
and nothing upstream of them -- were both created with
``netCDF4.Dataset(temp, "w", format="NETCDF4_CLASSIC")`` on the bare default
of their doors, under the estate's "writing is not decoding" reading.  The law
says "NetCDF read/write", both words.

These tests pin the flip in the wrfout tape's shape:

* the DEFAULT writes the classic container through
  :mod:`woof.io.nc_writer_bridge`, and reaches netCDF4 for nothing but the
  two character variables the Rust decoder cannot hand back;
* the netCDF4 writer stays reachable ONLY by naming
  ``WOOF_OBS_GRID_WRITER=python``;
* the two engines write the same observation: same inventory, same dimension
  extents, same attributes, and every value bit identical;
* the DA read side consumes what the default wrote.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from woof.io import nc_writer_bridge
from woof.obs.goes_cwp import GriddedCwp
from woof.obs.goes_grid import read_goes_grid, write_goes_grid
from woof.obs.grid_product import (OBS_GRID_WRITER_ENV,
                                    ObsGridWriterUnavailable,
                                    open_obs_grid_product,
                                    resolve_obs_grid_engine)
from woof.obs.radar_grid import read_radar_grid, write_radar_grid
from woof.obs.superob import (SuperobParams, merge_contributions,
                               superob_volume)
from woof.obs.target_grid import TargetGrid

from test_obs_radar_grid import _grid as _analytic_grid, _volume as _analytic_volume

netCDF4 = pytest.importorskip("netCDF4")

NZ = 6


def _require_writer() -> None:
    reason = nc_writer_bridge.unavailable_reason()
    if reason is not None:
        pytest.skip(f"the Rust NetCDF writer is not built here: {reason}")


def _grid() -> TargetGrid:
    return _analytic_grid(nx=21, ny=21, nz=NZ)


def _volume(grid: TargetGrid, *, site_id: str = "KTLX"):
    return _analytic_volume(
        grid, site_id=site_id, azimuths=(90.0, 135.0),
        reflectivity=np.linspace(5.0, 55.0, 40, dtype=np.float32),
        velocity=np.linspace(-20.0, 20.0, 40, dtype=np.float32))


def _write_radar(path: Path, grid: TargetGrid, engine: str | None,
                 monkeypatch) -> dict:
    return _write_multi(path, grid, engine, monkeypatch, count=1)


def _cwp(grid: TargetGrid) -> GriddedCwp:
    shape = (grid.ny, grid.nx)
    mask = np.zeros(shape, dtype=np.int8)
    mask[2:6, 3:8] = 1
    observed = mask.astype(bool)
    values = np.zeros(shape, dtype=np.float64)
    values[observed] = np.linspace(0.0, 900.0, int(observed.sum()))
    classes = np.full(shape, -1, dtype=np.int8)
    classes[observed] = 2
    # A clear-sky cell is a genuine 0.0, which is the one case the writer
    # cross-checks between the class and the value.
    classes[2, 3] = 0
    values[2, 3] = 0.0
    errors = np.zeros(shape, dtype=np.float64)
    errors[observed] = 50.0
    levels = np.full(shape, -1, dtype=np.int32)
    levels[observed] = 2
    tops = np.full(shape, np.nan, dtype=np.float64)
    tops[observed] = 8000.0
    return GriddedCwp(
        cwp_obs=values, cwp_mask=mask, cwp_err=errors, cwp_class=classes,
        cwp_count=(mask * 4).astype(np.int32),
        cwp_pixels=(mask * 5).astype(np.int32),
        cloud_top_height_m=tops, obs_level=levels,
        counts={"observed": int(observed.sum())},
        provenance={"join": None, "error_model": {"note": "synthetic"}})


def _write_goes(path: Path, grid: TargetGrid, engine: str | None,
                monkeypatch) -> dict:
    if engine is None:
        monkeypatch.delenv(OBS_GRID_WRITER_ENV, raising=False)
    else:
        monkeypatch.setenv(OBS_GRID_WRITER_ENV, engine)
    return write_goes_grid(path, _cwp(grid), grid,
                           valid_time="2026-08-04T18:01:17Z", overwrite=True)


class _ForbiddenDataset:
    def __call__(self, *args, **kwargs):
        raise AssertionError(
            "the bare observation-product write opened netCDF4.Dataset"
            f"{args[1:2]}")


# --------------------------------------------------------------- defaults
def test_the_default_engine_is_rust_with_nothing_set(monkeypatch):
    monkeypatch.delenv(OBS_GRID_WRITER_ENV, raising=False)
    assert resolve_obs_grid_engine() == "rust"
    monkeypatch.setenv(OBS_GRID_WRITER_ENV, "python")
    assert resolve_obs_grid_engine() == "python"
    # An explicit argument still wins, so a caller can pin an engine.
    assert resolve_obs_grid_engine("rust") == "rust"


def test_an_unknown_engine_is_refused_by_name(monkeypatch):
    monkeypatch.setenv(OBS_GRID_WRITER_ENV, "hdf5")
    with pytest.raises(ValueError, match="unknown observation-product writer"):
        resolve_obs_grid_engine()


def test_the_bare_radar_default_writes_classic_without_netcdf4(
        tmp_path: Path, monkeypatch):
    """The positive claim: a classic container, and netCDF4 never opened.

    The concrete breakage: a product written by the C library on the door
    that is supposed to be Rust, with nothing in the file saying which.
    """

    _require_writer()
    grid = _grid()
    path = tmp_path / "radar-grid.nc"
    monkeypatch.setattr(netCDF4, "Dataset", _ForbiddenDataset())
    receipt = _write_radar(path, grid, None, monkeypatch)

    assert receipt["bytes"] == path.stat().st_size
    assert path.read_bytes()[:4] == b"CDF\x05"


def test_the_bare_satellite_default_writes_classic_without_netcdf4(
        tmp_path: Path, monkeypatch):
    _require_writer()
    grid = _grid()
    path = tmp_path / "goes-grid.nc"
    monkeypatch.setattr(netCDF4, "Dataset", _ForbiddenDataset())
    receipt = _write_goes(path, grid, None, monkeypatch)

    assert receipt["bytes"] == path.stat().st_size
    assert path.read_bytes()[:4] == b"CDF\x05"


def test_the_radar_read_reaches_netcdf4_only_for_the_character_variables(
        tmp_path: Path, monkeypatch):
    """Every observation array is decoded in Rust; the ids are not.

    ``netcrust`` exposes no character read at all, so the radar ids and
    per-volume timestamps are the one part of the file the bridge cannot
    hand back.  Everything else must not be quietly riding along with
    them.
    """

    _require_writer()
    grid = _grid()
    path = tmp_path / "radar-grid.nc"
    _write_radar(path, grid, None, monkeypatch)

    opened: list[tuple] = []
    real = netCDF4.Dataset

    def watched(*args, **kwargs):
        opened.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(netCDF4, "Dataset", watched)
    document = read_radar_grid(path, expected_grid=grid)

    assert len(opened) == 1, (
        f"the radar read opened netCDF4 {len(opened)} times; exactly one "
        f"open, for the two character variables, is the contract")
    assert document["radars"][0]["id"] == "KTLX"
    assert document["variables"]["z_obs"].shape == (grid.nz, grid.ny, grid.nx)


# ------------------------------------------------------------- dual run
def _inventory(path: Path) -> dict:
    with netCDF4.Dataset(path) as dataset:
        return {
            "variables": list(dataset.variables),
            "dimensions": {name: len(dimension)
                           for name, dimension in dataset.dimensions.items()},
            "attributes": {name: dataset.getncattr(name)
                           for name in dataset.ncattrs()},
        }


def _values(path: Path) -> dict[str, bytes]:
    with netCDF4.Dataset(path) as dataset:
        dataset.set_auto_mask(False)
        return {name: np.asarray(variable[:]).tobytes()
                for name, variable in dataset.variables.items()}


def _variable_attributes(path: Path) -> dict:
    with netCDF4.Dataset(path) as dataset:
        return {
            name: {key: (np.asarray(variable.getncattr(key)).dtype.str,
                         np.asarray(variable.getncattr(key)).tolist())
                   for key in variable.ncattrs()}
            for name, variable in dataset.variables.items()}


@pytest.mark.parametrize("product", ["radar", "goes"])
def test_the_two_engines_write_the_same_observation(tmp_path: Path,
                                                    monkeypatch, product):
    """Dual run: same input, two writers, every value bit identical.

    The container differs -- CDF-5 against HDF5 -- so the FILES are not
    byte-identical and could not be.  What must be identical is everything
    that describes the observation: which variables exist and in what
    order, how long each dimension is, every attribute (type as well as
    value), and every value bit.
    """

    _require_writer()
    grid = _grid()
    write = _write_radar if product == "radar" else _write_goes
    rust = tmp_path / f"{product}-rust.nc"
    python = tmp_path / f"{product}-python.nc"
    rust_receipt = write(rust, grid, None, monkeypatch)
    python_receipt = write(python, grid, "python", monkeypatch)

    assert rust.read_bytes()[:4] == b"CDF\x05"
    assert python.read_bytes()[:8] == b"\x89HDF\r\n\x1a\n"

    left, right = _inventory(rust), _inventory(python)
    assert left["variables"] == right["variables"]
    assert left["dimensions"] == right["dimensions"]
    assert set(left["attributes"]) == set(right["attributes"])
    for name in left["attributes"]:
        assert np.all(np.asarray(left["attributes"][name])
                      == np.asarray(right["attributes"][name])), name
        assert (np.asarray(left["attributes"][name]).dtype.str
                == np.asarray(right["attributes"][name]).dtype.str), name

    assert _variable_attributes(rust) == _variable_attributes(python)

    rust_values, python_values = _values(rust), _values(python)
    assert set(rust_values) == set(python_values)
    differing = [name for name in rust_values
                 if rust_values[name] != python_values[name]]
    assert not differing, f"value bits differ for {differing}"

    # The receipts describe the same observation set, and the digests are
    # over file bytes, so they are expected to differ -- the receipt is a
    # description of a FILE, not of the observation.
    assert (rust_receipt["grid_identity_sha256"]
            == python_receipt["grid_identity_sha256"])
    assert rust_receipt["sha256"] != python_receipt["sha256"]


def test_a_python_written_product_still_reads(tmp_path: Path, monkeypatch):
    """The workaround engine's output stays a first-class input.

    Every radar-grid file on disk before this flip is HDF5, and the read
    side must not have quietly become classic-only.
    """

    _require_writer()
    grid = _grid()
    path = tmp_path / "radar-grid.nc"
    _write_radar(path, grid, "python", monkeypatch)
    document = read_radar_grid(path, expected_grid=grid)
    assert document["radars"][0]["id"] == "KTLX"


# ------------------------------------------------------------- DA uptake
# -------------------------------------------------------------- refusals
def test_a_missing_rust_library_refuses_and_names_the_workaround(
        tmp_path: Path, monkeypatch):
    """No silent downgrade, and the refusal says what a fallback would cost."""

    monkeypatch.delenv(OBS_GRID_WRITER_ENV, raising=False)
    monkeypatch.setattr(nc_writer_bridge, "unavailable_reason",
                        lambda: "FileNotFoundError: not built here")
    with pytest.raises(ObsGridWriterUnavailable) as caught:
        open_obs_grid_product(tmp_path / "radar-grid.nc")
    message = str(caught.value)
    assert "no automatic fallback" in message
    assert "depends on which box" in message
    assert OBS_GRID_WRITER_ENV in message
    assert "cargo build --release --offline" in message


#: The NetCDF writer's checkout build as each shell must receive it,
#: written out rather than derived so a generator that loses its shell
#: rule cannot also rewrite what it is judged against.  Windows
#: PowerShell 5.1 rejects `&&` with a parser error.
NC_WRITER_BUILD_FOR_SHELL = {
    False: "cd tools/rustwx && cargo build --release --offline && cd ../..",
    True: "cd tools/rustwx; cargo build --release --offline; cd ../..",
}


@pytest.mark.parametrize("windows", (False, True))
def test_the_missing_writer_refusal_spells_the_build_for_the_shell(
        tmp_path: Path, monkeypatch, windows):
    """The build line in this refusal is one a Windows user can paste.

    It printed `cd tools/rustwx && cargo build ...` on every OS, which
    Windows PowerShell 5.1 cannot parse.
    """

    from woof import bridges

    monkeypatch.setattr(bridges, "WINDOWS_SHELL", windows)
    monkeypatch.delenv(OBS_GRID_WRITER_ENV, raising=False)
    monkeypatch.setattr(nc_writer_bridge, "unavailable_reason",
                        lambda: "FileNotFoundError: not built here")
    with pytest.raises(ObsGridWriterUnavailable) as caught:
        open_obs_grid_product(tmp_path / "radar-grid.nc")
    message = str(caught.value)
    assert NC_WRITER_BUILD_FOR_SHELL[windows] in message, message
    if windows:
        assert "&&" not in message, (
            f"Windows PowerShell 5.1 cannot parse '&&': {message}")


def test_a_partially_written_variable_is_refused(tmp_path: Path):
    """A hole cannot be written, so it cannot be read back as data.

    The classic container has no in-band "never filled" signal: an
    unwritten slab reads back as whatever the filesystem left there.
    """

    _require_writer()
    from woof.io.classic_product import ClassicProductError  # noqa: PLC0415

    path = tmp_path / "partial.nc"
    with pytest.raises(ClassicProductError, match="written whole"):
        with open_obs_grid_product(path) as dataset:
            dataset.createDimension("south_north", 4)
            dataset.createDimension("west_east", 4)
            variable = dataset.createVariable("HGT", "f4",
                                              ("south_north", "west_east"))
            variable[0:2] = np.zeros((2, 4), dtype=np.float32)
    assert not path.exists()


def test_a_declared_but_unassigned_variable_is_refused(tmp_path: Path):
    _require_writer()
    from woof.io.classic_product import ClassicProductError  # noqa: PLC0415

    path = tmp_path / "hole.nc"
    with pytest.raises(ClassicProductError, match="never assigned"):
        with open_obs_grid_product(path) as dataset:
            dataset.createDimension("south_north", 4)
            dataset.createVariable("HGT", "f4", ("south_north",))
    assert not path.exists()


def test_hdf5_storage_tuning_is_refused_rather_than_ignored(tmp_path: Path):
    """Accepting `zlib=True` would tell a caller the file is compressed."""

    _require_writer()
    from woof.io.classic_product import ClassicProductError  # noqa: PLC0415

    with pytest.raises(ClassicProductError, match="HDF5 storage tuning"):
        with open_obs_grid_product(tmp_path / "tuned.nc") as dataset:
            dataset.createDimension("south_north", 4)
            dataset.createVariable("HGT", "f4", ("south_north",), zlib=True)


def test_the_environment_is_not_read_behind_the_callers_back(monkeypatch):
    """An explicit engine argument is not overridden by the environment."""

    monkeypatch.setenv(OBS_GRID_WRITER_ENV, "python")
    assert resolve_obs_grid_engine("rust") == "rust"
    assert os.environ[OBS_GRID_WRITER_ENV] == "python"


# --------------------------------------- the metadata a legacy file carries
def _multi_radar(grid: TargetGrid, count: int):
    """A product from ``count`` radars -- the ordinary DA case, not one site.

    Two radars is enough to push ``provenance`` past the point where an HDF5
    object header spills its attributes into a continuation block, which is
    the condition the reader used to lose them under.
    """

    params = SuperobParams()
    sites = ["KTLX"] if count == 1 else [f"K{chr(65 + i)}{chr(65 + i)}X"
                                        for i in range(count)]
    contributions = [superob_volume(_volume(grid, site_id=site), grid, params=params)
                     for site in sites]
    return merge_contributions(contributions, grid, params=params,
                               z_reduce="max"), params


def _write_multi(path: Path, grid: TargetGrid, engine: str | None,
                 monkeypatch, count: int = 2) -> dict:
    if engine is None:
        monkeypatch.delenv(OBS_GRID_WRITER_ENV, raising=False)
    else:
        monkeypatch.setenv(OBS_GRID_WRITER_ENV, engine)
    observations, params = _multi_radar(grid, count)
    return write_radar_grid(path, observations, grid,
                            valid_time="2026-07-28T20:03:16Z", params=params,
                            overwrite=True)


@pytest.mark.parametrize("engine", [None, "python"])
def test_a_two_radar_product_keeps_its_provenance(tmp_path: Path, monkeypatch,
                                                  engine):
    """The assimilation record survives the round trip on BOTH containers.

    Found by running the real `woof obs radar grid` door on a real KFTG
    Level-II volume: the file written by the netCDF4 workaround engine read
    back with an EMPTY ``provenance``, so the DA side recorded
    ``dealiasing: None`` for velocities that had in fact been masked on four
    risk signatures.  Every radar-grid file written before this release is
    HDF5, so this is the legacy read path, and it failed silently -- the
    reader defaulted a missing attribute to ``{}`` instead of refusing.
    """

    if engine is None:
        _require_writer()
    grid = _grid()
    path = tmp_path / "two-radar.nc"
    receipt = _write_multi(path, grid, engine, monkeypatch)
    assert len(receipt["radars"]) == 2

    document = read_radar_grid(path, expected_grid=grid)
    with netCDF4.Dataset(path) as dataset:
        written = dataset.getncattr("provenance")
    assert len(written) > 4000, (
        "the fixture no longer reaches the attribute spill it exists to "
        f"cover ({len(written)} bytes)")
    assert document["provenance"], (
        "the reader handed back an empty provenance for a file that carries "
        f"{len(written)} bytes of it")
    assert document["provenance"]["dealiasing"], (
        "the dealiasing statement is what the DA manifest records about "
        "these velocities; it cannot come back None")
    assert document["superob_params"]
    assert document["clear_air_source"] is not None


def test_both_engines_read_back_the_same_metadata(tmp_path: Path, monkeypatch):
    """Which library wrote the file cannot change what the reader reports."""

    _require_writer()
    grid = _grid()
    rust = tmp_path / "rust.nc"
    python = tmp_path / "python.nc"
    _write_multi(rust, grid, None, monkeypatch)
    _write_multi(python, grid, "python", monkeypatch)

    left = read_radar_grid(rust, expected_grid=grid)
    right = read_radar_grid(python, expected_grid=grid)
    for key in ("schema", "status", "valid_time", "grid_identity_sha256",
                "clear_air_source", "superob_params", "provenance"):
        assert left[key] == right[key], key


def test_the_reader_reports_every_attribute_the_file_carries(
        tmp_path: Path, monkeypatch):
    """No metadata key may be quietly defaulted away.

    The Rust bridge cannot see an HDF5 attribute that landed in an object
    header continuation block, and returns ``None`` for it rather than
    failing.  A reader that reaches for metadata with a default therefore
    turns "this build cannot read the attribute" into "the file does not
    have one", which is the quiet kind of wrong this schema exists to
    prevent.
    """

    grid = _grid()
    path = tmp_path / "legacy.nc"
    _write_multi(path, grid, "python", monkeypatch)
    document = read_radar_grid(path, expected_grid=grid)

    import json as _json

    with netCDF4.Dataset(path) as dataset:
        for name, value in (("schema", None), ("status", None),
                            ("valid_time", None),
                            ("grid_identity_sha256", None),
                            ("clear_air_source", None)):
            assert document[name] == dataset.getncattr(name), name
        assert document["provenance"] == _json.loads(
            dataset.getncattr("provenance"))
        assert document["superob_params"] == _json.loads(
            dataset.getncattr("superob_params"))
