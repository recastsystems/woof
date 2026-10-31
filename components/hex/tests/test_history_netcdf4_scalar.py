"""A lossless NetCDF-4 history frame holding a scalar variable opens in netCDF-C (rw_mpas_stream).

THE BREAKAGE THIS PREVENTS (hex 0.4.10 candidate, a hex regional run on 2026-10-29): every
MPAS-route history.*.nc of a hex regional run was refused by netCDF-C 4.9.3 (ncdump) and netCDF4-python with
"NetCDF: HDF error", so xarray, ncview and NCL could not read a history frame.  rw_mpas_stream's NetCDF-4 writer put an
empty _Netcdf4Coordinates attribute on the scalar variables of MPAS history (isice_lu, iswater_lu); netCDF-C never
writes one on a scalar and cannot read it back.  Fixed in the engine's netcdf-writer crate (woof dd8c2e697c) and
reaches hex through the engine-bridges.json re-pin of rw_mpas_stream.

The frame goes through the stream bridge the hex ladder resolves (the wheel's libexec, WOOF_HEX_RW_MPAS_STREAM, ...),
written the way woof.hex.restart_v842.write_stream writes a history file (compression member from
woof.hex.history_compression).  Skips where no stream bridge is installed or neither netCDF4 nor h5py is importable.
"""
import numpy as np
import pytest

from woof.hex import history_compression, restart_v842
from woof.hex.engines import EngineRefusal

ISICE_LU = 15
TEMPERATURE = np.array([288.5, 290.25, 301.0], dtype=np.float32)


def _bridge_or_skip():
    try:
        return restart_v842._bridge()
    except EngineRefusal:
        pytest.skip("no rw_mpas_stream installed (wheel libexec, WOOF_HEX_RW_MPAS_STREAM or woof's bridges)")


def _write_history_frame(tmp_path):
    raw_scalar = tmp_path / "isice_lu.raw"
    np.asarray(ISICE_LU, dtype=np.int32).tofile(raw_scalar)
    raw_t2m = tmp_path / "t2m.raw"
    TEMPERATURE.tofile(raw_t2m)
    compression = history_compression.for_stream(
        "output", {"output_interval": "01:00:00", "filename_template": "history.$Y-$M-$D_$h.$m.$s.nc"},
        append=False, record_index=0)
    assert compression is not None, "history frames are written lossless NetCDF-4 by default"
    target = tmp_path / "history.2026-10-28_12.00.00.nc"
    scalar_attrs = {"units": "unitless", "long_name": "Index of ice in land-use table"}
    t2m_attrs = {"units": "K", "long_name": "2-meter temperature"}
    request = {
        "operation": "write", "path": str(target), "format": "netcdf",
        "dimensions": {"Time": 1, "StrLen": 64, "nCells": TEMPERATURE.size},
        "dimension_order": ["Time", "StrLen", "nCells"],
        "variables": [
            {"name": "xtime", "dtype": "S1", "dimensions": ("Time", "StrLen"), "attrs": {"units": "unitless"},
             "attribute_order": ["units"], "text": ["2026-10-28_12:00:00"]},
            # isice_lu: an MPAS Registry scalar (dimensions ""), the variable netCDF-C refused.
            {"name": "isice_lu", "dtype": "i4", "dimensions": (), "attrs": scalar_attrs,
             "attribute_order": list(scalar_attrs), "filename": str(raw_scalar), "shape": (),
             "input_dimensions": (), "input_dtype": "i4", "connectivity_offset": 0},
            {"name": "t2m", "dtype": "f4", "dimensions": ("Time", "nCells"), "attrs": t2m_attrs,
             "attribute_order": list(t2m_attrs), "filename": str(raw_t2m), "shape": TEMPERATURE.shape,
             "input_dimensions": ("nCells",), "input_dtype": "f4", "connectivity_offset": 0},
        ],
        "global_attrs": {"model_name": "mpas"}, "global_attribute_order": ["model_name"],
        "header_alignment": 0, "append": False, "record_index": 0, "compression": compression,
    }
    restart_v842._call(request, tmp_path)
    return target


def test_a_lossless_history_frame_with_a_scalar_opens_in_netcdf_c(tmp_path):
    _bridge_or_skip()
    target = _write_history_frame(tmp_path)
    try:
        import netCDF4
    except ImportError:
        netCDF4 = None
    if netCDF4 is not None:
        # netCDF-C itself: the old writer's file stops here with "NetCDF: HDF error".
        with netCDF4.Dataset(target) as frame:
            assert frame.data_model == "NETCDF4_CLASSIC"
            assert frame.variables["isice_lu"].dimensions == ()
            assert int(frame.variables["isice_lu"][...]) == ISICE_LU
            np.testing.assert_array_equal(frame.variables["t2m"][0, :], TEMPERATURE)
            assert frame.variables["t2m"].units == "K"
        return
    h5py = pytest.importorskip("h5py", reason="neither netCDF4 nor h5py is installed")
    with h5py.File(target, "r") as frame:
        scalars = [name for name, item in frame.items() if isinstance(item, h5py.Dataset) and item.shape == ()]
        assert "isice_lu" in scalars
        carrying = [name for name in scalars if "_Netcdf4Coordinates" in frame[name].attrs]
        assert not carrying, f"scalar variables carry _Netcdf4Coordinates, which netCDF-C cannot open: {carrying}"
        assert int(frame["isice_lu"][()]) == ISICE_LU
