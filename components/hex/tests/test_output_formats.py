"""The hex doors share the engine's native output table and writer."""
import json
from pathlib import Path
import subprocess

import pytest
from woof import output_formats
from woof.hex.cli import build_parser


def test_table_is_the_installed_engine_table():
    row, = output_formats.selected('icechunk')
    assert row['writer'] == 'rw_icechunk'
    assert row['chunks']['time'] == 1


def test_all_forecast_parsers_advertise_output_formats():
    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == 'command').choices
    doors = [sub[n] for n in ('forecast', 'global', 'ensemble', 'moving-ensemble', 'add-nest', 'pair', 'country', 'mesh-plan', 'global-cycle', 's2s', 'regional')]
    moving = next(a for a in sub['moving']._actions if hasattr(a, 'choices') and isinstance(a.choices, dict)).choices
    doors.extend(moving[n] for n in ('run', 'fixed'))
    for door in doors:
        assert '--output-format' in door.format_help()
        assert '--icechunk-store' in door.format_help()
        assert '--output-config' in door.format_help()


def test_missing_writer_refuses_hex_forecast(tmp_path, monkeypatch):
    from argparse import Namespace
    from woof.hex.forecast_door import run_forecast
    monkeypatch.setattr(output_formats, 'find_binary', lambda row: None)
    args = Namespace(out=tmp_path/'out', output_format='icechunk')
    with pytest.raises(ValueError, match='without its requested store'):
        run_forecast(args)
    assert not args.out.exists()


def test_native_cell_history_roundtrip(tmp_path):
    nc = pytest.importorskip('netCDF4')
    np = pytest.importorskip('numpy')
    icechunk = pytest.importorskip('icechunk')
    xr = pytest.importorskip('xarray')
    row, = output_formats.selected('icechunk')
    binary = output_formats.find_binary(row)
    if binary is None:
        pytest.skip('build or install rw_icechunk')
    source = tmp_path/'history.2026-10-28_00.00.00.nc'
    with nc.Dataset(source, 'w', format='NETCDF3_64BIT_OFFSET') as f:
        for name, size in [('Time', None), ('StrLen', 19), ('nCells', 7), ('nVertLevels', 3)]:
            f.createDimension(name, size)
        f.createVariable('xtime', 'S1', ('Time', 'StrLen'))[:] = np.array([
            list('2026-10-28_00:00:00'), list('2026-10-28_01:00:00')], dtype='S1')
        v = f.createVariable('theta', 'f8', ('Time', 'nCells', 'nVertLevels'))
        v[:] = np.arange(42, dtype='f8').reshape(2, 7, 3) + 280
        v.units = 'K'
        f.createVariable('latCell', 'f8', ('nCells',))[:] = np.arange(7)*0.1
        f.createVariable('lonCell', 'f8', ('nCells',))[:] = np.arange(7)*0.2
    store = tmp_path/'store'
    request = tmp_path/'request.json'
    request.write_text(json.dumps(dict(schema=row['request_schema'], mode='run', store=str(store),
                                      inputs=[str(source)], chunks=row['chunks'])))
    proc = subprocess.run([str(binary), '--request', str(request)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    events = list(map(json.loads, proc.stdout.splitlines()))
    assert [e['event'] for e in events] == ['committed', 'committed', 'finalized']
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store)))
    with xr.open_zarr(repo.readonly_session('main').store, consolidated=False, mask_and_scale=False) as ds:
        np.testing.assert_array_equal(ds.theta, np.arange(42, dtype='f8').reshape(2, 7, 3)+280)
        assert ds.theta.dtype == np.dtype('float64')
        for t in range(2):
            np.testing.assert_array_equal(ds.latCell.isel(time=t), np.arange(7)*0.1)
        assert ds.sizes['time'] == 2
    with xr.open_zarr(repo.readonly_session(snapshot_id=events[0]['snapshot']).store,
                     consolidated=False) as initial:
        assert initial.sizes['time'] == 1

def test_real_history_excerpt(tmp_path):
    nc = pytest.importorskip('netCDF4')
    np = pytest.importorskip('numpy')
    icechunk = pytest.importorskip('icechunk')
    xr = pytest.importorskip('xarray')
    row, = output_formats.selected('icechunk')
    binary = output_formats.find_binary(row)
    if binary is None:
        pytest.skip('build rw_icechunk or install a platform wheel')
    source = Path(__file__).parent/'fixtures/icechunk/native-history.nc'
    store = tmp_path/'store'
    request = tmp_path/'request.json'
    request.write_text(json.dumps(dict(schema=row['request_schema'], mode='run',
        inputs=[str(source)], store=str(store), chunks=row['chunks'])))
    result = subprocess.run([str(binary), '--request', str(request)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store)))
    session = repo.readonly_session('main')
    with xr.open_zarr(session.store, consolidated=False, mask_and_scale=False, decode_times=False) as ds, nc.Dataset(source) as native:
        native.set_auto_maskandscale(False)
        for name, variable in native.variables.items():
            expected = variable[:]
            if 'Time' in variable.dimensions:
                expected = np.take(expected, 0, axis=variable.dimensions.index('Time'))
            if expected.dtype.kind == 'S':
                expected = expected.view('u1')
            np.testing.assert_array_equal(ds[name].isel(time=0).values, expected)
        assert ds.sizes['time'] == 1
