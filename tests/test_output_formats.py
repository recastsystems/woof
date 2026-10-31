"""Native output snapshots cannot expose partial frames or duplicate resumed times."""
from argparse import Namespace
import json
from pathlib import Path
import subprocess

import pytest

from woof import output_formats as formats


def test_table_default_and_bridge():
    from woof.bridge_assets import BUNDLED_ARTIFACTS
    from woof.bridges import BRIDGE_ABI_MARKERS
    assert [r['id'] for r in formats.selected()] == ['netcdf']
    row, = formats.selected('icechunk')
    assert row['chunks']['time'] == 1
    assert any(a.name == row['writer'] for a in BUNDLED_ARTIFACTS)
    assert BRIDGE_ABI_MARKERS[row['writer']] == formats.ABI_MARKER.encode()


@pytest.mark.parametrize('io_launcher', [True, False])
def test_writer_spawn_has_background_priority_and_bounded_pools(tmp_path, monkeypatch, io_launcher):
    row, = formats.selected('icechunk')
    monkeypatch.setattr(formats, 'require_binary', lambda _: Path('/writer'))
    monkeypatch.setattr(formats.os, 'process_cpu_count', lambda: 24, raising=False)
    monkeypatch.setattr(formats.os, 'sched_getaffinity', lambda _: set(range(24)), raising=False)
    monkeypatch.setattr(formats.shutil, 'which', lambda name:
                        '/bin/'+name if name == 'nice' or io_launcher else None)
    monkeypatch.setenv('RAYON_NUM_THREADS', '48')
    monkeypatch.delenv('TOKIO_WORKER_THREADS', raising=False)
    launches = []
    def run(command, **kwargs):
        launches.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(formats.subprocess, 'run', run)
    worker = formats.HistoryOutputs(tmp_path, [row])
    worker._invoke(row, Path('/writer'), '/store', [], 'finalize')
    command, kwargs = launches[0]
    if formats.os.name == 'nt':
        assert kwargs['creationflags'] == subprocess.BELOW_NORMAL_PRIORITY_CLASS
        # The writer path is str(Path('/writer')), '\writer' on Windows (public ci cpu (windows-2025), 2.9.10 P
        # f2d381407731, 2026-10-30).
        assert command[:2] == [str(Path('/writer')), '--request']
    else:
        prefix = ['/bin/ionice', '-c', '2', '-n', '7'] if io_launcher else []
        assert command[:-1] == prefix + ['/bin/nice', '-n', '10', '/writer', '--request']
    assert kwargs['env']['RAYON_NUM_THREADS'] == '12'
    assert kwargs['env']['TOKIO_WORKER_THREADS'] == '12'
    assert 'preexec_fn' not in kwargs


def test_writer_pool_respects_affinity_and_smaller_inherited_limit(monkeypatch):
    row, = formats.selected('icechunk')
    monkeypatch.setattr(formats.os, 'process_cpu_count', lambda: 24, raising=False)
    monkeypatch.setattr(formats.os, 'sched_getaffinity', lambda _: {0, 1, 2, 3}, raising=False)
    monkeypatch.setattr(formats.shutil, 'which', lambda name: '/bin/'+name)
    monkeypatch.setenv('RAYON_NUM_THREADS', '1')
    monkeypatch.setenv('TOKIO_WORKER_THREADS', '0')
    _, kwargs = formats.writer_launch(row, ['/writer'])
    assert kwargs['env']['RAYON_NUM_THREADS'] == '1'
    assert kwargs['env']['TOKIO_WORKER_THREADS'] == '2'


def test_missing_writer_refuses_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(formats, 'find_binary', lambda row: None)
    args = Namespace(command='run', outdir=tmp_path/'absent', output_format='icechunk')
    with pytest.raises(ValueError, match='run would finish without its requested store'):
        with formats.command_scope(args):
            pytest.fail('model must not start')
    assert not args.outdir.exists()


def test_config_and_cli_precedence(tmp_path):
    path = tmp_path/'output.toml'
    path.write_text('[output]\nformats = ["icechunk"]\nicechunk_store = "s3://bucket/prefix"\n')
    args = Namespace(output_config=path, output_format=None, icechunk_store=None)
    rows, store = formats.options(args)
    assert rows[0]['id'] == 'icechunk' and store == 's3://bucket/prefix'
    args.icechunk_store = '/local'
    assert formats.options(args)[1] == '/local'


def test_publication_scan_ignores_retired_column_spool(tmp_path):
    retained = tmp_path/'members/member-0000/wrfout'
    retired = tmp_path/'columns/.frames/member-0000'
    name = 'wrfout_d01_2026-10-28_00_00_00'
    for folder in (retained, retired):
        folder.mkdir(parents=True)
        (folder/'.history-written').write_text(name+'\n')
    (retained/name).touch()
    worker = formats.HistoryOutputs(tmp_path, [])
    assert worker._published() == [retained/name]


def write_history(path, hour, *, cells=False):
    nc = pytest.importorskip('netCDF4')
    np = pytest.importorskip('numpy')
    with nc.Dataset(path, 'w', format='NETCDF3_64BIT_OFFSET') as f:
        dimensions = [('Time', None), ('StrLen', 19), ('nCells', 7), ('nVertLevels', 3)] if cells else [
            ('Time', None), ('StrLen', 19), ('bottom_top', 3), ('south_north', 5), ('west_east', 7)]
        for name, size in dimensions:
            f.createDimension(name, size)
        name = 'xtime' if cells else 'Times'
        f.createVariable(name, 'S1', ('Time', 'StrLen'))[:] = np.array(
            [list(f'2026-10-28_{hour:02d}:00:00')], dtype='S1')
        dims = ('Time', 'nCells', 'nVertLevels') if cells else ('Time', 'bottom_top', 'south_north', 'west_east')
        shape = (1, 7, 3) if cells else (1, 3, 5, 7)
        v = f.createVariable('temperature' if cells else 'T', 'f4', dims)
        v[:] = np.arange(np.prod(shape), dtype='f4').reshape(shape) + hour
        v.units = 'K'
        v.description = 'native temperature field'
        cdims = ('nCells',) if cells else ('Time', 'south_north', 'west_east')
        for name in (('latCell', 'lonCell') if cells else ('XLAT', 'XLONG')):
            f.createVariable(name, 'f4', cdims)[:] = hour + 0.25
        f.frame_number = hour


def exercise_writer(tmp_path, *, cells=False):
    np = pytest.importorskip('numpy')
    icechunk = pytest.importorskip('icechunk')
    xr = pytest.importorskip('xarray')
    row, = formats.selected('icechunk')
    binary = formats.find_binary(row)
    if binary is None:
        pytest.skip('build rw_icechunk or install a platform wheel')
    files = [tmp_path/f'frame-{i}.nc' for i in range(2)]
    for i, path in enumerate(files):
        write_history(path, i, cells=cells)
    store = tmp_path/'store'
    def invoke(inputs, mode='append'):
        path = tmp_path/'request.json'
        path.write_text(json.dumps(dict(schema=row['request_schema'], mode=mode,
                                       inputs=list(map(str, inputs)), store=str(store), chunks=row['chunks'])))
        return subprocess.run([str(binary), '--request', str(path)], capture_output=True, text=True)
    first = invoke(files[:1])
    assert first.returncode == 0, first.stderr
    snapshot = json.loads(first.stdout)['snapshot']
    second = invoke(files, 'run')
    assert second.returncode == 0, second.stderr
    events = list(map(json.loads, second.stdout.splitlines()))
    assert [e['event'] for e in events] == ['kept', 'committed', 'finalized']
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store)))
    ds = xr.open_zarr(repo.readonly_session('main').store, consolidated=False, mask_and_scale=False)
    assert ds.sizes['time'] == 2
    with xr.open_zarr(repo.readonly_session(snapshot_id=snapshot).store,
                      consolidated=False, mask_and_scale=False) as initial:
        assert initial.sizes['time'] == 1
    nc = pytest.importorskip('netCDF4')
    for i, path in enumerate(files):
        with nc.Dataset(path) as source:
            source.set_auto_mask(False)
            for name, variable in source.variables.items():
                expected = variable[:]
                if variable.dimensions[0] == 'Time':
                    expected = expected[0]
                if expected.dtype.kind == 'S':
                    expected = expected.view('u1')
                np.testing.assert_array_equal(ds[name].isel(time=i).values, expected)
    tagged = xr.open_zarr(repo.readonly_session(tag=events[-1]['tag']).store,
                          consolidated=False, mask_and_scale=False)
    assert tagged.sizes['time'] == 2
    ds.close(); tagged.close()
    # A repeated time with changed values must not move the committed branch.
    with nc.Dataset(files[0], 'a') as source:
        source.variables['temperature' if cells else 'T'][:] += 1
    refused = invoke(files[:1])
    assert refused.returncode != 0 and 'different values or metadata' in refused.stderr
    assert repo.lookup_branch('main') == events[-1]['snapshot']


def test_native_roundtrip_atomic_resume(tmp_path):
    exercise_writer(tmp_path)


def test_failed_model_does_not_finalize(tmp_path, monkeypatch):
    row, = formats.selected('icechunk')
    monkeypatch.setattr(formats, 'require_binary', lambda _: Path('/unused'))
    scope = formats.HistoryOutputs(tmp_path, [row])
    calls = []
    monkeypatch.setattr(scope, '_scan', lambda: None)
    monkeypatch.setattr(scope, '_invoke', lambda *args: calls.append(args))
    scope.repositories.add((0, 'repo'))
    with scope:
        scope.finish(2)
    assert not calls
    assert json.loads((tmp_path/'output-formats/receipt.json').read_text())['status'] == 'failed'


def test_explicit_store_separates_member_and_segment(tmp_path):
    config = tmp_path/'output.toml'
    config.write_text('[output]\nformats=["icechunk"]\nicechunk_store="s3://bucket/base"\n')
    args = Namespace(output_config=config)
    forwarded = formats.argument_values(args, relative='member-001/segment-002')
    assert forwarded[-2:] == ['--icechunk-store', 's3://bucket/base/member-001/segment-002']


def test_cuda_history_waits_for_durable_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(formats, 'require_binary', lambda _: Path('/unused'))
    row, = formats.selected('icechunk')
    scope = formats.HistoryOutputs(tmp_path, [row], engine='woof hex')
    source = tmp_path/'cuda-history.2026-10-28_00.00.00.nc'
    source.write_bytes(b'landed')
    assert scope._published() == []
    stat = source.stat()
    source.with_name(source.name+'.durable.json').write_text(json.dumps({
        'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}))
    assert scope._published() == [source]


def test_single_scope_does_not_enable_ensemble_retention(tmp_path, monkeypatch):
    monkeypatch.setattr(formats, 'require_binary', lambda _: Path('/unused'))
    monkeypatch.setattr(formats.HistoryOutputs, '__enter__', lambda self: self)
    monkeypatch.setattr(formats.HistoryOutputs, '__exit__', lambda *args: False)
    args = Namespace(command='go', outdir=tmp_path/'run', output_format='icechunk')
    with formats.command_scope(args):
        assert not getattr(args, 'keep_member_files', False)
    config = tmp_path/'ensemble.toml'
    config.write_text('[ensemble]\nmembers=2\n')
    args.config = config
    with formats.command_scope(args):
        assert args.keep_member_files


def test_explicit_store_omits_history_container(tmp_path, monkeypatch):
    monkeypatch.setattr(formats, 'require_binary', lambda _: Path('/unused'))
    row, = formats.selected('icechunk')
    scope = formats.HistoryOutputs(tmp_path, [row], 's3://bucket/base')
    assert scope._store_for(row, tmp_path/'run/wrfout/wrfout_d01_time') == 's3://bucket/base/d01'
    assert scope._store_for(row, tmp_path/'chain/run/wrfout/wrfout_d01_time') == 's3://bucket/base/d01'


def test_restart_inherits_existing_local_store(tmp_path, monkeypatch):
    monkeypatch.setattr(formats, 'require_binary', lambda _: Path('/unused'))
    monkeypatch.setattr(formats.HistoryOutputs, '__enter__', lambda self: self)
    monkeypatch.setattr(formats.HistoryOutputs, '__exit__', lambda *args: False)
    previous = tmp_path/'original/run/wrfout/icechunk'
    previous.mkdir(parents=True)
    checkpoint = previous.parent.parent/'checkpoint.npz'
    args = Namespace(command='go', outdir=tmp_path/'continued', restart=checkpoint,
                     output_format='icechunk')
    with formats.command_scope(args) as scope:
        assert scope.store == str(previous)


def test_real_history_excerpt(tmp_path):
    nc = pytest.importorskip('netCDF4')
    np = pytest.importorskip('numpy')
    icechunk = pytest.importorskip('icechunk')
    xr = pytest.importorskip('xarray')
    row, = formats.selected('icechunk')
    binary = formats.find_binary(row)
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
