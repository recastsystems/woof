"""Exercise the real offline fetch/extract path with tiny sealed ZIP assets."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
import zipfile

import pytest

from woof.hex.mpasphys import resources_build as rb
from woof.hex.mpasphys import resources_fetch as rf


@pytest.fixture(autouse=True)
def _no_site_resource_source(monkeypatch):
    # A site that stages DATA offline exports WOOF_HEX_PHYSICS_RESOURCES_FROM/_URL; the fixtures here
    # build their own sources, so an inherited value redirected the online test to the site mirror
    # (G0 on 0702c603: test_online_upstream_uses_packaged_url_in_parallel_with_release read
    # <mirror>/resources.zip). Tests that need a source set it themselves.
    monkeypatch.delenv(rf.FROM_ENV, raising=False)
    monkeypatch.delenv(rf.URL_ENV, raising=False)


def sealed(tmp_path, monkeypatch, *, member='tables/fixture.bin', mode=stat.S_IFREG | 0o644):
    source=tmp_path/'assets';source.mkdir()
    data=b'fixture, not a native binary\n'
    name='resources.zip'
    info=zipfile.ZipInfo(member);info.external_attr=mode << 16
    with zipfile.ZipFile(source/name,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(info,data)
    row={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'executable':False}
    pins={'schema':rf.SCHEMA,'release_assets':{},'assets':[
        {'name':name,'pin_sets':list(rb.PIN_SETS),'files':{member:row},
         'sha256':rb.sha256_file(source/name),'bytes':(source/name).stat().st_size}],
         'resources':{'thompson-tables':{'path':'tables','sha256':row['sha256'],'pin_sets':list(rb.PIN_SETS)}}}
    monkeypatch.setattr(rf,'load_manifest',lambda:(pins,'a'*64))
    def assemble(directory, **kwargs):
        path=directory/member
        return {'pool-reader':{'path':str(path),'sha256':row['sha256']}}
    monkeypatch.setattr(rb,'assemble_framework_resources',assemble)
    return source,pins


def test_offline_fetch_verifies_installs_and_relocates(tmp_path, monkeypatch):
    source,pins=sealed(tmp_path,monkeypatch)
    cache=tmp_path/'cache'
    result=rf.fetch(cache,from_directory=source,jobs=4)
    assert result['source_build'] is False and result['workers']==4
    assert not set(result['resources']) & {'native-library','native-library-rrfs','mpas-source'}
    assert 'parallel_extract_verify_seconds' in result['stages']
    receipt,=list((cache/'receipts').glob('fetch-*.json'))
    assert json.loads(receipt.read_text())==result
    path=rb.resolve_resource('pool-reader',cache)
    assert path.read_bytes()==b'fixture, not a native binary\n'
    assert not path.stat().st_mode & 0o111
    assert not list((cache/'downloads').rglob('*.zip'))
    moved=tmp_path/'moved';cache.rename(moved)
    assert rb.resolve_resource('pool-reader',moved).read_bytes()==b'fixture, not a native binary\n'
    assert 'verify_seconds' in rf.fetch(moved,from_directory=source)['stages']
    rb.resolve_resource('pool-reader',moved).write_bytes(b'corrupt')
    with pytest.raises(rb.PhysicsResourceError,match='differs from pinned'):
        rf.fetch(moved,from_directory=source)


def test_wrong_offline_bytes_never_publish(tmp_path,monkeypatch):
    source,_=sealed(tmp_path,monkeypatch)
    (source/'resources.zip').write_bytes(b'wrong bytes')
    with pytest.raises(rb.PhysicsResourceError,match='differs from pinned'):
        rf.fetch(tmp_path/'cache',from_directory=source)
    assert not (tmp_path/'cache/installed.json').exists()


def test_missing_packaged_manifest_refuses_without_source_build(tmp_path,monkeypatch):
    monkeypatch.setattr(rf,'MANIFEST',tmp_path/'not-packaged.json')
    with pytest.raises(rb.PhysicsResourceError,match='no sealed physics release manifest'):
        rf.load_manifest()


@pytest.mark.parametrize('jobs',[0,1,65])
def test_invalid_worker_count_refuses_before_copy(tmp_path,monkeypatch,jobs):
    source,_=sealed(tmp_path,monkeypatch)
    with pytest.raises(rb.PhysicsResourceError,match='between 2 and 64'):
        rf.fetch(tmp_path/'cache',from_directory=source,jobs=jobs)
    assert not (tmp_path/'cache').exists()


@pytest.mark.parametrize('member,mode',[('../escape.bin',stat.S_IFREG),('/absolute.bin',stat.S_IFREG),
                                      ('link.bin',stat.S_IFLNK),('pipe.bin',stat.S_IFIFO)])
def test_unsafe_members_never_publish(tmp_path,monkeypatch,member,mode):
    source,_=sealed(tmp_path,monkeypatch,member=member,mode=mode)
    with pytest.raises(rb.PhysicsResourceError,match='unsafe|non-regular'):
        rf.fetch(tmp_path/'cache',from_directory=source)
    assert not (tmp_path/'cache/installed.json').exists()
    assert not list((tmp_path/'cache/releases').rglob('.partial-*'))


def test_pool_reader_comes_from_wheel_without_fetch(tmp_path,monkeypatch):
    path=tmp_path/'reader';path.write_bytes(b'wheel reader')
    monkeypatch.setattr(rb,'stage_pool_reader',lambda *a: {'path':str(path),'sha256':rb.sha256_file(path)})
    def forbidden(*args,**kwargs):
        pytest.fail('pool reader requested release DATA or source build')
    monkeypatch.setattr(rf,'fetch',forbidden)
    assert rb.ensure_route_resources(('pool-reader',),cache=tmp_path/'cache') == {'pool-reader':path}


def test_doctor_reports_unstaged_resources_without_writes_or_fetch(tmp_path, monkeypatch):
    from woof.hex import doctor
    monkeypatch.setenv('WOOF_HEX_CACHE_ROOT', str(tmp_path/'empty'))
    monkeypatch.setattr(rf, 'fetch', lambda *a, **kw: pytest.fail('doctor fetched DATA'))
    rows = doctor.check_physics_resources()
    assert len(rows) == 3
    assert all(row.status == doctor.INFO for row in rows)
    assert all(row.remedy == 'woof hex physics-resources fetch' for row in rows)
    assert not doctor.blocking_gaps(rows)
    assert not (tmp_path/'empty').exists()
    # Check the real collector calls this probe, without unrelated import/device probes.
    for name in ('check_interpreter', 'check_python_dependencies', 'check_physics_seam',
                 'check_gpu_runtime', 'check_gpu_architecture', 'check_engines', 'check_assets',
                 'check_on_demand_assets'):
        monkeypatch.setattr(doctor, name, lambda: [])
    assert [row.subject for row in doctor.collect()] == [row.subject for row in rows]


@pytest.mark.parametrize('tamper', [None, 'thompson-tables', 'physics-resources', 'pool-reader'])
def test_doctor_verifies_staged_resources_and_names_corruption(tmp_path, monkeypatch, tamper):
    from woof.hex import doctor
    source, _ = sealed(tmp_path, monkeypatch)
    monkeypatch.setenv('WOOF_HEX_CACHE_ROOT', str(tmp_path/'root'))
    cache = rb.default_cache()
    rf.fetch(cache, from_directory=source, jobs=2)
    packet = cache/'packet.json'; packet.write_text('{}')
    reader = cache/'reader'; reader.write_bytes(b'pool reader fixture')
    rb.record_installed(cache, {key: {'path': str(path), 'sha256': rb.sha256_file(path)}
                               for key, path in [('physics-resources', packet), ('pool-reader', reader)]})
    if tamper:
        path = rb.resolve_resource(tamper)
        (path/'fixture.bin' if path.is_dir() else path).write_bytes(b'changed')
    before = {p: p.read_bytes() for p in cache.rglob('*') if p.is_file()}
    monkeypatch.setattr(rf, 'fetch', lambda *a, **kw: pytest.fail('doctor fetched DATA'))
    rows = doctor.check_physics_resources()
    assert len(rows) == 3
    assert {p: p.read_bytes() for p in cache.rglob('*') if p.is_file()} == before
    gaps = doctor.blocking_gaps(rows)
    if tamper:
        index = ['thompson-tables', 'physics-resources', 'pool-reader'].index(tamper)
        assert gaps == [rows[index]]
        assert 'changed' in gaps[0].detail or 'differs from pinned' in gaps[0].detail
        assert gaps[0].remedy == 'woof hex physics-resources fetch'
    else:
        assert not gaps
        assert all(row.status == doctor.VERIFIED for row in rows)
        assert rows[0].evidence['verified_files'] == 1


def test_doctor_names_invalid_installed_record(tmp_path, monkeypatch):
    from woof.hex import doctor
    monkeypatch.setenv('WOOF_HEX_CACHE_ROOT', str(tmp_path))
    record = rb.installed_record_path(); record.parent.mkdir()
    record.write_text('invalid JSON')
    rows = doctor.check_physics_resources()
    assert len(doctor.blocking_gaps(rows)) == 3
    assert all('invalid physics resource record' in row.detail for row in rows)


def test_shipped_manifest_is_data_only():
    pins,_=rf.load_manifest()
    assert len(pins['assets'])==6 and len(pins['upstream_files'])==16
    assert set(pins['resources'])=={'thompson-tables'}
    assert pins['release_assets']['tag']=='physics-resources-v0.4.1'
    for asset in pins['assets']:
        for name,row in asset['files'].items():
            assert not row['executable']
            assert Path(name).suffix in ('.DBL','.bin')


def test_fetch_rejects_native_manifest_before_download(tmp_path,monkeypatch):
    source,pins=sealed(tmp_path,monkeypatch)
    pins['resources']['native-library']={'path':'native/lib.so'}
    with pytest.raises(rb.PhysicsResourceError,match='DATA resources only'):
        rf.fetch(tmp_path/'cache',from_directory=source)
    assert not (tmp_path/'cache').exists()


@pytest.mark.parametrize('executable',[False,True])
def test_fetch_rejects_executable_data(tmp_path,monkeypatch,executable):
    source,pins=sealed(tmp_path,monkeypatch,mode=stat.S_IFREG | 0o755)
    pins['assets'][0]['files']['tables/fixture.bin']['executable']=executable
    with pytest.raises(rb.PhysicsResourceError,match='executable'):
        rf.fetch(tmp_path/'cache',from_directory=source)
    assert not (tmp_path/'cache/installed.json').exists()


def test_rrfs_public_run_refuses_before_hydration(tmp_path,monkeypatch):
    # RRFS_REFUSAL (resources_build, item A) holds only while the fork's ported framework
    # (route_table framework row of mpasphys_rrfs) is absent from the distribution.
    from woof.hex.mpasphys.forecast_v842 import run_forecast
    from woof.hex.mpasphys.route_table import framework_absence
    config=SimpleNamespace(pin_set='rrfs',table_key='mpasphys_rrfs')
    if framework_absence(config) is None:
        pytest.skip('the RRFS ported framework is in this distribution: no library refusal')
    monkeypatch.setenv('WOOF_HEX_CACHE_ROOT',str(tmp_path/'empty'))
    monkeypatch.delenv('MPAS_NATIVE_LIBRARY',raising=False)
    with pytest.raises(ValueError) as error:
        run_forecast(SimpleNamespace(),config,'')
    assert str(error.value)==rb.RRFS_REFUSAL
    assert 'no native-free framework port yet' in str(error.value)
    assert 'Fortran runtime' in str(error.value)
    assert '`woof hex physics-resources build --pin-set rrfs`' in str(error.value)
    assert not (tmp_path/'empty').exists()


def test_cli_offline_directory_reaches_real_fetch(tmp_path,monkeypatch,capsys):
    source,_=sealed(tmp_path,monkeypatch)
    parser=argparse.ArgumentParser();rb.add_physics_resources_parser(parser.add_subparsers())
    args=parser.parse_args(['physics-resources','fetch','--from',str(source),
                            '--cache',str(tmp_path/'cache'),'--pin-set','rrfs','--jobs','2'])
    assert args.handler(args)==0
    assert json.loads(capsys.readouterr().out)['pin_sets']==['rrfs']


def test_packet_selects_run_mode_arch_and_landuse_after_move(tmp_path,monkeypatch):
    from woof.hex import numerics
    cache=tmp_path/'cache';(cache/'packets').mkdir(parents=True)
    path=cache/'packets/strict-sm120.json'
    packet={'sf_noah':{'landuse':rb.DEFAULT_LANDUSE},'bl_mynnedmf':{'image':'strict-sm120'}}
    path.write_text(json.dumps(packet))
    row={'path':str(path),'sha256':rb.sha256_file(path),'bytes':path.stat().st_size}
    rb.record_installed(cache,{'physics-resources':{**row,'variants':{'strict':{'sm_120':row}}}})
    monkeypatch.setattr(numerics,'mode',lambda:'strict')
    monkeypatch.setitem(sys.modules,'cupy',SimpleNamespace(cuda=SimpleNamespace(Device=lambda:SimpleNamespace(compute_capability='120'))))
    moved=tmp_path/'moved';cache.rename(moved)
    # A later installation update must rebase nested packet variants too.
    rb.record_installed(moved,{'pool-reader':{'path':str(moved/'reader'),'sha256':'a'*64}})
    selected=rf.packet_path(moved,landuse='USGS')
    assert selected.is_relative_to(moved)
    actual=rb.load_packet(selected)
    assert actual['sf_noah']['landuse']=='USGS'
    assert actual['bl_mynnedmf']['image']=='strict-sm120'
    selected=rf.packet_path(moved,landuse=rb.DEFAULT_LANDUSE)
    assert rb.load_packet(selected)['sf_noah']['landuse']==rb.DEFAULT_LANDUSE


def test_release_packet_repins_relocated_helper_and_retains_table_pin(tmp_path):
    import runpy
    recipe=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/build_physics_release.py'))
    cache=tmp_path/'cache';(cache/'packets').mkdir(parents=True)
    helper=cache/'decoders/reader';helper.parent.mkdir();helper.write_bytes(b'relocated ELF fixture')
    table=cache/'table';table.write_bytes(b'unchanged table words')
    packet={'rrtmg_lw':{'decoder':{'path':str(helper),'sha256':'old'},
                        'table':{'path':str(table),'sha256':rb.sha256_file(table)}}}
    source=cache/'packets/old.json'
    source.write_text(json.dumps(rb.portable_packet(packet,cache)))
    result=recipe['repin_packet'](source,cache,{helper:rb.sha256_file(helper)})
    target=cache/result['path'];actual=rb.load_packet(target)
    assert rb.sha256_file(target)==result['sha256']
    assert actual['rrtmg_lw']['decoder']['sha256']==rb.sha256_file(helper)
    assert actual['rrtmg_lw']['table']==packet['rrtmg_lw']['table']
    assert str(cache) not in target.read_text()


@pytest.mark.parametrize('pack_hit',[False,True])
def test_explicit_mynn_arch_does_not_query_cuda_device(tmp_path,monkeypatch,pack_hit):
    import base64
    import lzma
    from woof.hex import mynn_glibc_math
    from woof.hex.cuda_backend import aot,aot_build
    expected=b'explicit-target CUBIN fixture'
    def forbidden(*args,**kwargs):
        pytest.fail('explicit architecture construction queried a CUDA device or ordinary compiler')
    monkeypatch.setitem(sys.modules,'cupy',SimpleNamespace(__file__=str(tmp_path/'cupy/__init__.py'),
                        cuda=SimpleNamespace(Device=forbidden)))
    monkeypatch.setattr(mynn_glibc_math,'mynn_glibc_source',lambda:'fixture maths')
    monkeypatch.setattr(aot,'direct_image',forbidden)
    seen=[]
    class Pack:
        def select(self,key,source,options,arch,**kwargs):
            assert arch=='sm_89'
            return ({'fixture':True} if pack_hit else None),'fixture'
        def image(self,entry):
            return expected
    def compile_entry(row,arch):
        seen.append(arch)
        assert row['compiler_route']=='direct-nvrtc'
        return {'cubin_xz_base64':base64.b64encode(lzma.compress(expected)).decode()}
    monkeypatch.setattr(aot,'Pack',Pack)
    monkeypatch.setattr(aot_build,'compile_entry',compile_entry)
    result=rb.build_mynnedmf_image(tmp_path,arch='sm_89',log=lambda _:None)
    assert Path(result['image']).read_bytes()==expected
    assert seen==([] if pack_hit else ['sm_89'])
    assert rb.build_mynnedmf_image(tmp_path,arch='sm_89',log=lambda _:None)==result


def add_upstream(source, pins):
    data=b'pinned upstream table, excluded from release ZIP\n'
    row={'name':'UPSTREAM.TBL','url':'https://raw.githubusercontent.com/MPAS-Dev/MPAS-Data/pinned/UPSTREAM.TBL',
         'commit':'pinned','pin_sets':list(rb.PIN_SETS),'paths':['tables/UPSTREAM.TBL','mpas/files/UPSTREAM.TBL'],
         'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    pins['upstream_files']=[row]
    (source/row['name']).write_bytes(data)
    return row,data


def test_offline_upstream_files_have_independent_pins_and_reverify(tmp_path,monkeypatch):
    source,pins=sealed(tmp_path,monkeypatch)
    row,data=add_upstream(source,pins)
    cache=tmp_path/'cache'
    result=rf.fetch(cache,from_directory=source)
    assert result['upstream_files']==1
    root=Path(result['resources']['pool-reader']['path']).parents[1]
    for relative in row['paths']:
        assert (root/relative).read_bytes()==data
    (root/row['paths'][1]).write_bytes(b'bad')
    with pytest.raises(rb.PhysicsResourceError,match='differs from pinned'):
        rf.fetch(cache,from_directory=source)


def test_online_upstream_uses_packaged_url_in_parallel_with_release(tmp_path,monkeypatch):
    import io
    import threading
    source,pins=sealed(tmp_path,monkeypatch)
    row,data=add_upstream(source,pins)
    pins['release_assets']={'repository':'fixture/release','tag':'fixture'}
    archive=(source/'resources.zip').read_bytes()
    barrier=threading.Barrier(2)
    seen=[]
    def request(url,timeout):
        seen.append(url)
        barrier.wait(timeout=10)
        return io.BytesIO(data if url==row['url'] else archive)
    monkeypatch.setattr(rf.urllib.request,'urlopen',request)
    assert rf.fetch(tmp_path/'cache',jobs=2)['upstream_files']==1
    assert set(seen)=={row['url'],'https://github.com/fixture/release/releases/download/fixture/resources.zip'}


@pytest.mark.parametrize('damage',['corrupt','missing','overlap','traversal'])
def test_upstream_failure_never_publishes(tmp_path,monkeypatch,damage):
    source,pins=sealed(tmp_path,monkeypatch)
    row,_=add_upstream(source,pins)
    if damage=='corrupt': (source/row['name']).write_bytes(b'wrong')
    elif damage=='missing': (source/row['name']).unlink()
    elif damage=='overlap': row['paths'].append('tables/fixture.bin')
    else: row['paths'].append('../escape')
    cache=tmp_path/'cache'
    with pytest.raises(rb.PhysicsResourceError):
        rf.fetch(cache,from_directory=source)
    assert not (cache/'installed.json').exists()
