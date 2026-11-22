"""Tiny real transport/array fixtures; storage evidence, not a forecast claim."""
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from woof.hex import native_geometry as ng


@pytest.fixture
def native(tmp_path):
    # Different cell and edge extents and nontrivial float words exercise axes.
    arrays = dict(indexToCellID=np.array([1, 2, 3], '<i4'),
        indexToEdgeID=np.array([1, 2, 3, 4], '<i4'),
        latCell=np.array([.125, -.25, .5], '<f4'), lonCell=np.array([1., 2., 3.], '<f4'),
        latEdge=np.array([.25, .5, .75, 1.], '<f4'), lonEdge=np.array([-1., -2., -3., -4.], '<f4'))
    init = str(tmp_path / 'actual-source.nc')
    rows = []
    for n, (dim, kind) in ng.FIELDS.items():
        # Include native's real dummy entity and prove it never reaches output.
        b = np.append(arrays[n], np.array([0], dtype=arrays[n].dtype)).astype(arrays[n].dtype).tobytes(order='F')
        payload = f'{n}.bin'; (tmp_path / payload).write_bytes(b)
        rows.append(dict(name=n, pool='mesh', kind=kind, dimensions=[dim],
            shape=[len(arrays[n])+1], payload=payload, sha256=hashlib.sha256(b).hexdigest(), origins=[f'{init}:{n}']))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema='mpas-native-pool-payload-v1', fields=rows)))
    request = tmp_path / 'native-physics-request.json'
    request.write_text(json.dumps(dict(schema='mpas-native-door-v1', init=init,
        start='2000-01-01_00:00:00', stop='2000-01-01_00:01:00', dt='20', steps=3,
        pool_manifest=str(manifest), identity_checked=False)))
    return arrays, request, '2000-01-01_00:01:00'


def test_seal_preserves_actual_words_and_native_dummy_semantics(native):
    arrays, request, valid = native
    before = {n: a.tobytes() for n, a in arrays.items()}
    record = ng.seal(arrays, request, valid, 3)
    assert ng.validate(arrays, record, request, valid) == record
    assert all(a.tobytes() == before[n] for n, a in arrays.items())
    assert all(r['source']['shape'][0] == r['array']['shape'][0]+1 for r in record['fields'].values())
    assert record['role'] == 'static-identity-not-analysis-control'


@pytest.mark.parametrize('mode', ['permutation', 'geometry', 'clock', 'source', 'payload', 'missing', 'no-origin'])
def test_exact_bad_authority_refuses(native, mode):
    arrays, request, valid = native
    record = ng.seal(arrays, request, valid, 3)
    if mode == 'permutation': arrays = {n: a[::-1].copy() for n, a in arrays.items()}
    elif mode == 'geometry': arrays['latCell'][1] = np.nextafter(arrays['latCell'][1], np.float32(1))
    elif mode == 'clock': valid = '2000-01-01_00:00:40'
    elif mode == 'source':
        r = json.loads(request.read_text()); r['identity_checked'] = True; request.write_text(json.dumps(r))
    elif mode == 'payload':
        p = request.parent/'latCell.bin'; p.write_bytes(bytes(len(p.read_bytes())))
    elif mode == 'missing': del arrays['indexToCellID']
    else:
        p=request.parent/'manifest.json'; m=json.loads(p.read_text()); m['fields'][0]['origins']=[];p.write_text(json.dumps(m))
    with pytest.raises(ValueError): ng.validate(arrays, record, request, valid)


def test_publication_selects_actual_registry_slots_only():
    @dataclass(frozen=True)
    class Stream:
        name: str
        directions: tuple
        fields: tuple
    rows = [dict(name=n, pool='mesh', kind=k, dimensions=[d]) for n,(d,k) in ng.FIELDS.items()]
    streams = [Stream('output', ('output',), ('theta',)), Stream('restart', ('output',), ('rho',))]
    result = ng.publish_native_identity(streams, SimpleNamespace(fields=rows))
    assert result[0].fields == ('theta', *ng.FIELDS)
    assert result[1] == streams[1] and streams[0].fields == ('theta',)
    with pytest.raises(ValueError, match='actual Registry'):
        ng.publish_native_identity(streams, SimpleNamespace(fields=rows[1:]))


def test_native_ids_are_carried_from_source_and_never_replaced_by_a_range(native):
    arrays,request,valid=native
    arrays['indexToCellID']=np.array([7,3,11],'<i4')
    b=np.append(arrays['indexToCellID'],np.int32(0)).astype('<i4').tobytes(order='F')
    (request.parent/'indexToCellID.bin').write_bytes(b)
    p=request.parent/'manifest.json';m=json.loads(p.read_text())
    next(r for r in m['fields'] if r['name']=='indexToCellID')['sha256']=hashlib.sha256(b).hexdigest()
    p.write_text(json.dumps(m))
    r=ng.seal(arrays,request,valid,3)
    assert r['fields']['indexToCellID']['array']==ng.fingerprint(np.array([7,3,11],'<i4'))


@pytest.mark.parametrize('class_rate', [None, 1, 8])
@pytest.mark.parametrize('class_step', [0, 3])
@pytest.mark.parametrize('publication', [False, True])
def test_actual_encoder_hands_words_and_clock_to_seal(native, monkeypatch, class_rate, class_step, publication):
    from woof.hex.mpasphys.streams_output_v842 import PhysicsStreams
    from woof.hex.output import HistoryField
    arrays, request, valid = native
    rate = class_rate or 1
    r = json.loads(request.read_text())
    r.update(dt=str(20 / rate), steps=3 * rate)
    if publication:
        r['native_state_publication'] = True
    request.write_text(json.dumps(r))
    if class_step == 0:
        valid = r['start']
    prognostic = np.array([0x80000000, 1, 0x3f800001], np.uint32).view(np.float32)
    snapshot = {n: HistoryField(a.copy(), (ng.FIELDS[n][0],), {}) for n,a in arrays.items()}
    snapshot['theta'] = HistoryField(prognostic.copy(), ('nCells',), {})
    seen = []
    def write(target, stream, mesh, fields, when, **options):
        seen.append({n: (x.values.tobytes(), x.values.shape, x.values.dtype) for n,x in fields.items()})
        target.write_bytes(b'existing-encoder-fixture')
    if class_rate is None:
        writer_type = PhysicsStreams
    else:
        from woof.hex.mpasphys.lts_v842 import LtsClassExecution
        writer_type = LtsClassExecution._streams_type(SimpleNamespace(index=0, rate=class_rate))
    w = object.__new__(writer_type)
    w.output = request.parent; w.layout = SimpleNamespace(configs={}, packages=())
    w.initial = datetime(2000,1,1); w.mesh = SimpleNamespace(); w.field_attrs = {}; w.streams = {}
    target = request.parent/'history.nc'; w._file_attributes = {target:{}}
    w._bind_file_attributes = lambda jobs: None; w._publish_closed = lambda jobs, step: None
    w.contract = SimpleNamespace(writer=SimpleNamespace(write_stream=write))
    from woof.hex.streams_v842 import Stream
    stream = Stream('output', {'type': 'output'}, (('var', 'theta', ''),))
    records = w._encode([(stream, target, 0)], snapshot, datetime.strptime(valid, ng.CLOCK), class_step)
    assert records[0]['step'] == class_step
    assert target.read_bytes() == b'existing-encoder-fixture'
    assert seen[0]['theta'] == (prognostic.tobytes(), prognostic.shape, prognostic.dtype)
    if publication:
        actual = json.loads(ng.sidecar(target).read_text())['0']
        assert actual == ng.seal(arrays, request, valid, class_step * rate)
    else:
        assert not ng.sidecar(target).exists()
    for n,a in arrays.items(): assert seen[0][n] == (a.tobytes(), a.shape, a.dtype)


def test_copied_transport_validates_after_original_directory_is_removed(native):
    import shutil
    arrays, request, valid = native
    evidence = ng.seal(arrays, request, valid, 3)
    original = request.parent
    relocated = original.with_name(original.name + '-relocated')
    shutil.copytree(original, relocated)
    original.rename(original.with_name(original.name + '-retired'))
    actual = ng.validate(arrays, evidence, relocated/request.name, valid)
    assert actual['request']['sha256'] == evidence['request']['sha256']
    assert actual['request']['path'] != evidence['request']['path']


def test_rebound_transport_locators_preserve_science_but_settings_changes_refuse(native):
    import shutil
    arrays, request, valid = native
    evidence = ng.seal(arrays, request, valid, 3)
    original = request.parent
    relocated = original.with_name(original.name + '-rebound')
    shutil.copytree(original, relocated)
    original.rename(original.with_name(original.name + '-retired'))
    new_request = relocated/request.name
    for path in (new_request, relocated/'manifest.json'):
        path.write_text(path.read_text().replace(str(original), str(relocated)))
    actual = ng.validate(arrays, evidence, new_request, valid)
    assert actual['request']['sha256'] != evidence['request']['sha256']
    assert actual['scientific_identity'] == evidence['scientific_identity']
    settings = json.loads(new_request.read_text())
    settings['identity_checked'] = True
    new_request.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match='binding differs'):
        ng.validate(arrays, evidence, new_request, valid)


def test_native_reader_bytes_are_bound_after_relocation(native):
    import shutil
    arrays, request, valid = native
    reader = request.parent/'pool-reader'
    reader.write_bytes(b'actual native reader')
    settings = json.loads(request.read_text())
    settings['rust_pool_reader'] = str(reader)
    request.write_text(json.dumps(settings))
    evidence = ng.seal(arrays, request, valid, 3)
    original = request.parent
    relocated = original.with_name(original.name + '-reader-moved')
    shutil.copytree(original, relocated)
    original.rename(original.with_name(original.name + '-retired'))
    new_request = relocated/request.name
    for path in (new_request, relocated/'manifest.json'):
        path.write_text(path.read_text().replace(str(original), str(relocated)))
    ng.validate(arrays, evidence, new_request, valid)
    (relocated/'pool-reader').write_bytes(b'changed native reader')
    with pytest.raises(ValueError, match='binding differs'):
        ng.validate(arrays, evidence, new_request, valid)


@pytest.mark.parametrize('module', ['streams_v842', 'streams_rrfs'])
def test_runtime_publication_preserves_packages_and_restart(module):
    from importlib import import_module
    Stream = import_module('woof.hex.' + module).Stream
    stream = Stream('output', {'type': 'output'}, (('var', 'theta', ''),))
    restart = Stream('restart', {'type': 'output'}, (('var', 'rho', ''),))
    rows = [dict(name=n, pool='mesh', kind=k, dimensions=[d]) for n,(d,k) in ng.FIELDS.items()]
    output, unchanged = ng.publish_native_identity([stream, restart], SimpleNamespace(fields=rows))
    assert output.fields() == ('theta', *ng.FIELDS)
    assert stream.fields() == ('theta',)
    assert unchanged is restart
    assert ng.publish_native_identity([output], SimpleNamespace(fields=rows)) == [output]


@pytest.mark.parametrize('identity', [False, True])
def test_explicit_stream_encoder_hands_words_and_clock_to_seal(native, monkeypatch, identity):
    from woof.hex.mpasphys.streams_output_v842 import PhysicsStreams
    from woof.hex.output import HistoryField
    from woof.hex.streams_v842 import Stream
    arrays, request, valid = native
    prognostic = np.array([0x80000000, 1, 0x3f800001], np.uint32).view(np.float32)
    snapshot = {n: HistoryField(a.copy(), (ng.FIELDS[n][0],), {}) for n,a in arrays.items()}
    snapshot['theta'] = HistoryField(prognostic.copy(), ('nCells',), {})
    seen = []
    def write(target, stream, mesh, fields, when, **options):
        seen.append({n: (x.values.tobytes(), x.values.shape, x.values.dtype) for n,x in fields.items()})
        target.write_bytes(b'existing-encoder-fixture')
    w = object.__new__(PhysicsStreams)
    w.output = request.parent; w.layout = SimpleNamespace(configs={}, packages=())
    w.initial = datetime(2000,1,1); w.mesh = SimpleNamespace(); w.field_attrs = {}; w.streams = {}
    target = request.parent/'history.nc'; w._file_attributes = {target:{}}
    w._bind_file_attributes = lambda jobs: None; w._publish_closed = lambda jobs, step: None
    w.contract = SimpleNamespace(writer=SimpleNamespace(write_stream=write))
    # Use the production stream API and exercise included-stream expansion.
    members = (('stream', 'native_geometry'),) if identity else ()
    stream = Stream('output', {'type': 'output'},
                    (('var', 'theta', ''),) + tuple((*m, '') for m in members))
    w.streams = {'output': stream, 'native_geometry': Stream('native_geometry',
        {'type': 'output'}, tuple(('var', n, '') for n in ng.FIELDS))}
    w._encode([(stream, target, 0)], snapshot, datetime.strptime(valid, ng.CLOCK), 3)
    assert seen[0]['theta'] == (prognostic.tobytes(), prognostic.shape, prognostic.dtype)
    if identity:
        actual = json.loads(ng.sidecar(target).read_text())['0']
        assert actual == ng.seal(arrays, request, valid, 3)
    else:
        assert not ng.sidecar(target).exists()
    for n,a in arrays.items(): assert seen[0][n] == (a.tobytes(), a.shape, a.dtype)
