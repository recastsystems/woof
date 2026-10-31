from concurrent.futures import ThreadPoolExecutor
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys import moving_tables_v842 as tables


def test_certified_tables_share_only_read_only_data_and_reauthenticate_changed_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, '_TABLES', {})
    path = tmp_path/'data';path.write_bytes(b'original')
    packet = dict(table=dict(path=str(path), sha256=hashlib.sha256(b'original').hexdigest()))
    calls = []
    data = np.array([0x7fc00001, 0x80000000], dtype=np.uint32).view(np.float32)
    data.flags.writeable = False
    def load(packet, device):
        calls.append(1)
        if hashlib.sha256(path.read_bytes()).hexdigest() != packet['table']['sha256']:
            raise ValueError('certificate differs')
        return dict(tables={'values':data}, labels=['original'])
    module = SimpleNamespace(__name__='woof.hex.mpasphys.rrtmg_lw_v842', load_native_lw_resources=load)
    device = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda:0),
        get_current_stream=lambda:SimpleNamespace(synchronize=lambda:None)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i:tables.load(module, 'load_native_lw_resources', packet, device), range(4)))
    assert len(calls) == 1
    assert all(r['tables']['values'] is data for r in results)
    assert results[0]['tables']['values'].tobytes() == data.tobytes()
    results[0]['tables']['added'] = 'session-only';results[0]['labels'].append('session-only')
    assert 'added' not in results[1]['tables'] and results[1]['labels'] == ['original']
    path.write_bytes(b'changed!')
    with pytest.raises(ValueError, match='certificate differs'):
        tables.load(module, 'load_native_lw_resources', packet, device)
    assert len(calls) == 2
    device.cuda.runtime.getDevice = lambda:1
    path.write_bytes(b'original')
    tables.load(module, 'load_native_lw_resources', packet, device)
    assert len(calls) == 3


def test_shortwave_sessions_keep_private_workspaces_and_share_initialized_tables():
    engine_type = type('MPASShortwave', (), {'__module__':'woof.hex.mpasphys.rrtmg_sw_v842'})
    template = engine_type()
    template.tab_gpu = object();template._tables_ready = object()
    template._shape_modules = {'shape':object()}
    template._table_streams = {};template._scratch_by_stream = {};template._chunk_by_nlayers = {}
    template.max_nlay = None
    a, b = tables._session_copy(template), tables._session_copy(template)
    assert a is not b and a.tab_gpu is b.tab_gpu is template.tab_gpu
    assert a._tables_ready is template._tables_ready
    for name in ('_shape_modules','_table_streams','_scratch_by_stream','_chunk_by_nlayers'):
        assert getattr(a, name) is not getattr(b, name)
    a._scratch_by_stream[1] = object();a.max_nlay = 55
    assert not b._scratch_by_stream and b.max_nlay is None


def test_unknown_resource_loader_is_never_retained(monkeypatch):
    monkeypatch.setattr(tables, '_TABLES', {})
    calls = []
    module = SimpleNamespace(__name__='woof.hex.mpasphys.other', load=lambda *a, **kw:calls.append(1))
    tables.load(module, 'load', {}, None);tables.load(module, 'load', {}, None)
    assert calls == [1, 1] and not tables._TABLES


def test_resolved_packet_metadata_is_private_and_reloads_when_an_input_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, '_PACKETS', {})
    source = tmp_path/'packet.json';source.write_text('{}')
    data = tmp_path/'table';data.write_text('original')
    calls = []
    def reader(path):
        calls.append(path)
        return dict(data=dict(path=str(data)), options=['native'])
    a = tables.packet(source, reader)
    a['options'].append('session-only')
    assert tables.packet(source, reader)['options'] == ['native']
    assert len(calls) == 1
    data.write_text('changed')
    tables.packet(source, reader)
    assert len(calls) == 2
    source.with_suffix('.receipt.json').write_text('{"cache":"elsewhere"}')
    tables.packet(source, reader)
    assert len(calls) == 3
