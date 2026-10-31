"""Exercise resource selection without a GPU, compiler, download, or weather run."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex.mpasphys import resources_build as rb
from woof.hex.mpasphys import device_pools_v842 as pools
from woof.hex.mpasphys.driver_v842 import Backend


@pytest.fixture
def builders(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(rb, 'default_cache', lambda: tmp_path / 'cache')
    monkeypatch.setattr(rb, 'build_maths', lambda cache: {
        key: {'path': key} for key in ('bl_mynn', 'mp_thompson', 'mp_tempo')})
    monkeypatch.setattr(rb, 'stage_decoders', lambda *args: {
        key: key for key in ('lookup_decode', 'rw_mpas_rrtmg_lw_tables',
                            'rw_mpas_rrtmg_sw_tables', 'table_decode')})
    monkeypatch.setattr(rb, 'fetch', lambda *args, **kwargs: args[4])
    monkeypatch.setattr(rb, 'build_thompson_strict', lambda *args, **kwargs: tmp_path / 'tables')
    monkeypatch.setattr(rb, '_module_present', lambda name: False)
    monkeypatch.setattr(rb, '_rrfs_module_present', lambda name: True)
    monkeypatch.setattr(rb, 'decoded_digests', lambda packet: {'test_fixture': 'only'})
    monkeypatch.setattr(rb, '_toolchain', lambda compiler: {'test_fixture': 'only'})
    def image(cache, log, *, arch=None, module=None):
        calls.append('compile_rrfs')
        return {'image': 'fixture.cubin', 'metadata': 'fixture.json'}
    monkeypatch.setattr(rb, 'build_mynnedmf_image', image)
    monkeypatch.setenv('WOOF_HEX_PHYSICS_RESOURCES_FIRST_USE', 'on')
    monkeypatch.setattr(rb, 'record_installed', lambda *args: calls.append('record_install'))
    return calls


def test_explicit_packet_build_still_includes_rrfs(builders, tmp_path):
    receipt = rb.build_packet(tmp_path)
    assert json.loads(Path(receipt['packet']).read_text())['bl_mynnedmf']['image'] == 'fixture.cubin'
    # One MYNN-EDMF image per pinned RRFS row (resources_build.rrfs_mynnedmf_images: c942418 for 1d6e370e,
    # c6d08709 for 44c0849), each built once.
    assert builders == ['compile_rrfs'] * len(rb.rrfs_mynnedmf_images())


@pytest.mark.parametrize('backend,active,expected', [
    (Backend.HEX_CUDA, True, True),
    (Backend.NATIVE_CPU, True, False),
    (Backend.HEX_CUDA, False, False),
])
def test_slot_registration_requests_prebuilt_image_only_for_selected_cuda_slot(monkeypatch, builders,
                                                            backend, active, expected):
    # Put a different resource slot first: selection must see all active slots,
    # not just the callback that triggered the first-use packet build.
    config = SimpleNamespace(
        table_key='test', selectors={'microphysics': 'mp_thompson', 'pbl': 'bl_mynnedmf'},
        backends={'microphysics': Backend.HEX_CUDA, 'pbl': backend},
        active=lambda slot: slot != 'pbl' or active)
    descriptor = {'module': 'fixture', 'call': 'run',
                  'resources': {'module': 'fixture', 'call': 'load', 'attribute': 'loaded'}}
    monkeypatch.setattr(pools, 'load_table', lambda: {'test': {'slots': {
        slot: {'cuda_entries': {scheme: descriptor}} for slot, scheme in config.selectors.items()}}})
    packet = rb.default_cache() / 'prebuilt.json'
    packet.parent.mkdir(parents=True, exist_ok=True)
    packet.write_text(json.dumps({'mp_thompson': {}, 'bl_mynnedmf': {'image': 'prebuilt.cubin'}}))
    requests = []
    def prebuilt(**kwargs):
        requests.append(kwargs)
        return packet
    monkeypatch.setattr(rb, 'ensure_packet', prebuilt)
    loaded = []
    module = SimpleNamespace(run=lambda call: None, load=lambda packet, cp: loaded.append(packet))
    monkeypatch.setattr(pools, 'importlib', SimpleNamespace(import_module=lambda *args: module))
    provider = SimpleNamespace(native=SimpleNamespace(get_text_field=lambda *args: 'USGS'), cp=None)
    registered = []
    registry = SimpleNamespace(register=lambda *args, **kwargs: registered.append(args))
    pools.register_cuda_slots(registry, config, provider)
    assert builders == []
    assert requests == [dict(landuse='USGS', include_mynnedmf=expected)]
    assert len(registered) == len(loaded) == (2 if active and backend == Backend.HEX_CUDA else 1)
    assert loaded == ([{}, {"image": "prebuilt.cubin"}] if expected else [{}])
