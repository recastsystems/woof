"""History must not turn untouched host fields into persistent GPU copies."""
from datetime import timedelta
import numpy as np
from woof.hex.mpasphys.streams_output_v842 import PhysicsStreams
from test_mpasphys_streams_output_v842 import session, stream, DeviceArray


def test_unbound_input_history_keeps_host_authority_and_exact_words(tmp_path, monkeypatch):
    monkeypatch.setenv('WOOF_HEX_MEMBER_MEMORY', 'compact')
    state, fields, _, start = session(resident=True)
    state.device_pools.resident_field = lambda *a, **k: None
    state.device_pools.field = lambda *a: (_ for _ in ()).throw(AssertionError('idle upload'))
    state.native.get_field = lambda pool, name, **k: fields[pool, name].copy()
    expected = fields['state', 'scalars'][..., :3].copy()
    with PhysicsStreams(state, (stream(),), tmp_path,
                        final_time=start+timedelta(minutes=2)) as writer:
        snapshot, ready = writer._capture(tuple(writer.streams.values()))
        ready()
        fields['state', 'scalars'][...] = 7
        assert snapshot['qv'].values.tobytes() == expected[0].tobytes()
        assert snapshot['qc'].values.tobytes() == expected[1].tobytes()
        assert state.device_pools.transfers['publication_download_bytes'] == 0


def test_bound_device_history_does_not_read_stale_host(tmp_path, monkeypatch):
    monkeypatch.setenv('WOOF_HEX_MEMBER_MEMORY', 'compact')
    state, fields, _, start = session(resident=True)
    words = fields['state', 'scalars'].copy().view(DeviceArray)
    state.device_pools.resident_field = lambda pool, name, **k: words if name == 'scalars' else None
    state.device_pools.field = lambda *a: (_ for _ in ()).throw(AssertionError('rebinding'))
    with PhysicsStreams(state, (stream(),), tmp_path,
                        final_time=start+timedelta(minutes=2)) as writer:
        snapshot, ready = writer._capture(tuple(writer.streams.values()))
        ready()
        assert snapshot['qv'].values.tobytes() == words[0, :, :3].tobytes()
        assert state.device_pools.transfers['publication_download_bytes'] == words[..., :3].nbytes
