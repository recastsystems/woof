import json
from pathlib import Path

from woof.hex import prep_cache
from woof.hex import moving_prep_maintenance as maintenance


def test_background_eviction_protects_entries_published_after_request(tmp_path, monkeypatch):
    assert maintenance.drain()
    monkeypatch.setattr(prep_cache, '_memory', {})
    monkeypatch.setenv(prep_cache.CAP_ENVIRONMENT, '0.000000001')
    root = tmp_path/'cache'
    for name in ('held','late','spent'):
        entry = root/'aa'/name;entry.mkdir(parents=True)
        (entry/'manifest.json').write_text(json.dumps({'data':name}))
        (entry/'0.npy').write_bytes(b'field words')
    maintenance.request(root, {'held'})
    # This entry was not in request's snapshot. It still must survive the sweep.
    prep_cache._memory['late'] = dict(disk=True)
    assert maintenance.drain()
    assert (root/'aa/held/0.npy').read_bytes() == b'field words'
    assert (root/'aa/late/0.npy').read_bytes() == b'field words'
    assert not (root/'aa/spent').exists()
