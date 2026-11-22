"""Two multi-host edge cases found on 20 Nov 2026, and their fixes.

(b) a joined host with no shared filesystem resumes from its own shards: rank 0 mirrors each seal's manifest (without
    the global mesh and vertical grid) to every joined host's first rank, which seals it beside its host's shards.
(a) WOOF_HEX_PER_RANK_START stops by name where a rank still needs the global mesh it lets go of.
"""
from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import multihost
from woof.hex.mpasphys import checkpoint_shards_v842 as shards
from woof.hex.mpasphys import per_rank_start


class TinyRoute:
    """The two calls direct_carrier makes of the frozen route."""

    def carried_rows(self, layout):
        return [dict(pool='state', name='theta', dimensions=['nCells'], nlevels=1, kind='float')]

    def physics_clock_offset(self, dt):
        return (0, 0)


def _manifest(token, n_cells=10):
    return dict(schema=shards.SCHEMA, token=token, ranks=2, step=6, dt=600.0, time_seconds=3600.0,
                origin_time='2024-05-21_18:00:00', valid_time='2024-05-21_19:00:00',
                route=pickle.dumps(TinyRoute(), protocol=5), suite='mpas', manager={'alarms': []},
                names=['qv'], layout={'dimensions': {'nCells': n_cells}},
                partitions=[SimpleNamespace(n_owned_cells=5), SimpleNamespace(n_owned_cells=5)],
                mesh={'big': np.arange(4096, dtype=np.float64)}, vertical={'zgrid': np.ones((n_cells, 3))},
                n_cells=n_cells, metadata={'dt': 600.0})


def _seal_two_hosts(tmp_path, token):
    rv = multihost.Rendezvous(host='127.0.0.1', port=0, token='tok-fold')
    spec = dict(address='127.0.0.1', port=rv.port, token=rv.token, timeout=20.0)
    host0 = tmp_path / 'h0' / 'checkpoint.f001.pkl'
    host1 = tmp_path / 'h1' / 'checkpoint.f001.pkl'
    host0.parent.mkdir()
    host1.parent.mkdir()
    try:
        w0 = shards.ShardWriter(0, 2, rendezvous=rv, mirror_ranks=[1], timeout=20.0)
        w1 = shards.ShardWriter(1, 2, multihost_connect=spec, mirror=True, timeout=20.0)
        w1.submit(host1, token, {'rank': 1, 'theta': np.full(5, 1.5)}, step_seconds=0.01)
        w0.submit(host0, token, {'rank': 0, 'theta': np.full(5, 0.5)}, step_seconds=0.01,
                  manifest=_manifest(token))
        w0.wait()
        w1.wait()
    finally:
        rv.close()
    return host0, host1


def test_joined_host_seals_a_manifest_mirror_beside_its_own_shards(tmp_path):
    host0, host1 = _seal_two_hosts(tmp_path, 'seal-b-1')
    # rank 0 holds the whole manifest; host 1 holds the mirror, sealed, beside its only shard
    full = pickle.loads(host0.read_bytes())
    mirror = pickle.loads(host1.read_bytes())
    assert full['mesh'] is not None and 'mirror' not in full
    assert mirror['mesh'] is None and mirror['vertical'] is None and mirror['mirror'] is True
    assert mirror['shards'] == full['shards'] and len(mirror['shards']) == 2
    receipt = json.loads(host1.with_name(host1.name + '.json').read_text())
    assert receipt['path'] == str(host1)
    assert receipt['bytes'] == host1.stat().st_size and receipt['bytes'] < full_bytes(host0)
    assert receipt['mirror_of']['path'] == str(host0)
    assert receipt['sharded']['directory'] == str(shards.shard_directory(host1))
    assert sorted(p.name for p in shards.shard_directory(host1).glob('*.pkl')) == ['rank0001.pkl']


def full_bytes(path):
    return path.stat().st_size


def test_joined_host_restores_its_ranks_from_its_own_disk(tmp_path, monkeypatch):
    _, host1 = _seal_two_hosts(tmp_path, 'seal-b-2')
    monkeypatch.setenv(multihost.ENV_HOST_INDEX, '1')
    monkeypatch.delenv(multihost.ENV_JOIN, raising=False)
    manifest = pickle.loads(host1.read_bytes())
    token = shards._DIRECT_RANKS.set(2)
    try:
        carrier = shards.restore_sharded(host1, manifest)
    finally:
        shards._DIRECT_RANKS.reset(token)
    direct = carrier['direct_shards']
    assert direct['directory'] == str(shards.shard_directory(host1))
    words = shards.read_shard(direct['directory'], direct['shards'][1])
    assert words['rank'] == 1 and np.array_equal(words['theta'], np.full(5, 1.5))
    assert carrier['mesh'] is None and carrier['valid_time'] == '2024-05-21_19:00:00'


def test_joined_host_refuses_by_name_what_it_cannot_assemble(tmp_path, monkeypatch):
    _, host1 = _seal_two_hosts(tmp_path, 'seal-b-3')
    monkeypatch.setenv(multihost.ENV_HOST_INDEX, '1')
    manifest = pickle.loads(host1.read_bytes())
    token = shards._DIRECT_RANKS.set(3)          # another rank count than the seal's
    try:
        with pytest.raises(ValueError, match='host 1 cannot resume'):
            shards.restore_sharded(host1, manifest)
    finally:
        shards._DIRECT_RANKS.reset(token)


def test_joined_host_names_an_old_checkpoint_without_its_shards(tmp_path, monkeypatch):
    host0, _ = _seal_two_hosts(tmp_path, 'seal-b-4')
    monkeypatch.setenv(multihost.ENV_HOST_INDEX, '1')
    manifest = pickle.loads(host0.read_bytes())   # rank 0's whole manifest, read on a host that lacks rank 0's shard
    (shards.shard_directory(host0) / 'rank0000.pkl').unlink()
    token = shards._DIRECT_RANKS.set(2)
    try:
        with pytest.raises(ValueError, match='host 1 cannot resume.*no manifest mirror'):
            shards.restore_sharded(host0, manifest)
    finally:
        shards._DIRECT_RANKS.reset(token)


def test_joined_carrier_takes_the_host_folder_copy(tmp_path, monkeypatch):
    given = tmp_path / 'out' / 'checkpoint.f003.pkl'
    mine = tmp_path / 'out' / '.multihost' / 'host1' / 'checkpoint.f003.pkl'
    mine.parent.mkdir(parents=True)
    monkeypatch.delenv(multihost.ENV_JOIN, raising=False)
    monkeypatch.delenv(multihost.ENV_HOST_INDEX, raising=False)
    assert shards.joined_carrier(str(given)) == str(given)          # one host: unchanged
    monkeypatch.setenv(multihost.ENV_JOIN, json.dumps(dict(host=1, address='x', port=1, token='t')))
    assert shards.joined_carrier(str(given)) == str(given)          # no copy sealed there: the given path
    mine.with_name(mine.name + '.json').write_text('{}')
    assert shards.joined_carrier(str(given)) == str(mine)


def test_per_rank_start_refuses_a_limited_area_mesh_by_name(monkeypatch):
    lam = SimpleNamespace(configs={'config_apply_lbcs': True})
    globe = SimpleNamespace(configs={'config_apply_lbcs': False})
    monkeypatch.delenv(per_rank_start.ENV_START, raising=False)
    monkeypatch.delenv(per_rank_start.ENV_SHARDS, raising=False)
    per_rank_start.refuse_unsupported(lam)                         # the word off: nothing to refuse
    monkeypatch.setenv(per_rank_start.ENV_START, '1')
    with pytest.raises(ValueError, match='WOOF_HEX_PER_RANK_START is refused on a limited-area mesh'):
        per_rank_start.refuse_unsupported(lam)
    per_rank_start.refuse_unsupported(globe)


def test_per_rank_start_refuses_the_one_card_helicity_order(monkeypatch):
    from woof.hex.mpasphys import route_table

    class ForkOutputDiagnostics:
        pass
    monkeypatch.setattr(route_table, 'output_diagnostics_type', lambda config=None: ForkOutputDiagnostics)
    monkeypatch.setenv(per_rank_start.ENV_START, '1')
    monkeypatch.delenv('WOOF_HEX_UH_RANK_ORDER', raising=False)
    globe = SimpleNamespace(configs={})
    with pytest.raises(ValueError, match='helicity'):
        per_rank_start.refuse_unsupported(globe, SimpleNamespace())
    monkeypatch.setenv('WOOF_HEX_UH_RANK_ORDER', 'native')
    per_rank_start.refuse_unsupported(globe, SimpleNamespace())
