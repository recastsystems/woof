"""GPUWM_HEX_DIRECT_SHARD_RESTORE (checkpoint_shards_v842): which resumes take each rank's own shard, on CPU.

The two-rank identity (a resume with the word on against off, every frame word compared) was run on a card;
these check the choice: who is eligible, which carrier a reader gets, and when prepare_moving must assemble instead.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import speed_words
from woof.hex.mpasphys import checkpoint_shards_v842 as shards

WORD = speed_words.DIRECT_SHARD_RESTORE


def args(**kw):
    base = dict(samehost_devices='0,1', da_analysis=None, hosts=None, waves='off')
    base.update(kw)
    return SimpleNamespace(**base)


def test_word_on_by_default_and_off_when_set(monkeypatch):
    monkeypatch.delenv(WORD, raising=False)
    assert speed_words.on(WORD) is True
    assert WORD in speed_words.WORDS
    for off in ('0', 'off', 'false', 'no'):
        monkeypatch.setenv(WORD, off)
        assert speed_words.on(WORD) is False
    monkeypatch.setenv(WORD, '1')
    assert speed_words.on(WORD) is True


def test_eligible_ranks(monkeypatch):
    monkeypatch.setenv(WORD, '1')
    monkeypatch.delenv(shards.ENVIRONMENT, raising=False)
    assert shards._eligible_ranks(args(), None) == 2
    assert shards._eligible_ranks(args(samehost_devices='0,0,1,1'), None) == 4
    assert shards._eligible_ranks(args(samehost_devices=None), None) is None
    assert shards._eligible_ranks(args(da_analysis='x.nc'), None) is None
    assert shards._eligible_ranks(args(hosts='a:0,b:0'), None) == 2
    assert shards._eligible_ranks(args(samehost_devices=None, hosts='a:0-1,b:0-1'), None) == 4
    assert shards._eligible_ranks(args(samehost_devices=None, hosts='bad_spec'), None) is None
    assert shards._eligible_ranks(args(waves='on'), None) is None
    assert shards._eligible_ranks(args(), SimpleNamespace(operator='op.nc')) is None
    monkeypatch.setenv(shards.ENVIRONMENT, '0')
    assert shards._eligible_ranks(args(), None) is None
    monkeypatch.delenv(shards.ENVIRONMENT)
    monkeypatch.setenv(WORD, '0')
    assert shards._eligible_ranks(args(), None) is None


def test_restore_sharded_chooses(monkeypatch):
    monkeypatch.setenv(WORD, '1')
    calls = []
    monkeypatch.setattr(shards, 'direct_carrier', lambda path, manifest: calls.append('direct') or 'direct')
    monkeypatch.setattr(shards, 'assemble_sharded', lambda path, manifest: calls.append('assembled') or 'assembled')
    manifest = dict(ranks=2, shards=[{}, {}])
    assert shards.restore_sharded('c.pkl', manifest) == 'assembled'          # no same-host reader around it
    with shards.direct_restore_for(args(), None):
        assert shards.restore_sharded('c.pkl', manifest) == 'direct'
    with shards.direct_restore_for(args(samehost_devices='0,1,2'), None):
        assert shards.restore_sharded('c.pkl', manifest) == 'assembled'      # another rank count
    assert shards.restore_sharded('c.pkl', manifest) == 'assembled'          # the context is gone again
    assert calls == ['assembled', 'direct', 'assembled', 'assembled']


def part(owned, cells, edges, owned_edges):
    return SimpleNamespace(n_owned_cells=owned, n_owned_edges=owned_edges, cell_l2g=np.array(cells),
                           edge_l2g=np.array(edges))


def test_direct_matches():
    sealed = [part(2, [0, 1, 2], [0, 1], 1), part(1, [2, 1], [1, 0], 1)]
    direct = dict(n_cells=3, names=['qv', 'qc'], ranks=2, manifest=dict(partitions=sealed))
    layout = SimpleNamespace(dimensions=dict(nCells=3))
    same = [part(2, [0, 1, 2], [0, 1], 1), part(1, [2, 1], [1, 0], 1)]
    assert shards.direct_matches(direct, layout, ['qv', 'qc'], same) is None
    assert 'partitions' in shards.direct_matches(direct, layout, ['qv', 'qc'], None)
    assert shards.direct_matches(direct, SimpleNamespace(dimensions=dict(nCells=4)), ['qv', 'qc'], same) == 'another mesh'
    assert shards.direct_matches(direct, layout, ['qv'], same) == 'other species'
    assert 'ranks' in shards.direct_matches(direct, layout, ['qv', 'qc'], same[:1])
    cut = [part(2, [0, 1, 2], [0, 1], 1), part(1, [2, 0], [1, 0], 1)]
    assert shards.direct_matches(direct, layout, ['qv', 'qc'], cut) == 'rank 1 is cut differently from the seal'
