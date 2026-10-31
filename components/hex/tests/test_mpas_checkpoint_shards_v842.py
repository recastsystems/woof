"""Same-host rank checkpoints sealed as per-rank shards beside the step loop (checkpoint_shards_v842).

THE BREAKAGE (rrfs review 2026-10-29): a due checkpoint on same-host ranks gathered every rank's words to rank 0,
which assembled, pickled and hashed one card's carrier on one core while every card waited.  These tests hold the
replacement to the words: two ranks seal their own shards, rank 0 seals the manifest, and ``load_carrier`` returns
the payload the one-file gather seal holds, word for word.
"""
from copy import copy
from fractions import Fraction
import json
import threading
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from woof.hex.mpasphys import checkpoint_shards_v842 as shards
from woof.hex.mpasphys.checkpoint_v842 import resolve_route
from woof.hex.mpasphys.moving_mpi_v842 import DistributedMoving, HostArrays, PoolWords, ServiceWords, WindowWords
from woof.hex.mpasphys.moving_v842 import SAVED_FIELDS, STATE_FIELDS, load_carrier

TEXT = "&nhyd_model\nconfig_start_time='2026-10-17_00:00:00'\n/\n"
CONFIG = SimpleNamespace(receipt=lambda: dict(suite='convection_permitting', options={}))
CELLS = 4
IDS = (np.array([2, 0, 3, 1], np.int32), np.array([3, 1, 2, 0], np.int32))   # 2 owned cells per rank, 2 halo


def partitions():
    return [SimpleNamespace(partition=rank, cell_l2g=ids, edge_l2g=ids, vertex_l2g=ids, n_owned_cells=2,
                            n_owned_edges=2, n_owned_vertices=2, n_local_cells=4, n_local_edges=4,
                            n_local_vertices=4, _global_cells=CELLS, _global_edges=CELLS, _global_vertices=CELLS)
            for rank, ids in enumerate(IDS)]


def enabled(item):
    return True


def layout():
    fields = []
    for pool, name, dimensions, shape, levels, kind in (
            ('diag_physics', 'qke', ['nVertLevels', 'nCells'], [2, CELLS + 1], 2, 'real'),
            ('sfc_input', 'tslb', ['nSoilLevels', 'nCells'], [3, CELLS + 1], 1, 'real'),
            ('sfc_input', 'sst', ['nCells'], [CELLS + 1], 1, 'real'),
            ('tend', 'rt_diabatic_tend', ['nVertLevels', 'nCells'], [2, CELLS + 1], 1, 'real'),
            *(('sfc_input', name, ['nCells'], [CELLS + 1], 1, 'real') for name in ('xland', 'xice', 'isltyp', 'ivgtyp')),
            ('sfc_input', 'isice_lu', [], [], 1, 'integer')):
        fields.append(dict(pool=pool, name=name, pool_name=name, dimensions=dimensions, shape=shape,
                           nlevels=levels, kind=kind, components=[]))
    return SimpleNamespace(fields=fields, configs=dict(config_do_restart=False, config_do_DAcycling=False),
                           dimensions=dict(nCells=CELLS, nEdges=CELLS, nVertices=CELLS, nVertLevels=2, nSoilLevels=3),
                           root=ET.fromstring('<registry><streams><stream name="restart"><var name="qke"/>'
                                              '<var name="tslb"/></stream></streams></registry>'),
                           enabled=enabled)


def global_words(seed=7):
    """One card's words: pools with the native dummy column, the dycore state without it."""
    rng = np.random.default_rng(seed)
    rows = layout().fields
    pools = {}
    for row in rows:
        for level in range(1, row['nlevels'] + 1):
            dtype = np.float32 if row['kind'] == 'real' else np.int32
            words = (rng.standard_normal(row['shape']).astype(dtype) if row['shape']
                     else np.array(3, dtype))
            pools[row['pool'], row['name'], level] = np.asfortranarray(words) if words.ndim else words
    pools['diag_physics', 'qke', 1].view(np.uint32)[0, 0] = 0x80000000          # -0.0 survives
    state = {name: rng.standard_normal((3, 2, CELLS) if name == 'scalars' else (3, CELLS) if name == 'rho_w'
                                       else (2, CELLS)).astype(np.float32) for name in STATE_FIELDS}
    saved = {name: rng.standard_normal((3, CELLS) if name == 'vertical_velocity' else (2, CELLS)).astype(np.float32)
             for name in SAVED_FIELDS}
    window = {'rain': rng.standard_normal(CELLS).astype(np.float32), 'steps': np.arange(CELLS, dtype=np.int32)}
    return pools, state, saved, window


class Collectives:
    """gather/bcast between rank threads of one test."""

    def __init__(self, size):
        self.size, self.barrier, self.slots = size, threading.Barrier(size), [None] * size

    def comm(self, rank):
        shared = self

        class Comm:
            aborted = None

            def Get_rank(self):
                return rank

            def Get_size(self):
                return shared.size

            def gather(self, value, root=0):
                shared.slots[rank] = value
                shared.barrier.wait()
                result = list(shared.slots) if rank == root else None
                shared.barrier.wait()
                return result

            def bcast(self, value, root=0):
                if rank == root:
                    shared.slots[0] = value
                shared.barrier.wait()
                result = shared.slots[0]
                shared.barrier.wait()
                return result
        return Comm()


def member_inputs(registry):
    mesh = SimpleNamespace(arrays=dict(areaCell=np.arange(CELLS, dtype=np.float32) + 1))
    grid = SimpleNamespace(vertical_grid=SimpleNamespace(zgrid=np.arange(12, dtype=np.float32).reshape(3, 4),
                                                         dzw=np.ones((2, 4), np.float32)))
    reference = SimpleNamespace(**{name: np.full((2, CELLS), index + 1, np.float32) for index, name in
                                   enumerate(('rho_base', 'rho_theta_base', 'pressure_base', 'exner_base'))})
    return (mesh, grid, None, None, reference, None, ('qv', 'qc', 'tke'))


def seal(tmp_path, monkeypatch, sharded):
    """Two rank threads take one due checkpoint at step 6 (360 s) and finish; returns the checkpoint path."""
    monkeypatch.setenv(shards.ENVIRONMENT, '1' if sharded else '0')
    registry = layout()
    inputs = member_inputs(registry)
    route, _ = resolve_route(SimpleNamespace(checkpoint_hours='0.1', history_every_minutes=6), TEXT)
    route.resolve_schedule(Fraction(720), Fraction(60), [])
    route.begin('2026-10-17_00:00:00')
    route.bind_inputs(inputs, CONFIG)
    pools, state, saved, window = global_words()
    parts = partitions()
    collectives = Collectives(2)
    out = tmp_path / ('sharded' if sharded else 'gathered')
    out.mkdir()
    errors = []

    def rank_run(rank):
        try:
            ids = IDS[rank]
            native = PoolWords(registry)
            for (pool, name, level), words in pools.items():
                native.data[pool, name, level] = (np.asfortranarray(np.concatenate(
                    (words[..., ids], words[..., -1:]), axis=-1)) if words.ndim else words.copy())
            local = lambda words: np.ascontiguousarray(np.concatenate((words[..., ids], words[..., :1] * 0), -1))
            dycore = SimpleNamespace(cp=HostArrays, atmosphere=SimpleNamespace(
                state=SimpleNamespace(**{k: local(v) for k, v in state.items()}),
                saved=SimpleNamespace(**{k: local(v) for k, v in saved.items()})))
            session = SimpleNamespace(native=native, config=SimpleNamespace(suite='convection_permitting'),
                                      precipitation_window=WindowWords({k: v[ids] for k, v in window.items()}),
                                      surface_service=ServiceWords({'february': 29}), driver=None)
            prepared = dict(route=copy(route), control=None, layout=registry, mesh=inputs[0],
                            vertical=inputs[1].vertical_grid)
            moving = DistributedMoving(collectives.comm(rank), prepared, parts, None)
            # A checkpoint-only step's columns are the resident wind reconstruction, with no psfc_p (runtime_v842
            # prepares psfc_p only for a frame or the shock series): neither seal may read it.
            values = dict(steps=12, dt=60, columns=SimpleNamespace(u_p=None, v_p=None), session=session,
                          dycore=dycore, mesh=None,
                          vertical=None, names=inputs[6], layout=None, out=out)
            assert moving.after_step(6, **values) is False
            moving.finish(**values)
        except BaseException as error:
            errors.append(error)
            collectives.barrier.abort()
    threads = [threading.Thread(target=rank_run, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return out / 'checkpoint.t00000360.pkl'


def same(left, right, where='payload'):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left, right = np.asarray(left), np.asarray(right)
        assert left.dtype == right.dtype and left.shape == right.shape, where
        assert left.tobytes() == right.tobytes(), where
    elif isinstance(left, dict):
        assert set(left) == set(right), where
        for key in left:
            same(left[key], right[key], f'{where}.{key}')
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), where
        for index, (a, b) in enumerate(zip(left, right)):
            same(a, b, f'{where}[{index}]')
    elif isinstance(left, SimpleNamespace):
        same(vars(left), vars(right), where)
    else:
        assert left == right, where


def test_rank_shards_load_as_the_one_file_carrier_word_for_word(tmp_path, monkeypatch):
    gathered = seal(tmp_path, monkeypatch, sharded=False)
    sharded = seal(tmp_path, monkeypatch, sharded=True)
    receipt = json.loads(sharded.with_name(sharded.name + '.json').read_text())
    assert receipt['sharded']['ranks'] == 2 and len(receipt['sharded']['shards']) == 2
    assert receipt['checkpoint']['schema'] == 'mpas-native-checkpoint/v1'
    assert receipt['checkpoint'] == json.loads(gathered.with_name(gathered.name + '.json').read_text())['checkpoint']
    assert (shards.shard_directory(sharded) / 'rank0001.pkl').is_file()
    one, two = load_carrier(gathered), load_carrier(sharded)
    same(one, two)
    np.testing.assert_array_equal(two['state']['rho'], global_words()[1]['rho'])
    assert two['physics']['pools']['diag_physics.qke.1'].view(np.uint32)[0, 0] == 0x80000000
    assert two['physics']['continuation'] is not None


def test_rank_checkpoint_between_stream_frames_publishes_before_sealing():
    """A due seal must fold its diagnostic sample, rather than leave it deferred to the next step."""
    from woof.hex.mpasphys.runtime_v842 import moving_publication_due
    route, _ = resolve_route(SimpleNamespace(checkpoint_hours='0.5', history_every_minutes=30), TEXT)
    moving = DistributedMoving(None, dict(route=route, control=None), [], None)
    assert not moving_publication_due(moving, step=89, steps=180, dt=20)
    assert moving_publication_due(moving, step=90, steps=180, dt=20)


def test_a_damaged_shard_is_refused_by_its_sealed_hash(tmp_path, monkeypatch):
    sharded = seal(tmp_path, monkeypatch, sharded=True)
    shard = shards.shard_directory(sharded) / 'rank0001.pkl'
    data = bytearray(shard.read_bytes())
    data[-40] ^= 1
    shard.write_bytes(bytes(data))
    with pytest.raises(ValueError, match='checkpoint shard does not match'):
        load_carrier(sharded)


def test_rank_zero_waits_for_this_seals_shards_not_a_stale_one(tmp_path):
    writer = shards.ShardWriter(0, 2, timeout=0.3)
    directory = tmp_path / 'checkpoint.f001.pkl.shards'
    directory.mkdir()
    shards.write_json(directory / 'rank0001.pkl.json', dict(token='old', name='rank0001.pkl', bytes=1, sha256='x'))
    with pytest.raises(TimeoutError, match=r'ranks \[0, 1\]'):
        writer._peers(directory, 'new')
