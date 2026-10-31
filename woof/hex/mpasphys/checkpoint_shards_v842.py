"""Checkpoints on same-host ranks: every rank seals its own shard, beside the step loop.

THE BREAKAGE THIS FIXES (rrfs review, 2026-10-29): a due checkpoint on N same-host ranks sent every rank's words
through rank 0's links, assembled one card's carrier on rank 0, then pickled it and hashed it there on one core
(DistributedMoving._snapshot, then CheckpointRoute.export) while every card waited.  At about 31 KB a cell that is
some 70 GB and minutes of idle cards per checkpoint at conus3km's 2.34 M cells, every RRFS v2 cycle.

Now the step path keeps only what has to happen at the step boundary: the publication and each rank's own
device-to-host copies, on every rank at once.  A writer thread of each rank then seals that rank's words in
``<checkpoint>.shards/rankNNNN.pkl`` (pickle protocol 5, SHA256 taken from the bytes as they are written, fsync)
while the step loop runs on.  Rank 0's writer waits for every rank's shard receipt and seals the manifest at
``<checkpoint>`` (mesh, vertical grid, layout, partitions, the route as it was at the seal, rank 0's alarm phase,
the shard hashes), then the receipt ``<checkpoint>.json`` last, so a reader that sees the receipt sees a complete
checkpoint (as moving_v842._write_sealed).

``moving_v842.load_carrier`` reads the shards in parallel, each against its sealed hash, assembles them with the
code the gather used (``moving_mpi_v842.assemble_snapshot``) and builds the payload a one-card seal holds
(``MovingRoute.carrier_payload``).  So every reader (one card or N ranks, a resume or a DA analysis) gets the same
words as from the one-file seal.

``WOOF_HEX_CHECKPOINT_SHARDS=0`` keeps the gather-to-rank-0 seal (one file).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import threading
import time
from types import SimpleNamespace

SCHEMA = 'gpuwm-hex.mpas-checkpoint-shards/v1'
ENVIRONMENT = 'WOOF_HEX_CHECKPOINT_SHARDS'
#: How long rank 0's writer waits for the other ranks' shards before it gives the checkpoint up (seconds).
PEER_TIMEOUT_SECONDS = 3600.0


def enabled():
    """Same-host ranks seal checkpoints as shards unless WOOF_HEX_CHECKPOINT_SHARDS is 0/false/off/no."""
    return os.environ.get(ENVIRONMENT, '1').strip().lower() not in ('0', 'false', 'off', 'no')


def shard_directory(target):
    target = Path(target)
    return target.with_name(target.name + '.shards')


def shard_name(rank):
    return f'rank{int(rank):04d}.pkl'


def _now():
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(directory):
    try:
        handle = os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    except OSError:      # a platform whose directories cannot be opened (Windows)
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def seal_file(path, value):
    """Pickle ``value`` (protocol 5) to ``path``: SHA256 of the bytes as written, fsync, then the rename."""
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    digest = hashlib.sha256()

    class HashingWriter:
        def __init__(self, stream):
            self.stream = stream

        def write(self, data):
            digest.update(data)
            return self.stream.write(data)

    with temporary.open('wb') as stream:
        pickle.dump(value, HashingWriter(stream), protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def write_json(path, value):
    """``value`` as JSON at ``path``, replaced whole (a reader never sees half a receipt)."""
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    os.replace(temporary, path)


def manifest(*, route, session, layout, partitions, mesh, vertical, names, step, dt, token, metadata):
    """Rank 0's manifest at the seal: everything but the ranks' words (taken now, pickled by the writer)."""
    from copy import copy
    from datetime import timedelta
    from .moving_v842 import TIME_FORMAT, _manager_state
    frozen = copy(route)
    # The run's cycling hooks and controller hold live objects and play no part in the payload; earlier receipts
    # are this run's, not the checkpoint's.
    frozen.cycling, frozen.control, frozen.checkpoints = None, None, []
    model_seconds = route.offset_seconds + step * dt
    valid = (datetime.strptime(route.origin_time, TIME_FORMAT)
             + timedelta(seconds=model_seconds)).strftime(TIME_FORMAT)
    return dict(schema=SCHEMA, token=token, ranks=len(partitions), step=int(step), dt=dt,
                time_seconds=float(model_seconds), origin_time=route.origin_time, valid_time=valid,
                route=pickle.dumps(frozen, protocol=5), suite=getattr(session.config, 'suite', None),
                manager=_manager_state(session), names=list(names), layout=layout, partitions=list(partitions),
                mesh=mesh, vertical=vertical, n_cells=int(layout.dimensions['nCells']), metadata=metadata)


class ShardWriter:
    """One rank's checkpoint writer: at most one seal in flight, on a daemon thread beside the step loop.

    A daemon, so a failed run is never held at exit by a seal that cannot complete: its shards stay ``.partial``
    or unreferenced and no manifest is written, so nothing reads them.
    """

    def __init__(self, rank, size, *, aborted=None, timeout=PEER_TIMEOUT_SECONDS):
        self.rank, self.size = int(rank), int(size)
        self.timeout = float(timeout)
        self.aborted = aborted or (lambda: None)
        self._thread = None
        self._error = None
        self.records = []

    def wait(self):
        """Block until this rank's previous seal is on disk; returns the seconds waited (step-path seconds)."""
        began = time.perf_counter()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join()
        error, self._error = self._error, None
        if error is not None:
            raise RuntimeError(f'rank {self.rank}: checkpoint shard writer failed: '
                               f'{type(error).__name__}: {error}') from error
        return time.perf_counter() - began

    def submit(self, target, token, words, *, step_seconds, publish_seconds=0.0, manifest=None, on_sealed=None):
        if self._thread is not None:
            raise RuntimeError('a checkpoint seal is already in flight on this rank')
        self._thread = threading.Thread(target=self._run, name=f'hex-checkpoint-shard{self.rank}', daemon=True,
                                        args=(Path(target), token, words, (step_seconds, publish_seconds),
                                              manifest, on_sealed))
        self._thread.start()

    def _run(self, target, token, words, seconds, manifest, on_sealed):
        step_seconds, publish_seconds = seconds
        try:
            directory = shard_directory(target)
            directory.mkdir(parents=True, exist_ok=True)
            began, started = time.perf_counter(), _now()
            path = directory / shard_name(self.rank)
            sealed = seal_file(path, words)
            words = None
            record = dict(schema=SCHEMA, token=token, rank=self.rank, name=path.name, **sealed,
                          step_path_seconds=step_seconds, publish_seconds=publish_seconds,
                          write_seconds=time.perf_counter() - began,
                          started_utc=started, finished_utc=_now(),
                          parallelism=(f'rank {self.rank} writer thread (pid {os.getpid()}) beside the step loop; '
                                       'SHA256 taken as the bytes are written; fsync'))
            write_json(path.with_name(path.name + '.json'), record)
            self.records.append(record)
            if manifest is not None:
                receipt = seal_manifest(target, manifest, self._peers(directory, token))
                if on_sealed is not None:
                    on_sealed(receipt)
        except BaseException as error:   # reported on the step thread at the next wait()
            self._error = error

    def _peers(self, directory, token):
        """Every rank's shard receipt of this seal (same token), in rank order."""
        deadline = time.monotonic() + self.timeout
        found = {}
        while True:
            for rank in range(self.size):
                if rank in found:
                    continue
                try:
                    record = json.loads((directory / (shard_name(rank) + '.json')).read_text())
                except (OSError, ValueError):
                    continue
                if record.get('token') == token:
                    found[rank] = record
            if len(found) == self.size:
                return [found[rank] for rank in range(self.size)]
            reason = self.aborted()
            if reason:
                raise RuntimeError(f'checkpoint not sealed: {reason}')
            if time.monotonic() > deadline:
                missing = sorted(set(range(self.size)) - set(found))
                raise TimeoutError(f'checkpoint not sealed: ranks {missing} wrote no shard in {self.timeout:.0f} s')
            time.sleep(0.05)


def seal_manifest(target, manifest, shards):
    """Seal the manifest at ``target`` and its receipt last; returns the receipt."""
    from .moving_v842 import SCHEMA as CARRIER
    began, started = time.perf_counter(), _now()
    target = Path(target)
    metadata = manifest.pop('metadata')
    manifest['shards'] = [dict(name=record['name'], bytes=int(record['bytes']), sha256=record['sha256'])
                          for record in shards]
    sealed = seal_file(target, manifest)
    total = sealed['bytes'] + sum(int(record['bytes']) for record in shards)
    receipt = dict(schema=CARRIER, path=str(target), bytes=sealed['bytes'], sha256=sealed['sha256'],
                   time_seconds=manifest['time_seconds'], valid_time=manifest['valid_time'], registers_empty=True,
                   sharded=dict(schema=SCHEMA, ranks=manifest['ranks'], directory=str(shard_directory(target)),
                                shards=[{key: record[key] for key in ('rank', 'name', 'bytes', 'sha256',
                                         'publish_seconds', 'step_path_seconds', 'write_seconds', 'started_utc',
                                         'finished_utc')}
                                        for record in shards]),
                   bytes_total=total, bytes_per_cell=total / max(1, manifest['n_cells']),
                   # On the step path, every rank at once: the publication the seal reads (also taken by the one-file
                   # seal), then this rank's own words.  Everything else ran on the writer threads.
                   publish_seconds=max(float(record['publish_seconds']) for record in shards),
                   step_path_seconds=max(float(record['step_path_seconds']) for record in shards),
                   wall_seconds=time.perf_counter() - began)
    receipt['checkpoint'] = metadata
    receipt['checkpoint_export'] = dict(
        started_utc=min(record['started_utc'] for record in shards), manifest_started_utc=started,
        finished_utc=_now(), seconds=max(float(record['write_seconds']) for record in shards),
        parallelism=(f"{manifest['ranks']} rank writer threads beside the step loop, one shard each (SHA256 as "
                     "written, fsync); rank 0 seals the manifest; the step path holds the publication and each "
                     "rank's own device copies only"))
    # The receipt is written last: its presence means the checkpoint is complete.
    write_json(target.with_name(target.name + '.json'), receipt)
    return receipt


def assemble_sharded(path, manifest):
    """The one-card carrier payload of a sharded checkpoint: shards read and verified in parallel, assembled."""
    from .moving_v842 import _read_verified
    from .moving_mpi_v842 import ManagerWords, assemble_snapshot
    directory = shard_directory(path)
    shards = manifest['shards']
    if len(shards) != int(manifest['ranks']):
        raise ValueError('sharded checkpoint does not list one shard per rank')

    def read(entry):
        data = _read_verified(directory / entry['name'], int(entry['bytes']), entry['sha256'], 'checkpoint shard')
        return pickle.loads(memoryview(data))

    with ThreadPoolExecutor(max_workers=max(1, min(len(shards), os.cpu_count() or 1)),
                            thread_name_prefix='hex-checkpoint-read') as pool:
        pieces = list(pool.map(read, shards))
    route = pickle.loads(manifest['route'])
    layout = manifest['layout']
    session, dycore, _ = assemble_snapshot(pieces, layout, manifest['partitions'],
                                           config=SimpleNamespace(suite=manifest['suite']),
                                           manager=ManagerWords(manifest['manager']))
    pieces = None
    payload, _ = route.carrier_payload(session=session, dycore=dycore, mesh=manifest['mesh'],
                                       vertical=manifest['vertical'], names=manifest['names'], layout=layout,
                                       step=manifest['step'], dt=manifest['dt'])
    if (payload['time_seconds'] != manifest['time_seconds'] or payload['valid_time'] != manifest['valid_time']):
        raise ValueError('sharded checkpoint clock differs from its manifest')
    return payload
