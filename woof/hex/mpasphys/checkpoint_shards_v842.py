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
import sys
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

    def __init__(self, rank, size, *, aborted=None, timeout=PEER_TIMEOUT_SECONDS,
                 multihost_connect=None, rendezvous=None, mirror_ranks=None, mirror=False):
        self.rank, self.size = int(rank), int(size)
        self.timeout = float(timeout)
        self.aborted = aborted or (lambda: None)
        self.multihost_connect = multihost_connect
        self.rendezvous = rendezvous
        # Several hosts: rank 0 hands each joined host's first rank (``mirror_ranks``) a mirror of the manifest, and
        # that rank (``mirror``) seals it beside its host's shards, so the host resumes from its own disk.
        self.mirror_ranks = [int(r) for r in (mirror_ranks or ())]
        self.mirror = bool(mirror)
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
            if self.multihost_connect is not None:
                from .. import multihost
                multihost.send_shard_receipt(self.multihost_connect, token, self.rank, record)
                if self.mirror:
                    receive_mirror(self.multihost_connect, token, self.rank, target)
            self.records.append(record)
            if manifest is not None:
                receipt = seal_manifest(target, manifest, self._peers(directory, token))
                if self.mirror_ranks and self.rendezvous is not None:
                    send_mirror(self.rendezvous, token, self.mirror_ranks, manifest, receipt,
                                timeout=self.timeout, aborted=self.aborted)
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
                record = None
                try:
                    record = json.loads((directory / (shard_name(rank) + '.json')).read_text())
                except (OSError, ValueError):
                    record = None
                if (record is None or record.get('token') != token) and self.rendezvous is not None:
                    from .. import multihost
                    remote = multihost.recv_shard_receipt(self.rendezvous, token, rank)
                    if remote is not None and remote.get('token') == token:
                        record = remote
                        try:
                            write_json(directory / (shard_name(rank) + '.json'), record)
                        except OSError:
                            pass
                if record is not None and record.get('token') == token:
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


def light_manifest(manifest):
    """The manifest a joined host restores from: rank 0's sealed manifest without the global mesh and vertical grid.
    Only a remapped move and the wave restore read those (moving_v842.resume_dycore, waves), and neither restores
    rank by rank (``_eligible_ranks``), so a joined host never needs them.  At 10 km they are most of the manifest's
    56.7 GB (a 10 km run's receipt)."""
    light = dict(manifest)
    light['mesh'] = None
    light['vertical'] = None
    light['mirror'] = True
    return light


def send_mirror(rendezvous, token, ranks, manifest, receipt, *, timeout=PEER_TIMEOUT_SECONDS, aborted=None):
    """Rank 0's writer after its manifest seal: each joined host's first rank takes the light manifest and rank 0's
    receipt over the rendezvous.  THE BREAKAGE THIS FIXES (a multi-host edge case found on 20 Nov 2026): on
    hosts with no shared filesystem a joined host held its own shards but no manifest, so its resume read rank 0's
    carrier path, which on that host was absent or another file ('moving carrier does not match its sealed receipt',
    a 10 km resume, host 1 exit 2)."""
    from .. import multihost
    data = pickle.dumps(light_manifest(manifest), protocol=5)
    header = dict(token=token, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                  receipt=json.loads(json.dumps(receipt, default=str)))
    for rank in ranks:
        multihost.send_manifest_mirror(rendezvous, token, rank, header, data, timeout=timeout, failed=aborted)


def receive_mirror(spec, token, rank, target):
    """A joined host's first rank: seal rank 0's light manifest at this host's ``target`` (beside its shards) and the
    receipt last, so the folder reads as a sealed checkpoint whose shards are this host's."""
    from .. import multihost
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.partial')
    header = multihost.recv_manifest_mirror(spec, token, rank, temporary)
    os.replace(temporary, target)
    _fsync_directory(target.parent)
    origin = header['receipt']
    receipt = dict(origin, path=str(target), bytes=int(header['bytes']), sha256=header['sha256'],
                   mirror_of=dict(path=origin.get('path'), bytes=origin.get('bytes'), sha256=origin.get('sha256')))
    receipt['sharded'] = dict(origin.get('sharded') or {}, directory=str(shard_directory(target)))
    write_json(target.with_name(target.name + '.json'), receipt)
    return receipt


def joined_carrier(path):
    """The checkpoint a joined host resumes from: its own mirror under rank 0's folder/.multihost/host<N>/ when the
    seal left one there (a joined host's --out), else the path it was given (an external launch names its own)."""
    from ..multihost import joined_host_index
    host = joined_host_index()
    if host is None or not path:
        return path
    given = Path(path)
    mine = given.parent / '.multihost' / f'host{host}' / given.name
    if mine.with_name(mine.name + '.json').is_file():
        print(f'woof hex: host {host} resumes from its own copy of the checkpoint, {mine} (its shards and the '
              'manifest mirror rank 0 sent at the seal)', file=sys.stderr, flush=True)
        return str(mine)
    return path


def _joined_refusal(path, directory, manifest, why):
    from ..multihost import joined_host_index
    host = joined_host_index()
    return (f'host {host} cannot resume from {path}: {why}. A joined host restores rank by rank from its own shards '
            f'in {directory}; it cannot assemble the globe, whose other shards are on the other hosts. Resume on the '
            'hosts and partitions that sealed it, or on one host from rank 0\'s checkpoint')


def read_shard(directory, entry):
    """One rank's sealed words, read against the hash its manifest holds.  A site may put a reader of a raw
    form over this name; every caller looks it up here at call time, so either form is read."""
    from .moving_v842 import _read_verified
    data = _read_verified(Path(directory) / entry['name'], int(entry['bytes']), entry['sha256'], 'checkpoint shard')
    return pickle.loads(memoryview(data))


# ----------------------------------------------------------------------------------------------- direct restore
# GPUWM_HEX_DIRECT_SHARD_RESTORE (speed_words): found by the audit of 19 Nov 2026, made an engine
# change.  THE COST THIS REMOVES: a same-host resume read every rank's shard on rank 0's process, assembled the whole
# globe (assemble_sharded), built the one-card carrier, restored it into a global pool set, then projected it back onto
# the very partitions that sealed it (project_restored_pools) and pickled each rank's share to disk for the rank to
# read again.  On the 8-card CONUS globe the audit read 396 s of a 417 s restart as that assembly, and the
# parent's memory on a resume reached 255 GiB before the release word let go of it.  With the word, when the resume runs on
# the same number of ranks with the same partitions, the parent reads only the manifest; each rank reads and verifies
# its own shard and restores its own words (moving_mpi_v842.DistributedMoving).  Anything else (another rank count,
# other partitions, a DA analysis, several hosts, waves, a remapped move, one card) assembles as before.

import contextvars
from fractions import Fraction

_DIRECT_RANKS = contextvars.ContextVar('hex_direct_shard_ranks', default=None)


class direct_restore_for:
    """``with direct_restore_for(arguments, route):`` around a carrier read: the ranks a same-host resume will launch,
    when each may restore its own shard (the word on, ranks on this host only, no analysis, no waves, no remap)."""

    def __init__(self, arguments, route=None):
        self.ranks = _eligible_ranks(arguments, route)

    def __enter__(self):
        self._token = _DIRECT_RANKS.set(self.ranks)
        return self

    def __exit__(self, *exc):
        _DIRECT_RANKS.reset(self._token)
        return False


def _eligible_ranks(arguments, route):
    from ..speed_words import DIRECT_SHARD_RESTORE, on
    if not on(DIRECT_SHARD_RESTORE) or not enabled():
        return None
    devices = getattr(arguments, 'samehost_devices', None)
    if not devices and getattr(arguments, 'hosts', None):
        try:
            from ..multihost import parse_hosts
            hosts = parse_hosts(arguments.hosts)
            devices = ','.join(str(card) for host in hosts for card in host.cards)
        except ValueError:
            return None
    if not devices or getattr(arguments, 'da_analysis', None):
        return None
    if str(getattr(arguments, 'waves', 'off') or 'off') != 'off':
        return None     # rank 0 restores the carried wave spectrum from the global carrier (moving_mpi_v842.wave_state)
    if route is not None and getattr(route, 'operator', None):
        return None     # a remapped move maps the global carrier onto a new member
    try:
        from ..drivers.samehost_forecast_v841 import device_list
        return len(device_list(devices))
    except ValueError:
        return None


def restore_sharded(path, manifest):
    """The carrier of a sharded checkpoint for this reader: the direct carrier when the reader is a same-host resume on
    the ranks that sealed it (``direct_restore_for``), else the assembled one-card payload (``assemble_sharded``)."""
    ranks = _DIRECT_RANKS.get()
    from ..multihost import joined_host_index
    joined = joined_host_index() is not None
    if ranks is not None and int(ranks) == int(manifest['ranks']) and len(manifest['shards']) == int(ranks):
        carrier = direct_carrier(path, manifest)
        if joined and not manifest.get('mirror'):
            directory = Path(carrier['direct_shards']['directory'])
            missing = [entry['name'] for entry in manifest['shards'] if not (directory / entry['name']).is_file()]
            if missing:
                raise ValueError(_joined_refusal(path, directory, manifest,
                                                 f'{len(missing)} of its {len(manifest["shards"])} rank shards are '
                                                 f'not on this host ({missing[0]} first) and it has no manifest '
                                                 'mirror (sealed before multi-host mirrors)'))
        return carrier
    if manifest.get('mirror'):
        raise ValueError(_joined_refusal(path, shard_directory(path), manifest,
                                         'this resume does not run on the ranks that sealed it, or '
                                         'GPUWM_HEX_DIRECT_SHARD_RESTORE is 0'))
    return assemble_sharded(path, manifest)


def direct_carrier(path, manifest):
    """What the parent of a same-host resume reads of a sharded checkpoint without assembling it: the clock, the mesh,
    the names, the physics alarms and their clock (as ``MovingRoute.carrier_payload`` builds them from the frozen route
    the manifest holds), and where each rank's shard is.  ``moving_mpi_v842.prepare_moving`` checks the partitions and
    either hands each rank its shard or assembles (``assemble_direct``)."""
    from .moving_v842 import SCHEMA as CARRIER
    route = pickle.loads(manifest['route'])
    layout, step, dt = manifest['layout'], manifest['step'], manifest['dt']
    rows = route.carried_rows(layout)
    elapsed = (Fraction(str(route.physics_clock_offset(dt)[1])) + Fraction(str(step)) * Fraction(str(dt)))
    physics = dict(suite=manifest['suite'], manager=manifest['manager'], manager_elapsed=str(elapsed),
                   fields={f"{r['pool']}.{r['name']}": dict(dimensions=r['dimensions'], nlevels=r['nlevels'],
                                                             kind=r['kind']) for r in rows},
                   pools={}, continuation={})
    return dict(schema=CARRIER, route='mpas-suite', time_seconds=float(manifest['time_seconds']),
                origin_time=manifest['origin_time'], valid_time=manifest['valid_time'], step=int(step), dt=dt,
                mesh=manifest['mesh'], vertical=manifest['vertical'], scalar_names=list(manifest['names']),
                physics=physics, stochastic=None, chemistry=None, waves=None,
                direct_shards=dict(path=str(path),
                                   directory=str(os.environ.get('WOOF_HEX_SHARD_DIR') or shard_directory(path)),
                                   ranks=int(manifest['ranks']),
                                   n_cells=int(manifest['n_cells']), names=list(manifest['names']),
                                   shards=list(manifest['shards']), manifest=manifest))


def direct_matches(direct, layout, names, partitions):
    """Why this resume cannot restore rank by rank (None when it can): the cells, names and every rank's partition
    (owned cells, cell and edge ids) must be the ones that sealed."""
    import numpy as np
    if partitions is None:
        return 'this route does not hand partitions to its moving resume'
    if int(layout.dimensions['nCells']) != int(direct['n_cells']):
        return 'another mesh'
    if list(names) != list(direct['names']):
        return 'other species'
    saved = direct['manifest']['partitions']
    if len(saved) != len(partitions) or len(partitions) != int(direct['ranks']):
        return f'{len(partitions)} ranks against {len(saved)} sealed'
    for rank, (mine, theirs) in enumerate(zip(partitions, saved)):
        if (int(mine.n_owned_cells) != int(theirs.n_owned_cells)
                or getattr(mine, 'n_owned_edges', None) != getattr(theirs, 'n_owned_edges', None)
                or not np.array_equal(mine.cell_l2g, theirs.cell_l2g)
                or not np.array_equal(mine.edge_l2g, theirs.edge_l2g)):
            return f'rank {rank} is cut differently from the seal'
    return None


def assemble_direct(direct):
    """The assembled one-card payload of a direct carrier (the resume could not take its shards rank by rank)."""
    return assemble_sharded(Path(direct['path']), direct['manifest'])


def assemble_sharded(path, manifest):
    """The one-card carrier payload of a sharded checkpoint: shards read and verified in parallel, assembled."""
    from .moving_mpi_v842 import ManagerWords, assemble_snapshot
    directory = shard_directory(path)
    shards = manifest['shards']
    if len(shards) != int(manifest['ranks']):
        raise ValueError('sharded checkpoint does not list one shard per rank')

    def read(entry):
        return read_shard(directory, entry)

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
