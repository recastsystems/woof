"""Owner assembly at MPAS moving boundaries, using the existing carrier.

Remapping remains in moving_v842. This adapter assembles the global carrier
at a frame, and projects its restored words before the next rank launch.
"""
from __future__ import annotations

import os
import pickle
from copy import copy, deepcopy
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from . import moving_v842 as mv
from .partition_v842 import gather_owned_field, project_buffer
from .pool_v842 import NativePoolHydration


class PoolWords:
    """Host pool storage for the existing remapper and carrier writer."""

    def __init__(self, layout, hydration=None):
        self.layout, self.hydration = layout, hydration
        self.data = {}
        #: Keys put_field wrote: the carrier words a moving resume hands to the ranks/classes.
        self.written = set()
        self._fields = {(row['pool'], row['pool_name']):
            (np.dtype('<f4' if row['kind'] == 'real' else '<i4'),
             tuple(row['shape']), row['nlevels'])
            for row in layout.fields if row['kind'] != 'character'}

    def get_field(self, pool, name, *, level=1):
        key = (pool, name, level)
        if key not in self.data:
            if self.hydration is None:
                raise ValueError(f'boundary did not assemble {key}')
            row = next(row for row in self.layout.fields
                       if (row['pool'], row['pool_name']) == (pool, name))
            self.data[key] = self.hydration.field_buffer(pool, row['name']).copy(order='F')
        return self.data[key]

    def put_field(self, pool, name, value, *, level=1, nlevels=1):
        self.data[pool, name, level] = np.array(value, copy=True, order='F')
        self.written.add((pool, name, level))

    def restore_mapped_fields(self, jobs, n_target):
        """Install independent remapped fields without copying discarded cold columns.

        A new move replaces all real columns. Only padding retains the member's
        cold words. Each worker owns a new F-order array; publishing the arrays
        afterwards preserves put_field's private-storage contract.
        """
        from ..host_parallel import run_tasks
        rows = {(row['pool'], row['pool_name']): row for row in self.layout.fields}

        def restore(job):
            row, level, finite, mapped = job
            pool, name = row['pool'], row['pool_name']
            dtype, shape, _ = self._fields[pool, name]
            if finite and not np.isfinite(mapped).all():
                raise ValueError(f'moving physics remap produced a non-finite value in {mv._key(row, level)}')
            out = np.empty(shape, dtype=dtype, order='F')
            out[..., :n_target] = mapped
            if shape[-1] > n_target:
                key = (pool, name, level)
                cold = self.data.get(key)
                if cold is None:
                    cold = self.hydration.field_buffer(pool, rows[pool, name]['name'])
                out[..., n_target:] = cold[..., n_target:]
            return (pool, name, level), out

        # Four workers bound concurrent host traffic beside the atmosphere.
        for key, words in run_tasks([lambda job=job: restore(job) for job in jobs], workers=4):
            self.data[key] = words
            self.written.add(key)

    def written_words(self):
        """Only the words put_field wrote, never a field get_field merely read from the member's own inputs.

        THE BREAKAGE THIS PREVENTS (box-aws2x 2026-10-24, MPAS-route storm-following, member statics from a
        WPS_GEOG with MODIS lakes): restore_physics reads the target's land-use classes from the hydrated
        static, so ``data`` held its raw ivgtyp with class 21. Handed over as carrier words, it overwrote the
        ranks'/classes' lake fold (runtime_v842.fold_lake_class) and every moved segment died at its first
        Noah call: "MPAS Noah native fatal condition at source line 2441" (VEGTYP > LUCATS), three of three
        retries.
        """
        return {key: value for key, value in self.data.items() if key in self.written}


class WindowWords:
    def __init__(self, data=None):
        self.data = data

    def restore(self, data):
        self.data = {key: np.array(value, copy=True) for key, value in data.items()}

    def snapshot(self):
        return self.data


class ServiceWords:
    def __init__(self, data=None):
        self.data = data

    def restore_restart_state(self, data):
        self.data = dict(data)

    def restart_state(self):
        return self.data


class HostArrays:
    """Carrier/frame API over already assembled host arrays."""
    asnumpy = staticmethod(np.asarray)
    abs = staticmethod(np.abs)


def rank_carrier(carrier):
    """The carrier words a rank's route still reads after the global restore and projection (prepare_moving).

    The physics alarms' phase and its clock (restore_alarm_phase), the carried field list (CheckpointRoute
    carried_rows) and a continuation's scheme flags and start hour (restore_continuation); its accumulators reach
    each rank projected onto its cells (project_restored_words).  Never the state or the pools: the global carrier
    stays with the process that restored it.  THE BREAKAGE THIS FIXES: the rank routes held an empty carrier, so a
    resume on same-host ranks had no alarm phase, physics clock, scheme flags or accumulators to continue (the
    one-card continuation, fix/hex-rrfs-restart-exact daa3e531), and a resumed rank run that sealed again read
    ``carrier['physics']`` from an empty dict (CheckpointRoute.carried_rows).
    """
    physics = carrier['physics']
    kept = {key: physics[key] for key in ('fields', 'manager', 'manager_elapsed') if key in physics}
    continuation = physics.get('continuation')
    if continuation is not None:
        kept['continuation'] = {key: value for key, value in continuation.items() if key != 'output_accumulators'}
    return dict(physics=kept)


def prepare_moving(route, inputs, config, layout, payload, arguments):
    mesh, grid, profiles, state, reference, saved, names = inputs
    bind_inputs = getattr(route, 'bind_inputs', None)
    if bind_inputs is not None:
        bind_inputs(inputs, config)
    route.begin(layout.configs['config_start_time'] if route.carrier is None else None)
    native = PoolWords(layout, NativePoolHydration(None, layout, payload))
    session = SimpleNamespace(native=native, config=config, precipitation_window=None,
                              surface_service=None)
    restored = None
    if route.carrier is not None:
        target_grid = getattr(arguments, 'grid', None) or arguments.init
        physics = route.carrier['physics']
        if physics.get('window') is not None:
            session.precipitation_window = WindowWords()
        if physics.get('surface_service') is not None:
            session.surface_service = ServiceWords()
        early = getattr(route, 'move_weights', None)
        if os.environ.get('WOOF_HEX_MOVING_PHYSICS_BESIDE', '1') == '0':
            early = None   # the comparison knob: the physics restore after the atmosphere remap, as before
        weights = None if early is None else early(mesh=mesh, grid=grid, target_grid=target_grid)
        if weights is None:
            state, saved, weights = route.resume_dycore(mesh=mesh, grid=grid,
                reference=reference, state=state, saved=saved, names=names, target_grid=target_grid)
            restored = route.restore_physics(session, layout, weights, mesh)
        else:
            # The physics restore reads only the carrier's physics, this member's pools and the map (never the
            # remapped atmosphere, and none of the map's per-remap words: the below-ground extension and the
            # domain receipts belong to the density-weighted products alone), so it runs beside the atmosphere
            # remap.  THE COST THIS REMOVES: 2.5-2.7 s of every storm-following move on a 90,320-cell region
            # (RTX 5090, 2026-10-29), after the 8 s atmosphere remap.  Each is the same call on the same
            # inputs, so every carried word is unchanged.
            from ..host_parallel import run_tasks
            (state, saved, weights), restored = run_tasks([
                lambda: route.resume_dycore(mesh=mesh, grid=grid, reference=reference, state=state, saved=saved,
                                            names=names, target_grid=target_grid, weights=weights),
                lambda: route.restore_physics(session, layout, weights, mesh)])
            if getattr(route, 'remap_receipt', None) is not None:
                route.remap_receipt['physics'] = restored
    result = (mesh, grid, profiles, state, reference, saved, names)
    # MovingControl holds a thread lock. Recreate it in each worker rather
    # than pickling that lock, or retaining the complete source carrier.
    control = (None if route.control is None else
        dict(directory=route.control.directory, seal_every_seconds=route.control.seal_every_seconds))
    worker_route = copy(route)
    worker_route.control = None
    worker_route.carrier = None if route.carrier is None else rank_carrier(route.carrier)
    continuation = None if route.carrier is None else route.carrier['physics'].get('continuation')
    return result, dict(route=worker_route, control=control, layout=layout, mesh=mesh, vertical=grid.vertical_grid,
        restored=native.written_words() if route.carrier is not None else {},
        restore_receipt=restored,
        accumulators=None if continuation is None else continuation.get('output_accumulators'),
        window=None if session.precipitation_window is None else session.precipitation_window.snapshot(),
        surface_service=None if session.surface_service is None else session.surface_service.restart_state())


def project_restored_words(moving, layout, partitions, *, workers=None, gpu=False):
    """The restored carrier words of every partition, in memory, every field on its own worker.

    THE COST THIS REMOVES: the MPAS LTS class executor projected the restored pools class after class, field
    after field (each call rebuilding the partition's sorted id maps), then pickled them to disk for the class
    to read back: 16 s of a resumed storm-following segment's preparation (MPAS-ROUTE-SPEED-2026-10-24).
    """
    from concurrent.futures import ThreadPoolExecutor
    from .partition_v842 import _maps, _project_buffer
    by_key = {(r['pool'], r['pool_name']): r for r in layout.fields}
    maps = [_maps(layout, partition) for partition in partitions]
    keys = list(moving['restored'])
    device_projection = None
    began = time.perf_counter()
    if (gpu and os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            and os.environ.get('WOOF_HEX_MOVING_CLASS_WORDS_GPU', '1') != '0'):
        from .. import moving_pool_projection_cuda as device_projection
        device_maps = device_projection.prepare(maps, layout)

    def project(key):
        if device_projection is not None:
            return device_projection.project(moving['restored'][key], by_key[key[:2]], layout, maps, device_maps)
        return [_project_buffer(moving['restored'][key], by_key[key[:2]], layout, mapping) for mapping in maps]

    with ThreadPoolExecutor(max_workers=max(1, min(len(keys), workers or
                            (4 if device_projection is not None else os.cpu_count()) or 1))) as pool:
        projected = list(pool.map(project, keys))
    if device_projection is not None and isinstance(moving.get('restore_receipt'), dict):
        moving['restore_receipt']['class_word_projection'] = dict(
            backend='CUDA uint32 gather', seconds=time.perf_counter()-began,
            source_bytes=sum(v.nbytes for v in moving['restored'].values()))
    window = moving['window']
    accumulators = moving.get('accumulators')
    result = []
    for which, partition in enumerate(partitions):
        cells = partition.cell_l2g[:partition.n_owned_cells]
        owned = None if window is None else {
            key: np.ascontiguousarray(value[cells]) for key, value in window.items()}
        # A continuation's running accumulators, owned cells only (DistributedMoving.restore_continuation).
        sums = None if not accumulators else {
            name: np.ascontiguousarray(np.asarray(value)[..., cells]) for name, value in accumulators.items()}
        result.append(dict(pools={key: words[which] for key, words in zip(keys, projected)}, window=owned,
                           service=moving['surface_service'], accumulators=sums))
    return result


def project_restored_pools(moving, layout, partition, local_layout, payload):
    restored = project_restored_words(moving, layout, [partition])[0]
    with (Path(payload) / 'moving-restore.pickle').open('wb') as handle:
        pickle.dump(restored, handle, protocol=5)


def apply_restored(session, restored):
    """Write a rank's projected carrier words into its session (DistributedMoving.restore_physics)."""
    put = session.native.put_field
    if os.environ.get('WOOF_HEX_MOVING_RESIDENT') == '1':
        put = getattr(session.native,'adopt_restored_field',put)
    for (pool, name, level), value in restored['pools'].items():
        put(pool, name, value, level=level,nlevels=session.native._fields[pool,name][2])
    window = session.precipitation_window
    if (window is None) != (restored['window'] is None):
        raise ValueError('moving precipitation window differs on native rank')
    if window is not None:
        window.restore(restored['window'])
    if session.surface_service is not None and restored['service'] is not None:
        session.surface_service.restore_restart_state(restored['service'])


def assemble_columns(buffers, partitions, *, entity):
    """Assemble a CUDA carrier field without ever consulting a halo value."""
    if not partitions or len(buffers) != len(partitions):
        raise ValueError('moving assembly needs one buffer per rank')
    ids_name, owned_name, total_name = {
        'cell': ('cell_l2g', 'n_owned_cells', '_global_cells'),
        'edge': ('edge_l2g', 'n_owned_edges', '_global_edges'),
    }[entity]
    first = np.asarray(buffers[0])
    total = getattr(partitions[0], total_name)
    result = np.empty((*first.shape[:-1], total), dtype=first.dtype)
    seen = np.zeros(total, dtype=bool)
    for buffer, partition in zip(buffers, partitions):
        ids, owned = getattr(partition, ids_name), getattr(partition, owned_name)
        if (buffer.shape[:-1] != first.shape[:-1] or buffer.dtype != first.dtype or
                buffer.shape[-1] not in (owned, len(ids), len(ids) + 1)):
            raise ValueError('moving rank field shape/dtype differs')
        ids = ids[:owned]
        if np.any(seen[ids]):
            raise ValueError('moving rank field has duplicate owners')
        result[..., ids] = buffer[..., :owned]
        seen[ids] = True
    if not np.all(seen):
        raise ValueError('moving rank field lacks global coverage')
    return result


def snapshot_wanted(route, layout):
    """The pool fields a moving frame, verdict, seal or carrier export reads (Registry (pool, name) pairs)."""
    wanted = {(r['pool'], r['name']) for r in route.carried_rows(layout)}
    wanted.update(('sfc_input', name) for name in (*mv.CATEGORY_FIELDS, 'isice_lu'))
    wanted.update(('diag', name) for name in
                  ('pressure', 'theta', 'uReconstructZonal', 'uReconstructMeridional'))
    wanted.update(('diag_physics', name) for name in ('u10', 'v10'))
    # The tracker frame's setup-target fields (moving_v842.write_tracker_frame: qv, refl10cm_max, t2m, q2).
    # THE BREAKAGE THIS PREVENTS: an assembled frame without them raised "boundary did not assemble" on the
    # same-host ranks and the LTS classes once the tracker frame started writing them (rel/hex-0.4.5 f2c58ed0).
    wanted.update(('diag_physics', name) for name in ('refl10cm_max', 't2m', 'q2'))
    wanted.add(('state', 'scalars'))
    return wanted


def pool_keys(layout, wanted):
    """The (row, level) pairs of the wanted pool fields, in Registry order (the order every rank sends them)."""
    return [(row, level) for row in layout.fields if (row['pool'], row['name']) in wanted
            for level in range(1, row['nlevels'] + 1)]


def assemble_pool_words(layout, partitions, keys, gathered, *, maps=None):
    """Every rank's buffers of ``keys`` (``gathered[rank][index]``) as one card's PoolWords, on all cores.

    Carrier ABI: native dummy storage appended.
    """
    from concurrent.futures import ThreadPoolExecutor
    from .partition_v842 import partition_maps
    maps = partition_maps(layout, partitions) if maps is None else maps

    def assemble(index):
        row, _ = keys[index]
        words = gather_owned_field(row, layout, partitions, [ranks[index] for ranks in gathered], maps=maps)
        # Carrier ABI includes native dummy storage. Its value is disposable and no solved column reads it.
        if row['dimensions'] and row['dimensions'][-1] in ('nCells', 'nEdges', 'nVertices'):
            words = np.concatenate((words, np.zeros((*words.shape[:-1], 1), words.dtype)), axis=-1)
        return words

    native = PoolWords(layout)
    with ThreadPoolExecutor(max_workers=max(1, min(len(keys), os.cpu_count() or 1))) as pool:
        for (row, level), words in zip(keys, pool.map(assemble, range(len(keys)))):
            native.data[row['pool'], row['pool_name'], level] = np.asfortranarray(words)
    return native


def gather_pool_words(comm, layout, partitions, wanted, local_field, *, maps=None):
    """Every rank's owned words of the wanted pool fields in ONE collective; rank 0 assembles them on all cores.

    ``local_field(row, level)`` is this rank's buffer as gather_owned_field takes it.  Returns the assembled
    PoolWords on rank 0 (carrier ABI: native dummy storage appended), None elsewhere.
    THE BREAKAGE THIS PREVENTS: the frame and the carrier export gathered field by field and level by level (two
    barriers each) and rebuilt every partition's sorted id maps per call, about 11 s per frame on the
    storm-following 3 km region (23 s on a free-threaded host), three times a moving segment on the MPAS route
    against nothing on the WOOF route (MPAS-ROUTE-SPEED-2026-10-24).  The words written are the same.
    """
    keys = pool_keys(layout, wanted)
    gathered = comm.gather([local_field(row, level) for row, level in keys], root=0)
    if comm.Get_rank() != 0:
        return None
    return assemble_pool_words(layout, partitions, keys, gathered, maps=maps)


def assemble_continuation(ranks, partitions):
    """Every rank's continuation words (checkpoint_v842.rank_continuation) as one card's: rank 0's scheme flags and
    start hour (every rank calls every scheme on the same clock) and the running accumulators over every rank's
    owned cells, in the one-card carrier layout (native dummy cell appended), so either kind resumes the other."""
    first = ranks[0]
    accumulators = None
    if first['output_accumulators'] is not None:
        accumulators = {}
        for name in first['output_accumulators']:
            words = assemble_columns([rank['output_accumulators'][name] for rank in ranks], partitions,
                                     entity='cell')
            accumulators[name] = np.concatenate((words, np.zeros((*words.shape[:-1], 1), words.dtype)), axis=-1)
    return dict(first, output_accumulators=accumulators)


class ManagerWords:
    """A sealed alarm manager's restart words, where the snapshot proxy reads a live manager's (_manager_state)."""

    def __init__(self, state):
        self.state = state

    def restart_state(self):
        return self.state


def rank_words(route, layout, session, dycore, columns=None, *, continuation=False):
    """This rank's words of a snapshot: exactly what DistributedMoving gathers to rank 0, and what a rank seals in
    its own checkpoint shard (checkpoint_shards_v842).  Every array is a fresh host copy (the native pools'
    get_field, the device downloads), so a writer thread may hold it while the step loop runs on."""
    keys = pool_keys(layout, snapshot_wanted(route, layout))
    window = session.precipitation_window
    service = session.surface_service
    if continuation:
        from .checkpoint_v842 import rank_continuation
    return dict(
        keys=[(row['pool'], row['name'], level) for row, level in keys],
        pools=[session.native.get_field(row['pool'], row['pool_name'], level=level) for row, level in keys],
        state={name: dycore.cp.asnumpy(getattr(dycore.atmosphere.state, name)) for name in mv.STATE_FIELDS},
        saved={name: dycore.cp.asnumpy(getattr(dycore.atmosphere.saved, name)) for name in mv.SAVED_FIELDS},
        window=None if window is None else window.snapshot(),
        service=None if service is None else deepcopy(service.restart_state()),
        psfc=None if columns is None else dycore.cp.asnumpy(columns.psfc_p),
        continuation=rank_continuation(session) if continuation else None)


def assemble_snapshot(pieces, layout, partitions, *, config, manager, maps=None):
    """Every rank's ``rank_words`` as one card's (session, dycore, columns) proxies, which the route's frame, seal
    and carrier export read (``MovingRoute.carrier_payload``).  Halo words are never consulted."""
    keys = pieces[0]['keys']
    if any(piece['keys'] != keys for piece in pieces[1:]):
        raise ValueError('ranks sealed different pool fields')
    rows = {(row['pool'], row['name']): row for row in layout.fields}
    native = assemble_pool_words(layout, partitions, [(rows[pool, name], level) for pool, name, level in keys],
                                 [piece['pools'] for piece in pieces], maps=maps)
    state, saved = {}, {}
    for target, names, kind in ((state, mv.STATE_FIELDS, 'state'), (saved, mv.SAVED_FIELDS, 'saved')):
        for name in names:
            target[name] = assemble_columns([piece[kind][name] for piece in pieces], partitions,
                entity='edge' if name in ('rho_u', 'normal_velocity') else 'cell')
    services = [piece['service'] for piece in pieces]
    if any(service != services[0] for service in services):
        raise ValueError('moving surface service calendars differ between ranks')
    windows = [piece['window'] for piece in pieces]
    global_window = None
    if windows[0] is not None:
        global_window = {key: assemble_columns([window[key] for window in windows],
            partitions, entity='cell') for key in windows[0]}
    continuations = [piece['continuation'] for piece in pieces]
    # Rank 0's physics alarms: their phase is in model seconds, the same on every rank.  THE BREAKAGE THIS
    # FIXES: the proxy had no driver, so every carrier sealed on same-host ranks held no alarm phase.
    proxy = SimpleNamespace(native=native, config=config,
        driver=SimpleNamespace(manager=manager),
        precipitation_window=None if global_window is None else WindowWords(global_window),
        surface_service=None if services[0] is None else ServiceWords(services[0]),
        continuation=None if continuations[0] is None else assemble_continuation(continuations, partitions))
    global_dycore = SimpleNamespace(cp=HostArrays, atmosphere=SimpleNamespace(
        state=SimpleNamespace(**state), saved=SimpleNamespace(**saved)))
    global_columns = None if pieces[0]['psfc'] is None else SimpleNamespace(
        psfc_p=assemble_columns([piece['psfc'] for piece in pieces], partitions, entity='cell'))
    return proxy, global_dycore, global_columns


class DistributedMoving:
    """All ranks enter a frame; rank zero alone runs the moving controller."""

    def __init__(self, comm, prepared, partitions, execution):
        self.comm, self.prepared, self.partitions = comm, prepared, partitions
        self.execution = execution
        self.route = prepared['route']
        if prepared['control'] is not None:
            from ..moving_control import MovingControl
            self.route.control = MovingControl(**prepared['control'])
        self.carrier = self.route.carrier
        # A checkpoint, restart or DA analysis on the run's own mesh (checkpoint_v842 CheckpointRoute.fixed_mesh)
        # moves nothing, on every rank as on one card.  THE BREAKAGE THIS FIXES: this class did not carry the flag,
        # so runtime_v842 read every same-host rank as a moving run and refused a limited-area run ('takes the plain
        # regional dycore: no moving wrapper ...'): NOAA's conus3km on 8 cards could not --checkpoint-hours or
        # --restart-from at all (V8XB, venv RB bec7ce68, 2026-10-29 19:04Z), while one card and global meshes could.
        self.fixed_mesh = bool(getattr(self.route, 'fixed_mesh', False))
        self._receipt = self.route.receipt()

    def begin(self, start):
        self.route.begin(start)

    def resume_dycore(self, **values):
        # Global remapping ran once before rank projection and CUDA upload.
        return values['state'], values['saved'], None

    def restore_physics(self, session, layout, weights, mesh):
        with (session.hydration.payload / 'moving-restore.pickle').open('rb') as handle:
            restored = pickle.load(handle)
        apply_restored(session, restored)
        self._accumulators = restored.get('accumulators')
        return self.prepared['restore_receipt']

    # The rest of the one-card resume, which runtime_v842 calls after the physics init, on this rank's session.
    # THE BREAKAGE THIS FIXES: this class had none of them, so runtime_v842's restore_alarm_phase call failed every
    # resume on same-host ranks, and a correct-mode continuation (fix/hex-rrfs-restart-exact daa3e531) would have
    # restarted the physics step counter, the start hour, the scheme flags and the accumulators on several cards:
    # RRFS v2 cycling on conus3km is multi-card, and every DA analysis restart goes through here.
    def restore_alarm_phase(self, session, config):
        """The carried alarms' phase on this rank (its model seconds are every rank's)."""
        return self.route.restore_alarm_phase(session, config)

    def physics_clock_offset(self, dt):
        return self.route.physics_clock_offset(dt)

    def checkpoint_due(self, step, dt):
        """Expose the seal boundary to runtime_v842.moving_publication_due.

        Without this hook a checkpoint between stream frames deferred that step's diagnostic update, then
        sealed the accumulators before the next synchronize folded it. A continuation skips its initial
        update because the checkpoint instant is already included, so that sample was lost (lam7k hrrrv5,
        two ranks, correct mode: 49,303 diagnostic words at 13Z after a 12:30 checkpoint).
        """
        return getattr(self.route, 'checkpoint_due', lambda step, dt: False)(step, dt)

    def restore_continuation(self, *, session, diagnostics, dt):
        """The route's continuation restore, with the carried accumulators on this rank's owned cells.

        Halo words keep this run's own values: the accumulator update writes solve cells only and the writer
        gathers owned cells only, so no halo accumulator word is read.
        """
        restore = getattr(self.route, 'restore_continuation', None)
        if restore is None:
            return None
        owned, self._accumulators = getattr(self, '_accumulators', None), None
        local = None
        if owned and diagnostics is not None:
            count = int(self.partitions[self.comm.Get_rank()].n_owned_cells)
            local = {}
            for name, value in owned.items():
                buffer = diagnostics.buffers.get(('diag', name))
                if buffer is None:
                    continue
                words = np.array(diagnostics.cp.asnumpy(buffer), copy=True)
                words[..., :count] = value
                local[name] = words
        return restore(session=session, diagnostics=diagnostics, dt=dt, accumulators=local)

    def _snapshot(self, session, dycore, columns=None, *, continuation=False):
        layout = self.prepared['layout']
        pieces = self.comm.gather(rank_words(self.route, layout, session, dycore, columns,
                                             continuation=continuation), root=0)
        if self.comm.Get_rank() != 0:
            return None
        return assemble_snapshot(pieces, layout, self.partitions, config=session.config,
                                 manager=getattr(getattr(session, 'driver', None), 'manager', None))

    def after_step(self, step, **values):
        model_seconds = self.route.offset_seconds + step * values['dt']
        frame = self.route.control is not None and not model_seconds % self.route.frame_seconds
        due = getattr(self.route, 'checkpoint_due', lambda step, dt: False)(step, values['dt'])
        if not frame and not self.route.shock and not due:
            return False
        if due:
            # As CheckpointRoute.after_step on one card: the step's publication ran before its output event, whose
            # resets (refl10cm_1km_max, the hail maxima) changed the device words since.  Every rank seals what its
            # next step reads.
            published = time.perf_counter()
            publish = getattr(values['session'], 'publish', None)
            if callable(publish):
                publish()
            published = time.perf_counter() - published
            if not frame and not self.route.shock and hasattr(self.route, 'checkpoint_target'):
                from .checkpoint_shards_v842 import enabled
                if enabled():
                    self._seal_shards(step, values, publish_seconds=published)
                    result, self._receipt = self.comm.bcast((False, self.route.receipt()), root=0)
                    return result
        # The prepared columns (psfc_p) only where the route reads them: a frame or the shock series, the steps
        # runtime_v842 prepares them on (moving_columns_due).  THE BREAKAGE THIS FIXES: at a checkpoint-only step
        # the columns are the resident wind reconstruction, which has no psfc_p, so every same-host rank seal on the
        # RRFS route died with "'SimpleNamespace' object has no attribute 'psfc_p'" (HB18 774efbcb, hrrrv5
        # x1.2562, 2 rank processes, --checkpoint-hours 1).
        snapshot = self._snapshot(values['session'], values['dycore'],
                                  values['columns'] if frame or self.route.shock else None,
                                  continuation=due and hasattr(self.route, 'restore_continuation'))
        result = False
        if self.comm.Get_rank() == 0:
            session, dycore, columns = snapshot
            result = self.route.after_step(step, **{**values, 'session': session,
                'dycore': dycore, 'columns': columns, 'mesh': self.prepared['mesh'],
                'vertical': self.prepared['vertical'], 'layout': self.prepared['layout']})
        result, self._receipt = self.comm.bcast((result, self.route.receipt()), root=0)
        return result

    def _seal_shards(self, step, values, *, publish_seconds=0.0):
        """This rank's shard of a due checkpoint, sealed beside the step loop (checkpoint_shards_v842).

        On the step path: the previous seal's end (normally long done), one tiny collective for the seal's token,
        and this rank's own words (fresh host copies); the pickling, SHA256, fsync and rank 0's manifest run on the
        rank's writer thread while the next steps run.
        """
        from . import checkpoint_shards_v842 as shards
        began = time.perf_counter()
        rank, size = self.comm.Get_rank(), self.comm.Get_size()
        writer = getattr(self, '_shards', None)
        if writer is None:
            writer = self._shards = shards.ShardWriter(rank, size,
                                                       aborted=lambda: getattr(self.comm, 'aborted', None))
        writer.wait()
        route, layout = self.route, self.prepared['layout']
        token = self.comm.bcast(os.urandom(16).hex() if rank == 0 else None, root=0)
        target = route.checkpoint_target(step, values['dt'], values['out'])
        words = rank_words(route, layout, values['session'], values['dycore'],
                           continuation=hasattr(route, 'restore_continuation'))
        manifest = on_sealed = None
        if rank == 0:
            manifest = shards.manifest(route=route, session=values['session'], layout=layout,
                partitions=self.partitions, mesh=self.prepared['mesh'], vertical=self.prepared['vertical'],
                names=values['names'], step=step, dt=values['dt'], token=token,
                metadata=route.checkpoint_metadata(route.background_at(step)))
            index = len(route.checkpoints)
            route.checkpoints.append(dict(path=str(target), step=int(step),
                                          sealing='rank shards on writer threads beside the step loop'))

            def on_sealed(receipt, index=index):
                route.checkpoints[index] = receipt
        writer.submit(target, token, words, step_seconds=time.perf_counter() - began,
                      publish_seconds=publish_seconds, manifest=manifest, on_sealed=on_sealed)

    def finish(self, **values):
        writer = getattr(self, '_shards', None)
        if writer is not None:
            # Every rank's last shard on disk, and rank 0's manifest and receipt after it, before the run reports.
            writer.wait()
            self._receipt = self.comm.bcast(self.route.receipt() if self.comm.Get_rank() == 0 else None, root=0)
        if not self.route.export_path:
            return
        snapshot = self._snapshot(values['session'], values['dycore'],
                                  continuation=hasattr(self.route, 'restore_continuation'))
        if self.comm.Get_rank() == 0:
            session, dycore, _ = snapshot
            self.route.finish(**{**values, 'session': session, 'dycore': dycore,
                'mesh': self.prepared['mesh'], 'vertical': self.prepared['vertical'],
                'layout': self.prepared['layout']})
        self._receipt = self.comm.bcast(self.route.receipt(), root=0)

    def receipt(self):
        return self._receipt
