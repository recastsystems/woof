"""Rust output for the MPAS physics route, without a Fortran stream manager.

MPAS v8.4.2 mpas_atm_core.F:663-705,877-902 owns alarm/reset order;
mpas_io_streams.F:1659-1857 owns constituent and owned-entity selection.
"""
import json
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import numpy as np

from ..output import HistoryField
from ..restart_v842 import NativePoolStreams, _date, inspect_stream
from ..streams_v842 import Stream, StreamFieldCache
from .stream_metadata_v842 import global_attributes, input_field_attributes, NativeFileIds


def output_inventory(streams, layout, *, produced=()):
    """Retain the route's computed field inventory, including scalar aliases.

    MPAS Registry.xml:650-925,1360-4020. Allocated diagnostics without a
    computed authority are not made valid by changing their writer.
    """
    active = {r['name']: r for r in layout.fields}
    active.update({c: r for r in layout.fields for c in r['components']})
    diagnostics = {'rho', 'theta', 'pressure', 'pressure_base', 'pressure_p',
                   'exner', 'exner_base', 'rho_base', 'rho_p', 'rtheta_base',
                   'rtheta_p', 'uReconstructZonal', 'uReconstructMeridional',
                   # The microphysics callbacks already own these words;
                   # thompson_v842.py:4778-4781,4835-4843 binds the native
                   # interface.F:948-962 and microphysics.F:798-850 writes.
                   'surface_pressure', 'relhum'}
    diagnostics.update(produced)
    result = {}
    for stream in streams:
        result[stream.name] = [name for name in stream.fields
            if name not in active or
            (active[name]['pool'] == 'diag' and name not in diagnostics) or
            (active[name]['pool'] == 'tend' and name not in
             ('rt_diabatic_tend', 'tend_sfc_pressure'))]
    return result


def output_definitions(streams, layout, unavailable, *, stream_type=Stream):
    """Expand the already parsed door lists without changing their order.

    ``stream_type`` is the route's contract ``Stream`` (route_table.StreamContract.registry).
    """
    arrays = {r['name']: tuple(r['components']) for r in layout.fields if r['components']}
    definitions = {}
    for stream in streams:
        options = dict(getattr(stream, 'options', None) or {})
        # MPAS mpas_io.F:419-422 omits fformat when creating SMIOL files;
        # smiolf.F90:429 consequently selects CDF5, including default streams.
        options.setdefault('io_type', 'pnetcdf,cdf5')
        options.update(type=';'.join(stream.directions),
                       filename_template=stream.filename,
                       input_interval=stream.input_interval,
                       output_interval=stream.output_interval,
                       filename_interval=stream.filename_interval)
        members = []
        for name in stream.fields:
            if name not in unavailable.get(stream.name, ()):
                members.extend(('var', n, '') for n in arrays.get(name, (name,)))
        definitions[stream.name] = stream_type(stream.name, options, tuple(members))
    return definitions


def _one_frame_per_file(stream):
    """Whether every output time of ``stream`` opens a file of its own (MPAS mpas_stream_manager.F:4136-4201)."""
    output = stream.options.get('output_interval', 'none')
    if output in ('none', 'initial_only', 'final_only'):
        return output == 'initial_only'
    bucket = stream.options.get('filename_interval')
    return bucket in (None, 'output_interval', output)


def _ready_now():
    return None


def _wire_dtype(dtype):
    """The integer view that moves a field's words untouched (None: move the dtype itself)."""
    dtype = np.dtype(dtype)
    return {4: np.uint32, 8: np.uint64}.get(dtype.itemsize) if dtype.kind == 'f' else None


class _FailedStartReflectivity:
    """The f000 reflectivity could not start: the frame keeps its placeholder and the receipt says why."""

    def __init__(self, error):
        self.error = error

    def apply(self, snapshot, field_type):
        return {'applied': False, 'reason': f'start reflectivity not formed: {self.error!r}'}


class PhysicsStreams(NativePoolStreams):
    """Snapshot selected pool words, then encode away from the forecast thread.

    One outstanding frame bounds memory. A single worker preserves append
    order and uses independent Rust processes for NetCDF, so no libnetcdf
    thread-safety assumption crosses into the CUDA driver. Errors propagate
    before the next capture and on close. Restart timestamps follow durable
    file publication. No live device/host pool view reaches the worker.
    """

    def __init__(self, session, streams, output, *, final_time, alarms=None, diagnostics=None, contract=None):
        """``contract``: the stream contract to write with; None takes the session's route row's
        (route_table.stream_contract of ``session.config``; MPAS-A v8.4.2's when it names none)."""
        from .route_table import stream_contract
        contract = contract if contract is not None else stream_contract(getattr(session, 'config', None))
        self.session = session
        # Correct mode sets this (runtime_v842): the initial frame's surface-layer
        # diagnostics come from the first physics call (woof.hex.f000_first_call);
        # exact mode writes native's.
        # The held f000 frame's reflectivity, formed from its start state beside step 1 (tempo_refl_v303), and
        # the side thread that compiles its module from the moment the frame will be held.
        self._f000_reflectivity = None
        self._f000_warm = None
        self.first_frame_from_first_call = False
        self._held_first = None
        self.first_frame_receipt = None
        self.diagnostics = diagnostics
        produced = () if diagnostics is None else diagnostics.produced
        self.missing = output_inventory(streams, session.layout, produced=produced)
        # Native only registers fields allocated by active Registry packages.
        # An allocated field requested by an enabled output stream must also
        # have an actual writer. Dropping it would silently change the deck.
        active = {r['name'] for r in session.layout.fields}
        active.update(c for r in session.layout.fields for c in r['components'])
        unsupported = {s.name: [name for name in self.missing[s.name] if name in active]
                       for s in streams if 'output' in s.directions and s.output_interval != 'none'}
        unsupported = {name: fields for name, fields in unsupported.items() if fields}
        if unsupported:
            fields = '; '.join(f'{name}: {", ".join(fields)}' for name, fields in unsupported.items())
            raise ValueError(f'MPAS output fields lack computed authority: {fields}; '
                             'cannot omit active fields from an enabled output stream')
        definitions = output_definitions(streams, session.layout, self.missing,
                                         stream_type=contract.registry.Stream)
        super().__init__(session.native, session.layout, None, output,
                         stream_definitions=definitions, final_time=final_time,
                         alarms=alarms, contract=contract)
        self.file_ids = NativeFileIds()
        self._file_attributes = {}
        self.metadata_receipt = dict(file_id_seed=self.file_ids.seed,
                                     file_id_algorithm='GNU/Linux rand, MPAS random_id.c')
        source = next((s.input_file for s in streams if s.name == 'input'), None)
        if source is not None:
            header = inspect_stream(source, scratch=self.output/'stream-scratch')
            pinned = {} if contract.git_version is None else {'git_version': contract.git_version}
            self.attrs = global_attributes(session.layout, header,
                configs=getattr(session, 'output_config_snapshot', None), **pinned)
            self.field_attrs = input_field_attributes(session.layout, header,
                field_definition=contract.registry.field_definition)
        # mpas_atm_core.F:264-277: restarts retain the original CF epoch.
        if 'initial_time' in self.rows:
            row, _ = self.rows['initial_time']
            self.initial = _date(self.native.get_text_field(row['pool'], row['pool_name'], level=1))
            if alarms is None:
                self.alarms = contract.alarms(self.streams, self.start, self.initial, self.final,
                                              self.control, self.layout.packages)
        # Frames encode side by side, each in its own rw_mpas_stream process (see ENCODE_WORKERS).
        self._executor = ThreadPoolExecutor(max_workers=self.ENCODE_WORKERS, thread_name_prefix='mpas-output')
        # The previous frame's completion (its files written and published); see _submit_encode.
        self._last_done = threading.Event()
        self._last_done.set()
        self._publish_after = {}
        # The forecast supplies a factory that snapshots this frame's deferred
        # device checks. Its returned validator runs before any file is encoded.
        self.output_validator = None
        # Download wait and owner assembly (several ranks) run on their own
        # worker, so frame N+1 is assembled while frame N encodes.
        self._assembler = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mpas-output-assemble')
        # Frames in flight, oldest first: (encode future, receipt, host slot).
        self._inflight = deque()
        self._closed = False
        # Pinned writer-owned host storage, one set per frame in flight. A
        # second set is allocated only when a frame arrives while the previous
        # one is still being written (FRAMES_IN_FLIGHT bounds memory).
        self._host_sets = [{}]
        self._host_storage = self._host_sets[0]
        # Asynchronous frame capture (_capture): the writer thread waits for a
        # frame's download, never the forecast thread.
        self._copy_stream = None
        self.capture_receipt = dict(mode='asynchronous download, waited on by the writer thread',
                                    staged_frames=0, in_stream_frames=0, staging_bytes_max=0)
        if diagnostics is not None:
            diagnostics.bind(self)

    def current_time(self, *, step, dt):
        """Use the ported clock's committed time at the stream boundary."""
        clock = getattr(self.native, 'clock_state', None)
        if clock is not None:
            return _date(clock.current_time.stamp())
        return self.start + timedelta(seconds=step * dt)

    def required(self, when):
        """MPAS mpas_atm_core.F:675-701: poll without resetting an alarm.

        Running maxima no longer force a publication every step: on a step
        without a due stream the forecast defers the update to the next
        step's outer-physics synchronize of the same state (see
        ResidentOutputDiagnostics.defer), which costs no host round trip.
        """
        return any('output' in s.options['type'].split(';') and
                   self.stream_fields(s) and
                   self.alarms.due(name, when) for name, s in self.streams.items())

    def stream_fields(self, stream):
        """``stream.fields(self.layout.packages, streams=self.streams)``, expanded once (StreamFieldCache)."""
        cache = getattr(self, '_stream_field_cache', None)
        if cache is None:
            cache = self._stream_field_cache = StreamFieldCache()
        return cache.fields(self.streams, self.layout.packages, stream)

    def _snapshot(self, selected):
        """The frame's host words, complete on return (the synchronous contract).

        Subclasses that override this (the MPAS LTS class executor) keep the
        forecast-thread capture; ``write`` sends everyone else through
        ``_capture``, with the download wait and ``_assemble`` on the writer thread.
        """
        local, ready = self._capture(selected)
        ready()
        return self._assemble(local)

    def _assemble(self, local):
        """The written frame from this process's captured words (one card: unchanged)."""
        return local

    def _capture(self, selected):
        """Capture the frame's words without waiting for their download.

        Returns ``(snapshot, ready)``: ``ready()`` blocks until every host array
        in ``snapshot`` holds its words. Resident buffers move as integer words
        (every floating word kept, NaN payloads included) into pinned
        writer-owned host storage, queued on the device and waited on by
        ``ready``:

        * staged (the card has room): one device copy of the frame on the step
          stream into a staging block held only until the download lands, then
          the download on a separate copy stream ordered after it by an event,
          so the next steps' kernels overlap the download;
        * in stream (no room for the frame twice): the download queued on the
          step stream itself, so later kernels wait for it on the device, never
          on the host, and no frame-sized block is held.

        Either way the forecast may mutate its pools as soon as this returns.
        THE BREAKAGE THIS PREVENTS: the blocking pinned download of every frame
        (1.6 GB of history plus diag at 60 km) held the step thread 0.31 s per
        frame on one card and about 0.6 s on two (MPAS-ROUTE-SPEED-2026-10-24),
        where the WOOF route captures in the background.
        Never publish resident values through native.put_field.
        """
        storage, result, queued = {}, {}, []
        views = getattr(self.session, 'device_pools', None)
        resident = views is not None and getattr(views, 'resident', False)
        cp = getattr(views, 'cp', None) if views is not None else None
        # A queued download needs a real device array (a CPU stand-in has no streams).
        device_array = getattr(cp, 'ndarray', None)
        for name in dict.fromkeys(n for s in selected for n in self.stream_fields(s)):
            if name in ('xtime', 'initial_time'):
                continue
            row, component = self.rows[name]
            pool, field = row['pool'], row['pool_name']
            if row['kind'] == 'character':
                result[name] = self.native.get_text_field(pool, field, level=1)
                continue
            key = pool, field
            dims = tuple(row['dimensions'])
            expected = tuple(self.layout.dimensions[d] for d in dims)
            if key not in storage:
                diagnostic = (None if self.diagnostics is None or name == 'Time' else
                              self.diagnostics.field(pool, field))
                if (resident and name != 'Time') or diagnostic is not None:
                    # The resident provider caches owned buffers. Request its
                    # normal view, not a conflicting full/dummy allocation.
                    previous = views.level
                    try:
                        views.level = 1
                        device = diagnostic
                        if device is None:
                            device = views.field(pool, field)
                        slices = tuple(slice(0, size) for size in expected)
                        device = device[slices]
                        if hasattr(cp, 'cuda'):
                            # Pinned writer-owned storage, reused; _drain
                            # completed the previous frame before any reuse.
                            host = self._host_storage.get(key)
                            if host is None:
                                owner = cp.cuda.alloc_pinned_memory(device.nbytes)
                                host = np.ndarray(device.shape, dtype=device.dtype,
                                                  buffer=owner, order='C')
                                self._host_storage[key] = host
                            if host.shape != device.shape or host.dtype != device.dtype:
                                raise ValueError(f'{pool}.{field}: resident output layout changed')
                            wire_dtype = _wire_dtype(device.dtype)
                            wire = device.view(wire_dtype) if wire_dtype is not None else device
                            target = host.view(wire_dtype) if wire_dtype is not None else host
                            if device_array is not None and isinstance(wire, device_array):
                                queued.append((wire, target))
                            else:
                                wire.get(out=target, blocking=True)
                        elif device.dtype.itemsize == 4:
                            host = device.view(cp.uint32).get(order='C').view(device.dtype)
                        else:
                            host = device.get(order='C')
                        views.transfers['publication_download_bytes'] += host.nbytes
                    finally:
                        views.level = previous
                else:
                    host = self.native.get_field(pool, field, level=1)
                    host = host[tuple(slice(0, size) for size in expected)].copy()
                if host.shape != expected:
                    raise ValueError(f'{pool}.{field}: output shape {host.shape} != {expected}')
                storage[key] = host
            value = storage[key]
            if component is not None:
                value, dims = value[component], dims[1:]
            attrs = dict(self.field_attrs.get(name, {}))
            if name == 'Time':
                attrs['units'] = f'seconds since {self.initial:%Y-%m-%d %H:%M:%S}'
            result[name] = HistoryField(value, dims, attrs)
        if not queued:
            return result, _ready_now
        return result, self._queue_download(cp, queued)

    #: A subclass that overrides ``_snapshot`` (its synchronous frames) but takes the writer-thread path for every
    #: other frame (the MPAS LTS class executor, ``lts_v842``) sets this and overrides ``_assemble_captured``.
    WRITER_THREAD_CAPTURE = False

    #: Frames written at once (each holds its own pinned host storage). THE BREAKAGE THIS PREVENTS: with one
    #: frame in flight, a 60 km frame on 2 cards (owner assembly ~0.7 s plus a 3.6-4.6 s encode) outlasted the
    #: 3.6 s between hourly frames, so the next frame's step waited 1.2 s for the writer
    #: (MPAS-ROUTE-SPEED-2026-10-24, frame-async 2-card arm).
    FRAMES_IN_FLIGHT = 2

    #: Frames encoded at once.  THE COST THIS REMOVES (30 km globe over 4 cards, V5090 box, 2026-10-27): one
    #: rw_mpas_stream process encodes a frame on one core (netCDF-C), 24-36 s for a 30 km history frame against
    #: ~23 s of stepping per forecast hour, so frames queued behind one encoder and the run waited 40-63 s after
    #: its last step.  Each frame is its own process now, up to FRAMES_IN_FLIGHT at once; a frame that appends to
    #: a file (or writes a restart) waits for every earlier frame, each file's id is drawn in frame order on the
    #: forecast thread, and frames are published (durable marker, history-ready.jsonl) in frame order.
    ENCODE_WORKERS = FRAMES_IN_FLIGHT

    #: Device memory a staged frame leaves free (bytes, or a fraction of the card, whichever is larger):
    #: the staging block is transient, but it must not take the room a physics slot's peak needs.
    STAGING_RESERVE_BYTES = 2 << 30
    STAGING_RESERVE_FRACTION = 0.10

    def _queue_download(self, cp, queued):
        """Queue the frame's downloads; return the writer's wait (see ``_capture``)."""
        step_stream = cp.cuda.get_current_stream()
        align = 512
        sizes = [-(-wire.nbytes // align) * align for wire, _ in queued]
        total = int(sum(sizes))
        block = None
        free, card = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        reserve = max(self.STAGING_RESERVE_BYTES, int(self.STAGING_RESERVE_FRACTION * card))
        if free + pool.free_bytes() - total >= reserve:
            try:
                block = cp.empty(total, dtype=cp.uint8)
            except cp.cuda.memory.OutOfMemoryError:
                block = None
        if block is None:
            for wire, target in queued:
                source = wire if wire.flags.c_contiguous else cp.ascontiguousarray(wire)
                cp.cuda.runtime.memcpyAsync(target.ctypes.data, source.data.ptr, source.nbytes,
                                            cp.cuda.runtime.memcpyDeviceToHost, step_stream.ptr)
            done = cp.cuda.Event(disable_timing=True)
            done.record(step_stream)
            self.capture_receipt['in_stream_frames'] += 1
            return done.synchronize
        offset, staged = 0, []
        for (wire, target), size in zip(queued, sizes):
            stage = block[offset:offset + wire.nbytes].view(wire.dtype).reshape(wire.shape)
            stage[...] = wire
            staged.append((stage, target))
            offset += size
        copied = cp.cuda.Event(disable_timing=True)
        copied.record(step_stream)
        if self._copy_stream is None:
            self._copy_stream = cp.cuda.Stream(non_blocking=True)
        stream = self._copy_stream
        stream.wait_event(copied)
        for stage, target in staged:
            cp.cuda.runtime.memcpyAsync(target.ctypes.data, stage.data.ptr, stage.nbytes,
                                        cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
        done = cp.cuda.Event(disable_timing=True)
        done.record(stream)
        self.capture_receipt['staged_frames'] += 1
        self.capture_receipt['staging_bytes_max'] = max(self.capture_receipt['staging_bytes_max'], total)
        held = [block, staged]

        def ready():
            done.synchronize()
            # The staging block goes back to the pool only once its download landed.
            held.clear()

        return ready

    def _read_first_call(self, names):
        """Fresh host copies of the first call's fields (never the writer's pinned storage)."""
        views = getattr(self.session, 'device_pools', None)
        resident = views is not None and getattr(views, 'resident', False)
        values = {}
        for name in names:
            if name not in self.rows:
                continue
            row, component = self.rows[name]
            if row['kind'] == 'character' or component is not None:
                continue
            pool, field = row['pool'], row['pool_name']
            expected = tuple(self.layout.dimensions[d] for d in row['dimensions'])
            slices = tuple(slice(0, size) for size in expected)
            diagnostic = None if self.diagnostics is None else self.diagnostics.field(pool, field)
            if resident or diagnostic is not None:
                previous = views.level
                try:
                    views.level = 1
                    device = diagnostic if diagnostic is not None else views.field(pool, field)
                    host = device[slices].get(order='C') if hasattr(device, 'get') else np.asarray(device[slices])
                finally:
                    views.level = previous
            else:
                host = np.array(self.native.get_field(pool, field, level=1)[slices], copy=True, order='C')
            values[name] = host
        return values

    def complete_first_frame(self):
        """After step 1: complete the held initial frame from the first call, then publish it."""
        held, self._held_first = self._held_first, None
        if held is None:
            return None
        jobs, snapshot, when, step, receipt, extras, slot = held
        from .. import f000_first_call as first
        fields = {name: snapshot[name] for name in first.MPAS_FIELDS if name in snapshot}
        arrays = {name: np.asarray(item.values) for name, item in fields.items()}
        call = self._read_first_call(tuple(arrays)) if self.first_frame_from_first_call else None
        record = first.complete_start_frame(arrays, call, analysis=first.MPAS_ANALYSIS,
                                            placeholders=first.MPAS_PLACEHOLDERS, humidity=first.MPAS_HUMIDITY)
        for name in record.get('fields', {}):
            item = fields[name]
            snapshot[name] = HistoryField(arrays[name], item.dimensions, item.attrs)
        self.first_frame_receipt = record
        receipt['f000_first_call_diagnostics'] = record
        pending, self._f000_reflectivity = self._f000_reflectivity, None
        if pending is not None:
            receipt['f000_reflectivity'] = pending.apply(snapshot, HistoryField)
        self._inflight.append((self._submit_encode(jobs, step, self._encode, jobs, snapshot, when, step, extras),
                               receipt, slot))
        return record

    @property
    def first_frame_from_first_call(self):
        return getattr(self, '_first_frame_from_first_call', False)

    @first_frame_from_first_call.setter
    def first_frame_from_first_call(self, value):
        self._first_frame_from_first_call = bool(value)
        if value and getattr(self, '_f000_warm', None) is None:
            # Correct mode will hold the f000 frame: compile its reflectivity module now, during setup.
            try:
                from .rrfs.tempo_refl_v303 import warm_start_reflectivity
                import cupy as cp
                self._f000_warm = warm_start_reflectivity(
                    self.layout.configs, int(self.layout.dimensions['nVertLevels']), device=cp.cuda.Device().id)
                if self._f000_warm is None:
                    # WSM6's start-radar unit (wsm6_refl_start_v842) is its own compile: here, not at the f000 write.
                    from .wsm6_refl_start_v842 import warm_start_reflectivity as warm_wsm6
                    self._f000_warm = warm_wsm6(self.layout.configs, int(self.layout.dimensions['nVertLevels']),
                                                device=cp.cuda.Device().id)
            except Exception:
                self._f000_warm = None

    def _start_reflectivity(self, snapshot):
        """Start forming the held f000 frame's reflectivity from its own start state (None: nothing to form).

        THE BREAKAGE THIS ENDS: the start frame is written before any microphysics, so its refl10cm,
        refl10cm_max and refl10cm_1km were the zero placeholder on every start, an IAU analysis start full of
        rain included (hrrrv5 DA, Vermont 2023-07-10: the 10:15 analysis map was empty and unscorable).
        Correct mode only (the frame is held only there); a failure is named in the receipt, never the run's.
        """
        try:
            from .rrfs.tempo_refl_v303 import start_frame_reflectivity
            from .thompson_refl_start_v842 import start_frame_reflectivity as thompson_start
            from .wsm6_refl_start_v842 import start_frame_reflectivity as wsm6_start
            views = getattr(self.session, 'device_pools', None)
            cp = getattr(views, 'cp', None)
            if cp is None:
                import cupy as cp
            device = cp.cuda.Device().id
            # TEMPO (the RRFS suites), else Thompson (convection_permitting, hrrr-like-aerosol), else WSM6
            # (mesoscale_reference): the same breakage on the MPAS-A suites' routes, whose start frame showed
            # "No values to plot" over the first guess's own rain (hex-meshes-sources review, 2026-10-29).
            pending = start_frame_reflectivity(self.layout.configs, snapshot, device=device)
            if pending is None:
                pending = thompson_start(self.layout.configs, snapshot, device=device)
            if pending is None and str(self.layout.configs.get('config_microp_scheme', '')).strip() == 'mp_wsm6':
                pending = wsm6_start(self.layout.configs, snapshot, device=device,
                                     constants=None if views is None else views['constants'])
            return pending
        except Exception as error:
            return _FailedStartReflectivity(error)

    def _release_first_frame(self):
        """No first call will come (no step, or the run stopped): publish the frame as captured."""
        if self._held_first is not None:
            enabled, self.first_frame_from_first_call = self.first_frame_from_first_call, False
            try:
                self.complete_first_frame()
            finally:
                self.first_frame_from_first_call = enabled

    def _drain(self, keep=0):
        """Finish frames until at most ``keep`` are in flight (errors propagate here)."""
        if self._held_first is not None:
            self._release_first_frame()
        while self._inflight and (len(self._inflight) > keep or self._inflight[0][0].done()):
            future, receipt, _ = self._inflight.popleft()
            receipt.setdefault('streams', []).extend(future.result())

    def _free_slot(self):
        """A host storage set no frame in flight (or the held f000 frame) still reads."""
        self._drain(keep=self.FRAMES_IN_FLIGHT - 1)
        busy = {slot for _, _, slot in self._inflight}
        if self._held_first is not None:
            busy.add(self._held_first[-1])
        for slot in range(len(self._host_sets)):
            if slot not in busy:
                return slot
        self._host_sets.append({})
        return len(self._host_sets) - 1

    def write(self, *, step, dt, receipt):
        if self._closed:
            raise RuntimeError('MPAS output writer is closed')
        when = self.current_time(step=step, dt=dt)
        if self.diagnostics is not None:
            self.diagnostics.before_write(step=step, when=when)
            receipt['output_diagnostics'] = self.diagnostics.receipt()
        selected = [s for name, s in self.streams.items()
                    if 'output' in s.options['type'].split(';') and
                    self.stream_fields(s) and
                    self.alarms.due(name, when)]
        from ..forecast_start import radar_selected
        selected = radar_selected(selected, self.alarms, when)
        if any(s.name == 'radar' for s in selected):
            from ..forecast_start import radar_slot
            if not radar_slot(self.output):
                selected = [s for s in selected if s.name != 'radar']
        if not selected:
            return False
        from ..forecast_start import radar_minutes, radar_clock
        minutes = radar_minutes(self.output)
        if minutes:
            radar_clock(self.output, step, dt,
                self.start.strftime('%Y-%m-%d_%H:%M:%S'), minutes,
                history=any(s.name == 'output' for s in selected))
        waiting = perf_counter()
        slot = self._free_slot()
        self._host_storage = self._host_sets[slot]
        started = perf_counter()
        held = (step == 0 and self.first_frame_from_first_call and not self.control.config_do_restart
                and all(s.name != 'restart' for s in selected))
        # A held f000 frame is completed on the forecast thread from the global
        # words (complete_first_frame), and a subclass that overrides _snapshot
        # keeps its own forecast-thread capture unless it sets
        # WRITER_THREAD_CAPTURE (its _assemble_captured then does that work).
        asynchronous = not held and (self.WRITER_THREAD_CAPTURE or
                                     type(self)._snapshot is PhysicsStreams._snapshot)
        try:
            if asynchronous:
                snapshot, ready = self._capture(selected)
            else:
                snapshot, ready = self._snapshot(selected), None
        except Exception as error:
            if all(s.name == 'radar' for s in selected):
                from ..forecast_start import radar_error
                radar_error(self.output, error)
                return False
            raise
        # Variables outside the Registry for the history stream (the suite route's chemistry), captured now.
        extras = (None if getattr(self, 'extra_history', None) is None
                  or not any(s.name in ('output', 'history') for s in selected) else self.extra_history())
        if self.diagnostics is not None:
            # Reset only after immutable capture, while this event's stream
            # alarms still ring (mpas_atm_core.F:877-910).
            if any(s.name != 'radar' for s in selected):
                self.diagnostics.after_snapshot()
            else:
                # Radar reads instantaneous columns. Resetting the history's
                # convective accumulators here would change its next frame.
                self.diagnostics._end_event()
        jobs = []
        for stream in selected:
            target = (self.output / stream.filename(when, self.initial)).resolve()
            if not target.is_relative_to(self.output):
                raise ValueError(f'{stream.name}: filename escapes output directory')
            record = self.records.get(target, 0)
            # never_modify disallows preexisting files, but an open native
            # stream may append its own subsequent frames to the same file.
            publication = stream
            if record:
                publication = replace(stream, options={**stream.options, 'clobber_mode': 'append'})
            jobs.append((publication, target, record))
        if held:
            # Held until step 1's physics has run (complete_first_frame).
            self._held_first = (jobs, snapshot, when, step, receipt, extras, slot)
            self._f000_reflectivity = self._start_reflectivity(snapshot)
        else:
            if asynchronous:
                assembled = self._assembler.submit(self._assemble_captured, ready, snapshot)
                future = self._submit_encode(jobs, step, self._encode_assembled, assembled, jobs, when, step, extras)
            else:
                future = self._submit_encode(jobs, step, self._encode, jobs, snapshot, when, step, extras)
            self._inflight.append((future, receipt, slot))
        for stream, target, record in jobs:
            self.records[target] = record + 1
            self.alarms.reset(stream.name, when)
        receipt['output_capture_mode'] = dict(self.capture_receipt)
        receipt.setdefault('output_capture', []).append(dict(step=step,
            seconds=perf_counter()-started, encoding='background Rust process',
            download='copy stream, waited on by the writer thread' if asynchronous else 'forecast thread',
            wait_seconds=started-waiting, frames_in_flight=len(self._inflight),
            host_storage_sets=len(self._host_sets),
            fields=len(snapshot), resident=bool(getattr(getattr(self.session, 'device_pools', None), 'resident', False))))
        return any(s.name == 'diagnostics' for s in selected)

    def _assemble_captured(self, ready, snapshot):
        """Assembly worker: wait for the frame's download, then assemble it."""
        ready()
        return self._assemble(snapshot)

    def _encode_assembled(self, assembled, jobs, when, step, extras=None):
        """Encode worker, in frame order: the assembled frame, encoded."""
        return self._encode(jobs, assembled.result(), when, step, extras)

    def _bind_file_attributes(self, jobs):
        """Each new file's global attributes, its id drawn from the native sequence in frame order."""
        for stream, target, record in jobs:
            if target not in self._file_attributes:
                ids = self.file_ids
                if stream.name == 'radar':
                    # Scratch must not advance the native history's file-id RNG.
                    if not hasattr(self, 'radar_file_ids'):
                        self.radar_file_ids = NativeFileIds(seed=self.file_ids.seed)
                    ids = self.radar_file_ids
                self._file_attributes[target] = dict(self.attrs, file_id=ids.next(),
                                                     **self.contract.stream_file_attributes(stream))
                if stream.name in ('output', 'history') and getattr(self, 'extra_history_attrs', None):
                    self._file_attributes[target].update(self.extra_history_attrs)

    def _submit_encode(self, jobs, step, function, *arguments):
        """Queue one frame's encode (see ENCODE_WORKERS); called on the forecast thread, in frame order."""
        validate = self.output_validator() if self.output_validator is not None else None
        self._bind_file_attributes(jobs)
        previous, done = self._last_done, threading.Event()
        self._last_done = done
        self._publish_after[step] = previous
        ordered = any(record or stream.name == 'restart' for stream, _, record in jobs)

        def encode():
            try:
                if ordered:
                    previous.wait()
                if validate is not None:
                    # Capture can proceed while checks are in flight, but a
                    # rejected frame must never reach staging or the writer.
                    validate()
                return function(*arguments)
            except Exception as error:
                if all(stream.name == 'radar' for stream, _, _ in jobs):
                    from ..forecast_start import radar_error
                    radar_error(self.output, error)
                    return []
                raise
            finally:
                self._publish_after.pop(step, None)   # a rank that encodes nothing never publishes
                done.set()
        return self._executor.submit(encode)

    def _native_sentinels(self, snapshot):
        """A limited-area cut's "no neighbour" written as native writes it: 0, not the garbage element.

        runtime_v842.remap_limited_area_sentinels gives the pools native's in-memory garbage element (nCells+1,
        nEdges+1, nVertices+1, as mpas_block_creator does at read time); native's output maps that element back to
        0, the cull's stored value, because it has no global id.  THE BREAKAGE THIS FIXES: hex restart files of a
        limited-area run carried nCells+1 / nEdges+1 there (lam7k: cellsOnEdge 772, edgesOnEdge 63,442, cellsOnVertex
        1,161 words different from native), so a native restart from them would index past its arrays.
        """
        if not self.layout.configs.get('config_apply_lbcs', False):
            return snapshot
        from .runtime_v842 import LIMITED_AREA_CONNECTIVITY
        sizes = {}
        for item in snapshot.values():
            if isinstance(item, HistoryField) and item.dimensions:
                for dim, size in zip(item.dimensions, np.shape(item.values)):
                    sizes.setdefault(dim, int(size))
        out = None
        for name, kind in LIMITED_AREA_CONNECTIVITY.items():
            item = snapshot.get(name)
            if not isinstance(item, HistoryField) or kind not in sizes:
                continue
            values = np.asarray(item.values)
            garbage = values == sizes[kind] + 1
            if garbage.any():
                if out is None:
                    out = dict(snapshot)
                values = values.copy()
                values[garbage] = 0
                out[name] = replace(item, values=values)
        return snapshot if out is None else out

    def _encode(self, jobs, snapshot, when, step, extras=None):
        records = []
        snapshot = self._native_sentinels(snapshot)
        self._bind_file_attributes(jobs)
        for stream, target, record in jobs:
            started = perf_counter()
            options = dict(initial_time=self.initial, active_packages=self.layout.packages,
                           streams=self.streams, global_attrs=self._file_attributes[target],
                           field_attrs=self.field_attrs, scratch=self.output/'stream-scratch',
                           record_index=record,
                           registry_attributes=True)
            writer = self.contract.writer
            if stream.name == 'restart':
                writer.write_restart(target, self.mesh, snapshot, when, stream=stream,
                                     timestamp_path=self.output/'restart_timestamp', **options)
            else:
                if extras and stream.name in ('output', 'history'):
                    options['extra_fields'] = extras
                writer.write_stream(target, stream, self.mesh, snapshot, when, **options)
            records.append(dict(name=stream.name, step=step, seconds=perf_counter()-started,
                writer='rw_mpas_stream', path=str(target), record_index=record))
        self._publish_closed(jobs, step)
        return records

    def _publish_closed(self, jobs, step):
        """Publish each history frame the moment its file is closed for good.

        THE BREAKAGE THIS PREVENTS (parity run group 4, 2026-10-24): this route marked its frames durable in one
        batch after the forecast, so the live first look (which follows ``history-ready.jsonl`` and the durable
        sidecars) rendered nothing until the end: f001 came 55.6 s after its frame closed on the 120 km globe and 194 s
        on the 80 -> 10 km zoom globe, f000 84 s and 364 s. The WOOF route lists each frame as it closes.
        Here, on the writer thread and after every stream of this alarm is on disk (a frame's diag companion
        included), an ``output`` stream file that holds one frame (its filename bucket is its output interval)
        gets its durable sidecar and its ``history-ready.jsonl`` row. A file that later records append to keeps
        the post-run marking. Nothing on the forecast thread waits for this.
        """
        from ..history_writer import mark_closed_frame_durable
        waiting = getattr(self, '_publish_after', None)
        previous = None if waiting is None else waiting.pop(step, None)
        if previous is not None:
            previous.wait()   # frames are published in frame order (ENCODE_WORKERS)
        for stream, target, record in jobs:
            if stream.name == 'radar' and record == 0:
                from ..forecast_start import radar_ready
                radar_ready(self.output, target, step)
                continue
            if stream.name != 'output' or record != 0 or not _one_frame_per_file(stream):
                continue
            mark_closed_frame_durable(target)
            with (self.output / 'history-ready.jsonl').open('a', encoding='utf-8') as ready:
                ready.write(json.dumps({'step': int(step), 'path': str(target)}) + '\n')

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self._release_first_frame()
                self._drain()
            finally:
                self._executor.shutdown(wait=True)
                self._assembler.shutdown(wait=True)
                for storage in self._host_sets:
                    storage.clear()
                self._copy_stream = None
                close_diagnostics = getattr(self.diagnostics, 'close', None)
                if close_diagnostics is not None:
                    close_diagnostics()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
