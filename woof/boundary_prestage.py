"""Resident boundary preparation, suspended at the independent-analysis seam.

The callable passed to ``prestage_boundaries`` uses the normal mapped prep
front door in this process. Fetching may be part of that callable. No model
names, alternate decoder, interpolation, or boundary arithmetic live here.
"""
from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from dataclasses import replace
from pathlib import Path
import threading
import time
from types import SimpleNamespace

_CURRENT = ContextVar("boundary_prestage", default=None)


def current_boundary_prestage():
    return _CURRENT.get()


def _arrays(intervals):
    for interval in intervals:
        for field in interval.fields.values():
            for name in ("west", "east", "south", "north"):
                yield from (array for _, array in getattr(field, name).array_items())


class ResidentIntervals(Sequence):
    """The streamed reader protocol with every interval already in memory."""
    def __init__(self, intervals):
        self._intervals = tuple(intervals)
        self.bounds = tuple((iv.start_seconds, iv.end_seconds) for iv in intervals)
        self.fields = tuple(intervals[0].fields)
        self.validate = None
        self._validated = set()
        self.read_seconds = {}

    def __len__(self):
        return len(self._intervals)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._intervals[index]
        started = time.perf_counter()
        interval = self._intervals[index]
        if self.validate is not None and index not in self._validated:
            self.validate(interval)
            self._validated.add(index)
        self.read_seconds[index] = self.read_seconds.get(index, 0.0) + time.perf_counter() - started
        return interval

    def ready_prefix(self):
        return len(self)


class _PinnedTables:
    """Register immutable table pages without copying or weakening immutability."""
    def __init__(self, intervals):
        self.arrays = ()
        self.regions = []
        self.add(intervals)

    def add(self, intervals):
        import os
        import cupy as cp
        arrays = tuple(_arrays(intervals))
        self.arrays += arrays
        page = os.sysconf("SC_PAGE_SIZE")
        ranges = sorted((int(a.ctypes.data) // page * page,
                         (int(a.ctypes.data) + a.nbytes + page - 1) // page * page)
                        for a in arrays if a.nbytes)
        merged = []
        for start, end in ranges:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        try:
            for start, end in merged:
                # Separate immutable allocations may share a malloc page.
                # Register only pages not already owned by this handle.
                uncovered = [(start, end)]
                for lower, upper in self.regions:
                    pieces = []
                    for a, b in uncovered:
                        if b <= lower or a >= upper:
                            pieces.append((a, b))
                        else:
                            if a < lower:
                                pieces.append((a, lower))
                            if b > upper:
                                pieces.append((upper, b))
                    uncovered = pieces
                for a, b in uncovered:
                    if getattr(cp.cuda.runtime, "is_hip", False):
                        from woof.hip_runtime import host_register
                        host_register(a, b - a, 1)
                    else:
                        cp.cuda.runtime.hostRegister(a, b - a, 1)  # portable across devices
                    self.regions.append((a, b))
        except BaseException:
            self.close()
            raise

    def close(self):
        import cupy as cp
        for start, _ in reversed(self.regions):
            if getattr(cp.cuda.runtime, "is_hip", False):
                from woof.hip_runtime import host_unregister
                host_unregister(start)
            else:
                cp.cuda.runtime.hostUnregister(start)
        self.regions.clear()


def _device_owner(boundaries):
    import cupy as cp
    from woof.ingest.lateral_bc import attach_lateral_boundaries
    owner = SimpleNamespace(_scratch={})
    def scratch(shape, slot):
        result = cp.empty(shape, dtype=cp.float32)
        owner._scratch[slot] = result
        return result
    owner.scratch = scratch
    attach_lateral_boundaries(owner, boundaries)
    return owner


class BoundaryPrestage:
    """Own all boundary tables until the forecast and its CUDA work finish.

    ``device_budget_bytes`` is the caller's *admitted boundary allocation*,
    after reserving forecast state/physics/tiles. Free VRAM alone is not an
    admission. Devices without that budget keep the pinned host route.
    """
    def __init__(self, *, devices=(), device_budget_bytes=None, pin=True):
        self.devices = tuple(int(d) for d in devices)
        self.device_budget_bytes = dict(device_budget_bytes or {})
        self.pin = bool(pin)
        self._ready = threading.Event()
        self._initial = threading.Event()
        self._done = threading.Event()
        self._cancelled = False
        self._error = None
        self._initial_path = None
        self._pins = []
        self._device_tail = {}
        self._device_complete = {}
        self._tail = ()
        self.boundaries = None
        self.prepared_root = None
        self.result = None
        self.receipt = {"schema": "gpuwm-boundary-prestage-v1", "status": "PREPARING",
                        "started_wall": time.time(), "stages": [],
                        "analysis_dependent": ["start-frame coupling and first tendency",
                                               "initial atmosphere and land/soil",
                                               "analysis aerosol donor binding"],
                        "device_resident_bytes": {}}

    def wait_ready(self, timeout=None):
        if not self._ready.wait(timeout):
            return False
        if self._error is not None:
            raise self._error
        return True

    def finish_initial(self, initial_inputs, timeout=None):
        """Supply the existing initial-inputs JSON, then join normal preparation."""
        if self._initial.is_set():
            raise RuntimeError("initial analysis was already supplied or prestage closed")
        self._initial_path = Path(initial_inputs).resolve()
        self._initial.set()
        if not self._done.wait(timeout):
            raise TimeoutError("initial-state preparation has not completed")
        if self._error is not None:
            raise self._error
        return self.result

    def prepare_independent(self, *, forcing, times, build, release, workers,
                            output_root):
        """Use the route's admitted workers and exact producer, before analysis."""
        from woof.ingest.lateral_bc import LateralBoundaries
        started = time.perf_counter()
        started_wall = time.time()
        self.prepared_root = Path(output_root).resolve()
        count = len(times)
        tail = []
        # Bounded parallelism, as in BoundaryStreamWriter: at most workers
        # initialized states at a time, and deterministic frame consumption.
        pending = {}
        following = 1
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="boundary-prestage") as pool:
            def submit():
                nonlocal following
                if following < count:
                    pending[following] = pool.submit(copy_context().run, build, following)
                    following += 1
            for _ in range(min(workers, count - 1)):
                submit()
            try:
                for index in range(1, count):
                    met, initialized = pending.pop(index).result()
                    forcing.add_state(initialized.state, index=index)
                    del met, initialized
                    release()
                    if index > 1:
                        tail.append(forcing.interval(index - 1, times))
                        if index > 2:
                            forcing.release(index - 1)
                    submit()
                if count > 2:
                    forcing.release(count - 1)
            except BaseException:
                for future in pending.values():
                    future.cancel()
                raise
        self._tail = tuple(tail)
        self._widths = dict(spec_bdy_width=forcing.spec_bdy_width,
                            spec_zone=forcing.spec_zone, relax_zone=forcing.relax_zone)
        if self.pin and tail:
            self._pins.append(_PinnedTables(tail))
        tail_boundaries = LateralBoundaries(tuple(tail), **self._widths)
        if tail and self.devices:
            import cupy as cp
            from woof.ingest.lateral_bc import boundary_storage_shapes
            # Include the first interval that will arrive with the analysis.
            needed = sum(a.size * 4 for a in _arrays(tail))
            needed += sum(a.size * 4 for a in _arrays(tail[:1]))
            # Nonlinear producers need an additional evaluation buffer; the
            # full-domain hand-off below currently serves linear mapped tables.
            linear = "lbc_evaluated_tables" not in boundary_storage_shapes(tail_boundaries)
            for device in self.devices:
                budget = int(self.device_budget_bytes.get(device, 0))
                if not linear or budget < needed:
                    continue
                with cp.cuda.Device(device):
                    free, _ = cp.cuda.runtime.memGetInfo()
                    if free < needed:
                        continue
                    self._device_tail[device] = _device_owner(tail_boundaries)
        self.receipt.update(status="READY", ready_wall=time.time(),
                            forcing_times=count, independent_intervals=len(tail),
                            host_table_bytes=sum(a.nbytes for a in _arrays(tail)),
                            pinned_host_bytes=sum(end-start for p in self._pins for start,end in p.regions))
        self.receipt["stages"].append({"name": "independent_boundaries",
            "start_wall": started_wall, "end_wall": time.time(),
            "wall_seconds": time.perf_counter()-started, "workers": workers,
            "parallelism": {"workers": workers}, "status": "DONE"})
        self._ready.set()
        self._initial.wait()
        if self._cancelled:
            raise RuntimeError("boundary prestage cancelled before analysis")
        from woof.initial_source import read_initial_inputs
        return read_initial_inputs(self._initial_path)

    def complete(self, forcing, times):
        """Only interval zero depends on analysis; every tail object is reused."""
        from woof.ingest.lateral_bc import LateralBoundaries, _DeviceLateralBoundaries
        from types import MappingProxyType
        started = time.perf_counter()
        started_wall = time.time()
        first = forcing.interval(0, times)
        if self.pin:
            if self._pins:
                self._pins[0].add((first,))
            else:
                self._pins.append(_PinnedTables((first,)))
        sequence = ResidentIntervals((first, *self._tail))
        self.boundaries = LateralBoundaries(sequence, **self._widths)
        object.__setattr__(self.boundaries, "_prestage_owner", self)
        for device, tail in self._device_tail.items():
            import cupy as cp
            with cp.cuda.Device(device):
                head = _device_owner(LateralBoundaries((first,), **self._widths))
                scratch = {"lbc_prestaged_head": head._scratch["lbc_forcing_tables"],
                           "lbc_prestaged_tail": tail._scratch["lbc_forcing_tables"]}
                resident = _DeviceLateralBoundaries(
                    head._lateral_boundary_device.intervals + tail._lateral_boundary_device.intervals,
                    MappingProxyType({id(iv): i for i, iv in enumerate(sequence)}),
                    {}, set(scratch), sum(a.nbytes for a in scratch.values()))
                event = cp.cuda.Event()
                event.record()  # hand-off across prep/forecast threads, no host synchronization
                self._device_complete[device] = (resident, scratch, event)
                self.receipt["device_resident_bytes"][str(device)] = resident.device_nbytes
        self.receipt["stages"].append({"name": "analysis_boundary_tendency",
            "start_wall": started_wall, "end_wall": time.time(),
            "wall_seconds": time.perf_counter()-started,
            "parallelism": {"devices": list(self.devices)}, "status": "DONE"})
        self.receipt["status"] = "COMPLETE"
        self.receipt["host_table_bytes"] = sum(a.nbytes for a in _arrays(sequence))
        self.receipt["pinned_host_bytes"] = sum(end-start for p in self._pins for start,end in p.regions)
        return self.boundaries

    def try_attach(self, state, boundaries):
        """Transfer preallocated buffers to a matching full-domain state."""
        if boundaries is not self.boundaries or getattr(state, "_host_setup_state", False):
            return False
        import cupy as cp
        from woof.ingest.lateral_bc import _carried_external_clock, _release_resident_scratch
        entry = self._device_complete.get(int(cp.cuda.runtime.getDevice()))
        if entry is None:
            return False
        resident, scratch, event = entry
        cp.cuda.get_current_stream().wait_event(event)
        clock = _carried_external_clock(state)
        _release_resident_scratch(state)
        state._scratch.update(scratch)
        state._lateral_boundary_device = replace(resident, weights={}, scratch_slots=set(scratch), clock=clock)
        state.lateral_boundaries = boundaries
        state.elapsed_seconds = 0.0
        return True

    def for_forecast(self, inputs):
        if not self._done.is_set() or self._error is not None or self.boundaries is None:
            raise RuntimeError("boundary prestage must finish initial preparation before forecast")
        if Path(inputs.prepared_root).resolve() != self.prepared_root:
            raise ValueError("resident boundaries belong to another prepared root")
        if getattr(inputs, "stream_head", None) is not None:
            raise ValueError("resident boundaries require the completed preparation seal")
        if inputs.cache_reader.content_sha256 != self.cache_content_sha256:
            raise ValueError("resident boundaries differ from the verified prepared cache")
        self.boundaries.intervals.read_seconds.clear()
        return self.boundaries

    def close(self):
        """Cancel a waiting producer; after forecast completion release residency."""
        self._cancelled = True
        self._initial.set()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=60)
            if self._thread.is_alive():
                raise TimeoutError("boundary producer still owns its buffers; close after it exits")
        for pin in reversed(self._pins):
            pin.close()
        self._pins.clear()
        self._device_complete.clear()
        self._device_tail.clear()
        self._tail = ()
        self.boundaries = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def prestage_boundaries(prepare, *, devices=(), device_budget_bytes=None, pin=True):
    """Start normal fetch/prep in process and return its resident hand-off handle."""
    handle = BoundaryPrestage(devices=devices, device_budget_bytes=device_budget_bytes, pin=pin)
    context = copy_context()
    def run():
        token = _CURRENT.set(handle)
        try:
            if handle.devices:
                import cupy as cp
                cp.cuda.Device(handle.devices[0]).use()
            handle.result = prepare()
            if handle.boundaries is None:
                raise RuntimeError("preparation did not reach the mapped boundary prestage seam")
        except BaseException as error:
            handle._error = error
            handle.receipt.update(status="FAILED", error=str(error))
        finally:
            _CURRENT.reset(token)
            handle._done.set()
            handle._ready.set()
    handle._thread = threading.Thread(target=context.run, args=(run,), name="boundary-prestage", daemon=True)
    handle._thread.start()
    return handle
