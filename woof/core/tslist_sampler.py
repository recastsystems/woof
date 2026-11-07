"""Bounded CUDA sampling and asynchronous Rust station output.

The domain adapter binds live carriers and the committed model clock. A
streamed domain supplies its owning tile's carriers before buffer reuse.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np


class TslistSampler:
    """Sample every call; copy and write once per ts_buf_size samples.

    CUDA event waits and device-to-host copies run on the output worker.
    Two bounded slots exert backpressure only at a batch boundary. There is
    no device scalar read, weather-array host copy, or CUDA synchronization
    on the forecast thread in sample(). All arrays stay strongly referenced
    until close() drains the final batch.
    """
    def __init__(self, *, ij, surface, moisture, u, v, w, phb, ph, theta,
                 pressure, cosa, sina, fnm, fnp, levels, buffer_size,
                 unstagger, theta_m, writer, theta_base=None, staged_winds=False):
        import cupy as cp
        from woof.core.kernels import get_kernel

        if type(buffer_size) is not int or buffer_size < 1:
            raise ValueError("ts_buf_size must be a positive integer")
        if type(levels) is not int or not 1 <= levels <= pressure.shape[0]:
            raise ValueError("max_ts_level must fit the mass levels")
        self.nz, self.ny, self.nx = map(int, pressure.shape)
        if self.nz < 2:
            raise ValueError("WRF calc_p8w needs at least two mass levels")
        ij = np.asarray(ij)
        if ij.ndim != 2 or ij.shape[1] != 2 or ij.dtype.kind not in "iu":
            raise ValueError("tslist ij must be an integer [station,2] array")
        if (np.any(ij < 0) or np.any(ij[:, 0] >= self.nx) or np.any(ij[:, 1] >= self.ny)):
            raise ValueError("tslist station lies outside the supplied domain")
        if unstagger and not staged_winds and (np.any(ij[:, 0] + 1 >= self.nx)
                          or np.any(ij[:, 1] + 1 >= self.ny)):
            raise ValueError("edge stations require the staged native wind-neighbor "
                             "plan supplied by StationOutput")
        if len(surface) != 13 or len(moisture) != 6:
            raise ValueError("tslist requires 13 surface and six moisture carriers in native order")
        nz, ny, nx = self.nz, self.ny, self.nx
        checked = [*surface, *moisture, u, v, w, phb, ph, theta,
                   pressure, cosa, sina, fnm, fnp]
        shapes = ([(ny, nx)] * 13 + [(nz, ny, nx)] * 6
                  + [(nz, ny, nx + 1), (nz, ny + 1, nx), (nz + 1, ny, nx),
                     phb.shape, (nz + 1, ny, nx), (nz, ny, nx),
                     (nz, ny, nx), (ny, nx), (ny, nx), (nz,), (nz,)])
        if phb.shape not in ((nz + 1,), (nz + 1, ny, nx)):
            raise ValueError("tslist PHB must be a native column or full interface field")
        device = cp.cuda.Device().id
        for array, shape in zip(checked, shapes, strict=True):
            if (not isinstance(array, cp.ndarray) or array.shape != shape
                    or array.dtype != cp.float32 or not array.flags.c_contiguous
                    or array.device.id != device):
                raise ValueError(f"tslist requires contiguous float32 carriers of shape {shape} on one device")
        if type(unstagger) is not bool or type(theta_m) is not bool:
            raise ValueError("tslist wind/theta selectors must be booleans")
        self.device = device
        self.writer = writer
        self.points = int(ij.shape[0])
        self.levels = levels
        self.buffer_size = buffer_size
        self.stride = 15 + 7 * levels
        self.ij = cp.asarray(ij, dtype=cp.int32)
        self.surface_ptrs = cp.asarray([a.data.ptr for a in surface], dtype=cp.uint64)
        self.moisture_ptrs = cp.asarray([a.data.ptr for a in moisture], dtype=cp.uint64)
        self.references = checked
        if theta_base is not None:
            if (not isinstance(theta_base, cp.ndarray)
                    or theta_base.dtype != cp.float32
                    or theta_base.shape not in ((nz,), (nz, ny, nx))
                    or not theta_base.flags.c_contiguous
                    or theta_base.device.id != device):
                raise ValueError("tslist theta base must be a native column or full mass field")
            self.references.append(theta_base)
        self.theta_base = theta_base if theta_base is not None else np.uint64(0)
        self.theta_base_column = np.int32(theta_base is not None and theta_base.ndim == 1)
        self.carriers = (u, v, w, phb, ph, theta, pressure, cosa, sina, fnm, fnp)
        self.selectors = tuple(np.int32(i) for i in (unstagger and not staged_winds, phb.ndim == 1, theta_m))
        self.kernel = get_kernel("tslist", "tslist_gather")
        self._allocate()

    @classmethod
    def output_only(cls, *, shape, points, levels, buffer_size, writer):
        """A tiled domain binds its live tile carriers at gather time."""
        import cupy as cp
        from woof.core.kernels import get_kernel
        self = cls.__new__(cls)
        self.nz, self.ny, self.nx = shape
        self.points, self.levels = points, levels
        self.buffer_size, self.writer = buffer_size, writer
        self.stride = 15 + 7 * levels
        self.device = cp.cuda.Device().id
        self.kernel = get_kernel("tslist", "tslist_gather")
        self._allocate()
        return self

    def _allocate(self):
        import cupy as cp
        from woof import perf_timing
        self.measure = perf_timing.enabled()
        self.timings = [[], []]
        self.slots = []
        for _ in range(2):
            shape = (self.buffer_size, self.points, self.stride)
            device_array = cp.empty(shape, dtype=cp.float32)
            pinned = cp.cuda.alloc_pinned_memory(device_array.nbytes)
            host_array = np.ndarray(shape, dtype=np.float32, buffer=pinned)
            self.slots.append((device_array, host_array, pinned))
        self.pending = [None, None]
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tslist-output")
        self.slot = 0
        self.count = 0
        self.closed = False

    def _write(self, slot, count, ready):
        import cupy as cp
        from woof import perf_timing
        import time
        began = time.perf_counter()
        with cp.cuda.Device(self.device):
            device_array, host_array, _pinned = self.slots[slot]
            stream = cp.cuda.Stream(non_blocking=True)
            stream.wait_event(ready)
            cp.cuda.runtime.memcpyAsync(
                host_array.ctypes.data, device_array.data.ptr,
                count * self.points * self.stride * 4,
                cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
            # This wait belongs to the worker. sample() can enqueue the next
            # batch while Rust formats and writes the preceding one.
            stream.synchronize()
            self.writer.write_batch(host_array[:count], count)
            if self.measure:
                gpu_seconds = sum(cp.cuda.get_elapsed_time(a, b)
                                  for a, b in self.timings[slot]) / 1000
                perf_timing.record("tslist.gather", gpu_seconds,
                    samples=count, stations=count*self.points)
                perf_timing.record("tslist.copy_write", time.perf_counter()-began,
                    samples=count, bytes=count*self.points*self.stride*4)

    def _submit(self):
        if not self.count:
            return
        import cupy as cp
        ready = cp.cuda.Event(disable_timing=True)
        ready.record()
        self.pending[self.slot] = self.pool.submit(self._write, self.slot, self.count, ready)
        self.slot = 1 - self.slot
        self.count = 0

    def begin_sample(self):
        if self.closed:
            raise RuntimeError("tslist sampler is closed")
        if not self.points:
            return
        if self.count == 0 and self.pending[self.slot] is not None:
            # Never overwrite a slot while its copy or write remains live.
            self.pending[self.slot].result()
            self.pending[self.slot] = None
        if self.count == 0:
            self.timings[self.slot].clear()

    def gather(self, hour, *, ij=None, output_indices=None, surface_ptrs=None,
               moisture_ptrs=None, carriers=None, theta_base=None,
               theta_base_column=None, shape=None, selectors=None):
        import cupy as cp
        points = self.points if ij is None else ij.shape[0]
        if not points:
            return
        nx, ny, nz = (self.nx, self.ny, self.nz) if shape is None else shape
        if ij is None:
            arguments = (self.ij, self.surface_ptrs, self.moisture_ptrs, *self.carriers,
                self.slots[self.slot][0],
                *(np.int32(i) for i in (nx, ny, nz, self.points, self.levels, self.count)),
                *self.selectors, np.float32(hour), self.theta_base, self.theta_base_column,
                np.uint64(0), np.int32(points))
        else:
            arguments = (ij, surface_ptrs, moisture_ptrs, *carriers,
                self.slots[self.slot][0],
                *(np.int32(i) for i in (nx, ny, nz, self.points, self.levels, self.count)),
                *selectors, np.float32(hour), theta_base, theta_base_column,
                output_indices, np.int32(points))
        if self.measure:
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record()
        self.kernel((points,), (128,), arguments, shared_mem=4 * (nz + 1))
        if self.measure:
            end.record()
            self.timings[self.slot].append((begin, end))

    def end_sample(self):
        if not self.points:
            return
        self.count += 1
        if self.count == self.buffer_size:
            self._submit()

    def sample(self, hour):
        self.begin_sample()
        self.gather(hour)
        self.end_sample()

    def checkpoint(self):
        if self.closed:
            raise RuntimeError("tslist sampler is closed")
        self._submit()
        for i, future in enumerate(self.pending):
            if future is not None:
                future.result()
                self.pending[i] = None
        return self.writer.checkpoint()

    def close(self):
        if self.closed:
            return
        try:
            self.checkpoint()
        finally:
            self.pool.shutdown(wait=True)
            self.writer.close()
            # A moving nest releases its outgoing state before building the
            # replacement. An already closed output must not retain weather
            # pointers or device slots while its small cursor is carried on.
            self.references = ()
            self.carriers = ()
            self.theta_base = None
            self.ij = self.surface_ptrs = self.moisture_ptrs = None
            self.slots.clear()
            self.closed = True
