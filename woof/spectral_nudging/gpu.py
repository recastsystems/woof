"""Persistent cuFFT graph for physical fields on their own horizontal grids.

The graph has no host readbacks, host FFTs or allocations during replay.
The owner refreshes analysis endpoints at forcing seams, and checks the error
word at an already-required output/forcing synchronization. This is an operator,
not a whole-model dispatcher or a forecast-qualified scheme.
"""
from dataclasses import dataclass
import math
from pathlib import Path

import cupy as cp
import numpy as np
from cupyx.scipy.fft import get_fft_plan

from .config import NudgingConfig


@dataclass(frozen=True)
class FieldBinding:
    values: object
    analysis_start: object
    analysis_end: object
    height_agl_m: object
    pbl_height_m: object
    vertical_weight: object = None
    held_tendency: object = None


class SpectralNudger:
    def __init__(self, config: NudgingConfig, bindings, *, elapsed_seconds,
                 dt_seconds, interval_start, interval_end,
                 record_before=None, record_after=None):
        if not config.enabled:
            raise ValueError("use build_nudger for the allocation-free off path")
        if set(bindings) - {"u", "v", "temperature", "geopotential", "qv"}:
            raise ValueError("bind the requested wind, temperature, geopotential or water vapor fields")
        self.config = config
        self.bindings = dict(bindings)
        self._validate_clock(elapsed_seconds, dt_seconds, interval_start, interval_end)
        self.device = cp.cuda.runtime.getDevice()
        for name, binding in self.bindings.items():
            config.validate_grid(binding.values.shape)
            shape = binding.values.shape
            for field in ("values", "analysis_start", "analysis_end", "height_agl_m", "pbl_height_m"):
                array = getattr(binding, field)
                expected = shape[-2:] if field == "pbl_height_m" else shape
                if (not isinstance(array, cp.ndarray) or array.dtype != cp.float32
                        or array.shape != expected or not array.flags.c_contiguous
                        or array.device.id != self.device):
                    raise ValueError(f"{name}.{field} must be contiguous GPU float32 {expected}")
            for field in ("vertical_weight", "held_tendency"):
                array = getattr(binding, field)
                if array is not None and (not isinstance(array, cp.ndarray)
                        or array.dtype != cp.float32 or array.shape != shape
                        or not array.flags.c_contiguous or array.device.id != self.device):
                    raise ValueError(f"{name}.{field} must be contiguous GPU float32 {shape}")
            if config.convention == "wrf" and binding.vertical_weight is None:
                raise ValueError("WRF nudging requires its field-specific vertical weights")
            if config.cadence_steps(dt_seconds) > 1 and binding.held_tendency is None:
                raise ValueError("fgdt requires restart-carried held tendencies")
        # An output overlapping any other binding silently changes the next
        # field's mismatch. Check address ranges at setup, without a readback.
        outputs = [(name, b.values) for name, b in self.bindings.items()]
        for name, output in outputs:
            lo, hi = output.data.ptr, output.data.ptr + output.nbytes
            for other_name, binding in self.bindings.items():
                for field in ("values", "analysis_start", "analysis_end", "height_agl_m", "pbl_height_m"):
                    if other_name == name and field == "values":
                        continue
                    array = getattr(binding, field)
                    if max(lo, array.data.ptr) < min(hi, array.data.ptr + array.nbytes):
                        raise ValueError(f"{name} output overlaps {other_name}.{field}")
        self.clock = cp.asarray([elapsed_seconds, dt_seconds, interval_start, interval_end], dtype=cp.float64)
        self.error = cp.zeros(1, dtype=cp.uint32)
        # GPU global-timer stamps are graph nodes. Read at output seams only.
        self.stage_times_ns = cp.zeros(2 * len(bindings), dtype=cp.uint64)
        self.stage_sums_ns = cp.zeros(len(bindings), dtype=cp.uint64)
        self.stage_count = cp.zeros(1, dtype=cp.uint64)
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.pool = cp.cuda.MemoryPool()
        source = Path(__file__).with_name("kernels.cu").read_text()
        self.module = cp.RawModule(code=source, options=("--std=c++11",))
        from woof.certify.kernel_manifest import record_module
        record_module("woof.spectral_nudging.gpu:kernels", source=source,
                      options=("--std=c++11",), module=self.module)
        self.kernels = {name: self.module.get_function(name)
                        for name in ("difference", "truncate_modes", "relax", "tick", "stamp")}
        # Warm plans only on scratch. No model values change during construction.
        cp.cuda.get_current_stream().synchronize()
        with self.stream, cp.cuda.using_allocator(self.pool.malloc):
            self.scratch = {name: cp.zeros(b.values.shape, dtype=cp.complex64)
                            for name, b in self.bindings.items()}
            self.held = {name: (b.held_tendency if b.held_tendency is not None else
                               cp.zeros_like(b.values)) for name, b in self.bindings.items()}
            self.plans = ({name: get_fft_plan(work, axes=(-2, -1), value_type="C2C")
                          for name, work in self.scratch.items()} if config.mode == "spectral" else {})
            for name in self.plans:
                work = self.scratch[name]
                with self.plans[name]:
                    cp.fft.ifft2(cp.fft.fft2(work, axes=(-2, -1)), axes=(-2, -1))
        self.stream.synchronize()
        with self.stream, cp.cuda.using_allocator(self.pool.malloc):
            self.stream.begin_capture()
            try:
                if record_before is not None:
                    record_before()
                self._record()
                if record_after is not None:
                    record_after()
            finally:
                self.graph = self.stream.end_capture()
        # Retain all scratch and private pool allocations for graph lifetime.

    def _validate_clock(self, elapsed, dt, start, end):
        stop = (None if self.config.gfdda_end_h is None else self.config.gfdda_end_h*3600.)
        past_end = stop is not None and elapsed >= stop
        if (not all(math.isfinite(v) for v in (elapsed, dt, start, end))
                or not 0 < dt <= self.config.relaxation_seconds
                or start >= end or elapsed < start
                or (not past_end and elapsed + dt > end)):
            raise ValueError("forcing must bracket the step and 0 < dt <= relaxation_seconds")
        self.clock_metadata = (elapsed, dt, start, end)

    def _record(self):
        cfg = self.config
        wrf = np.int32(cfg.convention == "wrf")
        cadence = np.int32(cfg.cadence_steps(float(self.clock_metadata[1])))
        end = np.float64(-1 if cfg.gfdda_end_h is None else cfg.gfdda_end_h * 3600.)
        ramp = np.float64(cfg.dtramp_min * 60. if cfg.if_ramping else 0.)
        for stage, (name, binding) in enumerate(self.bindings.items()):
            self.kernels["stamp"]((1,), (1,), (self.stage_times_ns, self.stage_sums_ns, np.int32(2*stage)))
            shape = binding.values.shape
            ny, nx = shape[-2:]
            n = binding.values.size
            grid, block = ((n + 255) // 256,), (256,)
            ints = tuple(np.int32(v) for v in (n, nx, ny))
            work = self.scratch[name]
            self.kernels["difference"](grid, block, (
                binding.values, binding.analysis_start, binding.analysis_end,
                work, self.clock, self.error, ints[0], wrf, cadence, end, ramp,
                np.int32(cfg.mode == "grid")))
            if cfg.mode == "spectral":
                with self.plans[name]:
                    spectrum = cp.fft.fft2(work, axes=(-2, -1))
                self.kernels["truncate_modes"](grid, block, (
                    spectrum, *ints, np.int32(cfg.cutoff_x), np.int32(cfg.cutoff_y)))
                with self.plans[name]:
                    filtered = cp.fft.ifft2(spectrum, axes=(-2, -1))
            else:
                filtered = work
            self.kernels["relax"](grid, block, (
                binding.values, filtered, binding.height_agl_m, binding.pbl_height_m,
                self.clock, *ints, np.int32(cfg.boundary_cells),
                np.float32(cfg.minimum_height_m), np.float32(cfg.ramp_depth_m),
                np.float32(1.0 / cfg.coefficient(name)),
                binding.vertical_weight if binding.vertical_weight is not None else binding.height_agl_m,
                self.held[name], wrf, cadence, np.int32(cfg.fgdtzero), end, ramp))
            self.kernels["stamp"]((1,), (1,), (self.stage_times_ns, self.stage_sums_ns, np.int32(2*stage+1)))
        self.kernels["tick"]((1,), (1,), (self.clock, self.error, self.stage_count))

    def enqueue(self, stream=None):
        """Launch one captured operator on the caller's stream, without a sync."""
        self.graph.launch(stream=stream or cp.cuda.get_current_stream())

    def check_at_seam(self):
        """Call only at an existing output or forcing seam, never each step."""
        if int(self.error.get()[0]):
            raise RuntimeError("spectral nudging step escaped its analysis interval; state was not nudged")

    def replace_interval(self, analyses, *, elapsed_seconds, dt_seconds,
                         interval_start, interval_end):
        """Refresh existing endpoint buffers at a synchronized forcing seam.

        analyses maps each field to (start, end) device arrays. Buffers keep
        their addresses, so the cuFFT graph remains valid. Geometry and PBL
        buffers likewise must be updated in place by the forecast adapter.
        """
        self._validate_clock(elapsed_seconds, dt_seconds, interval_start, interval_end)
        if set(analyses) != set(self.bindings):
            raise ValueError("a forcing interval must replace all four fields")
        for name, pair in analyses.items():
            if len(pair) != 2:
                raise ValueError("each field needs two analysis endpoints")
            for value in pair:
                target = self.bindings[name].values
                if (not isinstance(value, cp.ndarray) or value.shape != target.shape
                        or value.dtype != cp.float32 or value.device.id != self.device):
                    raise ValueError(f"invalid analysis endpoint for {name}")
        self.check_at_seam()
        for name, (a, b) in analyses.items():
            self.bindings[name].analysis_start[...] = a
            self.bindings[name].analysis_end[...] = b
        self.clock.set(np.asarray([elapsed_seconds, dt_seconds, interval_start, interval_end], dtype=np.float64))
