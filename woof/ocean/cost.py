"""Wave stage timings without per-step device waits or scalar downloads."""
from __future__ import annotations

from collections import deque
import time


class StageMeter:
    def __init__(self, capacity):
        import cupy as cp
        self.pairs = [(cp.cuda.Event(), cp.cuda.Event()) for _ in range(capacity)]
        self.free = list(range(capacity))
        self.pending = deque()
        self.calls = self.measured = self.unmeasured = 0
        self.device_ms = self.launch_wall_seconds = 0.
        self.active = None

    def begin(self):
        self.calls += 1
        self.wall_start = time.perf_counter()
        self.active = self.free.pop() if self.free else None
        if self.active is not None:
            self.pairs[self.active][0].record()
        else:
            self.unmeasured += 1

    def end(self):
        if self.active is not None:
            self.pairs[self.active][1].record()
            self.pending.append(self.active)
        self.launch_wall_seconds += time.perf_counter()-self.wall_start
        self.active = None

    def collect(self):
        """Only at existing output/finalization cadence; never waits."""
        import cupy as cp
        while self.pending:
            i = self.pending[0]
            if not self.pairs[i][1].done:
                break
            self.device_ms += cp.cuda.get_elapsed_time(*self.pairs[i])
            self.measured += 1
            self.free.append(self.pending.popleft())

    def report(self):
        self.collect()
        return {"calls": self.calls, "measured_calls": self.measured,
                "unmeasured_calls": self.unmeasured, "pending_calls": len(self.pending),
                "device_ms": self.device_ms,
                "launch_wall_seconds": self.launch_wall_seconds,
                "parallelism": "CUDA on caller stream; 256 threads per block or solver backend"}


class WaveCost:
    def __init__(self, *, atmosphere_dt, output_interval, coupling_interval):
        import math
        steps = max(1, math.ceil(output_interval/atmosphere_dt))
        # Two output windows tolerate an asynchronous writer still holding the
        # preceding interval. A saturated ring is reported, never a hidden sync.
        self.stages = {
            "forcing_exchange": StageMeter(2*steps+8),
            "roughness_feedback": StageMeter(4*steps+8),
            "wave_solver": StageMeter(2*math.ceil(output_interval/coupling_interval)+8),
        }

    def collect(self):
        for meter in self.stages.values():
            meter.collect()


def receipt(physics, *, forecast_seconds, atmosphere_steps):
    ocean = getattr(physics, "ocean", None)
    if ocean is None:
        return {"schema": "gpuwm.waves.cost/v1", "status": "not-run"}
    stages = {} if ocean.cost is None else {key: meter.report() for key, meter in ocean.cost.stages.items()}
    complete = bool(stages) and all(row["measured_calls"] == row["calls"] for row in stages.values())
    device_ms = sum(row["device_ms"] for row in stages.values())
    count = max(1, int(atmosphere_steps))
    # Query memory only at finalization. Count distinct resident allocations,
    # not views or the whole shared CuPy pool. Core owns its inventory.
    arrays = list(ocean.buffers.values())+[ocean.alpha, ocean.zero, ocean.z0]
    arrays += [ocean.grid.latitude,ocean.grid.longitude,ocean.grid.cosalpha,ocean.grid.sinalpha,
               ocean.bathymetry.depth,ocean.bathymetry.elevation,ocean.bathymetry.wet,ocean.bathymetry.coastal]
    if ocean.exchange is not None:
        arrays += [ocean.exchange.zero,ocean.exchange.roughness,*ocean.exchange.history.values(),
                   *ocean.exchange.geometry[:3]]
    if ocean.timeline is not None:arrays.append(ocean.timeline.partitions)
    if ocean.output is not None:
        arrays += list(ocean.output.point_spectra)
        for values in ocean.output.point_fields.values():arrays += values
    if ocean.waves is not None:
        inventory = getattr(ocean.waves, "resident_arrays", None)
        if callable(inventory):
            arrays += list(inventory().values())
        elif callable(getattr(ocean.waves, "restart_state", None)):
            arrays += list(ocean.waves.restart_state()["arrays"].values())
    allocations = {}
    for array in arrays:
        memory = array.data.mem
        allocations[int(memory.ptr)] = int(memory.size)
    resident_bytes = sum(allocations.values())
    import cupy as cp
    wet_cells = int(cp.count_nonzero(ocean.bathymetry.wet>.5))
    return {
        "schema": "gpuwm.waves.cost/v1", "status": "complete" if complete else "incomplete-timing",
        "stages": stages, "atmosphere_steps": int(atmosphere_steps),
        "solver_output_diagnostics": ocean.diagnostics,
        "coupling_steps": ocean.steps, "interval_seconds": ocean.interval,
        "roughness_feedback": ocean.roughness_feedback,
        "wave_ms_per_atmosphere_step": device_ms/count,
        "percent_of_inclusive_forecast_step": (100*device_ms/(forecast_seconds*1000)
                                                if forecast_seconds>0 else None),
        "forecast_wall_seconds": float(forecast_seconds),
        "resident_bytes_counted": resident_bytes, "wet_cells": wet_cells,
        "bytes_counted_per_wet_cell": resident_bytes/wet_cells if wet_cells else None,
        "memory_inventory_complete": callable(getattr(ocean.waves, "resident_arrays", None)),
        "measurement": "CUDA events on caller stream, harvested at output cadence; wall timestamps cover enqueue work; no added step synchronization",
    }
