#!/usr/bin/env python3
"""Partial, metadata-only CuPy launch observation for binder development.

This records calls visible at the Python RawModule/Module boundary. It also
records CuPy allocator leases and their release, using an UnownedMemory owner
to retain the original pooled allocation. It does
not observe CUDA runtime calls made inside compiled CuPy code, nor all native
library calls; consequently its output MUST NOT be treated as a sealed replay.
It never copies device buffers.  Run only in an instrumented referee process.
"""
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import dataclasses
import functools
import gzip
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def forecast_step():
    frame = sys._getframe(1)
    while frame:
        if (frame.f_code.co_name == "_run_native_dynamics"
                and frame.f_code.co_filename.endswith("/hexcore/mpasphys/runtime_v842.py")):
            return int(frame.f_locals.get("step", 0))
        frame = frame.f_back
    return 0


def allocation_stack():
    """Record source locations, not local variables or file contents."""
    result = []
    frame = sys._getframe(1)
    while frame:
        name = frame.f_code.co_filename
        if ("/hexcore/" in name and "/hexcore/ad/record/" not in name) or "/woof/" in name:
            result.append({"path": name, "line": frame.f_lineno,
                           "function": frame.f_code.co_name})
            if len(result) == 6:
                break
        frame = frame.f_back
    return result


class Observer:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        (root / "images").mkdir(exist_ok=True)
        self.output = gzip.open(root / "launches.jsonl.gz", "wt", encoding="utf-8", compresslevel=1)
        self.state_output = gzip.open(root / "state.jsonl.gz", "wt", encoding="utf-8", compresslevel=1)
        self.boundary_output = gzip.open(root / "boundaries.jsonl.gz", "wt", encoding="utf-8", compresslevel=1)
        self.module_output = gzip.open(root / "modules.jsonl.gz", "wt", encoding="utf-8", compresslevel=1)
        self.allocation_output = gzip.open(root / "allocations.jsonl.gz", "wt", encoding="utf-8", compresslevel=1)
        self.slot_output = gzip.open(root / "host-slots.jsonl.gz", "wt", encoding="utf-8", compresslevel=1)
        self.slot_count = 0
        self.slot_marker_present = False
        self.lock = threading.RLock()
        self.thread = threading.local()
        self.raw_images = {}
        self.count = 0
        self.module_events = 0
        self.allocation_count = 0
        self.allocation_live = {}
        self.allocation_anomalies = 0
        self.previous_allocator = None

    def allocation_id(self, mem):
        with self.lock:
            return self.allocation_live.get(int(mem.ptr))

    def module_event(self, action, module, function, name, image):
        with self.lock:
            self.module_output.write(json.dumps({"action": action,
                "clock_ns": time.monotonic_ns(), "module": module,
                "function": function, "name": name, "image_sha256": image,
                "complete": False}, separators=(",", ":")) + "\n")
            self.module_events += 1
            if self.module_events % 128 == 0:
                self.module_output.flush()

    def array_metadata(self, array):
        mem = array.data.mem
        return {"base": int(mem.ptr), "allocation_id": self.allocation_id(mem),
                "bytes": int(mem.size), "offset": int(array.data.ptr) - int(mem.ptr),
                "dtype": array.dtype.str, "shape": list(array.shape),
                "strides": list(array.strides)}

    def pool_metadata(self, cp, session):
        views = getattr(session, "device_pools", None)
        if views is None:
            return {}
        result = {}
        for label in ("buffers", "resident_buffers"):
            for key, entry in tuple(getattr(views, label, {}).items()):
                if len(entry) > 1 and isinstance(entry[1], cp.ndarray):
                    result[label + ":" + ".".join(map(str, key))] = self.array_metadata(entry[1])
        return result

    def outer_boundary(self, frame, step, cp):
        """Sample named device views at the same outer boundary as the digest.

        These are ownership candidates, not a complete census. In particular,
        host-held clocks and persistent physics arrays still need discovery.
        """
        dycore = frame.f_locals["dycore"]
        atmosphere = dycore.atmosphere
        state = atmosphere.state
        fields = {name: self.array_metadata(getattr(state, name)) for name in
                  ("rho", "rho_theta", "rho_u", "rho_w", "scalars")}
        saved = atmosphere.saved
        saved_fields = {field.name: self.array_metadata(value)
                        for field in dataclasses.fields(saved)
                        if isinstance((value := getattr(saved, field.name)), cp.ndarray)}
        session = frame.f_locals["session"]
        with self.lock:
            self.boundary_output.write(json.dumps({
                "boundary": "outer_after_step", "step": int(step),
                "clock_ns": time.monotonic_ns(), "fields": fields,
                "saved_fields": saved_fields,
                "physics_pools": self.pool_metadata(cp, session),
                "complete": False}, separators=(",", ":")) + "\n")
            self.boundary_output.flush()

    def install_allocator(self, cp):
        observer = self
        self.previous_allocator = cp.cuda.get_allocator()
        previous = self.previous_allocator

        class Lease:
            def __init__(self, original, ident, base):
                self.original = original
                self.ident = ident
                self.base = base

            def __del__(self):
                try:
                    with observer.lock:
                        if observer.allocation_live.get(self.base) == self.ident:
                            del observer.allocation_live[self.base]
                        else:
                            observer.allocation_anomalies += 1
                        if not observer.allocation_output.closed:
                            observer.allocation_output.write(json.dumps({
                                "action": "free", "allocation_id": self.ident,
                                "base": self.base, "step": forecast_step(), "clock_ns": time.monotonic_ns(),
                                "stack": allocation_stack()}, separators=(",", ":")) + "\n")
                except Exception:
                    # Never perturb forecast cleanup during interpreter shutdown.
                    pass

        def allocate(nbytes):
            original = previous(nbytes)
            base = int(original.ptr)
            memory = original.mem
            # Zero-byte allocations and nonzero pool offsets are not mapped by
            # this prototype. Passing them through preserves forward behaviour.
            if not base or base != int(memory.ptr):
                with observer.lock:
                    observer.allocation_anomalies += 1
                    observer.allocation_output.write(json.dumps({
                        "action": "unresolved", "base": base,
                        "requested_bytes": int(nbytes), "step": forecast_step(), "clock_ns": time.monotonic_ns(),
                        "stack": allocation_stack()}, separators=(",", ":")) + "\n")
                return original
            with observer.lock:
                observer.allocation_count += 1
                ident = observer.allocation_count
                if base in observer.allocation_live:
                    observer.allocation_anomalies += 1
                observer.allocation_live[base] = ident
                observer.allocation_output.write(json.dumps({
                    "action": "allocate", "allocation_id": ident, "base": base,
                    "bytes": int(memory.size), "requested_bytes": int(nbytes),
                    "device_id": int(memory.device_id), "step": forecast_step(), "clock_ns": time.monotonic_ns(),
                    "stack": allocation_stack()}, separators=(",", ":")) + "\n")
            lease = Lease(original, ident, base)
            wrapped = cp.cuda.UnownedMemory(base, int(memory.size), lease, memory.device_id)
            return cp.cuda.MemoryPointer(wrapped, 0)

        cp.cuda.set_allocator(allocate)

    def image(self, content):
        if isinstance(content, str):
            content = content.encode()
        if not isinstance(content, bytes):
            return None
        digest = hashlib.sha256(content).hexdigest()
        path = self.root / "images" / (digest + ".gz")
        if not path.exists():
            with gzip.open(path, "wb", compresslevel=1) as stream:
                stream.write(content)
        return digest

    def record(self, cp, name, image, grid, block, args, kwargs):
        import numpy as np

        step = forecast_step()
        entries = []
        for arg in args:
            if isinstance(arg, cp.ndarray):
                mem = arg.data.mem
                entries.append({"kind": "device", "base": int(mem.ptr),
                                "allocation_id": self.allocation_id(mem),
                                "bytes": int(mem.size), "offset": int(arg.data.ptr) - int(mem.ptr),
                                "object_id": id(mem), "dtype": arg.dtype.str,
                                "shape": list(arg.shape), "strides": list(arg.strides)})
            elif isinstance(arg, (np.generic, int, float, bool)):
                value = np.asarray(arg)
                entries.append({"kind": "scalar", "dtype": value.dtype.str,
                                "bytes_hex": value.tobytes().hex()})
            else:
                entries.append({"kind": "unresolved", "type": type(arg).__qualname__})
        stream = kwargs.get("stream")
        if stream is None:
            stream = cp.cuda.get_current_stream()
        with self.lock:
            self.count += 1
            record = {"ordinal": self.count, "step": step, "clock_ns": time.monotonic_ns(), "kernel": name,
                      "image_sha256": image, "grid": list(grid), "block": list(block),
                      "shared_bytes": int(kwargs.get("shared_mem", 0)),
                      "stream": int(stream.ptr), "arguments": entries,
                      "complete": False}
            self.output.write(json.dumps(record, separators=(",", ":")) + "\n")

    def install(self):
        import cupy as cp
        from cupy.cuda import function

        observer = self
        self.install_allocator(cp)
        original_module, original_raw_module = function.Module, cp.RawModule
        original_raw_kernel = cp.RawKernel

        class Function:
            def __init__(self, function_, name, image):
                self.function, self.name, self.image = function_, name, image

            def __call__(self, grid, block, args, **kwargs):
                observer.record(cp, self.name, self.image, grid, block, args, kwargs)
                return self.function(grid, block, args, **kwargs)

            def __getattr__(self, name):
                return getattr(self.function, name)

        class Module(original_module):
            def __init__(self):
                self.image = None
                caller = sys._getframe(1).f_code.co_filename
                self.is_hex = ("/hexcore/" in caller and "/hexcore/ad/record/" not in caller) or "/woof/" in caller
                super().__init__()

            def load(self, image):
                self.image = observer.image(image)
                active = getattr(observer.thread, "raw", None)
                if active is not None:
                    active.ad_image = self.image
                result = super().load(image)
                observer.module_event("load", int(self.ptr), None, None, self.image)
                return result

            def get_function(self, name):
                kernel = super().get_function(name)
                observer.module_event("function", int(self.ptr), int(kernel.ptr), name, self.image)
                return Function(kernel, name, self.image) if self.is_hex else kernel

        class RawModule(original_raw_module):
            def __init__(self, *args, **kwargs):
                path = kwargs.get("path")
                self.ad_image = observer.image(Path(path).read_bytes()) if path else None
                self.ad_key = (kwargs.get("code"), tuple(kwargs.get("options", ())),
                               kwargs.get("backend", "nvrtc"), kwargs.get("jitify", False),
                               kwargs.get("translate_cucomplex", False),
                               kwargs.get("enable_cooperative_groups", False), path)
                super().__init__(*args, **kwargs)

            def get_function(self, name):
                previous = getattr(observer.thread, "raw", None)
                observer.thread.raw = self
                try:
                    kernel = super().get_function(name)
                finally:
                    observer.thread.raw = previous
                if self.ad_image is not None:
                    observer.raw_images[self.ad_key] = self.ad_image
                else:
                    self.ad_image = observer.raw_images.get(self.ad_key)
                native_function = kernel.kernel
                observer.module_event("raw_function", int(native_function.module.ptr),
                                      int(native_function.ptr), name, self.ad_image)
                return Function(kernel, name, self.ad_image)

        class RawKernel(original_raw_kernel):
            def __init__(self, code, name, *args, **kwargs):
                self.ad_name = name
                self.ad_image = None
                options = args[0] if args else kwargs.get("options", ())
                self.ad_key = (code, tuple(options), kwargs.get("backend", "nvrtc"),
                               kwargs.get("jitify", False), kwargs.get("translate_cucomplex", False),
                               kwargs.get("enable_cooperative_groups", False), None)
                super().__init__(code, name, *args, **kwargs)

            def __call__(self, grid, block, args, **kwargs):
                if self.ad_image is None:
                    previous = getattr(observer.thread, "raw", None)
                    observer.thread.raw = self
                    try:
                        self.compile()
                    finally:
                        observer.thread.raw = previous
                    if self.ad_image is not None:
                        observer.raw_images[self.ad_key] = self.ad_image
                    else:
                        self.ad_image = observer.raw_images.get(self.ad_key)
                observer.record(cp, self.ad_name, self.ad_image, grid, block, args, kwargs)
                native_function = self.kernel
                observer.module_event("raw_kernel", int(native_function.module.ptr),
                                      int(native_function.ptr), self.ad_name, self.ad_image)
                return super().__call__(grid, block, args, **kwargs)

        function.Module = Module
        cp.RawModule = RawModule
        cp.RawKernel = RawKernel

        # This is a metadata sample of the named prognostic arrays, taken at
        # the dycore entry.  Outer physics can already have run by this point.
        # It is not a census of all carried physics and scratch allocations.
        from woof.hex.cuda_driver import CudaDryDycoreDriver
        from woof.hex.cuda_driver_rrfs import CudaRRFSDycoreDriver
        from woof.hex.mpasphys.session_v842 import NativePhysicsSession

        original_begin = NativePhysicsSession.begin_step

        @functools.wraps(original_begin)
        def observe_begin(session, itimestep, time_seconds):
            frame = sys._getframe(1)
            while frame and frame.f_code.co_name != "_run_native_dynamics":
                frame = frame.f_back
            dycore = frame.f_locals.get("dycore") if frame else None
            state = getattr(getattr(dycore, "atmosphere", None), "state", None)
            state_fields = {}
            if state is not None:
                for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars"):
                    value = getattr(state, name)
                    state_fields[name] = observer.array_metadata(value)
            with observer.lock:
                observer.boundary_output.write(json.dumps({
                    "boundary": "before_begin_step", "step": int(itimestep),
                    "clock_ns": time.monotonic_ns(),
                    "time_seconds": float(time_seconds), "fields": state_fields,
                    "physics_pools": observer.pool_metadata(cp, session),
                    "complete": False}, separators=(",", ":")) + "\n")
                observer.boundary_output.flush()
            return original_begin(session, itimestep, time_seconds)

        NativePhysicsSession.begin_step = observe_begin

        def make_observer(original_step):
            @functools.wraps(original_step)
            def observe_step(driver, *args, **kwargs):
                state = driver.atmosphere.state
                fields = {}
                for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars"):
                    value = getattr(state, name)
                    fields[name] = observer.array_metadata(value)
                saved = driver.atmosphere.saved
                saved_fields = {field.name: observer.array_metadata(value)
                                for field in dataclasses.fields(saved)
                                if isinstance((value := getattr(saved, field.name)), cp.ndarray)}
                with observer.lock:
                    observer.state_output.write(json.dumps({"step": forecast_step(),
                        "clock_ns": time.monotonic_ns(),
                        "time_seconds": state.time_seconds, "fields": fields,
                        "saved_fields": saved_fields,
                        "complete": False}, separators=(",", ":")) + "\n")
                    observer.state_output.flush()
                return original_step(driver, *args, **kwargs)

            return observe_step

        CudaDryDycoreDriver._step_device_v841 = make_observer(CudaDryDycoreDriver._step_device_v841)
        CudaRRFSDycoreDriver._step_device_v841 = make_observer(CudaRRFSDycoreDriver._step_device_v841)

        # Both drivers implement _call independently; wrap each original once.
        from woof.hex.mpasphys.driver_v842 import PhysicsDriver
        from woof.hex.mpasphys.rrfs.driver_rrfs import PhysicsDriver as RRFSPhysicsDriver
        marker = getattr(ctypes.CDLL(None), "binder_mark_slot", None)
        if marker is not None:
            marker.argtypes = [ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint]
            marker.restype = ctypes.c_int
        install_slot_observer(observer, marker, (PhysicsDriver, RRFSPhysicsDriver))

    def close(self):
        if self.previous_allocator is not None:
            import cupy as cp
            cp.cuda.set_allocator(self.previous_allocator)
        self.output.close()
        self.state_output.close()
        self.boundary_output.close()
        self.module_output.close()
        with self.lock:
            self.slot_output.close()
            self.allocation_output.close()
            live_count = len(self.allocation_live)
        (self.root / "status.json").write_text(json.dumps({
            "schema": "ad.binder.partial-observation/v1", "complete": False,
            "launches": self.count,
            "observed_allocations": self.allocation_count,
            "observed_live_at_close": live_count,
            "allocator_anomalies": self.allocation_anomalies,
            "observed_host_slots": self.slot_count,
            "driver_slot_markers": self.slot_marker_present,
            "limitations": ["compiled CuPy and native calls are not intercepted",
                            "only CuPy allocator leases, not all device allocations, are observed",
                            "allocation ownership and lifetime across host boundaries are unresolved",
                            "device copies, clears, host boundaries and events are not observed",
                            "some lazy compiled module images are unresolved"]}, indent=2) + "\n")


def install_slot_observer(observer, marker, driver_types):
    """Record slot metadata; the original routine still computes all results."""
    observer.slot_marker_present = marker is not None

    def wrap(original):
        @functools.wraps(original)
        def observed(driver, operation, step, time_seconds, flags, pools, *, slot=None):
            encoded = operation.encode("utf-8")
            if marker is not None and marker(encoded, int(step), 0):
                raise RuntimeError("driver slot begin marker failed")
            begin = time.monotonic_ns()
            calls_before = len(driver.calls)
            failure = None
            try:
                return original(driver, operation, step, time_seconds, flags, pools, slot=slot)
            except BaseException as error:
                failure = type(error).__qualname__
                raise
            finally:
                phase = 2 if failure else 1
                marker_status = marker(encoded, int(step), phase) if marker is not None else 0
                row = driver.calls[calls_before] if len(driver.calls) > calls_before else {}
                if row.get("operation") != operation or row.get("itimestep") != step:
                    row = {}
                record = {"operation": operation, "slot_argument": slot, "step": int(step),
                          "time_seconds": float(time_seconds), "begin_ns": begin,
                          "thread_id": threading.get_native_id(),
                          "end_ns": time.monotonic_ns(), "backend": row.get("backend"),
                          "driver_slot": row.get("slot"), "flags": dataclasses.asdict(flags),
                          "exception_type": failure, "driver_marker": marker is not None,
                          "driver_marker_status": marker_status, "complete": False}
                with observer.lock:
                    observer.slot_output.write(json.dumps(record, separators=(",", ":")) + "\n")
                    observer.slot_count += 1
                if marker_status:
                    raise RuntimeError("driver slot end marker failed")
        return observed

    originals = [(kind, kind._call) for kind in driver_types]
    for kind, original in originals:
        kind._call = wrap(original)


def boundary_trace(observer):
    """Mark the outer loop, including commands preceding physics begin_step.

    The exact source lines are located from the pinned function rather than
    assuming the begin_step hook is the beginning of the forecast step.
    """
    import linecache
    import cupy as cp
    marker = ctypes.CDLL(None).binder_mark_step
    marker.argtypes = [ctypes.c_uint]
    marker.restype = ctypes.c_int
    def trace(frame, event, arg):
        if event != "call" or frame.f_code.co_name != "_run_native_dynamics":
            return None
        if not frame.f_code.co_filename.endswith("/hexcore/mpasphys/runtime_v842.py"):
            return None
        lines = linecache.getlines(frame.f_code.co_filename)
        loop = next(i for i, line in enumerate(lines)
                    if "for step in range(steps+1):" in line)
        before = next(i + 1 for i in range(loop + 1, len(lines))
                      if "if step and nest_look is not None:" in lines[i])
        finish = next(i + 1 for i in range(loop + 1, len(lines))
                      if lines[i].strip() == "deferred_checks.finish()")
        def local(current, kind, value):
            if kind == "line" and current.f_lineno == before:
                step = current.f_locals.get("step", 0)
                if step and marker(step):
                    raise RuntimeError("initial device-allocation snapshot failed")
                if step:
                    observer.outer_boundary(current, step - 1, cp)
                    for output in (observer.output, observer.allocation_output, observer.module_output, observer.slot_output):
                        output.flush()
            elif kind == "line" and current.f_lineno == finish:
                step = current.f_locals.get("step", 0)
                if step and marker(step + 1):
                    raise RuntimeError("final forecast boundary marker failed")
                if step:
                    observer.outer_boundary(current, step, cp)
                    for output in (observer.output, observer.allocation_output, observer.module_output, observer.slot_output):
                        output.flush()
            return local
        return local
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("forecast", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if not options.forecast or options.forecast[0] != "--":
        parser.error("put -- before the ordinary hex CLI arguments")
    observer = Observer(options.out)
    cupti = ctypes.CDLL(None)
    cupti.binder_cupti_start.restype = ctypes.c_int
    status = cupti.binder_cupti_start()
    if status:
        raise RuntimeError(f"CUPTI subscription failed: {status}")
    frames = None
    try:
        observer.install()
        from observe_frames import install as install_frames
        frames = install_frames(options.out / "frames.jsonl.gz")
        from woof.hex.cli import main as hex_main
        sys.settrace(boundary_trace(observer))
        try:
            return hex_main(options.forecast[1:])
        finally:
            sys.settrace(None)
    finally:
        if frames is not None:
            frames.close()
        observer.close()
        status = cupti.binder_cupti_stop()
        if status:
            raise RuntimeError(f"CUPTI unsubscribe failed: {status}")


if __name__ == "__main__":
    sys.exit(main())
