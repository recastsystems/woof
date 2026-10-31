"""Discover selected column-physics requests on a first run, then compile in processes.

Factories run only as far as the compiler request. Both packages' capture
sentinels prevent loader manifests and physics-parameter records from being
published by warm-up. The forecast still owns those side effects and loads
its own module. Requests share Hex's ordinary single-flight table.
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import os
import threading
import time
import warnings

from . import kernel_plan

_lock = threading.Lock()
_runs = {}


def _tag(run):
    material = [kernel_plan._state.get("tag_text"), dataclasses.asdict(run)]
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()


def _selection(run):
    from woof import kernel_prewarm
    from woof.core import preflight
    from types import SimpleNamespace

    experiment = kernel_prewarm.single_domain_experiment(run)
    experiment.vertical = SimpleNamespace(p_top=getattr(run, "p_top", 0.0))
    # The engine's static inventory also contains its WRF dynamics. Hex uses
    # its own dynamics, so only selected physics and their inline helpers go
    # through this path. No unrelated saved plans or full catalog are read.
    modules = set(preflight.physics_kernel_modules(experiment)) - set(preflight.CORE_KERNEL_MODULES)
    if "rrtmgp_rte" in modules:
        modules.update(("rrtmgp_validation", "rrtmg_legacy_solar"))
    jobs = [job for job in kernel_prewarm.static_jobs(experiment)
            if job[0].split("[", 1)[0] in modules or job[0].startswith("woof.core.")]
    return jobs


def _capture(factory, args, engine):
    # This explicit compatibility guard is needed because the engine's cache
    # hits do not pass Hex's compiler funnel. Its own loader guards must also
    # stop capture before recording a physics parameter or manifest entry.
    local = getattr(engine, "_capture", None)
    sentinel = getattr(engine, "_Captured", None)
    if not isinstance(local, threading.local) or not isinstance(sentinel, type) or not issubclass(sentinel, BaseException):
        raise RuntimeError("engine static-capture guards are unavailable")
    previous = getattr(local, "on", False)
    previous_hex = getattr(kernel_plan._capture, "on", False)
    local.on = kernel_plan._capture.on = True
    try:
        factory(*args)
    except kernel_plan._CapturedRequest as request:
        if request.arguments is None:
            raise RuntimeError("static request uses an unsupported compiler route") from None
        return request.arguments, request.key
    except sentinel as request:
        if request.key is not None:
            raise RuntimeError("engine unexpectedly owns static compiler capture") from None
        return None
    finally:
        local.on = previous
        kernel_plan._capture.on = previous_hex
    return None


def _thread_compile(arguments, arch, original, device):
    kernel_plan._use(device)
    wall, cpu = time.perf_counter(), time.thread_time()
    result = original(arguments["source"], tuple(arguments["options"]), arch,
                      arguments["filename"], arguments["name_expressions"], None,
                      arguments["cache_in_memory"], None, arguments["method"])
    return result, dict(pid=os.getpid(), seconds=time.perf_counter() - wall,
                        cpu_seconds=time.thread_time() - cpu)


def _run(run, device, record):
    pool = None
    try:
        from woof import kernel_plan as engine
        from . import kernel_warm
        from .cuda_backend.aot import effective_options
        from .kernel_compile_worker import compile_request
        kernel_plan._use(device)
        jobs = _selection(run)
        record["selected"] = len(jobs)
        arch = kernel_plan._device_arch()
        identity = None
        identity_resolved = False
        original = kernel_plan._state["original"]
        for label, factory, args in jobs:
            try:
                captured = _capture(factory, args, engine)
                if captured is None:
                    record["cached"] += 1
                    record["jobs"].append(dict(label=label, result="cached"))
                    continue
                arguments, key = captured
                target = arch if arguments["arch"] == "device" else arguments["arch"]
                with kernel_plan._lock:
                    known = key in kernel_plan._flight
                if known:
                    record["known"] += 1
                    record["jobs"].append(dict(label=label, result="known", sha256=key))
                    continue
                # A direct route can miss its own disk cache while the exact
                # warm artifact exists. Restore it before starting processes.
                # This validates source, headers, compiler and image bytes.
                image = None
                if getattr(kernel_plan._compiler(), "__name__", None) == "cupy.cuda.compiler":
                    image = kernel_warm.restore(arguments, target)
                if image is not None:
                    future = concurrent.futures.Future()
                    future.set_result(image)
                    with kernel_plan._lock:
                        held = kernel_plan._flight.setdefault(key, future)
                    result = "restored" if held is future else "known"
                    record[result] += 1
                    record["jobs"].append(dict(label=label, result=result, sha256=key))
                    continue
                if not identity_resolved:
                    identity = kernel_plan._process_compiler_identity(arch)
                    identity_resolved = True
                with kernel_plan._lock:
                    if key in kernel_plan._flight:
                        record["known"] += 1
                        record["jobs"].append(dict(label=label, result="known", sha256=key))
                        continue
                    if pool is None:
                        if identity is not None:
                            # The plan pool's lifetime, for this pool too: its workers end with this
                            # process, and a worker left idle at exit is stopped instead of joined.
                            pool = kernel_plan.compiler_pool(kernel_plan._workers())
                            record["compiler_executor"] = "process"
                        else:
                            pool = concurrent.futures.ThreadPoolExecutor(
                                kernel_plan._workers(), thread_name_prefix="kernel-plan-static-compile")
                            record["compiler_executor"] = "thread"
                    if identity is not None:
                        arguments = dict(arguments, options=list(effective_options(arguments["options"])))
                        pending = pool.submit(compile_request, arguments, target, identity,
                                              kernel_warm.worker_config())
                    else:
                        pending = pool.submit(_thread_compile, arguments, target, original, device)
                    future = kernel_plan._ProcessCompileFuture(pending, record)
                    kernel_plan._flight[key] = future
                def completed(pending, future=future, label=label):
                    future.completed(pending)
                    if pending.cancelled():
                        with kernel_plan._lock:
                            record["cancelled"] += 1
                    elif future.exception() is not None:
                        error = future.exception()
                        with kernel_plan._lock:
                            record["errors"].append(dict(label=label, error=f"{type(error).__name__}: {error}"))
                        warnings.warn(f"Hex static compile {label}: {error}; foreground compiler remains active",
                                      RuntimeWarning, stacklevel=2)
                pending.add_done_callback(completed)
                record["submitted"] += 1
                record["jobs"].append(dict(label=label, result="submitted", sha256=key))
            except Exception as error:
                record["errors"].append(dict(label=label, error=f"{type(error).__name__}: {error}"))
                warnings.warn(f"Hex static kernel {label}: {error}; foreground compiler remains active",
                              RuntimeWarning, stacklevel=2)
        record["status"] = "submitted"
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(dict(label="setup", error=f"{type(error).__name__}: {error}"))
        warnings.warn(f"Hex static kernel setup: {error}; foreground compiler remains active",
                      RuntimeWarning, stacklevel=2)
    finally:
        if pool is not None:
            pool.shutdown(wait=False)
        record["discovery_seconds"] = time.monotonic() - record["started"]


def start(run, device_id=None):
    """Dispatch once per selected configuration; do not wait for discovery or compilation."""
    from . import kernel_prewarm
    if not kernel_prewarm.enabled() or not kernel_plan._state["installed"]:
        return None
    try:
        tag = _tag(run)
        device = kernel_plan._current_device() if device_id is None else int(device_id)
        key = (tag, device)
        with _lock:
            if key in _runs:
                return _runs[key]["future"]
            record = dict(status="queued", tag=tag, device_id=device,
                          selected=None, submitted=0, cached=0, known=0, restored=0, cancelled=0,
                          jobs=[], errors=[], compiler_executor=None,
                          pool_compiled=0, pool_failed=0, pool_seconds=0.0,
                          pool_cpu_seconds=0.0, pool_processes=[], last_done=None,
                          started=time.monotonic(), discovery_seconds=None)
            pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="kernel-plan-static")
            try:
                future = pool.submit(_run, run, device, record)
                _runs[key] = dict(record=record, future=future)
            finally:
                pool.shutdown(wait=False)
            return future
    except Exception as error:
        warnings.warn(f"Hex static kernel selection: {error}; foreground compiler remains active",
                      RuntimeWarning, stacklevel=2)
        return None


def receipt():
    import copy
    with _lock, kernel_plan._lock:
        return [copy.deepcopy({k: v for k, v in row["record"].items()
                               if k not in ("started", "last_done")}) for row in _runs.values()]
