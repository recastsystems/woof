"""Compile the previous run's CUDA modules in the background during host preparation.

THE COST THIS REMOVES, MEASURED (2026-10-05, RTX 5090 and RTX PRO 6000
Blackwell boxes): on a card whose CuPy kernel cache is empty -- every freshly
rented box -- the first forecast step compiled about 200 NVRTC modules one
after another, 114-125 s on conus12km and 140 s on the published 30 km mesh,
while every CPU core but one sat idle.  The same modules had been compiled,
from the same sources with the same options, by the previous run of the
same port.

WHAT THIS DOES.  While a forecast runs, every call into CuPy's compile-and-
cache entry (``cupy.cuda.compiler._compile_with_cache_cuda``) is recorded --
source, options, architecture, backend, name expressions -- into a manifest
in the shared kernel cache, one manifest per
CuPy release, NVRTC version and CuPy cache key.  At the next forecast start
the door replays that manifest on a thread pool through the SAME entry point,
so each module is compiled and written to CuPy's own disk cache exactly as
the forecast would have compiled it, while the main thread is still reading
files and preparing host state.  When the forecast reaches a kernel, CuPy
takes the already loaded module on the same CUDA context, or finds the cubin
on disk when the context differs. A replayed module is the bytes the forecast would
have produced itself: same function, same arguments, same compiler.

A manifest travels with the kernel cache, so a box that is handed
a prebuilt cache directory compiles its kernels in the background instead of
on the critical path.

EVERY ROUTE, EVERY CORE (2026-10-16).  The manifest above sees only CuPy's
cached route.  Route-B and RRFS physics, the engine's direct exact-image
cache and the in-memory carriers compile through NVRTC directly, and those
compiles stayed serial inside step 1 (a 7.4 MB MYNN-EDMF image held one core
for 8 minutes before the first step on an RTX PRO 6000 box; x1.40962 on an
RTX 4090 spent 111.5 s of its first step compiling at 1.1 cores).
:mod:`woof.hex.kernel_plan` now owns the one funnel every route ends in: it
records the run's whole kernel set per configuration under
``<kernel cache>/kernel-plans`` and, at the next start, compiles all of it on a
pool as wide as the usable cores, longest first, while the host prepares.
This module's manifest replay runs on that same pool. ``WOOF_HEX_KERNEL_PREWARM=off``
disables both recording and replay. Disabling host preparation alone does not
disable the kernel cache or serialise first-use compilation.

The kernel root is ``WOOF_HEX_KERNEL_CACHE_ROOT``, then
``WOOF_HEX_CACHE_ROOT``, then ``$XDG_CACHE_HOME/gpuwm-hex/prep`` (or
``~/.cache/gpuwm-hex/prep``). Member-local ``WOOF_HEX_PREP_CACHE`` and
``CUPY_CACHE_DIR`` do not relocate it. Services can bake this root into an
image or share it between their processes. Exact compiler/request keys
and per-artifact process locks permit one compilation per uncached request.

THE GUARDS.  A replay that raises is counted and dropped: the forecast then
compiles that module itself, as it always did.  Same-host multi-card runs
record and replay too (since 2026-10-13): CuPy binds its disk cache
directory when it is imported, so every rank reads the images the pool
writes, and a cubin is keyed by its architecture, not by the card that
compiled it.  The manifest is rewritten atomically and capped,
so a broken or oversized file cannot grow without bound or half-write.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import warnings
from typing import Any

from . import kernel_plan, kernel_warm, prep_cache

SCHEMA = "gpuwm-hex.kernel-prewarm/v1"
PACK_SCHEMA = "gpuwm.kernel-pack/v1"
ENVIRONMENT = "WOOF_HEX_KERNEL_PREWARM"
MAX_ENTRIES = 4000
_POOL_PREFIX = "hex-kernel-prewarm"
_FIELDS = ("source", "options", "arch", "extra_source", "backend",
           "enable_cooperative_groups", "name_expressions", "log_stream",
           "cache_in_memory", "jitify", "to_ltoir")

_lock = threading.Lock()
_recorded: dict[str, dict[str, Any]] = {}
_state: dict[str, Any] = {"recording": False, "replay": None}
_pool: concurrent.futures.ThreadPoolExecutor | None = None
#: Replay futures by entry digest.  A forecast that reaches a module the pool
#: has not started takes it back (the future is cancelled and the forecast
#: compiles it, as it always did); one the pool is compiling is waited for
#: and then found in CuPy's disk cache.  Without this the forecast compiled a
#: module a second time beside the pool's copy -- measured on conus12km at
#: 54.9 s of main-thread compiles in a run whose pool compiled all 200.
_inflight: dict[str, concurrent.futures.Future] = {}


def enabled() -> bool:
    flag = os.environ.get(ENVIRONMENT, "").strip().lower()
    return flag not in ("off", "0", "no", "false")


def _compiler():
    from cupy.cuda import compiler

    return compiler


def _cupy_version() -> str:
    import cupy

    return str(cupy.__version__)


def manifest_path() -> Path | None:
    root = kernel_warm.cache_root()
    compiler = _compiler()
    identity = json.dumps([SCHEMA, _cupy_version(), list(compiler._get_nvrtc_version()),
                           str(compiler._get_cupy_cache_key())])
    configuration = kernel_plan._state.get("tag_text") or kernel_plan.default_tag_text()
    return (root / "kernels" / hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24] /
            (hashlib.sha256(configuration.encode("utf-8")).hexdigest() + ".json"))


def _compiler_identity() -> dict[str, Any]:
    compiler = _compiler()
    return {"cupy": _cupy_version(), "nvrtc": list(compiler._get_nvrtc_version()),
            "cache_key": str(compiler._get_cupy_cache_key())}


def _entry(args: tuple, kwargs: dict) -> dict[str, Any] | None:
    bound = dict(zip(_FIELDS, args))
    bound.update(kwargs)
    if bound.get("log_stream") is not None or bound.get("cache_in_memory"):
        return None  # nothing reaches the disk cache to be found later
    source = bound.get("source")
    if not isinstance(source, str):
        return None
    names = bound.get("name_expressions")
    entry = {
        "source": source,
        "options": [str(option) for option in (bound.get("options") or ())],
        "arch": bound.get("arch"),
        "extra_source": bound.get("extra_source"),
        "backend": bound.get("backend", "nvrtc"),
        "enable_cooperative_groups": bool(bound.get("enable_cooperative_groups", False)),
        "name_expressions": None if names is None else [str(name) for name in names],
        "jitify": bool(bound.get("jitify", False)),
        "to_ltoir": bool(bound.get("to_ltoir", False)),
    }
    if not all(isinstance(entry[key], (str, type(None))) for key in ("arch", "extra_source", "backend")):
        return None
    return entry


def _digest(entry: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode("utf-8")).hexdigest()


def _plan_directory() -> Path | None:
    return kernel_warm.cache_root() / "kernel-plans"


def install_recorder() -> None:
    """Record every compile-and-cache call this process makes; saved at exit."""

    from .cuda_backend.compile_contract import hash_nvrtc_library_once

    hash_nvrtc_library_once()  # before the first compiler identity below
    if os.environ.get(ENVIRONMENT, "").strip().lower() not in ("off", "0", "no", "false"):
        try:
            kernel_plan.install(plan_dir=_plan_directory())
        except Exception:
            pass  # the forecast compiles as it always did
    if _state["recording"] or not enabled():
        return
    compiler = _compiler()
    original = compiler._compile_with_cache_cuda

    def recording(*args: Any, **kwargs: Any) -> Any:
        future = None
        try:
            entry = _entry(args, kwargs)
            if entry is not None:
                digest = _digest(entry)
                with _lock:
                    _recorded.setdefault(digest, entry)
                    if not threading.current_thread().name.startswith(_POOL_PREFIX):
                        future = _inflight.pop(digest, None)
        except Exception:
            future = None
        if future is not None:
            record = _state["replay"]
            if future.cancel():
                _count(record, "taken_back")
            else:
                waiting = not future.done()
                try:
                    loaded = future.result()
                except Exception:
                    loaded = None
                if waiting:
                    _count(record, "waited_for")
                if loaded is not None:
                    module, context = loaded
                    if context is not None and context == _context_token():
                        _count(record, "served_replay")
                        return module
                    _count(record, "replay_context_mismatch")
        return original(*args, **kwargs)

    recording.__wrapped__ = original  # type: ignore[attr-defined]
    compiler._compile_with_cache_cuda = recording
    _state["recording"] = True
    _state["original"] = original
    atexit.register(save_manifest)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        try:
            # The interchange is data. A published 2.8.7 engine need not
            # import a newer orchestration helper to consume its pack.
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or document.get("schema") != PACK_SCHEMA:
                return []
            if document.get("compiler") != _compiler_identity():
                return []
            records = document.get("entries")
            if not isinstance(records, list) or len(records) > MAX_ENTRIES:
                return []
            entries = []
            for row in records:
                entry = row.get("arguments") if isinstance(row, dict) else None
                if not isinstance(entry, dict) or not isinstance(entry.get("source"), str):
                    return []
                if row.get("sha256") != _digest(entry):
                    return []
                entries.append(entry)
            return entries
        except (OSError, ValueError, TypeError):
            return []
    # Existing image caches remain readable; new writes use the shared pack.
    path = path.with_suffix(".jsonl")
    entries = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    entry = json.loads(line)
                    if isinstance(entry, dict) and isinstance(entry.get("source"), str):
                        entries.append(entry)
    except (OSError, ValueError):
        return []
    return entries[:MAX_ENTRIES]


def save_manifest() -> None:
    """Merge this run's recorded compiles into the manifest (atomic rewrite)."""

    try:
        path = manifest_path()
        if path is None:
            return
        with _lock:
            new = dict(_recorded)
        if not new:
            return
        existing = {_digest(entry): entry for entry in _load_manifest(path)}
        if all(key in existing for key in new):
            return
        merged = {**existing, **new}
        document = {"schema": PACK_SCHEMA, "compiler": _compiler_identity(),
                    "entries": [{"sha256": _digest(entry), "arguments": entry}
                                for entry in list(merged.values())[-MAX_ENTRIES:]]}
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".kernel-pack.", dir=path.parent)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        pass


def _count(record: dict[str, Any] | None, key: str) -> None:
    if record is not None:
        with _lock:
            record[key] = record.get(key, 0) + 1


def _use_device(device: int) -> None:
    import cupy

    cupy.cuda.Device(device).use()


def _context_token() -> tuple[int, int] | None:
    """Bind a loaded module to its device and live CUDA context.

    Disk images are portable across contexts; loaded module handles are not.
    If context identity is unavailable, retain ordinary image-cache loading.
    """
    try:
        from cupy.cuda import driver, runtime
        context = int(driver.ctxGetCurrent())
        return (int(runtime.getDevice()), context) if context else None
    except Exception:
        return None


def start_replay_async(device_id: int | None = None) -> concurrent.futures.Future | None:
    """Restore the exact plan on a coordinator thread during host preparation.

    Recording must already be installed on the caller. The existing plan owns
    compiler parallelism and foreground single-flight; this thread only moves
    plan loading and validation off the door's foreground path.
    """
    if not enabled():
        return None
    with _lock:
        pending = _state.get("setup_future")
        if pending is not None:
            return pending
        device = kernel_plan._current_device() if device_id is None else int(device_id)
        record = {"status": "queued", "seconds": None, "error": None, "device_id": device}
        _state["setup"] = record

        def setup():
            began = time.monotonic()
            with _lock:
                record["status"] = "running"
            try:
                if device is not None:
                    _use_device(device)
                result = _start_replay(device)
            except Exception as exc:
                with _lock:
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                warnings.warn(f"kernel prewarm background setup failed; foreground fallback remains active: {exc}",
                              RuntimeWarning, stacklevel=2)
                return None
            else:
                with _lock:
                    record["status"] = "complete"
                return result
            finally:
                with _lock:
                    record["seconds"] = time.monotonic() - began

        pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="hex-prewarm-setup")
        try:
            future = pool.submit(setup)
            _state["setup_future"] = future
        finally:
            pool.shutdown(wait=False)
        return future


def start_replay(device_id: int | None = None) -> dict[str, Any] | None:
    """Set up replay synchronously, or join an already dispatched setup."""
    with _lock:
        pending = _state.get("setup_future")
    if pending is not None:
        return pending.result()
    return _start_replay(device_id)


def _start_replay(device_id: int | None = None) -> dict[str, Any] | None:
    """Validate the plan, then dispatch its compilation and manifest replay."""

    global _pool
    plan_pool = False
    if os.environ.get(ENVIRONMENT, "").strip().lower() not in ("off", "0", "no", "false"):
        try:
            plan_pool = kernel_plan.start(device_id) is not None
        except Exception as exc:
            warnings.warn(f"kernel plan setup failed; manifest and foreground fallback remain active: {exc}",
                          RuntimeWarning, stacklevel=2)
            plan_pool = False
    try:
        return _start_manifest_replay(device_id, plan_pool)
    finally:
        if plan_pool:
            kernel_plan.close()


def _start_manifest_replay(device_id: int | None, plan_pool: bool) -> dict[str, Any] | None:
    global _pool
    if not enabled() or _state["replay"] is not None:
        return _state["replay"]
    path = manifest_path()
    entries = [] if path is None else _load_manifest(path)
    record: dict[str, Any] = {"manifest": None if path is None else str(path),
                              "entries": len(entries), "compiled_or_found": 0,
                              "failed": 0, "seconds": None, "started_monotonic": time.monotonic()}
    _state["replay"] = record
    if not entries:
        return record
    if device_id is None:
        import cupy

        device_id = int(cupy.cuda.runtime.getDevice())
    device = int(device_id)
    compile_entry = _state.get("original") or _compiler()._compile_with_cache_cuda
    workers = kernel_plan._workers()

    def replay(entry: dict[str, Any]):
        try:
            _use_device(device)
            module = compile_entry(
                entry["source"], tuple(entry["options"]), entry["arch"], entry["extra_source"],
                entry["backend"], entry["enable_cooperative_groups"],
                None if entry["name_expressions"] is None else list(entry["name_expressions"]),
                None, False, entry["jitify"], entry["to_ltoir"],
            )
            return module, _context_token()
        except Exception:
            return None

    def finished(future: concurrent.futures.Future) -> None:
        if future.cancelled():
            return
        with _lock:
            if future.result() is not None:
                record["compiled_or_found"] += 1
            else:
                record["failed"] += 1
            if record["compiled_or_found"] + record["failed"] + record.get("taken_back", 0) >= record["entries"]:
                record["seconds"] = round(time.monotonic() - record["started_monotonic"], 3)

    own_pool = not plan_pool
    if own_pool:
        _pool = concurrent.futures.ThreadPoolExecutor(workers, thread_name_prefix=_POOL_PREFIX)

    def pool_replay(entry: dict[str, Any]):
        # On the plan's pool: name the thread as this module's pool so the
        # recorder treats its calls as replays, not as the forecast's.
        thread = threading.current_thread()
        name = thread.name
        thread.name = f"{_POOL_PREFIX}-{name}"
        try:
            return replay(entry)
        finally:
            thread.name = name

    for entry in entries:
        future = kernel_plan.submit(pool_replay, entry) if plan_pool else None
        if future is None:
            # The plan has no pool to take it: kernel_plan.start hands back the record of a plan already
            # started in this process after its pool was closed (a second forecast in one process, e.g. the
            # suite route after the door's own run), and submit then returns None.  THE BREAKAGE THIS
            # PREVENTS: AttributeError 'NoneType' object has no attribute 'add_done_callback' (land-1017 G0).
            if not own_pool:
                own_pool = True
                _pool = concurrent.futures.ThreadPoolExecutor(workers, thread_name_prefix=_POOL_PREFIX)
            future = _pool.submit(replay, entry)
        with _lock:
            _inflight[_digest(entry)] = future
        future.add_done_callback(finished)
    if own_pool:
        _pool.shutdown(wait=False)
    return record


def start_static(run, device_id: int | None = None):
    """Discover selected column-physics requests after the seam seals its run options."""
    from . import kernel_static
    return kernel_static.start(run, device_id)


def receipt() -> dict[str, Any]:
    from . import kernel_static
    record = _state["replay"]
    with _lock:
        setup = dict(_state["setup"]) if "setup" in _state else None
        recorded = len(_recorded)
        replay = None if record is None else {k: v for k, v in record.items() if k != "started_monotonic"}
    return {"schema": SCHEMA, "enabled": enabled(), "recorded_this_run": recorded, "replay": replay, "setup": setup,
            "kernel_plan": kernel_plan.receipt(),
            "warm_pack": kernel_warm.receipt(), "static": kernel_static.receipt()}
