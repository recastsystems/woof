"""Compile a run's whole CUDA kernel set at once, on every core, before step 1.

THE DEFECT (2026-10-16: "no single-CPU path on the critical path").  On
a cold kernel cache -- a new card architecture, a fresh box, a wiped cache --
a forecast sat at 100 % of ONE core with its card at 0 % for minutes before
step 1, compiling 160-220 NVRTC images one after another as each kernel was
first reached (x1.40962 on an RTX 4090: step 1 111.5 s against 0.3 s steady
steps; on a 5090 214-301 s; route-B MPAS physics on an RTX PRO 6000: ~10 min).

WHAT THIS DOES.  Every compile route in woof and woof hex -- CuPy's cached
``RawModule``/``RawKernel``/``ElementwiseKernel`` route, the engine's direct
exact-image cache (``woof.nvrtc_ptx_cache``), the in-memory physics carriers
and every unit that calls ``compile_using_nvrtc`` itself -- ends in one CuPy
function, ``cupy.cuda.compiler._compile_using_nvrtc_no_warning``.  This module
wraps that one function:

* **Plan.**  Each request a run makes (source, options, architecture, program
  name, name expressions) is recorded into a plan file for this run's
  configuration (``tag``) and compiler identity.  The plan is the run's kernel
  set, in the very keys the compiler is called with.
* **Compile ahead.** Supported cubin requests use CPU compiler processes,
  as wide as the usable cores, including the first run's static captures.
  Each worker has an independent NVRTC instance. GPU loader work stays on a
  thread pool; no CUDA context enters a compiler process.
* **Single flight.**  A request that arrives while its compile is running
  waits for it; one still queued is taken back and compiled by the caller, as
  before; one already finished is returned at once.  Two threads never
  compile the same request twice.

WHY THE OUTPUT IS UNCHANGED.  The wrapper returns exactly what NVRTC returned
for the identical inputs, and every route still writes its own cache entry,
under its own key, from those bytes -- nothing else about a route changes.
Program names and memory-cache mode remain part of every match: NVRTC can
embed them in internal symbols even without assertions or debugging. The
architecture is recorded as "the device's" and resolved on the card the run
uses, so a plan made on one card architecture compiles for another.

A ROW THE ROUTE WILL FIND IN ITS CACHE IS NOT COMPILED AHEAD (2026-10-17).
The plan holds the requests that reached NVRTC, i.e. cache misses when they
were recorded.  Replaying them all at the next start recompiled kernels the
routes would load from disk anyway: on an RTX 4090 with a warm cache and plans
present, ``woof warm-kernels`` took 122.5 s and 78 NVRTC compiles against
2.5 s and 0 without the plans, and the second run on any box paid the same.
So each row records, per architecture, the disk entry its route stores the
image under: CuPy's cached route names its program after that entry
(``<sha1>.cubin.cu``), and a route with its own disk cache names its entry
around the compile (:func:`cache_entry`) and may name a probe that computes
the entry for another architecture.  At replay a row whose entry exists for
this card is skipped (``pool_skipped_cached``); rows without one (in-memory
routes, an architecture never seen) compile ahead as before.  A CuPy row
recorded on one card finds its entry on another by CuPy's own key form, used
only after it reproduces the entry recorded on the first card.

``WOOF_KERNEL_PLAN=off`` turns it off; ``WOOF_KERNEL_PLAN_WORKERS`` caps the
pool; ``WOOF_KERNEL_PLAN_DIR`` moves the plans.  The same file ships in
woof (``woof.kernel_plan``) and woof hex (``woof.hex.kernel_plan``); the
first one installed in a process owns the wrapper.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import contextlib
import copy
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import queue
import re
import sys
import tempfile
import threading
import time
from typing import Any, Iterable

SCHEMA = "gpuwm.kernel-plan/v1"
ENVIRONMENT = "WOOF_KERNEL_PLAN"
WORKERS_ENVIRONMENT = "WOOF_KERNEL_PLAN_WORKERS"
DIRECTORY_ENVIRONMENT = "WOOF_KERNEL_PLAN_DIR"
MAX_ENTRIES = 4000
#: Entries not requested in this many runs of their tag leave the plan.
KEEP_RUNS = 4
POOL_PREFIX = "kernel-plan"
_MARK = "__gpuwm_kernel_plan__"

_lock = threading.Lock()
_install_lock = threading.Lock()
_flight: dict[str, concurrent.futures.Future] = {}
_requested: dict[str, dict[str, Any]] = {}
_state: dict[str, Any] = {"installed": False, "plan": None, "tag": None, "tag_text": None,
                          "device_arch": None, "replay": None, "counts": {}}


def enabled() -> bool:
    return os.environ.get(ENVIRONMENT, "").strip().lower() not in ("off", "0", "no", "false")


def _compiler():
    from cupy.cuda import compiler

    return compiler


def compiler_identity() -> dict[str, Any]:
    import cupy

    compiler = _compiler()
    if getattr(compiler, "__name__", None) == "cupy.cuda.compiler":
        from woof.kernel_aot import compiler_identity as full_identity
        return full_identity()
    return {"cupy": str(cupy.__version__), "nvrtc": list(compiler._get_nvrtc_version()),
            "cache_key": str(compiler._get_cupy_cache_key()),
            "use_ptx": bool(getattr(compiler, "_use_ptx", False)),
            "use_pch": bool(getattr(compiler, "_use_pch", False))}


def default_tag_text(argv: Iterable[str] | None = None) -> str:
    """The run's configuration as far as kernels can see it.

    Command-line words that name a file or carry a number (paths, dates,
    hours, mesh sizes) are dropped: they change the run, not its kernel set.
    The numerics and exactness switches that change compile options stay.
    """

    words = [str(word) for word in (sys.argv[1:] if argv is None else argv)]
    kept = [word for word in words if "/" not in word and "\\" not in word
            and not any(ch.isdigit() for ch in word)]
    from woof import numerics
    kept.append(f"GPUWM_NUMERICS={numerics.mode()}")
    for name in ("WOOF_HEX_NUMERICS", "GPUWM_WRF_EXACT", "CUPY_CUDA_COMPILE_WITH_DEBUG"):
        if os.environ.get(name):
            kept.append(f"{name}={os.environ[name]}")
    return " ".join(kept)


def _device_arch() -> str:
    arch = _state["device_arch"]
    if arch is None:
        arch = str(_compiler()._get_arch())
        _state["device_arch"] = arch
    return arch


def _arguments(source, options, arch, filename, name_expressions, cache_in_memory, method):
    """The recorded form of one request (architecture relative to the card)."""

    options = tuple(str(option) for option in (options or ()))
    device = _device_arch()
    return {"source": source, "options": list(options),
            "arch": "device" if arch is None or str(arch) == device else str(arch),
            "filename": str(filename), "cache_in_memory": bool(cache_in_memory),
            "name_expressions": None if name_expressions is None else [str(n) for n in name_expressions],
            "method": None if method is None else str(method)}


def _key(arguments: dict[str, Any]) -> str:
    """What decides the image: the plan's entry digest and the flight key."""

    source, options = arguments["source"], tuple(arguments["options"])
    material = {"source": source, "options": list(options), "arch": arguments["arch"],
                "name_expressions": arguments["name_expressions"], "method": arguments["method"],
                # NVRTC embeds the basename in internal symbols even without __FILE__, asserts or
                # debug options (WSM5/WSM7's portable libm tables). Serving wsm5.cu for kern.cu
                # changed the exact-mode cubin in a serial GPU battery. The in-memory route also
                # changes the program name CuPy hands NVRTC.
                "filename": arguments["filename"], "cache_in_memory": arguments["cache_in_memory"]}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()


def _current_device() -> int | None:
    try:
        import cupy

        return int(cupy.cuda.runtime.getDevice())
    except Exception:
        return None


def _use(device: int | None) -> None:
    """Make ``device`` current on this thread (a no-op without CuPy)."""

    if device is None:
        return
    import cupy

    cupy.cuda.Device(int(device)).use()


#: CuPy's cached route compiles ``<cache entry>.cu``: the entry is a SHA-1 of its key plus the extension.
_CUPY_PROGRAM = re.compile(r"^([0-9a-f]{40}\.(?:cubin|ltoir))\.cu$")
CUPY_ROUTE = "cupy-disk"
_SHARED = "_gpuwm_kernel_plan_shared"


def _shared() -> dict[str, Any]:
    """State both packages' copies of this module share (kept on CuPy's compiler module)."""

    return vars(_compiler()).setdefault(_SHARED, {"local": threading.local()})


@contextlib.contextmanager
def cache_entry(route: str, path: str | Path):
    """Name the disk entry a compile made inside this block is stored under.

    ``route`` is ``"package.module:function"`` naming a probe
    ``function(arguments, arch) -> path | None`` that computes the entry of a
    recorded request for another architecture (or any other text for none).
    """

    try:
        local = _shared()["local"]
    except Exception:
        yield
        return
    previous = getattr(local, "entry", None)
    local.entry = (str(route), str(path))
    try:
        yield
    finally:
        local.entry = previous


def _cache_root() -> Path | None:
    """CuPy's resolved disk cache directory (relative entries live under it)."""

    compiler = _compiler()
    for name in ("get_cache_dir", "_get_cache_dir"):
        getter = getattr(compiler, name, None)
        if callable(getter):
            try:
                return Path(getter())
            except Exception:
                return None
    disk_type = getattr(compiler, "_DiskKernelCacheBackend", None)
    backend = getattr(compiler, "_kernel_cache_backend", None)
    if isinstance(disk_type, type) and isinstance(backend, disk_type):
        return Path(backend._cache_dir)
    return None


def _entry_of(filename, cache_in_memory) -> tuple[str, str] | None:
    """(route, entry) of the request being compiled, entry relative to the cache root when under it."""

    entry = getattr(_shared()["local"], "entry", None)
    if entry is None:
        match = None if cache_in_memory else _CUPY_PROGRAM.match(os.path.basename(str(filename)))
        if match is None:
            return None
        return CUPY_ROUTE, match.group(1)
    route, path = entry
    root = _cache_root()
    if root is not None and os.path.isabs(path):
        try:
            path = Path(path).relative_to(root).as_posix()
        except ValueError:
            pass
    return route, path


def _probe(route: str):
    if ":" not in route:
        return None
    module, _, function = route.partition(":")
    try:
        return getattr(importlib.import_module(module), function)
    except Exception:
        return None


def _cupy_extra_sources() -> tuple[Any, ...]:
    """The ``extra_source`` values CuPy hashes into its key: none, or its own header text (RawModule,
    elementwise kernels).  NVRTC never reads it, so the funnel cannot record it; the replay tries both."""

    try:
        from cupy._core import core

        return (None, core._get_header_source())
    except Exception:
        return (None,)


def _cupy_entry(arguments: dict[str, Any], arch: str, extra_source: Any = None) -> str:
    """The disk entry CuPy's cached route names for this request on ``arch``.

    Forms CuPy's own key (``_compile_with_cache_cuda``) from the options the
    funnel received, which already carry what that route appended.  The caller
    trusts it only after it reproduces the entry recorded on another card, so
    a CuPy release that keys differently (or a request with extra_source)
    compiles ahead as before instead of being skipped.
    """

    compiler = _compiler()
    options = tuple(arguments["options"])
    env = ((str(arch), options, compiler._get_nvrtc_version(), "nvrtc")
           + tuple(compiler._get_arch_for_options_for_nvrtc(str(arch))))
    cache = compiler._empty_file_preprocess_cache
    base = cache.get(env)
    if base is None:
        base = compiler._preprocess("", options, str(arch), "nvrtc")
        cache[env] = base
    key_src = "%s %s %s %s %s" % (env, base, arguments["source"], extra_source, compiler._get_cupy_cache_key())
    names = arguments["name_expressions"]
    if names:
        key_src += " ne%d " % compiler._cache_payload_version
        key_src += "\0".join(sorted(names))
    return compiler._hash_hexdigest(key_src.encode("utf-8")) + (".ltoir" if "-dlto" in options else ".cubin")


def _cupy_entry_elsewhere(arguments: dict[str, Any], entries: dict, target: str) -> str | None:
    """A CuPy row's entry for an architecture it was not recorded on, once the key form is proven."""

    for arch, name in entries.items():
        if not isinstance(name, str):
            return None
        try:
            for extra in _cupy_extra_sources():
                if _cupy_entry(arguments, str(arch), extra) == name:
                    return _cupy_entry(arguments, target, extra)
        except Exception:
            return None
        return None
    return None


def _cached(row: dict[str, Any], device_arch: str, root: Path | None) -> bool:
    """Whether this row's route will load its image from disk on this card."""

    arguments = row["arguments"]
    target = device_arch if arguments["arch"] == "device" else str(arguments["arch"])
    entries = row.get("cache_entries")
    entry = entries.get(target) if isinstance(entries, dict) else None
    if not isinstance(entry, str) and row.get("route") == CUPY_ROUTE and isinstance(entries, dict):
        entry = _cupy_entry_elsewhere(arguments, entries, target)
    if not isinstance(entry, str):
        probe = _probe(str(row.get("route") or ""))
        if probe is None:
            return False
        try:
            entry = probe(arguments, target)
        except Exception:
            return False
        if not entry:
            return False
    path = Path(entry)
    if not path.is_absolute():
        if root is None:
            return False
        path = root / path
    return path.is_file()


def _retarget_cupy_program(row: dict[str, Any], device_arch: str) -> dict[str, Any]:
    """Replay a CuPy disk request with the program name this card's caller will use."""
    arguments = row["arguments"]
    entries = row.get("cache_entries")
    if (row.get("route") != CUPY_ROUTE or arguments["arch"] != "device"
            or not isinstance(entries, dict) or device_arch in entries):
        return row
    entry = _cupy_entry_elsewhere(arguments, entries, device_arch)
    if entry is None:
        return row
    arguments = dict(arguments, filename=str(Path(arguments["filename"]).with_name(entry + ".cu")))
    return dict(row, arguments=arguments, sha256=_key(arguments),
                cache_entries={**entries, device_arch: entry})


def _count(name: str, amount: float = 1) -> None:
    counts = _state["counts"]
    counts[name] = counts.get(name, 0) + amount


def _copy(result):
    image, mapping = result
    return image, (dict(mapping) if isinstance(mapping, dict) else mapping)


def install(plan_dir: str | Path | None = None, tag_text: str | None = None) -> bool:
    """Wrap the NVRTC funnel (once per process) and bind this run's plan file."""

    # Concurrent members must not reread another installer's funnel as the
    # original compiler after installing the pack bridge. A nested funnel
    # waits on its own flight forever, holding the caller's factory lock;
    # static ahead() workers likewise wait on the future they just created.
    # Keep this separate from _lock: compilation never takes this lock, and
    # bridge/metadata setup must not hold the flight-table lock.
    with _install_lock:
        return _install(plan_dir, tag_text)


def _install(plan_dir: str | Path | None, tag_text: str | None) -> bool:
    if _state["installed"]:
        return True
    if not enabled():
        return False
    compiler = _compiler()
    original = compiler._compile_using_nvrtc_no_warning
    if getattr(original, _MARK, False):
        return False  # the other package's copy already owns the funnel
    directory = plan_dir or os.environ.get(DIRECTORY_ENVIRONMENT)
    if directory is None:
        from woof.kernel_warm_pack import cache_root
        directory = cache_root() / "plans"
    text = default_tag_text() if tag_text is None else str(tag_text)
    try:
        identity = compiler_identity()
    except Exception:
        return False
    record_raw = None
    if getattr(compiler, "__name__", None) == "cupy.cuda.compiler":
        import cupy
        from woof import kernel_aot

        from woof.kernel_warm_pack import install_metadata, cache_root
        _state["compiler_metadata"] = install_metadata(compiler, cache_root(), identity)
        kernel_aot.install(cupy)
        # The pack's final-NVRTC bridge/recorder must sit inside this funnel:
        # static capture exits below before it can load or compile anything.
        original = compiler._compile_using_nvrtc_no_warning
        if os.environ.get("WOOF_AOT_RECORD"):
            record_raw = kernel_aot.record_raw
    identity_digest = hashlib.sha256(json.dumps([SCHEMA, identity], sort_keys=True).encode()).hexdigest()[:20]
    tag = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    _state.update(identity=identity, root=Path(directory).expanduser() / identity_digest,
                  tag=tag, tag_text=text, original=original)
    _state["plan"] = _state["root"] / f"{tag}.json"

    def funnel(source, options=(), arch=None, filename="kern.cu", name_expressions=None,
               log_stream=None, cache_in_memory=False, jitify=None, method=None):
        route_filename = filename
        def call():
            return original(source, options, arch, filename, name_expressions, log_stream,
                            cache_in_memory, jitify, method)

        capturing = _capturing()
        if getattr(compiler, "__name__", None) == "cupy.cuda.compiler":
            from woof.kernel_aot import effective_options
            from woof.kernel_warm_pack import program_name
            options = effective_options(options)
            if isinstance(source, str):
                filename = program_name(source, filename)
        if log_stream is not None or jitify or not isinstance(source, str):
            if capturing:
                raise _Captured()
            return call()
        try:
            arguments = _arguments(source, options, arch, filename, name_expressions,
                                   cache_in_memory, method)
            key = _key(arguments)
        except Exception:
            if capturing:
                raise _Captured() from None
            return call()
        if capturing:
            # A static-plan capture (:func:`ahead`): hand back the request, compile nothing.
            raise _Captured(arguments, key)
        if record_raw is not None:
            record_raw(arguments)
        images = _warm_images()
        cached_image = None if images is None else images.get(arguments)
        if cached_image is not None:
            _count("warm_image_hits")
            return cached_image
        try:
            entry = _entry_of(route_filename, cache_in_memory)
            target = _device_arch() if arguments["arch"] == "device" else arguments["arch"]
        except Exception:
            entry = None
        # A pool thread's request is a warm-up, not the run's own: it is not
        # recorded into the plan.  Any thread takes back a queued compile, so
        # a pool thread never waits on a job queued behind it.
        in_pool = POOL_PREFIX in threading.current_thread().name
        owner = None
        with _lock:
            if not in_pool:
                row = _requested.setdefault(key, {"arguments": arguments, "seconds": None})
                if entry is not None:
                    row.setdefault("cache_entries", {})[target] = entry[1]
                    row["route"] = entry[0]
            pending = _flight.get(key)
            if pending is None or pending.cancel():
                if pending is not None:
                    _count("taken_back")
                owner = concurrent.futures.Future()
                owner.set_running_or_notify_cancel()
                _flight[key] = owner
        if owner is None:
            try:
                result = pending.result()
            except Exception:
                _count("failed_then_inline")
                return call()
            if images is not None:
                images.put(arguments, result)
            _count("served")
            return _copy(result)
        started = time.perf_counter()
        process_compile = (_state.get("compiler_pool") is not None
                           and _state.get("closer") is None and arguments.get("method") is None)
        try:
            if process_compile:
                target = _device_arch() if arguments["arch"] == "device" else arguments["arch"]
                result = _submit_and_wait(arguments, target, _state["replay"])
            else:
                result = call()
        except BaseException as error:
            with _lock:
                _flight.pop(key, None)
            owner.set_exception(error)
            raise
        seconds = time.perf_counter() - started
        owner.set_result(result)
        if images is not None:
            images.put(arguments, result)
        with _lock:
            prefix = "dynamic_pool" if process_compile else "inline"
            _count(prefix + "_compiles")
            _count(prefix + "_seconds", seconds)
            if key in _requested:
                _requested[key]["seconds"] = round(seconds, 3)
        return _copy(result)

    setattr(funnel, _MARK, True)
    funnel.__wrapped__ = original  # type: ignore[attr-defined]
    compiler._compile_using_nvrtc_no_warning = funnel
    _state["installed"] = True
    atexit.register(save)
    return True


def _read(path: Path) -> list[dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if (not isinstance(document, dict) or document.get("schema") != SCHEMA
            or document.get("compiler") != _state.get("identity")):
        return []
    rows = document.get("entries")
    if not isinstance(rows, list) or len(rows) > MAX_ENTRIES:
        return []
    good = []
    for row in rows:
        arguments = row.get("arguments") if isinstance(row, dict) else None
        if not isinstance(arguments, dict) or not isinstance(arguments.get("source"), str):
            return []
        try:
            if row.get("sha256") != _key(arguments):
                return []
        except Exception:
            return []
        good.append(row)
    return good


def load_plan() -> tuple[list[dict[str, Any]], str]:
    """Direct lookup for this shape. Static jobs cover a first use, without a pool scan."""
    path = _state["plan"]
    rows = _read(path) if path is not None and path.is_file() else []
    return rows, ("tag" if rows else "none")


def _warm_images():
    if not _state.get("installed") or getattr(_compiler(), "__name__", None) != "cupy.cuda.compiler":
        return None
    # Device and compiler identities are fixed before any compile is submitted.
    with _lock:
        images = _state.get("warm_images")
        if images is None:
            from woof.kernel_warm_pack import WarmImages, cache_root
            images = WarmImages(cache_root(), _state["identity"], _state["tag_text"], _device_arch())
            _state["warm_images"] = images
        return images


def _workers() -> int:
    """Compiler threads and processes: CPU affinity capped by the cgroup CPU quota.

    U-E5 (2026-10-26): affinity alone sized this pool, so a container with
    nproc 256 and a 61.44-core CFS quota (a rented box) opened 256 compiler
    threads and 256 spawn compiler processes for a cold kernel cache, and
    the first native parent step of a WOOF Regional case took 426.9 s.
    """
    from woof.ingest.preparation_workers import cpu_budget

    cores = cpu_budget()["available_cpus"]
    limit = os.environ.get(WORKERS_ENVIRONMENT, "").strip()
    if limit.isdigit() and int(limit) > 0:
        cores = min(cores, int(limit))
    return max(1, cores)


class _ProcessCompileFuture(concurrent.futures.Future):
    """The shared hex compiler proxy: cancellation must reach the real job."""

    def __init__(self, pending, record, *, request_sha256=None, arch=None):
        super().__init__()
        self.pending = pending
        self.record = record
        self.request_sha256 = request_sha256
        self.arch = arch

    def cancel(self):
        return self.pending.cancel() and super().cancel()

    def completed(self, pending):
        if pending.cancelled():
            super().cancel()
            return
        try:
            result, timing = pending.result()
        except BaseException as error:
            with _lock:
                self.record["pool_failed"] += 1
                self.record["last_done"] = time.monotonic()
            self.set_exception(error)
            return
        with _lock:
            record = self.record
            record["pool_compiled"] += 1
            record["pool_seconds"] = round(record["pool_seconds"] + timing["seconds"], 3)
            record["pool_cpu_seconds"] = round(record["pool_cpu_seconds"] + timing["cpu_seconds"], 6)
            record["last_done"] = time.monotonic()
            worker = next((r for r in record["pool_processes"] if r["pid"] == timing["pid"]), None)
            if worker is None:
                worker = {"pid": timing["pid"], "entries": 0, "cpu_seconds": 0.0}
                record["pool_processes"].append(worker)
            if "compile_intervals" in record and "compile_begin_mono_ns" in timing:
                record["compile_intervals"].append({
                    "request_sha256": self.request_sha256, "arch": self.arch,
                    "pid": timing["pid"],
                    "begin_mono_ns": timing["compile_begin_mono_ns"],
                    "end_mono_ns": timing["compile_end_mono_ns"],
                    "cpu_seconds": timing["cpu_seconds"]})
            worker["entries"] += 1
            worker["cpu_seconds"] = round(worker["cpu_seconds"] + timing["cpu_seconds"], 6)
        self.set_result(result)


def _process_compiler_identity(arch):
    """Use the CPU-only cubin worker where CuPy supports the explicit target."""
    try:
        compiler = _compiler()
        if getattr(compiler, "__name__", None) != "cupy.cuda.compiler" or compiler._use_ptx:
            return None
        from cupy_backends.cuda.libs import nvrtc
        if int(str(arch).removeprefix("sm_")) not in nvrtc.getSupportedArchs():
            return None
        from woof.kernel_aot import compiler_identity as identity
        return identity()
    except (ImportError, AttributeError, ValueError):
        return None


def _submit_compile(arguments, arch, record, *, key=None):
    """Submit identical final options to the isolated compiler processes."""
    from woof.kernel_aot import effective_options
    from woof.kernel_compile_worker import compile_request

    arguments = dict(arguments, options=list(effective_options(arguments["options"])))
    with _lock:
        if key is not None and key in _flight:
            return _flight[key]
        pending = _state["compiler_pool"].submit(
            compile_request, arguments, arch, _state["worker_identity"],
            **({"trace": True} if "compile_intervals" in record else {}))
        future = _ProcessCompileFuture(
            pending, record,
            request_sha256=(_key(arguments) if "compile_intervals" in record else None),
            arch=str(arch))
        if key is not None:
            _flight[key] = future
    # A completed job invokes callbacks immediately. Publish its proxy first,
    # then register outside the lock used by completed() and the NVRTC funnel.
    pending.add_done_callback(future.completed)
    return future


def _aot_coverage():
    """Scheduling hint only; the loader still checks the full pack contract."""
    if getattr(_compiler(), "__name__", None) != "cupy.cuda.compiler":
        return set()
    from woof.kernel_aot import Pack
    pack = Pack()
    if pack.reason:
        return set()
    held = set()
    for row in pack.rows:
        if row.get("compiler_route") == "raw-nvrtc":
            metadata = row.get("compile_request") or {}
            if metadata.get("cache_in_memory") or metadata.get("method") is not None:
                continue
            held.add(("raw", row["source_sha256"], tuple(row["options"]),
                      tuple(metadata.get("name_expressions") or ()), row["arch"],
                      metadata.get("filename")))
            continue
        options = list(row["options"])
        if row.get("compiler_route") != "direct-nvrtc":
            options.append("-ftz=true")
        names = (row.get("compile_request") or {}).get("name_expressions")
        held.add((row["source_sha256"], tuple(options), tuple(names or ()), row["arch"]))
    return held


def _aot_cached(arguments, arch, held):
    from woof.kernel_aot import effective_options, portable_options, source_hash
    if arguments.get("method") is not None:
        return False
    target = arch if arguments["arch"] == "device" else arguments["arch"]
    signature = (source_hash(arguments["source"]),
                 tuple(portable_options(effective_options(arguments["options"]))),
                 tuple(arguments.get("name_expressions") or ()),
                 "sm_" + str(target).removeprefix("sm_"))
    return (signature in held or
            (not arguments.get("cache_in_memory")
             and ("raw", *signature, arguments.get("filename", "kern.cu")) in held))


def start(device_id: int | None = None) -> dict[str, Any] | None:
    """Open a pool as wide as the usable cores and submit the plan; returns at once.

    :func:`submit` adds a package's own warm-up jobs to the same pool before
    :func:`close`; a job that compiles goes through the funnel, where it joins
    the single-flight table like any other request.
    """

    if not _state["installed"] or _state["replay"] is not None:
        return _state["replay"]
    device = _current_device() if device_id is None else int(device_id)
    if device is not None and _state["device_arch"] is None:
        import cupy

        with cupy.cuda.Device(device):
            _device_arch()
    arch = _device_arch()
    rows, source = load_plan()
    rows = [_retarget_cupy_program(row, arch) for row in rows]
    # A covered request will use the pack loader, which still validates its
    # complete contract. Avoid restoring a second image and its dependencies
    # merely to discard them here. If the loader rejects, the ordinary funnel
    # still tries the warm image and then compiles the exact request.
    held = _aot_coverage()
    _state["aot_coverage"] = held
    aot_cached = [row for row in rows if held and _aot_cached(row["arguments"], arch, held)]
    covered = {row["sha256"] for row in aot_cached}
    rows = [row for row in rows if row["sha256"] not in covered]
    images = _warm_images()
    image_cached = set()
    if images is not None:
        for row in rows:
            result = images.get(row["arguments"])
            if result is not None:
                future = concurrent.futures.Future()
                future.set_result(result)
                _flight[row["sha256"]] = future
                image_cached.add(row["sha256"])
    rows = [row for row in rows if row["sha256"] not in image_cached]
    workers = _workers()
    root = _cache_root()
    pending = [row for row in rows if not _cached(row, arch, root)]
    disk_cached = len(rows) - len(pending)
    record: dict[str, Any] = {"plan": str(_state["plan"]), "plan_source": source, "tag_text": _state["tag_text"],
                              "entries": len(rows) + len(aot_cached), "device_arch": arch, "workers": workers,
                              "pool_skipped_images": len(image_cached),
                              "pool_skipped_cached": disk_cached, "pool_skipped_aot": len(aot_cached),
                              "pool_compiled": 0, "pool_failed": 0, "pool_seconds": 0.0,
                              "started": time.monotonic(), "last_done": None}
    rows = pending
    _state["replay"] = record
    _state["device"] = device
    original = _state["original"]
    rows.sort(key=lambda row: -(row.get("seconds") or 0.0))

    def compile_row(arguments: dict[str, Any]):
        _use(device)
        started = time.perf_counter()
        try:
            result = original(arguments["source"], tuple(arguments["options"]),
                              arch if arguments["arch"] == "device" else arguments["arch"],
                              arguments["filename"], arguments["name_expressions"], None,
                              arguments["cache_in_memory"], None, arguments["method"])
        except BaseException:
            with _lock:
                record["pool_failed"] += 1
                record["last_done"] = time.monotonic()
            raise
        with _lock:
            record["pool_compiled"] += 1
            record["pool_seconds"] = round(record["pool_seconds"] + time.perf_counter() - started, 3)
            record["last_done"] = time.monotonic()
        return result

    pool = concurrent.futures.ThreadPoolExecutor(workers, thread_name_prefix=POOL_PREFIX)
    _state["pool"] = pool
    identity = _process_compiler_identity(arch)
    if identity is not None:
        cpu_pool = compiler_pool(workers)
        _state["compiler_pool"] = cpu_pool
        _state["worker_identity"] = identity
        record.update(compiler_executor="process", pool_cpu_seconds=0.0, pool_processes=[])
        if os.environ.get("WOOF_KERNEL_PLAN_TRACE") == "1":
            record["compile_intervals"] = []
            record["compile_intervals_scope"] = (
                "Successful process compiler calls only; same-host monotonic_ns; "
                "excludes imports/identity checks, queue wait, failures, inline and "
                "thread compiles. Not low-level NVRTC or GPU execution intervals.")
    else:
        record["compiler_executor"] = "thread"
    for row in rows:
        arguments = row["arguments"]
        if identity is not None and arguments.get("method") is None:
            target = arch if arguments["arch"] == "device" else arguments["arch"]
            # Keep GPU warm threads free while CPU compilers run. The funnel
            # cancels or joins the actual process job through this shared proxy.
            _submit_compile(arguments, target, record, key=row["sha256"])
        else:
            with _lock:
                if row["sha256"] not in _flight:
                    _flight[row["sha256"]] = pool.submit(compile_row, arguments)
    return record


def _submit_and_wait(arguments, arch, record):
    return _submit_compile(arguments, arch, record).result()


#: The thread that is capturing a static-plan request (:func:`ahead`).
_capture = threading.local()


class _Captured(BaseException):
    """A capturing thread's request, raised by the funnel instead of compiling it.

    A ``BaseException`` so no route's ``except Exception`` fallback can catch it
    and compile anyway.  ``key`` is None when nothing is to be compiled ahead:
    the route found its image on disk (:func:`refuse_while_capturing`) or the
    request is one the plan never replays (a log stream, jitify).
    """

    def __init__(self, arguments: dict[str, Any] | None = None, key: str | None = None):
        super().__init__("kernel-plan capture")
        self.arguments = arguments
        self.key = key


def _capturing() -> bool:
    return bool(getattr(_capture, "on", False))


def refuse_while_capturing() -> None:
    """Stop a capturing thread before it records anything a real load records.

    Called first by the per-process records a loader writes after its image
    exists (the physics-parameter guard's compiled set, the kernel manifest).
    A loader whose route found its image on disk reaches them without passing
    the funnel; under :func:`ahead` it must stop there, so the plan's warm-up
    never marks a module compiled or adds a manifest row the run did not make.
    """

    if _capturing():
        raise _Captured()


def ahead(call, *args: Any, **kwargs: Any) -> str:
    """Compile, on this thread, the NVRTC request ``call(*args, **kwargs)`` would make.

    THE STATIC PLAN (2026-10-17).  A recorded plan only exists from a
    configuration's second start; the first forecast on a fresh box still
    compiled its kernels one after another inside step 1 on one core (box
    door test on an RTX PRO 6000: noah 15.6 s, smag2d 12.5 s + 11.7 s,
    ysu_480 3.8 s, all serial).  ``call`` is a loader's own uncached factory
    (e.g. ``load_module.__wrapped__``): it runs on this thread up to its NVRTC
    request, where the funnel hands the request back instead of compiling
    (:class:`_Captured`).  This thread then compiles exactly that request and
    leaves the result in the single-flight table, where the run's own call --
    same source, same options, same architecture, hence the same key -- finds
    it.  The run's loader still writes its cache entry, its manifest row and
    its physics-parameter record itself, from the same bytes, at the moment it
    always did.  A request the run never makes is compiled for nothing; one
    whose bytes differ at run time (a parameter set declared later) misses the
    table and compiles inline as before.

    Returns ``"compiled"``, ``"known"`` (already compiled or compiling),
    ``"cached"`` (the route found its image; nothing to compile),
    ``"failed"`` or ``"off"``.
    """

    if not _state["installed"]:
        return "off"
    previous = _capturing()
    _capture.on = True
    captured = None
    try:
        call(*args, **kwargs)
    except _Captured as request:
        captured = request
    except Exception:
        return "failed"
    finally:
        _capture.on = previous
    record = _state["replay"]
    if captured is None or captured.key is None:
        with _lock:
            if record is not None:
                record["static_cached"] = record.get("static_cached", 0) + 1
        return "cached"
    arguments, key = captured.arguments, captured.key
    # As in recorded-plan replay, coverage avoids an unused image restore.
    arch = _device_arch()
    held = _state.get("aot_coverage")
    if held and _aot_cached(arguments, arch, held):
        with _lock:
            if record is not None:
                record["static_skipped_aot"] = record.get("static_skipped_aot", 0) + 1
        return "cached"  # the foreground loader still verifies the complete image contract
    images = _warm_images()
    result = None if images is None else images.get(arguments)
    if result is not None:
        future = concurrent.futures.Future()
        future.set_result(result)
        with _lock:
            _flight[key] = future
            if record is not None:
                record["static_skipped_images"] = record.get("static_skipped_images", 0) + 1
        return "cached"
    with _lock:
        if key in _flight:
            if record is not None:
                record["static_known"] = record.get("static_known", 0) + 1
            return "known"
        owner = concurrent.futures.Future()
        owner.set_running_or_notify_cancel()
        _flight[key] = owner
    started = time.perf_counter()
    try:
        target = arch if arguments["arch"] == "device" else arguments["arch"]
        if _state.get("compiler_pool") is not None and arguments.get("method") is None:
            result = _submit_and_wait(arguments, target, record)
        else:
            result = _state["original"](arguments["source"], tuple(arguments["options"]),
                                        target, arguments["filename"], arguments["name_expressions"], None,
                                        arguments["cache_in_memory"], None, arguments["method"])
    except BaseException as error:
        with _lock:
            if _flight.get(key) is owner:
                _flight.pop(key, None)
            if record is not None:
                record["static_failed"] = record.get("static_failed", 0) + 1
        owner.set_exception(error)
        return "failed"
    owner.set_result(result)
    if images is not None:
        images.put(arguments, result)
    with _lock:
        if record is not None:
            record["static_compiled"] = record.get("static_compiled", 0) + 1
            record["static_seconds"] = round(record.get("static_seconds", 0.0)
                                             + time.perf_counter() - started, 3)
    return "compiled"


def submit(job, *args: Any) -> concurrent.futures.Future | None:
    """Run ``job(*args)`` on the plan's pool (on the run's card); None if no pool."""

    pool = _state.get("pool")
    if pool is None:
        return None
    device = _state.get("device")

    def run():
        _use(device)
        try:
            return job(*args)
        finally:
            with _lock:
                if _state["replay"] is not None:
                    _state["replay"]["last_done"] = time.monotonic()

    return pool.submit(run)


#: The thread every compile worker is spawned from (:func:`compiler_pool`), per process.
_spawner: dict[str, Any] = {"pid": None, "thread": None, "jobs": None}
_spawner_lock = threading.Lock()


def _spawner_loop(jobs: queue.SimpleQueue) -> None:
    while True:
        job, done = jobs.get()
        if done.set_running_or_notify_cancel():
            try:
                done.set_result(job())
            except BaseException as error:
                done.set_exception(error)


def _on_spawner(job):
    """``job()`` on this process's spawner thread, which ends only when the process ends."""
    with _spawner_lock:
        thread = _spawner["thread"]
        if _spawner["pid"] != os.getpid() or thread is None or not thread.is_alive():
            jobs: queue.SimpleQueue = queue.SimpleQueue()
            thread = threading.Thread(target=_spawner_loop, args=(jobs,), name="kernel-compile-spawner",
                                      daemon=True)
            thread.start()
            _spawner.update(pid=os.getpid(), thread=thread, jobs=jobs)
        jobs = _spawner["jobs"]
    if threading.current_thread() is thread:
        return job()
    done: concurrent.futures.Future = concurrent.futures.Future()
    jobs.put((job, done))
    return done.result()


def compiler_pool(workers: int) -> concurrent.futures.ProcessPoolExecutor:
    """The spawn NVRTC compile pool, its workers bound to this process's life.

    THE BREAKAGE (2026-10-29 08:17-08:21Z; woof hex fix/hex-compile-worker-lifetime,
    this file is its twin): compile workers of runs that had ended -- here a WOOF kernel
    prewarm whose process was stopped by a signal (signal_report re-delivers it at the default
    action, so no exit hook runs) -- held memory until killed by PID; with hex's OOM-killed
    forecasts 62.5 GB in all, and the OOM killer took release containers and lanes.  Each worker
    now binds itself to this process (:func:`woof.kernel_compile_worker.bind_to_parent`: a
    parent-death SIGKILL plus a parent watch), so it ends when this process ends, however it ends.

    The parent-death signal fires when the THREAD that spawned the worker ends, and the pool spawns
    a worker lazily inside ``submit()`` on whichever thread submits (the funnel and :func:`ahead`
    submit from any thread).  So this pool's ``submit`` (``map`` goes through it) runs on one
    spawner thread that lives as long as the process.  The exit drop is registered here too.
    """
    import multiprocessing
    from woof.kernel_compile_worker import bind_to_parent

    pool = concurrent.futures.ProcessPoolExecutor(
        int(workers), mp_context=multiprocessing.get_context("spawn"),
        initializer=bind_to_parent, initargs=(os.getpid(),))
    direct = pool.submit

    def submit(fn, /, *args, **kwargs):
        return _on_spawner(functools.partial(direct, fn, *args, **kwargs))

    pool.submit = submit
    register = getattr(threading, "_register_atexit", None)
    if register is not None:
        # Registered after concurrent.futures' own exit hook, so it runs first.
        # The dict and the result queue, not the attributes: shutdown() sets the
        # attributes to None while the manager thread keeps the objects.
        register(functools.partial(_drop_at_exit, pool, getattr(pool, "_processes", None),
                                   getattr(pool, "_result_queue", None)))
    return pool


def _drop_at_exit(cpu_pool, processes=None, result_queue=None) -> None:
    """At interpreter exit, stop the compiler pool's workers instead of joining them.

    THE BREAKAGE: a hex forecast (the hl-3 headline run, 2026-10-24) wrote its receipt and its
    products, then sat in multiprocessing's exit join on one idle spawn compile worker of
    this pool until the worker was killed (twice, 16 min the first time). Unrequested
    compiles are pure waste at exit; a worker returns its image and writes nothing itself.
    Each step runs on its own so a shutdown that raises still terminates the workers.
    """
    try:
        cpu_pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    try:
        for process in list((processes or {}).values()):
            if process.is_alive():
                process.terminate()
    except Exception:
        pass
    # A worker stopped above while it was sending its compiled image leaves the pool's manager thread
    # blocked reading the rest of that result, and concurrent.futures' own exit hook (it runs next)
    # joins that thread with no timeout. The read ends only at end-of-file, and this process holds a
    # write end of the result pipe itself. THE BREAKAGE THIS ENDS (woof hex fix/hex-exit-hang 8137639c,
    # RRFS restart proofs, 2026-10-29; this file is its twin): finished forecasts (receipt
    # completed, every file written) never exited -- manager thread in the pipe read, main thread in
    # concurrent.futures' exit join, the call queue's feeder blocked writing to workers that were gone --
    # and held their card lease until killed by PID. Once the stopped workers are gone, closing this end
    # makes that read end-of-file: the manager takes the pool's broken path, which closes the call
    # queue (freeing its feeder) and ends the thread.
    try:
        if result_queue is not None:
            result_queue._writer.close()
    except Exception:
        pass


def close(*, wait: bool = False) -> None:
    """One shutdown owner drains workers; later callers join that owner."""
    with _lock:
        closer = _state.get("closer")
        if closer is None:
            pool = _state.pop("pool", None)
            cpu_pool = _state.get("compiler_pool")
            if pool is None and cpu_pool is None:
                return
            if pool is not None:
                pool.shutdown(wait=False)
            def finish():
                try:
                    # Captures already queued can still submit compiler work.
                    if pool is not None:
                        pool.shutdown(wait=True)
                    if cpu_pool is not None:
                        cpu_pool.shutdown(wait=True)
                finally:
                    with _lock:
                        if _state.get("compiler_pool") is cpu_pool:
                            _state.pop("compiler_pool", None)
                        _state.pop("closer", None)
            closer = threading.Thread(target=finish, name="kernel-plan-close", daemon=True)
            _state["closer"] = closer
            closer.start()
    if wait:
        closer.join()


def save() -> None:
    """Merge this run's requests into its tag's plan (atomic rewrite)."""

    try:
        images = _state.get("warm_images")
        if images is not None:
            images.save()
        path = _state["plan"]
        if path is None:
            return
        with _lock:
            fresh = copy.deepcopy(_requested)
        if not fresh:
            return
        document_runs = 0
        old: dict[str, dict[str, Any]] = {}
        if path.is_file():
            try:
                document_runs = int(json.loads(path.read_text(encoding="utf-8")).get("runs", 0))
            except (OSError, ValueError, TypeError, AttributeError):
                document_runs = 0
            old = {row["sha256"]: row for row in _read(path)}
        run = document_runs + 1
        entries: dict[str, dict[str, Any]] = {}
        for key, row in old.items():
            if run - int(row.get("last_run", 0)) < KEEP_RUNS:
                entries[key] = row
        for key, value in fresh.items():
            seconds = value["seconds"]
            if seconds is None and key in old:
                seconds = old[key].get("seconds")
            entries[key] = {"sha256": key, "arguments": value["arguments"], "seconds": seconds,
                            "last_run": run}
            # Each architecture's cache entry is kept: a plan baked on two cards skips on both.
            known = old.get(key, {}).get("cache_entries")
            merged = dict(known) if isinstance(known, dict) else {}
            merged.update(value.get("cache_entries") or {})
            route = value.get("route") or old.get(key, {}).get("route")
            if merged:
                entries[key]["cache_entries"] = merged
            if route:
                entries[key]["route"] = route
        rows = sorted(entries.values(), key=lambda row: -int(row["last_run"]))[:MAX_ENTRIES]
        document = {"schema": SCHEMA, "compiler": _state["identity"], "tag_text": _state["tag_text"],
                    "runs": run, "entries": rows}
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".kernel-plan.", dir=path.parent)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        pass


def rebind(tag_text: str) -> bool:
    """Save this process's plan and bind a new configuration tag (one worker, several profiles).

    The breakage this prevents: a ``warm-kernels`` worker that warmed several
    profiles wrote every profile's kernels under its first profile's tag
    (11 plan files for 68 profile runs on a 4090), so a later run of the other
    profiles found no plan of its own.
    """

    if not _state["installed"]:
        return False
    text = str(tag_text)
    if text == _state["tag_text"]:
        return True
    # Captures still queued under this tag can submit compiler work. Drain
    # them before saving and replacing the state their callbacks use.
    close(wait=True)
    save()
    images = _state.pop("warm_images", None)
    if images is not None:
        images.close()
    with _lock:
        _requested.clear()
        _state["counts"] = {}
    tag = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    _state.update(tag=tag, tag_text=text, plan=_state["root"] / f"{tag}.json", replay=None)
    return True


def receipt() -> dict[str, Any]:
    with _lock:
        replay = None if _state["replay"] is None else {
            key: value for key, value in _state["replay"].items() if key not in ("started", "last_done")}
        if replay is not None and _state["replay"]["last_done"] is not None:
            replay["pool_wall_seconds"] = round(_state["replay"]["last_done"] - _state["replay"]["started"], 3)
        if replay is not None and "compile_intervals" in replay:
            replay["compile_intervals"] = [dict(row) for row in replay["compile_intervals"]]
        counts = dict(_state["counts"])
        requested = len(_requested)
    if "inline_seconds" in counts:
        counts["inline_seconds"] = round(counts["inline_seconds"], 3)
    return {"schema": SCHEMA, "enabled": enabled(), "installed": _state["installed"],
            "requested_this_run": requested, "replay": replay, "calls": counts,
            "compiler_metadata": (_state["compiler_metadata"].receipt() if _state.get("compiler_metadata") is not None else None),
            "warm_images": (_state["warm_images"].receipt() if _state.get("warm_images") is not None else None)}


__all__ = ["CUPY_ROUTE", "SCHEMA", "ahead", "cache_entry", "close", "compiler_identity", "default_tag_text", "enabled",
           "install", "load_plan", "rebind", "receipt", "refuse_while_capturing", "save", "start", "submit"]
