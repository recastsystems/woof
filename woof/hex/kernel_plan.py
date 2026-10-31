"""Compile a run's whole CUDA kernel set at once, on every core, before step 1.

THE DEFECT (2026-10-16, owner ruling: "no single-CPU path on the critical path").  On
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
* **Compile ahead.** At the next start supported cubin requests run in CPU
  processes as wide as the usable cores, longest compile first. The measured
  CUDA 12 replay held about one core even with 48 active compiler threads;
  separate CPU compiler processes remove that serialization. Other compiler
  routes retain the thread executor. This overlaps host input preparation.
* **Single flight.**  A request that arrives while its compile is running
  waits for it; one still queued is taken back and compiled by the caller, as
  before; one already finished is returned at once.  Two threads never
  compile the same request twice.

WHY THE OUTPUT IS UNCHANGED.  The wrapper returns exactly what NVRTC returned
for the identical inputs, and every route still writes its own cache entry,
under its own key, from those bytes -- nothing else about a route changes.
Two inputs are not part of the match when they cannot change the image: the
program name and whether CuPy wrote the source to a temporary file (measured
on NVRTC 13: one image for every name and path).  When they CAN change it --
the source uses ``assert`` or ``__FILE__``, or the options ask for line or
debug info -- they are part of the match.  (Such images embed CuPy's random
temporary path, so two serial compiles of them already differ.)  The
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
#: ``off`` keeps a first run with no usable pack from compiling the shipped inventory ahead.
CATALOG_ENVIRONMENT = "WOOF_KERNEL_PLAN_CATALOG"
MAX_ENTRIES = 4000
#: Entries not requested in this many runs of their tag leave the plan.
KEEP_RUNS = 4
POOL_PREFIX = "kernel-plan"
_MARK = "__gpuwm_kernel_plan__"
_NAME_SENSITIVE_OPTIONS = ("-G", "--device-debug", "-lineinfo", "--generate-line-info")
_capture = threading.local()


class _CapturedRequest(BaseException):
    """Stop a static factory before compilation or loader side effects."""
    def __init__(self, arguments=None, key=None):
        super().__init__("Hex static compiler capture")
        self.arguments, self.key = arguments, key


_lock = threading.Lock()
_flight: dict[str, concurrent.futures.Future] = {}
#: Raw NVRTC units (woof.hex.cuda_backend.aot.raw_image) never reach the funnel; their pool compiles wait here.
RAW_ROUTE = "raw-nvrtc"
_raw_flight: dict[str, concurrent.futures.Future] = {}
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
    return {"cupy": str(cupy.__version__), "nvrtc": list(compiler._get_nvrtc_version()),
            "cache_key": str(compiler._get_cupy_cache_key()),
            "use_ptx": bool(getattr(compiler, "_use_ptx", False)),
            "use_pch": bool(getattr(compiler, "_use_pch", False))}


#: Options whose value only names where this run's data comes from or what date it starts.
#: The flag itself stays in the tag (an IAU run compiles IAU kernels); its value does not.
#: Breakage it prevents: every DA member and cycle passes its own --init/--iau-analysis path
#: and start time, so each wrote a new ~22 MB plan + manifest that no later run could find.
#: A box-wide kernel cache then filled with dead files and every member replayed no plan.
_INPUT_OPTIONS = ("--init", "--init-source", "--iau-analysis", "--iau-increment", "--start-time")
_OUTPUT_OPTIONS = ("--out", "--outdir", "--output", "--scratch", "--progress-json")


def default_tag_text(argv: Iterable[str] | None = None) -> str:
    """Bind shape and options; output destinations and input data values do not affect the plan."""
    words = [str(word) for word in (sys.argv[1:] if argv is None else argv)]
    kept = []
    skip = None
    for word in words:
        if skip is not None:
            if skip:
                kept.append("<input>")
            skip = None
            continue
        if word in _OUTPUT_OPTIONS or word in _INPUT_OPTIONS:
            skip = word in _INPUT_OPTIONS
            if skip:
                kept.append(word)
            continue
        if any(word.startswith(option + "=") for option in _OUTPUT_OPTIONS):
            continue
        name = word.split("=", 1)[0]
        if "=" in word and name in _INPUT_OPTIONS:
            kept.extend((name, "<input>"))
            continue
        kept.append(word)
    environment = {name: os.environ.get(name, "") for name in
                   ("WOOF_HEX_NUMERICS", "GPUWM_NUMERICS", "GPUWM_WRF_EXACT", "CUPY_CUDA_COMPILE_WITH_DEBUG")}
    return json.dumps({"argv": kept, "environment": environment}, sort_keys=True)


def _name_sensitive(source: str, options: tuple[str, ...]) -> bool:
    if "assert" in source or "__FILE" in source:
        return True
    return any(str(option).split("=", 1)[0] in _NAME_SENSITIVE_OPTIONS for option in options)


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
                "name_expressions": arguments["name_expressions"], "method": arguments["method"]}
    if _name_sensitive(source, options):
        material["filename"] = arguments["filename"]
        material["cache_in_memory"] = arguments["cache_in_memory"]
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()


def _raw_key(source: str, options, arch: str, filename: str) -> str:
    """What decides a raw NVRTC image: source, exact options, target and program name (always passed)."""

    material = {"route": RAW_ROUTE, "source": source, "options": [str(option) for option in options],
                "arch": str(arch), "filename": str(filename)}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()


def raw_planned(source: str, options, arch, filename: str):
    """The image the plan's pool compiled for this raw NVRTC request, or None for the caller's own path.

    THE BREAKAGE THIS ENDS (an RTX 5090 host and V8X6000, 2026-10-29): with no prebuilt pack, raw NVRTC
    units (Grell-Freitas on every MPAS convection_permitting run) were left out of the catalog the
    pool compiles ahead, so each compiled in the foreground on one core at its first call while the
    card waited.  A row still queued is taken back and compiled by the caller exactly as before; a
    running or finished one is waited for.  The pool runs the very compile the caller would
    (:func:`woof.hex.cuda_backend.aot.raw_nvrtc`), so the image is the same bytes.
    """

    if not _raw_flight:
        return None
    try:
        target = str(arch).removeprefix("sm_")
        key = _raw_key(source, options, "device" if target == _state.get("device_arch") else target, filename)
    except Exception:
        return None
    with _lock:
        future = _raw_flight.get(key)
    if future is None:
        return None
    if future.cancel():
        with _lock:
            _raw_flight.pop(key, None)
            _count("raw_taken_back")
        return None
    try:
        image = future.result()
    except Exception:
        with _lock:
            _count("raw_failed_then_inline")
        return None
    if not isinstance(image, bytes):
        return None
    with _lock:
        _count("raw_served")
    return image


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


def _count(name: str, amount: float = 1) -> None:
    counts = _state["counts"]
    counts[name] = counts.get(name, 0) + amount


def _copy(result):
    image, mapping = result
    return image, (dict(mapping) if isinstance(mapping, dict) else mapping)


def install(plan_dir: str | Path | None = None, tag_text: str | None = None) -> bool:
    """Wrap the NVRTC funnel (once per process) and bind this run's plan file."""

    if _state["installed"]:
        return True
    if not enabled():
        return False
    compiler = _compiler()
    # Check ownership before the warm installer replaces this callable. Its
    # wrapper has no plan marker, so checking afterward can claim a second
    # owner when the engine installed its funnel first.
    if getattr(compiler._compile_using_nvrtc_no_warning, _MARK, False):
        return False
    # Install below the plan wrapper so foreground and thread-pool compilers
    # both load exact artifacts, including requests without CuPy disk entries.
    if getattr(compiler, "__name__", None) == "cupy.cuda.compiler":
        from . import kernel_warm
        try:
            kernel_warm.install(tag=default_tag_text() if tag_text is None else str(tag_text))
        except (ImportError, OSError, ValueError) as error:
            import warnings
            warnings.warn(f"hex warm pack unavailable: {error}; ordinary compilation retained")
    original = compiler._compile_using_nvrtc_no_warning
    if getattr(original, _MARK, False):
        return False  # the other package's copy already owns the funnel
    directory = plan_dir or os.environ.get(DIRECTORY_ENVIRONMENT)
    if directory is None:
        directory = Path.home() / ".cache" / "woof" / "kernel-plans"
    text = default_tag_text() if tag_text is None else str(tag_text)
    try:
        identity = compiler_identity()
    except Exception:
        return False
    identity_digest = hashlib.sha256(json.dumps([SCHEMA, identity], sort_keys=True).encode()).hexdigest()[:20]
    tag = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    _state.update(identity=identity, root=Path(directory).expanduser() / identity_digest,
                  tag=tag, tag_text=text, original=original)
    _state["plan"] = _state["root"] / f"{tag}.json"

    def funnel(source, options=(), arch=None, filename="kern.cu", name_expressions=None,
               log_stream=None, cache_in_memory=False, jitify=None, method=None):
        def call():
            return original(source, options, arch, filename, name_expressions, log_stream,
                            cache_in_memory, jitify, method)

        capturing = bool(getattr(_capture, "on", False))
        if log_stream is not None or jitify or not isinstance(source, str):
            if capturing:
                raise _CapturedRequest()
            return call()
        try:
            arguments = _arguments(source, options, arch, filename, name_expressions,
                                   cache_in_memory, method)
            key = _key(arguments)
        except Exception:
            if capturing:
                raise _CapturedRequest() from None
            return call()
        if capturing:
            if method is not None:
                raise _CapturedRequest()
            raise _CapturedRequest(arguments, key)
        try:
            entry = _entry_of(filename, cache_in_memory)
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
            _count("served")
            return _copy(result)
        started = time.perf_counter()
        try:
            result = call()
        except BaseException as error:
            with _lock:
                _flight.pop(key, None)
            owner.set_exception(error)
            raise
        seconds = time.perf_counter() - started
        owner.set_result(result)
        with _lock:
            _count("inline_compiles")
            _count("inline_seconds", seconds)
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
    """Read only this configuration's plan, without scanning unrelated plans.

    An unknown or damaged tag used to load the union of every configuration
    for this compiler. That scanned other shapes and could compile their
    uncached kernels although the current run never requests them. A missing
    exact plan now leaves warm-up to the existing package callbacks and pack
    loader; actual misses still use the normal single-flight compiler path.
    """

    path = _state["plan"]
    rows = _read(path) if path is not None and path.is_file() else []
    return rows, ("tag" if rows else "none")


def _workers() -> int:
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    limit = os.environ.get(WORKERS_ENVIRONMENT, "").strip()
    if limit.isdigit() and int(limit) > 0:
        cores = min(cores, int(limit))
    return max(1, cores)


class _ProcessCompileFuture(concurrent.futures.Future):
    """Cancel a queued compiler only when its actual process job cancels."""
    def __init__(self, pending, record):
        super().__init__()
        self.pending = pending
        self.record = record

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
            worker["entries"] += 1
            worker["cpu_seconds"] = round(worker["cpu_seconds"] + timing["cpu_seconds"], 6)
        self.set_result(result)


def _process_compiler_identity(arch):
    """Use processes for the measured CuPy cubin route, retain other routes."""
    try:
        compiler = _compiler()
        if getattr(compiler, "__name__", None) != "cupy.cuda.compiler" or compiler._use_ptx:
            return None
        from cupy_backends.cuda.libs import nvrtc
        if int(str(arch).removeprefix("sm_")) not in nvrtc.getSupportedArchs():
            return None
        from .cuda_backend.aot import compiler_identity as identity
        return identity()
    except (ImportError, AttributeError, ValueError):
        return None


def _aot_coverage(pack):
    """A scheduling hint only. Every actual load keeps the full AOT contract.

    A replay must not compile an image the upstream loader already holds.
    A false positive here can only cause the normal on-demand compile, never
    serve an image: source, compiler, metadata and image checks remain there.
    """
    if pack.reason:
        return set()
    held = {}
    for row in pack.rows:
        if row.get("compiler_route") == "raw-nvrtc":
            continue
        options = list(row["options"])
        if row.get("compiler_route") != "direct-nvrtc":
            options.append("-ftz=true")
        names = (row.get("compile_request") or {}).get("name_expressions")
        signature = (row["source_sha256"], tuple(options), tuple(names or ()), row["arch"])
        held.setdefault(signature, set()).add((row.get("compile_request") or {}).get("filename"))
    return held


def _aot_cached(arguments, arch, held):
    from .cuda_backend.aot import effective_options, portable_options, source_hash
    options = tuple(effective_options(arguments["options"]))
    if arguments.get("method") is not None:
        return False
    target = arch if arguments["arch"] == "device" else arguments["arch"]
    signature = (source_hash(arguments["source"]), tuple(portable_options(options)),
                 tuple(arguments.get("name_expressions") or ()), "sm_" + str(target).removeprefix("sm_"))
    if signature not in held:
        return False
    # Assertion/file/debug requests must retain the bound program name too.
    return (not _name_sensitive(arguments["source"], options)
            or arguments["filename"] in held[signature])


def _pack_serves(pack, arch: str) -> bool:
    """Whether the verified prebuilt pack holds images for this compiler and card."""
    return not pack.reason and ("sm_" + str(arch).removeprefix("sm_")) in pack._architectures


def _catalog_rows(arch: str) -> list[dict[str, Any]]:
    """The shipped kernel inventory as plan rows, for a run with no plan and no usable pack.

    THE BREAKAGE THIS ENDS (2026-10-24): a first run on
    a new box whose loaded NVRTC matched no prebuilt pack had no plan either,
    so every module compiled in the foreground on one core (about 4 min before
    step 1 on a global run).  The inventory is the exact request set the
    shipped packs are built from (:func:`woof.hex.cuda_backend.aot_build.inventory`,
    this process's numerics mode), formed into the funnel's own arguments the
    way the pack builder calls the compiler, so the foreground finds each one in
    the single-flight table.  Raw NVRTC units bypass the funnel: their rows carry
    ``route: raw-nvrtc`` and wait in :data:`_raw_flight`, where
    :func:`raw_planned` hands their images to the unit's own raw call (2026-10-29:
    left out, Grell-Freitas compiled on one core at its first call on every cold
    MPAS-route start).  A row the foreground never requests costs pool time only.
    """
    from .cuda_backend.aot import compiler_identity, local_options
    from .cuda_backend.aot_build import cached_program_filename, inventory
    compiler = _compiler()
    identity = compiler_identity()
    debug = compiler._get_bool_env_variable("CUPY_CUDA_COMPILE_WITH_DEBUG", False)
    target = str(arch).removeprefix("sm_")
    rows = []
    for row in inventory(compiler_family=identity["nvrtc"][0], compiler=identity):
        route = row.get("compiler_route")
        if route == RAW_ROUTE:
            # The raw call adds nothing to its options (no CuPy include, --ftz or debug flags).
            request = row.get("compile_request") or {}
            arguments = {"source": row["source"], "options": [str(o) for o in local_options(row["options"])],
                         "arch": "device", "filename": str(request.get("filename") or "kern.cu"),
                         "cache_in_memory": False, "name_expressions": None, "method": None}
            plan_row = {"sha256": _raw_key(arguments["source"], arguments["options"], "device",
                                           arguments["filename"]),
                        "arguments": arguments, "seconds": None, "route": RAW_ROUTE}
            if row.get("physics_family"):
                plan_row["physics_family"] = row["physics_family"]
            rows.append(plan_row)
            continue
        request = row.get("compile_request") or {}
        options = tuple(local_options(row["options"]))
        cupy_route = route != "direct-nvrtc"
        if cupy_route and not str(row.get("module_key", "")).startswith("cupy.cache."):
            # A KernelCache declaration reaches NVRTC through CuPy's RawModule,
            # which wraps its options in CuPy's own include directories (a
            # recorded cupy.cache recipe already carries them). On
            # 2026-10-24, without them the 14 dycore and physics modules never
            # matched and compiled in the foreground.
            from cupy._core import core as cupy_core
            options = tuple(cupy_core.assemble_cupy_compiler_options(options))
        if cupy_route:
            options += ("-ftz=true",)
            if debug:
                options += ("--device-debug", "--generate-line-info")
            try:
                filename = cached_program_filename(row, options, target)
            except Exception:
                continue  # headers differ from this CuPy: the route compiles it itself
        else:
            filename = request.get("filename") or "kern.cu"
        arguments = {"source": row["source"], "options": list(options), "arch": "device",
                     "filename": str(filename), "cache_in_memory": bool(request.get("cache_in_memory", False)),
                     "name_expressions": (None if request.get("name_expressions") is None
                                          else [str(n) for n in request["name_expressions"]]),
                     "method": None}
        plan_row = {"sha256": _key(arguments), "arguments": arguments, "seconds": None}
        if row.get("physics_family"):
            plan_row["physics_family"] = row["physics_family"]
        if cupy_route and not arguments["cache_in_memory"]:
            match = _CUPY_PROGRAM.match(os.path.basename(str(filename)))
            if match is not None:
                plan_row.update(route=CUPY_ROUTE, cache_entries={target: match.group(1)})
        rows.append(plan_row)
    unique = {}
    for row in rows:
        unique.setdefault(row["sha256"], row)
    return list(unique.values())


def prioritize_physics(config) -> None:
    """Compile selected scheme families first; leave out another physics table's own families.

    A cold catalog holds hundreds of unrelated programs. The selected
    physics used to wait behind them while the mesh was ready. Every image
    keeps its exact source/options identity.

    THE BREAKAGE THE SKIP ENDS (an RTX 5090 host, 2026-10-29 07:55Z): a cold strict hl-3
    (MPAS v8.4.2 convection_permitting) compiled the RRFS fork's catalog rows
    too, and two pool workers held 35-37 GB of host RAM each on a 123 GB box
    that was down to 4 GB free (the OOM kills of that morning).  A run on the
    v8.4.2 table never calls the RRFS fork's modules, so their rows
    (:func:`_rrfs_only`) are not compiled ahead; were one ever requested it
    compiles at that call as before.  Families are NOT matched to slots for
    the skip: a slot's scheme uses images of other families (thompson_aero
    runs thompson_workspace, sf_mynn runs sfclay_mm5; measured on V8X6000).
    """
    from .physics_suites import load_table

    families = set()
    selectors = getattr(config, "selectors", None) or {}
    if selectors:
        slots = load_table()[getattr(config, "table_key", "mpasphys_v842")]["slots"]
        active = getattr(config, "active", lambda slot: True)
        for slot, scheme in selectors.items():
            if not active(slot):
                continue
            descriptor = slots[slot].get("cuda_entries", {}).get(scheme) or {}
            module = str(descriptor.get("module", "")).rsplit(".", 1)[-1]
            if module:
                families.add(re.sub(r"_v[0-9]+$", "", module))
    with _lock:
        _state["priority_families"] = tuple(sorted(families))
        _state["skip_rrfs_only"] = (bool(selectors)
                                    and not str(getattr(config, "table_key", "mpasphys_v842")).startswith("mpasphys_rrfs"))


def _family_selected(family: str, selected) -> bool:
    """Whether a catalog physics family belongs to a selected scheme's module stem.

    Inventory families name the module's image (thompson_workspace for thompson_v842, noah_seaice for
    seaice_v842, bl_mynnedmf for rrfs.bl_mynnedmf_c942418, rrfs_sfclay_mm5 for rrfs sfclay_mm5), so the
    match is generous at '_' boundaries: an extra row costs pool time, a missing one a foreground compile.
    """
    return any(family == name or family.startswith(name + "_") or name.startswith(family + "_")
               or family.endswith("_" + name) or ("_" + name + "_") in family for name in selected)


def _rrfs_only(family: str) -> bool:
    """A catalog family only the RRFS fork's modules (woof.hex.mpasphys.rrfs) compile (aot_build.physics_inventory)."""
    return family == "bl_mynnedmf" or family.startswith(("rrfs_", "tempo_rrfs"))


def _unselected(row) -> bool:
    """A catalog physics row of the RRFS fork on a run of another physics table (see prioritize_physics)."""
    family = row.get("physics_family", "")
    return bool(_state.get("skip_rrfs_only") and family and row.get("last_run") is None and _rrfs_only(family))


def _compile_priority(row):
    selected = _state.get("priority_families", ())
    family = row.get("physics_family", "")
    if not selected or row.get("last_run") is not None:
        tier = 0
    elif family and _family_selected(family, selected):
        tier = 0
    else:
        tier = 2 if family else 1
    # Longest first: a recorded compile time, else the source length (a catalog row has no time yet), so the
    # minutes-long physics images do not start last behind hundreds of small kernels.
    seconds = row.get("seconds")
    return (tier, -(seconds or 0.0),
            0 if seconds is not None else -len(str((row.get("arguments") or {}).get("source", ""))))


def start(device_id: int | None = None) -> dict[str, Any] | None:
    """Open a pool as wide as the usable cores and submit the plan; returns at once.

    :func:`submit` adds a package's own GPU warm-up jobs to its thread pool before
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
    workers = _workers()
    root = _cache_root()
    pack = None
    catalog_error = None
    is_cupy = getattr(_compiler(), "__name__", None) == "cupy.cuda.compiler"
    if (is_cupy and os.environ.get(CATALOG_ENVIRONMENT, "").strip().lower()
            not in ("off", "0", "no", "false")):
        # A first run or an incomplete plan. When the prebuilt
        # pack cannot serve this compiler or card, compile the shipped
        # inventory ahead on every core instead of one module at a time.
        from .cuda_backend.aot import Pack
        pack = Pack()
        if pack.reason != "aot_disabled" and not _pack_serves(pack, arch):
            try:
                # A failed forecast can leave a short but valid plan. When
                # this compiler has no matching pack, replaying only that
                # prefix leaves the rest compiling serially in step 1.
                # A matching pack is checked by the release inventory gate;
                # scanning that inventory here would compile unused recipes
                # and slow the ordinary first run on a supported compiler.
                known = {row["sha256"] for row in rows}
                catalog = [row for row in _catalog_rows(arch) if row["sha256"] not in known]
                if catalog:
                    source = (source + "+catalog") if rows else "catalog"
                    rows = [*rows, *catalog]
            except Exception as error:
                catalog_error = f"{type(error).__name__}: {error}"
    raw_rows = [row for row in rows if row.get("route") == RAW_ROUTE]
    entries = len(rows)
    rows = [row for row in rows if row.get("route") != RAW_ROUTE]
    pending = [row for row in rows if not _cached(row, arch, root)]
    disk_cached = len(rows) - len(pending)
    # A covered request will be served by the upstream AOT loader. Check its
    # validated pack before binding header trees for an artifact lookup that
    # cannot be used. An invalid/missing pack leaves every request on the
    # ordinary warm-artifact or compiler path; the AOT hint serves no image.
    held = set()
    if pending and is_cupy:
        from .cuda_backend.aot import Pack
        held = _aot_coverage(pack if pack is not None else Pack())
    aot_cached = [row for row in pending if held and _aot_cached(row["arguments"], arch, held)]
    covered = {row["sha256"] for row in aot_cached}
    pending = [row for row in pending if row["sha256"] not in covered]
    restored = 0
    if getattr(_compiler(), "__name__", None) == "cupy.cuda.compiler":
        from . import kernel_warm
        missing = []
        for row in pending:
            arguments = row["arguments"]
            target = arch if arguments["arch"] == "device" else arguments["arch"]
            image = kernel_warm.restore(arguments, target)
            if image is None:
                missing.append(row)
            else:
                future = concurrent.futures.Future()
                future.set_result(image)
                with _lock:
                    _flight.setdefault(row["sha256"], future)
                restored += 1
        pending = missing
    # A raw row the box's warm store already holds is read by the unit's own raw call: nothing to compile.
    raw_pending = []
    raw_restored = 0
    if raw_rows:
        from . import kernel_warm
        for row in raw_rows:
            arguments = row["arguments"]
            target = arch if arguments["arch"] == "device" else arguments["arch"]
            if kernel_warm.raw_cached(arguments["source"], arguments["options"], target, arguments["filename"]):
                raw_restored += 1
            else:
                raw_pending.append(row)
    record: dict[str, Any] = {"plan": str(_state["plan"]), "plan_source": source, "tag_text": _state["tag_text"],
                              "entries": entries, "device_arch": arch, "workers": workers,
                              "pool_skipped_cached": disk_cached, "pool_skipped_aot": len(aot_cached),
                              "pool_restored_warm": restored + raw_restored,
                              "pool_raw": len(raw_pending),
                              "pool_compiled": 0, "pool_failed": 0, "pool_seconds": 0.0,
                              "started": time.monotonic(), "last_done": None}
    if pack is not None:
        record["pack_reason"] = pack.reason
    if catalog_error is not None:
        record["catalog_error"] = catalog_error
    rows = [row for row in pending + raw_pending if not _unselected(row)]
    record["pool_skipped_rrfs_only"] = len(pending) + len(raw_pending) - len(rows)
    record["pool_raw"] = sum(1 for row in rows if row.get("route") == RAW_ROUTE)
    _state["replay"] = record
    _state["device"] = device
    original = _state["original"]
    record["priority_families"] = list(_state.get("priority_families", ()))
    rows.sort(key=_compile_priority)

    # Submit the entire plan before entering CUDA/NVRTC. On a cold MPAS
    # route the first worker can initialize the compiler while submit is still
    # feeding the executor, starving the remaining worker launches.
    ready = threading.Event()
    active = 0
    thread_rows: dict[str, dict[str, Any]] = {}
    record.update(pool_cpu_seconds=0.0, pool_peak_active=0, pool_threads=[])

    def compile_row(arguments: dict[str, Any]):
        nonlocal active
        ready.wait()
        _use(device)
        started = time.perf_counter()
        cpu_started = time.thread_time()
        name = threading.current_thread().name
        with _lock:
            active += 1
            record["pool_peak_active"] = max(record["pool_peak_active"], active)
        try:
            return compile_arguments(arguments)
        finally:
            cpu = time.thread_time() - cpu_started
            with _lock:
                active -= 1
                row = thread_rows.setdefault(name, {"thread": name, "entries": 0, "cpu_seconds": 0.0})
                row["entries"] += 1
                row["cpu_seconds"] = round(row["cpu_seconds"] + cpu, 6)
                record["pool_cpu_seconds"] = round(record["pool_cpu_seconds"] + cpu, 6)
                record["pool_threads"] = list(thread_rows.values())

    def compile_raw_row(arguments: dict[str, Any]):
        ready.wait()
        from .kernel_compile_worker import raw_compile
        started = time.perf_counter()
        try:
            result = raw_compile(arguments, arch if arguments["arch"] == "device" else arguments["arch"])
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

    def compile_arguments(arguments: dict[str, Any]):
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
    identity = _process_compiler_identity(arch) if rows else None
    if identity is not None and all(row["arguments"].get("method") is None for row in rows):
        from .kernel_compile_worker import compile_request
        from .cuda_backend.aot import effective_options
        cpu_pool = compiler_pool(workers)
        _state["compiler_pool"] = cpu_pool
        record.update(compiler_executor="process", pool_processes=[])
        record.pop("pool_peak_active", None)
        from . import kernel_warm
        for row in rows:
            if row.get("route") == RAW_ROUTE:
                from .kernel_compile_worker import compile_raw_request
                target = arch if row["arguments"]["arch"] == "device" else row["arguments"]["arch"]
                with _lock:
                    if row["sha256"] in _raw_flight:
                        continue
                    pending = cpu_pool.submit(compile_raw_request, dict(row["arguments"]), target, identity,
                                              kernel_warm.worker_config())
                    future = _ProcessCompileFuture(pending, record)
                    _raw_flight[row["sha256"]] = future
                pending.add_done_callback(future.completed)
                continue
            arguments = dict(row["arguments"])
            arguments["options"] = list(effective_options(arguments["options"]))
            target = arch if arguments["arch"] == "device" else arguments["arch"]
            with _lock:
                if row["sha256"] in _flight:
                    continue
                pending = cpu_pool.submit(compile_request, arguments, target, identity,
                                          kernel_warm.worker_config())
                future = _ProcessCompileFuture(pending, record)
                _flight[row["sha256"]] = future
            pending.add_done_callback(future.completed)
        return record
    record["compiler_executor"] = "thread"
    try:
        with _lock:
            for row in rows:
                if row.get("route") == RAW_ROUTE:
                    if row["sha256"] not in _raw_flight:
                        _raw_flight[row["sha256"]] = pool.submit(compile_raw_row, row["arguments"])
                elif row["sha256"] not in _flight:
                    _flight[row["sha256"]] = pool.submit(compile_row, row["arguments"])
    finally:
        ready.set()
    return record


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

    THE BREAKAGE (an RTX 5090 host, 2026-10-29 08:17-08:21Z): compile workers of forecasts that had ended
    (OOM-killed, or stopped by a signal whose default action runs no exit hook) held 62.5 GB until
    killed by PID, and the OOM killer took release containers and lanes.  Each worker now binds
    itself to this process (:func:`woof.hex.kernel_compile_worker.bind_to_parent`: a parent-death
    SIGKILL plus a parent watch), so it ends when this process ends, however it ends.

    The parent-death signal fires when the THREAD that spawned the worker ends, and the pool spawns
    a worker lazily inside ``submit()`` on whichever thread submits: this plan is submitted from the
    short-lived hex-prewarm-setup thread, and a worker bound there would be killed mid-compile when
    setup returned.  So this pool's ``submit`` (``map`` goes through it) runs on one spawner thread
    that lives as long as the process.  The exit drop is registered here too, so every compile pool
    (this plan's, woof.hex.kernel_static's) stops its workers at a normal exit.
    """
    import multiprocessing
    from .kernel_compile_worker import bind_to_parent

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
        # The dict and the result queue, not the attributes: shutdown() (close() below)
        # sets the attributes to None while the manager thread keeps the objects.
        register(functools.partial(_drop_at_exit, pool, getattr(pool, "_processes", None),
                                   getattr(pool, "_result_queue", None)))
    return pool


def _drop_at_exit(cpu_pool, processes=None, result_queue=None) -> None:
    """At interpreter exit, drop the pool compiles nobody is waiting for.

    A worker returns its image to this process and writes nothing itself, so
    an unrequested compile still running when the run ends is pure waste.
    THE BREAKAGE THIS ENDS (2026-10-24): a finished cold forecast held
    its exit for 70-90 s while one worker compiled, on one core, a catalog
    module the forecast never asked for.
    """
    # Each step on its own: a shutdown that raises (its wakeup pipe already closed) must not
    # skip the terminate, or an idle worker that never got its sentinel holds the exit in
    # multiprocessing's join (the hl-3 headline run, 2026-10-24: the forecast had written its receipt
    # and waited 16 min on one idle spawn worker).
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
    # A worker whose SIGTERM is blocked or ignored (a signal mask or disposition inherited from the
    # process that spawned it) outlives terminate() and still holds the exit. Breakage this ends
    # (hex 0.4.7 pass 1, HXCI battery 3.11, 2 of 3 runs): test_a_failing_shutdown_still_terminates_an_idle_worker
    # found the idle spawn worker alive 10 s after terminate(). The worker writes nothing itself, so
    # SIGKILL loses nothing.
    try:
        for process in list((processes or {}).values()):
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join(5)
    except Exception:
        pass
    # A worker stopped above while it was sending its compiled image leaves the pool's manager thread
    # blocked reading the rest of that result, and concurrent.futures' own exit hook (it runs next)
    # joins that thread with no timeout. The read ends only at end-of-file, and this process holds a
    # write end of the result pipe itself. THE BREAKAGE THIS ENDS (RRFS restart proofs, an RTX 5090 host,
    # 2026-10-29): finished forecasts (receipt "completed": true, every file written) never exited --
    # manager thread in the pipe read, main thread in concurrent.futures' exit join, the call queue's
    # feeder blocked writing to workers that were gone -- and held their card lease until killed by PID.
    # With the workers gone, closing this end makes that read end-of-file: the manager takes the pool's
    # broken path, which closes the call queue (freeing its feeder) and ends the thread.
    try:
        if result_queue is not None:
            result_queue._writer.close()
    except Exception:
        pass


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


def close() -> None:
    """No more jobs: queued compiler and warm-up work finishes, then workers exit."""

    pool = _state.pop("pool", None)
    if pool is not None:
        pool.shutdown(wait=False)
    cpu_pool = _state.pop("compiler_pool", None)
    if cpu_pool is not None:
        cpu_pool.shutdown(wait=False)


def save() -> None:
    """Merge this run's requests into its tag's plan (atomic rewrite)."""

    try:
        path = _state["plan"]
        if path is None:
            return
        with _lock:
            # Cache-entry dictionaries are updated by compile threads. Keep the
            # entire snapshot independent while JSON is serialized without the lock.
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
    save()
    close()
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
        counts = dict(_state["counts"])
        requested = len(_requested)
    if "inline_seconds" in counts:
        counts["inline_seconds"] = round(counts["inline_seconds"], 3)
    return {"schema": SCHEMA, "enabled": enabled(), "installed": _state["installed"],
            "requested_this_run": requested, "replay": replay, "calls": counts}


__all__ = ["CUPY_ROUTE", "SCHEMA", "cache_entry", "close", "compiler_identity", "default_tag_text", "enabled",
           "install", "load_plan", "rebind", "receipt", "save", "start", "submit"]
