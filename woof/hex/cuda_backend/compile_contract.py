"""Fail-closed identity and manifest for MPAS NVRTC translation units.

The numerical gates certify compiled kernels, not CUDA source in isolation.
This module binds every translation unit to woof's measured NVRTC platform
fingerprint and records the source, requested options, kernels resolved from
the module, and any effective compile evidence CuPy actually exposes.

CuPy's ``RawModule`` route appends ``-ftz=true`` after caller options on the
measured stack.  That is deliberately *not* inferred here.  When NVRTC really
fires, :class:`NvrtcCompileCapture` observes the options at its entry point.
A disk-cache hit, a changed private CuPy API, or an unexposed compiled image is
reported as unavailable rather than filled with a plausible value.
"""

from __future__ import annotations

import time

from contextlib import AbstractContextManager
from dataclasses import dataclass
import copy
import hashlib
import json
import os
import re
import threading
from typing import Any, Iterable, Mapping


COMPILE_MANIFEST_SCHEMA = "mpas-port.cuda-compile-manifest/v1"
STATUS_RESOLVED = "resolved"
STATUS_UNAVAILABLE = "unavailable"
UNAVAILABLE = "unavailable"

_REQUIRED_FINGERPRINT_FIELDS = (
    "nvrtc_build",
    "nvrtc_build_id",
    "nvrtc_library_sha256",
    "cuda_driver_version",
    "device_compute_capability",
    "cupy_version",
    "numpy_version",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


_NVRTC_DIGEST_LOCK = threading.Lock()


def hash_nvrtc_library_once() -> None:
    """Hash the loaded NVRTC library once per file per process, for every caller.

    THE BREAKAGE THIS ENDS (member-leg start-up measure, RTX 5090,
    2026-10-28): woof's ``compile_platform.nvrtc_library`` reads and SHA-256s
    the whole loaded library (115 MB, 72 ms) on every call, and one forecast
    start called it 8 times on its main thread warm and 12 cold (every
    compiler identity -- each ``aot.Pack()``, the warm store, the kernel plan --
    and every compile-platform fingerprint, which calls it twice): about 0.5 s
    warm and 0.9 s cold of one-core time while the card waited.

    The digest is reused only for the same file: the key is the library path
    woof resolves and that file's device, inode, size, mtime and ctime.  A
    replaced file has a new inode and a rewritten one a new ctime, and then the
    file is read and hashed again exactly as before.  An unresolved library
    (NVRTC not loaded yet) is never cached.  A stand-in ``nvrtc_library`` (a
    test's) is left alone: the wrapper goes over woof's own function only.
    Installed by ``kernel_prewarm.install_recorder`` (every forecast route,
    before its first compiler identity) and by the fingerprint below.
    """
    try:
        from woof.certify import compile_platform
    except Exception:  # no engine: nothing to hash, and the callers refuse by name
        return
    with _NVRTC_DIGEST_LOCK:
        original = compile_platform.nvrtc_library
        if getattr(original, "_hex_hashed_once", False):
            return
        locate = getattr(compile_platform, "_loaded_library_path", None)
        if locate is None or getattr(original, "__module__", None) != compile_platform.__name__:
            return
        digests: dict[tuple, dict[str, str]] = {}

        def nvrtc_library() -> dict[str, str]:
            path = locate()
            try:
                stat = os.stat(path) if path else None
            except OSError:
                stat = None
            if stat is None:
                return original()
            key = (path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            known = digests.get(key)
            if known is not None:
                return dict(known)
            library = original()
            if library.get("status") == STATUS_RESOLVED and library.get("path") == path:
                digests[key] = dict(library)
            return library

        nvrtc_library._hex_hashed_once = True
        nvrtc_library.__wrapped__ = original
        nvrtc_library.__doc__ = original.__doc__
        compile_platform.nvrtc_library = nvrtc_library
_MODULE_KEY_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
#: Guards installing the NVRTC-entry hook only; never held while NVRTC compiles.
_CAPTURE_LOCK = threading.RLock()
#: Per-thread: the captures open on this thread (innermost last), and whether this
#: thread is already inside an observing hook (an inner hook then passes through).
_CAPTURE_THREAD = threading.local()
_HOOK_MARK = "_hex_nvrtc_capture_hook"


def _thread_captures() -> list["NvrtcCompileCapture"]:
    stack = getattr(_CAPTURE_THREAD, "stack", None)
    if stack is None:
        stack = _CAPTURE_THREAD.stack = []
    return stack


def _install_capture_hook(compiler: Any) -> None:
    """Make an observing hook the outermost ``_compile_using_nvrtc_no_warning``.

    THE BREAKAGE THIS ENDS (VRFS 8x RTX 5090, 2026-10-28): the capture used to
    swap this process-global function in and out under one process-wide lock
    held for the whole compile, so every rank thread's first-use compile (the
    mesh-specific shape modules no pack can hold) waited for every other
    rank's: 8 ranks compiled 48 modules one at a time, a staircase of 1.05 s
    steps in fast numerics (ft-r700-8c) and ~4 s steps in strict (127 s first
    step).  The hook now stays installed and passes through unless the calling
    thread has a capture open; it records into that thread's open captures, so
    ranks compile concurrently and a capture never observes another thread's
    compile.  The lock guards installation only.  A wrapper installed on top
    later (the kernel plan funnel, the warm store) is covered by a new hook at
    the next capture, so a capture still observes at the outermost entry as
    before; the per-thread flag keeps nested hooks from recording a call twice.
    """
    with _CAPTURE_LOCK:
        current = compiler._compile_using_nvrtc_no_warning
        if getattr(current, _HOOK_MARK, False):
            return
        original = current

        def hook(source: Any, options: Iterable[str] = (), *args: Any, **kwargs: Any) -> Any:
            captures = getattr(_CAPTURE_THREAD, "stack", None)
            if not captures or getattr(_CAPTURE_THREAD, "inside", False):
                return original(source, options, *args, **kwargs)
            _CAPTURE_THREAD.inside = True
            started = time.perf_counter()
            try:
                result = original(source, options, *args, **kwargs)
            finally:
                _CAPTURE_THREAD.inside = False
                elapsed = time.perf_counter() - started
                for capture in captures:
                    capture.elapsed_seconds += elapsed
            observation = _observation(source, options, result)
            for capture in captures:
                capture.observations.append(observation)
            return result

        setattr(hook, _HOOK_MARK, True)
        hook.__wrapped__ = original  # type: ignore[attr-defined]
        compiler._compile_using_nvrtc_no_warning = hook


def _observation(source: Any, options: Iterable[str], result: Any) -> "NvrtcObservation":
    blob = result[0] if isinstance(result, tuple) else result
    source_text = (
        source if isinstance(source, str)
        else bytes(source).decode("utf-8", errors="replace")
    )
    normalized_options = tuple(str(option) for option in options)
    flags = tuple(option for option in normalized_options if not option.startswith("-I"))
    return NvrtcObservation(
        source_sha256=source_sha256(source_text),
        effective_flags=flags,
        include_path_count=len(normalized_options) - len(flags),
        compiled_image=_object_record(blob),
    )


class CompileContractError(RuntimeError):
    """A compiler identity or translation-unit claim cannot be proved."""


def canonical_sha256(document: Mapping[str, Any]) -> str:
    """SHA-256 of a mapping's canonical JSON representation."""

    encoded = json.dumps(
        dict(document), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_sha256(source: str) -> str:
    """SHA-256 of one UTF-8 CUDA translation unit."""

    if not isinstance(source, str):
        raise TypeError("CUDA source must be a string")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def validate_module_key(module_key: str) -> str:
    """Return a stable manifest key or refuse an ambiguous one."""

    if not isinstance(module_key, str) or not _MODULE_KEY_RE.fullmatch(module_key):
        raise CompileContractError(
            "CUDA module_key must be a non-empty stable identifier containing "
            "only letters, digits, '.', ':', '_' or '-'"
        )
    return module_key


#: The build id NVRTC 13.2.86 prints in its PTX banner ("Compiler Build ID:
#: UNKNOWN"; a system CUDA 13.2 toolkit, 2026-10-23).
#: That compiler carries no changelist, so its identity is the four-part build
#: plus the loaded library's SHA-256, which the fingerprint always requires.
#: THE BREAKAGE THIS ENDS: a bare install on a CUDA 13.2 machine refused its
#: first forecast with "nvrtc_build_id is invalid" and lanes had to install a
#: second NVRTC into their own venvs.
UNKNOWN_BUILD_ID = "UNKNOWN"


def build_id_accepted(build_id: str) -> bool:
    """A changelist id (``CL-...``), or NVRTC's own UNKNOWN pinned by its library digest."""

    return isinstance(build_id, str) and (build_id.startswith("CL-") or build_id == UNKNOWN_BUILD_ID)


def validate_compile_platform_fingerprint(
    measured: Mapping[str, Any],
) -> dict[str, str]:
    """Validate woof's fingerprint at certification resolution.

    The four-part NVRTC build (for example ``13.0.48``), build id and loaded
    library digest are load-bearing.  Falling back to ``nvrtc.getVersion()``
    would conflate compiler builds already measured to generate different
    kernel answers.
    """

    if not isinstance(measured, Mapping):
        raise CompileContractError(
            "woof compile_platform_fingerprint did not return a mapping"
        )
    missing = [name for name in _REQUIRED_FINGERPRINT_FIELDS if name not in measured]
    if missing:
        raise CompileContractError(
            f"woof compile_platform_fingerprint is missing fields {missing}"
        )
    normalized: dict[str, str] = {}
    for key, value in measured.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise CompileContractError(
                "woof compile_platform_fingerprint keys and values must be strings"
            )
        if not value or value == UNAVAILABLE:
            raise CompileContractError(
                f"woof compile_platform_fingerprint.{key} is unavailable"
            )
        normalized[key] = value
    if re.fullmatch(r"\d+\.\d+\.\d+", normalized["nvrtc_build"]) is None:
        raise CompileContractError(
            "woof compile_platform_fingerprint.nvrtc_build does not identify "
            "the compiler at build resolution"
        )
    if not build_id_accepted(normalized["nvrtc_build_id"]):
        raise CompileContractError(
            "woof compile_platform_fingerprint.nvrtc_build_id is invalid"
        )
    if _SHA256_RE.fullmatch(normalized["nvrtc_library_sha256"]) is None:
        raise CompileContractError(
            "woof compile_platform_fingerprint.nvrtc_library_sha256 is invalid"
        )
    return dict(sorted(normalized.items()))


def resolve_compile_platform_fingerprint() -> dict[str, str]:
    """Measure and validate woof's compiler fingerprint, or refuse.

    woof owns the fingerprint contract.  MPAS binds it rather than carrying a
    second implementation which could silently drift to different semantics.
    """

    try:
        from woof.certify.compile_platform import compile_platform_fingerprint
    except Exception as error:
        raise CompileContractError(
            "woof.certify.compile_platform.compile_platform_fingerprint is "
            f"unavailable: {type(error).__name__}: {error}"
        ) from error
    try:
        # The fingerprint hashes the loaded NVRTC library twice per call;
        # hash that file once per process.
        hash_nvrtc_library_once()
        measured = compile_platform_fingerprint()
    except Exception as error:
        raise CompileContractError(
            "woof compile_platform_fingerprint measurement failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    return validate_compile_platform_fingerprint(measured)


#: This process's compile-platform binding (the loaded NVRTC and CuPy do not change within a process).
_PLATFORM_BINDING: dict[str, Any] | None = None


def compile_platform_binding() -> dict[str, Any]:
    """JSON-ready full fingerprint and its canonical SHA-256 binding, measured once per process.

    THE COST THIS REMOVES (RRFS hrrrv5, 2026-10-29): every KernelCache (each StrictPhysicsKernelCache, ~5 per rank
    start) measured the fingerprint again.  A refusal is not kept: the next call measures again and refuses again.
    """

    global _PLATFORM_BINDING
    if _PLATFORM_BINDING is None:
        fingerprint = resolve_compile_platform_fingerprint()
        _PLATFORM_BINDING = {
            "fingerprint": fingerprint,
            "sha256": canonical_sha256(fingerprint),
        }
    return copy.deepcopy(_PLATFORM_BINDING)


def _object_record(blob: Any) -> dict[str, Any]:
    if isinstance(blob, str):
        payload = blob.encode("utf-8")
        kind = "ptx"
    elif isinstance(blob, (bytes, bytearray, memoryview)):
        payload = bytes(blob)
        kind = "cubin" if payload[:4] == b"\x7fELF" else "ptx"
    else:
        return {
            "status": STATUS_UNAVAILABLE,
            "reason": "NVRTC returned no string or byte image to the capture",
        }
    return {
        "status": STATUS_RESOLVED,
        "kind": kind,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class NvrtcObservation:
    """One invocation observed at CuPy's NVRTC entry point."""

    source_sha256: str
    effective_flags: tuple[str, ...]
    include_path_count: int
    compiled_image: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "effective_flags": list(self.effective_flags),
            "include_path_count": self.include_path_count,
            "include_paths_omitted": (
                "-I entries describe this machine's toolkit layout and are "
                "counted but not copied into the arithmetic contract"
            ),
            "compiled_image": dict(self.compiled_image),
        }


class NvrtcCompileCapture(AbstractContextManager["NvrtcCompileCapture"]):
    """Observe effective NVRTC options if CuPy compiles inside the context.

    The wrapped name is private CuPy API, so absence is an ordinary explicit
    ``unavailable`` result.  Observation is per thread (see
    :func:`_install_capture_hook`): no lock is held while NVRTC compiles, so
    the rank threads of a multi-card run compile their first-use modules
    concurrently.
    """

    def __init__(self) -> None:
        self.observations: list[NvrtcObservation] = []
        self.elapsed_seconds = 0.0
        self.unavailable_reason: str | None = None
        self._open = False

    def __enter__(self) -> "NvrtcCompileCapture":
        try:
            from cupy.cuda import compiler
            getattr(compiler, "_compile_using_nvrtc_no_warning")
        except Exception as error:
            self.unavailable_reason = (
                "CuPy exposes no usable NVRTC entry capture: "
                f"{type(error).__name__}: {error}"
            )
            return self
        _install_capture_hook(compiler)
        _thread_captures().append(self)
        self._open = True
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._open:
            stack = _thread_captures()
            for index in range(len(stack) - 1, -1, -1):
                if stack[index] is self:
                    del stack[index]
                    break
            self._open = False
        return False

    def evidence(self, expected_source_sha256: str) -> dict[str, Any]:
        matched = [
            observation.as_dict() for observation in self.observations
            if observation.source_sha256 == expected_source_sha256
        ]
        if matched:
            return {
                "status": STATUS_RESOLVED,
                "method": (
                    "wrapped cupy.cuda.compiler."
                    "_compile_using_nvrtc_no_warning at the NVRTC entry point"
                ),
                "observations": matched,
            }
        reason = self.unavailable_reason or (
            "NVRTC did not fire while the RawModule function was resolved; "
            "the compiled image may have been a CuPy disk-cache hit"
        )
        return {
            "status": STATUS_UNAVAILABLE,
            "reason": (
                reason + "; effective terminal -ftz=true is not claimed "
                "without a real NVRTC-entry capture"
            ),
        }


class CompileManifest:
    """Per-cache inventory of translation units bound to one compiler."""

    def __init__(self, binding: Mapping[str, Any]) -> None:
        if not isinstance(binding, Mapping) or "fingerprint" not in binding:
            raise CompileContractError(
                "compile platform binding is missing its fingerprint document"
            )
        fingerprint = validate_compile_platform_fingerprint(binding["fingerprint"])
        digest = str(binding.get("sha256", ""))
        expected = canonical_sha256(fingerprint)
        if digest != expected:
            raise CompileContractError(
                "compile platform fingerprint SHA-256 does not match its document"
            )
        self._binding = {"fingerprint": fingerprint, "sha256": digest}
        self._modules: dict[str, dict[str, Any]] = {}
        self._shape_origins: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @property
    def fingerprint_sha256(self) -> str:
        return str(self._binding["sha256"])

    def platform(self) -> dict[str, Any]:
        """The compile-platform binding alone: ``snapshot()["compile_platform"]`` without the module inventory."""

        return {"fingerprint": dict(self._binding["fingerprint"]), "sha256": self._binding["sha256"]}

    def declare_module(
        self,
        module_key: str,
        *,
        source: str,
        requested_options: Iterable[str],
        module_cache_key: str,
    ) -> None:
        key = validate_module_key(module_key)
        options = tuple(str(option) for option in requested_options)
        entry = {
            "source_sha256": source_sha256(source),
            "requested_options": list(options),
            "compile_platform_fingerprint_sha256": self.fingerprint_sha256,
            "module_cache_key": str(module_cache_key),
            "effective_compile": {
                "status": STATUS_UNAVAILABLE,
                "reason": (
                    "the RawModule has not produced observable NVRTC evidence; "
                    "its effective terminal -ftz=true is not claimed without a "
                    "real NVRTC-entry capture"
                ),
            },
            "resolved_kernels": [],
        }
        with self._lock:
            previous = self._modules.get(key)
            if previous is not None:
                identity_fields = (
                    "source_sha256", "requested_options",
                    "compile_platform_fingerprint_sha256", "module_cache_key",
                )
                if any(previous[name] != entry[name] for name in identity_fields):
                    raise CompileContractError(
                        f"CUDA module_key {key!r} was reused for a different "
                        "translation unit, options, or compiler"
                    )
                return
            self._modules[key] = entry

    def record_effective_compile(
        self, module_key: str, evidence: Mapping[str, Any]
    ) -> None:
        key = validate_module_key(module_key)
        with self._lock:
            if key not in self._modules:
                raise CompileContractError(
                    f"CUDA module_key {key!r} was not declared"
                )
            current = self._modules[key]["effective_compile"]
            if (current.get("status") == STATUS_RESOLVED
                    and evidence.get("status") != STATUS_RESOLVED):
                return
            self._modules[key]["effective_compile"] = json.loads(
                json.dumps(dict(evidence), sort_keys=True)
            )

    def record_kernel(self, module_key: str, kernel_name: str) -> None:
        key = validate_module_key(module_key)
        if not isinstance(kernel_name, str) or not kernel_name:
            raise CompileContractError("CUDA kernel name must be a non-empty string")
        with self._lock:
            if key not in self._modules:
                raise CompileContractError(
                    f"CUDA module_key {key!r} was not declared"
                )
            names = self._modules[key]["resolved_kernels"]
            if kernel_name not in names:
                names.append(kernel_name)
                names.sort()

    def record_shape_origin(
        self,
        module_key: str,
        *,
        generic_source_sha256: str,
        dimensions: Iterable[tuple[str, int]],
    ) -> None:
        """Bind a shape-specialized module to the generic source it came from.

        A shape module bakes one rank's own mesh dimensions into its source, so
        its source digest differs between ranks by design. The generic source
        and the dimension names are what two ranks must share.
        """

        key = validate_module_key(module_key)
        if not isinstance(generic_source_sha256, str) or _SHA256_RE.fullmatch(
            generic_source_sha256
        ) is None:
            raise CompileContractError("shape origin needs the generic source SHA-256")
        origin = {
            "generic_source_sha256": generic_source_sha256,
            "dimensions": [[str(name), int(value)] for name, value in dimensions],
        }
        with self._lock:
            if key not in self._modules:
                raise CompileContractError(
                    f"CUDA module_key {key!r} was not declared"
                )
            previous = self._shape_origins.get(key)
            if previous is not None and previous != origin:
                raise CompileContractError(
                    f"CUDA shape module {key!r} was bound to two generic origins"
                )
            self._shape_origins[key] = origin

    def snapshot(self) -> dict[str, Any]:
        """Return an isolated, deterministic, JSON-ready manifest."""

        with self._lock:
            modules = json.loads(json.dumps(self._modules, sort_keys=True))
            shape_origins = json.loads(json.dumps(self._shape_origins, sort_keys=True))
        document = {
            "schema": COMPILE_MANIFEST_SCHEMA,
            "compile_platform": {
                "fingerprint": dict(self._binding["fingerprint"]),
                "sha256": self._binding["sha256"],
            },
            "modules": modules,
        }
        if shape_origins:
            document["shape_origins"] = shape_origins
        return document


def kernel_cache_platform(cache: Any) -> dict[str, Any]:
    """A ``KernelCache``'s compile-platform binding (``compile_manifest()["compile_platform"]``) without the JSON
    round trip of every module the cache has compiled.

    Read here, not as a ``KernelCache`` method: ``cuda_backend/runtime.py`` is one of the sources the regional kernel
    set digests (``regional_admission.MINTED_KERNEL_SET_SHA256``), so a method added there lapses every minted
    regional class and refuses every limited-area CUDA run.  THE BREAKAGE THIS PREVENTS: fix/hex-host-bound 6ce2da3e
    added ``KernelCache.compile_platform`` and nine regional-mint / wheel-reach tests went red (2026-10-27).
    """

    return cache._compile_manifest.platform()


__all__ = [
    "COMPILE_MANIFEST_SCHEMA",
    "CompileContractError",
    "CompileManifest",
    "NvrtcCompileCapture",
    "STATUS_RESOLVED",
    "STATUS_UNAVAILABLE",
    "canonical_sha256",
    "kernel_cache_platform",
    "compile_platform_binding",
    "resolve_compile_platform_fingerprint",
    "source_sha256",
    "validate_compile_platform_fingerprint",
    "validate_module_key",
]
