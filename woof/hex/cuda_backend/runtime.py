"""CUDA runtime, capability, compilation-cache, and launch authority.

CuPy cubins persist across process launches. A user-owned architecture cache
is used unless ``MPAS_PORT_CUDA_CACHE_DIR`` or ``CUPY_CACHE_DIR`` is declared.
CuPy keys compiled images by compiler inputs, including source and options.
The in-process :class:`KernelCache` then owns compiled RawModule functions.
"""

from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

from .arch_admission import admitted_architecture, below_floor_refusal
from .compile_contract import (
    CompileContractError,
    CompileManifest,
    NvrtcCompileCapture,
    compile_platform_binding,
    validate_module_key,
)


class CudaRefusal(RuntimeError):
    """A named CUDA capability or runtime contract was not satisfied."""


def _cupy() -> Any:
    try:
        import cupy as cp
    except Exception as error:  # pragma: no cover - exercised on non-CUDA hosts
        raise CudaRefusal(f"CuPy CUDA backend is unavailable: {error}") from error
    return cp


def _cache_directory(device_id: int, compute: tuple[int, int], raw: str | Path | None) -> Path:
    declared = raw
    if declared is None:
        declared = os.environ.get("MPAS_PORT_CUDA_CACHE_DIR")
    if declared is None:
        declared = os.environ.get("CUPY_CACHE_DIR")
    if declared is None:
        declared = (
            Path.home() / ".cache" / "woof hex" / "cuda"
            / f"sm_{compute[0]}{compute[1]}"
        )
    result = Path(declared).expanduser().resolve()
    result.mkdir(parents=True, exist_ok=True)
    os.environ["CUPY_CACHE_DIR"] = str(result)
    return result


@dataclass(frozen=True, slots=True)
class CudaCapability:
    device_id: int
    name: str
    compute_major: int
    compute_minor: int
    total_memory_bytes: int
    multiprocessor_count: int
    runtime_version: int
    driver_version: int
    nvrtc_version: tuple[int, int]
    cupy_version: str
    cache_directory: str

    @property
    def compute(self) -> tuple[int, int]:
        return (self.compute_major, self.compute_minor)

    @property
    def sm(self) -> str:
        return f"sm_{self.compute_major}{self.compute_minor}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "compute_capability": f"{self.compute_major}.{self.compute_minor}",
            "sm": self.sm,
            "total_memory_bytes": self.total_memory_bytes,
            "multiprocessor_count": self.multiprocessor_count,
            "runtime_version": self.runtime_version,
            "driver_version": self.driver_version,
            "nvrtc_version": list(self.nvrtc_version),
            "cupy_version": self.cupy_version,
            "cache_directory": self.cache_directory,
        }


#: Admissions already granted in this process, keyed by the exact arguments
#: they were granted for.  Each entry keeps the cupy module it was measured
#: against, so a stubbed cupy in a test can never be answered from a real
#: card's admission (or the other way round), and an address recycled for a
#: fresh module object cannot alias a stale entry.
_ADMISSIONS: dict[tuple[Any, ...], tuple[Any, CudaCapability]] = {}


def forget_admissions() -> None:
    """Drop every memoised admission; the next ``require_cuda`` re-measures."""

    _ADMISSIONS.clear()


def require_cuda(
    *,
    device_id: int | None = None,
    min_compute: tuple[int, int] = (12, 0),
    required_compute: tuple[int, int] | None = None,
    min_runtime_version: int = 12_000,
    cache_dir: str | Path | None = None,
) -> CudaCapability:
    """Select a real CUDA device or refuse with the failing capability name.

    The runtime floor is CUDA 12, the oldest major this distribution's
    extras install (``gpu-cu12``, cupy-cuda12x) and the oldest CuPy 14
    supports.  It was 13000 until 0.4.0, a value that named no breakage:
    the kernels need nothing a CUDA 12 toolkit lacks (plain stream capture,
    and the NVRTC flags ``--fmad=false --ftz=false --prec-div=true
    --prec-sqrt=true -std=c++17``), and it refused every CUDA 12 driver,
    which is the driver many operational centres run.  Whether this
    install's NVRTC can build for the card is the toolchain question that
    can actually break a run, and it is asked of NVRTC itself below.

    An admission is measured once per process for each distinct argument
    set and answered from memory after that.  The dycore's translation
    units re-enter through their ``_cupy()`` helpers on every launch group,
    59 times a forecast step on the 43,884-cell point mesh, and
    ``cudaGetDeviceProperties`` costs 1.05 ms a call on the CUDA 13.3
    driver: 62-82 ms of host time per 0.38 s step, measured in
    ``evidence-gallery/hex-perf-profile-2026-09-13``, spent re-deciding a
    question whose answer cannot change while the process lives.  A
    refusal is never memoised, so a refused configuration is re-measured
    every time it is asked.
    """

    cp = _cupy()
    if device_id is None:
        device_id = int(cp.cuda.runtime.getDevice())
    key = (
        int(device_id),
        tuple(int(v) for v in min_compute),
        None if required_compute is None else tuple(int(v) for v in required_compute),
        int(min_runtime_version),
        None if cache_dir is None else str(cache_dir),
    )
    remembered = _ADMISSIONS.get(key)
    if remembered is not None and remembered[0] is cp:
        return remembered[1]
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as error:
        raise CudaRefusal(f"cuda.device_count unavailable: {error}") from error
    if device_id < 0 or device_id >= count:
        raise CudaRefusal(f"cuda.device_id={device_id} is outside [0,{count})")
    cp.cuda.Device(device_id).use()
    # Blocking sync, not CUDA's spin wait: a spinning host thread held one core
    # at 100 % for the whole GPU-bound step loop (cuda_backend/schedule.py).
    from .schedule import apply as _apply_wait_mode

    _apply_wait_mode(device_id)
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    compute = (int(properties["major"]), int(properties["minor"]))
    # Architectures at or above the floor keep the original checks unchanged.
    # Below the floor an architecture runs only on its own verified anchor
    # (contract receipt + frozen-authority set); see cuda_backend/arch_admission.
    anchored = None
    if compute < tuple(min_compute):
        anchored = admitted_architecture(compute)
        if anchored is None:
            raise CudaRefusal(below_floor_refusal(compute, tuple(min_compute)))
    if (
        required_compute is not None
        and compute != tuple(required_compute)
        and anchored is None
    ):
        raise CudaRefusal(
            f"cuda.compute_capability={compute[0]}.{compute[1]} != required "
            f"{required_compute[0]}.{required_compute[1]}"
        )
    runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
    if runtime_version < min_runtime_version:
        raise CudaRefusal(
            f"cuda.runtime_version={runtime_version} < required {min_runtime_version}"
        )
    from cupy.cuda import nvrtc

    supported = tuple(int(value) for value in nvrtc.getSupportedArchs())
    architecture = compute[0] * 10 + compute[1]
    if architecture not in supported:
        raise CudaRefusal(
            f"cuda.nvrtc_arch=compute_{architecture} is not supported; available={supported}"
        )
    cache = _cache_directory(device_id, compute, cache_dir)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    admitted = CudaCapability(
        device_id=device_id,
        name=str(name),
        compute_major=compute[0],
        compute_minor=compute[1],
        total_memory_bytes=int(properties["totalGlobalMem"]),
        multiprocessor_count=int(properties["multiProcessorCount"]),
        runtime_version=runtime_version,
        driver_version=int(cp.cuda.runtime.driverGetVersion()),
        nvrtc_version=tuple(int(value) for value in nvrtc.getVersion()),
        cupy_version=str(cp.__version__),
        cache_directory=str(cache),
    )
    _ADMISSIONS[key] = (cp, admitted)
    return admitted


@dataclass(frozen=True, slots=True)
class KernelTiming:
    compile_seconds: float
    first_launch_ms: float
    first_wall_ms: float
    mean_launch_ms: float
    min_launch_ms: float
    max_launch_ms: float
    repeats: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "compile_seconds": self.compile_seconds,
            "first_launch_ms": self.first_launch_ms,
            "first_wall_ms": self.first_wall_ms,
            "mean_launch_ms": self.mean_launch_ms,
            "min_launch_ms": self.min_launch_ms,
            "max_launch_ms": self.max_launch_ms,
            "repeats": self.repeats,
        }


_KERNEL_COMPILE_SECONDS: dict[int, float] = {}


class _PostLaunchKernel:
    """A resolved kernel with a caller-owned observer on every launch.

    The regional (limited-area) lane uses this to hold native's garbage
    element at its pool value: native dimensions every array ``n+1`` and
    never enters a loop for the extra element, but a device launch cannot
    skip its last element because ``lidx(k,i,n)`` makes the thread bound and
    the array stride the same integer.  The observer therefore restores the
    pool value after each launch, which is the residency form of the CPU
    authority's pad-compute-strip.

    The wrapper exists so the discipline reaches EVERY entrypoint the step
    resolves -- ``cuda_horizontal``, ``cuda_transport``, ``cuda_acoustic*``
    and ``cuda_driver`` alike -- without one of those translation units
    changing by a byte.  It is inert unless a cache sets
    :attr:`KernelCache.post_launch`, which nothing on the global lane does.
    """

    __slots__ = ("_kernel", "_cache", "_name", "_module_key")

    def __init__(
        self,
        kernel: Any,
        cache: "KernelCache",
        name: str,
        module_key: str = "",
    ) -> None:
        self._kernel = kernel
        self._cache = cache
        self._name = name
        self._module_key = module_key

    def __call__(self, grid: Any, block: Any, args: Any, **kwargs: Any) -> Any:
        result = self._kernel(grid, block, args, **kwargs)
        hook = self._cache.post_launch
        if hook is not None:
            # The MODULE is passed as well as the kernel name.  An observer
            # that rewrites memory after a launch has to be able to tell whose
            # launch it was: one cache serves the dycore AND the physics seam,
            # and only one of them wants its arrays touched.
            hook(self._name, args, module_key=self._module_key)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kernel, name)


class KernelCache:
    """In-process RawModule cache with compiler-bound compile inventory.

    ``module_key`` is mandatory at every resolution site.  It names the CUDA
    translation unit in receipts while the internal cache key binds its source,
    requested options, target SM and woof compile-platform fingerprint.
    """

    def __init__(
        self,
        *,
        capability: CudaCapability | None = None,
        cache_dir: str | Path | None = None,
        base_options: Sequence[str] | None = None,
    ) -> None:
        self.capability = (
            require_cuda(cache_dir=cache_dir) if capability is None else capability
        )
        selected_cache = _cache_directory(
            self.capability.device_id,
            self.capability.compute,
            cache_dir or self.capability.cache_directory,
        )
        self.cache_directory = selected_cache
        # None: the process's numerics mode (woof.hex.numerics; fast by default, strict for bitwise proofs).
        self.base_options = tuple(str(option) for option in (
            _hex_numerics.kernel_base() if base_options is None else base_options))
        from .aot import Pack, install
        self._aot_pack = Pack()
        install(_cupy(), self._aot_pack)
        try:
            self._compile_manifest = CompileManifest(compile_platform_binding())
        except CompileContractError as error:
            raise CudaRefusal(
                f"cuda.compile_platform_fingerprint refused: {error}"
            ) from error
        self._modules: dict[str, Any] = {}
        self._kernels: dict[str, Any] = {}
        self._compile_seconds: dict[str, float] = {}
        self._observed: dict[str, _PostLaunchKernel] = {}
        #: Resolution memo: (name, module_key, source, options) -> (kernel_key, kernel, module_key).
        #: A repeat request with the same arguments returns the kernel its first resolution produced,
        #: skipping the two sha256 passes over the source and the manifest bookkeeping that the first
        #: resolution already recorded. Breakage it prevents: U-MPASPHYS-1, every per-step launch on
        #: the MPAS-physics route re-hashed its whole translation unit on the forecast's one host thread.
        self._resolved: dict[tuple[Any, ...], tuple[str, Any, str]] = {}
        # Operational timing belongs outside the byte-identity manifest.
        self._load_timings: list[dict[str, Any]] = []
        #: Optional per-launch observer, set only by the regional lane.  When
        #: None -- which is every global-lane cache -- ``raw_kernel`` returns
        #: the bare CuPy kernel and nothing about resolution changes.
        self.post_launch: Any | None = None

    def _observe(
        self, kernel_key: str, kernel: Any, name: str = "", module_key: str = ""
    ) -> Any:
        """Wrap a resolved kernel only when this cache carries an observer."""

        if self.post_launch is None:
            return kernel
        wrapped = self._observed.get(kernel_key)
        if wrapped is None:
            wrapped = _PostLaunchKernel(kernel, self, name, module_key)
            self._observed[kernel_key] = wrapped
            _KERNEL_COMPILE_SECONDS[id(wrapped)] = _KERNEL_COMPILE_SECONDS.get(
                id(kernel), 0.0
            )
        return wrapped

    def _module_key(self, source: str, options: Sequence[str]) -> str:
        digest = hashlib.sha256()
        digest.update(self.capability.sm.encode("ascii"))
        digest.update(b"\0compile-platform\0")
        digest.update(self._compile_manifest.fingerprint_sha256.encode("ascii"))
        digest.update(source.encode("utf-8"))
        for option in (*self.base_options, *tuple(options)):
            digest.update(b"\0")
            digest.update(str(option).encode("utf-8"))
        return digest.hexdigest()

    def raw_kernel(
        self,
        name: str,
        source: str,
        *,
        module_key: str,
        options: Sequence[str] = (),
    ) -> Any:
        memo_key = (name, module_key, source, tuple(str(value) for value in options))
        hit = self._resolved.get(memo_key)
        if hit is not None:
            return self._observe(hit[0], hit[1], name, hit[2])
        kernel = self._raw_kernel_resolve(name, source, module_key=module_key, options=options)
        kernel_key = f"{self._module_key(source, memo_key[3])}:{name}"
        self._resolved[memo_key] = (kernel_key, self._kernels[kernel_key], validate_module_key(module_key))
        return kernel

    def _raw_kernel_resolve(
        self,
        name: str,
        source: str,
        *,
        module_key: str,
        options: Sequence[str] = (),
    ) -> Any:
        try:
            stable_module_key = validate_module_key(module_key)
        except CompileContractError as error:
            raise CudaRefusal(f"cuda.module_key refused: {error}") from error
        cp = _cupy()
        extra_options = tuple(str(value) for value in options)
        compiled_key = self._module_key(source, extra_options)
        requested_options = (
            *self.base_options,
            *extra_options,
        )
        try:
            self._compile_manifest.declare_module(
                stable_module_key,
                source=source,
                requested_options=requested_options,
                module_cache_key=compiled_key,
            )
        except CompileContractError as error:
            raise CudaRefusal(f"cuda.compile_manifest refused: {error}") from error
        kernel_key = f"{compiled_key}:{name}"
        cached = self._kernels.get(kernel_key)
        if cached is not None:
            try:
                self._compile_manifest.record_kernel(stable_module_key, name)
            except CompileContractError as error:
                raise CudaRefusal(
                    f"cuda.compile_manifest refused: {error}"
                ) from error
            return self._observe(kernel_key, cached, name, stable_module_key)
        module = self._modules.get(compiled_key)
        load_source = "jit"
        load_reason = None
        aot_entry = None
        capture = None
        shared = shared_key = module_bank = None
        effective_compile = None
        started = time.perf_counter()
        if (module is None and os.environ.get('WOOF_HEX_MOVING_RESIDENT') == '1'
                and os.environ.get('WOOF_HEX_MOVING_KERNEL_MODULES', '1') != '0'):
            from .. import moving_kernel_modules as module_bank
            from .aot import effective_options
            shared_key = (self.capability.device_id, compiled_key, stable_module_key,
                          getattr(self._aot_pack, 'resident_identity', None),
                          tuple(effective_options(requested_options)))
            shared = module_bank.get(shared_key)
            if shared is not None:
                module, effective_compile, load_source = shared
        if module is None:
            os.environ["CUPY_CACHE_DIR"] = str(self.cache_directory)
            module, load_reason, aot_entry = self._aot_pack.load(
                cp, self.cache_directory, stable_module_key, source,
                requested_options, self.capability.sm,
            )
            if module is not None:
                load_source = "aot"
            else:
                module = cp.RawModule(
                    code=source,
                    options=requested_options,
                    backend="nvrtc",
                    enable_cooperative_groups=False,
                )
        if compiled_key in self._modules:
            kernel = module.get_function(name)
        elif shared is not None:
            kernel = module.get_function(name)
            self._compile_manifest.record_effective_compile(stable_module_key, effective_compile)
        elif aot_entry is not None:
            kernel = module.get_function(name)
            effective_compile = aot_entry['effective_compile']
            self._compile_manifest.record_effective_compile(
                stable_module_key, aot_entry["effective_compile"])
        else:
            with NvrtcCompileCapture() as capture:
                kernel = module.get_function(name)
            try:
                effective_compile = capture.evidence(hashlib.sha256(source.encode('utf-8')).hexdigest())
                self._compile_manifest.record_effective_compile(
                    stable_module_key,
                    effective_compile,
                )
            except CompileContractError as error:
                raise CudaRefusal(
                    f"cuda.compile_manifest refused: {error}"
                ) from error
        # A shared module was published only after its first synchronization.
        # Resolving another bare function queues no work and must not drain
        # the running class's stream merely because a new class cache asks.
        if shared is None:
            cp.cuda.runtime.deviceSynchronize()
        if module_bank is not None and shared is None and effective_compile is not None:
            module_bank.put(shared_key, module, effective_compile, load_source)
        elapsed = time.perf_counter() - started
        if compiled_key not in self._modules and shared is None:
            from .aot import mark
            mark(load_source, stable_module_key, elapsed, reason=load_reason)
        self._modules[compiled_key] = module
        self._kernels[kernel_key] = kernel
        self._compile_seconds[kernel_key] = elapsed
        self._load_timings.append({
            "module_key": stable_module_key, "kernel": name,
            "module_cache_key": compiled_key, "device": self.capability.device_id,
            "module_load_seconds": elapsed,
            "route": load_source, "aot_miss_reason": load_reason,
            "nvrtc_calls": None if capture is None else len(capture.observations),
            "nvrtc_seconds": None if capture is None else capture.elapsed_seconds,
            "cache_status": ("moving_resident_module" if shared is not None else "resident_module" if compiled_key in self._modules
                and capture is None and aot_entry is None else
                "aot" if aot_entry is not None else
                "nvrtc_compile" if capture is not None and capture.observations else
                "not_observed" if capture is None or capture.unavailable_reason else
                "nvrtc_not_called_possible_disk_hit"),
        })
        _KERNEL_COMPILE_SECONDS[id(kernel)] = elapsed
        try:
            self._compile_manifest.record_kernel(stable_module_key, name)
        except CompileContractError as error:
            raise CudaRefusal(f"cuda.compile_manifest refused: {error}") from error
        return self._observe(kernel_key, kernel, name, stable_module_key)

    def record_shape_origin(
        self, module_key: str, generic_source: str,
        dimensions: Sequence[tuple[str, int]],
    ) -> None:
        """Record which generic source a shape-specialized module came from."""

        try:
            self._compile_manifest.record_shape_origin(
                module_key,
                generic_source_sha256=hashlib.sha256(
                    generic_source.encode("utf-8")).hexdigest(),
                dimensions=dimensions,
            )
        except CompileContractError as error:
            raise CudaRefusal(f"cuda.compile_manifest refused: {error}") from error

    def raw_kernels(
        self,
        names: Sequence[str],
        source: str,
        *,
        module_key: str,
        options: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Resolve several extern-C functions from one compiled RawModule."""

        selected = tuple(str(name) for name in names)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("raw_kernels names must be non-empty and unique")
        return {
            name: self.raw_kernel(
                name, source, module_key=module_key, options=options
            )
            for name in selected
        }

    def compile_manifest(self) -> dict[str, Any]:
        """JSON-ready snapshot for a CUDA step or long-run receipt."""

        return self._compile_manifest.snapshot()

    def compile_seconds(self, kernel: Any) -> float:
        return float(_KERNEL_COMPILE_SECONDS.get(id(kernel), 0.0))

    def timing_receipt(self) -> dict[str, Any]:
        return {"cache_directory": str(self.cache_directory),
                "loads": [dict(row) for row in self._load_timings],
                "scope": "host module resolution includes synchronization; NVRTC calls are nested, not additive"}

    def clear_memory_cache(self) -> None:
        self._resolved.clear()
        self._kernels.clear()
        self._modules.clear()
        self._compile_seconds.clear()


def _check_launch_error(cp: Any) -> None:
    checker = getattr(cp.cuda.runtime, "peekAtLastError", None)
    if checker is None:
        checker = getattr(cp.cuda.runtime, "getLastError", None)
    # CuPy 14's CUDA 13 Windows runtime binding exports neither helper.
    # RawKernel launch and the following event synchronization still surface
    # launch/configuration/execution failures as CUDARuntimeError.
    if checker is None:
        return
    code = int(checker())
    if code != 0:
        try:
            message = cp.cuda.runtime.getErrorString(code)
        except Exception:
            message = f"CUDA error {code}"
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        raise RuntimeError(f"CUDA kernel launch failed: {message}")


def _one_launch(
    cp: Any,
    kernel: Any,
    grid: tuple[int, ...],
    block: tuple[int, ...],
    args: tuple[Any, ...],
    stream: Any,
) -> tuple[float, float]:
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    wall = time.perf_counter()
    start.record(stream)
    kernel(grid, block, args, stream=stream)
    _check_launch_error(cp)
    end.record(stream)
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end)), (time.perf_counter() - wall) * 1_000.0


def launch_checked(
    kernel: Any,
    grid: tuple[int, ...],
    block: tuple[int, ...],
    args: tuple[Any, ...],
    *,
    stream: Any | None = None,
) -> None:
    """Launch exactly once on the stream and check the launch error state.

    No event is recorded and nothing is synchronized: the launch is queued
    and the host moves on.  This is the step path's launch; ``launch_timed``
    below is the benchmark's, and the difference is what the profile
    measured (``evidence-gallery/hex-perf-profile-2026-09-13``): every
    recovery kernel that went through ``launch_timed`` with
    ``timing_repeats=1`` was launched twice, event-synchronized twice and
    stream-synchronized once, 36 host stalls and 10.7 ms of repeated device
    work per forecast step.
    """

    cp = _cupy()
    if stream is None:
        kernel(tuple(grid), tuple(block), tuple(args))
    else:
        kernel(tuple(grid), tuple(block), tuple(args), stream=stream)
    _check_launch_error(cp)


def launch_timed(
    kernel: Any,
    grid: tuple[int, ...],
    block: tuple[int, ...],
    args: tuple[Any, ...],
    *,
    warmup: int = 1,
    repeats: int = 1,
    stream: Any | None = None,
) -> KernelTiming:
    """Launch, check the CUDA error state, synchronize, and event-time work."""

    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    cp = _cupy()
    selected = cp.cuda.get_current_stream() if stream is None else stream
    first_ms, first_wall_ms = _one_launch(
        cp, kernel, tuple(grid), tuple(block), tuple(args), selected
    )
    for _ in range(warmup):
        kernel(tuple(grid), tuple(block), tuple(args), stream=selected)
        _check_launch_error(cp)
    selected.synchronize()
    samples = [
        _one_launch(cp, kernel, tuple(grid), tuple(block), tuple(args), selected)[0]
        for _ in range(repeats)
    ]
    return KernelTiming(
        compile_seconds=float(_KERNEL_COMPILE_SECONDS.get(id(kernel), 0.0)),
        first_launch_ms=first_ms,
        first_wall_ms=first_wall_ms,
        mean_launch_ms=float(sum(samples) / len(samples)),
        min_launch_ms=float(min(samples)),
        max_launch_ms=float(max(samples)),
        repeats=repeats,
    )


__all__ = [
    "CudaCapability",
    "CudaRefusal",
    "KernelCache",
    "KernelTiming",
    "forget_admissions",
    "launch_checked",
    "launch_timed",
    "require_cuda",
]
