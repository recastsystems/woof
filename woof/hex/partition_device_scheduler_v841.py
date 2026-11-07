"""Device-parallel segment scheduler for the partitioned v841 executor (brief-3).

This module is the *assignment policy and barrier machinery* for running the
streamed partition executor across more than one GPU.  It contains no dycore,
no physics and no field arithmetic: it decides which partition runs on which
device, keeps one long-lived worker thread bound to each device, and enforces
the segment barrier that the streamed executor's double-buffer swap depends on.

Model (fixed by brief 3):

* Partitions are split into contiguous ascending blocks, one block per device
  (``P = 16``, ``devices = (0, 1)`` -> ``0..7`` on device 0, ``8..15`` on
  device 1).
* One worker thread per device, created once and bound to its
  ``cupy.cuda.Device`` for its whole life, holding its own per-device resources
  (kernel cache, memory-pool cap).
* Within a segment a worker walks ITS partitions in ascending order, reading
  the shared host ``in`` buffers and scattering owned outputs into the shared
  host ``out`` buffers.  Owned regions are element-disjoint across every
  partition (brief-1 exact cover), so concurrent scatters touch disjoint
  elements and never race.
* After every segment all workers meet a barrier; only then does the *single*
  caller thread swap the buffers.  Truth between segments therefore lives only
  in host global arrays, exactly as in the single-device case.

Because the barrier includes the caller, the swap provably happens after every
partition's scatter and before any partition's next gather.

``serial=True`` runs the identical work in the caller thread, device group by
device group, partitions ascending.  It is the debug arm: results must be
bitwise identical to threaded mode, which is an empirical gate, not an
assumption.

Nothing here imports cupy at module scope, so the assignment law, the barrier
semantics and the manifest/coverage guards are all testable on a CPU-only box.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

__all__ = [
    "SCHEDULER_VERSION",
    "DeviceAssignment",
    "DeviceSegmentScheduler",
    "SchedulerError",
    "DeviceContextLeak",
    "DeviceManifestMismatch",
    "SegmentFailure",
    "FakeDeviceContexts",
    "CupyDeviceContexts",
    "assert_identical_compile_manifests",
    "assert_disjoint_owned_partitions",
    "apply_device_memory_cap",
    "PARTITION_FLOOR_DERIVATION",
    "partition_min_free_bytes",
    "require_devices",
]

SCHEDULER_VERSION = "partition-device-scheduler-v841/1"

_GIB = 1 << 30


class SchedulerError(RuntimeError):
    """Base class for refusals raised by this module."""


class DeviceContextLeak(SchedulerError):
    """A worker was executing against a device other than the one it owns."""


class DeviceManifestMismatch(SchedulerError):
    """Two devices did not produce identical kernel-compilation manifests."""


class SegmentFailure(SchedulerError):
    """One or more device workers raised inside a segment."""

    def __init__(self, label: str, failures: Sequence[tuple[int, BaseException]]):
        self.label = label
        self.failures = tuple(failures)
        detail = "; ".join(
            f"device {device}: {type(exc).__name__}: {exc}" for device, exc in failures
        )
        super().__init__(f"segment {label!r} failed on {len(failures)} device(s): {detail}")


# ---------------------------------------------------------------------------
# assignment law
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceAssignment:
    """Which partitions each device owns.

    ``groups[i]`` is the ascending partition tuple owned by ``devices[i]``.
    """

    devices: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("at least one device is required")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError(f"duplicate device ordinals: {self.devices}")
        if any(int(d) < 0 for d in self.devices):
            raise ValueError(f"device ordinals must be non-negative: {self.devices}")
        if len(self.groups) != len(self.devices):
            raise ValueError("groups and devices must be parallel sequences")
        seen: set[int] = set()
        for group in self.groups:
            if not group:
                raise ValueError(
                    f"device group is empty; {len(self.devices)} devices cannot "
                    "share fewer partitions than devices"
                )
            if list(group) != sorted(group):
                raise ValueError(f"partitions within a device must ascend: {group}")
            for part in group:
                if part in seen:
                    raise ValueError(f"partition {part} assigned to more than one device")
                seen.add(int(part))
        expected = set(range(self.n_partitions))
        if seen != expected:
            missing = sorted(expected - seen)
            extra = sorted(seen - expected)
            raise ValueError(
                "assignment is not an exact cover of 0..P-1 "
                f"(missing={missing}, unexpected={extra})"
            )

    @classmethod
    def contiguous_blocks(
        cls, n_partitions: int, devices: Sequence[int]
    ) -> "DeviceAssignment":
        """Split ``0..n_partitions-1`` into contiguous ascending blocks.

        Equal blocks when the count divides; otherwise the leading blocks carry
        the one extra partition each.  For P=16 and two devices this is the
        brief's ``0..7`` / ``8..15``.
        """

        devices = tuple(int(d) for d in devices)
        n_partitions = int(n_partitions)
        if n_partitions < len(devices):
            raise ValueError(
                f"{n_partitions} partitions cannot fill {len(devices)} devices"
            )
        base, remainder = divmod(n_partitions, len(devices))
        groups: list[tuple[int, ...]] = []
        start = 0
        for index in range(len(devices)):
            size = base + (1 if index < remainder else 0)
            groups.append(tuple(range(start, start + size)))
            start += size
        return cls(devices=devices, groups=tuple(groups))

    @property
    def n_partitions(self) -> int:
        return sum(len(group) for group in self.groups)

    @property
    def n_devices(self) -> int:
        return len(self.devices)

    def partitions_for(self, device: int) -> tuple[int, ...]:
        for candidate, group in zip(self.devices, self.groups):
            if candidate == device:
                return group
        raise KeyError(f"device {device} is not part of this assignment")

    def device_of(self, partition: int) -> int:
        for device, group in zip(self.devices, self.groups):
            if partition in group:
                return device
        raise KeyError(f"partition {partition} is not assigned")

    def receipt(self) -> dict[str, Any]:
        return {
            "scheduler_version": SCHEDULER_VERSION,
            "devices": list(self.devices),
            "n_partitions": self.n_partitions,
            "groups": {
                str(device): list(group)
                for device, group in zip(self.devices, self.groups)
            },
        }


# ---------------------------------------------------------------------------
# device context providers
# ---------------------------------------------------------------------------


class FakeDeviceContexts:
    """CPU stand-in used by the unit tests.

    Tracks a per-thread current-device stack so the same context-leak assertion
    that guards the real run can be exercised without a GPU.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def _stack(self) -> list[int]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = [-1]
            self._local.stack = stack
        return stack

    @contextmanager
    def bind(self, device: int) -> Iterator[None]:
        stack = self._stack()
        stack.append(int(device))
        try:
            yield
        finally:
            stack.pop()

    def current(self) -> int:
        return self._stack()[-1]


class CupyDeviceContexts:
    """The real provider: ``cupy.cuda.Device`` binding plus a runtime read-back."""

    def __init__(self, cupy_module: Any = None) -> None:
        if cupy_module is None:
            import cupy  # local import: this module stays importable without CUDA

            cupy_module = cupy
        self._cp = cupy_module

    def bind(self, device: int) -> Any:
        return self._cp.cuda.Device(int(device))

    def current(self) -> int:
        return int(self._cp.cuda.runtime.getDevice())


# ---------------------------------------------------------------------------
# the scheduler
# ---------------------------------------------------------------------------

WorkFn = Callable[[int, int, Any], None]
"""``work(partition, device, resources) -> None``."""


class DeviceSegmentScheduler:
    """Runs segment bodies across per-device worker threads with a hard barrier.

    Usage::

        with DeviceSegmentScheduler(assignment, contexts=ctx,
                                    resource_factory=make_res) as sched:
            for segment in SEGMENTS:
                sched.run_segment(segment.name, segment.body)
                swap_host_buffers()          # caller thread, single swapper

    ``resource_factory(device)`` is invoked once per device *inside* that
    device's worker thread and inside its device context, so per-device objects
    (kernel cache, memory-pool cap, drivers, backends) are created and used on
    one thread only.
    """

    def __init__(
        self,
        assignment: DeviceAssignment,
        *,
        contexts: Any,
        resource_factory: Callable[[int], Any] | None = None,
        serial: bool = False,
        thread_name_prefix: str = "v841-dev",
    ) -> None:
        self.assignment = assignment
        self.contexts = contexts
        self.resource_factory = resource_factory
        self.serial = bool(serial)
        self._thread_name_prefix = thread_name_prefix

        self._resources: dict[int, Any] = {}
        self._threads: list[threading.Thread] = []
        self._started = False
        self._stopping = False
        self._work: WorkFn | None = None
        self._label = ""
        self._failures: list[tuple[int, BaseException]] = []
        self._failure_lock = threading.Lock()
        self._segment_count = 0

        parties = assignment.n_devices + 1
        self._start_barrier = threading.Barrier(parties)
        self._end_barrier = threading.Barrier(parties)
        self._ready_barrier = threading.Barrier(parties)

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "DeviceSegmentScheduler":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._started:
            raise SchedulerError("scheduler already started")
        self._started = True
        if self.serial:
            for device in self.assignment.devices:
                with self.contexts.bind(device):
                    self._resources[device] = (
                        self.resource_factory(device) if self.resource_factory else None
                    )
            return

        for device in self.assignment.devices:
            thread = threading.Thread(
                target=self._worker_main,
                args=(device,),
                name=f"{self._thread_name_prefix}{device}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        # Wait for every worker to bind its device and build its resources.
        self._ready_barrier.wait()
        if self._failures:
            failures = list(self._failures)
            self._failures.clear()
            self.stop()
            raise SegmentFailure("startup", failures)

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        if self.serial:
            self._resources.clear()
            return
        self._stopping = True
        try:
            self._start_barrier.wait(timeout=60.0)
        except threading.BrokenBarrierError:
            self._start_barrier.abort()
        for thread in self._threads:
            thread.join(timeout=60.0)
        self._threads.clear()
        self._resources.clear()

    # -- the segment call --------------------------------------------------

    def run_segment(self, label: str, work: WorkFn) -> None:
        """Run ``work`` for every partition, then return once ALL have finished.

        On return it is safe -- and required -- for the caller to swap the host
        double buffers.  Any worker exception is re-raised here as
        ``SegmentFailure``; the run is then over (no partial steps).
        """

        if not self._started:
            raise SchedulerError("scheduler is not running")
        self._segment_count += 1
        self._label = label
        self._work = work
        with self._failure_lock:
            self._failures.clear()

        if self.serial:
            for device in self.assignment.devices:
                self._run_group(device, work, label)
        else:
            self._start_barrier.wait()
            self._end_barrier.wait()

        self._work = None
        if self._failures:
            failures = list(self._failures)
            self._failures.clear()
            raise SegmentFailure(label, failures)

    @property
    def segment_count(self) -> int:
        return self._segment_count

    def resources_for(self, device: int) -> Any:
        return self._resources[device]

    # -- internals ---------------------------------------------------------

    def _record_failure(self, device: int, exc: BaseException) -> None:
        with self._failure_lock:
            self._failures.append((device, exc))

    def _run_group(self, device: int, work: WorkFn, label: str) -> None:
        """Execute one device's partitions, ascending, inside its context."""

        resources = self._resources.get(device)
        try:
            with self.contexts.bind(device):
                for partition in self.assignment.partitions_for(device):
                    current = self.contexts.current()
                    if current != device:
                        raise DeviceContextLeak(
                            f"segment {label!r} partition {partition}: worker owns "
                            f"device {device} but the runtime reports device {current}"
                        )
                    work(partition, device, resources)
        except BaseException as exc:  # noqa: BLE001 -- recorded and re-raised by caller
            self._record_failure(device, exc)

    def _worker_main(self, device: int) -> None:
        bound = None
        try:
            bound = self.contexts.bind(device)
            bound.__enter__()
        except BaseException as exc:  # noqa: BLE001
            self._record_failure(device, exc)
            bound = None
        try:
            if bound is not None:
                try:
                    self._resources[device] = (
                        self.resource_factory(device) if self.resource_factory else None
                    )
                except BaseException as exc:  # noqa: BLE001
                    self._record_failure(device, exc)
            self._ready_barrier.wait()

            while True:
                try:
                    self._start_barrier.wait()
                except threading.BrokenBarrierError:
                    return
                if self._stopping:
                    return
                work = self._work
                try:
                    if work is None:
                        raise SchedulerError("worker woke with no segment body")
                    resources = self._resources.get(device)
                    with_ctx_ok = self.contexts.current()
                    if with_ctx_ok != device:
                        raise DeviceContextLeak(
                            f"worker for device {device} entered a segment while the "
                            f"runtime reported device {with_ctx_ok}"
                        )
                    for partition in self.assignment.partitions_for(device):
                        current = self.contexts.current()
                        if current != device:
                            raise DeviceContextLeak(
                                f"segment {self._label!r} partition {partition}: "
                                f"worker owns device {device} but the runtime "
                                f"reports device {current}"
                            )
                        work(partition, device, resources)
                except BaseException as exc:  # noqa: BLE001
                    self._record_failure(device, exc)
                finally:
                    try:
                        self._end_barrier.wait()
                    except threading.BrokenBarrierError:
                        return
        finally:
            if bound is not None:
                try:
                    bound.__exit__(None, None, None)
                except BaseException:  # noqa: BLE001 -- teardown is best effort
                    pass


# ---------------------------------------------------------------------------
# preconditions for a bitwise multi-device run
# ---------------------------------------------------------------------------


def _normalise_manifest(manifest: Any) -> Any:
    if isinstance(manifest, Mapping):
        return {str(key): _normalise_manifest(value) for key, value in sorted(manifest.items())}
    if isinstance(manifest, (list, tuple)):
        return [_normalise_manifest(item) for item in manifest]
    return manifest


def assert_identical_compile_manifests(manifests: Mapping[int, Any]) -> Any:
    """Refuse the run unless every device compiled identically.

    Kernel-binary identity across the cards is a *precondition* of bitwise
    identity, not a consequence of it: two same-model cards can still take
    different driver JIT paths.  Returns the shared manifest.
    """

    if not manifests:
        raise DeviceManifestMismatch("no compile manifests were supplied")
    items = [(int(device), _normalise_manifest(value)) for device, value in manifests.items()]
    items.sort()
    reference_device, reference = items[0]
    for device, manifest in items[1:]:
        if manifest != reference:
            differing = _manifest_diff(reference, manifest)
            raise DeviceManifestMismatch(
                f"device {device} compile manifest differs from device "
                f"{reference_device}: {differing}"
            )
    return reference


def _manifest_diff(left: Any, right: Any) -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = sorted(set(left) | set(right))
        out: list[str] = []
        for key in keys:
            if key not in left:
                out.append(f"+{key}")
            elif key not in right:
                out.append(f"-{key}")
            elif left[key] != right[key]:
                out.append(f"{key}: {left[key]!r} != {right[key]!r}")
        return out
    return [f"{left!r} != {right!r}"]


def assert_disjoint_owned_partitions(layouts: Sequence[Any], assignment: DeviceAssignment) -> dict[str, int]:
    """Prove the owned regions the devices scatter into never overlap.

    Concurrency safety of the shared-host scatter rests entirely on this: every
    global cell / edge / vertex must be owned by exactly one partition.  Brief-1
    asserts the exact cover at asset-build time; this re-asserts it against the
    layouts actually loaded, and additionally reports the cross-device split.
    """

    import numpy as np

    if len(layouts) != assignment.n_partitions:
        raise ValueError(
            f"{len(layouts)} layouts for {assignment.n_partitions} assigned partitions"
        )
    counts: dict[str, int] = {}
    for kind, l2g_attr, owned_attr, total_attr in (
        ("cells", "cell_l2g", "n_owned_cells", "_global_cells"),
        ("edges", "edge_l2g", "n_owned_edges", "_global_edges"),
        ("vertices", "vertex_l2g", "n_owned_vertices", "_global_vertices"),
    ):
        total = int(getattr(layouts[0], total_attr))
        tally = np.zeros(total, dtype=np.int32)
        owner = np.full(total, -1, dtype=np.int32)
        for partition, layout in enumerate(layouts):
            l2g = np.asarray(getattr(layout, l2g_attr))
            owned = l2g[: int(getattr(layout, owned_attr))]
            tally[owned] += 1
            owner[owned] = assignment.device_of(partition)
        duplicated = int(np.count_nonzero(tally > 1))
        missing = int(np.count_nonzero(tally == 0))
        if duplicated or missing:
            raise SchedulerError(
                f"{kind}: owned regions are not an exact cover "
                f"(duplicated={duplicated}, missing={missing}) -- concurrent "
                "scatter would race"
            )
        counts[kind] = total
        for device in assignment.devices:
            counts[f"{kind}_device{device}"] = int(np.count_nonzero(owner == device))
    return counts


# ---------------------------------------------------------------------------
# device admission -- answers from woof.hex.device_admission (the ONE
# admission surface) since 2026-08-25; sm_86 lives here
# ---------------------------------------------------------------------------


def apply_device_memory_cap(device: int, cap_bytes: int, *, cupy_module: Any = None) -> int:
    """Cap the default memory pool for ``device``.  Call inside its context."""

    if cupy_module is None:
        import cupy

        cupy_module = cupy
    with cupy_module.cuda.Device(int(device)):
        cupy_module.get_default_memory_pool().set_limit(size=int(cap_bytes))
    return int(cap_bytes)


#: How the per-partition floor is derived, machine-readably.  Rerouted
#: 2026-08-25 (stale-guard audit #347, finding 5): this module used to be a
#: SECOND admission surface -- a 22 GiB whole-mesh constant scaled linearly
#: through the origin with no fixed term, times a 1.25 multiplier, beside a
#: separate 20 GiB ``min_free_bytes`` default -- the exact shape the L6
#: ``NATIVE_DEVICE_FLOOR`` re-derivation retired everywhere else the same
#: day.  A linear floor is UNDER-protective for a partition: every rank's
#: process pays the card's full fixed term (CUDA context, kernel
#: local-memory backing store, module images) no matter how few cells its
#: partition holds, so a floor with no fixed term admits a small partition
#: on a device that cannot even hold the process.  The floor now answers
#: from ``woof.hex.device_admission`` like every other free-memory gate.
#:
#: The per-partition APPLICATION of that measured row is a DECLARED
#: DERIVATION: the row was fitted from whole-mesh single-GPU #264 sessions,
#: and no 2-GPU #264 ledger row exists at the converged pin, so halo-buffer
#: and exchange residency are modeled as inside the whole-mesh slope rather
#: than measured.  The direction is conservative (a partition's local mesh
#: is charged the full whole-mesh per-cell slope plus the full fixed term),
#: and the first real measurement replaces this label.
PARTITION_FLOOR_DERIVATION: Mapping[str, str] = {
    "status": "DERIVED, NOT MEASURED",
    "derived": "2026-08-25",
    "model": "hexcore.device_admission.FOOTPRINT_MODEL",
    "sum": "required_free_bytes(n_local_cells): affine row + shared headroom",
    "replaces": (
        "22 GiB whole-mesh constant * (local/global) * 1.25 -- linear, no "
        "fixed term, beside a separate 20 GiB require_devices default; "
        "retired 2026-08-25 with the L6 floor re-derivation"
    ),
    "replaced_by": (
        "one 2-GPU #264 ledger run at the converged pin (hex 7fe514b + "
        "engine 26daaab7e) recording per-device peak and pool high-water; "
        "until it runs, this row must be quoted with its label"
    ),
}


def partition_min_free_bytes(
    n_local_cells: int,
    n_global_cells: int,
    *,
    model: Any = None,
    headroom_bytes: int | None = None,
) -> int:
    """The admission floor for ONE partition's device: the ONE admission sum.

    ``device_admission.required_free_bytes`` over the partition's LOCAL
    (owned + halo) cell count -- the measured affine row plus the shared
    headroom, the same sum the forecast door and the single-GPU driver
    admit on.  See :data:`PARTITION_FLOOR_DERIVATION` for why applying the
    whole-mesh row per partition is a labeled derivation, and for the
    retired linear shape this replaces.
    """

    from . import device_admission

    if n_local_cells <= 0 or n_global_cells <= 0:
        raise ValueError("cell counts must be positive")
    if n_local_cells > n_global_cells:
        raise ValueError("a partition cannot exceed the global mesh")
    return device_admission.required_free_bytes(
        int(n_local_cells), model, headroom_bytes
    )


def require_devices(
    devices: Sequence[int],
    *,
    min_compute: tuple[int, int] = (8, 6),
    min_free_bytes: int | None = None,
    cupy_module: Any = None,
) -> list[dict[str, Any]]:
    """Admission for the partitioned tools.

    ``min_free_bytes`` defaults to the ONE admission surface's whole-mesh
    floor (``device_admission.native_device_floor_bytes()``, the re-derived
    measured requirement) rather than an asserted constant; partitioned
    callers MUST pass the :func:`partition_min_free_bytes` scaled floor
    instead (the whole-mesh default refuses a 16 GB card whose partition
    share is a fraction of it).  Returns a per-device receipt.
    """

    if min_free_bytes is None:
        from . import device_admission

        min_free_bytes = device_admission.native_device_floor_bytes()
    if cupy_module is None:
        import cupy

        cupy_module = cupy
    count = cupy_module.cuda.runtime.getDeviceCount()
    receipts: list[dict[str, Any]] = []
    for device in devices:
        device = int(device)
        if device >= count:
            raise SchedulerError(f"device {device} requested but only {count} present")
        with cupy_module.cuda.Device(device):
            props = cupy_module.cuda.runtime.getDeviceProperties(device)
            major = int(props["major"])
            minor = int(props["minor"])
            free, total = cupy_module.cuda.runtime.memGetInfo()
        is_hip = bool(getattr(cupy_module.cuda.runtime, "is_hip", False))
        if not is_hip and (major, minor) < tuple(min_compute):
            raise SchedulerError(
                f"device {device} compute capability {major}.{minor} < required "
                f"{min_compute[0]}.{min_compute[1]}"
            )
        if int(free) < int(min_free_bytes):
            raise SchedulerError(
                f"device {device} has {int(free) / _GIB:.2f} GiB free, "
                f"{min_free_bytes / _GIB:.2f} GiB required"
            )
        name = props["name"]
        receipts.append(
            {
                "device": device,
                "name": name.decode() if isinstance(name, bytes) else str(name),
                "compute_capability": None if is_hip else f"{major}.{minor}",
                **({"architecture": __import__("woof.hip_backend", fromlist=["architecture"]).architecture(cupy_module, device), "backend": "hip"} if is_hip else {}),
                "free_bytes": int(free),
                "total_bytes": int(total),
            }
        )
    return receipts
