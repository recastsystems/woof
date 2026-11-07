"""Continuously observed GPU-memory high-water marks for run receipts.

The forecast runners used to assemble ``gpu_peak_used_bytes_observed``
from samples taken only inside step-boundary callbacks (the history
handler and the per-period progress callback).  ``execute_experiment``
trims the CuPy pool per STEP op and again at period commit BEFORE the
progress callback fires, so every sample landed after the intra-step
transient working set had already been released: the receipt reported
19.41 GiB against a 22.34 GiB true high-water mark on the four-domain
tree shape (-13%).  A receipt that understates VRAM is worse than no
receipt -- it invites launching a shape that demotes or OOMs.

:class:`GpuPeakMemoryWatcher` closes the gap with a lightweight daemon
thread that folds probe readings into running maxima every
``interval_seconds``, alongside the explicit boundary and end-of-run
``sample()`` calls the runners already made.  Probes are injected so
the accounting path is testable CPU-side, and each probe carries an
accurate ``scope`` label saying WHAT its number measures -- the
``cudaMemGetInfo`` view is not the whole card on a WDDM host (see
:func:`woof.core.preflight.device_wide_used_bytes`), and the CuPy pool
views are in-process only.

WHOSE BYTES.  Every view above is either in-process (the pool) or
device-wide (``cudaMemGetInfo`` and NVML's ``memory.used`` both answer
for the whole card, every process included).  A receipt built from
those alone cannot tell a run that grew from a card that filled up
underneath it: a 5 h pair of nested runs on a 32 GB card was read as
"one leg climbs to 29.8 GB and runs 2.3x slower", when the leg itself sat at
8.2-8.6 GB throughout and a second process was holding 15-17.8 GB of the
same card and running kernels on it for 98% of the leg.  The
:func:`nvidia_smi_process_probes` views split the card by process from
NVML's per-process accounting (``nvidia-smi --query-compute-apps``): the
bytes THIS process holds, the bytes every OTHER process holds, and the
device-wide figure, plus -- through the watcher's time-weighted
``nonzero_seconds`` -- how long the card was shared.  They are sampled
on their own slow cadence (one ``nvidia-smi`` pass every few seconds)
and are never strict: a host without ``nvidia-smi`` on PATH, or a WDDM
driver that reports per-process memory as N/A, records that fact in the
receipt and the forecast runs on.

WHO WAITS FOR NVIDIA-SMI.  Only a background thread of its own.  The
NVML views are ``background_only`` probes: the step-boundary
``sample()`` the runners call on the forecast's main thread never reads
them, and the 20 Hz peak thread never reads them either.  Measured
defect this replaces (2026-10-05, shared 8-card box): ``sample()`` ran
the due NVML probes inline, two ``nvidia-smi`` subprocesses every 2 s on
the main thread; on a busy box one call takes 0.5 s to several seconds
(the process sits in D state), and a 750 m city run that stepped in
0.02 s per step took 4.5-8.8 s per outer step -- 240 s per forecast
hour against about 24.  On an idle single-card host the same pair still
cost about 2 x 50 ms of main-thread wall every 2 s.  The receipts keep
every view; what changed is WHEN the NVML views are read: on the NVML
thread's own 2 s cadence from ``start()`` until ``stop()``, no longer
also at a boundary sample, and a pass still in flight when ``stop()`` is
called is dropped rather than waited for (the summary says so).

Overhead: one probe pass per tick.  The watcher machinery itself
measures in single-digit microseconds per pass CPU-side; at the default
20 Hz the device-facing cost is one ``cudaMemGetInfo`` plus two pool
counter reads per tick -- the same queries the boundary callbacks
already issued, now on a fixed, low cadence instead of per event.  The
NVML views add one ``nvidia-smi`` subprocess pair per
:data:`NVIDIA_SMI_INTERVAL_SECONDS` from their own daemon thread, each
call bounded by :data:`NVIDIA_SMI_TIMEOUT_SECONDS`; the main thread's
boundary ``sample()`` costs three in-process counter reads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import os
import subprocess
import threading
import time


#: Poll cadence for the background sampler.  Root periods run for
#: seconds on nested shapes, and the transients worth catching persist
#: for a meaningful fraction of a step, so 50 ms resolves them while
#: keeping the query load negligible.
DEFAULT_INTERVAL_SECONDS = 0.05

#: Cadence of the per-process NVML views.  ``nvidia-smi`` is a
#: subprocess costing tens of milliseconds; a foreign process that
#: shares the card does so for minutes, so a reading every two seconds
#: resolves it while costing well under one percent of a host core.
NVIDIA_SMI_INTERVAL_SECONDS = 2.0

#: Upper bound on one ``nvidia-smi`` call, paid only by the NVML thread.
#: A call that runs past it is a TRANSIENT failure (a busy box, the
#: driver in D state): it is counted in the receipt and the next pass
#: tries again, instead of retiring the per-process views for the rest
#: of the run.
NVIDIA_SMI_TIMEOUT_SECONDS = 10.0

_MIB = 1024 ** 2


@dataclass(frozen=True)
class MemoryProbe:
    """One byte counter plus the accurate description of its scope.

    ``interval_seconds`` is the probe's own cadence; ``None`` reads it on
    every watcher tick and every boundary sample.  ``strict`` probes
    fail a boundary ``sample()`` loud (the pool and runtime views, which
    cannot fail on a working device); a non-strict probe's error is
    recorded for the receipt and the probe retired, on either path,
    unless it is a :class:`TransientProbeError`, which is counted and
    retried on the next due pass.

    ``background_only`` marks a probe whose read may block (a
    subprocess, a slow driver query): it is read only on the watcher's
    dedicated background thread, never by ``sample()`` -- which runs on
    the forecast's main thread -- and never by the fast peak thread.
    Such a probe cannot be strict: nothing on a boundary path reads it.
    """

    name: str
    scope: str
    read: Callable[[], int]
    interval_seconds: float | None = None
    strict: bool = True
    background_only: bool = False


def default_cupy_probes() -> tuple[MemoryProbe, ...]:
    """The three views a CuPy-backed forecast can report.

    Imports CuPy lazily so that constructing probes (and every CPU-side
    test of the watcher) never touches the GPU stack.
    """
    import cupy as cp

    def cuda_device_used() -> int:
        free, total = cp.cuda.runtime.memGetInfo()
        return int(total - free)

    def pool_total() -> int:
        return int(cp.get_default_memory_pool().total_bytes())

    def pool_used() -> int:
        return int(cp.get_default_memory_pool().used_bytes())

    return (
        MemoryProbe(
            name="cuda_device_used",
            scope=(
                "cudaMemGetInfo total-free: device bytes the CUDA runtime "
                "sees as used.  NOT the whole card on a WDDM host -- other "
                "processes' residency (e.g. the desktop compositor) is "
                "invisible to it, though its deltas track the NVML "
                "device-wide series closely "
                "(woof.core.preflight.device_wide_used_bytes)"),
            read=cuda_device_used),
        MemoryProbe(
            name="cupy_pool_total",
            scope="in-process CuPy default-pool bytes held (live + cached)",
            read=pool_total),
        MemoryProbe(
            name="cupy_pool_used",
            scope="in-process CuPy default-pool bytes live",
            read=pool_used),
    )


class ProcessMemoryUnavailable(RuntimeError):
    """NVML cannot attribute device memory to processes on this host."""


class TransientProbeError(RuntimeError):
    """A probe read that failed for now but may succeed on the next pass.

    The watcher counts it in the receipt and does NOT retire the probe.
    """


def _run_nvidia_smi_text(arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            [*__import__('woof.gpu_smi', fromlist=['command']).command(), *arguments], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise TransientProbeError(
            f"nvidia-smi {' '.join(arguments)} ran past "
            f"{NVIDIA_SMI_TIMEOUT_SECONDS:g} s") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:200]
        raise ProcessMemoryUnavailable(
            f"nvidia-smi {' '.join(arguments)} exited "
            f"{result.returncode}: {detail}")
    return result.stdout


def _mib_field(text: str, *, what: str) -> int:
    value = text.strip()
    if not value or value.upper() in ("[N/A]", "N/A", "[NOT SUPPORTED]"):
        shown = repr(value) if value else "empty"
        raise ProcessMemoryUnavailable(
            f"nvidia-smi reports {what} as {shown}; NVML does not "
            "attribute device memory per process on this host (WDDM "
            "drivers answer N/A here)")
    return int(float(value))


@dataclass(frozen=True)
class NvidiaSmiReading:
    """One consistent snapshot of NVML's per-process and device views."""

    this_process_bytes: int
    other_processes_bytes: int
    device_used_bytes: int
    other_pids: tuple[int, ...]

    @property
    def shared(self) -> bool:
        return self.other_processes_bytes > 0


def parse_nvidia_smi_views(compute_apps_csv: str, gpus_csv: str, *,
                           pid: int) -> NvidiaSmiReading:
    """Split NVML's process list by ``pid`` on the card that carries it.

    ``compute_apps_csv`` is ``--query-compute-apps=pid,used_memory,gpu_uuid``
    and ``gpus_csv`` is ``--query-gpu=uuid,memory.used``, both
    ``--format=csv,noheader,nounits`` (MiB).  A process appears on the
    GPU it holds a context on; before this process has created its
    context (or on a host that lists nothing) the first GPU stands in,
    so a foreign process already resident there is still reported.
    """
    apps: list[tuple[int, int, str]] = []
    for line in compute_apps_csv.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            raise ProcessMemoryUnavailable(
                f"unexpected nvidia-smi compute-apps row {line!r}")
        row_pid = int(parts[0])
        used = _mib_field(parts[1], what="a process's used_memory")
        uuid = parts[2] if len(parts) > 2 else ""
        apps.append((row_pid, used, uuid))
    devices: list[tuple[str, int]] = []
    for line in gpus_csv.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            raise ProcessMemoryUnavailable(
                f"unexpected nvidia-smi gpu row {line!r}")
        devices.append((parts[0], _mib_field(parts[1], what="memory.used")))
    if not devices:
        raise ProcessMemoryUnavailable("nvidia-smi listed no GPU")
    ours = [row for row in apps if row[0] == pid]
    card = ours[0][2] if ours else devices[0][0]
    on_card = [row for row in apps if row[2] == card or not row[2]]
    this_bytes = sum(used for row_pid, used, _ in on_card if row_pid == pid)
    others = [(row_pid, used) for row_pid, used, _ in on_card
              if row_pid != pid]
    device_used = next(
        (used for uuid, used in devices if uuid == card), devices[0][1])
    return NvidiaSmiReading(
        this_process_bytes=this_bytes * _MIB,
        other_processes_bytes=sum(used for _, used in others) * _MIB,
        device_used_bytes=device_used * _MIB,
        other_pids=tuple(sorted({row_pid for row_pid, _ in others})))


class _NvidiaSmiSnapshot:
    """One nvidia-smi pass shared by the three per-process probes.

    The watcher reads the probes of a pass within microseconds of each
    other; caching the snapshot for ``max_age_seconds`` keeps that pass
    at one pair of subprocess calls instead of three.  A failed pass is
    cached the same way, so a timed-out ``nvidia-smi`` is not re-run by
    the pass's second and third probe.
    """

    def __init__(self, *, pid: int, run: Callable[[list[str]], str],
                 max_age_seconds: float = 0.5):
        self._pid = int(pid)
        self._run = run
        self._max_age = float(max_age_seconds)
        self._lock = threading.Lock()
        self._reading: NvidiaSmiReading | None = None
        self._error: BaseException | None = None
        self._read_at = float("-inf")

    def reading(self) -> NvidiaSmiReading:
        with self._lock:
            if time.perf_counter() - self._read_at > self._max_age:
                self._reading = None
                self._error = None
                try:
                    apps = self._run([
                        "--query-compute-apps=pid,used_memory,gpu_uuid",
                        "--format=csv,noheader,nounits"])
                    gpus = self._run([
                        "--query-gpu=uuid,memory.used",
                        "--format=csv,noheader,nounits"])
                    self._reading = parse_nvidia_smi_views(
                        apps, gpus, pid=self._pid)
                except BaseException as error:  # noqa: BLE001 - re-raised
                    self._error = error
                # Stamped after the calls: a slow pass must not age out
                # before its sibling probes read it.
                self._read_at = time.perf_counter()
            if self._error is not None:
                raise self._error
            assert self._reading is not None
            return self._reading


def nvidia_smi_process_probes(
        *, pid: int | None = None,
        interval_seconds: float = NVIDIA_SMI_INTERVAL_SECONDS,
        run: Callable[[list[str]], str] | None = None,
) -> tuple[MemoryProbe, ...]:
    """The per-process split of the card, from NVML via ``nvidia-smi``.

    ``run`` is injectable for CPU-side tests; ``pid`` defaults to this
    process.  The probes are non-strict, share one slow cadence, and are
    ``background_only``: a subprocess is never run on the thread that
    calls the watcher's ``sample()``.
    """
    snapshot = _NvidiaSmiSnapshot(
        pid=os.getpid() if pid is None else int(pid),
        run=_run_nvidia_smi_text if run is None else run)
    return (
        MemoryProbe(
            name="nvml_this_process_used",
            scope=("NVML per-process accounting (nvidia-smi "
                   "--query-compute-apps): device bytes THIS process holds"),
            read=lambda: snapshot.reading().this_process_bytes,
            interval_seconds=interval_seconds, strict=False,
            background_only=True),
        MemoryProbe(
            name="nvml_other_processes_used",
            scope=("NVML per-process accounting: device bytes held by "
                   "every OTHER process on the same card; its "
                   "nonzero_seconds is how long the card was shared"),
            read=lambda: snapshot.reading().other_processes_bytes,
            interval_seconds=interval_seconds, strict=False,
            background_only=True),
        MemoryProbe(
            name="nvml_device_used",
            scope=("NVML device-wide memory.used of the card this process "
                   "runs on: every process included"),
            read=lambda: snapshot.reading().device_used_bytes,
            interval_seconds=interval_seconds, strict=False,
            background_only=True),
    )


class GpuPeakMemoryWatcher:
    """Running maxima over injected byte probes, thread- and event-fed.

    ``sample()`` is the strict path the runners call at step boundaries
    and at end of run: a strict probe's error propagates, exactly as the
    old ``record_memory`` closures failed loud.  The daemon thread is the
    lenient path: a probe that breaks mid-run must not kill a forecast,
    so its first error is recorded for the receipt and that probe is
    retired while its siblings keep sampling.  Non-strict probes take
    the lenient path everywhere.  Either way the receipt can say what
    was measured, how often, and whether observation was complete.

    Two daemon threads: the fast one polls the in-process probes every
    ``interval_seconds``; a second one owns every ``background_only``
    probe (the ``nvidia-smi`` views), so a subprocess that takes seconds
    on a busy box neither stalls the fast peak thread nor ever runs on
    the caller of ``sample()``.

    Beside the peak, every probe accumulates ``nonzero_seconds``: the
    wall time over which its last reading was above zero, weighted by
    the gap to the next reading.  For a view of OTHER processes' bytes
    that is the time the card was shared.
    """

    _MECHANISM = "background-polling-thread+boundary-samples"

    def __init__(self, probes: Iterable[MemoryProbe], *,
                 interval_seconds: float = DEFAULT_INTERVAL_SECONDS):
        probes = tuple(probes)
        if not probes:
            raise ValueError("at least one memory probe is required")
        names = [probe.name for probe in probes]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate probe names: {names}")
        if not interval_seconds > 0.0:
            raise ValueError("interval_seconds must be positive")
        for probe in probes:
            if (probe.interval_seconds is not None
                    and not probe.interval_seconds > 0.0):
                raise ValueError(
                    f"probe {probe.name!r}: interval_seconds must be "
                    "positive")
            if probe.background_only and probe.strict:
                raise ValueError(
                    f"probe {probe.name!r}: a background_only probe is "
                    "never read on a boundary path, so it cannot be strict")
        self._probes = probes
        self._boundary_probes = tuple(
            probe for probe in probes if not probe.background_only)
        self._background_probes = tuple(
            probe for probe in probes if probe.background_only)
        self._interval = float(interval_seconds)
        self._lock = threading.Lock()
        self._peaks = {probe.name: 0 for probe in probes}
        self._counts = {probe.name: 0 for probe in probes}
        self._last_value: dict[str, int] = {}
        self._last_at: dict[str, float] = {}
        self._attempted_at: dict[str, float] = {}
        self._nonzero_seconds = {probe.name: 0.0 for probe in probes}
        self._errors: dict[str, str] = {}
        self._transient = {probe.name: 0 for probe in probes}
        self._last_transient: dict[str, str] = {}
        self._dropped_at_stop = 0
        self._background_busy = False
        self._in_flight_at_stop = False
        self._stop_event = threading.Event()
        self._started = False
        self._threads: list[threading.Thread] = []
        self._background_thread: threading.Thread | None = None

    # -- sampling ---------------------------------------------------------

    def _fold(self, name: str, value: int) -> None:
        now = time.perf_counter()
        with self._lock:
            if value > self._peaks[name]:
                self._peaks[name] = value
            self._counts[name] += 1
            previous_at = self._last_at.get(name)
            if (previous_at is not None
                    and self._last_value.get(name, 0) > 0):
                self._nonzero_seconds[name] += max(0.0, now - previous_at)
            self._last_value[name] = value
            self._last_at[name] = now

    def _due(self, probe: MemoryProbe) -> bool:
        if probe.interval_seconds is None:
            return True
        with self._lock:
            last = self._attempted_at.get(probe.name)
        return last is None or (
            time.perf_counter() - last >= probe.interval_seconds)

    def _attempt(self, probe: MemoryProbe) -> None:
        if probe.interval_seconds is not None:
            with self._lock:
                self._attempted_at[probe.name] = time.perf_counter()

    def _read_lenient(self, probe: MemoryProbe) -> None:
        with self._lock:
            retired = probe.name in self._errors
        if retired:
            return
        self._attempt(probe)
        try:
            value = int(probe.read())
        except TransientProbeError as error:
            with self._lock:
                self._transient[probe.name] += 1
                self._last_transient[probe.name] = str(error)
            return
        except BaseException as error:  # noqa: BLE001 - receipt-recorded
            with self._lock:
                self._errors[probe.name] = (
                    f"{type(error).__name__}: {error}")
            return
        if probe.background_only and self._stop_event.is_set():
            # The pass outlived stop(): the receipt is what was observed
            # while the run was being watched, not a reading taken after.
            with self._lock:
                self._dropped_at_stop += 1
            return
        self._fold(probe.name, value)

    def sample(self) -> None:
        """Strict pass over every due boundary probe; the first strict
        error propagates.

        Never reads a ``background_only`` probe: this is called on the
        forecast's main thread at step boundaries, and nothing here may
        wait on a subprocess.  Healthy probes are folded before the
        error is raised, so a boundary sample never discards
        information it already read.
        """
        first_error: BaseException | None = None
        for probe in self._boundary_probes:
            if not self._due(probe):
                continue
            if not probe.strict:
                self._read_lenient(probe)
                continue
            self._attempt(probe)
            try:
                value = int(probe.read())
            except BaseException as error:  # noqa: BLE001 - re-raised below
                if first_error is None:
                    first_error = error
                continue
            self._fold(probe.name, value)
        if first_error is not None:
            raise first_error

    def _all_retired(self, probes: tuple[MemoryProbe, ...]) -> bool:
        with self._lock:
            return all(probe.name in self._errors for probe in probes)

    def _loop(self) -> None:
        while not self._stop_event.wait(self._interval):
            for probe in self._boundary_probes:
                if self._due(probe):
                    self._read_lenient(probe)
            if self._all_retired(self._boundary_probes):
                return  # every probe retired; nothing left to watch

    def _background_loop(self) -> None:
        # Reads first, then waits: the run's first NVML pass lands as
        # early as the driver answers.
        while True:
            for probe in self._background_probes:
                if self._stop_event.is_set():
                    return
                if self._due(probe):
                    self._background_busy = True
                    try:
                        self._read_lenient(probe)
                    finally:
                        self._background_busy = False
            if self._all_retired(self._background_probes):
                return
            if self._stop_event.wait(self._interval):
                return

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._started:
            raise RuntimeError("the memory watcher was already started")
        self._started = True
        if self._boundary_probes:
            thread = threading.Thread(
                target=self._loop, name="gpu-mem-watch", daemon=True)
            self._threads.append(thread)
            thread.start()
        if self._background_probes:
            self._background_thread = threading.Thread(
                target=self._background_loop, name="gpu-mem-watch-nvml",
                daemon=True)
            self._background_thread.start()

    def stop(self) -> None:
        """Idempotent; safe whether or not the threads ever started.

        Joins the fast thread (in-process reads only).  Does NOT wait
        for the background thread: a pass still inside ``nvidia-smi``
        finishes on its own, and its reading is dropped and counted.
        """
        if not self._stop_event.is_set():
            self._in_flight_at_stop = bool(self._background_busy)
        self._stop_event.set()
        for thread in self._threads:
            thread.join()

    # -- reporting --------------------------------------------------------

    def peak_bytes(self, name: str) -> int:
        with self._lock:
            return self._peaks[name]

    def observed(self, name: str) -> bool:
        """True once the probe has folded at least one reading."""
        with self._lock:
            return self._counts[name] > 0

    def peak_bytes_observed(self, name: str) -> int | None:
        """The peak, or ``None`` for a probe that never read anything --
        a receipt must not print 0 bytes for "could not be measured"."""
        with self._lock:
            return self._peaks[name] if self._counts[name] > 0 else None

    def nonzero_seconds(self, name: str) -> float:
        with self._lock:
            return self._nonzero_seconds[name]

    def nonzero_seconds_observed(self, name: str) -> float | None:
        with self._lock:
            if self._counts[name] == 0:
                return None
            return self._nonzero_seconds[name]

    def summary(self) -> dict[str, object]:
        """Receipt-facing provenance: mechanism, cadence, per-probe
        peaks with their accurate scope labels, sample counts, time above
        zero, where each probe is read, and any mid-run observation
        failure."""
        with self._lock:
            return {
                "mechanism": self._MECHANISM,
                "interval_seconds": self._interval,
                "background_pass_in_flight_at_stop": self._in_flight_at_stop,
                "background_readings_dropped_at_stop": self._dropped_at_stop,
                "probes": {
                    probe.name: {
                        "peak_bytes": self._peaks[probe.name],
                        "scope": probe.scope,
                        "samples": self._counts[probe.name],
                        "interval_seconds": (
                            self._interval if probe.interval_seconds is None
                            else probe.interval_seconds),
                        "read_on": (
                            "background-thread-only"
                            if probe.background_only
                            else "boundary-samples+background-thread"),
                        "nonzero_seconds": round(
                            self._nonzero_seconds[probe.name], 3),
                        "last_bytes": self._last_value.get(probe.name),
                        "error": self._errors.get(probe.name),
                        "transient_errors": self._transient[probe.name],
                        "last_transient_error": self._last_transient.get(
                            probe.name),
                    }
                    for probe in self._probes
                },
            }


def process_memory_receipt(watch: GpuPeakMemoryWatcher) -> dict[str, object]:
    """The receipt rows that say WHOSE bytes the card carried.

    ``None`` where NVML could not attribute memory per process (no
    ``nvidia-smi``, or a WDDM driver answering N/A); the probe's error
    string sits in the watcher summary beside it.
    """
    shared_seconds = watch.nonzero_seconds_observed(
        "nvml_other_processes_used")
    return {
        "this_process_peak_bytes_nvml": watch.peak_bytes_observed(
            "nvml_this_process_used"),
        "other_processes_peak_bytes_nvml": watch.peak_bytes_observed(
            "nvml_other_processes_used"),
        "device_wide_peak_bytes_nvml": watch.peak_bytes_observed(
            "nvml_device_used"),
        "card_shared_seconds": (
            None if shared_seconds is None else round(shared_seconds, 1)),
        "card_shared": (
            None if shared_seconds is None else bool(shared_seconds > 0.0)),
    }
