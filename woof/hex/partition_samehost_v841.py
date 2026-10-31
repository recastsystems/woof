"""Concurrent resident ranks sharing one model clock and byte-only halos.

The driver keeps its existing acoustic, RK and transport round sites. Each
rank has its own CUDA compute stream, including when several ranks share a
card for the partition-invariance check. No field arithmetic lives here.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import threading
import time
from typing import Any, Callable, Sequence

from .partition_executor_v841 import ExecutorError, HaloExchanger, expected_round_sequence


class SameHostHaloExchanger(HaloExchanger):
    """Use the existing driver callbacks and schedule on the same-host link."""

    # mem/state-levels: every in-step round (B, C, D, E) receives into the
    # arrays it is handed -- the recovered RK candidate, the acoustic state,
    # the transport stage block, the limiter scales -- and never into the
    # step-start level; A and P run outside the step.  The v8.4.1 driver may
    # therefore bind its start level instead of copying it
    # (CudaDryDycoreDriver._shares_start_levels).
    start_levels_read_only = True

    def __init__(self, tables: Any, transport: Any, *, expected_sequence=None):
        self.tables = tables
        self.link = transport
        self._expected = (expected_round_sequence() if expected_sequence is None
                          else tuple(expected_sequence))
        self.fct_scale_parts = self._expected.count("E")
        self._step = None
        self._observed = []
        self.stats = {
            letter: {"rounds": 0, "bytes_sent": 0, "bytes_received": 0, "seconds": 0.0}
            for letter in "ABCDEP"
        }

    def _physics_fields(self, state, saved):
        # The adapter prepares every local column, including K3 storage
        # outside the native thermodynamic K2 exchange. Those incomplete
        # outer rows must receive their owner's state before physics reads
        # them. Distinct packet names select the complete layout indices;
        # native B/C/D/E dynamics packets retain their field-specific widths.
        return [("physics/" + name, array, kind)
                for name, array, kind in self._state_diag_fields(
                    state, saved, include_scalars=True)]

    def round_physics_candidate(self, atmosphere):
        """P: owner truth for the unpublished post-RK physics input."""
        self._round("P", self._physics_fields(atmosphere.state, atmosphere.saved))

    def round_step_boundary(self, state, saved, dynamics_tendencies=None):
        """A: owner truth before the next phase-one physics preparation.

        Physics runs on every local column, so every input it reads must be
        owner truth there. The GF advective forcing (rthdynten, rqvdynten) is
        formed from the step's dynamics tendencies over all local cells, and
        the outer halo rows' tendencies have incomplete stencils: unexchanged,
        the next step's GF on seam columns read them and a 2-card run left the
        1-card run at step 2, at the partition seam.
        """
        fields = self._physics_fields(state, saved)
        if dynamics_tendencies is not None:
            fields += [("physics/" + name, getattr(dynamics_tendencies, name), "cell")
                       for name in ("rthdynten", "rqvdynten")]
        self._round("A", fields)

    def _round(self, letter, fields, *, interior_work=None):
        if self._step is None:
            raise ExecutorError(f"round {letter} issued outside an open step")
        position = len(self._observed)
        expected = self._expected[position] if position < len(self._expected) else "<end>"
        if expected != letter:
            error = ExecutorError(f"step {self._step} round {position}: expected {expected}, got {letter}")
            self.link.abort(str(error))
            raise error
        started = time.perf_counter()
        # Native regional arrays retain one allocation-only garbage column.
        # Halo indexes describe real mesh elements, so leave that pool slot
        # untouched while exchanging a strided view of the real columns.
        real_fields = []
        for name, array, kind in fields:
            extent = getattr(self.tables, "n_local_cells" if kind == "cell" else "n_local_edges")
            if array.shape[-1] == extent + 1:
                array = array[..., :extent]
            real_fields.append((name, array, kind))
        extra = {} if interior_work is None else {"interior_work": interior_work}
        moved = self.link.exchange(self.tables.rank, f"{self._step}:{position}:{letter}", real_fields, **extra)
        stats = self.stats[letter]
        stats["rounds"] += 1
        if moved:
            stats["bytes_sent"] += moved["bytes_sent"]
            stats["bytes_received"] += moved["bytes_received"]
        stats["seconds"] += time.perf_counter() - started
        self._observed.append(letter)

    def round_acoustic(self, acoustic, *, interior_work=None):
        self._round("C", [("rho_pp", acoustic.rho_pp, "cell"),
                          ("rtheta_pp", acoustic.rtheta_pp, "cell"),
                          ("ru_p", acoustic.ru_p, "edge")], interior_work=interior_work)


def _bind_rank_thread(cp: Any, device: int) -> None:
    """Optionally pin a rank's worker thread to its card's NUMA-local CPUs.

    ``WOOF_HEX_RANK_AFFINITY=numa`` reads the card's ``local_cpulist`` from
    sysfs and binds the calling thread there, so its kernel launches, pinned
    staging and Python objects stay on the socket that owns the card's PCIe
    root.  Default ``off``; any failure to read or bind leaves the thread
    unbound (placement is a speed choice, never a refusal).
    """
    import os
    if os.environ.get("WOOF_HEX_RANK_AFFINITY", "off") != "numa" or cp is None:
        return
    try:
        bus = str(cp.cuda.Device(device).pci_bus_id).lower()
        text = open(f"/sys/bus/pci/devices/{bus}/local_cpulist").read().strip()
        cpus = set()
        for part in text.split(","):
            low, _, high = part.partition("-")
            cpus.update(range(int(low), int(high or low) + 1))
        allowed = os.sched_getaffinity(0)
        chosen = cpus & allowed
        if chosen:
            os.sched_setaffinity(0, chosen)
    except (OSError, ValueError, AttributeError):
        return


def warn_serialised_ranks(ranks: int, reason: str = "") -> bool:
    """Say once, on stderr, when rank threads on several cards share one interpreter lock.

    THE BREAKAGE THIS NAMES (woof.hex.rank_processes): on a Python with the GIL every rank thread's host launch
    work shares one core, so more cards can be slower (8x RTX 5090, 2026-10-28: 4 cards 0.70 s/step against
    2 cards 0.51).  Both physics routes run one process per card by default on Linux; rank threads remain where
    they are asked for (--samehost-ranks threads), where the host cannot run processes, and on the active
    full-step LTS rank group, so those say so instead of slowing down silently.
    """
    import os
    import sys
    gil = getattr(sys, "_is_gil_enabled", None)
    if (gil is not None and not gil()) or os.environ.get("WOOF_HEX_RANK_GIL_WARNED"):
        return False
    os.environ["WOOF_HEX_RANK_GIL_WARNED"] = "1"
    print(f"woof hex: {ranks} same-host ranks run as threads of one process on a Python with the GIL"
          f"{' (' + reason + ')' if reason else ''}, so their host work shares one core and more cards can be "
          "slower; the default (--samehost-ranks auto, on Linux) runs one process per card, and a free-threaded "
          "Python (3.14t) lifts this for rank threads", file=sys.stderr, flush=True)
    return True


class SameHostRanks:
    """A dedicated worker, device and stream per rank, and one committed clock.

    A failed rank aborts the transport so peers cannot remain in a halo
    rendezvous. A failed step never advances the coordinator clock. The
    caller must discard the run after a failure; no rank can continue alone.
    """

    def __init__(self, devices: Sequence[int], transport: Any, *, cp: Any = None, physics_scope: Any = None,
                 streams: Sequence[Any] | None = None):
        self.devices = tuple(int(device) for device in devices)
        if not self.devices or min(self.devices) < 0:
            raise ValueError("at least one non-negative device ordinal is required")
        if cp is not None and len(set(self.devices)) > 1:
            # Ranks on distinct cards share one process: the engine's land kernels must be held per card
            # (woof.hex.engine_per_card names the breakage).  Idempotent; the forecast door binds it earlier.
            # Real CUDA contexts only: a stand-in cp must never rebind the engine.
            from .engine_per_card import bind_engine_modules_per_card
            bind_engine_modules_per_card(cp)
        if cp is not None and len(set(self.devices)) > 1:
            warn_serialised_ranks(len(self.devices))
        if streams is not None and (cp is None or len(streams) != len(self.devices)):
            # Prevents a rank running on another rank's queue, or a CPU group
            # holding CUDA streams.
            raise ValueError("caller streams need CuPy and exactly one stream per rank")
        self.transport = transport
        self.cp = cp
        self.physics_scope = physics_scope
        self.time_seconds = None
        self.failed = False
        self._failure_lock = threading.Lock()
        self._first_error = None
        self._initialization_errors = [None] * len(self.devices)
        self.sync_seconds = [0.0] * len(self.devices)
        self.sync_calls = [0] * len(self.devices)
        self.streams = []
        # A caller may hand in the rank streams it already built the ranks on,
        # so construction and stepping share one CuPy pool arena per rank
        # (woof.hex.drivers.samehost_forecast_v841, WOOF_HEX_SAMEHOST_SETUP_STREAM).
        for device in (() if streams is not None else self.devices):
            if cp is None:
                self.streams.append(None)
            else:
                with cp.cuda.Device(device):
                    self.streams.append(cp.cuda.Stream(non_blocking=True))
        if streams is not None:
            self.streams = list(streams)
        self.pools = [
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"hex-rank-{rank}",
                               initializer=self._initialize_worker, initargs=(rank,))
            for rank in range(len(self.devices))
        ]

    def _initialize_worker(self, rank: int) -> None:
        _bind_rank_thread(self.cp, self.devices[rank])
        if self.cp is not None:
            try:
                # A worker retains its rank device between calls. Entering
                # and leaving that same Device then preserves the assignment.
                self.cp.cuda.Device(self.devices[rank]).use()
            except BaseException as error:
                # Raising here would break the pool before _call could abort
                # peers or drain the rank stream with the original failure.
                self._initialization_errors[rank] = error

    def _remember_failure(self, error: BaseException) -> bool:
        with self._failure_lock:
            self.failed = True
            first = self._first_error is None
            if first:
                self._first_error = error
            return first

    def _abort_failure(self, error: BaseException, reason: str, label: str) -> None:
        try:
            self.transport.abort(reason)
        except BaseException as abort_error:
            BaseException.add_note(error, f"{label} transport abort also failed: {type(abort_error).__name__}: {abort_error}")

    @staticmethod
    def _drain_futures(futures) -> None:
        for future in futures:
            if future is not None:
                try:
                    future.result()
                except BaseException:
                    pass

    def _call(self, rank: int, operation: Callable[[int], Any]) -> Any:
        try:
            error = self._initialization_errors[rank]
            if error is not None:
                raise error
            device = nullcontext() if self.cp is None else self.cp.cuda.Device(self.devices[rank])
            stream = nullcontext() if self.cp is None else self.streams[rank]
            resources = (nullcontext() if self.physics_scope is None
                         else self.physics_scope.rank(rank, self.devices[rank]))
            with device, stream, resources:
                result = operation(rank)
                if self.cp is not None:
                    began = time.perf_counter()
                    try:
                        self.streams[rank].synchronize()
                    finally:
                        self.sync_seconds[rank] += time.perf_counter() - began
                        self.sync_calls[rank] += 1
                observer = getattr(operation, "after_synchronize", None)
                if observer is not None:
                    observer(rank)
                return result
        except BaseException as error:
            self._remember_failure(error)
            self._abort_failure(error, f"rank {rank}: {type(error).__name__}: {error}", f"rank {rank}")
            if self.cp is not None:
                try:
                    with self.cp.cuda.Device(self.devices[rank]):
                        self.streams[rank].synchronize()
                except BaseException as drain_error:
                    BaseException.add_note(error, f"rank {rank} CUDA stream drain also failed: {type(drain_error).__name__}: {drain_error}")
            observer = getattr(operation, "after_synchronize", None)
            if observer is not None:
                observer(rank)
            raise

    def call(self, operation: Callable[[int], Any]) -> list[Any]:
        if self.failed:
            raise ExecutorError("a failed rank group cannot resume")
        futures = [None] * len(self.devices)
        try:
            for rank, pool in enumerate(self.pools):
                futures[rank] = pool.submit(self._call, rank, operation)
        except BaseException as error:
            self._remember_failure(error)
            self._abort_failure(error, f"rank {rank} submission failed: {type(error).__name__}: {error}",
                                f"rank {rank} submission")
            self._drain_futures(futures)
            if self._first_error is error:
                raise
            raise self._first_error
        try:
            return [future.result() for future in futures]
        except BaseException as error:
            if self._remember_failure(error):
                self._abort_failure(error, f"rank result failed: {type(error).__name__}: {error}", "rank result")
            # Drain all ranks before a caller can free their device storage.
            self._drain_futures(futures)
            if self._first_error is error:
                raise
            raise self._first_error

    def step(self, operation: Callable[[int], Any], clock: Callable[[int], float], *, dt: float):
        before = [float(clock(rank)) for rank in range(len(self.devices))]
        if len(set(before)) != 1 or (self.time_seconds is not None and before[0] != self.time_seconds):
            self.failed = True
            self.transport.abort("rank clocks disagree before a step")
            raise ExecutorError(f"rank clocks disagree before a step: {before}")
        results = self.call(operation)
        after = [float(clock(rank)) for rank in range(len(self.devices))]
        expected = before[0] + float(dt)
        if any(value != expected for value in after):
            self.failed = True
            self.transport.abort("rank clocks disagree after a step")
            raise ExecutorError(f"rank clocks after a step {after}, expected {expected}")
        self.time_seconds = expected
        return results

    def close(self):
        error = None
        for rank, pool in enumerate(self.pools):
            try:
                pool.shutdown(wait=True)
            except BaseException as shutdown_error:
                self._remember_failure(shutdown_error)
                self._abort_failure(shutdown_error, f"rank {rank} worker shutdown failed: {type(shutdown_error).__name__}: {shutdown_error}",
                                    f"rank {rank} worker shutdown")
                if error is None:
                    error = shutdown_error
                else:
                    BaseException.add_note(error, f"rank {rank} worker shutdown also failed: {type(shutdown_error).__name__}: {shutdown_error}")
        if error is not None:
            raise error
        close = getattr(self.transport, "close", None)
        if close is not None:
            close()
            return
        if hasattr(self.transport, "synchronize"):
            self.transport.synchronize()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, error, traceback):
        try:
            self.close()
        except BaseException as close_error:
            if error is None:
                raise
            BaseException.add_note(error, f"same-host rank group close also failed: {type(close_error).__name__}: {close_error}")
        return False
