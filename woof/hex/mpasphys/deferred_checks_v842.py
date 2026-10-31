"""Device-side refusals read off the forecast's critical path.

Native MPAS stops at a fatal condition before the next slot runs. The CUDA
slots record the same conditions in device words. Reading each word as it is
written makes the single host thread wait for the GPU queue to drain several
times per step, so host launch work and device work never overlap.

THE BREAKAGE THIS PREVENTS: U-MPASPHYS-1. The MPAS-physics route spent most
of its step in an outer-physics stage that ran on one host thread while the
card idled between per-scheme synchronizations (GWDO, Noah, sea ice and the
outer wind reconstruction each read a flag every step).

Every check stays on. Inside a forecast step loop a slot hands its flag to
``defer``: the flag is reduced on the device at once (stream ordered, so it
captures the value at that point), and ``flush`` at the end of the step queues
one asynchronous copy of all of that step's words into pinned host memory. The
copy is inspected without blocking when it has landed, at the latest at the
next flush, and ``finish`` waits for every outstanding copy before the forecast
writes its last output. A refusal therefore still stops the forecast, at most
one step later than an immediate read, and names the condition and the step.

Outside a forecast loop (unit replays, referee helpers, other routes) no
collector is active and ``defer`` reads the flag immediately, as before.
"""
from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Callable

_ACTIVE: contextvars.ContextVar["DeferredChecks | None"] = contextvars.ContextVar(
    "hexcore_mpasphys_deferred_checks", default=None)


def _message(message: str | Callable[[int], str], value: int) -> str:
    return message(value) if callable(message) else message.format(value=value)


def defer(flag: Any, message: str | Callable[[int], str], *, reduce: str = "max",
          error: type[Exception] = RuntimeError) -> None:
    """Refuse with ``error`` when ``flag`` reduces to a nonzero word.

    ``flag`` is a device integer array; ``reduce`` is ``max`` (the largest word,
    e.g. a native source line) or ``first`` (the first nonzero word in order).
    ``message`` is a format string with ``{value}``, or a callable of it.
    ``error`` is the exception type the slot raised before; both paths keep it.
    """
    checks = _ACTIVE.get()
    if checks is None:
        value = int(_reduce(flag, reduce).item())
        if value:
            raise error(_message(message, value))
        return
    checks.defer(flag, message, reduce=reduce, error=error)


def _array_module(flag: Any) -> Any:
    """The flag's own array module: CuPy for a device word, NumPy for a host one.

    THE BREAKAGE THIS PREVENTS (hex 0.4.5 gate, public CI battery on a CPU runner with no CuPy): the
    resident-synchronization replays hand host flags to ``defer`` outside a forecast loop, and an
    unconditional ``import cupy`` refused them with "No module named 'cupy'". A device flag still
    reduces on the device, exactly as before.
    """
    if type(flag).__module__.partition(".")[0] == "cupy":
        import cupy
        return cupy
    import numpy
    return numpy


def _reduce(flag: Any, reduce: str) -> Any:
    cp = _array_module(flag)
    if reduce == "max":
        return cp.max(flag)
    if reduce == "first":
        nonzero = flag != 0
        return flag.ravel()[cp.argmax(nonzero.ravel())]
    raise ValueError(f"unknown deferred reduction {reduce!r}")


class ForecastAborted(RuntimeError):
    """The forecast was aborted (a rank process failed); its checks queue no more device work."""


class DeferredChecks:
    """One forecast's deferred device refusals."""

    def __init__(self, cp: Any):
        self.cp = cp
        self.step = 0
        self._pending: list[tuple[Any, Any, Any]] = []
        self._inflight: list[tuple[Any, Any, list[Any], int]] = []
        self.flushed = 0
        self.words = 0
        self.aborted: str | None = None

    def abort(self, reason: Any) -> None:
        """Stop queuing device work for these checks; may be called from another thread.

        THE BREAKAGE THIS PREVENTS (an RTX 5090 host, x1.2562 hrrrv5, 4 rank processes, 2026-10-29,
        MULTICARD/ckpt-shards/tip-s4-oom-hang.log): after rank 2 died, rank 0's next flush ran ``cupy.stack``, which
        loaded a kernel module; a module load synchronizes the context, and the context held a halo copy waiting on
        a flag no live rank would write, so rank 0 hung 4+ min.  After an abort every call raises
        :class:`ForecastAborted` (or returns, for reads) without touching the card, and a refusal read on halos the
        dead rank never sent cannot replace the abort as the named cause.
        """
        if self.aborted is None:
            self.aborted = str(reason)

    def _refuse_if_aborted(self) -> None:
        if self.aborted is not None:
            self._pending = []
            raise ForecastAborted(f"forecast aborted: {self.aborted}")

    def defer(self, flag: Any, message: Any, *, reduce: str = "max",
              error: type[Exception] = RuntimeError) -> None:
        self._refuse_if_aborted()
        if flag.size == 1 and reduce in ("max", "first"):
            # A single word needs a snapshot/cast, not a reduction plus a cast.
            # copy=True also isolates an int64 input from subsequent flag reuse.
            value = flag.reshape(()).astype(self.cp.int64, copy=True)
        else:
            value = _reduce(flag, reduce).astype(self.cp.int64, copy=False)
        # A slot may run on its own stream: the flush stream waits for this point.
        self._pending.append((value, (message, error), self.cp.cuda.get_current_stream().record()))

    def flush(self, step: int | None = None) -> None:
        """Queue this step's words to pinned memory; inspect copies that have landed."""
        self._refuse_if_aborted()
        if step is not None:
            self.step = step
        # A fast host can enqueue several steps before the GPU catches up. Polling
        # only ready events silently exceeded the promised one-step refusal bound.
        # Usually the preceding copy has landed; wait only when it is overdue.
        self.poll(block=False, through_step=self.step - 1)
        if not self._pending:
            return
        cp = self.cp
        stream = cp.cuda.get_current_stream()
        for _, _, ready in self._pending:
            stream.wait_event(ready)
        values = cp.stack([value for value, _, _ in self._pending])
        messages = [message for _, message, _ in self._pending]
        self._pending = []
        host = cp.cuda.alloc_pinned_memory(values.nbytes)
        import numpy as np
        out = np.frombuffer(host, dtype=np.int64, count=values.size)
        values.get(out=out, stream=stream, blocking=False)
        event = stream.record()
        # values must outlive its asynchronous copy; host keeps the pinned block alive.
        self._inflight.append((event, (values, host, out), messages, self.step))
        self.flushed += 1
        self.words += int(values.size)

    def poll(self, *, block: bool, through_step: int | None = None) -> None:
        if self.aborted is not None:
            return   # no wait on a card whose halos a failed rank never sent; in-flight buffers stay referenced
        keep = []
        for event, buffers, messages, step in self._inflight:
            if not event.done:
                if block or (through_step is not None and step <= through_step):
                    event.synchronize()
                else:
                    keep.append((event, buffers, messages, step))
                    continue
            out = buffers[2]
            for value, (message, error) in zip(out.tolist(), messages):
                if value:
                    # Do not release a later pinned buffer while DMA still owns it,
                    # even when an earlier refusal is unwinding the forecast.
                    for later, _, _, _ in self._inflight:
                        if not later.done:
                            later.synchronize()
                    self._inflight = []
                    refusal = error(f"{_message(message, int(value))} (deferred device check, step {step})")
                    refusal.deferred_check_step = step
                    raise refusal
        self._inflight = keep

    def output_validator(self):
        """Snapshot checks on the forecast thread; validate on the file writer.

        The forecast keeps its copies for the one-step refusal deadline. The
        reader owns the same immutable buffers until its independent validation
        finishes, so polling the forecast collector cannot release writer data.
        Checks raised by publication itself are included by the flush here.
        """
        self.flush()
        reader = DeferredChecks(self.cp)
        reader._inflight = list(self._inflight)
        device = self.cp.cuda.Device()

        def validate():
            with device:
                reader.poll(block=True)

        return validate

    def before_output(self) -> None:
        """Flush and read every word before a forecast writes or hands out state.

        THE BREAKAGE THIS PREVENTS: a word deferred at the step being written (e.g. the RRFS dycore's
        validation flag) could still be in flight when that step's history is captured, so a refused state
        would reach a file before the refusal. Only unfinished copies need a wait at this boundary.
        """
        # Publication itself can enqueue checks after the end-of-step flush
        # (resident wind reconstruction, for example). Include those words too.
        self.flush()
        self.poll(block=True)

    def prefer_earlier(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """ExitStack exit hook: when the forecast unwinds on an error, an earlier deferred refusal wins.

        THE BREAKAGE THIS PREVENTS: a word deferred at step N is read up to one step later, so a state refused
        at step N could first raise a different, later error (a slot's own immediate check on that state) that
        names the wrong cause. Every word still pending or in flight is read; a refusal among them is raised in
        place of the later error, chained to it. A failed read (e.g. the card is gone) leaves the error as is.
        """
        if not isinstance(exc, Exception) or getattr(exc, "deferred_check_step", None) is not None:
            return False
        if self.aborted is not None or (not self._pending and not self._inflight):
            return False   # an aborted forecast keeps the abort (the failed rank) as its named cause
        try:
            self.finish()
        except Exception as refusal:
            if getattr(refusal, "deferred_check_step", None) is None:
                return False
            raise refusal from exc
        return False

    def finish(self) -> None:
        """Read every outstanding word (the forecast's last boundary)."""
        self.flush()
        self.poll(block=True)

    def receipt(self) -> dict[str, Any]:
        return dict(flushes=self.flushed, words=self.words,
                    reader="pinned asynchronous copy per step; waits only for overdue checks or output/final boundaries",
                    parallelism="GPU reductions; host inspection off the step path")


def active() -> "DeferredChecks | None":
    """The forecast loop's collector, or None outside one (``defer`` then reads at once)."""
    return _ACTIVE.get()


@contextlib.contextmanager
def collecting(checks: DeferredChecks):
    token = _ACTIVE.set(checks)
    try:
        yield checks
    finally:
        _ACTIVE.reset(token)
