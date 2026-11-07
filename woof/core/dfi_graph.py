"""Compile a finite, advancing-clock model pass before GPU integration.

Each call to the Python model driver here happens under CUDA stream capture:
it constructs graph nodes but does not integrate the atmosphere. Every step
gets its own actual clock, physics schedule and boundary kernel arguments.
Rust submits the completed graph once. There is no warm-up model step and
no repeated graph with a frozen capture-time clock.
"""
from __future__ import annotations

from dataclasses import dataclass


class DfiBoundaryClock:
    """Setup-time signed WRF clock for captured boundary kernel arguments.

    The finite graph contains these distinct arguments at every solve.
    Its runtime never consults or mutates this host clock.
    """

    def __init__(self, interval, dt):
        import numpy as np
        self.interval = interval
        self.tick_den = interval.tick_den
        self.step_ticks = interval.step_ticks
        self.step_count = 0
        self.ticks = interval.start_ticks
        self.dt_fp32 = np.float32(dt)
        self.dtbc_fp32 = np.float32(interval.start_ticks/interval.tick_den)

    @property
    def elapsed_seconds(self):
        return self.ticks/self.interval.tick_den

    @property
    def dtbc_launch_fp32(self):
        return self.dtbc_fp32

    @property
    def elapsed_seconds_fp32(self):
        import numpy as np
        whole = abs(self.ticks)//self.tick_den
        whole = -whole if self.ticks < 0 else whole
        num = self.ticks-whole*self.tick_den
        return np.float32(whole)+np.float32(num)/np.float32(self.tick_den)

    def mark_force(self):
        import numpy as np
        self.dtbc_fp32 = np.float32(0)

    def prepare_step(self):
        import numpy as np
        self.dtbc_fp32 = np.float32(self.dtbc_fp32+self.dt_fp32)

    def advance(self):
        self.ticks += self.step_ticks
        self.step_count += 1

    def prepare(self, index):
        import numpy as np
        self.ticks = self.interval.start_ticks+index*self.interval.step_ticks
        self.prepare_step()


@dataclass(frozen=True)
class PassInterval:
    start_ticks: int
    stop_ticks: int
    step_ticks: int
    tick_den: int = 1

    def __post_init__(self):
        for value in (self.start_ticks, self.stop_ticks, self.step_ticks):
            if not isinstance(value, int) or not -(1 << 63) <= value < (1 << 63):
                raise ValueError('DFI clock must fit signed 64-bit native ticks')
        duration = self.stop_ticks-self.start_ticks
        if (not isinstance(self.tick_den, int) or self.tick_den <= 0 or
                self.step_ticks == 0 or duration*self.step_ticks <= 0 or
                duration % self.step_ticks or not -(1 << 63) <= duration < (1 << 63)):
            raise ValueError('DFI pass must contain whole steps in its signed direction')

    @property
    def steps(self):
        return (self.stop_ticks-self.start_ticks)//self.step_ticks

    def times(self):
        """Setup-only clock enumeration; no array or weather data is processed."""
        return tuple((self.start_ticks+i*self.step_ticks)/self.tick_den
                     for i in range(self.steps))


class CompiledModelPass:
    """Single-use graph with an owned allocator, ledger and model lifetime."""

    def __init__(self, state, graph, stream, pool, ledger, interval, checks):
        self.state, self.graph, self.stream = state, graph, stream
        self.pool, self.ledger, self.interval = pool, ledger, interval
        self.submitted = False
        self.completed = False
        self.checks = checks

    def submit(self):
        if self.submitted:
            raise RuntimeError('a compiled DFI pass may be submitted only once')
        from woof.core.dfi_runtime import submit_compiled_pass
        # Mark before submission: a driver failure may leave partial writes.
        self.submitted = True
        submit_compiled_pass(self.graph, self.stream,
            start_ticks=self.interval.start_ticks, stop_ticks=self.interval.stop_ticks,
            step_ticks=self.interval.step_ticks, steps=self.interval.steps)

    def finish(self):
        """Sole pass-completion synchronization and health readback."""
        if not self.submitted:
            raise RuntimeError('the DFI pass has not been submitted')
        if not self.completed:
            self.stream.synchronize()
            self.ledger.drain()
            from woof.core.deferred_device_checks import drain_collected
            drain_collected(self.checks)
            self.completed = True


def capture_model_pass(state, cfg, interval, *, stream, pool=None,
                       before_step=None, after_step=None, at_end=None):
    """Build the graph at setup; callbacks also execute only during capture.

    Immutable device tables and clock-specific radiation/forcing inputs must
    be prepared before entry. Captured temporaries stay in an owned private
    pool; returning them to an unrelated allocator before launch would leave
    dangling graph pointers. The original state allocations remain owned by
    the supplied state. A failed capture is terminal for this setup.
    """
    import cupy as cp
    from woof.core.dycore import step
    from woof.core.health_ledger import HealthLedger, deferring
    from woof.core.dfi_runtime import load

    if not isinstance(interval, PassInterval):
        raise TypeError('DFI capture needs a validated PassInterval')
    if float(cfg.dt) != interval.step_ticks/interval.tick_den:
        raise ValueError('DFI graph dt differs from its signed tick interval')
    if float(state.elapsed_seconds) != interval.start_ticks/interval.tick_den:
        raise ValueError('DFI graph state is not at its interval start')
    load()  # Fail before capture if the compiled runtime is unavailable.
    if pool is None:
        pool = cp.cuda.MemoryPool()
    ledger = HealthLedger(label='compiled DFI model pass')
    from woof.core.deferred_device_checks import collecting
    checks = []
    stream.synchronize()  # Setup boundary, before any integration.
    with cp.cuda.using_allocator(pool.malloc), stream, deferring(ledger), collecting(checks):
        stream.begin_capture()
        try:
            for i, elapsed in enumerate(interval.times()):
                state.elapsed_seconds = elapsed
                if before_step is not None:
                    before_step(i)
                step(state, cfg)
                if after_step is not None:
                    after_step(i)
            state.elapsed_seconds = interval.stop_ticks/interval.tick_den
            if at_end is not None:
                at_end()
            graph = stream.end_capture()
        except BaseException:
            try:
                stream.end_capture()
            except BaseException:
                pass
            raise
    return CompiledModelPass(state, graph, stream, pool, ledger, interval, checks)


def capture_tree_pass(model, configurations, bindings, coefficients, interval,
                      *, stream, reset, backward, history=None, timing=None):
    """Capture WRF's recursive solve/FORCE walk, including repeated endpoints.

    All siblings are forced from the same advanced parent before recursion.
    Feedback is disabled during DFI by native dfi_*_init. Each recursive
    integration call samples its endpoint, even when the next call samples
    that same coefficient index again (module_integrate.F).
    """
    import cupy as cp
    from woof.core.dycore import step
    from woof.core.health_ledger import HealthLedger, deferring
    from woof.core.dfi_runtime import load
    load()
    pool = cp.cuda.MemoryPool()
    ledger = HealthLedger(label='compiled nested DFI pass')
    from woof.core.deferred_device_checks import collecting
    checks = []

    def integrate(node, stop):
        clock = node.clock
        gid = node.cfg.grid_id
        binding, cfg = bindings[gid], configurations[gid]
        while clock.ticks != stop:
            if history is not None:
                history(node)
            if timing is not None:
                timing.step(node,0)
            binding.accumulate(coefficients[gid])
            if timing is not None:
                timing.step(node,1)
            state = node.state
            state.elapsed_seconds = float(clock.elapsed_seconds_fp32)
            if node.parent is None:
                _prepare_external_clock(state, clock)
            clock.prepare_step()
            step(state,cfg)
            if timing is not None:
                timing.step(node,2)
            binding.native.advance()
            clock.advance()
            for child in node.children:
                child.coupler.force(child)
            for child in node.children:
                integrate(child,clock.ticks)
        if history is not None:
            history(node)
        if timing is not None:
            timing.endpoint(node,0)
        binding.accumulate(coefficients[gid])
        if timing is not None:
            timing.endpoint(node,1)

    stream.synchronize()
    with cp.cuda.using_allocator(pool.malloc), stream, deferring(ledger), collecting(checks):
        stream.begin_capture()
        try:
            integrate(model.root,interval.stop_ticks)
            for node in model.walk_parent_first():
                if timing is not None:
                    timing.reset(node,0)
                if reset:
                    binding = bindings[node.cfg.grid_id]
                    binding.native.reset(backward=backward)
                    binding.import_filtered()
                if timing is not None:
                    timing.reset(node,1)
            graph = stream.end_capture()
        except BaseException:
            try:
                stream.end_capture()
            except BaseException:
                pass
            raise
    return CompiledModelPass(model,graph,stream,pool,ledger,interval,checks)


def _prepare_external_clock(state, clock):
    """Reset root dtbc when its served forcing record changes at a seam."""
    boundaries = getattr(state,'lateral_boundaries',None)
    if boundaries is None or not hasattr(boundaries,'intervals'):
        return
    elapsed = clock.elapsed_seconds
    first = boundaries.intervals[0]
    record = first if elapsed < first.start_seconds else boundaries.interval_at(elapsed)
    previous = getattr(clock,'_external_record',None)
    if previous is not record:
        import numpy as np
        clock.dtbc_fp32 = np.float32(elapsed-record.start_seconds)
        clock._external_record = record
