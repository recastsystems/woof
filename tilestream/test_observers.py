"""THE SAFETY OBSERVERS UNDER STREAMING: they watch a corpse.

``woof.runtime.integrate_prepared_case`` guards a forecast with three
whole-domain observers, and all three are handed ``state``::

    stepper(state, integration_cfg, **step_kwargs)          # runtime.py
    ...
    health.require_healthy(phase=...)                       # every 4th substep
    report = stability_report(state, integration_cfg, boundary_width=width)
    ...
    if not nan_free:
        raise RuntimeError("real-case integration produced a non-finite ...")
    ...
    step_swdown_peak = float(cp.max(state.physics.fields["swdown"]))

``woof.core.model.execute_experiment`` does the same with
``StateHealthValidator(node.state)``.

Under ``[tiles] store = "host"`` -- the out-of-core mode, the whole point
of the feature -- ``state`` is NOT where the domain lives.
:func:`woof.core.streaming.attach` builds the store with::

    store = {name: _gather.pinned_copy(arr) for name, arr in live.items()}

``pinned_copy`` COPIES.  From that instant the domain is the store, the sweep
gathers tiles out of it and scatters interiors back into it, and the
``DomainState`` the observers hold is a snapshot of t=0 that nothing ever
writes again.  (With ``store = "device"`` the seam does ``store = live`` --
the same arrays -- so the observers are correct there, and the difference
between the two is the whole finding.)

WHY THIS IS THE WORST POSSIBLE SHAPE FOR A DEFECT
-------------------------------------------------
The observers are pure -- they carry nothing into the answer -- so a wrong
reading cannot change a single bit of the forecast, which is exactly why
nothing caught it.  What it changes is whether the run is GUARDED.  The
snapshot is healthy at t=0, so:

* ``nan_free`` stays ``True`` forever and the ``RuntimeError`` never fires;
* ``w_max``/``boundary_w_max``/``interior_w_max`` freeze at the t=0 values
  and are written into every restart checkpoint's ``run_trackers`` as
  though they were the run's;
* ``swdown_peak`` freezes at the t=0 peak;
* ``StateHealthValidator.require_healthy`` passes on a state that has not
  moved.

A domain that went non-finite in the store therefore completes
"successfully", writes its history frames and writes a checkpoint that
records ``nan_free: true``.  A silently disarmed NaN guard is
indistinguishable from a well-behaved forecast.  And it is not free: the
two-stage device reduction over u/w/thp runs EVERY substep on data nobody
is using, so the mode pays a real per-substep tax for an answer that is
wrong.

WHAT THIS MODULE MEASURES
-------------------------
Nine legs at dx 3 km, full physics + Kain-Fritsch (mp10, YSU, Noah, RRTMGP,
KF), a real Lambert projection, real terrain and SPECIFIED lateral
boundaries, in three sets of three.  Every set runs resident, streamed as
shipped, and streamed with the fold, from the same seed and the same
boundary tables, so the only difference inside a set is which memory the
observers read.

``blowup-*``   dt 18 s, 60 steps.  A GENUINE blow-up, not an injection:
               :func:`stability_ladder` measured this harness's initial
               condition going non-finite at dt 18 s on a 3 km grid, and the
               resident run raises the run loop's own ``RuntimeError`` when
               it does.  This is the narrative the defect is about -- a
               domain that blew up in the store -- with nothing put there by
               hand, so no argument about whether the poison was fair can
               reach it.
``*-clean``    dt 12 s, 200 steps, which the same ladder measured stable.
               The reading: the streamed run's reported ``w_max`` must equal
               the resident run's to the last bit, substep by substep.  The
               store's TRUE per-step ``w_max`` is folded on the host as a
               third trace, so the record shows both what the observer SAID
               and what was actually happening.
``*-poisoned`` dt 12 s, 200 steps, one cell of the STORE set to NaN at
               substep 50.  Kept even though the blow-up set is stronger,
               because it separates "the domain went bad" from "one cell of
               the store went bad" -- and because its resident leg exposed
               something worth knowing (see :func:`classify_guard`).

200 steps at dt 12 s is 40 forecast minutes, so radiation (radt 12 min =
every 60 steps) fires 4 times and cumulus (cudt 5 min = every 25 steps) 9
times INSIDE every window, on BOTH sides.  Those counts are PRINTED for
every leg, because three of this project's six false results were the same
mistake: a number measured in a window where radiation and cumulus never
fired.  The 60-step blow-up legs are shorter than a radiation period on
purpose -- they are a correctness gate and nothing timed is quoted from them.

Every leg also reports the per-substep wall cost of ``stability_report`` and
of ``StateHealthValidator.require_healthy``, so the fold can be priced
against what it replaces.

    python -m tilestream.test_observers --suite --nx 336 --tile 84
    python -m tilestream.test_observers --verdict obs/*.json
    python -m tilestream.test_observers --ladder 18 12 9 6 --nx 336

WHAT IS FIXED HERE AND WHAT IS NOT
----------------------------------
The stability record IS folded (``woof.core.streaming.StreamedStability``,
``woof/core/kernels/health_tile.cu``), and so are ``swdown``'s peak and the
physics call counters, which were reading the same corpse.
``StateHealthValidator`` is NOT: it is a descriptor kernel over up to 1024
whole fields with no windowing, so it cannot be aimed at a tile's interior,
and the foldable form -- validating each buffer after its gather and before
its step -- observes the domain one step behind and multiplies the launch by
the tile count.  Under a host store that validator still inspects the t=0
snapshot.  It is said here and in the run loop rather than left to be
discovered.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time

import numpy as np

from woof.core import streaming
from tilestream import driver, gather, harness
from tilestream import physics_inventory as physinv
from tilestream.test_join import build_domain, join_cfg, tile_factory

# --------------------------------------------------------------------------
# the configuration
# --------------------------------------------------------------------------

#: 672 divides by 168 exactly (4x4 = 16 tiles).  A tile that does not DIVIDE
#: the domain leaves a ragged trailing tile, which is read right through in
#: ring mode -- 22.7% of the store instead of 2.4% in one measured case.
NX = NY = 672
NZ = 49
TX = TY = 168
NSTEPS = 200
#: 3 km with WRF's ~6 s/km rule of thumb.  200 x 18 s = one forecast hour,
#: which is the window the 2.01x streaming tax was measured over.
DX = 3000.0
DT = 18.0
RUNG = "full(real74)+KF"
#: The substep the poison is injected at.  Deep enough that both cadences
#: have already fired several times, far enough from the end that a run
#: which fails to notice has 150 more substeps to look guarded in.
POISON_STEP = 50
#: One cell, in the interior of tile (1, 1) at the full size -- nowhere near
#: a tile seam, a ring band or a domain edge, so nothing but the observer
#: can be blamed for missing it.
POISON_INDEX = (20, 300, 300)

QUICK = dict(nx=336, ny=336, tile=84, nsteps=60, poison_step=20,
             index=(20, 150, 150))


def config(nx: int, ny: int, dt: float = DT) -> object:
    return join_cfg(nx, ny, NZ, rung=RUNG, dx=DX, dy=DX, dt=float(dt))


def stability_ladder(nx: int, ny: int, dts, nsteps: int) -> dict:
    """First non-finite substep of the RESIDENT reference, per ``dt``.

    ``test_join.RUNG_DT`` carries the same measurement at dx = 12 km: dry is
    clean from 3 s to 60 s but the moist rungs go non-finite at 60 s and are
    clean at 30 s, and that is a property of the harness's random INITIAL
    CONDITION, not of streaming.  A 3 km grid resolves the same
    perturbations on a quarter of the spacing, so the ladder has to be
    re-measured here rather than scaled.  A reference that is not finite has
    nothing to compare a streamed run against, and a gate that compared one
    run's NaNs with another's would agree beautifully.
    """
    import cupy as cp

    out = {}
    for dt in dts:
        cfg = config(nx, ny, dt)
        bnd = boundaries_one_state_at_a_time(cfg)
        cp.get_default_memory_pool().free_all_blocks()
        leg = resident_leg(cfg, nsteps, boundaries=bnd)
        out[float(dt)] = {
            "steps_completed": leg["steps_completed"],
            "raised": leg.get("raised"),
            "w_max_ms": leg["w_max_ms"],
            "boundary_w_max_ms": leg["boundary_w_max_ms"],
            "interior_w_max_ms": leg["interior_w_max_ms"],
            "final_w_max": (leg["true_trace"][-1] if leg["true_trace"]
                            else None),
            "call_counts": leg["call_counts"],
        }
        print(f"  dt={dt:5.1f} s  {out[float(dt)]}", flush=True)
        del bnd, leg
        cp.get_default_memory_pool().free_all_blocks()
    return out


# --------------------------------------------------------------------------
# the run loop's own observers, transcribed
# --------------------------------------------------------------------------

#: The shape of ``integrate_prepared_case`` this module reproduces, and the
#: reason a lookalike is admissible at all: these lines are checked against
#: the real loop rather than described.  A lookalike of a loop that has moved
#: measures nothing.
#:
#: ``AS_SHIPPED`` is the defect -- every observer's argument is ``state`` --
#: and it is what :class:`SafetyObservers` executes with ``read=None``,
#: whatever the tree in front of it says.  So the broken leg stays a faithful
#: reproduction of the pre-fix loop even after the tree is fixed, which is
#: what lets both legs run out of one checkout.
AS_SHIPPED = (
    'report = stability_report(',
    'state, integration_cfg, boundary_width=width)',
    'health.require_healthy(phase=phase + ".post-step")',
    'step_swdown_peak = float(cp.max(state.physics.fields["swdown"]))',
    'surface_forcing_updates = state.physics.call_counts["radiation"]',
    '"real-case integration produced a non-finite state at "',
)

#: ``AS_FIXED`` is what the loop says once the observers are asked of the
#: STEPPER.  The stability call keeps its exact text -- the name
#: ``stability_report`` is rebound to what ``streaming.stability_observer``
#: returned, which for a resident domain is ``dycore.stability_report``
#: ITSELF -- so a resident run executes the identical call it always did and
#: the pinned-source assertions in tests/test_perf_dycore_cpu.py still hold.
AS_FIXED = (
    'stability_report = stability_observer(stepper)',
    'report = stability_report(',
    'state, integration_cfg, boundary_width=width)',
    'step_swdown_peak = domain_field_max(',
    'surface_forcing_updates = domain_call_counts(',
    '"real-case integration produced a non-finite state at "',
)


def check_runtime_still_looks_like_this() -> dict:
    """Which shape ``integrate_prepared_case`` is in, checked not claimed."""
    from woof import runtime

    src = inspect.getsource(runtime.integrate_prepared_case)
    shipped = [line for line in AS_SHIPPED if line not in src]
    fixed = [line for line in AS_FIXED if line not in src]
    return {
        "as_shipped_missing": shipped,
        "as_fixed_missing": fixed,
        "shape": ("as-shipped" if not shipped else
                  "as-fixed" if not fixed else "UNRECOGNISED"),
        # True in BOTH shapes, and the reason the health validator is still
        # listed as unfixed in the verdict.
        "health_still_binds_state":
            "health = StateHealthValidator(state)" in src,
        "loop_calls_stepper":
            "stepper(state, integration_cfg, **step_kwargs)" in src,
    }


#: The RuntimeError text ``integrate_prepared_case`` raises when its NaN
#: guard fires.  Matched rather than assumed, because the point of every leg
#: below is WHICH guard stopped the run -- and a forecast has more than one.
NAN_GATE_MESSAGE = "produced a non-finite state at"


def classify_guard(exc) -> dict:
    """Which safety net caught the run, out of the several a forecast has.

    This mattered more than expected.  A poisoned ``state.w`` on the RESIDENT
    leg does not reach the run loop's NaN gate at all: it reaches
    ``PhysicsDriver.accept_microphysics`` first, INSIDE ``dycore.step``, and
    dies there with ``FloatingPointError: microphysics RAINNC contains a
    non-finite value``.  That guard is scheme-local -- it lives in the
    microphysics acceptance path, it exists in mp10 and not in a dry run --
    and, crucially, under streaming it runs on the TILE BUFFER, which is real
    current data.  So the streamed mode is not totally unguarded, and saying
    it was would have been a second false result.

    What the streamed mode loses is the run's OWN gate: the whole-domain
    ``nan``/``w_max``/CFL record that ``integrate_prepared_case`` raises on,
    writes into every checkpoint's ``run_trackers`` and reports as the run
    summary.  Distinguishing the two is the whole job of this function.
    """
    name = type(exc).__name__
    text = str(exc)
    gate = (isinstance(exc, RuntimeError) and NAN_GATE_MESSAGE in text)
    where = ""
    tb = exc.__traceback__
    while tb is not None:
        where = f"{tb.tb_frame.f_code.co_filename}:{tb.tb_lineno}"
        tb = tb.tb_next
    return {"guard": ("run-loop NaN gate" if gate else f"in-step {name}"),
            "is_run_loop_gate": bool(gate),
            "type": name, "message": text[:200], "raised_in": where}


class SafetyObservers:
    """``integrate_prepared_case``'s per-substep safety gates, in one object.

    ``read`` is where the reduction gets its numbers.  ``None`` -- the
    default and what the run loop does today -- means ``stability_report``
    on the ``DomainState``.  A streamed run may pass a callable that folds
    the same reduction over the domain the sweep actually wrote; that
    substitution IS the proposed fix, and having both here means the two are
    driven by identical code above the read.
    """

    def __init__(self, state, cfg, *, width: int, read=None,
                 health_cadence: int = 4):
        from woof.core.health import StateHealthValidator

        self.state = state
        self.cfg = cfg
        self.width = int(width)
        self.read = read
        self.health = StateHealthValidator(state)
        self.health_cadence = int(health_cadence)
        self.nan_free = True
        self.w_max = 0.0
        self.w_max_boundary_row = None
        self.boundary_w_max = 0.0
        self.interior_w_max = 0.0
        self.swdown_peak = -np.inf
        self.trace: list[dict] = []
        self.report_seconds = 0.0
        self.report_calls = 0
        self.health_seconds = 0.0
        self.health_calls = 0

    def _stability(self) -> dict:
        from woof.core.dycore import stability_report

        if self.read is not None:
            return self.read()
        return stability_report(self.state, self.cfg,
                                boundary_width=self.width)

    def after_substep(self, step_index: int) -> None:
        """Everything runtime.py does between one substep and the next."""
        import cupy as cp

        if step_index % self.health_cadence == 0:
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            self.health.require_healthy(phase=f"substep-{step_index + 1}")
            cp.cuda.runtime.deviceSynchronize()
            self.health_seconds += time.perf_counter() - t0
            self.health_calls += 1

        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        report = self._stability()
        cp.cuda.runtime.deviceSynchronize()
        self.report_seconds += time.perf_counter() - t0
        self.report_calls += 1

        self.nan_free = self.nan_free and not report["nan"]
        step_w_max = float(report["w_max"])
        if step_w_max > self.w_max:
            max_index = np.unravel_index(report["w_argmax"],
                                         self.state.w.shape)
            _k, j, i = (int(index) for index in max_index)
            distance = min(j, self.cfg.ny - 1 - j, i, self.cfg.nx - 1 - i)
            self.w_max_boundary_row = (distance if distance < self.width
                                       else None)
            self.w_max = step_w_max
        self.boundary_w_max = max(self.boundary_w_max,
                                  report["boundary_w_max"])
        self.interior_w_max = max(self.interior_w_max,
                                  report["interior_w_max"])
        self.trace.append({"step": step_index + 1,
                           "w_max": step_w_max,
                           "u_max": float(report["u_max"]),
                           "cfl": (None if report["cfl"] is None
                                   else float(report["cfl"])),
                           "nan": bool(report["nan"])})
        if not self.nan_free:
            raise RuntimeError(
                "real-case integration produced a non-finite state at "
                f"dynamics substep {step_index + 1}")

    def outer(self) -> None:
        """The per-outer-step observer: swdown's peak."""
        import cupy as cp

        peak = float(cp.max(self.state.physics.fields["swdown"]))
        self.swdown_peak = max(self.swdown_peak, peak)

    def summary(self) -> dict:
        return {
            "nan_free": self.nan_free,
            "w_max_ms": float(self.w_max),
            "w_max_boundary_row": self.w_max_boundary_row,
            "boundary_w_max_ms": float(self.boundary_w_max),
            "interior_w_max_ms": float(self.interior_w_max),
            "swdown_peak_wm2": float(self.swdown_peak),
            "stability_report_calls": self.report_calls,
            "stability_report_ms_per_call": (
                1000.0 * self.report_seconds / max(1, self.report_calls)),
            "health_calls": self.health_calls,
            "health_ms_per_call": (
                1000.0 * self.health_seconds / max(1, self.health_calls)),
        }


# --------------------------------------------------------------------------
# the truth: what the domain is actually doing
# --------------------------------------------------------------------------

def boundaries_one_state_at_a_time(cfg, *, seeds=(20_260_731, 20_260_732),
                                   seconds: float = 21600.0):
    """The domain's specified forcing without two domains ever coexisting.

    ``test_join.domain_boundaries`` holds both source states at once, which
    at 672x672x49 is two un-stepped full-physics domains -- MEASURED 15.8 GB
    of device allocation before the second one finishes building, and an
    OutOfMemoryError on a 24 GB card that has room for the forecast itself.
    :class:`woof.ingest.lateral_bc.StateBoundaryFrames` is WOOF-ArWen's own
    accumulator for exactly this: it keeps only the four ``spec_bdy_width``
    perimeter frames and lets the caller drop the state at once, and its
    docstring pins that the result is element-for-element what the
    all-at-once builder returns.  Two genuinely different seeds are still
    used, because a repeated snapshot gives a ZERO time tendency and would
    quietly disarm ``dtbc`` -- and with it the clock the whole sweep depends
    on.
    """
    from woof.ingest.lateral_bc import StateBoundaryFrames

    frames = StateBoundaryFrames(spec_bdy_width=int(cfg.spec_bdy_width),
                                 spec_zone=int(cfg.spec_zone),
                                 relax_zone=int(cfg.relax_zone))
    for seed in seeds:
        state, _geo = build_domain(cfg, seed=seed, warmup=0)
        frames.add_state(state)
        del state
        # NOT free_all_blocks: the blocks go back to cupy's POOL and stay
        # reserved from the driver.  On a card a dozen other agents are
        # fighting over, a process that returns its arena between phases
        # loses it and dies at its next allocation -- measured, three times.
    return frames.build([0.0, float(seconds)])


def store_w_max(store) -> float:
    """``max |w|`` over the WHOLE domain, read where the domain lives.

    NumPy over the pinned host array.  ``max`` is exact and order-free in
    floating point, so this is the same number the device reduction would
    return for the same bytes -- which is the point: it is the reading the
    run loop should have had.
    """
    w = np.asarray(store["state/w"])
    return float(np.abs(w).max())


def store_nonfinite(store) -> int:
    total = 0
    for value in store.values():
        arr = np.asarray(value)
        if arr.dtype.kind == "f":
            total += int(np.count_nonzero(~np.isfinite(arr)))
    return total


def state_w_max(state) -> float:
    import cupy as cp

    return float(cp.abs(state.w).max())


def digest(array) -> str:
    host = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(host.tobytes(order="C")).hexdigest()[:16]


# --------------------------------------------------------------------------
# the two legs
# --------------------------------------------------------------------------

def _cadence_counts(state_or_scalars) -> dict:
    counts = getattr(getattr(state_or_scalars, "physics", None),
                     "call_counts", None)
    if counts is None and isinstance(state_or_scalars, dict):
        counts = state_or_scalars.get("call_counts")
    return dict(counts or {})


def resident_leg(cfg, nsteps: int, *, poison=None, poison_step=None,
                 boundaries=None) -> dict:
    """The control: one resident domain, WOOF-ArWen's ordinary ``dycore.step``."""
    import cupy as cp
    from woof.core.dycore import step as dycore_step

    state, _geo = build_domain(cfg, boundaries=boundaries, warmup=1)
    stepper = streaming.make_stepper(state, cfg, streaming.OFF)
    assert stepper is dycore_step, "the control must be the dycore's own step"

    obs = SafetyObservers(state, cfg, width=int(cfg.spec_bdy_width))
    out: dict = {"leg": "resident", "poisoned": poison is not None,
                 "w_digest_at_t0": digest(cp.asnumpy(state.w))}
    cp.cuda.runtime.deviceSynchronize()
    wall0 = time.perf_counter()
    step_seconds = 0.0
    true_trace = []
    try:
        for istep in range(nsteps):
            if poison is not None and istep == poison_step:
                state.w[poison] = float("nan")
                out["poison_applied_at_step"] = istep + 1
                out["poison_target"] = "state.w (the resident domain)"
            t0 = time.perf_counter()
            stepper(state, cfg, refl_10cm_due=False)
            cp.cuda.runtime.deviceSynchronize()
            step_seconds += time.perf_counter() - t0
            true_trace.append(state_w_max(state))
            obs.after_substep(istep)
            obs.outer()
    except Exception as exc:                             # noqa: BLE001
        out["raised"] = str(exc)
        out["raised_at_step"] = len(obs.trace)
        out.update(classify_guard(exc))
    cp.cuda.runtime.deviceSynchronize()
    out["wall_seconds"] = time.perf_counter() - wall0
    out["step_seconds"] = step_seconds
    out["steps_completed"] = len(obs.trace)
    out["step_ms_per_step"] = 1000.0 * step_seconds / max(1, len(obs.trace))
    out.update(obs.summary())
    out["call_counts"] = _cadence_counts(state)
    out["observed_trace"] = [t["w_max"] for t in obs.trace]
    out["true_trace"] = true_trace
    out["final_state_w_digest"] = digest(cp.asnumpy(state.w))
    return out


def _builder(cfg, domain, boundaries):
    """The route-owned construction ``make_stepper`` needs, as test_join's."""

    def build(state, run_cfg, decision):
        geo_inv = {k: gather.pinned_copy(v) for k, v in
                   driver.geography_inventory(domain).items()}
        per_tile = streaming.tile_boundary_tables(
            boundaries, streaming.tile_specs(run_cfg, decision), seam="zeros")
        factory = tile_factory(run_cfg, per_tile[0])
        return streaming.attach(
            state, run_cfg, decision, tile_state_factory=factory,
            geography=geo_inv, boundary_tables=per_tile,
            scalars=physinv.carrier_scalars(domain), check_geography=False)

    return build


def streamed_leg(cfg, nsteps: int, tile_nx: int, tile_ny: int, *,
                 poison=None, poison_step=None, boundaries=None,
                 fixed: bool = False, trace_truth: bool = True) -> dict:
    """The subject: the domain in pinned host RAM, one tile at a time.

    ``fixed=False`` is what WOOF-ArWen ships: the observers read ``state``.
    ``fixed=True`` routes the same reduction through the streamed stepper,
    which is the proposed fix and the negative control for it -- the poison
    below MUST be missed with ``fixed=False`` and MUST be caught with
    ``fixed=True``, from otherwise identical code.
    """
    import cupy as cp

    domain, _geo = build_domain(cfg, boundaries=boundaries, warmup=1)
    options = streaming.StreamingOptions(
        mode="on", tile_nx=int(tile_nx), tile_ny=int(tile_ny), nbuffers=2,
        store="host")
    decision = streaming.decide(cfg, options)
    stepper = streaming.make_stepper(domain, cfg, options, decision=decision,
                                     build=_builder(cfg, domain, boundaries))
    assert streaming.is_streaming(stepper)
    store = stepper.store

    out: dict = {"leg": "streamed", "poisoned": poison is not None,
                 "fixed": bool(fixed),
                 "decision": stepper.decision.explain()}
    # THE FINDING, stated as an identity check rather than as prose: the
    # array the observers reduce over is not the array the sweep writes.
    # ``store = {name: pinned_copy(arr)}`` in streaming.attach, so the store
    # entry is a pinned HOST array and state.w is a device one -- not two
    # views of one buffer, two buffers.  The digests below are equal at t=0
    # and the state's is unchanged at the end, which is the proof that
    # matters: nothing ever writes it again.
    out["store_w_is_host_array"] = isinstance(store["state/w"], np.ndarray)
    out["w_digest_at_t0"] = digest(cp.asnumpy(domain.w))
    out["store_w_digest_at_t0"] = digest(store["state/w"])
    out["store_keys"] = len(store)

    read = None
    if fixed:
        from woof.core.dycore import stability_report as dycore_report

        folded = streaming.stability_observer(stepper)
        assert folded is not dycore_report, (
            "stability_observer handed back the dycore's own reduction for a "
            "STREAMED domain; the fold is not wired")

        def read():
            return folded(domain, cfg,
                          boundary_width=int(cfg.spec_bdy_width))
    else:
        # THE NEGATIVE CONTROL for the fix, and it has to be switched off at
        # the source rather than merely unused: with the observer still
        # installed the sweep would pay for a fold nothing reads, and the
        # cost comparison below would be measuring the fixed transport while
        # reporting the broken one's numbers.
        stepper.stability = None
        stepper.tiled_run.observer = None
    obs = SafetyObservers(domain, cfg, width=int(cfg.spec_bdy_width),
                          read=read)

    cp.cuda.runtime.deviceSynchronize()
    wall0 = time.perf_counter()
    step_seconds = 0.0
    true_trace = []
    try:
        for istep in range(nsteps):
            if poison is not None and istep == poison_step:
                store["state/w"][poison] = float("nan")
                out["poison_applied_at_step"] = istep + 1
                out["poison_target"] = "store['state/w'] (where the domain is)"
            t0 = time.perf_counter()
            stepper(domain, cfg, refl_10cm_due=False)
            cp.cuda.runtime.deviceSynchronize()
            step_seconds += time.perf_counter() - t0
            if trace_truth:
                true_trace.append(store_w_max(store))
            obs.after_substep(istep)
            obs.outer()
    except Exception as exc:                             # noqa: BLE001
        out["raised"] = str(exc)
        out["raised_at_step"] = len(obs.trace)
        out.update(classify_guard(exc))
    cp.cuda.runtime.deviceSynchronize()
    out["wall_seconds"] = time.perf_counter() - wall0
    out["step_seconds"] = step_seconds
    out["steps_completed"] = len(obs.trace)
    out["step_ms_per_step"] = 1000.0 * step_seconds / max(1, len(obs.trace))
    out.update(obs.summary())
    out["call_counts"] = _cadence_counts(stepper.scalars)
    out["observed_trace"] = [t["w_max"] for t in obs.trace]
    out["true_trace"] = true_trace
    out["store_nonfinite_at_end"] = store_nonfinite(store)
    out["final_store_w_digest"] = digest(store["state/w"])
    out["final_state_w_digest"] = digest(cp.asnumpy(domain.w))
    out["state_w_max_at_end"] = state_w_max(domain)
    return out


# --------------------------------------------------------------------------
# the whole suite, in ONE process
# --------------------------------------------------------------------------

#: Every leg, and what each one is for.  ``(name, dt, steps, poison, fixed,
#: leg)``.
#:
#: Set A is a GENUINE blow-up, not an injection: the ladder measured this
#: harness's initial condition going non-finite at dt = 18 s on a 3 km grid,
#: and the resident run raises the run loop's own
#: ``RuntimeError("...produced a non-finite state at...")`` when it does.
#: That is exactly the narrative the defect is about -- a domain that blew up
#: in the store -- with nothing injected by hand, so no argument about
#: whether the poison was fair can reach it.
#:
#: Set B is the same configuration at a dt the ladder measured stable, for
#: the READING: the streamed run's reported w_max must equal the resident
#: run's to the last bit.
#:
#: Set C is the surgical poison of the brief.  It is kept even though set A
#: is the stronger evidence, because it separates "the domain went bad" from
#: "one cell of the STORE went bad", and because its resident leg exposed
#: something worth knowing: a poisoned ``state.w`` dies in
#: ``accept_microphysics`` inside ``dycore.step``, not at the run loop's gate.
SUITE = (
    ("blowup-resident",         18.0,  60, False, False, "resident"),
    ("blowup-streamed",         18.0,  60, False, False, "streamed"),
    ("blowup-streamed-fixed",   18.0,  60, False, True,  "streamed"),
    ("resident-clean",          12.0, 200, False, False, "resident"),
    ("streamed-clean",          12.0, 200, False, False, "streamed"),
    ("streamed-clean-fixed",    12.0, 200, False, True,  "streamed"),
    ("resident-poisoned",       12.0, 200, True,  False, "resident"),
    ("streamed-poisoned",       12.0, 200, True,  False, "streamed"),
    ("streamed-poisoned-fixed", 12.0, 200, True,  True,  "streamed"),
)


def wait_for_vram(need_bytes: int, *, timeout_s: float = 5400.0,
                  poll_s: float = 20.0) -> None:
    """Block until the card has ``need_bytes`` free.

    These boxes are shared with a dozen other agents and a leg that starts
    30 seconds too early dies at its first allocation, which looks exactly
    like a code failure in the log.  ``cupy.cuda.runtime.memGetInfo`` is the
    only reading trusted here; ``nvidia-smi``'s used/free is unreliable under
    WSL2 and merely second-hand everywhere else.
    """
    import cupy as cp

    deadline = time.time() + timeout_s
    while True:
        try:
            free, total = cp.cuda.runtime.memGetInfo()
        except Exception as exc:                          # noqa: BLE001
            # A card with nothing left refuses to create the primary
            # CONTEXT, so the query itself raises cudaErrorMemoryAllocation.
            # That is the fullest the card can be, not an error in this
            # process -- keep waiting rather than dying with a traceback
            # that reads like a bug in the code under test.
            print(f"  [vram] card refuses a context ({exc}); waiting",
                  flush=True)
            if time.time() > deadline:
                raise
            time.sleep(poll_s)
            continue
        if free >= need_bytes:
            print(f"  [vram] {free / 2**30:.1f} GiB free of "
                  f"{total / 2**30:.1f}, need {need_bytes / 2**30:.1f} -- go",
                  flush=True)
            return
        if time.time() > deadline:
            raise RuntimeError(
                f"waited {timeout_s:.0f} s for {need_bytes / 2**30:.1f} GiB "
                f"and the card never had more than {free / 2**30:.1f} free")
        print(f"  [vram] {free / 2**30:.1f} GiB free, need "
              f"{need_bytes / 2**30:.1f} -- waiting", flush=True)
        time.sleep(poll_s)


def _reserve(need_gib: float) -> None:
    """Claim ``need_gib`` into cupy's pool, in chunks.

    In CHUNKS, not one block: a card reporting 8.5 GiB free frequently
    cannot serve a single contiguous 8 GiB, and the reservation then dies
    exactly where it was meant to prevent a death.  Sixteen blocks reserve
    the same total and are the granularity the pool hands back out anyway.
    """
    import cupy as cp

    chunks = []
    step_bytes = int(need_gib * 2**30) // 16
    try:
        for _ in range(16):
            chunks.append(cp.empty(step_bytes // 4, dtype=cp.float32))
    except Exception as exc:                              # noqa: BLE001
        print(f"  [vram] reservation stopped at {len(chunks)} chunks "
              f"({exc})", flush=True)
    del chunks
    pool = cp.get_default_memory_pool()
    print(f"  [vram] reserved {pool.total_bytes() / 2**30:.2f} GiB into the "
          "cupy pool", flush=True)

def run_suite(nx: int, tile: int, outdir, *, need_gib: float = 9.0,
              only=None) -> dict:
    """Every leg, sequentially, in one process that KEEPS its device pool.

    One process, not nine, and ``free_all_blocks`` is deliberately NOT called
    between legs: on a contended card a process that hands its arena back
    loses it to somebody else and the next leg dies at its first allocation.
    The pool is released only inside a leg, where it has to be.
    """
    import cupy as cp

    from pathlib import Path

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    boundaries: dict[float, object] = {}
    reserved = False
    for name, dt, steps, poisoned, fixed, leg in SUITE:
        if only and name not in only:
            continue
        cfg = config(nx, nx, dt)
        print(f"\n=== {name}: {leg}, dt={dt:g} s, {steps} steps, "
              f"poison={poisoned}, fixed={fixed} ===", flush=True)
        wait_for_vram(int((need_gib + 2.0) * 2**30))
        if not reserved:
            # CLAIM IT, then hand it to cupy's pool rather than to the
            # driver.  Between the successful memGetInfo above and this
            # process's first real allocation there is a window, and on
            # these boxes another agent took it three times running -- the
            # leg then died with CUDA_ERROR_OUT_OF_MEMORY in its setup,
            # which reads in the log exactly like a defect in the code under
            # test.  One block, split by the pool from here on.
            # In CHUNKS, not one block: a card reporting 8.5 GiB free
            # frequently cannot serve a single contiguous 8 GiB, and the
            # reservation then dies exactly where it was meant to prevent a
            # death.  Sixteen 512 MiB-ish blocks reserve the same total and
            # are what the pool will hand back out anyway.
            _reserve(need_gib)
            reserved = True
        if dt not in boundaries:
            t0 = time.perf_counter()
            boundaries[dt] = boundaries_one_state_at_a_time(cfg)
            print(f"  boundaries for dt={dt:g} built in "
                  f"{time.perf_counter() - t0:.1f} s", flush=True)
        pindex = (POISON_INDEX[0], nx // 2 + 6, nx // 2 + 6)
        pstep = min(50, steps // 4)
        # RETRIED, not recorded as a result.  Half the allocations a leg
        # makes are outside cupy's pool -- the module loads, the stream
        # handles, the scheme tables -- so a reservation cannot cover them,
        # and on a card a dozen agents share the window closes between the
        # check and the allocation.  A leg that died that way has measured
        # nothing, and letting it into the record as a datum would be worse
        # than waiting.
        out = None
        for attempt in range(1, 25):
            try:
                if leg == "resident":
                    out = resident_leg(cfg, steps,
                                       poison=(pindex if poisoned else None),
                                       poison_step=pstep,
                                       boundaries=boundaries[dt])
                else:
                    out = streamed_leg(cfg, steps, tile, tile,
                                       poison=(pindex if poisoned else None),
                                       poison_step=pstep,
                                       boundaries=boundaries[dt], fixed=fixed)
                out["attempts"] = attempt
                break
            except Exception as exc:                      # noqa: BLE001
                text = f"{type(exc).__name__}: {exc}"
                if "out of memory" not in text.lower():
                    out = {"leg": leg, "setup_failed": text}
                    print(f"  LEG FAILED (not memory): {text}", flush=True)
                    break
                print(f"  attempt {attempt} lost the card ({text}); "
                      "releasing and waiting", flush=True)
                cp.get_default_memory_pool().free_all_blocks()
                reserved = False
                time.sleep(45)
                wait_for_vram(int((need_gib + 2.0) * 2**30))
                _reserve(need_gib)
                reserved = True
        if out is None:
            out = {"leg": leg, "setup_failed": "never got the card"}
        out["name"] = name
        out["config"] = {"nx": nx, "ny": nx, "nz": NZ, "dx": DX, "dt": dt,
                         "tile": tile, "halo": harness.halo_radius(cfg),
                         "nsteps": steps, "rung": RUNG,
                         "poison_step": pstep, "poison_index": list(pindex)}
        out["runtime_shape"] = check_runtime_still_looks_like_this()
        results[name] = out
        with open(outdir / f"{name}.json", "w") as handle:
            json.dump(out, handle, default=str)
        brief = {k: v for k, v in out.items()
                 if k in ("raised", "raised_at_step", "guard",
                          "is_run_loop_gate", "steps_completed", "nan_free",
                          "w_max_ms", "store_nonfinite_at_end",
                          "w_digest_at_t0", "final_state_w_digest",
                          "final_store_w_digest",
                          "stability_report_ms_per_call",
                          "health_ms_per_call", "step_ms_per_step",
                          "call_counts", "setup_failed")}
        print(f"  -> {json.dumps(brief, default=str)}", flush=True)
        cp.cuda.runtime.deviceSynchronize()
    return results


# --------------------------------------------------------------------------
# the fold, gated on its own, in seconds and on any card
# --------------------------------------------------------------------------

#: Keys of :func:`woof.core.dycore.stability_report` that must agree EXACTLY
#: between a monolithic reduction and the per-tile fold.  All of them, not a
#: chosen subset -- ``w_argmax`` is in here because a tile-local index would
#: have made the reported location depend on the tiling, and
#: ``boundary_w_max``/``interior_w_max`` because those are cut against the
#: DOMAIN's spec_bdy_width from inside a tile that only knows its own offset.
STABILITY_KEYS = ("u_max", "w_max", "th_max", "nan", "cfl", "horizontal_cfl",
                  "vertical_cfl", "boundary_w_max", "interior_w_max",
                  "w_argmax")


def fold_gate(nx: int = 96, ny: int = 96, nz: int = 24, tile: int = 24,
              nsteps: int = 6, periodic: bool = True) -> int:
    """Is the folded record the monolithic record?  Cheap enough to always run.

    A dry, seeded domain small enough to fit beside anything -- MEASURED
    under 400 MB -- so the arithmetic of the fold can be gated without
    waiting for a card that has room for a forecast.  The physics rungs are
    proven by the suite; what this proves is the reduction itself, including
    the two things a per-tile fold can get wrong and still look right:
    the DOMAIN flat index behind ``w_argmax`` and the DOMAIN boundary cut
    behind ``boundary_w_max``.

    Three controls, each of which must fire:

    ``window="buffer"``   fold each tile's whole gathered window instead of
                          its interior.  The obvious implementation.  The
                          halo was stepped with insufficient neighbours, so
                          it reports maxima the domain never had.
    ``a skipped tile``    read the record after ``begin_sweep`` with tiles
                          missing -- must REFUSE, because a stale record
                          folded in with current ones looks healthy.
    ``the corpse``        the same run read the way WOOF-ArWen shipped it, off the
                          DomainState, must DISAGREE -- otherwise this whole
                          module is testing nothing.
    """
    import cupy as cp

    from woof.core.dycore import stability_report
    from woof.core.dycore import step as dycore_step

    # NON-PERIODIC WITH EVERY LATERAL FLAG OFF.  Deliberate: the geometry
    # is what this gate needs -- ``periodic=False`` clamps the compute
    # windows and makes ``TileSpec.owns_x_alias`` mean ``i1 == nx``, which is
    # the branch that decides whether u's closing face is folded in exactly
    # once -- while ``specified=False`` keeps ``dycore.step`` from calling
    # ``apply_state_lateral_boundaries``, so no forcing tables are needed
    # and this gate stays runnable in seconds.  The SPECIFIED case, with
    # real tables and real physics, is the suite's job.
    cfg = harness.make_config(
        nx, ny, nz, periodic=periodic,
        **({} if periodic else dict(specified=False, open_x=False,
                                    open_y=False, nested=False, map_proj=0)))
    halo = harness.halo_radius(cfg)
    width = int(getattr(cfg, "spec_bdy_width", 0)) or None
    print(f"-- FOLD GATE  {nx}x{ny}x{nz} dry, tile {tile}, halo {halo}, "
          f"{nsteps} steps, periodic={periodic}, boundary_width={width}",
          flush=True)

    def monolithic():
        state = harness.make_state(cfg)
        harness.run_steps(state, cfg, 1)
        out = []
        for _ in range(nsteps):
            dycore_step(state, cfg)
            out.append(stability_report(state, cfg, boundary_width=width))
        cp.cuda.runtime.deviceSynchronize()
        return out

    def streamed(window="interior", store="host"):
        state = harness.make_state(cfg)
        harness.run_steps(state, cfg, 1)
        options = streaming.StreamingOptions(
            mode="on", tile_nx=tile, tile_ny=tile, nbuffers=2, store=store)
        decision = streaming.decide(cfg, options)

        def build(st, run_cfg, dec):
            return streaming.attach(
                st, run_cfg, dec,
                tile_state_factory=lambda tc: _warm_tile(tc),
                scalars=physinv.carrier_scalars(st),
                check_geography=False,
                stability_window=window)

        stepper = streaming.make_stepper(state, cfg, options,
                                         decision=decision, build=build)
        folded, corpse = [], []
        for _ in range(nsteps):
            stepper(state, cfg)
            folded.append(stepper.stability(state, cfg, boundary_width=width))
            corpse.append(stability_report(state, cfg, boundary_width=width))
        cp.cuda.runtime.deviceSynchronize()
        return stepper, folded, corpse

    def _warm_tile(tile_cfg):
        st = harness.make_state(tile_cfg)
        harness.run_steps(st, tile_cfg, 1)
        return st

    failures = []
    ref = monolithic()
    cp.get_default_memory_pool().free_all_blocks()
    stepper, got, corpse = streamed("interior")

    def compare_reports(a, b):
        bad = []
        for i, (x, y) in enumerate(zip(a, b)):
            for key in STABILITY_KEYS:
                if key not in x and key not in y:
                    continue
                if x.get(key) != y.get(key):
                    bad.append((i + 1, key, x.get(key), y.get(key)))
        return bad

    bad = compare_reports(ref, got)
    ok = not bad
    if not ok:
        failures.append(f"the folded record differs: {bad[:6]}")
    print(_line("folded record == monolithic record, every key", ok,
                f"{len(ref)} steps x {len(STABILITY_KEYS)} keys"
                if ok else f"{len(bad)} disagreements"))

    stale = compare_reports(ref, corpse)
    print(_line("the SHIPPED reading (off the DomainState) must DISAGREE",
                bool(stale), f"{len(stale)} disagreements"
                if stale else "AGREED -- this gate is testing nothing"))
    if not stale:
        failures.append("the corpse reading agreed with the monolithic one; "
                        "the defect this module exists for is not reproduced")

    # ------------------------------------------------ control: a short sweep
    stepper.stability.begin_sweep()
    try:
        stepper.stability(None, cfg, boundary_width=width)
        fired = False
    except streaming.StreamingRefused:
        fired = True
    if not fired:
        failures.append("a record with no tiles in it was served without "
                        "complaint")
    print(_line("a sweep missing tiles is REFUSED, not folded", fired))

    del stepper, got, corpse
    cp.get_default_memory_pool().free_all_blocks()

    # ------------------------------- the DISCRIMINATOR: store = "device"
    # The same tiling, the same sweep, the same observers -- and correct.
    # ``attach`` does ``store = live`` for a device store, i.e. the store IS
    # the DomainState's own arrays, so the scatter writes the very memory
    # the observers reduce over.  If the corpse reading agrees here and
    # disagrees with a host store, the defect is in WHICH MEMORY, not in the
    # tiling, not in the halo and not in the physics -- which is the whole
    # claim, and it is worth a control rather than a paragraph.
    _s3, folded3, corpse3 = streamed("interior", store="device")
    dev_fold = compare_reports(ref, folded3)
    dev_corpse = compare_reports(ref, corpse3)
    ok_dev = not dev_fold and not dev_corpse
    if not ok_dev:
        failures.append(f"with a DEVICE store the fold and/or the state "
                        f"reading disagreed: fold {dev_fold[:3]}, state "
                        f"{dev_corpse[:3]}")
    print(_line("store='device': BOTH readings match monolithic", ok_dev,
                "the tiling is innocent; the host store is the defect"))
    del _s3, folded3, corpse3
    cp.get_default_memory_pool().free_all_blocks()

    # --------------------------------------- control: fold the whole buffer
    _s2, buffered, _c2 = streamed("buffer")
    bad_buf = compare_reports(ref, buffered)
    print(_line("window='buffer' (halo folded in) must DIFFER",
                bool(bad_buf),
                f"{len(bad_buf)} disagreements, e.g. {bad_buf[:2]}"
                if bad_buf else "AGREED -- the windowing is not essential"))
    if not bad_buf:
        failures.append("folding each tile's whole gathered window gave the "
                        "same answer as folding its interior; either the "
                        "control is broken or the halo is not being stepped")
    del _s2, buffered, _c2
    cp.get_default_memory_pool().free_all_blocks()

    print()
    if failures:
        print(f"FOLD GATE FAILED -- {len(failures)} problem(s):")
        for line in failures:
            print(f"  * {line}")
        return 1
    print("FOLD GATE PASSED")
    return 0


def _line(label: str, ok: bool, detail: str = "") -> str:
    return f"  {'PASS' if ok else 'FAIL':4s}  {label:56s} {detail}"


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------

def _fmt(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def verdict(paths) -> int:
    """Read the suite's JSON and say, per set, what happened.

    Nothing here asserts that the streamed-AS-SHIPPED leg is healthy: it is
    not, and the point of printing it is that the failure is invisible from
    inside the run.  What IS asserted is that the resident control works,
    that the fold catches what the resident control catches, and that the
    fold's reading is bit-equal to the resident reading -- because a fix that
    merely raises more often could be a fix that raises wrongly.
    """
    legs = {}
    for path in paths:
        with open(path) as handle:
            payload = json.load(handle)
        legs[payload.get("name", str(path))] = payload

    failures: list[str] = []
    print("=" * 78)
    print("THE RUN LOOP'S SAFETY OBSERVERS UNDER STREAMING")
    print("=" * 78)
    any_leg = next(iter(legs.values()), {})
    cfgs = {json.dumps(v.get("config", {}), sort_keys=True) for v in
            legs.values()}
    print(f"  runtime shape: {any_leg.get('runtime_shape', {}).get('shape')}"
          f"   ({len(legs)} legs, {len(cfgs)} distinct configurations)")

    # ------------------------------------------------------- per-leg record
    print()
    print("-- EVERY LEG, AND WHICH GUARD (IF ANY) STOPPED IT")
    head = (f"   {'leg':26s} {'steps':>5s} {'guard':>22s} {'at':>4s} "
            f"{'nan_free':>8s} {'w_max_ms':>10s} {'rad':>4s} {'cu':>4s}")
    print(head)
    for name, leg in legs.items():
        if leg.get("setup_failed"):
            print(f"   {name:26s} SETUP FAILED: {leg['setup_failed'][:60]}")
            failures.append(f"{name} never ran: {leg['setup_failed'][:80]}")
            continue
        counts = leg.get("call_counts", {})
        print(f"   {name:26s} {leg['steps_completed']:5d} "
              f"{str(leg.get('guard', '-- none --')):>22s} "
              f"{str(leg.get('raised_at_step', '-')):>4s} "
              f"{str(leg['nan_free']):>8s} {leg['w_max_ms']:10.4f} "
              f"{str(counts.get('radiation', '?')):>4s} "
              f"{str(counts.get('cumulus', '?')):>4s}")

    # ------------------------------------------- SET A: the genuine blow-up
    print()
    print("-- SET A: A DOMAIN THAT GENUINELY BLOWS UP (dt above the measured "
          "stability ladder)")
    a_res = legs.get("blowup-resident", {})
    a_str = legs.get("blowup-streamed", {})
    a_fix = legs.get("blowup-streamed-fixed", {})
    if not a_res:
        print("   (not run)")
    ok = bool(a_res.get("is_run_loop_gate"))
    if a_res and not ok:
        failures.append(
            "the RESIDENT control did not raise the run loop's own NaN gate "
            "on a domain that genuinely went non-finite; the feature is "
            "broken independently of streaming")
    if a_res:
        print(f"  {_fmt(ok)}  resident: {a_res.get('guard')} at substep "
              f"{a_res.get('raised_at_step')} of "
              f"{a_res.get('config', {}).get('nsteps')}")
    caught = bool(a_str.get("is_run_loop_gate"))
    nonfin = a_str.get("store_nonfinite_at_end")
    if a_str:
        print(f"  {'DEFECT' if not caught else 'unexpected':4s}  streamed "
              f"as shipped: run-loop gate fired = {caught}; guard = "
              f"{a_str.get('guard', '-- none --')}; nan_free reported "
              f"{a_str.get('nan_free')}; {nonfin} non-finite cells in the "
              f"STORE at the end; completed "
              f"{a_str.get('steps_completed')} substeps")
    okf = bool(a_fix.get("is_run_loop_gate"))
    if a_fix and not okf:
        failures.append("the folded streamed gate did not fire on a genuine "
                        "blow-up")
    if a_fix:
        print(f"  {_fmt(okf)}  streamed with the fold: {a_fix.get('guard')} "
              f"at substep {a_fix.get('raised_at_step')} "
              f"(resident: {a_res.get('raised_at_step')})")
        same = (a_fix.get("raised_at_step") == a_res.get("raised_at_step"))
        print(f"  {_fmt(same)}  the fold fires on the SAME substep as the "
              f"resident control")
        if not same:
            failures.append("the folded gate fired on a different substep "
                            "from the resident control")

    # --------------------------------------------- SET B: what was reported
    print()
    print("-- SET B: THE READING (no poison, a dt the ladder measured "
          "stable)")
    ref = legs.get("resident-clean")
    for key in ("streamed-clean", "streamed-clean-fixed"):
        got = legs.get(key)
        if ref is None or got is None or got.get("setup_failed"):
            continue
        a, b = ref["observed_trace"], got["observed_trace"]
        truth = got.get("true_trace") or []
        n = min(len(a), len(b))
        same = sum(1 for i in range(n) if a[i] == b[i])
        tsame = sum(1 for i in range(min(n, len(truth))) if a[i] == truth[i])
        frozen = len(set(b[:n])) == 1
        ok = (same == n)
        if key.endswith("fixed") and not ok:
            failures.append("the folded streamed w_max is not bit-equal to "
                            "the resident reading")
        print(f"  {_fmt(ok)}  {key}: {same}/{n} substeps bit-equal to the "
              "resident reading"
              + ("   [FROZEN: one value for the whole run]" if frozen else ""))
        print(f"        the STORE's true w_max matched the resident reading "
              f"{tsame}/{min(n, len(truth))} substeps -- the DOMAIN is "
              "bit-exact either way; only the observer is not")
        print(f"        reported w_max_ms  resident {ref['w_max_ms']!r}  "
              f"{key} {got['w_max_ms']!r}")
        print(f"        reported boundary/interior w_max  resident "
              f"{ref['boundary_w_max_ms']!r} / {ref['interior_w_max_ms']!r}"
              f"   {key} {got['boundary_w_max_ms']!r} / "
              f"{got['interior_w_max_ms']!r}")
        print(f"        w_max_boundary_row (needs the DOMAIN argmax)  "
              f"resident {ref['w_max_boundary_row']!r}   {key} "
              f"{got['w_max_boundary_row']!r}")
        if truth:
            print(f"        the store's true final w_max = {truth[-1]!r}")
        d0, d1 = got.get("w_digest_at_t0"), got.get("final_state_w_digest")
        print(f"        DomainState w digest: t0 {d0}  end {d1}  "
              f"{'UNCHANGED -- the state is a corpse' if d0 == d1 else 'moved'}"
              f";  store w digest end {got.get('final_store_w_digest')}")

    # ------------------------------------------------- SET C: the poison
    print()
    print("-- SET C: ONE CELL OF THE STORE SET TO NaN AT SUBSTEP 50")
    for key in ("resident-poisoned", "streamed-poisoned",
                "streamed-poisoned-fixed"):
        leg = legs.get(key)
        if leg is None or leg.get("setup_failed"):
            continue
        print(f"        {key:26s} target={leg.get('poison_target', '-')}  "
              f"guard={leg.get('guard', '-- none --')}  at="
              f"{leg.get('raised_at_step', '-')}  "
              f"run-loop gate={leg.get('is_run_loop_gate', False)}  "
              f"store non-finite at end="
              f"{leg.get('store_nonfinite_at_end', '-')}")

    # --------------------------------------------------------- the price
    print()
    print("-- WHAT THE OBSERVERS COST, PER CALL, ON BOTH SIDES")
    print(f"   {'leg':26s} {'stability ms':>13s} {'health ms':>10s} "
          f"{'step ms':>9s} {'stability % of wall':>20s}")
    for name, leg in legs.items():
        if leg.get("setup_failed"):
            continue
        share = (100.0 * leg["stability_report_ms_per_call"] * 1e-3
                 * leg["stability_report_calls"]
                 / max(1e-9, leg["wall_seconds"]))
        print(f"   {name:26s} {leg['stability_report_ms_per_call']:13.3f} "
              f"{leg['health_ms_per_call']:10.3f} "
              f"{leg['step_ms_per_step']:9.1f} {share:19.2f}%")

    print()
    print("=" * 78)
    if failures:
        print(f"OBSERVER GATE FAILED -- {len(failures)} problem(s):")
        for line in failures:
            print(f"  * {line}")
        return 1
    print("OBSERVER GATE PASSED -- the resident control fires, the fold "
          "fires where the control fires, and its reading is bit-equal.")
    return 0



# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg", choices=["resident", "streamed"])
    parser.add_argument("--name", default=None)
    parser.add_argument("--poison", action="store_true")
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--tile", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--poison-step", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--ladder", nargs="*", type=float, default=None)
    parser.add_argument("--verdict", nargs="*", default=None)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--outdir", default="obs")
    parser.add_argument("--need-gib", type=float, default=9.0)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--fold-gate", action="store_true")
    args = parser.parse_args(argv)

    if args.verdict:
        return verdict(args.verdict)

    import cupy as cp

    nx = ny = args.nx or (QUICK["nx"] if args.quick else NX)
    tile = args.tile or (QUICK["tile"] if args.quick else TX)
    nsteps = args.steps or (QUICK["nsteps"] if args.quick else NSTEPS)
    pstep = args.poison_step or (
        QUICK["poison_step"] if args.quick else POISON_STEP)
    pindex = (QUICK["index"] if args.quick else POISON_INDEX)
    if nx != (QUICK["nx"] if args.quick else NX):
        pindex = (POISON_INDEX[0], nx // 2 + 6, nx // 2 + 6)
    dt = args.dt or DT

    try:
        free, total = cp.cuda.runtime.memGetInfo()
        name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        print(f"{name}  {free / 2**30:.1f} GiB free of {total / 2**30:.1f}  "
              f"cupy {cp.__version__}", flush=True)
    except Exception as exc:                              # noqa: BLE001
        print(f"card is too full to report itself yet ({exc})", flush=True)

    if args.fold_gate:
        rc = fold_gate(periodic=True)
        rc |= fold_gate(nx=128, ny=96, nz=24, tile=32, periodic=False)
        return rc

    if args.suite:
        run_suite(nx, tile, args.outdir, need_gib=args.need_gib,
                  only=args.only)
        return 0

    if args.ladder is not None:
        dts = args.ladder or [18.0, 12.0, 9.0, 6.0, 3.0]
        print(f"-- STABILITY LADDER of the RESIDENT reference, {nx}x{ny}x{NZ} "
              f"dx={DX:g} m, {nsteps} steps", flush=True)
        out = stability_ladder(nx, ny, dts, nsteps)
        print(json.dumps(out, indent=2, default=str), flush=True)
        if args.out:
            with open(args.out, "w") as handle:
                json.dump(out, handle, default=str)
        return 0

    cfg = config(nx, ny, dt)
    halo = harness.halo_radius(cfg)
    print(f"domain {nx}x{ny}x{NZ}  dx={cfg.dx:g} m  dt={cfg.dt:g} s  "
          f"tile {tile}x{tile}  halo {halo}  "
          f"compute window {tile + 2 * halo} cells  "
          f"{nsteps} steps = {nsteps * cfg.dt / 60:.0f} forecast minutes",
          flush=True)
    print(f"radt {cfg.radt_minutes:g} min = every "
          f"{cfg.radt_minutes * 60 / cfg.dt:.1f} steps;  "
          f"cudt {cfg.cudt_minutes:g} min = every "
          f"{cfg.cudt_minutes * 60 / cfg.dt:.1f} steps", flush=True)
    faithful = check_runtime_still_looks_like_this()
    print(f"runtime.integrate_prepared_case is in the "
          f"{faithful['shape']} shape:  {faithful}", flush=True)
    if faithful["shape"] == "UNRECOGNISED":
        raise SystemExit(
            "integrate_prepared_case matches neither the shipped nor the "
            "fixed observer shape; this module's transcription is stale and "
            "anything it measures is about itself")

    # The boundary sources are built and freed BEFORE the domain, so the two
    # un-stepped states and the 15.6 GiB prepared domain never coexist.
    t0 = time.perf_counter()
    bnd = boundaries_one_state_at_a_time(cfg)
    cp.get_default_memory_pool().free_all_blocks()
    print(f"boundaries built in {time.perf_counter() - t0:.1f} s", flush=True)

    poison = pindex if args.poison else None
    if args.leg == "resident":
        result = resident_leg(cfg, nsteps, poison=poison, poison_step=pstep,
                              boundaries=bnd)
    else:
        result = streamed_leg(cfg, nsteps, tile, tile, poison=poison,
                              poison_step=pstep, boundaries=bnd,
                              fixed=args.fixed)
    result["name"] = args.name or (
        f"{args.leg}-{'poisoned' if args.poison else 'clean'}"
        + ("-fixed" if args.fixed else ""))
    result["config"] = {"nx": nx, "ny": ny, "nz": NZ, "dx": cfg.dx,
                        "dt": cfg.dt, "tile": tile, "halo": halo,
                        "nsteps": nsteps, "rung": RUNG}
    result["runtime_faithful"] = faithful

    show = {k: v for k, v in result.items()
            if k not in ("observed_trace", "true_trace")}
    print(json.dumps(show, indent=2, default=str), flush=True)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, default=str)
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
