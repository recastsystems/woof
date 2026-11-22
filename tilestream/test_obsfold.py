"""DEFECT 2: the run loop's safety gates observe a state the sweep never writes.

WHAT IS WRONG
-------------
``woof.runtime.integrate_prepared_case`` guards every dynamics substep with
three observers, and all three read the resident :class:`DomainState` object:

    runtime.py:2234   stability_report(state, integration_cfg, ...)  -> nan / w / CFL
    runtime.py:2212   health.require_healthy(...)   (StateHealthValidator(state))
    runtime.py:2232   health.require_healthy(...)   on a cadence
    runtime.py:2257   cp.max(state.physics.fields["swdown"])

``woof.core.model.execute_experiment`` does the same with
``StateHealthValidator(node.state)`` (model.py:843).

Under ``[tiles]`` with ``store = "host"`` the domain's arrays do not live
on that state.  :func:`woof.core.streaming.attach` copies the prepared
state's carriers into a pinned host store and every model step is a SWEEP over
that store: tiles are gathered to the card, stepped, and their interiors
scattered back to the store.  ``state`` is passed to ``StreamedDomain.__call__``
only to be CHECKED for identity -- its device arrays are never written again.

So the observers read a corpse.  It was healthy when the store was filled from
it, and it stays healthy forever:

* ``nan_free`` stays ``True`` no matter what the store does,
* ``w_max`` stays at the value it had at t=0,
* the CFL never moves,
* ``swdown_peak`` never moves,

and a run that went non-finite in the store completes "successfully" and
writes a checkpoint.  A silently disarmed NaN guard is indistinguishable from
a well-behaved forecast.  It is also a real GPU reduction per substep on data
nobody is using, so it is simultaneously wrong AND a measurable tax.

THE FIX, AND WHY IT IS LEGITIMATE
---------------------------------
Every quantity these observers compute is a MAX fold or an OR fold:
``u_max``, ``w_max``, ``th_max``, the boundary/interior ``w`` maxima and the
vertical CFL rate are maxima; ``nan`` is an OR of NaN flags.  Max and OR are
associative, commutative and idempotent, so they can be folded per tile inside
the sweep and combined -- and because float max is exact (it selects an
operand, it never rounds) the folded answer is BIT-IDENTICAL to the monolithic
one, not merely close.

The defect is which memory the reduction reads, not the mathematics.

Three details make the per-tile fold equal to the whole-domain one rather than
merely similar, and each is a place where a plausible implementation is wrong:

1.  **Interiors only.**  A tile's gathered array is interior + halo.  The halo
    is a copy of a neighbour's cells that this tile did not integrate, and on
    a domain edge it is seam fill (``zeros``/``poison``).  Folding it would
    reduce over values no monolithic run ever holds -- and with the poison
    seam it would report NaN on a perfectly healthy forecast.  The interiors
    of a plan partition the domain exactly once, which is what makes the union
    of per-tile folds the domain fold.

2.  **Domain indices, not tile indices.**  ``w_argmax`` breaks ties by LOWEST
    FLAT INDEX (``health.cu``'s ``update_max``), and the boundary/interior
    split is a function of the DOMAIN's ``(j, i)`` against
    ``spec_bdy_width``.  A fold that compares tile-local indices picks a
    different winner among equals and mislabels every tile's rim as a domain
    boundary.

3.  **The staggered faces are owned, not shared.**  ``u`` is ``(nz, ny, nx+1)``
    and a tile owns faces ``[i0, i1)`` plus, for exactly one tile, the closing
    alias/boundary slot ``nx``; likewise ``v`` in ``y``.  Folding ``[i0, i1]``
    on every tile would double-count shared faces (harmless for a max) but
    folding ``[i0, i1)`` on every tile and forgetting the closing slot DROPS a
    face -- and the dropped face is a domain edge, exactly where a blow-up
    starts.  :mod:`tilestream.spec` already owns this arithmetic; the fold
    asks it rather than re-deriving it.

WHAT THIS MODULE DOES
---------------------
    python -m tilestream.test_obsfold --demonstrate   # the disarm + control
    python -m tilestream.test_obsfold --verify        # bit-exact w_max
    python -m tilestream.test_obsfold --price         # per-substep cost

``--demonstrate`` poisons the STORE (never the state) through the tile hook
and shows the streamed run completing with ``nan_free=True``; the identical
poison applied to the resident leg's state MUST raise, which is the positive
control that proves the comparison is capable of failing at all.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from woof.core import streaming
from tilestream import harness, physics_inventory as physinv
from tilestream import test_join

# --------------------------------------------------------------------------
# the configuration under test
# --------------------------------------------------------------------------

#: 672 x 672 x 49 at dx = 3 km is the geometry the streaming lane's measured
#: forecast-hour number was taken on (vanilla 6.27 min vs streamed 12.59 min
#: on a 4090, identical digest over 229 carriers at step 240).  Tile 168
#: DIVIDES 672 -- 4 x 4 = 16 tiles -- which the transport requires.
NX = NY = 672
TILE = 168
NZ = 49
DX = 3000.0
#: dt = 3 s is a 3 km grid's own step, and it is the step the physics cadences
#: below are quoted against: radt = 12 min fires every 240 steps and
#: cudt = 5 min every 100.  A window that crosses NEITHER boundary measures a
#: configuration in which radiation and cumulus never ran.
DT = 3.0
RUNG = "full+MYNN+Noah-MP"
SEED = test_join.SEED

#: The poisoned cell, from the task: a mid-column w in the interior of a tile
#: that is NOT the first tile of the sweep, so the poison written at tile 0's
#: hook is gathered by its owner later in the SAME sweep.
POISON_K, POISON_J, POISON_I = 20, 300, 300
POISON_FIELD = "state/w"


def poison_index(cfg) -> tuple[int, int, int]:
    """The poisoned cell for this domain: ``(20, 300, 300)`` at 672.

    Scaled only when a smaller domain is being used to shake the harness out;
    the scaled cell keeps the property the demonstration depends on, which is
    that it is NOT in tile 0 and NOT in the boundary zone.
    """
    if int(cfg.nx) >= 672 and int(cfg.ny) >= 672:
        return POISON_K, POISON_J, POISON_I
    k = min(POISON_K, int(cfg.nz) - 1)
    return k, int(cfg.ny) * 300 // 672, int(cfg.nx) * 300 // 672


def obs_cfg(nx: int = NX, ny: int = NY, *, nz: int = NZ, rung: str = RUNG,
            dx: float = DX, dt: float = DT):
    """The real forecast configuration: Lambert, terrain, specified BCs."""
    return test_join.join_cfg(nx, ny, nz=nz, rung=rung, dx=dx, dy=dx, dt=dt)


def sequential_boundaries(cfg, *, seed: int = SEED, seed_b: int | None = None,
                          seconds: float = test_join.BDY_SECONDS):
    """The domain's specified forcing, built ONE STATE AT A TIME.

    ``test_join.domain_boundaries`` holds both snapshot states at once, which
    at 672 x 672 x 49 full physics is ~2 x 15 GiB and does not fit on a 24 GB
    card.  :class:`woof.ingest.lateral_bc.StateBoundaryFrames` exists for
    exactly this: it keeps only the four ``spec_bdy_width`` perimeter frames
    of each state and lets the caller drop the state immediately, and its
    docstring records that the result is element-for-element what the
    all-at-once builder returns.  So this is the same forcing, not a cheaper
    approximation of it.

    Two DIFFERENT seeds, because a repeated snapshot gives a zero time
    tendency and quietly disarms the clock -- see ``test_join``.
    """
    import cupy as cp
    from woof.ingest.lateral_bc import StateBoundaryFrames

    seed_b = seed + 1 if seed_b is None else seed_b
    frames = StateBoundaryFrames(spec_bdy_width=int(cfg.spec_bdy_width),
                                 spec_zone=int(cfg.spec_zone),
                                 relax_zone=int(cfg.relax_zone))
    for one in (seed, seed_b):
        state, _geo = test_join.build_domain(cfg, seed=one, warmup=0)
        frames.add_state(state)
        del state
        cp.get_default_memory_pool().free_all_blocks()
    return frames.build([0.0, float(seconds)])


# --------------------------------------------------------------------------
# the run loop's observers, extracted verbatim
# --------------------------------------------------------------------------

class RuntimeObservers:
    """``integrate_prepared_case``'s per-substep safety block, as it is today.

    Lifted from ``runtime.py`` lines 2231-2260 so that both legs of every
    comparison run THE SAME accounting and the only difference between them is
    where the report came from.  The cadence logic is the runtime's:
    ``stability_report`` every substep, ``require_healthy`` every 4th substep
    plus the mandatory instants, and ``nan_free`` latched across the run.

    ``raised`` records the substep the RuntimeError came from, because "the
    run completed" and "the run raised at substep 51" is the entire finding.
    """

    def __init__(self, cfg, *, width: int):
        self.cfg = cfg
        self.width = int(width)
        self.nan_free = True
        self.w_max = 0.0
        self.w_max_boundary_row = None
        self.boundary_w_max = 0.0
        self.interior_w_max = 0.0
        self.cfl_max = 0.0
        self.swdown_peak = 0.0
        self.raised: str | None = None
        self.reports = 0

    def observe(self, report: dict, step_index: int) -> None:
        """One substep's accounting.  Raises exactly where the runtime does."""
        self.reports += 1
        self.nan_free = self.nan_free and not report["nan"]
        step_w_max = float(report["w_max"])
        if step_w_max > self.w_max:
            max_index = int(report["w_argmax"])
            _k, j, i = np.unravel_index(
                max_index, (self.cfg.nz + 1, self.cfg.ny, self.cfg.nx))
            distance = min(int(j), self.cfg.ny - 1 - int(j),
                           int(i), self.cfg.nx - 1 - int(i))
            self.w_max_boundary_row = (distance if distance < self.width
                                       else None)
            self.w_max = step_w_max
        self.boundary_w_max = max(self.boundary_w_max,
                                  float(report["boundary_w_max"]))
        self.interior_w_max = max(self.interior_w_max,
                                  float(report["interior_w_max"]))
        cfl = report.get("cfl")
        if cfl is not None and np.isfinite(cfl):
            self.cfl_max = max(self.cfl_max, float(cfl))
        if not self.nan_free:
            self.raised = (
                "real-case integration produced a non-finite state at "
                f"dynamics substep {step_index}")
            raise RuntimeError(self.raised)

    def summary(self) -> dict:
        return {
            "nan_free": bool(self.nan_free),
            "w_max_ms": float(self.w_max),
            "w_max_boundary_row": self.w_max_boundary_row,
            "boundary_w_max_ms": float(self.boundary_w_max),
            "interior_w_max_ms": float(self.interior_w_max),
            "cfl_max": float(self.cfl_max),
            "swdown_peak_wm2": float(self.swdown_peak),
            "reports": int(self.reports),
            "raised": self.raised,
        }


# --------------------------------------------------------------------------
# poisoning the store through the tile hook
# --------------------------------------------------------------------------

class StorePoison:
    """Writes one NaN into the STORE, from inside the sweep, at one step.

    The poison goes in through the ``tile_hook`` -- the sweep's own per-tile
    callback -- so it lands MID-STEP, between two tiles of the same sweep,
    which is where a real blow-up would appear.  It never touches the resident
    ``DomainState``: that is the whole point, because the observers read the
    state and the forecast lives in the store.

    Fired at tile 0 of the target step so that the owning tile (which is not
    tile 0) gathers the poisoned cell LATER IN THE SAME SWEEP, integrates it,
    and scatters NaN back.  If it were fired after its owner had already
    scattered, the store would still end the step poisoned -- the fold would
    simply catch it one step later.
    """

    def __init__(self, *, step: int, index, value: float = float("nan"),
                 field: str = POISON_FIELD, at_tile: int = 0):
        self.step = int(step)
        self.field = field
        self.index = tuple(int(v) for v in index)
        self.value = float(value)
        self.at_tile = int(at_tile)
        self.store: dict | None = None
        self.istep = 0
        self.fired = False
        self.fired_at: tuple[int, int] | None = None

    def begin_step(self, istep: int) -> None:
        self.istep = int(istep)

    def maybe_fire(self, itile: int) -> None:
        if self.fired or self.store is None:
            return
        if self.istep == self.step and itile == self.at_tile:
            self.store[self.field][self.index] = self.value
            self.fired = True
            self.fired_at = (self.istep, int(itile))

    def install(self, module=streaming):
        """Patch ``make_tile_hook`` so the built hook also poisons.

        Patched only across the BUILD: the hook object the driver keeps is
        created once, and it closes over this instance, so the poison is
        controlled by mutating ``self`` during the run rather than by leaving
        a module attribute patched.
        """
        poison = self

        class _Patch:
            def __enter__(self):
                self.real = module.make_tile_hook

                def patched(per_tile, **kwargs):
                    inner = self.real(per_tile, **kwargs)

                    def hook(tile_state, tspec, itile, stream):
                        inner(tile_state, tspec, itile, stream)
                        poison.maybe_fire(itile)

                    return hook

                module.make_tile_hook = patched
                return poison

            def __exit__(self, *exc):
                module.make_tile_hook = self.real
                return False

        return _Patch()


def owning_tile(cfg, decision, j: int, i: int) -> int:
    """Which tile of the plan owns domain mass cell ``(j, i)``."""
    for itile, spec in enumerate(streaming.tile_specs(cfg, decision)):
        if spec.j0 <= j < spec.j1 and spec.i0 <= i < spec.i1:
            return itile
    raise AssertionError(f"no tile owns ({j}, {i})")


# --------------------------------------------------------------------------
# the two legs
# --------------------------------------------------------------------------

def _swdown_peak(state) -> float:
    """``runtime.py:2257``'s per-outer-step read, tolerant of a dry state.

    The runtime does this unconditionally because a prepared real case always
    has a physics driver; the dry rung here has none, and the dry rung is the
    control that isolates the NaN gate from the physics schemes' own input
    validation, so it has to be runnable.
    """
    import cupy as cp

    physics = getattr(state, "physics", None)
    if physics is None:
        return 0.0
    field = physics.fields.get("swdown")
    return 0.0 if field is None else float(cp.max(field))


def fire_counts(obj) -> dict:
    """``call_counts`` for whichever side of the comparison this is.

    Printed on BOTH legs of every comparison and checked, because a number
    measured in a window where radiation and cumulus never fired is the single
    most common false result this project has produced.
    """
    if isinstance(obj, dict):
        counts = dict(obj.get("call_counts") or {})
    else:
        counts = dict(getattr(getattr(obj, "physics", None),
                              "call_counts", {}) or {})
    return {k: int(v) for k, v in sorted(counts.items())}


def run_resident(cfg, nsteps, *, boundaries, poison_step=None,
                 seed=SEED, warmup=1, observer="state", index=None,
                 value: float = float("nan")):
    """The reference leg: one resident domain, WOOF-ArWen's own ``dycore.step``.

    ``observer`` is accepted and must be ``"state"``: on a resident domain the
    state IS the forecast, so there is nothing to fold and reading the state is
    correct.  It exists so the caller cannot ask for a fold here by accident.
    """
    import cupy as cp
    from woof.core.dycore import stability_report, step as dycore_step

    if observer != "state":
        raise ValueError("the resident leg observes its own state")
    pk, pj, pi = poison_index(cfg) if index is None else index
    state, _geo = test_join.build_domain(cfg, seed=seed, boundaries=boundaries,
                                         warmup=warmup)
    obs = RuntimeObservers(cfg, width=int(cfg.spec_bdy_width))
    poisoned_at = None
    trace: list[dict] = []
    crashed = None
    t0 = time.perf_counter()
    try:
        for istep in range(1, int(nsteps) + 1):
            if poison_step is not None and istep == int(poison_step):
                # The IDENTICAL poison, applied where this leg keeps its
                # forecast: the state.  Before the step, matching the tile
                # hook, which fires before its tile is integrated.
                state.w[pk, pj, pi] = float(value)
                poisoned_at = istep
            dycore_step(state, cfg, refl_10cm_due=False)
            report = stability_report(state, cfg,
                                      boundary_width=int(cfg.spec_bdy_width))
            trace.append({"step": istep,
                          "w_max": float(report["w_max"]),
                          "u_max": float(report["u_max"]),
                          "th_max": float(report["th_max"]),
                          "cfl": (None if report["cfl"] is None
                                  else float(report["cfl"])),
                          "boundary_w_max": float(report["boundary_w_max"]),
                          "interior_w_max": float(report["interior_w_max"]),
                          "w_argmax": int(report["w_argmax"]),
                          "nan": bool(report["nan"])})
            obs.observe(report, istep)
            peak = _swdown_peak(state)
            obs.swdown_peak = max(obs.swdown_peak, peak)
    except RuntimeError:
        pass
    except ValueError as exc:
        crashed = f"{type(exc).__name__}: {exc}"
    cp.cuda.runtime.deviceSynchronize()
    out = obs.summary()
    out.update(leg="resident", steps=int(nsteps), poisoned_at=poisoned_at,
               wall_s=time.perf_counter() - t0, fires=fire_counts(state),
               elapsed_seconds=float(state.elapsed_seconds),
               trace=trace, crashed=crashed, completed=len(trace))
    return out, state


def run_streamed(cfg, nsteps, *, boundaries, tile=TILE, poison_step=None,
                 seed=SEED, warmup=1, nbuffers=2, observer="state",
                 store="host", index=None, truth_every: int = 0,
                 value: float = float("nan"), control: str | None = None):
    """The streamed leg, driven through ``streaming.make_stepper``.

    ``observer="state"``  -- today's runtime: ``stability_report(state, ...)``
                             on the resident DomainState the sweep abandoned.
    ``observer="fold"``   -- the fix: the maxima the sweep folded per tile out
                             of the STORE, which is where the forecast is.
    """
    import cupy as cp
    from woof.core.dycore import stability_report

    options = streaming.StreamingOptions(
        mode="on", tile_nx=int(tile), tile_ny=int(tile),
        nbuffers=int(nbuffers), store=store)
    decision = streaming.decide(cfg, options)
    pk, pj, pi = poison_index(cfg) if index is None else index
    poison = None
    if poison_step is not None:
        owner = owning_tile(cfg, decision, pj, pi)
        if owner == 0:
            raise AssertionError(
                "the poisoned cell is owned by tile 0, so firing the poison "
                "at tile 0's hook would not be gathered later in the sweep")
        poison = StorePoison(step=int(poison_step), index=(pk, pj, pi),
                             value=value)

    domain, geo = test_join.build_domain(cfg, seed=seed,
                                         boundaries=boundaries, warmup=warmup)
    build = test_join._make_builder(
        cfg, domain, geo, boundaries=boundaries, seam="zeros", snapshot=None,
        geography=True, carry_clock=True, window_tables=True)
    if poison is not None:
        with poison.install():
            stepper = streaming.make_stepper(domain, cfg, options,
                                             decision=decision, build=build)
        poison.store = stepper.store
    else:
        stepper = streaming.make_stepper(domain, cfg, options,
                                         decision=decision, build=build)
    assert streaming.is_streaming(stepper)
    if control is not None:
        # THE NEGATIVE CONTROL: break exactly one of the fold's three
        # essential properties, in place, after the run is built.  Each
        # one MUST make the comparison against the resident reduction fail;
        # a control that passes is a control that proves nothing.
        fold = stepper.tiled_run.health
        if fold is None:
            raise AssertionError("this streamed run has no health fold to "
                                 "break, so the control cannot fire")
        fold.control = control

    obs = RuntimeObservers(cfg, width=int(cfg.spec_bdy_width))
    trace: list[dict] = []
    crashed = None
    t0 = time.perf_counter()
    try:
        for istep in range(1, int(nsteps) + 1):
            if poison is not None:
                poison.begin_step(istep)
            stepper(domain, cfg, refl_10cm_due=False)
            if observer == "state":
                # THE DEFECT, exactly as runtime.py:2234 has it.
                report = stability_report(
                    domain, cfg, boundary_width=int(cfg.spec_bdy_width))
                peak = _swdown_peak(domain)
            elif observer == "fold":
                report = stepper.report["health"]
                # Absent at the dry rung, which has no physics driver and so
                # no swdown at all -- the same case ``_swdown_peak`` handles
                # on the resident leg.
                peak = float(report.get("swdown_max", 0.0))
            else:
                raise ValueError(f"unknown observer {observer!r}")
            row = {"step": istep,
                   "w_max": float(report["w_max"]),
                   "u_max": float(report["u_max"]),
                   "th_max": float(report["th_max"]),
                   "cfl": (None if report.get("cfl") is None
                           else float(report["cfl"])),
                   "boundary_w_max": float(report["boundary_w_max"]),
                   "interior_w_max": float(report["interior_w_max"]),
                   "w_argmax": int(report["w_argmax"]),
                   "nan": bool(report["nan"])}
            if truth_every and (istep % truth_every == 0 or istep == 1
                                or (poison is not None
                                    and istep == poison.step)):
                row["store"] = store_probe(stepper.store)
            trace.append(row)
            obs.observe(report, istep)
            obs.swdown_peak = max(obs.swdown_peak, peak)
    except RuntimeError:
        pass
    except ValueError as exc:
        # A physics scheme's OWN finite check, not the safety gate.  MYNN's
        # mass-flux driver validates its inputs, so at that rung a NaN in the
        # store can kill the process before the (disarmed) gate would ever
        # have had an opinion.  Recorded separately, because "the run died"
        # and "the gate noticed" are different claims.
        crashed = f"{type(exc).__name__}: {exc}"
    cp.cuda.runtime.deviceSynchronize()
    out = obs.summary()
    out.update(leg=f"streamed[{observer}]", steps=int(nsteps),
               poisoned_at=(poison.fired_at if poison else None),
               wall_s=time.perf_counter() - t0,
               fires=fire_counts(stepper.scalars),
               elapsed_seconds=float(stepper.elapsed_seconds),
               tiles=len(streaming.tile_specs(cfg, decision)),
               trace=trace, crashed=crashed, completed=len(trace))
    return out, stepper


# --------------------------------------------------------------------------
# store-side truth, for checking the observers against
# --------------------------------------------------------------------------

def store_probe(store) -> dict:
    """What the STORE holds right now, reduced on the host with numpy.

    An INDEPENDENT third opinion: it shares no code with the device kernel
    the observers use, so when it and the observer disagree the disagreement
    is evidence rather than a tautology.  Semantics are the kernel's --
    ``health.cu`` excludes NaN from the maxima and reports it separately --
    so ``w_absmax`` is the max over FINITE cells only, which is what
    ``stability_report`` would return for the same array.
    """
    w = np.asarray(store["state/w"])
    finite = np.isfinite(w)
    nnan = int(w.size - np.count_nonzero(finite))
    return {
        "w_absmax": float(np.abs(w[finite]).max()) if finite.any()
        else float("nan"),
        "nonfinite_w": nnan,
        "nan": bool(nnan),
    }


def store_truth(store) -> dict:
    """What the STORE actually holds, computed on the host with numpy.

    Deliberately not the kernel and not the fold: an independent third opinion
    that answers "did the forecast really go non-finite" without sharing a
    line of code with either observer.  Slow, and only used on the handful of
    steps a demonstration inspects.
    """
    w = np.asarray(store["state/w"])
    u = np.asarray(store["state/u"])
    th = np.asarray(store["state/thp"])
    finite = {name: int(np.count_nonzero(~np.isfinite(np.asarray(arr))))
              for name, arr in (("w", w), ("u", u), ("thp", th))}
    return {
        "nonfinite_cells": finite,
        "nan_anywhere": bool(sum(finite.values()) > 0),
        "w_absmax_finite": float(np.nanmax(np.abs(w[np.isfinite(w)])))
        if np.isfinite(w).any() else float("nan"),
    }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


FOLD_KEYS = ("w_max", "u_max", "th_max", "cfl", "boundary_w_max",
             "interior_w_max", "w_argmax", "nan")


def compare_traces(ref: list, got: list) -> dict:
    """Element-for-element equality of two legs' per-substep reports.

    Exact ``==`` on the floats, deliberately: the fold is a reduction over
    maxima, and a maximum SELECTS an operand rather than combining operands,
    so a correct per-tile fold is bit-identical to the whole-domain one.  A
    tolerance here would accept exactly the reordering bugs the fold could
    plausibly have.
    """
    n = min(len(ref), len(got))
    diffs = []
    for row_r, row_g in zip(ref[:n], got[:n]):
        for key in FOLD_KEYS:
            a, b = row_r.get(key), row_g.get(key)
            if a is None and b is None:
                continue
            if isinstance(a, float) and isinstance(b, float):
                same = (a == b) or (np.isnan(a) and np.isnan(b))
            else:
                same = a == b
            if not same:
                diffs.append({"step": row_r["step"], "key": key,
                              "resident": a, "streamed": b})
    return {"steps_compared": n, "len_ref": len(ref), "len_got": len(got),
            "mismatches": diffs}



def price(cfg, nsteps, tile, bnd, *, seed=SEED, repeats=3):
    """What the per-substep safety observation COSTS, before and after.

    BEFORE is ``stability_report(state, ...)`` on the resident DomainState --
    which is what a streamed run pays today, every substep, to reduce a domain
    nobody is integrating.  AFTER is the marginal cost of folding the same
    reductions per tile inside the sweep, measured as the SAME streamed run
    with the fold enabled and disabled: same tiles, same transport, same
    physics, one launch per tile added.
    """
    import cupy as cp
    from woof.core.dycore import stability_report

    width = int(cfg.spec_bdy_width)
    out = {}

    # -- BEFORE: the whole-domain reduction on the resident state -----------
    state, _geo = test_join.build_domain(cfg, seed=seed, boundaries=bnd,
                                         warmup=1)
    for _ in range(5):
        stability_report(state, cfg, boundary_width=width)
    cp.cuda.runtime.deviceSynchronize()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(20):
            stability_report(state, cfg, boundary_width=width)
        cp.cuda.runtime.deviceSynchronize()
        samples.append((time.perf_counter() - t0) / 20.0)
    out["resident_stability_report_s"] = min(samples)
    del state
    cp.get_default_memory_pool().free_all_blocks()

    # -- AFTER: the same run with the fold on and off -----------------------
    options = streaming.StreamingOptions(
        mode="on", tile_nx=int(tile), tile_ny=int(tile), nbuffers=2,
        store="host")
    decision = streaming.decide(cfg, options)
    domain, geo = test_join.build_domain(cfg, seed=seed, boundaries=bnd,
                                         warmup=1)
    build = test_join._make_builder(
        cfg, domain, geo, boundaries=bnd, seam="zeros", snapshot=None,
        geography=True, carry_clock=True, window_tables=True)
    stepper = streaming.make_stepper(domain, cfg, options, decision=decision,
                                     build=build)
    fold = stepper.tiled_run.health
    if fold is None:
        raise AssertionError("this streamed run has no health fold to price")
    timings = {}
    for label, enabled in (("warm", True), ("fold_on", True),
                           ("fold_off", False), ("fold_on2", True)):
        fold.enabled = enabled
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        for _ in range(int(nsteps)):
            stepper(domain, cfg, refl_10cm_due=False)
        cp.cuda.runtime.deviceSynchronize()
        timings[label] = (time.perf_counter() - t0) / int(nsteps)
    fold.enabled = True
    on = min(timings["fold_on"], timings["fold_on2"])
    out.update(streamed_step_fold_on_s=on,
               streamed_step_fold_off_s=timings["fold_off"],
               fold_marginal_s=on - timings["fold_off"],
               fold_share=(on - timings["fold_off"]) / on if on else 0.0,
               tiles=len(streaming.tile_specs(cfg, decision)),
               raw=timings)
    del stepper, domain
    cp.get_default_memory_pool().free_all_blocks()
    return out


def _line(label: str, ok: bool, detail: str = "") -> str:
    return f"  [{'PASS' if ok else 'FAIL'}] {label}" + (
        f"   {detail}" if detail else "")


def _report_line(tag: str, res: dict) -> str:
    return (f"  {tag:<26} nan_free={str(res['nan_free']):<5} "
            f"w_max={res['w_max_ms']:.9g}  reports={res['reports']:<4} "
            f"raised={'YES' if res['raised'] else 'no'}  "
            f"{res['wall_s']:.1f} s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=NX)
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--poison-step", type=int, default=50)
    ap.add_argument("--rung", default=RUNG)
    ap.add_argument("--dx", type=float, default=DX)
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--observer", default="state", choices=("state", "fold"))
    ap.add_argument("--demonstrate", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--price", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--compare", nargs=2, default=None,
                    metavar=("RESIDENT_JSON", "STREAMED_JSON"))
    ap.add_argument("--control", default=None,
                    choices=("halo", "tileindex", "dropface"),
                    help="deliberately break one property of the fold; the "
                         "verify comparison MUST then fail")
    ap.add_argument("--legs", default="both",
                    choices=("both", "resident", "streamed"))
    ap.add_argument("--truth-every", type=int, default=10)
    ap.add_argument("--poison-value", type=float, default=float("nan"),
                    help="NaN (default) exercises the nan gate; a large "
                         "FINITE value exercises w_max/CFL without tripping "
                         "the physics schemes' own input validation")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import cupy as cp

    # ARM THIS BRANCH'S FOLD.  Two branches fixed the same defect and both
    # implementations are in the tree; ``attach`` arms exactly one, and its
    # default is the other (StreamedStability, through TiledRun's observer
    # hook), because that is the arm the production routes read.  This module
    # gates TileHealthFold specifically -- ``stepper.tiled_run.health`` below
    # is that object, and its ``control`` / ``enabled`` levers are what the
    # negative controls and the price measurement need -- so it selects it
    # here rather than measuring whichever arm happened to be default.
    streaming.HEALTH_FOLD_DEFAULT = True

    cfg = obs_cfg(args.n, args.n, rung=args.rung, dx=args.dx, dt=args.dt)
    if args.n % args.tile:
        raise SystemExit(f"tile {args.tile} must divide {args.n}")
    print(f"# {args.n}x{args.n}x{cfg.nz} dx={cfg.dx/1000:g} km dt={cfg.dt:g} s "
          f"rung={args.rung!r} tile={args.tile} halo={harness.halo_radius(cfg)}")
    print(f"# radiation fires every {cfg.radt_minutes*60/cfg.dt:g} steps, "
          f"cumulus every {cfg.cudt_minutes*60/cfg.dt:g}; "
          f"this window is {args.steps} steps")
    bnd = None
    if not args.compare:
        # --compare reads two finished runs' JSON and integrates nothing, so
        # it must not spend a minute building forcing it will never use.
        print("# building the domain's specified lateral forcing "
              "(one state at a time)")
        t0 = time.perf_counter()
        bnd = sequential_boundaries(cfg)
        print(f"#   forcing built in {time.perf_counter()-t0:.1f} s")

    if args.compare:
        # The two legs are compared ACROSS PROCESSES, deliberately.  Running
        # them in one process means the resident leg's 15 GiB is still held
        # by the allocator when the streamed leg builds its own domain plus
        # tile buffers, and the streamed leg dies of OOM on a card that fits
        # either leg comfortably on its own.  Separate processes also remove
        # any chance of one leg warming a cache the other reads.
        blobs = []
        for path in args.compare:
            with open(path) as fh:
                blobs.append(json.load(fh))
        ref = blobs[0].get("resident") or blobs[0]
        got = blobs[1].get("streamed") or blobs[1]
        cmp = compare_traces(ref["trace"], got["trace"])
        print(f"\n-- COMPARE {args.compare[0]}  vs  {args.compare[1]}")
        print(_report_line("resident", ref))
        print(f"    fires={ref['fires']}")
        print(_report_line(got.get("leg", "streamed"), got))
        print(f"    fires={got['fires']}")
        print("\n-- THE GATE")
        print(_line("both legs ran every step",
                    cmp["len_ref"] == cmp["len_got"] > 0,
                    f"resident {cmp['len_ref']}, streamed {cmp['len_got']}"))
        rad = min(ref["fires"].get("radiation", 0),
                  got["fires"].get("radiation", 0))
        cu = min(ref["fires"].get("cumulus", 0), got["fires"].get("cumulus", 0))
        print(_line("radiation and cumulus FIRED on both legs",
                    rad > 0 and cu > 0,
                    f"radiation={rad}, cumulus={cu} (min over the two legs)"))
        print(_line("fires agree exactly between the legs",
                    ref["fires"] == got["fires"],
                    f"{ref['fires']} vs {got['fires']}"))
        print(_line("every substep's report is bit-identical",
                    not cmp["mismatches"],
                    f"{len(cmp['mismatches'])} mismatch(es) over "
                    f"{cmp['steps_compared']} substeps x {len(FOLD_KEYS)} "
                    f"fields"))
        for bad in cmp["mismatches"][:8]:
            print(f"      step {bad['step']} {bad['key']}: "
                  f"resident={bad['resident']!r} streamed={bad['streamed']!r}")
        moved = len({r["w_max"] for r in got["trace"]}) > 1
        print(_line("the fold is ALIVE (w_max moves, not frozen at t=0)",
                    moved,
                    f"{len({r['w_max'] for r in got['trace']})} distinct "
                    f"w_max over {len(got['trace'])} substeps; "
                    f"first={got['trace'][0]['w_max']!r}"))
        return 0

    results = {}
    if args.coverage:
        from tilestream.health_fold import fold_coverage

        options = streaming.StreamingOptions(
            mode="on", tile_nx=args.tile, tile_ny=args.tile, nbuffers=2,
            store="host")
        decision = streaming.decide(cfg, options)
        specs = streaming.tile_specs(cfg, decision)
        cov = fold_coverage(specs, nx=int(cfg.nx), ny=int(cfg.ny),
                            periodic=False)
        print(f"\n-- COVERAGE of the fold's windows over {len(specs)} tiles")
        print(_line("every mass cell folded exactly once", cov["mass_ok"],
                    f"{cov['cells']} cells, counts seen {cov['mass_counts']}"))
        print(_line("every u face folded exactly once, closing face included",
                    cov["faces_ok"],
                    f"{cov['faces']} faces, counts seen {cov['face_counts']}; "
                    f"closing face i={cfg.nx} counts "
                    f"{cov['closing_face_counts']}"))
        # And the same question of a RAGGED plan, which the ship geometry
        # never exercises: tiles that own fewer columns than tile_nx.
        from tilestream import spec as _spec

        for rn, rt in ((200, 64), (300, 128), (671, 168)):
            rspecs = _spec.plan_tiles(rn, rn, rt, rt,
                                      harness.halo_radius(cfg), False)
            rcov = fold_coverage(rspecs, nx=rn, ny=rn, periodic=False)
            print(_line(f"ragged {rn}/{rt} ({len(rspecs)} tiles) covers once",
                        rcov["mass_ok"] and rcov["faces_ok"],
                        f"mass {rcov['mass_counts']}, faces "
                        f"{rcov['face_counts']}"))
        return 0

    if args.price:
        print("\n-- PRICE: what one substep's safety observation costs")
        res = price(cfg, args.steps, args.tile, bnd)
        print(f"  BEFORE  stability_report on the resident {cfg.nx}^2 state: "
              f"{res['resident_stability_report_s']*1e3:.3f} ms/substep")
        print(f"          (under streaming this reduces a domain nobody is "
              f"integrating)")
        print(f"  AFTER   streamed step, fold ON : "
              f"{res['streamed_step_fold_on_s']*1e3:.1f} ms")
        print(f"          streamed step, fold OFF: "
              f"{res['streamed_step_fold_off_s']*1e3:.1f} ms")
        print(f"          marginal cost of the fold: "
              f"{res['fold_marginal_s']*1e3:.3f} ms/step over "
              f"{res['tiles']} tiles ({res['fold_share']*100:.2f}% of a step)")
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(res, fh, indent=2, default=str)
            print(f"\n# wrote {args.json}")
        return 0

    if args.verify:
        # A CLEAN run, both legs, no poison: does the fold report the same
        # numbers the resident reduction does, to the last bit, at every
        # substep?  If the streamed leg comes back with the t=0 value it is
        # still reading the corpse.
        print("\n-- VERIFY: resident reduction vs per-tile fold, clean run")
        res_r, state = run_resident(cfg, args.steps, boundaries=bnd,
                                    poison_step=None)
        print(_report_line("resident", res_r))
        print(f"    fires={res_r['fires']}")
        del state
        cp.get_default_memory_pool().free_all_blocks()
        res_s, stepper = run_streamed(cfg, args.steps, boundaries=bnd,
                                      tile=args.tile, poison_step=None,
                                      observer="fold", control=args.control)
        print(_report_line("streamed[fold]", res_s))
        print(f"    fires={res_s['fires']}  tiles={res_s['tiles']}")
        cmp = compare_traces(res_r["trace"], res_s["trace"])
        results = {"resident": res_r, "streamed": res_s, "compare": cmp}
        print("\n-- THE GATE")
        both_ran = cmp["len_ref"] == cmp["len_got"] == args.steps
        print(_line("both legs completed every step", both_ran,
                    f"resident {cmp['len_ref']}, streamed {cmp['len_got']}, "
                    f"asked {args.steps}"))
        rad = min(res_r["fires"].get("radiation", 0),
                  res_s["fires"].get("radiation", 0))
        cu = min(res_r["fires"].get("cumulus", 0),
                 res_s["fires"].get("cumulus", 0))
        print(_line("radiation and cumulus FIRED on both legs",
                    rad > 0 and cu > 0,
                    f"radiation={rad}, cumulus={cu} (min over the two legs)"))
        if args.control:
            steps_hit = len({d["step"] for d in cmp["mismatches"]})
            keys_hit = sorted({d["key"] for d in cmp["mismatches"]})
            print(_line(
                f"NEGATIVE CONTROL {args.control!r} FIRED",
                bool(cmp["mismatches"]),
                f"{len(cmp['mismatches'])} mismatch(es) on "
                f"{steps_hit}/{cmp['steps_compared']} substeps "
                f"(miss rate {1 - steps_hit / max(1, cmp['steps_compared']):.1%}), "
                f"fields {keys_hit}"))
        print(_line("every substep's report is bit-identical",
                    not cmp["mismatches"],
                    f"{len(cmp['mismatches'])} mismatch(es) over "
                    f"{cmp['steps_compared']} substeps x {len(FOLD_KEYS)} "
                    f"fields"))
        for bad in cmp["mismatches"][:6]:
            print(f"      step {bad['step']} {bad['key']}: "
                  f"resident={bad['resident']!r} streamed={bad['streamed']!r}")
        moved = len({r["w_max"] for r in res_s["trace"]}) > 1
        print(_line("the fold is ALIVE (w_max moves, not frozen at t=0)",
                    moved,
                    f"{len({r['w_max'] for r in res_s['trace']})} distinct "
                    f"w_max over {len(res_s['trace'])} substeps"))
        del stepper
        cp.get_default_memory_pool().free_all_blocks()
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(results, fh, indent=2, default=str)
            print(f"\n# wrote {args.json}")
        return 0

    if args.legs in ("streamed", "both"):
        print("\n-- STREAMED leg: the store is the forecast; the state is not")
        res, stepper = run_streamed(
            cfg, args.steps, boundaries=bnd, tile=args.tile,
            poison_step=(args.poison_step or None),
            observer=args.observer,
            truth_every=args.truth_every, value=args.poison_value)
        res["store_truth"] = store_truth(stepper.store)
        results["streamed"] = res
        print(_report_line(f"streamed[{args.observer}]", res))
        print(f"    tiles={res['tiles']}  fires={res['fires']}")
        print(f"    completed {res['completed']}/{args.steps} observed steps"
              + (f"; DIED: {res['crashed']}" if res["crashed"] else ""))
        print(f"    store at the end: {res['store_truth']}")
        print("\n    step | OBSERVER (reads the state) | STORE (the forecast)")
        print("         |   w_max        nan          |   w_absmax     nan")
        for row in res["trace"]:
            if "store" not in row:
                continue
            st = row["store"]
            print(f"    {row['step']:>4} | {row['w_max']:>12.6f}  "
                  f"{str(row['nan']):<5}       | {st['w_absmax']:>12.6f}  "
                  f"{str(st['nan']):<5} ({st['nonfinite_w']} cells)")
        del stepper
        cp.get_default_memory_pool().free_all_blocks()

    if args.legs in ("resident", "both"):
        print("\n-- RESIDENT leg (POSITIVE CONTROL): the identical poison, "
              "applied where this leg keeps its forecast")
        res, state = run_resident(
            cfg, args.steps, boundaries=bnd,
            poison_step=(args.poison_step or None), value=args.poison_value)
        results["resident"] = res
        print(_report_line("resident", res))
        print(f"    fires={res['fires']}")
        print(f"    completed {res['completed']}/{args.steps} observed steps"
              + (f"; DIED: {res['crashed']}" if res["crashed"] else ""))
        del state
        cp.get_default_memory_pool().free_all_blocks()

    if args.demonstrate:
        print("\n-- THE FINDING")
        st = results.get("streamed")
        rs = results.get("resident")
        if st is not None:
            wmaxes = {row["w_max"] for row in st["trace"]}
            frozen = len(wmaxes) == 1
            moved = [row for row in st["trace"] if "store" in row]
            store_moved = len({r["store"]["w_absmax"] for r in moved}) > 1
            print(_line(
                "the observer is FROZEN: one w_max value over the whole run",
                frozen,
                f"{len(wmaxes)} distinct value(s) over {len(st['trace'])} "
                f"steps, first={st['trace'][0]['w_max']:.6f}"
                if st["trace"] else "no steps"))
            print(_line(
                "the STORE meanwhile moves: the forecast is really advancing",
                store_moved,
                f"{len({r['store']['w_absmax'] for r in moved})} distinct "
                f"w_absmax over {len(moved)} sampled steps"))
            if args.poison_step:
                after = [r for r in st["trace"]
                         if "store" in r and r["step"] >= args.poison_step]
                blind = [r for r in after
                         if r["store"]["nan"] and not r["nan"]]
                print(_line(
                    "the gate is BLIND to a non-finite store",
                    bool(blind),
                    f"{len(blind)} sampled step(s) where the store held NaN "
                    f"and the gate reported nan_free"))
        if rs is not None:
            control_fired = bool(rs["raised"])
            print(_line(
                "POSITIVE CONTROL fired (resident poison raises)",
                control_fired,
                rs["raised"] or "the control did NOT fire -- this comparison "
                "is incapable of failing and proves nothing"))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"\n# wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
