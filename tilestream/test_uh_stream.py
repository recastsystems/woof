"""UP_HELI_MAX and the two tracker windows, streamed: the RESETS are the bug.

``nwp_diagnostics = 1`` folds a 9-point-smoothed column updraft helicity into
a running max EVERY step (``woof/core/uh_diag.py``), into three separate
window slots that three DIFFERENT consumers zero on three DIFFERENT rhythms:

===========================  =========================  ====================
slot                         reset by                   rhythm
===========================  =========================  ====================
``up_heli_max``              the history writer          every wrfout frame
``uh_follow_window``         the relocation runner       every evaluation
``uh_spawn_window``          the spawn-watch runner      every leg boundary
===========================  =========================  ====================

The FOLD is not the problem.  It is pointwise after a horizontal-radius-1
smooth, so it tiles like any other carrier and needs nothing the halo does not
already carry.  The problem is that every one of those three consumers resets
STATE, not the store::

    reset_up_heli_max(node.state)             woof/runtime.py:2166,2272,2709
    reset_up_heli_max(state)                  woof/prepared_single_domain_forecast.py:4019
    reset_tracker_window(node.parent.state,   woof/core/relocation_runner.py:329
                         UH_FOLLOW_WINDOW_SLOT)
    reset_tracker_window(parent_state,        woof/core/spawn_runner.py:218
                         UH_SPAWN_WINDOW_SLOT)

and under ``[tiles]`` the domain's arrays are in the STORE.  The state the
model still holds -- the one every one of those call sites reaches -- was
copied into that store at ``streaming.attach`` and has been a corpse ever
since.  So the fold lands in the store, the reset lands on the corpse, and
each window silently stops meaning "max since I last looked" and starts
meaning "max since the run began".

That is a wrong forecast with no NaN and no warning, and it biases in ONE
direction: a running max that is never cleared can only ever be too big, so
the tracker and the spawn trigger fire early and never release.  The
monotone-in-one-direction signature is what separates this defect from a
transport error, and this gate asserts it in the negative control rather than
describing it.

It is also INVISIBLE in a short run.  A run that never crosses a reset
boundary is bit-exact either way -- which is the same shape as the three false
results this project has already produced, all three of them numbers measured
in a window where the cadence under test never fired.  So this gate PRINTS the
number of resets each accumulator actually took, on BOTH sides, and refuses
its own result if that number is below three.

    python -m tilestream.test_uh_stream           # the whole gate
    python -m tilestream.test_uh_stream --quick   # the dry rung only

THE SECOND DEFECT, WHICH THE FIRST ONE HIDES
--------------------------------------------
``tilestream.physics_inventory.carrier_manifest`` -- the streaming carrier set
-- is ``woof/io/restart.py``'s manifest, deliberately, so that a field added
to woof tomorrow is streamed the day it is added.  But restart classifies a
scratch slot by whether it is CHECKPOINTED, and the tracker windows are
deliberately not: a window means "max since that consumer last looked" and a
checkpoint cannot know when the consumer will next look.  Having only
``serialize``/``rebuild`` to say that in, they were filed ``rebuild`` -- and
the transport reads ``rebuild`` as "not cross-step state", so it neither
gathers nor scatters them.  Under a host store they were never in the store at
all: each tile buffer accumulated its own window over whatever tiles that
buffer happened to serve, at that buffer's coordinates, and the domain had no
window anywhere.

The two questions -- "does this survive a checkpoint" and "is this cross-step
state a tile buffer must not lose" -- are different questions, and one
classification was answering both.  ``restart.CARRIED_SCRATCH_SLOTS`` is the
third answer; ``streamed_windows_are_carriers`` below is the control that
fails when it is taken away.
"""

from __future__ import annotations

import pathlib
import sys
import time
import warnings

import numpy as np

from woof.core import streaming
from woof.core.uh_diag import (UH_FOLLOW_WINDOW_SLOT, UH_SPAWN_WINDOW_SLOT,
                                UP_HELI_MAX_SLOT, reset_tracker_window,
                                reset_up_heli_max)
from tilestream import driver, gather, harness
from tilestream import physics_inventory as physinv
from tilestream import test_join as tj

# --------------------------------------------------------------------------
# the configuration
# --------------------------------------------------------------------------

#: test_join's own geometry: 256x192 at 12 km is a real regional footprint and
#: 32x32 tiles put 24 tiles with no true edge, 20 with one and 4 with two into
#: the same plan.  A 64-cell compute window is far below the ~500 cells a
#: TIMING would need; this is a bit-exactness gate, and a small window makes
#: the halo and the seams do more work per cell, not less.
NX, NY, NZ = 256, 192, 49
TX, TY = 32, 32

#: 24 steps, and the three cadences are deliberately COPRIME with each other
#: and with the tile count so no two consumers ever share a boundary by
#: accident: history every 6 (4 resets), follow every 5 (4 resets), spawn
#: every 7 (3 resets).  Three is the floor the module docstring commits to and
#: :func:`assert_window_actually_reset` enforces it.
NSTEPS = 24
HISTORY_EVERY = 6
FOLLOW_EVERY = 5
SPAWN_EVERY = 7
MIN_RESETS = 3

#: ``(label, slot, period)``, in the order the model would run them.
SCHEDULE: tuple[tuple[str, str, int], ...] = (
    ("history", UP_HELI_MAX_SLOT, HISTORY_EVERY),
    ("follow", UH_FOLLOW_WINDOW_SLOT, FOLLOW_EVERY),
    ("spawn", UH_SPAWN_WINDOW_SLOT, SPAWN_EVERY),
)

STORE_KEY = {label: f"scratch/{slot}" for label, slot, _ in SCHEDULE}


def uh_cfg(rung: str = "dry", nx: int = NX, ny: int = NY, nz: int = NZ,
           **over):
    """test_join's joined configuration with the NWP diagnostics lane on.

    ``nwp_diagnostics = 1`` is what allocates all three accumulators
    (``woof/core/state.py``) and what makes ``dycore.step`` call
    ``update_up_heli_max`` every step.  ``cfg.specified`` is already true in
    :func:`tilestream.test_join.join_cfg` and is REQUIRED: WRF's periodic
    ``cal_helicity`` branch is not transcribed and ``update_up_heli_max``
    raises rather than guessing.
    """
    return tj.join_cfg(nx, ny, nz, rung, nwp_diagnostics=1, **over)


# --------------------------------------------------------------------------
# reading and resetting an accumulator, on either side
# --------------------------------------------------------------------------

def _host(array) -> np.ndarray:
    import cupy as cp

    return (cp.asnumpy(array) if isinstance(array, cp.ndarray)
            else np.asarray(array))


def _read_state(state, slot):
    buf = state.existing_scratch(slot)
    return None if buf is None else _host(buf).copy()


def _read_store(store, label):
    arr = store.get(STORE_KEY[label])
    return None if arr is None else np.asarray(arr).copy()


def _model_reset(state, label, slot) -> None:
    """The reset WOOF-ArWen's own run loops call.  Nothing is faked here.

    This is the whole point of the gate: the model calls these two functions
    and only these two, so whether a streamed domain's windows reset is a
    property of the model's own API, not of a test harness that knows about
    stores.
    """
    if label == "history":
        reset_up_heli_max(state)
    else:
        reset_tracker_window(state, slot)


def _state_only_reset(state, label, slot) -> None:
    """THE NEGATIVE CONTROL: today's path, zeroing only the state copy.

    Byte-for-byte what ``reset_up_heli_max`` and ``reset_tracker_window`` did
    before this branch -- reach the scratch pool, zero the buffer, return.  On
    a resident domain that is the domain.  On a streamed one it is the corpse.
    """
    existing_scratch = getattr(state, "existing_scratch", None)
    if existing_scratch is None:
        return
    buf = existing_scratch(slot)
    if buf is not None:
        buf[...] = 0.0


# --------------------------------------------------------------------------
# the two runs
# --------------------------------------------------------------------------

class Result(dict):
    """``samples``/``final``/``resets``/``physics`` for one run."""


def _physics_fires(scalars) -> dict:
    """Radiation and cumulus call counts -- PRINTED on both sides, always.

    Three of this project's six false results were one number measured in a
    window where radiation and cumulus never fired.  A bit-exactness gate is
    not a timing, but the same trap applies: a window in which the expensive
    cadences never fire has not tested the cadence-carrying machinery either,
    so the counts are reported rather than assumed.
    """
    counts = dict(scalars.get("call_counts") or {})
    return {"radiation": int(counts.get("radiation", 0)),
            "cumulus": int(counts.get("cumulus", 0)),
            "pbl": int(counts.get("pbl", 0)),
            "microphysics_updates": int(
                scalars.get("microphysics_updates", 0))}


def resident_run(cfg, boundaries, *, nsteps=NSTEPS, seed=tj.SEED,
                 warmup=1) -> Result:
    """The control: one resident domain, WOOF-ArWen's own ``dycore.step``.

    Same loop shape as :func:`streamed_run`, with the same reset calls at the
    same instants, so the only difference between the two runs is which
    callable ``make_stepper`` returned.
    """
    import cupy as cp

    from woof.core.dycore import step as dycore_step

    state, _geo = tj.build_domain(cfg, seed=seed, boundaries=boundaries,
                                  warmup=warmup)
    stepper = streaming.make_stepper(state, cfg, streaming.OFF)
    assert stepper is dycore_step, (
        "the resident control must run the dycore's own step, not a wrapper")
    before = physinv.carrier_scalars(state)
    samples: dict[str, list] = {label: [] for label, _s, _p in SCHEDULE}
    pre_reset: dict[str, list] = {label: [] for label, _s, _p in SCHEDULE}
    post_reset: dict[str, list] = {label: [] for label, _s, _p in SCHEDULE}
    for k in range(int(nsteps)):
        stepper(state, cfg, refl_10cm_due=False)
        for label, slot, period in SCHEDULE:
            if (k + 1) % period == 0:
                samples[label].append(_read_state(state, slot))
                pre_reset[label].append(
                    float(np.nanmax(samples[label][-1])))
                _model_reset(state, label, slot)
                # The reset is OBSERVED, not assumed: the accumulator is
                # read back immediately afterwards and must be exactly
                # zero.  A control that only checked "the next peak is
                # smaller" would pass on a window that never reset at all
                # whenever the flow happened to weaken.
                post_reset[label].append(
                    float(np.nanmax(_read_state(state, slot))))
    cp.cuda.runtime.deviceSynchronize()
    after = physinv.carrier_scalars(state)
    final = {label: _read_state(state, slot) for label, slot, _p in SCHEDULE}
    carriers = {name: _host(arr) for name, arr
                in physinv.carrier_inventory(state, None).items()}
    del state
    cp.get_default_memory_pool().free_all_blocks()
    return Result(samples=samples, final=final, carriers=carriers,
                  pre_reset=pre_reset, post_reset=post_reset,
                  resets={label: len(v) for label, v in samples.items()},
                  physics=_delta_fires(before, after))


def _delta_fires(before, after) -> dict:
    a, b = _physics_fires(before), _physics_fires(after)
    return {k: b[k] - a[k] for k in b}


def _inventory_dropping_windows(obj, names=None):
    """The transport as it was: the tracker windows are not carriers.

    ``restart.classify_scratch_slot`` used to answer ``rebuild`` for both
    window slots -- the only answer it had for "cross-step state that is
    deliberately not checkpointed" -- and ``carrier_manifest`` reads
    ``rebuild`` as "not cross-step".  Reproducing it here, rather than by
    monkeypatching the classification, keeps the control accurate: the store
    genuinely never holds the windows, exactly as before.
    """
    inv = physinv.carrier_inventory(obj, names)
    return {k: v for k, v in inv.items()
            if k not in (STORE_KEY["follow"], STORE_KEY["spawn"])}


def streamed_run(cfg, boundaries, *, nsteps=NSTEPS, seed=tj.SEED, warmup=1,
                 tile_nx=TX, tile_ny=TY, nbuffers=2, halo=None,
                 reset=_model_reset, carry_windows=True,
                 report: dict | None = None) -> Result:
    """The same run, streamed one sweep per model step through the seam.

    ``reset`` and ``carry_windows`` are the two negative controls and each
    disables exactly one half of the fix.
    """
    import cupy as cp

    domain, geo = tj.build_domain(cfg, seed=seed, boundaries=boundaries,
                                  warmup=warmup)
    options = streaming.StreamingOptions(
        mode="on", tile_nx=int(tile_nx), tile_ny=int(tile_ny),
        nbuffers=int(nbuffers), halo=halo, store="host")
    decision = streaming.decide(cfg, options)
    inventory_fn = (physinv.carrier_inventory if carry_windows
                    else _inventory_dropping_windows)

    def build(state, run_cfg, dec):
        geo_inv = {k: gather.pinned_copy(v) for k, v in
                   driver.geography_inventory(domain).items()}
        per_tile = streaming.tile_boundary_tables(
            boundaries, streaming.tile_specs(run_cfg, dec), seam="zeros")
        factory = tj.tile_factory(run_cfg, per_tile[0])
        return streaming.attach(
            state, run_cfg, dec, tile_state_factory=factory,
            geography=geo_inv, boundary_tables=per_tile,
            inventory_fn=inventory_fn,
            scalars=physinv.carrier_scalars(domain), check_geography=False)

    stepper = streaming.make_stepper(domain, cfg, options, decision=decision,
                                     build=build)
    assert streaming.is_streaming(stepper)
    before = dict(stepper.scalars)
    samples: dict[str, list] = {label: [] for label, _s, _p in SCHEDULE}
    pre_reset: dict[str, list] = {label: [] for label, _s, _p in SCHEDULE}
    post_reset: dict[str, list] = {label: [] for label, _s, _p in SCHEDULE}
    for k in range(int(nsteps)):
        stepper(domain, cfg, refl_10cm_due=False)
        for label, slot, period in SCHEDULE:
            if (k + 1) % period == 0:
                got = _read_store(stepper.store, label)
                samples[label].append(got)
                pre_reset[label].append(
                    float("nan") if got is None else float(np.nanmax(got)))
                reset(domain, label, slot)
                back = _read_store(stepper.store, label)
                post_reset[label].append(
                    float("nan") if back is None else float(np.nanmax(back)))
    cp.cuda.runtime.deviceSynchronize()
    final = {label: _read_store(stepper.store, label)
             for label, _s, _p in SCHEDULE}
    carriers = {name: np.asarray(arr) for name, arr in stepper.store.items()}
    if report is not None:
        report.update(stepper.report)
        report["decision"] = stepper.decision.explain()
    out = Result(samples=samples, final=final, carriers=carriers,
                 pre_reset=pre_reset, post_reset=post_reset,
                 resets={label: len(v) for label, v in samples.items()},
                 physics=_delta_fires(before, stepper.scalars))
    domain = None  # rebound, not deleted: a closure above reads it
    del stepper
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return out


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def compare_samples(ref: Result, got: Result) -> dict:
    """Frame by frame, per accumulator, plus the monotone-bias signature.

    ``monotone`` is ``True`` when every differing sample is >= the reference
    everywhere.  That is the fingerprint of a window that never reset: a
    running max that missed a zeroing can only ever be too big.  A transport
    error moves cells in both directions and would report ``False`` here,
    which is how the two are told apart.
    """
    out: dict = {}
    for label, _slot, _p in SCHEDULE:
        a_list, b_list = ref["samples"][label], got["samples"][label]
        rows = []
        for i, (a, b) in enumerate(zip(a_list, b_list)):
            if b is None:
                rows.append({"frame": i + 1, "equal": False,
                             "missing": True, "monotone": False,
                             "ncells": 0, "max_abs": float("inf"),
                             "ref_max": float(np.nanmax(a))})
                continue
            equal = bool(np.array_equal(a, b))
            diff = b.astype(np.float64) - a.astype(np.float64)
            rows.append({
                "frame": i + 1,
                "equal": equal,
                "missing": False,
                # ">= everywhere", the one-directional signature.
                "monotone": bool(np.all(diff >= 0.0)),
                "ncells": int(np.count_nonzero(diff != 0.0)),
                "max_abs": float(np.abs(diff).max()) if diff.size else 0.0,
                "ref_max": float(np.nanmax(a)),
            })
        fa, fb = ref["final"][label], got["final"][label]
        final_equal = (fb is not None and np.array_equal(fa, fb))
        out[label] = {
            "frames": rows,
            "nframes": len(rows),
            "all_equal": all(r["equal"] for r in rows) and final_equal,
            "first_differing": next(
                (r["frame"] for r in rows if not r["equal"]), None),
            "monotone_where_differing": all(
                r["monotone"] for r in rows if not r["equal"]),
            "final_equal": bool(final_equal),
            "final_missing": fb is None,
        }
    return out


def compare_carriers(ref: Result, got: Result) -> dict:
    """Every streamed carrier, so a UH-only verdict cannot hide a transport
    regression that happens to leave UP_HELI_MAX alone."""
    a, b = ref["carriers"], got["carriers"]
    shared = sorted(set(a) & set(b))
    differing = [k for k in shared if not np.array_equal(a[k], b[k])]
    return {"ntotal": len(shared), "ndiff": len(differing),
            "differing": differing[:8],
            "only_reference": sorted(set(a) - set(b)),
            "only_streamed": sorted(set(b) - set(a))}


def assert_window_actually_reset(res: Result) -> list[str]:
    """Refuse the run's own result if the cadence under test never fired.

    Two ways this gate could pass while testing nothing, and both have
    happened to this project in other lanes:

    * fewer than :data:`MIN_RESETS` resets -- a window that never reset cannot
      show a missing reset;
    * an accumulator that is identically zero -- every comparison of two
      all-zero fields is bit-exact.
    """
    problems = []
    for label, _slot, _p in SCHEDULE:
        n = res["resets"][label]
        if n < MIN_RESETS:
            problems.append(
                f"{label}: only {n} reset(s) in the run; a window that does "
                f"not reset at least {MIN_RESETS} times cannot see this bug")
        peaks = res["pre_reset"][label]
        if not peaks or max(peaks) <= 0.0:
            problems.append(
                f"{label}: the accumulator never became non-zero "
                f"(peaks {peaks}); two all-zero fields compare bit-exact and "
                "the comparison would mean nothing")
        # Every window must have accumulated something of its own, or a
        # frame that is bit-exact is bit-exact because it is empty.
        if any(p <= 0.0 for p in peaks):
            problems.append(
                f"{label}: a window peaked at zero "
                f"(peaks {[f'{p:.4g}' for p in peaks]}); that frame's "
                "comparison is a comparison of two empty fields")
        # The reset is OBSERVED: the accumulator reads exactly zero
        # immediately after every one of them, in the RESIDENT control.
        # This is the check that makes the whole gate non-vacuous, and it
        # is an equality, not a trend.
        after = res["post_reset"][label]
        if after and max(after) != 0.0:
            problems.append(
                f"{label}: the accumulator did not read zero after a reset "
                f"in the RESIDENT control (post-reset peaks {after}); the "
                "resets under test are not reaching it even resident")
    return problems


# --------------------------------------------------------------------------
# the seam itself, before any stepping
# --------------------------------------------------------------------------

def seam_contract(nx: int = 32, ny: int = 24, nz: int = NZ) -> list[str]:
    """The reader and the writer both reach the STORE, proved by disagreement.

    Costs no sweep and needs no tiling: a state is attached to a store whose
    copy of each accumulator holds a value the state's copy does not, so any
    code path that reaches the state instead of the store gives the WRONG
    NUMBER rather than a coincidentally right one.  Three claims, and each
    fails on the pre-fix code:

    * ``reset_up_heli_max`` / ``reset_tracker_window`` zero the STORE;
    * ``storm_tracking``'s field getter -- the plane the relocation runner
      and the spawn watcher actually steer on -- returns the STORE's;
    * with the binding removed, the getter returns the state's, which is the
      control that proves the first two are not passing by accident.
    """
    import cupy as cp

    from woof.core import storm_tracking

    problems: list[str] = []
    cfg = uh_cfg("dry", nx=nx, ny=ny, nz=nz)
    state, _drv = harness.make_physics_state(
        cfg, tj.SEED, geography=harness.make_geography(
            cfg, terrain=True, periodic_faces=False))
    store = {}
    for _label, slot, _p in SCHEDULE:
        state.scratch((ny, nx), slot)[...] = 1.0
        host = np.full((ny, nx), 7.0, dtype=np.float32)
        store[f"scratch/{slot}"] = gather.pinned_copy(cp.asarray(host))
    setattr(state, streaming.STREAMED_SCRATCH_ATTR,
            {k.split("/", 1)[1]: v for k, v in store.items()})

    for slot in (UH_FOLLOW_WINDOW_SLOT, UH_SPAWN_WINDOW_SLOT):
        plane = storm_tracking._plane_from_state(state, "uh", uh_slot=slot)
        if float(plane.max()) != 7.0:
            problems.append(
                f"the tracker read {slot!r} as {plane.max()} -- the STORE "
                "holds 7.0 and the state's corpse holds 1.0, so the signal "
                "the nest steers on is the frozen copy")
    for label, slot, _p in SCHEDULE:
        _model_reset(state, label, slot)
        left = float(np.asarray(store[f"scratch/{slot}"]).max())
        if left != 0.0:
            problems.append(
                f"the reset for {slot!r} left {left} in the STORE; it "
                "zeroed the state's copy only, which is the defect")
        if float(cp.asnumpy(state.existing_scratch(slot)).max()) != 0.0:
            problems.append(
                f"the reset for {slot!r} did not zero the state's own copy; "
                "the two views must not be allowed to disagree")

    # THE CONTROL.  Take the binding away and the getter must fall back to
    # the state -- i.e. to the value a pre-fix build would always have used.
    delattr(state, streaming.STREAMED_SCRATCH_ATTR)
    state.existing_scratch(UH_FOLLOW_WINDOW_SLOT)[...] = 3.0
    plane = storm_tracking._plane_from_state(
        state, "uh", uh_slot=UH_FOLLOW_WINDOW_SLOT)
    if float(plane.max()) != 3.0:
        problems.append(
            "with no store bound the tracker did not read the state's own "
            f"plane (got {plane.max()}, expected 3.0); the resident path "
            "must be exactly what it always was")
    del state
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return problems


# --------------------------------------------------------------------------
# the checkpoint, which the carrier change could have broken
# --------------------------------------------------------------------------

def checkpoint_contract(nx: int = 64, ny: int = 48, nz: int = NZ
                        ) -> list[str]:
    """A store carries the windows; a FILE must not; a restore must zero them.

    Making the tracker windows carriers moved a set that two shipped modules
    had (correctly, until now) assumed was one set: the streamed checkpoint
    writers emit every member of the store, and woof's own reader REFUSES any
    ``scratch/`` member it does not classify ``serialize``.  So without the
    ``checkpointed_carriers`` filter this change would have produced streamed
    checkpoints that no reader accepts -- and the two gates for those modules
    run at ``nwp_diagnostics = 0``, where the filter is a no-op and would
    never have noticed.  This is the case it exists for.

    Three claims and one control:

    * the archive's members are exactly the store's minus the two windows;
    * restoring into a store whose windows are DIRTY leaves them zero, which
      is what a resident restore gets for free (a fresh state allocates them
      zeroed and ``restore_restart`` never writes them) and what a store
      restored in place has to be told;
    * every other carrier round-trips bit-exact;
    * CONTROL: write the whole store, filter defeated, and the reader must
      refuse -- otherwise the filter is decoration.
    """
    import tempfile

    import cupy as cp

    from tilestream import restart_stream
    from tilestream import test_uh_mgstream as mg

    problems: list[str] = []
    cfg = mg.mg_cfg(nx, ny, nz)          # open BCs, flat, nwp_diagnostics=1
    state, _geo = mg.build_domain(cfg)
    # Fold a few steps so the windows hold something a checkpoint could
    # wrongly carry.
    harness.run_steps(state, cfg, 3)
    store = {k: gather.pinned_copy(v) for k, v in
             physinv.carrier_inventory(state, None).items()}
    windows = {STORE_KEY["follow"], STORE_KEY["spawn"]}
    if not windows <= set(store):
        return [f"the store does not carry {sorted(windows - set(store))}; "
                "this check has nothing to test"]
    if max(float(np.asarray(store[k]).max()) for k in windows) <= 0.0:
        problems.append("both tracker windows are zero in the store, so a "
                        "checkpoint that carried them would look identical "
                        "to one that did not")
    setup = restart_stream.capture_domain_setup(state)
    scalars = physinv.carrier_scalars(state)

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "wrfrst_uh.npz"
        restart_stream.write_streamed_restart(
            path, store, cfg, scalars=scalars, setup=setup,
            template_state=state)
        with np.load(path) as archive:
            members = {name for name in archive.files
                       if not name.startswith("__")}
        carried = members & windows
        if carried:
            problems.append(
                f"the archive carries {sorted(carried)}; a tracker window "
                "means 'max since that consumer last looked' and no file "
                "knows when the consumer will next look")
        missing = (set(store) - windows) - members
        if missing:
            problems.append(
                f"the archive is missing {sorted(missing)[:6]} -- the filter "
                "dropped more than the carry class")

        # Restore into a store whose windows are DIRTY and whose carriers are
        # wrong, so a member that is not restored is visibly not restored.
        store2 = {k: gather.pinned_copy(cp.asarray(np.asarray(v)))
                  for k, v in store.items()}
        for k in store2:
            np.asarray(store2[k])[...] = 0.0
        for k in windows:
            np.asarray(store2[k])[...] = 5.0
        restart_stream.read_streamed_restart(
            path, store2, cfg, setup=setup, template_state=state)
        for k in windows:
            left = float(np.asarray(store2[k]).max())
            if left != 0.0:
                problems.append(
                    f"{k} was {left} after the restore; a restart starts the "
                    "tracker windows EMPTY, and a store restored in place "
                    "keeps whatever it held unless it is told otherwise")
        differing = sorted(k for k in set(store) - windows
                           if not np.array_equal(np.asarray(store[k]),
                                                 np.asarray(store2[k])))
        if differing:
            problems.append(
                f"{len(differing)} carriers did not round-trip: "
                f"{differing[:6]}")

        # THE CONTROL: defeat the filter and the reader must refuse.
        whole = physinv.checkpointed_carriers
        physinv.checkpointed_carriers = lambda arrays: dict(arrays)
        try:
            bad = pathlib.Path(tmp) / "wrfrst_uh_unfiltered.npz"
            restart_stream.write_streamed_restart(
                bad, store, cfg, scalars=scalars, setup=setup,
                template_state=state)
            refused = False
            try:
                restart_stream.read_streamed_restart(
                    bad, store2, cfg, setup=setup, template_state=state)
            except Exception:                             # noqa: BLE001
                refused = True
            if not refused:
                problems.append(
                    "a checkpoint carrying the tracker windows was ACCEPTED; "
                    "the filter is then decoration, not a fix")
        finally:
            physinv.checkpointed_carriers = whole

    del state, store
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return problems


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

RUNGS = ("dry", "full(real74)+KF", "full fast cadence")


def _line(label: str, ok: bool, detail: str = "") -> str:
    return f"  {'PASS' if ok else 'FAIL':4s}  {label:50s} {detail}"


def _fires(res: Result) -> str:
    p = res["physics"]
    return (f"rad={p['radiation']} cu={p['cumulus']} pbl={p['pbl']} "
            f"mp={p['microphysics_updates']}")


def _samples_line(label: str, cmp_row: dict) -> str:
    rows = cmp_row["frames"]
    return (f"{cmp_row['nframes']} frames, "
            f"first differing={cmp_row['first_differing']}, "
            f"final_equal={cmp_row['final_equal']}, "
            f"peak_ref={max((r['ref_max'] for r in rows), default=0.0):.6g}")


def run_rung(rung: str, failures: list[str], *, nsteps=NSTEPS,
             shape=(NX, NY), tile=(TX, TY)) -> None:
    import cupy as cp

    cfg = uh_cfg(rung, nx=shape[0], ny=shape[1])
    tile_nx, tile_ny = tile
    print()
    print(f"-- RUNG {rung}  (dt={cfg.dt:g} s, {nsteps} steps, "
          f"nwp_diagnostics={cfg.nwp_diagnostics})")
    t0 = time.perf_counter()
    bnd_a, _ = tj.build_domain(cfg, seed=tj.SEED, warmup=0)
    bnd_b, _ = tj.build_domain(cfg, seed=tj.SEED + 1, warmup=0)
    bnd = tj.domain_boundaries(cfg, bnd_a, bnd_b)
    del bnd_a, bnd_b
    cp.get_default_memory_pool().free_all_blocks()

    ref = resident_run(cfg, bnd, nsteps=nsteps)
    problems = assert_window_actually_reset(ref)
    for p in problems:
        failures.append(f"{rung}: RESIDENT CONTROL VACUOUS -- {p}")
        print(_line("resident control is non-vacuous", False, p))
    if not problems:
        print(_line("resident control is non-vacuous", True,
                    "resets " + ", ".join(
                        f"{k}={v}" for k, v in ref["resets"].items())
                    + f"; peaks " + ", ".join(
                        f"{k}={max(v):.4g}"
                        for k, v in ref["pre_reset"].items())))
    print(f"        resident physics fires: {_fires(ref)}")
    print("        resident post-reset peaks (must be 0): "
          + ", ".join(f"{k}={max(v):g}"
                      for k, v in ref["post_reset"].items() if v))

    report: dict = {}
    got = streamed_run(cfg, bnd, nsteps=nsteps, tile_nx=tile_nx,
                       tile_ny=tile_ny, report=report)
    print(f"        streamed physics fires: {_fires(got)}   "
          f"({report.get('tiles')} tiles, halo {report.get('halo')})")
    print("        streamed post-reset peaks (must be 0): "
          + ", ".join(f"{k}={max(v):g}"
                      for k, v in got["post_reset"].items() if v))
    if ref["physics"] != got["physics"]:
        failures.append(f"{rung}: the two runs fired different physics "
                        f"{ref['physics']} vs {got['physics']}")
    if ref["resets"] != got["resets"]:
        failures.append(f"{rung}: the two runs took different reset counts "
                        f"{ref['resets']} vs {got['resets']}")

    cmp = compare_samples(ref, got)
    for label, _slot, _p in SCHEDULE:
        row = cmp[label]
        ok = row["all_equal"]
        if not ok:
            failures.append(f"{rung}: {label} window is not bit-exact")
        print(_line(f"{label:8s} bit-exact, frame by frame", ok,
                    _samples_line(label, row)))

    carriers = compare_carriers(ref, got)
    ok = carriers["ndiff"] == 0 and not carriers["only_reference"]
    if not ok:
        failures.append(f"{rung}: {carriers['ndiff']} carriers differ "
                        f"{carriers['differing']}")
    print(_line("every other carrier bit-exact too", ok,
                f"{carriers['ntotal']} carriers, ndiff={carriers['ndiff']}, "
                f"missing from store={carriers['only_reference']}"))
    print(f"        {time.perf_counter() - t0:.1f} s")

    # ---------------------------------------------------------- the negatives
    print("   negative controls (each MUST differ, and HOW it differs "
          "is the diagnosis)")
    broken = streamed_run(cfg, bnd, nsteps=nsteps, tile_nx=tile_nx,
                          tile_ny=tile_ny, reset=_state_only_reset)
    cmpb = compare_samples(ref, broken)
    for label, _slot, _p in SCHEDULE:
        row = cmpb[label]
        first = row["first_differing"]
        monotone = row["monotone_where_differing"]
        # A differing FINAL value is not enough: the claim is that the
        # published frames are wrong, so the control has to move a frame.
        differs = first is not None
        # Frame 1 closes before the first reset, so it CANNOT carry the
        # defect.  Requiring that it is clean is what proves the divergence
        # is the missed reset and not something the run does from step 1.
        clean_first = row["frames"][0]["equal"] if row["frames"] else False
        ok = differs and monotone and clean_first and first >= 2
        if not differs:
            failures.append(
                f"{rung}: control 'reset the state copy only' did not fire "
                f"for {label}; the gate cannot see the defect it exists for")
        elif not clean_first or first < 2:
            failures.append(
                f"{rung}: control 'reset the state copy only' moved {label} "
                f"at frame {first}; frame 1 closes before the first reset "
                "and must be identical, so a frame-1 difference means the "
                "control is reproducing something other than a missed reset")
        elif not monotone:
            failures.append(
                f"{rung}: control 'reset the state copy only' fired for "
                f"{label} but NOT monotonically; streamed >= resident "
                "everywhere is the signature that identifies a missed reset "
                "rather than a transport error, and it is absent")
        print(_line(f"state-only reset -> {label} diverges, monotone", ok,
                    f"frame 1 clean={clean_first}, first differing "
                    f"frame={first}, monotone={monotone}, "
                    f"max|d|={max((r['max_abs'] for r in row['frames']), default=0.0):.6g}"))
    del broken

    nocarry = streamed_run(cfg, bnd, nsteps=nsteps, tile_nx=tile_nx,
                           tile_ny=tile_ny, carry_windows=False)
    cmpc = compare_samples(ref, nocarry)
    for label in ("follow", "spawn"):
        row = cmpc[label]
        differs = not row["all_equal"]
        if not differs:
            failures.append(
                f"{rung}: control 'windows are not carriers' did not fire "
                f"for {label}")
        print(_line(f"windows not carried -> {label} diverges", differs,
                    f"absent from the store={row['final_missing']}, "
                    f"first differing frame={row['first_differing']}"))
    hist = cmpc["history"]
    if not hist["all_equal"]:
        failures.append(
            f"{rung}: dropping the tracker windows from the inventory also "
            "moved UP_HELI_MAX; the control is not isolating what it claims")
    print(_line("windows not carried -> UP_HELI_MAX unaffected",
                hist["all_equal"],
                "the two defects are independent"))
    del nocarry, ref, got
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def main(argv=None) -> int:
    import cupy as cp

    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    small = "--small" in argv
    rungs = ("dry",) if (quick or small) else RUNGS
    nsteps = NSTEPS
    # --small is a SMOKE geometry for the gate's own plumbing, never a
    # result: 96x80 split 16x16 keeps every property that matters here
    # (tiles with no true edge, tiles with two, an exact tiling) at a
    # fraction of the wall time.  Every number quoted anywhere comes from
    # the full geometry.
    shape = (96, 80) if small else (NX, NY)
    tile = (16, 16) if small else (TX, TY)

    free, total = cp.cuda.runtime.memGetInfo()
    print(f"cupy {cp.__version__}  "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  "
          f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f}")
    cfg0 = uh_cfg("dry", nx=shape[0], ny=shape[1])
    halo = harness.halo_radius(cfg0)
    print("=" * 78)
    print("UP_HELI_MAX AND THE TWO TRACKER WINDOWS, STREAMED")
    print(f"  {shape[0]}x{shape[1]}x{NZ} at dx=12 km, specified LBC + real "
          f"real terrain,")
    print(f"  tile {tile[0]}x{tile[1]}, halo {halo} = harness.halo_radius, "
          f"N={nsteps} steps")
    print(f"  resets: history every {HISTORY_EVERY}, follow every "
          f"{FOLLOW_EVERY}, spawn every {SPAWN_EVERY} steps")
    print("=" * 78)

    failures: list[str] = []

    print()
    print("-- THE SEAM: both the reset and the tracker's read reach the "
          "STORE")
    seam = seam_contract()
    for problem in seam:
        failures.append(f"seam: {problem}")
        print(_line("store-backed read/write seam", False, problem))
    if not seam:
        print(_line("store-backed read/write seam", True,
                    "store holds 7.0 and the state's corpse 1.0; the "
                    "tracker read 7.0, both resets zeroed the store, and "
                    "with the binding removed the read fell back to the "
                    "state"))

    untested: list[str] = []
    print()
    print("-- THE CHECKPOINT: the store carries the windows, the FILE must "
          "not")
    ck = checkpoint_contract()
    for problem in ck:
        failures.append(f"checkpoint: {problem}")
        print(_line("streamed checkpoint excludes the carry class", False,
                    problem))
    if not ck:
        print(_line("streamed checkpoint excludes the carry class", True,
                    "archive = store minus the two windows; a restore zeroes "
                    "them and round-trips everything else; a deliberately "
                    "unfiltered archive is refused"))

    for rung in rungs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                run_rung(rung, failures, nsteps=nsteps, shape=shape,
                         tile=tile)
        except cp.cuda.memory.OutOfMemoryError as exc:
            # These cards are shared.  An OOM is a card that filled up under
            # another process, not a result -- but it is also not a pass, so
            # it is reported as UNTESTED in its own section rather than
            # swallowed into the failure list or, worse, into silence.
            untested.append(f"{rung}: {exc}")
            print(_line(f"{rung}", False,
                        "NOT TESTED -- the card ran out of memory "
                        "mid-rung (shared GPU), so this rung has no result"))
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    print()
    print("=" * 78)
    if untested:
        print(f"NOT TESTED -- {len(untested)} rung(s) never ran:")
        for u in untested:
            print(f"  ? {u}")
        print()
    if failures:
        print(f"UH STREAMING GATE FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  * {f}")
        return 1
    if untested:
        print(f"UH STREAMING GATE INCOMPLETE -- every rung that RAN passed, "
              f"but {len(untested)} never ran; that is not a pass.")
        return 2
    print("UH STREAMING GATE PASSED -- all three accumulators stream "
          "bit-exact across their resets, and both negative controls fired.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
