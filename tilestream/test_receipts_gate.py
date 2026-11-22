"""The receipts gate: a domain-wide FP64 sum, taken from a domain that is
not on the card.

``dycore.domain_mass_measure`` is an FP64 area-weighted sum of ``mu/msft**2``
over the whole domain and it is the number WOOF-ArWen's conservation closure is
built on.  It is a RECEIPT, not a carrier -- nothing in the forecast reads it
-- which is exactly why it needs a gate of its own: a carrier that goes wrong
changes the answer and the bit-exactness matrix catches it, and a receipt that
goes wrong changes nothing except what the operator believes.

WHAT THIS GATE ASSERTS, IN TWO PARTS THAT ARE DELIBERATELY NOT THE SAME CLAIM
-----------------------------------------------------------------------------
(a) The receipt taken from the pinned store in the MONOLITHIC traversal order
    is BIT-IDENTICAL to the resident run's.  Asserted.

(b) The same receipt FOLDED per tile is reported with its exact ulp distance
    from (a) and is NOT asserted equal.  Published as data, because it is
    data: floating-point summation is not associative and 16 partial sums
    rounded then added is a different rounding from one sum of 49152 terms.

The two rows of (b) are the reason this gate runs two configurations rather
than one.  On the milestone-one periodic domain the fold is 0 ulp -- and that
zero is a TRAP.  ``msft == 1`` exactly there, so the area weight is exactly
1.0, every term is an FP32 column mass (a multiple of 2**-7 near 65000), and
the 49152-term sum needs 39 of FP64's 53 bits.  It is EXACT, so no
association can move it.  A gate that ran only that configuration would
publish "0 ulp, folding is fine" and be wrong the first time anybody pointed
it at a real projection.  The mapped case is where the question has an
answer.

THE CONTROLS
------------
Every one of these fails when the thing it guards is broken, and each was
observed FIRING before its fix:

* the receipt read from ``state`` on the streamed leg must be GROSSLY wrong
  -- it is the t=0 mass, because a streamed domain's state is never stepped
  again.  MEASURED 376832 ulp from the truth after 24 steps;
* the receipt folded over GATHERED windows (what a tile-local observer that
  simply called ``domain_mass_measure(tile_state)`` returns) must come back
  at the plan's redundancy, 2.5x;
* ``step_kwargs`` must reach ``dycore.step``.  ``TiledRun.sweep`` documented
  that it did and did not do it, so every keyword WOOF-ArWen's run loop hands a
  streamed domain was silently unset.  The control sweeps with
  ``acoustic=False`` and demands a DIFFERENT answer from the default sweep;
* ``mass_flux_observer`` and ``mass_flux_accumulator`` must be REFUSED by a
  sweep, because a tile can only offer its own window's faces;
* ``tilestream.receipts.tke_budget`` must REFUSE, and the naive
  sum-over-buffers fold is printed beside the resident answer so the size of
  the lie is on the record.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import traceback
import warnings

import numpy as np

from tilestream import harness

NX, NY, NZ = 256, 192, 49
TX, TY = 64, 48
NSTEPS = 24
SEED = harness.DEFAULT_SEED


def ulp_distance(a: float, b: float) -> int:
    """Signed distance in representable float64 steps between ``a`` and ``b``.

    Reported instead of a tolerance on purpose.  "Within 1e-9" hides whether
    two numbers are the same number; "1 ulp apart" says exactly how far a
    different association moved the sum, and a reader can decide.
    """
    def key(x: float) -> int:
        bits = struct.unpack("<q", struct.pack("<d", x))[0]
        return bits if bits >= 0 else (1 << 63) - bits
    return key(b) - key(a)


def _host(array) -> np.ndarray:
    return np.asarray(array.get() if hasattr(array, "get") else array)


def digest_of(arrays) -> str:
    """SHA-256 over a store, in sorted key order."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        host = np.ascontiguousarray(_host(arrays[name]))
        digest.update(name.encode() + host.dtype.str.encode()
                      + np.asarray(host.shape, np.int64).tobytes()
                      + host.tobytes(order="C"))
    return digest.hexdigest()


def fold_measure(mup, mub2d, msft, windows) -> float:
    """The same integral summed window by window, in ``windows`` order."""
    from woof.core.dycore import domain_mass_measure

    from tilestream.receipts import StoreDomainView

    total = 0.0
    for (j0, j1, i0, i1) in windows:
        sub_mub = mub2d[j0:j1, i0:i1] if np.ndim(mub2d) == 2 else mub2d
        total += domain_mass_measure(
            StoreDomainView(mup[j0:j1, i0:i1], sub_mub, msft[j0:j1, i0:i1]))
    return float(total)


# --------------------------------------------------------------------------
# the two configurations
# --------------------------------------------------------------------------

def build_flat(nsteps: int):
    """Milestone-one periodic domain: identity map factors, flat terrain."""
    cfg = harness.make_config(NX, NY, NZ, km_opt=2, tke_budget=1)
    state = harness.make_state(cfg, seed=SEED)
    return cfg, state, {}


def build_mapped(nsteps: int):
    """Real Lambert conformal grid, real terrain-following base state.

    The configuration the reduction-order question has an answer on: ``msft``
    spans 0.9848 to 1.0051, so ``1/msft**2`` carries a full FP64 mantissa and
    the domain sum genuinely rounds.
    """
    from tilestream import driver

    cfg = harness.make_config(NX, NY, NZ, km_opt=2, tke_budget=1,
                              **harness.GEOGRAPHY_OVERRIDES)
    geo = harness.make_geography(cfg, periodic_faces=True)
    state, _drv = harness.make_physics_state(cfg, SEED, geography=geo)
    return cfg, state, {"geography": driver.geography_inventory(state)}


CONFIGURATIONS = (
    ("periodic, map_proj=0, flat terrain", build_flat),
    ("periodic, real Lambert + real terrain", build_mapped),
)


def receipt_case(label, build, nsteps: int) -> dict:
    """One configuration, both legs, every receipt and every control."""
    import cupy as cp

    from woof.core import dycore, tke_budget
    from tilestream import driver, gather, receipts
    from tilestream import physics_inventory as physinv
    from tilestream import spec as tspec

    cfg, state, extra = build(nsteps)
    halo = harness.halo_radius(cfg)
    mapped = bool(extra)
    inventory = (physinv.carrier_inventory if mapped
                 else harness.state_arrays)

    start = {k: _host(v).copy() for k, v in inventory(state).items()}
    mub2d, msft = _host(state.mub2d).copy(), _host(state.msft).copy()

    measure_start = dycore.domain_mass_measure(state)
    harness.run_steps(state, cfg, nsteps)
    measure_end = dycore.domain_mass_measure(state)
    resident_digest = digest_of(inventory(state))
    resident_tke = tke_budget.drain(state, cfg)
    del state
    cp.get_default_memory_pool().free_all_blocks()

    # The state a model still holds after handing its carriers to the
    # transport: prepared, copied out, and never stepped again.
    stale_cfg, stale_state, stale_extra = build(nsteps)

    specs = tspec.plan_tiles(NX, NY, TX, TY, halo, True)
    tspec.validate_plan(specs, NY, NX)
    store = {k: gather.pinned_copy(v) for k, v in start.items()}
    kwargs: dict = {}
    if mapped:
        geo_store = {k: gather.pinned_copy(_host(v))
                     for k, v in stale_extra["geography"].items()}
        kwargs = driver.geography_run_kwargs(
            cfg, None, geography=geo_store,
            geography_fn=harness.neutral_geography)
        kwargs["scalars"] = physinv.carrier_scalars(stale_state)

    report: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run = driver.TiledRun(store, cfg, TX, TY, halo, 2, periodic=True,
                              write_mode="ring", **kwargs)
        run.sweep(nsteps, report=report)
    cp.cuda.runtime.deviceSynchronize()

    streamed_digest = digest_of(store)
    mup = _host(receipts.column_mass(store))

    # (a) from the store, in the monolithic traversal order
    store_measure = receipts.domain_mass_measure(store, stale_state)

    # (b) folded per tile INTERIOR, in tile order and reversed
    interiors = [(s.j0, s.j1, s.i0, s.i1) for s in specs]
    covered = np.zeros((NY, NX), np.int32)
    for (j0, j1, i0, i1) in interiors:
        covered[j0:j1, i0:i1] += 1
    folded = fold_measure(mup, mub2d, msft, interiors)
    reversed_ = fold_measure(mup, mub2d, msft, list(reversed(interiors)))
    # A 16-way fold is a weak probe of associativity: with only 16 partial
    # sums the extra rounding is one or two ulp and its SIGN wanders with the
    # step count, so a single number from it invites the reader to conclude
    # "0 ulp, folding is fine".  The row fold -- NY partial sums, the finest
    # decomposition the same values admit -- is the same question asked
    # loudly, and it is here so the policy rests on a trend rather than on
    # one draw.
    survey = {}
    for ty, tx in ((2, 2), (4, 4), (8, 8), (16, 16), (NY, 1), (1, NX)):
        hy, hx = NY // ty, NX // tx
        windows = [(j * hy, (j + 1) * hy, i * hx, (i + 1) * hx)
                   for j in range(ty) for i in range(tx)]
        value = fold_measure(mup, mub2d, msft, windows)
        survey[f"{ty}x{tx}"] = {
            "windows": len(windows),
            "ulp": ulp_distance(store_measure, value),
            "rel": abs(value - store_measure) / store_measure}
    rows_fold = survey[f"{NY}x1"]

    # control: folded over GATHERED windows -- what a tile-local observer
    # calling domain_mass_measure(tile_state) actually returns.
    gathered_total = 0.0
    for s in specs:
        rows = np.arange(s.cj0, s.cj0 + s.cny) % NY
        cols = np.arange(s.ci0, s.ci0 + s.cnx) % NX
        sub_mub = (mub2d[np.ix_(rows, cols)] if np.ndim(mub2d) == 2
                   else mub2d)
        gathered_total += dycore.domain_mass_measure(
            receipts.StoreDomainView(mup[np.ix_(rows, cols)], sub_mub,
                                     msft[np.ix_(rows, cols)]))

    # control: the receipt read from the stale state
    stale_measure = dycore.domain_mass_measure(stale_state)

    record = {
        "label": label,
        "halo": halo, "tiles": len(specs), "steps": nsteps,
        "msft_min": float(msft.min()), "msft_max": float(msft.max()),
        "msft_identity": bool(np.all(msft == 1.0)),
        "carriers": len(start),
        "bitexact": streamed_digest == resident_digest,
        "resident_digest": resident_digest,
        "streamed_digest": streamed_digest,
        "measure_start": measure_start, "measure_end": measure_end,
        "delta": measure_end - measure_start,
        "store_measure": store_measure,
        "store_ulp": ulp_distance(measure_end, store_measure),
        "store_bit_identical": measure_end.hex() == store_measure.hex(),
        "interior_partition_exact": bool(np.all(covered == 1)),
        "folded": folded,
        "folded_ulp": ulp_distance(store_measure, folded),
        "folded_abs": abs(folded - store_measure),
        "folded_rel": abs(folded - store_measure) / store_measure,
        "reversed_ulp": ulp_distance(store_measure, reversed_),
        "fold_spread_ulp": abs(ulp_distance(folded, reversed_)),
        "fold_survey": survey,
        "row_fold_ulp": rows_fold["ulp"],
        "row_fold_rel": rows_fold["rel"],
        "row_fold_windows": NY,
        "fold_survey_max_abs_ulp": max(abs(v["ulp"]) for v in survey.values()),
        "gathered_fold": gathered_total,
        "gathered_ratio": gathered_total / store_measure,
        "redundancy": report["efficiency"]["redundancy"],
        "stale_measure": stale_measure,
        "stale_is_t0": stale_measure.hex() == measure_start.hex(),
        "stale_ulp": ulp_distance(measure_end, stale_measure),
        "resident_tke_residual_rel": (None if resident_tke is None
                                      else resident_tke["residual_rel"]),
        "resident_tke_steps": (None if resident_tke is None
                               else resident_tke["steps"]),
    }

    # the TKE budget, where a streamed caller would look for it
    record["streamed_tke_from_state"] = tke_budget.drain(stale_state, cfg)
    buffers = []
    for tile in run.tiles:
        drained = tke_budget.drain(tile, run.tile_cfg, reset_window=False)
        steps = float(_host(tile.scratch((1,), "tke_budget_steps",
                                         dtype=np.float64))[0])
        buffers.append({"steps": steps,
                        "volume": None if drained is None
                        else drained["volume"]})
    record["tke_buffers"] = buffers
    if resident_tke is not None and all(b["volume"] for b in buffers):
        record["tke_naive_fold_ratio"] = {
            name: (sum(b["volume"][name] for b in buffers)
                   / resident_tke["volume"][name])
            for name in resident_tke["volume"]
            if resident_tke["volume"][name]}

    del run, stale_state
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return record


# --------------------------------------------------------------------------
# contracts
# --------------------------------------------------------------------------

def contract_step_kwargs_reach_the_tiles(nsteps: int = 2) -> str:
    """``sweep(step_kwargs=...)`` must actually reach ``dycore.step``.

    THE control for the defect this gate was written around.  ``_sweep``
    built the keyword dict and then called ``step(tile, tile_cfg)`` without
    it, so every keyword WOOF-ArWen's run loop hands a streamed domain -- the
    history step's ``refl_10cm_due`` above all -- was silently unset, and
    nothing anywhere said so.

    ``acoustic=False`` is the probe because it is a whole different
    integration (the Phase-1 advection-only path, w/phi'/mu' frozen), so a
    sweep that received it CANNOT agree with one that did not.  The gate
    demands both halves: the forwarded sweep must differ from the default
    sweep, and it must equal the MONOLITHIC advection-only answer -- the
    first half alone would pass on any corruption, the second pins the value.
    """
    import cupy as cp

    from woof.core import dycore
    from tilestream import driver, gather

    cfg = harness.make_config(128, 96, NZ)
    halo = harness.halo_radius(cfg)
    state = harness.make_state(cfg, seed=SEED)
    # ``acoustic=False`` refuses a state with w != 0 (advection.py:152: its
    # vertical flux is only the eta mass flux Omega at rest), so the probe
    # state starts from rest in w.  The advection-only path then freezes w,
    # phi' and mu', which is what makes the two sweeps disagree.
    state.w[...] = 0.0
    start = {k: _host(v).copy() for k, v in harness.state_arrays(state).items()}
    for _ in range(nsteps):
        dycore.step(state, cfg, acoustic=False)
    cp.cuda.runtime.deviceSynchronize()
    advective = digest_of(harness.state_arrays(state))
    del state
    cp.get_default_memory_pool().free_all_blocks()

    digests = {}
    for name, kw in (("default", None), ("acoustic=False", {"acoustic": False})):
        store = {k: gather.pinned_copy(v) for k, v in start.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            driver.TiledRun(store, cfg, 64, 48, halo, 2, periodic=True,
                            write_mode="ring").sweep(nsteps, step_kwargs=kw)
        cp.cuda.runtime.deviceSynchronize()
        digests[name] = digest_of(store)
        del store
        cp.get_default_pinned_memory_pool().free_all_blocks()

    if digests["default"] == digests["acoustic=False"]:
        raise AssertionError(
            "a sweep with step_kwargs={'acoustic': False} produced the SAME "
            "digest as one without it: step_kwargs is not reaching "
            "dycore.step, so every keyword a streamed domain is handed is "
            "silently unset")
    if digests["acoustic=False"] != advective:
        raise AssertionError(
            "the forwarded sweep differs from the default one but does NOT "
            "match the monolithic advection-only run, so something reached "
            "the tiles and it was not what the caller passed: tiled "
            f"{digests['acoustic=False'][:16]} vs monolithic "
            f"{advective[:16]}")
    return (f"step_kwargs reaches every tile: acoustic=False sweep "
            f"{digests['acoustic=False'][:16]} == monolithic advection-only, "
            f"and != the default sweep {digests['default'][:16]}")


def contract_domain_scope_observers_are_refused() -> list[str]:
    """A sweep must refuse the two receipts a tile cannot accurately serve."""
    import cupy as cp

    from woof.core.dycore import MassFluxAccumulator
    from tilestream import driver, gather

    cfg = harness.make_config(128, 96, NZ, periodic=False, open_x=True,
                              open_y=True, map_proj=0)
    halo = harness.halo_radius(cfg)
    state = harness.make_state(cfg, seed=SEED)
    store = {k: gather.pinned_copy(_host(v))
             for k, v in harness.state_arrays(state).items()}
    del state
    cp.get_default_memory_pool().free_all_blocks()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run = driver.TiledRun(store, cfg, 64, 48, halo, 2, periodic=False,
                              write_mode="ring")
    lines = []
    for name, value in (("mass_flux_observer", [].append),
                        ("mass_flux_accumulator", MassFluxAccumulator())):
        try:
            run.sweep(1, step_kwargs={name: value})
        except driver.TiledRunError as exc:
            lines.append(f"{name} refused: {str(exc).splitlines()[0][:70]}...")
        else:
            raise AssertionError(
                f"a sweep ACCEPTED {name}: every tile would report its own "
                "compute-window faces as domain edges and the receipt would "
                "be a plausible wrong number")
    del run, store
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return lines


def contract_receipts_refuse_what_they_cannot_take() -> list[str]:
    """The refusals in :mod:`tilestream.receipts`, exercised."""
    from tilestream import receipts

    lines = []
    periodic = harness.make_config(64, 48, NZ, km_opt=2, tke_budget=1)
    terms = receipts.mass_budget_terms(None, periodic)
    if set(terms) != {"lateral_flux_integral", "lbc_mass_forcing_integral",
                      "specified_zone_mass_reset"} or any(terms.values()):
        raise AssertionError(f"periodic budget terms are not all zero: {terms}")
    lines.append("mass_budget_terms on a periodic domain: all three terms "
                 "measured zero (no lateral boundary exists)")

    forced = harness.make_config(64, 48, NZ, periodic=False, open_x=True,
                                 map_proj=0)
    try:
        receipts.mass_budget_terms(None, forced)
    except receipts.ReceiptRefused:
        lines.append("mass_budget_terms on an open-boundary domain: refused "
                     "rather than defaulted to zero")
    else:
        raise AssertionError(
            "mass_budget_terms returned a flux-only closure on a forced "
            "domain, which is the exact false receipt conservation_closure "
            "refuses to write")

    off = harness.make_config(64, 48, NZ)
    if receipts.tke_budget(None, off) is not None:
        raise AssertionError("tke_budget must be None when the budget is off")
    try:
        receipts.tke_budget(None, periodic)
    except receipts.ReceiptRefused:
        lines.append("tke_budget with cfg.tke_budget=1: refused (a 3-D "
                     "volume integral accumulated on the tile buffer)")
    else:
        raise AssertionError(
            "tke_budget returned a value for a streamed domain; the "
            "accumulator is not in the store")
    return lines


def contract_streamed_domain_seam_returns_the_number(nsteps: int) -> str:
    """``StreamedDomain.domain_mass_measure()`` is the discoverable seam.

    The gate above drives :mod:`tilestream.receipts` directly, which proves
    the arithmetic and proves nothing about whether a user of the MODE can
    reach it.  ``[tiles] mode = "on"`` hands a route a
    :class:`woof.core.streaming.StreamedDomain`, and the obvious call on the
    state it still holds is the wrong one, so the right one has to be on the
    object the route already has.  Exercised here through
    ``streaming.attach`` -- the same path a route takes -- against the
    resident answer.
    """
    import cupy as cp

    from woof.core import dycore, streaming
    from tilestream import driver

    cfg = harness.make_config(128, 96, NZ)
    halo = harness.halo_radius(cfg)
    state = harness.make_state(cfg, seed=SEED)
    resident = harness.make_state(cfg, seed=SEED)
    harness.run_steps(resident, cfg, nsteps)
    truth = dycore.domain_mass_measure(resident)
    del resident
    cp.get_default_memory_pool().free_all_blocks()

    decision = streaming.StreamingDecision(
        True, "gate", tile_nx=64, tile_ny=48, nbuffers=2, halo=halo,
        store="host")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        streamed = streaming.attach(
            state, cfg, decision,
            tile_state_factory=driver.make_tile_state,
            inventory_fn=None, scalars=None, check_geography=False)
        for _ in range(nsteps):
            streamed(state, cfg)
    cp.cuda.runtime.deviceSynchronize()

    seam = streamed.domain_mass_measure()
    stale = dycore.domain_mass_measure(state)
    if seam.hex() != truth.hex():
        raise AssertionError(
            f"StreamedDomain.domain_mass_measure() returned {seam!r}, "
            f"{ulp_distance(truth, seam)} ulp from the resident {truth!r}")
    if stale.hex() == truth.hex():
        raise AssertionError(
            "the CONTROL did not fire: reading the receipt off the state "
            "the streamed domain was attached to gave the right answer, so "
            "this configuration cannot tell the two apart and proves "
            "nothing")
    del streamed, state
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return (f"StreamedDomain.domain_mass_measure() == resident ({seam!r}); "
            f"the same read off .state is {ulp_distance(truth, stale)} ulp "
            "away")


def contract_multigpu_receipt_is_the_same_number(nsteps: int) -> list[str]:
    """The receipt must not know how many cards stepped the tiles.

    :mod:`tilestream.mgstream` hands G GPUs their own share of the tiles of
    ONE pinned host domain, so the store is still one store and the receipt
    still reads THE STORE -- the number cannot depend on the partition.  That
    is an argument, and arguments are what this project has been wrong about,
    so it is measured on three legs: one GPU, two workers on one GPU (which
    separates "the partition is wrong" from "two physical cards interfere"),
    and two workers on two GPUs.

    Skipped, loudly, where :mod:`tilestream.mgstream` is not in the tree or
    only one device is visible -- a skipped leg is reported, never silently
    counted as a pass.
    """
    import cupy as cp

    from woof.core import dycore
    from tilestream import driver, gather, receipts

    try:
        from tilestream import mgstream
    except ImportError:
        return ["SKIPPED multi-GPU: tilestream.mgstream is not in this tree"]

    cfg = harness.make_config(128, 96, NZ, km_opt=2, tke_budget=1)
    halo = harness.halo_radius(cfg)
    state = harness.make_state(cfg, seed=SEED)
    start = {k: _host(v).copy()
             for k, v in harness.state_arrays(state).items()}
    harness.run_steps(state, cfg, nsteps)
    truth = dycore.domain_mass_measure(state)
    reference = {k: _host(v).copy()
                 for k, v in harness.state_arrays(state).items()}
    geo_state = harness.make_state(cfg, seed=SEED)   # geography is INPUT
    del state
    cp.get_default_memory_pool().free_all_blocks()

    ndev = cp.cuda.runtime.getDeviceCount()
    legs = [("1 GPU, run_tiled",
             lambda s: driver.run_tiled(s, cfg, 64, 48, halo=halo,
                                        nsteps=nsteps, nbuffers=2,
                                        write_mode="shadow")),
            ("2 workers, 1 GPU",
             lambda s: mgstream.run_mgstream(s, cfg, 64, 48, halo, nsteps,
                                             devices=(0, 0), nbuffers=2,
                                             write_mode="shadow"))]
    if ndev >= 2:
        legs.append(("2 workers, 2 GPUs",
                     lambda s: mgstream.run_mgstream(s, cfg, 64, 48, halo,
                                                     nsteps, devices=(0, 1),
                                                     nbuffers=2,
                                                     write_mode="shadow")))
    lines = []
    for label, runner in legs:
        store = {k: gather.pinned_copy(v) for k, v in start.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            runner(store)
        cp.cuda.runtime.deviceSynchronize()
        differing = sorted(k for k in reference
                           if np.asarray(store[k]).tobytes()
                           != reference[k].tobytes())
        measure = receipts.domain_mass_measure(store, geo_state)
        ulp = ulp_distance(truth, measure)
        if differing:
            raise AssertionError(
                f"{label}: carriers differ from monolithic in {differing}; "
                "the receipt comparison below would be meaningless")
        if measure.hex() != truth.hex():
            raise AssertionError(
                f"{label}: store receipt {measure!r} is {ulp} ulp from the "
                f"resident {truth!r}; a receipt taken from the store cannot "
                "depend on how many cards stepped the tiles")
        lines.append(f"{label}: carriers bit-exact, receipt {ulp} ulp "
                     f"({measure!r})")
        del store
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.get_default_memory_pool().free_all_blocks()
    if ndev < 2:
        lines.append(f"SKIPPED the two-card leg: {ndev} device(s) visible")
    return lines


def contract_receipt_closes_the_three_term_budget(record: dict) -> str:
    """The store receipt drives ``conservation_closure`` end to end."""
    from woof.verify import conservation_closure as cc

    from tilestream import receipts

    terms = {name: 0.0 for name in cc.MASS_BUDGET_TERMS}
    residual = cc.mass_budget_residual(record["measure_start"],
                                       record["store_measure"], terms)
    relative = cc.relative_residual(residual, record["measure_start"])
    entry = cc.residual_entry(
        relative, tier="observability",
        definition="|(store receipt at N) - (state receipt at 0) - "
                   "sum(three lateral terms)| / measure_start, periodic "
                   "domain so every lateral term is measured zero")
    provenance = receipts.receipt_provenance("streamed-store")
    if entry["bound"] is not None:
        raise AssertionError("no bound is pinned here")
    return (f"three-term budget closes on the store receipt: residual "
            f"{residual:+.6e} ({relative:.3e} relative), tier "
            f"{entry['tier']}, reduction_order "
            f"{provenance['reduction_order']!r}")


# --------------------------------------------------------------------------

def _fmt(record: dict) -> list[str]:
    lines = [
        f"{record['label']}  ({record['tiles']} tiles, halo "
        f"{record['halo']}, {record['steps']} steps, "
        f"{record['carriers']} carriers)",
        f"msft in [{record['msft_min']:.6f}, {record['msft_max']:.6f}]"
        f"{'  (identity -- the sum is EXACT, see the docstring)' if record['msft_identity'] else ''}",
        f"resident receipt {record['measure_start']!r} -> "
        f"{record['measure_end']!r}  (delta {record['delta']:+.6g})",
        f"(a) from the store, monolithic order: {record['store_measure']!r}  "
        f"{record['store_ulp']} ulp from resident",
        f"(b) folded per tile interior:         {record['folded']!r}  "
        f"{record['folded_ulp']:+d} ulp, {record['folded_abs']:.3e} abs, "
        f"{record['folded_rel']:.3e} rel  [DATA, not asserted]",
        f"    reversed fold order:              {record['reversed_ulp']:+d} "
        f"ulp, spread across associations {record['fold_spread_ulp']} ulp",
        f"    fold survey over 6 partitions:    max |ulp| "
        f"{record['fold_survey_max_abs_ulp']}  "
        + "  ".join(f"{k}:{v['ulp']:+d}"
                    for k, v in record["fold_survey"].items())
        + "  [DATA, not asserted]",
        f"CONTROL folded over GATHERED windows: "
        f"{record['gathered_ratio']:.6f}x the truth "
        f"(plan redundancy {record['redundancy']:.2f})",
        f"CONTROL receipt read from state:      {record['stale_ulp']} ulp "
        f"from the truth, equals the t=0 mass: {record['stale_is_t0']}",
        f"resident TKE budget: {record['resident_tke_steps']} steps, "
        f"residual_rel {record['resident_tke_residual_rel']:.3e}; "
        f"streamed drain from state: {record['streamed_tke_from_state']}",
    ]
    if "tke_naive_fold_ratio" in record:
        ratios = record["tke_naive_fold_ratio"]
        worst = max(ratios, key=lambda k: abs(ratios[k] - 1.0))
        lines.append(
            f"CONTROL naive TKE fold over {len(record['tke_buffers'])} "
            f"buffers ({record['tke_buffers'][0]['steps']:.0f} steps each "
            f"vs the domain's {record['resident_tke_steps']:.0f}): "
            f"shear {ratios.get('shear', float('nan')):.4f}x, worst term "
            f"{worst!r} at {ratios[worst]:.4g}x")
    return lines


def main(argv=None) -> int:
    import cupy as cp

    argv = list(sys.argv[1:] if argv is None else argv)
    nsteps = 8 if "--quick" in argv else NSTEPS
    emit = "--json" in argv

    free, total = cp.cuda.runtime.memGetInfo()
    print(f"cupy {cp.__version__}  "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  "
          f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f}")
    print()
    failures: list[str] = []
    records: list[dict] = []

    print("=" * 78)
    print("CONTRACTS")
    print("=" * 78)
    for name, fn in (
            ("step_kwargs reach the tiles", contract_step_kwargs_reach_the_tiles),
            ("domain-scope observers refused",
             contract_domain_scope_observers_are_refused),
            ("receipts refuse what they cannot take",
             contract_receipts_refuse_what_they_cannot_take),
            ("StreamedDomain seam returns the number",
             lambda: contract_streamed_domain_seam_returns_the_number(nsteps)),
            ("multi-GPU receipt is the same number",
             lambda: contract_multigpu_receipt_is_the_same_number(nsteps))):
        try:
            out = fn()
            for line in ([out] if isinstance(out, str) else out):
                print(f"  PASS  {line}")
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"contract {name}: {exc}")
            print(f"  FAIL  {name}: {exc}")
            traceback.print_exc()
    print()

    print("=" * 78)
    print("DOMAIN MASS RECEIPT  (a) asserted bit-identical, (b) published")
    print("=" * 78)
    for label, build in CONFIGURATIONS:
        try:
            rec = receipt_case(label, build, nsteps)
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"receipt {label}: raised {exc!r}")
            print(f"  ERROR {label}: {exc!r}")
            traceback.print_exc()
            continue
        records.append(rec)
        checks = (
            ("streamed carriers bit-exact", rec["bitexact"], True),
            ("tile interiors partition the domain",
             rec["interior_partition_exact"], True),
            ("(a) store receipt bit-identical to resident",
             rec["store_bit_identical"], True),
            ("CONTROL state receipt fires",
             rec["stale_measure"] != rec["measure_end"], True),
            ("CONTROL state receipt is the t=0 mass", rec["stale_is_t0"], True),
            ("CONTROL gathered-window fold fires",
             abs(rec["gathered_ratio"] - 1.0) > 0.1, True),
            ("CONTROL streamed TKE drain from state is unavailable",
             rec["streamed_tke_from_state"] is None, True),
        )
        ok = all(got is want for _n, got, want in checks)
        print(f"  {'PASS' if ok else 'FAIL':4s}  "
              + "\n        ".join(_fmt(rec)))
        for name, got, want in checks:
            if got is not want:
                failures.append(f"{label}: {name} = {got!r}, want {want!r}")
                print(f"        FAILED CHECK: {name} = {got!r}")
        try:
            print("  PASS  "
                  + contract_receipt_closes_the_three_term_budget(rec))
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"closure {label}: {exc}")
            print(f"  FAIL  closure {label}: {exc}")
        print()

    print("=" * 78)
    print("REDUCTION-ORDER POLICY  (what several later items depend on)")
    print("=" * 78)
    for rec in records:
        print(f"  {rec['label']}:")
        survey = "  ".join(
            f"{k}:{v['ulp']:+d}" for k, v in rec["fold_survey"].items())
        print(f"      the tiling's own 4x4 fold {rec['folded_ulp']:+d} ulp "
              f"({rec['folded_rel']:.3e} rel), spread across associations "
              f"{rec['fold_spread_ulp']} ulp")
        print(f"      fold survey (partition:ulp)  {survey}    max |ulp| "
              f"{rec['fold_survey_max_abs_ulp']}"
              + ("\n      ^ identity map factors: the weight is exactly 1.0, "
                 "every term is an FP32 column mass and the whole sum is\n"
                 "        EXACT in FP64, so these zeros say nothing about "
                 "associativity.  Read the mapped row instead."
                 if rec["msft_identity"] else ""))
    print("  POLICY: take domain reductions from the STORE in the monolithic")
    print("          traversal order.  A 2-D reduction always fits on the")
    print("          card even when the 3-D domain does not.")
    print()

    if emit:
        print("@@JSON@@")
        print(json.dumps(records, indent=2, default=str))

    print("=" * 78)
    if failures:
        print(f"FAILURES ({len(failures)})")
        for line in failures:
            print(f"  - {line}")
    else:
        print("ALL RECEIPT CHECKS PASS")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
