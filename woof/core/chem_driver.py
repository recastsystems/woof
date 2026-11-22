"""The chem step: WRF-Chem's ``chem_driver`` as one operator per model step.

WRF runs chemistry after the whole dynamics and physics step
(``share/solve_interface.F``: ``solve_em`` then ``chem_driver``).  WOOF-ArWen does
the same from the end of :func:`woof.core.dycore.step`, after microphysics,
so tiles and nests reach it through that step with no extra wiring.

Inside the step the order is WRF's (``chem/chem_driver.F``): emissions with
plume rise (:818), optics on radiation steps (:926), deposition velocities
and vertical mixing with deposition as its lower boundary (:1049), settling,
the (GOCART sulfur) mechanism (:1242), aerosol aging (:1577), large-scale wet
removal (:1620/:1689), then the Thompson coupling (a WOOF-ArWen addition, DESIGN
6.7).  :data:`CHEM_STEP_ORDER` is that order as process keys; a process runs
when an active row names it, and decides its own cadence (``chemdt``,
radiation steps, WRF's ``ktau > 2`` for deposition) from ``ctx.clock``.  The
PM sums and the other table diagnostics (:1700) depend only on the fields at
output time, so :mod:`woof.core.chem_outputs` evaluates them when a history
frame is written rather than every step.

The mass ledger files every kilogram of every transported row: the dycore's
change (advection, lateral boundaries, the positive-definite clamp, and
anything else that moved the fields between chem steps, such as nest
feedback) under ``transport``, and each process's change under its declared
``LEDGER`` bucket, measured by the driver before and after the process so a
process cannot book mass itself.  The accounting telescopes by construction
(``initial + sum(buckets) == current`` in float64), so it is a budget, not a
proof; that the transport itself conserves mass is proven by the tests on
periodic and zero-inflow domains.  Its vectors are ``chemdiag_ledger_*``
DomainState attributes, so a checkpoint carries it and a resumed run keeps
counting.
"""

from __future__ import annotations

from datetime import datetime

import cupy as cp
import numpy as np

from woof.core import constants as c
from woof.core.chem_context import (
    CHEM_STEP_ORDER, LEDGER_BUCKETS, ChemClock, ChemContext, chem_processes,
)
from woof.core.chem_state import ledger_attr, process_attr

__all__ = ["CHEM_STEP_ORDER", "M_AIR_G_MOL", "chem_step", "chem_pre_transport", "chem_processes",
           "ledger_report", "chem_ledger_receipt", "boundary_inflow"]

#: Molar mass of dry air, g mol-1: WRF-Chem's ``mwdry`` (28.966), the value
#: its ppmv conversions use (e.g. chem/dry_dep_driver.F:786).
M_AIR_G_MOL = 28.966


def _kg_per_unit(row) -> float:
    """kg of species per kg of dry air for one unit of the row's field."""
    if row.units == "ug kg-1":
        return 1e-9
    if row.units == "ppmv":
        return 1e-6 * float(row.molar_mass_g_mol) / M_AIR_G_MOL
    raise ValueError(f"chem row {row.name!r}: no ledger conversion for "
                     f"units {row.units!r}")


def _masses(state, cfg, rows, fields, mup) -> cp.ndarray:
    """Each row's mass in kilograms (float64, on the device).

    ``kernels/chem_ledger.cu`` sums ``q * C(mu) / msft^2 * (-dnw)`` per model
    row for every field in one launch (WRF's dry mass per unit area of a
    layer is ``(c1h*mu + c2h) * dnw / g``, a cell's area ``dx*dy/msft^2``);
    the partials are then summed along the row axis, scaled by
    ``dx*dy/g`` and converted from the row's units.  One read per field, no
    full-size temporary, the same bytes for the same inputs.
    """
    from woof.core.kernels import get_kernel

    if not fields:
        return cp.zeros(0, dtype=cp.float64)
    nz, ny, nx = state.p.shape
    table = cp.asarray([f.data.ptr for f in fields], dtype=cp.uint64)
    partial = cp.empty((len(fields), nz * ny), dtype=cp.float64)
    get_kernel("chem_ledger", "chem_row_mass")(
        (nz * ny, len(fields)), (256,),
        (table, state.mub2d, mup, state.c1h, state.c2h, state.dnw,
         state.msft, np.int32(bool(state.has_msf)), np.int32(nz),
         np.int32(ny), np.int32(nx),
         *map(np.int32, getattr(getattr(state, "chem", None), "ledger_window", (0, ny, 0, nx))), partial))
    scale = float(cfg.dx) * float(cfg.dy) / float(c.G)
    units = cp.asarray([_kg_per_unit(row) * scale for row in rows],
                       dtype=cp.float64)
    return partial.sum(axis=1) * units


def _clock(state, cfg, ktau: int) -> ChemClock:
    start = getattr(cfg, "start_time", None)
    if start is None:
        # RunConfig carries no start time; the physics driver hands the
        # run's to the chem state (woof/core/chem_geometry.py).
        start = getattr(getattr(state, "chem", None), "start_time", None)
    gmt = 0.0
    julday = 1
    if isinstance(start, datetime):
        gmt = start.hour + start.minute / 60.0 + start.second / 3600.0
        julday = int(start.timetuple().tm_yday)
    return ChemClock(gmt=gmt, julday=julday,
                     curr_secs=float(state.elapsed_seconds), ktau=ktau,
                     dt=float(cfg.dt))


def _context(state, cfg, ktau: int) -> ChemContext:
    chem = state.chem
    processes = getattr(chem, "processes", None)
    if processes is None:
        processes = chem.processes = chem_processes(chem.table)
    diag = {}
    for _key, module in processes:
        for alloc in getattr(module, "ALLOCATES", ()):
            diag[alloc.name] = getattr(state, process_attr(alloc))
    ddvel = getattr(chem, "ddvel", None)
    if ddvel is None:
        nz, ny, nx = state.p.shape
        ddvel = chem.ddvel = cp.zeros((len(chem.table.rows), ny, nx),
                                      dtype=cp.float32)
    prep = getattr(chem, "prep", None)
    if prep is None:
        from woof.core.chem_prep import ChemPrep
        prep = chem.prep = ChemPrep(state)
    physics_fields = (getattr(state.physics, "fields", {})
                      if getattr(state, "physics", None) is not None else {})
    return ChemContext(state, cfg, chem.table, prep=prep,
                       physics_fields=physics_fields, diag=diag, ddvel=ddvel,
                       clock=_clock(state, cfg, ktau))


def _ledger_begin(state, cfg, rows):
    """Open accounting at the untouched RK time-t state, once per domain."""
    chem = state.chem
    started = getattr(state, ledger_attr("started"))
    current = getattr(state, ledger_attr("current"))
    # One host read of the flag per state, not per step: afterwards the
    # ChemState remembers it (a restored checkpoint brings started = 1).
    if not getattr(chem, "ledger_started", False):
        chem.ledger_started = bool(cp.asnumpy(started)[0])
    if not chem.ledger_started:
        # The first chem step after construction: the time-t fields and
        # column mass (chem0_<row>, mup0) are still intact, so the initial
        # masses are exact without a separate call at initialization.
        initial = _masses(state, cfg, rows,
                          tuple(getattr(state, row.time_attr)
                                for row in rows), state.mup0)
        getattr(state, ledger_attr("initial"))[...] = initial
        current[...] = initial
        started[...] = 1
        chem.ledger_started = True


def _initialize_processes(ctx, processes):
    chem = ctx.state.chem
    if not getattr(chem, "initialized", False):
        for key, module in processes:
            missing = [name for name in getattr(module, "REQUIRES", ()) if not ctx.has(name)]
            if missing:
                raise RuntimeError(f"chem process {key} reads {missing}, which neither "
                    "chem_prep nor this run's physics publishes, so it would act on fields nothing wrote")
            module.init(ctx)
        chem.initialized = True


def chem_pre_transport(state, cfg, dt: float) -> None:
    """Run declared native physics sources before the tracer RK loop.

    Existing chemistry processes remain post-step. Source mass is booked
    here by the same driver, so the later transport bucket cannot count it
    again. Only source-affected rows refresh their saved time-t field.
    """
    chem = state.chem
    processes = getattr(chem, "processes", None)
    if processes is None:
        processes = chem.processes = chem_processes(chem.table)
    early = [(key, module) for key, module in processes
             if (getattr(module, "TIMING", "post_step") == "pre_transport"
                 and getattr(module, "pre_transport_active", lambda _s, _c: True)(state, cfg))]
    if not early:
        return
    if getattr(chem, "externally_managed_ledger", False):
        ktau = int(round(state.elapsed_seconds / float(cfg.dt))) + 1
        ctx = _context(state, cfg, ktau)
        ctx.prep.refresh(state)
        _initialize_processes(ctx, processes)
        for _key, module in early:
            module.step(ctx, dt, ktau)
            for row in module.rows(chem.table):
                getattr(state, row.time_attr)[...] = getattr(state, row.state_attr)
        return
    rows = chem.transported
    _ledger_begin(state, cfg, rows)
    current = getattr(state, ledger_attr("current"))
    mass = _masses(state, cfg, rows, chem.fields(state), state.mup)
    getattr(state, ledger_attr("transport"))[...] += mass - current
    ktau = int(round(state.elapsed_seconds / float(cfg.dt))) + 1
    ctx = _context(state, cfg, ktau)
    ctx.prep.refresh(state)
    _initialize_processes(ctx, processes)
    index = {row.name: i for i, row in enumerate(rows)}
    for _key, module in early:
        acted = [row for row in module.rows(chem.table) if row.name in index]
        module.step(ctx, dt, ktau)
        if acted:
            idx = cp.asarray([index[row.name] for row in acted])
            now = _masses(state, cfg, acted,
                          tuple(getattr(state, row.state_attr) for row in acted), state.mup)
            getattr(state, ledger_attr(module.LEDGER))[idx] += now - mass[idx]
            mass[idx] = now
            for row in acted:
                getattr(state, row.time_attr)[...] = getattr(state, row.state_attr)
    current[...] = mass

def chem_step(state, cfg, dt: float) -> None:
    """Close one model step with the chem operator (see module docstring)."""
    chem = state.chem
    external = getattr(chem, "externally_managed_ledger", False)
    # Source-only streamed profiles retain their pre-transport owner. Resident
    # ranks additionally own an interior window and run every post-step process.
    if external and not hasattr(chem, "ledger_window"):
        for _, module in chem_processes(chem.table):
            if getattr(module, "TIMING", "post_step") != "pre_transport":
                raise ValueError("post-step chemistry needs an owned interior ledger window; "
                                 "otherwise overlapping halos double-count process mass")
            refresh = getattr(module, "refresh_output", None)
            if refresh is not None:
                refresh(state)
        return
    rows = chem.transported
    ktau = int(round(state.elapsed_seconds / float(cfg.dt))) + 1
    fields = chem.fields(state)
    mass = _masses(state, cfg, rows, fields, state.mup)
    if external:
        # One step's deltas only. The rank owner folds these into the domain's
        # persisted vectors; halos never participate in a mass reduction.
        before = _masses(state, cfg, rows,
                         tuple(getattr(state, row.time_attr) for row in rows), state.mup0)
        for bucket in LEDGER_BUCKETS:
            getattr(state, ledger_attr(bucket))[...] = 0
        getattr(state, ledger_attr("initial"))[...] = before
        getattr(state, ledger_attr("current"))[...] = before
    else:
        _ledger_begin(state, cfg, rows)
    current = getattr(state, ledger_attr("current"))
    getattr(state, ledger_attr("transport"))[...] += mass - current
    processes = getattr(chem, "processes", None)
    if processes is None:
        processes = chem.processes = chem_processes(chem.table)
    later = [(key, module) for key, module in processes
             if getattr(module, "TIMING", "post_step") == "post_step"]
    if later:
        ctx = _context(state, cfg, ktau)
        ctx.prep.refresh(state)
        ctx.ddvel[...] = 0
        _initialize_processes(ctx, processes)
        index = {row.name: i for i, row in enumerate(rows)}
        for _key, module in later:
            acted = [row for row in module.rows(chem.table)
                     if row.name in index]
            module.step(ctx, dt, ktau)
            if acted:
                idx = cp.asarray([index[row.name] for row in acted])
                now = _masses(state, cfg, acted,
                              tuple(getattr(state, row.state_attr)
                                    for row in acted), state.mup)
                getattr(state, ledger_attr(module.LEDGER))[idx] += (
                    now - mass[idx])
                mass[idx] = now
    current[...] = mass
    for _key, module in processes:
        refresh = getattr(module, "refresh_output", None)
        if refresh is not None:
            refresh(state)


def chem_ledger_receipt(state, *, store=None) -> dict | None:
    """The forecast receipt's chem ledger for one domain, None when chem is
    off (so a chem-off receipt keeps its bytes).

    Every row's buckets, totals and closure in kg (:func:`ledger_report`),
    plus the worst row's closure relative to its initial or current mass,
    whichever is larger: the one number that says the ledger balanced.
    """
    if getattr(state, "chem", None) is None:
        return None
    report = ledger_report(state, store=store)
    worst = 0.0
    for entry in report["rows"].values():
        scale = max(abs(entry["initial_kg"]), abs(entry["current_kg"]))
        if scale > 0.0:
            worst = max(worst, abs(entry["closure_kg"]) / scale)
    report["max_relative_closure"] = worst
    report["inflow"] = boundary_inflow(state.chem.table)
    for key, module in chem_processes(state.chem.table):
        method = getattr(module, "source_receipt", None)
        if method is not None:
            report.setdefault("sources", {})[key] = method(state)
    return report


def boundary_inflow(table) -> dict:
    """Where each transported row's lateral inflow comes from, in words.

    A row whose boundary sources ``chem_sources`` leaves out takes its
    ``default_inflow`` at every inflow face (WRF's no-``have_bcs_chem``
    branch), so whatever the row carries outside the domain never enters
    it; the receipt says so rather than leaving a zero to be read as a
    measured clean boundary (DESIGN 5.1).
    """
    out = {}
    for row in table.transported:
        named = [ref["source"] for ref in row.boundary]
        enabled = [name for name in named if name in table.enabled_sources]
        if enabled:
            out[row.name] = f"{row.name} inflow from {enabled[0]}"
        else:
            out[row.name] = (
                f"{row.name} from outside the domain is absent: no enabled "
                f"boundary source ({', '.join(named) or 'none named'}), so "
                f"inflow is the row's default {row.default_inflow:g} "
                f"{row.units}")
    return out


def ledger_report(state, *, store=None) -> dict:
    """Host copy of the ledger: per row, every bucket, the totals, closure."""
    chem = state.chem
    from woof.core.streaming import domain_store
    if store is None:
        store = domain_store(state)
    def vector(name):
        key = 'state/' + ledger_attr(name)
        if store is not None:
            return store[key]
        return getattr(state, ledger_attr(name))
    host = {name: cp.asnumpy(vector(name))
            for name in ("initial", "current", *LEDGER_BUCKETS)}
    out = {}
    for i, row in enumerate(chem.transported):
        entry = {f"{name}_kg": float(host[name][i]) for name in LEDGER_BUCKETS}
        entry["initial_kg"] = float(host["initial"][i])
        entry["current_kg"] = float(host["current"][i])
        total = entry["initial_kg"] + sum(float(host[name][i])
                                          for name in LEDGER_BUCKETS)
        entry["closure_kg"] = total - entry["current_kg"]
        out[row.name] = entry
    return {"started": bool(np.asarray(cp.asnumpy(vector('started')))[0]), "rows": out}
