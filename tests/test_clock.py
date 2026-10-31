"""Phase-5 Task 9 (lane L2): integer-tick clocks + flat schedule executor.

Architecture section C pins (docs/superpowers/specs/
2026-07-16-phase5-nesting-architecture.md, 2026-07-16 amendment):

- schedule pin: the flat op table equals an INDEPENDENT reference
  recursion (transcribed here from frame/module_integrate.F with Fraction
  absolute clocks -- structurally distinct from the generator's per-period
  tick expansion) for chains (1,4,3,3), (1,3,3), (1,5,3), (1,4,4,2),
  pinned for BOTH an interior period AND the final period, INCLUDING the
  last-io tail feedback (module_integrate.F:521-525: the per-kid guard
  :439-445 suppresses the loop feedback in the final period and the tail
  RE-ISSUES it -- 17 relocated FEEDBACKs, identical op sequence on
  linear chains, reordered on branching trees; F8 as corrected by the
  p5t9 review round);
- bundle-chain pins: tick = 1/3 s, step_ticks 180/45/15/5, 53 STEPs + 17
  FORCEs (+17 dormant FEEDBACKs) per d01 step in EVERY period, 12 h =
  129,600 ticks, ops counts + a SHA-256 table hash;
- chained-FP32 dt bit-pattern pins (60, 15, 5, 1.6666666) CONSUMED from
  T1's run.dt and validated BIT-EXACTLY against WRF's REAL construction
  (a one-ULP neighbour is rejected, shadow F2);
- dtbc phase pins (shadow F1): boundary kernels see the POST-INCREMENT
  value (dyn_em/solve_em.F:371-372 before :932-952) -- first three d04
  launches 0x3FD55555/0x40555555/0x40A00000; the root resets at external
  LBC seams (share/mediation_integrate.F:1522, 21600 s bundle stream);
- WRF REAL curr_secs mirror (shadow F3): ``elapsed_seconds_fp32``
  reproduces ``dt_whole + dt_num/REAL(dt_den)``
  (dyn_em/adapt_timestep_em.F) bit-for-bit -- 0x3FD55556 at d04's 5/3 s,
  one ULP from the FP64-then-cast value;
- WRF-recurrent dtbc scope (PROVENANCE D4): the stored FP32 accumulator
  resets at force/LBC seams and recurs before every solve, bitwise-identical
  to WRF on real74 and a nontrivial 1:7 chain;
- calendar pins: STEPRA 12/12/12/36, cudt = 900 ticks d01-only, bldt = 0
  -> every step, history 10800/2700/2700/2700, restart 32400 on the d01
  clock, lbc_interval 64800 ticks root-only; every cadence lands on
  exact ticks;
- clock_dt consumer pins (section C traced list): per-domain diff_6th
  coefficient equals the plain factor 0.12/0.10/0.08/0.06 at each
  domain's dt; Davies weights keyed by the child dt via
  ``_resident_weights``; the KF calendar exists on d01 only;
- the code-audit test enforcing the no-float-elapsed-accumulation rule
  (hardened per shadow F4; scope stated accurately, dycore legacy sites
  pinned).

All CPU.
"""

from __future__ import annotations

import ast
import dataclasses
import struct
import sys
import types
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_case_inputs

from woof.case_data import load_experiment_case
from woof.config import RunConfig
from woof.experiment import (DomainConfig, ExperimentConfig,
                              VerticalConfig, load_experiment)
from woof.core.clock import (FEEDBACK, FORCE, STEP, Op, Schedule,
                              TickClock, build_schedule, execute_schedule,
                              resolve_clock)

REPO = Path(__file__).resolve().parents[1]

#: These tests load configs/real74_4dom.toml with its declared inputs
#: required, so they run only where the WRF 1974 reference bundle its
#: [case_data] names is on disk, and skip naming the absent file elsewhere.
requires_4dom_inputs = requires_case_inputs(
    Path(__file__).resolve().parents[1] / "configs" / "real74_4dom.toml")

#: SHA-256 of the canonical bundle schedule serialization (ops counts +
#: hash pin, plan Task 9).  Any reordering, count change, or tick change
#: in the 4-domain table moves this digest.  (Changed in the review fix
#: round: the final period carries its 17 tail-relocated FEEDBACKs,
#: module_integrate.F:521-525.)
BUNDLE_TABLE_SHA256 = ("e2f1a02d3039cbdaf6a94098b46a09549c38b15fa8"
                       "9e5bf731730fbf009fa1c3")

#: (total STEPs, total FORCEs) per d01 period for each pinned ratio
#: chain; EVERY period carries equally many FEEDBACKs as FORCEs -- the
#: interior period's from the per-kid loop (:443), the final period's
#: from the last-io tail (:523), same positions on linear chains.
CHAIN_PERIOD_COUNTS = {
    (1, 4, 3, 3): (53, 17),
    (1, 3, 3): (13, 4),
    (1, 5, 3): (21, 6),
    (1, 4, 4, 2): (53, 21),
}


def _fp32_bits(value) -> int:
    return struct.unpack("<I", np.float32(value).tobytes())[0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bundle_exp() -> ExperimentConfig:
    return load_experiment_case(REPO / "configs" / "real74_4dom.toml")[0]


@pytest.fixture(scope="module")
def bundle_clock(bundle_exp) -> TickClock:
    # 21600 s is the bundle's ERA5 external-boundary stream cadence: the
    # root's dtbc-reset calendar (share/mediation_integrate.F:1522).
    return resolve_clock(bundle_exp, lbc_interval_s=21600.0)


@pytest.fixture(scope="module")
def bundle_schedule(bundle_exp, bundle_clock) -> Schedule:
    return build_schedule(bundle_exp, bundle_clock)


@pytest.fixture(scope="module")
def bundle_execution(bundle_schedule):
    """One full 12 h executor walk with recording handlers."""
    rec = {"step_gids": [], "history": {1: [], 2: [], 3: [], 4: []},
           "restart": [], "force_dtbc": [], "feedback": [],
           "lbc": [], "d04_launch": [], "d01_launch": []}

    def on_step(gid, dom):
        rec["step_gids"].append(gid)
        # launch-phase dtbc as a boundary kernel inside this solve
        # would see it (post-increment, dyn_em/solve_em.F:371-372).
        if gid == 4 and len(rec["d04_launch"]) < 6:
            rec["d04_launch"].append(_fp32_bits(dom.dtbc_launch_fp32))
        if gid == 1 and dom.ticks in (64620, 64800, 129420):
            rec["d01_launch"].append((dom.ticks,
                                      float(dom.dtbc_launch_fp32)))

    report = execute_schedule(
        bundle_schedule,
        on_step=on_step,
        on_history=lambda gid, ticks: rec["history"][gid].append(ticks),
        on_restart=lambda ticks: rec["restart"].append(ticks),
        on_lbc_reset=lambda ticks: rec["lbc"].append(ticks),
        on_force=lambda cid, pid, child, parent:
            rec["force_dtbc"].append((cid, child.dtbc)),
        on_feedback_prepare=lambda cid, pid, child, parent:
            rec["feedback"].append((cid, child.ticks)))
    return report, rec


def _chain_experiment(tratios, *, run_seconds, history_s=60.0):
    """Synthetic linear-chain experiment (grid ratios == time ratios).

    ``run.dt`` carries the chained-FP32 kernel dt exactly as T1's loader
    writes it (np.float32 division down the chain,
    share/set_timekeeping.F:368; woof/experiment.py:827-833).
    """
    assert tratios[0] == 1
    domains = []
    dt32 = None
    for i, ratio in enumerate(tratios):
        gid = i + 1
        dt32 = np.float32(60) if i == 0 else dt32 / np.float32(ratio)
        run = RunConfig(nx=30, ny=30, nz=4, dx=12000.0, dy=12000.0,
                        ztop=12000.0, dt=float(dt32),
                        run_seconds=run_seconds,
                        output_interval_s=history_s,
                        restart_interval_s=0.0,
                        specified=(i == 0), nested=(i > 0), grid_id=gid)
        domains.append(DomainConfig(
            grid_id=gid, parent_id=i, i_parent_start=1, j_parent_start=1,
            parent_grid_ratio=ratio, parent_time_step_ratio=ratio,
            history_interval_s=history_s, run=run,
            time_step=60 if i == 0 else None))
    return ExperimentConfig(
        name="chain" + "x".join(map(str, tratios)),
        start_time=datetime(1974, 4, 3, 12, 0, 0),
        run_seconds=run_seconds,
        vertical=VerticalConfig(eta_levels=(), p_top=0.0, hybrid_opt=0,
                                etac=0.2),
        projection=None, restart_interval_s=0.0, domains=tuple(domains))


def _with_delayed_start(exp, grid_id: int, offset_s: int):
    domains = tuple(
        dataclasses.replace(
            dc, start_time=exp.start_time + timedelta(seconds=offset_s))
        if dc.grid_id == grid_id else dc
        for dc in exp.domains)
    return dataclasses.replace(exp, domains=domains)


# ---------------------------------------------------------------------------
# Independent reference recursion (the schedule-pin oracle)
# ---------------------------------------------------------------------------

def _wrf_reference_ops(exp):
    """Transcription of frame/module_integrate.F with Fraction clocks.

    INDEPENDENT of the generator by construction: one whole-run call on
    the head grid with per-domain ABSOLUTE Fraction times and explicit
    ``stop_subtime`` bookkeeping, versus build_schedule's per-period
    integer-tick expansion.  Represents the terminal stop-time feedback
    guard exactly as the source does (:439-445: the LOOP feedback is
    skipped whenever the head grid, the parent, or the child is at stop
    time) AND the last-io tail (:507-526: on every nested integrate()
    return, feedback is RE-ISSUED at :523 when the head grid or any grid
    on the chain up to the head is at stop time).
    """
    dt = {dc.grid_id: exp.dt_exact(dc.grid_id) for dc in exp.domains}
    kids = {dc.grid_id: [c.grid_id for c in exp.children_of(dc.grid_id)]
            for dc in exp.domains}
    parent_of = {dc.grid_id: dc.parent_id for dc in exp.domains}
    stop_time = Fraction(exp.run_seconds)
    head = exp.root.grid_id
    current = {gid: Fraction(0) for gid in dt}
    stop_subtime = {gid: Fraction(0) for gid in dt}
    ops = []

    def is_stop_time(gid):  # domain_clockisstoptime
        return current[gid] >= stop_time

    def integrate(gid):
        if is_stop_time(gid):                       # :318
            return
        while current[gid] < stop_subtime[gid]:     # :328
            ops.append((STEP, gid, 0))              # :393 solve
            current[gid] += dt[gid]                 # :396 clockadvance
            for kid in kids[gid]:                   # :411-423
                ops.append((FORCE, kid, gid))       # :416 med_nest_force
                stop_subtime[kid] = current[gid]    # :420-421
            for kid in kids[gid]:                   # :425-434
                integrate(kid)                      # :430
            for kid in kids[gid]:                   # :436-453
                if not (is_stop_time(head) or is_stop_time(gid)
                        or is_stop_time(kid)):      # :439-441
                    ops.append((FEEDBACK, kid, gid))  # :443
        if gid != head:                             # :509/:511
            last_io = is_stop_time(head)            # :513
            walk = gid                              # :514
            while walk != head:                     # :515-520
                if is_stop_time(walk):
                    last_io = True                  # :516-518
                walk = parent_of[walk]              # :519
            if last_io:                             # :521
                ops.append((FEEDBACK, gid, parent_of[gid]))  # :523

    stop_subtime[head] = stop_time
    integrate(head)
    return ops


def _split_periods(ops, head_gid):
    """Split a whole-run op list at head-grid STEP boundaries."""
    periods, current = [], None
    for op in ops:
        if op[0] == STEP and op[1] == head_gid:
            if current is not None:
                periods.append(current)
            current = []
        current.append(op)
    periods.append(current)
    return periods


# ---------------------------------------------------------------------------
# resolve_clock: bundle tick + calendar pins
# ---------------------------------------------------------------------------

@requires_4dom_inputs
def test_bundle_tick_pins(bundle_clock):
    """tick = 1/3 s, step_ticks 180/45/15/5, 12 h = 129,600 ticks."""
    assert bundle_clock.tick_den == 3
    assert bundle_clock.tick_exact() == Fraction(1, 3)
    assert bundle_clock.run_ticks == 129_600
    assert [s.step_ticks for s in bundle_clock.domains] == [180, 45, 15, 5]
    assert [s.grid_id for s in bundle_clock.domains] == [1, 2, 3, 4]
    # 36 d04 steps of 5 ticks == 180 ticks == one d01 step, exactly.
    assert 36 * 5 == 180 == bundle_clock.root.step_ticks
    assert bundle_clock.dt_exact(4) == Fraction(5, 3)


@requires_4dom_inputs
def test_chained_fp32_dt_consumed_bit_pins(bundle_exp, bundle_clock):
    """WRF REAL grid%dt bit patterns, CONSUMED from T1's run.dt."""
    expected_bits = {1: 0x42700000, 2: 0x41700000,
                     3: 0x40A00000, 4: 0x3FD55555}
    # Independent chained division (set_timekeeping.F:368) for the
    # consume-not-re-derive check.
    chained = np.float32(60)
    independent = {1: chained}
    for gid, ratio in ((2, 4), (3, 3), (4, 3)):
        chained = chained / np.float32(ratio)
        independent[gid] = chained
    for dc in bundle_exp.domains:
        spec = bundle_clock.spec(dc.grid_id)
        assert _fp32_bits(spec.dt_fp32) == expected_bits[dc.grid_id]
        # consumed from the schema, not recomputed:
        assert float(spec.dt_fp32) == dc.run.dt
        assert spec.dt_fp32 == independent[dc.grid_id]
    # d04's REAL dt is WRF's chained 1.6666666, never float(5/3).
    assert float(bundle_clock.spec(4).dt_fp32) == 1.6666666269302368


@requires_4dom_inputs
def test_bundle_calendar_pins(bundle_clock):
    """STEPRA 12/12/12/36; cudt 900 ticks d01-only; bldt=0 every step;
    history 10800/2700/2700/2700; restart 32400 on the d01 clock."""
    stepra = {s.grid_id: s.stepra for s in bundle_clock.domains}
    assert stepra == {1: 12, 2: 12, 3: 12, 4: 36}
    radt = {s.grid_id: s.radt_ticks for s in bundle_clock.domains}
    assert radt == {1: 2160, 2: 540, 3: 180, 4: 180}

    root = bundle_clock.root
    assert root.stepcu == 5 and root.cudt_ticks == 900
    for gid in (2, 3, 4):
        spec = bundle_clock.spec(gid)
        assert spec.stepcu is None and spec.cudt_ticks is None

    # bldt = 0 -> every step on each domain's own clock.
    for spec in bundle_clock.domains:
        assert spec.stepbl == 1
        assert spec.bldt_ticks == spec.step_ticks

    hist = {s.grid_id: s.history_ticks for s in bundle_clock.domains}
    assert hist == {1: 10800, 2: 2700, 3: 2700, 4: 2700}

    assert root.restart_ticks == 32400
    # external-LBC seam calendar: 21600 s ERA5 stream = 64800 ticks on
    # the d01 clock ONLY (children take nest forces, not external LBCs)
    assert root.lbc_interval_ticks == 64800
    for gid in (2, 3, 4):
        assert bundle_clock.spec(gid).restart_ticks is None
        assert bundle_clock.spec(gid).lbc_interval_ticks is None


@requires_4dom_inputs
def test_every_cadence_lands_on_exact_ticks(bundle_clock):
    """Each calendar is an exact multiple of its domain's step_ticks, and
    the WRF driver predicates fire with tick spacing == the calendar."""
    from woof.core.physics import (_cumulus_step_due,
                                    _radiation_step_due,
                                    _surface_pbl_step_due)

    for spec in bundle_clock.domains:
        for ticks in (spec.radt_ticks, spec.cudt_ticks, spec.bldt_ticks,
                      spec.history_ticks, spec.restart_ticks,
                      spec.lbc_interval_ticks):
            if ticks is not None:
                assert ticks % spec.step_ticks == 0
        assert bundle_clock.run_ticks % spec.step_ticks == 0

    for spec in bundle_clock.domains:
        steps_per_hour = 10800 // spec.step_ticks
        due = [it for it in range(1, steps_per_hour + 1)
               if _radiation_step_due(it, spec.stepra, 1.0)]
        spacing = {(b - a) * spec.step_ticks for a, b in zip(due, due[1:])}
        assert spacing == {spec.radt_ticks}
        due = [it for it in range(1, steps_per_hour + 1)
               if _surface_pbl_step_due(it, spec.stepbl, 0.0)]
        assert due == list(range(1, steps_per_hour + 1))  # every step
    root = bundle_clock.root
    due = [it for it in range(1, 181)
           if _cumulus_step_due(it, root.stepcu, 5.0)]
    spacing = {(b - a) * root.step_ticks for a, b in zip(due[1:], due[2:])}
    assert spacing == {root.cudt_ticks} == {900}


def test_physics_interval_steps_generalization():
    """Integer-tick division with exactness assertion (Fraction dt);
    the WRF REAL nint path is unchanged for float dt."""
    from woof.core.physics import _physics_interval_steps

    # exact rational division (section C)
    assert _physics_interval_steps(1.0, Fraction(5, 3)) == 36
    assert _physics_interval_steps(12.0, Fraction(60)) == 12
    assert _physics_interval_steps(5.0, Fraction(60)) == 5
    assert _physics_interval_steps(0.0, Fraction(60)) == 1  # every step
    # A non-integral quotient rounds as wrf.exe's NINT does (180/7 -> 26),
    # and the professor's cudt = 5 at dt = 45 s is 20/3 -> 7 steps.
    assert _physics_interval_steps(1.0, Fraction(7, 3)) == 26
    assert _physics_interval_steps(5.0, Fraction(45)) == 7
    assert _physics_interval_steps(5.0, 45.0) == 7

    # WRF nint on the chained-FP32 dt agrees on the bundle chain
    d04_dt = float(np.float32(60) / np.float32(4) / np.float32(3)
                   / np.float32(3))
    assert _physics_interval_steps(1.0, d04_dt) == 36
    assert _physics_interval_steps(12.0, 60.0) == 12
    # legacy float rounding frozen: Fortran NINT(2.5) rounds up
    assert _physics_interval_steps(25.0, 600.0) == 3
    assert _physics_interval_steps(-1.0, 60.0) == 1


def test_resolve_clock_validation_failures():
    base = _chain_experiment((1, 3, 3), run_seconds=180.0)

    # cadence not a whole number of the owning domain's steps
    bad = dataclasses.replace(
        base, domains=tuple(
            dataclasses.replace(
                dc, history_interval_s=50.0,
                run=dataclasses.replace(dc.run, output_interval_s=50.0))
            for dc in base.domains))
    with pytest.raises(ValueError, match="history_interval_s"):
        resolve_clock(bad)

    # run must end on a d01 step boundary
    bad = dataclasses.replace(base, run_seconds=90.0)
    with pytest.raises(ValueError, match="root-domain steps"):
        resolve_clock(bad)

    # run_seconds must be whole ticks
    bad = dataclasses.replace(base, run_seconds=100.1)
    with pytest.raises(ValueError, match="whole number of ticks"):
        resolve_clock(bad)

    # run.dt must be the binary64 image of an FP32 value (T1's chain)
    doms = list(base.domains)
    doms[2] = dataclasses.replace(
        doms[2], run=dataclasses.replace(doms[2].run,
                                         dt=float(Fraction(20, 3))))
    bad = dataclasses.replace(base, domains=tuple(doms))
    with pytest.raises(ValueError, match="binary64 image"):
        resolve_clock(bad)

    # an FP32-clean but WRONG dt is rejected bit-exactly
    doms = list(base.domains)
    doms[2] = dataclasses.replace(
        doms[2], run=dataclasses.replace(doms[2].run,
                                         dt=float(np.float32(6.5))))
    bad = dataclasses.replace(base, domains=tuple(doms))
    with pytest.raises(ValueError, match="not WRF's REAL dt"):
        resolve_clock(bad)

    # lbc_interval must be whole root steps
    with pytest.raises(ValueError, match="lbc_interval_s"):
        resolve_clock(base, lbc_interval_s=90.0)


@requires_4dom_inputs
def test_chained_dt_validation_is_bit_exact(bundle_exp):
    """Shadow F2 repro: the 1-ULP neighbour of d04's chained-FP32 dt
    (0x3FD55556 vs WRF's 0x3FD55555) must be REJECTED -- the validation
    compares bit patterns against WRF's REAL construction
    (set_timekeeping.F:343 root, :368 chained child), never a relative
    tolerance."""
    doms = list(bundle_exp.domains)
    good = doms[3].run.dt
    bad_dt = float(np.nextafter(np.float32(good), np.float32(np.inf)))
    assert _fp32_bits(bad_dt) == 0x3FD55556          # the injected ULP
    assert _fp32_bits(good) == 0x3FD55555            # WRF's value
    doms[3] = dataclasses.replace(
        doms[3], run=dataclasses.replace(doms[3].run, dt=bad_dt))
    bad = dataclasses.replace(bundle_exp, domains=tuple(doms))
    with pytest.raises(ValueError, match="not WRF's REAL dt"):
        resolve_clock(bad)
    # the root is validated against REAL(time_step) +
    # REAL(num)/REAL(den) (set_timekeeping.F:343)
    doms = list(bundle_exp.domains)
    root_bad = float(np.nextafter(np.float32(doms[0].run.dt),
                                  np.float32(0)))
    doms[0] = dataclasses.replace(
        doms[0], run=dataclasses.replace(doms[0].run, dt=root_bad))
    bad = dataclasses.replace(bundle_exp, domains=tuple(doms))
    with pytest.raises(ValueError, match="not WRF's REAL dt"):
        resolve_clock(bad)


def test_chain_tick_resolutions():
    """tick_den = LCM of dt denominators on every pinned chain."""
    expected = {
        (1, 4, 3, 3): (3, [180, 45, 15, 5]),
        (1, 3, 3): (3, [180, 60, 20]),
        (1, 5, 3): (1, [60, 12, 4]),
        (1, 4, 4, 2): (8, [480, 120, 30, 15]),
    }
    for tratios, (tick_den, steps) in expected.items():
        clock = resolve_clock(_chain_experiment(tratios,
                                                run_seconds=240.0))
        assert clock.tick_den == tick_den
        assert [s.step_ticks for s in clock.domains] == steps


# ---------------------------------------------------------------------------
# Schedule: recursion pin, terminal guard, bundle table hash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tratios", sorted(CHAIN_PERIOD_COUNTS),
                         ids=lambda t: "x".join(map(str, t)))
def test_flat_table_equals_reference_recursion(tratios):
    """The flat table reproduces the WRF recursion exactly -- terminal
    guard AND last-io tail included -- pinned for an interior AND the
    final period."""
    exp = _chain_experiment(tratios, run_seconds=240.0)  # 4 d01 periods
    sched = build_schedule(exp)
    ref = _wrf_reference_ops(exp)
    assert list(sched.full_ops()) == ref

    ref_periods = _split_periods(ref, exp.root.grid_id)
    assert len(ref_periods) == sched.periods == 4
    assert tuple(ref_periods[1]) == sched.interior_period
    assert tuple(ref_periods[-1]) == sched.final_period

    n_steps, n_forces = CHAIN_PERIOD_COUNTS[tratios]
    assert sched.op_counts(sched.interior_period) == {
        STEP: n_steps, FORCE: n_forces, FEEDBACK: n_forces}
    # the final period's feedbacks are RELOCATED to the last-io tail
    # (:521-525), never suppressed: same counts, and on a LINEAR chain
    # the tail positions coincide with the loop positions exactly.
    assert sched.op_counts(sched.final_period) == {
        STEP: n_steps, FORCE: n_forces, FEEDBACK: n_forces}
    assert sched.final_period == sched.interior_period


@requires_4dom_inputs
def test_bundle_schedule_counts_and_hash(bundle_schedule):
    """53 STEPs + 17 FORCEs + 17 dormant FEEDBACKs per d01 step in EVERY
    period (final-period feedbacks tail-relocated, not suppressed); 720
    periods; table hash pin."""
    sched = bundle_schedule
    assert sched.periods == 720
    assert sched.period_ticks == 180
    assert sched.op_counts(sched.interior_period) == {
        STEP: 53, FORCE: 17, FEEDBACK: 17}
    assert len(sched.interior_period) == 87
    assert sched.op_counts(sched.final_period) == {
        STEP: 53, FORCE: 17, FEEDBACK: 17}
    assert len(sched.final_period) == 87
    # linear chain: the tail positions (:523) coincide with the loop
    # positions (:443) -- identical op sequence, 720 identical periods.
    assert sched.final_period == sched.interior_period
    assert sched.table_hash() == BUNDLE_TABLE_SHA256
    # period_ops routing: every interior index shares one table object
    assert sched.period_ops(0) is sched.period_ops(360)
    assert sched.period_ops(719) is sched.final_period
    with pytest.raises(IndexError):
        sched.period_ops(720)


@requires_4dom_inputs
def test_bundle_schedule_equals_reference_recursion(bundle_exp,
                                                    bundle_schedule):
    """Full 12 h flat walk == the reference recursion (62,640 ops)."""
    ref = _wrf_reference_ops(bundle_exp)
    full = list(bundle_schedule.full_ops())
    assert len(full) == 87 * 720 == 62_640
    assert full == ref


@requires_4dom_inputs
def test_bundle_interior_period_ordering(bundle_schedule):
    """Leading and trailing op-order pins for the (1,4,3,3) period."""
    ops = bundle_schedule.interior_period
    assert ops[:9] == (
        Op(STEP, 1, 0),      # d01 solve + advance
        Op(FORCE, 2, 1),     # med_nest_force d01->d02 (:416)
        Op(STEP, 2, 0),
        Op(FORCE, 3, 2),
        Op(STEP, 3, 0),
        Op(FORCE, 4, 3),
        Op(STEP, 4, 0), Op(STEP, 4, 0), Op(STEP, 4, 0),
    )
    assert ops[9] == Op(FEEDBACK, 4, 3)   # d04 window closes (:443)
    assert ops[-3:] == (Op(FEEDBACK, 4, 3), Op(FEEDBACK, 3, 2),
                        Op(FEEDBACK, 2, 1))


def test_chain_133_interior_period_explicit():
    """Hand-derived (1,3,3) period against the generator."""
    exp = _chain_experiment((1, 3, 3), run_seconds=180.0)
    sched = build_schedule(exp)
    d03_block = [Op(STEP, 3, 0)] * 3
    d02_iter = [Op(STEP, 2, 0), Op(FORCE, 3, 2), *d03_block,
                Op(FEEDBACK, 3, 2)]
    expected = (Op(STEP, 1, 0), Op(FORCE, 2, 1),
                *d02_iter, *d02_iter, *d02_iter, Op(FEEDBACK, 2, 1))
    assert sched.interior_period == expected


def test_multi_child_tree_three_loop_order():
    """WRF forces ALL kids, then integrates ALL kids, then feedbacks ALL
    kids (three separate per-kid loops, :411/:425/:436) -- pinned on a
    branching tree and against the reference recursion."""
    def _dom(gid, parent, ratio, dt32):
        return DomainConfig(
            grid_id=gid, parent_id=parent, i_parent_start=1,
            j_parent_start=1, parent_grid_ratio=ratio,
            parent_time_step_ratio=ratio, history_interval_s=60.0,
            run=RunConfig(nx=30, ny=30, nz=4, dx=12000.0, dy=12000.0,
                          ztop=12000.0, dt=float(dt32),
                          run_seconds=120.0, output_interval_s=60.0,
                          restart_interval_s=0.0,
                          specified=(parent == 0), nested=(parent != 0),
                          grid_id=gid),
            time_step=60 if parent == 0 else None)

    tree = ExperimentConfig(
        name="tree", start_time=datetime(1974, 4, 3, 12),
        run_seconds=120.0,
        vertical=VerticalConfig(eta_levels=(), p_top=0.0, hybrid_opt=0,
                                etac=0.2),
        projection=None, restart_interval_s=0.0,
        domains=(_dom(1, 0, 1, np.float32(60)),
                 _dom(2, 1, 2, np.float32(60) / np.float32(2)),
                 _dom(3, 1, 3, np.float32(60) / np.float32(3))))
    sched = build_schedule(tree)
    assert sched.interior_period == (
        Op(STEP, 1, 0),
        Op(FORCE, 2, 1), Op(FORCE, 3, 1),
        Op(STEP, 2, 0), Op(STEP, 2, 0),
        Op(STEP, 3, 0), Op(STEP, 3, 0), Op(STEP, 3, 0),
        Op(FEEDBACK, 2, 1), Op(FEEDBACK, 3, 1))
    # FINAL period: the tail (:521-525) re-issues each kid's feedback at
    # its OWN integrate() return -- kid1's feedback precedes kid2's
    # window, unlike the interior per-kid loop where both follow the
    # last window.  Same counts, reordered positions.
    assert sched.final_period == (
        Op(STEP, 1, 0),
        Op(FORCE, 2, 1), Op(FORCE, 3, 1),
        Op(STEP, 2, 0), Op(STEP, 2, 0), Op(FEEDBACK, 2, 1),
        Op(STEP, 3, 0), Op(STEP, 3, 0), Op(STEP, 3, 0),
        Op(FEEDBACK, 3, 1))
    assert list(sched.full_ops()) == _wrf_reference_ops(tree)


# ---------------------------------------------------------------------------
# Executor: integer walk, sync asserts, alarms, dtbc
# ---------------------------------------------------------------------------

@requires_4dom_inputs
def test_executor_full_bundle_walk(bundle_execution):
    report, rec = bundle_execution
    assert report.steps == 53 * 720 == 38_160
    assert report.forces == 17 * 720 == 12_240
    # dormant FEEDBACK hook dispatched in EVERY period: loop family in
    # interior periods (:443), tail family in the final one (:523)
    assert report.feedback_calls == 17 * 720 == 12_240
    assert rec["step_gids"][:10] == [1, 2, 3, 4, 4, 4, 3, 4, 4, 4]
    # every domain ends exactly at run stop; elapsed is the exact FP64
    # rendering of 12 h, refreshed from ticks
    for clock in report.clocks.values():
        assert clock.ticks == 129_600
        assert clock.elapsed_seconds == 43200.0
        assert clock.at_stop_time


@requires_4dom_inputs
def test_executor_history_restart_alarms(bundle_execution):
    """History fires at the top of the step (med_before_solve_io,
    module_integrate.F:375) incl. t=0, plus the run-stop frame; restart
    on the d01 clock only, never at t=0.  All four domain alarms coincide
    at 15-min sync ticks (frames time-consistent)."""
    report, rec = bundle_execution
    assert report.histories == {1: 13, 2: 49, 3: 49, 4: 49}
    assert rec["history"][1] == list(range(0, 129_601, 10_800))
    for gid in (2, 3, 4):
        assert rec["history"][gid] == list(range(0, 129_601, 2_700))
    # d01's hourly frames are a subset of the children's 15-min frames
    assert set(rec["history"][1]) <= set(rec["history"][2])
    assert report.restarts == 4
    assert rec["restart"] == [32_400, 64_800, 97_200, 129_600]


@requires_4dom_inputs
def test_executor_dtbc_launch_phase_and_lbc_seam(bundle_execution):
    """Shadow F1 pins: a boundary kernel launched during a step sees the
    POST-INCREMENT dtbc (WRF increments at dyn_em/solve_em.F:371-372
    before consuming at :932-952) -- first three d04 launches are 5/3,
    10/3, 5 s; and the root's dtbc resets at the 21600 s external-LBC
    seam (share/mediation_integrate.F:1522), so the first post-seam d01
    launch sees 60 s."""
    report, rec = bundle_execution
    triple = [0x3FD55555, 0x40555555, 0x40A00000]
    assert rec["d04_launch"][:3] == triple
    assert rec["d04_launch"][3:6] == triple  # next force window repeats
    # seams: t=0 (first interval read) and 21600 s = 64800 ticks; the
    # run-stop boundary starts no step, so exactly two resets in 12 h.
    assert rec["lbc"] == [0, 64_800]
    assert report.lbc_resets == 2
    # last pre-seam launch spans the full 21600 s interval; first
    # post-seam launch is one root step; final launch spans the last
    # interval again (64800 -> 129600 ticks).
    assert rec["d01_launch"] == [(64_620, 21_600.0), (64_800, 60.0),
                                 (129_420, 21_600.0)]


@requires_4dom_inputs
def test_executor_force_sync_and_dtbc_reset(bundle_execution):
    """At every FORCE the parent leads by exactly one parent step; the
    child's recurrent FP32 dtbc equals one parent interval (its Davies
    clock ran a full window) and resets to 0 (mediation_force_domain.F,
    fresh-force marker dyn_em/solve_em.F:344)."""
    report, rec = bundle_execution
    parent_dt = {2: 60.0, 3: 15.0, 4: 5.0}
    first_seen = set()
    for cid, dtbc in rec["force_dtbc"]:
        if cid not in first_seen:
            first_seen.add(cid)
            assert dtbc == 0.0   # t=0 force: nothing elapsed yet
        else:
            assert dtbc == parent_dt[cid]
    # FEEDBACK positions: child has exactly caught its parent
    for cid, ticks in rec["feedback"]:
        assert ticks % {2: 180, 3: 45, 4: 15}[cid] == 0


def test_executor_dormant_feedback_and_f13_bypass():
    """The dormant hook is dispatched at every table position -- the
    per-kid loop family in the interior period and the tail-relocated
    family in the final period (same positions on this linear chain);
    skip_feedback_path=True (F13 CLOSING run B) bypasses the call path
    while leaving STEP/FORCE and every clock identical."""
    exp = _chain_experiment((1, 3, 3), run_seconds=120.0)  # 2 periods
    sched = build_schedule(exp)
    calls = []

    def record(phase):
        return lambda cid, pid, child, parent: calls.append(
            (phase, cid, pid, child.ticks))

    report_a = execute_schedule(
        sched,
        on_feedback_prepare=record("prepare"),
        on_feedback_commit=record("commit"),
        on_feedback_finalize=record("finalize"))
    assert report_a.feedback_calls == 8  # 4 loop (period 0) + 4 tail
    transactions = [calls[i:i + 3] for i in range(0, len(calls), 3)]
    assert [[call[0] for call in txn] for txn in transactions] == [
        ["prepare", "commit", "finalize"]] * 8
    assert [(txn[0][1], txn[0][2]) for txn in transactions] == [
        (3, 2), (3, 2), (3, 2), (2, 1)] * 2
    assert [txn[0][3] for txn in transactions] == [
        60, 120, 180, 180, 240, 300, 360, 360]

    report_b = execute_schedule(
        sched, skip_feedback_path=True,
        on_feedback_prepare=lambda *args: pytest.fail("bypass leaked prepare"),
        on_feedback_commit=lambda *args: pytest.fail("bypass leaked commit"),
        on_feedback_finalize=lambda *args: pytest.fail("bypass leaked finalize"))
    assert report_b.feedback_calls == 0
    assert (report_b.steps, report_b.forces) == (report_a.steps,
                                                 report_a.forces)
    for gid, clock in report_b.clocks.items():
        assert clock.ticks == report_a.clocks[gid].ticks


def test_delayed_child_starts_on_forcing_seam_without_changing_parent():
    """A dormant child is absent from recursion and output until its seam.

    Activation occurs after the parent reaches the seam, so the production
    callback can initialize from that live parent state before the first
    child frame and solve.
    """
    base = _chain_experiment((1, 3), run_seconds=300.0)
    delayed = _with_delayed_start(base, 2, 120)
    schedule = build_schedule(
        delayed, resolve_clock(delayed, lbc_interval_s=60))
    starts = []
    history = {1: [], 2: []}
    steps = {1: [], 2: []}
    report = execute_schedule(
        schedule,
        on_domain_start=lambda gid, clock:
            starts.append((gid, clock.ticks)),
        on_history=lambda gid, ticks: history[gid].append(ticks),
        on_step=lambda gid, clock: steps[gid].append(clock.ticks))

    baseline_parent_steps = []
    baseline_parent_history = []
    execute_schedule(
        build_schedule(base, resolve_clock(base, lbc_interval_s=60)),
        on_step=lambda gid, clock: (
            baseline_parent_steps.append(clock.ticks) if gid == 1 else None),
        on_history=lambda gid, ticks: (
            baseline_parent_history.append(ticks) if gid == 1 else None))

    assert starts == [(2, 120)]
    assert history[2][0] == 120
    assert steps[2][0] == 120
    assert all(tick >= 120 for tick in history[2] + steps[2])
    assert steps[1] == baseline_parent_steps
    assert history[1] == baseline_parent_history
    assert report.clocks[2].step_count == 9


def test_resume_before_delayed_child_start_activates_it_once_at_seam():
    """A PERIOD_BEGIN checkpoint before child birth resumes the dormant
    clock, then runs the normal activation callback at the configured seam."""
    exp = _with_delayed_start(
        _chain_experiment((1, 3), run_seconds=300.0), 2, 120)
    schedule = build_schedule(exp, resolve_clock(exp, lbc_interval_s=60))
    clocks = {spec.grid_id: schedule.clock.domain_clock(spec.grid_id)
              for spec in schedule.clock.domains}
    # Exact state restored from a t=60 PERIOD_BEGIN checkpoint.  The child
    # clock was synchronized virtually but has executed no steps.
    for clock in clocks.values():
        clock.ticks = 60
    clocks[1].step_count = 1
    starts = []
    child_history = []
    report = execute_schedule(
        schedule, clocks=clocks, start_period=1, started_grid_ids=(1,),
        committed_initial_history_grid_ids=(1,),
        on_domain_start=lambda gid, clock:
            starts.append((gid, clock.ticks)),
        on_history=lambda gid, ticks: (
            child_history.append(ticks) if gid == 2 else None))

    assert starts == [(2, 120)]
    assert child_history[0] == 120
    assert report.clocks[2].step_count == 9


def test_subhour_boundary_cadence_and_delayed_start_constraints():
    accepted = _with_delayed_start(
        _chain_experiment((1, 3), run_seconds=1200.0), 2, 600)
    clock = resolve_clock(accepted, lbc_interval_s=300)
    assert clock.root.lbc_interval_ticks == 300
    assert clock.domain_clock(2).spec.start_ticks == 600

    not_a_forcing_seam = _with_delayed_start(
        _chain_experiment((1, 3), run_seconds=1200.0), 2, 600)
    with pytest.raises(ValueError, match=(
            r"d02.*boundary-forcing cadence.*420 s.*offset/cadence = 10/7")):
        resolve_clock(not_a_forcing_seam, lbc_interval_s=420)

    with pytest.raises(ValueError, match=(
            r"310 s.*whole number of root-domain steps.*cadence/dt = 31/6")):
        resolve_clock(
            _chain_experiment((1, 3), run_seconds=1200.0),
            lbc_interval_s=310)


@requires_4dom_inputs
def test_domain_clock_elapsed_and_dtbc_pins(bundle_clock):
    """Elapsed seconds stay tick-derived while dtbc mirrors WRF's REAL
    recurrence (dyn_em/solve_em.F:371-372)."""
    clock = bundle_clock.domain_clock(4)
    assert clock.spec.step_ticks == 5
    for _ in range(11):
        clock.advance()
    assert clock.ticks == 55 and clock.step_count == 11
    # exact FP64 rendering of 55/3 s -- NOT an accumulated float sum
    assert clock.elapsed_seconds == 18.333333333333332
    accumulated = np.float32(0)
    for _ in range(11):
        accumulated = accumulated + np.float32(clock.spec.dt_fp32)
    assert float(accumulated) != clock.elapsed_seconds

    clock2 = bundle_clock.domain_clock(4)
    for _ in range(9):
        clock2.advance()
    clock2.mark_force()
    # prepare_step is solve_em's pre-boundary recurrence: the kernel inside
    # the k-th solve after a force sees k*dt, never (k-1)*dt.
    assert clock2.dtbc == 0.0
    clock2.prepare_step()
    assert _fp32_bits(clock2.dtbc_launch_fp32) == 0x3FD55555   # 5/3 s
    clock2.advance()
    assert _fp32_bits(clock2.dtbc) == 0x3FD55555
    clock2.prepare_step()
    assert _fp32_bits(clock2.dtbc_launch_fp32) == 0x40555555   # 10/3 s
    clock2.advance()
    assert _fp32_bits(clock2.dtbc) == 0x40555555
    clock2.prepare_step()
    assert _fp32_bits(clock2.dtbc_launch_fp32) == 0x40A00000   # 5 s

    assert clock.history_due() is False  # 55 % 2700 != 0
    fresh = bundle_clock.domain_clock(4)
    assert fresh.history_due() is True   # t = 0 frame
    assert fresh.restart_due() is False  # children never restart-alarm
    assert fresh.lbc_reset_due() is False  # LBC seams are d01-only
    # Is_alarm_tstep position (module_domain.F:2503-2516): rings during
    # the step ENDING on the boundary -- the reflectivity-stash step.
    ring = bundle_clock.domain_clock(4)
    for _ in range(539):
        ring.advance()                   # ticks 2695: last step of the
    assert ring.history_rings_within_step() is True   # 15-min window
    assert ring.history_due() is False
    ring.advance()                       # ticks 2700: frame boundary
    assert ring.history_due() is True
    assert ring.history_rings_within_step() is False


@requires_4dom_inputs
def test_elapsed_seconds_fp32_mirrors_wrf_real_time(bundle_clock):
    """Shadow F3: WRF's REAL curr_secs is ``dt_whole + dt_num /
    REAL(dt_den)`` (dyn_em/adapt_timestep_em.F real_time; consumed by
    first-RK physics, dyn_em/solve_em.F:330-333/:800-815) -- TWO FP32
    roundings, one ULP above the FP64-then-cast value at d04's 5/3 s."""
    clock = bundle_clock.domain_clock(4)
    clock.advance()                      # the 2nd d04 solve sits at 5/3 s
    assert clock.elapsed_seconds == 1.6666666666666667
    assert _fp32_bits(clock.elapsed_seconds_fp32) == 0x3FD55556
    assert _fp32_bits(np.float32(clock.elapsed_seconds)) == 0x3FD55555
    # exactly representable instants agree between the two expressions
    clock.advance()
    clock.advance()                      # 5.0 s
    assert clock.elapsed_seconds_fp32 == np.float32(clock.elapsed_seconds)
    assert float(clock.elapsed_seconds_fp32) == 5.0
    root = bundle_clock.domain_clock(1)
    for _ in range(720):
        root.advance()
    assert float(root.elapsed_seconds_fp32) == 43200.0  # 12 h exact


@requires_4dom_inputs
def test_elapsed_seconds_fp32_enforces_exact_integer_guarantee_bound(
        bundle_clock):
    """D4: whole/num/den stay inside the conservative exact-integer bound."""
    clock = bundle_clock.domain_clock(1)

    # The inclusive 2**24 integer boundary is exactly representable.
    clock.ticks = (1 << 24) * clock.tick_den
    assert clock.elapsed_seconds_fp32 == np.float32(1 << 24)

    # 2**24 + 2 is exactly representable, but deliberately outside the
    # conservative bound.  The failure message must not claim otherwise.
    clock.tick_den = 1
    clock.ticks = (1 << 24) + 2
    assert int(np.float32(clock.ticks)) == clock.ticks
    with pytest.raises(AssertionError, match=r"whole=.*outside.*bound"):
        _ = clock.elapsed_seconds_fp32

    clock.tick_den = (1 << 24) + 3
    clock.ticks = (1 << 24) + 2
    assert int(np.float32(clock.ticks)) == clock.ticks
    with pytest.raises(AssertionError, match=r"num=.*outside.*bound"):
        _ = clock.elapsed_seconds_fp32

    clock.tick_den = (1 << 24) + 2
    clock.ticks = 1
    assert int(np.float32(clock.tick_den)) == clock.tick_den
    with pytest.raises(AssertionError, match=r"den=.*outside.*bound"):
        _ = clock.elapsed_seconds_fp32


@pytest.mark.parametrize("ratio", [3, 4, 7, 11])
def test_dtbc_recurrence_is_wrf_identical_for_general_ratios(ratio):
    """PROVENANCE D4 evidence: DomainClock executes WRF's REAL recurrence
    verbatim for nontrivial ratios, including the former 1:7 counterexample."""
    exp = _chain_experiment((1, ratio), run_seconds=60.0)
    clock = resolve_clock(exp).domain_clock(2)
    wrf = np.float32(0.0)
    for _ in range(ratio):
        wrf = np.float32(wrf + clock.spec.dt_fp32)
        clock.prepare_step()
        assert _fp32_bits(clock.dtbc_launch_fp32) == _fp32_bits(wrf)
        clock.advance()
    clock.mark_force()
    assert _fp32_bits(clock.dtbc_launch_fp32) == 0x00000000


def test_dtbc_recurrence_1to7_absolute_bit_anchors():
    """Merge-round retention of the shadow's absolute 1:7 anchors: the
    recurrence must land on WRF's accumulated bits, not merely track a
    local reimplementation of the same loop (review MINOR-1)."""
    exp = _chain_experiment((1, 7), run_seconds=60.0)
    clock = resolve_clock(exp).domain_clock(2)
    anchors = {5: 0x422B6DB6, 6: 0x424DB6DA, 7: 0x426FFFFE}
    for k in range(1, 8):
        clock.prepare_step()
        if k in anchors:
            assert _fp32_bits(clock.dtbc_launch_fp32) == anchors[k], k
        clock.advance()


def test_executor_detects_desync():
    """A hand-corrupted table trips the tick-exact sync assertions."""
    exp = _chain_experiment((1, 3, 3), run_seconds=120.0)
    sched = build_schedule(exp)
    # drop one d03 STEP: the FEEDBACK sync assert must fire
    broken = tuple(op for i, op in enumerate(sched.interior_period)
                   if not (i == 4 and op == Op(STEP, 3, 0)))
    bad = dataclasses.replace(sched, interior_period=broken)
    with pytest.raises(RuntimeError, match="sync violated"):
        execute_schedule(bad)


# ---------------------------------------------------------------------------
# clock_dt consumer pins (section C traced list)
# ---------------------------------------------------------------------------

@requires_4dom_inputs
def test_diff6_per_domain_plain_factor(bundle_exp):
    """With clock_dt retired (0.0), the clock-scaled diff_6th coefficient
    short-circuits to the plain per-domain factor 0.12/0.10/0.08/0.06 at
    each domain's own dt (core/dycore.py:841-854)."""
    from woof.core.dycore import _clock_scaled_diff6_factor

    expected = {1: 0.12, 2: 0.10, 3: 0.08, 4: 0.06}
    for dc in bundle_exp.domains:
        assert dc.run.clock_dt == 0.0
        factor = _clock_scaled_diff6_factor(dc.run)
        assert factor == dc.run.diff_6th_factor == expected[dc.grid_id]


@requires_4dom_inputs
def test_davies_weights_keyed_by_child_dt(bundle_exp, monkeypatch):
    """_resident_weights caches Davies fcx/gcx keyed by the child's own
    dt (ingest/lateral_bc.py:282-302); lateral_boundary_clock_dt resolves
    to run.dt for every experiment-path domain (:214-217).  cupy is
    shimmed to numpy so the keying/caching logic runs CPU-only."""
    from woof.ingest import lateral_bc

    monkeypatch.setitem(
        sys.modules, "cupy",
        types.SimpleNamespace(asarray=np.asarray, float32=np.float32))

    class _FakeResident:
        def __init__(self):
            self.weights = {}
            self.scratch_slots = set()
            self.device_nbytes = 0
            self.next_weight_slot = 0

    class _FakeState:
        def __init__(self):
            self._lateral_boundary_device = _FakeResident()

        def scratch(self, shape, slot):
            return np.zeros(shape, dtype=np.float32)

    state = _FakeState()
    for dc in bundle_exp.domains[1:]:
        dt = lateral_bc.lateral_boundary_clock_dt(dc.run)
        assert dt == dc.run.dt  # clock_dt retired in the experiment path
        fcx, gcx = lateral_bc._resident_weights(
            state, 5, 1, 4, np.float32(dt), 0.0, wrf_real=True)
        # nested branch of lbc_fcx_gcx (module_bc_em.F:1325-1339): no
        # sponge.  Preserve every default-REAL left-to-right round point.
        numerator = np.float32(3.0)
        denominator = np.float32(3.0)
        expected_f = np.float32(np.float32(np.float32(0.1) / np.float32(dt))
                                * numerator)
        expected_f = np.float32(expected_f / denominator)
        expected_g = np.float32(np.float32(1.0) / np.float32(dt))
        expected_g = np.float32(expected_g / np.float32(50.0))
        expected_g = np.float32(expected_g * numerator)
        expected_g = np.float32(expected_g / denominator)
        assert fcx[0] == 0.0                       # spec row
        assert fcx[1] == expected_f
        assert gcx[1] == expected_g

    # d04's chained 5/3 step is the discriminating one-ULP fixture.
    assert np.float32(dt).view(np.uint32) == np.uint32(0x3FD55555)
    assert gcx[1].view(np.uint32) == np.uint32(0x3C449BA5)

    resident = state._lateral_boundary_device
    # The key still carries the child dt, so a DIFFERENT dt still mints a
    # fresh entry rather than reusing another domain's weights -- which is
    # what the three distinct calls above proved by producing three
    # distinct fcx/gcx pairs.
    #
    # What is held here is the BOUND on how many of them a resident keeps
    # (ingest/lateral_bc._MAX_LBC_WEIGHT_SLOTS).  The old assertion of 3
    # measured an UNBOUNDED cache, and it read 3 only because this fixture
    # shares one fake resident across three domains; in production a
    # resident belongs to one domain, which under a fixed clock has one dt
    # for the life of the run.  Unbounded is not survivable under an
    # adaptive clock: dt moves almost every step, each distinct value
    # minted a fresh `lbc_weights_<n>` scratch slot, and the
    # canonical-state digest audits that prefix and knows only slot 0.
    # WRF recomputes these on every dt change for the same reason
    # (adapt_timestep_em.F:430 calls lbc_fcx_gcx), so a cache that never
    # hits is pure growth.
    assert lateral_bc._MAX_LBC_WEIGHT_SLOTS == 1
    assert len(resident.weights) == lateral_bc._MAX_LBC_WEIGHT_SLOTS
    # The survivor is the LAST dt asked for, and the slot it occupies is
    # slot 0 -- the only one the digest knows.
    assert ({key[3] for key in resident.weights}
            == {np.float32(bundle_exp.domains[-1].run.dt)})
    assert resident.scratch_slots == {"lbc_weights_0"}
    # same dt hits the cache instead of minting a new entry
    before = resident.device_nbytes
    lateral_bc._resident_weights(
        state, 5, 1, 4, np.float32(bundle_exp.domains[-1].run.dt), 0.0,
        wrf_real=True)
    assert len(resident.weights) == lateral_bc._MAX_LBC_WEIGHT_SLOTS
    # A recompute REUSES slot 0, so the reported device footprint must not
    # grow with it: an accounting that added the same buffer's bytes on
    # every dt change would report an unbounded VRAM figure for a cache
    # that is bounded by construction.
    assert resident.device_nbytes == before
    lateral_bc._resident_weights(
        state, 5, 1, 4, np.float32(bundle_exp.domains[1].run.dt), 0.0,
        wrf_real=True)
    assert len(resident.weights) == lateral_bc._MAX_LBC_WEIGHT_SLOTS
    assert resident.scratch_slots == {"lbc_weights_0"}
    assert resident.device_nbytes == before


@requires_4dom_inputs
def test_kf_calendar_d01_only(bundle_exp, bundle_clock):
    """KF's clock-defined dt resolves to the domain dt (core/kf.py:22-31)
    and the cumulus calendar exists on d01 only (bundle cu_physics)."""
    from woof.core import kf
    from woof.core import physics

    for dc in bundle_exp.domains:
        assert kf._model_clock_dt(dc.run) == dc.run.dt
        assert physics._model_clock_dt(dc.run) == dc.run.dt
    assert bundle_exp.root.run.cu_physics == 1
    assert bundle_clock.root.stepcu == 5
    for gid in (2, 3, 4):
        assert bundle_exp.domain(gid).run.cu_physics == 0
        assert bundle_clock.spec(gid).stepcu is None
        assert bundle_clock.spec(gid).cudt_ticks is None


# ---------------------------------------------------------------------------
# Code audit: the no-float-elapsed-accumulation rule (grep-able, L2 bait)
# ---------------------------------------------------------------------------

def _target_identifier(node):
    """The human-facing identifier of an assignment target."""
    while isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def test_no_float_elapsed_accumulation_audit():
    """AST audit of the executor-facing clock modules (section C).

    SCOPE (stated accurately, shadow F4): this is an AST-level tripwire
    over woof/core/clock.py -- plus woof/core/model.py once T14 lands
    it -- not a type-system proof.  It enforces:

    1. augmented assignment only ever targets the integer tick/op
       counters, and its VALUE expression contains no call and no float
       constant (closes the ``ticks += np.float32(...)`` escape);
    2. no assignment target carries an elapsed/seconds name; the sole
       permitted dtbc target is the registered FP32 WRF accumulator;
    3. reflection/dynamic mutation is banned outright: no
       setattr/getattr/delattr/vars/eval/exec identifiers, and no
       ``out=`` keyword anywhere (closes the setattr and ufunc-out
       escapes);
    4. the integer hot-path functions (clock advance, alarm predicates,
       the schedule walk, the period expansion) contain no division, no
       float constant, and no reference to numpy at all; dtbc reset and
       recurrence are separately source-pinned as FP32 operations;
    5. the elapsed-seconds derivation is a single tick division.

    LEGACY NOTE (tracked, not scanned): the single-domain integrator
    still float-accumulates ``state.elapsed_seconds += cfg.dt`` at
    woof/core/dycore.py:1332/:1438 -- the frozen pre-experiment path,
    slated for T14 replacement by DomainClock; pinned below so its
    removal (or growth) surfaces here and in PROVENANCE D4.
    """
    scan = [REPO / "woof" / "core" / "clock.py"]
    model_py = REPO / "woof" / "core" / "model.py"
    if model_py.exists():
        scan.append(model_py)

    allowed_augassign = {"ticks", "step_count", "steps", "forces",
                         "feedback_calls", "restarts", "lbc_resets",
                         # `period` is the adaptive walk's period index
                         # (clock.adaptive_periods).  It is the same class
                         # as the rest: an integer counter with no
                         # calendar meaning.  Under a FIXED clock the
                         # walk is `range(...)` and no accumulation
                         # happens at all; the adaptive walk cannot use a
                         # range because with a varying step the period
                         # COUNT is not known ahead of time -- ending the
                         # run is a condition on ticks, not a count.  The
                         # rule's actual teeth, contains_call_or_float on
                         # the value expression, still apply and `+= 1`
                         # passes them.
                         "period",
                         "histories"}
    forbidden_fragments = ("elapsed", "second", "dtbc")
    banned_names = {"setattr", "getattr", "delattr", "vars", "eval",
                    "exec", "__setattr__", "__dict__"}

    def contains_call_or_float(node):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                return True
            if isinstance(sub, ast.Constant) and isinstance(sub.value,
                                                            float):
                return True
        return False

    for path in scan:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.AugAssign):
                name = _target_identifier(node.target)
                assert name in allowed_augassign, (
                    f"augmented assignment to {name!r} in {path.name} -- "
                    "only integer tick/op counters may accumulate")
                assert not any(bad in name.lower()
                               for bad in forbidden_fragments)
                assert not contains_call_or_float(node.value), (
                    f"augmented assignment to {name!r} in {path.name} "
                    "computes its value through a call or float "
                    "constant -- integer literals/locals only")
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                targets = [node.target]
            for target in targets:
                name = _target_identifier(target).lower()
                forbidden = any(bad in name for bad in forbidden_fragments)
                assert not forbidden or name == "dtbc_fp32", (
                    f"assignment to {name!r} in {path.name} -- "
                    "elapsed values are tick-derived and dtbc_fp32 is the "
                    "only permitted recurrent float")
            if isinstance(node, ast.Name):
                assert node.id not in banned_names, (
                    f"{node.id!r} in {path.name}: reflection/dynamic "
                    "mutation is banned in clock modules")
            if isinstance(node, ast.Attribute):
                assert node.attr not in banned_names
            if isinstance(node, ast.keyword):
                assert node.arg != "out", (
                    f"ufunc out= mutation in {path.name} is banned in "
                    "clock modules")

    # hot-path rules bind on clock.py's known function set
    source = scan[0].read_text(encoding="utf-8")
    tree = ast.parse(source)
    hot = {"advance", "history_due",
           "history_rings_within_step", "restart_due", "lbc_reset_due",
           "at_stop_time", "execute_schedule", "full_ops", "period_ops",
           "op_counts", "expand", "integrate", "at_stop",
           # The two period sources the executor now draws from.  Both
           # are hot path and both must stay integer-only; the whole
           # point of splitting the controller out into
           # `on_period_steps` is that its REAL(4) arithmetic happens
           # OUTSIDE these, and what crosses back is an integer number
           # of ticks.  Audited so that stays true.
           "fixed_periods", "adaptive_periods"}
    audited = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in hot:
            audited.add(node.name)
            for sub in ast.walk(node):
                if isinstance(sub, ast.BinOp):
                    assert not isinstance(sub.op, ast.Div), (
                        f"division inside hot-path {node.name}()")
                if isinstance(sub, ast.Constant):
                    assert not isinstance(sub.value, float), (
                        f"float constant inside hot-path {node.name}()")
                if isinstance(sub, ast.Name):
                    assert sub.id != "np", (
                        f"numpy reference inside hot-path {node.name}()")
    assert audited == hot  # every hot-path function was found and walked

    # The elapsed-seconds derivation remains tick-exact.  dtbc is the one
    # WRF-recurrent FP32 state, with positive-zero reset and post-increment
    # recurrence pinned independently.
    assert "return self.ticks / self.tick_den" in source
    assert source.count("self.dtbc_fp32 = np.float32(0.0)") == 2
    # The recurrence now takes the LIVE dt, not the configured one.
    # Changed deliberately, and this pin is why it could not happen
    # quietly: dtbc drives the Davies relaxation and the nest boundary
    # weight, so under an adaptive clock accumulating spec.dt_fp32 forces
    # the lateral boundaries at one timestep while the interior
    # integrates at another.  That corrupts the boundary zone instead of
    # raising anything, and it is what killed the first live adaptive
    # runs.  Under a FIXED clock clock.dt_fp32 IS spec.dt_fp32 for the
    # whole run, so no existing trajectory moves -- gated at 9/9 wrfout
    # byte-identical with the feature off.
    assert ("self.dtbc_fp32 = np.float32(self.dtbc_fp32 + "
            "self.dt_fp32)" in source)
    assert "return self.dtbc_fp32" in source
    assert "last_force_ticks" not in source

    # dycore legacy accumulator pin (tracked for T14; do not grow)
    dycore_src = (REPO / "woof" / "core" / "dycore.py").read_text(
        encoding="utf-8")
    assert dycore_src.count("elapsed_seconds += ") == 2, (
        "woof/core/dycore.py's legacy float elapsed accumulators "
        "changed: update the T14 replacement plan and PROVENANCE D4")


def test_provenance_d4_registered():
    """D4 records recurrent dtbc as identical and retains clock scope."""
    text = (REPO / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "### D4" in text
    for needle in ("dyn_em/solve_em.F:371-372",
                   "share/mediation_force_domain.F:203-206",
                   "share/mediation_integrate.F:1522",
                   "dyn_em/adapt_timestep_em.F",
                   "frame/module_domain.F:2503-2516",
                   "implemented-identical", "dtbc_fp32",
                   "module_integrate.F:439-445"):
        assert needle in text, f"PROVENANCE D4 lost cite {needle}"
