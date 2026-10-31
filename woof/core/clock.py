"""Integer-tick clock hierarchy and flat schedule executor (Phase 5, L2).

Implements section C of the RATIFIED Phase-5 nesting architecture
(docs/superpowers/specs/2026-07-16-phase5-nesting-architecture.md, 2026-07-16
controller amendment): ``fractions.Fraction`` at CONFIG-RESOLUTION time
only, integer TICKS at runtime.

WRF ORACLE (verified against the bundled v4.6.1 source): the ARW clock is
exact integer-rational -- namelist ``time_step`` plus
``time_step_fract_num/den`` (Registry.EM_COMMON:2245-2246) resolve to the
root REAL dt (share/set_timekeeping.F:343) and the rational stepTime
interval (``WRFU_TimeIntervalSet(stepTime, S=, Sn=, Sd=)``,
share/set_timekeeping.F:359); a child's stepTime is the parent's rational
interval divided by ``parent_time_step_ratio`` EXACTLY, while the REAL
fed to dynamics/physics is the chained single-precision division
``grid%dt = parent%dt / ratio`` (share/set_timekeeping.F:366-368).
:func:`resolve_clock` takes each domain's exact rational dt from
:meth:`woof.experiment.ExperimentConfig.dt_exact`, derives ``tick_den`` =
LCM of the dt denominators, and runs everything after load on integer
ticks.  Bundle chain (1,4,3,3): tick = 1/3 s, step_ticks = 180/45/15/5,
12 h = 129,600 ticks -- 36 d04 steps of 5 ticks == 180 ticks == one d01
step, exactly, forever.

KERNEL dt: T1's chained-FP32 value is CONSUMED from the merged schema --
``DomainConfig.run.dt`` carries the binary64 image of WRF's REAL
``grid%dt`` (the chained division lives in woof/experiment.py, never
here).  :func:`resolve_clock` verifies the FP32 round trip
(``np.float32(run.dt)`` bit-identical) and validates it BIT-EXACTLY
against WRF's REAL construction -- root ``REAL(time_step) +
REAL(fract_num)/REAL(fract_den)`` (set_timekeeping.F:343), child
``parent%dt / ratio`` chained from the consumed parent value (:368); a
value one ULP off is a hard error.  The recomputation is validation
only; ``run.dt`` remains the stored authority.

FP32/FP64 POLICY (section C): seconds are NEVER accumulated in floating
point; :attr:`DomainClock.elapsed_seconds` is refreshed from ticks (one
FP64 division, no drift) and is the internal (interval-selection /
restart) authority.  The KERNEL-FACING running seconds mirror WRF's REAL
``curr_secs`` expression bit-for-bit
(:attr:`DomainClock.elapsed_seconds_fp32`): WRF derives it per step from
the rational clock interval as ``dt_whole + dt_num / REAL(dt_den)``
(dyn_em/adapt_timestep_em.F ``real_time``; built at dyn_em/
solve_em.F:330-333, consumed by first-RK physics :800-815, widened to
REAL*8 only afterwards :4752-4755) -- one ULP from the
FP64-then-cast value at times like d04's 5/3 s (0x3FD55556 vs
0x3FD55555).  :class:`DomainClock` mirrors WRF's REAL boundary-clock
state machine directly: ``dtbc_fp32`` resets to positive FP32 zero at
every nested force (share/mediation_force_domain.F:203-206) and, on the
root, at every external LBC interval seam (``IF (currentTime .EQ.
grid%this_bdy_time) grid%dtbc = 0.``, share/mediation_integrate.F:1522),
then recurs as ``fl32(dtbc_fp32 + dt_fp32)`` immediately before each
solve, matching dyn_em/solve_em.F:371-372 before the boundary consumers
at :932-952.  Because every nested window and root LBC interval starts
from a reset, this recurrence has no cross-interval drift and is
WRF-identical for every ratio chain (PROVENANCE D4).  Exact integer
ticks remain the only sync, alarm, elapsed-time, and restart authority.

AUDIT SCOPE: tests/test_clock.py audits this module's AST (and
woof/core/model.py once it lands): augmented assignments may target
only the integer tick/op counters and their value expressions may
contain no call and no float constant; setattr/getattr/eval/exec and
ufunc ``out=`` mutation are banned outright; the hot-path functions
contain no division, no float constant, and no numpy reference.  The sole
stored-float exception is the source-pinned FP32 dtbc reset/recurrence.
This is an AST-level tripwire, not a type-system proof.  The LEGACY
single-domain accumulators ``state.elapsed_seconds += cfg.dt``
(woof/core/dycore.py:1332/:1438) are outside this executor path and
tracked for T14 replacement (pinned by the audit test).

PHYSICS CALENDARS (section C): WRF computes STEPRA = nint(radt*60/dt),
min 1, per domain (phys/module_physics_init.F); the generalized
``woof.core.physics._physics_interval_steps`` rounds the exact quotient
the same way when handed the rational dt, and for fixed clocks
:func:`resolve_clock` asserts the exact and
WRF-REAL (nint on the chained-FP32 dt) paths agree.  Bundle: STEPRA =
12/12/12/36 for d01..d04; KF cudt = 5 d01 steps (900 ticks), d01 only;
bldt = 0 -> every step on each domain's own clock.  history/restart are
tick calendars validated as exact multiples of the owning domain's
step_ticks on fixed clocks. Adaptive history and physics calendars are
exact tick deadlines; the root shortens its step to each deadline in the
tree. Restart alarms evaluate on the d01 clock ONLY (section B).

SCHEDULE (section C): :func:`build_schedule` expands WRF's integration
recursion into a periodic flat per-d01-step op table; the EXECUTOR
(:func:`execute_schedule`) walks precomputed ops with integer comparisons
only -- no Fraction and no float arithmetic on the hot path (the
host-overhead mitigation).  The recursion transcribed, verified verbatim
from the bundled frame/module_integrate.F::

    DO WHILE ( .NOT. domain_clockisstopsubtime(grid) )              ! :328
       CALL solve_interface ( grid_ptr )                            ! :393
       CALL domain_clockadvance ( grid_ptr )                        ! :396
       ...
       CALL med_nest_force ( grid_ptr , grid_ptr%nests(kid)%ptr )   ! :416
       ...%stop_subtime = domain_get_current_time(grid)             ! :420-421
       ...
       CALL integrate ( grid_ptr%nests(kid)%ptr )                   ! :430
       ...
       IF ( .NOT. ( domain_clockisstoptime(head_grid              ) ! :439
              .OR.  domain_clockisstoptime(grid                   ) ! :440
              .OR.  domain_clockisstoptime(grid_ptr%nests(kid)%ptr) ! :441
                  ) ) THEN
         CALL med_nest_feedback ( grid_ptr , ... )                  ! :443
       END IF                                                       ! :445
    ...after the subtime loop, on every NESTED integrate() return:
    IF ( grid%id .EQ. 1 ) THEN                                      ! :509
       ... med_last_solve_io ...                                    ! :510
    ELSE
       should_do_last_io = domain_clockisstoptime( head_grid )      ! :513
       DO WHILE ( grid_ptr%id .NE. 1 )  ! zip up the tree           ! :515
          IF ( domain_clockisstoptime( grid_ptr ) ) ...= .TRUE.     ! :516
          grid_ptr => grid_ptr%parents(1)%ptr                       ! :519
       ENDDO
       IF ( should_do_last_io ) THEN                                ! :521
          CALL med_nest_feedback ( grid_ptr%parents(1)%ptr, grid )  ! :523
          ... med_last_solve_io ...                                 ! :524
       ENDIF                                                        ! :525

TERMINAL FEEDBACK: RELOCATED, NOT SUPPRESSED (F8 as corrected by the
p5t9 review round -- the ratified F8 wording shares the omission and is
being amended by the controller).  The per-kid guard (:439-445) skips
the LOOP feedback whenever the head grid, the parent, or the child is at
stop time; but WRF RE-ISSUES the feedback at the last-io tail
(:521-525): on every nested ``integrate()`` return, if the head grid or
any grid on the chain from the returning grid up to the head is at stop
time, ``med_nest_feedback(parent, grid)`` runs (:523) immediately before
that grid's last-io check (:524).  The guard exists precisely to avoid
DOUBLING these tail calls (comment at :507).  In the final d01 period
the head is at stop time from its own clockadvance onward, so every
child window of that period emits a TAIL feedback instead of a loop
feedback: for the bundle, 12x d04->d03, 4x d03->d02, 1x d02->d01 -- 17
per period in BOTH interior and final periods.  For a LINEAR chain the
tail op lands at the same flat-table position as the loop op, so the
final period's op sequence EQUALS the interior period's; for a
multi-child tree the positions differ (kid1's tail feedback precedes
kid2's window; the interior loop feedbacks all follow the last window).
Bundle: 53 STEPs + 17 FORCEs + 17 FEEDBACKs per period, 720 identical
periods.  Both this generator and the independent reference recursion in
tests/test_clock.py represent the guard AND the tail (pinned for an
interior AND the final period, on chains (1,4,3,3), (1,3,3), (1,5,3),
(1,4,4,2), and a branching tree where the orders differ).

DORMANT FEEDBACK OPS (Phase-5b activation contract): FEEDBACK ops --
both the loop family (:443) and the terminal tail family (:523) -- are
PRESENT in the table at the exact WRF call positions and INERT this
phase.  As in WRF, the caller dispatches the hook at every op and the
``feedback .NE. 0`` test lives INSIDE the mediation layer
(share/mediation_integrate.F:1035) -- here the handler's
``feedback_prepare``/``feedback_commit``/``feedback_finalize`` transaction
no-ops at feedback = 0, and the experiment loader rejects feedback != 0
until Phase 5b.  Activation flips the loader gate and fills the transaction
with the copy_fcn transliteration
(BOTH parity branches, share/interp_fcn.F:1463-1477 odd / :1565-1654
even); the schedule table -- including the final period's 17 tail
feedbacks -- is already complete for two-way coupling, so activation
changes the handler, never the table.  ``skip_feedback_path`` is the F13
CLOSING run-B switch: bypass the FEEDBACK call path entirely and
whole-output bytes must match run A's no-op execution.

HISTORY/RESTART POSITION: history alarms fire at the top of the step
before solve (the med_before_solve_io position, frame/
module_integrate.F:375), including the t = 0 frame; the run-stop frame is
the med_last_solve_io analogue fired after the walk.  For the bundle
cadence all four domain alarms coincide at 15-min sync ticks, so frames
are time-consistent.  Restart alarms fire at d01 top-of-step boundaries
plus the run-stop boundary, never at t = 0.  The root's external-LBC
dtbc reset fires at interval seams inside the same top-of-step position
(med_latbound_in at the tail of med_before_solve_io,
share/mediation_integrate.F:1522).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from math import lcm
from typing import Iterator, NamedTuple

import numpy as np

from woof.experiment import ExperimentConfig
from woof.config import CU_SCHEMES as _CU_SCHEMES

#: Op kinds of the flat schedule table (section C adjudication: the
#: transcribed recursion is the GENERATOR, the flat table the EXECUTOR).
STEP = "STEP"
FORCE = "FORCE"
FEEDBACK = "FEEDBACK"

#: The denominator an adaptive clock needs as a factor: WRF requantises
#: every adaptive interval to hundredths, so the tick lattice must hold
#: them exactly or the controller's dt cannot be represented at all.
_ADAPTIVE_TICK_DEN = 100

#: Binary32 represents every integer through 2**24 inclusive.  Larger even
#: integers can also be exactly representable, so this is a conservative
#: all-integers-exact bound, not the definition of FP32 representability.
_FP32_ALL_INTEGERS_EXACT_BOUND = 1 << 24


def _fp32_hex(value: np.float32) -> str:
    """Bit pattern of an FP32 scalar for fail-loud dt messages."""
    return f"0x{int(np.float32(value).view(np.uint32)):08X}"


class Op(NamedTuple):
    """One schedule table entry.

    ``kind`` is :data:`STEP`, :data:`FORCE`, or :data:`FEEDBACK`.
    ``grid_id`` is the ACTING domain: the stepping domain for STEP, the
    CHILD for FORCE/FEEDBACK (WRF's per-kid loops).  ``parent_id`` is the
    forcing/receiving parent for FORCE/FEEDBACK and 0 for STEP.
    """

    kind: str
    grid_id: int
    parent_id: int


@dataclass(frozen=True)
class DomainTicks:
    """Per-domain integer-tick timing spec (config-resolution output).

    All calendars are integer ticks on the domain's OWN clock, validated
    by :func:`resolve_clock` as exact multiples of ``step_ticks``.
    ``None`` marks an absent calendar (scheme not configured on this
    domain -- e.g. cumulus off d01 in the bundle; restart on children,
    which evaluate restart on the d01 clock only; ``lbc_interval_ticks``
    is the ROOT's external-boundary interval, the d01 dtbc-reset
    calendar, ``None`` on children and when no external LBC stream is
    configured).  ``dt_fp32`` is WRF's REAL kernel dt, consumed from
    T1's chained-FP32 ``run.dt``.
    """

    grid_id: int
    parent_id: int
    parent_time_step_ratio: int
    step_ticks: int
    dt_fp32: np.float32
    history_ticks: int
    restart_ticks: int | None
    radt_ticks: int | None
    stepra: int | None
    cudt_ticks: int | None
    stepcu: int | None
    bldt_ticks: int | None
    stepbl: int | None
    lbc_interval_ticks: int | None = None
    start_ticks: int = 0
    #: WRF's history_begin, in ticks after ``start_ticks``, already rounded
    #: up to this domain's step lattice (the alarm rings at the first step
    #: at or after its RingTime, external/esmf_time_f90/ESMF_Clock.F90).
    #: 0 writes the start frame, which is every experiment without one.
    history_begin_ticks: int = 0
    #: WRF's history_end, in ticks after ``start_ticks``: no history frame
    #: after it (the alarm stops once the clock passes its StopTime).
    history_end_ticks: int | None = None

    history_interval2_ticks: int | None = None
    history_interval_change_ticks: int = 0

    def next_history_offset(self, elapsed: int) -> int:
        """First history alarm strictly after elapsed (domain-relative ticks)."""
        origin = self.history_begin_ticks
        interval = self.history_ticks
        if self.history_interval2_ticks is not None:
            change = self.history_interval_change_ticks
            if elapsed >= change:
                origin, interval = change, self.history_interval2_ticks
            else:
                due = origin if elapsed < origin else origin + ((elapsed-origin)//interval+1)*interval
                if due > change:
                    return change + self.history_interval2_ticks
                return due
        return origin if elapsed < origin else origin + ((elapsed-origin)//interval+1)*interval

    def history_offset_due(self, elapsed: int) -> bool:
        """Whether a frame is due ``elapsed`` ticks after this domain's
        start: on the interval lattice anchored at the begin offset, and
        inside the begin/end window."""
        if elapsed < self.history_begin_ticks:
            return False
        if (self.history_end_ticks is not None
                and elapsed > self.history_end_ticks):
            return False
        if (self.history_interval2_ticks is not None
                and elapsed > self.history_interval_change_ticks):
            return (elapsed - self.history_interval_change_ticks) % self.history_interval2_ticks == 0
        return (elapsed - self.history_begin_ticks) % self.history_ticks == 0

    def first_history_ticks(self) -> int:
        """Ticks (absolute) of this domain's first history frame after
        its start frame -- or of its first frame at all when a begin
        offset suppresses the start frame."""
        if self.history_begin_ticks:
            return self.start_ticks + self.history_begin_ticks
        return self.start_ticks + self.next_history_offset(0)


@dataclass(frozen=True)
class TickClock:
    """The experiment-wide integer-tick timing authority.

    ``tick_den`` is the LCM of every domain's rational-dt denominator
    (ticks per second); ``run_ticks`` is the whole run in ticks, a
    multiple of the root's ``step_ticks`` (runs end on a d01 boundary).
    Domain specs are stored parent-before-child, matching
    ``ExperimentConfig.domains``.
    """

    tick_den: int
    run_ticks: int
    start_time: datetime
    domains: tuple[DomainTicks, ...]

    @property
    def root(self) -> DomainTicks:
        return self.domains[0]

    def spec(self, grid_id: int) -> DomainTicks:
        for spec in self.domains:
            if spec.grid_id == grid_id:
                return spec
        raise KeyError(f"no domain with grid_id={grid_id} in the clock; "
                       f"have {[s.grid_id for s in self.domains]}")

    def tick_exact(self) -> Fraction:
        """One tick as an exact rational second (diagnostics only --
        Fractions never enter the runtime path)."""
        return Fraction(1, self.tick_den)

    def dt_exact(self, grid_id: int) -> Fraction:
        """A domain's exact rational dt reconstructed from ticks."""
        return Fraction(self.spec(grid_id).step_ticks, self.tick_den)

    def domain_clock(self, grid_id: int) -> "DomainClock":
        return DomainClock(self.spec(grid_id), self.tick_den,
                           self.run_ticks)

    def clocks(self) -> dict[int, "DomainClock"]:
        """Fresh runtime clocks for every domain, keyed by grid_id."""
        return {spec.grid_id: DomainClock(spec, self.tick_den,
                                          self.run_ticks)
                for spec in self.domains}


class DomainClock:
    """Mutable per-domain runtime clock with integer calendar authority.

    ``advance()`` is WRF's ``domain_clockadvance`` (frame/
    module_integrate.F:396) on the tick lattice -- pure integer
    increments, no floating point ever (AST-audited).  Elapsed seconds
    are DERIVED from ticks on demand (one FP64 division), never
    accumulated.  The one deliberately recurrent floating value is WRF's
    kernel-facing REAL ``dtbc`` accumulator; it resets at every boundary
    interval and is not calendar authority.
    """

    __slots__ = ("spec", "tick_den", "run_ticks", "ticks", "step_count",
                 "dtbc_fp32", "step_ticks", "dt_fp32",
                 "adaptive_state")

    def __init__(self, spec: DomainTicks, tick_den: int, run_ticks: int):
        self.spec = spec
        self.tick_den = tick_den
        self.run_ticks = run_ticks
        self.ticks = 0
        self.step_count = 0
        self.dtbc_fp32 = np.float32(0.0)
        # THE LIVE STEP, distinct from the CONFIGURED one.
        #
        # ``spec.step_ticks`` is config-resolution output and stays frozen:
        # it is what the namelist asked for, and the prepared-cache and
        # restart identities are stated against it.  ``self.step_ticks`` is
        # what the model is integrating with RIGHT NOW.  Under a fixed
        # namelist clock they are equal for the whole run and every
        # consumer sees exactly what it saw before; under
        # ``use_adaptive_time_step`` only this one moves.
        #
        # The distinction has to be pushed to every reader, because
        # reading the configured value where the live one was meant is a
        # silent wrong answer rather than a crash -- a nest forced with
        # the wrong parent interval, or a diagnostic timestamped a step
        # off.  docs/ADAPTIVE-TIMESTEP.md section 9 carries the inventory.
        self.step_ticks = spec.step_ticks
        # THE LIVE KERNEL dt, on the same footing as step_ticks and for a
        # sharper reason: dt_fp32 is WRF's REAL grid%dt, and it drives the
        # BOUNDARY clock (dtbc, :407), the nest boundary interpolation
        # weight (core/nest.py:592) and the Davies relaxation
        # (ingest/lateral_bc.py:418).  Leaving it on the configured value
        # under an adaptive clock forces the lateral boundaries at one dt
        # while the interior integrates at another -- which corrupts the
        # boundary zone rather than raising anything, and is what killed
        # the first live adaptive runs.
        self.dt_fp32 = spec.dt_fp32
        #: The adaptive controller's memory, or None.  Parked on the CLOCK
        #: because that is the object relocation does not replace and the
        #: checkpoint writer can already see -- the controller itself
        #: lives on the driver, which the writer cannot reach.  None on
        #: every fixed-clock run, and absent from those checkpoints.
        self.adaptive_state = None

    def advance(self) -> None:
        """One model step: integer tick increment (clockadvance :396)."""
        self.ticks += self.step_ticks
        self.step_count += 1

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since start, REFRESHED from ticks (section C): one
        FP64 division, never accumulated in floating point.  This is the
        INTERNAL authority (interval selection, restart bookkeeping);
        kernel-facing consumers on the WRF-compatible path take
        :attr:`elapsed_seconds_fp32`."""
        return self.ticks / self.tick_den

    @property
    def elapsed_seconds_fp32(self) -> np.float32:
        """WRF's REAL ``curr_secs``, mirrored bit-for-bit.

        WRF derives the running seconds per step from the rational clock
        interval as REAL via ``dt_whole + dt_num / REAL(dt_den)``
        (dyn_em/adapt_timestep_em.F ``real_time``; the interval is built
        at dyn_em/solve_em.F:330-333 and consumed by first-RK physics at
        :800-815, widened to REAL*8 only afterwards, :4752-4755).  The
        same whole/num/den decomposition falls out of the integer tick
        state; the two FP32 roundings (division, then addition) differ
        from ``np.float32(elapsed_seconds)`` by one ULP at times like
        d04's 5/3 s (0x3FD55556 vs 0x3FD55555), so this expression -- not
        a cast of the FP64 property -- is the WRF-compatible launch
        value.  Scaling num/den by a common factor is guaranteed to leave the
        FP32 quotient unchanged when both integers have absolute value <=
        2**24 (and hence are exactly representable).  This is sufficient, not
        necessary: some larger integers are exactly representable and some
        rounded conversions can coincidentally preserve the quotient.  The
        defensive assertions below enforce the conservative bound for
        whole/num/den; current clock bounds satisfy it.
        """
        whole = self.ticks // self.tick_den
        num = self.ticks - whole * self.tick_den
        den = self.tick_den
        assert 0 <= num < den
        for label, value in (("whole", whole), ("num", num), ("den", den)):
            assert abs(value) <= _FP32_ALL_INTEGERS_EXACT_BOUND, (
                f"elapsed_seconds_fp32 {label}={value} is outside the "
                "guaranteed FP32 integer bound (requires |value| <= 2**24)")
        return (np.float32(whole)
                + np.float32(num) / np.float32(den))

    @property
    def at_stop_time(self) -> bool:
        """WRF ``domain_clockisstoptime`` on the tick lattice."""
        return self.ticks >= self.run_ticks

    def mark_force(self) -> None:
        """Reset the boundary-tendency clock (``grid%dtbc = 0.``).

        Fires at every nested force (share/mediation_force_domain.F:
        203-206; the fresh-force marker is ``abs(grid%dtbc) < 1e-4`` at
        dyn_em/solve_em.F:344) and, on the root, at every external LBC
        interval seam (``IF (currentTime .EQ. grid%this_bdy_time)
        grid%dtbc = 0.``, share/mediation_integrate.F:1522).
        """
        self.dtbc_fp32 = np.float32(0.0)

    def prepare_step(self) -> None:
        """Advance WRF's REAL boundary clock before a solve.

        This is the exact ``grid%dtbc = grid%dtbc + grid%dt`` recurrence
        at dyn_em/solve_em.F:371-372.  It runs after any force/LBC-seam
        reset and before the solve callback, so boundary consumers see the
        post-increment FP32 value for every ratio chain.
        """
        self.dtbc_fp32 = np.float32(self.dtbc_fp32 + self.dt_fp32)

    @property
    def dtbc(self) -> float:
        """WRF's stored REAL boundary clock, exposed as a Python float."""
        return float(self.dtbc_fp32)

    @property
    def dtbc_launch_fp32(self) -> np.float32:
        """The FP32 dtbc a boundary kernel launched DURING the current
        step must see.  :meth:`prepare_step` has already executed WRF's
        ``grid%dtbc = grid%dtbc + grid%dt`` recurrence
        (dyn_em/solve_em.F:371-372) before the relax/spec routines consume
        it (:932-952).  First three d04 launches after a force: 5/3, 10/3,
        5 s (0x3FD55555, 0x40555555, 0x40A00000)."""
        return self.dtbc_fp32

    def history_due(self) -> bool:
        """History alarm predicate at the top-of-step position
        (med_before_solve_io, frame/module_integrate.F:375) -- true at
        t = 0 (WRF writes the initial frame) and at every interval
        boundary on this domain's own clock."""
        return (self.ticks >= self.spec.start_ticks
                and self.spec.history_offset_due(
                    self.ticks - self.spec.start_ticks))

    def history_rings_within_step(self) -> bool:
        """WRF ``Is_alarm_tstep`` for the step NOW being solved
        (frame/module_domain.F:2503-2516, the crossing-alarm predicate;
        consumed as ``diag_flag`` in dyn_em/solve_em.F:356/:365): true
        when the history alarm rings at the END of the current step --
        the reflectivity-stash position (trunk ``_refl_10cm_due``).
        Integer arithmetic only; advisory helper for T14 so the stash
        predicate is not re-derived per consumer."""
        return (self.ticks >= self.spec.start_ticks
                and self.spec.history_offset_due(
                    self.ticks + self.step_ticks - self.spec.start_ticks))

    def restart_due(self) -> bool:
        """Restart alarm predicate: d01 clock only (children carry
        ``restart_ticks = None``), never at t = 0."""
        rt = self.spec.restart_ticks
        return rt is not None and self.ticks > 0 and self.ticks % rt == 0

    def lbc_reset_due(self) -> bool:
        """External-LBC dtbc reset predicate (root only): true at every
        boundary interval seam INCLUDING t = 0 (WRF's med_latbound_in
        runs in the first med_before_solve_io and zeroes dtbc whenever
        the current time equals the interval start,
        share/mediation_integrate.F:1522)."""
        it = self.spec.lbc_interval_ticks
        return it is not None and self.ticks % it == 0


def _cadence_ticks(label: str, seconds: Fraction, step_ticks: int,
                   tick_den: int, grid_id: int, *,
                   require_step_multiple: bool = True) -> int:
    """Resolve a positive cadence, requiring fixed-step alignment when used.

    Adaptive history alarms are exact tick deadlines. The adaptive driver
    lands on them by shortening the live step, independently of the
    nominal step used to build the domain.
    """
    ticks = seconds * tick_den
    if ticks.denominator != 1:
        raise ValueError(
            f"{label} = {seconds} s on domain grid_id={grid_id} is not a "
            f"whole number of ticks (tick = 1/{tick_den} s).")
    count = int(ticks)
    if count <= 0 or (require_step_multiple and count % step_ticks != 0):
        raise ValueError(
            f"{label} = {seconds} s on domain grid_id={grid_id} is not a "
            f"positive whole number of that domain's steps "
            f"({count} ticks, step_ticks = {step_ticks}).")
    return count


def resolve_clock(exp: ExperimentConfig, *,
                  lbc_interval_s: float | None = None,
                  live_born_children=(),
                  tick_den_floor: int | None = None) -> TickClock:
    """Resolve the experiment's exact-rational clock into integer ticks.

    ``fractions.Fraction`` appears HERE ONLY (config resolution, section
    C): per-domain rational dt from ``exp.dt_exact`` (WRF's exact
    rational division chain, share/set_timekeeping.F:366-368),
    divisibility validation, ``tick_den`` = LCM of the dt denominators,
    every calendar as exact ticks.  The chained-FP32 kernel dt is
    CONSUMED from ``run.dt`` (T1 owns the chain; woof/experiment.py) --
    verified as a bit-exact FP32 round trip AND validated bit-for-bit
    against WRF's REAL construction (root set_timekeeping.F:343, child
    :368 chained from the consumed parent value); the recomputation is
    validation only, ``run.dt`` stays the stored authority.

    ``lbc_interval_s`` is the ROOT's external lateral-boundary interval
    (the bundle's ERA5 stream: 21600 s), owned by the case-data
    configuration (T2); when given it becomes the d01 dtbc-reset
    calendar (share/mediation_integrate.F:1522), validated as an exact
    whole number of root steps.

    ``live_born_children`` is forwarded to
    :func:`woof.experiment.validate_boundary_timing`: the grid ids of
    children initialized from their live parent at their start, which
    the forcing-seam rule does not bind.
    """
    # Lazy: physics.py imports cupy at module scope; the clock itself is
    # pure-CPU config machinery.
    from woof.core.physics import _physics_interval_steps
    from woof.config import radiation_enabled
    from woof.experiment import validate_boundary_timing

    if lbc_interval_s is not None:
        if (not float(lbc_interval_s).is_integer()
                or int(lbc_interval_s) != lbc_interval_s):
            raise ValueError(
                f"lbc_interval_s = {lbc_interval_s!r} must be a whole "
                "number of seconds for the boundary-forcing clock.")
        validate_boundary_timing(
            exp, int(lbc_interval_s),
            source="runtime input catalog lbc_interval_s",
            live_born_children=live_born_children)

    dt_exact = {dc.grid_id: exp.dt_exact(dc.grid_id) for dc in exp.domains}
    tick_den = lcm(*(dt.denominator for dt in dt_exact.values()))
    if tick_den_floor is not None:
        # A tree that GROWS mid-run (a dormant nest spawning) re-mints its
        # clocks over a larger domain set and carries every live clock's
        # integer ticks across.  The tick has to mean the same second on
        # both sides, so the leg's denominator is the declared tree's
        # (woof.experiment.declared tick denominator), never the startup
        # subset's: with HRRR's 20 s root and a ratio-3 nest it was 1 then
        # 3, and a newborn born at 1800 ticks read model second 600.
        tick_den = lcm(tick_den, int(tick_den_floor))
    if any(dc.run.use_adaptive_time_step for dc in exp.domains):
        # An adaptive clock emits hundredths -- calc_dt requantises every
        # interval to n/100 (adapt_timestep_em.F:179-184) -- so the
        # lattice must be able to hold one.  Widening here rather than
        # refusing later is what lets a fixed-dt tree and its adaptive
        # twin share a configuration: with the feature off this line does
        # nothing, and every existing tick_den is unchanged.
        tick_den = lcm(tick_den, _ADAPTIVE_TICK_DEN)

    run_ticks_f = Fraction(exp.run_seconds) * tick_den
    if run_ticks_f.denominator != 1:
        raise ValueError(
            f"run_seconds = {exp.run_seconds!r} is not a whole number of "
            f"ticks (tick = 1/{tick_den} s).")
    run_ticks = int(run_ticks_f)

    specs: list[DomainTicks] = []
    step_by_id: dict[int, int] = {}
    fp32_by_id: dict[int, np.float32] = {}
    for dc in exp.domains:
        run = dc.run
        dt_ex = dt_exact[dc.grid_id]
        start_f = exp.domain_start_offset_exact(dc.grid_id) * tick_den
        if start_f.denominator != 1:
            raise ValueError(
                f"start_time for grid_id={dc.grid_id} is not a whole "
                f"number of ticks (tick = 1/{tick_den} s).")
        start_ticks = int(start_f)
        step_f = dt_ex * tick_den
        if step_f.denominator != 1:  # unreachable: tick_den is the LCM
            raise ValueError(
                f"dt = {dt_ex} s on grid_id={dc.grid_id} is not a whole "
                f"number of ticks (tick = 1/{tick_den} s).")
        step_ticks = int(step_f)
        if dc.parent_id != 0:
            parent_step = step_by_id[dc.parent_id]
            if parent_step != step_ticks * dc.parent_time_step_ratio:
                raise ValueError(
                    f"grid_id={dc.grid_id}: parent step {parent_step} "
                    f"ticks != child step {step_ticks} ticks x "
                    f"parent_time_step_ratio {dc.parent_time_step_ratio}.")

        # --- consume T1's chained-FP32 kernel dt (never re-derive) ------
        dt_fp32 = np.float32(run.dt)
        if float(dt_fp32) != run.dt:
            raise ValueError(
                f"run.dt = {run.dt!r} on grid_id={dc.grid_id} is not the "
                "binary64 image of an FP32 value: the kernel dt must be "
                "T1's chained-FP32 WRF REAL grid%dt "
                "(share/set_timekeeping.F:368, owned by "
                "woof/experiment.py).")
        # BIT-EXACT validation against WRF's REAL construction (shadow
        # review F2 -- a 1-ULP neighbour like 0x3FD55556 on d04 must be
        # rejected, never absorbed by a tolerance): root dt =
        # REAL(time_step) + REAL(fract_num)/REAL(fract_den)
        # (set_timekeeping.F:343); child dt = parent%dt / ratio (:368),
        # chained from the CONSUMED (already-validated) parent value.
        # The recomputation is validation only -- dt_fp32 from run.dt
        # stays the stored authority.
        if dc.parent_id == 0:
            expected = (np.float32(dc.time_step)
                        + np.float32(dc.time_step_fract_num)
                        / np.float32(dc.time_step_fract_den))
        else:
            expected = (fp32_by_id[dc.parent_id]
                        / np.float32(dc.parent_time_step_ratio))
        if dt_fp32.tobytes() != expected.tobytes():
            raise ValueError(
                f"run.dt = {run.dt!r} ({_fp32_hex(dt_fp32)}) on grid_id="
                f"{dc.grid_id} is not WRF's REAL dt: the chained "
                "single-precision construction "
                "(share/set_timekeeping.F:343/:368) gives "
                f"{float(expected)!r} ({_fp32_hex(expected)}); the "
                "kernel dt is T1's chained-FP32 value consumed "
                "bit-for-bit.")

        # --- physics calendars (section C), exact-tick validated --------
        radt_ticks = stepra = None
        if radiation_enabled(run):
            minutes = run.radt if run.radt > 0.0 else run.radt_minutes
            radt_ticks, stepra = _physics_calendar(
                "radt", minutes, dt_ex, step_ticks, tick_den, dc.grid_id,
                run.dt, _physics_interval_steps,
                adaptive=run.use_adaptive_time_step)
        cudt_ticks = stepcu = None
        # Every admitted scheme gets a calendar (woof.config.CU_SCHEMES);
        # 5 (Grell 3D) pins cudt_minutes = 0 like GF and New Tiedtke.
        if run.cu_physics != 0 and run.cu_physics in _CU_SCHEMES:
            # GF pins cudt_minutes = 0: the calendar records the every-step
            # cadence the physics driver actually runs, same as radt = 0.
            # New Tiedtke pins it for the same reason and one more -- its
            # RAINCV is a per-call rate with no persistence, so a hold
            # would reapply it every step (config.py's cu_physics == 16
            # arm). Adding 16 here is what makes the F14 three-way check
            # run for it at all; outside the tuple the scheme still runs,
            # on a cadence nothing validated.  96 (New SAS) honours cudt
            # as WRF does: its rates are formed over dt*stepcu and held.
            cudt_ticks, stepcu = _physics_calendar(
                "cudt", run.cudt_minutes, dt_ex, step_ticks, tick_den,
                dc.grid_id, run.dt, _physics_interval_steps,
                adaptive=run.use_adaptive_time_step)
        bldt_ticks = stepbl = None
        if (run.bl_pbl_physics != 0 or run.sf_sfclay_physics != 0
                or run.sf_surface_physics != 0):
            bldt_ticks, stepbl = _physics_calendar(
                "bldt", run.bldt, dt_ex, step_ticks, tick_den, dc.grid_id,
                run.dt, _physics_interval_steps,
                adaptive=run.use_adaptive_time_step)

        history_ticks = _cadence_ticks(
            "history_interval_s", Fraction(dc.history_interval_s),
            step_ticks, tick_den, dc.grid_id,
            require_step_multiple=not run.use_adaptive_time_step)
        history_begin_ticks, history_end_ticks = _history_window_ticks(
            dc, step_ticks, tick_den, adaptive=run.use_adaptive_time_step)

        # Restart alarms evaluate on the d01 clock ONLY (section B/C).
        restart_ticks = None
        if dc.parent_id == 0 and exp.restart_interval_s > 0.0:
            restart_ticks = _cadence_ticks(
                "restart_interval_s", Fraction(exp.restart_interval_s),
                step_ticks, tick_den, dc.grid_id)

        # External-LBC interval: the ROOT's dtbc-reset calendar
        # (med_latbound_in, share/mediation_integrate.F:1522).
        lbc_interval_ticks = None
        if dc.parent_id == 0 and lbc_interval_s is not None:
            lbc_interval_ticks = _cadence_ticks(
                "lbc_interval_s", Fraction(lbc_interval_s), step_ticks,
                tick_den, dc.grid_id)

        history_interval2_ticks = None
        history_interval_change_ticks = 0
        if run.history_interval2_s:
            history_interval2_ticks = _cadence_ticks(
                "history_interval2_s", Fraction(str(run.history_interval2_s)),
                step_ticks, tick_den, dc.grid_id,
                require_step_multiple=not run.use_adaptive_time_step)
            change = Fraction(str(run.history_interval_change_s)) * tick_den
            if change.denominator != 1 or (not run.use_adaptive_time_step and int(change) % step_ticks):
                raise ValueError("history_interval_change_s must lie on the model clock")
            history_interval_change_ticks = int(change)

        specs.append(DomainTicks(
            grid_id=dc.grid_id, parent_id=dc.parent_id,
            parent_time_step_ratio=dc.parent_time_step_ratio,
            step_ticks=step_ticks, dt_fp32=dt_fp32,
            history_ticks=history_ticks, restart_ticks=restart_ticks,
            radt_ticks=radt_ticks, stepra=stepra,
            cudt_ticks=cudt_ticks, stepcu=stepcu,
            bldt_ticks=bldt_ticks, stepbl=stepbl,
            lbc_interval_ticks=lbc_interval_ticks,
            start_ticks=start_ticks,
            history_begin_ticks=history_begin_ticks,
            history_end_ticks=history_end_ticks,
            history_interval2_ticks=history_interval2_ticks,
            history_interval_change_ticks=history_interval_change_ticks))
        step_by_id[dc.grid_id] = step_ticks
        fp32_by_id[dc.grid_id] = dt_fp32

    root_step = specs[0].step_ticks
    if run_ticks <= 0 or run_ticks % root_step != 0:
        raise ValueError(
            f"run_seconds = {exp.run_seconds!r} ({run_ticks} ticks) is "
            f"not a positive whole number of root-domain steps "
            f"(step_ticks = {root_step}); runs end on a d01 boundary.")

    return TickClock(tick_den=tick_den, run_ticks=run_ticks,
                     start_time=exp.start_time, domains=tuple(specs))


def _history_window_ticks(dc, step_ticks: int,
                          tick_den: int, *,
                          adaptive: bool = False) -> tuple[int, int | None]:
    """A domain's history begin/end offsets in ticks.

    With fixed steps the begin rounds UP to the domain's step lattice:
    WRF's alarm rings
    once at the first step whose time is at or after start + begin
    (external/esmf_time_f90/ESMF_Clock.F90, the RingTime test in
    ESMF_ClockAdvance) and then every interval from that step.  The end
    is a bound, not a ring time, so it is not rounded. With adaptive
    steps the driver lands exactly on the requested begin alarm.
    """
    begin = Fraction(str(dc.history_begin_s or 0.0))
    begin_ticks = begin * tick_den
    if begin_ticks.denominator != 1:
        raise ValueError(
            f"history_begin_s = {float(begin):g} on grid_id={dc.grid_id} is "
            f"not a whole number of ticks (tick = 1/{tick_den} s).")
    begin_ticks = int(begin_ticks)
    if not adaptive:
        begin_ticks = -(-begin_ticks // step_ticks) * step_ticks
    end = dc.history_end_s
    end_ticks = None
    if end is not None:
        end_f = Fraction(str(end)) * tick_den
        end_ticks = int(end_f)  # a bound: floor keeps "not after" exact
    return begin_ticks, end_ticks


def _physics_calendar(label: str, minutes: float, dt_ex: Fraction,
                      step_ticks: int, tick_den: int, grid_id: int,
                      dt_float: float, interval_steps, *,
                      adaptive: bool = False) -> tuple[int, int]:
    """One WRF minutes-cadence calendar as (ticks, steps).

    ``minutes <= 0`` is WRF's every-step convention (STEPRA/STEPCU/STEPBL
    = 1). Under adaptive stepping a positive cadence is an exact tick
    deadline and its initial driver count uses WRF rounding. Otherwise
    the cadence runs on WRF's whole steps, ``MAX(NINT(minutes*60/DT), 1)``
    (phys/module_physics_init.F), as wrf.exe runs it: ``cudt = 5`` at a
    45 s step is 7 steps (5.25 min).  The generalized
    ``_physics_interval_steps`` rounds the exact rational and the WRF-REAL
    path (nint on the chained-FP32 dt) must agree -- the runtime
    PhysicsDriver recomputes the nint form, so a disagreement would split
    the single timing authority (F14) and is a load error.  Before 2.9.10
    a cadence that was not a whole number of steps was refused here and
    in the loader, where wrf.exe runs it.
    """
    if minutes <= 0.0:
        return step_ticks, 1
    if adaptive:
        ticks = _cadence_ticks(label, Fraction(str(minutes)) * 60, step_ticks,
                               tick_den, grid_id, require_step_multiple=False)
        # The integer deadline governs adaptive calls. The initial driver
        # count retains WRF's nominal-step rounding until the live clock
        # sets its per-step due override.
        return ticks, interval_steps(minutes, dt_float)
    exact = interval_steps(minutes, dt_ex)
    nint = interval_steps(minutes, dt_float)
    if exact != nint:
        raise ValueError(
            f"{label} = {minutes} min on domain grid_id={grid_id}: WRF "
            f"nint of the exact division {exact} steps, WRF nint on the "
            f"FP32 dt {nint} steps -- the single timing authority "
            "requires both to agree (F14).")
    return nint * step_ticks, nint


# ---------------------------------------------------------------------------
# Schedule: flat op table expanded from the module_integrate.F recursion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Schedule:
    """Periodic flat op table (section C: generator = the transcribed WRF
    recursion, executor = this table).

    ``interior_period`` is one non-terminal d01 step's ops (loop-family
    FEEDBACKs, module_integrate.F:443); ``final_period`` is the last
    period, where the per-kid guard (:439-445) suppresses the loop
    feedbacks and the last-io tail (:521-525) re-issues them at the
    integrate()-return positions -- identical op sequence on linear
    chains, reordered on branching trees.  A run is ``periods - 1``
    interior walks plus one final walk.
    """

    clock: TickClock
    interior_period: tuple[Op, ...]
    final_period: tuple[Op, ...]
    period_tables: tuple[tuple[Op, ...], ...] | None = None
    #: ``(start_ticks, step_now) -> ops`` -- the SAME transcription of
    #: frame/module_integrate.F that produced the tables above, kept
    #: callable so an adaptive clock can re-expand one period at the
    #: steps the controller just chose.  Single-sourced deliberately: a
    #: second copy of that recursion is how the two would drift.
    expander: object | None = None

    @property
    def period_ticks(self) -> int:
        return self.clock.root.step_ticks

    @property
    def periods(self) -> int:
        return self.clock.run_ticks // self.period_ticks

    def period_ops(self, index: int) -> tuple[Op, ...]:
        if index < 0 or index >= self.periods:
            raise IndexError(f"period {index} outside 0..{self.periods - 1}")
        if self.period_tables is not None:
            return self.period_tables[index]
        if index == self.periods - 1:
            return self.final_period
        return self.interior_period

    def full_ops(self) -> Iterator[Op]:
        """Every op of the whole run, in execution order."""
        for index in range(self.periods):
            yield from self.period_ops(index)

    @staticmethod
    def op_counts(ops) -> dict[str, int]:
        counts = {STEP: 0, FORCE: 0, FEEDBACK: 0}
        for op in ops:
            counts[op.kind] = counts[op.kind] + 1
        return counts

    def table_hash(self) -> str:
        """SHA-256 over the canonical schedule serialization (the pin)."""
        digest = hashlib.sha256()
        digest.update(
            f"tick_den={self.clock.tick_den};"
            f"run_ticks={self.clock.run_ticks};"
            f"period_ticks={self.period_ticks};"
            f"periods={self.periods}\n".encode())
        tables = (("INTERIOR", self.interior_period),
                  ("FINAL", self.final_period))
        if self.period_tables is not None:
            tables = tuple(
                (f"PERIOD {index}", ops)
                for index, ops in enumerate(self.period_tables))
        for label, ops in tables:
            digest.update(f"[{label}]\n".encode())
            for op in ops:
                digest.update(
                    f"{op.kind} {op.grid_id} {op.parent_id}\n".encode())
        return digest.hexdigest()


def build_schedule(exp: ExperimentConfig,
                   clock: TickClock | None = None) -> Schedule:
    """Expand the WRF recursion into the periodic flat op table.

    Transcribes frame/module_integrate.F onto the tick lattice: the
    subtime loop (:328), solve + clockadvance (:393/:396), the per-kid
    FORCE loop (:411-423), the per-kid recursive integrate loop
    (:425-434), the per-kid FEEDBACK loop (:436-453) with the terminal
    stop-time guard (:439-445), AND the last-io tail (:507-526): on
    every nested integrate() return, if the head grid or any grid on the
    chain from the returning grid to the head is at stop time, the
    feedback is RE-ISSUED at :523 (the guard exists to avoid doubling
    these tail calls, comment :507).  So the final period carries the
    same 17 FEEDBACK ops as an interior period -- relocated to the tail
    positions, which coincide with the loop positions on linear chains
    and differ on branching trees.  Tick-exact parent-child sync is
    asserted after every period expansion, and periodicity is
    self-checked against a middle period.
    """
    if clock is None:
        clock = resolve_clock(exp)
    kids = {dc.grid_id: tuple(k.grid_id
                              for k in exp.children_of(dc.grid_id))
            for dc in exp.domains}
    parent_of = {dc.grid_id: dc.parent_id for dc in exp.domains}
    step_cfg = {spec.grid_id: spec.step_ticks for spec in clock.domains}
    starts = {spec.grid_id: spec.start_ticks for spec in clock.domains}
    head = exp.root.grid_id
    run_ticks = clock.run_ticks

    def expand(start_ticks: int, step_now=None) -> tuple[Op, ...]:
        # ``step_now`` is the LIVE per-domain step, or None for the
        # configured one.  Under a fixed clock every caller passes None
        # and this is the same function it always was; an adaptive clock
        # re-expands one period at a time with the steps the controller
        # just chose.  One period is exactly one root step, which is why
        # re-expansion at this granularity is WRF's own shape rather than
        # a reinterpretation of it (frame/module_integrate.F recurses per
        # root step).
        #
        # The tick-exact assertion at the tail is what makes this safe:
        # a child step that does not divide its parent's exactly leaves
        # the child off the boundary and is refused here, which is the
        # same requirement WRF states as
        # num_small_steps = CEILING(parent%dt / dt).
        step = dict(step_cfg) if step_now is None else dict(step_now)
        ticks = {gid: start_ticks for gid in step}
        ops: list[Op] = []
        active = {gid for gid in step if starts[gid] <= start_ticks}

        def at_stop(gid: int) -> bool:
            # domain_clockisstoptime on the tick lattice (:439-441).
            return ticks[gid] >= run_ticks

        def integrate(gid: int, stop_subtime: int) -> None:
            while ticks[gid] < stop_subtime:            # :328
                ops.append(Op(STEP, gid, 0))            # :393
                ticks[gid] += step[gid]                 # :396
                for kid in kids[gid]:                   # :411-423
                    if kid not in active:
                        continue
                    ops.append(Op(FORCE, kid, gid))
                for kid in kids[gid]:                   # :425-434
                    if kid not in active:
                        continue
                    integrate(kid, ticks[gid])
                for kid in kids[gid]:                   # :436-453
                    if kid not in active:
                        continue
                    if not (at_stop(head) or at_stop(gid)
                            or at_stop(kid)):           # :439-441 (F8)
                        ops.append(Op(FEEDBACK, kid, gid))  # :443
            # Last-io tail (:507-526): on every NESTED integrate()
            # return, zip up the tree; if the head grid or any grid on
            # the chain (the returning grid included) is at stop time,
            # feedback is RE-ISSUED here, immediately before the grid's
            # last-io check -- the loop guard above exists to avoid
            # doubling these tail calls (comment :507).
            if gid != head and gid in active:           # :509/:511
                last_io = at_stop(head)                 # :513
                walk = gid                              # :514
                while walk != head:                     # :515-520
                    if at_stop(walk):
                        last_io = True                  # :516-518
                    walk = parent_of[walk]              # :519
                if last_io:                             # :521
                    ops.append(Op(FEEDBACK, gid, parent_of[gid]))  # :523

        integrate(head, start_ticks + step[head])
        boundary = start_ticks + step[head]
        for gid in step:
            if gid not in active:
                ticks[gid] = boundary
        for gid, t in ticks.items():
            if t != boundary:
                raise RuntimeError(
                    f"tick-exact sync violated after period expansion: "
                    f"grid_id={gid} at {t} ticks, boundary {boundary}.")
        return tuple(ops)

    periods = run_ticks // step_cfg[head]
    if any(value for value in starts.values()):
        tables = tuple(
            expand(index * step_cfg[head]) for index in range(periods))
        return Schedule(
            clock=clock, interior_period=tables[0],
            final_period=tables[-1], period_tables=tables,
            expander=expand)
    interior = expand(0)
    final = expand(run_ticks - step_cfg[head])
    if periods >= 3:
        middle = expand((periods // 2) * step_cfg[head])
        if middle != interior:
            raise RuntimeError(
                "schedule is not periodic: a middle period differs from "
                "the first interior period.")
    return Schedule(clock=clock, interior_period=interior,
                    final_period=final, expander=expand)


def fixed_periods(schedule, start_period: int, period_ticks: int):
    """Today's walk: the precomputed tables, in order.

    Yields ``(period, ops, boundary)``.  Integer arithmetic only -- this
    is inside the AST-audited hot path.
    """
    for period in range(start_period, schedule.periods):
        yield (period, schedule.period_ops(period),
               (period + 1) * period_ticks)


def adaptive_periods(schedule, clocks, root_id: int, on_period_steps,
                     start_period: int = 0):
    """One period at a time, at whatever step the controller just chose.

    ``on_period_steps(period, clocks)`` runs BEFORE the expansion and is
    where the controller lives: it may set ``clock.step_ticks`` on any
    domain, and it is deliberately outside this generator so that every
    float the controller touches -- ``calc_dt``'s REAL(4) factor above
    all -- stays outside the integer-audited walk.  What crosses the
    boundary is an integer number of ticks and nothing else.

    A period is exactly one ROOT step, which is the granularity WRF's own
    recursion works at (frame/module_integrate.F integrates the head grid
    one step and recurses).  The run ends when the root reaches
    ``run_ticks``; with a varying step that is a condition, not a count,
    which is why this is a while loop where the fixed walk is a range.
    """
    root = clocks[root_id]
    run_ticks = schedule.clock.run_ticks
    # NUMBERED FROM WHERE THE RESUME STARTED, like fixed_periods.  This
    # counted from 0 on every resume, so a resumed run's period indices
    # collided with the ones already taken -- latent while the three
    # period callbacks read `clocks` rather than `period`, and a trap the
    # moment one of them reads the index.
    period = int(start_period)
    while root.ticks < run_ticks:
        on_period_steps(period, clocks)
        steps = {gid: dom.step_ticks for gid, dom in clocks.items()}
        for gid, value in steps.items():
            # A zero or negative step is an INFINITE LOOP inside
            # `integrate`'s `while ticks[gid] < stop_subtime`, which
            # presents as a wedged run rather than as a bug -- found by
            # a test that asserted the walk would not hang, and it did.
            # A controller that computes a nonsense dt (a CFL of zero
            # through a clamp of zero, say) must be refused here, by
            # name, on the period it happens.
            if value <= 0:
                raise ValueError(
                    f"adaptive step for grid_id={gid} is {value} ticks at "
                    f"period {period}; a step must be a positive number "
                    f"of ticks, and a non-positive one would hang the "
                    f"integration rather than fail it.")
        ops = schedule.expander(root.ticks, steps)
        yield period, ops, root.ticks + root.step_ticks
        period += 1


# ---------------------------------------------------------------------------
# Executor: precomputed op-table walk, integer comparisons only
# ---------------------------------------------------------------------------

@dataclass
class ExecutionReport:
    """Counters and final clocks from one schedule walk."""

    steps: int = 0
    forces: int = 0
    feedback_calls: int = 0
    restarts: int = 0
    lbc_resets: int = 0
    histories: dict[int, int] = field(default_factory=dict)
    clocks: dict[int, DomainClock] = field(default_factory=dict)


def execute_schedule(schedule: Schedule, *,
                     on_step=None, on_force=None,
                     on_feedback_prepare=None, on_feedback_commit=None,
                     on_feedback_finalize=None,
                     on_history=None, on_restart=None, on_lbc_reset=None,
                     on_period_begin=None, on_period_end=None,
                     on_period_commit=None,
                     on_domain_start=None,
                     skip_feedback_path: bool = False,
                     clocks: dict[int, DomainClock] | None = None,
                     start_period: int = 0,
                     started_grid_ids=None,
                     committed_initial_history_grid_ids=(),
                     on_period_steps=None,
                     stop_ticks: int | None = None
                     ) -> ExecutionReport:
    """Walk the flat op table with integer-tick clocks (the hot loop).

    Every comparison is integer arithmetic on precomputed tick calendars
    -- no Fraction, no float division, no float accumulation (AST-audited
    by tests/test_clock.py).  Handlers receive the acting grid_id and its
    live :class:`DomainClock` (plus the parent clock for FORCE/FEEDBACK):

    ``stop_ticks`` returns after committing that complete synchronized
    period, retaining the configured full-run calendar. A later invocation
    resumes with the same clocks and its committed boundary histories marked
    in ``committed_initial_history_grid_ids``. It never manufactures a final
    period or rewrites the run's configured stop. A requested boundary that an
    adaptive period would cross is refused before that period executes.

    - ``on_history(grid_id, ticks)`` fires before the next solve at that
      instant (med_before_solve_io position, module_integrate.F:375),
      including t = 0.  Alarms reached at a complete d01 synchronization
      boundary fire tree-wide after the child recursion/feedback block has
      returned.  This is the true period boundary: every due history product
      is consumed before a same-instant restart can publish.
    - ``on_restart(ticks)`` fires on the d01 clock only, after all domains
      have reached the interval boundary and all same-instant history
      handlers have completed, never at t = 0.
    - ``on_lbc_reset(ticks)`` fires on the root at external-boundary
      interval seams (including t = 0, the first interval read) in the
      same top-of-step position (med_latbound_in at the tail of
      med_before_solve_io); the executor resets the root's dtbc clock
      there (``grid%dtbc = 0.``, share/mediation_integrate.F:1522), so
      the first post-seam launch sees one root step of dtbc.
    - ``on_step(grid_id, clock)`` is the solve; the clock advances after
      it (solve_interface :393 then domain_clockadvance :396).  NEST
      boundary kernels launched from inside the solve take
      ``clock.dtbc_launch_fp32`` -- the post-increment value WRF's
      solve_em applies at :371-372 before consuming it at :932-952 --
      and WRF-compatible running seconds from
      ``clock.elapsed_seconds_fp32``.  Davies clock bind (2026-07-28,
      retires the F20 adjudication): the ROOT's external Davies
      launches consume ``dtbc_launch_fp32`` too -- the tree build binds
      the root boundary clock to the external mirror
      (``bind_lateral_boundary_clock``), so the ``on_lbc_reset`` seam
      reset above plus ``prepare_step()`` reproduce WRF's exact
      reset/post-increment recurrence for root relaxation, held moist
      targets, and the final specified-ring overwrite (old record at
      ``dtbc = T_bdy`` on the pre-seam step).  The retired elapsed-based
      consumption survives only on unbound legacy direct paths; the
      pre-bind Phase-4 anchor bytes encode it, and the N-series ratchets
      regenerate against the seam-closure anchor epoch (PROVENANCE.md
      "Root external-boundary dtbc").
    - ``on_force(child_id, parent_id, child_clock, parent_clock)``: the
      parent is exactly one parent-step ahead (asserted tick-exact); the
      child's dtbc resets to 0 afterwards (mediation_force_domain.F:
      203-206).
    - ``on_feedback_prepare(...)``, ``on_feedback_commit(...)``, then
      ``on_feedback_finalize(...)`` form one DORMANT transaction at each
      exact WRF feedback position -- the per-kid loop family (:443,
      interior periods) and the last-io tail family (:523, final period).
      Missing handlers are no-ops at feedback = 0.  Phase 5b activates
      the three handlers, not the table; ``feedback_calls`` counts
      transactions, not individual phases.  ``skip_feedback_path=True``
      is the F13 CLOSING run-B bypass.
    """
    if clocks is None:
        clocks = schedule.clock.clocks()
    else:
        clocks = dict(clocks)
        expected = {spec.grid_id for spec in schedule.clock.domains}
        if set(clocks) != expected:
            raise ValueError(
                f"executor clocks {sorted(clocks)} do not match schedule "
                f"domains {sorted(expected)}")
    if start_period < 0 or start_period >= schedule.periods:
        raise ValueError(
            f"start_period {start_period} outside 0..{schedule.periods - 1}")
    start_ticks = start_period * schedule.period_ticks
    for gid, dom in clocks.items():
        if dom.ticks != start_ticks:
            raise ValueError(
                f"executor clock grid_id={gid} starts at {dom.ticks} ticks, "
                f"expected PERIOD_BEGIN {start_ticks} for period "
                f"{start_period}")
    report = ExecutionReport(
        histories={gid: 0 for gid in clocks},
        clocks=clocks)
    period_ticks = schedule.period_ticks
    run_ticks = schedule.clock.run_ticks
    if stop_ticks is not None:
        if isinstance(stop_ticks, bool) or not isinstance(stop_ticks, int):
            raise TypeError("stop_ticks must be an integer clock tick")
        if not start_ticks < stop_ticks <= run_ticks:
            raise ValueError("stop_ticks must follow the start and not exceed the configured stop")
        if on_period_steps is None and stop_ticks % period_ticks:
            raise ValueError("stop_ticks must be a complete period boundary")
    target_ticks = run_ticks if stop_ticks is None else stop_ticks
    root_id = schedule.clock.root.grid_id
    restart_enabled = clocks[root_id].spec.restart_ticks is not None
    if started_grid_ids is None:
        started = {
            gid for gid, dom in clocks.items()
            if dom.spec.start_ticks == 0}
    else:
        started = {int(gid) for gid in started_grid_ids}
    unknown_started = started - set(clocks)
    if unknown_started:
        raise ValueError(
            f"started domains are not in the schedule: "
            f"{sorted(unknown_started)}")
    required_started = {
        gid for gid, dom in clocks.items()
        if dom.spec.start_ticks < start_ticks}
    missing_started = required_started - started
    if missing_started:
        raise ValueError(
            f"domains whose configured starts precede PERIOD_BEGIN "
            f"{start_ticks} are not marked started: "
            f"{sorted(missing_started)}")

    committed_initial = frozenset(int(gid)
                                  for gid in committed_initial_history_grid_ids)
    unknown = committed_initial - set(clocks)
    if unknown:
        raise ValueError(
            "committed initial history domains are not in the schedule: "
            f"{sorted(unknown)}")
    last_history_ticks: dict[int, int] = {}

    def emit_history(grid_id: int) -> None:
        dom = clocks[grid_id]
        if (grid_id not in started or not dom.history_due()
                or last_history_ticks.get(grid_id) == dom.ticks):
            return
        last_history_ticks[grid_id] = dom.ticks
        if (dom.ticks == start_ticks and start_ticks > 0
                and grid_id in committed_initial):
            return
        report.histories[grid_id] += 1
        if on_history is not None:
            on_history(grid_id, dom.ticks)

    # Fresh t=0 frames and resume-boundary ownership are decided for every
    # domain before any solve.  A fixed checkpoint needs no serialized REFL
    # handoff: restore marks exactly the already-durable, due domain frames as
    # committed, and these initial callbacks are suppressed one domain at a
    # time.
    for gid in clocks:
        dom = clocks[gid]
        if (gid not in started and dom.spec.start_ticks == start_ticks):
            if on_domain_start is not None:
                on_domain_start(gid, dom)
            started.add(gid)
        emit_history(gid)

    if on_period_steps is None:
        periods = fixed_periods(schedule, start_period, period_ticks)
    else:
        periods = adaptive_periods(schedule, clocks, root_id,
                                   on_period_steps, start_period)
    for period, ops, boundary in periods:
        if boundary > target_ticks:
            raise ValueError("next complete period would cross stop_ticks")
        if on_period_begin is not None:
            on_period_begin(period, clocks)
        for op in ops:
            if op.kind == STEP:
                dom = clocks[op.grid_id]
                # Non-boundary child alarms can occur within a ratio block.
                # Equal-tick boundary alarms were emitted either before the
                # first solve or at the preceding complete period boundary.
                emit_history(op.grid_id)
                if dom.lbc_reset_due():
                    # med_latbound_in tail of med_before_solve_io: the
                    # root's dtbc zeroes at the new boundary interval
                    # (share/mediation_integrate.F:1522).
                    report.lbc_resets += 1
                    dom.mark_force()
                    if on_lbc_reset is not None:
                        on_lbc_reset(dom.ticks)
                dom.prepare_step()
                if on_step is not None:
                    on_step(op.grid_id, dom)
                dom.advance()
                report.steps += 1
            elif op.kind == FORCE:
                child = clocks[op.grid_id]
                parent = clocks[op.parent_id]
                if parent.ticks - child.ticks != parent.step_ticks:
                    raise RuntimeError(
                        f"nest-force sync violated: parent "
                        f"{op.parent_id} at {parent.ticks} ticks must "
                        f"lead child {op.grid_id} at {child.ticks} ticks "
                        f"by exactly {parent.step_ticks} ticks.")
                if on_force is not None:
                    on_force(op.grid_id, op.parent_id, child, parent)
                child.mark_force()
                report.forces += 1
            else:  # FEEDBACK (dormant; F8-guarded at generation time)
                child = clocks[op.grid_id]
                parent = clocks[op.parent_id]
                if child.ticks != parent.ticks:
                    raise RuntimeError(
                        f"feedback sync violated: child {op.grid_id} at "
                        f"{child.ticks} ticks has not caught parent "
                        f"{op.parent_id} at {parent.ticks} ticks.")
                if not skip_feedback_path:
                    report.feedback_calls += 1
                    if on_feedback_prepare is not None:
                        on_feedback_prepare(op.grid_id, op.parent_id,
                                            child, parent)
                    if on_feedback_commit is not None:
                        on_feedback_commit(op.grid_id, op.parent_id,
                                           child, parent)
                    if on_feedback_finalize is not None:
                        on_feedback_finalize(op.grid_id, op.parent_id,
                                             child, parent)
        for gid, dom in clocks.items():
            if gid not in started:
                dom.ticks = boundary
        for gid, dom in clocks.items():
            if dom.ticks != boundary:
                raise RuntimeError(
                    f"tick-exact sync violated after period {period}: "
                    f"grid_id={gid} at {dom.ticks} ticks, boundary "
                    f"{boundary}.")
        # WRF returns from the complete parent/child recursion before the
        # next boundary solve begins.  Consume every same-instant output
        # product now, while all clocks and states describe this boundary.
        for gid, dom in clocks.items():
            if (gid not in started
                    and dom.spec.start_ticks == boundary):
                if on_domain_start is not None:
                    on_domain_start(gid, dom)
                started.add(gid)
        for gid in clocks:
            emit_history(gid)
        if on_period_end is not None:
            on_period_end(period, clocks)
        root = clocks[root_id]
        if restart_enabled and root.restart_due():
            report.restarts += 1
            if on_restart is not None:
                on_restart(root.ticks)
        if on_period_commit is not None:
            on_period_commit(period, clocks)
        if boundary == target_ticks:
            break

    for gid, dom in clocks.items():
        if dom.ticks != target_ticks:
            raise RuntimeError(
                f"run did not end on the stop boundary: grid_id={gid} at "
                f"{dom.ticks} ticks, target_ticks {target_ticks}.")
    return report


__all__ = [
    "STEP", "FORCE", "FEEDBACK", "Op", "DomainTicks", "TickClock",
    "DomainClock", "Schedule", "ExecutionReport", "resolve_clock",
    "build_schedule", "execute_schedule",
]
