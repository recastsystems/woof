"""WRF's adaptive time-step controller (``dyn_em/adapt_timestep_em.F``).

Ported against the Fortran, not against a description of it, and graded by
a captured oracle (``tools/calc_dt_wrf_oracle/``) at ``max_ulp == 0``.

TWO KINDS OF ARITHMETIC LIVE HERE, and keeping them apart is the whole
correctness argument.

* The **factor** is REAL(4).  ``calc_dt`` computes it in single precision
  and then quantises it to an integer numerator over ``precision`` (100).
  Every operation below that touches ``factor`` is done in ``np.float32``
  in upstream's exact order, because the quantisation ``INT(x*100 + 0.5)``
  sits right on a tie for values like 1.51 -- whose float32 image is
  1.50999999, so ``INT(150.999999 + 0.5)`` is 151 and not the 152 a
  decimal reading would give.  That is not a rounding curiosity: it is one
  part in 100 of every subsequent timestep.

* The **interval** is exact rational.  ESMF's ``ESMF_BaseTimeQuotI8``
  (external/esmf_time_f90/ESMF_BaseTime.F90:183) is integer arithmetic --
  ``n = S*Sd + Sn; d = Sd*divisor``, then ``simplify`` (gcd) and
  ``normalize_basetime`` -- so ``last_dtInterval * num / den`` carries no
  floating-point error at all.  :class:`fractions.Fraction` reproduces it
  exactly, which is also what ``woof.core.clock`` already uses at
  config-resolution time.  So a varying dt does NOT push WOOF-ArWen's clock
  into floating point; it only requires ``tick_den`` to carry a factor of
  ``precision``.

Fortran operator precedence matters in ``last_dtInterval * num / den``:
that is ``(last * num) / den``, multiply first.  Both steps are exact, so
the grouping does not change the value -- but it does change which
intermediate could overflow ESMF's INTEGER(8), and the port keeps the
same order for that reason.

Graded against **WRF 4.8.0**, ``dyn_em/adapt_timestep_em.F`` sha256
9d5f1a18ee7b22d676f9747ceeb67a74f71cd4e373b0c926cd015aa3565744f5 -- the
digest rather than a path, so the reference is checkable from any
checkout.  A moving-nest WRF tree differs from a stationary one there by
a local restart patch; ``calc_dt`` itself is identical in both.
"""

from __future__ import annotations

from fractions import Fraction
from math import floor

import numpy as np


def nint(x: float) -> int:
    """Fortran ``NINT``: round half AWAY from zero.

    Python's ``round`` and ``numpy.rint`` are round-half-to-EVEN, and the
    two disagree on every exact half.  The oracle grades ``calc_dt``,
    which uses upstream's ``INT(x + 0.5)`` directly, so nothing was
    grading the three sites that reached for ``rint`` instead -- and one
    of them takes a float32, which lands exactly on a half for real
    inputs (dt = 3.125 s gives 312.5: upstream 313, banker's 312).
    """
    return int(floor(x + 0.5)) if x >= 0 else -int(floor(-x + 0.5))

#: WRF's ``INTEGER, PARAMETER :: precision = 100`` (adapt_timestep_em.F:44).
#: Every timestep the controller produces is a rational with this
#: denominator, which is what keeps an integer-tick clock exact.
PRECISION = 100

#: ``calc_dt``'s "the CFL is essentially zero" cut.  Below it the routine
#: ignores the CFL entirely and grows by the full increase factor.  This
#: branch is why a restart that has lost its CFL memory overshoots -- see
#: docs/ADAPTIVE-TIMESTEP.md §5.
NEGLIGIBLE_CFL = 0.001

#: The floor upstream clamps a reducing factor to.  BPR's comment gives
#: the reason as ``normalize_basetime`` refusing a negative denominator,
#: not CFL robustness: the dampened formula goes negative once
#: ``max_cfl > 3*target_cfl``.
MIN_FACTOR = 0.1


def _f32(value) -> np.float32:
    return np.float32(value)


def _quantise(factor: np.float32, precision: int) -> int:
    """WRF's ``num = INT(factor * precision + 0.5)``.

    ``INT`` truncates toward zero, and the arithmetic is REAL(4): the
    integer ``precision`` is promoted to REAL(4) and the ``0.5`` literal is
    a default real.  Doing this in float64 disagrees with upstream on
    exactly the ties that matter (1.51 -> 151, not 152).
    """
    scaled = _f32(_f32(factor) * _f32(precision)) + _f32(0.5)
    return int(np.trunc(np.float32(scaled)))


def calc_dt(last_dt: Fraction, max_cfl: float, max_increase_factor: float,
            target_cfl: float, precision: int = PRECISION) -> Fraction:
    """``SUBROUTINE calc_dt`` (adapt_timestep_em.F:451-513), exactly.

    ``last_dt`` and the result are exact rationals in seconds.  The three
    reals are REAL(4) on the Fortran side, so they are narrowed here
    before use rather than after -- a float64 ``target_cfl`` of 1.2 is not
    the same number as the REAL(4) 1.2 the namelist gives WRF.
    """
    cfl = _f32(max_cfl)
    target = _f32(target_cfl)

    if cfl < _f32(NEGLIGIBLE_CFL):
        # Grow by the maximum allowable amount; the CFL is not consulted.
        num = _quantise(_f32(max_increase_factor), precision)
    elif cfl > target:
        # Reduce, undershooting target by half the excess -- upstream's
        # comment: "This tends to keep the model more stable."
        factor = _f32(
            _f32(target - _f32(_f32(0.5) * _f32(cfl - target))) / cfl)
        factor = _f32(max(_f32(MIN_FACTOR), factor))
        num = _quantise(factor, precision)
    else:
        # At or under target: grow linearly.
        num = _quantise(_f32(target / cfl), precision)

    # (last * num) / den, both steps exact -- ESMF_BaseTimeQuotI8.
    return Fraction(last_dt * num, precision)


def real_time_fp32(dt: Fraction) -> np.float32:
    """WRF's ``real_time`` (adapt_timestep_em.F:517-543), exactly.

        out_time = dt_whole + dt_num / REAL(dt_den)

    in REAL(4), and this is the value that becomes ``grid%dt`` at :404 --
    the number every kernel and every boundary consumer sees.  Built by
    that construction rather than by casting ``float(Fraction)``, because
    the two differ in the last bit for intervals like 47/100 s and that
    bit is compared against WRF's chained-FP32 dt elsewhere in this
    package to the ULP.
    """
    whole = int(dt)                       # truncation, as INT does
    frac = dt - whole
    if frac == 0:
        return _f32(whole)
    return _f32(_f32(whole)
                + _f32(_f32(frac.numerator) / _f32(frac.denominator)))


def limit_increase(dt: Fraction, last_dt: Fraction,
                   max_increase_factor: float,
                   precision: int = PRECISION) -> Fraction:
    """The growth bound applied AFTER ``calc_dt`` (adapt_timestep_em.F:170-177).

    Upstream uses ``NINT`` here and ``INT(x + 0.5)`` inside ``calc_dt``.
    For the positive values these take the two agree, but they are written
    differently upstream and are kept different here so a future negative
    or NaN input diverges where upstream diverges rather than silently
    agreeing.  Both are round-half-AWAY-from-zero; ``numpy.rint``, which
    stood here, is round-half-to-even.
    """
    num = nint(float(np.float64(_f32(max_increase_factor)) * precision))
    bound = Fraction(last_dt * num, precision)
    return bound if dt > bound else dt


def requantise(dt: Fraction, precision: int = PRECISION) -> Fraction:
    """``dt = real_time(dtInterval); num = NINT(dt*precision)`` (:179-184).

    Upstream's stated reason is overflow: without this the denominator
    compounds every step.  It also happens to be what makes every adaptive
    timestep a clean ``n/100`` s, and therefore representable exactly on an
    integer-tick clock.

    ``real_time`` returns REAL(4), so the round trip is through single
    precision and is genuinely lossy for a long interval -- reproduced
    rather than improved.
    """
    whole = int(dt)
    frac = dt - whole
    as_real4 = _f32(whole) + _f32(_f32(frac.numerator) / _f32(frac.denominator))
    return Fraction(nint(float(np.float64(as_real4) * precision)), precision)


def clamp(dt: Fraction, min_dt: Fraction | None,
          max_dt: Fraction | None) -> Fraction:
    """``max_time_step`` then ``min_time_step`` (:186-214), in that order.

    Order is upstream's and is observable: with ``min_time_step >
    max_time_step`` the minimum wins, because it is applied second.  That
    combination is refused at config admission rather than relied on, but
    the port keeps upstream's order so the two agree if it ever is not.
    """
    if max_dt is not None and dt > max_dt:
        dt = max_dt
    if min_dt is not None and dt < min_dt:
        dt = min_dt
    return dt


# ---------------------------------------------------------------------------
# The controller: adapt_timestep's sequence, composed from the pieces above.
#
# SCOPE OF EVIDENCE, stated plainly because it is not uniform.  `calc_dt`
# above is graded bit-exact against a captured oracle -- upstream's own
# routine compiled against WRF's ESMF time manager, 1728 rows.  The
# SEQUENCE below is not, and cannot be cheaply: `adapt_timestep` USEs
# module_domain, module_configure, module_dm and module_bc_em, so a
# standalone harness for it is a different order of work from the one
# `calc_dt` needed.  It is transcribed against the Fortran line by line
# and its ORDER is pinned by tests, which is weaker evidence than the
# oracle and is labelled as such rather than blurred into it.
# ---------------------------------------------------------------------------


class AdaptiveTimestepController:
    """One domain's adaptive clock state (``adapt_timestep_em.F:1-449``).

    Pure: Fractions in, Fractions out, no device and no clock.  The CFLs
    are handed in because reading them is the caller's job -- which is
    also what keeps every float in here outside
    ``woof.core.clock``'s AST-audited walk.

    ``stepping_to_time`` and ``use_last2`` are upstream's, and they are
    the subtle part: when a step is SHORTENED to land on an output or
    boundary time, the shortened value must NOT become the baseline the
    next step grows from, or the clock ratchets down every time it lands
    on a frame.  Upstream keeps the pre-shortening interval in
    ``last_dtInterval`` for exactly that reason (:392-402).
    """

    __slots__ = ("target_cfl", "target_hcfl", "max_increase_factor",
                 "min_dt", "max_dt", "precision", "last_dt",
                 "last_max_vert_cfl", "last_max_horiz_cfl",
                 "stepping_to_time", "started")

    def __init__(self, *, target_cfl: float, target_hcfl: float,
                 max_step_increase_pct: int,
                 starting_dt: Fraction,
                 min_dt: Fraction | None = None,
                 max_dt: Fraction | None = None,
                 precision: int = PRECISION):
        self.target_cfl = target_cfl
        self.target_hcfl = target_hcfl
        # 1 + pct/100 (adapt_timestep_em.F:116).
        self.max_increase_factor = 1.0 + max_step_increase_pct / 100.0
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.precision = precision
        self.last_dt = Fraction(starting_dt)
        self.last_max_vert_cfl = 0.0
        self.last_max_horiz_cfl = 0.0
        self.stepping_to_time = False
        self.started = False

    def first_step(self) -> Fraction:
        """The step taken before any CFL has been measured (:124-150).

        No ``restart`` argument, deliberately: upstream's two arms differ
        only in WHERE the interval comes from, and both then take it
        unchanged, so the distinction belongs to whoever builds the
        controller rather than to this call.

          fresh start -- ``starting_dt`` is the configured
          ``starting_time_step`` (or ``4*dx`` when it is left at -1);
          upstream zeroes its LOCAL ``last_dtInterval`` here, and the
          growth bound is separately guarded by ``currentTime /=
          startTime`` (:174), which is what ``self.started`` reproduces.

          restart -- ``starting_dt`` is the RESTORED ``last_dtInterval``
          from the checkpoint, NOT the configured value.  That is the
          MOVING tree's local patch and the distinction decides the answer: the CFL
          memory is not in the checkpoint, so a controller that measured
          here would read cfl ~ 0, take calc_dt's ``max_cfl < 0.001``
          branch, grow by the full increase factor and overshoot into a
          blow-up minutes later.  docs/ADAPTIVE-TIMESTEP.md section 5.

        Passing the wrong one is the whole bug, so the caller is made to
        say which it has rather than handed a flag that reads as if it
        decides something here.
        """
        self.started = True
        return self.last_dt

    def next_dt(self, *, max_vert_cfl: float, max_horiz_cfl: float
                ) -> Fraction:
        """One step's dt from the CFLs just measured (:152-214).

        Two calls to ``calc_dt``, the more restrictive wins, then the
        growth bound, the 1/100 requantisation, and the clamps -- in
        upstream's order, which is observable (see :func:`clamp`).
        """
        if self.stepping_to_time:
            # The step just taken was shortened to land on a time, so its
            # CFL describes a step nobody will take again.  Upstream
            # reaches back to the previous one (:154-159).
            vert, horiz = self.last_max_vert_cfl, self.last_max_horiz_cfl
        else:
            vert, horiz = max_vert_cfl, max_horiz_cfl

        dt_vert = calc_dt(self.last_dt, vert, self.max_increase_factor,
                          self.target_cfl, self.precision)
        dt_horiz = calc_dt(self.last_dt, horiz, self.max_increase_factor,
                           self.target_hcfl, self.precision)
        dt = dt_vert if dt_vert < dt_horiz else dt_horiz

        if self.started:
            # ":168 -- only when this is not the first time on this domain"
            dt = limit_increase(dt, self.last_dt, self.max_increase_factor,
                                self.precision)
        dt = requantise(dt, self.precision)
        dt = clamp(dt, self.min_dt, self.max_dt)
        return dt

    def step_to_time(self, dt: Fraction, time_remaining: Fraction,
                     quantise=None) -> tuple[Fraction, bool]:
        """Land exactly on the next output/boundary time (:299-374).

        Upstream looks TWO steps ahead so it never manufactures a very
        short step -- its own comment says short steps cause instability:

          * remainder between one and two steps -> halve, so the next two
            are equal rather than one long and one tiny;
          * remainder at or under one step -> take exactly the remainder.

        Returns ``(dt, stepping_to_time)``.  This is the mechanism that
        makes woof's whole-number-of-steps cadence rule satisfiable at
        all under a varying dt, which is why the config refuses adaptive
        stepping with ``step_to_output_time`` off.
        """
        if time_remaining <= 0:
            return dt, False
        if dt < time_remaining < dt * 2:
            # HALVING IS NOT LATTICE-SAFE.  An odd number of ticks halves
            # to a half-tick -- 9435 ticks gave 47.175 s, which killed a
            # live run at the frame boundary.  WRF's rational clock takes
            # it; an integer tick lattice cannot.  ``quantise`` snaps DOWN
            # to the lattice, and the shortfall is picked up by the
            # `time_remaining <= dt` arm on the following step, so the
            # frame is still landed on exactly -- just by two unequal
            # steps rather than two equal ones.
            half = time_remaining / 2
            return (quantise(half) if quantise else half), True
        if time_remaining <= dt:
            return time_remaining, True
        return dt, False

    def accept(self, dt: Fraction, *, max_vert_cfl: float,
               max_horiz_cfl: float, stepping_to_time: bool) -> None:
        """Commit the step (:392-406).

        ``use_last2``: when the step was shortened to land on a time,
        ``last_dt`` and the CFL memory are NOT advanced -- the next step
        grows from the interval BEFORE the shortening, so landing on a
        frame does not ratchet the clock down.

        Upstream has a real quirk here and it is reproduced rather than
        tidied: in the use_last2 branch it writes
        ``grid%last_max_vert_cfl = grid%last_max_vert_cfl`` (a no-op),
        and then two lines later, OUTSIDE the branch, unconditionally
        assigns ``grid%last_max_vert_cfl = grid%max_vert_cfl`` (:405).
        So the VERTICAL memory is clobbered even when stepping to a time
        while the HORIZONTAL one is preserved.  The two are asymmetric in
        upstream, and a port that "fixed" it would diverge.
        """
        self.stepping_to_time = stepping_to_time
        if not stepping_to_time:
            self.last_dt = dt
            self.last_max_horiz_cfl = max_horiz_cfl
        # :405, outside the branch in upstream -- deliberate asymmetry.
        self.last_max_vert_cfl = max_vert_cfl
