"""How fast this plan will actually run, said BEFORE it runs.

THE HOLE THIS FILLS.  Every memory figure this package publishes is
careful, itemized and provenanced, and none of them is the number a user
acts on.  A 399,119-column domain on a 10 GiB card is priced correctly,
sent down the streamed road correctly, and started -- and nothing
anywhere says what a streamed step at that size COSTS, so a three-hour
forecast that is going to take hours of wall clock looks, from the
outside, exactly like a stall.  A plan document that answers "will it
fit" and refuses to answer "how long" has answered the easier half of
the question.

NOTHING HERE IS INVENTED.  Every rate is a measurement this repository
already carries, with the card, the grid and the timestep named; where a
figure is a bound rather than a measurement its ``basis`` says
``unmeasured-bound`` in those words, and :attr:`StepRate.measured` is
the field a caller branches on.

THE MEASUREMENTS
----------------
RESIDENT, full physics (Thompson/Morrison + RTE-RRTMGP + YSU + Noah +
Kain-Fritsch), per COLUMN per step at nz=49:

* ``docs/public/receipts/obsbattery/grounding-3km-conus-run-report.json``
  -- 796x636x49 at 3 km, dt 15 s, 1440 steps in 1366.03 s on an RTX 5090
  (Linux, sole occupant confirmed) = 0.9486 s/step = **1.874e-6 s per
  column-step**.
  This is the best-loaded measured run in the fleet and it is the FAST
  end.
* ``docs/public/HARDWARE.md`` -- 438x352x49 at 12 km, dt 60 s, 360 steps
  in 400 s on an RTX 4090 (Linux) = 1.111 s/step = **7.21e-6 s per
  column-step**.  A slower card AND a radiation-heavy cadence (at dt
  60 s a 12-minute ``radt`` fires every 12 steps instead of every 48),
  which is why it is the SLOW end.

STREAMED, full physics, per column per step:

* ``tilestream/HANDOFF-case-imagery.md`` -- a real HRRR case,
  1200x900x49 at 3 km, dt 15 s, tile 400x300 with halo 16 and 2
  buffers, on a single RTX 4090: **11.66 s/step measured**, with 9.6
  s/step on a less contended box and 22.6 s/step with a second forecast
  on the same card.  At 1,080,000 columns that is **8.89e-6 to 2.09e-5 s
  per column-step**.  The same receipt records that the design "moves
  ~27 GB of pinned host RAM per step" and a redundancy of 1.1952x, and
  :func:`streamed_transfer_bytes_per_step` reproduces BOTH from the
  tiling alone -- 27.1 GB and 1.1952x -- which is what licenses using
  that byte model on tilings nobody has timed.

A STREAMED STEP IS NOT ONLY ITS COLUMNS.  The rows above are whole steps
of the tiling they were measured on (1.1952x, nine tiles), per domain
column.  A tiling that does more halo work pays for it column for column,
so the column term is scaled by the plan's redundancy; and every tile
pays its own kernel sequence and sync whatever its size, which a column
rate cannot see at all.  :data:`TILE_SECONDS_LOW`/:data:`TILE_SECONDS_HIGH`
carry that per-tile cost, each end fitted from a streamed forecast of
hundreds of tiles once the column term at the same end is taken out, and
are charged for the tiles past the nine the rows already contain
(:data:`STREAMED_REFERENCE_TILES`).  Without both terms a 1,190-tile
sweep at 49.95x was quoted 0.45-1.2 s per step and ran at 237-547 s
(measured 2026-09-26).

THE THREE EFFECTS THAT MOVE A RATE, all of them measured here:

* **Card.** The identical speedrun grid measured 7.885e-6 s per
  column-step on an RTX 5070 Ti (Linux) and 1.381e-5 on an RTX 3080
  (Windows) -- **1.75x** -- and the 1.5.0 envelope pair measured 7.17e-7
  against 1.231e-6, **1.72x**, independently.  :data:`CARD_SPREAD`
  carries it, and it already contains the Windows/WDDM platform gap
  because the slow member of each pair is the Windows box.
* **Radiation cadence.** Radiation is 69% of a reference step (486 ms of
  it, ``docs/lead-handoff-2026-07-26.md``), and it fires on a wall-clock
  period, so a longer ``dt`` amortizes it over fewer steps and every
  step costs more.  The bracket spans two measured cadences rather than
  modelling this.
* **Launch-bound floor.** Below roughly 100,000 columns the card is not
  loaded and the per-column rate degrades sharply -- measured 1.11e-5 at
  250x200 against 1.874e-6 at 796x636 on the same card class
  (``docs/da-vs-wofs.md``: "small enough to be launch-bound").
  :data:`LAUNCH_BOUND_COLUMNS` is where the sentence says so.

Session-to-session variance on one box is up to 30% (stated in
``docs/public/HARDWARE.md`` and twice more), which is a further reason
every figure here is a bracket and none is presented as exact.

THE COLUMN BOUND.  :func:`resident_column_limit` is the largest column
count whose whole peak -- the domain, the per-process fixed cost, the
CUDA context and the rung's measured RRTMGP transient reservation --
still fits the card's allowance.  It is inverted from
``tilestream.autoplan``'s own cost model and checked against
``autoplan.plan`` at the boundary in the tests, because a report that
advised a size the planner then streamed anyway would be two models of
one card.  It is the actionable number: the memory report says the
current domain does not fit, and this says which one would be fast.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

GIB = 1 << 30


class _Unpriced:
    """'Nobody has resolved the road yet', distinct from 'resident'.

    ``streamed=None`` is a positive statement -- this plan runs
    resident -- so it cannot also mean 'not asked'.  Conflating them
    made a bare ``pace_advisory(exp)`` answer for the RESIDENT road on
    a config whose [tiles] table streams, which is the exact class of
    silent-wrong-road defect this module exists to end.
    """

    def __repr__(self) -> str:              # pragma: no cover
        return "<unpriced>"


UNPRICED = _Unpriced()

#: Every rate below is quoted per column at this level count, and scales
#: LINEARLY in ``nz``.  That is measured, not assumed: the P2 LES sweep
#: (``docs/internal/receipts/les/p2-nz-tier-2026-08-04``) ran nz =
#: 64/96/128/160/192 on one card and ``steps/s x nz`` came out 8,743 /
#: 9,146 / 9,102 / 9,042 / 8,915 -- 4.5% peak to peak across a 3x range.
#: "Throughput scales as the level count and no faster", in the
#: receipt's own words.
REFERENCE_NZ = 49

#: Spread between the fastest and slowest cards this fleet has measured
#: on IDENTICAL work, and it is two independent measurements of the same
#: number: the speedrun's 178x144x49 course ran 7.885e-6 s per
#: column-step on an RTX 5070 Ti (Linux) and 1.381e-5 on an RTX 3080
#: (Windows), 1.75x; the 1.5.0 operational envelope measured 7.17e-7 on
#: an RTX 5090 against 1.231e-6 on an RTX 5070 Ti, 1.72x.  The
#: Windows/WDDM platform gap is INSIDE this figure, not on top of it.
CARD_SPREAD = 1.75

#: Below this many columns the card stops being loaded and the
#: per-column rate degrades well outside the bracket -- measured 1.11e-5
#: s per column-step at 250x200 against 1.874e-6 at 796x636 on the same
#: card class.  The pace sentence says so rather than quoting a bracket
#: that does not apply.
LAUNCH_BOUND_COLUMNS = 100_000


@dataclass(frozen=True)
class StepRate:
    """Seconds per COLUMN per model step, at :data:`REFERENCE_NZ`.

    ``measured`` is the field a caller branches on and the fact a reader
    is owed: a rung this project has never timed carries a bound derived
    from the rungs on either side of it, and says so in both places.
    """

    rung: str
    road: str
    low: float
    high: float
    reference_card: str
    basis: str
    measured: bool = True

    def seconds_per_step(self, columns: int, nz: int) -> tuple[float, float]:
        """The bracket, in seconds, for ``columns`` columns at ``nz``."""
        scale = float(columns) * float(nz) / REFERENCE_NZ
        return (self.low * scale, self.high * scale)


_RESIDENT_FULL_BASIS = (
    "MEASURED full-physics resident forecasts, per column-step at nz=49: "
    "the fast end is 1.874e-6 from the 3 km CONUS grounding run "
    "(796x636x49, dt 15 s, 1440 steps in 1366.0 s on an RTX 5090 under "
    "Linux, sole occupant confirmed -- docs/public/receipts/obsbattery/"
    "grounding-3km-conus-run-report.json), and the slow end is 7.21e-6 "
    "from 438x352x49 at 12 km, dt 60 s, 360 steps in 400 s on an RTX "
    "4090 under Linux (docs/public/HARDWARE.md).  The bracket spans a "
    "card gap AND a radiation-cadence gap: radiation fires on a "
    "wall-clock period and is 69% of a reference step, so dt 60 s "
    "amortizes it over a quarter as many steps as dt 15 s does")

_RESIDENT_MYNN_BASIS = (
    _RESIDENT_FULL_BASIS + ".  Scaled by the MEASURED 1.86x MYNN + "
    "Noah-MP premium: the d04 reference step is 0.699 s amortized and "
    "the MYNN 5/5 step is 1.30 s on the same domain "
    "(docs/lead-handoff-2026-07-26.md), which is a measurement of the "
    "RUNG and not of this grid")

_RESIDENT_DRY_BASIS = (
    "MEASURED monolithic resident dry steps at nz=49 on an RTX 5090 "
    "(tilestream/RESULTS.md s1 and s12.7): 3.325 ns/cell at 1448^2, "
    "3.433 at 1536^2, 3.712 at 1950^2, which is 1.63e-7 to 1.82e-7 s per "
    f"column-step; the high end carries the {CARD_SPREAD:.2f}x measured "
    "card spread.  A dry rate is a TRANSPORT DIAGNOSTIC "
    "(tilestream/NO-DRY-NUMBERS.md) and no forecast a user runs is dry")

_RESIDENT_MOIST_BASIS = (
    "unmeasured-bound: this package has never timed a moisture-only "
    "forecast step, so the bracket is bounded BELOW by the measured dry "
    "rate and ABOVE by the measured full-physics one.  It is a bound and "
    "not a measurement, and it is wide because those two rungs are an "
    "order of magnitude apart.  Timing one mp10 step at a known column "
    "count closes it")

_STREAMED_FULL_BASIS = (
    "MEASURED streamed full-physics forecast: a real HRRR case at "
    "1200x900x49, 3 km, dt 15 s, tile 400x300 with halo 16 and 2 "
    "buffers on a single RTX 4090 ran 11.66 s/step, with 9.6 s/step "
    "measured on a less contended box and 22.6 s/step with a second "
    "forecast sharing the card (tilestream/HANDOFF-case-imagery.md).  At "
    "1,080,000 columns that is 8.89e-6 to 2.09e-5 s per column-step.  "
    "The same receipt records ~27 GB of pinned host RAM moved per step "
    "and a redundancy of 1.1952x, and this module's byte model "
    "reproduces both from the tiling alone (27.1 GB, 1.1952x), which is "
    "what licenses pricing a tiling nobody has timed.  A second MEASURED "
    "point sets the fast end: a 399,119-column 1 km HRRR case at dt 5 s "
    "on an RTX 3080 under Windows/WDDM ran 2.398 s/step median over 251 "
    "settled steps through the real run-plan door -- 6.01e-6 s per "
    "column-step (evidence/streamed-pace-3080-20260824.md)")

_STREAMED_MYNN_BASIS = (
    _STREAMED_FULL_BASIS + ".  Scaled by the MEASURED 1.86x MYNN + "
    "Noah-MP premium (docs/lead-handoff-2026-07-26.md)")

_STREAMED_TAX_BASIS = (
    "unmeasured-bound at this rung: no streamed forecast has been timed "
    "below full physics, so the resident rate is multiplied by the "
    "MEASURED dry tiling tax -- 1.05x at the headline out-of-core size "
    "(tilestream/RESULTS.md s1) up to 1.359x at the worst tile size in "
    "the 1024^2 sweep (docs/manual/06-pipeline.md).  The transfer floor "
    "below is priced from bytes either way, so a tiling that is actually "
    "bus-bound is not quoted a compute-bound answer")

#: The MEASURED premium of the MYNN + Noah-MP rung over the reference
#: physics bundle: 1.30 s amortized against 0.699 s on the same d04
#: domain and card (``docs/lead-handoff-2026-07-26.md``).
MYNN_PREMIUM = 1.86

#: The measured dry tiling tax, used only where no streamed forecast has
#: been timed at the rung: 1.05x at the headline out-of-core size,
#: 1.359x at the worst tile size of the 1024^2 sweep.
STREAM_TAX_LOW, STREAM_TAX_HIGH = 1.05, 1.359

#: The redundancy the measured streamed rows above were taken at: the HRRR
#: receipt's 400x300 tiles with halo 16 over 1200x900 do 1.1952x the
#: necessary work, reproduced from the tiling by
#: :func:`streamed_transfer_bytes_per_step`'s arithmetic.  A streamed row is
#: a per-DOMAIN-column rate at that redundancy, so a tiling doing R times
#: the necessary work costs ``R / 1.1952`` of it per domain column: the
#: halo cells are computed exactly like interior ones.
STREAMED_REFERENCE_REDUNDANCY = 1.1952

#: The tiles one step of the measured streamed rows swept: the HRRR
#: receipt's 400x300 tiles over 1200x900 are a 3x3 tiling.  The rows are
#: WHOLE steps, so those nine tiles' own cost is already inside them, and
#: :func:`estimate_pace` charges the per-tile term only for the tiles past
#: nine.  Charging all of them counted nine tiles twice, 0.16-1.5 s on the
#: receipt's 6.5-22.6 s step.  The fast end's 3080 run did not record its
#: tiling and is taken at the same nine.  A plan of nine tiles or fewer is
#: charged no per-tile term and keeps the rows as measured.
STREAMED_REFERENCE_TILES = 9

#: What one TILE costs a streamed step apart from the columns it computes,
#: in seconds per tile per model step: its whole kernel sequence launched
#: on a window, and the sync that ends it.  A column rate cannot see this,
#: and on a nearly full card it is most of the step.
#:
#: BOTH ENDS ARE FITTED NET OF THIS MODULE'S OWN COLUMN TERM: a measured
#: step, less the redundancy-scaled column term at the same end of the
#: bracket, over the tiles past the rows' nine.  A per-tile figure fitted
#: beside some other column term cannot be added to this one, because
#: part of every tile's work is then charged twice.
#: ``tilestream/bench_tile_overhead.py`` fits 18.0 ms a tile beside its own
#: 3.59e-6 s per window column; this module charges 5.03e-6 s per window
#: column at its fast end, and the 18 ms added to that prices the LOW
#: forecast below at 17 s a step against the 11.1 s it ran at.
#:
#: * LOW, 0.008 s -- a 208x204x49 3 km forecast (Morrison, YSU, Noah,
#:   RTE+RRTMGP, no cumulus, dt 15 s, radiation every 12 min) streamed in
#:   598 tiles of 8x9 with halo 18 (27.90x redundancy, one buffer, pinned
#:   host store) on an idle RTX 5070 Ti under Linux settled at 11.12 s a
#:   step: the median of 102 steps, 11.08 s the fastest and 11.44 s the
#:   mean with its radiation steps.  The fast-end column term of that
#:   tiling is 5.95 s, so 5.17 s over the 589 tiles past the rows' nine is
#:   8.8 ms a tile.  The constant is that rounded down, so the fastest
#:   settled step lies inside the bracket with room for a second idle run.
#: * HIGH, 0.170 s -- a run measured 2026-09-26: a 206x204x49 domain
#:   swept in 1,190 tiles of 6x6 with halo 18 (49.95x redundancy, one
#:   buffer) on an RTX 3080 under Windows/WDDM that other programs were
#:   drawing on, settled at 237.4 s per 15 s step (296.0 s the step
#:   before).  The slow-end column term of that tiling is 36.7 s, so
#:   200.7 s over the 1,181 tiles past the rows' nine is the per-tile
#:   cost.  A profile of that run put 81% of its wall time in the scheme
#:   status readbacks each tile's physics makes
#:   (:func:`woof.core.health_ledger.read_status` with no ledger
#:   installed): each one waits for the card, and on a WDDM card another
#:   program is drawing on that wait is long, so this end belongs to that
#:   platform and not to a Linux card.
#:
#: HOW THE ENDS ARE CHECKED.  Each end is fitted from one run, so quoting
#: that run back checks the arithmetic and not the model.  The checks that
#: can fail are runs that took no part in either fit, pinned in
#: ``tests/test_streamed_auto_tiling.py``: on the same domain at other
#: tilings, cards and machines, each measured step has to lie inside the
#: bracket.
TILE_SECONDS_LOW = 0.008
TILE_SECONDS_HIGH = 0.170

_RESIDENT_FULL = (1.874e-6, 7.21e-6)
_STREAMED_FULL = (6.01e-6, 2.09e-5)
_RESIDENT_DRY = (1.63e-7, 1.82e-7 * CARD_SPREAD)

#: Keyed by ``tilestream.autoplan.rung_of`` and by road, so every road
#: the planner can choose has a row.  A missing row would answer ``None``
#: where a pace is due, which is the silence this module exists to end.
STEP_RATES: dict[tuple[str, str], StepRate] = {
    ("dry", "resident"): StepRate(
        "dry", "resident", *_RESIDENT_DRY, "RTX 5090", _RESIDENT_DRY_BASIS),
    ("moist", "resident"): StepRate(
        "moist", "resident", _RESIDENT_DRY[1], _RESIDENT_FULL[0],
        "RTX 5090 / RTX 4090", _RESIDENT_MOIST_BASIS, measured=False),
    ("full", "resident"): StepRate(
        "full", "resident", *_RESIDENT_FULL, "RTX 5090 (Linux)",
        _RESIDENT_FULL_BASIS),
    ("full+mynn+noahmp", "resident"): StepRate(
        "full+mynn+noahmp", "resident",
        _RESIDENT_FULL[0] * MYNN_PREMIUM, _RESIDENT_FULL[1] * MYNN_PREMIUM,
        "RTX 5090 (Linux)", _RESIDENT_MYNN_BASIS),
    ("dry", "streamed"): StepRate(
        "dry", "streamed", _RESIDENT_DRY[0] * STREAM_TAX_LOW,
        _RESIDENT_DRY[1] * STREAM_TAX_HIGH, "RTX 5090",
        _RESIDENT_DRY_BASIS + ".  " + _STREAMED_TAX_BASIS, measured=False),
    ("moist", "streamed"): StepRate(
        "moist", "streamed", _RESIDENT_DRY[1] * STREAM_TAX_LOW,
        _RESIDENT_FULL[0] * STREAM_TAX_HIGH, "RTX 5090 / RTX 4090",
        _RESIDENT_MOIST_BASIS + ".  " + _STREAMED_TAX_BASIS,
        measured=False),
    ("full", "streamed"): StepRate(
        "full", "streamed", *_STREAMED_FULL, "RTX 3080 / RTX 4090",
        _STREAMED_FULL_BASIS),
    ("full+mynn+noahmp", "streamed"): StepRate(
        "full+mynn+noahmp", "streamed",
        _STREAMED_FULL[0] * MYNN_PREMIUM, _STREAMED_FULL[1] * MYNN_PREMIUM,
        "RTX 4090", _STREAMED_MYNN_BASIS),
}

#: Pinned host<->device bandwidth used for the transfer floor when no
#: probe answers.  Both ends are MEASURED on this fleet: 28.17 GB/s H2D
#: and 28.59 GB/s D2H median on an RTX 5090 at PCIe 4.0 x16
#: (``skeptic-results/duplex_5090.json``), 28.45 GB/s = 90.3% of
#: theoretical on the 4x5070 Ti box
#: (``tilestream/BOX-4X5070TI-BLOCKERS.md``), and 25.8 GB/s near-node /
#: 23.6 GB/s far-node on an RTX 4090 across NUMA
#: (``skeptic-results/numa_4090b.json``).
PCIE_PINNED_BYTES_PER_SECOND_HIGH = int(28.45e9)

#: The slow end: the MEASURED bidirectional figure, which is what a
#: gather and a scatter in the same step actually contend for -- 26.36
#: GB/s median on the RTX 5090 duplex probe, and 10.93 GB/s on a 4090
#: whose duplex collapses against 13.43 H2D
#: (``tilestream/skeptic_duplex.py``).  A degraded link is worse again:
#: ``tilestream/OVERLAP-ATTRIBUTION.md`` measured 2.34 GB/s pinned H2D
#: on a box that had negotiated gen1 x16.
PCIE_PINNED_BYTES_PER_SECOND_LOW = int(10.93e9)

_PCIE_BASIS = (
    f"the transfer floor is priced at "
    f"{PCIE_PINNED_BYTES_PER_SECOND_LOW / 1e9:.1f}-"
    f"{PCIE_PINNED_BYTES_PER_SECOND_HIGH / 1e9:.1f} GB/s of pinned "
    "host<->device bandwidth, both ends MEASURED on this fleet: 28.45 "
    "GB/s H2D at 90.3% of theoretical on PCIe 4.0 x16 "
    "(tilestream/BOX-4X5070TI-BLOCKERS.md) and 28.17/28.59 GB/s H2D/D2H "
    "median on an RTX 5090 (skeptic-results/duplex_5090.json) at the "
    "fast end; the slow end is that box's measured DUPLEX collapse, "
    "10.93 GB/s, which is what a gather and a scatter in one step "
    "actually contend for")


def step_rate(rung: str, road: str = "resident") -> StepRate | None:
    """The rate row for ``rung`` on ``road``, or ``None`` if there is none."""
    return STEP_RATES.get((str(rung), str(road)))


def slowest_recorded_rate(road: str = "resident") -> StepRate:
    """The slowest rate this package has RECORDED, for a rung nothing names.

    A caller that must quote a wall clock cannot answer "no row" with
    silence and cannot answer it with a guess either.  This is the third
    answer: the slowest row in :data:`STEP_RATES` for that road, which is
    a figure this repository measured or bounded and therefore the most
    conservative basis on record.  It is never cheaper than the true rung
    unless a future rung is slower than every rung measured so far, and
    the row it returns carries its own ``basis`` so the substitution is
    quoted rather than implied.

    ``measured`` on the returned row describes that row alone; a caller
    that substitutes it must additionally say that the row is a
    SUBSTITUTE, which :func:`estimate_pace` does in its basis when
    ``conservative`` is set.
    """
    rows = [rate for rate in STEP_RATES.values() if rate.road == str(road)]
    return max(rows or list(STEP_RATES.values()), key=lambda rate: rate.high)


def measured_pinned_bytes_per_second(*, device: int = 0,
                                     nbytes: int = 256 << 20,
                                     reps: int = 3) -> int | None:
    """Time a real pinned round trip on this box, or answer ``None``.

    GRACEFUL ABSENCE, the same discipline every other probe in this
    package keeps: no CuPy, no card, or any refusal from the runtime
    answers ``None`` and the caller falls back to the measured table
    constants with a basis that says which it used.  A probe that raised
    would put a hardware question in front of an estimate whose whole
    purpose is to be answerable before anything is allocated.

    THIS CREATES A CUDA CONTEXT.  It is therefore never called from
    ``run-plan --estimate``, which promises the opposite in
    :func:`woof.runplan.estimate_plan`'s own docstring; a caller that
    already owns a context, or one running out of process, may use it.
    """
    try:
        import time

        import cupy as cp
        import numpy as np

        with cp.cuda.Device(device):
            host = cp.cuda.alloc_pinned_memory(int(nbytes))
            view = np.frombuffer(host, dtype=np.uint8, count=int(nbytes))
            buffer = cp.empty(int(nbytes), dtype=cp.uint8)
            buffer.set(view)                      # warm the path
            cp.cuda.Stream.null.synchronize()
            best = None
            for _ in range(max(1, int(reps))):
                start = time.perf_counter()
                buffer.set(view)
                buffer.get(out=view)
                cp.cuda.Stream.null.synchronize()
                elapsed = time.perf_counter() - start
                if elapsed > 0 and (best is None or elapsed < best):
                    best = elapsed
        if not best:
            return None
        return int(2 * int(nbytes) / best)
    except Exception:
        # Including ImportError, CuPy's own runtime errors, and a driver
        # that refuses to pin.  ``None`` means "not measured here", which
        # is a different and accurate answer from a guess.
        return None


def resident_column_limit(cfg, machine, *, footprint=None) -> int | None:
    """The largest column count that still fits the RESIDENT road.

    Inverted from :meth:`tilestream.autoplan.Footprint.resident_bytes`
    against :func:`tilestream.autoplan.budget_for` -- the same two calls
    ``autoplan.plan`` makes to decide resident against tiled -- so the
    advice this produces and the decision the run takes cannot disagree.
    The budget already carries the rung's measured RRTMGP transient as a
    reservation and the price already carries the CUDA context and the
    per-process fixed cost, so the bound is about the whole PEAK and not
    about the domain arrays alone::

        resident_bytes(cells) = (CUDA_CONTEXT + process_fixed
                                 + buffer_fixed + b * cells) * VRAM_SAFETY

    solved for ``cells`` against ``budget_for(machine, fp)`` and divided
    by ``nz``.

    ``None`` when ``machine`` is ``None``: with no allowance there is no
    bound, and a number here would be invented.
    """
    if machine is None:
        return None
    from tilestream import autoplan

    nz = int(getattr(cfg, "nz", 0) or 0)
    if nz <= 0:
        return None
    fp = footprint or autoplan.footprint_for(cfg)
    budget = autoplan.budget_for(machine, fp)
    if budget <= 0 or fp.bytes_per_cell <= 0:
        return None
    cells = ((budget / autoplan.VRAM_SAFETY)
             - autoplan.CUDA_CONTEXT_BYTES
             - fp.process_fixed_bytes
             - fp.buffer_fixed_bytes) / fp.bytes_per_cell
    limit = int(cells // nz)
    if limit <= 0:
        return None
    # Settle the boundary against the pricing itself rather than trusting
    # the float division: what matters is that K fits and K+1 does not.
    # The price is monotone in cells, so the boundary is bracketed by
    # doubling steps and then bisected.  The earlier one-column walk
    # reached the same K but, on the radiation rungs, whose price carries
    # a column-dependent transient the linear inversion above leaves out,
    # it walked about 1,300 columns at a millisecond each: 1 s per card
    # asked, 170 s for the physics catalog's 156 questions.
    def fits(columns: int) -> bool:
        return columns <= 0 or fp.resident_bytes(columns * nz) <= budget

    step = 1
    if fits(limit):
        low = limit
        while fits(low + step):
            low += step
            step *= 2
        high = low + step
    else:
        high = limit
        while not fits(high - step):
            high -= step
            step *= 2
        low = max(0, high - step)
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low or None


def streamed_transfer_bytes_per_step(envelope, cfg) -> int:
    """Bytes crossing the bus per model step, from the chosen tiling.

    The round trip a streamed sweep actually makes, priced off the tiling
    the stream-init decision SELECTED rather than an idealised one: every
    tile gathers its whole compute window (tile plus a halo on both sides
    of both axes -- the window is what a buffer holds, never the tile),
    and every tile scatters its interior back, which sums to the domain
    exactly once.  ``store_bytes_per_cell`` is the carrier inventory,
    which is what a pinned store holds and is far smaller than the VRAM
    per-cell figure.

    VALIDATED AGAINST A MEASUREMENT, which is the only reason it is
    allowed to price a tiling nobody has timed: for the HRRR case in
    ``tilestream/HANDOFF-case-imagery.md`` -- 1200x900x49, tile 400x300,
    halo 16 -- this returns 27.1 GB and the receipt says the design
    "moves ~27 GB of pinned host RAM per step".
    """
    if envelope is None:
        return 0
    from tilestream import autoplan

    nx, ny, nz = int(cfg.nx), int(cfg.ny), int(cfg.nz)
    fp = autoplan.footprint_for(cfg)
    ntx = -(-nx // max(1, int(envelope.tile_nx)))
    nty = -(-ny // max(1, int(envelope.tile_ny)))
    window_cells = int(envelope.window_nx) * int(envelope.window_ny) * nz
    gather = ntx * nty * window_cells
    scatter = nx * ny * nz
    return int((gather + scatter) * fp.store_bytes_per_cell)


def streamed_tiling(envelope, cfg) -> tuple[int, float]:
    """``(tiles per step, redundancy)`` of the tiling ``envelope`` carries.

    Read off the envelope where it carries them
    (:attr:`woof.core.streaming.StreamedEnvelope.ntiles`), otherwise
    derived from its tile and halo by the planner's own arithmetic, so an
    envelope built by hand is priced the same way.
    """
    ntiles = int(getattr(envelope, "ntiles", 0) or 0)
    redundancy = float(getattr(envelope, "redundancy", 0.0) or 0.0)
    if ntiles > 0 and redundancy > 0.0:
        return ntiles, redundancy
    from woof.core.streaming import tiling_shape

    return tiling_shape(cfg, envelope.tile_nx, envelope.tile_ny,
                        envelope.halo)


def tile_overhead_seconds(ntiles: int) -> tuple[float, float]:
    """What a tiling of ``ntiles`` tiles adds to one streamed step.

    Charged for the tiles past :data:`STREAMED_REFERENCE_TILES` only: the
    streamed rows are whole steps of a nine-tile tiling and already carry
    those nine tiles' cost.
    """
    extra = max(0, int(ntiles) - STREAMED_REFERENCE_TILES)
    return (extra * TILE_SECONDS_LOW, extra * TILE_SECONDS_HIGH)


def tiling_step_seconds(cfg, *, tile_nx: int, tile_ny: int, halo: int
                        ) -> tuple[float, float] | None:
    """Seconds per step of streaming ``cfg`` at this tiling, as a bracket.

    The column term at the measured streamed rate scaled by this tiling's
    redundancy, plus the per-tile term: the same arithmetic
    :func:`estimate_pace` quotes, without the bus floor.  ``None`` where
    no streamed rate names the rung.  A refusal quotes it so the reader
    sees what a declined tiling would have cost.
    """
    from tilestream import autoplan

    from woof.core.streaming import tiling_shape

    rate = step_rate(autoplan.rung_of(cfg), "streamed")
    if rate is None:
        return None
    ntiles, redundancy = tiling_shape(cfg, tile_nx, tile_ny, halo)
    low, high = rate.seconds_per_step(int(cfg.nx) * int(cfg.ny), int(cfg.nz))
    scale = redundancy / STREAMED_REFERENCE_REDUNDANCY
    tile_low, tile_high = tile_overhead_seconds(ntiles)
    return (low * scale + tile_low, high * scale + tile_high)


def resident_step_seconds(cfg) -> tuple[float, float] | None:
    """Seconds per step of ``cfg`` on the resident road, as a bracket."""
    from tilestream import autoplan

    rate = step_rate(autoplan.rung_of(cfg), "resident")
    if rate is None:
        return None
    return rate.seconds_per_step(int(cfg.nx) * int(cfg.ny), int(cfg.nz))


def format_span(low: float, high: float) -> str:
    """``LOW-HIGH`` at the precision the bracket is known to."""
    return f"{_number(low)}-{_number(high)}"


def _domain_steps(domain, run_seconds: float) -> int:
    dt = float(getattr(domain.run, "dt", 0.0) or 0.0)
    if dt <= 0.0:
        return 0
    return max(1, int(math.ceil(float(run_seconds) / dt)))


def _number(value: float) -> str:
    """A figure at the precision it is actually known to.

    Never more than three significant figures, because none of these
    brackets is known to more than two and a long decimal reads as a
    measurement.  A small NONZERO value keeps two significant figures
    rather than rounding to ``0.00``: a sub-centisecond step is a real
    and useful answer, and printing it as zero reads as a broken
    estimator.
    """
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}"
    if value <= 0.0:
        return "0"
    return f"{value:#.2g}"


def _duration(seconds: float) -> str:
    """A wall clock in the unit a reader thinks in.

    Hours for a forecast that takes hours, minutes for one that takes
    minutes.  Quoting "3.5e-06 h" is arithmetically fine and useless: the
    whole point of this sentence is that somebody reads it and decides
    whether to start the run.
    """
    if seconds >= 3600.0:
        return f"{_number(seconds / 3600.0)} h"
    if seconds >= 60.0:
        return f"{_number(seconds / 60.0)} min"
    return f"{_number(seconds)} s"


def _span(low: float, high: float) -> str:
    """``~LOW-HIGH unit``, with the unit named once when both share it."""
    for cut, scale, unit in ((3600.0, 3600.0, "h"), (60.0, 60.0, "min"),
                             (0.0, 1.0, "s")):
        if high >= cut:
            if low >= cut:
                return f"~{_number(low / scale)}-{_number(high / scale)} {unit}"
            # The two ends straddle a unit boundary, so each carries its
            # own rather than rounding the small one to zero of the big.
            return f"~{_duration(low)}-{_duration(high)}"
    return f"~{_number(low)}-{_number(high)} s"


@dataclass(frozen=True)
class PaceEstimate:
    """The pace bracket, and everything a reader needs to judge it."""

    road: str
    seconds_per_step_low: float
    seconds_per_step_high: float
    wall_seconds_low: float
    wall_seconds_high: float
    steps: int
    resident_column_limit: int | None
    resident_seconds_per_step_low: float
    resident_seconds_per_step_high: float
    transfer_bytes_per_step: int
    transfer_seconds_per_step_low: float
    transfer_seconds_per_step_high: float
    columns: int
    measured: bool
    reference_card: str
    basis: str
    run_seconds: float
    #: True when no rate row named this rung and road and the estimate
    #: was taken from :func:`slowest_recorded_rate` instead.  A reader
    #: branches on it exactly as on ``measured``; the basis says which
    #: row stood in and for what.
    substituted: bool = False
    #: The streamed road's tiles per step and their redundancy, and what
    #: the tiles past the rates' own nine add to a step
    #: (:func:`tile_overhead_seconds`).  ``None`` and zero on the resident
    #: road.
    tiles: int | None = None
    redundancy: float | None = None
    tile_seconds_per_step_low: float = 0.0
    tile_seconds_per_step_high: float = 0.0

    @property
    def realtime_ratio_low(self) -> float:
        """Simulated seconds per wall second, at the SLOW end.

        Paired with ``wall_seconds_high`` deliberately: a document that
        put the fast wall against the low ratio would read as a run being
        fastest exactly when it is slowest.
        """
        return (0.0 if self.wall_seconds_high <= 0.0
                else self.run_seconds / self.wall_seconds_high)

    @property
    def realtime_ratio_high(self) -> float:
        return (0.0 if self.wall_seconds_low <= 0.0
                else self.run_seconds / self.wall_seconds_low)

    def sentence(self) -> str:
        """One sentence naming BOTH roads, because both are actionable.

        The road the plan is on tells the reader what they are about to
        wait for; the column bound tells them what to change.  A sentence
        with only the first half is a complaint.
        """
        shape = ("" if not self.tiles else
                 f" ({int(self.tiles):,} tiles at "
                 f"{float(self.redundancy or 0.0):.2f}x redundancy)")
        head = (f"{self.road} road{shape}: expect roughly "
                f"{_number(self.seconds_per_step_low)}-"
                f"{_number(self.seconds_per_step_high)} s per model step "
                f"({_span(self.wall_seconds_low, self.wall_seconds_high)} "
                f"wall for this {_duration(self.run_seconds)} forecast) on "
                f"this card")
        if self.resident_column_limit is None:
            tail = ("; the resident road's column bound needs a card to "
                    "price against and none was readable here")
        else:
            tail = (f"; the resident road at <= "
                    f"{self.resident_column_limit:,} columns runs about "
                    f"{_number(self.resident_seconds_per_step_low)}-"
                    f"{_number(self.resident_seconds_per_step_high)} s/step")
        if self.columns and self.columns < LAUNCH_BOUND_COLUMNS:
            tail += (f".  Under {LAUNCH_BOUND_COLUMNS:,} columns the card is "
                     "not loaded and the per-column rate degrades outside "
                     "this bracket, so treat it as an upper bound only")
        return head + tail

    def to_json(self) -> dict:
        """The document ``run-plan --estimate`` and ``woof check`` carry.

        Key names are a published surface: a front end renders them
        verbatim, so they are added to and never renamed.
        """
        return {
            "road": self.road,
            "seconds_per_step_low": round(self.seconds_per_step_low, 6),
            "seconds_per_step_high": round(self.seconds_per_step_high, 6),
            "wall_seconds_low": round(self.wall_seconds_low, 3),
            "wall_seconds_high": round(self.wall_seconds_high, 3),
            "realtime_ratio_low": round(self.realtime_ratio_low, 6),
            "realtime_ratio_high": round(self.realtime_ratio_high, 6),
            "steps": self.steps,
            "columns": self.columns,
            "resident_column_limit": self.resident_column_limit,
            "resident_seconds_per_step_low": round(
                self.resident_seconds_per_step_low, 6),
            "resident_seconds_per_step_high": round(
                self.resident_seconds_per_step_high, 6),
            "transfer_bytes_per_step": self.transfer_bytes_per_step,
            "transfer_seconds_per_step_low": round(
                self.transfer_seconds_per_step_low, 6),
            "transfer_seconds_per_step_high": round(
                self.transfer_seconds_per_step_high, 6),
            "launch_bound_columns": LAUNCH_BOUND_COLUMNS,
            "tiles": self.tiles,
            "redundancy": (None if self.redundancy is None
                           else round(self.redundancy, 4)),
            "tile_overhead_seconds_per_step_low": round(
                self.tile_seconds_per_step_low, 6),
            "tile_overhead_seconds_per_step_high": round(
                self.tile_seconds_per_step_high, 6),
            "measured": self.measured,
            "reference_card": self.reference_card,
            "basis": self.basis,
            "sentence": self.sentence(),
        }


def estimate_pace(exp, *, streamed=UNPRICED, machine=None,
                  pinned_bytes_per_second: int | None = None,
                  conservative: bool = False
                  ) -> PaceEstimate | None:
    """The pace bracket for ``exp``, on the road it will actually take.

    ``streamed`` is the :class:`woof.core.streaming.StreamedEnvelope`
    the caller already priced -- ``None`` meaning the resident road,
    and OMITTED meaning nobody has resolved it, in which case this
    resolves it from the config so a bare call never answers for the
    wrong road --
    and is taken rather than re-derived so the pace describes the tiling
    the memory report quoted.  Under ``[tiles] mode = "auto"`` those two
    genuinely disagree when the card's occupancy moves between the calls,
    which is the same reason
    :func:`woof.core.preflight.streaming_advisory` accepts one.

    ``machine`` is the allowance the column bound is computed against.
    When it is ``None`` the configured ``[tiles] vram_budget_bytes`` is
    used if there is one -- that is exactly what
    :func:`woof.core.streaming.decide` does with the key -- and
    otherwise the bound is ``None`` rather than a guess.

    ``conservative`` is for a caller whose answer is a PRICE rather than
    a sentence: when the rung and road have no row at all, the estimate
    is taken from :func:`slowest_recorded_rate` instead of answering
    ``None``, and the basis names both the missing rung and the row that
    stood in for it.  A missing row is then a stated substitution rather
    than a refusal to price, which is what a plan review owes a reader.
    ``None`` still answers a request with no domain to price at all.

    ``None`` when the experiment carries no domain to price.
    """
    domains = list(getattr(exp, "domains", ()) or ())
    if not domains:
        return None
    from tilestream import autoplan

    if streamed is UNPRICED:
        # Resolve it rather than defaulting to the cheap road.  Never
        # raises: streamed_forecast_envelope answers None for a config
        # that cannot be priced here, which IS the resident answer.
        from woof.core.preflight import streamed_forecast_envelope

        streamed = streamed_forecast_envelope(exp, machine=machine)
    root = domains[0]
    cfg = root.run
    road = "resident" if streamed is None else "streamed"
    rung = autoplan.rung_of(cfg)
    rate = step_rate(rung, road)
    substituted = ""
    if rate is None:
        if not conservative:
            return None
        rate = slowest_recorded_rate(road)
        substituted = (
            f"no rate row names the {rung!r} rung on the {road} road, so "
            f"this is priced from the slowest rate on record for that "
            f"road ({rate.rung}, {rate.high:.3g} s per column-step at "
            f"nz={REFERENCE_NZ}): a SUBSTITUTE basis, not a measurement "
            f"of this configuration, and the figure to beat by timing "
            f"one step of it")
    run_seconds = float(getattr(exp, "run_seconds", 0.0) or 0.0)
    machine = machine if machine is not None else _configured_machine(exp)
    nz = int(cfg.nz)
    columns = int(cfg.nx) * int(cfg.ny)

    step_low, step_high = rate.seconds_per_step(columns, nz)
    parts = [f"{rung} rung on the {road} road: {rate.basis}"]
    if substituted:
        parts.insert(0, substituted)

    # ------------------------------------------------ the tiling's own cost
    # THE TWO TERMS A COLUMN RATE CANNOT SEE (measured 2026-09-26).
    # A tiling of 1,190 tiles at 49.95x was quoted 0.45-1.2 s per step off
    # the column rate and the bus, and ran at 237-547 s: its halo work was
    # fifty times the domain's and every one of its tiles paid a whole
    # kernel sequence and a sync.  The column term is scaled by this
    # tiling's redundancy against the redundancy the rate was measured at,
    # and every tile past the nine inside the rate adds its measured
    # per-step cost.
    tiles = redundancy = None
    tile_low = tile_high = 0.0
    if streamed is not None:
        tiles, redundancy = streamed_tiling(streamed, cfg)
        scale = redundancy / STREAMED_REFERENCE_REDUNDANCY
        tile_low, tile_high = tile_overhead_seconds(tiles)
        step_low = step_low * scale + tile_low
        step_high = step_high * scale + tile_high
        parts.append(
            f"this plan sweeps {tiles:,} tile(s) a step doing "
            f"{redundancy:.2f}x the necessary work, so the column rate is "
            f"scaled by {redundancy:.2f}/{STREAMED_REFERENCE_REDUNDANCY} (the "
            f"redundancy it was measured at), and each tile past the "
            f"{STREAMED_REFERENCE_TILES} the rate was measured with adds "
            f"{TILE_SECONDS_LOW:.3g}-{TILE_SECONDS_HIGH:.3g} s per step for "
            f"its own kernel sequence and sync, {_number(tile_low)}-"
            f"{_number(tile_high)} s here: MEASURED, each end fitted from a "
            f"streamed forecast less its column term, the low end 598 tiles "
            f"of a 208x204x49 domain on an idle RTX 5070 Ti under Linux "
            f"(11.1 s per step settled), the high end 1,190 tiles of a "
            f"206x204x49 domain on an RTX 3080 under Windows/WDDM shared "
            f"with other programs (237.4 s per step settled), where the "
            f"scheme status readbacks each tile makes waited on the busy "
            f"card")

    # ------------------------------------------------------ the bus floor
    transfer_bytes = transfer_low = transfer_high = 0.0
    if streamed is not None and not _store_is_device(exp):
        transfer_bytes = streamed_transfer_bytes_per_step(streamed, cfg)
        fast = pinned_bytes_per_second or PCIE_PINNED_BYTES_PER_SECOND_HIGH
        slow = pinned_bytes_per_second or PCIE_PINNED_BYTES_PER_SECOND_LOW
        transfer_low = transfer_bytes / fast
        transfer_high = transfer_bytes / slow
        # A FLOOR, never a replacement.  The measured streamed rate above
        # is a whole-step figure and already contains the transfer it was
        # measured with; this only raises the answer when THIS tiling
        # moves more bytes per column than the measured one did, which is
        # exactly the case a small card produces (small tiles, more
        # tiles, more halo per useful cell).
        step_low = max(step_low, transfer_low)
        step_high = max(step_high, transfer_high)
        parts.append(
            f"this plan's tiling ({streamed.tile_nx}x{streamed.tile_ny} + "
            f"halo {streamed.halo}, {streamed.nbuffers} buffer(s)) moves "
            f"{transfer_bytes / GIB:.2f} GiB per step, so the bus alone "
            f"costs {_number(transfer_low)}-{_number(transfer_high)} s per "
            f"step and the quoted pace is never below it; {_PCIE_BASIS}")
    elif streamed is not None:
        parts.append(
            "the store is on the DEVICE, so nothing crosses the bus and "
            "the only streaming cost is the tiling tax")

    # ---------------------------------------------------- the whole plan
    steps = _domain_steps(root, run_seconds)
    wall_low, wall_high = step_low * steps, step_high * steps
    for domain in domains[1:]:
        # Every NEST runs resident: the streamed envelope is priced for
        # the root alone (preflight.streamed_forecast_envelope), so a
        # tree's wall clock is the root's road plus resident nests.
        nest_rate = step_rate(autoplan.rung_of(domain.run), "resident")
        if nest_rate is None:
            continue
        nest_low, nest_high = nest_rate.seconds_per_step(
            int(domain.run.nx) * int(domain.run.ny), int(domain.run.nz))
        nest_steps = _domain_steps(domain, run_seconds)
        wall_low += nest_low * nest_steps
        wall_high += nest_high * nest_steps

    # --------------------------------------------------- the column bound
    limit = resident_column_limit(cfg, machine)
    resident_rate = step_rate(rung, "resident") or rate
    if limit is None:
        resident_low = resident_high = 0.0
        parts.append(
            "no VRAM allowance was readable at estimate time, so the "
            "resident column bound is omitted rather than guessed; declare "
            "one with [tiles] vram_budget_bytes, or ask `woof check`, "
            "which measures the card")
    else:
        resident_low, resident_high = resident_rate.seconds_per_step(
            limit, nz)
        parts.append(
            "the resident column bound is inverted from the same "
            "tilestream.autoplan pricing the stream/resident decision uses "
            "-- Footprint.resident_bytes against budget_for, so the CUDA "
            "context, the per-process fixed cost and the rung's measured "
            "RRTMGP transient reservation are all inside it -- and is the "
            "largest column count whose whole peak still fits this card")
    if pinned_bytes_per_second:
        parts.append(
            f"the transfer rate is a PROBED "
            f"{pinned_bytes_per_second / 1e9:.1f} GB/s on this box, not the "
            "table constant")
    parts.append(
        f"rates are per column at nz={REFERENCE_NZ} and scale linearly in "
        "nz (MEASURED: steps/s x nz is constant to 4.5% over nz 64-192, "
        "docs/internal/receipts/les/p2-nz-tier-2026-08-04).  One box "
        "varies up to 30% between sessions, so this is a BRACKET and never "
        "an exact figure")
    return PaceEstimate(
        road=road,
        seconds_per_step_low=step_low, seconds_per_step_high=step_high,
        wall_seconds_low=wall_low, wall_seconds_high=wall_high,
        steps=steps, resident_column_limit=limit,
        resident_seconds_per_step_low=resident_low,
        resident_seconds_per_step_high=resident_high,
        transfer_bytes_per_step=int(transfer_bytes),
        transfer_seconds_per_step_low=transfer_low,
        transfer_seconds_per_step_high=transfer_high,
        columns=columns, measured=rate.measured and not substituted,
        substituted=bool(substituted),
        reference_card=rate.reference_card,
        basis="; ".join(part for part in parts if part),
        run_seconds=run_seconds,
        tiles=tiles, redundancy=redundancy,
        tile_seconds_per_step_low=tile_low,
        tile_seconds_per_step_high=tile_high)


def _store_is_device(exp) -> bool:
    options = getattr(exp, "tiles", None)
    return str(getattr(options, "store", "host")) == "device"


def _configured_machine(exp):
    """A ``Machine`` from ``[tiles] vram_budget_bytes``, or ``None``.

    The same substitution :func:`woof.core.streaming.decide` makes with
    that key: the configured number IS the budget, so the headroom
    multiplier is dropped rather than applied on top of it -- applying it
    is the defect that turned every declared budget into a smaller one.
    Host RAM is irrelevant to the column bound and is filled from the
    declared pinned budget purely so the dataclass is complete.
    """
    options = getattr(exp, "tiles", None)
    budget = getattr(options, "vram_budget_bytes", None)
    if budget is None:
        return None
    from tilestream import autoplan

    host = getattr(options, "host_budget_bytes", None) or int(budget)
    return autoplan.Machine(
        vram_bytes=int(budget), host_bytes=int(host),
        name="[tiles] vram_budget_bytes", vram_headroom=0.0,
        pinned_fraction=1.0, host_source="explicit")


def pace_advisory(exp, *, streamed=UNPRICED, machine=None) -> str | None:
    """The pace sentence for ``woof check``'s text surface.

    Advisory, never a gate: it changes no exit code and blocks nothing,
    on the same posture as every other entry in
    :func:`woof.core.preflight.check_advisories`.  It exists because the
    memory report is complete and silent about the one thing a user
    watching a run needs to know before starting it -- and a streamed run
    at a size that looks stalled is the concrete breakage.
    """
    estimate = estimate_pace(exp, streamed=streamed, machine=machine)
    if estimate is None:
        return None
    tail = ("" if estimate.measured else
            "  This rung and road have no measured step time: the bracket "
            "is a BOUND, not a measurement, and the basis says which.")
    return f"PACE: {estimate.sentence()}." + tail


__all__ = [
    "CARD_SPREAD", "LAUNCH_BOUND_COLUMNS", "MYNN_PREMIUM",
    "PCIE_PINNED_BYTES_PER_SECOND_HIGH", "PCIE_PINNED_BYTES_PER_SECOND_LOW",
    "PaceEstimate", "REFERENCE_NZ", "STEP_RATES", "STREAMED_REFERENCE_REDUNDANCY",
    "STREAMED_REFERENCE_TILES",
    "StepRate", "TILE_SECONDS_HIGH", "TILE_SECONDS_LOW", "estimate_pace",
    "format_span", "measured_pinned_bytes_per_second", "pace_advisory",
    "resident_column_limit", "resident_step_seconds",
    "slowest_recorded_rate", "step_rate", "streamed_tiling",
    "streamed_transfer_bytes_per_step", "tile_overhead_seconds",
    "tiling_step_seconds",
    "UNPRICED",
]
