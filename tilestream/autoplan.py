"""Choose the tiling, so that nobody has to choose a tile size ever again.

``run_tiled`` takes a tile size, a buffer count and a halo.  All three are
traps, and two of them are silent:

* **the halo.**  Too small is bit-wrong AND faster, so a benchmark rewards
  it and a one-step test certifies it (``harness.halo_radius``'s docstring
  has the numbers: halo 13 at ``full+MYNN+Noah-MP`` is bit-exact at N=1 and
  wrong in 111 carriers at N=8).  This module never measures a halo.  It
  copies ``harness.halo_radius(cfg)`` and refuses to accept an override.
* **the tile size.**  Too big does not fit; too small spends the card on
  halo cells (MEASURED: a 160-cell window on a 1024^2 dry domain does 1.56x
  the necessary work and takes 1.36x the time); and a tile that does not
  DIVIDE the domain multiplies the ring arena by 4.2x (MEASURED: 3276^2
  costs 24.46% of the store at tile 512 and 5.84% at tile 546).
* **the buffer count.**  Worth ~32% when it fits and an out-of-memory crash
  when it does not, and whether it fits turns on a question this module had
  to answer with a measurement -- see below.

Give it a ``RunConfig`` and a :class:`Machine` and it answers: resident or
tiled; if tiled, what tile, how many buffers, how much pinned host RAM; and
if neither, WHICH resource is binding.  ``python -m tilestream.autoplan``
prints the plan and the reasoning for it.


THE COST MODEL, AND THE QUESTION NOBODY HAD CHECKED
---------------------------------------------------
The project's working figure was "~2.5 GiB FIXED plus ~656 B/cell, both for a
resident domain and for each tile buffer".  If the fixed part really were per
buffer, ``nbuffers=2`` would cost 5 GiB of fixed on a 12 GB card and the
overlap would be unaffordable there.  MEASURED instead, by building 1..4
physics states in ONE process and reading ``cudaMemGetInfo`` after each
(``--measure`` re-runs it)::

    full+MYNN+Noah-MP, 128^2 x 49, RTX 4090
        buffer 1   3392 MiB      buffer 3   1336 MiB
        buffer 2   1332 MiB      buffer 4   1340 MiB

The first buffer costs 2.0 GiB more than every buffer after it.  **The fixed
part is per PROCESS, not per buffer** -- and it has to be, because CUDA
context, module code and the RRTMGP k-distribution tables are process-wide.
What IS per buffer is much smaller, and only one rung has any of it at all:

===================  ==============  =============  ==========  ============
rung                 process fixed   buffer fixed   B/cell      carrier B/cell
===================  ==============  =============  ==========  ============
dry                       128 MiB          0 MiB     155           32.3
mp10 (moist only)          96 MiB        200 MiB     391          181.0
full(real74)+KF          3232 MiB         16 MiB     541          233.3
full+MYNN+Noah-MP        2560 MiB       1024 MiB     612          279.5
===================  ==============  =============  ==========  ============

so the model this module plans with is

    VRAM = context + process_fixed + nbuffers * (buffer_fixed + b * cells)

The consequence is the whole point: a second buffer on a 12 GB card costs
``buffer_fixed + b * window_cells``, NOT another 2.5 GiB, so overlap is
affordable on a small card and the planner takes it.  Three ways this could
have been got wrong and was not: the numbers come from a fit over 29 points
spanning k = 1..4 buffers and 0.8 to 20 Mcell on two different cards (4090
and 5090, agreeing to 6% on ``b`` at every rung); the fit is non-negative, so
none of the three terms is an artefact of the other two absorbing a sign; and
the worst UNDER-prediction over all points is -4.3%, which is what
:data:`VRAM_SAFETY` covers.

MULTIPLE DOMAINS IN ONE PROCESS
--------------------------------
The measurement above says the fixed part is per PROCESS.  A tree of nested
domains is ONE process, so the same sentence applies to DOMAINS and not only
to buffers, and :meth:`Footprint.process_overhead_bytes` is the part a second
domain does not pay again::

    VRAM(tree) = process_overhead + SUM over domains of marginal_bytes(d)

``woof.core.streaming.steppers_for_tree`` charges it exactly once.  It used
to charge ``vram_bytes`` -- context INCLUDED -- once per domain, which at the
``full`` rung is 3.760 GiB of phantom bytes for every domain after the first:
7.519 GiB on a three-domain tree, enough that a tree priced at 12.41 GiB was
refused outright on a 16.30 GiB card.

THE RADIATION TRANSIENT, AND WHY IT IS A RESERVATION AND NOT A CLAIM
---------------------------------------------------------------------
The cost model above prices what a forecast HOLDS.  A radiation step also
allocates, uses and frees a working set that nothing holds between calls, and
on the measured three-domain run below that transient was **+2.74 GiB** -- a
fifth of the whole forecast, recurring every radiation period and lasting
60-75 s of it.  Ignoring it is not conservative in either direction: it makes
the planner spend the bytes on a bigger tile and then meet the transient at
the first radiation call.

It is charged as a RESERVATION against free VRAM (see :func:`budget_for`),
not as a claim added to a domain's price, and it REPLACES the percentage
:data:`VRAM_HEADROOM` rather than stacking on it -- that guess exists for
exactly these first-use transients, so charging both counts the same bytes
twice.  All three of those choices are decided by the same measurement, a
9/3/1 km three-domain Poland forecast on node 1's 15.92 GiB (16,303 MiB)
card with 15.245 GiB free, RTE+RRTMGP at 49 levels:

======================================  ==========  =========================
quantity                                     GiB    source
======================================  ==========  =========================
d01 200x160x49 resident                      4.614   this model, MATCHED
d02 402x300x49 resident                      6.932   this model, MATCHED
predicted total, d03 tiled 175x375          12.358   this model
measured steady state                       12.72    NVML, -2.9% miss
measured radiation peak                     15.46    NVML, +2.74 over steady
peak headroom against the card               0.456   467 MiB, and it RAN
the tiling one step larger (350x250)        ~14.34   would peak 17.08: DEAD
======================================  ==========  =========================

Both arms are measured, and together they pin the arithmetic:

* as a CLAIM the tree prices 12.358 + 2.74 = 15.098 GiB against a 14.025 GiB
  budget and the run that completed is refused;
* as a reservation STACKED on the 8% headroom the budget is
  15.245 - 1.22 - 2.74 = 11.285 GiB, which buys a 175x250 tile -- 20 tiles
  where 12 ran, so a run that worked is planned slower for no reason;
* as a reservation REPLACING the headroom the budget is
  15.245 - 2.74 = 12.505 GiB, which selects 175x375 -- the tiling that
  actually ran -- and refuses 350x250, the tiling that would have died.

Only the third arm reproduces the observation on both sides, so that is the
one implemented.  :data:`RADIATION_TRANSIENT_BYTES` is the measured number and
the rungs that do not run RRTMGP carry zero.

WHAT THE CONSTANT DOES NOT YET KNOW.  It was measured at ONE column count and
one ``nz``; the run's own instrumentation says it "will not grow: RRTMGP's
workspace is a function of column count, which is fixed", and the same figure
priced both tilings above, so it is carried as a per-PROCESS constant rather
than scaled by a law nobody has measured.  ``woof.core.preflight``'s
shape-derived transient rail computes 1.464 GiB for this same tree -- 1.9x
under the measurement -- so the two rails disagree and the measurement wins
here.  ``tilestream.rrtmgp_ceiling`` already forces a radiation step and reads
the pool; running it at two or three column counts is what would turn this
constant into a slope.

Two things in the table deserve a second look.  ``full(real74)+KF`` has
3.2 GiB of process fixed and no per-buffer fixed; ``full+MYNN+Noah-MP`` has
2.6 GiB of process fixed and 1.0 GiB per buffer.  MYNN and Noah-MP move a
gigabyte of workspace from process scope to driver scope, so the rung with
the SMALLER one-time cost is the expensive one to run two buffers of.  And
the per-cell figures are 541-612 B/cell, not the 656 the project had been
using -- 7 to 20% high, which at the 5395^2 capacity headline is a third of a
tile.


WHAT THE PLANNER OPTIMISES, AND WHAT IT REFUSES TO OPTIMISE
-----------------------------------------------------------
Subject to VRAM and pinned-host budgets, it minimises

    redundancy(tile) / overlap_gain(nbuffers)

where ``redundancy = ntiles_x*ntiles_y*(Tx+2h)*(Ty+2h) / (nx*ny)`` -- the
useful work divided into the work actually done, which counts a ragged
trailing tile's waste automatically because ``plan_tiles`` gives every tile
the SAME compute window and only the trailing interiors shrink.

It does NOT try to optimise the compute window for occupancy, because that
turned out not to be the effect it was assumed to be.  MEASURED here, a
monolithic dry step in ns per cell against the window edge:

    4090   128: 5.115   192: 3.692   256: 3.713   512: 4.109   1280: 4.539
    5090   128: 4.527   192: 3.228   256: 2.721   512: 2.806   1280: 3.027

Both cards are within 4% of their best by a 256-cell window and slowly get
WORSE above ~640 (cache, not occupancy); full physics saturates at 256 as
well (48.5 ns/cell at 256^2 on the 4090 against 82.8 at 128^2).  So "never
benchmark below ~500 cells" is a rule about the TILING TAX and not about the
arithmetic rate of the window.  :data:`MIN_COMPUTE_WINDOW` is therefore a
warning threshold, not a constraint.

Radiation and cumulus fired ZERO times inside every window timed above
(``radt=12 min`` and ``cudt=5 min`` against ``dt=3 s`` are 240 and 100 steps
apart, and the driver's ``call_counts`` were read on both sides to prove it);
sfclay/Noah-MP/PBL fired every step.  Those curves are a fast-cadence step,
which is 239 of every 240.

And the tiling tax itself, MEASURED against the objective it is supposed to
justify -- 1024^2 dry on a 4090, VRAM store so PCIe is out of it, nbuffers=2,
150 steps per call:

    tile  window  tiles  redundancy   tax    tax/redundancy
     128     160     64      1.562    1.359      0.870
     256     288     16      1.266    1.217      0.961
     512     544      4      1.129    1.346      1.192

Redundancy predicts the measured tax within 13% over an 8x range of tile
size, which is what makes it a usable objective.  It is not a perfect one:
the best row here is the MIDDLE tile, and the largest tile -- which this
planner would choose -- came out 10.6% worse than it.  Two effects pull
against each other and redundancy only sees one: a bigger window has lower
redundancy but a slightly worse arithmetic rate (the curve above), and four
tiles against two buffers leaves the pipeline almost nothing to overlap.  The
second is the more likely culprit and it is why a plan with fewer than three
tiles per buffer says so.  A one-configuration result is not enough to
rebuild the objective on, so the objective stands and the caveat is stated.

A warning for anyone who repeats that measurement: ``run_tiled`` builds its
``nbuffers`` tile states on every CALL, and a 544-cell dry buffer takes 3.3 s
to build.  At 4 steps per call that is 1650 ms/step of construction against
244 ms/step of physics, and the tax comes out at 8.7x with the largest tile
looking WORST by a factor of four.  At 30 steps the tile-512 row still read
2.25x.  Only at 150 steps do the numbers above stop moving.


HOST RAM, AND WHY /proc/meminfo IS NOT ALLOWED TO ANSWER
--------------------------------------------------------
``hoststore`` will not pin more than 47% of MemTotal and page-locking walls
at 50% of it, both MEASURED.  But inside a container ``/proc/meminfo`` is the
HOST's -- on the box these constants were measured on it reports 503 GiB
while ``/sys/fs/cgroup/memory.max`` says 241.7 GiB, and a store sized from
the first number is a plan to be OOM-killed.  :meth:`Machine.detect` reads
the cgroup limit first, takes the smaller of the two, and refuses to guess at
all if it can see it is containerised and there is no limit to read.

``/proc/meminfo`` is the LINUX source, not the only one.  Windows has no
procfs, so :func:`_host_memtotal` asks the OS directly there
(``GlobalMemoryStatusEx``/``ullTotalPhys``), and the cgroup reads simply
find nothing and answer ``None`` -- which is the correct answer for a box
that is not containerised.  A planner that had no host source on Windows
did not fall back to a guess, it RAISED, and because
:func:`woof.core.streaming.decide` probes the machine before it reads the
configured budget, that refusal stood in front of every ``[tiles]`` run on
the platform the product's own front door ships on -- including the ones
that configured ``host_budget_bytes`` precisely to say what the budget was.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Any, NamedTuple, Sequence

from tilestream import harness as _harness
from tilestream import spec as _spec

GIB = 1 << 30
MIB = 1 << 20


# --------------------------------------------------------------------------
# measured constants
# --------------------------------------------------------------------------

#: VRAM an initialised CUDA process holds before any model exists, MEASURED at
#: 397 MiB on a headless RTX 4090 (CUDA 13, CuPy 14.1.1).  It is charged once,
#: not per buffer.  :meth:`Machine.detect` budgets against FREE VRAM, so a
#: desktop compositor's share is already excluded and this covers only us.
CUDA_CONTEXT_BYTES = 400 * MIB

#: Worst under-prediction of the fitted model over all 29 measured points was
#: -4.3% (dry, 1152^2, where CuPy's pool rounding runs ahead of the fit).  A
#: planner that under-predicts hands out a tile that OOMs three hours into a
#: forecast, so the model is inflated past its worst measured miss.
VRAM_SAFETY = 1.06

#: The staggered ``+1`` columns make a store slightly dearer per cell than the
#: flat figure; MEASURED at 279.7 B/cell at 128^2 against 279.5 at 384^2, i.e.
#: 0.07%, so 1% covers it everywhere a real domain lives.
STORE_SAFETY = 1.01

#: Below this compute-window edge the TILING TAX -- halo redundancy plus
#: per-tile overhead -- is what stops being negligible; the window's own
#: arithmetic rate is flat from ~256 up (see the module docstring).  This is a
#: warning threshold, never a constraint: on a 12 GB card at full physics
#: there may be no window this large, and running is better than refusing.
MIN_COMPUTE_WINDOW = 500

#: What a second buffer is worth: gathers for tile i+1 overlap compute for
#: tile i.  1.32x on dry, measured by this project before this module existed.
#: A third buffer is taken only when it costs nothing, so it carries no
#: separate credit here.
OVERLAP_GAIN = 1.32

#: Tiles per buffer below which the pipeline has little left to overlap.  In
#: the measured tax curve the four-tile / two-buffer row is the one where the
#: largest tile lost to a smaller one despite less redundancy; three is a
#: threshold for saying so, not a measured cliff.
MIN_TILES_PER_BUFFER = 3

#: Candidates within this band of the best redundancy are decided on ring
#: arena instead, because redundancy does not price the arena and the arena is
#: what raggedness actually costs: MEASURED 5.84% of the store at 3276^2/546
#: (exact) against 24.46% at 3276^2/512 (13 ragged tiles), 4.2x the arena for
#: 6.6% of the tile.
#:
#: Note this is a strictly better rule than "prefer a tile that divides the
#: domain", and it disagrees with it in a case that comes up: on nx=4098
#: (= 2*3*683) the exact ladder jumps from 683 to 1366, so a 1025 tile leaves
#: a 1023-cell trailing tile -- ragged by two cells.  That tiling has BOTH
#: less redundancy (1.065x against 1.096x) and less arena (1.6% against 2.3%)
#: than the exact 683 one.  Raggedness is expensive when the trailing tile is
#: much narrower than the others, not when it is merely unequal, and the arena
#: is the quantity that knows the difference.
ARENA_TIE_BAND = 0.05

#: A ring arena above this fraction of the store is called out.  An exact
#: tiling at a realistic tile size measured 2-6%; the badly ragged plans
#: measured 22-25%.
EXPENSIVE_ARENA = 0.10

#: The most halo work a tiling may do, as a multiple of the necessary work,
#: before :func:`plan` refuses it.  ``[tiles] max_redundancy`` replaces it
#: and ``false`` lifts it; ``mode = "auto"`` never lifts it on its own.
#:
#: THE BREAKAGE IT PREVENTS is a streamed step fifty or more times the
#: resident step, which a user cannot tell from a hang.  MEASURED
#: 2026-09-26: on a card that missed a 206x204x49 domain's resident budget
#: by 0.3 GB, auto dropped this limit and swept the domain in 1,190 tiles
#: of 6x6 at 49.95x; each 15 s step took 237-547 s against 0.6-1.9 s
#: resident on the same card, so a 24 h forecast would have run for about
#: 16 days.
#:
#: WHAT IT DOES NOT BOUND is the tile count.  A domain much larger than
#: the card still streams in many tiles within the limit, and every tile
#: past nine is priced by :func:`woof.core.pace.estimate_pace`, so the
#: review quotes that cost before the run.
#:
#: WHY 4.0 AND NOT ANOTHER NUMBER, measured by
#: ``tilestream/bench_tile_overhead.py`` on an idle RTX 5070 Ti under
#: Linux (192x192x49 full physics, halo 16, one buffer): the streamed step
#: as a multiple of the resident step was 3.8x at 1.78x redundancy, 6.0x at
#: 2.25x, 9.6x at 2.78x and 16x at 4.0x, then 50x at 5.44x, 61x at 9.0x,
#: 174x at 13.4x and 288x at 25x.  The cliff is between 4.0x and 5.44x,
#: where a window stops being much wider than its two halos and the fixed
#: cost every tile pays (18 ms a step in that sweep's fit) outgrows its
#: columns.  At and below the limit :func:`woof.core.pace.estimate_pace`
#: quotes the slower step, per tile and per redundancy, so the review says
#: what it costs.
MAX_REDUNDANCY = 4.0

#: Fraction of MemTotal that may be pinned.  ``hoststore`` measured the wall
#: at 0.4998 x MemTotal and refuses past 0.47; this is that refusal, restated
#: where the plan is made rather than where the allocation fails.
PINNED_FRACTION = 0.47

#: VRAM left unplanned, on top of :data:`VRAM_SAFETY`.  CuPy's pool fragments
#: across a long run, cuFFT/cuBLAS workspaces appear on first use, and the
#: cost of being wrong is a dead forecast.  It is a PERCENTAGE GUESS at those
#: transients; where one of them has been measured instead, the measurement
#: replaces this rather than adding to it -- see :func:`budget_for`.
VRAM_HEADROOM = 0.08

#: The RRTMGP radiation call's per-process TRANSIENT working set, by rung.
#: MEASURED at +2.74 GiB over steady state on the three-domain 9/3/1 km run
#: in the module docstring (NVML: 13,030 MiB steady, excursions to
#: 15,700-15,836 MiB every ~5.8 min lasting 60-75 s, which is exactly the
#: radiation period).  It is allocated, used and freed inside one call, so it
#: is reserved from the budget and never added to a domain's price, and it is
#: charged ONCE per process because the chunk workspace is shared and the
#: domains of a tree step strictly sequentially.
#:
#: The two RRTMGP rungs carry the same figure: the transient measured is the
#: RADIATION one, and MYNN's and Noah-MP's own per-call transients are not in
#: it and have not been measured here.  The rungs without radiation carry
#: zero, which is why every dry and moist plan is byte-identical to what it
#: was before this constant existed.
RADIATION_TRANSIENT_BYTES: dict[str, int] = {
    "dry": 0,
    "moist": 0,
    "full": int(2.74 * GIB),
    "full+mynn+noahmp": int(2.74 * GIB),
}


@dataclass(frozen=True)
class Footprint:
    """The VRAM and host-store cost of one physics rung, MEASURED.

    ``process_fixed_bytes`` is charged once per process; ``buffer_fixed_bytes``
    once per tile buffer; ``bytes_per_cell`` per cell of each buffer's compute
    WINDOW (not of the domain).  ``store_bytes_per_cell`` is the carrier
    inventory -- what a pinned host store actually holds -- and is a different
    and much smaller number than the VRAM per-cell, because a resident state
    also carries scratch, tendencies and physics workspace that never leaves
    the card.
    """

    rung: str
    process_fixed_bytes: int
    buffer_fixed_bytes: int
    bytes_per_cell: float
    store_bytes_per_cell: float
    source: str = "measured"
    # Grid diagnostics survive relocation and are shared by tile buffers.
    domain_fixed_bytes: int = 0
    # (nz, pressure top, selected common column cap), resolved by the owning
    # experiment. The base measured rung still supplies all other prices.
    classic_lw_context: tuple[int, float, int] | None = None
    prepared_memory: object | None = None

    def classic_call_bytes(self, window_cells: int) -> int:
        if self.classic_lw_context is None:
            return 0
        from woof.core.rrtm_inventory import call_workspace_bytes
        nz, p_top, cap = self.classic_lw_context
        columns, remainder = divmod(int(window_cells), nz)
        if remainder:
            raise ValueError("radiation window cells must contain whole vertical columns")
        return call_workspace_bytes(columns, nz, p_top, cap)

    def buffer_bytes(self, window_cells: int, shape=None) -> float:
        """VRAM one tile buffer of ``window_cells`` costs.

        ``shape`` is the compute window ``(window_nx, window_ny)`` when the
        caller knows it.  The measured rungs price by the cell alone and
        ignore it; the itemized prepared model prices the rectangle it is
        given, and without one prices a bound over every legal rectangle
        of that area (:meth:`woof.core.prepared_tile_memory.
        PreparedTileMemory.sizing_shape`).
        """
        if self.prepared_memory is not None:
            return self.prepared_memory.buffer_bytes(window_cells, shape)
        return self.buffer_fixed_bytes + self.bytes_per_cell * window_cells

    def vram_bytes(self, window_cells: int, nbuffers: int, shape=None) -> float:
        """VRAM a process holding ``nbuffers`` such buffers costs, with safety.

        ONE domain in ONE process.  A caller pricing a SECOND domain into a
        process that already holds one wants :meth:`marginal_bytes`, which
        is this number without the part the process already paid.
        ``shape`` as in :meth:`buffer_bytes`.
        """
        if self.prepared_memory is not None:
            return self.prepared_memory.vram_bytes(window_cells, nbuffers, shape)
        raw = (CUDA_CONTEXT_BYTES + self.process_fixed_bytes + self.domain_fixed_bytes
               + nbuffers * self.buffer_bytes(window_cells))
        # The empirical rung already reserves a radiation call separately
        # in budget_for. Charge only the excess of this actual window/cap;
        # adding the entire classic call would charge that reserve twice.
        classic = self.classic_call_bytes(window_cells) * VRAM_SAFETY
        excess = max(0, classic - self.radiation_transient_bytes)
        return raw * VRAM_SAFETY + excess

    @property
    def process_overhead_bytes(self) -> float:
        """The part of :meth:`vram_bytes` a SECOND domain does not pay again.

        The CUDA context, the module images and the rung's k-distribution
        tables are process-wide -- that is the measurement the module
        docstring reports for tile BUFFERS, and the same fact holds for
        DOMAINS, because a tree of domains is one process too.  Charging
        this once per domain was worth 3.76 GiB of phantom bytes per extra
        ``full``-rung domain; see MULTIPLE DOMAINS IN ONE PROCESS above.
        """
        if self.prepared_memory is not None:
            return self.prepared_memory.process_overhead_bytes
        return (CUDA_CONTEXT_BYTES + self.process_fixed_bytes) * VRAM_SAFETY

    def marginal_bytes(self, window_cells: int, nbuffers: int, shape=None) -> float:
        """VRAM one MORE domain costs in a process that already runs one."""
        return self.vram_bytes(window_cells, nbuffers, shape) \
            - self.process_overhead_bytes

    def terms(self, window_cells: int, nbuffers: int, shape=None) -> dict:
        """Every term of :meth:`vram_bytes`, named, for a refusal to print.

        Byte-valued entries end in ``_bytes``; the rest are labels.  The
        itemized prepared model reports its own itemization; a measured rung
        reports the affine fit it is: the CUDA context, the rung's fixed
        costs, the per-cell rate times the window times the buffers, the
        safety factor and the reserved radiation transient.
        """
        if self.prepared_memory is not None:
            terms = dict(self.prepared_memory.terms(window_cells, nbuffers, shape))
            terms["rung"] = self.rung
            return terms
        nbuffers = int(nbuffers)
        window_cells = int(window_cells)
        per_buffer = self.buffer_bytes(window_cells)
        raw = (CUDA_CONTEXT_BYTES + self.process_fixed_bytes + self.domain_fixed_bytes
               + nbuffers * per_buffer)
        classic = self.classic_call_bytes(window_cells) * VRAM_SAFETY
        excess = max(0, classic - self.radiation_transient_bytes)
        return {
            "rung": self.rung,
            "window": (f"{int(shape[0])}x{int(shape[1])} = {int(shape[0]) * int(shape[1]):,} columns"
                       if shape is not None else f"{window_cells:,} cells"),
            "buffer/per_cell": f"{self.bytes_per_cell:.1f} B/cell x {window_cells:,} cells",
            "buffer/fixed_bytes": int(self.buffer_fixed_bytes),
            "buffer/total_bytes": int(per_buffer),
            "buffers": f"{nbuffers} x {per_buffer / GIB:.3f} GiB",
            "buffers_bytes": int(nbuffers * per_buffer),
            "fixed/cuda_context_bytes": int(CUDA_CONTEXT_BYTES),
            "fixed/process_bytes": int(self.process_fixed_bytes),
            "fixed/domain_bytes": int(self.domain_fixed_bytes),
            "safety": f"x {VRAM_SAFETY:.2f}",
            "classic_lw_excess_bytes": int(excess),
            "radiation_transient_bytes": int(self.radiation_transient_bytes),
            "vram_bytes": int(self.vram_bytes(window_cells, nbuffers)),
        }

    def resident_bytes(self, cells: int) -> float:
        """VRAM the whole domain costs with no tiling: one buffer, no halo."""
        return self.vram_bytes(cells, 1)

    def marginal_resident_bytes(self, cells: int) -> float:
        """:meth:`resident_bytes` for a domain that is not the first."""
        return self.marginal_bytes(cells, 1)

    @property
    def radiation_transient_bytes(self) -> int:
        """The rung's per-process radiation transient, RESERVED not claimed.

        Zero for a rung this project has not measured one on, which is the
        accurate answer and also the one that leaves every dry and moist plan
        exactly where it was; see :data:`RADIATION_TRANSIENT_BYTES`.
        """
        if self.prepared_memory is not None:
            # Its independently retained unfused LW/SW storage is already
            # inside vram_bytes, including the actual radiation peak.
            return 0
        return int(RADIATION_TRANSIENT_BYTES.get(self.rung, 0))

    def store_bytes(self, cells: int) -> float:
        """Pinned host bytes one full-domain carrier store costs."""
        return self.store_bytes_per_cell * cells * STORE_SAFETY


#: MEASURED on an RTX 4090 (rented, headless) and an RTX 5090 (local, sharing
#: the card with a desktop), CUDA 13 / CuPy 14.1.1, by non-negative least
#: squares over 1..4 buffers x 0.8..20 Mcell.  Each entry is the CONSERVATIVE
#: envelope of the two cards, because the planner is asked about machines it
#: cannot see.  Per-card values, ``(process MiB, buffer MiB, B/cell)``:
#:
#: ==================  ====================  ====================
#: rung                4090                  5090
#: ==================  ====================  ====================
#: dry                 54,     0,   154.3    122,    0,   149.0
#: mp10                70,   193,   389.6     88,  185,   390.4
#: full(real74)+KF     2720,    9,   540.8   3228,    9,   540.8
#: full+MYNN+Noah-MP   2024,  907,   611.2   2551,  974,   574.4
#: ==================  ====================  ====================
#:
#: The two cards agree on ``B/cell`` to 6% at every rung and on the per-buffer
#: fixed to 8%, which is the cross-check that this is a property of the model
#: and not of one machine.  ``store_bytes_per_cell`` is the carrier inventory
#: measured directly (``physics_inventory.carrier_inventory``), and its dry
#: value reproduces the project's independently derived 32.26 B/cell.
FOOTPRINTS: dict[str, Footprint] = {
    "dry": Footprint("dry", 128 * MIB, 0, 155.0, 32.3),
    "moist": Footprint("moist", 96 * MIB, 200 * MIB, 391.0, 181.0),
    "full": Footprint("full", 3232 * MIB, 16 * MIB, 541.0, 233.3),
    "full+mynn+noahmp": Footprint("full+mynn+noahmp",
                                  2560 * MIB, 1024 * MIB, 612.0, 279.5),
}


def budget_for(machine: "Machine", fp: Footprint) -> int:
    """What ``machine`` may actually spend on a process running ``fp``'s rung.

    ``Machine.vram_budget_bytes`` withholds :data:`VRAM_HEADROOM`, a
    PERCENTAGE GUESS at the transients that appear on first use.  At the
    RRTMGP rungs one of those transients has been measured instead
    (:data:`RADIATION_TRANSIENT_BYTES`), and the measurement is the bigger
    number on any card below ~34 GiB.

    The reservation is therefore the LARGER of the two and never their sum.
    Stacking them double-counts one set of bytes, and the module docstring
    has the arithmetic: on the card this was measured on, stacking plans a
    tiling 40% smaller than the one that ran, while charging neither plans
    the tiling that would have died at the first radiation call.

    A rung with no measured transient gets exactly
    ``machine.vram_budget_bytes`` back, so every dry and moist plan is
    unmoved.

    Written as ``budget - EXCESS`` rather than as ``vram - max(...)``
    deliberately: the second form derives the headroom by subtraction, so a
    machine whose ``vram_budget_bytes`` is nonsense gets quietly clamped
    back to something plausible -- which defeats the control that breaks
    the budget on purpose to prove the tile search reads it at all.  The
    budget stays the base; only the part of the measured transient that the
    percentage has NOT already withheld comes off it.
    """
    headroom = int(machine.vram_bytes * machine.vram_headroom)
    excess = max(0, fp.radiation_transient_bytes - headroom)
    return int(machine.vram_budget_bytes) - excess


class CannotPlan(RuntimeError):
    """No plan exists.  ``resource`` names what is binding.

    ``resource`` is one of ``"vram"``, ``"host"``, ``"geometry"``.  It is an
    attribute rather than prose because a caller that wants to retry on a
    bigger box needs to know WHICH box to ask for.
    """

    def __init__(self, message: str, resource: str,
                 detail: dict | None = None):
        super().__init__(message)
        self.resource = resource
        self.detail = detail or {}


# --------------------------------------------------------------------------
# the rung, from the config
# --------------------------------------------------------------------------

def rung_of(cfg) -> str:
    """Which measured footprint row ``cfg`` belongs to.

    Deliberately coarse and deliberately pessimistic: a config that turns on
    radiation lands on a ``full`` row even if its scheme selectors differ from
    the ones measured, and anything with MYNN or Noah-MP lands on the dearest
    row of all.  Getting this wrong in the cheap direction hands out a tile
    that does not fit, so where the rungs are ambiguous the expensive one
    wins.  Pass ``footprint=`` to :func:`plan` to override with a measurement
    of your own configuration.

    The "is anything on at all" questions are asked of ``woof.config`` and
    ``woof.core.physics`` rather than of the raw selectors, because the raw
    selectors lie: ``make_config``'s dry default leaves ``ra_sw_physics`` and
    ``ra_lw_physics`` at ``-1``, which is "unset, resolve from the suite" and
    not "RRTMGP is running".  A truthiness test on those two fields alone
    plans every dry domain as though it carried 3.2 GiB of radiation.
    """
    from woof.config import radiation_enabled
    # physics_inventory, not physics: that module's body imports cupy and
    # this classification runs on the estimator path (`woof check` and
    # the wizard price pace through it on installs with no GPU runtime).
    from woof.core.physics_inventory import physics_enabled

    pbl = int(getattr(cfg, "bl_pbl_physics", 0) or 0)
    surface = int(getattr(cfg, "sf_surface_physics", 0) or 0)
    if pbl == 5 or surface == 4:
        return "full+mynn+noahmp"
    if radiation_enabled(cfg) or physics_enabled(cfg):
        return "full"
    if bool(getattr(cfg, "moist", False)) or int(
            getattr(cfg, "mp_physics", 0) or 0):
        return "moist"
    return "dry"


def output_carrier_bytes_per_cell(cfg) -> float:
    """Bytes per domain cell of the persistent output carriers ``cfg`` enables.

    Exactly the slots the store and the tile buffers carry
    (woof.core.output_diagnostics.PERSISTENT) at the shapes their
    initializer allocates, float32, spread over the nx * ny * nz cells the
    footprint is priced in.  Zero when the output controls are off.
    """
    import math
    from woof.core.output_diagnostics import PERSISTENT, scratch_shapes
    shapes = scratch_shapes(cfg)
    carried = sum(4 * math.prod(shapes[slot]) for slot in PERSISTENT if slot in shapes)
    cells = int(cfg.nx) * int(cfg.ny) * int(cfg.nz)
    return carried / cells if carried and cells else 0.0


def footprint_for(cfg, *, rung: str | None = None, radiation_context=None,
                  follower_slots=()) -> Footprint:
    """The :class:`Footprint` to plan ``cfg`` with."""
    key = rung or rung_of(cfg)
    if key not in FOOTPRINTS:
        raise KeyError(f"unknown rung {key!r}; have {sorted(FOOTPRINTS)}")
    from dataclasses import replace
    from woof.core.cfl_inventory import WRF_CFL_BYTES, wrf_cfl_recording_requested
    fp = FOOTPRINTS[key]
    if follower_slots:
        # One independent FP32 horizontal window in each tile buffer and in
        # the canonical full-domain store. Window cells include nz; tracker
        # planes do not. Halo/ring/shadow accounting then follows the ordinary
        # carrier footprint without pricing a full-domain plane per tile.
        per_cell = 4 * len(set(follower_slots)) / int(cfg.nz)
        fp = replace(fp, bytes_per_cell=fp.bytes_per_cell + per_cell,
                     store_bytes_per_cell=fp.store_bytes_per_cell + per_cell,
                     source=fp.source + "; declared follower window carriers")
    output_cell = output_carrier_bytes_per_cell(cfg)
    if output_cell:
        # The operational output controls' carriers (interval precipitation
        # buckets, the gsd_diagnostics maxima and rel_vort) ride the store and
        # every tile buffer (woof.core.streaming's persistent output slots),
        # and no measured rung carries them.  Breakage this prevents: a
        # streamed HRRR-recipe root (1562 x 926 x 50) was refused restoring its
        # store, "store needs 21.12 GiB but budget_bytes is 20.74 GiB"
        # (box D, 2026-10-21 21:xxZ).
        fp = replace(fp, bytes_per_cell=fp.bytes_per_cell + output_cell,
                     store_bytes_per_cell=fp.store_bytes_per_cell + output_cell,
                     source=fp.source + "; operational output carriers")
    if wrf_cfl_recording_requested(cfg):
        fp = replace(fp, domain_fixed_bytes=fp.domain_fixed_bytes + WRF_CFL_BYTES)
    if radiation_context is not None and radiation_context.cam_ozone:
        from woof.core.cam_ozone import memory_increment_per_cell
        device, carried = memory_increment_per_cell(cfg)
        fp = replace(fp, bytes_per_cell=fp.bytes_per_cell + device,
                     store_bytes_per_cell=fp.store_bytes_per_cell + carried,
                     source=fp.source + "; derived CAM ozone allocation bound")
    from woof.config import DEFAULT_COLUMN_CHUNK, radiation_scheme_ids
    if radiation_scheme_ids(cfg)[0] == 1:
        # Bare RunConfig callers retain the preflight API's shared defaults;
        # experiment routes supply their validated explicit vertical/cap pair.
        cap, top = ((DEFAULT_COLUMN_CHUNK, 5000.0) if radiation_context is None else
                    (radiation_context.column_chunk, radiation_context.p_top))
        if top == 0:
            return replace(fp, source=fp.source + "; classic LW pressure top unresolved until initialization")
        fp = replace(fp, classic_lw_context=(int(cfg.nz), float(top), int(cap)))
    return fp


def _forced(cfg) -> bool:
    """``dycore._boundary_forced``: specified or nested forces BOTH axes."""
    return (bool(getattr(cfg, "specified", False))
            or bool(getattr(cfg, "nested", False)))


def edge_band(cfg) -> int:
    """Cells from each forced edge that only an edge-owning tile's interior
    and halo may reach.

    The relaxation frame's width (``lateral_bc._launch_state_relaxation``),
    for a boundary-forced config; 0 otherwise.  See
    :func:`tilestream.spec.edge_band_unowned` for what a plan that breaks
    it would do.
    """
    if not _forced(cfg):
        return 0
    return max(int(getattr(cfg, "spec_zone", 1)),
               int(getattr(cfg, "relax_zone", 4)))


def is_periodic_x(cfg) -> bool:
    """``not dycore._boundary_x(cfg)`` -- does the model wrap in x?

    This is the flag ``spec.plan_tiles`` must be given for the x axis, and it
    is a transcription of the dycore's own predicate rather than a guess: the
    advection launchers, ``slow_pgf``, ``stage_fluxes`` and the small-step uv
    kernels all take ``_boundary_x(cfg) = open_x or specified or nested`` and
    clamp when it is set, wrap when it is not.
    """
    return not (bool(getattr(cfg, "open_x", False)) or _forced(cfg))


def is_periodic_y(cfg) -> bool:
    """``not dycore._boundary_y(cfg)`` -- does the model wrap in y?"""
    return not (bool(getattr(cfg, "open_y", False)) or _forced(cfg))


def is_periodic(cfg) -> bool:
    """True when every lateral-boundary flag is off, i.e. BOTH axes wrap.

    A specified/open/nested domain is NOT periodic, and ``spec.plan_tiles``
    then requires the compute window to fit inside the domain rather than
    wrapping -- a constraint the planner has to honour, because a window
    wider than the domain is a ``ValueError`` and not a slow plan.

    THIS IS THE CONSERVATIVE ANSWER AND IT IS NOT ALWAYS THE RIGHT PLAN.
    ``open_x=True, open_y=False`` is non-periodic in x and genuinely periodic
    in y; planning both axes off this one boolean clamps the y windows and
    silently corrupts the two y-boundary tile rows (measured: nine of nine
    dry carriers after ONE step, rows 0-11 and 180-191 at every x).  A caller
    that owns a plan must ask :func:`is_periodic_x` and :func:`is_periodic_y`
    separately; this stays because "can this domain wrap at all" is still a
    question the sizing model asks.
    """
    return is_periodic_x(cfg) and is_periodic_y(cfg)


# --------------------------------------------------------------------------
# the machine
# --------------------------------------------------------------------------

#: Where the cgroup file systems are mounted, the file naming the cgroup
#: this process runs in on each hierarchy, and the kernel's memory report.
#: Module names so the readers below can be pointed at stand-in files.
_CGROUP_ROOT = "/sys/fs/cgroup"
_PROC_SELF_CGROUP = "/proc/self/cgroup"
_PROC_MEMINFO = "/proc/meminfo"

#: A cgroup v1 limit at or above this is the "no limit" sentinel (about
#: 2**63 rounded down to a page), not a limit.
_CGROUP_UNLIMITED = 1 << 62


def _cgroup_memory_limit() -> int | None:
    """The smallest memory cgroup limit binding this process, or None.

    A total, where :func:`_cgroup_memory_headroom` is room: the ceiling
    :meth:`Machine.detect` sizes the pinned host store against and
    ``woof.core.streaming._host_total_bytes`` prices host RAM against.
    Every limit from the process's own cgroup up to its mount binds
    (:func:`_cgroup_memory_limits`), so the smallest is the ceiling.  None
    when no limit binds, so the caller falls back to MemTotal.

    THE BREAKAGE: this read only the mount root's ``memory.max`` (or v1
    ``memory.limit_in_bytes``).  Inside a systemd scope with
    ``MemoryMax=2G`` on a 30 GiB worker the root carries no limit, so it
    read none, the host total read 32.8 GB, the planner sized its pinned
    host store to the whole host, and the kernel killed the run at the
    scope's limit.  The renderer's and the MPAS builder's readers
    (``rw_host_memory::cgroup_memory_limit``) are held to the same table.
    """
    limits = [limit for limit, _room in _cgroup_memory_limits()]
    return min(limits) if limits else None


def _cgroup_bytes(path: str) -> int | None:
    """One cgroup file's integer, or None for absent, ``max`` or unreadable."""
    try:
        with open(path, "r", encoding="ascii") as handle:
            return int(handle.read().strip())
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def _cgroup_stat_field(path: str, key: str) -> int | None:
    """The value of ``key`` in a ``memory.stat`` file, or None."""
    try:
        with open(path, "r", encoding="ascii") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) >= 2 and fields[0] == key:
                    return int(fields[1])
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return None


def _cgroup_memory_headroom() -> int | None:
    """Bytes this process may still allocate under its memory cgroup limits.

    ``None`` when no limit binds it: the least room of
    :func:`_cgroup_memory_limits`.

    THE BREAKAGE: reading only the mount's root limit and never the usage
    told a process in a limited systemd scope or slice (whose
    ``memory.max`` sits below the root), or one that had already used most
    of its container's limit, that the whole host, or the whole limit, was
    free.  The renderer's reader (``rusty_weather::host_memory``, the
    ``rw-host-memory`` crate) reads the same way, and both are held to one
    table, ``host_memory_cgroup_cases.json`` in that crate.
    """
    rooms = [room for _limit, room in _cgroup_memory_limits()]
    return min(rooms) if rooms else None


def _cgroup_memory_limits() -> list[tuple[int, int]]:
    """``(limit, room)`` for every memory cgroup limit binding this process.

    Every directory from the process's own cgroup (``/proc/self/cgroup``)
    up to its mount is read (:func:`_cgroup_memory_walk`), because a limit
    on an ancestor (a systemd slice, a pod) binds as hard as one on the
    leaf.  The room is the limit minus the working set.  The file pages in
    ``memory.stat`` (``inactive_file`` and ``active_file``, v1
    ``total_inactive_file`` and ``total_active_file``) are not counted as
    used: the kernel reclaims them inside the cgroup before it kills
    anything, so a container that had read its inputs through the page
    cache still has that room.  Measured (2.9.2, the hosted-runner
    emulation, a 16 GiB cgroup after a release build of every crate): the
    cache sat on the ACTIVE list, 15 GiB of it, and counting only the
    inactive list read 0.6 GiB of room, so a CPU preparation was refused
    beside 14 GiB the kernel would have handed over.  A limit whose usage
    cannot be read is the room.  The same answer as
    ``rw_host_memory::cgroup_memory_limits``.
    """
    return [(level.limit, level.room) for level in _cgroup_memory_walk()
            if level.limit is not None]


class _CgroupMemoryLevel(NamedTuple):
    """One directory on this process's memory cgroup path, as read.

    ``limit`` is None when the directory publishes no limit (absent,
    ``max``, unreadable, or the v1 "unlimited" sentinel); ``usage`` is None
    when its usage cannot be read, and ``inactive`` (the reclaimable file
    cache: inactive and active file pages) is then 0.
    """

    limit_file: str
    limit: int | None
    usage: int | None
    inactive: int

    @property
    def room(self) -> int | None:
        """The limit minus the working set; the limit when usage is unread."""
        if self.limit is None:
            return None
        if self.usage is None:
            return self.limit
        return max(0, self.limit - max(0, self.usage - self.inactive))


def _cgroup_memory_walk() -> list[_CgroupMemoryLevel]:
    """Every memory cgroup directory on this process's path, deepest first.

    The one walk every host-memory reader in this package takes: the
    planner's total and headroom (:func:`_cgroup_memory_limits`), the tile
    host store's ``tilestream.hoststore.host_memory``, and the diagnostics
    in ``tilestream.da_stream``, ``tilestream.endure`` and
    ``tilestream.node_probe``.  From the process's own cgroup
    (``/proc/self/cgroup``) up to its mount: cgroup v2 ``memory.max``, used
    ``memory.current``; cgroup v1 ``memory.limit_in_bytes``, used
    ``memory.usage_in_bytes``.  An unreadable ``/proc/self/cgroup`` reads
    the mounts' own roots.  A directory that publishes neither a limit nor
    a usage is left out.

    THE BREAKAGE: each of those readers used to open the mount root's own
    files.  Inside a systemd scope with ``MemoryMax=2G`` on a 30 GiB worker
    the root carries no limit and, on cgroup v2, no usage, so the host
    store admitted a 3 GiB pinned store against the whole host, the DA
    ensemble report said no limit applied and the endurance trace's cgroup
    column read NaN.
    """
    memberships = []
    try:
        with open(_PROC_SELF_CGROUP, "r", encoding="utf-8") as handle:
            for line in handle.read().splitlines():
                fields = line.split(":", 2)
                if len(fields) == 3:
                    memberships.append((fields[1], fields[2]))
    except (OSError, UnicodeDecodeError):
        pass
    if not memberships:
        memberships = [("", "/"), ("memory", "/")]
    levels = []
    for controllers, relative in memberships:
        if controllers == "":
            mounts = [_CGROUP_ROOT]
            names = ("memory.max", "memory.current", ("inactive_file", "active_file"))
        elif "memory" in controllers.split(","):
            mounts = list(dict.fromkeys([os.path.join(_CGROUP_ROOT, controllers),
                                         os.path.join(_CGROUP_ROOT, "memory")]))
            names = ("memory.limit_in_bytes", "memory.usage_in_bytes",
                     ("total_inactive_file", "total_active_file"))
        else:
            continue
        limit_name, usage_name, reclaimable_keys = names
        parts = [part for part in relative.split("/") if part and part != ".."]
        for mount in mounts:
            for depth in range(len(parts), -1, -1):
                directory = os.path.join(mount, *parts[:depth])
                limit_file = os.path.join(directory, limit_name)
                limit = _cgroup_bytes(limit_file)
                if limit is not None and (limit <= 0
                                          or limit >= _CGROUP_UNLIMITED):
                    limit = None
                usage = _cgroup_bytes(os.path.join(directory, usage_name))
                if limit is None and usage is None:
                    continue
                inactive = 0
                if usage is not None:
                    stat_path = os.path.join(directory, "memory.stat")
                    inactive = sum(_cgroup_stat_field(stat_path, key) or 0
                                   for key in reclaimable_keys)
                levels.append(_CgroupMemoryLevel(limit_file, limit, usage,
                                                 inactive))
    return levels


def _in_container() -> bool:
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r", encoding="ascii") as handle:
            body = handle.read()
    except OSError:
        return False
    return any(tag in body for tag in ("docker", "kubepods", "containerd",
                                       "lxc", "libpod"))


@dataclass(frozen=True)
class Machine:
    """What the plan is allowed to spend.

    ``vram_bytes`` is what the card will actually give this process now, not
    what the box has -- :meth:`detect` reads FREE VRAM so another tenant's
    allocation is already deducted.  ``host_bytes`` is the machine's RAM;
    ``host_budget_bytes`` applies the page-locking limit to it.
    """

    vram_bytes: int
    host_bytes: int
    name: str = "unknown"
    vram_headroom: float = VRAM_HEADROOM
    pinned_fraction: float = PINNED_FRACTION
    host_source: str = "explicit"
    device_profile: object | None = None

    @property
    def vram_budget_bytes(self) -> int:
        return int(self.vram_bytes * (1.0 - self.vram_headroom))

    @property
    def host_budget_bytes(self) -> int:
        return int(self.host_bytes * self.pinned_fraction)

    @classmethod
    def detect(cls, *, host_bytes: int | None = None, device: int = 0,
               use_free_vram: bool = True) -> "Machine":
        """Read the machine, refusing to guess where guessing is unsafe.

        VRAM comes from ``cudaMemGetInfo`` capped by device-wide NVML free
        when readable: FREE by default, because a card with a desktop or
        another tenant on it does not have its total to give.  Host RAM comes from the cgroup limit and ``MemTotal``,
        whichever is smaller -- ``/proc/meminfo`` inside a container reports
        the HOST's memory (MEASURED: 503 GiB reported against a 241.7 GiB
        cgroup limit), and a pinned store sized from that number is a plan to
        be OOM-killed.  Containerised with no cgroup limit to read, this
        raises rather than inventing a number.
        """
        import cupy as cp

        selected_device = cp.cuda.Device(device)
        if use_free_vram:
            # Share check's machine-wide cap: on WDDM CUDA may count memory
            # obtainable only by evicting another process. PCI identity keeps
            # CUDA-visible device ordering from selecting a different NVML GPU.
            # The one reading every "free on the card" line uses, so the
            # forecast's stream-init line cannot print a second figure.
            from woof.core.preflight import device_free_and_total_bytes
            free, total = device_free_and_total_bytes(device)
        else:
            with selected_device:
                free, total = cp.cuda.runtime.memGetInfo()
        props = cp.cuda.runtime.getDeviceProperties(device)
        name = props["name"].decode() if isinstance(props["name"], bytes) \
            else str(props["name"])
        profile = None
        if "multiProcessorCount" in props and "maxThreadsPerMultiProcessor" in props:
            from woof.core.preflight import DeviceLocalMemoryProfile
            with selected_device:
                stack_limit = int(cp.cuda.runtime.deviceGetLimit(0))
            profile = DeviceLocalMemoryProfile(
                name, int(props["multiProcessorCount"]),
                int(props["maxThreadsPerMultiProcessor"]),
                stack_limit)

        source = "explicit"
        if host_bytes is None:
            limit = _cgroup_memory_limit()
            meminfo = _host_memtotal()
            memsource = _memtotal_source()
            if limit is not None and meminfo is not None:
                host_bytes = min(limit, meminfo)
                source = ("cgroup limit" if limit <= meminfo else memsource)
            elif limit is not None:
                host_bytes, source = limit, "cgroup limit"
            elif meminfo is not None and not _in_container():
                host_bytes, source = meminfo, memsource
            elif meminfo is not None:
                raise CannotPlan(
                    "containerised with no cgroup memory limit to read, so "
                    f"{memsource} describes the HOST and not this container "
                    "-- pass host_bytes= explicitly rather than letting a "
                    "pinned store be sized from someone else's RAM",
                    "host")
            else:
                # NOT the container case: nothing readable answered at all.
                # Saying "containerised" here was wrong on every Windows box
                # -- where the only unreadable source was procfs, which that
                # OS does not have -- and the wrong word sent the reader
                # looking for a container that was not there.
                raise CannotPlan(
                    f"no host-memory source on this platform ({sys.platform}): "
                    f"{memsource} could not be read and no cgroup limit is "
                    "published -- pass host_bytes= explicitly rather than "
                    "letting a pinned store be sized from a guess",
                    "host")
        return cls(vram_bytes=int(free if use_free_vram else total),
                   host_bytes=int(host_bytes), name=name, host_source=source,
                   device_profile=profile)


def _windows_memory_status() -> tuple[int, int] | None:
    """This Windows box's physical RAM, via ``GlobalMemoryStatusEx``.

    The Win32 equivalent of ``MemTotal``, and the reason it is needed here:
    Windows has no ``/proc``, so the procfs read below finds nothing and a
    planner that treated "no procfs" as "unknowable" refused every streamed
    run on the platform.  ``ullTotalPhys`` is physical RAM exactly as
    ``MemTotal`` is, so the 47% pinning fraction applies to it unchanged.
    """
    import ctypes

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    try:
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(status)):
            return None
    except (AttributeError, OSError):
        # No windll (non-Windows), or the call was refused.  Answering None
        # keeps the "refuse rather than guess" contract intact.
        return None
    total, available = int(status.ullTotalPhys), int(status.ullAvailPhys)
    return (total, available) if total > 0 and 0 <= available <= total else None


def _windows_memtotal() -> int | None:
    status = _windows_memory_status()
    return None if status is None else status[0]


def _host_memavailable() -> int | None:
    """Available physical RAM from the OS, excluding swap/pagefile capacity."""
    if sys.platform == "win32":
        status = _windows_memory_status()
        return None if status is None else status[1]
    try:
        with open(_PROC_MEMINFO, "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    return available if available >= 0 else None
    except (OSError, ValueError, IndexError):
        pass
    return None


def _host_memtotal() -> int | None:
    """This box's physical RAM, from whichever source this OS has.

    Windows first, because on Windows the procfs read below cannot succeed
    and its failure is not evidence of anything.  Everywhere else this is
    ``/proc/meminfo``'s ``MemTotal``, unchanged.
    """
    if sys.platform == "win32":
        return _windows_memtotal()
    try:
        with open(_PROC_MEMINFO, "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _memtotal_source() -> str:
    """What :func:`_host_memtotal` read, named for a receipt."""
    return ("GlobalMemoryStatusEx ullTotalPhys" if sys.platform == "win32"
            else "/proc/meminfo MemTotal")


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def tile_candidates(n: int, *, max_tiles: int = 4096) -> list[int]:
    """Interior tile extents worth considering along an axis of length ``n``.

    ``ceil(n/k)`` for every tile count ``k``, which is what ``plan_tiles``
    would produce for ``k`` tiles and which contains every divisor of ``n`` as
    the subset where the division is exact.  Deduplicated and sorted
    descending, because the search wants the largest first.
    """
    n = int(n)
    out = {n}
    for k in range(1, min(n, max_tiles) + 1):
        out.add(-(-n // k))
    return sorted(out, reverse=True)


def ring_arena_fraction(nx: int, ny: int, tile_nx: int, tile_ny: int,
                        halo: int) -> float:
    """Ring arena as a fraction of the store, for ``write_mode="ring"``.

    Two facts decide it and the obvious formula gets both wrong:

    * **only bands a LATER tile reads are saved.**  A band its predecessors
      read is protected by ordering the read ahead of the write, which halves
      the saved set against a four-sided ``interior - erosion(halo)``
      estimate.  So the kept core is eroded on ONE side per axis, not two.
    * **a ragged trailing tile is read right through.**  Its compute window is
      the same width as every other tile's, so it reaches ``tile + halo - r``
      cells past its own interior (``r`` = the ragged remainder) and its
      neighbour has to save a band that deep instead of ``halo`` deep.

    Per axis, then, the saved depth is ``halo`` for every tile plus that
    overhang once, and the two axes compose as ``1 - (1-fx)(1-fy)``.
    VALIDATED against every arena ``rings.py`` has measured in bytes -- this
    is cells, and the two agree because the staggered ``+1``s are a small
    correction:

    ==================  ========  =========
    plan                measured  this
    ==================  ========  =========
    1950^2, tile 650      4.97%     4.85%
    3276^2, tile 546      5.84%     5.77%
    4608^2, tile 768      4.17%     4.12%
    5120^2, tile 512      6.19%     6.15%
    5395^2, tile 1079     2.98%     2.94%
    5412^2, tile 1353     2.39%     2.35%
    192^2,  tile 64      44.29%    43.75%
    3276^2, tile 512     24.46%    23.92%   <- 13 ragged tiles
    5182^2, tile 512     22.73%    22.70%   <- 21 ragged tiles
    ==================  ========  =========

    Every row is within 0.6 percentage points and always low, which is why the
    caller inflates it; the two ragged rows are the ones that prove the
    overhang term is doing something, since a model without it returns the
    exact-tiling answer for both.
    """
    def axis(n: int, t: int) -> float:
        n, t = int(n), int(t)
        ntiles = -(-n // t)
        remainder = n - (ntiles - 1) * t
        first = halo if remainder == t else min(n, t + halo - remainder)
        return min(1.0, (first + (ntiles - 1) * halo) / n)

    fx, fy = axis(nx, tile_nx), axis(ny, tile_ny)
    return 1.0 - (1.0 - fx) * (1.0 - fy)


#: Work done divided by work needed, for a tiling: the tile geometry's own
#: arithmetic (:func:`tilestream.spec.redundancy`), the one function and not
#: a copy, so the planner and a pinned tiling that never imports the planner
#: state the same number (A177).
redundancy = _spec.redundancy


def suggest_friendly_domains(nx: int, want_tile: int, *, span: int = 64,
                             count: int = 4) -> list[tuple[int, int]]:
    """``(n, tile)`` near ``nx`` that divide exactly at about ``want_tile``.

    Offered when the requested extent has no usable divisor, because "your
    domain is prime" is only half an answer.
    """
    out: list[tuple[int, int]] = []
    for delta in range(0, span + 1):
        for n in ({nx + delta, nx - delta} if delta else {nx}):
            if n < 2 * want_tile:
                continue
            best = max((t for t in _divisors(n)
                        if t <= want_tile), default=0)
            if best and best >= 0.75 * want_tile:
                out.append((n, best))
                if len(out) >= count:
                    return out
    return out


def _divisors(n: int) -> list[int]:
    out = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            out.append(i)
            if i != n // i:
                out.append(n // i)
        i += 1
    return sorted(out)


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    """What to run, and why.

    ``run_kwargs`` hands the whole thing to ``driver.run_tiled`` so that a
    caller never retypes a tile size::

        plan = autoplan.plan(cfg, autoplan.Machine.detect())
        driver.run_tiled(store, cfg, nsteps=N, **plan.run_kwargs)
    """

    mode: str                      # "resident" | "tiled"
    nx: int
    ny: int
    nz: int
    rung: str
    halo: int
    periodic: bool
    tile_nx: int
    tile_ny: int
    nbuffers: int
    write_mode: str
    ntiles_x: int
    ntiles_y: int
    window_nx: int
    window_ny: int
    ragged: bool
    redundancy: float
    vram_bytes: float
    vram_budget_bytes: int
    store_bytes: float
    arena_bytes: float
    host_budget_bytes: int
    footprint: Footprint
    machine: Machine
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    #: Per-axis periodicity, which is what ``plan_tiles`` actually needs.
    #: ``periodic`` above is their conjunction and is kept because it is what
    #: the sizing question ("can a window wrap at all") asks.  Defaulted so a
    #: caller that builds a Plan by hand still gets the old meaning.
    periodic_x: bool = True
    periodic_y: bool = True

    @property
    def cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def window_cells(self) -> int:
        return self.window_nx * self.window_ny * self.nz

    @property
    def ntiles(self) -> int:
        return self.ntiles_x * self.ntiles_y

    @property
    def host_bytes(self) -> float:
        return self.store_bytes + self.arena_bytes

    @property
    def run_kwargs(self) -> dict[str, Any]:
        """Keywords for ``driver.run_tiled`` -- empty for a resident plan."""
        if self.mode == "resident":
            return {}
        return dict(tile_nx=self.tile_nx, tile_ny=self.tile_ny,
                    halo=self.halo, nbuffers=self.nbuffers,
                    write_mode=self.write_mode, periodic=self.periodic,
                    periodic_x=self.periodic_x, periodic_y=self.periodic_y)

    def explain(self) -> str:
        """The plan and the reasoning for it, as text."""
        g = lambda b: f"{b / GIB:.2f} GiB"                      # noqa: E731
        prepared = self.footprint.prepared_memory
        pricing = (f"   ({self.footprint.bytes_per_cell:.0f} B/cell of VRAM,"
                   f" {self.footprint.store_bytes_per_cell:.1f} B/cell of store)"
                   if prepared is None else
                   "   (itemized independent prepared buffers, full unfused radiation peak)")
        lines = [
            "WOOF-ArWen out-of-core plan",
            f"  domain        {self.nx} x {self.ny} x {self.nz}"
            f"  = {self.cells / 1e6:.1f} Mcell"
            f"   ({'periodic' if self.periodic else 'NOT periodic'})",
            f"  rung          {self.rung}{pricing}",
            f"  machine       {self.machine.name}:"
            f" {g(self.machine.vram_bytes)} VRAM"
            f" (budget {g(self.vram_budget_bytes)}),"
            f" {g(self.machine.host_bytes)} host RAM"
            f" (pinnable {g(self.host_budget_bytes)},"
            f" from {self.machine.host_source})",
            "",
            f"DECISION: {self.mode.upper()}",
        ]
        if self.mode == "resident":
            lines += [
                f"  the whole domain fits on the card: {g(self.vram_bytes)}"
                f" of {g(self.vram_budget_bytes)}",
                "  no tiling, no halo, no host store, no pinned memory",
            ]
        else:
            lines += [
                f"  tile          {self.tile_nx} x {self.tile_ny}"
                f"   ({self.ntiles_x} x {self.ntiles_y} = {self.ntiles} tiles,"
                f" {'RAGGED' if self.ragged else 'exact: no ragged tile'})",
                f"  halo          {self.halo}"
                f"   = harness.halo_radius(cfg) = 10 + 3*ns//2"
                f"   (never measured, never tuned)",
                f"  window        {self.window_nx} x {self.window_ny} x"
                f" {self.nz} = {self.window_cells / 1e6:.1f} Mcell"
                f"   -> redundancy {self.redundancy:.3f}x",
                f"  buffers       {self.nbuffers}"
                f"   ({g(self.footprint.buffer_bytes(self.window_cells, (self.window_nx, self.window_ny)))}"
                + (f" each above a {g(self.footprint.process_fixed_bytes)}"
                   " per-process fixed cost)" if prepared is None else
                   " of named pool storage each; separate compute streams)"),
                f"  write mode    {self.write_mode}"
                f"   (arena {self.arena_bytes / max(self.store_bytes, 1):.1%}"
                f" of the store)",
                f"  VRAM          {g(self.vram_bytes)} of"
                f" {g(self.vram_budget_bytes)}",
                f"  pinned host   store {g(self.store_bytes)}"
                f" + arena {g(self.arena_bytes)}"
                f" = {g(self.host_bytes)} of {g(self.host_budget_bytes)}",
            ]
            if prepared is not None:
                fixed = prepared.fixed_terms()
                lines.append(
                    f"  memory basis  {g(fixed['template_resident_bytes'])} retained loader template; "
                    f"{g(fixed['cuda_context_bytes'] + fixed['local_memory_bytes'])} CUDA/local memory; "
                    f"15% allocator headroom and 0.50 GiB residual; "
                    f"{fixed['loader_rows']}-row initialization peak guarded")
        for note in self.notes:
            lines.append(f"  - {note}")
        for warn in self.warnings:
            lines.append(f"  ! {warn}")
        return "\n".join(lines)


def _prepared_memory_basis(fp: Footprint) -> str:
    """Describe the actual itemized costs in a prepared-buffer refusal."""
    fixed = fp.prepared_memory.fixed_terms()
    return (
        f"The itemized envelope includes {fixed['template_resident_bytes'] / GIB:.2f} "
        f"GiB for the retained loader template, "
        f"{(fixed['cuda_context_bytes'] + fixed['local_memory_bytes']) / GIB:.2f} "
        f"GiB for CUDA context and kernel local memory, "
        f"{fixed['k_tables_bytes'] / GIB:.2f} GiB of shared lookup tables, "
        f"each buffer's independent state, scratch and unfused radiation "
        f"storage, 15% allocator headroom and "
        f"{fixed['unmodelled_bytes'] / GIB:.2f} GiB of residual allowance. "
        f"The {fixed['loader_rows']}-row initialization peak is also guarded.")


def plan(cfg, machine: Machine, *, footprint: Footprint | None = None,
         rung: str | None = None, write_mode: str = "ring",
         prefer_resident: bool = True, max_nbuffers: int = 3,
         allow_ragged: bool = True, prefer_exact: bool = True,
         max_redundancy: float | None = MAX_REDUNDANCY,
         minimum_halo: int | None = None) -> Plan:
    """Decide how to run ``cfg`` on ``machine``, or refuse and say why.

    ``write_mode="ring"`` keeps one store plus a few per cent; ``"shadow"``
    keeps two full stores and is only worth asking for when a concurrent
    reader needs the untouched time-t domain.

    ``footprint`` / ``rung`` override the cost model.  Pass ``footprint`` when
    the configuration is not one of the four measured rungs and you have run
    :func:`measure_footprint` on it; ``rung`` when you just want a different
    row of :data:`FOOTPRINTS`.

    ``minimum_halo`` reserves a larger dependency envelope when the caller
    knows the acoustic count can grow, as under adaptive stepping. It never
    reduces the current config's halo; the same value sizes every candidate,
    arena and reported byte count.

    ``max_nbuffers`` caps the pipeline depth.  Two are taken whenever they
    fit even at the cost of a smaller tile (worth 1.32x, measured); a third
    only when it is free at the tile already chosen.

    ``allow_ragged`` and ``prefer_exact`` control what happens when no tile
    divides the domain: by default a ragged tiling is allowed and an exact
    one is preferred inside :data:`ARENA_TIE_BAND`.  ``allow_ragged=False``
    turns "no exact tile fits" into a ``geometry`` refusal, which is the
    accurate answer for an extent like a prime -- a bigger card will not help.

    ``max_redundancy`` refuses a tiling that does more than that multiple of
    the necessary work -- the failure mode where a card is so small that the
    only window that fits is barely wider than two halos, and the run would
    technically proceed while spending most of the machine on halo cells.
    Pass ``None`` to allow it anyway.  :data:`MAX_REDUNDANCY` names what it
    prevents.  The refusal's ``detail`` carries the tiling it refused
    (``tile``, ``ntiles``, ``window``, ``halo``, ``redundancy``, ``limit``)
    so a caller can quote it without parsing the sentence.

    The boundary condition is read off ``cfg`` and constrains the search: on
    a non-periodic domain ``plan_tiles`` clamps the window inside the domain
    instead of wrapping and REFUSES a window wider than the domain, so tile
    plus two halos has to fit in ``nx`` and ``ny``.

    Raises :class:`CannotPlan` with ``.resource`` set to the binding resource
    -- ``"vram"``, ``"host"`` or ``"geometry"``.
    """
    if write_mode not in ("ring", "shadow"):
        raise ValueError(f"write_mode must be 'ring' or 'shadow', "
                         f"got {write_mode!r}")

    nx, ny, nz = int(cfg.nx), int(cfg.ny), int(cfg.nz)
    cells = nx * ny * nz
    halo = _harness.halo_radius(cfg)
    if minimum_halo is not None:
        halo = max(halo, int(minimum_halo))
    band = edge_band(cfg)
    # PER AXIS.  ``open_x`` alone leaves y wrapping, and a plan that clamps
    # a wrapping axis corrupts its two boundary tile rows -- see
    # :func:`is_periodic`.
    periodic_x, periodic_y = is_periodic_x(cfg), is_periodic_y(cfg)
    periodic = periodic_x and periodic_y
    fp = footprint or footprint_for(cfg, rung=rung)
    # NOT ``machine.vram_budget_bytes``: at the RRTMGP rungs the radiation
    # call's measured transient is a bigger reservation than the percentage
    # headroom, and it is the one that decides the tile on a 16 GiB card.
    vram_budget = budget_for(machine, fp)
    host_budget = machine.host_budget_bytes
    # AN ALLOWANCE THE RESERVATIONS ALREADY EXCEED IS REFUSED HERE, before
    # the tile search ever sees the number.  The search itself is total: it
    # finds no window and its refusal prints the budget raw -- which on
    # 2026-08-24 was "no tile fits in -2.68 GiB of VRAM" on a card sitting
    # at 8.38 GiB free, because a front end had frozen a 62 MiB
    # vram_budget_bytes from a probe taken while the card was running
    # another forecast, and the full rung's 2.74 GiB radiation reservation
    # took it negative.  A negative byte count in a user-facing message is
    # itself the defect: the breakage is not "no tile fits", it is that the
    # allowance is spent before any tile is priced, and the refusal has to
    # itemize WHO spent it or the reader cannot tell a full card from a
    # stale number.
    if vram_budget <= 0:
        headroom = int(machine.vram_bytes * machine.vram_headroom)
        excess = max(0, fp.radiation_transient_bytes - headroom)
        floor = fp.vram_bytes((2 * halo + 1) ** 2 * nz, 1)
        if fp.prepared_memory is not None:
            raise CannotPlan(
                f"the VRAM allowance is spent before any tile is priced: "
                f"the planner was handed {machine.vram_bytes / GIB:.2f} GiB "
                f"and {machine.vram_headroom:.0%} first-use headroom withholds "
                f"{headroom / GIB:.2f} GiB. The smallest legal compute window "
                f"at halo {halo} is {2 * halo + 1}^2 x {nz}; its complete "
                f"one-buffer envelope is {floor / GIB:.2f} GiB. "
                f"{_prepared_memory_basis(fp)} Free VRAM or raise the allowance "
                f"before choosing a tile.",
                "vram", dict(vram_allowance_bytes=int(machine.vram_bytes),
                             headroom_withheld_bytes=int(headroom),
                             radiation_transient_bytes=int(fp.radiation_transient_bytes),
                             radiation_transient_excess_bytes=int(excess),
                             floor_bytes=float(floor), vram_budget=int(vram_budget)))
        raise CannotPlan(
            f"the VRAM allowance is spent before any tile is priced: the "
            f"planner was handed {machine.vram_bytes / GIB:.2f} GiB, the "
            f"{machine.vram_headroom:.0%} first-use headroom withholds "
            f"{headroom / GIB:.2f} GiB of it and the {fp.rung} rung's "
            f"measured radiation transient reserves another "
            f"{excess / GIB:.2f} GiB (allocated and freed inside each "
            f"radiation call, so it is held back once per process rather "
            f"than priced into any tile), leaving the tile search "
            f"{max(0, -vram_budget) / GIB:.2f} GiB short of empty.  No tile "
            f"size can help: the smallest window at halo {halo} would still "
            f"add {floor / GIB:.2f} GiB on top "
            f"({fp.process_fixed_bytes / GIB:.2f} GiB of that the "
            f"per-process fixed cost of the {fp.rung} rung), so nothing "
            f"plans below roughly "
            f"{(fp.radiation_transient_bytes + floor) / GIB:.2f} GiB; free "
            f"VRAM, raise the allowance past that, or -- if the allowance "
            f"was captured from a probe while the card was busy -- "
            f"re-derive it from the card as it is now.",
            "vram", dict(vram_allowance_bytes=int(machine.vram_bytes),
                         headroom_withheld_bytes=int(headroom),
                         radiation_transient_bytes=int(
                             fp.radiation_transient_bytes),
                         radiation_transient_excess_bytes=int(excess),
                         floor_bytes=float(floor),
                         vram_budget=int(vram_budget)))
    notes: list[str] = []
    warnings: list[str] = []
    if fp.radiation_transient_bytes:
        notes.append(
            f"{fp.radiation_transient_bytes / GIB:.2f} GiB is reserved for "
            f"the {fp.rung} rung's RRTMGP per-call transient, which is "
            f"allocated and freed inside one radiation step and so is never "
            f"part of a domain's price; it REPLACES the "
            f"{VRAM_HEADROOM:.0%} headroom rather than adding to it")

    # ---------------------------------------------------------- resident?
    resident = fp.resident_bytes(cells)
    if prefer_resident and resident <= vram_budget:
        return Plan(mode="resident", nx=nx, ny=ny, nz=nz, rung=fp.rung,
                    halo=halo, periodic=periodic, periodic_x=periodic_x,
                    periodic_y=periodic_y, tile_nx=nx, tile_ny=ny,
                    nbuffers=1, write_mode="n/a", ntiles_x=1, ntiles_y=1,
                    window_nx=nx, window_ny=ny, ragged=False, redundancy=1.0,
                    vram_bytes=resident, vram_budget_bytes=vram_budget,
                    store_bytes=0.0, arena_bytes=0.0,
                    host_budget_bytes=host_budget, footprint=fp,
                    machine=machine,
                    notes=("a resident run is always faster than a tiled one:"
                           " no halo redundancy, no gather, no scatter",))
    notes.append(
        f"resident would need {resident / GIB:.2f} GiB of VRAM against a "
        f"{vram_budget / GIB:.2f} GiB budget, so the domain is tiled")

    # ------------------------------------------------------- host store?
    store = fp.store_bytes(cells)
    if write_mode == "shadow" and 2.0 * store > host_budget:
        raise CannotPlan(
            f"shadow mode needs two full stores, {2 * store / GIB:.2f} GiB of "
            f"pinned host RAM, and only {host_budget / GIB:.2f} GiB may be "
            f"pinned ({PINNED_FRACTION:.0%} of "
            f"{machine.host_bytes / GIB:.1f} GiB, from {machine.host_source})."
            f"  write_mode='ring' needs one store plus a few per cent.",
            "host", dict(store_bytes=store, host_budget=host_budget))
    if store > host_budget:
        biggest = _largest_square_store(fp, host_budget, nz)
        raise CannotPlan(
            f"the domain store alone is {store / GIB:.2f} GiB of pinned host "
            f"RAM and only {host_budget / GIB:.2f} GiB may be pinned "
            f"({PINNED_FRACTION:.0%} of {machine.host_bytes / GIB:.1f} GiB, "
            f"from {machine.host_source}).  The largest square domain this "
            f"machine can hold at {fp.rung} is {biggest}^2 x {nz}.",
            "host", dict(store_bytes=store, host_budget=host_budget,
                         largest_square=biggest))

    # ------------------------------------------------------ the tile search
    best: dict | None = None
    ragged_only = False
    for nbuffers in range(1, max(1, int(max_nbuffers)) + 1):
        window_cells = _max_window_cells(fp, nbuffers, vram_budget)
        if window_cells <= 0:
            continue
        cand = _fitting_tile(fp, nx, ny, nz, halo, window_cells, nbuffers,
                             vram_budget, periodic_x, periodic_y,
                             allow_ragged, prefer_exact, band=band)
        if cand is None:
            continue
        if cand.get("ragged_only"):
            ragged_only = True
            continue
        score = cand["redundancy"] / (OVERLAP_GAIN if nbuffers >= 2 else 1.0)
        cand.update(nbuffers=nbuffers, score=score)
        if best is None or score < best["score"] - 1e-12:
            best = cand

    if best is None and ragged_only:
        want = _max_window_cells(fp, 1, vram_budget) // nz
        suggestions = suggest_friendly_domains(nx, int(want ** 0.5) or 1)
        raise CannotPlan(
            f"allow_ragged=False and no tile that DIVIDES {nx}x{ny} has a "
            f"compute window small enough to fit "
            f"{vram_budget / GIB:.2f} GiB of VRAM"
            + (f"; {', '.join(f'nx={n} divides by {t}' for n, t in suggestions)}"
               if suggestions else "")
            + ".  Tiles that do not divide the domain would fit; pass "
              "allow_ragged=True to accept the ring arena they cost.",
            "geometry", dict(nx=nx, ny=ny))

    if best is None:
        # WHICH resource emptied the search is derived, not asserted: the
        # same search with an unbounded window says whether ANY tile is
        # geometrically legal.  A scaling sweep hit exactly this seam -- its
        # 32^2 forced-tiled arm was refused "no tile fits in 13.26 GiB of
        # VRAM" while every larger arm planned, because on a non-periodic
        # axis spec.plan_tiles refuses tile + 2*halo > n, so at nx <= 2*halo
        # the candidate set is empty AT ANY BUDGET and the card was never
        # the constraint.  Blaming VRAM there sends the user shopping for a
        # bigger card that would change nothing.
        if _geometry_admits_no_tile(nx, ny, nz, halo, periodic_x, periodic_y,
                                    allow_ragged, band=band):
            raise CannotPlan(
                _too_small_to_tile_message(nx, ny, halo, periodic_x,
                                           periodic_y, fp, cells,
                                           vram_budget, band=band),
                "geometry", dict(nx=nx, ny=ny, halo=halo,
                                 periodic_x=periodic_x,
                                 periodic_y=periodic_y, band=band))
        # The smallest window the geometry admits.  A forced domain's
        # relaxation zone rules out every tiling that puts a seam between
        # halo and zone + halo cells from an edge (spec.edge_band_unowned),
        # so with a zone sized in parent cells that window is far wider
        # than a one-cell tile's, and pricing the one-cell window would
        # tell the user the card holds a tile it cannot run.
        wnx = _smallest_legal_tile(nx, halo, periodic_x, band) + 2 * halo
        wny = _smallest_legal_tile(ny, halo, periodic_y, band) + 2 * halo
        floor = fp.vram_bytes(wnx * wny * nz, 1)
        cost_basis = (
            f". {_prepared_memory_basis(fp)}" if fp.prepared_memory is not None
            else f" ({fp.process_fixed_bytes / GIB:.2f} GiB of that is the "
            f"per-process fixed cost of the {fp.rung} rung, which no tile "
            f"size can reduce).")
        window = (f"{wnx}^2 x {nz}" if wnx == wny else
                  f"{wnx} x {wny} x {nz}")
        zone = (f" (every seam at least zone + halo = {band + halo} cells "
                f"from a forced edge or no more than the halo's {halo} "
                f"cells from it, for the {band}-cell relaxation zone)"
                if band > 0 else "")
        raise CannotPlan(
            f"no tile fits in {vram_budget / GIB:.2f} GiB of VRAM: the "
            f"smallest legal compute window at halo {halo}{zone} is "
            f"{window} and one buffer of it already costs "
            f"{floor / GIB:.2f} GiB"
            f"{cost_basis}",
            "vram", dict(vram_budget=vram_budget, floor_bytes=floor))

    # A second buffer that costs a smaller tile is still usually worth it, but
    # a THIRD is taken only when it is free at the tile already chosen -- the
    # measured 1.32x is what two buffers buy, and nothing was measured for a
    # third beyond "it does not hurt".
    tile_nx, tile_ny = best["tile_nx"], best["tile_ny"]
    window_shape = (best["window_nx"], best["window_ny"])
    window_cells = best["window_nx"] * best["window_ny"] * nz
    nbuffers = best["nbuffers"]
    ntiles = best["ntiles_x"] * best["ntiles_y"]
    # AT THE TILE'S OWN WINDOW from here on.  The search above admitted this
    # tile through a shape-free bound over every rectangle of its area; the
    # plan's price, and the buffer count it can afford, are those of the
    # rectangle it actually chose -- which never costs more than the bound
    # that admitted it, so nothing admitted here was refused there.
    while (nbuffers < max_nbuffers and nbuffers < ntiles
           and fp.vram_bytes(window_cells, nbuffers + 1, window_shape) <= vram_budget):
        nbuffers += 1
    if nbuffers > ntiles:
        nbuffers = max(1, ntiles)
        notes.append("buffer count clamped to the tile count: a buffer that "
                     "never serves a tile is dead VRAM")

    if nbuffers == 1:
        warnings.append(
            "only one tile buffer fits, so no gather overlaps any compute; "
            "the measured cost of that on dry dynamics is 1.32x")

    # ------------------------------------------------------------- arithmetic
    vram = fp.vram_bytes(window_cells, nbuffers, window_shape)
    if write_mode == "ring":
        frac = ring_arena_fraction(nx, ny, tile_nx, tile_ny, halo)
        arena = store * frac * 1.05          # the model runs 0.1-0.6 pp low
    else:
        arena = store
    if store + arena > host_budget:
        raise CannotPlan(
            f"store {store / GIB:.2f} GiB + {write_mode} arena "
            f"{arena / GIB:.2f} GiB = {(store + arena) / GIB:.2f} GiB of "
            f"pinned host RAM against a {host_budget / GIB:.2f} GiB budget."
            + ("  The arena is this large because the tile does not divide "
               "the domain and a ragged trailing tile is read right through."
               if best["ragged"] else ""),
            "host", dict(store_bytes=store, arena_bytes=arena,
                         host_budget=host_budget))

    if max_redundancy is not None and best["redundancy"] > max_redundancy:
        refused = dict(redundancy=best["redundancy"], tile=(tile_nx, tile_ny),
                       ntiles=int(ntiles),
                       window=(best["window_nx"], best["window_ny"]),
                       halo=int(halo), limit=float(max_redundancy),
                       nbuffers=int(nbuffers), vram_bytes=float(vram),
                       vram_budget=int(vram_budget))
        # The resource is derived the same way as the empty-search case
        # above: if an UNBOUNDED budget still cannot beat the limit, the
        # tile is capped by the domain's own geometry (a non-periodic axis
        # admits no tile wider than n - 2*halo; a small periodic domain is
        # halo-dominated whatever the tile) and no card fixes it.  Only when
        # a bigger budget WOULD admit a better tile is this a VRAM problem.
        roomy = _best_tile(nx, ny, nz, halo,
                           _unbounded_window_cells(nx, ny, nz, halo),
                           periodic_x, periodic_y, allow_ragged, prefer_exact,
                           band=band)
        geometry_capped = (roomy is None or roomy.get("ragged_only")
                           or roomy["redundancy"] > max_redundancy)
        if geometry_capped:
            resident_note = (
                f"  This domain's resident footprint is "
                f"{resident / GIB:.2f} GiB against the {vram_budget / GIB:.2f}"
                f" GiB budget, so run it RESIDENT ([tiles] off, or mode = "
                f"'auto') instead of streaming it."
                if resident <= vram_budget else
                f"  Resident needs {resident / GIB:.2f} GiB against "
                f"{vram_budget / GIB:.2f} GiB, so neither shape of this run "
                f"fits this card.")
            clamp = (" a non-periodic axis admits no tile wider than "
                     f"n - 2*halo (x cap {nx - 2 * halo}, y cap "
                     f"{ny - 2 * halo}),"
                     if not (periodic_x and periodic_y) else
                     " every compute window carries 2*halo cells per axis,")
            raise CannotPlan(
                f"the domain is too small to tile efficiently at halo "
                f"{halo}:{clamp} so the best legal tiling is "
                f"{tile_nx}x{tile_ny} inside a "
                f"{best['window_nx']}x{best['window_ny']} window and does "
                f"{best['redundancy']:.2f}x the necessary work (limit "
                f"{max_redundancy:.2f}x) AT ANY BUDGET -- a bigger card "
                f"changes nothing.{resident_note}  Pass max_redundancy=None "
                f"([tiles] max_redundancy = false in a forecast "
                f"configuration) to stream it anyway.",
                "geometry", refused)
        cost_basis = (
            f"the {vram_budget / GIB:.2f} GiB budget limits the complete "
            f"independent-buffer envelope. {_prepared_memory_basis(fp)} "
            if fp.prepared_memory is not None else
            f"{fp.process_fixed_bytes / GIB:.2f} GiB of the "
            f"{vram_budget / GIB:.2f} GiB budget is the {fp.rung} rung's "
            f"per-process fixed cost before any tile exists. ")
        raise CannotPlan(
            f"the largest tile that fits is {tile_nx}x{tile_ny} inside a "
            f"{best['window_nx']}x{best['window_ny']} window, so "
            f"{best['redundancy']:.2f}x of the necessary work would be done "
            f"on halo cells (limit {max_redundancy:.2f}x).  This is a VRAM "
            f"problem: {cost_basis} Pass "
            f"max_redundancy=None ([tiles] max_redundancy = false in a "
            f"forecast configuration) to run it anyway.",
            "vram", refused)

    # ----------------------------------------------------------------- notes
    fraction = arena / store if store else 0.0
    if best["ragged"]:
        trailing = (nx - (best["ntiles_x"] - 1) * tile_nx,
                    ny - (best["ntiles_y"] - 1) * tile_ny)
        line = (f"the tile does NOT divide the domain: the trailing tile is "
                f"{trailing[0]}x{trailing[1]} against {tile_nx}x{tile_ny}, "
                f"and a tile narrower than its neighbours' compute window is "
                f"read right through")
        if write_mode == "ring" and fraction > EXPENSIVE_ARENA:
            suggestions = suggest_friendly_domains(nx, max(tile_nx, tile_ny))
            warnings.append(
                line + f" -- {fraction:.1%} of the store in ring arena, "
                f"against the 2-6% an exact tiling costs"
                + (f"; {', '.join(f'nx={n} divides by {t}' for n, t in suggestions)}"
                   if suggestions else ""))
        else:
            notes.append(
                line + f", but only just: the ring arena is {fraction:.1%} of "
                f"the store, inside the 2-6% an exact tiling costs")
    else:
        notes.append("the tile divides the domain exactly, so there is no "
                     "ragged trailing tile to be read right through")

    if nbuffers > 1 and ntiles < MIN_TILES_PER_BUFFER * nbuffers:
        warnings.append(
            f"{ntiles} tiles against {nbuffers} buffers leaves the pipeline "
            f"little to overlap; in the measured tax curve this is the shape "
            f"where the largest tile lost 10.6% to a smaller one despite "
            f"doing less work.  A tile one step down the ladder may be "
            f"faster here -- pass a smaller max_nbuffers, or accept it")

    if min(best["window_nx"], best["window_ny"]) < MIN_COMPUTE_WINDOW:
        warnings.append(
            f"compute window {best['window_nx']}x{best['window_ny']} is below "
            f"the {MIN_COMPUTE_WINDOW}-cell edge this project benchmarks "
            f"above; the window's arithmetic rate is flat from ~256 up "
            f"(measured), so what this costs is the tiling tax in the "
            f"{best['redundancy']:.3f}x redundancy above, not occupancy")

    notes.append(
        f"halo {halo} from harness.halo_radius(cfg) at time_step_sound="
        f"{int(cfg.time_step_sound)}; a halo below it is bit-wrong AND "
        f"faster, so it is never inferred from a measurement")
    if not periodic:
        clamped = ", ".join(a for a, p in (("x", periodic_x), ("y", periodic_y))
                            if not p)
        notes.append(
            f"the {clamped} axis is specified/open/nested, so its tile "
            f"windows are clamped inside the domain rather than wrapped and "
            f"{best['window_nx']}x{best['window_ny']} had to fit inside "
            f"{nx}x{ny} -- which is what bounds the tile here")

    return Plan(mode="tiled", nx=nx, ny=ny, nz=nz, rung=fp.rung, halo=halo,
                periodic=periodic, periodic_x=periodic_x,
                periodic_y=periodic_y, tile_nx=tile_nx, tile_ny=tile_ny,
                nbuffers=nbuffers, write_mode=write_mode,
                ntiles_x=best["ntiles_x"], ntiles_y=best["ntiles_y"],
                window_nx=best["window_nx"], window_ny=best["window_ny"],
                ragged=best["ragged"], redundancy=best["redundancy"],
                vram_bytes=vram, vram_budget_bytes=vram_budget,
                store_bytes=store, arena_bytes=arena,
                host_budget_bytes=host_budget, footprint=fp, machine=machine,
                notes=tuple(notes), warnings=tuple(warnings))


def _max_window_cells(fp: Footprint, nbuffers: int, budget: int) -> int:
    """Largest compute window, in cells, that ``nbuffers`` buffers can hold."""
    if fp.prepared_memory is not None:
        return fp.prepared_memory.max_window_cells(nbuffers, budget)
    room = (budget / VRAM_SAFETY - CUDA_CONTEXT_BYTES
            - fp.process_fixed_bytes - fp.domain_fixed_bytes)
    room = room / nbuffers - fp.buffer_fixed_bytes
    return int(room / fp.bytes_per_cell) if room > 0 else 0


def _unbounded_window_cells(nx: int, ny: int, nz: int, halo: int) -> int:
    """A window budget no legal candidate can exceed, for feasibility probes.

    The widest window any candidate produces is the whole domain plus a full
    halo on every side, so handing :func:`_best_tile` this many cells asks a
    pure GEOMETRY question: does any tile exist at any budget?
    """
    return (nx + 2 * halo) * (ny + 2 * halo) * nz


def _geometry_admits_no_tile(nx: int, ny: int, nz: int, halo: int,
                             periodic_x: bool, periodic_y: bool,
                             allow_ragged: bool, band: int = 0) -> bool:
    """Whether the tile search is empty AT ANY BUDGET.

    True exactly when the domain's own geometry -- in practice a
    non-periodic axis, where ``spec.plan_tiles`` refuses any window wider
    than the axis, so no tile exists once ``n <= 2*halo`` -- is what empties
    the search.  On a fully periodic domain this is never true: ``tile = n``
    is always a legal candidate there.
    """
    probe = _best_tile(nx, ny, nz, halo,
                       _unbounded_window_cells(nx, ny, nz, halo),
                       periodic_x, periodic_y, allow_ragged, band=band)
    return probe is None or bool(probe.get("ragged_only"))


def _smallest_legal_tile(n: int, halo: int, periodic: bool,
                         band: int) -> int:
    """The narrowest interior extent the tile search admits along one axis.

    1 without a relaxation band, which keeps the one-cell window the VRAM
    refusal has always priced.  With one, the narrowest candidate whose
    window fits the axis and whose tiles' interiors, widened by the halo,
    reach no zone of an edge their windows do not reach
    (:func:`tilestream.spec.edge_band_unowned`); ``n`` when no candidate
    does, a case the geometry refusal reports first.
    """
    if periodic or band <= 0:
        return 1
    for tile in sorted(tile_candidates(n)):
        if tile + 2 * halo <= n and not _spec.edge_band_unowned(
                n, tile, halo, band):
            return tile
    return int(n)


def _too_small_to_tile_message(nx: int, ny: int, halo: int, periodic_x: bool,
                               periodic_y: bool, fp: Footprint, cells: int,
                               vram_budget: int, *, band: int = 0) -> str:
    """The geometry refusal, with the remedy that actually helps.

    A domain in this state is SMALL -- the clamp only empties the search at
    ``n <= 2*halo`` per non-periodic axis -- so the remedy is almost always
    to run it resident, and the message computes that instead of gesturing
    at it.
    """
    caps = []
    if not periodic_x:
        caps.append(f"x admits no tile wider than nx - 2*halo = "
                    f"{nx - 2 * halo}")
    if not periodic_y:
        caps.append(f"y admits no tile wider than ny - 2*halo = "
                    f"{ny - 2 * halo}")
    resident = fp.resident_bytes(cells)
    remedy = (f"this domain's resident footprint is {resident / GIB:.2f} GiB "
              f"against the {vram_budget / GIB:.2f} GiB budget, so run it "
              f"RESIDENT ([tiles] off, or mode = 'auto')"
              if resident <= vram_budget else
              f"resident needs {resident / GIB:.2f} GiB against "
              f"{vram_budget / GIB:.2f} GiB, so neither shape of this run "
              f"fits this card")
    zone = ""
    if band > 0:
        # spec.edge_band_unowned: a zone cell of an edge the tile does not
        # own, within the halo of its interior, runs unrelaxed and the
        # tiled run stops matching the resident one.
        zone = (f"  The domain is boundary-forced with a {band}-cell "
                f"relaxation zone, and every seam has to sit at least "
                f"zone + halo = {band + halo} cells from a forced edge or "
                f"no more than the halo's {halo} cells from it, or the "
                f"tile beside it would run zone cells with no "
                f"relaxation.")
    return (f"the domain cannot be tiled at all at halo {halo}: on a "
            f"non-periodic axis the transport refuses any compute window "
            f"wider than the domain (tile + 2*halo <= n), and at "
            f"{nx}x{ny} that leaves no legal tile -- {'; '.join(caps)}."
            f"{zone}  "
            f"No card size changes this; {remedy}.  A larger domain tiles "
            f"fine, which is why a size sweep sees its SMALLEST arm refused.")


#: How many times :func:`_fitting_tile` narrows the window before giving up
#: on a buffer count; each pass drops at least one column, and in practice
#: one pass clears a step in the price.
_FIT_PASSES = 64


def _fitting_tile(fp: Footprint, nx: int, ny: int, nz: int, halo: int,
                  window_cells: int, nbuffers: int, vram_budget: int,
                  periodic_x: bool, periodic_y: bool, allow_ragged: bool,
                  prefer_exact: bool = True, *, band: int = 0) -> dict | None:
    """:func:`_best_tile`, held to the budget at the window it chose.

    ``window_cells`` comes from a shape-free bound priced at an area at least
    the window's, which admits only what fits while no smaller window costs
    more.  A term that steps DOWN as the window grows breaks that: Morrison
    takes one call (the whole WORK stack, up to 128 MiB) below its floor and
    one-volume chunks above it, so a 224 x 145 window cost 119 MB more than
    the 32,945-column bound that admitted it and the plan exceeded its own
    budget (2.9.7 R5, tests/test_prepared_tile_memory.py, 563 x 326 Morrison
    at 6.41 GiB).  The chosen window is priced exactly; when it does not fit,
    the search runs again strictly below it.  Only the itemized prepared
    model carries such a term; the rung footprint's bound is its price.
    """
    for _ in range(_FIT_PASSES):
        cand = _best_tile(nx, ny, nz, halo, window_cells, periodic_x,
                          periodic_y, allow_ragged, prefer_exact, band=band)
        if cand is None or cand.get("ragged_only") or fp.prepared_memory is None:
            return cand
        cells = cand["window_nx"] * cand["window_ny"] * nz
        if fp.vram_bytes(cells, nbuffers,
                         (cand["window_nx"], cand["window_ny"])) <= vram_budget:
            return cand
        window_cells = cells - nz
        if window_cells <= 0:
            return None
    return None


def _best_tile(nx: int, ny: int, nz: int, halo: int, window_cells: int,
               periodic_x: bool, periodic_y: bool, allow_ragged: bool,
               prefer_exact: bool = True, *, band: int = 0) -> dict | None:
    """The best tile whose compute window fits ``window_cells``.

    Lowest redundancy wins; everything within :data:`ARENA_TIE_BAND` of it is
    decided on ring arena instead, and a tie there goes to the tiling with
    fewer tiles.  The two axes are searched independently, which matters: on a
    12 GiB card a 4096^2 dry domain comes out as 1024 x 512 tiles at 1.096x
    redundancy, against 1.129x for the largest SQUARE tile whose window fits
    the same VRAM.  A square-tile search would have left that on the table.

    ``ragged_only`` in the result says the search found candidates and
    rejected all of them for raggedness, which is a geometry refusal and not
    a VRAM one -- the caller needs to tell the user which.
    """
    cap = window_cells // nz
    if cap <= 0:
        return None
    saw_ragged = False
    feasible: list[tuple[int, int, bool, float]] = []
    for tile_nx in tile_candidates(nx):
        wnx = tile_nx + 2 * halo
        if not periodic_x and wnx > nx:
            continue
        if wnx * (1 + 2 * halo) > cap:
            continue
        # ``band`` (edge_band): no tile's interior, widened by its halo,
        # may reach a forced domain's relaxation zone along an edge its
        # window does not reach -- tilestream.spec.edge_band_unowned.
        if not periodic_x and _spec.edge_band_unowned(nx, tile_nx, halo,
                                                      band):
            continue
        for tile_ny in tile_candidates(ny):
            wny = tile_ny + 2 * halo
            if not periodic_y and wny > ny:
                continue
            if wnx * wny > cap:
                continue
            if not periodic_y and _spec.edge_band_unowned(ny, tile_ny, halo,
                                                          band):
                continue
            ragged = (nx % tile_nx != 0) or (ny % tile_ny != 0)
            if ragged and not allow_ragged:
                saw_ragged = True
                continue
            feasible.append((tile_nx, tile_ny, ragged,
                             redundancy(nx, ny, tile_nx, tile_ny, halo)))
    if not feasible:
        return {"ragged_only": True} if saw_ragged else None

    # Two passes on purpose.  The band has to be measured against the BEST
    # redundancy, and a single pass that prunes against a running incumbent
    # measures it against whatever happened to be first -- which changes the
    # answer, because `_prefer` can replace the incumbent with a candidate
    # whose redundancy is slightly higher.
    floor = min(row[3] for row in feasible)
    best: dict | None = None
    for tile_nx, tile_ny, ragged, red in feasible:
        if red > floor * (1 + ARENA_TIE_BAND):
            continue
        cand = dict(tile_nx=tile_nx, tile_ny=tile_ny,
                    window_nx=tile_nx + 2 * halo, window_ny=tile_ny + 2 * halo,
                    ragged=ragged, redundancy=red,
                    arena_fraction=ring_arena_fraction(nx, ny, tile_nx,
                                                       tile_ny, halo),
                    ntiles_x=-(-nx // tile_nx), ntiles_y=-(-ny // tile_ny))
        if best is None or _prefer(cand, best, prefer_exact):
            best = cand
    return best


def _prefer(cand: dict, best: dict, prefer_exact: bool) -> bool:
    """Is ``cand`` a better tiling than ``best``?

    Both are already inside :data:`ARENA_TIE_BAND` of the lowest redundancy
    available -- :func:`_best_tile` filters on that first -- so what is left
    to decide is: a tile that DIVIDES the domain wins, then the smaller ring
    arena, then fewer tiles, then redundancy after all.

    Division is the primary tie-break rather than the arena because the
    arena model, accurate as it is against every plan ``rings.py`` has
    measured, has never been checked in the barely-ragged regime -- a
    trailing tile two cells short of the others.  Where a measurement exists
    the two rules agree; where none exists the rule that has been measured
    wins.  ``prefer_exact=False`` swaps them, which is worth about 3% of
    redundancy on an extent like 4098 whose divisor ladder has a hole in it.
    """
    if prefer_exact and cand["ragged"] != best["ragged"]:
        return best["ragged"]
    if abs(cand["arena_fraction"] - best["arena_fraction"]) > 1e-9:
        return cand["arena_fraction"] < best["arena_fraction"]
    cand_tiles = cand["ntiles_x"] * cand["ntiles_y"]
    best_tiles = best["ntiles_x"] * best["ntiles_y"]
    if cand_tiles != best_tiles:
        return cand_tiles < best_tiles
    return cand["redundancy"] < best["redundancy"]


def _largest_square_store(fp: Footprint, host_budget: int, nz: int) -> int:
    """Largest ``n`` whose full-domain store fits ``host_budget``."""
    lo, hi = 1, 2
    while fp.store_bytes(hi * hi * nz) <= host_budget:
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fp.store_bytes(mid * mid * nz) <= host_budget:
            lo = mid
        else:
            hi = mid
    return lo


def largest_runnable_domain(machine: Machine, *, rung: str = "dry",
                            nz: int = 49, cfg_kwargs: dict | None = None,
                            **plan_kwargs) -> tuple[int, Plan | None]:
    """The largest square domain ``machine`` can run at ``rung``, and its plan.

    Bisection on a real :func:`plan` call rather than on an inequality, so
    every reported size is one the planner will actually accept -- including
    the tile-geometry and redundancy limits, which an inequality on bytes
    would miss.
    """
    cfg_kwargs = dict(cfg_kwargs or {})

    def ok(n: int) -> Plan | None:
        cfg = _config_for_rung(n, n, nz, rung, **cfg_kwargs)
        try:
            return plan(cfg, machine, rung=rung, **plan_kwargs)
        except CannotPlan:
            return None

    lo, hi = 16, 32
    if ok(lo) is None:
        return 0, None
    while ok(hi) is not None:
        lo, hi = hi, hi * 2
        if hi > 1 << 20:
            break
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if ok(mid) is not None:
            lo = mid
        else:
            hi = mid
    return lo, ok(lo)


#: Config overrides per rung, so the dry-run CLI and the tests can build a
#: config for a named rung without importing the benchmark.  ``test_gate`` is
#: the authority on which of these are bit-exact; this is only their shape.
#: ``ztop=20000`` is essential: the harness default is an 8 km top and
#: RRTMGP then pads past its own 128-layer limit and raises.
_MOIST = dict(moist=True, mp_physics=10, ztop=20000.0)
_FULL = dict(_MOIST, km_opt=4, sf_sfclay_physics=91, bl_pbl_physics=1,
             bldt=0.0, sf_surface_physics=2, ra_sw_physics=4,
             ra_lw_physics=4, radt_minutes=12.0, cu_physics=1,
             cudt_minutes=5.0)
RUNG_OVERRIDES: dict[str, dict] = {
    "dry": dict(),
    "moist": dict(_MOIST),
    "full": dict(_FULL),
    "full+mynn+noahmp": dict(_FULL, sf_sfclay_physics=5, bl_pbl_physics=5,
                             sf_surface_physics=4),
}


def _config_for_rung(nx: int, ny: int, nz: int, rung: str, **overrides):
    kwargs = dict(RUNG_OVERRIDES[rung])
    kwargs.update(overrides)
    return _harness.make_config(nx, ny, nz, **kwargs)


# --------------------------------------------------------------------------
# re-measuring the constants
# --------------------------------------------------------------------------

def measure_footprint(rung: str = "dry", sizes: Sequence[int] = (128, 256),
                      buffers: int = 3, nz: int = 49, warmup: int = 1,
                      report=print) -> Footprint:
    """Re-derive :data:`FOOTPRINTS`'s row for ``rung`` on THIS machine.

    Builds ``buffers`` states of each size in one process and reads
    ``cudaMemGetInfo`` after every one.  The split falls out of the
    arithmetic without any fitting: the marginal cost of buffers 2..k is the
    per-buffer cost, its slope across two sizes is ``bytes_per_cell``, its
    intercept is ``buffer_fixed``, and whatever the FIRST buffer cost above
    the marginal is the per-process fixed cost.

    ``warmup`` must be at least 1.  Kain-Fritsch's ``cumulus/w0avg`` and the
    radiation packer are allocated on first use, so a buffer that has never
    stepped is missing footprint that a running one has -- and radiation
    fires on ``itimestep == 1`` regardless of ``radt``, so one step is
    enough.  The driver's ``call_counts`` are printed to show it did.
    """
    import cupy as cp

    from tilestream import physics_inventory as physinv

    sizes = tuple(int(s) for s in sizes)
    if len(sizes) < 2:
        raise ValueError("two sizes are needed to separate the fixed part "
                         "from the per-cell part")

    def used() -> int:
        free, total = cp.cuda.runtime.memGetInfo()
        return total - free

    warm = cp.zeros(1024, dtype=cp.float64) + 1.0
    cp.cuda.runtime.deviceSynchronize()
    del warm
    cp.get_default_memory_pool().free_all_blocks()
    context = used()
    report(f"CUDA context before any model: {context / MIB:.0f} MiB "
           f"(shipped constant {CUDA_CONTEXT_BYTES / MIB:.0f} MiB)")
    if context > 1.5 * CUDA_CONTEXT_BYTES:
        report("  -- this reading is total-minus-free, so it includes every "
               "OTHER process on the card (a desktop compositor is ~1.5 GiB). "
               "The shipped constant was measured headless; Machine.detect "
               "budgets against FREE VRAM, so the difference is already out.")

    marginal: dict[int, float] = {}
    one_time: list[float] = []
    for n in sizes:
        cfg = _config_for_rung(n, n, nz, rung)
        keep = []
        deltas = []
        # Each size builds from a PRIVATE, empty pool.  A pool that already
        # holds free cached blocks it cannot return (they share a chunk with
        # a live block) cuts the new arrays from that slack with no device
        # allocation, and cudaMemGetInfo under-counts them: MEASURED 0.70 GiB
        # short on a 2.74 GiB dry build at 640^2 x49 (RTX 5090, 2026-10-09).
        # Under-counting here is an under-predicting row, a tile that OOMs.
        pool = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(pool.malloc):
            base = used()
            for _ in range(max(2, int(buffers))):
                state, drv = physinv.default_builder(cfg, 4242)
                if warmup:
                    _harness.run_steps(state, cfg, int(warmup))
                cp.cuda.runtime.deviceSynchronize()
                keep.append(state)
                now = used()
                deltas.append(now - base)
                base = now
        calls = dict(drv.call_counts) if drv is not None else {}
        marg = sum(deltas[1:]) / len(deltas[1:])
        marginal[n * n * nz] = marg
        one_time.append(deltas[0] - marg)
        report(f"  {rung} {n}^2 x {nz}: first buffer "
               f"{deltas[0] / MIB:.0f} MiB, then "
               f"{'/'.join(f'{d / MIB:.0f}' for d in deltas[1:])} MiB"
               f"   -> one-time {(deltas[0] - marg) / MIB:.0f} MiB"
               f"   physics fired {calls or '{}'}")
        del keep, state, drv
        pool.free_all_blocks()
        cp.get_default_memory_pool().free_all_blocks()

    (c0, m0), (c1, m1) = sorted(marginal.items())[:2]
    per_cell = (m1 - m0) / (c1 - c0)
    buffer_fixed = max(0.0, m0 - per_cell * c0)
    process_fixed = max(0.0, sum(one_time) / len(one_time))
    got = Footprint(rung, int(process_fixed), int(buffer_fixed), per_cell,
                    FOOTPRINTS[rung].store_bytes_per_cell, source="measured "
                    "here")
    shipped = FOOTPRINTS[rung]
    report(f"  measured: process {process_fixed / MIB:.0f} MiB, "
           f"buffer {buffer_fixed / MIB:.0f} MiB, {per_cell:.1f} B/cell")
    report(f"  shipped:  process {shipped.process_fixed_bytes / MIB:.0f} MiB, "
           f"buffer {shipped.buffer_fixed_bytes / MIB:.0f} MiB, "
           f"{shipped.bytes_per_cell:.1f} B/cell")
    return got


# --------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------

#: The machines this module's table is quoted for.  ``(VRAM GiB, host GiB)``.
KNOWN_MACHINES: dict[str, tuple[float, float]] = {
    "5090": (32.0, 125.0),
    "4090": (24.0, 128.0),
    "5070": (12.0, 64.0),
}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m tilestream.autoplan",
        description="print the tiling plan and the reasoning for it")
    ap.add_argument("--nx", type=int, default=2048)
    ap.add_argument("--ny", type=int, default=0, help="defaults to --nx")
    ap.add_argument("--nz", type=int, default=49)
    ap.add_argument("--rung", default="dry", choices=sorted(RUNG_OVERRIDES))
    ap.add_argument("--machine", default="", choices=[""] + sorted(KNOWN_MACHINES))
    ap.add_argument("--vram", type=float, default=0.0, help="GiB")
    ap.add_argument("--host", type=float, default=0.0, help="GiB")
    ap.add_argument("--detect", action="store_true",
                    help="read this machine instead of --vram/--host")
    ap.add_argument("--write-mode", default="ring", choices=("ring", "shadow"))
    ap.add_argument("--time-step-sound", type=int, default=4)
    ap.add_argument("--specified", action="store_true",
                    help="specified lateral boundaries (not periodic)")
    ap.add_argument("--table", action="store_true",
                    help="the machine x rung capacity table instead")
    ap.add_argument("--measure", action="store_true",
                    help="re-derive this rung's footprint on this machine")
    ap.add_argument("--measure-sizes", default="128,256")
    args = ap.parse_args(argv)

    if args.table:
        return _print_table()
    if args.measure:
        measure_footprint(args.rung,
                          [int(s) for s in args.measure_sizes.split(",")],
                          nz=args.nz)
        return 0

    if args.detect:
        machine = Machine.detect()
    elif args.machine:
        vram, host = KNOWN_MACHINES[args.machine]
        machine = Machine(int(vram * GIB), int(host * GIB),
                          name=f"RTX {args.machine}")
    elif args.vram and args.host:
        machine = Machine(int(args.vram * GIB), int(args.host * GIB),
                          name="specified")
    else:
        ap.error("give --detect, --machine, or both --vram and --host")

    ny = args.ny or args.nx
    overrides: dict[str, Any] = dict(time_step_sound=args.time_step_sound)
    if args.specified:
        overrides.update(periodic=False, specified=True)
    cfg = _config_for_rung(args.nx, ny, args.nz, args.rung, **overrides)
    try:
        print(plan(cfg, machine, write_mode=args.write_mode).explain())
    except CannotPlan as exc:
        print(f"CANNOT RUN  (binding resource: {exc.resource})\n\n  {exc}")
        return 2
    return 0


#: Reference domains for the capacity table.  1024^2 is where a dry domain
#: still fits resident on a big card and full physics already does not;
#: 4096^2 is past every card; 8192^2 is past every host store but one.
TABLE_DOMAINS = (1024, 2048, 4096, 8192)


def _print_table() -> int:
    """What each machine plans, at each rung, for a ladder of domains."""
    machines = [Machine(int(v * GIB), int(h * GIB), name=f"RTX {k}")
                for k, (v, h) in KNOWN_MACHINES.items()]
    rungs = ("dry", "full+mynn+noahmp")

    print("PLANS  (nz=49, time_step_sound=4 so halo=16, write_mode=ring)\n")
    print(f"{'machine':<10}{'rung':<18}{'domain':>10}{'decision':>10}"
          f"{'tile':>13}{'tiles':>8}{'buf':>5}{'redund':>8}"
          f"{'VRAM':>10}{'pinned host':>13}")
    for machine in machines:
        for rung in rungs:
            for n in TABLE_DOMAINS:
                cfg = _config_for_rung(n, n, 49, rung)
                try:
                    p = plan(cfg, machine, rung=rung)
                except CannotPlan as exc:
                    print(f"{machine.name:<10}{rung:<18}{f'{n}^2':>10}"
                          f"{'REFUSED':>10}   binding resource: {exc.resource}")
                    continue
                tile = (f"{p.tile_nx}x{p.tile_ny}" if p.mode == "tiled"
                        else "-")
                print(f"{machine.name:<10}{rung:<18}{f'{n}^2':>10}"
                      f"{p.mode:>10}{tile:>13}"
                      f"{(p.ntiles if p.mode == 'tiled' else 1):>8}"
                      f"{p.nbuffers:>5}{p.redundancy:>8.3f}"
                      f"{p.vram_bytes / GIB:>9.1f}G"
                      f"{p.host_bytes / GIB:>12.1f}G")

    print("\nCAPACITY  (largest square domain that plans at all)\n")
    print(f"{'machine':<10}{'rung':<18}{'largest resident':>18}"
          f"{'largest tiled':>16}{'tile':>12}{'buf':>5}{'redund':>8}"
          f"{'binding':>10}")
    for machine in machines:
        for rung in rungs:
            n, p = largest_runnable_domain(machine, rung=rung)
            resident = _largest_resident(FOOTPRINTS[rung], machine)
            if p is None:
                print(f"{machine.name:<10}{rung:<18}{'cannot run':>18}")
                continue
            # What actually stops it: ask the planner about one cell more and
            # read the resource off the refusal, rather than guessing.
            try:
                plan(_config_for_rung(n + 1, n + 1, 49, rung), machine,
                     rung=rung)
                binding = "?"
            except CannotPlan as exc:
                binding = exc.resource
            print(f"{machine.name:<10}{rung:<18}{f'{resident}^2':>18}"
                  f"{f'{n}^2':>16}{f'{p.tile_nx}x{p.tile_ny}':>12}"
                  f"{p.nbuffers:>5}{p.redundancy:>8.3f}{binding:>10}")
    return 0


def _largest_resident(fp: Footprint, machine: Machine, nz: int = 49) -> int:
    lo, hi = 1, 2
    budget = machine.vram_budget_bytes
    while fp.resident_bytes(hi * hi * nz) <= budget:
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fp.resident_bytes(mid * mid * nz) <= budget:
            lo = mid
        else:
            hi = mid
    return lo


if __name__ == "__main__":
    raise SystemExit(main())
