"""Memory preflight: itemized estimator, scratch registry, N0 ``--alloc``.

Phase-5 Task 11 (panel lane L4; folds robust-5), implementing architecture
section E of docs/superpowers/specs/2026-07-16-phase5-nesting-architecture.md:
the all-resident four-domain decision is gated by an ENFORCED memory
estimate -- ``woof check --alloc`` constructs every persistent device
allocation for the experiment, runs zero steps, and reports the measured
pool/device numbers against the estimate and the measured WDDM budget.
Measured > estimate is a FAILING GATE (milestone N0, ledger records
``alloc_fits_wddm_budget`` / ``alloc_measured_le_estimate`` /
``alloc_estimate_le_wddm_budget`` -- F11 amendment: every leg blocks).

Three-tier model.  Tier 1 is exact arithmetic; tiers 2/3 are PROVISIONAL
POLICY calibrated on two controller measurements (the d01 run fixture
``n0-preflight-baseline.log`` and the N0
allocation probe ``n0-alloc-probe-r2.json``) -- the tier-2/3 constants
parameterize the reserve proposal for controller ratification; the tests
pin their ALGEBRA (calibration consistency), they do not and cannot
validate the model against independent evidence:

* TIER 1 -- LIVE ARRAYS.  Exact itemized persistent residency from shape
  formulas: (a) ``DomainState`` arrays (transcribed from
  ``woof/core/state.py``); (b) ``PhysicsDriver`` persistents per scheme
  (``woof/core/physics.py``: the surface fields dict, held tendency
  stacks, conditionally retained ``last_ysu`` output, scratch-aliased
  microphysics diagnostics, KF W0AVG + LUT d01-only, RRTMGP lat/lon +
  ozone); (c)
  every named scratch slot from the static registry below; (d) LBC
  residents -- d01 eager interval tables, children's rolling
  one-interval ``nest_*`` tables + F16's arena-aliased full-parent field
  + SINT geometry (the F4 NEST ALLOCATION MANIFEST); (e) the RRTMGP shared
  chunk workspace from
  the phase-maximum chunk formula, plus the per-domain radiation column
  packing and physics-prep transients that coexist with it inside a
  step.  d01 fixture cross-check: itemized residency 1.4544 GiB vs the
  measured full-physics pool-used peak 1.47 GiB (ratio 0.989; the
  residual ~17 MB is per-call KF/coupling transient tails owned by the
  15% headroom).
* TIER 2 -- POOL RETENTION (provisional).  CuPy's pool retains freed
  transient blocks it cannot re-bin.  The N0 probe measured alloc-time
  retention NIL (held - used = 16 MB); the d01 RUN fixture showed
  5.52 GiB held vs 1.47 used, i.e. retention is a run-churn phenomenon.
  ``pool_retention_residual_bytes()`` (fixture held minus the d01
  alloc-estimate basis) is the run-time reserve term; it belongs to the
  N5/N6 run gates, not the N0 allocation gate (split proposed below).
* TIER 3 -- DEVICE-SIDE FOOTPRINT.  ``cudaMemGetInfo`` sees the CUDA
  context, JIT modules, and non-pool allocations on top of the pool.
  The N0 probe measured a fresh allocation-only process at
  ``PROBE_DEVICE_OVERHEAD_BYTES`` = 1.39 GiB; the run fixture's
  apparent 5.72 GiB gap is thereby attributed to 12 h of other-process/
  WDDM drift and RETIRED from the model (kept only as
  ``CAL_FIXTURE_OVERHEAD_BYTES`` for the record).  The probe number is
  a lower bound on run-time overhead (zero steps JIT-compile almost
  nothing), so the run-gate reserve keeps margin above it.

RESERVE POLICY (plan Task 11, amended; controller ratifies at N0,
``--reserve-gib`` overrides): the budget is ALWAYS the MEASURED free
VRAM at startup minus the configured reserve -- never nominal 32 GiB.
Two proposals, split by gate (PENDING RATIFICATION):

* ``ReservePolicy.n0_alloc()`` -- the N0 allocation-gate reserve:
  probe-measured non-pool overhead + external margin (alloc-time
  retention measured nil).  The ``--alloc`` default.
* ``ReservePolicy.run_time()`` -- the N5/N6 run-gate reserve: adds the
  fixture-calibrated run-churn retention residual on top.

Robust-5 fold-ins: a headroom check runs before state construction and
before each high-water phase; persistent scratch is prewarmed at setup;
the RRTMGP ``column_chunk`` is the FIRST over-budget lever
(:func:`recommend_column_chunk`); OOM means record diagnostics and
terminate -- never ``free_all_blocks()``-and-continue.

STAGED-RESIDENCY CONTINGENCY (documented sketch ONLY -- no machinery, per
architecture section E fallback lever (3)): if the enforced chain cannot
fit even after the chunk lever and a transient-scratch shared arena, a
``StagedPlan`` would demote d04 (the dominant block) to staged residency:
its DomainState lives host-side pinned, uploaded per d03-step window,
with the coupler writing boundary tables into a device staging pool.
That is a DESIGN REOPEN materially changing lanes L6/L7 and is flagged as
the top risk; nothing here implements it.

CPU/GPU split: every estimator path is CPU-only (no cupy import); only
:func:`run_alloc_preflight` (the ``--alloc`` mode, controller-run) touches
the device.  The ``woof check`` CLI wiring ships as
:func:`register_cli` per the F2 ownership map -- the one-line ``cli.py``
hookup is a controller handoff commit at merge.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterable, Mapping


import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    # Only the PhaseMemoryEstimate.streamed annotation names it; the
    # streaming module is imported where the envelope is computed.
    from woof.core.streaming import StreamedEnvelope

from woof.config import (CUMULUS_ADVECTIVE_FORCING_SCHEMES,
                          DEFAULT_COLUMN_CHUNK, MYJ_PBL_SCHEME,
                          MYJ_SFCLAY_SCHEME, SASE_PBL_SCHEME,
                          UW_PBL_SCHEME, RunConfig,
                          mynn_mixscalars_active,
                          radiation_enabled, radiation_scheme_ids,
                          soil_layer_count)
from woof.core import kernel_frame_recordings as _kernel_frame_recordings
from woof.core.noahmp_kernel_sources import NOAHMP_PRICING_MODULES
from woof.experiment import DomainConfig, ExperimentConfig
from woof.wrf_exact import ENABLED as WRF_EXACT
# The preparation price's inventories, in the leaf the RW-WPS wheel
# stages; re-exported here under their historical names.
from woof.core.device_inventory import (  # noqa: F401
    CONTEXT_RUNTIME_GROWTH_BYTES, MEASURED_LOCAL_MEMORY_PROFILE,
    MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD,
    DeviceLocalMemoryProfile, _lbc_field_dims, lbc_interval_values,
    state_array_shapes)

# ---------------------------------------------------------------------------
# Calibration fixture (n0-preflight-baseline.log,
# controller-measured 2026-07-16 -- the plan's [PRE-FLIGHT] values).
# ---------------------------------------------------------------------------

GIB = 1024 ** 3

#: WDDM baseline at fixture time: free / total, and the implied non-gpuwm
#: residency (30.27 of 31.84 GiB free -> 1.57 GiB other processes).  The
#: BUDGET is never taken from these numbers -- it is re-measured at every
#: ``--alloc`` startup; the fixture values calibrate tests and projections.
CAL_WDDM_FREE_BYTES = int(30.27 * GIB)
CAL_WDDM_TOTAL_BYTES = int(31.84 * GIB)

#: d01 full-physics fixture: CuPy pool USED peak (step-boundary sampling ->
#: persistent residency), pool HELD (used + retained churn blocks), and the
#: cudaMemGetInfo device-side footprint (held + non-pool overhead).
CAL_D01_POOL_USED_PEAK_BYTES = int(1.47 * GIB)
CAL_D01_POOL_HELD_BYTES = int(5.52 * GIB)
CAL_D01_DEVICE_FOOTPRINT_BYTES = int(11.24 * GIB)

#: Plan-mandated allocator-headroom factor over the itemized subtotal.
#: Since A163 it prices the PREPARATION (ingest) pool and the Windows d01
#: fixture's retention basis only.  The FORECAST's pool margin is
#: :data:`FORECAST_POOL_HEADROOM`, which is measured, because 1.15 under
#: the forecast stacked with the legacy-RRTMG pool slack on the same
#: headroom (see that constant).
ALLOCATOR_HEADROOM = 1.15

#: The default-chunk RRTMGP workspace AS THE d01 FIXTURE ABOVE RAN IT.
#: Pinned, not recomputed.  :func:`pool_retention_residual_bytes` subtracts
#: an enumerated basis from that fixture's measured held bytes, and both
#: sides have to describe the same run: recomputing the workspace term from
#: today's layout means any change that shrinks the workspace shrinks the
#: basis, inflates the "unexplained" residual by exactly what it saved, and
#: hands the run gate a LARGER reserve as its reward for using less memory.
#: The tightening of the RTE phase layouts (978.18 -> 738.50 MiB at this
#: chunk) is the change that surfaced it.  Re-measure the fixture and this
#: constant moves with it; until then it records what was measured.
CAL_D01_WORKSPACE_BYTES = 1025700000

#: Fixture memGetInfo gap = footprint - pool held = 5.72 GiB.  RETIRED as
#: an overhead model by the N0 probe (a fresh allocation-only process
#: measures 1.39 GiB): the difference is attributed to other-process/WDDM
#: drift across the fixture's 12 h run window.  Kept for the record only.
CAL_FIXTURE_OVERHEAD_BYTES = (CAL_D01_DEVICE_FOOTPRINT_BYTES
                              - CAL_D01_POOL_HELD_BYTES)

#: Tier-2 calibration: pool retention beyond live arrays = held - used =
#: 5.52 - 1.47 = 4.05 GiB on the d01 RUN fixture (run churn; the N0
#: allocation probe measured alloc-time retention nil).  The estimate's
#: workspace + transient terms and the 15% headroom already cover most of
#: that churn; the run-gate reserve carries the calibrated residual (see
#: :func:`pool_retention_residual_bytes`, computed against the d01
#: fixture's own alloc-estimate basis so nothing is double-counted).
CAL_D01_POOL_RETENTION_BYTES = (CAL_D01_POOL_HELD_BYTES
                                - CAL_D01_POOL_USED_PEAK_BYTES)

#: Controller N0 allocation probe (
#: n0-alloc-probe-r2.json, 2026-07-16, ``--alloc --reserve-gib 2`` on the
#: fresh box): the full four-domain manifest-driven allocation completed;
#: pool retention at allocation time is nil and the non-pool device
#: overhead of a fresh process is 1.39 GiB.  These are the tier-2/3
#: RE-CALIBRATION measurements the fixture could not provide.
PROBE_POOL_USED_PEAK_BYTES = 26_581_917_184
PROBE_POOL_HELD_BYTES = 26_598_071_296
PROBE_DEVICE_FOOTPRINT_BYTES = 28_088_020_992
PROBE_FREE_BYTES = 32_499_564_544
#: Fresh-process non-pool overhead = footprint - held = 1,489,949,696 B.
#: LOWER BOUND on run-time overhead: zero steps JIT-compile almost none
#: of the kernel modules; the run-gate reserve keeps margin above it.
PROBE_DEVICE_OVERHEAD_BYTES = (PROBE_DEVICE_FOOTPRINT_BYTES
                               - PROBE_POOL_HELD_BYTES)

#: Reserve margin for non-gpuwm residency GROWTH during a 12 h run (the
#: baseline 1.57 GiB other-process residency is already outside "free").
EXTERNAL_MARGIN_BYTES = GIB // 2

#: Default ERA5 forcing cadence for the d01 eager LBC table formula; the
#: real value is owned by CaseDataConfig (Task 2/3) and passed through.
DEFAULT_FORCING_INTERVAL_SECONDS = 21600.0

#: N0 ledger record names (woof/verify/nest_gates.py, F11 amendment).
N0_GATE_METRICS = ("alloc_fits_wddm_budget", "alloc_measured_le_estimate",
                   "alloc_estimate_le_wddm_budget")


def gate_display_name(metric: str, *, vram_gib: float | None = None) -> str:
    """A gate's name as THIS platform should print it.

    The strings in :data:`N0_GATE_METRICS` are pre-registered N0 ledger
    record names, read by ``woof/verify/nest_gates.py`` and written into
    certification receipts, so they are identifiers and must not change
    with the host.  What a reader sees may: "WDDM" is a Windows display
    driver model, and printing ``alloc_fits_wddm_budget`` in the first
    ``woof check`` a Linux user ever runs describes their machine with a
    word that does not apply to it (A-6).  The prose beside these lines
    has been platform-correct since ``envelope_platform`` was introduced;
    only the gate names were left behind.

    Display only.  The key stays the key everywhere it is recorded.
    """
    if envelope_platform(vram_gib=vram_gib) == "linux":
        return metric.replace("_wddm_", "_vram_")
    return metric

#: RETIRED multiplicative machine-peak factors, kept as the historical
#: record and for the standalone child fit (`woof downscale`), which
#: still reads them as a deliberately conservative bound.
#:
#: WINDOWS / WDDM -- 1.75.  The Thompson 12-18Z matched rerun (2026-07-28,
#: the only multi-domain Windows run with whole-run machine-wide VRAM
#: sampling; receipts in docs/thompson-rematch-20260728.md and the campaign
#: handoff) measured a true machine peak of 29,004 MiB = 30,412,898,304 B
#: against this estimator's footprint projection of 17,416,429,288 B -- a
#: ratio of 1.7462.  The projection misses CuPy pool retention across the
#: run and output/checkpoint write transients, so under WDDM it
#: under-projects the number the budget actually confronts.  The single
#: observation is rounded UP to 1.75 (an envelope must not under-round).
#:
#: LINUX -- 1.45 over the ITEMIZED ALLOC ESTIMATE, measured-preliminary.
#:
#: Three independent first-run pilots (2026-07-30) instrumented the
#: machine-wide peak across whole forecasts with nvidia-smi sampling:
#:
#:   ======  =====  ==========  ===========  ==========  ==========
#:   node    card   alloc est.  footprint    machine pk  pk / alloc
#:   ======  =====  ==========  ===========  ==========  ==========
#:   node 1  4090   7.20 GiB    11.31 GiB    9.54 GiB    1.32
#:   node 2  4090   7.29 GiB    11.39 GiB    8.99 GiB    1.23
#:   node 3  4070   3.51 GiB     4.90 GiB    4.04 GiB    1.15
#:   ======  =====  ==========  ===========  ==========  ==========
#:
#: Two things follow, and the second is the important one.
#:
#: 1. The peak lands at 0.79-0.82x the FOOTPRINT PROJECTION, not 1.75x.
#: 2. The footprint projection itself is wrong on Linux, because the two
#:    grid-independent constants it adds to the alloc estimate --
#:    ``pool_retention_residual_bytes`` (2.73 GiB) and
#:    ``PROBE_DEVICE_OVERHEAD_BYTES`` (1.39 GiB) -- are Windows-pool
#:    artifacts that did not appear in any of the three measurements.
#:    At node 3's refused 60x48 minimum layout they were 4.12 GiB of a
#:    5.38 GiB projection: 77% of the floor was constants, which is why
#:    shrinking the grid could not help and a 12 GiB card could not be
#:    sized at all while its GPU sat 66% idle.
#:
#: So on Linux the projection IS the itemized alloc estimate (see
#: :func:`platform_projection_constants`), and this factor is the
#: envelope over THAT.  1.45 clears the worst of the three observations
#: (1.32) by 10%.  Three runs on two card models are still not a
#: calibration: re-derive rather than tune.  Receipts:
#: ARWEN-NODE1-4090-PILOT-20260730.md,
#: ARWEN-NODE2-4090-CUDA128-WORLDWIDE-20260730.md,
#: ARWEN-NODE3-4070-12GB-FLOOR-20260730.md.
#:
#: Both are presented as OBSERVED envelopes, not models, and neither
#: changes any gate: the enforced numbers remain the itemized estimate and
#: the measured --alloc legs.
#: 2026-08-01 AMENDMENT -- the multiplier has no intercept, so its ERROR
#: CHANGES SIGN.  A 16 GiB fleet node (RTX 4080, Linux, driver 595.58.03,
#: machine-wide nvidia-smi sampled at 250 ms, GPU otherwise idle) measured
#: whole forecasts across a 6.6x span of grid sizes and found the x1.45
#: envelope OPTIMISTIC below ~3.5 GiB of itemized estimate and pessimistic
#: by 25-30% above it.  A 224x180 first-run domain was declared 3.99 GiB
#: and peaked at 4.38.  The two fleet datapoints that looked contradictory
#: -- a 5090 under-predicted by ~19%, this 4080 over-predicted by 25-30%
#: -- are one model with a missing intercept read at two grid sizes.
#:
#: The intercept is not a fitted nuisance: it is the NON-POOL residency
#: this module already itemizes (:func:`non_pool_device_bytes` -- the CUDA
#: context plus the launch-time local-memory backing store), which the
#: multiplicative form charged in proportion to the grid when it scales
#: with neither.  See :data:`ENVELOPE_UNMODELLED_BYTES` for the affine
#: replacement and the residuals behind it.
#:
#: These factors are RETIRED from every gate: the 2026-08-19 RTX 3080
#: calibration (below) replaced the WDDM multiplier with a measured
#: affine term, and the Linux lane had already moved to the affine form.
#: The dict is kept as the historical record and for `woof downscale`'s
#: deliberately conservative standalone child fit.
PEAK_ENVELOPE_FACTORS = {"windows": 1.75, "linux": 1.45}

#: How much evidence each RETIRED factor rested on.
PEAK_ENVELOPE_BASIS = {
    "windows": "measured, 1 WDDM run (retired multiplier)",
    "linux": "measured-preliminary, 3 runs (retired multiplier)",
}

#: Retained name for the WDDM factor (the original single-platform value).
OBSERVED_PEAK_OVER_FOOTPRINT = PEAK_ENVELOPE_FACTORS["windows"]

#: WDDM pool slack: what a Windows machine-wide peak carries beyond the
#: affine terms, as a FRACTION of the itemized estimate.
#:
#: MEASURED 2026-08-19, RTX 3080 10 GiB / Windows 11 WDDM / driver-level
#: desktop resident on the same card, machine-wide ``nvidia-smi`` at
#: 0.25 s beside the runtime's own GpuPeakMemoryWatcher receipts, across
#: six whole bare-default ``woof go`` forecasts (single domain, 12 km,
#: 60x48 to 240x192, 6 h GFS, rte-rrtmgp and legacy-RRTMG suites).
#: ``machine-wide peak - desktop baseline`` against
#: ``estimate + itemized non-pool (live profile)``:
#:
#:   =========  ========  =========  =========  =========
#:   run        estimate  non-pool   measured   residual
#:   =========  ========  =========  =========  =========
#:   g60x48     1.26 GiB  1.22 GiB   2.28 GiB   -0.20 GiB
#:   g110x88    1.54 GiB  1.22 GiB   2.60 GiB   -0.16 GiB
#:   g170x136   2.10 GiB  1.22 GiB   3.25 GiB   -0.07 GiB
#:   g240x192   3.05 GiB  1.22 GiB   4.11 GiB   -0.16 GiB
#:   t60x48     2.94 GiB  1.42 GiB   4.62 GiB   +0.27 GiB
#:   t110x88    3.16 GiB  1.42 GiB   5.53 GiB   +0.95 GiB
#:   =========  ========  =========  =========  =========
#:
#: The rte-rrtmgp lane's pool tracks the itemization at 0.95-1.00x and
#: its residuals are NEGATIVE: the affine form alone bounds it.  The
#: only positive residuals are the legacy-RRTMG lane's, where the CuPy
#: pool held up to 1.47x the itemized estimate (call-peak retention the
#: itemization does not model), and they grow with the estimate:
#: +0.09x at t60x48, +0.30x at t110x88 -- worst 0.30x, of which the
#: 0.5 GiB unmodelled constant absorbed 0.16x.  0.20x of the estimate on
#: top of that constant covers the worst measured point with 0.33 GiB to
#: spare and is charged on Windows ONLY: the Linux lane keeps its own
#: measured form untouched (re-measure there before exporting this term
#: -- the retention mechanism is a CuPy pool behaviour, not WDDM's, but
#: no Linux legacy-RRTMG run has been instrumented).
#:
#: What this REPLACES on Windows: the 1.75 multiplier over a footprint
#: projection carrying 4.12 GiB of 5090-derived pool constants.  On the
#: 3080 walk that model predicted 9.91 GiB for a run that measured
#: 2.6 GiB of own contribution (3.8x), refused a fitting card, and its
#: printed remedy refused at every grid size because 78% of the floored
#: envelope was grid-independent.  Receipts:
#: docs/public/receipts/wddm/rtx3080-wddm-calibration-20260819.json
#: (every run's measured and priced terms), beside the walk capture in
#: the RTX 3080 walk capture.
#: 2026-08-20 AMENDMENT (task 206) -- THIS TERM IS NOT WDDM'S, AND IT IS
#: NOT EVERY SUITE'S.  It is the LEGACY-RRTMG call-peak retention, and it
#: splits on the radiation lane on both driver models and all four cards
#: anyone has instrumented.  Pool HELD over the itemized estimate:
#:
#:   ==============  ==========  =================  =================
#:   card            driver      rte-rrtmgp         legacy-RRTMG
#:   ==============  ==========  =================  =================
#:   RTX 3080        WDDM        0.939-0.988        up to 1.47
#:   RTX 5090        Linux       0.879-0.921        1.134, 1.166
#:   RTX 5070 Ti     Linux       0.910-0.956        1.172-1.186
#:   RTX 4080        Linux       0.94-1.00          (not instrumented)
#:   ==============  ==========  =================  =================
#:
#: Three campaigns, two driver models, and the same boundary each time:
#: the legacy engines' LW/SW call-peak workspace is retained by the pool
#: between calls and the itemization does not model it, while the
#: rte-rrtmgp lane's pool tracks the itemization to within a few percent.
#: Charging every suite for a mechanism only one of them has cost an
#: rte-rrtmgp configuration 20% of its estimate for nothing.
#:
#: What was wrong BEFORE this amendment, and it is the safety half: the
#: term was charged by DRIVER MODEL, which meant Linux paid none of it at
#: all.  The Linux envelope then under-predicted every one of fifteen
#: instrumented legacy-RRTMG forecasts:
#: paragraph above said so: "re-measure there before exporting this term
#: -- the retention mechanism is a CuPy pool behaviour, not WDDM's, but
#: no Linux legacy-RRTMG run has been instrumented".  Fifteen have been
#: now, on two Linux cards, and the Linux envelope UNDER-PREDICTED every
#: one of them:
#:
#:   ==============  =======  ==========  ==========  ==========  ========
#:   card            runs     estimate    held/est    measured    residual
#:   ==============  =======  ==========  ==========  ==========  ========
#:   RTX 5070 Ti     10       5.47 GiB    1.172-1.186  7.61-7.69  +0.39..+0.47
#:   RTX 5090         5       5.47 GiB    1.176-1.186  9.08-9.13  +0.69..+0.74
#:   ==============  =======  ==========  ==========  ==========  ========
#:
#: The 2.5.0 release battery's shared 300x300x49 legacy-RRTMG domain,
#: whole 6 h forecasts, each run's own 20 Hz watcher receipt; residual is
#: against the Linux envelope AS IT SHIPPED, i.e. with no slack term at
#: all.  The worst point needs 0.095 of the estimate on top of the
#: 0.5 GiB constant; the WDDM lane's already-shipped 0.20 covers it with
#: better than 2x margin and is the same mechanism measured on the same
#: allocator, so the term is charged on every DRIVER MODEL and priced by
#: RADIATION LANE instead.  Receipts:
#: docs/public/receipts/linux/linux-vram-calibration-20260820.json,
#: docs/public/receipts/wddm/rtx3080-wddm-calibration-20260819.json.
#:
#: RE-MEASURED 2026-08-26 on the same RTX 3080, post-#310 build, after
#: the stale-guard audit of 2026-08-25 asked whether this was calibrated
#: against pre-#310 RRTMG pool behaviour.  Legacy lane, measured as
#: (pool_total_peak - alloc_estimate) / alloc_estimate, which is
#: in-process CuPy-pool accounting and therefore free of the desktop
#: compositor sharing this card:
#:
#:   ==========  ===============  ================
#:   grid        pre-#310 (6 h)   post-#310 (2 h)
#:   ==========  ===============  ================
#:   60x48       0.269            0.213
#:   110x88      0.468            0.150
#:   290x232     --               0.019
#:   ==========  ===============  ================
#:
#: So the retention did fall, and post-#310 it also SHRINKS with grid
#: size (0.32, 0.26, 0.07 GiB absolute) instead of growing -- meaning
#: 0.20 is no longer a worst-case bound: it is marginally exceeded at
#: 60x48 and hugely conservative at 290x232, where memory actually
#: binds.  The term is KEPT and NOT re-fitted here, because the
#: comparison is not like-for-like: the pre-#310 rows are 6 h runs and
#: these are 2 h, and pool retention can build with run length.  A
#: re-fit needs the full six-forecast 6 h protocol -- recorded as a
#: named follow-up in
#: evidence/2026-08-25-stale-guards-engine/named-follow-ups.md.  The
#: rte-rrtmgp lane measured 0.099-0.102 on 6 h runs, uncharged by
#: design and covered by ENVELOPE_UNMODELLED_BYTES at these sizes.
#:
#: RETIRED FROM THE ENVELOPE 2026-09-30 (A163).  It was charged as a
#: fraction of an estimate that already carried the 1.15 allocator
#: headroom, so the legacy lane paid two margins on one headroom, and the
#: call-peak workspace whose retention it priced is itself itemized in the
#: subtotal.  The forecast's one proportional margin is now
#: :data:`FORECAST_POOL_HEADROOM`, measured over a battery that includes
#: the legacy lane.  The value is kept because receipts of earlier
#: releases are written in its terms; no gate reads it.
POOL_SLACK_FRACTION = 0.20

#: The name this term shipped under while it was believed to be a WDDM
#: property.  Kept so the 2.5.0 receipts written in its terms still read.
WDDM_POOL_SLACK_FRACTION = POOL_SLACK_FRACTION

#: What the Windows affine envelope rests on, printed beside the number.
ENVELOPE_WDDM_BASIS = (
    "measured, RTX 3080 10 GiB / Windows 11 WDDM, six whole bare-default "
    "forecasts machine-wide at 0.25 s over a 2.5x span of itemized "
    "estimate, rte-rrtmgp + legacy-RRTMG suites")

#: ...and what the Linux one rests on, since 2026-08-20.
ENVELOPE_LINUX_POOL_BASIS = (
    "measured, RTX 5070 Ti 16 GiB and RTX 5090 32 GiB / Linux, fifteen "
    "whole release-battery forecasts sampled at 20 Hz, CuPy pool held "
    "1.17-1.19x the itemized estimate")


#: The platform names this accounting has evidence for.  ``linux``
#: covers WSL and Linux containers, which report ``linux`` too.
_LINUX_PLATFORMS = ("linux",)
_WINDOWS_PLATFORM_PREFIXES = ("win", "msys")
_WINDOWS_PLATFORM_NAMES = ("cygwin",)



#: What `woof check` says about an experimental two-way configuration.
#:
#: `feedback = 1` is a legal schema value, so a test host G validation run
#: authored one, got a clean PASS and exit 0 here -- output identical to
#: the feedback=0 twin -- and discovered only at prepare, after a 26 s
#: hierarchy build, that the prepared-hierarchy route refused two-way
#: nesting outright.  THAT ROUTE REFUSAL IS LIFTED and this text moved
#: with it: the prepared executor now passes
#: ``skip_feedback_path=(feedback == 0)``
#: (woof/prepared_domain_tree_forecast.py:2203) and the hierarchy
#: stamps the experiment's own setting instead of refusing everything
#: but 0 (woof/source_hierarchy.py:129).  A wizard-built three-domain
#: tree taken through the shipped front doors on real forcing recorded
#: 640 feedback transactions against 0 on the arm that differed only in
#: the two [experiment] keys.  What survives is the experimental stamp
#: and the three shapes the coupler refuses BY NAME when it is built
#: (woof/core/nest.py:232-249) -- those are what a two-way author needs
#: before the ingest rather than after it.  This is an advisory, not a
#: gate: it changes no exit code and blocks nothing.
FEEDBACK_TWO_WAY_ADVISORY = (
    "experimental: feedback = 1 selects two-way nest feedback, which is "
    "stamped experimental and runs on BOTH routes -- the native "
    "experiment-runner route (`woof run`) and the prepared-hierarchy "
    "route, `rw-wps` preparation followed by the domain-tree runner.  "
    "What refuses is not the route but the tree: the nest coupler names "
    "three shapes it cannot feed back when it is built -- unequal "
    "parent/child vertical level counts, mixed parent/child "
    "microphysics, and mismatched active prognostic field inventories.  "
    "A tree clear of those three runs two-way wherever it is launched; "
    "feedback = 0 output is unchanged either way."
)


def feedback_advisory(exp) -> str | None:
    """The two-way advisory when it applies, else None."""

    return (FEEDBACK_TWO_WAY_ADVISORY
            if int(getattr(exp, "feedback", 0) or 0) == 1 else None)


def spawn_reservation_advisories(exp) -> list[str]:
    """One plain sentence per DORMANT (spawn-declared) nest.

    The reservation contract, said where the numbers are: a declared
    spawn-triggered nest is priced by this preflight exactly as if it
    were live -- that is what makes VRAM deterministic and lets this
    report refuse accurately -- so its residency is spent for the whole
    run even if the trigger never fires, and it costs zero compute
    until it spawns.
    """
    import dataclasses as _dc

    lines: list[str] = []
    dormant = [dc for dc in exp.domains
               if getattr(dc, "spawn", None) is not None]
    if not dormant:
        return lines
    full = estimate_experiment(exp).alloc_estimate_bytes
    for dc in dormant:
        without = _dc.replace(exp, domains=tuple(
            d for d in exp.domains if d.grid_id != dc.grid_id))
        delta = full - estimate_experiment(without).alloc_estimate_bytes
        lines.append(
            f"d{dc.grid_id:02d} is a DORMANT spawn-triggered nest "
            f"(trigger {dc.spawn.trigger!r}): declaring it costs "
            f"{delta / GIB:.2f} GiB of this plan's alloc estimate, "
            "reserved from startup and spent even if the trigger never "
            "fires; it costs zero compute until it spawns. Every figure "
            "in this report already includes it.")
    return lines


def anisotropic_w_mixing_advisories(exp) -> list[str]:
    """One line per domain whose per-axis mixing length is over the limit.

    The same sentence :func:`woof.config.warn_anisotropic_w_mixing`
    prints at config load, repeated HERE because that is not where a
    reader is looking.  The 1.6.0 instability that aborted an LES run
    had already been warned about at load, hours earlier, in a stream
    nobody re-read; the run then died 5,467 steps in on a health bound
    that named a vertical velocity and not a cause.  A preflight report
    is the door a user opens before paying for the run, and ``--json``
    puts the same text under ``advisories`` where a script can gate on
    it.

    Advisory, exactly like its neighbours: it changes no exit code and
    blocks nothing.  See ``warn_anisotropic_w_mixing`` for why the
    criterion is not a refusal.
    """

    from woof.config import (anisotropic_w_mixing_advice,
                              auto_mix_isotropic_selection)
    from woof.experiment import (anisotropic_w_mixing_exposure,
                                  auto_selected_isotropic_mixing)

    auto = set(getattr(exp, "auto_mix_isotropic", ()) or ())
    dz_max, exposed, ladder = anisotropic_w_mixing_exposure(exp)
    lines: list[str] = []
    for domain in exposed:
        _, advice = anisotropic_w_mixing_advice(
            where=f"d{domain.grid_id:02d}",
            km_opt=domain.run.km_opt,
            mix_isotropic=domain.run.mix_isotropic,
            mix_upper_bound=domain.run.mix_upper_bound,
            dx=domain.run.dx, dy=domain.run.dy, dz_max=dz_max,
            ladder=ladder, forced=domain.grid_id not in auto)
        if advice:
            lines.append(advice)
    # The domains whose isotropic length was the MODEL'S choice (the
    # 2026-08-16 auto-switch): the report states what the run WILL do --
    # the selection, the ratio and the limit -- rather than an advisory
    # that something is wrong.  A written mix_isotropic = 1 is a
    # legitimate configuration and stays out of this list entirely.
    selected, selected_ladder = auto_selected_isotropic_mixing(exp)
    for grid_id, ratio in selected:
        lines.append(auto_mix_isotropic_selection(
            where=f"d{grid_id:02d}", ratio=ratio, ladder=selected_ladder))
    return lines


#: The child step growth a generated adaptive nest carries
#: (woof.domain_wizard), and the value the advisory below measures from.
GENERATED_NEST_STEP_GROWTH_PCT = 5


def nest_step_growth_advisories(exp) -> list[str]:
    """One line per adaptive nest whose step may grow faster than 5 percent.

    The breakage this names: a 500 m child at ``max_step_increase_pct =
    51`` -- WRF's own nest value, so an imported namelist carries it --
    took a freshly measured CFL after an output alarm, grew to about
    three times its spacing rule within a few steps and went non-finite
    in vertical velocity many hours into a nested forecast.  The same
    forecast at 5 percent ran to completion at the same pace.

    Advisory, not a refusal: 51 is a legal WRF setting that many trees
    run without incident, and refusing it would turn away every imported
    nested namelist.  It changes no exit code and blocks nothing.
    """

    lines: list[str] = []
    for dc in exp.domains:
        run = dc.run
        if int(getattr(dc, "parent_id", 0) or 0) == 0:
            continue
        if not getattr(run, "use_adaptive_time_step", False):
            continue
        growth = int(getattr(run, "max_step_increase_pct",
                             GENERATED_NEST_STEP_GROWTH_PCT))
        if growth <= GENERATED_NEST_STEP_GROWTH_PCT:
            continue
        lines.append(
            f"d{dc.grid_id:02d} is an adaptive nest with "
            f"max_step_increase_pct = {growth}: after an output alarm a "
            "child at this growth has been reproduced taking steps about "
            "three times its spacing rule and then failing with a "
            "non-finite vertical velocity hours into a nested forecast. "
            f"max_step_increase_pct = {GENERATED_NEST_STEP_GROWTH_PCT} on "
            "the nest ran the same forecast to completion at the same "
            "pace; generated nests already use it.")
    return lines


#: "The caller did not price this" -- distinct from a caller that priced
#: it and got ``None``, which is the answer "this config does not stream".
_UNPRICED = object()


def streaming_advisory(exp, *, machine=None,
                       envelope=_UNPRICED, tree_road=_UNPRICED,
                       source=None) -> str | None:
    """Say out loud WHICH allocation this report prices.

    Every ENUMERATION in this module -- ``alloc_estimate_bytes``, the
    itemized peak envelope, the reserve, the whole N0 chain -- prices a
    domain resident in VRAM, and none of them has a concept of a tile
    buffer.  With ``[tiles]`` configured that is a fact the reader has to
    be told, because a report that shows only those figures is otherwise
    indistinguishable from one describing the run they actually asked for.

    What the report's BINDING figures now carry is the streamed envelope
    (:func:`estimate_phases` replaces the forecast term with it, and
    ``check_main`` compares the gate leg and the exit code against it), so
    the sentence below is a statement about the run the config asks for
    rather than a warning that the numbers are about a different one.

    REWRITTEN AT 2.2.0, TWICE OVER.  Both halves of what this used to say
    are now false, and a stale advisory is worse than none: it was the
    release's own headline feature telling users it does not work.

    * "this estimator has no model of a streamed domain" -- it does now.
      :func:`streamed_forecast_envelope` prices the tile working set, the
      buffers and the pinned store off the same measured
      :class:`tilestream.autoplan.Footprint` the run attaches with, and
      :func:`estimate_phases` puts that number in the forecast term.  So a
      refusal here is a statement about the run the config asks for.
    * "'on' is refused by the forecast routes, which wire no
      streamed-domain builder" -- they wire one as of 2.2.0
      (``prepared_single_domain_forecast``, ``prepared_domain_tree_forecast``),
      which is the whole point of the release.

    THE NESTED HALF WAS REWRITTEN AGAIN AT 2.6.1, for the same reason.
    It used to say a refusal or a fit below "describes the resident tree,
    not the mixed-road one the run will take" -- an advisory admitting the
    report had priced a run nobody asked for, and it sat beside an exit 1
    on trees whose mixed road (child streamed, parent resident) fits and
    completes.  Reproduced on the published 2.5.8 and 2.6.0 wheels.  A
    user read that pairing as "streaming has no point".  ``tree_road`` is
    the run door's own per-domain walk
    (:func:`woof.core.streaming.tree_road_plan`, the decide pass of
    ``steppers_for_tree``), so the sentence now states the road the run
    takes and the verdict beside it is asked of the same road.

    What remains worth saying is the one thing that is still true and still
    surprising: under ``mode = "auto"`` the DECISION is the planner's, taken
    against free VRAM at the instant the run starts, so a report written
    now can describe the other branch if the card's occupancy changes.

    Advisory, never a gate: it changes no exit code and blocks nothing,
    on the same posture as every other entry in :func:`check_advisories`.

    ONE DECISION PER REPORT.  ``envelope`` is accepted so a caller that
    has already priced the streamed forecast hands that envelope over
    rather than having this function derive a second one.  Under
    ``mode = "auto"`` the two derivations genuinely disagree: this one
    reached ``Machine.detect`` and planned against the card under the
    desk while the report's verdict planned against the card the reader
    declared, and one ``woof check --budget-gib 6`` printed "2 tile
    buffer(s) of 220x174 = 5.80 GiB" at the top and "2 tile buffer(s) of
    311x146 ... 6.17 GiB" in its binding-phase line.  ``machine`` is the
    same escape for a caller that has a card but not yet an envelope.
    ``source`` is the forcing source a caller that prices here itself
    knows, so the boundary tables it publishes are priced as the review's
    own (:func:`admission_estimate`).
    """
    from woof.core import streaming as _streaming

    # THE TABLES THAT GOVERN THIS TREE'S DOMAINS, not the tree-wide one
    # read raw.  A configuration whose only enabled table is a
    # ``[[domain]]`` row's own returned here with no sentence at all,
    # while every run door streamed that domain -- the report was silent
    # about the one thing this function exists to say out loud.  The mode
    # NAMED is the one that put the configuration on the tiled road: the
    # tree-wide table where that is enabled, and otherwise the first
    # domain table that is.
    options = getattr(exp, "tiles", None) or _streaming.OFF
    domains = tuple(getattr(exp, "domains", ()) or ())
    governing = [_streaming.options_for_domain(dc, options) for dc in domains]
    if not (options.enabled or any(entry.enabled for entry in governing)):
        return None
    mode = (options.mode if options.enabled
            else next(entry.mode for entry in governing if entry.enabled))
    nested_note = ""
    if len(domains) > 1:
        if tree_road is _UNPRICED:
            try:
                # THE SHARED ADMISSION, handed down.  Left out, the walk
                # falls back to an ``estimate_experiment`` of its own and
                # this sentence described a road priced from a third
                # basis, beside a verdict priced from the admission.
                tree_road = _streaming.tree_road_plan(
                    exp, machine=machine,
                    resident_estimate=admission_estimate(
                        exp, machine=machine, source=source),
                    source=source)
            except Exception:        # a report never dies on its estimate
                tree_road = None
        if tree_road is None:
            nested_note = (
                "  This tree is NESTED: roads are assigned per domain "
                "(streaming.steppers_for_tree walks parent-first and prices "
                "each domain against the budget its predecessors left), and "
                "that walk could not be priced here, so the figures below "
                "describe the resident tree.")
        elif tree_road.refusal is not None:
            nested_note = (
                "  This tree is NESTED, and the run door's own per-domain "
                "walk REFUSES it: " + tree_road.refusal
                + "  The figures below therefore describe the resident "
                "tree, which is the only road left to quote.")
        elif getattr(tree_road, "report_error", None):
            nested_note = (
                "  This tree is NESTED and the per-domain pricing walk "
                "FAILED before it could price it (" + tree_road.report_error
                + "); that is a defect in the report, not a refusal of the "
                "tree, so the figures below describe the resident tree.")
        elif not tree_road.streams_any:
            nested_note = (
                "  This tree is NESTED and the run door's own per-domain "
                "walk leaves EVERY domain resident, so the figures below "
                "describe the road the run takes.")
        else:
            # THE WHOLE SENTENCE, not a note appended to a per-domain one.
            # Every sentence below says "this domain", which a tree does
            # not have -- and with the mixed road in the forecast term the
            # envelope handed in IS this plan, so letting it fall through
            # printed the same walk twice, once under a heading claiming a
            # single streamed domain.
            return (
                f"[tiles] mode = '{mode}' over a NESTED tree: the memory "
                "numbers in this report price the MIXED ROAD the run door "
                "actually takes, walked by the same parent-first decide "
                "pass the run uses (streaming.steppers_for_tree), which "
                "prices each domain against the budget its predecessors "
                "left -- " + tree_road.summary() + ".")
    if envelope is _UNPRICED:
        envelope = streamed_forecast_envelope(exp, machine=machine,
                                              source=source)
    if getattr(envelope, "rows", None) is not None:
        # A tree plan reached here on a road that does not stream, or one
        # the walk refused; ``nested_note`` above already states it and
        # the resident figures are the ones being quoted.
        envelope = None
    if envelope is not None:
        return (f"[tiles] mode = '{mode}' streams this domain, so the memory "
                "numbers in this report price the STREAMED allocation and "
                f"not a resident one: {envelope.summary()}." + nested_note)
    if mode == "auto":
        return (
            "[tiles] mode = 'auto' is configured, so whether this domain "
            "streams is decided at run time by tilestream.autoplan against "
            "the free VRAM of that moment.  The numbers below price the "
            "RESIDENT allocation, which is the branch auto takes when the "
            "domain fits; if it does not fit, the run streams instead and "
            "holds a few tile buffers rather than the whole domain, so a "
            "refusal here is not the last word." + nested_note)
    return (
        f"[tiles] mode = '{mode}' is configured but no streamed envelope "
        "could be priced for this domain, so the numbers below price the "
        "RESIDENT allocation.  That happens when the planner can fit no "
        "tile in this card's budget at all, in which case streaming would "
        "not have saved the run either." + nested_note)


from woof.core.pace import UNPRICED as _PACE_UNPRICED


def pace_machine_from_free_bytes(free_bytes: int | None):
    """An ``autoplan.Machine`` for the card THIS REPORT measured.

    The column bound has to be priced against the allowance the reader
    actually has, and ``woof check`` is the one surface that has already
    measured it.  ``vram_headroom`` is zeroed because ``free_bytes`` is
    already what the card will give this process now, and
    ``autoplan.budget_for`` applies the rung's radiation reservation on
    top -- letting the percentage headroom stand as well would withhold
    the same bytes twice, which is the double count
    :func:`tilestream.autoplan.budget_for` exists to avoid.

    ``None`` when nothing measured the card, so the bound is omitted
    rather than guessed.
    """
    if not free_bytes or int(free_bytes) <= 0:
        return None
    from tilestream import autoplan

    host = _host_total_bytes_or_none()
    return autoplan.Machine(
        vram_bytes=int(free_bytes),
        host_bytes=int(host or free_bytes), name="measured free VRAM",
        vram_headroom=0.0, host_source="woof check")


def _host_total_bytes_or_none() -> int | None:
    from woof.core.streaming import _host_total_bytes

    try:
        return _host_total_bytes()
    except Exception:
        return None


def host_available_bytes() -> int | None:
    """Available physical RAM, capped at the process's total memory ceiling.

    Uses Linux MemAvailable or Windows GlobalMemoryStatusEx ullAvailPhys
    through the planner's shared OS probe, capped by the room left under
    the memory cgroup limits this process runs in
    (:func:`tilestream.autoplan._cgroup_memory_headroom`), the same reading
    the renderer's ``rusty_weather::host_memory`` makes. Unknown stays
    unknown; zero means no available RAM rather than a skipped comparison.
    """
    from tilestream.autoplan import _cgroup_memory_headroom, _host_memavailable

    available = _host_memavailable()
    if available is None:
        return None
    caps = [int(available), _host_total_bytes_or_none(), _cgroup_memory_headroom()]
    return min(int(cap) for cap in caps if cap is not None)


def pace_advisory(exp, *, streamed=_PACE_UNPRICED, machine=None) -> str | None:
    """The pace sentence, re-exported so ``check_main`` reads one name.

    The default is the PACE module's own sentinel, not ``None``.  Passing
    ``None`` here would be the positive claim "this plan runs resident",
    and a re-export that quietly made that claim reported the resident
    road for a config whose [tiles] table streams.
    """
    from woof.core.pace import pace_advisory as _pace_advisory

    return _pace_advisory(exp, streamed=streamed, machine=machine)


def pace_estimate_for_report(exp, *, streamed=None, free_bytes=None):
    """The pace this report should publish, priced against its own card."""
    from woof.core.pace import estimate_pace

    return estimate_pace(exp, streamed=streamed,
                         machine=pace_machine_from_free_bytes(free_bytes))


def check_advisories(exp, config_path=None, *, machine=None,
                     streamed=_UNPRICED, tree_road=_UNPRICED,
                     source=None) -> list[str]:
    """Every route advisory this config earns, in report order.

    Same posture as ``feedback_advisory``: these change no exit code and
    block nothing.  They exist because a legal config that a later stage
    silently ignores is worse than one it refuses -- the user learns
    after paying for the run instead of before it.

    ``machine`` and ``streamed`` are relayed to :func:`streaming_advisory`
    so a report that has already resolved ``[tiles]`` describes the
    tiling it is about to quote numbers for, and not a second one;
    ``source`` too, for a report that prices here itself.
    """

    from woof.checkpoint_routes import (
        checkpoint_route_advisory, config_has_case_data)

    # ONE DECISION, resolved once and given to both sentences below.
    # Deriving it twice is the defect ``streaming_advisory``'s docstring
    # records: under ``mode = "auto"`` two derivations planned against
    # two different cards and one report printed two tilings.
    if streamed is _UNPRICED:
        streamed = streamed_forecast_envelope(exp, machine=machine,
                                              source=source)
    advisories = [feedback_advisory(exp)]
    advisories.extend(spawn_reservation_advisories(exp))
    advisories.extend(anisotropic_w_mixing_advisories(exp))
    advisories.extend(nest_step_growth_advisories(exp))
    advisories.append(streaming_advisory(exp, machine=machine,
                                         envelope=streamed,
                                         tree_road=tree_road,
                                         source=source))
    # THE PACE, immediately after the sentence that says which road this
    # config takes -- the two belong together, because "this domain
    # streams" is only actionable next to what streaming costs and what
    # domain size would not.
    # THE PACE IS NOT AN ADVISORY, and putting it here was wrong.  Every
    # entry in this list is EARNED -- a config that turned something on
    # gets told what that costs it -- and the negative control in
    # tests/test_checkpoint_route_contract.py pins that a plain config
    # earns none, so "a green check stays green for everybody else".  The
    # pace is owed to every run, including the plainest, so it is printed
    # on its own line by ``check_main`` and published as the
    # ``expected_pace`` object in the JSON, not smuggled into a list
    # whose contract is the opposite.
    if config_path is not None:
        advisories.append(checkpoint_route_advisory(
            domain_count=len(exp.domains),
            has_case_data=config_has_case_data(config_path),
            restart_interval_s=getattr(exp, "restart_interval_s", 0.0)))
    return [line for line in advisories if line]


def host_platform() -> str:
    """The platform every default in this module prices for.

    One seam instead of three bare reads of :data:`sys.platform`, so a
    test that wants the Windows envelope on a Linux box patches THIS
    function and nothing else.  Patching ``sys.platform`` itself reaches
    every library in the process: ``shutil.which`` reads it to decide
    whether to consult a Windows-only ``_winapi`` attribute, and the
    bridge build ``woof check`` runs for an ERA5 forcing schedule asks
    ``shutil.which("cargo")``, so three preflight tests that patched the
    global name died on Linux with ``AttributeError:
    NeedCurrentDirectoryForExePath`` before reaching the report they
    assert on (proof/node-reds-276).
    """

    return sys.platform


def platform_is_measured(platform: str | None = None) -> bool:
    """Has this platform's memory behaviour actually been observed?

    Windows (including Cygwin and MSYS, which run on WDDM) and Linux
    have measurements behind them.  Nothing else does.
    """

    name = host_platform() if platform is None else str(platform)
    return (name.startswith(_WINDOWS_PLATFORM_PREFIXES)
            or name in _WINDOWS_PLATFORM_NAMES
            or name.startswith(_LINUX_PLATFORMS))


def unknown_platform_note(platform: str | None = None) -> str | None:
    """One line for a platform with no measurements, else ``None``."""

    name = host_platform() if platform is None else str(platform)
    if platform_is_measured(name):
        return None
    return (
        f"note: platform {name!r} has no VRAM measurements behind it "
        f"(only Windows and Linux do), so the conservative Windows/WDDM "
        f"accounting is applied -- it may size a smaller domain than "
        f"this machine can run.")


def envelope_platform(platform: str | None = None,
                      vram_gib: float | None = None) -> str:
    """Which envelope family applies: ``windows`` or ``linux``.

    PLATFORM defaults to :func:`host_platform`.

    Two platforms have measurements: Windows -- with Cygwin and MSYS,
    which are the same WDDM driver under a different shell -- and Linux,
    which is also what WSL and Linux containers report.  Anything else
    is a platform nobody has measured, and it takes the **conservative**
    (Windows) accounting rather than the optimistic one.

    That is a change from v1.0.0, which returned ``linux`` for every
    non-Windows name and so quietly priced an unknown platform with the
    envelope that omits 4.12 GiB of fixed constants.  Fail-open on an
    unsupported platform is the wrong direction: the Linux numbers are
    not a default, they are three runs on two Linux cards.  Callers
    should print :func:`unknown_platform_note` beside the sizing so the
    substitution is visible rather than silent.

    VRAM_GIB is accepted for callers that know the card and is
    deliberately IGNORED.  It used to select an experimental
    "windows-small" tier at or under 12 GiB, which is how the wizard
    (which knows the card size) and ``woof check`` / ``woof go``
    (which measure free VRAM and passed no size) priced the very same
    bytes with two different formulas -- the wizard's inline check said
    PASS and the standalone check exited 4 seconds later (open task
    #162; the 2026-08-19 3080 walk).  One machine gets ONE envelope
    family, decided by the platform alone; the 3080 calibration that
    made the Windows family measured (:data:`WDDM_POOL_SLACK_FRACTION`)
    is what retired the tier.
    """

    name = host_platform() if platform is None else str(platform)
    if name.startswith(_LINUX_PLATFORMS):
        return "linux"
    return "windows"


def peak_envelope_factor(platform: str | None = None,
                         vram_gib: float | None = None) -> float:
    """The RETIRED multiplicative factor for this platform.

    Nothing gate-side reads it any more; ``woof downscale``'s
    standalone child fit keeps it as a deliberately conservative bound.
    """

    return PEAK_ENVELOPE_FACTORS[envelope_platform(platform, vram_gib)]


def platform_projection_constants(
        platform: str | None = None,
        vram_gib: float | None = None) -> tuple[int, int]:
    """``(retention_residual, device_overhead)`` for the projection.

    Both are grid-independent constants calibrated on one Windows/5090
    fixture, and neither showed up in any of the three instrumented
    Linux runs -- whose peaks tracked the itemized alloc estimate to
    within 1.15-1.32x.  Adding 4.12 GiB of Windows pool accounting to a
    Linux projection is not conservatism, it is a wrong number: it put
    the 12 GiB tier out of reach entirely.  Zero on Linux, unchanged on
    Windows.

    These feed the TIER 2/3 projection DISPLAY lines and nothing else:
    since the 3080 calibration they are not envelope terms on any
    platform (:attr:`ExperimentMemoryEstimate.envelope_intercept_bytes`
    is the itemized non-pool residency alone).
    """

    family = envelope_platform(platform, vram_gib)
    if family == "windows":
        return pool_retention_residual_bytes(), PROBE_DEVICE_OVERHEAD_BYTES
    return 0, 0


#: What the machine-wide peak carries beyond the itemized estimate and
#: the itemized non-pool residency: allocator fragmentation, the driver's
#: own working set, and whatever the shape formulas do not enumerate.
#:
#: MEASURED 2026-08-01, RTX 4080 16 GiB / Linux / driver 595.58.03, GPU
#: otherwise idle, machine-wide ``nvidia-smi`` sampled every 250 ms across
#: whole prepared-cache forecasts.  Fitting ``peak = a x subtotal + b``
#: over the single-domain runs returns a = 0.98 -- i.e. the itemization
#: predicts the pool 1:1 and the residue is a CONSTANT, which is exactly
#: what a CUDA context plus a launch-time local-memory backing store is.
#: :data:`ENVELOPE_UNMODELLED_BYTES` is the part of that constant this
#: module does not already itemize, rounded UP over the worst residual
#: (an envelope must never round down).
#:
#: Residuals of ``measured - (alloc estimate + non-pool)``, single domain,
#: staged prepared-cache route unless marked:
#:
#:   ===========  ==========  ==========  ==========  ==========
#:   run          grid        estimate    measured    residual
#:   ===========  ==========  ==========  ==========  ==========
#:   s07          170x136      2.07 GiB    3.65 GiB   +0.05 GiB
#:   small8       224x180      2.75 GiB    4.14 GiB   -0.14 GiB
#:   small8 (go)  224x180      2.75 GiB    4.38 GiB   +0.10 GiB
#:   s11          340x272      4.82 GiB    5.95 GiB   -0.41 GiB
#:   edge15 (go)  448x360      7.56 GiB    8.75 GiB   -0.34 GiB
#:   L12    (go)  474x378      8.27 GiB    9.25 GiB   -0.55 GiB
#:   over22 (go)  594x476     12.38 GiB   12.59 GiB   -1.33 GiB
#:   big24  (go)  630x504     13.76 GiB   13.88 GiB   -1.42 GiB
#:   ===========  ==========  ==========  ==========  ==========
#:
#: The worst positive residual is +0.10 GiB; the same three-node Linux
#: pilot set of 2026-07-30, re-read through this model, leaves +0.46 GiB
#: of margin at its tightest (node 1, 4090).  A round half-gibibyte
#: covers both with room, and unlike a multiplier it does not grow with
#: the grid -- which is the whole point of having an intercept.
ENVELOPE_UNMODELLED_BYTES = GIB // 2

#: Per NEST beyond the root, as a FRACTION of the itemized estimate.
#:
#: A tree lands consistently above the single-domain line, because the
#: shared scratch arena and the shared dycore-state workspace are priced
#: at their per-slot maximum while the nest coupler's own buffers, the
#: extra per-domain physics drivers and the per-domain output staging are
#: not on that maximum.  Measured, same card and instrument, as
#: ``measured - (estimate + non-pool)`` over the estimate:
#:
#:   =========  =======  ==========  ==========  ==============
#:   run        domains  estimate    residual    per extra dom.
#:   =========  =======  ==========  ==========  ==============
#:   n10        2         4.12 GiB   +0.18 GiB      4.3%
#:   n2_16      2         8.22 GiB   +0.34 GiB      4.2%
#:   c07        4         2.06 GiB   +0.26 GiB      4.2%
#:   =========  =======  ==========  ==========  ==============
#:
#: Three trees, two depths, a 4x span of estimate, and the same number
#: each time -- so it is PROPORTIONAL, not a flat allowance, and 0.05
#: rounds it up.  Zero for a single domain by construction, so no
#: single-domain number in this release moves because of it.
ENVELOPE_PER_NEST_FRACTION = 0.05

#: What the affine envelope rests on, printed beside the number it makes.
ENVELOPE_AFFINE_BASIS = (
    "measured, RTX 4080 16 GiB / Linux, whole forecasts machine-wide at "
    "250 ms across a 6.6x span of itemized estimate, 1/2/4 domains")

#: A163: THE FORECAST'S POOL MARGIN, MEASURED AND COUNTED ONCE.
#:
#: THE DEFECT.  The envelope multiplied the itemized subtotal by the
#: plan's 1.15 allocator headroom, and then, on the legacy-RRTMG lane,
#: added 20% of THAT product as pool slack.  Both margins price the same
#: thing -- what the CuPy pool holds beyond the itemized arrays -- so the
#: legacy lane paid it twice: 1.38x the subtotal before the non-pool
#: intercept.  HRRR physics on HRRR's full grid (1797 x 1057 x 50, RTX PRO
#: 6000) itemized 83.35 GiB and was priced at 119.09 GiB against a
#: 93.93 GiB budget, where the probe's measured ratio puts it near
#: 93.4 GiB; urban BEP+BEM on the two 750 m city trees was priced at
#: 29.6 GiB and ran to the end on a 24 GiB RTX 4090.  On the rte-rrtmgp
#: lane the same 1.15 was all the margin there was.  The legacy engines'
#: LW/SW call-peak workspace the 20% term was measured against is itself
#: an ITEMIZED term of the subtotal (``legacy_radiation_vram_bytes``, the
#: pure-legacy ``workspace_bytes``), so its retention is inside the
#: measured ratio below and not a separate mechanism.
#:
#: THE MEASUREMENT.  ``(device peak - itemized non-pool - unmodelled
#: residue - held arrays) / (itemized subtotal - held arrays)`` over the
#: A163 battery: the smallest margin under which the envelope, which adds
#: the itemized non-pool, :data:`ENVELOPE_UNMODELLED_BYTES` and the held
#: arrays (the urban arrays priced at their allocated size,
#: :func:`forecast_pool_estimate_bytes`; zero off urban) each once, holds
#: that row's measured peak.  A ratio taken over the residue as well would
#: price the residue twice, once in the margin and once beside it, the
#: defect this section exists to retire.  The rows: 20 min HRRR-driven
#: forecasts spanning the first radiation call and a history write, a 6 h
#: legacy-RRTMG forecast, a ``[tiles]`` forecast and three trees, each
#: forecast's own process tree sampled at 100 ms through NVML beside the
#: runtime's 20 Hz watcher, on an RTX 5070 Ti 16 GB (test host D, card held
#: alone) and an RTX 5090 32 GB (test host B, the run's own processes).  The
#: two CONUS 3 km forecasts of 2026-09-30 on the RTX PRO 6000 are carried
#: as rows too.  The table lives beside the constant it calibrates:
#: :data:`FORECAST_PEAK_BATTERY`.
#:
#: The intercept is the ITEMIZED non-pool term, which covers what the
#: cards measured (device peak minus pool-held peak) by a configuration's
#: own amount: 0.72 GB measured against 1.02 GB itemized for the default
#: suite without Kain-Fritsch on the 5070 Ti, against 1.86 GB with it, and
#: 1.2-1.5 against 2.19-4.23 on the 5090.  So the ratio is taken over the
#: whole device peak, never over the pool with a second allowance beside
#: it.
#:
#: Over every battery row whose itemization is complete (pool USED
#: 0.54x the subtotal on the ``[tiles]`` row, 0.92-1.06x on the rest) the
#: worst is 1.085: the default suite without Kain-Fritsch at 500 x 500 x
#: 49 with its default products (render while running), 20 min, on the
#: 5070 Ti, whose itemized non-pool covers the measured one by the least;
#: the same configuration on the 5090 measured 1.004.  The row that set the
#: first repair's margin before counted a second CUDA process that was not
#: the forecast's, and it is replaced by that 5090 re-measure, which names
#: every process of the run.  The legacy lane, which the 20% term was
#: charged to, measured 0.34-1.06 on current code, and its peak does not
#: grow with run length: 6 h of the default suite with legacy RRTMG at
#: 500 x 500 x 49 peaked 139 s after launch at 1.002 and the 6 h after
#: never exceeded it.  The default-suite legacy-RRTMG 300 x 300 x 49 shape
#: the 2026-08-20 Linux receipts measured at 1.34-1.41 (before #310
#: narrowed the legacy batch width) now measures 0.82 on the 5070 Ti and
#: 0.43 on the 5090.
#:
#: ONE ROW IS SHORT IN ITS ITEMIZATION, not in its margin, and it is a
#: table row (:data:`FORECAST_ITEMIZATION_GAP_ROWS`) rather than the
#: margin every configuration pays: HRRR physics with RTE-RRTMGP.
FORECAST_PEAK_RATIO_MEASURED = 1.09

#: Configurations whose pool the battery measured USING more than the
#: itemization lists, with the measured worst ratio each pays instead of
#: :data:`FORECAST_PEAK_RATIO_MEASURED` until those arrays are itemized.
#: A configuration matches when it runs RTE-RRTMGP on a domain that also
#: selects ANY of the listed schemes; the battery separates the pair but
#: not the scheme within it (the same schemes under legacy RRTMG used
#: 1.04x, lean RTE-RRTMGP 1.02x), so every scheme of the measured suite
#: is listed.  Measured: HRRR physics with RTE-RRTMGP, 330 x 330 x 49,
#: pool used 1.161x the subtotal and the peak 1.141x over the itemized
#: non-pool and residue on the 5070 Ti (55 MB under the OLD envelope),
#: 1.123x and 0.871x on the 5090.
FORECAST_ITEMIZATION_GAP_ROWS = (
    {"label": "HRRR physics schemes under RTE-RRTMGP",
     "radiation": "rte-rrtmgp",
     "any_of": (("mp_physics", 28), ("bl_pbl_physics", 5),
                ("sf_sfclay_physics", 5), ("sf_surface_physics", 3)),
     "ratio": 1.15},
)

#: The stated safety term over the measured worst ratio: the card-to-card
#: swing of the pool-held ratio on the battery configuration the
#: itemization prices most exactly (lean 528 x 528 x 49, pool used 1.022x
#: the subtotal on both cards; held 1.125 on the 5070 Ti, 1.161 on the
#: 5090: 0.036), rounded up.  That swing is the allocator's own
#: variation between cards, which a card outside the battery can add to
#: the worst row.  The grid-independent residue stays
#: :data:`ENVELOPE_UNMODELLED_BYTES`.
FORECAST_PEAK_RATIO_SAFETY = 0.04

#: The forecast's pool margin over the itemized subtotal: the measured
#: worst ratio plus the stated safety term, and the ONLY proportional
#: margin a forecast envelope carries.  Every door prices with it, through
#: :func:`forecast_pool_headroom`: the estimate
#: (:attr:`ExperimentMemoryEstimate.alloc_estimate_bytes`), the
#: ``[tiles]`` window price, and the mid-run spawn check.
FORECAST_POOL_HEADROOM = round(
    FORECAST_PEAK_RATIO_MEASURED + FORECAST_PEAK_RATIO_SAFETY, 2)


def forecast_pool_headroom(runs) -> float:
    """The measured forecast margin for these domains' run configs.

    :data:`FORECAST_POOL_HEADROOM`, or a matching
    :data:`FORECAST_ITEMIZATION_GAP_ROWS` row's measured ratio plus the
    same safety term when any domain runs a configuration the battery
    measured the itemization short on.  One number per forecast: the
    domains share one pool.
    """
    from woof.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant

    ratio = FORECAST_PEAK_RATIO_MEASURED
    for run in runs:
        # A DomainConfig carries its RunConfig as ``run``.
        run = getattr(run, "run", run)
        if 4 not in radiation_scheme_ids(run):
            continue
        lane = ("legacy-rrtmg" if rrtmg_variant(run) == RRTMG_VARIANT_LEGACY
                else "rte-rrtmgp")
        for row in FORECAST_ITEMIZATION_GAP_ROWS:
            if row["radiation"] != lane:
                continue
            if any(int(getattr(run, key, -1)) == int(value)
                   for key, value in row["any_of"]):
                ratio = max(ratio, float(row["ratio"]))
    return round(ratio + FORECAST_PEAK_RATIO_SAFETY, 2)

#: The A163 battery, one row per measured forecast: ``(label, card,
#: radiation lane, domains, itemized subtotal, itemized non-pool, measured
#: peak, measured pool-held peak, held arrays)``, bytes.  The peak is NVML
#: at 100 ms: device-wide on test host D, whose card was held alone, and the
#: sum of the forecast's own process tree on test host B's shared card; the
#: pool-held peak is the runtime's 20 Hz ``cupy_pool_total`` receipt.  The
#: subtotal, non-pool and held arrays (the urban arrays priced at their
#: allocated size, :func:`forecast_pool_estimate_bytes`) are
#: ``woof check --json`` on the card that ran it.  HRRR-prs 2026-09-29
#: 12Z, 20 min forecasts (70 min for the delayed-nest tree), radiation
#: from step 0, history every 10 min; the second repair's rows are
#: HRRR-prs 2026-09-30 00Z, labelled with their own length.  The PRO 6000
#: rows are the CONUS 3 km HRRR-clone 18 h run report's
#: (2026-09-30, section 3) at its
#: printed two decimals, pool-held not recorded (0).
FORECAST_PEAK_BATTERY = (
    ("lean 528x528x49", "RTX 5070 Ti", "rte-rrtmgp", 1,
     9_967_081_120, 1_679_081_472, 12_046_041_088, 11_209_578_496, 0),
    ("lean 528x528x49", "RTX 5090", "rte-rrtmgp", 1,
     9_967_081_120, 3_790_159_872, 12_989_759_488, 11_570_135_040, 0),
    ("HRRR physics 300x300x49", "RTX 5070 Ti", "legacy-rrtmg", 1,
     5_221_590_392, 1_550_057_472, 6_803_161_088, 6_027_823_616, 0),
    ("HRRR physics 300x300x49", "RTX 5090", "legacy-rrtmg", 1,
     7_528_838_520, 3_476_815_872, 9_648_996_352, 8_311_704_064, 0),
    ("HRRR physics, RTE-RRTMGP 330x330x49", "RTX 5070 Ti", "rte-rrtmgp", 1,
     5_944_617_376, 1_550_057_472, 8_867_807_232, 7_782_245_376, 0),
    ("HRRR physics, RTE-RRTMGP 330x330x49", "RTX 5090", "rte-rrtmgp", 1,
     7_016_786_336, 3_476_815_872, 10_127_147_008, 8_659_322_368, 0),
    ("default suite, legacy RRTMG 300x300x49", "RTX 5070 Ti",
     "legacy-rrtmg", 1,
     3_792_969_400, 1_861_722_112, 5_503_975_424, 4_574_360_576, 0),
    ("default suite, legacy RRTMG 300x300x49", "RTX 5090", "legacy-rrtmg", 1,
     5_028_048_568, 4_233_715_712, 6_920_601_600, 5_705_550_848, 0),
    ("default suite, legacy RRTMG 500x500x49", "RTX 5070 Ti",
     "legacy-rrtmg", 1,
     9_052_755_000, 1_861_722_112, 11_723_079_680, 11_005_778_432, 0),
    # The same suite, 6 h (1,440 steps, a history write every hour), the
    # run length the 2026-08-20 receipts the pool slack rested on were
    # measured over: the peak came 139 s after launch and the 6 h after
    # never exceeded it.  This run integrated to 06Z and then stopped at
    # the forecast's provenance check, because a file of the tree it ran
    # from was rewritten under it; pool-held is therefore not read.
    ("default suite without KF, legacy RRTMG 500x500x49, 6 h",
     "RTX 5070 Ti",
     "legacy-rrtmg", 1,
     8_927_513_384, 1_018_478_592, 10_502_537_216, 0, 0),
    # The 750 m BEP+BEM city tree scaled to 180 x 180 (2.25 km) with a
    # 135 x 135 750 m nest, 59 levels, legacy RRTMG; pool-held not read.
    ("urban BEP+BEM tree 180x180 + 135x135 x 59", "RTX 5090",
     "legacy-rrtmg", 2,
     10_754_875_210, 3_476_815_872, 12_046_041_088, 0, 6_649_469_352),
    # The same tree at 216 x 216 + 162 x 162 x 59, sized for the 16 GB
    # card (A163 second repair).  Refused before A163 (19.39 GB envelope
    # against a 15.83 GB budget) and at the first repair (16.77 GB);
    # admitted (14.89 GB at the 1.16 margin it ran under, 14.79 GB at
    # 1.13) and ran 50 steps to the end: one CUDA process at the peak,
    # 12.34 GB of it the forecast's own.  Pool used 0.92x the
    # subtotal: the BEP+BEM column workspace is priced at every column
    # urban and the parent's city is smaller than that.
    ("urban BEP+BEM tree 216x216 + 162x162 x 59", "RTX 5070 Ti",
     "legacy-rrtmg", 2,
     11_931_666_412, 1_704_886_272, 12_360_613_888, 11_440_669_696,
     8_630_518_752),
    # Default suite, a 1 km nest delayed to 1 h and spawned mid-run, 70 min
    # forecast across the spawn; pool-held not read.
    ("tree 360x360 + delayed nest 360x360 x 49", "RTX 5090", "rte-rrtmgp", 2,
     6_394_161_884, 4_233_715_712, 9_036_627_968, 0, 0),
    ("tree 360x360 + delayed nest 360x360 x 49", "RTX 5070 Ti",
     "rte-rrtmgp", 2,
     6_394_161_884, 1_861_722_112, 8_367_636_480, 0, 0),
    ("default suite 500x500x49", "RTX 5070 Ti", "rte-rrtmgp", 1,
     9_573_167_984, 1_861_722_112, 12_220_104_704, 11_524_418_048, 0),
    # The 5090 row of the shape above, re-measured on the A163 second
    # repair's tree with every CUDA process of the run's process tree
    # named, with no products and with the default products (render while
    # running): at the peak there is one CUDA process, the forecast
    # (`woof go`), 12.26 GB both times; the renderer holds no CUDA
    # context, and the launch's CuPy readiness probe (a `python -c` that
    # imports CuPy and counts devices, 0.5 GB) exits before the forecast
    # allocates.  The row it replaces (1.115) counted a second CUDA process
    # that was not the forecast's: 3.60 GB of device peak over pool-held,
    # against 1.37 GB here, of which the Kain-Fritsch workspace that row's
    # suite carried and this one does not explains about 0.4 GB.
    ("default suite without KF 500x500x49", "RTX 5090", "rte-rrtmgp", 1,
     9_503_787_168, 2_185_838_592, 12_264_144_896, 10_894_410_240, 0),
    # The same configuration with its default products (render while
    # running) on the 16 GB card held alone: one CUDA process at the peak,
    # the forecast; the renderer holds no CUDA context and the launch
    # probe (0.24 GB) exits before the forecast allocates.  The worst row.
    ("default suite without KF 500x500x49, default products",
     "RTX 5070 Ti",
     "rte-rrtmgp", 1,
     9_503_787_168, 1_018_478_592, 11_869_880_320, 11_145_529_856, 0),
    # [tiles] auto on 640 x 640 x 49 (the default suite, 20 min): the
    # planner streams it in 8 tiles of 320 x 160 through 2 buffers of
    # 356 x 196.  The subtotal is the window price's pool (2 buffers + the
    # retained template + the k-tables), the non-pool its CUDA context and
    # local-memory terms: the window price is this envelope.
    ("[tiles] auto 640x640x49, 2 buffers of 356x196", "RTX 5070 Ti",
     "rte-rrtmgp", 1,
     10_312_908_352, 889_454_592, 9_634_316_288, 8_802_425_344, 0),
    ("CONUS lean 1797x1057x50, 18 h", "RTX PRO 6000", "rte-rrtmgp", 1,
     int(64.88 * GIB), int(3.89 * GIB), int(69.66 * GIB), 0, 0),
    ("CONUS HRRR physics 1620x953x50, 1 h", "RTX PRO 6000",
     "legacy-rrtmg", 1,
     int(64.68 * GIB), int(3.56 * GIB), int(72.48 * GIB), 0, 0),
)

#: What the A163 margin rests on, printed beside the number it makes.
ENVELOPE_A163_BASIS = (
    "measured (A163), RTX 5070 Ti 16 GB and RTX 5090 32 GB / Linux, "
    "20 min forecasts: lean, the default suite with RTE-RRTMGP and with "
    "legacy RRTMG, HRRR physics with legacy RRTMG and with RTE-RRTMGP, "
    "two BEP+BEM city trees, a tree whose nest spawns at 1 h, a [tiles] "
    "forecast, and 6 h of the default suite with legacy RRTMG; NVML at "
    "100 ms; plus the RTX PRO 6000 CONUS runs of 2026-09-30")

#: A13: THE POOL RESIDUE IS BOUNDED, AND THE RANKED ROAD PRICES IT SO.
#:
#: THE DEFECT.  Full HRRR (1797 x 1057 x 50, HRRR physics) on four RTX
#: 5090 cards was refused: the 2x2 split priced 31.23-31.66 GiB per card
#: against a 30.86 GiB budget, and the four cards measured 27.6-29.5 GiB
#: at their peaks (a rented box, 2026-10-06, the A13 receipts).  Each rank's
#: price is the A163 envelope of a 25-26 GiB itemized subtotal: x1.13
#: (the measured worst ratio 1.09 plus the 0.04 safety) prices a 3.3 GiB
#: pool margin per rank.  The ratio was measured on 3.5-11 GiB subtotals
#: (:data:`FORECAST_PEAK_BATTERY`), where the pool held 0.63-1.82 GiB
#: beyond the itemization and that residue reads as 9-17% of the
#: subtotal.  On the 24.6-26.2 GiB ranks the pool held 0.42-0.72 GiB
#: beyond the itemization (the run report's per-card ``pool_held_bytes``
#: against this module's itemization of each rank, the rows below): the
#: residue did not grow with the subtotal, so the ratio priced 2.5 GiB
#: per card that no card used, and the price refused a run that fits
#: with 1.3-3.2 GiB to spare.
#:
#: THE MEASUREMENT.  Over every forecast whose pool-held peak was
#: recorded (:data:`FORECAST_PEAK_BATTERY` rows with a held receipt,
#: subtotals 3.5-11 GiB, and the four ranks below at 25-28 GiB) the pool
#: held at most 1.82 GiB beyond the itemization (the default suite with
#: legacy RRTMG at 500 x 500 x 49 on the 5070 Ti), and the card-to-card
#: swing of the held residue on the configuration the itemization prices
#: most exactly (lean 528 x 528 x 49) is 0.34 GiB.  The bound is their
#: sum rounded up to the next quarter gibibyte: 2.25 GiB.  A rank's pool
#: margin is the smaller of the A163 ratio's margin and this bound, so
#: below 17.3 GiB of itemized pool nothing changes, and above it the
#: margin is the bound.
#:
#: ONE CARD, THE SAME BOUND (2026-10-17).  The single-card road kept the
#: ratio because one RTX PRO 6000 row of 2026-09-30 (HRRR physics,
#: 1620 x 953 x 50) peaked 4.24 GiB over the pool it was itemized at then
#: (64.68 GiB) plus its non-pool.  The itemization has since priced the
#: arrays that run held unlisted (the RUC slabs, the Thompson tables and
#: accumulators, moist_cq, ...): today's pricing of that grid is a
#: 78.1 GiB subtotal, so its bounded envelope (82.3 GiB) sits 9.9 GiB above
#: the 72.48 GiB the card measured.  The residue is the pool's, not the
#: rank's: a one-domain forecast of a suite on this table carries the
#: bound on one card too (:func:`single_card_pool_residue_cap_bytes`).
#: THE BREAKAGE: full HRRR (1797 x 1057 x 50) on one 96 GB RTX PRO 6000
#: was priced 108.05 GiB against 94.47 GiB free, 12.1 GiB of it the
#: ratio's margin, and refused resident; it runs resident at 1.70 s/step,
#: and was streamed instead at 4.5 s/step (lane/292-hrrr-admission).
#:
#: Each row: ``(label, card, layout, rank, itemized pool bytes (the
#: rank's subtotal plus its seam buffers, this module's pricing with the
#: prepared statics' lake bound), pool-held peak bytes, device peak
#: bytes, itemized non-pool bytes)``.  Box E, four RTX 5090, HRRR physics
#: on the full HRRR grid, 2026-10-03 21Z, 3 h, hourly history,
#: ``[devices] 1x4``; pool-held from the run report
#: (``peak_memory_per_card``, post-step samples: the CuPy pool never
#: shrinks, so a post-step high-water mark is the run's), device peak
#: from NVML at 200 ms.
#:
#: The two band rows are the bounded price's own run, admitted BY
#: DEFAULT (no refusal override): a 1797 x 557 x 50 exact crop of the
#: HRRR lattice, HRRR physics, 2024-05-21 18Z, 1 h, ``[devices] 1x2`` on
#: two RTX 5090 (a rented box, 2026-10-06), whose ranks (557 x 926 and 557 x
#: 927) are the rank shape of the full grid split 2x2.  Priced
#: 30.70 / 30.75 GiB per card against 30.82 / 30.84 budgets with the
#: MYNN chunk widened to 24,576 / 28,672 columns; the pool held 0.52 /
#: 0.40 GiB beyond the itemization and the cards peaked 28.35 / 28.44
#: GiB, 2.3 GiB under the price.
RANK_POOL_RESIDUE_BATTERY = (
    ("full HRRR 1797x1057x50, HRRR physics, 1x4", "RTX 5090", "1x4", 0,
     27_201_817_616, 27_974_908_928, 29_605_494_784, 1_475_592_192),
    ("full HRRR 1797x1057x50, HRRR physics, 1x4", "RTX 5090", "1x4", 1,
     29_344_842_752, 30_016_247_296, 31_610_372_096, 1_475_592_192),
    ("full HRRR 1797x1057x50, HRRR physics, 1x4", "RTX 5090", "1x4", 2,
     29_668_728_384, 30_123_004_416, 31_717_326_848, 1_475_592_192),
    ("full HRRR 1797x1057x50, HRRR physics, 1x4", "RTX 5090", "1x4", 3,
     27_570_478_776, 28_188_711_936, 29_773_266_944, 1_475_592_192),
    ("HRRR lattice band 1797x557x50, HRRR physics, 1x2 (bounded price, "
     "default admission)", "RTX 5090", "1x2", 0,
     28_324_674_648, 28_882_237_952, 30_435_966_976, 1_475_592_192),
    ("HRRR lattice band 1797x557x50, HRRR physics, 1x2 (bounded price, "
     "default admission)", "RTX 5090", "1x2", 1,
     28_590_739_680, 29_016_986_624, 30_538_727_424, 1_475_592_192),
)


def _pool_residue_receipts() -> tuple[tuple[str, int, int], ...]:
    """``(label, itemized pool bytes, pool-held peak bytes)`` for every
    measured forecast with a pool-held receipt: the A163 battery rows
    that recorded one and the ranked rows."""
    rows = [(row[0], row[4], row[7]) for row in FORECAST_PEAK_BATTERY
            if row[7]]
    rows += [(f"{row[0]} rank {row[3]}", row[4], row[5])
             for row in RANK_POOL_RESIDUE_BATTERY]
    return tuple(rows)


#: The widest pool residue any receipt measured, bytes: pool-held peak
#: less the itemized pool, worst over :func:`_pool_residue_receipts`.
POOL_RESIDUE_MEASURED_BYTES = max(
    held - itemized for _label, itemized, held in _pool_residue_receipts())

#: The card-to-card swing of the pool-held residue on the configuration
#: the itemization prices most exactly (lean 528 x 528 x 49, the two
#: cards of :data:`FORECAST_PEAK_BATTERY`), bytes: the same swing the
#: A163 safety term is, in bytes rather than as a fraction.
POOL_RESIDUE_SWING_BYTES = abs(
    [row[7] for row in FORECAST_PEAK_BATTERY if row[0] == "lean 528x528x49"][0]
    - [row[7] for row in FORECAST_PEAK_BATTERY if row[0] == "lean 528x528x49"][1])

#: The bound a device rank's pool margin never exceeds, bytes: the worst
#: measured residue plus the swing, rounded up to the next quarter
#: gibibyte.  2.25 GiB at the receipts above.
RANK_POOL_RESIDUE_BOUND_BYTES = int(math.ceil(
    (POOL_RESIDUE_MEASURED_BYTES + POOL_RESIDUE_SWING_BYTES) / (GIB // 4))
    * (GIB // 4))

#: What the rank bound rests on, printed beside the number it makes.
RANK_POOL_RESIDUE_BASIS = (
    f"the bound is the widest pool residue any receipt measured "
    f"({POOL_RESIDUE_MEASURED_BYTES / GIB:.2f} GiB, pool held less itemized, "
    f"over {len(_pool_residue_receipts())} forecasts of 3.5-28 GiB on the "
    "RTX 5070 Ti and RTX 5090 including the four ranks of full HRRR on 4 "
    "x RTX 5090 and the two ranks of its 1797 x 557 lattice band under this "
    f"bound) plus the card-to-card swing ({POOL_RESIDUE_SWING_BYTES / GIB:.2f} "
    "GiB), rounded up to the quarter gibibyte (A13)")

#: Why a rank keeps the ratio when an itemization-gap row raised its
#: headroom, printed beside the verdict in the bound's place.
RANK_POOL_RESIDUE_GAP_BASIS = (
    "the byte bound (A13) covers the pool's residue over a COMPLETE "
    "itemization; this suite matches an itemization-gap row "
    "(FORECAST_ITEMIZATION_GAP_ROWS: pool USED above the subtotal, a "
    "shortfall that grows with the rank, 4.1 GiB on a 25 GiB rank of "
    "full HRRR under RTE-RRTMGP where the bound would price 2.25), so "
    "the measured ratio stands until those arrays are itemized")

#: The suites whose pool residue was measured AT RANK SCALE (every row
#: of :data:`RANK_POOL_RESIDUE_BATTERY`), one row per suite: the
#: radiation lane (RRTMG on both streams, legacy or RTE-RRTMGP) and the
#: scheme ids every domain must carry.  Only a suite on this table
#: carries the byte bound; a suite off it keeps the ratio
#: (:func:`rank_pool_residue_cap_bytes`).  A new suite joins the
#: table with its own rank receipts, as rows of the battery and a row
#: here; nothing else changes.
#:
#: Why the table (A13 review): "the residue does not grow with the
#: rank" is shown for one suite only.  The single-card battery
#: (:data:`FORECAST_PEAK_BATTERY`) shows the default suite's pool-held
#: residue growing WITH the subtotal, 0.73 GiB at 300 x 300 x 49 and
#: 1.82 GiB at 500 x 500 x 49 (3.53 and 8.43 GiB itemized, 20.6% and
#: 21.6%), and by the device-peak measure 0.52-1.26 GiB on 8.3-8.9 GiB
#: (6-14%).  At that rate a 27 GiB rank of the default suite holds
#: 1.6-3.8 GiB beyond its itemization, up to 1.5 GiB over the 2.25 GiB
#: bound: the gate would admit a card that then allocates past its
#: price, the mid-run out-of-memory the gate exists to refuse.  HRRR
#: physics under legacy RRTMG measured 0.40-0.72 GiB on six 25-28 GiB
#: ranks (1.5-2.8%), which is what the bound covers.
RANK_POOL_RESIDUE_SUITES = (
    {"label": "HRRR physics under legacy RRTMG, no cumulus",
     "radiation": "legacy-rrtmg",
     "all_of": (("mp_physics", 28), ("bl_pbl_physics", 5),
                ("sf_sfclay_physics", 5), ("sf_surface_physics", 3),
                ("cu_physics", 0))},
)

#: Why a rank keeps the ratio when its suite has no rank-scale receipt,
#: printed beside the verdict in the bound's place.
RANK_POOL_RESIDUE_UNMEASURED_BASIS = (
    "the byte bound (A13) rests on rank-scale receipts of the suites in "
    "RANK_POOL_RESIDUE_SUITES only; this suite has none, and the "
    "single-card battery shows the default suite's pool residue growing "
    "with the subtotal (0.73 GiB at 3.5 GiB, 1.82 GiB at 8.4 GiB, about "
    "21%), which on a 27 GiB rank would exceed the 2.25 GiB bound by up "
    "to 1.5 GiB, so the measured ratio stands until a ranked run of this "
    "suite above 17 GiB per rank records its pool-held peak")


def rank_pool_residue_suite(runs):
    """The :data:`RANK_POOL_RESIDUE_SUITES` row every one of these
    domains' run configs matches, or ``None``: the domains share one
    pool, so one domain off the table takes the whole forecast off it."""
    from woof.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant

    matched = None
    for run in runs:
        run = getattr(run, "run", run)
        # Every rank receipt ran RRTMG on BOTH streams; a run that pairs
        # RRTMG with another scheme on one stream is a suite without a
        # receipt (A13 review 3), whatever its other scheme ids say.
        if tuple(radiation_scheme_ids(run)) != (4, 4):
            return None
        lane = ("legacy-rrtmg" if rrtmg_variant(run) == RRTMG_VARIANT_LEGACY
                else "rte-rrtmgp")
        row_for_run = None
        for row in RANK_POOL_RESIDUE_SUITES:
            if row["radiation"] != lane:
                continue
            if all(int(getattr(run, key, -1)) == int(value)
                   for key, value in row["all_of"]):
                row_for_run = row
                break
        if row_for_run is None or (matched is not None
                                   and row_for_run is not matched):
            return None
        matched = row_for_run
    return matched


def rank_pool_residue_cap_bytes(runs) -> int | None:
    """The byte bound a device rank's pool margin carries for these
    domains' run configs (A13), or ``None`` when the ratio must stand.

    The bound (:data:`RANK_POOL_RESIDUE_BOUND_BYTES`) was measured as
    the pool's residue over a complete itemization: what the CuPy pool
    holds beyond arrays the itemization already counts, which the rank
    receipts show does not grow with the rank FOR THE SUITES THEY
    MEASURED (:data:`RANK_POOL_RESIDUE_SUITES`); a suite with no
    rank-scale receipt keeps the ratio, since the single-card battery
    shows the default suite's residue growing with the subtotal and a
    27 GiB rank of it would exceed the bound.  An itemization-gap row
    (:data:`FORECAST_ITEMIZATION_GAP_ROWS`) names the opposite: a suite
    whose pool is USED beyond the subtotal (1.16x, HRRR physics under
    RTE-RRTMGP), arrays the forecast allocates and the itemization does
    not list, a shortfall proportional to the rank by construction.
    Capping that margin at the bound would price 2.25 GiB on a 25 GiB
    rank whose unlisted arrays alone take about 4.1 GiB, so the gate
    would admit a card that then allocates past its price, the
    mid-run out-of-memory the gate exists to refuse.  Such a suite
    keeps the ratio its gap row measured."""
    runs = tuple(runs)
    if forecast_pool_headroom(runs) > FORECAST_POOL_HEADROOM:
        return None
    if rank_pool_residue_suite(runs) is None:
        return None
    return int(RANK_POOL_RESIDUE_BOUND_BYTES)


def rank_pool_residue_basis(runs) -> str:
    """What the ranked road's pool margin rests on for these domains'
    run configs, printed beside the gate's verdict: the bound's receipts,
    the itemization-gap row that keeps the ratio, or the missing
    rank-scale receipt that keeps it."""
    runs = tuple(runs)
    if forecast_pool_headroom(runs) > FORECAST_POOL_HEADROOM:
        return RANK_POOL_RESIDUE_GAP_BASIS
    if rank_pool_residue_suite(runs) is None:
        return RANK_POOL_RESIDUE_UNMEASURED_BASIS
    return RANK_POOL_RESIDUE_BASIS


def single_card_pool_residue_cap_bytes(exp) -> int | None:
    """The A13 byte bound a ONE-CARD resident forecast's pool margin
    carries, or ``None`` when the ratio stands.

    The same bound and the same suite table as a device rank
    (:func:`rank_pool_residue_cap_bytes`): what the CuPy pool holds beyond
    a complete itemization is a property of the suite's allocations, not
    of the forecast being one rank of several, and the one single-card
    row that kept the ratio is covered at today's itemization (see the A13
    note above :data:`RANK_POOL_RESIDUE_BATTERY`).  One domain only: the
    receipts are one HRRR grid, so a tree keeps the ratio."""
    domains = tuple(getattr(exp, "domains", ()) or ())
    if len(domains) != 1:
        return None
    return rank_pool_residue_cap_bytes([domains[0].run])


def single_card_residue_terms(exp) -> dict:
    """``estimate_experiment`` keywords for the one-card residue bound:
    empty where the ratio stands, so a ratio-priced call is the call it
    always was."""
    cap = single_card_pool_residue_cap_bytes(exp)
    return {} if cap is None else {"pool_residue_cap_bytes": cap}


def mynn_width_adjustable(exp) -> bool:
    """Can the resident road narrow this forecast's MYNN width to fit?

    True when a domain runs MYNN and neither a pin nor an operator
    override names the width (those are walked as asked)."""
    from woof.core.mynn_pbl_scratch import mynn_rank_chunk_candidates
    runs = [dc.run for dc in (getattr(exp, "domains", ()) or ())
            if int(getattr(dc.run, "bl_pbl_physics", 0) or 0) == 5]
    return bool(runs) and len(mynn_rank_chunk_candidates(int(runs[0].nz))) > 1


@contextmanager
def required_mynn_pricing(exp, width: int | None = None, *,
                          only_given: bool = False):
    """Price the resident MYNN workspace at ``width`` columns, or at the
    narrowest width a run may walk when ``width`` is None (A3c).

    The MYNN column width is a speed choice that never changes a byte of
    output (:func:`woof.core.mynn_pbl_scratch.choose_mynn_column_chunk`).
    The ranked road prices each rank at the narrowest width and widens it
    into what the budget leaves (open-defects ledger A3c).  On one card the
    run picks its width from the card's free memory before anything is
    allocated (98,304 columns on a 96 GB card, 5.3 GiB of workspace at 50
    levels against 0.45 GiB at the narrowest), so a forecast whose state
    and physics fit only at a narrower width was refused.  The resident
    door prices the narrow width when the walked one does not fit, then
    fits and pins the widest width that does
    (:func:`woof.core.streaming.admit_resident_road`).  ``only_given``
    leaves the width alone when ``width`` is None.  A pin or an operator
    override is the width priced either way."""
    from woof.core.mynn_pbl_scratch import (
        mynn_pricing_resident_chunk, mynn_rank_chunk_candidates)
    runs = [dc.run for dc in (getattr(exp, "domains", ()) or ())
            if int(getattr(dc.run, "bl_pbl_physics", 0) or 0) == 5]
    if not runs or (width is None and only_given):
        yield None
        return
    if width is None:
        width = int(mynn_rank_chunk_candidates(int(runs[0].nz))[0])
    with mynn_pricing_resident_chunk(int(width)):
        yield int(width)


def machine_peak_envelope_bytes(
        *, alloc_estimate_bytes: int, non_pool_bytes: int,
        domains: int = 1,
        footprint_projection_bytes: int | None = None,
        family: str = "linux",
        legacy_radiation: bool = True,
        held_exact_bytes: int = 0) -> int:
    """The machine-wide peak a run of this configuration should reach.

    AFFINE, not multiplicative.  The old form was ``factor x projection``
    with no intercept, and a model with no intercept cannot describe a
    cost with a large fixed term: it under-predicts small configurations
    and over-predicts large ones, which is precisely what the 16 GiB
    fleet node measured (a 3.99 GiB declaration that peaked at 4.38, and
    a 19.95 GiB declaration that peaked at 13.88).

    The three terms are each something this module already knows:

    * the itemized allocation estimate -- the pool side, which the
      measurements track to within a few percent;
    * :func:`non_pool_device_bytes` -- the CUDA context plus the
      launch-time local-memory backing store, which scale with the DEVICE
      and the kernel set, not with the grid;
    * :data:`ENVELOPE_UNMODELLED_BYTES` (+ a per-nest term) -- the
      measured residue, stated as a constant because that is what it
      measured as.

    The legacy-RRTMG pool slack (:data:`POOL_SLACK_FRACTION`) is no
    longer a term (A163): it priced the same pool headroom the estimate's
    margin already prices, and the measured margin
    (:data:`FORECAST_POOL_HEADROOM`) covers the legacy lane's measured
    peaks on its own.  ``legacy_radiation`` and
    ``footprint_projection_bytes`` are accepted for signature
    compatibility and no longer read.

    ``held_exact_bytes`` is the part of the estimate held at exactly its
    priced size for the whole run (the urban arrays,
    :func:`forecast_pool_estimate_bytes`).  The per-nest term is a
    fraction of the REST: what it measured is the nest coupler's buffers,
    the extra per-domain drivers and the per-domain output staging, which
    scale with the prognostic and output fields, and no coupler or
    history writer touches a held urban array.
    """

    nests = max(0, int(domains) - 1)
    pooled = max(0, int(alloc_estimate_bytes) - int(held_exact_bytes))
    return (int(alloc_estimate_bytes) + int(non_pool_bytes)
            + ENVELOPE_UNMODELLED_BYTES
            + math.ceil(ENVELOPE_PER_NEST_FRACTION * nests * pooled))


def forecast_pool_estimate_bytes(subtotal_bytes: int, *,
                                 held_exact_bytes: int = 0,
                                 headroom: float,
                                 residue_cap_bytes: int | None = None) -> int:
    """The forecast's pool estimate: the measured margin on what the pool
    turns over, and the held arrays at their allocated size.

    ``residue_cap_bytes`` bounds the margin in bytes (A13, the ranked
    road: :data:`RANK_POOL_RESIDUE_BOUND_BYTES`).  The margin is a ratio
    measured on 3.5-11 GiB subtotals, where the pool's residue over the
    itemization reads as 9-17%; on 25-28 GiB device ranks every pool
    receipt measured that residue at 0.40-0.72 GiB, so the ratio alone
    priced 2.5 GiB per card that no card used.  Below the crossover
    (``residue_cap / (headroom - 1)``, 17.3 GiB at the measured margin
    and bound) the cap is inactive and the price is unchanged.  The cap
    bounds a residue over a COMPLETE itemization: a caller prices a
    suite that matches an itemization-gap row with the ratio alone
    (:func:`rank_pool_residue_cap_bytes`; :func:`estimate_experiment`
    drops the cap for such a suite itself).

    A163.  The margin (:data:`FORECAST_POOL_HEADROOM`) is what the CuPy
    pool holds beyond the itemization: blocks a step frees and the pool
    keeps, transients of one size cached for another.  An array allocated
    once at its exact shape and held for the run takes exactly its size
    from the pool, so multiplying it by the margin prices headroom it
    never uses.  The urban arrays are that family, and they are most of a
    BEP+BEM forecast: 12.73 of 18.77 GiB itemized on the 750 m city tree
    of 288 x 288 + 216 x 216 x 59, where the margin on the whole subtotal
    priced 1.7 GiB of headroom the pool never holds for them.  Measured:
    :data:`FORECAST_PEAK_BATTERY`'s urban rows, whose peak less the held
    arrays stays under the margin on the rest
    (``tests/test_memory_gate_a163.py``); on the 216 x 216 + 162 x 162
    tree the 16 GB card measured a 12.36 GB device peak against a
    14.79 GB envelope.
    """
    held = max(0, min(int(held_exact_bytes), int(subtotal_bytes)))
    turnover = int(subtotal_bytes) - held
    margin = math.ceil(float(headroom) * turnover) - turnover
    if residue_cap_bytes is not None:
        margin = min(margin, max(0, int(residue_cap_bytes)))
    return turnover + margin + held


def urban_held_item_names(run) -> frozenset:
    """The ``fields/...`` physics item names priced at their allocated size
    (:func:`woof.core.urban_state.urban_held_array_shapes`); empty when no
    urban model runs."""
    if int(getattr(run, "sf_urban_physics", 0) or 0) <= 0:
        return frozenset()
    from woof.core.urban_state import urban_held_array_shapes
    return frozenset(f"fields/{name}" for name in urban_held_array_shapes(run))


def bem_workspace_bound_note(estimate) -> str | None:
    """What a configuration-door price says beside a BEP+BEM workspace.

    The configuration door (``woof check CONFIG``, ``woof go``'s gate
    before the fetch) runs before the land cover exists, so it prices
    BEP+BEM's column workspace at every column urban, the upper bound; the
    prepared doors read the land cover and price the plan the run builds
    (:func:`woof.core.urban_state.prepared_urban_columns`, A176).  Said
    beside the figure, so a refusal here is read as the bound it is.
    ``None`` when the estimate holds no such workspace.
    """
    total = sum(item.nbytes for domain in estimate.domains
                for item in domain.items
                if item.name == "fields/bem_column_workspace")
    if not total:
        return None
    return (f"the BEP+BEM column workspace is priced here at every column "
            f"urban ({total / GIB:.2f} GiB of it), because the land cover "
            "does not exist before preparation; a prepared run (woof go "
            "--prepared-root, woof sim) prices it from the prepared land "
            "cover's urban columns, which is tighter")


def urban_held_bytes(run, *, urban_columns: int | None = None) -> int:
    """Device bytes of :func:`urban_held_item_names` for one domain, at
    the domain's urban column count where it is known (A176)."""
    if int(getattr(run, "sf_urban_physics", 0) or 0) <= 0:
        return 0
    from woof.core.urban_state import urban_held_array_shapes
    total = 0
    for shape in urban_held_array_shapes(
            run, urban_columns=urban_columns).values():
        n = 4
        for extent in shape:
            n *= int(extent)
        total += n
    return total


def observed_peak_envelope_bytes(
        footprint_projection_bytes: int,
        *, platform: str | None = None,
        vram_gib: float | None = None,
) -> int:
    """The RETIRED multiplicative machine-peak envelope.

    Kept because the WDDM factor is a real measurement over a real
    projection and the Windows lane still quotes it, and because the
    receipts of three releases are written in its terms.  New callers
    want :func:`machine_peak_envelope_bytes`, which has an intercept.
    """

    return int(footprint_projection_bytes
               * peak_envelope_factor(platform, vram_gib))


# ---------------------------------------------------------------------------
# TIER 3, RE-MEASURED: what the CuPy pool never sees
# ---------------------------------------------------------------------------
#
# ``PROBE_DEVICE_OVERHEAD_BYTES`` above (1.39 GiB) was taken from a process
# that ran ZERO steps, and its own docstring flags it as a lower bound
# "because zero steps JIT-compile almost none of the kernel modules".  It is
# worse than a lower bound: it models the wrong thing.  Measured 2026-07-26
# on the user's RTX 5090 (170 SMs, 1536 resident threads per SM, driver
# 610.74) by running real integrations with ``cudaMemGetInfo`` and the CuPy
# pool sampled together at 1 s, and bracketing the FIRST launch of every
# kernel symbol:
#
#   ============================  ==========  ==========  ==========
#   term                          3 domains   4 domains   scales with
#   ============================  ==========  ==========  ==========
#   CuPy pool, reserved peak      13,697 MiB  21,072 MiB  columns
#   CUDA context                     432 MiB     430 MiB  nothing
#   local-memory backing store     5,738 MiB      64 MiB  ONE kernel
#   other non-pool                  ~400 MiB    ~430 MiB  nothing
#   ============================  ==========  ==========  ==========
#
# NVRTC module images are not on that list because they do not measure: all
# 51 kernel modules compiled and loaded cost 2 MiB of device memory between
# them, and resolving every symbol another 18 MiB.
#
# The dominant term is the LOCAL-MEMORY BACKING STORE.  When a kernel's
# per-thread local frame exceeds the context's default stack limit, the
# driver answers the kernel's first launch by allocating a backing store for
# the device's whole resident-thread capacity -- not for the launched grid --
# and keeps it for the life of the process.  One allocation serves the
# process, sized by the LARGEST per-thread frame launched so far, so the term
# is a MAXIMUM over launched kernels and is completely independent of domain
# count.  That is why the old estimator was short by the same ~5.5 GiB at
# three domains and at four.
#
# The law is exact on this device, verified against a synthetic kernel at
# five local-frame sizes and against two real forecasts:
#
#     reservation = (max_local_size_bytes - default_stack_limit_bytes)
#                   * max_threads_per_multiprocessor * multiprocessor_count
#
# ``kf_column`` measures 24,064 B of local frame, so
# (24064 - 1024) * 1536 * 170 = 6,016,204,800 B = 5,737.5 MiB; the measured
# step across its first launch was 5,738.0 MiB, twice, in two separate runs.
# The baseline term ``default_stack_limit_bytes * capacity`` = 255 MiB is
# already inside :data:`CUDA_CONTEXT_BYTES`, which is why it is subtracted.

#: How far the CuPy pool's RESERVED high-water runs past the enforced
#: estimate.  ``alloc_estimate_bytes`` bounds pool USED, which is what
#: ``alloc_measured_le_estimate`` checks; a rail is spent on what the pool
#: HOLDS.  Measured on the two traced forecasts of 2026-07-26:
#:
#:   ==========  ==============  ================  ==========
#:   domains     estimate (MiB)  pool reserved     over
#:   ==========  ==============  ================  ==========
#:   3           13,373          13,697            2.42%
#:   4           20,480          21,072            2.89%
#:   ==========  ==============  ================  ==========
#:
#: 3% is the rounded-up bound over both, carried as the N0 reserve's
#: retention term whenever the caller supplies the estimate it applies to.
#: It is a fitted constant from two points and is documented as such; what
#: keeps it accurate is that it moves the gate in the refusing direction.
POOL_RESERVED_OVER_ESTIMATE_FRACTION = 0.03

#: Device memory a fresh CUDA context holds before woof allocates anything:
#: measured 432 MiB (three fresh processes: 433, 432, 430), of which
#: 1024 B x 1536 x 170 = 255 MiB is the default-stack local-memory store.
CUDA_CONTEXT_BYTES = 432 * 1024 ** 2


def non_pool_basis(profile: "DeviceLocalMemoryProfile",
                   exp: "ExperimentConfig | None" = None) -> str:
    """One sentence naming the card row a non-pool charge came from.

    Printed beside the number.  A grid-independent term large enough to
    refuse a card on its own has to be traceable to the reading that
    made it, or the reader has no way to tell a measurement from an
    assumption -- which is the whole history of this module.

    With ``exp`` given and selecting Noah-MP, the sentence also says
    where the Noah-MP frames came from: this card's own recorded platform
    ("measured on this card's compile platform sm_86/12.9.86 ..."), or
    the ceiling over the recorded platforms with the reason this card had
    no row of its own -- so a user reading a "fits" verdict can see
    whether the Noah-MP term was measured on the card in front of them
    (:func:`woof.core.noahmp_frame_provenance.noahmp_frame_basis`).
    """
    sentence = _non_pool_profile_basis(profile)
    if exp is not None and selects_noahmp(exp):
        from woof.core.noahmp_frame_provenance import noahmp_frame_basis

        basis = noahmp_frame_basis(physics_kernel_modules(exp), profile)
        if basis is not None:
            sentence = f"{sentence}; {basis.sentence()}"
    if exp is not None:
        # A module with no reading anywhere is priced, not refused, and
        # the number it is priced at is an assumption -- so the sentence
        # beside it says so and names the modules it covers.
        assumed = sorted(assumed_bound_modules(physics_kernel_modules(exp)))
        if assumed:
            bound = _kernel_frame_recordings.assumed_frame_bound()
            sentence = (
                f"{sentence}; {', '.join(assumed)} priced at the assumed "
                f"bound {bound} B per thread "
                f"({_kernel_frame_recordings.ASSUMED_BOUND_PHRASE}; the "
                "widest frame any module in this tree has been recorded at)")
    return sentence


def _non_pool_profile_basis(profile: "DeviceLocalMemoryProfile") -> str:
    """The profile half of :func:`non_pool_basis`: the card, its context
    and the compile platform its standalone frames were read at."""

    # The frames half of the backing store is a reading of the card's
    # compile platform (kernel_frame_recordings.py keys every row on the
    # target architecture and the NVRTC build), so a profile that carries
    # one names it: the reader can then tell a frame priced from THIS
    # platform's row from one priced off the cross-platform ceiling.
    platform = ""
    if profile.compile_platform is not None:
        capability, build = profile.compile_platform
        platform = (f", frames read at its compile platform sm_{capability} / "
                    f"NVRTC {build}")
    if profile.context_is_measured:
        return (
            f"measured on this card ({profile.name}, "
            f"{profile.multiprocessor_count} SMs x "
            f"{profile.max_threads_per_multiprocessor} threads): CUDA "
            f"context {profile.cuda_context_bytes / GIB:.2f} GiB "
            f"(bare {profile.bare_context_bytes / GIB:.2f} + "
            f"{CONTEXT_RUNTIME_GROWTH_BYTES / GIB:.2f} module-load growth) "
            f"plus the local-memory backing store of its kernel set{platform}")
    if profile is not MEASURED_LOCAL_MEMORY_PROFILE:
        # A card that was READ -- its name and shader census are its own
        # -- priced from that census.  The reference profile below is
        # the one object every absent, undeclared card is priced on, so
        # identity with it is what tells "nobody's card" from "this
        # one"; a profile's fields cannot (the reference carries a real
        # card's name and census too).
        return (
            f"read on this card ({profile.name}, "
            f"{profile.multiprocessor_count} SMs x "
            f"{profile.max_threads_per_multiprocessor} threads): CUDA context "
            f"{profile.cuda_context_bytes / GIB:.2f} GiB modelled from its "
            f"shader census at the measured "
            f"{MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD} B per "
            f"resident thread, plus the local-memory backing store of its "
            f"kernel set{platform}")
    return (
        f"modelled for an absent card ({profile.name}, "
        f"{profile.multiprocessor_count} SMs x "
        f"{profile.max_threads_per_multiprocessor} threads): CUDA context "
        f"{profile.cuda_context_bytes / GIB:.2f} GiB from the measured "
        f"{MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD} B per resident "
        f"thread, plus the local-memory backing store of its kernel set")


#: RETIRED 2026-08-03 -- the per-class SM discount for cards not in this
#: machine.  Each row claimed the LARGEST SM count sold at that capacity
#: ("(frame - default stack) x SMs x threads per SM" is a property of the
#: DEVICE), so a 12 GiB tier was priced at 70 SMs instead of the 170-SM
#: reference.  Two things killed it, both from the first user-zero
#: cross-architecture stress run (RTX 4090, sm_89):
#:
#: * IT WAS NOT A BOUND.  The rows were a market survey, not a
#:   measurement: a 12 GiB RTX 3080 Ti ships 80 SMs against the row's
#:   70, so the "upper bound" under-priced real cards in the class.
#: * IT MADE THE ABSENT-CARD PATH MORE OPTIMISTIC THAN THE PRESENT-CARD
#:   PATH.  Sizing for a card not in the machine used a 1.45 GiB
#:   non-pool intercept where the same code on the real 4090 measured
#:   2.30 GiB; a config certified "fits with 0.27 GiB to spare" landed
#:   0.015 GiB from the budget -- a margin 18x smaller than advertised,
#:   on exactly the sizing-for-a-card-you-intend-to-buy path, where the
#:   number cannot be checked until the money is spent.
#:
#: Kept because three sizing receipts are written in its terms.  No
#: caller consults it: :func:`card_local_memory_profile` prices every
#: absent card against the measured reference profile.
CARD_CLASS_MULTIPROCESSORS = ((12.0, 70), (16.0, 84), (24.0, 128),
                              (32.0, 170))


def card_local_memory_profile(
        vram_gib: float | None) -> DeviceLocalMemoryProfile:
    """The device profile for a card that is NOT in this machine.

    Always the measured reference profile -- the max of known-device
    intercepts -- whatever capacity is declared.  An absent card's SM
    count is unknown, and the retired per-class discount above priced a
    certified margin 18x too generous on the one machine that could
    check it; hardware that cannot be measured gets the conservative
    intercept, never a discount.  A live device always overrides this
    (see :func:`live_device_local_memory_profile`): the absent-card path
    can over-price relative to the card eventually bought, but it can
    never be MORE optimistic than a present card's own census.
    """

    return MEASURED_LOCAL_MEMORY_PROFILE


def local_memory_profile_from_device(cp, *, device_id: int = 0) -> DeviceLocalMemoryProfile:
    """Read the profile off the attached device: its name, shader census,
    default stack limit and compile platform.

    Nothing here is a sample.  Every field is a constant of the card and
    its toolchain, so two reads of one card are one profile, and the
    context term priced from it is one number
    (:data:`MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD` says why the
    bare context is priced from the census rather than read: the NVML
    delta this used to take either side of the first CUDA call moved by
    whole MiB between reads of one idle card).
    """
    # ``woof multi-run`` masks one physical UUID into each check process;
    # CUDA ordinal 0 is therefore the selected logical device, not a claim
    # that every run belongs on the machine's physical index zero.
    # Ensemble packing samples each physical card in its active Device
    # context. The existing masked single-card callers keep ordinal zero.
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    name = props["name"]
    stack_limit = int(cp.cuda.runtime.deviceGetLimit(0))
    return DeviceLocalMemoryProfile(
        name=name.decode() if isinstance(name, bytes) else str(name),
        multiprocessor_count=int(props["multiProcessorCount"]),
        max_threads_per_multiprocessor=int(
            props["maxThreadsPerMultiProcessor"]),
        default_stack_limit_bytes=stack_limit,
        compile_platform=read_compile_platform(),
    )


def read_compile_platform() -> tuple[str, str] | None:
    """``(device_compute_capability, nvrtc_build)`` of THIS process's card
    and compiler, or ``None`` when either could not be resolved.

    The two :func:`woof.certify.compile_platform.compile_platform_fingerprint`
    keys that decide code generation, in the shape
    :mod:`woof.core.kernel_frame_recordings` keys its rows on.  An
    unresolved half is ``None`` for the pair: "unavailable" must never
    match a recording.  Device contact -- only called from readers that
    are already touching the card.
    """
    try:
        from woof.certify.compile_platform import (
            UNRESOLVED, compile_platform_fingerprint)

        fingerprint = compile_platform_fingerprint()
    except Exception:  # noqa: BLE001
        return None
    pair = (fingerprint.get("device_compute_capability"),
            fingerprint.get("nvrtc_build"))
    if any(not isinstance(v, str) or not v or v == UNRESOLVED for v in pair):
        return None
    return (str(pair[0]), str(pair[1]))


#: Per-module MAXIMUM static local frame per thread, in bytes, as the CUDA
#: driver reports it in ``CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`` after NVRTC
#: compiles ``woof/core/kernels/<module>.cu`` with the shipped options and
#: NO integer defines injected -- i.e. at each module's unspecialized bound.
#:
#: THIS IS A CEILING OVER NAMED COMPILE PLATFORMS, not one box's reading.
#: A frame is what NVRTC emitted for one target architecture at one
#: compiler build, and measured three ways it moves with both: ``gf``,
#: ``noah``, ``thompson_aerosol_warm`` and ``ysu`` move with the
#: architecture at a fixed compiler, ``nssl2_fused_gs``, ``rrtmgp_cloud``
#: and ``shinhong`` move with the compiler build at a fixed architecture,
#: and ``noahmp_leaves`` moves with both.  The readings themselves, each
#: naming its box, its ``sm_`` target and its NVRTC build, live in
#: :mod:`woof.core.kernel_frame_recordings`; every row below is the
#: element-wise MAXIMUM over them, checked against them at import.
#:
#: Under-pricing is what a rail gate cannot survive -- one byte of frame
#: is one byte times the whole resident-thread capacity of device memory
#: nobody charged for -- so a platform nobody has measured is priced at
#: the ceiling, never at an average or at the nearest box.  Rows recorded
#: 2026-07-26 through 2026-08-20; regenerated and compared against the
#: compiler in front of it by ``tests/test_preflight.py::
#: test_the_recorded_local_frames_match_the_driver``, which asserts EXACT
#: equality only on a platform this tree has a recording for, and the
#: never-below-a-measurement invariant on every other.
#:
#: Two modules do not launch at their unspecialized bound: ``refl`` and
#: ``wdm6_refl`` compile their column arrays to the
#: configuration's own level count
#: (:data:`LEVEL_SPECIALIZED_KERNEL_FRAMES`), so their rows here are the
#: CEILING, not the price.  Three more are TIERED, and those are the only
#: ones that can price ABOVE their row.  ``acoustic`` compiles the implicit
#: w''-phi'' solve at a coarse ``WPHI_MAX_LEV`` tier chosen by ``nz``
#: (:data:`ACOUSTIC_TIER_FRAME`), so this row is its price at the shipped
#: 129 tier -- every ``nz <= 128`` configuration -- and the deeper tiers add
#: to it.  ``wdm6`` and ``wsm6`` are the same shape, each with a two-rung
#: ladder (:data:`WDM6_TIER_FRAME`, :data:`WSM6_TIER_FRAME`): each row is
#: that module's ``KMAX = 64`` default, every ``nz <= 64`` configuration,
#: and the 80 tier prices above it.
#: Everything else launches as compiled.
#:
#: A module maximum is an UPPER bound over the kernels a scheme can launch --
#: e.g. Morrison's launched sedimentation kernel measures 1,280 B against the
#: module's 5,120 B.  Over-pricing is the safe direction for a rail gate and
#: under-pricing is what put a run 1,630 MiB over; the bound is stated, not
#: silently tightened.
KERNEL_MAX_LOCAL_SIZE_BYTES: dict[str, int] = {
    # Production loaders, RTX 3080 sm_86 / NVRTC 13.0.48, 2026-10-21.
    "output_diagnostics": 0, "gsd_diagnostics": 0, "gsd_helicity": 0,
    "climate_diagnostics": 0,
    "gsd_initial_reflectivity": 0,
    # WRF's ocean mixed layer, read at 0 B through the production loader
    # by the OML port lane (port289-1-ocean1-oml, tools/vram_reserve_probe.py).
    "oml": 0,
    # Production loader, V58 sm_120 / NVRTC 12.8.93, 2026-10-20.
    "hail": 7392,
    # 27,408 B on test host B (RTX 5090, sm_120, NVRTC 13.4.92, 2026-10-17);
    # the G2 reading under NVRTC 12.9.86 was 26,736 B (kernel_frame_recordings).
    # Ceiling: 28,576 B, test host A (RTX 4090, sm_89, NVRTC 13.4.92, fast
    # numerics, 2026-10-22).
    "fast_sbm": 28576,
    "thompson_column": 50016,
    # 512 B: widest driver-read G3 ensemble variant on sm_120 / NVRTC 13.2.86;
    # 2,272 B on sm_89 / NVRTC 13.4.92 (2026-10-22), the ceiling.
    "g3": 2272,
    # New port units: measured production-composed driver frames, V12 12.8.93.
    'fer_hires': 0,

    "wsm6r": 0,  # sm_120, NVRTC 12.9.86, measured scalar/external-work kernels.
    # Production WSM-family tier 64, sm_120 / NVRTC 12.8.93.
    # The family metadata below prices a deeper launched tier separately.
    "wsm7": 7216,
    # 2.9.0 round 2: the radiation, lightning and New SAS units joined with no
    # row (golden/ports-rad, golden/ports-cu).  Read 2026-10-13 through the
    # production loader (tools/vram_reserve_probe.py frames) on test host D's RTX
    # 5070 Ti, sm_120, NVRTC 13.4, at 2f7194f53: nsas through its no-FTZ
    # route, lightning through its direct NVRTC site.
    "aer_opt2": 0,
    "lightning": 0,
    "nsas": 64,
    "rrtmg_aer1": 0,
    # Three units the 2.8.6 staging line shipped without a row, read
    # 2026-10-05 with tools/vram_reserve_probe.py frames on RTX 5090 /
    # NVRTC 13.4.92 through the production loader: 0 B each.  The fork's
    # order-five vertical scalar flux (f82847049), the fork's saved-wind
    # limiter (4b2bd1665) and the prescribed-smoke time blend (ef8325e32).
    "pd_vertical_sl": 0,
    "rrtmg_smoke_manifest": 0,
    "upper_wind_limiter": 0,
    # The coupled fire's standalone units (lane/ec-sfire), 0 B each on
    # sm_120 / NVRTC 13.4.92 (test host D RTX 5070 Ti, 2026-10-06).
    "chem_sfire": 0,
    "sfire_coupling": 0,
    "sfire_ideal": 0,
    "sfire_ideal_atmos": 0,
    # Initialization-only parameter-table scaler, measured 2026-10-04
    # through the production loader on RTX 5090 / NVRTC 13.4.92: 0 B.
    "physics_params": 0,
    # The ensemble member bookkeeping, the stochastic pattern generator and
    # the RUC hydraulic SPP operator, read 2026-10-03 on RTX 5090 /
    # NVRTC 13.4.92: 0 B each (no kernel of the three holds a local array).
    "ensemble_bookkeeping": 0,
    "ensemble_stochastic": 0,
    "ruc_spp": 0,
    # Classic Thompson (mp=8), the transcription of WRF v4.6.1
    # mp_thompson: 0 B at every entry point, sm_120 / NVRTC 13.4,
    # 2026-10-05 (kernel_frame_recordings.SM120_NVRTC_13_4_92).
    # 64 B on sm_89 / NVRTC 13.4.92 (2026-10-22), the ceiling.
    "thompson_classic": 64,
    # Production lake loader.  Measured 2026-10-03 at 14,224 B (RTX 5090,
    # NVRTC 13.4.92) and 14,400 B (RTX PRO 6000, NVRTC 12.8.93) while the
    # column arrays lived in the local frame; that frame was the widest of
    # every HRRR-physics run and its backing store cost 3.3 GiB per 5090.
    # The arrays moved to a priced device arena (``lake/arena`` in
    # physics_array_shapes, kernels/lake_support.cuh), and the same probe
    # read 2026-10-05 on RTX 5090 / cupy-cuda13x through the production
    # loader: step 1,240 B, init 0 B; the card's non-pool residency after a
    # 40,800-column step fell from 3,288 MiB to 58 MiB.  The same source
    # reads 4,496 B on an RTX 4090 (sm_89, NVRTC 13.4.92, 2026-10-22), the
    # ceiling: priced at 1,240 B a 4090 lake run was under-charged 0.6 GiB.
    "lake": 4496,
    # Test-only math grading unit, read on sm_89 and sm_120, NVRTC 13.4.59.
    "portable_libm64_grade": 48,
    # Added with this box's recording at 0 B. 'ntiedtke' is what
    # cu_physics = 16 needs to be priceable at all.
    # The GOCART kernels (lane/aq-gocart), read 2026-09-30 on sm_120 at
    # NVRTC 13.4.92 (RTX 5090) and sm_89 at 13.0.48 (RTX 4090): the maximum
    # of the two, which is sm_89's for chem_dust (72 B against 0) and
    # chem_optics (1,904 B against 1,648).  lane/aq-core's chem rows and
    # the dycore-host units follow in their own places below.
    "chem_ageing": 0,
    "chem_drydep_gocart": 0,
    "chem_dust": 72,
    "chem_inventory": 0,
    "chem_mp_coupling": 0,
    "chem_optics": 1904,
    "chem_rrtmgp_aerosol": 0,
    # cap-aer-optics: NVRTC sm_120 measured 0 B, gpu-tests.json 2026-10-20.
    "chem_radiation": 0,
    # cap-fzra (aa11dd9dc41): the saved-frame freezing-rain/FRAM unit,
    # saved_frame_ice 0 B on sm_120 at NVRTC 13.4.92 (read 2026-10-20,
    # lab box RTX 5090, its own --std=c++17 site and the loader alike).
    "ice_accretion": 0,
    "chem_seasalt": 0,
    "chem_settling": 0,
    "chem_sulfur": 32,
    # WRF-Chem's Wesely gas dry deposition (lane/aq-cams): 0 B on sm_120 and
    # sm_89 at NVRTC 13.4.92 (read 2026-09-30, RTX 5090 and RTX 4090).
    "chem_drydep_wesely": 0,
    # Added with this box's recording; all three are 0 B and none
    # was in the ceiling before. 'ntiedtke' is what cu_physics = 16
    # needs to be priceable at all.
    'ntiedtke': 0,
    "acoustic": 544,
    "advection": 0,
    # The chem program (woof/core/chem_*.py, lane/aq-core), read on the
    # sm_120 NVRTC 13.4.92 recording.  chem_vertmx is level-specialized
    # (LEVEL_SPECIALIZED_KERNEL_FRAMES): 2,048 B is its CHEM_NZ = 256
    # ceiling; a 59-level run launches 472 B.  mono_advection (WRF's
    # advect_scalar_mono) is 0 B; its row sits in order below.
    "chem_bdy": 0,
    # The smoke lane's units (lane/aq-smoke), the same recording: the fire
    # classifier's 16 group sums, and the Freitas plume state in its global
    # workspace rather than the thread frame.
    "chem_fire": 64,
    "chem_ledger": 0,
    "chem_outputs": 0,
    "chem_plumerise": 0,
    "chem_prep": 0,
    "chem_vertmx": 2048,
    "chem_wetdep_ls": 0,
    "coriolis_map": 0,
    "diagnostics": 0,
    "diff6": 0,
    "diff6_seam": 0,
    "diffusion": 0,
    "dycore": 0,
    # The dycore-host speed lane's four point-local units (66af318bc),
    # launched by every integration: 0 B each, read 2026-09-30 on sm_120
    # (RTX 5090) and sm_89 (RTX 4090) at NVRTC 13.4.92.  No array in any
    # of their kernels, so none can reserve backing store.
    "face_mass": 0,
    # zadvect_implicit = 1 (A158, kernels/ieva.cu) at its shipped
    # IEVA_KMAX = 65: the five column solves hold two double columns of
    # 65 words each, 1,040 B, and the split and column-mass kernels none.
    # Read 2026-10-01 on sm_120 (RTX 5090), NVRTC 13; IEVA_TIER_FRAME
    # prices the deeper tiers (2,064 B at 129, 4,112 B at 257, read alike).
    "ieva": 1040,
    "held_heating": 0,
    "rk_bookkeeping": 0,
    "surface_w": 0,
    # The FTZ receipt's probe rides the production loader from the same
    # kernels directory, so the local-frame sweep sees it like any model
    # module; it holds no local frame.
    "ftz_probe": 0,
    # Grell-Freitas.  One thread still owns one whole GFDRV column, but
    # the column arrays no longer live in the per-thread local frame:
    # woof/core/kernels/gf.cu keeps them in a global workspace that
    # woof/core/gf.py sizes to the threads IN FLIGHT, because the driver
    # sizes the local-memory backing store to the threads the card can
    # ever hold.  MEASURED on test host A (RTX 5070 Ti, sm_120): the frame went
    # 22,416 -> 72 B on NVRTC 13.3 and 22,416 -> 88 B on NVRTC 13.0.48,
    # and the launch-time reservation went 2,200.0 MiB -> 4.0 MiB.  Both
    # readings are under the 1,024 B default stack, so the row reserves
    # nothing on any card, and it no longer moves with nz -- the frame is
    # 72 B at the 40, 49, 55 and 64 tiers alike, which retires the
    # level-specialization gap this row used to carry.
    # The workspace itself is priced by :func:`gf_column_workspace_bytes`.
    # Re-read 2026-09-28, every sm_120 build (13.0.48 through 13.4.92)
    # compiles the current source to 72 B; the 88 here is the sm_86
    # recording's 2026-08-21 reading, which is re-read at the cut.
    "gf": 88,
    "health": 0,
    # The tile-streamed health reduction (woof/core/streaming.py:1728).
    # It holds no local frame on either measured architecture.  The row
    # exists because the ``.cu`` does: it shipped with the out-of-core
    # merge after the last 5090 reading, so the regeneration gate had a
    # module it could not even enumerate until 2026-08-20.
    "health_tile": 0,
    # The batched symmetric eigensolver the radar-DA analysis factors with
    # (woof/core/jacobi_eigh.py).  It holds NO local frame at any tier: the
    # whole k x k problem lives in dynamic SHARED memory, which is priced at
    # launch and released with the block rather than reserved per resident
    # thread for the life of the process.  That is the entire reason the
    # kernel is written around shared memory instead of per-thread arrays.
    "jacobi_eigh": 0,
    "kessler": 5120,
    # Kain-Fritsch.  One thread owns one whole column, and every column
    # array lives in a global workspace woof/core/kf.py sizes to the threads
    # IN FLIGHT, because the driver sizes the local-memory backing store to
    # the threads the card can ever hold.  MEASURED on test host A (RTX 5070 Ti,
    # 70 SMs x 1,536, sm_120, NVRTC 13.0.48): the pre-workspace frame was
    # 9,216 B and its launch-time reservation 840.0 MiB.
    #
    # The row is the element-wise ceiling over the recorded platforms
    # (kernel_frame_recordings), 512 B from the readings of the kernel that
    # kept two arrays on the stack.  The WRF port of 2026-10-07 keeps none
    # and reads 0 B (W1, RTX PRO 6000 Blackwell, sm_120, NVRTC 12.9), so
    # the ceiling over-prices by at most half the default stack, i.e. it
    # still reserves nothing on any card.
    # The workspace itself is priced by :func:`kf_column_workspace_bytes`.
    "kf": 512,
    "kf_validation": 0,
    "lbc_flow": 0,
    "lbc_state": 0,
    # Both boundary-time entry points: driver-read0B on sm86 with NVRTC
    # 13.0.48 (Windows) and12.8.93 (WSL), 2026-09-05. See the recordings.
    "lbc_time": 0,
    # Milbrandt-Yau.  MEASURED on an RTX 5090 over all seven kernels of the
    # module: milbrandt2_sediment_256 is the only one with a frame worth
    # naming at 2,048 B -- exactly its two per-thread column arrays
    # ``float VVQ[KMAX]; float VVN[KMAX]`` at MY2_KMAX_GENERIC = 256
    # (2 * 256 * 4) -- with milbrandt2_sediment_64 at 512 B on the same
    # arrays at MY2_KMAX_SHALLOW = 64, and prelim/geometry/warm/cold/
    # diagnostics holding no local frame at all.
    #
    # This row is a CEILING in the same sense Morrison's is: the launcher
    # picks the 64-level kernel for nz <= 64 (woof/core/milbrandt2.py:185),
    # so a shallow configuration is priced 1,536 B per thread over what it
    # reserves.  It is NOT a LEVEL_SPECIALIZED_KERNEL_FRAMES case -- both
    # tiers are compiled into the shipped unit at fixed bounds rather than
    # recompiled at the configuration's nz -- and it needs no tiered entry
    # above the row, because milbrandt2's VERTICAL_LEVEL_BOUNDS = (3, 256)
    # refuses anything deeper than the 256 tier this 2,048 B measures.
    "milbrandt2": 2048,
    # The pure Z block lifted out of milbrandt2.cu so the radar
    # observation operator can launch it without the scheme's state
    # update (woof/core/kernels/milbrandt2_zet.cu).  It reads one cell
    # and holds no column, so it reserves nothing -- MEASURED 0 B on the
    # RTX 3080 at NVRTC 13.0.48, the one recording that carries a row for
    # this module (kernel_frame_recordings.SM86_NVRTC_13_0_48).  This
    # table is the element-wise maximum over the recordings and is checked
    # for exact equality against them at import, so the 0 here IS that
    # reading rather than a second statement of it.
    "milbrandt2_zet": 0,
    "mono_advection": 0,
    "morrison": 5120,
    "microphysics_validation": 0,
    "wsm5": 7072,  # sm_120, NVRTC 12.9.86, unflushed WSM5_KMAX=80, 2026-10-08.
    # MYJ.  MEASURED on an RTX 5090 by the driver sweep this table is
    # checked against, at the kernel's compiled MYJ_KMAX = 128 tier: the
    # PBL translation unit holds 9,232 B and the Eta surface layer holds
    # none.  One thread owns one whole column and MYJPBL carries MIXLEN's
    # GM/GH/EL/Q2, DIFCOF's AKM/AKH, VDIFH's tridiagonal coefficients and
    # the species stack all at once, which is where the frame goes; the
    # surface layer is scalar per column, so it has nothing to hold.
    #
    # This closes the port's own open question ("nobody has measured the
    # local-memory cost of this kernel at production width").  By the
    # reservation model at the head of docs/kernel_local_memory_bounds.md
    # that frame reserves (9232 - 1024) * 1536 * 170 = 2,143,764,480 B
    # ~ 2.00 GiB on first launch, for the life of the process -- more than
    # SASE's and a third of GF's, and it is the price of selecting MYJ at
    # all, not of the domain size.  Like SASE's, it should be linear in
    # the compiled level bound and is therefore specializable; that is a
    # separate change, and until then the stated bound is the compiled
    # ceiling, which is the safe direction for a rail gate.
    "myjpbl": 9232,
    "myjsfc": 0,
    # SASE.  MEASURED on an RTX 5090 over all 29 kernels in the module at
    # the closure's compiled tier (SASE_KMAX = 128): the maximum is
    # sase_plume_vent_flux at 6,272 B; sase_moist_n2 5,120 B,
    # sase_vertical_channel 4,096 B, the two Thomas sweeps 3,072 B each,
    # and the remaining 24 kernels hold no local frame at all.  By the
    # reservation model at the head of docs/kernel_local_memory_bounds.md
    # this frame reserves (6272 - 1024) * 1536 * 170 = 1,370,357,760 B
    # ~ 1.28 GiB on first launch, for the life of the process.
    #
    # The frame is EXACTLY LINEAR in the compiled level bound -- measured
    # 1,568 / 3,136 / 6,272 B at SASE_KMAX 32 / 64 / 128, i.e. 49 B per
    # level with a zero intercept -- so this module is specializable the
    # way kf and refl are, and at a 49-level configuration the frame
    # would be 2,401 B and the reservation ~359 MiB.  It is NOT entered
    # in LEVEL_SPECIALIZED_KERNEL_FRAMES here, because that table prices
    # what the launcher actually compiles and this launcher compiles at
    # the fixed tier.  Specializing it is a real ~0.93 GiB saving and a
    # separate change; until then the stated bound is the compiled
    # ceiling, which is the safe direction for a rail gate.
    "sase": 6272,
    # The two MYNN-EDMF qn-family modules (4a0bb3f69) landed declared
    # UNMEASURED because their lane had no measurement platform; the
    # 2026-08-31 sm_86 campaign measured both at 0 B (RTX 3080, NVRTC
    # 13.0.48; test host A's sm_120 at NVRTC 13.0.88 also read 0 B), which
    # retired that declaration.  Scalar per column, no per-thread stack
    # arrays -- the same shape as mynn_pbl itself.
    "mynn_dmp_sibling": 0,
    "mynn_pbl": 0,
    "mynn_scalar_mix": 0,
    "mynn_surface": 0,
    "nest": 0,
    "nest_microphysics": 0,
    # sm_86 compiles Noah 48 B wider than sm_120 (224 against 176); sm_89
    # compiles the 2026-10-12 source to 320 B (re-read 2026-10-22), so the
    # ceiling is the sm_89 reading.  Well under the default stack either
    # way, so it reserves nothing on any card.
    "noah": 320,
    "noahmp_bareflux": 0,
    "noahmp_fluxprep": 0,
    "noahmp_leaves": 272,
    "noahmp_radiation": 0,
    "noahmp_sflx": 0,
    "noahmp_snow": 200,
    "noahmp_soilwater": 0,
    "noahmp_vegeflux": 0,
    "noahmp_vegprecip": 0,
    "noahmp_water": 224,
    "nssl2": 15504,
    "nssl2_diagnostics": 0,
    "nssl2_driver_support": 15504,
    # NVRTC 13.3.33 emits 216 B here where 13.0.48 emitted 112, at the
    # same sm_120 target: a COMPILER-build move, not an architecture one.
    "nssl2_fused_gs": 216,
    "nssl2_nucond": 0,
    "nssl2_qvexcess": 0,
    "openbc": 0,
    "pd_advection": 0,
    "refl": 18432,
    # WDM6's own reflectivity translation unit (see wdm6_refl.cu's
    # header for why it is not part of refl.cu).  63 B/level against
    # refl.cu's widest 72: WDM6 diagnoses one N0 array from nr where
    # Morrison carries three.
    "lin": 37808,  # G2 and V12, sm_120, NVRTC 12.9.86, LIN_KMAX=128
    "lin_refl": 14080,  # V12/V13, same compiler, unspecialized REFL_KMAX=256
    "wdm6_refl": 16128,
    # The legacy RRTMG adapter's device glue and its device wrapper prep:
    # 0 B each, read on sm_120 at NVRTC 13.4.92 (the only recording that
    # has them).
    "rrtmg_legacy_adapter": 0,
    "rrtmg_legacy_prep": 0,
    # The adapter's WRF v4.6.1 radconst + calc_coszen (WOOF's own float32
    # trig): 0 B for both entry points, read on sm_120 at NVRTC 12.9.86.
    "rrtmg_legacy_solar": 0,
    # The exact chain's strict Exner unit (physics_exner.cu): 0 B, read
    # on sm_120 at NVRTC 13.4.92.
    "physics_exner": 0,
    "rrtmg_lw": 0,
    "rrtmg_mcica_wrf": 0,
    # Another compiler-build move at a fixed sm_120: 0 B on NVRTC 13.0.48,
    # 40 B on 13.3.33.
    "rrtmgp_cloud": 40,
    "rrtmgp_gas": 512,
    "rrtmgp_mcica": 0,
    # The RRTMGP optimisation took this from 5,152 to 3,600:
    # rrtmgp_sw_2stream dropped denom/dif_dn/dif_up.  sm_86 was re-read at
    # NVRTC 13.0.48 when it landed; the sm_120 recordings kept the older
    # 5,152 until every sm_120 build was re-read on 2026-09-28 (RTX 5090
    # and RTX 5070 Ti, NVRTC 13.0.48, 13.0.88, 13.3.33 and 13.4.92, all
    # 3,600).  The stale 5,152 was not free: it was the widest frame a
    # Morrison or Kessler configuration with RRTMGP radiation launched
    # (both are 5,120), so those runs reserved 32 B per resident thread
    # more backing store than any card compiles, 8,355,840 B on a 170-SM
    # card, and a P3 or microphysics-free one the whole 1,552 B,
    # 405,258,240 B.
    "rrtmgp_rte": 3600,
    "rrtmgp_validation": 0,
    "ruc": 144,
    "saxpy": 0,
    "sfclay": 0,
    # Shin-Hong (bl_pbl_physics=11).  Its column arrays moved into a
    # global workspace on 2026-09-30 (the ysu.cu route): 17,160 -> 0 B,
    # re-read at the new source on sm_120 under NVRTC 13.0.48, 13.0.88,
    # 13.3.33 and 13.4.92, on sm_89 under 13.4.92 and on sm_86 under
    # 13.0.48 (woof/core/kernel_frame_recordings.py).  The workspace is
    # priced separately by shinhong_column_workspace_bytes.
    "shinhong": 0,
    "shinhong_validation": 0,
    "smag2d": 0,
    # Coordinate horizontal diffusion: both exports read 0 B on sm_120
    # through the production loader at all five recorded NVRTC builds.
    # The frame recordings name the RTX 5090-only measurement.
    "diff_opt1": 0,
    # Merged bandwidth and native cold-start leaf modules: every raw export
    # reads 0 B on RTX 5090 at the same five sm_120 compiler builds.
    "bandwidth_glue": 0,
    "noah_init": 0,
    "wrf_cold_start_w": 0,
    "spec_bdy": 0,
    "thompson": 11264,
    # mp_physics=28 translation units, measured 2026-07-31 the same way on
    # the same RTX 5090.  ``thompson_aerosol_sed``'s 9,216 B is its
    # 256-LEVEL cloud sedimentation variant
    # (``thompson_aa_cloud_sediment_256``); a run with nz <= 64 launches the
    # 64-level entry point and a much smaller frame, so this row is the
    # module CEILING exactly as the header above describes for Morrison.
    # ``thompson_aerosol_probe`` is a table row but never a priced module:
    # no ``physics_kernel_modules`` selector names it (it is the
    # device-helper oracle unit), and it is listed here only so the
    # driver-regeneration test covers every ``.cu`` in the directory.
    #
    # RE-MEASURE THESE SIX before the mp=28 wave closes.  They were taken
    # while a sibling package was still moving shared device helpers into
    # thompson_aerosol_common.cuh; sat/state/probe were measured against the
    # current header and warm/sed reproduce their values with the duplicate
    # definitions stripped, but ``thompson_aerosol_cold`` was measurable
    # only against the pre-move header.  Nothing has to be remembered for
    # that to be caught: ``test_the_recorded_local_frames_match_the_driver``
    # regenerates the whole table from NVRTC + the driver and fails loudly
    # on any row that moved.
    "thompson_aerosol_cold": 0,
    "thompson_aerosol_probe": 0,
    # 0 B on sm_89 and sm_120 at NVRTC 13.4.59 and 13.4.92; 48 B on sm_120
    # at NVRTC 12.9.86 (SM120_NVRTC_12_9_86, read 2026-10-01).
    "thompson_cold_start": 48,
    "thompson_aerosol_sat": 0,
    "thompson_aerosol_sed": 9216,
    # RE-MEASURED 2026-08-03 on the reference RTX 5090: 40, not 48.  The
    # module's only framed kernel is ``thompson_aa_init_profile``, and 40 is
    # what NVRTC 13.0 (system CUDA 13.0), NVRTC 12.9 (wheel, forced) and an
    # offline ``nvcc`` 13.0.48 all produce from the byte-identical source, at
    # every option variant this tree could plausibly use.  Nothing reproduces
    # 48 -- see the commit message.  The row cannot bind the reservation
    # either way: it is a MAX over the selected modules, and mp_physics=28
    # always co-selects ``thompson`` at 11,264 B.
    "thompson_aerosol_state": 40,
    # Architecture move at a fixed NVRTC 13.0.48: 0 B on sm_120, 112 B on
    # sm_86.
    "thompson_aerosol_warm": 112,
    "tke_budget": 0,
    "topo_radiation": 40,
    # WRF swint_opt = 1 and aer_opt = 3 (lane 286-aer-swint): 0 B on sm_120
    # at NVRTC 13.4.92 (kernel_frame_recordings).
    "swint": 0,
    # Production loader on RTX 5090, NVRTC 13.4.92, 2026-10-04.
    "solar_albedo": 0,
    # Dudhia shortwave (lane/verify-dudhia-sw): 0 B, 48 registers, on sm_120
    # at NVRTC 12.9.86 and 13.4.92 (kernel_frame_recordings).
    "dudhia_sw": 0,
    "rrtmg_aer3": 24,  # sm_89 / NVRTC 13.4.92, 2026-10-22; 0 B on sm_120.
    "uh_diag": 0,
    # The UW moist-turbulence PBL (bl_pbl_physics = 9): uwpbl_columns keeps
    # CAM's automatic arrays in a global per-column workspace
    # (:func:`uwpbl_column_workspace_bytes`), so what is left on the stack
    # is its binary64 scalars at 255 registers: 3,184 B on sm_89 and 864 B
    # on sm_120, both at NVRTC 13.4.92 (kernel_frame_recordings).
    "uwpbl": 3184,
    "satmedmf": 112,  # Recorded SM89/NVRTC 13.4.92 module maximum.
    # Fused preparation interpolation, 16 B on sm_89 and sm_120, NVRTC 13.4.92.
    # Portable binary64 RH kernel is 0 B; regular interpolation remains 16 B.
    "horizontal": 16,
    # Device initialize_real: real_init 0 B on sm_89 and sm_120; real_init_math
    # 0 B on sm_120 and 48 B on sm_89 (real_thermo), NVRTC 13.4.92.
    "real_init": 0,
    "real_init_math": 48,
    "vert_interp": 768,
    # The original cu6 kernel (cu_physics = 6): 24 B on sm_120, NVRTC
    # 12.9.86 (RTX 5090, 2026-10-09). Under the default stack, so it
    # reserves nothing.
    "tiedtke": 24,
    # WDM6 (mp_physics=16).  MEASURED 2026-08-10 on the reference RTX 5090
    # the same NVRTC + driver way as every other row: the module's one
    # kernel, ``wdm6_column``, holds 9,776 B at the source's ``#ifndef``
    # default ``WDM6_KMAX = 64`` (88 registers).  One thread owns a whole
    # column, so the frame is that bound's ~30 per-thread work arrays.
    #
    # This row is the 64 TIER, not a ceiling: the launcher compiles the 80
    # tier for 65 <= nz <= 80 and that frame is WIDER (12,208 B measured).
    # :data:`WDM6_TIER_FRAME` carries the ladder, and the measurements
    # behind it -- exactly 152 B per level between the two rungs -- are
    # re-read off the driver by tests/test_kernel_local_bounds.py.
    #
    # 308c2d39e (WDM6 rain mass and number conservation) shrank the frame:
    # 9,264 B at 64 and 11,568 B at 80, 144 B per level, on every sm_120
    # build read 2026-09-28 (RTX 5090: NVRTC 12.9.86, 13.0.48, 13.4.92;
    # RTX 5070 Ti: 13.0.88, 13.3.33), and the three sm_120 recordings in
    # woof/core/kernel_frame_recordings.py carry it.  This row is the
    # element-wise maximum over the recordings, so it stays at the sm_86
    # recording, which the RTX 3080 re-read at the 2.8.0 cut's Windows
    # step (2026-09-29): 9,264 B at 64 and 11,568 B at 80 on sm_86 too,
    # with NVRTC 13.4.92 and 13.0, so every recorded card now agrees.
    "wdm6": 9264,
    # WSM6 (mp_physics=6).  This row is the 64 TIER: the launcher compiles
    # the 80 tier for 65 <= nz <= 80 and that frame is WIDER.
    # :data:`WSM6_TIER_FRAME` carries the ladder, and
    # tests/test_kernel_local_bounds.py re-reads both rungs off the driver.
    # Since lane/verify-wsm6-mp6 (2026-10-07, the WRF-exact rewrite) every
    # sm_120 build reads 6,448 B here (NVRTC 12.9.86, 13.0.48, 13.0.88,
    # 13.3.33, 13.4.92); the row stays at the 7,216 B the sm_86 and sm_89
    # recordings carry from the previous source, the element-wise maximum,
    # until those cards are read again.  Since the column workspace
    # (2026-10-16, memory census item 12) both kernels read 0 B on sm_120
    # (NVRTC 13.4.92): the arrays live in a per-call global workspace of
    # (25 nz + 8) x 4 B per column in flight.  This row stays the ceiling,
    # which over-prices that workspace (safe direction) until the
    # workspace is priced on its own and every recording is re-read.
    "wsm6": 7216,
    "wsm3": 5664,
    # YSU (bl_pbl_physics = 1, the SHIPPED DEFAULT).  One thread still owns
    # one whole column, but the column arrays no longer live in the
    # per-thread local frame: woof/core/kernels/ysu.cu keeps them in a
    # global workspace that woof/core/ysu.py sizes to the threads IN
    # FLIGHT, because the driver sizes the local-memory backing store to
    # the threads the card can ever hold.
    #
    # MEASURED on test host A (RTX 5070 Ti, 70 SMs x 1,536,
    # sm_120) through the real launcher at nz=49: the frame went 9,232 -> 0
    # B on BOTH NVRTC 13.0 and 13.3, and the launch-time reservation went
    # 842.0 MiB -> nothing.  A zero frame reserves nothing on any card, and
    # it no longer moves with nz -- the workspace extent is a RUNTIME
    # argument, so a 49-level run holds 50 levels of arrays where the frame
    # had to hold 128.
    #
    # This row mattered more than any other: bl_pbl_physics = 1 is the
    # wizard's default (woof/domain_wizard.py:714), so this was the
    # widest frame a BARE DEFAULT run launched, and every default run paid
    # the 842.0 MiB.
    # The workspace itself is priced by :func:`ysu_column_workspace_bytes`.
    "ysu": 0,
    "ysu_480": 0,
    "ysu_validation": 0,
    # The urban canopy models (sf_urban_physics 1-3), read 2026-09-30 at
    # NVRTC 13.4.92 on sm_89 (RTX 4090) and sm_120 (RTX 5090):
    # urban_ucm 72 B on sm_89 and 0 B on sm_120 since its green-roof
    # constants moved to __constant__ memory (both under the 1,024 B
    # default stack, so neither reserves anything); urban_bep and
    # urban_bep_couple 0 B; myjurb, the MYJ column under BEP, 10,256 B on
    # both, which DOES reserve: (10,256 - 1,024) B x SMs x threads per SM.
    # A167: sm_120 under NVRTC 12.9.86, the default [gpu] extra's compiler,
    # compiles three of them wider (kernel_frame_recordings.
    # SM120_NVRTC_12_9_86): myjurb 12,304 B, which was 0.205 GiB of
    # reservation uncharged on an RTX 5070 Ti; urban_ucm 552 B and
    # urban_bep 16 B, both under the default stack.
    "myjurb": 12304,
    "urban_bep": 16,
    "urban_bep_couple": 0,
    "urban_ucm": 552,
    # 2026-10-17, test host B (RTX 5090, sm_120, NVRTC 13.4.92): the two
    # modules that left UNMEASURED_KERNEL_MODULES on the 2.9.2 line.
    # goddard at its source default GDD_MAX_LEVELS tier; urban_ucm480 is
    # the WRF 4.8 full-canopy SLUCM (kernel_frame_recordings, same date).
    "goddard": 43664,
    "urban_ucm480": 848,  # sm_89 / NVRTC 13.4.92, 2026-10-22 (352 B on sm_120).
    # RTX 5070 Ti production loaders, 2026-10-18, both numerics modes.
    "nextgen_cfe": 0, "nextgen_reservoirs": 0,
    "nextgen_inundation": 0, "nextgen_terrain": 56,
    # The Fitch wind-farm unit (labs/windfarm, 2.9.11): RTX 4090 sm_89,
    # NVRTC 12.9.86 and 13.4.92, fast and strict, 2026-10-30.
    "wind_farm": 56,
}

#: The readings the ceiling above is made of, each naming its box, its
#: target architecture and its NVRTC build.
KERNEL_LOCAL_FRAME_RECORDINGS = _kernel_frame_recordings.\
    KERNEL_LOCAL_FRAME_RECORDINGS

#: Re-export so a caller that has a recording can price against it
#: without importing a second module.
KernelFrameRecording = _kernel_frame_recordings.KernelFrameRecording

if dict(KERNEL_MAX_LOCAL_SIZE_BYTES) != _kernel_frame_recordings.frame_ceiling():
    _low = sorted(
        module for module, frame
        in _kernel_frame_recordings.frame_ceiling().items()
        if KERNEL_MAX_LOCAL_SIZE_BYTES.get(module, -1) != frame)
    raise RuntimeError(
        "KERNEL_MAX_LOCAL_SIZE_BYTES must be exactly the element-wise "
        "maximum over woof.core.kernel_frame_recordings: "
        f"{', '.join(_low)} disagree.  A row below a measurement "
        "under-charges the local-memory backing store on the platform "
        "that measured it")


def kernel_frame_recording_for(fingerprint) -> "KernelFrameRecording | None":
    """The frame recording taken on THIS compile platform, or ``None``.

    Matched on the two
    :func:`woof.certify.compile_platform.compile_platform_fingerprint`
    keys that decide code generation -- the target architecture and the
    NVRTC build.  ``None`` means "nobody has measured this compiler on
    this architecture", which is the case the ceiling exists for.
    """

    return _kernel_frame_recordings.recording_for(fingerprint)


@dataclass(frozen=True)
class UnderPricedKernelFrame:
    """A module whose real frame is wider than the shipped row.

    ``unpriced_device_bytes`` is the whole point: the reservation is
    ``(frame - default stack) x resident-thread capacity``, so a frame
    delta is a device-memory delta the fit gate never charged.  A run
    admitted on the short number does not fail at the gate -- it fails
    later, out of memory, with nothing pointing back here.
    """

    module: str
    shipped_bytes: int
    observed_bytes: int
    unpriced_device_bytes: int


def under_priced_kernel_frames(
        observed, *, profile: "DeviceLocalMemoryProfile | None" = None
) -> dict[str, UnderPricedKernelFrame]:
    """Modules this compiler emits WIDER than :data:`KERNEL_MAX_LOCAL_SIZE_BYTES`.

    Empty is the only acceptable answer on any machine: over-pricing
    costs headroom, under-pricing breaches the rail.  A module the
    shipped table does not know about is reported too, priced against
    zero -- an unpriced module is the worst case of the same defect.
    """

    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    capacity = profile.resident_thread_capacity
    over: dict[str, UnderPricedKernelFrame] = {}
    for module, frame in dict(observed).items():
        shipped = int(KERNEL_MAX_LOCAL_SIZE_BYTES.get(module, 0))
        if int(frame) <= shipped:
            continue
        over[module] = UnderPricedKernelFrame(
            module=module, shipped_bytes=shipped, observed_bytes=int(frame),
            unpriced_device_bytes=(int(frame) - shipped) * capacity)
    return over


@dataclass(frozen=True)
class LevelSpecializedFrame:
    """A kernel module whose per-thread local frame is compiled to ``nz``.

    ``refl.cu`` declares its per-thread column arrays against a
    ``#ifndef``-guarded compile-time bound, and its launcher specializes
    that bound to the field's own level count through
    ``woof.core.kernels.get_kernel_int_defines``.  The column arrays are the
    only thing in the kernel that scales with the bound, so the frame the
    driver reports is ``bytes_per_level * levels`` rounded up to the local
    frame's 8-byte granularity.

    Measured on the RTX 5090 (driver 610.74, NVRTC 13.x, ``-std=c++17``),
    module maximum per bound:

      ==========  ============  ==========  ==========
      module      bound         frame       reserved
      ==========  ============  ==========  ==========
      refl        256 (ceiling)  18,432 B   4,334 MiB
      refl         49            3,528 B      626 MiB
      refl         30            2,160 B      286 MiB
      ==========  ============  ==========  ==========

    ``kf`` used to be the widest row in this table (24,064 B at its 128
    ceiling, 5,738 MiB reserved).  It is not here any more: its column
    arrays live in a global workspace and its frame stopped following
    ``nz``.

    ``tests/test_kernel_local_bounds.py`` re-measures every row against the
    driver, so a compiler or source change that breaks the linear form fails
    loudly instead of silently mispricing a rail gate.
    """

    module: str
    define: str
    unspecialized_levels: int
    bytes_per_level: int
    alignment_bytes: int = 8

    def frame_bytes(self, levels: int) -> int:
        levels = int(levels)
        if levels < 1:
            raise ValueError(
                f"{self.module}: level-specialized frame needs levels >= 1, "
                f"got {levels}")
        if levels > self.unspecialized_levels:
            raise ValueError(
                f"{self.module}: {levels} levels exceeds the {self.define} "
                f"ceiling of {self.unspecialized_levels}")
        raw = self.bytes_per_level * levels
        remainder = raw % self.alignment_bytes
        return raw if remainder == 0 else raw + self.alignment_bytes - remainder


#: The two modules whose local frame follows the configuration's ``nz``
#: exactly.  (``acoustic`` and ``wdm6`` follow a TIER of ``nz`` instead and
#: live in :class:`TieredKernelFrame` below.)  ``bytes_per_level`` is fixed
#: by construction against the unspecialized row of
#: :data:`KERNEL_MAX_LOCAL_SIZE_BYTES` (checked at import below), so the two
#: tables cannot drift apart.
LEVEL_SPECIALIZED_KERNEL_FRAMES: dict[str, LevelSpecializedFrame] = {
    # ``kf`` USED TO SIT HERE at 188 B/level, and it is gone rather than
    # re-fitted.  52 of its 54 column arrays moved into a global workspace
    # on 2026-08-21 (woof/core/kernels/kf.cu), which left 188 B/level
    # wrong by a factor of 47, and -- because the two that stayed are the
    # only thing ``KF_KMAX`` still sizes, at 8 B/level -- made specializing
    # the bound worth 312 B of frame that reserves nothing either way.  So
    # woof/core/kf.py stopped recompiling the module per level count, and
    # the ONE binary that can now launch holds a frame that does not move
    # with ``nz`` at all.  MEASURED on test host A (RTX 5070 Ti, sm_120, NVRTC
    # 13.0.48): 512 B at the shipped KF_KMAX = 128 -- half the 1,024 B
    # default stack, so it reserves nothing on any card.  The flat row in
    # KERNEL_MAX_LOCAL_SIZE_BYTES is the whole model.
    "refl": LevelSpecializedFrame("refl", "REFL_KMAX", 256, 72),
    # 16-byte granularity, not refl.cu's 8: measured against the driver at
    # ten bounds from 256 down to 10 (63*n rounded up to 16 reproduces every
    # one).  At bound 1 the compiler eliminates the frame entirely and the
    # model over-prices by 64 B, which is the safe direction and the reason
    # tests/test_wdm6.py pins the realistic bounds rather than that one.
    "wdm6_refl": LevelSpecializedFrame(
        "wdm6_refl", "REFL_KMAX", 256, 63, alignment_bytes=16),
    # WRF-Chem's vertmx (woof/core/chem_vertmx.py specializes CHEM_NZ to
    # the column): its two Thomas recurrences, 8 B per level, read at nine
    # bounds from 11 to 256 on the sm_120 NVRTC 13.4.92 recording.
    "chem_vertmx": LevelSpecializedFrame("chem_vertmx", "CHEM_NZ", 256, 8),
}

# The launchers select only 64-level fallout variants at nz <= 64.
# The full module ceilings remain in KERNEL_MAX_LOCAL_SIZE_BYTES for
# deeper runs and the compilation census. Charging a loaded but unlaunched
# 256-level variant reserved memory the driver never allocates. Measured
# through the production loader on RTX 4090: the maximum over every plain
# and 64-level export is 2,816 / 2,304 bytes. The classic value also matches
# the earlier RTX 5090 recording in kernel_frame_recordings.py.
THOMPSON_SHALLOW_KERNEL_FRAMES = {
    "thompson": 2816,
    "thompson_aerosol_sed": 2304,
}
THOMPSON_SHALLOW_LEVELS = 64


THOMPSON_WRF39_KERNEL_FRAMES = {
    "thompson_aerosol_sed": (11264, 2816),
}

for _spec in LEVEL_SPECIALIZED_KERNEL_FRAMES.values():
    if (_spec.frame_bytes(_spec.unspecialized_levels)
            != KERNEL_MAX_LOCAL_SIZE_BYTES[_spec.module]):
        raise RuntimeError(
            f"{_spec.module}: level-specialized frame model disagrees with "
            "the driver-measured unspecialized frame")
del _spec


@dataclass(frozen=True)
class TieredKernelFrame:
    """A module whose local frame follows a compile-time TIER, not ``nz``.

    ``refl`` compiles exactly to the configuration's level count;
    ``acoustic`` instead picks the smallest of a short ladder of
    ``WPHI_MAX_LEV`` values (``woof.core.acoustic.WPHI_LEVEL_TIERS``), so
    its frame is a step function of ``nz`` and is constant within a tier.

    The frame at the shipped tier is the driver-MEASURED row of
    :data:`KERNEL_MAX_LOCAL_SIZE_BYTES`.  Deeper tiers are priced from it by
    the one object the bound sizes -- the ``real rhs[WPHI_MAX_LEV]`` column
    that ``advance_w_phi``/``advance_w_phi_msf`` each declare -- at
    ``bytes_per_level`` per added full level.  That extrapolation is
    PROVISIONAL until ``tests/test_kernel_local_bounds.py`` reads the deeper
    tiers back off the driver; a register-allocation change at a deeper tier
    could add more, and under-pricing is the direction that hurts, so the
    device test is a gate rather than a formality.
    """

    module: str
    define: str
    shipped_tier: int
    bytes_per_level: int
    alignment_bytes: int = 8
    # The standalone reading may use a deeper bound than the minimum tier.
    reference_tier: int | None = None

    def frame_bytes(self, tier: int) -> int:
        tier = int(tier)
        if tier < self.shipped_tier:
            raise ValueError(
                f"{self.module}: {self.define}={tier} is below the shipped "
                f"tier {self.shipped_tier}")
        anchor = self.shipped_tier if self.reference_tier is None else self.reference_tier
        raw = (KERNEL_MAX_LOCAL_SIZE_BYTES[self.module]
               + self.bytes_per_level * (tier - anchor))
        remainder = raw % self.alignment_bytes
        return raw if remainder == 0 else raw + self.alignment_bytes - remainder


#: ``acoustic.cu``'s ``WPHI_MAX_LEV`` sizes one FP32 column per thread.
ACOUSTIC_TIER_FRAME = TieredKernelFrame("acoustic", "WPHI_MAX_LEV", 129, 4)

#: ``ieva.cu``'s ``IEVA_KMAX`` sizes two FP64 columns per thread, 16 B per
#: level: 1,040 / 2,064 / 4,112 B at the 65 / 129 / 257 tiers
#: (:data:`woof.core.ieva_constants.IEVA_LEVEL_TIERS`), read exactly so.
IEVA_TIER_FRAME = TieredKernelFrame("ieva", "IEVA_KMAX", 65, 16)

#: ``wdm6.cu``'s ``WDM6_KMAX`` sizes the whole per-thread column stack, and
#: ``woof/core/wdm6.py`` compiles it at one of two rungs
#: (``wdm6_constants.WDM6_KERNEL_LEVEL_TIERS`` = 64, 80) rather than at
#: ``nz``.  That is why WDM6 is priced HERE and not in
#: :data:`LEVEL_SPECIALIZED_KERNEL_FRAMES`: an nz-linear model would price
#: a 49-level WDM6 run at 7,056 B when the kernel it actually launches
#: holds 9,264 B, and under-pricing a rail gate is the direction that put a
#: run 1,630 MiB over.
#:
#: MEASURED on the reference RTX 5090 over sixteen bounds from 2 to 80, and
#: re-read after 308c2d39e on every sm_120 build and on the RTX 3080 (sm_86).
#: The two rungs the launcher can compile are exactly linear in the bound --
#: 9,264 B at 64 and 11,568 B at 80, 2,304 B over 16 levels = 144 B/level --
#: which is what this frame reproduces, exactly, at both.  (Between the
#: rungs the driver's frame wanders by up to 16 B against the same line; the
#: launcher never compiles there, and the model is a ceiling on that band,
#: which is the safe direction.)
WDM6_TIER_FRAME = TieredKernelFrame("wdm6", "WDM6_KMAX", 64, 144)

LIN_TIER_FRAME = TieredKernelFrame("lin", "LIN_KMAX", 128, 60)
# The linear frame is evaluated at the launcher's 128/256/512 tiers.
LIN_REFL_TIER_FRAME = TieredKernelFrame(
    "lin_refl", "REFL_KMAX", 128, 55, reference_tier=256)

#: ``wsm6.cu``'s ``WSM6_KMAX`` sizes the whole per-thread column stack, and
#: ``woof/core/wsm6.py`` compiles it at one of two rungs
#: (``wsm6_constants.WSM6_KERNEL_LEVEL_TIERS`` = 64, 80) rather than at
#: ``nz``.  Until 1.8.9 the flat row above WAS the price for every WSM6
#: configuration, so an ``nz = 72`` run -- the six shipped tornado-LES
#: configs -- was priced at the 64-tier 7,216 B while the kernel it
#: actually launches holds 9,008 B.  1,792 B/thread of backing store the
#: pool never reports, which is the direction that put a run 1,630 MiB
#: over.
#:
#: RE-MEASURED 2026-08-10 on the reference RTX 5090, NVRTC + driver,
#: eighteen bounds from 2 to 80: ``wsm6_column`` holds 7,216 B at the
#: source's ``#ifndef`` default of 64 and 9,008 B at 80 -- 1,792 B over 16
#: levels = exactly 112 B per level -- which this frame reproduces at both
#: rungs.  (Between the rungs the driver's frame sits up to 16 B below the
#: same line; the launcher never compiles there, and the model is a stated
#: ceiling on that band, which is the safe direction.)
#:
#: RE-MEASURED 2026-10-07 after the WRF-exact rewrite (lane/verify-wsm6-mp6)
#: on W1's RTX PRO 6000 at five sm_120 NVRTC builds: 6,448 B at 64 and
#: 8,048 B at 80, 100 B per level, and the module's reflectivity kernel
#: (3,392 B / 4,240 B) is narrower.  The 112 B slope and the 7,216 B row
#: stay as the ceiling over the sm_86/sm_89 readings of the old source.
WSM6_TIER_FRAME = TieredKernelFrame("wsm6", "WSM6_KMAX", 64, 112)

# Additional column families declare their measured compile tiers here.
# WSM3 reads 5664 B at 64 and 7072 B at 80 in both arithmetic arms.
from woof.core.wsm3_constants import wsm3_level_tier as _wsm3_level_tier
CONDITIONAL_COLUMN_TIERED_FRAMES = {
    "wsm3": (TieredKernelFrame("wsm3", "WSM3_KMAX", 64, 88), _wsm3_level_tier),
}


#: Kernel source files that have no standalone local frame because they
#: never compile alone: ``noahmp_driver.cu``, ``noahmp_energy.cu``,
#: ``noahmp_thermal.cu``, ``noahmp_libm_slab.cu`` and ``noahmp_glacier.cu``
#: are FRAGMENTS that borrow ``noahmp_leaves.cu``'s single audited libm
#: transcription (and the glacier its macros), and NVRTC refuses each one
#: handed over by itself.  The units the model launches are the
#: compositions :mod:`woof.core.noahmp_kernel_sources` builds from them,
#: and THOSE are priced -- from a reading of the composed unit on the
#: card's own compile platform
#: (:data:`woof.core.kernel_frame_recordings.NOAHMP_COMPOSED_FRAME_RECORDINGS`,
#: keys ``noahmp_*_composed``; :func:`kernel_local_frame_bytes`).  No
#: selector row names a fragment; a row that did would be a table defect,
#: and :func:`kernel_local_frame_bytes` prices it at the assumed bound
#: (:func:`assumed_bound_modules`) and says so, rather than refusing.
#: The legacy-RRTMG members are the same shape of thing: ``rrtmg_sw.cu``,
#: ``rrtmg_lw_chain.cu`` and the ``rrtmg_lw_taugb*.cu`` band fragments
#: compile only through their own chained translation unit
#: (woof/core/rrtmg_lw.py / rrtmg_sw.py), never as standalone NVRTC
#: modules, and no ``physics_kernel_modules`` selector row names them
#: directly -- a legacy 4/4 radiation request prices its TRANSIENT VRAM
#: through the call-peak envelope (``legacy_radiation_vram_bytes``) and
#: its LOCAL-MEMORY frame through the driver-measured composite rows in
#: :data:`CHAINED_TRANSLATION_UNIT_FRAMES`, which cover these fragments
#: as the translation units they actually launch in.
UNMEASURED_KERNEL_MODULES = frozenset({
    # New Goddard (golden/ports-rad) sat here while its standalone compile
    # failed; it compiles alone now (read 2026-10-17 on test host B's RTX
    # 5090, sm_120, NVRTC 13.4.92: 43,664 B, its source default
    # GDD_MAX_LEVELS tier) and is a recorded row in
    # kernel_frame_recordings.KERNEL_LOCAL_FRAME_RECORDINGS.  A run's own
    # tier (woof/core/radiation_composition.py, get_kernel_int_defines)
    # is priced through its TieredKernelFrame declaration where one exists.
    # Noah mosaic (sf_surface_mosaic = 1) launches only through its own
    # unit, woof/core/noah_mosaic.py _mosaic_module, compiled with
    # --fmad=false beside the loader's -std=c++17.  A default-option
    # standalone compile is a module nothing launches, so its frame would
    # price nothing; the unit that does launch is noah_mosaic_unit in
    # CHAINED_TRANSLATION_UNIT_FRAMES below.
    "noah_mosaic",
    # The fork Thompson unit (thompson_version = "wrf_39_noaa") launches
    # only through woof/core/thompson_fork.py's own NVRTC compile
    # (its MODULE_OPTIONS).  The loader's default-option standalone
    # compile is a module nothing launches; the unit that does is
    # thompson_fork_unit in CHAINED_TRANSLATION_UNIT_FRAMES below.
    "thompson_fork",
    "noahmp_driver", "noahmp_energy", "noahmp_thermal", "noahmp_libm_slab",
    "noahmp_glacier",
    # P3 borrows r_pow/r_exp/r_log from noahmp_leaves.cu rather than
    # carrying a second copy of the tree's one glibc transcription, so
    # p3.cu alone fails NVRTC exactly as the three Noah-MP fragments do.
    # Its composed unit is woof/core/p3_device.p3_source(), and the frame
    # that unit compiles to is measured in CHAINED_TRANSLATION_UNIT_FRAMES
    # below -- so unlike the Noah-MP fragments this one is priced, not a
    # refusal.
    "p3",
    # P3's radar reflectivity operator (acb8faf25) is appended to that
    # composed unit by woof/core/p3_device.p3_reflectivity_source() and
    # calls p3.cu's own __device__ helpers (p3_get_rain_dsd2,
    # p3_access_lookup_table, ...), so p3_zdiag.cu alone fails NVRTC
    # (read 2026-10-07 on test host D's RTX 5070 Ti, sm_120, NVRTC 13.4.92:
    # 'identifier "P3_RD" is undefined').  It is the DA observation
    # operator (woof/da/obsop.py), launched by no forecast scheme, so no
    # mp_physics row selects it and no forecast price reads it.  The unit
    # that launches it, p3_reflectivity_module, read p3k_reflectivity at
    # 0 B local frame (40 registers) on the same card the same day.
    "p3_zdiag",
    "rrtmg_sw", "rrtmg_lw_chain", "rrtmg_lw_chain_coalesced", "rrtmg_lw_zbatched",
    "rrtmg_lw_taugb02_10_11_12", "rrtmg_lw_taugb03_05",
    "rrtmg_lw_taugb06_09", "rrtmg_lw_taugb13_16",
    # BEP+BEM's column kernel needs urban_bem.cuh and the glibc headers
    # woof/core/urban_bem.py composes; priced as urban_bem_composed.
    "urban_bep_bem",
    # The drag unit borrows glibc helpers from the composed source in
    # woof.core.terrain_drag.module_source(); it cannot compile alone.
    "terrain_drag",
    # The coupled fire's fragments (lane/ec-sfire): woof/core/sfire_core.py
    # composes them with their shared headers into the units it compiles,
    # and NVRTC refuses each handed over alone (test host D, 2026-10-06).  No
    # selector row names them; a row that did is priced at the assumed bound.
    "sfire_atm", "sfire_core", "sfire_moisture", "sfire_phys",
    "sfire_spotting", "sfire_wind",
    # golden/wrf-fixes: the WRF 4.8 full-canopy SLUCM (urban_ucm_generation
    # = "wrf_480") sat here until a card read it; it left this set on
    # 2026-10-17 with the row test host B's RTX 5090 read (352 B, sm_120,
    # NVRTC 13.4.92), as the solar geometry unit (rrtmg_legacy_solar) did at
    # the 2.9.1 exact-chain intake.  Breakage the two retirements fix:
    # tests/test_preflight.py::test_the_recorded_local_frames_match_the_driver
    # asserts the set of modules NVRTC refuses alone is exactly this set, and
    # both compiled alone on the 2.9.2 line.
    })

#: Priced units whose committed ``.cu`` compiles only through its own
#: production loader (a shared header spliced in, a composed prefix), so the
#: generic loader cannot read them: ``module -> "package.module:function"``,
#: the function returning ``(cupy module, the source text it compiled)``.
#: The frame regenerator (``tools/vram_reserve_probe.py frames``) and the
#: frame gate (tests/test_preflight.py) both measure these through
#: :func:`load_production_kernel_unit`, so the number recorded is the frame
#: of the image the run launches.  Breakage this prevents: the 2.9.4 hydro
#: rows were priced in KERNEL_MAX_LOCAL_SIZE_BYTES from their loaders while
#: the gate tried the generic loader, failed, and called them unpriced.
PRODUCTION_LOADED_KERNEL_UNITS = {
    "nextgen_cfe": "woof.hydro.cfe:frame_module",
    "nextgen_reservoirs": "woof.hydro.reservoirs:frame_module",
}


def load_production_kernel_unit(module: str):
    """``(cupy module, compiled source)`` for one production-loaded unit."""
    from importlib import import_module
    package, _, function = PRODUCTION_LOADED_KERNEL_UNITS[module].partition(":")
    return getattr(import_module(package), function)()
# face_mass, held_heating, rk_bookkeeping and surface_w (the dycore-host
# speed lane's point-local units, 66af318bc) sat in this set on
# 2026-09-30 under the same "no measurement platform in this lane"
# declaration, because the sweep that entered them ran CPU-only.  The
# same day an RTX 5090 (sm_120) and an RTX 4090 (sm_89) read all four at
# 0 B at NVRTC 13.4.92, which retired the declaration: they compile
# standalone and carry rows in kernel_frame_recordings.SM120_NVRTC_13_4_92.
# mynn_scalar_mix and mynn_dmp_sibling (4a0bb3f69) sat in this set from
# 2026-08-31 under a "no measurement platform in this lane" declaration;
# the sm_86 campaign measured both the same day (0 B, RTX 3080 at NVRTC
# 13.0.48 -- rows in kernel_frame_recordings.SM86_NVRTC_13_0_48), which
# retired the declaration: they compile standalone and are priced like
# any other module.


@dataclass(frozen=True)
class ChainedTranslationUnitFrame:
    """A chained translation unit's driver-measured widest local frame.

    The legacy-RRTMG kernels never load per ``.cu`` file: the LW chain
    concatenates ``rrtmg_lw.cu`` + ``rrtmg_lw_chain.cu`` + the four
    ``rrtmg_lw_taugb*.cu`` band fragments into ONE NVRTC translation unit
    (woof/core/rrtmg_lw.py section 10), and the SW composition compiles
    ``rrtmg_sw.cu`` through its own unit (woof/core/rrtmg_sw.py).  The
    fragments therefore stay in :data:`UNMEASURED_KERNEL_MODULES` --
    selecting one standalone prices it at the assumed bound -- while the
    unit that DOES launch carries the frame the driver measured for it.

    ``covers`` names the unmeasured fragments this measurement subsumes;
    the import-time checks below keep the two tables consistent.
    """

    module: str
    max_local_size_bytes: int
    covers: frozenset[str]


#: Measured 2026-07-27 (cupy 14.0.1 NVRTC on sm_120, RTX 5090) over every
#: kernel of each chained unit -- the record and its drift-bound re-audit
#: live in ``tests/test_rrtmg_lw_cuda.py`` (``LOCAL_FRAME_BOUNDS``); the
#: prose record is docs/rrtmg_legacy_integration.md section 6:
#:
#: * LW unit: ``rlw_rtrn_march`` 2,048 B/thread (exactly its four
#:   128-float per-thread work arrays atrans/atot/bbugas/bbutot at the
#:   fixed RLW_MAXLAY = 128 bound -- NOT nz-specialized), ``rlw_cldprmc``
#:   64 B, every other kernel 0 B.  Machine-wide store ~510 MiB, of which
#:   the 1,024 B default-stack half already sits in CUDA_CONTEXT_BYTES.
#: * SW unit: every kernel 0 B after the spcvmc workspace restructure
#:   (the old ~1.65 GiB hidden lmem reservation became pool-priced
#:   transient VRAM, priced by ``legacy_radiation_vram_bytes``).
#:
#: A re-measure past these values must move this table in the same diff.
CHAINED_TRANSLATION_UNIT_FRAMES: dict[str, ChainedTranslationUnitFrame] = {
    "rrtmg_lw_legacy_chain": ChainedTranslationUnitFrame(
        module="rrtmg_lw_legacy_chain",
        max_local_size_bytes=2048,
        # rrtmg_lw_chain_coalesced joined 2026-09-30: the address-only
        # twin of the chain kernels the batched engine launches, measured
        # with the unit on sm_120 at NVRTC 13.4.92 (cldprmc 64 B, march
        # 2,048 B, the rest 0 B -- the rows its originals already had),
        # so the unit's 2,048 B does not move.  rrtmg_lw_zbatched joined
        # the same day: the one-launch taumol (the sixteen band bodies as
        # device functions) and the shared-memory staged accumulation,
        # 0 B each on sm_120 and sm_89 at NVRTC 13.4.92; its later
        # per-g-point prol and row accumulation read 0 B on both too, and
        # the unit's rows (march 2,048 B, cldprmc 64 B) did not move.
        covers=frozenset({
            "rrtmg_lw_chain", "rrtmg_lw_chain_coalesced", "rrtmg_lw_zbatched",
            "rrtmg_lw_taugb02_10_11_12", "rrtmg_lw_taugb03_05",
            "rrtmg_lw_taugb06_09", "rrtmg_lw_taugb13_16"})),
    "rrtmg_sw_legacy": ChainedTranslationUnitFrame(
        module="rrtmg_sw_legacy",
        max_local_size_bytes=0,
        # Re-read 2026-09-30 after the fused spcvmc layer pass and the
        # coalesced slabs: all 17 kernels 0 B on sm_120 and sm_89 at
        # NVRTC 13.4.92.  Re-read 2026-10-02 after rsw_sfluxzen_body's
        # store on every path (fix/rrtmg-sw-sfluxzen): all 17 kernels 0 B
        # on sm_120 (RTX 5070 Ti) at NVRTC 13.4.92, before and after.
        covers=frozenset({"rrtmg_sw"})),
    # P3 one-category (mp=50).  ``noahmp_leaves.cu`` + ``p3.cu`` compiled as
    # ONE unit by woof/core/p3_device.p3_source(), the same shape the
    # legacy-RRTMG chain and the Noah-MP driver use.
    #
    # Measured 2026-08-29, cupy 14.2.0 NVRTC on sm_120 (RTX 5070 Ti,
    # test host A), over all twelve kernels of the unit: local_size_bytes = 0
    # for every one, both arms.  The widest register user is
    # ``p3k_kloopmain`` at 244 registers (the fused ``p3k_fused_process``
    # at 250), which costs occupancy but spills nothing.  A re-measure past
    # zero must move this row in the same diff.
    "p3_composed": ChainedTranslationUnitFrame(
        module="p3_composed",
        max_local_size_bytes=0,
        covers=frozenset({"p3"})),
    # BEP+BEM (sf_urban_physics = 3): glibc_flt32.cuh + glibc_trig_flt32.cuh
    # + urban_bem.cuh + urban_bep_bem.cu compiled by woof/core/urban_bem.py
    # as one unit.  Both kernels read 2026-10-01 at the two compilers a
    # recast-woof[gpu] install can carry:
    #
    #   sm_89  (RTX 4090), NVRTC 12.9.86: 4,368 B 13.4.92: 4,144 B
    #   sm_120 (RTX 5090), NVRTC 12.9.86: 5,128 B 13.4.92: 2,312 B
    #
    # bep_bem_class_init is 0 B at all four readings.  Price the widest:
    # the default install's NVRTC 12.9.86 leaves 984 B per resident thread
    # uncharged on sm_120 if this row carries only the 4,144 B reading.
    # tests/test_urban_bem_wrf471_parity.py::
    # test_the_composed_bem_unit_compiles_within_the_priced_frame re-reads
    # both kernels through the production loader on whatever card runs it.
    "urban_bem_composed": ChainedTranslationUnitFrame(
        module="urban_bem_composed",
        max_local_size_bytes=5128,
        covers=frozenset({"urban_bep_bem"})),
    # Noah mosaic (sf_surface_mosaic = 1): glibc_flt32.cuh + noah_mosaic.cu,
    # compiled by woof/core/noah_mosaic.py _mosaic_module with
    # MOSAIC_NVRTC_OPTIONS (-std=c++17 --fmad=false).  noah_mosaic_column's
    # local_size_bytes through that loader, read 2026-10-01 at both
    # compilers a recast-woof[gpu] install can carry:
    #
    #   sm_89  (RTX 4090), NVRTC 12.9.86: 240 B   13.4.92: 240 B
    #   sm_120 (RTX 5090), NVRTC 12.9.86: 688 B   13.4.92: 176 B
    #
    # Priced at the widest.  NVRTC 12.9.86 is the default install's compiler
    # (A167), and on sm_120 it spills the column to 255 registers and 688 B;
    # the 240 B this row carried until then was the 13.4.92 reading alone.
    # Still under the 1,024 B default stack, so the row reserves nothing.
    "noah_mosaic_unit": ChainedTranslationUnitFrame(
        module="noah_mosaic_unit",
        max_local_size_bytes=688,
        covers=frozenset({"noah_mosaic"})),
    # The same tile loop composed with the unchanged urban_ucm.cu
    # (sf_urban_physics = 1), noah_mosaic_ucm_column, the same four readings:
    #
    #   sm_89  (RTX 4090), NVRTC 12.9.86: 400 B   13.4.92: 400 B
    #   sm_120 (RTX 5090), NVRTC 12.9.86: 1,040 B 13.4.92: 288 B
    #
    # Priced at the widest, which is past the 1,024 B default stack: on the
    # default install's compiler on sm_120 this unit does reserve backing
    # store, 16 B per resident thread, which the 400 B row left uncharged.
    # tests/test_noah_mosaic_driver.py::
    # test_the_mosaic_units_compile_to_the_frames_they_are_priced_at re-reads
    # both units on whatever card runs it.
    "noah_mosaic_ucm_unit": ChainedTranslationUnitFrame(
        module="noah_mosaic_ucm_unit",
        max_local_size_bytes=1040,
        covers=frozenset()),
    # Production loader, NVRTC 13.4.59 on sm_89 and sm_120: topo_wind
    # and the SASE top helper 0 B, KIM 640 B, GSL 1,184 B on both cards.
    "terrain_drag_composed": ChainedTranslationUnitFrame(
        module="terrain_drag_composed",
        max_local_size_bytes=1184,
        covers=frozenset({"terrain_drag"})),
    # The fork Thompson unit (thompson_version = "wrf_39_noaa", mp_physics
    # = 28, the HRRR and RAP doors' default): thompson_fork.cu +
    # thompson_fork_constants.cuh compiled by woof/core/thompson_fork.py.
    # Its column arrays first sat in the thread's local frame, 48 + 448 B
    # per level of a compiled TF_KMAX (read at fourteen bounds from 8 to
    # 256): 25,136 B at a 50-level column, a 6.5 GiB launch-time
    # reservation on a 188-SM RTX PRO 6000 that nothing priced.  They moved
    # to a column arena sized to the launcher's tile
    # (thompson_fork_column_workspace_bytes), and the unit's frame is now
    # the register spill alone: 784 B at 255 registers, read 2026-10-07 off
    # the driver on that card (sm_120) at NVRTC 12.9.86, the default
    # install's compiler, in the default and the strict build alike.
    # Under the 1,024 B default stack, so it reserves nothing.
    "thompson_fork_unit": ChainedTranslationUnitFrame(
        module="thompson_fork_unit",
        max_local_size_bytes=784,
        covers=frozenset({"thompson_fork"})),
}

_seen_covers: set[str] = set()
for _tu_name, _tu in CHAINED_TRANSLATION_UNIT_FRAMES.items():
    if _tu_name != _tu.module or _tu_name in KERNEL_MAX_LOCAL_SIZE_BYTES:
        raise RuntimeError(
            f"{_tu_name}: chained-unit frame must carry its own key and "
            "must not shadow a standalone-measured module")
    if not _tu.covers <= UNMEASURED_KERNEL_MODULES:
        raise RuntimeError(
            f"{_tu_name}: covers must name only fragments that cannot be "
            "measured standalone (UNMEASURED_KERNEL_MODULES); anything "
            "else is priced from KERNEL_MAX_LOCAL_SIZE_BYTES")
    if _seen_covers & _tu.covers:
        raise RuntimeError(
            f"{_tu_name}: a fragment may be covered by exactly one "
            "chained translation unit")
    _seen_covers |= _tu.covers
del _seen_covers, _tu_name, _tu

#: Kernel modules every integration loads regardless of the physics
#: selectors: dynamics, diffusion, nesting, lateral boundaries, health and
#: output diagnostics.  Their maximum local frame is 768 B (``vert_interp``),
#: below the default stack limit, so this set reserves nothing at all.
CORE_KERNEL_MODULES = frozenset({
    "acoustic", "advection", "bandwidth_glue", "coriolis_map", "diagnostics", "diff6",
    "diffusion", "dycore", "health", "lbc_flow", "lbc_state", "lbc_time", "nest",
    "nest_microphysics", "openbc", "pd_advection", "saxpy", "smag2d",
    "spec_bdy", "tke_budget", "vert_interp", "wrf_cold_start_w"})

#: ``mp_physics`` -> the kernel modules that scheme launches.  Keys are
#: exactly ``woof/config.py``'s accepted set (0, 1, 3, 6, 8, 9, 10, 16, 18,
#: 28, 50), which is priced here ahead of its driver dispatch so a run can
#: never reach ``kernel_local_frame_bytes`` with an unpriced selector.
#: That "exactly" is now MEASURED rather than asserted in prose --
#: tests/test_composition_pricing.py walks every composition the loader
#: accepts and prices each one, which is what caught mp=50 and MYJ
#: shipping accepted-but-unpriceable into the 1.9 assembly.
_MICROPHYSICS_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    30: ("fast_sbm", "microphysics_validation"),
    0: (),
    1: ("kessler", "microphysics_validation"),
    2: ("lin", "microphysics_validation"),
    # Ferrier-Aligo: one translation unit (init and column kernels) plus
    # the shared validator its three canonical slots go through.  ``refl``
    # is not priced: the column kernel writes its own DBZ_col.
    5: ("fer_hires", "microphysics_validation"),
    # WSM3: one column kernel (woof/core/kernels/wsm3.cu) and the shared
    # validator; its REFL_10CM diagnostic loads refl.cu's WSM6 routine and
    # is priced by _REFLECTIVITY_MICROPHYSICS below.
    3: ("wsm3", "microphysics_validation"),
    4: ("wsm5", "microphysics_validation"),
    6: ("wsm6", "microphysics_validation"),
    # WSM6R: its own translation unit (the column kernel and the WSM6
    # effective-radius diagnosis) plus the shared validator; it never loads
    # the WSM6 unit.
    106: ("wsm6r", "microphysics_validation"),
    8: ("thompson", "microphysics_validation"),
    # Milbrandt-Yau.  All seven kernels live in the one ``milbrandt2``
    # translation unit (woof/core/milbrandt2.py:179-187), and the scheme
    # takes the NATIVE validation path -- ``accept_microphysics`` routes
    # everything but mp=18 through ``_validate_native_microphysics``
    # (woof/core/physics.py:1753-1765) -- so it launches the shared
    # validator exactly as Kessler/WSM6/Thompson/Morrison do.  ``refl`` is
    # deliberately NOT priced for this selector and 9 is absent from
    # _REFLECTIVITY_MICROPHYSICS below: mp=9 fills the REFL_10CM slot from
    # its own diagnostics kernel and only stashes the array
    # (woof/core/milbrandt2.py:291-296), so no refl kernel is ever loaded.
    9: ("milbrandt2", "microphysics_validation"),
    10: ("morrison", "microphysics_validation"),
    # WDM6 launches ONE column kernel plus the shared moisture validation
    # pass; its cold half is transcribed inside wdm6.cu rather than shared
    # with wsm6.cu, so mp=16 never loads the WSM6 translation unit.
    16: ("wdm6", "microphysics_validation"),
    # WSM7: ONE translation unit holds the column and the REFL_10CM
    # kernels (woof/core/wsm_family.py compiles it), plus the shared
    # moisture validation pass.
    24: ("wsm7", "microphysics_validation"),
    18: ("nssl2", "nssl2_driver_support", "nssl2_diagnostics",
         "nssl2_fused_gs", "nssl2_nucond", "nssl2_qvexcess"),
    # Thompson aerosol-aware.  ``thompson`` is genuinely launched: mp=28
    # reuses the frozen mp=8 ice/snow/graupel/rain sedimentation and classic
    # graupel-number launchers unchanged.  ``thompson_aerosol_probe`` is
    # excluded on purpose -- it exists only for the device-helper oracle
    # gate and no forecast path loads it, so pricing it here would reserve
    # local memory for a module the run never compiles.
    28: ("thompson", "thompson_aerosol_state", "thompson_aerosol_sat",
         "thompson_aerosol_cold", "thompson_aerosol_warm",
         "thompson_aerosol_sed"),
    # P3 one-category.  ``p3_composed`` is the noahmp_leaves.cu + p3.cu
    # translation unit woof/core/p3_device.p3_source() assembles; it is
    # priced through CHAINED_TRANSLATION_UNIT_FRAMES because p3.cu cannot
    # compile standalone (it borrows the tree's single glibc r_pow/r_exp/
    # r_log rather than carrying a second copy).
    #
    # This row said ONE module until the CUDA port landed, and the reason
    # it gave -- "P3 is a HOST float32 transcription ... there is no
    # woof/core/kernels/p3.cu" -- was true when written and is now false:
    # woof/core/p3.py:apply launches the device kernels and the prognostic
    # state never leaves the card.  The reference/debug arm
    # (run.p3_backend = "reference") still takes the host path and launches
    # nothing, which is why it is a debug selection and not a second
    # priced composition.
    #
    # ``microphysics_validation`` IS launched, on the same footing as
    # Kessler/WSM6/Thompson/Morrison/Milbrandt-Yau: mp=50 has a five-slot
    # canonical row (woof/core/physics.py:603-608) and P3's diagnostics
    # are exactly those canonical scratch arrays, so ``accept_microphysics``
    # takes the native path and launches the shared validator
    # (woof/core/microphysics.py:385-386).
    #
    # 50 stays deliberately ABSENT from _REFLECTIVITY_MICROPHYSICS below,
    # for mp=9's reason: P3 computes REFL_10CM inside its own kernels and
    # ``stash_refl_10cm`` only parks the array on the driver
    # (woof/core/refl.py:651-663), so no ``refl`` kernel is ever loaded.
    # That absence is RECORDED, with its authority and its measured cost,
    # in _SELF_REFLECTIVITY_MICROPHYSICS below -- and it is now a decision
    # the rail forces rather than one a reader has to notice.
    50: ("p3_composed", "microphysics_validation"),
}

# Named schemes compile their column kernel through their own runtime
# module (woof/core/tempo_runtime.py, state in global work arrays, no large
# local frame) and take the native validation path.
from woof.microphysics_schemes import NAMED_SCHEMES as _NAMED_MP_SCHEMES_K
from woof.microphysics_schemes import SCHEMES_BY_ID as _NAMED_MP_ROWS_BY_ID
_MICROPHYSICS_KERNEL_MODULES.update({
    _s.mp_id: ("microphysics_validation",) for _s in _NAMED_MP_SCHEMES_K.values()})

#: ``mp_physics`` values with a REFL_10CM path in ``woof/core/refl.py``.
#: The ``refl`` module is priced only when a history frame can come due
#: during the run (:func:`refl_diagnostic_reachable`), because the kernel is
#: launched from the ``refl_10cm_due`` branch of the microphysics drivers and
#: from nowhere else.
#: 28 is included: mp=28 routes REFL_10CM through the SAME Thompson
#: reflectivity kernel as mp=8 (calc_refl10cm takes no droplet number and
#: never re-reads rc), so the ``refl`` module is loaded on exactly the same
#: cadence -- which is not a judgement made here any more.  This set and
#: :data:`_SELF_REFLECTIVITY_MICROPHYSICS` are held equal to the operator's
#: own two tables at import by
#: :func:`_hold_reflectivity_rail_equal_to_the_operator`, so 28 is here
#: because ``woof/core/refl.py`` dispatches it, and a rail that stopped
#: agreeing with the operator -- in either direction, over-priced or under
#: -- is refused rather than reasoned about.
#:
#: WHAT THIS SET DECIDES, which is narrower than its name: not "has a
#: reflectivity diagnostic" but "reserves the SHARED reflectivity
#: translation unit's per-thread frame".  A moist scheme that fills
#: REFL_10CM from its OWN kernels and hands the finished array to
#: ``stash_refl_10cm`` loads no ``refl`` module and belongs in
#: :data:`_SELF_REFLECTIVITY_MICROPHYSICS` instead.  Every accepted moist
#: selector must appear in exactly one of the two, and
#: :func:`domain_kernel_modules` refuses one that appears in neither.
#: 18 IS NOT HERE, and was, until audit R-052.  NSSL two-moment carries its
#: own S-band diagnostic, radardd02, ported at woof.core.nssl2_diagnostics
#: and already priced by this module's own microphysics row for mp=18; it
#: reaches ``compute_and_stash_refl_10cm`` never -- woof/core/microphysics
#: .py routes 9, 18 and 50 to their adapters and only 1, 6, 8, 10, 16 and
#: 28 to the shared operator -- so an NSSL domain reserved refl.cu's
#: per-thread frame for a translation unit the run cannot launch, and a
#: reservation that is too large refuses a run that would have fit.
_REFLECTIVITY_MICROPHYSICS = frozenset({1, 2, 3, 6, 8, 10, 16, 28, 106})

#: The DELIBERATE absences from :data:`_REFLECTIVITY_MICROPHYSICS`, each
#: with the reason it is a decision and not an omission.  The pattern is
#: ``microphysics_transition``'s named-refusal tables: a scheme that is out
#: of an admission set says so by name, so the next reader finds a ruling
#: rather than a gap.  Adding a moist selector to
#: :data:`_MICROPHYSICS_KERNEL_MODULES` without landing it in one set or
#: the other now fails closed at ``woof check``.
_SELF_REFLECTIVITY_MICROPHYSICS: dict[int, str] = {
    30: ("FAST_SBM writes no REFL_10CM: its polar-scattering diagnostic is "
         "not ported and woof.core.fast_sbm_runtime.apply refuses a due "
         "frame, so it neither loads refl.cu nor fills stash_refl_10cm and "
         "the history lane (physics_inventory.REFL_10CM_MICROPHYSICS) never "
         "asks it for one"),
    5: (
        "Ferrier-Aligo computes its reflectivity column inside "
        "EGCP01COLUMN (DBZ_col from the scheme's own rain and "
        "precipitating-ice moments, module_mp_fer_hires.F:2232-2251, "
        ":2339-2345, :2494-2496).  kernels/fer_hires.cu writes that column "
        "on output steps and woof/core/fer_hires.py hands it to "
        "stash_refl_10cm, so no refl kernel is loaded.  The shared "
        "operator could not stand in: its Rayleigh sums read qs and qg, "
        "and the fer_mp_hires package carries one total-ice mass and "
        "neither."
    ),
    4: "The column scheme row launches its own radar kernel for stash_refl_10cm in wsm5.cu; no shared refl.cu is loaded.",
    18: (
        "NSSL two-moment fills the REFL_10CM slot from its own radardd02 "
        "diagnostic (woof.core.nssl2_diagnostics, priced by this module's "
        "mp=18 kernel row as nssl2_diagnostics) and hands the finished "
        "array to stash_refl_10cm, so no refl kernel is loaded.  The cost "
        "of pricing refl.cu anyway is the one measured for mp=50 in the "
        "row below -- refl compiles to 3,600 B per thread at nz=50, the "
        "widest frame either scheme has -- charged to a domain that never "
        "launches it."
    ),
    24: (
        "WSM7's refl10cm_wsm7 kernel lives in its own wsm7 translation "
        "unit (woof/core/kernels/wsm7.cu), priced by this module's "
        "mp=24 kernel row, and uses stash_refl_10cm for consume-once output; "
        "refl.cu is never loaded for it."
    ),
    9: (
        "Milbrandt-Yau fills the REFL_10CM slot inside its own "
        "milbrandt2 diagnostics kernel and hands the finished array to "
        "stash_refl_10cm (woof/core/milbrandt2.py:291-296), so no refl "
        "kernel is loaded."
    ),
    50: (
        "P3 one-category computes REFL_10CM inside p3_main's own final-"
        "checks-and-diagnostics loop -- ze_rain from the rain gamma "
        "moment, ze_ice from ice lookup-table column 9, summed to dBZ "
        "(phys/module_mp_p3.F:4722-4895, transcribed at "
        "woof/core/p3.py:1567-1626) -- and returns it as an OUTPUT of "
        "mp_p3_wrapper_wrf (:690-932).  Both arms hand that finished "
        "array straight to stash_refl_10cm (woof/core/p3.py:1831-1835 "
        "reference, :1982-1984 device), never to "
        "compute_and_stash_refl_10cm, so no refl kernel is ever loaded.  "
        "It could not be: refl.cu transcribes the calc_refl10cm family, "
        "whose Rayleigh sums read qs and qg, and P3 carries ONE ice "
        "category with a rime pair (qir/qib) and has neither -- its "
        "accumulator row is five slots with no graupel "
        "(woof/core/physics_inventory.py:163-169).  MEASURED cost of "
        "pricing it anyway: refl compiles to 3,600 B per thread at "
        "nz=50 and is the WIDEST frame an mp=50 configuration has, so "
        "admitting 50 moves the local-memory reservation from 0 to "
        "672,645,120 B (0.63 GiB) on a 200x200x50 domain and refuses "
        "runs that fit, for a kernel the run never launches."
    ),
}


# Named schemes computing their own REFL_10CM (capability row): the adapter
# stashes the scheme's field, so no refl kernel is loaded.
from woof.microphysics_schemes import NAMED_SCHEMES as _NAMED_MP_SCHEMES
_SELF_REFLECTIVITY_MICROPHYSICS.update({
    _s.mp_id: (f"{_s.label} computes REFL_10CM in its own column call and "
               "its adapter hands the finished array to stash_refl_10cm")
    for _s in _NAMED_MP_SCHEMES.values() if _s.native_reflectivity})

def _hold_reflectivity_rail_equal_to_the_operator() -> None:
    """HELD EQUAL, AT IMPORT, to the reflectivity operator's own tables.

    These two sets answer one question -- "does this scheme reach
    ``woof.core.refl``'s shared translation unit?" -- and
    ``woof/core/refl.py`` answers exactly the same question in
    ``REFL_10CM_INPUT_SPECIES`` (the schemes it dispatches) and
    ``SCHEME_NATIVE_REFL_10CM`` (the schemes that compute their own dBZ,
    with the reason each is out).  Two hand-kept copies of one decision is
    how mp=18 came to be priced for a kernel it cannot launch, so the copy
    that prices is refused at import if it stops agreeing with the copy
    that dispatches.  The REASONS stay separate on purpose: the operator
    says why a scheme has no generic call, this module says what pricing
    it anyway costs, and neither sentence is the other's.
    """

    from woof.core.refl import (REFL_10CM_INPUT_SPECIES,
                                 SCHEME_NATIVE_REFL_10CM)

    problems: list[str] = []
    from woof.core.column_schemes import COLUMN_SCHEMES
    if set(REFL_10CM_INPUT_SPECIES) - set(COLUMN_SCHEMES) != set(_REFLECTIVITY_MICROPHYSICS):
        problems.append(
            "the schemes woof.core.refl.compute_refl_10cm dispatches "
            f"({sorted(REFL_10CM_INPUT_SPECIES)}) are not the schemes this "
            "module prices refl.cu for "
            f"({sorted(_REFLECTIVITY_MICROPHYSICS)})")
    if set(SCHEME_NATIVE_REFL_10CM) | set(COLUMN_SCHEMES) != set(_SELF_REFLECTIVITY_MICROPHYSICS):
        problems.append(
            "the schemes woof.core.refl names as computing their own "
            f"REFL_10CM ({sorted(SCHEME_NATIVE_REFL_10CM)}) are not the "
            "schemes this module excuses from the refl rail "
            f"({sorted(_SELF_REFLECTIVITY_MICROPHYSICS)})")
    if problems:
        raise RuntimeError(
            "woof/core/preflight.py's reflectivity-rail sets disagree with "
            "woof/core/refl.py: " + "; ".join(problems)
            + ".  A scheme belongs to exactly one of them, and the operator "
            "is the authority for which.")


_hold_reflectivity_rail_equal_to_the_operator()

_CUMULUS_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    0: (), 1: ("kf", "kf_validation"),
    # One translation unit carries all of GFDRV (deep, shallow, driver).
    3: ("gf",),
    6: ("tiedtke",),
    # And one carries all twenty-one New Tiedtke stages. The table is
    # FAIL-CLOSED -- domain_kernel_modules raises on a selector with no
    # entry -- so 16 announced itself here rather than going quiet, which
    # is why this was the one item of the seven that could not be
    # forgotten.
    16: ("ntiedtke",),
    # Grell 3D: one translation unit carries G3DRV/CUP_enss_3d
    # (g3_column) and conv_grell_spread3d (g3_spread).
    5: ("g3",),
    # New SAS: one unit carries cu_nsas, nsas2d and nscv2d.
    96: ("nsas",)}
_PBL_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    0: (), 1: ("ysu", "ysu_validation"),
    # MYJ (Mellor-Yamada-Janjic).  ONE module: woof/core/myjpbl.py's only
    # kernel load is ``get_kernel("myjpbl", "myjpbl_column")`` (:111), one
    # thread per column, and the scheme does its own implicit vertical
    # diffusion inside that kernel rather than handing tendencies to a
    # separate diffusion launcher (woof/core/physics.py:2574-2582).
    #
    # No validation module, and that is a measured absence rather than an
    # oversight: YSU and Shin-Hong each launch a batched device validator
    # (``ysu_validation``, ``shinhong_validation``), while MYJ's
    # ``validate_myj_pbl_outputs`` is a host reduction over the returned
    # fields (woof/core/physics.py:2625) and compiles nothing.
    #
    # The frame this selects is the widest of any PBL in the tree --
    # KERNEL_MAX_LOCAL_SIZE_BYTES["myjpbl"] = 9,232 B at the compiled
    # MYJ_KMAX = 128 tier, ~2.00 GiB of reservation on the reference card
    # by the model at the head of docs/kernel_local_memory_bounds.md.  It
    # is a flat row and not a tiered one because woof/core/myjpbl.py
    # compiles at the source's fixed bound rather than at nz, so it is the
    # price of selecting MYJ at all; the row's own comment records that
    # specializing it is a real saving and a separate change.
    MYJ_PBL_SCHEME: ("myjpbl",),
    # MYNN-EDMF.  THREE modules (row widened by the 2026-08-31 scheme-5
    # enumeration fix; it said one module and hid two): the launcher stack
    # woof/core/mynn_pbl_gpu.py loads ``mynn_pbl`` always, and under
    # bl_mynn_mixscalars=1 dispatches the sibling DMP unit
    # (kernels/mynn_dmp_sibling.cu, :1329) plus the qn flux/mix unit
    # (kernels/mynn_scalar_mix.cu via woof/core/mynn_scalar_mix_gpu.py,
    # both added by 4a0bb3f69).  The two conditional modules are priced
    # FLAT rather than dispatched on the option (the ra_physics=4
    # pattern) because over-pricing is the safe direction for a rail gate
    # and both measured 0 B on SM86_NVRTC_13_0_48 (803ab8e58), so the
    # flat row refuses nothing -- while a module this table cannot see is
    # exactly where a future non-zero frame would hide.
    5: ("mynn_pbl", "mynn_dmp_sibling", "mynn_scalar_mix"),
    # The UW moist-turbulence PBL: ONE module.  woof/core/uwpbl.py's only
    # kernel load is ``load_module("uwpbl")`` (the loader assembles the
    # glibc_flt64/uwpbl_* headers in front of uwpbl.cu), one thread per
    # column, and like MYJ it performs its own implicit diffusion inside
    # that kernel.  No validation module: _run_uwpbl's finiteness check is
    # a host reduction and compiles nothing.
    UW_PBL_SCHEME: ("uwpbl",),
    901: ("satmedmf",),
    # Shin-Hong launches its column kernel plus its own batched output
    # validator, the YSU pair's shape (woof/core/shinhong.py).
    11: ("shinhong", "shinhong_validation"),
    SASE_PBL_SCHEME: ("sase",)}
_SURFACE_LAYER_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    0: (), 1: ("sfclay",),
    # Eta similarity, MYJ's own surface layer.  woof/core/myjsfc.py loads
    # exactly ``get_kernel("myjsfc", "myjsfc_column")`` (:102) and nothing
    # else; the module measures 0 B because the scheme is scalar per
    # column and holds no per-thread stack.  It is a separate row from the
    # PBL's on purpose: sf_sfclay_physics is its own selector, and
    # woof.config.validate_myj_pairing is what ties the two values
    # together -- this table prices whichever the resolved config names.
    MYJ_SFCLAY_SCHEME: ("myjsfc",),
    5: ("mynn_surface",), 91: ("sfclay",)}
_LAND_SURFACE_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    0: (),
    2: ("noah",),
    3: ("ruc",),
    4: NOAHMP_PRICING_MODULES,
}
#: ``ra_physics = 4`` is two implementations behind one selector value;
#: the row here is the modern RTE+RRTMGP set and
#: :func:`_radiation_44_kernel_modules` dispatches on
#: ``RunConfig.ra_rrtmg_variant`` -- the legacy variant launches the
#: chained translation units + the device McICA twin instead, and an
#: unknown variant refuses rather than pricing either row (fail-closed).
_RADIATION_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    0: (),
    4: ("rrtmgp_cloud", "rrtmgp_gas", "rrtmgp_mcica", "rrtmgp_rte"),
    90: (),
}

#: What a legacy 4/4 selection launches: the two chained translation
#: units (:data:`CHAINED_TRANSLATION_UNIT_FRAMES`), the standalone
#: ``rrtmg_mcica_wrf.cu`` device McICA twin, the adapter's device glue
#: ``rrtmg_legacy_adapter.cu`` and the device wrapper prep
#: ``rrtmg_legacy_prep.cu`` (all measured 0 B like every other standalone
#: row).
_RRTMG_LEGACY_KERNEL_MODULES: tuple[str, ...] = (
    "rrtmg_mcica_wrf", "rrtmg_lw_legacy_chain", "rrtmg_sw_legacy",
    "rrtmg_legacy_adapter", "rrtmg_legacy_prep", "rrtmg_legacy_solar")


def _radiation_44_kernel_modules(run) -> tuple[str, ...]:
    """Kernel modules a resolved 4/4 radiation request launches."""
    from woof.physics_compat import (
        RRTMG_VARIANT_LEGACY, RRTMG_VARIANT_RTE_RRTMGP, rrtmg_variant)
    variant = rrtmg_variant(run)
    if variant == RRTMG_VARIANT_RTE_RRTMGP:
        return _RADIATION_KERNEL_MODULES[4]
    if variant == RRTMG_VARIANT_LEGACY:
        return _RRTMG_LEGACY_KERNEL_MODULES
    raise ValueError(
        f"no kernel-module row for ra_physics=4 with "
        f"ra_rrtmg_variant={variant!r}; add one to "
        "woof/core/preflight.py before this configuration can be "
        "priced or gated.")

_SELECTOR_TABLES = (
    ("mp_physics", _MICROPHYSICS_KERNEL_MODULES),
    ("cu_physics", _CUMULUS_KERNEL_MODULES),
    ("bl_pbl_physics", _PBL_KERNEL_MODULES),
    ("sf_sfclay_physics", _SURFACE_LAYER_KERNEL_MODULES),
    ("sf_surface_physics", _LAND_SURFACE_KERNEL_MODULES),
    ("ra_physics", _RADIATION_KERNEL_MODULES),
)

#: ``sf_urban_physics`` -> the kernel modules the urban model launches:
#: urban_ucm.cu (woof/core/urban_ucm.py), urban_bep.cu with the surface
#: couple urban_bep_couple.cu (woof/core/urban_bep.py, urban_bep_couple.py),
#: and BEP+BEM's composed unit ``urban_bem_composed`` (woof/core/
#: urban_bem.py) with the same couple.  Under MYJ, options 2 and 3 replace
#: the MYJ PBL call with MYJURB (woof/core/physics.py), whose
#: myjurb.cu frame is 10,256 B on sm_89 at NVRTC 13.4.92: on an RTX 4090
#: that is a 1.7 GiB launch-time reservation the fit gate did not charge
#: before this row existed.
_URBAN_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    0: (),
    1: ("urban_ucm",),
    2: ("urban_bep", "urban_bep_couple"),
    3: ("urban_bem_composed", "urban_bep_couple"),
}


def _urban_kernel_modules(run, grid_id: int) -> tuple[str, ...]:
    urban = int(getattr(run, "sf_urban_physics", 0) or 0)
    if urban not in _URBAN_KERNEL_MODULES:
        raise ValueError(
            f"no kernel-module row for sf_urban_physics={urban} on "
            f"d{grid_id:02d}; add one to woof/core/preflight.py before "
            "this configuration can be priced or gated.")
    modules = _URBAN_KERNEL_MODULES[urban]
    if urban in (2, 3) and int(run.bl_pbl_physics) == MYJ_PBL_SCHEME:
        modules += ("myjurb",)
    return modules


def refl_diagnostic_reachable(exp: ExperimentConfig) -> bool:
    """Can a history frame carrying REFL_10CM come due inside the run?

    The reflectivity kernels are launched from the microphysics drivers'
    ``refl_10cm_due`` branch, which follows the history cadence.  A run
    shorter than every domain's history interval never reaches one -- both
    traced forecasts behind this model wrote their t=0 frames and never
    launched a ``refl`` kernel.  Anything else prices it.
    """
    intervals = [float(getattr(dc, "history_interval_s", 0.0) or 0.0)
                 for dc in exp.domains]
    if not intervals or min(intervals) <= 0.0:
        return True
    return float(exp.run_seconds) >= min(intervals)


def domain_kernel_modules(dc: DomainConfig, *,
                          prices_refl: bool) -> frozenset[str]:
    """Kernel modules ONE domain's selectors can launch (no core modules).

    FAIL-CLOSED: a selector value with no table entry raises rather than
    quietly pricing zero.  A new scheme that reaches production without a row
    here stops ``woof check``; it does not slip past it.
    """
    modules: set[str] = set()
    for selector, table in _SELECTOR_TABLES:
        value = int(getattr(dc.run, selector))
        if value not in table:
            raise ValueError(
                f"no kernel-module row for {selector}={value} on "
                f"d{dc.grid_id:02d}; add one to woof/core/preflight.py "
                "before this configuration can be priced or gated.")
        if selector == "ra_physics" and 4 in radiation_scheme_ids(dc.run):
            # One selector value, two implementations: dispatch on the
            # trajectory-bound ra_rrtmg_variant (fail-closed inside).
            modules.update(_radiation_44_kernel_modules(dc.run))
        else:
            modules.update(table[value])
    if int(dc.run.bl_pbl_physics) == 1:
        from woof.ysu_generations import ysu_kernel_selection
        from woof.wrf_exact import ENABLED
        modules.discard("ysu")
        modules.add(ysu_kernel_selection(dc.run.ysu_version, strict=ENABLED)[0])
    modules.update(_urban_kernel_modules(dc.run, dc.grid_id))
    if int(getattr(dc.run, "diff_opt", 2)) == 1:
        modules.add("diff_opt1")
    if int(dc.run.sf_surface_physics) == 2:
        # Native Noah starts initialize SH2O through its own module.  It
        # remains a possible cold-start load under the mosaic forecast.
        modules.add("noah_init")
    if (int(getattr(dc.run, "topo_wind", 0) or 0)
            or int(getattr(dc.run, "gwd_opt", 0) or 0)):
        modules.add("terrain_drag_composed")
    if int(getattr(dc.run, "zadvect_implicit", 0) or 0) > 0:
        modules.add("ieva")                    # A158, woof/core/ieva.py
    # WRF swint_opt = 1 (woof/core/swint.py) and aer_opt = 3 on the
    # legacy RRTMG shortwave (woof/core/rrtmg_aerosol_optics.py).
    if int(getattr(dc.run, "swint_opt", 0) or 0) == 1:
        modules.add("swint")
    if int(getattr(dc.run, "alb_sol", 0) or 0) == 1:
        modules.add("solar_albedo")
    # Dudhia shortwave (ra_sw_physics = 1) launches its own column kernel
    # (woof/core/dudhia.py, lane/verify-dudhia-sw).
    if int(getattr(dc.run, "ra_sw_physics", 0) or 0) == 1:
        modules.add("dudhia_sw")
    if int(getattr(dc.run, "aer_opt", 0) or 0) == 1:
        modules.add("rrtmg_aer1")
    if int(getattr(dc.run, "aer_opt", 0) or 0) == 3:
        modules.add("rrtmg_aer3")
    if int(getattr(dc.run,"aer_opt",0) or 0) == 2:
        modules.add("aer_opt2")
    from woof.core.hail import enabled as hail_enabled
    if hail_enabled(dc.run):
        modules.add("hail")
    if (float(getattr(dc.run, "prec_acc_dt", 0)) > 0
            or float(getattr(dc.run, "prec_acc_dt1", 0)) > 0):
        modules.add("output_diagnostics")
    if int(getattr(dc.run, "output_diagnostics", 0)):
        modules.add("climate_diagnostics")
    if int(getattr(dc.run, "gsd_diagnostics", 0)):
        modules.update(("gsd_diagnostics", "gsd_helicity"))
    from woof.core.gsd_initial_reflectivity import enabled as gsd_initial_enabled
    if gsd_initial_enabled(dc.run):
        modules.add("gsd_initial_reflectivity")
    if int(getattr(dc.run, "sf_lake_physics", 0)) == 1:
        modules.add("lake")
    if int(getattr(dc.run, "sf_ocean_physics", 0)) == 1:
        modules.add("oml")
    mp_physics = int(dc.run.mp_physics)
    from woof.thompson_generations import CLASSIC_GENERATIONS
    classic_generation = (CLASSIC_GENERATIONS.get(getattr(dc.run,"thompson_version","wrf_461"))
                          if mp_physics == 8 else None)
    generated_thompson = classic_generation is not None and classic_generation.backend == "column"
    if generated_thompson:
        modules.discard("thompson")
        modules.update(("thompson_column","thompson_aerosol_state"))
    if (mp_physics and mp_physics not in _REFLECTIVITY_MICROPHYSICS
            and mp_physics not in _SELF_REFLECTIVITY_MICROPHYSICS):
        # FAIL-CLOSED on the reflectivity rail too, not just on the scheme
        # row above.  A moist selector missing from BOTH sets prices no
        # reflectivity frame at all, and silence here is the dangerous
        # direction: the run passes ``woof check`` against a budget that
        # never counted refl.cu's per-thread frame and then breaches it at
        # the first history step.  Being out is a legitimate answer -- it
        # is what mp=9 and mp=50 are -- but it has to be WRITTEN.
        raise ValueError(
            f"no reflectivity-rail decision for mp_physics={mp_physics} on "
            f"d{dc.grid_id:02d}.  If the scheme's REFL_10CM comes from "
            "woof.core.refl.compute_refl_10cm it loads refl.cu (or "
            "wdm6_refl.cu) and must join _REFLECTIVITY_MICROPHYSICS; if it "
            "fills the slot from its own kernels and only calls "
            "stash_refl_10cm it loads neither, and must say so in "
            "_SELF_REFLECTIVITY_MICROPHYSICS with the reason.  Leaving it "
            "out of both under-prices the local-memory reservation.")
    if (WRF_EXACT and mp_physics == 28
            and getattr(dc.run, "thompson_version", "wrf_461")
            == "wrf_39_noaa"):
        # Under GPUWM_WRF_EXACT=1 the fork generation's whole driver runs in
        # its own column unit (woof.core.microphysics_aerosol.
        # _apply_thompson_fork_call); a default run keeps the aerosol units.
        modules.add("thompson_fork_unit")
    if int(getattr(dc.run, "sf_surface_mosaic", 0)) == 1:
        # lsm_mosaic replaces lsm (module_surface_driver.F:2662-2778): the
        # mosaic unit runs every Noah column.  noah.cu is still loaded: the
        # driver calls seaice_noah after lsm_mosaic too (:2917-2929), and
        # noah_seaice_column lives in noah.cu, so its frame stays priced.
        modules.add("noah_mosaic_ucm_unit" if int(getattr(dc.run, "sf_urban_physics", 0)) == 1
                    else "noah_mosaic_unit")
    if prices_refl and mp_physics in _REFLECTIVITY_MICROPHYSICS and not generated_thompson:
        # mp=16 launches its reflectivity from its OWN translation unit, so
        # a WDM6 domain must never reserve refl.cu's wider frame and a
        # non-WDM6 domain must never reserve wdm6_refl.cu's.
        modules.add({2:"lin_refl",16:"wdm6_refl"}.get(mp_physics,"refl"))
    if getattr(dc.run, "chem_sets", ""):
        # A chem domain's own units (woof/core/chem_context.py:
        # chem_kernel_modules), fail-closed on a process that does not
        # name the kernels its step launches.
        from woof.chem_table import load as load_chem_table
        from woof.core.chem_context import chem_kernel_modules

        modules.update(chem_kernel_modules(load_chem_table(dc.run),
                                           dc.run.chem_adv_opt))
        if getattr(dc.run, "aer_ra_feedback", 0):
            modules.update(("chem_radiation", "chem_rrtmgp_aerosol"))
    return frozenset(modules)


def physics_kernel_modules(exp: ExperimentConfig) -> frozenset[str]:
    """Every kernel module the experiment's selectors can launch."""
    prices_refl = refl_diagnostic_reachable(exp)
    modules = set(CORE_KERNEL_MODULES)
    for dc in exp.domains:
        modules |= domain_kernel_modules(dc, prices_refl=prices_refl)
    return frozenset(modules)


def assumed_bound_modules(modules: Iterable[str]) -> frozenset[str]:
    """The launched modules no recording in this tree covers.

    Two shapes reach here: a fragment that never compiles alone
    (:data:`UNMEASURED_KERNEL_MODULES`, reached when a selector row names
    the fragment instead of the composed unit that launches it), and a
    module with no row in any frame table at all.  Neither is refused.
    Each is priced at
    :func:`woof.core.kernel_frame_recordings.assumed_frame_bound` -- the
    widest frame any module has been recorded at, so the reservation is
    never short -- and :func:`non_pool_basis` says the number is an
    assumed bound and names the modules it covers.
    """
    modules = set(modules)
    return frozenset(
        (modules & UNMEASURED_KERNEL_MODULES)
        | (modules - set(KERNEL_MAX_LOCAL_SIZE_BYTES)
           - set(CHAINED_TRANSLATION_UNIT_FRAMES)
           - set(NOAHMP_PRICING_MODULES)))


def kernel_local_frame_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> dict[str, int]:
    """Widest per-thread local frame each launched module compiles to.

    A module launched by several domains is priced at the deepest of them:
    the specialized frame is monotone in the level count, and the driver's
    reservation is a maximum over everything the process ever launches.

    Every module that compiles alone is priced from the cross-platform
    ceiling and needs no ``profile``.  The Noah-MP composed units are
    priced from the reading taken on the card's own compile platform when
    the profile carries a recorded one (``None`` reads this machine's
    card when no card was declared), and from the ceiling over the
    recorded Noah-MP platforms otherwise, with the basis stated beside
    the number by :func:`non_pool_basis` -- see
    :mod:`woof.core.noahmp_frame_provenance`.  A module with no reading
    anywhere is priced at the assumed bound
    (:func:`assumed_bound_modules`), never refused.
    """
    modules = physics_kernel_modules(exp)
    assumed = assumed_bound_modules(modules)
    assumed_bound = _kernel_frame_recordings.assumed_frame_bound()

    from woof.core.noahmp_frame_provenance import noahmp_frames as _noahmp

    noahmp_frames = _noahmp(modules, _noahmp_pricing_profile(exp, profile))
    prices_refl = refl_diagnostic_reachable(exp)
    frames = {module: (
                  assumed_bound if module in assumed else
                  noahmp_frames[module] if module in noahmp_frames else
                  CHAINED_TRANSLATION_UNIT_FRAMES[module].max_local_size_bytes
                  if module in CHAINED_TRANSLATION_UNIT_FRAMES
                  else KERNEL_MAX_LOCAL_SIZE_BYTES[module])
              for module in modules
              if module not in LEVEL_SPECIALIZED_KERNEL_FRAMES}
    for dc in exp.domains:
        levels = int(dc.run.nz)
        for module in domain_kernel_modules(dc, prices_refl=prices_refl):
            specialized = LEVEL_SPECIALIZED_KERNEL_FRAMES.get(module)
            if specialized is None:
                continue
            frame = specialized.frame_bytes(levels)
            if frame > frames.get(module, -1):
                frames[module] = frame
    # ``acoustic`` is in CORE_KERNEL_MODULES, so it is priced above at its
    # shipped-tier row; a domain deeper than the shipped tier raises that
    # row to its own tier.  The launcher owns the ladder (it is the thing
    # that compiles the tier) and imports no CuPy at module scope, so this
    # still prices a configuration on a host with no device.
    from woof.core.acoustic import wphi_level_tier

    for dc in exp.domains:
        tier = wphi_level_tier(int(dc.run.nz))
        frame = ACOUSTIC_TIER_FRAME.frame_bytes(tier)
        if frame > frames.get(ACOUSTIC_TIER_FRAME.module, -1):
            frames[ACOUSTIC_TIER_FRAME.module] = frame
    for dc in exp.domains:
        selected = domain_kernel_modules(dc, prices_refl=prices_refl)
        for module, (frame_row, select_tier) in CONDITIONAL_COLUMN_TIERED_FRAMES.items():
            if module not in selected:
                continue
            frame = frame_row.frame_bytes(select_tier(int(dc.run.nz)))
            if frame > frames.get(module, -1):
                frames[module] = frame
    # ``wdm6`` and ``wsm6`` are the CONDITIONAL tiered modules, so
    # unlike acoustic it is priced PER DOMAIN THAT SELECTS IT.  Both halves
    # of that matter: a 40-level WDM6 child beside a 100-level WSM6 parent
    # must be priced at WDM6's own 64 tier, not raised to a tier WDM6 has no
    # kernel for -- and the parent's 100 levels must not reach WDM6's
    # level-bound refusal at all, because the parent never launches it.  The
    # ladder lives in the CuPy-free constants leaf, so this still prices on
    # a host with no device.
    from woof.core.wdm6_constants import wdm6_level_tier

    for dc in exp.domains:
        if WDM6_TIER_FRAME.module not in domain_kernel_modules(
                dc, prices_refl=prices_refl):
            continue
        frame = WDM6_TIER_FRAME.frame_bytes(wdm6_level_tier(int(dc.run.nz)))
        if frame > frames.get(WDM6_TIER_FRAME.module, -1):
            frames[WDM6_TIER_FRAME.module] = frame
    for dc in exp.domains:
        modules = domain_kernel_modules(dc, prices_refl=prices_refl)
        for declaration in (LIN_TIER_FRAME, LIN_REFL_TIER_FRAME):
            if declaration.module in modules:
                tier = next(t for t in (128, 256, 512) if t >= dc.run.nz)
                frames[declaration.module] = max(
                    frames.get(declaration.module, 0), declaration.frame_bytes(tier))
    # ``wsm6`` prices the same way, and the same two halves matter: a
    # 40-level WSM6 child beside a 100-level Morrison parent takes WSM6's
    # own 64 tier rather than a tier WSM6 has no kernel for, and the
    # parent's 100 levels never reach WSM6's level-bound refusal on behalf
    # of a domain that does not launch it.
    from woof.core.wsm6_constants import wsm6_level_tier

    for dc in exp.domains:
        if WSM6_TIER_FRAME.module not in domain_kernel_modules(
                dc, prices_refl=prices_refl):
            continue
        frame = WSM6_TIER_FRAME.frame_bytes(wsm6_level_tier(int(dc.run.nz)))
        if frame > frames.get(WSM6_TIER_FRAME.module, -1):
            frames[WSM6_TIER_FRAME.module] = frame
    # Each WSM-family row prices its own launched tier. A deeper domain
    # running another scheme cannot raise this module's reservation.
    from woof.core.wsm_family import (
        family_selectors, local_frame_bytes as family_frame_bytes, row as family_row)

    for dc in exp.domains:
        mp = int(dc.run.mp_physics)
        if mp not in family_selectors():
            continue
        unit = family_row(mp).unit
        frame = family_frame_bytes(mp, int(dc.run.nz))
        frames[unit] = max(frames.get(unit, -1), frame)
    # ``ieva`` prices per domain that turns zadvect_implicit on, at that
    # domain's own tier, exactly as wdm6/wsm6 do.
    from woof.core.ieva_constants import level_tier as ieva_level_tier

    for dc in exp.domains:
        if IEVA_TIER_FRAME.module not in domain_kernel_modules(
                dc, prices_refl=prices_refl):
            continue
        frame = IEVA_TIER_FRAME.frame_bytes(ieva_level_tier(int(dc.run.nz)))
        if frame > frames.get(IEVA_TIER_FRAME.module, -1):
            frames[IEVA_TIER_FRAME.module] = frame
    for module, shallow_frame in THOMPSON_SHALLOW_KERNEL_FRAMES.items():
        selected_levels = [int(dc.run.nz) for dc in exp.domains
                           if module in domain_kernel_modules(
                               dc, prices_refl=prices_refl)]
        if selected_levels and max(selected_levels) <= THOMPSON_SHALLOW_LEVELS:
            frames[module] = shallow_frame
    # The fork generation is priced at its own frames whenever any domain
    # launching the unit selects it; the 64-level rule is the one above
    # (every launching domain at or under THOMPSON_SHALLOW_LEVELS).
    for module, (deep_frame, shallow_frame) in THOMPSON_WRF39_KERNEL_FRAMES.items():
        launching = [dc for dc in exp.domains
                     if module in domain_kernel_modules(dc, prices_refl=prices_refl)]
        if not any(getattr(dc.run, "thompson_version", "wrf_461") == "wrf_39_noaa"
                   for dc in launching):
            continue
        shallow = max(int(dc.run.nz) for dc in launching) <= THOMPSON_SHALLOW_LEVELS
        frame = shallow_frame if shallow else deep_frame
        if frame > frames.get(module, -1):
            frames[module] = frame
    # Full-column statement generations compile to one of these three
    # capacities, not to nz. All three are driver readings on V12,
    # sm_120 / NVRTC 12.8.93, tools/thompson_generation_oracle/frame.py.
    for dc in exp.domains:
        if "thompson_column" in domain_kernel_modules(dc, prices_refl=prices_refl):
            capacity = next((k for k in (64, 128, 256) if int(dc.run.nz) <= k), None)
            if capacity is None:
                raise ValueError("generated Thompson column capacity is 256 levels; a larger column would overwrite private working arrays")
            frame = {64: 50016, 128: 100000, 256: 199968}[capacity]
            frames["thompson_column"] = max(frames.get("thompson_column", 0), frame)
    return frames


def selects_noahmp(exp: ExperimentConfig) -> bool:
    """Does any domain launch the Noah-MP land surface (scheme 4)?"""
    return any(int(dc.run.sf_surface_physics) == 4 for dc in exp.domains)


def _noahmp_pricing_profile(
        exp: ExperimentConfig,
        profile: DeviceLocalMemoryProfile | None, *,
        declared_card: bool = False) -> DeviceLocalMemoryProfile | None:
    """The profile a Noah-MP configuration is priced on, or the caller's
    own ``profile`` untouched for every other configuration.

    The Noah-MP composed frames are readings of one compile platform, and
    the card's own reading is preferred whenever there is one to prefer:

    * ``profile`` given: it is used as given.  Its own row prices the
      frames when its platform has one; otherwise the ceiling over the
      recorded platforms does, and the basis says so.  The requested card
      is never swapped for a recorded one -- that is the retired
      CARD_CLASS_MULTIPROCESSORS defect in a new coat.
    * ``profile`` is ``None`` and the caller declared a card
      (``--vram-gib``): the estimate is for a machine that is elsewhere,
      so this machine's card is NOT read on its behalf; ``None`` is
      returned and the caller prices the reference profile, whose
      platform is nobody's, from the ceiling.
    * ``profile`` is ``None`` and no card was declared: the question is
      about this machine, so this machine's card is read
      (:func:`live_device_local_memory_profile`) and priced from its own
      row when it has one.  Nothing to read -- ``GPUWM_NO_LOCAL_GPU``
      set, no runtime, no device -- returns ``None`` and the caller
      prices the reference profile from the ceiling, the basis naming
      that the card was not read.

    Non-Noah-MP configurations return ``profile`` unchanged, ``None``
    included, so every existing default (the reference profile in the
    byte formulas, ``card_local_memory_profile`` in the estimate) is
    applied by the caller exactly as before.
    """
    if not selects_noahmp(exp) or profile is not None or declared_card:
        return profile
    return live_device_local_memory_profile()


def kernel_local_memory_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """The launch-time local-memory backing store the experiment reserves.

    One allocation per process, sized by the largest per-thread local frame
    among the kernels the configuration launches -- so this is a maximum, not
    a sum, and it does not grow with domain count.  It DOES grow with the
    level count, because the two widest frames in the tree (``kf`` and
    ``refl``) compile their column arrays to ``nz``.
    """
    profile = _noahmp_pricing_profile(exp, profile)
    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    widest = max(kernel_local_frame_bytes(exp, profile=profile).values(),
                 default=0)
    return profile.reservation_bytes(widest)


def non_pool_device_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Device residency of a woof process that the CuPy pool never reports:
    the CUDA context plus the local-memory backing store.

    Both terms are properties of the DEVICE, so both are read off the
    profile.  The context used to be a flat
    :data:`CUDA_CONTEXT_BYTES` -- one card's 2026-07-26 reading charged
    to every card on every platform, which measured 48 MiB high on a
    5070 Ti and 215 MiB LOW on a Linux 5090 (task 206).
    """
    profile = _noahmp_pricing_profile(exp, profile)
    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    return (profile.cuda_context_bytes
            + kernel_local_memory_bytes(exp, profile=profile)
            + column_workspace_bytes(exp, profile=profile))


def kf_column_workspace_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Device bytes the Kain-Fritsch column workspace holds, or zero.

    This is the term that REPLACED KF's local-memory reservation.
    ``woof/core/kernels/kf.cu`` keeps all 66 of ``kf_column``'s column
    arrays in a global workspace instead of the per-thread local frame,
    and ``woof/core/kf.py`` sizes that workspace to the columns it keeps
    in flight -- ``SMs x KF_TILE_BLOCKS_PER_SM x _TPB`` -- rather than to
    the resident-thread capacity the driver would have charged.  That
    ratio is the whole saving: MEASURED on test host A (RTX 5070 Ti, 70 SMs x
    1,536) at nz = 49, 840.0 MiB of reservation became 174.2 MiB of
    workspace.

    Unlike GF's, this term follows ``nz`` LINEARLY and not a compiled
    tier: the workspace extent is the runtime level count, which is a
    saving the compile-time frame could never give.

    Priced here, beside the context and the backing store, for the same
    reason both of those are: it is a property of the DEVICE and of the
    level count, not of the grid.  Bounded by the column count -- a
    domain smaller than one tile never allocates a whole tile.

    Zero for every configuration that does not select ``cu_physics = 1``.
    """
    from woof.core.kf import (
        KF_TILE_BLOCKS_PER_SM, _TPB, kf_workspace_floats)

    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    tile_cap = profile.multiprocessor_count * KF_TILE_BLOCKS_PER_SM * _TPB
    worst = 0
    for dc in exp.domains:
        if int(dc.run.cu_physics) != 1:
            continue
        columns = min(int(dc.run.nx) * int(dc.run.ny), tile_cap)
        worst = max(worst, kf_workspace_floats(int(dc.run.nz), columns) * 4)
    return int(worst)


def column_workspace_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Device bytes held by the column workspaces, together.

    A SUM and not a maximum, deliberately.  The three workspaces are
    ordinary allocations owned by three different launchers, and a
    configuration holds every one whose scheme it selects at the same
    time -- unlike the local-memory backing store above, which is one
    per-context allocation the driver sizes to the widest frame.  Each
    term is already zero for a configuration that does not select its
    scheme, so this costs nothing where only one is reachable.
    """
    return (gf_column_workspace_bytes(exp, profile=profile)
            + kf_column_workspace_bytes(exp, profile=profile)
            + ysu_column_workspace_bytes(exp, profile=profile)
            + shinhong_column_workspace_bytes(exp, profile=profile)
            + thompson_fork_column_workspace_bytes(exp, profile=profile)
            + gsd_initial_reflectivity_workspace_bytes(exp, profile=profile)
            + ntiedtke_column_workspace_bytes(exp, profile=profile)
            + tiedtke_column_workspace_bytes(exp)
            + g3_column_workspace_bytes(exp, profile=profile)
            + uwpbl_column_workspace_bytes(exp, profile=profile)
            + column_cumulus_workspace_bytes(exp, profile=profile)
            + satmedmf_column_workspace_bytes(exp))


def column_cumulus_workspace_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Device bytes the table-driven column cumulus workspace holds, or zero.

    Every row of woof.core.column_cumulus.COLUMN_CUMULUS_SCHEMES (New SAS,
    cu_physics = 96, first) keeps its column arrays in one global workspace
    of ``slots * (nz + 2)`` floats per column in flight, sized to the
    card's width (SMs x TILE_BLOCKS_PER_SM x BLOCK) and bounded by the
    column count.  The formula is the launcher's own
    (``column_cumulus.workspace_bytes``), not restated.  The worst domain:
    the workspace is a call transient.

    Zero for every configuration that selects no row of the table.
    """
    from woof.core.column_cumulus import (COLUMN_CUMULUS_SCHEMES,
                                           workspace_bytes)

    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    worst = 0
    for dc in exp.domains:
        if int(dc.run.cu_physics) not in COLUMN_CUMULUS_SCHEMES:
            continue
        worst = max(worst, workspace_bytes(
            int(dc.run.cu_physics), int(dc.run.nz),
            int(dc.run.nx) * int(dc.run.ny), profile.multiprocessor_count))
    return int(worst)


def tiedtke_column_workspace_bytes(exp: ExperimentConfig) -> int:
    """cu6's per-call column arenas, summed over the domains that select it.

    Each arena holds the complete j rows one launch walks, as many as
    ``woof.core.tiedtke.rows_per_launch`` admits under its byte budget;
    the row count and the byte formula both come from that module.
    """
    from woof.core.tiedtke import rows_per_launch, workspace_bytes
    total = 0
    for dc in exp.domains:
        if int(dc.run.cu_physics) != 6:
            continue
        nz, nx = int(dc.run.nz), int(dc.run.nx)
        total += workspace_bytes(nz, nx * rows_per_launch(nz, nx, int(dc.run.ny)))
    return total


def g3_column_workspace_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Device bytes the Grell 3D column workspace holds, or zero.

    ``woof/core/kernels/g3.cu`` keeps CUP_enss_3d's level arrays in a
    global workspace of ``G3_SLOT_COUNT`` slots per column in flight, and
    ``woof/core/g3.py`` walks the written band in tiles bounded by
    ``G3_WORKSPACE_BYTES``.  The tile and the slot count both come from
    ``woof.core.g3``; neither is restated here.  It does not depend on the
    device, so ``profile`` is accepted for the common signature only.

    Zero for every configuration that does not select ``cu_physics = 5``.
    """
    from woof.core.g3 import g3_workspace_bytes

    worst = 0
    for dc in exp.domains:
        if int(dc.run.cu_physics) != 5:
            continue
        worst = max(worst, g3_workspace_bytes(
            int(dc.run.nz), int(dc.run.ny), int(dc.run.nx),
            slots=158 + 27 + 6 * int(dc.run.maxens2)))
    return int(worst)


def satmedmf_column_workspace_bytes(exp: ExperimentConfig) -> int:
    worst=0
    for dc in exp.domains:
        if int(dc.run.bl_pbl_physics)!=901:
            continue
        n,k=int(dc.run.nx)*int(dc.run.ny),int(dc.run.nz)
        capacity=(220+16*64)*k+4096
        per_column=capacity*8+((20+2*64)*k+23)*8
        chunk=min(n,max(1,(512*1024*1024)//per_column))
        # Full input/output bindings, finite-step humidity conversion and pool.
        worst=max(worst,chunk*per_column+n*(180*k+32)*8+7501*8)
    return int(worst)


def thompson_fork_column_workspace_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Device bytes of the fork Thompson unit's column arena, or zero.

    thompson_version = "wrf_39_noaa" (mp_physics = 28) keeps the fork's
    column arrays in an arena sized to the launcher's tile of threads in
    flight (woof.core.thompson_fork: TILE_BLOCKS_PER_SM blocks of BLOCK on
    every multiprocessor, never more than the columns need), 377 B per
    level + 1 per thread.  The sizing is the launcher's own formula
    (thompson_fork.workspace_bytes), so this term and the allocation are
    one formula.  Breakage prevented: an unpriced arena of 0.86 GiB at 50
    levels on a 188-SM card, met as an out-of-memory at the first
    microphysics call of a run admitted near the card's limit.  The domains
    run one at a time, so the deepest/widest domain prices it.
    """
    from woof.core.thompson_fork import tile_threads, workspace_bytes

    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    if not WRF_EXACT:
        # The fork unit launches only under GPUWM_WRF_EXACT=1
        # (microphysics_aerosol._apply_thompson_fork_call); a default run
        # allocates no arena, so pricing one would refuse runs that fit.
        return 0
    worst = 0
    for dc in exp.domains:
        if (int(getattr(dc.run, "mp_physics", 0) or 0) != 28
                or getattr(dc.run, "thompson_version", "wrf_461")
                != "wrf_39_noaa"):
            continue
        threads = tile_threads(profile.multiprocessor_count,
                               int(dc.run.nx) * int(dc.run.ny))
        worst = max(worst, workspace_bytes(int(dc.run.nz), threads))
    return int(worst)


def gsd_initial_reflectivity_workspace_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Bounded initial-history radar arena; prevents unpriced t=0 allocation.

    Each enabled domain uses 87 * (nz + 1) bytes per thread in flight.
    The same formula sizes the launcher's three device buffers.
    Domains and host/store slabs are diagnosed sequentially.
    """
    from woof.core.gsd_initial_reflectivity import enabled, tile_threads, workspace_bytes

    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    return max((workspace_bytes(int(dc.run.nz), tile_threads(
                    profile.multiprocessor_count, int(dc.run.nx)*int(dc.run.ny)))
                for dc in exp.domains if enabled(dc.run)), default=0)


def uwpbl_column_workspace_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Device bytes the UW moist-turbulence PBL's column pools hold, or zero.

    ``woof/core/uwpbl.py`` keeps CAM's automatic arrays (binary64, the
    scheme's own precision) in two per-column global pools sized from the
    level count, and walks the domain in column chunks whose pools fit
    ``UWPBL_WORKSPACE_BUDGET_BYTES``.  The sizing lives in
    woof.core.physics_inventory, which the launcher allocates from, so this
    term and the allocation are one formula.  Bounded by the column count:
    a domain smaller than one chunk allocates only its own columns.  The
    worst domain, because the pools are call transients released between
    domains.

    Zero for every configuration that does not select
    ``bl_pbl_physics = 9``.
    """
    from woof.core.physics_inventory import uwpbl_workspace_bytes

    worst = 0
    for dc in exp.domains:
        if int(dc.run.bl_pbl_physics) != UW_PBL_SCHEME:
            continue
        worst = max(worst, uwpbl_workspace_bytes(
            int(dc.run.nz), int(dc.run.nx) * int(dc.run.ny)))
    return int(worst)


def gf_column_workspace_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Device bytes the Grell-Freitas column workspace holds, or zero.

    This is the term that REPLACED most of GF's local-memory reservation.
    ``woof/core/kernels/gf.cu`` keeps GFDRV's column arrays in a global
    workspace instead of the per-thread local frame, and
    ``woof/core/gf.py`` sizes that workspace to the columns it keeps in
    flight -- ``SMs x GF_TILE_BLOCKS_PER_SM x GF_BLOCK`` -- rather than to
    the resident-thread capacity the driver would have charged.  That
    ratio is the whole saving: MEASURED on test host A (RTX 5070 Ti, 70 SMs x
    1,536) at the nz<=40 tier, 2,200.0 MiB of reservation became 422.1
    MiB of workspace.

    It is priced here, beside the context and the backing store, for the
    same reason both of those are: it is a property of the DEVICE and of
    the level count, not of the grid.  It is bounded by the column count
    -- a domain smaller than one tile never allocates a whole tile.

    Zero for every configuration that does not select ``cu_physics = 3``.
    """
    from woof.core.gf import (
        GF_BLOCK, GF_TILE_BLOCKS_PER_SM, gf_workspace_floats)

    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    tile_cap = (profile.multiprocessor_count * GF_TILE_BLOCKS_PER_SM
                * GF_BLOCK)
    worst = 0
    for dc in exp.domains:
        if int(dc.run.cu_physics) != 3:
            continue
        columns = min(int(dc.run.nx) * int(dc.run.ny), tile_cap)
        worst = max(worst, gf_workspace_floats(int(dc.run.nz), columns) * 4)
    return int(worst)


def ysu_column_workspace_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Device bytes the YSU column workspace holds, or zero.

    The term that REPLACED YSU's local-memory reservation, and the one a
    BARE DEFAULT run pays, because ``bl_pbl_physics = 1`` is the wizard's
    default.  ``woof/core/kernels/ysu.cu`` keeps the scheme's column
    arrays in a global workspace instead of the per-thread local frame,
    and ``woof/core/ysu.py`` sizes it to the columns it keeps in flight
    -- ``SMs x YSU_TILE_BLOCKS_PER_SM x YSU_BLOCK`` -- rather than to the
    resident-thread capacity the driver charged.

    MEASURED on test host A (RTX 5070 Ti, 70 SMs x 1,536,
    sm_120) at nz=49 and 102,400 columns: an 842.0 MiB reservation became
    123.0 MiB of workspace, for +1.8% on the kernel's wall clock.

    Unlike the frame it replaced, this term follows ``nz`` rather than the
    kernel's 128-level bound: the extent is a runtime argument, so a
    49-level run holds 50 levels of arrays where the frame held 128.

    Zero for every configuration that does not select
    ``bl_pbl_physics = 1``.
    """
    # physics_inventory, not ysu: the launcher imports cupy at module
    # scope and this pricing must be readable on installs with no GPU
    # runtime (the wizard's estimator runs here).
    from woof.core.physics_inventory import (
        YSU_BLOCK, YSU_TILE_BLOCKS_PER_SM, ysu_workspace_floats)

    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    tile_cap = (profile.multiprocessor_count * YSU_TILE_BLOCKS_PER_SM
                * YSU_BLOCK)
    worst = 0
    for dc in exp.domains:
        if int(dc.run.bl_pbl_physics) != 1:
            continue
        columns = min(int(dc.run.nx) * int(dc.run.ny), tile_cap)
        worst = max(worst, ysu_workspace_floats(int(dc.run.nz), columns) * 4)
    return int(worst)


def shinhong_column_workspace_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Price the largest Shin-Hong tile without importing the GPU runtime."""
    from woof.core.physics_inventory import (
        SHINHONG_BLOCK, SHINHONG_TILE_BLOCKS_PER_SM, shinhong_workspace_floats)

    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    tile_cap = (profile.multiprocessor_count * SHINHONG_TILE_BLOCKS_PER_SM
                * SHINHONG_BLOCK)
    worst = 0
    for dc in exp.domains:
        if int(dc.run.bl_pbl_physics) != 11:
            continue
        columns = min(int(dc.run.nx) * int(dc.run.ny), tile_cap)
        worst = max(worst,
                    shinhong_workspace_floats(int(dc.run.nz), columns) * 4)
    return int(worst)


def ntiedtke_column_workspace_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Device bytes New Tiedtke's column workspace holds, or zero.

    THIS WAS A REFUSAL until 2026-08-29, and the refusal was right for as
    long as it stood: the distinct-array count was not derivable, and a
    silent zero in this sum would have let a run proceed under-budgeted
    with nothing to show for it.  Two things changed.

    First, the count is now MEASURED rather than estimated, because
    ``NtWorkspace`` allocates and can be counted: 89 level arrays, 37
    surface, and 11 level-sized scratch slabs ``cutypen`` slices out of
    one pointer.  The old docstring's worry -- that the union of kernel
    signatures gives 90 names of which many are the same storage under
    different Fortran spellings -- was exactly right, and 41 of them turned
    out to be aliases (docs/ntiedtke/PORT-RECORD.md section 33).

    Second, the old text argued New Tiedtke does NOT cap, because
    ``NtLaunchGeometry`` launches over the full column count.  True when
    written and false since: the ASSEMBLER caps at GF's tile and walks the
    domain in chunks, and the chunking gate proves the answer is
    byte-identical at 32, 64 and 108 columns.  So the analogy to
    :func:`gf_column_workspace_bytes` that the old text warned against is
    now the correct shape.

    BOTH THE CAP AND THE BYTE COUNT COME FROM ``woof.core.ntiedtke``.
    Neither is restated here.  A census formula living apart from the
    thing it counts is what put 75 arrays where 89 belonged, and this
    module would be the second copy.

    Zero for every configuration that does not select ``cu_physics = 16``.
    """
    from woof.core.ntiedtke import nt_tile_columns, nt_workspace_bytes

    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    worst = 0
    for dc in exp.domains:
        if int(dc.run.cu_physics) != 16:
            continue
        columns = nt_tile_columns(int(dc.run.nx) * int(dc.run.ny),
                                  profile.multiprocessor_count)
        worst = max(worst, nt_workspace_bytes(int(dc.run.nz), columns))
    return int(worst)


# ---------------------------------------------------------------------------
# Itemized shape formulas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryItem:
    """One named device allocation: shape formula result + byte width."""

    name: str
    category: str  # state | physics | scratch | lbc | nest | sase
    #                | transient
    shape: tuple[int, ...]
    itemsize: int = 4
    dtype: str = "float32"
    #: Allocated ONCE at exactly this shape and held for the whole run?
    #: Then the pool holds it at its size and it is priced there, outside
    #: the measured pool margin (:func:`forecast_pool_estimate_bytes`,
    #: A163).  A class constant, not a field, so every inventory that
    #: holds no such array is the dataclass it always was:
    #: :class:`HeldMemoryItem` carries it.
    held_exact: ClassVar[bool] = False

    @property
    def nbytes(self) -> int:
        n = self.itemsize
        for extent in self.shape:
            n *= int(extent)
        return n


@dataclass(frozen=True)
class HeldMemoryItem(MemoryItem):
    """A :class:`MemoryItem` allocated once at exactly its shape and held
    for the run.  Only the urban arrays are one
    (``woof.core.urban_state.urban_held_array_shapes``), the family the
    battery measured that way (A163)."""

    held_exact: ClassVar[bool] = True


def _items(category: str, shapes: dict[str, tuple[int, ...]],
           itemsize: int = 4) -> tuple[MemoryItem, ...]:
    return tuple(MemoryItem(name, category, tuple(shape), itemsize)
                 for name, shape in shapes.items())


def _nest_items(shapes: dict[str, tuple[int, ...]],
                dtypes: dict[str, str]) -> tuple[MemoryItem, ...]:
    """F4 nest items with their semantic dtype recorded explicitly."""
    if shapes.keys() != dtypes.keys():
        raise RuntimeError("nest shape/dtype registries drifted")
    return tuple(MemoryItem(name, "nest", tuple(shape), 4, dtypes[name])
                 for name, shape in shapes.items())


@dataclass(frozen=True)
class PhysicsArrayLifetime:
    """Closed-world lifetime proof for one exact physics-array set."""

    names: tuple[str, ...]
    disposition: str
    evidence: str
    proof: str


_YSU_3D = ("du", "dv", "dtheta", "dqv", "dqc", "dqi",
           "exch_h", "exch_m")
_YSU_2D = ("hpbl", "kpbl", "wstar", "delta", "topdown_radsum",
           "wstar3_2", "cloudflg")
_TENDENCY_COMPONENTS = ("ru", "rv", "rtheta", "rqv", "rqc", "rqr",
                        "rqi", "rqs", "rw")
_MICROPHYSICS_COMPONENTS = ("rainnc", "rainncv", "sr", "snownc",
                            "snowncv", "graupelnc", "graupelncv", "hailnc",
                            "hailncv")

#: Persistent-physics counterpart to ``SCRATCH_SLOT_LIFETIME_AUDIT``.
#: Exact names make the three reclamations closed-world: a future diagnostic
#: or tendency component receives no aliasing without a new reviewed row.
PHYSICS_ARRAY_LIFETIME_AUDIT = (
    PhysicsArrayLifetime(
        ("gf_rthblten", "gf_rqvblten"),
        "retained_family_state", "woof/core/physics.py:_couple_pbl_slot",
        "GF/New Tiedtke read raw PBL forcing between producer calls; stable "
        "buffers are carried through restart, streaming and relocation"),
    PhysicsArrayLifetime(
        tuple(f"pbl_raw_rates/{name}" for name in
              ("du", "dv", "dw", "dtheta", "dqv", "dqc", "dqi")),
        "retained_family_state", "woof/core/physics.py:recouple_after_relocation",
        "positive cadence retains A-grid rates for coupling on relocated "
        "mass; theta/qv alias the GF pair where present and are priced once"),
    PhysicsArrayLifetime(
        tuple(f"last_ysu/{name}" for name in (*_YSU_3D, *_YSU_2D)),
        "transient_when_bldt_zero", "woof/core/physics.py:862-893",
        "field copies and coupling are the only readers; bldt=0 releases "
        "the dict after them; positive cadence retains diagnostics while "
        "raw-rate entries alias the canonical PBL buffers"),
    PhysicsArrayLifetime(
        tuple(f"microphysics/{name}" for name in _MICROPHYSICS_COMPONENTS),
        "aliases_serialized_scratch", "woof/core/dycore.py:1429-1439; "
        "woof/core/physics.py:514-534,569-631,842-856; "
        "woof/runtime.py:571-585",
        "pre-RK Noah reads precede the post-RK scheme write; output reads "
        "after accept, so the driver can alias the canonical mp_* set"),
    PhysicsArrayLifetime(
        tuple(f"tendencies/{name}" for name in _TENDENCY_COMPONENTS if name != "rw"),
        "aliases_fresh_pbl_at_bldt_zero", "woof/core/physics.py:779-819,"
        "895-959",
        "bldt=0 YSU replaces pbl_tendencies before every composition; "
        "positive cadence retains the separate target unchanged"),
    PhysicsArrayLifetime(
        ("tendencies/rw",), "retained_family_state",
        "woof/core/physics.py:_compose_tendencies",
        "the composed target borrows the held PBL z-face tendency read-only "
        "at every cadence; no second buffer or sum is allocated"),
    PhysicsArrayLifetime(
        tuple(f"{stack}/{name}" for stack in
              ("pbl_tendencies", "radiation_tendencies",
               "cumulus_tendencies") for name in _TENDENCY_COMPONENTS),
        "retained_family_state", "woof/core/physics.py:633-777,895-959",
        "radiation/cumulus carry between due calls; PBL is the proven "
        "bldt=0 composition backing but is otherwise held family state"),
)


def physics_array_lifetime(name: str) -> PhysicsArrayLifetime | None:
    """Return the unique exact-name physics lifetime row, if audited."""
    matches = [row for row in PHYSICS_ARRAY_LIFETIME_AUDIT
               if name in row.names]
    if len(matches) > 1:
        raise RuntimeError(f"physics lifetime audit overlaps for {name!r}")
    return matches[0] if matches else None




#: Strain/stress component order (sase.py launch_strain / authority) and
#: the Germano-lift pair order (authority ``_PAIRS``).
_SASE_S6 = ("xx", "yy", "zz", "xy", "xz", "yz")
_SASE_P6 = ("uu", "vv", "ww", "uv", "uw", "vw")


def sase_workspace_phases(cfg: RunConfig
                          ) -> dict[str, dict[str, tuple[tuple[int, ...],
                                                         int]]]:
    """SASE model-path transient live sets, one dict per phase.

    Exact transcription of the driver-coupled step's per-call device
    allocations (``woof/core/physics.py`` ``_run_sase`` +
    ``woof/core/sase.py`` ``launch_sase_step`` and the launchers it
    composes) on the MODEL path (per-column 3-D ``dz_col``, Task 6).
    Task-6 decision, documented here per the S3-5 pairing contract: the
    per-step temporaries REMAIN CuPy-pool allocations at this stage
    rather than moving into the shared scratch arena -- the pool
    reuses the freed blocks across steps so the steady-state device
    footprint equals this transcribed peak, while arena preallocation
    would require threading an allocator through six nested launchers
    and re-auditing ~58 slot lifetimes (revisit if the estimate ever
    pinches).  Any change to either side must update BOTH the launcher/
    driver and this transcription (the byte pin in tests/test_sase.py
    enforces the pairing).

    * ``solve`` -- the peak inside ``launch_dynamic_solve`` while
      ``launch_germano_lift`` runs on the second (width-4) test level:
      the driver-held work set (u/v A-grid work copies, destaggered w +
      its work copy, n2 plus its S4-2 M1 moist companion n2_eff -- the
      ``launch_moist_n2`` output held beside the dry field for the whole
      step (physics.py ``_run_sase`` step 2; both go down together into
      ``launch_sase_step``), and the ``heat`` output -- 7 full fields;
      the S3-6e governed scalar channel rides the step's exported km_h
      field, so the v0 pre-step e copy is RETIRED), the fused step's
      three PER-COLUMN z-stencil coefficient FIELDS (3-D dz_col mode;
      the (nz,) arrays of the shared-column test path are superseded on
      the model path), the six fine strains, six premultiplied
      eddy-basis integrands, three filtered velocities, six coarse
      strains, six refiltered basis fields, the lift's three filtered
      velocities + six velocity products + six filtered products + six
      lift outputs, and the width-2 iteration's still-referenced FP64
      ``(5, nblocks)`` partial-sum buffer: 58 full fields + partials
      (S4-3 amendment: was 57 pre-M1; the n2_eff field is 1/57 ~ 1.75%
      of the previous peak -- the S4-2 report note-1 obligation).  The
      S3-11b ``(ny, nx)`` float32 rho1 surface moist-density plane
      (``sase_surface_rho1``, computed once before the e source and held
      by the driver through the step-5 scalar-flux deposit, so alive in
      BOTH phases) is transcribed with it -- de-minimis at
      1/(57*nz) ~ 0.04% of the peak (the S3-11b report note-3 flag,
      absorbed explicitly here rather than left to the (nz,)-pack
      de-minimis precedent since this touch amends the pairing anyway).
      S4-5 amendment (the S4-3 in-task pairing law): the SASE-M2 deposit
      seam holds ONE new driver field in BOTH phases -- the pre-step
      ``e_sgs`` copy ``e_pre`` that freezes the venting limb's amplitude
      input (physics.py ``_run_sase``; theta/qv/qc/pressure are
      read-only through the whole slot, so ``e`` is the only state the
      limb needs frozen, and the limb cannot be evaluated beside the
      copy because its (1 - f) two-product blend needs the step's USED
      f).  One full 4*ncell field, 1/58 ~ 1.72% of the previous peak.
      It is NOT the v0 pre-step e copy the S3-6e governed scalar channel
      retired (that one fed ``launch_scalar_mix``'s coefficient mode and
      stays retired); it is a new field with a new consumer, and it is
      transcribed under the DISTINCT key ``driver_vent_e_pre``.  S4-5b
      Item 4b: the S4-5 build gave it the retired field's own key
      ``driver_e_pre`` and deleted the retirement guard that asserted
      that key absent, while stating here that ``driver_e_pre`` "is
      deliberately not its name" -- it was.  The suffixed key restores
      the guard's meaning: ``driver_e_pre`` is absent because the v0
      copy is retired, and the M2 seam's frozen copy is a different
      name.  The byte pin does not move (one ``(nz, ny, nx)`` float32
      field either way; re-derived at
      ``test_sase.test_sase_workspace_accounting_is_exact``).
      The launcher EXPLICITLY drops the previous width's six-field lift
      binding (``lift = None``) before the next width's allocations --
      without that drop the peak would be 6 fields higher; this
      transcription and the launcher are a bound pair (S3-5 review fix),
      so removing the drop requires re-pinning here.  The S3-6f
      partition-bound sub-moment (the ``(ny, nx)`` z_i column field and
      the FP64 ``(5, nblocks)`` w-sensor partials) runs AFTER the solve
      transients release and drops its references before the apply
      allocations -- transcribed in this phase as a covering superset
      (net far below the 57-field peak).
    * ``apply`` -- re-derived for the S3-6e governed split step's true
      allocation profile.  The peak sits at the ``sase_split_tendencies``
      launch: the same driver-held 7 + coefficient 3, the vertical
      channel's kv + leps fields (S3-6h: leps = the BL89 RANS
      dissipation limb, formerly the bare l_B), six strains, six
      stresses, the governed
      diffusivity km + smag-share r fields (S3-6e), TWO horizontal
      e-flux integrands (the vertical leg retired with the explicit
      step), and FIVE tendency fields (du, dv, dw, P_h,e, P_h,heat --
      the S3-6e production split) -- 33 full fields with the S4-3 M1
      n2_eff amendment, + the rho1 plane (S3-6f doc fix, kept for the
      record: the S3-6e text said 33 when the enumeration summed to 32
      pre-M1; the byte pin never drifted).  The launcher then
      DROPS the 13-field strain/stress/r binding before the
      Thomas/production/partials allocations (bound pair with this
      transcription, the S3-5 idiom), so the later sub-moments
      (momentum-Thomas FP64 ``(3, nblocks_col)`` partials -- the third
      row is the S3-6j dKE_sfc drag-work channel -- P_v, the
      S3-6e damping-taper weight field under damp_opt=3, the e-update
      FP64 ``(4, nblocks)`` partials) arrive net NEGATIVE; the partials
      buffers, P_v, and the taper field are transcribed anyway as a
      covering superset.  S4-5: the SASE-M2 seam's THREE
      ``(nz + 1, ny, nx)`` float32 face-flux planes and its
      ``(ny, nx)`` FP64 cap-rescale plane are allocated in this phase,
      after the split step returns and before the scalar loop, and are
      transcribed here for the same covering-superset reason (they land
      after the 13-field strain/stress/r drop, so the apply phase's own
      peak does not move; the solve phase continues to dominate the
      category bound by **20.01 full fields** at 49x250x250 and at
      49x501x501, 19.59 at the test configuration -- S4-5b Item 5b,
      MEASURED this session from :func:`sase_workspace_phases` itself
      as ``(sum(solve) - sum(apply))/(4*nz*ny*nx)``; the S4-5 text said
      "~25", which was never the number.  The M2 seam's own four apply
      entries are what closed the gap: at the BASE commit a5b8d7e the
      same probe measures 23.11 (23.21 at the test configuration), and
      3*(nz + 1)*4 + 8 bytes per column against 4*nz is exactly the
      3.10-field difference).  The Thomas sweeps themselves hold their
      r/c'/d' arrays in per-thread registers/local memory (3x SASE_KMAX
      doubles per thread) -- NO global workspace, which is why no
      Thomas entry appears here.  S3-12 (the in-task pairing law): the
      additive dissipation channel's ONE new device field -- the
      state-independent Blackadar reference length
      (``launch_blackadar_length``), allocated after the 13-field drop
      and only under ``cfg.sase_additive_dissipation`` -- is
      transcribed unconditionally as the covering-superset entry
      ``lb_ref``, exactly the taper_g idiom; the solve phase keeps
      dominating (~20-field margin) so neither the category bound nor
      the byte pin moves.  The post-step scalar phase (kv + km_h
      held + 2 horizontal flux + 4 mixed rates + the coupled set)
      stays far below the solve bound.

    The category bound is the MAXIMUM over phases (the
    :func:`rrtmgp_workspace_phases` idiom); ``solve`` always dominates.
    The solve-internal (nz,) coefficient packs of its uniform-dz strain
    calls remain de-minimis and untranscribed, exactly as before.
    """
    if cfg.bl_pbl_physics != SASE_PBL_SCHEME:
        return {}
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    m = (nz, ny, nx)
    ncell = nz * ny * nx
    # Single-sourced with the device define through the closure's own
    # compile-time tier, so this transcription's block count cannot
    # drift from the block size the kernels are actually compiled at.
    from woof.core.sase_limits import THREADS_PER_BLOCK
    tpb = THREADS_PER_BLOCK
    nblocks = (ncell + tpb - 1) // tpb
    partials = ((5, nblocks), 8)              # FP64 in-kernel reductions

    common: dict[str, tuple[tuple[int, ...], int]] = {"heat": (m, 4)}
    for name in ("zcm", "zc0", "zcp"):        # per-column coefficient pack
        common[name] = (m, 4)
    # S4-3 amendment (S4-2 report note-1 obligation): n2_eff = the M1
    # launch_moist_n2 output, the 6th driver-held work field, alive
    # beside the dry n2 through the whole step (both phases).
    # S4-5 amendment: the SASE-M2 seam's frozen pre-step e_sgs copy is
    # the 7th driver-held work field, alive from before the surface e
    # source through the step-5 scalar loop (both phases).  S4-5b
    # Item 4b: keyed ``vent_e_pre``, NOT ``e_pre`` -- the latter is the
    # retired v0 pre-step e copy's name and stays absent by name.
    for name in ("u_work", "v_work", "w_half", "w_work", "n2", "n2_eff",
                 "vent_e_pre"):
        common[f"driver_{name}"] = (m, 4)     # _run_sase held work set
    # S3-11b note-3 absorbed (same pairing touch): the (ny, nx) float32
    # rho1 surface moist-density plane (sase_surface_rho1), held from
    # before the e source through the step-5 scalar-flux deposit.
    common["driver_rho1"] = ((ny, nx), 4)

    solve = dict(common)
    for comp in _SASE_S6:
        solve[f"s_fine_{comp}"] = (m, 4)
        solve[f"premul_{comp}"] = (m, 4)
        solve[f"s_coarse_{comp}"] = (m, 4)
        solve[f"refilt_{comp}"] = (m, 4)
    for comp in ("u", "v", "w"):
        solve[f"filt_{comp}"] = (m, 4)
        solve[f"lift_filt_{comp}"] = (m, 4)
    for comp in _SASE_P6:
        solve[f"lift_prod_{comp}"] = (m, 4)
        solve[f"lift_fprod_{comp}"] = (m, 4)
        solve[f"lift_{comp}"] = (m, 4)
    solve["partials"] = partials
    # S3-6f partition-bound sub-moment (covering superset -- docstring):
    # the z_i column field and the w-sensor FP64 reduction buffer.
    solve["zi"] = ((ny, nx), 4)
    solve["w_partials"] = ((5, nblocks), 8)

    apply_phase = dict(common)
    for name in ("kv", "leps"):               # vertical-channel fields
        apply_phase[name] = (m, 4)
    for comp in _SASE_S6:
        apply_phase[f"strain_{comp}"] = (m, 4)
        apply_phase[f"tau_{comp}"] = (m, 4)
    for name in ("km_h", "r_smag"):           # S3-6e governed stress
        apply_phase[name] = (m, 4)
    for comp in ("x", "y"):                   # horizontal e-flux only
        apply_phase[f"e_hflux_{comp}"] = (m, 4)
    for comp in ("du", "dv", "dw", "ph_e", "ph_heat"):
        apply_phase[f"tend_{comp}"] = (m, 4)
    # Post-drop sub-moments (net below the 32-field peak; covering
    # superset -- see the docstring): implicit-flux production, the
    # S3-6e damping-taper weight field, and the two FP64 reduction
    # buffers.
    apply_phase["prod_v"] = (m, 4)
    apply_phase["taper_g"] = (m, 4)
    # S3-12: the additive channel's state-independent Blackadar
    # reference field (launch_blackadar_length), allocated after the
    # 13-field strain/stress/r drop and only under
    # cfg.sase_additive_dissipation -- transcribed unconditionally as a
    # covering superset exactly like taper_g (the solve phase dominates
    # by ~20 fields, so the category bound does not move; the byte pin
    # rides the solve phase and is untouched).
    apply_phase["lb_ref"] = (m, 4)
    # S4-5 SASE-M2 deposit seam: the three face-registered flux planes
    # and the per-column FP64 cap-rescale plane.
    for name in ("vent_f_theta", "vent_f_qv", "vent_f_qc"):
        apply_phase[name] = ((nz + 1, ny, nx), 4)
    apply_phase["vent_scale"] = ((ny, nx), 8)
    ncol_blocks = (ny * nx + tpb - 1) // tpb
    # S3-6j: third row = the dKE_sfc drag-work reduction channel.
    apply_phase["partials_mom"] = ((3, ncol_blocks), 8)
    apply_phase["partials_e"] = ((4, nblocks), 8)

    return {"solve": solve, "apply": apply_phase}


def sase_workspace_shapes(cfg: RunConfig
                          ) -> dict[str, tuple[tuple[int, ...], int]]:
    """The SASE step-transient bound: the phase-maximum simultaneous set,
    ``{"sase/<phase>/<name>": (shape, itemsize)}`` (see
    :func:`sase_workspace_phases`)."""

    def total(items):
        return sum(math.prod(shape) * size for shape, size in items.values())

    phases = sase_workspace_phases(cfg)
    if not phases:
        return {}
    phase = max(phases, key=lambda name: total(phases[name]))
    return {f"sase/{phase}/{name}": spec
            for name, spec in phases[phase].items()}


def shared_dycore_state_symbols() -> frozenset[str]:
    """Restart-REBUILT symbols eligible for sequential-domain sharing.

    The restart manifest is the sole inventory authority.  Keeping this as a
    function avoids a second module-level set that could drift independently.
    """
    from woof.io.restart import STATE_REBUILT_ATTRS

    # Strict mode's p_perturbation is rebuilt by a restart but CARRIED
    # between steps in a live run: rk_step 1 reads the previous step's
    # words without diagnosing again (core/dycore.py:_wrf_reuse_diagnostics)
    # and every history frame writes it as P.  Shared, a two-domain strict
    # run read the last-stepped domain's pressure in another shape: on the
    # UAE 9 km/3 km pair d01's history P differed from wrf.exe 4.6.1 in
    # 527,901 of 528,000 words at 45 s and was 0 outside the nest at t = 0.
    # Each domain keeps its own, like the carried wrf_exact_muts slot.
    return STATE_REBUILT_ATTRS - {"p_perturbation"}


def shared_dycore_state_workspace_shapes(
        domains: tuple[object, ...]) -> dict[str, tuple[int, ...]]:
    """Maximum requested shape for each active restart-REBUILT symbol."""
    maxima: dict[str, tuple[int, ...]] = {}
    rebuilt = shared_dycore_state_symbols()
    for dc in domains:
        shapes = state_array_shapes(dc.run)
        for symbol in rebuilt & shapes.keys():
            shape = shapes[symbol]
            previous = maxima.get(symbol)
            if previous is None or math.prod(shape) > math.prod(previous):
                maxima[symbol] = shape
    return {symbol: maxima[symbol] for symbol in sorted(maxima)}


def shared_dycore_state_workspace_bytes(
        domains: tuple[object, ...]) -> int:
    """Exact bytes for one float32 maximum backing per active symbol."""
    return sum(4 * math.prod(shape) for shape in
               shared_dycore_state_workspace_shapes(domains).values())


#: initialize_physics's own 2-D allocations before the SFCLAY/Noah unions
#: (physics.py:935-950).
_PHYSICS_INIT_FIELDS_2D = (
    "landmask", "xland", "tsk", "sst", "pblh", "mavail", "lakemask",
    "ivgtyp", "isltyp", "vegfra", "tmn", "xice", "xicem", "swdown", "glw",
    "snow", "snowh", "sstsk",
    "achfx", "aclhf", "acgrdflx",
)


def physics_field_names_2d(cfg: RunConfig | None = None) -> tuple[str, ...]:
    """The surface ``fields`` dict's 2-D inventory, reconstructed from the
    same name tuples ``initialize_physics`` consumes (physics.py:935-990):
    init fields | SFCLAY_OUTPUTS | NOAH _F2D, plus ``ebal``/``kpbl``.

    ``initialize_physics`` additionally allocates MYNN's extra persistent
    surface diagnostics when ``sf_sfclay_physics == 5``, and the Eta
    layer's own set when it is 2, and deliberately allocates neither
    otherwise.  Passing ``cfg`` reproduces that selection; omitting it
    returns the MM5/Noah union, which is what every caller without a
    configuration in hand means.  Under-counting here is a correctness bar
    on this hardware, not a cosmetic one.
    """
    from woof.core.noah import _F2D as NOAH_FIELDS_2D
    # physics_inventory, not sfclay/mynn_*: those modules import cupy
    # at module scope and this inventory must be readable on installs
    # with no GPU runtime (the wizard's estimator runs here).
    from woof.core.physics_inventory import SFCLAY_OUTPUTS

    union = dict.fromkeys(_PHYSICS_INIT_FIELDS_2D)
    if cfg is not None:
        # WRF's ocean mixed layer carries seven planes and reads GSW.
        from woof.surface_ocean import ocean_fields
        if int(getattr(cfg, "sf_ocean_physics", 0)) == 2:
            union.update(dict.fromkeys(("om_ml", "_pwp_mx", "_pwp_my", "_pwp_land")))
        else:
            union.update(dict.fromkeys(ocean_fields(cfg)))
        if int(getattr(cfg, "sf_ocean_physics", 0)):
            union["gsw"] = None
            union.update(dict.fromkeys(("ocean_sst", "ocean_sst_change")))
    if cfg is not None and cfg.nwp_diagnostics == 0 and not cfg.gsd_diagnostics:
        union["refd_max"] = None
    union.update(dict.fromkeys(SFCLAY_OUTPUTS))
    if cfg is not None and int(cfg.sf_sfclay_physics) == 5:
        from woof.core.physics_inventory import MYNN_SURFACE_OUTPUTS
        union.update(dict.fromkeys(MYNN_SURFACE_OUTPUTS))
    elif (cfg is not None
          and int(cfg.sf_sfclay_physics) == MYJ_SFCLAY_SCHEME):
        # The Eta layer's persistent set, the same tuple initialize_physics
        # allocates for this selector.  Without it 17 surface planes per
        # domain (akhs, akms, thz0 and the rest) were allocated unpriced.
        from woof.core.physics_inventory import MYJ_SFCLAY_FIELDS_2D
        union.update(dict.fromkeys(MYJ_SFCLAY_FIELDS_2D))
    union.update(dict.fromkeys(NOAH_FIELDS_2D))
    if cfg is not None and radiation_enabled(cfg):
        from woof.core.topo_radiation import topo_shortwave_active
        if not topo_shortwave_active(cfg):
            union["swnorm"] = None
    if cfg is not None:
        from woof.config import radiation_scheme_ids
        if radiation_scheme_ids(cfg)[1]:
            union.update(dict.fromkeys(("swddir", "swddif", "swddni")))
        from woof.core.physics_inventory import radiation_boundary_flux_names
        for name in radiation_boundary_flux_names(cfg):
            union[name] = union["ac"+name] = None
        if "swupt" in radiation_boundary_flux_names(cfg):
            union["coszen"] = None
    union.update(dict.fromkeys(("ebal", "kpbl")))
    if cfg is not None and int(cfg.bl_pbl_physics) == 5:
        from woof.core.physics_inventory import (
            MYNN_PBL_DIAGNOSTICS_2D, MYNN_PBL_DIAGNOSTICS_INT_2D,
        )
        union.update(dict.fromkeys(MYNN_PBL_DIAGNOSTICS_2D))
        union.update(dict.fromkeys(MYNN_PBL_DIAGNOSTICS_INT_2D))
    if cfg is not None and int(cfg.sf_surface_physics) == 3:
        from woof.core.ruc_runtime import (
            RUC_DIAGNOSTICS_2D, RUC_FRACTIONAL_SEAICE_FIELDS, RUC_STATE_2D,
        )
        from woof.core.surface_forcing import SURFACE_PRECIPITATION_FIELDS
        union.update(dict.fromkeys(RUC_STATE_2D))
        union.update(dict.fromkeys(RUC_DIAGNOSTICS_2D))
        union.update(dict.fromkeys(RUC_FRACTIONAL_SEAICE_FIELDS))
        if int(cfg.sf_sfclay_physics) == 5:
            # MYNN_SEAICE_WRAPPER performs a full second surface call.  WRF
            # keeps most results as automatic locals and exposes only the
            # wait-for-LSM subset; ArWen retains the full result so the GPU
            # launch allocates nothing outside this preflight inventory.
            from woof.core.physics_inventory import MYNN_SURFACE_OUTPUTS
            union.update(dict.fromkeys(
                f"{name}_sea" for name in MYNN_SURFACE_OUTPUTS))
        union.update(dict.fromkeys(SURFACE_PRECIPITATION_FIELDS))
        union["gsw"] = None
    if cfg is not None:
        # A chem process that reads GSW keeps the plane outside RUC too
        # (physics.initialize_physics, the same chem_physics_reads answer).
        from woof.core.chem_context import chem_physics_reads
        if "gsw" in chem_physics_reads(cfg) or int(cfg.aer_opt) == 1:
            union["gsw"] = None
        # Grell 3D reads GSW for its capping depth (physics.initialize_
        # physics keeps the plane from the same table).
        from woof.config import CU_GSW_READING_SCHEMES
        if int(cfg.cu_physics) in CU_GSW_READING_SCHEMES:
            union["gsw"] = None
    if cfg is not None and int(cfg.sf_surface_physics) == 4:
        from woof.core.noahmp_runtime import (
            NOAHMP_DIAGNOSTICS_2D, NOAHMP_STATE_2D, NOAHMP_STATE_INT_2D,
        )
        union.update(dict.fromkeys(NOAHMP_STATE_2D))
        union.update(dict.fromkeys(NOAHMP_STATE_INT_2D))
        union.update(dict.fromkeys(NOAHMP_DIAGNOSTICS_2D))
        from woof.core.surface_forcing import SURFACE_PRECIPITATION_FIELDS
        union.update(dict.fromkeys(SURFACE_PRECIPITATION_FIELDS))
        union["coszen"] = None
    if cfg is not None and int(getattr(cfg, "alb_sol", 0)) == 1:
        union.update(dict.fromkeys(("albsol", "albbcksol")))
    return tuple(union)


def physics_array_shapes(cfg: RunConfig, *, cam_ozone: bool = False,
                         urban_columns: int | None = None,
                         lake_columns: int | None = None
                         ) -> dict[str, tuple[int, ...]]:
    """``PhysicsDriver`` persistents per selected scheme (physics.py).

    Includes the surface/Noah ``fields`` dict, held family tendencies, a
    separate composed target except in the proven bldt=0/PBL identity path,
    positive-cadence raw YSU retention, radiative heating rates, mp=0 output
    placeholders, KF W0AVG/LUT, and RRTMGP setup grids.  Active microphysics
    diagnostics and KF ``cu_*`` persistence live in the scratch registry.
    ``urban_columns`` is the domain's urban column count where a prepared
    door has read its land cover
    (:func:`woof.core.urban_state.urban_array_shapes`).
    """
    # The runtime-free inventory module, NOT woof.core.physics: that
    # module's body imports cupy, and this estimator is what `woof
    # domain` runs on CPU-only installs (it used to refuse every one of
    # them from exactly this import).
    from woof.core.physics_inventory import (PBL_RQI_MICROPHYSICS,
                                              hmix_k_diag_names,
                                              microphysics_scratch_slots,
                                              physics_driver_required,
                                              physics_retains_ysu_output,
                                              physics_reuses_pbl_composition)

    if not physics_driver_required(cfg) and not cam_ozone:
        return {}
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    m = (nz, ny, nx)
    s2 = (ny, nx)
    shapes: dict[str, tuple[int, ...]] = {}
    if cam_ozone:
        shapes["radiation/o33d_grid"] = m
    shapes["fields/p_hyd"] = m
    from woof.core.physics_inventory import scm_forcing_fields
    for name in scm_forcing_fields():
        shapes[f"fields/{name}"] = ()
    if cfg.bl_pbl_physics:
        from woof.core.physics_inventory import pbl_history_rates
        for name in pbl_history_rates(cfg):
            shapes[f"fields/{name}"] = m
    if cfg.cu_physics:
        from woof.core.physics_inventory import (CUMULUS_HISTORY_RATES,
                                                  history_tendency_rows)
        if history_tendency_rows():
            for name in CUMULUS_HISTORY_RATES:
                shapes[f"fields/{name}"] = m
    for name in physics_field_names_2d(cfg):
        shapes[f"fields/{name}"] = s2
    if int(getattr(cfg, "sf_ocean_physics", 0)) == 2:
        from woof.ocean.mixed_layer import PROFILE_FIELDS
        for name in (*PROFILE_FIELDS, "_pwp_w", "_pwp_p"):
            shapes[f"fields/{name}"] = (cfg.ocean_levels, ny, nx)
        for name in ("_pwp_base", "_pwp_tend"):
            shapes[f"fields/{name}"] = (4*cfg.ocean_levels, ny, nx)
        shapes["fields/_pwp_status"] = (1,)
    n_soil = soil_layer_count(cfg)
    for name in ("smois", "tslb", "sh2o", "smcrel"):
        shapes[f"fields/{name}"] = (n_soil, ny, nx)
    if int(cfg.sf_surface_physics) == 3:
        # Configuration-only estimates allow the largest supported RUC
        # tables (USGS-RUC 28 and STAS-RUC 19). Actual category arrays keep
        # their source dimensions, so this is a conservative bound.
        if cfg.mosaic_lu:
            shapes["fields/landusef"] = (28, ny, nx)
        if cfg.mosaic_soil:
            shapes["fields/soilctop"] = (19, ny, nx)
    if int(getattr(cfg, "sf_lake_physics", 0)) == 1:
        from woof.core.lake_schema import (
            LAKE_STATE_WORDS, LAKE_STATIC_WORDS,
            LAKE_FORCING_WORDS, LAKE_OUTPUT_WORDS)
        count = ny * nx if lake_columns is None else int(lake_columns)
        if not 0 <= count <= ny * nx:
            raise ValueError("lake_columns must lie within the domain")
        # Restart/rank carriers retain horizontal axes; only computation
        # uses the sparse lake-column work arrays.
        shapes["fields/lake_columns"] = (LAKE_STATE_WORDS, ny, nx)
        shapes["fields/lake_static"] = (LAKE_STATIC_WORDS, ny, nx)
        shapes["fields/lake_latitude"] = s2
        shapes["lake/indices"] = (count,)
        shapes["lake/columns"] = (LAKE_STATE_WORDS, count)
        shapes["lake/static"] = (LAKE_STATIC_WORDS, count)
        shapes["lake/latitude"] = (count,)
        shapes["lake/forcing"] = (LAKE_FORCING_WORDS, count)
        shapes["lake/outputs"] = (LAKE_OUTPUT_WORDS, count)
        shapes["lake/errors"] = (count,)
        # Bound library workspace for indexed horizontal gathers even when
        # an out buffer is supplied; this also covers the smaller init seed.
        shapes["lake/gather_work"] = (LAKE_STATE_WORDS, count)
        shapes["lake/precipitation"] = s2
        # The column arrays' device arena (kernels/lake_support.cuh), one
        # launch window wide, in place of the 14,400 B local frame the
        # backing store used to charge every resident thread of the card.
        from woof.core.lake import LAKE_ARENA_SLOTS, LAKE_LAUNCH_COLUMNS
        shapes["lake/arena"] = (2 * LAKE_ARENA_SLOTS,
                                max(1, min(count, LAKE_LAUNCH_COLUMNS)))
    if int(getattr(cfg, "sf_surface_mosaic", 0)) == 1:
        from woof.core.noah_mosaic import mosaic_array_shapes
        for name, (shape, dtype) in mosaic_array_shapes(
                cfg.mosaic_cat, ny, nx, urban=int(getattr(cfg, "sf_urban_physics", 0)) == 1).items():
            shapes[f"fields/{name}"] = shape  # All tile dtypes are 4 bytes.
    shapes["fields/exch_h"] = m
    shapes["fields/exch_m"] = m
    if int(cfg.bl_pbl_physics) == 5:
        # initialize_physics allocates MYNN's ten carried 3-D arrays only for
        # this selector.  Missing them here under-counts VRAM by 10*nz*ny*nx
        # FP32 words, which on a four-domain nest is not a rounding error.
        from woof.core.physics_inventory import MYNN_PBL_STATE_3D
        for name in MYNN_PBL_STATE_3D:
            shapes[f"fields/{name}"] = m
        shapes["fields/tke_pbl"] = (nz + 1, ny, nx)
    if int(cfg.bl_pbl_physics) == 11:
        # Shin-Hong's Registry carriers TKE_PBL and EL_PBL (initialize_physics).
        shapes["fields/tke_pbl"] = (nz + 1, ny, nx)
        shapes["fields/el_pbl"] = m
    if int(cfg.bl_pbl_physics) == MYJ_PBL_SCHEME:
        # MYJ's carried TKE_MYJ and EL_MYJ, allocated by initialize_physics
        # for this selector only.  Missing them under-counts by 2*nz*ny*nx
        # FP32 words per domain: with the Eta layer's planes above, 0.87 GiB
        # at 1792x1024x55, enough to admit a run that cannot allocate.
        from woof.core.physics_inventory import MYJ_PBL_STATE_3D
        for name in MYJ_PBL_STATE_3D:
            shapes[f"fields/{name}"] = m
    if int(cfg.bl_pbl_physics) == UW_PBL_SCHEME:
        # The UW moist-turbulence PBL's carried and published fields,
        # allocated by initialize_physics for this selector only and read
        # from the same inventory tuples: the interface diffusivities and
        # diagnostics at nz+1 levels, the residual stress and the surface
        # diagnostics, and the held radiation cloud fraction.  Missing them
        # under-counts by (5*(nz+1) + nz + 5) planes per domain.
        from woof.core.physics_inventory import (
            UWPBL_DIAGNOSTICS_2D, UWPBL_DIAGNOSTICS_FULL, UWPBL_HELD_3D,
            UWPBL_STATE_2D, UWPBL_STATE_FULL)
        for name in UWPBL_STATE_FULL + UWPBL_DIAGNOSTICS_FULL:
            shapes[f"fields/{name}"] = (nz + 1, ny, nx)
        for name in UWPBL_STATE_2D + UWPBL_DIAGNOSTICS_2D:
            shapes[f"fields/{name}"] = s2
        for name in UWPBL_HELD_3D:
            shapes[f"fields/{name}"] = m
    if int(cfg.sf_surface_physics) == 3:
        # RUC's two Registry-package soil-column arrays, SMFR3D and
        # KEEPFR3DFLAG.  At nine levels that is 18*ny*nx FP32 words per
        # domain on top of the four generic soil arrays.
        from woof.core.ruc_runtime import RUC_STATE_3D
        for name in RUC_STATE_3D:
            shapes[f"fields/{name}"] = (n_soil, ny, nx)
    if int(cfg.sf_surface_physics) == 4:
        # Noah-MP's snow stack.  Three (NSNOW, ny, nx) arrays plus one
        # (NSNOW + n_soil, ny, nx); missing them under-counts by
        # (4*NSNOW + n_soil)*ny*nx FP32 words per domain.
        from woof.core.noahmp_runtime import (
            NOAHMP_STATE_SNOWSOIL_3D, NOAHMP_STATE_SNOW_3D, NSNOW,
        )
        for name in NOAHMP_STATE_SNOW_3D:
            shapes[f"fields/{name}"] = (NSNOW, ny, nx)
        for name in NOAHMP_STATE_SNOWSOIL_3D:
            shapes[f"fields/{name}"] = (NSNOW + n_soil, ny, nx)
    if int(getattr(cfg, "sf_urban_physics", 0)) > 0:
        # The urban canopy model's Registry arrays, all carried in
        # ``fields`` (woof/core/urban_state.py).  BEP+BEM's wall and floor
        # stacks run to thousands of words per column, so leaving them out
        # would admit a run that cannot allocate.
        from woof.core.urban_state import urban_array_shapes
        for name, shape in urban_array_shapes(
                cfg, urban_columns=urban_columns).items():
            shapes[f"fields/{name}"] = shape

    from woof.core.physics_inventory import (
        PBL_OPTIONAL_TENDENCIES, terrain_drag_array_shapes)
    shapes.update(terrain_drag_array_shapes(cfg))

    stacks = ["pbl_tendencies", "radiation_tendencies", "cumulus_tendencies"]
    reuse_pbl = physics_reuses_pbl_composition(cfg)
    if (radiation_enabled(cfg) or cfg.cu_physics) and not reuse_pbl:
        stacks.append("tendencies")  # composed target, physics.py:426-428
    for stack in stacks:
        shapes[f"{stack}/ru"] = (nz, ny, nx + 1)
        shapes[f"{stack}/rv"] = (nz, ny + 1, nx)
        for comp in ("rtheta", "rqv", "rqc"):
            shapes[f"{stack}/{comp}"] = m
    # The logical zero arrays keep their restart names, but radiation's
    # ru/rv/rqv/rqc share one contiguous backing; held KF's ru/rv share
    # another. Price the largest shape of each backing exactly once.
    for stack, components in (
            ("radiation_tendencies", ("ru", "rv", "rqv", "rqc")),
            ("cumulus_tendencies", ("ru", "rv") if cfg.cu_physics == 1 else ())):
        if components:
            largest = max(components, key=lambda comp: math.prod(shapes[f"{stack}/{comp}"]))
            for comp in components:
                if comp != largest:
                    shapes.pop(f"{stack}/{comp}")
    for component in PBL_OPTIONAL_TENDENCIES.get(cfg.bl_pbl_physics, ()):
        shapes[f"pbl_tendencies/{component}"] = m
        if (radiation_enabled(cfg) or cfg.cu_physics) and not reuse_pbl:
            shapes[f"tendencies/{component}"] = m
    if cfg.cu_physics:
        # Mixed-phase KF returns QR/QI/QS independently.  At bldt=0 the fresh
        # PBL stack is the composed target; positive cadence keeps the
        # historical separate target.
        target = "pbl_tendencies" if reuse_pbl else "tendencies"
        for comp in ("rqr", "rqi", "rqs"):
            shapes[f"cumulus_tendencies/{comp}"] = m
            shapes[f"{target}/{comp}"] = m
    if cfg.bl_pbl_physics and cfg.mp_physics in PBL_RQI_MICROPHYSICS:
        # Mixed-phase states carry qi; YSU returns dqi and rqi survives
        # composition (physics.py:263-264, :681-692).  mp=28 belongs by
        # Registry/Registry.EM_COMMON:3036 -- the thompsonaero package
        # declares moist:qv,qc,qr,qi,qs,qg, so WRF's F_QI is true and
        # module_first_rk_step_part1.F:1112's CALL pbl_driver hands
        # moist(...,P_QI), F_QI=F_QI (:1199) to the PBL driver.  This budget
        # mirrors physics._pbl_optional_tendency_components; the two sets
        # must stay identical, so they are now ONE constant.
        shapes["pbl_tendencies/rqi"] = m
        if (radiation_enabled(cfg) or cfg.cu_physics) and not reuse_pbl:
            shapes["tendencies/rqi"] = m

    if cfg.bl_pbl_physics == SASE_PBL_SCHEME:
        # The composed target aliases this held z-face component.
        shapes["pbl_tendencies/rw"] = (nz + 1, ny, nx)

    if physics_retains_ysu_output(cfg):
        # Positive cadence preserves the historical retained diagnostic.
        # At bldt=0 the same arrays are step transients itemized below.
        from woof.core.physics_inventory import pbl_raw_rate_names
        carried_raw = set(pbl_raw_rate_names(cfg))
        for name in _YSU_3D:
            if name not in carried_raw:
                shapes[f"last_ysu/{name}"] = m
        for name in _YSU_2D:
            shapes[f"last_ysu/{name}"] = s2

    shapes["rthratenlw"] = m
    shapes["rthratensw"] = m
    if radiation_enabled(cfg):
        shapes["rthraten"] = m
    if WRF_EXACT and radiation_enabled(cfg):
        shapes["_wrf_rthraten"] = m
    if cfg.cu_physics in CUMULUS_ADVECTIVE_FORCING_SCHEMES:
        # Raw PBL rates held for GF/New Tiedtke across calls, tile swaps
        # and checkpoints; these are distinct from coupled tendencies.
        shapes["gf_rthblten"] = m
        shapes["gf_rqvblten"] = m
    from woof.core.physics_inventory import (PBL_SHARED_FORCING,
                                                pbl_raw_rate_names)
    for name in pbl_raw_rate_names(cfg):
        if (name in PBL_SHARED_FORCING
                and cfg.cu_physics in CUMULUS_ADVECTIVE_FORCING_SCHEMES):
            continue  # the held GF/New Tiedtke buffer is the canonical owner
        shapes[f"pbl_raw_rates/{name}"] = m
    shapes["_pending_rainbl"] = s2
    if cfg.bl_pbl_physics == SASE_PBL_SCHEME and cfg.sase_flux_diag:
        # SPLIT SUBGRID-FLUX DIAGNOSTIC (physics.py PhysicsDriver
        # __init__): four z-FACE (nz+1, ny, nx) FP32 driver persistents
        # holding the venting and K_v channels of the closure's vertical
        # subgrid moisture/heat flux.  RESIDENT, not step transient --
        # output reads them after the step ends -- which is why they are
        # itemized here and NOT in sase_workspace_phases, whose
        # solve-phase byte pin is therefore unmoved.  Gated on the key,
        # so every estimate that does not set it is byte-identical by
        # construction.
        for name in ("fqv_vent", "fqv_diff", "fth_vent", "fth_diff"):
            shapes[f"sase_flux_diag/{name}"] = (nz + 1, ny, nx)
    if cfg.hmix_k_diag and hmix_k_diag_names(cfg):
        # HORIZONTAL EDDY-VISCOSITY DIAGNOSTIC (physics.py PhysicsDriver
        # __init__): two mass-grid (nz, ny, nx) FP32 driver persistents.
        # RESIDENT for the flux diagnostic's reason -- output reads them
        # after the step ends -- and gated on the key, so every estimate
        # that does not set it is byte-identical by construction.  The
        # producer's own K field is NOT counted again here: the km_opt=4
        # smag_km/smag_kh scratch and the closure's step-transient km_h
        # are already accounted where they are allocated; these two are
        # the published copies.
        for name in hmix_k_diag_names(cfg):
            shapes[f"hmix_k_diag/{name}"] = m
    # Active microphysics diagnostics alias the carrying mp_* scratch set.
    # With mp_physics=0 there is no canonical set, so the historical three
    # zero-filled output-plumbing arrays remain driver-owned.
    if not microphysics_scratch_slots(cfg.mp_physics):
        for comp in ("rainnc", "rainncv", "sr"):
            shapes[f"microphysics/{comp}"] = s2
    from woof.core.physics_inventory import cumulus_state_layout
    shapes.update({f"fields/{name}": shape for name, (shape, _dtype) in cumulus_state_layout(cfg).items()})
    if cfg.cu_physics == 1:
        shapes["cumulus/w0avg"] = m  # kf.py:174, WRF Registry r-flagged
        # Once-per-process lru_cache device LUT (kf.py load_kf_table +
        # _device_table): temperature/qsat (250,220) + thetae_base (220,)
        # + log_ratio (200,) FP32 = 441,680 B.  Counted on the cumulus
        # domain -- exactly one in every ratified config (cu is d01-only);
        # the cache itself is process-wide.
        shapes["cumulus/kf_lut_temperature"] = (250, 220)
        shapes["cumulus/kf_lut_qsat"] = (250, 220)
        shapes["cumulus/kf_lut_thetae_base"] = (220,)
        shapes["cumulus/kf_lut_log_ratio"] = (200,)
    ra_lw_physics, ra_sw_physics = radiation_scheme_ids(cfg)
    if ra_lw_physics or ra_sw_physics:
        shapes["radiation/latitude_deg"] = s2
        shapes["radiation/longitude_deg"] = s2
    if 4 in (ra_lw_physics, ra_sw_physics):
        # RRTMGPRadiation.__post_init__ ozone climatology profiles
        # (rrtmgp.py:1076-1077): two 60-level FP32 device arrays, 480 B.
        shapes["radiation/_ozone_logp"] = (60,)
        shapes["radiation/_ozone_vmr"] = (60,)
    if ra_lw_physics in (1, 4):
        # The OLR publication buffer (physics.py PhysicsDriver __init__):
        # one resident (ny, nx) FP32 driver persistent holding WRF's TOA
        # outgoing longwave for output, allocated when the attached
        # longwave adapter declares ``publishes_olr``.  BOTH built-in 4/4
        # adapters declare it, and so does the 1/1 WRF RRTM + Dudhia
        # composition, which is what this gate reproduces; a
        # caller-injected adapter that does not leaves this counted but
        # unallocated, and over-counting by one 2-D field is the safe
        # direction for a VRAM estimate.
        shapes["olr"] = s2
    if int(getattr(cfg, "aer_opt", 0) or 0) == 1:
        shapes["aodtot"] = s2
        shapes["radiation/aerodm"] = (int(cfg.ny)*int(cfg.nx), 12, 12, 6)
        # Classic LW call/packing arrays are priced as transients below.
    if ra_lw_physics in (1, 4) or ra_sw_physics == 4:
        # The CLDFRA publication buffer (physics.py PhysicsDriver
        # __init__): one resident (nz, ny, nx) FP32 driver persistent
        # holding the radiation cloud fraction for output, allocated when
        # the attached radiation callable declares ``publishes_cldfra``.
        # RTE+RRTMGP and legacy RRTMG declare it on either spectrum, the
        # classic RRTM longwave on its own; Dudhia alone computes no
        # fraction.  Over-counting a caller-injected adapter that does not
        # declare it is the safe direction for a VRAM estimate.
        shapes["cldfra"] = m
    if int(getattr(cfg, "ifire", 0) or 0) == 2:
        shapes.update(fire_array_shapes(cfg))
    return shapes


#: Halo of the coarse spotting accumulation grids (sfire_spotting), cells.
_SPOTTING_HALO = 4


def fire_array_shapes(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """The SFIRE coupler's resident arrays at ``ifire`` 2, as priced shapes.

    The breakage this prevents: ``PhysicsDriver.fire`` (the FireCoupler) and
    ``fire_tendencies`` were serialized by the restart manifest but priced by
    nothing, so a refined fire grid (sr_x * sr_y fine cells per column, 43
    padded fine planes) could be admitted on a card it does not fit.

    Every count is the inventory of an actual FireCoupler read on an RTX
    4090 (tests/test_sfire_preflight_price_gpu.py re-reads it and requires
    these shapes to equal the allocated bytes): 43 padded fine float32
    planes; coarse planes (20 without spotting, 12 with it, whose particle
    owner replaces eight); two coarse 3-D heating tendencies; the two
    staggered fire winds; the moisture model's three (nfmc, ny, nx) class
    fields, one (2, ny, nx) pair and one scalar; and with spotting the
    unpadded fine landing field, eight halo-4 coarse accumulators, eleven
    particle vectors of fs_array_maxsize and a three-word counter.  Integer
    arrays are int32, so every entry prices at four bytes a word.
    """
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    sr_x, sr_y = int(cfg.sr_x), int(cfg.sr_y)
    fine_y, fine_x = ny * sr_y, nx * sr_x
    spotting = int(getattr(cfg, "fs_firebrand_gen_lim", 0) or 0) > 0
    shapes = {
        "fire/fine_padded": (43, fine_y + 2, fine_x + 2),
        "fire/coarse": (12 if spotting else 20, ny, nx),
        "fire/heating": (2, nz, ny, nx),
        "fire/uah": (ny, nx + 1),
        "fire/vah": (ny + 1, nx),
        "fire/moisture_classes": (3, int(cfg.nfmc), ny, nx),
        "fire/moisture_pair": (2, ny, nx),
        "fire/lfn_time": (1,),
        # PhysicsTendencies.zeros: ru, rv and three mass-point scalars.
        "fire_tendencies/ru": (nz, ny, nx + 1),
        "fire_tendencies/rv": (nz, ny + 1, nx),
        "fire_tendencies/scalars": (3, nz, ny, nx),
    }
    if spotting:
        halo = 2 * _SPOTTING_HALO
        shapes.update({
            "fire/spotting_landing": (fine_y, fine_x),
            "fire/spotting_coarse": (8, ny + halo, nx + halo),
            "fire/spotting_particles": (11, int(cfg.fs_array_maxsize)),
            "fire/spotting_counter": (3,),
        })
    return shapes


def _perimeter_count(ny: int, nx: int, width: int) -> int:
    """Cells in ``width`` nested frames (lateral_bc.py:220-223 verbatim)."""
    return sum(2 * (nx - 2 * d) + 2 * max(ny - 2 * d - 2, 0)
               for d in range(width))


def lbc_host_series_bytes(cfg: RunConfig, intervals: int, *,
                          source=None) -> int:
    """HOST bytes of a specified domain's lateral forcing series.

    ``LateralBoundaries`` keeps value and tendency in float64, four sides,
    every field, every interval: :func:`lbc_interval_values` elements per
    interval at eight bytes each.  A streamed domain keeps this series on
    the host for the whole run and cuts each tile's edge from it, so it is
    part of what a streamed forecast holds in host RAM.  Zero for a domain
    that carries no tabulated series (a nest's forcing is its parent's
    rolling device frame).

    ``source`` names the forcing source (or is the mapping document a
    mapped route prepares from); the analysed hydrometeors its table row
    publishes ride the series too
    (:func:`woof.boundary_fields.source_boundary_species`), so a series
    priced without it is a HRRR-forced run's host claim short by those
    tables.
    """
    if not bool(getattr(cfg, "specified", False)) or bool(
            getattr(cfg, "nested", False)):
        return 0
    from woof.boundary_fields import source_boundary_species
    return 8 * lbc_interval_values(
        cfg, boundary_species=source_boundary_species(source)) * max(
            0, int(intervals))


def lbc_intervals(run_seconds: float, forcing_interval_seconds: float, *,
                  retained_intervals: int | None = None) -> int:
    """Root boundary count, using the retained input window when known."""
    if retained_intervals is not None:
        if (isinstance(retained_intervals, bool)
                or not isinstance(retained_intervals, int) or retained_intervals < 1):
            raise ValueError("retained forcing intervals must be a positive integer")
        return retained_intervals
    return max(1, math.ceil(run_seconds / float(forcing_interval_seconds)))


def mynn_pbl_column_chunk(cfg: RunConfig, *, tile_buffer: bool = False) -> int:
    """Columns per MYNN call for this domain.

    The process's derived chunk, capped by the domain: a 50x20 verification
    grid has 1,000 columns and asks for one chunk of 1,000, while a 600x600
    nest asks for as many chunks as the derived width needs.  The workspace
    is therefore the same size on every domain at or above that width, which
    is the property that lets a launch gate refuse a configuration before it
    allocates.

    :func:`woof.core.mynn_pbl_scratch.resolve_mynn_column_chunk` settles the
    width once per process and memoises it, so THIS function, the shared
    arena it sizes and the solver that walks the domain all name the same
    number.  It is the widest width the card's memory admits up to the
    98,304-column cap (the 2026-09-30 sweep on the level-major kernels:
    wider is faster up to one chunk per domain), never below 8,192, and
    the cap itself off a card without a target memory sample. A domain
    fit supplies its selected card's memory sample and uses the same
    run-width policy without consulting the host's memo. An operator
    override replaces it. It is NOT ``MYNN_PBL_COLUMN_CHUNK_FLOOR``, which bounds the
    derivation that rides the receipt.

    ``tile_buffer=True`` prices a streamed tile buffer, which walks
    :func:`woof.core.mynn_pbl_scratch.resolve_mynn_tile_column_chunk`
    instead: each buffer holds its own workspace.
    """
    from woof.core.mynn_pbl_scratch import (
        resolve_mynn_column_chunk, resolve_mynn_tile_column_chunk)
    resolve = (resolve_mynn_tile_column_chunk if tile_buffer
               else resolve_mynn_column_chunk)
    return max(1, min(resolve(int(cfg.nz)), int(cfg.ny) * int(cfg.nx)))


def mynn_pbl_scratch_slots(cfg: RunConfig, *, tile_buffer: bool = False
                           ) -> dict[str, tuple[int, ...]]:
    """Every ``bl_pbl_physics=5`` scratch slot and its exact shape.

    Split out of :func:`scratch_slot_registry` so the MYNN workspace can be
    priced, diffed and gated on its own; ``tests/test_mynn_pbl_scratch.py``
    checks this against the slot names the solver actually requests.
    """
    from woof.core.mynn_pbl_scratch import (
        mynn_pbl_flag_shapes, mynn_pbl_index_shapes,
        mynn_pbl_scratch_shapes, mynn_pbl_tendency_field_shapes,
    )
    chunk = mynn_pbl_column_chunk(cfg, tile_buffer=tile_buffer)
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    slots: dict[str, tuple[int, ...]] = {}
    slots.update(mynn_pbl_scratch_shapes(chunk, nz,
                                       bl_mynn_version=cfg.bl_mynn_version))
    slots.update(mynn_pbl_index_shapes(chunk, nz))
    slots.update(mynn_pbl_flag_shapes())
    slots.update(mynn_pbl_tendency_field_shapes(nz, ny, nx))
    return slots


def mynn_pbl_scratch_bytes_for(cfg: RunConfig) -> int:
    """Device bytes the MYNN workspace occupies for this domain."""
    total = 0
    for shape in mynn_pbl_scratch_slots(cfg).values():
        values = 1
        for extent in shape:
            values *= int(extent)
        total += values * 4
    return total


def _wsm_family_selectors() -> frozenset[int]:
    from woof.core.wsm_family import family_selectors
    return family_selectors()


def _wsm_family_preparation_slots() -> tuple[str, ...]:
    from woof.core.wsm_family import preparation_slots
    return tuple(slot for mp in _wsm_family_selectors()
                 for slot in preparation_slots(mp))


def scratch_slot_registry(cfg: RunConfig, *,
                          n_lbc_intervals: int = 0,
                          tile_buffer: bool = False
                          ) -> dict[str, tuple[int, ...]]:
    """Static registry: every named ``DomainState.scratch`` slot and its
    exact shape formula, keyed by the RunConfig features that create it.

    The completeness test (tests/test_preflight.py) AST-scans every
    ``scratch(...)`` call site in ``gpuwm/`` against this registry -- an
    unclassified slot is an error.  Sources cited per family.  Child
    ``nest_*`` slots live in :func:`nest_allocation_manifest` (the F4
    manifest), not here.
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    m = (nz, ny, nx)
    xs = (nz, ny, nx + 1)
    ys = (nz, ny + 1, nx)
    fl = (nz + 1, ny, nx)
    s2 = (ny, nx)
    slots: dict[str, tuple[int, ...]] = {}
    from woof.core.hail import scratch_shapes as hail_scratch_shapes
    slots.update(hail_scratch_shapes(cfg))
    from woof.core.output_diagnostics import scratch_shapes as output_shapes
    slots.update(output_shapes(cfg))
    if cfg.lightning_option:
        from woof.lightning_contract import LIGHTNING_OPTIONS
        slots.update({"ltng_" + k: s2 for k in LIGHTNING_OPTIONS[cfg.lightning_option]["fields"]})
        slots.update(ltng_iccg_in_num=(12, ny, nx), ltng_iccg_in_den=(12, ny, nx),
                     ltng_ltngacttime=(1,), ltng_calendar=(6,), ltng_lat=s2, ltng_lon=s2)
        if cfg.lightning_option == 11:
            slots["ltng_ktop_deep"] = s2

    # dycore.py rk stage fluxes (:102, :154-155) + integration health
    # (:1471-1474; nblocks = min(256, max(1, ceil(largest/256)))).
    slots.update(rk_ww=fl, rk_ru=xs, rk_rv=ys)
    largest = max(nz * ny * (nx + 1), (nz + 1) * ny * nx)
    nblocks = min(256, max(1, (largest + 255) // 256))
    slots["integration_health_partial"] = (nblocks, 9)
    slots["integration_health_result"] = (8,)
    # health.py StateHealthValidator: descriptor metadata and compact result.
    # These are the fixed descriptor-ceiling footprint, independent of case
    # size; inventory additions fail at that explicit cap (MAX_HEALTH_FIELDS,
    # or FSBM_HEALTH_FIELDS on a FAST_SBM domain -- health.py).
    from woof.core.health import health_field_ceiling_for_config
    ceiling = health_field_ceiling_for_config(cfg)
    health_words = (ceiling * 2,)
    slots.update(
        integration_health_field_ptr=health_words,
        integration_health_aux_ptr=health_words,
        integration_health_field_size=health_words,
        integration_health_bounds=(ceiling, 2),
        integration_health_flags=(ceiling,),
        integration_health_planes=(ceiling,),
        integration_health_status_bits=health_words,
        integration_health_rim=(ceiling, 3),
        integration_health_validation=(4,),
    )
    if int(getattr(cfg, "zadvect_implicit", 0) or 0) > 0:
        # ieva.py: the explicit/implicit eta mass-flux split (reused by the
        # scalar split after the dynamics solves) and the column masses.
        slots.update(ieva_wwe=fl, ieva_wwi=fl, ieva_mut=s2,
                     ieva_mut_old=s2, ieva_mut_new=s2)
    # advection.py:161-163.
    slots.update(adv_ru=xs, adv_rv=ys, adv_rw=fl)
    # Acoustic coefficients also stage horizontal w diffusion primitives.
    # Diffusion completes before calc_coefs rewrites every consumed element.
    slots.update(acoustic_mu_pp_old=s2, acoustic_th_pp_old=m,
                 acoustic_c2a=m, acoustic_a=fl, acoustic_alpha=fl,
                 acoustic_gamma=fl)
    from woof.wrf_exact import ENABLED as wrf_exact
    if wrf_exact:
        slots["acoustic_wrf_muts"] = s2
        # Strict mode's own allocations: the step-carried acoustic mass
        # (dycore.py WRF_MUTS_SLOT, WRF_MU_HALO_SLOT on state.mup's plane),
        # calc_ww_cp's coupled face fluxes (dycore.py:_calc_ww_cp_fluxes)
        # and the one held species source WRF's rk_update_scalar forms
        # (moist.py:_advance_species_wrf).
        slots.update(wrf_exact_muts=s2, wrf_exact_mu_halo=s2,
                     rk_ww_ru=xs, rk_ww_rv=ys)
        if cfg.moist:
            slots["moist_sc_t"] = m
    if cfg.moist and cfg.moist_cq:
        # acoustic.py:prepare_moist_cq.  These stage-fixed face arrays alias
        # the disjoint advection-only adv_ru/rv/rw arena backings below.
        slots.update(acoustic_cqu=xs, acoustic_cqv=ys, acoustic_cqw=fl)
    if cfg.open_x:
        slots["openbc_upp_faces"] = (nz, ny, 2)     # acoustic.py:121
    if cfg.open_y:
        slots["openbc_vpp_faces"] = (nz, 2, nx)     # acoustic.py:125
    if cfg.emdiv > 0.0:
        slots.update(acoustic_mudf=s2)

    chem_on = bool(getattr(cfg, "chem_sets", ""))
    if cfg.moist or cfg.km_opt == 2 or chem_on or (cfg.bl_pbl_physics == 5 and cfg.bl_mynn_tkeadvect):
        # dycore.py acoustic time-averaged mass fluxes (moist scalars,
        # the km_opt=2 TKE carrier and the chem rows advect with them;
        # dycore.step accumulates them for a chem state even when dry).
        slots.update(rk_ru_m=xs, rk_rv_m=ys, rk_ww_m=fl)
        slots["moist_rq_t"] = m                     # moist.py:247
        allocated = state_array_shapes(cfg)
        if "qi" in allocated and "qg" not in allocated:
            # woof/core/moist.py::absent_mass_plane.  The two fused
            # moist-array sums (calc_cq, slow_buoyancy's q_total) select
            # their species by an integer mode with no one-ice-mass arm,
            # so a scheme with qi and no qs/qg hands the absent pair this
            # single shared zero plane.  P3 (mp=50) is the only such
            # scheme today, but the RUNTIME guards are presence-based, so
            # this predicate is too -- a cfg-keyed list here would
            # under-count memory for the next one-ice-category port
            # instead of failing.  Spelled as a literal because this
            # module must stay importable without cupy;
            # tests/test_p3_port.py pins it to moist.ABSENT_MASS_SLOT.
            slots["moist_absent_mass"] = m
        pd = ((cfg.moist and cfg.moist_adv_opt == 1) or cfg.km_opt == 2
              or (cfg.bl_pbl_physics == 5 and cfg.bl_mynn_tkeadvect
                  and cfg.scalar_adv_opt == 1)) and not (cfg.open_x or cfg.open_y)
        # woof/core/chem_transport.py: the chem rows' limited final stage
        # borrows the same six face buffers and fold copy, positive
        # definite (chem_adv_opt 1, not on open domains) or monotonic (2).
        chem_limited = chem_on and (
            cfg.chem_adv_opt == 2
            or (cfg.chem_adv_opt == 1 and not (cfg.open_x or cfg.open_y)))
        if pd or chem_limited:
            slots.update(pd_fxl=xs, pd_fxc=xs, pd_fyl=ys, pd_fyc=ys,
                         pd_fzl=fl, pd_fzc=fl)      # moist.py:283-288
            slots["moist_pd_q0"] = m                # moist.py:197
        if chem_on and cfg.chem_adv_opt == 2:
            from woof.core.chem_context import (
                MONO_IMPLICIT_SLOT, MONO_SCRATCH_SLOTS,
            )
            for slot in MONO_SCRATCH_SLOTS:
                slots[slot] = m
            slots[MONO_IMPLICIT_SLOT] = fl
        if cfg.specified:
            from woof.boundary_fields import potential_external_scalar_fields
            for name in potential_external_scalar_fields(cfg):
                slots[f"lbc_{name}_held"] = m

    if cfg.nwp_diagnostics == 1 or cfg.gsd_diagnostics == 1:
        # woof/core/uh_diag.py: the serialized UP_HELI_MAX running max
        # (eagerly allocated by DomainState.__init__) plus two per-launch
        # work planes (column UH and the use_column flags).
        slots.update(up_heli_max=s2, uh_diag_col=s2, uh_diag_use=s2)
    if cfg.nwp_diagnostics == 1:
        # The two consumer-owned tracking windows (woof/core/uh_diag.py:
        # TRACKER_WINDOW_SLOTS).  Priced on the same gate that allocates
        # them, because they are allocated whenever the diagnostic runs
        # rather than only when a follow/spawn block is declared: the
        # relocation and spawn tables live on the ExperimentConfig, which
        # DomainState.__init__ does not see, and inventing a RunConfig
        # field to carry them would move the frozen-config surface for
        # two (ny, nx) FP32 planes.
        slots.update(uh_follow_window=s2, uh_spawn_window=s2)

    if cfg.mp_physics == 1:
        # microphysics.py:143-163 (Kessler prep + accumulators).
        slots.update(mp_th=m, mp_rho=m, mp_pii=m, mp_z=m, mp_dz8w=m,
                     mp_z8w=fl, mp_rainnc=s2, mp_rainncv=s2,
                     mp_kessler_sr=s2)
    if cfg.mp_physics == 2:
        slots.update(lin_theta=m,lin_rho=m,lin_pii=m,lin_dz=m,lin_z=m,lin_z8w=fl,
                     mp_rainnc=s2,mp_rainncv=s2,mp_snownc=s2,mp_snowncv=s2,
                     mp_graupelnc=s2,mp_graupelncv=s2,mp_sr=s2,refl_t=m,refl_10cm=m)
    if cfg.mp_physics == 5:
        # woof/core/fer_hires.py::apply: the WRF preparation bracket, the
        # three canonical surface slots and the scheme's own reflectivity.
        slots.update(mp_th=m, mp_rho=m, mp_pii=m, mp_dz8w=m, mp_z8w=fl,
                     mp_rainnc=s2, mp_rainncv=s2, mp_sr=s2, refl_10cm=m)
    if cfg.mp_physics == 3:
        # wsm3.py preparation (w on the lower faces included), persistent
        # precipitation, and the due reflectivity with its phase-split
        # inputs (woof.core.refl.wsm3_precipitation_phases).
        slots.update(wsm3_theta=m, wsm3_rho=m, wsm3_pii=m, wsm3_dz=m,
                     wsm3_z8w=fl, wsm3_w=m, mp_rainnc=s2, mp_rainncv=s2,
                     mp_snownc=s2, mp_snowncv=s2, mp_sr=s2, refl_t=m,
                     refl_10cm=m, refl_wsm3_rain=m, refl_wsm3_snow=m,
                     refl_wsm3_graupel=m)
    from woof.core.column_schemes import COLUMN_SCHEMES
    row = COLUMN_SCHEMES.get(cfg.mp_physics)
    if row is not None:
        slots.update(mpcol_theta=m, mpcol_rho=m, mpcol_pii=m, mpcol_dz=m,
                     mpcol_z8w=fl, refl_t=m, refl_10cm=m)
        slots.update({name: s2 for name in row.surface})
    if cfg.mp_physics == 6:
        # wsm6.py preparation, persistent precipitation and due reflectivity.
        slots.update(wsm6_theta=m, wsm6_rho=m, wsm6_pii=m, wsm6_dz=m,
                     wsm6_z8w=fl, mp_rainnc=s2, mp_rainncv=s2,
                     mp_snownc=s2, mp_snowncv=s2, mp_graupelnc=s2,
                     mp_graupelncv=s2, mp_sr=s2, refl_t=m, refl_10cm=m)
    if cfg.mp_physics == 106:
        # wsm6._apply_wsm6r: the shared preparation, fallk's (3, nz, ny, nx)
        # fall-speed scratch, RAINNC/RAINNCV/SR and due reflectivity.
        slots.update(wsm6r_theta=m, wsm6r_rho=m, wsm6r_pii=m, wsm6r_dz=m,
                     wsm6r_z8w=fl, wsm6r_work=(3,) + m, mp_rainnc=s2,
                     mp_rainncv=s2, mp_sr=s2, refl_t=m, refl_10cm=m)
    if cfg.mp_physics in _wsm_family_selectors():
        # woof/core/wsm_family.py::apply -- the WRF driver preparation
        # under the row's name, one (nz+1) interface plane, the
        # accumulator pair of every precipitating class the row carries,
        # SR, and the due reflectivity.
        from woof.core.wsm_family import row as _wsm_row
        _r = _wsm_row(cfg.mp_physics)
        slots.update({f"{_r.name}_theta": m, f"{_r.name}_rho": m,
                      f"{_r.name}_pii": m, f"{_r.name}_dz": m,
                      f"{_r.name}_z8w": fl, "mp_sr": s2,
                      "refl_t": m, "refl_10cm": m})
        for _cls in _r.precip:
            slots[f"mp_{_cls}nc"] = s2
            slots[f"mp_{_cls}ncv"] = s2
    if cfg.mp_physics == 6 and cfg.ny == 1:
        # mpas_column_batch.py:769-773 (run_phase2 adapter pair):
        # alt = 1/rho_dry and php = z_interface*g, fully rewritten by every
        # phase-2 call before the microphysics dispatch reads them.  The
        # column-batch seam is the only allocator and always builds its
        # RunConfig with ny == 1 (one row of columns), so a plane-shaped
        # WSM6 forecast neither allocates nor is priced for either slot.
        slots.update(physics_column_alt=m, physics_column_php=fl)
    if cfg.mp_physics == 16:
        # wdm6.py preparation, persistent precipitation and due reflectivity.
        # Same slot shape as mp=6: WDM6's three extra moments are STATE, not
        # scratch, so nothing here grows with the double-moment warm rain.
        slots.update(wdm6_theta=m, wdm6_rho=m, wdm6_pii=m, wdm6_dz=m,
                     wdm6_z8w=fl, wdm6_count_status=(1,), mp_rainnc=s2, mp_rainncv=s2,
                     mp_snownc=s2, mp_snowncv=s2, mp_graupelnc=s2,
                     mp_graupelncv=s2, mp_sr=s2, refl_t=m, refl_10cm=m)
    if cfg.mp_physics == 8 and not WRF_EXACT:
        # microphysics.py:_apply_thompson_default preparation,
        # persistent precipitation, output-due private graupel-number shadow,
        # and Thompson REFL_10CM staging.  The three reference fields are
        # deliberately distinct registry entries even where the runtime can
        # lifetime-alias their arena backings.
        slots.update(
            mp_th=m, mp_pii=m, mp_dz8w=m, mp_z8w=fl,
            mp_thompson_temperature=m,
            mp_thompson_frozen_reference_density=m,
            mp_thompson_frozen_reference_temperature=m,
            mp_thompson_rain_reference_density=m,
            mp_thompson_snow_melt_marker=m,
            mp_thompson_graupel_melt_marker=m,
            mp_thompson_snow_velocity_boost=m,
            mp_thompson_graupel_number_shadow=m,
            mp_rainnc=s2, mp_rainncv=s2, mp_snownc=s2, mp_snowncv=s2,
            mp_graupelnc=s2, mp_graupelncv=s2, mp_sr=s2,
            refl_t=m, refl_10cm=m,
            # WRF's no-microphysics column flag (:2020), taken on the entry
            # state and read by the terminal apply for the :3974 vapour
            # floor: one float32 per column.
            mp_thompson_micro_columns=s2,
        )
        if getattr(cfg, "thompson_cloud_surface_booking", False) is True:
            # Correct mode WB1 (woof/core/thompson_cloud_booking.py): the
            # settled-cloud total and its per-call amount.
            slots.update(mp_cloudnc=s2, mp_cloudncv=s2)
    elif cfg.mp_physics == 8:
        # GPUWM_WRF_EXACT=1 (microphysics._apply_thompson_strict):
        # woof/core/thompson_classic.py:apply_classic, the slots it draws,
        # from its own lists: theta, Exner, temperature, layer depths, the
        # pre-microphysics theta, the private graupel number, the
        # plane-stacked tendency / working / column buffers, the int32 cell
        # and column flags, persistent precipitation and REFL_10CM staging.
        from woof.core import thompson_classic as tcl
        for name, planes in tcl.SCRATCH_VOLUME_SLOTS:
            slots[name] = (planes * cfg.nz, ny, nx)
        for name, planes in tcl.SCRATCH_INTERFACE_SLOTS:
            slots[name] = (planes * (cfg.nz + 1), ny, nx)
        for name in tcl.SCRATCH_INT_VOLUME_SLOTS:
            slots[name] = m
        for name in tcl.SCRATCH_INT_SURFACE_SLOTS:
            slots[name] = s2
        slots.update(
            mp_rainnc=s2, mp_rainncv=s2, mp_snownc=s2, mp_snowncv=s2,
            mp_graupelnc=s2, mp_graupelncv=s2, mp_sr=s2, refl_10cm=m,
        )
        if getattr(cfg, "thompson_cloud_surface_booking", False) is True:
            # Correct mode WB1 (woof/core/thompson_cloud_booking.py): the
            # settled-cloud total and its per-call amount.
            slots.update(mp_cloudnc=s2, mp_cloudncv=s2)
    if cfg.mp_physics == 30:
        from woof.core.fast_sbm_contract import STATE_FIELDS
        slots.update(mp_th=m,mp_rho=m,mp_pii=m,mp_dz8w=m,mp_z8w=fl,
            mp_rainnc=s2,mp_rainncv=s2,mp_snownc=s2,mp_snowncv=s2,
            mp_graupelnc=s2,mp_graupelncv=s2,mp_sr=s2)
        ni,nk,nj=nx+2,nz+1,ny+2
        columns=(nx-2*max(int(cfg.spec_zone),0))*(ny-2*max(int(cfg.spec_zone),0)) if cfg.specified or cfg.nested else nx*ny
        n=ni*nk*nj
        slots.update(fsbm_dims=(18,),fsbm_pointers=(len(STATE_FIELDS),),
            fsbm_work=(7*n+2*columns*(nz+2),),fsbm_cloud_base=(columns,),
            fsbm_tally=(columns,17),fsbm_status=(columns,))
        surfaces={"xland","rainnc","rainncv","snownc","snowncv","graupelnc","graupelncv"}
        for name in STATE_FIELDS:
            slots["fsbm_frame_"+name]=((143,nj,nk,ni) if name=="chem" else
                ((nj,ni) if name in surfaces else (nj,nk,ni)))
    if cfg.mp_physics == 50:
        # P3 one-category (woof/core/p3.py::apply).  The WRF preparation
        # bracket, five precipitation slots and the reflectivity staging,
        # plus P3's three own ice diagnostics.  NO graupel accumulators:
        # P3 has a single ice category and its driver arm passes no
        # GRAUPELNC (module_microphysics_driver.F:1590-1595).
        #
        # vmi3d/di3d/rhopo3d are WRF grid STATE for mp=50
        # (Registry.EM_COMMON:3038) but nothing downstream of the scheme
        # reads them in woof yet, so they are registered as scratch rather
        # than promoted to DomainState fields -- the accurate place for an
        # output the model computes and does not consume.  They are
        # registered rather than left unclassified so the allocation gate
        # sees their true size.
        #
        # THE DEVICE COMPANIONS (the CUDA port, 2026-08-29).  Eighteen
        # (nz, ny, nx) float32 arrays: twelve that carry values between the
        # kernels and six of sedimentation workspace the three sedimentation
        # steps share, plus the two column-scope logical flags as one
        # (2, ny, nx) slot.  That is 72 bytes per grid cell of scratch, and
        # it is registered here rather than allocated behind the gate's back
        # -- woof/core/p3.py:apply hands DomainState.scratch to
        # p3_device.make_workspace as its allocator for exactly that reason.
        # The names are woof.core.p3_device.SCRATCH_SLOTS and SEDW_SLOTS
        # with the p3_/p3_sed_ prefixes that function applies;
        # tests/test_p3_cuda.py pins the two lists against each other so a
        # slot cannot appear on one side only.
        #
        # ``mp_pii`` is deliberately GONE.  The host path built a full
        # (nz, ny, nx) Exner array; the kernels build the same factor in a
        # register where the authority builds it (module_mp_p3.F:2293), so
        # keeping the array would reserve a domain-sized allocation nothing
        # reads.  The reference arm (run.p3_backend = "reference") does not
        # use it either: it recomputes tmparr1 the same way.
        slots.update(
            mp_th=m, mp_dz8w=m, mp_z8w=fl,
            mp_rainnc=s2, mp_rainncv=s2, mp_snownc=s2, mp_snowncv=s2,
            mp_sr=s2,
            p3_vmi=m, p3_di=m, p3_rhopo=m, p3_effc=m, p3_effi=m,
            p3_nc=m, p3_ssat=m,
            p3_prt_liq=s2, p3_prt_sol=s2,
            p3_rho=m, p3_inv_rho=m, p3_qvs=m, p3_qvi=m, p3_sup=m,
            p3_supi=m, p3_rhofacr=m, p3_rhofaci=m, p3_acn=m, p3_t=m,
            p3_tmparr1=m, p3_qv_cld=m,
            p3_sed_v_q=m, p3_sed_v_n=m, p3_sed_flux_q=m, p3_sed_flux_n=m,
            p3_sed_flux_qir=m, p3_sed_flux_bir=m,
            p3_flags=(2, ny, nx),
            refl_10cm=m,
        )
    if (WRF_EXACT and cfg.mp_physics == 28
            and getattr(cfg, "thompson_version", "wrf_461") == "wrf_39_noaa"):
        # Under GPUWM_WRF_EXACT=1 the operational fork's Thompson
        # (thompson_version = "wrf_39_noaa") runs as one column unit (woof/core/kernels/thompson_fork.cu) that
        # holds the fork's tendencies and working arrays per column in its
        # column arena (priced by thompson_fork_column_workspace_bytes, not
        # here).  Its adapter
        # (microphysics_aerosol._apply_thompson_fork_call) draws exactly
        # microphysics_aerosol.FORK_SCRATCH_SLOTS: the preparation bracket,
        # the seven precipitation slots and the reflectivity staging.
        slots.update(
            mp_th=m, mp_pii=m, mp_dz8w=m, mp_z8w=fl,
            mp_rainnc=s2, mp_rainncv=s2, mp_snownc=s2, mp_snowncv=s2,
            mp_graupelnc=s2, mp_graupelncv=s2, mp_sr=s2,
            refl_10cm=m,
        )
    elif cfg.mp_physics == 28:
        # Thompson AEROSOL-AWARE (mp=28).  A deliberate near-clone of the
        # mp=8 block above -- the aerosol adapter reuses every classic
        # preparation, precipitation, melt-marker and reflectivity slot with
        # the same shape and the same lifetime -- plus the aerosol-only
        # working set.  The mp=8 block is NOT shared: mp=8's rows are pinned
        # byte-for-byte by tests/test_mp8_frozen.py (receipt R4) and a shared
        # literal is exactly how a future mp=28 slot would leak into the
        # frozen mp=8 arena layout.
        slots.update(
            mp_th=m, mp_pii=m, mp_dz8w=m, mp_z8w=fl,
            mp_thompson_temperature=m,
            mp_thompson_frozen_reference_density=m,
            mp_thompson_frozen_reference_temperature=m,
            mp_thompson_rain_reference_density=m,
            mp_thompson_snow_melt_marker=m,
            mp_thompson_graupel_melt_marker=m,
            mp_thompson_snow_velocity_boost=m,
            mp_thompson_graupel_number_shadow=m,
            mp_rainnc=s2, mp_rainncv=s2, mp_snownc=s2, mp_snowncv=s2,
            mp_graupelnc=s2, mp_graupelncv=s2, mp_sr=s2,
            refl_t=m, refl_10cm=m,
            # WRF's no-microphysics column flag (:2020), taken on the entry
            # state and read by the terminal apply for the :3974 vapour
            # floor: one float32 per column.
            mp_thompson_micro_columns=s2,
        )
        if getattr(cfg, "thompson_cloud_surface_booking", False) is True:
            # Correct mode WB1 (woof/core/thompson_cloud_booking.py): the
            # settled-cloud total and its per-call amount, as for mp=8.
            slots.update(mp_cloudnc=s2, mp_cloudncv=s2)
        # --- The aerosol working set -------------------------------------
        # WRF runs mp=28 as ONE monolithic column loop that freezes
        # nc1d/nwfa1d/nifa1d at entry (module_mp_thompson.F:1795-1812),
        # accumulates ncten/nwfaten/nifaten across widely separated regions,
        # and applies them exactly ONCE with a shared clamp (:3972-4021).
        # ArWen's fused network launchers write state in place, so the port
        # has to materialize that entry-state / accumulator split as device
        # arrays.  Every slot below is one of those, named for the launcher
        # parameter it feeds:
        #
        #   ncten/nwfaten/nifaten  -- the three shared per-kg-per-second
        #       accumulators (:1679-1681 zero them; :3972-4021 apply them).
        #       Written by the cold network, the warm network, the ncten
        #       balance limiter, the saturation adjustment, rain
        #       evaporation, cloud sedimentation and the final phase
        #       cleanup; read by exactly one terminal kernel.
        #   entry_density          -- WRF's entry rho at :1802, the density
        #       rc/nc/the ncten limiter/the terminal clamp are all formed
        #       on.  Distinct from mp_thompson_frozen_reference_density,
        #       which the saturation adjustment OVERWRITES mid-call.
        #   tau1_density           -- the REFRESHED density of :3193.
        #   nwfa_work_m3           -- the :3211 working CCN snapshot, which
        #       has no 9999E6 ceiling, uses tau+1 density and feeds
        #       activ_ncloud only.
        #   qcten                  -- WRF's cloud water accumulator
        #       (:1670 zero, :3975 apply), written by every stage that moves
        #       cloud water so the cloud is rounded once per call.  The entry
        #       cloud itself needs no copy: state.qc stays qc1d until :3975.
        #   qrten/nrten/qiten/niten -- v4.6.1 generation: WRF's rain and
        #       ice accumulators, applied once with the :4023-4053 bounds
        #       (state.qr/nr/qi/ni stay the entry state, so no copy).
        #   ni_entry               -- fork generation only: frozen ni1d,
        #       credited to ncten by the cloud-ice melt branch of the final
        #       phase cleanup (:3943-3966), whose sources write ni in place.
        #   qvten/tten             -- v4.6.1 generation: WRF's vapour and
        #       temperature accumulators (:1668-1669 zero), written by the
        #       two source networks, the condensation and the rain
        #       evaporation, each of which re-forms qv1d + DT*qvten and
        #       t1d + DT*tten as WRF does (state.qv stays qv1d until the
        #       rain evaporation has run, so no copy).
        #   condensation_rate      -- prw_vcd, held so rain evaporation can
        #       reproduce the :3502 gate that suppresses evaporation in a
        #       cell that just condensed.
        #
        # The entry aerosol and droplet-distribution launchers also expose
        # six optional probe outputs. The forecast does not consume those
        # arrays, so it requests no storage for them. Their state-updating
        # operations still run before every later consumer.
        slots.update(
            mp_thompson_aero_ncten=m,
            mp_thompson_aero_nwfaten=m,
            mp_thompson_aero_nifaten=m,
            mp_thompson_aero_entry_density=m,
            mp_thompson_aero_tau1_density=m,
            mp_thompson_aero_nwfa_work_m3=m,
            mp_thompson_aero_qcten=m,
            mp_thompson_aero_condensation_rate=m,
        )
        if getattr(cfg, "thompson_version", "wrf_461") == "wrf_39_noaa":
            # The fork keeps its in-place rain and ice: no rain or ice
            # accumulator, and the entry ice number held for the cleanup.
            slots.update(mp_thompson_aero_ni_entry=m)
        else:
            slots.update(
                mp_thompson_aero_qrten=m,
                mp_thompson_aero_nrten=m,
                mp_thompson_aero_qiten=m,
                mp_thompson_aero_niten=m,
                mp_thompson_aero_qvten=m,
                mp_thompson_aero_tten=m,
                mp_thompson_aero_qsten=m,
                mp_thompson_aero_qgten=m,
                mp_thompson_aero_ngten=m,
            )
    if cfg.mp_physics == 9:
        # milbrandt2.py::apply -- the WRF prep pair, the thirteen scratch
        # volumes the six kernels hand to one another (Part 1 leaves DZ/iDZ
        # frozen while Part 3 refreshes DE/iDE, so both pairs are named
        # slots rather than temporaries), the nine-slot precipitation row
        # its WRF driver arm binds, and the reflectivity the scheme writes
        # itself (module_microphysics_driver.F:1878 binds Zet to
        # refl_10cm, so refl_t is deliberately absent -- no generic radar
        # operator runs under mp=9).
        slots.update(my2_theta=m, my2_pii=m, my2_t=m, my2_z=m, my2_z8w=fl,
                     my2_psfc=s2,
                     my2_pres=m, my2_de=m, my2_ide=m, my2_dz=m, my2_idz=m,
                     my2_gamfact=m, my2_qsw=m, my2_qsi=m,
                     my2_qc_in=m, my2_qr_in=m, my2_nc_in=m, my2_nr_in=m,
                     mp_rainnc=s2, mp_rainncv=s2, mp_snownc=s2,
                     mp_snowncv=s2, mp_graupelnc=s2, mp_graupelncv=s2,
                     mp_hailnc=s2, mp_hailncv=s2,
                     mp_sr=s2, refl_10cm=m,
                     # woof/core/milbrandt2.py::reflectivity H_Z(x): the
                     # temperature the lifted Z block reads when the caller
                     # has no microphysics-time pair to hand it.  Priced
                     # with the scheme, exactly as mp=18 prices da_nssl_t
                     # for the same operator's NSSL arm -- a mass-shaped
                     # float32 the arena estimator otherwise never saw.
                     my2_zet_t=m)
    if cfg.mp_physics == 10:
        # woof/core/morrison.py apply, the slots it draws from its own
        # lists (morrison_constants, CPU-only): the WRF preparation fields,
        # the WORK stack for one column call (tendencies, the scheme's
        # temperature, the stage hand-offs; the scheme runs in
        # ceil(ncol / NWORK)-column calls, so WORK is one volume plus at
        # most NWORK * nz values), the int32 LTRUE column flag, persistent
        # precipitation and REFL_10CM staging with refl_t, the post-call
        # temperature each column call copies out of its WORK plane.
        # morr_z8w is not allocated: the preparation kernel forms each dz
        # in registers.  It stays priced, an over-estimate of
        # (nz + 1) * ny * nx * 4 bytes, until the d01 calibration fixture
        # (measured with it) is re-measured.
        from woof.core import morrison_constants as mo
        for name in mo.SCRATCH_VOLUME_SLOTS:
            slots[name] = m
        slots[mo.SCRATCH_WORK_SLOT] = mo.work_slot_shape(cfg.nz, ny * nx)

        for name in mo.SCRATCH_INT_SURFACE_SLOTS:
            slots[name] = s2
        slots.update(morr_z8w=fl,
                     mp_rainnc=s2, mp_rainncv=s2, mp_snownc=s2,
                     mp_snowncv=s2, mp_graupelnc=s2, mp_graupelncv=s2,
                     mp_sr=s2, refl_t=m, refl_10cm=m)
    if cfg.mp_physics:
        # microphysics.py spec-zone ring guard: per-edge snapshot buffers
        # for the WRF tile-clip exclusion (specified/nested only; exact
        # shapes from the single-source helper).
        # physics_inventory, not microphysics: that module imports cupy
        # at module scope and this registry is priced on CPU-only installs.
        from woof.core.physics_inventory import spec_zone_ring_save_slots
        slots.update(spec_zone_ring_save_slots(cfg))
    from woof.microphysics_schemes import scheme as _named_mp_scheme
    if _named_mp_scheme(cfg.mp_physics) is not None:
        # A NAMED scheme's adapter (woof/core/<name>.py): the shared mp_*
        # preparation slots, its own temperature / w / graupel-volume work
        # planes, the standard precipitation slots and the reflectivity
        # handoff.  The column work area lives in the runtime, batched to
        # WOOF_TEMPO_WORK_BYTES, outside the arena.
        slots.update(
            mp_th=m, mp_pii=m, mp_dz8w=m, mp_z8w=fl,
            mp_tempo_temperature=m, mp_tempo_w=m, mp_tempo_qb=m,
            mp_rainnc=s2, mp_rainncv=s2, mp_snownc=s2, mp_snowncv=s2,
            mp_graupelnc=s2, mp_graupelncv=s2, mp_sr=s2,
            refl_10cm=m,
        )
        from woof.core.physics_inventory import spec_zone_ring_save_slots
        slots.update(spec_zone_ring_save_slots(cfg))
    if cfg.mp_physics == 18:
        # nssl2_runtime.py exact moist-physics prep, post-process radar
        # temperature, output handoff, and persistent precipitation state.
        # The mp_* prep names intentionally reuse the already classified
        # Kessler rebuild slots: their lifetime/shape contract is identical.
        slots.update(mp_th=m, mp_rho=m, mp_pii=m, mp_dz8w=m, mp_z8w=fl,
                     nssl2_driver_state=(16, nz, ny, nx),
                     nssl2_driver_surface_export=(5, ny, nx),
                     nssl2_driver_ignored_accumulator=s2,
                     nssl2_fused_temperature=m,
                     nssl2_primary_ice_target=m,
                     nssl2_nucond_ss=m, refl_t=m, refl_10cm=m,
                     mp_rainnc=s2, mp_rainncv=s2, mp_snownc=s2,
                     mp_snowncv=s2, mp_graupelnc=s2,
                     mp_graupelncv=s2, mp_hailnc=s2, mp_hailncv=s2,
                     mp_sr=s2)
        # woof/da/obsop.py:_nssl_reflectivity H(x) temporaries: dry-air
        # density and (when no temperature is passed) diagnosed T, both
        # mass-shaped, filled and consumed inside one operator call.
        slots.update(da_nssl_rho=m, da_nssl_t=m)

    from woof.config import wrf_mixing_package_active
    mixing_package = wrf_mixing_package_active(cfg)
    if mixing_package:
        slots.update(smag_km=m, smag_kh=m)
        if cfg.diff_opt == 1:
            slots.update(diff1_theta_initial=m, diff1_theta_work=m,
                         smag_mut=s2)
    if cfg.km_opt in (2, 3) or (mixing_package and cfg.km_opt == 1):
        # These closures carry the vertical exchange-coefficient pair; BN2
        # borrows the diff6_x face-workspace prefix and needs no slot.
        # km_opt=1 keeps its kvdif pair there (launch_wrf_isotropic_km).
        slots.update(smag_kmv=m, smag_khv=m)
    if cfg.km_opt == 2:
        # Prognostic-TKE forward tendency, its doubling temporary, and the
        # tke_rhs coupled-mass staging.
        slots.update(smag_rtke=m, smag_tke_tmp=m, smag_mut=s2)
        if getattr(cfg, "tke_budget", 0):
            # woof/core/tke_budget.py: the packed per-term field buffer,
            # the pre-bound_tke carrier snapshot, the two coupled-mass
            # planes the reduction reads, the FP64 slab accumulator, and
            # its step counter.  Priced only when the diagnostic is on --
            # it is a report-only toggle, not part of any trajectory.
            from woof.core.tke_budget import TERM_FIELDS, TERMS
            slots.update(
                tke_budget_terms=(len(TERM_FIELDS), nz, ny, nx),
                tke_budget_raw=m,
                tke_budget_mu0=s2, tke_budget_mu=s2,
                tke_budget_acc=(len(TERMS), nz),
                tke_budget_steps=(1,))
    if mixing_package or cfg.diff_6th_opt:
        # dycore.py prepare_fixed_tendencies: carrying WRF forward
        # tendencies shared by Smagorinsky and sixth-order diffusion.
        slots.update(smag_ru=xs, smag_rv=ys, smag_rw=fl, smag_rth=m)
        if cfg.bl_pbl_physics == 5 and cfg.bl_mynn_tkeadvect:
            slots["smag_rqke_adv"] = m
        if cfg.moist:
            for name in ("qv", "qc", "qr"):
                slots["smag_r" + name] = m
            # The three ICE MASS held tendencies, so the set is exactly
            # the schemes whose transported inventory STARTS FROM
            # woof/core/moist.py::ICE_MASS_SPECIES ("qi", "qs", "qg").
            # It is a PRICING MIRROR, not an admission gate: dycore
            # allocates one carrying buffer per row of _smag2d_specs
            # (dycore.py:1248-1252), which is SPECIES plus
            # extra_moist_species(state) -- and that helper dispatches on
            # FIELD PRESENCE, never on an integer.  moist.py imports cupy
            # at module scope (:95) while this registry is priced on
            # CPU-only installs, so the species tuples are transcribed
            # here as integers, and only a test can hold the two
            # spellings equal (tests/test_preflight.py, the mp=28 and
            # mp=50 held-tendency gates).
            #
            # mp=50 (P3) IS EXCLUDED ON PURPOSE, and the exclusion is not
            # a refusal -- nothing is denied, the scheme simply has no
            # snow or graupel to hold.  It is the one ported scheme with
            # qi and no qs/qg: Registry.EM_COMMON:3038 declares its
            # package as "moist:qv,qc,qr,qi;scalar:qni,qnr,qir,qib", so
            # WRF's own moist array carries no snow or graupel index
            # under P3, and the mixing loop that PRODUCES these
            # tendencies -- "do im = PARAM_FIRST_SCALAR, n_moist" at
            # module_diffusion_em.F:3036, bounded by that package extent
            # -- never reaches one.  Adding 50 here would price two full
            # (nz, ny, nx) fields no mp=50 DomainState allocates and
            # prepare_fixed_tendencies never writes, inflating the
            # headroom estimate on the card the estimate exists to
            # protect.  P3's own row is the mp == 50 arm below.
            from woof.core.column_schemes import COLUMN_SCHEMES
            column_row = COLUMN_SCHEMES.get(int(cfg.mp_physics))
            if column_row is not None:
                for name in column_row.species:
                    if name not in ("qv", "qc", "qr"):
                        slots["smag_r" + name] = m
            if cfg.mp_physics in (2, 6, 8, 9, 10, 16, 18, 28, 30, 106):
                for name in ("qi", "qs", "qg"):
                    slots["smag_r" + name] = m
            if cfg.mp_physics == 30:
                from woof.core.fast_sbm_contract import EXTRA_SPECIES
                for name in EXTRA_SPECIES: slots["smag_r"+name]=m
            if cfg.mp_physics == 24:
                for name in ("qi", "qs", "qg", "qh"):
                    slots["smag_r" + name] = m
            if cfg.mp_physics == 9:
                # One held tendency per TRANSPORTED species beyond the
                # ice masses; the set is woof/core/moist.py::MY2_SPECIES,
                # which is what prepare_fixed_tendencies iterates (1.9.1
                # D1's route: mp=9 had no arm here at all).
                for name in ("qh", "nc", "nr", "ni", "ns", "ng", "nh"):
                    slots["smag_r" + name] = m
            if cfg.mp_physics == 16:
                for name in ("nn", "nc", "nr"):
                    slots["smag_r" + name] = m
            if cfg.mp_physics == 8:
                for name in ("nr", "ni"):
                    slots["smag_r" + name] = m
            if cfg.mp_physics == 28:
                # One held tendency per TRANSPORTED species; the set is
                # woof/core/moist.py::THOMPSON_AERO_NUMBER_SPECIES, which
                # is what prepare_fixed_tendencies iterates.  nc is here
                # and is NOT here for mp=10, exactly as in moist.py.
                for name in ("nr", "ni", "nc", "nwfa", "nifa"):
                    slots["smag_r" + name] = m
            if cfg.mp_physics == 10:
                for name in ("nr", "ni", "ns", "ng"):
                    slots["smag_r" + name] = m
            if cfg.mp_physics == 5:
                # woof/core/moist.py::FERRIER_SPECIES.
                slots["smag_rqi"] = m
            if cfg.mp_physics == 50:
                # One held tendency per TRANSPORTED species, and the set is
                # woof/core/moist.py::P3_SPECIES -- which is what
                # prepare_fixed_tendencies iterates.  qs/qg are absent for
                # the same reason they are absent from the state.
                for name in ("qi", "ni", "nr", "qir", "qib"):
                    slots["smag_r" + name] = m
            if cfg.mp_physics == 18:
                for name in ("qh", "qndrop", "qnr", "qni", "qns", "qng",
                             "qnh", "qnn", "qvolg", "qvolh"):
                    slots["smag_r" + name] = m
            # A named scheme (woof.microphysics_schemes): one held tendency
            # per species its capability row transports -- the same rows
            # moist.extra_moist_species returns for it.
            named = _NAMED_MP_ROWS_BY_ID.get(int(cfg.mp_physics))
            if named is not None:
                for name in (named.ice_mass_species + named.moment_species
                             + named.aerosol_species):
                    slots["smag_r" + name] = m
        if getattr(cfg, "chem_sets", ""):
            # One held tendency per transported chem row, dry state or
            # moist: dycore._smag2d_specs appends the chem rows whenever
            # the state carries chem (their slots are smag_rchem_<row>).
            from woof.chem_table import load as load_chem_table
            for row in load_chem_table(cfg).transported:
                slots["smag_r" + row.state_attr] = m
    if mixing_package or cfg.diff_6th_opt:
        # Smagorinsky reuses the x/y face workspaces for u/v staging,
        # W stresses and metric scalar fluxes (km_opt=3 stages BN2 in the
        # diff6_x prefix during the K computation); sixth-order diffusion
        # subsequently overwrites them.  z/m are required only by diff6.
        slots.update(diff6_x=xs, diff6_y=ys)
    if cfg.diff_6th_opt:
        slots.update(diff6_z=fl, diff6_m=m)
        # The NOAA WRFV3.9 fork's edge-to-edge form (diff_6th_form =
        # "noaa_wrf39", the HRRR recipe request default) on a specified or
        # nested domain: the padded field, padded tendency, padded planes
        # and, under the slope taper, padded base geopotential that
        # dycore.launch_diff6_to_edge filters (diff6_edge_workspace.py).
        from woof.core.diff6_edge_workspace import diff6_edge_slot_shapes
        slots.update(diff6_edge_slot_shapes(cfg))

    from woof.core.physics_inventory import physics_enabled
    if physics_enabled(cfg):
        slots["physics_qtot"] = m                   # physics.py:369
        if not cfg.moist:
            slots.update(physics_dry_qv=m, physics_dry_qc=m)
        # physics.py:1541-1550 substitutes a zero-filled scratch plane only
        # when the state has no qi/qs of its own, PER FIELD.  mp=28 and
        # mp=16 allocate both, so listing them there would price two full
        # 3-D fields the run never asks for; mp=50 (P3) allocates qi and
        # NOT qs, so it is the one scheme that needs exactly one of the
        # two -- the conditions are therefore split rather than sharing a
        # tuple.
        #
        # BOTH TESTS ARE NEGATED, so mp=50's presence in the first and its
        # ABSENCE FROM THE SECOND is a decision rather than an oversight:
        # keeping 50 out of the qs tuple is what PRICES the plane every
        # physics-enabled mp=50 step allocates.  state.py:464-476 gives a
        # P3 state qi/ni/nr/qir/qib and no qs -- one ice category with a
        # rime mass/volume pair instead of split snow and graupel, so WRF
        # registers the package as moist:qv,qc,qr,qi
        # (Registry.EM_COMMON:3038) and its driver binds no snow array at
        # all in the mp=50 call shape (module_microphysics_driver.F:
        # 1569-1602).  The prep therefore substitutes qs and only qs.
        # Adding 50 below, which is what a mechanical "this scheme set
        # omits 50" sweep would do, drops a full nz*ny*nx float32 field
        # from the envelope of every mp=50 domain.  Enforced by
        # tests/test_preflight.py::test_mp50_prices_the_absent_snow_plane_
        # and_not_the_present_ice_one, which measures the prep's own
        # scratch requests on a real mp=50 state; before it existed that
        # edit left this whole module green.
        if cfg.mp_physics not in (2, 5, 6, 8, 10, 16, 18, 24, 28, 30, 50, 106):
            slots["physics_qi"] = m                 # physics.py:1541-1545
        if cfg.mp_physics not in (2, 6, 8, 10, 16, 18, 24, 28, 30, 106):
            slots["physics_qs"] = m                 # physics.py:1546-1550
    # FOUR producers write this one uint32 word and the disjunction has to
    # name every one of them: YSU (physics.py:_run_ysu), Shin-Hong
    # (physics.py:_run_shinhong), the native microphysics validator
    # (physics.py:2052) and KF (physics.py:_validate_native_kf_result).
    # A producer this registry does not name is not a refusal -- the word
    # is still allocated, just outside the audited arena, because
    # DomainState.scratch falls back to a private per-state allocation for
    # any slot the arena has no registry row for (state.py:933-939).
    #
    # The microphysics leg is DERIVED, not spelled.  The literal it
    # replaces -- (1, 6, 8, 10, 16, 28) -- was the ported-scheme list as it
    # stood when mp=28 landed, and it went stale twice with nobody deciding
    # to exclude anything: mp=9 and mp=50 both take the native arm and
    # neither was in it.  ``accept_microphysics`` routes a scheme through
    # ``_validate_native_microphysics`` exactly when it has a canonical
    # surface-diagnostic row and is not mp=18 (physics.py:2036-2052), so
    # that predicate IS the membership rule, read from the same table the
    # runtime reads instead of copied out of it.  18 stays out because
    # accept_microphysics excludes it BY NAME (physics.py:2046): NSSL owns
    # every accumulator in its own contract and takes the per-field arm,
    # which launches no validator.
    #
    # P3 (mp=50) belongs, and its one ice category is the reason it looked
    # like it did not.  No qs and no qg means its driver arm binds FIVE
    # surface diagnostics where mp=6/8 bind seven -- RAINNC/RAINNCV/SR/
    # SNOWNC/SNOWNCV and no graupel argument
    # (module_microphysics_driver.F:1590-1595, transcribed at
    # physics_inventory.py:157-170).  Five is still a canonical row, and
    # the SR bound at :1592 is P3's own solid-to-total ratio
    # pcprt_sol/(pcprt_liq+pcprt_sol+1.e-12) (module_mp_p3.F:898), which
    # the validator range-checks every step.  Both P3 backends return those
    # exact canonical scratch arrays (p3.py:1838-1840 reference,
    # p3.py:1987-1989 CUDA), so the native arm is taken and the word IS
    # asked for on every P3 step.  mp=9 arrives on the same predicate for
    # the same reason (milbrandt2.py:300-303), and this file already priced
    # the validator kernel module for both selectors
    # (_MICROPHYSICS_KERNEL_MODULES 9 and 50) while omitting its status
    # word -- the contradiction this derivation removes.
    #
    # bl=11 rides the same word as bl=1: Shin-Hong validates its outputs
    # through the identical batched-status policy.
    from woof.core.physics_inventory import microphysics_scratch_slots
    native_microphysics_validation = bool(
        microphysics_scratch_slots(int(cfg.mp_physics))
        and int(cfg.mp_physics) != 18)
    if (int(cfg.bl_pbl_physics) in (1, 11)
            or native_microphysics_validation
            or int(cfg.cu_physics) == 1):
        slots["physics_validation_status"] = (1,)
    if int(cfg.bl_pbl_physics) == 5:
        # MYNN's whole working set, declared in
        # woof/core/mynn_pbl_scratch.py.  Before it was declared the solver
        # allocated it fresh on every call and this registry knew none of it:
        # measured at nz = 49 on the RTX 5090, 46,160 bytes per column in 439
        # pool allocations per step, which is 15,847.6 MiB at the 360,000
        # columns of a d04 nest against a preflight estimate that did not
        # move at all.  A headroom check that reassuring on a card with no
        # ECC is a hazard, not a safeguard.
        #
        # These shapes are written against the COLUMN CHUNK, not ny*nx, which
        # is what makes the estimate bounded: mynn_pbl_runtime walks the
        # domain in chunks of that width and the split is bitwise identical
        # to the wide call.  Only the six returned A-grid tendency fields are
        # full width, because couple_ysu_tendencies consumes whole fields.
        slots.update(mynn_pbl_scratch_slots(cfg, tile_buffer=tile_buffer))
    if int(cfg.bl_pbl_physics) == UW_PBL_SCHEME:
        # physics.py _run_uwpbl: WRF's zero QNC_CURR/WSEDL3D (and the zero
        # QNI_CURR where the microphysics carries no P_QNI), one plane.
        slots["uwpbl_zero"] = m
    if cfg.cu_physics:
        # physics.py:451-470 KF driver persistence (restart-serialized).
        slots.update(cu_rainc=s2, cu_nca=s2, cu_pratec=s2, cu_raincv=s2,
                     cu_expiring=s2,
                     cu_rthcuten=m, cu_rqvcuten=m, cu_rqccuten=m,
                     cu_rqicuten=m, cu_rqrcuten=m, cu_rqscuten=m)

    if cfg.specified:
        # lateral_bc.py resident attachment: held relax tendencies (:612),
        # Davies weights (:294), the MU boundary frame (:664), and the
        # packed eager forcing tables (:545) when interval count is known.
        slots.update(lbc_relax_u=xs, lbc_relax_v=ys, lbc_relax_theta=m,
                     lbc_relax_phi=fl)
        if getattr(cfg, "relax_w", False) and not cfg.nested:
            # The held w relaxation of a specified domain that relaxes w
            # (lateral_bc.apply_state_lateral_boundaries, relax_w).
            slots["lbc_relax_w"] = fl
        if cfg.moist:
            from woof.boundary_fields import potential_external_scalar_fields
            for name in potential_external_scalar_fields(cfg):
                slots[f"lbc_{name}_held"] = m
        slots["lbc_weights_0"] = (2, cfg.spec_bdy_width)
        slots[f"lbc_old_mup_frame_{cfg.spec_zone}"] = (
            _perimeter_count(ny, nx, cfg.spec_zone),)
        if cfg.specified and n_lbc_intervals > 0:
            slots["lbc_forcing_tables"] = (
                n_lbc_intervals * lbc_interval_values(cfg),)
        if cfg.specified and getattr(cfg, "chem_sets", ""):
            # WRF-Chem's chem_b/chem_bt of the rows a data-store source
            # fills (have_bcs_chem): the outer row of every side, packed
            # per served interval for the chem flow boundary launch
            # (lateral_bc.chem_boundary_rows), one row per transported row.
            from woof.chem_source_init import chem_boundary_fields
            from woof.chem_table import load as load_chem_table
            from woof.ingest.lateral_bc import CHEM_BOUNDARY_SLOTS
            table = load_chem_table(cfg)
            if chem_boundary_fields(table, cfg):
                rows = len(table.transported)
                for index, slot in enumerate(CHEM_BOUNDARY_SLOTS):
                    slots[slot] = (rows, nz, ny if index < 4 else nx)
    elif cfg.nested:
        # Rolling tables themselves live in the F4/F16 nest manifest.
        # Only the tiny Davies weights and MU finalizer frame use the legacy
        # LBC scratch registry. Nested held increments are recomputed from
        # RK time-t copies, so no extra full-domain carrying slots exist.
        frame_width = min(max(cfg.spec_zone, cfg.relax_zone + 1),
                          cfg.spec_bdy_width)
        slots["lbc_weights_0"] = (2, frame_width)
        slots[f"lbc_old_mup_frame_{cfg.spec_zone}"] = (
            _perimeter_count(ny, nx, cfg.spec_zone),)
        # One temporary is reused serially for u/v/w/theta/phi.  W is the
        # largest field only when horizontal extents exceed nz; retain the
        # true maximum so valid skinny/high-top grids remain capacity-safe.
        slots["lbc_nested_relax"] = _full_field_capacity(cfg)
        if WRF_EXACT:
            # Strict mode takes WRF's solve_em order on a nest too
            # (dycore._wrf_relaxation_fold: specified OR nested):
            # lateral_bc.capture_folded_relaxation holds u/v/theta/w and
            # _apply_state_lateral_boundaries_wrf holds phi, each at THIS
            # domain's own shape, in the same lbc_relax_* slots a specified
            # parent declares above.  The shared arena sizes a slot by the
            # largest domain that declares it, so with only the parent
            # declaring them a nest larger than its parent got the parent's
            # backing: the two-domain strict UAE case (d01 121x100,
            # d02 211x150, 44 levels) stopped at its first nest step with
            # "scratch slot 'lbc_relax_u' arena capacity is 532400 values
            # ((44, 100, 121)), requested 1392600 values ((44, 150, 211))".
            slots.update(lbc_relax_u=xs, lbc_relax_v=ys, lbc_relax_theta=m,
                         lbc_relax_phi=fl, lbc_relax_w=fl)
    return slots


# ---------------------------------------------------------------------------
# F4 NEST ALLOCATION MANIFEST -- authoritative; Tasks 10/13 register slots
# matching these names/shapes EXACTLY (registry-equality gate; any drift is
# a plan amendment).
# ---------------------------------------------------------------------------

from woof.core.nest_fields import nest_field_kinds


def _kind_dims(kind: str, nz: int, ny: int, nx: int) -> tuple[int, int, int]:
    """(levels, ny-extent, nx-extent) of one field kind on the child grid."""
    if kind == "u":
        return (nz, ny, nx + 1)
    if kind == "v":
        return (nz, ny + 1, nx)
    if kind in ("w", "ph"):
        return (nz + 1, ny, nx)
    if kind == "mu":
        return (1, ny, nx)
    return (nz, ny, nx)


def _full_field_capacity(cfg: RunConfig) -> tuple[int, ...]:
    """Flat capacity of the largest full field for one domain."""
    shapes = (_kind_dims(kind, cfg.nz, cfg.ny, cfg.nx)
              for kind in nest_field_kinds(cfg))
    return (max(math.prod(shape) for shape in shapes),)


def nest_slot_shapes(dc: DomainConfig, spec_bdy_width: int,
                     parent: DomainConfig | None = None
                     ) -> dict[str, tuple[int, ...]]:
    """Every persistent ``nest_*`` device slot for ONE child domain.

    Three families (architecture section E item (d), F4/F16 amendments):

    * Rolling one-interval boundary tables, WRF Registry naming
      (``u_bxs``/``u_btxs`` style): per kind, per side, value
      (``nest_{kind}_b{xs,xe,ys,ye}``) and tendency
      (``nest_{kind}_bt{side}``).  x-sides ``(lev, ny_ext, W)``, y-sides
      ``(lev, W, nx_ext)`` -- the exact ``_field_boundary`` layout the
      Phase-4 resident-table CUDA machinery consumes
      (lateral_bc.py:149-167), refreshed every parent step with dtbc
      reset (mediation_force_domain.F semantics; registered deviation:
      rolling device tables instead of WRF's persistent host bdy
      arrays).
    * TWO simultaneously live force-local coupled fields
      (``nest_parent_field`` and ``nest_child_field``), shared through the
      sequential-domain arena when two distinct dead RK backings have enough
      capacity and otherwise allocated explicitly.  Each is overwritten
      before every ``bdy_interp1`` read.  F16 retires all four-side donor-strip
      rows.
    * SINT geometry: six arrays per stagger class, matching T10's landed
      ``device_tables`` exactly: ``ci/ip`` at ``nx_ext`` and ``cj/jp`` at
      ``ny_ext`` (int32), plus ``xig`` at ``nri`` and ``xjg`` at ``nrj``
      (float32).
    """
    run = dc.run
    nz, ny, nx = run.nz, run.ny, run.nx
    ratio = dc.parent_grid_ratio
    # interp_fcn.F:2517, identical to nest_interp.bdy_width().  The rolling
    # manifest must describe the exact arrays bdy_interp1 writes, even when
    # a case configures a wider maximum boundary allocation.
    width = min(max(int(run.spec_zone), int(run.relax_zone) + 1),
                int(spec_bdy_width))
    shapes: dict[str, tuple[int, ...]] = {}

    for kind in nest_field_kinds(run):
        lev, ny_ext, nx_ext = _kind_dims(kind, nz, ny, nx)
        for prefix in ("b", "bt"):
            shapes[f"nest_{kind}_{prefix}xs"] = (lev, ny_ext, width)
            shapes[f"nest_{kind}_{prefix}xe"] = (lev, ny_ext, width)
            shapes[f"nest_{kind}_{prefix}ys"] = (lev, width, nx_ext)
            shapes[f"nest_{kind}_{prefix}ye"] = (lev, width, nx_ext)

    parent_run = run if parent is None else parent.run
    shapes["nest_parent_field"] = _full_field_capacity(parent_run)
    shapes["nest_child_field"] = _full_field_capacity(run)

    for stag, (nx_ext, ny_ext) in (("m", (nx, ny)), ("x", (nx + 1, ny)),
                                   ("y", (nx, ny + 1))):
        shapes[f"nest_sint_ci_{stag}"] = (nx_ext,)
        shapes[f"nest_sint_ip_{stag}"] = (nx_ext,)
        shapes[f"nest_sint_cj_{stag}"] = (ny_ext,)
        shapes[f"nest_sint_jp_{stag}"] = (ny_ext,)
        shapes[f"nest_sint_xig_{stag}"] = (ratio,)
        shapes[f"nest_sint_xjg_{stag}"] = (ratio,)
    return shapes


def nest_slot_dtypes(dc: DomainConfig, spec_bdy_width: int,
                     parent: DomainConfig | None = None) -> dict[str, str]:
    """Semantic dtype of every F4/F16 nest slot (all are four bytes)."""
    shapes = nest_slot_shapes(dc, spec_bdy_width, parent)
    return {name: ("int32" if name.startswith(("nest_sint_ci_",
                                                "nest_sint_ip_",
                                                "nest_sint_cj_",
                                                "nest_sint_jp_"))
                   else "float32")
            for name in shapes}


def nest_allocation_manifest(exp: ExperimentConfig
                             ) -> dict[int, dict[str, tuple[int, ...]]]:
    """The frozen nest allocation manifest: ``grid_id -> {slot: shape}``
    for every CHILD domain.  N0's ``--alloc`` allocates exactly these
    entries as real device allocations (F4: the residency proof covers
    the actual coupler footprint, not proxies)."""
    manifest: dict[int, dict[str, tuple[int, ...]]] = {}
    by_id = {dc.grid_id: dc for dc in exp.domains}
    for dc in exp.domains:
        if dc.parent_id == 0:
            continue
        manifest[dc.grid_id] = nest_slot_shapes(
            dc, exp.spec_bdy_width, by_id[dc.parent_id])
    return manifest


# ---------------------------------------------------------------------------
# Scratch-slot lifetime audit (architecture section E lever 2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScratchSlotLifetime:
    """One reviewed row in the scratch sharing admission table.

    Patterns ending in ``*`` classify a generated slot family.  ``kind`` is
    either ``write_before_read`` (safe on the sequential-domain arena),
    ``carrying`` (observed after the producing step/call), or
    ``excluded_unproven`` (correctness-first exclusion).
    """

    patterns: tuple[str, ...]
    kind: str
    evidence: str
    rationale: str

    @property
    def arena_eligible(self) -> bool:
        return self.kind == "write_before_read"


# Committed audit table. tests/test_preflight.py expands every registry and
# F4-manifest slot through this table, rejects gaps/overlaps, and separately
# proves that every arena-admitted row is write-before-read classified.
from woof.core.hail import PERSISTENT as _HAIL_PERSISTENT

from woof.core.output_diagnostics import PERSISTENT as _OUTPUT_PERSISTENT

SCRATCH_SLOT_LIFETIME_AUDIT = (
    ScratchSlotLifetime(_OUTPUT_PERSISTENT + ("cl_zero", "output_zero", "gsd_zero", "gsd_uh_use", "gsd_pressure_w", "gsd_temperature", "gsd_density", "gsd_wind10", "gsd_wind80"), "carrying",
        "woof/core/output_diagnostics.py:update_precipitation",
        "Interval buckets survive history and checkpoint; absent inputs stay zero"),
    ScratchSlotLifetime(
        _HAIL_PERSISTENT
        + ("hail_zero",), "carrying", "woof/core/hail.py:update,reset_window",
        "Embryos, history windows, run maxima and updraft duration survive "
        "steps; the absent-species plane must remain zero"),
    ScratchSlotLifetime(
        ("hail_temperature", "hail_height", "hail_density", "hail_new_mask",
         "hail_new_duration"), "write_before_read", "woof/core/hail.py:update",
        "Preparation and duration kernels overwrite every entry before use"),
    ScratchSlotLifetime(("ieva_*",), "excluded_unproven", "woof/core/ieva.py (zadvect_implicit = 1)", "the explicit/implicit eta mass-flux split and column masses are produced by the dynamics and reused by the scalar split later in the step; no audit has proven a sibling domain cannot interleave, so each domain keeps its own (found by the first HRRR-recipe tree, 2026-10-21)"),
    ScratchSlotLifetime(("fsbm_*",), "excluded_unproven", "woof/core/fast_sbm_runtime.py:apply", "mixed uint64 pointers, int32 status and float64 tallies are not compatible with the float32 arena; all are rewritten inside each call"),
    # Strict WRF mode (woof.wrf_exact.ENABLED).  acoustic_wrf_muts entered
    # the registry (9190ec5f2) without a row here, so every multi-domain
    # strict run raised KeyError in scratch_slot_uses_arena while
    # woof/core/model.py built the shared arena, and the estimator raised
    # the same on the same road.  The two mass planes cross the step
    # (woof/io/restart.py carries them); the rest have no reviewed
    # sequential-domain lifetime proof, and strict mode is the byte-identity
    # referee, so each domain keeps its own backing, exactly as a
    # single-domain strict run (which builds no arena) always has.
    ScratchSlotLifetime(
        ("wrf_exact_muts", "wrf_exact_mu_halo"), "carrying",
        "woof/core/dycore.py:_capture_wrf_muts, _wrf_step_epilogue; "
        "woof/io/restart.py:CARRIED_SCRATCH_SLOTS",
        "the acoustic loop's MUT + MU'' and the pre-boundary mu ring are "
        "read by the next step's first RK stage"),
    ScratchSlotLifetime(
        ("acoustic_wrf_muts", "rk_ww_ru", "rk_ww_rv", "moist_sc_t"),
        "excluded_unproven",
        "woof/core/acoustic.py:wrf_acoustic_muts; "
        "woof/core/dycore.py:_calc_ww_cp_fluxes; "
        "woof/core/moist.py:_advance_species_wrf",
        "strict-mode only; kept off the shared arena until a lifetime "
        "review admits them, so strict multi-domain runs allocate them per "
        "domain as single-domain strict runs do"),
    ScratchSlotLifetime(
        ("wsm3_theta", "wsm3_rho", "wsm3_pii", "wsm3_dz", "wsm3_z8w", "wsm3_w"),
        "write_before_read", "woof/core/wsm3.py:apply_wsm3",
        "all six fields are assigned over their complete shapes before the column launch"),
    ScratchSlotLifetime(
        ("refl_wsm3_rain", "refl_wsm3_snow", "refl_wsm3_graupel"),
        "write_before_read", "woof/core/refl.py:compute_refl_10cm",
        "the phase split fills rain and snow and zeroes graupel before the radar launch"),
    ScratchSlotLifetime(
        ("wsm6r_theta", "wsm6r_rho", "wsm6r_pii", "wsm6r_dz",
         "wsm6r_z8w", "wsm6r_work"), "write_before_read",
        "woof/core/wsm6.py:_wsm6_prepare and _apply_wsm6r; "
        "woof/core/kernels/wsm6r.cu:column preparation and fall velocity",
        "Preparation rewrites theta, density, Exner, layer thickness and "
        "interfaces; the column initializes each fall-velocity workspace "
        "element before using it in the sedimentation sweep"),
    ScratchSlotLifetime(
        ("diff1_theta_initial",), "carrying",
        "woof/core/dycore.py:initialize_coordinate_reference; "
        "woof/io/restart.py:SERIALIZED_SCRATCH_SLOTS",
        "the original thermal reference persists across steps, restart "
        "and streamed slabs and cannot share a sequential-domain arena"),
    ScratchSlotLifetime(
        ("diff1_theta_work",), "write_before_read",
        "woof/core/dycore.py:_compute_coordinate_tendencies",
        "the reconstructed WRF thermal field is filled before each "
        "coordinate flux launch"),
    ScratchSlotLifetime(
        ("rk_ww", "rk_ru", "rk_rv", "rk_ru_m", "rk_rv_m", "rk_ww_m"),
        "write_before_read",
        "woof/core/dycore.py:102-113,154-161,1373-1414; "
        "woof/core/nest.py:_coupled_child_field",
        "stage/moist fluxes are filled before stage reads; FORCE borrows "
        "the matching rk_ru/rk_rv/rk_ww staggered backing only after the "
        "prior step and before the next stage rewrite"),
    ScratchSlotLifetime(
        ("adv_ru", "adv_rv", "adv_rw"), "write_before_read",
        "woof/core/advection.py:161-180",
        "the advection-only path fills all three flux arrays before launch"),
    ScratchSlotLifetime(
        ("acoustic_mu_pp_old", "acoustic_th_pp_old", "acoustic_c2a",
         "acoustic_a", "acoustic_alpha", "acoustic_gamma", "acoustic_mudf",
         "acoustic_cqu", "acoustic_cqv", "acoustic_cqw",
         "openbc_upp_faces", "openbc_vpp_faces"),
        "write_before_read",
        "woof/core/acoustic.py:115-190,223-235; woof/core/dycore.py:1452-1464",
        "substep histories/coefficient/filter slots are seeded in their stage; "
        "the W primitive path borrows acoustic_a/acoustic_c2a before the "
        "RK loop, and calc_coefs rewrites every consumed coefficient before "
        "the first substep; acoustic_a row zero is not consumed"),
    ScratchSlotLifetime(
        ("smag_km", "smag_kh", "smag_ru", "smag_rv", "smag_rw",
         "smag_rth", "smag_rqv", "smag_rqc", "smag_rqr", "smag_rqi", "smag_rqke_adv",
         "smag_rqs", "smag_rqg", "smag_rnr", "smag_rni", "smag_rns",
         # mp=16 (WDM6) transported number moments.  smag_rnc/smag_rnr are
         # already above -- Morrison named them first -- so only the CCN
         # reservoir is new.
         "smag_rnn",
         # mp=9 (Milbrandt-Yau) hail number.  The rest of its transported
         # set was already audited here by the schemes that named the
         # slots first; nh is the one name no other scheme transports
         # (1.9.1 D1's route, third table: registered by
         # scratch_slot_registry without a row here, invisible until a
         # TREE reached shared_scratch_arena_shapes -- the identical
         # class as the km_opt=2/3 row below).
         "smag_rnh",
         "smag_rng", "smag_rqh", "smag_rqndrop", "smag_rqnr",
         "smag_rqni", "smag_rqns", "smag_rqng", "smag_rqnh",
         "smag_rqnn", "smag_rqvolg", "smag_rqvolh",
         # mp=28 transported number/aerosol moments.  Same construction,
         # same lifetime: prepare_fixed_tendencies writes each held
         # tendency once before the RK loop and every stage only reads it.
         "smag_rnc", "smag_rnwfa", "smag_rnifa",
         # mp=50 (P3) rime mass and rime volume.  Same construction, same
         # lifetime: they are transported scalars like the number moments,
         # so prepare_fixed_tendencies writes each held tendency once
         # before the RK loop and every stage only reads it.
         "smag_rqir", "smag_rqib",
         # A named scheme's graupel volume (TEMPO's volg); its numbers
         # share the names above.  Same construction, same lifetime.
         "smag_rvolg",
         # Chem rows' held tendencies (woof/core/chem_transport.py), one
         # per transported row, named by the chem table.  Same
         # construction, same lifetime.
         "smag_rchem_*", "smag_rff1i*", "smag_rff5i*", "smag_rff6i*", "smag_rff8i*"),
        "write_before_read",
        "woof/core/dycore.py:prepare_fixed_tendencies",
        "time-t K and its held tendencies are written and consumed before "
        "the RK loop; K is dead before acoustic alpha/gamma overwrite the "
        "borrowed backings, while all three RK stages read only the held "
        "tendencies"),
    # The km_opt=2/3 closure's own slots.  They were registered by
    # scratch_slot_registry (km_opt in (2, 3) above) without a row here,
    # which is invisible to a single-domain run -- only a TREE reaches
    # shared_scratch_arena_shapes, and no LES tree had been built.  The
    # first nested LES domain hit it as
    # `KeyError: scratch slot 'smag_kmv' has no lifetime audit row`.
    ScratchSlotLifetime(
        ("smag_kmv", "smag_khv", "smag_rtke", "smag_tke_tmp", "smag_mut"),
        "write_before_read",
        "woof/core/dycore.py:836-837,928-929 (written by "
        "launch_wrf_tke_km / launch_wrf_smag3d_km), :1207-1208 (read by "
        "launch_wrf_smag2d_vertical), :1288 (smag_rtke zeroed); "
        "woof/core/moist.py:advance_tke_stage",
        "the vertical exchange pair is filled and consumed inside one "
        "prepare_fixed_tendencies call exactly as the horizontal "
        "smag_km/smag_kh pair above, and smag_rtke is a held time-t "
        "tendency on the same footing as the smag_r* row -- zeroed before "
        "the RK loop, read by all three stages, never carried across a "
        "step; smag_tke_tmp and smag_mut are within-call staging"),
    ScratchSlotLifetime(
        ("tke_budget_terms", "tke_budget_raw", "tke_budget_mu0",
         "tke_budget_mu"), "write_before_read",
        "woof/core/tke_budget.py:clear_fields,accumulate; "
        "woof/core/dycore.py:1288,2380",
        "clear_fields zeroes the packed term buffer at the top of the "
        "step and accumulate folds it, the pre-bound_tke snapshot and the "
        "two coupled-mass planes into the accumulator before the same "
        "step() returns"),
    ScratchSlotLifetime(
        ("tke_budget_acc", "tke_budget_steps"), "carrying",
        "woof/core/tke_budget.py:accumulator,reset,accumulate,drain",
        "the window accumulator and its step counter are read many steps "
        "after the step that wrote them (drain ends the window), and both "
        "are float64 while ScratchArena is float32-only"),
    ScratchSlotLifetime(
        ("diff6_x", "diff6_y", "diff6_z", "diff6_m"),
        "write_before_read", "woof/core/dycore.py:prepare_fixed_tendencies; "
        "woof/core/dycore.py:apply_diff6",
        "the diff6 target loops consume one temporary at a time; however, "
        "Smagorinsky uses x/y simultaneously for momentum staging, W "
        "stress faces and scalar fluxes, so every configuration retains "
        "two distinct face backings"),
    ScratchSlotLifetime(
        ("diff6_edge_field", "diff6_edge_tend", "diff6_edge_planes",
         "diff6_edge_phb"),
        "write_before_read",
        "woof/core/dycore.py:launch_diff6_to_edge,_diff6_edge_work; "
        "woof/core/diff6_edge_workspace.py",
        "every padded copy (field, column mass, map factors, base "
        "geopotential) is written whole by _edge_pad_into and the padded "
        "tendency is zeroed before the kernel adds into it, all inside one "
        "launch_diff6_to_edge call whose result is added into the caller's "
        "diff6_* temporary before it returns; nothing is read across "
        "calls, so no arena neighbour can be observed through them"),
    ScratchSlotLifetime(
        ("chem_mono_qmin", "chem_mono_qmax", "chem_mono_si",
         "chem_mono_so", "chem_mono_ht", "chem_mono_zt"),
        "write_before_read",
        "woof/core/chem_advect_mono.py:launch_mono_fluxes; "
        "woof/core/kernels/mono_advection.cu:mono_fluxes,mono_scales,"
        "mono_apply",
        "mono_fluxes writes both extrema of every cell, mono_scales both "
        "scales of every cell and mono_apply both diagnostics of every cell, "
        "each before the next kernel reads them, once per chem row"),
    ScratchSlotLifetime(
        ("chem_mono_wwi",), "carrying",
        "woof/core/chem_transport.py:advance_chem_stage",
        "WRF's implicit vertical mass flux, read as the zeros the slot is "
        "allocated with on every monotonic stage and never written; lending "
        "its backing to any other slot would hand the limiter a nonzero "
        "implicit flux"),
    ScratchSlotLifetime(
        ("moist_pd_q0", "moist_rq_t", "moist_absent_mass",
         "pd_fxl", "pd_fxc", "pd_fyl",
         "pd_fyc", "pd_fzl", "pd_fzc"), "write_before_read",
        "woof/core/moist.py:197-203,247-308; "
        "woof/core/moist.py::absent_mass_plane",
        "source copies, tendencies, and six PD fluxes are filled before "
        "use; the absent-mass plane is zeroed inside the same call that "
        "hands it to calc_cq or slow_buoyancy, immediately before the "
        "launch, so no arena neighbour can be observed through it"),
    ScratchSlotLifetime(
        ("mp_th", "mp_rho", "mp_pii", "mp_z", "mp_dz8w", "mp_z8w",
         "mp_tempo_temperature", "mp_tempo_w", "mp_tempo_qb",
         "nssl2_driver_state", "nssl2_driver_surface_export",
         "nssl2_driver_ignored_accumulator",
         "nssl2_fused_temperature", "nssl2_primary_ice_target",
         "nssl2_nucond_ss"),
        "write_before_read", "woof/core/microphysics.py:143-169; "
        "woof/core/nssl2_runtime.py:_prepare_fields; "
        "woof/core/nssl2_fused_gs.py:launch_fused_gs; "
        "woof/core/kernels/nssl2_fused_gs.cu:nssl2_prepare_fused_gs; "
        "woof/core/nssl2_nucond.py:139-143; "
        "woof/core/kernels/nssl2_nucond.cu:96-107",
        "Kessler and NSSL preparation fields are rebuilt for every scheme "
        "call; gather overwrites the driver state and sediment export planes "
        "while the ignored accumulator is explicitly reset before RMW; the "
        "fused-GS prepass overwrites temperature and primary-ice target "
        "snapshots before the process kernel reads them; NUCOND overwrites "
        "its supersaturation filter before reading it"),
    ScratchSlotLifetime(
        # The eighteen device companions plus the two column-scope logical
        # flags.  Every one is written before it is read WITHIN a call and
        # nothing in any of them crosses a call boundary: p3k_prep and
        # p3k_kloop1 fill the twelve carriers for every level of every
        # column before p3k_kloopmain reads one, each sedimentation step
        # zeroes the six workspace arrays over the whole column before its
        # own substep loop touches them (woof/core/kernels/p3.cu, the
        # `for (int k = 0; k < nk; ++k) AT(W[...], k) = 0.0f` prologues),
        # and p3k_prep clears both flags before p3k_kloop1 sets them.
        #
        # The cross-step carriers of this scheme are th_old and qv_old, and
        # they are DomainState FIELDS serialized by woof/io/restart.py --
        # deliberately not scratch, because the diagnosed-ssat branch reads
        # them on the next call (module_mp_p3.F:2325-2337, :5018-5021).
        ("p3_rho", "p3_inv_rho", "p3_qvs", "p3_qvi", "p3_sup", "p3_supi",
         "p3_rhofacr", "p3_rhofaci", "p3_acn", "p3_t", "p3_tmparr1",
         "p3_qv_cld", "p3_sed_v_q", "p3_sed_v_n", "p3_sed_flux_q",
         "p3_sed_flux_n", "p3_sed_flux_qir", "p3_sed_flux_bir",
         "p3_flags", "p3_nc", "p3_ssat", "p3_prt_liq", "p3_prt_sol"),
        "write_before_read",
        "woof/core/p3_device.py:SCRATCH_SLOTS/SEDW_SLOTS (allocation via "
        "DomainState.scratch); woof/core/kernels/p3.cu p3_step_prep_col, "
        "p3_step_kloop1_col, the three p3_step_sed_*_col prologues; "
        "woof/core/p3.py:apply (ssat zeroed on entry, the wrapper's "
        "module_mp_p3.F:851)",
        "the twelve carriers are filled by prep/kloop1 before kloopmain "
        "reads them, the six sedimentation arrays are zeroed over the whole "
        "column at the top of each sedimentation step, the flag pair is "
        "cleared in prep, nc is respecified from nccnst/rho every call at "
        "the specified-Nc setting (:2350) and ssat is zeroed by the adapter "
        "and diagnosed in kloop1 -- none of them carries a value across a "
        "call boundary"),
    ScratchSlotLifetime(
        ("p3_vmi", "p3_di", "p3_rhopo", "p3_effc", "p3_effi"),
        "write_before_read",
        "woof/core/p3.py:1749-1751 (allocation), :p3_main diagnostic pass; "
        "phys/module_mp_p3.F:1965-1967 (intent(out)), :2282-2284 (zeroed on "
        "entry), :4856-4858 (written from the post-update ice state)",
        "P3's diagnostics are intent(out) of p3_main: it presets them on "
        "entry (:2278-2288) and refills them from the updated state before "
        "returning, so nothing in them survives a call boundary and the "
        "driver's read-back always follows that call's own write.  effc and "
        "effi joined this row with the CUDA port: the device kernels write "
        "them in metres and the adapter converts into state.effc/effi in "
        "microns, where the host path used two host arrays instead"),
    ScratchSlotLifetime(
        ("mp_thompson_temperature",
         "mp_thompson_frozen_reference_density",
         "mp_thompson_frozen_reference_temperature",
         "mp_thompson_rain_reference_density",
         "mp_thompson_snow_melt_marker",
         "mp_thompson_graupel_melt_marker",
         "mp_thompson_snow_velocity_boost",
         "mp_thompson_graupel_number_shadow"),
        "write_before_read",
        "woof/core/microphysics.py:_apply_thompson; "
        "woof/core/kernels/thompson.cu:129-217,1999-2020,5674-5682",
        "the adapter fills temperature before its first consumer; cloud/rain "
        "reference kernels write every cell before fallout; the warm source "
        "writes both held melt markers for every cell before their consumers; "
        "the fused cold source resets every velocity boost; output-due "
        "graupel number is initialized across the complete field before "
        "source/fallout reads"),
    # Classic Thompson's own working set (woof/core/thompson_classic.py).
    # Every call writes every word it reads: tc_entry the temperature,
    # theta, Exner, layer depths, private graupel number and both column
    # flags (filled to zero first); tc_sources every tendency of every cell
    # (zero in a no-microphysics column), every working field of every cell
    # of a column with microphysics, and every cell flag; tc_column zeroes
    # its fall-speed planes before it reads them and writes each flux
    # before the update that reads it, and touches no column whose flags
    # say it has no microphysics.  Nothing crosses a call.
    ScratchSlotLifetime(
        ("mp_thompson_saved_theta",
         "mp_thompson_classic_ten",
         "mp_thompson_classic_work",
         "mp_thompson_classic_colbuf",
         "mp_thompson_classic_flags",
         "mp_thompson_classic_micro",
         "mp_thompson_classic_kmelt"),
        "write_before_read",
        "woof/core/thompson_classic.py:apply_classic; "
        "woof/core/kernels/thompson_classic.cu:tc_entry, tc_sources, "
        "tc_column",
        "WRF's mp_thompson zeroes its tendencies and recomputes every "
        "working value at the top of each column call (module_mp_thompson.F"
        ":1650-1777, :1798-2002); the unit's buffers hold exactly those "
        "values for the length of one call"),
    # WRF's per-column no_micro flag (module_mp_thompson.F:1646, :2020),
    # registered for mp=8 and mp=28 by the Thompson repair G.  The slot
    # shipped in 217e84e18 without this row, so every mp=8 and mp=28
    # configuration's scratch_slot_uses_arena raised KeyError on it; the two
    # Thompson scratch-completeness tests in tests/test_preflight.py now
    # require its arena admission.
    ScratchSlotLifetime(
        ("mp_thompson_micro_columns",),
        "write_before_read",
        "woof/core/microphysics.py:_apply_thompson and "
        "microphysics_aerosol.py:_apply_thompson_aerosol (launch before the "
        "first source kernel); woof/core/kernels/thompson.cu:"
        "thompson_microphysics_columns, thompson_aerosol_state.cu:"
        "thompson_aa_micro_columns",
        "each call takes the flag from its own entry state: the flag kernel "
        "writes 1 or 0 at every column, unconditionally, before the "
        "terminal apply that is its only reader, so nothing in it crosses "
        "a call boundary"),
    # mp=28's aerosol working set.  WRITE-BEFORE-READ, and the evidence is
    # structural rather than incidental: WRF's own column loop freezes the
    # entry state and ZEROES the three accumulators at the top of every call
    # (module_mp_thompson.F:1679-1681), so the adapter must explicitly fill
    # every one of these at entry before any network runs.  That is not a
    # convenience -- woof/core/state.py's scratch pool persists across
    # steps by design, so an accumulator that were merely "usually
    # overwritten" would carry the previous step's aerosol tendency forward
    # as a slow, bounded, entirely plausible-looking drift that no
    # single-step column test could see.  The entry snapshots are written by
    # the entry kernels over the complete field (no branch leaves a cell
    # unassigned) before the first consumer, and the terminal apply/clamp
    # reads each accumulator exactly once.
    ScratchSlotLifetime(
        ("mp_thompson_aero_ncten",
         "mp_thompson_aero_nwfaten",
         "mp_thompson_aero_nifaten",
         "mp_thompson_aero_entry_density",
         "mp_thompson_aero_tau1_density",
         "mp_thompson_aero_nwfa_work_m3",
         "mp_thompson_aero_qcten",
         "mp_thompson_aero_qrten",
         "mp_thompson_aero_nrten",
         "mp_thompson_aero_qiten",
         "mp_thompson_aero_niten",
         "mp_thompson_aero_ni_entry",
         "mp_thompson_aero_qvten",
         "mp_thompson_aero_tten",
         "mp_thompson_aero_qsten",
         "mp_thompson_aero_qgten",
         "mp_thompson_aero_ngten",
         "mp_thompson_aero_condensation_rate"),
        "write_before_read",
        "woof/core/thompson_aerosol_state.py:"
        "zero_aerosol_accumulators,launch_aerosol_entry_snapshot,"
        "launch_aerosol_entry_cloud_number,launch_tau1_density,"
        "launch_aerosol_working_number; "
        "woof/core/kernels/thompson_aerosol_state.cu",
        "the adapter zeroes the three accumulators and fills every entry "
        "snapshot at call entry, before any source network reads them; the "
        "terminal state-finalize kernel is the single consumer of the "
        "accumulators and runs after every writer"),
    ScratchSlotLifetime(
        ("lin_theta", "lin_rho", "lin_pii", "lin_dz", "lin_z", "lin_z8w"),
        "write_before_read", "woof/core/lin.py:apply",
        "Lin preparation assigns every volume before its kernel reads it"),
    ScratchSlotLifetime(
        ("mpcol_theta", "mpcol_rho", "mpcol_pii", "mpcol_dz", "mpcol_z8w"),
        "write_before_read", "woof/core/microphysics.py:_apply_column_scheme",
        "The shared column prologue assigns the complete fields before each launch."),
    ScratchSlotLifetime(
        _wsm_family_preparation_slots(),
        "write_before_read", "woof/core/wsm_family.py:apply",
        "WSM-family preparation fully assigns every array before the "
        "column kernel reads it; precipitation accumulators are separate "
        "carrying slots"),
    ScratchSlotLifetime(
        ("wsm6_theta", "wsm6_rho", "wsm6_pii", "wsm6_dz",
         "wsm6_z8w"), "write_before_read",
        "woof/core/wsm6.py:89-98",
        "WSM6 preparation fully assigns each array before the scheme launch "
        "or any dependent read"),
    ScratchSlotLifetime(
        ("wdm6_theta", "wdm6_rho", "wdm6_pii", "wdm6_dz",
         "wdm6_z8w", "wdm6_count_status"), "write_before_read",
        "woof/core/wdm6.py:apply",
        "WDM6 preparation fully assigns each array before the scheme launch "
        "or any dependent read"),
    ScratchSlotLifetime(
        ("morr_theta", "morr_rho", "morr_pii", "morr_dz",
         "morr_work", "morr_ltrue", "morr_z8w"), "write_before_read",
        "woof/core/morrison.py:apply; woof/core/kernels/morrison.cu:"
        "mo_process, mo_sediment_64/_256, mo_finalize",
        "Morrison preparation fields are rebuilt for every scheme call; "
        "LTRUE is zeroed before the first column call; each column call's "
        "mo_process writes every WORK plane of every cell of its columns "
        "(tendencies, temperature, CPM, LAMI, ACN and the zeroed fallout "
        "tendencies); that call's sedimentation and final loop read only "
        "what it wrote (WRF's MORR_TWO_MOMENT_MICRO zeroes its tendencies "
        "at :1275-1418 of every column call)"),

    ScratchSlotLifetime(
        ("my2_theta", "my2_pii", "my2_t", "my2_z", "my2_z8w", "my2_psfc"),
        "write_before_read", "woof/core/milbrandt2.py::apply",
        "Milbrandt-Yau preparation assigns every element of each array "
        "before the launch, and the surface pressure is derived from the "
        "same call's geopotential"),
    ScratchSlotLifetime(
        ("my2_pres", "my2_de", "my2_ide", "my2_gamfact", "my2_qsw",
         "my2_qsi", "my2_qc_in", "my2_qr_in", "my2_nc_in", "my2_nr_in"),
        "write_before_read", "woof/core/kernels/milbrandt2.cu"
        "::milbrandt2_prelim",
        "the Part 1 kernel writes every one of these for every cell before "
        "any later kernel reads them; entry contents are never consulted"),
    ScratchSlotLifetime(
        ("my2_dz", "my2_idz"), "write_before_read",
        "woof/core/kernels/milbrandt2.cu::milbrandt2_geometry",
        "the geometry kernel writes both for every cell from the Part 1 "
        "density and pressure, before the cold/warm/sedimentation kernels "
        "read them"),
    ScratchSlotLifetime(
        ("refl_t",), "write_before_read", "woof/core/refl.py:349-372; "
        "woof/core/nssl2_runtime.py:apply_nssl2_production; "
        "woof/core/morrison.py:apply (every column call copies its T1D)",
        "post-scheme temperature preparation is assigned after the final "
        "condensation hook and before the reflectivity launch"),
    ScratchSlotLifetime(
        ("physics_qtot", "physics_qi", "physics_qs"), "write_before_read",
        "woof/core/physics.py:387-414",
        "physics preparation explicitly zeroes these arrays before reads"),
    ScratchSlotLifetime(
        ("uwpbl_zero",), "write_before_read",
        "woof/core/physics.py:_run_uwpbl",
        "the UW PBL call assigns zero to every element before the launch "
        "reads it as WRF's zero QNC_CURR/WSEDL3D, and nothing reads it "
        "after the launch"),
    ScratchSlotLifetime(
        ("physics_validation_status",), "write_before_read",
        "woof/core/physics.py:_run_ysu; "
        "woof/core/physics.py:_run_shinhong; "
        "woof/core/physics.py:_validate_native_microphysics; "
        "woof/core/physics.py:_validate_native_kf_result; "
        "woof/core/ysu.py:validate_ysu_outputs; "
        "woof/core/shinhong.py:invalid_shinhong_outputs",
        "the float32 backing is viewed as uint32 and reset before every "
        "validation launch; its blocking scalar read completes before the "
        "next sequential domain can reuse the shared-arena word"),
    ScratchSlotLifetime(
        ("lbc_qv_held", "lbc_nwfa_held", "lbc_nifa_held", "lbc_relax_u", "lbc_relax_v",
         "lbc_relax_theta", "lbc_relax_phi", "lbc_relax_w"),
        "write_before_read",
        "woof/core/moist.py:263-281; woof/ingest/lateral_bc.py:611-628",
        "RK stage 1 captures held tendencies before later stages consume them"),
    ScratchSlotLifetime(
        ("lbc_old_mup_frame_*",), "write_before_read",
        "woof/ingest/lateral_bc.py:658-728",
        "the MU install kernel writes the frame before fused field finalizers"),
    ScratchSlotLifetime(
        ("lbc_nested_relax",), "write_before_read",
        "woof/ingest/lateral_bc.py:apply_state_lateral_boundaries; "
        "woof/core/acoustic.py:prepare_acoustic_coefficients",
        "each nested field overwrites the temporary before immediate use; "
        "the following acoustic preparation overwrites its aliased backing "
        "before any acoustic read"),
    ScratchSlotLifetime(
        ("cu_expiring",), "excluded_unproven",
        "woof/core/physics.py:_advance_cumulus_clock,finish_step",
        "the mask is cleared immediately after the pre-RK compose, but "
        "internal substeps can read that carried zero without a same-step "
        "overwrite; retain per-domain "
        "identity rather than overclaim write-before-read sharing"),
    # EXCLUDED (unproven): each snapshot is fully written at capture and
    # read back at restore within one microphysics.apply call, but the
    # whole scheme dispatch (which allocates and writes its own scratch)
    # runs BETWEEN that write and read -- sharing a backing with any
    # dispatch-written slot would corrupt the ring restore.  Tiny
    # (~2*(nx+ny)*nz per field); correctness beats the savings.
    ScratchSlotLifetime(
        ("mp_ring_save_*",), "excluded_unproven",
        "woof/core/microphysics.py:_capture_spec_zone_ring,"
        "_restore_spec_zone_ring",
        "the snapshot must survive the full scheme dispatch between its "
        "capture write and restore read; retain per-domain identity"),
    ScratchSlotLifetime(
        ("integration_health_partial", "integration_health_result"),
        "write_before_read",
        "woof/core/dycore.py:1474-1486",
        "the two reduction kernels overwrite their outputs before host read"),
    # EXCLUDED (carrying): T15's validator launch tables are immutable
    # setup-time category maps filled once in HealthValidator.__init__
    # (woof/core/health.py:476-491, "immutable setup-time category maps")
    # and reused every step; arena-sharing across domains would corrupt
    # them.  Tiny (~48 KB/domain); classified at the T15 merge.
    ScratchSlotLifetime(
        ("integration_health_field_ptr", "integration_health_aux_ptr",
         "integration_health_field_size", "integration_health_bounds",
         "integration_health_flags", "integration_health_planes",
         "integration_health_status_bits", "integration_health_validation",
         "integration_health_rim"),
        "carrying",
        "woof/core/health.py:476-496",
        "setup-time launch tables persist across steps, and the status/"
        "result buffers may be read asynchronously by supervision between "
        "steps; per-domain only (conservative; ~61 KB/domain with the "
        "12 KiB strict-rim geometry table)"),

    # EXCLUDED (carrying): these microphysics accumulators are read-modify-
    # write and restart-serialized (restart.py:148-158; kessler.cu:91-92;
    # morrison.cu:1087-1103), so per-domain identity must be preserved.
    ScratchSlotLifetime(
        ("mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
         "mp_graupelnc", "mp_graupelncv", "mp_hailnc", "mp_hailncv",
         "mp_sr", "mp_kessler_sr"),
        "carrying", "woof/io/restart.py:148-158",
        "restart-visible live microphysics accumulator/diagnostic state"),
    # EXCLUDED (carrying): correct mode WB1's settled-cloud total and its
    # per-call amount (woof/core/thompson_cloud_booking.py), read-modify-
    # write and restart-serialized like the precipitation accumulators.
    ScratchSlotLifetime(
        ("mp_cloudnc", "mp_cloudncv"),
        "carrying", "woof/core/thompson_cloud_booking.py; woof/io/restart.py "
        "SERIALIZED_SCRATCH_SLOTS",
        "restart-visible correct-mode cloud-booking accumulator"),
    # EXCLUDED (carrying): KF initialization stores these arrays on the
    # driver and later calls update them (physics.py:474-500); restart.py:
    # 148-158 serializes the same slots.
    ScratchSlotLifetime(
        ("cu_rainc", "cu_nca", "cu_pratec", "cu_raincv", "cu_rthcuten",
         "cu_rqvcuten", "cu_rqccuten", "cu_rqicuten", "cu_rqrcuten",
         "cu_rqscuten"), "carrying",
        "woof/core/physics.py:474-500; woof/io/restart.py:148-158",
        "KF timers, rates, and precipitation persist across scheme calls"),
    # EXCLUDED (carrying): the output driver retains this exact view after the
    # producing step until consumption (physics.py:468-472; refl.py:376-399).
    ScratchSlotLifetime(
        ("refl_10cm",), "carrying", "woof/core/refl.py:376-399",
        "one-frame output handoff can outlive the producing domain step"),
    # DA reflectivity-operator temporaries: both are filled in full at the
    # top of _nssl_reflectivity (rho from 1/alt, T from theta and Exner)
    # before diagnose_radardd02_if_due reads them, inside one H(x) call.
    ScratchSlotLifetime(
        ("da_nssl_rho", "da_nssl_t"), "write_before_read",
        "woof/da/obsop.py:_nssl_reflectivity",
        "the observation operator fills dry-air density and temperature "
        "before the shared NSSL diagnostic reads them; both are dead when "
        "the call returns"),
    # The Milbrandt-Yau arm of the same operator: milbrandt2.reflectivity
    # diagnoses T from theta and Exner when its caller has no
    # microphysics-time pair, and the lifted Z kernel reads it in the same
    # call.
    ScratchSlotLifetime(
        ("my2_zet_t",), "write_before_read",
        "woof/core/milbrandt2.py:reflectivity",
        "the reflectivity operator fills absolute temperature before the "
        "lifted Z block reads it; it is dead when the call returns"),
    # EXCLUDED (carrying): UP_HELI_MAX is a restart-serialized elementwise
    # running max, read-modify-written every step and consumed by history
    # frames; per-domain identity must be preserved.
    ScratchSlotLifetime(
        ("up_heli_max",), "carrying",
        "woof/core/uh_diag.py:update_up_heli_max,reset_up_heli_max; "
        "woof/io/restart.py:SERIALIZED_SCRATCH_SLOTS",
        "restart-visible running-max accumulator with a reset only at "
        "history writes"),
    # UP_HELI_MAX work planes: uh_columns writes every cell of both planes
    # (edge threads included) before uh_smooth_max reads them, all inside
    # one update_up_heli_max call.
    # The consumer-owned tracking windows: the same running-max operator
    # as up_heli_max, folded in the same pass, but reset by the consumer
    # that read them rather than by the history writer, and NOT restart
    # visible -- a restart starts them empty and the first post-restart
    # evaluation may under-read, which is the tolerated-experiment
    # posture the moving-nest and spawn restart rulings already take.
    ScratchSlotLifetime(
        ("uh_follow_window", "uh_spawn_window", "uh_follow_window.d*"), "carrying",
        "woof/core/uh_diag.py:update_up_heli_max,reset_tracker_window; "
        "woof/io/restart.py:CARRIED_SCRATCH_SLOTS",
        "per-consumer running-max windows, reset at every evaluation of "
        "the consumer that owns them and never emitted"),
    ScratchSlotLifetime(
        ("uh_diag_col", "uh_diag_use"), "write_before_read",
        "woof/core/kernels/uh_diag.cu:uh_columns,uh_smooth_max; "
        "woof/core/uh_diag.py:update_up_heli_max",
        "per-launch column UH and use_column planes, fully rewritten by "
        "every diagnostic call before the smoother reads them"),
    # EXCLUDED (unproven): unlike physics_qi, these dry placeholders are only
    # zero-initialized by DomainState.scratch and are not rewritten in
    # _prepare_atmosphere (physics.py:404-414). Correctness beats tiny savings.
    ScratchSlotLifetime(
        ("physics_dry_qv", "physics_dry_qc"), "excluded_unproven",
        "woof/core/physics.py:404-414",
        "constant-zero placeholders lack a per-step write-before-read"),
    # EXCLUDED (unproven): the column-batch adapter pair is fully rewritten
    # by each run_phase2 call (cp.divide/cp.multiply with out=) before the
    # microphysics dispatch reads it, but the whole WSM6 dispatch -- which
    # allocates and writes its own scratch -- runs BETWEEN that write and
    # the scheme's reads, the mp_ring_save_* hazard.  The seam never builds
    # a shared arena anyway; correctness beats the savings.
    ScratchSlotLifetime(
        ("physics_column_alt", "physics_column_php"), "excluded_unproven",
        "woof/core/mpas_column_batch.py:769-774",
        "the adapter pair must survive the WSM6 dispatch between its write "
        "and the scheme's reads; retain per-domain identity"),
    # EXCLUDED (carrying/setup): weights are cached by key and forcing views
    # remain attached across all steps (lateral_bc.py:282-301,535-577).
    ScratchSlotLifetime(
        ("lbc_weights_*", "lbc_forcing_tables", "lbc_evaluated_tables",
         # The chem rows' packed outer boundary rows, cut once per served
         # interval and read by every RK stage until it changes.
         "lbc_chem_*"), "carrying",
        "woof/ingest/lateral_bc.py:282-301,535-577,chem_boundary_rows",
        "resident forcing tables and cached weights are cross-step setup"),
    # MYNN's declared workspace.  Split three ways on purpose.
    #
    # WRITE-BEFORE-READ: the kernel that owns the slot fills the whole
    # requested prefix before anything reads it, verified against the CUDA
    # source rather than assumed.  Two of them needed a code change to earn
    # the classification: mynn_pbl.cu:945 returns before k == 0, so the nine
    # mym_turbulence products and the seven full-column level-2 fields kept
    # the surface zero of a fresh allocation.  mynn_pbl_gpu now zeroes that
    # one element explicitly.  MynnPblScratch.poison() is the runtime lever
    # and tests/test_mynn_pbl_scratch.py drives it: NaN every slot in this
    # group, run a carried forecast, require the same hash.
    ScratchSlotLifetime(
        ("mynn_pbl_prep", "mynn_pbl_zw", "mynn_pbl_surface",
         "mynn_pbl_delt", "mynn_pbl_diss_heat", "mynn_pbl_exchange",
         "mynn_pbl_pblh", "mynn_pbl_level2_pairs", "mynn_pbl_level2_out",
         "mynn_pbl_level2_full", "mynn_pbl_mixlength",
         "mynn_pbl_mixlength_work", "mynn_pbl_turbulence",
         "mynn_pbl_predict", "mynn_pbl_predict_work",
         "mynn_pbl_condensation", "mynn_pbl_initialize",
         "mynn_pbl_gsd41_condensation_work", "mynn_pbl_gsd41_pblh_thetav",
         "mynn_pbl_initialize_work", "mynn_pbl_plume_layer",
         "mynn_pbl_plume_face", "mynn_pbl_plume_column",
         "mynn_pbl_plume_work", "mynn_pbl_plume_scratch",
         "mynn_pbl_gsd41_condensation_work", "mynn_pbl_gsd41_thvl",
         "mynn_pbl_tendency", "mynn_pbl_tendency_work",
         "mynn_pbl_tendency_face", "mynn_pbl_stage_layer",
         "mynn_pbl_stage_dx", "mynn_pbl_out_du", "mynn_pbl_out_dv",
         "mynn_pbl_out_dtheta", "mynn_pbl_out_dqv", "mynn_pbl_out_dqc",
         "mynn_pbl_out_dqi"),
        "write_before_read",
        "woof/core/mynn_pbl_scratch.py; woof/core/mynn_pbl_gpu.py; "
        "woof/core/kernels/mynn_pbl.cu:945,2482-2489,2799-2821,1400-1406; "
        "mynn_condensation_gsd41_columns; mynn_gsd41_thvl_columns",
        "each solver leaf fills its own outputs and work vectors before any "
        "reader; the six returned A-grid tendency fields are fully written "
        "across the chunk walk and are dead the moment "
        "couple_ysu_tendencies has multiplied them into new arrays; GSD "
        "condensation writes all five consumed work vectors before its "
        "second loop reads them, and the GSD driver fills liquid-water "
        "virtual theta before every PBL-height reader",
    ),
    # EXCLUDED (constant): nothing writes these.  WRF passes them to systems
    # this lane's pinned identity switches off, and every reader requires
    # them to be zero -- so they are read-before-write by construction, the
    # same reason physics_dry_qv/physics_dry_qc are excluded above.  Sharing
    # a backing with a slot that IS written would put a nonzero mass flux
    # into a tendency solve that is supposed to have none.
    ScratchSlotLifetime(
        ("mynn_pbl_zero_layer", "mynn_pbl_zero_face",
         "mynn_pbl_zero_column", "mynn_pbl_plume_zero_layer",
         "mynn_pbl_plume_zero_face", "mynn_pbl_tendency_zero"),
        "excluded_unproven",
        "woof/core/mynn_pbl_scratch.py:MYNN_PBL_CONSTANT_ZERO_SLOTS; "
        "woof/core/mynn_pbl_gpu.py:mynn_bl_driver_cuda",
        "constant-zero feeds for the mass-flux, subsidence, detrainment, "
        "snow, ozone, stochastic and ocean-current systems this identity "
        "disables; they have no per-step write to be before",
    ),
    # EXCLUDED (int32): ScratchArena is float32-only by construction, so an
    # index slot cannot draw a view from it.  Excluding them here is what
    # makes DomainState.scratch fall back to a per-state int32 allocation
    # instead of raising on the dtype.
    ScratchSlotLifetime(
        ("mynn_pbl_kpbl", "mynn_pbl_pblh_index", "mynn_pbl_plume_index",
         "mynn_pbl_validity_flags"),
        "excluded_unproven",
        "woof/core/state.py:ScratchArena.view; "
        "woof/core/mynn_pbl_scratch.py:MYNN_PBL_INDEX_SLOTS",
        "int32 one-based level indices and the validity words; the shared "
        "arena admits float32 only",
    ),
    ScratchSlotLifetime(
        ("nest_parent_field", "nest_child_field"), "write_before_read",
        "woof/core/nest.py:_coupled_parent_field,_coupled_child_field,force",
        "each field coupling overwrites the full requested prefix before "
        "bdy_interp1 reads it; the two simultaneous fields use distinct "
        "backings and FORCE ends before any aliased RK-stage read"),
    ScratchSlotLifetime(
        # THE LIST IS nest_field_kinds' RANGE and must stay it.  A kind that
        # crosses a nest edge and has no row here is not a slow path: the
        # arena audit RAISES KeyError, so `woof domain --ladder 12-3` --
        # the wizard, the front door for every nested run -- dies with an
        # unhandled traceback before it can emit anything.  P3's rime mass
        # and rime volume were exactly that: nest_field_kinds gained
        # "qir"/"qib" with the mp=50 arm and this tuple did not, so no
        # nested P3 configuration could be authored at all.  MEASURED:
        # `KeyError: scratch slot 'nest_qir_bxs' has no lifetime audit row`.
        # tests/test_p3_front_door.py walks every shipped mp_physics against
        # this row so the next scheme cannot repeat it -- and that walk
        # found a SECOND hole of the same class while it was being written:
        # Milbrandt-Yau (mp=9) forces "nh", its hail number moment, and had
        # no row either, so a nested mp=9 wizard emission died the same way.
        tuple(f"nest_{kind}_b*" for kind in
              ("u", "v", "w", "t", "ph", "mu", "qv", "qc", "qr",
               "qi", "qs", "qg", "nr", "ni", "ns", "ng", "nh", "qh",
               "qir", "qib",
               "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
               "qvolg", "qvolh", "nc", "nn", "nwfa", "nifa",
               # MYNN TKE advection's scalar (nest_field_kinds appends it
               # when bl_pbl_physics=5 and bl_mynn_tkeadvect).  MEASURED
               # without it: every nested run with TKE advection on (the
               # rrfs-v1 suite, HRRR's MYNN settings) died at preflight
               # with `KeyError: scratch slot 'nest_qke_adv_bxs' has no
               # lifetime audit row` (LA 750 m, woof 2.9.9, 2026-10-29).
               "qke_adv"))
        # Chem rows' rolling frames, one per transported row a nest carries
        # (woof/core/nest_fields.py), named by the chem table.
        + ("nest_chem_*", "nest_ff1i*", "nest_ff5i*", "nest_ff6i*", "nest_ff8i*"),
        "carrying", "woof/core/nest.py:force; "
        "woof/ingest/lateral_bc.py:attach_nest_boundaries",
        "rolling value/tendency frames are consumed through the complete "
        "child subcycle and must remain child-owned"),
    ScratchSlotLifetime(
        ("nest_sint_*",), "carrying",
        "woof/core/nest.py:_bind_geometry; "
        "woof/core/nest_interp.py:NestRegistration.device_tables",
        "setup geometry is bound once and reused by every force"),
)


_scratch_lifetime_index = (None, None)


def _index_scratch_lifetimes(audit):
    """Index literal rules once; memoize the much smaller prefix search.

    Domain sizing repeatedly prices the same slot names at different shapes.
    Scanning every pattern of every row for each request made that metadata
    walk the wizard's single-core tail. The audit is immutable; replacing it
    (including in an audit test) builds a new index and a new lookup cache.
    Row numbers, rather than patterns, preserve the unique-ROW overlap rule.
    """
    from functools import lru_cache

    exact, prefixes = {}, []
    for index, row in enumerate(audit):
        for pattern in row.patterns:
            if pattern.endswith("*"):
                prefixes.append((pattern[:-1], index))
            else:
                exact.setdefault(pattern, set()).add(index)

    @lru_cache(maxsize=8192)
    def lookup(slot):
        matches = set(exact.get(slot, ()))
        matches.update(index for prefix, index in prefixes if slot.startswith(prefix))
        if len(matches) > 1:
            raise RuntimeError(f"scratch lifetime audit overlaps for {slot!r}")
        return audit[next(iter(matches))] if matches else None

    return lookup


def scratch_slot_lifetime(slot: str) -> ScratchSlotLifetime | None:
    """Return the unique audit row for ``slot``, or ``None`` if unclassified."""
    global _scratch_lifetime_index
    audit, lookup = _scratch_lifetime_index
    if audit is not SCRATCH_SLOT_LIFETIME_AUDIT:
        audit = SCRATCH_SLOT_LIFETIME_AUDIT
        lookup = _index_scratch_lifetimes(audit)
        _scratch_lifetime_index = audit, lookup
    return lookup(slot)


def scratch_slot_uses_arena(slot: str) -> bool:
    row = scratch_slot_lifetime(slot)
    if row is None:
        raise KeyError(f"scratch slot {slot!r} has no lifetime audit row")
    return row.arena_eligible


def shared_scratch_arena_shapes(
        domains: tuple[DomainConfig, ...],
        tree: tuple[DomainConfig, ...] | None = None,
        ) -> dict[str, tuple[int, ...]]:
    """Max request shape per admitted slot for the sequential-domain arena.

    Max is by element count because :class:`ScratchArena` returns contiguous
    reshaped prefix views. Ties retain the first (parent-first) request. The
    runtime builder and the estimator both call this exact function.

    ``domains`` are the domains that SHARE the arena (the resident ones);
    ``tree`` is the whole configured tree they belong to, consulted only to
    find a resident child's parent when that parent is not itself resident.
    A root that streams while its child stays resident is a legal road
    (the plan report prices it), and the child's force slots are sized
    from the parent's own field capacity either way; looking the parent up
    among the resident domains alone raised ``KeyError`` on exactly that
    road.  With ``tree`` omitted every parent is expected among ``domains``.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    for dc in domains:
        for slot, shape in scratch_slot_registry(
                dc.run, n_lbc_intervals=0).items():
            if not scratch_slot_uses_arena(slot):
                continue
            shape = tuple(shape)
            if slot not in shapes or math.prod(shape) > math.prod(shapes[slot]):
                shapes[slot] = shape
    if all(hasattr(dc, "grid_id") and hasattr(dc, "parent_id")
           for dc in domains):
        by_id = {dc.grid_id: dc for dc in (tree if tree is not None else ())}
        by_id.update({dc.grid_id: dc for dc in domains})
        for dc in domains:
            if dc.parent_id == 0:
                continue
            if dc.parent_id not in by_id:
                raise KeyError(
                    f"domain {dc.grid_id} names parent {dc.parent_id}, which "
                    "is neither among the arena's domains nor in the tree "
                    "handed to shared_scratch_arena_shapes; pass the whole "
                    "configured tree as `tree`")
            force_shapes = {
                "nest_parent_field": _full_field_capacity(
                    by_id[dc.parent_id].run),
                "nest_child_field": _full_field_capacity(dc.run),
            }
            for slot, shape in force_shapes.items():
                if (slot not in shapes
                        or math.prod(shape) > math.prod(shapes[slot])):
                    shapes[slot] = shape
    return shapes


def shared_scratch_arena_aliases(
        domains: tuple[DomainConfig, ...],
        tree_domains: tuple[DomainConfig, ...] | None = None) -> dict[str, str]:
    """Disjoint-lifetime arena aliases admitted by the reviewed audit.

    FORCE occurs between complete domain steps.  All three RK backings are
    dead after the preceding STEP and overwritten before their next stage
    read.  Parent and child coupled fields are simultaneously live, so the
    alias assignment below admits them only onto two distinct capacity-safe
    RK backings.  Preference order preserves F16's historical parent->rk_ru,
    child->rk_ww addresses on ordinary production grids; an explicit logical
    backing remains when no safe pair exists.

    Sixth-order diffusion itself processes targets serially.  On a diff6-only
    configuration x/y/m can all reuse the largest z backing.  The combined
    Smagorinsky path also borrows x/y, and needs both face buffers
    simultaneously for momentum staging and scalar fluxes.  If any domain
    enables km_opt=4, x and y therefore remain distinct while x/m may still
    reuse z.

    Smagorinsky K_m/K_h are consumed only while the pre-RK preparation builds
    the held mixing tendencies.  Each stage later prepares acoustic alpha and
    gamma from scratch before its first acoustic read. Strict coordinate
    scalar diffusion reads Kh after the first acoustic loop, so that arm
    keeps Kh separate from acoustic gamma.
    The two mass-point K arrays can therefore borrow prefixes of those two
    independent full-level coefficient backings.

    ``tree_domains`` supplies the complete tree when only resident domains
    share the arena. Alias capacity uses the same parent-aware shapes as
    the allocation and byte estimate, including a streamed parent.
    """
    aliases = {}
    shapes = shared_scratch_arena_shapes(domains, tree_domains)
    # The standalone advection-only adv_* path and the acoustic RK path are
    # mutually exclusive inside dycore.step.  cq overwrites all three faces
    # once per RK stage before calc_coefs/advance_uv/advance_w reads them.
    for cq, adv in (("acoustic_cqu", "adv_ru"),
                    ("acoustic_cqv", "adv_rv"),
                    ("acoustic_cqw", "adv_rw")):
        if (cq in shapes and adv in shapes
                and math.prod(shapes[cq]) <= math.prod(shapes[adv])):
            aliases[cq] = adv
    parent_slot = "nest_parent_field"
    child_slot = "nest_child_field"
    rk_targets = ("rk_ru", "rk_ww", "rk_rv")
    pair_preferences = (
        ("rk_ru", "rk_ww"), ("rk_ru", "rk_rv"),
        ("rk_rv", "rk_ww"), ("rk_ww", "rk_ru"),
        ("rk_ww", "rk_rv"), ("rk_rv", "rk_ru"),
    )

    def force_slot_fits(slot: str, target: str) -> bool:
        return (slot in shapes and target in shapes
                and math.prod(shapes[slot]) <= math.prod(shapes[target]))

    force_pair = next((
        (parent_target, child_target)
        for parent_target, child_target in pair_preferences
        if force_slot_fits(parent_slot, parent_target)
        and force_slot_fits(child_slot, child_target)
    ), None)
    if force_pair is not None:
        aliases[parent_slot], aliases[child_slot] = force_pair
    else:
        # Correctness-first fallback: alias only the larger logical request
        # when one dead RK backing can hold it; the other remains explicit.
        logical_slots = sorted(
            (slot for slot in (parent_slot, child_slot) if slot in shapes),
            key=lambda slot: math.prod(shapes[slot]), reverse=True)
        for slot in logical_slots:
            target = next((candidate for candidate in rk_targets
                           if force_slot_fits(slot, candidate)), None)
            if target is not None:
                aliases[slot] = target
                break
    if ("lbc_nested_relax" in shapes and "acoustic_a" in shapes
            and math.prod(shapes["lbc_nested_relax"])
            <= math.prod(shapes["acoustic_a"])):
        aliases["lbc_nested_relax"] = "acoustic_a"
    from woof.config import wrf_mixing_package_active
    smag_uses_xy = any(wrf_mixing_package_active(dc.run) for dc in domains)
    if "diff6_z" in shapes:
        candidates = (("diff6_x", "diff6_m") if smag_uses_xy
                      else ("diff6_x", "diff6_y", "diff6_m"))
        for slot in candidates:
            if (slot in shapes
                    and math.prod(shapes[slot])
                    <= math.prod(shapes["diff6_z"])):
                aliases[slot] = "diff6_z"
    keep_scalar_kh = WRF_EXACT and any(
        dc.run.diff_opt == 1 and dc.run.km_opt in (2, 3, 4) for dc in domains)
    for slot, target in (("smag_km", "acoustic_alpha"),
                         ("smag_kh", "acoustic_gamma")):
        if slot == "smag_kh" and keep_scalar_kh:
            continue
        if (slot in shapes and target in shapes
                and math.prod(shapes[slot]) <= math.prod(shapes[target])):
            aliases[slot] = target
    return aliases


def shared_scratch_arena_bytes(
        domains: tuple[DomainConfig, ...],
        tree: tuple[DomainConfig, ...] | None = None) -> int:
    shapes = shared_scratch_arena_shapes(domains, tree)
    aliases = shared_scratch_arena_aliases(domains, tree)
    return sum(4 * math.prod(shape) for slot, shape in shapes.items()
               if slot not in aliases)


# ---------------------------------------------------------------------------
# RRTMGP workspace + per-step transient formulas
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _gas_table_meta() -> dict[str, int]:
    """ngpt/ngas per band from the shipped k-distributions (lazy import --
    the estimator stays CPU-only; table loading is host NetCDF I/O).
    Read from the packaged receipt when the installed tables match it, so a
    plan needs no native NetCDF decoder (see woof.core.rrtmgp.packaged_table_facts)."""
    from woof.core.rrtmgp import gas_table_shape

    lw = gas_table_shape("lw")
    sw = gas_table_shape("sw")
    return {"ngpt_lw": lw.ngpt, "ngpt_sw": sw.ngpt,
            "ngas_lw": lw.ngas, "ngas_sw": sw.ngas,
            "nband_lw": lw.nband, "nband_sw": sw.nband}


@lru_cache(maxsize=1)
def k_distribution_bytes() -> int:
    """Device bytes of the lru_cache-shared k-distribution/cloud tables,
    counted ONCE per process (rrtmgp.py:324/:436 -- baseline behavior,
    never claimed as savings).  Uses the to_device dtype rule
    (rrtmgp.py:266-282): float -> f32, int -> i32, bool -> 1 byte.
    Read from the packaged receipt when the installed tables match it."""
    from woof.core.rrtmgp import packaged_table_facts

    facts = packaged_table_facts()
    if facts is not None:
        return int(facts["bytes"])
    return _decoded_k_distribution_bytes()


def _decoded_k_distribution_bytes() -> int:
    import numpy as np
    from woof.core.rrtmgp import load_cloud_tables, load_gas_tables

    total = 0
    for tables in (load_gas_tables("lw"), load_gas_tables("sw"),
                   load_cloud_tables("lw"), load_cloud_tables("sw")):
        for value in vars(tables).values():
            if isinstance(value, np.ndarray):
                itemsize = 1 if value.dtype == bool else 4
                total += value.size * itemsize
    return total


def rrtmgp_workspace_phases(nz: int, column_chunk: int, p_top: float = 5000.0
                            ) -> dict[str, dict[str, tuple[tuple[int, ...],
                                                           int]]]:
    """Per-chunk SIMULTANEOUS live sets, one dict per solver phase.

    ONE shared working set sized to the fixed column chunk, reused by all
    four domains (architecture section E RRTMGP CHUNK POLICY: legal
    because domains step strictly sequentially and workspace shape is
    f(chunk_columns, nz, p_top) with nz/p_top identical everywhere).  The
    workspace
    bound is the MAXIMUM over phases of each phase's exact live set --
    not a union with substitutions (p5t11 shadow review F2):

    * ``lw_optics`` (rrtmgp.py:1291-1306): gas tau + finalized tau alive
      together with the McICA bool mask, per-chunk VMR, band cloud
      optics, and col_dry (:1615, retained by BOTH the gas and finalized
      optics results -- one shared array, counted once).
    * ``lw_rte`` (:1307-1317): only what it READS BACK -- the finalized tau
      and the VMR Planck consumes -- plus its own Planck lay/lev/sfc sources
      (:1747-1749), emissivity/incident g-point arrays and two flux outputs.
      gas_tau, the three band cloud cubes and col_dry are dead the moment
      the finalized optics exist, so this phase lies its outputs over them
      rather than appending after them.
    * ``sw_optics`` (:1353-1368): gas tau/ssa + finalized tau/ssa/g (five
      g-point cubes) + mask + VMR + band cloud optics + col_dry.
    * ``sw_rte`` (:1369-1381): the three finalized cubes it reads, plus
      albedo/incidence g-point arrays, the materialized (chunk,nz) mu0
      broadcast (:1690-1691) and three (chunk,nz+1) flux arrays over the
      dead gas/cloud/VMR tail.  SW builds no Planck source, so unlike LW it
      does not carry vmr either.
    """
    from woof.core.rrtmgp import rrtmgp_above_model_layer_counts

    meta = _gas_table_meta()
    c = int(column_chunk)
    lw_g, sw_g = meta["ngpt_lw"], meta["ngpt_sw"]
    lw_upper, sw_upper = rrtmgp_above_model_layer_counts(p_top)
    lw_nz, sw_nz = int(nz) + lw_upper, int(nz) + sw_upper
    if max(lw_nz, sw_nz) > 128:
        raise ValueError(
            "RRTMGP workspace cannot exceed the 128-layer CUDA solver limit")

    # CARRIED FIRST.  An RTE phase reads back only a few of its optics
    # phase's slots, and `phase()` assigns offsets by walking the layout in
    # order -- so a carried slot keeps its address only if everything ahead
    # of it does too.  Listing the carried ones first makes the rest ONE
    # contiguous tail that the RTE phase lays its own outputs over and then
    # stops, with no padding anywhere.  Reordering inside an optics phase is
    # free: every slot there is written before it is read in that same phase
    # (RRTMGP_WORKSPACE_LIFETIME_AUDIT).
    #
    # Dropping the dead slots WITHOUT the reorder recovers much less --
    # the holes are scattered, each has to be padded to hold the next slot
    # in place, and lev_source misses gas_tau's hole by a few percent.
    # The finalize is fused into the solvers, so the finalized optics
    # cubes DO NOT EXIST.  The carried set is what the fused solver reads:
    # the gas cube(s), the band cloud cubes it consumes, and the McICA
    # mask.  The mask is 1-byte and sits LAST among the carried slots; its
    # byte count is c*nz*ngpt with ngpt a multiple of 32, so every 4-byte
    # slot after it stays aligned, and `phase()` refuses an unaligned one
    # anyway.
    lw_carried = {
        "gas_tau": ((c, lw_nz, lw_g), 4),
        "vmr": ((c, lw_nz, meta["ngas_lw"] + 1), 4),
        "cld_tau": ((c, lw_nz, meta["nband_lw"]), 4),
        "cld_ssa": ((c, lw_nz, meta["nband_lw"]), 4),
        "mcica_mask": ((c, lw_nz, lw_g), 1),
    }
    lw_dead_in_rte = {
        "cld_asy": ((c, lw_nz, meta["nband_lw"]), 4),
        "col_dry": ((c, lw_nz), 4),
    }
    sw_carried = {
        "gas_tau": ((c, sw_nz, sw_g), 4),
        "gas_ssa": ((c, sw_nz, sw_g), 4),
        "cld_tau": ((c, sw_nz, meta["nband_sw"]), 4),
        "cld_ssa": ((c, sw_nz, meta["nband_sw"]), 4),
        "cld_asy": ((c, sw_nz, meta["nband_sw"]), 4),
        "mcica_mask": ((c, sw_nz, sw_g), 1),
    }
    sw_dead_in_rte = {
        "vmr": ((c, sw_nz, meta["ngas_sw"] + 1), 4),
        "col_dry": ((c, sw_nz), 4),
    }
    return {
        "lw_optics": {**lw_carried, **lw_dead_in_rte},
        # lay_source/lev_source/sfc_source do not exist: the LW solver
        # derives the Planck sources in registers.  That is 455 MiB of this
        # phase at the default chunk, and it is why lw_rte stopped being
        # the maximum.
        "lw_rte": {**lw_carried,
                   "emiss_gpt": ((c, lw_g), 4),
                   "incident": ((c, lw_g), 4),
                   "flux_up": ((c, lw_nz + 1), 4),
                   "flux_dn": ((c, lw_nz + 1), 4)},
        "sw_optics": {**sw_carried, **sw_dead_in_rte},
        "sw_rte": {**sw_carried,
                   "albedo_gpt": ((c, sw_g), 4),
                   "inc_gpt": ((c, sw_g), 4),
                   "mu0": ((c, sw_nz), 4),
                   "flux_up": ((c, sw_nz + 1), 4),
                   "flux_dn": ((c, sw_nz + 1), 4),
                   "flux_dir": ((c, sw_nz + 1), 4)},
    }


def rrtmgp_workspace_shapes(nz: int, column_chunk: int, p_top: float = 5000.0
                            ) -> dict[str, tuple[tuple[int, ...], int]]:
    """The shared chunk workspace: the phase-maximum simultaneous set,
    ``{"<phase>/<name>": (shape, itemsize)}`` (see
    :func:`rrtmgp_workspace_phases`)."""

    def total(items):
        return sum(math.prod(shape) * size for shape, size in items.values())

    phases = rrtmgp_workspace_phases(nz, column_chunk, p_top)
    phase = max(phases, key=lambda name: total(phases[name]))
    return {f"{phase}/{name}": spec
            for name, spec in phases[phase].items()}


def _named_three_radius_schemes() -> frozenset[int]:
    """Selectors outside WRF's numbering (woof.microphysics_schemes, ids
    from 900) whose cloud-optics family reads effc/effi/effs off the state
    the way Thompson's row does: TEMPO (900) couples as "thompson"
    (woof/core/rrtmgp.py _MP_CLOUD_OPTICS_SCHEME), so its radiation call
    packs the same three radius columns and they are priced here.
    """
    from woof.microphysics_schemes import NAMED_SCHEMES
    return frozenset(int(scheme.mp_id) for scheme in NAMED_SCHEMES.values()
                     if scheme.cloud_optics_family in ("thompson", "wsm6", "wsm3"))


def rrtmgp_column_shapes(
        cfg: RunConfig, p_top: float = 5000.0, *,
        column_chunk: int = DEFAULT_COLUMN_CHUNK,
) -> dict[str, tuple[tuple[int, ...], int]]:
    """Per-domain radiation column packing transients (rrtmgp.py
    ``RRTMGPRadiation.__call__``): full-``ncol`` model arrays plus the
    chunk-local above-model profile/path/cloud/interpolation arrays that
    coexist with the shared solver workspace.  Freed at call end; domains
    step sequentially, so the experiment estimate takes the MAX over
    domains."""
    if 4 not in radiation_scheme_ids(cfg):
        return {}
    from woof.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
    if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
        # The legacy adapter's transients are priced as ONE shared
        # call-peak envelope in estimate_experiment (LW/SW run
        # sequentially per chunk, one domain at a time), not as
        # per-domain RRTMGP column shapes.
        return {}
    from woof.core.rrtmgp import rrtmgp_above_model_layer_counts

    if (isinstance(column_chunk, bool)
            or not isinstance(column_chunk, int)
            or column_chunk < 1):
        raise ValueError("column_chunk must be a positive integer")
    nz = cfg.nz
    lw_upper, sw_upper = rrtmgp_above_model_layer_counts(p_top)
    peak_upper = max(lw_upper, sw_upper)
    peak_nz = nz + peak_upper
    ncol = cfg.ny * cfg.nx
    cap_ncol = min(ncol, column_chunk)
    shapes: dict[str, tuple[tuple[int, ...], int]] = {}
    lay = ["play", "tlay", "qv", "exner", "qc", "qr", "qi", "qs", "cldfra",
           "clwp", "ciwp", "reliq", "dgice"]
    if cfg.mp_physics == 10:
        lay += ["nc", "nr", "ni", "ns", "effc", "effr", "effi", "effs"]
    elif (cfg.mp_physics in (3, 4, 6, 8, 16, 18, 24, 28, 106)
          or cfg.mp_physics in _named_three_radius_schemes()):
        # WSM3 (3) packs its three radii for the same reason: WSM3SCHEME is
        # in the use_mp_re disjunction (module_physics_init.F:1008) and its
        # "wsm3" coupling reads effc/effi/effs (woof/core/rrtmgp.py).
        # WRF's use_mp_re scheme table lists THOMPSONAERO explicitly and
        # separately from THOMPSON in the same disjunction
        # (phys/module_physics_init.F:1005 THOMPSON, :1006 THOMPSONAERO), and
        # the P3/Jensen-Ishmael has_reqs=0 override at :1026-1033 does not
        # exclude it, so has_reqc/has_reqi/has_reqs are all 1 for mp=28 --
        # the same authority woof/core/rrtmg_legacy.py's _MP_DECLARES_RADII
        # already carries.  The mp=28 state allocates effc/effi/effs on the
        # same terms as mp=8 (see state_array_shapes above).
        #
        # STATED RATHER THAN HIDDEN: the RTE+RRTMGP adapter's own scheme map
        # (woof/core/rrtmgp.py:1811) does not yet route 28 and currently
        # falls through to "kessler", which packs no radii columns at all.
        # Pricing them here first is the SAFE direction and the same
        # convention _REFLECTIVITY_MICROPHYSICS above already uses: an
        # over-priced rail refuses a run that would have fit, an under-priced
        # one lets a run breach the budget.  The legacy-RRTMG 4/4 variant is
        # unaffected either way -- it returns {} from this function and is
        # priced as one shared call-peak envelope.
        lay += ["effc", "effi", "effs"]
    elif cfg.mp_physics == 50:
        # P3 packs TWO radius columns, not three, and the missing one is
        # the whole shape of its coupling: WRF puts has_reqs back to 0 for
        # the P3 family (phys/module_physics_init.F:1027-1034) because P3's
        # single ice category has no snow species to have a radius, so
        # Registry.EM_COMMON:3043 gives it state:re_cloud,re_ice and
        # woof/core/state.py allocates effc/effi and no effs.  Pricing an
        # effs column here would budget a per-column array the run never
        # allocates -- over-pricing in the direction that HIDES a real
        # allocation, since it would then look priced for a scheme whose
        # adapter branch refuses one.
        #
        # UNREACHABLE UNTIL NOW, which is why the row is only landing with
        # the coupling: this function returns {} for the legacy-RRTMG 4/4
        # variant, and the RTE+RRTMGP variant refused mp=50 outright until
        # woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME gained ``50: "p3"``.
        # With that refusal retired, an mp=50 RTE+RRTMGP run reaches this
        # branch, and without the row it would reach it priced for no
        # radius columns at all -- the under-priced direction, which lets a
        # run breach the budget rather than refusing it.
        lay += ["effc", "effi"]
    for name in lay:
        shapes[f"columns/{name}"] = ((ncol, nz), 4)
    for name in ("metadata_jt", "metadata_jp", "metadata_iatm",
                  "metadata_ftemp", "metadata_fpress"):
        shapes[f"columns/{name}"] = ((cap_ncol, peak_nz), 4)
    for name in ("plev", "tlev", "lw_up", "lw_dn", "sw_up", "sw_dn"):
        shapes[f"columns/{name}"] = ((ncol, nz + 1), 4)
    # LW's 25-layer the reference case cap is the peak phase.  The live model columns above
    # remain needed for SW, but only one solver chunk's expanded thermo/cloud
    # copies and metadata coexist.  With no representable upper layer the
    # adapter returns the model slices directly and creates no duplicate cap.
    if peak_upper:
        for name in ("play", "tlay", "qv", "cldfra",
                     "clwp", "ciwp", "reliq", "dgice"):
            shapes[f"columns/upper_peak_{name}"] = ((cap_ncol, peak_nz), 4)
        for name in ("plev", "tlev"):
            shapes[f"columns/upper_peak_{name}"] = (
                (cap_ncol, peak_nz + 1), 4)
    shapes["columns/emiss_bands"] = ((ncol, 16), 4)
    for name in ("tsfc", "mu0", "albedo_surface", "daylight"):
        shapes[f"columns/{name}"] = ((ncol,), 4)
    # WRF's top/bottom fluxes (rrtmgp.py RRTMGPRadiation.__call__): the
    # per-flux rows, and one chunk's clear-sky solve -- fresh flux outputs
    # (SW: up, down, direct), the LW zero incident g-point plane and the
    # SW zero band cloud optics.
    from woof.core.physics_inventory import radiation_boundary_flux_names
    boundary = radiation_boundary_flux_names(cfg)
    if boundary:
        shapes["columns/boundary_fluxes"] = ((len(boundary), ncol), 4)
        for name in ("up", "dn", "dir"):
            shapes[f"columns/clear_sky_flux_{name}"] = (
                (cap_ncol, peak_nz + 1), 4)
        shapes["columns/clear_sky_incident"] = ((cap_ncol, 256), 4)
        shapes["columns/clear_sky_cloud"] = ((cap_ncol, peak_nz, 16), 4)
    return shapes


def classic_rrtm_column_shapes(cfg: RunConfig, p_top: float = 5000.0, *,
                               column_chunk: int = DEFAULT_COLUMN_CHUNK
                               ) -> dict[str, tuple[tuple[int, ...], int]]:
    """Classic selector-1 call: actual window packing plus capped chunk peak."""
    if radiation_scheme_ids(cfg)[0] != 1:
        return {}
    # Zero is the existing ideal/legacy wrapper's UNKNOWN pressure top. Its
    # initializer supplies state.p_top later. Retain the previously unpriced
    # classic workspace for that incomplete pre-initialization metadata; do
    # not turn the placeholder into a new refusal or invent buffer layers.
    if p_top == 0:
        return {}
    from woof.core.rrtm_inventory import call_workspace_shapes
    return {f"classic_rrtm/{name}": item for name, item in call_workspace_shapes(
        int(cfg.nx)*int(cfg.ny), int(cfg.nz), p_top, column_chunk).items()}


def dudhia_column_shapes(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Conservative device-transient envelope for Dudhia shortwave.

    The adapter packs ten top-down layer fields.  SWPARA then carries five
    full-column work arrays plus a heating/output reversal; the driver holds
    its returned/coupled copies while validating.  Twenty-four layer-sized
    arrays and forty column-sized vectors bound those simultaneously live
    values and CuPy expression temporaries without pretending they share the
    RRTMGP chunk workspace.
    """
    if radiation_scheme_ids(cfg)[1] != 1:
        return {}
    ncol = cfg.ny * cfg.nx
    return {
        "dudhia/layer_envelope": (24, ncol, cfg.nz),
        "dudhia/column_envelope": (40, ncol),
        "dudhia/lookup_tables": (2, 4, 5),
    }


def atmosphere_transient_shapes(cfg: RunConfig, *, cam_ozone: bool = False
                                ) -> dict[str, tuple[int, ...]]:
    """``_prepare_atmosphere`` per-call transients (physics.py:338-400):
    fresh device arrays alive for the whole physics call, including
    through radiation."""
    from woof.core.physics_inventory import physics_enabled

    if not physics_enabled(cfg) and not cam_ozone:
        return {}
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    m = (nz, ny, nx)
    fl = (nz + 1, ny, nx)
    return {"atmosphere/theta": m, "atmosphere/temperature": m,
            "atmosphere/pressure": m, "atmosphere/exner": m,
            "atmosphere/u": m, "atmosphere/v": m, "atmosphere/dz": m,
            "atmosphere/rho": m,
            "atmosphere/p_interface": fl, "atmosphere/z_interface": fl}


#: Only YSU allocates this raw output bundle. MYNN returns six views of
#: the already priced ``mynn_pbl_out_*`` scratch and writes its diagnostics
#: into the already priced driver fields. Charging the YSU bundle to MYNN
#: counted those arrays twice and added YSU-only diagnostics it never owns.
_YSU_OUTPUT_BUNDLE_SCHEMES = (1,)


def mynn_mixscalars_memory_items(cfg: RunConfig, *, tile_buffer: bool = False
                                ) -> tuple[MemoryItem, ...]:
    """Conditional qn storage outside the always-on MYNN scratch arena.

    The previous four coupled scalars remain held by ``self.tendencies``
    until composition (physics.py:_compose_tendencies). During replacement,
    four raw rates, four new coupled fields and the mass field coexist:
    thirteen volume fields including the previous four. This also bounds
    the earlier mass-expression temporaries, before new coupled fields exist.

    The chunk term bounds the wider DMP phase, including five previous
    chunk tendencies kept by the wrapper's old ``out`` dict. The subsequent
    five scalar solves need fewer bytes, including both returned qn2 arrays
    during assignment and their (7*nz+1)-wide solve scratch. Summing this
    chunk ceiling beside the full coupling peak is conservative; the two
    phases are sequential. Probe-only export_sink copies are not forecast
    allocations. Owners: mynn_pbl_runtime.py, mynn_pbl_gpu.py and
    mynn_scalar_mix_gpu.py.
    """
    # Priced for the ACTIVE key only: with a microphysics that carries no
    # qn family the key is inert (config.mynn_mixscalars_active) and the
    # runtime stages none of this storage.
    if not mynn_mixscalars_active(cfg):
        return ()
    from woof.core.mynn_pbl import DMP_NUP
    from woof.core.mynn_scalar_mix import QN_SOLVE_ORDER

    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    chunk = min(nx * ny, mynn_pbl_column_chunk(cfg, tile_buffer=tile_buffer))
    mass = (nz, ny, nx)
    prefix = "mynn_mixscalars/"
    items = [MemoryItem(prefix + "held_" + name, "physics", mass, 4)
             for name in ("nc", "ni", "nwfa", "nifa")]
    for family in ("raw_", "coupled_"):
        items.extend(MemoryItem(prefix + family + name, "transient", mass, 4)
                     for name in ("nc", "ni", "nwfa", "nifa"))
    items.append(MemoryItem(prefix + "coupling_mass", "transient", mass, 4))
    shapes = {}
    for species in QN_SOLVE_ORDER:
        shapes["input_" + species] = (chunk, nz)
        shapes["previous_tendency_" + species] = (chunk, nz)
        shapes["flux_" + species] = (chunk, nz + 1)
    shapes.update(
        plume_area=(chunk, nz + 1, DMP_NUP),
        plume_velocity_copy=(chunk, nz + 1, DMP_NUP),
        plume_area_copy=(chunk, nz + 1, DMP_NUP),
        plume_entrainment_copy=(chunk, nz, DMP_NUP),
        flux_workspace=(chunk, (nz + 1) * DMP_NUP),
        plume_weight=(chunk,), plume_limiter=(chunk,))
    items.extend(MemoryItem(prefix + name, "transient", shape, 4)
                 for name, shape in shapes.items())
    items.append(MemoryItem(prefix + "plume_active", "transient",
                            (chunk,), 4, "int32"))
    return tuple(items)


def ysu_output_transient_shapes(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Raw per-call YSU outputs before field-copy/coupling consumption.

    These allocations exist on every YSU call.  Positive ``bldt`` retains
    the result after the call (and is also counted as persistent); bldt=0
    releases it at the last consumer, so it appears only in this category.
    """
    # Keyed on the SCHEME, never on the truthiness of the selector: SASE
    # holds the same driver slot but allocates none of YSU's output
    # bundle -- it returns its rates through the coupling helper
    # directly -- so a truthiness test priced a SASE run for fifteen
    # arrays it never asks for.  This mirrors the driver's own rule that
    # a selector VALUE, not its truthiness, decides what runs.
    if int(cfg.bl_pbl_physics) not in _YSU_OUTPUT_BUNDLE_SCHEMES:
        return {}
    m = (cfg.nz, cfg.ny, cfg.nx)
    s2 = (cfg.ny, cfg.nx)
    shapes = {f"ysu_output/{name}": m for name in _YSU_3D}
    shapes.update({f"ysu_output/{name}": s2 for name in _YSU_2D})
    return shapes


def mynn_scalar_transient_shapes(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Peak live FP32 storage added by MYNN ``scalar_pblmix=1``.

    The local solve keeps four raw grid rates and the previous four coupled
    rates alive. Its chunk adds 5*nz+1 work values, two current outputs,
    one packed input and one time step per column, plus the scalar time-step
    input. Outputs are dropped after scatter before the next species.
    Coupling can keep old and new four-field
    coupled banks beside the four raw rates and its mass coefficient.
    Charge the larger phase, rather than summing nonconcurrent phases.
    The scalar-off estimate is unchanged.
    """
    from woof.config import (SCALAR_PBLMIX_SPECIES, scalar_pblmix_exchange,
                              scalar_pblmix_mixes)
    if int(cfg.scalar_pblmix) != 1 or not scalar_pblmix_mixes(cfg):
        return {}
    nz, columns = int(cfg.nz), int(cfg.ny) * int(cfg.nx)
    if not scalar_pblmix_exchange(cfg)["mixes_in_scheme"] or int(cfg.mp_physics) != 28:
        # The shared post-PBL arm (physics._scalar_pblmix_rates, after YSU,
        # MYJ, Shin-Hong or UW, and after MYNN for every microphysics but
        # mp=28, whose family MYNN's wrapper solves): one raw rate per
        # species, its coupled rate and the mass coefficient live together
        # (N = 4 under mp=28 gives the historical 9); each 32768-column
        # chunk adds four packed inputs, 5*nz+1 work values and two outputs
        # per column.  A scheme with no diffused scalar keeps only the mass
        # coefficient.  The YSU key keeps its historical name.
        species = len(SCALAR_PBLMIX_SPECIES.get(int(cfg.mp_physics), ()))
        chunk = min(32768, columns) if species else 0
        key = ("ysu_scalar/local_diffusion_peak" if int(cfg.bl_pbl_physics) == 1
               else "pbl_scalar/local_diffusion_peak")
        return {key: ((2 * species + 1) * nz * columns + (11 * nz + 1) * chunk,)}
    chunk = mynn_pbl_column_chunk(cfg)
    solve = 8 * nz * columns + (8 * nz + 2) * chunk + 1
    coupling = 13 * nz * columns
    return {"mynn_scalar/local_diffusion_peak": (max(solve, coupling),)}


#: PBL schemes that allocate the Shin-Hong per-call output bundle
#: (woof/core/shinhong.py launch_shinhong: one empty_like comprehension
#: for the nine 3-D fields, four cp.empty for the 2-D ones -- the five
#: allocation sites the physics allocation inventory prices).  Its own
#: constant rather than a second member of _YSU_OUTPUT_BUNDLE_SCHEMES
#: because the bundles differ: Shin-Hong returns 9 3-D + 4 2-D fields
#: against YSU's set, and keying both off one tuple would price the
#: wrong roster for whichever scheme joined second.
_SHINHONG_OUTPUT_BUNDLE_SCHEMES = (11,)

#: The launcher's exact per-call output roster (single-sourced by test
#: against woof/core/shinhong.py, which is CuPy-importing and therefore
#: not imported here -- the ysu constant-pair idiom above).
_SHINHONG_3D = ("du", "dv", "dtheta", "dqv", "dqc", "dqi",
                "exch_h", "tke", "el")
_SHINHONG_2D = ("hpbl", "kpbl", "wstar", "delta")


def shinhong_output_transient_shapes(
        cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Raw per-call Shin-Hong outputs before field-copy/coupling use.

    The :func:`ysu_output_transient_shapes` contract for scheme 11:
    these allocations exist on every _run_shinhong call and are released
    at the last consumer (the scheme has no positive-cadence retention
    -- ``last_ysu`` stays None -- so no persistent counterpart exists).
    ``kpbl`` is int32; every other field is float32, so one 4-byte
    itemsize covers the whole roster.
    """
    if int(cfg.bl_pbl_physics) not in _SHINHONG_OUTPUT_BUNDLE_SCHEMES:
        return {}
    m = (cfg.nz, cfg.ny, cfg.nx)
    s2 = (cfg.ny, cfg.nx)
    shapes = {f"shinhong_output/{name}": m for name in _SHINHONG_3D}
    shapes.update({f"shinhong_output/{name}": s2 for name in _SHINHONG_2D})
    return shapes


#: PBL schemes that allocate the MYJ per-call output bundle
#: (woof/core/myjpbl.py myj_pbl_step: one cp.empty comprehension for the
#: eight 3-D fields plus three cp.empty for the 2-D ones -- the four
#: allocation sites the physics allocation inventory prices).  Its own
#: constant for the reason the Shin-Hong pair above gives: the rosters
#: differ, and keying two schemes off one tuple prices the wrong one.
_MYJ_OUTPUT_BUNDLE_SCHEMES = (2,)

#: The launcher's exact per-call output roster, single-sourced against
#: woof/core/myjpbl.py -- which is CuPy-importing and therefore not
#: imported here -- by tests/test_myj_port.py::
#: test_preflights_myj_output_roster_matches_the_launchers, the
#: ysu/shinhong constant-pair idiom above.  ``tke``
#: is absent on purpose: MYJ's TKE column is CARRIED state that
#: initialize_physics allocates once, not a per-call transient, and it is
#: already priced as a driver field.
_MYJ_3D = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
           "rqiblten", "el_myj", "exch_h")
_MYJ_2D = ("pblh", "kpbl", "mixht")
#: What MYJURB (sf_urban_physics 2/3 under MYJ) allocates beyond that
#: roster; tests/test_myjurb_wrf471_parity.py holds it to the launcher.
_MYJURB_EXTRA_3D = ("exch_m",)


def myj_output_transient_shapes(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Raw per-call MYJ PBL outputs before coupling consumes them.

    The :func:`ysu_output_transient_shapes` contract for scheme 2: these
    allocations exist on every ``_run_myj_pbl`` call and are released at
    the last consumer.  ``kpbl`` is int32; every other field is float32,
    so one 4-byte itemsize covers the whole roster.

    The Eta surface layer (sf_sfclay_physics=2) has no counterpart here
    and needs none: woof/core/myjsfc.py allocates NOTHING per call -- it
    writes into the driver's own persistent surface fields -- which the
    physics allocation inventory records as an empty row.
    """
    if int(cfg.bl_pbl_physics) not in _MYJ_OUTPUT_BUNDLE_SCHEMES:
        return {}
    m = (cfg.nz, cfg.ny, cfg.nx)
    s2 = (cfg.ny, cfg.nx)
    shapes = {f"myj_output/{name}": m for name in _MYJ_3D}
    shapes.update({f"myj_output/{name}": s2 for name in _MYJ_2D})
    if int(getattr(cfg, "sf_urban_physics", 0) or 0) in (2, 3):
        # Under BEP/BEM the call goes to woof/core/myjurb.py instead, whose
        # bundle is the same roster plus EXCH_M (module_bl_myjurb.F
        # publishes both exchange coefficients): one more 3-D plane.
        shapes.update({f"myj_output/{name}": m for name in _MYJURB_EXTRA_3D})
    return shapes


def uwpbl_output_transient_shapes(
        cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Per-call UW PBL transients before coupling consumes them.

    The :func:`ysu_output_transient_shapes` contract for scheme 9, read
    from the launcher's own roster (woof.core.physics_inventory, which
    woof/core/uwpbl.py allocates from, so there is no second copy to
    drift): the seven mass-level tendencies, the three interface
    diagnostics, the four surface diagnostics and the int32 KPBL; plus the
    two mass-level inputs _run_uwpbl forms (the mass-level heights and
    phy_prep's density) with the one intermediate plane either expression
    holds while it is formed, and, where the microphysics carries the ice
    number, the mass-coupling factor and the coupled tendency.  Every word
    is four bytes.  The column workspace is priced separately
    (:func:`uwpbl_column_workspace_bytes`).
    """
    if int(cfg.bl_pbl_physics) != UW_PBL_SCHEME:
        return {}
    from woof.config import UW_PBL_ICE_NUMBER_SPECIES
    from woof.core.physics_inventory import (
        UWPBL_DIAGNOSTICS_FULL, UWPBL_MASS_OUTPUTS, UWPBL_SURFACE_OUTPUTS)
    m = (cfg.nz, cfg.ny, cfg.nx)
    fl = (cfg.nz + 1, cfg.ny, cfg.nx)
    s2 = (cfg.ny, cfg.nx)
    shapes = {f"uwpbl_output/{name}": m for name in UWPBL_MASS_OUTPUTS}
    shapes.update({f"uwpbl_output/{name}": fl
                   for name in UWPBL_DIAGNOSTICS_FULL})
    shapes.update({f"uwpbl_output/{name}": s2
                   for name in UWPBL_SURFACE_OUTPUTS + ("kpbl2d",)})
    shapes["uwpbl_input/z_mass"] = m
    shapes["uwpbl_input/rho"] = m
    # Each of the two is one CuPy expression with one intermediate plane
    # alive while it is formed.
    shapes["uwpbl_input/expression_temporary"] = m
    if int(cfg.mp_physics) in UW_PBL_ICE_NUMBER_SPECIES:
        shapes["uwpbl_output/ice_number_mass_factor"] = m
        shapes["uwpbl_output/ice_number_coupled"] = m
    return shapes


def noahmp_lsm_transient_shapes(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Noah-MP land-surface per-call device transients, both paths.

    Shapes here are ``(columns, bytes-per-column)`` with ``itemsize=1``,
    because the underlying allocations are dozens of named arrays whose
    itemization lives with their owners (the allocation-inventory rows in
    ``tests/test_physics_allocation_inventory.py``); what preflight owes is
    the bound, not the roster.

    * ``slab_chunk_transients`` is the forecast path's per-chunk cost:
      :func:`woof.core.noahmp_column_slab.evaluate_sflx_slab` over
      ``SLAB_COLUMN_CHUNK`` land columns.  The ceiling and the bound are the
      runtime's own constants, so price and bound cannot drift apart.
      Measured on the RTX 5090: 2,723 B of peak allocator demand per column
      for one 65,536-column chunk call, priced at 4,096.  Demand rather than
      CuPy pool growth, because growth reads only what the pool had to
      acquire from the driver and a warm pool serves the transient from
      blocks it already holds; see ``SLAB_TRANSIENT_BYTES_PER_COLUMN``.
    * ``slab_grid_transients`` is the same call's whole-grid residue: the
      device prologue intermediates (Q_ML through FICEOLD), the land index
      arrays and the pool's cross-chunk fragmentation, which scale with
      ``nx*ny`` and not with the chunk.  Measured at 360,000 columns: the
      whole-call pool growth of 311.8 MiB minus the chunk term's measured
      172.0 MiB is 407 B per grid column, priced at 512.
    * ``staged_leaf_batches`` is the per-column seam kept as the paired
      second implementation: four leaf batches at 620 B per staged column
      (52 + 76 + 296 + 196, derived in the allocation inventory), bounded by
      ``COLUMN_BATCH``.

    Only one path runs per call; all are priced because together they are
    still decided by bounds rather than by the nest, and the staged term is
    three orders of magnitude below the slab ones.  At d04 the two slab
    terms price 431.8 MiB against the measured 311.8.
    """
    if getattr(cfg, "sf_surface_physics", 0) != 4:
        return {}
    from woof.core.noahmp_runtime import (
        COLUMN_BATCH, SLAB_COLUMN_CHUNK, SLAB_GRID_TRANSIENT_BYTES_PER_COLUMN,
        SLAB_TRANSIENT_BYTES_PER_COLUMN)

    columns = int(cfg.ny) * int(cfg.nx)
    return {
        "noahmp_lsm/slab_chunk_transients":
            (min(SLAB_COLUMN_CHUNK, columns),
             SLAB_TRANSIENT_BYTES_PER_COLUMN),
        "noahmp_lsm/slab_grid_transients":
            (columns, SLAB_GRID_TRANSIENT_BYTES_PER_COLUMN),
        "noahmp_lsm/staged_leaf_batches":
            (min(COLUMN_BATCH, columns), _NOAHMP_STAGED_BYTES_PER_COLUMN),
    }


def noahmp_lsm_cache_shapes(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Noah-MP's slab caches, held on the card between calls.

    ``(columns, bytes-per-column)`` with ``itemsize=1``, as the transients
    above.  The surface layout and the per-identity parameter slabs are
    reused from one call to the next while the land identities stay the
    same, so they are resident for the domain's life rather than per call:
    ``nx*ny`` columns at :func:`woof.core.noahmp_runtime
    .slab_cache_bytes_per_column` (522 B), every column priced as land.
    """
    if getattr(cfg, "sf_surface_physics", 0) != 4:
        return {}
    from woof.core.noahmp_runtime import slab_cache_bytes_per_column

    return {"noahmp_lsm/slab_caches":
            (int(cfg.ny) * int(cfg.nx), slab_cache_bytes_per_column())}


#: 620 B of leaf-batch rows per staged column: bare_flux 52 + radiation 76 +
#: water 296 + sflx_pre 196, the derivation the allocation inventory records.
_NOAHMP_STAGED_BYTES_PER_COLUMN = 620


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainMemoryEstimate:
    """Itemized memory for one domain (architecture section E contract)."""

    grid_id: int
    items: tuple[MemoryItem, ...]

    def category_bytes(self, category: str) -> int:
        return sum(item.nbytes for item in self.items
                   if item.category == category)

    @property
    def resident_bytes(self) -> int:
        """Persistent tier-1 residency, including per-grid diagnostics.

        The SASE closure's own working set is a per-step transient like
        the radiation columns, so it is excluded here and counted below.
        """
        return sum(item.nbytes for item in self.items
                   if item.category not in ("transient", "sase"))

    @property
    def held_exact_bytes(self) -> int:
        """Resident bytes allocated once at their priced size and held for
        the run (:attr:`MemoryItem.held_exact`), which the measured pool
        margin does not multiply (A163)."""
        return sum(item.nbytes for item in self.items if item.held_exact)

    @property
    def transient_bytes(self) -> int:
        """Per-step transients (radiation columns + physics prep/YSU outputs)
        coexisting with the chunk workspace; freed between steps."""
        return sum(item.nbytes for item in self.items
                   if item.category in ("transient", "sase"))

    @property
    def arena_scratch_bytes(self) -> int:
        """This domain's requests admitted by the lifetime audit.

        The value remains part of :attr:`resident_bytes` so a bare/single-
        domain estimate keeps the historical per-state accounting. The
        experiment estimate replaces the multi-domain sum with one per-slot
        maximum.
        """
        return sum(item.nbytes for item in self.items
                   if item.category in ("scratch", "lbc", "nest")
                   and scratch_slot_uses_arena(item.name))

    @property
    def rebuilt_state_bytes(self) -> int:
        """This domain's restart-REBUILT state-array requests."""
        rebuilt = shared_dycore_state_symbols()
        return sum(item.nbytes for item in self.items
                   if item.category == "state" and item.name in rebuilt)


def estimate_domain(dc: DomainConfig, *, spec_bdy_width: int | None = None,
                    cfl_recording: bool | None = None,
                    cam_ozone: bool = False,
                    follower_slots: tuple[str, ...] = (),
                    parent: DomainConfig | None = None,
                    n_lbc_intervals: int = 0,
                    lateral_boundaries=None,
                    p_top: float = 5000.0,
                    column_chunk: int = DEFAULT_COLUMN_CHUNK,
                    boundary_species=(),
                    tile_buffer: bool = False,
                    urban_columns: int | None = None,
                    lake_columns: int | None = None,
                    ) -> DomainMemoryEstimate:
    """Itemized :class:`DomainMemoryEstimate` for one domain.

    ``spec_bdy_width`` (the experiment's) sizes child ``nest_*`` tables;
    ``n_lbc_intervals`` sizes the root's eager forcing tables.  Both
    default from the domain's own RunConfig / to zero intervals for a
    bare single-domain estimate.  ``tile_buffer`` prices ``dc`` as one
    streamed tile buffer, whose MYNN workspace is its own
    (:func:`mynn_pbl_column_chunk`).  ``urban_columns`` is the domain's
    urban column count read off its prepared land cover
    (:func:`woof.core.urban_state.prepared_urban_columns`); ``None``, the
    configuration door, prices BEP+BEM's column workspace at every
    column urban (A176).
    """
    run = dc.run
    width = run.spec_bdy_width if spec_bdy_width is None else spec_bdy_width
    items: list[MemoryItem] = []
    items += _items("state", state_array_shapes(run))
    held = urban_held_item_names(run)
    items += tuple(
        (HeldMemoryItem if name in held else MemoryItem)(
            name, "physics", tuple(shape), 4)
        for name, shape in physics_array_shapes(
            run, cam_ozone=cam_ozone, urban_columns=urban_columns,
            lake_columns=lake_columns).items())
    from woof.core.cfl_inventory import (
        WRF_CFL_SHAPE, wrf_cfl_recording_requested)
    if wrf_cfl_recording_requested(run, adaptive=cfl_recording):
        items.append(MemoryItem("wrf_cfl_ring", "diagnostic",
                                WRF_CFL_SHAPE, 4, "uint32"))
    registry = scratch_slot_registry(
        run, n_lbc_intervals=(n_lbc_intervals if run.specified else 0),
        tile_buffer=tile_buffer)
    registry.update({slot: (int(run.ny), int(run.nx)) for slot in follower_slots})
    if boundary_species and run.specified and n_lbc_intervals > 0:
        # The analysed hydrometeors the source publishes ride the root's
        # tables (woof.boundary_fields); their spec+relax tendency is
        # recomputed per stage, so they add tables and no held slot.
        registry["lbc_forcing_tables"] = (
            n_lbc_intervals * lbc_interval_values(
                run, boundary_species=boundary_species),)
    if lateral_boundaries is not None:
        if not run.specified:
            raise ValueError("external boundary storage requires a specified domain")
        from woof.ingest.lateral_bc import boundary_storage_shapes
        registry.update(boundary_storage_shapes(lateral_boundaries))
    # The (d) LBC residents live in the scratch pool but report under
    # their own itemization category (architecture section E contract).
    items += _items("lbc", {slot: shape for slot, shape in registry.items()
                            if slot.startswith("lbc_")})
    wide_fsbm={"fsbm_pointers":"uint64","fsbm_tally":"float64"}
    items += tuple(MemoryItem(slot,"scratch",tuple(shape),
                    8 if slot in wide_fsbm else 4,
                    wide_fsbm.get(slot,"float32"))
              for slot,shape in registry.items() if not slot.startswith("lbc_"))
    if dc.parent_id != 0:
        shapes = nest_slot_shapes(dc, width, parent)
        items += _nest_items(shapes, nest_slot_dtypes(dc, width, parent))
    items += _items("transient", atmosphere_transient_shapes(run, cam_ozone=cam_ozone))
    items += _items("transient", ysu_output_transient_shapes(run))
    items += _items("transient", mynn_scalar_transient_shapes(run))
    items += mynn_mixscalars_memory_items(run, tile_buffer=tile_buffer)
    from woof.core.physics_inventory import terrain_drag_transient_shapes
    items += _items("transient", terrain_drag_transient_shapes(run))
    items += _items("transient", shinhong_output_transient_shapes(run))
    items += _items("transient", myj_output_transient_shapes(run))
    items += _items("transient", uwpbl_output_transient_shapes(run))
    items += _items("transient", noahmp_lsm_transient_shapes(run),
                    itemsize=1)
    items += _items("physics", noahmp_lsm_cache_shapes(run), itemsize=1)
    if int(run.sf_surface_physics) == 3:
        # The RUC fused driver and generated soil/snow/sea-ice solver keep
        # their own slabs outside DomainState.scratch. Omitting them hides
        # more than a GiB on a wide rank. Price their GPU-censused shape
        # inventory, including per-allocation pool rounding.
        from woof.core.ruc_memory import ruc_runtime_memory_bytes
        for name, size in ruc_runtime_memory_bytes(
                int(run.nx) * int(run.ny), soil_layer_count(run)).items():
            category = "transient" if name == "sfctmp_outputs" else "physics"
            items.append(MemoryItem("ruc/" + name, category, (size,), 1))
    items += tuple(MemoryItem(name, "transient", shape, size)
                   for name, (shape, size)
                   in rrtmgp_column_shapes(
                       run, p_top, column_chunk=column_chunk).items())
    items += tuple(MemoryItem(name, "transient", shape, size)
                   for name, (shape, size) in classic_rrtm_column_shapes(
                       run, p_top, column_chunk=column_chunk).items())
    items += _items("transient", dudhia_column_shapes(run))
    if getattr(run, "chem_sets", ""):
        from woof.core.chem_radiation import memory_shapes
        from woof.core.device_inventory import chem_process_memory_shapes
        process_held, process_work = chem_process_memory_shapes(run)
        items += _items("physics", process_held)
        items += _items("transient", process_work)
        optics_held, optics_transient = memory_shapes(run, p_top, column_chunk)
        items += _items("physics", optics_held)
        items += _items("transient", optics_transient)
    # The SASE closure's step working set gets its OWN category rather
    # than joining "transient": on a wide domain its dynamic-solve peak
    # is the single largest transient in the run, and a user reading a
    # preflight needs to see that it is the closure asking, not the
    # radiation columns.  It is still counted as a step transient, never
    # as residency.
    items += tuple(MemoryItem(name, "sase", shape, size,
                              "float64" if size == 8 else "float32")
                   for name, (shape, size)
                   in sase_workspace_shapes(run).items())
    return DomainMemoryEstimate(dc.grid_id, tuple(items))


@dataclass(frozen=True)
class ExperimentMemoryEstimate:
    """The experiment-level three-tier estimate.

    ``alloc_estimate_bytes`` is THE enforced number of the N0/N5/N6
    chain: ``ALLOCATOR_HEADROOM x (resident + shared chunk workspace +
    max-over-domains step transients)``. Multi-domain resident scratch uses
    the lifetime-audited per-slot maximum; a single domain keeps the original
    per-state sum. ``held/footprint`` projections add the calibrated
    tier-2/tier-3 terms for reserve-policy visibility. ``workspace_bytes`` is
    not a pool-reuse allowance: Task 14 constructs one byte allocation of
    exactly this size and every domain adapter consumes its phase views; the
    builder refuses any runtime/ledger byte drift.
    """

    domains: tuple[DomainMemoryEstimate, ...]
    k_tables_bytes: int
    workspace_bytes: int
    scratch_arena_bytes: int
    uses_shared_scratch_arena: bool
    dycore_state_workspace_bytes: int
    uses_shared_dycore_state_workspace: bool
    column_chunk: int
    #: The measured forecast margin (A163), not the plan's 1.15.
    headroom: float = FORECAST_POOL_HEADROOM
    #: A byte bound on the margin ``headroom`` prices (A13): set by the
    #: ranked road to :data:`RANK_POOL_RESIDUE_BOUND_BYTES`, ``None``
    #: (the ratio alone) on every other road.
    pool_residue_cap_bytes: int | None = None
    retention_residual_bytes: int = field(default=0)
    device_overhead_bytes: int = field(
        default_factory=lambda: platform_projection_constants()[1])
    #: CUDA context + launch-time local-memory backing store for the
    #: kernel set THIS configuration launches, on the device profile it
    #: was priced against.  The affine envelope's intercept.
    non_pool_device_bytes: int = 0
    #: Which envelope family priced it (``linux``/``windows``/...).
    envelope_family: str = "linux"
    #: Does this configuration run the LEGACY RRTMG engines?  Reported
    #: (``envelope_legacy_radiation`` in ``woof check --json``) and passed
    #: through to :func:`machine_peak_envelope_bytes`, which no longer
    #: reads it: the pool-slack term it used to decide
    #: (:data:`POOL_SLACK_FRACTION`) is retired from every gate (A163),
    #: because the legacy lane's call-peak workspace is itemized and its
    #: measured peaks sit inside :data:`FORECAST_POOL_HEADROOM`.
    uses_legacy_radiation: bool = True
    # Mixed variants keep the modern workspace resident during legacy calls.
    # Pure-legacy estimates retain their historical workspace-envelope field.
    legacy_call_peak_by_domain: tuple[int, ...] = ()
    # Carry the same resolved device into the independent prepared tile
    # inventory. It is internal pricing context, never an extra GPU query.
    local_memory_profile: DeviceLocalMemoryProfile | None = None
    retained_forcing_intervals: int = 0
    # Included in non_pool_device_bytes, but bounded by grid column counts.
    # Keep it separate for fixed-floor diagnostics, not a second admission.
    column_workspace_bytes: int = 0
    # Immutable microphysics coefficients uploaded once per device, not
    # once per domain. Separate from radiation's shared k-tables so a
    # legacy radiation workspace is never mistaken for a fixed floor.
    physics_tables_bytes: int = 0
    # Table-selected hydro is priced outside the static RunConfig physics
    # roster. Missing owner hooks are explicit gaps in this same envelope.
    hydro_memory: dict | None = None

    @property
    def fixed_envelope_bytes(self) -> int:
        """A grid-independent LOWER BOUND evaluated by the same envelope.

        The historical intercept also contains grid-sized column workspaces;
        it cannot prove that resizing is futile. Exclude those, and retain
        the context/kernel floor plus shared RRTMGP tables/chunk workspace.
        Pure-legacy call workspaces can scale with columns and are excluded.
        No forecast or admission number changes when this diagnostic is read.
        """
        fixed_pool = (self.k_tables_bytes + self.physics_tables_bytes
                      + (self.workspace_bytes if self.k_tables_bytes else 0))
        return machine_peak_envelope_bytes(
            alloc_estimate_bytes=math.ceil(self.headroom * fixed_pool),
            non_pool_bytes=self.non_pool_device_bytes - self.column_workspace_bytes,
            domains=len(self.domains), family=self.envelope_family,
            legacy_radiation=self.uses_legacy_radiation)

    @property
    def resident_bytes(self) -> int:
        per_domain = sum(d.resident_bytes for d in self.domains)
        if self.uses_shared_scratch_arena:
            per_domain -= self.scratch_arena_request_bytes
            per_domain += self.scratch_arena_bytes
        if self.uses_shared_dycore_state_workspace:
            per_domain -= self.dycore_state_request_bytes
            per_domain += self.dycore_state_workspace_bytes
        return (per_domain + self.k_tables_bytes + self.physics_tables_bytes
                + int((self.hydro_memory or {}).get("resident_bytes", 0)))

    @property
    def dycore_state_request_bytes(self) -> int:
        """Unshared sum of restart-REBUILT requests across all domains."""
        return sum(d.rebuilt_state_bytes for d in self.domains)

    @property
    def dycore_state_saved_bytes(self) -> int:
        """Resident bytes removed by per-symbol maximum sharing."""
        if not self.uses_shared_dycore_state_workspace:
            return 0
        return (self.dycore_state_request_bytes
                - self.dycore_state_workspace_bytes)

    @property
    def scratch_arena_request_bytes(self) -> int:
        """Unshared sum of arena-admitted requests across all domains."""
        return sum(d.arena_scratch_bytes for d in self.domains)

    @property
    def scratch_arena_saved_bytes(self) -> int:
        """Resident bytes removed by max-per-slot sharing."""
        if not self.uses_shared_scratch_arena:
            return 0
        return self.scratch_arena_request_bytes - self.scratch_arena_bytes

    @property
    def transient_peak_bytes(self) -> int:
        """Domains step strictly sequentially (architecture section E
        chunk-policy adjudication); one domain's step transients live at
        a time, so the peak takes the max, not the sum."""
        return int((self.hydro_memory or {}).get("transient_bytes", 0)) + max((d.transient_bytes + (self.legacy_call_peak_by_domain[index]
                    if self.legacy_call_peak_by_domain else 0)
                    for index, d in enumerate(self.domains)), default=0)

    @property
    def subtotal_bytes(self) -> int:
        return (self.resident_bytes + self.workspace_bytes
                + self.transient_peak_bytes)

    @property
    def held_exact_bytes(self) -> int:
        """The subtotal's arrays held at exactly their priced size (the
        urban arrays), summed over the domains."""
        return sum(d.held_exact_bytes for d in self.domains)

    @property
    def alloc_estimate_bytes(self) -> int:
        return forecast_pool_estimate_bytes(
            self.subtotal_bytes, held_exact_bytes=self.held_exact_bytes,
            headroom=self.headroom,
            residue_cap_bytes=self.pool_residue_cap_bytes)

    @property
    def pool_residue_cap_active(self) -> bool:
        """Is the pool margin this estimate carries the byte bound rather
        than the ratio (A13)?  False whenever no cap was given or the
        ratio's margin is already under it."""
        cap = self.pool_residue_cap_bytes
        if cap is None:
            return False
        turnover = self.subtotal_bytes - max(
            0, min(self.held_exact_bytes, self.subtotal_bytes))
        return math.ceil(self.headroom * turnover) - turnover > int(cap)

    @property
    def held_projection_bytes(self) -> int:
        return self.alloc_estimate_bytes + self.retention_residual_bytes

    @property
    def footprint_projection_bytes(self) -> int:
        return self.held_projection_bytes + self.device_overhead_bytes

    @property
    def envelope_intercept_bytes(self) -> int:
        """Everything in the envelope that does not scale with the grid.

        The itemized non-pool residency, and nothing else.  The 5090
        zero-step probe constant (``device_overhead_bytes``) and the
        pool-retention constant stay in the TIER 2/3 projection display
        they were calibrated for; charging them here on top of the
        itemized non-pool term is how a 10 GiB card came to carry
        4.12 GiB of another machine's accounting (the 3080 walk).
        """
        return self.non_pool_device_bytes + int((self.hydro_memory or {}).get("non_pool_bytes", 0))

    @property
    def peak_envelope_bytes(self) -> int:
        """The affine machine-peak envelope for this forecast."""
        return machine_peak_envelope_bytes(
            alloc_estimate_bytes=self.alloc_estimate_bytes,
            non_pool_bytes=self.envelope_intercept_bytes,
            domains=len(self.domains),
            family=self.envelope_family,
            legacy_radiation=self.uses_legacy_radiation,
            held_exact_bytes=self.held_exact_bytes)

    @property
    def envelope_basis(self) -> str:
        """The evidence behind this configuration's envelope terms.

        One basis for every radiation lane since A163: the margin is
        measured over a battery that carries both lanes, and the
        intercept and residue rest on the affine campaign.
        """
        basis = f"{ENVELOPE_A163_BASIS}; {ENVELOPE_AFFINE_BASIS}"
        if self.hydro_memory is not None:
            gaps = self.hydro_memory["gaps"]
            basis += ("; hydro allocation inventory PARTIAL: " + "; ".join(gaps)
                      if gaps else "; hydro owner allocation contracts (whole-run qualification pending)")
        return basis

    def peak_envelope_terms(self) -> str:
        """The envelope's arithmetic, exactly as it was evaluated.

        Printed by both `woof check` and the wizard, from one place, so
        the two can never show a sum whose parts do not add up to it --
        which is what happens the moment a second branch exists and only
        one of them is described.
        """
        nests = max(0, len(self.domains) - 1)
        held = self.held_exact_bytes
        nest_term = (f" + {ENVELOPE_PER_NEST_FRACTION:.0%} of the estimate"
                     f"{' less the held arrays' if held else ''} "
                     f"x {nests} nest(s)" if nests else "")
        margin = (f"x {self.headroom:.2f}: measured peak/itemized "
                  f"{self.headroom - FORECAST_PEAK_RATIO_SAFETY:.2f} + safety "
                  f"{FORECAST_PEAK_RATIO_SAFETY:.2f}")
        if self.pool_residue_cap_active:
            margin = (f"+ {int(self.pool_residue_cap_bytes) / GIB:.2f}: the "
                      f"measured pool residue bound (A13), under the "
                      f"{margin}")
        pool = (f"subtotal {self.subtotal_bytes / GIB:.2f} {margin}"
                if not held else
                f"subtotal less held arrays "
                f"{(self.subtotal_bytes - held) / GIB:.2f} {margin}, + "
                f"urban arrays held at their allocated size "
                f"{held / GIB:.2f}")
        return (f"estimate {self.alloc_estimate_bytes / GIB:.2f} ({pool}) + "
                f"non-pool {self.envelope_intercept_bytes / GIB:.2f} (CUDA "
                f"context + local-memory backing store + GF, KF and "
                f"YSU column workspaces) + "
                f"{ENVELOPE_UNMODELLED_BYTES / GIB:.2f} unmodelled"
                f"{nest_term} = "
                f"{self.peak_envelope_bytes / GIB:.2f} GiB")


# ---------------------------------------------------------------------------
# Preprocessing (ingest) phase
#
# Everything above prices the FORECAST.  For a long time that was the only
# phase anybody priced, and on a measured 414x330x49 CONUS 12 km case the
# forecast peaked at 7.10 GiB while preprocessing the very same case peaked
# at 14.94 GiB -- so `woof check` reported an envelope for a phase that
# was not the binding one, and a domain sized to a 12 GB card downloaded
# 81 GFS files and then died in preprocessing at 15.82 GB.  Ingest is now
# streamed (woof/ingest/lateral_bc.py:StateBoundaryFrames), and the model
# below is what it actually holds.
# ---------------------------------------------------------------------------

#: Forcing times a streaming ingest holds on the device at once: the ONE
#: currently being built.  Every other time contributes its perimeter
#: frames -- host memory, O(perimeter) -- and is released before the next
#: one is interpolated.
#:
#: THIS WAS 2, AND THE SECOND ONE WAS THE START TIME.  Nothing reads the
#: start time until the boundaries are complete (the prepared cache, the
#: wrfinput export and the surface analysis are all written from it after
#: the loop), but a prepare loop that walked the times in order built it
#: FIRST and therefore held it while every later time was built
#: underneath.  The domain-tree (hierarchy) preparations and the
#: single-domain ERA5 preparation with a water-temperature overlay build it
#: LAST (woof/ingest/lateral_bc.py:start_last_forcing_order) and retain
#: nothing else, which is a pure reordering: the perimeter frames are
#: accumulated against their positions and the intervals come out
#: byte-identical.  At 800x800x49, mp=10, three GFS times, that is 14.67
#: GiB of device residency dropping to 7.66 and a peak envelope of 23.92
#: GiB dropping to 15.86 -- the whole reason such a domain can be
#: prepared on a 16 GiB card at all.
#:
#: Every other single domain is now prepared START FIRST again, and still
#: holds one: the start time is written into the prepared head and
#: released before the next time is built (woof/ingest/boundary_stream.py,
#: chained preparation), so what it held under every later build is gone.
#: Its forecast may run beside the producer; the producer holds one
#: forcing time on its own device while it does, and
#: ``boundary_stream.chained_admission`` admits the pair only when the
#: forecast's peak envelope plus that one time fits the card, and
#: ``boundary_stream.host_admission`` only when host RAM holds the forecast
#: process, its head and its whole boundary series beside the producer.
#:
#: SCOPE, because this number is a gate input and an optimistic gate is
#: the failure mode this section exists to prevent: it describes the
#: prepared-cache adapters `woof/gfs_direct.py`, `woof/era5_direct.py`
#: and `woof/mapped_direct.py`, which are what `woof go` and the domain
#: wizard price.  `woof/runtime.py:prepare_real_case` -- the verify-case
#: preparer, off those routes -- has not been reordered and still holds
#: two, so for a run prepared through THAT path this estimate is
#: optimistic by exactly one `per_time_bytes`.
INGEST_RESIDENT_FORCING_TIMES = 1

#: Pressure levels each forcing product decodes onto the target grid.
#: GFS: the certified 21-level ladder the Rust bridge gates on
#: (woof/gfs_direct.py:_validate_ladder).  A case whose p_top sits above
#: 100 hPa is fetched with the levels its top needs, and
#: :func:`source_analysis_levels` counts those for it.
#: ERA5: the 37 standard pressure levels of the reanalysis product.
#:
#: This map IS the priced-source inventory: a product absent from it is
#: reported NOT PRICED rather than given a plausible number, and every
#: other table in this section is keyed by exactly these names.  The
#: native-hybrid-level ingest lane is deliberately not among them --
#: nothing here has measured it.  A wrong ingest estimate is the defect;
#: an absent one that says so is not.
SOURCE_ANALYSIS_LEVELS = {"era5": 37, "gfs": 21}

#: Cadence each PRICED product is fetched at.  The ingest phase's time
#: COUNT comes from this, not from the forecast's LBC interval: a GFS
#: chain fetched 3-hourly has nine forcing times over 24 h where the
#: 6-hourly default would claim five.  Keys track SOURCE_ANALYSIS_LEVELS
#: exactly; the wizard keeps its own, wider table for the forecast side,
#: which prices products this one does not.
INGEST_FORCING_CADENCE_SECONDS = {"era5": 21600.0, "gfs": 10800.0}

#: Fields each product carries on those levels: T, RH, GHT on mass points
#: plus U and V on their own staggers.
SOURCE_ANALYSIS_MASS_LEVEL_FIELDS = 3
SOURCE_ANALYSIS_WIND_LEVEL_FIELDS = 1  # each of U and V, on its own stagger

#: Single-level fields interpolated alongside them (surface state, skin,
#: snow, ice, land mask and the four soil moisture/temperature layers).
#: ERA5 includes the three explicit lake-state fields in the current CDS
#: request. These nominal counts apply only before an actual decoded catalog
#: supplies its exact inventory; older/smaller inputs retain their own price.
SOURCE_ANALYSIS_SURFACE_FIELDS = {"era5": 23, "gfs": 19}

#: HOST bytes one decoded SOURCE field point occupies.  The GRIB1 bridge
#: dump is float64 (`woof/ingest/grib.py:550`, ``np.fromfile(...,
#: "<f8")``) and :class:`woof.ingest.grib.Era5Snapshot` REFUSES anything
#: narrower on both its axes and its fields (`grib.py:220-221`,
#: `:237-238`), so nothing on this road is ever taken to float32.  This is
#: the source grid, not the target: horizontal interpolation happens
#: after the decode, so the bytes below are charged on the file's mesh.
INGEST_HOST_DECODE_BYTES_PER_POINT = 8

#: How many times the decode of ONE forcing time is held in host memory
#: SIMULTANEOUSLY, which is the number an out-of-memory kill turns on.
#: The window is preparation: both copies exist from the decode until
#: ``woof.ingest.grib.clear_forcing_caches`` runs, which is the moment
#: the root's case is prepared, and one of them -- the input catalog's
#: own frozen tuple, which a nest re-ingests from -- outlives that for
#: the run.  A command that only decodes and never prepares (``woof
#: check`` itself) holds both until it exits.
#:
#: KEYED BY SOURCE, and it prices ONE of them, because this is a property
#: of a DECODER and not of a product.  ERA5 is the native-GRIB1 route
#: through :mod:`woof.ingest.grib`, which is the road that retains; GFS
#: and HRRR feed the rw-wps/gpuwm-wrf-init front door, whose host
#: behaviour nothing here has measured.  A source absent from this map is
#: reported NOT PRICED rather than given this road's number -- the same
#: convention :data:`SOURCE_ANALYSIS_LEVELS` states one table up.
#:
#: TWO, and both are named.  (1) The decoder's own arrays, reached through
#: ``@lru_cache(maxsize=8) _decode_era5_forcing_partials_resolved``
#: (`woof/ingest/grib.py:934`).  (2) The frozen snapshot's, because
#: :meth:`Era5Snapshot.__post_init__` COPIES every axis and every field
#: (`grib.py:226`, `:251`) and the result is reached through a second
#: ``@lru_cache(maxsize=8)``, ``_decode_era5_gribs_resolved``
#: (`grib.py:958`).  The two byte sets are disjoint and neither cache
#: evicts on size: ``maxsize`` counts ENTRIES, and one entry is a whole
#: forcing window.
#:
#: A FLOOR, AND THE REPORT SAYS SO.  Two terms this deliberately does not
#: claim, because both are conditional on facts the estimator cannot see
#: without performing the decode it exists to price:
#:   * the flat bridge buffer.  ``_load_bridge_partials`` reads the whole
#:     dump into one array (`grib.py:550`), stacks the pressure levels
#:     into new cubes (`:604-607`) but keeps every SURFACE field as a VIEW
#:     into that buffer (`:565`, `:573`).  A live view pins the whole
#:     allocation, so a file carrying both pressure and surface records
#:     retains one more copy of itself; separate CDS pressure/single-level
#:     downloads -- the split `build_input_catalog` tells users to make --
#:     free the pressure file's buffer after the stack and do not.
#:   * a second merged copy.  A run reaches the module twice for one
#:     product, under catalog discovery and again under the catalog's time
#:     selection, and those are different cache keys.
#: Understating is the correct direction here: this figure gates a
#: REFUSAL, and a refusal must never fire on a run that would have
#: completed.
INGEST_HOST_RETAINED_COPIES = {"era5": 2}


def source_analysis_levels(source: str, *,
                           p_top_pa: float | None = None) -> int:
    """Pressure levels one forcing time of SOURCE decodes, for a model top.

    :data:`SOURCE_ANALYSIS_LEVELS` is the ladder a run gets when its
    model top sits inside it.  A run whose ``[shared].p_top`` sits above a
    source's certified ladder is fetched with the levels that top needs
    (``woof fetch --p-top-pa``, which ``woof go`` and ``run-plan`` pass
    on their own), so the count follows the same registry answer and the
    same ladder function the fetch uses: a default 50 hPa GFS run decodes
    23 levels, not 21.  A top above everything the product publishes is
    refused by the fetch before any download; it is counted here as every
    published level, the most that fetch could ever take.
    """
    key = str(source).strip().lower()
    try:
        levels = SOURCE_ANALYSIS_LEVELS[key]
    except KeyError:
        raise ValueError(
            f"no forcing-analysis level inventory for source {source!r}; "
            f"known: {sorted(SOURCE_ANALYSIS_LEVELS)}") from None
    from woof.source_adapters import fetch_model_top_pa

    top = fetch_model_top_pa(key, p_top_pa)
    if top is None:
        return levels
    from woof.fetch import container_subset_levels
    from tools.download_gfs_native_subset import CERTIFIED_AVAILABLE_LEVELS_HPA

    try:
        fetched = len(container_subset_levels(key, top_pressure_pa=top))
    except ValueError:
        fetched = len(CERTIFIED_AVAILABLE_LEVELS_HPA)
    return max(levels, fetched)


def source_analysis_fields_per_time(source: str, *,
                                    p_top_pa: float | None = None) -> int:
    """Two-dimensional SOURCE fields one forcing time decodes to.

    The same inventory :func:`ingest_analysis_shapes` prices on the device,
    counted as flat 2-D fields on the file's own mesh: every pressure level
    of every level field, plus the single-level fields.  ERA5: 37 x (3 mass
    + U + V) + 19 = 204.  ``p_top_pa`` is the run's model top, which sets
    how many levels its fetch takes (:func:`source_analysis_levels`).

    Keyed by exactly the products :data:`SOURCE_ANALYSIS_LEVELS` prices, so
    a source this module reports NOT PRICED on the device side cannot
    acquire a host figure here by a different route.

    NOMINAL, NOT MEASURED.  Nothing obliges a config to carry these
    levels or these surface fields, so a caller holding a decoded catalog
    passes its own count instead (:func:`ingest_host_geometry`); this
    stands in only when none is in hand, and a figure built on it is not
    one to refuse a run over.
    """
    key = str(source).strip().lower()
    try:
        levels = source_analysis_levels(key, p_top_pa=p_top_pa)
        surface = SOURCE_ANALYSIS_SURFACE_FIELDS[key]
    except KeyError:
        raise ValueError(
            f"no forcing-analysis level inventory for source {source!r}; "
            f"known: {sorted(SOURCE_ANALYSIS_LEVELS)}") from None
    per_level = (SOURCE_ANALYSIS_MASS_LEVEL_FIELDS
                 + 2 * SOURCE_ANALYSIS_WIND_LEVEL_FIELDS)
    return levels * per_level + surface


#: The setup a CUDA preparation builds beside each forcing time's state is
#: ITEMIZED, not a fraction of it: the vertical-interpolation plans and
#: outputs, and the temporaries around them, priced by
#: :mod:`woof.ingest.preparation_price` from the analysis inventory and
#: the model's levels, with the pool headroom measured for this phase
#: (:data:`woof.ingest.preparation_price.PREPARATION_POOL_HEADROOM`).
#: The 0.65-of-one-forcing-time transient it replaces priced the 3 km
#: CONUS GFS preparation's setup at 13.4 GiB where 5.2 GiB was live at the
#: failure, and did not follow the source's level count at all (A65).
#:
#: What that measurement was, printed beside the number it produces.
INGEST_PEAK_ENVELOPE_BASIS = (
    "itemized analysis, model state and vertical setup, x1.10 setup "
    "residual and x1.20 pool headroom, measured on four CUDA preparations "
    "(1792x1024x55 to a 3:1 nest, H100, 2026-09-28), + CUDA context")

#: What the CPU preparation holds beyond
#: :attr:`IngestMemoryEstimate.host_preprocess_floor_bytes`: the
#: interpolation and initialization scratch built at the SOURCE's levels
#: on the target grid, the statics and the interpreter, charged as
#: multiples of the root's one-time analysis, because that is what it
#: scales with. It prices a single-domain preparation; a nested tree is
#: priced by CPU_PREPARATION_TREE_ACTIVE_ANALYSIS_MULTIPLE and
#: CPU_PREPARATION_RETAINED_NEST_ANALYSIS_MULTIPLE below. At one grid
#: it did not move between 49 and 76
#: model levels, which is why it is not a fraction of the state.
CPU_PREPARATION_ANALYSIS_MULTIPLE = 10.0

#: Child input mapping can overlap, but state initialization follows parent
#: order. Price the largest domain's active initialization/export work
#: separately from the calibrated retained child inputs/results.
#: CPU initialization diagnoses and adjusts each child against its parent;
#: export can peak on the root when it is larger than its children. The
#: whole tree's states remain resident and are already in the floor.
#: Measured process-tree RSS: MEASURED_NESTED_CPU_PREPARATIONS in
#: tests/test_cpu_preparation_host_ram.py. Summing the root's 10x scratch
#: with every child's scratch overestimated equal-grid trees by 18-24%.
CPU_PREPARATION_TREE_ACTIVE_ANALYSIS_MULTIPLE = 13.0
#: The retained term charges every child except the widest one, which the
#: measurements place inside the active envelope even when the root is the
#: active domain.
CPU_PREPARATION_RETAINED_NEST_ANALYSIS_MULTIPLE = 9.0

#: And the part that grows with the forcing window (the decoded source
#: times, all held until the boundaries are built), per forcing interval,
#: as a multiple of the root's one-time analysis.
CPU_PREPARATION_ANALYSIS_MULTIPLE_PER_INTERVAL = 0.3

#: Where the CPU preparation's two host figures were measured against the
#: real thing, printed beside them.  Peak resident memory of the whole
#: preparation process tree, on the default install (Rust static fields
#: and NetCDF writer); ``tests/test_cpu_preparation_host_ram.py`` carries
#: every case.  The multiples put the estimate at or above the highest
#: peak seen for each case, which moved by up to 6% between repeated runs
#: of one configuration. Every case ran eight preparation threads. Wider
#: pools add 32 MiB per extra worker for allocator and column scratch,
#: covering the saved 16/32/64-worker process-tree peaks in that test file.
CPU_PREPARATION_PEAK_BASIS = (
    "measured on real GFS preparations at 3 km, 474x380 to 902x720, 49 to "
    "96 levels, 3 to 25 forcing times, with eight worker threads, plus "
    "32 MiB allocator and column scratch per worker above eight: "
    "the floor came to 0.74 to 0.89 of "
    "the peak and the estimate to 1.01 to 1.11; for nested trees of 2 and "
    "3 domains the floor came to 0.68 to 0.94 of the measured peak and "
    "the estimate to 1.01 to 1.12 of it")


def ingest_analysis_shapes(cfg: RunConfig, *, source: str,
                           actual_shapes: Mapping[str, tuple[int, ...]] | None = None,
                           p_top_pa: float | None = None,
                           ) -> dict[str, tuple[int, ...]]:
    """One forcing time, horizontally interpolated onto the target grid.

    The source-level fields land on the model's OWN horizontal grid --
    that is what horizontal interpolation is -- so they are sized by the
    target ny/nx and the SOURCE's level count, not the model's nz.  That
    count follows the run's model top ``p_top_pa`` (``p_top`` is not a
    :class:`RunConfig` field, so the experiment's is passed in), because
    the fetch takes the levels that top needs.
    """
    if actual_shapes is not None:
        if not isinstance(actual_shapes, Mapping) or not actual_shapes:
            raise ValueError("actual analysis shapes must be a nonempty field inventory")
        result = {}
        for name, shape in actual_shapes.items():
            if (not isinstance(name, str) or not name or not isinstance(shape, (tuple, list))
                    or len(shape) not in (2, 3) or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape)):
                raise ValueError("actual analysis inventory requires named positive 2-D/3-D integer shapes")
            if tuple(shape[-2:]) not in ((cfg.ny,cfg.nx),(cfg.ny,cfg.nx+1),(cfg.ny+1,cfg.nx)):
                raise ValueError(f"actual analysis field {name} differs from the target C grid")
            result[name] = tuple(shape)
        return result
    key = str(source).strip().lower()
    try:
        levels = source_analysis_levels(key, p_top_pa=p_top_pa)
        surface = SOURCE_ANALYSIS_SURFACE_FIELDS[key]
    except KeyError:
        raise ValueError(
            f"no forcing-analysis level inventory for source {source!r}; "
            f"known: {sorted(SOURCE_ANALYSIS_LEVELS)}") from None
    ny, nx = int(cfg.ny), int(cfg.nx)
    shapes: dict[str, tuple[int, ...]] = {}
    for index in range(SOURCE_ANALYSIS_MASS_LEVEL_FIELDS):
        shapes[f"analysis_mass_level_{index}"] = (levels, ny, nx)
    for index in range(SOURCE_ANALYSIS_WIND_LEVEL_FIELDS):
        shapes[f"analysis_u_level_{index}"] = (levels, ny, nx + 1)
        shapes[f"analysis_v_level_{index}"] = (levels, ny + 1, nx)
    for index in range(surface):
        shapes[f"analysis_surface_{index}"] = (ny, nx)
    return shapes


@dataclass(frozen=True)
class IngestMemoryEstimate:
    """Itemized device memory for the preprocessing phase of one TREE.

    Preprocessing builds one complete :class:`DomainState` per forcing
    time from one horizontally interpolated analysis per forcing time.
    Streaming means ``resident_times`` of each coexist rather than all of
    them, which is the whole difference between this phase costing twice
    the forecast and costing less than half of it.  ``resident_times`` is
    now 1 rather than 2: see
    :data:`INGEST_RESIDENT_FORCING_TIMES` for which routes that describes
    and which one it does not.

    2026-08-01 AMENDMENT -- IT IS NOT ONE ROOT.  v1.4.0 priced this phase
    on the root domain alone, so the PREDICTION FELL as nests were added
    (5.46 -> 1.89 -> 1.30 GiB across one, two and four domains) while the
    16 GiB fleet node measured it FLAT at 4.0-4.8 GiB -- under by 2.1x at
    two domains and 3.4x at four, in the unsafe direction, on the very
    number the before-the-fetch gate half-relies on.  The mechanism was
    plain in the tool's own itemization: a deeper ladder has a SMALLER
    root, and only the root was priced, so adding domains made the answer
    shrink.  In the two-domain case the unpriced d02 had 4.9x the cells
    of the priced root.

    The hierarchy is initialized and exported as ONE transaction, so
    every domain's initial state is on the device together at the moment
    it is written: :attr:`nest_state_bytes` prices the nests, and the
    per-call setup transient is charged against the WIDEST domain in the
    tree rather than the root.
    """

    grid_id: int
    items: tuple[MemoryItem, ...]
    resident_times: int
    n_forcing_times: int
    boundary_frame_bytes: int
    #: Initial-state residency of every domain BELOW the root, summed:
    #: the hierarchy export holds the whole tree at once.  Zero for a
    #: single-domain configuration, which is why this amendment cannot
    #: move any single-domain number.
    nest_state_bytes: int = 0
    #: Analysis + state of the widest domain in the tree, whatever its
    #: depth; the setup transient is charged against this.
    widest_domain_time_bytes: int = 0
    #: ``(grid_id, state_bytes)`` per nest, for the itemized report.
    nest_state_items: tuple[tuple[int, int], ...] = ()
    sequential_domains: bool = False
    #: HOST, NOT DEVICE, and the only three fields here that are.  Points
    #: on the SOURCE grid, 2-D source fields per valid time, and valid
    #: times the decoder will hold.  ``0`` is "not known", which is how a
    #: caller that cannot see the forcing files says so: the host figure
    #: is then ``None`` and the report prints NOT PRICED rather than a
    #: plausible number, the same convention
    #: :data:`SOURCE_ANALYSIS_LEVELS` states for an unpriced product.
    source_grid_points: int = 0
    host_fields_per_time: int = 0
    decoded_valid_times: int = 0
    host_retained_copies: int = 0
    headroom: float = ALLOCATOR_HEADROOM
    context_bytes: int = CUDA_CONTEXT_BYTES
    device_overhead_bytes: int = field(
        default_factory=lambda: platform_projection_constants()[1])
    preprocess_backend: str = "cuda"
    #: HOST bytes of the root's float64 lateral-forcing series
    #: (:func:`lbc_host_series_bytes`).  The CPU preparation keeps it on
    #: the state beside the float32 tables converted from it and the
    #: perimeter frames it was built from.  Zero for a root with no
    #: tabulated series.
    host_boundary_series_bytes: int = 0
    #: One forcing time's analysis on every NEST, summed: the target-grid
    #: size of the source fields each child's own preparation maps.
    nest_analysis_bytes: int = 0
    #: Largest child's one-time analysis. Child state initialization is
    #: sequential even when the independent input mappings overlap.
    widest_nest_analysis_bytes: int = 0
    #: The itemized setup of the most expensive build in the tree: its
    #: vertical plans and outputs times the setup residual, plus the
    #: residual on its analysis (:mod:`woof.ingest.preparation_price`).
    setup_bytes: int = 0

    def category_bytes(self, category: str) -> int:
        return sum(item.nbytes for item in self.items
                   if item.category == category)

    @property
    def per_time_bytes(self) -> int:
        """One ROOT forcing time: its analysis plus the state built from it."""
        return sum(item.nbytes for item in self.items
                   if item.category in ("analysis", "state"))

    @property
    def transient_basis_bytes(self) -> int:
        """The domain the per-call setup transient is charged against."""
        return max(self.widest_domain_time_bytes, self.per_time_bytes)

    @property
    def forcing_table_bytes(self) -> int:
        """The completed boundary tables, uploaded onto the initial state."""
        return self.category_bytes("lbc")

    @property
    def resident_bytes(self) -> int:
        if self.sequential_domains:
            # A prepared-cache publisher releases each domain before building
            # the next; completed root boundaries remain on the host.
            return max(self.resident_times * self.per_time_bytes + self.forcing_table_bytes,
                       self.widest_domain_time_bytes)
        return (self.resident_times * self.per_time_bytes
                + self.forcing_table_bytes + self.nest_state_bytes)

    @property
    def unstreamed_resident_bytes(self) -> int:
        """What this phase cost before it streamed -- every time at once."""
        return (self.n_forcing_times * self.per_time_bytes
                + self.forcing_table_bytes + self.nest_state_bytes)

    @property
    def host_forcing_bytes(self) -> int | None:
        """HOST residency of the decoded forcing, or ``None`` -- NOT PRICED.

        Not a device figure and not part of any total above it: this is
        what the decoding process holds in RAM, before the first byte
        reaches the card.  A FLOOR -- see
        :data:`INGEST_HOST_RETAINED_COPIES` for the two terms it does not
        claim.

        ``decoded_valid_times`` is the count the DECODER will take, which
        is not the forecast's ``n_forcing_times``: the catalog selects the
        longest contiguous run of times present in the forcing files
        (`woof/ingest/preflight.py:470-497`) and never sees
        ``run_seconds``, so a window fetched longer than the run is
        decoded in full and held in full.
        """
        if not (self.source_grid_points and self.host_fields_per_time
                and self.decoded_valid_times and self.host_retained_copies):
            return None
        return (INGEST_HOST_DECODE_BYTES_PER_POINT
                * int(self.source_grid_points)
                * int(self.host_fields_per_time)
                * int(self.decoded_valid_times)
                * int(self.host_retained_copies))

    @property
    def host_preprocess_floor_bytes(self) -> int:
        """HOST arrays the CPU preparation certainly holds at one moment.

        The start time is built last and kept (``gfs_direct``'s
        start-last loop), and when its boundaries are attached these are
        all alive together: that time's analysis and state, every nest's
        initial state, the float32 forcing tables on the state, the
        float64 series they were converted from, and the float64
        perimeter frames that series was built from.  Nothing else is
        counted: no setup temporaries, no statics, no decoder buffers and
        no allocator headroom.  This is the figure a host-RAM refusal
        weighs, so it has to stay under the real peak at every forecast
        length and level count; :data:`CPU_PREPARATION_PEAK_BASIS` says
        where that was measured.  Zero on the device road.
        """
        if self.preprocess_backend != "cpu":
            return 0
        tables = sum(item.nbytes for item in self.items
                     if item.name == "lbc_forcing_tables")
        return (self.resident_times * self.per_time_bytes
                + self.nest_state_bytes + tables
                + self.host_boundary_series_bytes + self.boundary_frame_bytes)

    @property
    def tree_analysis_bytes(self) -> int:
        """One forcing time's analysis on every domain of the tree."""
        return self.category_bytes("analysis") + self.nest_analysis_bytes

    @property
    def host_preprocess_bytes(self) -> int:
        """CPU working-set estimate, distinct from the decoded forcing.

        A single domain charges its state floor plus
        :data:`CPU_PREPARATION_ANALYSIS_MULTIPLE` times its analysis.
        A nested tree charges the whole tree's state floor plus the
        largest domain's active scratch and retained child inputs/results.
        The retained term excludes the widest child, which the measurements
        place inside the active envelope even when the root is active.
        Both cases carry
        :data:`CPU_PREPARATION_ANALYSIS_MULTIPLE_PER_INTERVAL` times the
        root's analysis for every forcing interval.  A best estimate of
        the real peak, calibrated to sit at or just above it
        (:data:`CPU_PREPARATION_PEAK_BASIS`), which is what sizing steers
        on.  The device model's setup and pool headroom are not reused
        here: both describe a CUDA pool, and carried onto the host its
        earlier form priced a 6 h preparation at about 1.5 times what it
        held.  The decoder's separate retained input
        footprint can still be unknown; this term must never be
        presented as all host RAM.
        """
        if self.preprocess_backend != "cpu":
            return 0
        intervals = max(0, int(self.n_forcing_times) - 1)
        root_analysis = self.category_bytes("analysis")
        from woof.ingest.preparation_workers import effective_workers, WORKER_SCRATCH_BYTES
        # The original peak calibration used eight workers. Wider pools
        # reserve allocator/column scratch per extra worker instead of
        # imposing a machine-wide eight-thread ceiling.
        worker_bytes = max(0, effective_workers() - 8) * WORKER_SCRATCH_BYTES
        interval_bytes = (CPU_PREPARATION_ANALYSIS_MULTIPLE_PER_INTERVAL
                          * intervals * root_analysis)
        if not self.nest_analysis_bytes:
            return math.ceil(
                self.host_preprocess_floor_bytes
                + CPU_PREPARATION_ANALYSIS_MULTIPLE * root_analysis
                + interval_bytes + worker_bytes)
        active_analysis = max(root_analysis, self.widest_nest_analysis_bytes)
        tree_peak = (
            self.host_preprocess_floor_bytes
            + CPU_PREPARATION_TREE_ACTIVE_ANALYSIS_MULTIPLE * active_analysis
            + CPU_PREPARATION_RETAINED_NEST_ANALYSIS_MULTIPLE
            * (self.nest_analysis_bytes - self.widest_nest_analysis_bytes))
        return math.ceil(tree_peak + interval_bytes + worker_bytes)

    @property
    def host_peak_estimate_bytes(self) -> int | None:
        """Known decoder plus CPU working set; unknown decode stays unknown."""
        forcing = self.host_forcing_bytes
        return None if forcing is None else forcing + self.host_preprocess_bytes

    @property
    def host_floor_with_forcing_bytes(self) -> int | None:
        """Known decoder plus the CPU floor; unknown decode stays unknown."""
        forcing = self.host_forcing_bytes
        return (None if forcing is None
                else forcing + self.host_preprocess_floor_bytes)

    @property
    def transient_bytes(self) -> int:
        """The itemized setup of the ONE domain being built (A65)."""
        return int(self.setup_bytes)

    @property
    def subtotal_bytes(self) -> int:
        return self.resident_bytes + self.transient_bytes

    @property
    def alloc_estimate_bytes(self) -> int:
        """Subtotal under the same allocator headroom the forecast uses."""
        return math.ceil(self.headroom * self.subtotal_bytes)

    @property
    def peak_envelope_bytes(self) -> int:
        """What a machine watching this phase should expect to see.

        Pool allocations plus the non-pool CUDA context.  No second
        envelope multiplier: unlike the forecast, this phase has no
        per-step churn for a retention factor to model -- it builds one
        forcing time at a time, keeps ``resident_times`` of them, and
        drops the rest.
        """
        if self.preprocess_backend == "cpu":
            return 0
        return (self.alloc_estimate_bytes + self.context_bytes
                + self.device_overhead_bytes)


def estimate_ingest(exp: ExperimentConfig, *, source: str,
                    forcing_interval_seconds: float
                    = DEFAULT_FORCING_INTERVAL_SECONDS,
                    forcing_intervals: int | None = None,
                    vram_gib: float | None = None,
                    profile: DeviceLocalMemoryProfile | None = None,
                    source_grid_points: int | None = None,
                    decoded_valid_times: int | None = None,
                    source_fields_per_time: int | None = None,
                    analysis_shapes_by_domain: Mapping[int, Mapping[str, tuple[int, ...]]] | None = None,
                    sequential_domains: bool = False,
                    preprocess_backend: str = "cuda",
                    route: str | None = None,
                    ) -> IngestMemoryEstimate:
    """Itemize the preprocessing phase of ``exp``'s WHOLE DOMAIN TREE.

    ``route`` names the preparation route row
    (:data:`woof.ingest.preparation_price.PREPARATION_ROUTES`) the door
    prepares on, and the allocator headroom is that row's, as the door's
    own price takes it; ``None`` takes the measured default.

    The root carries the streamed forcing times and the boundary tables.
    Every nest carries one complete initial state, and they are resident
    together: the hierarchy is verified and exported as a single atomic
    transaction, not domain by domain with a release in between.

    ``source_grid_points`` and ``decoded_valid_times`` describe the FORCING
    FILES rather than the experiment, so nothing in ``exp`` can supply
    them.  Given both, the estimate also carries
    :attr:`IngestMemoryEstimate.host_forcing_bytes`; omitted -- the
    default, and every existing caller -- the host term is ``None`` and
    the report says NOT PRICED.  ``source_fields_per_time`` is the same
    kind of fact and is the DECODED count when a caller holds a catalog;
    without it the nominal :func:`source_analysis_fields_per_time`
    inventory stands in, which is right for the itemization and too
    coarse to gate a refusal on.
    """
    if not isinstance(sequential_domains, bool):
        raise TypeError("sequential_domains must be boolean")
    if preprocess_backend not in ("cpu", "cuda", "auto"):
        raise ValueError("preprocess backend must be cpu, cuda or auto")
    if analysis_shapes_by_domain is not None:
        if not isinstance(analysis_shapes_by_domain, Mapping) or set(analysis_shapes_by_domain) != {domain.grid_id for domain in exp.domains}:
            raise ValueError("actual analysis inventory must name every experiment domain exactly once")
    # The model top sets how many source levels the fetch takes, and one
    # top serves the whole tree: the vertical block is shared.
    p_top_pa = getattr(getattr(exp, "vertical", None), "p_top", None)
    def analysis_shapes(domain):
        return ingest_analysis_shapes(domain.run, source=source,
            actual_shapes=None if analysis_shapes_by_domain is None else analysis_shapes_by_domain[domain.grid_id],
            p_top_pa=p_top_pa)
    dc = exp.root
    run = dc.run
    n_intervals = lbc_intervals(exp.run_seconds, forcing_interval_seconds,
                                retained_intervals=forcing_intervals)
    items: list[MemoryItem] = []
    items += _items("analysis", analysis_shapes(dc))
    items += _items("state", state_array_shapes(run))
    from woof.ingest.preparation_price import (
        SETUP_RESIDUAL, SourceInventory, route_pool_headroom,
        vertical_setup_bytes)

    headroom = route_pool_headroom(route)

    def setup_of(domain, analysis_nbytes):
        inventory = SourceInventory.from_shapes(analysis_shapes(domain))
        return math.ceil(
            SETUP_RESIDUAL * vertical_setup_bytes(domain.run, inventory)
            + (SETUP_RESIDUAL - 1.0) * analysis_nbytes)

    setup = setup_of(dc, sum(item.nbytes for item in items
                             if item.category == "analysis"))
    registry = scratch_slot_registry(
        run, n_lbc_intervals=(n_intervals if run.specified else 0))
    items += _items("lbc", {slot: shape for slot, shape in registry.items()
                            if slot.startswith("lbc_")})
    # Every NEST: one complete initial state each, all resident for the
    # export transaction.  A nest is initialized from its parent, so it
    # carries no second copy of the source analysis -- but the setup
    # transient is charged against whichever domain is widest, which on
    # a real ladder is usually a nest and not the root.
    nest_items: list[tuple[int, int]] = []
    nest_analysis = 0
    widest_nest_analysis = 0
    widest = sum(item.nbytes for item in items
                 if item.category in ("analysis", "state"))
    for child in exp.domains:
        if child.grid_id == dc.grid_id:
            continue
        child_run = child.run
        state = sum(4 * math.prod(shape)
                    for shape in state_array_shapes(child_run).values())
        analysis = sum(
            4 * math.prod(shape) for shape in
            analysis_shapes(child).values())
        nest_items.append((child.grid_id, state))
        setup = max(setup, setup_of(child, analysis))
        nest_analysis += analysis
        widest_nest_analysis = max(widest_nest_analysis, analysis)
        widest = max(widest, state + analysis)
    # The host-side perimeter frames StateBoundaryFrames retains: float64,
    # four sides, every forcing time.  Reported so the phase's HOST cost
    # is visible too; it is not part of the device residency above.
    width = int(run.spec_bdy_width)
    from woof.boundary_fields import source_boundary_species
    frame_elements = sum(
        2 * width * (dims[1] + dims[2]) * dims[0]
        for dims in _lbc_field_dims(
            run, boundary_species=source_boundary_species(source)).values())
    return IngestMemoryEstimate(
        grid_id=dc.grid_id, items=tuple(items),
        resident_times=INGEST_RESIDENT_FORCING_TIMES,
        n_forcing_times=n_intervals + 1,
        boundary_frame_bytes=8 * frame_elements * (n_intervals + 1),
        nest_state_bytes=0 if sequential_domains else sum(nbytes for _, nbytes in nest_items),
        sequential_domains=sequential_domains,
        widest_domain_time_bytes=widest,
        nest_state_items=tuple(nest_items),
        source_grid_points=int(source_grid_points or 0),
        host_fields_per_time=(int(source_fields_per_time) if source_fields_per_time is not None else
            math.ceil(sum(math.prod(shape) for shape in analysis_shapes(dc).values()) / (run.nx*run.ny))
            if analysis_shapes_by_domain is not None else source_analysis_fields_per_time(source, p_top_pa=p_top_pa)),
        decoded_valid_times=int(decoded_valid_times or 0),
        host_retained_copies=INGEST_HOST_RETAINED_COPIES.get(
            str(source).strip().lower(), 0),
        # This card's context, not the retired flat constant: ingest
        # stands up the same CUDA context the forecast does.
        context_bytes=(0 if preprocess_backend == "cpu" else
                       (MEASURED_LOCAL_MEMORY_PROFILE if profile is None
                        else profile).cuda_context_bytes),
        device_overhead_bytes=(0 if preprocess_backend == "cpu" else
                               platform_projection_constants(vram_gib=vram_gib)[1]),
        preprocess_backend=preprocess_backend,
        host_boundary_series_bytes=lbc_host_series_bytes(
            run, n_intervals, source=source),
        nest_analysis_bytes=nest_analysis,
        widest_nest_analysis_bytes=widest_nest_analysis,
        setup_bytes=setup,
        headroom=headroom,
    )


@dataclass(frozen=True)
class HostStateInitializationEstimate:
    """Remaining initialization, with CUDA transforms and a host setup state.

    The device transient uses the established ingest model. Host bytes are
    a known allocation floor, not a claimed upper bound on initializer
    temporaries. Already decoded source arrays are excluded: callers take
    available-memory readings while those inputs are live.
    """

    device: IngestMemoryEstimate
    host_state_bytes: int
    host_boundary_bytes: int

    @property
    def host_floor_bytes(self) -> int:
        return self.host_state_bytes + self.host_boundary_bytes


def estimate_host_state_initialization(
        cfg: RunConfig, *, analysis_shapes: Mapping[str, tuple[int, ...]],
        forcing_times: int, vram_gib: float | None = None,
        profile: DeviceLocalMemoryProfile | None = None
        ) -> HostStateInitializationEstimate:
    """Price the remaining real initialization from actual mapped fields.

    This does not price horizontal interpolation, and never selects a
    preprocessing backend. The caller retains CUDA and changes only where
    the finished state resides. The old resident estimate is untouched.
    """
    if isinstance(forcing_times, bool) or not isinstance(forcing_times, int) or forcing_times < 2:
        raise ValueError("forcing_times must be an integer of at least two")
    shapes = ingest_analysis_shapes(cfg, source="", actual_shapes=analysis_shapes)
    analysis = _items("analysis", shapes)
    state_bytes = sum(item.nbytes for item in _items("state", state_array_shapes(cfg)))
    width = int(cfg.spec_bdy_width)
    frame_elements = sum(
        2 * width * (dims[1] + dims[2]) * dims[0]
        for dims in _lbc_field_dims(cfg).values())
    frame_bytes = 8 * frame_elements * forcing_times
    # Value+tendency in immutable FP64 series plus the FP32 host-state
    # attachment, while StateBoundaryFrames still retains its perimeter.
    boundary_bytes = frame_bytes + 24 * frame_elements * (forcing_times - 1)
    from woof.ingest.preparation_price import (
        PREPARATION_POOL_HEADROOM, SETUP_RESIDUAL, SourceInventory,
        vertical_setup_bytes)
    analysis_nbytes = sum(item.nbytes for item in analysis)
    setup = math.ceil(
        SETUP_RESIDUAL * vertical_setup_bytes(
            cfg, SourceInventory.from_shapes(shapes))
        + (SETUP_RESIDUAL - 1.0) * analysis_nbytes)
    device = IngestMemoryEstimate(
        grid_id=int(cfg.grid_id), items=analysis, resident_times=1,
        n_forcing_times=forcing_times, boundary_frame_bytes=frame_bytes,
        widest_domain_time_bytes=sum(item.nbytes for item in analysis) + state_bytes,
        setup_bytes=setup, headroom=PREPARATION_POOL_HEADROOM,
        context_bytes=(MEASURED_LOCAL_MEMORY_PROFILE if profile is None
                       else profile).cuda_context_bytes,
        device_overhead_bytes=platform_projection_constants(vram_gib=vram_gib)[1])
    return HostStateInitializationEstimate(device, state_bytes, boundary_bytes)


@dataclass(frozen=True)
class PhaseMemoryEstimate:
    """Both phases of one run, and which of them binds the card.

    A configuration fits when the LARGEST phase fits.  Pricing only the
    forecast is what let a wizard size a domain to a card, watch the user
    download 81 GFS files, and then fail in preprocessing.
    """

    forecast: ExperimentMemoryEstimate
    ingest: IngestMemoryEstimate | None
    forecast_envelope_bytes: int
    ingest_envelope_bytes: int | None
    source: str | None = None
    #: The STREAMED forecast envelope, when ``[tiles]`` resolves to streaming
    #: this domain.  Present means ``forecast_envelope_bytes`` above is the
    #: number that binds -- it is already the streamed one, not the resident
    #: one -- and this carries the tiling that produced it so the verdict can
    #: name it.  ``None`` is every resident run, where nothing changed.
    #: For a NESTED tree this is the mixed-road
    #: :class:`woof.core.streaming.TreeRoadPlan` (recognisable by its
    #: ``rows``), on the same contract: present means the forecast term is
    #: the mixed road's.
    streamed: "StreamedEnvelope | None" = None
    #: The mixed-road walk of a nested ``[tiles]`` tree, kept even when it
    #: cannot replace the forecast term (an all-resident decision, a
    #: refused walk) so the report can print the run door's answer -- the
    #: roads, the claims, or the refusal -- beside the resident context.
    tree_road: object | None = None
    #: The resident forecast envelope, kept beside the streamed one so a
    #: report can say what streaming BOUGHT.  Equal to
    #: ``forecast_envelope_bytes`` on a resident run.
    resident_forecast_envelope_bytes: int | None = None
    preprocess_backend: str = "cuda"
    #: RAM of the machine this estimate was priced for (its planner
    #: ``Machine.host_bytes``), or ``None`` when no machine was given.  What
    #: :meth:`host_preparation_refusal` weighs the CPU preparation against.
    host_ram_bytes: int | None = None
    #: The source whose published hydrometeors the forecast terms' root
    #: boundary tables carry (:func:`estimate_phases`' ``boundary_source``),
    #: so a surface re-taking a decision from these phases prices the same
    #: tables.  ``None`` prices water vapour only.
    boundary_source: object = None

    @property
    def ingest_priced(self) -> bool:
        return self.ingest_envelope_bytes is not None

    @property
    def streamed_forecast(self) -> bool:
        return self.streamed is not None

    @property
    def mixed_road(self) -> bool:
        """Whether the forecast term is a nested tree's MIXED road.

        The two things :attr:`streamed` can be are told apart by the one
        thing only the tree plan has: per-domain ``rows``.
        """
        return getattr(self.streamed, "rows", None) is not None

    @property
    def pace_streamed(self) -> object | None:
        """The envelope the PACE model should be handed, not the memory one.

        :func:`woof.core.pace.estimate_pace` prices the ROOT's road and
        charges every nest resident, so it reads ``tile_nx``/``halo``/
        ``nbuffers`` off a single domain's envelope.  A
        :class:`woof.core.streaming.TreeRoadPlan` has no single tiling --
        that is the whole point of a mixed road -- so the root's own
        envelope is what goes to pace, and ``None`` when the walk left the
        root RESIDENT.  Handing the tree plan over instead would have the
        pace line say "streamed road" about a root that is never tiled and
        quote a bus floor for bytes that never cross the bus.
        """
        if not self.mixed_road:
            return self.streamed
        return getattr(self.streamed, "root_envelope", None)

    @property
    def binding_phase(self) -> str:
        if not self.ingest_priced:
            return "forecast"
        return ("ingest"
                if self.ingest_envelope_bytes > self.forecast_envelope_bytes
                else "forecast")

    @property
    def peak_envelope_bytes(self) -> int:
        if not self.ingest_priced:
            return self.forecast_envelope_bytes
        return max(self.forecast_envelope_bytes, self.ingest_envelope_bytes)

    @property
    def host_preparation_bytes(self) -> int:
        """HOST RAM the preparation phase holds when it runs on the CPU.

        The CPU preparation road (:mod:`woof.preprocess_policy`) takes
        the whole ingest working set off the card, which is why
        ``ingest_envelope_bytes`` is zero there, and puts it in host RAM.
        The ingest admission term does not disappear on that road; it
        changes memories, and this is the figure it becomes.  Zero on the
        device road and when the ingest phase is not priced.

        A BEST ESTIMATE of the peak
        (:attr:`IngestMemoryEstimate.host_preprocess_bytes`), which is
        what sizing steers on and what a verdict reports.  A refusal
        weighs :attr:`host_preparation_floor_bytes` instead.  The decoded
        forcing is held beside the working set, so where the forcing files
        are visible this is both together
        (:attr:`IngestMemoryEstimate.host_peak_estimate_bytes`); where they
        are not it is the working set alone.
        """
        if self.ingest is None or self.ingest.preprocess_backend != "cpu":
            return 0
        peak = self.ingest.host_peak_estimate_bytes
        if peak is not None:
            return int(peak)
        return int(self.ingest.host_preprocess_bytes)

    @property
    def host_preparation_floor_bytes(self) -> int:
        """HOST RAM the CPU preparation certainly holds at one moment.

        :attr:`IngestMemoryEstimate.host_preprocess_floor_bytes`, plus the
        decoded forcing where the files are visible (itself a floor).  No
        temporaries and no headroom, so a preparation refused on it could
        not have completed.  Zero on the device road.
        """
        if self.ingest is None or self.ingest.preprocess_backend != "cpu":
            return 0
        floor = self.ingest.host_floor_with_forcing_bytes
        if floor is not None:
            return int(floor)
        return int(self.ingest.host_preprocess_floor_bytes)

    def _weighed_host(self, host_bytes):
        if host_bytes is None:
            host_bytes = self.host_ram_bytes
        if host_bytes is None or isinstance(host_bytes, bool):
            return None
        return int(host_bytes)

    def host_preparation_refusal(self, host_bytes: int | None = None
                                 ) -> str | None:
        """Why this CPU preparation cannot be held in ``host_bytes`` of RAM.

        ``host_bytes`` defaults to :attr:`host_ram_bytes`.  ``None`` when it
        fits, when the preparation runs on the card, and when the machine's
        RAM is unknown: unknown RAM never refuses.  Weighed against the
        machine's WHOLE RAM rather than a page-locking share, because
        preparation memory is ordinary pageable memory, and weighed with
        :attr:`host_preparation_floor_bytes`, the arrays the preparation
        cannot run without holding together.  More of those than all of
        the RAM runs the machine out of memory after the forcing has been
        downloaded, and on Linux the kernel kills the process from outside
        with no woof message: that is the breakage this names.  A best
        estimate over the RAM with a floor under it is
        :meth:`host_preparation_warning`, never a refusal.
        """
        host_bytes = self._weighed_host(host_bytes)
        floor = self.host_preparation_floor_bytes
        if not floor or host_bytes is None or floor <= host_bytes:
            return None
        held = ("the decoded forcing, the start time's analysis and state "
                "and the lateral-boundary tables, series and frames"
                if self.ingest.host_floor_with_forcing_bytes is not None else
                "the start time's analysis and state and the "
                "lateral-boundary tables, series and frames; the forcing "
                "decode is held on top of it")
        return (f"preparing this configuration on the CPU holds at least "
                f"{floor / GIB:.2f} GiB of host RAM at once ({held}), and "
                f"about {self.host_preparation_bytes / GIB:.2f} GiB at its "
                f"peak, more than the {host_bytes / GIB:.2f} GiB this "
                "machine has, so the preparation would run out of memory "
                "after the download; a smaller domain, fewer vertical "
                "levels, a shorter forecast or a machine with more RAM "
                "moves it")

    def streamed_host_refusal(self) -> str | None:
        """Why the streamed forecast cannot be held in this host's RAM.

        THE ONE HOST ADMISSION FOR A STREAMED FORECAST.  ``woof go``
        refuses on it before the download, ``woof check`` fails on it and
        ``woof domain`` sizes against it, so a configuration one of them
        admits is never one another refuses.  Weighed is everything the
        streamed run holds in host RAM for the whole run
        (:attr:`StreamedEnvelope.host_bytes`: the pinned store and its
        arena plus the lateral-boundary series) against the page-locking
        budget of the machine it was priced for.

        The tile planner weighs the store and arena alone, so a domain
        whose store fits and whose boundary series does not was sized by
        ``woof domain`` and passed by ``woof check`` while ``woof go``
        refused it (measured: a 1158x928x55 3 km GFS domain at 14.20 GiB
        against a 14.13 GiB budget on a 30 GiB worker).  A forecast over
        this budget runs the host out of memory after the download.

        ``None`` for a resident forecast, when it fits, and when the host
        RAM is unknown: unknown RAM never refuses.
        """
        env = self.streamed
        if env is None:
            return None
        budget = getattr(env, "host_budget_bytes", None)
        if budget is None or int(env.host_bytes) <= int(budget):
            return None
        boundary = int(getattr(env, "boundary_table_bytes", 0) or 0)
        if boundary:
            return (f"the streamed forecast holds "
                    f"{env.host_bytes / GIB:.2f} GiB of host RAM "
                    f"({env.pinned_bytes / GIB:.2f} GiB pinned store and "
                    f"arena plus {boundary / GIB:.2f} GiB of lateral-boundary "
                    f"tables) against a {budget / GIB:.2f} GiB host budget, "
                    "which is more host RAM than this machine allows a "
                    "forecast to hold, and the run would find that out "
                    "after the download")
        return (f"the pinned host store is {env.host_bytes / GIB:.2f} GiB "
                f"against a {budget / GIB:.2f} GiB page-locking budget, "
                "which is where a streamed domain actually lives")

    def host_preparation_warning(self, host_bytes: int | None = None
                                 ) -> str | None:
        """What to say when the best estimate, not the floor, passes RAM.

        Such a preparation may complete or may run out of memory, depending
        on what else the machine holds, so it is admitted and told.
        ``None`` whenever :meth:`host_preparation_refusal` would refuse, and
        whenever the estimate fits.
        """
        host_bytes = self._weighed_host(host_bytes)
        need = self.host_preparation_bytes
        if (not need or host_bytes is None or need <= host_bytes
                or self.host_preparation_refusal(host_bytes) is not None):
            return None
        return (f"preparing this configuration on the CPU is estimated to "
                f"peak at {need / GIB:.2f} GiB of host RAM, above the "
                f"{host_bytes / GIB:.2f} GiB this machine has, so it may run "
                "out of memory after the download; it is not refused "
                f"because the {self.host_preparation_floor_bytes / GIB:.2f} "
                "GiB it certainly holds at once fits; a smaller domain, "
                "fewer vertical levels or a shorter forecast makes room")

    def _ingest_clause(self) -> str:
        """The ingest term inside a verdict's parenthesis.

        On the CPU road the card holds nothing in that phase, and a bare
        "ingest 0.00 GiB" reads as a phase that costs nothing; the host RAM
        it holds instead is said beside it.
        """
        text = f", ingest {self.ingest_envelope_bytes / GIB:.2f} GiB"
        host = self.host_preparation_bytes
        if host:
            text += (f" of card and about {host / GIB:.2f} GiB of host RAM "
                     "on the CPU")
        return text

    def fits(self, budget_bytes: int) -> bool:
        return self.peak_envelope_bytes <= int(budget_bytes)

    def verdict(self, budget_bytes: int | None) -> str:
        """One sentence naming the binding phase and its number.

        Under ``[tiles]`` the forecast term is the STREAMED envelope, so the
        sentence has to say so: a reader who sees "the forecast needs 6.61
        GiB" for a 550x550 domain that manifestly cannot fit in 6.61 GiB
        resident is owed the reason in the same breath, and the tiling is
        the reason.
        """
        if self.streamed_forecast:
            return self._streamed_verdict(budget_bytes)
        if not self.ingest_priced:
            text = (f"the forecast needs "
                    f"{self.forecast_envelope_bytes / GIB:.2f} GiB peak "
                    f"envelope, and preprocessing for --source "
                    f"{self.source} is NOT PRICED here, so this is the "
                    "forecast phase only")
        else:
            phase = self.binding_phase
            label = ("preprocessing (ingest)" if phase == "ingest"
                     else "the forecast")
            text = (f"{label} is the memory-binding phase at "
                    f"{self.peak_envelope_bytes / GIB:.2f} GiB peak "
                    f"envelope (forecast "
                    f"{self.forecast_envelope_bytes / GIB:.2f} GiB"
                    f"{self._ingest_clause()})")
        if budget_bytes is None:
            return text
        if self.fits(budget_bytes):
            return (f"{text}; it fits the "
                    f"{budget_bytes / GIB:.2f} GiB budget with "
                    f"{(budget_bytes - self.peak_envelope_bytes) / GIB:.2f} "
                    "GiB to spare")
        return (f"{text}; that EXCEEDS the {budget_bytes / GIB:.2f} GiB "
                f"budget by "
                f"{(self.peak_envelope_bytes - budget_bytes) / GIB:.2f} GiB")

    def _budget_tail(self, text: str, budget_bytes: int | None) -> str:
        if budget_bytes is None:
            return text
        if self.fits(budget_bytes):
            return (f"{text}; it fits the {budget_bytes / GIB:.2f} GiB "
                    "budget with "
                    f"{(budget_bytes - self.peak_envelope_bytes) / GIB:.2f} "
                    "GiB to spare")
        return (f"{text}; that EXCEEDS the {budget_bytes / GIB:.2f} GiB "
                f"budget by "
                f"{(self.peak_envelope_bytes - budget_bytes) / GIB:.2f} GiB")

    def _tree_road_verdict(self, budget_bytes: int | None) -> str:
        env = self.streamed
        phase = self.binding_phase
        label = ("preprocessing (ingest)" if phase == "ingest"
                 else "the mixed-road forecast")
        parts = [f"{label} is the memory-binding phase at "
                 f"{self.peak_envelope_bytes / GIB:.2f} GiB peak envelope "
                 f"(mixed-road forecast "
                 f"{self.forecast_envelope_bytes / GIB:.2f} GiB"]
        if self.ingest_priced:
            parts.append(self._ingest_clause())
        parts.append(")")
        text = "".join(parts)
        text += "; " + env.summary()
        if self.resident_forecast_envelope_bytes is not None:
            text += (f", against "
                     f"{self.resident_forecast_envelope_bytes / GIB:.2f} GiB "
                     "with the whole tree resident")
        boundary = int(getattr(env, "boundary_table_bytes", 0) or 0)
        if boundary:
            text += (f", with the streamed domain(s) in "
                     f"{env.pinned_bytes / GIB:.2f} GiB of pinned host RAM "
                     f"plus {boundary / GIB:.2f} GiB of the root's "
                     f"lateral-boundary tables")
        elif env.host_bytes:
            text += (f", with the streamed domain(s) in "
                     f"{env.host_bytes / GIB:.2f} GiB of pinned host RAM")
        return self._budget_tail(text, budget_bytes)

    def _streamed_verdict(self, budget_bytes: int | None) -> str:
        env = self.streamed
        if getattr(env, "rows", None) is not None:
            return self._tree_road_verdict(budget_bytes)
        phase = self.binding_phase
        label = ("preprocessing (ingest)" if phase == "ingest"
                 else "the streamed forecast")
        parts = [f"{label} is the memory-binding phase at "
                 f"{self.peak_envelope_bytes / GIB:.2f} GiB peak envelope "
                 f"(streamed forecast "
                 f"{self.forecast_envelope_bytes / GIB:.2f} GiB"]
        if self.ingest_priced:
            parts.append(self._ingest_clause())
        parts.append(")")
        text = "".join(parts)
        if self.resident_forecast_envelope_bytes is not None:
            # The tiling, its tile count and its redundancy: the numbers a
            # streamed step's pace follows (measured 2026-09-26: a 1,190-tile
            # sweep printed only its 45x45 window here).
            shape = (f" ({env.tiling_text()})"
                     if hasattr(env, "tiling_text") else "")
            text += (f"; {env.nbuffers} tile buffer(s) of "
                     f"{env.window_nx}x{env.window_ny}{shape} instead of the "
                     f"whole domain, against "
                     f"{self.resident_forecast_envelope_bytes / GIB:.2f} GiB "
                     "resident")
        if env.radiation_transient_bytes:
            # NAMED WHERE THE FIGURE IS READ.  A forecast term larger than
            # the tile buffers it describes reads as an arithmetic error
            # unless the reservation is stated beside it.
            text += (f"; the tiles hold {env.vram_bytes / GIB:.2f} GiB and "
                     f"the RRTMGP call adds a measured "
                     f"{env.radiation_transient_bytes / GIB:.2f} GiB "
                     f"transient on top of them, which is the peak the card "
                     "has to hold")
        if getattr(env, "boundary_table_bytes", 0):
            text += (f", with the forecast itself in "
                     f"{env.pinned_bytes / GIB:.2f} GiB of pinned host RAM "
                     f"plus {env.boundary_table_bytes / GIB:.2f} GiB of "
                     f"lateral-boundary tables")
        else:
            text += (f", with the forecast itself in "
                     f"{env.host_bytes / GIB:.2f} GiB of pinned host RAM")
        if budget_bytes is None:
            return text
        if self.fits(budget_bytes):
            return (f"{text}; it fits the {budget_bytes / GIB:.2f} GiB "
                    "budget with "
                    f"{(budget_bytes - self.peak_envelope_bytes) / GIB:.2f} "
                    "GiB to spare")
        return (f"{text}; that EXCEEDS the {budget_bytes / GIB:.2f} GiB "
                f"budget by "
                f"{(self.peak_envelope_bytes - budget_bytes) / GIB:.2f} GiB")


def streamed_forecast_envelope(exp: ExperimentConfig, *, machine=None, resident_estimate=None,
                               forcing_interval_seconds: float | None = None,
                               forcing_intervals: int | None = None,
                               source=None):
    """The ROOT domain's streamed envelope under this config's ``[tiles]``.

    ``None`` whenever this configuration does not stream, which includes the
    cases where the question cannot be answered here: ``mode = "auto"`` with
    no pinned tiling has to consult the planner, and the planner needs a
    card.  Passing ``machine`` (built from an out-of-process probe, never
    from ``Machine.detect`` inside a long-lived CLI) is what lets ``auto``
    be priced.

    NEVER RAISES.  This runs inside an admission gate whose whole job is to
    answer before the user spends a download, and a planner refusal
    ("no tile fits in this budget") is a legitimate answer meaning "streaming
    will not save this either" -- the caller then prices the resident
    envelope and refuses on that, which is the correct and conservative
    outcome.

    The root domain only, deliberately: a nested tree is refused by
    ``prepared_domain_builder`` for a nest anyway, so pricing a nest's
    streamed envelope would describe a run that cannot happen.

    ``forcing_interval_seconds`` / ``forcing_intervals`` are the schedule
    the root's lateral forcing series is priced at (its host bytes are part
    of ``host_bytes``); omitted, the estimator's default cadence stands in.
    ``source`` is the forcing source, whose published hydrometeors that
    series carries (:func:`lbc_host_series_bytes`).
    """
    if not getattr(exp, "domains", None):
        return None
    from woof.core import streaming

    # THE ROOT'S OWN TABLE, through the one resolution every other surface
    # takes (:func:`woof.core.streaming.options_for_domain`).  Read raw,
    # ``exp.tiles`` answers for the TREE: a root carrying
    # ``tiles = {...}`` was priced here on a table that does not govern
    # it, so a root the run door streams was priced resident and a root
    # the run door keeps resident was priced streamed -- the same
    # two-answers-to-one-configuration this seam exists to close, one
    # table down.
    options = streaming.options_for_domain(
        exp.domains[0], getattr(exp, "tiles", None))
    if not options.enabled:
        return None
    try:
        return streaming.streamed_envelope(
            exp.domains[0].run, options, machine=machine, resident_estimate=resident_estimate,
            forcing_interval_seconds=forcing_interval_seconds,
            forcing_intervals=forcing_intervals, source=source)
    except Exception:                    # a gate never dies on its estimate
        return None


#: :func:`estimate_phases`' default ``boundary_source``: the ``source`` it
#: prices the ingest phase for.  A sentinel rather than ``None``, because
#: ``None`` is itself a boundary source (one that publishes nothing).
_SAME_AS_SOURCE = object()


def estimate_phases(exp: ExperimentConfig, *, source: str,
                    column_chunk: int | None = None,
                    forcing_interval_seconds: float
                    = DEFAULT_FORCING_INTERVAL_SECONDS,
                    forcing_intervals: int | None = None,
                    ingest_forcing_interval_seconds: float | None = None,
                    vram_gib: float | None = None,
                    profile: DeviceLocalMemoryProfile | None = None,
                    machine=None,
                    source_grid_points: int | None = None,
                    decoded_valid_times: int | None = None,
                    source_fields_per_time: int | None = None,
                    analysis_shapes_by_domain: Mapping[int, Mapping[str, tuple[int, ...]]] | None = None,
                    sequential_domains: bool = False,
                    preprocess_backend: str | None = None,
                    boundary_source=_SAME_AS_SOURCE,
                    preparation_route: str | None = None,
                    urban_columns=None,
                    resident_forcing_intervals: int | None = None,
                    ) -> PhaseMemoryEstimate:
    """Price every phase of ``exp`` and say which one binds the card.

    ``preparation_route`` is relayed to :func:`estimate_ingest` as its
    ``route`` (the preparation row whose allocator headroom it takes).

    ``ingest_forcing_interval_seconds`` defaults to the SOURCE's own
    fetch cadence rather than the forecast's LBC interval, because the
    ingest phase's time count is set by what was downloaded.

    ``[tiles]`` REPLACES THE FORECAST TERM (2.2.0).  Every enumeration in
    this module itemizes a domain resident in VRAM, so with streaming
    configured the forecast term described a run that was not going to
    happen -- and the gate built on it refused, by default, the one
    configuration class streaming exists to enable.  When
    :func:`streamed_forecast_envelope` returns a number the forecast term
    becomes that number. The shared preparation policy selects CPU for GFS
    host-store tiling declarations; that route retains its host itemization
    and consumes no GPU memory during preparation. Explicit CUDA/auto and
    other source contracts retain the existing device envelope.

    ``machine`` is only consulted for ``mode = "auto"`` with no pinned
    tiling, where the decision belongs to the planner.

    ``source_grid_points``/``decoded_valid_times``/
    ``source_fields_per_time`` are relayed to :func:`estimate_ingest`
    unchanged; they price the ingest phase's HOST residency, which no
    term above touches.

    ``resident_forcing_intervals`` is how many forcing intervals the
    RESIDENT forecast holds on the card when that differs from the schedule:
    a forecast started on a chained preparation's head streams its root's
    intervals through one device slot
    (:func:`woof.ingest.lateral_bc.attach_streaming_lateral_boundaries`),
    which is what its door prices (``forcing_intervals=1`` on a stream
    head).  ``None`` holds the whole schedule, as a sealed restore does.
    Only the resident forecast term reads it; the ingest phase and a
    streamed domain's host series keep ``forcing_intervals``.

    ``boundary_source`` is the source whose published hydrometeors the
    root's boundary tables carry, in every forecast term here; it is
    ``source`` unless given.  A surface that prices no ingest phase
    (``woof run-plan --estimate`` passes ``source=None``) names its
    configuration's forcing source here instead, so its forecast figure
    is the one ``woof check`` prices for the same file.
    """
    if boundary_source is _SAME_AS_SOURCE:
        boundary_source = source
    from woof.preprocess_policy import resolve_preprocess_backend
    preprocess_backend = resolve_preprocess_backend(
        source=source, experiment=exp, requested=preprocess_backend)
    from woof.boundary_fields import source_boundary_species
    forecast = estimate_experiment(
        exp, column_chunk=column_chunk,
        forcing_intervals=(forcing_intervals
                           if resident_forcing_intervals is None
                           else max(1, int(resident_forcing_intervals))),
        forcing_interval_seconds=forcing_interval_seconds,
        vram_gib=vram_gib, profile=profile,
        boundary_species=source_boundary_species(boundary_source),
        urban_columns=urban_columns,
        # The one-card residue bound, as the admission prices it.
        **single_card_residue_terms(exp))
    key = None if source is None else str(source).strip().lower()
    ingest = None
    if key in SOURCE_ANALYSIS_LEVELS or analysis_shapes_by_domain is not None:
        cadence = ingest_forcing_interval_seconds
        if cadence is None:
            cadence = INGEST_FORCING_CADENCE_SECONDS.get(
                key, forcing_interval_seconds)
        ingest = estimate_ingest(
            exp, source=key, forcing_interval_seconds=float(cadence),
            forcing_intervals=forcing_intervals,
            vram_gib=vram_gib, profile=profile,
            source_grid_points=source_grid_points,
            decoded_valid_times=decoded_valid_times,
            source_fields_per_time=source_fields_per_time,
            analysis_shapes_by_domain=analysis_shapes_by_domain, sequential_domains=sequential_domains,
            preprocess_backend=preprocess_backend, route=preparation_route)
    resident_forecast = forecast.peak_envelope_bytes
    tree_road = None
    if len(getattr(exp, "domains", ()) or ()) > 1:
        # A NESTED tree with [tiles] is priced by the same per-domain walk
        # the run door performs (streaming.steppers_for_tree's decide
        # pass), because the road the run actually takes is mixed --
        # each domain resident or streamed against the budget its
        # predecessors leave.  The root-only streamed envelope below
        # cannot describe that run, and pricing only the resident tree
        # is the defect that had `woof check` exit 1 against configs
        # whose mixed road fits and completes.
        from woof.core.streaming import tree_road_plan

        try:
            # Priced from the ADMISSION estimate, not from this report's
            # own forecast term: the run door asks the same question of the
            # same function, and a review that admitted a tree the door
            # then refused is the defect :func:`admission_estimate`
            # documents.  The report's resident term below is unchanged.
            # The forecast's own LBC schedule prices a streamed root's
            # forcing series, as it does on the single-domain road below.
            tree_road = tree_road_plan(
                exp, machine=machine,
                resident_estimate=admission_estimate(
                    exp, machine=machine, source=boundary_source,
                    urban_columns=urban_columns),
                forcing_interval_seconds=forcing_interval_seconds,
                forcing_intervals=forcing_intervals, source=boundary_source)
        except Exception:            # a gate never dies on its estimate
            tree_road = None
        streamed = (tree_road if tree_road is not None and tree_road.usable
                    else None)
    else:
        # The forecast's own LBC schedule, the one ``estimate_experiment``
        # sized the resident device tables with above: the streamed domain
        # keeps that series on the host, and it is part of its host claim.
        streamed = streamed_forecast_envelope(
            exp, machine=machine, resident_estimate=forecast,
            forcing_interval_seconds=forcing_interval_seconds,
            forcing_intervals=forcing_intervals, source=boundary_source)
    return PhaseMemoryEstimate(
        forecast=forecast, ingest=ingest,
        # THE PEAK, NOT THE HOLD.  ``vram_bytes`` is what a streamed
        # forecast holds between radiation calls; the RRTMGP call's
        # measured per-process transient is on the card too at the instant
        # it matters, and the first call is itimestep == 1.  Pricing the
        # hold here let a card that fits 6.71 GiB admit a run that reaches
        # 9.45 -- every surface below reads this field.
        forecast_envelope_bytes=(resident_forecast if streamed is None
                                 else int(streamed.peak_vram_bytes)),
        ingest_envelope_bytes=(None if ingest is None
                               else ingest.peak_envelope_bytes),
        source=key, streamed=streamed, tree_road=tree_road,
        resident_forecast_envelope_bytes=resident_forecast,
        preprocess_backend=preprocess_backend,
        host_ram_bytes=_machine_host_bytes(machine),
        boundary_source=boundary_source,
    )


def _machine_host_bytes(machine) -> int | None:
    """A planner machine's RAM as a positive int, or ``None``."""
    host = getattr(machine, "host_bytes", None)
    if isinstance(host, bool) or not isinstance(host, int) or host <= 0:
        return None
    return host


def pool_retention_residual_bytes() -> int:
    """Tier-2 RUN-gate reserve term (provisional policy): the d01 RUN
    fixture's pool-held bytes beyond the d01 alloc-estimate basis
    (measured used-peak + default-chunk workspace, with headroom).
    Calibrated, clamped at zero: if the enumerated workspace already
    over-covers the fixture's retention, nothing extra is reserved (no
    double count).  The N0 probe measured allocation-time retention NIL,
    so this term belongs to the N5/N6 run gates, not the N0 alloc gate
    (:meth:`ReservePolicy.run_time` vs :meth:`ReservePolicy.n0_alloc` --
    split pending controller ratification)."""
    meta_free_basis = math.ceil(ALLOCATOR_HEADROOM * (
        CAL_D01_POOL_USED_PEAK_BYTES + CAL_D01_WORKSPACE_BYTES))
    return max(0, CAL_D01_POOL_HELD_BYTES - meta_free_basis)


def _workspace_total_bytes(nz: int, column_chunk: int,
                           p_top: float = 5000.0) -> int:
    return sum(math.prod(shape) * size for shape, size in
               rrtmgp_workspace_shapes(nz, column_chunk, p_top).values())


def estimate_devices(exp, **kwargs):
    """The resident per-card rank admission, imported only for a split."""
    from woof.core.devices_memory import estimate_devices as estimate
    return estimate(exp, **kwargs)


@lru_cache(maxsize=2)
def thompson_coefficient_bytes(aerosol: bool = False) -> int:
    """Exact immutable device-table allocations, without loading an asset.

    Both Thompson variants upload the complete classic float64 contract.
    The aerosol variant adds CCN activation and the small derived arrays;
    its droplet-evaporation table aliases the classic owner and adds zero.
    Round each allocation to the device pool's 512-byte quantum.
    """
    from woof.core.thompson_contract import (
        AUXILIARY_TABLE_RECORDS, GENERATED_TABLE_FILES)

    records = tuple(record for group in GENERATED_TABLE_FILES.values()
                    for record in group) + AUXILIARY_TABLE_RECORDS
    sizes = [record.payload_bytes for record in records]
    if aerosol:
        from woof.core.thompson_aerosol_contract import (
            CCN_ACTIVATION_VALUES, derived_constant_arrays)
        sizes.append(CCN_ACTIVATION_VALUES * 8)
        sizes.extend(array.size * 8 for array in derived_constant_arrays().values())
    return sum(((int(size) + 511) // 512) * 512 for size in sizes)


def thompson_generation_coefficient_bytes(cfg) -> int:
    """Immutable device allocations of the selected statement generation."""
    from woof.thompson_generations import CLASSIC_GENERATIONS
    if int(cfg.mp_physics) not in (8, 28):
        return 0
    if int(cfg.mp_physics) == 28 or CLASSIC_GENERATIONS[cfg.thompson_version].backend != "column":
        return thompson_coefficient_bytes(int(cfg.mp_physics) == 28)
    from woof.core.thompson_column import TABLE_ARGUMENTS
    from woof.core.thompson_contract import AUXILIARY_TABLE_RECORDS, GENERATED_TABLE_FILES, TableRecord
    records = tuple(TableRecord(n, (37,37,37,37)) for n in TABLE_ARGUMENTS[:5])
    records += tuple(r for name in ("qr_acr_qsV2.dat", "freezeH2O.dat") for r in GENERATED_TABLE_FILES[name])
    records += AUXILIARY_TABLE_RECORDS[:5]
    return sum(((r.payload_bytes + 511) // 512) * 512 for r in records)


def experiment_thompson_coefficient_bytes(exp) -> int:
    """Each resident generation owns its tables once, across all domains."""
    owners = {}
    for dc in exp.domains:
        cfg = dc.run
        if int(cfg.mp_physics) not in (8, 28):
            continue
        key = "legacy" if int(cfg.mp_physics) == 28 or cfg.thompson_version == "wrf_461" else cfg.thompson_version
        owners[key] = max(owners.get(key, 0), thompson_generation_coefficient_bytes(cfg))
    return sum(owners.values())


def estimate_experiment(
        exp: ExperimentConfig, *,
        column_chunk: int | None = None,
        forcing_interval_seconds: float = DEFAULT_FORCING_INTERVAL_SECONDS,
        forcing_intervals: int | None = None,
        lateral_boundaries=None,
        vram_gib: float | None = None,
        profile: DeviceLocalMemoryProfile | None = None,
        boundary_species=(),
        urban_columns=None,
        tile_buffer: bool = False,
        lake_columns=None,
        pool_residue_cap_bytes: int | None = None,
) -> ExperimentMemoryEstimate:
    """Sum the per-domain itemizations; count the lru_cache-shared
    k-distribution tables ONCE (rrtmgp.py:324/:436 -- baseline behavior,
    never claimed as savings); replace audited scratch with its per-slot max
    for multi-domain experiments; size the ONE explicitly allocated and
    adapter-consumed shared chunk workspace; apply the 15% allocator-headroom
    factor. The result is compared against the
    MEASURED budget (free VRAM at startup minus the configured reserve --
    never nominal 32 GiB).

    ``tile_buffer`` prices the MYNN workspace a prepared tile or device
    rank actually allocates. The loader's retained template is a resident
    domain and keeps the default False.

    ``urban_columns`` maps grid id to the domain's urban column count, the
    prepared doors' reading of their land cover
    (:func:`woof.core.urban_state.prepared_urban_columns`); a domain it
    does not name is priced at every column urban (A176). ``tile_buffer``
    prices the workspace policy used by ``prepared_tile_state_factory``;
    permanently resident device ranks use that factory too.

    ``lake_columns`` maps grid id to an upper bound on the domain's CLM
    lake columns (a device rank's count of lake-eligible columns in its
    compute window, :func:`woof.core.devices_memory.rank_lake_column_bounds`);
    a domain it does not name prices the sparse lake work arrays at every
    column, the configuration door's bound.

    ``pool_residue_cap_bytes`` bounds the pool margin in bytes
    (:func:`forecast_pool_estimate_bytes`); the ranked road passes
    :data:`RANK_POOL_RESIDUE_BOUND_BYTES` for its device ranks (A13)."""
    column_chunk = (exp.column_chunk if column_chunk is None
                    else column_chunk)
    if (isinstance(column_chunk, bool)
            or not isinstance(column_chunk, int)
            or column_chunk < 1):
        raise ValueError(
            "column_chunk must be a positive integer number of radiation "
            f"columns, got {column_chunk!r}.")
    n_int = lbc_intervals(exp.run_seconds, forcing_interval_seconds,
                          retained_intervals=forcing_intervals)
    by_id = {dc.grid_id: dc for dc in exp.domains}
    from woof.core.cam_ozone import cam_ozone_domain_ids
    ozone_domains = cam_ozone_domain_ids(exp)
    from woof.core.uh_diag import declared_follower_slots
    follower_slots = declared_follower_slots(exp.domains)
    domains = tuple(
        estimate_domain(
            dc, spec_bdy_width=exp.spec_bdy_width,
            cam_ozone=dc.grid_id in ozone_domains,
            follower_slots=follower_slots.get(int(dc.grid_id), ()),
            cfl_recording=bool(exp.root.run.use_adaptive_time_step),
            parent=(None if dc.parent_id == 0 else by_id[dc.parent_id]),
            n_lbc_intervals=n_int,
            lateral_boundaries=(lateral_boundaries if dc.parent_id == 0 else None),
            boundary_species=(boundary_species if dc.parent_id == 0 else ()),
            p_top=exp.vertical.p_top,
            column_chunk=column_chunk,
            tile_buffer=tile_buffer,
            urban_columns=(urban_columns or {}).get(int(dc.grid_id)),
            lake_columns=(lake_columns or {}).get(int(dc.grid_id)))
        for dc in exp.domains)
    from woof.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
    variants = {rrtmg_variant(dc.run) for dc in exp.domains
                if 4 in radiation_scheme_ids(dc.run)}
    uses_rrtmgp = any(value != RRTMG_VARIANT_LEGACY for value in variants)
    uses_legacy = RRTMG_VARIANT_LEGACY in variants
    # The device the non-pool terms are priced against: the caller's
    # profile, else the reference profile for the (absent or undeclared)
    # card.  A Noah-MP configuration adds one step to that default: its
    # frames are readings of a compile platform, so with no profile and
    # no declared card it reads this machine's card first, to price from
    # that card's own row when it has one (_noahmp_pricing_profile).  It
    # never prices the requested card on another card's geometry; a card
    # with no row is priced from the Noah-MP ceiling, and the basis says so.
    device_profile = _noahmp_pricing_profile(
        exp, profile, declared_card=vram_gib is not None)
    if device_profile is None:
        device_profile = card_local_memory_profile(vram_gib)
    legacy_calls = ()
    legacy_envelope = 0
    if uses_legacy:
        from woof.core.rrtmg_legacy import legacy_radiation_vram_bytes
        from woof.core.chem_radiation import feedback_row_count
        # Modern tables/workspace persist while another domain executes legacy
        # radiation. Per-domain call peaks combine with that domain's other
        # transients; unrelated domains' transient maxima are not summed.
        # Price the workspace from the same device profile as the non-pool
        # terms. A live CuPy query here both mixed devices' assumptions and
        # left a CUDA context in a process promising CPU-only estimation.
        #
        # THE RESOLVED PROFILE, not the caller's raw one, and the
        # difference is a shipped refusal.  This line used to pass
        # ``resident_threads=0`` whenever the caller handed no profile,
        # which rrtmg_sw.sw_batch_column_chunk reads as "no device" and
        # answers with the fixed no-device width.  That was the same
        # answer as the reference profile's until c5f942ad128 (2026-09-15)
        # retired the 2,048-column ceiling and made the width the device's
        # own saturation width: on the reference profile the shortwave
        # chunk is 2,560 columns, and the no-device width stayed where the
        # retired ceiling left it, 2,048.  So one command priced one file
        # twice.
        #
        # READ at this tree rather than carried: sw_batch_column_chunk
        # answers 2,048 at resident_threads=0 and 2,560 at this profile's
        # 261,120, at every layer count from 30 to 128, because host-side
        # pricing passes no free_bytes and the width is then the
        # saturation width alone.  The 3,328 columns this comment used to
        # name is the saturation width of a 170-SM part at 2,048 threads
        # per SM (tests/test_rrtmg_chunk_narrow.py, ONE_SEVENTY_SM_2048);
        # MEASURED_LOCAL_MEMORY_PROFILE declares 1,536, so it is not this
        # profile's width and the gap this line closes is 2,048 against
        # 2,560.
        # Measured at 674133103, `woof domain --point=39.7,-96.6 --card
        # 16gb --ladder 12-3 --source hrrr`: the wizard sized the ladder at
        # a 2.89 GiB workspace (13.73 GiB envelope, inside the 14.54 GiB
        # budget) and the check it runs on its own output priced the same
        # file at 3.62 GiB (14.72 GiB envelope) and refused it, exit 4,
        # because `woof check` fills an absent profile from
        # card_local_memory_profile and this line did not.  A declared card
        # is not a measured one, and the conservative reference intercept
        # is what every other term in this function already gives it.
        legacy_calls = tuple(
            legacy_radiation_vram_bytes(
                ncol=dc.run.ny * dc.run.nx, nz=dc.run.nz,
                p_top=exp.vertical.p_top, column_chunk=None,
                longwave=radiation_scheme_ids(dc.run)[0] == 4,
                shortwave=radiation_scheme_ids(dc.run)[1] == 4,
                resident_threads=device_profile.resident_thread_capacity,
                aer_opt=int(getattr(dc.run, "aer_opt", 0)),
                chem_optics_rows=feedback_row_count(dc.run))
            if (4 in radiation_scheme_ids(dc.run)
                and rrtmg_variant(dc.run) == RRTMG_VARIANT_LEGACY) else 0
            for dc in exp.domains)
        legacy_envelope = max(legacy_calls, default=0)
    uses_arena = len(exp.domains) > 1
    arena_shapes = (shared_scratch_arena_shapes(exp.domains)
                    if uses_arena else {})
    nz = exp.domains[0].run.nz
    from woof.hydro.memory import forecast_plan
    hydro_plan = forecast_plan(exp)
    return ExperimentMemoryEstimate(
        hydro_memory=None if hydro_plan is None else hydro_plan.inventory(),
        domains=domains,
        k_tables_bytes=k_distribution_bytes() if uses_rrtmgp else 0,
        workspace_bytes=(_workspace_total_bytes(
            nz, column_chunk, exp.vertical.p_top)
                          if uses_rrtmgp else legacy_envelope),
        scratch_arena_bytes=(shared_scratch_arena_bytes(exp.domains)
                             if uses_arena else 0),
        uses_shared_scratch_arena=uses_arena,
        dycore_state_workspace_bytes=(
            shared_dycore_state_workspace_bytes(exp.domains)
            if uses_arena else 0),
        uses_shared_dycore_state_workspace=uses_arena,
        column_chunk=column_chunk,
        headroom=forecast_pool_headroom(dc.run for dc in exp.domains),
        # A suite an itemization-gap row raised the headroom for keeps
        # the ratio whatever the caller passed: its shortfall is pool
        # USED beyond the subtotal, proportional to the rank, which the
        # byte bound would under-price (rank_pool_residue_cap_bytes).
        pool_residue_cap_bytes=(
            None if pool_residue_cap_bytes is None
            or rank_pool_residue_cap_bytes(
                dc.run for dc in exp.domains) is None
            else int(pool_residue_cap_bytes)),
        retention_residual_bytes=(
            platform_projection_constants(vram_gib=vram_gib)[0]
            if uses_rrtmgp else 0),
        device_overhead_bytes=platform_projection_constants(
            vram_gib=vram_gib)[1],
        non_pool_device_bytes=non_pool_device_bytes(exp, profile=device_profile),
        envelope_family=envelope_platform(vram_gib=vram_gib),
        uses_legacy_radiation=uses_legacy,
        legacy_call_peak_by_domain=legacy_calls if uses_rrtmgp else (),
        local_memory_profile=device_profile,
        retained_forcing_intervals=n_int,
        column_workspace_bytes=column_workspace_bytes(
            exp, profile=(card_local_memory_profile(vram_gib)
                          if profile is None else profile)),
        physics_tables_bytes=experiment_thompson_coefficient_bytes(exp),
    )


def admission_estimate(exp: ExperimentConfig, *, machine=None, source=None,
                       urban_columns=None,
                       mynn_column_chunk: int | None = None
                       ) -> ExperimentMemoryEstimate:
    """THE estimate a tree's ``[tiles]`` admission is judged from.

    ONE function, called with the same arguments wherever the question is
    asked, because the question has one answer.  The plan review
    (:func:`estimate_phases`, through
    :func:`woof.core.streaming.tree_road_plan`) and the run door
    (:func:`woof.core.streaming.cold_tree_streaming_decision`)
    used to price it from two different calls: the review with an explicit
    ``column_chunk`` and the target card's profile, the door with the
    prepared cache's retained forcing interval count and its real lateral
    boundaries and no profile at all.  MEASURED on the 12/3 km moving-nest
    cyclone tree, those two inputs move the envelope by 60,193,971 and by
    up to 432,788,799 bytes, so for any budget in between the review
    admitted the tree resident and the door then raised ``StreamingRefused``
    AFTER authority, fetch, manifest and prepare had run -- a refusal the
    user paid for twice over and could have had before the download.

    So the admission is priced from the CONFIGURATION and the machine, and
    from nothing a cold surface cannot see: ``column_chunk`` is the
    experiment's own, the forcing schedule is the one the configuration
    declares, and no lateral boundary SET is folded in.  The run keeps
    its own, richer estimate for its memory LEDGER -- that one has the
    cache's real interval count and belongs to a different question -- and
    the two are never compared against each other.

    THE MACHINE IS THE ONLY DEVICE TERM, AND THERE IS NO SECOND WAY IN.
    This used to take a ``profile`` as well, defaulting to the machine's,
    and that optional second way in was a third estimate rather than a
    convenience: the review passed its own profile while the door passed
    none and took ``machine.device_profile``, so the very argument
    asymmetry this function was written to close reopened one field
    lower.  MEASURED on one card, the same tree and the same budget:
    4,009,919,677 bytes on a detect-built profile, 4,271,801,533 on a
    probe-built one carrying the measured bare context and compile
    platform, 5,141,378,237 on none -- and for any budget between two of
    those, one side admits what the other refuses.  So the device comes
    from the machine, every caller passes the machine its own surface
    already holds (:func:`woof.core.streaming.planner_machine` now
    carries the profile for the review surfaces, and
    :meth:`tilestream.autoplan.Machine.detect` has always carried it for
    the door), and the same card gives the same envelope on both sides.

    THE SOURCE IS PART OF THE CONFIGURATION'S TABLES.  The root's forcing
    tables at the declared schedule are priced, and they carry the
    analysed hydrometeors the forcing source publishes
    (:func:`woof.boundary_fields.source_boundary_species`, read off its
    table row, off the mapping document a mapped route passes as
    ``source``, or off the masses a prepared door's cache carries, which
    that door passes instead of its source's name).  Priced without it, a
    HRRR-forced tiled, streamed or resident-admitted run was weighed
    against the card short by those tables while :func:`estimate_phases`
    priced them, so the review and the door answered two different
    envelopes again.  ``None`` (a source
    the caller does not know, or one that publishes none) prices water
    vapour only, which is what such a run holds.

    THE LAND COVER IS THE ONE PREPARED READING.  ``urban_columns`` (grid
    id to urban columns, :func:`woof.core.urban_state
    .prepared_urban_columns`) is what a prepared door reads off the static
    it is about to restore, and it prices BEP+BEM's column workspace at
    the plan the run builds.  The configuration has no land cover, so the
    review prices every column urban, the upper bound: a prepared door can
    admit what the review refused, never the other way (A176).

    THE MEASURED POOL RESIDUE.  A one-domain forecast of a suite with
    rank receipts carries the A13 pool residue bound
    (:func:`single_card_pool_residue_cap_bytes`), as each device rank
    does.  ``mynn_column_chunk`` prices the MYNN workspace at that width
    instead of the width the run walks; only the resident road's width
    fit asks for it (:func:`required_mynn_pricing`).
    """
    from woof.boundary_fields import source_boundary_species
    with required_mynn_pricing(exp, mynn_column_chunk, only_given=True):
        return estimate_experiment(
            exp, column_chunk=exp.column_chunk,
            profile=getattr(machine, "device_profile", None),
            boundary_species=source_boundary_species(source),
            urban_columns=urban_columns, **single_card_residue_terms(exp))


# ---------------------------------------------------------------------------
# Reserve policy + gate evaluation (F11 chain, exact comparisons)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReservePolicy:
    """Configured reserve: what the MEASURED budget subtracts from free
    VRAM.  Two proposals, split by gate, PENDING CONTROLLER RATIFICATION
    at N0 (``--reserve-gib`` overrides with a flat value):

    * :meth:`n0_alloc` -- the allocation-gate reserve: the probe-measured
      fresh-process non-pool overhead (1.39 GiB) + external margin;
      alloc-time pool retention was measured nil, so no retention term.
    * :meth:`run_time` -- the N5/N6 run-gate reserve: adds the
      fixture-calibrated run-churn retention residual.  The probe
      overhead is a lower bound at run time (JIT module loads), which
      the external margin partially covers -- flagged, not resolved.
    """

    retention_residual_bytes: int
    device_overhead_bytes: int
    external_margin_bytes: int = EXTERNAL_MARGIN_BYTES

    @classmethod
    def n0_alloc(cls, exp: ExperimentConfig | None = None, *,
                 profile: DeviceLocalMemoryProfile | None = None,
                 estimate_bytes: int | None = None) -> "ReservePolicy":
        """Allocation-gate reserve.

        With an experiment, ``device_overhead_bytes`` is the MEASURED
        non-pool residency of a process running THAT configuration -- CUDA
        context plus the local-memory backing store its widest launched
        kernel reserves (:func:`non_pool_device_bytes`).  Without one it
        falls back to the 2026-07-16 zero-step probe constant, which is the
        number that was 5.5 GiB short of every real run.

        With ``estimate_bytes``, the retention term carries
        :data:`POOL_RESERVED_OVER_ESTIMATE_FRACTION` -- the measured gap
        between what the pool hands out and what it holds onto.
        """
        overhead = (PROBE_DEVICE_OVERHEAD_BYTES if exp is None
                    else non_pool_device_bytes(exp, profile=profile))
        retention = (0 if estimate_bytes is None else math.ceil(
            POOL_RESERVED_OVER_ESTIMATE_FRACTION * int(estimate_bytes)))
        return cls(retention_residual_bytes=retention,
                   device_overhead_bytes=overhead)

    @classmethod
    def run_time(cls, exp: ExperimentConfig | None = None, *,
                 profile: DeviceLocalMemoryProfile | None = None
                 ) -> "ReservePolicy":
        overhead = (PROBE_DEVICE_OVERHEAD_BYTES if exp is None
                    else non_pool_device_bytes(exp, profile=profile))
        return cls(retention_residual_bytes=pool_retention_residual_bytes(),
                   device_overhead_bytes=overhead)

    @classmethod
    def flat(cls, reserve_bytes: int) -> "ReservePolicy":
        return cls(retention_residual_bytes=0, device_overhead_bytes=0,
                   external_margin_bytes=int(reserve_bytes))

    @property
    def reserve_bytes(self) -> int:
        return (self.retention_residual_bytes + self.device_overhead_bytes
                + self.external_margin_bytes)

    def budget_bytes(self, measured_free_bytes: int) -> int:
        return int(measured_free_bytes) - self.reserve_bytes


def evaluate_alloc_gates(*, measured_used_bytes: int | None,
                         estimate_bytes: int,
                         measured_free_bytes: int | None,
                         reserve: ReservePolicy) -> dict[str, bool | None]:
    """The F11 enforced chain, evaluated EXACTLY (no tolerance), keyed by
    the pre-registered N0 ledger records (verify/nest_gates.py).  ``None``
    marks a leg whose measurement is unavailable (estimator-only mode);
    a missing measurement can never report a pass."""
    budget = (None if measured_free_bytes is None
              else reserve.budget_bytes(measured_free_bytes))
    return {
        "alloc_fits_wddm_budget": (
            None if (measured_used_bytes is None or budget is None)
            else measured_used_bytes <= budget),
        "alloc_measured_le_estimate": (
            None if measured_used_bytes is None
            else measured_used_bytes <= estimate_bytes),
        "alloc_estimate_le_wddm_budget": (
            None if budget is None else estimate_bytes <= budget),
    }


def device_wide_used_bytes(*, device_id: str | None = None) -> int:
    """Device memory held by EVERY process on the card, from NVML.

    ``cudaMemGetInfo`` is not a substitute here.  On this WDDM host it
    reported 1,614 MiB used at a moment NVML reported 3,903 MiB -- it does
    not see the desktop compositor's allocations, so a budget built from its
    ``free`` over-states the card by ~2.3 GiB.  Its DELTAS are exact (a
    traced run's memGetInfo-derived process peak agreed with the NVML
    device-wide series to 27 MiB), which is why the estimator uses it for
    growth and NVML for the whole-machine rail.

    Fails closed through ``supervisor.GPUPreflightError`` when nvidia-smi is
    unavailable: a rail that cannot be measured must never read as headroom.
    """
    from woof.supervisor import _run_nvidia_smi

    arguments = ["--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if device_id is not None:
        arguments.append(f"--id={device_id}")
    text = _run_nvidia_smi(arguments)
    return int(text.strip().splitlines()[0]) * 1024 ** 2


def device_physical_total_bytes(*, device_id: str | None = None) -> int | None:
    """Physical VRAM total of the card, from NVML, or None if unreadable.

    NVML and not ``cudaMemGetInfo``: capacity is the one device question
    that must be answerable in CPU mode, and ``memGetInfo`` cannot be
    asked without standing up a CUDA context -- which is exactly what an
    estimator-mode preflight promises not to do.  ``device_wide_used_bytes``
    already argues NVML is the authority on this host anyway.

    Unreadable is None, never a number: this figure is used as a CEILING
    only, so a card that cannot be read simply imposes no ceiling.
    """
    try:
        from woof.supervisor import _run_nvidia_smi

        arguments = ["--query-gpu=memory.total", "--format=csv,noheader,nounits"]
        if device_id is not None:
            arguments.append(f"--id={device_id}")
        text = _run_nvidia_smi(arguments)
        total = int(text.strip().splitlines()[0]) * 1024 ** 2
    except Exception:
        return None
    return total if total > 0 else None


def cap_free_to_device_wide(free_bytes: int, *, device_id: str | None = None
                            ) -> tuple[int, bool]:
    """Cap CUDA's eviction-inclusive free figure by the same device's NVML free.

    An unreadable additional ceiling does not manufacture capacity or replace
    the successful CUDA observation. Supplied/declared Machine values never
    call this helper. The optional PCI/UUID selector follows the CUDA device.
    """
    free = int(free_bytes)
    arguments = {} if device_id is None else {"device_id": device_id}
    try:
        total = device_physical_total_bytes(**arguments)
        used = device_wide_used_bytes(**arguments)
    except Exception:
        return free, False
    if total is None or used is None:
        return free, False
    capped = min(free, max(0, int(total) - int(used)))
    return capped, capped < free


def release_unreachable_device_memory(array_module=None) -> None:
    """Hand back the device memory nothing in this process can reach.

    Call it immediately before a card reading that decides something (an
    admission, a pack, a tile plan).  Two kinds of bytes are this process's
    own and free in every sense that matters, yet ``cudaMemGetInfo`` counts
    them as used:

    * arrays reachable only through a reference cycle.  The driver/state
      attachment of a finished forecast is such a cycle, and its arrays stay
      allocated until the cyclic collector happens to run; when it runs
      depends on everything the process allocated before, not on the run
      being admitted.
    * blocks the default CuPy pool holds unused after earlier work.

    So a reading taken without this depended on what ran earlier in the
    process.  MEASURED (RTX 5090 held to 15 GiB free, the release gate's
    one-process order): after the two member-sources ensemble identity
    tests, the native decline gate read 2.87 GiB free and its first member
    was refused (it needs 4.77 GiB); with one collection between the tests
    and nothing else changed it read 13.3 GB and passed.  The C6 skeptic
    saw the same reading admit the gate's four-member native pack as two
    packs.  The same garbage is reachable in production wherever a process
    runs a second forecast or ensemble after a first.

    Live arrays are untouched (``free_all_blocks`` releases unreferenced
    blocks only), so this has no numerical effect.  It is a whole-process
    collection, which is why it belongs at the few decision readings and
    never on a per-step path.  A process with no reachable card has nothing
    to hand back and returns quietly.
    """
    import gc

    gc.collect()
    if array_module is None:
        try:
            import cupy as array_module
        except Exception:                         # noqa: BLE001 - no CuPy
            return
    try:
        array_module.cuda.get_current_stream().synchronize()
        array_module.get_default_memory_pool().free_all_blocks()
    except Exception:                             # noqa: BLE001 - no card
        return


def device_free_and_total_bytes(device: int | None = None) -> tuple[int, int]:
    """``(free, total)`` for one CUDA device, free as the tiling planner reads it.

    ``cudaMemGetInfo`` capped by the same device's NVML free
    (:func:`cap_free_to_device_wide`).  One function for every reader of
    "free on the card right now", because two instruments on one card
    printed two answers in one run: under Windows WDDM ``memGetInfo``
    counts memory obtainable only by evicting other processes, so the
    streamed forecast's init line said 8.88 GiB free of 10.00 GiB while
    the planner (``tilestream.autoplan.Machine.detect``) measured 3.99 to
    4.32 GiB, with about 6 GiB held by other programs.  On Linux the two
    agree and the cap is a no-op.

    The reading is taken after :func:`release_unreachable_device_memory`,
    so it does not depend on what this process ran before.

    ``device`` None reads the current device, as ``cupy.cuda.Device()``
    does.
    """

    import cupy as cp

    selected = cp.cuda.Device() if device is None else cp.cuda.Device(device)
    with selected:
        release_unreachable_device_memory(cp)
        free, total = cp.cuda.runtime.memGetInfo()
    free, _ = cap_free_to_device_wide(free, device_id=selected.pci_bus_id)
    return int(free), int(total)


def cap_free_to_physical(free_bytes: int, *,
                         card_total_bytes: int | None,
                         measured_total_bytes: int | None
                         ) -> tuple[int, int | None]:
    """Clamp a free-VRAM figure to what the card physically has.

    A DERIVED free figure is arithmetic (declared budget + reserve) and
    nothing in that arithmetic knows how big the card is: the ``--card
    16gb`` wizard tier produced a notional free of 16.68 GiB on a card
    whose physical total is 15.57 GiB, which then bought the estimate a
    gigabyte of budget that does not exist.  Free can never exceed total,
    on any card, so the smaller of the two capacity statements we have --
    the caller's declared card size and a live measurement of it -- is an
    ADDITIONAL ceiling, in the same idiom as the device rail: never a
    replacement and never a widening.

    Returns ``(free, cap)`` where ``cap`` is the ceiling that actually
    bound, or None when the figure was already within capacity.
    """
    caps = [int(c) for c in (card_total_bytes, measured_total_bytes)
            if c is not None and int(c) > 0]
    if not caps:
        return int(free_bytes), None
    cap = min(caps)
    if int(free_bytes) <= cap:
        return int(free_bytes), None
    return cap, cap


def device_rail_free_bytes(rail_bytes: int, *,
                           other_process_bytes: int) -> int:
    """Bytes this process may hold before the WHOLE CARD crosses the rail.

    The rail is a property of the host, not of the model: this box has no
    ECC and has already corrupted fine-domain output near capacity, so the
    bar is on total device residency, and every other process on the card --
    a 3.4 GiB desktop, here -- spends it.
    """
    return int(rail_bytes) - int(other_process_bytes)


def recommend_column_chunk(exp: ExperimentConfig, budget_bytes: int, *,
                           start_chunk: int | None = None,
                           floor: int = 1) -> int | None:
    """FIRST over-budget lever (robust-5 / architecture section E): the
    largest halving of ``start_chunk`` (down to ``floor``) whose estimate
    fits the budget, or None if none fits."""
    chunk = int(exp.column_chunk if start_chunk is None else start_chunk)
    while chunk >= floor:
        estimate = estimate_experiment(exp, column_chunk=chunk)
        if estimate.alloc_estimate_bytes <= budget_bytes:
            return chunk
        chunk //= 2
    return None


# ---------------------------------------------------------------------------
# N0 --alloc runner (GPU; controller-run)
# ---------------------------------------------------------------------------

class PreflightHeadroomError(RuntimeError):
    """Free VRAM is short of the remaining estimate BEFORE construction.

    Carries structured fields so ``check_main`` can still emit the full
    JSON report (estimate-side legs evaluated, abort reason recorded)
    with an exit code distinct from a gate-leg FAIL.
    """

    def __init__(self, message: str, *, phase: str, free_bytes: int,
                 total_bytes: int, reserve_bytes: int,
                 remaining_bytes: int):
        super().__init__(message)
        self.phase = phase
        self.free_bytes = int(free_bytes)
        self.total_bytes = int(total_bytes)
        self.reserve_bytes = int(reserve_bytes)
        self.remaining_bytes = int(remaining_bytes)


class PreflightAllocError(RuntimeError):
    """Device allocation failed during --alloc; diagnostics attached."""

    def __init__(self, message: str, *, phase: str,
                 free_bytes: int | None = None):
        super().__init__(message)
        self.phase = phase
        self.free_bytes = None if free_bytes is None else int(free_bytes)


@dataclass
class AllocReport:
    """Measured N0 numbers + gate legs from one --alloc run."""

    estimate: ExperimentMemoryEstimate
    reserve: ReservePolicy
    free_before_bytes: int
    total_bytes: int
    pool_used_peak_bytes: int
    pool_held_peak_bytes: int
    free_at_peak_bytes: int
    free_after_release_bytes: int
    gates: dict[str, bool | None]

    @property
    def device_footprint_bytes(self) -> int:
        return self.free_before_bytes - self.free_at_peak_bytes

    @property
    def measured_overhead_bytes(self) -> int:
        """Re-calibrated tier-3 term: measured footprint minus pool held."""
        return self.device_footprint_bytes - self.pool_held_peak_bytes

    @property
    def passed(self) -> bool:
        return all(leg is True for leg in self.gates.values())


def _require_headroom(cp, remaining_bytes: int, reserve: ReservePolicy,
                      phase: str) -> None:
    """Robust-5: headroom check before state construction and before each
    high-water phase.  Fails loudly BEFORE the allocation that would OOM."""
    free, total = cp.cuda.runtime.memGetInfo()
    if free - reserve.reserve_bytes < remaining_bytes:
        raise PreflightHeadroomError(
            f"headroom check failed before {phase}: free {free / GIB:.2f} "
            f"GiB minus reserve {reserve.reserve_bytes / GIB:.2f} GiB is "
            f"short of the remaining itemized need "
            f"{remaining_bytes / GIB:.2f} GiB (device total "
            f"{total / GIB:.2f} GiB)",
            phase=phase, free_bytes=free, total_bytes=total,
            reserve_bytes=reserve.reserve_bytes,
            remaining_bytes=remaining_bytes)


def _synthetic_root_boundaries(cfg: RunConfig, n_intervals: int, *,
                               boundary_species=()):
    """Zero-valued LateralBoundaries with the exact the reference case table shapes,
    so attach_lateral_boundaries performs the REAL eager device upload.

    ``boundary_species`` is the source's published hydrometeor inventory
    (:func:`woof.boundary_fields.source_boundary_species`): the tables
    carry the masses and seeded numbers the run's root boundary carries,
    so the measurement is of the tables the run holds."""
    import numpy as np
    from woof.ingest.lateral_bc import build_lateral_boundaries

    snapshot = {name: np.zeros(dims, dtype=np.float64)
                for name, dims in _lbc_field_dims(
                    cfg, boundary_species=boundary_species).items()}
    # mu snapshots are 2-D in domain_boundary_snapshot; (1, ny, nx) works
    # identically through _field_boundary, but keep the exact contract.
    snapshot["mu"] = np.zeros((cfg.ny, cfg.nx), dtype=np.float64)
    times = [float(i) * DEFAULT_FORCING_INTERVAL_SECONDS
             for i in range(n_intervals + 1)]
    return build_lateral_boundaries(
        [snapshot] * (n_intervals + 1), times,
        spec_bdy_width=cfg.spec_bdy_width, spec_zone=cfg.spec_zone,
        relax_zone=cfg.relax_zone)


#: ``mp_physics`` values for which ``woof check --alloc`` advances the
#: driver's behavior-gating microphysics counter before it measures
#: residency.
#:
#: WHAT THIS SET DECIDES, which is not what its two values suggest.  Not
#: "which schemes are ported", not "which schemes own a canonical
#: accumulator row", not "which schemes predict radii".  It decides which
#: schemes reach an allocation a CONSTRUCTED driver does not hold but a
#: RUNNING one holds from its first microphysics call onward -- the only
#: thing ``PhysicsDriver.microphysics_updates`` gates that can move a byte
#: this measurement reports.  The tree has exactly one such allocation:
#: ``woof/core/rrtmgp.py:2535`` withholds the Morrison effective-radius
#: column pack (``effc``/``effr``/``effi``/``effs``, four ``(ncol, nz)``
#: float32 packs per radiation call) until the counter has accepted one
#: update, because until then ``state.eff*`` is zero-filled storage rather
#: than a PSD diagnostic.  Leave the counter at zero and ``--alloc``
#: measures a radiation call four column packs lighter than the run's.
#:
#: ``10`` is Morrison, the scheme that gate is written for.  ``6`` shares
#: the row because :func:`rrtmgp_column_shapes` prices its
#: ``effc``/``effi``/``effs`` pack off the same state arrays, so seeding
#: the counter states the same steady state for it; there the seeding is
#: faithful rather than essential, because the ``wsm6`` coupling arm
#: takes its radii unconditionally.
ALLOC_COUNTER_ADVANCED_MICROPHYSICS = (6, 10)

#: Selectors deliberately absent from
#: :data:`ALLOC_COUNTER_ADVANCED_MICROPHYSICS`, each with the reason it is
#: absent.  An omission is not a decision; this is where the decision is
#: written down.  Pinned by ``tests/test_preflight.py::
#: test_the_alloc_counter_advance_records_why_p3_is_out``.
ALLOC_COUNTER_INERT_MICROPHYSICS = {
    50: (
        "P3 one-category reaches NO reader of microphysics_updates on any "
        "radiation pairing woof.config admits, so advancing the counter "
        "would move zero measured bytes while putting mp=50 in a set whose "
        "membership means 'reaches a counter-gated allocation'.  "
        "(1) The counter's only allocation gate lives inside the RTE "
        "adapter's scheme == 'morrison' arm.  mp=50 DOES reach the "
        "adapter since the 2.6.1 cloud-optics coupling "
        "(_MP_CLOUD_OPTICS_SCHEME row 50: 'p3'; the old admission "
        "refusal, validate_p3_radiation, retired with the defect it "
        "guarded), but the p3 arm copies its two radii UNCONDITIONALLY "
        "-- it reads the counter nowhere, so there is still no "
        "counter-gated allocation for mp=50 to reach.  The two other "
        "admitted pairings (legacy RRTMG 4/4, Dudhia 0/1) never "
        "construct the adapter and price zero RTE+RRTMGP columns.  "
        "(2) The legacy RRTMG adapter -- P3's only 4/4 pairing -- reads "
        "the counter nowhere.  It takes its radii from WRF's has_req* "
        "table, and module_physics_init.F:1017 names P3_1CATEGORY in the "
        "use_mp_re disjunction while the :1027-1033 override sets "
        "has_reqs=0, so P3's radii are a scheme property rather than a "
        "first-call one.  "
        "(3) P3 has no 'radii not yet valid' phase for such a gate to "
        "express: woof/core/state.py:494-495 seeds state.effc/effi at "
        "10/25 microns at construction, which is exactly what p3_main "
        "presets on entry to every call before any condensate test "
        "(module_mp_p3.F:2279, :2281).  "
        "(4) P3's persistent set is already whole when _materialize_"
        "physics runs.  Its FIVE accumulator slots -- "
        "physics_inventory.microphysics_scratch_slots(50), no graupel, "
        "because module_microphysics_driver.F's CASE (P3_1CATEGORY) at "
        ":1557 binds RAINNC/RAINNCV/SR/SNOWNC/SNOWNCV and diag_effc_3d/"
        "diag_effi_3d and nothing else -- and its p3_* scratch slots are "
        "all in scratch_slot_registry, which run_alloc_preflight prewarms "
        "before this function is called.  "
        "WHAT WOULD FLIP THIS: a row for 50 in "
        "woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME, or any read of "
        "microphysics_updates appearing in woof/core/rrtmg_legacy.py.  "
        "The pinning test fails on either, rather than letting --alloc "
        "quietly understate a P3 run's radiation call."
    ),
}


def _materialize_physics(state, cfg: RunConfig, start_time: datetime,
                         *, glw=None):
    """initialize_physics + the steady-state extras the first steps would
    allocate (composed rqr/rqi/rqs, Morrison accumulator optionals, KF W0AVG),
    so the alloc proof covers the run's real persistent driver set.

    ``glw`` is the experiment's DECLARED constant downward longwave, or
    None: initialize_physics refuses to invent a GLW that something
    would consume or publish, so an alloc proof for a config that
    declared the constant (``constant-downward-longwave-v1``) must type
    the same declaration here that the run preparers type -- otherwise
    ``woof check --alloc`` false-refuses the very configs whose device
    footprint it exists to measure (the MYNN no-radiation d04 pair)."""
    import cupy as cp
    import numpy as np
    from woof.core.physics import (PBL_RQI_MICROPHYSICS, initialize_physics,
                                    physics_retains_ysu_output,
                                    physics_reuses_pbl_composition)
    from woof.core.state import DTYPE

    ny, nx = cfg.ny, cfg.nx
    grid = np.zeros((ny, nx), dtype=np.float64)
    driver = initialize_physics(
        state, cfg, landmask=1.0, tsk=290.0, soil_temperature=285.0,
        soil_moisture=0.30, glw=glw, radiation_start_time=start_time,
        radiation_latitude=grid, radiation_longitude=grid + 1.0)
    m = state.p.shape
    zero_m = lambda: cp.zeros(m, dtype=DTYPE)
    zero_s = lambda: cp.zeros((ny, nx), dtype=DTYPE)
    if cfg.cu_physics:
        target = (driver.pbl_tendencies
                  if physics_reuses_pbl_composition(cfg)
                  else driver.tendencies)
        for comp in ("rqr", "rqi", "rqs"):
            if getattr(driver.cumulus_tendencies, comp) is None:
                setattr(driver.cumulus_tendencies, comp, zero_m())
            if getattr(target, comp) is None:
                setattr(target, comp, zero_m())
        adapter = driver.cumulus_callable
        if (int(cfg.cu_physics) == 1 and adapter is not None
                and getattr(adapter, "w0avg", None) is None):
            adapter.w0avg = zero_m()          # kf.py:174 shape contract
            adapter._history_state = state
    if cfg.bl_pbl_physics and cfg.mp_physics in PBL_RQI_MICROPHYSICS:
        # The materialization side of the pbl_tendencies/rqi budget above.
        # 28 belongs for the same reason: Registry/Registry.EM_COMMON:3036
        # gives the thompsonaero package qi in moist, so WRF's F_QI is true;
        # 16 (wdm6scheme, :3031) declares the same moist inventory.
        # This set and physics._pbl_optional_tendency_components must agree,
        # or the --alloc measurement stops covering true runtime residency --
        # which is why it is now ONE named constant read by all three sites
        # (the shapes budget above, this materializer, and the physics
        # module), pinned equal by
        # tests/test_preflight.py::test_the_rqi_budget_shapes_materialization
        # _and_physics_name_one_set.
        if driver.pbl_tendencies.rqi is None:
            driver.pbl_tendencies.rqi = zero_m()
        if ((radiation_enabled(cfg) or cfg.cu_physics)
                and not physics_reuses_pbl_composition(cfg)):
            if driver.tendencies.rqi is None:
                driver.tendencies.rqi = zero_m()
    if cfg.mp_physics in ALLOC_COUNTER_ADVANCED_MICROPHYSICS:
        # PhysicsDriver initialization already aliases and materializes all
        # seven carrying mp_* scratch slots; only the behavior-gating counter
        # changes after the first real scheme call.  Which selectors belong
        # here -- and why mp=50 does not -- is
        # ALLOC_COUNTER_ADVANCED_MICROPHYSICS /
        # ALLOC_COUNTER_INERT_MICROPHYSICS above.
        driver.microphysics_updates = 1
    if physics_retains_ysu_output(cfg):
        # Materialize the positive-cadence retained YSU output set so the
        # --alloc measurement covers true runtime residency, not just
        # construction (review fix round): same shapes/dtypes as
        # launch_ysu's out dict (ysu.py:79-92), zero-filled.
        last_ysu = {
            name: (driver.pbl_raw_rates[name] if name in driver.pbl_raw_rates
                   else zero_m())
            for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi",
                         "exch_h", "exch_m")}
        for name in ("hpbl", "wstar", "delta", "topdown_radsum",
                     "wstar3_2"):
            last_ysu[name] = zero_s()
        for name in ("kpbl", "cloudflg"):
            last_ysu[name] = cp.zeros((ny, nx), dtype=cp.int32)
        driver.last_ysu = last_ysu
    return driver


def run_alloc_preflight(
        exp: ExperimentConfig, *,
        column_chunk: int | None = None,
        forcing_interval_seconds: float = DEFAULT_FORCING_INTERVAL_SECONDS,
        forcing_intervals: int | None = None,
        reserve: ReservePolicy | None = None,
        profile: DeviceLocalMemoryProfile | None = None,
        boundary_species=()) -> AllocReport:
    """N0: construct all DomainStates + drivers + d01 LBC + the F4 nest
    manifest allocations + the RRTMGP workspace on the real device, run
    ZERO steps, report pool used/peak vs estimate, free.

    ``boundary_species`` is the forcing source's published hydrometeor
    inventory: the d01 tables built and measured carry them, and the
    estimate beside the measurement prices the same tables.

    OOM policy (robust-5): any device allocation failure records the
    phase and pool/device diagnostics and TERMINATES the worker via
    :class:`PreflightAllocError` -- never ``free_all_blocks()`` and
    continue.
    """
    import cupy as cp

    from woof.core.state import (DTYPE, DomainState,
                                  build_shared_dycore_state_workspace,
                                  build_shared_scratch_arena)
    from woof.ingest.lateral_bc import attach_lateral_boundaries

    reserve = ReservePolicy.n0_alloc() if reserve is None else reserve
    # The one-card residue bound (A13) the estimate path already prices:
    # without it the measured leg compared a x1.13 estimate (95.75 GiB for
    # full-grid HRRR on a 96 GB card) against the budget while the plain
    # check and the go gate priced the bounded 86.98 GiB for the same run.
    estimate = estimate_experiment(
        exp, column_chunk=column_chunk, forcing_intervals=forcing_intervals,
        forcing_interval_seconds=forcing_interval_seconds, profile=profile,
        boundary_species=boundary_species, **single_card_residue_terms(exp))
    column_chunk = estimate.column_chunk
    n_int = lbc_intervals(exp.run_seconds, forcing_interval_seconds,
                          retained_intervals=forcing_intervals)

    pool = cp.get_default_memory_pool()
    free_before, total = cp.cuda.runtime.memGetInfo()
    # Growth from entry, not the pool's absolute level: the estimate prices
    # THIS run's allocations, and free_before/free_at_peak are already a
    # delta.  Breakage it prevents (2.9.4 R6 sweep, RTX 5070 Ti, 2026-10-22):
    # tests/test_preflight.py::test_alloc_preflight_d01_measured_le_estimate
    # failed in a pytest process whose earlier GPU test files still held
    # pool memory (alone: 1.88 GB measured <= 2.66 GB estimate, green),
    # because the absolute used level charged their arrays to this run.
    # A fresh `woof check --alloc` process starts at zero, so its numbers
    # are unchanged.
    used_base = pool.used_bytes()
    held_base = pool.total_bytes()
    used_peak = 0
    held_peak = 0
    free_at_peak = free_before

    def sample(phase: str) -> None:
        nonlocal used_peak, held_peak, free_at_peak
        cp.cuda.runtime.deviceSynchronize()
        used_peak = max(used_peak, pool.used_bytes() - used_base)
        held_peak = max(held_peak, pool.total_bytes() - held_base)
        free_at_peak = min(free_at_peak, cp.cuda.runtime.memGetInfo()[0])

    per_domain = {}
    for d in estimate.domains:
        resident = d.resident_bytes
        if estimate.uses_shared_scratch_arena:
            resident -= d.arena_scratch_bytes
        if estimate.uses_shared_dycore_state_workspace:
            resident -= d.rebuilt_state_bytes
        per_domain[d.grid_id] = resident
    remaining = (sum(per_domain.values()) + estimate.scratch_arena_bytes
                 + estimate.dycore_state_workspace_bytes
                 + estimate.workspace_bytes + estimate.k_tables_bytes)
    holdings: list[object] = []
    phase = "startup"
    arena = None
    dycore_state_workspace = None
    try:
        if estimate.uses_shared_dycore_state_workspace:
            phase = "shared dycore-state workspace"
            _require_headroom(cp, remaining, reserve, phase)
            dycore_state_workspace = build_shared_dycore_state_workspace(
                exp.domains)
            if (dycore_state_workspace.nbytes
                    != estimate.dycore_state_workspace_bytes):
                raise RuntimeError(
                    "runtime shared dycore-state workspace drifted from "
                    f"the estimator: {dycore_state_workspace.nbytes} != "
                    f"{estimate.dycore_state_workspace_bytes} bytes")
            holdings.append(dycore_state_workspace)
            sample(phase)
            remaining -= estimate.dycore_state_workspace_bytes

        if estimate.uses_shared_scratch_arena:
            phase = "shared transient-scratch arena"
            _require_headroom(cp, remaining, reserve, phase)
            arena = build_shared_scratch_arena(exp.domains)
            if arena.nbytes != estimate.scratch_arena_bytes:
                raise RuntimeError(
                    "runtime scratch arena drifted from the estimator: "
                    f"{arena.nbytes} != {estimate.scratch_arena_bytes} bytes")
            holdings.append(arena)
            sample(phase)
            remaining -= estimate.scratch_arena_bytes

        states: dict[int, DomainState] = {}
        by_id = {dc.grid_id: dc for dc in exp.domains}
        for dc in exp.domains:
            phase = f"domain d{dc.grid_id:02d} construction"
            _require_headroom(cp, remaining, reserve, phase)
            # Preserve the historical constructor call on the default/single-
            # domain path; only a multi-domain experiment receives workspaces.
            state_kwargs = {}
            if arena is not None:
                state_kwargs["scratch_arena"] = arena
            if dycore_state_workspace is not None:
                state_kwargs["dycore_state_workspace"] = (
                    dycore_state_workspace)
            state = DomainState(dc.run, **state_kwargs)
            states[dc.grid_id] = state
            # Prewarm every registry scratch slot (robust-5: persistent
            # scratch prewarmed at setup).  The root's packed forcing
            # tables come from the REAL attach below, not a prewarm.
            registry = scratch_slot_registry(dc.run, n_lbc_intervals=0)
            for slot, shape in registry.items():
                state.scratch(shape, slot)
            if dc.parent_id == 0 and dc.run.specified:
                attach_lateral_boundaries(
                    state, _synthetic_root_boundaries(
                        dc.run, n_int, boundary_species=boundary_species))
                # Davies weights are created on first force; prove their
                # bytes now under the exact resident slot name.
                state.scratch((2, dc.run.spec_bdy_width), "lbc_weights_0")
            if dc.parent_id != 0:
                shapes = nest_slot_shapes(
                    dc, exp.spec_bdy_width, by_id[dc.parent_id])
                dtypes = nest_slot_dtypes(
                    dc, exp.spec_bdy_width, by_id[dc.parent_id])
                for slot, shape in shapes.items():
                    state.scratch(shape, slot, dtype=dtypes[slot])
            from woof.core.physics_inventory import physics_driver_required
            if physics_driver_required(dc.run):
                # The experiment's declared constant GLW (or None),
                # exactly as prepare_real_case/prepare_child_case type
                # it: a config that legitimately declared
                # constant-downward-longwave-v1 must reach the device
                # here too, or --alloc refuses the very footprint
                # measurement it exists for.
                from woof.runtime import declared_constant_glw
                _materialize_physics(state, dc.run, exp.start_time,
                                     glw=declared_constant_glw(exp))
            sample(phase)
            remaining -= per_domain[dc.grid_id]

        phase = "rrtmgp k-tables + shared chunk workspace"
        _require_headroom(cp, remaining, reserve, phase)
        from woof.physics_compat import (RRTMG_VARIANT_LEGACY,
                                          rrtmg_variant)
        legacy_44 = [dc for dc in exp.domains
                     if 4 in radiation_scheme_ids(dc.run)
                     and rrtmg_variant(dc.run) == RRTMG_VARIANT_LEGACY]
        if legacy_44:
            # Legacy variant: hold the priced call-peak envelope (the
            # estimate's shared term) so the residency proof covers the
            # adapter's worst single call.
            phase = "legacy-RRTMG call-peak envelope"
            from woof.core.rrtmg_legacy import legacy_radiation_vram_bytes
            from woof.core.chem_radiation import feedback_row_count
            envelope = max(
                legacy_radiation_vram_bytes(
                    ncol=dc.run.ny * dc.run.nx, nz=dc.run.nz,
                    p_top=exp.vertical.p_top, column_chunk=None,
                    longwave=radiation_scheme_ids(dc.run)[0] == 4,
                    shortwave=radiation_scheme_ids(dc.run)[1] == 4,
                    aer_opt=int(getattr(dc.run, "aer_opt", 0)),
                    chem_optics_rows=feedback_row_count(dc.run))
                for dc in legacy_44)
            holdings.append(cp.zeros(envelope, dtype=cp.uint8))
        if any(4 in radiation_scheme_ids(dc.run)
               and rrtmg_variant(dc.run) != RRTMG_VARIANT_LEGACY
               for dc in exp.domains):
            from woof.core.rrtmgp import load_cloud_tables, load_gas_tables
            for kind in ("lw", "sw"):
                holdings.append(load_gas_tables(kind).to_device())
                holdings.append(load_cloud_tables(kind).to_device())
            nz = exp.domains[0].run.nz
            for name, (shape, size) in rrtmgp_workspace_shapes(
                    nz, column_chunk, exp.vertical.p_top).items():
                dtype = cp.bool_ if size == 1 else DTYPE
                holdings.append(cp.zeros(shape, dtype=dtype))
        if any(dc.run.cu_physics == 1 for dc in exp.domains):
            # The once-per-process KF device LUT (kf.py _device_table):
            # materialized so the residency proof covers it.
            from woof.core.kf import _device_table
            holdings.append(_device_table())
        sample(phase)

        gates = evaluate_alloc_gates(
            measured_used_bytes=used_peak,
            estimate_bytes=estimate.alloc_estimate_bytes,
            measured_free_bytes=free_before, reserve=reserve)
    except cp.cuda.memory.OutOfMemoryError as exc:
        free_now, _ = cp.cuda.runtime.memGetInfo()
        raise PreflightAllocError(
            f"--alloc OOM during {phase}: pool used "
            f"{pool.used_bytes() / GIB:.2f} GiB, held "
            f"{pool.total_bytes() / GIB:.2f} GiB, device free "
            f"{free_now / GIB:.2f} GiB of {total / GIB:.2f} GiB; itemized "
            f"estimate {estimate.alloc_estimate_bytes / GIB:.2f} GiB. "
            "Terminating (never free_all_blocks-and-continue); first "
            "lever: shrink the RRTMGP column_chunk.",
            phase=phase, free_bytes=free_now,
        ) from exc

    # Zero steps by contract.  Free everything and report the release.
    # The driver <-> state attachment is a reference cycle; collect it so
    # the arrays actually return to the pool before free_all_blocks.
    import gc
    holdings.clear()
    states.clear()
    # Loop locals otherwise keep the final state (and through it the shared
    # arena) alive past free_all_blocks().
    state = None
    arena = None
    dycore_state_workspace = None
    gc.collect()
    pool.free_all_blocks()
    cp.cuda.runtime.deviceSynchronize()
    free_after, _ = cp.cuda.runtime.memGetInfo()
    return AllocReport(
        estimate=estimate, reserve=reserve,
        free_before_bytes=int(free_before), total_bytes=int(total),
        pool_used_peak_bytes=int(used_peak),
        pool_held_peak_bytes=int(held_peak),
        free_at_peak_bytes=int(free_at_peak),
        free_after_release_bytes=int(free_after), gates=gates)


# ---------------------------------------------------------------------------
# CLI (`woof check`) -- registrar only; cli.py hookup is a controller
# handoff commit (F2 ownership map).
# ---------------------------------------------------------------------------

def _load_experiment_any(path: Path) -> ExperimentConfig:
    import io
    import tomllib

    from woof.case_data import load_experiment_case
    from woof.config import load_config
    from woof.config_authority import read_config_authority
    from woof.experiment import (build_experiment_from_config_tables,
                                  experiment_from_run_config,
                                  is_experiment_toml_bytes)

    authority = read_config_authority(path)
    if is_experiment_toml_bytes(authority.payload):
        raw = tomllib.load(io.BytesIO(authority.payload))
        if "case_data" not in raw:
            # `woof domain --source gfs|hrrr` deliberately emits no
            # [case_data]: those tables feed the native front door.  The
            # memory preflight needs only the experiment geometry, so the
            # companion tables ([fetch] hints, [static.highres]) are
            # validated by their owners and split off, as every file
            # loader does.  Splitting only [fetch] stopped `woof go` on
            # every such config that turned [static.highres] on.
            exp = build_experiment_from_config_tables(
                raw, source=str(path), base_dir=Path(authority.base_dir))
            return exp
        exp, _case_data = load_experiment_case(path)
        return exp
    from dataclasses import replace
    from woof.config import load_device_options
    return replace(experiment_from_run_config(load_config(path),
                                              datetime(1970, 1, 1)),
                   devices=load_device_options(path))


@dataclass(frozen=True)
class PreparedCheckInputs:
    """A check's forecast price, bound to the artifacts the runner restores."""

    experiment: ExperimentConfig
    prepared_root: Path
    source: str
    boundary_source: object
    forcing_interval_seconds: float
    forcing_intervals: int
    urban_columns: Mapping[int, int] | None
    prepared_head_sha256: str | None = None


def _prepared_check_inputs(args) -> PreparedCheckInputs | None:
    """Verify a sealed bundle or live head before using its land-cover price."""
    root = getattr(args, "prepared_root", None)
    if root is None:
        return None
    cached = getattr(args, "_prepared_check_inputs", None)
    if cached is not None:
        return cached
    from woof import stage_cli
    from woof import prepared_domain_tree_forecast as tree
    from woof import prepared_single_domain_forecast as single

    root = Path(root).resolve()
    config = Path(args.config).resolve()
    try:
        bundle = stage_cli.resolve_bundle(root)
    except stage_cli.StageRefusal:
        bundle = stage_cli.unsealed_head_bundle(root)
        if bundle is None:
            raise
    head_digest = bundle.get("head_sha256")
    prepared = Path(bundle.get("root", bundle["document"].parent))
    if bundle["layout"] == "tree":
        digests = stage_cli.tree_digests(bundle, config)
        inputs = tree.preflight_prepared_tree(
            prepared_root=prepared,
            **({"prepared_head_sha256": head_digest} if head_digest else
               {"preparation_receipt_sha256": digests["preparation_receipt"]}),
            experiment_config=config,
            experiment_config_sha256=digests["experiment_config"])
        reader = tree._root_bundle(inputs).cache_reader
        counts = tree.tree_urban_columns(inputs)
    else:
        wps = getattr(args, "wps_namelist", None)
        if wps is None:
            candidates = (config.with_suffix(".namelist.wps"),
                          config.parent / "namelist.wps")
            wps = next((path for path in candidates if path.is_file()), None)
        if wps is None:
            raise ValueError("a prepared single-domain check requires "
                             "--wps-namelist: the runner binds that namelist "
                             "to prevent restoring a different prepared grid")
        if head_digest:
            binding = {
                "prepared_head_sha256": head_digest,
                "source_manifest_sha256": bundle.get("source_manifest_sha256")}
        else:
            digests = stage_cli.single_domain_digests(bundle)
            binding = {
                "proof_sha256": digests["proof"],
                "source_manifest_sha256": digests["source_manifest"],
                "prepared_content_sha256": digests["prepared_content"]}
        exp = _load_experiment_any(config)
        inputs = single.preflight_prepared_forecast(
            source=bundle["source"], prepared_root=prepared,
            **binding,
            experiment_config=config, wps_namelist=Path(wps),
            physics_profile=None, run_seconds=float(exp.run_seconds),
            history_interval_seconds=float(exp.root.history_interval_s))
        reader = inputs.cache_reader
        counts = single.single_urban_columns(inputs)
    retained = 1 if head_digest else single._retained_interval_count(reader)
    if retained is None:
        raise ValueError("the prepared cache records no retained boundary "
                         "interval count, so its memory cannot be priced")
    boundary_source = single._priced_boundary_source(reader, inputs.source)
    if head_digest:
        from woof.boundary_fields import sealed_boundary_species

        inventory = inputs.stream_head["basis"]["cache"]["lbc"]
        carried = sealed_boundary_species([inventory])
        if carried is not None:
            boundary_source = carried
    result = PreparedCheckInputs(
        experiment=inputs.experiment, prepared_root=prepared,
        source=inputs.source,
        boundary_source=boundary_source,
        forcing_interval_seconds=float(inputs.boundary_interval_seconds),
        forcing_intervals=int(retained), urban_columns=counts,
        prepared_head_sha256=head_digest)
    args._prepared_check_inputs = result
    return result


def _check_bem_basis(estimate, prepared, *, alloc: bool = False) -> str | None:
    if prepared is None:
        return bem_workspace_bound_note(estimate)
    if alloc:
        note = bem_workspace_bound_note(estimate)
        return (None if note is None else
                "the allocation probe constructs synthetic all-urban land "
                "cover, so its BEP+BEM workspace keeps the every-column "
                "bound; omit --alloc to price the verified prepared land cover")
    from woof.core.urban_state import urban_columns_line

    return urban_columns_line(prepared.urban_columns)


def _recorded_source_forcing_interval(raw: dict) -> float | None:
    """The boundary cadence a config with no staged case still states.

    An emitted ``[fetch]`` table carries ``cadence`` only for the
    producers whose fetch takes the flag.  Every other one -- the ones
    retrieved as a whole window in a single call -- gets a table that
    names the producer and no cadence at all, and this read answered
    ``None`` for it.  The callers then priced the file at
    :data:`DEFAULT_FORCING_INTERVAL_SECONDS`, 21,600 s, while the
    ``namelist.wps`` emitted BESIDE that very file says
    ``interval_seconds = 3600``.

    MEASURED, at a declared 16 GiB card on the 12/3 km ladder: the
    wizard sized the ladder at the producer's own 3,600 s and printed
    ``peak envelope 13.74 GiB``, and the memory check it then ran on its
    own output priced the same file at 21,600 s and printed ``forecast
    needs 13.69 GiB``.  One command, one file, two prices, and the lower
    of the two was the one that described a run that is not going to
    happen: the boundaries this file will be prepared with are hourly,
    which is the number its own namelist carries.

    The cadence is the PRODUCER's published fact, so it is read from the
    registry row rather than from a second table here, and a newly
    registered producer reaches this answer without an edit.  A row that
    declares no cadence, and a table naming nothing registered, still
    answer ``None``: a guessed cadence would ask the preparation for
    valid times the producer never published, which is the breakage
    :func:`woof.source_adapters.source_forcing_interval_seconds`
    refuses outright.
    """
    from woof.source_adapters import source_forcing_interval_seconds

    fetch = raw.get("fetch")
    if not isinstance(fetch, dict):
        return None
    recorded = fetch.get("source")
    if not isinstance(recorded, str):
        return None
    try:
        return source_forcing_interval_seconds(recorded)
    except ValueError:
        return None


def recorded_forcing_interval_seconds(raw: dict) -> float | None:
    """The boundary cadence a configuration with no staged case states,
    for every surface that prices one: a declared ``[fetch] cadence`` in
    hours, else the recorded producer's published cadence
    (:func:`_recorded_source_forcing_interval`), else ``None``.

    ONE read for ``woof check``, ``woof go`` and ``run-plan
    --estimate``.  The plan review used to read only the declared
    cadence and price a producer with none at
    :data:`DEFAULT_FORCING_INTERVAL_SECONDS`, while the check priced the
    same file at the producer's own cadence -- the same one-file-two-prices
    defect :func:`_recorded_source_forcing_interval` closed on the check,
    left open on the document a front end renders.
    """

    cadence = (raw.get("fetch") or {}).get("cadence")
    if cadence is not None:
        return float(cadence) * 3600.0
    return _recorded_source_forcing_interval(raw)


def config_forcing_schedule(
        path: Path, exp: ExperimentConfig, *, input_catalog=None,
        fetch_cadence_hours: float | None = None,
) -> tuple[float | None, int | None]:
    """Validated/configured cadence and actual retained boundary count.

    A staged case's time inventory takes precedence over advisory fetch
    hints. Missing inputs leave the count unknown; no values are decoded to
    answer this memory question. Actual catalogs are reused when available.

    With no staged case and no cadence hint, the cadence is the one the
    recorded producer publishes
    (:func:`_recorded_source_forcing_interval`), which is the number the
    ``namelist.wps`` emitted beside such a config already carries.  It
    used to be ``None`` there, and ``None`` is what every caller turns
    into :data:`DEFAULT_FORCING_INTERVAL_SECONDS`.
    """
    import io
    import tomllib

    from woof.config_authority import read_config_authority
    from woof.experiment import is_experiment_toml_bytes

    path = Path(path)
    authority = read_config_authority(path)
    if not is_experiment_toml_bytes(authority.payload):
        return None, None
    raw = tomllib.load(io.BytesIO(authority.payload))
    table = raw.get("case_data")
    if not isinstance(table, dict):
        if fetch_cadence_hours is not None:
            return float(fetch_cadence_hours) * 3600.0, None
        return recorded_forcing_interval_seconds(raw), None
    from woof.case_data import build_case_data
    data = build_case_data(table, source=str(path), base_dir=path.parent,
                           require_inputs=False, require_met_inputs=False)
    return case_forcing_schedule(data, exp, input_catalog=input_catalog)


def case_forcing_schedule(data, exp: ExperimentConfig, *, input_catalog=None
                          ) -> tuple[float | None, int | None]:
    """Price an already resolved case's complete retained forcing schedule.

    The same inventory applies to configs loaded from a path, inline plan
    text, and generated intent. Paths have already been resolved by the
    config loader; this helper neither rewrites nor decodes the input values.
    """
    from woof.ingest.grib import inspect_era5_forcing_times
    from woof.ingest.preflight import _select_contiguous_times

    interval = data.forcing_interval_s
    times = tuple(getattr(input_catalog, "valid_times", ()) or ())
    if not times and data.forcing and data.vtable.is_file() and all(
            forcing.is_file() for forcing in data.forcing):
        raw_times = inspect_era5_forcing_times(data.forcing, data.vtable)
        times, _, _ = _select_contiguous_times(raw_times, interval)
    if not times:
        return interval, None
    if exp.start_time not in times:
        raise ValueError(f"forcing is missing the requested start time {exp.start_time}")
    usable = tuple(when for when in times if when >= exp.start_time)
    if len(usable) < 2:
        raise ValueError("forcing needs at least two valid times at/after the requested start")
    if (usable[-1] - exp.start_time).total_seconds() < exp.run_seconds:
        raise ValueError(f"forcing ends at {usable[-1]}, before the requested forecast end")
    if interval is not None:
        for earlier, later in zip(usable, usable[1:]):
            if (later - earlier).total_seconds() != interval:
                raise ValueError(
                    f"declared forcing_interval_s = {interval:g} disagrees with "
                    f"the supplied schedule {earlier} to {later}")
    if interval is None:
        deltas = {(later - earlier).total_seconds()
                  for earlier, later in zip(usable, usable[1:])}
        if len(deltas) == 1:
            interval = deltas.pop()
    return interval, len(usable) - 1


def config_forcing_source(path: Path, *,
                          priced_only: bool = True) -> str | None:
    """The forcing product a config records, or None if it records none.

    The preprocessing phase is priced against the SOURCE's level count
    and field inventory, so an unpriceable source has to be said out
    loud rather than silently reported as "the forecast is the whole
    story" -- which is exactly how a domain sized to a 12 GB card came
    to die in preprocessing after the download.

    ``priced_only`` (the default, and the contract every existing caller
    was written against) folds "records a source this estimator cannot
    price" into the same ``None`` as "records no source at all".  Those
    are different facts and a REPORT must not conflate them: told only
    ``None``, :func:`unpriced_ingest_note` says "this config records no
    forcing product at all" to a reader looking at a ``[fetch]`` table
    they typed themselves.  Pass ``priced_only=False`` to get the
    recorded name back so the note can say which source it was.
    """
    import io
    import tomllib

    from woof.config_authority import read_config_authority
    from woof.experiment import is_experiment_toml_bytes

    path = Path(path)
    authority = read_config_authority(path)
    if not is_experiment_toml_bytes(authority.payload):
        return None
    return recorded_forcing_source(
        tomllib.load(io.BytesIO(authority.payload)), priced_only=priced_only)


def recorded_forcing_source(raw: dict, *,
                            priced_only: bool = True) -> str | None:
    """The forcing product a parsed configuration records, or None.

    :func:`config_forcing_source`'s reading of an already parsed TOML,
    so a surface that holds the configuration's text rather than its
    path (``woof run-plan --estimate``) reads the same name ``woof
    check`` reads off the file.  ``priced_only`` as there.
    """
    table = raw.get("fetch")
    if isinstance(table, dict) and isinstance(table.get("source"), str):
        source = table["source"].strip().lower()
        if source in SOURCE_ANALYSIS_LEVELS or not priced_only:
            return source
        return None
    # The config-driven front door decodes ERA5 GRIB1; a [case_data]
    # table is that route by definition.
    return "era5" if "case_data" in raw else None


def config_preparation_route(path: Path) -> str | None:
    """The preparation route row a config's own door prepares on, or None.

    A config whose forcing :func:`config_forcing_source` reads from its
    ``[case_data]`` table (no ``[fetch]`` source) is prepared by
    ``woof run`` itself, the ``experiment`` row of
    :data:`woof.ingest.preparation_price.PREPARATION_ROUTES`, whose door
    prices its allocator reserve at that row's measured headroom.  Any
    other config is prepared by a door that picks its route, and the
    estimate keeps the default headroom.

    The run door takes the ``experiment-host-store`` row instead when its
    streaming plan, made against the card at run time, puts the case in
    a host store; the config alone cannot say that, so such a config is
    priced here at the ``experiment`` row's higher headroom on purpose,
    at or above what that door charges.
    """
    import io
    import tomllib

    from woof.config_authority import read_config_authority
    from woof.experiment import is_experiment_toml_bytes

    authority = read_config_authority(Path(path))
    if not is_experiment_toml_bytes(authority.payload):
        return None
    raw = tomllib.load(io.BytesIO(authority.payload))
    table = raw.get("fetch")
    if isinstance(table, dict) and isinstance(table.get("source"), str):
        return None
    return "experiment" if "case_data" in raw else None


def unpriced_ingest_note(path: Path, source: str | None = None) -> str:
    """Why the preprocessing phase could not be priced for this config.

    Said in full rather than shortened to a footnote, because the number
    beside it is a FORECAST number and a reader who takes it for the
    whole run is making the exact mistake this phase estimate exists to
    prevent.
    """
    known = ", ".join(sorted(SOURCE_ANALYSIS_LEVELS))
    if source:
        why = (f"--source {source} ingests on a lane this estimator does "
               f"not model (priced sources: {known})")
    else:
        why = ("this config records no forcing product at all, so there "
               f"is no ingest lane to price (priced sources: {known})")
    return (f"preprocessing (ingest) NOT PRICED: {why}, so the envelope "
            "beside it covers the FORECAST only.  Preprocessing has its "
            "own peak and it is not always the smaller one.")


def ingest_host_geometry(args) -> tuple[int, int, int] | None:
    """``(source grid points, valid times, 2-D fields per time)`` DECODED.

    Both are properties of the forcing FILES and of nothing in the
    experiment TOML, so they are read off the input catalog that ``woof
    check`` has already built by the time this section runs -- the cheap
    CPU input preflight runs first and only a zero return advances to the
    memory estimator (``woof/cli.py`` combined check policy), and it
    leaves its catalog on ``args``.

    NEVER BUILDS ONE.  Decoding the forcing in order to price the decode
    would spend the exact host memory this section exists to warn about,
    and on the configuration that matters it would spend it before the
    warning could be printed.  No catalog means ``None``, and the report
    then says NOT PRICED with the reason -- the convention this module
    already applies to an unpriceable ingest source.

    ``valid_times`` is the CATALOG's count, which is not the forecast's:
    the catalog selects the longest contiguous run of times present in the
    files and never sees ``run_seconds``, so a user who fetched a wider
    window than they integrate pays for the whole window here.

    THE FIELD COUNT IS THE DECODED ONE, not
    :func:`source_analysis_fields_per_time`.  That table is a nominal
    inventory -- 37 levels and 19 surface fields for ERA5 -- and nothing
    requires a config to match it: ``_check_levels``
    (`woof/ingest/preflight.py`) asks only that the levels be finite,
    strictly increasing, reach ``p_top`` and go down to 1000 hPa, so a
    legal 13-level CDS subset decodes 82 fields where the table charges
    204.  Charging the table there would over-state by 2.5x, and this
    figure gates a REFUSAL: a refusal must never fire on a run that would
    have completed.  The exact count is free -- the catalog's own
    snapshots are the decoded arrays -- so it is the one taken.
    """
    catalog = getattr(args, "input_catalog", None)
    shape = getattr(getattr(catalog, "spatial_coverage", None), "shape", None)
    times = getattr(catalog, "valid_times", None)
    snapshots = getattr(catalog, "snapshots", None)
    if not shape or not times or not snapshots:
        return None
    points = 1
    for extent in shape:
        points *= int(extent)
    # Horizontal slices, not variables: a pressure-level cube counts once
    # per level, which is what the host bytes are actually made of.  A
    # field of any other rank is not a shape this arithmetic describes, so
    # the geometry is withheld and the report says NOT PRICED rather than
    # publishing a number built on a guess.
    fields = 0
    for value in getattr(snapshots[0], "fields", {}).values():
        extent = tuple(getattr(value, "shape", ()))
        if len(extent) == 2:
            fields += 1
        elif len(extent) == 3:
            fields += int(extent[0])
        else:
            return None
    if points <= 0 or fields <= 0:
        return None
    return points, len(times), fields


#: ``woof check`` exit code for "every gate passed, but the observed peak
#: envelope exceeds the budget".  Nonzero, because the report says in prose
#: that the run may not fit and a script must be able to see that; distinct
#: from 1, because no gate failed and the levers are different.
_EXIT_ENVELOPE_OVER_BUDGET = 4

#: ``woof check`` exit code for "the ingest phase's HOST residency does not
#: fit this machine's available RAM".  A refusal rather than a note, and the
#: reasons it is one: it is a property of the configuration and not of this
#: command's flags, the lever is real and printed (fetch a narrower area or
#: a shorter forcing window), and the failure it describes is the one no
#: other door in this product can see -- a host OOM kills the worker from
#: outside with no traceback and nothing for woof to print.
#:
#: Distinct from 4 because the budget is a different one -- RAM, not VRAM --
#: and the levers do not overlap: no tiling, no column chunk and no smaller
#: card moves this number.  Softer than 1/2/3, which are about the gates
#: themselves.
_EXIT_HOST_MEMORY_OVER_BUDGET = 5
#: `woof check` exit when the run door's own [tiles] walk refuses the
#: configured execution plan.  The report still prices the resident road
#: (the only one left to quote), but a plan the run refuses is not a
#: preflight pass: `woof go` on the same file stops at that refusal, and
#: this command exited 0 beside the sentence that said so.
_EXIT_EXECUTION_PLAN_REFUSED = 6


def _format_bytes(n: int | None) -> str:
    return "n/a" if n is None else f"{n / GIB:7.2f} GiB"


def _leg_text(value: bool | None) -> str:
    return {True: "PASS", False: "FAIL", None: "not measured"}[value]


def absent_gate_metrics(gates: dict[str, bool | None]) -> tuple[str, ...]:
    """The N0 legs nothing measured, in :data:`N0_GATE_METRICS` order."""
    return tuple(metric for metric in N0_GATE_METRICS
                 if gates.get(metric) is None)


def memory_gate_verdict(gates: dict[str, bool | None]) -> str:
    """``"fail"`` / ``"incomplete"`` / ``"pass"`` over the N0 chain.

    THE SAME THREE-VALUED REDUCTION the verify lane already uses --
    :func:`woof.verify.spectral_receipt.evaluate_gates` and
    the reference case's chain's N5S compound, whose policy
    field spells it ``incomplete-not-pass``.  One product, one word for
    one state; this is not a second vocabulary.

    WHY THE MEMORY SECTION NEEDED IT.  :func:`evaluate_alloc_gates` is
    already careful -- "a missing measurement can never report a pass",
    and it returns ``None`` for every leg it could not evaluate.  What
    was missing is a reduction that can SAY so: the report reduced the
    chain by dropping the absent legs and taking ``all()`` over the
    survivors, and ``all([True])`` is ``all([True, True, True])``.  One
    evaluated leg out of three then read exactly like three of three, in
    a section that prints no verdict of its own, under a headline
    (``woof input preflight: PASS``) emitted by a different module about
    nineteen file/time/table checks that are not about memory at all.

    Absence outranks a pass and a failure outranks absence: a leg that
    FAILED is a measured refusal and is the harder verdict.
    """
    legs = [gates.get(metric) for metric in N0_GATE_METRICS]
    if any(leg is False for leg in legs):
        return "fail"
    if any(leg is None for leg in legs):
        return "incomplete"
    return "pass"


def _warn_unstaged_physics_tables(exp) -> None:
    """One line when a selected scheme's lookup tables are not staged.

    A WARNING, not a gate.  ``woof check`` is the memory preflight and
    the page that calls it "Preflight"; a person who runs it before an
    mp8 case should hear that the tables are absent while the fix is
    still one command and no download has started, and the run doors
    (``--materialize-authorities`` and both prepared runners) are where
    the same condition is actually refused.  Sizing a domain whose
    tables are elsewhere is a legitimate thing to do, so nothing here
    changes an exit code.
    """

    try:
        selected = {int(domain.run.mp_physics) for domain in exp.domains}
        if not selected & {8, 28}:
            return
        from woof.core.thompson_contract import CLASSIC_TABLE_ASSETS
        from woof.table_assets import require_thompson_tables

        assets = CLASSIC_TABLE_ASSETS
        if 28 in selected:
            from woof.core.thompson_aerosol_contract import (
                AEROSOL_TABLE_ASSETS)

            assets = (*CLASSIC_TABLE_ASSETS, *AEROSOL_TABLE_ASSETS)
        # stage=False: a memory preflight must not start a 315 MiB
        # download; the run doors stage the set on first use.
        require_thompson_tables(assets=assets, stage=False)
    except FileNotFoundError as error:
        from woof.explain import warn

        warn(str(error),
             why="The tables are read at load and validated byte for "
                 "byte, so a run cannot start without them.  This "
                 "preflight sizes memory and does not need them, which "
                 "is why it says so and continues.")
    except Exception:  # pragma: no cover - never let an advisory throw
        return


#: The one live read of this machine's card, kept for the process.  A
#: profile is device constants (name, shader census, default stack
#: limit, compile platform), so a second read answers the same profile
#: and costs an NVRTC probe for nothing.  A Noah-MP configuration reads
#: the card on every estimate call that is handed no profile --
#: ``recommend_column_chunk``'s loop, the composition walk's rows --
#: which is what made the repeat cost visible.  Filled only by a
#: successful read; ``GPUWM_NO_LOCAL_GPU`` is consulted on every call so
#: the switch keeps its meaning after a read.
_LIVE_DEVICE_PROFILE: list[DeviceLocalMemoryProfile] = []


def forget_live_device_local_memory_profile() -> None:
    """Drop the process's cached live read (a test that swaps the runtime
    under the module, or a caller that knows the device changed)."""
    _LIVE_DEVICE_PROFILE.clear()
def _warn_unmet_run_preparation(exp) -> None:
    """One line per machine precondition this install does not meet.

    THE SAME POSTURE as the table warning above, for the same reason, and
    it is here because this command is the one a person runs to hear
    "PASS" before spending anything.  A precondition in
    ``woof.config.RUN_PREPARATION_PRECONDITIONS`` is a question about the
    INSTALL, and ``woof check`` is also the portable sizing door -- a
    declared ``--budget-gib`` sizes for a machine that is not this one,
    where "the dataset is not here" says nothing about whether the run
    fits there.  So it reports and does not change an exit code; the doors
    that COMMIT to building a forecast (``woof go``'s stage composer and
    the experiment run dispatch) raise the identical sentence from the
    identical inventory.
    """

    try:
        from woof.config import experiment_preparation_refusals

        unmet = experiment_preparation_refusals(exp)
    except Exception:  # pragma: no cover - never let an advisory throw
        return
    if not unmet:
        return
    from woof.explain import warn

    for label, sentence in unmet:
        warn(f"{label}: {sentence}",
             why="This preflight sizes memory and can size for a machine "
                 "that is not this one, which is why it reports the gap "
                 "and continues.  `woof go` and `woof run` refuse it "
                 "before they fetch anything.")


def live_device_local_memory_profile() -> DeviceLocalMemoryProfile | None:
    """This machine's own local-memory profile, or ``None``.

    Read off the device once per process and kept
    (:data:`_LIVE_DEVICE_PROFILE`); the switch below is honoured on
    every call.

    The local-memory backing store is ``(frame - default stack) x SM
    count x threads per SM``, so it is a property of the DEVICE, and the
    reference profile in this module is a 170-SM RTX 5090 -- roughly
    2.2x the resident-thread capacity of a 76-SM 4080.  Charging every
    card the 5090's backing store is the same mistake as charging Linux
    the WDDM pool constants: an accounting term measured somewhere else.

    Read only when the free-VRAM figure is being MEASURED off this
    device, i.e. when the answer is about this machine.  A declared
    ``--budget-gib`` says "size for a machine that is not this one", and
    that machine's SM count is unknown, so it keeps the reference
    profile -- which over-prices rather than under-prices.
    """

    from woof.local_gpu import no_local_gpu

    if no_local_gpu():
        # Same switch, same scope as the probe subprocess below: reading
        # the device's SM census is device contact, and a caller that
        # cannot measure prices against the reference profile, which
        # over-prices rather than under-prices.
        return None
    if _LIVE_DEVICE_PROFILE:
        return _LIVE_DEVICE_PROFILE[0]
    try:
        import cupy as cp

        profile = local_memory_profile_from_device(cp)
    except Exception:
        return None
    _LIVE_DEVICE_PROFILE.append(profile)
    return profile


#: The device probe lives in :mod:`woof.core.device_probe`, a leaf the
#: standalone RW-WPS package stages without this module, so its automatic
#: preparation backend reads the same card load woof does.  Every name is
#: re-exported here for the callers and tests that read it from preflight.
from woof.core.device_probe import (  # noqa: E402,F401
    DEVICE_MEMORY_PROBE_TIMEOUT_SECONDS,
    PROBE_EXIT_CARD_UNREAD,
    PROBE_EXIT_NO_RUNTIME,
    PROBE_REASON_NO_RUNTIME,
    _DEVICE_MEMORY_PROBE_SOURCE,
    _device_memory_probe,
    _probe_error,
    _probe_failure_report,
    _probe_last_line,
    device_memory_probe_reason,
    device_memory_probe_subprocess,
)


def profile_from_device_probe(payload) -> DeviceLocalMemoryProfile | None:
    """The probe payload's device-profile half, typed, or ``None``.

    ``None`` falls back exactly like :func:`live_device_local_memory_profile`
    returning ``None``: the callers price against the reference profile,
    which over-prices rather than under-prices.

    This build's own probe ships no ``bare_context_bytes``; the context is
    priced from the census (see
    :data:`MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD`).  A payload
    that STATES one -- a target-hardware sizing document, an older node's
    probe -- is priced as stated, because a stated number reads the same
    every time.
    """

    profile = payload.get("profile") if isinstance(payload, dict) else None
    if not isinstance(profile, dict):
        return None
    bare = profile.get("bare_context_bytes")
    if isinstance(bare, bool) or not isinstance(bare, int) or bare <= 0:
        # Absent or unusable prices from the census, never from zero: a
        # zero-byte CUDA context is not a thing this could mean.
        bare = None
    platform = profile.get("compile_platform")
    # Both halves as non-empty strings, or the platform is unread: an
    # older probe does not carry the field, and "unavailable" is not a
    # platform anyone measured.
    if (not isinstance(platform, (list, tuple)) or len(platform) != 2
            or any(not isinstance(v, str) or not v or v == "unavailable"
                   for v in platform)):
        platform = None
    else:
        platform = (str(platform[0]), str(platform[1]))
    try:
        return DeviceLocalMemoryProfile(
            name=str(profile["name"]),
            multiprocessor_count=int(profile["multiprocessor_count"]),
            max_threads_per_multiprocessor=int(
                profile["max_threads_per_multiprocessor"]),
            default_stack_limit_bytes=int(
                profile["default_stack_limit_bytes"]),
            bare_context_bytes=bare,
            compile_platform=platform,
        )
    except (KeyError, TypeError, ValueError):
        return None


#: How close a declared card size has to be to the local card's measured
#: total to BE the local card.  Capacities are reported in whole MiB and
#: converted through GiB floats on the way in, so an exact comparison
#: would fail on rounding alone; 0.5% is far tighter than the gap between
#: any two card tiers.
LOCAL_CARD_MATCH_TOLERANCE = 0.005


def declares_the_local_card(card_total_gib: float | None) -> bool:
    """Is ``--vram-gib`` naming the card that is in this machine?

    A declaration is normally a statement about hardware that is
    somewhere else, and that is priced against the conservative
    reference profile.  But the wizard declares the size of the card it
    just MEASURED when it hands the emitted config to ``woof check``,
    and substituting a 170-SM reference under a 68-SM card there made
    the two doors disagree about one machine.

    False whenever the local card cannot be read: an unreadable card
    cannot be the one being described, and the reference profile is the
    safe answer.
    """

    if card_total_gib is None:
        return False
    return declares_this_card(card_total_gib, device_physical_total_bytes())


def declares_this_card(card_total_gib: float | None,
                       total_bytes: int | None) -> bool:
    """Is a declared capacity naming the card whose total this is?

    The rule :func:`declares_the_local_card` applies to the card it reads
    itself, for a caller that already holds a card's total (the probe
    subprocess reports one beside the profile).  ``False`` with no
    declaration and with no total: an unread card cannot be the one
    being described.
    """

    if card_total_gib is None or not total_bytes:
        return False
    declared = float(card_total_gib) * GIB
    return abs(declared - total_bytes) <= LOCAL_CARD_MATCH_TOLERANCE * total_bytes


def _required_memory_without_kernels(exp, args, *,
                                     readiness: str | None = None) -> dict:
    """The CPU estimate ``woof check`` makes when this machine's kernels
    are not proven to compile and run.

    No budget, allocation or automatic tiling decision is made and no
    kernel is compiled.  The card IS read, through the same short-lived
    probe subprocess ``woof run-plan --estimate`` reads it with
    (:func:`device_memory_probe_subprocess`): the readiness gate is about
    running kernels, and pricing the card in the machine needs only its
    census, so a card that answers the probe is priced from that census
    and the two surfaces quote one figure for one file.  This route used
    to price the reference profile whenever readiness was unmet, so a
    machine whose CuPy wheel carries no CUDA headers read 1,353,931,428
    bytes from ``woof check`` and 1,205,295,780 from the estimate
    document for one plan on one RTX 4090.

    ``readiness`` is the verdict that sent :func:`check_main` here, and
    it is stated beside the figure either way: with a card read, as the
    reason its kernels were not compiled; with none, as part of why the
    reference card is priced.  A declared ``--vram-gib`` naming a
    capacity that is not this card's keeps the reference profile,
    exactly as the measured route does (:func:`declares_the_local_card`):
    a declaration is about a machine that is elsewhere.  A Noah-MP
    configuration is priced from the read card's own platform row when
    the probe read a platform and from the ceiling when it did not, and
    the basis says which (:func:`non_pool_basis`).
    """
    prepared = _prepared_check_inputs(args)
    urban_columns = (None if prepared is None or args.alloc
                     else prepared.urban_columns)
    configured_interval, forcing_intervals = (
        (prepared.forcing_interval_seconds, prepared.forcing_intervals)
        if prepared is not None else config_forcing_schedule(
            args.config, exp, input_catalog=getattr(args, "input_catalog", None)))
    explicit_interval = args.forcing_interval_s
    if (explicit_interval is not None and forcing_intervals is not None
            and configured_interval is not None
            and explicit_interval != configured_interval):
        raise ValueError("The requested forcing interval disagrees with the supplied forcing cadence")
    interval = (explicit_interval if explicit_interval is not None else
                configured_interval if configured_interval is not None else
                DEFAULT_FORCING_INTERVAL_SECONDS)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("--forcing-interval-s must be finite and positive")
    declared_gib = getattr(args, "vram_gib", None)
    probe = device_memory_probe_subprocess()
    read = profile_from_device_probe(probe)
    total = None if probe is None else probe.get("total_bytes")
    total = (int(total) if isinstance(total, int)
             and not isinstance(total, bool) and total > 0 else None)
    verdict = readiness or "this machine's GPU readiness is unjudged"
    device_read = read is not None and (
        declared_gib is None or declares_this_card(declared_gib, total))
    if device_read:
        profile = read
        vram_gib = (float(declared_gib) if declared_gib is not None
                    else None if total is None else total / GIB)
        device_basis = "measured local device; kernels not compiled here"
        basis = ("CPU-only metadata; resident alternative priced on the card "
                 f"read in this machine, kernels not compiled here ({verdict}): "
                 f"{non_pool_basis(profile, exp)}")
    else:
        profile = card_local_memory_profile(declared_gib)
        vram_gib = declared_gib
        total = None
        if probe is None:
            unread = (device_memory_probe_reason()
                      or "no card answered the probe")
        elif read is None:
            unread = "the probe answered with no device profile"
        else:
            unread = (f"--vram-gib {declared_gib:g} names a card that is not "
                      f"the one read in this machine ({read.name}), so the "
                      "declared card is priced as one that is elsewhere")
        device_basis = f"conservative reference; local device unmeasured ({unread})"
        basis = ("CPU-only metadata; resident alternative with conservative "
                 "reference GPU overhead")
        if selects_noahmp(exp):
            # Say what kept this card unread beside the ceiling basis: a
            # user who asked about the card in front of them should read
            # that it was not the one priced.
            from woof.core.noahmp_frame_provenance import noahmp_frame_basis

            frames = noahmp_frame_basis(physics_kernel_modules(exp), profile)
            basis = (f"{basis} ({verdict}; its card was not read on this "
                     f"route: {unread}); {frames.sentence()}")
    # THE TABLES THE RUN HOLDS, as check_main's declared and measured
    # routes and ``run-plan --estimate`` price them: the root's boundary
    # carries the analysed hydrometeors the recorded source publishes.
    # Priced without them, a bare check on a machine whose kernels are not
    # proven read a 550 x 550 x 49 Morrison root whose source publishes
    # five masses 24.8 MB below the figure its own suggested
    # --free-gib/--vram-gib follow-up printed for the same file.
    from woof.boundary_fields import source_boundary_species

    recorded = (prepared.boundary_source if prepared is not None else
                config_forcing_source(args.config, priced_only=False))
    estimate = estimate_experiment(
        exp, column_chunk=args.column_chunk, forcing_intervals=forcing_intervals,
        forcing_interval_seconds=interval, vram_gib=vram_gib,
        profile=profile, boundary_species=source_boundary_species(recorded),
        urban_columns=urban_columns)
    source = (None if prepared is not None else
              recorded if recorded in SOURCE_ANALYSIS_LEVELS else None)
    ingest = (estimate_ingest(
        exp, source=source, forcing_interval_seconds=interval,
        forcing_intervals=forcing_intervals,
        vram_gib=vram_gib, profile=profile,
        route=config_preparation_route(args.config))
        if source in SOURCE_ANALYSIS_LEVELS else None)
    return {
        "status": "estimated", "basis": basis,
        "column_chunk": estimate.column_chunk,
        "domains": {f"d{domain.grid_id:02d}": {
            "resident_bytes": domain.resident_bytes,
            "transient_bytes": domain.transient_bytes,
            "by_category": {category: domain.category_bytes(category) for category in
                            ("state", "physics", "scratch", "lbc", "nest", "diagnostic", "sase", "transient")},
        } for domain in estimate.domains},
        "resident_bytes": estimate.resident_bytes,
        "transient_peak_bytes": estimate.transient_peak_bytes,
        "workspace_bytes": estimate.workspace_bytes,
        "k_tables_bytes": estimate.k_tables_bytes,
        "physics_tables_bytes": estimate.physics_tables_bytes,
        "alloc_estimate_bytes": estimate.alloc_estimate_bytes,
        "resident_forecast_peak_envelope_bytes": estimate.peak_envelope_bytes,
        "bem_column_workspace_basis": _check_bem_basis(
            estimate, prepared, alloc=getattr(args, "alloc", False)),
        "prepared_urban_columns": urban_columns,
        # THE DEVICE BESIDE THE FIGURE: which card the non-pool terms
        # were priced on and whether it is the one in this machine, so
        # this figure and the estimate document's can be told apart by
        # their receipts when they differ and matched when they agree.
        "device_read": device_read,
        "device_basis": device_basis,
        "local_memory_profile": profile.name,
        "device_total_bytes": total,
        "non_pool_device_bytes": estimate.non_pool_device_bytes,
        "non_pool_basis": non_pool_basis(profile, exp),
        "ingest": None if ingest is None else {
            "source": source, "resident_bytes": ingest.resident_bytes,
            "peak_envelope_bytes": ingest.peak_envelope_bytes},
        "budget_bytes": None, "measured_free_bytes": None,
        "memory_verdict": "unavailable",
        "note": "Free VRAM and the admission budget are unavailable. No fit or streamed-road judgment was made; declare target memory to compare.",
    }


def target_capacity_source(sampled, target_hardware) -> str:
    """Where the check's ``--vram-gib`` card capacity came from, in words.

    The flag reaches this check two ways and only one of them is a
    declaration.  Typed by a person, it states the size of a card that
    need not be in this machine.  Passed by ``woof domain`` beside its
    shared sizing sample, it is the total that sample MEASURED, off this
    card or off the selected node's hardware snapshot.  Every capacity
    was printed ``(declared)``, so the wizard's own follow-up check
    called the card it had just read "declared" one line under
    ``Physical GPU capacity: ... (measured)`` for the same card, and a
    reader could not tell which figure anyone had actually read.
    """

    if sampled is None:
        return "declared"
    return "measured on the selected GPU" if target_hardware else "measured"


def check_main(args) -> int:
    """``woof check CONFIG [--alloc]``: memory section of the preflight.

    Exit codes: 0 = every requested leg measured and PASSED; 1 = a leg
    FAILED; 2 = fail-closed -- the requested legs could not be evaluated
    (estimator mode with no budget, or an ``--alloc`` that produced no
    measurements); 3 = the ``--alloc`` run ABORTED before measuring
    (headroom precheck / OOM) -- the JSON/text report is still emitted
    with the estimate-side legs and the abort reason; 4 = every gate
    passed but the observed peak envelope EXCEEDS the budget, which the
    report warns about in prose.  4 is distinct from 1 because the gates
    genuinely passed and the levers differ, but it is nonzero because a
    report whose own text says the run may not fit must never read green
    to a script.  5 = the ingest phase's HOST residency exceeds this
    machine's available RAM -- a different budget with different levers,
    and the one failure mode no gate above can see.  A harder verdict
    wins: 1, 2 and 3 outrank 5, which outranks 4.
    """
    declared_free_gib = getattr(args, "free_gib", None)
    declared_memory = args.budget_gib is not None or declared_free_gib is not None
    sampled = getattr(args, "_shared_sizing_budget", None)
    target_hardware = getattr(args, "_target_hardware_supplied", False)
    target_machine = getattr(args, "_shared_target_machine", None)
    if target_hardware:
        from tilestream.autoplan import Machine
        if (target_hardware is not True or args.alloc or args.budget_gib is not None
                or args.rail_mib is not None or declared_free_gib is None
                or args.vram_gib is None
                or (target_machine is not None and (
                    not isinstance(target_machine, Machine)
                    or target_machine.vram_bytes != int(declared_free_gib * GIB)
                    or type(target_machine.host_bytes) is not int
                    or target_machine.host_bytes <= 0))):
            raise ValueError("The internal target machine must match the check's exact declared GPU budget")
    elif target_machine is not None:
        raise ValueError("An internal target machine requires its selected-hardware binding")
    if sampled is not None:
        from woof.domain_wizard import SizingBudget
        if (not isinstance(sampled, SizingBudget) or not sampled.measured
                or args.alloc or args.budget_gib is not None or args.rail_mib is not None
                or declared_free_gib is None or args.vram_gib != sampled.vram_gib
                or int(declared_free_gib * GIB) != sampled.free_bytes):
            raise ValueError("The internal sizing sample must match the check's exact measured budget")
    if declared_free_gib is not None:
        if not math.isfinite(declared_free_gib) or declared_free_gib <= 0:
            raise ValueError("--free-gib must be a finite positive amount of free VRAM")
    # The other declarations are byte arithmetic below: int(inf * GIB)
    # was an OverflowError traceback, NaN an unnamed "cannot convert"
    # sentence, and a negative or zero card total or a negative reserve
    # was accepted and printed a verdict for a card that cannot exist.
    # A zero budget or reserve is a real statement and still prices.
    for option, value, zero_ok, what in (
            ("--vram-gib", getattr(args, "vram_gib", None), False,
             "a card capacity"),
            ("--budget-gib", args.budget_gib, True, "an allocation budget"),
            ("--reserve-gib", getattr(args, "reserve_gib", None), True,
             "a reserve")):
        if value is not None and not (math.isfinite(value) and (
                value > 0 or (zero_ok and value == 0))):
            bound = "zero or more" if zero_ok else "above zero"
            raise ValueError(f"{option} {value:g} is not {what}: pass a "
                             f"finite number of GiB, {bound}")
    rail_mib = getattr(args, "rail_mib", None)
    if rail_mib is not None and rail_mib < 1:
        raise ValueError(f"--rail-mib {rail_mib} is not a device ceiling: pass "
                         "a whole number of MiB, 1 or more")
    if args.alloc and declared_memory:
        raise ValueError("--alloc measures this GPU; omit --free-gib and --budget-gib")
    prepared = _prepared_check_inputs(args)
    exp = (prepared.experiment if prepared is not None
           else _load_experiment_any(args.config))
    urban_columns = (None if prepared is None or args.alloc
                     else prepared.urban_columns)
    from woof.output_disk import forecast_projection

    if getattr(args, "devices", None) is not None:
        # ``woof go --devices N``'s count, priced the way the run will be:
        # one card's envelope for a domain the split exists for is a
        # refusal the run itself would not give.  MEASURED: a 2560 x 1440
        # x 55 domain checked at 158.66 GiB on one card (refused) while
        # `woof go --devices 2` admitted it at 86.1 and 85.2 GiB per card.
        from dataclasses import replace as _replace
        from woof.core.devices import override_device_count
        exp = _replace(exp, devices=override_device_count(
            exp.devices, args.devices,
            log=lambda line: print(line, file=sys.stderr)))
    output_disk = forecast_projection(exp)
    import tomllib
    from woof.config_authority import read_config_authority
    from woof.runplan import drivability_for
    hints = tomllib.loads(read_config_authority(args.config).payload.decode("utf-8")).get("fetch") or {}
    if (prepared is None and
            drivability_for(hints.get("source")).get("requires_source_root")):
        from woof.local_preparation import review_local_inputs
        review_local_inputs(hints, base_dir=Path(args.config).resolve().parent)
    if (target_hardware and target_machine is None
            and (getattr(exp.tiles, "mode", "off") != "off"
                 or any(getattr(getattr(domain, "tiles", None), "mode", "off") != "off"
                        for domain in exp.domains))):
        raise ValueError("Selected target host memory is required to check streamed domains")
    # The companion is a base dependency, including for CPU table sizing.
    # Resolve its presence/version before a kernel probe can hide the install
    # failure behind an unrelated GPU refusal. This reads no table arrays.
    from woof.data_assets import companion_root

    companion_root()
    _warn_unstaged_physics_tables(exp)
    if prepared is None:
        _warn_unmet_run_preparation(exp)
    # Declared-budget sizing is deliberately portable and needs no GPU.
    # A check of this machine must prove kernels can run before allocating
    # a forecast or claiming its measured memory budget is usable.
    readiness = {"status": "not_checked", "detail": (
        "estimate using the shared measured sizing sample; no allocation attempted"
        if sampled is not None else "declared-budget estimate")}
    if not declared_memory:
        from woof.doctor import _cuda_headers_check

        checked = _cuda_headers_check()
        readiness = {"status": checked.status, "detail": checked.detail,
                     "action": checked.action}
        if checked.status != "verified":
            command = ["woof", "check", str(args.config), "--free-gib", "FREE_GIB",
                       "--vram-gib", "CAPACITY_GIB"]
            if prepared is not None:
                command += ["--prepared-root", str(prepared.prepared_root)]
                if getattr(args, "wps_namelist", None) is not None:
                    command += ["--wps-namelist", str(args.wps_namelist)]
            planning = {"command": command,
                        "inputs": "Replace FREE_GIB and CAPACITY_GIB with the target GPU's declared free memory and capacity in GiB.",
                        "scope": "CPU-only estimate; does not verify local GPU readiness"}
            # No UnqualifiedNoahMP arm here any more: a tree with no
            # usable Noah-MP reading prices the units at the assumed
            # bound and says so in the basis, so this route always has a
            # number and a portable planning command.
            try:
                required = _required_memory_without_kernels(
                    exp, args, readiness=(
                        f"this machine's GPU readiness is {checked.status} "
                        f"({checked.brief or checked.detail})"))
            except (OSError, ValueError, RuntimeError, TypeError, AttributeError) as error:
                required = {"status": "unavailable", "reason": str(error),
                            "budget_bytes": None, "memory_verdict": "unavailable"}
            if args.json:
                print(json.dumps({"config": str(args.config),
                                  "gpu_readiness": readiness,
                                  "required_memory": required,
                                  "cpu_planning": planning,
                                  "output_disk": output_disk}, indent=2, allow_nan=False))
            else:
                label = "FAILED" if checked.status == "missing" else "UNVERIFIED"
                print(f"woof GPU readiness: {label}. {checked.brief or checked.detail}.")
                print(f"Forecast output disk: {output_disk['total_bytes'] / GIB:.2f} GiB "
                      "for configured history and checkpoints; "
                      "source downloads and preparation excluded.")
                print(f"  Next: {checked.action or 'woof doctor --explain'}")
                if required["status"] == "estimated":
                    # The card beside the figure: the one read in this
                    # machine, its kernels not compiled, or the reference
                    # card because none was read.
                    priced = (f"on {required['local_memory_profile']} as read "
                              "here, kernels not compiled" if required["device_read"]
                              else "on the reference card, none read here")
                    print("  CPU required-memory estimate (resident alternative): "
                          f"{required['alloc_estimate_bytes'] / GIB:.2f} GiB allocations; "
                          f"{required['resident_forecast_peak_envelope_bytes'] / GIB:.2f} GiB forecast envelope; {priced}.")
                    print("  Budget and fit judgment: unavailable. No GPU allocation or streaming decision was attempted.")
                    if required["device_read"] or selects_noahmp(exp):
                        # The basis where the number is: the card's census
                        # and, for Noah-MP, whether its frames came from
                        # this card's platform row or from the ceiling.
                        print(f"  Basis: {required['basis']}")
                else:
                    print(f"  CPU required-memory estimate unavailable: {required['reason']}")
                if planning["command"] is None:
                    print("  CPU-only planning: none.  " + planning["inputs"])
                else:
                    import shlex
                    print("  CPU-only planning: " + shlex.join(planning["command"]))
                    print("    " + planning["inputs"])
                if getattr(args, "explain", False):
                    print(checked.detail)
                    if checked.remedy:
                        print(checked.remedy)
            return 1 if checked.status == "missing" else 2
    configured_interval, forcing_intervals = (
        (prepared.forcing_interval_seconds, prepared.forcing_intervals)
        if prepared is not None else config_forcing_schedule(
            args.config, exp, input_catalog=getattr(args, "input_catalog", None)))
    explicit_interval = args.forcing_interval_s
    if (explicit_interval is not None and forcing_intervals is not None
            and configured_interval is not None
            and explicit_interval != configured_interval):
        raise ValueError(
            f"--forcing-interval-s = {explicit_interval:g} disagrees with "
            f"the supplied forcing cadence {configured_interval:g} s")
    forcing_interval = (
        explicit_interval if explicit_interval is not None
        else configured_interval if configured_interval is not None
        else DEFAULT_FORCING_INTERVAL_SECONDS)
    if not math.isfinite(forcing_interval) or forcing_interval <= 0:
        raise ValueError("--forcing-interval-s must be finite and positive")
    ingest_interval = (explicit_interval if explicit_interval is not None
                       else configured_interval)
    # THE FORCING SLOTS THE CARD HOLDS, priced the way `woof go` prices
    # them.  A config `woof go` runs chained (one domain, one card) streams
    # its lateral intervals through one device slot; a verified prepared
    # bundle is a sealed restore and holds the whole schedule.  Pricing the
    # whole schedule here refused a full-grid 18 h HRRR on a 96 GB card by
    # 0.50 GiB (19 hourly intervals, 8.24 GiB of boundary tables) that the
    # go gate admits.  The ingest phase keeps the whole schedule.
    from woof.ingest.boundary_stream import chained_resident_forcing_slots
    resident_slots = (chained_resident_forcing_slots(exp)
                      if prepared is None else None)
    resident_intervals = (forcing_intervals if resident_slots is None
                          else resident_slots if forcing_intervals is None
                          else min(int(forcing_intervals), resident_slots))
    #: Whether the free-VRAM figure below was measured off the device or
    #: derived from a declared --budget-gib.  Printed, because the two
    #: must never wear the same label.
    free_source = "measured"
    physical_total_bytes = (int(sampled.vram_gib * GIB) if sampled is not None else None)
    #: Declared physical capacity of the card this preflight is sizing for
    #: (``--vram-gib``, which the wizard passes from its card tier).  A
    #: ceiling on the free figure, never a source of one.
    card_total_gib = getattr(args, "vram_gib", None)
    #: Whether that capacity was declared or measured, as printed.
    capacity_source = target_capacity_source(sampled, target_hardware)
    #: The capacity ceiling that actually bound the free figure, if any.
    capped_to = None
    # Read the card BEFORE anything in this process touches CUDA, so the
    # other-process residency the rail must respect is not inflated by our
    # own context (which the reserve already carries).
    rail_bytes = (None if args.rail_mib is None
                  else int(args.rail_mib) * 1024 ** 2)
    other_process_bytes = None if rail_bytes is None else (
        device_wide_used_bytes())
    chunk = (exp.column_chunk if args.column_chunk is None
             else args.column_chunk)
    #: The device the non-pool terms are priced against.  This machine's
    #: own, whenever the free figure is measured off it; the reference
    #: 5090 profile when a declared budget says the target is elsewhere.
    profile = (sampled.device_profile if sampled is not None else
               None if declared_memory else live_device_local_memory_profile())
    if (sampled is None and not target_hardware and profile is None
            and declares_the_local_card(card_total_gib)):
        # A DECLARED budget for THIS card.  ``--budget-gib`` alone means
        # "the caller states the budget", not "the caller is describing
        # another machine" -- and the wizard's own follow-up check is
        # exactly that case: it sizes the card it just measured, states
        # the budget it sized against, and used to have the reference
        # 5090 profile substituted underneath it.  The wizard then read
        # 68 SMs and the check it printed read 170, on one card, and the
        # emission failed its own verification (task 206).
        #
        # Recognised by CAPACITY: ``--vram-gib`` naming the same total
        # this machine's card reports IS this machine's card.  A
        # declaration for any other size keeps the conservative
        # reference profile, which is what sizing hardware you do not
        # have is supposed to get.
        profile = live_device_local_memory_profile()
    if profile is None:
        # The reference profile's compile platform is nobody's; a
        # Noah-MP configuration on it prices its frames from the ceiling
        # over the recorded platforms and the NON-POOL BASIS line says
        # so, naming that the card was not read (non_pool_basis).
        profile = card_local_memory_profile(card_total_gib)
    #: What the config RECORDS, priceable or not.  A report that is about
    #: to tell the reader their ingest phase is unpriced has to name the
    #: source it could not price; folding it into the same ``None`` as
    #: "no [fetch] table at all" printed "this config records no forcing
    #: product at all" at a reader looking at the one they wrote.
    #: :func:`estimate_phases` keys off the same table either way, so an
    #: unpriceable name still leaves the ingest term absent.
    ingest_source = (prepared.source if prepared is not None else
                     config_forcing_source(args.config, priced_only=False))
    #: THE TABLES THE RUN HOLDS.  The root's boundary tables carry the
    #: analysed hydrometeors the recorded source publishes (its table row,
    #: :func:`woof.boundary_fields.source_boundary_species`), and
    #: :func:`estimate_phases` prices them.  The itemization this report
    #: prints, the reserve and the allocation gate are priced on the same
    #: tables; without them a 550 x 550 x 49 HRRR-forced Morrison root
    #: itemized 21.6 MB less boundary table than it holds, and reported an
    #: observed envelope 24.8 MB below the peak envelope printed beside it.
    #: ``--alloc`` builds and measures synthetic tables carrying the same
    #: species, so its measurement, its estimate side and the observed
    #: envelope are of the tables the run holds too; it used to build
    #: water-vapour-only tables and report that observed envelope beside a
    #: peak envelope priced with the hydrometeors.
    from woof.boundary_fields import source_boundary_species

    priced_boundary_source = (prepared.boundary_source if prepared is not None
                              else ingest_source)
    boundary_species = source_boundary_species(priced_boundary_source)
    # The retention term is a fraction OF the estimate, so the estimate is
    # formed first.  It is pure arithmetic and the runners re-derive it from
    # the same inputs, so there is no second source of truth.
    if getattr(getattr(exp, "devices", None), "enabled", False):
        from woof.core.devices import probe_devices, validate_device_count
        from woof.core.devices_memory import devices_gate, include_preparation
        from woof.free_threading import refuse_multi_card_under_gil
        refuse_multi_card_under_gil(exp.devices.count, "woof preflight")
        if args.alloc:
            raise ValueError("[devices] --alloc is not wired into the ranked road; "
                             "refused to prevent allocating a one-card reference domain")
        measured = None if declared_memory else probe_devices()
        if measured is not None:
            validate_device_count(exp.devices, measured["visible_count"])
        profiles = (None if measured is None else
                    {int(dev): profile_from_device_probe(row)
                     for dev, row in measured["cards"].items()})
        declared_budget = args.budget_gib if args.budget_gib is not None else declared_free_gib
        budgets = ({dev: int(declared_budget * GIB) for dev in exp.devices.device_ids()}
                   if declared_budget is not None else None if measured is None else
                   {int(dev): int(row["free_bytes"])
                    for dev, row in measured["cards"].items()})
        if len(exp.domains) > 1:
            # A split tree: each grid on the cards it runs on, the pricing
            # the `woof go` gate and the tree runner use.
            from woof.core.devices_memory import estimate_devices_tree
            estimate = estimate_devices_tree(
                exp, forcing_intervals=forcing_intervals,
                forcing_interval_seconds=forcing_interval, vram_gib=card_total_gib,
                profile=profile, source=priced_boundary_source, profiles=profiles)
        else:
            estimate = estimate_devices(
                exp, forcing_intervals=forcing_intervals,
                forcing_interval_seconds=forcing_interval, vram_gib=card_total_gib,
                profile=profile, source=priced_boundary_source, profiles=profiles,
                budgets=budgets)
        gate = devices_gate(estimate, budgets=budgets, host_budget=host_available_bytes())
        if prepared is None:
            include_preparation(
                gate, exp, source=ingest_source, budgets=budgets, host_budget=host_available_bytes(),
                forcing_intervals=forcing_intervals, forcing_interval_seconds=forcing_interval,
                ingest_forcing_interval_seconds=ingest_interval,
                vram_gib=card_total_gib, profile=profile)
        else:
            gate["preparation"] = {
                "priced": False, "already_completed": True,
                "reason": "verified prepared forecast inputs; preparation already completed"}
        gate["forcing_interval_seconds"] = forcing_interval
        gate["retained_forcing_intervals"] = forcing_intervals
        gate["boundary_species"] = list(boundary_species or ())
        if urban_columns is not None:
            gate["bem_column_workspace_basis"] = (
                "rank pricing keeps the conservative every-column BEP+BEM bound; "
                "prepared_urban_columns records verified full-grid counts, "
                "not their distribution among rank halos")
        unmeasured = [int(row["card"]) for row in estimate["cards"]
                      if budgets is None or budgets.get(row["card"]) is None]
        exit_code = 4 if gate["refuse"] else 2 if unmeasured else 0
        gate["evaluable"] = not unmeasured
        gate["unmeasured_cards"] = unmeasured
        gate["check_exit_code"] = exit_code
        if unmeasured:
            gate["refuse"] = True
            gate["verdict"] += (
                f"\nREFUSED: no declared or measured VRAM budget for card(s) "
                f"{unmeasured}; the split estimate cannot be verified.")
        from woof.core.devices import describe_split
        sentence = "go: " + describe_split(exp, halo=estimate["halo"])
        if args.json:
            print(json.dumps({**gate, "plan": sentence,
                              "output_disk": output_disk,
                              "prepared_urban_columns": urban_columns}, indent=2))
        else:
            print(sentence)
            print(gate["verdict"])
            print(f"Forecast output disk: {output_disk['total_bytes'] / GIB:.2f} GiB "
                  "for configured history and checkpoints; "
                  "source downloads and preparation excluded.")
        if exit_code == 2:
            print("woof check: REFUSED (exit 2, fail-closed): no declared "
                  f"or measured VRAM budget for card(s) {unmeasured}. "
                  "Declare the card budget and re-run with "
                  "--budget-gib <N> --vram-gib <card GiB>.", file=sys.stderr)
        return exit_code
    reserve = ReservePolicy.n0_alloc(
        exp, profile=profile, estimate_bytes=estimate_experiment(
            exp, forcing_intervals=resident_intervals, column_chunk=chunk,
            forcing_interval_seconds=forcing_interval,
            vram_gib=card_total_gib, profile=profile,
            boundary_species=boundary_species, urban_columns=urban_columns,
            **single_card_residue_terms(exp)
        ).alloc_estimate_bytes)
    if args.reserve_gib is not None:
        # Flat controller-ratified reserve replacing the proposed stack.
        reserve = ReservePolicy.flat(int(args.reserve_gib * GIB))

    report = None
    abort = None
    if args.alloc:
        try:
            report = run_alloc_preflight(
                exp, column_chunk=chunk, forcing_intervals=resident_intervals,
                forcing_interval_seconds=forcing_interval,
                reserve=reserve, profile=profile,
                boundary_species=boundary_species)
        except (PreflightHeadroomError, PreflightAllocError) as exc:
            abort = exc
        if report is not None:
            estimate = report.estimate
            measured_used = report.pool_used_peak_bytes
            free = report.free_before_bytes
            physical_total_bytes = report.total_bytes
            gates = report.gates
        else:
            # Aborted before measurement: still report the estimate side
            # (F1 fix) -- the measured legs stay None and can never pass.
            estimate = estimate_experiment(
                exp, forcing_intervals=resident_intervals, column_chunk=chunk,
                forcing_interval_seconds=forcing_interval,
                vram_gib=card_total_gib, profile=profile,
                boundary_species=boundary_species,
                **single_card_residue_terms(exp))
            measured_used = None
            free = getattr(abort, "free_bytes", None)
            physical_total_bytes = getattr(abort, "total_bytes", None)
            gates = evaluate_alloc_gates(
                measured_used_bytes=None,
                estimate_bytes=estimate.alloc_estimate_bytes,
                measured_free_bytes=free, reserve=reserve)
    else:
        estimate = estimate_experiment(
            exp, forcing_intervals=resident_intervals, column_chunk=chunk,
            forcing_interval_seconds=forcing_interval,
            vram_gib=card_total_gib, profile=profile,
            boundary_species=boundary_species, urban_columns=urban_columns,
            **single_card_residue_terms(exp))
        measured_used = None
        free = None
        if declared_memory:
            # CPU-mode DECLARED budget: the caller states the budget and
            # the reserve is added back to recover a notional free
            # figure.  It is arithmetic, not a measurement, and it must
            # never print under the same label as one -- the wizard's
            # inline check reported "measured free 19.31 GiB" on a
            # machine with 11.44 GiB free, which is exactly the kind of
            # number a user then trusts.
            # The wizard knows free VRAM before either allocation or
            # envelope reserves. Preserve that number instead of adding
            # an allocation reserve to an already reduced envelope budget.
            free = (int(declared_free_gib * GIB) if declared_free_gib is not None
                    else int(args.budget_gib * GIB) + reserve.reserve_bytes)
            declared_option = "--free-gib" if declared_free_gib is not None else "--budget-gib"
            free_source = f"declared ({declared_option})"
            if sampled is not None:
                free_source = "measured (shared sizing sample)"
            # ...and, once a card is named, it is capped by THAT card --
            # the declared one, and only it.  The arithmetic above knows
            # the budget but not the capacity, so on its own it can and
            # did synthesise a free figure larger than the whole card.
            #
            # This box's own physical total is NOT a second ceiling here.
            # --budget-gib is explicitly a "size for a machine that is not
            # this one" flag, and clamping a declared 24 GiB target to a
            # 16 GB card under the desk is how the wizard's own inline
            # check came to refuse a config it had just sized correctly:
            # the check reported a budget of 11.14 GiB for a 24 GiB
            # target, which is this machine's number, not the target's.
            if card_total_gib is not None:
                free, capped_to = cap_free_to_physical(
                    free, card_total_bytes=int(card_total_gib * GIB),
                    measured_total_bytes=None)
            if capped_to is not None:
                free_source = (f"declared ({declared_option}), capped at the "
                               "card's physical total")
        else:
            try:
                import cupy as cp
                device_free, device_total = cp.cuda.runtime.memGetInfo()
                free = int(device_free)
                physical_total_bytes = int(device_total)
                free_source = "measured"
                # ...and never MORE than the card actually has free.
                # ``memGetInfo`` answers "free if every other process
                # were evicted", which under WDDM it can be: measured
                # 2026-08-20 on a loaded RTX 3080 desktop, four
                # consecutive samples, memGetInfo said 9,097 MiB free
                # while NVML said 3,375-3,405 -- 5.7 GiB of a 10 GiB
                # card.  Spending that is spending a desktop's memory.
                # CUDA_VISIBLE_DEVICES may reorder CUDA ordinals relative
                # to nvidia-smi. Both NVML capacity and residency must belong
                # to the current CUDA device, just as in Machine.detect.
                device_id = cp.cuda.Device().pci_bus_id
                free, device_wide_capped = cap_free_to_device_wide(
                    free, device_id=device_id)
                if device_wide_capped:
                    free_source = ("measured machine-wide (the CUDA "
                                   "runtime reported more, counting "
                                   "memory the driver would have to "
                                   "evict from other processes)")
            except Exception:
                free = None
            if free is not None and card_total_gib is not None:
                # Sizing for a SMALLER card than the one under the desk is
                # a legitimate use of --vram-gib, and it must not inherit
                # this box's headroom.
                free, capped_to = cap_free_to_physical(
                    free, card_total_bytes=int(card_total_gib * GIB),
                    measured_total_bytes=None)
                if capped_to is not None:
                    free_source = ("measured, capped at the declared "
                                   "card's physical total")
        from woof.core.device_probe import sizing_free_memory
        if free is not None:
            lease_free, lease_note = sizing_free_memory(free)
            if lease_note:
                free = min(free, lease_free) if declared_memory else lease_free
                free_source = lease_note
        gates = evaluate_alloc_gates(
            measured_used_bytes=None,
            estimate_bytes=estimate.alloc_estimate_bytes,
            measured_free_bytes=free, reserve=reserve)

    # The whole-machine rail, when one is configured, is an ADDITIONAL
    # ceiling on top of measured free VRAM -- never a replacement and never a
    # widening.  ``free`` becomes the smaller of what the driver will hand
    # out and what the rail leaves after every other process on the card.
    rail = None
    if rail_bytes is not None:
        rail_free = device_rail_free_bytes(
            rail_bytes, other_process_bytes=other_process_bytes)
        rail = {"rail_bytes": rail_bytes,
                "other_process_bytes": other_process_bytes,
                "rail_free_bytes": rail_free}
        free = rail_free if free is None else min(int(free), rail_free)
        gates = evaluate_alloc_gates(
            measured_used_bytes=measured_used,
            estimate_bytes=estimate.alloc_estimate_bytes,
            measured_free_bytes=free, reserve=reserve)

    budget = None if free is None else reserve.budget_bytes(free)
    #: A reserve larger than free VRAM leaves NO budget, not a negative
    #: capacity.  ``budget = free - reserve`` is unbounded below, and a
    #: 4000x4000 config drove it to -7.15 GiB, which the report then
    #: printed as a figure to compare an envelope against.  Clamp at
    #: zero and say what happened, in the report, once.
    budget_underwater_bytes = 0
    if budget is not None and budget < 0:
        budget_underwater_bytes = -budget
        budget = 0
    forecast_envelope = estimate.peak_envelope_bytes
    # PREPROCESSING IS A PHASE TOO.  Reporting only the forecast is what
    # let a 12 GB-sized domain download 81 GFS files and then OOM in
    # ingest at 15.82 GB.  A config whose source this estimator cannot
    # price says so; it never silently reports the forecast as the peak.
    # ``ingest_source`` is read above, beside the boundary species.
    #: THE FORCING FILES, not the experiment: the source mesh and the
    #: number of valid times the decoder will hold.  ``None`` whenever this
    #: command has no catalog to read them off, and the ingest section then
    #: prices no host bytes and says so.
    host_geometry = None if prepared is not None else ingest_host_geometry(args)
    from woof.core.streaming import planner_machine

    phases = estimate_phases(
        exp, source=None if prepared is not None else ingest_source,
        column_chunk=chunk, boundary_source=priced_boundary_source,
        urban_columns=urban_columns,
        preparation_route=config_preparation_route(args.config),
        forcing_intervals=forcing_intervals,
        resident_forcing_intervals=resident_slots,
        ingest_forcing_interval_seconds=ingest_interval,
        forcing_interval_seconds=forcing_interval,
        vram_gib=card_total_gib, profile=profile,
        source_grid_points=(None if host_geometry is None
                            else host_geometry[0]),
        decoded_valid_times=(None if host_geometry is None
                             else host_geometry[1]),
        source_fields_per_time=(None if host_geometry is None
                                else host_geometry[2]),
        # THE CARD THIS REPORT IS ABOUT, and not the one printing it.
        # ``mode = "auto"`` with no pinned tiling is the planner's
        # decision, and asked with no Machine the planner reaches for
        # ``Machine.detect`` -- which reads whatever card is under the
        # desk, or fails outright with no CuPy and silently leaves the
        # RESIDENT estimate standing.  ``free`` here is either measured
        # off this card or derived from a ``--budget-gib`` naming a card
        # that is not in this machine at all; both are the card the
        # reader asked about, and neither costs this process a CUDA
        # context.  ``woof go``'s gate builds the same Machine from its
        # out-of-process probe, through the same function.
        # ...CARRYING THIS REPORT'S OWN DEVICE PROFILE.  The admission
        # estimate prices its non-pool terms off the machine, so a review
        # machine with no profile priced a different envelope from the run
        # door's ``Machine.detect``, which always has one.  ``profile``
        # here is this card's when the card was read and the reference
        # profile when a declared budget names another box; either way it
        # is the one every other number on this page was priced against.
        machine=(target_machine if target_hardware else
                 planner_machine(vram_bytes=free, name="woof check budget",
                                 device_profile=profile)))
    #: AN UNPRICED INGEST LANE COSTS THE INGEST SECTION, NOT THE PHASE
    #: ESTIMATE.  This used to be ``phases = None``, which threw away the
    #: streamed forecast term along with the ingest one -- and the streamed
    #: term has nothing to do with the source.  Every ``[tiles]`` config
    #: forced to a source outside :data:`SOURCE_ANALYSIS_LEVELS` was then
    #: reported, and refused, on a resident envelope describing a run that
    #: was not going to happen: measured on a 550x550x49 config with 200x200
    #: tiles, 14.30 GiB reported where the run holds 6.71 GiB, exit 4 from
    #: the envelope and exit 1 from the alloc gate.  The section's ABSENCE
    #: is what gets reported now (:func:`unpriced_ingest_note`), which is
    #: the gap that is real.
    ingest_priced = phases.ingest_priced
    ingest_note = (
        "preparation is still producing boundary intervals; this check "
        "prices the verified prepared-head forecast only"
        if prepared is not None and prepared.prepared_head_sha256 else
        "preprocessing already completed; this check prices the verified "
        "prepared forecast only" if prepared is not None else
        unpriced_ingest_note(args.config, ingest_source))
    #: The envelope every verdict below compares: the largest phase, not
    #: whichever phase happens to be modelled.
    envelope = phases.peak_envelope_bytes
    binding_phase = phases.binding_phase
    #: What the ENVELOPE is compared against.  NOT ``budget``: that is
    #: the allocation gate's budget and it has already subtracted the
    #: CUDA context and the local-memory backing store, which
    #: ``peak_envelope_bytes`` carries as its intercept.  Comparing the
    #: two charged one process for its own non-pool residency twice --
    #: 2.91 GiB of a 10 GiB card, on the walk that opened task 206 -- and
    #: warned that a configuration would not fit a card it fits.
    #:
    #: The envelope models the whole device residency this process
    #: reaches, so what is left outside it is other processes, which is
    #: :data:`EXTERNAL_MARGIN_BYTES`.  Same seam the wizard sizes with
    #: (:func:`woof.domain_wizard.sizing_budget_bytes`), so the door
    #: that emits a config and the door that verifies it cannot disagree.
    envelope_budget = (None if free is None
                       else max(0, int(free) - EXTERNAL_MARGIN_BYTES))
    phase_verdict = (
        phases._budget_tail(
            f"the verified prepared forecast needs {envelope / GIB:.2f} "
            f"GiB peak envelope; {ingest_note}",
            envelope_budget)
        if prepared is not None and not phases.streamed_forecast else
        phases.verdict(envelope_budget))
    #: The report's own prose says this configuration may not fit.  It is
    #: read here, before either renderer, because the exit code has to
    #: carry it whether or not anybody reads the text.
    envelope_over_budget = (envelope_budget is not None
                            and envelope > envelope_budget)
    #: THE OTHER MEMORY.  Every figure above this line is device memory;
    #: the ingest phase also decodes the forcing into HOST RAM, holds two
    #: copies of it at once while the root case is prepared, and no door
    #: in this product has ever priced that.  It is the failure this section could
    #: not have caught even in principle: a host OOM is delivered from
    #: outside the process, so a run that dies of it prints nothing at all.
    #:
    #: ``None`` for both is the accurate answer on a config whose forcing
    #: this command cannot see, or a box whose RAM it cannot read; the
    #: comparison is then omitted rather than guessed.
    host_forcing_bytes = (phases.ingest.host_forcing_bytes
                          if ingest_priced else None)
    #: READ BEFORE THIS COMMAND DECODED ANYTHING, when the input preflight
    #: that runs first left it here.  That half of ``woof check`` decodes
    #: the forcing into caches nothing clears, so this process is already
    #: holding the bytes being priced: measuring the headroom now would
    #: subtract them from the rail they are being compared against and
    #: refuse a run that fits.  Falling back to a fresh read is correct
    #: for the callers that reach this section without that half, because
    #: nothing has decoded anything on those paths either.
    # A transported host snapshot measures total RAM for tile planning, not
    # currently available RAM. The desktop's available memory cannot fill in
    # that unknown for a different target.
    host_available = (None if target_hardware else
                      getattr(args, "host_available_at_entry", None))
    if host_available is None and not target_hardware:
        host_available = host_available_bytes()
    host_over_available = (host_forcing_bytes is not None
                           and host_available is not None
                           and host_forcing_bytes > host_available)
    #: THE WAY THROUGH.  ``MemAvailable`` is a reading of this second, so a
    #: busy workstation, a shared login node or a box beside another job
    #: can be momentarily short of RAM a run would have had; and the figure
    #: on the other side is a floor over a decoder, not a measurement of
    #: this run.  The product's other memory refusal already has an
    #: override -- ``woof go --no-memory-gate`` -- and a refusal with no
    #: way past it is one that gets worked around by not running the check
    #: at all.  Spelled for the budget it skips, because every other gate
    #: in this command is about the card and ``--no-memory-gate`` here
    #: would read as all of them.
    host_gate_skipped = bool(getattr(args, "no_host_memory_gate", False))
    host_refused = host_over_available and not host_gate_skipped
    #: THE PREPARATION'S OWN HOST TERM, on the CPU road.  There the ingest
    #: working set is host RAM and its device envelope is zero, so no card
    #: gate above sees it; weighed against the RAM of the machine this
    #: report prices, the same comparison the wizard sizes against and
    #: ``woof go`` refuses on.  With no card figure there is no planner
    #: machine, and this box's RAM is read directly, as ``woof go`` does;
    #: a declared target's RAM is never replaced by this box's.  Unknown
    #: RAM never refuses.
    preparation_host = phases.host_ram_bytes
    if preparation_host is None and not target_hardware:
        from woof.core.streaming import _host_total_bytes
        preparation_host = _host_total_bytes()
    preparation_host_refusal = phases.host_preparation_refusal(
        preparation_host)
    preparation_host_warning = phases.host_preparation_warning(
        preparation_host)
    preparation_refused = (preparation_host_refusal is not None
                           and not host_gate_skipped)
    #: THE STREAMED FORECAST'S HOST RAM, the admission ``woof go`` refuses
    #: on before the download.  This report printed the figure and its
    #: budget and passed regardless, so a configuration whose forecast
    #: needed more host RAM than the budget allows passed here and was
    #: refused by the run door.  Skipped with the other host gates.
    streamed_host_refusal = phases.streamed_host_refusal()
    streamed_host_refused = (streamed_host_refusal is not None
                             and not host_gate_skipped)
    #: THE ALLOC GATE PRICES THE RUN THE CONFIG ASKS FOR.
    #:
    #: Every leg above was fed ``estimate.alloc_estimate_bytes``, which
    #: itemizes a domain RESIDENT in VRAM.  Under ``[tiles]`` that domain
    #: is never allocated -- the card holds ``nbuffers`` tile buffers and
    #: the domain lives in pinned host RAM -- so the leg was refusing, at
    #: exit 1, the one configuration class streaming exists to enable.
    #: Measured on the 550x550x49 / 200x200 fixture: 11.35 GiB of resident
    #: pool request weighed against a budget the 6.71 GiB streamed run fits
    #: with room to spare.
    #:
    #: COMPARED AS AN ENVELOPE, AGAINST THE ENVELOPE BUDGET, and that pair
    #: is the decision rather than a convenience.
    #: :attr:`StreamedEnvelope.vram_bytes` is envelope-shaped by
    #: construction -- CUDA context, the rung's per-process fixed cost and
    #: the tile buffers, with autoplan's safety factor -- and there is no
    #: alloc-shaped figure to be had from it.  ``Footprint`` lumps the
    #: pool-resident k-distribution tables in with the non-pool module
    #: images inside one measured ``process_fixed_bytes``, so any
    #: subtraction that made it alloc-shaped would be inventing a number.
    #: Both candidate subtractions were priced on this fixture and both
    #: are wrong in a way that matters: taking off the whole
    #: ``process_overhead_bytes`` leaves 2.95 GiB against a run that really
    #: holds 6.71, an UNDER-charge of 1.31 GiB that would have the gate
    #: admit an OOM; taking off only the CUDA context leaves 6.29 GiB and
    #: charges the 2.03 GiB local-memory backing store a second time, on
    #: top of the reserve that already holds it -- the task-206 double
    #: count this file spent a release removing.
    #:
    #: So the streamed figure is compared against ``envelope_budget``
    #: (free VRAM less the other-process margin), which is the same pair
    #: ``woof go``'s memory gate admits on and the same pair the
    #: over-budget WARNING below prints.  The leg's contract --
    #: "the estimate fits the budget" -- is unchanged; what it prices is.
    #:
    #: ``--alloc`` is excluded on purpose: ``run_alloc_preflight``
    #: constructs the RESIDENT domain on the real card, so its measured
    #: legs describe a resident allocation, and swapping the estimate leg
    #: underneath them would have ``alloc_measured_le_estimate`` compare a
    #: resident measurement against a streamed estimate and fail for a
    #: reason that is not about this configuration at all.
    streamed_alloc_gate = (phases.streamed_forecast and not args.alloc
                           and envelope_budget is not None)
    if streamed_alloc_gate:
        gates = dict(gates)
        # THE RADIATION PEAK, not the steady hold: the leg admits a run,
        # and a run admitted on the hold meets the RRTMGP transient at
        # itimestep == 1.  See StreamedEnvelope.peak_vram_bytes.
        gates["alloc_estimate_le_wddm_budget"] = (
            int(phases.streamed.peak_vram_bytes) <= envelope_budget)
    if args.json:
        payload = {
            "config": str(args.config), "experiment": exp.name,
            "output_disk": output_disk,
            "preprocess_backend": phases.preprocess_backend,
            "gpu_readiness": readiness,
            "column_chunk": estimate.column_chunk,
            "domains": {
                f"d{d.grid_id:02d}": {
                    "resident_bytes": d.resident_bytes,
                    "arena_scratch_request_bytes": d.arena_scratch_bytes,
                    "transient_bytes": d.transient_bytes,
                    "by_category": {c: d.category_bytes(c) for c in
                                    ("state", "physics", "scratch", "lbc",
                                     "nest", "diagnostic", "sase", "transient")},
                } for d in estimate.domains},
            "k_tables_bytes": estimate.k_tables_bytes,
            "physics_tables_bytes": estimate.physics_tables_bytes,
            "workspace_bytes": estimate.workspace_bytes,
            "scratch_arena_bytes": estimate.scratch_arena_bytes,
            "scratch_arena_request_bytes":
                estimate.scratch_arena_request_bytes,
            "scratch_arena_saved_bytes": estimate.scratch_arena_saved_bytes,
            "uses_shared_scratch_arena":
                estimate.uses_shared_scratch_arena,
            "dycore_state_workspace_bytes":
                estimate.dycore_state_workspace_bytes,
            "dycore_state_request_bytes":
                estimate.dycore_state_request_bytes,
            "dycore_state_saved_bytes": estimate.dycore_state_saved_bytes,
            "uses_shared_dycore_state_workspace":
                estimate.uses_shared_dycore_state_workspace,
            "resident_bytes": estimate.resident_bytes,
            "transient_peak_bytes": estimate.transient_peak_bytes,
            "subtotal_bytes": estimate.subtotal_bytes,
            "alloc_estimate_bytes": estimate.alloc_estimate_bytes,
            "held_projection_bytes": estimate.held_projection_bytes,
            "footprint_projection_bytes":
                estimate.footprint_projection_bytes,
            "observed_peak_envelope_platform":
                envelope_platform(vram_gib=card_total_gib),
            "observed_peak_envelope_bytes": forecast_envelope,
            "non_pool_device_bytes": estimate.non_pool_device_bytes,
            "envelope_unmodelled_bytes": ENVELOPE_UNMODELLED_BYTES,
            "envelope_per_nest_fraction": ENVELOPE_PER_NEST_FRACTION,
            # Both slack keys read 0.0 since A163: the term is retired
            # from the envelope, and the keys stay because 2.5.0 and
            # later receipts read them.
            "envelope_wddm_pool_slack_fraction": 0.0,
            "envelope_pool_slack_fraction": 0.0,
            "forecast_pool_headroom": estimate.headroom,
            "held_exact_bytes": estimate.held_exact_bytes,
            "bem_column_workspace_basis": _check_bem_basis(
                estimate, prepared, alloc=args.alloc),
            "prepared_root": (None if prepared is None else
                              str(prepared.prepared_root)),
            "prepared_head_sha256": (None if prepared is None else
                                     prepared.prepared_head_sha256),
            "prepared_urban_columns": urban_columns,
            "forcing_interval_seconds": forcing_interval,
            "retained_forcing_intervals": forcing_intervals,
            "forecast_peak_ratio_measured": FORECAST_PEAK_RATIO_MEASURED,
            "forecast_peak_ratio_safety": FORECAST_PEAK_RATIO_SAFETY,
            "envelope_legacy_radiation": estimate.uses_legacy_radiation,
            "envelope_basis": estimate.envelope_basis,
            "local_memory_profile": estimate.non_pool_device_bytes and
                profile.name,
            "peak_envelope_bytes": envelope,
            "binding_phase": binding_phase,
            "reserve_bytes": reserve.reserve_bytes,
            "reserve_components": {
                "retention_residual_bytes":
                    reserve.retention_residual_bytes,
                "device_overhead_bytes": reserve.device_overhead_bytes,
                "external_margin_bytes": reserve.external_margin_bytes,
            },
            "run_time_reserve_bytes": ReservePolicy.run_time(
                exp).reserve_bytes,
            "pool_reserved_over_estimate_fraction":
                POOL_RESERVED_OVER_ESTIMATE_FRACTION,
            "cuda_context_bytes": CUDA_CONTEXT_BYTES,
            # PRICED AGAINST THE DEVICE THIS REPORT IS ABOUT.  Without
            # the profile this defaulted to the 170-SM reference while
            # the sibling "local_memory_profile" field named the real
            # card, so the two disagreed: a 70-SM RTX 5070 Ti reported
            # 5.20 GiB where its own profile gives 2.14, and the
            # difference propagated into the envelope as a spurious
            # over-budget warning.
            "kernel_local_memory_bytes": kernel_local_memory_bytes(
                exp, profile=profile),
            "kernel_modules": sorted(physics_kernel_modules(exp)),
            # THE BASIS IN WORDS, the same sentence the text report prints
            # as NON-POOL BASIS: which card the grid-independent terms
            # were priced on and, for a Noah-MP configuration, whether its
            # frames were measured on this card's compile platform or
            # taken from the ceiling over the recorded platforms.
            "non_pool_basis": non_pool_basis(profile, exp),
            "measured_free_bytes": free,
            "physical_total_bytes": physical_total_bytes,
            "declared_capacity_bytes": (None if card_total_gib is None else int(card_total_gib * GIB)),
            "capacity_source": None if card_total_gib is None else capacity_source,
            "free_bytes_source": free_source,
            # A declared budget sizes hardware that is not in this
            # machine; every figure in this report is then an ESTIMATE
            # for hardware not present, priced against the conservative
            # measured reference profile above -- never a measurement.
            "sized_for_hardware_not_present": declared_memory and sampled is None,
            "free_bytes_capped_to_physical_bytes": capped_to,
            "budget_bytes": budget,
            "budget_underwater_bytes": budget_underwater_bytes,
            "envelope_budget_bytes": envelope_budget,
            "observed_peak_envelope_exceeds_budget": (
                None if envelope_budget is None else envelope_over_budget),
            "gates": gates,
            # THE VERDICT THIS SECTION REACHED, as a word rather than as
            # something a reader has to reconstruct from three legs and an
            # exit code.  "incomplete" is the state the reduction used to
            # have no name for.
            "memory_verdict": memory_gate_verdict(gates),
            "gates_evaluated": sum(
                1 for metric in N0_GATE_METRICS if gates[metric] is not None),
            "gates_total": len(N0_GATE_METRICS),
            "gates_absent": list(absent_gate_metrics(gates)),
            "host_available_bytes": host_available,
            "ingest_host_forcing_bytes": host_forcing_bytes,
            "ingest_host_forcing_exceeds_available": (
                None if host_forcing_bytes is None or host_available is None
                else host_over_available),
            "host_memory_gate_skipped": host_gate_skipped,
            "ingest_host_preparation_bytes": phases.host_preparation_bytes,
            "ingest_host_preparation_floor_bytes":
                phases.host_preparation_floor_bytes,
            "host_ram_bytes": preparation_host,
            "ingest_host_preparation_refusal": preparation_host_refusal,
            "ingest_host_preparation_warning": preparation_host_warning,
            "streamed_host_refusal": streamed_host_refusal,
        }
        # WHICH FORECAST FIGURE THE READER GOT, said in a field rather
        # than inferred from the size of the number.
        payload["streamed_forecast"] = phases.streamed_forecast
        # THE RUN DOOR'S OWN PER-DOMAIN WALK, published whether or not it
        # replaced the forecast term: a reader whose tree the walk refuses,
        # or leaves all-resident, is owed the same answer as one whose tree
        # streams.  ``None`` for every config the question does not arise
        # for (single domain, no [tiles] anywhere).
        payload["tree_road"] = (None if phases.tree_road is None
                                else phases.tree_road.to_json())
        if phases.streamed_forecast:
            env = phases.streamed
            payload["streamed"] = {
                # THE HOLD, kept because it is the figure an NVML
                # steady-state reading can be compared against...
                "vram_bytes": int(env.vram_bytes),
                # ...and the two terms that make the PEAK, which is what
                # peak_envelope_bytes above carries and every gate weighs.
                "radiation_transient_bytes":
                    int(env.radiation_transient_bytes),
                "peak_vram_bytes": int(env.peak_vram_bytes),
                "resident_forecast_envelope_bytes":
                    phases.resident_forecast_envelope_bytes,
                "host_bytes": int(env.host_bytes),
                # host_bytes' lateral forcing series (ordinary host RAM,
                # beside the pinned store); on a tree's mixed road, the
                # root's, and zero when the root is resident.
                "boundary_table_bytes":
                    int(getattr(env, "boundary_table_bytes", 0) or 0),
                # The named terms the figures above add up from, so a
                # report carries the arithmetic and not only its total.
                "terms": {str(k): v for k, v in getattr(env, "terms", ())},
                # The pair the alloc leg above compared, named, so a
                # script never has to guess which budget it was.
                "alloc_gate_basis": (
                    "the streamed envelope against envelope_budget_bytes"
                    if streamed_alloc_gate else
                    "resident (--alloc measures a resident allocation)"),
            }
            if phases.mixed_road:
                # A TREE HAS NO SINGLE TILING.  These keys describe one
                # streamed domain, and emitting them for a mixed road would
                # have a script read the child's tile as the tree's -- so
                # the road field says which shape this block is, and the
                # per-domain tilings live in ``tree_road.rows`` above.
                payload["streamed"]["road"] = "mixed (nested tree)"
            else:
                payload["streamed"].update({
                    "road": "streamed (single domain)",
                    "host_budget_bytes": env.host_budget_bytes,
                    "tile_nx": env.tile_nx, "tile_ny": env.tile_ny,
                    "window_nx": env.window_nx, "window_ny": env.window_ny,
                    "nbuffers": env.nbuffers, "halo": env.halo,
                    "rung": env.rung, "write_mode": env.write_mode,
                })
        # The verdict compares ``envelope_budget``, not ``budget``: it is
        # an ENVELOPE sentence, and the allocation budget has already
        # subtracted the CUDA context and the local-memory backing store
        # that the envelope carries as its intercept.  This field said
        # otherwise while the printed line beside it said this, which is
        # one report giving two answers about one card (task 206).
        payload["phase_verdict"] = phase_verdict
        if not ingest_priced:
            payload["ingest"] = None
            payload["ingest_not_priced_reason"] = ingest_note
        else:
            ingest = phases.ingest
            payload["ingest"] = {
                "source": ingest_source,
                "preprocess_backend": ingest.preprocess_backend,
                "allocation_memory": "host" if ingest.preprocess_backend == "cpu" else "device",
                "forcing_times": ingest.n_forcing_times,
                "resident_forcing_times": ingest.resident_times,
                "per_forcing_time_bytes": ingest.per_time_bytes,
                "analysis_bytes": ingest.category_bytes("analysis"),
                "state_bytes": ingest.category_bytes("state"),
                "forcing_table_bytes": ingest.forcing_table_bytes,
                "resident_bytes": ingest.resident_bytes,
                "nest_state_bytes": ingest.nest_state_bytes,
                "nest_state_by_grid": {
                    f"d{grid:02d}": nbytes
                    for grid, nbytes in ingest.nest_state_items},
                "widest_domain_time_bytes":
                    ingest.widest_domain_time_bytes,
                "transient_bytes": ingest.transient_bytes,
                "subtotal_bytes": ingest.subtotal_bytes,
                "unstreamed_resident_bytes":
                    ingest.unstreamed_resident_bytes,
                "boundary_frame_host_bytes": ingest.boundary_frame_bytes,
                # HOST, and named so no reader takes it for a device
                # figure.  ``None`` when the forcing files were not
                # visible to this command; the terms beside it say what
                # the figure would have been made of.
                "host_forcing_bytes": ingest.host_forcing_bytes,
                "host_preprocess_bytes": ingest.host_preprocess_bytes,
                "host_peak_estimate_bytes": ingest.host_peak_estimate_bytes,
                "host_source_grid_points": ingest.source_grid_points,
                "host_fields_per_time": ingest.host_fields_per_time,
                "host_decoded_valid_times": ingest.decoded_valid_times,
                "host_retained_copies": ingest.host_retained_copies,
                "alloc_estimate_bytes": ingest.alloc_estimate_bytes,
                "peak_envelope_bytes": ingest.peak_envelope_bytes,
                "context_bytes": ingest.context_bytes,
                "peak_envelope_basis": (
                    "CPU preprocessing: no device allocations; host working set and decoder reported separately"
                    if ingest.preprocess_backend == "cpu" else INGEST_PEAK_ENVELOPE_BASIS),
            }
        # The same document ``run-plan --estimate`` publishes, on the
        # surface that has actually measured the card: a machine-facing
        # reader of `woof check --json` gets the pace as fields rather
        # than having to parse it back out of the advisory sentence.
        pace = pace_estimate_for_report(exp, streamed=phases.pace_streamed,
                                        free_bytes=free)
        payload["expected_pace"] = None if pace is None else pace.to_json()
        advisories = check_advisories(exp, args.config,
                                      streamed=phases.streamed,
                                      tree_road=phases.tree_road)
        if advisories:
            payload["advisories"] = advisories
        if rail is not None:
            payload["device_rail"] = rail
        if abort is not None:
            payload["abort"] = {
                "error": type(abort).__name__,
                "phase": abort.phase,
                "reason": str(abort),
                "free_bytes": getattr(abort, "free_bytes", None),
            }
        if report is not None:
            payload["alloc"] = {
                "pool_used_peak_bytes": report.pool_used_peak_bytes,
                "pool_held_peak_bytes": report.pool_held_peak_bytes,
                "device_footprint_bytes": report.device_footprint_bytes,
                "measured_overhead_bytes": report.measured_overhead_bytes,
                "free_after_release_bytes":
                    report.free_after_release_bytes,
            }
        print(json.dumps(payload, indent=2))
    elif getattr(args, "explain", False):
        print(f"Physical GPU capacity: {_format_bytes(physical_total_bytes).strip()} "
              + ("(measured)" if physical_total_bytes is not None else "(not measured)"))
        if card_total_gib is not None:
            print(f"Target GPU capacity: {card_total_gib:g} GiB ({capacity_source})")
        if readiness["status"] == "verified":
            print("woof GPU readiness: PASS (cold compile and execution).")
        else:
            print("woof GPU readiness: NOT CHECKED (declared-budget estimate).")
        print(f"woof check: memory preflight for {exp.name!r} "
              f"({len(exp.domains)} domain(s); column_chunk "
              f"{estimate.column_chunk})")
        # THIS SECTION'S OWN VERDICT, in this section's own words.  It had
        # none: the only verdict word on the page came from
        # ``woof/ingest/preflight.py``, about nineteen CPU file/time/table
        # checks, and a reader applied it to the memory report below it
        # because nothing here said otherwise.  Three-valued, so the state
        # the N0 chain is actually in on every invocation without
        # ``--alloc`` -- two legs absent -- has a name it can be printed
        # under instead of being reduced away.
        evaluated = sum(1 for metric in N0_GATE_METRICS
                        if gates[metric] is not None)
        print(f"woof memory preflight: "
              f"{memory_gate_verdict(gates).upper()} "
              f"({evaluated} of {len(N0_GATE_METRICS)} allocation gates "
              f"evaluated)")
        for advisory in check_advisories(
                exp, args.config, streamed=phases.streamed,
                tree_road=phases.tree_road):
            print(f"  {advisory}")
        # THE PLAN, PER DOMAIN, because the sentence above states the road
        # and this states its arithmetic.  Printed for every nested tree
        # the walk could price -- including one it refuses, where the
        # roads it got as far as deciding are what the reader needs to see
        # -- and never gated behind a flag: a user whose mixed road fits
        # was being handed exit 1 and no way to see why.
        if phases.tree_road is not None:
            print("  MIXED-ROAD PLAN (the per-domain walk the run door "
                  "performs; streaming.steppers_for_tree):")
            for line in phases.tree_road.row_lines():
                print(f"    {line}")
            if phases.tree_road.total_budget_bytes:
                configured_bound = (phases.tree_road.configured_mixed_envelope_bytes
                    > phases.tree_road.vram_hold_bytes + phases.tree_road.radiation_transient_bytes)
                peak_basis = (" after configured resident/global and tile obligations"
                              if configured_bound else " with the radiation reservation")
                print(f"    tree budget {_format_bytes(int(phases.tree_road.total_budget_bytes))}"
                      f"; process floor "
                      f"{_format_bytes(int(phases.tree_road.process_overhead_bytes))}"
                      f"; card holds "
                      f"{_format_bytes(int(phases.tree_road.vram_hold_bytes))}"
                      f", peak "
                      f"{_format_bytes(int(phases.tree_road.peak_vram_bytes))}"
                      f"{peak_basis}")
            if phases.tree_road.refusal is not None:
                print(f"    REFUSED: {phases.tree_road.refusal}")
        # UNCONDITIONAL, unlike the advisories above.  This is the line
        # whose absence let a streamed run at 399,119 columns look like a
        # stall: the memory report was complete and said nothing about
        # what a step costs.  Priced against the card THIS REPORT just
        # measured, so the column bound is about the reader's own
        # allowance and not a declared one.
        pace_line = pace_advisory(
            exp, streamed=phases.pace_streamed,
            machine=pace_machine_from_free_bytes(free))
        if pace_line:
            print(f"  {pace_line}")
        for d in estimate.domains:
            cats = ", ".join(
                f"{c} {d.category_bytes(c) / GIB:.3f}"
                for c in ("state", "physics", "scratch", "lbc", "nest",
                          "diagnostic", "sase")
                if d.category_bytes(c))
            print(f"  d{d.grid_id:02d}: resident "
                  f"{d.resident_bytes / GIB:6.2f} GiB ({cats}); step "
                  f"transients {d.transient_bytes / GIB:.2f} GiB")
        print(f"  shared: k-tables {estimate.k_tables_bytes / GIB:.3f} GiB; "
              f"chunk workspace {estimate.workspace_bytes / GIB:.2f} GiB; "
              f"scratch arena {estimate.scratch_arena_bytes / GIB:.2f} GiB "
              f"(saves {estimate.scratch_arena_saved_bytes / GIB:.2f} GiB); "
              f"dycore state {estimate.dycore_state_workspace_bytes / GIB:.2f} "
              f"GiB (saves {estimate.dycore_state_saved_bytes / GIB:.2f} GiB)")
        print(f"  TIER 1  resident: {_format_bytes(estimate.resident_bytes)}"
              f"   subtotal (+workspace+transient peak): "
              f"{_format_bytes(estimate.subtotal_bytes)}")
        print(f"  ESTIMATE ("
              + (f"+{int(estimate.pool_residue_cap_bytes) / GIB:.2f} GiB "
                 f"measured pool residue bound (A13), under "
                 if estimate.pool_residue_cap_active else "")
              + f"x{estimate.headroom:.2f}: measured peak/itemized "
              f"{estimate.headroom - FORECAST_PEAK_RATIO_SAFETY:.2f} + safety "
              f"{FORECAST_PEAK_RATIO_SAFETY:.2f}"
              + (f", on the subtotal less "
                 f"{_format_bytes(estimate.held_exact_bytes)} of urban "
                 f"arrays held at their allocated size"
                 if estimate.held_exact_bytes else "")
              + f"): {_format_bytes(estimate.alloc_estimate_bytes)}")
        bem_note = _check_bem_basis(estimate, prepared, alloc=args.alloc)
        if bem_note is not None:
            print(f"  NOTE: {bem_note}")
        print(f"  TIER 2  pool-held projection: "
              f"{_format_bytes(estimate.held_projection_bytes)}"
              f"   TIER 3 device-footprint projection: "
              f"{_format_bytes(estimate.footprint_projection_bytes)}")
        # Same profile the estimate used (see the JSON field above):
        # this feeds both the printed backing-store figure and the
        # re-measured footprint projection, so pricing it against a
        # card that is not in the machine inflates the envelope.
        widest = kernel_local_memory_bytes(exp, profile=profile)
        gf_ws = gf_column_workspace_bytes(exp, profile=profile)
        kf_ws = kf_column_workspace_bytes(exp, profile=profile)
        ysu_ws = ysu_column_workspace_bytes(exp, profile=profile)
        shinhong_ws = shinhong_column_workspace_bytes(exp, profile=profile)
        fork_ws = thompson_fork_column_workspace_bytes(exp, profile=profile)
        initial_refl_ws = gsd_initial_reflectivity_workspace_bytes(exp, profile=profile)
        # Kept out of the f-string: a line break inside an f-string
        # expression is PEP 701 (Python 3.12+) syntax, and the supported
        # floor is 3.11 -- the 1.2.0 release workflow failed on exactly
        # this line before any wheel was published.
        # THIS CARD's context, not the retired flat constant: the two
        # numbers used to disagree on the same page, because this display
        # line kept CUDA_CONTEXT_BYTES while the envelope beside it moved
        # to the per-card term (task 206).
        context_bytes = profile.cuda_context_bytes
        remeasured_bytes = (estimate.alloc_estimate_bytes + widest
                            + gf_ws + kf_ws + ysu_ws + shinhong_ws + fork_ws + initial_refl_ws
                            + context_bytes
                            + reserve.retention_residual_bytes)
        gf_ws_term = (f" + GF column workspace {_format_bytes(gf_ws)}"
                      if gf_ws else "")
        gf_ws_term += (f" + KF column workspace {_format_bytes(kf_ws)}"
                       if kf_ws else "")
        ysu_ws_term = (f" + YSU column workspace {_format_bytes(ysu_ws)}"
                       if ysu_ws else "")
        ysu_ws_term += (" + Shin-Hong column workspace "
                        f"{_format_bytes(shinhong_ws)}"
                        if shinhong_ws else "")
        ysu_ws_term += (f" + initial GSD radar workspace {_format_bytes(initial_refl_ws)}"
                       if initial_refl_ws else "")
        print(f"  NON-POOL: CUDA context "
              f"{_format_bytes(context_bytes)} + local-memory backing "
              f"store {_format_bytes(widest)} "
              f"({len(physics_kernel_modules(exp))} kernel modules selected)"
              f"{gf_ws_term}"
              f"{ysu_ws_term}"
              f"; RE-MEASURED device-footprint projection "
              f"{_format_bytes(remeasured_bytes)}"
              " (the TIER 3 line above is the retired zero-step probe model)")
        print(f"  NON-POOL BASIS: {non_pool_basis(profile, exp)}")
        family = envelope_platform(vram_gib=card_total_gib)
        if family == "windows":
            provenance = (
                "affine, calibrated on this card class: six whole "
                "bare-default forecasts on an RTX 3080 10 GiB Windows/WDDM "
                "desktop measured machine-wide peaks of estimate + "
                "itemized non-pool within -0.20..+0.95 GiB, so the "
                "envelope is that affine sum with the measured A163 "
                "margin -- the retired 1.75 multiplier predicted 3.8x the "
                "measured peak on the same card and is gone from every "
                "gate")
        else:
            provenance = (
                "affine, not a multiplier: a multiplier with no intercept "
                "under-predicts small configurations and over-predicts "
                "large ones, which is what the 16 GiB fleet node measured "
                "(3.99 GiB declared, 4.38 measured; 19.95 declared, 13.88 "
                "measured).  The intercept is the non-pool line above, "
                "which scales with the device and the kernel set, not "
                "with the grid")
        # THE ITEMIZATION IS RESIDENT, AND SAYS SO UNDER [tiles].  Every
        # term above enumerates a domain living in VRAM; with streaming
        # configured the streamed term below replaces this figure, and a
        # reader who takes this line for the answer has the wrong number
        # by the whole point of the feature.  The advisory at the top of
        # this report promises its numbers price the STREAMED allocation:
        # the label is what keeps that promise true of this line too.
        forecast_label = ("RESIDENT FORECAST PEAK ENVELOPE (replaced by the "
                          "streamed term below)"
                          if phases.streamed_forecast
                          else "FORECAST PEAK ENVELOPE")
        print(f"  {forecast_label} ({estimate.peak_envelope_terms()}"
              f"; {estimate.envelope_basis}): "
              f"{_format_bytes(forecast_envelope)} -- {provenance}.")
        if not ingest_priced:
            print("  " + ingest_note)
        else:
            ingest = phases.ingest
            backend_label = ", backend cpu, host RAM" if ingest.preprocess_backend == "cpu" else ""
            print(f"  INGEST (preprocessing, --source {ingest_source}{backend_label}): "
                  f"root {ingest.n_forcing_times} forcing times x "
                  f"{_format_bytes(ingest.per_time_bytes)} each "
                  f"(analysis {_format_bytes(ingest.category_bytes('analysis'))}"
                  f" + state {_format_bytes(ingest.category_bytes('state'))}); "
                  f"{ingest.resident_times} resident at a time = "
                  f"{_format_bytes(ingest.resident_bytes)} resident")
            if ingest.nest_state_items:
                nested = ", ".join(
                    f"d{grid:02d} {nbytes / GIB:.2f}"
                    for grid, nbytes in ingest.nest_state_items)
                print(f"    + NESTS ({len(ingest.nest_state_items)} of them, "
                      f"one initial state each, all resident for the single "
                      f"export transaction): {nested} = "
                      f"{_format_bytes(ingest.nest_state_bytes)}")
            if ingest.preprocess_backend == "cpu":
                print(f"    INGEST GPU PEAK ENVELOPE: 0.00 GiB (CPU preparation); "
                      f"HOST preprocessing working set {_format_bytes(ingest.host_preprocess_bytes)} "
                      f"estimated at its peak, at least "
                      f"{_format_bytes(ingest.host_preprocess_floor_bytes).strip()} "
                      f"held at once ({CPU_PREPARATION_PEAK_BASIS}), "
                      f"plus the separately reported forcing decode.")
            else:
                print(f"    INGEST OBSERVED PEAK ENVELOPE "
                      f"(x{ingest.headroom:.2f} headroom + "
                      f"{_format_bytes(ingest.context_bytes)} CUDA context; "
                      f"{INGEST_PEAK_ENVELOPE_BASIS}): "
                      f"{_format_bytes(ingest.peak_envelope_bytes)}   "
                      f"[streaming; holding all "
                      f"{ingest.n_forcing_times} times would resident "
                      f"{_format_bytes(ingest.unstreamed_resident_bytes)}]")
            # HOST RAM, ON THE SAME PAGE AS THE CARD.  Every line above
            # is device memory -- this class says so in its first
            # sentence -- and the decode that feeds them is a host cost
            # that nothing in this product has ever weighed.  It is not
            # added to any total above: they are different memories, with
            # different levers, and a run can fit one and not the other.
            if ingest.host_forcing_bytes is None:
                why = ("this command has no input catalog to read them off "
                       "(a config with no [case_data] table, or a run of "
                       "this section on its own), and pricing them by "
                       "decoding would spend the memory this line exists "
                       "to weigh"
                       if ingest.host_retained_copies else
                       f"--source {ingest_source} ingests through the "
                       f"native front door, whose host retention nothing "
                       f"here has measured (priced: "
                       f"{', '.join(sorted(INGEST_HOST_RETAINED_COPIES))})")
                print(f"    HOST FORCING NOT PRICED: the decode's host "
                      f"residency is set by the SOURCE grid, the number of "
                      f"valid times in the forcing files and how many "
                      f"copies the decoder retains -- {why}.")
            else:
                print(f"    HOST FORCING (RAM, not VRAM): "
                      f"{ingest.decoded_valid_times} decoded valid times x "
                      f"{ingest.host_fields_per_time} source fields x "
                      f"{ingest.source_grid_points} points x "
                      f"{INGEST_HOST_DECODE_BYTES_PER_POINT} B float64 x "
                      f"{ingest.host_retained_copies} retained copies = "
                      f"{_format_bytes(ingest.host_forcing_bytes)}, held "
                      f"at once while the case is prepared, against "
                      f"{_format_bytes(host_available)} of RAM this box "
                      f"had available before this command ran.  A "
                      f"FLOOR: the decoder's flat message buffer and the "
                      f"second merged copy a run makes are real and not "
                      f"claimed here.")
                horizon_times = lbc_intervals(exp.run_seconds, forcing_interval) + 1
                if ingest.decoded_valid_times > horizon_times:
                    print(f"    NOTE: the forecast horizon needs "
                          f"{horizon_times} forcing times; preparation retains "
                          f"{ingest.n_forcing_times} and the "
                          f"decoder takes all "
                          f"{ingest.decoded_valid_times} in the files.  "
                          f"The catalog selects the longest contiguous run "
                          f"of valid times present in the forcing and "
                          f"never sees run_seconds, so shortening the "
                          f"forecast does not shorten the decode -- "
                          f"re-fetching a narrower window does.")
        # ``envelope_budget``, NOT ``budget``.  This one printed line
        # was the last place the task-206 double count survived: it
        # compared the machine-peak ENVELOPE -- which carries the
        # CUDA context and the local-memory backing store as its
        # intercept -- against the ALLOCATION budget, from which the
        # allocation reserve has already subtracted those same
        # bytes.  Measured 2026-08-20 on the loaded RTX 3080, 5.75
        # GiB free: it printed "5.40 GiB peak envelope; that EXCEEDS
        # the 3.79 GiB budget by 1.61 GiB" while the exit code (read
        # off ``envelope_over_budget``, which uses the line below)
        # said 0.15 GiB, and `woof go` on the same config seconds
        # later admitted it and ran to rc 0 with output byte-
        # identical to the roomy run.  A report whose sentence and
        # whose exit code disagree by 1.46 GiB of budget teaches the
        # reader to ignore one of them.
        #
        # PRINTED WHETHER OR NOT THE INGEST LANE COULD BE PRICED.  It
        # used to live inside the priced branch, so the one line that
        # names the STREAMED forecast term -- and the tiling that
        # produced it -- was withheld from exactly the configs whose
        # envelope the streaming replaced.
        print(f"  BINDING PHASE: {phase_verdict}.")
        if streamed_alloc_gate:
            # The gate leg below prices this run as it will actually be
            # allocated, and it compares a different budget from the one
            # on the reserve line: say so where the two are read, so
            # neither number reads as contradicting the other.
            # WHY THE ITEMIZED ESTIMATE IS NOT THE SUBJECT, and the reason
            # differs by road.  On a single streamed domain the resident
            # allocation never happens at all.  On a MIXED road it partly
            # does -- a resident parent really is allocated -- so saying
            # "the resident domain is never allocated" there would be
            # false about the very domain the reader can see in the plan
            # above.  What is true of both is that the walk's figure is
            # already whole-process, which is the part that matters for
            # which budget it is weighed against.
            basis = (
                "the walk above already prices every domain's road "
                "together, resident claims included"
                if phases.mixed_road else
                "with [tiles] the resident domain is never allocated")
            label = "MIXED ROAD" if phases.mixed_road else "STREAMED"
            print(f"  ALLOC GATE, {label}: the leg below weighs the "
                  f"envelope "
                  f"{_format_bytes(int(phases.streamed.peak_vram_bytes))} "
                  f"against "
                  f"the envelope budget {_format_bytes(envelope_budget)} "
                  f"(free VRAM less the "
                  f"{_format_bytes(EXTERNAL_MARGIN_BYTES)} other-process "
                  f"margin), not the itemized resident estimate against the "
                  f"allocation budget: {basis}, and the figure already "
                  f"carries the CUDA context and the per-process fixed "
                  f"cost that the allocation reserve holds separately.")
        print(f"  reserve {reserve.reserve_bytes / GIB:.2f} GiB "
              f"(retention {reserve.retention_residual_bytes / GIB:.2f} + "
              f"overhead {reserve.device_overhead_bytes / GIB:.2f} + "
              f"external {reserve.external_margin_bytes / GIB:.2f}); "
              f"{free_source} free {_format_bytes(free)}; budget "
              f"{_format_bytes(budget)}")
        if sampled is not None:
            print("  SHARED MEASURED SIZING SAMPLE: capacity, free memory and "
                  "device profile are the same snapshot used to fit this "
                  "configuration. This is a CPU estimate; no allocation "
                  "was attempted. Recheck live memory before launch.")
        elif declared_memory:
            # The 4090 stress run certified "fits with 0.27 GiB to
            # spare" off this path and the config landed 0.015 GiB from
            # the budget on real hardware.  A declared budget is sizing
            # a card that is not in this machine, and the report has to
            # say so beside the verdict, not leave "fits" to read as a
            # measurement.
            if declares_the_local_card(card_total_gib):
                # Declared, but declared about THIS card -- which is what
                # the wizard's own follow-up check does.  Saying "hardware
                # not present" about the card in the machine, and pricing
                # it on a reference profile to match, is how one box got
                # two answers for one card.
                print(f"  DECLARED BUDGET, MEASURED CARD: the free figure "
                      f"above is declared rather than sampled, so it does "
                      f"not move with what else is on the card right now; "
                      f"the card itself is this machine's and its "
                      f"grid-independent terms are "
                      f"{non_pool_basis(profile, exp)}.")
            else:
                print(f"  ESTIMATE FOR HARDWARE NOT PRESENT: the free "
                      f"figure above is declared, not measured -- this "
                      f"preflight is sizing a card that is not in this "
                      f"machine.  Non-pool terms are priced against the "
                      f"conservative measured reference device profile "
                      f"({profile.name}, "
                      f"{profile.multiprocessor_count} SMs), the largest "
                      f"known-device intercept, so the estimate is never "
                      f"more optimistic than a present-card measurement; "
                      f"verify with `woof check` on the real card before "
                      f"trusting the margin.")
        if budget_underwater_bytes:
            print(f"  NO BUDGET AT ALL: the reserve alone is "
                  f"{_format_bytes(reserve.reserve_bytes)} against "
                  f"{_format_bytes(free)} free, so it exceeds the card by "
                  f"{_format_bytes(budget_underwater_bytes)} before this "
                  f"configuration asks for a single byte.  The budget "
                  f"above is clamped to zero: a negative capacity is not "
                  f"a number anything can be compared against.  The "
                  f"reserve's retention term scales with the "
                  f"configuration, so a smaller one is the lever.")
        if capped_to is not None:
            print(f"  CAPPED: the declared free figure exceeded the card's "
                  f"physical total and was clamped to "
                  f"{_format_bytes(capped_to)}; free VRAM cannot exceed "
                  f"the card")
        if rail is not None:
            print(f"  DEVICE RAIL {_format_bytes(rail['rail_bytes'])} "
                  f"whole-machine; other processes hold "
                  f"{_format_bytes(rail['other_process_bytes'])}, leaving "
                  f"{_format_bytes(rail['rail_free_bytes'])} for this run")
        if report is not None:
            print(f"  --alloc measured: pool used peak "
                  f"{_format_bytes(report.pool_used_peak_bytes)}; held "
                  f"{_format_bytes(report.pool_held_peak_bytes)}; device "
                  f"footprint {_format_bytes(report.device_footprint_bytes)}"
                  f" (non-pool overhead re-calibration "
                  f"{_format_bytes(report.measured_overhead_bytes)}); free "
                  f"after release "
                  f"{_format_bytes(report.free_after_release_bytes)}")
        if abort is not None:
            print(f"  ABORTED before measurement ({type(abort).__name__} "
                  f"during {abort.phase}): {abort}")
        for metric in N0_GATE_METRICS:
            print(f"  {gate_display_name(metric, vram_gib=card_total_gib)}: "
                  f"{_leg_text(gates[metric])}")
        absent = absent_gate_metrics(gates)
        if absent:
            # ``not measured`` used to be the entire message.  It names no
            # remedy, does not say that nothing allocated anything, and
            # does not say that a verdict over the surviving leg is not
            # the verdict these three legs describe.  The module already
            # writes this paragraph for the case where EVERY leg is absent
            # (the fail-closed refusal below); the partial case -- which is
            # every invocation without ``--alloc`` -- got two words.
            names = ", ".join(
                gate_display_name(metric, vram_gib=card_total_gib)
                for metric in absent)
            declared = ("  The leg that did evaluate compares an ESTIMATE "
                        "against the budget you declared, not against a "
                        "measurement of this card."
                        if free_source.startswith("declared") else "")
            # WHY they are absent, which differs by how this command was
            # invoked.  Telling an --alloc run that ran and aborted that
            # "no allocation was attempted" would be false, and pointing
            # it at --alloc would be advice it has already taken.
            why = ("the allocation run aborted before it could measure them"
                   if args.alloc else
                   "no allocation was attempted in this command")
            print(f"  INCOMPLETE: {len(absent)} of {len(N0_GATE_METRICS)} "
                  f"legs above were not measured ({names}): {why}, so those "
                  f"legs have nothing to compare -- they are ABSENT, not "
                  f"passing, and a verdict over the rest is not the verdict "
                  f"these three describe.{declared}")
            if not args.alloc:
                print(f"  to measure them: woof check {args.config} --alloc "
                      f"(constructs every persistent allocation on the real "
                      f"card, runs zero steps, reports measured vs estimate)")
        if envelope_over_budget:
            if binding_phase != "forecast":
                print(f"  WARNING: the binding phase here is "
                      f"{binding_phase}, not the forecast; the envelope "
                      "named below is that phase's.")
            # The exit code this block ANNOUNCES has to be the one the
            # process will really return.  It used to assert "(exit code
            # 4: gates passed)" unconditionally -- including when a gate
            # had just failed and the process therefore exited 1, which
            # is a printed contract a script can be written against and
            # then mis-handle.  Read the gates here, once.
            gate_failed = any(leg is False for leg in gates.values())
            unevaluable = not [leg for leg in gates.values()
                               if leg is not None]
            if gate_failed:
                code_note = ("exit code 1: a gate above FAILED as well, "
                             "and the harder verdict wins")
            elif unevaluable:
                code_note = ("exit code 2: no gate above could be "
                             "evaluated, which fails closed")
            else:
                code_note = (f"exit code {_EXIT_ENVELOPE_OVER_BUDGET}: "
                             "gates passed, envelope did not")
            budget_word = ("WDDM budget" if envelope_platform(
                vram_gib=card_total_gib) == "windows" else "budget")
            # The tail sentence exists to explain a gate that PASSES
            # beside an envelope that does not: the resident legs weigh
            # the itemized pool request, which is a smaller thing than
            # the envelope.  Under [tiles] the alloc leg weighs this very
            # envelope against this very budget, so that explanation
            # would be describing a disagreement that cannot happen.
            if streamed_alloc_gate:
                tail = (" The alloc leg above weighs the same streamed "
                        "envelope against the same budget, so it does not "
                        "disagree with this line -- trim the tiling or "
                        "free VRAM.")
            else:
                tail = (" The gates above "
                        "compare the itemized estimate; the envelope is "
                        "what the machine is measured to reach, so this "
                        "configuration may run out of budget even though "
                        "the estimate gate passes -- trim the "
                        "configuration or free VRAM before trusting the "
                        "pass.")
            print(f"  WARNING: observed peak envelope "
                  f"{_format_bytes(envelope)} exceeds the {budget_word} "
                  f"{_format_bytes(envelope_budget)} -- free VRAM less the "
                  f"{_format_bytes(EXTERNAL_MARGIN_BYTES)} other-process "
                  f"margin, which is all the envelope does not already "
                  f"model -- by {_format_bytes(envelope - envelope_budget)}."
                  f"{tail}  ({code_note}.)")
        if streamed_alloc_gate:
            # A STREAMED RUN'S LEVER IS THE TILE, not the RRTMGP column
            # chunk.  The resident alloc estimate is above the budget for
            # essentially every streamed config -- it describes a domain
            # this run never allocates -- so the block below would print
            # "OVER BUDGET; first lever --column-chunk N" beside a
            # verdict that had just said the run fits, and the lever it
            # named would move a number nothing compares.
            if (phases.mixed_road
                    and int(phases.streamed.peak_vram_bytes)
                    > envelope_budget):
                # A TREE HAS NO SINGLE TILE TO TRIM.  The remedy below
                # names one, and printing it for a mixed road would send a
                # reader to shrink a tiling the overshoot may not even be
                # in -- the plan above already says which domain claims
                # what, which is where the lever actually is.
                print(f"  OVER BUDGET, MIXED ROAD: the per-domain walk "
                      f"holds "
                      f"{_format_bytes(int(phases.streamed.vram_bytes))} "
                      f"and reaches "
                      f"{_format_bytes(int(phases.streamed.peak_vram_bytes))}"
                      f" against the {_format_bytes(envelope_budget)} "
                      f"envelope budget.")
                print("  remedy: a smaller [tiles] tile_nx/tile_ny on the "
                      "domain whose claim the plan above shows is dearest, "
                      "or nbuffers = 1 to trade overlap for room, or free "
                      "VRAM and re-run")
            elif (not phases.mixed_road
                    and int(phases.streamed.peak_vram_bytes)
                    > envelope_budget):
                env = phases.streamed
                # WHICH TERM IS OVER.  With the radiation transient in the
                # figure, a reader told to trim the tile has to be able to
                # see how much of the overshoot the tile can actually
                # move: the transient is a property of the RUNG and no
                # tile size reduces it by a byte.
                transient = (
                    "" if not env.radiation_transient_bytes else
                    f" plus the measured "
                    f"{_format_bytes(int(env.radiation_transient_bytes))} "
                    f"RRTMGP transient, which no tile size reduces,")
                print(f"  OVER BUDGET, STREAMED: {env.nbuffers} buffer(s) "
                      f"of the {env.window_nx}x{env.window_ny} compute "
                      f"window (tile {env.tile_nx}x{env.tile_ny} + halo "
                      f"{env.halo}) hold "
                      f"{_format_bytes(int(env.vram_bytes))}{transient} "
                      f"and reach "
                      f"{_format_bytes(int(env.peak_vram_bytes))} against "
                      f"the {_format_bytes(envelope_budget)} envelope "
                      f"budget.")
                # THE ARITHMETIC, under the verdict.  A refusal that names
                # one total sends the reader to guess which term to trim;
                # one that itemizes it shows the buffers a smaller tile
                # shrinks beside the floors no tile moves, and a
                # screenshot of it can be checked with a calculator.
                for line in getattr(env, "terms_lines", tuple)():
                    print(f"    {line}")
                print("  remedy: a smaller [tiles] tile_nx/tile_ny, or "
                      "nbuffers = 1 to trade overlap for room, or free "
                      "VRAM and re-run")
        elif budget is not None and estimate.alloc_estimate_bytes > budget:
            lever = recommend_column_chunk(exp, budget)
            if lever:
                print("  OVER BUDGET; first lever (radiation column_chunk): "
                      f"--column-chunk {lever}")
            else:
                # It used to end "staged residency (DESIGN REOPEN) per
                # section E".  No pip user has a section E, and the
                # sentence names no action; the actionable one already
                # exists one layer up, in `woof go`'s refusal.  The
                # remedy it then printed -- `woof domain --vram-gib
                # <free>` -- fed a free-VRAM figure to a flag that names
                # a CARD, and on the 3080 walk that recursion refused at
                # every grid size.  The bare wizard measures the card
                # itself, which is the number this remedy actually means.
                print("  OVER BUDGET, and the radiation column_chunk lever "
                      "cannot close it: no chunk halving fits after the "
                      "shared-scratch arena, so the grid itself is what "
                      "has to come down.")
                print("  remedy: re-size against this machine -- woof "
                      "domain ... (bare, it measures this card) -- or "
                      "pick a lighter --physics-profile, or free VRAM "
                      "and re-run")
    else:
        evaluated = sum(value is not None for value in gates.values())
        print(f"woof memory preflight: {memory_gate_verdict(gates).upper()} "
              f"({evaluated} of {len(N0_GATE_METRICS)} allocation gates evaluated)")
        print(f"Configuration: {exp.name} ({len(exp.domains)} domains)")
        print("GPU readiness: " + ("verified by cold compile and execution"
              if readiness["status"] == "verified" else f"not checked; {readiness['detail']}"))
        print(f"Physical GPU capacity: {_format_bytes(physical_total_bytes).strip()} "
              + ("(measured)" if physical_total_bytes is not None else "(not measured)"))
        if card_total_gib is not None:
            print(f"Target GPU capacity: {card_total_gib:g} GiB ({capacity_source})")
        free_label = ("inferred from --budget-gib; not measured" if declared_memory
                      and sampled is None and declared_free_gib is None else free_source)
        if free_label.startswith("measured machine-wide"):
            free_label = "measured machine-wide"
        elif free_label == "measured (shared sizing sample)":
            free_label = "measured; shared sizing sample"
        print(f"Free VRAM used for sizing: {_format_bytes(free).strip()} ({free_label})")
        if args.budget_gib is not None:
            print(f"Configured allocation budget: {args.budget_gib:g} GiB requested")
        print(f"Effective allocation budget: {_format_bytes(budget).strip()}; "
              f"whole-process budget: {_format_bytes(envelope_budget).strip()}")
        print(f"Reserved from free VRAM: {reserve.reserve_bytes / GIB:.2f} GiB for allocations; "
              f"{EXTERNAL_MARGIN_BYTES / GIB:.2f} GiB for the whole-process estimate")
        road = "mixed resident/streamed" if phases.mixed_road else "streamed" if phases.streamed_forecast else "resident"
        mode = getattr(getattr(exp, "tiles", None), "mode", "off")
        print(f"Forecast execution: {road} ([tiles] mode={mode})")
        if resident_slots is not None:
            print(f"Lateral boundaries: {resident_intervals} forcing slot(s) on the card; "
                  "the chained preparation streams the rest of the schedule "
                  "(WOOF_CHAINED_PREP=0 holds the whole schedule)")
        if phases.tree_road is not None:
            for line in phases.tree_road.row_lines():
                print(f"  {line}")
        if phases.streamed is not None:
            boundary = int(getattr(phases.streamed, "boundary_table_bytes", 0) or 0)
            parts = ("" if not boundary else
                     f" ({_format_bytes(phases.streamed.pinned_bytes).strip()} pinned, "
                     f"{_format_bytes(boundary).strip()} lateral-boundary tables)")
            print(f"Streaming host memory: {_format_bytes(phases.streamed.host_bytes).strip()} needed{parts}; "
                  f"budget {_format_bytes(getattr(phases.streamed, 'host_budget_bytes', None)).strip()}")
        print(f"BINDING PHASE: {binding_phase} needs {envelope / GIB:.2f} GiB; "
              f"whole-process budget {_format_bytes(envelope_budget).strip()}")
        if envelope_over_budget:
            print(f"WARNING: observed peak envelope {envelope / GIB:.2f} GiB exceeds the "
                  f"{envelope_budget / GIB:.2f} GiB budget.")
        elif envelope_budget is not None:
            print(f"GPU fit estimate: fits with {(envelope_budget - envelope) / GIB:.2f} GiB headroom")
        else:
            print("GPU fit estimate: unavailable without a measured or declared budget")
        if selects_noahmp(exp):
            # THE BASIS BESIDE THE VERDICT.  A Noah-MP price is either a
            # reading of this card's compile platform or the ceiling over
            # the recorded platforms, and the user is owed the word
            # without asking for --explain.
            from woof.core.noahmp_frame_provenance import noahmp_frame_basis

            frames = noahmp_frame_basis(physics_kernel_modules(exp), profile)
            if frames is not None:
                print(f"Noah-MP frame basis: {frames.sentence()}")
        if not ingest_priced:
            print(ingest_note if prepared is not None else
                  "Forcing preparation GPU memory: not priced for this source")
        if host_forcing_bytes is None:
            print(f"Host RAM: {_format_bytes(host_available).strip()} available; forcing decode memory not priced")
        else:
            print(f"Host forcing decode: {_format_bytes(host_forcing_bytes).strip()} needed; "
                  f"{_format_bytes(host_available).strip()} available")
        for metric, verdict in gates.items():
            if verdict is False:
                print(f"  {gate_display_name(metric, vram_gib=card_total_gib)}: FAIL")
        if abort is not None:
            print(f"Allocation measurement stopped: {abort}")
        if phases.tree_road is not None and phases.tree_road.refusal:
            print(f"Execution plan refused: {phases.tree_road.refusal}")
        for advisory in check_advisories(exp, args.config, streamed=phases.streamed, tree_road=phases.tree_road):
            print(f"Note: {advisory}")
        print("Use --explain for the full memory breakdown and remedies; --alloc measures allocations on the target GPU.")
    if not args.json:
        print(f"Forecast output disk: {output_disk['total_bytes'] / GIB:.2f} GiB "
              "for configured history and checkpoints; "
              "source downloads and preparation excluded.")
    evaluable = [leg for leg in gates.values() if leg is not None]
    # WHICH CODE THIS COMMAND IS ABOUT TO RETURN.  1, 2 and 3 outrank the
    # host refusal and return before it, so a paragraph printed ahead of
    # them told a reader "REFUSED (exit 5)" and then handed them 3, 2 or
    # 1.  Read here, from the same three conditions the returns below use,
    # so the sentence and the status cannot disagree.
    harder_verdict = (
        abort is not None
        or (not all(leg is True for leg in gates.values()) if args.alloc
            else (not evaluable or not all(evaluable))))
    if host_over_available and host_gate_skipped:
        print(f"woof check: host memory gate SKIPPED by "
              f"--no-host-memory-gate: the ingest phase decodes "
              f"{host_forcing_bytes / GIB:.2f} GiB of forcing into HOST "
              f"RAM, against {host_available / GIB:.2f} GiB this box had "
              f"available before this command ran.  Exit code unchanged.",
              file=sys.stderr)
    elif host_refused and not harder_verdict and (args.json or getattr(args, "explain", False)):
        # A REFUSAL, not a note.  The gates above are about the card; this
        # is about the box, and it is the one budget whose exhaustion this
        # product cannot report after the fact -- the kernel or a userspace
        # watchdog kills the worker from outside, woof installs no signal
        # handler, and the shell prints one word.  Nothing above would have
        # gone red: on the transcript that opened this finding every device
        # figure fitted, with GiB to spare.
        #
        # The levers named here are the only ones that move this number.
        # Column chunk, tiling, dropping a nest and a bigger card are VRAM
        # levers and do nothing for it; shortening the forecast does not
        # either, because the decode is charged on what is in the FILES.
        print(f"woof check: REFUSED (exit "
              f"{_EXIT_HOST_MEMORY_OVER_BUDGET}): the ingest phase decodes "
              f"{host_forcing_bytes / GIB:.2f} GiB of forcing into HOST "
              f"RAM and holds it, against {host_available / GIB:.2f} GiB "
              f"this box had available before this command ran -- over by "
              f"{(host_forcing_bytes - host_available) / GIB:.2f} GiB, and "
              f"that figure is a floor.  A host allocation this size is "
              f"refused by the kernel or reaped by a watchdog, which kills "
              f"the run from OUTSIDE the process: no traceback, no woof "
              f"message.", file=sys.stderr)
        print(f"  remedy: fetch less forcing.  The decode is charged on "
              f"what is in the files, so shortening the forecast does not "
              f"reduce it:\n"
              f"    * narrow the area -- host bytes fall with the ratio of "
              f"the source footprints, and this is the larger lever by "
              f"far;\n"
              f"    * fetch only the valid times the forecast consumes -- "
              f"the catalog decodes all "
              f"{phases.ingest.decoded_valid_times} times it finds.\n"
              f"  # VRAM levers (column chunk, [tiles], dropping a nest, a "
              f"bigger card) do not move this number.",
              file=sys.stderr)
    elif host_refused and not harder_verdict:
        print(f"woof check: REFUSED (exit {_EXIT_HOST_MEMORY_OVER_BUDGET}): "
              f"forcing decode needs {host_forcing_bytes / GIB:.2f} GiB host RAM; "
              f"{host_available / GIB:.2f} GiB is available. Reduce the source area "
              "or number of forcing times. GPU tiling does not reduce source decode RAM.", file=sys.stderr)
    if preparation_host_refusal is not None and host_gate_skipped:
        print(f"woof check: host memory gate SKIPPED by "
              f"--no-host-memory-gate: {preparation_host_refusal}.  Exit "
              f"code unchanged.", file=sys.stderr)
    elif preparation_refused and not harder_verdict:
        print(f"woof check: REFUSED (exit "
              f"{_EXIT_HOST_MEMORY_OVER_BUDGET}): "
              f"{preparation_host_refusal}.", file=sys.stderr)
    if streamed_host_refusal is not None and host_gate_skipped:
        print(f"woof check: host memory gate SKIPPED by "
              f"--no-host-memory-gate: {streamed_host_refusal}.  Exit "
              f"code unchanged.", file=sys.stderr)
    elif streamed_host_refused and not harder_verdict:
        print(f"woof check: REFUSED (exit "
              f"{_EXIT_HOST_MEMORY_OVER_BUDGET}): "
              f"{streamed_host_refusal}; a smaller domain moves it.",
              file=sys.stderr)
    if preparation_host_warning is not None:
        print(f"woof check: WARNING: {preparation_host_warning}.",
              file=sys.stderr)
    if abort is not None:
        return 3
    if args.alloc:
        # A requested measurement run fails CLOSED: every leg must have
        # been measured AND passed (shadow F5 / review F6).
        if not all(leg is True for leg in gates.values()):
            return 1
        if host_refused or preparation_refused or streamed_host_refused:
            return _EXIT_HOST_MEMORY_OVER_BUDGET
        return _EXIT_ENVELOPE_OVER_BUDGET if envelope_over_budget else 0
    if not evaluable:
        # Fail closed, and SAY SO.  This exit used to be silent: the
        # wizard prints `woof check CONFIG` as its own step 2, and on a
        # box with no measurable card that command printed the estimate,
        # three "not measured" gate lines, and exit 2 with no sentence
        # naming why or what to type next (UX finding R1, replay walk C
        # step 1).  A refusal names the breakage -- nothing here to
        # verify an estimate against -- and prints a remedy the reader
        # can type: the declared-budget form of THIS command, and the
        # wizard door that prints that form with the numbers filled in.
        print("woof check: REFUSED (exit 2, fail-closed): no gate could "
              "be evaluated -- no VRAM budget was declared and no card "
              "could be measured in this machine (CuPy or a CUDA device "
              "is absent), so the estimate above has nothing to be "
              "verified against.", file=sys.stderr)
        print(f"  remedy: declare the card this config is sized for and "
              f"re-run:\n"
              f"    woof check {args.config} --budget-gib <N> "
              f"--vram-gib <card GiB>\n"
              f"  # this configuration's peak envelope is "
              f"{envelope / GIB:.2f} GiB; a budget at or above it "
              f"passes.\n"
              f"  # `woof domain --card <tier>` (or --vram-gib N) "
              f"prints this exact check line, numbers filled in, as the "
              f"comment under its step 2.", file=sys.stderr)
        return 2  # nothing verifiable: fail closed at the command boundary
    if not all(evaluable):
        return 1
    if phases.tree_road is not None and phases.tree_road.refusal:
        # The run door refuses this [tiles] plan; the figures above price
        # the resident road the run will not take, so this is a refusal,
        # not a pass, and the sentence is the run door's own.
        print(f"woof check: REFUSED (exit {_EXIT_EXECUTION_PLAN_REFUSED}): "
              f"the run door refuses this [tiles] plan, so the figures above "
              f"price a road the run will not take. {phases.tree_road.refusal}",
              file=sys.stderr)
        return _EXIT_EXECUTION_PLAN_REFUSED
    # Gates passed.  The report may still have said, in its own words,
    # that the machine peak lands above the budget -- that sentence and
    # exit 0 cannot both be true, and the sentence is the accurate one.
    # The host refusal outranks it: a run that cannot be held in RAM never
    # reaches the phase whose envelope the other code is about.
    if host_refused or preparation_refused or streamed_host_refused:
        return _EXIT_HOST_MEMORY_OVER_BUDGET
    return _EXIT_ENVELOPE_OVER_BUDGET if envelope_over_budget else 0


def register_cli(subparsers) -> None:
    """Register ``woof check`` (memory section + ``--alloc``).  Per the F2
    ownership map the one-line ``cli.py`` hookup is a controller handoff
    commit at merge; Task 3's input-catalog section joins the same
    subcommand at its own handoff."""
    p = subparsers.add_parser(
        "check",
        help="memory preflight: itemized estimate vs the measured WDDM "
             "budget; --alloc performs the enforced N0 allocation run")
    p.add_argument("config", type=Path, metavar="CONFIG", nargs="?",
                   help="experiment TOML (or legacy RunConfig TOML, "
                        "wrapped as a one-domain experiment); omit only for "
                        "woof check --json context-free box inventory")
    p.add_argument("--prepared-root", type=Path, default=None, metavar="DIR",
                   help="verify prepared forecast inputs and price their "
                        "land cover and retained boundary tables; "
                        "bind a finished bundle or its live prepared head")
    p.add_argument("--wps-namelist", type=Path, default=None, metavar="FILE",
                   help="WPS namelist bound by a single-domain prepared "
                        "bundle (otherwise the config's sibling namelist)")
    p.add_argument("--alloc", action="store_true",
                   help="construct every persistent allocation on the "
                        "device, zero steps, report measured vs estimate "
                        "(N0; GPU required)")
    p.add_argument("--column-chunk", type=int, default=None,
                   metavar="COLS", help="Radiation column-cap override (the first "
                   "over-budget lever)")
    p.add_argument("--reserve-gib", type=float, default=None, metavar="GIB",
                   help="override the calibrated reserve policy with a "
                        "flat reserve")
    declared = p.add_mutually_exclusive_group()
    declared.add_argument("--budget-gib", type=float, default=None, metavar="GIB",
                         help="declared allocation budget (free VRAM minus "
                              "allocation reserve); estimate only")
    declared.add_argument("--free-gib", type=float, default=None, metavar="GIB",
                         help="declared free VRAM before reserves, as used by "
                              "the domain wizard; estimate only")
    p.add_argument("--vram-gib", type=float, default=None, metavar="GIB",
                   help="physical VRAM total of the card being sized for.  "
                        "A CEILING on the free figure, never a source of "
                        "one: a declared --budget-gib plus the reserve can "
                        "otherwise synthesise more free VRAM than the card "
                        "physically has")
    p.add_argument("--rail-mib", type=int, default=None, metavar="MIB",
                   help="whole-machine device residency ceiling: the budget "
                        "is additionally capped at RAIL minus what every "
                        "other process on the card already holds (read from "
                        "NVML before this process touches CUDA).  A property "
                        "of the host, so there is no default")
    p.add_argument("--forcing-interval-s", type=float,
                   default=None, metavar="S",
                   help="override the configured or measured forcing cadence "
                        "for memory sizing (otherwise defaults to ERA5 6-hourly)")
    p.add_argument("--no-host-memory-gate", action="store_true",
                   dest="no_host_memory_gate",
                   help="report the HOST RAM of the forcing decode, the CPU "
                        "preparation and a streamed forecast but do not "
                        "refuse on it.  The counterpart of `woof go "
                        "--no-memory-gate` for the other budget: "
                        "MemAvailable is a reading of this second, and a "
                        "busy box can be momentarily short of RAM a run "
                        "would have had")
    p.add_argument("--devices", type=int, default=None, metavar="N",
                   help="price the run split into N resident slabs, one per "
                        "card (or as [devices] ids places them): replaces "
                        "[devices] count the way `woof go --devices N` "
                        "does, and reports the memory envelope of every "
                        "card and of the pinned host store")
    p.add_argument("--json", action="store_true",
                   help="machine-readable report")
    p.set_defaults(func=check_main)


__all__ = [
    "CORE_KERNEL_MODULES", "CUDA_CONTEXT_BYTES", "DeviceLocalMemoryProfile",
    "POOL_RESERVED_OVER_ESTIMATE_FRACTION",
    "KERNEL_MAX_LOCAL_SIZE_BYTES", "KERNEL_LOCAL_FRAME_RECORDINGS",
    "KernelFrameRecording", "kernel_frame_recording_for",
    "UnderPricedKernelFrame", "under_priced_kernel_frames",
    "MEASURED_LOCAL_MEMORY_PROFILE",
    "CHAINED_TRANSLATION_UNIT_FRAMES", "ChainedTranslationUnitFrame",
    "UNMEASURED_KERNEL_MODULES", "local_memory_profile_from_device",
    "cap_free_to_physical", "cap_free_to_device_wide", "device_physical_total_bytes",
    "device_rail_free_bytes", "device_wide_used_bytes",
    "column_workspace_bytes", "gf_column_workspace_bytes",
    "kf_column_workspace_bytes", "ysu_column_workspace_bytes",
    "shinhong_column_workspace_bytes", "thompson_fork_column_workspace_bytes",
    "gsd_initial_reflectivity_workspace_bytes",
    "kernel_local_memory_bytes", "non_pool_device_bytes",
    "physics_kernel_modules", "refl_diagnostic_reachable",
    "LEVEL_SPECIALIZED_KERNEL_FRAMES", "LevelSpecializedFrame",
    "ACOUSTIC_TIER_FRAME", "TieredKernelFrame", "WDM6_TIER_FRAME",
    "WSM6_TIER_FRAME",
    "domain_kernel_modules", "kernel_local_frame_bytes",
    "ALLOCATOR_HEADROOM", "AllocReport", "CAL_D01_DEVICE_FOOTPRINT_BYTES",
    "CAL_D01_WORKSPACE_BYTES",
    "CAL_D01_POOL_HELD_BYTES", "CAL_D01_POOL_RETENTION_BYTES",
    "CAL_D01_POOL_USED_PEAK_BYTES", "CAL_WDDM_FREE_BYTES",
    "CAL_WDDM_TOTAL_BYTES", "DEFAULT_COLUMN_CHUNK",
    "DEFAULT_FORCING_INTERVAL_SECONDS", "CAL_FIXTURE_OVERHEAD_BYTES",
    "DomainMemoryEstimate", "EXTERNAL_MARGIN_BYTES",
    "ExperimentMemoryEstimate", "GIB", "MemoryItem", "N0_GATE_METRICS",
    "PROBE_DEVICE_FOOTPRINT_BYTES", "PROBE_DEVICE_OVERHEAD_BYTES",
    "PROBE_FREE_BYTES", "PROBE_POOL_HELD_BYTES",
    "PROBE_POOL_USED_PEAK_BYTES",
    "PHYSICS_ARRAY_LIFETIME_AUDIT", "PhysicsArrayLifetime",
    "PreflightAllocError", "PreflightHeadroomError", "ReservePolicy",
    "SCRATCH_SLOT_LIFETIME_AUDIT", "ScratchSlotLifetime",
    "atmosphere_transient_shapes", "check_main", "dudhia_column_shapes",
    "admission_estimate", "estimate_domain", "estimate_experiment",
    "evaluate_alloc_gates",
    "gate_display_name",
    "k_distribution_bytes", "lbc_interval_values", "lbc_intervals",
    "myj_output_transient_shapes", "uwpbl_output_transient_shapes",
    "uwpbl_column_workspace_bytes",
    "nest_allocation_manifest", "nest_field_kinds", "nest_slot_dtypes",
    "nest_slot_shapes",
    "physics_array_lifetime", "physics_array_shapes",
    "physics_field_names_2d",
    "pool_retention_residual_bytes", "recommend_column_chunk",
    "register_cli", "rrtmgp_column_shapes", "rrtmgp_workspace_phases",
    "rrtmgp_workspace_shapes",
    "run_alloc_preflight", "scratch_slot_lifetime",
    "scratch_slot_registry", "scratch_slot_uses_arena",
    "shared_dycore_state_symbols", "shared_dycore_state_workspace_bytes",
    "shared_dycore_state_workspace_shapes",
    "shared_scratch_arena_aliases", "shared_scratch_arena_bytes",
    "shared_scratch_arena_shapes", "shinhong_output_transient_shapes",
    "state_array_shapes",
    "ysu_output_transient_shapes",
    "ENVELOPE_AFFINE_BASIS", "ENVELOPE_A163_BASIS", "ENVELOPE_PER_NEST_FRACTION",
    "FORECAST_PEAK_RATIO_MEASURED", "FORECAST_PEAK_RATIO_SAFETY",
    "FORECAST_POOL_HEADROOM", "FORECAST_PEAK_BATTERY",
    "RANK_POOL_RESIDUE_BATTERY", "RANK_POOL_RESIDUE_BOUND_BYTES",
    "RANK_POOL_RESIDUE_BASIS", "POOL_RESIDUE_MEASURED_BYTES",
    "POOL_RESIDUE_SWING_BYTES", "RANK_POOL_RESIDUE_GAP_BASIS",
    "RANK_POOL_RESIDUE_SUITES", "RANK_POOL_RESIDUE_UNMEASURED_BASIS",
    "rank_pool_residue_cap_bytes", "rank_pool_residue_suite",
    "single_card_pool_residue_cap_bytes", "single_card_residue_terms",
    "mynn_width_adjustable", "required_mynn_pricing",
    "rank_pool_residue_basis",
    "HeldMemoryItem", "bem_workspace_bound_note",
    "forecast_pool_estimate_bytes", "urban_held_bytes",
    "urban_held_item_names",
    "ENVELOPE_UNMODELLED_BYTES", "ENVELOPE_WDDM_BASIS",
    "WDDM_POOL_SLACK_FRACTION", "CARD_CLASS_MULTIPROCESSORS",
    "card_local_memory_profile", "live_device_local_memory_profile",
    "machine_peak_envelope_bytes", "observed_peak_envelope_bytes",
    "peak_envelope_factor", "PEAK_ENVELOPE_FACTORS",
    "PEAK_ENVELOPE_BASIS", "envelope_platform", "estimate_ingest",
    "estimate_phases", "IngestMemoryEstimate", "PhaseMemoryEstimate",
    "INGEST_HOST_DECODE_BYTES_PER_POINT", "INGEST_HOST_RETAINED_COPIES",
    "source_analysis_fields_per_time", "source_analysis_levels",
    "host_available_bytes",
    "ingest_host_geometry", "absent_gate_metrics", "memory_gate_verdict",
]
