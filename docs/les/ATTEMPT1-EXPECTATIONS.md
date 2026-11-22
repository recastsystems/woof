# Tornado-LES attempt #1: the registered expectation document

**Registered 2026-08-05, BEFORE the run. Case updated 2026-08-06, still
before the run.** Config `configs/les_tornado_100m_mayfield_20211210.toml`.
Rulings `docs/internal/specs/P6-LES-DECISIONS-RATIFIED-2026-08-05.md`
(sessions 2 and 3).

## The case, and what kind of box this is

The quad-state supercell of **2021-12-10/11**, and the **Mayfield, KY**
segment of its long-track tornado. Initiation in northeast Arkansas around
02Z on 12-11, Mayfield struck near 03:27Z, roughly a 250 km track. Window
00Z-06Z on 12-11, 6 h.

The case moved here from Dodge City 2016-05-24 on 2026-08-06 so that an
HRRRv1 data-identity question would not need a ruling before anything could
run. Dodge City is parked, not adjudicated. **No screen below changed as a
result of the move** -- the chain, the vertical grid, the parents, the inflow
treatment and every band and provenance are the same. What changed is the
geometry the screens are computed over, and one framing:

**d04 IS A TRANSIT BOX, NOT A GENESIS BOX.** It is centred about 6 km
southwest of Mayfield so the storm enters already mature and tornadic and
transits past the town. Convective initiation happens far upstream, inside
d01/d02 and outside d03/d04 by design: the parents resolve genesis, the LES
children resolve the tornadic transit. **Nothing here grades genesis**, and
no screen may be read as if it did.

The real track is ~250 km and no fixed 20 km box can follow it. Verified
against this geometry (`docs/les/attempt1/mayfield-placement.json`):
Princeton KY at ~04:30Z is inside d03 but outside d04, and Bremen KY at
~05:05Z is outside d03. The box holds one segment of one storm and claims
nothing about the rest of the track.

One accurate note on the vertical grid: this is a **nocturnal cool-season**
event. The 1.7 km threshold the ladder was ratified against is a daytime CBL
depth and does not describe this boundary layer. Every screen below that
needs a depth takes it from the parent's diagnosed PBLH rather than assuming
one, and the "levels in the CBL" framing inherited from the shipped
demonstration does not transfer.

This document exists so that attempt #1 cannot be scored after the fact against
whatever it happened to produce. Every screen below is fixed now. Every one of
them is **committed whatever it reads**. A screen that fails is a finding that
steers attempt #2: that is the entire point of running this ahead of P3/P5/P6
completion.

## The discipline this document is written under

**Bands are cited, never invented.** Where a committed receipt supplies a
principled band, the band is stated with the receipt it was cut from. Where no
principled band exists, the metric is registered **REPORT-ONLY** and says so.
A prior single measurement is recorded as context and is explicitly *not* a
band: the shipped demonstration ran a different case, a different refinement
ratio and a different vertical grid, and its numbers do not transfer.

**No obs-skill claim.** Nothing in this run is scored against MRMS, ASOS, the
storm reports, or any other observation, and no receipt from it may be cited as
if it were. Skill against observations is the flagship program's business
(D-L10, carried forward). This run claims structural behaviour and determinism
on one named case, or it claims nothing.

**This is not a milestone.** Attempt #1 is deliberately ahead of P3 completion
(the inflow campaign), P5 (own-reference validation) and P6 (the 250 m
certification). It cannot discharge any of their acceptance criteria and no
receipt it produces is admissible as R2/R3 conformance evidence.

---

## A. HARD screens: a failure here stops the claim

| # | Screen | Band | Provenance |
|---|---|---|---|
| A1 | Run status | `status: PASS` end to end, all four domains, full 28,800 s | The route's own contract; the demonstration's PASS is `nested-les-scored-2026-08-02.md:35` |
| A2 | Dual-run byte identity | The two runs are byte-identical with `ENVIRONMENTAL_FIELDS` excluded. **Any mismatch is a corruption finding that stops the claim, full stop** | Standing corruption detector on the no-ECC card; les-completion spec §8.1 item 4 / AC-P6.3 |
| A3 | Stability gate | `stability_gate_failed` never fires (`woof/core/dycore.py:2521-2538`) | The gate is the tree's own hard bound; nothing softens it for LES |
| A4 | Parseval closure | Spectral residual `<= 1e-10 %` on every domain and every frame | `nested-les-scored-2026-08-02.md:95-100`: worst measured residual **3.7e-14 %** across both runs and both domains, after the correction that the first version of the figure failed. The band is three orders looser than the measurement and is a closure check on the instrument, not on the physics |
| A5 | Restart exercised | At least one `gpuwmrst_d*` checkpoint set is written and one resume from it reproduces the parent trajectory | `restart_interval_s = 3600.0`; the demonstration's `= 0.0` is the named defect this repeats-not (`nested-les-scored-2026-08-02.md:147-148`) |
| A6 | Vertical grid as ratified | 33 half levels below 1.7 km on the grid the run actually integrated | G2 ruling; `docs/les/attempt1/eta72-ladder.json`, reproduced from the run's own eta array |

## B. GRADED structural screens: bands where a receipt supplies one

### B1. var(w) child/parent ratio: **floor banded, magnitude REPORT-ONLY**

Resolved vertical-velocity variance, child against the parent's own footprint,
over identical ground, at the level nearest 0.5 z_i.

- **Band (floor):** `ratio > 1.0`, sustained across the convective window, on
  **both** child/parent pairs (d03 over d02, d04 over d03). A refined domain
  that does not carry more resolved variance than the domain that fed it has
  not refined anything. This is the only claim the demonstration's evidence
  supports at a different refinement ratio.
- **REPORT-ONLY:** the magnitude. Context, not a band: the demonstration
  measured **4.12x to 8.42x** sustained over seven hours at a 250 m child on a
  750 m parent (ratio 3, 49 levels, a non-severe corridor case),
  `nested-les-scored-2026-08-02.md:44-56`. Attempt #1 refines 2x then 5x on 72
  levels over a supercell. Nothing licenses transferring that band.
- **Method, pinned now:** z_i comes from **d02**, never from a child. A PBL-off
  domain has `PBLH` identically 0.0 in every frame, and scoring the
  demonstration from the child's own PBLH pinned every "mid-CBL" level near
  107 m and understated the ratio by 2-3x (7.17x read as 2.26x):
  `nested-les-scored-2026-08-02.md:58-70`. **Both** d03 and d04 are PBL-off
  here, so both take z_i from d02.

### B2. Variance-profile shape: **REPORT-ONLY**

Peak of the `var(w)` profile expressed as z/z_i, in a pinned window.

- **No principled band exists.** The spec asks for "peak z/z_i in a pinned
  window" (§8.1 item 2) without registering the window, and no committed
  receipt supplies one. **Registered REPORT-ONLY.**
- Context, not a band: the demonstration's child peaked near 1000 m with
  z_i = 1741 m, i.e. **z/z_i ~ 0.55**, which it called the canonical CBL
  profile (`nested-les-scored-2026-08-02.md:81-85`). One measurement, one
  case, a non-severe boundary layer.
- The window must be registered from the P5 own-reference family before any
  run is graded pass/fail on this. Flagged as an open gap in the battery.

### B3. Updraft area fraction: **floor banded, magnitude REPORT-ONLY**

Fraction of the footprint with `w > 0.5 m s-1` at mid-CBL.

- **Band (floor):** child fraction `>` parent fraction, on both pairs.
- **REPORT-ONLY:** the magnitude and the ratio. Context: the demonstration's
  child roughly doubled its parent (0.062-0.219 child vs 0.021-0.121 parent),
  `nested-les-scored-2026-08-02.md:60-64`.

### B4. Parseval-normalised spectral overlay: **structure banded**

- **Band (structure):** the child and parent spectra converge at large scales
  and separate **only** in the band between the two effective resolutions.
  Effective resolution is 7dx: **d02 7.0 km, d03 3.5 km, d04 0.7 km**.
  Separation outside that band is not refinement.
- **Band (closure):** A4 above.
- Provenance: `nested-les-scored-2026-08-02.md:73-79` and spec §8.1 item 2.
  Both the scale artefact (unnormalised `|fft2|^2` grows as N^4) and the
  window-correlation closure error are corrected in the shipped instrument;
  this run uses that instrument unmodified.

### B5. Resolvability precondition: **banded, and expected to be tight on d03**

`z_i / dx` from d02's PBLH over each child's footprint, at peak heating.

- **Band:** `z_i/dx >= 7` on **d04**. The demonstration named `PBLH/dx = 7.0`
  as "the precondition for any of this meaning anything"
  (`nested-les-scored-2026-08-02.md:66-68`). At 100 m and a Plains May
  boundary layer this should clear comfortably.
- **Pre-registered expectation, stated before the run:** **d03 at 500 m is
  expected to be MARGINAL or to FAIL this screen, and more surely here than it
  would on a daytime case.** A nocturnal cool-season boundary layer outside
  the storm's own circulation is shallow and stably stratified, so `z_i/dx` at
  500 m may be well under 7 for most of the window. d03 is a bridge domain
  whose job is to hand d04 a boundary, not to be a canonical LES. If it fails
  here that is the registered expectation being met, not a surprise.
- **Method note, fixed now:** on this case the meaningful turbulence is inside
  and around the storm, not in a quiescent CBL. `z_i/dx` is reported for
  completeness and as the demonstration's stated precondition, but it is NOT
  the screen that decides whether d04 resolved anything -- B1, B4 and the
  section C series are. Recorded so the screen is not later promoted past what
  it can carry on a nocturnal case.

## C. STRUCTURAL screens on d04: **REPORT-ONLY, committed whatever they read**

These are the tornado-scale screens. They are registered as **structural
observations, not as a claim of any kind**, and no band is attached to any of
them because no committed receipt in this program has ever measured them.

| # | Screen | Registered as |
|---|---|---|
| C1 | d04 vertical-vorticity maximum, full time series at 5 min cadence | REPORT-ONLY |
| C2 | Swirling-updraft time series (co-located `w` and vertical vorticity above their own thresholds), thresholds recorded with the series | REPORT-ONLY |
| C3 | d04 maximum wind and its height, full series | REPORT-ONLY, **and it is the sizing input for attempt #2** |
| C4 | Whether a resolved vortex exists at all, anywhere in the d04 window | REPORT-ONLY. A run that produces none is a **finding**, not a failure |
| C5 | Whether the storm is in the box at all during the window, and for how long | REPORT-ONLY, and it conditions C1-C4. A transit box can be missed by a track error in the parents; if it is, C1-C4 are unscoreable rather than negative |

**Explicitly not claimed by C1-C5:** that any vortex is a tornado; that its
intensity, track, timing or duration corresponds to the real 2021-12-10 event;
any EF-scale equivalence; any comparison to storm reports or damage surveys.
The 100 m grid and the 51.5 m boundary-layer spacing are nowhere near
tornado-core resolution, the d04 box is fixed while the real storm moved ~250
km, and the run has no observational reference attached to it. **In
particular: that the run is centred on Mayfield is a statement about where the
grid is, not a claim that anything the model produces there resembles what
happened there.**

## D. Numerical-health series: **one band, rest REPORT-ONLY**

| # | Screen | Band |
|---|---|---|
| D1 | Maximum CFL per domain, full series | REPORT-ONLY. Context from the plan's sizing: `dt = 5*dx` puts CFL near 0.5 at 100 m s-1, and a core above ~130 m s-1 breaks it |
| D2 | `w_damping` activation count and locations, per domain | REPORT-ONLY. The count is the accurate measure of how much the run leaned on the damper |
| D3 | The run completes without the stability gate firing | Banded: this is A3 |
| D4 | Mass-conservation residual via the committed conservation machinery | REPORT-ONLY unless the machinery carries its own committed gate, in which case that gate applies unchanged |

Whatever D1/D2 read is the sizing input for attempt #2's timestep. That is
their purpose; they are instruments, not grades.

## E. Fetch hygiene: **the scored footprint is defined before the numbers are**

| # | Screen | Registered as |
|---|---|---|
| E1 | D90 spin-up fetch, measured by the committed meter, on **both** d03 and d04 | REPORT-ONLY (the value) |
| E2 | Every number in sections B and C is computed on the **fetch-clean interior only** | **BINDING METHOD.** Not a band: a definition of the scored footprint, fixed before the run |

Provenance: `INFLOW-FETCH-D90-2026-08-03.md` measured the shipped
demonstration's scored footprint as **56-100 % spin-up fetch**, i.e. the
published score was computed largely over ground that had not yet grown its own
turbulence. That contamination is the reason E2 is binding here.

**Expected, and pre-registered: sharper on a transit box than it would be on
a genesis box.** Two things compound here:

1. **The storm ENTERS d03 rather than forming in it.** Verified against this
   geometry, the track crosses only about **50-75 km of d03** before reaching
   d04 (Cayce KY is inside d03, upstream of the box; initiation is outside d03
   entirely). At 25-30 m s-1 storm motion that is **30-50 minutes** of
   in-domain transit for the inflow air to develop resolved turbulence before
   it reaches the 100 m child. A genesis box would have had hours.
2. **d04 is 20 km wide.** If the measured D90 fetch approaches that, **there
   is no fetch-clean interior on d04 at all** and section C becomes
   unscoreable on this geometry.

Either outcome is a first-class finding, and together they are the single most
likely way attempt #1 falsifies its own design. `inflow_perturbation` is ON at
**d03** partly to shorten (1); whether it shortens it *enough* on a
50-75 km fetch is exactly what E1 measures. A failure here sends attempt #2 to
a larger 100 m box (96 GB territory per the plan's sizing verdict), to a
d03 shifted upstream to buy fetch, or to a stronger inflow treatment, and
learning which, for 0.3 h of card time, is why this run is staged early.

### AMENDED 2026-08-06, before the run: `inflow_perturbation` OFF at d04

This document said the perturbation was ON at *both* LES domains. It is now
ON at d03 and **OFF at d04**, ruled after the model refused the smoke in
setup:

```
ValueError: inflow_perturbation defines its vertical extent from the
parent-diagnosed PBLH and therefore requires a parent running a PBL
scheme; parent bl_pbl_physics=0
```

The generator exists for the mesoscale→LES handoff: a parent running a PBL
scheme diagnoses PBLH because it *parameterizes* the turbulence its child
must spin up from nothing. That is d02 (YSU) → d03, and the perturbation
stays on there. d03 → d04 is a different problem: the parent is itself LES
at 500 m, its **resolved** eddies advect through d04's inflow faces and *are*
the inflow turbulence, so injecting synthetic perturbations on top of them
would double-count. The generator was never designed for a resolved-turbulence
parent, which is why its contract demands a parent PBLH. The refusal is the
model correctly encoding its own applicability boundary.

**This ruling has a registered falsifier, and it is E1.** The d04 D90
spin-up fetch receipt stays a graded screen and stays committed whatever it
reads. The ruling asserts that d03's resolved eddies arrive at d04 already
developed; E1 is the measurement that can contradict it. If the d03 → d04
fetch is insufficient, **attempt #2 revisits this ruling**: that is the
disposition, fixed now rather than argued afterwards.

Amended before relaunch, not after the numbers, per section "The discipline
this document is written under".

## F. What is deliberately absent

- **No obs-skill screen.** See the discipline note above.
- **No P5 reference-family comparison.** P5 has not run; there is no own-WRF
  reference for this case. Every comparison here is WOOF-ArWen against WOOF-ArWen.
- **No km_opt=2 arm.** The plan makes it contingent on the km3 pair succeeding
  first. It is not part of attempt #1.
- **No Shin-Hong parent comparison.** G3 ruled the YSU fallback; nothing here
  bears on P4 Finding 1 in either direction.
- **No claim about the vertical grid being sufficient.** G2 ratified 33 levels
  below 1.7 km as better than 18. Whether it is *enough* is what B1/B2/B4
  measure, and one case cannot settle it.

## G. Disposition rule, fixed now

1. **A1-A6 all pass** → the receipts are published as attempt #1's result, with
   B/C/D/E reported at whatever they read.
2. **Any A screen fails** → the receipts are published as a **finding**, the
   structural sections are published beside them marked unscoreable, and no
   structural statement is made. A2 in particular stops everything.
3. **A passes, a B floor fails** → published as a finding against the chain
   design, with the failing pair named. This steers attempt #2's ratios.
4. **Section C is empty of any vortex** → published as a finding about the
   configuration, explicitly not as a statement about the atmosphere.
5. **The storm never transits d04** (C5 near zero) → sections C1-C4 are
   published as unscoreable, and the finding is about the parents' track and
   the box placement, not about the LES closure. This is a real possibility on
   a transit box and is registered so it cannot be reported as a negative
   result about turbulence.

In all four dispositions the receipts are committed. There is no outcome of
this run that goes unpublished.
