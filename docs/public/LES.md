# Large-eddy simulation in WOOF

WOOF implements two of WRF's resolved-turbulence closures. This page states
what they are, what has been measured, and (at least as carefully) what has
**not**.

The one-sentence scope, which the rest of this page only elaborates:

> `km_opt=2` and `km_opt=3` are implemented, FP64-mirror-verified per kernel,
> restart-bit-identical, and produce canonical CBL statistics on a
> single-domain dry doubly-periodic idealized case at 100 m on one GPU.
> **Both** additionally run as a 250 m nested child inside a real moist
> terrain-following HRRR tree, carrying up to 8.4x (`km_opt=3`) and 9.9x
> (`km_opt=2`) the 750 m parent's resolved vertical-velocity variance
> over the same ground. Those nested measurements used `mix_isotropic = 0`
> on configurations now archived under `configs/frozen/`. The shipped
> configurations have changed since and have not been re-scored (section 4).

LES enters WOOF's existing maturity ladder as **implemented-unverified**.
There is no separate "LES-verified" tier and none is claimed.

---

## 1. The two closures

| `km_opt` | closure | SGS TKE | selectable on a nest |
|---|---|---|---|
| 2 | 1.5-order prognostic TKE, `K = c_k sqrt(e) l` | prognostic carrier, advected | yes; under a `km_opt=2` parent it loads with a not-yet-verified warning (see §4) |
| 3 | 3-D Smagorinsky, `K = (c_s l)^2 \|S\|` | diagnostic | yes |

Both require `diff_opt=2`. `km_opt=2` additionally requires
`bl_pbl_physics=0`: WRF will evolve TKE with the PBL on, but that combination
has no vertical TKE mixing, because `vertical_diffusion_2` is PBL-off-gated.
WOOF refuses it rather than run a half-wired scheme.

The parameter row (`c_s`, `c_k`, `mix_isotropic`, `mix_upper_bound`,
`tke_upper_bound`, `tke_heat_flux`, `tke_drag_coefficient`) is per-domain,
matching WRF's `max_domains` Registry spelling, and importable from a WRF
namelist. `isfflx` is the exception and is discussed in §4.

### Vertical scalar mixing with the PBL off

This is the single most misread property of a PBL-off run, so it is stated
flatly:

| configuration | vertical **momentum** mixing | vertical **scalar** mixing |
|---|---|---|
| `km_opt=2` or `3`, PBL off | yes | **yes** |
| `km_opt=1` or `4`, PBL off | yes | **no** |

Under `km_opt` 1 and 4 with the PBL off there is no vertical exchange
coefficient pair at all, so `theta` and every moist species are mixed
horizontally only: a 2-D Smagorinsky run with the PBL off does not mix heat
or moisture vertically by any route. Supplying one of the LES closures is
what fills `smag_kmv`/`smag_khv` and puts the scalar rows into the vertical
pass (`woof/core/dycore.py`, the `cfg.km_opt in (2, 3)` branch inside
`if cfg.bl_pbl_physics == 0`).

---

## 2. What has been measured

Hardware: one RTX 5090 (sm_120, 32,607 MiB), driver 575.57.08, CUDA 12.9,
CuPy 14.1.1. Case: dry convective boundary layer, 96x96x64 at dx = 100 m,
ztop 2400 m, dt = 0.5 s, 2 h, surface heat flux 0.24 K m s-1.

**Kernel level.** 37 GPU tests across `tests/test_smag3d.py`,
`tests/test_tke_km2.py` and `tests/test_tke_budget.py` pass on the card. Each
checks a kernel against an FP64 NumPy mirror of the WRF formula, with the
formula authority anchored in `handoffs/P6-LES-WPL0-AUTHORITY-RECEIPT.md`.

**Boundary-layer statistics.** `wth_res_max_over_qs` is the resolved buoyancy
flux as a fraction of the prescribed surface flux; it is the
closure-independent resolved-fraction measure, because its subgrid
counterpart comes from the live `smag_khv` field that both closures fill.

Every row below is a receipt committed at
`docs/superpowers/receipts/les/cbl-2026-08-02/`, produced by the tree
this page describes. The five older receipts beside that directory come
from the LES engine branch and an earlier code state; they are kept as
history and **do not** correspond to this table.

| case | closure | surface | lateral | resolved flux / surface flux | entrainment ratio |
|---|---|---|---|---|---|
| `km2` | 2 | prescribed | periodic | 0.8380 | −0.143 |
| `km3` | 3 | prescribed | periodic | 0.8439 | −0.165 |
| `km2most` | 2 | MOST (`isfflx=2`) | periodic | 0.8686 | −0.199 |
| `km3most` | 3 | MOST (`isfflx=2`) | periodic | 0.8322 | −0.218 |
| `km2walled` | 2 | prescribed | specified | 0.9424 | −0.246 |
| `km3walled` | 3 | prescribed | specified | 0.9722 | −0.282 |

The entrainment column is `wth_total_min_over_qs_mean`, the mean of the
trailing 20 samples. **Do not cite the single-frame
`wth_total_min_over_qs`** that sits beside it in the receipts: measured
frame by frame over one run's final hour it wanders between −0.072 and
−0.184, an 81% spread about its own mean, and two receipts differing by
64% on it are two draws from that distribution rather than two different
answers.

For `km_opt=2`, which has a prognostic carrier, the TKE-based mixed-layer
resolved fraction is **0.894**: above the conventional 0.8 threshold for a
run to be called an LES rather than a very fine mesoscale run. That metric is
reported as `null` for `km_opt=3`, which has no prognostic SGS TKE to divide
by; see §5.

**Grid refinement moves the partition the right way.** Halving the spacing
(192x192x96 at dx = 50 m, dt = 0.25 s, 28,800 steps) raises the `km_opt=2`
resolved fraction from 0.894 to **0.932** and the resolved flux fraction from
0.838 to **0.896**, in 858 s on one card. `km_opt=3` moves the same way over
the same refinement, 0.844 to **0.904**, in 753 s. That direction is the
property that distinguishes an LES closure from a parameterization: refining
the mesh moves work from the subgrid model to the resolved field, rather than
leaving the partition where it was. The TKE budget tightens with it, closing
to 7.88e-10 relative.

Both 50 m runs are committed beside the 100 m six, at
`docs/superpowers/receipts/les/cbl-2026-08-02/{km2fine,km3fine}/`: receipt
and per-minute profile arrays alike, from the same driver and the same code
state as the 100 m pair, so the refinement is a difference between two runs
of one instrument rather than between two printed numbers. Every scalar in
this section is recoverable from those arrays alone. **One exception:
`resolved_fraction_ml` must not be recomputed from the `km_opt=3` arrays.**
Its `e_sgs` plane is identically zero there (that closure carries no
prognostic SGS TKE) so the ratio evaluates to exactly 1.000 from a missing
denominator. The receipt reports it as `null` for that reason, and
`wth_res_max_over_qs` is the measure defined under both closures.

Neither 50 m receipt is a single draw. Both were run twice, independently,
on the same card, and reproduced byte-identically: all 24 arrays for
`km_opt=2`, all 11 for `km_opt=3`, and every physics field of both receipts.
On a card without ECC that comparison is the corruption detector, and it is
clean here as it is at 100 m.

One field did **not** reproduce, and this page previously published it.
`vram_gib` read 3.11 GiB on the first `km_opt=2` run and 1.84 GiB on its
byte-identical repeat. It is measured as whole-device occupancy at exit, not
the run's own footprint, so on a shared card it counts other tenants: the
committed 100 m `km2` receipt reads 14.6 GiB for a case an eighth the size,
which is the same artifact. It is not a sizing figure and is no longer cited
as one. The field has since been split so the dual-run screen cannot trip
on it: `vram_pool_gib` is this process's own live allocation (a run
property, compared by the screen) and `vram_device_gib` is the device-wide
figure (environment-dependent, excluded from the comparison along with the
wall-clock timings). When comparing receipts, compare every field except
those in `ENVIRONMENTAL_FIELDS`: `vram_gib` in receipts written before
the split is the device-wide figure and is likewise excluded.

This is a convergence *direction*, measured at two spacings on one case. It
is not a grid-convergence study and no order of convergence is claimed.

**WRF itself has now run this case.** A WRF v4.6.1 able to initialise
`em_les` was built independently from pristine source for exactly this
comparison; the build recipe and every scorer are committed at
`tools/wrf_em_les_oracle/`, and the machine-readable run receipts ship at
[receipts/les/](receipts/les/README.md). With both models reduced by one
routine, the two implementations agree at 100 m to **0.32 %** on z_i,
**0.11 %** on w\*, **0.15–0.29 %** on the resolved flux fraction and
**0.27 %** on the prognostic subgrid TKE: the
`wrf_oracle_same_instrument_*.json` receipts there. No pass band is cut
anywhere in that comparison: differences are reported, not judged, and the
comparison is statistical because it can only be, WRF's ideal-case
perturbation is unseeded and decomposition-dependent, so the two initial
conditions cannot be made equal.

**WRF also refines the same way, judged against measured noise, not
assumed noise.** WOOF's own realisation spread (n=18 at 100 m, n=9 at
50 m) is measured and committed beside WRF's own draws in
[receipts/les/ARWEN-REALISATION-SPREAD.md](receipts/les/ARWEN-REALISATION-SPREAD.md),
its correction history kept in place. On the flux-fraction arm WOOF's
spread is **8.7x** WRF's (0.01414 against 0.00163 stdev at 100 m: an open
question, recorded rather than diagnosed); on the TKE-based resolved
fraction the two are at parity (0.00231 against 0.00247). That receipt's
composite statement is the citable form: *both refinement arms land at or
below a third of their measured single-draw noise, inside a band registered
before the data existed; the TKE arm's band is a sharp test and the flux
arm's is not, because WOOF's own 50 m flux-fraction scatter (0.017, n=9)
exceeds the band.* An instrument-qualification control (`km_opt=4`, which
must **not** present a credible LES partition here) and a
`-fno-tree-vectorize` toolchain arm ship with the same set.

**TKE budget closure.** With `--tke-budget`, `km_opt=2` accumulates the
term-by-term budget on device. Over a 120-step window the terms close to a
relative residual of **2.49e-09**: machine precision for FP32 state
accumulated in FP64. Shear and dissipation dominate (1.62e7 against
-1.87e7 in mu-weighted units) with buoyancy 2.46e6, which is the expected
partition for a sheared CBL.

**Restart bit-identity.** Running straight through, versus checkpointing at
60 min and restoring into a freshly built state, reproduces the end state
exactly: 10 members compared for `km_opt=2` and 9 for `km_opt=3`, none
differing. The comparison is non-trivial: the TKE carrier was 1.98 m2 s-2
after restore against 0.0 in the cold-started state, so the archive and not
the trajectory supplied the developed turbulence field.

**Determinism and silent corruption.** Two independent 14,400-step
`km_opt=2` integrations produced byte-identical state
(`120ca74150f55b7b…`) and receipts identical in every field but wall time.
On a card without ECC this dual-run comparison is the corruption detector,
and it is clean. Since extended across hardware: the same seed is
bit-identical on three different cards (two of them different sm_120
silicon) and seed-equivalent against a different architecture (sm_89),
where the worst receipt field differs by 1.81 difference-sigma against the
n=18 realisation spread. The receipt, its per-card provenance and the
claim's stated boundary are at
[receipts/les/ARWEN-CROSS-CARD-DETERMINISM.md](receipts/les/ARWEN-CROSS-CARD-DETERMINISM.md).

### Nested, on a real case

`km_opt=3` has run as a 250 m child inside a 3 km / 750 m HRRR tree
(moist, terrain-following, one-way nested, PBL off on the child only) for
seven hours of a real convective afternoon. `status: PASS`, 3010 s wall,
31 frames. Against the parent's own 134x134-cell footprint, so the two
cover identical ground:

| valid | mid-CBL z | var(w) child | var(w) parent | ratio | PBLH |
|---|---|---|---|---|---|
| 18Z | 670 m | 0.564 | 0.079 | 7.2 | 1256 m |
| 19Z | 807 m | 1.010 | 0.120 | **8.4** | 1535 m |
| 21Z | 806 m | 1.311 | 0.212 | 6.2 | 1741 m |
| 23Z | 803 m | 1.093 | 0.265 | 4.1 | 1563 m |

The spectra converge above ~20 km wavelength (the child inherits its
large scales through the lateral boundary) and separate only between the
parent's 7dx limit (5.25 km) and the child's (1.75 km), which is where
refinement is supposed to add energy. Full detail, including what it does
not establish, in
`docs/superpowers/receipts/les/nested-les-scored-2026-08-02.md`.

**A PBL-off domain carries no PBLH.** WRF diagnoses PBLH inside the PBL
scheme, so an LES child running `bl_pbl_physics=0` has `PBLH` identically
zero. Anything keyed on boundary-layer depth over such a domain must take
it from the parent or rediagnose it from the resolved profile. Scoring the
run above from the child's own PBLH understated the variance ratio by a
factor of two to three.

### A second regime: does the closure obey mixed-layer similarity?

Everything above is one surface forcing. Driving the same CBL across an
**eightfold** range of surface heat flux tests whether the closures follow
the similarity scaling they should, resolved `var(w)` growing as w²,
where w = (g/θ · Q · zi)^(1/3), or merely follow the forcing.

| Q (K m s-1) | w* | zi | zi/Δx | peak var(w), km2 | peak var(w), km3 |
|---|---|---|---|---|---|
| 0.06 | 1.35 | 1266 m | 12.7 | 0.643 | 0.621 |
| 0.12 | 1.79 | 1456 m | 14.6 | 1.205 | 1.096 |
| 0.24 | 2.37 | 1689 m | 16.9 | 2.191 | 2.033 |
| 0.48 | 3.20 | 2089 m | 20.9 | 4.465 | 3.850 |

Over that range w rises 2.37x, so pure similarity predicts a **5.62x**
rise in peak `var(w)`. Measured: **6.94x** for `km_opt=2` and **6.20x**
for `km_opt=3`. Both closures therefore track w² to within 23% and 10%
respectively over an 8x forcing change, and the CBL deepens
monotonically (1266 → 2089 m) as it must.

The residual is **not** unexplained. zi/Δx climbs 12.7 → 20.9 across the
sweep, so the deeper boundary layer is also the better-resolved one at
fixed 100 m spacing, and the vertical table below shows exactly that
sense of drift. Similarity and resolution are confounded here by
construction; separating them would need the grid refined in step with
the forcing, which was not run. What can be said is that the closures
reproduce mixed-layer similarity to first order, and that the residual
has the sign and roughly the size the resolution change predicts.

### What the vertical grid costs: measured, not asserted

The nested child runs its grandparent's 49 shared levels. At peak heating
that is **18 levels below the 1741 m boundary-layer top, an effective
dz of 96.7 m**. To price that, the same idealized CBL was run at seven
vertical resolutions with everything else fixed (96x96 at dx = 100 m,
`km_opt=2`, 2 h). Every diagnostic is averaged over the last 20 minutes:
the instantaneous entrainment minimum wanders enough between frames to
invent a trend that is not there.

| nz | dz | levels in BL | resolved fraction | var(w)peak / w*² | zi |
|---|---|---|---|---|---|
| 16 | 150 m | 11.1 | 0.858 | 0.351 | 1670 m |
| **24** | **100 m** | **17.0** | **0.873** | **0.373** | 1696 m |
| 32 | 75 m | 22.8 | 0.880 | 0.367 | 1709 m |
| 48 | 50 m | 34.4 | 0.889 | 0.379 | 1722 m |
| 64 | 37.5 m | 45.0 | 0.894 | 0.390 | 1689 m |
| 96 | 25 m | 67.3 | 0.907 | 0.399 | 1683 m |
| 128 | 18.8 m | 89.6 | **0.921** | 0.439 | 1679 m |

The nested child sits on the bold row. **The cost of the shared vertical
grid is 4.8 points of resolved TKE fraction**: 0.873 against 0.921 at
`nz=128`. That is the deepest rung of this sweep, not the model's ceiling:
§4 records receipted runs at nz=160 and nz=192 and a solver that admits 256,
with radiation the binding constraint on a real-data column.
Put the other way round, the subgrid model carries **12.7% of the
turbulence instead of 7.9%: about 60% more of it is parameterized** than
at the ceiling. Peak resolved `var(w)` normalised by w² is 15% low
(0.373 against 0.439; Deardorff's canonical CBL value is ≈0.4, which the
grid reaches only above ~45 levels in the boundary layer).

Two things this does **not** show, and both were checked before being
ruled out:

- **Boundary-layer depth is not measurably biased.** zi spans
  1670–1722 m across a factor of eight in dz, with no monotonic trend:
  the scatter is sampling noise, not resolution. An earlier
  single-snapshot version of this table appeared to show a 6% high bias
  at coarse dz; time-averaging removed it entirely.
- **Coarse vertical resolution does not disqualify the run.** Even at
  11 levels in the boundary layer the resolved fraction is 0.858, above
  the conventional 0.8 threshold. The vertical grid degrades this LES; it
  does not demote it to something else.

---

## 3. How to select it

The idealized case is reachable directly:

```
python -m woof.verify.cases.convective_boundary_layer \
    --km-opt 2 --minutes 120 --tke-budget --out ./out --tag cbl
```

For a real, config-driven run see `configs/`: a shipped LES configuration
selects the closure on a nested child rather than requiring hand-written
TOML. The parameter row is importable from a WRF namelist, so an existing
`namelist.input` carrying per-domain `c_s`/`c_k` columns round-trips.

Every door that makes a grid applies the same rule, so a deep chain needs no
hand edits:

- `woof domain --les-below-m M` (default 250) writes PBL off, the 3-D
  closure, `mix_isotropic = 1` and `isfflx = 1` on every real-data domain at
  or below `M` metres, the top grid included when it takes specified
  boundaries. `--les-closure smagorinsky|tke` picks `km_opt` 3 or 2.
  `--les-inflow-seeding on|off` (default on) turns the inflow generator on
  for the first LES nest under a parent that still runs a 1-D PBL.
- `woof downscale --les-below-m M --les-closure ...` does the same for a
  nest-down child and for every segment of a moving step-down, on top of the
  child's suite.
- `--les-below-m 0` keeps the suite's 1-D PBL at any spacing.

---

## 4. Bounds: what constrains a run today

These are limits of the current build, not opinions about LES.

- **`nz <= 192` has been RUN and receipted; the solver admits 256.** The
  acoustic solver carries one per-thread stack column sized `WPHI_MAX_LEV`
  (`woof/core/kernels/acoustic.cu`), and that bound is compiled from a tier
  ladder chosen by `nz` (`woof/core/acoustic.py`, tiers 129/193/257). Every
  `nz <= 128` configuration compiles the same kernel it always did: the
  unspecialized module, no define injected. Above 256 the host raises before
  any launch, so it is a loud refusal and not a silently skipped kernel.
  The `nz=160` acceptance battery registered at
  `docs/superpowers/specs/2026-08-04-p2-nz-tier-acceptance.md` **ran on
  2026-08-04 and passed every leg**: mass drift 6.27e-08, TKE budget residual
  1.23e-09, restart bit-identical member for member, and a determinism pair
  with 0 of 54 screened receipt fields differing. The level sweep and the
  perf numbers ran with it. `steps/s x nz` is flat to 4.5% peak to peak from
  nz=64 to nz=192 with no step at the 129->193 tier boundary, and no rung
  reserves any local-memory backing store: the widest frame is 800 B against
  the 1,024 B default stack limit, so the ~8.0 MiB reservation the acceptance
  document prices arrives only above nz=192. Receipt:
  `docs/superpowers/receipts/les/p2-nz-tier-2026-08-04/P2-DEVICE-ACCEPTANCE.md`.
  **`nz > 192` is still unrun**: the 257 tier is compiled, frame-measured
  (1,056 B) and gated, but no integration has been run there.
- **Radiation, not the acoustic solver, is the vertical ceiling on a
  real-data run.** Every bound below is refused by name at the door
  (`woof check`; `woof/physics_compat.py`, single-sourced from
  `woof/physics_vertical_contract.py`). The radiation bounds count model
  levels PLUS the cap layers above the model top, so they move with `p_top`:

  | component | the bound | max `nz` at `p_top = 5,000 Pa` |
  |---|---|---|
  | legacy RRTMG, shortwave | model + wrapper layers <= 64 | **63** |
  | RTE+RRTMGP, longwave | model + cap layers <= 128 | **115** |
  | legacy RRTMG, longwave | model + cap layers <= 128 | **115** |
  | WSM6 microphysics | `nz <= 80` | 80 |
  | Kain-Fritsch cumulus | `nz <= 128` | 128 |
  | YSU / Shin-Hong / MYJ PBL, SASE | `nz <= 128` | 128 |
  | Kessler / Thompson / Thompson-aerosol / Morrison / NSSL-2 | `nz <= 256` | 256 |
  | MYNN PBL, Grell-Freitas cumulus | none | 256 (the solver's) |
  | acoustic solve | `nz <= 256` | 256 |

  The deep runs above were idealized dry CBL cases with radiation off, which
  is why they reached 160 and 192. **A full-physics real-data column is capped
  at 115 levels by RRTMGP at the usual `p_top`, and at 63 if legacy RRTMG
  shortwave is selected**, both below the acoustic solver's 256, and the
  shortwave one below every other ceiling in the model. The cap-layer counts
  fall as `p_top` rises: at `p_top = 10,000 Pa` the RRTMGP ceiling is 103.
  §2 prices what the vertical grid costs: at 128 the resolved fraction is
  0.921, and the nested child's shared 49-level grid gives up 4.8 points
  of it. The battery extends that sweep, strictly monotone: **0.92476 at
  nz=160 and 0.92941 at nz=192**.
- **Vertical nesting is impossible in an INLINE nest tree, and available
  offline.** The vertical grid of a running tree is single-sourced from
  `ExperimentConfig.vertical`, and per-domain vertical keys are rejected
  outright (`woof/experiment.py`, `_DOMAIN_VERTICAL_KEYS`): the inline
  corridor interpolates boundaries every parent step on the GPU and has no
  vertical operator there. A 250 m child running *inside* the tree does
  therefore run its parents' level count. To give it more, run it as an
  offline child off archived parent history: `woof downscale
  --child-levels 128,2.5` builds it on its own ladder, remapping the
  initial state and the whole lateral-boundary table set once at
  preparation in FP64. That is the door that recovers the 4.8 points of
  resolved TKE the shared 49-level grid gives up, measured in §2 above.
- **`km_opt=2` on a nest cold-starts its own TKE, under any parent.** WRF
  gives `tke` no `i` (nest-interpolation) and no `f` (feedback) Registry
  flag, so a child cold-starts its own TKE and never feeds it back, and
  WOOF's nest forcing carries no TKE row either. Under a parent carrying
  no TKE that case has been run: a 250 m `km_opt=2` child under a
  `km_opt=4` parent, 7 h, PASS, carrying 4.8x to 9.9x the parent's resolved
  w variance and *leading the `km_opt=3` child at every output time*
  despite the cold start. Under a `km_opt=2` parent the child does the
  same thing and the parent's TKE stays on the parent, as in WRF; that
  tree loads with a warning that it is implemented and not yet verified.
- **`km_opt=2` requires `bl_pbl_physics=0`** (see §1).
- **Per-domain `isfflx` is an WOOF-over-WRF extension.** `isfflx` is
  `nentries=1` in the WRF Registry: a scalar. WOOF's TOML schema admits it
  per domain, which WRF cannot express, so a configuration that uses it has
  left WRF-expressible territory and cannot be round-tripped back to a
  namelist. The namelist importer deliberately reads it as the scalar WRF
  spells.
- **One registered runtime divergence, D8, and it is inert in most of
  what is measured here.** WRF hands `vertical_diffusion_w_2` the
  *horizontal* momentum coefficient `xkmh`, while every other stress
  operator takes the coefficient of its own directions. WOOF hands it
  `xkmv`. That can only change an answer where the two coefficients
  differ, i.e. `km_opt` 2 or 3 **with `mix_isotropic = 0`**. The
  idealized CBL case runs `mix_isotropic = 1`, where the two are the same
  number (the WRF oracle lane measured `XKMH == XKMV` at all 589,824
  points, maximum difference exactly 0) so **every idealized result on
  this page is unaffected by D8**. It was essential on the nested
  250 m child while that child ran `mix_isotropic = 0`: at dx 250 m
  against dz 17 m the pairing is roughly two orders of magnitude in the
  explicit diffusion number, and taking WRF's is unstable. **Since
  2026-08-09 no config offered as a starting point runs
  `mix_isotropic = 0` on an LES child** (below), so D8 is arithmetically
  inert on everything a user would begin from. It is still live on the
  five archived records that reproduce committed runs, and it stays
  registered because the divergence is in the operator, not in the
  configs. Full argument in `PROVENANCE.md`, section D8.

### The mixing length on every shipped LES child, and why it is 1

`mix_isotropic = 0` gives WRF per-axis mixing lengths, and on a grid
whose layers are much deeper than the grid is wide that is a trap that
has cost this project a run. WRF hands `horizontal_diffusion_w_2` the
**vertical** exchange coefficient, `smag_km` both builds and caps that
coefficient on the layer depth (`xkmv <= mix_upper_bound*dz²/dt`), and
the operator then differences it over `dx`. Nothing compares the two, so
the reachable `K·dt/dx²` is `mix_upper_bound·(dz_max/dx)²`: independent
of `dt`, because the cap carries `1/dt` and the ratio carries `dt`. An
explicit Laplacian multiplies a 2Δx mode by `1 − 4K·dt/dx²` per step:
past **1/4** the sign flips, past **1/2** the mode grows.

| tree | domain | ratio when it ran at `mix_isotropic = 0` | now |
|---|---|---|---|
| nested 250 m (`les_nest_250m_km3`, `les_nest_250m_grayzone`) | d03, 250 m | **0.702** (2.8×) | not on the path |
| 100 m tornado (`les_tornado_100m_*`) | d03, 500 m | 0.169 | not on the path |
| 100 m tornado | d04, 100 m | **4.23** (17×) | not on the path |

The `attempt2` and `attempt2b` files at the top level of `configs/`, and
the three archives under `configs/frozen/`, still run at those ratios.
They are records of committed runs rather than configurations to start
from, and each is pinned to its sha256 so the exemption cannot widen.

The 4.23 tree aborted at step 5467 with `w = 239.48 m/s`, bit-identical
across three instrumented reproductions. The 0.702 trees completed
multi-hour runs at that ratio, which is the whole reason the criterion
is an **advisory and not a refusal**: it is the worst case the cap
admits, and five completed runs at 2.8× the limit are the receipt-backed
evidence that a flow need not reach it. There is a second, weaker
observation pointing the same way: the attempt #2b post-mortem records
a realised 1.55 at the failing cell against a criterion of 4.23, written
down in prose in
`configs/les_tornado_100m_mayfield_20211210_attempt3.toml:232` and
nowhere else, with no instrument output behind it. Treat the ~3×
overshoot it implies as an indication, not a measurement; the refusal
argument does not rest on it.

What ships instead is a clean set plus a check that keeps it clean:
`tests/test_shipped_configs_mixing_stability.py` fails if any config
under `configs/` arrives on the exposed path, and `woof check`
repeats the mixing lines in its report (and under `advisories` in
`--json`) rather than leaving them at config load. The check reads the
layer depths from the config's eta ladder, and **resolves that ladder
from `nz`/`ztop` when a config does not write one out**: a config is
not safer for having left the interfaces implicit.

**The stable length is the default (auto-switch, 2026-08-16).** A
domain that leaves `mix_isotropic` unset (or writes the sentinel
`"auto"`) and violates the criterion **runs `mix_isotropic = 1`**: the
selection is made once at config load, announced with one line naming
the ratio and the limit, and `woof check` reports that the run WILL
use isotropic mixing rather than advising that something is wrong. A
config that **writes `mix_isotropic = 0` keeps it**, in the danger zone
too; what it gets is the advisory above, now carrying the override
state. Because `mix_isotropic` is inside the restart fingerprint, a
checkpoint written under the old anisotropic default does not
bit-continue under the auto-selected isotropic form, both restart
doors say so in one line, and writing `mix_isotropic = 0` explicitly is
the way to resume such a checkpoint.

**The receipts on this page belong to the pre-change bytes.** Every
measured nested number in §2 was produced with `mix_isotropic = 0`; the
exact files are archived at `configs/frozen/` and the receipts' sha256
digests still resolve there. The shipped configs have not been
re-scored, and `mix_isotropic` is inside the restart fingerprint, so no
checkpoint crosses the change.

---

## 5. Open: what is NOT claimed

- **The WRF `em_les` comparison is measurement, not a maturity
  promotion.** §2's head-to-head against the independently built v4.6.1
  reference cut no pass band anywhere; differences are reported, not
  judged, and the one registered corroboration band was committed before
  the data existed. The comparison is statistical because it can only be:
  WRF's ideal-case perturbation is unseeded and decomposition-dependent,
  so the two initial conditions cannot be made equal and no deterministic
  run-for-run comparison exists to schedule. No IC-perturbed ensemble was
  run on either side, so no envelope exists and nothing weaker is
  reported in its place. LES stays **implemented-unverified**.
- **No resolved-fraction figure for `km_opt=3`.** The TKE-based metric needs
  a prognostic SGS TKE and Smagorinsky has none. Until a diagnostic SGS TKE
  is derived from the Smagorinsky `K`, the flux-based measure in §2 is the
  only resolved-fraction statement for that closure.
- **Idealized coverage is moist in transport only, and flat.** The MOST
  surface mode runs `moist=True` with `mp_physics=0`, so water vapour is
  advected and mixed by the LES closure through
  `vertical_diffusion_2`'s moist rows, but nothing condenses, so no
  phase change or latent heating is exercised idealized. **No idealized
  terrain LES exists at all.** The nested run is both moist (WSM6) and
  terrain-following over 225–947 m orography, but it is one case scored
  against its own parent rather than against a reference, so it
  demonstrates the closure works there: it does not verify it there.
- **The nested results are one case, in one window, under one parent
  closure.** Both `km_opt=3` and `km_opt=2` have run there, but the parent
  was `km_opt=4` in both; a `km_opt=2` parent is admitted and unverified.
- **The nested child is COARSE LES at the gray-zone edge**, because it runs
  its 3 km grandparent's 49 shared levels: measured, 18 inside the 1741 m
  boundary layer, an effective dz of 96.7 m. Vertical resolution, not the
  250 m spacing, is the binding constraint, and §2 now says what it costs
  rather than leaving it as a caveat.
- **Per-domain LES selection is a configuration capability,
  implemented-unverified.**
- **The WP-L9 archive is not a like-for-like reference.** It holds 18
  distinct weather cases rather than perturbed realizations of one, and it
  was produced with `km_opt=5` (SMS-3DTKE) on WRF v4.7.1: a different
  closure on a different version. It cannot verify these two configurations or validate them against observations.
