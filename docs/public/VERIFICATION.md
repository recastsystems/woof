# Verification and validation evidence

This page separates implementation checks, numerical-error estimates and
comparisons against observations. Evidence is tied to its reference version,
engine revision, configuration and measured quantities. A historical receipt
is not a measurement of a later engine revision. Per-option evidence and
declared deviations are listed in [PHYSICS.md](PHYSICS.md).

## Verification and validation

The terminology follows [ASME V&V 10/20](https://www.asme.org/codes-standards/publications-information/verification-validation-uncertainty),
the [AIAA G-077 guide, summarized in the NASA V&V tutorial](https://www.grc.nasa.gov/www/wind/valid/tutorial/tutorial.html),
and [Verification and Validation in Scientific Computing](https://assets.cambridge.org/97805211/13601/frontmatter/9780521113601_frontmatter.htm).

- **Code verification:** does the code solve its intended equations correctly?
  Evidence includes reference-implementation comparisons, component oracles,
  ULP and byte-identity checks, ensemble consistency tests, and analytic,
  manufactured-solution or convergence tests. Each checks only the operations
  and conditions it exercises; agreement between two codes can share errors.
- **Solution verification:** estimating numerical error in a particular run,
  including spatial and temporal discretization, iteration and roundoff error.
  A stability check or agreement with another model does not supply this estimate.
- **Validation:** assessing how well the model represents reality by comparing
  with observations for a stated use, including uncertainties in both simulation
  and observations. Meteorology calls scoring against observations **forecast
  verification**; that activity is validation in computational-science V&V terms.
  This page uses the V&V terms below. Input and configuration validation retain
  their separate software meaning.

**Matching WRF is verification. It never validates a forecast.** WOOF can
inherit WRF's published validation record only to the extent that the two are
statistically indistinguishable for that configuration and the quantities and
conditions covered by that record. A passed consistency test has finite power;
it does not prove equivalence for unmeasured fields, regimes or later revisions.
Neither a WRF match nor an isolated observation score establishes general
forecast accuracy. The [ASME V&V 20 scope](https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-fluid-dynamics-and-heat-transfer)
likewise ties accuracy assessment to specified variables and validation points.

### Evidence by kind and reference version

| Kind | Evidence and reference | Scope and remaining gap |
|---|---|---|
| Code verification | Historical component Fortran oracles and ULP checks: **WRF v4.6.1**, commit `d66e442fccc04111067e29274c9f9eaccc3cef28`; sections 1 and 5 and [PHYSICS.md](PHYSICS.md). Newer component ports identify **v4.7.1** explicitly in their option rows, for example UW PBL and urban physics. | Coverage differs by routine; transcription and self-consistency are distinguished from an independent Fortran oracle. A reference version belongs to the individual fixture, not to every component in the release. |
| Code verification | Historical six-hour, four-domain comparison, initial-state digest and decay tables below: **WRF v4.6.1**, same commit; WOOF run of record `152f7d31`, 2026-07-28. | Initial states fail the recorded ceilings on all four domains. Differences combine initialization and forecast evolution; the tables are not evidence for all current configurations. |
| Code verification | Byte identity for named component oracles and same-build replay/restart checks; [DETERMINISM.md](DETERMINISM.md). | WRF-based oracle claims use that oracle's version. Same-build checks have no external WRF reference and do not establish physical accuracy. |
| Code verification | Light ensemble consistency test, 2026-10-02: 100 six-hour references per regime from **WRF v4.7.1**, commit `f52c197ed39d12e087d02c50f412d90d418f6186`, strict GNU flags `-O2 -fno-fast-math -ffp-contract=off`; WOOF `e459f79ed4c5ae74e8a23d6e1c6f66c1519dc1e1`. | Three WOOF runs per regime; hour-six area-weighted means of 11 fields. Convective and winter pass; coastal fails, so the overall three-regime result fails. The fixed rule rejects at least one regime in 33/5,000 held-out WRF resampling trials (0.66%); this is conditional on the sampled cases and can miss spatial error. |
| Code verification | Descriptive 20-member envelope across five 24-hour cases: **WRF v4.7.1**, same commit and strict flags; corrected WOOF `1b15157c0f9803f75c2ecd9fb3a100c706bcf453`. | WOOF is at or below the largest reference-member RMSE from the WRF mean in 1,098/1,440 field-hours; a different WRF compiler build is within it in 941/1,440. The declared whole-case consistency result is 0/5. The envelope and accompanying PCA screen have no calibrated significance level and do not establish statistical indistinguishability. |
| Code verification | Analytic and component convergence checks, including the NumPy advection reference's smooth-profile convergence test in `tests/test_advection.py::test_convergence_order`; no WRF version applies. | These are component tests, not a full-model manufactured-solution campaign or an error estimate for a weather forecast. |
| Solution verification | No published run-specific numerical-error budget for the weather forecasts described here. | Grid/time-step refinement studies and quantified discretization, iteration and roundoff contributions remain gaps. The nested-grid comparisons below and ensemble spread are not substitutes. |
| Validation against observations | Limited ASOS case scoring is described in the [snow/soil initialization note](../soil-texture-downscaling.md); the [physics selection record](PHYSICS.md#mynn-scope-note-what-composes-and-what-is-pinned) reports a small station/ceilometer fog comparison without a published case count or score receipt. MRMS reflectivity scoring of one model-top A/B has a [retained receipt](receipts/ptop-default-ab/RECEIPT.md). WRF version: not applicable as the reference is observations. | These specific comparisons are distinct from the preregistered observation battery, which has no scored forecast result in its published receipts. They do not establish skill across seasons, regions, physics suites or leads, or a complete numerical and observational uncertainty budget. |

The [WOOF-versus-WRF ensemble consistency summary](receipts/wrf-consistency-20261002/SUMMARY.md)
and its charts record the current fidelity instrument, including the coastal
failure and the limits of the descriptive envelope. They describe the engine
snapshots identified above, not a new test of every release assembled from them.

The observation scoreboard is in development. Its implementation and data
plumbing are not completed multi-case validation results. A broad observation
campaign with stated uncertainty and coverage remains a gap. The two ensemble
campaigns above are retained evidence summaries; their full replay bundles are
not distributed with this page, so this checkout alone cannot reproduce them.
The reproduction recipe in section 7 applies only to the historical v4.6.1
comparison. The v4.7.1 ensemble results do not replace or re-label it.

## 1. Methodology

The historical WRF v4.6.1 comparison uses three measurements and a review
process; their conclusions are not interchangeable:

1. **Component ULP comparisons.** An independent reference oracle
   drives the byte-unmodified WRF v4.6.1 Fortran (compiled from
   the pinned commit) over fixture columns and dumps inputs and
   outputs; the CUDA port is compared field by field in units of FP32
   ULP (units in the last place). Transcription and self-consistency
   fixtures are separate forms of evidence. Available tools and decks include
   `tools/noah_wrf461_oracle`, `tools/ysu_wrf461_oracle`,
   `tools/morrison_wrf461_oracle`, the Noah-MP and RUC column oracles,
   and the legacy-RRTMG fixture decks. Where a routine is bit-exact the
   gate pins max ULP 0 (for example, the batched legacy-RRTMG LW and SW
   engines are bit-identical to their transcription oracles over the
   full fixture decks at four chunk sizes); where it is not, the
   measured distance and any identified cause are recorded in the physics registry
   rather than hidden behind a tolerance (see
   [PHYSICS.md](PHYSICS.md)).

2. **t=0 full-state digest.** WOOF's own ingest opens the same
   analysis data as the WRF reference chain, and the two initial
   states are then scored array by array -- every registered carrier
   group, on every domain both runs wrote, against ceilings pinned in
   `woof/verify/nest_gates.py` before any of these frames existed.
   The measurement is published as it comes out. On the reference case
   the two t=0 states do not agree within those ceilings: verdict
   **FAIL**, on all four domains
   ([receipt](../../woof/data/certification/t0_state_parity_digest.json),
   [table](../../woof/data/certification/t0_state_parity_digest.md)).
   The artifact's historical name is "full-state digest", but its
   coverage rule requires at least one scored array per required group,
   not every runtime carrier. Absent variables and unavailable boundaries
   remain unmeasured. A history frame also need not contain restart-only
   state. Establish initial time, expected domains, field applicability,
   units, staggering and vertical coordinates separately before using
   this receipt as evidence of a shared initial condition.

3. **Matched-run protocol.** The model integrates a real case with
   physics, geometry, and output cadence matched to a WRF v4.6.1 CPU
   reference run, and a streaming comparator
   (`tools/matched_wrfout_stream_compare.py`) scores every output frame
   on the interior grid (5-row rim excluded): T2 MAE, PSFC MAE,
   composite-reflectivity correlation and MAE, CSI at the 20 dBZ
   threshold, W correlation, and 10 m wind correlation. Nothing is
   summarized until every frame is scored; the full decay tables are
   published, not just the flattering leads.

4. **Automated adversarial review.** AI review tools examined the ports
   and their evidence for contradictions and unsupported claims. This is
   a review process, not a measurement or independent verification by
   outside scientists.

## 2. The reference case

The historical matched-run case is a historical severe-weather reference day
(ERA5-initialized), four one-way nested domains at 12 km / 3 km / 1 km
/ 500 m, integrated
12Z-18Z with Thompson microphysics (WRF's own tables, hash-pinned),
YSU PBL, MM5 surface layer, Noah LSM, Kain-Fritsch on the root, and the
legacy-RRTMG transcription -- the same option set as the CPU reference.

The CPU reference is WRF v4.6.1 at the pinned commit, built with **GNU
gfortran 15.2.0** (WRF `configure` option 34, gfortran/gcc, dmpar; Intel
oneAPI supplies the MPI layer only), run on 48 MPI ranks. The GPU run is
one RTX 5090. Both write the same history cadence (d01-d03 hourly, d04
half-hourly).

> **Corrected in 1.4.1.** This page previously described that build as
> Intel `ifx`. It was not: the pinned binary's own bytes carry exactly
> one compiler stamp, `GCC: (Ubuntu 15.2.0-16ubuntu1) 15.2.0`, and no
> Intel Fortran signature. The measurement, the `configure` transcript
> and the full flag set are in the
> [build recipe](wrf-reference/10d6e178d96648e35b826217cc001a1f598232bf6ceabd9ce07074a1f65e2031.build-recipe.md)
> committed beside the manifest that pins the binary. The reference
> stream itself is unchanged -- only the description of how it was
> compiled was wrong.
>
> One config in this repository still carries the old wording as a bare
> statement, and it is named here by its contents rather than fixed:
> the four-domain 12-18Z Thompson / legacy-RRTMG reference config under
> `configs/`, whose header
> comment reads `WRF v4.6.1 d66e442f, ifx, 48 ranks`. That file's
> SHA-256 **is** the `config_sha256` this manifest, the acceptance band
> and the certification capsule are all addressed by, so correcting a
> comment inside it would move a digest the certification chain depends
> on. Read that line as `gfortran 15.2.0, 48 ranks`; the `48 ranks` and
> the pinned WRF commit in it are correct.

### t=0 comparator metrics (all four domains)

| t=0 | T2 MAE / corr | PSFC MAE | 10 m wind corr |
|---|---|---|---|
| d01 | 0.000 K / 1.000 | 0.1 Pa | 0.999 |
| d02 | 0.000 K / 1.000 | 0.9 Pa | 1.000 |
| d03 | 0.000 K / 1.000 | 0.9 Pa | 1.000 |
| d04 | 0.000 K / 1.000 | 0.1 Pa | 1.000 |

Those are the streaming comparator's own four metrics on the interior
grid, and they are not a statement about the initial state. Scored in
full -- every registered carrier group, every element of every array,
against the pinned ceilings -- the two t=0 states do not agree: verdict
**FAIL** on all four domains
([receipt](../../woof/data/certification/t0_state_parity_digest.json),
[table](../../woof/data/certification/t0_state_parity_digest.md)).
Only the accumulation group passes as a whole on every domain; some
individual arrays in other groups are also bit-identical. On d01 the
reported maximum absolute differences include 66 Pa in perturbation
pressure, 0.75 m in terrain height, 296 K in soil temperature, and a
land-use code difference of ten. The receipt does not record the
extremum's index, soil depth, active mask or both compared values, so it
does not by itself locate or explain the soil-temperature discrepancy.
Land-use codes are categories, not a physical distance scale. The receipt
carries the per-array numbers for all four domains. The tables below
therefore contain initial-state
differences as well as forecast divergence, and the digest is where the
size of the former is written down.

## 3. Matched-run results (2026-07-28 rerun)

Two scored snapshots are highlighted because they bracket convective
initiation: 15Z (+3 h, squall line organizing) and 18Z (+6 h, mature
cell-scale convection). "Old" is the previous matched run of the same
case (2026-07-27, before a series of seam closures); "new" is the
development lineage at 2026-07-28, commit `152f7d31`. Neither is the
current release engine, and later changes have not been rerun on this
case. The new run is closer to the WRF reference on most of these metrics
at these two leads. Each column is one deterministic run: small changes,
such as W correlation 0.130 to 0.138, cannot be separated from trajectory
sensitivity without a matched control ensemble and are not evidence of
improved accuracy. Differences in the tables are signed new-minus-old.

### d03 (1 km), 18Z -- the verdict lead

| metric | old | new | new minus old |
|---|---|---|---|
| T2 MAE | 0.565 K | 0.347 K | -0.218 K |
| PSFC MAE | 22.91 Pa | 20.45 Pa | -2.46 Pa |
| refl-comp corr | 0.717 | 0.715 | -0.002 |
| refl-comp MAE | 9.81 dBZ | 9.75 dBZ | -0.06 dBZ |
| CSI (20 dBZ) | 0.377 | 0.425 | +0.048 |
| W corr | 0.130 | 0.138 | +0.008 |
| wind10 corr | 0.891 | 0.901 | +0.010 |

### d03 (1 km), 15Z

| metric | old | new | new minus old |
|---|---|---|---|
| T2 MAE | 0.281 K | 0.046 K | -0.235 K |
| PSFC MAE | 7.15 Pa | 5.20 Pa | -1.95 Pa |
| refl-comp corr | 0.976 | 0.981 | +0.005 |
| refl-comp MAE | 1.63 dBZ | 1.06 dBZ | -0.57 dBZ |
| CSI (20 dBZ) | 0.660 | 0.670 | +0.010 |
| W corr | 0.366 | 0.333 | -0.033 |
| wind10 corr | 0.937 | 0.963 | +0.026 |

### d02 (3 km), both leads

| d02 | old 15Z | new 15Z | old 18Z | new 18Z |
|---|---|---|---|---|
| T2 MAE | 0.2365 K | 0.0514 K | 0.4449 K | 0.165 K |
| PSFC MAE | 4.666 Pa | 3.155 Pa | 15.480 Pa | 11.55 Pa |
| refl corr | 0.9782 | 0.9848 | 0.9127 | 0.929 |
| refl MAE | 1.821 dBZ | 1.326 dBZ | 4.768 dBZ | 4.02 dBZ |
| CSI (20 dBZ) | 0.8078 | 0.8573 | 0.6518 | 0.682 |

At d02 15Z the new run has 14,230 pixels at or
above 20 dBZ against WRF's 14,227 -- a 3-pixel difference in echo
coverage where the old run was off by 295. The mean reflectivity difference
from WRF changed from -0.311 dBZ to -0.004 dBZ. This is one snapshot
chosen after the fact. Matching coverage does not show that echoes occupy
the same locations, and neither number measures bias against observations.

### Full decay table (all domains, all leads)

Interior grid, 5-row rim excluded. F1..F6 are forecast hours; d04 is
scored half-hourly. CSI `nan` means neither model had any >=20 dBZ
echo -- by design, not an error.

| dom | lead | T2 MAE K | PSFC MAE Pa | refl corr | refl MAE dBZ | CSI20 | W corr | wind10 corr |
|---|---|---|---|---|---|---|---|---|
| d01 | F1 | 0.012 | 0.50 | 0.970 | 1.58 | 0.863 | 0.984 | 1.000 |
| d01 | F2 | 0.019 | 0.80 | 0.971 | 1.84 | 0.854 | 0.975 | 1.000 |
| d01 | F3 | 0.025 | 1.09 | 0.963 | 2.13 | 0.845 | 0.963 | 1.000 |
| d01 | F4 | 0.030 | 1.37 | 0.953 | 2.44 | 0.820 | 0.945 | 1.000 |
| d01 | F5 | 0.034 | 1.78 | 0.945 | 2.72 | 0.800 | 0.922 | 1.000 |
| d01 | F6 | 0.041 | 2.21 | 0.939 | 2.86 | 0.785 | 0.909 | 1.000 |
| d02 | F1 | 0.014 | 1.02 | 0.996 | 0.41 | 0.931 | 0.994 | 1.000 |
| d02 | F2 | 0.029 | 1.87 | 0.994 | 0.80 | 0.897 | 0.872 | 0.998 |
| d02 | F3 | 0.051 | 3.15 | 0.985 | 1.33 | 0.857 | 0.749 | 0.997 |
| d02 | F4 | 0.076 | 4.24 | 0.975 | 1.85 | 0.794 | 0.743 | 0.995 |
| d02 | F5 | 0.116 | 6.92 | 0.951 | 2.80 | 0.750 | 0.580 | 0.992 |
| d02 | F6 | 0.165 | 11.55 | 0.929 | 4.02 | 0.682 | 0.375 | 0.988 |
| d03 | F1 | 0.010 | 0.92 | 0.996 | 0.26 | nan | 0.999 | 1.000 |
| d03 | F2 | 0.015 | 1.21 | 0.997 | 0.33 | 0.708 | 0.986 | 0.998 |
| d03 | F3 | 0.046 | 5.20 | 0.981 | 1.06 | 0.670 | 0.333 | 0.963 |
| d03 | F4 | 0.105 | 7.93 | 0.933 | 2.70 | 0.428 | 0.358 | 0.946 |
| d03 | F5 | 0.186 | 10.00 | 0.795 | 5.35 | 0.313 | 0.370 | 0.954 |
| d03 | F6 | 0.347 | 20.45 | 0.715 | 9.75 | 0.425 | 0.138 | 0.901 |
| d04 | F0.5 | 0.004 | 0.34 | 0.990 | 0.08 | nan | 0.998 | 1.000 |
| d04 | F1 | 0.007 | 0.35 | 0.997 | 0.11 | nan | 0.998 | 1.000 |
| d04 | F1.5 | 0.006 | 0.46 | 0.995 | 0.13 | nan | 0.994 | 1.000 |
| d04 | F2 | 0.007 | 0.70 | 0.997 | 0.14 | nan | 0.988 | 1.000 |
| d04 | F2.5 | 0.014 | 0.63 | 0.999 | 0.25 | 0.023 | 0.942 | 0.998 |
| d04 | F3 | 0.026 | 2.10 | 0.998 | 0.40 | 0.136 | 0.904 | 0.995 |
| d04 | F3.5 | 0.047 | 4.25 | 0.923 | 0.76 | 0.249 | 0.735 | 0.982 |
| d04 | F4 | 0.107 | 7.27 | 0.870 | 3.01 | 0.516 | 0.508 | 0.951 |
| d04 | F4.5 | 0.173 | 10.99 | 0.832 | 5.09 | 0.328 | 0.257 | 0.915 |
| d04 | F5 | 0.268 | 16.37 | 0.667 | 8.32 | 0.418 | 0.216 | 0.879 |
| d04 | F5.5 | 0.309 | 18.78 | 0.680 | 11.36 | 0.391 | 0.112 | 0.842 |
| d04 | F6 | 0.434 | 22.70 | 0.577 | 14.20 | 0.222 | 0.110 | 0.795 |

### Determinism

The run survived two external process kills; frames produced before
each kill were copied aside and byte-compared when regenerated after
relaunch: SHA256-identical (d03 and d04 checked explicitly). WOOF
reproduces its own trajectory bit-for-bit under restart-free relaunch
on the same hardware and build.

## 4. Late fine-mesh divergence: measured, not causally attributed

On d03, W correlation falls from 0.986 at F2 to 0.333 at F3 and ends at
0.138 at F6. On d04 it ends at 0.110. These are measured differences
between the historical forecasts, not a diagnosis of their cause.

Roundoff-sized differences can grow in convective flow. In the separate
[idealized warm-bubble comparison](wrf-comparison/mp28-matched-trajectory.md),
section 10, WRF against its own one-flag recompilation differed by 0.10%
RMS in `w` at 600 s, versus 3.2% for the port against WRF, and the WRF
control had not saturated by 7200 s. That control does not explain away
the much larger port difference. Ensemble consistency tests compare the
port with a distribution of perturbed reference runs; the v4.7.1 tests
above are that kind of instrument, but do not cover this historical
four-domain case. Convective sensitivity and displacement are plausible contributors.
However, the initial-state digest fails, boundary tables were not
retained, and these tables contain no matched reference-versus-reference
control that establishes a nondegenerate sensitivity envelope for this
configuration and window. The tables therefore do not exclude ingest,
geometry, coupling, physics or dynamics errors. Alternating signs in
peak-reflectivity differences do not establish absence of intensity bias.

At d02 F6, reflectivity correlation is 0.929, 10 m wind correlation is
0.988 and T2 MAE is 0.165 K. That narrower agreement remains useful
model-versus-model evidence; it neither proves observed forecast skill
nor supplies a causal explanation for differences on the finer domains.

A zero sensitivity envelope is not a passing chaos allowance. These
historical scores do not establish a tolerance for a later forecast.

## 5. Worldwide projections: what their shallower tier means

Lambert conformal (both hemispheres), Mercator, and polar
stereographic (both poles) run end to end -- wizard, config, static
build, ERA5/GFS ingest, native WRF export -- including
antimeridian-crossing domains. Their verification tier is stated here
with the same prominence as the reference case because it is
deliberately **shallower**: transcription oracle plus GPU smoke
integrations, **not** matched-run.

### Projection transcription oracle (binary64)

The projection mathematics is transcribed from the pinned WRF v4.6.1
`share/module_llxy.F` (plus the WPS v4.6.0 geogrid map-factor and
rotation-angle formulas) and gated against a committed fixture of
IEEE-754 binary64 words produced by that unmodified Fortran compiled
at real-8 (gfortran 13.3.0 / glibc 2.39). Nine projection
configurations cover Lambert (NH secant, SH secant, SH tangent),
Mercator (tropical, subtropical, antimeridian) and polar
stereographic (NH, SH, pole-anchored). Every transform -- setup
constants, lat/lon to grid, grid to lat/lon, map factor, wind
rotation -- is compared in binary64 ULPs against per-quantity
ceilings pinned exactly in `tests/test_projection_oracle.py`
(`ULP_CEILINGS`):

| projection | worst pinned ceiling | worst slot |
|---|---|---|
| Lambert conformal | 32 ULP | lat/lon -> j |
| Mercator | 8 ULP | grid -> latitude |
| Polar stereographic | 2 ULP | grid -> lat/lon |

The only drift source is numpy libm vs glibc libm (the transcriptions
are operation-identical). The Lambert map factor additionally carries
a 2.3e-16 relative bound: the product ships the ARW tech-note form
referenced to `truelat1`, geogrid the mathematically identical form
referenced to `truelat2`. A mutation control (truelat1 perturbed by
1e-3 must overflow every ceiling) proves the suite can fail.

### GPU smoke integrations (four worldwide sites)

Wizard-emitted single-domain 12 km configs (116x94x49, the default
physics suite), integrated on the GPU from GFS initialization at the
2026-07-29T06 cycle:

| site | projection exercised | simulated | verdict |
|---|---|---|---|
| Brisbane | Lambert, southern hemisphere | 3 h | PASS -- finite everywhere, no NaN |
| Singapore | Mercator, near-equatorial | 2 h | PASS -- finite everywhere, no NaN |
| Fairbanks | polar stereographic | 3 h | PASS -- finite everywhere, no NaN |
| Fiji | Mercator across the antimeridian | 3 h | PASS -- finite everywhere, no NaN |

Each run's final model state is digested field-by-field (157 arrays)
and the digests, configs, and per-site reports are retained as
machine receipts in the development tree (`evidence/worldwide-smoke/`;
retained outside the release snapshot).

### The boundary, stated plainly

**No matched-run verification exists for the new projections.** No
WRF twin has been integrated on Mercator, polar stereographic, or
southern-hemisphere Lambert; the matched-run family of sections 2-4
is northern-hemisphere Lambert only. A smoke integration proves the
pipeline executes and stays finite -- it does not measure forecast
agreement. Configurations on the new projections inherit the
transcription oracle above, the component-level physics evidence
([PHYSICS.md](PHYSICS.md), including its projection maturity rows),
and these smoke receipts -- nothing more.

## 6. What is claimed, and what is not

Claimed, each with its receipt above or in the linked pages:

- A published full-state t=0 digest of the reference case on all four
  domains, carrying the result it returned: the two t=0 states do not
  agree within the pinned ceilings -- verdict **FAIL**
  ([receipt](../../woof/data/certification/t0_state_parity_digest.json)).
  Parity of the initial state at the FP32/operator floor is not
  claimed; see "Not claimed" below.
- Named component routines bit-exact or measured-ULP-close to
  unmodified WRF v4.6.1 Fortran, per the physics registry's per-option
  records ([PHYSICS.md](PHYSICS.md)).
- Historical model-versus-model forecast agreement on the reference
  case at the levels tabulated in section 3, between runs that did not
  start from the same state.
  This is not a pure test of time integration and does not verify
  subsequent engine changes or a different physics configuration.
- Bit-deterministic re-execution on fixed hardware and build.
- Unchanged stock WRF v4.6.1 accepts and integrates this
  preprocessor's outputs, within the stated boundaries
  ([WRF-INTEROP.md](WRF-INTEROP.md)).
- Projection transcription parity at the pinned binary64 ULP ceilings
  for all three projections, plus finite-state GPU smoke integrations
  at four worldwide sites (section 5).

**Not claimed:**

- **The initial states do not match at the FP32 floor.** The
  full-state t=0 digest scores five of its six covered carrier groups
  outside the pinned ceilings on every domain
  ([receipt](../../woof/data/certification/t0_state_parity_digest.json)):
  verdict **FAIL**. Section 2's four surface metrics are small, and
  they were never evidence for the rest of the state. Neither side of
  the staged pair retained a `wrfbdy_d01`, so the lateral-boundary
  tables are recorded as unavailable and no boundary statement is made
  anywhere on this page.
- **The new projections are not matched-run verified.** Mercator,
  polar stereographic, and southern-hemisphere Lambert carry the
  section-5 tier only; no WOOF-vs-WRF forecast comparison exists on
  them.

- **No end-to-end bit-exactness with WRF.** The model state is FP32,
  GPU transcendentals differ from glibc/Intel libm at the ULP level,
  and several deliberate deviations from WRF are registered in
  [PROVENANCE.md](../../PROVENANCE.md). "Bitwise" statements are always
  scoped to a named comparison (a kernel oracle, a restart identity, a
  dual-run byte comparison) -- never to WRF output files. What the
  dual-run byte comparison covers, and what it cannot detect in place
  of ECC, is [DETERMINISM.md](DETERMINISM.md).
- **One historical case has detailed matched-run tables.** Those tables cover one
  meteorological situation, one season, one region, one option set.
  Other cases, seasons and physics combinations require their own evidence.
  The v4.7.1 ensemble campaigns above are separate, narrowly scoped checks.
- **Part of the 2026-07-28 improvement is by construction.** That
  rerun matched the reference configuration (legacy RRTMG, matched
  cadence and geometry); it shows the agreement reached by the matched
  configuration, not a universally improved solver. The old/new
  comparison also spans two output-writer versions (4-byte file-size
  difference; frames are identified by SHA256, never by size).
- **No observation-validation claim from the historical comparison.**
  Sections 1-7 describe model-versus-model and implementation checks.
  The separate ASOS/MRMS evidence listed above assesses particular runs
  against observations; it does not establish general forecast skill.
- **No resolved tornado dynamics.** The 500 m nest and the STP/UH
  severe suite characterize the tornadic-supercell environment and
  mesocyclone-scale morphology; they do not resolve the near-surface
  corner flow, suction vortices, or tornado-scale wind intensity
  (tornado-like-vortex resolution in the literature sits below
  roughly 25 m horizontal and 10 m vertical).
  Everything on this page is convection-permitting to sub-kilometer
  case-study evidence, not a tornado-resolving claim.
- **FP32 subnormals are flushed on the compile route this product
  uses.** Kernels here are built through CuPy, which appends
  `-ftz=true` after the options the caller passed, and the compiler
  honors the last occurrence -- so FP32 subnormals are flushed to
  zero in arithmetic on those modules even where `--ftz=false` was
  requested. That is toolchain behavior, not a property of the
  silicon: on the measured device (sm_120) the same source compiled
  without the appended flag keeps full IEEE subnormals, and the
  routes in this codebase that bypass the append measure IEEE
  agreement on the same card ([HARDWARE.md](HARDWARE.md) records the
  per-route result). The flushing is real wherever the append
  reaches, so the countermeasures stand; the specific measured
  consequences (branch flips on subnormal inputs) and the
  countermeasures taken are recorded per scheme in the registry.

## 6b. The registered cases, and which of them need data

`woof cases` lists every registered case and, beside each, the command
that runs it. Two doors appear there and they are not
interchangeable: `woof verify NAME` grades a case against its
registered `GATES`, and only cases carrying the `verify` capability have
any; the rest are run as `python -m woof.verify.cases.NAME` and print
their own result. That is why the listing is longer than `woof verify
--help`'s choice list.

Every verify case is **self-contained and needs only the GPU runtime**,
with one exception:

- **`real74_d01`** is a frozen comparison against an external WRF v4.6.1
  run of 3 April 1974, so it needs that run's reference bundle -- its
  `namelists/`, `met_em/`, `static/` and `wrfout_reference/` subtrees.
  The bundle is third-party output of several gigabytes, is not
  redistributable, and there is no `woof fetch` for it. Point
  `WOOF_REAL74_REFERENCE_BUNDLE` at a directory holding it, or place it
  under the case data root as `WRF_1974_MP55_reference_bundle`. Without it the case
  refuses by name and says the same thing.

The two LES tornado cases are the module half of a case whose other half
is a config under the repository's `configs/` directory. As of 2.5.0 that
directory ships in the wheel and the sdist, so an install already has the
file and needs no argument. Through 2.4.1 it shipped in neither: on an
older install, or for a config you keep outside the tree, name the file
with `--config` or point `WOOF_CONFIGS_ROOT` at the directory holding
it. `--help` works either way.

## 7. Reproduce this

The headline comparison (section 3) was produced as follows.

- **Config:** `configs/real74_thompson_1218z_rrtmg_legacy_4dom.toml`
  (ships in this repository) -- Thompson mp8 with the packaged,
  SHA-256-validated WRF tables; `ra_rrtmg_variant = "rrtmg_legacy"`;
  four domains, 12Z-18Z (21,600 s); history cadence 60/60/60/30 min.
- **Commit:** the run of record executed at internal development
  commit `152f7d31` (2026-07-28, recorded here for provenance; that
  history predates this public repository). The configuration and
  comparator ship unchanged in this release, so the comparison runs
  from any checkout of it.
- **Inputs you must stage yourself** (none are redistributable by this
  repository): the ERA5 case retrieval for 3 April 1974 (see
  [DATA.md](DATA.md) for the CDS walkthrough; the config's
  `[case_data]` table names the exact files), the NCAR WPS_GEOG static
  tree, and -- for the CPU side of the comparison -- a WRF v4.6.1 run
  of the same case built from the pinned commit (the reference run's
  namelist is recorded with the run metadata; its option set is the one
  named above).
- **Commands:**

```bash
woof check configs/real74_thompson_1218z_rrtmg_legacy_4dom.toml --alloc
woof run   configs/real74_thompson_1218z_rrtmg_legacy_4dom.toml --outdir out/rematch --directory-input-hash content

python tools/matched_wrfout_stream_compare.py \
  --gpu-dir out/rematch --cpu-dir /path/to/wrf-reference-wrfouts \
  --out-csv out/rematch/metrics.csv --exclude-rows 5 \
  --start-time 1974-04-03_12:00:00 --done-file out/rematch/run.done

python tools/matched_wrfout_t0_state_digest.py \
  --candidate-dir out/rematch --reference-dir /path/to/wrf-reference-wrfouts \
  --out-json out/rematch/t0_state_parity_digest.json \
  --out-md out/rematch/t0_state_parity_digest.md
```

Three flags here are not decoration:

- `--directory-input-hash content` hashes every byte of the static geography
  tree instead of its listing. Two people staging that tree separately get the
  same digest only under `content`, and certification refuses a run whose
  geography was bound by listing.
- `--start-time` is what fills the `forecast_hour` column. Without it every
  lead cell is written empty, and a comparison with no lead cannot be placed
  against an acceptance band, which is keyed by lead.
- `--done-file` is the comparator's terminating condition: it exits once that
  file exists and no pair is still pending. Without it the poll loop has no way
  to finish, so the command runs until it is killed. Touch that file when the
  run completes.

These commands were executed through their production entry points against a
two-frame fixture pair before this paragraph was written; the exact argv, exit
code, and resulting lead column are recorded in
`woof/data/certification/recipe-receipt/receipt.json` beside the metrics CSV
it produced. The `woof run` line is the one half that is not executed there --
no fixture stands in for a four-domain six-hour integration -- and the receipt
says so, recording instead that the production parser accepts the line as
written.

- **Expected resources:** on one RTX 5090 the full four-domain window
  took 6.7 h of wall time (67.2 wall-seconds per simulated minute
  whole-tree; 61.4 pre-convective, 72.9 after convective initiation)
  and 31.5 GB of disk (20.1 GB wrfouts + 11.4 GB checkpoints). Measured
  machine-wide VRAM peak was 29,004 MiB (28.3 GiB) on a 32 GiB card --
  this case is sized for a 32 GiB card and will not fit smaller ones;
  size your own case with `woof domain` ([HARDWARE.md](HARDWARE.md)).
- **Expected result:** compare the new per-frame metrics with section 3
  and retain all differences. This page supplies no calibrated tolerance
  for a new run or a later build; it does not establish a chaos allowance.
  Same-build determinism is a separate, explicitly tested property.
  The digest command scores the two t=0 states array by array. Preserve
  the verdict it measures. The historical result was **FAIL**, not a
  requirement that later initial states must also fail; compare the
  per-array numbers with the committed
  [receipt](../../woof/data/certification/t0_state_parity_digest.json).
  Missing coverage remains unmeasured rather than equivalent.
