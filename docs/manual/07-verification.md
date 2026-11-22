# 7. Verification and validation instruments

This chapter separates code verification against reference computations,
numerical-resolution diagnostics, sensitivity studies and observation
tooling. The matched WRF comparison in 7.1 and known-answer checks of the
scorer in 7.2 are code verification. The first spectral campaign is a
sensitivity study of two runs with different starts. Section 7.3 measures
effective resolution; it informs solution verification but supplies no
run-specific numerical-error bound. Section 7.4 describes observation
tooling, not a completed multi-case validation result. Section 7.5 is an
demo with limited case scores, not broad validation; 7.6 is a
reproducibility screen for hardware faults.
Measurements are made from retained output and scored with tools checked
against known answers; neither a log nor a successful run proves accuracy.

## 7.1 The matched-run protocol against WRF

The historical WRF-comparison case (code verification, not an observation
comparison) is ERA5-initialized, four one-way nested domains at
12/3/1/0.5 km, integrated over a 6 h window with Thompson microphysics (WRF's own
hash-pinned tables), YSU, MM5 surface layer, Noah, Kain-Fritsch on the root, and
the legacy-RRTMG transcription: the same option set as the CPU reference (WRF
v4.6.1, gfortran 15.2.0, dmpar, 48 ranks; the GPU run one RTX 5090)
[docs/public/VERIFICATION.md, historical comparison and limits]. Three measurements and a review process: component ULP oracles, a
t=0 full-state digest, a matched-run streaming comparator scoring every frame on
the interior grid (5-row rim excluded), and automated review (not an independent scientific measurement)
[docs/public/VERIFICATION.md, historical comparison and limits].

The stated limit published beside the wins: the t=0 full-state comparison verdict
is FAIL on all four domains. Only the accumulation group passes as a whole on every domain; some
individual arrays in other groups are also bit-identical at t=0; on d01 the largest disagreements are 66 Pa in perturbation pressure,
0.75 m in terrain height, 296 K in soil temperature (the receipt does not identify the depth or active mask), and ten categories in
the land-use index, so the decay tables contain initial-state differences as well
as forecast divergence [docs/public/VERIFICATION.md, historical comparison and limits].

Selected rows of the historical 2026-07-28 comparison (engine commit
`152f7d31`, legacy RRTMG, one case; interior grid). These are not current
release measurements: later kernel and dynamics changes alter trajectories.
The historical certification band is an uncalibrated documented margin
around this table. Part of the change from the previous run came from
matching its reference configuration more closely; no row scores observations
[docs/public/VERIFICATION.md, historical comparison and limits]:

| dom | lead | T2 MAE K | PSFC MAE Pa | refl corr | refl MAE dBZ | CSI@20 | W corr | wind10 corr |
|---|---|---|---|---|---|---|---|---|
| d01 12 km | F6 | 0.041 | 2.21 | 0.939 | 2.86 | 0.785 | 0.909 | 1.000 |
| d02 3 km | F3 | 0.051 | 3.15 | 0.985 | 1.33 | 0.857 | 0.749 | 0.997 |
| d02 3 km | F6 | 0.165 | 11.55 | 0.929 | 4.02 | 0.682 | 0.375 | 0.988 |
| d03 1 km | F6 | 0.347 | 20.45 | 0.715 | 9.75 | 0.425 | 0.138 | 0.901 |
| d04 0.5 km | F6 | 0.434 | 22.70 | 0.577 | 14.20 | 0.222 | 0.110 | 0.795 |

At +3 h on the 3 km domain, the highlighted lead has reflectivity
correlation 0.985 against WRF, with 14,230 pixels at or above 20 dBZ
against 14,227. The previous 2026-07-27 run differed by 295 pixels. The
mean model-minus-WRF reflectivity difference changed from -0.311 to
-0.004 dBZ. Matching a coverage count does not establish the same storm
locations or structure; this is one selected lead, not bias against
observations. Earlier correlations are higher and later/finer-grid
agreement falls [docs/public/VERIFICATION.md, Matched-run results].

Small perturbations can grow in convective flow, but that does not
explain away every model difference. On d03 W correlation falls from
0.986 to 0.333 when convection initiates, stays near 0.33-0.37 for two
hours, then ends at 0.138. The initial states differ and this historical
case has no matched WRF-versus-WRF control ensemble. Mixed signs in peak
reflectivity differences do not establish the absence of intensity bias;
the tables do not exclude ingest, geometry, coupling, physics or dynamics
errors. The original sign count has no cited retained receipt and is not
used as evidence here.

The [recent ensemble consistency tests](../public/receipts/wrf-consistency-20261002/SUMMARY.md)
compare tested engine snapshots with perturbed WRF v4.7.1 references.
They are the appropriate kind of fidelity instrument, with a coastal
failure and stated power/coverage limits. They do not retroactively
provide a control for this four-domain historical case. Worldwide
projections have their stated projection-oracle and finite-state smoke
evidence, not a matched forecast [docs/public/VERIFICATION.md, Worldwide projections].

Determinism evidence from the same run: frames produced before two external
process kills were byte-compared when regenerated after relaunch,
SHA256-identical [docs/public/VERIFICATION.md, historical comparison and limits].

## 7.2 Spectral comparison v2: the instrument

The tool compares two gridded fields scale by scale. With a reference-model
run it supports code verification; with observations it supports validation.
The two-initialization campaign below is a sensitivity study.

`woof spectral` is an additive model-to-reference comparison class beside the
frozen v1 chaos-envelope metric, whose pins did not move (v1 pin hash
`f3d1d17f...`, still asserted in the suite) [docs/public/SPECTRAL_VERIFICATION.md;
tests/test_spectral_compare_v2.py]. Six subcommands: `register` (pin bands,
fields, and gates before opening any output), `score` (emit a self-hashed
receipt), `run` (register then score, the registration durably written before the
first output byte is opened), `check`, `plot` (generated only from the validated
receipt, never recomputing a metric), `pins`. The arithmetic is pinned prose per
key: mode power `|F|^2/N^2` (Parseval-normalized), the complete isotropic disk
through the smaller-axis Nyquist, a separable Hann window normalized to unit mean
square, least-squares plane detrend, float64/complex128 throughout, half-open
physical wavelength bands that may leave gaps but may not overlap. Per band, per
field a receipt reports candidate and reference power, power and amplitude
ratios, signed spectral correlation, coherence squared, weighted mean absolute
per-mode phase error, normalized error power, and an exact decomposition of error
power into amplitude mismatch and phase/location terms; horizontal wind repeats
all of it for total, rotational, and divergent kinetic energy
[docs/public/SPECTRAL_VERIFICATION.md].

Gates are fail-closed with no built-in empirical thresholds: no gates means an
`informational` receipt, not a pass claim; any violated gate fails; any missing or
low-signal target is `incomplete`; a band the domain cannot resolve is
`unresolved` and a gate targeting it becomes INCOMPLETE, never PASS. Thresholds
come only from a predeclared known-good population via the calibration tool, and
the candidate being judged may never be in the calibration population
[docs/public/SPECTRAL_VERIFICATION.md].

**Instrument checks against known answers, both directions, before any real use**
(code verification of the scoring tool)
[gallery:spectral-v2-20260818/instrument-validation/]:

| test | measured | truth |
|---|---|---|
| phase recovery, 9 shifts 1-175 deg | worst miss 0.0004 deg | exact |
| error attribution, moved-only field | 100.0000000000% position | 100% |
| error attribution, rescaled-only field | 0.0000000000% position | 0% |
| field against itself | agreement 1.000, position 0.00 deg | 1 / 0 |
| same field shifted 30 deg | 0.866 / 30.00 deg | 0.866025 / 30 |
| two unrelated noise fields | 0.054 / 84.6 deg | 0 / 90 |
| whole known-answer run | 73 of 73 predictions met | |

Resolution limit published with the instrument: better than 0.001 deg of phase
for a properly resolved wave in a well-populated band, degrading to about
0.01 deg at a band boundary or near a 180 deg shift. Reading rule: 90 deg of
position error means no information; 85-95 deg in a real comparison means the two
runs are unrelated at that feature size.

**Two findings from the known-answer checks, stated because under-claiming is the
point** [gallery:spectral-v2-20260818/captions.md]:

1. The shipped gate-calibration demo is degenerate: its known-good population is
   copies of one field times a scalar, so three of five metrics are
   mathematically constant and the derived threshold lands on a machine-precision
   boundary (4 ULP below 1.0 on the author's box, 0 below on another, so the same
   member passes there and fails here by 2 ULP). The gate arithmetic is fine; the
   demo must not be used as a template, and the calibration tool does not yet
   warn on zero-variance metrics.
2. `coherence_squared` and `spectral_correlation` are the same measurement for
   real-valued fields (checked across all 15 band rows, agreeing to the last
   bit); a calibrated gate set double-counts it. Open pin-owner item.

**First real campaign** (preregistered, verdict deliberately `informational`
because no known-good population exists for the pairing): two WOOF-ArWen runs,
identical 300x300 3 km grid, identical physics, identical 6 h window, differing
only in initialization (RRFS 3 km analysis vs GEFS ~25 km global), scored at 0,
3, 6 h [gallery:spectral-v2-20260818/campaign/]:

- storm-scale (18-50 km) wind-energy ratio, sharp vs smooth start: 780x at h0,
  24x at h3, 3.4x at h6;
- agreement above 200 km: 0.85-0.91, nearly flat with lead;
- agreement 50-200 km: 0.19-0.30;
- agreement below 50 km: 0.02-0.08 at every lead, position error 80-95 deg,
  which the instrument's own reading rule renders as no information;
- rotational/divergent partition: the sharp start is more divergent at h0; by
  h3/h6 it flips, the smooth start up to 30 percentage points more divergent at
  storm scale;
- fail-closed on real data: at h0 vertical motion is exactly zero in both runs,
  and the tool reported those bands unresolved rather than dividing by zero.

The claimable statement, printed as the limit it is: WOOF-ArWen rebuilds most of the
small-scale energy a sharp start began with inside about 6 hours, but not the
same small-scale weather; below about 200 km, changing the initialization gives
a different forecast, not a shifted one.

Instrument limits the doc states and this manual keeps: a taper changes a pure
Helmholtz mode (Hann convolves neighboring modes); this version does not regrid
(regridding inside the scorer would hide a second numerical operator under the
metric); model levels are not always common physical levels; a regional FFT is
not a global spherical-harmonic transform; power agreement is not forecast skill
by itself [docs/public/SPECTRAL_VERIFICATION.md:301-336].

## 7.3 Effective resolution

Method, preregistered before any spectrum was computed: kinetic-energy spectrum
of horizontal wind at ~510 hPa; -5/3 fitted over the resolved mesoscale band;
effective resolution read as the wavelength where the spectrum falls below the
fitted extension by a factor of 2; every spectrum through the spectral-v2
instrument with register-then-score receipts (the preregistration predates the
first receipt by 72 s) [receipt:ARWEN-EFFECTIVE-RESOLUTION-2026-08-18.md, with a
verifier-corrections block appended after an adversarial recompute; the
corrections govern].

**The one claimable number: 6.7 dx (3.35 km) on a 500 m grid.** Total spread
across every band/threshold perturbation 1.23 dx; +/-1 sigma 6.09 to 7.48 dx; an
own-slope diagnostic on the same spectrum reads 5.6 dx (2.79 km); a secondary
level (~690 hPa) reproduced it within 30%. Conditions: the 500 m nest (424x336,
49 levels) of the radiation-cadence A/B chain, control arm, RTX 5090, GFS
2026-08-17 12Z init, forecast hour 2. The configuration is Morrison mp10, YSU
PBL, MM5 surface layer, Noah, RTE+RRTMGP, `km_opt = 4`, cumulus off; the config's
own generated header flags it as a gray-zone configuration (sub-km dynamics with
a 1-D PBL scheme active) [receipt:ARWEN-EFFECTIVE-RESOLUTION-2026-08-18.md;
gallery:radt-subkm-fix-20260817/config-control.toml]. Against the widely cited
~7 dx effective resolution of WRF-class models (Skamarock 2004, Mon. Wea. Rev.,
a literature figure, not an in-tree measurement), the reading is consistent
with it within its one-sigma spread, 6.09-7.48 dx. It does not show a
sharper model: the literature figure uses a different criterion and
configuration. The tail has no energy pileup but is strongly over-damped. One caveat the receipts do not resolve: the reading
is at forecast hour 2, and the report's own guidance is that small scales need
about 6 h of spin-up; the verifier accepted the 500 m number on its clean fit
band, but no document explains why the spin-up objection that voided the 2 km
reading cannot apply at 500 m. Treat the number as the study reports it, with
that flag attached.

**What is not claimable, with the same prominence:**

- The 3 km readings (16.6 and 21.0 dx) are not measurements and must not be
  quoted as WOOF-ArWen's effective resolution at 3 km: no inertial range in the fit
  band (free slopes -2.5 to -2.9), fit scatter exceeding the factor-2 criterion
  itself, no knee in the local slope profile, sensitivity spread 15-22x the
  stability bar, and the f02-f06 trajectory (10.2 to 16.6 dx, monotone, with
  domain-mean KE still rising 19%/h) shows the case still spinning up at f06. A
  fixed -5/3 reference is the wrong yardstick on a 900 km domain at these leads.
- The 2 km reading (26.6 dx) is a 2 h spin-up artifact by the report's own text.
- The 6 km parent produced no number at all: the preregistered fit band spans
  0.30 decades on a 196x156 domain against a preregistered 0.4-decade floor, so
  no number was forced. That refusal is a methodology result.

**The limit that must sit beside the win:** the spectral tail is strongly
over-damped everywhere measured (local slopes -8 to -12 near the grid scale),
verifier-confirmed, with zero pileup. Sharpness is not being bought with
under-dissipation, and the over-damping means the "make it sharper" direction
has real headroom; it also means every grid's small-scale energy is low against
a -5/3 extension. Never generalize the 500 m reading to other grid spacings:
effective resolution has been measured on exactly one grid. At the 250 m LES
target, 6-7 dx scales to roughly 1.5-1.8 km effective resolution; that is a
projection from one measurement, labeled as such in the report body and neither
endorsed nor disputed by the verifier; mesocyclone-scale rotation sits inside
that resolved range, and tornado-core structure (a few hundred metres) remains
a downstream target that needs the grid
[receipt:ARWEN-EFFECTIVE-RESOLUTION-2026-08-18.md]. What a proper 3 km
measurement needs is queued as post-cut science: a larger domain, 12-24 h of
spin-up, and the own-slope knee criterion formalized in place of fixed -5/3.

Figures safe to reuse from this study:
[gallery:effective-resolution-20260818/money-chain500m.png] (the one surviving
spectrum) and
[gallery:effective-resolution-20260818/verify-effective-resolution-adversarial.png]
(the disconfirming chart, published deliberately). The gallery's `captions.md`
and the 3 km summary charts predate the corrections and present retracted
numbers; they are not to be used.

## 7.4 Observation tooling for validation

The broad observation battery is designed and preregistered, with a
seven-case menu and hash-pinned MRMS, ASOS and Stage IV archives. Its
published receipts state that no forecast has been scored and no model
skill claim is made. The commands below are observation-ingest tools;
their availability is not a validation result. The intended reflectivity
reference is MRMS and the surface reference is ASOS-class stations.
Specific case scores reported elsewhere are listed in
[VERIFICATION.md](../public/VERIFICATION.md); they do not complete this
programme or establish general forecast skill. The ingest doors are
`woof obs asos | goes | mrms | odim | opera | stage4 | radar`; command
details remain in [CLI-OPTIONS.md](../public/CLI-OPTIONS.md).

## 7.5 DA and nowcasting: present, demo-grade, self-labelled

The DA/nowcast pipeline (`python -m tools.da_nowcast`, a checkout tool, not a
pip-installed subcommand) runs eight receipted stages: survey a radar site, size
a domain around the echo, fetch the GFS background, prepare, run a georeference
forecast, build per-cycle observation files, run six 15-minute LETKF cycles with
ten members, draw the gallery, and hand the case to a detached verifier
[docs/da-nowcast-quickstart.md]. Its gallery identifies the demo as outside
a registered validation campaign. A frame becomes `SCORED` when an observed
radar composite is available and compared with it; that badge means a
comparison was made, not that the forecast was right. An observation-unavailable
frame remains unscored. The retained demo uses ten members, one 3 km domain,
no radiation and a GFS background. `docs/da-vs-wofs.md` reports a limited
single-case radar comparison and states how the
configuration differs from the Warn-on-Forecast System and why its FSS does not
sit next to a published WoFS number.

The launcher's planner prints, before anything launches, the grid the box fits
(from the wizard, including the memory preflight), which radars cover it, an
estimated cost per cycle with its basis (3.15 s per trajectory per 900 s leg on a
132x132x49 grid at dt 15 s, RTX 5090, from a real cycled run), and the estimate's
two known ways of being wrong, both optimistic: short legs cost more than
predicted, and large ensembles cost more because integration is linear in N
(measured 3.06 s per member-leg at N = 10, 20, 36) while the LETKF solve is not
(6.3, 10.1, 71.9 s at those sizes). Measured on a rented RTX 4080 (16 GB, sm_89),
full six-cycle demo at 136x134x49: N=4 431 s, N=10 727 s, N=20 1163 s, with
whole-card peak near 15.9 GB at all three sizes because members advance one after
another [docs/da-nowcast-quickstart.md; tests/test_da_nowcast*.py].

The single-case radar scores do not establish a broadly validated forecast
system, and a later release does not inherit those measurements without
matching their configuration and evidence scope.

## 7.6 The dual-run screen

The dual-run byte comparison (section 4.4) is part of the verification kit: on
cards without ECC it is the corruption screen, run as two sequential integrations
from one prepared root with byte comparison of state digests and outputs. Its
scope and non-claims are stated in section 4.4 and are not repeated here.
