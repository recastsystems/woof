# WOOF's own realisation spread, measured, and what it does to the headlines

Measured 2026-08-02 on the local RTX 5090. Until now every WOOF-vs-WRF number
compared **one** WOOF run against WRF runs whose spread had been measured. The
error budget carried WRF's noise and assumed WOOF's. This measures WOOF's.

`gpuwm.verify.cases.convective_boundary_layer`, 96x96x64, dx = 100 m, dt = 0.5 s,
2 h, `c_s` 0.18, `ztop` 2400 m: the exact configuration of the committed
`cbl-2026-08-02` receipts.

## 1. Determinism: bit-identical

Same `--seed`, same card, run twice. Every numeric field of the receipt
compared, excluding wall-clock timings:

| closure | fields compared | differing |
|---|---|---|
| `km_opt=3` @ 100 m | 18 | **0** |
| `km_opt=2` @ 100 m | 19 | **0** |

This is the RTX 5090 datapoint 1.4.1 lists as a known limit, and a clean pass of
the dual-run screen the release notes tell users to run. Repeatability on this
card is not in question; what follows is physics, not hardware.

*Superseded upward 2026-08-03: `ARWEN-CROSS-CARD-DETERMINISM.md` extends this
single-card datapoint to three cards and two architectures (bit-identical
within sm_120 across different silicon, seed-equivalent against sm_89) and
retires the known-limit wording.*

## 2. Realisation spread: ~9x WRF's on the flux arm, parity on the TKE arm

Independent seeds 1–18 (extended from n=6 by the second local sweep,
`EXT-SWEEP-LOG.jsonl`, 37 runs, 0 failures, configs identical to the first
six):

| quantity | WOOF stdev (n=18) | WRF stdev | ratio |
|---|---|---|---|
| `wth_res_max_over_qs`, `km_opt=3` | **0.01414** | 0.00163 (n=4) | **8.7x** |
| `resolved_fraction_ml`, `km_opt=2` | **0.00231** | **0.00247** (n=6) | **0.93x** |

*History of this table, kept because the receipt's own spread estimates carry
n-noise too. At n=6 the `km_opt=3` cell read 0.00855 (5.2x): the first six
seeds were a narrow draw (seeds 7–18 alone give 0.01487) so the extension
made the flux-arm excess larger, not smaller. The `km_opt=2` cell read "not
yet measured", then 0.00300 vs WRF's 0.00247 (1.2x) at n=6; at n=18 WOOF is
marginally the tighter of the two. WRF cells from
`wrf_oracle_spread_km2_100m_r1..r6.json` (truncation convention; n=7 with the
original draw 0.00295, averaging 0.00261) and the n=4 `km_opt=3` set. The
same runs put the flux-fraction spread under `km_opt=2` at WOOF 0.01499
(n=18) vs WRF 0.00280 (5.4x) so the excess is a property of the
flux-fraction metric under both closures, not of the closure, and the
TKE-based resolved fraction is at parity.*

WOOF `km_opt=3` @ 100 m across 18 seeds: min 0.82342, max 0.86912, range
**0.04571**, mean 0.84264.

**Why WOOF is noisier than WRF here is not established.** Candidates are the
perturbation amplitude, the seeding mechanism, and a genuinely different level
of run-to-run variability in the closure. Recorded as an open question, not
diagnosed.

## 3. What this does to the verdicts, both ways

**It strengthens the corroboration.** Both arms move from "about 1x noise" to
comfortably inside it:

| arm | observed delta difference | % of registered band | x combined noise |
|---|---|---|---|
| `km_opt=3` | 0.00184 | 12.3% | **0.13x** |
| `km_opt=2` | 0.00112 | 7.5% | **0.33x** |

*Twice updated: the `km_opt=2` row read 0.37x with WRF's spread unmeasured
and implicitly zero, then 0.29x with it measured at WOOF n=6; the `km_opt=3`
row read 0.21x at WOOF n=6. At n=18 the combined noises are
sqrt(0.01414² + 0.00163²) = 0.01423 and sqrt(0.00231² + 0.00247²) = 0.00338.
The `km_opt=2` ratio rose slightly because WOOF's spread shrank with n; the
`km_opt=3` ratio fell because it grew. The convention in this table, both
rows, is the two models' 100 m single-run spreads in quadrature; §6 gives the
all-endpoints version.*

*One endpoint-provenance correction, found while reconciling the node state:
the +0.00112 in `KM2-50M-REFINEMENT-VERDICT.md` conditions on a WOOF 100 m
same-instrument endpoint of 0.889297, the pre-`eab04061` reduction against
the superseded WOOF receipts. Test host 2 still holds that stale file, and the
hand collection quoted it. Against WOOF's current 100 m receipt (0.891629,
the value committed in `wrf_oracle_same_instrument_km2_100m.json`) the delta
difference is **−0.00122**: 8.1% of the band, 0.30x measured noise. The
verdict is unaffected (corroborates either way) and the sign flipping when
one WOOF draw is swapped for another is the cleanest demonstration this
receipt could ask for that the sign of a 0.3x-noise quantity is not
information.*

**It weakens the precision we quoted.** Two claims need retiring:

- The published single-run WOOF value at `km_opt=3` / 100 m is **0.84386**.
  At n=6 this receipt called it "roughly the 95th percentile of its own
  realisation spread"; at n=18 it sits at the **67th percentile** (12 of 18
  draws below it): a typical draw after all. The percentile claim was itself
  a noisy n=6 statistic and is corrected here rather than defended. What
  stands is the structural point: every comparison used one draw from a
  distribution whose measured width is 0.014.
- The 50 m headline was reported as **0.07% agreement** on flux fraction. That
  is 0.00062 absolute, against a one-run realisation spread now measured at
  0.01719 (n=9 at 50 m): **~28x finer than the noise of a single run**. It is
  a fortunate sample, not a demonstrated precision, and must not be repeated
  as though it were the latter.

The honest form of the result is unchanged and is the one to use: *WOOF's
refinement of the resolved/subgrid partition matches WRF's, and the difference
between them is smaller than either model's run-to-run scatter.* Any statement
tighter than that is over-reading one sample.

## 4. What this changes going forward

Single-run WOOF receipts should carry their realisation uncertainty, or be
replaced by an ensemble mean over seeds, wherever they are compared against
another model. The runs are cheap (87 s at 100 m on an RTX 5090) so there is
no reason to keep quoting one draw.

## 5. The 50 m spreads, measured on both sides

Added with collection 2c9d4b58. Everything above was measured at 100 m; the
50 m refinement endpoints carried assumptions. Both sides now have measured
50 m spread.

**WOOF, n=9 per closure**: seeds 1–9 (extended from n=3 by the second
sweep), config identical to the committed 50 m receipts (192x192x96,
dx = 50 m, dt = 0.25 s), all rc=0. The seed-0 determinism pair at 50 m is
bit-identical (0 numeric receipt fields differ, timings excluded), so as at
100 m this is physics, not hardware.

| quantity | seeds 1–9, sorted | stdev (n=9) | range |
|---|---|---|---|
| `wth_res_max_over_qs`, `km_opt=3` @ 50 m | 0.86199, 0.86785, 0.87971, 0.88432, 0.88597, 0.89834, 0.90318, 0.90385, 0.91270 | **0.01719** | 0.05072 |
| `resolved_fraction_ml`, `km_opt=2` @ 50 m | 0.92930, 0.92963, 0.92996, 0.93056, 0.93069, 0.93167, 0.93203, 0.93262, 0.93285 | **0.00131** | 0.00355 |

At n=3 these cells read 0.01798 and 0.00115: the wide flux-arm spread **held**
under tripling of the sample (a 9-sample stdev still carries ~25% relative
standard error, but a factor-of-two n=3 fluke is now excluded).

**WRF, n=3 per closure** (r3 runs collected at 390fa782; each run's
`*_binaries.sha256` matches the lane's identity hashes, so all draws are the
same bytes). WRF's ideal-case perturbation is unseeded and
decomposition-dependent, so the draws are genuinely independent. Flux-arm
ledger `match_km3_50m` (test host 2) / `_r2` (test host 1) / `_r3` (test host 1):

| quantity | r1 | r2 | r3 | stdev (n=3) |
|---|---|---|---|---|
| `wth_res_max_over_qs` | 0.882523 | 0.883176 | 0.883313 | **0.000422** (range 0.000791) |
| zi_m | 1662.086 | 1662.079 | 1662.076 | 0.0050 |
| theta_ml | 301.3247 | 301.3233 | 301.3217 | 0.0015 |
| tke_res_max (trunc) | 2.16543 | 2.17577 | 2.18478 | 0.0097 |
| entrainment_min_over_qs | −0.13228 | −0.13441 | −0.13291 | 0.0011 |

The `km_opt=2` ledger (`match_km2_50m_node1` / `match_km2_50m` /
`match_km2_50m_r3`, test host 1, test host 2, test host 2): resolved fraction (truncation)
0.927470 / 0.927673 / 0.929116, stdev **0.000898**, range 0.001646. The
third draw corrects an n=2 artifact: `CONTROLS-MEASURED.md` reported the
pair gap of 0.000203 and called the 50 m endpoint "five times tighter than
assumed"; at n=3 the spread is 0.000898, essentially the 0.0010 the
registered band assumed. The first two draws were simply close. A 3-sample
stdev still carries ~50% relative standard error, on every n=3 number in
this section.

Three things this settles:

1. **The noise asymmetry is extreme on the flux arm at 50 m.** WOOF's 50 m
   flux-fraction spread (0.01719, n=9) is ~40x WRF's (0.000422, n=3) and
   roughly flat against its own 100 m spread (0.01414, n=18). Refinement
   halves WOOF's noise on the `km_opt=2` quantities (resolved fraction
   0.00231 → 0.00131, flux fraction 0.01499 → 0.00669) and cuts WRF's by
   3–4x (0.00247 → 0.000898; 0.00163 → 0.000422 on the flux arm), but
   leaves WOOF's `km_opt=3` flux fraction wide. An earlier revision of
   this item said that spread "doubles under refinement": that was the
   narrow n=6 100 m baseline, not real growth. The §2 open question is
   sharpened, not diagnosed.

2. **The "0.07% agreement" stays retired, with prejudice.** WRF's three
   50 m draws cluster within 0.0008 of one another, and WOOF's single
   published draw happens to sit inside that cluster: gaps of 0.00059,
   0.00006 and 0.00020 to the three. The cluster is tight because WRF is
   tight; WOOF's own measured 50 m spread is 0.017, so the typical gap for
   a fresh WOOF draw would be ~0.014. One lottery ticket, not three.

3. **The published km3fine draw is a high-side draw, inside the measured
   range.** 0.904 has 8 of 9 seeds below it, but the max draw (0.91270) is
   well above it: high, no longer the "top of the spread" the n=3 sample
   suggested, consistent with §3's corrected reading of the 100 m published
   value.

## 6. The 50 m refinement corroboration, restated with measured noise

Registered band (31205814, committed before the data existed): a WRF
100 m → 50 m delta within ±0.015 of WOOF's corroborates. Measured flux-arm
delta difference: **+0.00184**, 12.3% of the band. That verdict stands
exactly as registered.

What the measured noise adds, in both directions:

- With all four endpoints measured, the single-draw noise on the flux-arm
  delta difference is sqrt(0.01414² + 0.01719² + 0.00163² + 0.00042²) ≈
  **0.0223** at WOOF n=18/n=9 and WRF n=4/n=3: every term now an honest
  stdev (this bullet read 0.0200 at WOOF n=6/n=3 with a pair-gap stand-in
  for WRF's 50 m term). The observed +0.00184 is **0.08x** that: deeper
  inside the noise than the 0.13x §3 quotes from 100 m spreads alone.
- The same number cuts against the test's power: **0.0223 is 1.5x the ±0.015
  band itself.** The band was built generously from ~0.0010 endpoint noise;
  the measured one-draw noise floor exceeds it, so two identical models
  would land inside the band only about half the time, and a single draw
  just outside it would not have been evidence of a real difference. The
  registered rule's "inconclusive just outside the band" clause was doing
  real work. What the flux arm shows is consistency, not a sharp test.
- Across the three WRF 50 m endpoint draws the flux-arm delta difference is
  **+0.00184 / −0.00249 / −0.00262** (12.3%, 16.6%, 17.5% of the band),
  all three corroborate, and the sign flips between draws; swapping the
  WOOF endpoint draw would move it by ±0.017. The sign is noise, as in §3.
- The TKE arm is where the discriminating power lives. All four endpoints
  measured: sqrt(0.00231² + 0.00131² + 0.00247² + 0.00090²) ≈ **0.00373**,
  a noise floor 4.0x smaller than the band. Observed 0.00112 = **0.30x**
  (−0.00122 / 0.33x against the current-receipts endpoint, per the §3
  correction). Identical models would pass that band >99.9% of the time
  and a real difference of band size would be caught; this arm's
  corroboration means something on its own.

The honest composite statement, replacing all precision claims: *both
refinement arms land at or below a third of their measured single-draw
noise, inside a band registered before the data existed; the TKE arm's band
is a sharp test and the flux arm's is not, because WOOF's own 50 m
flux-fraction scatter (0.017, n=9) exceeds the band.* Nothing tighter
survives the measured spreads.
