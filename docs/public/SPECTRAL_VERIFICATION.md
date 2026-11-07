# Scale-resolved spectral comparison

WOOF has two deliberately separate spectral metric classes.
Comparisons with WRF support code verification; comparisons with observations
support validation for the scored quantities. Each receipt must state whether
its reference is another model or observations. Neither kind of comparison
estimates the numerical error of a run. See
[Verification and validation](VERIFICATION.md#verification-and-validation).

1. `woof.verify.spectral` is the existing frozen `gpuwm.spectral-pins/v1`
   chaos-envelope metric. It compares radially averaged power for the pinned
   `W` and `REFL_10CM` column reductions. Its pin hash and existing receipts
   must remain unchanged.
2. `woof.verify.spectral_compare` is the additive model-to-reference class.
   It compares arbitrary preregistered scalar fields and horizontal wind in
   physical wavelength bands, including amplitude, signed phase/location
   agreement, error power, and rotational/divergent kinetic energy.

The second class was re-pinned on 2026-08-20 under
`gpuwm.spectral-comparison-pins/v2`, because two calibrated guards changed
what the arithmetic does: a wavelength within 8 ulp of a declared band edge is
snapped onto that edge before the inclusive/exclusive rule, and a band with no
resolvable variance is `unresolved` rather than scored. Both are measured, and
both are described below. A receipt written under the previous pin is refused
by name rather than reinterpreted; the frozen v1 chaos-envelope metric and its
hash are untouched.

The second class does **not** replace pointwise RMSE, FSS, object comparisons,
conservation receipts, or the v1 chaos envelope. It answers a different
question:

> At which spatial scales does WOOF carry the same amount and kind of
> structure as the reference field, and at which scales is the disagreement
> amplitude, displacement, or numerical noise?

## Why this is useful

A gridpoint metric can grade a translated but otherwise realistic storm very
harshly. A power-only spectrum can show that the storm has the right scale and
amplitude, but cannot tell whether the structure is in the same place. The v2
comparison therefore carries both power and a complex cross-spectrum.

For every preregistered wavelength band it reports:

- candidate and reference power;
- power ratio and amplitude ratio;
- signed spectral correlation;
- coherence squared;
- weighted mean absolute per-mode phase error;
- total error power normalized by reference power;
- the part of error attributable to amplitude mismatch; and
- the remaining phase/location error.

For a horizontal-wind field it repeats those quantities for:

- total kinetic energy;
- rotational kinetic energy; and
- divergent kinetic energy.

It also records the divergent-energy fraction on both sides. This is useful for
finding high-wavenumber acoustic/divergent noise that can be invisible in a
single wind RMSE.

## The arithmetic contract

The regional comparison is intentionally not a global spherical-harmonic
transform. Current WOOF domains are limited-area projected grids; treating the
unmodeled remainder of Earth as zero would create a boundary discontinuity and
spectral leakage.

The v2 regional contract is:

1. Read both fields on the same horizontal grid.
2. Apply the preregistered equal-width crop to remove the lateral-boundary or
   relaxation frame.
3. Remove a least-squares plane `a + b*x + c*y`.
4. Apply a separable symmetric Hann window normalized to unit mean square.
5. Compute a float64/complex128 two-dimensional FFT.
6. Keep only `k > 0` through the smaller physical axis Nyquist. This circular
   support disk has complete directional coverage; diagonal-only corner modes
   are not scored.
7. Sum mode contributions in half-open physical wavelength bands. A mode's
   wavelength is recovered as `1/|k|`, and that round trip is not exact, so a
   wavelength within 8 ulp of a declared edge is snapped onto the edge before
   the inclusive/exclusive rule is applied. Measured over 2,414,712 retained
   modes on 48 grids, the unsnapped comparison graded 80 modes into the wrong
   band, and every one of them was a `|j|` = 1 domain-scale mode: the most
   energetic the disk retains. Which band it landed in depended on the float
   spelling of `dx`, so `n` = 100 at `dx` = 1000 m and `n` = 200 at
   `dx` = 500 m, describing the same physical wavelength, disagreed. Snapping
   neither widens nor narrows a band: a shared edge snaps identically in both
   bands that touch it, and a declared gap between two bands stays a gap.

For a scalar transform `F`, mode power is

```text
|F|² / N²
```

where `N = ny*nx`. Summing all modes therefore closes to the spatial mean square
by Parseval.

For two transforms `A` and `R`, a band carries

```text
P_A = Σ |A|² / N²
P_R = Σ |R|² / N²
C   = Σ A conj(R) / N²
E   = Σ |A - R|² / N²
```

and

```text
power_ratio          = P_A / P_R
spectral_correlation = Re(C) / sqrt(P_A P_R)
coherence_squared    = |C|² / (P_A P_R)
weighted_phase_error = Σ |A conj(R)| |arg(A conj(R))| / Σ |A conj(R)|
```

The signed correlation distinguishes equal-power phase reversal from genuine
agreement. Coherence remains high for a consistent phase offset. The weighted
absolute phase metric is evaluated per Fourier mode, so the conjugate pair of
a real field cannot cancel its phase to zero. Use these diagnostics together,
not one alone.

The minimum possible error if the two band amplitudes were perfectly aligned
is

```text
amplitude_mismatch_power = (sqrt(P_A) - sqrt(P_R))²
```

and the remaining error is recorded as `phase_location_error_power`.

## Quick start

Start from the example:

```text
configs/verify/spectral_compare_example.toml
```

The safe two-command workflow is:

```bash
# 1. This reads only policy. It does not open either wrfout.
woof spectral register \
  configs/verify/spectral_compare_example.toml \
  --output out/spectral-registration.json

# 2. This validates the registration, hashes each input, scores, and receipts.
woof spectral score \
  out/spectral-registration.json \
  --output out/spectral-receipt.json \
  --plot-dir out/spectral-plots

# Audit receipt bytes and, optionally, rehash all source files.
woof spectral check out/spectral-receipt.json --rehash-inputs

# Compare two boxes' receipts by value under the declared tolerance.
woof spectral cross-box out/box-a-receipt.json out/box-b-receipt.json
```

A convenience command preserves the same publication order:

```bash
woof spectral run configs/verify/spectral_compare_example.toml \
  --registration out/spectral-registration.json \
  --receipt out/spectral-receipt.json \
  --plot-dir out/spectral-plots
```

`run` durably writes the registration before the scoring function opens the
first output byte.

The command is CPU-only. It is intentionally not placed behind the CuPy/GPU
capability gate.

## Source formats

The reader supports:

- WRF-shaped NetCDF files (`.nc`, `.nc4`, `.cdf`);
- `.npz` fixtures with named arrays; and
- one-array `.npy` fixtures.

NetCDF time selection is applied only when the leading dimension explicitly
identifies time.

### The ceiling on non-finite cells is zero

Masked, missing, empty, and non-finite values are refusals, and the ceiling is
**zero cells**, not conservatism. A two-dimensional FFT is a dense sum over
every cell, so one NaN reaches every retained Fourier mode: every band power,
every correlation and every gate on the whole plane becomes NaN, and a receipt
would then grade the entire field on the cells that were bad. There is no
fraction of a plane at which that stops being true, so there is no fraction to
tune.

The refusal says how many cells, which kind, and where:

```text
.../left.npz:W carries 2 of 4096 non-finite cells (nan=1, inf=1), at
(10, 10), (11, 12). A spectral score is a two-dimensional Fourier
transform, so one non-finite cell reaches every retained Fourier mode and
every band power, correlation and gate on the whole plane becomes NaN --
the ceiling on non-finite cells is 0 and cannot be raised. Either score a
variable that is finite everywhere on this grid, or set crop_cells in the
registration so the scored window excludes the affected frame, or fix the
field in the producing run -- this reader will not substitute a fill value
and score it as data.
```

An empty selection and a masked source each get their own message, so a
mistyped level index does not read as a corrupt field.

### Reductions

A scalar or vector component can use:

- `plane`: the source is already two-dimensional;
- `level`: select the preregistered zero-based vertical index;
- `surface`: select vertical index zero;
- `column_max`;
- `column_max_abs`; or
- `column_mean`.

### Staggering

The reader can remove one C-grid stagger before the vertical reduction:

- `x`: average adjacent west-east points;
- `y`: average adjacent south-north points;
- `z`: average adjacent vertical points;
- `auto`: use a single NetCDF dimension ending in `_stag`; or
- `none`.

For a standard WRF horizontal wind, explicitly set U to `x` and V to `y`.
For W, use `z` if the campaign intends mass-level vertical velocity. The
choice is registration policy and must not be changed after seeing output.

## Recommended first WOOF campaign

Use matched WOOF/WRF history frames on the same grid and score these fields
first:

1. `T` at fixed model levels or pressure-interpolated common levels;
2. horizontal U/V at matched levels, with C-grid destaggering;
3. W at a matched mass level or a preregistered column reduction;
4. pressure/geopotential fields; and
5. optionally vorticity/divergence planes derived by one shared authority.

Reflectivity is useful as a morphology diagnostic, but dBZ power is not
physical atmospheric energy. Keep CSI, FSS, object timing, and neighborhood
comparisons as the primary reflectivity measures. State whether each reference
is a model run (code verification) or observed radar (validation).

For multi-domain campaigns, each `[[pairs]]` record carries its own `dx_m` and
`dy_m`. A physical band that a domain cannot resolve is marked `unresolved`.
Any gate targeting it becomes `INCOMPLETE`, never `PASS`.

## Reading the result

Typical interpretations:

| Power ratio | Signed correlation | Meaning |
|---:|---:|---|
| near 1 | near 1 | same amount of structure as the reference, with aligned phase/location |
| near 1 | low | same power as the reference, but displaced/rephased structure |
| below 1 | high | aligned but too weak or over-diffused |
| above 1 | low | excess or misplaced structure; possible noise |
| above 1 in divergent KE only | variable | possible acoustic/split-explicit or boundary noise |

A small reference power can produce an enormous ratio from a physically small
candidate signal. Every row retains absolute powers, and gates may declare a
`minimum_reference_power`. Such a gate becomes `INCOMPLETE` below its floor;
it does not silently skip to green.

## Gates and calibration

There are no built-in empirical thresholds. A source TOML with no `[[gates]]`
records produces an `informational` receipt, not a pass claim.

Use a predeclared known-good population to build a chaos/equivalence envelope,
then derive a candidate gate fragment:

```bash
python tools/spectral_gate_calibrate.py \
  out/member01-receipt.json \
  out/member02-receipt.json \
  out/member03-receipt.json \
  --percentile 95 \
  --output out/calibrated-spectral-gates.toml
```

The calibrator can derive:

- a symmetric log-space envelope around a power or amplitude ratio of one;
- lower-tail signed-correlation and coherence limits;
- an upper weighted mean absolute per-mode phase-error limit;
- an upper nearest-rank normalized-error limit; and
- an upper nearest-rank absolute divergent-fraction difference.

Every source receipt SHA-256 is written into the fragment header. Review and
paste the fragment into the campaign TOML **before** scoring the candidate.
Never include the candidate being judged in the calibration population.

### A gate targets a component and a metric that component carries

A gate names a `component` and a `metric`, and not every component produces
every metric. `scalar`, `total`, `rotational` and `divergent` carry the full
band metric set; the synthetic `partition` row carries only
`left_divergent_energy_fraction`,
`reference_divergent_energy_fraction` and their
`divergent_energy_fraction_difference`.

A gate naming a pair the table does not have is refused at **registration**,
before any model output is opened, and the refusal lists what that component
does carry. It used to be accepted: the partition row republished the total
component's `reference_power` so the `minimum_reference_power` floor could
reach it, so a gate declaring `component = "partition"`,
`metric = "reference_power"` silently graded the total-KE row, and
`spectral_gate_calibrate.py` built two gates out of one measurement. The floor
now travels under a key that is not a metric.

Gate outcomes are fail closed:

- any violated gate: `fail`;
- any missing, unresolved, or low-signal gate target: `incomplete`;
- every evaluated gate inside its preregistered interval: `pass`; and
- no gates: `informational`.

## Receipt identity

A registration binds:

- source TOML SHA-256;
- candidate/reference labels;
- crop;
- physical wavelength bands;
- pair paths, geometry, time indices, domain, and valid time;
- field variables, reductions, levels, and staggering;
- gates;
- v2 arithmetic pins and hash; and
- the existing v1 spectral pin hash.

A registration carries two digests. `registration_sha256` binds the resolved
absolute paths, which is what makes it a durable pin on one box.
`registration_policy_sha256` is the same policy with the pair paths reduced to
basenames (the campaign, without where the bytes happened to sit) and it is
what two boxes may compare.

A score receipt additionally binds:

- full-file SHA-256 and size for every source;
- resolved source dimensions and reductions;
- the evaluator, under `code`: the woof version and git commit, resolved by
  the same builder the certification capsule uses, so a run's capsule and its
  spectral receipt cannot name two different commits;
- the declared reproducibility rule, under `reproducibility`;
- all scalar/vector comparison rows;
- gate rows and verdict, each evaluated row carrying a `row_id` unique per
  matched pair (a `pair = "*"` gate produces one row per pair, and they used
  to share the gate's single `id`); and
- a self-hash over the complete receipt.

`woof spectral check --rehash-inputs` detects a source file changed after the
receipt was produced.

## Plot contract

Plots are generated only from the validated receipt. They never reopen model
output or recompute a metric. Each PNG receives a SHA-256 in
`plots/manifest.json`.

The plotter emits separate charts for:

- candidate/reference band power;
- power ratio;
- signed spectral correlation; and
- normalized error power.

Vector fields receive one set for total, rotational, and divergent energy.

## Limits and cautions

### A taper changes a pure Helmholtz mode, and by how much

A nonperiodic regional field needs a taper. Multiplication by the Hann window
convolves neighboring Fourier modes, so a mathematically pure divergent or
rotational test wave acquires a small component in the other partition. The
synthetic controls require strong separation, not impossible post-window
purity.

The size of that effect is measured, not left as a caution. Write `m` for the
number of times a band's longest retained wavelength fits across the scored
window. Then

```text
leakage ~= 0.34 / m²
```

of that mode's energy lands in the other partition. Measured 2026-08-20 over
38 cases, `n` = 64 to 256, `m` = 2 to 32, crop 0 and 8: the fitted coefficient
rises from 0.314 at `m` = 2 to 0.340 at `m` = 32, and rotational and divergent
leak by the same amount to every digit printed. The estimate is meaningful
only for `m` at least 1; a band whose wavelength is longer than the scored
window has no partition worth reading.

| Wavelengths across the window | Leakage into the other partition |
|---:|---:|
| 2 | 8% |
| 6 | 1% |
| 18 | 0.1% |
| 32 | 0.03% |

Every vector band in a receipt now carries its own
`wavelengths_across_scored_window` and `helmholtz_leakage_estimate`, and the
result carries a `helmholtz_leakage` block naming the model and the scored
window. **A divergent-fraction difference smaller than the leakage at that
band's scale is the window talking, not the model.** Cropping shrinks the
window without shortening the wave, so a large crop makes the longest bands
worse, not better.

### Two boxes agree on the numbers, not on the receipt hash

A receipt self-hash is a **this-box identity**. It says what this machine
measured from these input bytes; it is not portable, and comparing two boxes
by hash reports "different" every time.

Measured 2026-08-20 on sha256-identical input and module bytes, over two
independent pairs (a 192x192 scalar and vector probe, and a 128x128 pair
scored through the real `woof spectral run` door) between the Windows
desktop (Python 3.13.7, numpy 2.2.6, UCRT) and test host 1 (Python 3.14.4,
numpy 2.3.5, glibc 2.43): 169 of 489 metric values were bit-identical and the
rest differed. Repeating a run on either box reproduced it to the bit, so the
spread is the boxes, not the run.

The declared rule is `gpuwm-spectral-cross-box-v1`, quoted into every receipt
under `reproducibility`, and it compares values in four classes:

| Class | Compared against | Measured worst |
|---|---|---:|
| exact (`mode_count`, band edges) | nothing; must be equal | 0 |
| power-dimensioned (`left_power`, `error_power`, the cross-spectrum, …) | the band's own reference power | 7.7304e-16 |
| bounded (`spectral_correlation`, `coherence_squared`, the phase error in degrees, the fractions) | the metric's own declared range | 1.1102e-14 |
| unbounded ratios (`power_ratio`, `normalized_error_power`, …) | the larger of the two magnitudes | 7.7498e-15 |

The declared tolerance is **1e-12**: 90x the measured worst case and 23x the
`sqrt(N_modes)·eps` accumulation bound (4.26e-14) for the larger plane. That
bound reaches the tolerance near a 4500x4500 plane, so a campaign scoring a
larger one remeasures. Any real arithmetic defect (a wrong band, window, or
normalization) moves a metric by order 1e-3, nine decades clear of this.

Power-dimensioned and bounded metrics get a scale of their own because
several of them are analytically zero. The imaginary cross-spectrum of two
real fields cancels across conjugate pairs, and every phase metric collapses
when the candidate is a pure rescaling of the reference: two numbers that
are both zero have no ratio worth quoting.

Compare two boxes with the door, not with `sha256sum`:

```bash
woof spectral cross-box out/windows-receipt.json out/node1-receipt.json
```

It exits 0 when every value is inside the tolerance and 1 when one is not,
naming the pair, field, band, component and metric that moved. Both receipts
must come from **one source TOML**; differing absolute paths are fine, because
the check is on `registration_policy_sha256`, which is the campaign policy
with the pair paths reduced to basenames. The raw `registration_sha256` binds
the resolved paths and never matches across boxes.

### A band with no variance is unresolved, not perfect

A plane held at a constant does not detrend to exactly zero: it detrends to
float cancellation residue. Measured: a constant 273.15 K field scored a
`spectral_correlation` of 0.9999999999999997 against itself, which is a 0.95
correlation gate passing on a field with no structure in it, and whether it
happens depends on whether the constant is a dyadic rational (1.0 and
101325.0 give exact zero; 273.15 does not).

A band whose power is at or below **1e-22** of the plane's own mean square is
therefore reported `unresolved`, and any gate on it becomes `INCOMPLETE`,
never `PASS`. That floor is calibrated between two measured populations:
constant planes (84 cases, `n` = 8 to 512, 12 plausible constants) peaked at
2.784e-31, and real structure at the float32 storage quantization limit (the
finest structure a history file can carry) measured 1.24e-14. The floor sits
8.6 decades above the noise and 8.1 decades below the signal.

### This version does not regrid

Both sides must already be on one common grid. Regridding inside the scorer
would hide a second numerical operator under the comparison metric. For
science-mode statements across different projections, generate a separately
receipted common-grid product first.

### Model levels are not always common physical levels

Comparing `level_index = 20` is valid only when both outputs share the same
vertical coordinate. For different model configurations, interpolate both
sides to preregistered pressure/height surfaces using one shared authority.

### Regional FFT is not global SH

The linked Spherical Harmonic Exponentials work inspired the broader
scale-space direction, but no graphics reflectance code is copied here. A
future global WOOF/MPAS parent can add a true spherical-harmonic backend under
a separate pin schema. It must not reinterpret these regional receipts.

### Power agreement is not forecast skill by itself

A forecast can have an excellent spectrum and put every storm in the wrong
county. Keep pointwise, neighborhood, object, conservation and distributional
comparisons beside this class, and score against observations for any claim
about forecast skill. Model-to-model spectral agreement does not supply that
validation evidence.

## Future extensions

Additions should use new schemas rather than modifying v1/v2 receipt meanings:

- pressure-level preprocessing with one pinned interpolation authority;
- equal-area science-grid preparation;
- lead-time × wavelength error-growth matrices;
- direct connection to the existing chaos-envelope builder;
- optional CuPy FFT batching after CPU/GPU bit/tolerance controls exist;
- global scalar/vector spherical harmonics for a global parent; and
- offline shadow-mode large-scale nudging diagnostics.
