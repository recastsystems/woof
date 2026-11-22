# Longwave radiation extension spacing

WOOF-ArWen extends the model column above its top before calling legacy RRTMG
longwave radiation. WRF 4.7.1 uses 4 hPa steps and overwrites the final
interface with zero. The layer count remains WRF's rounded count, including
its positive half-way rounding.

The former positivity guard divided the actual top pressure by the number
of added layers even when the retained interfaces were already positive.
For a 50 hPa top and 13 added layers, it selected 50/13 hPa instead of
4 hPa. This was an implementation defect in the guard, not a deliberate
radiation formulation. It affected scalar, batched and device preparation.

The corrected default preserves 4 hPa whenever the top exceeds
4 times the number of retained interior interfaces. Only the first
`n_extra - 1` decrements are retained: the final interface is overwritten.
At equality or below, the existing conservative reduced-step fallback
remains, so variable-top columns keep positive interior pressures. The
shortwave preparation and the longwave layer-count rule are unchanged.

An admitted compiled-WRF capture supplies independent pressure,
temperature and ozone expectations in
`tests/data/wrf471_lw_extension/`. All seven complete observer histories
match the accepted executable. The native interfaces at this model top
are 50, 46, 42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 2, 0 hPa. Tests check
those words, the full five captured profiles, partial GPU batch widths,
threshold neighbors and the retained shallow-column fallback.

The coastal forecast retest uses the original Light-ECT reference,
registration and thresholds. Matching this preparation boundary alone
is not a claim that the full forecast passes that test.

The admitted RTX 5090 replay first reproduces all 29 recorded WOOF-ArWen
preparation arrays and its surface longwave flux. On native input words,
the corrected guard then reproduces all 25 captured native preparation
arrays, all eight solver output arrays and all 245 physical-layer heating
values at each of four sampled calls across five columns. The former
heating error peaks between 1.38e-7 and 1.48e-7 K/s; the correction removes
it. The surface longwave change is at most about 0.0026 W/m2.

Three six-hour corrected coastal arms, with default arithmetic, still
fail the original sealed test: six of eleven recurrent components and
critical statistic 3.356648, versus nine and 5.348833 before the fix.
Descriptive native-envelope coverage improves from 63 to 96 of 216
field-hours. The defect contributes to the rejection but does not explain
the complete remaining mismatch. The test is a model comparison, not an
observation-based forecast-skill assessment.

The committed-default GPU regression completes 36 focused cases without
skips or failures. It includes independent native profiles, exact guard
neighbors, shallow and single-extra-layer paths, partial batches and
ordinary width-5000 LW/SW dual runs. The changed unit's four kernels still
use zero bytes of per-thread local memory on the RTX 5090, sm_120,
NVRTC 13.4.92. A broader affected suite timed out and is not represented
as passed.

The pressure, temperature and ozone CPU expectations use only basic
float32 arithmetic on captured input words. They do not depend on platform
exp, log, pow or trigonometric functions; a regression refuses those host
calls during native-profile preparation. The touched CPU tests pass on
Ubuntu 24.04 with glibc 2.39 and on glibc 2.43. The RTX 4090 device path
also matches 1,604,940 CPU profile words exactly, including width-5000
batches, with the same four zero-byte local-frame readings.
