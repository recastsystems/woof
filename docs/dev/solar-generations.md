# WRF solar generations

`solar_generation = "wrf_480"` is the generic default. `wrf_461` preserves
the earlier solar consumer arithmetic for reproduction. The corrected path
uses the shared float32 device sun, WOOF math, explicit single-rounded
operations and the shared no-FTZ loader. No atmospheric radiation scheme,
table, interpolation fit or surface prognostic equation changes here.

WRF source pins:

| Release | Commit |
| --- | --- |
| v4.6.1 | d66e442fccc04111067e29274c9f9eaccc3cef28 |
| v4.7.0 | eeacb821626feb34c62c3c1e169594d886d13adb |
| v4.7.1 | f52c197ed39d12e087d02c50f412d90d418f6186 |
| v4.8.0 | 06d4240ae989cc3e50af412bb472df3d9048783c |

Official source: https://github.com/wrf-model/WRF/blob/v4.8.0/phys/module_radiation_driver.F

The census identifies two executable hunks in `calc_coszen`:

| Hunk ID | Source | Disposition |
| --- | --- | --- |
| 9dfea7131715a9e30ece1da10de0085181a8d716a2bf1aa5f3e2591c4d60a0e2 | 4.7.1 to 4.8.0, module_radiation_driver.F:3527 | Corrected generation uses julian minus 0.5; legacy retains minus 1 |
| 8bb0803cb131da8960f51f326a992fc5179fb38e9ef0e4a29f6e74d9eb1cc170 | 4.7.1 to 4.8.0, module_radiation_driver.F:3529 | Corrected generation uses 0.040849; legacy retains 0.04089 |
| 830b0f130d096a5d24b3774e3ec31956104073e863086071ad32b7ef2cb05aba | module_ra_rrtmg_swf.F:11692 | Comment only; the scheme consumes the driver's solar outputs |

The endpoint 4.6.1 to 4.8.0 comparison repeats these changes. Neither solar
routine adds a Registry state field, a table dependency, a tendency or an
output array. `DECLIN`, `SOLCON`, `COSZEN` and `HRANG` remain the outputs.
The radiation wrapper still shifts XTIME by the radiation midpoint and
keeps JULIAN at call time. Per-step interpolation has no midpoint offset.
The corrected radiation-free Noah-MP carrier advances the call calendar
at midnight; the legacy branch retains its earlier clock behavior.

Consumers are legacy RRTMG, Dudhia, RRTMGP, solar interpolation, terrain
radiation, solar albedo, urban, Noah-MP and the geometric COSZEN carriers
of analytic and longwave-only radiation. Analytic flux retains its declared
NOAA approximation. Chemistry's separate orbital implementation is outside
the WRF radiation driver's equation-of-time change.

The stock WRF namelist has no spelling for this implementation selector.
Ordinary import therefore resolves the corrected generic default. Both
generations round-trip through the existing physics-selector comment table
and `[shared]` configuration. The selector is global across a domain tree.
Restart echoes omit only `wrf_461`, which an older header without this key
already means. Corrected headers bind the explicit selector and the solar
algorithm identity. A generation change refuses the checkpoint.

The referee extracts unmodified `radconst` and `calc_coszen` from each
pinned release, with an empty `module_wrf_error` stub. Those routines use no
error-module state. GNU Fortran uses REAL(4), `-O0 -ffp-contract=off
-fcheck=all -finit-real=snan -ffree-line-length-none`. Every solar word is
graded over ten dates and 4,370 points per date. The fixture build receipts
pin source hashes, compiler, flags, kinds and dependencies. This is a solar
routine referee, not a full WRF 4.8 radiation-scheme or forecast claim.

Reproduce on a box:

```
python -m tools.solar_geometry_oracle.prepare_versions /work/<lane>/sources
python -m pytest -q tests/test_solar_generation.py
# Within a GPU queue hold, once for default and once with GPUWM_WRF_EXACT=1:
python -m tools.solar_geometry_oracle.compare_versions tests/data/oracles/solar_generations /work/<lane>/solar.json
python -m pytest -q tests/test_solar_generation_gpu.py
```

The one-bit negative control flips mantissa bit 20 of the corrected sine
coefficient in a private image. Production source stays intact. The lowest
coefficient bit was also tried and rounded away before the output clock;
it is not claimed as a successful negative control.

## WRF 4.8.0 native replay and the frame calendar (2026-11-09)

`tools/wrf480_solar_replay` links a small driver against the referee build's
own `module_radiation_driver.o` (WRF 4.8.0 06d4240ae989 and 4.7.1
f52c197ed39d, strict serial, GNU Fortran 13.3.0, `-O2 -fno-fast-math
-ffp-contract=off -fno-tree-vectorize`) and calls `radconst` and
`calc_coszen` the way `radiation_driver` does: the radiation call at
`xtime+radt*0.5`, and the swint call at `xtime`. Inputs are WRF's frame
calendar words for the A071 and HRRR-recipe hours on an A071-shape grid,
plus a whole leap year, the binade-edge days and a 30-minute start across a
new year on a global 2-degree grid: 257,248,756 words (DECLIN, SOLCON,
COSZEN, HRANG, COSZEN_LOC, HRANG_LOC).

The replay found one defect in the corrected generation: the radiation
calendar rounded JULIAN once, as `REAL(day)`, where WRF forms
`REAL(dayOfYear_r8) - 1.0`. On days 0, 1, 3, 7, 15, 31, 63, 127 and 255
about half of the calls carried a different JULIAN word, and DECLIN, the
equation of time, COSZEN and HRANG followed it. `wrf_480` now takes its
JULIAN, XTIME and GMT from `woof.core.solar.radiation_calendar` (WRF's
frame words, `woof.core.wrf_radiation_clock.wrf_calendar_scalars`) in
every solar consumer: legacy RRTMG (solar, ozone and aerosol of the call
read one word), `swint.calendar_scalars` and its callers, `geometry_at`
(Dudhia, RRTMGP, urban, Noah-MP and the analytic carriers), terrain
radiation and RRTMGP SOLCON. `wrf_461` keeps its earlier words byte for
byte.

Result: `wrf_480` against native 4.8.0 is 0 of 257,248,756 words different
when native runs on the host's glibc 2.43 libm, the runtime the A071 gate
uses. Inside the build container (glibc 2.39) native's own ASIN rounds
DECLIN differently on some calls; the same binary on the two libms differs
by exactly the residue (1,227,093 words; 4 DECLIN words in 82 A071/HRRR-
recipe calls), so that gap is the maths library, not the port.
