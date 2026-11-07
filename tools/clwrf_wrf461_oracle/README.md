# Stock CLWRF climate diagnostics

Authority: public WRF v4.6.1 commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`,
`phys/module_diag_cl.F`. The file is byte-identical in the repository's
WRF v4.7.1 pin, commit `f52c197ed39d12e087d02c50f412d90d418f6186`.
Its SHA-256 is
`2ec1e964a9ae94244d513fac38e0ce9a33212b21ca232fcc01de890afd152a54`.

The source is distributed under WRF's public-domain statement:
<https://github.com/wrf-model/WRF/blob/v4.7.1/LICENSE.txt>.
The source has no additional restrictive licence notice. This module
requires no external table. The script compiles the unchanged source,
with small framework stubs, using the flags recorded in its receipt.

Run from the repository root, passing a directory containing the public
source and a private output directory:

```sh
python tools/clwrf_wrf461_oracle/run.py /path/to/WRF/phys /path/to/output
GPUWM_WRF_EXACT=1 python -m pytest -q tests/test_clwrf.py::test_native_stock_words_at_every_call
```

The committed fixture contains 41 calls and 8,897 native output words.
Inputs are constructed. Cases cover interval reset/finalization,
nonintegral step counts, restart calls, a one-step interval and large
integer-valued clocks. These are module checks, not a real forecast
capture or verification against observations.

The CUDA path preserves the native 273 K Celsius conversion, native
timestamps and untouched wind maxima on reset. Native scalar `NSTEPS`
is represented by equal per-column integer counters so tile reuse does
not serialize the calculation. History uses the native Registry names,
including `T2MIN` and `RAINCVMAX`, rather than the routine's formal names
containing `CL`.

Stock namelists select this path with `output_diagnostics=1` and a
positive `auxhist3_interval` on every domain. The interval is in minutes.
The existing operational fork path retains `diag_int` and `wind_int`.
The 2026-11-01 continuation checks native auxiliary-stream alarms, a real
native restart-file warm start, checkpoint continuation, both nest feedback
modes and the ordinary production ensemble. Auxiliary streams write one frame
per file; native multi-frame file grouping is not reproduced. The three-hour
stock `woof go` case is 2024-05-25 18--21 UTC with MP8 and a 12 s step.
Hourly Rust-rendered CLWRF means agree closely with the native case, with
differences concentrated near cloud edges. Surface observations supply context;
this is not a precipitation-statistics skill assessment.

For actual native inputs, instrument a private source copy before building
with strict native numerics:

```sh
python tools/clwrf_wrf461_oracle/capture_wrf461.py \
  --wrf-source PRIVATE_WRF_COPY --patch clwrf-instrumented-native.patch
```

Run that executable with `output_diagnostics=1, auxhist3_interval=60`.
The first three monitor-rank calls write `lane-real-clwrf-001.bin`
through `003.bin`. Each records seven input planes, the actual incoming
36 carrier planes and scalar counter, clocks and native outputs. No model
field or clock is changed. The capture selects up to 16 by 16 points
around the monitor rank's precipitation maximum.

```sh
GPUWM_WRF_EXACT=1 python tools/clwrf_wrf461_oracle/replay_capture.py \
  lane-real-clwrf-001.bin lane-real-clwrf-002.bin lane-real-clwrf-003.bin \
  --out clwrf-real.json
```

The 2026-10-16 09 UTC MP6 case matched 27,651 output words across real
calls at 0, 15 and 30 seconds, with zero differences. Those calls cover
reset and accumulation. Native per-step precipitation was zero during
these initial calls; nonzero precipitation and interval finalization are
covered only by the separate constructed replay. Raw captures are deleted
after the small JSON receipts record their hashes and comparison counts.
