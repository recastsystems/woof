# NSSL variant referee

The production entry supports eight resolved NSSL modes, including the native
one-moment options 19 and 21. Three moments remain refused. Public source pins,
licence checks, coefficient decisions and limits are in `SOURCE_PIN.json`.
No external tables or competitor implementation are used.

The lane evidence qualifies CCN 1e7, shape 2.5/0 and independent drag 2/4 in
two-moment hail/predicted-CCN mode: each real native replay matches 6,228 active
words, and mixed-phase fixtures match 10,614 words. Each completes a 900-step
real forecast, viewed Rust native/observation maps at three hours, whole nests
with feedback 0/1, exact 65,987,832-word two-domain checkpoint continuation,
and two full ensemble members against ordinary controls and Rust aggregates.
Production-stage ratios to their nearest controls are 1.039, 0.964 and 0.964.

One-moment real replay matches 2,994 active words for option 19 and 2,406 for
option 21, with 882 native-unwritten radius words separately witnessed as
unchanged bytes, including NaN residue. Those arguments are not finite active
physics outputs. WRF disables `has_req*` for these modes; the production
radiation adapters follow that contract. Default and jointly nondefault
(CCN 1e7, shape 2.5/0, drag 2/4) mixed-phase fixtures each match all 10,614
words. `tests/test_nssl1_wrf461_parity.py` replays preserved small fixtures
through the same CUDA entry as the model, under `GPUWM_WRF_EXACT=1` and
`GPUWM_WRF_GLIBC=2.43` for the recorded native math ABI.

Both one-moment modes complete separate native and GPU 900-step real forecasts.
Rust maps at 19/20/21 UTC show closely matching surface patterns and a southwest
observed storm missed by both forecasts. Both modes complete both whole nest
feedback modes, exact two-domain continuation, and full two-member ensembles.
Their fast production-stage totals are 0.776352 and 0.695584 ms for 22,500
columns with 49 levels: 0.871162 and 0.805522 times the nearest two-moment
control. A complete default-control forecast retains four byte-identical
history files and all 62,383,800 checkpoint words. Fast math remains default;
strict math is the replay opt-in. The lane report gives commands, configurations,
source locations, stage numbers, observations and receipts.

Gathered CD selectors are qualified at graupel 7 / hail 2 with CCN 1e7.
A complete 74-column native row at timestep 112 preserves wet-64 context;
all 76,812 active words match exactly. Four independent frozen branches
each match all 10,614 words. Native and GPU each complete 900 steps, and
all 18 Rust native/observation comparison panels at three hours are viewed.
Both share the missed southwest storm and warm/dry station biases. Whole
feedback-0/1 nests, exact 65,987,832-word continuation, two concurrent full
ensemble members (4,478 array hashes and Rust aggregates), and default
history/checkpoint identity pass. On 22,200 real-capture columns x49 levels,
the fast total is 0.872704/0.861952 ms = 1.012474 times the nearest 6/6
control. Full native frozen NUCOND alone is 2.758671 times its fast warm
control (six extra fields, 86 versus 71 registers, no local spill). The
parallel gathered scan replaces the initial measured row-scan bottleneck.
CUDA events order shared table initialization across ensemble streams;
derived CD scratch is prepared, priced and rebuilt on restart.

The build pins the canonical-LF SHA-256 of the unchanged WRF 4.6.1 NSSL module.
It uses `-O2 -fno-tree-vectorize -fno-fast-math -ffp-contract=off` and records
the compiler, flags and source hash. It runs the actual public NSSL initializer
and driver, with the module's default internal namelist. Inputs are shared
float32 values. Outputs use hexadecimal float images, saved at every step.
The CSV reader refuses duplicate, missing and out-of-range rows.

There are six independent regimes and 19 configurations. The coefficients'
test endpoints are fixture choices, not bounds imposed by WRF. The cited
`share/module_check_a_mundo.F:3377-3465` contains selector consistency and
deprecated-ID remaps, not coefficient bounds. In the pinned 4.6.1 source,
the 19/21 remaps do not set `nssl_ccn_on`; the fixture uses its default of 1.
WRF 4.7.1 explicitly sets it to zero for option 21. That later selector
change is not imported into this 4.6.1 numerical oracle.

The stock parameter-array route uses `ABS(nssl_params(1))` for CCN and
assigns shape and drag options directly. The optional scalar initializer
arguments have different guards and are not the stock WRF caller. See
`SOURCE_PIN.json` and `NOTICE` for the public source and licence audit.

The comparator requires every selected configuration, every state and diagnostic
array at every step, finite values, identical input hashes and run settings, and
the full expected trajectory shape. The strict arm requires identical float32
images, including signed zero. It measures ULP, absolute and relative errors.
The default budgets are declared in `compare.py`; passing the comparator's
unit tests does not establish that the numerical implementation meets them.
Net tendencies are `(state_after - state_before) / dt` in float32, not internal
instantaneous process rates. Their default budget uses the corresponding state
budget in units per second.

CPU commands, with `T` set to the worktree before sourcing the box environment:

```sh
source /work/env.sh
bash tools/nssl2_wrf461_variant_oracle/build.sh /work/wrf/src/WRF-4.6.1 "$L/oracle-build"
python tools/nssl2_wrf461_variant_oracle/run_oracle.py "$L/oracle-build/nssl2_variant_column" "$L/oracle"
CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 python tools/nssl2_wrf461_variant_oracle/run_gpu.py "$L/admission" --admission-only
CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 python -m pytest -q tools/nssl2_wrf461_variant_oracle/test_compare.py tools/nssl2_wrf461_variant_oracle/test_oracle.py
```

The following fixture commands describe the synthetic harness; they are not
the required real native-run replay below. Check free GPU memory before a run.
Both engines generate their own fixtures from `cases.py`; do not transfer
numerical output between boxes.

```sh
python tools/nssl2_wrf461_variant_oracle/run_gpu.py "$L/gpu"
python tools/nssl2_wrf461_variant_oracle/compare.py "$L/oracle" "$L/gpu" "$L/comparison.json" --arm strict
```

The GPU runner calls the production coordinator, GS, condensation, diagnostics
and scatter. The final moist-physics callback is empty because this is an
isolated microphysics column. This does not exercise dynamics, boundaries or a
public-namelist forecast. The focused mutation test flips one bit in the CCN
coefficient inside NUCOND. It checks the production coefficient seam.

## Real native-run captures

`capture_native.py` adds output-only capture blocks to the pinned public module.
Removing those marked blocks recovers the exact source image. It records the
source and generated-source hashes. Run the resulting whole native WRF build
with a stock namelist; the first native driver call is captured by default.
`--wet-threshold 1e-5` instead selects the first post-initial call whose monitor
rank has a column with integrated condensate above that threshold. It saves
up to six of that rank's wettest columns, all model levels, their native grid
coordinates, actual resolved coefficients, and both sides of the driver call.
This selection is local to the monitor rank, not a global maximum or a promise
that every mixed-phase process has executed.

Inactive WRF Registry arguments may be PRESENT but null. The capture must not
read them: it writes zero placeholders and the replay excludes these words from
proof. It never fills active fields with manufactured meteorology. For gathered-CD selectors, `--complete-row` captures every X column at one
native Y in native order; version 2 validates that complete row. The binary
stream is little-endian version 1 or 2; `read_capture` validates its full length and
axes before any launch. `test_native_capture.py` checks that contract.

```sh
python tools/nssl2_wrf461_variant_oracle/capture_native.py \
  /path/to/official/module_mp_nssl_2mom.F /path/to/own/phys/module_mp_nssl_2mom.F \
  --wet-threshold 1e-5
# Build and run the whole native WRF executable in owned scratch.
GPUWM_WRF_EXACT=1 python tools/nssl2_wrf461_variant_oracle/replay_native_capture.py \
  /path/to/native/nssl-native-capture.bin /path/to/replay.json
python tools/nssl2_wrf461_variant_oracle/benchmark_native_capture.py \
  /path/to/native/nssl-native-capture.bin /path/to/stage-time.json --columns 22500
```

The strict replay counts float32 images, including signed zero, requires finite
active native and GPU results, and admits no tolerance. Its receipt identifies
unported selectors without counting refused execution as compared words. The
benchmark uses seven alternating pairs of production CUDA launches on replicated
captured columns, excludes the first two warm-up pairs, and reports stage medians
and the ratio to otherwise identical default-CCN option 18. It measures the
microphysics stages only. Neither tool replaces the whole forecast, observation
maps, restart, nesting, ensemble, or default-identity checks required for a port.


## Nondefault graupel and hail shapes

`generate_shape_kernels.py` derives separate numerical units from the pinned
transcriptions using checked anchors. `--check` detects stale generation.
The default source images remain unchanged. Both real-valued coefficients
are native FP32 words in compile-time definitions, cached per device. There
are no per-step coefficient uploads or host reductions. Source table bounds
come from the 4.7.1 audit; alpha 15 is a lookup clamp, not a coefficient cap.
The 2.5/0 qualification includes a real 6,228-word replay, mixed-phase driver
fixture, full real forecast with Rust native/observation maps, both nest
feedback modes, checkpoint continuation and whole ensemble forecasts.


## Gathered scalar CD inheritance

The positive graupel selectors >6 and hail selectors 2 or >6 reuse CD from
an earlier mass/number loop in native `setvtz` (:7135-7417). They are admitted.
`carry_transcription.py` preserves both the GS gathered wet-64 ordering
(:13626-13677) and the entire-column ordering before sedimentation. Parallel
piece scans, a scan of piece counts, group last-point selection and a final
lookup avoid traversing a full meteorological row on one block. Scratch is
priced, allocated before stepping and rebuilt after restart. No per-step
host synchronization or CPU physics is introduced. Four mixed-phase branch
fixtures each preserve all 10,614 words, including the known-graupel-number
source for inherited hail CD. The full real-row replay, forecast, Rust map,
lifecycle and timing receipts are recorded in the lane report.
