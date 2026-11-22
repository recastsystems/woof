# Compiled WRF v4.7.1 Runge-Kutta reference

The harnesses call the unchanged `rk_addtend_dry`, `rk_update_scalar`,
`rk_step_prep`, and `rk_tendency` routines from WRF v4.7.1 `module_em.F`.
Their extracts retain the original argument lists and every executable line.
The source pin is commit `f52c197ed39d12e087d02c50f412d90d418f6186`.
`rk_build.py` refuses different source bytes and records each extract hash.
The actual WRF `module_model_constants.F` is compiled, and the generated WRF
configuration module is copied from a native build rather than replaced.

The isolated scalar and dry references use gfortran 15 at `-O0`, with
contraction disabled and bounds checking enabled. The complete tendency
reference compiles the unchanged RK routines at those same settings and links
copied native WRF dependency libraries. Those libraries were built at
`-O2 -fno-fast-math -ffp-contract=off`. The full build receipt pins every
dependency library and extra object.

The source folder supplied to these commands contains copied
`module_em.F` and `module_model_constants.F`. The native build folder contains
the generated configuration module and WRF libraries. Neither source folder
is modified.

```bash
nice -n 10 python tools/bigstep_wrf471_oracle/rk_build.py \
    "$WRF_SOURCE" "$WRF_NATIVE_BUILD" "$BUILD_DIR/rk"
nice -n 10 python tools/bigstep_wrf471_oracle/rk_pack.py \
    "$BUILD_DIR/rk/rk_oracle.so" \
    tests/data/wrf471_bigstep/state-real.npz \
    tests/data/wrf471_bigstep/rk-cases.npz
nice -n 10 python tools/bigstep_wrf471_oracle/rk_full_build.py \
    "$WRF_SOURCE" "$WRF_NATIVE_BUILD" "$BUILD_DIR/full"
nice -n 10 python tools/bigstep_wrf471_oracle/rk_tendency_pack.py \
    "$BUILD_DIR/full/rk_tendency_harness" \
    tests/data/wrf471_bigstep/state-real.npz \
    tests/data/wrf471_bigstep/rk-tendency.npz "$BUILD_DIR/full-run"
nice -n 10 python tools/bigstep_wrf471_oracle/rk_tendency_pack.py \
    "$BUILD_DIR/full/rk_tendency_harness" \
    tests/data/wrf471_bigstep/state-real.npz \
    tests/data/wrf471_bigstep/rk-tendency-stored-theta.npz \
    "$BUILD_DIR/stored-theta-run" --stored-theta
```

The reference uses a real 12 by 10 by 49 state crop, native staggering,
four horizontal halo rows, and the full top row. The dry cases use explicit
state-derived tendencies: initial field divided by 64, held field divided by
2048, and held theta heating `T/4096`. Boundary save arrays are zero, and the
held geopotential physics tendency is zero. The scalar cases use six real
moisture species, a held rate equal to each species divided by 128, and a
common advection tendency `QVAPOR/256`. They cover all three RK stages,
specified and nested boundary rows, map factors from 0.25 to 8, and values
scaled by `1e-25`. Scalar advection decomposition diagnostics are disabled,
as in the default run; their unchanged memory is still compared.

The complete tendency cases cover RK stages one and two plus reversed
hemisphere rotation, with specified boundaries and horizontal order five /
vertical order three. Their fluxes, pressure-point geopotential, masses,
inverse density and moisture coefficients come from the native
`rk_step_prep` call. Optional diffusion, damping and implicit vertical
advection are disabled. The stored-theta fixture feeds the literal WRF
theta-minus-300 words to both transport operators. The additional full-theta
fixture feeds WRF T+300 to both operators, which measures the physical-theta
carrier WOOF-ArWen normally transports. It is explicitly an operator probe,
not the literal WRF stored-T call. Native `solve_em.F` passes `grid%t_2`
and `grid%t_1` to these RK calls. Neither probe is a complete forecast
comparison.

The pressure adapter gives Fortran the perturbation word recovered from the
engine's total-pressure storage. It retains original input words and a
separate original-pressure reference. On this real crop the round trip
changes zero of 5,880 pressure words, with maximum loss zero Pa.

The GPU replay uses the production word-copy kernel, held mixing and heating
paths, fused scalar updater, complete slow-tendency driver and CFL recorder.
It compares every supplied output memory word, including unchanged halos.
Measurements include a SHA256 of every actual and reference array, so an
unchanged maximum or mismatch count cannot hide a changed output.

Under the applicable GPU ownership protocol:

```bash
nice -n 10 python tools/bigstep_wrf471_oracle/rk_measure.py \
    "$RECEIPT_DIR/rk-replay.json" --full
nice -n 10 python -m pytest -q tests/test_bigstep_rk_wrf471_parity.py
```

`rk-ulp-table.json` contains measured equality pins. They are drift checks,
not tolerances or a statement that every full-tendency output agrees with
WRF. The full routine is explicitly recorded as DIFFERENT. The timestep
mutation test demonstrates that the compiled scalar reference detects a
changed update interval.

## Results and representation differences

`rk_update_scalar` is bit-identical in all seven cases, including all copied
and unchanged output memory. `rk_addtend_dry` is bounded by one ULP in five
cases: WRF multiplies the V tendency by the separately rounded
`msfvx_inv`, while the production mixing path divides by `msfvx`. The other
dry outputs are bit-identical.

`rk_tendency` is DIFFERENT in both three-case probe families. The CUDA flux
helpers combine velocity into a numerator and divide once, while WRF
divides the central and dissipative interpolation terms separately before
multiplying by velocity. Their dissipative sums also associate differently.
The mapped scalar kernel combines horizontal components before applying
the map factor, while WRF applies each component in Y, X, vertical order.
These FP32 evaluation trees are part of the existing forecast kernel's
frozen Phase 2 arithmetic; see `advection.cu:47-74` and `:77-104`.
FMA contraction adds a separate rounding difference.

The full-theta discrepancy is completely reproduced by the diagnostic
`rk_advection_diagnostic.py`: disabling FMA alone does not remove it;
restoring WRF's flux evaluation order reduces it; restoring the mapped
accumulation order then gives bit identity for all 18,000 theta memory
words in each of the three cases. This control changes transient compiled
source only and does not change the forecast kernel. Full-tendency pins
record this difference rather than accepting it under a tolerance.
The same control run on the stored-theta fixture (`--fixture
rk-tendency-stored-theta.npz`) also reaches bit identity for all 18,000
theta words in each case, from a production maximum of 3,658,240 ULP and
0.0430755615234375 absolute (`receipts/rk-stored-theta-attribution.json`).
The theta tendency does not depend on the RK stage or on the rotation
sign, so the three cases give the same theta words; they differ in the
momentum outputs.

`receipts/rk-replay-b200.json` is an independent replay on an sm_100
card with CUDA runtime 13.2. Every standalone and full-call measurement,
including every actual and reference array hash, equals the pins recorded
on an sm_120 card.

WRF also overwrites the interior `cqw` mixing-ratio temporary with its
consumed reciprocal during `pg_buoy_w`. The engine builds that reciprocal
at preparation time, as explicitly documented in `acoustic.cu:112-113`.
Their 5,760 interior words match; 240 top and bottom words retain different
representations. The same three cases produce bit-identical held arrays,
save arrays, mass tendency, explicit/implicit flux split and both CFL
outputs.
