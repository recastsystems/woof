# WDM6 (mp_physics = 16) column oracle, WRF v4.6.1

Grades WOOF's WDM6 CUDA kernel (`woof/core/kernels/wdm6.cu`, launched by
`woof.core.wdm6.launch_wdm6`) and its 10 cm reflectivity
(`woof/core/kernels/wdm6_refl.cu`, `woof.core.refl.launch_refl10cm_wdm6`)
word for word against WRF v4.6.1's own Fortran.

## What it runs

`run_wdm6.F90` calls the public `wdm6` entry of `phys/module_mp_wdm6.F`
exactly as `module_microphysics_driver.F:2476-2512` does, after `wdm6init` as
`module_physics_init.F:4583` calls it, with every physical constant taken from
WRF's own `share/module_model_constants.F`, `diagflag = .true.`,
`do_radar_ref = 1` and `has_reqc = has_reqi = has_reqs = 1`. The columns are
the i dimension of one j row. It also dumps the words `wdm6init` and
`radar_init` leave in the modules' saved variables.

`build.sh WRF_SOURCE_ROOT BUILD_DIR` compiles it against the pinned source
(commit `d66e442f`, `module_mp_wdm6.F` sha256 `78b7c148...`; the build refuses
anything else) at `gfortran -O0`, with `frame/libmassv.F` for the `VREC`/
`VSQRT` loops and a service-only `module_wrf_error` stub. It fails if the
linked reference holds a libmvec vector symbol or an FMA instruction, and a
positive control proves the libmvec grep can fire. Five arms:

| executable | source |
|---|---|
| `run_wdm6_stock` | the unmodified file -- the comparison of record |
| `run_wdm6_defined` | the file with the three defined behaviours of `docs/wdm6_oracle_known_deltas.md` (sections 2, 5, 6) substituted by `defined_arm.py` |
| `run_wdm6_only2`, `_only5`, `_only6` | one substitution each, to attribute the stock differences |

`fixture.py OUTDIR` writes the 84-column fixture (convective land and ocean,
stable night, winter snow and ice, a melting layer, high terrain, a cold
anvil below -40 C, rain evaporating into dry air, numberless rain and cloud,
random mixtures and 16 edge cases: zeros, negatives from advection, CCN
outside its clamp, subnormals, exact thresholds, exactly 273.15 K, enormous
and tiny drops, strong supersaturation, heavy graupel, an XLAND just below 2,
ice only, warm cloud only, maritime autoconversion) in seven configurations:

| configuration | dt (s) | minor loops | calls | hail_opt | levels |
|---|---|---|---|---|---|
| dt20_hail0 | 20 | 1 | 1 | 0 | 50 |
| dt20_hail1 | 20 | 1 | 1 | 1 | 50 |
| dt20_hail0_coldstart (itimestep 1) | 20 | 1 | 1 | 0 | 50 |
| dt300_hail0 | 300 | 3 | 1 | 0 | 50 |
| dt150_hail1_x3 | 150 | 1 | 3 | 1 | 50 |
| dt60_hail0_x10 | 60 | 1 | 10 | 0 | 50 |
| deep72_dt30_hail0_x2 | 30 | 1 | 2 | 0 | 72 (WOOF's WDM6_KMAX=80 tier) |

`run_oracle.sh` runs every arm on every input. `run_woof.py` runs WOOF's
production launchers on the same inputs (the arithmetic follows
`GPUWM_WRF_EXACT`). `compare.py` reports differing words, max ULP (the
repository's `fp32_signed_ulp`) and max absolute difference per field and
step; `summarize.py` condenses its JSON. `mirror_main.cpp` compiles the kernel
source for the host (`WDM6_CPU_MIRROR`) for fast iteration on a CPU; it is an
aid, not evidence.

Evidence arms, never graded against WOOF: `measure_split.sh` builds
`only5t` (section 5's conservative transfer with WRF's own sub-step count,
`split_arm.py`) and `fine8`/`fine32` (section 5 at 8 and 32 times WOOF's
count); `convergence.py` reports each arm's distance to `fine32`.
`measure_o2.sh` builds the unmodified file at WRF's own GNU physics flags
(`-O2 -ftree-vectorize -funroll-loops`, libmvec) to show stock WRF's spread
around the `-O0` reference.

Effective radii: WRF publishes metres and WRF's RRTMG forms microns as
`re*1.E6` in REAL; WOOF stores that micron word, so `compare.py` scales WRF's
radii by that one float32 multiply before comparing.

## Rerun (W1 recipe)

```bash
T=/work/pverify/<lane>/src; source /work/pverify/base/env.sh
O=$T/tools/wdm6_wrf461_oracle
bash $O/build.sh /work/pverify/wrf-build/src/WRF-4.6.1 build
python3 $O/fixture.py runs/fixture
bash $O/run_oracle.sh build runs/fixture runs
# GPU, through the mutex: strict and default builds
card-lease <lane> --cards 1 --est 600 --wait 900 bash $O/gpu_run.sh $T runs/fixture runs/woof
python3 $O/compare.py runs/fixture runs/defined runs/woof/strict
python3 $O/compare.py runs/fixture runs/defined runs/woof/default
python3 $O/compare.py runs/fixture runs/stock runs/woof/strict
python3 $O/compare.py runs/fixture runs/stock runs/only5 --fortran-pair
# evidence arms (CPU)
bash $O/measure_split.sh /work/pverify/wrf-build/src/WRF-4.6.1 build runs/fixture runs
python3 $O/convergence.py runs/fixture runs
bash $O/measure_o2.sh /work/pverify/wrf-build/src/WRF-4.6.1 build runs/fixture runs
python3 $O/compare.py runs/fixture runs/stock runs/stockO2 --fortran-pair
```

The committed fixture and both reference arms are in
`tests/data/oracles/wdm6/` (gzip); `tests/test_wdm6_wrf461_oracle.py` grades
the device kernel on them (`-m gpu`, with and without `GPUWM_WRF_EXACT=1`),
pins the coefficient and radar words, and runs the host mirror on a glibc
host.

## Result (2026-10-07, RTX PRO 6000, NVRTC 12.9, CuPy 14.2)

* Strict build: every output word of every field and step of all seven
  configurations equals the defined arm (0 of 1,180,116 words differ).
* Against the stock arm: bitwise in the columns where none of the three
  behaviours fires; differences confined to the columns where section 5
  (conservative rain transfer) or section 6 (zero-rate condensation cap)
  fires. Section 2 (the kt clamp) never fires on the fixture.
* Default build (flush-to-zero route): state differences only in columns
  into which a subnormal moisture value enters a call; elsewhere a surface
  accumulator may lose a vanishing fallout (under 5e-34 mm).

Details and numbers: `docs/wdm6_oracle_known_deltas.md`.
