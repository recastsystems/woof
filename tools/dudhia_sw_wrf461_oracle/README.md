# Dudhia shortwave (ra_sw_physics=1) WRF v4.6.1 column oracle

Grades WOOF's Dudhia shortwave word for word against SUBROUTINE SWRAD of the
byte-unmodified WRF v4.6.1 `phys/module_ra_sw.F`.

| file | role |
|---|---|
| `make_columns.py` | builds the input columns (named regimes, or `--stress N` randomised columns) |
| `run_swrad.F90` | Fortran driver: `swinit`, one `SWRAD` call per case, and the radiation driver's `SWDOWN=GSW/(1.-ALBEDO)` |
| `build.sh` | checks the pinned WRF commit, compiles the reference (-O0, no FMA, scalar libm) and the stock-flags build, runs both, builds `libm_sweep.so` |
| `libm_sweep.c` | the C library's `log10f`/`powf` over a run of float32 words |
| `compare_gpu.py` | runs WOOF (column API and production adapter) and grades every word |
| `sweep_libm.py` | exhaustive check of the kernel's two libm calls on every float32 in [1, +inf] |
| `coverage.py` | which SWPARA branches the columns reach (descriptive only) |

## Rerun (box W1 layout)

```
T=/work/pverify/<lane>/src; source /work/pverify/base/env.sh; cd $T
PYTHON=python bash tools/dudhia_sw_wrf461_oracle/build.sh \
    /work/pverify/wrf-build/src/WRF-4.6.1 /work/pverify/<lane>/oracle/build
PYTHON=python bash tools/dudhia_sw_wrf461_oracle/build.sh \
    /work/pverify/wrf-build/src/WRF-4.6.1 /work/pverify/<lane>/oracle/stress \
    --stress 20000
# GPU work only through the mutex:
card-lease <lane> --cards 1 --est 900 --wait 1200 bash -c '
  T=/work/pverify/<lane>/src; source /work/pverify/base/env.sh; cd $T
  B=/work/pverify/<lane>/oracle
  GPUWM_WRF_EXACT=1 python tools/dudhia_sw_wrf461_oracle/compare_gpu.py $B/build
  python tools/dudhia_sw_wrf461_oracle/compare_gpu.py $B/build
  GPUWM_WRF_EXACT=1 python tools/dudhia_sw_wrf461_oracle/compare_gpu.py $B/stress
  python tools/dudhia_sw_wrf461_oracle/compare_gpu.py $B/stress
  GPUWM_WRF_EXACT=1 python tools/dudhia_sw_wrf461_oracle/sweep_libm.py $B/build
  python tools/dudhia_sw_wrf461_oracle/sweep_libm.py $B/build
  GPUWM_WRF_EXACT=1 python -m pytest -q tests/test_dudhia_sw_wrf461_parity.py
  python -m pytest -q tests/test_dudhia_sw_wrf461_parity.py'
```

`compare_gpu.py ... --write-fixture tests/data/oracles/dudhia_sw` rewrites the
committed fixture from a build directory.

## Scope

Graded: SWRAD/SWPARA on the meteorological path (every species flag set, no
WRF-Chem aerosol, no eclipse, no slope radiation) and the driver's SWDOWN.
Not graded here: the solar geometry (COSZEN, SOLCON), which the radiation
driver computes for every shortwave scheme; the oracle supplies those words.
