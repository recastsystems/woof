# Eta similarity surface layer: WRF v4.6.1 column oracle

`sf_sfclay_physics = 2` (paired with the MYJ PBL). WOOF module:
`woof/core/myjsfc.py`, `woof/core/myjsfc_tables.py`,
`woof/core/kernels/myjsfc.cu`. WRF module: `phys/module_sf_myjsfc.F`.

## What it is

`build.sh` compiles the byte-unmodified `phys/module_sf_myjsfc.F` and
`share/module_model_constants.F` from a WRF checkout at v4.6.1
(`d66e442fccc04111067e29274c9f9eaccc3cef28`; the script refuses any other
commit or an edited file) and links them with `run_myjsfc.F90`, a column
driver. No stub framework is needed: without `DM_PARALLEL` the module uses
only the constants module.

The driver calls `MYJSFCINIT` once (it builds the PSIM/PSIH tables) and then
`MYJSFC` several times in a row, carrying WRF's own INOUT state between
calls exactly as the surface driver does. It dumps the tables and the
post-call value of all 35 INOUT and output fields after every call.

`make_cases.py` writes three deterministic sets, 224 columns in all:

| set | columns | levels | calls | what it covers |
|---|---|---|---|---|
| `cold` | 96 | 40 | ITIMESTEP 1, 2, 3 | first-step branches (USTAR 0.1, TZ0 = TSK, QS = QLOW, AKMS = AKHS = CXCHS over a calm sea) and two warm calls |
| `warm` | 96 | 40 | ITIMESTEP 8, 9, 10 | seeded state; the sea branch starts in each viscous regime (USTAR below 0.225, 0.225 to 0.7, above 0.7) |
| `nz2` | 32 | 2 | ITIMESTEP 1, 2 | the two-level minimum column |

Each set cycles 17 regimes: convective land, stable night land, snow and
sea ice, warm ocean, cold ocean, high terrain (HT to 4.4 km), calm air,
cloud water at the lowest level, a TKE column that never drops below EPSQ2
(LPBL = 1), a dead TKE column, a boundary layer above 1000 m (BTGH branch),
extreme instability, extreme stability, a saturated cold column (shelter
clamp), storm-force wind over water, rough forest, and a lowest layer under
4 m (the 2 m diagnostic above the lowest level).

Receipts the build writes (copies in `tests/data/oracles/myjsfc/`):

- `compiler.txt`: gfortran 13.3.0, glibc 2.39, the CPU, the objects' libm
  symbols (`logf`, `expf`, `powf`, `atanf`, `__powisf2` at -O0), and the
  FMA instruction count (0).
- `o2-equality.txt`: WRF's own phys flags (`-O2 -ftree-vectorize
  -funroll-loops`) write the same bytes as the -O0 reference on every set.
- `coverage-myjsfc.txt`: every executable line of `MYJSFC` and `SFCDIF`
  runs; only `MYJSFCINIT`'s non-restart USTAR seed does not (the driver
  calls it with RESTART true and seeds USTAR itself).
- `oracle-sha256sums.txt`: the case and output streams and the fixture.

## Grading

- `woof/verify/myjsfc_oracle.py` grades the host tables and the CUDA
  kernel, every field bitwise, replaying WRF's pre-call state and
  free-running on the kernel's own.
- `woof/verify/myjsfc_oracle.py::measure_cpu_authority` grades the
  float32 CPU twin (`woof/verify/myj_ref.py::np_myjsfc_column`) the same
  way, on CPU.
- `compare_myjsfc.py` prints the table (max ULP, mismatch count, max abs
  difference, flushed subnormals per field) and exits 0 only on a full
  match; `--cpu-authority` grades the CPU twin instead of the kernel.
- `tests/test_myjsfc_wrf461_parity.py` is the gate (tables and the CPU
  twin on CPU, kernel on GPU, plus a negative control that zeroes HT).
- `libm_sweep.py` compares the kernel's float32 log, exp and pow forms with
  the oracle host's C library at every one of the 2**32 float32 inputs.

## Random-regime stress measurement

`stress.sh BUILD_DIR STRESS_DIR [NCOL]` reuses a finished build and writes
two further sets of NCOL columns each (default 4096; 50 levels; 4 calls;
ITIMESTEP from 1 and from 5), every column's regime drawn at random. It is
not committed (16 MB); `compare_myjsfc.py --fixture STRESS_DIR/myjsfc-stress.npz`
grades it the same way.

Measured on W1 (RTX PRO 6000 Blackwell, driver 595.91.07, NVRTC 12.9) with
NCOL 4096, 32,768 column-calls: 0 mismatching words on all 35 fields,
replayed and free-running, under `GPUWM_WRF_EXACT=1` and under default
arithmetic, and 0 for the CPU twin (`--cpu-authority`, 22 s on one core).

## Rerun (box W1)

```bash
# oracle (CPU only, about 1 minute)
bash tools/myjsfc_wrf461_oracle/build.sh /work/pverify/wrf-build/src/WRF-4.6.1 /tmp/myjsfc-oracle
cmp /tmp/myjsfc-oracle/myjsfc-wrf461.npz tests/data/oracles/myjsfc/myjsfc-wrf461.npz \
  || python - <<'EOF'   # npz zip timestamps differ; compare the arrays
import numpy as np
a = np.load("/tmp/myjsfc-oracle/myjsfc-wrf461.npz"); b = np.load("tests/data/oracles/myjsfc/myjsfc-wrf461.npz")
assert a.files == b.files and all(a[k].tobytes() == b[k].tobytes() for k in a.files)
print("fixture reproduced")
EOF

# kernel against the oracle, one card through the mutex
card-lease <lane> --cards 1 --est 600 --wait 1800 bash -c '
  GPUWM_WRF_EXACT=1 python tools/myjsfc_wrf461_oracle/compare_myjsfc.py
  python tools/myjsfc_wrf461_oracle/compare_myjsfc.py
  GPUWM_WRF_EXACT=1 python tools/myjsfc_wrf461_oracle/libm_sweep.py'

# the CPU twin, no GPU
python tools/myjsfc_wrf461_oracle/compare_myjsfc.py --cpu-authority
```
