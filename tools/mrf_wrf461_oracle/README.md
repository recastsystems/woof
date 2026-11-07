# MRF PBL oracle, WRF v4.6.1

Column oracle for `bl_pbl_physics = 99`, WRF's `mrfscheme`
(`Registry/Registry.EM_COMMON:3177`): the Hong-Pan (1996) nonlocal PBL in
`phys/module_bl_mrf.F` (`MRF` wrapper, `MRF2D` column, `TRIDI2`, `mrfinit`).

```
source /work/wrf/env.sh
PYTHON=/work/venv/bin/python bash tools/mrf_wrf461_oracle/build.sh \
    "$WRF_SOURCE_ROOT" <BUILD_DIR> [/work/wrf/build/scalar/WRF-4.6.1]
```

`module_bl_mrf.F` must hash to
`a1306d0063043d001fa4a53b9219031b931999c7d7daea172fd97518bc9b19eb` (v4.6.1,
d66e442); build.sh refuses anything else, in the source root and in the
compiled WRF tree alike.

## Three arms, one referee

| arm | object | role |
| --- | --- | --- |
| `referee` | `phys/module_bl_mrf.o` from WRF's own `compile em_real` (scalar build: `-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`) | the fixture |
| `rebuild` | the pinned source compiled here at the FCOPTIM read from that tree's `configure.wrf` | must write the referee's bytes, or build.sh fails |
| `o0` | the pinned source at `-O0` | diagnostic only, counted in the receipt |

The `-O0` arm is not WRF as built: gfortran leaves `ZFAC**PFAC`
(`PFAC = 2.0`, `module_bl_mrf.F:978`) as a fifth `powf` call there and folds
it to `ZFAC*ZFAC` at `-O1` and above, and `powf(x, 2.0)` is not always the
correctly rounded square.  On the packaged cases 86 of 196,520 output words
differ for that reason alone (`build-receipt.txt`).  The port follows WRF as
built: the four remaining real-exponent powers call WOOF's own float32
`gfk_pow`, `ZFAC**PFAC` and every integer power are multiplies.

## The driver

`run_bl_mrf.F90` calls WRF's 3-D `MRF` entry once per column per step with a
single-column tile (`its = ite`), the memory layout of `module_pbl_driver.F`
(`kms:kme = 1:nz+1`, `kts:kte = 1:nz`, as `module_first_rk_step_part1.F`
passes `kte = min(k_end, kde-1)`), and WRF's constants spelled as
`share/module_model_constants.F` spells them.  Tendency arrays are filled
with a sentinel before every call; the packer refuses a run in which WRF
wrote memory level `kme`, or wrote `RQIBLTEN` with `FLAG_QI` false.

Between steps the driver advances its own state with the tendencies it got
(explicit float32, no contraction); the fixture records every step's inputs
as well as its outputs, so a consumer never has to reproduce that update.

## Cases (`gen_mrf_cases.py`, 73 columns x 4 steps x 40 levels)

convective land (12), stable night (10), neutral windy (6), cloud-topped
with liquid and ice and `FLAG_QI` both ways (10), very shallow (6), PBL
filling a short column so the thermal reaches `K = 1` (6), water (10), and
probes: calm lowest level (`WSPD1 = 1e-9`), zero shear (`SS = 1e-9`), the
land `UST` floor, the `GAMCRT`/`XKZMAX` clamps, the `PRMAX` clamp, dt = 10 s
and 180 s, `qc` exactly at the `0.01E-3` in-cloud test and one float32 step
above it, and `BR` at `+0`, `-0`, `+/-1.4e-45` and the smallest normal on the
`BR .GT. 0.0` compare.

## What the fixture shows about WRF

* `KPBL2D` is `KPBL0` (`:1284`), MRF's TOP-DOWN index: stable columns
  report `kte`, the model top, in a field every other PBL fills bottom-up.
  The port publishes the same level bottom-up and declares it.
* `PBL` is `PBL0` (`:1285`) only where `PBLFLG` survives; a stable column,
  and a column whose thermal reaches the top level, report the first-layer
  depth `ZQ(KL)`.
* `REGIME` and `ZOL` are never written (`:685-760` is commented out); they
  pass through as the surface layer wrote them.


## Real native forecast capture

`capture_native.py SOURCE OUTPUT` verifies the same module digest and adds
read-only stream writes before and after the MRF wrapper's physics body.
It records the center column's first four calls, including every consumed
input, all six tendencies, UST, HOL, PBL and native KPBL. The physics
expressions remain byte-for-byte unchanged. Compile this generated source
in a private strict native WRF build with the flags above, replace only
its MRF object, relink, and run a real-data forecast with PBL 99 and either
MM5 surface layer. The stream is `mrf-real-capture.bin` in the run folder.
Only the rank owning the center column writes it.

The retained `woof/data/mrf/oracle/native-real-capture.json` contains the
raw float32 words from four calls during a native WRF v4.7.1 real forecast
on 2026-10-16 at 09 UTC, a 300 by 300 mass grid at 3 km with 49 levels and
a 15 s step. The module itself is byte-identical to v4.6.1. Source, build,
namelist and input digests are in the fixture. This fixture is distinct
from the 73 synthetic columns above. Run its proof with:

```
GPUWM_WRF_EXACT=1 python -m woof.verify.mrf_real_capture replay.json
```

The measured replay on an RTX 5090 compared 1,192 defined output words,
with zero differing words. The launcher returns the unconverted native
KPBL integer separately from the engine's bottom-up diagnostic, so the
proof compares the native integer directly. No native-undefined residue
was needed on these four calls.
A module replay establishes module parity; it does not qualify a matched
whole forecast or a comparison with observations.
