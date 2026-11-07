# ACM2 PBL oracle, WRF v4.6.1 (bl_pbl_physics = 7)

WRF's own numbers for the Asymmetric Convective Model version 2 (Pleim
2007): `ACMPBL` from `phys/module_bl_acm.F`, driven over independent columns
for several consecutive steps, every step's float32 inputs recorded beside
everything the scheme returned.

```
bash tools/acm2_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR> [<REFEREE_BUILD>]
```

* `WRF_SOURCE_ROOT/phys/module_bl_acm.F` must hash to
  `f18685f7ea091f567c3f37e983f26357dcd7e9ec2eea9a6e70635c506a66d60c` (the
  pristine v4.6.1 tree, d66e442); build.sh refuses any other file.
* The module goes through WRF's own phys/ preprocessing (sed, cpp with WRF's
  define set, `tools/standard.c`, cpp) and WRF's own serial gfortran compile
  line (`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize` plus
  FCBASEOPTS). That is the program the combo-sweep referee `wrf.exe` runs.
* With `REFEREE_BUILD` (a compiled WRF-4.6.1 tree) the script proves it: the
  preprocessed `.f90` must be byte-identical to the referee's, and a second
  driver linked against the referee's own `phys/module_bl_acm.o` must write
  byte-identical fixtures.
* An -O0 build runs as evidence only (`O0-report.txt`): at -O0 gfortran keeps
  `(1-ZF/PBL)**P` (`P = 2.0`) as a `powf(x, 2.0)` call instead of folding it
  to a multiply, so it is a different program from WRF's.

## What it writes

| file | contents |
| --- | --- |
| `acm2-l40.{in,out}.bin`, `.index.json` | 74 columns, 40 levels, DTPBL 60 s, six steps from itimestep 1 |
| `acm2-l60.{in,out}.bin`, `.index.json` | 74 columns, 60 levels, DTPBL 180 s, four steps from itimestep 41 |
| `fatal.in.bin`, `fatal.stderr.txt` | two columns; WRF stops on column 2 ("RIBX never exceeds RIC") |
| `libm-report.txt` | the five `powf` call sites in WRF's object, by subroutine |
| `referee.txt`, `toolchain.txt`, `oracle-sha256sums.txt` | receipts |

The column families (`make_cases.py`) are built to reach the branches:
convective day with and without a superadiabatic first layer (both sides of
`NOCONV`), shallow convection under `KLPBL <= 3`, stable nights with and
without a low-level jet, neutral columns with exactly zero fluxes (the `TSTV`
floor), sea, cloud- and ice-topped columns (EDDYX's moist Richardson
adjustment), weak wind (the `WSSQ` floor), strong shear, deep convective
layers, seeded draws, and first-layer thicknesses from 12 m to 65 m (the
30 m source-layer search ends at level 1, 2 or 3). `XTIME` is WRF's
`itimestep`, so step 1 of the 40-level fixture is the never-convective
first step.

`run_acm2.F90` documents the stream layouts; `woof/verify/acm2_oracle.py`
reads them. The fixtures of record live in `tests/data/oracles/acm2/`.

These are constructed inputs. They are not inputs captured from a real
native WRF forecast. Current word grading explicitly selects strict
numerics; default fast numerics uses CUDA float32 powers and permits
contraction in the column solver.
Neither these fixtures nor the earlier short idealized CLI run establish
multi-hour forecast, observation, nest, or ensemble qualification.

The public v4.7.1 module has the same SHA-256 as the v4.6.1 pin. Both
versions distribute the module under WRF's public-domain statement in
`LICENSE.txt`, copied in `licenses/LICENSE-WRF-public-domain.txt`. ACM2
uses no external lookup-table file. Its packaged scalar constants are
derived from that source and pinned in `woof/acm2_constants.json`.

`stub_wrf.F90` supplies the one external symbol the module references,
`wrf_error_fatal3`: it prints WRF's message and stops with status 3.

## Captures from a whole native forecast

`capture_native.py` instruments the pinned pristine module with sparse
stream writes before and after ACMPBL. It changes no arithmetic. Apply it
to an owned strict WRF source copy before building `wrf.exe`:

```sh
python tools/acm2_wrf461_oracle/capture_native.py WRF_SOURCE_ROOT --patch capture.patch
```

The native run writes input/output pairs at domain step indices 1, 301 and
601, at every seventeenth horizontal point in each tile. Tile bounds are
part of each filename. The stream header records the tile, level count,
step, constants and coordinate list. It records all six incoming and
returned tendencies, carried fields and both exchange arrays. The native
compile must use big-endian stream I/O, as the pinned build does. Do not
run two forecasts in the same output folder.

Replay each completed pair under strict numerics:

```sh
GPUWM_NUMERICS=strict GPUWM_WRF_EXACT=1 python tools/acm2_wrf461_oracle/replay_native.py CASE_DIR --receipt replay.json
```

The native tendency addition is reproduced on the device before comparing
every returned word, including integer KPBL and zero signs. Receipts bind
the complete stream bytes by SHA-256 and report per-field differences,
constants, sampled locations and the compiled kernel manifest. A missing
capture or any differing word fails the replay. Raw streams are disposable
after recording the receipt. The separate forecast receipt must retain
the executable, object, source, input and configuration identities.

The 2026-11-01 real capture receipt is `native-real-receipt.json`: 96,471
words, zero differences, 243 column-time instances at 49 levels. It records
all input/output stream hashes, constants, coordinates and per-field word
counts without retaining disposable native streams. Native WRF v4.6.1 used
`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`; the uninstrumented
module remains byte-pinned above. The real case used PBL7/sfclay1, mp8,
3 km, 12-second steps and an 18-21 UTC 25 May 2024 forecast window.
This replay proves the captured module calls. Nest feedback, ensemble
execution, PX pairing and the literal `woof go` WRF-input route are not
qualified by it.
