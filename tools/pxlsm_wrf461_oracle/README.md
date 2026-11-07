# Pleim-Xiu LSM oracle, WRF v4.6.1 and v4.7.1

The WRF reference for `sf_surface_physics = 7`.  It compiles the
byte-unmodified `phys/module_sf_pxlsm.F` and `phys/module_sf_pxlsm_data.F`
of the pinned tree and drives the `pxlsm` driver itself over a tile of
columns for several consecutive calls.

```
bash tools/pxlsm_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR>
```

`source-pin.json` pins the SHA-256 of each of the four compiled source
files and the upstream license. All five files are byte-identical in
WRF v4.6.1 (`d66e442fccc04111067e29274c9f9eaccc3cef28`) and v4.7.1
(`f52c197ed39d12e087d02c50f412d90d418f6186`). A Git checkout must be at
one of those revisions. An exported source snapshot is accepted only
with those exact file bytes, including `LICENSE.txt`. `build.sh` writes
`source-provenance.json` and refuses a changed source or license before
compiling. The files carry no additional license terms; the WRF public
domain notice in `licenses/LICENSE-WRF-public-domain.txt` applies.

These are analytic fixtures. They are not a real-input module capture,
a coupled forecast, a restart proof, or evidence of stock-namelist admission.
The separate native-capture and coupled qualification receipts below provide
that evidence; the engine admits the qualified PX paths.

## Real native-input capture

`instrument_native.py PINNED_MODULE OUTPUT.F` produces an instrumented
copy of the pinned PX source. Install that generated file in a private
copy of an existing strict native build, then rebuild that copy. Every
original source line stays unchanged. Set `WOOF_PXLSM_CAPTURE_DIR` to
an empty scratch directory before running native WRF with PX enabled.
Only calls 1, 61 and 121 write snapshots, one entry and one exit per call.
At a 60-second model step these cover initialization, hour one and hour
two. The capture rejects decomposed domains, multiple tiles, a soil
layer count other than two, or a soil-category count other than 16.

The snapshot files use `schema.py`'s order and explicitly little-endian
words, including when the native build uses `-fconvert=big-endian` for
its other I/O. No uninitialized output residue is masked. Run the GPU
comparison under the node's shared-device ownership protocol:

```sh
GPUWM_NUMERICS=strict GPUWM_WRF_EXACT=1 python3 \
  tools/pxlsm_wrf461_oracle/replay_native.py CAPTURE_DIR --report replay.json
```

The report keeps per-field word counts, differing-word counts, input and
output hashes, and the actual kernel manifest. It fails on any differing
word. Retain the native namelist, build flags, executable and input
hashes, completion log, instrumentation diff and JSON comparison;
delete raw captures and forecast outputs once those receipts exist.
This module replay does not qualify the coupled WOOF option, surface
FDDA, nests, restarts, speed or forecast maps.

## Coupled qualification

`woof/data/pxlsm/coupled-proof.json` records the native two-layer input
door, whole three-hour `woof go` forecast, Rust maps beside native WRF
and ASOS at hours 0–3, production-stage timing, histories, checkpoint
continuation, both nest feedback modes and ensemble members with soil
nudging off. It retains the strict 525,600-word replay above.

`woof/data/pxlsm/soil-nudging-proof.json` qualifies `pxlsm_soil_nudge=1`
with native auxinput9 OLD/NEW T2/Q2/SN records, bounded Rust reads and
per-domain/member CUDA clocks. A second whole native run supplied calls
1, 61 and 121 with soil nudging on: 525,600 physical output words and
78,840 streamed endpoint words match exactly, with no exclusions.
The corresponding whole GPU forecast and four-hour native/ASOS Rust maps
are recorded there, along with record-boundary restart, tree checkpoint,
feedback 0/1 and independent-member clock checks.

For that live-reader replay, add `--auxinput9 PATH --start-time ISO_TIME`
to the command above. Use `GPUWM_WRF_GLIBC=2.43` for the retained native
v4.6.1 referee. The qualified forecasts use default fast numerics;
strict arithmetic is the replay opt-in. Native WRF disables auxinput9
when every `grid_sfdda` is zero, so the retained stock soil-nudging case
enables `grid_fdda=1, grid_sfdda=1` with atmospheric and surface nudging
coefficients zero. The forcing perturbations are constructed from real
native initialization; archived ASOS supplies the independent observation
comparison. Resident fixed-step domains are the qualified runtime scope.

The retained `woof/data/pxlsm/native-proof` receipts record a completed
WRF 4.7.1 three-hour January 2000 forecast. Calls 1, 61 and 121 covered
4,380 columns each. The strict RTX 4090 comparison checked 525,600 words,
with zero differences and zero exclusions. Soil nudging was off in this
run. The raw captures and prepared inputs were deleted after recording
their hashes. These receipts are numerical evidence, not a bundled
real-input regression fixture or coupled-engine qualification.

## What it writes

* `cases/case-*.bin`: 13 call geometries from `gen_cases.py` (12 x 8
  columns each, 6 calls), covering every land-use table pxlsm has, both
  `pxlsm_smois_init` values, soil nudging on and off, MODIS vegetation on
  and off, one, two and three PX sub-steps, day and night, and every
  surface kind the scheme branches on.  The field order is `schema.py`.
* `out-asbuilt/out-*.bin`: the driver's state after every call.
* `pxlsm-tables.txt`: every land-use table word as the compiled module
  holds it (`dump_tables.F90`); `pack_tables.py` turns it into the packaged
  `woof/data/pxlsm/pxlsm_tables.json`.
* `literals.txt` and `pxk_table.cuh`: every constant the CUDA kernel reads,
  as gfortran makes it, including the compile-time folds
  (`constants.py`).  The kernel's `pxk_bits` table is this file.
* `o0-vs-asbuilt.txt`: the word-level difference between the reference and
  an `-O0` build of the same sources.

The fixtures under `woof/data/pxlsm/oracle/` are copies of these files.

## The reference is WRF as built

The objects are compiled with the scalar `wrf.exe`'s own flags
(`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`).  An `-O0`
build is NOT the same program on gfortran 13.3: `X**5` becomes a call to
libgcc's `__powisf2` at `-O0` and an inline addition chain at `-O2`, and
SURFPX's dry-layer thickness `(1.-WG/WSAT)**5` is such a power.  Measured
over the fixtures: 173 of 299,520 words differ between the two builds.  The
port follows the as-built program.

## One case pins a WRF defect

`modis21-lake-defect` covers two land columns with MODIS category 21
(inland lakes) only.  real.exe folds lake fractions into ISWATER before any
LSM sees them, so a WRF run never presents this; pxlsm given it anyway forms
`RSTMIN = SUMLAI/SUMRSI = 0/0` and the columns go NaN.  The port defines
RSTMIN there (woof/core/pxlsm.py `PXLSM_DIVERGENCES`), and the parity test
asserts both halves: NaN in WRF, finite in the port, every other column
bit-identical.
