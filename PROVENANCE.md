# woof oracle provenance and deviation register

The historical WRF authority used by the evidence in this register is WRF v4.6.1
(upstream <https://github.com/wrf-model/WRF.git>, tag `v4.6.1`, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`). A verified checkout of that
tree is the source-reading and line-citation authority; its machine-local
staging path is deliberately not part of the contract. The optional real74
case bundle is selected with `WOOF_REAL74_REFERENCE_BUNDLE` in
`woof/verify/cases/real74_d01.py`. The WRF source tree is **not**, by itself,
the binding numerical oracle. Scheme-data provenance and scheme-local
deviations live next to their data (`woof/data/*/PROVENANCE.md`); this
file is the repo-level register for cross-module numerical authority,
arithmetic policy, deliberate deviations, and fail-loud scope cuts.

## Native GFS direct-WRF adapter evidence (2026-07-20)

- **Source**: NOAA/NCEP GFS `pgrb2.0p25`, cycle 2026-07-20 00Z, f000 and
  f003, fetched from the NOMADS `filter_gfs_0p25.pl` service for
  20..55 N and 110..55 W.  Input SHA-256 values are
  `24bf5b5b302962ed66da502e67a91b62120714a76ee4689c648c91539fb688e9`
  (f000) and
  `9ada4e2f8e086773417236781b2211d9a6007f4350dc7ffb8eed0104d50b0f40`
  (f003).  The complete input-manifest SHA-256 is
  `8d4f66f25f181f86611581d057115a1f70730f245f8aa8c777210431ae7eb548`.
- **Mapping authority**: WRF v4.6.1 `WPS/ungrib/Variable_Tables/Vtable.GFS`
  for GFS field semantics, plus `share/module_soil_pre.F` for Noah soil
  initialization.  The adapter uses GFS RH as WPS supplies it, declares
  initial hydrometeors zero (the stock GFS Vtable has none), copies the four
  exact GFS/Noah soil slabs without ERA5-style interpolation, and retains
  bitmap missingness through land-aware horizontal selection.
- **Decoder gate**: `tools/grib1_bridge/src/bin/gfs_grib2_bridge.rs` uses raw
  discipline/category/parameter codes, requires all five atmosphere fields
  at each of 21 pressure levels from 1000 through 100 hPa, and validates both
  fixed surfaces for every soil layer.  The certified slice permits only GRIB2
  simple packing (DRT 5.0), binds process identifier 81 at f000 and 96 for
  forecasts, and rejects wrong cycle/time/grid, duplicates, gaps, nonuniform
  cadence, stale or malformed bitmaps, nonfinite required values, and
  out-of-contract ranges before atomic publication.  Source files are decoded
  once and the proof binds source, inventory, decoded-product, implementation,
  git commit, and git tree hashes.
- **Horizontal-policy scope**: atmospheric fields use the native interpolation
  contract, while masked surface and soil fields use land-aware nearest donors
  and a globally proven nearest-water donor for model lake cells.  This is not
  a claim of numerical equivalence to WPS METGRID's four- and sixteen-point
  masked interpolation. The WRF acceptance gate below used the instrumented
  oracle build. It checks file structure, acceptance and a stable smoke step
  for this exact slice, not WPS bitwise or numerical parity.
- **Native runtime**: clean git commit
  `528e9d795da1090b239ff29367f463c52866e422` (tree
  `752ecac00c6964c8573e4d89935e6bfb394765ec`) completed decode,
  initialization, prepared-cache publication, and direct export in
  21.559 seconds internally (22.57 seconds process wall time).
- **WRF acceptance gate**: native initialization/direct export produced
  `wrfinput_d01` SHA-256
  `064b216010c0ed43ad06ac197b82cd3dbfe884890251e0ebac30a4405fe387f3`
  and `wrfbdy_d01` SHA-256
  `c74c5b9636f0e5a5cb54dbd9366d86a221ecbeb9a203fbbd6a3c0d39adc923f8`.
  The WRF v4.6.1 executable with SHA-256
  `f0fb585bf37b72fbdcece562047934cb8386db3958f153d6e4e6876e5fd997ac`
  accepted both, advanced to 2026-07-20 00:00:05, printed success, and wrote
  a finite history file.  Acceptance evidence SHA-256:
  `b8fca75fc263b23699b17ccadaf4a122199d92f460c83fbae3a8def702c05cc3`.
  This register identifies that executable below as the instrumented oracle
  built with an audited write-out-only patch. It is not an unchanged stock
  build. Claiming stock acceptance for these exact files would require a
  separate run with the clean tag build; none is claimed by this receipt.

## Binding numerical-oracle manifest

The binding numerical oracle is the following pinned instrumented build,
not an arbitrary executable produced from the WRF v4.6.1 tree:

- **Source/build recipe**: the source tree above, copied read-only into the
  disposable WSL build tree and built by
  `.superpowers/sdd/codex/wrf-build-spike-report.md`.  The environment is
  WSL `Ubuntu-24.04` (recorded as Ubuntu 24.04.4 LTS), GNU Fortran/GCC
  13.3.0, netCDF-Fortran 4.5.4 (netCDF-C 4.9.2), WRF's serial/STUBMPI
  execution (no MPI processes), no OpenMP (the generated OMP flags are
  commented out), one tile, and the `em_real` target.  Configure used the
  displayed GNU serial option 32 plus basic-nesting option 1 with
  `NETCDF=/usr` and `WRFIO_NCD_LARGE_FILE_SUPPORT=1`; generated
  `configure.wrf` supplied `-fallow-argument-mismatch` and
  `-fallow-invalid-boz`.  There were **no hand edits** to `configure.wrf`.
  The recipe's mandatory prerequisite normalized CRLF endings in the
  disposable copy (4,838 text files), never in the read-only bundle.
- **Effective oracle case**: `tools/wrf_instrumented/namelist.input.n1p5`,
  the d01+d02 one-outer-step case with Morrison (`mp_physics=10`), YSU
  (`bl_pbl_physics=1`), and revised-MM5 surface layer
  (`sf_sfclay_physics=91`) as the matched-physics scheme selection.  The
  case itself explicitly sets `use_theta_m = 0` in `&dynamics` and
  `nwp_diagnostics = 0` in `&time_control`.  The first setting is
  essential: WRF's default is `use_theta_m=1`
  (`Registry.EM_COMMON:2860`), under which the dumped `t` table family is the
  moist-theta `thm` state (`Registry.EM_COMMON:211`), differing from woof's
  dry-theta consumers by approximately the `(1 + 1.61*qv)` field factor.  The
  second
  setting keeps the case inside D2's output-due diagnostic scope.
  `tools/wrf_instrumented/instrument-med-force.patch` is the exact
  instrumentation applied after the pristine source-file SHA check; it
  captures the output boundary tables, the bracketing `PROG` samples,
  the solve-side `DBDY` clock records, and (p5n15 extension) the 25
  complete pre-coupling `INPT` input arrays per domain that make an
  independent candidate producer possible.  The p5n15 review audited
  every hunk as insertion-only/write-out-only: zero changes to any
  computed value, loop bound, or call ordering in the integration.
- **Reference dump status**: the binding pin lives in ONE place,
  `tools/wrf_instrumented/n1p5-dump.sha256`, currently
  `e4a930daac3f7295eef9e6d9e31a885d6e776a713d128c8ccb8b0a115805e060`
  (844,428,012 bytes; schema records: legacy tables + 4 `PROG`
  prognostic-diagnostic samples + 50 full pre-coupling `INPT` input
  arrays + 4 `DBDY` boundary-clock samples).  Produced by the p5n15
  worker lane and BYTE-REPRODUCED by the controller's independent WSL
  regeneration from pristine sources + the committed patch (identical
  SHA-256); the p5n15 full-scope review additionally proved the patch
  insertion-only/write-out-only hunk-by-hunk.  Superseded lineage,
  retained for provenance only: (1)
  `049ece3d96bb912d094304e2616b2e7021f9ea0ed815da280d95eaead360ebd3`,
  ran `nwp_diagnostics=1` and omitted `use_theta_m` (Registry default 1),
  so its `t` tables are moist-theta state, not binding for woof's
  dry-theta consumers; (2)
  `441da3cba30fd1fb958488ba55c7e3aa0e5b11854a92666a3ba73f020f8022cc`,
  corrected case but diagnostics-only schema (12 scalar `PROG` samples
  per domain), which cannot drive an independent nest-force candidate
  producer; any producer built on it would be tautological against the
  dumped output tables.  The instrumented executables (controller
  oracle build tree, outside this repository) have SHA-256
  `f0fb585bf37b72fbdcece562047934cb8386db3958f153d6e4e6876e5fd997ac`
  (`wrf.exe`) and
  `4b47e80e55009144410b5530e0bbbcd1a5426b427716301a078a39af7b6f4622`
  (`real.exe`), rebuilt from pristine + patch by the controller's
  verification run (the retired diagnostics-era `wrf.exe` was
  `e4ac817bd0dc1bb5dde85c76bd24f7a9674613c014e2452a94ecfde255bb9932`).

After the controller re-pins the regenerated reference dump,
numerical-fidelity comparisons are against dumps made by **this build** and
case and apply only the tolerances registered in the gate ledger
(`woof/verify/nest_gates.py`; N1.5 currently registers the FP32-relative
`1.0e-6` table band).  A “bitwise” statement is always scoped to its named
comparison, for example a kernel/mirror seam, restart identity, or a
controlled A/B refactor/feature-inertness comparison.  It is never a claim
against arbitrary WRF builds.  In particular, no gpuwm-versus-WRF output
file byte-identity claim is made; file-format, metadata, and I/O-library
bytes are outside the numerical oracle.

## Arithmetic contract

- **Compilation**: the COMMON CuPy `RawModule` loader
  (`woof/core/kernels/__init__.py`) passes only `-std=c++17`; it never
  enables `--use_fast_math` and leaves FMA contraction at NVRTC's default
  `--fmad=true`, so contraction is permitted.  CPU-mirror comparisons for
  those COMMON-loader kernels therefore sit behind registered tolerance
  seams unless explicit `__*_rn` intrinsics pin operation boundaries.  Only
  the dedicated nest/coupler loader in `woof/core/nest_interp.py` passes
  `-fmad=false`, so SINT/TR4, boundary interpolation, terrain adjustment,
  and dormant feedback arithmetic do not contract multiply/add pairs.
<!-- BEGIN GENERATED ftz-statement: provenance-compile-policy (tools/ftz_receipt/render_statement.py) -->
- **Rounding and subnormals**: ordinary FP32/FP64 operations and conversions
  use round-to-nearest-even.  FP32 subnormal handling is not one policy: it
  differs between the compile routes this codebase uses, so it is recorded
  per route, as measured on NVIDIA GeForce RTX 5090 (compute capability
  12.0, driver 13.3 (13030), NVRTC 13.3,
  CuPy 14.0.1) by `tools/ftz_receipt/probe.py` over
  6 arithmetic mechanisms:
  - `R1` loader RawModule (`woof/core/kernels/__init__.py:84`,
    woof.core.kernels.load_module), effective NVRTC options `-std=c++17`
    `-ftz=true`: `flush-to-zero` on 6 of 6 mechanisms [disassembly
    `tools/ftz_receipt/receipt/sass/r1.sass`]
  - `R1-ftztrue` loader RawModule + explicit --ftz=true (control)
    (`woof/core/kernels/__init__.py:84 + control flag`, cupy.RawModule),
    effective NVRTC options `-std=c++17` `--ftz=true` `-ftz=true`:
    `flush-to-zero` on 6 of 6 mechanisms [disassembly
    `tools/ftz_receipt/receipt/sass/r1_ftztrue.sass`]
  - `R2` direct NVRTC with the shortwave option tuple
    (`woof/core/rrtmg_sw.py:2906`, cupy.cuda.compiler.compile_using_nvrtc),
    effective NVRTC options `-std=c++17` `--ftz=false`: `ieee-agreement` on
    6 of 6 mechanisms [disassembly `tools/ftz_receipt/receipt/sass/r2.sass`]
  - `R3` direct NVRTC + cuda.function.Module (`woof/core/rrtmg_lw.py:3733`,
    cupy.cuda.compiler.compile_using_nvrtc), effective NVRTC options
    `-std=c++17` `--ftz=false`: `ieee-agreement` on 6 of 6 mechanisms
    [disassembly `tools/ftz_receipt/receipt/sass/r3.sass`]
  - `R4` CuPy-generated ReductionKernel (`woof/core/mynn_pbl_gpu.py:296`,
    cupy.ReductionKernel), effective NVRTC options `--std=c++17`
    `-ftz=true`: `flush-to-zero` on 6 of 6 mechanisms [disassembly of 6
    objects, `tools/ftz_receipt/receipt/sass/r4_0.sass` and siblings]
  - `R5` inline PTX without .ftz (`woof/core/kernels/__init__.py:84`,
    woof.core.kernels.load_module), effective NVRTC options `-std=c++17`
    `-ftz=true` (this route rides `R1`'s compile): `ieee-agreement` on 4 of
    6 mechanisms; `not-applicable` on 2 of 6 mechanisms [disassembly
    `tools/ftz_receipt/receipt/sass/r1.sass`, the object `R1` compiled]
  The option tuple each row names is the tuple NVRTC received, captured by
  wrapping the compiler entry point rather than re-deriving it: CuPy appends
  `-ftz=true` to whatever the caller passed, at
  `cupy.cuda.compiler` line 585 (`options += ('-ftz=true',)`), after the
  caller's options, and NVRTC honours the last occurrence.
  The inventory records 13 distinct caller-supplied option tuples across the
  209 compile sites in the shipped package, each listed here with a site
  that supplies it:
  - no caller options -- `woof/core/attribute_tracking.py:99`
    (xp.ElementwiseKernel), and 157 other site(s)
  - `-std=c++17` -- `woof/core/acoustic_fusion.py:403` (cp.RawModule), and
    13 other site(s)
  - `--std=c++17` -- `woof/ensemble/batch_perturbation.py:51`
    (xp.RawModule), and 11 other site(s)
  - `--std=c++11` -- `woof/hex/cuda_courant_fullstep_lts.py:52`
    (cp.RawKernel), and 5 other site(s)
  - `-std=c++17` `--ftz=false` -- `woof/core/chem_plumerise_cache.py:37`
    (compiler.compile_using_nvrtc), and 5 other site(s)
  - `-fmad=false` -- `woof/core/dycore.py:306` (cp.ElementwiseKernel), and 2
    other site(s)
  - `--fmad=false` -- `woof/hex/moving_vertical_cuda.py:21` (cp.RawKernel),
    and 1 other site(s)
  - `-std=c++17` `--fmad=false` -- `woof/core/kernels/__init__.py:689`
    (cp.RawModule), and 1 other site(s)
  - `-std=c++17` `--fmad=false` `--ftz=false` --
    `woof/core/thompson_fork.py:109` (compiler.compile_using_nvrtc), and 1
    other site(s)
  - `--fmad=false` `-std=c++14` -- `woof/da/letkf_device.py:363`
    (cp.RawModule)
  - `--std=c++17` a target flag built from the device architecture at run
    time -- `woof/ocean/waves/model.py:73` (cp.RawModule)
  - `-std=c++17` `--ftz=false` `--fmad=false` --
    `woof/core/kernels/__init__.py:805` (compiler.compile_using_nvrtc)
  - a target flag built from the device architecture at run time a target
    flag built from the device architecture at run time --
    `woof/hex/mpasphys/surface_update_v842.py:148`
    (compiler.compile_using_nvrtc)
  `R5` and `R1` are kernels inside ONE compiled object -- same device, same
  flags, one compile -- and they did not measure alike, so on this device
  the outcome follows the instruction the compiler emitted rather than the
  hardware alone.
  Evidence: the device bit table `tools/ftz_receipt/receipt/bitpatterns.csv` (both
  passes byte-identical: true), the objects the routes themselves compiled,
  `tools/ftz_receipt/receipt/cubin/`, and their disassembly,
  `tools/ftz_receipt/receipt/sass/` (Cuda compilation tools, release 13.0,
  V13.0.39), all recorded in `tools/ftz_receipt/receipt/receipt.json`.
<!-- END GENERATED ftz-statement: provenance-compile-policy -->
- **Operation boundaries**: explicit parity-critical kernels use
  `__fadd_rn`/`__fmul_rn`/related intrinsics where an operation boundary
  itself is part of the seam.  NVRTC's documented option defaults are
  `--ftz=false`, `--prec-div=true`, and `--prec-sqrt=true` (see the NVIDIA
  [NVRTC compile-option reference](https://docs.nvidia.com/cuda/nvrtc/index.html#supported-compile-options));
  what the compiler defaults to and what a given route ends up asking for
  are separate questions, and the measured answer to the second is the
  generated block above.
- **Reduction order**: reductions that mirror serial WRF loops are serial
  within each CUDA work item and retain the WRF iteration order.  One
  checkable example is the two `n=0..49` wet-snow/wet-graupel sums in
  `woof/core/kernels/refl.cu`, which accumulate the WRF 50-bin terms in
  order rather than using a parallel reduction tree.
- **Transcendentals**: device `exp`/`log`/`pow` results are **not** claimed
  bit-equal to gfortran's host math library.  Every consumer that matters
  to a fidelity claim terminates at a registered tolerance seam.  D2's
  density-offset pin is the local pattern:
  `tests/test_refl.py::test_prepared_pressure_seam_has_predicted_dry_dbz_offset`
  uses an absolute `5e-8` tolerance rather than a libm bit-identity claim.
- **Classic Thompson coefficient tables**: table-dependent native CUDA
  kernels consume the validated `ClassicTableSet.to_device` result.  In
  particular, rain-cloud accretion requires canonical `t_Efrw(100,100)`, and
  cloud-ice-to-snow autoconversion requires canonical `tps_iaus(64,55)` and
  `tni_iaus(64,55)`.  All are complete FP64, Fortran-contiguous WRF arrays.
  The production path has no WRF runtime dependency; WRF v4.6.1 is used only
  to generate and hash-pin the canonical external table asset and direct
  column fixtures.  The focused `warm-accrete` and `ice-auto` gates sparsify
  only their known-active table entries after verifying them against that
  asset, so they do not define alternate production tables.
- **Classic Thompson rain-evaporation ordering**: the ordinary
  Srivastava-Coen process updates temperature and vapor before sedimentation,
  while WRF retains the pre-evaporation volumetric rain mass and number for
  that same step's fallout.  The staged CUDA composition records the original
  FP32 density into an explicit scratch field and supplies it to the admitted
  rain-sedimentation kernel; fallout velocities still use the updated
  environmental density.  This is a numerical-ordering seam, not an external
  WRF runtime dependency.

## Operational WRF 3.9 fork Thompson tables (release assets, 2.8.7)

`thompson_version = "wrf_39_noaa"` reads its own four lookup tables
(`woof/core/thompson_contract.py`, `FORK_TABLE_ASSETS`, table set
`wrf-3.9-noaa-fork-thompson-gfortran15-v1`).  From 2.8.7 every release
carries them as assets named `thompson-wrf39-noaa-<table>`
(`tools/release/cut/fork_tables.py`), and `woof.thompson_fork_assets`
downloads them from the engine's own release, then from v2.8.7, verifying
each file's size and SHA-256 against those pins before installing it into
`~/.woof/tables/thompson-wrf39-noaa` (or `WOOF_THOMPSON_FORK_TABLE_ROOT`).

- **Source**: the HRRR (NOAA GSL) fork of WRF's `module_mp_thompson.F`,
  NOAA-EMC/HRRR tag `v4.1.21` (commit
  `a131d6c7f6ebf57ae3c43f198229ad6afdf4d886`),
  `sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_mp_thompson.F` sha256
  `4d60011188443eb432294f7693beb64bdbc8f812541a15c7800013060c877283` and
  `module_mp_radar.F` sha256
  `08329c87604b234efab53f7986163a4f6050f58823da9099cb464163c1920f08`, with
  the v4.6.1 `CCN_ACTIVATE.BIN` (sha256
  `f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd`).
- **Licence**: the fork's `WRFV3.9/README` (blob
  `cd40a3a3b28a15d5bb52f1d9b6f2bb25505a55d0` at that tag) opens with the
  WRF Public Domain Notice, word for word the text of
  `licenses/LICENSE-WRF-public-domain.txt`; neither source file has a
  copyright line, a licence clause or a use restriction, and the HRRR
  repository has no licence file of its own that says otherwise.  The
  tables ship under the WRF notice (NOTICE).
- **Generator**: the fork's own `thompson_init`, compiled unmodified, called
  by `tools/thompson_fork_oracle/generate_tables.F90` (sha256
  `65a0925b31003ae24135e1a0c69c48ccc08e03b0195c6a3c3282c3a5d0c03f77`) and
  built by `tools/thompson_fork_oracle/build.sh` (sha256
  `e454ac55c752c0f927fa457319a2364d1568bed8af9bff98650da696020558e3`) with
  `stub_wrf.F90` (sha256
  `e377d142027d294cb5d259f02c2c04ef68d698ff98816ba04de263af003396ba`);
  harness commit `43d87c73cf72554044108c74bb647d79a415916f`.  GNU Fortran
  15.2.0 (Ubuntu 15.2.0-16ubuntu1), glibc 2.43-2ubuntu2.4, Linux x86_64,
  `-O2 -fno-tree-vectorize`, no libmvec symbols (the build refuses a
  binary that links libmvec, which changes the table bits).
- **Tables** (bytes, sha256):
  `qr_acr_qg.dat` 51,518,256
  `8b9b5039ec560748c6261cc9ecbc4366332473a9a1d0666ee5a7f04686abf402`;
  `qr_acr_qs.dat` 33,118,944
  `39b4fd363d440a867cc8c37a22ebf2651c8ffcf54f8c05ab7a6f5e753e630ac0`;
  `freezeH2O.dat` 254,944,848
  `28c12192251969cd79a02e08e3174f26113de51490689f728cfc44af9cf76b2c`;
  `thompson_aux_tables.dat` 6,164,536
  `13d304911e5322fb5bcc2a0639effc20ad5381c59f076ced2386b56f2fb441ae`.
- **Reproduction**: a development machine's receipt
  (`tools/thompson_fork_oracle/PROVENANCE-node2.txt`) recorded these four
  digests; on 2026-10-06 two independent `build.sh` runs on a development machine (same
  compiler and libc, kernel 7.0.0-31) from a fresh download of the two
  fork sources produced byte-identical tables with the same four digests.
  The generator binaries' own digests differ from a development machine's receipt; the
  tables do not.  `woof.thompson_fork_build` repeats this
  build from the same pinned sources when no release answers, and its
  output must equal the same pins or it is refused.

## Operational HRRR v4 CONUS static file (release asset, 2.8.7)

The static-source row `hrrr-conus-v4`
(`woof/data/static_sources/static-sources.v1.toml`), which the HRRR
adapters select by default, pins NCO's operational HRRR v4 CONUS geogrid
output.  NCO serves it from a versioned folder that moves with every HRRR
code bump (the v4.1.19 path already returns 404), so 2.8.7 publishes the
same file, unmodified, as the release asset
`hrrr-conus-v4-hrrr_geo_em.d01.nc` (`tools/release/cut/static_mirror.py`),
and the row names
https://github.com/recastsystems/woof/releases/download/v2.8.7/hrrr-conus-v4-hrrr_geo_em.d01.nc
as its mirror.  The fetch tries NCO first, then the mirror, and installs
bytes from either only when they match the row's size and SHA-256.  Only
the 2.8.7 release carries the file; later releases point at it there.

- **Source**: NOAA/NCEP NCO production tree,
  https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/hrrr.v4.1.21/fix/conus/hrrr_geo_em.d01.nc
  (served with `Last-Modified: Tue, 15 Jul 2025 17:56:18 GMT`).
- **Licence**: operational HRRR v4 static data made by NOAA/NCEP and NOAA
  GSL, a work of the United States Government and in the public domain
  (NOTICE).  Redistributed unmodified.
- **File**: `hrrr_geo_em.d01.nc`, 1,112,724,420 bytes, sha256
  `a785b53fdb8493e376b88ece6e166ddd8046588dd1c0062181141bf2b012f6e2`.
- **How it was made**: as the row's `provenance` key records it: GEOGRID
  V3.9.1 output with HRRR's own land use, soil, vegetation and terrain
  choices and GSL's drag fields grafted in.
- **Acquisition for the 2.8.7 cut**: downloaded on 2026-10-07 from the NCO
  URL above on a Linux node, its size and SHA-256 checked against the pin
  before the cut staged it.

## Deliberate runtime deviations from WRF v4.6.1

### D1. Retired historical-reference compatibility-mode microphysics/h_diabatic cadence

- **WRF-native cadence**: the moist-physics prep/driver/finish block runs
  once per model step with the model-step `dtm` (solve_em.F:3604-3616
  description; prep call :3661-3672, driver :3689+, finish :4082-4094),
  and the retained heating `h_diabatic = mpten/dtm`
  (module_big_step_utilities_em.F:5745) is consumed by the NEXT model
  step's RK tendencies (rk_addtend_dry, module_em.F:1076-1080), a
  one-model-step lag.
- **Thermodynamic scope**: woof's `h_diabatic` transcription is the
  non-moist-theta (`use_theta_m=0`) branch in
  `woof/core/microphysics.py`.  The moist-theta branch changes the stored
  thermodynamic variable and the heating conversion
  (module_big_step_utilities_em.F:5505/:5735); it is not implemented.
  `woof/namelist_import.py` therefore rejects `use_theta_m=1` loudly,
  including WRF's Registry default of 1 when the key is omitted.
- **Retired woof compatibility mode**: before the native-dt ratification,
  the real74 compatibility integrator
  (`woof/verify/cases/real74_d01.py` `phase3_integration_config`,
  :161-163) replaced `dt = 60 s` with `DYNAMICS_SUBSTEPS = 8` uniform
  internal dynamics steps of `dt = 7.5 s` per 60 s WRF model-clock
  interval (`clock_dt = 60`; inner loop :1044-1048).  In that mode,
  `dycore.step` ran the full microphysics bracket once per INTERNAL step
  (woof/core/dycore.py, post-RK3 microphysics slot), so the heating was
  captured as `mpten/7.5`, the clamp was `mp_tend_lim*7.5` per call, and
  the lag was one 7.5 s internal step, not one 60 s clock step.
- **Why this is self-consistent (and not the W0AVG defect class)**: no
  WRF-clock-referenced constant enters the mechanism.  The capture dt and
  the apply window are the SAME internal step everywhere: each internal
  step's RK loop integrates the previous internal step's rate, removes
  exactly what it added (small_step_finish removal,
  module_small_step_em.F:416-426 mirrored at
  `dycore._finish_small_steps`), and applies the fresh microphysics
  increment once.  Conservation and the no-double-count invariant hold
  per internal step by the same algebra as WRF's per-model-step form.
- **Current enforcement**: the frozen real74 profile pins
  `DYNAMICS_SUBSTEPS == 1`; `phase3_integration_config` rejects any other
  value.  The experiment schema exposes no dynamics-substeps knob and no
  other `woof` package code reads that frozen-profile constant.  At the pin,
  `cfg.dt = clock_dt = 60`, so the internal step is the model step and the
  capture/apply/lag cadence is WRF-native.  The constant and no-reader
  invariant are pinned in `tests/test_real74_d01.py`.
- **Unsupported configuration**: any path with `DYNAMICS_SUBSTEPS != 1`
  is rejected.  The historical behavior was conservation-consistent but
  not WRF-equivalent.

### D2. REFL_10CM is computed on output-due microphysics steps only

- **WRF-native seam**: `refl_10cm` is computed INSIDE the microphysics
  driver call on history steps (`diagflag .and. do_radar_ref == 1`,
  phys/module_microphysics_driver.F `refl_10cm` argument; Morrison wrapper
  phys/module_mp_morr_two_moment.F:911-918).  The wrapper passes the
  scheme's post-call `t1d` and moments with the unchanged prepared `p1d`
  (:730, :780, :913-914), before `moist_physics_finish_em` applies its
  theta-increment clamp (module_big_step_utilities_em.F:5706-5707) or a
  later EOS refresh/boundary overwrite.
- **`nwp_diagnostics` scope**: WRF overrides the history cadence and sets
  `diag_flag = .true.` on every step when `nwp_diagnostics == 1`
  (dyn_em/solve_em.F:369).  woof's output-due seam matches the WRF cadence
  only under `nwp_diagnostics == 0`; `woof/namelist_import.py` rejects any
  nonzero value loudly instead of silently dropping it.
  *Amendment (STEP17, 2026-07-29):* `nwp_diagnostics = 1` now imports and
  enables the UP_HELI_MAX member of WRF's nwp_output family
  (`woof/core/uh_diag.py`, oracle-pinned to dyn_em/module_diffusion_em.F
  `cal_helicity`).  The REFL_10CM seam above is UNCHANGED and remains
  correct under either value: WRF's forced every-step `diag_flag` only
  computes instantaneous diagnostics WRF then discards between frames --
  the emitted history-time REFL_10CM is the same field either way.  The
  unimplemented family members (WSPD10MAX, W_UP_MAX/W_DN_MAX, W_MEAN,
  GRPL_MAX, HAIL_MAX*) stay absent from wrfouts.
- **woof arrangement**: the real74 output calendar marks only the final
  model step before each scheduled frame.  `dycore.step` threads that flag
  through `microphysics.apply`; the Morrison adapter computes from its
  post-call temperature/moments plus the pre-writeback prepared pressure,
  and the physics driver holds the FP32 `(nz,ny,nx)` result until
  `_write_case_output` consumes it exactly once.  A scheduled frame with no
  stash raises loudly.  The Kessler fallback uses the same timing.
- **Retired output-time seam**: evaluating after the dycore's
  post-microphysics EOS refresh changed the dry-branch density by
  `rho_new/rho_WRF = theta1*(1+(Rv/Rd)*qv1) /
  (theta0*(1+(Rv/Rd)*qv0))` at unchanged specific volume, hence changed
  dry reflectivity by exactly `10*log10(rho_new/rho_WRF)` before the floor.
  Representative latent-heating changes are 0.03-0.06 dB, already far
  above the mirror tolerance; near 273.15 K the reconstructed temperature
  could also flip the strict melting scan
  (module_mp_morr_two_moment.F:4586-4599) and its Blahak replacement.  That
  density-ratio analysis is why output-time evaluation was retired.
- **Scope (a), prognostic trajectory, inert**: woof launches the
  reflectivity kernel only on output-due steps and does not maintain an
  always-current field between frames.  `REFL_10CM` has no reader back into
  dynamics, physics tendencies/state updates, nesting/coupling, or the
  restart-resume path; the only consumer API call outside its defining
  module is runtime output assembly.  The repository-wide no-reader pin is
  `tests/test_refl.py::test_refl_stash_has_no_trajectory_or_restart_reader`.
- **Scope (b), scientific products, real differences**: two omissions are
  intentional and observable: (1) the cold-start frame omits `REFL_10CM`
  because no microphysics call precedes it; and (2) resume does not rewrite
  the restart-boundary frame, so that boundary also has no rebuilt
  reflectivity product.  The stash is transient driver-rebuilt state in
  `woof/io/restart.py`; normal loop order consumes output before a
  same-step restart write, and the next scheduled due step after resume
  rebuilds it.  Serializing an already-consumed/stale product is not part of
  trajectory identity, but the missing product frames remain product-level
  differences.
- **Scope (c), bytewise output, not claimed**: woof does not claim that a
  `wrfout` carrying this diagnostic is byte-identical to a WRF output file.
  Comparisons are field-level under the registered seam/tolerance, not
  NetCDF byte comparisons.
- **Input-domain adjudication (review F2)**: WRF activates each species only
  by mass (`q > 1e-9`) and then immediately forms the slope from its number
  moment (module_mp_morr_two_moment.F:4544-4584); it has no zero/negative/
  non-finite number guard, so such inputs do not have a defined
  meteorological result.  The production scheme clips number moments
  nonnegative and reconstructs active moments from bounded slopes
  (:1528-1635).  woof documents and tests this post-Morrison precondition;
  for an invalid active pair its kernel explicitly emits non-finite output
  instead of allowing CUDA `fmaxf` to disguise invalid slope arithmetic as
  the meteorological -35 dBZ clear-air floor.
- **Quadrature adjudication (review F3)**: the wet-particle path remains the
  exact WRF 50-log-bin weighted sum (module_mp_radar.F:83-90/:124-146;
  module_mp_morr_two_moment.F:4626-4667).  It is a compatibility rule, not
  a claim of uniformly converged composite-Simpson integration; no refined
  accuracy mode replaces it on the WRF-compatible path.

### D3. Experiment-schema fail-loud rejections (Phase 5, lane L1)

#### D3a. Fail-loud scope rejections

- **WRF-native behavior**: WRF accepts these namelist settings and runs
  (or silently ignores them); woof's experiment loader
  (`woof/experiment.py`, `woof/namelist_import.py`) rejects them loudly
  at load.  Each rejection message names the mechanism and the Phase that
  owns it, where one exists:
  - **Nonzero `spec_exp` on a nested child** is rejected rather than
    silently ignored.  WRF's nested `lbc_fcx_gcx` branch has no sponge
    term (dyn_em/module_bc_em.F:1297-1341; `spongeweight` only in the
    specified branch :1320, with the nested branch :1325-1337 carrying
    the sponge lines commented out).  woof's reuse of the Phase-4 Davies
    `_weights` with `spec_exp = 0` is therefore a stated transliteration of
    that branch instead of a default-value coincidence.
  - **Moving nests** (`num_moves`/`move_*`/`vortex_*`/corral keys),
    **vertical nesting** (per-domain `e_vert`/`eta_levels`/`p_top`
    differences and `vert_refine_method != 0`; WRF only calls
    `init_domain_vert_nesting` when a nest refines the vertical grid, guarded
    by `nest%e_vert /= parent%e_vert` at
    share/mediation_integrate.F:663 with the call at :665),
    **adaptive time step** (`use_adaptive_time_step`), and
    **`input_from_hires`** terrain are rejected design-scope options that the
    bundle namelist never exercises.
  - Only **`interp_method_type = 2`** (SINT, the WRF default per
    Registry.EM_COMMON:2301) is accepted.  Bilinear/nearest/quadratic and
    `nest_interp_coord = 1` isobaric re-interpolation are rejected as a scope
    cut, not a behavior change; the bundle namelist runs the defaults.  The
    SINT kernels themselves are lane L3.
  - **`feedback != 0` and `smooth_option != 0`** are rejected this phase
    (one-way nesting; the dormant feedback machinery has separate Phase-5b
    activation gates).
  - **`use_theta_m = 1`** is rejected because woof implements the
    non-moist-theta thermodynamic/heating branch only (D1).  Omission is
    also rejected because WRF's Registry default is 1; supported effective
    namelists must say `use_theta_m = 0` explicitly.
  - **`nwp_diagnostics != 0`** is rejected because it forces WRF's
    `diag_flag` every step and falls outside D2's output-due diagnostic
    cadence.  *Amendment (STEP17, 2026-07-29): retired -- the knob now
    imports, gating only the UP_HELI_MAX running-max diagnostic; D2's
    output-due REFL_10CM cadence is unaffected (see the amended
    `nwp_diagnostics` scope entry above).*
- **Inactive when**: none of the rejected options is requested.  Future
  implementation of any rejected mechanism is a roadmap item requiring its
  own verification against WRF and scope amendment; it is not a vanish condition for this
  entry.

#### D3b. Exact-rational derivation and chained-FP32 dt

- Child `dx` and the clock interval are never hand-typed authorities.  The
  loader derives `dx_child = dx_parent/parent_grid_ratio` and
  `dt_child = dt_parent/parent_time_step_ratio` as exact rationals
  (share/set_timekeeping.F:366-368; Registry.EM_COMMON:2245-2246), then
  cross-checks supplied decimals.  The bundle's truncated `333.333333` d04
  `dx` passes; a hand-typed “500 m” is a hard error.  WRF itself trusts the
  namelist `dx` decimals.
- The WRF REAL time-step chain is now **implemented bit-exactly**, not a
  future intent: `woof/core/clock.py` consumes the schema's chained-FP32
  values and validates the root construction and every child division by
  bit pattern.  `tests/test_clock.py::test_chained_fp32_dt_consumed_bit_pins`
  pins d01..d04 to `0x42700000`, `0x41700000`, `0x40A00000`, and
  `0x3FD55555`; `test_chained_dt_validation_is_bit_exact` proves a one-ULP
  neighbor is rejected.  D4 records the runtime tick representation and
  the remaining `dtbc` deviation.

#### D3c. Physics substitutions and comparison scope

- The importer maps ra_lw/ra_sw 4 (RRTMG) to RTE+RRTMGP, emitting a
  structured `SubstitutionReport`; every other unimplemented scheme id is a
  hard error.  That is a **model-form change with no error bound**.  mp 55
  (ISHMAEL) was mapped to Morrison 2-moment the same way until 2026-10-12;
  it is now refused by name, and a configuration that runs Morrison in its
  place asks for mp 10.  bl_pbl 11 (Shin-Hong) was a third
  substitution (to YSU) until the Shin-Hong port; it now imports natively
  as 11 and runs the ported scheme.  The campaign-era configs and
  comparisons below predate that admission and record the substitution
  they actually ran.
- Numerical fidelity gates in the N-ladder use the matched-physics
  instrumented oracle in the manifest above (Morrison+YSU, including the
  N1.5 registered tolerance).  Comparisons against the original ISHMAEL +
  Shin-Hong + RRTMG reference frames are statistical comparisons under the
  gate ledger, not numerical-equivalence claims.
- The flagship publication framing is the **effective namelist**.  Its
  reproducibility artifacts publish both the original WRF namelists and the
  effective woof namelist/config after the explicit, structured
  substitutions and scope selections; neither may be presented as though
  it were the other.

### D4. Integer-tick clock: WRF-recurrent dtbc / running seconds and the exact-calendar scope (Phase 5, lane L2)

- **WRF-native mechanism**: WRF's boundary-tendency clock is a REAL
  accumulator, `grid%dtbc = grid%dtbc + grid%dt` inside solve_em
  (dyn_em/solve_em.F:371-372), executed BEFORE the relax/spec boundary
  routines consume it (:932-952), reset to zero at every nested force
  (share/mediation_force_domain.F:203-206) and, on d01, at every
  external-boundary interval seam (`IF (currentTime .EQ.
  grid%this_bdy_time) grid%dtbc = 0.`, share/mediation_integrate.F:1522).
  Running seconds (`curr_secs`) are DERIVED per step from the rational
  clock interval as REAL via `dt_whole + dt_num / REAL(dt_den)`
  (dyn_em/adapt_timestep_em.F `real_time`; interval built at
  dyn_em/solve_em.F:330-333, consumed by first-RK physics :800-815,
  widened to REAL*8 only at :4752-4755).
- **woof mechanism** (`woof/core/clock.py`): dtbc is now
  implemented-identical to WRF.  `DomainClock.dtbc_fp32` resets to
  positive FP32 zero on the existing force/LBC-seam calendar, then
  `prepare_step()` evaluates
  `np.float32(dtbc_fp32 + spec.dt_fp32)` immediately before every solve;
  `dtbc_launch_fp32` exposes that post-increment accumulator.  The reset
  at every parent interval prevents cross-interval drift, so this is
  bit-faithful for every ratio chain, including the former 1:7
  counterexample.  The existing d04 pins remain 0x3FD55555,
  0x40555555, 0x40A00000.  Exact integer ticks remain the sole
  sync/alarm/restart authority.  Running seconds remain tick-derived:
  `DomainClock.elapsed_seconds_fp32` mirrors WRF's REAL expression
  exactly (`FP32(whole) + FP32(num)/FP32(den)` from the tick
  decomposition, two FP32 roundings), while `elapsed_seconds` (one FP64
  division) is the internal interval-selection/restart authority.
  Scaling num/den by a common factor is guaranteed to leave the FP32 quotient
  unchanged when both integers satisfy `|n| <= 2^24` (and hence are exactly
  representable); that sufficient condition is not the definition of FP32
  integer representability.  `clock.py` defensively asserts this conservative
  bound for `whole`, `num`, and `den` at the decomposition; the
  current `num < den <= tick_den` bounds make the condition trivial for the
  registered clocks, and `tests/test_clock.py` pins the boundary/fail-loud
  behavior.  (The former "dtbc deviation scope" bullet is retired: the
  WRF-recurrent accumulator above is bit-faithful for every ratio chain,
  including the 1:7 counterexample it documented, those bits are now
  reproduced, not deviated from, and remain test-pinned.)
- **curr_secs**: NOT a deviation on the WRF-compatible path,
  `elapsed_seconds_fp32` reproduces WRF's two-rounding expression
  bit-for-bit (one ULP ABOVE the single-rounded FP64→FP32 cast at times
  like d04's 5/3 s: 0x3FD55556 vs 0x3FD55555; equal at all
  whole-second and dyadic instants).  Consumers that instead take the
  FP64 `elapsed_seconds` (interval selection, restart bookkeeping) are
  deliberately exact; kernel-facing WRF-parity consumers must use the
  FP32 mirror.
- **Root external-boundary dtbc, deviation CLOSED (Davies clock bind,
  2026-07-28; supersedes the F20 adjudication of 2026-07-17)**: the
  ROOT's external Davies launches now consume the WRF-recurrent
  `dtbc_launch_fp32` accumulator exactly like nest launches.  The tree
  build binds the root boundary clock (`woof/core/model.py`
  `build_experiment` calls `bind_lateral_boundary_clock(root.state,
  root.clock)`; the N5S restored-model builder binds its manual root
  the same way), so root relaxation, the held moist/scalar targets, and
  the final specified-ring overwrite take WRF's post-increment dtbc
  (reset on boundary read, share/mediation_integrate.F:1515-1522;
  increment before the solve, dyn_em/solve_em.F:371-372: dt..T_bdy per
  interval) in place of the retired elapsed-based calculation
  (`state.elapsed_seconds - interval.start_seconds`, one dt BEHIND WRF
  in the relax zone).  The bound final ring additionally owns the OLD
  record at dtbc=T_bdy on the last pre-seam step (solve_em.F:4531-4639;
  the previous half-open lookup installed the new record at dtbc=0,
  equal-valued, not FP32-bit-identical), closing the once-per-interval
  seam micro-mismatch.  At the campaign cadence (dt=60 s, T_bdy=21600 s)
  every FP32 dtbc partial sum is exactly representable, so the fix is a
  pure phase correction with zero roundoff residue.  CONSEQUENCES:
  (1) the frozen Phase-3/4 reference bytes (out/real74-t7-final,
  out/real74-t7-final-r2, out/cardinal) encode the retired semantics
  and are historical evidence only, the N-series invariance ratchets
  regenerate against the seam-closure anchor epoch
  (out/real74-t7-final-r3) in the batched end-of-Wave-1 regeneration;
  (2) restart headers carry the `root_external_lbc_clock` semantic
  identity ("wrf-postincrement-v1" bound / "legacy-elapsed-v0" unbound;
  a header without the key is a pre-bind file), and restores refuse a
  semantic mismatch, so old unbound checkpoints fail closed;
  (3) legacy direct paths that attach external boundaries without a
  DomainClock (era5_direct/gfs_direct) REMAIN elapsed-based by scope
  decision: their checkpoints self-identify as legacy and cannot mix
  with the bound production trajectory.  Child launches were already
  bit-faithful to WRF per the pins above and are unchanged.
- **Exact-calendar scope restriction**: every cadence
  (radt/cudt/bldt/history/restart/lbc_interval) must be an exact whole
  number of the owning domain's steps, noncommensurate alarms are a
  load-time error (`_cadence_ticks`, resolve_clock).  WRF is more
  permissive: its `Is_alarm_tstep` crossing-alarm predicate
  (frame/module_domain.F:2503-2516, `PrevRingTime + RingInterval <=
  CurrTime + TimeStep`) rings an alarm on the step that CROSSES the
  ring time, so WRF accepts e.g. a 100 s history interval on a 60 s
  clock (frames at 120, 180, 300, ... s).  woof rejects such configs
  loudly (fail-loud scope cut; the bundle's calendars are all
  commensurate).  Note that for integer refinement a parent-commensurate
  cadence is automatically child-commensurate
  (`k*dt_parent = k*r*dt_child`), so the restriction binds only on
  genuinely noncommensurate namelist values.
- **Terminal feedback placement (schedule table)**: WRF does not
  suppress the final period's feedback; the per-kid loop guard
  (frame/module_integrate.F:439-445) skips it there and the last-io
  tail RE-ISSUES it (`med_nest_feedback` at :523 inside :507-526)
  whenever the head grid or any ancestor-chain grid is at stop time.
  woof's flat table carries both families at WRF's exact call
  positions (identical op sequence on linear chains; reordered on
  branching trees), so the dormant table is already complete for the
  Phase-5b two-way activation.  The ratified architecture's F8 wording
  ("final period omits its FEEDBACK ops") predates this correction and
  is amended by the controller (p5t9 review round, 2026-07-16).
- **Vanishes when**: dtbc is no longer a deviation; the recurrence is
  implemented-identical for all ratios.  Calendar scope, if a future
  case needs WRF crossing
  alarms, `Is_alarm_tstep` semantics would be implemented behind the
  same DomainClock predicates (a plan amendment, not a silent change).

### D5. SINT geometry precomputed FP64-on-host, stored FP32 (Phase 5, lane L3)

- **WRF-native behavior**: `SINT`/`SINTB` construct the XIG/XJG offset
  coefficient tables in REAL, on the fly, on every call
  (share/sint.F:13-14, :31, :46-57), alongside the donor-cell index
  arithmetic of the wrappers (share/interp_fcn.F:975-985, :2562-2575).
- **woof**: nests are static, so the geometry (donor index maps plus
  the XIG/XJG tables) is precomputed ONCE at registration
  (`woof/core/nest_interp.py` `sint_offsets`/`register_nest`), built in
  FP64 on host and stored FP32 on device; the FP64 verification mirrors
  (`woof/verify/npref.py`) consume the SAME FP32-rounded tables so the
  N1 fp32_floor oracle stays discriminating (architecture §D, F6
  amendment; ratified deviations list).  ONLY geometry is precomputed:
  the DONOR/TR4 flux and min/max-limiter arithmetic is field-dependent
  and is evaluated per field at force time, never baked into weights.
- **Bound**: donor index maps are integer arithmetic (no precision
  content).  For rr ∈ {1, 2, 3, 4}, every ratio the bundle chain
  (4/3/3) exercises, the FP64-built/FP32-stored XIG/XJG values
  coincide bitwise with WRF's per-op REAL construction, machine-proven
  against a REAL-build emulation of sint.F:49-57 in
  `tests/test_nest_interp.py::test_sint_offsets_vs_wrf_real_build_emulation`
  (the stored values themselves are additionally pinned in
  `test_sint_offset_tables_fp32_pins`).  At rr = 5 the two
  constructions diverge by exactly 1 ULP at ip = 3: WRF per-op
  `fl32(0.4f - 0.6f)` = -0.20000002 (0xBE4CCCCE) versus woof
  `fl32(fl64(-0.2))` = -0.2 (0xBE4CCCCD); every other (rr <= 5, ip)
  pair coincides.  The counterexample is recorded because the ratified
  schedule-pin family includes a (1,5,3) ratio chain, making rr = 5 a
  plausible future fixture ratio.  The general numeric consequence is
  owned by milestone N1.5 (the instrumented-WRF bdy-table oracle
  compares the full force pipeline at the pre-registered FP32-relative
  1e-6 band).
- **Vanishes when**: moving nests (a design non-goal) would force
  per-call geometry; or N1.5 evidence shows a discrepancy, which
  reopens this entry by controller adjudication.

### D6. adjust_tempqv evaluates in FP64 on device, stores FP32 (Phase 5, lane L3)

- **WRF-native behavior**: `adjust_tempqv`
  (dyn_em/nest_init_utils.F:812-890) evaluates in REAL: the post-blend
  theta/qv correction chain `tc = (th+300)*(p/1e5)**(2/7) - 273.15`
  (:853/:855/:881), Magnus saturation (:857/:882), RH conservation and
  qv reconstruction (:859, :883-884).
- **woof**: the `nest_adjust_tempqv` kernel
  (`woof/core/kernels/nest.cu`) transliterates the identical algorithm
  and constants but computes each point in FP64, storing the REAL/FP32
  result, an implementation-forced deviation.  The tc chain cancels
  ~275 K of magnitude and the Magnus exponential amplifies the residue:
  a REAL-internal kernel is irreducibly hundreds of ULPs from ANY FP64
  mirror in qv, which makes the pre-registered N1 gate
  (`adjust_tempqv_vs_fp64_mirror`, fp32_floor at
  `FP32_FLOOR_MAX_ULPS = 8`) unpassable.  The alternatives are strictly
  worse under gates-first discipline (lane-review adjudication A3):
  degrading the mirror to FP32 internals destroys the oracle's
  discriminating power, and loosening a pre-registered threshold
  requires the N1.5 attribution run in hand (nest_gates ledger).
- **Bound**: a one-shot child-initialization operator (called once per
  nest at build, mediation_integrate.F:749 position).  Outside the
  blend rows `mub == save_mub`, hence `p_new == p_old`, `dth = 0`, and
  theta is bit-unchanged with qv reconstructed through the same tc
  (identity up to FP64 roundoff): the deviation is CONFINED to the
  terrain-blend rows, where it is O(1e-5) relative in t/qv versus WRF's
  REAL evaluation; monitored by the N1 HGT/MUB static oracles, the N3
  blend-zone T2/TSK bias diagnostic, and the N3 statistical gates.
- **Vanishes when**: never by default; the controller may revisit with
  N1.5 evidence in hand (matched-physics instrumented WRF), which would
  arbitrate REAL-faithful internals against the gate design.

### D7. Noah sea-ice runtime thermodynamics are not implemented (Phase 5, lane L4)

- **RETIRED in 2.8.6 (lane/parity-noah)**: `seaice_noah` and `SFLX_SEAICE`
  are ported as `noah_seaice_column` (`woof/core/kernels/noah.cu`) on WRF's
  default namelist path (seaice_albedo_opt, seaice_thickness_opt and
  seaice_snowdepth_opt 0, fractional_seaice 0) and run after both `lsm` and
  `lsm_mosaic`, with the model time step, as the surface driver orders them.
  It is graded against a WRF v4.6.1 column oracle
  (`tools/noah_seaice_wrf461_oracle`, 24 columns x 3 fixtures) by
  `tests/test_noah_seaice_wrf461_parity.py`.  The text below records the
  deviation as it stood before the port.

- **WRF initialization semantics, verified in the bundled v4.6.1 source**:
  for `fractional_seaice=0`, `adjust_for_seaice_post` uses an XICE threshold
  of 0.5 (`share/module_soil_pre.F:250-260`), snaps qualifying ice to XICE=1
  (`:260-262`), sets water-point TMN=271.4 K (`:263-264`), constructs four
  equispaced TSLB levels through a 3 m ice column from TSK to TMN
  (`:289-295`), and initializes SMOIS=1/SH2O=0 (`:297-300`); sub-threshold
  XICE is zeroed (`:301-302`).  woof now mirrors all four cheap init
  behaviors in `woof/ingest/soil.py`.  On the first Noah call WRF then sets
  sea-ice SH2O=1 and returns (`phys/module_sf_noahdrv.F:1067-1076`), which
  `woof/core/kernels/noah.cu:938-941` already mirrors.
- **Registered runtime deviation**: WRF separately calls `seaice_noah` from
  the surface driver (`phys/module_surface_driver.F:2917-2929`; interface and
  updated fields in `phys/module_sf_noah_seaice_drv.F:14-23`).  That scheme
  evolves the sea-ice surface/ice-column energy state and its TSK, GRDFLX,
  HFX/QFX, snow, albedo, emissivity, and roughness coupling.  woof has no
  sea-ice scheme in this phase.  Its Noah land kernel correctly recognizes
  an ice column and returns, but no later scheme advances that column; TSK
  therefore remains frozen at its initialized SKINTEMP over ice, while the
  surface-layer flux path continues to use that frozen skin.
- **Observed scope**: CPU reproduction of the staged May 3 1999 ERA5 SEAICE
  interpolation on the configured CONUS grid found 11 water cells with
  positive ice and 6 cells at/above the 0.5 Noah threshold, in the Gulf of
  St. Lawrence.  This is a documented small-signal deviation for that
  May-1999 case.  It does **not** affect the frozen real74 profile: its GRIB
  inventory contains neither SEAICE nor SOILGEO, so XICE remains exact zero
  and every sea-ice-only init/runtime branch is inactive.
- **Vanishes when**: a WRF-v4.6.1-equivalent Noah sea-ice driver and column
  thermodynamics are implemented and verified.  Adding a different generic
  ice model does not silently close this WRF-parity entry.

### D8. vertical_diffusion_w_2 takes the VERTICAL momentum coefficient (P6 LES, nested-LES lane)

- **WHEN THIS IS LIVE, first, because the rest of the entry reads as
  though it always is.** The deviation can only change an answer where
  `xkmv` and `xkmh` actually differ, which means `km_opt` 2 or 3 **with
  `mix_isotropic = 0`**. Under `mix_isotropic = 1` the two coefficients
  are the same number and the divergence is arithmetically inert: the
  WRF `em_les` oracle lane measured `XKMH == XKMV` at **every one of
  589,824 points, maximum difference exactly 0**, with the two caps at
  2000 and 233–352 m² s⁻¹ against a realised maximum Km of 15.9, so
  neither cap was approached and roughly 15x more mixing would be needed
  before they could bind differently. Concretely, in this repository:
  the idealized CBL case runs `mix_isotropic = 1`, so **every idealized
  LES receipt is unaffected by D8**; the shipped nested configuration
  `configs/les_nest_250m_km3.toml` ran `mix_isotropic = 0` on its 250 m
  child, which is where the deviation was essential and where the
  failure it prevents was observed. **Since 2026-08-09 the only files
  under `configs/` that run `mix_isotropic = 0` on an LES child are the
  five archived records**: `configs/frozen/les_nest_250m_km3.toml`,
  `configs/frozen/les_nest_250m_grayzone.toml`,
  `configs/frozen/les_tornado_100m_mayfield_20211210.toml`, and the
  `attempt2` / `attempt2b` files that sit at the top level of `configs/`
  because a receipt names them there. Every config offered as something
  to run is off the path. The per-axis length exposes the horizontal `w`
  operator to a coefficient capped on the layer depth, at
  `mix_upper_bound*(dz_max/dx)^2` of 0.702 on that 250 m child and 4.23
  on a 100 m one, and the second aborted a run (`docs/public/LES.md`
  §4). D8 is therefore arithmetically inert on the configs a user starts
  from, and live on the five records, which is the point of keeping
  them: it is what makes their receipts reproducible. It stays
  registered for a second reason too: the divergence is in the operator
  rather than in any config, so a user who writes `mix_isotropic = 0`
  reaches it immediately. The bytes of each record are pinned by sha256
  in `tests/test_shipped_configs_mixing_stability.py`, which is also the
  screen that keeps the first sentence true.
- **WRF's own pairing, verified in the bundled v4.6.1 source.** Every
  subgrid stress is given the exchange coefficient of its own directions.
  tau11/tau22/tau12 are horizontal-horizontal and take `xkmh`
  (`dyn_em/module_diffusion_em.F:2978-2996`).  tau13/tau23 are mixed and
  take `xkmv` in BOTH operators that consume them:
  `vertical_diffusion_u_2`/`_v_2` (`:4128-4147`) and
  `horizontal_diffusion_w_2` (`:2998-3006`, whose dummy argument is
  spelled `xkmv` at `:3524`).  Scalars take `xkhh` horizontally
  (`:3007-3013`) and `xkhv` vertically (`:4274-4279`).
- **The one call that breaks it.** tau33 = 2 K dw/dz is the
  vertical-vertical stress; `vertical_diffusion_w_2`, the divergence of
  it, is handed `xkmh` (`:4145-4155`).
- **Registered runtime deviation**: WOOF hands that operator `xkmv`.
- **Why, in one inequality.**  The operator is explicit, so it needs
  `2*K*dt/dz^2 <= 1/2`.  `smag_km` caps the two coefficients on different
  length scales (`:1890-1908`): `xkmv <= mix_upper_bound*mlen_v^2/dt` with
  `mlen_v = dz`, and `xkmh <= mix_upper_bound*mlen_h^2/dt` with
  `mlen_h^2 = dx*dy`.  So `2*xkmv*dt/dz^2 <= 2*mix_upper_bound` = 0.2 at
  the Registry default -- WRF's own cap IS this operator's stability
  guarantee, and it guarantees it only for `xkmv`.  With `xkmh` the same
  quantity is bounded only by `2*mix_upper_bound*(dx/dz)^2`.  Transcribing
  WRF here means transcribing an operator that cannot run on any
  anisotropic LES grid.
- **Observed scope**: invisible for km_opt=1 (khdif/kvdif are normally
  equal) and for km_opt=4, where `smag2d_km` defines `xkmv = xkmh`
  outright (`:2035`, "v4.2 and later, this is used for hor. diff. of w").
  It is reachable only by km_opt=2/3 with `mix_isotropic = 0`.  All 15
  configs in `configs/` select km_opt=4, so no shipped trajectory moves.
  Measured on a 402x402 250 m km_opt=3 PBL-off child inside a 750 m nest
  (dx 250 m, dz 17-25 m, dt 1.25 s): with `xkmh` the diffusion number
  crossed 0.5 in the lateral boundary zone at child step 7 and reached 127
  by step 121, where the run died with u = -522 m/s at k = 0; with `xkmv`
  the same tree completes its 15-hour window.
- **Precedent in the same routine**: WRF corrected the identical class of
  coefficient-selection error for the prognostic-TKE self-diffusion --
  `vertical_diffusion_s` for tke took `xkhv` until wrf-model/WRF issue
  #2026, and v4.6.1 passes `xkmv` (`:4333-4341`), which WOOF follows.
- **Vanishes when**: WRF passes `xkmv` to `vertical_diffusion_w_2`.  Until
  then this entry is the divergence, and
  `tests/test_smag3d.py::test_mix_upper_bound_makes_the_vertical_w_operator_stable_only_with_kmv`
  is its standing proof.

### D9. Thompson aerosol-aware (mp_physics=28): admission, aerosol ingest, lateral boundaries and PBL mixing

The mp=28 physics is a line-for-line transcription of unmodified WRF v4.6.1
`phys/module_mp_thompson.F` at commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`.  Everything registered here is
*around* the scheme: how a run is admitted, where the aerosol comes from, what
happens at a boundary, and what mixes it.  Each item is a decision, not an
oversight, and each says what would close it.

- **D9a: RETIRED.  woof now runs the configuration `real.exe` admits.**
  This entry used to record the opposite: that
  `dyn_em/module_initialize_real.F:2734-2736`
  (`CALL wrf_error_fatal ('wif_input_opt=0 but mp_physics=28')`) refused the
  only mp=28 initial state woof could produce, because woof had no WIF
  (water/ice-friendly aerosol) ingest and fell back to `thompson_init`'s
  synthetic CCN/IN profile.  It named its own closing condition -- "vanishes
  when a QNWFA/QNIFA metgrid ingest lane lands and `wif_input_opt` becomes a
  real setting" -- and that condition is met.  `woof/ingest/wif_climatology.py`
  ports WRF's global monthly QNWFA/QNIFA climatology
  (`QNWFA_QNIFA_SIGMA_MONTHLY.dat`) with metgrid's `four_pt` horizontal
  interpolation, `real.exe`'s `monthly_interp_to_date` temporal weighting and
  its `vert_interp` onto the dry eta pressure, matched to `real.exe` to 1e-5;
  a real-data mp=28 run takes it BY DEFAULT (`RunConfig.mp28_aerosol_source`
  = `"auto"`), which is the same state `real.exe` reaches through
  `use_aero_icbc = .true.` -> `aer_init_opt = 1` with `wif_input_opt = 1`.
  A woof mp=28 run initialized this way and a WIF-initialized WRF mp=28 run
  ARE directly comparable, which is the whole point of closing it.
  The constant that carried the old warning,
  `woof.config.MP28_AEROSOL_SOURCE_DEVIATION`, is gone rather than softened:
  fixing a defect retires its guards.  What replaces it is
  `MP28_AEROSOL_SOURCE_DEFAULT` (what a run that found the dataset did) and
  `MP28_AEROSOL_SYNTHETIC_FALLBACK` (what a run that did not found instead,
  announced by name in the receipt, because a forecast started from the
  synthetic profile is a scientifically different forecast).  **Still true:**
  when no dataset can be located, `mp28_aerosol_source = "auto"` falls back to
  the synthetic profile and a run that did so is NOT comparable to a
  WIF-initialized WRF run -- the fallback says so in the receipt, and
  `mp28_aerosol_source = "climatology"` refuses instead of falling back.

- **D9b: aerosol BOUNDARY conditions do not exist as an ingest lane.**
  NARROWED, not retired: the INITIAL-condition half of this entry closed with
  D9a above -- `aer_init_opt = 1` with `wif_input_opt = 1` (`use_aero_icbc =
  .true.`) is ported and is the real-data default.  What remains open is
  everything downstream of initialization: `use_rap_aero_icbc = .false.`,
  `qna_update = 0`, no aerosol crosses a specified lateral boundary, and there
  is no `nbca`
  (black-carbon) species at all -- WRF's `thompsonaero` package registers
  `scalar:...,qnbca` (`Registry/Registry.EM_COMMON:3036`) and woof does not
  allocate it.  What fills the gap is WRF's own fallback: `thompson_init`'s
  synthetic profile, whose SHAPE is pinned by oracle fixtures
  `aero-init-profile` and `aero-sfc-emit` and whose INSTALLATION on a real
  domain is `woof/core/physics.py::initialize_physics` calling
  `woof/core/microphysics.py::microphysics_init` once per domain, at the seam
  WRF's `phy_init` calls `mp_init` (`phys/module_physics_init.F:1635`, whose
  `CASE (THOMPSONAERO)` arm at `:4522-4538` calls `thompson_init`).  A
  cold-start mp=28 domain therefore begins strictly ABOVE the terminal apply's
  clamps rather than pinned at them; a nest under an aerosol-bearing parent
  and a run about to restore a checkpoint both take WRF's own presence test
  (`:490`/`:493` for CCN, `:528`/`:531` for IN) and are left alone.
  `nifa2d` is allocated and stays exactly zero, matching WRF, where
  `thompson_init` never writes it.

  RECORDED BECAUSE IT WAS FALSE FOR FOUR WAVES.  Until 2026-08-01 nothing in
  `gpuwm/` called that hook: the fill was implemented and oracle-gated, and
  every mp=28 forecast integrated from `nwfa = nifa = 0` with WRF's floors
  holding CCN at a maritime-clean value for the whole run -- no NaN, no bound
  violation, no warning, and no column gate able to see it, because every
  fixture supplies its own aerosol.  Measured over two otherwise identical
  150-step forecasts, the cost was 5.6x fewer cloud droplets and +74.0%
  domain-total surface rain; the same measurement is now the SENSITIVITY of an
  mp=28 forecast to its aerosol initial condition and is published in
  `docs/public/validation/mp28-column-evidence.md` 6.1 and on the registry
  option under `extensions.aerosol_initialisation`. The folder name
  `docs/public/validation/` is historical: its contents compare woof
  with WRF, which is code verification, not validation against
  observations. The path is retained so existing evidence links keep
  resolving. Pinned in code by
  `tests/test_mp28_forecast_smoke.py::test_microphysics_init_has_a_production_call_site`
  (which also asserts there is exactly ONE caller, because once-per-domain
  silently becoming per-step would overwrite an advected, activated and
  scavenged aerosol field with the synthetic one) and by
  `test_a_freshly_initialised_mp28_domain_carries_the_profile_not_zero`.

- **D9c: external lateral boundaries carry no aerosol.**  WRF's Registry
  declares `qnwfa`/`qnifa` with the trailing `b` of `ikjftb`, which allocates
  real `qnwfa_b*`/`qnwfa_bt*` boundary arrays exactly as for `qv`, so WRF
  forces aerosol at a specified domain's edge from the boundary file.  woof
  couples ONLY `qv` from external boundary snapshots
  (`woof/ingest/lateral_bc.py` `_coupled_device_fields`) and gives every
  other scalar flow-dependent boundaries with zero inflow.  Consequence,
  stated precisely: aerosol-free air advects in at the upstream face and
  monotonically depletes `nwfa`/`nifa` in the boundary zone for as long as the
  run continues.  It does not produce a NaN, a negative or a health trip,
  because WRF's own terminal clamps hold the floor (`nwfa >= 11.1E6` and
  `nifa >= naIN1*0.01 == 5.0E3` per m3).  This is the SAME policy every other
  hydrometeor already takes here, applied to a field where the effect is
  qualitatively different, because aerosol number has no interior source term
  except the fixed surface emission.  **Vanishes when** the LBC ingest carries
  `qnwfa`/`qnifa`.  Registered rather than closed, and the reasoning is also
  carried in the `woof/core/moist.py` module docstring.

- **D9d: MYNN does not mix `nc`/`nwfa`/`nifa`.**  `woof/core/mynn_pbl.py`
  passes `flag_qnc`/`flag_qnwfa`/`flag_qnifa` as literal `False` at every call
  site, so those species never enter MYNN's scalar mixing.  WRF mixes them
  when `bl_mynn_mixscalars > 0` (`phys/module_bl_mynn.F:4735,:4777,:4957`), or
  outside MYNN through `scalar_pblmix` (`phys/module_pbl_driver.F:2251`).
  Measured accurately: at woof's pinned MYNN identity `bl_mynn_mixscalars = 0`,
  and WRF's own `check_a_mundo` raises `scalar_pblmix` to 1 ONLY when
  `use_aero_icbc` or `use_rap_aero_icbc` is set
  (`share/module_check_a_mundo.F:2477-2495`) -- both of which woof refuses --
  and forces it back to 0 on any MYNN domain with `bl_mynn_mixscalars = 1`
  (`:2497-2511`).  So WRF's own value in woof's scope is 0 as well, and in
  the v1 scope the two models agree: the divergence is LATENT rather than
  active.
  What is not latent is the mechanism: woof's withholding is structural and
  cannot be lifted by a namelist, and mp=28 is the first configuration in
  which those fields carry real prognostic values, so this is where the
  withholding first becomes physically meaningful.  **Vanishes when** the PBL
  area lights up the pass-through together with `bl_mynn_mixscalars`.

- **D9e: WRF silently overwrites two knobs under mp=28; woof refuses
  instead.**  `share/module_check_a_mundo.F:2459-2474` forces
  `grav_settling = 0` on every mp=28 domain unconditionally, and `:2477-2495`
  forces `scalar_pblmix = 1` on an mp=28 domain **only** when `use_aero_icbc`
  or `use_rap_aero_icbc` is set, with `:2497-2511` forcing it back to 0 on any
  MYNN domain at `bl_mynn_mixscalars = 1`.  All three write through
  `wrf_debug`/`wrf_message`, not `wrf_error_fatal`, so a WRF user is not told
  at error level that their namelist was rewritten.  woof's standing posture
  is to refuse where WRF overwrites, so both knobs are published
  `implemented: false` in the registry and a nonzero request is an error
  rather than a silent correction.

- **D9f: mixed mp=8 <-> mp=28 nest edges are refused by name.**  The registry
  registers no cross-scheme transition rule involving mp=28, so
  `validate_physics_plan` refuses such an edge with
  `unsupported-component-transition`, and
  `woof/core/microphysics_transition.py` refuses the same edge at runtime
  with a message that names the scheme rather than falling through to the
  generic "not a ported selector" text.  The obvious closure is WRF's own
  non-aerosol-aware fallback (`mp_gt_driver` uses `nc = Nt_c/rho`,
  `nwfa = 11.1E6/rho`, `nifa = naIN1*0.01/rho` when `is_aerosol_aware` is
  false), but applying it at a nest edge would hand the child FABRICATED
  aerosol instead of its parent's, indistinguishable from correct behaviour to
  every gate in this tree.  Refused rather than approximated.

- **D9g: mp=28 and mp=8 are deliberately NOT bit-identical thermodynamics.**
  `woof/core/kernels/thompson_aerosol_common.cuh` contraction-pins the
  `RSLF`/`RSIF` Horner chains; `woof/core/kernels/thompson.cu` leaves them
  contracted. nvrtc defaults to `--fmad=true` while the
  gfortran `-O2` baseline-x86-64 oracle emits no FMA, so the unpinned chain
  lands one ULP low.  That is not cosmetic: `module_mp_thompson.F:3401` opens
  the entire condensation / CCN-activation block on `ssatw > eps` with
  `eps = 1.E-15` (`:185`), so one ULP flips a branch. In mp=28 these two
  Horner chains are pinned to reproduce WRF's rounding; mp=8 retains its
  FMA-contracted saturation chains. This is code verification of those
  expressions, not a claim that all mp=28 arithmetic matches WRF. The
  mp=8 kernels changed in 2.7.4 and on 2026-09-23, after the 2026-07-28
  WRF matched-run comparison, so that trajectory describes an earlier
  build. WRF is the reference for this comparison; agreement between
  the two woof schemes is not the goal.

  CITATION REPOINTED 2026-08-01, from `:3400` to `:3401`.  Every publication
  of this deviation -- this entry, the registry warning, `docs/public/
  PHYSICS.md`, the evidence document and two source comments -- cited `:3400`,
  which is `orho = 1./rho(k)`.  The branch is one line lower:
  `module_mp_thompson.F:3401` is
  `if ( (ssatw(k).gt. eps) .or. (ssatw(k).lt. -eps .and.`.  Verified against
  BOTH reference trees, which hold byte-identical copies of the file
  (md5 `f4ec64c511d6656277c4d1f9346813d1`).  The physics claim is unchanged
  and correct; the line number was one off, which is exactly the rot
  `tools/check_registry_citations.py` exists to catch and could not, because
  WRF is not vendored here.  Two copies outside this package's ownership
  (`woof/core/kernels/thompson_aerosol_common.cuh:652` and
  `tests/test_thompson_aerosol_device_helpers.py:1399`) still say `:3400` and
  are filed as an integration request.

- **D9h: mp=8's own pre-existing gamma-ratio deviation, recorded not fixed.**
  Reproducing WRF's `REAL(4)` `WGAMMA`/`GAMMLN` in float32 gives
  `ccg(2,12)*ocg1(12) = 2729.9973` and `ccg(5,12)*ocg2(12) = 272.00012`, while
  `woof/core/kernels/thompson.cu` hardcodes `2730.0f` at :882, :999, :4005,
  :4128, :4680 and `272.0f` at :888, :1006.  At `thompson.cu:343` the literal
  `2730.0f` is CORRECT, because `calc_effectRad` genuinely uses WRF's INTEGER
  `g_ratio` PARAMETER.  So mp=8 carries a small deviation at five sites and
  none at a sixth.  mp=28 computes the ratio rather than inheriting the
  literal. This records the arithmetic of the build inspected at that time.
  That finding did not change mp=8 because doing so would have moved the
  trajectory used in its historical matched-run comparison against WRF
  (code verification). It does not imply a current byte freeze; later
  mp=8 changes are recorded in D9k and `docs/public/PHYSICS.md`.

- **D9i: `CCN_ACTIVATE.BIN` is a WRF-DISTRIBUTED INPUT, not a `thompson_init`
  product; woof redistributes WRF's bytes verbatim.**  `table_ccnAct`
  (`phys/module_mp_thompson.F:5110-5166`) computes nothing: it `OPEN`s the
  file by bare relative name and performs one sequential `READ`.  The numbers
  are tabulated output of an offline parcel model (Feingold & Heymsfield as
  modified by Eidhammer and Kreidenweis; WRF's own comment at `:5102-5108`),
  so no recompilation of WRF, no re-run of `thompson_init` and no woof code
  path regenerates it or any approximation of it.  This makes it categorically
  unlike the four generated Thompson tables, which this repository's own
  harness reproduces byte for byte.

  THE DECISION MOVED, AND BOTH STATES ARE RECORDED.  Through the port the
  decision was *do not commit*: the file was gitignored, absent from
  `tables/MANIFEST.sha256`, and excluded from the wheel (D9j).  On 2026-08-01
  the licence question was answered, the committed copy is WRF v4.6.1's
  `run/CCN_ACTIVATE.BIN` bit for bit (git blob `9026e073…`) and WRF's
  `LICENSE.txt` is a public-domain dedication with a notice *request*, not a
  condition, and the owner reversed it.  The file is now tracked, listed in
  `tables/MANIFEST.sha256`, and shipped in the wheel and sdist through
  `[tool.setuptools.package-data]`'s `data/**/*` glob;
  `AEROSOL_ASSET_REDISTRIBUTED` is `True` and the notice travels in
  `woof/data/wrf_radiation/LICENSE-WRF.txt`. A clean checkout can now
  run the mp=28 WRF column-oracle gates (code verification).

  WHAT DID NOT MOVE.  It stays absent from
  `thompson_contract.CLASSIC_TABLE_ASSETS` (so `TABLE_SET_ID` is unchanged and
  no mp=8 launch acquires a dependency on it) and from
  `table_assets.EXTERNALIZED_TABLE_FILENAMES` (so `woof fetch-tables`
  neither offers nor promises it; at 35 KB it simply ships).  Membership of
  `tables/MANIFEST.sha256` is not membership of the classic set: nothing
  reads that manifest at run time.  Shipping it is not the same as trusting
  it: 35,288 bytes and sha256
  `f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd` are
  pinned in `woof/core/thompson_aerosol_contract.py` and re-verified on every
  load, absence is fatal and never defaulted, and the file and root
  environment overrides still redirect a run to another copy.  One
  bookkeeping consequence: `AEROSOL_ASSET_REDISTRIBUTED` is written into the
  mp=28 restart identity, so an mp=28 checkpoint written before the reversal
  has a different `physics_setup_fingerprint` and will not resume against a
  build after it.  Asset-level detail is in
  `woof/data/thompson/PROVENANCE.md`.

- **D9j: packaging defect, measured 2026-07-31, CLOSED 2026-08-01, then
  SUPERSEDED the same day.**  The `[tool.setuptools.exclude-package-data]`
  entry landed and the three gates went green; hours later D9i was reversed
  and the entry was removed again, because the file now ships on purpose.
  The gates were not deleted, they were INVERTED: they assert the table
  reaches the wheel and that no exclusion entry survives, which is the same
  defect in the other direction, a stale exclusion would ship an mp=28 that
  fails closed on a missing activation table for every user.  The record of
  what the original defect was is kept, because "measured, not argued" is the
  standard the rest of this register is held to, and because the mechanism it
  exposed (`data/**/*` is a filesystem glob, so tracking status does not
  govern the wheel) is what makes the inverted pin necessary.

  WHAT IT WAS.  The wheel exclusion for D9i was NOT in place.
  `[tool.setuptools.package-data]` selects
  `data/**/*` off the FILESYSTEM, and every other control on
  `CCN_ACTIVATE.BIN` works by the file not being tracked -- none of which
  touches a wheel.  Measured with setuptools' own
  `build_py.get_data_files_without_manifest` against the real
  `pyproject.toml`: a wheel built from a working tree that has staged the file
  (as any tree running the mp=28 gates must) WOULD contain it, contradicting
  `AEROSOL_ASSET_REDISTRIBUTED = False`.

  THE CLOSURE was ONE line -- `"data/thompson/tables/CCN_ACTIVATE.BIN"` in
  `[tool.setuptools.exclude-package-data]` under the `woof` key.  It was
  measured rather than argued, against the real tree: the wheel went from 429
  files to 428, the ONLY file whose membership changed was
  `woof/data/thompson/tables/CCN_ACTIVATE.BIN`, and the packaging gates went
  from 0/3 green to 3/3 green.

  THE SUPERSESSION removed that same line, because D9i was reversed: the file
  is redistributed now, so the wheel is back to 429 files and the exclusion
  would be the defect.  The measurement that mattered survives either way --
  which side of the line the file belongs on is a decision, but that the
  wheel's contents are governed by a FILESYSTEM glob rather than by tracking
  status is a fact, and it is why both the old and the new pin have to be
  explicit.

  Three gates cover it, and the third exists because of a finding worth
  recording.  `test_wheel_exclusions_are_exactly_the_pinned_lists` and
  `test_the_thompson_aerosol_table_reaches_the_wheel` both
  route through a helper that begins `pytest.importorskip("setuptools")` --
  correct, because measuring real wheel contents must use setuptools rather
  than a second implementation of glob semantics, but it means those tests
  SKIP where setuptools is absent.  The project virtualenv this suite runs in
  is exactly such an environment, so the strongest control on this file's
  packaging was silently inert in the place it most needed to fire, and
  a summary line could not tell a skip from a pass.
  `test_the_redistributed_table_is_not_excluded_in_the_pyproject_declaration`
  asserts the DECLARATION out of `tomllib` instead, so it runs everywhere; it
  cannot prove the file reaches the wheel and the setuptools
  gates cannot say anything at all without setuptools, so the two are kept
  together and fail closed in both environments.

- **D9k: code-verification status against WRF column fixtures, stated as numbers.**
  These comparisons use WRF as the reference, not observations. mp=28 is registered
  `implemented: true`, `maturity: implemented-unverified`,
  `reachability.state: component-override`.  The evidence is 22 committed WRF
  column fixtures -- the 19 the port spec names plus three `wp08-*` columns
  from the same oracle build -- driven end to end through the shipped adapter
  and compared on 23 quantities each at a flat 2.0e-6 relative / 2.0e-4 dB
  gate.  **21 of 22 clear every quantity with nothing held out** (18 of the
  19 `aero-*`, plus `wp08-freeze`, `wp08-melt` and `wp08-nusweep`), and the
  gated count is the same 21 of 22: no allowance remains.  The 22nd,
  `aero-cold-overlap`, misses by the declared rain-graupel divergence alone:
  WRF v4.6.1 reads its rain-graupel collision tables out of bounds when the
  scheme is not hail aware (`module_mp_thompson.F:465`, `:607-615`,
  `:2527-2545`) and WOOF reads the one slab the tables hold, so it reads `qr`
  5.070e-04, `nr` 5.066e-04, `qi` 1.618e-04, `qg` 8.776e-05, `effi`
  5.394e-05, `ni` 1.464e-05, `effs` 6.574e-06 and 1.585e-03 dB of
  reflectivity from WRF; with WRF's read emulated in a measurement copy of
  the tree it clears all 23 quantities.  An RTX 4090 (sm_89) and an RTX 5070
  Ti (sm_120) read the whole table identically.

  WHAT CLOSED THE LAST FOUR (the 2.8.6 accumulator rework).  WRF's
  `mp_thompson` never writes a hydrometeor during the call: every process
  adds to a running tendency (`qcten`, `qrten`, `nrten`, `qiten`, `niten`),
  every stage reads `X1d + Xten*DT`, and the terminal apply
  (`module_mp_thompson.F:3972-4053`) rounds each species once, with the size
  bounds after the `:3943-3966` phase cleanup.  The port applied each stage
  to the state in turn, so a species was rounded once per stage, and the
  rain and ice fallout it shared with mp=8 folded the terminal size bounds in
  before the freeze.  mp=28 now carries the five tendencies, runs the
  `:3033-3055` / `:3070-3091` balances in tendency form, has its own
  tendency-form rain and ice fallout, applies once in a terminal kernel, and
  takes the cold network's rain-conservation ratio as REAL as `:1615`
  declares it.  The fork generation keeps its in-place rain and ice and the
  frozen mp=8 kernel file is unchanged.  Measured: `aero-cloud-freeze-nc` `qc`
  4.926e-06 -> bit-exact; `aero-cold-overlap` `qc` / `nc` / `effc` at level 4
  (1.000e+00 / 1.000e+00 / 8.102e-01, one float32 ULP of cloud flipping
  `:4007`) and `nr` 1.261e-04 / `qr` 4.443e-05 at level 6 -> bit-exact in
  2.8.6, before 2.8.8 moved rain collecting graupel onto the one slab WRF's
  tables hold (the declared divergence above); `wp08-nusweep` `qr` 4.642e-06 -> 5.532e-07; `wp08-freeze` `nr` level 0
  4.006e-07 -> bit-exact; `aero-reduces-to-classic` level 6 bit-exact in `qr`
  and `nr`, which retired the last allowance.

  THE ALLOWANCES, ALL RETIRED, NONE EVER WIDENED.  The relative bound on
  `aero-reduces-to-classic` went 2.5e-03 -> 1.0e-04 -> 1.0e-05 with its `qr`
  entry deleted, and was retired at the 1.4.1 merge (level 5 `nr` 5.700e-06
  -> 4.146e-07); the dB bound went 1.0e-02 -> 1.0e-03 and was retired when
  the residual fell to 3.242e-05 dB; the near-cancellation bound, which
  replaced a bare SKIP of level 6 with 32 ULP of the entry value, was retired
  by the 2.8.6 accumulator rework.  The mechanisms closed on the way were
  found rather than tolerated: `module_mp_thompson.F:3490` overwrites
  `rho(k)` inside the condensation loop while `:3384-3388` had already frozen
  the rain moments from the pre-condensation density (pre-fix `qr` 1.915e-03
  / `nr` 1.922e-03 at level 5); WRF builds the rain mass and number
  sedimentation consumes at `:3237-3238` from the `:3193` tau+1 density for
  every level with rain and rebuilds them at `:3568`/`:3570` only inside the
  `:3501-3502` gate (`qr` 7.813e-05 -> 1.788e-07, `nr` 4.832e-05 ->
  5.700e-06); and WRF's terminal apply is `q1d(k) = q1d(k) + qten(k)*DT`
  with no FMA in the `gfortran -O2` oracle, which nvrtc used to contract.

  NO SURFACE ACCUMULATION MISSES ON ANY FIXTURE: `RAINNC`, `RAINNCV` and `SR`
  are bitwise identical to WRF on all 22 columns.  Earlier revisions of this
  register carried `rainnc`/`rainncv`/`sr` rows on two fixtures and noted
  they were one number seen three ways (`module_mp_thompson.F:1298`,
  `:1299`, `:1308`); the note was right and the rows are gone.

  Three preregistered matched woof/WRF mp=28 comparisons have run on an
  idealized, single-domain, doubly periodic case. Their declared outcomes
  were HOLD (`docs/public/validation/mp28-matched-trajectory.md`, V3 failed),
  INCONCLUSIVE (`docs/public/validation/mp28-shortwindow-gate.md`) and HOLD
  (`docs/public/validation/mp28-distribution-gate.md`, D1 failed). None
  certified under its preregistered rule. These are code-verification
  comparisons; no real-data matched mp=28 run with decay tables is
  recorded here, and no mp=28 forecast score against observations is
  published here. Such observation scoring would be validation.

  Separate self-consistency runs integrate 150 steps x 12 s and, in a
  second gate, 600 steps (7200 s) on a specified-BC convective domain
  with 0 non-finite values, 0 bound violations and 0 spec-zone ring
  violations. Finite and bounded does not establish agreement with WRF
  or observations. These numbers are published on the registry option under
  `extensions.column_oracle_evidence`, and
  `tests/test_physics_registry.py::test_mp28_evidence_matches_the_bound_the_adapter_gate_actually_applies`
  binds them to the gate that produced them.

  WHAT MOVED ON 2026-08-01, INCLUDING AGAINST GPUWM.  `aero-ice-koop` --
  published by this register, the registry, `docs/public/PHYSICS.md` and the
  evidence document as the port's LARGEST GENUINE PHYSICS GAP at `qi`
  1.612e-03 / `ni` 1.764e-03 / `effi` 5.093e-05 -- now measures 1.534e-07 /
  3.396e-07 / 1.886e-07 and is CLEAN.  THAT CLOSURE IS NOT A PHYSICS FIX AND
  MUST NOT BE REPORTED AS ONE.  No `.cu` or `.cuh` file changed for it.  The
  cause was this port's own oracle harness:
  `tools/thompson_wrf461_oracle/run_column_aero.F90` built the Exner
  function with `rd_over_cp = 287.0/1004.0`, while WRF's own `rcp` is
  `r_d/cp` with `r_d = 287.` and `cp = 7.*r_d/2.` = 1004.5
  (`share/module_model_constants.F` lines 19, 20 and 31) -- exactly `2/7`,
  and 4774 float32 ULP from what the harness used.  Under the wrong constant
  47 of the deck's 528 entry levels (first published as 40; re-pinned
  2026-08-03, owner-ratified -- the count is the host libm's float32
  `power`, not the deck's, and the correction note in
  `docs/public/validation/mp28-column-evidence.md` section 3.4 records
  both) carry a `(p, T)` pair that no float32
  `theta` inverts exactly, the adapter perturbed the entry pressure by up to
  15 ULP to recover the recorded temperature, and the perturbed pressure
  drove different microphysics on seven fixtures including this one.  The
  deck was regenerated with WRF's own constant and the residual went with
  it.  Homogeneous haze freezing was never measured to be wrong: the claim
  is WITHDRAWN, not repaired.  Re-derivable from the committed CSVs by
  `tests/test_physics_md_aerosol_claims.py::test_the_committed_deck_carries_wrfs_own_exner_constant`
  (528 of 528 levels invert under `r_d/cp`; 481 of 528 under
  `287.0/1004.0`), and recorded in full in
  `docs/public/validation/mp28-column-evidence.md` section 3.4.
  `aero-cloud-freeze-nc` lost its `effc` row and its `qc` fell to a
  third of the published value; `aero-ice-demott-idxin` lost its `qc` row.
  Against that, `aero-cold-overlap` is WORSE than it was published, and
  `wp08-freeze` / `wp08-nusweep` had never been published at all because this
  register and the registry counted 19 fixtures while the gate drove 22.

  They are also RE-MEASURED rather than only cross-referenced.
  `test_mp28_published_residuals_still_equal_a_live_adapter_measurement` runs
  all twenty-two fixtures through the shipped adapter on the device and
  rebuilds the published partition from the result, so a fixture cannot be
  quietly reclassified as clean and a residual cannot quietly drift.  That gap
  was real, not theoretical: with `aero-ice-koop` moved from the residual list
  to the clean list and its row deleted -- the exact shape of an overclaim --
  every one of the other 152 registry and builder assertions still passed and
  only this gate went red.  It fired again this wave, in the accurate direction:
  it is what caught the registry publishing residuals up to four orders of
  magnitude LARGER than the tree produces.  Exact comparison at the published
  `%.3e` precision is legitimate here because the measurement was repeated
  three times end to end on one RTX 5090 and every value was bit-identical
  across repeats; if that ever stops holding, the right response is to publish
  the spread, not to widen the comparison.

  AND THE PUBLICATION LAYER IS DERIVED, NOT RETYPED.  The registry gate above
  binds the REGISTRY to a live measurement; it never bound the prose.  Five
  waves of this port shipped at least one of this register,
  `docs/public/PHYSICS.md`, `docs/public/validation/mp28-column-evidence.md`
  or `CHANGELOG.md` quoting a residual, a count or a call graph the code no
  longer produced, and each time a different auditor found it.
  `tests/test_physics_md_aerosol_claims.py` now imports the adapter gate's own
  pinned partition (`_G3_UNEXCEPTIONED_CLEAN`, `_G3_GATED_CLEAN`,
  `_G3_RESIDUALS`, `_END_TO_END_BOUNDS`) and REBUILDS what the pages publish:
  the two clean counts are computed rather than compared against literals;
  each miss table's row SET must equal the gate's residual set, so a closed
  residual cannot be left behind and an invented one cannot be added; every
  published residual must be the gate's own number at `%.3e` and must name
  every quantity it misses on; the maturity label is read out of the registry;
  section 6.1's aerosol sensitivity is RE-RUN on the device (both 150-step
  forecasts) rather than cross-referenced; and section 3.4's Exner claim is
  re-derived from the committed CSVs.  Thirteen fault injections -- dropping a
  residual fixture, rounding a residual toward the gate, inflating the clean
  count, reviving the superseded Koop number, overstating the maturity label,
  reviving the closed MYNN snow deviation, drifting the sensitivity table --
  each turn the corresponding gate red.

  ONE CLAIM IN THE PORT'S OWN BRIEF WAS WRONG AND IS CORRECTED HERE.  The
  brief for this work stated that `287.0/1004.0` "appears nowhere in WRF".  It
  appears exactly once in the stock v4.6.1 tree, at
  `phys/module_fdda_psufddagd.F:1247`, inside a PSU FDDA nudging diagnostic.
  It is not the model's Exner function, nothing in the dynamics or the physics
  prep uses it, and the conclusion is unaffected -- but "nowhere" was not
  true, and a register that repeats an unverified negative is doing the same
  thing this entry exists to prevent.

  The full evidence statement, graded by how strong each class of measurement
  actually is, is `docs/public/validation/mp28-column-evidence.md`.

### D10. LES-nest inflow perturbation (designed WOOF-over-WRF extension, P3)

- **Status: designed.**  `woof/core/inflow_perturbation.py` seeds a
  one-way LES child's inflow transition with CPM-style cell-blocked
  potential-temperature perturbations of the child's rolling
  nest-boundary VALUE tables, relax-zone rows only, refreshed on the
  FORCE cadence at `NestCoupler.force`.  Every constant is pinned in
  `docs/superpowers/receipts/les/INFLOW-GENERATOR-ACCEPTANCE-V2.md`,
  registered before any perturbed integration existed; the measured
  need is the D90 receipt beside it (fetch > 55 z_i, contamination
  verdict CONTAMINATED on the shipped 250 m child).
- **What stock v4.6.1 ships instead**: `perturb_bdy` (Registry
  `registry.bdy_perturb`) perturbs U/V/T boundary TENDENCIES with a
  SKEBS or user-supplied stochastic pattern inside the stoch package.
  No cell-perturbation (Muñoz-Esparza CPM) path exists in the pinned
  Registry extracts available on this box; the full source-authority
  read remains P3 item 2 and is still open: this entry records the
  extension as designed, not a WRF-parity claim.
- **WRF-parity safety**: the four `inflow_perturbation*` keys are
  per-domain, TOML-only (no namelist spelling; a config using them
  cannot round-trip to a namelist, like per-domain `isfflx`), default
  OFF, and the OFF path is gated byte-identical to a build without the
  mechanism (acceptance G1).  The keys are preparation-inert for the
  prepared-cache identity (`PREPARATION_INERT_RUN_FIELDS`,
  `woof/ingest/prepared_cache.py`) and remain inside the experiment
  fingerprint and restart identity.

### D11. RUC LSM runs the DEFINED one-layer answer where WRF reads an uninitialised `ilnb`

- **WRF-native behaviour, and why it is not a behaviour**: `SFCTMP`
  declares `ilnb` at `phys/module_sf_ruclsm.F:1385` and never initialises
  it, then reads it at `if(ilnb.gt.1)` under `if(snhei.gt.0.)` on thin
  snow.  Inside WRF's column loop the local therefore carries the PREVIOUS
  COLUMN's snow-layer count, so the selected `tsnav` formula (which feeds
  snow compaction) depends on grid traversal order.  That is not a physics
  choice WRF made; it is an undefined read, and Fortran gives it whatever
  the stack holds.
- **What WOOF does instead**: every column is passed the DEFINED value
  `ilnb = 1` with `ilnb_chain = False` (`woof/core/ruc_runtime.py:109-126`
  `DEFINED_ILNB`, runner binding `:774`, receipt `:559-560`, published
  statement `:336-341`).  One is the defined answer rather than merely the
  safe one: a column with `0 < snhei < snth` has exactly one snow layer.
- **Why this is a deviation and stays one**: where WRF is undefined, WOOF
  implements the defined behaviour and documents the divergence, a standing
  rule, and the reason there is no forecast-mode switch back.  The bug chain
  exists only as an oracle-fixture affordance for reproducing one specific
  WRF run (`woof/core/ruc.py:8663` `ilnb_chain`, described at `:8717-8726`),
  and the resident device driver REFUSES it outright
  (`tests/test_ruc_device_column.py:571-586`): column i's seed is column
  i-1's answer, a genuine sequential dependence a data-parallel lane cannot
  reproduce and should not pretend to.
- **Public statement**: `docs/public/PHYSICS.md:601-603`.
- **Divergence-ledger status**: entry L2, class A, already the shipped
  default; no toggle exists and none should.  It earns no observation-battery
  arm, and the three-way comparison prices it implicitly wherever WRF's RUC
  runs.

### Registered deviation: non-mutating force coupling (Phase 5 external review #4)

- **WRF-native transaction and arithmetic**: every `med_force_domain`
  call couples the parent in place, couples the child in place, performs
  the transfer/boundary interpolation, uncouples the child, uncouples the
  parent, clears `first_force`, and resets child dtbc, in that order
  (`share/mediation_force_domain.F:111-206`).  The forward coupling
  factors and reciprocal uncoupling factors are formed in separate REAL
  loops (`dyn_em/couple_or_uncouple_em.F:117-182` and `:193-261`), then
  applied to the full-patch prognostics in separate multiply passes
  (`:270-348`).  Consequently the FP32 round trip is not algebraically
  inert: `fl(fl(x*m)*fl(1/m)) != x` for a nontrivial subset of values.
  Both participating domains' full-patch prognostics can therefore be
  perturbed at every force call even when feedback is zero.
- **Bundle-chain exposure over 12 hours**: the 60 s root step and
  `parent_time_step_ratio = 1,4,3,3` are the bundled namelist values
  (`namelists/namelist.input:2,32-33,53,57`).  WRF's grouped force loop
  calls once per parent step for each sibling edge
  (`frame/module_integrate.F:409-423`), giving these per-domain in-place
  couple/uncouple round trips:

  | domain | as child | as parent | total round trips |
  |---|---:|---:|---:|
  | d01 | 0 | 720 | 720 |
  | d02 | 720 | 2,880 | 720 + 2,880 = 3,600 |
  | d03 | 2,880 | 8,640 | 2,880 + 8,640 = 11,520 |
  | d04 | 8,640 | 0 | 8,640 |

- **woof registered improvement**: `force()` couples one full parent field
  at a time into the shared write-before-read arena scratch, pairs it with a
  full child field in a dead-RK scratch borrow, and feeds the full-extent
  `bdy_interp1` interface.  It never mutates either domain for coupling; only
  the rolling child boundary value/tendency frames persist.  The WRF
  round-trip perturbations are absent by construction.  The parent-read-only
  rule applies to `force()` only; Phase 5b's explicit `feedback_commit()`/
  `feedback_finalize()` transaction is the registered parent-mutation
  boundary.
- **Consequences for verification**: (a) a true nested WRF d01 is **not**
  bit-identical to its no-nest run, while woof's d01 is; (b) the N3/N4/N5
  bitwise ratchets certify woof's own stronger invariance and protect this
  deviation, not WRF identity; and (c) GPU-vs-WRF agreement was intended
  to be judged by matched-physics shadow gates at ensemble-envelope
  tolerances designed to accommodate this deviation class. No valid
  result for those shadow gates is recorded here. The only green N5S
  report had degenerate envelopes and was retired, as the historical
  2026-07-28 ratchet record below explains. The separate WRF ensemble
  comparisons in `docs/public/VERIFICATION.md` do not adjudicate these
  older nest-shadow gates.

### Registered N1.5 harness seam: dumped-WRF theta restoration (Phase 5, lane p5n15)

- **Scope**: the N1.5 candidate producer ONLY
  (`tools/wrf_instrumented/produce_nest_force_candidate.py`).  No
  production code path is affected.
- **Mechanism**: WRF couples the stored dry `t_2` directly
  (`dyn_em/couple_or_uncouple_em.F:283-286`); woof's state carries
  `thb/thp`, and its production coupling converts natively via
  `fl(fl(t_init+thp)-300)` (`woof/core/kernels/lbc_state.cu:575-582`).
  The producer must restore the dumped WRF `t_2` INTO the woof
  representation first, `thp := fl(fl(t_2+300)-t_init)`, and that
  inverse map is lossy: the composition collapses to
  `fl(fl(t_2+300)-300)`, up to half a theta-scale FP32 ULP
  (2^-16 K at theta ~ 300 K).  Amplified through hybrid-mass coupling,
  the field-nonlinear SINT limiter, and the near-cancelling
  `(SINT(parent)-nfld)/cdt` tendency division, this appears as
  4-8e-2 coupled-unit tendency errors confined to `state.t.*.tendency`.
- **Attribution evidence** (p5n15 numerics shadow, empirical): child
  `nfld` is bit-exact on all four theta sides (0/441,000 mismatches);
  substituting direct `t_2` coupling for the parent donor makes ALL
  882,000 theta VALUE and TENDENCY elements bit-identical; the
  production N3+ recurrence never executes the inverse map (rolling
  boundary targets are rebuilt from woof's native live state,
  `woof/core/nest.py:178-196`).
- **Registered treatment** (F18 amendment): theta tendency tables are
  compared at the FP32 successor scale
  `E = fl32(VALUE + fl32(cdt*TENDENCY))` (exactly the expression WRF's
  boundary consumer applies) under the UNCHANGED 1e-6 band (measured
  2.06e-7..2.59e-7); all other tables keep the raw registered
  comparator.  The producer additionally hard-fails if woof's own
  stored-table recurrence deviates from its raw SINT successor
  (measured <= 4.4e-10).
- **Adjacent findings closed in the same investigation**: (1) the
  west/south u/v tendency misses were a REAL production defect, woof
  one-sided both physical faces where WRF applies the four-term
  duplicated-halo weight at low faces and repairs only the high faces
  (`couple_or_uncouple_em.F:140-166`), FIXED at p5n15 head 80eebb4
  (kernel + mirror), after which all sixteen u/v tables are
  bit-identical to instrumented WRF; not a deviation.  (2) WRF's
  intermediate-grid `t/qv/ph` re-coupling
  (`force_domain_em_part2.F:285-300`) was investigated and REFUTED as a
  contributor: it is dead code in this case (`vert_refine_method=0`
  guard), and direct coupling through woof's SINT path reproduces
  every WRF theta table bit-for-bit.
- **Vanishes when**: the N1.5 instrument gains a lossless
  representation bridge (e.g. WRF-side dumps of gpuwm-native
  `thb/thp`), or N1.5 is superseded by a stronger oracle.

### D12. Configured initial-state theta bubbles (`[perturbation]`, WOOF-over-WRF extension)

- **Status: implemented.**  `[[perturbation.bubbles]]`
  (`woof/experiment.py`) declares warm bubbles in geographic
  coordinates; `woof/ingest/init_perturbation.py` evaluates them per
  domain and `woof/ingest/real.py::initialize_real` applies them ONCE
  to the final theta/qv columns before alpha/geopotential are formed:
  constant analyzed pressure and column mass, geopotential rebalanced,
  the `em_quarter_ss` convention (`module_initialize_ideal.F`; the
  port's own transcription is `woof/verify/cases/moist_bubble.py`),
  so the initialized state passes `hydrostatic_residual` unchanged.
  Real-data nest init re-ingests the source analysis per domain, so
  each domain that starts with the experiment evaluates the same
  geographic bubble on its own grid; delayed-start domains take no
  fresh bubble.
- **What stock v4.6.1 ships instead**: warm bubbles exist only in the
  idealized initializers (`module_initialize_ideal.F`); real.exe has no
  config-driven initial-condition perturbation, and no namelist can
  express this block, so `import-namelist` never emits it and a config
  using it cannot round-trip to a namelist.
- **WRF-parity safety**: default ABSENT; the absent path builds no
  applier and executes not one instruction of the module, the
  prepared state is byte-identical to a build without the feature, and
  the restart-identity payload omits the key so pre-feature experiment
  fingerprints are preserved.  Present, the block binds the fingerprint
  and restart identity and is honored on two routes: the experiment
  runtime path (`woof run` / `woof ingest`, inside `initialize_real`
  with the geopotential rebalanced) and the prepared domain-tree
  forecast runner (applied to the restored sealed states before
  physics/diagnostics, then rebalanced at the held pressure: each
  warmed layer thickens by its ratio of new to old moist theta and the
  geopotential re-integrates up the column, the state
  `init_moist_balanced` builds for the idealized bubble; the sealed
  caches stay the pure analysis and both arms of an A/B can share one
  preparation).  Every other route (the prepared-cache front doors and
  the single-domain prepared runner) refuses it by name.  Both routes
  also refuse, before integration, a layer heated past the top of the
  radiation's temperature table and `rh_preserve` vapour past a
  measured limit (`init_perturbation.py`).  Application stats
  (per-domain cells touched, max theta delta, qv adjustment under
  `rh_preserve`) are published to `initial-perturbation.json` before
  integration starts.

## Divergence ledger v1 (the observation-battery promotion register)

Registered 2026-08-03 with the observation-verification battery
(`docs/superpowers/specs/2026-08-03-obs-verification-battery.md` section 5.2).
The D-entries above are deviations WOOF already ships.  THIS register is the
other half: the places WOOF may deliberately diverge, each with the state of
its evidence, so that a candidate cannot become a default by accumulating
confidence instead of receipts.

Its executable half is `woof/physics_mode.py`: the same entry ids, the same
faithful -> patched edges, and the `physics_mode` axis that resolves them.
Neither half may move without the other; `tests/test_physics_mode.py` pins the
entry set, the frozen patch-set membership, and the gates.

**Rewrite, don't append.**  An entry's verdict is EDITED here when its
receipts land; a superseded claim is replaced and said to be replaced, never
left standing beside its successor.  A promoted patch changes the WOOF
default and its entry becomes a numbered D-entry above.

**No observation-battery verdict is recorded here.** This ledger began
on 2026-08-03; its later edits, including the 2026-09-23 L1 correction,
do not record a completed battery. The promotion cells below read
UNDECIDED except L5, which DID NOT QUALIFY for battery arms. This is the
state of this battery record, not a claim that no other run has been
scored against observations. Specific observation comparisons are
reported separately in `docs/public/VERIFICATION.md`.

### Class A. Already the shipped default (no toggle exists or should)

| entry | state | battery consequence |
|---|---|---|
| **L1** Thompson rain-presence gate | The gate is WRF's conjunction and compares a mass concentration: `woof/core/kernels/thompson.cu:450-452`, landed `cb765336` (2026-08-01) with `5e4af4e3` beside it.  The roadmap listed this as a patch candidate; it was overtaken by events, and the register says so rather than quietly dropping it.  The one residual class that commit enumerated and left standing (`qr <= R1 < qr*rho` with `L_qr` true) was documented at the `wp08-freeze` row of `docs/public/PHYSICS.md`, reconciled to this kernel on 2026-08-03, and CLOSED for mp=28 on 2026-09-23: the mp=28 rain evaporation carries `L_qr` to the fallout as a zero reference density (and the `:3568` floor as a negative one) and the adapter launches the `_with_presence` entry points (`thompson.cu`); mp=8's entry points keep the stand-in. | No arm: there is one mp8 path. |
| **L2** RUC `ilnb` uninitialised read | D10 above. | No arm, and never one: a forecast-mode toggle back onto an undefined read would be bit-exactness to a bug. |

### Class B. Toggleable candidates, the battery's arms

| entry | key (mechanism-named) | faithful -> patched | evidence today | promotion |
|---|---|---|---|---|
| **L3** gray-zone PBL | `bl_pbl_physics` (existing selector; no new key) | 1 (YSU) -> 11 (Shin-Hong 2015 scale-aware) | 1.5.2 headline; CPU authority max-ULP-0 against the byte-frozen `module_bl_shinhong.F`; partition ladder inside registered Honnert-envelope bands (`docs/public/receipts/grayzone/PHASE1-SHINHONG-20260803.md`); maturity implemented-unverified.  At dx 3000 m the scheme sits above its gated ladder, which is stated in the arm's registration and is exactly what an obs referee is for.  **Route:** the composition an L3 arm resolves to is registered as `thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1` (wrf-matched-run-candidate, template_only, HRRR-only), which is what makes the arm runnable: its three refusing gates (emission physics, root-preparation profile, certified hierarchy slice) now admit it, each by an enumerated admission keyed to that registration rather than a widened rail. | UNDECIDED |
| **L4** 6th-order filter off the moist scalars | `moist_mix6_off` (WRF's own spelling) | `.false.` -> `.true.` | WRF v4.6.1 `Registry/Registry.EM_COMMON:2889` declares it verbatim: `rconfig   logical  moist_mix6_off   namelist,dynamics  max_domains  .false. rh  "moist_mix6_off"  "de-activate 6th-order horizontal filter for moisture"`.  It gates one call (`sixth_order_diffusion` on the moist array, `dyn_em/module_em.F:1421`, reached from `dyn_em/solve_em.F:2230`) so theta and the WRF `scalar`-package tracers keep their own filters.  This is therefore a **configuration-policy patch**: the WRF arm could be configured to match, and the battery tests it as an WOOF-DEFAULT candidate rather than as a capability.  Precedent: D-L4 ratified diff6 default-OFF in LES configs (`docs/superpowers/specs/P6-LES-DECISIONS-RATIFIED.md:35-39`).  The roadmap's "Sc-deck scales" framing has **no receipt in this tree** (verified: `diff_6th` and stratocumulus never co-occur; the Sc-deck finding is a SASE result, `woof/verify/sase_ref.py:1280-1292`), so the entry is registered on WRF-community evidence plus D-L4 and is measured at 3 km. | UNDECIDED |
| **L5** SASE closure | `bl_pbl_physics` -> 900 plus six mandatory companions (`docs/public/PHYSICS.md:718-723`) | 1 (YSU + `km_opt` 4) -> 900 | implemented-unverified on the WRF-conformance ladder (no WRF counterpart; this does not rule out code verification against its own reference implementation or validation against observations); met 2 of 7 frozen bars on its reference case; its first real-data forecast failed the health gate at 1 h. | **DID NOT QUALIFY.**  Registered entry gate: SASE earns battery arms only after one full 24 h battery-shape integration completes with `status: PASS`.  Until then it is scored on zero cases and reported as "did not qualify" -- accurate, and cheaper than crashed arms.  It is also not a bit-flip, so it enters as its own overlay and is deliberately not a member of patch-set v1. |

### Class C. Knob exists, dormant (not in the battery suite)

| entry | state | when it wakes |
|---|---|---|
| **L6** Morrison rimed-ice identity | `morr_rimed_ice: int = 1` (hail, = the WRF default) with validator, import round-trip and receipt binding (`woof/config.py:185-189`, `:1714`). | Only if an mp10 battery arm is ever registered; hail-vs-graupel is then its ablation pair. |
| **L7** gray-zone partition (SASE-side, Phase 0+) | Nothing to toggle at this tip. | Enters by its own receipt when it lands. |

### Patch-set v1, and how an arm is spelled

`patchset = "v1"` is **{L3, L4}**: the two toggleable entries that are
bit-flips with a real counterfactual at 3 km.  A registered patch-set version
is FROZEN: a later ledger entry gets a NEW version beside it, so a receipt
reading `patchset = "v1"` keeps meaning the vector it meant on the day it was
written.

| arm | `[experiment]` overlay |
|---|---|
| F (WOOF-faithful) | `physics_mode = "wrf-faithful"` |
| P-L3 | `physics_mode = "arwen-patched"` + `patches = ["L3"]` |
| P-L4 | `physics_mode = "arwen-patched"` + `patches = ["L4"]` |
| P-all | `physics_mode = "arwen-patched"` (patches omitted) |

Single-patch arms are the promotion evidence; P-all is reported and is never a
promotion basis, because interaction credit is not assignable to a patch.  A
case configuration carries no arm of its own: it omits every governed key and
the overlay writes them, so one case file serves every arm and the loader
refuses a second author rather than merging one.

The promotion rule itself (R1-R4, the Wilcoxon direction, the twin band, the
guardrail list) is registered in the battery spec section 6.5 and is **frozen
only by the owner's ratification**: an agent-invented gate is not a gate
until the owner rules.

## Seam-closure ratchet epoch (Davies clock bind, 2026-07-28): historical record

This section records the epoch at fix commit `439bc083`. The engine
has changed since, including the 2026-09-23 fallout changes in D9k.
These ratchet values measure that historical build, not the current
engine. This register records neither ratification of the two verdicts
marked pending below nor a later valid N5S regeneration.

The batched end-of-Wave-1 ratchet regeneration executed with the Davies
clock bind (branch `fix/davies-clock-bind`, fix commit `439bc083`):

- **Fresh N0 probe receipt** (`out/rungs/n0-alloc-probe-r3.json`,
  `woof check configs/real74_4dom.toml --alloc --reserve-gib 2`): all
  three legs TRUE, pool used peak 15,459,084,288 B <= alloc estimate
  21,421,799,373 B <= budget 30,352,080,896 B.  Supersedes the stale
  `n0-alloc-probe-r2.json` calibration (2026-07-16, pre-arena-sharing;
  its margin against the grown Wave-1 estimator had inverted to
  -1,202,489 B per the ring-lane ledger).
- **d01 anchor epoch `out/real74-t7-final-r3`** (replaces the retired
  `-r2` role; `-r2`/`real74-t7-final`/`cardinal` bytes encode pre-bind
  semantics and remain historical evidence only): d01-only 75-min run
  at the fix tip, dual-run byte-identical
  (`real74-t7-final-r3/anchor-r3-dualrun.json`); 13:00 frame sha256
  `b4ee56f6…12616e00` (old r2: `7f588501…`), 12:00 frame
  `42d0dae9…eba7bc78`.  Triple identity re-proven at the bound epoch
  (`out/rungs/N3-runA/d01-triple-identity.json`): anchor == nested
  straight d01 == d01-only ancestor control, byte-for-byte.
- **N3 regenerated** (`out/rungs/N3-runA`, report PASSED 12/12;
  ratchet `out/rungs/N3/manifest.json`, experiment fingerprint
  `13d3d884…` [old `8a1c9117…`, input catalog gained the committed
  SOILHGT assets and the Wave-1 output inventory grew], d02 frames
  12:00 `6a656c5e…` … 13:15 `fe1e97e5…`): straight + ancestor-control
  arms dual-run byte-identical across two full rung executions
  (`N3-runA/n3-dualrun.json`, 10/10 frames); restart split
  bit-identity PASSED (v5 + `root_external_lbc_clock` identity
  round-trip at production scale); values moved vs the 2026-07-18
  epoch because the trajectory legitimately changed under ring-MP +
  SEAM A + init-surface + diff6 + Thompson + Davies bind (e.g. MSLP
  corr 0.9936 -> 0.9996, T850 RMSE 0.0741 -> 0.0942 K); the
  d02_refl_10cm_structure verdict is lane-adjudicated
  (`out/rungs/N3-verdicts-r3.json`, evidence
  `N3-runA/refl_structure_comparison.png`, amplitude offset 58.1
  dBZ/7357 cells vs prior ratified 57.2/7177) pending controller
  ratification.
- **N4 regenerated** (`out/rungs/N4-runA`, report PASSED 9/9; ratchet
  `out/rungs/N4/manifest.json`, fingerprint `8ee7107c…` [old
  `2df55493…`]): d01-vs-anchor and d02-vs-regenerated-N3 bitwise
  ratchets TRUE; straight arm dual-run byte-identical (14/14 frames,
  `N4-runA/n4-dualrun.json`); d03 statistics against the reference
  frame changed (MSLP corr 0.9526 -> 0.9989). That reference is one
  WRF frame at +1:15 from the original run, whose physics differ from
  this configuration (D3c and `woof/verify/nest_gates.py`,
  `CHILD_REFERENCE_FRAMES`). This is a regression signal for that
  build, not evidence that either epoch is more accurate; accuracy
  needs observations. Both structural verdicts were
  lane-adjudicated (`out/rungs/N4-verdicts-r3.json`, evidence
  `N4-runA/d03_refl_structure_matched_comparison.png`, d03 interior
  w_max 1.83 m/s) pending controller ratification.
- **N5/N5B/N6**: no green recorded evidence exists at the old epoch on
  this box (no N5-report; N5B partial members m01-m03; N6 never run);
  their producers consume the regenerated N3/N4 manifests at next
  execution.
- **N5S, one green pre-bind report EXISTS and is RETIRED on paper**
  (correction 2026-07-28, adversarial-review finding; the original
  epoch record wrongly claimed no green N5S evidence existed): the
  main repo retains `out/n5s-shadow-report.json` (metric
  `N5S_matched_physics_wrf_shadow`, `passed: true`, generated
  2026-07-18T18:52:12Z, evaluator_commit `55e79f23`) together with its
  gpu evidence chain `out/n5s-gpu/` (`N5S-gpu-run.json`,
  `n5s-gpu-evidence.json`, `n5s-run-artifact.json`,
  `n5s-preregistration.json`, `restored_input_sha256.txt`).  That
  report is wholly VACUOUS, `all_envelopes_degenerate: true`, its
  pass auto-accepted under the f28 degenerate-envelope adjudication,
  and was never consumed by any N5-report; `_load_gate_report`
  re-scores with a `score_binding_sha256` refusal, so stale
  consumption fails closed.  Like the `-r2` anchor and the r2 N0
  receipt, it predates the Davies bind and encodes the retired
  unbound root-clock semantics (and, via piece 3, a shadow the
  regenerated N5S runner would no longer reproduce): it is SUPERSEDED
  as evidence, retained in place unmodified as historical record
  only, and must not be cited by any bind-era consumer.  N5S
  regeneration rides its next execution against the regenerated N3/N4
  manifests, with the now-bound restored-model builder.
- Every ratchet value above is derived from reference bytes produced
  by the fix-tip runs, never hand-tuned.

### Davies bind A/B measurement (2026-07-28, `out/rungs/davies-ab/`)

Root-domain-only production pair, 2 h at 15-min cadence, base
`a8b14b4c` (pre-bind) vs fix tip `439bc083`, identical t=0 bytes
(cross-checked); BOTH sides dual-run byte-identical
(`ab-dualrun.json`), zone-resolved deltas in `ab-zone-report.json`:

- **Specified ring**: T/MU/QVAPOR byte-zero at every lead, the final
  ring overwrite was already time-aligned away from interval seams,
  exactly as the dossier narrowed the seam.  (The staggered-U ring
  statistic in the JSON is projection-contaminated by the first
  interior face and is not a ring deviation.)
- **Relax rows (2-4)**: immediate quasi-steady offset, the closed
  one-step phase lag: mean |dT| 1.9e-3 -> 3.1e-3 K, mean |dMU| ~0.19 Pa
  (the recorded F20 one-step magnitude), mean |dU| 4e-3 -> 7e-3 m/s;
  local maxima 0.08-0.32 K / 0.29-0.74 m/s / 0.9-4.7 Pa in
  frontal/jet gradients.
- **Interior**: the structured phase correction seeds growing local
  differences while means stay small: |dT| max 0.19 K at +15 min ->
  ~2.6 K local at +105 min (mean 3.5e-3 K at +120 min); |dMU| max to
  ~36 Pa local (mean 0.31 Pa); |dQVAPOR| max ~1e-3 kg/kg local.
  A one-step phase change can grow during convective evolution, but
  this pair has no perturbed control that establishes an expected
  magnitude. The WRF recompilation control in
  `docs/public/validation/mp28-matched-trajectory.md` demonstrates
  growth on a different idealized case, not a bound for this one.
  These numbers show the correction changes the run; they do not
  establish whether either run is more accurate against observations.
- **Cost**: byte-inert on timing, 0.335 vs 0.335 wall s per sim-min
  (both sides, dual-run).
## EXPERIMENTAL ensemble / perturbation / DA-cycle route (v1.2)

Registered here because a consumer can reach this machinery from the repo
without ever seeing a finished manifest, and everything below is a scope
cut rather than a mechanism transcribed from an oracle. **Nothing in this
section is on a certified forecast path**, no product it draws has been
calibrated against a verification archive, and every manifest, receipt and
PNG it writes carries an `experimental` stamp.

- **Entry points, and only these**: `tools/ensemble_forecast.py` (`run`,
  `cycle`, `bench`, `status`), `woof enprod`, and
  `tools/da_synthetic_cycle.py`. `woof run` reaches none of it.
- **Authority**: no WRF mechanism is transcribed. The perturbation
  construction is a prescribed isotropic Gaussian random field, the
  probability-matched mean follows Ebert (2001) with the every-*M*-th
  pooled selection described in Flora et al. (2018) fn. 1, and the
  neighborhood-maximum ensemble probability is the standard
  max-in-neighborhood-per-member-then-ensemble-fraction order.
- **Scientific non-goals of `woof.da.perturb`**, stated before a run and
  repeated in every returned provenance dict: no mass balance (`mu'`
  untouched, no hydrostatic re-balance); no wind balance (the `u`/`v`
  increments are neither non-divergent nor in geostrophic/gradient balance
  with the temperature increment); one shared, unperturbed lateral
  boundary file for all members, so ensemble spread decays toward the rim
  by construction; lateral rim taper only, no vertical taper; no surface,
  soil, or physics-parameter perturbation and no `w`, `mu'` or
  hydrometeor perturbation; no flow dependence.
- **Registered artifact: vertical FFT periodicity.** The draw is periodic
  on every axis. Horizontally the rim taper hides it. Vertically nothing
  does: on a periodic column level 0 and level `nz-1` are ONE interval
  apart, so their correlation is approximately `exp(-1/(2 Lv^2))` --
  0.983 at the admitted quarter-column cap (`nz=24`, `Lv=6`), not the
  `exp(-2) ~ 0.135` earlier documentation claimed by confusing the seam
  with the maximum circular separation. The cap bounds the HALF-COLUMN
  correlation (0.270 at that setting, itself above `exp(-2)` because
  periodic images contribute covariance) and does not control the seam.
  Every draw reports its exact figures in
  `provenance["fields"][i]["vertical_wrap"]`, computed from the applied
  filter.
- **Registered scope cut: horizontal scale admission.** A prescribed
  length scale is admitted only up to `span/(2*pi)`, because the
  documented radial-spectrum peak `k = 1/L` is resolved only when
  `1/L >= 2*pi/span`. The previous quarter-span limit admitted
  configurations whose peak fell below the domain's lowest nonzero
  wavenumber (measured `peak * L = 2.356` at the old cap on 32-, 64- and
  128-point domains).
- **Determinism scope**: "byte-identical given the seed" holds on ONE
  software and device stack. The white noise is drawn on the host with
  NumPy Philox and hash-stamped, so `noise_sha256` is machine-independent;
  the filtered field is not, because cuFFT and pocketfft round
  differently, and the provenance records which FFT backend ran.
- **Ensemble-product NaN policy**: `mask` (default) excludes non-finite
  member values from every reduction at that point, shrinks the
  denominator with them, leaves a point with no finite member blank, and
  records the resulting coverage in the provenance and on the panel;
  `refuse` fails the product naming the members. The NMEP voting roster is
  fixed by the point's own data so the denominator does not move with the
  neighborhood radius and the advertised monotonicity is exact.
- **Ensemble-product roster override**: `--accept-status` beyond
  `DONE,complete` triggers a frame-inventory check of every admitted
  member against the manifest before anything renders, and stamps the
  override on every panel.

## Attribution defects in this repository's own history (2026-07-26)

This register exists so a provenance claim is never re-derived from memory.
Attribution is a provenance claim, so known defects in it are recorded here
rather than only in a lane's tracking document, which may be retired.

Every commit produced under agent delegation on this project must end with the
trailer
`Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
One commit does not:

| Commit | Subject | Defect |
|---|---|---|
| `ab60dc8` | `feat(noahmp): route sf_surface_physics=4 to a real runner and admit it` | no trailer at all; `git log -1 --format='%(trailers:only=true)' ab60dc8` prints empty |

**Not repaired, deliberately.**  The omission was found after the branch
(`codex/mynn-noahmp-ruc-cuda`) had advanced past that commit and other lanes
had built on it, so `git commit --amend`/rebase would have rewritten history
shared with concurrent work.  The lane flagged it instead of rewriting shared
history.  Recording it here is the repair: the commit's authorship is not in
doubt, the trailer is simply absent, and this row is what prevents a later
reader from treating the absence as evidence of an unattributed change.

Verification, on any checkout of the branch:

    git log --format='%h %s%n  trailers:[%(trailers:only=true,valueonly=true)]' -12

The narrative context of that commit -- Noah-MP's runtime admission -- is in
`docs/wrf_noahmp_runtime_admission.md`, and the same fact is duplicated in
`docs/mynn_noahmp_ruc_completion_plan.md` under "Provenance and process
record".
