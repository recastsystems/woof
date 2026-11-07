# Old Kain-Fritsch (cu_physics = 99) WRF v4.6.1 column oracle

The numerical authority for `woof/core/kernels/kf_old.cu`.  It drives the
byte-unmodified public `phys/module_cu_kf.F` through its own entry point,
`KFCPS`, one column per process, and records every output KFCPS writes.
`tests/test_kf_old_wrf461_parity.py` grades the GPU kernel against the
recorded fixture at zero differing words.

## Running it

    bash tools/kf_old_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR \
        [WRF_PRODUCTION_OBJECT]
    cp BUILD_DIR/kf-old-wrf461-oracle.npz tests/data/kf_old_wrf461_oracle/

`WRF_SOURCE_ROOT` is a WRF v4.6.1 tree (commit d66e442).  `build.sh`
refuses to run unless `phys/module_cu_kf.F` and
`share/module_model_constants.F` match the sha256 pins it carries.
`WRF_PRODUCTION_OBJECT` is optional: the `phys/module_cu_kf.o` of a
configure/compile build of the same tree.

`select_cases.py` chooses the wide-family columns (below) and rewrites
`selected_wide.json`; it needs a finished `build.sh` run in the same
`BUILD_DIR`.

## What is compiled

Two WRF files, nothing restated: `module_cu_kf.F` and the constants module
the cumulus driver takes `XLV0..SVPT0, CP, R_D, G, EP_2` from.  The scheme
USEs only `module_wrf_error`, which `stub_wrf.F90` stands in for (message
buffer, logging and abort; a fatal prints `WRF-FATAL: <message>` and exits
7, and `build.sh` records it as that column's outcome).

The harness (`run_kf_old.F90`) gives KFCPS a 1 x nz x 1 tile with
`kts=1, kte=nz, kme=kde=nz+1`, exactly the bounds
`module_cumulus_driver.F:958-989` passes, and calls it with the driver's
argument list including `UDR_KF..TIMEC_KF` and `KF_EDRATES`.

## Three arms, one answer

| arm | object | role |
| --- | --- | --- |
| A | `module_cu_kf.F` at `-O0` | the reference the parity test grades |
| N | the same source, `-finit-real=snan -finit-integer=-77777 -finit-logical=true` | every uninitialised local poisoned |
| B | WRF's own `phys/module_cu_kf.o` from the strict build (`-O2 -fno-fast-math -ffp-contract=off`) | the object WRF actually runs |

On all 256 fixture columns the three arms agree word for word, including
on which columns (none) reach a WRF fatal.  Arm N agreeing is what makes a
zero-initialising port gradeable at 0 ULP: the scheme reads several locals
before every path is sure to have set them (`AINCM1` in DO 166, `ISTOP`
on the `GOTO 255` path, `RL`, `EE1/UD1`), and N proves no graded output
depends on what the stack held.  Arm B agreeing is what makes the `-O0`
reference the model WRF runs rather than a debug build of it.  The
undefined-symbol receipt shows both objects call the scalar `expf`,
`logf`, `powf` and `lroundf` and nothing from libmvec.

## The cases (256)

* 216 curated columns: eight regimes (deep tropical, deep cold-top,
  mid-latitude, high terrain, capped/shallow, stable, dry, cold season) at
  40, 49 and 60 levels, nine draws each.
* 24 columns from the two real HRRR soundings `tests/data/kf_real74_12z_columns.npz`
  already carries for cu=1, perturbed.
* 16 wide-family columns: `select_cases.py` drew 30,000 columns with
  every sounding knob independent and wide, ran each through a `--coverage`
  build, and kept a candidate only if it executed a line of
  `module_cu_kf.F` no earlier case had.

Each column is crossed with the driver knobs KFCPS reads: DT from 20 s to
600 s, DX from 6 km to 30 km, STEPCU 1-10, CUDT, the adaptive-step flag,
the species flags of five WRF moist packages (passiveqv, Kessler, WSM3,
six-class, P3) and the NCA hold (cold start, held, exactly 0.5*DT, an
expiring timer).  107 curated and most wide columns convect; 21 are held.

Coverage (`kf-old-coverage.txt`): 1222 of the 1324 lines gcov counts as
executable in `module_cu_kf.F` are executed.  What is not:

* diagnostics only -- the ISTOP print block (:1884-1931) and FORMATs;
* `kfinit` (:2583-2664), which the driver calls at initialisation, not
  per step;
* unreachable by construction -- `LFS < KLCL` (:1381, :1415; LFS starts
  at KSTART+1 >= KLCL+1), TPDD's `RH < 1` tail (:2561-2575; the only call
  passes 1.0), the impossible species combination (:2022), a column with
  no 60 hPa layer below its top (:580);
* the three WRF fatals (:1586, :1617, :1951), the iteration cap
  (:1754), `FABE = 0` (:1761), `TU10 = TVQU` exactly (:881-884),
  `QICE < 0` after evaporation in TPMIX (:2425-2426), and an LCL above
  every level (:623) -- none was reached by 30,000 wide draws.

Five wide candidates were REJECTED rather than kept (`rejected_ill_posed`
in `selected_wide.json`).  In each, TPMIX's secant step divides zero by
zero for an extreme parcel (`module_cu_kf.F:2381`, trapped with
`-ffpe-trap=invalid`), and once a NaN is in the column the answer depends
on the build: arm B orders MIN/MAX operands differently from arm A, and
arm N differs from A through stack contents.  Such a column has no single
WRF answer to grade a port against.  The port runs the same arithmetic,
so a forecast column that reached it would carry the NaN exactly as WRF's
does, and woof's non-finite tendency gate stops the run.

## gfortran lowering the port reproduces

* `AMIN1/AMAX1` compile to `minss/maxss` with the first written operand as
  the destination, so `AMAX1(A,B)` is `(A > B) ? A : B`; with one literal
  operand the literal takes the destination whichever side it is written
  on.  Measured with a probe on this toolchain (gfortran 13.3.0), and it
  only matters for signed zeros and NaNs.
* `EXP(-4.5)` in PROF5 is folded at compile time to `0x3C360282`, the
  same word the reference `expf(-4.5)` returns, so the port's runtime call
  matches it.
* `NINT` on a default REAL is `lroundf`; `REAL**REAL` is `powf` (no
  integer or 0.5 exponent appears in the file); `SQRT` is correctly
  rounded on both sides.

## module_cu_kf.F against module_cu_kfeta.F (stage 1 of the port)

The two files share an ancestry and almost no code: `diff` reports 5,148
differing lines of 2,669 and 3,303 (the full unified diff is in the lane
receipts).  Routine by routine:

| old KF (module_cu_kf.F) | KF-eta (module_cu_kfeta.F) | relation |
| --- | --- | --- |
| KFCPS :13-331 | KF_eta_CPS :29-531 | same W0AVG running mean (:191-219 vs :219-249, including the adaptive-step form); old holds while `NCA > 0.5*DT` (:235), KF-eta while `NCA >= 0.5*DT` (:410); old has no trigger options (KF-eta's `trigger` 2/3, :253-398), no shallow branch, no CUTOP/CUBOT; old folds hydrometeors by `F_QI` (:1983-2023) where KF-eta folds by `F_QS` |
| KFPARA :334-2082 | KF_eta_PARA :533-2690 | rewritten: old builds a 60 hPa source layer from the surface up (:574-581), KF-eta a 50 hPa layer with a shallow branch (:746-880); old's updraft thermodynamics call TPMIX/DTFRZNEW/ENVIRTHT, KF-eta's call the table routines; old's downdraft starts at the minimum saturated theta-e above KSTART and runs TPDD, KF-eta's uses TPMIX2DD; both close on 90% CAPE removal by the AINC iteration |
| TPMIX :2308-2462 (secant iteration on Buck's formulas) | TPMIX2 :2695-2815 (KF_LUTAB bilinear lookup) | different algorithms |
| DTFRZNEW :2151-2261 (Buck liquid/ice weights, RATIO2) | DTFRZNEW :2817-2861 (simplified freezing) | different algorithms |
| ENVIRTHT :2464-2508 (liquid, ice and mixed branches) | ENVIRTHT :3044-3082 (liquid only, ALU table log) | different algorithms |
| TPDD :2518-2580 (iteration) | TPMIX2DD :2977-3042 (lookup) | different algorithms |
| CONDLOAD :2085-2148 | CONDLOAD :2863-2928 | three differences: `1.E-10` vs `1.E-8` guards in RATIO3/RATIO4, `WAVG=(..)/2.` vs `0.5*(..)`, and KF-eta's `IF(ABS(WTW).LT.1.E-4)WTW=1.E-4` floor |
| PROF5 :2275-2305 | PROF5 :2930-2975 | identical apart from continuation whitespace |
| kfinit :2583-2664 | kf_eta_init :3084+ | KF-eta also builds KF_LUTAB |

Driver side: `module_physics_addtendc.F` zeroes held KFSCHEME tendencies
once `NINT(NCA/DT) .le. 0` (:2155) and KFETASCHEME ones at `.le. 1`
(:2201), so the old scheme applies its tendencies one step longer.

**Reuse map.**  PROF5 is the only routine the Fortran shares, and kf.cu's
`kf_prof5` is not a WRF-order transcription of it (CUDA `expf`, and
`-0.5f*y*y` where the Fortran divides by -2), so kf_old.cu carries its
own.  What cu=99 reuses from cu=1 is the infrastructure: the
lane-interleaved global workspace and tile launcher, the W0AVG carrier and
its restart/tile-seam handling, the `KainFritsch` adapter (one class, a
`scheme` row), the NCA hold and RAINC/PRATEC driver machinery, and the
kernel loader.  No KF_LUTAB is involved: the old scheme has no table.


## Real native forecast replay

The retained generated fixture above is distinct from a real forecast capture.
`instrument_native.py PRIVATE_ROOT` adds opt-in entry/exit stream writes to a
private WRF v4.7.1 source copy at `PRIVATE_ROOT/WRF-4.7.1`. It verifies the
original source pin before editing. Set `WOOF_KF_CAPTURE_DIR` in that native
run, then grade the completed capture directory in a separate strict process:

    GPUWM_WRF_EXACT=1 python -m tools.kf_old_wrf461_oracle.replay_native \
        CAPTURE_DIRECTORY RECEIPT.json

The big-endian versioned captures bind the original column inputs and every
returned tendency, running mean, precipitation/timer word and optional
entrainment diagnostic. Sampling selects columns with eligible NCA, every
37th i and 43rd j, on the first call and every 240th model step. Held-column
preservation is checked separately by the adapter and lifecycle tests.
`replay_native.py` refuses an empty capture directory or an incomplete record.
Its JSON receipt records each capture's hash, location, model step, selector
flags, native activity and precipitation, and the differing-word count for
every returned field. It separates captured native output words from the
one zero-fatal status witness per successfully returned call; the fatal
status is inferred, not a native output argument. The arithmetic policy
and WRF_EXACT setting are recorded separately. Keep a compressed archive
of the small captures alongside the receipt so the proof can be replayed.

`GPUWM_NUMERICS=fast GPUWM_WRF_EXACT=0 python -m
tools.kf_old_wrf461_oracle.time_stage CAPTURE_DIRECTORY RECEIPT.json`
measures one paired CU99/CU1 batch at default arithmetic. It repeats the
first-call native columns across 300 by 300 horizontal points and records
seven warmed, interleaved CUDA-event samples per scheme (configurable with
`--samples`). Use `--ktau N` to repeat a later native capture instead. The
receipt binds the capture step, number of raining native columns,
arithmetic mode and assembled sources.
This is a column-stage measurement, excluding driver composition,
health readback, transfers and forecast IO. CU99 retains its optional
`kf_edrates` setting; CU1 has no corresponding output profiles. This timing
must not be described as whole-forecast throughput or a convective-workload
benchmark when those captured columns do not convect.

CU99's fast default uses CUDA's standard `expf`, `logf` and `powf` for the
same WRF equations and searches. `GPUWM_NUMERICS=strict` (also implied by
`GPUWM_WRF_EXACT=1`) selects the native-matching scalar helpers. Its compile
define belongs to the compiled-image cache key. Strict module replay proves
that mode; it is not a bitwise claim for fast numerics.
