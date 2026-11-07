# GFS PBL oracle, WRF v4.6.1 (bl_pbl_physics = 3)

The WRF words the GFS nonlocal PBL port (`woof/core/kernels/gfs_pbl.cu`,
`woof/verify/gfs_pbl_ref.py`) is graded against.

## Build and run

```
bash tools/gfs_pbl_wrf461_oracle/build.sh $WRF_SOURCE_ROOT $L/oracle-build
python tools/gfs_pbl_wrf461_oracle/make_cases.py $L/oracle-run
for f in $L/oracle-run/cases-*.in; do $L/oracle-build/run_bl_gfs $f ${f%.in}.out; done
python tools/gfs_pbl_wrf461_oracle/compare.py --impl ref $L/oracle-run/cases-*.out
GPUWM_NUMERICS=strict python tools/gfs_pbl_wrf461_oracle/compare.py --impl gpu $L/oracle-run/cases-*.out
```

`build.sh` compiles the byte-unmodified `phys/module_bl_gfs.F`,
`phys/module_gfs_machine.F`, `phys/module_gfs_physcons.F` and
`share/module_model_constants.F`, preprocessed with WRF's own ARW cpp line,
at the flags of the scalar WRF referee build (`configure.wrf` FCOPTIM:
`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`).  It refuses a
source whose SHA-256 is not the v4.6.1 file's and an object that calls a
vector libm entry.  On the sprint box (gfortran 13.3.0, glibc 2.39) the
object's only libm calls are `pow` and `powf`; its preprocessed source
differs from the WRF build's own `phys/module_bl_gfs.f90` in comment lines
only.

`run_bl_gfs.F90` calls `BL_GFS` exactly as `module_pbl_driver.F:1406-1424`
binds it and records, for every call, the complete input record and every
output: the six tendencies, UST, WSPD, PBLH and KPBL.  Between calls it
advances the state with the returned tendencies, so calls 2-4 see a state
the scheme produced.

## The columns

`make_cases.py`: 72 hydrostatic columns per grid -- eleven variants each of
convective daytime land, stable night, neutral windy, cloud-topped, very
shallow PBL, a deep PBL that reaches the KMPBL = KM/2 search ceiling -- and six
branch-point columns (BR exactly zero, zero buoyancy flux, a flux carried by
evaporation alone, calm wind, an extreme heat flux that caps HGAMT, a dry
upper column).  Four grids (16, 40, 61 levels, plus a lid-focused 16-level grid), two configurations (60 s
with cloud ice, 180 s without), four calls each: 468,864 graded words.
`cases-index.json` names every column.  The records are packaged under
`woof/data/gfs_pbl/oracle/` (xz) with their SHA-256 manifest.

## Lowering facts the port follows (measured here)

* `x**2` and `ZFAC**PFAC` (PFAC = 2.0) are `x*x`; the two other real powers
  call `powf` (`(PSFC*.00001)**ROVCP`) and `pow` (`(PRSL*.01)**ROVCP`).
* `minmax_probe.F90`: REAL(8) `MAX(a,b)` returns a only when a > b, else b;
  `MIN(a,b)` returns b only when b < a, else a (ties of signed zeros and NaN
  operands included).
* PARAMETERs and the con_* constants are default-REAL literals widened to
  binary64; `gen_constants.py` derives `gfs_pbl_constants.cuh` from the
  pinned copies in `upstream/`.

## WRF's own pairing law, measured

WRF v4.6.1 admits this scheme only with an isfc == 2 surface layer
(`module_physics_init.F:3742-3743`).  On the A071 iowa-convective case
(41 x 41 x 51, 3 km), the scalar WRF referee build:

* with `sf_sfclay_physics = 91`: stops at init, "use myjsfc scheme for this
  pbl option";
* with `sf_sfclay_physics = 2` (MYJSFC, which writes no GZ1OZ0, PSIM or
  PSIH): UST is +Inf at 35 points by 21:02, PBLH/HFX/T2 are NaN everywhere
  at 21:02, and the run dies with a segmentation fault in RRTMG LW taumol
  shortly after.

The production pair is GFS PBL 3 with GFS surface layer 3. The surface
layer publishes the precise PSIM, PSIH, GZ1OZ0, BR and WSPD definitions
BL_GFS consumes. The archived MM5 exception is retired. The Eta pairing
is held because it leaves those boundary fields undefined. The pairing
is enforced both at namelist admission and at the runtime entry point.

## Real native module replay

`instrument_native.py WRF_SOURCE_ROOT OUTPUT_DIRECTORY` checks both module
SHA-256 pins before adding stream writes around the two native calls.
It captures the first two calls on each MPI tile, 8 by 4 columns each,
including every input and output. Build only these instrumented copies in
an own native build, using the strict scalar flags above. Instrumentation
changes no arithmetic or call argument. Stream files are little endian.

```
GPUWM_WRF_EXACT=1 python tools/gfs_pbl_wrf461_oracle/replay_native.py \
  NATIVE_RUN_DIRECTORY receipts/gfs-native-replay.json
```

The November lane's real WRF v4.6.1 run uses the 2026-10-16 09Z prepared
Texas/Oklahoma case, 300 by 300 by 49, 3 km, dt 15 s, GFS 3/3, Noah 2,
WSM6 6 and legacy RRTMG 4/4. The eight-rank capture contains 152,576 PBL
and 10,240 surface words, all identical in strict arithmetic. The native
GFS module objects use strict flags; the reused rest of WRF is its existing
optimized build. This is a module proof, not a whole-WRF strict claim.
Fast arithmetic differs on 25 words in this capture, maximum wind speed
absolute difference 9.5367431640625e-7 m/s. Fast remains the default.

WRF's public-domain notice covers the source modules and FPVS table
construction. Source pins and build flags are in `build.sh` and the surface
oracle's `build.sh`; original source notices remain with the pinned files.



The lid-focused grid has a 3 km physical top, a 25 m first layer and
its upper half clustered near the lid. This is a boundary-condition
oracle grid, not a suggested forecast mesh. On the first call, nine
of its eleven deep columns return PBLH = 2826.233 m, 94.2% of the lid,
in both tracer configurations. The coverage test checks those actual
WRF outputs. The original 6500 m, 16-level grid reaches the search
ceiling well below the physical lid and is not called a near-lid case.

## Paired real forecast and timing

The fast GFS 3/3 forecast completed all 720 steps through `woof go` on
the 2026-10-16 09Z Texas/Oklahoma case above. Native WRF completed the
same three hours. Rust maps were inspected at 10, 11 and 12Z beside
native WRF and IEM ASOS temperature and wind reports. Scoring uses Rust
NetCDF sampling at 21 supported stations per hour. The evidence report is
`WRF-PORTS-2026-11-01/port-gfs-pbl-sfc.md`; its lane evidence retains
commands, pins, maps, scores and deletion receipts. Raw run data are removed.

A third pass shares the identical heat/vapour and condensate matrix
factorization, and reuses native geometry divisions after their original
workspace values are dead. Every native recurrence and binary64 operation
is retained. Strict fixture checks pass; fast before/after comparison has
zero differences in 468,864 words. The optimized whole forecast completed
again. All three hourly forecast NetCDF files (10, 11 and 12Z) are
byte-identical to the earlier GPU forecast. All six hourly temperature/
wind PNG files also match, with identical Rust renderer settings.

A new eight-rank native run captures both the first calls and late-hour
outlier regions. Strict replay matches all 167,904 real output words:
157,344 PBL and 10,560 surface words, zero differences. The independent
reference also matches all captured late native output words. At j261 i262,
the 12Z GPU pre-call Richardson number at level 3 is 0.2581141857, above
WRF's 0.25 first-crossing threshold, while native is 0.2167371867. The
same native equations give the shallow 54.568714 m GPU height and deep
920.698059 m native height on their respective evolved inputs. Native
11Z calls also alternate between shallow and deep heights as their
level-3 Richardson number crosses 0.25. This witnesses the discrete
threshold mechanism without smoothing or changing the native scheme.
It does not establish whole-forecast identity or retrospectively capture
every outlier in the earlier two-rank run.

The new 12Z comparison has three cells above 500 m out of 90,000,
maximum 896.351 m and p99 7.604 m. Broad patterns agree. Temperature
station RMSE is 0.9076 K GPU and 0.9053 K native; scalar wind RMSE is
2.8180 and 2.8187 m/s. Both underpredict observed speed by about 1.77 m/s.
ASOS supplies no PBL-height observation. Maturity remains
`implemented-unverified`: one exercised case does not establish general
forecast skill.

The speed pass on a shared RTX 5090 uses captured real 300x300x49
columns in the complete production driver stages. Three warmups and three
interleaved batches of twenty calls, with CUDA-event readback only after
each batch, measure PBL 2.638197 ms versus its starting-tip
implementation 3.283002 ms and YSU 2.292237 ms
(1.151x YSU, 19.64% less than the starting implementation).
Surface is 0.092896 ms versus 0.156133 ms
before and MM5 0.077598 ms (1.197x MM5,
40.50% less). Both meet 1.2x on this case/card;
no arithmetic-cost exception is claimed. The secondary 128x128x49
comparison is 1.100x YSU and 0.962x MM5.
The process selects fast numerics before importing test helpers and
records the actual mode/options. Earlier timing claims did not establish
that ordering and are superseded.

Nsight Systems traces actual GO call 3: PBL device operations fall from
72 to 14, including 63 to 10 kernel launches and six to one memset.
There is no synchronization or host transfer inside either stage range;
three PBL copies are device to device. The proof wrapper synchronizes
outside the ranges. Nsight Compute identifies math/FP64 throughput rather
than memory or spill traffic: original DRAM utilization is 3.03%, with
zero spills. Sharing binary64 divisions, pressure logarithms, roughness
expressions and drag factors cuts executed FP64 instructions from
2,840,548 to 1,625,438. Counter runs leave shared-card clocks/caches
uncontrolled; their 149.66 to 87.74 microsecond kernel durations are
profiles, not the controlled stage-speed result above. Current production
PBL uses 114 registers and surface 86, with zero local spill bytes.

The PBL streams matrix construction into native forward elimination and
back substitution into tendency output, caches repeated geometry, and
evaluates potential-temperature powers only through the levels reached
by the unchanged searches. Workspace falls from twelve to ten slots;
all tracers and native thresholds remain. One CUDA finite-check pass
replaces the repeated field reductions, retaining each named failure and
the deferred health-ledger drain. Surface fast algebra retains binary64;
strict mode keeps native operation order, powers and divisions.

The final source passes 32 focused checks including the 468,864-word
strict fixture and both PBL restart cadences. Replaying the twelve retained
late real-native streams matches 4,768 PBL and 320 surface words, zero
differences. The earlier 167,904-word full replay remains evidence for its
earlier source; its deleted early streams were not rerun or represented
as current proof. New fast old/new comparisons match all 468,864 fixture,
26,820,000 real PBL and 1,800,000 real surface words.

The final fast GO forecast completes all 720 steps. Nine new Rust PNGs
at 10, 11 and 12Z were viewed beside retained native WRF maps and ASOS
dots. Broad PBL, temperature and wind patterns remain. This new forecast
is not byte-identical to the starting forecast: at 12Z, mean absolute T2
change is 0.00791 K, maximum 0.72983 K; mean absolute PBLH change is
0.56225 m, maximum 842.40478 m. Local threshold switches remain and both
native and GPU underpredict some observed wind speeds. No new native
forecast, new station-score result, whole-forecast identity or general
skill claim is made for the speed pass. Exact source hashes match both
whole-run certification capsules. Evidence is retained under the lane's
speed-pass folder in the authorized local report delivery.

Focused GPU checks cover restart, positive PBL cadence, fractional-ice
RUC restart, both nest feedback modes and independent ensemble members.
With GFS options off, an eight-step candidate comparison retained 169
identical serialized fields (87,904 bytes), excluding only the restart
creation timestamp. The third-pass product edits touch only the GFS
kernel and its workspace count. These checks are not native nested or
ensemble real forecast proofs.
