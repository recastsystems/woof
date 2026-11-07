This records the initial unchanged-native failure for mp_physics=15.
The subsequent corrected-caller forecast and real-input GPU proof are documented
in [REAL-REPLAY.md](REAL-REPLAY.md). The original failure remains reproducible;
the corrected reference must not be called an unchanged WRF executable.

The reference is WRF v4.7.1 commit
`f52c197ed39d12e087d02c50f412d90d418f6186`. Its `LICENSE.txt:1` declares the
release public domain. The unchanged notice is retained in `WRF-LICENSE.txt`.
The inspected Ferrier module and expanded-rain table contain no additional
licence restriction. Source SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `phys/module_mp_fer_hires.F` | `69e937f0f10b4964e3c6742cb852ef68df451dcdeb05ee82451c23a15bb72a57` |
| `phys/module_physics_init.F` | `f48cf9793f782a72c3969e986eef24746252f030cd33cfba71cc82b81d128d90` |
| `Registry/Registry.EM_COMMON` | `6f3ee02175b76487c5c6c046ff2fb4c5d41980b86c0f06eb2e09e34dacc9623a` |
| `run/ETAMPNEW_DATA.expanded_rain` | `28e60e6c83eb55889c4ab65eaa33e9fc3b557e5ab45c3dde7ee7153c2ef95a13` |

The native initialization defect crosses these original source sites:

1. `Registry/Registry.EM_COMMON:3023` activates `f_ice_phy`, `f_rain_phy`,
   and `f_rimef_phy` for option 5. The option 15 package at line 3024 omits them.
2. `tools/gen_allocs.c:695` emits one-element allocations for inactive fields.
   The generated `inc/allocs_5.F:2904` allocates `f_ice_phy(1,1,1)`.
3. `dyn_em/start_em.F:1018` passes the actual grid pointers through `phy_init`.
   `phys/module_physics_init.F:4504` passes all three fractions to
   `fer_hires_init` for both options 5 and 15.
4. `phys/module_mp_fer_hires.F:2886` declares explicit domain-shaped optional
   arrays. Its presence test at line 2912 succeeds, and lines 2915-2920 write
   the entire tile through those arguments.

A full, unchanged strict native build was run on the official January 2000
30 km WRF test data. The archive is
`https://www2.mmm.ucar.edu/wrf/src/data/wps_30km.tar.gz`, SHA-256
`c60a0dc5cff57ceae1555b68c06b184314b54dcbee09bf6da9511dd990d693d8`.
Only d01 met files at 2000-01-24 12:00 and 15:00 were prepared. The native
sample `test/em_real/namelist.input.jan00` was adjusted to these file dimensions:
`num_metgrid_levels=27`, `num_land_cat=24`, `interval_seconds=10800`,
`max_dom=1`, `e_we=74`, `e_sn=61`, `e_vert=33`, and `mp_physics=15`.
The requested run was three hours with hourly history. Other physics remained
at the sample's options 4/4 radiation, 1 surface layer, 2 land surface,
1 boundary layer and 1 cumulus.

The compiler was GNU Fortran 15.2.0, with
`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`. The build command was
`nice -n 10 ./compile -j 8 em_real`. Both native executables were invoked as
`mpirun --bind-to none -np 4 /path/to/native/main/real.exe` and then
`mpirun --bind-to none -np 4 /path/to/native/main/wrf.exe`, from the case
directory, with `OMP_NUM_THREADS=1`.

The unchanged option 15 `real.exe` succeeded and wrote a stock WRF input with
`QRIMEF`, `QICE`, `QCLOUD` and `QRAIN`. Its `wrf.exe` exited 139 before the
initial history frame completed. The error was
`Program received signal SIGSEGV: Segmentation fault - invalid memory reference.`
The backtrace included `output_wrf_`, `open_hist_w_` and
`med_before_solve_io_`. That crash location alone does not identify the earlier
write that corrupted memory.

A control using the same unchanged binaries and met input, with option 5 and
`run_hours=0, run_seconds=180`, completed with exit 0 and
`wrf: SUCCESS COMPLETE WRF`. Its one native step took 0.73274 seconds across
four MPI ranks. This is a startup control, not a GPU stage timing comparison.

To observe the actual pointer sizes before native physics receives them, apply
`native_startup_trace.patch` to a separate copy of the pinned WRF tree:

```sh
git apply --check /path/to/native_startup_trace.patch
git apply /path/to/native_startup_trace.patch
nice -n 10 ./compile -j 8 em_real
```

Run the same option 15 case with that diagnostic executable. The patch adds
only messages before `phy_init`: actual fraction allocation sizes, tile bounds
and memory bounds. It neither fixes nor suppresses the native overwrite. Keep
this instrumented result distinct from the unchanged-native crash.

The diagnostic run reported `MP15_ACTUAL_FRACTION_SIZES 1 1 1` on all four
MPI ranks, then exited 139. Rank 0 reported tile bounds
`1 37 1 30 1 33` and memory bounds `-4 44 -4 37 1 33`. The messages are in
`native-paper/diagnostic-rsl.error.0000:27` through line 29; the corresponding
rank 1, 2 and 3 logs also contain the one-element sizes. This is an observed
allocation mismatch in a generated full WRF run. The earlier synthetic
one-element-pointer probe found 141 writes beyond logical allocations on a
4 by 3 by 4 extent. That synthetic count is not a full-forecast replay count.
No memory sanitizer or native initialization correction was used, so the
specific causal link between the overwrite and the later IO crash is inferred
from these source sites and observations, rather than claimed as a sanitizer
diagnosis.

The detailed command log, complete namelists, source manifest, original crash,
control run and diagnostic receipt are retained in the lane evidence for
2026-11-01, `port-mp15-fer-advect`. The initial audit stopped at the native failure. That earlier held status is
superseded by the three named caller corrections, real native captures and
whole forecast documented in REAL-REPLAY.md. The corrected reference leaves
Ferrier's column arithmetic unchanged. The recovered synthetic fixtures remain
regressions only; they do not supply the real-input proof.
