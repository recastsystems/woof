# Operational output controls

Authority: NOAA-EMC/HRRR commit
`40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827`, under
`sorc/hrrr_wrfarw.fd/WRFV3.9/`. This is the operational fork, not stock
WRF 4.6.1. Its diagnostic module is **phys/module_diag_gsd.F**;
`phys/module_diag_nwp.F` does not exist at this revision.

## Cadence and precipitation

`history_interval2_s` and `history_interval_change_s` are domain output
controls; a nonzero change requires the second interval. The packaged recipe writes every 900 seconds
through 64800 seconds, then every 3600 seconds. A 48-hour schedule has 103
frames including initialization. Both fixed and adaptive clocks carry the
change, as do native-namelist import/export and prepared history inventories.
Disk admission counts the two alarm phases algebraically, including the
exclusive resume boundary, instead of pricing every frame at the first interval.

`prec_acc_dt` and `prec_acc_dt1` retain WRF's units of minutes (60 and 15).
`module_diag_gsd.F:724-813` resets the corresponding bucket at step-start
`MOD(curr_secs2, interval*60)==0`, then adds the current step. History reads
never reset it. Each interval writes `PREC_ACC_C`, `PREC_ACC_NC`,
`SNOW_ACC_NC`, `GRAUP_ACC_NC`, with suffix `1` for the second interval.
Snow includes convective rain when T2 is below 273.15 K, exactly as the fork.
These are GPU accumulators and restart/streaming carriers.

## GSD inventory and reset authority

Registry.EM_COMMON:2992's `gsd_diags_rt` package contains 13 fields:
`WSPD10MAX`, `W_UP_MAX`, `W_DN_MAX`, `UP_HELI_MAX`, `W_MEAN`,
`GRPL_MAX`, `HAIL_MAXK1`, `HAIL_MAX2D`, `LWP`, `IWP`, `WSPD10`,
`WSPD80`, `MAXCLDFRA`. The module also fills the misc Registry fields
below. They are carried in history, including the fork's lowercase
vorticity variable names.

| Fields | Producer and meaning |
| --- | --- |
| WSPD10MAX, WSPD10UMAX, WSPD10VMAX | Per-step speed maximum and its U/V components |
| WSPD80MAX, WSPD80UMAX, WSPD80VMAX | Declared and reset in the authority, never assigned a nonzero value there |
| W_UP_MAX, W_DN_MAX | Per-step extrema below 100 hPa |
| W_MEAN | Time mean of column-mean W between sigma 0.5 and 0.8 |
| GRPL_MAX | Maximum column integrated graupel, kg m-2 |
| HAIL_MAXK1, HAIL_MAX2D | Thompson fork's 50-bin PSD tail diameter at the first mass level / column maximum, metres |
| REFD_MAX, REFDM10C_MAX | Derived reflectivity maximum at the first mass level / highest -10 C crossing |
| LTG1_MAX, LTG2_MAX, LTG3_MAX | Lightning threats; LTG3 scales by whole-domain LTG1/LTG2 maxima |
| UP_HELI_MAX/MIN, UH | 2-5 km AGL helicity extrema / instantaneous integral |
| UP_HELI_MAX16/MIN16, UH16 | 1-6 km AGL |
| UP_HELI_MAX02/MIN02, UH02 | 0-2 km AGL |
| UP_HELI_MAX03/MIN03, UH03 | 0-3 km AGL |
| rel_vort, rel_vort_max, rel_vort_max01 | Instantaneous 3-D vorticity and fork's 0-2 / 0-1 km extrema |
| WSPD10, WSPD80 | Model-level wind interpolation, filled on the first history-due call after the hourly sentinel reset |
| LWP, IWP, MAXCLDFRA | History-time cloud-water paths and boundary-layer cloud fraction |
| COMPOSITE_REFL_10CM, REFL_10CM_1KM, REFL_10CM_4KM | History-time reflectivity with native convective contribution, base-geopotential height selection and -10 dBZ floor |

`module_diag_gsd.F:817-874` uses `reset_interval1`, default **60 minutes**,
independent of history. It resets before folding the current step; its
position after first-RK helicity means the boundary step's helicity sample
is discarded. `module_diffusion_em.F:6068-6816` owns the four helicity bands
and vorticity, sampled before the dynamics solve. The stock completed-step
UH evaluation remains separate for moving-nest tracking; GSD output does
not change those consumers. All column math, stencil work and reductions
run on the GPU. Restart carries the interval state. Streamed history
recomputes LTG3 after full-domain carriers are assembled, avoiding tile-local
scaling. `tests/test_gsd_output.py` covers these boundaries and inertness.

The fork has an unusual vorticity overwrite: every vertical level compares
against the previous step's 0-2 km maximum, also for the 0-1 km field.
This implementation preserves that comparison rather than replacing it
with a mathematically different vertical reduction.
The previous-value selector also preserves `cal_helicity`'s integer
`MOD(itimestep-1, INT(reset_interval1*60/dt))`, including changes of dt.
Its GPU step counter survives restart and is not cleared by the separate
clock-hour reset in `module_diag_gsd`.

## What diag_int and wind_int actually gate

`diag_int=15` and `wind_int=5` do **not** gate GSD severe-weather maxima.
They gate `phys/module_diag_cl.F::clwrf_output_calc`, called under
`output_diagnostics=1` in `module_diagnostics_driver.F:1095`.
The climate-statistics module tests these periods at lines 171 and 292.
The GPU port carries all 48 produced fields: min/max, their times, mean and
population standard deviation for T2, Q2, SKINTEMP, SWRAD and SWNORM;
U10/V10 at the speed maximum, speed maximum and its time, and U10/V10/speed
means and standard deviations; RAINCV/RAINNCV flux maxima, their times,
means and standard deviations. Names follow the Registry (for example
T2MIN, TT2MIN, SWRADMEAN and TRAINNCVMAX). Times are forecast minutes.

RunConfig/import/export retain WRF's `output_diagnostics`, `diag_int` and
`wind_int`, with Registry defaults 0, 15 and 5. Both packaged recipe copies
enable the output. The default-off checkpoint echo omits these defaults;
prepared-cache identity treats them as default-tolerant output controls.

The port retains REAL MOD tolerance, INT(period/dt) counters, strict extrema
ties, reset timestamps left untouched by the source, and its speed-maximum
U/V reset quirk. Means/std slots hold raw sums until the closing step,
then the routine divides them in place; history does not finalize or reset
them. All 48 fields and both counters survive restart and tile transport.
WOOF-ArWen checkpoint continuation restores the absolute forecast clock and window
arrays. The replay API also exposes the native restart flag and restart-relative
clock; native WRF restart-file interchange is not claimed.

`tools/climate_output_oracle/run.py` compiles the entire unmodified pinned
module with gfortran and framework-only stubs. Its frozen 106-call receipt
covers every field at every call, both cadences, changing/nondivisible dt,
native restart suppression and untouched reset values. Strict-numerics GPU
results match all binary32 words. Fast production numerics retain FMA.
Tests also cover real-step trajectory inertness, asynchronous Rust history
snapshots, restart restore, and pinned host gather/update/scatter with tile reuse.

## Initial-time reflectivity

With `gsd_diagnostics=1`, `do_radar_ref=true` and Thompson microphysics
8/28, the first history frame calls the fork's
`module_diag_gsd.F:17-153::gsd_refl_first_time_step` on the GPU. Before
`calc_refl10cm`, nonpositive rain number with rain mass at least 1.e-12
uses the native Marshall-Palmer intercept of 8.e6 m-4. This fallback is
private to the radar calculation; prognostic NR and all other inputs stay
unchanged. The first-call flag disables melting-layer bright-band treatment,
and the output retains the native -35 dBZ floor. Later history frames still
consume the microphysics-time diagnostic.

The history route supports resident arrays, host arrays and store-direct
domains through bounded 64-row GPU slabs. It prepares theta, Exner and
moist density with the fork's initial phy_prep expressions. Exact mode uses
this helper for the HRRR configuration too; stock exact-mode t=0 zeros
remain for configurations without GSD. Its 87-byte-per-level/thread arena
is bounded to threads in flight and included in preflight pricing and
kernel prewarm. The launched helper reads zero local-frame bytes on
RTX 3080 / sm_86 / NVRTC 13.0.48.

`tools/gsd_initial_reflectivity_oracle/run.py` compiles the unmodified
extracted Fortran helper and `calc_refl10cm`, plus the complete native radar
module, using desktop WSL gfortran. Only framework stubs and a harness are
added. The 16-call reference covers both selectors, supplied rain numbers,
fallback thresholds, warm/cold/mixed columns and native state preparation.
Strict-numerics GPU results match all 3,456 binary32 output words. Tests also
write native reference values through the Rust NetCDF history path for all
three placements. Fast production numerics retain FMA; word identity is
claimed with strict numerics on these fixtures. The desktop fast-mode probe
differs in 50 of 3,456 words, with maximum absolute error 1.90735e-5 dBZ.

## Evidence and qualification limits

`tools/gsd_output_oracle/extract.py` extracts the unmodified Thompson hail
expressions, scalarizing array indices only. Frozen gfortran outputs and
binary64 bin words are under `tests/data/oracles/gsd_output`. Seven hail
probes match float32 output words exactly. Other tests use manufactured
columns, metric helicity, reset boundaries, restart and real GPU model steps;
they do not establish full operational WRF forecast or complete-file parity.
The output kernels join preflight memory pricing and the parallel static
compile plan. The output-window entries and launched initial-history helper
measured zero local frame through the production loaders on RTX 3080 /
sm_86 / NVRTC 13.0.48; the recording names that platform rather than claiming
measurements on other architectures. The initial-helper unit also compiles
an unused older Thompson entry from its device header; that entry is never
launched here and is excluded from the zero-frame claim.

The separate `diag_print=1` DPSDT/DMUDT console-budget path has no history
flag in Registry.EM_COMMON:1922-1923 and is not implemented here. The
restart-only 3-D `rlv`/`rlvn` extrema are not emitted history variables and
are not native restart interchange carriers. The Registry also declares
NCI/NCA convection-initiation counters and QR_MAX_CI/QG_MAX_CI, but the pinned
GSD producer does not update them. This lane does not invent values for them.
GSD hail tables for microphysics other than 0/8/28 are explicitly refused.

A full native output-file comparison remains outside the focused evidence.
The precipitation block's
additional runoff resets and ACSNOW/ACGRAUP/ACFRAIN accounting are not claimed
as part of the eight interval buckets; their physical land-surface carriers
need a separate units/ownership audit before being changed.

## Full-grid history cost

The resolved 1799 x 1059 x 50 recipe has 369 planned history fields and
**19,227,482,528 bytes (17.907 GiB)** per uncompressed full frame, including
the planner's bounded header allowance. At 48 hours, 103 frames total
1.801 TiB, compared with 49 hourly frames (before any data-dependent
compression). This is an inventory calculation; it allocates no full grid.

`woof/io/wrfout.py::AsyncDomainWrfoutWriter` already has a side CUDA stream,
pinned D2H buffers, a worker thread and the Rust NetCDF writer. The worker
waits for D2H and performs file output off the stepping thread. However,
snapshot capture has a producer-stream event dependency, the bounded queue
can apply backpressure, and borrowed host-store frames must be drained
before their arrays are mutated. Therefore **fully off-step-path, full-CONUS
throughput is not established** by this lane. The RTX 3080 cannot hold this
resident domain plus snapshots. A larger-card / streamed full-grid timing
receipt is needed to qualify sustained 15-minute output; no such run was
performed on the desktop.
