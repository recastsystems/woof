# Changelog

## 0.4.11

Requires woof>=2.9.11,<2.10.

New:
- `hrrrv5` on the 96 -> 3 km zoom globe: `woof hex mesh-plan --global-mesh ... --run --geog ROOT --forecast-arg=--physics-suite=hrrrv5` builds NOAA's RRFS start state and the GSL drag statistics itself. `--forecast-arg=--physics=config_convection_scheme=cu_grell_freitas_li` adds NOAA GSL's scale-aware Grell-Freitas on the GPU as the labelled variant "hrrrv5 + storm scheme", so the coarse cells keep their convective rain (hrrrv5 itself has no cumulus scheme); it fades out on the 3 km cells.
- `woof hex forecast|global --wind-farms uswtdb|PATH` runs Fitch wind-farm drag and turbine TKE inside MYNN on globes and zoom globes, with per-turbine and per-farm power and hub wind in history (opt-in).
- `hrrrv5` on any region: `woof hex country` and `woof hex regional` with `--forecast-arg=--physics-suite=hrrrv5` start from NOAA's RRFS init (case 7 on the region, case 9 boundaries) from GFS or IFS, on a circle, an ellipse (`--ellipse`) or a drawn outline (`--shape`, Shape row or GeoJSON), on one card or several (`--samehost-devices 0,1,2,3`).

Changed:
- A static built for hrrrv5 reads ERA5's deep-soil temperature (`soiltemp_era5_1deg`) instead of NOAA's ISLSCP-I `soiltemp_1deg`, whose licence is unclear: about 2 K warmer over land on average (docs/rrfs-init.md). `--deep-soil-temperature islscp-i` selects NOAA's by name. NOAA's own conus3km static is unchanged.
- Simulated radar during a hex forecast reads its radar columns from each committed state, and its backpressure waits follow the radar consumer's real progress, so radar keeps pace with the run.
- Hex products keep pace with a fast forecast: live frames are drawn side by side and converted once, a zoom globe's whole-globe view and its context view are converted at their pictures' pixels (no context tiles unless asked), and the frame cache leaves 10 % of its disk free. A whole globe finer than its picture is drawn on the picture's grid, so its renderer's memory no longer grows with the mesh.
- A zoom globe's `export --window global` is a stack, each region at the detail the model has there (the sphere at the outer spacing, a lat-lon box per ring, the core on its own box with the usual coordinates), and `export --follow` writes it beside the forecast. `woof hex regional` draws each frame as it lands.
- Storm-following regions on the MPAS-physics route move about four times faster: the forecast process stays on the card across moves, prepares the next region while the current one runs and hands the state over in memory (about 45 s to about 11 s per move on one RTX 5090). Output is unchanged; `--no-resident` keeps one process per segment.

Fixed:
- GFS starts on the Noah land model (the MPAS-physics default) from cycles after 2026-10-28 no longer stop at init ('soil contract does not cover this analysis date'): the cold-soil contract is keyed on the GFS analysis itself (its model, soil layers and land model, read off the donor file) instead of an end date, so an unchanged GFS keeps the soil remap.
- No case the cold-soil remap cannot serve stops an init any more: a GFS start from a later forecast hour (init-source, regions and moving cases), a land model other than Noah (the WOOF route's Noah-MP, RUC), a missing or late donor, a cycle before GFS v16, a GFS analysis from another land setup, an all-water region and the rest start from the source's interpolated soil, as before 0.4.10, and the init's log and receipt say why. `--soil-remap on` refuses instead.
- A limited-area region started from GFS (country and region doors, hourly boundaries) no longer stops at its first boundary hour ('this cold-soil donor row is analysis-only'): only the run's start takes GFS's cold-soil donor, as the regional door already did.
- RRFS v2 `hrrrv5` forecasts take `--stream-input invariant=FILE`: a registered input stream the run binds is read once at the start instead of being refused as inactive.
- RRFS v2 `hrrrv5` history draws maps again: its aerosol optical depth (`aod550`) declares CF units '1', so the converter no longer refuses it.
- `woof hex country --forecast-arg=--physics-suite=hrrrv5` no longer dies as its forecast starts (KeyError 'nscat'), from GFS or IFS.
- A physics-resource cache that lost RRFS or MPAS tables (a cleaned disk) fetches them again instead of failing on a missing table file.
- Map titles name the view and the spacing its picture shows ('Core zoom 1-4.1 km', 'Whole globe 1-129 km') instead of the whole mesh's range; `--view-label VIEW=TEXT` passes a caller's own words.
- RRFS Grell-Freitas-Li (`cu_grell_freitas_li`) runs on the GPU-resident route with no native library: it no longer stops at start ('init_convection for GFL / Tiedtke schemes has no ported path'), at its first step ('cached extent conflict for diag_physics.vcpool') or on its momentum carriers ('sub3d_rucuten_p is not allocated'), and its columns run in wide launches instead of 64 at a time.

## 0.4.10

Requires woof>=2.9.10,<2.10.

New:
- RRFS-MPAS `hrrrv5` runs with no native library: its physics runs GPU-resident on the ported fork framework, and its history and diagnostics are written by `rw_mpas_stream` with the fork's stream contract.
- `woof hex init --physics-suite hrrrv5` (and every RRFS-MPAS suite) makes the initial condition and hourly boundaries the way RRFS does from GFS: the RRFS fork's init_atmosphere ported to Rust, with RUC's nine soil levels, the L60 grid, TEMPO aerosols, the fork static and the GSL drag statistics. Equal to the fork's own init on every stored value (docs/rrfs-init.md).
- `hrrrv5` runs the MPAS commit NOAA's RRFS v2 runs (RRFSx/MPAS-Model 44c0849): a 6 h x1.2562 forecast in strict numerics writes every history and diagnostics word equal to native 44c0849. `--physics-pin-set rrfs` keeps the previous pin.
- RRFS `hrrrv5` runs NSSL two-moment microphysics (`mp_nssl2m`) on the GPU; a 3 km step costs about 1.17x the TEMPO default.
- RRFS-SD smoke and dust run on the GPU on the default 44c0849 pin; a 3 km step with real fires costs about 1.12x the same run without smoke.
- RRFS at 3 km steps faster: TEMPO microphysics (176 -> 143 ms of a 3 km step) and the limited-area dycore's stencils and acoustic solve (about 12 % of the step), with every word unchanged in strict numerics.
- The RRFS init engines (`rw_mpas_init_v831`, `rw_mpas_gsl_oro`), which woof 2.9.10 ships in its release bundle rather than its wheel, are staged automatically the first time an RRFS init needs them.
- Optional Icechunk output stores native forecast history as atomic Zarr v3 snapshots, locally or on S3 (`--output-format netcdf,icechunk` on every forecast door; needs woof 2.9.10, whose writer is fetched on first use).

Changed:
- MPAS-physics history and diagnostics files are written lossless-compressed (NetCDF-4, the same values in a fraction of the disk), compressed on every core by the engine's `rw_mpas_stream`. Init, restart and boundary files keep their native format. `GPUWM_HISTORY_COMPRESSION=none` writes them as before.
- Regional GFS cold starts (zoom globe, limited area, storm-following) default to correct numerics with the `hrrr-like-aerosol` profile: aerosol-aware Thompson with `nwfa2d` initialisation, the corrected MYNN cloud merge, cold-start aerosol emissions, soil moisture mapped by relative wetness and the land carry. Forecasts on those runs change; the step costs about 1.03x the plain suite's.
- A hex region driven by a global source (GFS, IFS, GEFS) takes its boundaries from every lead the source posts (GFS hourly, IFS and GEFS 3-hourly) instead of every 6 h, can start at any posted lead, and `--boundary-interval-hours` picks a coarser step. Boundary files are written in parallel.
- GEFS member inits and boundaries download only the records the init reads, about 34 MB instead of 106-118 MB per member and lead.

Fixed:
- A local-time-stepping run on MPAS physics records every class step, the coordinator's work between them and the loop's own measured rate in `native-runtime-timings.json`; it used to hold only the class that finished last, so most of each step was unlabelled.
- A local-time-stepping profile run (`WOOF_HEX_LTS_DEVICE_SPANS=1`) records each class step's device time. The default receipt's per-class rows are host wall, which on one card includes the device work queued ahead of a class, and the receipt now says so.
- The zoom-globe receipt (`global-vr-receipt.json`) names the physics mode the forecast ran. A run that named a suite but no mode ran correct mode, and the receipt said exact.
- A hex run on MPAS-A physics started from HRRR (or any first guess that carries hydrometeors) starts from its cloud, rain, ice, snow and graupel and HRRR's own numbers, as the hrrrv5 route does, instead of from no condensate (no echo at hour zero); `woof hex init --first-guess-hydrometeors no`, and strict numerics, keep MPAS-A v8.4.2 case 7's dry start. Needs woof 2.9.10's `rw_mpas_init`; an older engine is a named gap.
- A hex run started from GFS also starts from GFS's own cloud, rain, ice, snow and graupel (the GFS rows now decode them), instead of from no condensate.
- The hour-zero radar maps of the MPAS-A suites (Thompson microphysics) show the reflectivity of the starting state instead of nothing.
- So does the hour-zero radar map of a mesoscale_reference run (WSM6 microphysics).
- Map subtitles no longer cut a long `--model-label` mid-word: it ends on a whole word with '...', and `--label` names only the files.
- Limited-area runs with many tracers on few cards no longer crash in their first step: NOAA's conus3km RRFS domain now runs on 4 x 96 GB cards, with less device memory per card.
- Multi-card runs (same-host rank processes, such as RRFS `hrrrv5` on several cards) no longer recompile their GPU kernels in every rank process before the first step: each rank uses the forecast process's warm kernel cache, so step 1 on 4 ranks went from 11.5 s to 2.2 s, with every word unchanged.
- Multi-card limited-area runs read only their own rows of each boundary file: every rank process used to read each boundary file whole (13.3 GB a boundary hour on NOAA's conus3km) to keep its share. Each now gathers its rows from the shared page cache, with about 13x less kernel time and a fraction of the host memory per rank, and every word unchanged. Each rank's receipt also splits its CPU time into start-up, first step and stepping, with the kernel time per thread.
- DA cycles on the RRFS hrrrv5 suite accept their own soil state: soil moisture under the RUC land model is bounded by RUC's porosity, not Noah's.
- An MPAS-suite forecast refuses an RRFS init made by another fork pin's producer than its own route's, and the refusal names both ways to fix it.
- `country --polygon` gives a drawn region the grown halving ladder, so `--fine-dx-m` is the spacing it gets, and a large drawn region keeps every side of its grown bands.
- Hex starts from HRRR: 10 m wind and 2 m humidity are filled on regional boxes west of Greenwich; HRRR's own cloud, rain and ice numbers and its RUC soil nodes reach the start; the HRRR crop stops at HRRR's edge instead of refusing.
- Hex starts from HRRR with no whole HRRR file on disk download the records their decoder reads (cloud, rain and ice numbers, 2 m humidity and the analysed aerosols) instead of failing at the intermediate stage.
- On the RRFS suites the first history frame carries reflectivity (`refl10cm`, its column maximum and the 1 km level) from TEMPO's own operator instead of zeros; the exact mode keeps native MPAS's zero. On a multi-card run it is formed in column slabs (1.3 GB on rank 0's card at any mesh size; it was 8 GB on NOAA's conus3km), not from a whole copy of the global start state beside the first step.
- A regional run on the `hrrr-like-aerosol` default fits a 24 GB card (RTX 4090) where it fits: admission prices it from a measured run on that card instead of refusing it.
- `woof hex regional` runs RRFS `hrrrv5` end to end (`--forecast-arg=--physics-suite=hrrrv5`): RUC initial and boundary files, the resolved namelist staged, GSL drag statistics on any mesh, and no uncomputed or empty fields in the default streams; an initial file whose land soil came out at 0 K is refused by name.
- Same-host rank processes exchange halos with one socket header per field set instead of one per round, and check the round order in shared memory (output unchanged).
- Region and nest triggers on updraft (`wmax`), CAPE (`mucape`) and 2-5 km updraft helicity (`uh25`) arm and fire on both physics routes; they were refused at arm, so only reflectivity triggers worked.
- Same-host rank processes no longer each hold the whole shared mesh or a second, unused copy of the TEMPO lookup tables, so multi-card runs on large meshes use much less host memory.
- The forecast process of a multi-card run (same-host rank processes) releases the global initial state and the other ranks' inputs once every rank's inputs are cut, instead of holding them for the whole run. Each rank's launch receipt now records its own peak host memory (`max_rss_mib`); it used to show the forecast process's memory at launch.
- `WOOF_HEX_EXCHANGE_TIMING=1` records the same-host halo timings on the MPAS physics route too, for rank threads and rank processes; unset, nothing changes.
- MPAS physics forecasts no longer wait for the card at the end of every dynamics step: the step check is read at most one step later, before any output, so the host queues the next step while the card works. Output is unchanged.
- An RRFS `hrrrv5` run resumed with `--restart-from` continues the straight run word for word in correct mode, on one card, and on same-host ranks on global meshes.
- MPAS-physics runs on same-host rank processes (RRFS `hrrrv5` on several cards, global meshes) no longer crash at their first `--checkpoint-hours` checkpoint, and each rank now writes its own share of the checkpoint beside the step loop instead of rank 0 gathering and hashing the whole state while every card waits. `--restart-from` reads both the sharded and the one-file checkpoints.
- A multi-card run whose rank process dies (killed, or out of memory) now stops within seconds with that rank's failure named, instead of hanging until an outer timeout: the other rank processes are stopped at once, and a forecast process that does not stop by itself within 10 s exits with code 1.
- MPAS-physics history files open in netCDF-C again (ncdump, netCDF4-python, xarray, ncview, NCL): the lossless writer no longer puts an empty `_Netcdf4Coordinates` attribute on scalar variables such as `isice_lu`, which netCDF-C refused with "NetCDF: HDF error". Every value in the file is unchanged.
- Limited-area runs on one card on the MPAS physics route (RRFS hrrrv5) take the DA hand-offs: JEDI backgrounds and analyses, checkpoints and restarts with lateral boundaries, and IAU; a restart or an unchanged analysis continues the forecast word for word.
- Limited-area runs on several cards on the MPAS physics route take checkpoints and restarts with lateral boundaries; a restart continues the forecast and its diagnostics word for word in correct mode, including window accumulators at a checkpoint between diagnostics frames. IAU on a limited area stays on one card.
- A limited-area run on several cards whose namelist asks for IAU (`&iau config_IAU_option`) is refused when it starts, as the `--iau-*` flags already were, instead of on every rank after the whole multi-card launch; IAU on a limited area runs on one card.
- Ensemble members that share a card use less GPU memory; a single run keeps its memory use and speed.
- RRFS conus3km on 8 cards starts sooner: the limited-area inputs are read while the pools are transported, each rank receives the global mesh once, and the mesh checks run on many threads.
- A credit the start data asks for (`WOOF_HEX_SOURCE_CREDIT`, e.g. ECMWF open data's CC BY 4.0 line) is added to each map's own source line and fitted to the map's size, so ensemble member maps keep their member number and the credit is never cut.
- A finished forecast no longer hangs at exit (holding its card) when a kernel compile worker was stopped while it returned its image.
- Kernel compile workers end with the forecast that started them, however it ends (finished, failed, stopped by a signal or killed), instead of staying behind holding memory.
- Local time stepping on the MPAS physics route checkpoints and resumes (`--checkpoint-hours`, `--restart-from`); before, those options made it run the fixed step.
- A start with no kernel pack compiles every physics kernel on the parallel compile pool, Grell-Freitas and Thompson's column module included, and a run with its own cache root reads the box's kernel store and packs.
- Zoom globes and country regions keep their mesh's measured stable step when it is shorter than the MPAS suite's default step.
- Storm-following regions on a known path prepare their next regions from the start, by rotating the premade mesh, so moves land on time; the run summary names every decided move and hold.
- `--device N` on `woof hex moving run` and the moving doors is the Nth card the process already sees, so a card queue's lease holds instead of the run landing on physical card N.
- A GFS cold start whose mesh has land cells far from any GFS soil point (Kerguelen, Tahiti) starts: those cells keep the cold-init soil instead of refusing the whole init.
- `woof hex regional --shape` and `--ellipse` (drawn limited-area regions) pass the HRRR intermediate stage instead of refusing after the mesh is built.
- An init from ECMWF open data (`--source ifs`, `ifs-0p1`, `aifs`, `ifs-ens` or `aifs-ens`, any door) has sea ice. ECMWF publishes sea-ice thickness only (AIFS no sea-ice field), so every ECMWF init started ice-free; the rows now take the same cycle's GFS analysis sea-ice concentration (two records, about 130 kB) as the init's surface input, interpolated over water with its own land mask.
- `--source ifs-0p1` refuses its first cycle (2026-10-20 06Z, published without soil layers) before the mesh, the static or any download, naming the `ifs` row as the remedy, on `init-source`, `global`, `mesh-plan --run` and the regional doors; a cycle before that one is refused as predating the publication. A row declares this in its new `cycles` block.
- IFS, IFS 0.1, AIFS and IFS ENS starts download only the records the init reads, and IFS inits carry the real 2 m temperature and humidity.
- Hex with Regional nest maps are headlined `nest 3 km` instead of `d01 3 km` (with woof 2.9.10).
- `woof hex render --products qpf_1h` (or any windowed product) over several history hours draws the window instead of refusing every run: naming a windowed product puts the frames in one store, as `--series` does.
- A delayed start reads hex history under its own variable names and keeps soil water and snow, so delayed-start backgrounds take their surface from history instead of the initial state.
- A regional parent run shorter than 15 steps completes: the boundary mint runs inside the boundaries the run admitted.
- `woof hex moving check --device cpu` runs in about a minute instead of 20 on one core, with the same results.
- A missing geography set or static/initial file is refused with the command that fixes it.
- `woof hex forecast` takes linked input files (`--init`, `--grid`, `--static`): it reads the files the links reach and finds the inputs staged beside the links, so an RRFS run directory of links reads its invariant file. A deck input the run reads at start and cannot find is refused by name instead of running on the initial file's land use.
- Multi-card forecasts on the MPAS-physics route run one process per card, so a plain install scales on any Python (8 cards: 0.145 s per step on Python 3.12). `--samehost-ranks threads` keeps the one-process executor.
- Multi-card forecasts on the WOOF-physics route also run one process per card, so their host work no longer shares one core on Python with the GIL (output unchanged).
- On 8 cards Thompson and the other per-card physics modules no longer reload every step: their module caches hold every card (output unchanged).
- `woof hex mesh-plan` prices a globe on the physics it will run, per card at the card count it runs on (`--cards N`), and names the fewest cards that hold it, so a 96 -> 3 km zoom globe is no longer said to fit a card it overflows.
- One-card local time stepping writes history in a child process off the step thread and no longer synchronises the card at every class step (MPAS physics 62.1 -> 58.0 s per model hour, WOOF physics 63.3 -> 57.8; output unchanged).
- Strict runs no longer compile the corrected MYNN kernel on one core at the first physics step: the kernel packs carry it, and the kernel cache is shared safely between processes.
- Strict runs on a fresh box no longer compile kernels on one core at the first step: the kernel packs carry strict images of every recorded unit (strict first step 547.6 -> 22.9 s, output unchanged), and multi-card rank threads compile their first-use modules at the same time.
- Ensemble members start faster: kernels are cached box-wide and planned once per configuration, the doors import only what the command needs, and the dycore loader reads on host threads.
- The MPAS v8.4.2 WSM6 microphysics runs on the GPU: its maths header ships with the package (it was missing since 0.4.1).
- `mesoscale_reference` (WSM6) runs about 3x faster in fast numerics: each column runs on one warp with its levels across the lanes, fast arithmetic, and the WSM6 constants initialised once per card. Strict numerics is unchanged.
- YSU boundary-layer columns run in one launch instead of 128 at a time (output unchanged): `mesoscale_reference` on a 96 -> 3 km zoom globe (149,322 cells) steps in 0.33 s instead of 1.32 s on one RTX PRO 6000, the same as `convection_permitting`.
- A `mesoscale_reference` run no longer compiles the corrected WSM6 kernel on one core at start: the kernel packs carry it.
- New Tiedtke convection keeps one pipeline per card, so same-host ranks no longer rebuild it every step or read another card's.
- The IAU increment a DA member builds at start-up is written on all cores with its header defined once, so a DA member reaches its first kernel in seconds instead of half a minute; the increment file and its receipt are unchanged.
- MPAS's own `mesoscale_reference` suite (WSM6, New Tiedtke, YSU, MM5 surface layer), Kessler and Thompson-aerosol microphysics, and the original Tiedtke and Kain-Fritsch convection run on the GPU with no native library (24 h strict, each: every word equal to native MPAS-A v8.4.2). Thompson-aerosol and WSM6 step within about 1.2x of the default suite.
- The RRFS-MPAS `rrfs_convection_permitting` and `rrfs_mesoscale_reference` selections and New Tiedtke on `conus12km` run with no native library; RRFS runs can start from a DA state stream, and the non-stationary gravity-wave drag runs on the GPU.
- New Tiedtke runs on every engine release (the timestep argument is bound under either engine name), so `mesoscale_reference` no longer stops at its first convection call.
- A limited-area MPAS-physics run takes IAU on one card, so a regional DA cycle hands its analysis to the forecast.
- RRFS `hrrrv5` runs on several cards of one host, and each rank reads only its own rows of the boundary files.
- Multi-card runs validate the device mesh once and expand output stream field lists once, so 8 same-host ranks are no longer held back by host checks; `--samehost-transport peer` is honoured instead of silently dropped. RRFS start-up runs its stages off one core.
- Multi-card runs no longer re-check the mesh and stream lists every step (r700 8 cards: 0.276 -> 0.224 s per step).
- RRFS-MPAS (`hrrrv5`) on several cards runs one process per card too: 60 km global on 8 RTX 5090s steps at 3.9-4.6 s per forecast hour (was 6.8 s), 30 km 15.2 s (was 19.5 s), and 8 cards now beat 4. Same output words as rank threads in strict numerics at a fixed card count.
- Rank processes get their start-up inputs at once (60 km 8 cards 16.2 s to 2.1 s, 30 km 70 s to 6.7 s of start-up).
- Limited-area RRFS ranks read each boundary file once per process and read the next one ahead off the step path (boundary-hour step 11.2 -> 2.3 s on 2 cards, output unchanged).
- `--samehost-transport peer|staged` is honoured on the MPAS-physics and RRFS rank routes (it was replaced by `auto`); a peer request runs the ranks as threads, where device peer copies exist.
- RRFS-MPAS `hrrrv5` on the GPU is bitwise to the native fork build in every history and diagnostics word through a 6 h forecast (strict numerics): RRTMG now reads the ozone climatology's pressure levels (they were zero, so every level got surface ozone).
- `--native-physics-tables` reaches the RRFS framework's TEMPO lookup files, and the receipt records their sha256.
- RRFS hourly maximum reflectivity and accumulated hail reset after each diagnostics write on the GPU route.
- An RRFS run with a restart stream due no longer stops at that step.
- The RRFS door reads the deck's invariant file by default (land use and soil category at sea-ice cells).
- RUC's snow temperature in thin snow no longer depends on an undefined value.
- An `hrrrv5` run on a plain install no longer stops at the MYNN-EDMF slot: its CUDA image is built for the run's device on first use.
- RRFS TEMPO microphysics is about 12x faster: one warp per column with the substeps inside one kernel. An hrrrv5 step on x1.2562 (RTX 5090, fast numerics) goes from 0.263 s to 0.084 s, with identical output in both numerics modes. The kernel joins the prebuilt kernel-pack inventory, so release packs carry it instead of a new box compiling it on one core at the first step.
- RRFS MYNN-EDMF boundary layer is about 3x faster in fast numerics and 7x in strict numerics (its operations inline, its scratch arrays laid out per warp, and its strict image compiled with optimisation), and the RRFS scalar transport runs one thread per level. An hrrrv5 step on x1.2562 (RTX 5090) goes from 0.088 s to 0.070 s in fast numerics and from 0.337 s to 0.118 s in strict numerics, with identical output in both modes.
- RRFS `hrrrv5` boundary layer and RUC land surface hand data to their schemes through persistent device buffers and a few fused copy launches per call, instead of about 2,000 small copies and launches per step, and RUC no longer reads two words back from the GPU at every call. An hrrrv5 step on x1.2562 (RTX 5090, fast numerics) goes from 0.071 s to 0.059 s, with identical output in both numerics modes.
- RRFS `hrrrv5` radiation is about 8x cheaper: the longwave slot no longer builds per-column profile views that nothing reads, and the shortwave aerosol optics run one thread per column and level. On x1.40962 (RTX 5090, fast numerics) a radiation step goes from 1.60 s to 0.51 s and the mean step from 0.98 s to 0.46 s, with identical output in both numerics modes.
- The UGWP gravity-wave drag ports ship again, so RRFS hrrrv5's drag (`bl_ugwp_gwdo`) runs on the GPU with no opt-in. `--licence-opt-in ugwp-held` is retired: a run that still names it is told so and runs normally.
- The engine programs the package bundles are rebuilt from woof 2.9.10: maps of a history that leaves the terrain unset reduce sea-level pressure from the run's initial terrain (an RRFS 3 km frame showed a tangle of isobars around an 868 hPa low), sea-level pressure is smoothed over high ground, a limited-area mesh's boundary rings are left blank, composite reflectivity includes parameterised convection, map frames are written lossless-compressed (a 3 km frame 1.23 GB -> 306 MB), history frames are written about 4x faster with the same bytes, and RRFS boundary files carry the 2.9.10 numbers.

## 0.4.9

Requires woof>=2.9.7,<2.10.

Fixed:
- The s2s door wires its ocean on the MPAS-physics route: a prescribed surface series updates SST and sea ice, and the default prognostic ocean runs MPAS's mixed layer on the GPU.
- Storm rosters on the MPAS-physics route run SPPT and SKEB with local time stepping.
- Plain hex on MPAS physics no longer stops with non-finite columns after weeks: MYNN's lowest-layer pressure thickness can no longer be zero (correct mode). A checkpoint resumes under a changed timestep or a changed correction set.
- Automatic members per card prices MPAS-physics members on their own measured memory, so 30 km globes no longer run out of GPU memory.
- A mid-window start (delayed start) writes the whole initial_time stamp.
- The MPAS-physics pool decode runs on every core, and forecasts use the reader their wheel carries.
- Zoom globes rotate a stored template mesh instead of generating one, and fetch the start source beside the mesh and static.
- Hex init holds its thread pools to the container's CPU quota, and the main thread no longer spins a core waiting on the card.
- Storm-following moves keep one ocean-ice owner, and moved sea points take the role of their carried ice.
- MPAS-route runs keep far less host memory: device views use the framework's own storage.
- Thompson microphysics caps its column workspace at 1.5 GiB per call (`WOOF_HEX_THOMPSON_WORKSPACE_MIB`), so large ranks no longer run a 32 GB card out of memory at the first microphysics step. Output is unchanged.
- A bare local-time-stepping run on a refined mesh moves its default step onto the class clock, and two LTS blocks on one card no longer race on a spec upload.
- The observation scorer runs with no card visible, so card leases can be released when the forecast ends.
- MPAS-physics steps do their per-step host bookkeeping off the step thread: a storm-following local-time-stepping run goes from 43.1 to 29.5 s per model hour on one card. Output is unchanged.
- One-card local time stepping shares one work arena across its classes and runs the limiter one species per pass, so a 48/3 km storm-following run fits a 32 GB card.
- A global member that does not fit one card runs over several cards on the same host, and the next pipelined member's host memory is priced per rank.
- The MPAS-physics route applies `--acoustic-substeps`, `--time-integration-order` and `--monotonic`.
- A bare local-time-stepping run picks its step for the mesh's real coarsest class, so a 96 -> 3 km zoom globe steps 15 s instead of 12.5 s.
- The zoom globe's products pass keeps the column stores the live loop already wrote instead of rewriting them on one core.
- A stalled source download is retried on a new connection after 10 s instead of waiting 300 s.
- Local time stepping picks the fastest class plan for the mesh and runs the plain fixed step where that is faster; on a 96 -> 3 km zoom globe a bare LTS request now runs about 1.2x faster than the fixed step on MPAS physics.
- MPAS-physics local time stepping deposits and assembles its history frames off the class step threads, so the classes no longer wait 1-2 s at every frame.
- MPAS-physics start-up and history output no longer wait on one core: the pool decode, init reads, region preparation and frame encoding run in parallel (a 30 km member over 4 cards starts in 85 s instead of 112 s; output unchanged).
- Zoom-globe, global, regional and country runs give their card back to a card queue when the forecast ends and draw their products on the CPU.
- Local time stepping runs its rate classes on the faster dycore kernels (about 1.2x faster on a 3 km zoom globe).
- A storm-following move no longer stops the run when a member reused from disk is ready but unsafe for the storm, and a case with no safe radius is refused when it is loaded.
- Packaging links the pinned netCDF for every helper, and two exit-drop tests no longer race.
- Kernel packs (`kernel-packs-v0.4.9`) carry the corrected MYNN kernel, so a new box does not compile it on first use.

## 0.4.8

Requires woof>=2.9.7,<2.10.

Fixed:
- A global ensemble retries a member stage that fails once, and a member that still fails is recorded as missing while the other members finish.
- MPAS-physics runs that update SST, sea ice, greenness and deep soil from a surface series keep that update on the GPU instead of asking for the native MPAS library.
- A test now guards the Grell-Freitas scratch fix from 0.4.6: no GF output may read a workspace word before writing it.

## 0.4.7

Requires woof>=2.9.7,<2.10.

Fixed:
- Storm-following runs no longer refuse a move with "FP32 moving integral cannot meet its conservation tolerance".
- A cycle that is not posted yet is refused naming every address it tried (NOAA first) and when it posts; archives that lag a cycle are skipped for younger cycles.
- A start file without a 2 m analysis (IFS starts carry q2 as NaN and t2m as 0 K) takes its 2 m fields from the first physics call, so ensemble products no longer fail with "history q2 is not finite".
- A run's exit no longer waits on an idle kernel-compile worker that ignored the stop signal.

## 0.4.6

Requires woof>=2.9.7,<2.10.

New:
- MPAS physics is the default on every door: a run that names no physics runs MPAS-A's own `convection_permitting` suite on the GPU (the suite follows MPAS-A practice by mesh, `data/profiles/physics-default.v1.json`). WOOF physics stays one flag away: `--physics-suite arwen_mpas_column` (`--physics-row arwen` on `global`).
- The cycle door takes an MPAS physics forecast as its coarse and parent stream.
- Kernel packs (`kernel-packs-v0.4.6`, CUDA 12.9, 13.0 and 13.4) carry the MPAS physics kernels, so a new box starts MPAS physics runs without compiling them.
- Storm- and setup-following runs start from a premade mesh rotated to the storm (`moving template` builds one), and the static builder maps geography to global meshes about twice as fast, with identical output.

Fixed:
- MPAS physics runs with local time stepping no longer fail at the first step on a new box (Noah kernel cache).
- `forecast --receipt` writes its receipt on the MPAS physics route.
- Storm-following and other MPAS physics runs no longer stop with a non-finite state after a few hours (Grell-Freitas shallow convection read an unset cloud-top value).
- Local time stepping on two or more cards gives the same results as one card, on both physics routes; MPAS physics runs it on several cards.

## 0.4.5

Requires woof>=2.9.7,<2.10.

New:
- The MPAS physics route runs every hex shape: limited-area regions, storm-following regions and ensembles, local time stepping, Hex with Regional, add-nest, restart, IAU, stochastic physics, chemistry, member rosters and two or more cards, with no native library.
- The MPAS physics route is faster: the host is off the step path, and Thompson, MYNN, Grell-Freitas and radiation run their GPU fast paths. Strict numerics still matches native MPAS-A bit for bit over a whole 24 h convection-permitting forecast.
- Hex with Regional in three shapes: a nest placed at launch, a nest added mid-run, and a nest that spawns when a storm attribute you set crosses its threshold.
- Doors start from any posted forecast hour with `--lead-hours`, from GFS, GEFS, IFS 0.25 and 0.1 degree, IFS ENS, AIFS and AIFS ENS.
- Regions take drawn shapes (`--shape`, `--ellipse`), open-ended resolution, and can be kept as inits.
- Every run writes the any-point column store for every output hour, so soundings can be drawn anywhere after the run.
- Maps stream every hour on region doors and ensemble members too.
- GSL drag statistics are built for any hex mesh.
- The MPAS route writes reflectivity (REFC) from its diagnostics stream.
- Moving regions: `--lead-hours`, `--from-last-hour`, and following targets set at launch.

Fixed:
- 2-card runs: per-card modules, the compile check, the Noah stack limit, the Grell-Freitas halo and a local-time-stepping race.
- Kernel packs match the loaded NVRTC on new boxes, with a CUDA 13.0 row and all-cores compile.
- f000 carries the first physics call's surface diagnosis (10 m wind, fluxes, PBL height) on both routes.
- Local time stepping on a limited-area region imposes the interface flux.
- Storm-following runs keep their rendezvous at every check.
- Off-mesh sounding points are dropped with a note; limited-area runs default to cities on the mesh.
- A geography folder with a hidden file is accepted.
- `--skip score` queues no scorer, and the scorer never waits for frames valid in the future.
- Sonde scoring reads the year-to-date IGRA2 snapshot.
- GFS fetches fall back to the older AWS layout and the NCAR GDEX archive.
- A hydrometeor number neither carried nor diagnosed is never published.
- Region prepares build the parent's static from the region's halo instead of the whole globe.
- Storm-following segments resume on every core, and an MPAS-route segment moved over a geography with lakes no longer stops at its first land-surface call.
- MPAS-route exports carry terrain height and reflectivity, so keep-as-init, the column store and Hex with Regional work on that route.
- The WOOF route runs on woof 2.9.7.
- Moving runs on column suites (HRRR shape) and over sea ice move past their first move, and a new member's sea ice brings its own albedo, emissivity and ice fraction.

## 0.4.4

Requires woof>=2.9.4,<2.10.

New:
- Global and variable-resolution forecasts draw their first maps the moment the forecast writes its first frame, instead of after the whole forecast.
- 1 km limited-area runs start faster: a regional target of a global parent reads only the parent cells around it, boundary frames are remapped concurrently, and a premade variable-resolution mesh and its static are fetched instead of rebuilt.
- First maps open on 2 m temperature, 10 m wind and CAPE, and on reflectivity from its first echo.

Fixed: every hex map carries a labelled latitude and longitude graticule.

Fixed: `woof hex global`, `mesh-plan`, `regional`, `country`, `moving` and `forecast` take `--model-label` and `--brand` and pass them to the point soundings and meteograms; with woof 2.9.6 or later the brand is drawn in bold at the top right.

## 0.4.3

Requires woof>=2.9.4,<2.10.

Fixed: global and subseasonal forecasts with sea ice no longer stop at their first step on woof 2.9.4, and `woof hex s2s` runs of any length finish their energy products.

## 0.4.2

Requires woof>=2.9.4,<2.10.

New:
- `woof hex regional`: one-command limited-area forecasts from HRRR around a point, with mesh, boundaries, products and a stage receipt.
- Limited-area runs over any named region through `woof hex country --region`.
- Variable-resolution global meshes from `woof hex mesh-plan --global-mesh --generate --run`.
- Storm-following regions authored from tracks, best tracks or IBTrACS with `woof hex moving case`, `run` and `products`.
- Storm-following ensembles with `woof hex moving-ensemble`, including SPPT and SKEB under local time stepping.
- Mixed-layer ocean and sea ice on the mesh columns, on by default in subseasonal runs.
- Aerosol transport (advection, boundary-layer mixing, deposition, settling) in global forecasts, and `woof hex chem-sources` to prepare RAVE and GBBEPx smoke, GEOS-FP composition, dust and sea-salt sources.
- Subseasonal products: weekly means, anomalies, climatology and terciles, member history export, MJO and BSISO indices, weekly scores against ERA5, and hub-height wind and solar energy tables.
- An 80-level high-top row, `global-120km-high-top`, for stratospheric runs.
- Keyless ERA5 initialization from the public ARCO store as `--source era5`; the CDS route is `era5-cds`. Lagged ERA5 analysis members for subseasonal ensembles.
- IVT, freezing height and the 300 hPa jet in the default global products; 850 and 925 hPa winds in regional products.

Fixed:
- Observation scoring fetches the on-demand engine bridges (rw_asos, rw_prepbufr) before scoring.
- Radiosonde cells at hours other than 00Z and 12Z are reported as not applicable.
- An interrupted mesh download resumes.
- Global ensembles pipeline member groups on each card, size groups by the card's memory, and name each member in its maps.
- Subseasonal runs accept any length of 7 days or more; `--members` takes a total count.
- `woof hex physics list` and `resolve` report the MPAS suite routes the forecast door uses; a suite named through `--forecast-arg` is the planned and recorded physics.
- Storm-following members read the selected suite's sea-ice threshold.
- `woof hex moving run --resume` redraws the products of a segment whose product step was interrupted.
- PyPI uploads run only for version-tag releases.

## 0.4.1

Requires woof>=2.9.2,<2.10.

New:
- MPAS-A v8.4.2 dycore, GPU physics and LES closures, with four acoustic substeps and a native-free framework. WOOF physics remains the default.
- MPAS physics ports: WSM6, Thompson, Kain-Fritsch, Tiedtke, Grell-Freitas, revised MM5, MYNN, YSU, Noah, sea ice, RRTMG and CAM.
- Real-data and idealized initialization, vertical grids, orographic statics, aerosol climatology, SST and sea-ice updates, IAU, limited-area boundaries, streams, restart and diagnostics.
- Correct mode for MPAS and WOOF physics, with named corrections, switches and receipt records.
- Global and country workflows, source-table initialization, analysis bundles, mesh remapping, cycling and subseasonal ensembles.
- Forecast products, soundings, meteograms, NetCDF export and observation scorecards, including IGRA2 radiosondes for past cycles.
- Ensemble and preset preparation, member queues, shared static caches, a pinned preparation store and suite-aware soil and aerosol initialization.
- Storm-following regions with balanced conservative remapping, physics-state carry, restart and same-host multi-card execution.
- MPAS physics tables fetched from pinned data-only assets in `physics-resources-v0.4.1`.
- CUDA 12 and CUDA 13 installation extras, driver-aware diagnostics, bundled helpers and a manylinux_2_28 wheel.
- Asynchronous history output, cached input digests, parallel host preparation, fine-terrain sampling and diagnostic export.
- Kernel packs fetched on demand from `kernel-packs-v0.4.1`; `woof hex fetch-assets --from DIR` and `WOOF_HEX_ASSETS_FROM` support offline staging.

Fixed:
- MPAS shortwave shape specialization and YSU memory probes support woof 2.9.3.
- Physics routing, vegetation, soil columns and variable-mesh convection follow the selected suite. MPAS physics without a namelist selects `mesoscale_reference`; default YSU timesteps respect engine limits and explicit selections.
- WSM6 precipitation checks accept valid step scalings; limited-area forecasts avoid the first-step buffer failure, and CUDA graph replay refreshes heating and tendencies.
- Same-host forecasts handle IAU, per-card kernels and memory admission; local time stepping supports even acoustic counts and exchanges multi-card interface fluxes.
- Moving remaps extend below-ground states hydrostatically and evaporate subsaturated cloud water; unsupported move hooks are refused by name.
- Static construction preserves grid precision and maps lake categories; vertical templates use the bundled CDF-5 copier.
- Installed country, products and preset commands resolve their helpers; TEMPO checks aerosols before fetching tables, and table generators use pinned logarithm arithmetic.
- Strict checks compare data and attributes without file identifiers, authenticate installed engine wheels by archive and RECORD hashes, and use bundled float32 geometry math across host glibc versions, built from the shipped sources in source checkouts; reference checks include staged changes, shell isolation, timestep and dycore anchors.
- Windows source installs work without bash or netCDF-C development files; kernel and physics asset fetches, initialization, subseasonal runs and cycling use portable file locks.
- RRFS-MPAS selections name the required site build when their native library is unavailable.
- Public exports exclude UCAR Noah-MP v5 and UGWP implementations, with licence gates, native stubs and dependency notices.
- The public release workflow builds the wheel: the renderer and surface-series helpers download by pinned hash beside the engine bridges. Packaging includes helper licences and METIS, checks sizes and excludes kernel packs. Fetches verify whole batches; doctor recognizes offline packs and reports MPAS physics-resource readiness. Release checks validate engine availability and public CI before preparing a cut.

## 0.3.3 (2026-09-29)

Requires woof 2.8.0 (woof>=2.8.0,<2.8.1), as 0.3.2 did.

New:
- Every NVIDIA architecture NVRTC can compile the kernels for is admitted
  by default, with no flag: compute capability 7.5 (Turing) and newer on
  the CUDA 13 toolkit. sm_86 and sm_89 hold anchors, sm_120 is the proven
  contract floor (anchored, with no pin: the contract was first measured
  there), and a card of any other architecture runs unanchored and its
  receipts say so.
- sm_89 holds an anchor (an RTX 4090, 2026-09-24): the numerical contract
  measured against sm_120, 123 records identical, with two named
  deviations: the guard-cost timing control (timing only, 1.274725x
  against the borrowed 1.25x ceiling, bitwise identity held) and the
  v8.4.1 four-pass audit (inherited, not measured on the card: it refuses
  before any device work on every card at that tree, and the
  release-specific kernels were measured directly). Its own guard-cost
  ceiling is 1.55. Each anchor row pins its evidence by SHA-256
  (ArchAnchor.evidence_sha256).
- The preflight prints one line per card: anchored, with the evidence pin,
  or unanchored. Every receipt (door, driver, proof harness) records the
  compute capability and the same status.
- On an unanchored card the door measures the numeric route before the run
  in one tiny compile and launch (the FTZ route, the guarded subnormal
  fallback, the --fmad=false pin, IEEE ordinary arithmetic) and records it.
- `woof hex doctor` names the card, its compute capability and its anchor
  status. An unanchored card is never a gap; the one gap it reports is
  the forecast door's own refusal, a card NVRTC cannot compile for.

Fixed:
- An architecture holding no anchor was refused ("holds no
  per-architecture anchor"), a refusal that named no breakage.
  `woof hex forecast` now meets two architecture refusals, each naming
  one: a card NVRTC cannot compile for, and ordinary arithmetic that is not
  IEEE.
- The forecast driver and the x4 proof harness pinned sm_120
  (required_compute), which refused sm_121 and every newer part. Sixteen
  tools under tools/ did too and no longer do. The pin stays only on the
  two FTZ instruments and two tools whose bytes a recorded pin holds,
  where it refuses only parts newer than sm_120. Two-GPU runs are not a
  supported path in this release: the two-GPU forecast tool
  (run_cuda_v841_forecast_2gpu.py) is unchanged and still refuses a card
  below compute capability 12.0.
- `woof hex render` refused every frame with "rw_wrfbatch produced no
  catalog rows" on woof 2.8.0, whose renderer prints six fields per
  catalog row. The door reads five or more.
- The hex tests find the package by its import name, so they hold in any
  layout the package is installed or folded into.
- Every forecast's start frame (F000) published q2 as 0 and psfc as
  100000 Pa in every cell, the physics seam's values before its first call,
  so the hour-0 2 m dewpoint map was drawn from zero humidity while every
  later hour was right. The start frame now takes q2 from the start file
  when the start file carries it, and otherwise from the lowest model level
  bounded to saturation at the 2 m temperature (a specific-humidity start
  such as HRRR's leaves the start file's q2 at zero); psfc is the model's
  own surface pressure. Later frames and the model state are unchanged.

## 0.3.2 (2026-09-29)

New:
- woof hex forecast --preset NAME selects a row of woof.hex.forecast_preset,
  and the door turns the row into the surface/PBL cadence it already admits
  (an explicit --pbl-cadence still wins; the receipt says which decided).
  reference, the default, is the proven configuration. fast holds the
  surface layer, Noah-MP and YSU for up to 30 s between calls (six steps at
  dt 5 s, the weld where dt is 20 s or longer). Graded against MRMS
  reflectivity, Stage-IV hourly precipitation and ASOS on four 3 h
  HRRR-driven forecasts of a 937.5 m point cull, fast took about 6 per cent
  less time per step and forecast 10 m wind worse on all four cases and 2 m
  dewpoint and temperature worse on three, so reference stays the default.
  A row declaring a radiation, land-surface, surface-layer or PBL scheme the
  engine's column seam does not take as an argument is refused by name.

Changed:
- The physics seam re-pins to woof 2.8.0: woof>=2.8.0,<2.8.1. Eight of
  the sixteen pinned engine files moved between the published 2.7.4 and
  2.8.0 wheels (woof/config.py, woof/core/gf.py, woof/core/kernels/gf.cu,
  woof/core/kernels/__init__.py, woof/core/microphysics.py,
  woof/core/physics.py, woof/core/rrtmg_legacy.py, woof/io/restart.py);
  the column batch, the seam document and the Noah-MP files did not, and
  the composed glacier unit is unchanged. Measured through the seam on an
  RTX 5090: the seam A/B is 500 of 500 arrays byte-identical on both
  profiles and both arms under the published 2.7.4 and 2.8.0, and a
  10-minute forecast on a 12,795-cell 937.5 m point cull wrote both history
  frames byte-identical under the two engines; the cull's contract deck is
  8 of 8 decks bitwise and 22 of 22 kernels at the unchanged kernel-set
  digest. Nothing is declared for the move.
  verification/engine-verdicts/repin-280-20260929.json is the table of
  record; premeasure-280-20260929.json shows 2.7.5, 2.7.6, 2.7.7 and 2.8.0
  all failing the 2.7.4 manifest, so 0.3.1 installs kept resolving 2.7.4.
- The physics seam's per-step rollback export is taken only under
  --stop-on-refusal, the one option that reads it. It copied every persisted
  seam array to the host before every step (447 MB and 192 copies a step on
  a 43,884-cell mesh, about 24 ms of host time). Measured on that
  mesh, one session on an RTX 5090: 0.2678 to 0.2367 s a step, 3.21 to 2.84
  minutes per simulated hour. Every history
  frame is byte-identical. Without --stop-on-refusal a refused step ends the
  run as before, and the receipt records the seam as retired rather than
  restored (physics_rollback in the driver receipt).

Fixed:
- woof hex doctor reports a staged bridge that is present and not
  executable as missing, with chmod +x as its remedy. It reported the file
  found while the door refused it (measured on a pip install of 0.3.1,
  whose engine wheel's bridges arrived mode 664).
- The forecast receipt's arwen_commit and glacier digest name the engine
  the seam bytes matched, with its version beside them. They restated the
  x4 proof's constants, which name an older engine than the one that ran.
- Receipts from the forecast, init and pair doors, the contract deck and
  four measurement tools name the machine by a salted digest
  (woof.hex.host_identity, a per-user salt in ~/.woof/hex-host-salt)
  instead of its network name.
- The point door's printed forecast command no longer adds
  --gpuwm-checkout at a fixed engine version or the unused --repo, and two
  ingest refusals stop naming an engine version.
- Evidence citations in row notes and comments name the evidence folder
  instead of a machine label.
- Two tool pins that the text cleanup edited without re-deriving are
  re-derived: SOURCE_DRIFT_SINCE_CAMPAIGN.current in
  tests/test_regional_forecast_anchor.py and EXPECTED_RUNNER_SHA256 in
  tools/run_cuda_x1_163842_stabilized_products.py.
  tools/repin_source_tables.py reported both as drift and now reports none.
- woof hex forecast runs from an install. The three modules the door
  drives (the forecast driver, the proof harness it reaches the model
  through, and the mesh registry) and the limited-area contract deck ship
  in the package as woof.hex.drivers; through 0.3.1 they lived in the
  repository's tools/, which the wheel does not carry, and the door refused
  every install. tools/ keeps a same-named entry for each that hands back
  the packaged module. --repo is accepted, reported and not used.
- The physics seam defaults to the installed woof. --gpuwm-checkout is
  optional: with none given the run verifies the installed engine's sixteen
  pinned files and records it by version, the SHA-256 of pip's RECORD and
  direct_url; a git clone passed as --gpuwm-checkout is still recorded by
  HEAD, tree and dirty paths, and any other directory is refused by name.
  doctor and the pip remedy stop sending users to clone woof. Measured
  2026-09-26 from a wheel built from this tree and installed with pip into
  a clean venv, standing outside any checkout: the contract deck on a
  937.5 m point cull passed (8 of 8 decks bitwise, 22 of 22 kernels) in
  126 s, and a 10-minute forecast on it ran 120 steps of 5 s in 50.0 s of
  integration (73 s door wall, shared RTX 5090), receipt kind "installed".
- The cycle door no longer requires --gpuwm-checkout, runs the contract deck
  from the package, and its subprocesses import the hexcore the parent runs.
- Further physics-backend rows arrive through the woof.hex.rows entry-point
  group (a module that registers on import, or a callable returning rows);
  the row table names no provider module, a provider still cannot rebind the
  default row, and a declared provider that fails to load is named in the
  refusal for the row it would have carried.
- The published admitted-engine table is empty: a row names a build that is
  not on PyPI and belongs in that build's own tree. The mechanism and its
  tests are unchanged (the tests use a synthetic row).
- tools/point_source_placement.py places a tracer release line upwind of a
  target the user names, in the control leg's own wind (`place --target
  LAT LON --alt-m Z`), and writes the release table as before; the
  cloud-water targeting census is gone from the public tool. The pair
  gallery and its tests say control and treatment legs and tracer fields.
- --convection's help states the default and its reason: the cumulus
  scheme is off where the mesh's finest spacing is below 3 km, because
  Grell-Freitas is called every step and a fine mesh's short step calls it
  up to 24 times as often as the proven 120 s, which measured as a
  different solution.
- NOTICE credits the MPAS-Tools grid_rotate.f90 longitude formula that
  woof.hex.mesh reproduces, the MPAS-Atmosphere v8.2.3 loop bodies the
  Fortran oracles transcribe, and the WRF lineage of the gravity-wave-drag
  operator.
- Documentation and messages: private task numbers are replaced by the
  named breakage or instrument they cite, private evidence-folder paths,
  machine names and branch names are gone from the docs (the measured
  results stay), and the README headline and the repository README's
  engine range are current.
- Not changed, with the reason: four task-number citations remain in
  woof/hex/partition_device_scheduler_v841.py and a receipt path in
  woof/hex/dt_admission.py, whose bytes the frozen execution set pins by
  SHA-256, so moving them needs the pins re-derived and re-proven on a
  card. The architecture
  anchor record in woof.hex.cuda_backend.arch_admission keeps its receipt
  path because that path is the record's identity, printed on the
  ARCHITECTURE line and asserted by its tests.

## 0.3.1 (2026-09-15)

New:
- The physics seam re-pins to woof 2.7.4. The sixteen-file manifest
  carries the published 2.7.4 wheel's bytes (tag ed957997, whose tree
  agrees with all three platform wheels on every pinned path) and the
  derived range is woof>=2.7.4,<2.7.5. Two of the sixteen paths moved
  between the published 2.7.3 and 2.7.4 wheels, measured wheel against
  wheel (verification/engine-verdicts/repin-274-20260915.json): the config
  loader gained one settings-map reader of the radiation rule that the
  port never calls, and the microphysics driver passes two keyword
  arguments on the classic Thompson path, a scheme the port's WSM6 backend
  never runs; the other fourteen, the composed Noah-MP glacier unit
  included, are byte-identical. Measured the other way first, against the
  manifest 0.3.0 carried (premeasure-274-20260915.json), 2.7.4 fails by
  those two paths, which is the <2.7.4 ceiling holding installs on 2.7.3
  for the two days between the 2.7.4 publish and this re-pin. Under the
  re-pinned manifest 2.7.3 fails by the same two and every earlier engine
  by exactly the count it failed the 2.7.3 manifest by. What moved through
  the seam is measured and recorded in docs/declared-divergences.md: the
  seam-level A/B under both engines on both fixed-column profiles and the
  one-hour point-cull forecast under 2.7.4 against the 2.7.3 frames of
  record.
- The admitted private row is re-taken on the 2.7.4 base
  from its three platform wheels and its source tree (which agree with
  each other and with the tree's HEAD): the same five of sixteen paths
  differ from the public bytes as before and the composed glacier unit is
  the pin's. The row on the 2.7.3 base is dropped with the range, because
  pip no longer resolves 2.7.3+x.1 inside woof>=2.7.4,<2.7.5 and a row
  whose base the specifier refuses admits nothing; tests/test_engine_pin.py
  refuses such a row by name. The forecast driver's checkout guard is
  unchanged and follows the admitted row as before.
- The surface/PBL cadence is a selectable knob. woof hex forecast
  --pbl-cadence SECONDS calls the surface layer, the land-surface model and
  the PBL once every SECONDS/dt steps and holds the tendency between calls
  (the engine's own positive-bldt path); radiation keeps its own 600 s
  cadence. The default stays the weld (bldt = dt, the native x4 reference's
  semantics) because a held cadence changes the forecast; the run receipt
  records the decision (calls per hour, held steps, source) and the driver
  receipt carries the seam's own due/held count against the declared
  cadence (physics.cadence). Measured on the point mesh in
  docs/hex-point-hrrr.md: 30, 60 and 120 s each buy the same five per cent
  per step and each moves the one-hour forecast well past the 2.6.5 to
  2.7.3 seam movement, so none qualified as the default; the tree at auto
  is byte-identical to the tree before this change.
- A held cadence at an anchored timestep and cumulus selection is admitted
  through a row derived from the welded anchor (woof.hex.dt_admission.
  derived_held_cadence_anchor): the dycore half is the welded row's own,
  the host half is re-minted for the held cadence, the physics band is
  stamped NOT MEASURED, and the receipt names the welded row it derives
  from. A registered held row still wins the exact lookup; a held cadence
  at an unanchored timestep is refused as the welded run would be.
- woof.hex.cuda_solve_region_v841: four device kernels that check the solve
  region of a limited-area array and read back once. The per-step health
  gate takes the envelope of every field (min, max, a NaN verdict, the
  first-index argmax of |w|) from one launch pair and one read instead of
  31 strided CuPy reductions each drained to the host; the regional
  density and recovered-state validations set the step's flag on the
  device instead of materialising full-size temporaries and draining per
  array; the garbage discipline restores every padded argument of a
  launch in one launch instead of one cupy fill per column. The kernels
  compare, select and store what they are handed and do no arithmetic on
  the values, so the hex self-reproducibility is unchanged by construction;
  the measuring and audited forms of the discipline keep the per-column
  fills.
- launch_checked in woof.hex.cuda_backend: one launch, the error state
  checked, no event and no synchronize. recover_state uses it by default
  (timing_repeats=0); the timed path stays for the benchmark.
- tools/lts_forced_classing.py and --local-timestep-classing FILE on the
  forecast door and the driver tool: an A/B instrument that places a rate
  class where the spacing would not, recorded in receipts as
  rate_source: explicit.

Fixed:
- --pbl-cadence that is not a whole number of steps is refused by name at
  the door (both numbers, the multiples of dt on either side, and auto),
  on the run route and in --preflight, instead of tracing back after the
  mesh is bound.
- A forecast step no longer measures the CUDA admission fifty-nine times.
  require_cuda memoises a granted admission per argument set (a refusal is
  never memoised, and a stubbed cupy never inherits a real card's answer);
  the dycore's translation units re-entered it on every launch group at
  1.05 ms a call on the CUDA 13.3 driver.
- Every recovery kernel is launched once. The driver's two recover_state
  calls went through the timed launch path with timing_repeats=1, which
  launched each kernel twice with two event synchronizes and a stream
  synchronize: 36 host stalls and 10.7 ms of repeated device work a step.
- Measured on the 43,884-cell 937.5 m point cull (55 levels, dt 5 s, HRRR
  15Z, one hour = 720 steps) on an RTX 5090, both arms in one session on
  the same card: the composite step 0.3143 s before, 0.2957 s after
  (1.06x; integration 228.7 s to 215.3 s), the health gate 56.2 s to
  0.2 s an hour, the loop as a user waits for it 0.3973 to 0.3011 s per
  step (1.32x; door 328 s to 259 s), both history frames
  sha256-identical between the arms and across two hostpath runs; the
  contract deck on the cull's own rings passed at the new kernel-set digest
  (8 of 8 decks bitwise, 22 of 22 kernels, dual-run identical). Evidence:
  the hostpath folder of the evidence gallery, 2026-09-13.
- The regional kernel-set digest moved on three host sources
  (cuda_backend/runtime.py, cuda_driver.py, cuda_regional_forecast_v841.py)
  and no CUDA source string; the constant is re-derived, the one class row
  that carried a literal follows the constant, and the note at the constant
  records what moved and that the point class was re-minted at it.
- The campaign's changes measured together: the tree with the cadence, hostpath
  and local-time-stepping changes merged runs the 3 h point-cull forecast (2,160 steps, HRRR
  15Z, RTX 5090, both arms in one session) at 0.2978 s per step against the
  untouched 0.3.0 tree's 0.3193 (1.07x; the loop as a user waits for it
  679.6 s against 864.7 s, 1.27x; door 728 s against 907 s), with all four
  history frames byte-identical, the contract deck at the merged kernel set
  8 of 8 decks bitwise and dual-run identical, and the peak card memory
  unchanged at 9,485 MiB. The merged tree's kernel-set digest is the
  hostpath re-mint's. docs/hex-point-hrrr.md, the integrated section.
- Every level-independent dycore kernel launches one thread per
  (tracer, level, owner) element. Eight translation units
  (cuda_regional_v841, cuda_dynamics_v841, cuda_driver, cuda_acoustic,
  cuda_horizontal, cuda_horizontal_v841, cuda_transport,
  cuda_backend/recovery) walk a flat element index in a grid-stride loop
  instead of one thread per owner looping over the levels, and the
  forecast launches them per element; each element runs the former loop
  body for its level with the same operands in the same order, the
  vertically implicit column solve keeps its per-column form, and the
  momentum kernel gathers a once-per-run reference wind instead of
  evaluating cosf/sinf per neighbour per stage. The local-timestep unit
  gathers the two regional acoustic kernels it derives in their element
  form. Measured on the 43,884-cell point cull (RTX 5090): the change's own
  one-hour pair in one session 0.3146 to 0.2836 s per step (1.11x) with
  both frames byte-identical; the landed tree's 3 h arm (2,160 steps)
  0.2617 s per step after the first against the integrated tree's 0.2978
  (1.14x) and the untouched tree's 0.3193 (1.22x), the loop as a user waits
  for it 572 s against 680 s and 865 s, all four history frames equal to
  the digests both earlier arms recorded, peak card memory 8,914 MiB; the
  contract deck on the cull's own rings at the merged kernel set 8 of 8
  decks bitwise, 22 of 22 kernels, dual-run identical; a per-kernel
  capture-and-replay of all 49 re-mapped kernels launched by the run
  bitwise against the previous translation units. The boundary-zone
  kernels gain 5x to 17x per launch, the level-heavy cell kernels 1.3x to
  2.6x; a dozen small kernels are slower per launch in element form
  (named with their numbers in docs/hex-point-hrrr.md, the kernels
  section) and stay as follow-ups because they are bitwise and the
  composite step is what moved.
- tools/kernel_element_ab.py: a per-kernel capture-and-replay instrument
  that runs the forecast door in-process, captures the first launch of
  every re-mapped kernel after the warm-up steps with a device copy of
  every argument before and after, and replays this tree's kernel and the
  untouched tree's translation unit from the same inputs on the same card,
  byte for byte and event-timed. Its first run reported five kernels as
  differing whose arithmetic was byte-identical: four in the 55 values of
  an input's garbage column, rewritten by the post-launch scrub with the
  step's unit pool released, and one whose untouched form was launched at
  one owner because its last device argument is the (1,) invalid flag.
  The recorder now asks the garbage discipline which arguments are bound
  at the launch (bound_to_unit_pool) and re-binds them for every replayed
  launch, the owner-count rule names the edge count, every difference is
  also classified by garbage column, and tests/test_kernel_element_ab.py
  parses every target's signature so a flag or index array can never be
  the owner axis. tools/nvrtc_compile_check.py resolves every kernel of
  the eight units before card time is spent.
- The regional kernel-set digest moved again, this time on the CUDA source
  strings themselves (six of the fourteen sources), and once more on the
  landing that carries the host-path move and the re-map together; the
  constant is re-derived at the landed tree, the note at the constant
  records each move and that the four non-point classes carry the constant
  without a re-mint of their own, and the point class row records the
  re-mint the landing measured.
- The NVRTC reciprocal-rewrite census names woof.hex.cuda_solve_region_v841,
  which the census's own control (every CUDA-bearing module under src must
  be a listed translation unit) found unregistered on the GPU tier's first
  pass over the merged tree.
- The four point-door test files are named once on the CPU tier list; two
  changes had each added them and the tier census refuses a file named twice.
- tests/test_hrrr_intermediate.py, test_mesh_point.py, test_mesh_rows.py and
  test_regional_class_band.py are named by the CPU tier list; they arrived
  with the point door on no list, which tests/test_tier_membership.py
  refused.
- --local-timestep runs on a limited-area cull. The classing admits a
  ring-7 edge's missing cell, the attachment counts the regional memory
  model's padded cell and edge, the regional acoustic sub-step is rebound
  per rate class (its three kernels derived from the regional translation
  unit by the same asserted gather), every driven boundary cell is held at
  rate 1 and a class interface touching the driven zone is refused by
  name. On the culls the doors make this leaves one class and the run is
  byte-identical to the default; measured on the RTX 5090, with an
  interface forced into the interior it is 1.07x slower, so the option
  stays off by default on every mesh class. docs/local-timestep-lam.md.
- An explicit local-timestep classing's receipt reports the instrument's
  request as cells_requested instead of cells_qualified_by_spacing.

## 0.3.0 (2026-09-13)

New:
- A fine hex core at any point, from HRRR. woof hex mesh-plan --point
  LAT,LON writes the registered 937.5 m ladder spec centred on the point,
  prices parent and cull for a named card, and with --generate builds,
  admits, registers (a runtime row file beside the mesh, woof.hex.mesh_rows,
  read by tools/mpas_mesh_binding.py through $WOOF_HEX_MESH_ROWS) and
  culls it, minting the parent's vertical and culling that too. woof hex
  intermediate --source hrrr writes regular lat-lon WPS intermediates from
  the projected HRRR product through the engine's own decoder, operator and
  soil layering; woof hex lbc drives rw_mpas_lbc over the hourly
  intermediates. rw_mpas_static joins the engine ladder. The two
  sub-kilometre regional classes admit a same-ladder finest-edge band of
  five per cent. docs/hex-point-hrrr.md.
- The engine-verdict instrument ships in the tree. tools/measure_engine_verdicts.py
  downloads every wheel of every published woof from 2.5.0 up, hashes the
  sixteen pinned seam files inside each, reads the build road off the
  engine's tags, and writes the JSON that woof.hex.engine_pin's table is
  spliced from (verification/engine-verdicts/). The table gate in
  tests/test_engine_pin.py now runs on the published tree and in the sdist
  instead of skipping there, and a re-pin is one command
  (--pin-to VERSION --write-manifest --splice) rather than hand-typed
  digests.
- A seam-level engine A/B. tools/measure_engine_seam_ab.py drives the
  engine's column-batch seam on fixed columns under two engines and
  reports, field by field, what moved and by how much, with a no-cumulus
  control arm and a capped-profile control. It is how this release states
  its physics movement below, and how a re-pin taken without the x4 assets
  and a 32 GiB card can still say what changed.
- A private engine with a measured row is admitted. An engine that is not
  on PyPI, carrying a PEP 440 local version label on the pinned base,
  already satisfies the declared range (2.7.3+x.1 sits inside
  woof>=2.7.3,<2.7.4); what decides whether this port runs on it is now
  the same thing that decides for a published engine, its sixteen seam
  bytes. tools/measure_engine_verdicts.py --admit hashes those files
  inside the engine's own wheels and source tree (which must agree with
  each other and with the git tree's HEAD), computes the composed glacier
  digest by the engine's own hand, and writes the row to
  verification/engine-verdicts/admitted-engines.json, spliced into
  woof.hex.engine_pin.ADMITTED_ENGINES and gated by tests/test_engine_pin.py
  like the published table. The seam check, the forecast door, doctor and
  the adapter's checkout guard accept a tree whose bytes are exactly that
  row's under that version string, and every receipt and restart identity
  then names the row's build commit and digests instead of the public
  pin's; the same bytes under another version string, that version string
  over other bytes, and any engine with no row are refused by name as
  before. One private row shipped on the 2.7.3 base: five of the sixteen paths differ
  from the published 2.7.3 bytes (the config loader, the kernel loader,
  the microphysics driver, the phase-one driver and the restart identity
  table) and the composed glacier unit does not. On the public 2.7.3
  engine nothing this release writes changes.

Fixed:
- The forecast driver's checkout guard follows an admitted engine's row.
  verify_arwen_checkout_git in tools/run_cuda_v841_full_physics_x4.py read
  the public pin's sixteen digests unconditionally, so on the admitted
  private engine the frozen row's --preflight refused with
  "woof/core/physics.py does not match the proven manifest" while the
  adapter, the forecast door and doctor admitted the same tree (measured
  2026-09-13). The guard now asks engine_pin.inspect_seam which row the
  tree's bytes are and gates on that row's digests; a tree matching no
  row is still measured against the public pin and refused naming the
  pin's digest, as before. Three tests in tests/test_proof_guard_pins.py.
- woof hex init stamps the run declarations the engine's capsule copy
  left absent (config_start_time, the first-guess level counts, the
  moisture, sea-ice, extrapolation and land-use switches) onto the init on
  the compatibility-capsule route, and records in the receipt which were
  stamped and which the capsule already carried. A capsule minted at mesh
  time, which is what mesh-plan --point culls, carries no clock, and the
  forecast driver refused the first 937.5 m point cull with "init carries
  no config_start_time" after the contract deck had spent seven minutes on
  the card. Attributes a native capsule carries are left byte for byte.
- rw_mpas_lbc resolves through the engine ladder. The boundary producer was
  read from $RW_MPAS_LBC and PATH alone, so a box where woof fetch-bridges
  had staged it beside rw_mpas_init refused the lbc leg of the very chain
  whose init leg had just resolved from that directory. woof.hex.engines
  carries an LBC row ($WOOF_HEX_RW_MPAS_LBC, $RW_MPAS_LBC kept as the
  legacy spelling, $WOOF_RW_MPAS_LBC, the bridge directories, PATH), the
  cycle chain and woof hex lbc resolve through it, and doctor reports it.
- Requires woof 2.7.3 and refuses every earlier engine, 2.6.4 and 2.6.5
  included. Eleven of the sixteen pinned seam files moved between the
  published 2.6.5 wheel 0.2.3 pinned and the published 2.7.3 wheel: the
  Grell-Freitas adapter and its kernel, the phase-one physics driver, the
  microphysics driver, legacy RRTMG, the Noah-MP compile site and glacier
  unit with the kernel loader and the Noah-MP libm unit, the config loader
  and the restart identity table; the column batch, the contract document,
  the Noah-MP runtime and glacier sources did not move, and the seam's
  constructor takes the same keywords, so no adapter change was needed.
  2.7.0, 2.7.1 and 2.7.2 carry the same sixteen bytes as each other and
  fail the re-pinned manifest by eight. Measured the other way first, every
  2.7 engine failed the 0.2.3 manifest, so the previous ceiling kept
  installs on 2.6.5 for the ten days the 2.7 line was out. The composed
  glacier unit's digest and the adapter contract digest moved with the
  engine's compile-site change and are re-derived by the instrument, not
  typed.
- Measured physics movement at this pin, seam against seam on one card
  (eight columns, forty levels, twenty 120 s steps): Grell-Freitas, whose
  kernel now uses the engine's own correctly rounded gamma and includes the
  kbcon layer in the cloud-work integral, moves the convective rain bucket
  by up to 5.1e-4 mm in forty minutes (0.21 per cent of the largest
  bucket), the phase-one theta rate by up to 2.9e-5 K/s and theta by up to
  9.4e-3 K after forty minutes on columns where it fires, and nothing where
  it does not. The YSU boundary-layer kernel, a file the manifest does not
  pin, keeps a column whose first-guess boundary-layer top sits below the
  second model level in the local-K regime for the whole step, and moves
  the lowest three levels of five of eight capped columns by up to
  1.9e-4 m/s² in the u rate and 0.139 K in theta after forty minutes;
  swapping that one file alone reproduces 2.7.3 byte for byte. The x4 F001
  byte chain that spans 2.5.8 to 2.6.5 is not re-run at this pin and is not
  expected to survive it. Details and receipts in
  docs/declared-divergences.md and verification/engine-verdicts/.
- tools/repin_source_tables.py runs on a tree that holds evidence/ out: the
  gated-files table is reported as not carried instead of raising, and the
  stale runner pin it then found on the x1.163842 stabilized-products tool
  is re-derived.

## 0.2.3 (2026-09-02)

New:
- Aerosol-aware Thompson microphysics (mp_physics = 28) routes through the
  physics seam and the production forecast door. The eleven-species row (the
  six WSM6 masses, the ni/nr/nc number moments and the nwfa/nifa aerosol
  number tracers) is carried in the engine's own array order, nc prognostic,
  and the door builds the seam with microphysics_scheme="thompson_aero" and
  passes every species by name. The mp=28 refusal 0.2.2 declared, because the
  pinned engine carried rows for wsm6 and p3 only, retired when the engine
  published its thompson_aero row: the two-rows gate failed, the row took an
  engine_scheme, and no other change in the door was needed.
- Registry rows for a generated uniform 60 km global mesh and for the 937.5 m
  and 800 m limited-area culls of one 125 km cap, each with its minted class
  and contract deck, so a sub-kilometre limited-area run reaches the forecast
  door the same way every other registered row does. The 3.75 km cull class
  carries the same contract.

Fixed:
- Requires woof 2.6.4 or 2.6.5. Eight of the sixteen pinned seam files
  moved between the published 2.6.1 wheel 0.2.2 pinned and the published
  2.6.4: three at the 2.6.3 cut (the column batch's mp=28 row, a
  config-relative file key in the config loader, the contract document) and
  six at the 2.6.4 cut (the phase-one physics driver's cumulus adapter
  contract, the Grell-Freitas adapter's release method and its kernel
  source, the kernel loader, the config loader's cumulus table, the restart
  identity table), config.py moving at both; each re-measured against the
  published wheel. 2.6.5 carries all sixteen byte for byte and is admitted
  by the same measurement. 2.6.3 fails the re-pinned manifest by those six,
  and 2.6.2, whose publish job died, fails on bytes as well as on PyPI. The
  x4 frozen-source proof re-run at the 2.6.4 pin hashes all four snapshots
  byte-identical to the 2.6.1 proof (one mesh, one case, one hour).
- The render door passes its --init file to the converter, so pressure-level
  products render; without it every pressure-level product refused as
  missing fields on every history the door had rendered.
- The device-memory floor counts the bytes this process already holds in its
  CuPy pool, so a run the door admitted is no longer refused on the
  remainder at its first step.
- A driver failure keeps its chained cause and writes the full traceback
  beside the run instead of a bare abort summary.
- The repository page states the engine range the package enforces and no
  longer links to a file the public repository does not carry.

## 0.2.2 (2026-09-01)

New:
- P3 microphysics (mp_physics = 50) routes through the physics seam: the
  rime pair (qir/qirim, qib/birim) is carried, `mp_p3` is hexcore
  vocabulary, and the scalar ladder requires WRF's eight P3 species. The
  seam A/B proves the hex plumbing byte-identical to the engine's own call.
  The production forecast door still speaks WSM6 only; generalizing the
  v841 chain to eight species is its own campaign.

Fixed:
- Requires woof 2.6.1. Three of the sixteen pinned seam files moved at
  that cut, each re-measured against the published wheel; the x4
  frozen-source proof re-run at the new pins hashes all four snapshots
  byte-identical to the 2.6.0 proof (one mesh, one case, one hour).
- A staged engine bridge older than the pinned engine release refuses by
  name instead of skipping its gates; the remedy is the fetch-bridges door.
- The sub-kilometre registry row `v0.9.120.110533` re-pins to the bytes the
  published 2.6.1 toolchain makes, and its delivery proof is re-taken from
  published artifacts only: mesh, static, GFS init, a 6 h forecast at
  dt 5 s on a 16 GB card, and 637 rendered frames, all through the shipped
  doors.
- A published comment the release scrub had mangled reads correctly again,
  and the scrub no longer doubles an article in front of a substituted
  label.

## 0.2.1 (2026-08-31)

New:
- The physics seam's scalar-requirement ladder fails closed over its own
  domain: a woof microphysics selector with no requirement row is refused
  by name instead of being accepted while declaring water vapour alone.
  mp=50 (P3) carries its own refusal reason: its rime pair (qir/qib) is
  state no six-species scheme can source, and this seam does not yet
  transport it.

Fixed:
- The transition-band gate refuses a gradient nobody measured. The engine's
  steepest-gradient meter used to sample a sphere-uniform lattice (~101 km
  point spacing) and stepped over any refinement transition narrower than
  that, so a spec truly at 65.2 %/cell read 8.3 and was admitted at 7.3x
  the ceiling; the repaired engine probes where the regions are and stamps
  its probe coverage, this gate refuses a receipt whose coverage word is
  missing or not complete, and the widening helper's non-monotonicity note
  died with the sampling defect that caused it.
- The swath probe stops reporting a coverage-refused spec as having cleared
  the gradient gate; its verdict now fails closed over the refusal classes
  and its coverage fragments are pinned against the Rust that prints them.
- Requires woof 2.6.0: the engine release carrying the repaired
  steepest-gradient meter, P3 microphysics (mp_physics = 50) and the ten
  P3 reachability fixes.

## 0.2.0

New:
- **A limited-area forecast runs the full physics stack.** `woof hex
  forecast --lbc-dir` integrates a culled regional mesh behind boundary
  files built from its own coarse parent, with WSM6, Grell-Freitas, YSU,
  YSU-GWDO, revised-MO, NoahMP, cloud fraction and RRTMG all attached.
  Measured on a 10 GiB RTX 3080: six hours, 1,080/1,080 steps, 13 history
  frames on 11,020 cells, peak 6,224 MiB, median 0.271 s/step, 343 rendered
  products. Against a global full-physics run over the same ground at t+6 h:
  theta 1.117 K RMS (r = 0.999973), precipitation r = 0.95, reflectivity
  r = 0.82, vertical velocity r = 0.621. Before this, the regional path was a
  dry dycore carrying one passive moisture variable and published no
  renderable weather field at all.
- **`woof hex cull`** cuts a limited-area grid, static and initial condition
  out of a global case in about a second, where a native regional init took
  775 s on the 121,182-cell parent `v4.75.121182`, measured 2026-08-26 with
  `woof hex init` and NOT against native. Culling that init is the
  supported route into the
  regional lane.
- **`woof hex swath`** decides where the fine grid goes, from a coarse
  forecast's own fields: detection on sea-level-reduced pressure, a
  declarative threat grammar, ranking that is commensurable across phenomena
  carrying different units, and hysteresis so a placement does not chase
  noise. Four independently placed grids over four different kinds of weather
  each completed 1,080/1,080 full-physics steps on one card.
- **`woof hex cycle`** follows weather across cycles: plan, cull, force,
  forecast, render. Two cycles of one real case ran end to end in 1,058 s at
  peak 7,744 MiB, cycle 2 re-detecting six hours on and continuing all four
  slots: three reusing their mesh, one regenerating. Starting a corridor
  from transplanted parent state rather than from the beginning transplants
  in 0.83 s and saves **273.8 s, 43 %**, against a real baseline arm.
  `docs/cycle-door.md`.
- **The regional anchor is keyed to a configuration class, not to a cull's
  own boundary-mask digest.** A re-placed swath was a new digest and therefore
  a new anchor, at 5.5 to 8.7 minutes of card against 4 to 6 minutes for the
  forecast it admitted: roughly three forecasts of permission per forecast
  run, which is what made cycling unaffordable. Five concentric culls of one
  parent were minted independently and returned the same verdict at the same
  Courant margin, so not one input the mint reads distinguished them. The
  class is earned once (two 1,080-step runs, 13/13 masked digests identical);
  the contract deck stays per-geometry because it runs on the cull's own zone
  geometry. Residual per-geometry cost 147.3 / 136.2 s of deck against the
  288-382 s of mint it retired, and anchors now admit by presented content
  rather than by a row in source.

Changed:
- **BREAKING: the import namespace is `woof.hex`.** It was `mpas_port` through
  0.1.1. `import mpas_port` no longer resolves and there is deliberately no
  alias shim. Every module path underneath is unchanged, so the migration is
  one token: `from mpas_port.X import Y` becomes `from hexcore.X import Y`.
  The distribution name (`woof hex`), the console script (`woof hex ...`)
  and every command surface are untouched. The old name overclaimed: this
  project implements MPAS-A v8.4.x's **dycore numerics and mesh conventions**,
  pinned as a specification and checked with stated one-step dry tolerances, and deliberately does not match that model's physics,
  which is WRF physics run through MPAS's own plumbing. Naming the package
  after another project put that project's name in every user's import line
  for a relationship holding over half the model. `woof.hex` names what is
  actually pinned and matches the distribution.
- **The shipped limited-area cut is wider, and the width is measured.** Five
  concentric culls of one parent against a no-boundary control: every field
  improves monotonically with cut width, because slicing at the fine core's
  edge discards the parent's own resolution ramp, which IS the
  intermediate-resolution ladder and is already inside the mesh. The knee is
  **1.35x** (`w` r 0.624 to 0.744, 2 m temperature r 0.578 to 0.852) for
  +27 % cells, +25 s wall, +42 MiB and zero extra forecasts. All nine shipped
  placement rows carry `cull_pad_scale` 1.35.
- **Device-memory admission takes the card's shape.** See the closed items
  below: this replaces the affine row and the flat headroom outright.

Fixed:
- **A fresh `pip install woof hex` resolved an engine this port's own pin
  refuses.** The dependency was `woof>=2.5.5` with no ceiling, so pip took
  the newest published engine (at the time 2.5.7) and the forecast lane
  then refused at launch with two SHA-256 digests and no version number while
  `woof hex doctor` reported the estate healthy and exited 0. A green
  install and a dead run, with no route out by reading. Three changes, all
  default-on: the declared range is now **`woof>=2.5.8,<2.5.9`**, derived by
  `woof.hex.engine_pin` from a measured table of every published engine rather
  than typed, with an **exclusive ceiling at the first engine nobody has
  measured** so a future engine cut cannot re-open it; `doctor` hashes the
  pinned files that live in `site-packages` and reports the offending version
  and the fix; and the forecast door's refusal names the version it found, the
  version it wants, which files moved, and the two commands that close the
  gap.
- **The engine floor is 2.5.8, and it is the only usable engine.** The seam
  manifest was re-pinned on 2026-08-28 and every row of the verdict table
  moved with it, because `moved` is measured against *this port's* sixteen
  files: 2.5.6, the floor of the day before, now reads 4 of 16 moved. 2.5.8 is
  the only published woof whose bytes match, so this port has no fallback
  engine: stated because it is a real exposure, not because it is
  comfortable. The table is spliced from the instrument's JSON, never typed.
- **`doctor` printed two adjacent lines that contradicted each other, and the
  forecast lane refused for a reason that had stopped being true.** On a
  byte-perfect install of the pinned engine the report read `16 of 16 pinned
  files are in this install and all 16 match`, and the next line said an
  installed woof cannot satisfy the pin and that lane needs a source
  checkout. The second was a constant string written when the manifest pinned
  `docs/mpas-seam.md`, which no wheel carried; woof 2.5.8 ships it inside the
  wheel at the manifest's own key. Measured 2026-08-28 in a virtualenv holding
  only the published wheels: `inspect_seam` over the install returns
  `checked=16, matched=16, moved=(), absent=()`, `doctor` exits 0, and the
  forecast door ACCEPTS `--gpuwm-checkout <site-packages>` at its own byte
  check. What still refuses is the driver's `verify_arwen_checkout_git`, and
  for a different reason: it records the checkout's HEAD, tree and dirty paths
  into every receipt so the executed source can be named by commit, and
  `site-packages` is not a git working tree. That refusal was a bare
  `CalledProcessError` at exit 128 and is now named. The guard is KEPT and its
  reason is corrected everywhere it is stated: the driver, `engine_pin`,
  both doors, `doctor`, `tools/battery/gpu_gates.txt`, `pyproject.toml`,
  README and five manual chapters. Retiring it needs a receipt identity that
  does not spell a commit and is a named follow-up in
  `docs/release-checklist-0.2.md`.
- **The precipitation verdict was stated backwards on three user-facing
  pages.** README, the concepts chapter and the troubleshooting chapter all
  carried "net domain-mean precipitation runs about 15 % dry" with no referee
  attached. That figure is a global domain mean against native MPAS-A, the
  referee retired on 2026-08-20. The live referee (skill against
  observations) ran on 2026-08-25 against NCEP/EMC Stage-IV and returned the
  opposite sign in all four cases (+0.0247 mm/h paired, 95 %
  [+0.0041, +0.0606]; frequency bias at 1 mm/h 1.59 / 1.35 / 1.38 / 0.77, so
  it rains over too much area). Each page now names which referee each number
  belongs to and states the obs sample's limits rather than trading one
  overstatement for another: four cases, two of them the divergence cases
  themselves, one truncated at 23 h with the largest bias, one +41.5 % on
  almost no rain, and the two clean complete cases at +9.8 % and +2.4 %. The
  troubleshooting chapter routed a user whose run looked too WET to a page
  saying the model runs dry; its advice is rewritten around what the live
  referee measured. The three declared-divergence magnitudes also carry a
  tense now: they entered the tree already finished on 2026-08-20 with no
  receipt, no card and no run commit, under engine pin `629ddb6f0`, and
  whether any of them survived the three engine pin moves since is NOT
  MEASURED.
- **A shipped capability was documented as impossible.** The concepts chapter
  said divergence 3's reflectivity half "cannot be scored at all against this
  build, because the history stream carries no reflectivity field to
  compare". `refl10cm` has been computed in the due step's own WSM6 call and
  published in every history frame, default on, since `e3caf6c`. The chapter
  now carries what the run returned: on the one case re-run with the field,
  86 model 35 dBZ objects against 54 observed, 8 matched at a median 110.9 km
  displacement, point CSI 0.0916 at 20 dBZ and 0.0097 at 40 dBZ over 643,419
  pairs, with the resolution cost named.
- **The limited-area lane is reachable from published artefacts.** Through
  2.5.7 it was not: no published `rw_mpas_mesh` carried `--cull-parent` and
  `rw_mpas_lbc` was in no bundle and no published source, so the flagship
  feature of this release could be run only from a source-built engine.
  Measured 2026-08-28 on the pinned engine: `woof fetch-bridges` stages 26 of
  26 artifacts against its packaged pins, `rw_mpas_lbc` included, and
  `woof hex cull` drove the staged published `rw_mpas_mesh` through two real
  cuts of the 40,962-cell global parent (338 and 606 cells; grid, static and
  init each written; 0.9 s). Not yet measured from published artefacts: a
  boundary set written by the published `rw_mpas_lbc` and a `--lbc-dir`
  forecast behind it.
- **The distribution's own test battery could not pass on the tree that gets
  published.** Thirty tests read measurement receipts under `evidence/`,
  which every published surface holds out on purpose; they raised
  `FileNotFoundError` on paths that never existed on the reader's machine,
  and `ci.yml` runs on push. They now skip with a stated reason when the
  tree does not carry its receipts, and still FAIL when it does carry them
  and the receipt a row names is missing: a held-out record and a row citing
  a measurement nobody can check are different findings and get different
  answers. Two packaging guards required `docs/LANE-BRIEFING.md` to EXIST in
  order to check that it does not ship, which a published tree cannot
  satisfy; they now assert the outcome instead. `setuptools` is declared as
  the test dependency it always was (`ensurepip` stopped seeding it at Python
  3.12, so three packaging gates failed on 3.13 and not on 3.11). And the
  forecast-door tests substitute the card-shape seam, which
  `WOOF_HEX_NO_LOCAL_GPU` (set by this project's own CI) made refuse.
  Assembled public tree: **32 failed before, 0 after**. Unpacked sdist: **40
  failed before, 0 after**.
- **Three campaign scripts published home-directory paths.**
  `tools/device_memory_ledger/run_arm.sh`, `run_kern.sh` and
  `ledger_table.py` carried 24 absolute paths under one machine's home
  directory: a private working-tree layout and a venv path, in scripts that
  could not run anywhere else anyway. They read `HEX_REPO`, `HEX_WORK`,
  `HEX_ASSETS`, `HEX_PYTHON` and `ARWEN_CHECKOUT` now, and refuse by name
  when one is unset rather than defaulting somewhere wrong.
- **The device-memory gate had the wrong shape, and it failed in both
  directions at once.** The affine row charged card-sized workspace knees
  (Grell-Freitas, YSU and the RRTMG shortwave chunk, the three sites that do
  NOT scale 4x on a 4x mesh) as per-cell growth. Across seventeen recorded
  peaks its error spans **-41.42 % to +27.19 %**, and **six runs exceeded what
  the old gate demanded**: four on the limited-area path the cascade actually
  runs, short by 1,056 / 1,104 / 860 / 716 MiB, plus 96 and 28 MiB on two
  graded globals. The gate now asks
  `device_admission.model_for_card(card, configuration)` for
  `core(card, configuration)` plus a Grell-Freitas workspace at
  `min(cells, SMs x 4 x 64)` plus a YSU workspace at
  `min(cells, SMs x 16 x 32)` plus a per-cell term, and it covers all
  seventeen. The margin is named (that card's shortwave workspace plus
  11.2 MiB of instrument convention) instead of a flat 512 MiB. The retired
  arm stays computable at `device_admission.RETIRED_AFFINE_ROW_20260826` so
  the comparison can be re-made rather than re-argued.
- **The door reads the live card.** Two registered meshes that the shaped
  gate's predecessor had moved from admitted to refused on a 10 GiB card are
  admitted again, confirmed on the real hardware rather than derived. The
  workaround that reported itself as a workaround is retired with the defect.

Known issue:
- Across four placed variable-resolution meshes at one card, one engine pin
  and one schedule, the peak spans **+3.89 % to -6.76 %**, and one mesh with
  15,343 MORE cells peaked 318 MiB LOWER. The mechanism is named (allocator
  placement of the shortwave block, which stops being servable from the free
  list at the step where radiation and history capture coincide) but it is
  NOT separated by an A/B on those meshes. The limited-area core is an
  envelope over five samples, not a per-allocation fit.
- Pool retention is 20-30 % of the footprint with no arena owning it.
- A cycle is one parent integration read at successive times, not a parent
  regenerated per cycle. Regenerating it is the operational remedy for a
  corridor that has moved far, and it is not built. Two of four admitted
  slots per cycle are skipped as background culls by a measured minimum-edge
  ratio.
- A corridor started from transplanted parent state begins with no cloud ice,
  snow or graupel, because the initial-condition stream carries no slot for
  them. Hour-zero reflectivity does not correlate; one hour on, the
  microphysics has re-formed the ice and r = 0.863. Temperature agrees to
  five decimals throughout.
- No obs-skill score exists for a cycled case. Every limited-area verdict
  above is against a global run over the same ground, not against
  observations.

---

Added:
- Registry row `v16.66.195629`: `v16.66.195630` regenerated from its own spec
  row, unchanged, by a generator that no longer makes four-sided cells (woof
  2026-08-26).
  The cause of the old row's blow-up was the graded generator's insertion
  operator placing its new generator on the near-cocircular quad's own
  circumcentre, where its Delaunay ring is exactly the four quad cells, 18 of
  18 and 13 of 13 insertions measured, and readable in the shipped bytes
  themselves: cell 195615's four neighbours lie on a circle of radius
  20.783 km to within 0.10 km and the cell sits 0.564 km from its centre, 2.7 %
  of the radius. The surgery's local polish then PINNED the cell it had just
  damaged, and neither the repair loop nor the emit gate ever read a
  coordination number, because a quadrilateral plus the two heptagons the same
  operation makes leaves `sum(6 - nEdgesOnCell)` at exactly 12. All three
  graded spec rows now regenerate clean: `v16.66.195629` (195,629 cells,
  `{5: 1037, 6: 193568, 7: 1023, 8: 1}`), `v15.60.224210` (unchanged cell
  count, `{5: 1073, 6: 222076, 7: 1061}`, digests moved), and `v20.80.151649`,
  whose regenerated geometry is BIT-IDENTICAL to the registered bytes (every
  cell centre, ring and edge length exactly equal) so its completed 6 h
  forecast still describes what the generator emits.

- The polygon-complement defect is closed at the producer, and it was never reporting-only.
  `rw-mpas` `density.rs::polygon_contains` accepted on `|winding| > pi`, which
  cannot tell a point inside a ring from a point whose ANTIPODE is inside it,
  so every polygon region refined a congruent ghost of itself on the far side
  of the globe. That ghost's edge is a step, not a ramp (the signed distance
  jumps from −19,900 km to +19,900 km across it, both ends of a saturated
  `tanh`) so the spacing field fell from the region's spacing to the
  background across one cell and the generator's gradient gate refused every
  swath spec the placement layer emits, at `background/spacing - 1` per cell
  every time (1775 %/cell at 4 km in 75 km). Fixing the containment test alone
  cleared both symptoms: all four emitted specs now clear the gate at every
  spacing tried, `tools/probe_polygon_attainment.py` returns "no defect
  reproduced at this engine build", and the polygon arm tracks its cap control
  to four figures. One correction to the row: the ghost was in the FIELD, not
  only the report, at a 600 km half-width it cost 299,497 cells against the
  equivalent cap's 175,721.

Changed:
- `cell_coordination_admission`'s remedy is re-anchored, not retired. It used
  to hand "regenerate" to the reader as a coin flip and name a generator-side
  follow-up; that follow-up has landed, so the refusal now names the fixed
  generator, the mechanism it fixed, and `v16.66.195629` as this row's
  replacement. The gate itself stays and must: the mesh it refuses already
  exists, is still registered, and is still on disk.
- The `v16.66.195630` row records that it is superseded. It stays registered
  and stays refused, because it is the bytes that measured the cost.
- The device-memory row of record is re-fitted at the merged tip (2026-08-26): `5,016.5 MiB + 98,748 B/cell` on the 170 SM card,
  replacing the 2026-08-25 row measured at hex `7fe514b`. One per-allocation ledger session,
  both published meshes, same card, same engine pin, same protocol
  byte-for-byte. The tip did NOT
  shift the footprint uniformly: `x1.40962` rose 484.0 MiB (8,390.0 ->
  8,874.0) while `x4.163842` FELL 96.0 MiB (20,542.0 -> 20,446.0), so the
  fixed term rose 677.4 MiB, the slope fell 4,948 B/cell, and the two rows
  cross at about 143,554 cells. Quoted at this tip the old row left
  `x1.40962`'s requirement 27.9 MiB above its measured peak: 5 % of the
  shared 512 MiB headroom, on the smallest published mesh. The superseded row
  retires computably (`device_admission.RETIRED_ROW_20260825`,
  `retired_converged_row_floor_bytes`), a test refuses any governing surface
  that quotes it as the footprint (RED at the change's base, 18 offences) and any
  shipped caller of either retired arm, and every `CARD_TIER_ROWS` entry now
  carries `measured_at_pin` / `restated_at_tip`: a per-card row with no pin
  on it is how the 170 SM row outlived its tree. Cells admitted: the 32 GiB
  part 270,915 -> 277,297; the 16 GiB part 111,524 -> 109,919; the 10 GiB
  part 42,934 -> 37,892.
- The RTX 3080 tier row re-borrows its slope from the merged tip
  (2,483.0 MiB + 98,748 B/cell). The fixed term is a property of the card and
  the slope a property of the build, so a BORROWED slope must be the current
  build's; the fixed term is re-derived from that card's own measured
  6,340.5 MiB peak and reproduces it exactly. Arithmetic on one measurement,
  not a new one: the row still declares NOT RE-MEASURED AT THE MERGED TIP.

Known issue *(both entries below are CLOSED at 0.2.0 and are kept because they
are the before-arm of the fix. The first was measured again at its own
protocol and REVERSED SIGN: `v20.80.151649` measures 19,255.25 MiB against a
19,297.8 MiB prediction, so the row OVER-predicts by 0.22 %, the two peak
conventions agree to 0.75 MiB, and the attribution to mesh shape below was
wrong. The second is fixed by the door reading the live card. Do not quote
either as open.)*:
- The re-fitted row still under-predicts the one GRADED mesh measured against
  it, and now does so past the gate. `v20.80.151649` (151,649 cells) peaked
  19,838.0 MiB against a 19,297.8 MiB prediction: +540.2 MiB (+2.80 %), and
  **28.2 MiB above its own `required_free_bytes`**, a card offered exactly
  that requirement would be admitted and then overrun. The re-fit did not
  cause it and did not remove it: both fitted points are quasi-uniform global
  meshes and this row is EXACT on both, which is what makes the excess
  attributable to the mesh shape rather than to a stale term (the 08-25 row
  missed the same point by +2.60 % and missed both uniform meshes as well).
  At most 11.2 MiB is the whole-device sampling convention, measured side by
  side in the same session. Pinned by
  `test_the_graded_point_exceeds_the_shipped_rows_requirement_and_says_so`.
  The remedy is one per-allocation ledger arm on a graded mesh at this row's own protocol, not
  a wider gate; that arm has not run.
  Until it does, the 512 MiB headroom is not spendable and graded meshes want
  margin above what the row returns.
- Two registered meshes (`x1.40962` and `v15.150.38857`) move from admitted
  to refused on the DEFAULT row on the 10 GiB desktop card, which the 08-25
  row admitted. The refusal is conservative rather than correct: the default
  row carries the 170 SM card's fixed term, and that card was measured
  running `x1.40962` at a 6,340.5 MiB peak. Its own row admits the mesh with
  2,244 MiB to spare (`--device-fixed-mib 2483.0 --device-bytes-per-cell
  98748`, chapter 6). A door that selects a measured tier row from the
  detected card would make that the default; until then the flag is a
  workaround and is reported as one.

Fixed:
- A global mesh carrying an all-zero `bdyMask` triple is a global mesh. MPAS
  writes that triple all-zero on a sphere and the unified `rw_mpas_static`
  follows the convention, so every static this project generates ships it;
  classifying a mesh as a regional cull on the PRESENCE of the triple made a
  closed sphere a bounded disk and refused it for being a sphere: Euler
  characteristic 2, boundary rings 1..7 empty. Measured on `v20.80.151649`,
  RTX 5090: bound clean, refused at load. Every generated-static row was
  affected; the published statics are native-made and carry no triple, which
  is why no test caught it. The test is now a boundary ZONE, a nonzero mask
  value, and an incomplete triple is still refused on presence.
- `--preflight` answers the timestep question beside the memory one instead
  of exiting on it. A row declaring an unanchored timestep ended the preflight
  before the admission verdict printed, so "will this mesh fit my card?" went
  unanswered for exactly the meshes people ask it about.

Changed:
- The device admission floor is re-proved against hardware (2026-08-26) and the citations that outlived the constant it replaced are
  retired. The floor itself is unchanged (the measured affine row plus one
  512 MiB headroom, from the single `woof.hex.device_admission` surface,
  since 2026-08-25) but the graded-mesh work measured its capacity
  boundaries on a base that predated that change and merged the conclusions
  beside it, so three registry rows described a retired linear proxy as
  governing. `device_admission.retired_linear_floor_bytes` is now the one
  place that computes the retired arm, tests refuse any governing surface
  that quotes it as a requirement and any shipped caller of it, and
  `FLOOR_DERIVATION` records the decision, the finding and the measured
  consequence. Measured on an RTX 5090: the 224,210-cell graded mesh is
  admitted on device memory (26,511.7 MiB predicted against 31,642.6 MiB
  free) where the proxy demanded more than the card holds, the 32 GiB part
  carries 270,915 cells against the proxy's 210,952, and a 1 h full-physics
  `v20.80.151649` forecast ran rc 0 at the shared sum. That run is also the
  row's first out-of-sample point and it came in 2.60 % OVER prediction,
  finishing 9.7 MiB inside its own requirement on the headroom; re-fitting
  the row at the merged tip is a named follow-up.

New:
- **The frozen lane runs at five timesteps, not one.** From 2026-08-26 the v8.4.1 column-physics lane is no longer pinned to 120 s,
  and anchors were earned the same day at **100 s, 75 s, 20 s and 5 s**, each
  two forecasts on named hardware, finite at every step, every history frame
  identical between arms, minted on the already-registered `x1.40962` because
  an anchor is a property of the timestep and a Courant limit is an upper
  bound. A timestep with no anchor is still refused by name before anything is
  allocated. Only 120 s carries a native reference and only it ever can, so
  every new row records `native_reference=None` rather than being conflated
  with it. Three registered graded meshes go from refused at bind to runnable:
  `v16.66.195630` at **16.5 km** core spacing, and both 224k-cell rows at 75 s.
  Each anchor's health band is measured against a 120 s control on the same
  card, mesh and init, and reported as a trend as well as a min/max: 100 s and
  75 s track the control within parts in 1e4, while **20 s does not**, the
  vertical-velocity mean climbs monotonically to 5.53 m/s against 1.48 and
  keeps climbing, with `theta_m` max 5.1 K below the control. That run is
  finite at every step and byte-identical across arms, so it is a different
  solution rather than an unstable one; whether the cause is Grell-Freitas
  being called 180 times an hour instead of 30 or resolved dynamics is
  recorded as NOT MEASURED, with what would settle it. Convection-off is not
  covered: the frozen configuration pins `config_convection_scheme`, so the
  fine anchors certify the GF-on configuration and no other.

Fixed:
- **A global mesh was refused as a corrupt regional cull, closing the whole
  published family.** Every forecast on `x1.40962` (at any timestep,
  including the proven 120 s) died with `regional mesh is not a bounded disk:
  nCells-nEdges+nVertices = 2, not 1` and three empty-`bdyMask`-ring findings.
  Both were the proof the mesh is global read as proof of a broken cull: 2 is
  a closed sphere's Euler characteristic, and empty rings are what an all-zero
  mask means. `Mesh.validate` classified on the *presence* of the
  `bdyMaskCell/Edge/Vertex` triple, and native MPAS-A writes that triple into
  a global mesh's static file too, all zero: the published `x1.40962.static.nc`
  carries all three with zero nonzero entries. A cull has a boundary zone, so
  the rule is now the triple **plus a nonempty zone**; an all-zero triple on a
  mesh that is not a closed sphere is refused by name.

- The regional anchor is re-minted at the merged tip, and the NVRTC
  reciprocal defect is measured on a live sm_120 forecast. The NVRTC reciprocal-rewrite fix moved
  the third-order stencil denominator off a source literal in both
  `cuda_driver` and `cuda_transport`, superseding the pre-fix forecast pair;
  the anchor's source-binding check caught it and also showed the check was
  too narrow, so every translation unit the regional step launches through is
  named now. An A/B on the card (merged tip against the same tree with those
  two units reverted, the reverted arm reproducing the superseded digest
  exactly) measures the defect moving 97.8 % of interior `u` values and
  42.5 % of `theta` at three forecast hours, with the specified zone unmoved
  at every field and every lead. The same attribution probe re-run at the
  merged tip gives an unchanged 36,750 of 163,405 `kinetic_energy` values, so
  the reciprocal defect explains none of that divergence and its cause is
  still open; the count is pinned so a later change cannot adopt the fix as its
  explanation.
- The model timestep is admitted from an earned-anchor registry
  (`woof.hex.dt_admission`), on the same pattern as per-architecture and
  regional admission. The frozen v8.4.1 column-physics configuration refused
  any `config_dt` but 120.0 with a literal, and pinned `config_bldt_seconds`
  and `config_cudt_seconds` beside it; the premise of that refusal was
  "unproven at this timestep", not "wrong at this timestep". **The admitted
  set is unchanged** (120 s holds the only anchor and every other value is
  still refused before anything is allocated) but the refusal now names the
  evidence the anchor rests on and the procedure that mints another, and a
  second anchor is one table row rather than an edit to two files. An anchor
  carries a schedule receipt (host-derivable: physics cadence step counts, the
  Grell-Freitas `cudt == dt` law WRF pins for `cu_physics = 3`, the RK
  schedule's shape against the proven one, the WSM6 minor-loop split, and
  clock closure in binary64), an integration anchor (two byte-identical
  forecasts on named hardware), and a nullable native reference: nullable
  because the one native MPAS-A v8.4.1 integration this program holds was run
  at 120 s and no other timestep can ever have one. `tools/mint_dt_anchor.py`
  mints and verifies; its verifier certifies the registered row and fails all
  six fabricated variants of it, and it refuses to mint at all unless it
  reproduces the archived 120 s stage tables exactly. Registering a second
  anchor moves the frozen lane off its proven timestep and is a decision of
  record, not a tool run.
- `woof hex forecast --preflight` answers the timestep question from the
  registry row alone, with no card and no file, the same way it answers the
  architecture question. A row whose declared timestep holds no anchor is
  refused at argument resolution instead of after the mesh bytes are read.
- The regional (limited-area) forecast runs on the card, and its anchor is
  earned. `woof.hex.cuda_regional_forecast_v841` carries the device
  residency, the memory model and the stage sequencing that let the port's
  whole-step CUDA driver run `config_apply_lbcs=true` on a native-culled
  mesh; `mpas-port`'s driver gained a `regional_v841` hook mirroring its
  halo-exchanger hook, guarded at every one of its ten call sites so a
  whole-mesh run is bitwise untouched. Four independent processes ran three
  forecast hours on the 2,971-cell CONUS cull in 21.5 seconds of card time
  and produced masked-digest-identical history at all seven published frames
  while every whole-file digest differed. `ADMITTED_REGIONS` now holds one
  earned row, `conus-x1.2971`, naming L5's contract receipt and this
  forecast pair; every other regional configuration, including the larger x4
  cull of the same region, still refuses at the door by name.
- The device runs native MPAS's own garbage-element memory model. Every
  array carries one padded element per dimension with absent neighbours
  remapped to it, and the native pool value is restored into every garbage
  column after each launch: pad-compute-strip, held resident, because a
  device launch cannot skip its last element when the thread bound and the
  array stride are the same integer. The discipline is armed by an optional
  `KernelCache.post_launch` observer, so it reaches every entrypoint the step
  resolves without one shared CUDA translation unit changing by a byte. All
  twelve recorded division-by-garbage-geometry sites and the
  `divergence_damping_f32` sentinel early-out are retired without touching a
  kernel; two further blockers found by running it (a singular tridiagonal
  denominator at a zero reference state, and a recovered-state validator that
  demands positive density over its whole launch extent) are answered by a
  non-trapping reference pad and by running the identical test over the
  elements native solves.
- The v8.4.1 regional (limited-area) surface is a CUDA translation unit.
  `woof.hex.cuda_regional_v841` carries 22 kernels: the lateral-boundary
  pool with its four derived coupled fields and its device-side time
  interpolation, the specified-zone tendency assignment, the relaxation-zone
  Rayleigh and Laplacian stages with their hardwired 50/10-dt coefficients,
  the u/ru specified-zone overwrite and the w hard-zero, the end-of-step
  `reset_speczone_values`, the scalar boundary adjust/set/clamp stages, the
  acoustic specified-zone pressure-gradient masking and implicit-solve skip,
  and the scalar-transport mask-4/5 edge downgrade with its specified-zone
  cell skip. Each kernel mirrors exactly one function of the v8.4.1 CPU
  authority, which is its expected-bits oracle, and the four native quirks
  are replicated at their sites citing that work's anchors rather than
  re-derived. The kernels live in their own translation unit and under their
  own names so the global lane's sources stay byte-identical and every
  archived compile manifest, FTZ audit count and receipt that pins them
  stays valid. Proved on the 16 GiB proving card (RTX 5070 Ti, sm_120) by
  `tools/run_cuda_regional_contract.py` against the native-culled reference
  mesh: 8 of 8 contract decks bitwise identical over 10,443,332 float32
  values compared as raw bit patterns, 22 of 22 kernels covered with no
  kernel lacking a deck, dual-run stable both within a process and across
  two independent processes (84 of 84 payload digests identical), and every
  deck re-run with a deliberately wrong zone geometry FAILS, so each proof
  is shown to work in both directions.
- Regional CUDA execution is refused by name until a registered regional
  anchor exists (`woof.hex.cuda_backend.regional_admission`), decided
  2026-08-25, mirroring the per-architecture earned-anchor pattern. An
  anchor is a row naming a contract receipt and a byte-identical forecast
  pair that exist in this repository; adding a region is table work. The
  registry is empty, so every regional configuration refuses, and the
  refusal names the breakage it prevents: a regional forecast that carries a
  receipt nobody could verify. The two CUDA host validations that already
  refused a culled mesh now refuse through that gate instead of declaring
  the lane closed/global, a premise the kernels above retired.

Fixed:
- The dycore's outer step and the frozen physics seam's step come from one
  source. They were two: `bind_mesh` rebound `DT_SECONDS` in the proof and
  forecast modules and in the GWDO guards, and the sealed WOOF constructor
  read that rebound value, but the dycore takes its outer step from
  `config.config_dt` and the configuration was built from its dataclass
  default. MEASURED (2026-08-26, the 32 GiB proving card (RTX 5090)): a mesh row declaring 100 s
  bound clean, allocated 18,820 MiB, spent 285 s and died inside composite
  step 0 with `post-RK candidate time must equal the exact step endpoint:
  120.0 != 100.0`. The forecast host now builds its configuration at the bound
  row's timestep and derives all four seam clocks from that configuration, so
  the two cannot diverge by construction; a coherence gate refuses a divergent
  pair on the host, before device memory is taken, quoting what it prevents.
  The forecast door's step count was a third clock reading the registry row
  while the run stepped at `config_dt`; all three now agree.
- A registered graded row said "Declared dt 90 s" while declaring 75 s. The
  value moved when the radiation-cadence rule landed and the sentence did not.
  The note now states the derivation: 95.84 s Courant limit, and 75 s is the
  largest value at or below it that also divides the 600 s radiation cadence
  exactly and closes the model clock in binary64.
- Regional CUDA kernels no longer divide by a source literal. MEASURED on
  the 16 GiB proving card: NVRTC rewrites `x / <float literal>` as `x * (1/<literal>)`, so
  `mpas_div(x, 5.0f)` returns a value one ulp from the correctly-rounded
  float32 quotient the CPU authority computes, while a runtime divisor and
  `__fdiv_rn` are exact. It cost four kernels their bitwise identity at
  once and the contract deck is what caught it. The hardwired `nRelaxZone`
  denominator is now a runtime argument, and a test greps the translation
  unit so the defect cannot return. The same hazard is measured and
  recorded at eight further sites in shared, frozen-source-pinned
  translation units (`mpas_div(..., 12.0f)`: two in `cuda_transport`, six in
  `cuda_driver`), where a third of float32 arguments take a different value
  than the CPU authority's division, a named cause for the released
  `transport_vertical_flux` differing from the CPU authority at 51,258 of
  166,376 values on the reference cull. Those eight sites are fixed in the
  next entry.
- The eight shared literal divisors are gone, and the rewrite is an
  architecture boundary rather than a property of one stack. MEASURED on the
  desktop RTX 3080 (sm_86, NVRTC 13.0.48 `CL-36260728`, CUDA driver 13030):
  one compiler, one option set, one source, NVRTC emits `div.rn.f32`
  against the literal for every target up to `compute_90` and `mul.rn.f32`
  by the literal's float32 reciprocal from `compute_100` up, which covers
  every card this port runs production work on. A differential compile of
  all fifteen CUDA translation units puts the census at ten rewritten
  instructions over eight source sites: `transport_vertical_flux` in
  `cuda_transport` (inherited by `cuda_transport_v841`), and
  `vertical_u_flux_f32`, `theta_vertical_flux_f32` and `w_vertical_flux_f32`
  in `cuda_driver`. The flux3/flux4 denominator is now the translation-unit
  constant `mpas_third_order_denominator`, which the host can write and the
  compiler therefore may not fold; `mpas_div` still carries the division, so
  its FTZ subnormal guard stays on the path. On targets below the boundary
  every payload digest is byte-unchanged, measured on sm_86 against the CPU
  authority on the reference regional bytes.
- The `transport_vertical_flux` divergence is explained in full, not
  partly. Reproduced independently on the RTX 3080 by running the shipped
  kernel and an instruction-level emulation of the higher target's
  arithmetic on the same reference bytes: the shipped kernel matches the CPU
  authority's `_atmosphere_vertical_flux` at all 166,376 values, and the
  rewrite alone moves exactly 51,258 of them, the whole of the recorded
  divergence, with nothing left over. The other three kernels move 157,710
  of 474,032, 51,639 of 154,492 and 51,318 of 154,492 interior values.
- Two files that supply CUDA bytes to pinned translation units are pinned
  themselves. `cuda_transport_v841` compiles `cuda_transport._CUDA_SOURCE`
  and every unit prepends `cuda_fp32.CUDA_FTZ_HELPERS`, so while those two
  were unpinned an edit to either changed a pinned unit's compiled bytes
  with every pinned digest still matching. This change's own remedy landed in
  `cuda_transport.py` with the frozen-source proof reporting green, which is
  how the hole was found.

New:
- The v8.4.1 regional (limited-area) runtime runs in the CPU authority
  lane. `woof.hex.regional_v841` transcribes the complete surface of
  `mpas_atm_boundaries.F` and the `atm_srk3` regional insertions: the
  7-ring masks and `specZoneMask` derivation, `nearestRelaxationCell`, the
  limited-area admission checks, the two-level LBC value/tendency pool with
  the four derived coupled fields (`lbc_rho_zz`, `lbc_ru`, `lbc_rho_edge`,
  `lbc_rtheta_m`) that `woof.hex.lbc` deliberately left to the driver,
  `meshScalingRegional`, the spec/relax-zone tendency stages with their
  hardwired 50/10-dt Rayleigh and Laplacian coefficients, the acoustic
  specified-zone pressure-gradient masking and implicit-solve skip, the
  scalar-transport edge downgrade at masks 4-5, the u/ru specified-zone
  overwrite after recover, the w hard-zero, `bdy_adjust_scalars`,
  `bdy_set_scalars`, `reset_speczone_values`, the moist coefficients
  (`qtot`/`cqw`/`cqu`) and the unconditional end-of-step negative-scalar
  clamp of DO_PHYSICS builds. `config_apply_lbcs=True` is admitted only
  behind real, admitted LBC state (every other absence still refuses by
  name) and the two circular `transport.py` sentinel refusals now name a
  remedy that exists. Four native quirks are REPLICATED, not fixed, each
  documented where it is implemented and each a checked fact in
  `tests/test_regional_runtime.py`: the monotonic copy-back that admits
  only `bdyMaskCell <= nSpecZone` and so excludes relaxation rings 3-5; the
  Fortran operator precedence in the mask-4/5 edge condition, where
  `.and.` binds tighter than `.or.` and the mask-4 half therefore fires
  regardless of `config_apply_lbcs`; ring 1 never being nudged; and the
  `tend_rho` pool, which `atm_compute_dyn_tend_work` writes only at
  `rk_step == 1`, so the regional adjustments persist into RK stages 2 and
  3 and are applied again on top of themselves.

Fixed:
- `rvord` is the REAL(RKIND) quotient of the float32 constants, not the
  rounded float64 quotient. One ulp; it alone broke frame-0 `theta`
  bitwise identity against the compiled reference.

- One device-memory admission surface (`woof.hex.device_admission`), and
  the `NATIVE_DEVICE_FLOOR` re-derived from measurement (2026-08-25).
  Every free-memory gate (the forecast door, `--preflight`, the mesh
  binding's per-mesh floor, the driver's `MIN_FREE_DEVICE_BYTES` and the
  restart-worker floor) now computes the same sum, the converged-pin
  measured row (4,339.1 MiB + 103,696 B/cell, the 170 SM per-allocation ledger fit) plus one
  shared 512 MiB headroom, and the door forwards its resolved requirement
  into the driver argv (`--required-free-bytes`) so a card admitted on its
  own measured row cannot be refused downstream on the default model. The
  retired floor (an asserted 24 GiB scaled linearly per cell) refused
  meshes the measured row says fit and admitted x1.40962 below its measured
  peak; both retired breakages are test-pinned facts. Per-card rows at the
  converged stack measured the same day: RTX 5070 Ti two-point fit
  1,774.0 MiB + 115,143 B/cell (x1 6,272 / u96 8,802 MiB, u96 rc 0 under
  the new floor); RTX 3080 x1 6,340.5 MiB device-view, fixed 2,289.6 MiB
  with the slope borrowed and said so, x1.40962 is now admitted on the
  10 GiB desktop card, where the superseded row refused it. No 12 GiB card
  exists in the fleet: the 12 GiB tier figure ships as a DECLARED
  DERIVATION, labeled `DERIVED, NOT MEASURED`, and the label is
  test-pinned.
- Regional (limited-area) meshes are admitted. `Mesh.validate()` recognises
  the `bdyMaskCell/Edge/Vertex` triple a native cull adds and validates the
  measured regional contract: absent-neighbour sentinels tolerated in
  exactly the five arrays a cull zeroes (`cellsOnCell`, `cellsOnEdge`,
  `edgesOnEdge` inside the unshrunk row, `cellsOnVertex`, `edgesOnVertex`)
  and only on ring-7 elements; reciprocity exempt only where the absent
  element makes it undefined; Euler characteristic 1 for a disk; edge and
  vertex masks equal to the minimum of their present cells' masks; neighbour
  masks within 1; ring populations growing outward; incidence identities
  corrected by exactly the sentinel counts. Every refusal names the concrete
  breakage. Both native culls of the regional reference mint load through
  `Mesh.from_netcdf` and pass `mesh-check`. The registry gains regional
  ROW fields, `boundary_zone_width`, `bdy_mask_sha256`
  (`regional_mask_digest`), and a nullable `lbc_source`, cross-examined at
  bind for every row before any constant moves: a regional cull on a global
  row, a digest or width mismatch, and an empty boundary-source slot are
  each refused by name (no boundary stream exists yet to force the zone).
  `mesh-check` prints a `regional` receipt block (zone width, per-ring cell
  counts, the pinned mask digest) and accepts `--grid-only` because a
  culled grid exists before its static does.
- `woof.hex.lbc`: the lateral-boundary file reader and the two-level
  value/tendency pool, a transcription of `mpas_atm_boundaries.F` admission
  and timekeeping. `LbcInventory` applies the two stream rules by each
  file's own xtime (LATEST_BEFORE for the first admission,
  EARLIEST_STRICTLY_AFTER for every advance) and a missing interval refuses
  naming the rule, the model time and the timeline it searched. `LbcPool`
  holds the interval-end state and the float32 `(new - old) / dt` tendency,
  and `state_at` interpolates linearly backward from the interval end,
  `mpas_atm_get_bdy_state` verbatim. The reader pins the v8.4.1 lbc stream
  contract (seven float32 full-mesh fields on their measured dimensions) and
  refuses a missing, transposed or widened variable by name. Unit-tested on
  synthetic schema-correct files and on the three real native case-9 files
  of the 2026-08-25 regional oracle (`GPUWM_HEX_LBC_ORACLE_DIR`). Derived
  coupled fields (`lbc_rho_zz`, `lbc_ru`, `lbc_rho_edge`, `lbc_rtheta_m`)
  and driver wiring are deliberately out: they need mesh state and belong to
  the runtime lane; the pool refuses their names and says whose they are.
- Graded (variable-resolution) meshes are registrable and bindable. Four
- **A generated variable-resolution mesh completes a full-physics
  forecast** -- the first in this project. Registry row `v20.80.151649`
  carries 20 km resolution in its core inside an 80 km background at
  151,649 cells, and ran 6 h at dt 120 s on one RTX 5090: 180/180 steps,
  rc 0, finite at every step, 621.6 s wall, 19,226 MiB peak (0.57 % under
  the capacity model's prediction). Run twice, all seven history frames
  byte-identical. The uniform mesh at that resolution would be 1,472,535
  cells and fit no card this project owns; the graded mesh is 9.71x
  smaller and left 8.69 GiB of admission margin.
- Graded (variable-resolution) meshes are registrable and bindable. Five
  rows join the registry from the engine's new hierarchical-Goldberg
  generator, pinned by byte count and SHA-256 like every other row, each
  with its own dt admitted from the file's real `dcEdge` under the
  versioned Courant policy. A second row was produced by moving ONE
  coordinate in the spec JSON with zero code changes and binds through the
  real forecast door -- the arbitrary-acceptance test, passed on the door
  rather than on a diagram. Measured quality across the generated set:
  min `dvEdge/dcEdge` 0.0406-0.1023, 2.0x to 5.1x the admission floor, with
  zero edges under 0.04; the Fibonacci-seeded mesh this registry refuses
  reads 1.685e-4.
- `tools/probe_dv_floor_boundary.py` measures the real post-fix dvEdge
  load boundary through the actual loader on the actual published pair
  (edited copies at 50/200/1,000/5,000 m dual edges), and
  `tools/audit_donor_padding.py` reports which padding convention a grid or
  static carries -- the donor-padding defect's donor-readability surface, measured on four
  artifacts and handed over rather than silently changed.

Fixed:
- A mesh row whose dt the frozen v8.4.1 lane cannot step is refused AT BIND
  instead of inside composite step 0. Measured: a row at dt 100 s --
  Courant-admitted, dual-edge admitted, cadence-dividing -- bound clean,
  allocated 18,820 MiB, spent 285 s and died on `post-RK candidate time
  must equal the exact step endpoint: 120.0 != 100.0`, because the dycore
  takes its outer step from `config_dt`, which `V841MpasColumnPhysicsConfig`
  pins to exactly 120.0. The refusal names the frozen constant, the cost of
  not refusing, and the remedy. A companion guard refuses a dt that does not
  divide the 600 s radiation cadence, checked where the row is written.
- The WOOF seam pin leaves the pin-only lineage: `ARWEN_BUILD_COMMIT` moves
  to `26daaab7e`, the engine's seam-converge merge, where the refl10cm seam
  (`6e333822e`) folds into the release line (`613b681d3`). The sixteen-file
  manifest re-freezes on release-line bytes (seven digests move: physics,
  gf, noahmp_runtime, kernels/__init__, config, io/restart, docs/mpas-seam),
  the contract surface and adapter digests move with it, and the proof
  re-pins `cuda_arwen_physics_v841.py`. A checkout of the engine release
  line now verifies 16/16 with no recorded dirt, so the next public engine
  snapshot satisfies the port's pins as cut. No port behavior changes;
  the pinned engine bytes gain the release line's own seam-file evolution
  (per-PBL GF forcing wiring and the support files of that era).
- The default history stream publishes `refl10cm` and `q2`, the two fields
  whose absence left four registered obs-referee metrics unscorable on the
  first real run. `refl10cm` is computed inside the due step's own WSM6 call
  from post-call temperature and the unchanged prepared pressure (WRF's
  `diagflag` arrangement, the point where native MPAS-A computes the field),
  carried through the transactional seam, and consumed exactly once per
  frame; `q2` is published bitwise with `q2_products_allowed = "true"` and
  its occasional native-parity negatives preserved. `rw_mpas_convert` maps
  both (`REFL_10CM`, `Q2`), so `rw_wrfbatch` reflectivity products read the
  model's own field instead of the renderer's hydrometeor fallback, and the
  model bundle producer derives `reflectivity_dbz` (column maximum, the
  MRMS-comparable composite) and `dewpoint_k` (the engine's
  dewpoint-from-mixing-ratio on Q2/PSFC, transcribed exactly). The WOOF
  seam pin moves to `6e333822e` (the refl-capable seam on
  `pin/mpas-port-arwen-seam-v2`); snapshot schema v3.

Fixed:
- Stale-guard sweep, hex side (2026-08-25, stale-guard audit findings 5-10 +
  unknowns). The 2-GPU partition scheduler's floors route through
  `woof.hex.device_admission` (the retired 22 GiB linear shape and the
  20 GiB `require_devices` default are gone; the per-partition
  application of the measured row is labeled `DERIVED, NOT MEASURED`
  pending a 2-GPU per-allocation ledger run). `reservation_probe`'s self-validation
  control is re-pinned from the dead pre-frame-cut `gf_gfdrv_stage` 7,034 MiB
  premise to the post-cut widest frame (`wsm6_column`, 7,216 B) with a
  device-derived bound, and `module_image_probe` gains the registry
  route so the widest-frame module cannot escape the ledger.
  `copy_elision_accounting`'s "of record" arm is the converged row,
  test-pinned to `FLOOR_DERIVATION`. The FTZ guard-cost timing ceiling
  is per-architecture beside the arch-admission registry (sm_120 keeps
  1.25; sm_86 gets 1.75 from its recorded 1.47-1.57x deviation;
  unregistered architectures refuse by name). Bigcard refusal/marker
  strings compute from `X4_FULL_PHYSICS_BYTES` instead of restating the
  retired 26.4 GiB; README and manual chapter 1 stop asserting the
  superseded row and the un-re-derived floor. The dry-runner 16 GiB
  floor and the x1.163842 nominalMinDc dt rule are adjudicated frozen
  closed-case records with the determination written at the constants;
  STATE.md's declared engine floor reads the enforced 2.5.5.
- `forecast_door.FOOTPRINT_MODEL` no longer quotes the superseded
  6,296.5 MiB + 93,474 B/cell row (pin `0d04db712`): it is the of-record
  converged row, and `tests/test_device_admission.py` re-fits the raw
  evidence ledgers and pins the shipped coefficients to them, so a
  constant drifting from the evidence it cites fails by name (#340).
- The registered v15.150.38857 static is rebuilt on the unified 82-variable
  `rw_mpas_static` writer and re-registered (the unified static writer). The retired writer's
  drag band sampled terrain 180 degrees of longitude from every cell (the
  archive-origin assumption): measured corr(old, new) for var2d is +0.003
  at the same cell and +0.697 at lon+180, and the field-by-field compare
  shows oa/ol moving full scale on two thirds of cells. The rebuilt static
  matches a native init_atmosphere static for the same mesh at var2d
  corr +0.9999, oa1 +0.9961, land-only con +0.9928, and adds the operator
  tables and soil-composition group the retired writer omitted. The x4 and
  x1.40962 rows were measured to be native-built statics (v8.4.1 and the
  published v8.2.0 artifact) that never carried the band; each registry row
  now names its builder.
- A lake column (MODIS category 21) is folded to open water at the forecast
  loader boundary, the same conversion WRF applies without a lake model.
  The WOOF vegetation tables end at category 20, so before the fold any
  generated-mesh run with lakes died with an IndexError inside the Noah-MP
  cold start; the native x4 landuse never exceeds 19, which is why the
  proof path never saw it. The fold count is in every run receipt; on the
  x4 case the mask is empty and every array passes through untouched.
- The GWDO dt guard follows the mesh binding: a registered mesh runs the
  YSU-GWDO kernel at its own Courant-admitted timestep instead of dying
  at step 0 on "requires dt_seconds=120". The kernel takes dt as a runtime
  argument; on the frozen native mesh nothing is rebound and the guard
  still demands exactly 120 s.
- The x4 proof's restart leg is bit-identical again. GF's advective
  forcing pair (rthdynten/rqvdynten) is per-step carried state: each
  step's dynamics forms it and the next step's physics consumes it, and
  it lives outside both the MPAS atmosphere and the WOOF backend
  restart payload. The F030 checkpoint never captured it, so every
  restored run re-entered step 16 with zero forcing lanes while the
  unbroken run fed the real step-15 pair, and the step-16 identity gate
  failed deterministically on every arm (the step-16 restart divergence, 5/5 red on the reference node, red
  since the forcing lanes landed). Checkpoint schema v3 downloads the
  pair at F030, refuses to write a checkpoint without it, re-seeds it on
  restore in both the fresh-process worker and the in-process
  instrument, and gates the rehydration with its own fingerprint
  identity. A pre-v3 checkpoint is refused by name instead of resuming
  wrong.

## 0.1.1 (2026-08-25)

The forecast becomes a front door, and the referee runs.

New:
- woof hex forecast: a front door that binds the mesh, asks the card
  first against a measured per-card row, refuses by name with numbers,
  and prints the render command when it passes. --preflight gives the
  same answer without spending anything.
- The obs referee ships with its first scorecard: canonical model bundles
  gain a producer, and four metrics that could not score now score.
- The default history stream publishes refl10cm and q2, and
  rw_mpas_convert maps both, so reflectivity products read the model's
  own field instead of the renderer's hydrometeor fallback.
- A generated mesh completes a forecast end to end. A mesh whose Voronoi
  edges collapse is refused at bind, by name, before anything expensive.
- The engine seam pin moves to the woof release line and verifies 16/16
  clean, so an engine checkout at the pinned commit satisfies the pins
  as cut.
- A per-allocation device-memory ledger. Measured 2026-08-24 on an RTX
  5070 Ti: x1.40962 peaks at 5,604.0 MiB with the engine's device-sized
  radiation chunks.

Fixed:
- The registered v15 static is rebuilt on the unified 82-variable writer;
  the retired writer's drag band sampled terrain 180 degrees of longitude
  away. Every registry row now names its builder.
- A lake landuse column folds to open water at the forecast loader
  boundary; a generated mesh with lakes no longer dies in the Noah-MP
  cold start.
- The GWDO dt guard follows the mesh binding; a registered mesh runs at
  its own admitted timestep.
- Restart checkpoints carry GF's advective forcing pair (schema v3);
  restored runs are bit-identical again, and a pre-v3 checkpoint is
  refused by name.
- The forecast door leaves output creation to the driver; an admitted run
  no longer fails on a directory that already exists.

Requires woof 2.5.5 or newer for the seam bytes and the bundled engine
binaries.
