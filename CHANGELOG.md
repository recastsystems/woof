# Changelog

## 1.1.4 (2026-10-31)

WOOF 1.1.4 takes engine 2.9.11 and the hex model 0.4.11. Forecast answers change.

New:

- Hex runs NOAA's RRFS v2 `hrrrv5` suite with no native library, from
  RRFS's own start state, on zoom globes and on any region (circle, ellipse
  or drawn outline), on one card or several.
- Wind farms (`--wind-farms uswtdb|PATH`, opt-in): turbine drag and
  turbulence inside the boundary layer, with turbine and farm power in
  history, on regional runs and hex globes.
- Nudging toward the source analyses (`--nudging spectral|grid`, opt-in).
- Simulated radar at C- and X-band as well as S-band.
- Optional Icechunk output (`--output-format netcdf,icechunk`).

Changed:

- History frames are written lossless-compressed: the same values in about
  a third of the disk. `--history-compression none` writes classic frames.
- Starts from HRRR, GFS, GDAS, GEFS, RRFS, IFS, AIFS and ERA5 download only
  the records they read (an HRRR start moves about 40 % of the bytes); the
  prepared start is unchanged.
- Hex multi-card runs use one process per card and scale on any Python.

Fixed:

- Hail: HAILCAST now uses the hail stone model NOAA's operational HRRR runs,
  by default. WRF 4.6.1's model melted most stones before they reached the
  ground and left thin, broken hail swaths; it is still selectable with
  `hailcast_version = "wrf_461"`. HAILCAST is a diagnostic, so the forecast
  itself does not change.
- Runs with Kain-Fritsch cumulus and Thompson microphysics no longer show
  the cumulus scheme's drizzle as 20-30 dBZ echoes or strips along a coarse
  grid's edges.
- A source that carries cloud and rain (HRRR and others) no longer streams
  convective cells in through the grid's edges as straight rain bands.
- Nest-downs at 600 m or finer take 100 m land cover and soil, and those at
  250 m or finer run as LES.
- A nested run on a source that posts lead by lead starts on the first
  leads instead of waiting for the cycle's last one.
- HRRR-recipe runs with a companion export no longer exit 1 after a
  finished forecast.
- A case prepares the same lake temperatures from a cold or a warm cache.
- Starts from ECMWF open data and AIFS have sea ice.
- Sea-level pressure maps are smoothed over high terrain.
- Soundings show MOSHE as `--` without an inflow layer and colour the MCS
  Index the right way round.
- Hex starts from GFS and HRRR begin with the source's cloud, rain, ice,
  snow and graupel, so the first radar map shows the starting storms.
- Hex storm-following regions on MPAS physics move about four times faster.

## 1.1.3 (2026-10-28)

WOOF 1.1.3 takes engine 2.9.9 and the hex model 0.4.9. Forecast answers change.

New:

- A hurricane map theme, `woof-light-tc`: 10 m wind bands with edges at 34,
  50, 64, 83, 96, 113 and 137 kt, and isobars every 4 hPa.

Fixed:

- Forecasts that start a RUC or Noah land surface from another model's soil
  (ERA5, GFS, GEFS, IFS and others) map its soil moisture onto the land
  model's own scale, so the surface no longer starts too wet and too cold.
- Every physics choice the catalog offers runs.
- Runs of 48 h or longer with sea ice on the grid start again.
- A moving step-down draws hourly window products such as 1 h rain, over the
  ground the grid reached during the hour.
- A restart continues a checkpoint taken while GFS boundaries were still
  streaming, and runs that keep one checkpoint set no longer stop on a
  missing checkpoint.
- A forecast no longer waits up to five minutes on a stuck kernel warm-up,
  and no longer spins a CPU core while it waits for the card.
- The column store, GFS preparation, analysis bundles and restart checksums
  run on every core.
- Hex on MPAS physics no longer stops with non-finite columns after weeks of
  model time, and a checkpoint resumes under a changed timestep.
- Hex 30 km globes no longer run out of GPU memory, and large ranks fit a
  32 GB card.
- Hex local time stepping picks the fastest plan for the mesh: a 96 -> 3 km
  zoom globe runs about 1.2x faster than the fixed step on MPAS physics.
- Hex zoom globes rotate a stored mesh instead of generating one, and
  MPAS-physics start-up and history output run on every core.
- Hex storm-following runs no longer stop when a reused member is unsafe
  for the storm.
- A stalled hex source download is retried after 10 s instead of 300 s.

## 1.1.2 (2026-10-27)

WOOF 1.1.2 takes engine 2.9.8 and the hex model 0.4.8. Forecast answers change.

New:

- Hex runs MPAS physics by default: a run that names no physics runs
  MPAS-A's `convection_permitting` suite on the GPU. The regional model's
  physics stays one flag away; `woof hex physics list` shows every suite.

Fixed:

- `woof hex` with MPAS physics (the default), `woof hex physics list` and
  `woof hex physics resolve` no longer stop with "No module named 'hexcore'".
  The RRFS physics route, the ocean mixed layer, the UGWP orographic drag
  tables and the hex preparation cache work again too.
- Hex local time stepping on the full step, column-suite restore and
  rollback, product redraws after a moving run's controller restarts, and
  settings handed to hex child processes work again.
- `install.sh` and `install.ps1` run `woof` (they called the old command
  name), and package version lookups and documentation links name
  recast-woof.
- `pip install recast-woof` installs threadpoolctl and psutil, and the
  `hydro-conus` extra exists.
- `python -m tools.import_sweep` and `tools/battery/run_stage1.py` run on
  the public tree, and the test suite is green on it.
- Everything engine 2.9.8 and hex 0.4.6, 0.4.7 and 0.4.8 fix, as their notes
  list it. Among them: storm-following runs no longer refuse a move, a cycle
  that is not posted yet is fetched from the next source, and GOES-17 ABI
  windows fetch.

## 1.1.1 (2026-10-25)

WOOF 1.1.1 takes engine 2.9.7 and the hex model 0.4.5. Forecast answers change.

New:

- Soundings at any point and hour after a run: every run and ensemble member
  writes a column store that `woof soundings` draws from.
- Run controls: per-nest history intervals, a quick-look stream between full
  frames, an LES closure below 250 m, the storm nest size and an inner nest.
- Moving nest-downs: `woof downscale --follow storm|parent|fixed|target` and
  `--warm-from`. Children finer than 4 km default to the HRRR-shape suite.
- IFS HRES open data at 0.1 degree, and RRFS ensemble members that start with
  their same-cycle deterministic companion.
- Branched ensembles (`woof ensemble --branch-from`) share each card among
  several members and draw their maps on every core.
- A cyclone rain swath product, and 5 km CONUS forecasts that stream on a
  32 GB card.
- Hex: the MPAS physics route runs every hex shape (limited-area and
  storm-following regions, ensembles, local time stepping, Hex with Regional,
  restart, IAU, stochastic physics, two or more cards) and runs faster.
- Hex with Regional in three shapes: a nest placed at launch, a nest added
  mid-run, and a nest that spawns when a storm attribute crosses a threshold.
- Hex doors start from any posted forecast hour (`--lead-hours`) of GFS, GEFS,
  IFS, IFS ENS, AIFS and AIFS ENS; regions take drawn shapes and can be kept
  as inits; maps stream from the first hour.

Changed:

- The spectral global model (`woof global`) is retired. Global forecasts run
  on `woof hex global`, which exports to the regional model on both physics
  routes.

Fixed:

- Everything engine 2.9.6 and 2.9.7 and hex 0.4.4 and 0.4.5 fix, as their
  notes list it. Among them: `woof go` no longer waits on a RAP or HRRR native
  cycle that has already posted, preparation has no single-core stages, and
  hex exports on the MPAS physics route carry terrain height and
  reflectivity.

## 1.1.0 (2026-10-23)

WOOF 1.1.0 takes engine 2.9.5 and the hex model 0.4.3. Forecast answers change.

New:

- `woof hex` carries the GPU port of MPAS-Atmosphere v8.4.2: both physics
  suites and their scheme options, initialization and static fields, IAU,
  limited-area boundaries, streams, restart and diagnostics. WOOF's own
  physics stays the default suite.
- `woof hex global --row R --cycle C --work-dir W` runs a global forecast end
  to end: mesh, static, start, forecast, products, export and score. It is the
  main global model; the spectral `woof global` stays a full option.
- From hex 0.4.2 and 0.4.3: one-command limited-area forecasts
  (`woof hex regional`, `woof hex country --region`), variable-resolution
  global meshes, storm-following regions and ensembles (`woof hex moving`,
  `woof hex moving-ensemble`), subseasonal ensembles with weekly products and
  scores, a mixed-layer ocean and sea ice on the mesh, aerosol transport in
  global forecasts, keyless ERA5 starts, and an 80-level high-top row.
- WoFS-shape ensembles (`woof ensemble --recipe wofs`) from any start source,
  with ensemble soundings, and hurricane ensembles with cyclone products.
- From engine 2.9.4: the packaged HRRR recipe follows operational HRRR v4
  physics; `woof race` and `woof standby`; storm-following 1 km nests
  (`woof go --storm-nests N`); HRRR-Smoke with `--smoke`; coupled hydro wherever a
  hydrofabric covers the domain, opt-in with `--hydro on`; coupled ocean waves and storm surge (`woof ocean run`); a GPU
  mixed-layer ocean and prognostic sea ice; live aerosol optics; solar
  eclipse shading; hail size and HAILCAST; storm tracks fetched automatically
  (`woof cyclone-setup --storm auto|ID|NAME`) and `woof go --cyclone-products`;
  new default products (IVT, IWV, freezing level, atmospheric rivers,
  visibility and fog, snowfall, heat index, WBGT, HDWI, Haines, lightning
  threat); hub-height wind and solar power products; fire danger indices;
  archived GFS, dated GEFS and ECMWF ENS sources; polar stereographic domains;
  and `woof setup --with-geog`, a permissively licensed default geography set.

Changed:

- Coupled hydro is opt-in: a run has no hydro unless it asks with
  `--hydro on`.
- The bridge bundle carries the engine's tools and the global model's
  observation decoders. Maps read `WOOF 1.1.0`.

Fixed:

- Everything engine 2.9.4 and 2.9.5 and hex 0.4.2 and 0.4.3 fix, as their
  notes list it. From 2.9.5: coastal sea surface temperatures in the
  mixed-layer ocean no longer drift, hydro start-up runs its spin-up on the
  GPU, and `install.sh` installs the required Rust when the system Rust is
  too old.

## 1.0.3 (2026-10-06)

WOOF runs the operational HRRR configuration: its static file, its
Thompson, MYNN, RUC, surface-layer and radiation forms, its fixed 20 s
clock and fifth-order vertical advection, from `woof import-namelist` on
an HRRR namelist or from the shipped HRRR recipes. Ensembles, simulated
radar volumes and verification against observations join the forecast
doors. Isobaric heights, RUC soil water and legacy RRTMG longwave are
corrected. Forecast answers change.

New:

- HRRR configuration recipes ship in the installed package's `configs/recipes/`
  folder: `hrrr_configuration_cut.toml` (a 401 x 301 3 km cut, started from
  native RAP columns), `hrrr_configuration_clock.toml`,
  `conus_hrrr_configuration.toml` and `hrrr_v4_gsd41.toml`. Run one by its
  installed path:
  `RECIPES=$(python -c "import configs, pathlib; print(pathlib.Path(list(configs.__path__)[0], 'recipes'))")`
  then `woof fetch-geog` and `woof fetch-geog --static-source hrrr-conus-v4`
  (once; the operational static file is 1.1 GB) and
  `woof go "$RECIPES/hrrr_configuration_cut.toml"`. Each recipe ships the
  WPS namelist its staged route reads beside it. They select the
  operational branch forms below explicitly; generic configurations keep
  the WRF v4.6.1 defaults.
- `woof import-namelist` on a supported `hrrr_wrf.nl` selects the HRRR
  physics forms: GSD v4.1 MYNN, the operational Thompson, MYNN surface
  layer, RUC irrigation, snow, ground-vapour cold start and logarithmic
  2 m diagnostic, prescribed monthly leaf area and albedo, legacy RRTMG
  with the operational cloud form, aerosol optics 3, shortwave
  interpolation 1, sun-angle land albedo (`alb_sol = 1`), the sixth-order
  filter, microphysics zero-out and upper wind limiter of that branch, and
  fifth-order vertical advection. An adaptive clock whose start, minimum
  and maximum are equal (HRRR v4: 20/20/20) imports as a fixed step with
  `terrain_clock = "pinned"` and the acoustic substep count WRF derives
  for it (6 for HRRR's grid at 20 s, where its namelist spells 4). The
  import report lists each substitution. The namelist importer also
  accepts 43 inert `&time_control` output-stream settings it refused.
- `[static] source` selects a published static file. HRRR configurations
  take the pinned operational CONUS static file by default, on its native
  grid and contained cuts; soil, land use, terrain, monthly vegetation,
  lakes and drag fields keep the source values, and analyzed vegetation
  fraction comes from the initialization GRIB.
- `terrain_clock = "pinned"` runs `time_step` and `time_step_sound`
  exactly as written when `use_adaptive_time_step = false`. The terrain
  clock and steep-ground substep rule still report what they would have
  done, as advice in their receipts, and apply nothing. `"measured"`
  stays the default.
- New physics selectors, each off or at the WRF v4.6.1 form by default:
  `bl_mynn_version = "gsd_41"` (the operational RAP/HRRR MYNN boundary
  layer, with dew and frost water, mixing length 2, subgrid stratus and
  shallow cumulus, and an optional mass-flux plume population),
  `thompson_version = "wrf_39_noaa"` with its own lookup tables, fetched
  once from pinned public source with hash checks,
  `mynn_sfclay_variant = "gsl_wrf39"`, `fractional_seaice = 1`,
  `ruc_qvg_cold_start = "air"`, `ruc_2m_diagnostic = "log_profile"`,
  `ruc_irrigation = "wrf_45"`, `ruc_snow = "wrf_45"`, `alb_sol = 1`,
  `diff_6th_form`, `diff_6th_factor2`, `mp_zero_out`,
  `upper_wind_limiter_form`, and advection orders `v_sca_adv_order`,
  `v_mom_adv_order` and `h_mom_adv_order` (defaults 3, 3 and 5).
  `bl_mynn_version = "gsd_41"` lowered daytime 2 m dewpoint bias against
  the operational model on a 3 km cut from +0.93/+0.95 K to +0.69/+0.85 K
  at f01/f02. The GSD v4.1 MYNN port is partial.
- `rap-native` fetches and prepares RAP's 13 km hybrid-level columns.
  `hrrr-native` supplies a native hybrid analysis, with the same cycle's
  pressure product supplying the soil column. `woof prep --initial-inputs
  JSON` starts a domain from a separate analysis while every lateral frame
  comes from the boundary source. `use_rap_aero_icbc = true` takes
  aerosol number initial and boundary values from the analysis. HRRR
  plans stage the monthly aerosol climatology in the shared cache before
  forcing transfer, with exact size and SHA-256 checks; dry runs report
  it as pending.
- A domain that is its source's own Lambert grid (for `hrrr-prs` and
  `hrrr-native`, the full 1799 x 1059 HRRR grid) prepares untrimmed, each
  point copying its own source cell. It was refused before.
- MYNN supports WRF mixing length 2 and `scalar_pblmix = 1`. RUC mosaic
  land use and soil (`mosaic_lu`, `mosaic_soil`) and WRF's CLM lake model
  (`sf_lake_physics = 1`) run, off by default.
  `zadvect_implicit_variant = "wrf_legacy"` selects the older WRF
  implicit vertical-advection split.
- Ensembles: `woof ensemble CONFIG --members N` (also `woof go` and
  `woof run` with `--members N`, or `[ensemble] members = N`) runs N
  members, each from its own source trajectory: the source's operational
  ensemble (`gefs` for `gfs`, `aigefs` for `aigfs`, `ecmwf-ens` for
  `ecmwf-open-data`), `--recipe time-lagged`, or multi-model members from
  `--trajectories FILE`. A source with no ensemble and no recipe is
  refused before any download, because copies of one forecast have no
  spread. The run draws mean, spread, minimum, maximum, threshold
  probabilities, paintball and postage-stamp maps as members finish.
  Members of matching physics and forcing advance as one batch per card,
  with every product byte-identical to members run one at a time. Packed
  members can carry a parent and its nest. An interrupted streamed nested
  forecast or ensemble continues from its last checkpoint. Random
  perturbations (SPPT, SKEBS, SPP and related switches) are refused at
  every door, because their spread has not been calibrated against
  observations. `woof fetch --source ecmwf-ens` fetches ECMWF open-data
  ENS members p01 to p50 and `--source rrfs-ens` the five public RRFS
  members, which preparation refuses because their files carry no soil.
- `[simulated_radar]` writes native radar volumes (Level II and CfRadial 1
  by default; CfRadial 2 and ODIM optional) and PPI loops while a forecast
  runs, off by default. `woof simulated-radar` replays saved histories;
  `--describe` and `--estimate` report support, memory and output size.
  PPIs draw ZDR, CC and KDP when the volume carries them. Radar colours
  follow the AWIPS reflectivity table and a green-red velocity table;
  `[simulated_radar] color_tables = "classic"`, `woof render
  --radar-colors classic` or `RUSTWX_RADAR_COLORS=classic` restores the
  earlier colours byte for byte.
- A finished forecast is verified against observations by default:
  station dots and scorecards, and MRMS reflectivity and hourly rain
  panels, scored and drawn in Rust. The run waits at most 2 seconds for
  it; downloads and scoring continue in the background, and `woof
  verify-visuals RUN_DIRECTORY` runs it on demand. A verification failure
  never changes a forecast's result; `WOOF_VERIFY_VISUALS=0` turns it off.
  A regional rain scorer adds Rust hourly rain, echo and fractions skill
  score kernels with a native MRMS decoder.
- Comparison sheets draw MRMS and RRFS as reference panels, in a stated
  order. Maps and sections name the model WOOF for files WOOF wrote.
- Multi-card `[devices]` forecasts step their slabs at once under a
  free-threaded Python 3.14 (`python3.14t`); `woof` re-executes once with
  `PYTHON_GIL=0` on such a build. Full HRRR physics on the 1797 x 1057 x
  50 grid: 4 x RTX PRO 6000 130.0 s per forecast hour against 333.8
  before, 8 x RTX 5090 83.8 s. Histories are byte-identical across 2, 4
  and 8 cards and both interpreters.
- `woof fetch --source 20crv3-cf` reads the 20th Century Reanalysis v3
  ensemble mean (1836 to 2015), so a forecast can start from a historical
  date.
- `[physics_params]` (or `WOOF_PHYSICS_PARAMS`) applies a named set of
  physics parameter values, recorded in the run's identity. Default runs
  are unchanged.
- Generated adaptive nests declare a maximum step growth of 5 percent,
  and `woof check` names a nest whose growth is above 5 (WRF's own nest
  value, 51, ended a nested forecast with a non-finite vertical velocity).

Fixed:

- Isobaric heights no longer read high: each pressure-level height had
  paired a layer's mean height with its mid-level pressure (500 hPa read
  5.3 m high on a 3 km frame). The fix reaches height charts, pressure
  planes, soundings, post-processed files, ML export, the moving-nest
  tracker and GNSS-RO data assimilation. Answers change for every
  isobaric and sounding height product.
- RUC soil water no longer refills a dry top level at the WRF v4.6.1
  rate. `ruc_soilprop` defaults to `"wrf_45"`, the operational RAP and
  HRRR form; in one cycle 2 m dewpoint bias against stations went from
  +1.16/+1.59 K to +0.92/+1.22 K at f01/f02. `"wrf_461"` reproduces the
  earlier output. RUC also runs its water branch on lake cells when the
  lake model is off. Answers change for every RUC run.
- Legacy RRTMG longwave applies each stratospheric optical-depth
  correction once; parallel bands had raced on shared outputs. Legacy
  RRTMG shortwave follows WRF on columns with no layer above the
  troposphere switch, where it had read stale memory. Answers change.
- Fifth-order vertical scalar transport no longer creates mass: a 12 h
  3 km forecast had gained 181,000 t of cloud ice through its zero clamp,
  now 25 t. Order 3, the generic default, is unchanged.
- HRRR-configuration runs: surface shortwave excess against HRRR fell
  from about +31 to +4 W/m2 on a full-grid cycle, and the afternoon 2 m
  dewpoint bias came down from about +0.5 to +0.8 K to HRRR's own on two
  cycles, with 2 m temperature error unchanged. RUC keeps prescribed
  leaf area on those paths, and the Q2 cap applies over land and water
  under the logarithmic 2 m diagnostic.
- MYNN initializes turbulent moments from water vapour, as WRF does, and
  a physics template's own parameters win over its component options.
- An HRRR-native start no longer refuses when the source's top moist
  level sits a fraction of a millipascal above the model top. Native
  hybrid analysis can initialize at its own model-top interface.
- Nested forecasts: a nest keeps its fresh CFL after an output alarm on a
  full step, a nested health failure is retried at most twice from a
  verified checkpoint with a lowered step ceiling, and each retry is
  written to the receipt.
- Multi-card forecasts: the GPU memory watcher no longer stalls the
  forecast (a busy 8-card nested run went from 130.5 to 8.0 s per
  forecast hour), cross-NUMA halo copies are staged, slab threads run on
  their card's local CPUs, halo seams move in one launch each, history
  writes select the owning card, split domains pass orographic and
  topographic wind inputs to each tile, and admission prices MYNN, RUC,
  microphysics and boundary workspaces.
- Process start to first output on a 1797 x 1057 x 50 case fell from
  113.8 to 26.6 minutes with identical output, and such cases run within
  the usual 1,024 open-file limit. High-resolution static preparation
  holds about 160 MiB per reader instead of climbing to 322 GiB, with
  identical fields. GFS preparation budgets its decode against host
  memory.
- Preparation honours explicit worker counts, runs host interpolation and
  prepared-file writes in Rust workers, prepares grids larger than a
  card's workspace on that card, and starts split forecasts from the
  first boundary interval.
- More host work runs in Rust with unchanged bits: hex geometry and
  initialization checks (a 157,859-cell geometry job from 28.46 to
  0.223 s), observation scoring, Noah soil liquid water, health scans,
  ozone interpolation and prepared-store hashing. Legacy RRTMG frees
  326 MiB to 5,145 MiB of unused ozone grids.
- The shared NetCDF reader accepts arrays above 134,217,728 values and
  signed and unsigned 64-bit coordinates; compressed NetCDF-4 files with
  dense attributes read without a checksum mismatch. GRIB2 readers keep
  complex-packed all-missing fields as missing.
- Two WOOF versions on one machine no longer overwrite each other's Rust
  bridges: a release stages its bundle in its own folder under
  `~/.woof/bridges/`, and `woof doctor` names the folder in use.
- A completed forecast keeps its exit status and final digest when
  another process deletes history frames during the run. Simulated-radar
  requests survive preparation. Forecasts run one after another in one
  process release their microphysics and land tables.
- Windowed products (6 h precipitation and similar) on latitude-longitude
  grids, such as a WOOF Global map, no longer streak across the date line
  or shrink on a regional crop.
- High-resolution geography fetches serialize shared cache entries and
  verify hashes before reuse; damaged packaged tables in the user cache
  are repaired by verified replacement.
- A WPS namelist whose `ref_x`/`ref_y` name the grid centre imports to
  the same bytes on Linux and Windows.
- A packed data-assimilation step resumed with `--resume-ensemble`
  reproduces the uninterrupted run, and the radar observation operator no
  longer zeroes graupel in the model state.
- Scientific documentation separates code verification, solution
  verification and validation against observations; old physics profile
  IDs are accepted as aliases.

## 1.0.2 (2026-10-02)

WRF-fidelity fixes correct moisture pressure terms, edge vertical velocity,
diffusion and map factors, and the 2 m humidity cap. Forecast answers change.

New:

- `[devices]` in a single-domain or tree config, or `woof go --devices N`,
  splits a forecast across cards, with per-card memory checks and an
  execution receipt. `domains` selects split grids, defaulting to all;
  split nests take forcing each parent step and return two-way feedback
  to split parents. `woof sim --devices N` or `--devices-table` sets a
  prepared forecast's split without changing its configuration.
  `woof check CONFIG --devices N` prices every card, grid and pinned host
  store, exiting 2 for an unknown VRAM budget. Frames reach the host while
  the model steps on. Split parents over resident nests, moving nests and
  late-starting split grids are refused. Split grids refuse Noah mosaic
  and slope radiation (`slope_rad = 1`) before allocation, naming the
  missing state; resident grids retain both. Splitting is off by default.
- `woof run-plan` starts GFS and native HRRR single domains and trees from
  the first posted hours, as `woof go` does. Both start single `hrrr-prs`,
  `rap`, `rrfs`, `icon-eu` and `gem-gdps` domains the same way. A `[tiles]`
  root, including `woof cyclone-setup` trees, waits at each boundary seam,
  instead of for the whole cycle. Output is unchanged. An earlier forecast
  failure keeps its own error beside any later input timeout.
- `WOOF_FETCH_POLICY=aws` pins sources with an AWS host to that host without
  NOMADS fall-through; sources with no AWS host retain their usual host.
  `latest` selects the newest cycle whose first hours are on AWS;
  `--whole-cycle` retains the whole-cycle rule. Forecasts still stream from
  their first posted hours. Unset, fetching is unchanged. `woof fetch`
  probes hosts refusing HEAD with a one-byte GET.
- `woof import-namelist` accepts `domain_wizard`, `mod_levs`, `plotfmt` and
  bare `30s` geography, and fills 17 of 33 formerly required keys from WRF
  and WPS defaults. Unknown sections are refused by name.
- `diff_opt = 1` runs WRF's coordinate-surface horizontal diffusion with
  `km_opt` 2 or 4 and either `mix_full_fields` setting, including restarts
  and streamed forecasts. It is off by default (`diff_opt = 2`).
- `topo_wind` runs under YSU, excluding BEP and BEP+BEM;
  `gwd_opt = 1` or `3` runs KIM or GSL gravity-wave drag under every PBL
  scheme. GPU column checks match WRF 4.7.1 bit for bit. Static preparation
  adds orographic statistics from WPS geography, and namelist imports keep
  both keys. Orographic statistics are byte-identical on Linux and Windows.
  A streamed `[tiles]` domain stops at its first drag call.
  Both options are off by default.
- `woof ml-export` converts regional histories, converted hex frames,
  global tapes or their ZIPs into one Zarr store per domain. It offers
  WeatherBench 2's 13 pressure levels, ERA5's 37 or model levels, ERA5
  and WeatherBench 2 names and units, earth-relative winds, a
  `below_ground` mask over ERA5's below-ground fill, an optional regular
  latitude-longitude grid and a ZIP that opens in place. Levels above the
  model top are omitted and listed. `xarray.open_zarr` works with
  zarr-python 2.18 and 3. On Windows, ZIP history files with WRF's colon
  filenames unpack with underscores; see `docs/ml-export.md`.
- GPU preparation rotates winds, interpolates vertically and builds the
  Thompson cold start with the CPU route's arithmetic, matching its
  prepared arrays. Forecasts prepared on the card change. A full HRRR
  forcing preparation took 154 s instead of 640 s.
- Acoustic, advection, horizontal diffusion and per-step glue kernels move
  less memory, with byte-identical output. The horizontal diffusion change
  alone reduced an hour of an 800 x 600 x 50 forecast from 95.7 to 88.4 s
  on an RTX 5090.

Fixed:

- Legacy RRTMG longwave radiation keeps the correct spacing above the
  model top. A pressure guard had compressed valid layers, changing their
  temperatures and heating. Forecast answers change.
- `woof go --cycle` accepts an exact match to a prepared bundle,
  checkpoint or declared forcing's actual initial time, recording a no-op.
  Conflicting or unreadable input times remain refused. Forcing fetched by
  the run is checked after acquisition and before preparation.
- Every shipped profile and configuration leaving `moist_cq` unset applies
  WRF's moisture correction to pressure terms, including passive vapour
  with microphysics off. Dry states bypass it; an explicit setting wins.
  Unmatched configurations had omitted it, and WSM6 and Kessler profiles
  had disabled it. Forecast answers change.
- The dynamical core corrects vertical-velocity damping and float32
  rounding, geopotential tendencies at open and specified outer rows,
  vertical advection at an open model top, mapped open-boundary map
  factors, and diffusion's sixth-order filtering, boundary deformation,
  outer-row TKE and prescribed surface heat flux. Routine output words
  are checked against compiled WRF 4.7.1. Forecast answers change.
- Surface vertical velocity at open and specified edges uses WRF's
  clamped terrain slope. Copying the interior slope outward had doubled
  its terrain-following term there. Periodic domains and interior cells
  are unchanged; edge answers change.
- Land points cap 2 m humidity at 1.05 times the lowest model level's
  vapour after the surface schemes, as WRF's surface driver does.
  Answers change for 2 m humidity.
- WRF `wrfinput` starts initialize unfrozen Noah soil liquid water from
  soil moisture and terrain-following vertical wind from surface flow,
  instead of retaining zeros. First-hour latent heat flux RMSE against WRF
  fell from 239 to 0.4 W/m2 on a 3 km comparison. With `usemonalb` off,
  Noah keeps WRF's seasonal background albedo instead of the file's monthly
  albedo. Answers change for these forecasts.
- `woof downscale` children of saved runs, including `--point`, build
  terrain, land use and soil at their own spacing, using Copernicus GLO-30
  terrain at 1 km or finer. Parent states and every boundary frame are
  blended and rebalanced as WRF's `ndown` does, within a few float32
  epsilons of compiled WRF 4.7.1 on column checks. A WPS_GEOG tree is
  required (`woof fetch-geog` or `--geog-root`); a missing tree is refused.
  `--parent-terrain` retains the old route, and `[static]` records the
  choice. Answers change for these children.
- Downscaled children and DA nests take steep-terrain acoustic substeps
  and the `epssm` floor from their own terrain, keeping automatic parent
  choices. Lower explicit `epssm` values are refused. Answers change when
  the child's terrain needs the rule and the parent's terrain did not.
- Adaptive steps land on history, radiation, cumulus, surface and
  boundary-layer times; history intervals need only be whole seconds.
  Prepared forecasts honor adaptive history alarms and delayed domains
  start on time. Answers change where those times fell between steps.
- Disk estimates count each domain's physics fields, `history_vars` and
  output window. Trimmed runs that fit start; insufficient space is
  refused before starting.
- BEP+BEM workspace estimates count urban columns, including forecasts
  starting while their source posts, keeping a forecast that fits on the
  card. Configuration checks explain their upper bound.
- A resident nest under a streamed or split parent restores from the
  shared scratch arena without a `KeyError`.
- Streamed final-state digests use the resident forecast's model time.
  Fractional adaptive steps no longer leave identical forecasts with
  different digests through a 0.27 ms time difference after 3 h.
- Resident, streamed and out-of-core checkpoint writers record the run
  configuration alike. Checkpoints of the same state no longer differ in
  their header.
- CPU-only validation and SASE resource accounting work without CuPy.
  Windows numerical receipts resolve their own paths.

Available since 1.0.1:

- `sf_urban_physics = 1` runs single-layer urban canopy under Noah or
  Noah-MP; `2` runs BEP with YSU or MYJ; `3` adds BEP's building energy
  model. Local Climate Zones and NLCD urban classes are retained. A
  single-layer canopy above the first model level is refused. Urban
  physics is off by default; BEP under YSU applies nonurban drag once.
- `sf_surface_mosaic = 1` runs Noah land-use tiles per cell. With the
  single-layer canopy, `mosaic_urban_canopy = "every_tile"` also treats
  town tiles in mostly rural cells. Mosaic is off by default.
- The UW moist-turbulence PBL (`bl_pbl_physics = 9`) runs on the GPU and
  imports from WRF namelists. `[shared] zadvect_implicit = 1` runs
  implicit-explicit vertical advection with `w_crit_cfl`, including the
  two corrected boundary terms. Implicit advection is off by default;
  answers change.
- Terrain accepts WPS GEOGRID.TBL smoothers or no smoothing through
  `woof domain --terrain-smoothing`, domain static settings or GEOGRID.TBL
  import. `smooth_precision = "wps-float32"` or
  `--terrain-smoothing-precision` matches default `geogrid.exe` smoothing;
  float64 remains the default.
- WRF namelists retain `history_begin`, `history_end`, `smooth_cg_topo`,
  `topo_shading` and `slope_rad`, and imports name all unsupported keys
  together. Restarts can change history windows where cadence can change.
  WUDAPT Local Climate Zones import with their urban land cover.
- Byte-identical physics speedups remain included: per call, legacy
  RRTMG 7x, RUC 7 to 22x, Noah-MP 5.8x, MYNN 4.5x, NSSL 1.9 to 3.1x,
  P3 1.5 to 2.2x, RTE-RRTMGP 1.6x, Thompson 1.3 to 1.6x, Morrison
  1.2 to 1.5x, and Milbrandt-Yau, Kain-Fritsch, New Tiedtke, Shin-Hong
  and Grell-Freitas 1.1 to 1.5x. Dycore bookkeeping and nest forcing
  use fewer launches.
- `[shared] adaptive_nest_lattice = true` selects the root step with the
  fewest nest cell-steps; the three-domain 250 m comparison was 6% faster.
  It is opt-in and changes answers.
- `woof fetch` takes posted leads in order and records their first
  availability or an already-up marker; late leads exit 75, AIGFS waits
  for its GDAS donor, and events carry every source's leads.
  GFS, native HRRR and mapped-source trees prepare their start and nests
  first, then stream later boundaries; storm-following trees start at
  that head too.
  `woof sources` shows posting schedules and lateness budgets; fetch
  `latest`, `--readiness` and `--whole-cycle` remain available, and
  `woof go --cycle` selects the cycle. HRRR DA uses expected posting
  times, and `latest` probes AIFS f360 from +5 h 26 min. Config preparation
  can pin its CPU or GPU backend with `woof run --preprocess-backend` or
  `[case_data] preprocess_backend`, and `--preprocess-workers` sets decode
  concurrency. A 48 h HRRR decode on 24 cores took 140 s instead of 480 s.

## 1.0.1 (2026-10-01)

The engine's speed work, forecasts that start before their inputs finish
posting, and the fixes made since 1.0.0.

New:

- Faster physics, output byte-identical: legacy RRTMG radiation about 7x
  faster per call (product-suite forecasts up to 13% faster), RTE-RRTMGP
  1.6x faster per call, MYNN about 4.5x faster per call, and RUC land
  surface as six fused kernels instead of up to 3,000 launches (7 to 22x
  faster per call). The dycore step and nest forcing make fewer launches
  and host waits (nest forcing takes 25% less time).
- `WOOF_PHYSICS_MEGAKERNEL=1` runs the physics glue as a few fused kernels:
  a 100 x 80 default-suite step is 1.12x faster and forecasts are
  byte-identical. It is off by default, and a receipt names it when on.
- `[shared] adaptive_nest_lattice = true` picks the adaptive outer time step
  with the fewest nest cell-steps (6% faster on a three-domain 250 m tree).
  Opt-in: answers change.
- Sources are fetched as they post: `woof fetch` takes each forecast time as
  soon as a host has it, `--cycle latest` needs only the first ones, a time
  that passes its late limit exits 75, `--whole-cycle` keeps the old rule and
  `--readiness` says whether a run can start. `woof sources` shows each
  source's posting schedule and lateness budget.
- Domain trees from HRRR pressure levels, mapped sources and GFS prepare
  their start time and nests first, then one boundary interval at a time, and
  the forecast starts on that head, waiting at any interval not yet prepared.
  Output is identical to waiting for the whole preparation.
- The source decode uses every core (`woof prep --preprocess-workers` sets
  the count): a 48 h HRRR decode on 24 cores took 140 s instead of 480. Prepared arrays
  are unchanged.
- GEM GDPS, and every regular latitude/longitude source, decodes only the
  window its domains read: a 6 h GDPS preparation took 32 s and 1.5 GB of
  memory instead of 175 s and 12 GB.

Fixed:

- With `hypsometric_opt = 2`, which `woof domain` writes, a CPU-prepared
  state takes glibc's `log1pf` on every machine, so it no longer differs in
  the last bits on AVX-512 machines (the 1.0.0 known issue). The default
  hybrid coordinate and the preparation's exp, log and pow are also the
  same on every machine now.
- Under the adaptive clock, the steep-terrain cap on `max_time_step` comes
  from three-hour runs of the adaptive clock itself: a 2.25 km ridge under a
  24 m/s crest wind keeps 30 s and a 3 km CONUS domain keeps 24 s up to
  30 m/s, where 1.0.0 capped them shorter.
- `--cycle latest` waits each publisher's measured delay for that cycle hour
  and no longer skips posted cycles of seven sources. ECMWF's "slow down"
  answer is retried for up to 10 minutes.
- An AIFS forecast starting after f000, and a GEM GDPS window starting after
  f000, now prepare: the land mask and terrain come from the cycle's f000.
- A storm-following nest's 1 h rain draws at the hours it moved, and ground
  it moves onto starts from the parent's rain total.
- A nest that starts after the forecast draws live and through
  `woof render`, where every frame was refused.
- A forecast waiting for a boundary interval says so on its heartbeat,
  progress file and events, and `woof go` no longer stops it after 120 s.
- Two card preparations of the same inputs now publish identical prepared
  caches; per-run measurements stay in the receipt.
- In a cloud-cover picture, clear sky under the first level (10%) draws
  nothing, so the map shows through instead of a white sheet.
- Hex and global maps and cross-sections name their model, WOOF Hex and
  WOOF Global, in the line under the title, where they said WRF.
- The global model's quickstart configurations name the `woof` commands to
  fetch its analysis and run it.
- The identity texts name WOOF (the 1.0.0 known issue): the pin tables
  `woof global pins` and `python -m woof.globe.spectral pins` print, and the
  hex model's contract texts and messages. They carry new ids (global pins
  v3, Level-3 pins v2, hex adapter contract v3) and no arithmetic moved.
  1.0.1 reads the 1.0.0 ids as the same arithmetic, so a global checkpoint,
  export or receipt written by 1.0.0 resumes and validates.
- A global receipt's `libraries` block records the `recast-woof` version.
  1.0.0 receipts recorded `recast-woof-data` and not the package itself.

## 1.0.0 (2026-09-29)

Recast WOOF 1.0.0 is the regional engine of release 2.8.0 and the fixes made
since, under the WOOF name, with the hex and global models inside it and the
next plots.

New:

- One install, `pip install recast-woof`, carries the regional model (`woof`),
  the hex model (`woof hex`, preview) and the global model (`woof global`,
  preview). One version covers all three.
- The Python package is `woof`, the command is `woof`, the table companion is
  `recast-woof-data` and settings are spelled `WOOF_*`. Settings spelled
  `GPUWM_*` keep working.
- Plots take the shape of their domain: a square or tall domain fills the
  picture instead of leaving white space, the header runs the full width, the
  colour bar matches the map's length, wind barbs are larger and place names
  are drawn on every domain. County lines are left off at continental scale.
- `woof render --diff A_RUN B_RUN` draws any product as run A minus run B on a
  zero-centred colour bar, frame by frame at each valid time both runs hold.
  Runs on different grids or times are refused by name.
- `woof hex` runs on any NVIDIA GPU that can run its kernels (compute
  capability 7.5 and newer, with a CUDA 13 runtime). Each run's receipt
  records whether the card is anchored, meaning its architecture's numerical
  contract was measured against the reference card's: the RTX 40 series and
  the L40S (sm_89) are anchored, as are sm_86 (the RTX 30 series) and sm_120
  (the RTX 50 series). Any other card runs unanchored, and its receipt says
  so. `woof hex doctor` names the card, its compute capability and whether
  it is anchored.
- The global model reads its trace-gas and ozone climatology from a shipped
  table instead of the RFMIP input file; its forecasts are unchanged, bit for
  bit.

Fixed:

- A forecast from HRRR (native or pressure levels), ICON-D2 or a mapping that
  declares cloud water, rain, ice, snow and graupel now brings them in through
  its outer edges, instead of water vapour alone, so the analysed cloud and
  snow no longer drain out of the domain. Sources that publish none, such as
  GFS and ERA5, are unchanged. A forecast started from WPS met_em files or
  from wrfinput and wrfbdy files keeps water vapour alone at its edges, as
  WRF's real.exe writes them.
- Snow-covered land whose analysed top soil is more than 30 K colder than the
  snow surface, as older HRRR analyses carry under western snowpack, now
  starts with its soil rebuilt from the skin temperature down to the deep soil
  temperature. On a 2017 Idaho case the 2 m temperature bias at snow-covered
  stations over the first 3 hours went from -2.5 K to -0.2 K.
- A nested domain on its parent's vertical levels, including one spawned or
  moved during a run, now gets the same pressure correction as the outer
  domain. Its base state was held at single precision, which zeroed that
  correction.
- A nest spawned during a run is checked against the card's free memory
  before it is built. A card that cannot hold it stops the run with a message
  naming the nest's sizes and the memory free, instead of a CUDA out-of-memory
  error part-way through. Moving a nest from Python now stages through host
  memory by default, instead of holding the old and the new nest on the card
  at once.
- A forecast prepared with engine release 2.8.0 still runs, on
  water-vapour-only edges (it says so in one line) and with the soil it was
  prepared with. Prepare it again to get the fixes above.
- A forecast too big for the card's free memory is refused before anything is
  allocated, naming what it needs and what the card has, instead of stopping
  in a CUDA out-of-memory error part-way through. `woof run` (one domain or
  a domain tree, before anything is downloaded), `woof downscale`, the
  prepared and domain-tree forecasts and the loaders price the state,
  physics and boundary tables first, and `--no-memory-gate` skips the check
  wherever a forecast runs. A pinned `[tiles]` tiling that does not fit
  names the largest tile that does.
- Every memory check prices the tables that carry a source's cloud and
  precipitation at the edges: the `[tiles]` decision, the check before a
  forecast is built, the card reservations of `woof stream` and `woof
  multi-run`, and the host memory a streamed forecast keeps its edges in.
  They priced water vapour alone, so a HRRR-forced run could be let onto a
  card or a host that could not hold it. `woof check`, `woof run-plan
  --estimate` and the sizing table of `woof domain` count the same tables in
  every figure they print.
- A `woof run` preparation on the card prices the memory pool's reserve at
  1.25 times its arrays instead of 1.20, which a measured run exceeded, and
  `woof check` prices it the same way.
- A `woof run` domain tree whose nest starts after the forecast and streams
  through `[tiles]` now runs past the nest's start, where it stopped at the
  nest's first step. A nest that starts late is checked against the card's
  free memory before it is rebuilt at its start, and no longer holds two
  copies of itself in memory there.
- A domain finer than 1 km made by `woof domain` with no physics named now
  runs Thompson microphysics with MYNN and RUC instead of the source's YSU
  and Noah default. Scored against stations and ceilometers on marine fog
  days at 750 m, that suite kept more coastal fog and low stratus (stratus
  CSI 0.71, against 0.52 with Noah and 0.55 with YSU and Noah). This is
  limited observation evidence: the selection record publishes neither a
  case count nor a reproducible score receipt. A source whose soil RUC
  cannot start from keeps its own default.
- `woof domain` refuses a RUC land-surface suite on a source whose soil RUC
  cannot start from (GEM GDPS publishes one soil layer and RUC needs two),
  instead of leaving the refusal to the preparation after the download. A
  `--source hrrr` domain with nests now runs a RUC suite.
- `woof local-da` on a rung finer than 1 km with no physics named now runs
  Thompson with MYNN and RUC, as every other door does at that spacing,
  instead of the source's YSU and Noah default. A published local DA case on
  `--source hrrr` now launches, where it was refused as an incomplete roster.
- A nested domain no longer gets stronger sixth-order damping than its
  parent. A suite that sets `diff_6th_factor` below 0.10 on the outer grid
  wrote 0.10 on its first nest.
- A forecast with nests that names its physics suite now runs from every
  source, with that suite on the outer grid and the nests as `woof domain`
  wrote them. On every source but GFS and ERA5 it was refused after the
  download and the preparation.
- A stock-WRF export handed no physics choice writes the schemes its
  preparation was made with into each wrfinput and wrfbdy, instead of the
  WSM6 and Dudhia values of the bundled reference file.
- Under the adaptive clock, a 3 km domain over steep, high ground under a
  strong crest-level wind has its `max_time_step` capped where a longer step
  was measured to stop. A 3 km CONUS domain keeps the default 24 s under a
  20 m/s crest-level wind and is capped at 15 s from 30 to 40 m/s. A cap
  only ever shortens the step.
- A native `hrrr` configuration with no route namelists beside it now runs:
  the run writes them from the configuration. One the route cannot carry is
  refused by `woof go --dry-run`, where it used to pass the dry run and fail
  seconds into the real run.
- `woof warm-kernels --all-profiles` warms all 28 shipped physics suites. A
  suite that still fails is named with its remedy and the pass goes on to the
  rest, where the first failure used to stop it.
- `woof cycle --parent-kind mpas-cuda` checks the hex mesh against the card's
  free memory before the first leg, and refuses a mesh the card cannot hold
  with both numbers, where it used to fail inside the forecast worker.
- `woof cycle --parent-kind mpas-cuda` runs against an MPAS port checkout in
  the port's current package layout. The forecast worker imported only the
  package's older name, so such a checkout passed the memory check and then
  stopped at its first import. A checkout holding neither package is refused
  by name.
- A prepared DA cycle's memory check (`python -m tools.da_cycle_prepared`,
  which the storm nowcast also drives) counts each leg's analysis, sized the
  way the solve sizes its chunks, and each member perturbation's FFT work
  areas, which are now released after each draw instead of staying on the
  card.
- A state prepared on the CPU and the Python map projections no longer
  depend on NumPy 2.5's AVX-512 math, which most cloud Linux machines run:
  the prepared pressure takes glibc's `powf` on every machine, and the
  projections take tan, atan, log, exp and pow from the C library, as the
  native tools do. With `hypsometric_opt = 2` one step still does (see
  Known issues).
- `woof render` imports each frame once instead of twice. Every picture is
  byte-identical; the nine long windows of a 37-frame run took a quarter of
  the CPU time.
- The 10 m wind maximum pictures are drawn from the stored maximum
  (`WSPD10MAX`) when a history carries it. From an hourly history without
  one they are titled as the largest hourly snapshot, where they called the
  top-of-hour wind a maximum.
- `simulated_ir_satellite` drawn from a wrfout shows the model's cloud tops.
  Its cloud optical depths were a thousand times too small, so most cloud
  drew the ground's temperature; it now matches WRF-Python's cloud-top
  temperature to 2e-5 K in every column.
- A domain tree whose configuration carries a `[perturbation]` block now
  prepares from native HRRR, HRRR pressure levels, the other mapped sources
  and ERA5 as it did from GFS, and the forecast applies the bubbles at its
  start. Those preparations used to refuse the block.
- `woof hex doctor` reports a bridge that is present but not executable as a
  gap and prints the `chmod` that fixes it, instead of reporting it found.
- `woof hex render` reads the renderer's catalog rows, which gained a sixth
  field in release 2.8.0. The hex model on its own refused every render
  against that renderer with "produced no catalog rows".
- A `woof hex` forecast's hour-0 frame carries a real 2 m humidity and
  surface pressure. It published a 2 m humidity of 0 and a surface pressure
  of 100000 Pa in every cell, so the hour-0 2 m dewpoint map was drawn from
  zero humidity while every later hour was right.
- Hex receipts name the machine by a salted digest, never its network name.
- The global model's level weights take pow and exp from the C library, so a
  configuration's identity, which checkpoints and receipts record, no longer
  depends on the CPU's vector math. With numpy 2.5 on an AVX-512 machine every
  shipped global configuration had a different identity.
- `woof global doctor` exits 0 on a correct install. It graded two engine
  functions that no documented command reaches as gaps and exited 1; both
  now print as optional notes, and calling either still refuses by name.
- The global model's six observation decoders (`rw_atms`, `rw_gnssro`,
  `rw_ndbc`, `rw_igra2`, `rw_amv`, `rw_wis2`) ship in WOOF's bridge bundle,
  which now carries 34 programs and libraries. `woof global doctor` grades
  all fourteen Rust doors against WOOF's own pins and exits 0 on a clean
  install, and `woof global fetch-doors` says there is nothing for it to
  stage.

Known issues, each fixed in 1.0.1 for the reason it names:

- A configuration with `hypsometric_opt = 2`, which `woof domain` writes,
  still takes NumPy's float32 `log1p` when it is prepared on the CPU, so on
  a machine with AVX-512 its prepared `al`, `alt` and `p` can differ from
  another machine's in the last bits. The fix is made in the engine and
  ships in 1.0.1: it changes prepared bytes, and 1.0.0's release checks ran
  on the engine before it, so it goes in with the next engine re-point
  and its checks.
- Two kinds of identity text still print the engine's earlier name: the
  pin table `woof global pins` prints, and the contract texts `woof hex
  forecast --preflight` prints. Their bytes are hashed into every global
  pin and every hex receipt, so a changed word changes every digest, and a
  global checkpoint written before would stop resuming. They keep their
  bytes in 1.0.0 and are reworded in 1.0.1.
