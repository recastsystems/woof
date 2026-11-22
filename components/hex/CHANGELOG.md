# Changelog

## 0.6.2

Requires gpuwm>=3.0.0,<3.1. The kernel packs are new (release `kernel-packs-v0.6.2`): the TEMPO fix below changes
three of their modules, and the 0.6.1 packs would leave a correct-mode TEMPO run compiling those at every run. The
mesh generator `rw_mpas_mesh` and the history converter `rw_mpas_convert` are new (release `engine-bridges-v0.6.2`);
the other engine bridges are the ones 0.6.1 pinned.

Changed:
- **The meshes the doors generate are built under the generator's geometry 2 by default.** It grows the spacing by
  one rate per cell from each fine zone's own line, builds every spacing as asked (no halving ladder), and hands out
  no mesh with an obtuse triangle; the same ask is more columns, and the plan counts them: a few per cent on a zoom
  globe, and on the 3 km storm box over eastern Iowa (`woof hex regional --point 40.6,-91.3 --radius-km 300
  --fine-dx-m 3000 --background-km 96`) 51,671 columns against 38,278, whose forecast took 27.5 s of step loop a
  forecast hour against 20.8 s (2 h, one RTX PRO 6000). The 3 km CONUS zoom globe (1,959,938 cells) is one command,
  `mesh-plan --global-mesh --point 38.5,-96.5 --fine-dx-m 3000 --radius-km 2100 --background-km 96 --card
  rtx-pro-6000 --cards 4 --generate`: grid and static in 110 s on a 24-core host, with no obtuse triangle and no cell
  of four edges.
  `--geometry 1` (or a row's `"generator": {"geometry": 1}`) builds the halving ladder and the meshes of 0.6.1 byte
  for byte; a registered country row or region keeps the mesh it was made with.
- The mesh generator runs on at most 32 threads when nothing sets `RAYON_NUM_THREADS`
  (`RW_MPAS_MESH_MAX_THREADS` changes the cap); the mesh it writes is the same on any thread count.
- **Seven speed words for several-rank runs, each on unless set to 0; none changes a word written.**
  `GPUWM_HEX_LIMITER_BATCH` (the multi-rank monotonic limiter carries 7 species a pass: 13 tracers take two limiter
  halo exchanges a step, not thirteen), `GPUWM_HEX_HALO_PARALLEL_RECV` (each peer's halo receive on its own stream),
  `WOOF_HEX_HALO_TABLE_CACHE` (halo descriptor tables kept and uploaded from pinned memory),
  `WOOF_HEX_PUBLISH_LEAN` and `GPUWM_HEX_PUBLISH_FAST` (the host pools downloaded only when something reads them,
  and transposed on the card), `GPUWM_HEX_RESTART_RELEASE` (a same-host resume lets go of the whole-globe copy once
  every rank has its share) and `GPUWM_HEX_DIRECT_SHARD_RESTORE` (a resume from rank shards on the same ranks and
  partitions restores each rank from its own shard; rank 0 no longer assembles the globe). With every word set to 0
  a run is the bytes it was. They were made for many-card runs and change no speed on one card: two ranks of the
  3 km storm box over eastern Iowa on one RTX PRO 6000 step at 37 to 39 s a forecast hour with them on or off, as
  0.6.1 does.
- Updraft helicity on several same-host ranks is the same, word for word, as on one card: each rank sums a seam
  cell's vertex terms in one card's order (it differed in the last place at the seams).
  `WOOF_HEX_UH_RANK_ORDER=native` keeps a native MPAS rank's own order.

Fixed:
- **`woof hex regional --point` stopped at its init stage under geometry 2** ("362 of the 51671 cells ... lie
  outside the first guess", the 3 km storm box over eastern Iowa): the plan left out the width of the cut's seven
  boundary rings, so the HRRR window ended 30 km past the cut while the rings reach 66 km past it. The plan
  carries it, as geometry 1's always did, with each ring walked at its own spacing (past the cut geometry 2's
  spacing keeps growing), and the regional door draws the HRRR window over the cut's own cells.
- **A region whose cut and boundary rings leave the HRRR grid was found only after the mesh and the download.**
  The regional door now reads that circle against HRRR's grid first and refuses by name, with the side and by
  how many grid cells, before anything is built or fetched.
- **A host the door starts for `--hosts` could not resume from its seal**: it seals its shards under the run's
  `.multihost/host<N>/` folder, and `--restart-from` pointed it at rank 0's ("1 of its 2 rank shards are not on
  this host"). It resumes from its own copy.
- **Same-host ranks resumed from a seal began HAILCAST's updraft count again**, so the next hour's hail fields
  differed from the run that never stopped (1,246 words over 156 cells of the 3 km storm box on two ranks; every
  other word was equal). The carried count and maxima reach each rank.
- `--hosts` on a host with no NCCL names the pip line that installs it.
- **hrrrv5 with MPAS-A's Grell-Freitas convection runs on the GPU.** `--physics
  config_convection_scheme=cu_grell_freitas` with `--physics-suite hrrrv5` had no GPU route: the RRFS 44c0849 route
  had no GPU entry for the scheme, and with one its resident glue refused the pairing with TEMPO. The scheme now runs
  on the card beside TEMPO, MYNN-EDMF, RUC and UGWP with the port MPAS-A's convection_permitting suite uses (the fork
  compiles the same scheme file); the route table names the pairing as its own variant, "hrrrv5 + Grell-Freitas".
  A choice that still has no GPU slot (cu_tiedtke or cu_kain_fritsch with hrrrv5) stops at the door before any download (`woof hex regional` and `global` read it before the mesh), instead of
  fetching 1 GB of tables and ending on "the packaged physics manifest has no assets", and the message names the
  schemes of that slot that run on the GPU with the run's other choices, as `--physics` flags.
- **`woof hex render` given several history frames and no `--simulation-start` filed every frame as lead 0**, so
  each frame's pictures overwrote the one before, and the door still ended `failed=0`. The door now reads the run
  start every history carries (`config_start_time`) and says so in one `START` line; frames that name two starts,
  or none, are refused with the flag that settles it. One frame takes its run start the same way: a +3 h frame was
  drawn as "Init" at its own valid time and "F000".
- **`woof hex --help` stopped on Windows** with `No module named 'resource'` (and then `'fcntl'`): two derivative
  modules imported POSIX-only modules at the top, and `--help` builds every door. They are imported where they are
  used; the derivative commands still run on Linux only, and say so.
- **Ten doors' `--help` stopped on Windows when their output went to a pipe or a file** (`forecast`, `global`,
  `regional` and seven more): Python writes such a stream in the ANSI code page there, and the arrow in their help
  text stopped it with `UnicodeEncodeError`. A stream that is not UTF-8 now writes a character it cannot encode as
  a backslash escape.
- **The release build's netCDF-4 step could stop on a bad zlib download** ("zlib.tar.gz: FAILED"): one server
  sometimes answered with other bytes. The build now fetches the pinned zlib 1.3.1 from upstream's GitHub release
  first and zlib.net's fixed path second, tries each three times, and names the size, sum and first bytes of any
  answer that is not the pinned file. A wheel built from the source archive takes the same road.
- **The scorer (`hex_obs_scorecard`) and the energy tables program (`rw_s2s_energy`) built to different bytes in
  different folders**: Cargo hashes a path dependency outside the project by its whole path, so the folder entered
  the programs. Both now reach the engine through a link inside their project and remap both roots, so one source
  builds one program wherever it is built.
- **A false light echo over the oceans: TEMPO's droplet activation never took the land mask.** Upstream TEMPO
  (v3.2.5 and its main branch) calls droplet activation in `cloud_condensation` without the cell's land mask, so the
  scheme's own floor (at least 0.15 of the aerosol number activated over water and sea ice, none over land) never
  runs; CCPP Thompson, the scheme this activation comes from, passes the mask (`lsml`). Our port copied the call as
  written. On coarse ocean cells the resolved vertical motion sits at the activation table's floor, marine stratus
  activates very few droplets and drizzles, and the radar product shows a sheet of light echo where there is none.
  Correct mode (the default) now passes the cell's land mask on both TEMPO generations (fork correction `TP6`;
  `--physics-correct-skip TP6` turns it off). Exact mode is unchanged byte for byte. No floor on vertical motion is
  added: TEMPO, WRF Thompson and CCPP Thompson all activate on the resolved motion.
- A cycle's land state no longer refuses a rounded zero just below a lower bound (an `acsnom` of -3.9e-17).
- **`tests/test_moving_vertical_cuda.py` failed 12 of its 55 cases when run on its own on a card**: it compares
  the CUDA remap bit for bit with the CPU reference, which holds under strict numerics only, and it took the
  fast default outside the battery. It now selects strict numerics itself. No kernel changed.

Text corrected:
- The 12 km `adjoint-test`'s time on the README's first screen and in the first-hour page is 5 to 13 minutes (418 to
  708 s on builds of 0.6.1 and 769 s on 0.6.2's, on one RTX 5090); it said 5 to 11.
- The guarded route's reading (0.996 to 1.000) names its step sizes and the build it was measured on, and the
  per-species check names its full setting (`AD_DOT_SELF=1 AD_DOT_PLANES=scalars`) and that it misses on rain at
  12 km.
- The User Manual's 2.2 no longer names a version for doctor's `WPS_GEOG` line.
- The moving check's jolt figures in the README and the moving-nest guide are 0.6.2's own: 12.97 to 1.014 on the
  gentle case and 152.8 to 1.029 on the steep case on a card, 19.3 to 1.04 on the CPU. They gave a first
  measurement's 16.2 to 1.01 and 21.4 to 1.04, which today's check does not print. The User Manual's *What's new
  in 0.4* gives the same figures.
- The README's local-time-stepping section says what a one-class run is bit-identical to: the option-off run on the
  option's own schedule (RK3, six acoustic sub-steps, no FCT limiter), not a default run, which takes RK2 with four
  sub-steps and the limiter on.

Added:
- **Ranks on several hosts.** `woof hex forecast ... --hosts hostA:0-7,hostB:0-7` runs one rank process per listed
  card across machines, halos through NCCL; a checkpoint is sealed as rank shards on every host and a run resumes
  across hosts from them, a joined host from its own disk. A launcher that starts every host itself sets
  `WOOF_HEX_MULTIHOST_EXTERNAL` with `MASTER_ADDR`, `MASTER_PORT` and `WOOF_HEX_MULTIHOST_TOKEN`. Off by default:
  `WOOF_HEX_PER_RANK_OUTPUT=1` (each rank writes its own cells),
  `WOOF_HEX_PER_RANK_START=1` and `WOOF_HEX_START_SHARDS=DIR` (no whole-globe start held in memory). Every host
  needs NCCL, which no gpu extra installs: `pip install nvidia-nccl-cu12` in the same environment. Shown with the
  120 km globe, 6 h with a 3 h seal, one rank on an RTX 4090 host and one on an RTX 5070 Ti host on a home network:
  in strict numerics the words of two ranks on one RTX PRO 6000 (0 of 706,509,363 history and diagnostics words
  different) and, resumed from each host's own seal, of the run that never stopped (0 of 302,789,727); in fast
  numerics, the default, the door's ssh start and a launcher's start write the same words and the resume is exact,
  but fast words depend on the card family, so they match a one-host run only on the same families. No speed is
  claimed for several hosts.
- A start row `ifs-gfs-top`, selected by name only: ECMWF IFS open data at 0.25 degree with NCEP GFS above 50 hPa.
- **Three surface and boundary-layer closures on the hrrrv5 suite, each off unless named, one at a time:** a
  surface-flux network (`--physics config_surface_flux_net=r4_a50`), a neural boundary layer
  (`config_neural_pbl=v1`, or `v2` with its dry-air gate) and an 8-parameter closure with no network
  (`config_autophysics_pbl=v2`, `v3_bowen` or `v2_warm_night`); the `GPUWM_HEX_SURFACE_FLUX_NET`,
  `GPUWM_HEX_NEURAL_PBL`, `GPUWM_HEX_NEURAL_PBL_DRY_GATE` and `GPUWM_HEX_AUTOPHYSICS_PBL` words take the same values.
  They act after RUC's land step and after MYNN-EDMF. Their weights and parameter sets ship in the wheel, each
  checked by its SHA-256 before the first step; the two networks need the `[torch]` extra. Two at once, the dry-air
  gate alone, a route without RUC land and the MPAS-A v8.4.2 suites are refused by name. Off, the forecast is the
  plain model bit for bit. NOTICE section 20 names what each was trained or fitted on.
- **A learned column correction on the forecast's external-tendency slot**, off unless
  `WOOF_HEX_LEARNED_CORRECTION=/path/checkpoint.pt` names a network (`GPUWM_HEX_LEARNED_CORRECTION_ARCH`,
  `GPUWM_HEX_LEARNED_CORRECTION_SCALE`); one card only. No weights ship with the package. Unset, the forecast is the
  plain model bit for bit.
- **Hail size: HAILCAST, on by default wherever the mesh is finer than 4 km.** The forecast writes HRRR's hail fields
  to its diagnostics stream under HRRR's names and units, each the largest since the last whole-hour frame (so the
  hourly frame holds HRRR's hourly maximum): `HAILCAST_DIAM_MAX`, `HAILCAST_DIAM_MEAN`, `HAILCAST_DIAM_STD` and
  `HAILCAST_DHAIL1`-`5` in mm (HAILCAST's stone model, HRRR v4.1.21's by default), and `HAIL_MAXK1` / `HAIL_MAX2D`
  in m (the size distribution's largest hail at the lowest level and in the column: TEMPO's own on the RRFS route,
  woof's `hail_psd` on MPAS-A Thompson and WSM6). The stone model is woof's, compiled from the installed engine,
  whose columns match WRF's and HRRR's Fortran; a woof whose hail kernels differ is refused with one log line and the
  forecast goes on without these fields. Runs on several cards and runs resumed from a checkpoint give the same fields
  as one card and a run that never stopped. `WOOF_HEX_HAILCAST=0` turns it off, `=1` runs it in every cell,
  `WOOF_HEX_HAILCAST_VERSION=wrf_461` takes WRF 4.6.1's stones. It costs time where it runs: on the 3 km storm box
  over eastern Iowa (38,278 cells, HRRR 2024-05-21 15Z, 2 h, one RTX PRO 6000) the step loop took 20.8 s a forecast
  hour with it and 18.2 to 18.4 s with `WOOF_HEX_HAILCAST=0`, as 0.6.1 does (18.2 to 18.7 s); with it off that
  run's words are 0.6.1's. `woof hex render --products hail_max_hailcast,hail_max_surface,hail_max_column` draws
  them (they are not in the default map set): the converter the wheel bundles, which the door runs before any woof
  bridge, now maps `HAILCAST_DIAM_MAX` in mm and `HAIL_MAXK1` / `HAIL_MAX2D` in m and keeps their peaks. Before it, a
  hail product named at the door was refused ("not stored: hail_max_hailcast_surface") whatever woof was installed.

Derivative engine (tangent, adjoint, sensitivity):
- Added: `--inner-model simple` on `tangent`, `adjoint-test` and `sensitivity`, a simplified inner model; off by
  default.
- Added: `--physics config_radt_mcica_seed=cell` seeds RRTMG's McICA cloud sample by cell, so a recording's
  perturbed forwards and its linear model hold the same sample; `pressure`, MPAS's rule, stays the default, and a
  door asked for one rule on a plan recorded under the other refuses by name.
- Added: recordings as several rank threads on one card (`record --ranks N`, the doors' `--record-ranks N`) and a
  made row run as N ranks (`--rank-plans`); `--cumulus` records a plan with a cumulus scheme on. All off by default.
- Added: the external-tendency slot (`woof.hex.ad.external_tendency`), a caller's tendency added to the
  boundary-layer rates once a step, with its cotangent handed out each step of the reverse (the torch bridge).
- Changed: rows made for `--mesh` offer `--boundary-layer guarded` (the live MYNN-EDMF derivative); the made
  limited-area classes run it up to 30 minutes and the held closure past it.
- Changed: every whole-state dot reading prints its conditioning beside it.
- Fixed: a plan recorded from a default forecast lost part of three recover kernels' input cotangents (the short
  array lifetimes, `WOOF_HEX_MEM_ACOUSTIC`, have them write over their own inputs; a 16-step 12 km plan read a
  whole-state dot of 0.239). The derivative route records with `WOOF_HEX_MEM_ACOUSTIC=0`, the form every example
  plan was recorded in; the forecast's own default is unchanged. Sealing refuses a plan that aliases a checked
  kind's arguments in a pair no example plan ran.
- Fixed: the window had no rules for the model-top damping NOAA's hrrrv5 deck turns on (`upper_cam_f32`,
  `upper_rayleigh_f32`); a plan whose kernels the window cannot differentiate is refused at seal, naming each kind.
- Fixed: the regional transport's density check no longer writes a float copy of the field that has no derivative
  row (a 15-minute 12 km plan failed the release's own check).
- Fixed: New Tiedtke keeps one pipeline workspace per host thread, so rank threads sharing a card do not share it.
- Fixed: `woof hex regional --source hrrr-gfs` runs the blend row on a cut, and a failed blend leaves no start a
  rerun would reuse.

<!-- fork jacobbieker/woof fixes: begin -->
Fixed (brought from github.com/jacobbieker/woof, by Jacob Prince-Bieker, Apache-2.0; NOTICE section 19):
- **Every `--local-timestep` run stopped at step 0** with `TypeError: LocalTimestepAttachment.divergence_damping_3d()
  got an unexpected keyword argument 'native_stored_inverse'`. The local-time-stepping damping now takes the keyword
  the acoustic loop passes and forms its coefficient as the shared kernel does. From fork jacobbieker/woof PR 447.
- **`--local-timestep` runs no longer go eager whatever the graph policy says.** The rate-class launches are captured
  and replayed with the CUDA subcycle graph, with the same words and the same launch counts as eager
  (`WOOF_HEX_CUDA_GRAPHS_LTS=0` keeps them eager). From fork jacobbieker/woof PR 473.
- **A registry-id microphysics or land surface stopped the forecast door with a bare KeyError** (for example
  `--physics-suite arwen_mpas_column --physics mp_physics=2`: `KeyError: 'lin-mp2'`). A selector with no
  suite-table row is now an engine-only row. From fork jacobbieker/woof PR 379.
- **Every WOOF-route forecast recompiled kernels into its own scratch.** The v8.4.1 driver now keeps its kernel
  images in the box kernel cache (`<kernel cache root>/cupy`), keyed after the numerics options, so a run in a
  fresh work dir starts from the images the last one made (`WOOF_HEX_SHARED_KERNEL_IMAGES=0` keeps them per
  run). Output is byte-identical. From fork jacobbieker/woof PR 488.
- **A blow-up on the MPAS-suite route was reported as mesh corruption.** The wind-reconstruction refusal now says
  which check failed: the geometry against its sealed copy (naming the array), or a non-finite normal velocity
  (with its count and levels). From fork jacobbieker/woof PR 495.
- **A moving run started a cold process when a move went to another member than the prepared one, and prepared
  the next segment beside the running one even when the host had no room for it.** The resident record now
  follows the move, the preparation waits for host memory to hold a second copy
  (`WOOF_HEX_MOVING_PREPARE_HOST_GATE=0` turns the gate off), and freed heap is returned at each move. Same
  words. From fork jacobbieker/woof PR 497.
- **Every regional transport step waited on the host** in the FCT target-density check. The check is now one
  device OR over a broadcast mask, with the same flag. From fork jacobbieker/woof PR 216.
- **Small regional meshes ran the slower element FCT kernels**: the hoisted forms now start at 15,000 edges
  (was 30,000), with the same words, and `WOOF_HEX_EDGE_HOIST_CHECK` no longer refuses a regional mesh over
  its garbage column. From fork jacobbieker/woof PR 224.
- **A 256-level column could not be remapped by a moving run** (the guard refused more than 255 source layers;
  the one-byte layer index holds 256). From fork jacobbieker/woof PR 137.
- **`--physics slope_rad=1` stopped at step 0 with non-finite surface fluxes, and `topo_shading=1` changed every
  frame with a shadow the column call cannot compute.** Both (and `shadlen`) are now refused by name: the hex
  column call has no neighbour terrain. Mechanism from fork jacobbieker/woof PR 302.
- **The upper-damping flags did nothing on the MPAS suite route.** `--cam-coefficient`, `--cam-damping-levels`,
  `--rayleigh-u` and `--damping-start-m` were accepted and dropped (the history was word-for-word the run
  without them); they now reach `&damping` and the route's damping operators. From fork jacobbieker/woof PR 509.
- **Regional views lacked the lightning threat and the 10 m gust maps.** They now draw from the frames already
  stored. From fork jacobbieker/woof PR 433.
- **A cold kernel compile on a crowded host took 13 to 15 GB.** The compile pool is now sized by free host memory
  as well as by CPUs (1 GiB a worker within half of what is free), as woof does; `WOOF_KERNEL_PLAN_WORKERS`
  still sets it. From fork jacobbieker/woof PR 239.
<!-- fork jacobbieker/woof fixes: end -->

## 0.6.1

Requires gpuwm>=3.0.0,<3.1. A defect release: it fixes what a first user of 0.6.0 met, adds the command that finds
the examples now in the wheel (`woof hex examples`) and one option (`--boundary-layer guarded` on the 12 km example
row). The downloads 0.6.0 pinned are unchanged (the kernel packs, the engine bridges and the derivative engine's
files).

Fixed:
- **Installing `[gpu]` and then `[ad]` left two CuPy packages, and `doctor` said OK.** `woof hex doctor` now reports
  two CuPy packages as a required gap and prints the one command that leaves one. The pages say which install a user
  of the derivative commands needs: `pip install "recast-woof[ad]"` in a Python 3.14 environment of its own, not on top
  of `[gpu]` (pip does not remove one CuPy package when it installs the other).
- **The examples were not in the wheel.** The wheel now carries `run-32gb.sh`, `run-96gb.sh`, `own_measure.py`,
  `expected.json` and their README; `woof hex examples` prints their folder and `woof hex examples --copy DIR`
  copies them.
- **Surface-station (ASOS) scoring of a global run never measured.** The station list of the whole globe takes about
  9 minutes to fetch at the archive's pace with gpuwm 3.0 (533.6 s measured), and the scorer gave it 180 s, so all
  500 surface rows of a 24-hour global run read NOT_MEASURED. The request now gets the time it needs, and a run
  fetches the list once.
- **`woof hex global` and `woof hex country` ended "done ... score" while the scorer was still running.** The last
  line, in the terminal and in the door's log, now says scoring is still running in the background, with its process
  number and the file to watch, or gives the scorecard's status and its count of measured metrics when it has finished.
- **A correct-mode forecast printed internal names**: as the owner of a gap in its list of named gaps, and in the
  record line it prints when it finishes (the gaps' owners and the surface service's unit). The printed lines now give
  each gap and its reason; the receipt files, and the record line's surface service, name the part of hex concerned
  in plain words. A regional forecast's record line and receipt also carried development remarks in the evidence
  text of the configuration class it was admitted on; that text now states the measurements only.
- **The kernel pack was fetched with nothing in the terminal.** A pinned file fetched from the network now says one
  line first: its name, its size, that it is fetched once, and where from.
- **`rw_s2s_energy --help`** printed "No such file or directory". It prints its usage and exits 0.
- **`adjoint-test` on a measure of the storm itself** (`--metric qr`, `qc` or `w`) printed FAIL on a one-sided
  reading the forecast's own roughness decides. It now reads each step size at both signs beside a probe step of
  1e-06, counts a step size only where the roughness leaves it readable, and ends NOT PASSED (UNRESOLVED) with the
  reason when it cannot tell. No PASS covers these measures, as before; `tangent`, which takes no measure, no longer
  carries the call for their line.
- **`doctor`'s engine line** said the engine "this port was measured against is 2.9.7" on an install whose range is
  gpuwm 3.0. It now names the installed engine and the range, and says the 2.9.7 is where the seam manifest was last
  measured (information, not a gate).
- **Every `window.json` the derivative commands write carried an internal note with a date** in its `approximation`
  field. The field now states the approximation only.
- **An `hrrrv5` static build that lacked two geography sets said "stage them with woof fetch-geog"**, whose default
  set leaves those two out, so the same refusal came back. It now prints the exact command,
  `woof fetch-geog --root DIR --datasets NAME,NAME`, with the root and the missing sets.
- **`global-receipt.json` said `"ran": {"mode": "exact"}` for a correct-mode run on an MPAS suite.** It read only the
  WOOF-ArWen route's mode key; it now reads the MPAS suite route's as well and records the mode the run used.
- **A scorer command that failed with nothing on its error stream gave an empty reason** (`exited 2:`). It now gives
  the command's output, or the path of its log when it printed nothing.
- **`doctor` said this distribution "has no fetch path" for a mesh**, while `woof hex global` fetches its row's
  published mesh. The mesh line now says so, and that every other door takes `--grid` and `--static`.

Added as an option, not a default:
- **`--boundary-layer guarded` on the 12 km example row.** With nothing of the boundary layer held, a low-level
  temperature measure placed on the storm cell reads 0.996 to 1.000 of the gradient's prediction (centred, at steps
  0.1 to 0.01, measured on a 0.6.0 build with this row added), where the default
  route is 2.2 per cent short. It is not the default: the README's 12 km command ends NOT PASSED (UNRESOLVED) on it
  (466 s, card peak 23,610 MiB, on one RTX 5090, on a pre-release build of 0.6.1). `docs/differentiable/LIMITS.md` has both readings.

Text corrected:
- **The README's first install command stopped on a system Python** (`error: externally-managed-environment`, Ubuntu
  26.04). Every install the README prints now starts with the line that makes the Python environment it goes in.
- The package summary and the README's first screen say what the package is, with the install and check commands;
  "0.4 in short" moved under the earlier releases. The classifiers list Python 3.14 and no longer list Windows (the
  one wheel is Linux x86_64).
- `docs/differentiable/LIMITS.md` states the 12 km updraft reading plainly: it is the forecast's own melt-or-freeze
  switch in one column (the other 55 cells of the circle read 0.997 to 1.009), which no derivative follows.
- The 12 km example's seconds and memory on the README, the first-hour page, the limits page and the examples'
  page name the build they were measured on: 0.6.0's release builds (320 to 645 s on a first run), or a pre-release
  build of 0.6.1 (418 s). The 3 km answer checks are given against a bar of 4 single-precision roundings
  (1.786e-07 and 1.304e-07), not 1e-07. The counts of bundled programs (38 files, nine engine bridges), the engine
  versions named beside the 3.0 range and the manual's quickstart versions are corrected.
- **The global quick start stopped at its static stage on a fresh install** ("no geography root"): every static needs
  the geography, the published global meshes' included. The README's quick start and the global door's page print
  `woof fetch-geog` before the command, with its size, and the README and the first-hour page say which commands
  need it.
- GUIDE section 7 gives step 6's time and memory from an empty home, and the global door's page gives a fresh
  install's stage times, the run folder's size and how long scoring takes.
- The README no longer says that a nest under an `hrrrv5` parent stops: gpuwm 3.0.3, which pip resolves for this
  release, fixes it, and the README gives the measured run. It also gives the `woof fetch-geog` command an `hrrrv5`
  run needs on a fresh install.
- **The User Manual's quickstart stopped at two of its commands as printed.** `pip download --no-binary :all:
  woof hex` built every dependency from source and stopped in cmake's build on a machine without `make`; the page
  now prints `pip download --no-deps --no-binary :all: woof hex` and the line that copies the vertical specs out.
  Its render step named `cuda-history.*.nc`, which the default (MPAS) route does not write; it names `history.*.nc`,
  as do the forecast and render chapters and the transport products page, whose render of several frames now names
  the run's start (without it every frame was filed as lead 0 and the pictures replaced one another). The quickstart also prints the published
  mesh download and the `woof hex init-source` command that makes its met input, and says what a 0.6.1 forecast
  prints and writes. The first-hour page and the derivative engine's page say the `[ad]` line needs a `python3.14`,
  which Ubuntu 26.04 has and Ubuntu 24.04 has not.

## 0.6.0

Requires gpuwm>=3.0.0,<3.1.

Headline, each with its scope:
- **NOAA's hrrrv5 configuration, bit for bit.** A plain `woof hex forecast --physics-suite hrrrv5` runs NOAA's hrrrv5 deck, the configuration of RRFS v2. With `--physics-mode exact --numerics strict` its one-hour forecast on a limited-area mesh matches the patched native RRFS-MPAS 44c0849 reference in every stored word: on a 12 km cut of 2,890 cells through the plain door, with NOAA's 12 km cumulus on as the deck runs it, 0 of 86,751,014 words differ in 27 of 27 history, diagnostics and restart files; on a 3 km storm box of 38,278 cells through the plain door, 0 of 624,931,668 words differ in 11 of 11 files. Each hour starts from the HRRR 2024-05-21 15Z start and ran once, one process on one RTX 5090. The reference is 44c0849 built with a strict GNU recipe plus four labelled lines where native reads a value it never set (`docs/differentiable/GUIDE.md`, section 7). The default forecast (correct mode, fast numerics) is not bit for bit, by design, and other meshes, dates and lengths are not claimed by this line.
- **The tangent-linear and adjoint of NOAA's exact configuration.** `woof hex tangent`, `adjoint-test`, `sensitivity` and `param-grad` differentiate the exact-mode hrrrv5 forecast (`--physics-mode exact`, dynamics and physics), not the correct-mode forecast `woof hex forecast` runs by default. A gradient of correct mode is refused in one sentence, because this release has no derivative of the corrected arithmetic. The derivative commands run on NVIDIA cards of compute capability 12.0 only (GeForce RTX 50 series, RTX PRO Blackwell): the 12 km example on a 32 GB card, the 3 km and 120 km examples on a 96 GB card. The forecast also runs on older families: on one RTX 4090 the 12 km hour above matches the same reference in every word of the same 27 files (0 of 86,751,014 different).
- **How the derivative is checked, in plain words.** Two separate checks. The window check asks whether the adjoint is the transpose of the tangent-linear model: the two sides of its dot cannot agree to better than about one single-precision rounding, so a direction is inside when they agree within 4 roundings, and 8 or more directions must also not lean one way. On the 12 km example the README's direction reads +0.63 roundings, and seeds 1 to 8 read within 4 roundings with no one-signed drift (worst +2.26, mean +0.35, standard error 0.34; `adjoint-test --dot-directions 8`), through the command on one RTX 5090 with exit status 0; twelve seeded directions of the same window with the transport limiter's derivative live (the row holds it by default) read within 4 roundings, worst 3.38, mean -0.15, standard error 0.38. The gradient test asks whether the gradient predicts the forecast: the forecast is run again from the perturbed start at several step sizes, and a direction is inside when the real change is within 1 per cent of the change the gradient predicts. On the 12 km example the temperature bump and the steepest descent are inside and near-ground moisture is about 2 to 4 per cent short (named by the command); on the 3 km example the default route ends PASS at the row's own step sizes (0.3 to 0.03): the bump is inside (3.7e-04 from 1) and the steepest descent inside (1.9e-03 from 1), and the exact route ends FAIL; on the 120 km examples, over a 1-step window, the plain row ends PASS on the bump and near-ground moisture and the full-physics row ends PASS on three directions.
- **What the derivative holds, by name.** On the 12 km row the boundary layer's seven species outputs (the tendencies of cloud water, cloud ice and snow, and of cloud-droplet, ice and the two aerosol numbers) are held in both sweeps, the fix for single-precision scatter in the window check, and the cumulus scheme (New Tiedtke) is held by name: with its derivative live the 12 km window check misses by tens to hundreds of roundings, growing with the window, so the cumulus derivative waits for a double-precision form. The 12 km row and the 120 km default row hold the transport limiter's coefficients; the 3 km row's default route differentiates the limiter (its exact derivative) on windows up to 30 minutes, the example's 59 steps among them, and holds it on longer ones; the 120 km full-physics row differentiates it. The 12 km and 3 km rows hold the boundary layer's mixing length and the plume's own speed, area and condensation, and the lateral-boundary driving data (zero tangent; the relax zone is differentiated), and the 3 km default cloud route holds two riming terms (below); the 120 km default row holds the boundary layer's turbulence closure and whole plume and also radiation and cloud fraction; the 120 km full-physics row differentiates the plume, its speed on a ramp; both 120 km rows and the 60 km class hold the mountain-drag closure. The 120 km full-physics row's boundary layer runs `floored` by default: the live mixing's turbulence-energy path is bounded where the turbulence speed goes to zero (factor q*q/(q*q + 0.25), the forecast unchanged), because without the floor its descent reads OUTSIDE. Each receipt names what its row holds (`row_what`), and `docs/differentiable/LIMITS.md` lists every measured limit.
- **Measures on the storm itself are not covered by a PASS.** The PASSes of this release are for the README's measures (low-level temperature and moisture averaged over circles of 100 to 150 km radius). Read on the storm itself with `adjoint-test --metric`: on the 12 km example a rain-water measure fails (finite difference 0.118 against 0.01; read centred on both signs, the real change is 1.036 to 1.038 of the gradient's prediction at steps 0.1 and 0.03) and an updraft measure is unresolved (0.88 to 1.05); on a 3 km storm cell an updraft measure fails along the steepest descent (1.008 to 1.023), and so does a low-level temperature measure on the same cell (finite difference 0.245 against 0.01; the descent 1.022 to 1.025). The transpose passes on each. `sensitivity`, `adjoint-test` and `param-grad` print the row's measured line before they run for `qr`, `qc` and `w`; `LIMITS.md`, "Storm measures", has the readings.

New, the forecast:
- A plain `--physics-suite hrrrv5` forecast runs NOAA's hrrrv5 deck: the rrfs-workflow `parm/hrrrv5` namelist and, on a mesh within 1.2 times one of NOAA's spacings (12 km, 3 km, 1.25 km), that mesh's step, sub-steps and radiation interval (12 km: 60 s, 2 sub-steps, 30 minutes; 3 km: 20 s, 4 sub-steps, 15 minutes). The spacing is read from the init file when no `--mesh` is given, and the door prints the row it took. Any other mesh keeps the door's own step.
- New Tiedtke convection with the TEMPO cloud scheme runs on the card on the RRFS 44c0849 route (`config_convection_scheme = 'cu_ntiedtke'` with `config_gfl_sub3d = 1`, NOAA's 12 km cumulus).
- The README has a table of card families: what the forecast and the derivative commands were measured on. The forecast's 12 km hour on NOAA's hrrrv5 deck matches the native reference word for word on an RTX 5090 (Blackwell) and on an RTX 4090 (Ada Lovelace); the derivative commands run on Blackwell only.
- Correct mode, the default, fixes seven inherited RRFS defects wherever the scheme it runs still has them: MY1 (MYNN-EDMF's surface vapour-flux limiter deleted every downward moisture flux, dew and frost; upstream NCAR/MYNN-EDMF f3c9d9e), MY2 (MYNN surface air density from the virtual temperature; eb5029c), SF1 (the MYNN surface layer's cold-start reset only on a cold start), TP1 (TEMPO condensation stopped above an exactly saturated level; NCAR/TEMPO 881614f), TP2 (graupel volume from rain-ice collection missed its litre factor; 92aafb9), TP3 (TEMPO's first-call initialisation in every column; 4b40a93) and TP4 (the cloud-water over-depletion limiter scales the riming graupel volume with its mass; 71588d1). On NOAA's hrrrv5 pin set correct mode already runs TEMPO v3.2.5 and MYNN-EDMF v6.1.1, whose own text carries six of the seven; this release adds TP4 to TEMPO v3.2.5. `--physics-mode exact` keeps every one of them: its source texts are unchanged.
- Correct mode also takes MYNN-EDMF's upstream fixes for the ice-branch potential-temperature tendency and the condensate return (e447691, 87af8c6) and the boundary-layer height search and surface potential temperature (27da0f0) on the pinned scheme, and TEMPO's snow-moments correction (ace0eae: the mass moment at bm_s = 2), in the forecast and in the first frame's reflectivity. Exact mode is unchanged.
- The init door names each recipe switch it ignores.
- A region on the hrrrv5 suite says at once that the geography lacks the fork static's datasets, before it fetches any source hour.
- A region given `--vertical-spec FILE` on a producer suite (`--physics-suite hrrrv5`) is refused by name before any fetch, instead of running on the producer's L60 grid and ignoring the file; the default suite keeps the file.
- Regional views draw the severe-weather set (MLCAPE, SBCIN, MLCIN, MUCIN, 0-1 and 0-3 km helicity, 0-1 and 0-6 km shear, STP, EHI), and each map's theme carries the host's colour bars beside the credit line.
- `--init` from HRRR uses its own aerosols, hands the RUC soil over node for node when the first guess carries RUC nodes, and balances condensate against the producer's vapour; an HRRR-GFS blend start (HRRR inside its grid, GFS outside).
- The blend start takes `--outside-init INIT.nc`: a limited-area blend uses the mesh's own start (built with its boundaries) as its outside half, so a start at an hour with no GFS analysis (15Z) can be blended. A start valid at another hour is refused.
- `woof hex --version` prints the installed version.
- The source package carries `CHANGELOG.md`; `tools/native_dry_reference/run_native_case.sh` takes its MPI runtime and Python from `PATH` (no default folder).

New, the differentiable engine:
- The differentiable engine: the tangent-linear and adjoint model of the exact-mode forecast step, dynamics and physics, through four commands, `woof hex tangent`, `adjoint-test`, `sensitivity` and `param-grad`. It runs on one card of compute capability 12.0 (GeForce RTX 50 series, RTX PRO Blackwell); on a card of any other family the commands stop before the example downloads. Scope and limits: `docs/differentiable/README.md`.
- Three examples, fetched once on first use. Start with the 12 km cut of the storm box (row `hrrrv5-q12.96.2890.n40.60w91.30`, 2,890 cells, 15 steps of 60 s), which runs on a 32 GB card such as the RTX 5090; the README's first command is its `adjoint-test`. The 3 km storm box over a 15-minute window (`--row hrrrv5-q3.96.38278.n40.60w91.30`, 38,278 cells, 59 steps) and a 120 km global example of three steps (row `hrrrv5-x1.40962`; `--row hrrrv5-full-x1.40962` runs the same case with radiation and cloud in and the transport limiter differentiated, not held) need a 96 GB card. Install with `pip install "recast-woof[ad]"` in a Python 3.14 environment: the `ad` extra pins the CuPy and CUDA toolkit versions the examples replay.
- The answer check that `tangent`, `sensitivity` and `param-grad` print is a consistency check of the adjoint against the tangent-linear model, not a measure of how well the gradient predicts the forecast; `adjoint-test` makes that measure along its test directions. A measure built on radar reflectivity does not pass the gradient check, and the commands refuse it by name.
- Which code the derivative is of: the boundary layer's is of MYNN-EDMF c6d08709 (the code RRFS 44c0849 runs) and the surface layer's is of the MYNN surface layer v6.0.0, where a plain hrrrv5 forecast runs v6.1.1 and v6.1.0. A sensitivity in a cloud-topped boundary layer is that of the older boundary layer; the two surface layers differ only at the first step of a run.
- `adjoint-test` compares the gradient with the real model's change along four test directions (a temperature bump, the gradient's own descent direction, near-ground moisture and a smooth draw; `--directions` picks them) and prints each per step size. A PASS needs none outside the bar and two independent ones resolved; a direction the forecast cannot resolve is printed UNRESOLVED and never counts.
- `param-grad` gives a metric's gradient with respect to five physics settings; `--check` adds the parameter dot test. It prints and records the scheme bundle it runs in place of the row's (the mountain-drag closure is differentiated in it).
- Before the example downloads, a command checks the card's family and whole size, Python and the `ad` extra, and stops by name on a machine without them. A bump or region off the mesh is refused after the mesh file alone (2 MB for the 12 km cut). Free card memory, the host memory it needs, disk and, for `--init`, the recorder are checked after that mesh file and before the rest of the example (`--wait-card SECONDS` waits for card memory another job holds).
- `--strict-from` takes the strict check from a passing `adjoint-test` receipt of the same plan. `--seed N` sets the strict check's test direction in `tangent`, `sensitivity` and `param-grad`.
- The whole-state check scores the model's prognostic fields at the window end, one unit a field, where the plan's field map holds that boundary. Where it does not (a 15-step window on a plan recorded with 15 boundaries), it scores memory allocations and the command says so. `AD_DOT_SELF_UNITS=pool` asks the runner for memory allocations.
- The runner's window check has a per-species form (`AD_DOT_SELF=1 AD_DOT_PLANES=scalars`): each of the 13 scalar species is weighted and read on its own, and a species that carries no tangent is named. On the 12 km example it misses on rain (`docs/differentiable/LIMITS.md`), and no example runs it. It also names the form it ran; the independent form (a random second vector not made from the tangent, `--direction fieldxy` without `AD_DOT_SELF`) is in `docs/differentiable/CHECKS.md`.
- The per-launch transpose check of a replay builds its topology tables from the run's own mesh and refuses a stored table of another mesh by name.
- `sensitivity` draws its map over the whole globe. Every receipt names how its plan was recorded and the boundary of the plan at which the window end was read (`window_end_boundary`); `tangent` also writes the edge wind `u` at the window end.
- A perturbation that is zero on every cell is refused by name. A bump, bump level or metric region that misses the row's mesh is refused before the example downloads, from the mesh file alone (2 MB for the 12 km cut).
- On a card of another family the commands say which cards run the row (compute capability 12.0: GeForce RTX 50 series, RTX PRO Blackwell), which family the card is, and how much free memory the row needs. On the 12 km cut, as on the other examples, another Python or a missing `ad` extra is refused before any download, naming the file and what to install.
- `adjoint-test --box` takes the box as its metric region: the default circle went in beside it, and the two were refused together.
- `--card N` counts within a `CUDA_VISIBLE_DEVICES` a scheduler has already set. `WOOF_HEX_NO_LOCAL_GPU=1` stops every command before it reads a card, finds a runner or downloads anything.
- `WOOF_HEX_PAIR_ADD=plain` and `WOOF_HEX_OZONE_MONTHS=all` select the forecast forms used to record plans.
- The default and full-physics rows and the 60 km class hold the mountain-drag closure at the trajectory, because its derivative grows without bound past about an hour at 120 km; `param-grad` and rows made with `--suite hrrrv5-exact` differentiate it, and say so.
- On the 3 km storm row `sensitivity` leaves its program loaded for the next command, and a later gradient of the same forecast takes the forward of the gradient before (`--keep-forward on|off`). A command that finds a program loaded with other words stops it and names the word that differed; a later request whose start has moved is followed, and one whose start the forecast itself stops at is refused in a sentence while the program stays loaded for the next request.
- The 3 km default route's `adjoint-test` reading is in the command's `known limits`, the README and `LIMITS.md`: PASS, the bump inside (3.65e-04 from 1) and the steepest descent inside (1.92e-03 from 1) at the row's own step sizes 0.3, 0.2, 0.1, 0.06 and 0.03 (at 0.01 the descent moves the measure by only about 5 single-precision spacings of it, so that size reads rounding), read through the command on one 96 GB card, the same digits on two runs (the second on a 499 GiB host), with the number cloud route, the live transport limiter and the three double-arithmetic dynamics images; the exact route's reading is FAIL. The docs say what PASS, FAIL and NOT PASSED mean and which exit status each has.
- The 3 km storm row's default boundary-layer route (tape-free) differentiates the boundary layer's turbulence closure, the plume's heat, moisture and momentum and the returned boundary-layer height (of MYNN-EDMF c6d08709, under a weak-shear rule at 1e-5 s-2); the exact route (`--boundary-layer taped`) holds them, so the two differ by design and the command prints no fast-against-exact figure. `adjoint-test` and `tangent` run the row's boundary-layer route, the linear model a plain `sensitivity` runs, and say which (`--boundary-layer taped` tests the held form); the 12 km example row runs the same default route on each command it offers (`tangent`, `adjoint-test`, `sensitivity`; it refuses `param-grad`, which has not been run on it). The README's first command, `adjoint-test` on the 12 km example, passes with it (bump and descent inside the bar); near-ground moisture is about 2 to 4 per cent short there (unresolved at the command's step sizes; the held plume speed and condensation), and the held form fails along the descent and near-ground moisture directions.
- `sensitivity --boundary-layer guarded` on the 3 km storm row: the exact route with the boundary layer's mixing differentiated under a weak-shear guard, a named option for near-ground wind questions over short windows (not the default; the command says what was shown and what was not).
- The 3 km storm row's default cloud derivative is the number route (`--cloud-derivative number`): it holds the snow riming ratio where riming is slow and the riming fall-speed boost where a trace of cloud water would carry it to its cap, lets rain drop number and cloud droplet number follow the forecast, gives the rain-snow and rain-graupel collection tables their own slope between bins, and holds snow and cloud-ice deposition only where it would add the species' own mass within 200 s. It runs with the row's transport limiter live on windows up to 30 minutes; a longer window holds the limiter, runs the `gated` route and says so in one line. `--cloud-derivative gated` runs the first two holds with both numbers held at their forecast values; `--cloud-derivative shipped` runs the derivative without those holds. The global rows keep their own cloud images.
- `adjoint-test --plant-error KERNEL:SCALE` plants a known error in the backward code on purpose (`boundary-layer`, `acoustic` or `advection`; the adjoint side only), to show the transpose check catching it: it ends exit status 4 when the error is caught and 5 when it is not. On the 12 km example through the command on one RTX 5090, `boundary-layer:1.01` read 66.8 roundings against the bar of 4 and `acoustic:1.01` 271.5 roundings, each exit status 4; `advection` was not run on a card.
- `examples/differentiable/run-32gb.sh` and `run-96gb.sh` end with the highest exit status of their commands. With every card hidden, `run-32gb.sh` printed three one-sentence refusals and ended 0, so a run where nothing ran read as a success; it now ends 2.
- The 12 km example row states 47,104 MiB of host memory (was 46,080): the release check `adjoint-test --directions all --dot-directions 8` peaked at 46,681 MiB on one RTX 5090 host.
- The bit-for-bit claim can be walked: `docs/differentiable/GUIDE.md` section 7 rebuilds NOAA's patched reference from the source distribution's `tools/rrfs_44c0849_reference/` (`make_ref.sh` fetches RRFS-MPAS 44c0849 and its submodules at their pins, `SOURCE-PINS.json` names them, `rebuild.sh` builds it, `make_deck.py` renders NOAA's deck, `compare_outputs.py` compares every stored word). Steps 0 to 5 walked from an empty folder reproduced the reference of record's binaries byte for byte; the six-hour `x1.2562` hrrrv5 run from 2023-03-31 12Z, one process, read 0 of 129,420,310 stored words different from native with NOAA's published TEMPO tables (history 25,664,457, diagnostics 3,982,153, restart 99,773,700), on one RTX 5090 with its kernels already compiled.
- `tangent`, `adjoint-test`, `sensitivity` and `param-grad` say which forecast their linear model is of: the exact-mode one (`--physics-mode exact`, the mode every plan is recorded in), where `woof hex forecast` runs correct mode by default; each receipt names it (`physics_mode`). `--physics-mode correct` is refused before any download, because this build has no derivative rows of the corrected arithmetic. The recorder refuses `--physics-mode correct` among the words after `--` (it used to record the correct-mode forecast and label it exact), and the commands refuse a plan whose recorded forecast words end on another mode.
- The adjoint commands judge what the command asked before the machine's state: the card's family and whole size, Python and the `ad` extra, and a bump or region off the mesh are said before free card memory, host memory and disk. On a busy machine a wrong install or a point off the mesh was first told to free memory.
- On the 3 km storm row `adjoint-test`, `tangent` and `param-grad` state the host memory of their own window length on this host (they keep every step boundary, in host memory up to 0.4 of what is available) and are refused in one sentence on a host that cannot hold them: at 59 steps under 102,948 MiB available. Between that and the runner's default store (0.4 of what is available, which needs 158,958 MiB at 59 steps) they run with a smaller store in memory (`--bstore-ram-gib`, said in the output) and the rest in the run folder. `adjoint-test` there had stated 110,592 MiB and held 254,885 to 262,306 MiB on 512 GiB hosts; on a 499 GiB host it stated 303,410 MiB and held 278,824 MiB (19 of 60 boundaries in memory), PASS. A first `sensitivity --steps 59` with its answer check on that host held 240,043 MiB, its reverse sweep 191.3 s and 630.4 s in all, one 96 GB card. The row's default route asks the card for 96,600 MiB (was 96,000; a first `sensitivity` with its answer check held 96,214 MiB).
- `adjoint-test` prints the measured limits of the linear model it tests under its directions (`known limits`, each with what the run read along it) and names on its last line any direction its verdict does not cover because of one; the receipt records them as `known_limits`. The verdict and exit status are unchanged.
- New pages for the differentiable engine: `docs/differentiable/FIRST-HOUR.md` (cards, needs, the first command's output), `GUIDE.md` (a gradient of your own measure, the checks, recordings and their cost, the `libhexad-phys` C library as a variational system's linear model, exact and fast mode, the bit-for-bit claims) and `LIMITS.md`.
- `examples/differentiable/`: the 12 km example on a 32 GB card and the 120 km and 3 km examples on a 96 GB card as scripts, a Python example of two gradients through `SensitivitySession`, and the numbers to expect.
- The window check is judged in single-precision roundings of its dot. The model and its derivative run in single precision, so `<y, M x>` and `<M^T y, x>` cannot agree to better than about one rounding (2^-23 of the dot, 1.2e-07 relative). A direction is inside within 4 roundings (4.77e-07 relative), and over 8 or more directions (`--dot-directions 8`) their signed readings must not lean one way (the mean within 2 standard errors of zero). A missing or wrong derivative term reads tens to hundreds of roundings, or leans one way; rounding does neither. `--dot-bar strict` applies the earlier bar of 1e-07, which lies under one rounding. The command prints each reading in roundings beside the bar.
- On the 12 km row the boundary layer's seven species outputs (`rqcblten`, `rqiblten`, `rqsblten`, `rncblten`, `rniblten`, `rnwfablten`, `rnifablten`) are held in both sweeps of its default route, and `window.json` names them. The 3 km and 120 km rows run without this hold.
- The 12 km row holds the cumulus derivative (New Tiedtke) by name: in a 12 km plan recorded with it on, its 26 launch kinds run as identity in the tangent and the adjoint and `window.json` names them; every door on the row says so in one line. The forecast is unchanged, and the 12 km example's recorded forecast runs no cumulus scheme. The cumulus derivative is not in this release.
- The 120 km full-physics row (`--row hrrrv5-full-x1.40962`) runs its boundary layer `floored` by default: the live mixing's derivative through the turbulence speed is bounded by the factor q*q/(q*q + 0.25) of the forecast's own q, which leaves the forecast unchanged. `--boundary-layer taped` runs the row's own images, whose steepest descent reads OUTSIDE (the command names it). The plain 120 km row is unchanged.
- On the 120 km rows `adjoint-test` runs its own 1-step window (12 minutes) and reruns the real model at eight step sizes a factor of two apart; `tangent`, `sensitivity` and `param-grad` keep the row's three steps.
- `woof.hex.ad_session.SensitivitySession` runs on the 12 km row: each `gradient()` call there is the `sensitivity` command's own program for that measure, as `docs/differentiable/GUIDE.md` says (on the 3 km storm row the session keeps one program loaded between calls). `examples/differentiable/own_measure.py` ends a refusal in one sentence with exit status 2.
- A FAIL line names only what failed: each check that missed its bar, with its reading and bar (the whole-state dot in single-precision roundings, with its test direction), and the directions read OUTSIDE (with the known limit along one, where there is one); it adds the `window` known limit only when the whole-state dot failed, and says a window was not a valid derivative only when a window was not. The direction line of the whole-state check names the seed it ran (`test directions 1..8` for `--dot-directions 8`).
- `python -m woof.hex.ad.record` finds the wheel's own runner (`woof/hex/libexec/ad-binder`) without `--runner`.
- A host card queue tells a command its cards through `CARD_QUEUE_CARDS` beside `CUDA_VISIBLE_DEVICES` (and `CARD_QUEUE_RELEASE`, `CARD_QUEUE_MODE`, `CARD_QUEUE_VRAM_MIB`, `CARD_QUEUE_GPU_DONE`; `woof.hex.card_queue`); a site whose queue exports another prefix names it once in `WOOF_CARD_QUEUE_PREFIX` or `~/.config/gpuwm-hex/card-queue.json`, and hex reads both.

Fixed:
- `--point` (regional, the HRRR intermediate, the country cap) refuses a point at a pole given a longitude other than 0 and names the point it likely means: `--point -90,37.5` built a domain centred on the South Pole. A latitude out of range that reads as a longitude says so too. A core centred on the pole itself is written `--point -90,0`.
- `--point`, `--ellipse`, `--blend-core`, `--at` and `--box` take a value that starts with a minus after a space (`--point -33.9,151.2`); every point south of the equator stopped at "argument --point: expected one argument".
- A start refused because its cycle is not posted yet names the date when it is not today, and how long until it posts: a cycle typed five days ahead read "usually posts around 03:30Z, and it is 16:00Z".
- A start from ICON on the RUC-soil suites (hrrrv5 and the RRFS suites) has soil: ICON's eight soil levels are moved onto Noah's four layers before the start-state program, which reads only four layers or RUC's nine nodes. Before, every land cell started at 0 K and the start was refused.
- With `--physics-profile hrrr-like` (or `config_mynn_mixlength = 1` in correct mode) the MPAS-A MYNN's BouLac downward mixing length takes the sign NCAR/MYNN-EDMF corrected in v6.1.1 (PR 94): where two adjacent levels have exactly the same potential temperature, the length was short by twice the distance a parcel travels into that layer. Exact mode and MPAS-A's default mixing length (2), which never reaches this code, are unchanged.
- `woof hex country` builds its cold start for a profile named with `--forecast-arg=--physics-profile=...`: `hrrr-like` and `hrrr-like-aerosol` take their aerosol fields (`nc`, `nifa`, `nwfa`) from the monthly climatology, as the point and zoom-globe doors already did. Before, the region prepared, its start had no aerosol fields, and the forecast stopped at its first step with "physics profile 'hrrr-like' needs prepared init fields ['nifa', 'nwfa']", so no region could run either profile.
- IFS (`ifs`, ECMWF open data 0.25 degree) reads every archived cycle from 2024-03-18 12Z, at any posted hour. A cycle before 2024-12-02 18Z published no surface geopotential: its terrain comes from the 2024-12-03 00Z field, the same orography. The 2024 soil records (st, stl2-4, swvl1-4) decode, the 06 and 18 UTC runs before 2026-05-12 06Z are read from their scda path, and an hour past a past run's range (f240 at 00/12 UTC, f090 at 06/18 UTC, before 2024-11-12 06Z) is refused at planning. Before, a region from such a cycle started at a later hour failed at its first boundary file.
- IFS ENS (`ifs-ens`, one member) reads the same archived cycles. Its terrain donor reads the cycle's own oper or scda step-0 z, or the 2024-12-03 00Z z for a cycle before 2024-12-02 18Z, and a member selects the 2024 soil records (st, stl2-4, swvl1-4). Before, a member from a cycle before 2024-12-02 18Z failed at its first file, and a 2024 member had no soil.
- A start from IFS before the 2025-11-20 12Z run (`ifs`, `ifs-ens`), and every start from AIFS (`aifs`, `aifs-ens`), has snow on the ground. ECMWF open data published no snow before that run, and AIFS publishes none, so these starts began with none anywhere. They now take the same cycle's GFS snow water equivalent at the same lead (one 0.5 MB record). An IFS start from 2025-11-20 12Z on reads its own snow, as before.
- Hex regions from ICON (`icon`, DWD open data) get lateral boundaries through the run, and an ICON run can start at a later hour. The row reads every hour DWD posts (00/12 UTC: hourly to f078, then 3-hourly to f180; 06/18 UTC: the same to f120), each record is remapped with its own lead and the run's source cycle, and the GFS column above 30 hPa is read to f384. ICON's single-stream objects download 8 at once. Before, every ICON region was refused at planning.
- The regional door takes `--member` for an ensemble source (GEFS). The start and every boundary hour read that member, and a missing or out-of-range member is refused before the mesh. Before, every GEFS region on this door failed after its mesh was built.
- Hex regions from ERA5 take their boundaries from ERA5's own analyses, hourly. Before, both region doors refused every ERA5 region at planning ("era5 declares no forecast cadence for its boundaries"). Each boundary hour keeps only its intermediate and its receipt, so a 48 h ERA5 region no longer writes about 245 GB of downloads and composed frames.
- A region whose length is not a whole number of its source's posting steps reads the next posted hour past its end, on both region doors. Before, such a region (ICON 100 h, IFS 25 h, GFS 125 h) was refused at planning; a run past the cycle's last posted hour is still refused, and the refusal names that hour.
- A country region from ICON tells its boundaries the levels its boundary files hold (ICON's 19 and the GFS levels above them: 27). Before, every boundary hour was refused after it was decoded.
- `moving case --member N` writes the storm-following case's ensemble member, which `moving stage` reads, so a storm-following run from GEFS starts. A member on a source with no members, or one that is not a whole number of zero or more, is refused before the case is written.
- A country region reads as many source hours at once as its host has room for (half its processors, 2 to 8; `--endpoint-workers` still sets it). Its boundary files are the same bytes; ICON boundaries were ready in about half the time on a 16-core host.
- A new start row, `aifs-ifs-soil`: AIFS with the same cycle's IFS soil below 28 cm, so the RUC-soil suites (hrrrv5 and the RRFS suites) can start from AIFS. Before, such a start left every land cell at 0 K and was refused.
- Regional views draw the 2-5 km updraft-helicity tracks (the maximum over 1 h, 3 h and the run), as the WOOF-ArWen route does.
- Soundings and meteograms of a variable-resolution globe whose grid file is on the unit sphere were headed "Variable-resolution globe to 0 km"; the heading now gives the mesh's finest spacing.
- A global start whose pole cell lies a hair past 90 degrees (the x1.40962 mesh) is read as the pole instead of refused.
- `woof hex forecast` refuses, before it takes a card, a start file whose soil levels belong to another land model.
- `woof hex forecast` with a grid, its static and a start file built on that pair by `woof hex init` or `init-source` runs. Before, the start file's parent named only the static, and the forecast was refused before its first step ("static parent_id does not match grid file_id"); the manual's quickstart was one such run.
- `woof hex global` finds `rapidgzip` beside the Python interpreter when that folder is not on PATH.
- A wheel built from the source distribution downloads the nine pinned engine bridges by SHA-256.
- `pip install -e .` no longer fails while writing `kernel-set.json`.
- On a variable-resolution mesh the horizontal mixing's mesh scaling is computed with GNU `powf`, the power function native MPAS-A built with gfortran uses, in every mode. With it, in exact mode with strict numerics, one-hour hrrrv5 forecasts on a 3 km and a 12 km limited-area cut match native RRFS-MPAS 44c0849 in every word of every history, diag and restart file (the 12 km one through the plain door, and with New Tiedtke cumulus); before, the 3 km case differed from its first step. The regional kernel set is minted again at the new sources.
- In exact mode a limited-area run drives each boundary scalar from the boundary species native RRFS-MPAS 44c0849 applies in that slot (it exchanges rain number and ice number at the boundaries). Correct mode keeps each species on its own boundary field.
- A limited-area forecast with a physics suite clamps negative scalars once a step, before the cloud scheme, as native MPAS-A does. On the shared dycore driver a second clamp ran after the cloud scheme; the RRFS driver already ran one.
- A card hidden with an empty `CUDA_VISIBLE_DEVICES` (or one starting with -1) was handed to the derivative runner as card 0. Now the commands refuse in one sentence where the runner would start, and before they read a card's memory or family (`nvidia-smi` does not follow `CUDA_VISIBLE_DEVICES`).
- A 2024 IFS start can be fetched: an archive answering 429 or 503 ("slow down") is waited out on its own limit of 40 tries, at most 30 s apart; any other failure keeps the usual ten tries.
- `sensitivity.nc` and `tangent.nc` hold the latitude and longitude of each cell centre and edge midpoint in degrees (`latCell`, `lonCell`, `latEdge`, `lonEdge`), and each field names its pair in `coordinates`, so a gradient can be placed without the mesh file. `SensitivitySession.gradient()` returns `latCell` and `lonCell`.
- The `sensitivity` map of a `qv` measure draws `dJ_dqv`. Before, it showed `dJ_dtheta`. The map of a measure of a field other than theta, theta_m or qv draws `dJ_dtheta`. For every measure other than theta and theta_m, the command prints one line naming the field it draws, the level, and the measure's own levels.
- The 12 km example row has 59 levels, as its recorded plan does. It said 55, so `--level 56` to `--level 59` were refused.

## 0.5.0

Requires gpuwm>=3.0.0,<3.1.

New:
- Runs on gpuwm 3.0, released with it. On the regional, global and hrrrv5 gate cases the MPAS route gives output bit for bit the same as on gpuwm 2.9.13. WOOF-physics runs follow gpuwm 3.0's physics (its changelog), for example its new start for Noah-MP snow layers.
- Physics defaults are the latest upstream releases; every pinned version stays selectable by name, and the bitwise proofs run the pins.
- The RRFS route runs MYNN v6.1 by default (MYNN-EDMF v6.1.1 + MYNN-SFC v6.1.0). `--physics config_mynn_version=pinned` runs RRFS 44c0849's MYNN, bitwise to native.
- TEMPO v3.2.5 is the default TEMPO on the RRFS route, with the hail-hyperaware category on. `config_tempo_version = 'v3.0.3'` runs the RRFS 44c0849 TEMPO; `--physics-mode exact` runs v3.0.3 unless a version is named.
- MMM-physics moves to 550b5b4 (WRF v4.8.0) for YSU and the revised MM5 surface layer, with the KIM form-drag option (off by default). `mmm_version=a4baf7f` runs the MPAS-A v8.4.2 code.
- KIM orographic drag (MPAS PR #1409) is selectable with `--physics gwdo_version=kim2026`. The v8.4.2 drag stays the default. The KIM option builds its orographic statistics with gpuwm 3.0's static builder.
- `--physics-mode exact` runs every pinned scheme (TEMPO v3.0.3, MYNN 44c0849, MMM a4baf7f) unless the run names a version.
- Pin bookkeeping: UGWP MPAS_20261104, RUC ef5c949, GFL 873b414, rrtmgp-data v1.9.2, with identical output.

Fixed:
- The memory check for an hrrrv5 run uses hrrrv5's own measured runs. A 30 km hrrrv5 globe was admitted with about 14 GB less than it uses, on one card and split over two or four.
- The kernel-pack and physics-resource downloads retry when the release host answers with a temporary error.
- Limited-area runs and hrrrv5 runs load three more kernels from the kernel pack instead of compiling them at every start.
- WOOF-physics runs load the Grell-Freitas kernel and the engine's other physics kernels from the kernel pack. The Grell-Freitas kernel compiled for about 90 s on one core at every run that uses it, so every move of a storm-following region waited that long.
- TEMPO picked over an WOOF-physics suite loads its kernels from the kernel pack. It compiled one engine kernel for about 30 s on one core at every run.
- Two `woof hex global` runs started together on an empty cache both get the mesh; one of them used to stop in its mesh stage.
- A microphysics scheme picked over an WOOF-physics suite (NSSL 2-moment, P3, TEMPO, also TEMPO beside Grell-Freitas) runs: it gets its extra fields in the start file, on every door, and NSSL draws its first reflectivity frame. On gpuwm 2.9.13 these picks stopped before the forecast; they need gpuwm 3.0.
- TEMPO picked over an WOOF-physics suite steps at 150 s or less. A default step above that takes 150 s, and a longer `--dt` stops before the run and says why. On the 120 km globe, TEMPO at the mesh's own 600 s step stopped within its first 40 minutes, and at 300 s its rain water was still growing after an hour; at 150 s it ran 6 h with bounded fields. MPAS-A suites that carry TEMPO (hrrrv5) run on the MPAS route and are not affected.
- `woof hex score` on a limited-area run scores only observations inside the region.
- In correct mode, IAU applies every scalar increment the analysis file carries (cloud, rain, ice and graupel numbers, graupel volume, aerosol numbers), not only the moist masses. Exact mode is unchanged.

## 0.4.15

Requires gpuwm>=2.9.12,<2.10.

Fixed:
- A 60 km global run with MPAS physics starts on a 16 GB card. It was refused for memory it does not use.
- Turning on SST, sea-ice or deep-soil updating with `--physics` no longer raises the memory price.
- Shortwave radiation holds about 720 MB less at the 60 km memory peak, with the same results.
- Killed runs no longer leave their stream handoff files in /dev/shm; the next run clears them.
- The GFS source carries its dated surface forecast, and S2S runs take SST and sea-ice forcing from any source that has one.
- S2S products and energy rows run on the MPAS-physics route.
- An interrupted analysis start is rebuilt on resume, and start-source receipts are written whole.
- Global mesh-plan receipts are standard JSON: a uniform globe wrote `Infinity`, which strict JSON readers refuse.

## 0.4.14

Requires gpuwm>=2.9.12,<2.10.

Fixed:
- Native MPAS restart files now restart the dynamics on every physics suite. On the default route a restart continues the run that wrote it exactly; with `--numerics strict` it reproduces native MPAS's own restart.
- Checkpoint resumes and same-mesh resumes of storm-following regions on MPAS physics continue the run exactly, on one card or several ranks.
- Several ranks on one card use much less memory: radiation work space is shared between them and the TEMPO tables are held once per card.
- Memory admission prices MPAS-physics runs from measurements on this release: one member at 120, 60 and 30 km, co-resident members, ranks sharing a card and one rank per card. Shapes that were refused before now start.
- Same-host ranks no longer grow their memory over a long run.
- `--samehost-ranks threads` runs again on both physics routes.
- Waves run on several ranks and cards with one wave model per run, give the same numbers run to run, keep their state in checkpoints, and runs on one host download the bathymetry once.
- WOOF-physics runs with waves start on the native global meshes.
- WOOF-physics runs write the 2-5 km updraft-helicity maximum and draw its maps, on one card, with local time stepping and on several cards.
- Variable-resolution globes with a coarse background draw their "around the core" map instead of stopping the run.
- A region run no longer stops now and then with "boundary intermediate digest differs".
- Runs with observation scoring turned off finish instead of stopping after the forecast.
- RUC land: frozen-soil runoff matches native MPAS in the last bit.
- Strict runs refuse TEMPO tables that are not the strict ones, and a named table folder must hold every table.
- Prebuilt kernel packs cover NVRTC 13.5.37 and 13.4.59, so a fresh `[gpu-cu13]` install starts without compiling. The RRFS MYNN-EDMF kernel compiles to the same machine code every time.
- `woof hex[gpu-cu12]` and `[gpu-cu13]` also install woof's extra of the same name, so the pair resolves one CUDA stack: on CUDA 13 that is NVRTC 13.4.92, the one this release was gated on (before, a fresh `[gpu-cu13]` install took NVRTC 13.5.37, which gpuwm 3.0 excludes).

## 0.4.13

Requires gpuwm>=2.9.12,<2.10.

Fixed:
- A fresh `woof hex[gpu-cu13]` install loads prebuilt kernels again instead of compiling them for about 100 s: the extra keeps NVRTC below 13.5, which has no pack yet. A 13.5 pack comes in a later release.
- `woof hex doctor` warns when the loaded NVRTC has no prebuilt kernel pack.
- Storm-following and setup-following runs write the 0-1/0-2/0-3 km updraft-helicity maxima on every segment.
- The README, the PyPI page and the manual state this release's version, engine range and kernel packs, and the PyPI page's links work.

## 0.4.12

Requires gpuwm>=2.9.12,<2.10.

New:
- Regional forecasts start from RRFS v1, with separate boundary sources and cycles.
- HRDPS and RDPS regional starts, with deep-soil and upper-air donors.
- Held-Suarez forcing on global meshes (`ra_lw_physics = 31`).
- Opt-in hurricane line: cyclone tracking, HAFS starts and storm-following regions.
- Opt-in wave forecasts, stations and maps.
- SPPT and SKEB stochastic physics, written from the published method (opt-in).
- 0-1, 0-2 and 0-3 km updraft helicity maxima in history and on maps.
- A 10 km global mesh, with parallel download and unpack.
- An experimental AMD (ROCm/HIP) backend, opt-in. CUDA stays the default.
- Storm-following regions run on several cards.

Changed:
- RRFS conus3km transport went from 93.4 ms to 71.5 ms.
- hrrrv5 with the storm scheme runs at 1.18x the MPAS-physics step, down from 1.60x, and its final pass went from 227 s to 31 s.
- Less device memory: 120 km MPAS 10.44 to 5.49 GiB, 60 km engine 14.95 to 11.96 GiB, an hrrrv5 region 6.49 to 2.86 GiB.
- Less host memory during mesh preparation and history writes.
- Sources are prepared and maps finished across more cores, and a region's source preparation runs beside mesh preparation. Sub-area end-of-run maps went from 81.5 s to 13.5 s.
- Process-rank device memory is priced from its measured whole-run peak, and low-memory forecasts are priced before launch.

Fixed:
- Four-card MPAS-physics runs start: the process transport links only the ranks that share halo words.
- 1 km runs split across cards no longer stop at random: each card checks only the columns it owns.
- rrfs_v2_dev and every column suite without Noah-MP run again, on one card or several.
- Reflectivity maps include the rain from active cumulus schemes.
- Coastal water cells take their sea temperature by the WPS SKINTEMP rule.
- A refused cap mesh is built at another centre and turned to the requested point.
- Stale run files are recovered, MPAS device memory is checked before startup, and waiting ranks stop when the forecast coordinator fails.
- Native restarts, checkpoints and history timing are preserved.
- Runs that write checkpoints no longer write zeros for boundary-layer height, surface heat fluxes and reflectivity in their first frame.
- `--dynamics KEY=VALUE` now reaches runs on the MPAS-physics route.
- A run resumed from a checkpoint keeps its maps' start time, and frames written past the checkpoint are set aside.
- The 96 -> 3 km zoom globe is priced from its measured peak, so restarts that fit are admitted.
- Fixes to source soil layers, static culls, field units and worker refusals.

NOAA GSL UGWP is now credited under its Apache-2.0 licence in NOTICE.

## 0.4.11

Requires gpuwm>=2.9.11,<2.10.

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
- No case the cold-soil remap cannot serve stops an init any more: a GFS start from a later forecast hour (init-source, regions and moving cases), a land model other than Noah (the WOOF-ArWen route's Noah-MP, RUC), a missing or late donor, a cycle before GFS v16, a GFS analysis from another land setup, an all-water region and the rest start from the source's interpolated soil, as before 0.4.10, and the init's log and receipt say why. `--soil-remap on` refuses instead.
- A limited-area region started from GFS (country and region doors, hourly boundaries) no longer stops at its first boundary hour ('this cold-soil donor row is analysis-only'): only the run's start takes GFS's cold-soil donor, as the regional door already did.
- RRFS v2 `hrrrv5` forecasts take `--stream-input invariant=FILE`: a registered input stream the run binds is read once at the start instead of being refused as inactive.
- RRFS v2 `hrrrv5` history draws maps again: its aerosol optical depth (`aod550`) declares CF units '1', so the converter no longer refuses it.
- `woof hex country --forecast-arg=--physics-suite=hrrrv5` no longer dies as its forecast starts (KeyError 'nscat'), from GFS or IFS.
- A physics-resource cache that lost RRFS or MPAS tables (a cleaned disk) fetches them again instead of failing on a missing table file.
- Map titles name the view and the spacing its picture shows ('Core zoom 1-4.1 km', 'Whole globe 1-129 km') instead of the whole mesh's range; `--view-label VIEW=TEXT` passes a caller's own words.
- RRFS Grell-Freitas-Li (`cu_grell_freitas_li`) runs on the GPU-resident route with no native library: it no longer stops at start ('init_convection for GFL / Tiedtke schemes has no ported path'), at its first step ('cached extent conflict for diag_physics.vcpool') or on its momentum carriers ('sub3d_rucuten_p is not allocated'), and its columns run in wide launches instead of 64 at a time.

## 0.4.10

Requires gpuwm>=2.9.10,<2.10.

New:
- RRFS-MPAS `hrrrv5` runs with no native library: its physics runs GPU-resident on the ported fork framework, and its history and diagnostics are written by `rw_mpas_stream` with the fork's stream contract.
- `woof hex init --physics-suite hrrrv5` (and every RRFS-MPAS suite) makes the initial condition and hourly boundaries the way RRFS does from GFS: the RRFS fork's init_atmosphere ported to Rust, with RUC's nine soil levels, the L60 grid, TEMPO aerosols, the fork static and the GSL drag statistics. Equal to the fork's own init on every stored value (docs/rrfs-init.md).
- `hrrrv5` runs the MPAS commit NOAA's RRFS v2 runs (RRFSx/MPAS-Model 44c0849): a 6 h x1.2562 forecast in strict numerics writes every history and diagnostics word equal to native 44c0849. `--physics-pin-set rrfs` keeps the previous pin.
- RRFS `hrrrv5` runs NSSL two-moment microphysics (`mp_nssl2m`) on the GPU; a 3 km step costs about 1.17x the TEMPO default.
- RRFS-SD smoke and dust run on the GPU on the default 44c0849 pin; a 3 km step with real fires costs about 1.12x the same run without smoke.
- RRFS at 3 km steps faster: TEMPO microphysics (176 -> 143 ms of a 3 km step) and the limited-area dycore's stencils and acoustic solve (about 12 % of the step), with every word unchanged in strict numerics.
- The RRFS init engines (`rw_mpas_init_v831`, `rw_mpas_gsl_oro`), which gpuwm 2.9.10 ships in its release bundle rather than its wheel, are staged automatically the first time an RRFS init needs them.
- Optional Icechunk output stores native forecast history as atomic Zarr v3 snapshots, locally or on S3 (`--output-format netcdf,icechunk` on every forecast door; needs gpuwm 2.9.10, whose writer is fetched on first use).

Changed:
- MPAS-physics history and diagnostics files are written lossless-compressed (NetCDF-4, the same values in a fraction of the disk), compressed on every core by the engine's `rw_mpas_stream`. Init, restart and boundary files keep their native format. `GPUWM_HISTORY_COMPRESSION=none` writes them as before.
- Regional GFS cold starts (zoom globe, limited area, storm-following) default to correct numerics with the `hrrr-like-aerosol` profile: aerosol-aware Thompson with `nwfa2d` initialisation, the corrected MYNN cloud merge, cold-start aerosol emissions, soil moisture mapped by relative wetness and the land carry. Forecasts on those runs change; the step costs about 1.03x the plain suite's.
- A hex region driven by a global source (GFS, IFS, GEFS) takes its boundaries from every lead the source posts (GFS hourly, IFS and GEFS 3-hourly) instead of every 6 h, can start at any posted lead, and `--boundary-interval-hours` picks a coarser step. Boundary files are written in parallel.
- GEFS member inits and boundaries download only the records the init reads, about 34 MB instead of 106-118 MB per member and lead.

Fixed:
- Limited-area runs (RRFS `hrrrv5`, hex limited-area regions) take less card time per step: the vertically implicit acoustic solve runs level-parallel around its column sweeps and the acoustic flux sums read the mesh tables in thread order. Every word is unchanged in fast and strict numerics.
- A local-time-stepping run on MPAS physics records every class step, the coordinator's work between them and the loop's own measured rate in `native-runtime-timings.json`; it used to hold only the class that finished last, so most of each step was unlabelled.
- A local-time-stepping profile run (`WOOF_HEX_LTS_DEVICE_SPANS=1`) records each class step's device time. The default receipt's per-class rows are host wall, which on one card includes the device work queued ahead of a class, and the receipt now says so.
- The zoom-globe receipt (`global-vr-receipt.json`) names the physics mode the forecast ran. A run that named a suite but no mode ran correct mode, and the receipt said exact.
- A hex run on MPAS-A physics started from HRRR (or any first guess that carries hydrometeors) starts from its cloud, rain, ice, snow and graupel and HRRR's own numbers, as the hrrrv5 route does, instead of from no condensate (no echo at hour zero); `woof hex init --first-guess-hydrometeors no`, and strict numerics, keep MPAS-A v8.4.2 case 7's dry start. Needs gpuwm 2.9.10's `rw_mpas_init`; an older engine is a named gap.
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
- Hex with Regional nest maps are headlined `nest 3 km` instead of `d01 3 km` (with gpuwm 2.9.10).
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
- One-card local time stepping writes history in a child process off the step thread and no longer synchronises the card at every class step (MPAS physics 62.1 -> 58.0 s per model hour, ArWen physics 63.3 -> 57.8; output unchanged).
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
- The engine programs the package bundles are rebuilt from gpuwm 2.9.10: maps of a history that leaves the terrain unset reduce sea-level pressure from the run's initial terrain (an RRFS 3 km frame showed a tangle of isobars around an 868 hPa low), sea-level pressure is smoothed over high ground, a limited-area mesh's boundary rings are left blank, composite reflectivity includes parameterised convection, map frames are written lossless-compressed (a 3 km frame 1.23 GB -> 306 MB), history frames are written about 4x faster with the same bytes, and RRFS boundary files carry the 2.9.10 numbers.

## 0.4.9

Requires gpuwm>=2.9.7,<2.10.

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

Requires gpuwm>=2.9.7,<2.10.

Fixed:
- A global ensemble retries a member stage that fails once, and a member that still fails is recorded as missing while the other members finish.
- MPAS-physics runs that update SST, sea ice, greenness and deep soil from a surface series keep that update on the GPU instead of asking for the native MPAS library.
- A test now guards the Grell-Freitas scratch fix from 0.4.6: no GF output may read a workspace word before writing it.

## 0.4.7

Requires gpuwm>=2.9.7,<2.10.

Fixed:
- Storm-following runs no longer refuse a move with "FP32 moving integral cannot meet its conservation tolerance".
- A cycle that is not posted yet is refused naming every address it tried (NOAA first) and when it posts; archives that lag a cycle are skipped for younger cycles.
- A start file without a 2 m analysis (IFS starts carry q2 as NaN and t2m as 0 K) takes its 2 m fields from the first physics call, so ensemble products no longer fail with "history q2 is not finite".
- A run's exit no longer waits on an idle kernel-compile worker that ignored the stop signal.

## 0.4.6

Requires gpuwm>=2.9.7,<2.10.

New:
- MPAS physics is the default on every door: a run that names no physics runs MPAS-A's own `convection_permitting` suite on the GPU (the suite follows MPAS-A practice by mesh, `data/profiles/physics-default.v1.json`). ArWen physics stays one flag away: `--physics-suite arwen_mpas_column` (`--physics-row arwen` on `global`).
- The cycle door takes an MPAS physics forecast as its coarse and parent stream.
- Kernel packs (`kernel-packs-v0.4.6`, CUDA 12.9, 13.0 and 13.4) carry the MPAS physics kernels, so a new box starts MPAS physics runs without compiling them.
- Storm- and setup-following runs start from a premade mesh rotated to the storm (`moving template` builds one), and the static builder maps geography to global meshes about twice as fast, with identical output.

Fixed:
- MPAS physics runs with local time stepping no longer fail at the first step on a new box (Noah kernel cache).
- `forecast --receipt` writes its receipt on the MPAS physics route.
- Storm-following and other MPAS physics runs no longer stop with a non-finite state after a few hours (Grell-Freitas shallow convection read an unset cloud-top value).
- Local time stepping on two or more cards gives the same results as one card, on both physics routes; MPAS physics runs it on several cards.

## 0.4.5

Requires gpuwm>=2.9.7,<2.10.

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
- The WOOF-ArWen route runs on gpuwm 2.9.7.
- Moving runs on column suites (HRRR shape) and over sea ice move past their first move, and a new member's sea ice brings its own albedo, emissivity and ice fraction.

## 0.4.4

Requires gpuwm>=2.9.4,<2.10.

New:
- Global and variable-resolution forecasts draw their first maps the moment the forecast writes its first frame, instead of after the whole forecast.
- 1 km limited-area runs start faster: a regional target of a global parent reads only the parent cells around it, boundary frames are remapped concurrently, and a premade variable-resolution mesh and its static are fetched instead of rebuilt.
- First maps open on 2 m temperature, 10 m wind and CAPE, and on reflectivity from its first echo.

Fixed: every hex map carries a labelled latitude and longitude graticule.

Fixed: `woof hex global`, `mesh-plan`, `regional`, `country`, `moving` and `forecast` take `--model-label` and `--brand` and pass them to the point soundings and meteograms; with gpuwm 2.9.6 or later the brand is drawn in bold at the top right.

## 0.4.3

Requires gpuwm>=2.9.4,<2.10.

Fixed: global and subseasonal forecasts with sea ice no longer stop at their first step on gpuwm 2.9.4, and `woof hex s2s` runs of any length finish their energy products.

## 0.4.2

Requires gpuwm>=2.9.4,<2.10.

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

Requires gpuwm>=2.9.2,<2.10.

New:
- MPAS-A v8.4.2 dycore, GPU physics and LES closures, with four acoustic substeps and a native-free framework. ArWen physics remains the default.
- MPAS physics ports: WSM6, Thompson, Kain-Fritsch, Tiedtke, Grell-Freitas, revised MM5, MYNN, YSU, Noah, sea ice, RRTMG and CAM.
- Real-data and idealized initialization, vertical grids, orographic statics, aerosol climatology, SST and sea-ice updates, IAU, limited-area boundaries, streams, restart and diagnostics.
- Correct mode for MPAS and ArWen physics, with named corrections, switches and receipt records.
- Global and country workflows, source-table initialization, analysis bundles, mesh remapping, cycling and subseasonal ensembles.
- Forecast products, soundings, meteograms, NetCDF export and observation scorecards, including IGRA2 radiosondes for past cycles.
- Ensemble and preset preparation, member queues, shared static caches, a pinned preparation store and suite-aware soil and aerosol initialization.
- Storm-following regions with balanced conservative remapping, physics-state carry, restart and same-host multi-card execution.
- MPAS physics tables fetched from pinned data-only assets in `physics-resources-v0.4.1`.
- CUDA 12 and CUDA 13 installation extras, driver-aware diagnostics, bundled helpers and a manylinux_2_28 wheel.
- Asynchronous history output, cached input digests, parallel host preparation, fine-terrain sampling and diagnostic export.
- Kernel packs fetched on demand from `kernel-packs-v0.4.1`; `woof hex fetch-assets --from DIR` and `WOOF_HEX_ASSETS_FROM` support offline staging.

Fixed:
- MPAS shortwave shape specialization and YSU memory probes support gpuwm 2.9.3.
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

Requires gpuwm 2.8.0 (gpuwm>=2.8.0,<2.8.1), as 0.3.2 did.

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
  catalog rows" on gpuwm 2.8.0, whose renderer prints six fields per
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
- The physics seam re-pins to gpuwm 2.8.0: gpuwm>=2.8.0,<2.8.1. Eight of
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
- The physics seam re-pins to gpuwm 2.7.4. The sixteen-file manifest
  carries the published 2.7.4 wheel's bytes (tag ed957997, whose tree
  agrees with all three platform wheels on every pinned path) and the
  derived range is gpuwm>=2.7.4,<2.7.5. Two of the sixteen paths moved
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
  pip no longer resolves 2.7.3+x.1 inside gpuwm>=2.7.4,<2.7.5 and a row
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
  gpuwm>=2.7.3,<2.7.4); what decides whether this port runs on it is now
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
- Requires gpuwm 2.7.3 and refuses every earlier engine, 2.6.4 and 2.6.5
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
- Requires gpuwm 2.6.4 or 2.6.5. Eight of the sixteen pinned seam files
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
- Requires gpuwm 2.6.1. Three of the sixteen pinned seam files moved at
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
- A published comment the release tooling had mangled reads correctly again,
  and the tooling no longer doubles an article in front of a substituted
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
- Requires gpuwm 2.6.0: the engine release carrying the repaired
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
  refuses.** The dependency was `gpuwm>=2.5.5` with no ceiling, so pip took
  the newest published engine (at the time 2.5.7) and the forecast lane
  then refused at launch with two SHA-256 digests and no version number while
  `woof hex doctor` reported the estate healthy and exited 0. A green
  install and a dead run, with no route out by reading. Three changes, all
  default-on: the declared range is now **`gpuwm>=2.5.8,<2.5.9`**, derived by
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
  `docs/mpas-seam.md`, which no wheel carried; gpuwm 2.5.8 ships it inside the
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
  to refused on the DEFAULT row on a 10 GiB RTX 3080, which the 08-25
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
  forecast modules and in the GWDO guards, and the sealed WOOF-ArWen constructor
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
  architecture boundary rather than a property of one stack. MEASURED on an
  RTX 3080 (sm_86, NVRTC 13.0.48 `CL-36260728`, CUDA driver 13030):
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
  10 GiB RTX 3080, where the superseded row refused it. No 12 GiB card
  was measured: the 12 GiB tier figure ships as a DECLARED
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
  the runtime; the pool refuses their names and says where they belong.
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
- The WOOF-ArWen seam pin leaves the pin-only lineage: `ARWEN_BUILD_COMMIT` moves
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
  dewpoint-from-mixing-ratio on Q2/PSFC, transcribed exactly). The WOOF-ArWen
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
  The WOOF-ArWen vegetation tables end at category 20, so before the fold any
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
  it lives outside both the MPAS atmosphere and the WOOF-ArWen backend
  restart payload. The F030 checkpoint never captured it, so every
  restored run re-entered step 16 with zero forcing lanes while the
  unbroken run fed the real step-15 pair, and the step-16 identity gate
  failed deterministically on every arm (the step-16 restart divergence, 5/5 red on the reference machine, red
  since the forcing changes landed). Checkpoint schema v3 downloads the
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

Requires gpuwm 2.5.5 or newer for the seam bytes and the bundled engine
binaries.
