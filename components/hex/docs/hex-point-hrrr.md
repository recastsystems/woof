# A fine hex core at any CONUS point, from HRRR

`woof hex regional` builds and culls the mesh, fetches HRRR, writes the
intermediates, initial state and hourly boundaries, checks the regional
contract, runs the forecast and invokes the products door:

```sh
woof hex regional --point LAT,LON --fine-dx-m 937.5 --radius-km 100 \
    --source hrrr --cycle YYYY-MM-DDTHH --hours 3 --work-dir RUN \
    --geog /path/to/WPS_GEOG
```

The door passes its mesh registry and contract evidence internally. There is
no environment variable to set by hand. `RUN/regional-receipt.json` records
start, finish, wall time and outcome for every stage, the mesh and product
counts. Preparation stages can reuse their files; the forecast destination
must be fresh. `--card` selects a measured sizing row when supplied.

The source is a `limited_area` row in `data/init-sources.v1.json`; run shapes
are `limited_area` rows in `data/countries.v1.json`. A new configuration class
earns its runtime mint through two independent strict dry forecasts with
finite, byte-identical fields, then runs the contract deck on its own rings.
Each mint samples up to 15 complete steps within the requested forecast
window. A one-hour forecast at 600 s therefore mints six steps; both receipts
record that window and must contain all six fields at every step, including
the initial state. It does not require parent boundaries past the forecast.
This is a reproducibility check, not a weather-skill certificate.

Measured on 2026-10-20 at 36.5 N, 97.5 W, HRRR 2024-05-06 18Z, three hours:
43,360 cells, 55 levels, 5 s timestep, 2,160 steps and four history frames.
The forecast stage completed in 1,908.79 s on a shared RTX PRO 6000 Blackwell
Server Edition with CUDA graphs disabled. The installed hex 0.4.1 native
binaries and woof 2.9.3 dependencies were used with this branch's Python
overlay. Maps rendered: 70, with two expected hour-zero skips and no failures;
the view is `refined-core`, with no global view.

Sampled forecast VRAM peaked at 6,508 MiB under a 9,000 MiB pool reservation.
Stage walls were mesh 1,246.24 s, intermediate 328.85 s, init 3.44 s,
boundaries 9.11 s, contract 85.19 s and forecast 1,908.79 s. These are shared
host timings. The original command hung in the engine's `rw_hex_points`
bridge after the forecast passed. The engine fix on
`lane/improve-32b-hex-points-hang` (for 2.9.4) masks missing surface wind
values and bounds raster work to the image canvas. The exact retained bridge
invocation then completed in 0.53 s. Products-only recovery on
the retained forecast completed in 13.81 s with 70 maps and
6 point pictures, no failures and no global view. The merged regional
receipt preserves the failed attempt and measured recovery separately. This
was not an uninterrupted successful command wall, and the forecast was not
repeated. F000 surface winds are unavailable, so point wind diagnostics show
gaps or n/a; later frames have valid winds.

The component doors remain available for inspecting each stage. `--spec`
hands the generator an authored spec unchanged; `--point` writes one and
records it. Their mesh and HRRR mechanics follow.

## `mesh-plan --point`: the ladder is data

The spec is a pure function of six numbers (`woof.hex.mesh_point.ladder_spec`):
the point, the fine spacing, the core radius, the global background, the ramp
factor and the ring factor. It is the registered `v0.9.120.110533` recipe
restated -- nested caps on a 120 km background, each rung halving the spacing,
each ramp eighteen times its own spacing, each cap three ramps of the next
finer rung outside it -- and at 35.0 N 97.0 W, 937.5 m, 100 km it reproduces
the spec of record to the digit (tested). Moving the core moves the centre of
every cap and nothing else.

A fine spacing that is not a rung of the halving ladder is refused naming the
rungs on either side, because the generator snaps a request onto the nearest
rung ALWAYS FINER: a 900 m request would silently build 468.75 m and cost four
times the cells the card was priced for.

The plan prices two things and admits on the second, exactly as the swath
layer does:

- the GLOBAL parent, through the generator's own `--dry-run` with the gates a
  build applies (`woof.hex.swath.sizing`);
- the limited-area CULL, an area integral over the ladder's own spacing
  profile at the generator's attained fine spacing (`basis: area_integral`,
  a bound the cull receipt replaces with the count), priced on the admission
  surface's limited-area row for the named card with that row's own margin
  (`woof.hex.device_admission`). The verdict names the cells the card holds
  and the largest core radius that fits at this spacing and pad.

`--card` is a row in `woof.hex.mesh_point.CARD_ALIASES` joined to a measured
row of the admission surface; a card nobody measured is refused rather than
given another card's number.

## `--generate`: build, admit, register, cull

With `--generate` the door builds the pair (`rw_mpas_mesh`, then
`rw_mpas_static` -- the static builder joins the engine ladder as
`woof.hex.engines.STATIC`), measures the pair's own admissions before
registering anything (dual edges, cell coordination, Courant at the largest
anchored timestep the mesh's own `min(dcEdge)` admits), writes the RUNTIME ROW,
cuts the cull at the shipped 1.35x pad, admits and registers that too, and
with `--vertical-spec` mints the parent's native-free vertical artifact and
culls it.

**Why the vertical is minted on the parent and culled.** `woof hex init`
refuses a limited-area grid in native-free mode, correctly: the closed-sphere
vertical authority does not invent exterior state. A global init needs global
meteorology and HRRR covers only its own domain. Minting the vertical on the
GLOBAL parent (grid + static + spec, no meteorology) and culling the artifact
with the same region as the grid and static gives the cull a vertical that IS
the parent's, cell for cell; the culled artifact is then handed to
`woof hex init --capsule/--reference` with the regional intermediate.

The receipt is `<stem>.point-generate.json` (`gpuwm-hex.point-generate/v1`)
and its `next` list is the rest of the chain with every path filled in.

## Runtime rows: `mesh-rows.json` beside the mesh

`woof hex forecast` resolves `--mesh` against `src/hexcore/drivers/mpas_mesh_binding.py`
and refuses a name it does not hold. A mesh generated at an arbitrary point
cannot be a row in that file, and a checkout edit per point would be the
hand-written-row-per-swath defect the cascade retired. `woof.hex.mesh_rows`
is the answer: the door writes a `gpuwm-hex.mesh-rows/v1` document beside the
mesh, and the registry module applies it -- before the cascade rows, so a
cascade may cull a generated parent -- when `$WOOF_HEX_MESH_ROWS` names it,
returning the mapping unchanged on every ordinary run.

What a registry row buys is kept: the bytes are pinned by count and SHA-256
and re-hashed at bind; Courant, dual-edge, cell-coordination and regional
admissions are measurements of the files at bind; the timestep is refused
unless an anchor covers the configuration; the regional opener re-measures
the class key and requires a contract deck on THESE rings. What stands in for
a person reading the row is the door's own admission pass, recorded in the
row. Refused by name: a runtime row that shadows a shipped row; a cull row
whose parent is neither in the file nor registered; a row whose bytes moved;
a cull row with no `lbc_source`; a document of another schema.

## `intermediate`: HRRR onto the lat-lon record the engines read

The gap, measured in `docs/source-matrix.md`: the engine's intermediate writer
refuses to mint a projected grid as `iproj 0`, and `rw_mpas_init` /
`rw_mpas_lbc` invert projection code 0 only. This door resamples the
projected product onto a regular 0.025-degree lat-lon box over the cull (cut
radius plus the boundary halo plus a 30 km margin) and writes the WPS
version-5 records the engines already read. Everything with numbers in it is
the engine's own, driven, not re-implemented:

- decode: `hrrr_grib2_bridge`, resolved through `woof.bridges`, over the
  `wrfnat` + soil pair `woof fetch --source hrrr` writes, on its
  `--series-workers` route (`HOUR<TAB>WRFNAT<TAB>SOIL` rows), into the native
  window `woof.ingest.hrrr` maps;
- regrid: the engine's projected-source plan (`_ProjectedCpuPlan`, WPS's
  overlapping-parabolic operator with FP64 donor selection, on the engine's
  CPU bridge), aimed at the lat-lon target; winds rotated to the earth basis
  on the source grid with the engine's own rotation first, because an
  `iproj 0` record is earth-relative by definition;
- soil: the engine's node-to-Noah-layer interpolation over its own HRRR
  depth nodes, moved by nearest neighbour;
- write: the inverse of this tree's own frozen reader
  (`woof.hex.wps_intermediate`); every file is read back through that reader
  before the receipt is signed, and a slab carrying a non-finite value is
  refused.

Fifty hybrid levels are carried as level-indexed slabs with the 3-D
`PRESSURE` beside them (the init engine's model-level branch, the ERA5-ML
convention), plus the 200100.0 surface level, so `--nfglevels 51
--nfgsoillevels 4 --use-spechumd yes --extrap-airtemp constant`. Hydrometeors
are not carried, although the engine's window decodes them: the init stream
has slots for cloud water and rain (`qc`, `qr`), but the init and boundary
writers read no hydrometeor record from an intermediate file and write both
as zero, and it has no slot for cloud ice, snow or graupel, so hour zero has
no ice (README, limited-area lane).

The source is a ROW (`woof.hex.hrrr_intermediate.SOURCE_ROWS`): decoder key,
loader module, file names, field map, soil nodes, level count. Nothing below
the table branches on the string `hrrr` (tested); a second projected source
with an engine decoder and loader is a second row.

## `lbc`: one boundary file per intermediate valid time

`rw_mpas_lbc` on its wps-intermediate route over the hourly intermediates,
each admitted by the valid time in its own header, never its file name. A
series that does not cover `--start-time..--stop-time` at both ends, or that
is unevenly spaced, is refused by name: a boundary series with a missing end
freezes the boundary there without saying so.

The binary resolves through the engine ladder (`woof.hex.engines.LBC`):
`--lbc-exe`, `$WOOF_HEX_RW_MPAS_LBC`, `$RW_MPAS_LBC`, `$WOOF_RW_MPAS_LBC`,
woof's bridge directories (`libexec/bridges` beside the installed package and
`~/.woof/bridges`, where `woof fetch-bridges` stages the pinned engine's
bundle, which carries it), then `PATH`. `woof hex doctor` reports it beside the other
engines.

## The init carries its own clock

`rw_mpas_init` writes the init's global attributes by copying the
capsule's and does not stamp `config_start_time` itself. A native capsule
carries it from its namelist; the constructed route writes it into the
vertical artifact; the vertical `--generate` culls at mesh time is minted
before any start time exists and carries neither. The forecast driver
asserts `--start-time` against that attribute and refuses an init without
one, so the first point cull reached the card, spent seven minutes on the
contract deck, and stopped at the forecast opener. `woof hex init` now
stamps every run declaration the capsule copy left absent
(`woof.hex.init_door.stamp_run_declarations`, the same nine `config_*`
attributes the constructed route writes, encoded by the one encoder both
share) and records `run_declarations.stamped` and
`run_declarations.already_present` in the init receipt. Attributes the
capsule carried are left byte for byte, so the native x4 route is untouched;
a carried clock that disagrees with `--start-time` is reported on stderr and
left for the forecast door's own refusal, because the cascade's delayed
start moves it on purpose.

## The same-ladder class band

The regional forecast opener measures a class key off the run (zone width,
column count, finest edge, timestep, kernel set) and refuses a configuration
no forecast mint covers. Until 0.3.0 the finest edge compared exact to the
millimetre, which was protection against a printed number, not real drift: two
culls of one parent carry `min(dcEdge)` in identical bits. A parent generated
at another point from the same ladder spec delivers the same rung and a finest
edge that differs by relaxation noise, so the two sub-kilometre classes
(`graded-869m-dt5-z7`, `graded-711m-dt5-z7`) now declare a five per cent band
(`SAME_LADDER_FINEST_EDGE_BAND`), which cannot reach from one to the other
(18 % apart) and which the Courant admission re-measures against the cull's
own `dcEdge` regardless. Exact matches are tried first. The contract deck on
the cull's own rings is still required and still run; nothing about the
geometry half is relaxed.

## Measured, 2026-09-13

Point 41.95 N 94.79 W (central Iowa), HRRR cycle 2026-09-13 15Z, f00 to
f03, an RTX 5090 (32,607 MiB, sm_120), woof 2.7.3 from PyPI with its
2.7.3 bridge bundle, woof hex 0.3.0. Every number below is copied from a
run receipt.

The mesh (`mesh-plan --point 41.95,-94.79 --fine-dx-m 937.5 --radius-km 100
--card 32gb --generate`, 276 s wall on the same machine's CPUs):

- Ladder 120 / 60 / 30 / 15 / 7.5 / 3.75 / 1.875 / 0.9375 km, transition
  factor 18, fine core radius 100 km, fine flat radius 83.1 km, cut radius
  135 km (pad scale 1.35), spacing at the cut 2.88 km, halo 20.2 km.
- Parent `p0.9375.120.112246.n41.95w94.79`: 112,246 cells, 336,732 edges;
  `rw_mpas_mesh` 90.0 s, `rw_mpas_static` 100.6 s; min `dcEdge` 841.324 m,
  min `dvEdge/dcEdge` 0.0402 (floor 0.02), coordination {5: 2000, 6: 108260,
  7: 1984, 8: 2}, dt 5 s against a 6.058 s Courant limit (1.21x).
- Cull `q0.9375.120.43884.n41.95w94.79`: 43,884 cells (43,421 predicted,
  1.1 % over), 131,956 edges, seven boundary rings, attained fine spacing
  0.93751 km; the same finest edge and dt; registered as the second runtime
  row in `mesh-rows.json` with `lbc_source` naming the boundary directory.
- Priced for the 32 GB card: 11,827.9 MiB predicted for the cull on the
  measured limited-area row (96,582 B per cell over a 663 MB core), margin
  1,756.8 MiB, 248,102 cells would fit, so the largest fine core this card
  admits at 937.5 m is 240 km in radius.

The state (`intermediate`, `init`, `lbc`, all CPU):

- `intermediate --source hrrr --hours 0-3 --workers 4`: `hrrr_grib2_bridge`
  decoded the four `wrfnat` + `soil` pairs in 17.6 s onto the cull's lat-lon
  box (rows 908 to 1043, columns 591 to 726 of the HRRR grid); each WPS file
  holds 318 records (51 levels of TT, UU, VV, SPECHUMD; 50 of GHT and
  PRESSURE; the surface and soil set) in 31.2 MB, regridded in 0.08 s.
- `init`: `rw_mpas_init` 1.44 s, 317 met records used, 55 levels; the door
  stamped all nine run declarations onto the init (`run_declarations.stamped`
  in its receipt; `already_present` empty), which is what the section above
  is about.
- `lbc`: four boundary files (15Z to 18Z, 87.3 MB each) in 2.08 s through
  `rw_mpas_lbc` resolved from the venv's own bridge directory.

The card (both passes; 23.3 minutes of card time in total):

- Contract deck on the cull's own rings (`run_cuda_regional_contract.py
  --class-id graded-869m-dt5-z7`): 8 decks, 8 passed, 22 of 22 regional
  kernels covered, every deck bitwise, every control with teeth, dual run
  identical; 422 s, peak 1,026 MiB.
- `forecast --hours 3 --history-every-minutes 60 --lbc-dir ...`: admitted on
  the measured limited-area row (11,874.3 MiB predicted, 31,642.6 MiB free,
  margin 1,756.8 MiB), convection off by resolution (841 m against the 3 km
  threshold), surface/PBL every step (the welded cadence, 720 calls an hour);
  2,160 steps of 5 s in 721.3 s of integration, **0.334 s per step**, 912.2 s
  driver execution, 943.6 s door wall including the pinned-source
  verification of 23 modules; four history frames; status passed. Peak
  memory on the card 9,485 MiB (nvidia-smi at 5 s cadence), utilisation
  75 to 97 %. Load average at launch 0.34 with nothing else on the node's
  CPUs; another job had held the card for 420 s before this pass.
- The first pass stopped at the forecast opener on the clockless init after
  the deck had run; the deck's receipt pins the boundary mask, the cell count
  and the kernel set, none of which the remade init changes, so the second
  pass reused it (`SKIP_DECK=1` in the chain script).

The picture (`render --products composite_reflectivity,2m_temperature_10m_winds`
on the 18Z frame): `rw_mpas_convert` framed the mesh window at 233 x 235
columns (924 m, mean nearest-neighbour 0.51 km) in 0.81 s and `rw_wrfbatch`
rendered both panels; 5 s in all. `theta_e_2m_10m_winds` is in the renderer's
catalog but not realised by the wrfout import lane for this history, and the
door refuses it by name before drawing anything.


## The surface/PBL cadence, measured 2026-09-13

Same point mesh, same HRRR 15Z init and boundary files (one `intermediate`,
`init` and `lbc` pass, digests in the run receipts), an RTX 5090
with no other process on the card during any arm (a co-tenant job had
finished before the first arm; load average 1.0 to 1.3 throughout), one
session, one hour (720 steps of 5 s), history every 60 minutes. Five arms
back to back in one session: the untouched 0.3.0 tree
in the shared venv at the welded default, then this tree at 120, 60 and
30 s, then this tree at the welded default. Card time 1,656 s (688 s for
the first two arms, 968 s for the last three; a fresh-scratch refusal split
the chain once).

**The default is unchanged, byte for byte.** This tree at `auto` wrote both
history frames byte-identical to the untouched tree's (15Z
`85cb8196...`, 16Z `4ed8001f...`, the same two digests the 2026-09-13 profile
measurement recorded), at 0.3156 s per step after the first against the untouched
tree's 0.3148 (median composite step 0.2994 s in both).

**The hold is real, and the receipt proves it.** At 120 s the seam
reported the surface/PBL stack due on steps 1, 24, 48, ... -- 31 of 720,
689 held -- with the engine's own `sfclay`, `noah` and `ysu` counters all
reading 31 at the last step and radiation due 6 times, exactly the count
the declared cadence predicts (`physics.cadence` in the receipt:
`consistent: true`). 60 s: 61 due, 659 held. 30 s: 121 due, 599 held.

**What it buys: about five per cent per step, the same at every cadence.**

| arm | s/step after the first | median composite step | ratio (median) | due / held | peak card memory |
|---|---|---|---|---|---|
| every step, untouched tree | 0.3148 | 0.2994 | 1.000 | 720 / 0 | 8,914 MiB |
| every step, this tree | 0.3156 | 0.2994 | 1.000 | 720 / 0 | 8,914 MiB |
| 30 s (6 steps) | 0.3022 | 0.2845 | 0.950 | 121 / 599 | 8,914 MiB |
| 60 s (12 steps) | 0.3005 | 0.2842 | 0.949 | 61 / 659 | 8,914 MiB |
| 120 s (24 steps) | 0.3013 | 0.2844 | 0.950 | 31 / 689 | 9,421 MiB |

The saving is 14.9 to 15.2 ms per step and does not grow past 30 s: what a
held step skips is the surface layer, Noah-MP and YSU (the profile's 16 to
19 ms of Noah-MP host time is most of it), and what it keeps -- the
phase-one prep, the 447 MB rollback snapshot, GWDO, WSM6, the commit, the
host health gate and the dycore -- is 95 per cent of the step. The 120 s
arm's larger peak memory and its 356.5 s door wall (against 318 to 330 s
for the others) are its first step: 34.3 s of cold NVRTC compilation into a
fresh kernel cache, which the other arms of this tree then hit; the
per-step numbers above exclude the first step.

**What it costs: a different forecast, everywhere.** Differences at 16Z
against the every-step arm (same init, same card, same session; the
every-step arm's own dual-run noise floor is zero):

| field | 30 s max / RMS | 60 s max / RMS | 120 s max / RMS | 2.6.5 to 2.7.3 seam move (max) |
|---|---|---|---|---|
| theta, K | 1.245 / 0.0235 | 1.839 / 0.0340 | 2.949 / 0.0540 | 0.139 |
| qv, kg/kg | 1.04e-3 / 1.3e-5 | 1.20e-3 / 2.0e-5 | 1.80e-3 / 3.2e-5 | 3.7e-4 |
| u, m/s | 1.469 / 0.0262 | 2.487 / 0.0469 | 5.038 / 0.0859 | |
| v, m/s | 1.074 / 0.0238 | 1.820 / 0.0412 | 3.040 / 0.0737 | |
| w, m/s | 1.563 / 0.0204 | 3.001 / 0.0382 | 4.692 / 0.0705 | |
| T2, K | 0.456 / 0.0771 | 0.507 / 0.0929 | 0.657 / 0.1074 | |
| Q2, kg/kg | 1.66e-4 / 1.4e-5 | 1.85e-4 / 1.9e-5 | 2.97e-4 / 2.8e-5 | |
| U10, m/s | 0.384 / 0.0382 | 0.439 / 0.0558 | 0.530 / 0.0855 | |
| rain, mm | 0.045 / 0.0029 | 0.100 / 0.0056 | 0.220 / 0.0110 | |
| sensible heat flux, W m-2 | 58.5 / 7.3 | 62.4 / 10.1 | 95.4 / 13.0 | |
| PBL height, m | 420 / 19 | 480 / 26 | 417 / 32 | |

Every surface cell differs (43,884 of 43,884 for T2, Q2, U10 and the
fluxes) and 98 per cent of the theta and qv column points do; the maxima sit
in the convective cells (level 19 to 29, where the reference w reaches
23 m/s) and the surface fields move because the surface layer and Noah-MP
integrate with the cadence as their step. The yardstick the campaign named
is the 2.6.5 to 2.7.3 engine seam movement recorded in
`declared-divergences.md` -- a column-seam number, eight columns over forty
minutes, theta up to 0.139 K and qv up to 3.7e-4 kg/kg -- and the smallest
cadence exceeds it nine times over in theta and three times in qv at the
maximum; the card's own contribution, the other yardstick, is zero.

**The decision.** No cadence qualifies: none keeps its one-hour differences
below either yardstick, and the speed it buys is the same five per cent at
30 s as at 120 s. The knob ships selectable with the every-step default,
which is what this tree does. If a held cadence is ever graded against
observations (MRMS, ASOS; the referee `docs/obs-referee.md` names), 30 s is
the arm to grade: the same saving as 120 s with the smallest differences.
The time in a step is elsewhere -- the host health gate (a fifth of it),
the per-step rollback snapshot and the double-launched recovery kernels
the profile named -- and that is where the next changes should look.

The reflectivity panels for the every-step arm and the 120 s arm
(`composite_reflectivity` and `2m_temperature_10m_winds` on the 16Z frame,
through `woof hex render`) are in the evidence folder beside this section's
receipts; they are model results, not evidence about the atmosphere.

## The step's host path, measured 2026-09-14

The same cull, the same init and boundaries, one hour (720 steps of 5 s),
two arms back to back in one session on an RTX 5090, the
untouched 0.3.0 tree first and the hostpath tree second, the card otherwise idle (no other GPU process during any arm; load average 1.05 at the second mutex hold, 1.5 at the first):

- Contract deck on the cull's own rings at the hostpath tree's kernel-set
  digest (three host sources moved, no CUDA source string; the regional
  translation unit's source digest in the receipt is the one the 0.3.0
  deck recorded): 8 of 8 decks bitwise, 22 of 22 kernels, dual run
  identical, every control with teeth; 430 s.
- Untouched tree: door 328.0 s, integration 228.7 s,
  0.3143 s per step after the first, the health gate
  56.2 s of it, peak 8,914 MiB, utilisation
  77.7 % mean.
- Hostpath tree: door 258.7 s, integration 215.3 s,
  0.2957 s per step after the first (1.06x), the health gate
  0.2 s, peak 8,914 MiB, utilisation 74.2 % mean.
  Run a second time in its own process: 214.3 s of integration and
  the same two frames byte for byte.
- Both history frames (15Z, 16Z) sha256-identical between the two trees,
  and the 720 `step_health` envelopes identical entry for entry.

Item by item at this cull's shapes (host milliseconds per step, a
microbenchmark on the same card in the same session, median of twenty,
each call closed by a stream synchronize): the admission 47.2 to 0.02;
the recovery launches 56.3 to 0.07 (and half of their device work gone
with the second launch); the health gate 78.4 to 0.20, the receipt it
returns identical; the density validation 0.15 to 0.01; the
recovered-state validation 1.19 to 0.04; the garbage scrub 15.6 to 5.9.
The loop as a user waits for it (integration, gate and history) went
from 0.3973 to 0.3011 s per step, 1.32x. The pictures and every receipt are
in the run receipts.

## The integrated performance tree, measured 2026-09-14

The integrated performance tree is the 0.3.0 tree with the cadence, host-path
and local-time-stepping changes merged in that order (the kernels change was not
merged at this
measurement: its per-kernel capture-and-replay instrument reported 49 kernels
compared and one, `transport_standard_finish_regional_v841`, not bitwise; the
section after this one is that change's landing, once the replay was bitwise
for all 49). At this measurement the merged tree's regional kernel-set
digest was the one the host-path change re-minted (`191af5f9...`), computed on the
merged sources and equal to `MINTED_KERNEL_SET_SHA256`; the four frozen-source
digests the two changes moved (`dt_admission.py`, `pbl_cadence.py` from cadence;
`cuda_driver.py`, `cuda_backend/recovery.py` from hostpath) were each moved by
exactly one change and the CPU battery's pin tests hold them on the merged tree.

Same point cull (`q0.9375.120.43884.n41.95w94.79`, 43,884 cells, 55 levels,
dt 5 s; grid `e4fb6704...`, init `4c6deb1c...`), same HRRR 15Z init and
boundary files, an RTX 5090 (32,607 MiB, sm_120), one session, the
card otherwise idle (no other GPU process during any step; load average 1.14
at the first arm's launch, 3.91 at the second's, the first arm's own writer
tail), woof 2.7.3 from PyPI with its 2.7.3 bridge bundle in both venvs. Every
number is from a run receipt.

- Contract deck on the cull's own rings at the merged tree's kernel set
  (`run_cuda_regional_contract.py --class-id graded-869m-dt5-z7`): 8 of 8
  decks bitwise, 22 of 22 regional kernels covered, dual run identical, every
  control with teeth; 424 s, peak 1,597 MiB.
- `forecast --hours 3 --history-every-minutes 60` at the defaults each change
  recommended (surface/PBL every step, local time stepping off, the hostpath
  device checks on), 2,160 steps of 5 s, both arms back to back:

| arm | door wall | driver execution | integration | s/step after the first | median typical step | health gate | loop incl. I/O | peak card memory | utilisation, mean |
|---|---|---|---|---|---|---|---|---|---|
| 0.3.0 tree (untouched, shared venv) | 906.8 s | 875.5 s | 692.1 s | 0.3193 s | 0.3007 s | 168.6 s | 864.7 s | 9,485 MiB | 85.1 % |
| integrated performance tree (its own venv) | 728.1 s | 696.6 s | 675.0 s | 0.2978 s | 0.2797 s | 0.6 s | 679.6 s | 9,485 MiB | 79.8 % |
| ratio | 1.245 | 1.257 | 1.025 | 1.072 | 1.075 | 293 | 1.272 | 1.000 | |

  The integration ratio reads low because the second arm's first step was
  32.0 s of cold NVRTC compilation into its venv's fresh kernel cache against
  2.7 s in the first arm (whose cache was warm); excluding the first step the
  integration is 689.4 s against 642.9 s, 1.072x, the same number as the
  per-step column. The door wall carries that 29 s as well; with a warm cache
  the door ratio would be about 1.30, which is not a measured number and is
  not claimed. The health gate's 168 s over three hours became 0.6 s, which is
  where the loop's 1.27x comes from.
- Identity: all four history frames (15Z, 16Z, 17Z, 18Z) sha256-identical
  between the arms (`85cb8196...`, `4ed8001f...`, `8943dcf9...`,
  `9f332aa8...`; the first two are the digests every arm of the campaign
  recorded for the untouched tree), the 2,160 `step_health` envelopes
  identical entry for entry, and the 18Z composite-reflectivity panel through
  `woof hex render` the same PNG byte for byte (`3b02745f...`). The receipt's
  `physics.cadence` reads consistent: 2,160 surface-layer, Noah-MP and YSU
  calls, 18 radiation calls, nothing held.
- The tree's GPU tier (`-m "gpu and not bigcard and not assets"`) on the card
  in the same session: one failure on the first pass, the NVRTC
  reciprocal-rewrite census finding `cuda_solve_region_v841.py` unregistered
  (the control that test exists for); registered, the tier is 47 passed, 1
  skipped, and the CPU battery on a CPU node is 1,194 passed, 70 skipped, 16
  deselected. Card time for the whole chain 2,080 s, plus 8 s for the tier's
  re-run.
- Not run: the x4 full-physics dycore anchor. Its nine authority assets are not
  on that machine (no `work/v841-vr-static/` or
  `work/v841-full-physics-gf-gwdo-native-authority-20260820a/` under any
  checkout there), and the tool refuses by name without them.

What ships as the default from this campaign is the host-path change's host path
and, from the section below, the element-parallel launch geometry of the
level-independent dycore kernels, neither of which changes a byte; the cadence
knob and local time stepping ship selectable and off, each with the
measurement that kept it off in its own section above and in
`docs/local-timestep-lam.md`.

## The element-parallel kernels, measured 2026-09-14

What moved. Every dycore kernel whose levels are independent used to launch
one thread per owner (cell, edge or vertex) and loop over the levels inside
the thread, so a 43,884-cell cull put 343 blocks of 128 threads on a card with
170 streaming multiprocessors and the boundary-zone kernels, whose owners are
the 601 to 3,581 cells and edges of the relaxation rings, ran on five to
twenty-eight blocks. Eight translation units (`cuda_regional_v841`,
`cuda_dynamics_v841`, `cuda_driver`, `cuda_acoustic`, `cuda_horizontal`,
`cuda_horizontal_v841`, `cuda_transport`, `cuda_backend/recovery`) now walk a
flat `(tracer, level, owner)` element index in a grid-stride loop
(`REGIONAL_ELEMENT_LOOP` and its siblings, the same text in every unit, held
by `tests/test_element_parallel_kernels.py`), and the forecast launches one
thread per element. Each element runs the former loop body for its level with
the same operands in the same order; the vertically implicit column solve
keeps its per-column form; the momentum kernel gathers a once-per-run
reference wind instead of evaluating `cosf`/`sinf` per neighbour per stage. The
regional kernel-set digest moved (the CUDA source strings themselves moved
this time) and is re-derived at `8a716f28...` on the landed tree, with the
note at the constant saying what moved and that the four non-point classes
carry it through the constant without a re-mint of their own; the point class
row carries the re-mint sentence. The two acoustic kernels the local-timestep
unit derives from the regional unit are gathered in their element form
(`cuda_acoustic_lts.py`), because the derivation anchored on the two
per-owner preambles the re-map removed and refused to build.

The replay verdict and its resolution. The kernels change's own chain
(2026-09-14 16:14 to 16:39Z, RTX 5090, load average 0.08, no other GPU
process at any step) passed the NVRTC compile of
all eight units, the contract deck at the change's kernel set (`82a0b9ae...`; 8
of 8 decks bitwise, 22 of 22 kernels, dual run identical, 420 s), and three
one-hour arms: the untouched tree 0.3146 s per step after the first (door
328.5 s, loop as a user waits for it 286.5 s), the change's tree 0.2836 s (1.109x;
door 305.6 s, loop 264.1 s, 1.085x) and 0.2860 s in a second process, both
frames (15Z `85cb8196...`, 16Z `4ed8001f...`) byte-identical across all
three. Its per-kernel capture-and-replay (`tools/kernel_element_ab.py`, which
runs the forecast door in-process, captures the first launch of every
re-mapped kernel after the warm-up steps with a device copy of every argument
before and after, then replays this tree's kernel and the untouched tree's
translation unit from the same inputs on the same card) compared 49 kernels
and reported five as differing. None of the five was the arithmetic, and the
receipt says so in its own numbers: `recover_pressure_f32`,
`recover_edge_velocity_f32`, `recover_terrain_w_f32` and
`transport_standard_finish_regional_v841` each differed in exactly 55 values,
every one at flat index 43,884 of a `(55, 43,885)` array, which is the garbage
column (column `n_cells_solve`, every level) of an INPUT the kernel never
writes (the RK stage's `rho` for the three recovery kernels, `rho_zz_old` for
the transport finish), 0.0 in the replay against 1.0 recorded, and this tree's
own kernel did not reproduce its own recorded output either
(`replay_reproduces_recorded_output: false`). The cause was the replay
environment: every dycore launch resolves through the kernel cache the
regional runtime arms with the garbage discipline's scrub as its post-launch
hook, which rewrites the garbage column of every padded float32 argument after
every launch, 1.0 into an array bound to the step's unit pool and 0.0 into any
other, and the replay ran after the forecast had released its pool. The fifth,
`acoustic_ru_regional_v841`, differed in 7,218,551 values because the
untouched kernel was launched at one owner: the owner count of a kernel with
no entry in the instrument's table was the trailing dimension of its last
device argument, and that kernel's last device argument is the `(1,)` acoustic
invalid flag; the instrument's own control launch of the untouched kernel at
the element geometry was byte-identical, which is the arithmetic's verdict.
The instrument was corrected rather than the kernels (`4e469d3`): the recorder
asks the discipline at the launch which arguments are bound
(`bound_to_unit_pool`, a reading of a dictionary that launches nothing) and
the replay re-binds them before every launch of either translation unit; the
owner-count rule names the edge count; `tests/test_kernel_element_ab.py`
parses every target's signature so a flag or index array can never be the
owner axis, and holds the re-bind and the garbage-column split. Each
difference is also classified by whether it lies in a garbage column, but the
verdict of record stays the whole-array comparison. So neither of the two
resolutions a differing kernel would have needed (restoring an evaluation
order, or proving the differing slots have no consumer) was taken, because
no kernel's output differed; what differed was an input slot neither kernel
wrote, written by the instrument itself. `tools/nvrtc_compile_check.py`
resolves every kernel of the eight units before any card time is spent.

The landing, measured. The landed tree is the integrated performance tree with
the element-parallel kernels change merged (`04c4c69`: the recovery launches keep the host path's
once-per-kernel untimed helper and take the re-map's element grids), the
instrument fix and the local-timestep gather above. On an RTX 5090
(32,607 MiB), one session 2026-09-14 22:22:20Z to 22:41:43Z (1,163 s of
card time against the 35 min budget), no other GPU process at any step (the
session waited 120 s for another process to finish and took the card
empty), load average 0.83 at the start and 1.02 to 1.03 at every step after,
woof 2.7.3 from PyPI, the venv cloned from the integrated tree's and
editable to a copy of this tree that hashes file for file to `0648789`:

- NVRTC compile check of the eight units, 104 kernels resolved, 7 s.
- Contract deck on the cull's own rings at the merged kernel set
  (`run_cuda_regional_contract.py --class-id graded-869m-dt5-z7`, kernel set
  `8a716f28...`): 8 of 8 decks bitwise, 22 of 22 regional kernels covered,
  dual run identical, every control with teeth; 422 s, peak 1,026 MiB.
- Per-kernel replay through the door against the integrated tree's
  translation units (dumped by that tree's own venv): 49 kernels compared,
  `all_bitwise: true` on the whole-array comparison, every replay reproducing
  its recorded output, every control launch of the untouched kernel at the
  element geometry identical; capture 85.4 s (36 steps), replay 13.0 s, peak
  19,354 MiB (the fifty records' before-and-after argument images).
- One 3 h arm with the integrate's exact invocation (`forecast --hours 3
  --history-every-minutes 60`, 2,160 steps of 5 s, the same mesh, init and
  boundary files) compared with the integrate's two arms, which ran in an
  earlier session the same day (16:46 to 17:18Z) and not beside this one, so
  the ratios below are across sessions on the same card and mesh, not the
  same-session pair the campaign's method asks for; the same-session pair for
  this re-map is the change's own one-hour chain above (1.109x):

| arm | door wall | driver execution | integration | s/step after the first | median typical step | health gate | loop incl. I/O | peak card memory | utilisation, mean |
|---|---|---|---|---|---|---|---|---|---|
| 0.3.0 tree (integrate's baseline arm) | 906.8 s | 875.5 s | 692.1 s | 0.3193 s | 0.3007 s | 168.6 s | 864.7 s | 9,485 MiB | 85.1 % |
| integrated performance tree (integrate's arm) | 728.1 s | 696.6 s | 675.0 s | 0.2978 s | 0.2797 s | 0.6 s | 679.6 s | 9,485 MiB | 79.8 % |
| landed kernels tree (this arm) | 614.5 s | 583.0 s | 567.7 s | 0.2617 s | 0.2442 s | 0.6 s | 572.1 s | 8,914 MiB | 80.9 % |
| ratio, hex-perf / kernels-land | 1.185 | 1.195 | 1.189 | 1.138 | 1.145 | 1.0 | 1.188 | 1.064 | |
| ratio, baseline / kernels-land | 1.476 | 1.502 | 1.219 | 1.220 | 1.231 | 281 | 1.511 | 1.064 | |

  This arm's first step was 2.64 s (its venv's kernel cache was warm from the
  deck and the replay in the same hold), so no cold-compilation correction is
  owed here; the integrate's hex-perf arm carried 32.0 s of it, which is why
  its door and integration ratios above read higher than its per-step ratio.
- Identity: all four history frames (15Z `85cb8196...`, 16Z `4ed8001f...`,
  17Z `8943dcf9...`, 18Z `9f332aa8...`) sha256-equal to the digests both
  integrate arms recorded, and to the untouched tree's own; the receipt's
  `physics.cadence` reads consistent (2,160 surface-layer, Noah-MP and YSU
  calls, 18 radiation calls, nothing held).
- The tree's GPU tier (`-m "gpu and not bigcard and not assets"`) in the same
  session: 47 passed, 1 skipped, 19 s. The CPU battery on a CPU node (no card,
  `GPUWM_NO_LOCAL_GPU=1`, load average 0.55 to 0.97): 1,346 passed, 70
  skipped, 16 deselected, 65 s.

Per kernel, from the landed tree's replay (device microseconds, median of
twenty launches, the captured launch's own inputs; one launch each, so the
sum is not weighted by how often a step launches each kernel): the 49
captured launches sum to 26,840 us on the untouched translation units and
21,594 us on this tree's, 1.24x. The boundary-zone kernels gain most, 5x to
17x (`regional_bdy_adjust_scalars_compute` 1,560 to 93 us,
`regional_relaxzone_filter_cell` 746 to 44, `regional_relaxzone_filter_edge`
836 to 105, `regional_bdy_set_scalars` 67 to 12), because their owner sets
had filled a handful of blocks; the level-heavy cell kernels next
(`recover_terrain_w` 945 to 388, `tendency_w_to_omega` 884 to 378,
`w_vertical_flux` 69 to 27, `transport_standard_finish_regional` 1,960 to
1,465, `theta_vertical_flux` 77 to 48); the two edge-flux kernels and the
momentum kernel, which were already wide, gain 3 to 8 per cent. Slower in
element form on the captured launch: the eight small elementwise kernels
(`add_inplace`, `vertical_u_finish`, the three `split_flux_*`, 21 to 23 us
before, 31 to 32 after, the grid-stride loop's overhead on 7.3 million
elements), `theta_finish` and `w_finish` (139 to 185 us),
`mass_flux_divergence` (228 to 286), `pv_cell` (393 to 470),
`cell_diagnostics` (764 to 819), `transport_vertical_flux` (370 to 401), and
`acoustic_ru_regional_v841`, 53 to 355 us: the captured launch was a stage's
first acoustic sub-step, whose body is two loads and two stores per element,
and the element form pays the edge's connectivity and mask loads and the
`rgas / (cp - rgas)` division per element where the per-owner form paid them
once per edge; three launches a step take that path (the first sub-step of
each Runge-Kutta stage), about 0.9 ms of a 244 ms step. All of these are
bitwise; each is a named follow-up for a later change (a per-owner text for the
first sub-step, or hoisting the per-edge loads out of the level axis), and
none of them holds the landing, because the composite step, which is what a
user waits for, went from 0.2978 to 0.2617 s with the frames unchanged.

## The forecast presets, measured 2026-09-27

`woof hex forecast --preset NAME` selects a row of `woof.hex.forecast_preset`.
Two rows ship. `reference`, the default, is the proven configuration: legacy
RRTMG, Noah-MP, revised MO and YSU, with the surface layer, Noah-MP and YSU
called every model step. `fast` keeps the same schemes and holds the
surface/PBL stack for up to 30 s between calls (six steps at dt 5 s; the weld
where dt is 20 s or longer). The door turns the row into `--pbl-cadence` at
the mesh's timestep and hands it to the same admission; an explicit
`--pbl-cadence` wins, and the receipt's `configuration.preset` and
`pbl_cadence_source` say which decided.

**Every row got faster, with the frames unchanged.** Before phase one of every
step the physics adapter copied every persisted seam array to the host (447 MB
and 192 copies a step on this cull) so that a refused step could be rolled
back. Only `--stop-on-refusal` reads that export, so the forecast now takes it
only under that option. The 3 h forecast of 2026-09-13 wrote all four history
frames with the same SHA-256 as before (`85cb8196...`, `4ed8001f...`,
`8943dcf9...`, `9f332aa8...`).

**Speed, one session.** The 1 h forecast of 2026-09-13 (720 steps of 5 s),
four arms back to back on an RTX 5090 in one mutex hold, the card otherwise
idle:

| arm | s per step after the first | minutes per simulated hour | median step | radiation step | door wall | 16Z frame |
|---|---|---|---|---|---|---|
| before: the default at `c9f2edf` | 0.2678 | 3.21 | 0.2517 s | 2.56 s | 240.3 s | `4ed8001f...` |
| after: `reference`, the default | 0.2367 | 2.84 | 0.2216 s | 2.49 s | 219.9 s | `4ed8001f...`, the same bytes |
| after: `--preset fast` | 0.2225 | 2.67 | 0.2051 s | 2.39 s | 205.5 s | differs |
| measurement arm: RTE-RRTMGP seam | 0.2228 | 2.67 | 0.2222 s | 0.33 s | 201.3 s | differs |

The mutex hold ran 14:27 to 14:49Z (load average 1.0 at the first arm and a
transient 18.5 as the second started; that arm's step time is within 2 per
cent of the same row's other 1 h runs of the day). The 15Z frame is
`85cb8196...` in all four. The before arm reproduces the September reference
number for this case (0.2617 s a step on a 3 h arm) within 2.3 per cent.

The 3 h forecasts agree: 0.2683 s a step before the change and 0.2451 s after
it on 2026-09-13; on the four graded cases the reference row ran 0.2353 to
0.2451 s a step, `fast` 0.2220 to 0.2258 and the RTE-RRTMGP arm 0.2203 to
0.2242.

**Skill against observations.** Four 3 h forecasts from HRRR 15Z on this
cull: 2026-09-13 (the profile's case) and 2026-09-25, 2026-05-22 and
2026-08-04, the three wettest days over the cull in Stage-IV hourly
precipitation at 16 to 18Z between May and September, at least a week apart.
The observation referee (`tools/run_obs_referee.py`) scored each arm against
MRMS composite reflectivity, Stage-IV hourly precipitation (the MRMS door
decodes reflectivity only) and ASOS 2 m temperature, 2 m dewpoint and 10 m
wind, paired by case, 5,000 bootstrap replicates. Means over the four cases,
each arm minus the reference row, in the score's own units:

| score | reference | `fast` minus reference | verdict | RTE-RRTMGP arm minus reference | verdict |
|---|---|---|---|---|---|
| ASOS 2 m temperature RMSE, K | 2.369 | +0.013 (worse on 3 of 4) | indistinguishable | -0.046 (better on 3 of 4) | favors the arm |
| ASOS 2 m dewpoint RMSE, K | 1.500 | +0.017 (worse on 3 of 4) | indistinguishable, guardrail fails | -0.001 | indistinguishable |
| ASOS 10 m wind speed RMSE, m/s | 1.760 | +0.049 (worse on 4 of 4) | favors reference | +0.011 (worse on 3 of 4) | favors reference |
| Stage-IV 1 h precipitation RMSE, mm | 1.556 | +0.004 | favors reference | +0.009 | favors reference |
| Stage-IV 1 h precipitation CSI, 1 mm | 0.259 | -0.0001 | indistinguishable | +0.0017 | favors the arm |
| Stage-IV 1 h precipitation FSS, 1 mm, 4 cells | 0.471 | +0.0001 | indistinguishable | +0.0021 | favors the arm |
| Stage-IV 1 h precipitation CSI, 10 mm | 0.037 | 0.0000 | indistinguishable | -0.0003 | indistinguishable |
| MRMS composite reflectivity CSI, 20 dBZ | 0.366 | -0.0003 | indistinguishable | +0.0002 | indistinguishable |
| MRMS composite reflectivity CSI, 40 dBZ | 0.032 | +0.0001 | indistinguishable | +0.0001 | indistinguishable |
| MRMS object centroid displacement, km | 34.4 | -0.8 | indistinguishable | +3.7 | indistinguishable |

The referee's verdict on both arms is DISFAVORED: its guardrails (the ASOS
scores) allow no adverse change, and each arm has one. The Stage-IV bias row
is left out of the table because the manifest scores it lower-is-better on a
negative bias; the RTE-RRTMGP arm's bias is closer to zero on three of four
cases.

**The decision.** `fast` does not score as well as `reference`: its
reflectivity and precipitation threshold scores are the same, and it loses
near the ground, 10 m wind on every case and 2 m dewpoint and temperature on
three of four, the fields the held surface layer, Noah-MP and YSU act on
directly. It stays selectable at about 6 per cent less time per
step, and `reference` stays the default.

**RTE-RRTMGP, measured but not selectable.** The engine's column seam writes
the radiation variant into its `RunConfig` as a literal
(`woof/core/mpas_column_batch.py`, one of the sixteen pinned files), so no
row can select RTE-RRTMGP through the pinned engine. The arm above was run by
a measurement script that substitutes the seam's `RunConfig` in-process; no
engine or hex file is changed, and no shipped option runs it. Its radiation
call takes about 0.10 s against legacy RRTMG's 2.2 s (the radiation step 0.33
s against 2.49 s), which alone buys about as much as the `fast` hold does,
with the surface stack still called every step. Its scores are mixed: better 2
m temperature and 1 mm precipitation placement, slightly worse 10 m wind and
hourly precipitation RMSE. It becomes a row here once the engine seam takes
the radiation variant as an argument.

**Where a step's time is now.** Nsight Systems over steps 101 to 160 of the
reference row under the profiler (the radiation step left out): 0.2256 s a
step on the host, 0.2027 s of it the card busy and 0.0229 s idle. The dycore
is 1,419 kernels, 550 copies and 184 memsets a step and keeps the card busy
for 0.189 s of its 0.192 s, so it is bound by the card, not by its launches;
its idle time is 2.9 ms a step. Of the remaining idle time, 14.5 ms a step is
Noah-MP inside the engine (1,613 kernels, 229 memsets and 83 copies a call for
2.4 ms of device work) and about 4 ms is the preparation, recovery, clamp and
gate flag reads. The CUDA admission is answered from memory (68 calls a step,
0.14 ms in all) and the health gate costs 0.3 ms a step.
