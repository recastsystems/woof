# woof hex

MPAS-Atmosphere v8.4.2 on the GPU, driven from the WOOF engine.

woof hex ports MPAS-A v8.4.2 to Python and CUDA, fully device-resident: the
dynamical core on its unstructured global mesh with regional refinement, the
MPAS physics suites, initialization and static fields, the incremental analysis
update (IAU), limited-area runs, streams, restart and diagnostics. It runs on
one consumer CUDA card. MPAS's own physics is the default; WOOF's physics
stays one flag away, and hex adds capabilities MPAS does not have:
one-command global forecasts, local time stepping, forecast products and scores
at the end of every run, and an experimental storm-following region.

*(MPAS-Atmosphere is a registered project of NCAR and LANL. This distribution
is an independent port and is not affiliated with or endorsed by them. The name
is used here only to say which model was ported.)*

**New here? Start with the [User Manual](docs/manual/index.md):** a
plain-language introduction, a quickstart in which every command was run
before it was printed, task guides for each door, and a troubleshooting index
built from the doors' own refusals. This README is the capability contract;
the manual is how you walk it. [What's new in 0.4](docs/manual/10-whats-new-0.4.md)
gives each new piece its command, its proof, a picture and what it does not
yet do.

## 0.4 in short

[Physics routes and measured comparison scopes](docs/physics-routes-0.4.1.md) describe the MPAS physics route (the default since 0.4.6), the WOOF route and public licence exclusions.

0.3.3 (2026-09-29) was the MPAS-A dynamical core on the GPU, running WOOF's
physics. 0.4.1 is the whole of MPAS-A v8.4.2 on the GPU, each piece proven
against the real thing before it landed:

1. **The dycore became exact.** Matching native MPAS-A's order and precision of
   operations, with no change to its equations, time scheme or damping, made
   the ported dycore give the same bits as native MPAS-A v8.4.2 built with a
   strict GNU recipe.
2. **All of MPAS-A was ported.** Both physics suites and their scheme options
   (the few not in this release are listed under *Not in this release*),
   initialization, static fields, IAU, limited-area boundaries, streams,
   restart and diagnostics. Since 0.4.6 MPAS's own physics is the default on
   every door; WOOF's physics stays, one flag away (`--physics-suite
   arwen_mpas_column`).
3. **Each piece proved itself before it landed:** bit for bit against native
   MPAS on real inputs captured from a native run, then a forecast with it on,
   compared with native map by map.
4. **The hardest test:** a 24-hour forecast with every physics scheme of the
   `convection_permitting` suite on the GPU gave the same bits as native MPAS
   in every field hex writes. It is now a gate: every change to the port must
   keep it exact before it lands.
5. **Matching exactly found real defects in native MPAS.** Five are fixed in
   correct mode, which is the default; matching native exactly is one flag
   away.
6. **Beyond MPAS:** `woof hex global` (one command from a global row to a
   scored, drawn and exported forecast), an experimental storm-following 3 km
   region, and regional forecast presets.



---

## What is proven, and how far

"Byte-identical", "bitwise" and "identical" below always mean every stored
word of every compared field, with no tolerance. Each claim states its scope.
Nothing outside the scope is claimed.

Kernels compile with fast numerics by default (FMA contraction, flush-to-zero); `--numerics strict` is the opt-in for bitwise identity to native MPAS-A, and every identity claim here was measured with it.

### The dynamical core

Byte-identical to native MPAS-A v8.4.2 built with the pinned strict recipe
(GNU Fortran, single precision, no FP contraction, no vectorised
reassociation), on these measured cases:

- `x1.40962`, 55 levels, real terrain, dry, 5 days, on two dates (2023-03-31
  12Z and 2022-12-22): 0 differing words at all 21 six-hourly samples;
- the same mesh with water vapour transported, and with water vapour plus five
  condensates, 5 days, on 1, 2 and 4 ranks sharing one card;
- `x1.163842` with water vapour, one step and 5 days;
- one dry step on the public `x1.2562` mesh (three schedules);
- 24-hour dry forecasts on `x1.2562` and `x1.10242`, one process, which the
  landing gate re-checks on every change to the port.

The identity is per host C library: hex evaluates the single-precision
geometry maths (`sinf`, `cosf`, `asinf`, `atan2f`, `powf` and related
functions) through the host's GNU libm, as the native executable does, and
glibc 2.39 and glibc 2.43 round some of them 1 ulp apart (`asinf` and `atan2f`
among them), so hex and the reference run on one host and the receipt names
its glibc. Against an optimised native build, hex
differs exactly as much as two native builds differ from each other. Not
claimed for the dycore alone: other meshes, more ranks, several cards, and
local time stepping, which is a different time scheme by design and is never
called identical. [MPAS compatibility](docs/mpas-compatibility.md) has the
tables; `woof hex verify-dycore` rebuilds the reference and checks it.

### MPAS physics, all on the GPU (the headline)

One 24-hour forecast of the `convection_permitting` suite on the public
`x1.2562` mesh, starting 2023-03-31 12Z, one process, exact mode:

- ten physics schemes run as hex's CUDA code: Thompson microphysics,
  Grell-Freitas convection, the MYNN boundary layer, the MYNN surface layer,
  Noah land surface, sea ice, YSU orographic drag, cloud fraction, and RRTMG
  longwave and shortwave radiation (the suite's other two options, the
  non-stationary gravity-wave drag and the ocean mixed layer, are off in it);
- every field native MPAS-A v8.4.2 writes in this forecast, hex writes too, and
  all of them match bit for bit (strict numerics): 650,457,072 values in 13,938
  written fields, 0 different;
- the forecast is a landing gate, so every change to the port keeps it exact.

The same gate runs both MPAS suites (`mesoscale_reference` and
`convection_permitting`) with every scheme on the native backend, and they are
bitwise in every written field too (`x1.2562`, 24 h, one process). The decks
are in [`verification/gates/`](verification/gates/README.md) and the
reference recipe in `tools/native_physics_reference/`.

### Every other MPAS-A physics scheme

Each is proven on its own: bitwise against native MPAS-A v8.4.2 on real inputs
captured from a native run, and a forecast with it on compared with native,
map by map. In this release:

| Kind | Schemes |
|---|---|
| Microphysics | Kessler, WSM6, Thompson, aerosol-aware Thompson |
| Convection | Kain-Fritsch, Tiedtke, New Tiedtke, Grell-Freitas |
| Surface layer | revised MM5 (both MPAS options), MYNN |
| Boundary layer | YSU, MYNN |
| Gravity-wave drag | YSU orographic drag; UGWP orographic drag and non-stationary gravity waves (NOAA GSL) |
| Land, sea and ice | Noah, sea ice, the ocean mixed layer; Noah-MP (with its urban switch) in private builds only (see *Licence*) |
| Radiation | the shared radiation code with the `cld_fraction` cloud fraction, RRTMG longwave and shortwave |

Not claimed: a whole-forecast match for each scheme one by one, or for suites
other than the headline. Not in this release: the CUDA port of Noah-MP v5.0
is withheld (see *Licence*), and CAM radiation and the `cld_incidence` and
`cld_fraction_thompson` cloud fractions do not run on the GPU route (see
*Not in this release*).

### Correct mode: MPAS with its defects fixed

Every door runs **correct mode** unless told otherwise. On an MPAS suite it
fixes 5 defects found in native MPAS-A v8.4.2:

| Id | Native defect | In correct mode |
|---|---|---|
| RP1 | MYNN removes cloud water, ice and snow on every call | the exact inverse conversion (GPU MYNN) |
| RP4 | `config_mynn_tkeadvect=true` resets TKE to a field that was never advected | the setting is refused, as a corrected native build does |
| RP6 | the PBL interval stretches the step while the schemes still run every step, so they integrate twice over | the surface and PBL schemes are called on their alarm |
| WB3 | precipitable water divides by (1+qv) a second time, so it reads about 0.6% low | the density statement corrected (GPU WSM6, Thompson, Kessler) |
| WB5 | WSM6 books the below-ground part of a falling cell flat while its remap moves it on a slope | the remap's profile is used (GPU WSM6) |

Each fix has a test that fails on native behaviour and passes with the fix,
and each is checked word for word against its own native referee, a native
build carrying that one fix. No combined correct-mode forecast is claimed
against a combined referee.

A correction a scheme does not carry yet is never a refusal. The run keeps
native behaviour for that finding, prints `CORRECT MODE: named gaps ...` at
start-up with the finding, scheme, reason and owner, and records the same in
its receipt. `--physics-mode exact` reproduces native bit for bit, and every
proof above was run in exact mode. `--physics-correct-skip ID` leaves one
correction out, for an observation A/B arm. On the WOOF physics route the
same flags, and correct mode by default, apply the corrections the installed
woof carries (WB4, New Tiedtke's dry-mass coupling; WB5, WSM6's surface
integral) and name the rest as gaps. Details: [MPAS physics seam](docs/manual/mpas-physics-seam.md#correct-mode-and-exact-mode).

### Initialization, statics and the services around the forecast

| Piece | Proof, against native MPAS-A v8.4.2 |
|---|---|
| Real-data initialization (`init_atmosphere` case 7) | 2,880 captured native calls replayed, 134,016 output words, 0 different; a cold start from it matches all 134 native fields on both measured meshes, and a forecast from it matches native in every written field (426,595,161 values) |
| Static fields (`rw_mpas_static`, the native static stream) | every word equal, except 1,473 `albedo12m` and 189 `soilcomp` values where native itself reads undefined memory: declared residue, never called identical |
| Gravity-wave-drag statics (`rw_mpas_static --native-gwd-static`) | the complete 79-field static file: 4 cases, 11,675,352 words, 0 different |
| GSL orographic statics, idealized initialization | bitwise on captured native inputs |
| GOCART aerosol climatology (initial and boundary) | 91,420,560 words, 0 different |
| SST and sea-ice update (`woof hex surface-update` builds the records; the forecast's surface update applies them) | a forecast with SST updates on matches native in every written field: 581,775,705 values, 0 different |
| IAU | MPAS-A's IAU kernels bitwise on captured native inputs; in a forecast IAU runs on the WOOF physics route (`--iau-route native\|hex`, IN1-IN3 applied in correct mode) and on the explicit MPAS suite route (MPAS's own `atm_add_tend_anal_incr` once per step) |
| Limited-area boundaries | 2,822,211,692 words, 0 different, on captured native inputs; seven regional cases' difference maps are zero |
| Streams and restart | bitwise on captured native inputs |
| Diagnostics: isobaric, convective, cloud, potential vorticity, soundings | bitwise on captured native inputs (isobaric: 471,380 words, 0 different) |

The real-data initialization's vertical grid comes from `rw_mpas_geometry`
speaking `hex-vertical-v1`, MPAS streams and restart use `rw_mpas_stream`,
and a mesh generated at a point needs an `rw_mpas_static` whose statics hold
the binary64 coordinates they declare. No released woof carries any of the
three in the version this release needs, so the wheel carries them in
`woof/hex/libexec` ([engine bridges](docs/engine-bridges.md)). A door that
needs one and cannot find it refuses by name.

**A forecast from an analysis.** `woof hex forecast --analysis BUNDLE` starts
a forecast from an MPAS analysis bundle ([analysis bundle](docs/analysis-bundle.md)):
by direct insertion, with an IAU window from the bundle's increment, as a
cycling restart, or with a remap onto another mesh. Measured on one convective
case (WOOF physics): with no bundle every history file is byte-identical to
the run before this release, and in correct mode a zero increment is bitwise a
plain run. IAU also runs across a restart (the checkpoint carries its window),
on the explicit MPAS suite route, on same-host multi-card ranks and in member
rosters ([analysis bundle](docs/analysis-bundle.md)); IAU on multi-card local
time stepping is refused by name.

### Hex global, in one command

`woof hex global --row global-120km --cycle 2026-10-11T12 --work-dir RUN`
goes from a global row and a cycle to a scored, drawn and exported forecast:
the NCAR-published mesh (fetched and pinned), the native static, the init, the
forecast on as many cards as the mesh needs, products, export and scores.
Rows are defined at 120, 60, 30 and 15 km, on WOOF physics and on the
woof global suite; the measured runs are the 120 km rows. **Hex global is
WOOF's global model** (the spectral global model was retired in WOOF
1.1.1). Hex global starts from an analysis you supply
(`woof hex forecast --analysis`).

On one case (GFS 2026-10-11 12Z, 24 h), the 500 hPa height error against the
GFS analysis was 10.0 m for hex at 120 km, against 14.9 and 13.5 m for the
spectral global model's T127 and T255 runs (retired in WOOF 1.1.1); T255, at twice hex's resolution, was best on
250 hPa wind (4.80 against 4.87 m/s). One case is not a skill claim. On the
120 and 60 km rows the physics step is capped at 300 s on woof 2.9.0: at the
mesh's own step the YSU boundary layer and New Tiedtke left single layers up to
24 K too hot, and one row ran away to 381 K. The cause is explicit tendencies
overshooting their donor layers; an engine carrying both donor bounds runs the
WOOF rows at the mesh's own step with no hot layer, and the door retires the
cap there by itself. See [Hex global](docs/global-door.md).

### Storm-following region (experimental)

`woof hex moving` keeps a 3 km region that moves with a storm inside a 48 km
global mesh, on the MPAS-exact dycore with the physics the case selects (the
measured runs used WOOF physics). One case file drives `stage`, `prepare`,
`run` and `fixed` (the matched control); `woof hex moving check` tests an
install with no inputs. With no moving command and no `WOOF_HEX_MOVING_*`
variable set, a run is unchanged (`tests/test_moving_inactive_noop.py`).
Guide: [Moving nests](docs/moving-nest.md).

Measured: three tropical cyclones, 48 hours each, five moves each, one card,
a 3 km region about 450 km in radius.

- The storm stayed inside the 3 km region at every hourly output (49 of 49 in
  each run).
- Mass and tracers held to within 1e-10 at each move.
- A restart across a move continued bit for bit (one move tested).
- Those runs used the remap before the balance fix: at 2 of the 15 moves, the
  short-term shock measured against a fixed-region run went over its 1.5 limit
  (1.5004 and 1.81). The balance fix is now the default; on `woof hex
  moving check` it brings the worst post-move jolt from 16.2 to 1.01 on a
  card with local time stepping and from 21.4 to 1.04 on the CPU (ordinary
  integrator); see [the jolt report](docs/moving-jolt-report.md). A
  full-cyclone replay of those two moves with the fix is not yet run. MPAS-A
  v8.4.2 suites move too, on one card: each move carries the suite's physics
  state, and on a 2 h western Pacific case with every convection_permitting
  scheme on the GPU the post-move jolt was 1.38 (15 min) and 0.69 (30 min)
  against the 1.5 limit (docs/moving-nest.md, *MPAS physics on a moving
  region*).

### RRFS-MPAS

NOAA's MPAS configuration, built on the GSL v8.3.1 fork of MPAS-Model
(RRFSx/MPAS-Model 1d6e370e), runs on the GPU with no native library: the
fork's framework, scheme initialization, stream services and per-step
interface glue are ported, and `hrrrv5` runs every physics slot GPU-resident
on one card. The v8.4.2 default is MPAS-A's own physics (convection_permitting). The
named MPAS physics suites remain selectable within their documented CUDA-port
scope. The public wheel does not contain a
native Fortran runtime or the held UCAR Noah-MP v5.0 port. Historical
fork-native replays below are references, not a bundled runtime path.

What is bitwise, on real inputs captured from the native fork build (global
and regional cases, one process):

- the fork's dynamical core, as a module replay: 1,235,615,738 words, 0
  different;
- the fork's physics driver layer, the same way: 104,976,732 words, 0
  different.

**A whole RRFS forecast is bitwise to the native fork build in every field hex
writes:** a one-hour `hrrrv5` run on `x1.2562` (2023-03-31 12Z, one process)
with the fork's dycore on the GPU and every physics slot on the native
library compares 40,525,734 words with the native fork build (strict recipe),
0 different, and 0 different at every 720 s step when every stream is written
each step. Scope: that case and process count; native also writes fields hex
does not write yet, so the claim is every field hex writes. That run put
every slot on the native library (`--physics-default-backend native-cpu`).

The measured RRFS GPU-port run used these schemes on `hrrrv5`:
TEMPO, MYNN-EDMF, the MYNN surface layer, RUC land and sea ice,
RRTMG longwave and shortwave and the cloud fraction (GFL, MYJ and the fork's
MYNN surface layer when a suite selects them). Each port is bitwise to the
native fork build as a module replay on real captured calls, except two RUC
sea-ice calls on an RTX 5090 that differ in one `tsnav` word each where the
native code reads an uninitialised value.

**With every slot on the GPU and no native library, a whole `hrrrv5` forecast
is bitwise to the native fork build (strict numerics):** a 6 h `x1.2562` run
(2023-03-31 12Z, one process) differs in 0 of 29,644,818 history and
diagnostics words from the native reference at every hour f000-f006. The run
names the reference's own strict-built TEMPO lookup tables with
`--native-physics-tables` (native ran with them too; a default run reads the
fork's shipped tables). With hourly restart files as well, 57 of 129,371,634
words differ, all of them RUC's snow temperature `tsnav` at thin-snow cells,
where native reads an uninitialised local (`ilnb`); a native build with that
local set to 0 matches the GPU run in every word. Scope: that case and
process count. `config_<slot>_backend = 'native-cpu'` puts one slot back on
the native library.

`hrrrv5` needs no native library: a 6 h `x1.2562` run (2023-03-31 12Z, one
card) with every copy of the native library hidden from the process completes
on the ported fork framework, writes its history and diagnostics through
`rw_mpas_stream`, and maps no MPAS library. Its fields stay as close to the
native fork build as the native-library route's do. The
`rrfs_convection_permitting` and `rrfs_mesoscale_reference` aliases and
`cu_ntiedtke` on conus12km also use resident CUDA table entries without a
native library. NSSL and smoke still take a site-built native library: `woof hex physics-resources build --pin-set rrfs` with the required
C and Fortran compilers, then that cache or `--native-physics-library` and
`--mpas-source`. Public release assets contain physics DATA only.
Noah-MP v5.0 stays a licence stub unless that build and the run opt in.
The wheel bundles the three Rust programs the fork's dycore calls.
`woof hex init --physics-suite hrrrv5` makes `init.nc`, the fork static and
the hourly boundaries from GFS, the fork's init ported to Rust
([RRFS init](docs/rrfs-init.md)). The deck's invariant file is read by default
from the deck's directory or the init's (`--stream-input invariant=FILE` names
another).

The RRFS-shaped presets (`--physics-suite rrfs_v2_dev`) are a different thing:
they run WOOF physics shaped after RRFS v2's `hrrrv5` suite, and they are
labelled RRFS-shaped, not RRFS-matched.

### Earlier releases, still true

Measured on an RTX 5090:

- 24 h global forecasts at 720/720 steps, about 3.07 s/step, **deterministic**:
  two arms of the same run compared byte-for-byte and identical.
- 2-D Smagorinsky horizontal mixing, ported from the native Registry default and
  **on by default**, with a full A/B pass.
- Two-node multi-GPU, bitwise partition-invariant against the single-GPU
  reference. Built and proven, **not shipped**: there is no door on it.
  (Several cards on **one** host is a door from 0.4: `--samehost-devices`;
  see *Limitations*.)
- 702 product PNGs rendered through the Rust renderer.
- The init door matches the recorded arrays of 92/92 carried fields against a native
  golden init, and an init it produced started the port in a 5-step smoke.
- The render door produced 31/31 renderable products from a native history.
- Restart: checkpoint and restore with bitwise-identical continuation; in the
  full-physics proof harness the restarted history file is byte-identical
  (same SHA-256) to the uninterrupted run's.
- The engine-pin move onto the GF local-memory frame cut left the trajectory
  untouched: 132/132 history variables and every per-step atmosphere
  fingerprint byte-identical across 30 full-physics composite steps, against a
  1-ULP flip instrument the comparator flagged.
- Device footprint, measured at two meshes in one session at the merged tip: the published 40,962-cell
  global mesh peaks at **8,874 MiB** and the native x4.163842 at
  **20,446 MiB** on the 170 SM card. The model those points feed is **not a
  line in cell count**: it is a per-card core, plus the Grell-Freitas and
  YSU column workspaces charged at `min(cells, tile)`, plus a per-cell term,
  and it reproduces both peaks exactly.
  Every admission gate (the forecast door, `--preflight`, and the driver's
  own floor) answers from that one surface (`woof.hex.device_admission`),
  which reads the card's multiprocessor count at the moment of the decision.

Measured on an RTX 3080, a 10 GiB Ampere card:

- **A limited-area forecast runs the full WOOF physics stack** (WSM6,
  Grell-Freitas, YSU, YSU-GWDO, revised-MO, NoahMP, cloud fraction and RRTMG)
  behind lateral boundary conditions built from its own coarse parent. Six
  hours, 1,080/1,080 steps, 13 history frames on an 11,020-cell culled mesh,
  peak **6,224 MiB**, median 0.271 s/step, 343 rendered products. Against a
  global full-physics run over the same ground at t+6 h: theta 1.117 K RMS
  (r = 0.999973), precipitation r = 0.95, reflectivity r = 0.82, vertical
  velocity r = 0.621. The cull is
  11.0x fewer cells and 5.1x less memory than the global refined mesh that
  covers the same storm.
- **Four independently placed fine grids, over four different kinds of
  weather**, each built and run full physics in sequence on one card:
  1,080/1,080 steps every time.
- **The cascade follows weather across cycles.** `woof hex cycle run`
  re-detected six hours on and continued all four slots (three reusing their
  mesh, one regenerating) over two cycles of one real case, 1,058 s total,
  peak 7,744 MiB. Starting a corridor from transplanted parent state instead
  of from the beginning saved **273.8 s, 43 %**, against a real baseline arm.
- **The domain-size question is measured, not assumed.** Five concentric culls
  of one parent against a no-boundary control: every field improves
  monotonically with cut width and the knee is at **1.35x** the fine core
  (`w` r 0.624 to 0.744, 2 m temperature r 0.578 to 0.852) for +27 % cells and
  +25 s of wall. That is the shipped default.

Also proven:

- **MPAS-A's own defaults, with no certificate gates** (2026-10-10). A bare
  forecast runs MPAS-A v8.4.2's Registry schedule (RK order 2, four acoustic
  sub-steps (two through v8.4.1; `--acoustic-substeps 2` still selects it),
  three splits, the monotonic limiter, radiation every 1800 s,
  surface/PBL and cumulus every step) at 6 s per km of the mesh's nominal
  spacing (15 km -> 90 s, 30 km -> 180 s, 60 km -> 360 s), every value cited
  to its upstream line (`woof.hex.mpas_defaults`) and every knob overridable
  (`--dt`, `--time-integration-order`, `--acoustic-substeps`,
  `--radiation-seconds`, `--no-monotonic`, `--pbl-cadence`; any other
  dynamics option by its namelist name with `--dynamics KEY=VALUE`, or from
  a `--physics-namelist`). A run is refused
  only for a concrete breakage: a cadence the timestep cannot schedule, a
  clock that does not close, or a step above the acoustic stability limit
  or the advective stability estimate measured from the mesh's real dcEdge
  (`woof.hex.timestep_admission`; the estimate assumes a largest wind of
  125 m/s, which `--max-wind` restates for the case). The engine version is
  not a condition either: a forecast runs on a woof other than the one the
  seam was measured against and records the comparison
  (`arwen_engine_pin`; the one engine refusal is a radiation helper set no
  row admits, see *The engine pin*), and an edit to this port's own sources is recorded
  (`sources.matches_pins`) instead of closing the forecast until a re-pin.
- **Variable-resolution mesh generation**, spec-driven and deterministic, with
  a bind-time refusal for any mesh carrying a cell a Goldberg polyhedron
  cannot have. One graded mesh regenerates **bit-identical** to its registered
  bytes and completes a 6 h full-physics forecast.

## What is *not* claimed

- **Outside the measured scope, nothing.** The dycore identity holds on the
  cases listed above, on one host's C library; the all-GPU match holds for the
  one headline forecast; each other MPAS scheme is bitwise on captured inputs,
  not proven forecast by forecast.
- **Correct mode is not native MPAS.** By design it differs from native
  wherever a fix acts. Each fix is checked against its own referee build;
  there is no combined referee.
- **WOOF physics is not MPAS physics.** A forecast on the WOOF
  route (`--physics-suite arwen_mpas_column`) is not identical to native MPAS, and the dycore check does not
  qualify one. Its measured differences from native MPAS are registered, with
  mechanism, magnitude and observational referee, in
  [Declared divergences](docs/declared-divergences.md). Its referee is skill
  against observations (MRMS, Stage-IV, ASOS, radiosondes), not agreement
  with another model; a bias that is wrong against observations is still
  wrong.
- **RRFS-MPAS is matched on its measured case only.** Six hours of `hrrrv5` on
  `x1.2562`, one process, every slot on the GPU, is bitwise to the native fork
  build in every history and diagnostics word hex writes (strict numerics, the
  reference's TEMPO tables named); other meshes, dates, process counts and
  longer runs are not measured.
- **Local time stepping is never called identical.** It is a different time
  scheme, and native MPAS has none.
- A one-step dry pass does not qualify a forecast trajectory or its weather
  skill.

## Not in this release

| Item | Reason |
|---|---|
| The CUDA port of Noah-MP v5.0 (`sf_noahmp`, with its urban switch) | withheld: UCAR's terms forbid selling it or any work that contains it. The native backend runs Noah-MP v5.0 only with `--licence-opt-in ucar-noahmp`, from a native library built with the same opt-in (`woof hex physics-resources build --licence-opt-in ucar-noahmp`); the Noah-MP that hex runs without that opt-in is the WOOF route's WRF-lineage v4.x |
| CAM longwave and shortwave radiation (`cam_lw`, `cam_sw`) | on the GPU route a run that selects them is refused by name before step 1. The port has run only standalone column witnesses: its tables and module words come from the native library (the rw-mpas CAM table decoder is not built), and no whole forecast has run on it on any route |
| The `cld_incidence` and `cld_fraction_thompson` cloud fractions on the GPU route | they are in the shared radiation code but not bound to the resident GPU glue, so a GPU run that selects them is refused by name before step 1; they run only on a site-built native library (see *Requirements*). `cld_fraction`, the choice of both MPAS suites, runs on the GPU |
| NSSL two-moment microphysics and smoke/dust on the RRFS stage's GPU | they run on the native library: they are ported but not wired into the forecast. These two selections require a site-built native library |
| Zone carry (corridor cases) on a moving region with an MPAS suite, and the RRFS suite on a moving case | zone carry with an MPAS suite is refused by name (run the case without a corridor; its moves take the overlap remap). The RRFS suite takes the same route but has not been run on a moving case. MPAS-A v8.4.2 suites move on one card (docs/moving-nest.md) |
| A full-cyclone replay of the two over-limit moves with the balance fix | not yet run; it needs a card with about 64 GB or more (the 470,308-cell member holds about 54 GB, and a move about 62 GB) |
| An ensemble analysis inside hex global | not part of this package: hex global starts from an analysis you supply (`woof hex forecast --analysis`) |
| IAU on multi-card local time stepping, and `--balance-series` off one card | the LTS block executor has no held-tendency hook and the ranks no balance sampler; refused by name |
| A combined correct-mode native referee | one native build carrying every fix is a follow-up |
| A hex-specific maths-library pin | it changes the dycore arithmetic and needs every native reference rebuilt |
| YSU and New Tiedtke single hot layers at long steps on woof 2.9.0 | WOOF-physics runs with YSU at a step above about 300 s (meshes coarser than about 50 km at MPAS's 6 s per km) can show the single hot layers the global door caps away. The cause is YSU's explicit boundary-layer-top entrainment and New Tiedtke's explicit rates overshooting their donor layers. An engine whose YSU and New Tiedtke identities are `ysu-wrf-v4.8.0-entrainment-courant-v2` and `new-tiedtke-wrf461-cumastrn-donor-bound-v2` bounds both, and `woof hex global` then runs the WOOF arm at the mesh's own step. woof 2.9.0 predates this release's engine floor and has neither, so on it every door's default step with YSU is capped at 300 s (`woof hex global` and a bare `woof hex forecast` alike); an explicit `--dt` runs as given |

---

## Requirements

| | |
| --- | --- |
| Python | 3.11 or newer |
| GPU | A CUDA device: any NVIDIA card of compute capability 7.5 (Turing) or newer runs, on a CUDA 12 or CUDA 13 driver, with no flag. sm_86 and sm_89 are anchored (the numerical contract measured again on that architecture), sm_120 is the proven contract floor where it was first measured, and any other card runs unanchored and its receipt says so; `woof hex doctor` names your card and its status. Memory is set by mesh size *and* by the card, and not by a line through the two: the footprint is a per-card core, plus the Grell-Freitas and YSU column workspaces charged at `min(cells, tile)`, plus a per-cell term, all in `woof.hex.device_admission`. Measured 2026-08-26 on an RTX 5090 at the merged tip: the published 40,962-cell global mesh (x1.40962, about 120 km) peaked at **8,874 MiB, inside a 12 GiB card's budget**; the 163,842-cell mesh (x4.163842, about 24 km) peaked at **20,446 MiB**, and every free-memory gate admits at the measured floor (the prediction plus that card's own margin, **about 21.7 GiB free at x4**), so x4 remains in practice a 32 GiB-card configuration. The core is a property of the card, not the mesh, and the door reads the card's multiprocessor count at the decision rather than assuming a 5090: the 16 GiB and 10 GiB parts each carry their own measured row, and a card nobody has measured gets a derived row that is labelled derived. A card with its own per-allocation ledger can still supply it with `--device-fixed-mib` / `--device-bytes-per-cell`. |
| CUDA | CuPy matching your driver's CUDA major. There is no way for pip to detect it, so you choose (below). |
| Engine | The `woof` distribution, 2.9 (`woof>=2.9.11,<2.10`), which pip resolves for you: the WOOF physics seam, and the bundle that carries the MPAS bridge binaries the doors drive (including `met_intermediate` for init-source and the renderers for products). The forecast runs on the installed engine; a git clone at `v2.9.0` is optional (receipts then name its commit). See *The engine pin*. woof 2.8.9 cannot serve 0.4: it has no `met_intermediate`, its renderer knows none of the product windows, and it has no `rw_hex_points`. |
| Platforms | Linux x86_64 with glibc 2.28 or newer installs the prebuilt wheel (`manylinux_2_28`: RHEL 8 and 9, Ubuntu 20.04 and later). Anywhere else pip builds the sdist, which needs a Rust toolchain (`cargo`) for the bundled helpers; the observation scorer is built only when `GPUWM_HEX_ENGINE_SOURCE` names a woof source checkout, and otherwise `woof hex score` refuses by name and `doctor` reports the gap (forecasts are unaffected). There is no prebuilt Windows wheel. |
| Assets | A mesh grid file, a matching static file, and (for the init door) a vertical-grid declaration, normally a `--vertical-spec` JSON; a native-minted init file as a capsule is the compatibility mode. `woof hex global` fetches its published meshes itself; otherwise **woof hex ships none of these**. See *Assets you must supply*. |
| Rust binaries | The field engines (`rw_mpas_init`, `rw_mpas_static`, `rw_mpas_convert`, `rw_mpas_lbc`, `rw_wrfbatch` and the others) come from woof's bridge bundle or a woof source build. Eight of them ship in this wheel, with its own helpers: `rw_mpas_geometry`, `rw_mpas_stream`, `rw_mpas_static`, `rw_mpas_lbc` (native case-9 boundaries), `rw_igra2` (IGRA2 radiosondes for `woof hex score`) and the RRFS fork dycore's three programs. See *Getting the Rust engines*. |
| MPAS physics suites | The MPAS-A framework uses the wheel's helpers and pinned DATA from `physics-resources-v0.4.1`. `woof hex physics-resources fetch --from DIR` stages an offline mirror; `WOOF_HEX_PHYSICS_RESOURCES_FROM=DIR` selects it at first use. RRFS `hrrrv5`, both RRFS alias suites and New Tiedtke run on the same terms; selections without resident CUDA table entries require a site-built native library. See [MPAS physics resources](docs/physics-resources.md). |

### Install

```sh
pip install recast-woof
pip install "recast-woof[gpu]"         # the CUDA lane; = "recast-woof[gpu-cu12]"
```

**CUDA 12 and CUDA 13 drivers both run this port.** `[gpu]` installs
`cupy-cuda12x`, which runs on a driver serving CUDA 12 or CUDA 13 (a driver
runs its own CUDA major and every older one). On a CUDA 13 driver,
`[gpu-cu13]` installs `cupy-cuda13x`, the toolchain the byte-identity
results in this release were measured with. `cupy-cuda13x` on a CUDA 12
driver stops at the first device call with `cudaErrorInsufficientDriver`;
`woof hex doctor` reads the driver's CUDA major, reports a wheel the driver
cannot run as a gap, and names the extra for that driver. Whether the card is reachable is the installed NVRTC's
question: an RTX PRO 6000 or RTX 50-series card (compute capability 12.0)
needs CUDA 12.8 or newer, which both extras install.

```sh
woof hex version      # what is installed, and where
woof hex --help       # the doors
woof hex doctor       # what this install can actually reach
```

**Run `woof hex doctor` first.** A wheel for this project is deliberately
partial and says so: the engine's Rust programs, the CUDA runtime and the mesh
assets cannot travel inside it. Doctor checks each of those estates for real
and prints, for every gap, the exact command that closes it on your platform.
`woof hex doctor --explain` prints the evidence and the whole pasteable
remedy block; `--json` emits the same findings as data. It exits 1 while any
required item is missing, so it works as a gate in a script.

### Workflow commands

- `woof hex country` runs the stages selected by a country row.
- `woof hex surface-series` builds timed SST and sea-ice forcing.
- `woof hex ensemble-prepare` prepares source members and lagged cycles;
  `woof hex ensemble` runs the member forecasts and their products.
- `woof hex products RUN` draws maps, soundings and meteograms for a run;
  `woof hex export RUN` writes regridded NetCDF frames.
- `woof hex run-plan` prints a global row's commands and paths;
  `woof hex sources` lists initialization sources and global rows.
- `woof hex global-cycle` repeats a global row with schedule, retention
  and status records.
- `woof hex moving-ensemble` runs the members of a prepared storm-following case,
  each with its own SPPT/SKEB forcing ([moving-region ensembles](docs/moving-ensemble.md)).
- `woof hex chem-sources` lists, fetches and prepares smoke, composition, dust and
  sea-salt source files for aerosol transport ([chemistry sources](docs/chemistry-sources.md)).

### Kernel packs

Kernel packs are pinned assets of the `kernel-packs-v0.4.9` release, for CUDA 12.9, 13.0 and 13.4
NVRTC, and hold the WOOF and MPAS physics kernels.
`woof hex fetch-assets --list` lists them. Run
`woof hex fetch-assets` to stage them in the cache, or
`woof hex fetch-assets --from DIR` to verify and stage an offline copy.
`WOOF_HEX_ASSETS_FROM=DIR` selects an offline source directory for first use;
`WOOF_HEX_ASSETS_URL` selects a release-asset mirror.
`WOOF_HEX_ASSETS_FIRST_USE=off` disables first-use downloads.
`--cache DIR` chooses a cache and `--only NAME` selects an asset.
Each fetch checks the pinned size and SHA-256 and writes a receipt.
`woof hex doctor` reports staged and on-demand assets without fetching.
A run without a matching pack compiles its kernels through NVRTC.

### The import namespace

The distribution is `woof hex`, every command you type is `woof hex`, and the
**import namespace is `woof.hex`**: `import woof.hex`, not `import gpuwm_hex`.

It was `mpas_port` through 0.1.1 and the rename landed in 0.2.0. `woof.hex`
names what the package is built around, the hexagonal Voronoi mesh and its
core, and it matches the distribution name. **This was a breaking change for
code that imported the package directly.** There is no `mpas_port` alias shim:
rewrite `import mpas_port` as `import woof.hex`; every module path underneath it
is unchanged.

*(The `hex` in the name is the hexagonal Voronoi mesh the model runs on.)*

### The engine pin

woof hex 0.4.2 requires woof 2.9 (`woof>=2.9.4,<2.10`). The sixteen seam
files were measured against woof 2.9.0 at its public tag `v2.9.0` (`0db04746`),
from the three platform wheels PyPI serves
(`verification/engine-verdicts/repin-290-20261014.json`).

- **The range admits woof 2.9.4 and newer patches below 2.10.** 0.3.3's one-patch range replaced
  woof 2.8.7 with 2.8.0; that is why the range changed.
- **An engine inside the range whose radiation helper set this release has
  not measured is refused at the forecast door, by name:** the installed
  version, the version this release was measured on, and both remedies
  (`pip install "recast-woof==X"`, or the woof hex release that admits the
  installed engine). The receipt records the installed engine and its seam digests.
- **The range is derived, not typed.** `woof.hex.engine_pin` computes it from a
  table of measured verdicts, one row per published woof: on PyPI or not,
  whether its bytes satisfy the sixteen-file seam manifest, and whether its
  source builds offline. `tools/measure_engine_verdicts.py` takes that
  measurement from the published wheels and tags and splices the table into
  `woof.hex.engine_pin` (`--table` prints it), and
  `tests/test_packaging_declaration.py` fails if `pyproject.toml` and the
  derivation disagree. The ceiling is exclusive: the next minor is never
  resolved onto, because nobody has measured it.
- **The seam check in `woof hex doctor` is information, not a gate.** It
  reports how many of the sixteen seam files moved on the installed engine,
  and the forecast still runs.
- **An engine that is not on PyPI can be admitted, by measurement only.**
  `tools/measure_engine_verdicts.py --admit` hashes the sixteen files inside a
  private build's own wheels and source tree, refuses if they disagree, and
  writes a row to `verification/engine-verdicts/admitted-engines.json` that
  `--splice-admitted` renders into `woof.hex.engine_pin.ADMITTED_ENGINES`. The
  published distribution ships that table empty.

**Installing `woof` is all the forecast needs from the engine.** Since 0.3.2
the `forecast` door runs on the installed distribution: the pinned seam files
resolve from `site-packages`, and the run names the executed seam by the
install's version and the SHA-256 of pip's `RECORD` for it. `--gpuwm-checkout`
stays as an option: pass a clone of `v2.9.0` and the receipt records its HEAD,
tree and dirty paths instead. Any other directory that is neither a git
working tree nor the installed engine (an unpacked tarball, a copied
`site-packages`) is refused by name, because it names neither a commit nor a
record.

*(The woof source tree publishes too: `github.com/recastsystems/woof`
carries the release tags with the full tree, `docs/mpas-seam.md` and
`woof/core/mpas_column_batch.py` included.)*

**The init and render doors do not need any of that.** They import no `woof`
at all; they drive Rust binaries.

---

## Assets you must supply

Two doors fetch what their rows name: `woof hex global` fetches the
NCAR-published quasi-uniform meshes its rows use, checked against pinned
SHA-256, and `woof hex init-source` fetches the meteorological input of its
source rows. Beyond that, woof hex has **no fetch path for any of these**, and
no `woof hex fetch` command exists.

**1. A mesh grid file** (`x1.2562.grid.nc`, `x4.163842.grid.nc`, …). Meshes
come from the MPAS-Atmosphere project's published mesh downloads, from the
`MPAS-Tools` mesh generator, or from the engine's own `woof mesh` door, which
generates an icosahedral Goldberg grid **and its matching static** for a named
refinement region, sized against the measured footprint of a named card.
woof hex reads and validates meshes and registers a new pair as table work;
the generator lives in `woof`.

**2. A matching static file** (`*.static.nc`) carrying terrain, land use, soil
category and vegetation interpolated onto that mesh. Static files are produced
by the engine's `rw_mpas_static` (the same writer `woof mesh` drives, against
a WPS geographical dataset); a static file native `init_atmosphere_model` wrote
is read as it is. With
`--native-init --native-gwd-static` it writes the native static stream and the
native gravity-wave-drag statistics, word for word as native MPAS-A v8.4.2
writes them apart from the declared residue (see *Initialization, statics and
the services around the forecast*). woof hex does not build static files
itself; `woof hex global` drives `rw_mpas_static` for its rows. The grid and
static must be the same mesh: the doors cross-check `nCells` and refuse a
mismatched pair by name.

**3. For the init door: a vertical grid, one of two ways.** The Rust init
engine does not invent the vertical grid.

- **Native-free (the normal path):** pass `--vertical-spec` with a
  `gpuwm-hex.vertical-spec/v1` JSON declaration; the door constructs the
  v8.4.1 vertical contract itself and writes a durable vertical artifact with
  its receipt. No native file is read. A new vertical configuration is a JSON
  file, not a code branch. Measured against a native golden on 2026-08-24: the
  mint is schema-complete (134/134 variables), the constructed vertical sits
  within 3.9 mm of native `zgrid` with the per-field cost quantified, and the
  minted init runs the dycore; boundary in
  [`docs/native-free-init-admission.md`](docs/native-free-init-admission.md).
- **Native-capsule compatibility mode:** pass `--capsule`/`--reference` naming
  a native-minted init-class file; the door reads `zgrid, zz, fzm, fzp, dzu,
  rdzw, zb, zb3` and the smoothed terrain out of it and asserts the capsule's
  `zgrid` checked for exact array equality against the reference before trusting a single level.

The two modes are mutually exclusive and there is no hidden native read.

**4. Meteorological input**: built by `woof hex init-source` from a row of
its source table (GFS, ERA5 and the rest: the engine's Rust `met_intermediate`
decodes the GRIB, no Fortran). A WPS intermediate file you already have is read
as it is.

### Getting the Rust engines

The doors are orchestration only; the field data is handled entirely by Rust
binaries. The wheel carries sixteen compiled programs of its own in
`woof/hex/libexec`:

- `hex_contract_hostprep`, the host-preparation helper;
- `gpuwm-hex-netcdf-copy`, the init door's CDF-5 vertical template copier;
- `hex_obs_scorecard`, the observation scorer behind `woof hex score`;
- the four table decoders `woof hex physics-resources build` uses:
  `lookup_decode` (a shared library), `table_decode`,
  `rw_mpas_rrtmg_lw_tables` and `rw_mpas_rrtmg_sw_tables`;
- `rw_mpas_physics_pool`, the MPAS route's Registry pool reader;
- eight engine bridges that no released woof wheel carries in the version
  this release needs: `rw_mpas_geometry` (speaking `hex-vertical-v1`, for the
  init door), `rw_mpas_stream` (MPAS streams and restart), `rw_mpas_static`
  (statics for meshes generated at a point, which hold the binary64
  coordinates they declare), `rw_mpas_lbc` (boundary files equal to native
  `init_atmosphere` case 9), `rw_igra2` (the IGRA2 radiosondes `woof hex
  score` reads for a case NOMADS no longer serves) and the RRFS fork dycore's
  `rw_mpas_rrfs_connectivity`, `rw_mpas_rrfs_damping` and
  `rw_mpas_rrfs_deformation`. Their bytes are pinned in
  `woof/hex/data/engine-bridges.json`; see
  [engine bridges](docs/engine-bridges.md).

That is the prebuilt Linux wheel. A build from the sdist (it needs `cargo`)
always builds the host-preparation helper and the table decoders; the pool
reader and the template copier only where the netCDF-C development files are
installed; the scorer only when `GPUWM_HEX_ENGINE_SOURCE` names a woof source
checkout; and the eight bridges only when `WOOF_HEX_ENGINE_BRIDGES` names
the pinned prebuilt files or `GPUWM_HEX_ENGINE_SOURCE` names a woof source
that carries all of them. A door that needs one that was left out refuses by
name.

The other field engines come from the `woof` source tree and are staged onto
your machine one of two ways.

**The short way.** `woof` publishes prebuilt bundles, and one command stages
the whole set: `rw_mpas_mesh`, `rw_mpas_static`, `rw_mpas_init`,
`rw_mpas_convert` and `rw_wrfbatch`:

```sh
woof fetch-bridges
```

It writes into `~/.woof/bridges`, which woof hex reads directly. No
environment variables to set. This works wherever a bundle is published for
your platform; `woof hex doctor` tells you whether it worked.

**The build.** On a platform with no published bundle, build from a checkout:

```sh
cd <woof checkout>/tools/rustwx
cargo build --release --offline --locked -p rw-mpas       # rw_mpas_init, rw_mpas_convert
cargo build --release --offline --locked -p rw-wrfbatch   # rw_wrfbatch
```

Then point the doors at what you built:

```sh
export WOOF_HEX_RW_MPAS_INIT=<woof>/tools/rustwx/target/release/rw_mpas_init
export WOOF_HEX_RW_MPAS_CONVERT=<woof>/tools/rustwx/target/release/rw_mpas_convert
export WOOF_HEX_RW_WRFBATCH=<woof>/tools/rustwx/target/release/rw_wrfbatch
```

#### The resolution ladder

Every door resolves every engine the same way, best first:

1. the door's own flag (`--engine`, `--convert-exe`, `--renderer-exe`);
2. this distribution's variable, then any older spelling it has carried:
   `RW_MPAS_INIT`, `MPAS_PORT_RW_MPAS_CONVERT` and `MPAS_PORT_RW_WRFBATCH`
   still work and always will, because a rename must never invalidate an
   install line that already works;
3. `woof`'s own variable (`WOOF_RW_MPAS_INIT` and siblings);
4. this wheel's own `woof/hex/libexec`, which holds the seven bundled bridges
   (`rw_mpas_geometry`, `rw_mpas_stream`, `rw_mpas_static`, `rw_mpas_lbc`,
   `rw_mpas_rrfs_connectivity`, `rw_mpas_rrfs_damping` and
   `rw_mpas_rrfs_deformation`);
5. `woof`'s bridge directories, which is where `woof fetch-bridges` stages;
6. `PATH`.

A flag or an environment variable naming a missing file is a **hard error**,
never a fall through to the next rung: a ladder that silently skips a broken
setting runs the wrong engine build and reports success. When nothing is
found, the refusal names every rung it searched and both commands above.

#### What `woof fetch-bridges` supplies

The MPAS bridge binaries (`rw_mpas_init`, `rw_mpas_convert`, `rw_mpas_mesh`,
`rw_mpas_static`) entered woof's bundle at 2.5.3, and every engine the
declared range admits carries them, so pip cannot resolve you onto an engine
that strands the doors. You are not asked to track that. `woof hex doctor`
asks the woof you actually have which artifacts its bundle declares, and
every refusal offers the staging command only when it can really deliver the
file: a remedy that cannot work is worse than no remedy, because you run it,
it succeeds, and the door still refuses.

The MPAS route needs three more native pieces that are not in the bundle: the
native framework library, the Registry pool reader and an MPAS-Model source
tree. `woof hex physics-resources fetch` supplies all three (see *Running an
MPAS physics suite*). MPAS streams and restart need `rw_mpas_stream`, the
real-data initialization needs an `rw_mpas_geometry` that speaks
`hex-vertical-v1`, and `mesh-plan --point --generate` needs an
`rw_mpas_static` that gives a generated grid a static holding the binary64
coordinates it declares (woof 2.9.0's does not, and the door refuses its
statics by name). No released woof carries any of the three, so this wheel
carries them in `woof/hex/libexec`, pinned by SHA-256 to the woof commit they
were built from ([engine bridges](docs/engine-bridges.md)). A flag or variable
still wins over the bundled copy (`WOOF_HEX_RW_MPAS_STREAM` names the stream
program), and a door that needs one refuses by name when it is absent or too
old.

---

## Door 1: `woof hex init`

Build initial conditions without native Fortran `init_atmosphere_model`.

```sh
woof hex init \
  --met        WORK/MET:2025-03-14_12 \
  --static     assets/x4.163842.static.nc \
  --grid       assets/x4.163842.grid.nc \
  --vertical-spec verification/vertical-specs/tc55-v1.json \
  --out        run/init.nc \
  --start-time 2025-03-14_12:00:00 \
  --nfglevels 38 --nfgsoillevels 4 \
  --extrap-airtemp lapse-rate --use-spechumd no \
  --theta-adv-order 3 --coef-3rd-order 0.25 \
  --virtual-factor reproduce-fortran \
  --deep-soil-moisture reproduce-fortran \
  --landuse-table MODIFIED_IGBP_MODIS_NOAH \
  --frac-seaice yes --tsk-seaice-threshold 100.0 \
  --oned-underflow preserve
```

Writes `run/init.nc` and `run/init.nc.provenance.json` (SHA-256 of every input,
the engine binary, the argv, the engine's own receipt and the output), prints a
JSON summary, exits 0.

**Every physics switch is required and has no default.** That is deliberate:
each one changes the numbers in a file that opens cleanly and reads plausibly
either way, so a default would be a silent wrong answer. Each refusal prints the
native namelist key it corresponds to, so a captured `namelist.init_atmosphere`
transcribes without guessing. Full detail, including the sixteen named refusals:
[`docs/init-door.md`](docs/init-door.md).

What the door accepts is not a promise but a measured table:
[`docs/source-matrix.md`](docs/source-matrix.md) drives every source in the
RW-WPS registry with real bytes through intermediate, init and a
five-composite-step forecast, and records one verdict per source, a green
run with receipts, or the chain's refusal, verbatim.

`--vertical-spec` is the native-free path (see *Assets you must supply* and
[`docs/native-free-init-admission.md`](docs/native-free-init-admission.md));
`--capsule`/`--reference` with a native-minted init file is the compatibility
mode. The two are mutually exclusive.

## Door 2: `woof hex render`

History in, product PNGs out, entirely through the Rust path.

```sh
woof hex render \
  --history history.2025-03-15_00.00.00.nc \
  --mesh    assets/x4.163842.grid.nc \
  --out     ./png \
  --simulation-start 2025-03-14_12:00:00 \
  --products all
```

`rw_mpas_convert` resamples each history frame onto a render window and
`rw_wrfbatch` draws the products. PNGs are filed at render time into
`<out>/<domain>/<product>/<valid-day>/`, never flat, with a
`render-manifest.json` beside the tree carrying engine digests, per-frame
results and the exact invocations. Scratch lives in a sibling of `--out`, never
inside it, and is deleted after a clean run.

There is no fallback plotter. If the Rust renderer is absent the door refuses by
name; it never draws a weather field in Python. Full detail:
[`docs/render-door.md`](docs/render-door.md).

## Four smaller doors

`woof hex mesh-check --grid <grid> --static <static>` validates a mesh pair
before anything expensive touches it, and refuses a defective pair by name.
`woof hex oracle-gate` replays the source-extracted Fortran fixtures against
a mesh. `woof hex cull` cuts a limited-area grid, static and initial
condition out of a global case, which is what makes the regional lane cheap:
grid, static and init in about a second where a native regional init took
775 s. `woof hex pair <forecast flags> --source-table <table> --pair-out
<dir>` runs one forecast twice, without and with a point-source table, and
differences the matched frames: the control is derived from the same command,
the door refuses unless the two legs differ in the table alone, and its
summary says a difference between the legs is a model result, not a
measurement of the atmosphere. All four are listed by `woof hex --help`.

## The forecast lane

`woof hex forecast` is a door. It binds a registered mesh against its pinned
bytes, asks the card whether the run fits **before** the run starts, refuses
by name with numbers when it does not, and prints the render command when it
passes. `--preflight` gives the same answer without integrating anything.

It runs from an install. The drivers ship inside the package
(`woof/hex/drivers/`), and the physics seam is the installed `woof`, named in
the receipt by version and `RECORD` digest (*The engine pin*). Through 0.3.1
both were checkouts (a woof hex source checkout for the drivers under
`tools/` and a woof git clone for receipt provenance) so a wheel install
could not forecast at all. `tools/` keeps same-named entries for the three
drivers so older scripts keep working; they reach the packaged modules.

Its inputs are a mesh, a static file and an init. The registry makes the first
two table work; the init door makes the third. The card question is answered
from `woof.hex.device_admission`, on the 170 SM part the registered
40,962-cell mesh peaks at 8,874 MiB and the 163,842-cell x4 at 20,446 MiB,
which with that card's own margin puts x4 in practice on a 32 GiB card.

### Limited area, behind lateral boundary conditions

`--lbc-dir` runs the same full physics stack on a **culled** regional mesh
driven from its parent. A cull of a placed fine grid is around eleven times
fewer cells than the global refined mesh covering the same storm, and it runs
on a 10 GiB card. The outermost seven rings are boundary data driven from the
parent every step rather than the model's own answer, and products drawn over
the whole domain include them.

### A fine core at any point, from HRRR

`woof hex mesh-plan --point LAT,LON --fine-dx-m 937.5 --radius-km 100 --card 32gb`
writes the registered 937.5 m ladder spec centred on the point, prices the
global parent through the generator and the limited-area cull on the
admission surface's own row for the card, and with `--generate` builds the
pair, admits it, registers it as a RUNTIME ROW in `mesh-rows.json` beside the
mesh (never a checkout edit), cuts the cull and mints its vertical.
`woof hex intermediate --source hrrr` resamples HRRR onto the regular
lat-lon WPS intermediate the init and boundary engines read, through the
engine's own decoder and interpolation operator; `woof hex lbc` builds the
boundary series from the hourly intermediates. The whole chain, with what it
measured, is [`docs/hex-point-hrrr.md`](docs/hex-point-hrrr.md).

### Placement and cycling

`woof hex swath` decides where a fine grid should go: detection on
sea-level-reduced pressure, a declarative threat grammar, commensurable
ranking across phenomena that carry different units, and hysteresis so a grid
does not chase noise. `woof hex cycle` runs the loop: a coarse parent, the
corridors placed inside it, and the next cycle's re-detection. See
[`docs/cycle-door.md`](docs/cycle-door.md).

### Local time stepping, opt-in

On a variable-resolution mesh most columns are far coarser than the finest one,
and the acoustic sub-step is sized for the finest. `--local-timestep` lets a
coarse column take fewer, longer acoustic sub-steps, chosen from the grid
file's own `dcEdge`.

```bash
python src/hexcore/drivers/run_cuda_v841_forecast.py \
    --grid x4.163842.grid.nc --static x4.163842.static.nc \
    --init <init>.nc --hours 6 \
    --cache-root <cache> --output <out> \
    --local-timestep
```

`--local-timestep-rates` sets the ladder (default `1,3`, two classes) and
`--local-timestep-buffer-rings` the width of the finer-rate buffer around a
class boundary (default 1).

**It is off by default and that is deliberate.** Native MPAS-A v8.4.1 has no
local time stepping: `Registry.xml:64-68` offers SRK3 only, and `dt_dynamics`,
`rk_timestep`, `rk_sub_timestep` and `number_sub_steps` are scalars at
`mpas_atm_time_integration.F:2053-2092`. There is therefore no native-equivalent
implementation and there never can be. A user who turns it on takes a declared
divergence from native; a user who does not gets the pinned arithmetic
unchanged, byte for byte. This is a performance feature, not a correctness
remedy, so opt-in is the correct shape for it.

Two things follow from the mesh rather than the flag:

- On a **quasi-uniform** mesh every column lands in one class, so the option is
  inert and the run is bit-identical to a default run. Measured on x1.40962:
  all 40,962 cells in class 0, zero interface edges, identity permutation, and
  every history frame SHA-256-identical.
- The ladder is not free. A rate must divide every RK stage's acoustic
  sub-step count, and the released `(1, 3, 6)` schedule admits `1` and `3`
  only: `2` and `4` do not divide the RK2 stage's three sub-steps.

Class boundaries are refluxed, so mass and passive water vapour are conserved
to binary32 rounding rather than exactly; see *Local time stepping is
flux-conservative, not exactly conservative* under **Limitations**. A run with
the option on stays bit-reproducible run to run, so the dual-run byte
comparison that screens for memory corruption on cards without ECC still
works.

#### What it costs, measured

**On the published x4.163842 mesh the option does not pay, and the ceiling
says it cannot.** Dry lane, 163,842 columns, 55 levels, RTX 5070 Ti, one model
hour:

| | wall seconds per model step |
|---|---|
| default path | 1.268 |
| `--local-timestep` | 1.283 |

0.988x: about 1% slower. The reason is measurable rather than mysterious.
Re-timing the same arm at 12 acoustic sub-steps instead of 6 gives the cost of
one sub-step directly, and from it **the acoustic loop is 23.5% of a model
step**. This mesh admits a 23.0% acoustic saving, so the best a whole step
could do is **1.057x** before the option pays for its own bookkeeping, and the
bookkeeping is larger than that.

The ceiling is a property of the released schedule, not of this mesh. A rate
must divide every stage's sub-step count, so `(1, 3, 6)` admits only rate 3,
and a rate-3 column still runs 4 of the 10 sub-steps a fine column runs. Even
in the limit where **every** column is coarse the saving is 60% of the acoustic
loop, which at a 23.5% share is **1.16x for the whole step**. That number is
the real cap on this feature as the port stands.

What would move it: a mesh with a much steeper resolution gradient (the
generated 15 km-in-136 km box mesh classes to a 49.7% acoustic saving, a 1.13x
ceiling), and an acoustic schedule whose sub-step counts admit a rate above 3.

#### On a limited-area cull, measured 2026-09-14

On the culls the doors make the option is inert by construction: `mesh-plan`
cuts at 1.35 times the fine radius and the cull door cuts a cap, so the cells
whose spacing reaches three times the finest edge all sit in the seven driven
boundary rings, every driven cell is held at rate 1, and the one-cell buffer
demotes the rest. Measured on the RTX 5090 (point cull, 43,884 cells x 55
levels, HRRR-driven, 1 h): `--local-timestep` executes one class and both
history frames are byte-identical to the default run, at 0.992x the step
(inside run-to-run noise). Forcing an interface into the interior with the
`tools/lts_forced_classing.py` instrument (6,206 cells at rate 3, 1,456
interface edges, an 8.5 % arithmetic acoustic saving) makes the step 1.071x
SLOWER: every dycore kernel on a mesh this size runs in one wave on the card,
so the fine class costs what the whole mesh costs and each coarse-class
launch is a further wave. The corridor cull (40,520 cells, GFS-driven) gives
the same numbers (0.989x, 1.068x). What the forced interface does to the
fields is in [`docs/local-timestep-lam.md`](docs/local-timestep-lam.md).

#### The full-step form, measured to a no-go

The stronger form of this feature (every dycore kernel launched per rate
class, whole RK steps advancing at `rate * dt`) is not capped by the acoustic
share. Its arithmetic prize on real meshes is large: counting cell-steps under
a `(1,2,4)` ladder with one buffer ring, the published x4.163842 mesh admits
**1.47x** and the generated 15 km-in-136 km box mesh **2.73x**. Whether the
card can collect that prize is a question about launch cost: a kernel launched
over a class must cost proportionally less than a launch over the whole mesh,
and below the card's occupancy knee it does not.

`tools/probe_lts_fullstep_projection.py` measured it (RTX 5070 Ti, the port's
own pinned kernels and their landed index-list derivations, the real meshes'
own class index lists):

| mesh | cells | arithmetic prize | measured projection |
|---|---|---|---|
| x1.40962 uniform | 40,962 | 1.000x | 1.000x |
| x4.163842 published VR | 163,842 | 1.467x | **1.254x** |
| 15 km-in-136 km box | 38,857 | 2.725x | **0.979x** |

The uniform row is the instrument's known-answer: one class, and the
index-list launch prices within 0.2% of the pinned kernel. On the box mesh the
prize inverts into a projected slowdown because the whole mesh is already
below the occupancy knee (doubling its cells costs only 1.26–1.30x on the
cell kernels) so a launch over its 5,562-cell fine class costs nearly what
the whole mesh costs, and the fine class launches four times per macro step.
The steeper the refinement, the smaller the classes, the harder the floor
bites: `(1,2,4,8)` projects worse (0.904x), not better. Only the x4-size mesh
keeps every class above the knee, and even there the trio projects 1.254x
**before** interface bookkeeping; the shipped acoustic form measured its own
bookkeeping at about 7% of a step on that mesh, and the full-step form pays a
cost of the same character plus time interpolation at class boundaries.

So the verdict, measured rather than estimated: rebuilding the whole step
loop per class would buy roughly 1.1–1.2x on the one registered mesh large
enough to profit, and a slowdown on the small steep meshes the prize was
supposed to come from. Not worth a rewrite of every kernel's launch path.
What could reopen it: meshes an order of magnitude larger, where every class
sits above the occupancy knee, or overlapping different classes' independent
work in concurrent streams, neither is measured here, and the probe is the
instrument to re-run when either becomes real.

### The surface/PBL cadence, selectable

`config_bldt_seconds` is welded to `config_dt` by default: the surface layer,
the land-surface model and the PBL are called on every model step, as the
native x4 v8.4.1 reference ran them. At the 120 s that reference used that is
30 calls an hour; at the 5 s a sub-kilometre mesh declares it is 720, and the
per-step profile of 2026-09-13 (RTX 5090, the 43,884-cell point mesh) put the
seam's surface/PBL phase at a fifth of every step.

`--pbl-cadence SECONDS` calls the stack once every `SECONDS / dt` steps and
holds the tendency it produced on the steps between -- the engine's own
positive-`bldt` path, ARW's `stepbl`: the held rate is applied unchanged on
every non-due step, and the surface layer and land-surface model integrate
their own state with the cadence as their step. Radiation keeps its own
cadence either way (1800 s by default, MPAS-A's); it was never welded to `dt`.

```bash
woof hex forecast --mesh <row> ... --pbl-cadence 60
```

**It changes the forecast, so it is selectable and never silently the
default.** `auto` is the weld and changes no run. An explicit cadence records
itself as `source: "explicit"` in the run receipt with its calls per hour and
the steps held between calls, and the driver receipt carries the seam's own
count of due and held steps against the count the declared cadence predicts
(`physics.cadence` in `forecast-receipt.json`), so a held arm that reproduced
the welded arm exactly would be caught by its own receipt.

Two refusals, each by name and before any card is touched: a cadence that
is not a whole number of steps (the message names both numbers, the multiples
of `dt` on either side and `auto`; nothing is rounded), and a request that is
not a number of seconds. Whether a held cadence matches anchor evidence is
recorded, not required: at an anchored `(dt, cumulus)` the receipt names the
row *derived* from the welded anchor, whose physics band is stamped NOT
MEASURED at the held cadence rather than borrowed.

The measurement of record is in `docs/hex-point-hrrr.md`: the every-step
arm against 30, 60 and 120 s on the point mesh (43,884 cells, dt 5 s, HRRR
15Z, RTX 5090, one session). Every held cadence buys the same five per cent
per step (median composite step 0.2994 s to 0.2842 to 0.2845 s) and every
one changes the one-hour forecast well past the 2.6.5 to 2.7.3 seam
movement (theta up to 1.25 K at 30 s, 2.95 K at 120 s, against 0.139 K), so
none qualified as the default and the weld stays. This tree at `auto` wrote
history byte-identical to the untouched tree's.

### Forecast presets

`woof hex forecast --preset NAME` selects a row of `woof.hex.forecast_preset`;
the door turns the row into the cadence knob above, so a preset is data and
adding one is adding a row. `reference`, the default, is the proven
configuration. `fast` holds the surface/PBL stack for up to 30 s between calls
(six steps at dt 5 s). An explicit `--pbl-cadence` wins, and the receipt says
which decided.

Graded against observations (MRMS reflectivity, Stage-IV hourly
precipitation, ASOS 2 m temperature, dewpoint and 10 m wind) on four 3 h
HRRR-driven forecasts of the point cull, `fast` took about 6 per cent less
time per step and scored the same on reflectivity and precipitation
thresholds, and it forecast 10 m wind worse on all four cases and 2 m
dewpoint and temperature worse on three. So `reference` stays the default and
`fast` is there for when time matters more than the near-surface forecast.
The same measurement is in `docs/hex-point-hrrr.md`.

### Running an MPAS physics suite

**MPAS physics is the default (0.4.6).** Every door (`forecast`, `global`,
`mesh-plan --run`, `regional`, `country`, `moving`, ensembles, `cycle`, Hex
with Regional parents, CUDA JEDI backgrounds) runs MPAS-A's own physics on the
MPAS route when the run names no physics. The suite follows MPAS-A practice by
mesh (`src/hexcore/data/profiles/physics-default.v1.json`):

| Mesh | Suite | Why |
|---|---|---|
| finest cells under 10 km (regions, zoom globes, storm-following, any variable-resolution mesh) | `convection_permitting` | MPAS-A's recommended suite where convection-permitting cells are used, including variable-resolution meshes |
| quasi-uniform, 10 km and coarser | `convection_permitting` | MPAS-A tested it from several hundred km down to 3 km (scale-aware Grell-Freitas); `mesoscale_reference`, MPAS's Registry default there, does not run on the GPU yet (refused by name) |

WOOF physics stays one flag away: `--physics-suite arwen_mpas_column` runs
exactly what a bare run ran before 0.4.6 (on `global`, `--physics-row arwen`).
A `--physics KEY=VALUE` override, an explicit `--convection` scheme or a
`--physics-backend` row with no suite also selects the WOOF route.
`woof hex physics list` marks the default.

An MPAS suite can also be named, or read from a native `namelist.atmosphere`
and `streams.atmosphere` on the MPAS route (`--physics-backend mpasphys`):

```sh
woof hex physics-resources fetch      # optional; a forecast fetches missing resources
woof hex forecast --mesh x1.2562 --grid x1.2562.grid.nc --init x1.2562.init.nc \
    --physics-namelist namelist.atmosphere --streams streams.atmosphere \
    --physics-backend mpasphys \
    --out run/
```

- **The suite** is the namelist's `config_physics_suite`
  (`mesoscale_reference` or `convection_permitting`), with every scheme option
  read and checked as MPAS reads and checks it (`mpas_atmphys_control.F`).
- **The public v8.4.2 route** uses the Python framework and GPU physics ports.
  It uses the wheel's Registry and pool reader, plus pinned DATA tables,
  without a native MPAS library or a Fortran compiler. Public installs carry
  no native MPAS library. Missing tables are fetched on first use;
  `woof hex physics-resources fetch --from DIR` stages them offline.
  On this route `convection_permitting` runs on the GPU; other suites,
  `mesoscale_reference` included, are refused by name at the door until
  their GPU interface glue is ported.
- **Native comparison backends** require an explicit site build. Per-scheme
  backend selectors and `--physics-default-backend native-cpu` select that
  site's Fortran implementations. Build them with
  `woof hex physics-resources build --pin-set v8.4.2 --build-native-library`, then pass
  `--native-physics-library` or set `MPAS_NATIVE_LIBRARY`. RRFS-MPAS
  `hrrrv5`, both RRFS alias suites and New Tiedtke run on the ported fork
  framework with no native library. Selections without resident CUDA table
  entries require its separate site build,
  `woof hex physics-resources build --pin-set rrfs`, and a public install
  refuses the missing entry by name. See
  [MPAS physics resources](docs/physics-resources.md).
- **The mode** is correct unless `--physics-mode exact` is given (see
  *Correct mode* above).

`--preflight` checks routing and metadata without integrating.
`woof hex verify-mpas --deck DECK --receipt R.json --work-root DIR --ratchet RATCHET`
runs a deck and compares every stream word hex writes against native MPAS.
The published decks are in the sdist under `verification/gates/` (see its
README for the data layout). The headline deck runs every active physics
slot on the GPU, so it needs no native MPAS library, no MPAS-Model tree and no
compiler. Running it from a pip install:

```sh
pip install "recast-woof[gpu-cu13]"
woof hex physics-resources fetch
woof hex verify-mpas --deck verification/gates/decks/mpas-headline-convection-permitting.json \
    --ratchet verification/gates/ratchet.json --receipt hl-3.json --work-root hl-3-work
```

The comparator is the `verify` tool of woof's `rw_mpas_static` bridge,
found in the install. It loads the system netCDF-C library at run time
(`apt install libnetcdf-dev`, `dnf install netcdf`, or set `RW_MPAS_NETCDF_LIBRARY`). `--comparator` or
`WOOF_HEX_RW_MPAS_VERIFY` names another build. verify-mpas captures the run's
environment itself, before and after the forecast, and writes both captures
into the receipt: the engine (a git checkout's commit, or for a pip install
its installed files checked against their RECORD hashes), CUDA runtime,
NVRTC, CuPy, driver, card and every external math header. It refuses a run
whose numerics are not strict, whose capture is incomplete, or whose engine
files differ from their RECORD. A deck may pin a reference environment
(`strict_environment`). Identical words are GREEN in any environment, and the
receipt lists how the environment differs from the reference. Differing words
are a source regression only in the reference environment; elsewhere the
verdict is ENVIRONMENT, with the raw counts kept.

---

## Limitations

Read these before trusting a number.

### On the WOOF route, Grell-Freitas is a different generation from native

**This is about WOOF physics** (`--physics-suite arwen_mpas_column`). The MPAS suites, the default, run MPAS's own
Grell-Freitas, ported bit for bit (see *What is proven*).

**Declared, measured, and not a defect in the seam.** The seam-level non-parity
was closed: the four auxiliary forcing lanes, shallow-on, and per-cell `dx` all
reach the scheme the way native feeds them. What remains is that the port's
Grell-Freitas body is **WRF v4.6.1's Freitas-2018 generation**, while MPAS
v8.4.1's `module_cu_gf.mpas.F` is the **2013 ensemble fork**. Verified by source
count on both sides:

- native has zero occurrences of `dicycle` and zero of `tau_ecmwf`: those
  closures do not exist in it at all;
- native carries Fritsch-Chappell `AA0/1200s` closure members the port does not;
- native runs `c0=.002`; the port runs a temperature-scaled `c0=.004`;
- native's shallow scheme is non-precipitating (`c0=0`); the port's shallow
  scheme folds `prets` into `pratec`.

Closing this means porting native's `cup_gf`/`cup_gf_sh` bodies. That is a
program, not a seam edit, and whether it should close is the referee's call.
Every WOOF-route run receipt carries `gf_native_parity_claim: false` next to
`gf_declared_divergence` naming exactly the above, and so does every history
file written. It is a declared property of the WOOF route, judged against
observations. A user who wants MPAS's Grell-Freitas selects an MPAS suite.

### Local time stepping is flux-conservative, not exactly conservative

**Only reachable with `--local-timestep`, which is off by default.** When two
neighbouring columns advance on different acoustic sub-steps, the mass each
carries across the edge between them is integrated at two different rates and
the two no longer cancel. Mass is then created or destroyed at every class
boundary, and that is what kills local time stepping in practice.

The remedy here is Berger-Colella refluxing on the acoustic mass flux: the
coarse column keeps predicting with its own sub-step, the interface edge
accumulates the fine-rate integral minus that prediction, and the residual is
handed to the coarse column at the end of the RK stage. No term is dropped and
none is double counted, so the statement is **conservative in the flux and
approximate at binary32 rounding**: the coarse side applies one rounded sum
where the fine side applied a sequence of rounded increments.

Measured, not asserted. Dry lane, water vapour a passive scalar, published
x4.163842 variable-resolution mesh (5.46 max/min spacing, 1,063 interface
edges), one model hour:

| arm | dry-mass drift | passive-qv drift |
|---|---|---|
| default | 3.09e-11 | 9.02e-10 |
| `--local-timestep` | 2.58e-11 | 6.36e-10 |

against a 2.0e-8 bound. A full-physics qv budget cannot decide this: water
vapour there has sources and sinks and drifts about 1.8e-4 over six minutes
from the microphysics alone.

### Three bias-shaped differences between the WOOF route and native MPAS (declared divergences)

Not blockers, and not evidence of a broken dycore. They are the measured price
of running WOOF's physics instead of MPAS's on the WOOF route, stated so a
user knows what they are getting; the MPAS suites do not carry them. The
obs-skill comparison, which is the verification of record for WOOF's physics,
ran on 2026-08-25 and reaches two of the three; the register of
all three (the mechanism, the magnitude, the named observational referee and
now what that referee said) is
[`docs/declared-divergences.md`](docs/declared-divergences.md).

Measured over 24 h, two independent weather cases, two mixing regimes,
163,842-cell mesh, cell-aligned with no interpolation.

**When, exactly, because these three magnitudes carry no date of their own.**
They entered this repository already finished on 2026-08-20, and there is no
receipt directory, no card and no run commit for them anywhere in the tree.
What can be established is a ceiling: the commit that introduced them pinned
engine `629ddb6f0`, so they were measured at or before that pin. Ten engine
pin moves have landed since: `0d04db712` (2026-08-24), `26daaab7e`
(2026-08-25), `659962929` (2026-08-28), `7e34a48` (2026-08-31, woof
`2.6.0`), `df5f34c5c` (2026-09-01, woof 2.6.1), `636ab1b4b`
(2026-09-02, woof 2.6.3), `d60b883e7` (2026-09-02, woof 2.6.4),
`a7785421e` (2026-09-13, woof 2.7.3), `ed957997f` (2026-09-15, woof
2.7.4) and `0164ae0f2` (2026-09-29, woof 2.8.0). The **2.5.8** move is measured to change nothing: a four-arm byte A/B
on one card found the old and new pins identical on the atmosphere half of
the per-step fingerprint at all 31 steps and on 0 of 138 history variables, one mesh, one case, one hour. The
**2.6.0** move also crosses the executed seam (`woof/core/rrtmg_legacy.py`
among its three moved files) and is measured the same way at the same
scope: the x4 proof re-run at that pin wrote an F001 history byte-identical
to the 2.5.8 proof's. The
**2.6.1** move crosses it a third time, at its centre (the seam's own batch
driver `woof/core/mpas_column_batch.py` gains restart schema v2 and the P3
eight-species transport) and is measured the same way at the same scope:
the x4 proof re-run at that pin wrote all four snapshots (F000, F030, F001
and the restarted F001) byte-identical to the 2.6.0 proof's, so one F001
SHA-256 now spans three engines.
The **2.6.3** move touches the same batch driver (the aerosol-aware Thompson
species row) plus a config-relative file key off the physics path; its
one-hour byte arm was not run separately: the 2.6.4 arm below, taken
ninety minutes after that pin was superseded, covers it by inclusion. The
**2.6.4** move crosses the executed seam more widely than any since 2.5.8:
six of the sixteen pinned files moved between the two published wheels,
among them the phase-one driver `woof/core/physics.py` (the cumulus adapter
contract grew for a third cumulus scheme), the Grell-Freitas adapter
`woof/core/gf.py` and its kernel source `woof/core/kernels/gf.cu` (the
glibc float32 words moved into a header the loader prepends), the kernel
loader, the config loader and the restart identity table; the column batch
and the contract document did not move. Its one-hour byte arm is the x4
frozen-source proof re-run at the 2.6.4 pins: all four snapshots (F000, F030,
F001 and the restarted F001) byte-identical to the 2.6.1 proof's, so one
F001 SHA-256 now spans 2.5.8, 2.6.0, 2.6.1 and 2.6.4. The published
**2.6.5** moved none of the sixteen pinned files, so it is admitted
by the range without a pin move and the 2.6.4 arm covers it.
The **2.7.3** move (2026-09-13) crosses the executed seam more widely than
any before it (eleven of the sixteen pinned files moved between the
published 2.6.5 and 2.7.3 wheels) and it is the first move since 2.5.8
with **no x4 byte arm**: the re-pin was taken on a 16 GiB card holding none
of the pinned x4 assets, so whether the F001 SHA-256 that spans 2.5.8
through 2.6.5 survives it is **NOT MEASURED**, and given what follows it is
not expected to. What was measured instead is the seam itself, engine
against engine on fixed columns (`tools/measure_engine_seam_ab.py`,
receipts `verification/engine-verdicts/seam-ab-*.json`: one card, eight
columns, forty levels, twenty 120 s steps, WSM6, YSU, Noah-MP and legacy
RRTMG, with Grell-Freitas on one arm and off on the other). Two things
move. (1) **Grell-Freitas**, in the pinned `woof/core/kernels/gf.cu`: the
glibc `tgammaf` transcription is gone and the mass-flux shape normalisation
`fzu` now uses the engine's own correctly rounded gamma (the engine's own
figures: `fzu` moves by up to 4 ULP on 21 of the 26 (alpha, beta) pairs its
fixture reaches, and `xmb` moves by up to 7.3 per cent per ULP of `fzu`
through the closure), and the cloud-work integral's level loop now includes
the `kbcon` layer. On the convective profile, where GF fires on every
column, the convective rain bucket differs by up to 5.1e-4 mm after forty
minutes (0.21 per cent of the 0.245 mm the largest column accumulated), the
phase-one theta rate by up to 2.9e-5 K/s (8.7 per cent of the largest
rate), the vapour rate by up to 7.8e-8 kg/kg/s (15 per cent), and after
forty minutes theta by up to 9.4e-3 K and qv by 2.3e-5 kg/kg; with cumulus
off the same columns are byte-identical between the two engines on all 500
arrays captured. (2) **YSU**, in `woof/core/kernels/ysu.cu`, a file the
sixteen-file manifest has never pinned: a column whose first-guess
boundary-layer top sits below the second model level now stays in the
local-K regime for the whole step, and the post-scan clamp is WRF's two
independent statements, so a theta-li revival with `kpbl == 1` no longer
survives (the engine's reading of `bl_ysu.F90:703-728` and `:765-766`). On
the capped profile, where GF never fires and the two arms are
byte-identical, the lowest three levels move on five of eight columns from
step 7 on: the u rate by up to 1.9e-4 m/s² (55 per cent of the largest
rate), the theta rate by up to 1.8e-4 K/s (43 per cent), the vapour rate by
up to 5.6e-7 kg/kg/s, and after forty minutes theta by up to 0.139 K and qv
by 3.7e-4 kg/kg. That attribution is exact rather than inferred: 2.6.5 with
only that one kernel replaced by 2.7.3's is byte-identical to 2.7.3 on all
500 arrays of that profile. Both are the engine's changes, carried through
the seam without alteration; the port's physics is judged against
observations, not against the previous engine, and a pinned-file manifest
that does not reach the boundary-layer kernel is a named follow-up rather
than a claim this page makes.
The **2.7.4** move (2026-09-15) is the narrowest since 2.6.5 (two of the
sixteen pinned files moved between the published 2.7.3 and 2.7.4 wheels,
one added settings-map reader in the config loader and two keyword
arguments on the classic Thompson path, neither on the port's executed
path) and it is measured byte-neutral through the seam: the seam-level
A/B is 500 of 500 arrays identical on both profiles and both arms on two
cards, and the one-hour point-cull forecast under 2.7.4 wrote both history
frames byte-identical to the 2.7.3 frames of record on an RTX 5090,
with the cull's contract deck 8 of 8 bitwise at the unchanged kernel-set
digest (`docs/declared-divergences.md`). Nothing is declared for it.
The **2.8.0** move (2026-09-29) moved eight of the sixteen pinned files,
none of them the column batch, the seam document or a Noah-MP file, and it
is measured byte-neutral through the seam at the same three scopes: the
seam-level A/B is 500 of 500 arrays identical on both profiles and both
arms, a 10-minute forecast on a 12,795-cell 937.5 m point cull wrote both
history frames byte-identical under 2.7.4 and 2.8.0, and the cull's
contract deck is 8 of 8 bitwise at the unchanged kernel-set digest, all on
an RTX 5090 (`docs/declared-divergences.md`). Nothing is declared for it.
The two earliest moves have no such arm, and none of the three magnitudes
has been re-measured over 24 h since. Whether any of them moved is **NOT MEASURED**.
Read them as the numbers from the pre-2026-08-20 engine, not as today's.

1. **Upper-level warm drift.** Above level 45 the port warms relative to native
   at **+0.019 K/h, near-linear, one-signed**, reaching **+0.46 K at 24 h**,
   and it is identical in both weather cases and both mixing regimes. Case
   independence means it is a code path, not weather. The legacy-RRTMG radiation
   lane is the prime suspect. Fine at 24 h; extrapolated (not measured) a 7-day
   run carries about +3.2 K of stratospheric error, which disqualifies this
   version for long-range work.
2. **Convective-to-explicit precipitation repartition.** The port's GF produces
   about **a third less convective rain** (`rainc` -36 % / -34 % in the two
   cases); explicit microphysics makes up roughly half of it (`rainnc`
   +29 % / +25 %); **net domain-mean precipitation runs about 15 % dry against
   native MPAS-A**. This is the whole-model price of the declared GF
   generation gap above, quantified.

   **That number is against native MPAS, which is not the WOOF route's
   referee, and the observations returned the opposite sign.** "15 % dry" is a
   global domain mean against native MPAS-A. The verification of record for
   WOOF's physics is skill against observations; it ran on 2026-08-25
   against NCEP/EMC Stage-IV hourly QPE over a CONUS window and found the port
   **wet in all four cases** (paired case-block estimate **+0.0247 mm/h**,
   95 % interval **[+0.0041, +0.0606]**, 5,000 replicates) with the frequency
   bias at 1 mm/h above one in three of the four (1.59 / 1.35 / 1.38 / 0.77),
   so it rains over too much area rather than too little.

   **Read the obs result with its own limits: four cases is not a skill
   assessment.** Two of the four are the divergence cases themselves. One
   truncated at 23 h and carries the largest bias, +56.9 %. One is +41.5 % on
   almost no rain: 0.0156 against 0.0110 mm/h, an absolute difference of
   +0.0046 mm/h. The two clean, complete cases are +9.8 % and +2.4 %. Neither
   verdict cancels the other, because they are not the same domain or the same
   statistic: one is a global mean against a model, the other a CONUS window
   against gauge-and-radar analysis. What is settled is that being drier than
   native does not mean being drier than the atmosphere.
3. **Downstream condensate surplus.** With more rain made explicitly, the port
   carries **+50 % cloud water and +62 % rain water** in the domain mean by
   24 h, with much heavier point extrema (max-cell 24 h precipitation 502 vs
   308 mm). Probably a consequence of (2); it should be re-measured after any GF
   fix before anyone touches microphysics.

Everything else in the comparison is chaos-shaped: symmetric, growing with lead
time, driven by convective cells landing in different places, with envelope
statistics that match (peak updraft 11.49 vs 11.26 m/s; domain means within
0.05 %). That shape is expected between a single-precision CPU model and a CUDA
port and proves nothing wrong. The three above are one-signed and
case-independent, which chaos cannot be, and each names a lane to fix.

### Limited-area HRRR forecasts

`woof hex regional` is the single command for a fine hex core at a point:

```sh
woof hex regional --point 36.5,-97.5 --cycle 2024-05-06T18 --hours 3 \
    --work-dir RUN --geog /path/to/WPS_GEOG
```

It generates and culls the mesh, fetches HRRR, writes intermediates, init and
hourly boundaries with `rw_mpas_lbc`, earns a fresh configuration's runtime
mint, checks the contract on its rings, runs `forecast --lbc-dir`, and invokes
the products door. Mesh rows and contract evidence are passed internally.
The source and run shape are limited-area rows in the init-source and country
tables. See [the point/HRRR guide](docs/hex-point-hrrr.md).

Measured 2026-10-20 on an RTX PRO 6000: the command above completed a 43,360-cell,
55-level forecast with 2,160 steps at 5 s and four history frames. The forecast
stage took 1,908.79 s on a shared RTX PRO 6000 Blackwell Server Edition, with
CUDA graphs disabled. This used the installed hex 0.4.1 native binaries and
woof 2.9.3 dependencies with this branch's Python overlay. The published
`rw_mpas_lbc` wrote the complete hourly boundary set and the forecast consumed
it. The maps rendered 70 products, skipped two expected hour-zero products
and failed none, using only the refined-core view.

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

The run's `regional-receipt.json` records stage start/end/wall, outcome, mesh
and product counts. Runtime minting compares two independent strict dry
forecasts; it does not establish full-physics accuracy or observation skill.

The lane runs, and these are its edges rather than its promises.

- **One parent, windowed.** A cycle is one parent integration read at
  successive times, not a parent regenerated per cycle. Regenerating it is the
  operational remedy for a corridor that has moved far, and it is not built.
- **Two of four admitted slots per cycle** are skipped as background culls by
  a measured minimum-edge-length ratio, so a cycle can place fewer corridors
  than it detected.
- **Hour zero has no ice.** A corridor started from transplanted parent state
  begins with no cloud ice, snow or graupel, because the initial-condition
  stream carries no slot for them. Reflectivity does not correlate at hour
  zero; one hour on, the microphysics has re-formed the ice and r = 0.863.
  Temperature agrees to five decimals throughout.
- **No obs-skill score on a cycled case.** The limited-area verdicts above are
  against a global run over the same ground, not against observations.

### Other

- **Multi-node multi-GPU is built and proven but not shipped** (two-node,
  bitwise partition-invariant, 1.23x on 25 GbE). There is no door on it, and it
  has not been re-proven at the current engine pin.
- **Several cards on one host** is a door (`--samehost-devices`, manual
  chapter 12). A two-card global 30 km run restarted from a checkpoint matched
  the uninterrupted run in all 117 field hashes, but the door's own
  qualification (full-physics history identity, distinct-card scaling) is
  incomplete. Local time stepping uses one card unless `--samehost-devices`
  asks for more; multi-rank DA and multi-card LTS cycling are refused by name.
- **A case has been seen to refuse mid-run** on a vertical-velocity divergence
  at levels 44-47. The refusal is the model declining to publish a step it does
  not trust, which is the designed behaviour, but it means a given case can stop
  early rather than produce a bad forecast.

## Licence and derivation

woof hex is **Apache-2.0**, the same licence woof ships under, except where
an upstream's own terms govern a ported portion.

It ports MPAS-Atmosphere v8.4.2 (LANL/UCAR) to CUDA devices, with pinned
source-line citations: the dynamical core, initialization, IAU, limited-area
boundaries, restart, streams, diagnostics and the MPAS physics suites. MPAS is
BSD-3-Clause, whose terms govern the MPAS-derived portions and travel with every
copy: `NOTICE` reproduces the MPAS licence in full and carries the marking it
requires of derivative works. The MPAS physics suites also port NCAR
MMM-physics (BSD-3-Clause), WRF physics as MPAS distributes it (WRF
public-domain notice) and RRTMG (AER, BSD-3-Clause). `NOTICE` names each
upstream, `THIRD-PARTY-NOTICES.txt` lists the derived files and `licenses/`
holds the texts; all of them ship inside the wheel and the sdist. Every
third-party term in the distribution is permissive; `NOTICE` section 2 names
the one routine kept on a stated reliance instead (Numerical Recipes'
log-gamma in WSM6's reflectivity constants, as MMM-physics carries it).

WOOF's RTE+RRTMGP radiation, which any WOOF-route run that selects it
runs, reads a trace-gas and ozone
table that woof's data package ships under CC-BY-SA-4.0. woof hex does not
carry it. It is data: the licence permits sale, and its share-alike reaches
only adaptations of that table (`NOTICE` section 10).

The wheel's compiled helpers and the libraries they link (netCDF-C in the
template copier, the pool reader and the stream bridge, which also links HDF5
and zlib; the Rust crates in the observation scorer, the pool reader and the
two engine bridges) are listed in `NOTICE` sections 12 to 14. The texts of the Rust crates the
scorer and the bridges link are in
`licenses/THIRD-PARTY-LICENSES-helper-binaries.txt`.

**US Government code:** the CUDA ports of NOAA GSL's UGWP gravity-wave
drag (`bl_ugwp_gwdo`, `ugwp_ngw`) ship in the distribution, with their tests
and oracles, and the RRFS stage runs the fork's NOAA GSL physics (MYNN-EDMF,
MYNN-SFC, RUC, GFL, smoke) with no opt-in. Their upstreams carry no licence
file; the project treats them as works of the US Government (NOAA) by its
ruling of 2026-10-14. `NOTICE` sections 8 and 9 give their pins and
attribution.

**Withheld from the public distribution:** the UCAR Noah-MP v5.0 port, with
its tests and native oracles. `NOTICE` section 4 retains its upstream pin,
attribution and the reason for the hold. The public wheel contains no
Fortran runtime. A withheld selection is refused by name; an opt-in does not
restore code absent from the wheel.

The Noah-MP that hex runs without that opt-in is the WOOF route's WRF-lineage
v4.x (WRF v4.6.1's Noah-MP, WRF public-domain notice); it is not MPAS-bitwise.

This is **not** the version available from LANS and UCAR, and neither they nor
their contributors endorse it. Results from woof hex are not results from
MPAS-Atmosphere; where the two are known to differ, the differences are the
measured ones stated above.

---

## Development

Tests run from this directory:

```sh
PYTHONPATH=src python -m pytest tests -q
```

Three tiers gate themselves and each names why it is skipping: see
[`tools/battery/README.md`](tools/battery/README.md) for what each covers and
how to run the gated ones:

| tier | selector | needs |
| --- | --- | --- |
| unit + packaging | `-m "not gpu and not bigcard and not assets"` | nothing but Python |
| assets | `-m assets` | about 6.9 GiB of byte-pinned mesh/static/init/native-history files |
| big card | `-m bigcard` | a CUDA device clearing the measured x4 floor: about 21.7 GiB free (the measured 20,446 MiB peak plus that card's own margin, computed by `woof.hex.device_admission`; re-fitted at the merged tip 2026-08-26 and re-shaped 2026-08-27) |

`WOOF_HEX_NO_LOCAL_GPU=1` (or `GPUWM_NO_LOCAL_GPU=1`, honoured so a box
configured for the engine behaves the same) bans device contact outright.


### Subseasonal global ensembles

`woof hex s2s --row global-120km --source gfs --members 4 --days 30
--cycle 2026-10-17T00 --devices 0 --work-dir s2s-run` drives the persistent
ensemble workflow. See [the S2S door](docs/s2s/README.md) for lagged starts,
checkpoint recovery, products, baseline interpretation and measured costs.
Use `--dry-run` to inspect the plan.

Native forecast stores: [Icechunk output](docs/icechunk.md).
