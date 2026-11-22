# woof hex

MPAS-Atmosphere v8.4.2 and NOAA's hrrrv5 (RRFS-MPAS) configuration on NVIDIA GPUs, with the tangent-linear and
adjoint of the exact-mode hrrrv5 forecast.

**Install and check** (Linux x86_64, Python 3.11 or newer, one NVIDIA card), in a Python environment of its own:

```sh
python3 -m venv hexgpu && . hexgpu/bin/activate   # Debian and Ubuntu: sudo apt install python3-venv first
pip install "recast-woof[gpu]"      # forecasts
woof hex doctor                  # what this install can reach, and the command that closes each gap
```

**For the derivative commands, install the `ad` extra instead**, in a Python 3.14 environment of its own, not on top of
`[gpu]`. Each of the two extras brings its own CuPy package, one environment holds one, and pip does not remove the
other (`woof hex doctor` reports two CuPy packages as a gap and prints the command that leaves one). `[ad]` installs
what `[gpu-cu13]` installs (the CUDA 13 CuPy), so it needs a driver that serves CUDA 13; the forecast commands of this
release were run from a `[gpu]` environment.

```sh
python3.14 -m venv hexad && . hexad/bin/activate   # Debian and Ubuntu: sudo apt install python3-venv first
pip install "recast-woof[ad]"
woof hex doctor
woof hex adjoint-test --row hrrrv5-q12.96.2890.n40.60w91.30 --steps 15 --at 40.6,-91.3 --radius-km 150 --out cut12
```

The last command is the 12 km example. It needs one card of compute capability 12.0 with 28,700 MiB free (a 32 GB
GeForce RTX 5090 runs it) and 47,104 MiB of host memory, fetches its own 1.20 GB of inputs, and ends **PASS** in 5 to
13 minutes (320 to 645 s on 0.6.0's release builds; on one RTX 5090, 418 to 708 s on builds of 0.6.1 and 769 s on
a build of 0.6.2, its host's processors shared with a package build; 466 s on one RTX PRO 6000 96 GB on another;
*The differentiable engine*, below). `woof hex examples` prints where the runnable example scripts are, and
`woof hex examples --copy DIR` copies them.

CUDA remains the default. The [experimental AMD backend](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/amd-backend.md)
is opt-in with `--backend amd` or `WOOF_BACKEND=amd`; the linked page records
the installation paths, measured single-card scope and numerical limits.

woof hex ports MPAS-A v8.4.2 to Python and CUDA, fully device-resident: the
dynamical core on its unstructured global mesh with regional refinement, the
MPAS physics suites, initialization and static fields, the incremental analysis
update (IAU), limited-area runs, streams, restart and diagnostics. Its forecasts run on one consumer CUDA card. MPAS's own physics is the default; WOOF-ArWen's physics
stays one flag away, and hex adds capabilities MPAS does not have:
one-command global forecasts, local time stepping, forecast products and scores
at the end of every run, and an experimental storm-following region.

*(MPAS-Atmosphere is a registered project of NCAR and LANL. This distribution
is an independent port and is not affiliated with or endorsed by them. The name
is used here only to say which model was ported.)*

**New in 0.6.2:** each item has a short section in
[User Manual chapter 11](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/manual/11-new-in-0.6.2.md).

- The meshes the doors generate are built with the new generator's geometry 2 (no obtuse triangle, every spacing
  built as asked); `--geometry 1` gives 0.6.1's meshes.
- Hail size (HAILCAST, HRRR's hail fields) on meshes finer than 4 km, on by default.
- Ranks on several hosts (`--hosts`), with checkpoints sealed and resumed across hosts; seven speed settings for
  runs on several ranks, on by default, none of which changes a stored value; updraft helicity on several ranks the
  same as on one card.
- Opt-in, off by default: three surface and boundary-layer closures for `hrrrv5` (a surface-flux network, a neural
  boundary layer and an 8-parameter closure) and a learned column correction that takes your own checkpoint.
- A start row `ifs-gfs-top` (ECMWF IFS open data with GFS above 50 hPa).
- `hrrrv5` with Grell-Freitas convection (`--physics config_convection_scheme=cu_grell_freitas`) runs on the GPU.
- Fixes: `woof hex render` given several history frames files each at its own lead, every door's `--help` runs on
  Windows, correct mode no longer draws a false light echo over the oceans, `--local-timestep` no longer stops at
  step 0, and twelve fixes from github.com/jacobbieker/woof. The
  [CHANGELOG](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/CHANGELOG.md) lists each change, and 0.6.1's.

**What 0.6.0 added.**

- **The tangent-linear and adjoint of NOAA's exact hrrrv5 configuration:** of the exact-mode forecast
  (`--physics-mode exact`), not of the correct-mode forecast `woof hex forecast` runs by default, whose gradient is
  refused in one sentence. Four commands run it (`woof hex tangent`, `adjoint-test`, `sensitivity`, `param-grad`), on
  NVIDIA cards of compute capability 12.0 (GeForce RTX 50 series, RTX PRO Blackwell): the 12 km example on a 32 GB
  card, the 3 km and 120 km examples on a 96 GB card.
- **Two checks, printed by the commands.** The transpose check asks whether the adjoint is the tangent-linear model's
  transpose over the whole window: the two sides of its dot must agree within 4 single-precision roundings along every
  tested direction, and over 8 or more directions they must not lean one way. The gradient test compares the gradient's prediction with the real
  model's change and has a 1 per cent bar. What the derivative holds is named (*The differentiable engine*, below, and
  [the limits](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/differentiable/LIMITS.md)).
- **NOAA's hrrrv5 deck, bit for bit.** A plain `--physics-suite hrrrv5` run is NOAA's hrrrv5 deck, and with
  `--physics-mode exact --numerics strict` its one-hour forecasts on a 12 km and a 3 km limited-area mesh match the
  patched native RRFS-MPAS 44c0849 reference in every stored value (0 of 86,751,014 and 0 of 624,931,668 values
  different, each hour from the HRRR 2024-05-21 15Z start, one process on one RTX 5090; the CHANGELOG's 0.6.0 headline
  has the files). The forecast also runs on older card families, the RTX 4090 included.

**New here? Start with the [User Manual](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/manual/index.md):** a
plain-language introduction, a quickstart in which every command was run
before it was printed, task guides for each door, and a troubleshooting index
built from the doors' own refusals. This README is the capability contract;
the manual is how you walk it.

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
  12Z and 2022-12-22): 0 differing values at all 21 six-hourly samples;
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
called identical. [MPAS compatibility](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/mpas-compatibility.md) has the
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
are in [`verification/gates/`](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/verification/gates/README.md) and the
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
correction out, for an observation A/B arm. On the ArWen physics route the
same flags, and correct mode by default, apply the corrections the installed
woof carries (WB4, New Tiedtke's dry-mass coupling; WB5, WSM6's surface
integral) and name the rest as gaps. Details: [MPAS physics seam](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/manual/mpas-physics-seam.md#correct-mode-and-exact-mode).

### Initialization, statics and the services around the forecast

| Piece | Proof, against native MPAS-A v8.4.2 |
|---|---|
| Real-data initialization (`init_atmosphere` case 7) | 2,880 captured native calls replayed, 134,016 output values, 0 different; a cold start from it matches all 134 native fields on both measured meshes, and a forecast from it matches native in every written field (426,595,161 values) |
| Static fields (`rw_mpas_static`, the native static stream) | every value equal, except 1,473 `albedo12m` and 189 `soilcomp` values where native itself reads undefined memory: declared residue, never called identical |
| Gravity-wave-drag statics (`rw_mpas_static --native-gwd-static`) | the complete 79-field static file: 4 cases, 11,675,352 values, 0 different |
| GSL orographic statics, idealized initialization | bitwise on captured native inputs |
| GOCART aerosol climatology (initial and boundary) | 91,420,560 values, 0 different |
| SST and sea-ice update (`woof hex surface-update` builds the records; the forecast's surface update applies them) | a forecast with SST updates on matches native in every written field: 581,775,705 values, 0 different |
| IAU | MPAS-A's IAU kernels bitwise on captured native inputs; in a forecast IAU runs on the ArWen physics route (`--iau-route native\|hex`, IN1-IN3 applied in correct mode) and on the explicit MPAS suite route (MPAS's own `atm_add_tend_anal_incr` once per step) |
| Limited-area boundaries | 2,822,211,692 values, 0 different, on captured native inputs; seven regional cases' difference maps are zero |
| Streams and restart | bitwise on captured native inputs |
| Diagnostics: isobaric, convective, cloud, potential vorticity, soundings | bitwise on captured native inputs (isobaric: 471,380 values, 0 different) |

The real-data initialization's vertical grid comes from `rw_mpas_geometry`
speaking `hex-vertical-v1`, MPAS streams and restart use `rw_mpas_stream`,
and a mesh generated at a point needs an `rw_mpas_static` whose statics hold
the binary64 coordinates they declare. No released woof carries any of the
three in the version this release needs, so the wheel carries them in
`woof/hex/libexec` ([engine bridges](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/engine-bridges.md)). A door that
needs one and cannot find it refuses by name.

**A forecast from an analysis.** `woof hex forecast --analysis BUNDLE` starts
a forecast from an MPAS analysis bundle ([analysis bundle](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/analysis-bundle.md)):
by direct insertion, with an IAU window from the bundle's increment, as a
cycling restart, or with a remap onto another mesh. Measured on one convective
case (ArWen physics): with no bundle every history file is byte-identical to
the run before this release, and in correct mode a zero increment is bitwise a
plain run. IAU also runs across a restart (the checkpoint carries its window),
on the explicit MPAS suite route, on same-host multi-card ranks and in member
rosters ([analysis bundle](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/analysis-bundle.md)); IAU on multi-card local
time stepping is refused by name.

### Hex global, in one command

```sh
woof fetch-geog      # once per machine: the geography every static needs
woof hex global --row global-120km --cycle 2026-10-11T12 --work-dir RUN
```

The first line stages, once, the geography archive the static stage reads: 16 datasets, 2,705,039,107 bytes
(about 2.5 GiB) to download and about 31.3 GiB unpacked, by default into `~/.local/share/woof/WPS_GEOG`, where the
global command looks for it (69 s on one measured host, with gpuwm 3.0.3; `woof fetch-geog --explain` prints the plan without
downloading). On a fresh install without it the global command stops at its static stage, exit 2, with "no geography
root". The published global meshes need it too: their static is built on your machine.
The second line goes from a global row and a cycle to a scored, drawn and exported forecast:
the NCAR-published mesh (fetched and pinned), the native static, the init, the
forecast on as many cards as the mesh needs, products, export and scores.
Rows are defined at 120, 60, 30 and 15 km, on ArWen physics and on the
Morrison suite (Morrison, Grell-Freitas, YSU, revised MM5, Noah, RRTMGP); the
measured runs are the 120 km rows. Hex global starts from an analysis you
supply (`woof hex forecast --analysis`).

On one case (GFS 2026-10-11 12Z, 24 h), the 500 hPa height error against the
GFS analysis was 10.0 m for hex at 120 km, against 14.9 and 13.5 m for T127
and T255 runs of a spectral global model; T255, at twice hex's resolution, was best on
250 hPa wind (4.80 against 4.87 m/s). One case is not a skill claim. On the
120 and 60 km rows the physics step was capped at 300 s on gpuwm 2.9.0 (an engine older than this release installs): at the
mesh's own step the YSU boundary layer and New Tiedtke left single layers up to
24 K too hot, and one row ran away to 381 K. The cause is explicit tendencies
overshooting their donor layers; an engine carrying both donor bounds runs the
WOOF-ArWen rows at the mesh's own step with no hot layer, and the door retires the
cap there by itself. See [Hex global](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/global-door.md).

### Storm-following region (experimental)

`woof hex moving` keeps a 3 km region that moves with a storm inside a 48 km
global mesh, on the MPAS-exact dycore with the physics the case selects (the
measured runs used ArWen physics). One case file drives `stage`, `prepare`,
`run` and `fixed` (the matched control); `woof hex moving check` tests an
install with no inputs. With no moving command and no `WOOF_HEX_MOVING_*`
variable set, a run is unchanged (`tests/test_moving_inactive_noop.py`).
Guide: [Moving nests](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/moving-nest.md).

Measured: three tropical cyclones, 48 hours each, five moves each, one card,
a 3 km region about 450 km in radius.

- The storm stayed inside the 3 km region at every hourly output (49 of 49 in
  each run).
- Mass and tracers held to within 1e-10 at each move.
- A restart across a move continued bit for bit (one move tested).
- Those runs used the remap before the balance fix: at 2 of the 15 moves, the
  short-term shock measured against a fixed-region run went over its 1.5 limit
  (1.5004 and 1.81). The balance fix is now the default. On 0.6.2,
  `woof hex moving check --lts` brings the worst post-move jolt from 12.97 to
  1.014 on its gentle case and from 152.8 to 1.029 on its steep case (an RTX
  4090 and an RTX 5070 Ti print the same numbers), and `--device cpu` brings the
  gentle case from 19.3 to 1.04 (ordinary integrator); the first measurement is
  in [the jolt report](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/moving-jolt-report.md). A
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

- the fork's dynamical core, as a module replay: 1,235,615,738 values, 0
  different;
- the fork's physics driver layer, the same way: 104,976,732 values, 0
  different.

**A whole RRFS forecast is bitwise to the native fork build in every field hex
writes:** a one-hour `hrrrv5` run on `x1.2562` (2023-03-31 12Z, one process)
with the fork's dycore on the GPU and every physics slot on the native
library compares 40,525,734 values with the native fork build (strict recipe),
0 different, and 0 different at every 720 s step when every stream is written
each step. Scope: that case and process count; native also writes fields hex
does not write yet, so the claim is every field hex writes. That run put
every slot on the native library (`--physics-default-backend native-cpu`).

The measured RRFS GPU-port run used these schemes on `hrrrv5`:
TEMPO, MYNN-EDMF, the MYNN surface layer, RUC land and sea ice,
RRTMG longwave and shortwave and the cloud fraction (GFL, MYJ and the fork's
MYNN surface layer when a suite selects them). Each port is bitwise to the
native fork build as a module replay on real captured calls. (History, against
calls captured from the pristine build: two RUC sea-ice calls on an RTX 5090
differed in one `tsnav` word each, where pristine native reads the
uninitialised local `ilnb`; the reference of record sets it, see below.)

**With every slot on the GPU and no native library, a whole `hrrrv5` forecast
is bitwise to the native fork build (strict numerics):** a 6 h `x1.2562` run
(2023-03-31 12Z, one process) differs in 0 of 129,420,310 stored values from
the native reference of record (the patched referee below) at every hour
f000-f006: history 25,664,457, diagnostics 3,982,153 and hourly restart
99,773,700 values, `tsnav` included. Both runs read NOAA's published TEMPO
lookup tables and `ccn_activate.bin`: hex names the native run folder with
`--native-physics-tables`, as GUIDE section 7 step 6 says (a default run
reads the fork's shipped tables). Read on 0.6.0's wheel, one RTX 5090. (History, against the pristine build: 57 of 129,371,634 values
differed, all RUC's snow temperature `tsnav` at thin-snow cells, where pristine
native reads the uninitialised local `ilnb`.) Scope: that case and process
count.

**At the operational shape, 8 ranks, the whole `hrrrv5` forecast is bitwise to
the reference of record (strict numerics):** on `x1.40962` (120 km, 59 levels,
2023-03-31 12Z, every physics slot on the GPU), 8 same-host ranks on one card
against native at 8 MPI ranks on the same partition differ in 0 of
1,958,644,100 history, diagnostics and restart values over 24 h; one card
against native at one rank differs in 0 of 786,309,080 values over 24 h.

**Restart.** A native MPAS restart file restarts the dynamics from the restart
stream on every MPAS physics suite. With `--numerics strict` a restart
reproduces native MPAS's own restart: hex restarted from its own 24 h restart
matches native restarted from native's in 0 of 540,860,379 values (`x1.40962`,
one card against one rank). Native MPAS's own restart does not continue its
run exactly (it starts its step counter again, and first-step scheme blocks
run again). On the default route a restart continues the run that wrote it
exactly: restarted against uninterrupted, 0 of 1,208,871,000 values on
`x1.40962` at 8 ranks and 0 of 452,696,265 values on `x1.163842` (60 km, 8
ranks, restarted at +24 h, compared to +30 h). A restart file native MPAS
wrote keeps native's restart behaviour on either route.

**The RRFS reference of record is patched, and says so.** It is the pinned
RRFS-MPAS 44c0849 source built with the strict recipe plus exactly four
labelled lines, each where native reads a value it never set: two
`nullify(smoke_fine_fg, dust_fine_fg, dust_coarse_fg)` lines in the
initializer (native's case-7/9 init otherwise crashes on unset optional
smoke/dust pointers) and two `ilnb = 0` lines in RUC land and sea ice before
the snow calls (native otherwise reads an unset local for snow thinner than
`snth` and picks the `tsnav` formula from a stale stack word, which then feeds
snow compaction). Pristine against patched native reproduces the
hex-vs-pristine divergence word for word, and patched native is 0 different
run to run. It is rebuilt with
`RRFS_44C0849_REFEREE=patched tools/rrfs_44c0849_reference/rebuild.sh REF NEW_WORK atmosphere`
(and `init_atmosphere`), after `tools/rrfs_44c0849_reference/make_ref.sh REF`
fetches the pinned sources from their upstream repositories; every binary it
builds carries `referee: patched` and the exact `source-correction.diff`.
Native outputs are not published: the differentiable GUIDE, section 7, walks
the six-hour `x1.2562` case from an empty folder (build, tables, NOAA's deck,
init, native run, hex run, compare). It is never called pristine. The
restart-word `tsnav` exclusion used against the pristine build is retired. `config_<slot>_backend = 'native-cpu'` puts one slot back on
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
([RRFS init](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/rrfs-init.md)). The deck's invariant file is read by default
from the deck's directory or the init's (`--stream-input invariant=FILE` names
another).

The RRFS-shaped presets (`--physics-suite rrfs_v2_dev`) are a different thing:
they run ArWen physics shaped after RRFS v2's `hrrrv5` suite, and they are
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

- **A limited-area forecast runs the full ArWen physics stack** (WSM6,
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

### 0.4 in short

[Physics routes and measured comparison scopes](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/physics-routes-0.4.1.md) describe the MPAS physics route (the default since 0.4.6), the WOOF-ArWen route and public licence exclusions.

0.3.3 (2026-09-29) was the MPAS-A dynamical core on the GPU, running WOOF-ArWen's
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
   every door; WOOF-ArWen's physics stays, one flag away (`--physics-suite
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


[What's new in 0.4](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/manual/10-whats-new-0.4.md)
gives each piece of 0.4 its command, its proof, a picture and what it does not yet do.

## What is *not* claimed

- **Outside the measured scope, nothing.** The dycore identity holds on the
  cases listed above, on one host's C library; the all-GPU match holds for the
  one headline forecast; each other MPAS scheme is bitwise on captured inputs,
  not proven forecast by forecast.
- **Correct mode is not native MPAS.** By design it differs from native
  wherever a fix acts. Each fix is checked against its own referee build;
  there is no combined referee.
- **ArWen physics is not MPAS physics.** A forecast on the WOOF-ArWen
  route (`--physics-suite arwen_mpas_column`) is not identical to native MPAS, and the dycore check does not
  qualify one. Its measured differences from native MPAS are registered, with
  mechanism, magnitude and observational referee, in
  [Declared divergences](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/declared-divergences.md). Its referee is skill
  against observations (MRMS, Stage-IV, ASOS, radiosondes), not agreement
  with another model; a bias that is wrong against observations is still
  wrong.
- **The adjoint's consistency is not the gradient's accuracy, and one card is not several.** The dot tests show that
  the adjoint is the transpose of the tangent-linear model on the scope stated in *The differentiable engine*. How
  well the gradient predicts the forecast is claimed only for the measured rows of the table "How far the gradient can
  be trusted" in [The differentiable engine](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/differentiable/README.md): about 3 hours at
  120 km and 60 km, 14 min 45 s at 3 km, about 1 hour wherever the mountain-drag closure is differentiated. A
  composite-reflectivity measure does not pass the gradient check. Nothing is claimed for a gradient split across several cards (the split is experimental, and says so when used), or for any card family but compute capability 12.0.
- **RRFS-MPAS is matched on its measured cases only.** In exact mode with strict
  numerics, against the patched 44c0849 reference: six hours of `hrrrv5` on
  `x1.2562` at one process; 24 hours on `x1.40962` at one process and at 8
  same-host ranks on one card; and one hour on the 12 km cut and on the 3 km
  storm box from the HRRR 2024-05-21 15Z start, one process (*RRFS-MPAS*, above,
  and the CHANGELOG). Other meshes, dates, lengths, process counts and several
  cards are not measured.
- **Local time stepping is never called identical.** It is a different time
  scheme, and native MPAS has none.
- A one-step dry pass does not qualify a forecast trajectory or its weather
  skill.

## Not in this release

| Item | Reason |
|---|---|
| The CUDA port of Noah-MP v5.0 (`sf_noahmp`, with its urban switch) | withheld: UCAR's terms forbid selling it or any work that contains it. The native backend runs Noah-MP v5.0 only with `--licence-opt-in ucar-noahmp`, from a native library built with the same opt-in (`woof hex physics-resources build --licence-opt-in ucar-noahmp`); the Noah-MP that hex runs without that opt-in is the WOOF-ArWen route's WRF-lineage v4.x |
| CAM longwave and shortwave radiation (`cam_lw`, `cam_sw`) | on the GPU route a run that selects them is refused by name before step 1. The port has run only standalone column witnesses: its tables and module words come from the native library (the rw-mpas CAM table decoder is not built), and no whole forecast has run on it on any route |
| The `cld_incidence` and `cld_fraction_thompson` cloud fractions on the GPU route | they are in the shared radiation code but not bound to the resident GPU glue, so a GPU run that selects them is refused by name before step 1; they run only on a site-built native library (see *Requirements*). `cld_fraction`, the choice of both MPAS suites, runs on the GPU |
| NSSL two-moment microphysics and smoke/dust on the RRFS stage's GPU | they run on the native library: they are ported but not wired into the forecast. These two selections require a site-built native library |
| Zone carry (corridor cases) on a moving region with an MPAS suite, and the RRFS suite on a moving case | zone carry with an MPAS suite is refused by name (run the case without a corridor; its moves take the overlap remap). The RRFS suite takes the same route but has not been run on a moving case. MPAS-A v8.4.2 suites move on one card (docs/moving-nest.md) |
| A full-cyclone replay of the two over-limit moves with the balance fix | not yet run; it needs a card with about 64 GB or more (the 470,308-cell member holds about 54 GB, and a move about 62 GB) |
| An ensemble analysis inside hex global | not part of this package: hex global starts from an analysis you supply (`woof hex forecast --analysis`) |
| IAU on multi-card local time stepping, and `--balance-series` off one card | the LTS block executor has no held-tendency hook and the ranks no balance sampler; refused by name |
| A combined correct-mode native referee | one native build carrying every fix is a follow-up |
| A hex-specific maths-library pin | it changes the dycore arithmetic and needs every native reference rebuilt |
| YSU and New Tiedtke single hot layers at long steps on gpuwm 2.9.0 | WOOF-physics runs with YSU at a step above about 300 s (meshes coarser than about 50 km at MPAS's 6 s per km) can show the single hot layers the global door caps away. The cause is YSU's explicit boundary-layer-top entrainment and New Tiedtke's explicit rates overshooting their donor layers. An engine whose YSU and New Tiedtke identities are `ysu-wrf-v4.8.0-entrainment-courant-v2` and `new-tiedtke-wrf461-cumastrn-donor-bound-v2` bounds both, and `woof hex global` then runs the WOOF-ArWen arm at the mesh's own step. gpuwm 2.9.0 predates this release's engine floor and has neither, so on it every door's default step with YSU is capped at 300 s (`woof hex global` and a bare `woof hex forecast` alike); an explicit `--dt` runs as given |

---

## Requirements

| | |
| --- | --- |
| Python | 3.11 or newer |
| GPU | A CUDA device: any NVIDIA card of compute capability 7.5 (Turing) or newer runs, on a CUDA 12 or CUDA 13 driver, with no flag. sm_86 and sm_89 are anchored (the numerical contract measured again on that architecture), sm_120 is the proven contract floor where it was first measured, and any other card runs unanchored and its receipt says so; `woof hex doctor` names your card and its status. Memory is set by mesh size *and* by the card, and not by a line through the two: the footprint is a per-card core, plus the Grell-Freitas and YSU column workspaces charged at `min(cells, tile)`, plus a per-cell term, all in `woof.hex.device_admission`. Measured 2026-08-26 on an RTX 5090 at the merged tip: the published 40,962-cell global mesh (x1.40962, about 120 km) peaked at **8,874 MiB, inside a 12 GiB card's budget**; the 163,842-cell mesh (x4.163842, about 24 km) peaked at **20,446 MiB**, and every free-memory gate admits at the measured floor (the prediction plus that card's own margin, **about 21.7 GiB free at x4**), so x4 remains in practice a 32 GiB-card configuration. The core is a property of the card, not the mesh, and the door reads the card's multiprocessor count at the decision rather than assuming a 5090: the 16 GiB and 10 GiB parts each carry their own measured row, and a card nobody has measured gets a derived row that is labelled derived. A card with its own per-allocation ledger can still supply it with `--device-fixed-mib` / `--device-bytes-per-cell`. |
| CUDA | CuPy matching your driver's CUDA major. There is no way for pip to detect it, so you choose (below). |
| Engine | The `woof` distribution, 3.0 (`gpuwm>=3.0.0,<3.1`), which pip resolves for you: the ArWen physics seam, and the bundle that carries the MPAS bridge binaries the doors drive (including `met_intermediate` for init-source and the renderers for products). The forecast runs on the installed engine; a git clone of the engine is optional (receipts then name its commit). See *The engine pin*. |
| Platforms | Linux x86_64 with glibc 2.28 or newer installs the prebuilt wheel (`manylinux_2_28`: RHEL 8 and 9, Ubuntu 20.04 and later). Anywhere else pip builds the sdist, which needs a Rust toolchain (`cargo`) for the bundled helpers; the observation scorer is built only when `GPUWM_HEX_ENGINE_SOURCE` names a woof source checkout, and otherwise `woof hex score` refuses by name and `doctor` reports the gap (forecasts are unaffected). There is no prebuilt Windows wheel. |
| Assets | A mesh grid file, a matching static file, and (for the init door) a vertical-grid declaration, normally a `--vertical-spec` JSON; a native-minted init file as a capsule is the compatibility mode. `woof hex global` fetches its published meshes itself; otherwise **woof hex ships none of these**. See *Assets you must supply*. |
| Rust binaries | The field engines (`rw_mpas_init`, `rw_mpas_static`, `rw_mpas_convert`, `rw_mpas_lbc`, `rw_wrfbatch` and the others) come from woof's bridge bundle or a woof source build. Nine of them ship in this wheel, with its own helpers: `rw_mpas_geometry`, `rw_mpas_stream`, `rw_mpas_static`, `rw_mpas_mesh`, `rw_mpas_lbc` (native case-9 boundaries), `rw_igra2` (IGRA2 radiosondes for `woof hex score`) and the RRFS fork dycore's three programs. See *Getting the Rust engines*. |
| MPAS physics suites | The MPAS-A framework uses the wheel's helpers and pinned DATA from `physics-resources-v0.4.1`. `woof hex physics-resources fetch --from DIR` stages an offline mirror; `WOOF_HEX_PHYSICS_RESOURCES_FROM=DIR` selects it at first use. RRFS `hrrrv5`, both RRFS alias suites and New Tiedtke run on the same terms; selections without resident CUDA table entries require a site-built native library. See [MPAS physics resources](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/physics-resources.md). |

### Install

```sh
python3 -m venv hexgpu && . hexgpu/bin/activate   # Debian and Ubuntu: sudo apt install python3-venv first
pip install "recast-woof[gpu]"         # forecasts: the CUDA 12 CuPy; = "recast-woof[gpu-cu12]"
```

```sh
python3.14 -m venv hexad && . hexad/bin/activate  # INSTEAD of [gpu], for the derivative commands
pip install "recast-woof[ad]"
```

Each install goes in a Python environment of its own: a system Python that marks itself externally managed (Ubuntu
26.04's does) refuses `pip install` outside one with `error: externally-managed-environment`.

**One environment, one extra.** A bare `pip install recast-woof` installs no CuPy and cannot run a forecast. `[gpu]`
brings `cupy-cuda12x`; `[ad]` brings `cupy-cuda13x` with the CUDA 13.4 toolkit the derivative commands replay, and
needs a driver that serves CUDA 13 (the forecast commands of this release were run from `[gpu]`). The `ad` extra goes
in its own Python 3.14 environment, not on top of `[gpu]`: pip does not
remove one CuPy package when it installs the other, CuPy warns that it may not work with two, and
`woof hex doctor` reports two as a required gap with the one command that leaves one
(`pip uninstall -y cupy-cuda12x cupy-cuda13x && pip install "recast-woof[ad]"`).

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
required item is missing, so it works as a gate in a script. On a fresh install
the one missing item is `WPS_GEOG`, the geography archive the static builder
needs for any static, the published global meshes included (`woof hex global`
builds its static on your machine); the four adjoint commands do not need it
(their examples download what they replay). `woof fetch-geog` stages the
default set: 16 datasets, about 2.5 GiB to download and about 31.3 GiB
unpacked (gpuwm 3.0.3). An `hrrrv5` run, and a nest under one, also reads two
sets that the default leaves out and only NCAR serves:

```sh
woof fetch-geog --datasets lai_modis_30s,modis_landuse_20class_15s
```

That is 0.76 GiB to download and about 13.9 GiB unpacked; `--explain` prints
the plan without downloading. A static build that lacks them names them and
prints this command with your geography root.

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
  each with its own SPPT/SKEB forcing ([moving-region ensembles](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/moving-ensemble.md)).
- `woof hex chem-sources` lists, fetches and prepares smoke, composition, dust and
  sea-salt source files for aerosol transport ([chemistry sources](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/chemistry-sources.md)).

### Kernel packs

Kernel packs are pinned assets for NVRTC 12.9, 13.0, 13.2, 13.4 (13.4.59 and 13.4.92) and 13.5 (release
`kernel-packs-v0.6.2`), and hold the WOOF-ArWen and MPAS physics kernels of this release.
The `gpu-cu13` extra installs woof's own `gpu-cu13` toolchain with it, so the pair resolves one CUDA stack (NVRTC 13.4.92 today, the one this release was gated on), and a fresh install loads a prebuilt pack.
`woof hex fetch-assets --list` lists them. Run
`woof hex fetch-assets` to stage them in the cache, or
`woof hex fetch-assets --from DIR` to verify and stage an offline copy.
`WOOF_HEX_ASSETS_FROM=DIR` selects an offline source directory for first use;
`WOOF_HEX_ASSETS_URL` selects a release-asset mirror.
`WOOF_HEX_ASSETS_FIRST_USE=off` disables first-use downloads.
`--cache DIR` chooses a cache and `--only NAME` selects an asset.
Each fetch checks the pinned size and SHA-256 and writes a receipt.
`woof hex doctor` reports staged and on-demand assets without fetching, and warns when the
loaded NVRTC matches no pinned pack. A run without a matching pack compiles its kernels through
NVRTC.

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

woof hex 0.6.2 requires gpuwm 3.0.0 or newer inside 3.0 (`gpuwm>=3.0.0,<3.1`). The installed
engine version and seam digests are recorded in each forecast receipt.
The bridge source commit and measured helper source sets ship in the package manifests.

- **The range admits gpuwm 3.0.0 and newer patches below 3.1.**
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
stays as an option: pass a clone of the engine and the receipt records its HEAD,
tree and dirty paths instead. Any other directory that is neither a git
working tree nor the installed engine (an unpacked tarball, a copied
`site-packages`) is refused by name, because it names neither a commit nor a
record.

*(The woof source tree publishes too: `github.com/recastsystems/arwen`
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
  [`docs/native-free-init-admission.md`](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/native-free-init-admission.md).
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
binaries. The prebuilt wheel carries 38 compiled files of its own in
`woof/hex/libexec` (programs and shared libraries), among them:

- `hex_contract_hostprep`, the host-preparation helper;
- `gpuwm-hex-netcdf-copy`, the init door's CDF-5 vertical template copier;
- `hex_obs_scorecard`, the observation scorer behind `woof hex score`;
- the four table decoders `woof hex physics-resources build` uses:
  `lookup_decode` (a shared library), `table_decode`,
  `rw_mpas_rrtmg_lw_tables` and `rw_mpas_rrtmg_sw_tables`;
- `rw_mpas_physics_pool`, the MPAS route's Registry pool reader;
- nine engine bridges that no released woof wheel carries in the version
  this release needs: `rw_mpas_mesh` (mesh cuts, `--cull-parent`), `rw_mpas_geometry` (speaking `hex-vertical-v1`, for the
  init door), `rw_mpas_stream` (MPAS streams and restart), `rw_mpas_static`
  (statics for meshes generated at a point, which hold the binary64
  coordinates they declare), `rw_mpas_lbc` (boundary files equal to native
  `init_atmosphere` case 9), `rw_igra2` (the IGRA2 radiosondes `woof hex
  score` reads for a case NOMADS no longer serves) and the RRFS fork dycore's
  `rw_mpas_rrfs_connectivity`, `rw_mpas_rrfs_damping` and
  `rw_mpas_rrfs_deformation`. Their bytes are pinned in
  `woof/hex/data/engine-bridges.json`; see
  [engine bridges](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/engine-bridges.md).

That is the prebuilt Linux wheel. A build from the sdist (it needs `cargo`)
always builds the host-preparation helper and the table decoders; the pool
reader and the template copier only where the netCDF-C development files are
installed; the scorer only when `GPUWM_HEX_ENGINE_SOURCE` names a woof source
checkout; and the nine bridges only when `WOOF_HEX_ENGINE_BRIDGES` names
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
4. this wheel's own `woof/hex/libexec`, which holds the bundled bridges pinned in
   `woof/hex/data/engine-bridges.json` (among them `rw_mpas_geometry`, `rw_mpas_stream`, `rw_mpas_static`, `rw_mpas_lbc`,
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
coordinates it declares (gpuwm 2.9.0's does not, and the door refuses its
statics by name). No released woof carries any of the three, so this wheel
carries them in `woof/hex/libexec`, pinned by SHA-256 to the woof commit they
were built from ([engine bridges](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/engine-bridges.md)). A flag or variable
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
[`docs/init-door.md`](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/init-door.md).

What the door accepts is not a promise but a measured table:
[`docs/source-matrix.md`](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/source-matrix.md) drives every source in the
RW-WPS registry with real bytes through intermediate, init and a
five-composite-step forecast, and records one verdict per source, a green
run with receipts, or the chain's refusal, verbatim.

`--vertical-spec` is the native-free path (see *Assets you must supply* and
[`docs/native-free-init-admission.md`](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/native-free-init-admission.md));
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
[`docs/render-door.md`](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/render-door.md).

## Five smaller doors

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
measurement of the atmosphere. `woof hex add-nest RUN --lat LAT --lon LON`
adds a WOOF-ArWen nest to a hex run while it runs, on the box already running the
parent: it starts from a frame the parent has closed (`--start now` by default,
or `--start -2h`) and is fed by the parent as it writes each frame; `--watch`
starts one for every nest a `[hex_nests]` trigger writes. A nest under a
59-level parent (every `hrrrv5` run) needs gpuwm 3.0.3 or newer, which pip
resolves for this release (`gpuwm>=3.0.0,<3.1`): gpuwm 3.0.0 to 3.0.2 read hex
frames as 55 levels and stop at the nest's preparation ("geopotential_height
requires N+1 finite interface heights"), and `pip install -U "recast-woof"`
moves such an install on. From the public packages (hex 0.6.0 with gpuwm 3.0.3),
a 3 km nest 300 km across under an `hrrrv5` parent, 3 hours from 2024-05-21
18Z, ran every stage with exit 0 on one RTX 5090 and wrote all four of its
hourly frames; its products include updraft-helicity maps. All five are listed
by `woof hex --help`.

## The differentiable engine (tangent-linear and adjoint)

> **In WOOF 1.1.7** recording a plan from a start of your own (`--init FILE`, `--init example`) is not supported: the derivative commands run on the example rows' own plans (`--row`).

Scope: the tangent-linear and adjoint model of the woof hex forecast step, dynamics and physics, reached through four
commands on one card of the newest NVIDIA family (compute capability 12.0). Three examples ship. The first is a 12 km
cut of the storm box (2,890 cells, a 15-minute window of 15 steps), the one that runs on a 32 GB card such as the
RTX 5090: start there. The second is the 3 km storm box (38,278 cells, a 15-minute window of 59 steps), the size of
mesh the engine is built for, on a 96 GB card. The third is a 120 km global example of three steps (36 minutes), on
a card with 73,156 MiB free. The checks the commands
print are consistency checks: they show that the adjoint is the transpose of the tangent-linear model. They do not
show how well the gradient predicts the forecast; that was measured separately and is summed up below. A measure
built on composite radar reflectivity does not pass the gradient check, and the commands refuse it by name.

**New to it?** [The first hour](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/differentiable/FIRST-HOUR.md) says which cards run it, what each example
needs, and how to read the first command's output line by line; [the guide](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/differentiable/GUIDE.md) covers a gradient of your own measure, the checks, recordings and their cost,
the C library for a variational system's linear model, exact against fast mode and the bit-for-bit claims; [the limits](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/differentiable/LIMITS.md)
lists every measured limit in one place. Runnable scripts with the numbers to expect travel in the wheel: `woof hex examples` prints their folder and `woof hex examples --copy DIR` copies them (the source archive and the repository hold the same files as `examples/differentiable/`).

**Which forecast the linear model is of.** The tangent and adjoint are of the exact-mode forecast
(`--physics-mode exact`): every plan is recorded in that mode, and the derivative rows are of its arithmetic.
`woof hex forecast` runs correct mode unless that word is given. Each of the four commands says so in one line and
names the mode in its receipt (`physics_mode`). `--physics-mode correct` is refused before anything is downloaded,
and a plan recorded in correct mode is refused by the recorder and by the commands, because this build has no
derivative rows of the corrected arithmetic: the gradient would not be that forecast's, and the
tangent-against-adjoint check could not tell.

**Cumulus at 12 km.** The 12 km row's derivative holds the cumulus scheme (New Tiedtke) by name: in a 12 km plan recorded with it on, its 26 launch kinds run as identity in the tangent and the adjoint, and `window.json` names them in its refuse list. The forecast is unchanged, and the 12 km example's recorded forecast runs no cumulus scheme. The cumulus derivative is not in this release: with it live, the 12 km window check misses by tens to hundreds of single-precision roundings, growing with the window.

**Species outputs at 12 km.** On the 12 km row the default route holds seven outputs of the boundary layer in both sweeps: its tendencies of cloud water, cloud ice and snow and of cloud-droplet, ice and the two aerosol numbers (`rqcblten`, `rqiblten`, `rqsblten`, `rncblten`, `rniblten`, `rnwfablten`, `rnifablten`); `window.json` names them. Live, a few of their derivative words are very large, and single precision scattered the window check. On the row as shipped (the transport limiter held) the README's direction read 8.6e-08 (-0.72 roundings) before the hold and 7.5e-08 (+0.63) with it; with the limiter's derivative live it read 2.0e-07 (+1.72) without the hold, and with the hold twelve seeded directions read within 4 roundings, worst 3.38, mean -0.15, standard error 0.38 (no one-signed lean). The 3 km and 120 km rows run without this hold: at 3 km it added scatter.

**The 12 km example: first, on a 32 GB card.** The release's own gradient test on a 12 km cut of the storm box
(2,890 cells of 12 km around 40.6N 91.3W, 59 levels), over a 15-minute window of 15 steps of a recorded forecast from
the HRRR 2024-05-21 15Z start, with a 1 K temperature bump over 150 km at 40.6N 91.3W:

```bash
python3.14 -m venv hexad && . hexad/bin/activate   # Debian and Ubuntu: sudo apt install python3-venv first
pip install "recast-woof[ad]"
woof hex adjoint-test --row hrrrv5-q12.96.2890.n40.60w91.30 --steps 15 --at 40.6,-91.3 --radius-km 150 --out cut12
```

It needs one card of compute capability 12.0 with 28,700 MiB free and 47,104 MiB of host memory, and Python 3.14
with the `ad` extra as above. Before it downloads anything it stops and says what to do on another card family,
in another Python or without the extra; a bump or region off the cut is refused after the 2 MB mesh file alone. It
tests the linear model a plain `sensitivity` on this row runs: the default boundary-layer route (`tape-free`, the 3 km storm row's
default, *The default route* below), and it says so in a `linear model` line; on this row the boundary layer's
seven species outputs are held in both sweeps (*Species outputs at 12 km*, above). Measured on 0.6.0's release builds, from fresh
installs: 320 to 645 s on a first run, which also fetches and unpacks its 1.20 GB of downloads (458 s, about 7.5 minutes,
on one RTX 5090 with the downloads fetched from GitHub; 320 s on one RTX PRO 6000 96 GB; 588 and 645 s on RTX 5090 hosts
shared with other work); card peak 24,766 to 25,360 MiB, host peak 41,190 to 44,109 MiB. On builds of
0.6.1, from fresh installs with the downloads fetched from GitHub, one RTX 5090: 418 to 708 s, card peak
24,766 MiB, host peak 40,990 to 42,989 MiB, with the same printed readings. On 0.6.2's, the same way, on a host whose
processors a package build shared: 769 s, card peak 24,766 MiB, host peak 43,491 MiB, the same readings; on another
build of 0.6.2, on one RTX PRO 6000 96 GB: 466 s, card peak 25,360 MiB, host peak 42,690 MiB, the same readings. What it prints: the bump's dot,
finite-difference, Taylor and zero checks pass (dot 1.1e-08, finite-difference and Taylor 9.9e-04), the whole-state dot passes (7.5e-08, +0.63 single-precision roundings of the dot against a bar of 4), and
the command ends **PASS** (exit status 0) on two directions inside the 0.01 bar: the bump (9.9e-04 from 1) and the
gradient's own steepest descent (5.0e-04 from 1). The other two read UNRESOLVED, which a PASS does not count
(near-ground moisture 7.2e-03 and the smooth draw 1.5e-02 from 1), and the PASS says nothing about them. Near-ground moisture is not inside the
bar: read before the species hold at ten step sizes of both signs on the same window, the real change is 1.010 to 1.049 of the gradient's
prediction at every one of them (centred, 1.021 to 1.036), so near the ground the gradient is about 2 to 4 per cent
short; the forecast's own roughness along that direction (up to 3.7 per cent between neighbouring step sizes) is why
the command's five step sizes cannot resolve it. The term behind it is the boundary-layer plume's own speed and
condensation, which this route holds: with them live (a form not in this release) the centred readings are 0.998 to
1.013. `--boundary-layer guarded` (an option on this row from 0.6.1) holds nothing of the boundary layer: a low-level
temperature measure on the storm cell, 2.2 per cent short on the default route, reads 0.996 to 1.000 with it (centred,
at steps 0.1 to 0.01, measured on a 0.6.0 build with this row added), and this
command ends NOT PASSED (UNRESOLVED) on it (466 s, card peak 23,610 MiB, on a pre-release build of 0.6.1), which is why it is not the default
([the limits](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/differentiable/LIMITS.md), *Storm measures*).
`--boundary-layer taped` tests the exact route's held boundary layer instead, and that ends **FAIL** (the
gradient's steepest descent 0.955 and near-ground moisture 0.88 of the real change). A `sensitivity` on this row runs
the same default route: after the test, `woof hex sensitivity --row hrrrv5-q12.96.2890.n40.60w91.30 --steps 15
--circle 40.6,-91.3,150 --levels 1-10 --out cut12s` took 55 to 119 s in the shell over the runs of 0.6.0's release builds (74.6 s on one RTX 5090 from the PyPI
install; its answer check is taken from the test's kept verdict), card peak 27,048 MiB, host peak 36,904 MiB, and 108.8 s
(card peak 27,048 MiB, host peak 37,116 MiB) on the pre-release build of 0.6.1 above; its map is one positive disc over the circle. `--steps` takes 1 to 15 here.

Under its directions the command prints the measured limits of the linear model it tests (`known limits`), each
with what this run read along that direction, and its last line names any direction the verdict does not cover
because of one (here: near-ground moisture). A known limit never changes the verdict or the exit status; the
receipt carries them as `known_limits`.

**What the last line and the exit status mean.** `PASS` (exit status 0): the first window's dot, finite-difference,
Taylor and zero checks and the whole-state dot are inside their bars, every direction ran, no direction read OUTSIDE
the 0.01 bar, and at least two independent directions read INSIDE. A PASS is of the directions printed INSIDE and of
nothing else: an UNRESOLVED direction is not covered, and the line says so when a known limit lies along one. `FAIL`
(exit status 1): a check missed its bar, or a direction read OUTSIDE; the line names it. `NOT PASSED (UNRESOLVED)`
(exit status 1): nothing failed, but fewer than two directions could be resolved. A command that stops before it
runs exits 2, or 3 when the host's memory or the run folder's disk is short. What each example reads today, per
direction against the 0.01 bar: at 12 km on the default route the bump and the descent are INSIDE and near-ground
moisture and the smooth draw UNRESOLVED (PASS); read with centred differences of both signs, a reading this
release's command does not make, near-ground moisture is OUTSIDE (2.6e-02 from 1), so that PASS does not stand for
it. At 3 km the default route's bump and descent are both inside the bar, so the command ends PASS (exit status
0); the exact route ends FAIL (*The default route*, below).

**The 3 km storm example.** The sensitivity of low-level temperature in a 150 km circle (model levels 1 to 10, around
40.6N 91.3W) at the end of a 15-minute window to the state at its start, on a recorded forecast from the HRRR
2023-03-31 18Z start. It needs a card with 96,600 MiB free (93,000 MiB on the exact route): a first call with its
answer check held 96,214 MiB by its own count on each of two 96 GB RTX PRO 6000 Blackwell cards, and the card was
sampled at 96,571 MiB (95,666 and 96,023 MiB on the build before):

```bash
woof hex sensitivity --row hrrrv5-q3.96.38278.n40.60w91.30 --steps 59 --circle 40.6,-91.3,150 --levels 1-10 --out storm
```

What it needs and what it costs, on one 96 GB RTX PRO 6000 Blackwell card. Each figure was read from a single run
of this command unless it says otherwise, on builds of 0.6.0: the 3 km example was not run on a 0.6.1 build, and
0.6.1 changes none of its downloads. "The build before" and "earlier build" are earlier builds of 0.6.0; the build
before runs the 3 km routes 0.6.0 shipped: the forward kept between gradients, the refined cloud derivative as the
default, and the window end read where the plan's own map puts it. The window-end read moved
every 3 km answer by one step, so the build before's answers are compared with earlier builds' as readings only, not
byte for byte:

- **Card memory:** 96,600 MiB free on the default route, 93,000 MiB on the exact one. At peak, by the command's own
  count, the default route held 89,966 MiB and 91,648 MiB over two targets and 95,666 MiB on a first call with its
  answer check (the card itself was sampled at 96,023 MiB during that call); the exact route held 89,040 MiB, and
  90,014 MiB with its check (measured, the build before).
- **Host memory, three figures.** After the mesh file and before the rest of the example the command states what it will hold and which case
  it judged, and stops on a machine with less. On a machine with 400 GB of memory: 158,617 MiB for a gradient that
  runs no check and keeps no forward; **249,661 MiB for a gradient whose program keeps its forward for the next
  gradient of the same forecast**, which is this route's default (91,044 MiB more: nine stored states of
  10,116 MiB; the loaded program's own count of what it kept read 91,042 MiB); 258,057 MiB for a call
  that runs the answer check (the first call on a plan, or `--strict`). On a machine with room for the gradient
  and not for the kept forward the command runs without it and says so; `--keep-forward off` asks for that.
  Measured (the build before): a two-target command with the forward kept was admitted on 249,661 MiB and peaked
  at 233,301 MiB; the same command on the exact route, which keeps no forward, on 158,617 MiB and 151,835 MiB; a
  call with its answer check, admitted on 258,057 MiB, peaked at 241,895 MiB on the default route and 243,477 MiB
  on the exact one. A refusal names the case it judged, and says so when the gradient alone
  would fit. The command keeps ten states of the window in host memory, about 10 GiB each; on a machine with less
  memory to spare it writes some of them to the output folder instead (and then keeps no forward), and states that
  disk before it runs.
- **Host memory of `adjoint-test`, `tangent` and `param-grad` on this row.** Their windows keep every step
  boundary (60 of 10,116 MiB at 59 steps). The runner keeps them in host memory up to 0.4 of what is available
  when a window starts and writes the rest to the run folder, so the need follows the host, and the command
  states it for this host before it downloads the example: 1.05 x (the runner's own memory, 87,805 MiB at 59
  steps, plus the boundaries kept in memory). Measured, `adjoint-test --steps 59`, each once on a 512 GiB host:
  262,306 MiB on the exact route and 254,885 MiB on the default route, with at most 19 boundaries in memory and
  41 to 48 in the run folder (435 to 509 GB there). On a host too small for the runner's share the command gives
  it a smaller store in memory (`--bstore-ram-gib`, said in the output: the same numbers, more in the run folder):
  with 120,000 MiB available it runs on 119,076 MiB with 25 GiB of boundaries in memory and states 585,456 MiB of
  disk. A host with less than 102,948 MiB available is refused in one sentence; one with 180,000 MiB available is
  admitted on 167,796 MiB. On a 499 GiB host (2026-11-16, a 0.6.0 pre-release build) it stated 303,410 MiB and held
  278,824 MiB with 19 of 60 boundaries in memory, PASS with the same digits. Not measured: any host smaller than
  499 GiB, and the time the extra files cost.
  The candidate before this change stated 110,592 MiB for this command, a 3-step figure.
- **Disk:** about 2.7 GB of downloads, once, each checked by SHA-256 (the mesh, a one-hour plan in two parts, the
  operator files, the kernel set and three small files of compiled images). The output folder peaked at 1,043 MiB
  (measured); the command states 4,096 MiB.
- **Seconds, the first call:** 12.3 minutes (737.6 s in the shell, 698.6 s counted by the command) for one target on
  the default route: 243.0 s for the gradient, 416.7 s for the answer check, which runs once per plan, is kept, and
  read 2.78e-08 against the 1e-07 bar that build's runner used (pass), and the rest for fetching and checking the files
  (here from a local disk), the answer files and the map (measured, the build before). On 0.6.0's final wheel (one RTX PRO
  6000 96 GB) the same first call took 630.4 s and its answer check read 1.786e-07, which is -1.50 single-precision
  roundings against the bar of 4 (4.77e-07): the bar is counted in roundings, not at 1e-07. On the exact route a call with its check took
  707.8 s (11.8 minutes) on a program already loaded: 259.7 s for the gradient and 428.5 s for the check, which read
  1.75e-08 (measured, the build before).
- **Seconds, a later command on the same plan: the program stays loaded.** A `sensitivity` command on this row
  leaves its program loaded on the card, and the next command attaches to it instead of starting one: its first
  line reads `ATTACHED` or `STARTED FRESH`, with the reason. Measured (the build before): a later command that
  attached took 181.1 s in the shell (180.3 s counted, 176.9 s of it the gradient, which took the kept forward), and
  its answer was byte-equal to a fresh program's. Earlier builds read 205.3 s for it with no forward kept, and
  305.9 s where a later command started its own program. The first
  call on a plan leaves no program loaded (its answer check needs the card); the second command starts one and the
  third attaches. A command that attached and then runs a check of its own (`--strict`) ends the loaded program and
  waits up to 120 s for the card that program is giving back. A command that finds a program loaded with other
  words (another route, cloud option or window length) stops it and waits for the card and the host memory it
  gives back before it reads either. On the exact route a later command took 357 s, 339 s
  of it the gradient (measured, earlier build).
- **The forward is kept between gradients of one forecast.** On the default route a further target in the same
  command, and a later command that attaches, start from the same forecast start as the gradient before. The
  loaded program then does not run the forward again: it takes the forward it kept and goes straight to the
  backward sweep. The command prints which happened for every gradient (`forward: KEPT` or `forward: ran its own`).
  The program compares a fingerprint of its start with the kept forward's and runs the forward itself when they
  differ; through this command another plan, window length or route is another program, which runs its own
  forward. Measured through this command on the build before this one (59 steps, one 96 GB card, once each): a
  further target in the same command 180.5 s (24.4 s of forward steps not run), and a later command that attached
  176.9 s (181.1 s in the shell), with the same answer bytes as a fresh program. Earlier builds read 184.4 s for a
  further target with the forward kept, and 195.7 s and 198.2 s without it. A program's first gradient, which also
  fills the kept forward, took 239.5 s: the kept checkpoints' buffers are page-locked on a helper thread while the
  forward runs (`AD_KF_PREFILL=1` in the runner's words). An earlier build page-locked them inline in its sweep and
  read 247.3 s, against 236.3 s with `--keep-forward off`. It holds nine more states in host memory between
  gradients (91,042 MiB by the loaded program's own count; the command counts 91,044 MiB, and on a short window,
  where a later gradient holds them twice, twice that: a 12-step window states 219,313 MiB and held 182,610 MiB,
  measured).
  `--keep-forward off` runs without it. It is off on the exact route: with it the exact route gave the exact bytes
  over 12 steps, and no 59-step exact run with it has read.
- **Stopping a loaded program.** It ends by itself after 600 s without a request (`--session-idle-s N` changes
  that) and frees the card and the host memory it holds (213,931 MiB after two gradients with the forward kept, and
  142,632 MiB after two on the exact route, which keeps none; measured, the build before). `woof hex sensitivity --session-stop` ends it now, `--session-list` shows what is loaded, and
  `--session off` keeps none. A command attaches only to a program of the same plan, build, route, window length
  and words; anything else is stopped and started fresh, and the first line says what differed. In a batch job none
  is kept unless `--session on` is given. From Python, `woof.hex.ad_session.SensitivitySession` holds one program
  for a loop of targets.
- **Seconds, more targets: pass them in one command.** With `--objectives` every target after the first runs on the
  program already loaded. Measured (the build before, default route, two targets in one command, the second taking
  the first one's forward): the first gradient took 239.5 s and the second 180.5 s (243.0 s and 183.7 s with each
  target's answer files and map; 427.7 s for the whole command in the shell). On the exact route the first took
  289.6 s and the second 260.8 s (576.2 s in the shell; measured once, the build before). Earlier builds read
  247.3 s and 184.4 s with the forward kept, 239.5 s and 195.7 s without it, and 282.3 s and 254.8 s on the exact
  route.
  Those seconds were measured with the boundary-layer images of the builds before this one. With this build's default
  route, through this command on one 96 GB card (16 Nov): a first gradient 231.7 s, a further target in the same
  command 179.8 s, a later command that attached 178.1 s, and a first call with its answer check 617.6 s and 630.4 s
  (two runs). On the 12 km cut its gradient took 22.6 s and 14.7 s (first and later)
  against 23.4 s and 15.0 s with the images before, in one session on one card.

**The default route is fast, and differentiates more of the boundary layer than the exact route.** On this row
the boundary layer's backward runs without its tape, and from this build its linear model differentiates the
boundary layer's turbulence closure (the stability functions and the turbulence-energy terms), the plume's
transport of heat, moisture and momentum (with the plume's own coefficients held) and the boundary-layer height the
column hands to the surface layer, under a weak-shear rule: an interface whose forecast squared wind shear is under
1e-5 s-2 keeps its stability functions' value. The mixing length and the plume's own speed, area and condensation
stay held. The exact route holds all of these at the forecast's values, as the default route did before this
build, so the two routes' answers differ by design and not by rounding, and the command prints no
fast-against-exact figure: the figures the earlier builds printed were of the held form and are not true of this
one. On the 12 km cut of this storm box over 15 steps (one RTX 5090, 32 GB; `adjoint-test` on the 12 km example row,
which runs this route by default, above) the test along its four directions read PASS with this build's default
route, read on 0.6.0's release build: bump inside (9.91e-04 from 1); descent inside (5.00e-04 from 1); near-ground moisture and the smooth draw unresolved (7.2e-03 and 1.5e-02 from 1). With the held
form the same test read FAIL on the exact route (the gradient's own descent direction 4.4 per cent and near-ground
moisture 12 per cent off the real model's change) and FAIL (bump inside, 0.0086 per cent off; descent OUTSIDE, 4.4 per cent off; near-ground-moisture OUTSIDE, 12 per cent off; smooth-draw unresolved (the real change read 0.899 to 0.956 of the gradient's prediction over the step sizes that count)) on the default route of the builds before.
Near-ground moisture's UNRESOLVED is not a pass: at ten step sizes of both signs the real change read 1.010 to 1.049
of the gradient's prediction (about 2 to 4 per cent short; the held plume speed and condensation, above). The
window's consistency check on this build read +0.63, +0.33, -0.87, -0.25, +0.96, -0.34, +0.11 and +2.26
single-precision roundings of the dot on seeded directions 1 to 8 (`adjoint-test --dot-directions 8`, one window,
one RTX 5090, exit status 0): each within the bar of 4, and no one-signed drift (5 of 8 positive, mean +0.35,
standard error 0.34, within 2 standard errors of zero). With the transport limiter's derivative live (not this row's default), twelve seeded directions read within 4 roundings: worst 3.38, mean -0.15, standard error 0.38, no one-signed lean.
Through this command on the 3 km example both routes have
been read, each once on one 96 GB RTX PRO 6000 Blackwell, and `adjoint-test --row hrrrv5-q3.96.38278.n40.60w91.30
--steps 59` ended **PASS** on the default route (exit status 0) and **FAIL** on the exact route.
The default route (the number cloud route, the live transport limiter and the three double-arithmetic dynamics
images): window dot 1.317e-08 (bar 2e-05) and whole-state dot 1.786e-07 relative (-1.50 roundings; bar 4; the 12
model fields), both PASS; the first window's finite-difference and Taylor checks 3.653e-04 against the 1e-02 bar,
PASS; at the row's own step sizes 0.3, 0.2, 0.1, 0.06 and 0.03 the bump 0.9996 / 0.9993 / 0.9989 / 0.9987 /
0.9996 of the real change, INSIDE (3.65e-04 from 1), and the gradient's steepest descent 0.9842 / 0.9894 / 0.9947
/ 0.9951 / 0.9981, INSIDE (1.92e-03 from 1); 2,751 s, card peak 94,162 MiB, host peak 420,360 MiB (the command
asked for 450,866 MiB of host on this machine). The row's step sizes stop at 0.03 because at 0.01 the descent
moves the measure (a mean temperature near 348 K) by only about 5 single-precision spacings of it, so that size
reads rounding, not the derivative; at sizes 1 to 0.01 the same direction read 0.9332 to 1.0044, UNRESOLVED. The exact route (the held boundary layer, now `--boundary-layer taped`), through the build
before this one: its window dot read 2.825e-08 (bar
2e-05) and its whole-state dot 1.313e-08 (0.11 roundings; bar 4), both PASS; its first window's finite-difference and Taylor
checks read 1.017e-02 against the 1e-02 bar, FAIL; the bump read 1.0102 to 1.0133 of the real change, UNRESOLVED
(1.02e-02 from 1); the gradient's steepest descent read 0.873 to 0.971, OUTSIDE (2.86e-02 from 1). It took 2,642 s,
a card peak of 84,942 MiB and a host peak of 262,306 MiB by its own count. Two directions (the bump and the descent) are shown inside the bar
at 3 km on the default route, none on the exact route.
`--boundary-layer taped` runs the exact route of the held boundary layer, the one to use for byte-for-byte
comparisons. Against the default route of the build before this one it cost 50 s more on a first gradient and 80 s
more on a further target (289.6 s against 239.5 s, and 260.8 s
against 180.5 s, where the default route takes the forward kept from the first target and the exact route runs its
own; measured once through this command on the build before this one, both routes on the same card, kernel set and
radiation kernels). A call with its answer check and one without gave the same exact answer bytes; against earlier
builds its answers moved with the window-end read, as the default route's did.
**The exact route is for windows up to about 30 minutes at 3 km, measured on this example.** An exact derivative of
the transport limiter grows where the forecast itself oscillates from step to step, so a longer window needs the
limiter's coefficients held; windows past that are for the default route. Asked for a longer window, the exact
route prints a one-line warning and still runs. The default route's compiled images are built for this card
family only, and on another the command stops and says so.

**What the answer check shows, and in which direction.** The check is the whole-state dot test over the window asked
for: every value of the start state moved at once, in a random direction. In that direction the 3 km check read,
through this command, 2.78e-08 on the default route and 1.75e-08 on the exact one over the 15-minute window (59
steps; 0.23 and 0.15 single-precision roundings of the dot against a bar of 4, pass; measured on the build before, with the refined cloud derivative). Those readings are of the boundary-layer images of the builds before this one; with this build's default route the check read 1.786e-07 on the 3 km example (-1.50 roundings against the bar of 4, pass; through `adjoint-test` and through a first `sensitivity`, each on one 96 GB card) and 7.49e-08 on the 12 km example (+0.63 roundings). Earlier builds, with the
cloud derivative of the builds before, read 9.1e-09 and 1.3e-08. In the direction
of a storm sensitivity (a 1 K warm bump low under an updraft, against the mean upward wind above it; measured on a
3 km recording of another storm day, with this kernel set) it reads 1.7e-06 on the default route and 2.2e-06 on the
exact one over the same window (14 and 18 roundings): over the bar on both. Over 12 steps it reads 1.6e-07 at the
storm (1.3 roundings) and 6.3e-08 a hundred km from it (0.5): both within the bar (the bar before this build, 1e-07,
failed the first). The command prints this beside its own verdict, and the verdict it prints is the random
direction's only.

**The answer check's bar, and why it is in roundings.** The check compares two numbers the engine computes from the same window: the tangent-linear model's `<y, M x>` and the adjoint's `<M^T y, x>`. If the adjoint is the transpose of the tangent-linear model they are equal, except for rounding. The model and its derivative run in single precision, so the two cannot agree to better than about one single-precision rounding of the dot (2^-23 of its size, 1.2e-07 relative). The bar is therefore stated in those roundings: the two sides must agree within 4 roundings (4.8e-07 relative) on every direction tested, and over 8 or more directions (`adjoint-test --dot-directions 8`, what the release check runs) their signed differences must not lean one way (the mean within 2 standard errors of zero). A missing or wrong derivative term reads tens to hundreds of roundings, or leans one way over many directions; rounding does neither. The bar before this build, 1e-07, was below one rounding, so a correct build passed or failed it by luck; `--dot-bar strict` still applies it. The check shows that the adjoint is the transpose of the tangent-linear model within 4 single-precision roundings; it is not a claim of exactness, and it says nothing about how well the gradient predicts the forecast. That is the separate gradient test against the real model (the `fd`, `taylor` and direction lines), with its own 1 per cent bar.

**Open accuracy items at 3 km.** For moisture and vertical-wind targets, the linear model's response has not been
shown to follow the real model. In the storm test above, the linear answer for the mean upward wind was 0.00023 m/s
per K of warm bump, and the real model gave 0.00069, 0.00077 and 0.00044 m/s per K for bumps of 1, 0.5 and 0.25 K.
Moisture after a low-level warm bump is not predicted at any window tried. Read 3 km sensitivities of moisture and
vertical wind as not yet validated. The wind part of a one-card gradient now carries the first step's surface-layer
path, which every earlier build left out. **Near-ground wind on the exact route, and on the default route before this build, is not right.** Along a low-level
wind nudge over the whole box the 12-step gradient of the held boundary layer read 1.37 times the real model's change
with the gated cloud derivative (not yet run along it with the number-following default), about a third too large; with `--cloud-derivative shipped` its
terms cancelled to about zero (the last row of the table below; measured on a 3 km recording of another storm day).
The cause of the 1.37 is the boundary layer's mixing, which the held linear model keeps at the forecast's values while
the real model moves it with the wind. From this build the default route differentiates the turbulence closure
(above); it has not been run along that nudge through this command. With the mixing differentiated and nothing of the boundary layer held
the same test read 1.006 over 12 steps, and over 59 steps that form ran away in one calm column. The command prints
this caution on every 3 km sensitivity.

`--boundary-layer guarded` (on the 3 km storm row; from 0.6.1 the 12 km example row takes it too, as above; not the default) runs the exact route with the boundary layer's
mixing differentiated instead of held, under one rule: where the forecast's own squared wind shear between two
model levels is under 1e-4 s-2, that interface's stability functions keep their value and carry no slope. Where
the rule binds it also removes the slope of the scheme's plume and cloud floors and of the turbulence energy
through those functions. Shown: over 12 steps on a 3 km recording of another storm day the gradient read 1.003 of
the real model's change along the box-wide low-level wind nudge that reads 1.37 on the other routes; on a 12 km
cut over 15 steps its consistency check passes with three cloud image sets and three seeds where the unguarded
form fails. Through the command on one 96 GB card, 59 steps with the default cloud derivative (one target with its
answer check): on 0.6.0's final wheel the whole-state check read 1.304e-07, +1.09 single-precision roundings against
the bar of 4 (4.77e-07), in 1,487.7 s with a host peak of 241,845 MiB. On the build before (the same digits on two
builds) it read 5.34e-08, no start field had a one-value spike (the largest value's share of the temperature field
4.0e-03), and it took 1,478 s and 243,432 MiB of host memory. Not shown: the gradient against the real model at 59 steps, the storm-direction
consistency check with the default cloud images, and other
directions (a near-surface measure's own descent direction read 0.67 to 0.71 of the real change for surface wind
and 0.35 to 0.44 for surface temperature on the 12 km cut over 15 minutes). The rule holds 43 per cent of the ten
lowest interfaces at 12 km; at a weak-shear place the cross-wind response reads about 0.13 under the unguarded
form. It is slower than the exact route and keeps no forward between gradients.

At 3 km the gradient of a cloud or rain target can be wrong in size and in
sign at single columns, because the cloud scheme's derivative mishandles a ratio of two rates that are both near zero
there. Measured over a 12-step window on a 3 km recording of another storm day: in the worst column tested, a
moisture bump's predicted change had the wrong sign and a temperature bump's was 5.6 times too large, while at three
healthy test points in the storm (a wind patch, a warm bump and a moist bump) the gradient agreed with the real model
to within 3 per cent. Such columns held a map's largest single-cell values. **The fix is in the 3 km storm row's
default from 0.6.0 (`--cloud-derivative number`, which keeps it; `--cloud-derivative gated` runs it alone; `--cloud-derivative shipped` runs the derivative of the builds
before).** It holds that one ratio (snow riming over snow deposition) at its forecast value where riming is too slow
to add the snow's own mass in an hour, and the riming fall-speed boost where a trace of cloud water would carry it to
its cap, and keeps the true derivative elsewhere; the forecast is unchanged, and the gradient's bytes change wherever
snow rimes a trace of cloud water. Measured over the same 12-step window, the gradient's prediction over the real
model's change, the builds before then the fix:

| test direction | `--cloud-derivative shipped` | `gated` |
|---|---:|---:|
| vapour at the one worst word | -538 | 0.999 |
| +1 K at the worst column | 5.62 | 0.973 |
| +1 g/kg at the worst column | -2.10 | 0.999 |
| +1 K, a healthy point | 1.001 | 1.001 |
| +1 g/kg, a healthy point | 1.019 | 1.019 |
| a storm-scale wind patch | 0.975 | 0.970 |
| low-level wind over the whole box | 0 | 1.367 |

Six of the seven directions are within 0.1 to 3 per cent with the fix. The last, near-ground wind, is still 37
per cent high: that is the boundary layer's held mixing (above), not the cloud scheme. Over the whole 59-step window
(engine's own route, one 96 GB card) the largest single moisture word holds 0.4 per cent of the squared moisture
sensitivity, where it held about 40 per cent before; the consistency check in a random direction reads 9.2e-08 on
the default route and 7.8e-08 on the exact route (against the 1e-07 bar that build's runner used, pass), and in a storm direction 2.06e-06 against
1.67e-06 before (both over the bar). Those readings ran images whose tangent and taped forms held two process rates in
single precision; this release ships the repaired images (the default route's gradient runs the block reverse, which
is the same bytes). Through this command, as the default, over the whole window they read 2.78e-08 on the default
route and 1.75e-08 on the exact one (measured, the build before). On the first day's recording the repaired
images read 3.9e-08 over 12 steps and 6.9e-08 and 4.9e-08 over 59 steps (default and exact route; pass), and in
the storm direction 2.14e-06 over 59 steps, over the bar as before the repair.

**Reflectivity: rain and droplet number follow the forecast (`--cloud-derivative number`, the 3 km storm row's
default from 0.6.0).** The gated form above holds rain drop number and cloud droplet number at their forecast
values. That makes the gradient of reflectivity wrong in size where it rains. Measured with both numbers held (an earlier form of the derivative,
which gates deposition at 200 s where gated holds it; gated itself was not read on this measure)
on this row's example over 59 steps,
along a nudge of the rain mixing ratio read on both sides, the predicted change of reflectivity summed over the
volume was 1.251 / 1.392 / 1.461 times the real model's change at nudges of 50 / 15 / 5 per cent of the rain, and on
a 12 km storm cut over 15 steps the predicted change of echo over 35 dBZ was 0.37 to 0.46 of it. The default now
lets both numbers follow, gives the rain-snow and rain-graupel collection tables their own slope between bins, and
holds snow and cloud-ice deposition only where it would add the species' own mass within 200 s. Measured the same
way:

| along the rain nudge, 59 steps | nudge 50 % | 15 % | 5 % |
|---|---:|---:|---:|
| summed reflectivity, numbers held (the earlier form) | 1.251 | 1.392 | 1.461 |
| summed reflectivity, `number` (default) | 0.930 | 1.035 | 1.086 |
| snow mass, earlier form / `number` | 1.490 / 0.953 | 1.449 / 0.926 | 1.504 / 0.961 |
| vapour mass, earlier form / `number` | 0.849 / 0.930 | 0.864 / 0.947 | 0.862 / 0.945 |

What a reflectivity gradient can be compared with: below a 5 per cent nudge of the rain the real model's own change
of summed reflectivity is not steady (echo appears and vanishes word by word, and the collection tables step
between bins), so no gradient can be judged against it there; echo over 35 dBZ alone and the column maximum are not
steady at any nudge size on this example, which is why no composite-reflectivity measure is offered. These
readings ran the bare engine line's runner with this row's kernel set and only the cloud images changed, one 96 GB
card, 15 Nov 2026; they have not yet been run through this command. The window consistency check of these images on a 12 km storm cut over 14 steps reads within
four single-precision roundings of the dot on all twelve directions read, with no one-signed drift (mean
+0.18, standard error 0.54); holding both numbers read 0.04 to 0.36 on two of them, so the numbers' derivative widens the
single-precision scatter, inside the bar. The forecast is the same under every
`--cloud-derivative` word.
The number route needs the transport limiter's live derivative (`--fct-frozen 0`, which this row runs on windows up
to 30 minutes): with the limiter's coefficients held, rain number's tangent grew until it was not finite, and this row's
adjoint-test stopped at step 28 of 59 on rain mass, rain number and reflectivity. On a longer window the command runs
`gated` and says so in one line; `--cloud-derivative number` there is refused. With the limiter live, through this
command on one 96 GB RTX PRO 6000 Blackwell (15 Nov 2026, once, on the line's build with this route and the live
limiter), `adjoint-test --row hrrrv5-q3.96.38278.n40.60w91.30 --steps 59` read: window dot 1.758e-08 (bar 2e-05);
finite-difference and Taylor checks 3.20e-04 (bar 1e-02); the bump 1.0020 / 0.9996 / 0.9989 / 0.9996 / 1.0003 of the
real change at step sizes 1 / 0.3 / 0.1 / 0.03 / 0.01, INSIDE (3.20e-04 from 1); the gradient's steepest descent
0.933 / 0.984 / 0.994 / 1.000 / 0.992, INSIDE (1.50e-04 from 1); the whole-state dot 1.903e-07, 1.60 single-precision
roundings of the dot, over the 1e-07 bar that build's runner judged by and inside the four roundings this release's
check judges by; 2,977 s, card peak 94,198 MiB, host peak 218,404 MiB. One direction of the whole-state dot was read,
so the check's no-drift half is not shown at 3 km.

**A second storm day.** The same 12-step comparison was run on the example's own day (the 2023-03-31 18Z start)
with the repaired images, for the mean upward wind on levels 27 to 38 within 150 km of the box centre, one run
each on a 96 GB card. A row is counted where the real model's two sides agree within 25 per cent. Along the seven
such storm directions drawn on that day (temperature and moisture at low, middle and upper levels, wind aloft) the
gradient reads 0.995 to 1.004 of the real model's change; along seven more inside or within 20 km of the target,
0.986 to 1.025. Over 170 km from the centre, where the target barely moves, five of seven are within 10 per cent
and two read -0.105 and 1.255. Eighteen of the 39 rows are not counted: the real model's own two sides disagree
there. With `--cloud-derivative shipped` every counted row reads the same to three digits. So the second day shows that
the fix leaves healthy directions as they were. It does not show a bad column repaired a second time: none of
these directions meets one, and a cloud or rain target has not been measured on that day.

**Which rows run which cloud files.**

| row | cloud derivative files | why |
|---|---|---|
| the 3 km storm box | `number` (default): the three `cloud-number/*-0x06-rd3600n100-nfr-nfc-ts-dg200.cubin` images of `ad-aux-cloud-number-sm120-v1.tar.gz`; `--cloud-derivative gated`: the three `cloud-gated/*-0xc6-rd3600n100.cubin` images of `ad-aux-cloud-gated-sm120-v2.tar.gz`; `--cloud-derivative shipped`: the kernel set's own cloud images | the fix above, with rain and droplet number following (reflectivity, above): the bad columns and the held-number error were both found at 3 km |
| the 12 km cut of the storm box | the kernel set's own cloud images; no `--cloud-derivative` option | the gated form has been compared with the real model at 3 km only |
| the 120 km global row and any row made with `--mesh` | the kernel set's own cloud images; no `--cloud-derivative` option | measured on a 96 km globe, below |

On a 96 km globe (one seeded direction, one run each, one 96 GB card) the gated images, before that repair, failed
the window check over 8 steps: 5.2e-05, with 141,698 of the tangent's forward values differing from the
forecast's. That was the single-precision fault named above: the repaired images read 1.5e-08 over the
same 8 steps with every forward value equal, beside 9.8e-09 for the globe's own images. Over 16 steps (2 hours)
the repaired images read 1.6e-07, over the 1e-07 bar that build's runner used, where the globe's own images read 2.9e-09; with the
double-precision tangent of the same rule, which is packed but which no command runs, 2.7e-08. So no global row
runs the gated form in this release.

Not yet measured: windows of 30 and 60 minutes, and a cloud or rain target on a second storm day. Separately, the gradient
damps a near-ground wind nudge about twice as fast as the real model does in its first physics step (measured).

**Which boundary layer the derivative is of.** The tangent and the adjoint of the boundary layer are generated from
MYNN-EDMF at commit c6d08709, the code NOAA's RRFS 44c0849 runs, and those of the surface layer from the MYNN
surface layer v6.0.0. A plain hrrrv5 forecast of this release runs MYNN-EDMF v6.1.1 and the MYNN surface layer
v6.1.0 by default. c6d08709 leaves the latent heat of the scheme's own cloud mixing out of the boundary layer's temperature
tendency, so where the boundary layer mixes cloud (stratocumulus, low stratus under an inversion) a sensitivity is
that of the older boundary layer, not of the one a plain forecast runs. The two surface layers differ only at the
first step of a run. The default route's form (above) is of the same c6d08709 code: it differentiates more of that
boundary layer, not a newer one. The derivatives of v6.1.1 and v6.1.0 are not in this release (the derivative of
v6.1.1 is planned for the next build); `docs/differentiable/README.md` gives the size of the difference.

**Who it is for, and what card it needs today.** It is built for labs and agencies with large NVIDIA cards, 48 to
96 GB. This release runs on **one card**, and only on the newest card family (compute capability 12.0). The 12 km
example needs **28,700 MiB free** on its default route (25,000 MiB with `--boundary-layer taped`) and has run on a
32 GB RTX 5090; the 120 km example needs **73,156 MiB free**
(80,000 MiB for the full-physics row) and the 3 km storm box 96,600 MiB, and those have run on a 96 GB RTX PRO 6000
Blackwell. Before it downloads anything, a command stops and says why on a card of an older family (A100, H100,
L40S, RTX A6000, RTX 6000 Ada) and on a card whose whole memory is smaller than its row needs (a 32 or 48 GB card for
the 120 km and 3 km examples, an 80 GB card for the 3 km box). Free card memory is checked after the row's mesh file
(2 MB for the 12 km cut) and before the rest of the example, and a card with less free memory than its row needs is
refused there. Splitting one gradient over several cards (`--cards 0,1` on `adjoint-test` and `sensitivity`) is experimental and claimed for nothing: a split gradient has not yet been shown equal to the one-card gradient, and the command says so before it runs and in its receipt. Measured on an earlier build of the release: on the 3 km storm box a 59-step gradient on two cards took 436 s against 489 s on one card of the same machine, with the wind part of the split answer 7.6 per cent off the one-card answer. On 0.6.0's final wheel the one-card gradient takes 180 to 232 s, so a split gradient is not shown faster either.

**Card families, and what was measured on each.**

| Card family (compute capability) | Cards | The forecast | The derivative commands |
| --- | --- | --- | --- |
| Blackwell (12.0) | GeForce RTX 50 series, RTX PRO Blackwell | Runs. One hour of the 12 km cut's forecast on NOAA's hrrrv5 deck, in exact mode with strict numerics, matches the patched native RRFS-MPAS reference in every value of its 27 history, diagnostic and restart files, one RTX 5090: 85,508,314 values without the cumulus scheme (as the 12 km example runs), 86,751,014 values with NOAA's 12 km cumulus on (as the deck runs it). | Run. Every derivative figure in this README was read on an RTX 5090 or an RTX PRO 6000 Blackwell. |
| Ada Lovelace (8.9) | GeForce RTX 40 series, L4, L40S, RTX 6000 Ada | Runs. The same two hours match the same references in every value of the same 27 files (one RTX 4090, 24 GB, 2026-11-16). | Refused before any download. No derivative command has been run on this family. |
| Hopper (9.0), Ampere (8.0 and 8.6) | H100, H200, A100, A10, A40, RTX A6000 | Not run for this release. The forecast admits any card of compute capability 7.5 or newer by the same rule (*Requirements*, above). Not compared with the native reference on these families. | Refused before any download. No derivative command has been run on these families. |

The derivative commands are tied to a card family twice over: their compiled images are one family's code, and an
example plan replays only on the family it was recorded on, because its forward kernel images are that family's code.
The shipped images and plans are Blackwell's. The forecast is not tied this way: it compiles its kernels for the card
it finds.

A recording of an ordinary hex forecast is replayed launch by launch, and every kernel launch is paired with a
derivative rule or a derivative kernel, so the tangent and the adjoint follow the forecast the engine actually ran.
Gradients are a separate process that reads the recording; the forecast command does not run them.

**The 120 km example: the quick check that an install works on a large card.** The same install, four commands; the
first on a plan runs the answer check, and `tangent` and `sensitivity` after it take its verdict (seconds under "What
the commands use" below). They need a card with 73,156 MiB free (80,000 MiB with `--row hrrrv5-full-x1.40962`): at
peak `adjoint-test` held 71,156 MiB, and 78,206 MiB on the full-physics row (one 96 GB RTX PRO 6000 Blackwell, the
build before):

```bash
woof hex adjoint-test --out at
woof hex tangent      --steps 3 --out tl
woof hex sensitivity  --steps 3 --metric theta_m --box 30,48,-105,-80 --levels 1-10 --out se
woof hex param-grad   --steps 3 --check --out pg
```

On this example `adjoint-test` runs its own window, 1 step (12 minutes), and reruns the real model at eight step
sizes a factor of two apart (2 to 0.015); `tangent`, `sensitivity` and `param-grad` keep the row's 3 steps. On a
working install it ends **PASS**, exit status 0: the checks and the whole-state dot pass (fd and Taylor 2.1e-04
against 1e-02, whole-state dot 7.7e-09, 0.06 single-precision roundings of the dot against 4) and the bump and near-ground moisture are inside the bar (2.1e-04
and 8.5e-04 from 1); card peak 60,734 MiB, host peak 73,994 to 74,072 MiB, 271.9 to 307 s (two runs, one 96 GB RTX PRO 6000 Blackwell each, 15 Nov). The steepest
descent and the smooth draw stay UNRESOLVED and the command names both limits under its directions. Over the row's
3-step (36 minute) window the test cannot pass: at 120 km the real change per unit step moves by 1 to 6 per cent
between neighbouring step sizes, so only the bump resolves and `--steps 3` ends NOT PASSED (UNRESOLVED), exit status 1.
Over 2 steps (24 minutes) the steepest descent settles at 0.977 to 0.987 of the real change and reads OUTSIDE the bar
(1.74e-02 from 1), so `--steps 2` FAILS; centred (each step size at both signs) it reads 0.986 to 0.992, a gap
of about 1 per cent along minus the gradient that the 1-step window cannot show and that is not yet explained (`docs/differentiable/LIMITS.md`). Otherwise a failed check or an OUTSIDE
direction is what a broken install shows. Because the test's verdict is for its 1-step window, the first 3-step
command on a plan runs its own answer check. **With `--row hrrrv5-full-x1.40962` the same command also PASSES**, exit
status 0: 3 directions resolved (bump 1.3e-04, descent 1.4e-03, near-ground moisture 3.3e-04 from 1), whole-state dot
1.7e-08, 306 s on one 96 GB card. Its boundary layer runs the floored form by default (`--boundary-layer floored`):
the turbulence-energy path of the live mixing is bounded where the turbulence speed goes to zero. Without the floor
(`--boundary-layer taped`, the row's own images) the descent reads OUTSIDE, 0.949 and 0.954 of the real change centred
(`docs/differentiable/LIMITS.md`).

Use Python 3.14 and the `ad` extra: the example replays CUDA library files of exactly cupy-cuda13x 14.2.0 and
cuda-toolkit 13.4.3 built for CPython 3.14, and the `ad` extra pins both. In any other Python, or with another extra,
the commands stop before any download and say what to install. Add `--row hrrrv5-full-x1.40962` to any command for
the full-physics row. The example's files (four archives, about 2.4 GB) download once on first use and are checked by
SHA-256.

| command | what it gives |
|---|---|
| `woof hex tangent` | the tangent-linear forecast of a perturbation (default: a 1 K theta bump), written to `tangent.nc` |
| `woof hex adjoint-test` | on a temperature bump (`bump`): the dot test, a finite-difference check, a Taylor series over five bump sizes and a zero-direction check; the whole-state dot test; then the gradient's prediction against the real model's change along further test directions, one more window each (`descent`, the gradient's own steepest descent; `near-ground-moisture`; `smooth-draw`; `--directions` picks them). It checks the linear model a plain `sensitivity` on the same row runs: on the 3 km storm box and its 12 km cut that is the default boundary-layer route (`tape-free`), and `--boundary-layer taped` checks the exact route's held boundary layer instead; the command prints the route it checked (a `boundary layer` line before the windows and a `linear model` line with the verdict) and its receipt names it. PASS needs every check inside its bar, no direction outside the bar and at least two independent directions resolved; a direction the forecast cannot resolve is printed UNRESOLVED and never counts |
| `woof hex sensitivity` | the gradient of a scalar forecast measure (for example a box-mean theta) to the start state, written to `sensitivity.nc`, with a map drawn by the Rust renderer |
| `woof hex param-grad` | the gradient of a measure with respect to five physics constants; `--check` adds the parameter dot test |

**The 120 km example's near-ground maps are an install check, not a forecast sensitivity.** Measured on this example
(3 steps; nudges on model levels 2 to 4 within 900 km of 38.5N 93W; the target is the mean temperature over 30 to
48N, 105 to 80W, levels 1 to 10; scored on the 87 of 348 box columns where the real forecast answers a nudge
steadily; the same with the radiation's random cloud sample held or live): along a low-level east wind nudge of
0.01 to 0.3 m/s the gradient predicts 0.09 of the real model's change; along a low-level temperature nudge 0.77 of
it; along a moisture nudge 1.4 to 1.9 times it. The cause is the one named for the 3 km example: the boundary
layer's mixing is held in the linear model. With it differentiated the wind test read 1.00 (measured); that form is
not in this build. In some 60 to 80 columns of this example the real forecast flips under any nudge, however
small, so no derivative exists there. `sensitivity` prints this caution on the 120 km example.

**What the commands use** on the 120 km example (3 steps, one 96 GB RTX PRO 6000 Blackwell card, fresh install, one
run each). The 120 km example is the quick check that an install works: a mesh this small leaves the card mostly idle,
so its seconds and memory say nothing about a real mesh (for those, the 3 km storm box below).

| row | card memory at peak | host memory at peak | seconds: adjoint-test, tangent, sensitivity, param-grad |
|---|---|---|---|
| `hrrrv5-x1.40962` (default) | 71,156 to 71,282 MiB | 74,860 to 74,923 MiB | 271.9 to 307 (1 step), 158.1, 175.2, 186.1 |
| `hrrrv5-full-x1.40962` | 78,206 to 78,328 MiB | 104,871 to 105,017 MiB | 305.6 (1 step), 186.0, 220.0, 222.0 |

`adjoint-test` runs its own 1-step window with every test direction (above; its card peak 60,734 MiB on the default
row and 76,114 MiB on the full-physics row). On an earlier build, which ran it over the row's 3 steps with one
direction, it read 208.7 s and 253.2 s.

Each of these seconds includes the command's own answer check, and the first command also pays for the download.
The answer check is a whole-state dot test of the adjoint against the tangent-linear model and, on these rows, a
second window proving that the memory left out of that dot is never read: two extra windows, 127 s of the 175.2 s
`sensitivity` above. The check belongs to the recorded forecast and the engine, not to the measure you ask about, so
it runs once per plan: the first command on a plan runs it and keeps its verdict, and a later `tangent`,
`sensitivity` or `param-grad` on the same plan, engine build, kernel set, card family, step count and row prints one
line saying when the check passed and what it read, and does not run it again. Any change to those things runs it
again. `adjoint-test` always runs it, and `--strict` runs it on any command.

**A later command, and the test directions.** Measured on the build before, the four commands in the order above on
one card: `adjoint-test` 219.4 s (the answer check, one test direction, the bump, and the example's files taken
from a local folder), then `tangent` 32.9 s and `sensitivity` 47.3 s, each taking the kept verdict, and
`param-grad --check` 181.7 s, which runs its own check. On the full-physics row `adjoint-test` took 253.3 s and a
`tangent` after it 29.5 s. `adjoint-test` now runs three more test directions after the bump, one more window each, on its own 1-step window:
271.9 to 307 s on the default row and 305.6 s on the full-physics row (an earlier build of 0.6.0), ending PASS
there; with `--steps 3` it took 443 s and 548 s on the two rows and ended NOT PASSED (UNRESOLVED), only the bump
resolved ([the limits](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/differentiable/LIMITS.md)). A FAIL along a direction
is a reading of the gradient along it, not of the install: the receipt's `checks_passed` is the verdict of the
first window's checks and the whole-state dot test alone.

**The words of `sensitivity`.**

- `--objectives FILE` takes every measure in a JSON list (`metric`, `levels`, `box`, `circle`, `name`) in one command,
  each with its own folder, answer and receipt; on a row with a gradient line (the 3 km storm box) every measure runs
  on one loaded program, so the later ones do not pay the program's start again.
- `--checked-forward` compares every forward step of the gradient with the recording (slower: it turns the gradient
  line off).
- `--boundary-layer tape-free|taped|guarded` chooses how the boundary layer's backward runs on the 3 km storm box
  (`guarded` also on the 12 km example row, from 0.6.1).
  `tape-free`, the default there, is faster and from this build differentiates the boundary layer's turbulence
  closure, the plume's heat, moisture and momentum and the returned boundary-layer height, which the exact route
  holds (above). `taped` is the exact route of the held boundary layer, for windows up to about 30 minutes at 3 km (a longer one runs with a
  warning). `guarded` is the exact route with the boundary layer's mixing differentiated under a weak-shear guard
  (above): a named option for a near-ground wind question over a short window, slower, with no forward kept.
- `--keep-forward on|off` on the 3 km storm box's default route: `on`, the default, lets a further target and a
  later attached command take the forward of the gradient before (above); `off` runs every forward and holds
  91,044 MiB less host memory between gradients.
- `--cloud-derivative number|gated|shipped` on the 3 km storm box: `number`, the default from 0.6.0, runs the cloud
  scheme's corrected derivative with rain drop number and cloud droplet number following the forecast (below);
  `gated` runs the corrected derivative with both numbers held; `shipped` runs the derivative of the builds before.
- `--session on|off`, `--session-idle-s N`, `--session-name NAME`, `--session-list`, `--session-stop [NAME]`: the
  program a 3 km `sensitivity` command leaves loaded for the next one (above).
- `--first-gradient recorded|direct`: `recorded`, the default, takes the gradient from a recording of the forecast.
  `direct` is not in this release: it is refused, with what it covers and where it would pay.
- `--strict` runs the answer check on this card even where a verdict was kept.

**Larger meshes: what box to bring and how long a gradient takes.** These are measurements of the engine on one 96 GB
RTX PRO 6000 Blackwell card. The 60 km mesh is not a command of this release; the 3 km storm box is (below).

| mesh (cells) | window | card memory at peak | host memory at peak | one gradient |
|---|---|---|---|---|
| 60 km global (163,842) | 29 steps, 2 h 54 min | 85,080 MiB, with the tape streamed through the host | 161 to 193 GiB | 958 s from a recording already made; 1,163 s for the first gradient of a recording |
| 3 km storm box (38,278) | 59 steps of 15 s, about 15 min | through this command 89,966 MiB and 91,648 MiB on the default route, 89,040 MiB on the exact route (measured, the build before) | through this command (measured, the build before): default route with the forward kept, admitted on 249,661 MiB and peaked at 233,301 MiB; exact route, no forward kept, 158,617 MiB and 151,835 MiB; a first call with its answer check, admitted on 258,057 MiB, peaked at 241,895 MiB | through this command, from the packed recording (measured once, the build before): default route, a first gradient 239.5 s (243.0 s with its answer files and map), a further target in the same command 180.5 s (183.7 s), a later command that attached 181.1 s in the shell. Exact route: 289.6 s first and 260.8 s further. Earlier builds with no forward kept: a further target 195.7 s, a later command that attached 205.3 s. A new forecast is first recorded, about 28 min |

The card figures sit near the card's size because the tape and the working areas are given budgets that fill the
card; they show the job fits a 96 GB card, not what a smaller card would need.

**How far in time the gradient can be trusted.** Measured as the fraction of the real model's change that the
tangent-linear model explains (1 is perfect, 0 is no better than predicting no change), all levels, with 0.7 as the
line:

- 120 km, default row (mountain-drag closure held): about **3 hours**. At 3 hours temperature 0.78 to 0.82, wind 0.90
  to 0.91, moisture 0.76 to 0.79 on three dates; at 6 hours temperature falls to 0.61 to 0.64.
- 120 km, `param-grad` and rows made with `--suite hrrrv5-exact` (mountain-drag closure differentiated): about
  **1 hour**. The full row holds the closure from 0.6.0, as the default row does: held, its tangent explains 0.59
  to 0.70 of the real temperature change and 0.76 to 0.83 of the wind change at 6 hours on the three dates
  (tangent-only runs, 30 steps, one RTX 5090).
- 60 km: about **3 hours**; in one of two places temperature reads 0.63 to 0.65 near 3 hours.
- 3 km storm box: shown to **14 min 45 s** only, for temperature after a low-level warm bump (0.92). Moisture after
  that bump is not predicted at any window tried (0.13 to 0.30). For a 1 K nudge aloft, temperature falls to 0.60 at
  10 minutes (0.85 at half the size). For the mean upward wind above an updraft after a 1 K low-level warm bump, the
  linear answer was a third of the real change, and the real change was not in proportion to the bump (one point, one
  day): moisture and vertical-wind targets at 3 km are an open accuracy item.

Six hours is not claimed on any mesh. The full table, with each row's settings and where it falls off, is "How far
the gradient can be trusted" in [The differentiable engine](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/differentiable/README.md);
its no-model baseline is 0, a forecast of no change.

**A 3 km gradient taken on the 6 km mesh: measured, and not runnable on the example this release ships.** The
3 km storm example below refuses `--gradient-mesh 6km` in one sentence, before any download, and no other shipped
example has a 6 km companion: the route runs only with your own 3 km recording of 2024-05-21 15Z given with `--plan`
(why, at the end of this paragraph). What it is: `woof hex sensitivity --gradient-mesh 6km` on the 3 km storm box
(mesh `q3.96.38278.n40.60w91.30`) takes the gradient on a 6 km cut of the same region (10,014 cells against 38,278)
and carries it back to the 3 km cells. `--gradient-mesh native` takes it on the mesh itself, and so does the default,
`auto`, until the 6 km route has been shown on more targets. The receipt and `sensitivity.nc` say which mesh the
gradient was made on and that its answer check was made on the 6 km operator. Where it was shown: one storm (the HRRR
start of 2024-05-21 15Z over eastern Iowa), one target (the mean modified potential temperature over the 150 km circle
at 40.6N 91.3W, every level), one 885 s window, one 96 GB RTX PRO 6000 Blackwell card. There the 6 km gradient took 95
to 98 s against 499 to 502 s for the 3 km one (an earlier build's times; on 0.6.0's final wheel the 3 km gradient takes
180 to 232 s on such a card, and the 6 km one was not timed again), pointed the same way (cosine above 0.99 for temperature and moisture),
and the real 3 km cost fell 0.995 as much along it as along the 3 km gradient. Only cell fields are carried, not the
edge winds. A target, day or window outside that is refused by name. The 6 km plan that ships is of that
2024-05-21 15Z start, so on the storm example below (a forecast from 2023-03-31 18Z) the command refuses the 6 km
route in one sentence: its gradient would be of another forecast. Give `--plan` with a 3 km recording of
2024-05-21 15Z to use it. On the storm example's own forecast the 6 km route was measured and is not offered: for
the mean modified potential temperature of the lowest ten levels over the same circle the carried 6 km gradient read
cosine 0.992 for temperature and 0.986 for moisture against the 3 km one, and for the mean water vapour on every
level 0.80 and 0.96, where the day the route is offered on read above 0.99.

More on the 3 km storm example (the row `hrrrv5-q3.96.38278.n40.60w91.30`, near the top of this section). Its plan is
one hour of the HRRR 2023-03-31 18Z start: 3 steps by default, up to 239. `--init example` records a plan of its own
from that start (not yet run through to a gradient on a card in this release). A 59-step `sensitivity` keeps its checkpoints in host memory; `tangent`, `adjoint-test`,
`param-grad` and `--checked-forward` keep every step and write what does not fit to the output folder, and the
command states that disk before it runs and stops if the folder has less. Of its commands `sensitivity` and the 59-step `adjoint-test` have run from a plain install on a card; `tangent` and `param-grad` have not been run on this row that way. On this row `adjoint-test` runs two test directions by default, `bump` and
`descent`; `--directions all` runs the four, one more window of the box each.

Limits, in short. The 120 km example covers 1 to 3 steps, the 12 km example 1 to 15 and the 3 km storm box 1 to 239;
any other count is refused before anything downloads, with the counts the example covers. A plan recorded with `--init` has not yet passed
the strict check, and a plan replays only on the card family it was recorded on. What each row differentiates and
holds, how to read the output, and the rest of the limits:
[The differentiable engine](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/differentiable/README.md).

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

### Making a mesh

The generator ships in this wheel (`rw_mpas_mesh`, geometry 2), and `woof hex mesh-plan` makes a
grid, its static and the rows file the forecast doors read, in one command. A request that names no
geometry is built under geometry 2: every spacing as asked, one growth rate per cell from each fine
zone's outline, no obtuse triangle and no cell of four edges; `--geometry 1` builds the halving
ladder and the meshes of 0.6.1, byte for byte. The 3 km CONUS globe:

```sh
woof hex mesh-plan --global-mesh --point 38.5,-96.5 --fine-dx-m 3000 --radius-km 2100 \
    --background-km 96 --card rtx-pro-6000 --cards 4 --generate --out-dir conus3km
```

From a fresh install with an empty home on 20 Nov 2026 it wrote 1,959,938 cells, grid and static,
in 109.5 s on an otherwise idle 24-core host (the generator 33.7 s, the static 68.2 s), with the
smallest edge 0.79 of 3 km and no dual-triangle angle over 87.0 degrees. A region, or a zone drawn
to any outline, is a mesh row (`--spec ROW.json`; `regional --mesh-spec ROW.json` runs a forecast on
one): [mesh rows](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/mesh-rows.md)
has the words and more measured builds, and the User Manual's chapter 4 the rest.

### A fine core at any point, from HRRR

`woof hex mesh-plan --point LAT,LON --fine-dx-m 937.5 --radius-km 100 --card 32gb`
writes the generator's spec for a 937.5 m core centred on the point (geometry 2 unless
`--geometry 1` asks for the registered 937.5 m ladder), prices the
global parent through the generator and the limited-area cull on the
admission surface's own row for the card, and with `--generate` builds the
pair, admits it, registers it as a RUNTIME ROW in `mesh-rows.json` beside the
mesh (never a checkout edit), cuts the cull and mints its vertical.
`woof hex intermediate --source hrrr` resamples HRRR onto the regular
lat-lon WPS intermediate the init and boundary engines read, through the
engine's own decoder and interpolation operator; `woof hex lbc` builds the
boundary series from the hourly intermediates. The whole chain, with what it
measured, is [`docs/hex-point-hrrr.md`](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/hex-point-hrrr.md).

### Placement and cycling

`woof hex swath` decides where a fine grid should go: detection on
sea-level-reduced pressure, a declarative threat grammar, commensurable
ranking across phenomena that carry different units, and hysteresis so a grid
does not chase noise. `woof hex cycle` runs the loop: a coarse parent, the
corridors placed inside it, and the next cycle's re-detection. See
[`docs/cycle-door.md`](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/cycle-door.md).

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
  inert: the run is bit-identical to the option-off run on the same schedule.
  `--local-timestep` selects RK3 with six acoustic sub-steps and no FCT limiter,
  where the route's default is RK2 with four sub-steps and the limiter on, so the
  run to compare with is `--time-integration-order 3 --acoustic-substeps 6
  --no-monotonic`. Measured on 0.6.2 (x1.40962, WOOF-ArWen route, one hour, one RTX
  4090): one class, zero interface edges, identity permutation, and every history
  frame SHA-256-identical.
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
history frames are byte-identical to the option-off run on the same schedule, at 0.992x the step
(inside run-to-run noise). Forcing an interface into the interior with the
`tools/lts_forced_classing.py` instrument (6,206 cells at rate 3, 1,456
interface edges, an 8.5 % arithmetic acoustic saving) makes the step 1.071x
SLOWER: every dycore kernel on a mesh this size runs in one wave on the card,
so the fine class costs what the whole mesh costs and each coarse-class
launch is a further wave. The corridor cull (40,520 cells, GFS-driven) gives
the same numbers (0.989x, 1.068x). What the forced interface does to the
fields is in [`docs/local-timestep-lam.md`](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/local-timestep-lam.md).

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

### Stochastic physics (SPPT, SKEB)

hex carries its own SPPT and SKEB, written from the published methods. SPPT
follows Buizza, Miller & Palmer (1999) and Palmer et al. (2009); SKEB follows
Shutts (2005) and Berner et al. (2009). They run on the GPU on every route
and on every mesh: global, zoom globe, limited area, RRFS included.
`--sppt` and `--skeb` on `forecast` and `global --members` turn them on.
They are on by default in every hex ensemble of more than one member, and
`--no-sppt` or `--no-skeb` turns them off. On the RRFS suite route, `--sppt`
takes NOAA's RRFS v2 ensemble settings (one scale: 0.8, 450 km, 6 h).
[Stochastic physics](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/stochastic-physics.md) has the method, the
controls and the measured pattern statistics.

Measured on 2026-10-29:

- **Pattern statistics** (450 km, 6 h): correlation falls to `exp(-1/2)`
  at 450.5 km (Gaussian fit 451.0 km). Point variance is 1.001 and the lag
  time is 5.996 h.
- **Spread and skill** (10 GEFS members, 120 km, 2026-10-27 00Z, +6 h
  against the GFS analysis):

  | | z500 spread | z500 mean RMSE | t2m spread | t2m mean RMSE |
  | --- | --- | --- | --- | --- |
  | SPPT and SKEB off | 3.21 m | 4.065 m | 0.44 K | 1.698 K |
  | SPPT and SKEB on | 3.50 m | 4.078 m | 0.51 K | 1.692 K |

- **Cost:** under 1 % of the step.
- **Restart:** a run restarted from a 3 h checkpoint is identical to the
  straight run in every value.
- **Both off:** forecasts are identical, word for word, to those of the
  release before this change.

### Running an MPAS physics suite

**MPAS physics is the default (0.4.6).** Every door (`forecast`, `global`,
`mesh-plan --run`, `regional`, `country`, `moving`, ensembles, `cycle`, Hex
with Regional parents, backgrounds) runs MPAS-A's own physics on the
MPAS route when the run names no physics. The suite follows MPAS-A practice by
mesh (`src/hexcore/data/profiles/physics-default.v1.json`):

| Mesh | Suite | Why |
|---|---|---|
| finest cells under 10 km (regions, zoom globes, storm-following, any variable-resolution mesh) | `convection_permitting` | MPAS-A's recommended suite where convection-permitting cells are used, including variable-resolution meshes |
| quasi-uniform, 10 km and coarser | `convection_permitting` | MPAS-A tested it from several hundred km down to 3 km (scale-aware Grell-Freitas); `mesoscale_reference`, MPAS's Registry default there, does not run on the GPU yet (refused by name) |

ArWen physics stays one flag away: `--physics-suite arwen_mpas_column` runs
exactly what a bare run ran before 0.4.6 (on `global`, `--physics-row arwen`).
A `--physics KEY=VALUE` override, an explicit `--convection` scheme or a
`--physics-backend` row with no suite also selects the WOOF-ArWen route.
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
  [MPAS physics resources](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/physics-resources.md).
- **The mode** is correct unless `--physics-mode exact` is given (see
  *Correct mode* above).
- **Radiation's cloud sample** (`config_radt_mcica_seed`, from 0.6.2). RRTMG
  draws each column's sub-column clouds (McICA) from a random stream. The
  default, `pressure`, seeds it as MPAS does, from the fractional digits of
  the column's four lowest layer pressures, so any change of the state, down
  to one bit of one pressure, draws that column's clouds again at the next
  radiation call. `cell` seeds it from the cell's global index and the
  radiation call's time step (base seed 150), so two runs from nearby starts
  draw the same sub-columns, on any number of ranks. It is for derivative
  checks and twin experiments, where a small change
  of the start must not draw new clouds. It is not MPAS's rule, so a run that
  names it is not bit for bit with MPAS; with the word left out a forecast is
  the one it was before the option existed. Set it in the namelist's
  `&physics` group or with `--physics config_radt_mcica_seed=cell`.

`--preflight` checks routing and metadata without integrating.
`woof hex verify-mpas --deck DECK --receipt R.json --work-root DIR --ratchet RATCHET`
runs a deck and compares every stream value hex writes against native MPAS.
The published decks are in the sdist under `verification/gates/` (see its
README for the data layout). The headline deck runs every active physics
slot on the GPU, so it needs no native MPAS library, no MPAS-Model tree and no
compiler. Running it from a pip install:

```sh
python3 -m venv hexgpu && . hexgpu/bin/activate   # Debian and Ubuntu: sudo apt install python3-venv first
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
(`strict_environment`). Identical values are GREEN in any environment, and the
receipt lists how the environment differs from the reference. Differing values
are a source regression only in the reference environment; elsewhere the
verdict is ENVIRONMENT, with the raw counts kept.

---

## Limitations

Read these before trusting a number.

### On the WOOF-ArWen route, Grell-Freitas is a different generation from native

**This is about ArWen physics** (`--physics-suite arwen_mpas_column`). The MPAS suites, the default, run MPAS's own
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
file written. It is a declared property of the WOOF-ArWen route, judged against
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

### Three bias-shaped differences between the WOOF-ArWen route and native MPAS (declared divergences)

Not blockers, and not evidence of a broken dycore. They are the measured price
of running WOOF-ArWen's physics instead of MPAS's on the WOOF-ArWen route, stated so a
user knows what they are getting; the MPAS suites do not carry them. The
obs-skill comparison, which is the verification of record for WOOF-ArWen's physics,
ran on 2026-08-25 and reaches two of the three; the register of
all three (the mechanism, the magnitude, the named observational referee and
now what that referee said) is
[`docs/declared-divergences.md`](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/declared-divergences.md).

Measured over 24 h, two independent weather cases, two mixing regimes,
163,842-cell mesh, cell-aligned with no interpolation.

**When, exactly, because these three magnitudes carry no date of their own.**
They entered this repository already finished on 2026-08-20, and there is no
receipt directory, no card and no run commit for them anywhere in the tree.
What can be established is a ceiling: the commit that introduced them pinned
engine `629ddb6f0`, so they were measured at or before that pin. Ten engine
pin moves have landed since: `0d04db712` (2026-08-24), `26daaab7e`
(2026-08-25), `659962929` (2026-08-28), `7e34a48` (2026-08-31, woof
`2.6.0`), `df5f34c5c` (2026-09-01, gpuwm 2.6.1), `636ab1b4b`
(2026-09-02, gpuwm 2.6.3), `d60b883e7` (2026-09-02, gpuwm 2.6.4),
`a7785421e` (2026-09-13, gpuwm 2.7.3), `ed957997f` (2026-09-15, woof
2.7.4) and `0164ae0f2` (2026-09-29, gpuwm 2.8.0). The **2.5.8** move is measured to change nothing: a four-arm byte A/B
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
glibc float32 values moved into a header the loader prepends), the kernel
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

   **That number is against native MPAS, which is not the WOOF-ArWen route's
   referee, and the observations returned the opposite sign.** "15 % dry" is a
   global domain mean against native MPAS-A. The verification of record for
   WOOF-ArWen's physics is skill against observations; it ran on 2026-08-25
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
case-independent, which chaos cannot be, and each names a part of the port to fix.

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
tables. See [the point/HRRR guide](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/hex-point-hrrr.md).

Measured 2026-11-20 on a 0.6.2 build, one RTX PRO 6000 Blackwell Server Edition
on a shared 8-thread host: the command above, as printed, built a 57,940-cell,
55-level mesh (geometry 2, the default) and ran a 3 h forecast of 2,160 steps at
5 s with four history frames. The whole command took 487 s: mesh 102 s,
intermediate 14 s, init 11 s, boundaries 6 s, contract 38 s, forecast 313 s and
products 3 s (13 maps of the refined core, 7 skipped, none failed). An earlier
build's run of it found F000 surface winds unavailable, so point wind diagnostics
show gaps or n/a there; later frames have valid winds (not rechecked on 0.6.2).

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

- **Ranks on several hosts** are a door from 0.6.2 (`woof hex forecast --hosts
  hostA:0,hostB:0`, manual chapter 12); every host needs NCCL (`pip install
  nvidia-nccl-cu12`). With the 120 km globe, one rank on an RTX 4090 host and one
  on an RTX 5070 Ti host on a home network, strict numerics wrote the words of two
  ranks on one RTX PRO 6000 (0 of 706,509,363 different), and the resume from each
  host's own 3 h seal those of the run that never stopped. Fast numerics, the
  default, depend on the card family. No speed is claimed for several hosts.
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

WOOF-ArWen's RTE+RRTMGP radiation, which the global rows' Morrison suite runs
(as does any WOOF-route run that selects it), reads a trace-gas and ozone
table that woof's data package ships under CC-BY-SA-4.0. woof hex does not
carry it. It is data: the licence permits sale, and its share-alike reaches
only adaptations of that table (`NOTICE` section 10).

The wheel's compiled helpers and the libraries they link (netCDF-C in the
template copier, the pool reader and the stream bridge, which also links HDF5
and zlib; the Rust crates in the observation scorer, the pool reader and the
engine bridges) are listed in `NOTICE` sections 12 to 14. The texts of the Rust crates the
scorer and the bridges link are in
`licenses/THIRD-PARTY-LICENSES-helper-binaries.txt`.

**US Government code:** the CUDA ports of NOAA GSL's UGWP gravity-wave
drag (`bl_ugwp_gwdo`, `ugwp_ngw`) ship in the distribution, with their tests
and oracles, and the RRFS stage runs the fork's NOAA GSL physics (MYNN-EDMF,
MYNN-SFC, RUC, GFL, smoke) with no opt-in. Their upstreams carry no licence
file; the project has treated them as works of the US Government (NOAA)
since 2026-10-14. `NOTICE` sections 8 and 9 give their pins and
attribution.

**Withheld from the public distribution:** the UCAR Noah-MP v5.0 port, with
its tests and native oracles. `NOTICE` section 4 retains its upstream pin,
attribution and the reason for the hold. The public wheel contains no
Fortran runtime. A withheld selection is refused by name; an opt-in does not
restore code absent from the wheel.

The Noah-MP that hex runs without that opt-in is the WOOF-ArWen route's WRF-lineage
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
[`tools/battery/README.md`](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/tools/battery/README.md) for what each covers and
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
ensemble workflow. See [the S2S door](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/s2s/README.md) for lagged starts,
checkpoint recovery, products, baseline interpretation and measured costs.
Use `--dry-run` to inspect the plan.

Native forecast stores: [Icechunk output](https://github.com/recastsystems/gpuwm-hex/blob/v0.6.2/tree/docs/icechunk.md).
