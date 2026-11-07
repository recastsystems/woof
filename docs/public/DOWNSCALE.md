# Offline downscaling (`woof downscale`)

`woof downscale` re-runs a finer child grid from an **archived** parent
run -- the CUDA-native equivalent of WRF's offline `ndown` workflow. It
accepts woof and stock-WRF parents, proves the parent series before
touching the GPU, and advances only the child.

The downscale command itself is one invocation and it is not where people
stall. What stalls people is producing a parent archive the command will
accept, and **that part differs by source**. This page walks one source
end to end with every command written out, names what the other routes
change, and puts each unfixed rough edge at the step where it bites.

## What the parent archive must contain

Two preconditions, both checked at the front door before any GPU work.

**1. A woof restart file beside the history frames.** The restart is the
physics evidence `--parent-restart` binds -- the parent's microphysics
identity is never inferred from a variable inventory. Without it:

```
woof downscale: parent physics must be bound from companion evidence: pass --parent-restart (woof) or --parent-namelist (stock WRF)
```

This refusal fires on the real run **and** on `--dry-run`. Check for the
files before you plan anything else:

```bash
ls RUN/gpuwmrst_d0*.npz
```

The real filename pattern is `gpuwmrst_d0N_YYYY-MM-DD_HH_MM_SS.npz`, for
example `gpuwmrst_d01_2024-05-06_01_00_00.npz`. A stock-WRF parent uses
`--parent-namelist namelist.input` instead.

Choose the restart from the selected history domain. The front door checks
its domain identity, dimensions and grid spacing against the archive before
deriving child physics or time step. A restart from another nest is refused.
The child duration must also fit entirely inside the archived forcing window;
both `--point --hours` and a supplied child config are checked on `--dry-run`.

The `--point` route picks the child's time step for you. It is the parent's
step over the ratio when the run length, history, checkpoint and health
intervals and the parent's frame interval are all whole numbers of it, and
otherwise the largest shorter step they all are (a 12 km parent on 64.44 s at
ratio 3 gives 20 s instead of 21.48 s, and the plan says so). Any `--hours`
works. A child config you write yourself keeps the `dt` it names, and the
review refuses one that the run length or an interval is not a whole number of.

**2. For a full-physics child, nothing extra.** A child builds its own
terrain, land use and soil at its own spacing (next section) and puts the
parent's land-surface state on them. The rest of this item describes the
older route a child takes with `--parent-terrain`, where the land identity
comes from the parent. On that route, with no `--child-surface-from`,
`woof downscale` takes land identity and the soil warm start off the
parent's own history frame and puts them on the child grid through WRF's
nest-birth operators, the same route WRF uses for a nest with
`input_from_file = .false.`. It says so, once:

```
woof downscale: child surface derived from RUN/wrfout_d01_... (21 fields, MODIFIED_IGBP_MODIS_NOAH)
warning: child surface state interpolated from the parent's own history rather than built on the child grid -- the child's land-use, soil category and landmask are the PARENT's ...
```

That warning is the cost, and it is real: the child's coastlines, lakes
and islands are its parent's, not the ones its own spacing could resolve.
`--child-surface-from` is the higher-fidelity route and takes a
`wrfinput` or history file whose grid **equals** the child's. The grid
check on that flag is exact:

```
woof downscale: RUN\wrfout_d01_2024-05-06_00_00_00 south_north=48 does not match the child grid south_north=96; the surface source must be on the EXACT child grid
```

Where such a file comes from depends on your route -- see
[Route 2](#route-2-the-prepared-sources), which writes one per nest.

Matching dimensions alone do not establish the grid. An explicit surface
must include `XLAT` and `XLONG` matching the selected child placement. The
front door compares them with native parent-to-child mass coordinates,
allowing float32/projection differences up to one percent of a child cell
(at least 2 m), and checks `DX`/`DY` when present. A same-shaped surface from
another location is refused before preprocessing. The plan records the
measured coordinate separation and tolerance.

## The child's own terrain

A child runs on its **own** static geography by default: terrain, land use
and soil category built at the child's spacing by the same static builder a
standalone run of that grid uses, with Copernicus GLO-30 terrain at 1 km or
finer (the engine's high-resolution default). A 1 km child of a 3 km parent
therefore sees the passes, canyons and ridges its spacing resolves, not its
parent's 3 km ridges interpolated. The parent's atmosphere is moved onto that
terrain the way WRF's `ndown` moves it (WRF v4.7.1 `main/ndown_em.F`):

1. Every parent frame is interpolated onto the child grid and, with
   `--child-levels`, remapped onto the child's own levels.
2. The parent's interpolated terrain and the child's own are blended across
   the child's edges (WRF's `blend_terrain`): the specified and relaxation
   rows keep the parent's terrain, so the boundary data are applied on the
   terrain they were made on; the next 5 rows blend; the interior is the
   child's own.
3. The state is rebalanced onto the blended terrain (WRF's `rebalance`): the
   base state is rebuilt on it, each level's potential temperature keeps its
   departure from the reference atmosphere, the column's dry-mass
   perturbation is kept, and pressure and geopotential are integrated
   hydrostatically down and up the new column.

The initial state and every boundary frame take the same steps. The land
surface is the child's own land use and soil, with the parent's land state
(skin, soil temperature and moisture, snow) put on it by WRF's masked
interpolator and moved to the child's terrain with `real.exe`'s soil lapse
(-6.5 K per km). `--child-surface-from` still wins for the land surface when
it is given.

Both steps match WRF v4.7.1's own Fortran on column checks
(`tools/ndown_wrf471_oracle`, `tests/test_offline_child_geography.py`).

The plan and the run say which terrain the child is on:

```
woof downscale: child terrain: its own static geography at 1000 m, built from WPS_GEOG with copernicus-dem-glo30 terrain, blended into the parent's across the boundary zone (--parent-terrain keeps the parent's)
```

and the derived child config carries it in a `[static]` table:

```toml
[static]
terrain = "own"            # or "parent"
# geog_root = "/data/WPS_GEOG"     the WPS_GEOG tree (default: the staged one)
# geog_data_res = "default"        WPS geog_data_res tokens
# smooth_option = "1-2-1"          terrain smoothing, as [[domain]] static takes it
# smooth_passes = 1
# [static.highres]                 high-resolution terrain or land cover, as an experiment takes it
```

Flags win over the file: `--parent-terrain` runs the child on its parent's
interpolated terrain, land use and soil (the route every child ran before
children had their own geography), and `--geog-root DIR` names the WPS_GEOG
tree. A child whose own geography cannot be built (no WPS_GEOG tree, or a
parent whose stored coordinates contradict its projection) is refused before
anything is reserved, naming `--parent-terrain`; `--dry-run` prints the same
reason and continues so the plan can be read.

A downscaled run is itself a parent, and its grandchild builds its own
terrain the same way.

## Which route builds your parent -- ask the registry

Do not guess. Every source declares its route:

```bash
woof sources              # all 32 rows, one line each
woof sources era5         # one row in full
woof sources --json       # the same facts as JSON
```

The field that decides your chain is `run_plan.intent_chain`. It takes
three values on this release:

| `intent_chain` | Sources | Parent is built by |
|---|---|---|
| `experiment` | `era5` | `woof run CONFIG.toml` -- config-driven, no chain |
| `prepared:go` / `prepared:hrrr` | `gfs`, `hrrr` | `woof go CONFIG.toml` end to end |
| `prepared:staged` | `hrrr-prs`, `gem-gdps`, `icon-eu`, `aigfs`, `ecmwf-open-data`, `aifs`, `rap`, `rrfs` | staged chain: authority, manifest, `rw-wps`, tree runner |

These are different enough that a walkthrough for one does not transfer.
Route 1 below is `era5`, written out completely. Route 2 covers the
`prepared:*` sources.

`mapped` is not a fourth route to a parent: it is the generic declarative
adapter for bytes you already have, it has no fetch route, and
`wizard_planable` is `no`, so `woof domain` cannot plan it.

---

# Route 1: ERA5, from nothing to a downscaled child

Every command below was run in this order. Values are literal.

### 0. Find the console scripts

On an editable or `--user` install the console scripts land in your
**user** Scripts directory, which is often not on `PATH`:

```
C:\Users\<you>\AppData\Roaming\Python\Python313\Scripts\
```

`gpuwm.exe`, `rw-wps.exe` and `woof-prepared-tree-forecast.exe` are
there. If `woof` is not found, prepend that directory before continuing.

### 1. Emit the config FIRST, before fetching anything

The wizard computes the geographic box you must request. Fetching first
means fetching the wrong box.

```bash
woof domain --point 39.0,-98.0 --source era5 --cycle 2024-05-06T00 \
  --hours 12 --history-interval 900 --vram-gib 6 \
  --geog-root C:\WPS_GEOG --name demo \
  --out demo.toml --data-dir era5-raw
```

2.6 s. Writes `demo.toml`, `demo.namelist.wps` and `Vtable.ERA5_CDO`, and
emits `nx = 60, ny = 48, dx = 12000.0`.

**Pass `--geog-root` explicitly.** Without it the wizard writes
`geog_root = "${GPUWM_CASE_DATA_ROOT}/WPS_GEOG"` into the config, which
resolves to a path that may not hold your staged static data. This is a
**workaround**, not a fix: the emitted default is not validated at emission
time. `woof fetch-geog` stages the nine datasets (~16 GiB unpacked) if you
do not have them.

**Sizing on a 10 GiB card.** The grid-independent envelope is about
3.24 GiB (CUDA context plus the kernel local-memory backing store), so
small budget changes move the grid a long way. Measured on an RTX 3080
10 GiB, Windows/WDDM:

| flag | fitted grid |
|---|---|
| `--vram-gib 10` | 406 x 326 |
| `--vram-gib 7` | 84 x 68 plus a nest |
| `--vram-gib 6` | 60 x 48 |
| `--vram-gib 4` | refused, naming the breakage |

`--card` offers only `{12gb,16gb,24gb,32gb}`, so a 10 GiB card has no tier
and must use `--vram-gib`. A WDDM desktop idles 2.7-4.6 GiB, so the card is
never empty; `--vram-gib 6` is a realistic starting budget on a 10 GiB
card with a desktop running. For grids larger than the card can hold, see
[TILES.md](TILES.md).

### 2. Check that the parent will write checkpoints

Step 1 already wrote the checkpoint cadence into `demo.toml`:

```toml
restart_interval_s = 3600.0
```

Every configuration `woof domain` writes, single domain or nest ladder,
checkpoints hourly, or once at the end of a run shorter than an hour.
`woof run` honours it and writes `gpuwmrst_d01_*.npz` beside the wrfouts,
which is the file precondition 1 above asks for. There is nothing to edit.

A configuration written by hand, or by an older release, may carry
`restart_interval_s = 0.0`. That run writes no checkpoint and cannot be a
parent, so set a positive interval, such as `3600.0`, before running it.

### 3. Request the bytes

```bash
woof fetch --source era5 --cycle 2024-05-06T00 --hours 12 \
  --area 34.30,-104.39,43.63,-91.61 --out era5-raw
```

Use the `--area` the wizard printed in step 1, **widened**. This command
downloads nothing -- ERA5 acquisition is manual because the Copernicus CDS
API needs a personal account key that woof will not embed. It writes:

```
fetch era5: wrote era5-raw\era5-cds-request.json
fetch era5: wrote era5-raw\era5-cds-retrieve.py (runs the retrieval)
```

**Ask for more area than the wizard prints.** CDS snaps the requested box
inward to its grid. Requesting `34.30,-104.39,43.63,-91.61` delivered
`lat [34.5, 43.5] lon [-104.25, -91.75]` -- smaller on all four sides. The
preflight catches it later, but only after the download:

```
[spatial-coverage] domain point lat/lon=(34.0024, -105.184) lies outside forcing lat [34.5, 43.5] lon [-104.25, -91.75]
```

### 4. Retrieve, with the CDS client

`era5-cds-retrieve.py`, written beside the request, runs both requests and
concatenates their output. It resolves every path from its own location, so
the working directory does not matter and the files land in `--out`:

```bash
pip install cdsapi
python era5-raw/era5-cds-retrieve.py
```

The command is printed by step 3, with your absolute path filled in.

The client reads your key from `~/.cdsapirc` in the home directory of
**whichever interpreter runs it**. If you retrieve from WSL on a Windows
box, that is WSL's home, not `C:\Users\<you>`. `woof sources era5` reports
the Windows path and whether a key is present there, so on a WSL retrieval
its "no key" line can be correct about Windows and irrelevant to the run.
The fetch step prints the WSL form too, with the path already translated to
`/mnt/c/...`:

```bash
wsl sh -c "python3 -u /mnt/c/wx/era5-demo/era5-raw/era5-cds-retrieve.py"
```

Run it in the foreground. `nohup ... &` inside `wsl sh -c` does not
survive: the job dies when `wsl.exe` returns and no log is written.

Measured on this walk: 95 s for a 2-time, 9x9-point request; both parts plus
`era5-combined.grib` written into `--out`, nothing in the shell's directory.

### 5. Validate

```bash
woof fetch --source era5 --validate era5-combined.grib --area 34.30,-104.39,43.63,-91.61
```

The retrieval prints this command with your own `--area` already in it.
`era5 validation: PASS` lists the GRIB1 envelope count, the valid times, the
pressure ladder, the surface inventory, **and the delivered grid**:

```
ok: grid 9x9, 0.25 x 0.25 deg, lat [39.00, 41.00] lon [-99.00, -97.00]
ok: the delivered grid covers the requested box lat [39.00, 41.00] lon [-99.00, -97.00]
```

With `--area` given, a delivered box that falls more than one grid cell
short of the requested one on any edge is a FAIL, naming the edge and the
shortfall -- so the wrong-box file is caught here instead of at `woof
check` or mid-run. One cell of tolerance is the CDS's own inward snap
(step 3). Without `--area` the extent is still reported, followed by a line
saying it was checked against nothing.

### 7. Check, then run the parent

Order matters: `woof check` refuses before the bytes exist, so it comes
after retrieval, not before.

```bash
woof check demo.toml
woof run demo.toml --outdir demo-run
```

If the config is missing its bytes you get, at rc 0:

```
woof check: forcing file ...\era5-combined.grib declared in [case_data] of ...\demo.toml does not exist.
```

`woof run` is silent through preparation -- a banner, then roughly 40 s
with no output before anything else appears. That is expected.

Measured, RTX 3080 10 GiB, 12 h forecast at 60 x 48 x 49 and 12 km,
including ERA5 preprocessing: **51 s wall**, `status: complete`, 49
`wrfout_d01_*` frames at 900 s, 12 `gpuwmrst_d01_*.npz`. Machine-wide peak
4800 MiB VRAM, 1965 MiB host RSS.

Do not run `woof go` here. It refuses ERA5 configs by name and points at
the right command:

```
woof go: demo.toml declares a [case_data] table, which is the ERA5 config-driven route -- `woof run` executes that one directly and needs no chain.
  remedy: woof check demo.toml && woof run demo.toml
```

That is the POSIX spelling. In Windows PowerShell the remedy prints as
`woof check demo.toml; if ($?) { woof run demo.toml }`.

### 8. Derive and run the child

Which domain to downscale from: `woof downscale-parent demo-run` prints, as
JSON, every domain the run wrote with its grid spacing and frame count
(`domains: [{id, dx_m, frames}]`) and names the finest as `default_parent`.
`woof downscale` on a run with several domains still needs
`--parent-domain`; its refusal lists the domains with their spacing.

```bash
woof downscale demo-run --parent-domain 1 \
  --parent-restart demo-run/gpuwmrst_d01_2024-05-06_01_00_00.npz \
  --point 39.0,-98.0 --ratio 3 --child-size 120,96 \
  --vram-gib 6 --hours 2 --out child-run --dry-run
```

1.4 s, rc 0. It prints the placement and writes the derived TOML. A REAL
run writes it to `child-run/child.toml`, inside the run it describes. A
`--dry-run` cannot: the run claims `--out` for itself, so a dry run
writes `child-run.child.toml` beside it and says so.

**Re-running with the same `--out` is safe.** A run that refuses hands
the directory it claimed back, so the corrected command meets the tree
the first attempt found. An `--out` that already holds a run's output is
refused by name, saying what it holds and that you may pass a new `--out`
or remove the old directory; nothing in it is ever overwritten or merged
into, because the `report.json` a run publishes has to describe one run.

**Two downscales never share one `--out`.** The run that claims `--out`
holds `.gpuwm-output.owner` inside it until the run ends, whether it
created the folder or found it empty, and a second `woof downscale`
aimed at that folder meanwhile is refused with the process that holds
it. A refused run gives back only what it still owns: a folder it
created is removed, an empty folder it adopted is left empty, and a
folder another run has taken over is left alone.

**Pass `--vram-gib` or `--card` here too.** `woof downscale --card`
defaults to `24gb` while `woof domain` measures the local card, so a
10 GiB owner who omits it gets a child sized for a card they do not have.

**You do not need to check the derived placement by hand.** For a 120 x 96
child at ratio 3 inside a 60 x 48 parent, centred, the tool printed
`{"ratio": 3, "i_parent_start": 11, "j_parent_start": 9}`, matching the
arithmetic exactly. `--dry-run` is worth running to read the plan; it is
not a required rehearsal, because the real run refuses at the front door in
about a second, before the GPU, with the same remedy text.

Then drop `--dry-run` to run it. Measured on the RTX 3080: `contract_pass`,
`child_shape [49, 96, 120]` at 4000 m, 49 boundary frames, first frame
`wrfout_d02_2024-05-06_00_00_00` at 44,803,168 B. Machine-wide peak
8859 MiB VRAM, 3074 MiB host RSS -- close enough to a 10 GiB card's ceiling
that a loaded desktop matters.

#### A full-physics ERA5 child needs no extra file

This used to be the dead end of the whole page: `--child-surface-from` was
mandatory for a full-physics child, and **no ERA5 route produced a file on
an arbitrary child grid**. `woof run` -- the route ERA5 is steered onto --
writes no `wrf-native-input/`; the `rw-wps` prepared-tree route that does
write one needs a front-door manifest `woof fetch` authors for `gfs` only;
and building an ERA5 nest at the child geometry to make one hit its own
refusal in `initialize_child`. The refusal demanded a file the product
could not make, and said so only after the parent forecast was paid for.

It is closed. The parent's own history already carries the nine surface
fields and the landuse identity attributes -- any run with a land-surface
scheme publishes them -- and the child grid is an exact refinement of a
parent window, so `woof downscale` puts them where the child needs them
itself. Measured on test host 1 (RTX 5070 Ti), the 12 km ERA5 parent from
`configs/era5_demo.toml` downscaled to a 72 x 54 child at 4 km with
`sf_surface_physics = 2`, `sf_sfclay_physics = 91`, `bl_pbl_physics = 1`
and both radiation streams:

```
woof downscale RUN --parent-domain 1 --parent-restart RUN/gpuwmrst_d01_... \
  --point 35.3,-97.5 --ratio 3 --child-size 72,54 --hours 1 --out CHILD
```

```
woof downscale: child surface derived from RUN/wrfout_d01_... (21 fields, MODIFIED_IGBP_MODIS_NOAH)
warning: child surface state interpolated from the parent's own history rather than built on the child grid ...
```

`"result": "PASS"`, 180 steps, `nan: false`, two `wrfout_d02_*` frames and
a final restart, 2.8 s wall. `report.json` carries the whole derivation
under `child_surface_source`: which fields were donor-copied, the masked
interpolator's branch counts per field, the parent frame's sha256.

**What you give up.** The child's land-use, soil category and landmask are
its parent's, carried down from the parent cell each child column sits in,
so coastlines, lakes and islands the child's 4 km spacing could resolve are
not resolved. That is WRF's own answer for a nest with
`input_from_file = .false.`, and `--child-surface-from` remains the
higher-fidelity route when you have a child-grid file (see
[Route 2](#route-2-the-prepared-sources), which writes one per nest).

A **microphysics-only child** is still available and still not a
substitute: zeroing `sf_surface_physics` alone is not enough, the check
reads `sf_surface_physics`, `sf_sfclay_physics` and `bl_pbl_physics`
together. In the 2.5.1 walk, with all three zeroed, the contract passed and
the run failed mid-integration with
`FloatingPointError: microphysics RAINNC contains a non-finite value`.
Prefer the full-physics child.

---

# Route 2: the prepared sources

For `intent_chain` values `prepared:go`, `prepared:hrrr` and
`prepared:staged`, the parent is built by the prepared pipeline rather than
by `woof run`, and the downscale half of the page is unchanged.

Emit a config the same way, naming your source and a nest ladder:

```bash
woof domain --point 39.7,-84.0 --card 12gb --ladder 12-3 \
  --source gfs --cycle latest --hours 6 --history-interval 900 \
  --geog-root C:\WPS_GEOG --out tree.toml --data-dir gfs-raw --explain
```

This config checkpoints hourly too, as step 2 of Route 1 describes. With
`--explain` the wizard closes by printing
the fetch line with your area, cycle and model top filled in, then the
check, then the one `woof go` line that runs everything after the
fetch. Without `--explain` it prints only that `woof go` line. Run by
hand, the prepared chain is four commands, not one, in this order:

```bash
# a. fetch the bytes, using the --area, --cycle and --p-top-pa the
#    wizard printed (5000 is the config's own 50 hPa model top)
woof fetch --source gfs --cycle 2026-08-20T18 --hours 6 \
  --area 14.95,-115.69,63.65,-52.31 --p-top-pa 5000 --out gfs-raw

# b. confirm the sizing against the card that is actually present
woof check tree.toml

# c. materialize the physics authority
python -m woof.prepared_single_domain_forecast --materialize-authorities \
  --source gfs --base-experiment-config tree.toml \
  --base-wps-namelist tree.namelist.wps \
  --physics-profile morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1 \
  --output-directory tree-authority

# d. author the front-door manifest
woof fetch --source gfs --author-front-door-manifest \
  --out gfs-raw \
  --wps-namelist tree-authority/namelist.wps \
  --experiment-config tree-authority/experiment.toml
```

Step (d) prints a complete `rw-wps` command; run it, adding your
`--geog-root`. `rw-wps` in turn prints the runner command with both of its
values filled in:

```bash
woof-prepared-tree-forecast \
  --prepared-root <the directory rw-wps wrote> \
  --preparation-receipt-sha256 <the sha256 rw-wps printed>
```

Those last two values are produced by the preceding command and cannot be
known in advance; everything else above is complete as written. Section 3a
of [FIRST-LIGHT.md](FIRST-LIGHT.md) carries the same sequence.

For a single-domain `prepared:go` config (`--ladder 12`, the default),
`woof go tree.toml` runs the whole chain end to end.

**Before downscaling, confirm the archive actually has restarts** --
`ls RUN/gpuwmrst_d0*.npz` -- rather than assuming the emission's interval
reached the runner.

On this route `--child-surface-from` is reachable: the preparation writes
`wrfinput_d0N` for every nest under `<prepared>/wrf-native-input/`. Derive
the child at a nest's own geometry and that nest's `wrfinput` is the
surface source. To downscale the innermost nest to a brand-new finer grid,
emit a deeper ladder (`--ladder 12-3-1`) so the preparation builds a
`wrfinput_d03` on that finer grid; the forecast does not have to run the
extra nest, only the preparation has to build it.

---

# Deriving the child: the two forms

```bash
# Explicit: a child config plus its placement in the parent
woof downscale RUNDIR --parent-domain 3 \
  --parent-restart RUNDIR/gpuwmrst_d03_2024-05-06_01_00_00.npz \
  --child-config child.toml --ratio 2 \
  --i-parent-start 151 --j-parent-start 151 \
  --child-surface-from wrfinput_d04 \
  --max-boundary-interval-seconds 900 --out child-run

# Derived: geometry around a point, physics inherited verbatim from the
# parent's restart evidence, extent fitted to a VRAM budget
woof downscale RUNDIR --parent-domain 3 \
  --parent-restart RUNDIR/gpuwmrst_d03_2024-05-06_01_00_00.npz \
  --point 39.7,-84.0 --ratio 3 --vram-gib 6 --hours 3 \
  --max-boundary-interval-seconds 900 --out child-run
```

`--card` takes the same tiers `woof domain` does
(`12gb|16gb|24gb|32gb`, [HARDWARE.md](HARDWARE.md)) and `--vram-gib N`
covers anything between them; both feed the same budget model. Point mode
writes the derived TOML to `<out>/child.toml` for reuse, and inherits the
*parent's* per-domain tuning knobs verbatim (documented in the TOML
header).

`--hours` does not truncate the `parent_frames` list in the printed plan --
every archived frame is listed. The window lands in the derived config as
`run_seconds`.

### The lateral zone is sized for the parent

A derived child does not copy its parent's lateral zone. WRF's zone
(`spec_bdy_width = 5`, `spec_zone = 1`, `relax_zone = 4`) counts the
domain's own cells: at ratio 20 it is 0.75 km wide, a quarter of one 3 km
parent cell. And a root domain never relaxes `w`: its outer row copies the
first interior row, so an updraft that forms in the zone is carried onto
the boundary. The child's own storms and cold pools met the parent's state
head on at the edge, and the children made storms there.

`child.toml` therefore carries:

- `relax_zone` = two parent cells (`2 x ratio` child cells: 24 at ratio 12,
  40 at ratio 20), never less than WRF's 4 and never more than a quarter of
  the child's shorter side per edge, with `spec_bdy_width = spec_zone +
  relax_zone`;
- `relax_w = true`: `w` is relaxed toward the parent's `w` and specified
  from it on the outer row, as on a WRF nest, instead of being copied onto
  the boundary from the first interior row;
- `relax_timescale_s` = the time a 20 m/s flow takes to cross one child
  cell (12.5 s at 250 m, 7.5 s at 150 m), never shorter than 10 of the
  child's steps. It is set in seconds, so a step edited at review does not
  change how hard the zone pulls;
- `spec_exp = 0`: the zone ramps linearly whatever the parent's ramp. A
  parent's exponential ramp counts its own rows, and inherited it would
  cut the child's zone back to its outer few rows.

All four are ordinary `[run]` keys ([CONFIGURATION.md](CONFIGURATION.md))
and can be edited in the reviewed settings like any other. A `--child-config`
you write yourself keeps whatever it says.

Measured on a 2 h, 250 m child (300 x 300, ratio 12) of a 3 km HRRR parent
over the Front Range, 21 June 2023 from 18Z, with 15-minute boundaries:
the 99th percentile of column-maximum |w| within 5 cells of the edge
against the same statistic 40 or more cells in, at 20Z.

| zone | time scale | w on the edge | edge p99 (m/s) | interior p99 (m/s) | edge / interior |
|---|---|---|---|---|---|
| WRF's, 4 cells | 10 child steps | copied | 10.6 | 4.5 | 2.36 |
| 24 cells | 150 s (10 parent steps) | copied | 10.9 | 3.8 | 2.87 |
| 12 cells | 150 s | relaxed | 5.2 | 3.4 | 1.53 |
| 24 cells | 150 s | relaxed | 5.5 | 3.8 | 1.45 |
| 24 cells | 12.5 s | relaxed | 3.0 | 3.6 | 0.83 |

A wide zone alone does nothing while the edge copies `w`, and a zone that
relaxes slowly leaves the rows just inside the specified row free to part
from it: on the 150 s arm, rows 1 to 3 carried 5.5 to 6.0 m/s where the
parent has 3.1 to 3.3. The parent itself is not uniform over this ground:
its own |w| within 5 cells of the child's edge is 1.4 to 3.2 times its
interior value through the window, so a child that follows its parent at
the edge is not expected to read 1.0 there.

The shipped zone reads the same at the edge whatever the boundary cadence:
3.0 m/s at the edge against 3.4 m/s inside at 20Z with 30-minute and with
hourly boundaries. The cadence shows inside instead: after 2 h the mean
2 m temperature difference 40 or more cells in was 0.11 K between the
15-minute and 30-minute runs and 0.17 K between the 15-minute and hourly
runs, against 0.42 K between the child and its parent.

Where the relaxation lets go, the child makes some vertical motion of its
own: on the same arm, pooled over 19Z to 20Z, the 99th percentile 24 to 28
cells in was 4.4 to 4.9 m/s, against 3.0 to 3.4 on either side and 2.8 in
the parent. An exponential ramp (`spec_exp` 0.1 or 0.2) moved that band
outward and widened it without lowering it, so the derived zone keeps
WRF's linear ramp (`spec_exp = 0`).

A wider zone costs boundary memory in proportion to its width; the plan
review prices it. Streamed, a tile's interior seams relax nothing, and a
tile's own cells read everything within its halo each step, so a tile whose
interior, widened by its halo, reaches a relaxation zone has to own that
domain edge: its compute window has to reach the edge. Each seam therefore
sits at least zone + halo cells from a forced edge (58 cells for a 40-cell
zone under an 18-cell halo), or no more than the halo's width from it, where
the window of the tile beside it runs out to that edge. The tile planner
only proposes such tilings, and a pinned tile size that breaks the rule is
refused before the first step.

### A drawn extent on the measured card

`--child-size NX,NY` with `--auto-vram` prices the extent you chose on the
card in front of you: the door measures the local card once, prices the
given child on it (`memory.basis` is `measured-local`, `gpu_sizing` carries
the measurement) and reports `memory.fits`. The same holds for
`--child-config` beside `--auto-vram`. Only `--card` and `--vram-gib` are
exclusive with measuring, because a measured card and a declared capacity
are two answers to one budget.

An extent that reaches past the parent's interior around the point is not
refused: it shrinks to the largest centered extent the parent holds there,
and the plan's `warnings` carry one sentence saying what was asked, what
it became and why. Only a child that cannot exist at all (the smallest
legal extent already reaches past the interior) is refused, with the way
out.

### One price, one decision

The plan review prices the child once with the itemized estimator and
takes its `[tiles]` decision on that price, against the card it holds; the
run calls the same function on a card measured cold, before the child's own state was built on the device. The run measures the card whatever
`[tiles]` says, because the price is taken on it as well as the decision,
so a run without `--tiles` records the price its review gave. The plan document's `streaming`
block (`mode` resident or streamed, `why`, `budget_bytes`,
`peak_envelope_bytes`, `machine` and `machine_free_bytes` for the card it
was priced on, `tile` when streamed) sits beside the `memory`
block, and the child's `report.json` repeats it. With `[tiles]` set to
`auto`, `memory.fits` is judged on the budget the streaming decision used,
so `mode` resident comes with `fits` true and `mode` streamed with `fits`
false: read `mode` for whether the child runs, `fits` for whether it runs
resident. Without a `[tiles]` block the child is resident by configuration
and `fits` reports the fit ceiling. `mode` is null only when the review
holds no card to plan against; the run then decides on the card it starts
on. A child that cannot run on the card even streamed refuses at
plan review, and at run start before preprocessing, never after the
archive has been interpolated. `child_outline` gives the footprint's four
corner `[lat, lon]` pairs read from the parent's own grid, beside
`parent_domain` and `child_grid_id`, so a front end draws exactly what
will run.

### A downscaled run is itself a parent

The child honours its configuration's `restart_interval_s` (inherited from
the parent by `--point` derivation): it writes a checkpoint set at every
interval inside its window and once more at its end, under the instant
naming `--parent-restart latest` discovers
(`gpuwmrst_d02_YYYY-MM-DD_HH_MM_SS.npz` beside its `wrfout_d02_*` frames).
`restart_interval_s = 0` writes only the final set. The child keeps one
set, the newest: each new set is written whole before the older one is
removed, so a finished child leaves its end state and nothing else.
`--keep-checkpoints N` keeps the newest N (0 keeps every set), and
`WOOF_KEEP_CHECKPOINTS`, the variable `woof run-plan` and `woof go`
export, decides when the flag is absent. A child is re-run rather than
resumed, and the next downscale reads the newest set, so one is enough.
So a finished downscale chains:

```bash
woof downscale CHILD_RUN --parent-domain 2 --parent-restart latest   --point 40.55,-103.60 --ratio 3 --child-size 60,60 --auto-vram   --hours 2 --out GRANDCHILD
```

derives a grid 3 grandchild at a third of the child's spacing from the
d02 frames at the run root and the d02 checkpoint sets beside them. A
child is named after its parent as the parent's `run-manifest.json` names
it (the parent's run folder when it has none), so a child of the forecast
`Front Range 12 km` is `Downscale of Front Range 12 km · d02 ×3 · 4 km`, and
this grandchild adds its own grid to that name:
`Downscale of Front Range 12 km · d02 ×3 · 4 km · d03 ×3 · 1.33 km`. With `--parent-domain N`
the physics evidence is the d0N member of the newest checkpoint set, so a
multi-domain run root serves its nest's frames with that nest's own
physics; a set with no such member is refused naming the members it has.

### The disk a child needs

The plan's `disk` block is what the child will write, priced on its own
clock before it starts: `history_bytes`, `checkpoint_bytes` (the sets
held at once, one more than it keeps while a new set is written),
`picture_bytes` and `total_bytes`, beside `free_bytes` on the disk that
holds `--out`. A child and a forecast share one price per product from
`woof/data/picture-bytes.v1.json`, interpolated between its two
horizontal-grid brackets and held outside them. A windowed product is
charged only on the whole-hour frames that close its window.
`pictures_per_frame` is still the count per frame: it
is one for each product `--render-products` names, every product a local
run of that length can draw for `all`, and none for `none`, so a shorter list is a
smaller figure. Measured frame by frame, a 250 m child's pictures averaged
0.36 MB and a 3 km parent's 0.68 MB. `download_bytes` and
`preparation_bytes` are zero: a child reads its parent's files where they
lie and builds its start and boundaries in memory. The review prints the
block as one line, the page's review and the desktop's show the total
beside the free space, `report.json` carries it beside `checkpoints_written` and
`keep_checkpoints`, and a child whose total is larger than the free space
is refused before it starts, naming both figures and only the flags that
would shrink it; `--dry-run` prints the plan and says the same as a
warning.

## The contract, in order

1. **Prove the parent.** Complete frame inventory, frozen geometry,
   regular cadence, one producer -- checked before anything runs.
2. **Prove the physics.** The parent's microphysics identity must come
   from companion evidence -- a woof restart file or the WRF
   `namelist.input` -- never inferred from the variable inventory.
   Cross-scheme conversion is explicit and fail-closed (active condensate
   is never paired with a fabricated zero number moment; NSSL targets
   require the official `calcnfromq` diagnosis).
3. **The boundary cadence defaults to the archive's own.** With no cadence
   flag the tool uses the parent history cadence and says so in one
   warning line:

   ```
   warning: using the parent archive's own 43200 s history cadence as the boundary cadence; pass --max-boundary-interval-seconds SECONDS to bound it
   warning: parent cadence 43200 s is coarser than the 900 s guidance for downscaling
   ```

   `--max-boundary-interval-seconds` bounds it explicitly and
   `--accept-parent-cadence` is the warning-free spelling of the default
   (one or the other, never both). Boundary cadence changes the forcing
   supplied to the child. The single comparison below cannot rank that effect
   against other sources of divergence, so the guidance is not a measured
   error bound. A runnable job is never refused for it.
4. **Full-physics children get their land identity and soil warm start
   automatically.** With `--child-surface-from` they come from a `wrfinput`
   or history file on the exact child grid, mirroring `ndown`'s own
   requirement; without it they are interpolated off the parent's own
   history frame, which is WRF's route for a nest with
   `input_from_file = .false.` and costs the child its own coastlines. The
   run says which one it used, once, and records the whole derivation in
   `report.json`. Microphysics-only children need neither. The surface is
   resolved at the front door, before any preprocessing, so a parent whose
   history cannot seed a child refuses at plan time naming the missing
   fields; `--dry-run` warns instead of refusing so the derived placement
   can still be read.
5. **Every run writes `report.json`** with SHA-256 receipts of the parent
   frames, the physics evidence, the surface source, the boundary-clock
   identity, and the outputs. A failed run writes `failure-capsule.json`
   carrying the config path and its sha256, every input hash, the GPU uuid
   and driver, the git commit, and `last_phase`.

## The measured cadence cost

Acceptance measurement, RTX 5090, 2026-07-29: a 1 km parent domain
(501 x 501, Thompson) archived **hourly**, downscaled to a 500 m child
(400 x 400, ratio 2, dt 2.5 s, full physics) over 3 h spanning convective
initiation, then scored against the *live-nest* child of the same run on
the interior grid (5-row rim excluded).

At F+0.0 the offline cold start matches the live nest on the four metrics
this table scores. That row is the whole scope of that statement: four
comparator metrics on one pair of runs, with no full-state digest taken.
The later rows measure differences from the live nest, not errors against
observations. They combine the changed forcing path (hourly interval-linear
boundaries versus every-parent-step forcing), growth of small numerical
differences during convection, and any start-state differences the four F+0
metrics do not detect. No perturbed live-nest control was run, so this single
pair cannot separate those contributions.

| lead | T2 MAE (K) | T2 corr | PSFC MAE (Pa) | wind10 corr | refl corr | refl MAE (dBZ) | CSI@20 |
|---|---|---|---|---|---|---|---|
| F+0.0 | 0.000 | 1.000 | 0.1 | 1.000 | -- | -- | -- |
| F+0.5 | 0.133 | 0.992 | 37.5 | 0.978 | 0.922 | 0.31 | -- |
| F+1.0 | 0.127 | 0.988 | 23.0 | 0.957 | 0.906 | 0.80 | -- |
| F+1.5 | 0.156 | 0.989 | 26.6 | 0.943 | 0.845 | 1.20 | -- |
| F+2.0 | 0.253 | 0.977 | 23.9 | 0.916 | 0.262 | 3.05 | -- |
| F+2.5 | 0.289 | 0.975 | 29.8 | 0.871 | 0.248 | 13.06 | 0.000 |
| F+3.0 | 0.430 | 0.964 | 28.4 | 0.802 | 0.148 | 25.38 | 0.020 |

Two regimes:

- **Mesoscale envelope: close.** T2 correlation never drops below 0.96,
  PSFC MAE stays under 40 Pa, 10 m wind correlation is 0.80 at F+3 h. The
  child tracks the parent-constrained mesoscale state for the full window.
- **Convective scale: decorrelates at initiation.** Interior reflectivity
  maximum crosses 0 dBZ between F+1.5 and F+2.0; exactly there,
  reflectivity correlation collapses 0.845 -> 0.262 and CSI at 20 dBZ is
  near zero. Both runs make storms -- in different places.

Run cost: 4,320 child steps, 66.3 min wall, 7.98 GiB pool peak, no
non-finite values, receipts complete.

## The 15-minute guidance

**Write parent history at 15-minute or denser cadence when you plan to
downscale.** The table above measures the difference between an hourly-boundary
offline child and a live nest at 500 m across convective initiation in one
three-hour case. In that case, temperature, pressure and the wind envelope
stayed close while cell placement decorrelated. It is not a general skill
result or an isolated measurement of cadence error, and no denser-cadence arm
here establishes that 15-minute boundaries recover the live nest's storms.

The CLI prints this guidance whenever the archive is coarser than 15
minutes, however the cadence was chosen. `--accept-parent-cadence` is
recorded in `report.json` as
`boundary_cadence_provenance.accepted_parent_cadence: true` beside the
ceiling and the effective boundary interval; an explicit
`--max-boundary-interval-seconds` records `false` there instead.

Output cadence is a config decision on the parent run (`history` interval
per domain), so the time to make it is before the parent runs, not after.
On the ERA5 route that means `--history-interval 900` in step 1.

## Rendering the child

**The child draws itself.** `woof downscale` renders the finished child
the way every other WOOF forecast door renders its forecast: the same
stage, the same product catalog, the same folder shape. The pictures land
in `<out>/png/<domain>/<product>/<valid-day>/`, beside the frames they
came from, and each frame is drawn as it is written, one render at a time
in the background beside the integration, so every hour of the child is
readable while the rest is still integrating. The event stream carries
`first_products_ready` for the analysis frame and `live_products_ready`
for each frame after it. The end of the run draws only what is still
missing: the windowed pictures (`qpf_1h`, `qpf_total`, run maxima), which
need the whole series, and any frame whose pictures it cannot verify by
digest. The run's event stream carries the `finalize` stage and its
render summary, which is what fills the rendered-picture count a run
browser shows.

`--render-products` chooses the set, in `woof render --products`'
own spelling: a comma-separated list of catalog slugs, `all` (the
default, and the same default `woof go` takes), or `none` to keep only
the frames.

```bash
woof downscale RUN --parent-restart latest --point 39.5,-84.0 \
  --out out/child --render-products composite_reflectivity,mslp_10m_winds
```

The names are the renderer's own catalog slugs; `woof render
--list-products` prints every one this install can draw. A slug the
catalog does not carry is refused at plan review, before a single
archived frame is opened.

A requested set that produced no picture is a refusal naming the render
command to run by hand, and a render stage that exits nonzero is the same
kind of refusal: the forecast keeps its `PASS`, `report.json` gains a
`products` block saying the pictures failed and naming that command, and
the door exits 2 with the sentence rather than a traceback. A computer
with no staged Rust renderer is turned away before the child is
integrated rather than after, with `--render-products none` named as the
way to run the forecast anyway.

**A child that does not finish keeps the pictures it rendered.** A run that
stops partway through -- non-finite, a refusal raised mid-run, an
interrupt -- leaves every picture drawn while it ran exactly where it was
published, and adds the verdict those pictures cannot carry themselves.
A stop draws nothing more: the frame being drawn is abandoned and no
queued frame is drawn. A child that failed on its own first finishes
drawing the frames it had already written.

* `DID-NOT-FINISH.txt` at the top of `<out>/png/` says where the forecast
  stopped (model second and step, of how many), why it stopped, how many
  pictures are in the folder, which frames were written before the stop,
  and that every picture there is of a frame written before it.
* `render-summary.json` beside them carries `status: did-not-finish`
  with `pictures_on_disk` and the banner's path, so a run browser that
  reads that file finds the pictures rather than an absent folder.
* The event stream carries a `warning` with code `early_render_kept`
  naming the same count and banner.
* `report.json` is published for this outcome too, whatever stopped the
  run: `result` is `FAIL`, `failure` is the capsule naming what stopped
  it -- or, for a stop that composed no capsule, such as an interrupt,
  an `OSError` from a mount that dropped or a contract error raised
  mid-run, the sentence the banner carries (`summary`) with the whole of
  what was raised (`message`) and its class (`error_type`) -- and the
  `products` block reads `status: KEPT` with `pictures_on_disk`, the
  banner's path, and a next step that is the pictures rather than
  redrawing them. The banner, the render summary, the event stream and
  the report are one account of one run.
* A picture folder that cannot be listed -- a permission wall, a dropped
  mount, a path that is a file -- says so and carries the error, in the
  banner, in `render-summary.json` (`pictures_on_disk` null beside
  `pictures_on_disk_error`) and in the report. It is not counted as
  zero: a tree nobody could read is not a tree with nothing in it, and
  reading it as one is what tells somebody who still has their pictures
  that they have none.

The frames and the checkpoints are kept too, so the run can still be
drawn in full by hand at any time. Earlier releases removed the pictures
instead, which left a child that stopped part way through its forecast
with nothing to look at.

`woof render` handles the child's wrfouts like any other run's, and
sub-hourly cadences render exactly: every frame carries its precise
`valid_..._lead_...` stamp, so a 15-minute child cadence never rounds to a
fake whole hour. To see what downscaling changed, render the parent domain
and the child into separate directories and compose labeled pair sheets:

```bash
woof render --pair out/parent/png out/child/png --out out/compare
```

## When a child stops being finite

A child's own health check samples the state every
`--health-interval-seconds` of model time (60 s by default) and refuses the
forecast the first time a field is not finite. That refusal is a capsule,
not a step number. The climb in the example below is a real child's, read
off its own health record; the places, the box and the count are whatever
the run you are reading about records:

```
The child blew up: w_max ran 10.73, 13.22, 15.77, 18.12, 21.05, 22.97 m/s
over the 300 model seconds before the health check after step 6624 of 69120
(model second 2760 of 28800) found W non-finite in 4,812 cells, all inside
k 10-14, j 398-404, i 385-391.
Non-finite fields at that check, listed dynamics first and then moisture,
which is not the order they failed in:
  W: 4,812 cells of 31,840,200, all inside k 10-14, j 398-404, i 385-391
The check after step 6480, 144 steps earlier, found u, w and theta' finite;
the check reads only those three, and a survey taken afterwards cannot say
which cell or which field went first. The last |w| maximum measured, 22.97
m/s at the check after step 6480, was at (k=12, j=401, i=388), 388 cells in
from the west edge.
The last 7 health checks, 60 model seconds apart:
  step 5760  model second 2400  w_max 10.73 m/s at (k=11, j=399, i=386)  CFL 0.1903
  ...
  step 6624  model second 2760  w_max non-finite  CFL not computed
Next: every frame the run did reach is on disk and can be drawn by hand:
  woof render <out> --series ...
```

Here is what each part says, and what it cannot say. The **fields** are
every carrier the survey found non-finite at that check. They are listed in
a fixed order, dynamics first and then moisture, and that order is not the
order they failed in: the check reads only the u, w and theta' maxima, once
per health interval (48 to 144 steps of a typical child), so by the time it
finds them gone the other fields have had that long, or longer, to go with
them. The **box** is what the check measured: the k, j and
i ranges every non-finite cell falls inside and how many cells there are,
with a note when the box touches a lateral edge (j 0 is the south edge, the
last j the north, i 0 the west, the last i the east). A single cell is
named only when exactly one cell went; of many cells none is named, because
a survey taken after they went cannot know which went first. The **last
|w| maximum** is where |w| was largest at the last check
that could measure it, and how many cells in from the nearest lateral edge;
it is the nearest thing to where the climb started that the record holds.
Every row of the trend carries the same place, and so does every
`child_step` event line, as `w_max_cell` and `w_max_edge`, so a run's own
record says where |w| was largest at each check. The **model
second** says where in the forecast the check fell, which the step alone
does not. The **trend** is the health record read back over the window: a
CFL that never left its band while `w_max` doubled says plainly that the
time step was not what ran out.

Every document this outcome writes is JSON a strict reader can open.
`NaN` is not a JSON token, so a reading that went travels as `null` beside
a state word rather than as a number: `"w_max": null, "w_max_state":
"non-finite"` for a field that stopped being finite, and `"not computed"`
for a reading nothing ever produced, such as the CFL, which is not
computed at all from fields that are not finite. That is the shape in
`report.json`, on the `child_step` event line and in the run-plan event
stream alike, and it is why the table above prints the state word where
the number would be, with no unit after it.

`report.json` is written for this outcome too, with `"result": "FAIL"`, the
same capsule under `failure`, and a `products` block reading `"status":
"KEPT"` with `pictures_on_disk` and the path of the `DID-NOT-FINISH.txt`
banner standing over those pictures: nothing is removed from the picture
folder on any failure path. The run-plan `failed` event carries
the capsule's first sentence, which is the line a run browser shows. The
frames and checkpoints written before the refusal stay on disk and render
like any other run's, and `woof resume` reads that `"result"` rather than
the presence of the file: a directory holding a `FAIL` report is told to
run `woof downscale` again on a fresh `--out`, with the capsule's first
sentence quoted back as the reason it did not finish.

Which directories that reading applies to is decided by what the run
wrote in them, not by the configuration file beside them. `child.toml`
lands in `--out` only when `--point` derived it, so a run given its
configuration with `--child-config` records none; the route is named by
`downscale-plan.json`, or by `report.json` naming its pipeline, counting
its steps or carrying the failure capsule.

## Giving the child its own vertical levels

An LES child usually wants the levels, not just the columns: `docs/public/LES.md`
measures the nested child carrying 12.7% of its turbulence in the subgrid model
against 7.9% for a properly resolved column, because it inherits its parent's
ladder. `--child-levels` gives it its own:

```bash
woof downscale parent/ --parent-restart latest   --point 39.5,-84.0 --ratio 5 --child-levels 96,2.5 --out out/child
```

`N,STRETCH` is the level count and the tanh clustering toward the ground; the
derived `child.toml` carries the resulting ladder explicitly as `eta_levels`, so
the grid the child was prepared on is the grid it integrates on and both are
readable in the config. The initial state and the whole lateral boundary table
set are remapped once, at preparation, on the host; the integration loop is
unchanged.

The remap conserves what it moves. Because every admissible ladder makes the
reference dry pressure run from `p_s` at the surface to `p_top` at the model top
identically, two ladders sharing `p_top`/`hybrid_opt`/`etac` partition the same
column with coincident endpoints, and moving a field between them is exact
rebinning rather than interpolation. Measured on a prepared child: water
substance drifts 1.2e-16 relative, potential temperature 0.0, the column's dry
mass closes to 0.0 Pa, and the child's model top lands on the parent's exactly.

### The LES case is stated at the door

A child at or below **250 m** horizontal spacing is in a different regime
from the mesoscale run its parent's configuration was written for, and the
door says so before the run starts. 250 m is this tree's own number:
`docs/public/LES.md` ships its nested LES child at that spacing and calls it
coarse LES at the gray-zone edge.

The statement fires when the child is at or below that spacing **and**
any of the following is true of it:

- it inherits the parent's vertical ladder, which it does whenever
  `--child-levels` is absent and its configuration names no ladder of its
  own, or names one of the same depth as the parent tape's;
- it runs a 1-D boundary-layer scheme (`bl_pbl_physics` other than 0) with
  no 3-D closure (`km_opt` other than 2 or 3); or
- nothing mixes heat or moisture vertically at all, which is
  `bl_pbl_physics = 0` with `km_opt` 1 or 4: those two compute no vertical
  exchange pair of their own, and the scheme that would otherwise do it is
  off. (`km_opt = 0` is not in this list: this tree admits it only behind
  an acknowledgement written out in full, so that child was already told.)

It names the spacing, the threshold, which of the three it found, and the
shape that goes with it: a boundary-layer-scheme child at LES spacing
tends to grow vertical velocity check after check until the field goes
non-finite; a child with no vertical mixing has nothing but the motion it
resolves carrying heat and moisture between its levels; a child on a
ladder chosen for a coarser grid leaves more of its turbulence to the
subgrid model than a resolved column does, 12.7 percent against 7.9
(`docs/public/LES.md`). The ways out are named too: `--child-levels N,STRETCH` for the ladder,
`km_opt = 3` (3-D Smagorinsky) or `km_opt = 2` (prognostic TKE) with
`bl_pbl_physics = 0` in the `--child-config` TOML for the closure, and
`--child-surface-from` for the geography a grid this fine can resolve and
the parent's cannot. `--explain` adds why.

It is a **statement, not a refusal**: the shipped nested LES child is itself
a 250 m child on its grandparent's ladder, nothing about the run changes,
and the command still exits 0 with its plan. `downscale-plan.json` carries
the same numbers as fields under `les_regime`, `null` when the child is not
in that regime.

There is no downscale flag for the closure. `--child-config` is how a child
gets one, because the TOML is where `km_opt` and `bl_pbl_physics` live; a
`--point`-derived child takes the parent's physics verbatim
(`woof/downscale.py`, `_derive_child_run_config`), which is exactly how a
child arrives at LES spacing still running its parent's boundary-layer
scheme.

Two things it will refuse, both by name: a bare level count with no stretch
(a uniform ladder under a stretched parent is a different atmosphere, not a
finer sampling of one), and a ladder whose depth its own radiation cannot run
(RRTMGP tops out at model plus cap layers <= 128, so `nz = 120` at
`p_top = 5000 Pa` is refused at preparation rather than at the first radiative
call).

Those three coordinate parameters stay shared with the parent deliberately: they
are what make the endpoints coincide, and a per-domain `p_top` would leave the
remap extrapolating above the model top with no state to extrapolate from.

## Ozone on the child

A child under legacy RRTMG (`ra_lw_physics = 4`, `ra_sw_physics = 4`,
`ra_rrtmg_variant = "rrtmg_legacy"`) with `o3input = 2`, which is the
RunConfig default and the pairing a downscale of a default forecast
inherits, evaluates the packaged CAM ozone climatology on its own grid. It
has no parent in memory to take a field from, and it is configured as a WRF
root (`specified = true`, `nested = false`, lateral boundaries read from the
archive), which is the domain WRF hands its own `oznini` and `ozn_p_int`
evaluation; only a resident nest is given the parent's field. `report.json`
says which under `child_ozone_routing`: `child-grid-climatology` here,
`wrapper-o3data` under `o3input = 0`, and `null` when the child's radiation
carries no ozone routing at all. Earlier releases refused this pairing on
the offline route and named two exits the desktop cannot take; the refusal
is gone and a bare downscale runs.

What the choice is worth was measured on an RTX 4090 against the field a
resident nest would be given (the parent's climatology carried onto the
child grid by the same SINT operator the nest transfer uses), on a 72 x 72 x
30 parent at 3 km and its 60 x 60 x 30 ratio-3 child at 1 km, three parent
frames at 900 s: per-layer relative difference at most 1.29e-2 (mean
5.9e-3, the widest column at the child's corner), column-integrated ozone
1.3635e-3 against 1.3675e-3 kg m-2 (relative difference at most 2.96e-3),
and two 10-minute runs of that child differing in nothing but the ozone
field end 0.0086 K apart in potential temperature at the widest point
(mean 1.5e-4 K, RMS 2.2e-4 K, the widest layer near 350 hPa). That is the
nest-interpolation seam the legacy port already documents, not a different
atmosphere, so the parent's history is not read for ozone and needs no
`o3rad` in it. The receipts are under
`receipts/` (`ozone-field-reading.json`,
`ozone-heating-reading.json`) with the scripts that wrote them beside them.

## Known limits

- **Terrain is SINT-inherited** from the parent rather than rebuilt at the
  child's resolution. Downscaling a coarse-terrain source gives the child
  that coarse terrain at a fine grid spacing.
- **The child's eta ladder is its own only if you ask for one.** Without
  `--child-levels` the child keeps the parent's levels, exactly as before.
  With it the child is built on its own ladder through a conservative
  vertical remap; `p_top`, `hybrid_opt` and `etac` stay shared with the
  parent, and a level count given without a stretch is refused rather than
  filled in with a uniform ladder.
- **`woof downscale --card` defaults to `24gb`** while `woof domain`
  measures the local card.
- **A parent needs a checkpoint from the history domain you downscale.**
  Configurations from `woof domain` checkpoint hourly; one written by
  hand with `restart_interval_s = 0.0` writes none, so set a positive
  interval before running a parent from it.
- One fixed child per invocation; a downscaled run is itself a parent
  (its `restart_interval_s` checkpoints are discoverable), so repeat the
  command on the child's run directory with `--parent-domain 2` for the
  next level. A parent series whose geometry changes as
  a nest moves is refused, and the standalone child config does not accept
  a relocation block. Live moving nests use the prepared-corridor route in
  [TUI task modes](TUI-TASK-MODES.md#follow-weather-without-changing-what-the-tracker-means).
- Stock-WRF parents work (history plus `namelist.input` as physics
  evidence) with the same explicit-cadence contract; a point-mode child of
  a WRF parent needs `--child-config`, because a WRF namelist carries no
  complete woof run configuration.
- The child runs the same bound boundary-clock semantics as the production
  nest tree (WRF's `dtbc` recurrence), and its restarts record that
  identity, so checkpoints cannot silently mix with legacy-clock
  trajectories.

Full contract details:
[native-offline-child-contract.md](../native-offline-child-contract.md).
