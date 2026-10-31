# First light

This is the full path from a bare machine to rendered forecast
products, with real wall-clock timings. Every number on this page was
measured in a first-time-user acceptance transcript on 2026-07-29:
RTX 5090 (32 GiB), Windows 11, Python 3.13, a warm pip cache, and no
prior WOOF state. Your times will vary with network, disk, and card;
the shape of the path will not.

> WOOF is a research and educational tool, never a substitute for
> official warnings from your national meteorological service.

## 0. What you need

- Python 3.11+, git, and a Rust toolchain (`cargo`), Rust 1.94 or newer
  (`rustc --version`). The terminal and Zarr reader workspaces refuse an
  older compiler, and a distribution's packaged Rust can be older;
  `rustup update stable` brings a rustup install up to date.
- For the GPU forecast loop: an NVIDIA card with CUDA 12.x/13.x
  (tested through 13.0) and the CuPy extra that matches the box's CUDA
  major -- `[gpu-cu12]` on CUDA 12.x, `[gpu-cu13]` where CUDA is 13-only
  (`nvidia-smi` prints it; `woof doctor` names the right one). 8 GiB
  VRAM is enough for a first single-domain run; nest ladders are sized
  to your card in step 2.
- Disk: budget several GB. In the acceptance transcript the whole tree
  (venvs, data, outputs) reached 7.6 GB, dominated by hourly output
  frames at ~198 MB each.
- Static geography: the WPS_GEOG tree, staged by `woof fetch-geog`
  (~1.3 GB one-time download, ~16 GB unpacked; see
  [DATA.md](DATA.md#static-geography-wps_geog)).

## 1. Install (measured: a few minutes from clean)

| step | measured wall |
|---|---|
| `git clone` (local) | 3.0 s |
| `python -m venv` + `pip install -e recast-woof-data` + `pip install -e '.[gpu-cu12,render]'` | ~25 s cached; a fresh machine downloads ~150 MB (numpy, matplotlib, netCDF4, CuPy) |
| Optional `woof fetch-tables` (stage Thompson tables ahead of first use, a one-time ~243 MiB release-asset download from a checkout, SHA-256 verified; a no-op once staged) | connection-speed bound; instant when already present |
| `cargo build --release --locked --offline` in `tools/grib1_bridge` | ~8 s (vendored workspace, no network) |
| `cargo build --release --locked --offline` in `tools/rustwx` (the production render engine; `--no-render` skips it) | 67 s from clean (measured 2026-07-29, same box; vendored workspace, no network) |
| `cargo build --release --locked --offline` in `tools/zarr_bridge`, `tools/rw_wps` and `tools/region_global_dealias` (terminal, Zarr reader, mapped decode engine, dealiasing library) | 114 s together from clean, 78 s of it the Zarr reader (measured 2026-09-27 on a Linux node; vendored, no network) |
| `woof doctor` | seconds |

One command does all of it -- `bash install.sh` (POSIX; the universal
form, mode-bit independent) or `.\install.ps1`
(PowerShell) from the checkout root: a venv on free-threaded Python 3.14t
(found on PATH or installed with uv; multi-card forecasts run about twice
as fast on it, and `--python` / `-Python` picks another interpreter), the checkout's `recast-woof-data`
companion, `[gpu-cu12,render]` extras, the offline Rust builds of all nine
native workspaces (the `tools/grib1_bridge` GRIB bridges, the
`tools/rustwx` render engine, the
`tools/zarr_bridge` Zarr reader, the `tools/rw_wps` mapped decode engine,
the `tools/region_global_dealias` dealiasing library, the
`tools/nextgen_hydrofabric` hydro payload, `tools/nextgen_forcing` forcing
`tools/nextgen_verify` gauge verification and `tools/hydro_world`, `tools/hydro_world_params`,
`tools/nextgen_reservoir_prepare` and `tools/hydro_world_init` hydro preparation and `tools/ocean_prepare` and `tools/rw-bathy` ocean preparation, `tools/wind-table` wind-turbine tables, `tools/nextgen_output` hydro output and `tools/rw_icechunk` Icechunk store workspaces, fetching their
locked dependencies first; `--no-render` /
`-NoRender` skips the renderer, the long pole of install), and a closing
`woof doctor`; it offers rustup if `cargo` is missing and is safe to
re-run. The equivalent manual steps install the checkout's own
`recast-woof-data` first, because the engine requires the companion of its own
version:

POSIX:

```bash
git clone https://github.com/recastsystems/woof woof && cd woof
python -m venv .venv && source .venv/bin/activate
python -m pip install -e recast-woof-data
python -m pip install -e '.[gpu-cu12,render]'   # or gpu-cu13
woof fetch-tables     # optional: the first Thompson run stages missing tables
woof fetch-geog       # WPS_GEOG static tree: ~1.3 GB down, ~16 GB unpacked
for workspace in tools/grib1_bridge tools/rustwx tools/zarr_bridge tools/rw_wps tools/region_global_dealias tools/nextgen_hydrofabric tools/nextgen_forcing tools/nextgen_verify tools/hydro_world tools/hydro_world_params tools/nextgen_reservoir_prepare tools/hydro_world_init tools/ocean_prepare tools/rw-bathy tools/wind-table tools/nextgen_output tools/rw_icechunk; do
  (cd "$workspace" && cargo fetch --locked && cargo build --release --locked --offline)
done
woof doctor
```

Windows (PowerShell):

```powershell
git clone https://github.com/recastsystems/woof woof; cd woof
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install -e recast-woof-data
python -m pip install -e '.[gpu-cu12,render]'   # or gpu-cu13
woof fetch-tables     # optional: the first Thompson run stages missing tables
woof fetch-geog       # WPS_GEOG static tree: ~1.3 GB down, ~16 GB unpacked
foreach ($workspace in 'tools\grib1_bridge', 'tools\rustwx', 'tools\zarr_bridge', 'tools\rw_wps', 'tools\region_global_dealias', 'tools\nextgen_hydrofabric', 'tools\nextgen_forcing', 'tools\nextgen_verify', 'tools\hydro_world', 'tools\hydro_world_params', 'tools\nextgen_reservoir_prepare', 'tools\hydro_world_init', 'tools\ocean_prepare', 'tools\rw-bathy', 'tools\wind-table', 'tools\nextgen_output', 'tools\rw_icechunk') {
  Push-Location $workspace; cargo fetch --locked; cargo build --release --locked --offline; Pop-Location
}
woof doctor
```

`woof doctor` checks CuPy, the render extra, the rust render engine,
every Rust bridge executable this release declares, the CPU library, the
packaged physics
tables, and your data-root layout, and prints the exact command that
fixes anything missing. Run it until it is clean; everything downstream
assumes it is.

You do not have to build the render engine to have one. On a supported
platform `woof fetch-bridges` downloads this release's prebuilt
artifacts and verifies every byte against the pinned SHA-256 digests
packaged in the wheel; `woof setup` runs it as its first step. The
renderer's map assets -- the coastline, border, state and county
shapefiles -- arrive with the `recast-woof-data` package every install pulls,
and `woof` hands them to the renderer. Without them plots come out with
the weather drawn over a blank rectangle, so a run that finds none says so
in a `render_basemap_missing` warning event with the command that
restores them.

(The `tools/rustwx` build remains skippable, and doctor labels its
absence `info`, not a gap -- but `woof render` then REFUSES rather than
drawing with something else, because weather-field product plots come
from `rw_wrfbatch`. `--engine matplotlib` is still reachable by name;
it is a workaround -- five products against the rust catalog's 151 --
and it prints a `WORKAROUND:` line on every run.)

## 2. Size a domain to your card (measured: 1.7 s)

```bash
woof domain --point 35.3,-97.5 --card 24gb --cycle 1999-05-03T12 \
  --hours 6 --out configs/myarea.toml
```

The wizard centers a nest ladder on your point and bisects the grid
sizes through the real VRAM estimator until the projected machine peak
fits your card's budget. Output on the 24 GB tier (Windows):

```
  domain    dx        mass grid      dt         resident
  d01     12.000 km   164 x 130        60 s     0.59 GiB
  d02      3.000 km   328 x 256        15 s     2.09 GiB
  d03      1.000 km   354 x 276         5 s     2.44 GiB
  d04      0.500 km   284 x 220       5/2 s     1.58 GiB
  peak envelope: footprint 10.52 x 1.75 WDDM floor = 18.41 GiB, which is above the affine form (estimate 6.40 + non-pool 2.30 (CUDA context + local-memory backing store) + 0.50 unmodelled + 5% of the estimate x 3 nest(s) = 10.16 GiB) and therefore binds
    envelope basis: windows; measured, 1 WDDM run
  ingest (preprocessing): root 2 forcing times x 0.22 GiB each, 2 resident at a time + 3 nest initial state(s) 2.30 GiB, all resident for the single export transaction = 2.75 GiB resident; peak envelope 5.72 GiB
    ingest envelope basis: itemized analysis, model state and vertical setup, x1.10 setup residual and x1.20 pool headroom, measured on four CUDA preparations (1792x1024x55 to a 3:1 nest, H100, 2026-09-28), + CUDA context
  BINDING PHASE: the forecast is the memory-binding phase at 18.41 GiB peak envelope (forecast 18.41 GiB, ingest 5.72 GiB); it fits the 19.57 GiB budget with 1.16 GiB to spare
  budget 19.57 GiB (24 GiB card presents about 22.56 GiB free, minus this suite's 2.99 GiB reserve); headroom 1.16 GiB
```

The same command on Linux drops the Windows-only pool constants and the
WDDM floor with them, so the affine form binds, and the card sizes a
much larger grid -- roughly one card tier's worth
([HARDWARE.md](HARDWARE.md)). `--card 12gb` is a Linux tier for exactly
this reason.

Note what the budget line says: a 24 GiB card is priced at about
22.56 GiB free, not 24. No card hands over its nameplate capacity, and
a tier that assumes it does emits configs that fail the product's own
`woof check` on a real card of that tier.

It emits the experiment TOML, a matching `namelist.wps`, and prints the
exact `woof fetch` command for the data it needs. With `--source hrrr`
it emits the whole input set the native HRRR routes read -- the
`<stem>.d01-target.json` target-domain document, the native
`<stem>.namelist.input` and its stock-WRF twin
`<stem>.stock.namelist.input` beside them. The closing block prints
`woof go <experiment.toml>` to fetch,
prepare, forecast and render. Set up `WPS_GEOG` as described by
`woof setup`; preparation receipts are read automatically.
`--ladder` picks the
depth (`12`, `12-3`, `12-3-1`, `12-3-1-0.5`, or `auto`; the
single-domain `12` ladder supports the same checkpoint transport as the
nested ladders; new configurations checkpoint hourly or at the end of a
shorter run -- see section 7); `--vram-gib N` covers cards between the
named tiers (`--card 12gb|16gb|24gb|32gb`). How the sizing model works and where each platform's
envelope factor comes from: [HARDWARE.md](HARDWARE.md).

The presets are shortcuts, not the whole product. `--root-dx KM` and
`--chain R1,R2,...` build any ladder from an arbitrary root spacing and
a chain of integer refinement ratios, sized by the same estimator fit
loop and validated by the same `woof check`:

```bash
# 3 km -> 750 m
woof domain --point=35.3,-97.5 --card 24gb \
    --root-dx 3 --chain 4 \
    --cycle 1999-05-03T12 --hours 6 --out configs/myarea.toml
# 3 km -> 1 km -> 333 m -> 111 m
woof domain --point=35.3,-97.5 --card 32gb \
    --root-dx 3 --chain 3,3,3 \
    --cycle 1999-05-03T12 --hours 6 --out configs/myarea.toml
```

`--cycle` and `--out` are required on every `woof domain` call -- the
two lines above used to end in `...`, which hid them and made both
examples an argparse error when pasted. `--cycle latest` resolves the
newest complete cycle for `--source gfs`/`hrrr`; ERA5 and reanalysis
cases name the date they want.

Root `dt` follows the same convention at any spacing (5 s per km, 2.5
in the tropics) and is carried exactly, including half seconds, through
WRF's rational clock keys. When any domain lands below 1 km with a 1-D
PBL scheme active the wizard prints a **gray-zone advisory** -- not a
refusal -- into both the file and stdout: below that spacing the largest
boundary-layer eddies are partly resolved by the dynamics while the PBL
scheme parameterizes them as if they were not, and the proper tool at
those scales is a 3-D turbulence closure -- SASE, `bl_pbl_physics =
900`, which is implemented and selectable but experimental and not
WRF-verified. Read [PHYSICS.md](PHYSICS.md) before using it.

The wizard works worldwide: it picks the projection from your point's
latitude -- Mercator below 25 degrees, hemisphere-correct Lambert
conformal from 25 to 60, polar stereographic above 60, in either
hemisphere -- and `--projection` overrides the choice.
Antimeridian-crossing footprints are handled. It still refuses what
the pipeline cannot stand behind: domains containing or touching a
pole, and forcing footprints wider than 180 degrees of longitude --
though the 180-degree limit is now a bound the sizer respects rather
than a wall it walks into, so a card big enough to want a wider box
gets the largest domain one source crop can actually feed, and a line
on stderr saying the source, not the card, is what stopped it. The
new projections (Mercator, polar stereographic, southern-hemisphere
Lambert) have been checked against the WRF projection oracle and exercised
in finite-state smoke runs. They have no matched-run comparison against WRF;
the smoke runs check execution, not accuracy
([VERIFICATION.md](VERIFICATION.md)).

## 3. Get data

Two routes, accurately distinguished (details and disk sizing:
[DATA.md](DATA.md)):

- **ERA5 (the GPU forecast route).** `woof fetch --source era5 --era5-provider arco --cycle 1999-05-03T00 --hours 24 --area 30,-105,42,-90 --out data/era5-arco`
  downloads, validates and publishes `era5-combined.nc` with no account.
  The default CDS provider instead emits the two-part CDS request template;
  you retrieve it with your own Copernicus account, then validate with
  `woof fetch --source era5 --validate FILE...` (seconds, catches a wrong
  retrieval before anything expensive).
- **GFS / HRRR (the native preprocessor route, measured: 14.5 s).**

```bash
woof fetch --source gfs --cycle latest --hours 6 \
  --point 35.2,-97.4 --radius-km 350 --p-top-pa 5000 --out data/gfs-latest
```

resolved the newest complete cycle (2026-09-27 12Z), downloaded three
~146 KB subset files, and wrote a SHA-256 manifest and the series
inventory in 14.5 s. `--p-top-pa 5000` is the 50 hPa model top every
`woof domain` config carries: it adds the 70 and 50 hPa levels to the
100 hPa ladder the fetch takes without it, and preparation refuses a
folder that stops short of the config's top. The fetch line
`woof domain --explain` prints carries it already.
Every fetch is resumable; re-running with a different area or cycle
into the same directory refuses with the exact difference rather than
silently keeping the old files.

GFS/HRRR data feeds `rw-wps`, the native initialization front door
(measured: 44.7 s for a two-domain d01+d02 build -- 30-arcsec static
fields, decode, and initialization on the deterministic Rust CPU
backend), which emits `wrfinput_d01..dNN` + `wrfbdy_d01`. Those files
drive unchanged stock WRF ([WRF-INTEROP.md](WRF-INTEROP.md)) and the
downscale tool. The config-driven `woof run` loop in section 5 below
runs from ERA5 only -- **GFS and HRRR reach the GPU through a different
runner, documented in section 3a.**

## 3a. GFS -> GPU forecast: the complete route

**The short version is one command.** `woof go` runs the whole chain
below -- authority, fetch, front-door manifest, rw-wps, forecast,
render -- carrying each stage's digests to the next instead of asking
you to copy them, and printing a heartbeat while the long stages work:

```bash
woof domain --point=35.3,-97.5 --card 24gb --ladder 12 \
  --source gfs --cycle latest --hours 6 \
  --physics-profile morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1 \
  --out configs/myarea.toml
woof go configs/myarea.toml
```

Bare `woof domain` at a terminal asks four questions and supplies both
of those flags for you, so its emitted config is a `woof go` config.
`woof go CONFIG.toml --dry-run` validates the route and prints a launch
command without running it. The long form below explains the GFS stages;
read it when a stage refuses, or when you want to change one.

Downloads are managed automatically under the forecast workspace. Running a
different cycle, source or area selects a separate cache; repeating a matching
request reuses verified inputs. Existing data are preserved, including the flat
data folders created by older versions. You do not need to choose another folder
between forecasts. Use `--data-dir DIR` only when deliberately managing a specific
download directory yourself.

Configs that declare `[case_data]` use the same `woof go CONFIG.toml`
command. Their named inputs go directly through preparation and the
experiment runner, without fetching replacements. The default draws pictures
as frames become available and finishes the remaining products after the
forecast. Use `--products none` for a forecast without pictures, or a product
list such as `--products t2,refl` to select them. A `--geog-root` override is
honored and recorded; otherwise the config's geography root applies.
`--data-dir` is for download routes and is refused for declared inputs;
change `[case_data].forcing` to use a different set of files. The lower-level
`woof run` command remains available, and an ordinary experiment run plan
still produces no pictures unless its `render_products` option requests them.

The same `woof go CONFIG.toml` command selects the registered native
preparation route for HRRR and supported mapped sources such as ICON-EU.
It also accepts nested configs; the config selects the appropriate tree
runner. Use `woof go CONFIG.toml --dry-run` to check the route before
fetching. Native routes carry manifest digests internally. Their default
output reports stages and warnings; `--explain` shows detailed output, and
`launch.log` in the run folder preserves stage diagnostics. Sources without
an executable automatic route are refused before downloading, with their
missing capability available under `--explain`. Memory admission prices the forecast before launch and reports whether the
source's preparation phase is priced; the actual preparation and runtime
allocation checks remain in force.

When you already have input files, `woof prep` and `rw-wps` report
preparation stages and the selected backend while they run. They save full
diagnostics to a uniquely named `*-prep-*.log` beside the prepared directory.
Successful preparation prints a `woof sim` command with the configuration
paths filled in; you do not need to copy checksums. That command carries
`--render-products all`, so it draws every product from each output frame
as it lands while the forecast runs; name fewer products there, or `none`
for no pictures. A failure prints its
reason and the log path. Add `--explain` to also show the full diagnostic
output on the terminal. Inventory and `--dry-run` commands do not create logs.
Companion WRF export progress explicitly says whether files were written,
not requested, or not produced; optional export refusal does not discard a
successful native preparation.

**The order matters**: the runner binds the experiment config into the
prepared cache, so materializing the physics *after* preprocessing
means preprocessing again.

Three first-time-user pilots ran this route on rented Linux 4090s and
4070s on 2026-07-30; the ordering, the flags, and the timings are
theirs.

```bash
# 1. Size the domain.  --physics-profile is OPTIONAL: it binds the
#    config to a shipped suite that every later stage then enforces
#    switch for switch.  Without it you get the product default suite,
#    which runs as written too -- the receipts state its verification
#    status ("supported, not yet WRF-verified") and the run continues.
#    --explain is what makes the wizard print the separate fetch line
#    step 3 runs; without it the wizard ends with the one `woof go`
#    line that runs every step itself.  It also prints what the profile
#    you chose actually runs.  It is a modifier on a COMPLETE domain
#    command, not a query: `woof domain --explain` on its own is a
#    usage error.  --data-dir points that fetch line at data/myarea.
woof domain --point=35.3,-97.5 --card 24gb --ladder 12     --source gfs --cycle latest --hours 6     --physics-profile morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1     --out configs/myarea.toml --data-dir data/myarea --explain

# 2. Materialize the exact physics authority.  BEFORE rw-wps, not after.
#    The profile SUPPLIES every physics key your config is silent about.
#    It never replaces one your config states: where the two disagree,
#    step 2 refuses and names each key, your value, the profile's value,
#    and the remedy that fits -- the shipped profile your config matches
#    when it matches one, or omitting --physics-profile, which publishes
#    your config's own suite unchanged with its verification status in
#    the receipt.  Nothing is rewritten behind you.
python -m woof.prepared_single_domain_forecast --materialize-authorities     --source gfs     --base-experiment-config configs/myarea.toml     --base-wps-namelist configs/myarea.namelist.wps     --physics-profile morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1     --output-directory work/myarea-authority

# 3. Fetch.  The fetched directory is a front door (inputs.txt,
#    prep-command.txt, SHA256SUMS, fetch-manifest.json), and the input
#    manifest the next step needs is NOT yours to author: rw-wps /
#    `woof prep --source gfs` authors and digest-binds it from this
#    directory when --source-manifest is omitted.  --bridge is optional
#    there too: omitted, it resolves the built gfs_grib2_bridge this
#    install has (a checkout's own build, then libexec/, then the
#    ~/.woof/bridges that `woof setup` stages into) -- the same
#    resolver `woof go` uses.  `woof doctor` names the one it found.
#    (`woof fetch --author-front-door-manifest` still authors the
#    manifest standalone -- a tail series, or a different
#    namelist/config pairing -- and prints the bound rw-wps line.)
#    --p-top-pa 5000 is the config's own 50 hPa model top: it fetches
#    the 70 and 50 hPa levels above the 100 hPa ladder the fetch takes
#    without it, and rw-wps refuses a folder that stops short of the
#    top.  The fetch line step 1 printed already carries it, and the
#    area, cycle and folder below; paste that line.
woof fetch --source gfs --cycle <RESOLVED> --hours 6     --area=<THE BOX THE WIZARD PRINTED> --p-top-pa 5000 --out data/myarea

# 4. Run the front door (paste the line step 3 printed, plus these).
rw-wps ... --geog-root $GPUWM_CASE_DATA_ROOT/WPS_GEOG     --output-root out/myarea-init

# 5. Run the forecast.  rw-wps finishes by printing THIS command with
#    all three digests filled in -- copy it rather than retyping.  Its
#    --outdir is <output-root>-forecast, so with step 4's
#    --output-root out/myarea-init the printed line ends in
#    --outdir out/myarea-init-forecast.  Change it if you like; step 6
#    reads whatever you ran with.
python -m woof.prepared_single_domain_forecast     --source gfs --prepared-root out/myarea-init     --proof-sha256 <printed> --source-manifest-sha256 <printed>     --prepared-content-sha256 <printed>     --experiment-config work/myarea-authority/experiment.toml     --wps-namelist work/myarea-authority/namelist.wps     --physics-profile morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1     --io-mode history --outdir out/myarea-init-forecast

# 6. Render.  <frame> is one file under wrfout/, not the directory.
woof render out/myarea-init-forecast/wrfout/<frame> --out out/myarea-png
```

Measured on a Linux RTX 4070 12 GB (node 3, GFS 2026-07-29 18Z, single
domain 342x272x49 at 12 km):

| stage | wall |
|---|---:|
| `woof fetch` (3 files, 42.2 MB) | 50.3 s |
| `--materialize-authorities` | < 1 s |
| `rw-wps` (30-arcsec static + decode + init, 48 CPU cores) | 2 m 52 s |
| GPU forecast, 6 simulated hours, 2160 steps at dt 10 s | **523 s** |
| `woof render` (4 products, 1 frame) | 21.8 s |

**Multi-domain** products go to `python -m woof.prepared_domain_tree_forecast`
instead, which takes `--prepared-root` + `--preparation-receipt-sha256`.
Neither runner has a physics-profile whitelist: both run the suite your
config selects as written (the wizard's default suite included), and
`--physics-profile` is an optional assertion that the config is one of
the shipped suites. `rw-wps` names whichever runner applies to the
proof it just wrote, with the digests filled in.

**Watch progress** at `<outdir>/evidence/progress.json` (domain tree) or
`<outdir>/progress.json` (single domain), not the `run-progress.json`
that section 5's config-driven route writes.

## 4. Preflight (measured: 9.2 s)

```bash
woof check configs/myarea.toml
```

Two gates in one command: the input preflight (in the transcript, 19
checks -- real GRIB decode envelopes, level/temporal/spatial coverage,
geog tile coverage and hashes, physics table hashes) and the itemized
VRAM preflight (predicted 12.78 GiB peak envelope against a 27.25 GiB
measured budget in the transcript). `check` failing is the tool working:
it names the missing input or the memory shortfall and the remedy
before you spend GPU time.

A check of this machine also compiles and executes a tiny CUDA kernel
and a CuPy reduction from a fresh cache before memory admission. A
missing compiler or toolkit header stops the check with the matching
`woof doctor` remedy. `--budget-gib` remains an estimate for a declared
allocation budget and labels GPU readiness as not checked. The wizard
uses `--free-gib` for free VRAM before reserves, so its follow-up check
prices the same resident or streamed plan. Run ordinary `check` on the
forecast machine before launch. Host-memory admission reads
available physical RAM on both Windows and Linux.

## 5. Run (measured: 6 h forecast in 3.6 min)

```bash
woof run configs/myarea.toml --outdir out/myarea
```

The acceptance run: 6 simulated hours on a 250x200x49 12-km domain
with Morrison + RTE+RRTMGP + YSU + Noah + Kain-Fritsch at dt = 60 s
(the wizard default at the time of the transcript; it now emits
Thompson mp8 in the microphysics slot, Morrison stays selectable) --
361 steps in ~200 s of wall time (~0.55 s/step including output),
completed on the first attempt, ~6.3 GiB of device memory above the
desktop baseline, seven hourly wrfout frames of ~198 MB each.

While it runs:

- **Watch `run-progress.json`** in `--outdir` (schema
  `gpuwm.run-progress/v1`): status, model-elapsed seconds, outer step,
  newest durable output frame, newest checkpoint. It is rewritten
  atomically through the run. Do not watch a redirected stdout -- it is
  block-buffered and can stay empty until exit while the run is
  healthy. This filename belongs to the config-driven `woof run`
  route only; the `tools/` runners of section 3 write
  `<outdir>/evidence/progress.json` (domain tree) or
  `<outdir>/progress.json` (single domain).
- On failure the supervisor writes `failure-capsule.json` beside it
  with the exception, the step, and the state needed to report or
  resume.

## 6. Render (measured: 2.6 s for 16 PNGs)

```bash
woof render out/myarea/wrfout_d01_* --out out/myarea/png
```

With the `tools/rustwx` build from step 1, this runs the production
Rusty Weather engine by default.  Its vendored catalog carries 324
entries; 151 of them are implicit-render candidates the runtime
lister evaluates against every file (the rest are explicit-opt-in
ensemble/probabilistic families): reflectivity composite and 1 km,
the 2 m temperature/dewpoint/RH families with 10 m wind variants,
MSLP + winds, PWAT, cloud cover, the 200/250/300/500/700/850 mb chart
families (height/temperature/dewpoint/RH/absolute vorticity with
winds), SB/ML/MU CAPE and CIN, SRH, bulk shear, STP, the heavy ECAPE
family (`--heavy`), and multi-hour windowed accumulations (run-total
QPF, wind/UH maxima, ...) on whole-hour multi-frame runs.  Whatever
your output fields prove out renders -- measured on the committed
3 km UH-smoke case: 58/58 on a single frame and 238 renders with 0
failures across its four-frame store (receipt transcripts retained in
the development tree under `evidence/render-receipts/`) -- and
`woof render --list-products FILE` prints the whole catalog with the
per-file verdict and the exact field-level reason anything is
unavailable.
Charts draw coast/state/county basemaps, and sub-hourly output
cadences are stamped exactly (`valid_..._lead_003h30m00s`) in filename
and subtitle.  The first line of output names the engine in use.

Every filename carries the domain and its resolution
(`..._d02-3km_composite_reflectivity_...`; sub-kilometre nests read as
`_d05-111m_`), so several nests of one run render into one directory
without colliding, and the plot subtitle carries the same spacing as
`Δx 3 km`.  Plots are labelled **WOOF**; pass `--source-label` when
rendering wrfout files this model did not produce, so the sheet does
not claim them.

Without that build, `woof render` refuses and names `woof
fetch-bridges`: weather-field product plots come from `rw_wrfbatch`, so
nothing degrades to a second engine on its own. `--engine matplotlib`
asks for the workaround by name and prints a `WORKAROUND:` line every
run; it draws five products per frame -- composite reflectivity (NWS
color scale), 2 m temperature, 10 m wind speed and barbs, accumulated
precipitation, and TOA outgoing longwave as synthetic infrared -- via
the `wrf-rust` package (`pip install 'recast-woof[render]'` if you skipped the
extra; the error message names it). The transcript below predates the
OLR panel and measured the other four: 4 files x 4 products took 2.6 s.

To compare two runs product-by-product (a rerun, a physics variant, a
CPU WRF twin), render each into its own directory and compose labeled
side-by-side sheets. These are qualitative comparisons. Roundoff-level
differences can grow during convection, including between different WRF builds;
displaced cells alone establish neither a port defect nor statistical agreement.
An ensemble consistency test estimates whether differences fit the reference
model's sampled variability. The scoped WRF v4.7.1 ensemble results and their
limits are in [VERIFICATION.md](VERIFICATION.md); they do not qualify an arbitrary
pair of runs.

```bash
woof render --pair out/runA/png out/runB/png --out out/compare
```

## 7. Checkpoint and resume (measured: 65.6 s + 46.1 s)

**Which route this is.** `woof run` is the `[case_data]` route of
section 3. It and both prepared routes, single-domain and multi-domain,
support checkpoints. New wizard configurations use hourly checkpoints,
or the end of a shorter run, with a compatible native event clock.
An explicit `restart_interval_s = 0` disables checkpoint writing.
Resume requires a complete valid checkpoint and matching original inputs;
an output or log folder by itself is insufficient. Prepared runs retain
their prepared bundle and configuration for the resumed launch.

```bash
# a 2 h leg writing checkpoints every simulated hour
woof run configs/myarea.toml --outdir out/myarea    # restart_interval_s = 3600
# ... later, extend run_seconds in the config, then:
woof resume configs/myarea.toml --outdir out/myarea
```

In the transcript the 2 h leg took 65.6 s and wrote 2 checkpoints;
after extending the config to 3 h, `woof resume` found the newest
valid checkpoint set, verified its identity against the config
(fingerprints, physics identity, boundary-clock semantics -- mismatches
refuse loudly), and completed the third hour in 46.1 s. Torn or
truncated checkpoint sets are skipped with printed reasons.

**What may change between the leg and the resume:** the forecast length
(`run_seconds`), the output/restart cadence (`history_interval_s`,
`restart_interval_s`) and each domain's history window
(`history_begin_s`, `history_end_s`) -- that is the whole tolerance, on
every route that checkpoints, and extending the run is its point. Everything else
(geometry, timestep, physics, nesting, prepared inputs) is trajectory
identity: change one and the restart is refused by name, in one
sentence, exit 2.

## 8. Where to go next

- Downscale an archived run to a finer nest:
  [DOWNSCALE.md](DOWNSCALE.md) (write parent history at 15-minute
  cadence if you plan to).
- Drive unchanged stock WRF from the same preprocessor:
  [WRF-INTEROP.md](WRF-INTEROP.md).
- Import an existing WRF namelist pair:
  `woof import-namelist namelist.wps namelist.input` -- emits an
  experiment TOML plus an explicit substitution report of every option
  it mapped or refused.
- Read what the evidence measures: [VERIFICATION.md](VERIFICATION.md)
  separates code verification against WRF from limited comparisons with
  observations. Neither a matching picture nor a completed run establishes
  general forecast accuracy.
