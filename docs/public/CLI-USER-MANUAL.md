# Command-line user manual (engine 2.7)

WOOF's `woof` command creates configurations, acquires forcing, prepares model inputs, integrates forecasts, and renders saved history. You can run the complete workflow with `go` or use each stage independently.

This manual describes the 2.7 interface. Check your executing version with `woof version`; the command makes no network request unless `--check-pypi` is given. For the keyboard-and-mouse interface, see the [terminal workspace user manual](TUI-USER-MANUAL.md).

## Contents

1. [Command conventions and prerequisites](#1-command-conventions-and-prerequisites)
2. [Install and check the environment](#2-install-and-check-the-environment)
3. [Create and run a first forecast](#3-create-and-run-a-first-forecast)
4. [Read and revise a configuration](#4-read-and-revise-a-configuration)
5. [Select sources, dates, and forcing](#5-select-sources-dates-and-forcing)
6. [Check memory, fit, and tile](#6-check-memory-fit-and-tile)
7. [Create research and case workspaces](#7-create-research-and-case-workspaces)
8. [Prepare supplied data and run a prepared bundle](#8-prepare-supplied-data-and-run-a-prepared-bundle)
9. [Use WRF and WPS inputs](#9-use-wrf-and-wps-inputs)
10. [Downscale archived parent history](#10-downscale-archived-parent-history)
11. [Render existing forecast output](#11-render-existing-forecast-output)
12. [Continue or branch a forecast](#12-continue-or-branch-a-forecast)
13. [Read results, logs, and exit status](#13-read-results-logs-and-exit-status)
14. [Control durable jobs on Linux nodes](#14-control-durable-jobs-on-linux-nodes)
15. [Use structured interfaces in scripts](#15-use-structured-interfaces-in-scripts)
16. [Command reference](#16-command-reference)
17. [Troubleshooting](#17-troubleshooting)

## 1. Command conventions and prerequisites

The examples use `woof` from your activated Python environment. `python -m woof.cli` is the equivalent spelling when you need to select an interpreter explicitly.

```text
woof --help
woof --help-all
woof go --help
```

The short help emphasizes common tasks; `--help-all` lists every command. Subcommands provide their own help. Most operational commands support `--explain` for full evidence and detailed reasons. Use it when a concise summary is insufficient.

**Replace uppercase operands before running a template.** For example, `CONFIG.toml`, `PREPARED_DIR`, `JOB_ID`, and `YYYY-MM-DDTHH` denote your actual paths, identifiers, and dates. Lowercase paths such as `configs/first.toml` are proposed output names; choose a new name if one already exists. Quote paths containing spaces. The one-line examples work in PowerShell and POSIX shells unless a block names a specific shell.

| Task | Main prerequisites |
|---|---|
| Create/edit a configuration | Python environment; a measured GPU or declared planning capacity for sizing. |
| Fetch forcing | Network access or an existing local acquisition; any source-specific account/permissions. |
| Prepare inputs | Compatible source files, configuration and namelists, static geography, CPU RAM and disk; supported CPU preparation does not require a GPU. |
| Integrate locally | Compatible NVIDIA GPU, driver and CuPy, admitted GPU/host memory, complete model inputs. There is no CPU forecast fallback. |
| Render | Actual saved history and the native renderer with its map assets; no new model integration is required. |
| Use a node | Existing Linux WOOF installation and data, local OpenSSH, and working authentication/known-host configuration. |

Configuration admission and a successful forecast do not establish forecast skill. Research recipes expose questions and comparisons to evaluate; they are not validated forecasts of the named phenomenon.

## 2. Install and check the environment

### Published Python distribution

Use Python 3.11 or newer. Create an environment and activate it:

POSIX:

```sh
python3 -m venv .venv
. .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Choose the extra matching the CUDA installation on the forecast machine. These package-index commands select an available published release; verify its version after installation.

```text
python -m pip install "recast-woof[all-cu12]"
```

For a CUDA-13-only installation, use `recast-woof[all-cu13]` instead. For a machine used only for CPU preparation or remote control, install `woof` without a GPU extra. `woof doctor` reports the actual environment and the appropriate remedies; changing a package extra does not install an NVIDIA driver.

Then stage the release's native tools and reference tables:

```text
woof version
woof setup
woof doctor
```

`setup` verifies staged artifacts and reuses valid existing copies. To include static geography, use `woof setup --with-geog`, or run `woof fetch-geog` separately. Geography requires tens of gigabytes unpacked; use the command's current size estimate and allow space for forcing, preparations, history, checkpoints, and pictures.

For supplied offline bridge/table artifacts, `woof setup --from ARTIFACT_DIR` performs the same verification. Geography has its own acquisition options. Match artifacts to the installed release; arbitrary native executables are not interchangeable with its verified tools.

### Developer checkout

From the checkout root:

```sh
bash install.sh --cuda 12
. .venv/bin/activate
```

Or in PowerShell:

```powershell
.\install.ps1 -Cuda 12
.\.venv\Scripts\Activate.ps1
```

Use `13` when appropriate, or omit the CUDA option to use detection. The scripts install the checkout's matching `recast-woof-data` companion before the main editable package, stage external tables, and build all seventeen native workspaces: the GRIB bridges (`tools/grib1_bridge`), the renderer (`tools/rustwx`), the Zarr reader (`tools/zarr_bridge`), the mapped decode engine (`tools/rw_wps`), the dealiasing library (`tools/region_global_dealias`), hydro payload (`tools/nextgen_hydrofabric`), hydro forcing (`tools/nextgen_forcing`), gauge verification (`tools/nextgen_verify`) and hydro preparation (`tools/hydro_world`, `tools/hydro_world_params`, `tools/nextgen_reservoir_prepare`, `tools/hydro_world_init`) and ocean preparation (`tools/ocean_prepare`, `tools/rw-bathy`) and the wind-turbine table decoder (`tools/wind-table`) and hydro output (`tools/nextgen_output`) and the Icechunk store writer (`tools/rw_icechunk`). The hydro and ocean workspaces fetch their locked registry dependencies before the offline build. Missing Rust prompts for rustup consent; `--yes` / `-Yes` supplies that consent. The scripts make `.venv` on free-threaded Python 3.14t (a `python3.14t` on PATH, or one uv finds or installs), because multi-card `--devices` forecasts run about twice as fast on it; `--python` / `-Python` picks another interpreter, and when no 3.14t can be had the install goes on with a warning that names the cost. `--no-render` / `-NoRender` skips the renderer build, and `--no-fetch-tables` / `-NoFetchTables` defers the external table download.


The release's Linux x86-64 platform wheel targets manylinux 2.28. Use the artifact appropriate to your operating system and architecture, and verify the installed environment with Doctor. See [installation details](../install.md) for distribution-specific setup.

## 3. Create and run a first forecast

This example creates one 12 km domain centered near `35.3,-97.5` for a three-hour GFS forecast. It needs a usable local GPU for measured sizing and integration, network access for `latest` and acquisition, and staged geography. It is a workflow example, not a promised fit or weather-skill result.

```text
woof domain --point "35.3,-97.5" --source gfs --cycle latest --hours 3 --ladder 12 --out configs/first.toml
woof check configs/first.toml
woof go configs/first.toml --outdir runs/first --products t2,refl --dry-run
woof go configs/first.toml --outdir runs/first --products t2,refl
```

`domain` writes the TOML and companion authorities, including its WPS namelist, and prints the actual geometry and next steps. `check` evaluates configuration and resource readiness. `go --dry-run` validates the route and shows the launch without fetching or integrating. Removing `--dry-run` starts the chain.

On a fresh install, `domain` still exits 0 when its memory checks pass and only static geography is missing. It names the missing datasets and prints `woof fetch-geog --config CONFIG.toml` for the file it created. Stage that geography before launching. A failing memory check still makes configuration creation exit nonzero.

Bare `woof domain` opens guided questions. Specify `--source` in scripts: its default need not be GFS. For an 8 GiB planning target, add `--vram-gib 8`; this declares capacity rather than measuring current free memory. Check again on the machine that will run it.

Common variations:

| Request | Creation option |
|---|---|
| Preserve a study footprint | `--polygon study-area.geojson` instead of `--point`. |
| One 12 km root and a 3 km nest | `--ladder 12-3`. |
| Custom root spacing and refinement | `--root-dx 3 --chain 4` for 3 km → 750 m. |
| Set vertical detail | `--nz N`. |
| Set root or nest history cadence | `--history-interval SECONDS`, `--nest-history-interval SECONDS`. |
| Allow planned tile streaming | `--tiles auto`. |
| Forecast without pictures | `go --products none`. |

Changing resolution, extent, levels, nests, duration, or physics changes resource requirements. A point-based fit can change coverage. A polygon-based fit retains the required footprint or refuses it. Read the emitted geometry before downloading data.

By default, `go` places each launch in a fresh `run-...` directory under its output parent. Repeating the same saved configuration therefore creates a separate run. Existing explicit run directories receive stricter checks; use a new ordinary output parent when you want a clearly separate experiment.

## 4. Read and revise a configuration

Use a complete generated or imported TOML as your starting point. An experiment contains an `[experiment]` section and one or more `[[domain]]` sections, with optional data, output, tracking, scenario, and tiling settings. The actual file is the authority; a preset name is a starting suite or assertion, not a substitute for its settings.

Before launch, review:

- Initialization time, forecast duration, source, and cycle-relative start lead.
- Domain dimensions, spacing, projection, nest placement, and vertical grid.
- Physics and any explicit scientific acknowledgements.
- History and restart cadence, including compatibility with each domain's exact time step.
- Forcing, geography, namelist, preparation, and output paths.
- Following/lifecycle policies and the coverage needed throughout the run.

Save a new configuration for a new study and retain its companion files. Relative paths can change meaning when you move a TOML to another folder. Review them after moving or copying it.


A prepared bundle and a checkpoint are bound to particular inputs and configuration settings. Do not edit a receipt or checksum to make an old bundle accept a new experiment. Prepare matching inputs, or use a documented permitted restart/branch change.

## 5. Select sources, dates, and forcing

Inspect source capability rather than inferring it from a name:

```text
woof sources
woof sources gfs --explain
woof sources hrrr --json
woof prep --list-sources
woof prep --show-source era5
woof run-plan --physics-profiles
```

The registry distinguishes acquisition, decoding, preparation, and simulation routes. A source can support one stage without an end-to-end forecast route: a row with an acquisition route but no preparation runner is fetchable and not runnable, and `woof sources <id> --explain` says which stages a given row declares. Regional products have coverage limits, and source physics/cadence requirements still apply to aliases.

### Dates and forecast leads

Use UTC cycles such as `YYYY-MM-DDTHH`. `latest` may access the network. Public forecast sources are checked through their acquisition rules; delayed reanalyses may use a publication estimate instead of an object probe. An archive start date is not a guarantee that every requested file exists.

`--forecast-start-hour K` begins at lead `K` from a cycle; `--hours N` remains the length of the requested window. Thus `K=12`, `N=6` requires source leads 12 through 18 and initializes from a forecast, not the cycle's analysis. Review both the cycle and the resulting model start time.

Go downloads missing ERA5 using the saved `[fetch]` recipe. `woof domain` picks the provider: Copernicus CDS when a CDS key is present (`~/.cdsapirc`, `CDSAPI_RC`, or `CDSAPI_URL`/`CDSAPI_KEY`), otherwise Google's public hourly ARCO archive, which needs no key, and it prints which one it chose and why; `--era5-provider` names one explicitly. ERA5 is published about a week behind real time, so with no key a bare `woof domain` for a window the ARCO archive does not hold yet runs on GFS instead and says so (`--source era5` keeps ERA5 and is refused for such a window, naming the newest cycle the archive holds). A `[fetch]` table that names no provider uses the CDS. Set `era5_provider = "arco"` in `[fetch]` to use Google's public hourly ARCO archive without a key, and declare `era5-combined.nc` in `[case_data].forcing`; CDS produces `era5-combined.grib` in the same fetch output directory. The Google adapter reads all 37 pressure levels and the required surface and soil fields into native NetCDF for the existing preparation path. Long windows, upper atmospheric coverage, and regional bounds must match the source's available data.

For a separate acquisition, use `woof fetch --source era5 --era5-provider cds --retrieve` or `woof fetch --source era5 --era5-provider arco`, adding the desired `--cycle`, `--hours`, `--area`, `--cadence`, and `--out`. Without `--retrieve` or the ARCO provider, the legacy CDS request-template command remains available.

### Acquire data separately

For ordinary Go use, leave download caching to the launch route unless you need a specific existing acquisition. For a manual pipeline, add `--explain` to the `domain` command that creates the configuration, and `--data-dir DIR` to choose the download folder, then copy the acquisition command it prints, including its area and cadence. Where `woof go` can fetch the source itself, `domain` without `--explain` ends with that one `woof go` line and prints no separate acquisition command. The printed command already accounts for the fitted domain and required source margin.

The general fetch template is:

```text
woof fetch --source SOURCE --cycle YYYY-MM-DDTHH --hours HOURS --area SOUTH,WEST,NORTH,EAST --out data/case-forcing
```

Replace every uppercase operand with values admitted for your study. `--area` is a source-data crop, not the forecast grid. Cropping too tightly can omit required boundary or water/terrain donors.

Use a request-specific directory. Existing complete matching files are reused. Reusing a directory for different source requests can be refused; `--force-refetch` deliberately moves previous files aside and downloads again. It is not the first remedy for an unexplained validation error.

### ERA5 lake temperatures and lake ice

ERA5 acquisitions (CDS and ARCO) include lake mixed-layer temperature, lake
ice temperature and lake ice depth at the requested forcing times and source
grid. When a lake has no usable SST analysis of its own, WOOF uses the
explicit lake-model water field. This also covers small inland lakes that
occupy no majority-water ERA5 grid cell. The provider is reported as
`lake_model_ice_free_water`; it is a modelled lake state, not an observation
of that individual lake. Existing fields and source-file hashes remain
part of the forcing provenance.

The same lake model gives the lake its ice. A lake cell's ice cover is the
bilinear weight of the ERA5 points around it whose lake ice depth is above
zero, and its skin temperature is their lake ice temperature. Where that cover
is larger than ERA5's sea-ice cover, it replaces it, so a frozen lake starts
frozen instead of as open water near 273 K. With WRF's default
`fractional_seaice = 0`, a cell is ice at a cover of 0.5 or more, the same
threshold as sea ice. Like real.exe with a GFS lake's ice, the ice is carried
as the cell's sea-ice fraction (`XICE`), because the lake model is off by
default. The preparation log counts the cells that start as lake ice, and the
receipt's `lake_ice_mapping` lists the rule and the cells where a donor had no
usable depth or ice temperature. Those cells keep ERA5's sea-ice cover.

Open-water lake temperature needs every contributing bilinear donor to have
zero lake-ice depth. A one-sided numerical-zero convention admits only tiny
negative source roundoff within `2.220446049250313e-16 m`, records its use and
retains the raw values. A lake cell with a partly frozen or unknown donor that
the lake model does not cover with ice keeps the component skin temperature,
and the preparation log names those cells. Old forcing files that lack the
lake fields remain readable and use ERA5's sea-ice cover alone. Acquire the
current ERA5 inventory into a new output directory to preserve an older run's
inputs.

### Supply an explicit preparation donor

If a supported preparation requires a missing supplemental role, supply genuine compatible data with that role. For HRRR, Go accepts a PMSL donor within the chosen `--data-dir`:

```text
woof go CONFIG.toml --data-dir data/hrrr-case --supplement PMSL=data/hrrr-case/pmsl.grib2 --dry-run
```

Repeat `--supplement ROLE=PATH` for multiple files. Review, then remove `--dry-run` to launch. The donor must satisfy the actual time, grid, metadata, and composition contract; the command binds its bytes. Mean sea-level pressure and surface pressure are different quantities. Do not invent a field or relabel an unrelated file to satisfy a missing-role message.

## 6. Check memory, fit, and tile

### Measure or declare the budget

```text
woof check CONFIG.toml
woof check CONFIG.toml --explain
woof check CONFIG.toml --json
```

The concise report gives the outcome and actionable diagnostics. `--explain` adds detailed estimates and evidence; JSON is available for programs. GPU capacity, currently free VRAM, allocation budget, forecast peak, initialization requirements, and host RAM are different measurements. A large nominal card can be busy. Read measured physical capacity separately from any declared target capacity, and requested allocation budget separately from the effective budget.

When the memory checks pass but the configuration needs unstaged geography, `check` exits 6 and names the missing datasets with the remedy `woof fetch-geog --config CONFIG.toml`. A failing memory check keeps its own exit code. `go`, including `--dry-run`, refuses missing geography before downloading forcing data.

An estimate can report **INCOMPLETE** when only some gates could be evaluated. Exit zero then means those evaluated gates fit; it does not qualify an absent GPU or unevaluated requirements. An unpriced source-decode RAM requirement is unknown, not zero.

To estimate for a machine you are not using, declare both free memory and its physical ceiling. This example assumes an 8 GiB card with 7 GiB free:

```text
woof check CONFIG.toml --free-gib 7 --vram-gib 8
```

Alternatively, `--budget-gib` declares allocation budget after the allocation reserve; it is mutually exclusive with `--free-gib`. `--vram-gib` is a ceiling, not a source of free memory. Such an estimate does not qualify the target GPU. Without an available device, Check can report required estimates while still failing GPU readiness.

For an explicit device-allocation check:

```text
woof check CONFIG.toml --alloc
```

This allocates the persistent device state and takes zero forecast steps. It requires a GPU and can use substantial memory; run it when that GPU is available. It is not a stability or forecast-skill test.

`--card` takes any spelling that carries a capacity: a tier (`12gb`, `16gb`, `24gb`, `32gb`), a size written into the name (`8gb`, `"RTX 3080 10GB"`), or a model with a recorded size (`"RTX 3080"`, `rtx3080`, `"5070 Ti"`); `--vram-gib 8` beside a card is the capacity and the card is a label. Only a name with no capacity in it is refused. Research's separate `--hardware-class 8` is a profile, not a declaration of capacity.

### Fit an editable starter

Preview a fitted copy, then add `--write` to create it:

```text
woof domain-fit starter.toml --point "35.3,-97.5" --vram-gib 8 --out configs/fitted.toml
woof domain-fit starter.toml --point "35.3,-97.5" --vram-gib 8 --out configs/fitted.toml --write
```

Use `--polygon study-area.geojson` instead of a point when the full area is required. Omit the capacity option to measure the local GPU. `--start-time` and `--hours` are explicit changes; otherwise the template's time and duration remain. `--source` is needed when the starter has no fetch-source declaration.

Fitting retains scientific settings and reports the new layout. It may refuse a footprint or research minimum area that cannot fit. Inspect the result before preparation.

`--write` creates the configuration, the fit receipt, and every file the configuration's input route reads beside it: the `namelist.wps`, and on the regional route the two namelists and the target-domain document the run reads. They are rendered from the fitted layout, not copied, and each one is named in the printed output and in the receipt's `route_companions`. An existing file of any of those names is preserved and the fit refuses.

### Make a streaming copy

```text
woof domain-tiles CONFIG.toml --out configs/streamed.toml --mode auto
woof domain-tiles CONFIG.toml --out configs/streamed.toml --mode auto --write
```

This preserves configured area, resolution, time, and physics, validates the planner's GPU/host-memory requirements, and creates a new configuration with every file its input route reads beside it, named in the receipt's `route_companions`. Use `on` to force planner-selected streaming. In TOML, the ordinary automatic setting is:

```toml
[tiles]
mode = "auto"
```

`auto` keeps a fitting resident domain resident. Streaming stores the domain in system RAM and processes tiles on the GPU; it consumes host memory and transfer time. Do not add pinned tile dimensions to `auto`; use the documented explicit controls only when you intend them. A host-RAM failure during forcing decode is a separate problem that GPU tiling cannot remove.

See [Tiles](TILES.md) for advanced controls. `woof stream PLAN.toml` is a different feature: it follows uploading forecast cycles through sealed forecast legs, not tile-based memory management.

## 7. Create research and case workspaces

### Research catalog

```text
woof research catalog --json
woof research hardware --json
woof research attributes --json
```

The packaged catalog contains 114 starting configurations organized into 38 questions and 12 families. Select the exact configuration ID from the catalog; its method, comparison, limitations, and required inputs matter.

```text
woof research create RECIPE_ID --point "35.3,-97.5" --source gfs --cycle YYYY-MM-DDTHH --hardware-class auto --out configs/study.toml
```

Replace the recipe ID and date. Creation writes a new configuration, every file that configuration's input route reads beside it, a plot selection, and a research receipt, and starts no forecast. It reports actual fitted geometry and memory admission. Omitted duration keeps the recipe's duration. Use `--hours`, `--nz`, or `--physics-profile` only for changes you intend to review.

For a declared 8 GiB target, use `--hardware-class 8 --vram-gib 8`. Without `--vram-gib`, hardware selection measures current capacity and free memory. The class chooses a resource profile; it does not create memory or validate the requested study. If a larger class loses the minimum study area, retry the same question with `auto` or the suggested smaller class and keep the true capacity.

Recipes requiring archived parents or existing scenario state cannot create those inputs from an ordinary analysis. Use the archive/downscale or existing-state workflow. Fresh controlled-scenario creation works from every source: each source's domain-tree preparation records the bubble as deferred, and the tree forecast applies it at the start and validates warm-bubble coverage.

Research creation uses packaged diagnostic metadata and does not need to execute a built renderer. The current selector subset has 40 members; actual files and complete time windows still decide availability. Read **Additional analysis required** in the command output and `further_analysis` in the catalog. Gusts, visibility, precipitation-type categories, DCAPE, and several near-surface derived quantities are not supplied by this history-rendering route. Related state fields provide context, not an equivalent diagnostic.

### Historical-case catalog

The installed package includes 300 historical cases, combining the original 200 with their newer revisions and 100 additional worldwide cases. Omit `--catalog` to use them:

```text
woof case-catalog list --limit 20
woof case-catalog list --query "storm"
woof case-catalog default --json
```

Use `--catalog` for another JSON, TOML, or supported ZIP catalog:

```text
woof case-catalog list --catalog cases.json --limit 20
woof case-catalog list --catalog cases.json --query "storm"
woof case-catalog preview CASE_ID --catalog cases.json --tier lower
woof case-catalog create CASE_ID --catalog cases.json --tier lower --out configs/case.toml --vram-gib 8
```

Replace `CASE_ID` with a listed ID. Tiers are `lower`, `recommended`, and `upper`; inspect what each actually changes. `--source-option` chooses an initialization ID listed by the case. Creation preserves provenance and refuses existing destinations. Programs can bind creation to reviewed catalog bytes with `--expected-catalog-sha256`.

Catalog contents are data, including when supplied in a ZIP. They are not executed as authoring scripts. Selecting a historical event does not establish its data availability or simulation fidelity.

## 8. Prepare supplied data and run a prepared bundle

| Stage | Command | Input | Output |
|---|---|---|---|
| Acquire | `fetch` | Source request | Verified source files and acquisition records. |
| Prepare | `prep` | Supplied source files and exact configuration/namelists | Prepared model inputs and completion/identity records. |
| Simulate | `sim` | Finished prepared bundle and its authorities | History, checkpoints when enabled, progress, and run report. |
| Plot | `render` | Actual saved history | Requested PNG products and rendering diagnostics. |

### GFS preparation from an existing acquisition

After fetching the GFS files into `data/case-forcing` using the correct generated request, author the preparation handoff:

```text
woof fetch --source gfs --author-front-door-manifest --out data/case-forcing --wps-namelist configs/case.namelist.wps --experiment-config configs/case.toml
```

With the existing acquisition and no new cycle/hours arguments, this authors the manifest and prints the complete preparation command with its bound inputs. Copy that command. Add `--dry-run` to validate its route arguments without preprocessing, then execute the reviewed command to prepare.

For reference, the GFS preparation template has this shape:

```text
woof prep --source gfs --gfs-series SERIES.tsv --cycle YYYY-MM-DD_HH:MM:SS --wps-namelist CONFIG.namelist.wps --experiment-config CONFIG.toml --source-manifest MANIFEST.json --source-manifest-sha256 MANIFEST_SHA256 --geog-root WPS_GEOG --output-root prepared/case --preprocess-backend cpu
```

Use the real series, WRF-form cycle, manifest, digest, and paths printed by your handoff. A GFS series is a tab-separated hour/path inventory; it is not the same operand as a raw GRIB file. `prep` downloads nothing. Its supported CPU backend still needs sufficient RAM and the native preprocessing tools.

Other sources have their own operands: ERA5 combined GRIB uses `--grib`/`--vtable`; mapped sources use their mapping/composition and ordered `--input` or `--input-list`; HRRR uses its source manifest and domain/namelist authorities. Read `woof prep --show-source SOURCE` and `woof prep --help`. A successful custom preparation does not automatically establish an admitted forecast route.

### Run the finished bundle

First inspect the exact runner command:

```text
woof sim prepared/case --experiment-config configs/case.toml --wps-namelist configs/case.namelist.wps --outdir runs/prepared-case --print-command
```

Then run it:

```text
woof sim prepared/case --experiment-config configs/case.toml --wps-namelist configs/case.namelist.wps --outdir runs/prepared-case
```

The WPS operand is required for a single-domain bundle and unused by a domain-tree runner. The default runner selection reads the bundle's format and domain count. Keep its complete contents and original authority files intact. A partial directory or changed input identity is refused, with one exception: a chained preparation that is still being produced (see PIPELINE-STAGES.md) is bound by its head, and the forecast waits at each boundary interval that is not prepared yet. A preparation whose producer failed, was stopped or went silent is refused.

Sim performs no fetching. Rendering is off unless requested. `--render-products t2,refl` draws those products from each committed history frame of every grid as it lands, while integration continues. The `woof sim` line `woof prep` prints carries `--render-products all`.

Go can also reuse an existing preparation:

```text
woof go configs/case.toml --prepared-root prepared/case --wps-namelist configs/case.namelist.wps --outdir runs/reused --products t2,refl --dry-run
```

Review and remove `--dry-run` to run. Existing preparation is validated; it is not silently regenerated to match changed settings.

## 9. Use WRF and WPS inputs

For an existing WRF `real.exe` directory:

```text
woof run --wrfinput WRF_INPUT_DIR --outdir runs/from-wrf
```

Supply `wrfinput_d0*`, `wrfbdy_d01`, and the producing `namelist.input`. For a WPS metgrid directory:

```text
woof run --met-em MET_EM_DIR --outdir runs/from-met-em
```

Supply `met_em.d0*.nc` and the producing `namelist.input`. When its eta levels are absent and you intend native eta initialization, the met_em route accepts `--vertical-grid native`. `--run-seconds` can shorten a directory-input run within its forcing coverage.

The directory's geometry, times, and physics evidence are checked. `run --wrfinput` and `run --met-em` use the producing namelist rather than a separate positional TOML. Inspect any radiation substitution before explicitly changing `--rrtmg-variant`.

To translate a WRF namelist pair into an editable experiment and review substitutions:

```text
woof import-namelist namelist.wps namelist.input
woof import-namelist namelist.wps namelist.input --output configs/imported.toml
```

Without `--output`, the command reports without writing the resolved TOML. An import can substitute an implementation or refuse unsupported settings; read its report. Translation alone neither acquires the declared input data nor qualifies every WRF option. Use the producing files, not a loosely similar namelist.

Which WRF physics options import, and why each other one is refused, is listed in `docs/public/WRF-OPTIONS.md`. That page is generated by putting every option WRF v4.7.1 declares through this command, so it says what the installed importer does.

For a configuration that already declares local inputs in `[case_data]`, the direct route is:

```text
woof check CONFIG.toml
woof run CONFIG.toml --outdir runs/direct-case
```

Unlike Go and Sim's default timestamped layout, direct `run` writes into the specified output directory. Use a new directory for an independent forecast. On multi-GPU hosts, direct `run` and `resume` require `--gpu-uuid GPU-UUID` to identify the physical device they lock; choose the intended device from the machine's inventory.

## 10. Downscale archived parent history

`downscale` creates an offline child forecast from an existing parent archive. It does not add a live feedback nest to an ongoing parent simulation.

You need parent history covering the full intended window, the selected parent domain, authoritative parent physics evidence, a valid child placement, and any required child-grid surface warm start. The parent archive's cadence limits the lateral boundary information available to the child.

For a WOOF parent with a restart and a real child-grid surface file, a point-derived review template is:

```text
woof downscale PARENT_HISTORY_DIR --parent-domain 1 --parent-restart PARENT_CHECKPOINT.npz --point "35.3,-97.5" --ratio 3 --auto-vram --hours 3 --max-boundary-interval-seconds 900 --child-surface-from CHILD_SURFACE.nc --out downscale-review --dry-run
```

Replace the parent and surface paths, location, duration, and acceptable cadence with values appropriate to the archive. The child must lie inside usable parent coverage. Use `--accept-parent-cadence` instead of a maximum interval only when you intend to accept the archive's cadence; the options are mutually exclusive.

For a stock-WRF parent, supply `--parent-namelist` and, when needed, `--parent-namelist-domain`. Its explicit-child route uses `--child-config`, `--ratio`, and 1-based `--i-parent-start`/`--j-parent-start`. The child-config operand is the standalone RunConfig shape required by this command, not an arbitrary multi-domain experiment.

Dry-run validates and writes the derived configuration/plan without integration. After review, run the command without `--dry-run` and choose a **different unused** `--out` directory; the planning directory already exists. `--child-size` or explicit capacity replaces automatic point sizing. Surface-physics children need a compatible child-grid land/soil warm start; reusing a parent-grid file or disabling physics merely to bypass the requirement changes the experiment.

## 11. Render existing forecast output

Rendering needs actual history; it does not create a forecast. Check available products against the files first:

```text
woof render WRFOUT_FILE --engine rust --list-products
woof render FRAME_00.nc FRAME_01.nc FRAME_02.nc --series --engine rust --list-products
```

Replace the file operands with real paths. The second form combines compatible files into a timeline. `--series` keeps distinct runs, domains, grids, and lifecycle episodes separate. Listing the global catalog without a file only tells you names; it does not establish that your history can supply them.

Render a compact selection:

```text
woof render FRAME_00.nc FRAME_01.nc FRAME_02.nc --series --engine rust --products t2,refl,wind10 --size 1200x900 --out pictures
```

For hourly files forming a sufficient rain window, request `qpf_1h` or `qpf_6h`. A forecast directory containing separate hourly files must be supplied as actual file operands; the TUI's Plot history guide can discover and review them for you.

POSIX example for a known wrfout directory:

```sh
woof render runs/case/wrfout/wrfout_d01_* --series --products t2,refl --out pictures
```

PowerShell example:

```powershell
$historyFiles = Get-ChildItem -LiteralPath 'runs/case/wrfout' -File -Filter 'wrfout_d01_*' | Sort-Object Name | Select-Object -ExpandProperty FullName
woof render @historyFiles --series --products t2,refl --out pictures
```

Replace `runs/case` with the actual run directory. Review the selection before drawing unrelated files together.

| Option | Use |
|---|---|
| `--products t2,refl,wind10` | Common aliases for temperature, reflectivity, and 10 m wind. |
| `--products all` | Request the native catalog's available products; output can be large. |
| `--timeidx N` | Select an index within each file, or each timeline with `--series`; default is all. |
| `--source-label "Stock WRF"` | Label files produced by a model other than WOOF correctly. |
| `--size 1600x1200` | Change native PNG dimensions. |
| `--barbs` / `--streamlines` | Choose a wind overlay style. |
| `--heavy` | Compute additional heavy ECAPE diagnostics at import, with extra CPU cost; this does not supply every absent diagnostic. |
| `--theme dark` | Use the built-in dark theme. |
| `--products var:FIELD` | Plot a stored generic field offered by the actual catalog. A field stored in kg kg-1 is drawn in g kg-1. |
| `--products isotherm_height_0c,supercooled_water_path_0_3km,cloud_water_column_max` | Column products the wrfout import derives from its native levels: the height of the 0, -10 and -20 C isotherms (metres above sea level, the lowest crossing above the ground, blank where the column never reaches it), the supercooled liquid water path (`supercooled_water_path` for the whole column, `_0_3km` and `_3_6km` for those layers above ground, in g m-2, on the model's own dry-air mass per level), the column maximum of each hydrometeor mixing ratio (`cloud_water_`, `rain_water_`, `cloud_ice_`, `snow_` and `graupel_column_max`, g kg-1) and `simulated_ir_satellite`, a brightness temperature from the column's cloud optical depth. |
| `--products xsec:QCLOUD=0.01,0.1/wa` | A vertical cut whose fill is drawn on the named range (0.01 to 0.1 g kg-1 on every frame); `xsec:QCLOUD~log` draws the fill in log10 with the bar labelled in the field's own numbers, 0.1, 1, 10. |

The default native engine needs its matching executable and map assets. If they are absent, automatic rendering refuses and names the remedy. `--engine matplotlib` is an explicitly requested limited workaround; it is not an automatic substitute for the native product set.

### Interpret quantities and time windows correctly

| Requested quantity | What the current WOOF history route provides |
|---|---|
| 10 m wind and window maxima | Wind fields and documented snapshot/interval maxima. A snapshot maximum is not gust magnitude. |
| One- or six-hour precipitation | Accumulation products when their required saved times and fields exist. A run-total field is not a fixed-duration accumulation. |
| 24-hour temperature minimum/maximum | Complete required hourly history through 24 hours, not a shorter forecast stretched into a daily statistic. |
| Low/middle/high cloud | Layer cloud diagnostics. They do not establish total cloud fraction or visibility distance. |
| Plain-temperature lapse rates | `var:wrf_lapse_rate_0_3km` and `var:wrf_lapse_rate_700_500`, when listed, in degC/km. They are not virtual-temperature lapse rates. |
| Gusts, visibility, precipitation-type categories, DCAPE | Not supplied by this history-rendering route; require separate analysis or observations. |
| 2 m wet bulb, heat index, wind chill, VPD, dewpoint depression, equivalent potential temperature | Related state plots are context; these derived quantities are not supplied by the current route. |

Keep the renderer's strategy and skip messages with a scientific interpretation. A more frequent history cadence does not automatically satisfy every fixed-hour native window convention. A field with a similar name can represent a different height, layer, time interval, or physical quantity.

Default output is grouped under a timestamped render directory by domain, product, and valid day. `--layout flat` and `--run-stamp off` are compatibility options for consumers that require the older layout; they remove useful separation between results.

## 12. Continue or branch a forecast

History files and checkpoint files serve different purposes. Restart files are `gpuwmrst_*.npz`; multi-domain checkpoints require a complete compatible set. Preserve their companions and original placement. A forced stop does not guarantee that another checkpoint was written.

Enable the intended positive `restart_interval_s` in the experiment before running; zero disables checkpoint writing. Cadence must satisfy the configuration's time-step rules. Keep the original configuration and input identities available for restoration.

### Continue a config-driven run

```text
woof resume CONFIG.toml --outdir runs/direct-case --from latest
```

This locates the newest valid complete checkpoint set in the interrupted run's output directory. To select an explicit checkpoint, replace `latest` with its real path. `resume` uses the same direct-run restart contract; it does not accept a different scientific configuration merely because the filenames match.

The configuration argument is resolved against the run directory as well as
the working directory. An argument that is not a readable file is tried with
the `.toml` extension a file manager hides, then against `--outdir`, and last
against the configuration the run in `--outdir` recorded for itself:
`child.toml` for a downscaled child, `experiment.toml` for a prepared
forecast, `captured-config-<run id>.toml` for a supervised run. `--explain`
prints which of those answered and which did not; when none of them is a
readable configuration file the refusal names every path it tried.

A downscaled child's run directory is refused instead of resumed. A child is
integrated by `woof downscale` from an archived parent, and no door
continues one from its own checkpoint. A child that reached its last frame is
told there is nothing to resume and given the render line for the frames it
left; one that stopped inside its forecast is re-run with `woof downscale`.
Its checkpoints remain useful: they are what the next downscale reads as
parent evidence.

The explicit direct-run form is:

```text
woof run CONFIG.toml --restart CHECKPOINT.npz --outdir runs/direct-case
```

### Continue a prepared forecast

Use the matching prepared bundle, configuration, and WPS authority where required, and a fresh output parent:

```text
woof sim PREPARED_DIR --experiment-config CONFIG.toml --wps-namelist CONFIG.namelist.wps --restart CHECKPOINT.npz --outdir runs/continued --print-command
```

Review, then remove `--print-command` to integrate. Current prepared single-domain and hierarchy runners use the canonical checkpoint path; their configuration and input checks still apply. Go also exposes `--prepared-root` with `--restart` when reusing the bundle.

`--sealed-forcing-extension` is a separate prepared-hierarchy contract for an admitted append-only forcing prefix. It does not turn an ordinary single bundle into a hierarchy or add missing future forcing. Longer runtime needs matching forcing coverage as well as an allowed configuration change.

### Branch within permitted changes

`branch` reads the source run and creates a new run with permitted checkpoint-compatible changes. It does not allow arbitrary physics, grid, or state changes.

```text
woof branch CONFIG.toml --from-run runs/direct-case --from latest --outdir runs/branch-review --set tiles.mode=auto --prepare-only
```

This writes the reviewed branch and receipts without integrating. Inspect its output before continuing by the printed route. Allowed settings are listed by `woof branch --help`; they include certain timing, output, relocation, and tile controls. A requested change outside the restart contract is refused. The new directory must be empty and outside the source run.

## 13. Read results, logs, and exit status

Go and Sim normally use an output parent plus a timestamped `run-...` child. Read the actual path printed by the command rather than assuming the output parent itself holds `report.json`. Direct `run` writes straight into its `--outdir`.

| Artifact | Use |
|---|---|
| `wrfout...` / `wrfout/` | Saved model history for inspection and rendering. |
| `gpuwmrst_*.npz` and checkpoint companions | Validated continuation state when checkpoint writing is enabled. |
| Forecast `report.json` and receipts | Run status, health/validity results, and bound inputs. |
| `progress.json` or `run-progress.json` | Last published progress for the route. |
| `launch.log` | Detailed stage diagnostics for the staged Go route. |
| `worker-01.stdout.log`, `worker-01.stderr.log`, subsequent attempt numbers | Direct supervised-run worker output and failures. |

Check the process exit status and the model's report. A finished preparation is not a completed forecast. A PNG does not prove that integration passed its health checks. A failed plot stage can follow a completed model stage; inspect each outcome before deciding what to rerun.

In POSIX shells, inspect `$?` after the command. In PowerShell, inspect `$LASTEXITCODE`. For example:

```powershell
woof check CONFIG.toml
$LASTEXITCODE
```

Treat nonzero as a request to inspect the named failure. Codes have command-specific meanings; do not build automation around an assumed universal meaning for `1` versus `2`.


Collect diagnostics without submitting anything:

```text
woof report RUN_DIR --dry-run
woof report RUN_DIR --output diagnostics.zip
```

The first command shows the proposed manifest and redactions. The second writes an archive. Add `--log FILE` for a log saved elsewhere. Review the archive before sharing it.

## 14. Control durable jobs on Linux nodes

The remote commands use an existing Linux installation and existing remote files. They do not install WOOF or upload inputs. All remote paths below are placeholders to replace; `NODE_ALIAS` is an existing SSH alias or `user@host`.

Probe the environment:

```text
woof remote probe --host NODE_ALIAS --python /path/to/venv/bin/python --workspace /path/to/workspace --json
```

Review a launch:

```text
woof remote start --host NODE_ALIAS --python /path/to/venv/bin/python --workspace /path/to/workspace --config /path/to/workspace/case.toml --outdir /path/to/workspace/runs/new-case --products t2,refl --dry-run --json
```

The workspace, configuration, and any supplied geography/preparation must already exist remotely. The output directory is an **absolute remote path** laid out the way a local run lays it: it is a case folder, the run claims its own stamped run folder inside it, a second run of the same configuration claims a stamped sibling beside the first, and `latest-run.txt` in the case folder names the newest. A directory that is itself a run folder is honoured as it is, and a run folder that already exists is refused by name; name the directory above it, or a run folder that does not exist yet. The review and every later status, log, and gallery reply name the run folder the job actually used. Read the review, then remove `--dry-run` to start. `--expected-config-sha256` and the related expected-input flags can bind a scripted start to the reviewed bytes. Save the returned job ID.

List and inspect jobs:

```text
woof remote list --host NODE_ALIAS --python /path/to/venv/bin/python --workspace /path/to/workspace --json
woof remote status --host NODE_ALIAS --python /path/to/venv/bin/python --workspace /path/to/workspace --job JOB_ID --json
woof remote logs --host NODE_ALIAS --python /path/to/venv/bin/python --workspace /path/to/workspace --job JOB_ID --json
```

Use the log result's cursor with `--cursor` for the next bounded chunk. `--limit` controls the maximum requested bytes; it is not an instruction to download an unbounded log.

To stop the identified job, run `woof remote stop` with the same host, Python, workspace, and `--job` operands. This is an action, not a preview. To review continuation into new output:

```text
woof remote resume --host NODE_ALIAS --python /path/to/venv/bin/python --workspace /path/to/workspace --job JOB_ID --outdir /path/to/workspace/runs/continued-case --from latest --dry-run --json
```

The job survives local process exit and SSH disconnection. If a start is uncertain, refresh the job list before retrying, or retry the same request with the `--request-id` a reply or timeout printed so the node answers with the job that attempt already created instead of a sibling run. Resume output may land anywhere outside the source job's own run folder; a sibling run folder beside it is fine, and a resume aimed inside it is refused naming that folder. A connection failure is not confirmation of termination. Resume requires a complete valid checkpoint and supports expected-checkpoint hashes for reviewed automation.

`--identity` and `--ssh-config` identify files on the local computer. SSH uses existing keys/agent and strict known-host verification. Passwords and private-key contents are not put into node profiles.


## 15. Use structured interfaces in scripts

Prefer JSON surfaces over scraping aligned human tables:

```text
woof sources --json
woof research catalog --json
woof research hardware --json
woof check CONFIG.toml --json
woof run-plan --physics-profiles
```

`woof run-plan PLAN.json` executes a versioned run plan and emits JSONL events, also saved in the run's `events.jsonl`. Review a plan with `--resolve` or `--estimate` before execution. Its schema is documented in [run plans](../run-plan.md); a plan is not an arbitrary list of shell commands.

`woof sim ... --print-command` prints the fully bound runner invocation without running it. For progress consumed by a supervising program, Sim accepts `--progress-format jsonl`.

On a busy GPU host, `woof run-plan --probe --no-readiness` returns the NVML-only device inventory. Omitting `--no-readiness` adds execution-based environment checks and may create a CUDA context; do not treat that expanded probe as a passive polling loop.

Keep return codes, exact command arguments, configuration copies, and reported output paths with your experiment. For long-lived remote work, save job IDs rather than assuming an SSH process lifetime is the forecast lifetime.

## 16. Command reference

| Command | Purpose | Preview or inspection |
|---|---|---|
| `tui` | Open the terminal workspace. | `--snapshot FILE.html` captures the interface without starting a forecast. |
| `domain` | Create a configuration for a point/footprint and budget. | Review its emitted geometry and new files. |
| `domain-fit` | Fit a complete editable starter. | Omit `--write`. |
| `domain-tiles` | Create a validated streaming copy. | Omit `--write`. |
| `check` | Configuration and memory/readiness evaluation. | `--explain`, `--json`; `--alloc` explicitly uses the GPU. |
| `sources` | Inspect forcing routes. | `ID --explain`, `--json`. |
| `research` | Catalog, hardware, attributes, and new study creation. | `catalog --json`, `hardware --json`. |
| `case-catalog` | Browse built-in or supplied cases and create configurations. | `list`, `show`, `preview`, `default`. |
| `fetch` | Acquire source data or author supported handoffs. | Source-specific templates/validation; read its help. |
| `prep` | Prepare caller-supplied inputs. | `--dry-run`, `--show-source`, support reports. |
| `sim` | Run a finished prepared bundle. | `--print-command`. |
| `go` | Select and execute the preparation/forecast/render chain. | `--dry-run`. |
| `hydro` | Prepare pinned NextGen inputs or run a complete GPU hydrology realization. | `demo lower-colorado-tx --prepare-only`; see [HYDRO.md](HYDRO.md) for current runtime limits. |
| `run` | Run declared local case data, WRF real output, or WPS met_em. | Check/import the configuration or producing namelists first. |
| `import-namelist` | Translate a WRF namelist pair and report substitutions. | Omit `--output`. |
| `downscale` | Integrate an offline child from archived history. | `--dry-run` writes a derived plan into new output. |
| `render` | Draw saved history. | `--list-products`, preferably with actual inputs and `--series`. |
| `resume` | Continue a direct run from its valid checkpoint. | Inspect the selected original run and checkpoint. |
| `branch` | New run with permitted checkpoint-compatible changes. | `--prepare-only` writes branch artifacts without integration. |
| `remote` | Durable Linux-node jobs. | `probe`; `start/resume --dry-run`; `list/status/logs`. |
| `doctor` | Inspect the installed environment and remedies. | `--explain`, `--source ID`. |
| `setup` | Stage verified native artifacts and physics tables. | Geography is explicit with `--with-geog`. |
| `version` / `update` | Identify the environment / print its upgrade command. | `version` asks the package index only with `--check-pypi`; `update` does not install. |
| `report` | Write a diagnostic archive for a run. | `--dry-run`. |
| `run-plan` | Structured orchestration. | `--resolve`, `--estimate`, `--sources`, `--physics-profiles`. |

Specialist surfaces such as ensembles, observations, verification campaigns, cycle streaming, and mesh generation have separate contracts. Use `woof --help-all` and their command help before applying them to production data.

## 17. Troubleshooting

| Failure | What to check |
|---|---|
| Wrong version or missing command | `woof version`; ensure the intended environment is activated. Use `python -m woof.cli` to bind the interpreter explicitly. |
| Missing or mismatched `recast-woof-data` | Install the companion matching the main package's exact version. In a checkout, install the local `recast-woof-data` before the main editable package. |
| Missing native tool or wrong render contract | Run Doctor and stage the release's matching artifacts with `fetch-bridges`, or build the correct checkout workspace. |
| Missing geography | Use `fetch-geog` or point the route at a valid existing WPS_GEOG tree. |
| Source/cycle unavailable | Inspect its coverage, cycle hours, lead horizon, publication state, and acquisition requirements. A calendar entry does not prove file availability. |
| Configuration or companion already exists | Use a new filename and stem. Preserve the previous configuration and its authorities. |
| GPU memory refused | Read the binding phase and actual free budget. Fit a new copy, use planned streaming with enough RAM, or free the intended GPU. Declaring more capacity is not a repair. |
| Host memory refused | Check source decode and host-store costs. GPU tiling does not remove all initialization RAM requirements. |
| Preparation needs PMSL or another donor | Supply compatible data under the exact role; different quantities are not substitutes. |
| Prepared input/config hash mismatch | Use the original matched authorities or prepare anew. Do not edit the hashes. |
| No valid restart found | Confirm restart writing was enabled and preserve a complete checkpoint set from the original run. History/log files are not checkpoints. |
| Plot missing or a window blocked | List products against the actual compatible series; inspect fields, duration, cadence, and quantity-specific limitations. |
| Direct supervised run failed without a long terminal traceback | Read the worker stderr/stdout logs and `run-progress.json` in its output directory. |
| Remote start outcome uncertain | Refresh `remote list/status` before launching again, or retry with the printed `--request-id` so the node reconciles the attempt. Retain the returned job ID. |

Use `--explain` for detailed reasons and `woof report RUN_DIR --dry-run` to inspect a diagnostic collection. Keep the original files and reports available when comparing or reporting an outcome.
