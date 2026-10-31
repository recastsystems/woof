# Every option, every door

The complete command-line surface, read off the parsers themselves.  It exists because a flag that appears in no document is a feature nobody can reach: `--parent-namelist` gated the whole stock-WRF-parent route, `--tiles` gated the only streamed prepared route, and neither was written down anywhere.

`tests/test_docs_extras_agree_with_code.py` holds this page against the parsers in both directions, so it cannot fall behind the code and cannot name a flag that was removed.  Regenerate it with `python -m tools.build_cli_options_doc` after changing any option.

Everything is listed with the help text the tool itself prints.  A door's positional arguments come first, in the order they are written on the command line and under the names its `--help` usage line gives them: `[NAME]` is optional, `NAME [NAME ...]` repeats.  Where an argument is restricted to a fixed set of values, run that door with `--help` for the list -- it is read from the tree at run time rather than pinned here.  `--help` itself is omitted.

## `rw-wps`

`--validate-physics-plan`, `--canonical-physics-plan-output`, `--extend-root-preparation`, `--sealed-prepared-cache`, `--domain-source-orography`, `--validate-hrrr-domain` and `--no-stock-wrf-export` are gates on the preprocessing route; each is off unless named.

| option | what it does |
|---|---|
| `--accept-licence LICENCE` | explicitly accept a model row's exact restricted/unresolved weight licence; repeat for multiple licences |
| `--ack` | registry-owned expert physics acknowledgement id; repeatable |
| `--as-posted POSTING_DIR` | prepare as an as-posted fetch publishes the window's leads: POSTING_DIR is that fetch's posting/ folder; the preparation starts on the first leads and its seal writes the input manifest, so no --source-manifest pair is given (a mapped source names where with --author-input-manifest, beside the fetched files) |
| `--author-input-manifest` | create an exact mapped or 20CRv3 input manifest; conflicts with an existing --source-manifest/--source-manifest-sha256 pair |
| `--author-mapping` | create-only path for a mapping compiled from --descriptor; the adjacent *.authoring.json receipt binds descriptor/Vtable bytes |
| `--author-only` | author the requested create-only mapped contract or 20CRv3 member manifest and exit; requires --author-input-manifest and does not need run geometry |
| `--bridge` | prebuilt woof all-Rust source-specific GRIB bridge executable; omitted on the era5/gfs routes it resolves through the shared bridge ladder (environment override, a checkout build, staged bridges under ~/.woof/bridges) exactly as woof go does |
| `--canonical-physics-plan-output PATH` | create an exact canonical UTF-8 copy of the plan validated by --validate-physics-plan; refuses an existing output |
| `--child-workers` | bounded CPU worker budget for parallel d02..dNN initialization (1..32) |
| `--composition` | strict gpuwm-mapped-composition-v2 product join contract |
| `--contributing-mapping ROLE=PATH` | cross-source composition: a contributing source's own mapping document under the mapping_role its field_sources binding declares; bytes must hash to the composition's pinned SHA-256 |
| `--cpu-preprocess-bridge` | _(the parser declares no help text for this option)_ |
| `--cycle` | GFS cycle, YYYY-MM-DDTHH or YYYY-MM-DD_HH:MM:SS |
| `--descriptor` | explicit rw-wps.descriptor.v1 science contract; requires --author-mapping and, for GRIB, --vtable |
| `--domain-source-orography DNN=PATH` | ERA5 hierarchy source-orography binding; repeat once for every domain (d01..dNN). All bindings use --source-orography-variable |
| `--domain-spec` | strict gpuwm-hrrr-target-domain-v1 Lambert root-domain JSON; nested layouts come from --wps-namelist/--namelist-input |
| `--dry-run` | validate route-specific arguments and print the exact internal command |
| `--experiment-config` | _(the parser declares no help text for this option)_ |
| `--experiment-config-suite` | bind the explicitly configured physics suite in the native export |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--extend-root-preparation` | sealed HRRR predecessor to extend by exactly one forcing hour |
| `--forecast-end-hour` | inclusive absolute HRRR source lead |
| `--forecast-start-hour` | absolute cycle-relative HRRR lead used for model time zero |
| `--geog-root` | WPS_GEOG root used to build a domain-specific native static cache; requires --domain-spec and replaces --static-cache/--static-receipt |
| `--gfs-series` | tab-separated HOUR and GFS GRIB2 path inventory |
| `--grib` | combined ERA5 GRIB1 series |
| `--grib2-dump` | override the GRIB2 dump tool; omitted, it resolves through the shared bridge ladder exactly as --grib2-inventory does |
| `--grib2-inventory` | override the GRIB2 inventory tool; omitted, it resolves through the shared bridge ladder (WOOF_GRIB2_INVENTORY, a checkout build, the wheel's bundled copy, then the staged ~/.woof/bridges) |
| `--hierarchy-workers` | bounded mapped d02..dNN initialization workers (1..32) |
| `--history-interval-seconds` | positive output cadence used by HRRR preparation and the prepared-cache forecast identity |
| `--initial-inputs JSON` | separate packaged analysis inventory for the initial state; --source continues to supply every lateral boundary frame |
| `--input` | mapped source file; repeat in deterministic time/file order |
| `--input-list` | file naming the mapped source files, one path per line, in the same deterministic time/file order the repeated --input flag spells; the spelling that keeps a field-per-file source's hundreds of inputs inside the Windows 32 KB command-line limit |
| `--list-sources` | print the provenance-bound source capability manifest as JSON |
| `--mapped-engine {rust,python}` | which engine decodes mapped source bytes; omitted, the default engine runs. `python` is a documented WORKAROUND -- the slower Python decode path, kept reachable so a decode the Rust engine gets wrong has a way around it while the defect is fixed -- not a supported mode to prefer |
| `--mapping` | strict rw-wps.mapping.v1 field/coordinate/target contract |
| `--namelist-input` | _(the parser declares no help text for this option)_ |
| `--namelist-support-report` | classify --wps-namelist/--namelist-input and print the exact stock-WRF versus woof support report as JSON |
| `--no-stock-wrf-export` | prepare the forecast only, and do not attempt the bonus unchanged-WRF wrfinput/wrfbdy export |
| `--output-root` | _(the parser declares no help text for this option)_ |
| `--physical-base-prepared` | HRRR base preparation whose sealed bridge can be reused |
| `--physical-input-provider` | posted native physical provider with a frozen member plan |
| `--physical-input-store` | sealed native physical snapshots on the target grid |
| `--physical-member-index` | original recipe member index in the posted provider |
| `--physical-output-store` | capture native mapped snapshots before real initialization |
| `--physics-profile` | optional assertion that the experiment IS this shipped single-domain suite, refused on any switch drift; omitted, the config's own physics is prepared as written and its WRF-verification status is reported (the HRRR route still requires a shipped profile: its cold-start evidence contract is profile-keyed) |
| `--pipeline-workers` | _(the parser declares no help text for this option)_ |
| `--prepare-workers` | _(the parser declares no help text for this option)_ |
| `--preprocess-backend {cuda,cpu,auto}` | select CUDA or deterministic parallel CPU preprocessing |
| `--preprocess-backend-reason` | _(accepted, but not listed by --help)_ |
| `--preprocess-workers` | threads for CPU preprocessing (default: this machine's CPUs, at most 8, the count its host RAM estimate was measured at; a larger count peaks above that estimate); under --preprocess-backend cuda, the threads of the host steps (masked soil, snow, skin temperature and sea ice), default every CPU |
| `--provenance ROLE=PATH` | composition provenance binding |
| `--root-preparation` | sealed output of the native HRRR root-preparation command; enables parallel d01..dNN hierarchy export for max_dom 1..21; the two namelists remain the topology authority |
| `--run-seconds` | _(the parser declares no help text for this option)_ |
| `--sealed-prepared-cache` | opt in to a prefix-sealed operational HRRR root preparation |
| `--show-physics-registry` | print the canonical GPUWM-owned physics registry v2 as JSON |
| `--show-source MODEL` | print one source declaration as JSON |
| `--show-support-matrix` | print the versioned native WRF compatibility matrix as JSON |
| `--source MODEL` | native source adapter id |
| `--source-format {grib1,grib2,netcdf}` | input format; must agree with the sealed rw-wps.mapping.v1 document |
| `--source-manifest, --source-sha256s` | SHA-256 file manifest covering every downloaded source file |
| `--source-manifest-sha256, --source-sha256s-sha256` | expected SHA-256 of --source-sha256s |
| `--source-orography` | _(the parser declares no help text for this option)_ |
| `--source-orography-variable` | _(the parser declares no help text for this option)_ |
| `--source-root` | the folder holding the source's files: the fetched HRRR cycle, the 20CRv3 member files --author-only reads, or, for a source whose fetch-route row declares its folder layout, the folder whose inputs and supplements it binds itself, authoring DIR/inputs.json and preparing into CONFIG-prepared beside the experiment config (CONFIG-prepared-2 and on once that exists) unless --output-root names one |
| `--source-top-pressure-pa` | smallest pressure represented by the selected source; used by --namelist-support-report to reject vertical extrapolation |
| `--static-cache` | _(the parser declares no help text for this option)_ |
| `--static-input` | _(the parser declares no help text for this option)_ |
| `--static-receipt` | _(the parser declares no help text for this option)_ |
| `--statics-corridor GRID_IDS` | also seal child-resolution statics over the ground each child can reach (the moving-nest corridor); bare flag covers every child domain, or pass comma-separated child grid ids (e.g. 2,3). Required before the prepared tree runner will honor a [relocation] follow source |
| `--stock-wrf-export {optional,required,off}` | mapped preparation's WRF file product: optional by default, required with early configuration admission, or off |
| `--stock-wrf-namelist-input` | unchanged-stock-WRF namelist matching the native hierarchy except for the certified longwave selection and the stock-only ghg_input and do_radar_ref keys; both declare use_theta_m = 0, the dry theta the exported files hold |
| `--supplement ROLE=PATH` | composition supplement binding; repeat roles for multiple files |
| `--valid-time` | initial UTC time, YYYY-MM-DDTHH or WRF form YYYY-MM-DD_HH:MM:SS. On --source hrrr this is the CYCLE; model time zero is cycle + --forecast-start-hour and is derived for every stage |
| `--validate-hrrr-domain PATH` | validate a strict HRRR target domain and its complete native interpolation window |
| `--validate-physics-plan PATH` | validate and resolve a gpuwm-physics-plan-v2 JSON document |
| `--version` | show program's version number and exit |
| `--vtable` | ERA5 GRIB1 Vtable |
| `--wps-namelist` | standard WPS geometry/static-selection namelist |
| `--wrf-version {3,4}` | with --namelist-support-report: the WRF line the namelist was written for, which selects only the Registry default an omitted &dynamics/use_theta_m takes (3: 0, dry theta, the line operational HRRR v4 runs; 4, the default: 1, moist theta) |

## `woof`

| option | what it does |
|---|---|
| `--help-all` | show every command |

## `woof adapt`

| option | what it does |
|---|---|
| `--descriptor JSON` | completed rw-wps.descriptor.v1 document |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--grib2-dump EXE` | expert override paired with --grib2-inventory |
| `--grib2-inventory EXE` | expert override paired with --grib2-dump |
| `--input FILE` | your own GRIB2 or NetCDF input file, matching the descriptor's declared format (repeat for every file in the series) |
| `--output-dir DIR` | directory for create-only adapter authorities and manifest |
| `--skeleton JSON` | create a review-required descriptor scaffold and stop |
| `--vtable VTABLE` | 11-column WPS Vtable selector authority. Required for GRIB descriptors, and never defaulted: this command adapts arbitrary sources, and quietly reaching for a GFS Vtable would mis-map every other product. Must be omitted for NetCDF descriptors, whose selectors name CF variables directly. A worked GFS example installs with the package -- <woof package>/authorities/Vtable.GFS.rw-wps |

## `woof branch`

| argument | what it does |
|---|---|
| `CONFIG` | the config the source run used; the branch edits a copy of it and the restart identity check refuses any other |

| option | what it does |
|---|---|
| `--allow-shared-gpu` | UNSUPPORTED: permit another substantial CUDA compute context; device verification and the GPUWM UUID lock remain enforced |
| `--directory-input-hash {inventory,content}` | how declared directory inputs (the static geography tree) are bound to this run's identity: 'inventory' (default) uses relative path, size, and mtime; 'content' reads every file and uses its SHA-256. Use 'content' when two runs being compared for byte identity stage their geography separately, and when an mtime-preserving change to that tree must not go unnoticed (docs/public/DETERMINISM.md). Also settable as WOOF_DIRECTORY_INPUT_HASH. |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--from CKPT\|latest` | explicit gpuwmrst_*.npz checkpoint to branch from, or 'latest' (default) for the newest valid set in --from-run |
| `--from-run RUNDIR` | the source run's output directory -- where its gpuwmrst_*.npz checkpoints are. Optional only when --from names a checkpoint file explicitly |
| `--gpu-uuid GPU-UUID` | physical GPU UUID to lock (required on multi-GPU hosts) |
| `--health-debug` | enable debug phase health attribution hooks |
| `--icechunk-store URL` | Output store base: local path or s3://bucket/prefix; domain and member paths are appended (default RUN/icechunk) |
| `--keep-member-files` | also retain every member's full history files |
| `--members N` | make an N-member ensemble with aggregate products |
| `--no-memory-gate` | run a case whose priced peak envelope exceeds this card's free memory anyway, as `woof go --no-memory-gate` does: the envelope is an upper bound and the card's own allocation then decides; a model state too big to build at all is still refused |
| `--no-supervise` | run the experiment in this process (escape hatch; disables fresh-process recovery and exclusive-GPU supervision) |
| `--outdir OUT` | the NEW run's output directory; it must be empty and must not be inside the source run |
| `--output-config TOML` | Read [output] formats and icechunk_store from TOML; CLI values override it |
| `--output-format FORMATS` | Output-format table rows, comma separated (default netcdf); netcdf,icechunk adds atomic native Zarr v3 history |
| `--prep-timeout SECONDS` | optional preparation heartbeat timeout; default is no timeout until integration begins |
| `--prepare-only` | write the branch run directory, its config and its receipts, then stop without integrating -- the price-it-first step a what-if screen shows before committing a card |
| `--restart-roster JSON` | continue the exact original members from a durable ensemble restart roster |
| `--set KEY=VALUE` | a setting to change in the branched run, repeatable. Changeable from a checkpoint: run_seconds, restart_interval_s, acknowledgements, relocation.*, tiles.*, devices.*, output.*, simulated_radar.*, domain.<grid_id>.history_interval_s, domain.<grid_id>.history_begin_s, domain.<grid_id>.history_end_s, domain.<grid_id>.tiles.*, domain.<grid_id>.output.*, domain.<grid_id>.surface_energy_diag. Everything else is refused by name, because the restart identity binds it |
| `--supervisor-max-restarts N` | fresh-process recovery attempts (default 3) |

## `woof case-catalog`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof case-catalog create`

| argument | what it does |
|---|---|
| `case_id` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--ack` | explicit native scientific acknowledgement; never inferred from catalog prose |
| `--card` | _(the parser declares no help text for this option)_ |
| `--catalog` | custom JSON, TOML or ZIP catalog; defaults to the bundled historical cases |
| `--expected-catalog-sha256` | bind creation to the exact original catalog bytes displayed by preview |
| `--geometry-only` | validate and open declared geometry; defer GPU memory admission to target Review/Run |
| `--json` | emit compact JSON for the interface or scripts |
| `--native-overrides` | JSON shared/domains scientific overrides; validated against the native contract |
| `--out` | new experiment .toml; existing files are preserved |
| `--physics-profile` | explicit native profile replacing the catalog's selected profile |
| `--source-option` | listed source/initialization ID; default is the catalog's recommended option |
| `--tier {lower,recommended,upper}` | _(the parser declares no help text for this option)_ |
| `--vram-gib` | _(the parser declares no help text for this option)_ |

## `woof case-catalog default`

| option | what it does |
|---|---|
| `--json` | emit compact JSON for the interface or scripts |

## `woof case-catalog export`

| option | what it does |
|---|---|
| `--catalog` | custom JSON, TOML or ZIP catalog; defaults to the bundled historical cases |
| `--json` | emit compact JSON for the interface or scripts |
| `--original` | export the exact original JSON/TOML bytes; default is normalized JSON |
| `--out` | _(the parser declares no help text for this option)_ |

## `woof case-catalog list`

| option | what it does |
|---|---|
| `--catalog` | custom JSON, TOML or ZIP catalog; defaults to the bundled historical cases |
| `--event-kind` | _(the parser declares no help text for this option)_ |
| `--json` | emit compact JSON for the interface or scripts |
| `--limit` | _(the parser declares no help text for this option)_ |
| `--offset` | _(the parser declares no help text for this option)_ |
| `--query` | _(the parser declares no help text for this option)_ |
| `--source` | _(the parser declares no help text for this option)_ |

## `woof case-catalog native-settings`

| option | what it does |
|---|---|
| `--json` | emit compact JSON for the interface or scripts |

## `woof case-catalog preview`

| argument | what it does |
|---|---|
| `case_id` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--catalog` | custom JSON, TOML or ZIP catalog; defaults to the bundled historical cases |
| `--json` | emit compact JSON for the interface or scripts |
| `--native-overrides` | JSON shared/domains scientific overrides; validated against the native contract |
| `--physics-profile` | explicit native profile replacing the catalog's selected profile |
| `--source-option` | listed source/initialization ID; default is the catalog's recommended option |
| `--tier {lower,recommended,upper}` | _(the parser declares no help text for this option)_ |

## `woof case-catalog search`

| option | what it does |
|---|---|
| `--catalog` | custom JSON, TOML or ZIP catalog; defaults to the bundled historical cases |
| `--event-kind` | _(the parser declares no help text for this option)_ |
| `--json` | emit compact JSON for the interface or scripts |
| `--limit` | _(the parser declares no help text for this option)_ |
| `--offset` | _(the parser declares no help text for this option)_ |
| `--query` | _(the parser declares no help text for this option)_ |
| `--source` | _(the parser declares no help text for this option)_ |

## `woof case-catalog show`

| argument | what it does |
|---|---|
| `case_id` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--catalog` | custom JSON, TOML or ZIP catalog; defaults to the bundled historical cases |
| `--json` | emit compact JSON for the interface or scripts |

## `woof cases`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--json` | emit the registry as JSON for a front end |

## `woof cds-credentials`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--json` | return safe credential status as JSON |
| `--save` | read endpoint and key as JSON from stdin and save privately |

## `woof cells`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof cells analyze`

| argument | what it does |
|---|---|
| `WRFOUT [WRFOUT ...]` | wrfout NetCDF file(s), a directory of them, or a glob; one domain per invocation, sorted by name (WRF's stamp order) |

| option | what it does |
|---|---|
| `--json` | print the analyze receipt as JSON |
| `--ladder BOTTOM:TOP:STEP` | the fixed height ladder titan sees, metres above sea level, cell centres (default 250:18000:250: 72 levels) |
| `--no-catalog` | stop after titan analyze; write no catalog |
| `--no-temperature` | export reflectivity only; by default the model temperature rides along on the ladder so titan reports each cell's mean temperature |
| `--out DIR` | the case folder; the series lands under <DIR>/<domain>/cells/<first-valid-day>/ |
| `--profile {legacy,severe,research,operational}` | titan threshold profile (default severe) |
| `--titan PATH` | the titan binary (a separate program WOOF does not ship; without one, analyze is off); default resolves $WOOF_TITAN, then the bridge directories, then PATH |
| `--titan-config FILE` | a titan key=value config overriding the profile |

## `woof cells catalog`

| argument | what it does |
|---|---|
| `WRFOUT [WRFOUT ...]` | wrfout NetCDF file(s), a directory of them, or a glob; one domain per invocation, sorted by name (WRF's stamp order) |

| option | what it does |
|---|---|
| `--bundle DIR` | a titan analyze bundle (frames.jsonl, tracks.json, ...) |
| `--json` | print the catalog receipt as JSON |
| `--out DIR` | where the catalog files go |

## `woof cells export`

| argument | what it does |
|---|---|
| `WRFOUT [WRFOUT ...]` | wrfout NetCDF file(s), a directory of them, or a glob; one domain per invocation, sorted by name (WRF's stamp order) |

| option | what it does |
|---|---|
| `--json` | print the export receipt as JSON |
| `--ladder BOTTOM:TOP:STEP` | the fixed height ladder titan sees, metres above sea level, cell centres (default 250:18000:250: 72 levels) |
| `--no-temperature` | export reflectivity only; by default the model temperature rides along on the ladder so titan reports each cell's mean temperature |
| `--out DIR` | directory for input.tfs and export-receipt.json |

## `woof certify`

| option | what it does |
|---|---|
| `--band BAND` | acceptance band for this configuration; it is addressed by the configuration's SHA-256, and certify refuses a band keyed to another one |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--metrics-csv CSV` | matched-comparison metrics CSV for the run |
| `--out-verdict VERDICT` | write the verdict document here (it is printed either way) |
| `--run-capsule CAPSULE` | certification-capsule.json written by the run |
| `--wrf-reference-manifest MANIFEST` | WRF reference manifest naming the executable, build recipe, namelists and reference wrfout bytes the comparison was made against |

## `woof check`

| argument | what it does |
|---|---|
| `[CONFIG]` | experiment TOML (or legacy RunConfig TOML, wrapped as a one-domain experiment); omit only for woof check --json context-free box inventory |

| option | what it does |
|---|---|
| `--alloc` | construct every persistent allocation on the device, zero steps, report measured vs estimate (N0; GPU required) |
| `--budget-gib GIB` | declared allocation budget (free VRAM minus allocation reserve); estimate only |
| `--column-chunk COLS` | Radiation column-cap override (the first over-budget lever) |
| `--devices N` | price the run split into N resident slabs, one per card (or as [devices] ids places them): replaces [devices] count the way `woof go --devices N` does, and reports the memory envelope of every card and of the pinned host store |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--forcing-interval-s S` | override the configured or measured forcing cadence for memory sizing (otherwise defaults to ERA5 6-hourly) |
| `--free-gib GIB` | declared free VRAM before reserves, as used by the domain wizard; estimate only |
| `--json` | machine-readable report |
| `--no-host-memory-gate` | report the HOST RAM of the forcing decode, the CPU preparation and a streamed forecast but do not refuse on it. The counterpart of `woof go --no-memory-gate` for the other budget: MemAvailable is a reading of this second, and a busy box can be momentarily short of RAM a run would have had |
| `--prepared-root DIR` | verify prepared forecast inputs and price their land cover and retained boundary tables; bind a finished bundle or its live prepared head |
| `--rail-mib MIB` | whole-machine device residency ceiling: the budget is additionally capped at RAIL minus what every other process on the card already holds (read from NVML before this process touches CUDA). A property of the host, so there is no default |
| `--reserve-gib GIB` | override the calibrated reserve policy with a flat reserve |
| `--vram-gib GIB` | physical VRAM total of the card being sized for. A CEILING on the free figure, never a source of one: a declared --budget-gib plus the reserve can otherwise synthesise more free VRAM than the card physically has |
| `--wps-namelist FILE` | WPS namelist bound by a single-domain prepared bundle (otherwise the config's sibling namelist) |

## `woof companion-domains`

| option | what it does |
|---|---|
| `--availability` | report why each installed physics option is open or closed to a draft |
| `--capabilities` | report the actions, presets and limits this door offers, and write nothing |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--repairs` | check compatible physics replacements without writing a candidate |
| `--request` | a domain edit request document; the candidate it publishes carries every file its route reads, named in the result's route_companions |

## `woof companion-forcing`

| option | what it does |
|---|---|
| `--capabilities` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--request` | _(the parser declares no help text for this option)_ |
| `--schedule-request` | _(the parser declares no help text for this option)_ |

## `woof companion-query`

| argument | what it does |
|---|---|
| `config` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof companion-setups`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof companion-setups save`

| option | what it does |
|---|---|
| `--config CONFIG` | forecast configuration TOML to save as a setup |
| `--library DIR` | setups library directory the setup folder is created in |
| `--name NAME` | name to save this setup under; 1 to 80 characters |

## `woof companion-setups start`

| argument | what it does |
|---|---|
| `SETUP` | saved setup.toml to start a forecast from |

| option | what it does |
|---|---|
| `--cycle CYCLE` | source cycle as YYYY-MM-DDTHH (UTC), or latest |
| `--forecast-start-hour N` | forecast lead in hours after the cycle the run starts at |
| `--hours H` | forecast duration in hours |
| `--name NAME` | name of the new forecast |
| `--out TOML` | path of the new configuration TOML to write |

## `woof compose-source`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--jobs` | parallel frame readers (0: available cores) |
| `--manifest` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |
| `--prepare` | pass remaining options to woof prep using the composed inputs and supplements |

## `woof cycle`

| option | what it does |
|---|---|
| `--accept-snap-offset-seconds` | largest analysis-time offset from the parent-step lattice this run will accept by name (default 0.0: the time must land on a step) |
| `--allow-placement-clamp` | accept a placement clamped into the parent instead of refusing it (default off; a clamp is always receipted either way) |
| `--analysis-increment CYCLE=PATH` | the analysis increment applied at CYCLE, as an npz keyed by PROGNOSTIC FIELD NAME. Repeatable. Use CYCLE=null for an explicit NULL ARM (a zero increment that must be bit-stable through the anchor). The three-hash ingestion gate needs both arms to mean anything |
| `--child-dt-seconds` | child model step; must divide the parent step exactly (default: the parent step) |
| `--child-dx-m` | child grid spacing in metres (default 1000.0); must divide --parent-dx-m exactly |
| `--child-nx` | child grid points west-east (default 199) |
| `--child-ny` | child grid points south-north (default 199) |
| `--child-slots` | identically shaped dormant nests reserved at t=0 (default 0). The RESERVATION is fixed and the PLACEMENT is arbitrary: that is what keeps VRAM deterministic while a child can be anywhere |
| `--cycle-seconds` | model seconds between cycle boundaries; must be a whole number of parent steps |
| `--cycles` | how many cycles to run |
| `--dry-run` | print the boundary lattice and the resolved child ratios, refuse invalid combinations, and write nothing |
| `--epoch-anchor ISO8601` | the parent init's config_start_time (UTC); the ONLY datetime in the cycling spine, every other time is an integer tick from it |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--max-forecast-only-cycles` | consecutive cycles allowed with no analysis before the run halts STALE_ANALYSIS_BUDGET_EXHAUSTED (default 3): forecast-only is legitimate, forecast-only forever is a run that stopped being a DA cycle |
| `--min-separation-km` | two children are never planted on one storm (default 40.0). A request inside this radius of an assigned child is REFUSED by name, never silently dropped |
| `--no-resume` | start again at cycle 1 |
| `--parent-dt-seconds` | parent model step; must be a whole number of milliseconds and divide --cycle-seconds (default: 120 s when it divides the cycle, else the largest step that divides both 120 s and the cycle, e.g. 60 s for a 15-minute cycle) |
| `--parent-dx-m` | the parent's grid spacing in metres. Required with a placement provider: the child/parent refinement ratio is derived from it and a guessed spacing silently changes every placement |
| `--parent-geo-file PATH` | XLAT/XLONG for the parent's mass grid: a radar-grid NetCDF or an npz carrying both. Defaults to the first --placement-obs-file, whose grid IS the target model grid by contract |
| `--parent-kind {mpas-cuda,mpas-cuda-frames,arwen,replay}` | which engine advances the parent |
| `--parent-mesh-id ID` | the mesh identity written into every anchor. No default: an identity the spine guessed is an identity no downstream reader can trust |
| `--parent-state GLOB` | the parent's state series: a glob of npz frames in the SAME on-disk shape tools/cycle_mpas_leg.py --history reads (flat npz: prognostic fields, time_seconds, and the derived diagnostics beside them). Required for a real run; --dry-run does not need it |
| `--placement-obs-field NAME` | which observation plane the obs provider places on (default z_obs). The radar-grid contract ships z_obs, z_max and z_mean side by side and calls the choice the consumer's; an absent name is refused naming what the file does carry |
| `--placement-obs-file PATH` | radar-grid observation file the obs provider places on. Repeatable, ONE PER CYCLE in order; a single file is reused for every cycle. A storm that never moves between cycles is the defect that hid child retirement for a week |
| `--placement-provider {tracker,schedule,obs,none}` | where each cycle's child placements come from (default none: parent-only cycling) |
| `--placement-threshold` | trigger value a peak must reach to earn a child, in the placement field's own units (default 40.0) |
| `--placement-tracker-field NAME` | which PARENT plane the tracker provider places on (default composite_reflectivity) |
| `--port-config PATH` | the port's case configuration JSON. Required for a model parent kind |
| `--port-root PATH` | the MPAS port checkout the forecast worker runs from. Required for a model parent kind |
| `--port-steps` | dycore steps per cycle boundary. Required for a model parent kind; the step RECEIPTS the worker returns are counted against this number, and a leg that ran fewer steps than asked cannot earn the mpas-cuda stamp |
| `--port-timeout` | seconds to wait for one forecast segment (default: no timeout) |
| `--render-products LIST` | which products each boundary is drawn into ROOT/png as it lands, from its frame in ROOT/wrfout: a comma-separated list of catalog slugs, 'all', or 'none' to draw nothing (default: the renderer's default set). A boundary is drawn when the parent's planes sit on a latitude/longitude grid: its own XLAT/XLONG, --parent-geo-file or the first --placement-obs-file |
| `--resume` | continue after the last completed cycle in the ledger (default) |
| `--retire-below-strength` | a child with less than this much signal under it is retired and its reservation returns to the pool. Required with a placement provider and deliberately has NO default: its units are the trigger field's, so a default would be a hardcoded threshold for somebody else's field |
| `--root` | cycle root; the ledger, anchors and per-cycle receipts all live here |

## `woof cyclone-setup`

| option | what it does |
|---|---|
| `--accept-fit FIT_ID` | save only the exact reviewed proposal identified by fitting.fit_id |
| `--accept-licence LICENCE` | explicitly accept a model row's exact restricted/unresolved weight licence; repeat for multiple licences |
| `--advisory-position LAT,LON` | advisory center; bounds the field search and is the last fallback |
| `--card` | a tier (12gb/16gb/24gb/32gb), a size ('10gb') or a model with a recorded size ('RTX 3080') |
| `--cycle` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--hardware-json` | _(the parser declares no help text for this option)_ |
| `--history-interval SECONDS` | how often the 12 km parent writes a wrfout (default 3600) |
| `--hours` | _(the parser declares no help text for this option)_ |
| `--inner-nest RATIO,KM` | a third domain centred in the following nest and moving with it: RATIO to the following nest (3 turns 3 km into 1 km) and its side in km |
| `--isftcflx {0,1,2}` | surface flux over water on both grids (WRF isftcflx), as `woof domain --isftcflx`; default: the suite's own (0) |
| `--json` | emit the JSON result (the default) |
| `--latest-map` | _(the parser declares no help text for this option)_ |
| `--list-sources` | emit the planable sources with their members, cycle hours, forcing interval and coverage envelope |
| `--member MEMBER` | track member id with --tracks; otherwise member in the source route grammar |
| `--model MODEL` | optional model selection for a registry acquisition owner |
| `--name` | configuration name (default: the selected source's own title) |
| `--nest-budget-gib GIB` | grow the following nest, square and in whole parent cells, to the largest the priced tree holds inside this much memory; the parent is unchanged and the preset nest is the floor |
| `--nest-cells N` | the following nest's side in its own cells (a multiple of the parent ratio), instead of the preset or --nest-budget-gib |
| `--nest-history-interval SECONDS[,SECONDS]` | how often the following nest writes a wrfout (default 900); a longer interval is how a multi-day run fits its disk |
| `--nest-km {3.0,2.5}` | nest spacing; 2.5 km authors a 10 km parent at ratio 4 |
| `--out` | _(the parser declares no help text for this option)_ |
| `--point` | _(the parser declares no help text for this option)_ |
| `--quicklook-interval SECONDS` | a quick-look subset on the finest domain every SECONDS between its full frames (quicklook/wrfout_dNN_*); must divide that domain's history interval |
| `--search-area S,W,N,E` | limit track selection by position at the cycle within these bounds |
| `--seed-fields NPZ` | canonical source-analysis arrays carrying that source's own cycle and member identity, to locate the center from |
| `--seed-radius-km` | how far from the advisory position the field search may look |
| `--source SOURCE` | forcing source to initialize from (default gfs); --list-sources prints the planable set |
| `--start-hour N` | forecast lead of the selected cycle to begin at (default 0, the analysis); the run initialises from fN and is forced from fN onward at the source's own cadence |
| `--storm auto\|ID\|NAME` | strongest eligible system at the cycle, or a track id, ATCF id or storm name |
| `--target-host-memory-json` | _(the parser declares no help text for this option)_ |
| `--tiles {off,auto,on}` | _(the parser declares no help text for this option)_ |
| `--track-cache` | cache for automatic best-track acquisition |
| `--track-source {auto,nhc,hurdat2,ibtracs,jtwc}` | observed best-track archive used when no --tracks file is supplied |
| `--tracks FILE` | track JSON or CSV guidance for selection and placement only |
| `--vram-gib` | _(the parser declares no help text for this option)_ |

## `woof doctor`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--json` | emit the checks as JSON |
| `--since VERSION` | print what changed for an existing user between VERSION and this install (the results that move on a bare configuration, and the checkpoints and namelists that stop loading), then exit 0 without running the estate checks. The same note is printed once, automatically, on the first doctor run after an upgrade |
| `--source {20crv3,20crv3-cf,aifs,aigefs,aigfs,aorc,earth2,earth2-ace2,earth2-aifs,earth2-aifs-ens,earth2-aifs2,earth2-aifs2-ens,earth2-atlas,earth2-atlas-crps,earth2-atlas-si,earth2-aurora,earth2-aurora15,earth2-aurora15-6h,earth2-aurora15-ensemble,earth2-aurora15-ensemble-6h,earth2-corrdiff,earth2-derived-ws10m,earth2-dlesym,earth2-dlwp,earth2-fcn,earth2-fcn3,earth2-fengwu,earth2-fuxi,earth2-fuxi-s2s,earth2-gencast-mini,earth2-graphcast-operational,earth2-graphcast-small,earth2-interpmodafno,earth2-pangu,earth2-sfno,earth2-stormcast,earth2-stormcast-conus,earth2-ucast,earth2-weathernext2,earth2-weathernext2-mini,earth2-windgust,ecmwf-ens,ecmwf-open-data,ecmwf-open-data-0p1,era5,era5-l137,era5-land,gdas,gefs,gefs-reforecast,gefs-reforecast-land,gefs-reforecast-weekly,gefs-v11,gefs-v11-flat,gem-gdps,gfs,hex-global,hgefs,hiresw,href,hrrr,hrrr-ak,hrrr-native,hrrr-prs,icon-d2,icon-eu,icon-global,mapped,nam,nbm,ncep-r1-seaice,ngen-csv,nwm-retrospective-forcing,rap,rap-native,refs,rrfs,rrfs-a,rrfs-ens,rrfs-firewx,rrfs-public,rtma,sref,urma,wrf}` | report only this data route's own resolution (repeatable) alongside the shared estate: what its preparation will decode with, and the byte transport its fetch will use. The choices are the source registry -- the same list `woof fetch` and `woof prep` take. Omitted, every route this build knows is reported |

## `woof domain`

| option | what it does |
|---|---|
| `--accept-licence LICENCE` | explicitly accept a model row's exact restricted/unresolved weight licence; repeat for multiple licences |
| `--ack ID` | declare a governed experiment, written verbatim into the emitted [experiment].acknowledgements. Repeatable. This door used to write the nocturnal declaration for you, which silenced the load guard at check/run/go/run-plan and both prepared runners for the life of the file; it no longer does, and refuses instead. The id it accepts is asymmetric-radiation-nocturnal-window-v1: a longwave-OFF suite over a window that includes local night, which you are running deliberately as a daytime-only experiment |
| `--buffer-km KM[,KM...]` | with --polygon, nonnegative geometry buffer in kilometres; one value applies to every domain, or supply exactly one outer-to-inner value per level. Every value is measured from the polygon itself, not from the next inner grid: '800,300,0' puts the outer grid 800 km from the polygon, about 500 km beyond the middle one. With --ladder auto, a multi-value list selects the preset of that depth (default: zero) |
| `--cadence HOURS` | boundary spacing in whole hours, validated against the selected product; required for a source that declares no cadence of its own (hex-global, mapped), where it is the spacing of the history files handed to woof prep |
| `--card` | GPU to size for: a tier (12gb/16gb/24gb/32gb), a size ('10gb'), or a model with a recorded size ('RTX 3080', '5070 Ti'); sets the VRAM budget with no local probe. With no --card, --vram-gib or --hardware-json the wizard MEASURES the local card's capacity (short-lived probe, suppressed by GPUWM_NO_LOCAL_GPU) and refuses, naming both flags, when there is nothing to measure |
| `--chain R1,R2,...` | custom nest refinement ratios, integers in [2, 8] (e.g. --root-dx 3 --chain 4 for 3 km -> 750 m); omit for a single domain at --root-dx. Sized by the same estimator fit loop as the presets |
| `--clock {auto,adaptive,fixed}` | how the run steps. adaptive: each grid starts at the emitted time_step and then follows its own Courant number between 3 and 8 s per km of its spacing (WRF's use_adaptive_time_step, written into [shared]); the terrain clock still caps it over steep ground at launch. fixed: one step throughout. auto (default): adaptive when every grid starts inside those bounds and lies within the 500 m to 12 km spacings the terrain clock measured, fixed otherwise (the tropical clock's 2.5 s per km is always fixed). adaptive on a grid outside the bounds is refused, naming the grid |
| `--cumulus {suite,grid}` | who decides the root's cumulus scheme. suite: the suite's own scheme at any spacing (what naming --physics-profile means when this is left out). grid: the suite's scheme is turned off on a root finer than the 4 km convection-permitting bound, where the dynamics resolve deep convection (what an unnamed suite gets) |
| `--cycle YYYY-MM-DDTHH\|latest` | the forcing CYCLE (UTC), which is the run's start time unless --forecast-start-hour moves it; 'latest' probes the public mirrors for the newest complete gfs/hrrr cycle covering the whole window and prints what it picked; sources without a probe use their declared publication delay) |
| `--data-dir DIR` | explicit forcing directory; automatic go launches otherwise manage request-specific downloads. Manual acquisition and ERA5 paths default to data/<name> |
| `--earth2-batch-root` | shared generated-member batch directory |
| `--earth2-batch-size` | members per admitted GPU sub-batch |
| `--earth2-members` | complete comma-separated noise-id roster |
| `--earth2-weights-package` | local pinned model weight package |
| `--era5-product {reanalysis,ensemble_members}` | explicit ERA5 product; default reanalysis has no member axis |
| `--era5-provider {cds,arco}` | ERA5 provider (default: cds when a Copernicus CDS key is present, otherwise the keyless arco store); ensemble_members requires CDS |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--forcing GRIB` | era5: explicit forcing GRIB path(s) already on disk (default <data-dir>/era5-combined.grib) |
| `--forecast-start-hour K` | initialize the run from the cycle's f{K} FORECAST lead instead of its analysis, so start_time = cycle + K h and the boundaries come from f{K+i}. This is how a window deep in a forecast (say f174..f240) is reached without integrating from f000. Every source whose registry row publishes forecast leads takes it; a row that declares none refuses it by name. The initial condition is then itself a K-hour forecast, and every receipt says so |
| `--geog-root DIR` | staged WPS_GEOG tree, saved for every source (default ${GPUWM_CASE_DATA_ROOT}/WPS_GEOG) |
| `--hardware-json` | selected target hardware snapshot with measured GPU capacity, available memory and device profile; no local GPU probe |
| `--history-interval SECONDS` | how often the ROOT domain writes a wrfout, in seconds (default 3600). Must be a whole number of seconds and a whole number of that domain's time steps -- the loader checks both against the exact rational dt and refuses the emitted file otherwise, before it is written |
| `--hours N` | forecast length (run_seconds = N*3600) |
| `--isftcflx {0,1,2}` | surface flux over water on every grid (WRF isftcflx): 0 standard MM5 roughness, 1 Donelan drag with constant heat and moisture roughness (the tropical cyclone option), 2 Donelan drag with Garratt heat and moisture roughness. Default: the suite's own (0) |
| `--ladder {12,12-3,12-3-1,12-3-1-0.5,auto}` | preset nest dx chain in km (default: 12 -- one 12 km domain, the shape `woof go` runs end to end, same as the interactive session). Nest trees are explicit opt-in: a deeper preset, `auto` (the deepest preset that fits the card), or --root-dx / --chain for anything else; their closing block names the tree runner they route to |
| `--les-below-m METRES` | write the resolved-turbulence closure (PBL off, a 3-D closure, isotropic mixing) on every real-data domain at or below this spacing (default 250 m; 0 never) |
| `--les-closure {smagorinsky,tke}` | the 3-D closure an LES domain runs: smagorinsky (km_opt 3, the default) or tke (km_opt 2, 1.5-order prognostic TKE) |
| `--les-inflow-seeding {on,off}` | cell perturbations on the inflow faces of an LES nest whose parent runs a 1-D PBL (default on) |
| `--member` | ensemble trajectory member; defaults to the route's control |
| `--model` | model key from this source's declared model table |
| `--mosaic-cat` | Noah mosaic tile count on every grid (WRF mosaic_cat) |
| `--mosaic-urban-canopy {dominant,every_tile}` | where Noah mosaic runs the urban canopy with sf_urban_physics = 1: dominant (WRF's rule, the default: only cells that are mostly urban) or every_tile (also the town tiles of mostly rural cells) |
| `--name` | experiment name (default derived from the center) |
| `--nest-history-interval SECONDS[,SECONDS...]` | the same, for the NESTED domains (default 900): one value for every nest, or one per nest, outer to inner (1800,600 is d02 every 30 min and d03 every 10 min). Nests write more often than the root by default because resolving what the root cannot, over a shorter window, is the point of running one. Ignored for a single-domain ladder |
| `--nz N` | vertical mass levels (default: 49); resamples the default eta ladder while preserving its stretching |
| `--out TOML` | emitted experiment TOML path |
| `--physics-choices JSON` | schemes to run in place of the suite's own, family by family, as the physics composer picks them: '{"microphysics": "thompson-mp8", "pbl": "myj", "surface_layer": "eta-similarity"}'. Checked by the engine the way `woof physics-catalog --check` checks them and written into the config the way `--into` writes them, on every size the fit tries, so the card is priced for the schemes that run. The suite (--physics-profile, or the default at the finest grid) is the base the choices change; no suite is asserted, so a mix no named suite matches runs as written |
| `--physics-profile {morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1,nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-wrf-comparison-candidate-v1,nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-wrf-comparison-candidate-v1,thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1,thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1,thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1,p3-mp50-ysu-mm5-noah-rrtmg-legacy-v1,wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1,wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1,thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1,thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1,wsm6-ysu-mm5-noah-no-radiation-v1,wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1,wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1,wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1,thompson-mp8-mynn-mynn-ruc-dudhia-implemented-unverified-v1,kessler-mp1-ysu-mm5-noah-dudhia-v1,thompson-mp8-mynn-mynn-ruc-monthly-rrtmg-legacy-v1,thompson-mp8-mynn-mynn-ruc-monthly-solar-rrtmg-legacy-v1,thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1,wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1,wsm6-mynn-mynn-noahmp-no-radiation-expert-only-v1,wsm6-mynn-mynn-noahmp-rte-rrtmgp-expert-only-v1,20crv3-wsm6-ysu-mm5-noah-kf-rte-rrtmgp-implemented-unverified-v1,lin-mp2-ysu-mm5-noah-rrtmg-legacy-implemented-unverified-v1,wsm6-ysu-mm5-noah-nsas-rrtmg-legacy-v1,ccpp-tke-edmf-thompson-mm5-noah-gf-rte-v1,milbrandt2mom-mp9-ysu-mm5-noah-ntiedtke-rrtmg-legacy-v1,wdm6-mp16-ysu-mm5-noah-grell-freitas-rte-rrtmgp-v1,thompson-aerosol-mp28-myj-eta-noah-rte-rrtmgp-v1,wsm6-sase-revised-mm5-noah-closure-supplied-v1,wsm6-pbl-off-mm5-noah-tke-1-5-order-v1,wsm6-pbl-off-mm5-noah-smagorinsky-3d-v1,wsm6-pbl-off-mm5-noah-constant-k-v1}` | shipped physics suite to emit; taken verbatim from the registry the prepared-forecast runner validates against, so the emitted config passes its guard as written. Read the resolved radiation selectors: a suite with shortwave ON and longwave OFF is a daytime-only experiment; selecting it for a window that includes local night is REFUSED unless you declare it yourself with --ack. NOT every profile runs on every route: --source gem-gdps cannot prepare wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1 or thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1 or wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1 or wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1 or thompson-mp8-mynn-mynn-ruc-dudhia-implemented-unverified-v1 or thompson-mp8-mynn-mynn-ruc-monthly-rrtmg-legacy-v1 or thompson-mp8-mynn-mynn-ruc-monthly-solar-rrtmg-legacy-v1 or thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1 -- the wizard refuses those pairings and names the missing component rather than emitting a config the front door would reject. (--source era5, the default source, binds morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1; a run whose finest grid is under 1 km binds thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1 instead, on every source whose route admits it; every source has its own computed default and its own admissible set -- `woof run-plan --physics-profiles` prints the whole table) |
| `--point LAT,LON` | domain center in decimal degrees. \|lat\| 90 is refused. A point carries no extent, so the fit chooses one: the largest layout the budget affords, capped at --point-extent-km per axis, kept within the projection envelope, and kept to less than one trip around the globe in longitude. These caps SHRINK the domain rather than refuse it, and the plan summary states which one bound; the pole refusal is left for a center so close to one that even the smallest layout contains it. Draw a --polygon to ask for more ground than the cap. The projection is auto-selected from \|lat\| (<25 Mercator, 25-60 Lambert conformal, >60 polar stereographic) unless --projection is set. Negative (southern/western) values work in both forms: --point -33.87,151.21 and --point=-33.87,151.21 |
| `--point-extent-km KM` | largest root extent a --point request is sized to, on the longer root axis (x); the root keeps the ladder's 5:4 shape, so y is 0.8 of it (default 6000). The projection pole, one trip around the globe, the source's coverage and the card still bound the fit; an extent below the smallest root the ladder hosts gets that root. The plan summary states the extent used |
| `--polygon GEOJSON` | local GeoJSON Polygon, MultiPolygon, Feature, or FeatureCollection; the minimum antimeridian-aware bounds supply the center and every emitted level is fitted around the geometry |
| `--projection {auto,lambert,mercator,polar}` | map projection override (default: auto by center latitude; all three are oracle-gated against WRF v4.6.1 module_llxy) |
| `--quicklook-interval SECONDS` | a quick-look stream on the FINEST domain: a small subset (--quicklook-vars) every SECONDS between its full frames, written to quicklook/wrfout_dNN_<time>. Must divide that domain's history interval. Measured 13 B a cell against 172 for a full frame |
| `--quicklook-vars VAR[,VAR...]` | the quick-look subset's wrfout variables (default: REFL_10CM, UP_HELI_MAX, U10, V10, T2, Q2, PSFC, RAINNC, RAINC) |
| `--recipe ID\|TOML` | complete recipe from the packaged table or a local TOML, laid out on --region (or --polygon) at --spacing-km; a packaged row given neither uses its declared default region and spacing (hrrr_v4_gsd41: US at 3 km), stated in the output and the receipt |
| `--recipe-table` | recipe registry override; new recipes and aerosol policies are data |
| `--region ID\|GEOJSON` | with --recipe: region row id (e.g. US) or local polygon GeoJSON the recipe is laid out on; requires --spacing-km |
| `--region-table` | region JSON table override; adding a region requires data only |
| `--root-dx KM` | custom root grid spacing in km [0.05, 200]; use with --chain instead of --ladder |
| `--sf-surface-mosaic {0,1}` | Noah land-use tiles on every grid (WRF sf_surface_mosaic) |
| `--smoke` | with --recipe: turn on HRRR-Smoke (transported smoke, its radiative feedback, RAVE fires); in 2.9.4 it costs about 11x per step on full CONUS (measured on 8x RTX 5090) |
| `--source SOURCE` | forcing source: any registered source id or alias -- hrrr, hrrr-prs, gem-gdps, icon-global, icon-eu, icon-d2, gfs, gdas, gefs-v11-flat, gefs-reforecast, gefs-reforecast-weekly, gefs-v11, gefs, aigfs, aigefs, ecmwf-open-data, ecmwf-open-data-0p1, ecmwf-ens, aifs, earth2, rap-native, hrrr-native, rap, rrfs, rrfs-ens, era5, era5-l137, 20crv3, 20crv3-cf, earth2-fcn3, earth2-sfno, earth2-pangu, earth2-aurora, earth2-fengwu, earth2-aifs, earth2-fuxi, earth2-stormcast, earth2-atlas, earth2-atlas-si, earth2-aifs2, earth2-aifs-ens, earth2-aifs2-ens, earth2-aurora15, earth2-aurora15-6h, earth2-aurora15-ensemble, earth2-aurora15-ensemble-6h, earth2-gencast-mini, earth2-weathernext2, earth2-weathernext2-mini, earth2-ucast, earth2-graphcast-operational, earth2-graphcast-small, earth2-stormcast-conus, earth2-atlas-crps today, plus hex-global, mapped with --cadence naming the spacing of the history files it will be handed (`woof prep --list-sources` lists the whole registry). It sets the boundary cadence written into the companion namelist.wps, bounds the domain by the source's own grid where that grid is regional, and (era5) declares [case_data]. A source `woof fetch` cannot download still emits the same geometry: one whose registry row declares a local input contract gets a [fetch] table (source, cycle, hours and its staging source_root) with the staging step named beside it, and any other has the acquisition step named in place of the table |
| `--spacing-km` | with --recipe: grid spacing in kilometres; required whenever --region or --polygon is given with --recipe |
| `--target-host-memory-json` | selected target host-memory snapshot for an explicit --card or --vram-gib budget; no local RAM sizing |
| `--terrain-smoothing SPEC` | WPS terrain smoothing per domain, in domain order, the last repeating: none, 1-2-1, smth-desmth or smth-desmth_special, each with an optional :PASSES (e.g. none or smth-desmth_special,none); default: WPS's one smth-desmth_special pass |
| `--terrain-smoothing-precision {float64,wps-float32}` | arithmetic of every domain whose terrain smoother is WPS's default smth-desmth_special x1: wps-float32 reproduces geogrid.exe's HGT_M exactly; default float64, WOOF's own smoother. Every other smoother always runs WPS's float32 |
| `--tiles {off,auto,on}` | streaming mode (bare --tiles means auto); sizes with the forecast planner using the selected target's GPU and RAM when supplied, otherwise local hardware or an explicit card budget; on forces streaming |
| `--vram-gib N` | total VRAM in GiB (alternative to --card) |
| `--vtable` | era5: Vtable override (default: the packaged Vtable.ERA5_CDO, copied beside the TOML) |

## `woof domain-fit`

| argument | what it does |
|---|---|
| `template` | ordinary complete experiment TOML |

| option | what it does |
|---|---|
| `--buffer-km` | one polygon buffer, or one per domain in the template's parent-before-child order |
| `--card` | GPU to size for: a tier (12gb/16gb/24gb/32gb), a size ('10gb') or a model with a recorded size ('RTX 3080') |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--hardware-json` | selected node hardware snapshot with measured capacity, available memory and device profile |
| `--hours` | explicit new duration; otherwise preserve template |
| `--out` | new ordinary TOML path |
| `--point` | center LAT,LON; fit largest centered layout |
| `--point-extent-km KM` | largest root extent a --point fit is sized to, on the longer root axis (x); the root keeps its 5:4 shape, so y is 0.8 of it (default 6000); the projection pole, one trip around the globe, the source's coverage and the card still bound it, and an extent below the template's smallest root gets that root |
| `--polygon` | GeoJSON area; preserve its entire footprint |
| `--root-dx KM` | explicit root cell spacing in km; scale the domain tree's spacing while preserving physics, vertical levels and clocks |
| `--source` | input source, required only without [fetch].source |
| `--start-time` | explicit new UTC start; otherwise preserve template |
| `--target-host-memory-json` | selected target host-memory snapshot for an explicit --card or --vram-gib budget |
| `--vram-gib` | target total VRAM capacity in GiB; omit device flags to detect this machine's GPU |
| `--write` | write the reviewed TOML, the fit receipt, and every file its input route reads beside it |

## `woof domain-tiles`

| argument | what it does |
|---|---|
| `template` | ordinary complete experiment TOML |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--mode {auto,on}` | auto streams when needed; on forces planner-selected streaming |
| `--out` | new ordinary TOML path |
| `--write` | create the reviewed TOML, the tile receipt, and every file its input route reads beside it |

## `woof downscale`

`--parent-namelist` (with `--parent-namelist-domain`) is the entire stock-WRF-parent route this command's own summary advertises: without it, only a woof parent run can be downscaled.  `--tiles {on,auto}` and `--child-size` are the only way to stream a `--point`-derived child, because this command authors the child TOML itself and a `[tiles]` table you wrote by hand would be overwritten.  A child builds its own terrain, land use and soil at its own spacing by default; `--parent-terrain` keeps its parent's, interpolated, and `--geog-root` names the WPS_GEOG tree the child is built from.

| argument | what it does |
|---|---|
| `parent [parent ...]` | parent wrfout directory or explicit history files |

| option | what it does |
|---|---|
| `--accept-parent-cadence` | accept the archive's own cadence as the ceiling (prints the 15-min guidance when coarser); mutually exclusive with --max-boundary-interval-seconds |
| `--auto-vram` | measure local total AND free GPU memory and price the child on it: fits the extent when --child-size is absent, prices the given extent or child config otherwise; exclusive with --card and --vram-gib |
| `--card` | card for --point sizing (default 24gb): a tier (12gb/16gb/24gb/32gb), a size ('10gb') or a model with a recorded size ('RTX 3080'), the same spellings `woof domain` accepts |
| `--child-config` | legacy RunConfig TOML for the child (specified=true, nested=false) |
| `--child-config-sha256` | require the exact reviewed child configuration bytes |
| `--child-levels N[,STRETCH]` | give the child its own vertical ladder of N levels instead of inheriting the parent's, clustered toward the ground by STRETCH (the LES case: a 100 m child wants the levels, not just the columns). p_top, hybrid_opt and etac stay shared with the parent |
| `--child-size NX[,NY]` | explicit child extent for --point |
| `--child-surface-from` | child-grid wrfinput/history file with land identity + soil warm start (required for surface-physics children) |
| `--dry-run` | validate contracts, derive/print the plan, write the derived TOML, run nothing |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--follow {storm,parent,fixed,target}` | step down inside a moving parent and follow: 'storm' keeps the child on the storm (the parent run's storm track, else the moving parent's own centre), 'parent' keeps its place in the moving parent, 'fixed' holds it on the ground, 'target' follows a severe-weather setup (--follow-target). Needs --child-size; --point is where it starts (default: on the storm, else the moving parent's centre) |
| `--follow-boundary-field {dewpoint,theta_e,temperature}` | --follow-target boundary: 2 m dewpoint (dryline, the default), theta-e or temperature (fronts) |
| `--follow-every SECONDS` | how often a following child is re-placed (default 3600, a whole number of the parent's frame interval) |
| `--follow-field NAME` | --follow-target object: the frame field (default REFL_10CM, column maximum; UP_HELI_MAX for rotation) |
| `--follow-path FILE` | --follow-target path: timed points, JSON {"rows":[{"time","lat","lon"}]} or CSV valid_time,lat,lon |
| `--follow-segment LAT,LON,LAT,LON` | --follow-target boundary: the dryline or front as drawn at the start |
| `--follow-target {low,cape-shear,boundary,object,path}` | with --follow target: what the child follows, fixed on the parent's own frames: a surface low, the dominant MLCAPE x 0-6 km shear area, a dryline/front (--follow-segment), a thresholded feature (--follow-field), or the user's own timed path (--follow-path; not a tracker) |
| `--follow-threshold` | the target's threshold (cape-shear index 0.25, boundary 4 K/100 km, object 35 dBZ, low 1 hPa) |
| `--geog-root` | the WPS_GEOG tree the child's own static geography is built from (default: the child config's [static] geog_root, else the staged tree `woof fetch-geog` installs) |
| `--health-interval-seconds` | model seconds between child health lines (CFL, w_max, NaN check; default 60) |
| `--highres-cache` | the folder the child's high-resolution terrain tiles and land-cover rasters are read from and cached in (default: the child config's [static.highres] cache_root, else the engine's per-user cache). A child at 600 m or finer takes land cover and soil at source resolution (CGLC-MODIS-LCZ, SoilGrids) as well as Copernicus terrain, so point this at a cache that already holds the land-cover raster |
| `--hours` | --point run window in hours (default: the full parent archive window) |
| `--i-parent-start` | 1-based west-east parent index of the child's southwest corner (required with --child-config; --point derives it) |
| `--icechunk-store URL` | Output store base: local path or s3://bucket/prefix; domain and member paths are appended (default RUN/icechunk) |
| `--j-parent-start` | 1-based south-north parent index of the child's southwest corner (required with --child-config; --point derives it) |
| `--keep-checkpoints N` | how many complete checkpoint sets the child keeps in --out (default 1, the newest, which a downscale from this child binds to); 0 keeps every set |
| `--les-below-m METRES` | a --point child at or below this spacing runs the LES closure (PBL off, a 3-D closure, isotropic mixing) on top of its suite (default 250 m, the same rule `woof domain` writes on nests; 0 never) |
| `--les-closure {smagorinsky,tke}` | the LES child's 3-D closure: smagorinsky (km_opt 3, default) or tke (km_opt 2) |
| `--max-boundary-interval-seconds` | explicit ceiling on acceptable parent cadence (the scientific cadence contract); mutually exclusive with --accept-parent-cadence |
| `--no-memory-gate` | run a child whose priced peak envelope exceeds this card's free memory anyway, as `woof go --no-memory-gate` does: the envelope is an upper bound and the card's own allocation then decides; a child state too big to build at all is still refused |
| `--out` | create-only output directory for the child run (report.json, wrfout frames, restart) |
| `--output-config TOML` | Read [output] formats and icechunk_store from TOML; CLI values override it |
| `--output-format FORMATS` | Output-format table rows, comma separated (default netcdf); netcdf,icechunk adds atomic native Zarr v3 history |
| `--output-interval-seconds` | --point child history cadence (default: the parent cadence) |
| `--parent-domain` | parent domain id when the directory carries several (e.g. 3 for the innermost archived parent) |
| `--parent-namelist` | stock-WRF namelist.input of the parent run |
| `--parent-namelist-domain` | domain column of --parent-namelist (default 1) |
| `--parent-restart PATH\|latest` | woof restart of the parent run (authoritative physics evidence); 'latest' discovers the newest complete checkpoint set in the parent's own run directory |
| `--parent-terrain` | run the child on its parent's interpolated terrain, land use and soil. By default a child builds its own static geography at its own spacing (terrain, land use, soil; Copernicus 30 m terrain at 1 km or finer, CGLC-MODIS-LCZ land cover and SoilGrids soil at 600 m or finer) and the parent's state is blended and rebalanced onto it as WRF's ndown does |
| `--physics-profile ID\|parent` | the --point child's physics suite, a `woof domain --physics-profile` id, or 'parent' to keep the parent's physics verbatim (default: thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1, the HRRR-shape suite, for a child finer than 4 km; the parent's physics otherwise) |
| `--point LAT,LON` | derive the child around this point instead of --child-config (woof parents only) |
| `--preprocess-backend {cuda,cpu,auto}` | where the parent-to-child interpolation runs (default auto: on the card when its priced interpolation fits the card's free memory, else on the CPU; cuda refuses rather than move it; cpu runs it off-GPU) |
| `--ratio` | refinement ratio (child-config placement: required; --point default 3) |
| `--render-products LIST` | which products the child's frames are drawn into <out>/png, each as it is written: a comma-separated list of catalog slugs, 'all' (the default -- the renderer's whole catalog), or 'none' to keep only the frames. The same spelling `woof render --products` and `woof go --products` take |
| `--tiles {on,auto}` | write [tiles] mode into the config --point derives, so the child integrates out of a pinned host store instead of resident ('on' always, 'auto' when tilestream.autoplan says it does not fit) |
| `--vram-gib` | explicit VRAM capacity for --point sizing |
| `--warm-from CHECKPOINT` | start from the parent as usual, but carry this earlier child's state onto the ground both share (the seam a following child's segments join on); needs --warm-shift |
| `--warm-shift DI,DJ` | this child's placement minus the earlier one's, in child cells |
| `--warm-strip NPZ` | accumulated precipitation for the freshly covered strip of a warm start, per slot |

## `woof downscale-parent`

| argument | what it does |
|---|---|
| `parent_run_dir` | a run directory (or folder of wrfout frames) to read |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--parent-domain` | the domain to read; default: the finest domain the run wrote |

## `woof dual-run`

| option | what it does |
|---|---|
| `--capsule-a CAPSULE` | _(the parser declares no help text for this option)_ |
| `--capsule-b CAPSULE` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--out-report REPORT` | write the comparison document here |

## `woof earth2`

| option | what it does |
|---|---|
| `--accept-licence LICENCE` | explicitly accept a model row's exact restricted/unresolved weight licence; repeat for multiple licences |
| `--area` | south,west,north,east; without --nest, author a 3 km regional domain |
| `--card` | planning card name or capacity, passed to domain/cyclone-setup |
| `--cycle` | explicit initialization in UTC, YYYY-MM-DDTHH |
| `--diagnose` | diagnostic table rows: derived-ws10m, windgust |
| `--dry-run` | check the job against every refusal the real run applies and write its canonical JobSpec, without fetching or running |
| `--earth2-input` | reuse a completed fetch directory after checking model, cycle, horizon, coverage and artifact hashes; auto placement searches a regional input's area |
| `--ensemble` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--export {earth2}` | _(the parser declares no help text for this option)_ |
| `--geog-root DIR` | geography tree the regional or nest preparation reads (default: the staged WPS_GEOG tree); a dry run checks the named tree |
| `--hours` | _(the parser declares no help text for this option)_ |
| `--ic-source` | initial conditions: era5, gfs, hrrr, ifs |
| `--mix` | parent sources and counts, e.g. source:18,source:18 |
| `--model` | table id or alias |
| `--nest` | inner grid spacing, e.g. 2.5km |
| `--out` | _(the parser declares no help text for this option)_ |
| `--spec` | _(the parser declares no help text for this option)_ |
| `--spec-out` | _(the parser declares no help text for this option)_ |
| `--storm` | auto, ATCF id or storm name; auto ranks cycle strength within --area or the configured basin |

## `woof earth2-diagnose`

| argument | what it does |
|---|---|
| `[store]` | consumer export, alias of --input |

| option | what it does |
|---|---|
| `--accept-licence LICENCE` | explicitly accept a model row's exact restricted/unresolved weight licence; repeat for multiple licences |
| `--device` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--for` | comma-separated consumer rows |
| `--input` | _(the parser declares no help text for this option)_ |
| `--model` | derived-ws10m, windgust |
| `--out` | _(the parser declares no help text for this option)_ |

## `woof enprod`

| argument | what it does |
|---|---|
| `[ENS_ROOT]` | ensemble root holding member_NNN/ run directories and ensemble-manifest.json (schema gpuwm-ensemble-manifest.v1), or the run folder of woof go --members --keep-member-files (schema gpuwm-ensemble-output.v2, one manifest per domain) |

| option | what it does |
|---|---|
| `--accept-status LIST` | comma-separated manifest member statuses to accept (default DONE,complete); any other status is a refusal naming the members |
| `--domain dNN` | which domain to plot when members hold more than one (default: the single domain present, else a refusal) |
| `--dpi N` | PNG resolution (default 150) |
| `--engine {auto,rust,matplotlib}` | which renderer draws the panels (default auto). The render law puts weather fields on the Rust renderer; 'auto' uses rw_ensbatch when this checkout has built it and falls back to matplotlib with the reason named, 'rust' refuses rather than substituting, and 'matplotlib' selects the fallback outright |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--field LIST` | comma-separated product fields: visibility, fog_prob, refl, uh, precip, t2, wspd10, or 'all' (default refl; 'refl,uh' is the severe-convective pair) |
| `--make-fixture DIR` | write a synthetic ensemble (members + manifest) to DIR and exit, for exercising the suite without a real ensemble |
| `--members N` | --make-fixture member count (default 5) |
| `--nan-policy {mask,refuse}` | what to do with a non-finite member value -- NaN or +/-Inf (default mask): 'mask' excludes it from every reduction at that point, shrinks the denominator with it, and stamps the resulting coverage on the panel; 'refuse' fails the whole product naming the members |
| `--neighborhood-km LIST` | comma-separated neighborhood radii in km for the probability product (default 0 = point probability). Each member uses its maximum (minimum for visibility) within the radius before the ensemble fraction is taken |
| `--out DIR` | output directory for the PNGs (default out/enprod) |
| `--pmm-tie-rule {flat-index,average}` | how the probability-matched mean resolves equal means (default flat-index): 'flat-index' is Ebert's algorithm exactly and paints a deterministic but meaningless row-major gradient across a plateau; 'average' gives every point in a tie the group's mean intensity and gives up the exact pooled distribution |
| `--products LIST` | comma-separated products: mean, spread, prob, paintball, pmm, or 'all' (default) |
| `--source-label TEXT` | model/provenance label stamped on every plot (default WOOF) |
| `--threshold LIST` | comma-separated thresholds in the field's own units; default is the field's own (refl 40 dBZ, uh 75 m2 s-2; visibility strictly below 1000 and 400 m). Every threshold gets its own probability and paintball plot |
| `--timeidx N\|all` | index into the valid times every member shares, or 'all' (default) |

## `woof ensemble`

| argument | what it does |
|---|---|
| `[CONFIG]` | an experiment TOML from woof domain; its source and domain tree choose the preparation route |

| option | what it does |
|---|---|
| `--accept-licence LICENCE` | explicitly accept a model row's exact restricted/unresolved weight licence; repeat for multiple licences |
| `--branch-from PARENTS` | branch members from saved checkpoints in a parent-list JSON |
| `--branch-hours` | forecast hours after each parent checkpoint (default 1) |
| `--branch-surface` | also apply the WoFS seeded soil-moisture and SST draws after restore |
| `--card` | target card for domain sizing, e.g. 16gb |
| `--cards all\|0,1` | cards for checkpoint branches: all visible cards or visible ordinals |
| `--center` | regional domain center LAT,LON |
| `--cycle YYYY-MM-DDTHH\|latest` | download routes only: run the config at this cycle; wins over the config's [fetch] cycle the way --transport does. The config is re-timed (start time, delayed nests, namelists) into <outdir>/cycles/<cycle>/; latest is resolved once, under the run's posting rule (as posted: the newest cycle whose start needs are posted) |
| `--cyclone-products JSON` | JSON common_grid and optional grid_id/products; ordinary go accumulates max-wind, gust, min-pressure and track at every committed step |
| `--data-dir DIR` | download routes only: use this existing download instead of the automatically managed request cache |
| `--devices N` | resident slab count; replaces [devices] count |
| `--dry-run` | validate the route and show how to launch it; fetch and run nothing |
| `--ensemble-seed` | the member seed root; the same root draws the same members |
| `--era5-provider {arco,cds}` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--extent-km` | domain width when creating a domain (default 900 km) |
| `--family-weights ai=0.5,physics=0.5` | family probability masses, must sum to 1; defaults to member shares |
| `--geog-root DIR` | override the geography tree (default: the config's [case_data].geog_root or [static].geog_root, otherwise the staged WPS_GEOG tree) |
| `--grib2` | write native surface and pressure-level GRIB2 as each frame lands, then <run>/grib2-grib2.zip |
| `--history-compression {deflate,none}` | history frame storage: deflate (default; lossless NetCDF-4, the same values in about a third of the disk) or none (classic CDF-2). Also GPUWM_HISTORY_COMPRESSION |
| `--history-interval SECONDS` | history frame interval when creating a generated-source domain |
| `--hours` | forecast duration when creating a domain (default 3 h) |
| `--hydro on\|off\|SETTINGS.toml` | hydro is opt-in: --hydro on couples NextGen hydro wherever a hydrofabric covers the domain; a TOML file selects settings; without it the forecast has no hydro |
| `--icechunk-store URL` | Output store base: local path or s3://bucket/prefix; domain and member paths are appended (default RUN/icechunk) |
| `--init` | initial UTC analysis/cycle, YYYY-MM-DDTHH |
| `--initial-inputs JSON` | a separate packaged analysis for the initial state (woof prep --initial-inputs); the fetched source still supplies every lateral boundary, and the forecast starts at the preparation head while the later boundary times are built |
| `--keep-checkpoints N` | how many complete checkpoint sets the run keeps in its folder (default 1, enough to resume); 0 keeps every hourly set, which a later branch or downscale from an earlier checkpoint needs |
| `--keep-member-files` | also retain every member's full history files |
| `--late-after-minutes MIN` | download routes only: how far past its scheduled time a lead may be before the run stops with exit 75; wins over the config's [fetch] late_after_minutes and the source row's budget |
| `--live-products-width` | _(accepted, but not listed by --help)_ |
| `--max-lag-hours` | oldest lagged cycle (0 disables lags) |
| `--member ID` | select one generated source trajectory; fetch accepts member lists |
| `--member-sharing {auto,mps,contexts}` | how co-resident checkpoint branches share a card: a private MPS controller for this launch, or one CUDA context per member (default auto: MPS where the controller exists) |
| `--members N` | make an N-member ensemble with aggregate products |
| `--members-per-card auto\|N` | checkpoint branches run at once on each card, one process each (default auto: as many priced members as the card's free memory holds, up to 4) |
| `--nest-at LAT,LON[@TIME]` | spawn a storm-following nest at this point (at TIME, an ISO time or seconds into the run; default the first step); repeatable |
| `--no-memory-gate` | skip the before-launch memory check that refuses a configuration whose binding phase cannot fit this card's free VRAM, and the forecast runner's own check of the same envelope; a model state too big to build at all is still refused |
| `--no-probe` | with --readiness: compute the schedule from the source table only and ask no host |
| `--no-surface-fill` | disable fill soil/SST perturbations |
| `--no-verify-visuals` | skip postforecast observation verification; the physical run is unchanged |
| `--nudging {off,spectral,grid}` | relax toward prepared source analyses; spectral keeps only the configured large-scale modes |
| `--outdir DIR` | output root for one timestamped run folder per launch, with forecast files, pictures and diagnostics (default <config-stem>-go beside the config); an existing run-... folder is used directly |
| `--output-config TOML` | Read [output] formats and icechunk_store from TOML; CLI values override it |
| `--output-format FORMATS` | Output-format table rows, comma separated (default netcdf); netcdf,icechunk adds atomic native Zarr v3 history |
| `--physics-arms {fill,all,off}` | _(the parser declares no help text for this option)_ |
| `--prepare-only` | fetch and stream prepared inputs without starting a forecast |
| `--prepared-root DIR` | run this existing prepared bundle without fetch or preparation; add --restart to continue its checkpoint |
| `--products LIST` | which products the render stage draws: a comma-separated list of catalog slugs, 'all' (the default -- the renderer's whole catalog), or 'none' to stop after the forecast. The same spelling `woof render --products` takes |
| `--publish-cmd COMMAND` | optional delivery command; manifest JSON on stdin, no shell (default: off) |
| `--publish-root DIR` | local static web root (default: run output/www) |
| `--readiness` | print gpuwm.readiness.v1 for the config's fetch window on stdout and run nothing: exit 0 ready, 75 not yet, 2 refused |
| `--readiness-fold` | _(accepted, but not listed by --help)_ |
| `--recipe {wofs,time-lagged,multi-model,surface-state,member-roster}` | take each ensemble member from a real source trajectory: time-lagged runs earlier cycles of the config's own source over the same window, multi-model runs the trajectories --trajectories lists, surface-state runs seeded soil moisture scales and SST offsets from [ensemble.perturbation], member-roster runs named land and fixed surface arms from [ensemble.member_variants] |
| `--restart CHECKPOINT` | continue an existing checkpoint into fresh output; a checkpoint woof go wrote resumes from its own run folder's prepared bundle, other prepared-cache checkpoints also need --prepared-root |
| `--restart-roster JSON` | continue the exact original members from a durable ensemble restart roster |
| `--resume-ensemble RUN` | retry unfinished checkpoint branches in RUN; preserve completed members and failed attempts |
| `--run-stamp {on,off}` | put this run's forecast files, pictures and diagnostics in its own timestamped folder under --outdir (default on): --outdir/run-<YYYYMMDD>-<HHMMSS>Z_i<YYYYMMDD><HHMM>Z/ (launch instant UTC, then the model initialisation time; the _i part is omitted when the run's init time cannot be read). Successive runs of one configuration then never overwrite or interleave each other. 'off' writes straight into --outdir, which is what releases up to 2.4.1 did; it is kept only for a consumer still written against that and is a workaround, not a supported alternative |
| `--section lat,lon,lat,lon\|FILE.json` | the line the vertical-section products (xsec:<fill>[/<overlay>...] in --products) are cut along, the same value `woof render --section` takes; a JSON file gives {start, end} or a {points, extend_km} polyline. Spell a line that starts with a minus sign as --section=-33.9,151.2,-34.1,151.3. An xsec: product with no line is refused before anything is fetched |
| `--smoke` | turn on HRRR-Smoke (transported smoke, its radiative feedback, RAVE fires); in 2.9.4 it costs about 11x per step on full CONUS (measured on 8x RTX 5090) |
| `--smoke-radiation {on,off}` | explicit transported aerosol radiation feedback: on selects live SW/LW optics instead of aerosol climatology and shortwave-only smoke; off disables live feedback. Requires active optical mass rows; omitted preserves the recipe. See docs/solar-products.md |
| `--source SOURCE[:N],...` | create a 3 km WoFS-shape ensemble domain; mixed family counts sum to --members; eligible registry ids: hrrr, hrrr-prs, gem-gdps, icon-global, icon-eu, icon-d2, gfs, gdas, gefs-v11-flat, gefs-reforecast, gefs-reforecast-weekly, gefs-v11, gefs, aigfs, aigefs, ecmwf-open-data, ecmwf-open-data-0p1, ecmwf-ens, aifs, earth2, earth2-fcn3, earth2-sfno, earth2-pangu, earth2-aurora, earth2-fengwu, earth2-aifs, earth2-fuxi, earth2-stormcast, earth2-atlas, earth2-atlas-si, earth2-aifs2, earth2-aifs-ens, earth2-aifs2-ens, earth2-aurora15, earth2-aurora15-6h, earth2-aurora15-ensemble, earth2-aurora15-ensemble-6h, earth2-gencast-mini, earth2-weathernext2, earth2-weathernext2-mini, earth2-ucast, earth2-graphcast-operational, earth2-graphcast-small, earth2-stormcast-conus, earth2-atlas-crps, rap-native, hrrr-native, rap, rrfs, rrfs-ens, era5, era5-l137, 20crv3, 20crv3-cf |
| `--stage-only STAGE` | _(accepted, but not listed by --help)_ |
| `--static-cache-key` | _(accepted, but not listed by --help)_ |
| `--static-cache-root` | _(accepted, but not listed by --help)_ |
| `--storm-nests N\|auto\|off` | storm-following fine nests that spawn on their own: N dormant slots (or 'auto', priced from the card's free memory) watch the parent's 2-5 km updraft helicity, each nest follows its storm and stops when it decays, merges or leaves the domain. Overrides the config's [storm_nests] slots; 'off' removes the automatic slots (docs/storm-nests.md) |
| `--supplement ROLE=PATH` | explicit preparation donor; repeat for multiple files. HRRR accepts PMSL=GRIB inside --data-dir and binds its bytes in the preparation source manifest |
| `--tiles {off,auto,on}` | [tiles] mode for this launch's forecast: 'auto' streams a domain the card cannot hold through host memory, 'on' always streams, 'off' runs resident. A prepared bundle (--prepared-root, --restart) takes it as an execution control, never into its hash-bound config; a fetching launch runs a copy of the config with the mode re-stated, as --cycle does |
| `--trajectories FILE` | the multi-model member list: a JSON or TOML file of {source, cycle[, member]} entries, one per member (selects --recipe multi-model) |
| `--transport {auto,aws,dwd,ecmwf,gdex,google,msc,ncei-0p5,ncei-1p0,nomads,s3}` | download routes only: pin the fetch stage to one host of the source's endpoint ladder, the value `woof fetch --transport` takes; wins over the config's [fetch] transport, and the plan says which it used |
| `--warm-kernels` | _(accepted, but not listed by --help)_ |
| `--whole-cycle` | download routes only: the fetch stage waits for the whole cycle (the old rule) instead of taking each lead as it posts; wins over the config's [fetch] as_posted |
| `--wind-amplitude` | fill uniform U/V amplitude in m/s; 0 disables it |
| `--wind-farms uswtdb\|TABLE\|off` | opt-in Fitch turbine drag and TKE from USWTDB or a WRF/JSON turbine table |
| `--wps-namelist PATH` | with --prepared-root: the exact WPS authority required by a single-domain portable bundle |

## `woof ensemble-products`

| argument | what it does |
|---|---|
| `run` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--dx-m` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--member-map` | _(the parser declares no help text for this option)_ |
| `--mrms-cache` | _(the parser declares no help text for this option)_ |
| `--offline` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |
| `--renderer` | _(the parser declares no help text for this option)_ |
| `--valid-time` | _(the parser declares no help text for this option)_ |

## `woof export`

| argument | what it does |
|---|---|
| `[INPUT ...]` | history files, folders holding them, .gz history files, or ZIPs |

| option | what it does |
|---|---|
| `--append` | add these frames to the export in --out (one frame at a time) |
| `--channels` | additional lexicon channels, comma separated |
| `--chunk-yx SHAPE` | horizontal chunk shape: auto (default; 512x512 tiles for planes over 1024x1024 cells, one chunk otherwise), full, or NY,NX |
| `--config TOML` | the run's configuration; its SHA-256 is recorded |
| `--domains LIST` | e.g. d01,d02 |
| `--earth2 PROFILE` | _(the parser declares no help text for this option)_ |
| `--end TIME` | last valid time kept |
| `--every HOURS` | keep frames on multiples of HOURS from the run's start |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--finalize` | close the export in --out after --append calls |
| `--for CONSUMER[,CONSUMER]` | _(the parser declares no help text for this option)_ |
| `--json` | with --list: the options document; otherwise: progress as JSON lines |
| `--list` | print the level sets, variables and naming schemes |
| `--out DIR` | the export folder (one dNN.zarr per domain, README.txt, receipt) |
| `--overwrite` | replace an earlier export in --out |
| `--regrid {bilinear,area-mean}` | latlon method (default bilinear) |
| `--skip-unavailable` | write the variables the files can make and record the rest |
| `--start TIME` | first valid time kept |
| `--threads N` | worker threads (default: every core) |
| `--zip` | also write <out>-ml.zip (stored entries, opens in place) |

## `woof export-grib2`

| argument | what it does |
|---|---|
| `[INPUT ...]` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--append` | append chronologically ordered frames |
| `--bits` | quantization bits, default 20 |
| `--definitions {woof,renderer,arwen,upp}` | who computes the fields: woof (default, the WOOF post-processor, catalog woof-post/v1); renderer takes the fields the product maps draw from the renderer's own diagnostics (arwen is the same); upp is a one-release alias of woof |
| `--domains` | comma-separated d01,d02,... |
| `--end` | last UTC valid time to write |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--extrema-interval-seconds SECONDS` | also write maximum and minimum updraft helicity and maximum 10 m wind over this window ending at each frame (3600 = hourly), from the running extremes the history frames of the window carry; pass the whole run folder |
| `--fields {standard,surface,pressure,all}` | _(the parser declares no help text for this option)_ |
| `--finalize` | close an incremental export |
| `--gust {auto,similarity,tke}` | surface gust method: auto (default) is the mixed-layer TKE gust when the history carries TKE (TKE_PBL or QKE), else the surface-layer similarity gust (RULINGS 7, scored against ASOS) |
| `--json` | machine-readable options or progress |
| `--levels` | comma-separated hPa levels, default 100 through 1000 by 25 |
| `--list` | show field and packing options |
| `--out` | export directory |
| `--packing {complex,simple}` | _(the parser declares no help text for this option)_ |
| `--post-device {auto,gpu,cpu}` | where products are computed: auto (default) uses a GPU with room, else the CPU |
| `--start` | first UTC valid time to write |
| `--threads` | Rust worker threads |
| `--times` | comma-separated UTC valid times |
| `--upp-control {srw,rapr}` | composite reflectivity level label: srw (default) writes the entire-atmosphere layer 200, rapr writes 10 as RAP and HRRR files do |
| `--winds {grid,earth}` | wind component frame, default grid-relative; 100 m components are always earth-relative |
| `--zip` | write sibling <out>-grib2.zip |

## `woof fetch`

| option | what it does |
|---|---|
| `--accept-inventory-change` | proceed when the live provider inventory yields a different record count than this WOOF was certified against. Without it such a mismatch is a refusal naming both counts; with it the live count becomes the bar and the fetch manifest records the acceptance |
| `--accept-licence LICENCE` | explicitly accept a model row's exact restricted/unresolved weight licence; repeat for multiple licences |
| `--all-levels` | gfs/gdas only: take every isobaric level the product publishes instead of choosing a ladder. On the default NOMADS grib-filter transport this selects every level; with --mode full-file the whole object already carries every level and this declares them all for the decode. Either way level subsetting stays an opt-in bandwidth saver rather than a ceiling on the model top |
| `--analyzed-aerosol` | the preparation reads the source's analyzed aerosol numbers (QNWFA/QNIFA), so a subset fetch selects them too; implied by the native forecast chain when an mp=28 domain asks for the analyzed aerosol |
| `--area LAT0,LON0,LAT1,LON1` | bounding box corners in degrees (order free); allow several degrees of margin beyond the outer domain -- for gfs, 15 deg, so every model lake's nearest source-water donor lies inside the crop (a lake whose nearest donor may lie outside it is counted; `woof domain` suggests areas with this margin built in) |
| `--as-posted` | fetch each lead the moment one host holds all of it, in lead order, writing <out>/posting/ (schedule.json and one fNNN.json per verified lead) as the manifest grows; the default. --cycle latest is then the newest cycle whose first leads are posted. A lead later than its budget stops the fetch with exit 75 and posting/failed.json; a re-run resumes from the fetched prefix |
| `--author-front-door-manifest` | author the front-door input manifest for the fetched series; requires --wps-namelist and --experiment-config (--bridge defaults to the built decoder this install resolves) |
| `--bridge EXE` | built gfs_grib2_bridge executable; omit it and the same resolver `woof go` uses finds the one this install has (checkout build, libexec, then ~/.woof/bridges -- see woof doctor) |
| `--cache-dir DIR` | --engine rust only (hrrr, gfs/gdas --mode full-file): wx-core disk cache root, keyed by URL and byte range, so a re-run or an overlapping window re-reads bytes instead of re-downloading them |
| `--cadence HOURS` | forecast-hour cadence: gfs any positive whole-hour spacing whose requested leads are published (default 3); gdas any whole number of hours that divides --hours, on its hourly f000..f009 ladder (default 1; --hours 0 is a single lead, which a cadence has nothing to space); era5 any positive whole number of hours that divides --hours (default 6; the EDA product publishes 3-hourly, so it takes multiples of 3); hrrr is hourly. On a table route any whole number of hours at which the cycle's ladder publishes every lead of the window and the preparation takes the series, default the row's own -- a cadence off the publisher's ladder refuses and names the ladder |
| `--cycle YYYY-MM-DDTHH\|latest` | model cycle (UTC); 'latest' resolves the newest cycle this source can serve, from the initialization grid and publication lag its registry row or route declares -- probed against the mirrors where the source publishes objects to probe, and taken from the declared lag where it does not (a reanalysis published on a delay has a latest, and it is that delay). A source that declares neither is refused by name |
| `--earth2-batch-root` | shared generated-member batch folder |
| `--earth2-batch-size` | members per admitted GPU sub-batch |
| `--earth2-members` | complete comma-separated Earth-2 noise-id roster |
| `--earth2-weights-package` | local pinned Earth2Studio weight package |
| `--engine {auto,rust,python}` | hrrr, and gfs/gdas --mode full-file: which downloader moves the bytes. 'rust' is the vendored rw_fetch backbone (parallel range GETs, .idx coalescing, the cross-process NOMADS rate governor, a disk cache); 'python' is the stdlib transport and always works; 'auto' (default) uses the backbone when it is built |
| `--era5-product {reanalysis,ensemble_members}` | ERA5 product: reanalysis (default), or ten-member EDA with explicit --member 0..9 --retrieve, on a cadence that is a whole multiple of its three-hourly clock |
| `--era5-provider {cds,arco}` | ERA5 provider: cds uses Copernicus credentials; arco downloads Google's public hourly ERA5 Zarr archive without a key |
| `--experiment-config TOML` | the experiment TOML the front door will consume |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--fetch-max-connections N` | total engine fetch connections on this box (default 30, or WOOF_FETCH_MAX_CONNECTIONS). Shared across engine processes; use the same cap for every process. File and range widths are clipped to it; receipts record the cap and stream limits. |
| `--fetch-workers N` | how many FILES are in flight at once (default 10; every source but era5, which is a manual CDS retrieval). Bounded per host on top of the pool: NOMADS is capped at 2 in-flight requests and every request still passes the node-wide 2.5 s spacing governor, so concurrency overlaps service time without raising the request rate against a fragile public host. Every file keeps the exact serial verification -- envelope walk, record bar, sha256 -- and one failed file still refuses by name. 1 is the serial transport: a knob, not a workaround. The manifest receipts files, bytes, workers, wall and the effective speedup under 'concurrency' |
| `--force-refetch` | move every existing file in --out aside (nothing is deleted) and re-download this request. The receipts go first -- fetch-manifest.json, SHA256SUMS, the series -- so an interrupted force can never leave a manifest behind claiming payloads it has already replaced; then payloads, .idx indexes, stale parts and anything else in the directory. Files already set aside by an earlier quarantine are left untouched, and subdirectories are yours. Required when re-fetching a different area/cycle into the same --out |
| `--forecast-start-hour K` | every forecast source: the forecast lead the window BEGINS at (default f000, the analysis). --hours stays the window length, so --forecast-start-hour 174 --hours 66 fetches f174..f240 and nothing before it; an experiment whose start_time is cycle+K is then initialized from f{K} with its boundaries from f{K+i}. With --author-front-door-manifest on an already-fetched --out, this authors the manifest over that tail of the existing series instead of re-downloading it |
| `--hours N` | forecast window length: hours 0..N are fetched. gdas is certified for fetch and decode through f009; native mapped GDAS preparation uses the complete pressure ladder and specific humidity. --hours 0 is one analysis on each acquisition route; it is also how a hybrid source's donor is fetched. Forecast preparation still needs at least two forcing times. A window past the cycle's own horizon refuses and names both the horizon and which cycles reach farther |
| `--late-after-minutes MIN` | how far past its scheduled time a lead may be before the fetch stops with exit 75 (default: the source table's posting late_after_minutes) |
| `--manifest-out JSON` | manifest path (default <out>/gfs-input-manifest.json) |
| `--member ID` | ERA5 EDA: required encoded member 0..9. Ensemble routes (gefs, aigefs): which member to fetch (default the control). Member identity is a PATH component for these products, so the files land under their declared upstream-relative paths and `woof-member-prep --inputs` reads the directory as published |
| `--mode {auto,full-file,idx-subset}` | the byte transport. hrrr: 'auto' is the default -- the .idx record subset the preparation reads (about 40 % of the bytes), and the whole object for a file whose .idx is absent, malformed, or provably shorter than the object, each file's line saying which; 'full-file' takes every object whole (--engine rust); 'idx-subset' takes only subsets and refuses rather than degrading when the index cannot carry the selection (--engine rust). gfs/gdas: 'full-file' takes the whole pgrb2.0p25 objects from the S3 archive (either engine); 'idx-subset' takes from those objects only the records the grib-filter asks for, by their .idx byte ranges (an object whose host keeps no index moves whole); omitted or 'auto', the NOMADS grib-filter crop remains the default (the archive's records for a cycle the crop host no longer keeps, and under the hosted AWS policy). The other forecast sources take whole objects, and 'auto' there is that same full-file default |
| `--model MODEL` | --source earth2: which NVIDIA Earth-2 model to run through earth2studio (fcn3, sfno, pangu, aurora, fengwu, graphcast, aifs: a row of woof/authorities/earth2-models.v1.json); the forecast and the ERA5 analysis at --cycle are written to --out |
| `--no-probe` | with --readiness: compute the schedule from the table only and ask no host |
| `--out DIR` | output directory (created; complete files are skipped on re-run) |
| `--p-top-pa PA` | gfs/gdas only: the model top (Pa) the fetched atmosphere must reach. The pressure ladder is extended upward along whatever the live inventory publishes until a level sits at or above it, so --p-top-pa 5000 fetches the 70 and 50 hPa levels the certified 100 hPa ladder stops short of. Omitted, the certified 21-level ladder is fetched exactly as before (a 10000 Pa source top). woof go, run-plan and the desktop pass the config's own [shared].p_top when the certified ladder stops below it. A top the product cannot serve refuses and names the deepest it can |
| `--point LAT,LON` | center point; requires --radius-km |
| `--prepare-only` | Earth-2 sources: run only the CPU stage (weights, CPU model construction, initial conditions, donor) with no GPU, so it can run outside a card lease; the same command with a card then generates from it. Implied when CUDA devices are hidden |
| `--radius-km KM` | half-width of the box around --point |
| `--readiness` | print gpuwm.readiness.v1 for this window on stdout and fetch nothing: exit 0 ready (or nothing to probe), 75 not yet (with expected_ready_at and retry_after_seconds), 2 refused (the window can never start) |
| `--retrieve` | ERA5: download and validate with the selected provider (default CDS); otherwise write a CDS retrieval template |
| `--reuse-fetch DIR` | reuse a verified completed generated fetch covering this request before loading its model |
| `--runtime-surface FIELDS` | comma-separated runtime surface fields (e.g. VEGFRA) the start must read from this cycle itself. A field the source table declares but this list omits starts from its recorded fallback when the cycle publishes no record for it; a listed field the cycle does not publish is refused by name |
| `--source MODEL` | public data source: 20crv3-cf, aifs, aigefs, aigfs, earth2, earth2-ace2, earth2-aifs, earth2-aifs-ens, earth2-aifs2, earth2-aifs2-ens, earth2-atlas, earth2-atlas-crps, earth2-atlas-si, earth2-aurora, earth2-aurora15, earth2-aurora15-6h, earth2-aurora15-ensemble, earth2-aurora15-ensemble-6h, earth2-corrdiff, earth2-derived-ws10m, earth2-dlesym, earth2-dlwp, earth2-fcn, earth2-fcn3, earth2-fengwu, earth2-fuxi, earth2-fuxi-s2s, earth2-gencast-mini, earth2-graphcast-operational, earth2-graphcast-small, earth2-interpmodafno, earth2-pangu, earth2-sfno, earth2-stormcast, earth2-stormcast-conus, earth2-ucast, earth2-weathernext2, earth2-weathernext2-mini, earth2-windgust, ecmwf-ens, ecmwf-open-data, ecmwf-open-data-0p1, era5, gdas, gefs, gefs-reforecast, gefs-reforecast-land, gefs-reforecast-weekly, gefs-v11, gefs-v11-flat, gem-gdps, gfs, hrrr, hrrr-native, hrrr-prs, icon-d2, icon-eu, icon-global, ncep-r1-seaice, rap, rap-native, rrfs, rrfs-ens. Registry aliases work too (gdps, ifs, hrrr-wrfprs). A registered source with no public bytes -- the 20CRv3 every-member archive, the generic 'mapped' adapter -- refuses by name and points at `woof prep --source-root` |
| `--static-input NPZ` | optional prebuilt static cache (with --static-receipt); omit when the front door builds statics from --geog-root |
| `--static-receipt JSON` | receipt for --static-input |
| `--transport {auto,aws,dwd,ecmwf,gdex,google,msc,ncei-0p5,ncei-1p0,nomads,s3}` | pin one rung of the source's endpoint ladder. Every NCEP source declares an ORDERED ladder -- the operational server (nomads.ncep.noaa.gov) while it still holds the cycle, the AWS archive behind it -- and the default walks it. Retention decides which rungs are asked: a cycle older than the operational window goes straight to the archive. Throughput decides which one serves: each requested object is HEADed on the archive first and taken there when the archive already has it, because the operational server's head start is spent once both hosts have the same bytes; an object the archive has not caught up with comes from the operational server. A refusal, a 403/503 or a Retry-After moves to the next rung either way. Where a source's ladder carries both hosts they serve byte-identical objects under identical keys, so the choice never changes the data, except for AI-GEFS: NOMADS marks each member with ensemble type 6 where the AWS copy of the same member says 3, the AWS surface files carry an extra surface pressure record, and the AWS pressure-level files are repacked copies whose heights sit within 0.08 gpm of the NOMADS ones. Preparation reads AI-GEFS from either host the same way and derives surface pressure itself on both. Naming a host here is a decision: it skips the probe, disables fall-through, and refuses in that host's own words. A host a source does not carry refuses and lists the ones it does, because for some products the second copy is a DIFFERENT product (see `woof fetch --source aigfs`) |
| `--validate GRIB` | era5 only: validate user-supplied GRIB1 file(s) against what woof ingest expects instead of fetching |
| `--wait-for` | the same as --as-posted, for every source |
| `--wait-timeout-minutes MIN` | a cap on the whole window, for every source: exit 75 if the window is not all in by then, keeping the fetched prefix |
| `--whole-cycle` | the old rule: --cycle latest is the newest cycle whose final lead is posted, and a named cycle needs its final lead before anything moves |
| `--wif` | stage the SHA-256-verified monthly aerosol climatology in its shared cache before forcing transfer; implied by the native forecast chain when its selected physics needs that dataset |
| `--wps-namelist WPS` | the namelist.wps the front door will consume (e.g. the woof domain output) |

## `woof fetch-bridges`

| option | what it does |
|---|---|
| `--artifact NAME` | with --partial, select this artifact (repeatable); renderer launchers also fetch their shared rw_wrfbatch executable |
| `--dest DIR` | stage into DIR instead of this release's own ~/.woof/bridges/<release>-<digest> (woof finds the default on its own; anywhere else needs the per-artifact environment variables) |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--from DIR` | stage from a local directory instead of downloading (offline installs): either the bundle archive or the artifacts loose in it; verification is identical |
| `--keep-bundle` | keep the verified archive under <dest>/.fetch-bridges after staging (default: remove it) |
| `--list` | print the platform, bundle and per-artifact staged state, then exit without touching the network |
| `--partial` | fetch only missing or stale pinned binaries with HTTP byte ranges; map assets require a full fetch and servers must support ranges |

## `woof fetch-geog`

| option | what it does |
|---|---|
| `--allow-upstream-drift` | legacy spelling; refused by the native production route because unpinned geography changes prepared fields under the same dataset version |
| `--bundle` | fetch NCAR's single geog_high_res_mandatory.tar.gz (2.6 GiB) instead of the per-dataset tarballs and extract the requested datasets from it (fallback; NCAR only) |
| `--config FILE` | resolve only the geography this experiment's domains read; includes selected published static sources |
| `--datasets default\|all\|CONSUMER\|NAME,NAME` | which datasets to stage (default keeps the WRF and mesh set; chem-dust opts into the three upstream dust datasets; all selects every pin). A consumer name stands for one door's whole set: 'wrf' is the 8 the WRF static builder opens, 'mesh' is what woof mesh needs for the static half of its pair. Use '--datasets wrf' to skip the ~1.7 GiB of land-use pins that only woof mesh reads |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--keep-archives` | keep the verified tarballs under <root>/.fetch-geog after extraction (default: remove each one after its datasets validate) |
| `--list` | print the dataset/size/source table and per-dataset staged state, then exit without touching the network (--explain does the same and adds which door needs each dataset and its URL) |
| `--root DIR` | geog root to stage into (default: $GPUWM_CASE_DATA_ROOT/WPS_GEOG, exactly what woof doctor checks and wizard configs reference) |
| `--source {hf,ncar}` | download host: 'ncar' (default) is the upstream NCAR server (no upstream checksums; the packaged pins are enforced); 'hf' is a byte-for-byte mirror of the same tarballs on Hugging Face (CDN bandwidth, pinned bytes) |
| `--static-source ID` | stage only the published static file of this static-source row (woof/data/static_sources/static-sources.v1.toml), the file a configuration names with [static] source; verified against the row's size and SHA-256 and staged under <root>/static_sources/<ID>/ |

## `woof fetch-tables`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--from DIR` | stage from a local directory instead of downloading (offline installs); verification is identical |
| `--ocean` | also stage the 12 NOAA WOA18 monthly mixed-layer-depth files (about 85 MB) the default mixed-layer ocean initializes from, SHA-256 verified, into the cache forecasts read (WOOF_OCEAN_CACHE, else <case data root>/ocean), so no forecast fetches them from NCEI at start |
| `--ocean-only` | stage only the ocean climatology and leave the coefficient tables alone |
| `--ocean-root DIR` | stage the ocean climatology into DIR (same meaning as WOOF_OCEAN_CACHE) |
| `--replace-damaged` | with --schemes and --from, replace damaged files only after source pin verification |
| `--schemes FAMILY` | stage only named physics table families, e.g. p3 noah tempo rrtmgp; see woof.tables |
| `--thompson-fork` | also stage the pinned WRF 3.9 fork Thompson coefficient set (thompson_version = "wrf_39_noaa", the HRRR configurations' tables; 330 MiB) into ~/.woof/tables/thompson-wrf39-noaa: from --from DIR, packaged fork data, an explicitly selected mirror, or the release assets, SHA-256-verified against the packaged pins before install. Runs that select the fork acquire it automatically |
| `--thompson-fork-only` | stage only the fork coefficient set and leave classic tables alone |
| `--thompson-fork-root DIR` | stage fork tables into DIR instead of the selected fork cache |
| `--wif` | also stage QNWFA_QNIFA_SIGMA_MONTHLY.dat (215 MiB), the global monthly aerosol climatology the mp_physics=28 WIF ingest reads (aer_init_opt=1 with wif_input_opt=1), into ~/.woof/wif under the same SHA-256 contract. Opt-in: it is an input dataset, not a coefficient table. Forecast fetches acquire it automatically when selected physics needs it |
| `--wif-only` | with --wif, stage only that dataset and leave the coefficient tables alone |
| `--wif-root DIR` | stage the WIF dataset into DIR instead of ~/.woof/wif (same meaning as WOOF_WIF_DATA_ROOT) |

## `woof fire-ideal`

| argument | what it does |
|---|---|
| `config` | RunConfig TOML or native namelist.input with --namelist |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--input-directory` | native input files (default: configuration directory) |
| `--namelist` | read native WRF ideal namelist.input |
| `--outdir` | _(the parser declares no help text for this option)_ |
| `--prepare-only` | initialize the complete ideal state and write its receipt |
| `--restart` | restore the byte-exact fire and atmosphere checkpoint |
| `--run-seconds` | _(the parser declares no help text for this option)_ |
| `--sounding` | native input_sounding (default: input directory) |

## `woof fire-ignition`

| argument | what it does |
|---|---|
| `request` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--domain` | _(the parser declares no help text for this option)_ |
| `--experiment` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof fire-static`

| argument | what it does |
|---|---|
| `config` | static REQUEST.json or resolved EXPERIMENT.toml |

| option | what it does |
|---|---|
| `--domain` | experiment domain grid_id; defaults to its only SFIRE domain |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--output` | _(the parser declares no help text for this option)_ |
| `--receipt` | _(the parser declares no help text for this option)_ |
| `--request` | source routes, bound rasters and optional captured observed_perimeter JSON |

## `woof go`

`--no-memory-gate` is the only escape from the pre-fetch memory gate.  The gate runs before the chain downloads anything, and on a box whose card it cannot see it declines to refuse rather than blocking a run that would have worked.

| argument | what it does |
|---|---|
| `[CONFIG]` | an experiment TOML from woof domain; its source and domain tree choose the preparation route |

| option | what it does |
|---|---|
| `--accept-licence LICENCE` | explicitly accept a model row's exact restricted/unresolved weight licence; repeat for multiple licences |
| `--branch-from PARENTS` | branch members from saved checkpoints in a parent-list JSON |
| `--branch-hours` | forecast hours after each parent checkpoint (default 1) |
| `--branch-surface` | also apply the WoFS seeded soil-moisture and SST draws after restore |
| `--card` | target card for domain sizing, e.g. 16gb |
| `--cards all\|0,1` | cards for checkpoint branches: all visible cards or visible ordinals |
| `--center` | regional domain center LAT,LON |
| `--cycle YYYY-MM-DDTHH\|latest` | download routes only: run the config at this cycle; wins over the config's [fetch] cycle the way --transport does. The config is re-timed (start time, delayed nests, namelists) into <outdir>/cycles/<cycle>/; latest is resolved once, under the run's posting rule (as posted: the newest cycle whose start needs are posted) |
| `--cyclone-products JSON` | JSON common_grid and optional grid_id/products; ordinary go accumulates max-wind, gust, min-pressure and track at every committed step |
| `--data-dir DIR` | download routes only: use this existing download instead of the automatically managed request cache |
| `--devices N` | resident slab count; replaces [devices] count |
| `--dry-run` | validate the route and show how to launch it; fetch and run nothing |
| `--ensemble-seed` | the member seed root; the same root draws the same members |
| `--era5-provider {arco,cds}` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--extent-km` | domain width when creating a domain (default 900 km) |
| `--family-weights ai=0.5,physics=0.5` | family probability masses, must sum to 1; defaults to member shares |
| `--geog-root DIR` | override the geography tree (default: the config's [case_data].geog_root or [static].geog_root, otherwise the staged WPS_GEOG tree) |
| `--grib2` | write native surface and pressure-level GRIB2 as each frame lands, then <run>/grib2-grib2.zip |
| `--history-compression {deflate,none}` | history frame storage: deflate (default; lossless NetCDF-4, the same values in about a third of the disk) or none (classic CDF-2). Also GPUWM_HISTORY_COMPRESSION |
| `--history-interval SECONDS` | history frame interval when creating a generated-source domain |
| `--hours` | forecast duration when creating a domain (default 3 h) |
| `--hydro on\|off\|SETTINGS.toml` | hydro is opt-in: --hydro on couples NextGen hydro wherever a hydrofabric covers the domain; a TOML file selects settings; without it the forecast has no hydro |
| `--icechunk-store URL` | Output store base: local path or s3://bucket/prefix; domain and member paths are appended (default RUN/icechunk) |
| `--init` | initial UTC analysis/cycle, YYYY-MM-DDTHH |
| `--initial-inputs JSON` | a separate packaged analysis for the initial state (woof prep --initial-inputs); the fetched source still supplies every lateral boundary, and the forecast starts at the preparation head while the later boundary times are built |
| `--keep-checkpoints N` | how many complete checkpoint sets the run keeps in its folder (default 1, enough to resume); 0 keeps every hourly set, which a later branch or downscale from an earlier checkpoint needs |
| `--keep-member-files` | also retain every member's full history files |
| `--late-after-minutes MIN` | download routes only: how far past its scheduled time a lead may be before the run stops with exit 75; wins over the config's [fetch] late_after_minutes and the source row's budget |
| `--live-products-width` | _(accepted, but not listed by --help)_ |
| `--max-lag-hours` | oldest lagged cycle (0 disables lags) |
| `--member ID` | select one generated source trajectory; fetch accepts member lists |
| `--member-sharing {auto,mps,contexts}` | how co-resident checkpoint branches share a card: a private MPS controller for this launch, or one CUDA context per member (default auto: MPS where the controller exists) |
| `--members N` | make an N-member ensemble with aggregate products |
| `--members-per-card auto\|N` | checkpoint branches run at once on each card, one process each (default auto: as many priced members as the card's free memory holds, up to 4) |
| `--nest-at LAT,LON[@TIME]` | spawn a storm-following nest at this point (at TIME, an ISO time or seconds into the run; default the first step); repeatable |
| `--no-memory-gate` | skip the before-launch memory check that refuses a configuration whose binding phase cannot fit this card's free VRAM, and the forecast runner's own check of the same envelope; a model state too big to build at all is still refused |
| `--no-probe` | with --readiness: compute the schedule from the source table only and ask no host |
| `--no-surface-fill` | disable fill soil/SST perturbations |
| `--no-verify-visuals` | skip postforecast observation verification; the physical run is unchanged |
| `--nudging {off,spectral,grid}` | relax toward prepared source analyses; spectral keeps only the configured large-scale modes |
| `--outdir DIR` | output root for one timestamped run folder per launch, with forecast files, pictures and diagnostics (default <config-stem>-go beside the config); an existing run-... folder is used directly |
| `--output-config TOML` | Read [output] formats and icechunk_store from TOML; CLI values override it |
| `--output-format FORMATS` | Output-format table rows, comma separated (default netcdf); netcdf,icechunk adds atomic native Zarr v3 history |
| `--physics-arms {fill,all,off}` | _(the parser declares no help text for this option)_ |
| `--prepare-only` | fetch and stream prepared inputs without starting a forecast |
| `--prepared-root DIR` | run this existing prepared bundle without fetch or preparation; add --restart to continue its checkpoint |
| `--products LIST` | which products the render stage draws: a comma-separated list of catalog slugs, 'all' (the default -- the renderer's whole catalog), or 'none' to stop after the forecast. The same spelling `woof render --products` takes |
| `--publish-cmd COMMAND` | optional delivery command; manifest JSON on stdin, no shell (default: off) |
| `--publish-root DIR` | local static web root (default: run output/www) |
| `--readiness` | print gpuwm.readiness.v1 for the config's fetch window on stdout and run nothing: exit 0 ready, 75 not yet, 2 refused |
| `--readiness-fold` | _(accepted, but not listed by --help)_ |
| `--recipe {wofs,time-lagged,multi-model,surface-state,member-roster}` | take each ensemble member from a real source trajectory: time-lagged runs earlier cycles of the config's own source over the same window, multi-model runs the trajectories --trajectories lists, surface-state runs seeded soil moisture scales and SST offsets from [ensemble.perturbation], member-roster runs named land and fixed surface arms from [ensemble.member_variants] |
| `--restart CHECKPOINT` | continue an existing checkpoint into fresh output; a checkpoint woof go wrote resumes from its own run folder's prepared bundle, other prepared-cache checkpoints also need --prepared-root |
| `--restart-roster JSON` | continue the exact original members from a durable ensemble restart roster |
| `--resume-ensemble RUN` | retry unfinished checkpoint branches in RUN; preserve completed members and failed attempts |
| `--run-stamp {on,off}` | put this run's forecast files, pictures and diagnostics in its own timestamped folder under --outdir (default on): --outdir/run-<YYYYMMDD>-<HHMMSS>Z_i<YYYYMMDD><HHMM>Z/ (launch instant UTC, then the model initialisation time; the _i part is omitted when the run's init time cannot be read). Successive runs of one configuration then never overwrite or interleave each other. 'off' writes straight into --outdir, which is what releases up to 2.4.1 did; it is kept only for a consumer still written against that and is a workaround, not a supported alternative |
| `--section lat,lon,lat,lon\|FILE.json` | the line the vertical-section products (xsec:<fill>[/<overlay>...] in --products) are cut along, the same value `woof render --section` takes; a JSON file gives {start, end} or a {points, extend_km} polyline. Spell a line that starts with a minus sign as --section=-33.9,151.2,-34.1,151.3. An xsec: product with no line is refused before anything is fetched |
| `--smoke` | turn on HRRR-Smoke (transported smoke, its radiative feedback, RAVE fires); in 2.9.4 it costs about 11x per step on full CONUS (measured on 8x RTX 5090) |
| `--smoke-radiation {on,off}` | explicit transported aerosol radiation feedback: on selects live SW/LW optics instead of aerosol climatology and shortwave-only smoke; off disables live feedback. Requires active optical mass rows; omitted preserves the recipe. See docs/solar-products.md |
| `--source SOURCE[:N],...` | create a 3 km WoFS-shape ensemble domain; mixed family counts sum to --members; eligible registry ids: hrrr, hrrr-prs, gem-gdps, icon-global, icon-eu, icon-d2, gfs, gdas, gefs-v11-flat, gefs-reforecast, gefs-reforecast-weekly, gefs-v11, gefs, aigfs, aigefs, ecmwf-open-data, ecmwf-open-data-0p1, ecmwf-ens, aifs, earth2, earth2-fcn3, earth2-sfno, earth2-pangu, earth2-aurora, earth2-fengwu, earth2-aifs, earth2-fuxi, earth2-stormcast, earth2-atlas, earth2-atlas-si, earth2-aifs2, earth2-aifs-ens, earth2-aifs2-ens, earth2-aurora15, earth2-aurora15-6h, earth2-aurora15-ensemble, earth2-aurora15-ensemble-6h, earth2-gencast-mini, earth2-weathernext2, earth2-weathernext2-mini, earth2-ucast, earth2-graphcast-operational, earth2-graphcast-small, earth2-stormcast-conus, earth2-atlas-crps, rap-native, hrrr-native, rap, rrfs, rrfs-ens, era5, era5-l137, 20crv3, 20crv3-cf |
| `--stage-only STAGE` | _(accepted, but not listed by --help)_ |
| `--static-cache-key` | _(accepted, but not listed by --help)_ |
| `--static-cache-root` | _(accepted, but not listed by --help)_ |
| `--storm-nests N\|auto\|off` | storm-following fine nests that spawn on their own: N dormant slots (or 'auto', priced from the card's free memory) watch the parent's 2-5 km updraft helicity, each nest follows its storm and stops when it decays, merges or leaves the domain. Overrides the config's [storm_nests] slots; 'off' removes the automatic slots (docs/storm-nests.md) |
| `--supplement ROLE=PATH` | explicit preparation donor; repeat for multiple files. HRRR accepts PMSL=GRIB inside --data-dir and binds its bytes in the preparation source manifest |
| `--tiles {off,auto,on}` | [tiles] mode for this launch's forecast: 'auto' streams a domain the card cannot hold through host memory, 'on' always streams, 'off' runs resident. A prepared bundle (--prepared-root, --restart) takes it as an execution control, never into its hash-bound config; a fetching launch runs a copy of the config with the mode re-stated, as --cycle does |
| `--trajectories FILE` | the multi-model member list: a JSON or TOML file of {source, cycle[, member]} entries, one per member (selects --recipe multi-model) |
| `--transport {auto,aws,dwd,ecmwf,gdex,google,msc,ncei-0p5,ncei-1p0,nomads,s3}` | download routes only: pin the fetch stage to one host of the source's endpoint ladder, the value `woof fetch --transport` takes; wins over the config's [fetch] transport, and the plan says which it used |
| `--warm-kernels` | _(accepted, but not listed by --help)_ |
| `--whole-cycle` | download routes only: the fetch stage waits for the whole cycle (the old rule) instead of taking each lead as it posts; wins over the config's [fetch] as_posted |
| `--wind-amplitude` | fill uniform U/V amplitude in m/s; 0 disables it |
| `--wind-farms uswtdb\|TABLE\|off` | opt-in Fitch turbine drag and TKE from USWTDB or a WRF/JSON turbine table |
| `--wps-namelist PATH` | with --prepared-root: the exact WPS authority required by a single-domain portable bundle |

## `woof hex`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof hf-fetch`

| argument | what it does |
|---|---|
| `target` | Earth-2 model row or hf://[datasets/]<owner>/<repo>[@revision][/path] |

| option | what it does |
|---|---|
| `--accept-licence LICENCE` | explicitly accept a model row's exact restricted/unresolved weight licence; repeat for multiple licences |
| `--cache` | cache root (default WOOF_HF_CACHE or ~/.cache/woof/hf) |
| `--connections` | parallel HTTP connections, 1..30 (default 30) |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof hubwind`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof hubwind catalog`

Takes no options of its own.

## `woof hubwind fetch-observations`

| option | what it does |
|---|---|
| `--date` | UTC YYYY-MM-DD |
| `--network {noaa_bao_100m,noaa_bao_300m}` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |

## `woof hubwind products`

| option | what it does |
|---|---|
| `--request` | _(the parser declares no help text for this option)_ |

## `woof hydro`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof hydro ahead`

| argument | what it does |
|---|---|
| `config` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `woof hydro calibrate`

| argument | what it does |
|---|---|
| `request` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--device` | _(the parser declares no help text for this option)_ |

## `woof hydro demo`

| argument | what it does |
|---|---|
| `example` | pinned NextGen example id |

| option | what it does |
|---|---|
| `--data-dir` | verified immutable input cache |
| `--device` | CUDA device ordinal (default: 0) |
| `--outdir` | parent folder for separate example runs |
| `--prepare-only` | verify and materialize inputs without simulating water |
| `--seconds` | simulation duration in seconds (default: example table) |
| `--strict-numerics` | explicit strict proof mode; fast numerics is the default |

## `woof hydro fabric`

| argument | what it does |
|---|---|
| `config` | JSON with geopackage, output, data_license and optional grid/subset |

Takes no options of its own.

## `woof hydro forcing`

| argument | what it does |
|---|---|
| `request` | JSON naming source or explicit source_profile/profile_row |

Takes no options of its own.

## `woof hydro init-state`

| option | what it does |
|---|---|
| `--date` | forecast start, e.g. 2023-01-01T00:00:00Z |
| `--fabric` | HYDRO-WORLD contract GeoPackage |
| `--outdir` | _(the parser declares no help text for this option)_ |
| `--params` | CFE parameter row file when the fabric has no model_attributes |
| `--source` | row in init-state-sources.v1.json |
| `--spinup-days` | _(the parser declares no help text for this option)_ |

## `woof hydro prepare`

| argument | what it does |
|---|---|
| `config` | JSON with geopackage, output, data_license and optional grid/subset |

| option | what it does |
|---|---|
| `--cell-overlap` | transpose sealed catchment intersections onto an exact terrain grid |

## `woof hydro retrospective`

| option | what it does |
|---|---|
| `--cache` | Capture immutable compressed archive objects |
| `--candidate` | arwen.point-flow.v1 candidate JSON |
| `--end` | Last sample UTC, inclusive |
| `--mapping` | Explicit archive_feature_id/candidate_feature_id mapping rows JSON |
| `--output` | Output directory; repeated runs retain results in a new run subfolder |
| `--request` | Complete archive request JSON |
| `--source` | _(the parser declares no help text for this option)_ |
| `--source-table` | _(the parser declares no help text for this option)_ |
| `--start` | First sample UTC, inclusive; omit with --end to inspect coverage |
| `--store` | Local captured archive for offline replay |
| `--workers` | Bound Rust task concurrency, default available CPU cores |

## `woof hydro run`

| argument | what it does |
|---|---|
| `config` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--device` | _(the parser declares no help text for this option)_ |
| `--outdir` | _(the parser declares no help text for this option)_ |
| `--restart` | _(the parser declares no help text for this option)_ |
| `--seconds` | _(the parser declares no help text for this option)_ |

## `woof hydro verify-usgs`

Takes no options of its own.

## `woof hydro verify-usgs compare`

| option | what it does |
|---|---|
| `--label` | _(the parser declares no help text for this option)_ |
| `--observations` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |
| `--prediction` | _(the parser declares no help text for this option)_ |

## `woof hydro verify-usgs describe`

| option | what it does |
|---|---|
| `--out` | _(the parser declares no help text for this option)_ |
| `--source` | _(the parser declares no help text for this option)_ |
| `--table` | _(the parser declares no help text for this option)_ |

## `woof hydro verify-usgs discover`

| option | what it does |
|---|---|
| `--bbox` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |
| `--source` | _(the parser declares no help text for this option)_ |
| `--table` | _(the parser declares no help text for this option)_ |

## `woof hydro verify-usgs fetch`

| option | what it does |
|---|---|
| `--end` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |
| `--source` | _(the parser declares no help text for this option)_ |
| `--start` | _(the parser declares no help text for this option)_ |
| `--table` | _(the parser declares no help text for this option)_ |

## `woof hydro verify-usgs import`

| option | what it does |
|---|---|
| `--files` | _(the parser declares no help text for this option)_ |
| `--layout` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |
| `--table` | _(the parser declares no help text for this option)_ |

## `woof hydro verify-usgs match`

| option | what it does |
|---|---|
| `--fabric` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |
| `--stations` | _(the parser declares no help text for this option)_ |
| `--table` | _(the parser declares no help text for this option)_ |

## `woof hydro verify-usgs match-check`

| option | what it does |
|---|---|
| `--fabric` | _(the parser declares no help text for this option)_ |
| `--matches` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |

## `woof hydro verify-usgs observations`

| option | what it does |
|---|---|
| `--observations` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |

## `woof hydro verify-usgs replay`

| option | what it does |
|---|---|
| `--expected` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |
| `--raw` | _(the parser declares no help text for this option)_ |
| `--source` | _(the parser declares no help text for this option)_ |
| `--station` | _(the parser declares no help text for this option)_ |
| `--table` | _(the parser declares no help text for this option)_ |

## `woof hydro verify-usgs score`

| option | what it does |
|---|---|
| `--model` | _(the parser declares no help text for this option)_ |
| `--observations` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |
| `--reference` | _(the parser declares no help text for this option)_ |

## `woof hydro verify-usgs stations`

| option | what it does |
|---|---|
| `--attributes` | _(the parser declares no help text for this option)_ |
| `--feature-field` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |
| `--records` | _(the parser declares no help text for this option)_ |
| `--station-field` | _(the parser declares no help text for this option)_ |

## `woof import-namelist`

| argument | what it does |
|---|---|
| `WPS` | WPS namelist.wps (projection + nest layout) |
| `INPUT` | WRF namelist.input (domains/physics/dynamics/bdy_control) |

| option | what it does |
|---|---|
| `--ack ID` | declared-experiment acknowledgement id to write into [experiment].acknowledgements of the resolved TOML (repeatable). WRF namelists cannot spell woof governance declarations, so an import that needs one -- e.g. shortwave-on/longwave-off physics across a window that includes local night -- names the id it wants in its refusal |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--geogrid-tbl PATH` | GEOGRID.TBL (file or directory) whose HGT_M smooth_option/smooth_passes set every domain's terrain smoothing (default: the namelist.wps opt_geogrid_tbl_path, else ./geogrid/ beside it) |
| `--name NAME` | [experiment].name for the resolved TOML (default derived from start time and domain count) |
| `--output TOML` | write the resolved experiment TOML here (omit to print the report only) |
| `--rrtmg-variant {rte-rrtmgp,rrtmg_legacy}` | implementation for a WRF RRTMG 4/4 request: RTE+RRTMGP by default, or legacy RRTMG when the namelist selects GSD MYNN or aer_opt 1, 2 or 3; an explicit choice retains its implementation |
| `--static-cache-root DIR` | cache_root of the [static.highres] block a land-cover geog_data_res token (cglc_modis_lcz) imports as (default: the per-user high-resolution cache the engine's default terrain already uses); unused when the namelist names no such token |
| `--terrain-smoothing-precision {float64,wps-float32}` | arithmetic of every domain whose terrain smoother is WPS's default smth-desmth_special x1: wps-float32 reproduces geogrid.exe's HGT_M exactly, float64 is WOOF's own smoother (default: the GEOGRID.TBL HGT_M smooth_precision, else float64); every other smoother always runs WPS's float32 |
| `--urban-fvg` | _(the parser declares no help text for this option)_ |
| `--urban-table-dir` | _(the parser declares no help text for this option)_ |
| `--urban-treeoption {0,1}` | _(the parser declares no help text for this option)_ |
| `--urban-ttscheme {0,1}` | _(the parser declares no help text for this option)_ |
| `--urban-ucm-generation {wrf_471,wrf_480}` | single-layer urban numerical generation; omitted imports retain wrf_471 |
| `--wrf-version` | the WRF line or release the namelist was written for (3, 4, or a release such as 4.6.1). The line selects the Registry default an omitted &dynamics/use_theta_m takes (3: 0, dry theta, the line operational HRRR v4 runs; 4, the default: 1, moist theta, booked as a substitution). A V4 release also pins the scheme generations that release runs (4.6.1: WRF 4.6's YSU, sun, Noah-MP snow aging and urban water balance; a bare 4 keeps the newest). The report names the version and what chose it |

## `woof ingest`

| argument | what it does |
|---|---|
| `CONFIG` | experiment TOML ([experiment]/[[domain]] tables, as emitted by `woof domain` or `woof import-namelist`; config-driven runs declare their inputs in [case_data]). A legacy [run]-table RunConfig naming a registered case is also accepted. |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--output NPZ` | initialized-state NPZ output |

## `woof kernels`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof kernels pack`

| argument | what it does |
|---|---|
| `output` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--arch` | SM filter, e.g. sm_120 or sm_89; repeatable |
| `--cache` | override WOOF_KERNEL_CACHE |

## `woof kernels seed`

| argument | what it does |
|---|---|
| `pack` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--cache` | override WOOF_KERNEL_CACHE |
| `--sha256` | expected SHA256; defaults to PACK.sha256 |

## `woof local-da`

| option | what it does |
|---|---|
| `--budget-seconds` | advisory wall-time target for preparation, forecast and cycles; does not reduce cycles or impose a runtime deadline |
| `--cadence-seconds` | exact requested whole-second cadence; otherwise derived from scale; never shortened to meet a cost estimate |
| `--capabilities` | print the companion command and field contract without pricing |
| `--card-name` | label recorded beside the timing basis so a review names the card it was priced for |
| `--continuous WINDOWS` | cycle continuously for WINDOWS analysis windows at the reviewed cadence: each window restarts from the previous analysis, assimilates, forecasts and renders, and the boundary forcing is renewed from the same source cycle when a window reaches past it; --status and --stop address the saved plan |
| `--dry-run` | review only; no writes, downloads or device allocation |
| `--epoch` | initial UTC timestamp on a whole-second boundary, including Z or an explicit offset |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--forcing-cadence-hours` | whole-hour source boundary spacing, separate from the whole-second analysis cadence |
| `--forecast-seconds` | length of the forecast that follows the last analysis, in seconds |
| `--free-gib` | free card memory in GiB when less than the whole card is available; defaults to the declared total |
| `--host-gib` | declared host RAM in GiB for advisory comparison with estimated analysis arrays and observation tables |
| `--json` | emit the review as one JSON document on stdout, which this door always does; accepted so a companion can state it |
| `--launch` | launch or resume an existing local-da.json |
| `--obs-table` | existing neutral observation table; repeatable |
| `--out` | new directory to publish experiment.toml, ensemble.toml, experiment.namelist.wps, every other file the published configuration is read with on its own input route, and local-da.json into; refused if it exists |
| `--point` | latitude,longitude |
| `--prepared-config` | configuration authority consumed by the supplied prepared bundle |
| `--prepared-namelist` | WPS authority consumed by the supplied prepared bundle |
| `--prepared-root` | existing portable single-domain prepared bundle, verified by the ordinary forecast reader |
| `--profile` | physics profile name resolved by the authoring authority; defaults to the profile that authority selects at the grid spacing of the selected rung |
| `--radar-grid` | existing radar-grid observation file; repeatable |
| `--region` | west,south,east,north; east<west crosses the dateline |
| `--request-json` | arwen.local-da-request.v1 file, or - for stdin |
| `--run` | launch after publishing the reviewed configuration |
| `--satellite-grid` | existing cloud-water-path grid; repeatable |
| `--scale` | requested rung of the derived ladder; its domain, resolution, members and cycle count are preserved, with lower rungs shown as alternatives |
| `--score PLAN` | score every still-unscored nowcast lead of a saved plan now and exit: each completed window is graded against the MRMS composite nearest 15, 30, 45 and 60 minutes after its analysis, beside the radar-persistence baseline and the difference, and each window receipt is rewritten; a run scores its own leads by default, so this is for the leads whose valid time had not arrived when the run finished |
| `--seed` | base seed for member perturbation and the static covariance samples; the same seed reproduces the same analysis |
| `--source` | background source name resolved by its owner; defaults to the package background source |
| `--source-cycle` | explicit source cycle in UTC; omitted selects one published cycle covering the entire window |
| `--source-input` | ROLE=PATH for original source bytes needed to verify a supplied member identity; repeatable |
| `--source-member` | one source member, separate from the regional ensemble member count |
| `--source-product` | source product selector; source membership never changes the default product implicitly |
| `--source-provider` | provider supported by the selected product owner |
| `--source-root` | local source directory with the existing preparation handoff or native member inventory |
| `--speed-factor` | compute speed relative to the printed reference card, not memory capacity |
| `--status PLAN` | print the continuous status document of a saved plan with its controller liveness, and exit |
| `--stop PLAN` | ask the running continuous controller of a saved plan to stop after its current operation; the request is durable, and a launch made while no controller runs clears it and resumes, so ask again after that launch to stop it |
| `--supplement` | ROLE=PATH consumed by the preparation composition; repeatable |
| `--vram-gib` | declared card memory in GiB; required unless a request document supplies the card |

## `woof mesh`

| option | what it does |
|---|---|
| `--allow-rough-mesh` | WORKAROUND: emit a mesh rougher than the stated smoothness bound. Reported as a workaround in the receipt, never as a setting |
| `--background-km KM` | cell spacing far from every refinement region; with no --refine this is a uniform mesh at that spacing |
| `--card NAME` | size the mesh to this card's measured device footprint; --list-cards prints the ones that have been measured |
| `--cells N` | exact cell count, skipping the device model entirely |
| `--clobber` | replace an existing --out |
| `--dry-run` | size and cost the request, apply both gates, write nothing |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--geog DIR` | WPS_GEOG archive the static's terrain, land use, soil, green-ness and albedo come from. Defaults to $GPUWM_WPS_GEOG, then ~/.local/share/woof/WPS_GEOG |
| `--list-cards` | print the cards in the sizing table and the largest mesh each measured one holds, then exit |
| `--name TEXT` | label carried into the grid file's provenance attributes |
| `--no-static` | WORKAROUND: write only the grid. The result is NOT runnable -- the mesh registry pins grid AND static, so a lone grid file is refused before any dycore sees it |
| `--nominal-dx-m M` | the nominal spacing the static DECLARES, in metres. Defaults to the grid's own implied value. This scalar is compared FP32-bit-exactly by the mesh registry, so a declaration the grid disagrees with is refused |
| `--out GRID.nc` | grid file to write (not needed with --dry-run or --list-cards) |
| `--receipt JSON` | write the measured receipt here as well as to stdout |
| `--refine LAT,LON,KM,KM` | refine a circle: LAT,LON,RADIUS_KM,SPACING_KM and optionally a fifth TRANSITION_KM. Repeatable -- each row is one region and adding one is data, not a code path |
| `--refine-box LAT0,LAT1,LON0,LON1,KM` | refine a latitude/longitude box: LAT0,LAT1,LON0,LON1,SPACING_KM and optionally a sixth TRANSITION_KM. Repeatable |
| `--spec SPEC.json` | read the whole resolution spec from a JSON document instead of building it from --background-km/--refine |
| `--static-out STATIC.nc` | where the matching static goes. Defaults to --out with a .static.nc suffix beside the grid, because the mesh registry admits the two as a pair |
| `--sweeps N` | relaxation budget passed to the generator |
| `--tolerance X` | relaxation convergence tolerance passed to the generator |
| `--triangulation {rebuild,incremental}` | how the Delaunay is kept between relaxation sweeps. incremental (the default) keeps the facets and repairs them by Lawson flips; rebuild rebuilds it every sweep (slower). Both write the same bytes |
| `--vram-gib X` | device budget in GiB, instead of the named card's total memory (for a card that is shared with something else); needs --card, because the fixed term is per card |

## `woof ml-export`

| argument | what it does |
|---|---|
| `[INPUT ...]` | history files, folders holding them, .gz history files, or ZIPs |

| option | what it does |
|---|---|
| `--append` | add these frames to the export in --out (one frame at a time) |
| `--chunk-yx SHAPE` | horizontal chunk shape: auto (default; 512x512 tiles for planes over 1024x1024 cells, one chunk otherwise), full, or NY,NX |
| `--config TOML` | the run's configuration; its SHA-256 is recorded |
| `--domains LIST` | e.g. d01,d02 |
| `--end TIME` | last valid time kept |
| `--every HOURS` | keep frames on multiples of HOURS from the run's start |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--finalize` | close the export in --out after --append calls |
| `--grid GRID` | native (default), latlon, or latlon:DEG |
| `--json` | with --list: the options document; otherwise: progress as JSON lines |
| `--layout {analysis,forecast}` | analysis: time is valid time; forecast: WeatherBench 2's time + prediction_timedelta |
| `--levels SET` | wb13 (default), era5-37, model, model:LIST, or hPa list |
| `--list` | print the level sets, variables and naming schemes |
| `--names SCHEME` | wb2 (default, WeatherBench 2 long names) or era5 (short names) |
| `--out DIR` | the export folder (one dNN.zarr per domain, README.txt, receipt) |
| `--overwrite` | replace an earlier export in --out |
| `--regrid {bilinear,area-mean}` | latlon method (default bilinear) |
| `--skip-unavailable` | write the variables the files can make and record the rest |
| `--start TIME` | first valid time kept |
| `--threads N` | worker threads (default: every core) |
| `--variables LIST` | default, all, +NAME,... or NAME,... (table ids or names) |
| `--zip` | also write <out>-ml.zip (stored entries, opens in place) |

## `woof multi-run`

`--preflight {estimate,alloc,off}` is the only override of the plan's own preflight mode.

| argument | what it does |
|---|---|
| `PLAN.toml` | versioned plan with one or more [[run]] entries |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--preflight {estimate,alloc,off}` | override the plan's woof check mode: estimate, alloc, or off |
| `--summary SUMMARY.json` | summary path relative to the plan (default PLAN.summary.json) |

## `woof obs`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof obs airnow`

| argument | what it does |
|---|---|
| `ARGS ...` | arguments passed to the instrument's binary unchanged, --help included; woof's own flags must come before the instrument name |

Takes no options of its own.

## `woof obs asos`

| argument | what it does |
|---|---|
| `ARGS ...` | arguments passed to the instrument's binary unchanged, --help included; woof's own flags must come before the instrument name |

Takes no options of its own.

## `woof obs glm`

| argument | what it does |
|---|---|
| `ARGS ...` | arguments passed to `rw_goes glm` unchanged |

Takes no options of its own.

## `woof obs goes`

| argument | what it does |
|---|---|
| `ARGS ...` | arguments passed to the instrument's binary unchanged, --help included; woof's own flags must come before the instrument name |

Takes no options of its own.

## `woof obs hail`

| argument | what it does |
|---|---|
| `ARGS ...` | arguments passed to the instrument's binary unchanged, --help included; woof's own flags must come before the instrument name |

Takes no options of its own.

## `woof obs mrms`

| argument | what it does |
|---|---|
| `ARGS ...` | arguments passed to the instrument's binary unchanged, --help included; woof's own flags must come before the instrument name |

Takes no options of its own.

## `woof obs odim`

| argument | what it does |
|---|---|
| `ARGS ...` | arguments passed to the instrument's binary unchanged, --help included; woof's own flags must come before the instrument name |

Takes no options of its own.

## `woof obs opera`

| argument | what it does |
|---|---|
| `ARGS ...` | arguments passed to the instrument's binary unchanged, --help included; woof's own flags must come before the instrument name |

Takes no options of its own.

## `woof obs radar`

Takes no options of its own.

## `woof obs radar doctor`

Takes no options of its own.

## `woof obs radar fetch`

| option | what it does |
|---|---|
| `--archive ID` | archive id from woof/obs/data/radar_archives_world.json: jma-nict, bom-nci, ideam-s3, kma-apihub |
| `--max-range-km KM` | with --out: trim gates beyond this |
| `--out PACK` | also decode the fetched files into this sweep pack |
| `--out-dir DIR` | where the fetched files go |
| `--quantities Q,Q` | with --out: quantities to carry |
| `--site ID` | the site in that archive's own naming (the table's site_ids says what that is) |
| `--time ISO` | valid time, UTC, e.g. 2019-10-12T09:00Z; the nearest volume within the archive's tolerance is taken |
| `--tolerance-s S` | how far from --time a volume may be (default: the archive's, about half its cadence) |

## `woof obs radar grid`

| option | what it does |
|---|---|
| `--clear-air-from-censor` | build clear-air zeroes from the decoder's own gate codes as well as from finite below-floor gates. Needs a pack carrying censor planes (v2 or v3); a v1 pack is a hard error rather than a silent fallback. Range-folded and ambiguous gates stay excluded either way |
| `--dealias` | unfold radial velocity per sweep before gridding instead of masking every gate that might be folded. Requires scipy |
| `--grid-wrfout WRFOUT` | the wrfout whose georeference the observations are gridded onto; its SHA-256 joins the receipt |
| `--max-elevation-deg DEG` | likewise: the elevation ceiling, stated rather than defaulted |
| `--max-range-km KM` | THE range authority, required rather than defaulted: a build that quietly picked a different range than the one it is compared against produces a plausible, wrong answer |
| `--out NC` | observation file to write |
| `--overwrite` | replace an existing --out |
| `--pack PACK` | the sweep pack, from `pack` or from rw_nexrad |

## `woof obs radar nyquist`

| option | what it does |
|---|---|
| `--file FILE` | one radar file (ODIM geometry is read without payload; other formats are decoded) |
| `--site ID` | the station, for a multi-station file |

## `woof obs radar pack`

| option | what it does |
|---|---|
| `--dir DIR` | directory of single-sweep ODIM files (SCAN) to assemble into one volume, as Germany publishes them. Mutually exclusive with --file |
| `--file FILE` | one whole-volume file: ODIM PVOL, JMA GRIB2 tar, IRIS RAW, Universal Format, CfRadial 1 or DORADE (gzipped or not; the format is read from the file). Repeat it for one station split over files (JMA's N5 reflectivity and N6 velocity tars). Mutually exclusive with --dir |
| `--max-elevation-deg DEG` | drop cuts above this elevation. The 90-degree birdbath a Dutch volume opens with is a calibration cut, not an observation of anything a model column exists for |
| `--max-range-km KM` | trim gates beyond this range |
| `--out PACK` | sweep pack to write |
| `--quantities Q,Q` | ODIM quantity names to carry (DBZH,VRADH). Omitting this carries every quantity in the volume, which is nine of them on a Dutch scan |
| `--site ID` | the station to take from a file that holds several (a JMA tar holds the whole network; `stations` lists them) |
| `--stamp YYYYmmddTHHMMSSZ` | which volume in --dir to assemble, in the spelling `volumes` reports. Required only when the directory holds more than one: taking the newest silently would put a volume nobody asked for behind an ordinary-looking record |

## `woof obs radar sites`

| option | what it does |
|---|---|
| `--bbox W,S,E,N` | select the sites inside this lon/lat box |
| `--no-velocity` | do not require a radial-velocity moment in the assimilability check, for a reflectivity-only assimilation |
| `--require-assimilable` | also run the assimilability check over the selection and report its refusal in full. Without this the verdict field is null rather than true: a check that did not run has no verdict |
| `--site ID` | select one site by its table id |

## `woof obs radar stations`

| option | what it does |
|---|---|
| `--file FILE` | _(the parser declares no help text for this option)_ |

## `woof obs radar volumes`

| option | what it does |
|---|---|
| `--dir DIR` | directory of ODIM .h5 files, not searched recursively |

## `woof obs radar-feeds`

| argument | what it does |
|---|---|
| `ARGS ...` | arguments passed to the instrument's binary unchanged, --help included; woof's own flags must come before the instrument name |

Takes no options of its own.

## `woof obs scorecard`

| argument | what it does |
|---|---|
| `manifest` | versioned cases, arms, observation sources and metrics |

| option | what it does |
|---|---|
| `--acquire` | fetch and decode public sources named by acquisition metadata; delete these private inputs after scoring |
| `--data-root` | recorded input root substituted for ${DATA_ROOT} in request paths |
| `--out` | output folder for the JSON receipt, the case and campaign tables and the native sheets |
| `--timeout` | seconds allowed for each existing observation door |

## `woof obs stage4`

| argument | what it does |
|---|---|
| `ARGS ...` | arguments passed to the instrument's binary unchanged, --help included; woof's own flags must come before the instrument name |

Takes no options of its own.

## `woof obs streams`

Takes no options of its own.

## `woof obs streams anchors`

Takes no options of its own.

## `woof obs streams fetch`

| option | what it does |
|---|---|
| `--archive` | wis2: a subscriber's archive directory to decode for the window instead of subscribing now |
| `--bbox` | madis-*: keep reports inside S,N,W,E (degrees); default worldwide |
| `--end` | window end, ISO-8601 UTC |
| `--missions` | cdaac-ro: comma-separated missions (default cosmic2,paz,kompsat5) |
| `--networks` | iem-metar: comma-separated IEM networks, or 'all' (default) |
| `--out` | directory; the stream lands under <out>/<stream>/ |
| `--satellites` | goes-dmw: comma-separated, default G18,G19 |
| `--seconds` | wis2: how long to subscribe (default 600) |
| `--start` | window start, ISO-8601 UTC |
| `--stations` | igra2 or ndbc: comma-separated station ids (default all) |
| `--stream {awc-metar,cdaac-ro,gnss-ro,goes-dmw,iem-metar,igra2,madis-acars,madis-mesonet,madis-raob,ndbc,nws-upper-air-text,wis2}` | _(the parser declares no help text for this option)_ |

## `woof obs streams hours`

| option | what it does |
|---|---|
| `--cutoff` | information cutoff: an ISO-8601 instant, or 'analysis' for each hour's own instant; rows received later are dropped and counted, rows without a receipt time are kept and counted |
| `--cycle-s` | cycle length in seconds |
| `--end` | last analysis instant, ISO-8601 UTC |
| `--out` | directory for the hourly tables and hours.json |
| `--start` | first analysis instant, ISO-8601 UTC |
| `--tables` | neutral tables (v2 or v1) |
| `--thin-grid` | NLAT,NLON: keep one row per source, variable, cell and ln p layer, the one nearest the analysis instant |

## `woof obs streams streams`

Takes no options of its own.

## `woof obs streams subscribe`

| option | what it does |
|---|---|
| `--out` | directory; the archive lands under <out>/<stream>/archive/ |
| `--seconds` | how long to stay subscribed (default 600) |
| `--stream {wis2}` | the stream to subscribe to; only doors that read a live feed are listed |

## `woof obs streams summary`

| option | what it does |
|---|---|
| `--dir` | the directory `fetch` wrote the streams under |

## `woof ocean`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof ocean extract`

| argument | what it does |
|---|---|
| `request` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `woof ocean prepare`

| argument | what it does |
|---|---|
| `request` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `woof ocean prove-forcing`

| argument | what it does |
|---|---|
| `config` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--forcing` | _(the parser declares no help text for this option)_ |
| `--outdir` | _(the parser declares no help text for this option)_ |
| `--render` | _(the parser declares no help text for this option)_ |
| `--research-data-licence` | explicit opt-in to the named non-default data licence |

## `woof ocean render`

| argument | what it does |
|---|---|
| `request` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `woof ocean run`

| argument | what it does |
|---|---|
| `config` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--forcing` | _(the parser declares no help text for this option)_ |
| `--outdir` | _(the parser declares no help text for this option)_ |
| `--render` | _(the parser declares no help text for this option)_ |
| `--research-data-licence` | explicit opt-in to the named non-default data licence |

## `woof ocean waves`

Takes no options of its own.

## `woof ocean waves audit`

| argument | what it does |
|---|---|
| `request` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `woof ocean waves catalog`

| argument | what it does |
|---|---|
| `request` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `woof ocean waves forcing`

| argument | what it does |
|---|---|
| `request` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `woof ocean waves observations`

| argument | what it does |
|---|---|
| `request` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `woof ocean waves prepare`

| argument | what it does |
|---|---|
| `request` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `woof ocean waves products`

| argument | what it does |
|---|---|
| `request` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `woof ocean waves stage`

| argument | what it does |
|---|---|
| `request` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `woof ocean waves verify`

| argument | what it does |
|---|---|
| `request` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `woof physics-catalog`

| option | what it does |
|---|---|
| `--check JSON` | a check request as JSON text, a file, or - for stdin |
| `--emit` | with --check: print the experiment-file physics lines for a mix that runs |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--into EXPERIMENT` | with --check: print this experiment file with the mix's physics in it, after the experiment loader has accepted it |
| `--json` | print JSON for the GUI or scripts |
| `--out NEW.toml` | with --into: write the result here, with the experiment's companion files (its namelist.wps) copied under the new name, instead of printing it |
| `--preset` | print one preset row |
| `--source` | price and default against this data source's default suite |

## `woof prep`

| option | what it does |
|---|---|
| `--accept-licence LICENCE` | explicitly accept a model row's exact restricted/unresolved weight licence; repeat for multiple licences |
| `--ack` | registry-owned expert physics acknowledgement id; repeatable |
| `--as-posted POSTING_DIR` | prepare as an as-posted fetch publishes the window's leads: POSTING_DIR is that fetch's posting/ folder; the preparation starts on the first leads and its seal writes the input manifest, so no --source-manifest pair is given (a mapped source names where with --author-input-manifest, beside the fetched files) |
| `--author-input-manifest` | create an exact mapped or 20CRv3 input manifest; conflicts with an existing --source-manifest/--source-manifest-sha256 pair |
| `--author-mapping` | create-only path for a mapping compiled from --descriptor; the adjacent *.authoring.json receipt binds descriptor/Vtable bytes |
| `--author-only` | author the requested create-only mapped contract or 20CRv3 member manifest and exit; requires --author-input-manifest and does not need run geometry |
| `--bridge` | prebuilt woof all-Rust source-specific GRIB bridge executable; omitted on the era5/gfs routes it resolves through the shared bridge ladder (environment override, a checkout build, staged bridges under ~/.woof/bridges) exactly as woof go does |
| `--canonical-physics-plan-output PATH` | create an exact canonical UTF-8 copy of the plan validated by --validate-physics-plan; refuses an existing output |
| `--child-workers` | bounded CPU worker budget for parallel d02..dNN initialization (1..32) |
| `--composition` | strict gpuwm-mapped-composition-v2 product join contract |
| `--contributing-mapping ROLE=PATH` | cross-source composition: a contributing source's own mapping document under the mapping_role its field_sources binding declares; bytes must hash to the composition's pinned SHA-256 |
| `--cpu-preprocess-bridge` | _(the parser declares no help text for this option)_ |
| `--cycle` | GFS cycle, YYYY-MM-DDTHH or YYYY-MM-DD_HH:MM:SS |
| `--descriptor` | explicit rw-wps.descriptor.v1 science contract; requires --author-mapping and, for GRIB, --vtable |
| `--domain-source-orography DNN=PATH` | ERA5 hierarchy source-orography binding; repeat once for every domain (d01..dNN). All bindings use --source-orography-variable |
| `--domain-spec` | strict gpuwm-hrrr-target-domain-v1 Lambert root-domain JSON; nested layouts come from --wps-namelist/--namelist-input |
| `--dry-run` | validate route-specific arguments and print the exact internal command |
| `--experiment-config` | _(the parser declares no help text for this option)_ |
| `--experiment-config-suite` | bind the explicitly configured physics suite in the native export |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--extend-root-preparation` | sealed HRRR predecessor to extend by exactly one forcing hour |
| `--forecast-end-hour` | inclusive absolute HRRR source lead |
| `--forecast-start-hour` | absolute cycle-relative HRRR lead used for model time zero |
| `--geog-root` | WPS_GEOG root used to build a domain-specific native static cache; requires --domain-spec and replaces --static-cache/--static-receipt |
| `--gfs-series` | tab-separated HOUR and GFS GRIB2 path inventory |
| `--grib` | combined ERA5 GRIB1 series |
| `--grib2-dump` | override the GRIB2 dump tool; omitted, it resolves through the shared bridge ladder exactly as --grib2-inventory does |
| `--grib2-inventory` | override the GRIB2 inventory tool; omitted, it resolves through the shared bridge ladder (WOOF_GRIB2_INVENTORY, a checkout build, the wheel's bundled copy, then the staged ~/.woof/bridges) |
| `--hierarchy-workers` | bounded mapped d02..dNN initialization workers (1..32) |
| `--history-interval-seconds` | positive output cadence used by HRRR preparation and the prepared-cache forecast identity |
| `--initial-inputs JSON` | separate packaged analysis inventory for the initial state; --source continues to supply every lateral boundary frame |
| `--input` | mapped source file; repeat in deterministic time/file order |
| `--input-list` | file naming the mapped source files, one path per line, in the same deterministic time/file order the repeated --input flag spells; the spelling that keeps a field-per-file source's hundreds of inputs inside the Windows 32 KB command-line limit |
| `--list-sources` | print the provenance-bound source capability manifest as JSON |
| `--mapped-engine {rust,python}` | which engine decodes mapped source bytes; omitted, the default engine runs. `python` is a documented WORKAROUND -- the slower Python decode path, kept reachable so a decode the Rust engine gets wrong has a way around it while the defect is fixed -- not a supported mode to prefer |
| `--mapping` | strict rw-wps.mapping.v1 field/coordinate/target contract |
| `--namelist-input` | _(the parser declares no help text for this option)_ |
| `--namelist-support-report` | classify --wps-namelist/--namelist-input and print the exact stock-WRF versus woof support report as JSON |
| `--no-stock-wrf-export` | prepare the forecast only, and do not attempt the bonus unchanged-WRF wrfinput/wrfbdy export |
| `--output-root` | _(the parser declares no help text for this option)_ |
| `--physical-base-prepared` | HRRR base preparation whose sealed bridge can be reused |
| `--physical-input-provider` | posted native physical provider with a frozen member plan |
| `--physical-input-store` | sealed native physical snapshots on the target grid |
| `--physical-member-index` | original recipe member index in the posted provider |
| `--physical-output-store` | capture native mapped snapshots before real initialization |
| `--physics-profile` | optional assertion that the experiment IS this shipped single-domain suite, refused on any switch drift; omitted, the config's own physics is prepared as written and its WRF-verification status is reported (the HRRR route still requires a shipped profile: its cold-start evidence contract is profile-keyed) |
| `--pipeline-workers` | _(the parser declares no help text for this option)_ |
| `--prepare-workers` | _(the parser declares no help text for this option)_ |
| `--preprocess-backend {cuda,cpu,auto}` | select CUDA or deterministic parallel CPU preprocessing |
| `--preprocess-backend-reason` | _(accepted, but not listed by --help)_ |
| `--preprocess-workers` | threads for CPU preprocessing (default: this machine's CPUs, at most 8, the count its host RAM estimate was measured at; a larger count peaks above that estimate); under --preprocess-backend cuda, the threads of the host steps (masked soil, snow, skin temperature and sea ice), default every CPU |
| `--provenance ROLE=PATH` | composition provenance binding |
| `--root-preparation` | sealed output of the native HRRR root-preparation command; enables parallel d01..dNN hierarchy export for max_dom 1..21; the two namelists remain the topology authority |
| `--run-seconds` | _(the parser declares no help text for this option)_ |
| `--sealed-prepared-cache` | opt in to a prefix-sealed operational HRRR root preparation |
| `--show-physics-registry` | print the canonical GPUWM-owned physics registry v2 as JSON |
| `--show-source MODEL` | print one source declaration as JSON |
| `--show-support-matrix` | print the versioned native WRF compatibility matrix as JSON |
| `--source MODEL` | native source adapter id |
| `--source-format {grib1,grib2,netcdf}` | input format; must agree with the sealed rw-wps.mapping.v1 document |
| `--source-manifest, --source-sha256s` | SHA-256 file manifest covering every downloaded source file |
| `--source-manifest-sha256, --source-sha256s-sha256` | expected SHA-256 of --source-sha256s |
| `--source-orography` | _(the parser declares no help text for this option)_ |
| `--source-orography-variable` | _(the parser declares no help text for this option)_ |
| `--source-root` | the folder holding the source's files: the fetched HRRR cycle, the 20CRv3 member files --author-only reads, or, for a source whose fetch-route row declares its folder layout, the folder whose inputs and supplements it binds itself, authoring DIR/inputs.json and preparing into CONFIG-prepared beside the experiment config (CONFIG-prepared-2 and on once that exists) unless --output-root names one |
| `--source-top-pressure-pa` | smallest pressure represented by the selected source; used by --namelist-support-report to reject vertical extrapolation |
| `--static-cache` | _(the parser declares no help text for this option)_ |
| `--static-input` | _(the parser declares no help text for this option)_ |
| `--static-receipt` | _(the parser declares no help text for this option)_ |
| `--statics-corridor GRID_IDS` | also seal child-resolution statics over the ground each child can reach (the moving-nest corridor); bare flag covers every child domain, or pass comma-separated child grid ids (e.g. 2,3). Required before the prepared tree runner will honor a [relocation] follow source |
| `--stock-wrf-export {optional,required,off}` | mapped preparation's WRF file product: optional by default, required with early configuration admission, or off |
| `--stock-wrf-namelist-input` | unchanged-stock-WRF namelist matching the native hierarchy except for the certified longwave selection and the stock-only ghg_input and do_radar_ref keys; both declare use_theta_m = 0, the dry theta the exported files hold |
| `--supplement ROLE=PATH` | composition supplement binding; repeat roles for multiple files |
| `--valid-time` | initial UTC time, YYYY-MM-DDTHH or WRF form YYYY-MM-DD_HH:MM:SS. On --source hrrr this is the CYCLE; model time zero is cycle + --forecast-start-hour and is derived for every stage |
| `--validate-hrrr-domain PATH` | validate a strict HRRR target domain and its complete native interpolation window |
| `--validate-physics-plan PATH` | validate and resolve a gpuwm-physics-plan-v2 JSON document |
| `--vtable` | ERA5 GRIB1 Vtable |
| `--wps-namelist` | standard WPS geometry/static-selection namelist |
| `--wrf-version {3,4}` | with --namelist-support-report: the WRF line the namelist was written for, which selects only the Registry default an omitted &dynamics/use_theta_m takes (3: 0, dry theta, the line operational HRRR v4 runs; 4, the default: 1, moist theta) |

## `woof preset`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof preset check`

| argument | what it does |
|---|---|
| `name` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `woof preset export`

| argument | what it does |
|---|---|
| `name` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--out` | _(the parser declares no help text for this option)_ |
| `--seconds` | _(the parser declares no help text for this option)_ |
| `--start` | UTC YYYY-MM-DD_HH:MM:SS |

## `woof preset list`

Takes no options of its own.

## `woof preset prepare`

| argument | what it does |
|---|---|
| `name` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--cycle` | UTC YYYYMMDDHH |
| `--hours` | _(the parser declares no help text for this option)_ |
| `--mesh` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |

## `woof preset run`

| argument | what it does |
|---|---|
| `name` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--card` | _(the parser declares no help text for this option)_ |
| `--experimental` | run despite unmet requirements; a preset with none launches without it |
| `--fingerprint-every` | full-field hash cadence in steps; defaults to each history endpoint |
| `--grid` | _(the parser declares no help text for this option)_ |
| `--history-every-minutes` | _(the parser declares no help text for this option)_ |
| `--hours` | _(the parser declares no help text for this option)_ |
| `--init` | _(the parser declares no help text for this option)_ |
| `--init-source` | _(the parser declares no help text for this option)_ |
| `--invariant` | _(the parser declares no help text for this option)_ |
| `--lbc-dir` | _(the parser declares no help text for this option)_ |
| `--mesh` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |
| `--start-time` | _(the parser declares no help text for this option)_ |
| `--static` | _(the parser declares no help text for this option)_ |

## `woof preset show`

| argument | what it does |
|---|---|
| `name` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `woof preset verify-sources`

| argument | what it does |
|---|---|
| `name` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `woof race`

| argument | what it does |
|---|---|
| `recipe` | recipe ID or registry alias |

| option | what it does |
|---|---|
| `--box-label` | label for the measured box in a dry prediction |
| `--cycle` | HHZ (today UTC) or YYYY-MM-DDTHH |
| `--data-dir` | existing downloaded input cache |
| `--devices` | _(the parser declares no help text for this option)_ |
| `--dry` | predict from observed posting times; no forecast |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--hours` | forecast horizon (default 48; use 1 for the live door test) |
| `--hydro auto\|on\|off\|SETTINGS.toml` | hydro is opt-in: auto keeps the forecast default (off), on enables it, or select settings |
| `--initial-inputs` | separate analysis manifest for the go fallback |
| `--launch-delay` | seconds from analysis LastModified to launch (default 0) |
| `--launch-to-first-step` | seconds from launch to the first completed step |
| `--measurement-note` | measurement workload or prediction assumption, preserved in receipt |
| `--object-endpoint` | S3-compatible listing URL (supports a local HTTP test store) |
| `--outdir` | parent for a fresh receipt/run directory |
| `--poll-seconds` | source listing interval, at least 20 seconds (two sources; default 30) |
| `--posting-times` | timing study JSON; default fetch cycle's live S3 listing |
| `--products` | products defining online, passed to forecast |
| `--publish-lag` | seconds from hour close to products online |
| `--publish-receipt` | explicit hourly publish receipt |
| `--publish-root` | local web root; default RUN/www |
| `--receipt` | JSON with prediction_inputs (or top-level timing keys) |
| `--seconds-per-hour` | measured steady wall seconds per forecast hour, including output workload |
| `--smoke` | turn on HRRR-Smoke (transported smoke, its radiative feedback, RAVE fires); in 2.9.4 it costs about 11x per step on full CONUS (measured on 8x RTX 5090) |
| `--tiles {auto,off,on}` | _(the parser declares no help text for this option)_ |
| `--watch-timeout` | observation budget in seconds; never kills forecast |

## `woof remote`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof remote artifact-index`

| option | what it does |
|---|---|
| `--after-sequence` | last native sequence from the previous timeline page |
| `--domain` | selected committed domain, 1..999 |
| `--host` | existing SSH alias or user@host |
| `--identity` | existing local SSH identity path; contents are never copied |
| `--job` | job ID returned by start or list |
| `--json` | one versioned JSON result line; exit 0 or 2 |
| `--port` | SSH port (otherwise SSH configuration applies) |
| `--python` | absolute remote Python path with WOOF installed |
| `--ssh-config` | existing local OpenSSH configuration path |
| `--workspace` | existing absolute remote workspace directory |

## `woof remote list`

| option | what it does |
|---|---|
| `--host` | existing SSH alias or user@host |
| `--identity` | existing local SSH identity path; contents are never copied |
| `--json` | one versioned JSON result line; exit 0 or 2 |
| `--limit` | newest 1..100 jobs |
| `--port` | SSH port (otherwise SSH configuration applies) |
| `--python` | absolute remote Python path with WOOF installed |
| `--ssh-config` | existing local OpenSSH configuration path |
| `--workspace` | existing absolute remote workspace directory |

## `woof remote list-products`

| option | what it does |
|---|---|
| `--host` | existing SSH alias or user@host |
| `--identity` | existing local SSH identity path; contents are never copied |
| `--json` | one versioned JSON result line; exit 0 or 2 |
| `--port` | SSH port (otherwise SSH configuration applies) |
| `--python` | absolute remote Python path with WOOF installed |
| `--ssh-config` | existing local OpenSSH configuration path |
| `--workspace` | existing absolute remote workspace directory |

## `woof remote logs`

| option | what it does |
|---|---|
| `--cursor` | byte cursor from the previous log result |
| `--host` | existing SSH alias or user@host |
| `--identity` | existing local SSH identity path; contents are never copied |
| `--job` | job ID returned by start or list |
| `--json` | one versioned JSON result line; exit 0 or 2 |
| `--limit` | maximum bytes requested, 1..131072; each response may return less |
| `--port` | SSH port (otherwise SSH configuration applies) |
| `--python` | absolute remote Python path with WOOF installed |
| `--ssh-config` | existing local OpenSSH configuration path |
| `--workspace` | existing absolute remote workspace directory |

## `woof remote probe`

| option | what it does |
|---|---|
| `--host` | existing SSH alias or user@host |
| `--identity` | existing local SSH identity path; contents are never copied |
| `--json` | one versioned JSON result line; exit 0 or 2 |
| `--port` | SSH port (otherwise SSH configuration applies) |
| `--python` | absolute remote Python path with WOOF installed |
| `--ssh-config` | existing local OpenSSH configuration path |
| `--workspace` | existing absolute remote workspace directory |

## `woof remote resume`

| option | what it does |
|---|---|
| `--device` | card index or full GPU UUID on the node; omit to take the node's own default |
| `--dry-run` | review inputs and command; create and launch nothing |
| `--expected-checkpoint-set-sha256` | refuse any checkpoint set member changed since review |
| `--expected-checkpoint-sha256` | refuse a selected checkpoint changed since review |
| `--expected-config-sha256` | refuse inputs changed since the reviewed SHA-256 |
| `--expected-input-sha256` | refuse inputs changed since the reviewed SHA-256 |
| `--expected-prepared-sha256` | refuse a prepared receipt changed since review |
| `--expected-wps-sha256` | refuse inputs changed since the reviewed SHA-256 |
| `--from` | latest valid checkpoint, or its absolute remote path |
| `--geog-root` | existing absolute remote geography directory |
| `--host` | existing SSH alias or user@host |
| `--identity` | existing local SSH identity path; contents are never copied |
| `--job` | job ID returned by start or list |
| `--json` | one versioned JSON result line; exit 0 or 2 |
| `--outdir` | absolute remote output directory; the run claims its own stamped run folder inside it, so a directory that already collects runs takes another beside them, and a run folder that already exists is refused |
| `--port` | SSH port (otherwise SSH configuration applies) |
| `--prepared-root` | existing absolute remote prepared bundle; reuse it without fetch or preparation |
| `--products` | render catalog selectors, all, or none |
| `--python` | absolute remote Python path with WOOF installed |
| `--request-id` | name this launch attempt with the 32-character identity a previous reply or timeout printed, so the node answers a retry with the job that attempt already created; omit for a fresh attempt |
| `--section` | line for xsec: products: --section=LAT,LON,LAT,LON or a JSON file on the node; relative paths use the configuration directory (the source job's working directory on resume) |
| `--ssh-config` | existing local OpenSSH configuration path |
| `--workspace` | existing absolute remote workspace directory |
| `--wps-namelist` | with --prepared-root: exact absolute remote WPS authority required by a single-domain bundle |

## `woof remote review-plan`

| option | what it does |
|---|---|
| `--device` | card index or full GPU UUID on the node; omit to take the node's own default |
| `--expected-config-sha256` | _(the parser declares no help text for this option)_ |
| `--expected-plan-sha256` | _(the parser declares no help text for this option)_ |
| `--geog-root` | existing remote geography directory |
| `--host` | existing SSH alias or user@host |
| `--identity` | existing local SSH identity path; contents are never copied |
| `--json` | one versioned JSON result line; exit 0 or 2 |
| `--outdir` | new absolute remote output directory |
| `--plan` | saved local run-plan JSON to stage |
| `--port` | SSH port (otherwise SSH configuration applies) |
| `--prepared-root` | existing remote prepared bundle this plan's run option is relocated onto |
| `--python` | absolute remote Python path with WOOF installed |
| `--restart` | existing remote checkpoint this plan's run option is relocated onto |
| `--ssh-config` | existing local OpenSSH configuration path |
| `--workspace` | existing absolute remote workspace directory |
| `--wps-namelist` | existing remote WPS authority this plan's run option is relocated onto |

## `woof remote start`

| option | what it does |
|---|---|
| `--config` | existing absolute remote experiment TOML |
| `--device` | card index or full GPU UUID on the node; omit to take the node's own default |
| `--dry-run` | review inputs and command; create and launch nothing |
| `--expected-config-sha256` | refuse inputs changed since the reviewed SHA-256 |
| `--expected-input-sha256` | refuse inputs changed since the reviewed SHA-256 |
| `--expected-prepared-sha256` | refuse a prepared receipt changed since review |
| `--expected-wps-sha256` | refuse inputs changed since the reviewed SHA-256 |
| `--geog-root` | existing absolute remote geography directory |
| `--host` | existing SSH alias or user@host |
| `--identity` | existing local SSH identity path; contents are never copied |
| `--json` | one versioned JSON result line; exit 0 or 2 |
| `--outdir` | absolute remote output directory; the run claims its own stamped run folder inside it, so a directory that already collects runs takes another beside them, and a run folder that already exists is refused |
| `--port` | SSH port (otherwise SSH configuration applies) |
| `--prepared-root` | existing absolute remote prepared bundle; reuse it without fetch or preparation |
| `--products` | render catalog selectors, all, or none |
| `--python` | absolute remote Python path with WOOF installed |
| `--request-id` | name this launch attempt with the 32-character identity a previous reply or timeout printed, so the node answers a retry with the job that attempt already created; omit for a fresh attempt |
| `--section` | line for xsec: products: --section=LAT,LON,LAT,LON or a JSON file on the node; relative paths use the configuration directory (the source job's working directory on resume) |
| `--ssh-config` | existing local OpenSSH configuration path |
| `--workspace` | existing absolute remote workspace directory |
| `--wps-namelist` | with --prepared-root: exact absolute remote WPS authority required by a single-domain bundle |

## `woof remote start-plan`

| option | what it does |
|---|---|
| `--bundle-id` | _(the parser declares no help text for this option)_ |
| `--expected-bundle-sha256` | _(the parser declares no help text for this option)_ |
| `--expected-config-sha256` | _(the parser declares no help text for this option)_ |
| `--expected-input-sha256` | _(the parser declares no help text for this option)_ |
| `--expected-plan-sha256` | _(the parser declares no help text for this option)_ |
| `--host` | existing SSH alias or user@host |
| `--identity` | existing local SSH identity path; contents are never copied |
| `--json` | one versioned JSON result line; exit 0 or 2 |
| `--port` | SSH port (otherwise SSH configuration applies) |
| `--python` | absolute remote Python path with WOOF installed |
| `--request-id` | name this launch attempt with the 32-character identity a previous reply or timeout printed, so the node answers a retry with the job that attempt already created; omit for a fresh attempt |
| `--source-inputs-file` | completed local review whose selected raw inputs must still match |
| `--ssh-config` | existing local OpenSSH configuration path |
| `--workspace` | existing absolute remote workspace directory |

## `woof remote status`

| option | what it does |
|---|---|
| `--host` | existing SSH alias or user@host |
| `--identity` | existing local SSH identity path; contents are never copied |
| `--job` | job ID returned by start or list |
| `--json` | one versioned JSON result line; exit 0 or 2 |
| `--port` | SSH port (otherwise SSH configuration applies) |
| `--python` | absolute remote Python path with WOOF installed |
| `--ssh-config` | existing local OpenSSH configuration path |
| `--workspace` | existing absolute remote workspace directory |

## `woof remote stop`

| option | what it does |
|---|---|
| `--host` | existing SSH alias or user@host |
| `--identity` | existing local SSH identity path; contents are never copied |
| `--job` | job ID returned by start or list |
| `--json` | one versioned JSON result line; exit 0 or 2 |
| `--port` | SSH port (otherwise SSH configuration applies) |
| `--python` | absolute remote Python path with WOOF installed |
| `--ssh-config` | existing local OpenSSH configuration path |
| `--workspace` | existing absolute remote workspace directory |

## `woof remote sync-artifacts`

| option | what it does |
|---|---|
| `--cache-root` | owned local cache for this job's raw frame objects |
| `--domain` | selected committed domain, 1..999 |
| `--host` | existing SSH alias or user@host |
| `--identity` | existing local SSH identity path; contents are never copied |
| `--job` | job ID returned by start or list |
| `--json` | one versioned JSON result line; exit 0 or 2 |
| `--port` | SSH port (otherwise SSH configuration applies) |
| `--python` | absolute remote Python path with WOOF installed |
| `--reader-leases` | the visual reader retains shared OS leases for every frame clone |
| `--sequence` | exact native output commit sequence; omit for latest |
| `--ssh-config` | existing local OpenSSH configuration path |
| `--workspace` | existing absolute remote workspace directory |

## `woof remote sync-native-plots`

| option | what it does |
|---|---|
| `--cache-root` | local folder for selected native PNG galleries |
| `--domain` | selected committed domain, 1..999 |
| `--height` | panel height in pixels, 256..4096; default 900 |
| `--host` | existing SSH alias or user@host |
| `--identity` | existing local SSH identity path; contents are never copied |
| `--job` | job ID returned by start or list |
| `--json` | one versioned JSON result line; exit 0 or 2 |
| `--layout {auto,fixed}` | auto sizes each canvas from its domain; fixed keeps the requested size, default 1200x900 |
| `--port` | SSH port (otherwise SSH configuration applies) |
| `--products` | render catalog selectors separated by commas; omit for this run's own selection, empty for the node's default set |
| `--profile {viewer-2d-v1,full-science-v1}` | native processing profile the gallery draws from; omit for the compact viewer profile |
| `--python` | absolute remote Python path with WOOF installed |
| `--sequence` | exact native output commit sequence |
| `--ssh-config` | existing local OpenSSH configuration path |
| `--theme` | built-in theme or theme JSON path on the node; files may extend woof-light or woof-dark |
| `--width` | panel width in pixels, 256..4096; default 1200 |
| `--workspace` | existing absolute remote workspace directory |

## `woof remote sync-outputs`

| option | what it does |
|---|---|
| `--after-sequence` | last native sequence from the previous timeline page |
| `--cache-root` | owned local directory for this run's committed output set |
| `--domain` | selected committed domain, 1..999 |
| `--host` | existing SSH alias or user@host |
| `--identity` | existing local SSH identity path; contents are never copied |
| `--job` | job ID returned by start or list |
| `--json` | one versioned JSON result line; exit 0 or 2 |
| `--port` | SSH port (otherwise SSH configuration applies) |
| `--python` | absolute remote Python path with WOOF installed |
| `--ssh-config` | existing local OpenSSH configuration path |
| `--workspace` | existing absolute remote workspace directory |

## `woof remote sync-processed-frame`

| option | what it does |
|---|---|
| `--cache-root` | owned local directory for immutable native processed stores |
| `--domain` | selected committed domain, 1..999 |
| `--host` | existing SSH alias or user@host |
| `--identity` | existing local SSH identity path; contents are never copied |
| `--job` | job ID returned by start or list |
| `--json` | one versioned JSON result line; exit 0 or 2 |
| `--port` | SSH port (otherwise SSH configuration applies) |
| `--python` | absolute remote Python path with WOOF installed |
| `--sequence` | exact native output commit sequence; omit for latest |
| `--ssh-config` | existing local OpenSSH configuration path |
| `--workspace` | existing absolute remote workspace directory |

## `woof remote sync-processed-frame-v2`

| option | what it does |
|---|---|
| `--cache-bytes` | local viewer cache budget in bytes; default 2 GiB |
| `--cache-root` | owned bounded local cache for compact native fields |
| `--domain` | selected committed domain, 1..999 |
| `--expected-run-id` | require this exact native producer run identity |
| `--host` | existing SSH alias or user@host |
| `--identity` | existing local SSH identity path; contents are never copied |
| `--job` | job ID returned by start or list |
| `--json` | one versioned JSON result line; exit 0 or 2 |
| `--port` | SSH port (otherwise SSH configuration applies) |
| `--prefetch-sequences` | up to eight committed loop sequences separated by commas |
| `--products` | render catalog selectors separated by commas; an empty value takes the node's own default set |
| `--profile {viewer-2d-v1,full-science-v1}` | _(the parser declares no help text for this option)_ |
| `--python` | absolute remote Python path with WOOF installed |
| `--reader-leases` | viewer retains shared native-store object leases |
| `--sequence` | exact committed sequence; omit for latest |
| `--ssh-config` | existing local OpenSSH configuration path |
| `--workspace` | existing absolute remote workspace directory |

## `woof render`

`--pair-labels`, `--pair-subtitle` and `--pair-title` title and label the paired CPU-vs-GPU figure `--pair` composes.

| argument | what it does |
|---|---|
| `[WRFOUT ...]` | wrfout NetCDF files, a wrfout directory, or the output folder printed by woof go |

| option | what it does |
|---|---|
| `--annotate FILE.json` | rust engine: override the panel title and the three subtitle slots (title, title_suffix, subtitle_left, subtitle_center, subtitle_right). A short badge belongs in the centre slot; anything sentence-length belongs on the left, which owns the row's width |
| `--barbs` | rust engine: draw the wind as BARBS, overruling both the automatic choice and any inherited RUSTWX_WIND_STREAMLINES |
| `--bounds W,E,S,N` | CF geographic extent in degrees, e.g. --bounds=-125,-66,24,50 |
| `--cf FILE.nc` | render a native CF diagnostic grid with file source provenance |
| `--cf-table FILE.json` | CF variable product table (default: packaged production rows) |
| `--compare REFERENCE[,REFERENCE...]` | draw each frame beside ordered native references (for example hrrr,mrms or hrrr,rrfs,mrms), on one grid and colour scale. Forecast references share the valid time; observation panels label their actual observation time. The native --list-products catalogue lists references and supported products. WRFOUT may be frames or a run folder. The map rasters are filled on a CUDA card: the first one CUDA_VISIBLE_DEVICES exposes (CUDA's device 0 when it is unset; the run names the card it uses). RUSTWX_RASTER_DEVICE=cpu fills them on the CPU |
| `--compare-cache DIR` | where fetched reference subsets are kept between renders (default ~/.woof/cache/compare-reference) |
| `--compare-cycle YYYYMMDDHH` | compare against THIS reference cycle instead of the run's own start time |
| `--compare-difference {auto,on,off}` | run-minus-reference panels: 'auto' (default) draws continuous differences with one reference and only field panels with a reference list; 'on' adds a difference for each reference whose units have a difference ladder; 'off' never |
| `--compare-gallery DIR` | also copy every sheet, flat, into DIR; DIR may equal the output directory. Multi-reference sheets keep the requested panel order |
| `--compare-label TEXT` | the title over the run's panel (default WOOF) |
| `--compare-offline` | never fetch: use only --compare-reference-dir and the cache |
| `--compare-reference-dir DIR` | read the reference's GRIB2 files from DIR (by their published names, flat or under the bucket's own folders) before fetching anything |
| `--context-wrfout FILE` | _(accepted, but not listed by --help)_ |
| `--contour-file FILE.nc` | CF contour companion on the fill grid (default: --cf file) |
| `--contours NAME[,NAME]` | CF probability contours at each table row's contour_level |
| `--diff ('A_RUN', 'B_RUN')` | draw each product as run A minus run B: two folders of wrfout frames (or two files), paired by valid time; refused by name when the runs do not share a grid (rust engine; no wrfout arguments) |
| `--diff-labels ('A_NAME', 'B_NAME')` | the two runs' names on the difference panels (default: the two folder names) |
| `--diff-sheet` | also draw an A \| B \| A minus B sheet per product |
| `--diff-table {table,auto,FILE}` | the difference colour bars: table (default: the fixed per-product bars, so the frames of a loop share one), auto (every bar fitted to its own difference) or a JSON file whose keys replace the built-in table's (schema in tools/rustwx/crates/rustwx-render/data/difference.json; RUSTWX_DIFFERENCE_TABLE is the environment spelling). Under every choice a fixed bar that would draw a nonzero difference as blank is fitted to it, and every panel prints the largest \|A-B\| and how many cells differ |
| `--dpi N` | PNG resolution, matplotlib engine (default 150) |
| `--engine {auto,rust,matplotlib}` | render engine: the vendored Rusty Weather renderer (campaign plot quality; 151 implicit-render catalog candidates per file) or the matplotlib workaround; 'auto' (default) uses rust whenever its binary is built and probes as runnable, and REFUSES otherwise rather than drawing weather fields with matplotlib -- 'matplotlib' asks for that workaround by name and announces itself |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--grib2-out DIR` | also export every input frame as native surface and pressure-level GRIB2 |
| `--heavy` | rust engine: also compute the heavy ECAPE product family at import (SBECAPE/SBNCAPE/SBECIN, ECAPE SCP/EHI/...; adds substantial per-frame import time) |
| `--hubwind-config JSON\|TOML` | hub heights, turbine, exceedance and ramp settings; JSON options or [hubwind] TOML |
| `--inputs-from FILE` | _(accepted, but not listed by --help)_ |
| `--isotherms L,L,...[@H]` | rust engine: the isotherms (C) drawn on every section, with an optional highlighted one after '@' (e.g. 0,-5,-10,-15,-20@-10); default 0,-10,-20,-30,-40 |
| `--layout {nested,flat}` | how the PNGs are arranged inside this render's run folder: 'nested' (default) files each picture at <run folder>/<domain>/<product>/<valid-day>/<file>.png (domain as d02-3km / d05-111m / native_grid, valid-day as YYYY-MM-DD), so a run's thousands of frames are separated by nest, by chart and by day and a script can predict a path without globbing; 'flat' writes every picture directly into the run folder, which is what releases up to 2.4.1 did and is kept only for consumers still written against it (with --run-stamp off it is the v2.4.1 tree exactly) |
| `--list-products` | list the engine's product catalog with per-file availability (why each product is or is not renderable from this wrfout) instead of rendering |
| `--observation FILE` | CF observation grid or MRMS GRIB2/gz beside the diagnostic on one ladder and extent |
| `--observation-variable NAME` | variable in a CF observation file (default: --variable) |
| `--out DIR` | where the PNGs go (default out/render). Each render claims its own timestamped run folder under it, so two renders never overwrite each other; point it at an existing run-... folder to draw into that one |
| `--overlays FILE.json` | rust engine: draw map overlays given in geographic DEGREES on every panel -- lines, closed boxes, markers, labels and range rings. This is the seam a boundary-zone frame, tile seams, storm-report markers and radar sites needed; the schema is documented in tools/rustwx/crates/rustwx-products/src/geographic_overlays.rs. Omitted, the renderer runs no overlay code and the PNGs are byte-identical |
| `--pair ('A_DIR', 'B_DIR')` | compose two runs' rendered PNG directories into labeled side-by-side comparison sheets (no wrfout arguments) |
| `--pair-labels ('LEFT', 'RIGHT')` | panel labels (default: the two directory names) |
| `--pair-subtitle TEXT` | optional pair-sheet subtitle |
| `--pair-title TITLE` | pair-sheet title (default 'Paired comparison') |
| `--products LIST` | comma-separated products: refl, t2, wind10, precip, olr, or 'all' (default); with the rust engine these are refl=composite_reflectivity, t2=2m_temperature, wind10=10m_wind_speed_and_direction, precip=total_qpf, olr=var:wrf_olr, raw catalog slugs (sbcape, srh_0_1km, ...) also work and 'all' renders its full catalog |
| `--radar-colors {standard,classic}` | rust engine: the colour tables the reflectivity and radial velocity products draw with -- standard (the radar tables, the default) or classic (the reflectivity ladder and blue-red velocity scale before 2.8.5). One name selects every radar-table product; RUSTWX_RADAR_COLORS is the environment spelling, which `woof go` and `woof run` renders also read |
| `--receipt FILE.json` | CF input, output and per-stage timing receipt |
| `--run-stamp {on,off}` | put this run's PNGs in its own timestamped folder under --out (default on): --out/run-<YYYYMMDD>-<HHMMSS>Z_i<YYYYMMDD><HHMM>Z/ (launch instant UTC, then the model initialisation time; the _i part is omitted when the run's init time cannot be read). Successive runs of one configuration then never overwrite or interleave each other. 'off' writes straight into --out, which is what releases up to 2.4.1 did; it is kept only for a consumer still written against that and is a workaround, not a supported alternative |
| `--section lat,lon,lat,lon\|FILE.json` | rust engine: the line the vertical-section products (xsec:<fill>[/<overlay>...] in --products, any 3-D wrfout field on a height axis) are cut along; a JSON file gives {start, end} or a {points, extend_km} polyline |
| `--section-across KM` | rust engine: also draw each section product across the line, this many km long, through the fill's maximum column |
| `--section-size WxH` | the size a cross-section is drawn at; absent, a section is landscape 2:1 at the map's width, because a vertical cut handed the map's own size comes out portrait |
| `--section-top-km N` | rust engine: the ceiling of a section's height axis, 1-40 km; absent, the engine fits up to 14 km, which draws a shallow feature in the bottom fourteenth of the frame -- give it 3 for a boundary-layer cut |
| `--series` | render compatible files from each run/domain/episode as one timeline, including multi-hour products |
| `--size WxH\|auto` | output pixels, rust engine; 'auto' (the default) sizes each canvas from its domain's shape, WxH draws a fixed canvas |
| `--source-label TEXT` | model/provenance label stamped on every plot (default 'WOOF <the executing version>'); set it when rendering wrfout files this model did not produce, so the sheet does not claim them |
| `--streamlines` | rust engine: draw the wind as STREAMLINES instead of barbs on every product that carries a wind layer. Without either flag the engine keeps its automatic choice (streamlines on curvilinear and projected grids, barbs on plain lat/lon), and the RUSTWX_WIND_STREAMLINES environment variable still works; this flag and --barbs outrank it |
| `--theme NAME\|FILE.json` | rust engine: the render theme -- a built-in name (default, dark) or a JSON theme file naming the surface, the inks, the basemap linework, the colorbar chrome, the fonts and the colormap overrides (schema in tools/rustwx/crates/rustwx-render/src/theme.rs; RUSTWX_THEME is the environment spelling). Omitted, the engine draws its own look and the PNGs are byte-identical |
| `--timeidx N\|all` | frame index within each file (within each timeline with --series), or 'all' (default) |
| `--valid UTC` | CF valid time in ISO UTC (default: first record) |
| `--variable NAME` | CF field selected by variable name; product and units come from the CF table |

## `woof report`

| argument | what it does |
|---|---|
| `[RUNDIR]` | the run directory to collect from (default: the current directory, so `woof report` with no arguments inside a run works; when the current directory holds no receipt but out/run below it does, that one is read and the manifest says so) |

| option | what it does |
|---|---|
| `--dry-run, --list` | print the manifest -- everything that would be included, redacted and reported missing -- and write nothing |
| `--exit-code N` | the exit status the failing command returned, recorded in the manifest (nothing on disk records it) |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--log FILE` | an additional log file to include, for output that was redirected outside the run directory (repeatable) |
| `--output PATH` | where to write the zip: a file path, or a directory to name it in (default: the current directory, falling back to the system temporary directory and then your home directory if a write is refused) |

## `woof research`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof research attributes`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--json` | emit supported attributes, units, extrema, reductions and model-level semantics as compact JSON |

## `woof research catalog`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--json` | emit the complete native catalog and hardware profiles as compact JSON |

## `woof research create`

| argument | what it does |
|---|---|
| `configuration_id` | exact configuration id from research catalog; existing-state recipes must continue from their supplied archive or scenario |

| option | what it does |
|---|---|
| `--ack` | native source acknowledgement code; repeat for each required acknowledgement |
| `--cycle` | input cycle in UTC, YYYY-MM-DDTHH; latest uses native source discovery |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--hardware-class {auto,8,12,16,24,32}` | resource profile; does not declare available memory |
| `--hours` | explicit whole-hour study duration; omitted keeps the recipe's duration |
| `--json` | emit the created workspace receipt, geometry, resource budget and required next steps as compact JSON |
| `--name` | descriptive name for this study; omitted keeps the recipe title |
| `--nz` | explicit vertical-level override |
| `--out` | new experiment TOML path; existing configuration or companion files are never replaced |
| `--physics-profile` | native physics suite override; omitted keeps the admitted source default |
| `--point LAT,LON` | centre latitude and longitude; native sizing fits coverage to the actual budget and enforces the recipe's minimum span |
| `--polygon GEOJSON` | GeoJSON study footprint; retain its full required coverage |
| `--source` | native input source id (default gfs); source, cycle and research method must be compatible |
| `--tiles {off,auto,on}` | explicit native tile-streaming mode; host-memory and transfer costs remain subject to admission |
| `--vram-gib` | explicit capacity estimate; omit to measure actual total/free memory |

## `woof research hardware`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--hardware-class {auto,8,12,16,24,32}` | resource profile; does not declare available memory |
| `--json` | emit capacity, free-memory budget and selected resource profile as compact JSON |
| `--vram-gib` | explicit capacity estimate; omit to measure actual total/free memory |

## `woof resume`

| argument | what it does |
|---|---|
| `CONFIG` | the SAME config the interrupted run used; the restart identity check refuses any other. An argument that is not a readable file is tried with the .toml extension a file manager hides, then against --outdir, and last against the configuration the run in --outdir recorded for itself (child.toml, experiment.toml or captured-config-<run id>.toml) |

| option | what it does |
|---|---|
| `--allow-shared-gpu` | UNSUPPORTED: permit another substantial CUDA compute context; device verification and the GPUWM UUID lock remain enforced |
| `--directory-input-hash {inventory,content}` | how declared directory inputs (the static geography tree) are bound to this run's identity: 'inventory' (default) uses relative path, size, and mtime; 'content' reads every file and uses its SHA-256. Use 'content' when two runs being compared for byte identity stage their geography separately, and when an mtime-preserving change to that tree must not go unnoticed (docs/public/DETERMINISM.md). Also settable as WOOF_DIRECTORY_INPUT_HASH. |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--from CKPT\|latest` | explicit gpuwmrst_*.npz checkpoint, or 'latest' (default) to take the newest set in --outdir whose members validate |
| `--gpu-uuid GPU-UUID` | physical GPU UUID to lock (required on multi-GPU hosts) |
| `--health-debug` | enable debug phase health attribution hooks |
| `--icechunk-store URL` | Output store base: local path or s3://bucket/prefix; domain and member paths are appended (default RUN/icechunk) |
| `--keep-member-files` | also retain every member's full history files |
| `--members N` | make an N-member ensemble with aggregate products |
| `--no-memory-gate` | run a case whose priced peak envelope exceeds this card's free memory anyway, as `woof go --no-memory-gate` does: the envelope is an upper bound and the card's own allocation then decides; a model state too big to build at all is still refused |
| `--no-supervise` | run the experiment in this process (escape hatch; disables fresh-process recovery and exclusive-GPU supervision) |
| `--outdir OUT` | the interrupted run's wrfout/checkpoint directory (default out/run) |
| `--output-config TOML` | Read [output] formats and icechunk_store from TOML; CLI values override it |
| `--output-format FORMATS` | Output-format table rows, comma separated (default netcdf); netcdf,icechunk adds atomic native Zarr v3 history |
| `--prep-timeout SECONDS` | optional preparation heartbeat timeout; default is no timeout until integration begins |
| `--restart-roster JSON` | continue the exact original members from a durable ensemble restart roster |
| `--supervisor-max-restarts N` | fresh-process recovery attempts (default 3) |

## `woof run`

`--allow-shared-gpu`, `--gpu-uuid`, `--prep-timeout` and `--supervisor-max-restarts` are the command-line spellings of settings STREAMING.md documents only as run-plan keys.

| argument | what it does |
|---|---|
| `[CONFIG]` | experiment TOML ([experiment]/[[domain]] tables, as emitted by `woof domain` or `woof import-namelist`; config-driven runs declare their inputs in [case_data]). A legacy [run]-table RunConfig naming a registered case is also accepted. |

| option | what it does |
|---|---|
| `--allow-shared-gpu` | UNSUPPORTED: permit another substantial CUDA compute context; device verification and the GPUWM UUID lock remain enforced |
| `--analysis BUNDLE` | WRF inputs: start from an analysis bundle (wrf-analysis-bundle/1, docs/analysis-bundle-wrf.md): a directory or its manifest.json. Every file is checked by SHA-256, and a grid, vertical coordinate or static field other than the run's is refused by name. The analysis (or, with --iau on, its background) becomes the run's wrfinput; with --restart or --restart-from it is added at the restart instant |
| `--analysis-work DIR` | with --analysis: where the door writes what it derives (the run directory, the background, the rebased boundary, the plan); default <outdir>.analysis-work/<manifest digest> |
| `--benchmark` | WRF inputs: time the forecast as a benchmark. Draws no products and runs no post-run verification, then prints the read, init, first step, steps and write times WRF's rsl.out reports, and the process total (also OUTDIR/benchmark-timing.json) |
| `--checkpoint-at TIMES` | WRF inputs: write only these scheduled checkpoints, as comma-separated WRF valid times (2022-06-13_22:00:00) or `end`; each must be a restart_interval instant of the run. Absent writes every scheduled checkpoint |
| `--directory-input-hash {inventory,content}` | how declared directory inputs (the static geography tree) are bound to this run's identity: 'inventory' (default) uses relative path, size, and mtime; 'content' reads every file and uses its SHA-256. Use 'content' when two runs being compared for byte identity stage their geography separately, and when an mtime-preserving change to that tree must not go unnoticed (docs/public/DETERMINISM.md). Also settable as WOOF_DIRECTORY_INPUT_HASH. |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--family-weights ai=0.5,physics=0.5` | family probability masses, must sum to 1; defaults to member shares |
| `--gpu-uuid GPU-UUID` | physical GPU UUID to lock (required on multi-GPU hosts) |
| `--health-debug` | enable debug phase health attribution hooks |
| `--history-compression {deflate,none}` | history frame storage: deflate (default; lossless NetCDF-4, the same values in about a third of the disk) or none (classic CDF-2). Also GPUWM_HISTORY_COMPRESSION |
| `--hydro on\|off\|SETTINGS.toml` | hydro is opt-in: on couples it wherever a hydrofabric covers the domain, a TOML file selects settings (as on woof go); without it the run has no hydro |
| `--iau {on,off}` | with --analysis: 'on' starts from the background and forces the increment over the IAU window inside every RK stage; 'off' inserts the analysis at the start. Default: on when the bundle carries increment.nc, off otherwise |
| `--iau-boundary {background,analysis}` | with --analysis and IAU on a cold start: what the lateral boundary's first record is the coupling of. 'background' (the default, measured the smaller rim disturbance) keeps the boundary of the state the run starts from; 'analysis' rebases it onto the analysis and leaves the whole boundary zone to it |
| `--iau-fields {all,dynamics}` | with --analysis and IAU: 'all' (the default) forces every field the increment carries; 'dynamics' forces winds, theta, vapour and geopotential and puts the hydrometeors in at the window start |
| `--iau-mode {oneshot,3d,split,4d}` | with --analysis: how the increment enters. oneshot = --iau off; 3d (the default with --iau on) = a held tendency over a forward window. split and 4d are named gaps of this door and are refused by name |
| `--iau-window-seconds S` | with --analysis and IAU: the window, overriding the bundle's iau.window_s |
| `--icechunk-store URL` | Output store base: local path or s3://bucket/prefix; domain and member paths are appended (default RUN/icechunk) |
| `--keep-member-files` | also retain every member's full history files |
| `--members N` | make an N-member ensemble with aggregate products |
| `--met-em DIR` | WPS metgrid directory with met_em.d0*.nc and producing namelist.input; native WOOF initialization |
| `--no-memory-gate` | run a case whose priced peak envelope exceeds this card's free memory anyway, as `woof go --no-memory-gate` does: the envelope is an upper bound and the card's own allocation then decides; a model state too big to build at all is still refused |
| `--no-supervise` | run the experiment in this process (escape hatch; disables fresh-process recovery and exclusive-GPU supervision) |
| `--outdir OUT` | wrfout output directory |
| `--output-config TOML` | Read [output] formats and icechunk_store from TOML; CLI values override it |
| `--output-format FORMATS` | Output-format table rows, comma separated (default netcdf); netcdf,icechunk adds atomic native Zarr v3 history |
| `--prep-timeout SECONDS` | optional preparation heartbeat timeout; default is no timeout until integration begins |
| `--preprocess-backend {cuda,cpu,auto}` | CONFIG: where the root domain's preparation runs, overriding [case_data] preprocess_backend (default: that key, else auto, which prepares on the CPU when the card reads busy or cannot hold it); pin it so two runs you compare start from the same preparation. Nests prepare on the card either way |
| `--recipe {wofs,time-lagged,multi-model,surface-state,member-roster}` | take each ensemble member from a real source trajectory: time-lagged runs earlier cycles of the config's own source over the same window, multi-model runs the trajectories --trajectories lists, surface-state runs seeded soil moisture scales and SST offsets from [ensemble.perturbation], member-roster runs named land and fixed surface arms from [ensemble.member_variants] |
| `--restart RST` | resume from a gpuwmrst restart file written by an earlier run of the SAME config (only the forecast length / output and restart cadence and each domain's history window, history_begin_s / history_end_s, may differ); restart writing itself is the restart_interval_s config key |
| `--restart-from WRFRST` | continue a WRF run from its restart file (wrfrst_d01_<time>, any suffix): the state, physics and clock are the file's, and the namelist.input and wrfbdy_d01 beside it (or in --wrfinput DIR) supply the settings and the lateral forcing. With --run-seconds, the window after the restart instant |
| `--restart-roster JSON` | continue the exact original members from a durable ensemble restart roster |
| `--rrtmg-variant {rrtmg_legacy,rte-rrtmgp}` | WRF inputs: preserve legacy RRTMG by default; choose rte-rrtmgp to change radiation |
| `--run-seconds` | shorten a --wrfinput or --met-em run inside its forcing coverage |
| `--soil-source DIR` | WRF inputs: original met_em and Vtable directory for automatic soil-water recovery; defaults to the input directory |
| `--supervisor-max-restarts N` | fresh-process recovery attempts (default 3) |
| `--thompson-version {wrf_461,wrf_44}` | WRF inputs: override the classic Thompson generation selected from producer TITLE |
| `--trajectories FILE` | the multi-model member list: a JSON or TOML file of {source, cycle[, member]} entries, one per member (selects --recipe multi-model) |
| `--urban-fvg` | _(the parser declares no help text for this option)_ |
| `--urban-table-dir` | _(the parser declares no help text for this option)_ |
| `--urban-treeoption {0,1}` | _(the parser declares no help text for this option)_ |
| `--urban-ttscheme {0,1}` | _(the parser declares no help text for this option)_ |
| `--urban-ucm-generation {wrf_471,wrf_480}` | _(the parser declares no help text for this option)_ |
| `--vertical-grid` | met_em: native, wrf-auto, or explicit:PATH eta grid |
| `--vertical-levels` | met_em: requested level count for the selected vertical grid |
| `--wrfinput DIR` | WRF real.exe directory containing wrfinput_d0*, wrfbdy_d01 and producing namelist.input (instead of CONFIG) |

## `woof run-plan`

| argument | what it does |
|---|---|
| `[PLAN.json]` | a gpuwm.run-plan.v1 document: which route to execute, which config to execute it with, and where the outputs land |

| option | what it does |
|---|---|
| `--catalog` | print the renderer's product catalog as one JSON document -- what may be put in the render_products run option, and in local_run the products a local run can draw -- and run nothing; needs no plan, and a PLAN narrows local_run to that run's length |
| `--estimate` | print this plan's VRAM estimate, output-frame counts, download and disk bytes as one JSON document, and run nothing |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--no-probe` | with --readiness: compute the schedule from the source table only and ask no host |
| `--no-readiness` | with --probe, report the device inventory only: the NVML-only half, safe to poll on a card that is busy |
| `--physics-profiles` | print the per-source physics menu as one JSON document -- every registered source crossed with every shipped physics suite, saying which pairings this product can actually prepare, why each refused one is refused, which suite that source's bare run binds, and which suites run shortwave with longwave off -- and run nothing; needs no plan |
| `--probe` | print this machine's device inventory and runtime-estate readiness as one JSON document; needs no plan. The device inventory is NVML only and creates no CUDA context; the readiness half runs `woof doctor`'s checks, which verify by execution and do create one |
| `--readiness` | print gpuwm.readiness.v1 for the window this plan fetches and run nothing: exit 0 ready (or nothing to probe), 75 not yet (with expected_ready_at and retry_after_seconds), 2 refused (the window can never start) |
| `--resolve` | print the fully resolved configuration plus every automatic resolution as one JSON document, and run nothing |
| `--sources` | print the source registry as one JSON document -- every registered source, what each one's row declares, and which run-plan route can drive it from an intent -- and run nothing; needs no plan |

## `woof scoreboard`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--request` | woof.scoreboard.request.v1 JSON request for the Rust scorer |

## `woof setup`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--from DIR` | stage the bridge and table artifacts from a local directory instead of downloading (offline installs); verification is identical. Does not apply to --with-geog, which has its own --source |
| `--with-earth2` | also build the managed Earth2Studio environment (torch, earth2studio with the fcn3, cyclone and diagnostic extras) and the TempestExtremes tools that woof tc-tracks and woof earth2-diagnose run; several GB of wheels, built at WOOF_EARTH2_ENV (default ~/.local/share/woof/earth2-env) |
| `--with-geog` | also stage the WPS_GEOG static geography (~2.7 GB compressed, ~34 GB unpacked); the size is printed before anything downloads |

## `woof sim`

| argument | what it does |
|---|---|
| `PREPARED_ROOT` | a prepared tree written by `woof prep --output-root`, by the rw-wps console script, or by `woof go`'s preparation stage |

| option | what it does |
|---|---|
| `--devices N` | resident slab count; replaces [devices] count (on a tree, for every grid [devices] domains names) |
| `--devices-table JSON` | [devices] table as JSON (count, grid, ids, transport, and on a tree domains); validated by the runner, without modifying the prepared configuration or its digests |
| `--experiment-config TOML` | the experiment TOML this preparation was bound to (the tree runner binds its digest; the single-domain runner binds it through the proof) |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--health-debug` | write per-step health diagnostics with the prepared hierarchy runner |
| `--icechunk-store URL` | Output store base: local path or s3://bucket/prefix; domain and member paths are appended (default RUN/icechunk) |
| `--io-mode {history}` | history output (the only mode this seam offers; `--io-mode none` is a runner-level diagnostic, reachable through --print-command) |
| `--no-memory-gate` | restore a forecast whose priced peak envelope exceeds this card's free memory anyway, as `woof go --no-memory-gate` does: the envelope is an upper bound and the card's own allocation then decides; a model state too big to build at all is still refused |
| `--outdir DIR` | where this forecast's output goes. By default it is the parent of one timestamped run folder per forecast, so running the same prepared tree twice never merges two runs' wrfout frames and report.json. Point it at an existing run-... folder and that folder is used as given -- and refused if a forecast is already in it |
| `--output-config TOML` | Read [output] formats and icechunk_store from TOML; CLI values override it |
| `--output-format FORMATS` | Output-format table rows, comma separated (default netcdf); netcdf,icechunk adds atomic native Zarr v3 history |
| `--physics-profile ID` | optional assertion that the hash-bound experiment IS this shipped suite; omitted, the experiment's own suite runs as written |
| `--print-command` | print the exact runner command, with every digest filled in, and exit without running it -- the documented boundary a third-party script writes to. The --outdir in that line is this run's own timestamped folder, named but not created: asking the question spends nothing, and the runner makes the directory when you run the line. The line is quoted for PowerShell on Windows and for a POSIX shell elsewhere |
| `--progress-format {text,jsonl,off}` | how the run reports progress. Omitted, the runner's own default applies, which is the WRF-shaped `Timing for main:` line per step per domain on this terminal -- watching it run is the reason to run the stage alone. `jsonl` is what `woof go` passes, because it owns the runner's stdout; pass it here when you are hosting this stage the same way |
| `--render-dir DIR` | picture directory (default OUTDIR/png); ignored without --render-products |
| `--render-products SPEC` | render selected products from every committed history frame of every grid as it lands, while the forecast runs: comma-separated catalog selectors, 'all', or 'none'. Omitted means no rendering |
| `--restart RST` | resume a prepared single-domain checkpoint or any member of a hierarchy checkpoint set into a fresh output folder; the runtime validates config, inputs and checkpoint identity without changing settings |
| `--run-stamp {on,off}` | put this run's wrfout, report.json and receipts in its own timestamped folder under --outdir (default on): --outdir/run-<YYYYMMDD>-<HHMMSS>Z_i<YYYYMMDD><HHMM>Z/ (launch instant UTC, then the model initialisation time; the _i part is omitted when the run's init time cannot be read). Successive runs of one configuration then never overwrite or interleave each other. 'off' writes straight into --outdir, which is what releases up to 2.4.1 did; it is kept only for a consumer still written against that and is a workaround, not a supported alternative |
| `--runner {auto,single,tree}` | which runner arm to use. 'auto' (default) reads it off the bundle's own schema and domain count; the explicit values exist for a caller who knows better and wants to be refused precisely when they do not |
| `--sealed-forcing-extension` | use the existing prepared-tree append-only forcing prefix contract when writing or restoring checkpoints |
| `--simulated-radar-table JSON` | [simulated_radar] options as JSON; overrides output options without editing a prepared configuration |
| `--stream-init {auto,resident,store}` | single-domain streamed initialization: auto prices both roads; resident or store forces that road |
| `--tiles JSON` | single-domain streaming override as a JSON [tiles] mapping; validated by the runner, without modifying the prepared configuration or its digests |
| `--wps-namelist WPS` | the namelist.wps this preparation consumed; required for a single-domain forecast, unused by the tree runner |

## `woof simulated-radar`

| argument | what it does |
|---|---|
| `[history ...]` | history files or run directories |

| option | what it does |
|---|---|
| `--band {S,C,X}` | override the site band |
| `--config` | TOML with a [simulated_radar] table |
| `--describe` | inspect installed native capabilities, accepted inputs and routes as JSON |
| `--estimate` | report native scan dimensions and memory admission without generating radar |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--formats` | comma-separated output formats |
| `--input-kind {wrf,native-columns}` | full WRF histories or native-atmosphere.columns/v1 transports |
| `--list-sites` | show simulation sites and world reference rows with band metadata |
| `--outdir` | run root receiving radar/manifest.json |
| `--sites` | auto or comma-separated radar IDs |
| `--timing {history,scan}` | history scans each saved snapshot; scan lets rays use neighboring history times (overrides the --config table; default history) |
| `--wavelength-mm` | select legacy S at 107.068735 mm or a 53.5/33.3 mm C/X table |

## `woof solar`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof solar catalog`

Takes no options of its own.

## `woof solar fetch-observations`

| option | what it does |
|---|---|
| `--date` | _(the parser declares no help text for this option)_ |
| `--directory` | _(the parser declares no help text for this option)_ |
| `--network {surfrad,solrad}` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |
| `--prefix` | _(the parser declares no help text for this option)_ |

## `woof solar products`

| option | what it does |
|---|---|
| `--out` | _(the parser declares no help text for this option)_ |
| `--products` | comma-separated solar slugs; default all |
| `--request` | _(the parser declares no help text for this option)_ |
| `--systems` | _(the parser declares no help text for this option)_ |
| `--wrfout` | _(the parser declares no help text for this option)_ |

## `woof soundings`

| argument | what it does |
|---|---|
| `run` | single run, ensemble member folder, wrfout, or rw-store folder |

| option | what it does |
|---|---|
| `--columns DIR` | write the run's any-point column store into DIR instead of pictures (no --point needed) |
| `--domain` | domain identifier to select |
| `--every-minutes` | with --columns: only valid times on this cadence |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--interval-minutes` | profile cadence from first available valid time (default hourly) |
| `--mode` | mean, all (members + P10-P90), or member directory ID; repeat (default mean and all) |
| `--out` | render root (default RUN/render) |
| `--point, --station` | LAT,LON, NAME=LAT,LON or an ID in --stations; repeat for multiple points |
| `--setting KEY=VALUE` | website window setting, e.g. theme=light, parcel=ml, brand=..., init=2026-10-22T00:00Z (the lead's start) |
| `--stations` | JSON station catalogue: {"ID": [lat, lon]} |
| `--threads` | CPU workers (default 4 for pictures, every core for --columns) |

## `woof source-intermediate`

| option | what it does |
|---|---|
| `--accept-licence LICENCE` | explicitly accept a model row's exact restricted/unresolved weight licence; repeat for multiple licences |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--input` | _(the parser declares no help text for this option)_ |
| `--member` | the frame's ensemble member (the GRIB member number the compose decodes) that the WPS export selects |
| `--member-identity` | also seal --member into the input manifest under this identity policy (mapped_authoring); without it the member only selects the composed frame |
| `--out` | _(the parser declares no help text for this option)_ |
| `--profile` | packaged composition profile (overrides the source default) |
| `--source` | source registry name, including earth2-<table key> |
| `--supplement ROLE=FILE` | _(the parser declares no help text for this option)_ |
| `--valid-time` | frame time YYYY-MM-DDTHH:MM:SS |
| `--work-dir` | _(the parser declares no help text for this option)_ |
| `--workers` | _(the parser declares no help text for this option)_ |

## `woof sources`

| argument | what it does |
|---|---|
| `[ID]` | print ONE row in full, named by its registry id or any alias it declares (omit for the listing) |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--json` | emit the registry document instead of the table -- the gpuwm.run-plan.sources.v1 schema, narrowed to the one row when ID is given |

## `woof spectral`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof spectral check`

| argument | what it does |
|---|---|
| `RECEIPT.json` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--rehash-inputs` | _(the parser declares no help text for this option)_ |

## `woof spectral cross-box`

| argument | what it does |
|---|---|
| `RECEIPT.json` | _(the parser declares no help text for this option)_ |
| `OTHER-RECEIPT.json` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--tolerance` | override the declared tolerance (1e-12); the default is measured, so a campaign widening it says why in its record |

## `woof spectral pins`

Takes no options of its own.

## `woof spectral plot`

| argument | what it does |
|---|---|
| `RECEIPT.json` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--output-dir DIR` | _(the parser declares no help text for this option)_ |

## `woof spectral register`

| argument | what it does |
|---|---|
| `SPEC.toml` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--output REGISTRATION.json` | _(the parser declares no help text for this option)_ |

## `woof spectral run`

| argument | what it does |
|---|---|
| `SPEC.toml` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--plot-dir DIR` | _(the parser declares no help text for this option)_ |
| `--receipt RECEIPT.json` | _(the parser declares no help text for this option)_ |
| `--registration REGISTRATION.json` | _(the parser declares no help text for this option)_ |

## `woof spectral score`

| argument | what it does |
|---|---|
| `REGISTRATION.json` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--output RECEIPT.json` | _(the parser declares no help text for this option)_ |
| `--plot-dir DIR` | _(the parser declares no help text for this option)_ |

## `woof spectral-op`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof spectral-op benchmark`

| option | what it does |
|---|---|
| `--backend {numpy,cupy}` | _(the parser declares no help text for this option)_ |
| `--dx-m` | _(the parser declares no help text for this option)_ |
| `--dy-m` | _(the parser declares no help text for this option)_ |
| `--levels` | _(the parser declares no help text for this option)_ |
| `--nx` | _(the parser declares no help text for this option)_ |
| `--ny` | _(the parser declares no help text for this option)_ |
| `--output` | _(the parser declares no help text for this option)_ |
| `--repeats` | _(the parser declares no help text for this option)_ |

## `woof spectral-op calibrate`

| option | what it does |
|---|---|
| `--dt-s` | _(the parser declares no help text for this option)_ |
| `--input` | _(the parser declares no help text for this option)_ |
| `--output` | _(the parser declares no help text for this option)_ |
| `--protect-wavelength-m` | _(the parser declares no help text for this option)_ |

## `woof spectral-op check`

| argument | what it does |
|---|---|
| `receipt` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `woof spectral-op pins`

Takes no options of its own.

## `woof spectral-op response`

| option | what it does |
|---|---|
| `--dt-s` | _(the parser declares no help text for this option)_ |
| `--e-fold-time-s` | _(the parser declares no help text for this option)_ |
| `--maximum-damping-fraction` | _(the parser declares no help text for this option)_ |
| `--maximum-wavelength-m` | _(the parser declares no help text for this option)_ |
| `--minimum-wavelength-m` | _(the parser declares no help text for this option)_ |
| `--order` | _(the parser declares no help text for this option)_ |
| `--output` | _(the parser declares no help text for this option)_ |
| `--protect-wavelength-m` | _(the parser declares no help text for this option)_ |
| `--reference-wavelength-m` | _(the parser declares no help text for this option)_ |
| `--samples` | _(the parser declares no help text for this option)_ |
| `--wavelength-m` | _(the parser declares no help text for this option)_ |

## `woof speedrun`

| argument | what it does |
|---|---|
| `[COURSE]` | the course id to run (`--list` shows them). A course is a row in the shipped course table plus its two asset files; adding one is table work |

| option | what it does |
|---|---|
| `--cold-cache-dir DIR` | EMPTY this directory and point CUPY_CACHE_DIR at it for the run, so a cold-cache record can be set on a machine whose own cache is warm. Only a directory that is absent, empty, or already a CuPy kernel cache is emptied; anything else -- and the inherited cache, the working directory and your home directory by name -- is REFUSED rather than deleted |
| `--compare ('A', 'B')` | compare two records. REFUSED, by name, when they are not records of the same course, the same product set and the same compile mode |
| `--compile-mode {cold,warm}` | which kernel-cache class this record belongs to (default: whatever the course declares). The door MEASURES the cache before the clock starts and refuses a mismatch, because the one-time NVRTC compile is roughly a minute and it is always inside the clock -- a cold record and a warm record are different records and are never compared |
| `--determinism ('ARM_A', 'ARM_B')` | the dual-run byte screen: two capsules from two runs of one course on one machine. These cards carry no ECC, so this is the only thing that may set a determinism claim |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--geog-root DIR` | staged WPS_GEOG tree (default: the one `woof fetch-geog` stages into) |
| `--json` | with --list, emit the table as JSON, including each course's digest and product-set digest |
| `--leaderboard CAPSULE` | emit the SPEEDRUN.md tables for these capsules, one table per comparability class |
| `--list` | list the courses, their product sets and the off-the-clock command that stages each course's bytes |
| `--out DIR` | where the run tree and the capsule go (default speedrun/<course>). The capsule is written as speedrun-capsule.json inside the run's own timestamped folder |
| `--staged DIR` | the directory holding this course's already-staged input bytes. Required to run a course: the clock starts here, so the download must have happened before the door is called |
| `--verify CAPSULE` | verify one capsule's seal and evidence and print its record line, instead of running anything |

## `woof standby`

| option | what it does |
|---|---|
| `--cycle` | _(the parser declares no help text for this option)_ |
| `--data-dir` | _(the parser declares no help text for this option)_ |
| `--devices` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--fetch-workers` | _(the parser declares no help text for this option)_ |
| `--geog-root` | _(the parser declares no help text for this option)_ |
| `--hours` | _(the parser declares no help text for this option)_ |
| `--hydro auto\|off\|SETTINGS` | _(the parser declares no help text for this option)_ |
| `--outdir` | _(the parser declares no help text for this option)_ |
| `--poll-s` | _(the parser declares no help text for this option)_ |
| `--products, --render-products` | _(the parser declares no help text for this option)_ |
| `--publish-root` | _(the parser declares no help text for this option)_ |
| `--recipe` | _(the parser declares no help text for this option)_ |
| `--smoke` | turn on HRRR-Smoke (transported smoke, its radiative feedback, RAVE fires); in 2.9.4 it costs about 11x per step on full CONUS (measured on 8x RTX 5090) |
| `--tiles {auto,on,off}` | _(the parser declares no help text for this option)_ |
| `--wait-timeout-s` | _(the parser declares no help text for this option)_ |

## `woof static`

| argument | what it does |
|---|---|
| `CONFIG` | experiment TOML ([experiment]/[[domain]] tables, as emitted by `woof domain` or `woof import-namelist`; config-driven runs declare their inputs in [case_data]). A legacy [run]-table RunConfig naming a registered case is also accepted. |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--output NPZ` | static-field NPZ output |

## `woof storm-detect`

| option | what it does |
|---|---|
| `--basin` | basin id in the threshold table |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--manifest` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |
| `--reference` | _(the parser declares no help text for this option)_ |
| `--reference-license` | explicit research opt-in for the reference data licence |
| `--render` | _(the parser declares no help text for this option)_ |
| `--rows` | _(the parser declares no help text for this option)_ |
| `--workers` | _(the parser declares no help text for this option)_ |

## `woof storm-diagnostics`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--gravity` | gravity used by the producing model for PH/PHB |
| `--heights` | geometric heights AGL in metres, comma separated |
| `--input-list` | _(the parser declares no help text for this option)_ |
| `--jobs` | _(the parser declares no help text for this option)_ |
| `--label` | _(the parser declares no help text for this option)_ |
| `--out` | _(the parser declares no help text for this option)_ |
| `--render` | _(the parser declares no help text for this option)_ |

## `woof stream`

| argument | what it does |
|---|---|
| `PLAN.toml` | strict gpuwm-stream-plan-v1 orchestration plan |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof suite`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof suite apply`

| argument | what it does |
|---|---|
| `file` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--bindings` | prepared/deployment bindings tied to the resolved suite hash |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--state-dir` | _(the parser declares no help text for this option)_ |

## `woof suite events`

| option | what it does |
|---|---|
| `--after` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--follow` | _(the parser declares no help text for this option)_ |
| `--state-dir` | _(the parser declares no help text for this option)_ |

## `woof suite halt`

| argument | what it does |
|---|---|
| `row` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--state-dir` | _(the parser declares no help text for this option)_ |

## `woof suite init`

| option | what it does |
|---|---|
| `--boxes` | comma-separated woof check --json receipts |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--output` | _(the parser declares no help text for this option)_ |
| `--region` | _(the parser declares no help text for this option)_ |
| `--registry-dir` | _(accepted, but not listed by --help)_ |

## `woof suite plan`

| argument | what it does |
|---|---|
| `file` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--emit-commands` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--json` | _(the parser declares no help text for this option)_ |
| `--what-if` | _(the parser declares no help text for this option)_ |

## `woof suite rehearse`

| argument | what it does |
|---|---|
| `file` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--bindings` | prepared/deployment bindings tied to the resolved suite hash |
| `--cycles` | limit replay to first N cycles |
| `--days` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--from` | _(the parser declares no help text for this option)_ |
| `--state-dir` | _(the parser declares no help text for this option)_ |

## `woof suite resume`

| argument | what it does |
|---|---|
| `row` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--state-dir` | _(the parser declares no help text for this option)_ |

## `woof suite run`

| argument | what it does |
|---|---|
| `[file]` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--bindings` | prepared/deployment bindings tied to the resolved suite hash |
| `--detach` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--once` | _(the parser declares no help text for this option)_ |
| `--poll-seconds` | _(the parser declares no help text for this option)_ |
| `--resume` | _(the parser declares no help text for this option)_ |
| `--state-dir` | _(the parser declares no help text for this option)_ |

## `woof suite status`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--json` | _(the parser declares no help text for this option)_ |
| `--state-dir` | _(the parser declares no help text for this option)_ |

## `woof suite validate`

| argument | what it does |
|---|---|
| `file` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--json` | _(the parser declares no help text for this option)_ |

## `woof tc-strata`

| option | what it does |
|---|---|
| `--author-nests` | Author each selected member through cyclone-setup placement and pricing |
| `--best-track` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--hours` | _(the parser declares no help text for this option)_ |
| `--max-strata` | _(the parser declares no help text for this option)_ |
| `--member-map` | JSON mapping tracking member IDs to explicit source member tokens |
| `--out` | _(the parser declares no help text for this option)_ |
| `--render` | _(the parser declares no help text for this option)_ |
| `--rules` | _(the parser declares no help text for this option)_ |
| `--seed` | _(the parser declares no help text for this option)_ |
| `--select` | _(the parser declares no help text for this option)_ |
| `--start` | _(the parser declares no help text for this option)_ |
| `--storm {auto}` | _(the parser declares no help text for this option)_ |
| `--storm-id` | _(the parser declares no help text for this option)_ |
| `--target` | LAT,LON,UTC initial fix; exact common valid time, no extrapolation |
| `--tracks` | _(the parser declares no help text for this option)_ |
| `--workers` | _(the parser declares no help text for this option)_ |

## `woof tc-tracks`

| option | what it does |
|---|---|
| `--backend {wuduan,vitart,tempestextremes}` | _(the parser declares no help text for this option)_ |
| `--best-track` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--hex-init` | MPAS initialization supplying static fields absent from history |
| `--hex-window` | rw_mpas_convert window, default global 0.25 degree |
| `--input` | _(the parser declares no help text for this option)_ |
| `--manifest` | _(the parser declares no help text for this option)_ |
| `--mesh` | mesh/static file for native MPAS hex --input history |
| `--out` | _(the parser declares no help text for this option)_ |
| `--render` | Draw the plume with rw_wrfbatch |
| `--storm-id` | _(the parser declares no help text for this option)_ |
| `--track-cache` | _(the parser declares no help text for this option)_ |
| `--track-source {auto,nhc,hurdat2,ibtracs,jtwc}` | _(the parser declares no help text for this option)_ |
| `--workers` | _(the parser declares no help text for this option)_ |

## `woof timeline`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof timeline project`

| option | what it does |
|---|---|
| `--deltas` | measured lane delta JSON; repeatable, omitted for baseline |
| `--output` | _(the parser declares no help text for this option)_ |
| `--png` | projected waterfall, for one product |
| `--reference` | gpuwm.timeline.reference.v1 JSON; repeat per product |

## `woof timeline report`

| argument | what it does |
|---|---|
| `ledger [ledger ...]` | explicit timing JSON/JSONL paths; .jsonl.gz accepted |

| option | what it does |
|---|---|
| `--cards` | allocated card count, including idle cards |
| `--output` | JSON path; defaults to timeline.json |
| `--png` | waterfall path; defaults to the JSON path with .png suffix |
| `--product` | select product metadata; no product-specific accounting |
| `--proxy-end` | explicit measured boundary as zoned UTC or Unix seconds |
| `--proxy-start` | explicit measured boundary as zoned UTC or Unix seconds |
| `--rental-id` | select one rental when a source has multiple rentals |
| `--rental-requested` | explicit measured boundary as zoned UTC or Unix seconds |
| `--rental-terminated` | explicit measured boundary as zoned UTC or Unix seconds |
| `--run-id` | select a run while retaining its rental boundaries |

## `woof update`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `woof verify`

| argument | what it does |
|---|---|
| `case` | benchmark or code-verification case to run (no observation scoring) |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--outdir OUT` | directory for the PNG and wrfout NetCDF output (omit to compute metrics only) |

## `woof verify-exact`

| option | what it does |
|---|---|
| `--cases NAMES` | comma-separated case names |
| `--combos IDS` | comma-separated combo ids (default: every combo) |
| `--combos-file JSON` | combo list (default: RECORDINGS/provenance/sweep/combos.json) |
| `--default-arithmetic` | allow a run without GPUWM_WRF_EXACT=1 (not a 0 ULP proof) |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--fixtures DIR` | per-scheme column fixtures (default: RECORDINGS/fixtures when present) |
| `--keep-history` | keep each replay's wrfout files (default: hash and delete) |
| `--list` | print the selection and exit |
| `--no-replay` | score fixtures only |
| `--out DIR` | where WOOF's replays and the results are written |
| `--raw-overlay DIR` | extra root holding reference raw arrays in the recording layout |
| `--recordings DIR` | recording root (runs.csv, inputs/, <build>/<case>/<combo>/) |
| `--referee {scalar,strict}` | the WRF build a PASS/FAIL is decided against (its Thompson-fixed variant where recorded); every recorded build is reported |
| `--repeat {auto,off}` | auto: run WOOF twice for combos whose checks ask for a repeat |
| `--run-summary CSV` | per-run-summary.csv with clean_reference (default: RECORDINGS/...) |
| `--score-only` | no card: rescore WOOF replays already stored under --out |
| `--shard I/N` | run every N-th selected run starting at the I-th (1-based) |
| `--strata LETTERS` | comma-separated strata (A, B, C, D, G, S...) |
| `--write-needs JSON` | list the referee raw arrays that would turn each digest miss into a ULP measurement (input to tools/verify_exact_raw_subset.py) |

## `woof verify-visuals`

| argument | what it does |
|---|---|
| `run_dir` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--append-to` | append station and radar rows to STATIONS.md and MRMS.md |
| `--cycle` | UTC cycle, YYYY-MM-DDTHH[:MM:SS]Z |
| `--domain` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--field-arm LABEL=TEMPLATE` | NPZ path template, for example LABEL=fields/run-f{hour:02d}.npz |
| `--first-hour` | _(the parser declares no help text for this option)_ |
| `--grid` | native lat/lon NPZ for archived field arrays |
| `--last-hour` | _(the parser declares no help text for this option)_ |
| `--list-pending` | print the durable verification state without fetching |
| `--manifest` | JSON list of native per-hour requests, or {hours: [...]} object |
| `--point-arm LABEL=DIR` | _(the parser declares no help text for this option)_ |
| `--reference` | reference row in the native renderer table |
| `--reference-dir` | existing native reference files, named by its metadata table |
| `--refresh` | refetch observations and regenerate receipts |
| `--station-mode {observed,error}` | _(the parser declares no help text for this option)_ |
| `--station-table` | frozen station table used by point extracts |
| `--timeout` | seconds allowed for each public-source command |

## `woof version`

| option | what it does |
|---|---|
| `--check-pypi` | also ask pypi.org for the latest published version and say whether this install is behind, current or ahead of it. Off by default: `woof version` makes no network request unless this flag is given, and a lookup that does not answer prints nothing rather than an error |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--offline` | accepted for older scripts; names the default (no PyPI lookup) and changes nothing |
| `--pypi-timeout SECONDS` | seconds to wait for the index under --check-pypi (default 2.0) |

## `woof warm-kernels`

| option | what it does |
|---|---|
| `--all-profiles` | compile for every shipped physics profile |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--json` | print the report as one JSON document |
| `--levels` | vertical levels of the compile domain (default 40) |
| `--profile PROFILE` | a physics profile to compile for (repeatable); default: every profile a source defaults to |

## `woof-carried-channel`

Takes no options of its own.

## `woof-carried-channel candidate`

| option | what it does |
|---|---|
| `--carried` | _(the parser declares no help text for this option)_ |
| `--classified` | _(the parser declares no help text for this option)_ |
| `--expected-release-id` | independent trusted id, not an id copied from received JSON |
| `--manifest` | _(the parser declares no help text for this option)_ |
| `--out` | new output path; otherwise allocate a new owned generation |
| `--receiver` | _(the parser declares no help text for this option)_ |

## `woof-carried-channel classify`

| option | what it does |
|---|---|
| `--decisions` | _(the parser declares no help text for this option)_ |
| `--expected-review-id` | _(the parser declares no help text for this option)_ |
| `--out` | new output path; otherwise allocate a new owned generation |
| `--receiver` | _(the parser declares no help text for this option)_ |
| `--review` | _(the parser declares no help text for this option)_ |

## `woof-carried-channel emit`

| option | what it does |
|---|---|
| `--new` | _(the parser declares no help text for this option)_ |
| `--new-version` | _(the parser declares no help text for this option)_ |
| `--old` | _(the parser declares no help text for this option)_ |
| `--old-repo` | read the old endpoint from a separate public mirror |
| `--old-version` | _(the parser declares no help text for this option)_ |
| `--out` | new output path; otherwise allocate a new owned generation |
| `--repo` | _(the parser declares no help text for this option)_ |
| `--scope` | _(the parser declares no help text for this option)_ |

## `woof-carried-channel feedback`

| option | what it does |
|---|---|
| `--carried` | _(the parser declares no help text for this option)_ |
| `--classified` | _(the parser declares no help text for this option)_ |
| `--evidence` | _(the parser declares no help text for this option)_ |
| `--expected-release-id` | independent trusted id, not an id copied from received JSON |
| `--include-source` | explicit selected carried path whose whole bytes may be exported |
| `--index` | _(the parser declares no help text for this option)_ |
| `--manifest` | _(the parser declares no help text for this option)_ |
| `--out` | new output path; otherwise allocate a new owned generation |
| `--receiver` | _(the parser declares no help text for this option)_ |

## `woof-carried-channel review`

| option | what it does |
|---|---|
| `--bootstrap` | no inherited classifications |
| `--carried` | _(the parser declares no help text for this option)_ |
| `--expected-release-id` | independent trusted id, not an id copied from received JSON |
| `--manifest` | _(the parser declares no help text for this option)_ |
| `--out` | new output path; otherwise allocate a new owned generation |
| `--previous` | _(the parser declares no help text for this option)_ |
| `--receiver` | _(the parser declares no help text for this option)_ |

## `woof-carried-channel scope`

| option | what it does |
|---|---|
| `--out` | new output path; otherwise allocate a new owned generation |
| `--receiver` | _(the parser declares no help text for this option)_ |

## `woof-carried-channel verify-feedback`

| option | what it does |
|---|---|
| `--carried` | _(the parser declares no help text for this option)_ |
| `--classified` | _(the parser declares no help text for this option)_ |
| `--expected-feedback-id` | _(the parser declares no help text for this option)_ |
| `--expected-release-id` | independent trusted id, not an id copied from received JSON |
| `--feedback` | _(the parser declares no help text for this option)_ |
| `--manifest` | _(the parser declares no help text for this option)_ |
| `--receiver` | _(the parser declares no help text for this option)_ |
| `--require-pairs` | _(the parser declares no help text for this option)_ |

## `woof-carried-channel verify-git`

| option | what it does |
|---|---|
| `--manifest` | _(the parser declares no help text for this option)_ |
| `--old-repo` | _(the parser declares no help text for this option)_ |
| `--repo` | _(the parser declares no help text for this option)_ |

## `woof-mapped-inspect`

| option | what it does |
|---|---|
| `--grib1-bridge` | _(the parser declares no help text for this option)_ |
| `--grib2-dump` | _(the parser declares no help text for this option)_ |
| `--grib2-inventory` | _(the parser declares no help text for this option)_ |
| `--input` | _(the parser declares no help text for this option)_ |
| `--input-manifest` | _(the parser declares no help text for this option)_ |
| `--input-manifest-sha256` | _(the parser declares no help text for this option)_ |
| `--mapping` | _(the parser declares no help text for this option)_ |

## `woof-member-prep`

| option | what it does |
|---|---|
| `--cycle YYYY-MM-DDTHH` | the model cycle whose member is staged, in UTC; it selects the declared upstream-relative paths under --inputs and is recorded in the staging receipt |
| `--describe SET` | print one packaged member set's declared members, statistics and products, then exit |
| `--grib2-inventory` | override the resolved inventory executable |
| `--inputs ROOT` | root holding fetched files at their declared upstream-relative paths |
| `--list-member-sets` | print the packaged member sets and exit |
| `--member ID` | declared member id |
| `--member-set SET` | packaged member-set id (see --list-member-sets) |
| `--members-document JSON` | explicit rw-wps.members.v1 document instead of a packaged set (its SHA-256 is recorded in the receipt) |
| `--output ROOT` | root the member-addressed prepared tree is written under |
| `--products P,P` | comma-separated declared products (default: all declared) |
| `--steps H,H,...` | forecast hours to prepare, e.g. 0,3,6 |
| `--verify-only FILE` | verify one file's bytes against --member and print the evidence; nothing is staged |

## `woof-prepared-forecast`

`--tiles JSON` is the only way to stream this route: its hash-bound experiment cannot carry a `[tiles]` table, so the table rides on the flag.  `--render-products` (with `--render-dir`) is render-on-first-committed-frame, off by absence.  `--materialize-authorities` and `--show-capabilities` each select a DIFFERENT program with its own options and must be the first argument on the line.

| option | what it does |
|---|---|
| `--ack` | registry-owned expert acknowledgement id; repeat as needed. The hash-bound experiment's acknowledgements array delivers the same consent |
| `--devices N` | resident slab count; replaces [devices] count |
| `--devices-table JSON` | _(accepted, but not listed by --help)_ |
| `--domain-bundle` | explicit hierarchy d01 bundle; if omitted it is derived from the hash-bound domain-artifacts manifest |
| `--experiment-config` | _(the parser declares no help text for this option)_ |
| `--frame-markers` | publish OUTDIR/ready/<frame>.json after each history frame is fsynced, self-validated and renamed into place (the default). A marker that exists names a frame that is complete and readable, which is the signal to poll for instead of racing the writer with a size check |
| `--health-debug` | Validate the canonical whole-domain state each step. |
| `--history-interval-seconds` | history cadence; must equal the hash-bound experiment's d01 history_interval_s, and defaults to it when omitted |
| `--io-mode {history}` | _(the parser declares no help text for this option)_ |
| `--materialize-authorities` | create one hash-receipted named-source experiment/WPS authority pair for an exact physics profile, then exit. Run it first on the line and with --help after it for that mode's own options |
| `--no-frame-markers` | do not publish frame-ready markers |
| `--no-memory-gate` | restore a forecast whose priced peak envelope exceeds this card's free memory anyway, as `woof go --no-memory-gate` does: the envelope is an upper bound and the card's own allocation then decides; a model state too big to build at all is still refused |
| `--outdir` | _(the parser declares no help text for this option)_ |
| `--physics-profile` | optional assertion that the hash-bound experiment IS this shipped suite, refused on any switch drift; omitted, the experiment's own physics runs as written and its WRF-verification status is reported, never gating |
| `--prepared-content-sha256` | _(the parser declares no help text for this option)_ |
| `--prepared-head-sha256` | head_sha256 of boundary-stream/head.json: binds a chained preparation at its head, so the forecast starts while the later boundary intervals are still being prepared; the proof and cache digests are checked at the seal |
| `--prepared-root` | _(the parser declares no help text for this option)_ |
| `--progress-every N` | report every Nth model step (default 1, WRF's own cadence). The first and last step of every domain are always reported, and this thins ONLY `step` records -- output, restart and domain events are never thinned |
| `--progress-format {text,jsonl,off}` | how this run reports its progress. `text` (the default) prints one WRF-shaped `Timing for main:` line per model time step per domain on stdout and ALSO writes the machine stream to OUTDIR/progress.jsonl; `jsonl` writes only that stream, leaving stdout free of sentences; `off` disables per-step reporting entirely |
| `--progress-output PATH` | where the machine stream is written; defaults to OUTDIR/progress.jsonl. Append-only JSONL at gpuwm.step-log/v3 (gpuwm.step-log/v4 when the run carries an adaptive time step, which adds `dt` to every step record), one record per printed line, with a dense `sequence` so a consumer can detect a lost line. `-` sends the records to stdout instead of to a file, which with --progress-format jsonl is a pure record pipe |
| `--proof-sha256` | sha256 of the sealed preparation's proof.json; with --prepared-content-sha256, binds a finished preparation |
| `--publish-cmd COMMAND` | optional delivery command; manifest JSON on stdin, no shell (default: off) |
| `--publish-root DIR` | local static web root (default: run output/www) |
| `--radar-heating-table JSON` | [radar_heating] as JSON: radar latent heating windows for the first active_minutes of the run, without editing a prepared configuration |
| `--render-dir DIR` | where --render-products publishes; defaults to OUTDIR/png. Ignored without --render-products |
| `--render-products SPEC` | `woof render --products`' own spec -- a comma-separated product list, or `all`, or `none` -- for every frame this run commits, each drawn on a worker thread as it lands while the forecast is still integrating, one render at a time. Absent is off, and off is the default: there is deliberately no second switch, so "which products" has one answer that cannot disagree with itself. The first frame is the analysis at t = 0, durable before a single step is integrated |
| `--render-section lat,lon,lat,lon\|FILE.json` | `woof render --section`'s own value: the line every xsec: product in --render-products is cut along. Ignored without --render-products |
| `--restart` | Resume a canonical checkpoint with this exact sealed preparation/configuration. |
| `--run-seconds` | forecast length the model integrates; defaults to the hash-bound experiment's run_seconds when omitted. A shorter length runs the first part of the prepared forecast from the same prepared state; a length past the prepared boundary forcing refuses naming the last prepared lead |
| `--show-capabilities` | print this runner's capability JSON and exit; it must be the only argument |
| `--simulated-radar-table JSON` | [simulated_radar] options as JSON; overrides output options without editing a prepared configuration |
| `--source {20crv3,20crv3-cf,aifs,aigefs,aigfs,earth2,earth2-aifs,earth2-aifs-ens,earth2-aifs2,earth2-aifs2-ens,earth2-atlas,earth2-atlas-crps,earth2-atlas-si,earth2-aurora,earth2-aurora15,earth2-aurora15-6h,earth2-aurora15-ensemble,earth2-aurora15-ensemble-6h,earth2-fcn3,earth2-fengwu,earth2-fuxi,earth2-gencast-mini,earth2-graphcast-operational,earth2-graphcast-small,earth2-pangu,earth2-sfno,earth2-stormcast,earth2-stormcast-conus,earth2-ucast,earth2-weathernext2,earth2-weathernext2-mini,ecmwf-ens,ecmwf-open-data,ecmwf-open-data-0p1,era5,era5-l137,gdas,gefs,gefs-reforecast,gefs-reforecast-weekly,gefs-v11,gefs-v11-flat,gem-gdps,gfs,hex-global,hrrr,hrrr-native,hrrr-prs,icon-d2,icon-eu,icon-global,mapped,rap,rap-native,rrfs,rrfs-ens}` | _(the parser declares no help text for this option)_ |
| `--source-manifest-sha256` | sha256 of the preparation's portable source manifest; required, except with --prepared-head-sha256 naming an as-posted head, which binds its input plan instead (its seal writes the manifest, held to that plan) |
| `--stream-init {auto,resident,store}` | which road a STREAMED forecast builds its domain on. `resident` restores the prepared cache into one full-domain DomainState, attaches physics to it and lets the streaming seam copy it into the pinned host store -- the road with the parity proof, and the one that caps the domain at the size of the CARD rather than of the machine (MEASURED at nz = 49: the prepared case costs about 15 780 B/column, so 1024x1024 is refused on a 16 GB card while the streamed forecast it would have fed needs about 6 GiB). `store` fills the same store one ROW SLAB at a time and never allocates a domain-shaped device array, so the ceiling is the machine's pinned RAM. `auto`, the default, prices the resident state from the cache's own state/* manifest times the measured physics headroom and takes the resident road wherever it fits inside 0.80 of the card's free memory. Meaningful only when the run streams: with [tiles] off the resident state IS the domain and this flag changes nothing |
| `--tiles JSON` | the [tiles] table this forecast integrates under, as a JSON object with the keys woof.core.streaming.StreamingOptions takes (mode/tile_nx/tile_ny/nbuffers/halo/store/write_mode/pipeline/vram_budget_bytes/host_budget_bytes). For the caller whose hash-bound experiment cannot carry one: the native HRRR chain hands this runner the authority its preparer BUILT, which has no [tiles] table, so a user's block had nowhere to ride. Validated by the same StreamingOptions.from_mapping the config front door uses, and binds no identity -- omitted, the hash-bound experiment's own table (usually none) runs |
| `--wps-namelist` | _(the parser declares no help text for this option)_ |

## `woof-prepared-forecast --materialize-authorities`

| option | what it does |
|---|---|
| `--base-experiment-config` | _(the parser declares no help text for this option)_ |
| `--base-wps-namelist` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--output-directory` | _(the parser declares no help text for this option)_ |
| `--physics-profile` | shipped suite to materialize into the experiment; omitted, the base config's own physics is published unchanged and its WRF-verification status is reported |
| `--source {20crv3,20crv3-cf,aifs,aigefs,aigfs,earth2,earth2-aifs,earth2-aifs-ens,earth2-aifs2,earth2-aifs2-ens,earth2-atlas,earth2-atlas-crps,earth2-atlas-si,earth2-aurora,earth2-aurora15,earth2-aurora15-6h,earth2-aurora15-ensemble,earth2-aurora15-ensemble-6h,earth2-fcn3,earth2-fengwu,earth2-fuxi,earth2-gencast-mini,earth2-graphcast-operational,earth2-graphcast-small,earth2-pangu,earth2-sfno,earth2-stormcast,earth2-stormcast-conus,earth2-ucast,earth2-weathernext2,earth2-weathernext2-mini,ecmwf-ens,ecmwf-open-data,ecmwf-open-data-0p1,era5,era5-l137,gdas,gefs,gefs-reforecast,gefs-reforecast-weekly,gefs-v11,gefs-v11-flat,gem-gdps,gfs,hex-global,hrrr,hrrr-native,hrrr-prs,icon-d2,icon-eu,icon-global,mapped,rap,rap-native,rrfs,rrfs-ens}` | _(the parser declares no help text for this option)_ |

## `woof-prepared-tree-forecast`

`--sealed-forcing-extension` selects the append-only forcing-prefix checkpoint contract.

| option | what it does |
|---|---|
| `--devices N` | split every grid the tree's [devices] domains names (default every grid) into N resident slabs; replaces [devices] count |
| `--devices-table JSON` | the tree's [devices] table as JSON (count, grid, ids, transport, domains); validated here, without modifying the prepared configuration or its digests |
| `--experiment-config` | _(the parser declares no help text for this option)_ |
| `--experiment-config-sha256` | _(the parser declares no help text for this option)_ |
| `--frame-markers` | publish OUTDIR/ready/<frame>.json after each history frame is fsynced, self-validated and renamed into place (the default). A marker that exists names a frame that is complete and readable, which is the signal to poll for instead of racing the writer with a size check |
| `--health-debug` | _(the parser declares no help text for this option)_ |
| `--io-mode {history,none}` | _(the parser declares no help text for this option)_ |
| `--no-frame-markers` | do not publish frame-ready markers |
| `--no-memory-gate` | restore a tree whose priced peak envelope exceeds this card's free memory anyway, as `woof go --no-memory-gate` does: the envelope is an upper bound and the card's own allocation then decides; a model state too big to build at all is still refused |
| `--outdir` | _(the parser declares no help text for this option)_ |
| `--physics-profile ID` | assert every hash-bound domain uses the named suite; omit to preserve mixed per-domain physics |
| `--preparation-receipt-sha256` | sha256 of the sealed tree's preparation document (proof.json or receipt.json) |
| `--prepared-head-sha256` | head_sha256 of boundary-stream/head.json: binds a chained tree's preparation at its head, so the forecast starts while the root's later boundary intervals are prepared; the seal is bound at the end |
| `--prepared-root` | _(the parser declares no help text for this option)_ |
| `--progress-every N` | report every Nth model step (default 1, WRF's own cadence). The first and last step of every domain are always reported, and this thins ONLY `step` records -- output, restart and domain events are never thinned |
| `--progress-format {text,jsonl,off}` | how this run reports its progress. `text` (the default) prints one WRF-shaped `Timing for main:` line per model time step per domain on stdout and ALSO writes the machine stream to OUTDIR/progress.jsonl; `jsonl` writes only that stream, leaving stdout free of sentences; `off` disables per-step reporting entirely |
| `--progress-output PATH` | where the machine stream is written; defaults to OUTDIR/progress.jsonl. Append-only JSONL at gpuwm.step-log/v3 (gpuwm.step-log/v4 when the run carries an adaptive time step, which adds `dt` to every step record), one record per printed line, with a dense `sequence` so a consumer can detect a lost line. `-` sends the records to stdout instead of to a file, which with --progress-format jsonl is a pure record pipe |
| `--publish-cmd COMMAND` | optional delivery command; manifest JSON on stdin, no shell (default: off) |
| `--publish-root DIR` | local static web root (default: run output/www) |
| `--render-dir DIR` | picture directory (default OUTDIR/png); ignored without --render-products |
| `--render-products SPEC` | plot selectors for every committed frame of every grid, each drawn as it lands, 'all', or 'none'; omitted means no rendering |
| `--render-section lat,lon,lat,lon\|FILE.json` | the line every xsec: product is cut along, `woof render --section`'s own value; ignored without --render-products |
| `--restart` | resume from any member of a gpuwmrst checkpoint set written by an earlier run of this prepared tree. The forecast length (run_seconds), the output/restart cadence (history_interval_s, restart_interval_s) and each domain's history window (history_begin_s, history_end_s) may differ from the run that wrote it -- the same contract `woof run --restart` publishes. Under an adaptive clock the controller's targets and clamps (target_cfl, target_hcfl, the time-step bounds, max_step_increase_pct, the substep floor min_time_step_sound) may differ too: they govern future steps rather than model state, and a resume that retunes them is reported rather than refused, so a dead run can be recovered with the setting that would have saved it. Turning use_adaptive_time_step itself on or off is still refused, as is anything else |
| `--sealed-forcing-extension` | write/restore checkpoints using the explicit append-only forcing-prefix contract |
| `--show-capabilities` | print this runner's capability JSON and exit; it must be the only argument |
| `--simulated-radar-table JSON` | [simulated_radar] options as JSON; overrides output options without editing a prepared configuration |

## `woof-wrf-runtime-check`

| option | what it does |
|---|---|
| `--bridge-dir` | _(the parser declares no help text for this option)_ |
| `--contract` | _(the parser declares no help text for this option)_ |
| `--receipt` | _(the parser declares no help text for this option)_ |
| `--skip-gpu` | _(the parser declares no help text for this option)_ |

## `woof-wrf-init`

The same program as `rw-wps`, under its other installed name; every option above applies.
