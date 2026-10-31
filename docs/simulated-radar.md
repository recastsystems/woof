# Simulated radar

Python callers can use [`woof.simulated_radar.simulate`](simulated-radar-python.md)
with saved history or a georeferenced in-memory state.

## Radar bands

The same simulator now supports S, C and X bands. S keeps the existing bulk
operator and output bytes. C and X use finite-particle T-matrix tables for rain,
snow, ice, graupel and hail, with weighted melting. Rust integrates horizontal
and vertical specific attenuation along each beam and applies the two-way loss
to reflectivity and ZDR. Kdp and accumulated differential phase use the same
sampled atmosphere. These products do not feed back into the forecast.

An omitted band uses the site metadata: WSR-88D sites are S-band and TDWR sites
are C-band. An existing request with radar disabled is unchanged. Existing
TDWR requests now select C-band; use `band = "S"` to reproduce the earlier
S-band approximation. `woof simulated-radar --list-sites` lists the executable
catalog and the world reference catalog. Unknown reference bands stay unknown;
reference sites are not automatically executable site IDs. Copy their verified
coordinates and band into a custom site.

```sh
woof simulated-radar forecast/wrfout --outdir radar-c --sites TOKC --band C --formats cfradial1
```

Use `band = "C"` or `band = "X"` in `[simulated_radar]` to override all sites,
or give each custom site its own band:

```toml
[simulated_radar]
enabled = true
sites = [{ id = "SIM1", lat = 35.0, lon = -97.0, height_m = 350.0, band = "X" }]
formats = ["cfradial1"]
scan_strategy = "custom"
elevations_deg = [0.5]
fields = ["reflectivity", "zdr", "kdp", "rhohv", "phidp"]
```

```python
from woof.simulated_radar import RadarSite, simulate

site = RadarSite("SIM1", 35.0, -97.0, 350.0, band="X")
result = simulate("wrfout.nc", site=site, outdir="radar-x",
                  formats=["cfradial1"], fields=["reflectivity", "zdr", "kdp"])
```

The global Python `band=` option and CLI `--band` have the same meaning.
`wavelength_mm=` / `--wavelength-mm` selects the exact supported wavelength:
107.068735 mm for legacy S at 2.8 GHz, 53.5 mm for C, or 33.3 mm for X.
Other wavelengths and contradictory band/wavelength pairs are refused because
silently substituting a table would change the requested science. The C/X
tables represent a band, not the measured frequency of every individual radar.
The generated 111 mm research S table is not substituted for the legacy operator.

C/X require microphysics 8, 9, 18, 50 or 51 and their scheme-specific mass,
number and density inputs. Unsupported physics is a named refusal, with no
S-band fallback. The tables use equal-volume spheroids, a 0 C dielectric,
spherical Gaussian canting and dual Maxwell-Garnett mixing. These are research
forward-operator assumptions, not a calibration to a particular instrument.
CfRadial preserves floating-point moments; Level II is an encoded radar product
with its format's quantization and range limits. Existing manifest and folder
contracts remain in force. See the [whole-run demonstration](simulated-radar-bands-demo.md)
for measurements, sources, limitations and reproducible commands.

A forecast can produce virtual radar scans from its saved atmospheric columns;
WOOF drives the same forecast routes. The same native Rust path serves
forecasts and replay of history files. Python
validates options and queues file paths. Beam tracing, field sampling, radar
file writing and PPI drawing run in Rust.

Enable it in an experiment TOML:

```toml
[simulated_radar]
enabled = true
sites = "auto"
scan_strategy = "vcp212"
formats = ["level2", "cfradial1"]
timing = "history"
fields = "auto"
range_km = 230.0
gate_spacing_m = 250.0
azimuth_step_deg = 1.0
volume_duration_s = 300.0
```

Platform wheels carry the native radar engine. Fallback installs can obtain
it with `woof fetch-bridges --partial --artifact rw_simradar`.

An absent table disables this optional product. A present table enables it
unless `enabled = false`. The table changes output products only; it does not
change forecast state or restart identity. The option works through the
common durable-history stream in the single-domain, tree and prepared-cache
forecast runners. A run with history output disabled cannot produce radar.
Ensemble members and local-cycling members integrate without that stream, so
they refuse an enabled table by name rather than finish without radar.

Before it fetches, prepares or allocates a card, a forecast asking for radar
checks that `rw_simradar` is installed and answers this release's request
contract, that the scan's host memory fits for its largest grid, and prices the
radar output into its disk admission. `woof go`, `woof run`, `woof run-plan`
and the prepared runners behind `woof sim` all ask; `woof check` reports the
same gaps. The disk price is the native writer bound per site-volume (every
format at its widest gate word, uncompressed PPI images and GIF frames) times
sites times committed history frames. With `sites = "auto"` the sites are
chosen from each grid's coverage when its first history lands, so the disk
review names radar as unpriced instead of guessing a count; list the sites to
price them.

`sites = "auto"` selects radar coverage overlapping the domain. An explicit
list selects NEXRAD IDs, for example `sites = ["KTLX", "KVNX"]`. A list can
also contain custom sites:

```toml
sites = [{ id = "SIM1", lat = 35.0, lon = -97.0, height_m = 350.0 }]
```

`height_m` is antenna height above mean sea level, not height above ground.
Site IDs have four letters or digits because Level II carries a four-byte
station identifier. IDs must be unique within a request.
For Level II, custom IDs beginning with `T` are refused unless they are `TJUA`;
Py-ART treats the other `T` IDs as entries in its fixed terminal-radar table.

`scan_strategy = "vcp212"` uses the reference implementation's VCP timing.
`low_tilts` requests the low elevation ladder. A custom ladder uses
`scan_strategy = "custom"` with strictly increasing `elevations_deg`, for
example `[0.5, 0.9, 1.3]`. Giving a ladder without naming a strategy selects
`custom`. The manifest records the actual sweep elevations, including repeated
elevations when the named strategy uses separate scans.

`timing = "history"` scans a history snapshot for each saved time.
`timing = "scan"` supplies neighboring history times to the native scanner
so its rays can use evolving columns. The manifest records requested and used
timing separately. A first frame or an unbracketed scan can use a recorded
snapshot fallback; consumers must use `timing_used` for what happened.
Sub-hourly output improves this mode without changing the forecast stepper.
During a forecast each history waits for the next one on its grid, and the pair
publishes the earlier history's volume once; the last history of each grid
publishes when the forecast finishes, with no later neighbor.

Default formats are `level2` and `cfradial1`. The optional formats are
`cfradial2` and `odim`. `fields = "auto"` requests reflectivity, radial velocity
and supported dual-polarization moments. Explicit field names are
`reflectivity`, `velocity`, `zdr`, `rhohv`, `phidp` and `kdp`.
The manifest lists fields actually written and the dual-polarization status.
Radial velocity is positive away from the radar and negative toward it.
All writers support at most 16384 gates per radial. Increase `gate_spacing_m`
or reduce `range_km` for a larger request. Gate spacing uses whole metres.
Level II additionally supports at most 32 physical cuts per volume.

The lowest two distinct tilts have reflectivity and velocity PPI images,
and CC, ZDR and KDP PPI images when the volume carries those moments; a
volume without dual-polarization moments draws reflectivity and velocity
only. PHIDP is written to the radar files but not drawn. Every image uses
the Rust renderer's radar tables (`rustwx_render::RadarTable`, selected by
name through `scales::radar_scale_named`):

| Image field | Units | Scale |
| --- | --- | --- |
| `reflectivity` | dBZ | `radar_reflectivity`, 10 to 85 dBZ, nothing drawn below 10 dBZ |
| `velocity` | m/s | `radar_velocity`, fixed -60 to +60 m/s, greens toward the radar and reds away |
| `rhohv` | unitless | `correlation_coefficient`, 0.2 to 1.05 |
| `zdr` | dB | `differential_reflectivity`, -4 to 8 dB |
| `kdp` | deg/km | `specific_differential_phase`, -1 to 7 deg/km |

A CC, ZDR or KDP image is drawn only where the same gate's reflectivity is
drawn (10 dBZ and above), so the three cover the echo the reflectivity image
shows and nothing else. The radar files keep every gate.

These are the same tables `rw_wrfbatch` draws its reflectivity products and
observed radar grids with. The table sources are recorded in
`tools/rustwx/vendor/RADAR-COLOR-TABLES.md`.

`color_tables = "classic"` in the `[simulated_radar]` table draws the
reflectivity and velocity images with the scales they wore before 2.8.5
(`reflectivity_classic`, the twelve-step ladder to 70 dBZ, and
`radial_velocity_classic`, the blue-red scale). The default is
`"standard"`. The map products take the same two names from
`woof render --radar-colors classic`, or from `RUSTWX_RADAR_COLORS=classic`
in the environment of any run. The choice changes images only, never the
radar files, and an `rw_simradar` older than the key refuses it by name.

Replay saved history without running the model:

```console
woof simulated-radar RUN_DIRECTORY --config experiment.toml --outdir RUN_DIRECTORY
woof simulated-radar WRFOUT_FILE --sites KTLX --formats level2,cfradial1 --outdir OUTPUT_DIRECTORY
woof simulated-radar --describe
```

A missing or stale `rw_simradar` and a native refusal of an option or input
(an unknown site, a history without atmospheric columns, a scan the host
memory cannot hold) are one refusal at exit 2 that names the next step.

`--describe` returns configuration defaults, strategies, formats, field names,
accepted input contracts and installed capabilities as JSON. It probes the
resolved binary's request ABI, canonical-column ABI and feature declaration.
It does not fetch an artifact, run the model or allocate device state.
`supports.history_replay` and `supports.native_columns_adapter` describe
those installed native interfaces. `supports.resource_estimate` and
`supports.named_input_refusals` require the newer native feature response;
an older binary with the same volume ABI does not claim them.
Configuration editors can also read
`woof.config.declared_key_rows()["simulated_radar"]`.

For an existing prepared cache whose saved authority has no radar table,
`woof sim` and both prepared forecast runners accept
`--simulated-radar-table '{"sites":["KTLX"]}'`. This carries the same validated
output options without rewriting preparation documents or their digests. The
normal run-plan path forwards this table when a preparation step generates its
own authority document. The saved history selection must still keep the
required atmospheric columns.

The history selection must keep the columns the scanner needs. A configuration
that drops required coordinates, heights, thermodynamics or winds fails at
admission with the missing variable names. Unsupported or absent physical
fields are reported by the native scanner; a requested product failure is
raised when its queue drains. A successful forecast cannot silently claim a
radar file that failed to write. A forecast that fails or is interrupted does
not wait for queued scans: they are dropped, the running `rw_simradar` is
terminated, and every committed history left without a volume is named on
stderr with the replay command that scans it.

## Accepted inputs and model routes

The common replay input is a full WRF-shaped column scene. Required variables
are `XLAT`, `XLONG`, `HGT`, `T`, `PH`, `PHB`, `U`, `V` and `W`, with positive
`bottom_top`, `south_north` and `west_east` dimensions. Reflectivity needs
either a three-dimensional `REFL_10CM`, or `P`, `PB`, `T`, `QVAPOR` and
`QRAIN` for the model-field estimator. Snow and graupel fields, scheme identity
and actual number moments determine which additional physical products are
available. Explicitly requested unavailable moments fail by name.

Mass fields use `bottom_top, south_north, west_east`; geopotential and vertical
wind use `bottom_top_stag, south_north, west_east`. Horizontal winds use the
corresponding staggered horizontal axis. `T` requires a nonempty leading
`Time` dimension because the native reader obtains its grid shape from that
four-dimensional field, including when `REFL_10CM` supplies reflectivity.
Other fields may carry an optional leading `Time` dimension; multiple time
records are supported. Native scenes declaring
`RADAR_NATIVE_WINDS=earth-relative-mass-grid/v1` also require the exact
mass-grid `RADAR_U_EARTH` and `RADAR_V_EARTH` fields. Internal `Times` is
authoritative; any WRF filename timestamp fallback is reported in provenance.

Surface maps, composite reflectivity and converted display-only frames do
not supply a vertical atmosphere. They fail with `radar_input_missing_columns`
and the missing dimension and variable names before field loading or input
hashing. A field carrying a collapsed or wrong vertical shape fails with
`radar_input_invalid_columns`. Neither the site nor the worker should invent
vertical velocity or broadcast surface values into missing columns.

Another model can supply `native-atmosphere.columns/v1`, described below,
through the same explicit CLI adapter:

```console
woof simulated-radar SOURCE.nc --input-kind native-columns --config radar.toml --outdir OUTPUT_ROOT
```

For a directory, `native-columns` selects its direct `*.nc` children. Each
source is converted with `woof.rustwx.canonical_radar_scene` before the
ordinary native replay path. Converted scenes are kept below
`radar/native-scenes/`; separate source paths receive separate directories,
so identical input basenames cannot overwrite each other. Passing a raw
native transport as ordinary WRF input fails with `radar_input_needs_adapter`.
The command cannot reconstruct fields absent from its inputs.

The supported integration routes are distinct from this shared replay API:

| Model route | Native integration requirement |
| --- | --- |
| Regional and cyclone | The experiment `[simulated_radar]` table reaches the durable-history runner; prepared runs carry `--simulated-radar-table JSON` |
| HEX | Companion forecast door receives `--simulated-radar radar.toml --radar-window WINDOW`; its Rust converter must use `field_set=full`, the run mesh and initial vertical coordinate |
| Global | Companion experiment carries `[simulated_radar]`; its checkpoint producer writes native temperature and a geometric vertical-wind derivative using an actual model step |

The qualified candidate HEX source is
`edd87c72eae24187bf860f1d763874b0873bd9c8`; the global source is
`9f785b375e4d024233dd3c33233c8f3c4da2518d`. These changes retain the source
package versions `0.3.3` and `0.1.1`, respectively. Those version strings
alone do not identify the radar hooks or prove that a published package has
them. Both routes additionally require matching engine native artifacts.
HEX uses its engine compatibility admission. Global checks the canonical
temperature/wind ABI and the Rust NetCDF transport writer before allocation.
An undated global experiment also needs a UTC origin through `--start-date`;
the native producer records its one-step vertical-wind derivative interval.

`supports.live_routes` supplies each entry point, installed distribution
version, hook-module presence and qualified source commit. A present companion
module leaves `available = null` with `status = requires_companion_admission`:
its presence is not evidence that a forecast driver forwards the table.
A missing module or missing native simulator gives `available = false`.
Consumers must verify the actual companion entry point and its admission
before offering the option. The engine's `go` route does not forward options
into separately launched companion commands on a worker's behalf.

After forwarding the option, a worker consumes the native manifest while
forecasting continues. It must not also replay display frames for the same
run. For completed runs, replay accepts full qualified converted histories or
the documented native column transport. A display-only archive cannot recover
the native global vertical-wind derivative. Requested radar failures propagate
at queue drain; a worker must not mark radar complete after an ignored option
or failed conversion.

## Output contract

The output root is the run directory passed to the API or `--outdir`.
All manifest artifact paths are relative to that root and use `/` separators.

```text
radar/manifest.json
radar/<domain>/<site>/<generation>/<YYYYMMDDTHHMMSSZ>.ar2v
radar/<domain>/<site>/<generation>/<YYYYMMDDTHHMMSSZ>.cfradial1.nc
radar/<domain>/<site>/<generation>/<YYYYMMDDTHHMMSSZ>.cfradial2.nc
radar/<domain>/<site>/<generation>/<YYYYMMDDTHHMMSSZ>.odim.h5
radar/<domain>/<site>/<generation>/ppi/<field>/<YYYYMMDDTHHMMSSZ>_tilt-<index:02>.png
radar/<domain>/<site>/loops/<field>/tilt-NN-<loop_hash>.gif
```

`<domain>` distinguishes model grids, for example `d01`. `<generation>` is an
immutable generation identity, so reconfigured output cannot replace bytes
still named by the current manifest. A file exists only
when its format or image was written. Consumers use manifest entries instead
of guessing which optional outputs exist. The manifest is replaced atomically
after the files it names have been committed. Publication is serialized by a
run lock. Replaying the same domain, site and valid time replaces that volume's
manifest entry, and the replaced generation's files are deleted once the
saved manifest no longer names them. `config_sha256` records the settings used
for each volume.

The manifest schema is `simulated-radar.manifest/v1`. Top-level keys are:

| Key | Meaning |
| --- | --- |
| `schema` | Exact schema identifier |
| `simulated` | Always `true`; these are model-derived volumes |
| `model` | `WOOF` |
| `velocity_convention` | Sign and unit convention |
| `bowecho_source_commit` | Exact reference source revision |
| `bowecho_extraction_commit` | Exact reusable-library extraction revision |
| `writer_source_commit` | Exact recast-radar-tools revision |
| `updated_at` | Manifest publication time |
| `warnings` | Run-level coverage or output messages |
| `volumes` | Ordered volume records |
| `loops` | Rust-rendered GIF loops and their ordered frame lists |

Each volume record contains:

| Key | Meaning |
| --- | --- |
| `domain` | Domain identifier |
| `generation` | Immutable generation identifier |
| `implementation` | Exact implementation revisions, request contract and native source SHA-256 |
| `expected_sites` | Requested or resolved site inventory for this generation |
| `site` | `id`, `latitude_deg`, `longitude_deg`, `antenna_height_msl_m` |
| `valid_time` | Model valid time |
| `scan_start`, `scan_end` | Simulated scan interval |
| `timing_requested`, `timing_used` | Requested and actual time sampling |
| `source_times`, `source_history` | Input valid times and `basename#time-index:sha256=<full-file-hash>` identities |
| `tilts_deg` | Actual sweep elevations in sweep order |
| `fields` | Radar moments written |
| `config_sha256` | Resolved option identity |
| `reflectivity_source` | Reflectivity estimator or model field used |
| `dual_pol_status` | Availability statement, or null |
| `writer_reports` | Per-format writer decisions and any reported limitations |
| `files` | Downloadable radar file records |
| `images` | PPI frame records for constructing loops |
| `elapsed_seconds` | Processing wall time |

Every file record contains `path`, `format`, `sha256` and `bytes`. Every image
record adds `field`, `tilt_index`, `sweep_index` and `elevation_deg` to those keys.
`tilt_index` identifies a unique elevation; `sweep_index` preserves the actual
scan cut index when a strategy repeats an elevation. A loop is
the image records for one domain, site, field and tilt index ordered by volume
`valid_time`; no image-directory scan is needed. Download and image consumers
should verify length and SHA-256 after transfer. Preserve `simulated = true`
and the valid and scan times in any derived display or download listing.

Each loop record contains `domain`, `site_id`, `field`, `tilt_index`,
`elevation_deg`, `valid_times`, `frames`, `path`, `format`, `sha256` and `bytes`.
`format` is `gif`; `frames` identifies the ordered PPI image paths and
`valid_times` supplies their model times. A loop's hash-derived pathname is
immutable, so a new history frame publishes a new loop without changing a
previously listed download's bytes; the loop it replaces is deleted once the
saved manifest no longer names it. A new loop copies the previous loop's
encoded frames and encodes only the new ones, so loop work grows with the
number of new frames, not with the length of the run.

## Native interface

The Python entry point is
`woof.rustwx.simulate_radar(history_paths, *, outdir, config) -> dict`.
It invokes `rw_simradar --request REQUEST.json`. The request contains
`schema = "simulated-radar.request/v1"`, `history_paths`, `outdir` and the
resolved `config` table. The native bridge produces the manifest and returns
its JSON inventory. A native failure raises at the Python boundary.

The binary is part of the Rust bridge asset catalog. `WOOF_RW_SIMRADAR` can
name an explicit build. `woof doctor` checks its presence and contract marker
and gives the command that builds or stages it.

The request may also carry `volume_paths`, the histories that publish volumes
(the others are scan-timing neighbors only), and, for `--estimate` only,
`scene_shapes`, the `[nx, ny, nz]` grids a forecast will write. The `--abi`
marker names both keys, so an older binary is refused at the forecast door.

The simulator follows the BowEcho reference implementation and the radar
writers use recast-radar-tools. Source revision and licence records live with
the vendored crates. Decoder interoperability and scientific checks are
separate from successful file writing; the validation report states which
readers and cases were actually exercised.

## Native column input

`woof.rustwx.canonical_radar_scene(source_path, *, outdir) -> Path` invokes
`rw_simradar --canonical-atmosphere SOURCE.nc --out OUTPUT.nc`. It converts a
durable NetCDF transport with `schema = "native-atmosphere.columns/v1"` into
a WRF-shaped radar scene. It writes a temporary sibling, finishes and syncs
the native writer, then atomically renames it. A failed conversion preserves
any previous output. The Python helper uses `wrfout_d01_<source-stem>.nc`; the file's
`Times` field, rather than its basename, supplies the valid time.

Before converting, `woof.rustwx.canonical_radar_binary()` resolves the binary,
verifies the regular `--abi` request marker and verifies `--canonical-abi`
against
`native-atmosphere.columns/v1 temperature=temperature_k winds=earth-relative-mass-grid/v1`.
This rejects an older artifact that would silently ignore native temperature.

The transport has dimensions `level`, `interface = level + 1`, `latitude`
and `longitude`. Each spatial axis has at least two cells. Levels are ordered
from model top to surface. Latitude increases south to north; longitude
increases over less than 360 degrees without a repeated seam column.
Coordinates may be Gaussian. Conversion retains the native columns without
horizontal interpolation.

| Variables | Dimensions | Units and meaning |
| --- | --- | --- |
| `latitude_deg`, `longitude_deg` | Corresponding one-dimensional axis | Degrees |
| `pressure_pa` | `level, latitude, longitude` | Full pressure Pa |
| `potential_temperature_k` | `level, latitude, longitude` | Potential temperature K; required only when `temperature_k` is absent |
| Optional `temperature_k` | `level, latitude, longitude` | Actual native temperature K; when present it takes precedence over potential temperature |
| `eastward_wind_m_s`, `northward_wind_m_s` | `level, latitude, longitude` | Earth-relative mass-grid wind m/s |
| `qv_kg_kg`, `qc_kg_kg`, `qr_kg_kg`, `qi_kg_kg`, `qs_kg_kg`, `qg_kg_kg` | `level, latitude, longitude` | Native moist-air mass fractions for vapor, cloud, rain, ice, snow and graupel |
| Optional `nc_kg1`, `nr_kg1`, `ni_kg1`, `ns_kg1`, `ng_kg1` | `level, latitude, longitude` | Actual prognostic number per kg moist air; omit unavailable species numbers |
| `height_half_m`, `vertical_velocity_half_m_s` | `interface, latitude, longitude` | Interface height MSL m and upward geometric wind m/s |
| `terrain_height_m` | `latitude, longitude` | Terrain MSL m |

Required text attributes are `schema`, `valid_time`, `simulation_start`,
`source_model`, `source_checkpoint`, `config_sha256`, `microphysics_scheme`,
`vertical_velocity_method` and `derivative_stencil`. Times use
`YYYY-MM-DD_HH:MM:SS`. `source_checkpoint` is a basename. Required numeric
attributes are nonnegative integer `mp_physics`, positive
`derivative_interval_s` and positive `gravity_m_s2`. Optional integer
`morr_rimed_ice` is preserved as `MORR_RIMED_ICE`; it is never inferred.

The canonical scene reverses vertical order, writes pressure as `P` with
zero `PB`. When `temperature_k` is present, it encodes `T` so that the WRF
reader recovers that actual temperature with its own exponent 0.2857142857
and reference pressure 100000 Pa. Without it, `potential_temperature_k`
must already use those constants and `T` is that value minus 300 K. Models
with different thermodynamic constants should supply actual temperature.
Native mass
and number fractions are divided by `1 - qv` to obtain the WRF dry-air
convention. Interface heights are encoded with the reader's exact gravity
9.80665 m/s2, while the producing model's gravity is recorded separately.
Nonfinite fields, invalid humidity, nonmonotonic interface heights, missing
wind, and incompatible dimensions fail before publication.

Full native conversions carry `RADAR_U_EARTH` and `RADAR_V_EARTH` with
`RADAR_NATIVE_WINDS = "earth-relative-mass-grid/v1"`. The simulator uses
these exact mass-grid winds in place of the smoothed WRF staggering round
trip. Missing cells in a bounded mesh window keep their NaN masks. Ordinary
WRF input keeps its native staggered winds. Source column
SHA-256, checkpoint identity, config identity and vertical-wind derivative
provenance are stored in the canonical scene. The radar request and atomic
output manifest remain `simulated-radar.request/v1` and
`simulated-radar.manifest/v1`.

The [resource contract](simulated-radar-resources.md) specifies native geometry
limits, memory admission, the sampling counts used for a worker quote and the
scope of the measured CPU reference. It imposes no site-count cap.
