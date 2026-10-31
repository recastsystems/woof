# Sources at the domain wizard

`woof domain` plans a domain FOR a source. The source decides three
things about the file it emits: the boundary cadence written into the
companion `namelist.wps`, whether the domain has to sit inside a regional
grid, and how far into a forecast the window may reach. All three are
declared in that source's row in the source registry
(`woof/source_adapters.py`), so the wizard reads them rather than
carrying its own table -- and a source added to the registry reaches this
door with no change to the wizard.

List the whole registry with `woof prep --list-sources`, or one row with
`woof prep --show-source <id>`.

## What the wizard plans for today

`--source` takes any registered id or any of its aliases.

| source | aliases | boundary cadence | forecast horizon | native grid |
|---|---|---|---|---|
| `hrrr` | -- | 1 h | f048 | 1799x1059 Lambert at 3 km (CONUS) |
| `hrrr-prs` | `hrrr-pressure`, `hrrr-wrfprs` | 1 h | f048 | 1799x1059 Lambert at 3 km (CONUS) |
| `hrrr-native` | `hrrr-mapped-native` | 1 h | f048 | 1799x1059 Lambert at 3 km, native hybrid levels; same-cycle pressure product supplies analysis soil |
| `gem-gdps` | `gem`, `gdps`, `gem-global` | 3 h | f240 | global |
| `icon-global` | `icon`, `icon-13km`, `dwd-icon`, `dwd-icon-global` | 3 h | f180 (00/12Z), f120 (06/18Z) | global, 2,949,120-cell icosahedral mesh at nominally 13 km |
| `icon-eu` | `dwd-icon-eu`, `icon-eu-regular` | 1 h | f120 | lat 29.5..70.5, lon -23.5..62.5 |
| `icon-d2` | `icon-2km`, `dwd-icon-d2` | 1 h | f048 | lat 44.0..57.3, lon 0.0..17.3, 542,040-cell icosahedral mesh at nominally 2.2 km |
| `gfs` | `gfs-0p25`, `gfs-0.25` | 3 h | f384 | global |
| `gdas` | `gdas-0p25`, `gdas-0.25` | 1 h | f009 | global |
| `gefs` | `gefs-ensemble` | 3 h | f384 | global |
| `aigfs` | `ai-gfs` | 6 h | f384 | global |
| `aigefs` | `ai-gefs` | 6 h | f384 | global |
| `ecmwf-open-data` | `ecmwf`, `ifs` | 3 h | f360 | global |
| `ecmwf-ens` | `ifs-ens`, `ecmwf-ensemble` | 3 h | f360 | global, one perturbed member (`p01`..`p50`) |
| `aifs` | `aifs-v2`, `aifs-single` | 6 h | f360 | global |
| `rap` | `rap-awip32` | 1 h | f051 | AWIPS 221, 349x277 Lambert at 32 km (North America) |
| `rap-native` | `rap-awp130bgrb` | 1 h | f051 | AWIPS 130, 451x337 Lambert at 13.545 km, 50 native hybrid levels (CONUS) |
| `rrfs` | `rrfs-ops` | 1 h | f084 | 1799x1059 Lambert at 3 km (CONUS; HRRR's grid, measured identical) |
| `era5` | -- | 6 h | analysis only | global |
| `era5-l137` | `era5-model-level`, `era5-ml` | 1 h | analysis only | global |
| `20crv3` | `20cr`, `twentycrv3`, `20crv3-member` | 3 h | analysis only | global |
| `20crv3-cf` | `20crv3-netcdf`, `20cr-netcdf`, `20cr-cf` | 3 h | analysis only | global |
| `earth2` | `earth2studio` | 6 h | as far as the model was run (f8784) | 0.25-degree global, or the crop the file carries |

`rap-native` preserves RAP's 50 hybrid levels and their explicit pressure
fields. The producer interpolates those levels horizontally onto AWIPS grid
130. Its model lid is 10 hPa; `hrrr-native` has a 15 hPa lid. Both packaged
profiles carry analyzed cloud, ice, rain and aerosol number concentrations.
The older `rap` pressure product remains a separate 32 km source with a
50 hPa uppermost pressure level.

The boundary cadence column is the default. `--cadence N` writes another
spacing when the publisher posts every lead of the window at it and the
source's preparation takes it: `gdas`, `icon-eu`, `icon-d2`, `icon-global`,
`gefs`, `ecmwf-open-data`, `gem-gdps`, `hrrr-prs`, `rap`, `rrfs`, `aigfs`,
`aigefs` and `aifs` take any whole multiple of the spacing their publisher
posts (`icon-global` posts hourly to f078, then every 3 h), so
`woof domain --source gdas --hours 9 --cadence 3` writes
`interval_seconds = 10800` and prepares from f000, f003, f006 and f009.
A cadence the preparation does not take is refused by `woof domain`,
`woof fetch` and the `[fetch]` table check before anything is downloaded,
naming the spacing the preparation takes. A spacing whose leads the cycle
does not publish (`--cadence 1` past f078 on `icon-global`) is refused the
same way, naming the lead and the cycle's ladder.

A registered source that is NOT in this list refuses by name and says why:
either its row has no runnable initialization route yet, or its boundary
cadence is not a property of the source at all (`--source mapped` reads it
from the mapping document a caller supplies).

## Regional sources are bounded at plan time

Where a row declares a native grid, the wizard bounds the fitted ladder by
it and refuses a domain that cannot be reached -- before anything is
downloaded. The refusal names the offending point, where that point lands
in the source's OWN index space, and the window the source covers:

```
$ woof domain --point=38.5,-97.5 --card 16gb --root-dx 3 --hours 6 \
      --source icon-eu --cycle 2026-08-17T00 --out kansas.toml
woof domain: ladder 3 cannot be forced by icon-eu even at the minimum
layout (60x48 root): the 60x48 root's point at lat/lon (37.8377, -98.5408)
maps to source index x=-1200.652 y=133.404, and the source covers
x=0..1376 (lon -23.5..62.5) y=0..656 (lat 29.5..70.5); 2989 of 2989 root
mass points are outside it.  icon-eu's grid is centred at (50.00, 19.50)
-- move --point inside that grid, shrink the ladder, or choose a source
whose coverage includes this domain
```

(Verbatim, `woof 2.5.0`, exit 2, nothing written.)

That is the same answer the preparation stage gives on real bytes, in the
same coordinates, arrived at from the registry instead of from a download.
The 2026-08-17 model battery paid a full ICON-EU acquisition and 73
seconds of preprocessing to learn it, and read it as a traceback.

Where the domain fits but the *margined* fetch box overruns the grid, the
box is clamped into coverage and the clamp is reported as an advisory --
`--area` is a coverage check for a regional source, not a crop.

HRRR keeps its own, stricter check: its certified route needs real source
cells outside the target on every side for the interpolation stencil, the
surface-fallback halo and the donor search, so it refuses domains that the
bare grid rectangle would accept.

## Which sources `woof fetch` downloads

Every row in the table above except `era5-l137` and `20crv3` (see
"Sources with no fetch door" below). Fifteen of them are rows in the
packaged acquisition-route document
(`woof/authorities/rw-wps-fetch-routes.v1.json`), read by the one engine
in `woof/fetch_routes.py`; four keep the hand-written transports that
predate it, and `20crv3-cf` reads the NOAA PSL annual CF archive through
the native CF subset contract (`woof/authorities/native-cf-fetch.v1.json`,
1836 to 2015, 3-hourly). The route document has a sixteenth row,
`rrfs-ens`: `woof fetch` downloads its public member subset, and the
wizard does not plan it, because that column stops at 250 hPa and carries
no land state (its registry row is `explicit_composition_required`).
`docs/public/DATA.md` publishes the working command for each.

| how the bytes arrive | sources |
|---|---|
| table route (whole published objects, in parallel) | `hrrr-prs`, `hrrr-native`, `rap`, `rap-native`, `rrfs`, `rrfs-ens`, `gefs`, `aigfs`, `aigefs`, `ecmwf-open-data`, `ecmwf-ens`, `aifs`, `icon-global`, `icon-eu`, `icon-d2`, `gem-gdps` |
| native CF subset (the publisher's own subsetting service) | `20crv3-cf` (NOAA PSL THREDDS, annual per-variable files) |
| hand-written transport (publisher-side subsetting) | `gfs`, `gdas` (NOMADS grib-filter or the S3 archive), `hrrr` (`.idx` byte ranges, live-cycle wait), `era5` (the public ARCO Zarr store with no key, or a Copernicus CDS retrieval under your own key) |

`woof domain` emits the `[fetch]` table and the runnable step 1 for every
one of them, because the wizard asks the fetch module the question rather
than carrying a list:

```
next:
  1. woof fetch --source rap --cycle 2026-08-17T00 --hours 6 --out .../data/area_38p50n_97p50w
```

**`--area` is only for the four.** A table route takes whole published
objects -- there is no subsetting service in front of them -- so the
emitted `[fetch]` table carries no crop key for one, and passing `--area`
to it refuses by name. The crop for those sources happens at `woof prep`,
where the namelist geometry is the crop. The wizard's coverage advisories
still print: they are a statement about whether the source reaches your
domain, which is a different question from how many bytes come down.

## Sources with no fetch door

Three runnable rows refuse a download by name, and the refusal states the
breakage rather than reporting a gap:

| source | why there is no route | what to do instead |
|---|---|---|
| `era5-l137` | ERA5 model-level data is a queued Copernicus CDS MARS request (dataset `reanalysis-era5-complete`, `levtype=ml`), not files at a predictable URL, and the request runs under your own CDS account | submit the model-level request yourself into a folder DIR, fetch the same hours' surface analysis beside it with `woof fetch --source era5 --cycle CYCLE --hours HOURS --area AREA --retrieve --out DIR`, then `woof prep --source era5-l137 --source-root DIR --experiment-config CONFIG.toml --wps-namelist CONFIG.namelist.wps` |
| `20crv3` | the every-member GRIB2 archive is not published on an anonymously readable public endpoint; only the ensemble-MEAN NetCDF distribution is, and a member state is not a mean | stage the member's files in DIR, then `woof prep --source 20crv3 --source-root DIR --author-only --author-input-manifest DIR/member-manifest.json`, which prints the `--source-manifest` pair the run binds |
| `mapped` | the generic declarative adapter is not a product: it names no publisher, no bucket and no file grammar, so there is nothing to resolve | it *is* the door for bytes you already have -- supply the mapping and composition documents with the files (`--mapping`, `--composition`, `--provenance`, `--input`, `--supplement`) |

`--source-root` binds the folder by the source's own row in
`woof/authorities/rw-wps-fetch-routes.v1.json`: which files are the
ordered inputs (by name pattern and by the format their leading bytes
carry) and which file is each supplement role. It authors the input
manifest as `DIR/inputs.json`, and with no `--output-root` it writes the
prepared tree beside the experiment config as `CONFIG-prepared`, then
prints the `woof sim` line that runs the forecast and draws every
product from each output as it lands. Run again, it prepares into the
next free `CONFIG-prepared-N`, since a folder that exists is never
written over. It keeps `DIR/inputs.json` while the folder's files are
unchanged; after they or this WOOF's decoders change it writes a new
one and says which one it replaced. Each preparation keeps its own copy
of the manifest it was made from. Adding a hand-staged source is a row
there, not a code path.

For `era5-l137` and `20crv3`, `woof domain` emits a
`[fetch]` table carrying the source, cycle, hours and the folder the
bytes go in, and step 1 of the printed
next-steps block is the staging note rather than a `woof fetch` line
that would refuse: for `era5-l137` it spells the model-level request for
this config's hours and area and the `woof fetch --source era5 --retrieve` line
that writes the surface analysis into the same folder. Step 3, `woof go
CONFIG.toml --data-dir DIR`, binds that folder by the same row and runs
the whole chain. Everything else in the file is complete: geometry,
levels, physics, time step, radiation cadence and the boundary interval.

Fourteen further registry rows are registered but **not runnable**
(`hgefs`, `hiresw`, `href`, `hrrr-ak`, `nam`, `nbm`, `refs`, `rrfs-a`,
`rrfs-firewx`, `rrfs-public`, `rtma`, `sref`, `urma`, `wrf`). They are not
planned by `woof domain` and not downloaded by `woof fetch`, and both
doors say which of the four states the row is in -- `adapter_mapping_required`,
`explicit_composition_required`, `member_selection_and_mapping_required`
or `wrf_archive_mapping_required` -- rather than "invalid choice":

```
woof fetch: error: argument --source: --source nam: no fetch route.
  why: the registry row is not runnable (adapter_mapping_required);
  nothing in this WOOF could read the bytes a download produced.
  see: `woof sources` for what each registered source can do today.
```

Adding a download route for one of them is a row in the route document
plus the profile work its status names. It is not a new code path.

## ICON-D2

`--source icon-d2` (aliases `icon-2km`, `dwd-icon-d2`) starts a forecast
from DWD's convection-permitting ICON-D2: nominally 2.2 km, Germany and
its neighbours, a run every 3 hours (00, 03, ... 21 UTC), each to 48
hours at one-hour steps. DWD publishes a run about 45 minutes after its
start and all of it within about 1 h 25 min, and keeps each run for about
a day, so only the last day of starts can be downloaded.

```
woof domain --point 50.1,8.7 --card 16gb --root-dx 1 --hours 3 \
      --source icon-d2 --cycle latest --out frankfurt.toml
woof go frankfurt.toml
```

It reads DWD's native icosahedral objects through the same GDT-101
normalization as `icon-global`, remapped onto your domain plus a
0.25-degree halo at 0.02 degrees. DWD's regular lat/lon ICON-D2 product
is not used: it leaves the part of its bounding box outside the model
domain empty in every field.

The domain must lie inside lat 44.0..57.3, lon 0.0..17.3. DWD leaves the
outer 13 km of the model domain empty, and the model domain is a tilted
shape, so that box is the largest one whose every cell is published.

The default route reads all 65 model levels and their 66 invariant height
interfaces, extending to about 22 km. It reads pressure and water vapor on
those levels and initializes cloud water, ice, rain, snow and graupel from
the published fields. The generated configuration uses a 60 hPa model top;
the 5 km damping layer therefore stays above the deep troposphere.
Mass-level heights are the means of their bounding interfaces. Published
water fractions are converted to the forecast's mass basis by declared
mapping operations, including the condensate contribution to moist mass.

Data: Deutscher Wetterdienst, https://opendata.dwd.de, CC BY 4.0. Anything
you publish from it carries the attribution "Source: Deutscher
Wetterdienst".

## NVIDIA Earth-2 models

`--source earth2` starts a forecast from an NVIDIA Earth-2 model run through
Earth2Studio: FourCastNet 3, SFNO, Pangu, Aurora, FengWu, GraphCast or AIFS.
It reads the NetCDF file Earth2Studio's own `NetCDF4Backend` writes, as
written: one variable per field and level (`t850`, `z500`, `q700`, `u10m`,
`msl`), dimensions `(time, lead_time, lat, lon)`, no units in the file
(Earth2Studio's lexicon fixes them, and the packaged mapping carries that
lexicon). A domain may start at any lead of the file: valid times are the
initialization plus `lead_time`.

These models publish no soil, land mask, orography, skin temperature, SST,
sea ice, snow, surface pressure or 2 m humidity. The land surface comes from
the ERA5 analysis at the model's initialization (the same rule the `aigfs`
row applies with GDAS), and surface pressure is derived at every lead from
the model's own mean-sea-level pressure at the analysis terrain.

One command runs the model and fetches that analysis:

```
WOOF_EARTH2_PYTHON=/path/to/earth2studio-env/bin/python \
woof fetch --source earth2 --model fcn3 --cycle 2015-09-27T00 --hours 96 \
      --area 0,-105,55,-15 --out data/fcn3
woof domain --source earth2 --polygon box.geojson --root-dx 2.5 --hours 48 \
      --cycle 2015-09-27T00 --forecast-start-hour 48 --out storm.toml
woof prep --source earth2 --input data/fcn3/earth2-fcn3-20150927T00.nc \
      --geog-root ~/.local/share/woof/WPS_GEOG \
      --supplement physical_analysis_surface_data=data/fcn3/analysis/era5-combined.nc \
      --author-input-manifest data/fcn3/inputs.json \
      --experiment-config storm.toml --wps-namelist storm.namelist.wps \
      --output-root prepared
woof sim prepared --experiment-config storm.toml \
      --wps-namelist storm.namelist.wps --outdir forecast --render-products all
```

A file you produced with Earth2Studio yourself goes straight to `woof prep`
with the ERA5 analysis at its initialization
(`woof fetch --source era5 --era5-provider arco --retrieve --cycle INIT
--hours 0 --area AREA`, over an area that contains the file's crop).
`WOOF_EARTH2_PYTHON` names an environment with `earth2studio` and the
model's extra installed; nothing is installed and no account is created.
Which models qualify, where their weights live and what the others need is
the table `woof/authorities/earth2-models.v1.json`; adding a model whose
output carries the same vocabulary is a row there.

`woof earth2` combines acquisition, track-based storm placement, preparation,
simulation and native map rendering. Its `--dry-run --spec-out spec.json`
writes the canonical job contract; `woof earth2 --spec spec.json --out DIR`
runs that saved job. Runtime help lists the installed table rows.

Completed fetch directories can be supplied with `--earth2-input DIR`, including
with `--spec spec.json`. This runtime input does not change canonical JobSpec
bytes. Model, cycle, initial data, member, horizon and artifact hashes must match;
longer forecasts and spatial supersets are accepted. A bounded explicit input
defines automatic search reach, and the authored parent and moving corridor
must fit inside it before preparation starts. An incompatible explicit input
refuses instead of launching replacement inference. Omitting it retains normal
model acquisition when no matching cache exists.

Completed fetch-v2 receipts can also be shared through
`WOOF_EARTH2_FETCH_RECEIPTS=/path/to/earth2-run.json` (a path-separated list
also works). Reuse checks the complete model row, cycle, initial data
source, member, area, complete lead ladder and forecast file hashes. A longer
forecast may supply the requested leads; automatic cache discovery does not
replace a global storm search with a cropped forecast. The door reads shared arrays in place and
writes its own preparation handoff. A copied acquisition directory may resolve
retired producer paths relative to the original request output directory;
forecast hashes remain mandatory and the original paths stay in relocation
lineage. Original producer timings stay in the
parent receipt; the current job records verification time and zero model time.
The producer table digest is retained as lineage; an unrelated row added to
the installed table does not invalidate an identical model row. Original
single-member fetch-v1 receipts also require a checked installed default public
weight revision, native seed policy, runtime version and native NetCDF clock
and global extent. Nondefault loader settings without evidence are refused.
The local index defaults to `fetch-receipts` inside the model weight cache;
`WOOF_EARTH2_FETCH_CACHE` overrides that index directory. Retired arrays make
an ordinary cache miss; changed retained bytes cause a named refusal.

## Adding a source

Nothing in `woof/domain_wizard.py` names a model. A new row reaches this
door by declaring, in `woof/source_adapters.py`:

- `runnable=True` and the profile/runner the decode route uses;
- `forcing_interval_seconds` -- the spacing between valid times a door
  writes when none is named. For a row with a packaged profile this is the
  mapping document's `target.boundary_interval_seconds`, or a whole
  multiple of it that the mapping takes and the source's route row names
  as its `default_cadence` (`icon-global`: hourly published, 3 h by
  default), and a test fails otherwise, for every row that has one. It
  reproduces the number the 2026-08-17 battery typed into its hand-written
  namelists by hand. When
  the product can be read at any whole multiple of that spacing, the
  mapping's `target` also declares `"accept_boundary_interval_multiples":
  true`, and a test fails if the source's fetch offers a cadence the
  mapping does not take;
- `max_forecast_hour` -- 0 for an analysis or reanalysis, which also turns
  off `--forecast-start-hour` with that reason named;
- `coverage=` a `RegularLatLonWindow` or a `LambertGridWindow` if the
  product is regional; nothing at all if it is global.

`tests/test_wizard_sources.py` proves the claim rather than asserting it:
it installs a synthetic registry row and drives the real CLI, checking
that the emitted `namelist.wps` carries the cadence the row declared and
that a regional variant refuses outside the window the row declared, with
no code added anywhere.

## Parameterized model sources

A registry row can declare a model table. Its concrete source keys use
`<row>-<model>` and model aliases use the same prefix. `earth2` reads
`woof/authorities/earth2-models.v1.json`; bare model names are not source
aliases. Each row retains its own output cadence and weight licence.
Rows without a preparation profile refuse initialization with their declared
missing inputs. Restricted or unresolved weight terms require an explicit
`--accept-licence` naming those terms, before acquisition or planning.
The shared decoder accepts whole multiples of its minimum cadence.

| source | default output spacing | preparation profile | weight licence class |
|---|---|---|---|
| `earth2-fcn3` | 6 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-sfno` | 6 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-pangu` | 6 h | earth2-pl13-q-netcdf-v1 | research_only |
| `earth2-aurora` | 6 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-fengwu` | 6 h | earth2-pl13-q-netcdf-v1 | unresolved |
| `earth2-aifs` | 6 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-fuxi` | 6 h | earth2-pl13-rh-netcdf-v1 | unresolved |
| `earth2-fcn` | 6 h | unavailable, see model table needs | unresolved |
| `earth2-stormcast` | 1 h | unavailable, see model table needs | unresolved |
| `earth2-corrdiff` | unspecified h | unavailable, see model table needs | unresolved |
| `earth2-dlwp` | 6 h | unavailable, see model table needs | unresolved |
| `earth2-atlas` | 6 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-atlas-si` | 6 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-aifs2` | 6 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-aifs-ens` | 6 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-aifs2-ens` | 6 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-aurora15` | 1 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-aurora15-6h` | 6 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-aurora15-ensemble` | 1 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-aurora15-ensemble-6h` | 6 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-gencast-mini` | 12 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-weathernext2` | 6 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-weathernext2-mini` | 6 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-dlesym` | 6 h | unavailable, see model table needs | permissive |
| `earth2-ace2` | 6 h | unavailable, see model table needs | permissive |
| `earth2-ucast` | 12 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-graphcast-operational` | 6 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-graphcast-small` | 6 h | earth2-pl13-q-netcdf-v1 | permissive |
| `earth2-interpmodafno` | 1 h | unavailable, see model table needs | research_only |
| `earth2-fuxi-s2s` | 24 h | unavailable, see model table needs | research_only |

Hugging Face packages can be fetched independently of a forecast:

```console
woof hf-fetch fcn3
woof hf-fetch hf://owner/model@revision
woof hf-fetch hf://datasets/owner/dataset@revision/path
```

The Earth-2 table's `weights` object supplies `host: "huggingface"`,
`repository`, `revision`, and `files` records containing `path`, `size`, and
`sha256`. A null `sha256` denotes a small Git-stored file; its size and Git
blob ID from the pinned API response are verified. LFS files are checked
against both the table digest and the API digest. Dataset
rows additionally set `repo_type: "dataset"`. With no file list the command
fetches the repository, or the URL's file/directory prefix. Branches and tags
resolve to a commit before transfers start. The cache is
`WOOF_HF_CACHE` (default `~/.cache/woof/hf`), with a JSON receipt beside
each commit directory. Dataset packages live under its `datasets/` prefix.
Repeated fetches verify sizes and checksums before reusing files.

Only `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` from the environment supplies
credentials. Public packages work anonymously; denied packages raise an error
with the repository name and HTTP status. The command reads no token files.
`--connections` or `WOOF_HF_CONNECTIONS` sets the HTTP pool size (default and
maximum 30). Large files need at least two connections and download in
parallel resumable ranges. Fetches share a host admission lock, including across different cache roots.
`WOOF_HF_GOVERNOR` can relocate the lock file for an isolated environment. Interrupted `.part` files resume on the next invocation.

For Earth-2 rows whose `weights.host` is `huggingface`, the forecast door
fetches the pinned package and passes its local directory to the runtime.
The runtime uses `Package(local_directory, cache=False)` and `load_model`,
so it does not ask Earth2Studio to fetch a second package.

The ensemble door accepts registry keys and aliases, including RAP's hourly
cycles. Mixed families use `--source earth2-fcn3:18,gefs:18`, or `[ensemble]`
`sources = [{source = "earth2-fcn3", members = 18}, {source = "gefs", members = 18}]`.
Counts must sum to `--members` when it is supplied. A single uncounted source
takes `--members`, with 36 as its default. Each family fills its own source,
lagged-cycle and perturbation rungs. Member indices and seeds span all families.
The first family supplies the geometry; every other family must cover it.

Explicit storm names and ATCF ids also work with unlabelled model tracks.
The generic cyclone selector looks up the NHC catalogue entry active at the
cycle, then requires exactly one model track within 300 km of its cycle fix.
The catalogue covers the Atlantic and eastern/central North Pacific. Ambiguous
or absent matches refuse with the requested id and cycle. The receipt seals the
catalogue and best-track references; model fixes drive the forecast and moving
nest. `--storm auto` uses only model strength at the cycle and never consults
this identity catalogue.

Local schedulers may set `WOOF_EARTH2_STAGE_PREFIX` to a JSON object of stage names
and argv prefixes. Each prefix wraps the public stage command; unspecified
stages run normally. For example, a pool can admit `setup`, `prep` and `sim`
while tracking and Rust rendering run on the CPU. Receipts separate admission
wait from stage wall time and retain both the scheduler and underlying argv.
This execution setting does not change the canonical JobSpec.

<!-- model-sources:begin -->
## Model source IDs

Generated by `python tools/source_registry_doc.py` from the source registry.

Rows share acquisition and preparation routes. Runtime availability and weight
licence checks still apply; a listed ID does not imply a measured forecast.

| Source | Model | Packaged profile |
|---|---|---|
| `earth2-fcn3` | FCN3 | `earth2-pl13-q-netcdf-v1` |
| `earth2-sfno` | SFNO | `earth2-pl13-q-netcdf-v1` |
| `earth2-pangu` | Pangu6 | `earth2-pl13-q-netcdf-v1` |
| `earth2-aurora` | Aurora | `earth2-pl13-q-netcdf-v1` |
| `earth2-fengwu` | FengWu | `earth2-pl13-q-netcdf-v1` |
| `earth2-aifs` | AIFS | `earth2-pl13-q-netcdf-v1` |
| `earth2-fuxi` | FuXi | `earth2-pl13-rh-netcdf-v1` |
| `earth2-fcn` | FCN | Not declared |
| `earth2-stormcast` | StormCast | `earth2-stormcast-hrrr-netcdf-v1` |
| `earth2-corrdiff` | CorrDiff (km-scale diffusion downscaling) | Not declared |
| `earth2-dlwp` | DLWP | Not declared |
| `earth2-atlas` | Atlas CRPS, selected member | `earth2-pl13-q-netcdf-v1` |
| `earth2-atlas-si` | Atlas stochastic interpolant | `earth2-pl13-q-netcdf-v1` |
| `earth2-aifs2` | AIFS2 | `earth2-pl13-q-netcdf-v1` |
| `earth2-aifs-ens` | AIFSENS | `earth2-pl13-q-netcdf-v1` |
| `earth2-aifs2-ens` | AIFS2ENS | `earth2-pl13-q-netcdf-v1` |
| `earth2-aurora15` | Aurora1p5 | `earth2-pl13-q-netcdf-v1` |
| `earth2-aurora15-6h` | Aurora1p5_6h | `earth2-pl13-q-netcdf-v1` |
| `earth2-aurora15-ensemble` | Aurora1p5Ensemble | `earth2-pl13-q-netcdf-v1` |
| `earth2-aurora15-ensemble-6h` | Aurora1p5Ensemble_6h | `earth2-pl13-q-netcdf-v1` |
| `earth2-gencast-mini` | GenCastMini | `earth2-pl13-q-netcdf-v1` |
| `earth2-weathernext2` | WeatherNext2Cyclones | `earth2-pl13-q-netcdf-v1` |
| `earth2-weathernext2-mini` | WeatherNext2CyclonesMini | `earth2-pl13-q-netcdf-v1` |
| `earth2-dlesym` | DLESyMLatLon | Not declared |
| `earth2-ace2` | ACE2ERA5 | Not declared |
| `earth2-ucast` | UCast | `earth2-pl13-q-netcdf-v1` |
| `earth2-graphcast-operational` | GraphCastOperational | `earth2-pl13-q-netcdf-v1` |
| `earth2-graphcast-small` | GraphCastSmall | `earth2-pl13-q-netcdf-v1` |
| `earth2-interpmodafno` | InterpModAFNO | Not declared |
| `earth2-fuxi-s2s` | FuXiS2S | Not declared |
| `earth2-stormcast-conus` | StormCastCONUS | `earth2-stormcast-conus-hrrr-netcdf-v1` |
| `earth2-atlas-crps` | Atlas CRPS | `earth2-pl13-q-netcdf-v1` |
| `earth2-derived-ws10m` | Derived 10 m wind speed | Not declared |
| `earth2-windgust` | Wind Gust AFNO | Not declared |
<!-- model-sources:end -->
