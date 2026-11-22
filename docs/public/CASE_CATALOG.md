# Case catalogs

WOOF-ArWen includes a default catalog of **300 historical cases** and also reads
custom catalogs from ZIP, JSON or TOML. A catalog can describe
tornadoes, derechos, hurricanes and other studies without embedding commands or
inventing another forecast engine. The same catalog serves the interface and
the `woof case-catalog` commands.

Open **Case catalogs** to browse the included cases. CLI commands use the same
catalog when `--catalog` is omitted:

```sh
woof case-catalog list --query "El Reno"
woof case-catalog default --json
```

The bundled `historical.zip` (in the `recast-woof-data` companion, which every
`pip install recast-woof` installs, at `woof_data/data/case-catalog/`) combines the
supplied 200-case catalog and its worldwide expansion. All 200 earlier case IDs are
retained, with the newer complete records taking precedence, and 100 cases
are added. The result contains 120 tornado, 60 convective-wind, 70
tropical-cyclone and 50 synoptic cases. Source references, planning limitations,
initializations and scientific settings remain as supplied; merging does not
validate a forecast preset. The ZIP includes `merge-provenance.json` with both
original archive hashes, source titles and versions, and every retained ID.
The original archives are not modified. For publication, the catalog's
reader-facing text uses WOOF's name and plain punctuation and its engine
links point at this repository; `merge-provenance.json` records each rule
and the catalog digest before those edits. Pass `--catalog` to use another
catalog.

The supplied `woof/data/case-catalog/example.json` and `example.toml` contain
**synthetic format demonstrations, not historical cases or scientific
recommendations**. `schema.json` describes the format. Its native-setting list
comes from the current ArWen physics/configuration contract.

## Authoring a catalog

Use `schema = "arwen.case-catalog.v1"`, a `catalog` object with its ID, title,
version and provenance, and a `cases` array. Preserve real references, the
author's rationale, uncertainty and any initialization limitations. Do not
invent dates, archive coverage, resolution requirements or physics evidence.

Each case has a safe, stable ID; a title and event kind; source/initialization
options; and `lower`, `recommended` and `upper` tiers. Case IDs use lowercase
letters, digits, hyphens or underscores, at most 128 characters. IDs are not
paths. Duplicate IDs are refused.

Source options carry an option ID, a native source ID or alias, and
`cycle_utc`. An optional nonnegative `forecast_start_hour` selects a forecast
lead from that cycle. An explicit `recommended_source_option` identifies the
catalog author's preferred option. Multiple options without a recommendation
require the user or script to select one. Unknown future sources remain
browsable but cannot create a configuration until WOOF-ArWen supports them.

UTC timestamps may carry `Z` or an offset, which is converted to UTC. A source
initialization must resolve to an exact UTC hour. Event start/end metadata may
include minutes. A timestamp without an offset in a field named `_utc` means
UTC. Source selection uses WOOF-ArWen's source calendar and publication constraints;
catalog metadata never certifies that a server currently has the files.

Each tier carries:

- `bounds_degrees`: south, west, north and east in degrees. West greater than
  east explicitly crosses the antimeridian.
- `root_dx_km`: root grid spacing in kilometres.
- `nest_ratios`: integer refinement ratios; `[]` means a single domain.
- `run_hours`: a positive whole-hour run duration.
- Optional `nz` and `history_interval_s` in their native units.

The tiers describe increasing resource demands: lower coverage is contained
by recommended coverage, which is contained by upper coverage; grid spacing
progresses from coarser to finer; run durations do not decrease. Bounds are
required study coverage, not a promise of exact grid dimensions. The native
domain builder fits and reports the actual grids and refuses a footprint that
cannot fit the selected hardware.

## Scientific settings

Advisory `recommendations` and `metadata` remain data. Unknown advisory topics
are retained. They cannot run commands or silently change settings.

A case or tier may explicitly select `physics_profile`, using an actual native
profile ID. For advanced work it may also specify `native_overrides`:

```json
{
  "shared": {"num_soil_layers": 4, "opt_thcnd": 2},
  "domains": [
    {"grid_id": 1, "settings": {"diff_6th_factor": 0.1, "epssm": 0.4}}
  ]
}
```

These numbers demonstrate syntax only. Native override keys are derived from
WOOF-ArWen's registered scientific parameters/selectors and configuration scopes.
Unknown keys, unsupported values, invalid combinations, paths, commands,
environment settings and credentials are refused. No acknowledgement is taken
from catalog data. Required scientific declarations must be explicitly supplied
with `--ack`, exactly as on the existing domain command.

Overrides apply in this order: case, selected tier, explicit caller override.
A shared override applies to all domains; a specific domain override then wins.
The final experiment is parsed and admitted through WOOF-ArWen's existing physics
and memory validation after the changes. Creation is not a forecast result or
scientific qualification. The original catalog and all recommendation
provenance are retained beside the new configuration.

## Commands

```sh
woof case-catalog list --catalog cases.json --json
woof case-catalog search --catalog cases.toml --query "derecho" --json
woof case-catalog show CASE_ID --catalog cases.json --json
woof case-catalog preview CASE_ID --catalog cases.json --tier recommended --source-option OPTION_ID --json
woof case-catalog native-settings --json
woof case-catalog create CASE_ID --catalog cases.json --tier lower --source-option OPTION_ID --out study.toml --vram-gib 16 --json
woof case-catalog create CASE_ID --catalog cases.json --out advanced.toml --vram-gib 24 --native-overrides overrides.json --json
woof case-catalog export --catalog cases.toml --out normalized-cases.json
woof case-catalog export --catalog cases.toml --out original-cases.toml --original
```

Preview shows the chosen source, cycle, tier, explicit settings and source
availability guidance. Native geometry and memory admission run during create.
Creation writes a new TOML, every file that configuration's input route
reads beside it (its `namelist.wps`, and on the native regional route the
two namelists and the target-domain document the run reads), the exact
original catalog and a `.arwen-case.json` receipt. Existing files are
preserved. A selection those route files cannot state, which is a physics
suite outside the shipped profiles on that route, is refused at creation
naming the setting, and nothing is written. The receipt
records the original SHA-256, selected case, applied settings, actual domain
sizes and native admission. Continue through the existing `woof go` workflow.

Exports are create-only. Normalized exports are JSON; `--original` preserves
the exact original JSON or TOML bytes, including comments and formatting.

## Supplied research-proposal ZIP

The loader also reads the `arwen.case-catalog/v1` research-proposal format directly from JSON or a ZIP containing one `catalog.json`. Archive paths, member count and expanded size are checked; no member is extracted and no bundled script is run. The original ZIP and its SHA-256 are retained when a configuration is created.

Its minimum/preferred/large labels map to lower/recommended/upper. Source-specific schedules supply the actual cycle, duration and boundary cadence. Centered per-domain width/height intentions are rounded upward only for native cell/nest alignment; actual dimensions and memory admission are recorded. The original source records, sensitivity proposals and scientific limitations remain in the receipt. Unsupported executable settings are listed explicitly and block creation. Known `diff_opt=2` notation is recorded as the native full-diffusion form, using the existing namelist-import contract. Importing these proposals does not validate historical claims, source files or forecast skill.
