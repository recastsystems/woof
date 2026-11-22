# MPAS analysis bundle, version 1

An analysis bundle is an analysed atmosphere that a hex forecast can start from. It is the one
interface between whatever produced the analysis (an ensemble filter, a variational system, a
nudged run) and the forecast. The producer writes it; `woof hex forecast --analysis BUNDLE` reads
it. Nothing in the bundle or in hex knows how the analysis was made.

The code is `woof.hex.analysis_bundle` (format, checks, producer helpers) and
`woof.hex.analysis_door` (the forecast door's `--analysis`).

## Files

A bundle is one directory, or the same files under one object-store prefix
(`s3://bucket/prefix/`):

| File | Required | What it holds |
|---|---|---|
| `manifest.json` | yes | schema `mpas-analysis-bundle/1`, below |
| `analysis.nc` | yes | the analysed MPAS state at the valid time |
| `increment.nc` | no | analysis minus background, in MPAS IAU form |
| `noda.nc` | no | the matched no-assimilation state at the same valid time (for side-by-side forecasts; the forecast never reads it) |
| `static.nc` | no | the mesh's static file, carried with the bundle (SHA-256 `static.sha256`) |
| `grid.nc` | no | the mesh's grid file (SHA-256 `mesh.sha256`); omitted when the static file is also the grid |
| `mesh-row.json` | no | the `gpuwm-hex.mesh-rows/v1` row that binds the carried mesh (timestep, levels, admission) |

No other file may be listed. A file present in the directory but not listed is refused, because
no digest covers it.

### `analysis.nc`

`analysis.form` in the manifest says which of two forms it is.

- **`init`** (the default). An MPAS initial-conditions file of the bundle's mesh, the form
  `woof hex forecast --init` takes. It carries the uncoupled state: `rho` (dry density),
  `theta`, `u` (edge-normal wind), `w`, the water species `qv qc qr qi qs qg`, and any number
  concentrations the physics carries. It also carries the vertical grid (`zgrid`, `zz` and the
  rest of the init stream), the reference state (`rho_base`, `theta_base`), the land and surface
  state, and `xtime` equal to `valid_time`. It must not carry `rho_zz` or `theta_m`, because the
  init loader reads those in preference to `rho` and `theta`. The forecast recouples it exactly
  as it recouples any initial-conditions file (MPAS-A `atm_init_coupled_diagnostics`).
- **`mpasout`**. Only the fields of the RRFS v2 JEDI analysis stream
  (`woof.hex.jedi_contract.ANALYSIS_FIELDS`). Such a file has no reference state of its own, so it
  starts a forecast only as a DA-cycling restart (`--restart-from`), which takes the reference
  state from the run's own init (`woof.hex.jedi_contract.recouple_da_cycling`). An `mpasout`
  bundle carries no `increment.nc`.

### `increment.nc`

The fields of the MPAS-A v8.4.2 `iau` input stream (Registry `tend_iau`), each the analysis minus
the background it was computed against (an MPAS model can read the file as its `iau` stream):

| Variable | Dimensions | Units |
|---|---|---|
| `u_amb` | (Time, nEdges, nVertLevels) | m/s, edge-normal wind |
| `rho_amb` | (Time, nCells, nVertLevels) | kg/m3, dry density |
| `theta_amb` | (Time, nCells, nVertLevels) | K |
| `qv_amb`, and optionally `qc_amb qr_amb qi_amb qs_amb qg_amb` | (Time, nCells, nVertLevels) | kg/kg |

`u_amb`, `rho_amb`, `theta_amb` and `qv_amb` are required (write zeros for one the analysis left
unchanged). Any other prognostic `X` of `analysis.nc` over (nCells or nEdges, nVertLevels or
nVertLevelsP1) may carry `X_amb` too: number concentrations (`nr_amb`, `ni_amb` ...) and `w_amb`.
Each `X_amb` spans `X`'s own dimensions.

Values are float32 or float64. The background is `analysis - increment`, formed in float64 for
every `X_amb` and rounded to `X`'s type. A float64 increment (`a - b` of two float32 fields) gives
the producer's background bit for bit wherever the background value lies within the analysis
value's float64 resolution; where it does not (rain or snow analysed where the background had
almost none) the rebuilt value is within one float64 step of the analysis value of the producer's
(measured on a convective case: at most 2e-19 kg/kg in qr, qs and qg). `write_increment` records
exactness per variable, measured. On that case a forecast from the bundle was bit for bit the
producer's own IAU forecast from its first output (+15 min). Every variable with no `_amb` (the land and surface state) is the
analysis's own. A mixing ratio, number concentration or volume the subtraction leaves below zero
by rounding is set to zero and counted in the run receipt.

`python -m woof.hex.analysis_bundle increment --analysis A --background B --out F` writes this file
from two init-form files: float64 (`--precision f4` for MPAS's own single-precision type), every
differing prognostic (`--fields mpas` for the iau stream's nine only).

### Carried mesh

A bundle may carry its mesh, so that a forecast needs nothing but the bundle: `static.nc`,
`grid.nc` (unless `mesh.sha256 == static.sha256`, when the static file serves as the grid) and
`mesh-row.json`, one `gpuwm-hex.mesh-rows/v1` row named `mesh.id` whose `grid` and `static` are the
bundle's own file names (`grid.nc` or `static.nc`, and `static.nc`), whose digests and sizes are
the carried files', and whose receipt fields are digests (`sha256:...`), not paths. A bundle that
carries the files without the row is bound with `--mesh ROW` (a registered row such as an NCAR
mesh's).

## `manifest.json`

```json
{
  "schema": "mpas-analysis-bundle/1",
  "valid_time": "2022-06-13T23:15:00Z",
  "mesh": {
    "id": "v3.96.167266",
    "sha256": "<SHA-256 of the grid file>",
    "cells": 167266,
    "levels": 55,
    "vertical_spec": {"levels": 55, "zgrid_sha256": "<...>", "z_top_m": 30000.0, "spec_sha256": null}
  },
  "static": {"sha256": "<SHA-256 of the static file>", "builder": "<static builder and version>"},
  "physics": {"suite": "arwen_mpas_column", "mode": "correct"},
  "analysis": {"kind": "mean", "members": 36, "recipe_id": "<opaque id>", "recipe_sha256": "<...>",
               "cycles": 5, "window": 900, "form": "init"},
  "iau": {"window_s": 600},
  "versions": {"woof": "2.9.0", "gpuwm_hex": "0.4.0", "producer": "<producer and version>"},
  "files": {"analysis.nc": {"sha256": "<...>", "bytes": 1703460816},
            "increment.nc": {"sha256": "<...>", "bytes": 512000000}},
  "forecast": {"args": ["--dt", "18.75", "--fullstep-local-timestep", "--fullstep-lts-mode", "active",
                        "--fullstep-lts-max-rate", "8", "--physics-mode", "correct"]},
  "notes": {"producer_run": "<anything the producer wants to record>"}
}
```

| Key | Rule |
|---|---|
| `valid_time` | UTC, `YYYY-MM-DDThh:mm:ssZ` (a reader also takes MPAS's `YYYY-MM-DD_hh:mm:ss`); `analysis.nc` holds a record at this time |
| `mesh.sha256` | SHA-256 of the grid file's bytes |
| `mesh.vertical_spec.zgrid_sha256` | SHA-256 of `zgrid` as little-endian float64 in (nCells, nVertLevelsP1) order, so storage type and axis order do not change it. `spec_sha256` is the canonical hex vertical declaration a native-free init records (`gpuwm_hex_vertical_spec_sha256`), or null |
| `static.sha256` | SHA-256 of the static file's bytes |
| `physics.mode` | `correct` or `exact`, the mode the analysis was cycled with. A forecast in the other mode is allowed, and the door prints the difference |
| `analysis.kind` | `mean`, `control` or `member` |
| `analysis.recipe_id` | an opaque identifier of the producer's recipe (letters, digits, `. _ + : -`) |
| `analysis.window` | assimilation window length in seconds, or null |
| `iau.window_s` | present exactly when `increment.nc` is |
| `files` | every file, by SHA-256 and byte count |
| `notes` | optional: the producer's own provenance (any object, no paths). A reader records it in the run receipt and never acts on it |
| `forecast.args` | optional: the producer's own forecast configuration of this analysis, as `woof hex forecast` flags, only these: `--dt --fullstep-local-timestep --fullstep-lts-mode --fullstep-lts-max-rate --physics-suite --physics --physics-mode --physics-correct-skip --physics-pin-set --physics-profile --physics-default-backend --stochastic --convection --pbl-cadence --preset` (no path, no host binding). The door applies each one the run does not name, so a forecast from the bundle integrates the model the analysis was cycled with; a flag the run names is the run's, and the receipt lists both. Pin sets and profiles must be registered; the backend is `hex-cuda` or `native-cpu`; stochastic JSON uses `StochasticConfig` validation and retains its exact text, including member and seed. Route combinations are resolved after run overrides and may still be refused |

A v1 reader refuses any key it does not know, because a key it would ignore could change how the
bundle must be read. No value may be a path or an address (absolute paths, `~`, drive letters,
backslashes, `scheme://`): the mesh and static are named by SHA-256, and a producer's cache
resolves them.

`python -m woof.hex.analysis_bundle make ...` assembles a bundle and writes a manifest the reader
accepts; `python -m woof.hex.analysis_bundle verify BUNDLE` checks one.

## The forecast door

```
woof hex forecast --analysis <bundle dir | manifest.json | s3://bucket/prefix/> \
    [--mesh ROW --grid GRID --static STATIC] --hours N --history-every-minutes M --out DIR \
    [--iau on|off] [--iau-route native|hex] [--iau-window-seconds S] \
    [--physics-mode correct|exact] \
    [--remap ROW --remap-grid G --remap-static S --remap-template T [--lbc-dir D]] \
    [--restart-from CHECKPOINT --init ORIGINAL_INIT]
```

A bundle that carries its mesh needs no `--mesh`, `--grid` or `--static`: the door writes
`<work>/mesh-rows.json` binding the carried row to the bundle's files and puts it first in
`$WOOF_HEX_MESH_ROWS` for the run. Without a carried mesh, `--grid` and `--static` name the
bundle's mesh files and `--mesh` its row. The door checks every file by SHA-256 and refuses, by
name, a run whose `--grid`, `--static` or vertical grid (the invariant stream, or on a restart the
original init) is not the bundle's.
`--start-time` defaults to `valid_time` (a different one is refused), and `--init-source` to a
sentence naming the bundle's manifest digest, kind, members, recipe and valid time. It then
rewrites the request into the door's existing inputs, so the forecast runs exactly the code a
plain `--init` run does:

| Start | When | What runs |
|---|---|---|
| direct insertion | `--iau off`, the default without `increment.nc` | `--init analysis.nc` |
| IAU | `--iau on`, the default with `increment.nc` | `--init` is the background (analysis minus increment, written into the work directory); the increment enters over `iau.window_s` (or `--iau-window-seconds`) by `--iau-route` |
| DA-cycling restart | `--restart-from` | the checkpoint's physics with the analysis's dycore state; `--init` is the original run's init |
| remap | `--remap ROW` | the analysis (and the background) moved onto the target mesh with the moving nest's operators (`woof.hex.analysis_remap`); the run is then on the target. `--grid`/`--static` name the bundle's mesh, `--remap-grid`/`--remap-static`/`--remap-template` the target's. A limited-area target takes its boundaries from `--lbc-dir`, made from the parent forecast with rw_mpas_lbc |

What the door derives (the fetched bundle, the background, remapped files, the carried row)
goes into a fresh directory beside `--out` (`<out>.analysis-work`, or `--analysis-work`). The
run receipt's `analysis_bundle` block records the manifest digest, every check, the mesh binding
and every derived file.

Every forecast (with a bundle or without) writes `forecast-start.json` into `--out` before its
first step: `{"schema": "gpuwm-hex.forecast-start/1", "grid", "static", "init", "start_time"}`,
the mesh and init it runs on. A reader that draws each history hour as it lands needs nothing else,
even when the door derived the init (a background, a remap).

### Remap onto a nest

`--remap ROW` moves the analysis (and, with IAU, its background) onto a mesh made by hex's own
generator (a regional cull or nest of the analysis's region): the conservative horizontal map
(`woof.hex.moving_remap`), the height-conservative vertical map (`woof.hex.moving_vertical`), the
reconstruction on the target's Jacobian (`woof.hex.moving_state`) and, for a limited-area target,
the open-boundary mass-flux balance (`woof.hex.nest_handoff`). A mesh whose stored areas are not
the generator's sphere (an archived NCAR mesh) is refused by name. A limited-area target's
boundaries come from its parent's forecast trajectory:

```
python -m woof.hex.nest_boundary --source-grid G --source-static S --analysis PARENT_STATE_AT_START \
    --target-grid NEST_GRID --target-static NEST_STATIC --target-init NEST_INIT --operator OP.npz \
    --output LBC_DIR --history PARENT_HISTORY [--history ...]
```

The series' interval is the parent's history interval (15-minute histories force the nest every
15 minutes); `--lbc-dir LBC_DIR` then starts the nest.

## IAU routes

Both routes add their tendencies at the slot MPAS-A's `atm_add_tend_anal_incr` uses: onto the
held physics tendencies, after the column physics is coupled and before the dynamics, on the
fixed-step executor and on every rate class of the full-step local-time-step executor.

| Route | Increment | Each step |
|---|---|---|
| `native` (`woof.hex.iau_native`) | MPAS's uncoupled increments (`increment.nc`) | MPAS-A v8.4.2's own IAU kernels (`woof.hex.iau_v842`) convert them to coupled tendencies against that step's state. `--physics-mode exact` runs them word for word (whole steps up to NINT(window/dt), the physics vapour tendency counted again in the theta_m conversion, no `u*d(rho_edge)` term); `correct` applies IN1, IN2 and IN3 |
| `hex` (`woof.hex.iau_forcing`) | the coupled analysis minus background, formed once from the two recoupled files | added as a constant tendency, weighted by each step's overlap with the window |

The default is `native` in exact mode (exact mode is the MPAS match) and `hex` in correct mode
(the route already measured in cycling). Both stay selectable.

IAU on every door:

- **Across a restart.** A checkpoint carries its IAU window (route, mode, length, start and the
  SHA-256 of the increment it admits). A restart with the same increment continues that window,
  so the increment is admitted once across the restart; a restart with a new increment (a
  DA-cycling restart with `--iau on`: the checkpoint is the background and the bundle's
  `increment.nc` is admitted on the native route) begins a new window at the restart. Exact mode
  on the native route keeps MPAS-A v8.4.2's own rule, a new window at every run start (IN3). The
  hex route forms analysis minus `--init`, which is not a restarted run's background, so a new
  window on a restart takes the MPAS increment file and is refused by name on the hex route.
- **The explicit MPAS suite route** (`--physics-suite`) applies MPAS's own IAU
  (`atm_add_tend_anal_incr`, `woof.hex.iau_v842` on CUDA) once per step after `physics_get_tend`,
  and the density increment enters `tend_rho` as native's `tend_rho_physics` does. The door's
  flags and an MPAS namelist `&iau` group with the `iau` stream run the same code. A native-file
  restart inside a window in correct mode is refused by name (the pristine restart stream holds no
  window position); `config_do_DAcycling = true` begins a new window. One named difference from
  native in exact mode: native's convective dynamics forcing (`rthdynten`) also reads the IAU
  density source; hex's keeps the dynamics-only density tendency, so with Grell-Freitas or New
  Tiedtke the forcing differs by `theta_m*tend_rho_physics/rho_zz` inside the window.
- **Same-host multi-card ranks** admit the increment on each rank's own cells and edges, halos
  included, as one card does; the automatic multi-card selection takes the same path.
- **Member rosters**: a member record may name its bundle (docs/multi-member.md).
- Still one-card only: `--balance-series`, and IAU on multi-card local time stepping (its block
  executor has no held-tendency hook); both are refused by name.
- `--iau-route native` admits MPAS's `iau` stream only (u, rho, theta, qv..qg); another `X_amb`
  (number concentrations, `w`) starts at the background's value, named on stderr and in the
  receipt. `--iau-route hex` admits every prognostic.
