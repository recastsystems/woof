# WRF analysis bundle, version 1

An analysis bundle is an analysed atmosphere a WOOF-ArWen forecast can start from. It is the one
interface between whatever produced the analysis (an ensemble filter, a variational system, a
nudged run) and the forecast. The producer writes it; `woof run --wrfinput DIR --analysis BUNDLE`
reads it. Nothing in the bundle or in the engine knows how the analysis was made.

The code is `woof.io.analysis_bundle` (format, checks, producer helpers),
`woof.da.analysis_boundary` (the boundary after an analysis) and `woof.da.analysis_door` (the
forecast door). The MPAS bundle of woof hex (`mpas-analysis-bundle/1`) carries the same manifest
keys outside its identity block; this format's identity block is `grid` where that one's is `mesh`.

## Files

A bundle is one directory:

| File | Required | What it holds |
|---|---|---|
| `manifest.json` | yes | schema `wrf-analysis-bundle/1`, below |
| `analysis.nc` | yes | the analysed state at the valid time, WRF-named, on the run's own grid |
| `increment.nc` | no | analysis minus background, on WOOF-ArWen's own staggering |
| `noda.nc` | no | the matched no-assimilation state at the same valid time (for side-by-side forecasts; the forecast never reads it) |

No other file may be listed. A file present in the directory but not listed is refused, because
no digest covers it. Every NetCDF file of a bundle is NetCDF classic (CDF-1, CDF-2 or CDF-5): a
derived file (the background, an analysis made from an increment, a rebased boundary) is a patched
copy of its template with every other byte unchanged, written by the Rust classic writer. A
NetCDF-4 file is refused by name.

### `analysis.nc`

`analysis.form` in the manifest says which of two forms it is.

- **`wrfinput`** (the default). A real.exe-shaped initial-conditions file of the run's grid, the
  form `woof run --wrfinput` starts from: one record, `Times` and `START_DATE` at `valid_time`.
- **`wrfout`**. A history frame at `valid_time` (WOOF-ArWen's or WRF's wrfout). It has no base-state
  inputs a cold start needs, so it starts a forecast only as a cycling restart (`--restart` or
  `--restart-from`), which adds the increment to the restored state; it needs `increment.nc`.

### `increment.nc`

Analysis minus the background it was computed against, in WOOF-ArWen's own prognostics on WOOF-ArWen's own
staggering, under the WRF names and dimensions:

| Variable | Dimensions | Units |
|---|---|---|
| `U` | (Time, bottom_top, south_north, west_east_stag) | m/s, grid-relative |
| `V` | (Time, bottom_top, south_north_stag, west_east) | m/s, grid-relative |
| `W` | (Time, bottom_top_stag, south_north, west_east) | m/s |
| `T` | (Time, bottom_top, south_north, west_east) | K, dry potential temperature perturbation |
| `PH` | (Time, bottom_top_stag, south_north, west_east) | m2/s2, perturbation geopotential |
| `MU` | (Time, south_north, west_east) | Pa, perturbation dry column mass |
| `QVAPOR` and any other `Q*` species, number or volume the run carries | (Time, bottom_top, south_north, west_east) | kg/kg over dry air; numbers per kg |

`U V W T PH MU QVAPOR` are required (zeros where the analysis left a field alone). Values are
float64 (the default) or float32. The background is `analysis - increment`, formed in float64 and
rounded to the analysis variable's type, so a float64 increment rebuilds the producer's background
bit for bit wherever float64 resolves it; `write_increment` measures exactness per variable. Every
variable with no increment (the land and surface state among them) is the analysis's own.

### `PH` is the engine's

`make` (the default `--ph balance`) replaces the analysis's `PH` with the engine's own hydrostatic
integration (`woof.da.hydrostatic.mass_aware_geopotential_change`) of the `T`, loading-moisture
and `MU` changes at each column's own dry mass, so every bundle is balanced the same way whatever
produced it. A column none of them touches keeps its `PH` word for word. `--ph analysis` keeps the
producer's. A wrfout background carries no coordinate coefficients; `--coordinate` names a file of
the same grid that does (the run's wrfinput), checked by fingerprint.

## `manifest.json`

```json
{
  "schema": "wrf-analysis-bundle/1",
  "valid_time": "2022-06-13T23:15:00Z",
  "grid": {
    "id": "<opaque grid id>", "geo_em_sha256": null,
    "nx": 480, "ny": 420, "nz": 50, "dx_m": 3000.0, "dy_m": 3000.0,
    "projection": {"map_proj": 1, "truelat1": 38.5, "truelat2": 38.5, "stand_lon": -85.0,
                   "moad_cen_lat": 41.5, "cen_lat": 41.5, "cen_lon": -85.0},
    "horizontal_sha256": "<...>",
    "vertical": {"znw_sha256": "<...>", "p_top_pa": 1500.0, "hybrid_opt": 2, "etac": 0.2}
  },
  "static": {"sha256": "<...>", "builder": "<static builder and version>"},
  "physics": {"suite": "<suite>", "mode": "correct"},
  "analysis": {"kind": "mean", "members": 36, "recipe_id": "<opaque id>", "recipe_sha256": "<...>",
               "cycles": 5, "window": 900, "form": "wrfinput"},
  "iau": {"window_s": 600},
  "versions": {"woof": "<version>", "producer": "<producer and version>"},
  "files": {"analysis.nc": {"sha256": "<...>", "bytes": 1703460816},
            "increment.nc": {"sha256": "<...>", "bytes": 512000000}},
  "notes": {"producer_run": "<anything the producer wants to record>"}
}
```

| Key | Rule |
|---|---|
| `valid_time` | UTC, `YYYY-MM-DDThh:mm:ssZ` (a reader also takes WRF's `YYYY-MM-DD_hh:mm:ss`); `analysis.nc` holds a record at this time |
| `grid` | the run grid, measured from a WRF-form file of it: mass-point sizes, spacing, the projection as the file's global attributes state it, `horizontal_sha256` (SHA-256 of `XLAT XLONG MAPFAC_M MAPFAC_U MAPFAC_V`, each as little-endian float64 of its first record, so storage type does not change it) and `vertical` (`ZNW` the same way, `P_TOP`, `HYBRID_OPT`, `ETAC`). `geo_em_sha256` is the producer's record of the geogrid file, or null |
| `static.sha256` | SHA-256 of `HGT LANDMASK LU_INDEX`, the same way: the terrain and land under the analysed columns |
| `physics.mode` | `correct` or `exact`, the mode the analysis was cycled with |
| `analysis.kind` | `mean`, `control` or `member` |
| `analysis.recipe_id` | an opaque identifier of the producer's recipe (letters, digits, `. _ + : -`) |
| `iau.window_s` | present exactly when `increment.nc` is |
| `files` | every file, by SHA-256 and byte count |
| `notes` | optional: the producer's own provenance (any object, no paths). A reader records it and never acts on it |
| `forecast.args` | optional: the producer's model configuration as `woof run` flags, only `--rrtmg-variant --thompson-version --urban-ucm-generation --urban-treeoption --urban-ttscheme --urban-fvg`. The door applies each one the run does not name |

A v1 reader refuses any key it does not know. No value may be a path or an address.

```
python -m woof.io.analysis_bundle make --out DIR --background BG.nc (--analysis A.nc | --increment I.nc)
    --grid-id ID --static-builder TEXT --physics-suite TEXT --physics-mode correct|exact
    --kind mean|control|member --members N --recipe-id ID --recipe-sha256 SHA --cycles N
    --producer TEXT [--iau-window-s S] [--form wrfinput|wrfout] [--ph balance|analysis]
    [--coordinate WRFINPUT] [--fields differing|all] [--increment-precision f8|f4] [--noda N.nc]
python -m woof.io.analysis_bundle verify BUNDLE
python -m woof.io.analysis_bundle increment --analysis A --background B --out F
python -m woof.io.analysis_bundle background BUNDLE --out F
```

## The forecast door

```
woof run --wrfinput DIR --analysis BUNDLE [--iau on|off] [--iau-mode oneshot|3d]
          [--iau-fields all|dynamics] [--iau-boundary background|analysis]
          [--iau-window-seconds S] [--analysis-work DIR]
          [--restart CHECKPOINT | --restart-from WRFRST] ...
```

The door checks every file by SHA-256, checks the grid, vertical coordinate and static fields
against the run's (its wrfinput, or its WRF restart), and rewrites the request into the WRF-input
door's own inputs, so the forecast runs the code a plain `--wrfinput` run does:

| Start | When | What runs |
|---|---|---|
| direct insertion | `--iau off`, the default without `increment.nc` | the run's wrfinput is `analysis.nc` |
| IAU | `--iau on`, the default with `increment.nc` | the run's wrfinput is the background (analysis minus increment); the increment is forced over the window inside every RK stage (`woof.da.iau.IauForcing`, `--iau-mode 3d`) |
| warm start | `--restart-from WRFRST` at `valid_time` | the WRF restart's state, the increment added at the restart instant or forced from it |
| cycling restart | `--restart CHECKPOINT` at `valid_time` | the member's own WOOF-ArWen checkpoint, the increment added or forced from the restored instant |

The same command with `--restart` at a checkpoint inside the window resumes an interrupted
analysis leg: the forcing is rebuilt from the bundle and the model clock, and the leg continues
bit for bit (the forcing holds no state the bundle and the clock do not determine; the checkpoint
classifies `_iau_forcing` as infrastructure, `woof.io.restart`).

Around every start:

- **The boundary.** The first `wrfbdy_d01` record is the coupling of what the run pairs it with:
  on a direct insertion the analysis (the start itself); on an IAU cold start, by
  `--iau-boundary`, the background (the default) or the analysis. When that state is not byte for
  byte the wrfinput the boundary was made from, the first record is rebased onto it (WRFDA's
  `da_update_bc`: the record is the state's own coupling, `woof.ingest.wrfinput.
  wrf_initial_boundary_head`, and its tendency is re-derived so the end of the first interval is
  the boundary's own). Every later record and every table the WRF-input door does not read (`W`,
  `HT_SHAD`, `PC`, `qke_adv`) are the original's, and the WRF-input door's own pair check holds
  the rebased record to the state it names. With `--iau-boundary analysis` the forcing leaves the
  whole boundary zone (`spec_bdy_width` rows) to the boundary. Measured on a real 12 km case with
  an increment band that reaches the western and eastern rims (600 s window): the 3000 m vertical
  velocity's departure from no-DA at +15 min was at most 0.050 m/s on the rims with the background
  boundary and 0.095 m/s with the analysis boundary, so the background is the default. Neither
  ramps the rim with the window (named gap G8i), and a restart reads its own boundary at the
  restart instant (named gap G8b).
- **What the RK forcing cannot carry goes in at the window start, balanced.** A moving `MU` (a
  surface-pressure increment) has no dry-mass slot with the mass term every coupled tendency would
  need (named gap G9b), so it goes in at the window start with its own geopotential share
  (integrated on the background's columns); the rest of the geopotential is forced. Under
  `--iau-fields dynamics` the hydrometeor masses and numbers go in at the window start too.
- **Records.** `forecast-start.json` in the output directory names the start before the first
  step. The run receipt's `analysis_bundle` block records the bundle, every check, the plan,
  what went in at once, the IAU forcing's receipt (fraction applied, steps forced, forced
  seconds), the distance of a restored state from the bundle's background
  (`background_match`), and the balance series (`woof.da.maspt`: the interior mean
  surface-pressure tendency of the first hour, per step, with the insertion's own jump).

What the door derives (the run directory with the analysis or background, the rebased boundary,
the increment split, `door-plan.json`) goes into `<outdir>.analysis-work/<manifest digest>`, or
`--analysis-work`.

### Refusals

Each names the breakage it prevents:

- a door flag with no `--analysis`, or on the CONFIG or `--met-em` route (no analysis hook: the
  flag would be read by nothing);
- `--iau on` without `increment.nc`; `--iau on --iau-mode oneshot`; `--iau off --iau-mode 3d`;
  an IAU window flag on a direct insertion;
- `--iau-mode split` (named gap G12: the tree executor has no after-step hook) and `--iau-mode 4d`
  (named gap G11: a v1 bundle carries one increment, a 4D window needs slot records and a start
  half a window before the analysis);
- an IAU window that ends after the run (G10), refused before any step, and a run that reached
  the window's end with part of the increment unapplied, refused when the receipt is written;
- a run with nests (named gap G7);
- a `wrfout`-form analysis on a cold start;
- a restart before the analysis time;
- a cold start whose analysis time falls outside the boundary's first interval (a start inside
  that interval is rebased onto its own instant, `woof.da.analysis_boundary`);
- an increment of a field the run's physics does not carry;
- a stale or changed derived file, or a resume that asks for another IAU configuration than the
  leg it continues.

The default IAU mode and field set (3d, every field) stand until the IAU measurement on a cycled
convective case sets them; every mode stays selectable.
