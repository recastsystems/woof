# Hex global: one command

```
woof hex global --row global-60km --cycle 2026-10-11T12 --work-dir RUN
```

`woof hex global` runs a whole global forecast from one row of
`woof/hex/data/global-runs.v1.json` and one cycle. It composes the doors this
package already has and adds none of its own numerics:

| stage | what it does | door |
|---|---|---|
| mesh | fetches the NCAR-published quasi-uniform mesh the row names, once, into the mesh cache; refuses an archive or grid whose SHA-256 is not the pinned one | (fetch) |
| static | builds the native static from that grid, once: `rw_mpas_static --native-init --native-gwd-static` (the MPAS-A v8.4.2 init_atmosphere static stream and its native gravity-wave-drag statistics), and registers the pair as a runtime mesh row so the forecast binds it with its bytes pinned | `rw_mpas_static` |
| init | the row's source (GFS by default; any `init-source --list` row) at the cycle | `init-source` |
| forecast | the forecast door; it picks how many cards the mesh needs from its own capacity rows and spreads it over them (`--devices` names cards instead) | `forecast` |
| products | the standard product set through the Rust renderer | `products` |
| export | the history on a regular lat-lon grid as NetCDF, without drawing | `export` |
| score | the observation scorecard over the whole globe, scored on the CPU; radiosondes from GDAS prepBUFR while NOMADS has the cycle, then from the IGRA2 year-to-date snapshot that covers the run (every case since 2021; only the run's soundings are kept), then the per-station IGRA2 files (`rw_igra2`), so a past case scores too | `score` |

A stage with nothing to do says so in the log and in `events.jsonl`, with the
digest that made the skip safe: a second cycle on the same mesh reuses the
cached grid and static.

## Rows

| row | mesh | cells | physics |
|---|---|---|---|
| `global-120km` | x1.40962 | 40,962 | MPAS (default): convection_permitting |
| `global-60km` | x1.163842 | 163,842 | MPAS: convection_permitting |
| `global-30km` | x1.655362 | 655,362 | MPAS: convection_permitting |
| `global-15km` | x1.2621442 | 2,621,442 | MPAS: convection_permitting |
| `global-*-morrison` | as above | | the woof global suite: Morrison, Grell-Freitas, YSU, revised MM5, Noah, RRTMGP |

The `mpas` arm is the hex default (0.4.6): MPAS-A's own suite for the mesh, from
`src/hexcore/data/profiles/physics-default.v1.json`, at the forecast door's own
step for that suite. `--physics-row arwen` runs WOOF physics (the arm every
row ran before 0.4.6, with its long-step cap).

`--physics-row` swaps the physics arm of any row. A new resolution, mesh or
physics arm is a row in the table, not a code path.

## Step

A row runs at the mesh's own step (MPAS's 6 s per km, rounded down to divide
the 30 min radiation interval: 600 s at 120 km, 360 s at 60 km, 180 s at
30 km, 90 s at 15 km) unless its physics arm caps it. Both shipped arms run
YSU and carry a 300 s cap (`max_dt_seconds` in the table, with the measured
reason in `max_dt_why`). The cap binds only on an engine that lacks the fix
for its defect: `max_dt_retired_by_engine` names the engine algorithm
identities that carry it, and when the installed engine has all of them the
row runs the mesh's own step. On woof 2.9.0, which has neither bound, the
120 km and 60 km rows run at 300 s. The plan records the step, whether the cap
bound, and either the identities that retired it (`cap_retired`) or the ones
the engine lacks (`engine_lacks`), under `step` in `global-plan.json`;
`--forecast-arg=--dt=...` still overrides it.

The forecast door applies the same cap to its own default step whenever a run
selects YSU (`woof.hex.global_door.default_step_cap`), so a bare
`woof hex forecast` on these meshes takes 300 s on woof 2.9.0 and says so on
start; an explicit `--dt` or a namelist `config_dt` runs as given.

Why: at long steps YSU puts single layers at the boundary-layer top that are
5 to 25 K hotter than the layers above and below. On the 120 km globe over
24 h that happened in 3.2 cells per frame at 600 s (worst +24 K), 0.5 at
360 s (worst +23 K) and 0.2 at 300 s (worst +5 K), against 0.1 to 0.2 (worst
+6 to +8 K) in native MPAS-A's own mesoscale_reference runs. With Morrison and
Grell-Freitas one such layer ran away to 381 K in 6.5 h. At 300 s the scores
against sondes and the GFS analysis are the same as at 600 s and the maps
show the same weather.

The cause is two explicit tendencies applied over the whole step that
overshoot their donor layers. YSU's boundary-layer-top heat and vapour
entrainment reaches across several layers at a long step, so the receiving
layer gets more vapour than its donors hold, and microphysics condenses it and
heats that layer; MPAS's own YSU carries the same term. New Tiedtke can ask a
thin layer for more water than it holds: on the 120 km globe at +18 h a layer
over Tibet holding 2.18 g/kg was asked for 4.44 g/kg, and the layer above
became 7.9 K hotter than both neighbours. The engine bounds both: YSU's
entrainment by its donor Courant number (identity
`ysu-wrf-v4.8.0-entrainment-courant-v2`), and New Tiedtke's rates per column so
that no layer loses more water than it holds (identity
`new-tiedtke-wrf461-cumastrn-donor-bound-v2`). Strict reproduction
(`GPUWM_WRF_EXACT=1`) keeps upstream behaviour for both. With both bounds the
`global-120km` row at its own 600 s step showed no hot layer below level 30
over 24 h (13 with neither, 1 with the YSU bound alone), and its maps show the
same weather as before. The WOOF arm's cap retires on such an engine. The
Morrison arm runs Grell-Freitas instead of New Tiedtke and keeps its cap until
its own row has been run at the long step on such an engine.

## Machine surface

Under `--work-dir`:

- `global-plan.json`: the run plan (row, paths, every stage's argument vector,
  the card estimate and, where a measurement exists, the wall estimate).
  `woof hex run-plan ...` prints it and runs nothing.
- `status.json`: rewritten atomically on every transition (schema
  `gpuwm-hex-status-v1`: command, row, stage, stage_index, stage_count, step,
  step_count, started_utc, updated_utc, eta_s, state, reason). `state` is one of
  `running`, `done`, `failed`, `refused`; a refusal is an answer, not a crash.
- `events.jsonl`: one JSON object per event (schema `gpuwm-hex-events-v1`):
  start, stage-start, stage-skip with its reason, step, stage-end, done,
  failed, refused.
- `global.log`: the same events as text.
- `global-receipt.json`: per-stage return codes and seconds.

`woof hex sources --json` lists every init source and every global row for a
workspace to offer.

## Measured

`global-120km`, GFS 2026-10-11 12Z, 24 h, one RTX 5070 Ti (16 GB): mesh 2 s,
static 59 s, init 92 s, forecast 263 s door time (144 steps of 600 s), products
24 s, export 5 s; 7.5 minutes for the whole command. The door's predicted peak
was 7,853 MiB.

At the 300 s step (the YSU arms' cap) the same day's forecast took 133 s of
door time on the same card with a warm kernel cache (288 steps); the 263 s
above included the first run's kernel compilation.

## Export

```
woof hex export RUN/forecast --out RUN/export [--window global|region:...] [--field-set full|surface]
```

writes each history frame as a WRF-shaped NetCDF file (`wrfout_d01_<valid>`)
on the window, filed as `<out>/<view>/<valid-day>/`, through the same
`rw_mpas_convert` the render door runs, and keeps the converter's receipt as
`export-receipt.json`. `--remap product` (default) interpolates the state
linearly and remaps accumulations conservatively.

On a variable-resolution mesh (a zoom globe), `--window global` writes each
region at the detail the model has there, the way WRF writes a parent and its
nests: `wrfout_d01` is the whole sphere at the outer mesh's spacing (0.75
degrees for 96 km cells), `d02` onward are lat-lon boxes over each ring at that
ring's spacing (48, 24, 12 and 6 km rings at 0.4, 0.2, 0.1 and 0.05 degrees),
and the last domain is the refined core at its own spacing (0.025 degrees for
a 3 km core). Every file is a regular lat-lon grid, so each carries the 1-D
coordinates `lat`, `lon`, `Time`, `bottom_top`, `bottom_top_stag` and
`soil_layers_stag` as the single whole-sphere file did, and any of them can be
a WOOF domain's parent; only a core no lat-lon box holds at its spacing (one
reaching a pole) keeps the refined-core maps' Lambert grid. The receipt's
`stack` lists each grid. A 6 h zoom globe with a
550 km, 3 km core wrote 3.1 GB this way in 32 s, where the whole sphere at the
core's spacing wrote 51 GB in 2 min 43 s; a uniform mesh keeps one grid.

`--follow` exports each frame as the forecast closes it; a one-shot export
afterwards takes the frames the follower wrote (`export-ready.jsonl`) instead
of converting them again. The zoom-globe door (`mesh-plan --run`) starts that
follower itself when `WOOF_HEX_EXPORT_LIVE` holds the export's flags as a JSON
list, e.g. `["--out", "RUN/export", "--window", "global", "--field-set", "surface"]`.

## Every cycle: `--cycle latest` and `woof hex global-cycle`

`--cycle latest` picks the init source's newest cycle that is due, from the
source row's `posting` block in `data/init-sources.v1.json` (its cycle hours
and measured posting lag: GFS 211 min, GEFS 222, ICON 158, AIFS 326, IFS ENS
384, IFS HRES 450), and confirms it by asking the source's own product URLs.
A cycle that is due but not posted yet is waited for (`--max-wait-minutes`,
default 120; `--poll-seconds`, default 60), never replaced by an older one.
A newer cycle that posted early is taken. The plan records the choice under
`cycle_resolution`.

    woof hex global-cycle --row global-60km --store STORE            # the latest cycle, once (cron)
    woof hex global-cycle --row global-60km --store STORE --forever  # each cycle as it posts

The driver runs `woof hex global` for each cycle in its own process, into
the parent store:

| Path | What |
|---|---|
| `STORE/<row>/cycles/<YYYYMMDDHH>/` | the cycle's run folder (the door's layout) and `cycle.json`, listing its published frames by valid time and lead hour |
| `STORE/<row>/latest`, `latest.json` | the newest finished cycle; an older cycle run by hand never moves it back |
| `STORE/<row>/feed.json` | the status feed: driver state, the cycle running and its stage, the next cycle and when it is due, every cycle in the store, skipped cycles, what retention removed |
| `STORE/<row>/events.jsonl` | append-only events |

`--keep N` (default 4) finished cycles stay; older cycle folders the driver
made are deleted after each finished cycle. One driver holds a store row at a
time (`driver.lock`). A cycle already finished in the store is not run again
unless `--rerun`; a failed cycle is not retried in a loop, the schedule moves
on to the next cycle.

## History compression on every core

History frames are netCDF-4 with shuffle and zlib level 1. Inside the
background history writer, `woof.hex.history_deflate` hands every chunk to the
Rust library `rust/history-deflate`, which compresses them on worker threads
ahead of HDF5 and gives HDF5 the stream its own deflate would have made, so
each file is the same bytes. Measured on a 16-CPU host (8 workers): a 15 km
sized frame (2,621,442 cells, 11.8 GB raw) took 143 s on one thread and 53 s
this way, identical sha256; the rest of the 53 s is netCDF's own serial work
(the array transpose, HDF5's chunk copy and shuffle) and the 4.9 GB write.
`WOOF_HEX_HISTORY_DEFLATE=off` writes the single-thread way;
`WOOF_HEX_HISTORY_DEFLATE_THREADS` sets the worker count.

### Morrison precipitation validation

The four Morrison rows use the engine's native signed fallout buckets. Their
rain, snow and graupel accumulations and increments must remain finite, but may
be slightly negative or decrease. Hex preserves those native values. Other
schemes and extra buckets retain nonnegative, nondecreasing accumulation checks.
All schemes still validate finite volume fields and the surface snow ratio.
Validation errors and physics receipts name the selected scheme.
