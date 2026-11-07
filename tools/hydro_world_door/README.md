# Default-on hydro door

Project ruling, 2026-10-18 08:40Z: hydro is default on for every WOOF forecast and every
ensemble member whenever a hydrofabric covers the domain; `--hydro off` is the
opt-out; where no fabric exists hydro is off and the receipt names the gap.

## What runs

`woof go` and `woof ensemble` (the same entry, `woof.hydro.runtime.go_scope`)
call `woof.hydro.door.resolve_launch` before preparation. A runner started
without a launcher (`forecast_scope`) makes the same decision once.

| Case | Decision | Receipt `hydro_activation` |
|---|---|---|
| `--hydro off` | off, the table and the domain are never read | `reason: cli_off`, every member off |
| `--hydro FILE` or a `[hydro]` table naming `realization`, `stages` or `driver_factory` | on with those settings (unchanged behaviour) | `door.fabric: explicit settings` |
| `[hydro] enabled = false` | off | `reason: configuration_off` |
| nothing prepared (the default) | the door reads the parent grid and tries the provider rows | `fabric_covered` with the provider, or the named gap |
| bare `--hydro` and no covering fabric | refused before any work, naming every gap | (raises `HydroDoorRefusal`) |

Provider rows live in `woof/data/hydro/fabric-providers.v1.json` (extra rows from
`$WOOF_HYDRO_FABRIC_PROVIDERS`). A row is used only when all of these hold:

1. its coverage (WGS84 boxes, dateline-aware, or `global`) holds at least
   `min_covered_fraction` of a 33x33 lattice over the parent grid;
2. its data licence status is `verified`;
3. it names a coupled realization that `configurations.v1.json` admits for the
   parent LSM (`realization_by_parent_lsm`: Noah `parent-noah-routed`, RUC
   `parent-ruc-routed`, Noah-MP `parent-noahmp-routed`; Noah and RUC runoff is
   their accumulated SFCRUNOFF/UDRUNOFF differenced on the card at each LSM call);
4. the parent LSM is one coupled NextGen consumes (Noah, RUC or Noah-MP,
   `sf_surface_physics=2`, `3` or `4`);
5. its own tool answers the probe: the builder executable exists, has the
   `build` subcommand, `rw_hydro_world admit TABLE ROW` admits its source row, and
   its tile root exists or the path names an `acquire` hook (or, for a `prepared`
   row, its manifest exists). The provider tools are on-demand bundle artifacts, so
   a wheel install stages them on first use.

A path with an `acquire` hook (the global row's `woof.hydro.acquire`) is filled at
preparation: the DEM tiles `rw_hydro_world tiles` lists, the lake and dam archives,
the WorldCover tiles and SoilGrids subsets the parameter recipe needs, into
`$GPUWM_HYDRO_WORLD_TILES` (default `~/.woof/hydro-world`), and the shipped
recipe `woof/data/hydro/parameter-recipe.world.v1.json` is bound to them. On a
default run a failed preparation turns hydro off for that run only, with
`reason: fabric_preparation_failed` and the failure named in the receipt and on
stderr; bare `--hydro` refuses instead.

The first such row wins. The door then runs the builder
(`rw_hydro_world build --sources TABLE --row ROW --bbox W,S,E,N
--data-root DIR --out FABRIC`), runs `rw_hydrofabric prepare`
on the result, caches it under `$WOOF_HYDRO_FABRIC_CACHE` (default
`<case-data root>/hydro-fabric/<row>/<request key>/`, the root `woof setup`
and `fetch-geog` use; never beside the configuration), and writes
`hydro-door.toml` binding the prepared manifest into the realization. A rerun on
the same grid and source root reuses the build. Each builder `paths` entry can
name an `argument` (for example `data_root`): the door resolves its environment
variable or default, checks the directory, and substitutes that value in the
row's argv template. Resolved paths enter the cache request, so changing the
source root rebuilds the fabric. No provider-specific adapter is needed.
The decision travels to every runner and member
process in `WOOF_HYDRO_DOOR_DECISION_JSON`, never inside the settings, so the
sealed realization identity in checkpoints does not depend on how the run was
launched.

One hydro state per member is the coupling lane's resident driver: each member's
forecast prepares its own driver on its own card from its own settings copy, and
the driver's state is part of the atmosphere checkpoint.

## Cost and products in every coupled receipt

- `hydro_cost` (`woof.hydro.cost`): CUDA events recorded around the driver's
  `before_lsm` and `advance` hooks on the hydro stream (the LSM between them is
  excluded), harvested with non-blocking queries; no device sync in the step
  loop. Reports ms per LSM call, ms per atmosphere step, the fraction of the
  atmosphere step, and each stage's parallelism. Above 5 percent it is the named
  defect `hydro-cost-over-budget` (owner hydro-speed): the receipt and the
  forecast outputs are written first, then the run is refused with a non-zero exit.
- `hydro_products` (`woof.hydro.forecast_products`): at the output boundary only,
  the realization's GPU history goes through the Rust writer, its renderer
  request gets the gauges from the fabric's `gauges.json` that sit on the routed
  roster, and the Rust renderer draws hydrographs and streamflow into
  `<forecast out>/hydro-products/d01-<dx>/<product>/<valid-day>/`. Every missing
  piece (no history stage, no renderer build, no matched gauge) is a named gap in
  the receipt and on stderr.

## Today

The global row covers every domain on Earth; on a CONUS domain the CONUS row is
still rejected (licence pending, no realization) and the global row serves it:

| Domain | Decision | Realization |
|---|---|---|
| CONUS, Noah | on, `fabric_covered` by `hydro-world-global` | `parent-noah-routed` |
| CONUS, RUC | on, `fabric_covered` by `hydro-world-global` | `parent-ruc-routed` |
| CONUS, Noah-MP | on, `fabric_covered` by `hydro-world-global` | `parent-noahmp-routed` |

Measured on the 2.9.3 improve-1 run (1800 km Oklahoma domain, 600 x 480 at 3 km,
GFS, Noah): fresh-cache acquisition 91 s for 6.49 GB, then (preparation with the
cards hidden, 48 cores) builder 79 s,
parameters 43 s, projected fabric 153 s, initial water 121 s, reservoirs and
routing about 15 s; 126,593 catchments. The forecast then reused that cache;
at forecast start the continuous-runoff replay routed 2,160 hours (25,920 routing
steps) in 222 s on the card before the first atmosphere step, and the coupled
hydro cost 0.27 percent of the atmosphere step (0.63 ms per step). `receipts/door-decisions.json` holds the
decisions.

The global default row uses the Copernicus-derived network. Per the coordinator's
2026-10-18 11:50Z ruling, HydroSHEDS v1 is research / cross-check only: never
bundled, fetched by default, or used as the default source. Its limited WWF
licence remains a named refusal without an explicit research opt-in. The Rhine
run on that fabric is a pipeline proof, not a shippable fabric or a skill result.

The Rust global builder is merged. Physical source and realization admission
remain separate; a verified source licence alone never enables a forecast.

## Commands

```
# decisions and the coverage chart for any experiment files (CPU)
python -m tools.hydro_world_door.decide EXP.toml [...] --png coverage.png
# tests (CPU; the CUDA file skips without cupy)
python -m pytest -q tests/test_hydro_world_door.py tests/test_hydro_world_door_cuda.py
# on a card (an RTX 5070 Ti measured it) under the hydro pool's slot runner
env MPAS_UNIT=hydro-world-door python3 -B RUN/source/tools/hydro_world_door/node4_run.py RUN
```

Receipts: `receipts/door-decisions.json`, `receipts/door-coverage.png`,
`receipts/cpu-tests.json`, `receipts/node4-cost-meter.json`,
`receipts/case-b-coupled-run-blocker.json`.
