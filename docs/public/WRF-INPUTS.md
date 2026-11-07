# Use existing WPS or WRF inputs

Native WOOF preparation remains the recommended path for new forecasts. Existing
WPS and `real.exe` scripts can also hand their files to the same WOOF forecast
runtime.

After `real.exe`, keep `wrfinput_d0*`, `wrfbdy_d01`, and the producing
`namelist.input` together:

```console
woof run --wrfinput wrf-run --outdir forecast
```

To continue a WRF run from a restart file, keep `wrfrst_d0*_<time>` (any
suffix), `wrfbdy_d01` and a `namelist.input` together, and name the file:

```console
woof run --restart-from wrf-run/wrfrst_d01_2019-11-26_23:00:00 --outdir forecast
```

A run directory whose `namelist.input` says `restart = .true.` restarts
from wrf.exe's own file name with plain `woof run --wrfinput wrf-run` (or
`woof run wrf-run`). The state, soil and snow, held radiation, PBL and
cumulus persistence, accumulators and clock come from the file; moist theta
is converted to WOOF's dry theta. The run ends where the namelist ends;
`--run-seconds` sets the window after the restart instant. State the file
holds for a scheme WOOF cannot continue, or a scheme that differs from the
one that wrote it, is refused by name. NCAR's v4.4 benchmark folders
(https://www2.mmm.ucar.edu/wrf/users/benchmark/v44/, for example
`v4.4_bench_conus12km.tar.gz`, 517 MB, 3.2 GB unpacked) run as shipped.

To time a run the way WRF's `rsl.out` reports it, add `--benchmark`:

```console
woof run --benchmark --restart-from v4.4_bench_conus12km/wrfrst_d01_2019-11-26_23:00:00.gnu --outdir woof-restart
```

A benchmark run draws no products and runs no post-run verification. It
ends with one summary, also written to `woof-restart/benchmark-timing.json`:

- **read:** the input import.
- **init:** device restore, physics initialization and the first frame.
- **first step.**
- **steps:** the remaining steps, with their median and mean.
- **write:** from the last step until the last history frame is on disk.
- **finish:** the run receipt and certification capsule, and the worker's exit.
- **total:** from process start, so interpreter start-up and card selection
  are inside it, as they are inside `time woof run ...`.

On a warm start the run line and the step totals count only this run's
steps (for NCAR's 1 h restart leg, 50 steps from step 550). The `Timing for
main` lines keep WRF's own step numbers.

After WPS `metgrid.exe`, keep `met_em.d0*.nc` and the intended
`namelist.input` together. WOOF performs its native initialization:

```console
woof run --met-em metgrid-run --outdir forecast
```

Explicit `eta_levels` are preserved exactly. When they are absent, native
initialization generates the requested `e_vert` grid with WRF's automatic
algorithm. Both `auto_levels_opt = 1` and `2` are supported; `max_dz`, `dzbot`,
`dzstretch_s`, and `dzstretch_u` retain their namelist values. Omitted controls
take the WRF defaults (option 2, 1000 m, 50 m, 1.3, and 1.1). The resolved
grid and control values are recorded with the forecast. `--vertical-grid native`
explicitly chooses WOOF's native profile instead when no eta list is given.

Native model-level mappings can declare `model_top_pressure_pa`, the interface
above the highest mass level. This lets a native analysis initialize the model
on its own eta ladder and lid. Each target mass level must still be supported
by the decoded pressure column. A declared numerical divergence from WRF real's
strict top test co-locates a target with the highest source mass level when
the target lies above it by at most 2^-16 (1.53e-5) times that source pressure.
The gap is physical as well as numerical: the native top level is a full
pressure that carries the water vapour of its top half-layer, while the target
is the same level's dry pressure, and WRF's moisture integration leaves the top
level undried. At HRRR's 1731.475 Pa top mass level above a 1500 Pa lid this
is 231.475 Pa times the top-level vapour mixing ratio, plus GRIB and FP32
rounding: 0.00061 Pa on 2026-10-03 00Z and 0.00085 Pa on 2025-03-14 06Z, the
second past the earlier four-FP32-epsilon bound. 2^-16 holds a top-level vapour
mixing ratio up to 1e-5 kg/kg and is about 0.1 m in height. The target takes
the source endpoint value and no atmospheric layer is extrapolated. A target
further above the source top is refused, because its values would have to be
invented above the analysis. WRF real.exe has no top tolerance to copy here: it
refuses this start earlier, because its `find_p_top` takes the highest pressure
on the top source level and requires `p_top_requested` to be at least that.
The preprocessing receipt records
`native-pressure-top-colocation-relative-2pow-16-v2`.

The external WRF doors preserve WRF RRTMG when radiation scheme 4 is selected.
Changing it to RTE+RRTMGP requires `--rrtmg-variant rte-rrtmgp`. Other unsupported
controls fail with the missing capability named; the adapter does not silently
replace explicit physics. `--run-seconds 600` shortens a run within its supplied
forcing coverage.

Both routes use WOOF's shared clocks, physics setup, output, health checks, and
restart machinery. Input hashes, actual humidity/soil interpretation, selected
physics, and initialization choices are recorded under `forecast/input`; users
do not copy hashes into a launch command. The metgrid path prices initialization
and forecast memory separately before preparing a state.

Metgrid soil can contain depth-node stacks (`SOILT`, `SOILM`, `SOIL_LEVELS`) or
layer stacks (`ST`, `SM`, `SOIL_LAYERS`) with the corresponding WPS layer names.
The actual depths must support the selected land model's target geometry.
Flagged analyzed mass categories are retained according to the active physics
package. Flagged number concentrations (QNI, QNC, QNR, QNWFA, QNIFA) are
initialized natively for every package that transports them and are discarded,
as an inactive WRF species is, by a package that does not. The one flagged field
with no species in this build is QNBCA: under `mp_physics = 28` the door refuses
`FLAG_QNBCA = 1` by name, so regenerate the met_em without QNBCA. Likewise, a request for WRF's sea-level-pressure reconstruction is
distinct from the implemented source-pressure/terrain reconstruction.

Moving nests need forcing and static coverage at their future positions. A
directory containing only initial footprints cannot supply that coverage;
prepare a native statics corridor for those requests. These rectangular WRF
files do not establish compatibility with other mesh layouts.

## `use_theta_m` on the WRF doors

`use_theta_m = 1` selects moist potential temperature for dynamics. Physics and physical-temperature products use dry theta. `T` in history remains dry theta minus 300 K; `THM` holds the selected prognostic minus 300 K. Set `use_theta_m = 1` in the namelist or shared TOML configuration, or `RunConfig(use_theta_m=1)` in Python. The engine's own configuration default remains 0. Both choices use fast numerics by default.

An omitted namelist key takes the default of the declared WRF version: WRF 4 defaults to 1 and WRF 3.9 defaults to 0. `run --wrfinput` reads the version from the files' `TITLE`; `import-namelist` assumes WRF 4 unless given `--wrf-version 3`. The log and import receipt record the resolved value and its source.

For a requested value of 1, `run --wrfinput` requires matching `USE_THETA_M = 1` metadata and native `THM`, retains that prognostic directly, and retains moist boundary thermal fields. Missing or mismatched selected-variable inputs are refused explicitly. At 0 the existing dry conversion and arithmetic are retained.

Companion exports declare the selected value in `USE_THETA_M`. `T` is dry under both choices, `THM` is the selected prognostic, and boundary `T_B*` couples that prognostic to dry column mass. Use the same setting in stock WRF. Older exports written by 2.8.4 or earlier declared 1 over dry boundary rows and should be written again ([WRF-INTEROP.md](WRF-INTEROP.md)).

The moist route has completed a one-hour forecast matching all 190 common floating fields of native strict WRF 4.6.1 at every history time in the documented configuration. See `docs/dev/theta-m-port-audit.md` for that configuration, source coverage and concrete limits. This strict comparison does not change the fast default. Hydrostatic integration and digital filtering remain unavailable because their required solves and filter are absent.

A moist-theta engine checkpoint continues identically to an uninterrupted run in the tested configuration. Strict continuation from a native WRF restart file completes in the tested configuration with either thermal setting, but neither setting reproduces native continuation word for word. The source audit records this existing restart limitation and separates it from the engine checkpoint proof.

The first `wrfbdy_d01` record of an export is the coupling of the exported `wrfinput_d01` itself, as `real.exe` always writes it. A preparation whose start state comes from a separate analysis (`--initial-inputs`) built its boundary from the boundary source, so that first record and its tendency are rewritten the way WRFDA's `da_update_bc` rewrites them after an analysis: the values become the new start's coupling and the tendency still reaches the prepared boundary at the end of the first interval; later records are the prepared ones unchanged. When the preparation forces the Thompson-aerosol numbers at the boundary (`mp_physics = 28` with aerosols from input), `QNWFA_B*` and `QNIFA_B*` carry those tables; run WRF with `use_aero_icbc = .true.` to use them. Exports written by 2.8.6 or earlier with `--initial-inputs` held another atmosphere in that first record (stock WRF's outer row jumped by up to 10 m/s and 8 K in its first minute on a 3 km RAP crop, and `run --wrfinput` refused the pair), and mp 28 exports held zero aerosol tables; write them again.

What the files hold: WRF 4.0 through 4.7.1 `real.exe` writes `wrfinput` `T` as dry theta-300 under both settings and `THM` as the prognostic (moist theta-300 under `use_theta_m = 1`, equal to `T` otherwise), and `wrfbdy` `T_BXS` and its siblings couple the prognostic with dry column mass. At option 0 WOOF reads dry initial `T` and converts a moist boundary with its own vapour and mass. At option 1 it retains native `THM` and moist boundary values. Import checks that the first boundary record equals the coupling of the initial file's selected representation. A stock WRF 4.6.1 pair passes at both settings bit for bit. A `real.exe` build that fuses multiply-adds rounds that rebuild differently; the difference is bounded by a few FP32 steps of the full theta (a few hundred K) times the column mass, and the check allows that bound at each point. The refusal `wrfbdy T west does not match initial wrfinput_d01` states what was compared and how far apart the two files are, and names the case where the boundary holds dry theta under `USE_THETA_M = 1`: WRF 3.7 to 3.9.1.1 `real.exe` wrote that (`use_theta_m` entered the namelist in 3.7; no `THM` variable; its solver converted at run time), WRF 4.x never does. `use_theta_m = 0` in the producing namelist remains a fine setting: the files then hold dry theta everywhere and the import applies no vapour conversion.

## `fine_input_stream` on the WRF doors

`fine_input_stream = 2` is WRF's delayed-nest-start stream: the nest takes only its static and masked land-surface fields from its own input and interpolates the rest from the parent. `import-namelist`, `run --wrfinput` and `run --met-em` all accept it and announce it at the terminal as a declared divergence. The delayed child starts at its declared start time either way; what differs is where its masked surface state comes from, which in WOOF is the child's own-grid analysis rather than a `real.exe` `wrfinput`. `fine_input_stream = 0`, WRF's other defined value, is what WOOF already does for every domain and is recorded without a divergence. An index WRF does not define is refused by name, with both defined values stated. The RW-WPS support report and the importer read this answer from one function, so a namelist pair the report passes is a pair the importer imports.
