# Single-domain Python API

`woof.domain` exposes `API_VERSION = "1.0"`, `Domain`, `WarmBubble`,
`DomainSnapshot`, `device_free_bytes` and `estimate_device_bytes`.
Importing it does not import CuPy or open a device. Building a domain needs
the installed GPU runtime. This API runs the existing dynamics and physics;
it does not provide a CPU dynamics backend.

```python
import numpy as np
from woof.domain import Domain, WarmBubble

z = np.linspace(0, 20000, 401)
d = Domain.from_sounding(
    z=z, theta=300 + 0.003*z, qv=0.012*np.exp(-z/2500),
    nx=64, ny=64, nz=40, dx=1000, dy=1000, ztop=20000, dt=3,
    microphysics=8, boundary="periodic", bubble=WarmBubble(amplitude=3))
d.step(10)
checkpoint = d.snapshot()
d.dt = 1.5
d.advance(60)
w = d.to_host(["w"])["w"]
d.restore(checkpoint)  # original clock, fields, physics history and dt
```

Fetch coefficient tables with `woof fetch-tables`. A missing table raises
the engine's named error; this API never selects another physics scheme.
`examples/domain_api_warm_bubble.py` runs a 15-minute example and prints
finite-state checks, maximum vertical wind and wall seconds per step.

## Construction and time

`Domain.from_sounding(sounding=None, *, nx, ny, nz, dx, dy, ztop, dt, ...)`
accepts a mapping, a path, or separate keyword columns. Explicit columns
are `z` (metres) **or** `p` (Pa, strictly decreasing), `theta` (K), and
optional `qv` (kg/kg), `u`/`v` (m/s, zero when omitted). Arrays must be
finite, equally sized 1-D profiles. Heights must strictly increase from
zero through `ztop`. Pressure profiles start at `p_surf`; their heights
are obtained by hydrostatic integration using virtual potential temperature.
These heights feed the same dry-base/moist-balanced initialization as a
height sounding; pressure is a coordinate, not an imposed 3-D pressure field.

A CM1/WRF `input_sounding` path has a surface row
`pressure_hPa theta_K qv_g_per_kg`, followed by rows
`height_m theta_K qv_g_per_kg u_m_per_s v_m_per_s`. Blank lines and `#`
comments are accepted. If the first height is above zero, the surface
theta/qv row supplies height zero, with the nearest wind. Profiles are
linearly interpolated in height without extrapolating beyond their coverage.
`p_surf` defaults to 100000 Pa for height arrays, or the supplied surface
pressure for pressure arrays/files.

`microphysics` accepts the engine selector or scheme name; omitted means
no microphysics unless a `suite` is selected. `suite` accepts a named
single-domain physics profile. A conflicting microphysics selector raises.
`overrides` supplies other `RunConfig` keywords; explicit geometry and
boundary arguments cannot be overridden. `physics_options` supplies
`initialize_physics` keyword arguments, such as surface inputs and radiation
date/latitude/longitude required by a selected suite. Configuration and
physics validation are the existing engine validators. This constructor
builds flat, unprojected atmospheres. Use `from_state` for other preparation.

`boundary` is `periodic`, `open`, or `specified`. Specified construction
attaches constant side forcing from the initial atmosphere through
`cfg.run_seconds` (default 1e9 s); replace it with time-dependent tables
before stepping when needed. Open boundaries retain RunConfig defaults;
stabilization settings remain explicit overrides.

`WarmBubble(amplitude, radii, centre)` adds a cosine-squared ellipsoid during
balanced initialization. Amplitude is K; radii and centre are metres, x/y
relative to the horizontal domain centre and z above ground. The default is
3 K, radii `(10000, 10000, 1500)`, centre `(0, 0, 1500)`.

`Domain.from_state(state, cfg, physics=None)` wraps a prepared state without
changing its arrays or time. An explicit driver is attached; `None` retains
the existing driver. This factory validates configuration and grid shapes.
Preparation, including physics and boundaries, remains the caller's work.

`step(n=1)` calls `woof.core.dycore.step` exactly n times with the same
defaults. `time` reads elapsed model seconds. Setting positive finite `dt`
replaces the frozen RunConfig; `advance(seconds)` takes whole steps and a
shortened final step, then restores the original dt. Explicit `clock_dt`
and physics cadence settings keep their engine meanings. This API does not
install an adaptive physics-calendar policy: callers configuring one retain
that policy when using `from_state`.

## Arrays

`get(name)` returns the live device array, and `set(name, array)` copies into
its existing storage, converting dtype but requiring the exact shape.
Both return the live destination array. Array order is
`(vertical, south-to-north, west-to-east)`; float32 is the
normal state dtype. No halo is exposed. The extra face is part of the array.

| Field | Shape | Meaning/units |
|---|---|---|
| `u` | `(nz, ny, nx+1)` | x-face wind, m/s |
| `v` | `(nz, ny+1, nx)` | y-face wind, m/s |
| `w` | `(nz+1, ny, nx)` | vertical-face wind, m/s |
| `thp` | `(nz, ny, nx)` | perturbation potential temperature, K |
| `php` | `(nz+1, ny, nx)` | perturbation geopotential, m²/s² |
| `mup` | `(ny, nx)` | perturbation dry column mass in pressure units, Pa |
| `qv`, water mass fields | `(nz, ny, nx)` | dry-air mixing ratios, kg/kg |
| number/aerosol fields | `(nz, ny, nx)` | scheme-native number mixing ratios |
| `p`, `al`, `alt` | `(nz, ny, nx)` | engine EOS pressure/specific-volume diagnostics |

`prognostic_names()` lists checkpoint-carried state arrays, including
auxiliary carriers and end-of-step diagnostics required by the restart
contract. `moist_species()` lists the active water mass, number and aerosol
fields, including qv/qc/qr when allocated; it excludes PBL-owned scalars.
`to_host(names)` returns owned NumPy copies. `diagnostics()` refreshes EOS
diagnostics through `update_diagnostics` and returns live p/al/alt arrays.
After editing prognostics, call it before reading dependent diagnostics.

## Carry, snapshots and relocation

`carried_arrays()` returns `{"state": {name: verdict}, "scratch": ...}`
for allocated arrays, using the restart writer's unchanged classifiers.
State verdicts are `serialize`, `checkpoint_only`, `rebuild`, `setup`,
`derived_setup`, `infra`; scratch verdicts are `serialize`, `carry`,
`rebuild`. Unknown arrays raise the engine manifest error. `carry` is
cross-step state that ordinary restart files omit.

`snapshot()` returns owned host restart bytes and configuration. It uses
the engine restart writer, opting scratch `carry` slots in, and includes
physics fields, held tendencies, accumulators and clocks. Temporary NPZ
files are removed before returning. `restore(snapshot)` uses the engine
reader and requires the same prepared setup, grid, forcing and physics;
the snapshot's dt is restored. A mismatch raises before applying restart
arrays. This is an in-memory checkpoint interface to the engine format,
not a portable cross-release snapshot format. A previously prepared
second domain can restore the same snapshot.

`destination.copy_overlap(source, offset_i, offset_j)` maps source `(j,i)`
to destination `(j+offset_j,i+offset_i)`. Positive offsets move the copied
pattern east/north. Matching cell spacing, vertical size and microphysics
are required. It transfers carried horizontal state/scratch/physics arrays,
including the extra u/v/w faces, leaves uncovered destination cells at
their initialized values, and refreshes diagnostics. Missing carried
scratch storage is allocated with zero outside the overlap. It returns
copied element counts by `state/name`, `scratch/name` or `physics/name`. Overlap is clipped
independently on each staggering; no overlap is a no-op for that array.
Physics arrays use the restart writer's inventory, including surface fields
and held tendencies; matching destination storage must already be prepared.
Setup, physics clocks, scalar/object bookkeeping and elapsed time stay at the
destination. This operation supports explicit patch assembly; it is not
a complete moving-domain physics restart or a conservation remap.

## Radiation and lateral forcing

`force_radiation_next_step()` temporarily sets `radiation_due_override=True`
for the next step, then restores its previous value. A snapshot preserves
an outstanding request. Calling it without a configured radiation scheme
raises because an override cannot produce radiation without a solver.

`boundary_snapshot()` returns host float64 **mass-coupled** fields for the
existing lateral-boundary builder. They are not the primitive `get` units.
`extract_side(side, width=None)` wraps its side extractor; side is `west`,
`east`, `south` or `north`, width defaults to `cfg.spec_bdy_width`, and
east/north run outermost-first as in WRF boundary tables.

```python
side_names = ("west", "east", "south", "north")
first = {s: d.extract_side(s) for s in side_names}
# Supply a second set from an atmosphere at the interval's end.
d.set_lateral_boundaries(first=first, second=first,
                         start_seconds=0, end_seconds=3600)
```

Use this with a domain built with `boundary="specified"`. Alternatively
pass an existing `LateralBoundaries` object positionally or use
`intervals=[...]`. The wrapper delegates to `build_lateral_interval_from_sides`
and `attach_lateral_boundaries`, including attachment's clock reset and
forcing validation. Attach before stepping or restore a snapshot afterwards.

## Placing a domain and simulating radar

Use `woof.simulated_radar.simulate` to scan a placed domain between steps.
Supply `center_lat`, `center_lon` and an optional `start_time` to
`Domain.from_sounding`. `domain.grid` is the engine's `MercatorGrid`, with
true spacing at the requested centre and mass arrays shaped `(ny, nx)`.
The existing grid class requires `dx == dy`; a centre must be finite and
strictly between the poles. Without placement, `grid` is `None`.

Placement is geographic metadata for an idealized Cartesian atmosphere.
It does not install map factors or Coriolis, change RunConfig, or change a
prognostic word. Grid-relative winds follow the attached Mercator grid's
east/north axes. This places the idealized experiment on a map; it does
not turn it into a terrain-following real-data forecast.

```python
from datetime import datetime, timezone
from woof.domain import Domain, WarmBubble
from woof.simulated_radar import simulate

domain = Domain.from_sounding(
    "input_sounding", nx=64, ny=64, nz=40,
    dx=1000, dy=1000, ztop=20000, dt=3, microphysics=8,
    bubble=WarmBubble(amplitude=3), center_lat=35.55, center_lon=-97.10,
    start_time=datetime(2026, 1, 1, tzinfo=timezone.utc))
domain.advance(900)
result = simulate(domain, site="KTLX", formats=("level2", "cfradial1"),
                  outdir="radar-products", valid_time=domain.valid_time)
print(result.paths)
```

`valid_time` is `start_time + timedelta(seconds=domain.time)`, or `None`
without a start time. Naive start datetimes mean UTC; aware datetimes are
converted to UTC. Radar's history transport requires a whole-second valid
time. The radar call reads the state between steps and uses the attached
grid's `latlon_mass()` and `rotation_m()` methods. See
[the radar Python API](simulated-radar-python.md) for scan options and
scientific limits. `Domain.from_state(..., grid=grid, start_time=epoch)`
can attach an already prepared engine grid with matching dimensions and
spacing. Placement and epoch remain with the destination wrapper during
`restore` and `copy_overlap`; the numerical restart does not relocate them.

## Memory and compatibility

`device_free_bytes()` returns the existing admission reading, including
reusable pool bytes, or `None` when a device cannot be read.
`estimate_device_bytes(nx, ny, nz, physics=cfg, p_top=5000,
n_lbc_intervals=0)` delegates to the engine's itemized single-domain
resident-plus-transient estimate. `physics` is a full `RunConfig`.
It excludes the CUDA context, allocator margin and other domains; it is
not the forecast admission envelope or a promise that an allocation fits.

API_VERSION 1.x keeps these call names, argument meanings, layout/staggering,
ownership and offset conventions. Optional keyword arguments may be added.
Scheme-specific field inventories follow the selected engine scheme.
Numerical changes and restart-format compatibility follow engine releases;
API versioning does not freeze the solver or promise cross-release bits.
Existing core, physics, boundary and restart entry points remain available;
this module adds a public route beside them. `state` and `cfg` are available
for advanced integration, but arbitrary engine attributes are outside this
API's stability promise.
