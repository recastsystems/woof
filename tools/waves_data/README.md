# Native wave inputs and products

`rw_waves_data REQUEST.json` is the common native data door. Python only builds
requests, invokes the bridge and draws analysis charts. Acquisition, decoding,
regridding, wave metrics and observation parsing run in Rust with Rayon.

A `prepare` request can supply `grid`, `start`, `end` and `output` without a
staged source. It acquires the public forecast, bathymetry and station registry
using `woof/ocean/waves/data/public_sources.v1.json`. The start selects one
forecast cycle; all boundary and reference-wind hours use that same cycle.
This differs from the archived verification recipes, which splice short leads
from successive cycles. Source coverage failures remain explicit errors.

Supported grids are `latlon`, `curvilinear` and ordered `points`. The shared
Python functions `prepare_grid` and `prepare_mesh` preserve static-grid identity
and call this door. `rw_waves_coupled_prepare` preserves the atmospheric caller's
binary interface and uses the same automatic preparation module. It returns the
prepared folder in its receipt. No `source_prepared` option or environment
variable is required. An explicit staged recipe remains an optional shortcut.

Preparation controls include `station_max_distance_km`, `coast_fill_cells`,
`stations` (auto or IDs), `source_catalog`, `input_cache`, `shared_inputs`,
`bathymetry`, `field_table`, `coarsening` (atmospheric grids), and `threads`.
Python maps `prepare_threads` to `threads`. Automatic registry selection limits
stations to the domain's latitude and circular longitude envelope and the
specified distance to a wet cell. Registry positions do not establish historical
mooring locations. Point depth samples cell centres, not polygon averages.

A completed manifest records array shapes and SHA256 hashes, complete hourly
coverage, boundary indices, the native source request and stage times. The
`audit` operation reads those bytes and checks geometry, wet masks, bounds,
partition semantics, forcing and non-calm initial state.

`observations` takes `prepared`, `acquisition_receipt` and `output`. It writes
`observations.json` with schema `gpuwm.waves.observations/v1`, independent ISO UTC
observation times, Hs/Tp/Dp arrays, null missing values, source URLs and hashes.
The acquisition receipt must contain a real retrieval timestamp and match the
observation bytes. It never labels export time as retrieval time.

`products` prepares maps for `rw_wrfbatch` via `woof.rustwx`. Set
`join_existing_output: true` when joining a normal run root. Products then land
at `<out>/<domain>/waves/<product>/<valid-day>/`; duplicates fail without
replacing an existing PNG. Public directions are nautical FROM. Map arrows
show travel TOWARD. Use CPU rendering unless a pool lease covers GPU rendering.

Verification also retains per-station sampled Hs/Tp maxima and all upward Tp crossings of 16 and 20 seconds. Each crossing records the prior below-threshold time, first above-threshold time and bracket width. A signal present at the first valid sample is left-censored. These are bulk-period diagnostics, not unique swell identification or interpolated arrival times. Missing-record gaps remain visible in the bracket.
