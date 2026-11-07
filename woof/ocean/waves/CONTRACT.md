# Spectral device contract v1

Status: interface frozen for the waves lanes. This document specifies the new
implementation; the legacy WaveModel is not evidence that it is implemented.
The machine-readable companion is spectral_contract.json.

## State, coordinates and ownership

The core state is CUDA float32 C-contiguous action density
`N = E(f,theta)/(2*pi*f)`, shape `[nwet,nf,nd]`. Flattened offset is
`((cell*nf)+frequency)*nd+direction`. Exchange spectra are always variance
`E`, in m2 Hz-1 rad-1, with the same ordering. No conversion of the frequency
measure is implied by dividing by intrinsic angular frequency. Frequency
centres are `first_hz*ratio**i`; integration widths are
`f[i]*(sqrt(ratio)-1/sqrt(ratio))`. Default nf=36, nd=36, first_hz=.035,
ratio=1.1. Theta is TOWARD, counterclockwise from local east, in radians.
FROM nautical degrees are `(270-theta*180/pi)%360`.

All kernels, setters and steps enqueue on the caller's current CUDA stream.
Setters retain device references until replaced; callers must not change those
buffers before consumption completes. No setter or step performs a host scalar
read, device synchronization, file I/O or implicit CPU array conversion.
Geometry construction, restart validation and explicit output collection may
synchronize. Shapes/dtypes/strides are checked without reading device values.
Constructor validation rejects nonfinite geometry and inconsistent masks.

`wet_to_full` is CUDA int32 `[nwet]`, in ascending flattened full-grid index
order for structured grids; Voronoi uses ascending original cell index.
`full_to_wet` is CUDA int32 `[nfull]`, dry=-1. `ncell` means nwet in every
backend call. Full-grid structured arrays have shape `[ny,nx]`; mesh arrays
have shape `[nfull]`. Compact arrays never silently substitute for full arrays.

## Grid backend Python protocol

The importable protocol is `woof.ocean.waves.backend.GridBackend`.
Backend registration uses `config['backend']` (latlon, curvilinear, voronoi)
or an already constructed `config['grid_backend']`. A custom backend must have:

* `name`, `ncell`, `full_shape`, `wet_to_full`, `full_to_wet`;
* CUDA float32 `[ncell]` `lon`, `lat` in degrees, `depth` in metres,
  `area` in m2, `depth_gradient_east`, `depth_gradient_north`;
* CUDA int32 `[nboundary]` `boundary_cells`, in compact order; its ordering
  defines the boundary spectrum ordering, and may repeat a corner cell only
  once. Boundary-face normals and connectivity belong to the backend;
* `max_dt(cg_max: float) -> float`: conservative geographic CFL seconds for
  any group speed up to cg_max. This uses host geometry bounds computed once
  at construction, never a per-step device reduction. Core supplies the safe
  no-current envelope `g/(2*pi*first_hz)`. Current support must include its
  configured speed envelope in this bound;
* `advect(state, group_velocity, dt, stream, incident_boundary) -> array`:
  state `[ncell,nf,nd]`, group_velocity `[ncell,nf]` in m/s,
  incident_boundary either None or action `[nboundary,nf,nd]`, dt seconds,
  stream a CuPy CUDA Stream. Returns a distinct persistent scratch array of
  the same shape. Does not mutate state. The core copies the returned array
  into its persistent action allocation, preserving captured CUDA addresses;
* `set_depth(depth, stream)`: updates device depth and east/north gradients
  without changing wet-cell order, allocation or host CFL bounds. Prepared
  coast topology stays fixed; drying cells emit zero action.

Optional paired hooks: `frequency_steps(first_hz, ratio, nf)` returns tuples
`(first_bin, bin_count, maximum_dt)` covering the frequency axis exactly once;
`advect_group(..., first_bin, bin_count)` advances only those bins and returns
persistent full-shaped scratch. Unselected bins need not be initialized. The
core copies only selected bins and applies direction turning at that interval.
Backends without both hooks retain the single-interval `advect` path. The
structured backend uses the safe finite-depth envelope `0.6*g/(2*pi*f)`
at each group's lowest frequency, rather than the looser common `g/sigma`.

Optional hook: `turning_handled: bool`, default false. A backend setting this
true must rotate directional spectra between cell tangent frames during its
geographic flux exchange. The core then omits only its analytic spherical
`-cg*cos(theta)*tan(latitude)/R` term. Bathymetric refraction, depth-change
frequency transport and every source term remain enabled. The Voronoi backend
uses exact great-circle parallel transport through each shared edge and
conservative periodic linear interpolation of its directional bins. This
removes the coordinate singularity at a polar cell without masking that cell
or clipping latitude. The hook is included in the restart geometry identity.

The backend owns geographic propagation only. It must use true face lengths,
areas and east/north normals. Curvilinear geometry uses dx/dy divided by mass
map factors, and grid-x rotation `(cosalpha,sinalpha)`; grid-y is
`(-sinalpha,cosalpha)`. Forcing remains earth-relative. Refraction,
great-circle turning, frequency propagation and sources belong to the core.
Interior shared faces have one flux used with opposite signs by their cells.
Open faces use incident spectra only for inflow and the interior state for
outflow. Coasts have no incoming spectrum; outgoing action is absorbed.
This resolves SPEC's 'closed (outgoing leaves, nothing incoming)' wording as
an absorbing coast, not a reflecting wall.

## CUDA calling convention

CUDA calls use `extern "C" __global__`, float32 pointers, int32 dimensions,
int64 indexing for spectral offsets, float32 dt, and the supplied stream.
Launch block size is an implementation choice, not an ABI constraint. Kernels
do not allocate, synchronize with the host, or own input memory. Direction is
the fastest axis. A backend may implement its own kernels, including Voronoi;
there is no required C symbol shared by independently compiled RawModules.
The core calls only the Python protocol above. The canonical face convention
for reusable geographic kernels is:

```c
void wave_face_flux(const float* action, const float* cg,
 const float* incident, const int* left, const int* right,
 const int* boundary_slot, const float* normal_east,
 const float* normal_north, const float* length, float* flux,
 int nface, int nf, int nd);
void wave_cell_update(const float* action, const float* flux,
 const int* face_offsets, const int* face_indices, const int* face_sign,
 const float* area, float* result, int ncell, int nf, int nd, float dt);
```

Face left is always a wet compact cell; right is a wet compact cell or -1.
Normals point outward from left. boundary_slot=-1 denotes absorbing coast;
otherwise it indexes incident spectra. A null incident pointer means zero
incoming spectrum. Flux includes face length, positive left-to-right.
CSR face_sign is +1 for left and -1 for right; result is
`action - dt/area * sum(face_sign*flux)`. Higher-order reconstructions add
backend-owned stencil arrays without altering this sign/layout contract.

## Initial spectra and boundaries

`initialize_partitions(partitions, *, valid_time, shape_table)` accepts CUDA
float32 `[nfull,4,4]`: Hs metres, Tp seconds, Dp FROM degrees, kind 0=windsea,
1=swell. Missing partition: Hs=0, remaining fields ignored. Negative Hs,
nonpositive Tp for active partitions, and unknown active kinds are errors.
`shape_table` is the parsed partition_shapes.v1.json owned by the data lane;
its exact version and SHA256 go into the receipt and restart metadata.
Reconstruction is on GPU. Each partition is discretely normalized so its
integral is Hs**2/16, then the partitions are summed and divided by sigma.
The shape is JONSWAP in frequency and cos**(2*s)((theta-theta_p)/2)
in direction, with table-selected gamma, sigma_a, sigma_b and s.
The table schema is the data lane's authority; no second coefficient table.

`set_boundary_spectra(e0, e1, *, time0, time1)` accepts exchange E in CUDA
float32 `[nboundary,nf,nd]` and finite host time endpoints, time1>time0.
`set_boundary_partitions(p0,p1,*,time0,time1,shape_table)` accepts
`[nboundary,4,4]` and reconstructs both endpoints once on GPU.
At each geographic substep midpoint the core computes alpha from its host
clock and launches `N_b=((1-alpha)*E0+alpha*E1)/sigma` on GPU. It interpolates
spectra, never period or direction. Time outside the bracket raises before
enqueueing a step; no extrapolation or silent endpoint hold. Callers refresh
the bracket at forcing knots and split steps there. None is supported only
for an explicitly chosen idealised zero-incoming boundary.

## Shared ocean ABI and outputs

`create_solver(grid,bathymetry,*,config=None)` retains its signature.
`set_forcing(AtmosphereForcing)`, `set_water_level(eta)` and `step(dt)` retain
full-grid device input and asynchronous semantics. Existing `.outputs.hs`,
`.tp` and `.direction` remain full-grid persistent CUDA float32 views.
`direction` is legacy mean propagation TOWARD, clockwise from north.
`config['spectral']` retains frequencies/directions/first_hz/ratio.
`config['source_pack']` selects st4 or janssen. Selection for release is
pending buoy evidence; neither label denotes the legacy hybrid source.

Extended `.outputs` exposes hs, tp, tm01, tm02, dp, dir, dspr, ws_hs, ws_tp,
ws_dp, sw1_hs, sw1_tp, sw1_dp, sw2_hs, sw2_tp, sw2_dp, sw3_hs, sw3_tp,
sw3_dp, ustar, charnock, z0, direction. dp is peak FROM, dir is mean FROM,
dspr is directional spread in degrees. Partition directions are peak FROM.
Dry and calm bulk wave outputs are zero. Roughness outputs over wet calm
water follow the surface-layer solution; dry roughness is zero. Missing
partitions are zero, ordered windsea then swell by descending variance.
Additional `.outputs.tau_wave_east` and `.tau_wave_north` are wave-supported
stress in N/m2; they do not add required file product keys.

`set_stations(indices)` accepts CUDA int32 `[nstation]` original full-cell
indices, validated at initialization as wet and in range. `station_spectra()`
returns persistent CUDA float32 E `[nstation,nf,nd]` gathered on the current
stream. `station_outputs()` returns a mapping of the bulk keys to device
float32 `[nstation]`. The Rust writer owns points.nc with dimensions
`[station,time,nf,nd]`, FROM coordinate `(270-j*360/nd)%360` (not sorted).

`step(dt)` advances exactly dt with no growth-selected host or device
microsteps. Geographic substeps use the backend CFL bound. Spectral turning
uses an implicit periodic solve. Source packs use semi-implicit integration
with a published increment limiter. Time advances only after launch success;
CUDA execution failures invalidate the solver and must surface at collection.

## Restart

`restart_state()` returns `{metadata: dict, arrays: dict}`. Arrays are device
views, not a host copy: action, depth, water_level, plus model-owned prognostic
auxiliary state if any. Metadata includes schema `gpuwm.waves.restart/v1`,
time_seconds, spectral_contract version, spectral grid, source pack and its
parameter digest, backend name, geometry SHA256, wet-order SHA256 and partition
shape SHA256. Boundary and forcing data are external inputs and are rebound
at the restored time before stepping. Geometry/mask/spectral/parameter hash
mismatches fail closed. `restore_state(payload)` validates and enqueues
copies into existing buffers; it never changes their addresses. Persistence
is the caller's Rust checkpoint writer. Derived moments, geometry and outputs
are refreshed after restore; diagnostics are not prognostic state.

## Delivery dependencies

Implementation extension: graph capture retains the action buffer address.
The device clock is float64 `[1]` epoch seconds and device step count is int64
`[1]`; they advance in CUDA on every execution, including graph replay. Boundary
endpoints and their float64 time bracket use stable storage, refreshed before
replay when needed. Out-of-bracket replay sets a device input-error flag and
invalidates the spectrum; collection reports the failure. Normal Python calls
also validate their known host bracket before launch. No silent extrapolation.
`config.time_seconds` or initialize_partitions establishes the clock; forcing
setters do not reset it during capture. There is currently no device gate to
skip source work between coupling intervals; captured step(dt) advances every
execution and its effective cadence must be measured.

`checkpoint_arrays`, `checkpoint_metadata`, `validate_checkpoint` and
`restore_metadata` expose the same restart state to the coupling transport.
Additional persistent restart arrays are depth_previous, surface_iteration,
shelter_iteration, wind_input_iteration, clock and steps. The wind input is
retained because the next source moment uses its positive-input windsea mask.
The mask is now packed as uint32 [nwet,ceil(nf*nd/32)], low bit first within
each word. It replaces the former full float32 beta array, which is scratch.
The iterative surface state is retained to
make a continued solve reproduce its finite-iteration roughness history.
Callers pass partition_shapes_sha256 in config when constructing directly for
restore; it is the canonical sorted compact JSON SHA256 of the shape table.
Constructor geometry hashes and checkpoint metadata may synchronize; stepping
and checkpoint_arrays do not. `collect()` is the explicit output barrier.
Per-stage CUDA globaltimer marks accumulate on device over every graph replay.
They add no synchronization and report stage wall time, including shared-card
scheduling stalls. Host enqueue time is recorded separately.

Memory: source terms and watershed diagnostics share a persistent cell-tile
arena, default 1024 wet cells. Tiling is solely an execution layout; spectra
and source physics are unchanged. The structured backend borrows two RK
buffers and a positivity budget with bind_workspace(stage,result,budget).
After transport, the first RK buffer holds directional turning until the next
transport call. The geographic state stays at a stable address. A backend
without bind_workspace retains its own buffers, plus the core turning buffer.
Source diagnostic arrays are scratch views; use source_diagnostics(cell) at
output cadence rather than treating beta/loss/seed as full-grid state.

Core owns this ABI, reconstruction, spectral physics and structured backends.
The data lane owns shape-table schema, prepared-reader and product/verification
modules. The hex lane owns backends/voronoi*. The coupling lane owns the
shared ocean rows/history integration and checkpoint transport. Each must
record implemented and validated status separately from this interface.
