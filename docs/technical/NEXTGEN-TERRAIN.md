# Literature-based 2D terrain routing

SPDX-License-Identifier: Apache-2.0

This is an original numerical implementation. No external model source code,
software ABI, or numerical fixtures are used. The equations describe physical
laws; the discretization and boundary choices below are WOOF-ArWen's own. This is
not a reproduction of CASC2D, GSSHA, HEC-RAS or WRF-Hydro.

Julien, Saghafian and Ogden (1995), *Raster-based hydrologic modeling of
spatially-varied surface runoff*, JAWRA 31, 523–536,
[doi:10.1111/j.1752-1688.1995.tb04039.x](https://doi.org/10.1111/j.1752-1688.1995.tb04039.x),
provides the diffusive-wave terrain-routing context. The public
[USACE HEC-RAS 6.4 hydraulic reference](https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/6.4/theoretical-basis-for-one-dimensional-and-two-dimensional-hydrodynamic-calculations/2d-unsteady-flow-hydrodynamics)
describes the pressure/friction balance and continuity formulation.
[USGS Professional Paper 708, *Ground-water hydraulics*](https://pubs.usgs.gov/pp/0708/report.pdf),
equations 13–16, defines hydraulic conductivity, transmissivity and specific
yield. These are literature references, not software dependencies or sources
of copied implementation text. Publication copyright does not imply a licence
for associated model software; none is included here.

## Equations and discretization

Square cells have width `dx` in metres and area `A=dx²`. Surface water depth is
`h` (m). Groundwater storage `w` is water-equivalent depth (m), not saturated
thickness. Saturated thickness is `b=w/Sy`, where `Sy` is specific yield. Bedrock
is `z-D`, with terrain elevation `z` and soil depth `D`. Heads are `Hs=z+h` and
`Hg=z-D+b`.

Continuity is `∂h/∂t = r - ∇·q + e`. Manning diffusive flow has unit-width
discharge `q = -(h_face^(5/3)/n) sqrt(|∇Hs|) sign(∇Hs)`. Along each of the
four cell faces, a higher-head donor exports a nonnegative depth rate
`h_face^(5/3)/n_face * sqrt((Hs_i-Hs_j)/dx) / dx`.
Here `h_face=max(Hs_i-max(z_i,z_j),0)` and roughness is the arithmetic mean
of the two cells. This blocks flow across an emergent terrain sill and gives
zero flux for equal water-surface elevations, including over uneven terrain.
It is an axis-separated D4 diffusive discretization, not a full-inertia solver.

Darcy unit-width discharge is `qg=-K*b*∇Hg`. The D8 groundwater path is the
neighbor with the largest positive hydraulic gradient, with diagonal length
`sqrt(2)*dx`; equal-gradient ties follow W,E,N,S,NW,NE,SW,SE order. Conductivity
at an interface is the harmonic mean; transmissive thickness is the donor's
saturated column above the higher of the two bedrock elevations, capped at
the donor land surface. The resulting water-depth loss rate is
`K_face*b_face*(Hg_i-Hg_j)/(distance*dx)`. Selecting one D8 neighbor is an
explicit modeling choice; it is not an isotropic eight-face Darcy solver.

Nonnegative surface sources and recharge are rates in m/s. Each substep first
adds these rates over `dt`. Storage beyond `Sy*D` becomes surface water, both
before routing and after lateral inflow. This represents immediate saturation
excess return flow, without a separately resolved vertical pressure column.
Infiltration, evapotranspiration and unsaturated-zone physics belong to the
land-surface/runoff component; do not supply rainfall here if that component
already supplied the same water as runoff.

Coupled mode uses WOOF-ArWen's existing atmosphere/Noah-MP forcing. It does not
require an additional external meteorological forcing or PET stage. At the
parent routing exchange, `sfcheadrt` is surface-head **state in mm**;
`infxsrt` and `soldrain` are **mm per LSM step**; `smois`/`sh2o` aliases
are layers first. The coupling owner supplies the actual resident field
bindings, grid mapping and state writeback. Convert a step-depth source to
this terrain rate ABI as `depth_mm / (1000 * lsm_dt_seconds)` on CUDA.
Convert head state to metres as `head_mm / 1000`, with no division by time;
do not add that same stored head again as a source.

Do not infer units from `runsfxy`/`runsbxy` names. At base
`64fc403dfd4305510ca478b2473c263b24886e73`, the glacier writeback in
`woof/core/noahmp_runtime.py` multiplies `runsrf`/`runsub` by the LSM step
duration before assigning those carriers. The ordinary GPU water kernel
also multiplies runoff by `dt_soil` in
`woof/core/kernels/noahmp_water.cu`; the batch direct-write table subsequently
copies those already integrated depths. Both actual parent paths publish
step depths. A blanket instantaneous-mm/s interpretation is invalid.
Provider provenance must name the actual field semantics and LSM duration.

The provider must distinguish new aquifer recharge from subsurface runoff
already exported by the land model. A `soldrain` or other drainage field
can feed `recharge` only with an explicit compatible aquifer budget;
`runsbxy` must not be relabelled as recharge. Catchment-average fluxes also
require an explicit area-preserving grid mapping; their array order cannot
be bound as terrain cells. Initial groundwater storage, mapping and
physical parameters must be identified in restart provenance before an
event run can establish hydrologic skill.

The coupling owner's corrected parent exchange at `485b468bdf4` exposes
`infxsrt` as parent step-depth surface runoff and `soldrain` as genuine
bottom drainage integrated over the LSM step. Its m/s outputs are catchment
means; they are not arrays on this terrain grid. Terrain binding still
requires an explicit raster mapping and physical aquifer initialization.
Exposing `sfcheadrt` alone does not establish infiltration feedback.

Inactive cells and the rectangular exterior are impermeable. A supplied drain
width `B>0` opens a one-way connection to external receiving stage `Hc`. The
surface boundary uses Manning discharge across width B over distance `dx/2`,
with flow depth `max(Hs-max(z,Hc),0)`. Groundwater uses `K*b*(Hg-Hc)/(dx/2)`
across B when `Hg>Hc`. High receiving stage suppresses outflow. Channel-to-grid
reverse inundation is not modeled by this one-way boundary. The coupling must
explicitly route exported water and provide any prescribed external inflow as
a source; this choice must accompany inundation/skill results.

Each donor scales all its surface exports by `min(1,h/(dt*sum_rates))`, and
groundwater exports independently by `min(1,w/(dt*sum_rates))`. A double sum
and downward float conversion prevent floating-point overdraw. A second CUDA
grid gathers the exact stored outgoing rates from neighbors and updates both
stores in place. Internal transfers cancel by construction; storage conversion
to float32 adds ordinary roundoff. Positivity limiting alone is not a temporal
accuracy guarantee: choose `dt` for the water-depth/grid/slope regime and
report it. There is no host adaptive-time-step calculation.

## Resident ABI

`woof.nextgen.terrain.Terrain2D` prepares the component. Supply `dx`, `dt`,
`period_seconds` (an integer number of substeps), and a nonempty `identity`
object with grid, parameter and ordering provenance from the Rust input path.
All fields are same-card CUDA C-contiguous arrays of shape `(ny,nx)`:

| Field | Type | Units / meaning |
| --- | --- | --- |
| surface_depth | float32 | m, mutable carried depth |
| groundwater_storage | float32 | m water-equivalent, mutable carried store |
| elevation | float32 | m, terrain datum shared with receiving stage |
| soil_depth | float32 | m, positive |
| specific_yield | float32 | dimensionless, `(0,1]` |
| manning | float32 | s/m^(1/3), positive |
| conductivity | float32 | m/s, nonnegative |
| active | int32 | 0 sealed/inactive, 1 active |
| surface_source | float32 | m/s, nonnegative excess water source |
| recharge | float32 | m/s, nonnegative aquifer recharge |
| outlet_stage | float32 | m, externally maintained receiving stage |
| outlet_width | float32 | m, zero disables the cell's drain |
| surface_outlet_volume | float64 | m³, cumulative per-cell export |
| groundwater_outlet_volume | float64 | m³, cumulative per-cell export |
| exfiltration_volume | float64 | m³, cumulative internal return-flow transfer |

Optional output arrays are initialized to zero at setup if omitted. State and
output arrays cannot alias any other field. Read-only fields may share storage.
All values must be finite; depths and source rates must be nonnegative.

`advance(dt_seconds, stream=parent_stream)` (also named `step`) only enqueues two grid launches per
fixed substep. It does not allocate, read device scalars or synchronize. `.arrays`
contains every cross-step carrier including external forcing/parameters and
the uint32 `.health` bitfield. `.restart_identity` identifies the algorithm,
numerics, ordering and prepared durations. Pointers never alternate or change.
Scratch fluxes are fully overwritten before each update, so they are not restart
state. Inspect health outside the hot path: bit 1 invalid inputs, bit 2 nonfinite
flux, bit 4 invalid updated storage. Nonzero health means the run failed.

Surface depth is the terrain water-depth field for the independent verification
and products lanes. They own inundation diagnostics and USGS/NWM skill checks.
The stable `.depth_m` view and `.outputs` mapping expose depth and the three
cumulative volume fields without copying or changing their pointers.
There is no numerical acceptance against any UCAR executable or capture.

### Transfer and scalar ownership

Terrain owns the water-store updates, donor-limited transfer rates and all
three cumulative-volume arrays. Each cell has one update thread. That thread
gathers the stored float32 rates into float64 sums in fixed direction order,
updates both water stores and accumulates the float64 volume outputs every
substep. External surface and groundwater exports use precisely
`((double)dt * (double)stored_rate) * ((double)dx * (double)dx)`, from flux
planes 4 and 13 respectively. Exfiltration is added to surface storage in
that same substep; its volume is an internal transfer, not another external
source or outlet export. These are per-cell arrays, not global scalars.
There is no atomic floating-point reduction or host fold in this path.

The caller owns source/recharge and receiving-stage updates on the same
stream before the consuming terrain launches. Coupled-driver owns stage
ordering and checkpoints all public carriers through `.arrays`, including
the three cumulative arrays and health. A routing or coupling consumer that
needs a mapped transfer or scalar must complete its CUDA transfer/reduction
before using it, with the required operation order and precision. The private
14-plane flux scratch is overwritten at the next substep and is not a
published receiving-channel interface. Cumulative volumes alone do not
establish a channel inflow connection or an interval transfer: subtracting
large totals may lose small increments. No such channel consumer is bound
in the measured terrain workload.

Only diagnostic reductions with no downstream numerical consumer may be
deferred to an output boundary. Explicit strict checks still compare those
diagnostic carriers with their declared ordering and stored-word precision.
The replay lane owns the shared comparator and harness; this unit does not
implement a competing comparison tool. Native UCAR scalars and captures
remain excluded from this clean-room implementation.

The resident assembly entry point is `woof.nextgen.terrain:prepare`, using
the shared component-factory keyword signature. Its options contain exactly
`dx`, `dt`, `period_seconds`, `identity` and `fields`. Each fields entry names
`{"component": "prior_stage", "array": "resident_name"}`. Bind all twelve
constructor fields; the three cumulative-volume fields are optional. Providers
load static parameters, initialize water stores and update sources before this
stage. The factory resolves pointers once and retains provider identities and
the explicit bindings in restart identity. It refuses incomplete fields or
arrays on another card. It supplies no hydraulic defaults or file adapter.

Example stage options (expand `fields` for every constructor field):

```json
{
  "dx": 1000, "dt": 10, "period_seconds": 300,
  "identity": {"grid": "sealed terrain manifest SHA", "parameters": "parameter identity"},
  "fields": {
    "elevation": {"component": "static", "array": "elevation_m"},
    "surface_source": {"component": "runoff", "array": "terrain_excess_m_s"}
  }
}
```

Terrain owns the table-driven Rust/all-core DEM and static-grid preparation in
`tools/nextgen_terrain/README.md`. Hydrofabric supplies case bounds, CRS,
divides, network and catchment mapping; products uses the same sealed elevation
and grid for HAND. Source raster SHA256, request/table digests, data licence
and case/grid identity accompany the prepared fields. A mutable DEM service
URL alone is not a source pin. Its
coverage mask and topographic D8 field do not imply catchment or channel
assignment; the hydrofabric geometry provider must supply those relationships.

## Raw engineering capture transport

The verification lane owns the fresh `tools/nextgen_replay` comparator and
`woof.nextgen.raw_cuda.RawCuda` transport. The tools-path `raw_cuda` module
re-exports the same class for existing replay scripts. New isolated snapshots
stage both paths together; previously sealed snapshots retain their pinned
bridge. Product imports require only the product package on the import path. A setup manifest supplies explicit paths, SHA256,
dtype, shape and packed C/Fortran order. Terrain fields require little-endian
float32 or int32 in C order; the transport performs no transpose or conversion.
`RawCuda.load_many` allocates CUDA arrays and pinned host buffers at setup,
calls Rust `nextgen_raw_read_many` to check seals/sizes and fill those host
buffers, then enqueues CUDA host-to-device copies on the supplied stream.
It waits once before releasing the pinned buffers. A failed read batch is
discarded; it cannot become accepted input state.

`RawCuda.store_many` copies contiguous CUDA storage to pinned host buffers at
an output boundary, waits for those copies, then calls Rust
`nextgen_raw_write_many` to write new files without payload conversion. The
C ABI takes host pointers, never CUDA device pointers. No bridge call or
synchronization occurs inside `Terrain2D.advance`. Python handles metadata,
buffer lifetime and CUDA calls; Rust handles raw I/O, seals and comparison.
The comparator preserves signed-zero and NaN payload bits. These engineering
captures do not turn the clean-room solver into a UCAR numerical replay.

## Direct interval export and ordered channel-cell producer

`surface_interval_volume` and `groundwater_interval_volume` are stable f64 C(ny,nx)
outputs in m3, included in `.arrays` and `.outputs`. Every `advance` resets them
inside its first `terrain_update` launch, including inactive cells, then adds the
same promoted stored-flux increments used by cumulative outlet volumes. No
cumulative subtraction, host reset, extra reset launch or exfiltration source is
used. Their completed values describe exactly the most recently completed terrain
period. Consuming stages must run after that period on the ordered stream and
before the next terrain advance. Checkpoints preserve these carried arrays;
resuming at a completed-period boundary starts a fresh interval on next advance.
Low-level `update(reset_interval=True)` starts an interval explicitly; bare
`update()` continues it for off-path profiles or explicit substep callers.

`woof.nextgen.terrain:prepare_lateral` prepares the terrain-owned reduction
producer. Its options are exactly `terrain_component`, `mapping_component`,
`offsets_array`, `indices_array`, and nonempty `identity`. The latter must seal
terrain geometry, inverse map and current node axis; mapping-provider identity is
also retained. Bind the fabric owner's actual `routing_cell_offsets` i64[N+1]
and `routing_cell_indices` i32[M]. These enumerate ascending flattened cell
indices per current collapsed node. The map is channel-cell incidence, never an
assignment of arbitrary off-channel cells to a reach. Missing physical widths or
stages cannot be inferred from it.

After terrain, this stage runs one plain CUDA launch, 128 threads/block and one
thread/node. Each thread adds surface then groundwater interval m3 for each cell
in deterministic ascending order in f64, divides by the prepared period in f64,
and rounds once to f32. Its stable output is `terrain_lateral` f32[N] m3/s,
an **isolated contribution**, overwritten each period. It neither advances terrain
nor accumulates into an existing runoff vector. The driver must order terrain,
this producer, any owned CUDA source composition, then the receiver. There is no
host reduction, file operation, allocation, synchronization or graph per period.
Malformed row bounds/indices set this producer's health bit8; a nonfinite or
negative resulting rate sets bit16. Nonzero terrain or producer health invalidates
the run. Rust owns full map/grid/axis validation and uniqueness across rows.

All node rows, including reservoir rows, are preserved in the producer output.
Current MC expects interval-average f32 m3/s over its full current node axis;
its native-compatible reservoir gather defaults to zero lateral. MC commit
`81c30410af83fd60b2ea0668a2569cda2e6a70b3` adds the explicit boolean factory
option `reservoir_lateral_inflow`, default `false`. The reservoir owner confirmed
its receiving convention on 2026-10-18 at 00:40:41Z: after the ordered upstream
sum, add the node's interval-average lateral rate once, publish the total to
both reservoir inflow endpoints, and leave the separate reservoir lateral
argument zero. The two endpoints describe one current interval inflow; they
are not two volume contributions. Do not add that lateral again downstream.
The flag and inflow convention are checkpoint identity. It adds no receiving
launch or host computation. MC's focused 0/24-word receiving test uses a stub
for reservoir mathematics; it does not establish production-reservoir transfer
conservation.

For an explicitly selected **terrain-only** lateral source, the relevant MC
factory option fragment is:

```json
{
  "lateral_component": "terrain_exports",
  "lateral_array": "terrain_lateral",
  "reservoir_lateral_inflow": true
}
```

Here `terrain_exports` is the realization's name for this prepared producer.
This is a binding fragment, not a runnable physical realization. The driver
must select and seal the source partition: water supplied to terrain must not
also enter routing directly through a catchment runoff vector. If an independent
runoff contribution is retained, an owner-supplied CUDA composition must combine
the disjoint contributions on the same current node axis before MC consumes
them. Positive terrain drain exports must have a receiving map entry, or a
separately named external-boundary budget; the channel-incidence map excludes
off-channel cells. Width and mapping providers must establish that coverage
before a physical run. This producer supplies neither the source composition
nor its partition or coverage proof.
The stream order is parent exchange, terrain advance, interval producer,
optional agreed composition, then routing/reservoir consumption. Each component
uses the same prepared period. Read both terrain and producer health at the
output boundary; deferred diagnostics cannot substitute for consumed transfers.

**The producer remains unbound to a physical MC realization.** The receiving
convention is agreed; driver selection, source partition and production-receiver
execution are still pending. No channel elevation/datum or reverse backwater is
inferred. Hydrofabric owner e6398dc supplies the actual 6742-node,
39983-cell inverse map; this does not establish physical drain parameters,
actual parent-to-terrain forcing, or event validation.

The shared replay owner retains the comparator/bridge and numerical harness.
Current shared transport is `tools/nextgen_replay`, with `nextgen_raw_*` host ABI
returning 0/2 and Rust pinned-host I/O plus native CUDA copies at setup/export.
The historical `tools/hydro_replay` path is absent on the current NextGen tree;
its withdrawn ABI and captures are not used here. Existing depth/max/health
verdicts predate this extension; interval and reduction verification must be
reported from a new measured run, not inferred from earlier results.

Measured extension evidence: verification candidate f93cb5d3af5a618c413135243f95af494a328ada
ran terrain numerical candidate48d1ca4f2066b9a10ae37b779db5218a0003a888 on the
shared RTX4090 in default fast and explicit strict modes. The shared Rust
comparator reports 0/145 differing stored words in each mode. Two consecutive
advances check all carried volume arrays, reset of interval buffers seeded32,
a1m3 interval beside a cumulative2^60 total, immediate same-stream consumers,
empty mapping rows and a positive-volume rounding witness for ascending
surface-then-ground summation. Fixture SHA256:
536ec63908ecb843442f408d49226faccbab4d574b1beb27dc0e536ed41abd74.
This is analytical transfer evidence; no MC/reservoir binding or observed event
is implied. The owned real-DEM fast engineering workload with zero-width drains
measured4.886805ms per300s terrain period, or4.930376ms including the6742-node
producer. Separate off-path flux/update profiles and exact commands are in the
lane receipt; these shared-pool timings do not establish whole-hydro speed.
