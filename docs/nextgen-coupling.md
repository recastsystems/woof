# NextGen coupling and continuation

The runtime attaches one resident hydro driver to each forecast member's root
physics driver. The current attachment requires `grid_id = 1` and WOOF-ArWen
Noah-MP, selected with `sf_surface_physics = 4`. Moving child nests keep their
own atmospheric state and do not acquire another hydro state.

The `go --hydro` command selects hydro settings. A working coupled realization
also needs physical inputs and prepared component factories. The initialized
three-catchment standalone row and the prescribed-inflow Lower Colorado
routing row are separate development compositions; neither is a demonstrated
parent-atmosphere configuration. See their [measured scopes](nextgen-development-evidence.md).

## Setup and the parent land-surface interval

Before stepping, the realization prepares its forcing or parent exchange,
catchment state, network, reservoirs and diagnostic buffers on the parent's
card. Its ordered stage rows select factories and the phases described in the
[configuration contract](nextgen-configurations.md#runtime-stages-and-evidence).
The resident driver requires at least one stage after the land-surface call;
an empty descriptive baseline cannot advance a simulation.
The [prepared routing reader](nextgen-configurations.md#loading-prepared-routing-resources)
verifies payloads and the network roster at setup. Its availability does not
supply the physical inputs described below.

On each actual parent land-surface interval, the physics loop performs:

1. `hydro.before_lsm(stream=...)` enqueues the prepared stages before the LSM.
2. WOOF-ArWen advances its parent Noah-MP land surface.
3. `hydro.advance(dt_seconds, stream=...)` enqueues the prepared exchange and
   hydro stages after the LSM.

The hydro interval is the parent land-surface cadence, `physics.bldt_seconds`.
The driver rejects an explicit advance with a different duration or without
its matching `before_lsm` call. Standalone execution uses the same pair, with
its runoff modules in the prepared hydro sequence. It does not call the
atmospheric land surface between them.

Component launches receive the interval in seconds and the parent CUDA stream.
They use arrays, kernel arguments and ordering bound during setup. Numerical
processing, unit conversion and exchanges remain on the GPU; file I/O and
diagnostic publication run at output boundaries. The launch sequence uses
ordinary kernels and no CUDA graphs.

## Geometry and units required for a physical run

Catchment exchange requires the executed parent grid definition, projected
coordinate system, cell boundaries, array orientation and prepared overlap
weights with the same catchment ordering. Latitude/longitude cell centres
alone do not define these resources. A source-polygon or source-flowline
manifest is not a prepared resident network or overlap packet.

Conversions must follow the selected module's contract. Parent runoff or
drainage accumulated over a land-surface interval, a flux per second, stored
water depth and reach discharge have different units. Bottom drainage is
not interchangeable with total subsurface runoff. The realization must bind
the actual exchange and its soil, aquifer and initial-state resources before
it can claim physical coupling. Terrain and channel inputs also need an
explicit source partition so the same water is not routed twice.

The setup-only overlap exporter at hydrofabric revision
`dae3b50d0887992cb992aba573eb28194ea74b7a` connects 6,971 catchments to
the 737 × 582, north-first, 1 km EPSG:5070 terrain grid. Its raw intersections
preserve overlapping source polygons, including cells with coverage above
one. Positive overlap does not establish a DEM-valid active cell or a drain
receiver. Do not normalize those overlaps silently, treat them as the executed
parent grid's weights, or register the exporter as a resident time-step stage.
The selected physical realization still needs its overlap and water-budget
policy, producer inputs and receiving controls.

## Preparing terrain overlap

The integrated preprocessing door at `f02ed5fa833f3c0d31da943723249bed356aed73`
exposes the same Rust transpose as a setup command:

```sh
woof hydro prepare --cell-overlap CONFIG.json
```

This request consumes existing prepared geometry. Its JSON fields are:

| Field | Required input |
|---|---|
| `fabric_manifest` | Prepared hydrofabric manifest with the original catchment/grid intersections. |
| `fabric_sha256` | SHA-256 of that manifest. |
| `destination_manifest` | Prepared terrain-grid manifest. |
| `destination_sha256` | SHA-256 of that manifest. |
| `output` | Output root for the derived overlap packet. |

Rust checks the pinned manifests and exact grid, CRS and row orientation before
writing. The result uses `arwen.nextgen.catchment-cell-overlap.v1`; its sparse
rows follow flat C-order terrain cells and retain the source catchment axis.
Weights are intersection area divided by the full cell area. Coverage above
one is preserved, and the positive-overlap mask is not a DEM-valid or physical
active mask. This command does not acquire or clear data rights, initialize
soil or aquifer state, bind the executed atmospheric parent, or advance hydro.

The integration owner's setup comparison reports ten fields, 16,441,816
bytes and zero differing bytes against the retained intersections, using
24 Rust Rayon workers. This is geometric transport evidence; the physical
producer and water-budget requirements above still apply.

## Parent exchange state and diagnostics

At driver revision `15b52b6157be07baa3e2db594c9415d4de91997c`, the exchange
returns `PreparedCatchmentExchange.advance(dt_seconds, *, stream=None)`, ordered
after the actual parent LSM and before routing. Parent
fields are contiguous float32 arrays on the same card, with soil layers first;
the sparse weights use int64 offsets, int32 indices and float32 weights.
Surface head remains persistent storage in millimetres. This exchange does
not demonstrate reinfiltration or two-way soil feedback.

Retain its complete `.arrays` mapping in checkpoints, including aliased
fields such as `infxsrt` and `runsrf`. Restoring only published runoff fluxes
loses producer and soil state. Its identity includes the loaded fabric and
payload identities; setup verifies static payload hashes in Rust.

The resident int32 `status` diagnoses malformed weights, indices and
normalization. Consumers must inspect it at output or checkpoint boundaries.
It does not cover every nonfinite physical input or output, and the current
writer does not provide generic automatic fault rejection. Qualify those
boundary checks and restore complete producer/component state before retrying.
Noah's total subsurface runoff includes bottom drainage: adding that drainage
again as a disjoint recharge source would count the same water twice.

## Checkpoints and output boundaries

Every prepared component exposes a stable `.arrays` mapping of its cross-step
CUDA state and a JSON `.restart_identity`. Carriers belong to the parent card;
their shape, dtype and pointers remain fixed. The driver records the ordered
composition, numerics, cadence and component identities. Diagnostic history
and its cursor must be registered before the inventory is sealed.

A checkpoint is taken after the matching advance completes, rather than
between the pre-LSM and post-LSM sequences. Components with continuation
metadata implement all three hooks: `checkpoint_metadata()`,
`validate_checkpoint(metadata)` and `restore_metadata(metadata)`. These hooks
run at checkpoint boundaries. The reader validates identities, the complete
carrier inventory, shapes, dtypes and continuation metadata before restoring
device arrays in place.

In a member-scoped run, newer driver versions include the original ensemble
member ID in hydro identity and reject another member's checkpoint before
destination writes. Ordinary unscoped identity remains unchanged. See the
[member-binding versions and measured scope](nextgen-ensemble.md#driver-identity-and-checkpoint-restoration).

The standalone door accepts `--restart FILE`; its selected writer owns the
supported checkpoint format and output layout. A newer driver has measured
initialized standalone Noah/PET/CFE continuation: 24 hours, a fresh process
restoring `restart.nc`, then another 24 hours match an uninterrupted 48-hour
run across all 466 declared carriers. Its `history.nc` remains diagnostic.
The separate 148-array routing continuation and repeated-output recovery also
have measured scopes. See the [exact versions, output paths and continuation
scope](nextgen-output.md). These separate results do not establish combined
runoff/routing/terrain or parent-atmosphere continuation, qualify every
composition or qualify an installed product. The demo preparer independently
creates a unique run directory.
