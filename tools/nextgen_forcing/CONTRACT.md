# NextGen forcing and PET component contract

Forcing owner: hydro-forcing. Driver contract checked at
`15b52b6157be07baa3e2db594c9415d4de91997c`, 2026-10-18 UTC.
NOAA-OWP/ngen `be11a180565a78ff069d855b0b5dbfc23845e7d8` and
evapotranspiration `6c410f2a8bc843043ad25e13239928031b8c5e64` LICENSE:
“Apache License”, “Version 2.0, January 2004”. Complete upstream notices are
retained. These components use no withdrawn WRF-Hydro code, input or interface.

## Prepared assembly

| Factory | Purpose | Mode |
| --- | --- | --- |
| `woof.hydro.forcing_inputs:prepare_series` | Previously Rust-prepared forcing packet | standalone |
| `woof.hydro.forcing:prepare_parent` | Optional meteorological aggregation from explicit parent fields | coupled |
| `woof.hydro.pet:prepare` | PET consuming an earlier prepared forcing component | standalone composition, or an explicitly selected meteorological composition |

All factories accept keyword-only `settings, options, state, physics, device_id,
mode, components`. Bind `Stage(name, component, component.advance,
parallelism="CUDA, 128 threads/block; one thread/catchment")` once during
preparation. A standalone sequence places forcing before PET and runoff in
the ordered stages. Each launch is `advance(dt_seconds, *, stream=None)`:
one existing kernel, no synchronization, readback, allocation, compilation or
host numerical conversion. None selects the current stream. The caller owns
the current card and producer/consumer stream dependencies. No CUDA graphs.
Preparation leaves output fields uninitialized; enqueue the first forcing
launch before PET or any other consumer.

Noah-OWP factory snapshot `3561c32176c5d9c5e4a451c4d86d43dc3c98eefc`
accepts `forcing_component` in `woof.hydro.noahowp.factory:prepare_slab`.
It binds the earlier component's canonical **float32** `.fields`; its launch
copies U2D/V2D/T2D/PSFC/Q2D/LWDOWN/SWDOWN/RAINRATE into
UU/VV/SFCTMP/SFCPRS/Q2/LWDN/SOLDN/PRCPNONC on CUDA before numeric physics.
Q2 remains specific humidity. The hook copies the current sample; it does
not advance the forcing provider. Schedule one provider advance before the
consumer on the same stream. Native float64 PET/provider replay is a separate
explicit binding; do not narrow those fields in Python or supply them to this
float32 hook. Complete Rust-prepared Noah initial slabs/parameters and matching
catchment order remain prerequisites. This API inspection does not establish
a new physical case(b)/(c) initialization or a new numerical replay.

Existing WOOF Noah-MP already receives atmosphere forcing. Its NextGen
`noahowp.factory.prepare_exchange` runoff exchange needs no extra weather or
PET stage. Its signed-i64/i32/f32 CSR is distinct from this optional
meteorological helper's u64/u32/f64 CSR; preserve each owner's widths.

## Published fields

Forcing `.fields` is a stable mapping of contiguous same-card `[N]` views,
aliased into `.arrays["fields"]`, in hydrofabric/packet catchment order.
Default standalone and parent forcing are float32. Standalone
`native_precision=true` preserves float64; PET accepts either width without
narrowing. Output must retain the actual native dtype.

| Name | Units / meaning |
| --- | --- |
| T2D | K |
| Q2D | specific humidity, kg/kg |
| U2D, V2D | earth-relative m/s |
| PSFC | Pa |
| LWDOWN, SWDOWN | W/m2 |
| RAINRATE | mm/s |
| PET `potential_et_m_s` | float64 `[N]`, m/s |

PET publishes a rate. Its kernel does not use dt to turn the rate into a
step depth; any consumer conversion belongs in that consumer's CUDA kernel.
Status is resident int32, inspected only at an output/checkpoint boundary.
Forcing status1 means an invalid interval or input; PET status1 means invalid
forcing, status2 a nonfinite numerical result. Preserve diagnostics and status
as carriers, not substitute discharge/publication fields.

## Standalone packet and time

Setup doors are `forcing_inputs.prepare_external(request_path, *, library=None)`
and `load_series(manifest_path, *, library=None, device_id=0,
native_precision=False, start_seconds=0.0)`. Rust decodes, converts units,
regrids and hashes before GPU stepping. New source mappings are table rows.
`prepare_series` options are `manifest`, optional `library`, `native_precision`,
`start_seconds`, `time_origin` (must match packet), and `source` (checks the
shared init-source row and profile seal).

Frames are interval means at the packet's explicit UTC `time_origin` and
`interval_seconds`. CUDA duration-weights the window `[clock, clock+dt)` and
advances each resident clock. `start_seconds` is relative to that origin;
the driver continuation clock is separately relative to driver preparation.
Preserve both on resume. An invalid or uncovered window records status/NaN
and leaves the forcing clock unchanged. A covered window containing invalid
values instead publishes status/NaN and advances the clock to its end. Neither
error is an automatic retry boundary; restore the complete upstream/component
state before replaying a failed interval. Hour-aligned hourly native provider
replay is exact; historical native sub-hour quirks are not emulated.

All four standalone `.arrays` carriers:

| Key | Shape | Dtype / role |
| --- | --- | --- |
| frames | `[T,8,N]` | selected float32/float64 prepared source means |
| fields | `[8,N]` | same dtype, published interval means |
| clock_seconds | `[N]` | float64, next interval start relative to packet origin |
| status | `[N]` | int32 |

Identity names `arwen-duration-weighted-forcing-v1`, ngen source pin, numerics,
origin, cadence, shape, dtype, units and fields. Nested packet identity seals
catchment IDs, packet bytes, source profile/row/source ID and area weights.

## Optional parent meteorology and precipitation baseline

`prepare_parent` options require `hydrofabric_manifest`, `parent_fields`
(canonical names to dotted `state`/`physics`/earlier `components` paths),
`time_origin`, optional `humidity` and `rain_accumulators` paths.
`humidity="mixing_ratio"` converts each source cell `r/(1+r)` in CUDA before
area weighting; default is specific humidity. Winds must already be earth
relative. Physical parent projection-angle binding remains open.

With cumulative-mm precipitation, declare one or two `rain_accumulators` and
`rain_baseline="at_prepare"`. Setup copies the actual producer arrays as the
first baseline at the supplied epoch. The direct constructor
`PreparedCatchmentForcing` instead accepts explicit matching baseline grids.
Each launch differences the current total against the saved per-CSR-edge
baseline, divides by the actual dt and updates that baseline. Shared source
cells cannot race. A decreasing or nonfinite accumulator records an error but
still saves the current total as the next baseline. Do not retry the interval
against that changed baseline: recover the producer and baseline together, or
explicitly prepare a new baseline at a boundary. No forecast-start baseline is
guessed.

All parent `.arrays` carriers (14 direct-rate; 16 accumulated):

| Key | Shape | Dtype / role |
| --- | --- | --- |
| fields | `[8,N]` | float32 published values |
| status | `[N]` | int32 |
| area_offsets | `[N+1]` | uint64 |
| area_indices | `[E]` | uint32 |
| area_weights | `[E]` | float64 |
| previous_rain_mm | `[E]` accumulated, `[0]` direct | float64, saved total per edge |
| parent/T2D,Q2D,U2D,V2D,PSFC,LWDOWN,SWDOWN,RAINRATE | parent grid | eight float32 producer aliases |
| parent/rain_a, parent/rain_b | parent grid | accumulated only, float32 |

In accumulated mode parent/RAINRATE is an unused alias of parent/T2D;
parent/rain_b is a prepared zero grid when only one accumulator exists.
Identity `arwen-catchment-area-forcing-v2` seals fabric identity, declared
producer paths, epoch, grid shape, rain convention/count, humidity conversion,
sum order and numerics. Producer arrays and baseline must restore together.
Catchment ordering/overlap weights and physical geometry belong to hydrofabric.

## PET parameters, carriers and restart

`pet.prepare` requires `forcing_component` naming an earlier component and
either all twelve named `parameters` plus optional integer `method` (default5),
or ordered `catchment_parameters` rows with catchment_id/method/parameters.
The latter must exactly match the standalone forcing catchment IDs. No missing
physical parameters are inferred. `PARAMETER_NAMES` in `pet.py` is the table
column order; native mutable parameter updates are retained in place.

All twelve PET `.arrays` carriers:

| Key | Shape | Dtype / role |
| --- | --- | --- |
| forcing/T2D,Q2D,U2D,V2D,PSFC,LWDOWN,SWDOWN | `[N]` each | seven aliases, selected forcing dtype |
| parameters | `[N,12]` | float64, native mutable parameter table |
| methods | `[N]` | int32, native method1..5 |
| potential_et_m_s | `[N]` | float64, published rate |
| diagnostics | `[N,6]` | float64, internal native intermediates |
| status | `[N]` | int32 |

Identity `noaa-owp-pet-cuda-v2` names source pin, numerics, forcing dtype,
float64 arithmetic, maths implementation and corrected wind-norm expression;
nested identity seals forcing and complete parameter/method configuration.
Version1 checkpoints require fresh preparation under version2.

Every listed array is writable, pointer/shape/dtype stable and on one card.
Restore the complete inventory in place, including aliased inputs, mutable
parameters, precipitation baseline and forcing clock. These components have
no host continuation state and expose none of the optional three component
metadata hooks. Driver-owned steps/elapsed time are checkpointed by the driver.
Complete identity and inventory validation precedes destination writes; call
driver restore_metadata only after boundary copies complete. Alias duplication
in the inventory is intentional. Register separate stable history/cursor
buffers before the driver seals its inventory if output capture is selected.

## Timing and acceptance scope

Report setup separately from the completed launch batch. GPU event timings
bracket actual kernels on the caller's stream only in a separate profile, and
may include enqueue gaps; cold per-stage measurements are not additive warm
step costs. Name card, pool contention, numerics, active catchments and block
size. A forcing/PET microbenchmark does not establish whole-network10ms.

Existing measured native720h/three-catchment replay at numerical revision
`5e0598057fb46b5a15b3e1bc62aa8ed1dd6addba` is forcing0/17280 and PET0/2160
in both fast and strict. Initialized strict48h case(a) is0/864 stored words,
with five inspected Rust-rendered pictures. Receipts preserve their own pins,
commands, actual inputs and pooled4090 timings. They do not establish physical
parent/event/USGS acceptance, permitted public demo data or whole-network speed.
