# NextGen forcing inputs

This contract describes forcing owner
`370d635cb5f34a5015f2badf48604c5ac21c0786`, including source-grid wind rotation
`d21abe7ef3b133b5a761712e1b4982c4747e7dda`. It is a versioned implementation
snapshot; this documentation change does not import those runtime components
or qualify an installed package.
The resident interval conventions below also follow the owner's README
clarification `c89efe470cc39bf5fc77113db94ddbbf70331c0e`; it changes no runtime code.
Prepared factory and checkpoint details follow owner contract
`d84841227e364c9a5a575a8b466ea1532dc7754b`, checked against driver
`15b52b6157be07baa3e2db594c9415d4de91997c`. Its numerical sources retain the
PET replay revision `5e0598057fb46b5a15b3e1bc62aa8ed1dd6addba`.

## Standalone preparation

The setup command is `arwen-nextgen-forcing REQUEST.json`. It reads prepared
NetCDF grids or ordered catchment CSVs, converts declared quantities and
uses catchment area weights to produce typed forcing packets. A matching
Rust executable is required. This is preparation before simulation, not
another stage of the resident time step.

The request supplies `source_profile`, optional `profile_row`, ordered
`inputs`, `output`, `time_origin`, `interval_seconds` and ordered
`catchment_ids`. Gridded inputs also need `area_weights`; cumulative rain
needs an explicit `rain_baseline`. A source profile maps variables, units,
temporal meaning and orientation. Adding another compatible source is a
table row. A GFS, HRRR or ERA5 row consumes already decoded canonical grids;
it does not connect raw GRIB acquisition, decoding or missing-field completion.

Canonical packet fields are:

| Field | Quantity and units |
|---|---|
| `T2D` | Air temperature, K. |
| `Q2D` | Specific humidity, kg/kg. |
| `U2D`, `V2D` | Earth-east and earth-north wind, m/s. |
| `PSFC` | Surface pressure, Pa. |
| `LWDOWN`, `SWDOWN` | Downward longwave and shortwave radiation, W/m². |
| `RAINRATE` | Precipitation rate, mm/s. |

The profile defaults to specific humidity. Declaring
`"humidity": "mixing_ratio"` applies the declared field scale and offset,
then converts each source cell to specific humidity before area averaging.
Relative humidity and dewpoint conversion remain unconnected. An interval
precipitation total and a cumulative accumulator require different temporal
declarations; the preparer does not infer their meaning from a variable name.

Gridded times must match the continuous declared clock. The time coordinate
defaults to `time`; `time_variable` can select another coordinate and
`time_offset_seconds` can explicitly align interval-end samples. Reversed,
repeated or gapped times fail preparation. A multi-time weather variable must
carry its declared time dimension; its spatial axes must match the area
weights. Missing weather fields are not synthesized.

## AORC hourly windows

Forcing owner `cac14eb68797abd4dfe19fd55f24c877d6a09b88` corrects the
`aorc-grid` row to `time_offset_seconds = -3600`. AORC hourly precipitation
totals end at the source timestamp. To prepare `[start,end)`, supply source
stamps from `start + 1 hour` through `end`, including that final endpoint.
For example, `[00:00,02:00)` needs the 01:00 and 02:00 samples. A derived
start-aligned dataset needs its own explicit profile row.

Co-stamped nonprecipitation endpoint samples are held over the preceding
hour. This policy does not make them observed hourly means; a native
realization must select the same sampling policy before provider equivalence
can be claimed.

The setup API is `woof.hydro.forcing_inputs.prepare_external(request_path,
library=...)`. Select `source="aorc"`, ordered local NetCDF `inputs`,
`output`, UTC `time_origin`, `interval_seconds=3600`, ordered `catchment_ids`
and `area_weights`. Weights must index the exact local grid's flattened
spatial order and preserve catchment order, coordinate bounds and mask
coverage. Preparation then produces the packet consumed by `prepare_series`.

The inspected NOAA 2023 archive is Zarr with packed int16 weather fields.
Direct cloud subset/decode into this local NetCDF/CSV door remains unfinished;
fetching archive metadata does not supply event forcing. A Rust decoder must
preserve the actual scales, offsets, missing values and axis order. NOAA
forcing permissions remain separate from hydrofabric data rights.

Forcing revision `1ef318567cade4b3209a20d29f00860c73a7f46f` adds the optional
setup tool `forcing-zarr-coordinates REQUEST.json`. It decodes sealed local
coordinate chunks in Rust, preserving float64 values and source ordering;
it does not extract weather values or construct area weights. The inspected
2023 archive has ascending latitude/longitude axes of 4,201/8,401 centres and
C-order `[latitude,longitude]` weather grids. These geographic axes cannot
stand in for the north-first projected terrain grid.

Optional midpoint bounds are explicitly inferred, not provider-published.
The consumer must establish the spatial reference, cell areas, sampling and
exact subset ordering before preparing geographic weights. For the January
5–17 window, the coordinate export distinguishes 288 hourly forcing endpoints
from 289 inclusive discharge sample points. This coordinate export supplies
neither an event weather packet nor native provider equivalence.

## Source-grid wind orientation

Earth-relative wind is the default. For grid-relative source vectors, add
this object to the source profile:

```json
{
  "wind": {
    "frame": "grid_relative",
    "angle": {"variable": "grid_east_angle", "scale": 0.017453292519943295}
  }
}
```

The angle is positive counterclockwise from earth east to grid east. Its
declared scale and offset must produce radians; the example scale accepts
degrees. Rust rotates each cell into earth east/north **before** catchment
averaging. Averaging the original vectors and their angles first is incorrect
when grid orientation varies within a catchment.

The angle may be a static spatial grid or use the weather time dimension in
any supported axis position. A catchment CSV may instead name an angle column.
Angles are instantaneous, using `temporal = "rate"`, the default for a field
mapping. Missing angles, invalid shapes and unsupported frames fail preparation.
The profile and input hashes record the declared orientation. This option
does not derive angles from a projection or bind a WOOF-ArWen parent-grid provider.
Existing earth-relative rows retain their packet bytes.

## Resident and coupled use

The prepared-series factory consumes a packet manifest and preserves native
float64 forcing where requested. It uploads once and advances on the member's
card. Cross-step state, including clocks and precipitation baselines, belongs
to the checkpoint inventory. Numerical transformations and interval processing
remain in Rust or CUDA; no file reading or host numerical work occurs per step.

Preparation leaves published forcing fields uninitialized. Enqueue the first
`advance(dt_seconds, stream=...)` before a consumer, on the same stream or with
an explicit producer dependency. The standalone series returns interval means
over `[clock_seconds, clock_seconds + dt_seconds)`, then advances its resident
clock. Its UTC origin, cadence and starting offset must describe the intended
window; these are not interpolated endpoint samples.

For accumulated parent rain, the saved cumulative-mm baseline and its epoch
must describe the same producer state. The kernel differences the current
total against that baseline and divides by the actual elapsed LSM interval.
`rain_baseline="at_prepare"` samples the producer at preparation; it does not
infer a forecast-start baseline. Preserve those arrays and their identity on
restart, and aggregate after the rain producer completes its interval.

The existing coupled Noah-MP land-surface exchange supplies runoff and ET
directly. It needs neither an external weather packet nor an additional PET
stage. An optional parent meteorological helper is a separate contract: its
winds must already be earth-relative, and it needs actual parent geometry,
area weights and precipitation accumulator semantics. Source-file rotation
does not make a physical parent binding available. See the
[coupling requirements](nextgen-coupling.md) and the initialized row's
[prepared-resource bindings](nextgen-configurations.md#initialized-noahpetcfe-row).

## Prepared factories and state

These factories accept keyword-only `settings`, `options`, `state`, `physics`,
`device_id`, `mode` and earlier `components`. Bind their
`advance(dt_seconds, stream=...)` callbacks during preparation, with forcing
before its consumers. Each callback enqueues one kernel with 128 threads per
block, one thread per catchment. The caller supplies the card and stream
dependencies; `stream=None` uses the current stream.

| Factory | Required options and setup semantics |
|---|---|
| `woof.hydro.forcing_inputs:prepare_series` | Standalone `manifest`; optional `library`, `native_precision`, `start_seconds`, `time_origin` and `source`. The origin must match the packet; `source` checks the init-source row and profile seal. |
| `woof.hydro.forcing:prepare_parent` | Coupled `hydrofabric_manifest`, `parent_fields` and `time_origin`. Field paths resolve through `state`, `physics` or earlier `components`. Optional `humidity="mixing_ratio"` converts each cell before area weighting. One or two `rain_accumulators` require `rain_baseline="at_prepare"`. |
| `woof.hydro.pet:prepare` | Earlier `forcing_component`; either all twelve named `parameters` with optional `method` (default 5), or `catchment_parameters` rows with IDs, methods and parameters in exact packet order. `PARAMETER_NAMES` in `pet.py` defines column order; methods are integers 1–5. |

Standalone `start_seconds` is relative to the packet's UTC origin. Its resident
forcing clock and the driver's elapsed time since preparation are separate
state and both must survive resume. An uncovered window, nonpositive interval
or negative start produces status 1 and NaNs without advancing the forcing
clock. Invalid values within a covered window produce status 1 **and advance
the clock**. Inspect status at an output/checkpoint boundary before using the
result; retrying the same call is not a recovery protocol.

Forcing fields are contiguous `[N]` aliases of `fields[8,N]`, in packet or
hydrofabric order. Standalone defaults to float32;
`native_precision=true` preserves float64 through PET. Parent forcing uses
float32. PET publishes `potential_et_m_s[N]` as float64 **m/s**; its kernel
does not multiply by the step duration. PET status 1 denotes invalid forcing,
status 2 a nonfinite numerical result.

The checkpoint inventory includes inputs and aliases as well as outputs:

| Component | Resident carriers |
|---|---|
| Standalone forcing: 4 | `frames[T,8,N]` and `fields[8,N]` in the selected float width; `clock_seconds[N]` float64; `status[N]` int32. |
| Parent forcing: 14 direct-rate, 16 accumulated | `fields[8,N]` float32; `status[N]` int32; CSR `area_offsets[N+1]` uint64, `area_indices[E]` uint32 and `area_weights[E]` float64; `previous_rain_mm[E]` float64 (empty for direct rates); eight float32 `parent/` field aliases, plus `parent/rain_a` and `parent/rain_b` in accumulated mode. |
| PET: 12 | Seven `forcing/` aliases in the selected forcing width; mutable `parameters[N,12]` float64; `methods[N]` int32; `potential_et_m_s[N]` float64; `diagnostics[N,6]` float64; `status[N]` int32. |

Here `N` is catchments, `T` source intervals and `E` CSR edges. The parent
meteorological CSR uses uint64/uint32/float64; the existing Noah runoff
exchange has its own signed-int64/int32/float32 CSR contract. Preserve each
component's widths. Restore the complete stable inventory in place, including
parent producers and their precipitation baseline together. Even a decreasing
accumulator updates that baseline while marking the output invalid.

Validate identities `arwen-duration-weighted-forcing-v1`,
`arwen-catchment-area-forcing-v2` and `noaa-owp-pet-cuda-v2` before restore.
These components expose no host continuation hooks. Driver metadata owns its
step count and elapsed time; register separate history/cursor buffers before
the driver seals its inventory. See [checkpoint behavior](nextgen-output.md).

## Evidence limits

The AORC row correction passes four analytic Rust clock/amount cases, including
rejection of wrongly aligned or missing-first-hour inputs. Its hour-ending
packet matches the start-aligned reference in both float32 and float64 bytes.
The owner's eight-worker Rust preparation took 0.585640 ms on that small
fixture. This is input-transport evidence, with no event weather payload,
native physics replay, GPU run or new picture.

The owner's current-driver ABI check used two forcing components (float64 and
float32) plus PET, three catchments and 20 carriers. Across two hourly steps,
carrier pointers, shapes, dtypes and field aliases remained stable and driver
metadata reached 7,200 seconds. On a pooled RTX 5070 Ti, preparation took
940.529615 ms and the completed two-step batch took 0.076074 ms: six kernels,
one 128-thread block per stage, two boundary waits, zero timing events or CUDA
graphs. This checks the launch/inventory contract. It supplies no new native
comparison, serialization/restore proof, physical picture or isolated-card
speed measurement.

The owner records 14 analytic input-transport checks each for NetCDF-4 and
classic NetCDF, including varying orientations, static angles, time-axis
placement, CSV scaling and invalid inputs. Those checks compare exact typed
packet hashes; they are not native physics replay. Repreparing the three
catchments' 720-hour earth-relative inputs leaves both packet hashes unchanged.

PET owner `5e0598057fb46b5a15b3e1bc62aa8ed1dd6addba` corrects wind-norm
rounding that previously caused a 2 ULP difference at catchment 52, hour 674.
The owner's fresh 720-hour, three-catchment replay has zero differences in
17,280 forcing words and 2,160 PET words in both fast and strict modes;
initial and final mutable parameters also match. The numerical replay used
the corrected CUDA before a metadata-only identity update. The committed
identity is `noaa-owp-pet-cuda-v2`, with an explicit wind-norm identifier.
Older PET checkpoints require fresh preparation under that identity;
preserve checkpoint identity validation.

The owner also ran the committed version in the initialized strict 48-hour
Noah/PET/CFE composition. Its six captured fields match all 864 stored native
words; five Rust-rendered plots show coincident native/model curves. This
supersedes the earlier PET replay blocker at the named source pin. It does
not import that implementation through this documentation change.

Driver `15b52b6157be07baa3e2db594c9415d4de91997c` also proves
[PET v2 fast continuation](nextgen-output.md#pet-v2-fast-continuation):
24 hours plus a fresh-process restore for 24 more matches the uninterrupted
48-hour checkpoint and history. Its fast history still differs from native
in 88 of 864 words; PET itself matches all 144 words. The original
[strict continuation](nextgen-output.md#initialized-standalone-continuation)
retains its earlier source pin. Input transport, PET replay and
short standalone results do not establish physical event skill, parent
coupling, complete hydro-step speed or public-demo data admission.
