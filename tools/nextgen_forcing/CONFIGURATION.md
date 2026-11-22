# Forcing selection in a NextGen realization

Composition belongs to `woof.hydro.config.configure(settings, *, table=None)`
and `woof/data/hydro/configurations.v1.json`. The composition selector is
`settings["realization"]`; forcing does not introduce another realization
table or selector. The forcing source selector is the forcing stage's
`options.source`, resolved by `woof.source_adapters.get_source_adapter`.
`woof.hydro.forcing.source_profile(source)` returns the packaged
`woof/data/hydro/forcing-sources.v1.json` path and that adapter row's
`hydro_forcing_profile`. New source mappings use these shared tables.

## Current source declarations

| Canonical init-source ID | Forcing profile | Admitted prepared input |
|---|---|---|
| `ngen-csv` (alias `aorc-csv`) | `ngen-catchment-csv` | Ordered catchment CSV files with the declared native ngen columns |
| `aorc` | `aorc-grid` | Local NetCDF, explicit area weights, hourly interval-end time coordinates |
| `nwm-retrospective-forcing` (alias `nwm-forcing`) | `decoded-canonical-grid` | Local NetCDF containing all eight canonical fields |
| `gfs`, `hrrr`, `era5` | `decoded-canonical-grid` | Local NetCDF containing all eight canonical fields |

These are declarations for prepared input, not raw GRIB admission or a promise
that arbitrary upstream files already supply the required fields. The raw GRIB
connection remains incomplete. Local AORC Zarr subset transport is documented
in [ZARR-SUBSET.md](ZARR-SUBSET.md); geographic catchment sampling/weights are
separate. AORC uses `APCP_surface` interval amounts divided by the interval
duration and `time_offset_seconds=-3600`; other meteorological values are held
over that preceding hour. Other packaged profiles use interval-start times and
rain rates. All profiles use specific humidity and earth-relative winds.

## Bind the selected source at setup

For a realization that selects an external source, set `options.source` on its
`woof.hydro.forcing_inputs:prepare_series` stage. This existing factory checks
the packet's **canonical source ID, profile row, and complete profile-table
SHA-256** against the selected source before Rust packet loading or GPU upload.
Checking only the profile row is insufficient: several sources share
`decoded-canonical-grid`. Changing table provenance also changes this identity;
prepare a fresh packet rather than suppressing the mismatch.

Prepare the packet with `prepare_external` and request `source`, so the helper
records the canonical source ID in the Rust request and manifest. Alternatively
an explicit Rust request can carry `source_profile`, `profile_row`, and
`source_id`. A packet prepared with only the profile path/row has null
`source_id` and will fail a source-selected factory check.

If `options.source` is omitted, the factory uses packet provenance without
comparing it to the shared registry. This supports explicit custom profiles;
it is not source validation against the realization. The factory does **not**
infer a source from top-level `settings` or the atmosphere's init source. The
configuration owner must pass its declared forcing selection into this stage
option. A coupled atmosphere and a standalone forcing series may have different
sources; no implicit selection is made.

At config-owner revision `6ddb681c709e48909c2d16b0663d849d4ab4c590`,
`ngen-shipped-noah-pet-cfe-48h` already passes `source="ngen-csv"` and the
explicit December 2015 epoch to both forcing stages, using float64 for PET and
float32 for Noah-OWP. Its PET rows follow `cat-67`, `cat-27`, `cat-52` with
methods 1, 5, 3. This is the shipped-example composition, not a case(b)/(c)
weather or initialized-physical-domain substitute. The separate five-method
native PET replay retains its own evidence and is not implied by those three
configuration rows.

Factory options actually consumed:

| Option | Meaning |
|---|---|
| `manifest` | Required Rust-prepared packet manifest path |
| `library` | Optional setup Rust bridge path; otherwise normal bridge discovery |
| `native_precision` | Default false: upload float32; true: upload the prepared float64 packet |
| `start_seconds` | Default 0; simulation clock offset from the packet's explicit UTC origin |
| `source` | Optional shared init-source selection, with the checks above |
| `time_origin` | Optional exact epoch-string check against the component identity |

The factory is standalone-only. Packet metadata supplies interval duration and
ordered catchment IDs; there is no factory `interval_seconds` override. The
epoch check currently occurs after loading; source checks occur before loading.
Unknown option keys are not rejected by this factory, so the configuration
owner should admit only these keys rather than treat ignored keys as supported.
No `source_profile` or `profile_row` stage override is implemented. Those belong
to preparation requests, not resident factory options.

Rust records input hashes and the area-weight hash in the manifest, checks the
actual times/cadence, transforms and remaps, and emits both packet precisions.
The loader checks selected packet bytes through Rust; restart identity carries
the selected packet hash, ordered IDs, source/profile/table hash, weight hash,
origin and clock semantics. Source validation is setup metadata work. Each
`advance(dt_seconds, *, stream=None)` enqueues the resident kernel only; it
performs no source resolution, allocation, synchronization or host readback.
The provider averages over the requested interval and advances once before its
consumers on the same stream. Module, routing and forcing cadences remain
distinct; a configuration row must not make a physical hourly runoff module
advance at a shorter routing cadence without its owner's transfer contract.

## PET stage

`woof.hydro.pet:prepare` consumes `forcing_component` naming an earlier prepared
component. Supply either a complete twelve-scalar `parameters` mapping and
`method` (default 5), or `catchment_parameters` rows containing `catchment_id`,
`method`, and complete `parameters` in the forcing packet's exact catchment
order. Mixing these forms is rejected. Integer methods 1 through 5 are
implemented. The twelve keys are the actual `woof.hydro.pet.PARAMETER_NAMES`:

```text
wind_speed_measurement_height_m
humidity_measurement_height_m
vegetation_height_m
zero_plane_displacement_height_m
momentum_transfer_roughness_length_m
heat_transfer_roughness_length_m
surface_longwave_emissivity
surface_shortwave_albedo
canopy_resistance_sec_per_m
water_temperature_C
ground_heat_flux_W_per_sq_m
surface_skin_temperature_C
```

PET accepts the seven non-rain canonical forcing vectors in one common float32
or float64 dtype. Parameters and `potential_et_m_s` output are float64; the
method vector is int32. Setup broadcasts configured scalar rows on the GPU;
CUDA owns parameter mutation and all per-step numerics. Do not add PET to the
existing coupled WOOF-ArWen Noah-MP exchange, which already supplies ET/runoff.

## Provenance and admission scope

Forcing definitions follow NOAA-OWP/ngen
`be11a180565a78ff069d855b0b5dbfc23845e7d8` (`AorcForcing.hpp`); PET follows
NOAA-OWP/evapotranspiration `6c410f2a8bc843043ad25e13239928031b8c5e64`.
Their actual pinned LICENSE notices were read: “Apache License”; “Version 2.0,
January 2004”. Full notices remain with the implementation. This describes
software provenance. Packet hashes and a source-table declaration grant no
dataset rights or physical-case acceptance. Keep weather, geometry, derived
weights and initialization permissions with their preparation receipts.
Case(c)'s v2.2 fabric hold and case(b)'s unresolved source-specific dataset
terms remain separate from this generic configuration contract.
