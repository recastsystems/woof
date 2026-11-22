# Arbitrary but verified GRIB adapters

`woof adapt` is the user-facing front door for a GRIB2 product that does not
have a named WOOF-ArWen adapter. It combines three authorities:

1. a WPS Vtable, used only for exact numeric GRIB selector import;
2. an explicit `rw-wps.descriptor.v1`, which owns field meaning, units, axes,
   staggering, pressure levels, target requirements, and soil policy; and
3. the user's actual GRIB2 files, inventoried before any adapter output is
   published.

Nothing is inferred from a product name or filename. Passing the battery makes
the resulting mapped adapter **runnable, not stock-WRF certified**.

Passing the battery also does not make the descriptor physically correct. What
`adapt` validates for you, what it trusts from your declaration -- units,
absolute geolocation, cell registration, level sufficiency, time semantics,
land-mask polarity, soil depth labels -- and a self-check you can run for each
trusted row, are in
[adapt-validation-contract.md](adapt-validation-contract.md). Read it while you
write the descriptor.

## Start with a skeleton

Generate a review-required descriptor rather than writing JSON without a
schema-shaped example:

```bash
woof adapt \
  --vtable /data/Vtable.PRODUCT \
  --skeleton /case/product.descriptor.json
```

The generator fills selector references only where a Vtable description
exactly matches a canonical field name. Unresolved selectors retain a
`REPLACE_WITH_*` marker, `model_top_pa` is a marker, and the pressure-level
list is empty. The scaffold therefore cannot accidentally pass as a completed
adapter. Review every selector and unit, declare the complete pressure
inventory, and compare with the worked
`configs/rw-wps-gfs-pressure-grib2.descriptor.json` example.

The command **prints the list of values it left for you**, and
`--descriptor` refuses the unfilled scaffold against that same list, so
what you are told to do and what you are held to cannot drift:

```
adapt skeleton: wrote /case/product.descriptor.json; 4 value(s) need
you before --descriptor will author it:
  adapt.model_top_pa
  coordinates.vertical.levels
  name
  target.name
```

`name` and `target.name` are on that list because nothing else would
catch them: they are ordinary strings, every validator accepts them,
and a bundle published as `REPLACE_WITH_ADAPTER_NAME` is a real adapter
with a placeholder identity.

The descriptor's adapt-only policy is removed from the executable mapping but
is SHA-256-bound through provenance:

```json
{
  "adapt": {
    "model_top_pa": 10000,
    "soil_policy": {
      "kind": "identity_complete_layers"
    }
  }
}
```

`identity_complete_layers` requires contiguous bounded soil selectors from
the surface down and uses those exact layers as the target. A declared
`conservative_layer_means` policy instead requires `target_layers_m`, for
example `[[0.0, 0.1], [0.1, 0.4], [0.4, 1.0], [1.0, 2.0]]`; every target
layer must be completely covered by the source. There is no silent layer
copy, fill, or synthesis policy.

## Verify and author

Pass every real file needed to make each valid-time frame complete:

```bash
woof adapt \
  --vtable /data/Vtable.PRODUCT \
  --descriptor /case/product.descriptor.json \
  --input /data/product-f000.grib2 \
  --input /data/product-f003.grib2 \
  --output-dir /case/product-adapter
```

WOOF-ArWen uses its vendored GRIB2 inventory and dump tools by default. Installed
or development environments may explicitly provide both exact executables
with `--grib2-inventory` and `--grib2-dump`; supplying only one is refused.

Success creates, without overwriting:

| Artifact | Purpose |
|---|---|
| `adapter.mapping.json` | Executable `rw-wps.mapping.v1` |
| `adapter.composition.json` | Executable soil and in-band-terrain composition |
| `adapter.provenance.json` | Battery result; descriptor, Vtable, input, mapping, and composition SHA-256 bindings; accurate status |
| `adapter.inputs.json` | Runtime manifest binding inputs, the authority triple, and exact decoder binaries |

The mapping, composition, and provenance JSON files are the authority triple.
The manifest makes that triple runnable through the existing `mapped` source
route. Preserve all four files together with the input GRIB files; changing
any bound byte invalidates the manifest. The command's JSON result includes
`runtime_bindings` with the exact primary inputs, repeated in-band-terrain
supplement role, provenance role, and decoder paths needed by that route; the
normal WPS namelist, geography, experiment, and output controls remain the
run's separate authorities.

## Acceptance battery

The command fails closed, before publication, on every check:

| Check | What passes |
|---|---|
| Compiled selectors and record inventory | Every direct scalar field occurs exactly once at every valid time; every pressure field has exactly the descriptor's full level set; every soil selector occurs exactly once; members do not mix |
| Units, axes, and staggering | Each target-required field exactly matches the descriptor's canonical target units, axes, and location and is unstaggered; the existing decoder then unpacks the selected real messages and executes their unit transforms and axis binding |
| Soil layers | Temperature and moisture bind the same ordered, contiguous bounded layers; the declared identity or conservative policy has complete source coverage; a missing land soil value is rejected as a donor (or first repaired by the declared bounded column repair), and only a source with no soil value on any land cell is refused |
| Grid family | Every selected record uses one shared regular latitude/longitude GDT 0 grid and scan mode `0x40` |
| Vertical coverage | The lowest-pressure declared and observed source level is at or above `adapt.model_top_pa` |
| Time series and identity | Valid times do not mix members; a multi-time series is uniformly spaced at the descriptor's boundary interval; every input and decoder is byte/SHA-256 bound |

Refusals name the failed capability and the missing value. Examples include:

```text
record-inventory check failed ... field 'surface_pressure' missing selector ...
grid-family check failed ... uses GDT 30; generic adapt supports regular latitude/longitude GDT 0 only. Use the named-adapter path ...
soil-layer check failed: gap between source layers 0 and 1: 0.1 m to 0.2 m
vertical-coverage check failed: source top 10000 Pa does not cover model top 5000 Pa
```

The v1.1 front door is deliberately GRIB2-only. GRIB1/NetCDF mapped contracts
remain available through the lower-level mapped authoring path, but do not
pass through this GDT-checked command. Curvilinear, projected, or normalized
scan-order products need a named adapter whose decoder explicitly implements
and verifies that capability; there is no generic bypass.

## Runnable is not certified

The emitted machine status is:

```text
runnable_mapping_not_stock_wrf_certified
```

Runnable means the declared mapping and composition are executable by
WOOF-ArWen's mapped engine and this exact GRIB inventory passed the ingest battery.
It does not mean the descriptor is a correct physical interpretation of the
files ([adapt-validation-contract.md](adapt-validation-contract.md)). It does
not mean WOOF-ArWen produced equivalent `wrfinput`/`wrfbdy` files to an
unchanged stock WRF toolchain, nor that stock WRF accepted and integrated
them. Stock-WRF certification is a separate gate keyed to exact retained
mapping, composition, source, geometry, and evidence hashes. A newly authored
adapter cannot inherit the GFS, ERA5, HRRR, or any other named adapter's
certificate merely because it shares a decoder, grid family, or field list.
