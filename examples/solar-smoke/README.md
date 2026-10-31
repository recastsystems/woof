# Smoke and solar irradiance

`northeast.toml` is a 180 x 140 x 49, 3 km, 24-hour configuration starting
2023-06-07 00 UTC. It uses transported NOAA smoke and legacy RRTMG with a
12-second timestep and 15-minute radiation clock. The file has feedback OFF.

```sh
woof go examples/solar-smoke/northeast.toml --smoke-radiation on --outdir smoke-on --products none
woof go examples/solar-smoke/northeast.toml --smoke-radiation off --outdir smoke-off --products none
woof solar fetch-observations --network surfrad --prefix psu --directory Penn_State_PA --date 2023-06-07 --out observations
woof solar products --wrfout FRAME --out solar --products solar_ghi,solar_dni,solar_dhi
```

Replace `FRAME` with a native history file, or provide a chronological list.
For station verification and multiple frames, use the request format in
[solar products](../../docs/solar-products.md). GHI is `SWDOWN`, DNI is
`SWDDNI`, DHI is `SWDDIF`, and direct horizontal irradiance is `SWDDIR`.

Initial and boundary smoke supply the historical plume. The fire source reports
unavailable historical emissions explicitly; it does not synthesize missing
fires. The ON option changes aerosol ownership as documented and preserves
transport. These settings do not reproduce the operational model's own
shortwave-only feedback recipe.

Inputs are public NOAA data, with provenance in the acquisition receipts.
See [source licences](../../docs/solar-products.md#sources-and-licences).

The completed 24-hour example reduced Penn State SURFRAD daylight GHI RMSE
from 124.06 to 105.49 W/m2 over 14 matched hours. On one shared RTX 4090,
engine wall time increased from 379.88 to 395.06 s with feedback enabled.
Both runs still overpredicted irradiance and substantially underpredicted
GOES total AOD. See the [measured example](../../docs/solar-products.md#measured-example)
for the comparison limits, identity checks and radiation cost.
