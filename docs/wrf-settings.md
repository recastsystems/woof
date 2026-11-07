# Combined WRF settings

To select moist potential temperature, SMS-3DTKE turbulence and averaged
acoustic fluxes together, add these values to the ordinary experiment:

```toml
[shared]
use_theta_m = 1
km_opt = 5
diff_opt = 2
bl_pbl_physics = 0
sf_sfclay_physics = 1
do_avgflx_em = 1
```

Run `woof go experiment.toml`. Python `RunConfig` accepts the same names.
WRF namelists put the first three dynamics settings and `do_avgflx_em` in
`&dynamics`, and the two physics selectors in `&physics`. Per-domain WRF
values must include each domain to enable it there. The support report and
namelist import use the same admission checks.

The engine defaults remain dry theta and no averaged fluxes. Fast numerics
remain the default. An imported WRF namelist with no thermal selector uses
the selected WRF line's Registry default, which the import reports: WRF 4
selects moist theta and WRF 3 selects dry theta. This file convention does
not change the engine's own configuration default.

`run --wrfinput` keeps moist `THM` and the moist boundary thermal field when
both the file and requested thermal selector are 1. History `T` remains
dry theta minus 300 K; `THM` carries moist theta minus 300 K. Physics receives
dry theta. Turbulence diagnostics also use dry theta, while its implicit
thermal transport carries the selected prognostic.

Engine checkpoints retain the selected prognostic, turbulence state and
averaged-flux interval. Stock WRF restart-file import with averaged fluxes
is refused because that door does not restore the native history alarm.
The existing native-file continuation without averaged fluxes completes
but has documented numerical differences from native, even with strict
numerics. Physical multi-device execution has not been measured.

`woof.capabilities.features()` includes `wrf-settings-v1` for workers that
need to admit this combined configuration. It describes installed
implementation support; it does not bypass physics pairing or resource
checks and is not a universal bitwise-equivalence claim.

With strict numerics and moist theta, Noah's 2 m vapor diagnostic retains
WRF's original flux-inversion word, including a possible negative value.
Fast numerics and the established dry route retain the existing fallback
to lowest-level vapor when that diagnostic is nonpositive. This diagnostic
does not replace the model's vapor state.

Combined validation at numerical commit `2a9cee64f86` matches all 189 common
floating history fields, including AVGFLX, at seven times over one hour
against strict WRF 4.6.1. The parent is 90 by 81 by 39 mass cells at 10 km
and 30 seconds; the child is 111 by 96 by 39 at one-third spacing and a
10-second step, with feedback. Native uses four MPI ranks in a 2 by 2
layout; the engine uses one device with strict numerics. The schemes are
Thompson 8, RRTMG 4/4, Noah 2, surface layer 1, Tiedtke 6 and km5 with no
separate PBL. There are zero differing words in the 51,032,555-word
single-domain comparison and the 125,601,049-word two-domain comparison.
This is a claim about that configuration and numerical mode.

A six-hour fast Florida forecast uses all three settings, real GFS forcing
and the same Thompson reflectivity diagnostic as native. A midpoint engine
checkpoint reproduces all three subsequent hourly files byte for byte.
A separate ten-minute fast run gives identical history and final state
with one rank and two logical ranks on one physical device. A normal
30-minute real moving-nest run completes with finite fields and reviewed
weather, TKE and averaged-flux sheets. These functional checks do not
establish physical multicard performance or native moving-nest parity.

With the new selectors absent, ten-minute whole default-fast and
HRRR-recipe-strict runs match base `34951539d98` in every history word.
The HRRR recipe retains its established km_opt=4. Generic dry-route
repairs deferred to the release owner remain deferred.

On the 424 by 299 by 49 NCAR CONUS case at a 36-second step, the final
fast timing set measured median whole steps of 103.73 ms for the ordinary
km4/MYJ off control, 107.79 ms for km2 with PBL off, and 108.75 ms with all
three settings on: ratios 1.048 and 1.009. The six arms ran back to back
on shared hardware. These measured configuration-specific ratios do not
predict other meshes or physical multicard scaling. The integration
receipt retains per-stage times and both preflight and observed memory;
feature memory is not a universal fixed number of bytes per cell.
