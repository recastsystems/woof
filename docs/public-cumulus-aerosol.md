# Public monthly aerosol forcing

`aercu_opt=1` or `2` and `aercu_fct` retain their stock namelist meaning with
`cu_physics=11` and `mp_physics=40`. The default forcing is NASA MERRA-2 GMI
replay for 2001–2010: **written from papers, not bitwise to WRF**. It replaces
the uncleared CESM data file. The old filename is never opened by this route.
NASA GMAO and the MERRA-2 GMI project supply the public US government data.

Run a stock namelist with `woof go --wrfinput INPUT --outdir RUN`. On first
use the bundled Rust builder fetches public monthly source arrays, constructs
the pinned climatology, and caches it under `~/.woof/aerosol/`. This substantial
one-time preparation needs about 30 GB of source cache plus output/staging.
An interrupted preparation reuses complete source downloads. Subsequent runs
verify and reuse the output. No Earthdata login or credentials are needed.
A private installation can place `NASA_MERRA2_GMI_2001_2010_Aerosol.dat` beside
wrfinput, set `WOOF_PUBLIC_AEROSOL` to that file, or set
`WOOF_PUBLIC_AEROSOL_CACHE` to another cache directory. Every choice must
match the shipped output size and SHA-256. A wrong file fails before stepping.

For a source checkout build both providers with
`cargo build --manifest-path tools/aerosol_grid/Cargo.toml --release --locked --offline`.
The explicit conversion command is
`public-aerosol-build CACHE NASA_MERRA2_GMI_2001_2010_Aerosol.dat 2`.
Conversion and horizontal interpolation use bounded CPU thread pools; forecast
time/month/pressure interpolation and activation stay in CUDA. Python selects
files, transports arrays, and records provenance. Positive selectors consume
additional memory for all native-grid monthly fields; the ordinary memory
planner includes these arrays.

The defining [WRF aerosol–cloud paper](https://doi.org/10.1175/MWR-D-18-0267.1)
describes monthly CESM-NCSU/RCP4.5 climatology for 2001–2010, horizontal,
monthly-time and pressure interpolation. This replacement follows that
interface, rather than claiming the unavailable CESM realization is reproduced.
The runtime ABI has ten species and thirty ascending pressure levels. Output
records contain dimensions, latitude, longitude, pressure in Pa, and aerosol
concentrations in micrograms per cubic metre. Species are DUST1–4, SEASALT2,
SULFATE, BCPHO, BCPHI, OCPHO and OCPHI, in that order.
Pressure follows the executed public WRF `aer_p_int_cu` comparison with model
pressure in Pa; the legacy hPa comment and Registry millibar label do not
describe that calculation (the `*0.01` conversion is commented out).
The paper describes eleven CESM-NCSU bulk components; this file targets the
ten fields consumed by the public WRF cumulus driver. The selected GOCART
inventory has no separate nitrate, ammonium or secondary-organic-aerosol
field. Those missing components are not synthesized or folded into sulfate
or carbon. Matching this interface therefore does not reproduce CESM chemistry.

[The NASA project](https://acd-ext.gsfc.nasa.gov/Projects/GEOSCCM/MERRA2GMI/)
is a GMI/GOCART replay driven by MERRA-2 meteorology, without aerosol optical
depth assimilation. It is not the assimilated MERRA-2 aerosol product and does
not supply an RCP4.5 future scenario. The public
[OPeNDAP collection](https://opendap.nccs.nasa.gov/dods/merra2_gmi/instM_3d_aer_Nv)
provides 72 model layers; its `lev` values are layer indices, not pressure.
Layer midpoint pressure is 1 Pa plus accumulated DELP and half the current
layer thickness. Thirty declared source levels retain every near-surface
layer, with progressively wider stratospheric spacing. The grid is 1 degree
latitude by 1.25 degrees longitude, sampled from NASA's half-degree by
0.625-degree grid. Native-grid interpolation is bilinear and longitude periodic.

For each calendar month, concentration is the ten-year mean of each year's
monthly mass mixing ratio times monthly AIRDENS times 1e9. This omits within-month
density–mixing-ratio covariance. The algorithm does not interpolate aerosols
from pressure index labels. Source fill values, invalid DDS shapes, truncated
DAP2 responses and non-monotonic pressures fail conversion.

NASA dust radius edges 0.1, 1, 1.8, 3, 6 and 10 micrometres are doubled before
mapping to CAM4 diameter edges 0.1, 1, 2.5, 5 and 10 micrometres. Mass is assumed
uniform in log diameter inside each source bin. Dust above 10 micrometres
diameter is excluded, not relabelled as fine dust. This unresolved within-bin
size distribution is a scientific approximation. The four retained bins plus
excluded coarse tail conserve every input bin. See
[NASA's dust documentation](https://gmao.gsfc.nasa.gov/media/gmao_products/IqPeIBhIWLGrweILfVPj9Vmdoc/Dust_z72405udf3g1la8jwxmyspohi.pdf)
and [the CAM4 paper](https://doi.org/10.5194/acp-14-13043-2014).

All five NASA sea-salt bins are summed into WRF's bulk SEASALT2; its original
size spectrum is not reconstructed. NASA organic fields already represent
particulate organic matter; no extra 1.4 multiplier is applied. Sulfate is SO4
ion mass, without conversion to ammonium sulfate. These conventions follow
the [NASA FAQ](https://gmao.gsfc.nasa.gov/gmao-products/merra-2/faq_merra-2/).
No CESM-specific empirical bias fit is applied in preprocessing. Existing
public WRF activation-runtime multipliers remain in the inherited kernels.
No uncleared CESM code, data or fitted tables were read to derive this input.

Each positive run receipt identifies the source, output digest, mapping,
units and rewrite label. These also enter restart setup identity, alongside
the monthly array fingerprints and forecast epoch. A continuation cannot
silently change the forcing. Online nests construct monthly fields on each
domain's own coordinates; member executors retain member-local forcing and
activation state. Archived-parent offline MP40 aerosol coupling still requires
a parent binding that records CU11 and the aerosol controls; use an online
coupled nest until that separate binding is qualified.

With `aercu_opt=0` the acquisition, monthly allocations, interpolation and
receipt additions are skipped. Changing the unused `aercu_fct` does not affect
that branch. Native reference comparisons use the same public forcing under
the native program's legacy filename only inside the comparison scratch;
there is no comparison to the uncleared CESM data realization.

Native comparison files honor the reference build's unformatted-file byte
order. Reversing each four-byte word for a big-endian native build changes
neither dimensions nor floating-point values; reversing it back reproduces
the pinned little-endian SHA-256. The product uses the pinned little-endian
file.


Qualification on 2026-11-01 uses a 300×300×49, 3 km, 15 s Texas forecast
from 2026-10-16 09 to 12 UTC, MP40/CU11, factor 1, and ordinary fast numerics.
Both positive selectors complete all 720 steps through stock `woof go`.
The option-off control against merged prerequisite `44a3d61c773` has four
byte-identical hourly history files and 265 identical checkpoint arrays
(461,637,600 values). A selector-2 continuation from 10 to 12 UTC matches
266 arrays (505,737,600 values) and all 524 history variable entries of the
uninterrupted run; checkpoint container UUIDs are intentionally distinct.
Both selectors exercise the production online nest executor with feedback
0 and 1 and member-local MP40/CU11 ensemble executors. These are focused
lifecycle runs, rather than full real nested or ensemble forecasts.

At 12 UTC, 23 nearby ASOS stations give temperature RMSE 0.983, 0.979 and
0.977 degrees Celsius for off, selector 1 and selector 2. Rust MRMS maps show
an overly broad and strong southwest precipitation band and a missed
northeast observed line in every arm. Changing aerosol forcing does not
establish better forecast skill from this one case. The registry retains
`experimental-runtime` and scientific evidence `none`.

Positive selectors ship as **opt-in options, off by default**, with their measured
cost disclosed. They are **written from papers, not bitwise to WRF**. The follow-up
paired, interleaved fast-numerics physics stage measurement on an RTX PRO 6000
Blackwell was 47.5594 ms off, 175.3204 ms for selector 1 (3.6864x) and
175.1594 ms for selector 2 (3.6830x). The sequence was off, 1, off, 2,
using the same 300x300x49 case above and 40 real forecast steps per arm. CUDA
events exclude each first call; the denominator is the mean of the two off
controls. The GPU timing queue isolated card 1; the lane report records host
load and other-card co-tenants. Costs include the MSKF and MP40 parents.

Positive fast MSKF uses twice as many columns per launch tile (four tiles per
step on this card instead of eight); strict and off geometry are unchanged.
Earlier shared-card samples were 55.0439 ms off, 356.0892 ms for selector 1
(6.4692x), and 355.4803 ms for selector 2 (6.4581x). Changing shared load prevents
a causal before/after speed claim. The current lane report records the full
stage split, separate profiles, byte checks and one complete positive forecast.
A physics-cost exception is not required for this named opt-in route. No general
forecast skill is claimed.

Detailed commands, configurations, logs, independent data audit, native
comparisons, map review, timing and lifecycle receipts are delivered in the
local `WRF-PORTS-2026-11-01/lic-cesm-aerosol.md` lane report. The input is a
public-data replacement with declared bin/chemistry approximations, not a
qualification of the original CESM realization.
