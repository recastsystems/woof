# TC relocation primitives

`vortex.py` implements CUDA separation and translation on common isobaric regular latitude/longitude patches. The `woof prep --source gfs --vortex-init relocate --vortex-inputs REQUEST.json` source door applies a sealed canonical transform before parent and child interpolation. This route is implemented but not yet qualified by a complete native forecast.

The background filter is the 1993 published 11-pass, one-degree, zonal then meridional filter. The cylindrical boundary construction uses 24 azimuths by default and radial scale r0/5. This implements the original boundary interpolation, not the later HAFS optimum interpolation. Input patches must contain the complete source and destination disks plus the coarse filter's 11-degree halo. Truncated patches are refused.

`basic_field` returns the filtered coarse fields. `relocate_scalars` removes the original anomaly and inserts its spherical translated/resized version. `relocate_vectors` also rotates the Earth-relative wind anomaly between the source and destination radial/tangential bases. Outside both disks, output copies the original values exactly. Identity geometry returns a copy without subtract/add roundoff.

`radial_map` fits a monotone quadratic through the diagnosed RMW and the chosen outer filter radius. The latter is not automatically an observed quadrant R34. The caller must define its size constraint and preserve member anomalies. Invalid or folding transformations are refused.

Native GFS preparation performs its usual eta interpolation after the source transform. Hex mesh handoff and complete assimilation cycling remain unfinished. Never pass native eta levels as if they were common pressure levels. The new kernels compile with NVRTC; transform runtime and forecast evidence are pending. No default is changed.

Equations independently implemented from DOI 10.1175/1520-0493(1993)121<2030:AISOHM>2.0.CO;2 and DOI 10.3390/atmos11090968. No external implementation copied. Code is Apache-2.0 under the repository licence.

`gradient_wind_ratio` integrates paired axisymmetric tangential-wind profiles on their respective source/target radii, using the boundary-layer-top gradient-wind formulation from the 2020 paper. `thermal_balance` applies the resulting horizontal ratio to an axisymmetric temperature anomaly and uses the published same-pressure constant-RH approximation for vapour. `pressure_patch` composes these operations with diagnosed profiles, intensity adjustment and the pressure/geopotential anchor. Native rediagnosis and forecast spin-up still need validation.

`axisymmetric_profiles` diagnoses scalar, counterclockwise tangential and outward radial anomalies using spherical azimuthal quadrature. It uses each member's own separation and does not borrow another member's vortex. `woof.tc.frames.publish` sends resulting device fields to the Rust canonical writer, with sealed input and transform hashes. The writer preserves grid/clock identity and unchanged field payloads; the native source door verifies the input GRIB and both canonical manifests before exact-lattice storage selection.

`profile_field` evaluates inverse-mapped radial profiles at the destination with spherical wind-basis rotation. `intensity_gain` intersects the resolved vector-wind constraints to find a positive axisymmetric gain achieving the requested 10-m maximum while retaining the environment and asymmetric residual; incompatible targets are refused. `hydrostatic_increment` preserves the prior discrete column residual; `pressure_anchor` reconciles its constant-height offset and surface-pressure reduction with an explicit changed sea-level pressure at fixed terrain. The latter uses lowest-pressure-level/near-surface virtual-temperature extrapolation and must be checked by native rediagnosis and spin-up. These new kernels pass CPU-only NVRTC compilation; runtime evidence remains pending.

The request schema is `gpuwm-tc-source-init/1`, containing `before`, `after`, `library` and their SHA-256 identities. `frame_transform.relocate_frame` publishes the transformed canonical frame and its provenance. The GFS mapping includes the native 24-level ladder through 40 hPa, source specific humidity, and explicit mixed-phase RH. The native door leaves later forcing unchanged and selects original decoder bits at unchanged source cells. It refuses posted/incremental preparation and reuse routes that would bypass source admission.

`--vortex-init off` is the default and imports no TC transform. Whole-case bitwise equivalence is still pending. `vitals-da` is reserved for the private data-assimilation cycle; the public source-transform door refuses it explicitly. There is no automatic default promotion, independent MSLP fit or quadrant R34 fit. RMW and the outer separation radius constrain the radial map; pressure and R34 are diagnostics for qualification.

The optional canonical-to-WPS export policy `native-gfs-skin` retains the
native MPAS initial GFS surface contract: SKINTEMP is read as the initial
surface temperature; a separate SST slab is not required for this path.
It still requires snow water equivalent and sea ice from the source, plus
soil and land mask. `configs/rw-wps-gfs-tc-native.mapping.json` adds those
GFS fields to the TC pressure mapping. The normal `complete` policy exports
all fields in the standard WPS contract. No surface-update series is created.
Source field identities: NOAA GFS inventory
https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.f000.shtml
and NOAA GRIB2 tables 4.2-0-1 and 4.2-10-2. These are US government data;
no external implementation is copied. Native runtime qualification is pending.
The GFS snow mapping uses the existing `landmask_zero_water` policy from the
native GDAS donor mapping: missing snow over land is refused and snow over
source water cells is explicitly zero. This is a cold-start surface assumption,
including ice-covered water; it is not a snow analysis or an observational
claim. The land mask and policy are retained in canonical provenance. The
vortex transform never changes these surface fields.

The real GFS canonical transformation and WPS source export have now executed
on a full global initial field. Native 72-hour forecast, mesh initialization,
assimilation-cycle skill, and whole-case OFF equivalence remain unqualified.

`tools/tc_vitals` also builds `rw_tc_rmw_score DIAGNOSTICS.json BDECK.txt
STORM YYYYMMDDHH OUTPUT.json`. It supplements the shared verification harness
with exact-time RMW errors from the documented ATCF MRD field. It collapses
repeated isotachs, refuses conflicting MRD, and reports missing or zero MRD
without interpolation. Input diagnostic samples must be one deterministic
member, with `elapsed_seconds`, `variables` and `hofx`; its schema is the
native diagnostic receipt. The model quantity is grid-point maximum-wind
radius, with grid and observation sampling uncertainty. The shared harness
remains responsible for track, intensity and quadrant-radii comparisons.

The common-pressure transform now accepts `minimum_pressure_pa`. It constrains
one common wind gain using the quadratic gradient-wind streamfunction before
wind, pressure and temperature are updated. It never clips pressure separately.
When the retained vortex cannot reach the requested wind without overdeepening,
the pressure bound wins and the receipt exposes the remaining wind mismatch.
This is a constrained extension of the published balance relation, not a claim
that both observed intensity quantities can always be matched by one amplitude.
The public source door remains opt-in pending native forecast qualification.
The radial correction defaults to the paper's 15 percent per-cycle bound and
records both requested and applied radii; `size_change_limit` remains selectable.

TC WPS export explicitly requests `--longitude-origin zero`. The Rust exporter
cyclically permutes every row without interpolation, so the native MPAS
surface-wind reader receives its expected global longitude convention. The
receipt records the origin and rotation. Ordinary WPS export keeps its source
origin. For already retained WPS exports, `gpuwm_mapped_engine wps-reperiod
--input MET --out NEW_MET` performs the same exact byte permutation and records
both hashes; it never changes the input. This is storage repair, not a new
vortex transform.
