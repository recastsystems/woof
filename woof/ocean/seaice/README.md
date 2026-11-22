# Prognostic thermodynamic sea ice

`seaice_prognostic = true` is the default. A domain with an LSM and at least
one ocean-ice cell activates the CUDA model. Lake cells retain their lake
model. `false` explicitly selects the existing prescribed-thickness solver.
The ordinary `woof go CONFIG --outdir OUTPUT` door runs the coupling.
Render with `--products var:wrf_seaice,var:wrf_icethick,var:wrf_iceconcchange,var:wrf_icethickchange`.

The model implements the zero-layer ice and insulating snow balance from
https://doi.org/10.1175/1520-0485(1976)006%3C0379:AMFTTG%3E2.0.CO;2,
with original Apache-2.0 CUDA code. No Icepack source is included.

Ice concentration A, grid-cell ice volume V (m), and snow mass S (kg m-2)
are prognostic. Ice thickness is V/A, snow depth is S/(330 A).
The surface satisfies Qnet + (Tf - Ts)/(hi/2.03 + hs/0.31) = 0, up to the
surface melting point. Excess energy melts snow first, then ice. The ocean
boundary flux is positive into ice. Basal growth follows the Stefan balance
between upward conductive heat and that ocean flux. Sublimation/deposition
changes solid mass. Solid precipitation uses the microphysics phase fraction.
Melting reduces concentration with A proportional to sqrt(V); new ice fills
open area at the configured new-ice thickness. These are thermodynamic area
changes. There is no horizontal ice transport, ridging or mechanical lead
opening. The model is not an ice-sheet melt/runoff product.

The surface layer supplies the exchange velocity CHS. Ice fluxes use the
nonlinear ice saturation curve and feed HFX/QFX/LH and the PBL after the LSM.
Water fluxes are calculated at the supplied SST. Fluxes and skin temperature
are area weighted, with no second blend of an already blended LSM flux.
Noah's prescribed-thickness sea-ice call is replaced when the prognostic
column owns the state. Noah-MP and RUC use the same post-LSM owner.

`seaice_initial_thickness = 1.0` m is an explicit uniform prior, not an
observed thickness. Concentration and snow initialize from the public
forcing's XICE and SNOW fields. `seaice_new_ice_thickness = 0.5` m controls
new-ice area. `seaice_ocean_heat_flux = 2.0` W m-2 is a prescribed lower
boundary, not a forecast ocean circulation. Ice-free water only freezes
when the water-temperature owner supplies water at/below 271.35 K.
With `sf_ocean_physics=1`, the mixed layer consumes the ocean-side heat budget
where ice is present (`ice_ocean_coupled`, plus owned water the atmosphere sees
as land); ice-free open ocean keeps the atmosphere's own fluxes,
and updates the water SST separately from the blended ice/water skin. Any mixed-layer cooling below freezing forms additional ice with equal
and opposite latent heat, leaving the water at its freezing point. With
`sf_ocean_physics=2` (3D PWP, chosen by setting it), the
PWP step skips its atmospheric surface heat where ice is present (the columns
whose surface fluxes the ice step set, `ice_ocean_coupled`),
and the PWP mixed layer (the surface layer and every layer whose top lies
above OM_ML) takes the same ocean-side heat budget instead, spread evenly over
its depth, so the ocean-to-ice heat flux leaves the column. Each layer this
leaves below Tf freezes the same way, giving up (Tf-T) rho c dz. Heat too
small for a float32 temperature to resolve in one step waits in
`ice_ocean_owed` (J m-2) rather than being rounded away. Columns PWP does not
own are left alone. On a cold start the PWP
profile is seeded no colder than the freezing point where this model owns
the water, as SST is, so analysis water between its salinity freezing point
and 271.35 K is not turned into ice. With
that option absent, SST and the lower heat boundary remain prescribed.
Analysis SST masked to zero beneath sea ice is initialized at the seawater
freezing point. In coupled columns, entrained water is bounded below by that
freezing point and mixed-layer temperature follows liquid enthalpy; surface
cooling remains available to the latent-heat reconciliation.

All prognostic fields and cumulative budgets live in the existing restart
field dictionary. History includes ICEVOLUME, ICETHICK, ICESNOWMASS,
ICESNOWDEPTH, ICETSK, ICETHICKCHANGE, ICECONCCHANGE, ICEGROWTH, ICEMELT,
ICEENERGY, ICEENERGYERROR and
ICEOCEANEXCHANGE. Concentration is the existing SEAICE field. ICEENERGY
is accumulated solid-column external enthalpy including snowfall. Its
comparison energy is -333700*(917*ICEVOLUME + ICESNOWMASS) J m-2.
ICEENERGYERROR accumulates the actual stored-state energy increment minus
that budget, including float32 storage rounding. Melt energy left after
complete ice loss returns to the ocean; an unsupported snow load enters the
ocean with its negative fusion enthalpy, rather than disappearing.

Observed concentration initialization does not establish thickness accuracy
or forecast skill. Qualification requires a whole multi-day case, the ice
analysis comparison, a complete time sequence rendered through woof.rustwx,
and visual inspection, in addition to the conservation and restart tests.

## Initialization and verification sources

ERA5 sea-ice concentration is available through the existing public ARCO
forcing door, with the Copernicus/ECMWF CC BY 4.0 attribution:
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview
(DOI 10.24381/cds.adbb2d47). The current preparation path binarizes XICE;
the thermodynamic owner then evolves fractional concentration. It does not
recover fractional analysis information discarded during preparation.

Daily extent verification uses U.S. National Ice Center (2008), IMS Daily
Northern Hemisphere Snow and Ice Analysis, G02156 Version 1, NSIDC,
https://doi.org/10.7265/N52R3PMC. This US government analysis requires the
NSIDC dataset citation. IMS ice/ocean classes verify extent, rather than
thickness or fractional concentration. With downloaded 4 km IMS ASCII gzip:

```sh
python -m woof.ocean.seaice.verify --history FINAL_WRFOUT \
  --initial-history INITIAL_WRFOUT --analysis IMS_ASC_GZ --out VERIFICATION --render
```

A 72-hour ERA5-initialized Arctic run from 2025-02-20 completed 4,320 steps
on a 64x52x49, 25 km regional grid with the default ice owner and coupled
mixed layer. Maximum ice thickness grew from the declared 1 m prior to
1.037 m. Day-three IMS extent CSI was 0.872 versus initial-extent
persistence 0.868; accuracy was 95.84% versus 95.81%. This small difference
in one case does not establish general forecast skill or thickness skill.
The maximum independent relative solid-enthalpy residual was 7.12e-5.
All seven history times were rendered by rw_wrfbatch through woof.rustwx
and visually inspected. Existing WPS soil/deep-soil-temperature geography
used in that private run still lacks a complete source/version permission
audit; these tiles are not newly adopted or redistributed by this module.
