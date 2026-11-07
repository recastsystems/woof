# Noah-MP direct soil properties (WRF 4.7.1)

**native WRF 4.7.1 fails here; this fixes it, not bitwise to WRF**

Select `sf_surface_physics=4, opt_soil=4` explicitly. The default stays 1.
Supply nine binary32 fields in the input/restart file on
`(Time, soil_layers_stag, south_north, west_east)`, four soil layers:
`BEXP_3D`, `SMCDRY_3D`, `SMCWLT_3D`, `SMCREF_3D`, `SMCMAX_3D`,
`DKSAT_3D`, `DWSAT_3D`, `PSISAT_3D`, `QUARTZ_3D`.
Units are respectively dimensionless, four volume fractions, m/s, m2/s,
m, and dimensionless. Inputs are decoded by the existing Rust NetCDF batch
reader, gathered on the GPU each land call, and retained in history/restarts.
Missing fields refuse; values are never inferred or clipped.

The pinned WRF release is f52c197ed39d12e087d02c50f412d90d418f6186,
Noah-MP submodule e5c0859874407859936739e8be8741f9aed369ee.
In `drivers/wrf/module_sf_noahmpdrv.F:742-751` option 4 never assigns
SOILTYP, then `:974` indexes tables through TRANSFER_MP_PARAMETERS.
The direct-input declarations and assignments at `:177-187,:932-942`
are commented placeholders. Merely assigning SOILTYP would silently run
option 1 and would not implement option 4.

The completed arm first transfers category parameters, as native option 3
already does, then replaces the nine per-layer hydraulic vectors for natural
land. It retains category-derived KDT/FRZX, other ancillary parameters, and
the native urban overrides. No change is made to NOAHMP_SFLX equations or
the CUDA implementations of them. This is an explicit completion of an
unsupported WRF driver interface, not whole-model native option-4 parity.

The intended direct-property interpretation is confirmed by the
[upstream maintainer explanation](https://github.com/NCAR/noahmp/discussions/160)
and the [LIS 7.5 documentation, p.192](https://lis.gsfc.nasa.gov/sites/default/files/LIS/public_7_5_releases/LIS_users_guide.11-jan-2024.pdf).
The underlying physics is [Niu et al. (2011)](https://www2.mmm.ucar.edu/wrf/site_linked_files/phys_refs/land_surface/noah_mp_part1.pdf).
Only the pinned public-domain Noah-MP v4 code is used; see
`licenses/LICENSE-WRF-public-domain.txt`.

`replay.py` compares every formal output of captured working native
NOAHMP_SFLX calls, after routing their direct vectors through the completed
GPU selector. It requires GPUWM_WRF_EXACT=1 and the explicit WRF 4.7 snow
age generation. Native undefined/pass-through outputs retain the existing
source witnesses. These scoped zero-word replays do not claim that stock
WRF's broken opt_soil=4 driver ran successfully.

The retained [qualification receipt](qualification.json) records the completed
30-minute real GO case and its synthetic direct-property geography, all twelve
visually inspected Rust maps, option-off whole-file byte identity, scoped
native 0/5,880-word comparisons, actual GPU memory samples and the interleaved
complete-land-driver option-3/4 cost. Full commands and logs are retained at
the receipt's evidence path; every native capture input/reference and both
forecast arms have local paper manifests and SHA256SUMS.

The working/warm native replay is a routine-boundary claim. It does not turn
the stock broken WRF option-4 driver into a successful native forecast.
