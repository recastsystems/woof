# A real forecast of a real outbreak, streamed out of core, rendered by WOOF-ArWen

**Case.** 2019-05-28, the central/eastern Kansas dryline outbreak. A dryline
across central Kansas fired discrete supercells through the afternoon; the
Lawrence–Linwood EF4 was on the ground from 2306Z to about 0000Z across
Douglas and Leavenworth counties, and the same evening produced hail and wind
reports from Nebraska to Missouri. The forecast is initialised from the
operational **HRRR 15Z analysis** of that day, which puts model time zero
about five hours ahead of initiation and eight ahead of the violent tornado.

**What is real, and what is not.**

* the initial condition is the operational HRRR **f00 analysis** (50 native
  hybrid levels) from the NOAA archive `noaa-hrrr-bdp-pds`, decoded by the
  packaged `hrrr_grib2_bridge`, interpolated by
  `woof.ingest.hrrr.interpolate_hrrr_to_lambert` and turned into a model
  state by `woof.ingest.real.initialize_real`: the production ingest,
  called, not reimplemented;
* terrain, land use, soil category, greenness, albedo and deep soil
  temperature are the real WPS\_GEOG geogrid statics for the target grid;
* the lateral boundaries are the **same HRRR cycle's f01…f09**, on their real
  hourly cadence, linearly interpolated between them, never synthesised from
  the model's own state;
* **everything in the pictures is SIMULATED.** Reflectivity is the model's own
  `REFL_10CM`, not a radar mosaic. Observed storm reports, where they appear,
  are SPC's and are marked as observations.

**The lateral boundary is not WRF's.** A tile is stepped as a small periodic
domain of its own, so WRF's `spec_zone`/Davies treatment cannot be applied to
the three tile edges that are really interior seams. What this run applies
instead is a wide Davies frame at the sweep seam: the outer 20 cells are
replaced hard by the time-interpolated analysis and a 24-cell cosine taper
blends to the free interior. That is a real lateral boundary condition of the
AROME/HIRLAM family driven by real analysis data, and it is **not**
bit-comparable with a monolithic WRF-style run. Nothing inside the outer 44
cells is a forecast.

**Cumulus is off, deliberately.** `cu_physics = 0`. At Δx = 3 km the model is
convection-allowing and a cumulus parameterisation removes the instability the
explicit updraughts exist to release. The brief asked for Kain-Fritsch; KF is
the right choice at the 12 km of `real74_d01` and the wrong one here, so it is
off and this paragraph is why. Every other selector is the streaming gate's
own `full+MYNN+Noah-MP` rung: Morrison mp10, MYNN surface layer and PBL,
Noah-MP, RTE-RRTMGP on a 12-minute cadence.

## Two defects this case found, because it is the first time the HRRR path was stepped

Neither is a streaming defect. Both would happen to a resident run at a size
that fitted; the streamed run simply got there first.

**1. The deep-soil boundary is the wrong field, and by up to 26 K.**
`woof.ingest.hrrr_physics.initialize_hrrr_physics` hands Noah-MP the raw
geogrid `SOILTEMP` as its bottom boundary. `woof/static/build.py` already
builds what WRF uses (`TMN = SOILTEMP − 0.0065·HGT_M` on land, WRF v4.6.1
`share/module_soil_pre.F:973`) and ships it in the same static bundle,
unused by this path. Measured on this domain: **155 075 of 931 808 land
columns have a bottom boundary ≥10 K too warm, 4 432 ≥20 K, worst 26.0 K at
4 000 m.** At (j=441, i=276) the model was told 291.95 K under a soil column
sitting at 268.5 K, and 268.48 K is exactly `SOILTEMP − 0.0065·HGT`. The run
uses the corrected field (`--deep-soil tmn`).

**2. Sixteen Rocky Mountain soil columns go non-finite on the first Noah-MP
call.** 2717–3612 m, 37.6–40.6°N, sub-freezing soil layers, ordinary land use
(1 and 10) and ordinary soil categories (3 and 6). `fields/tslb` takes 64
non-finite values (16 columns × 4 layers) after one sweep; on the next
sweep they come back in as input, the surface fluxes go with them, and MYNN's
mass-flux guard refuses, **killing a 52.9-million-cell forecast over sixteen
mountain columns 1 500 km from the storms.** Correcting the deep-soil
boundary does *not* fix it (tested, same columns, same step) so it is a
separate defect and is reported as one. This run contains it: the land
surface only is held at its analysis value in cells that have gone
non-finite, at the sweep seam, never touching a prognostic, capped, and
counted into `run.json`. That is containment, not a fix.

**UP\_HELI\_MAX in these files is INSTANTANEOUS.** WRF's is a running maximum
between history writes. The device UH lane refuses a tiled periodic geometry
by design (`uh_diag._supported_boundary_geometry`), so what is computed is the
whole-domain host mirror of `cal_helicity` evaluated at each sweep seam: the
2–5 km updraft helicity at that instant. Every wrfout carries
`GPUWM_UP_HELI_MAX_SEMANTICS` saying so.
