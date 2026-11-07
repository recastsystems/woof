# Opt-in GECROS (WRF 4.7.1)

**native WRF 4.7.1 fails here; this fixes it, not bitwise to WRF**

Select Noah-MP (`sf_surface_physics=4`) and `opt_crop=2` in the WRF namelist.
The WPS crop geography (`CROPTYPE`, `SEASON_GDD`, planting and harvest fields)
is required. A cold start represents sowing; it does not represent an already
emerged crop. All sixty GECROS state words are GPU resident, checkpointed, and
carried across nest feedback. Rust writes history as sixty scalar planes
`GECROS01` through `GECROS60`; the native vector is not silently reshaped into
an incompatible NetCDF field. The option is never enabled by default.

## Native defect and the correction

The pinned source is WRF v4.7.1, commit
`f52c197ed39d12e087d02c50f412d90d418f6186`.
`phys/module_sf_noahmpdrv.F:884` selects crop-specific emergence parameters,
and `:2812` initializes GECROS. Flags 40–42 start at -1 and thermal time
(slot 43) at zero. `phys/module_sf_noahmplsm.F:3957` permits GECROS only when
flag 41 is positive and flag 42 negative. Its existing `EMERG` routine at
`:11000` accumulates temperature above the crop base and sets flag 41, but
there is no call to that routine anywhere in the pinned source. Thus the gate
cannot open after a cold start. The retained full strict native run confirms
zero thermal accumulation over both time steps; it is not a multi-day trial.

The correction calls that existing emergence calculation at the surface
flux boundary, using top-soil temperature and the driver's original crop
parameters. Wheat requires more than 65.5 degree-days above 2 C; maize
requires more than 81 degree-days above 8 C. Warm emerged states are left
alone. Before emergence, the ordinary Noah-MP conductance path supplies the
canopy coupling: native unconditionally consumes GECROS conductance outputs
although its inactive branch has not produced them. Harvest retains the
native thresholds, reset values, and reserved words. No threshold is shortened
to force a forecast to grow leaves.

The coupling's purpose is interactive crop phenology, canopy photosynthesis,
and water/energy exchange, as described in
[the Noah-MP–GECROS coupling paper](https://doi.org/10.1016/j.agrformet.2018.06.023)
and [the official UCAR coupling presentation, pp. 5–6](https://ral.ucar.edu/sites/default/files/docs/session4ingwersenmay24pptx.pdf).
Thermal emergence uses the implementation already supplied by WRF, rather
than introducing a new growth parameterization.

## Arithmetic and scope

`generate_gecros_complete.py` verifies the SHA256 of the complete upstream
`module_sf_gecros.F` before translating its 28 routines. WRF's public-domain
license is retained at `licenses/LICENSE-WRF-public-domain.txt`. The GPU uses
explicit FP32 operations, no contraction, and the repository's native-libm
implementations. All native constants and the native winter-wheat calibration
are retained, including where native uses that calibration for a maize crop
category. This is not a newly calibrated maize model. Native ASTRO stops for
crop latitudes beyond 67 degrees; initialization rejects that case explicitly.

The retained active-state oracle covers 128 seed/established crop states and
8448 output words. Its strict native compiler is GNU Fortran 15.2.0 with glibc
2.43. All words match on the qualified GPU platform. The release-box native
build uses glibc 2.39, whose trigonometric results differ: that platform is
**not** an exact replay claim. A whole cold forecast intentionally differs
from broken native. The oracle does not claim that the entire Noah-MP driver
or a full native forecast is bitwise matched.

Run the retained independent native oracle and GPU comparison:

```sh
FLAGS='-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize'
gfortran $FLAGS -ffree-form -c module_sf_gecros.F
gfortran $FLAGS -ffree-form -ffree-line-length-none \
  tools/port_gecros/oracle.f90 module_sf_gecros.o -o oracle
./oracle woof/data/gecros/oracle/inputs.bin native.bin
python -m pytest -q tests/test_noahmp_gecros.py
```

`woof/data/gecros/oracle/manifest.json` binds the captured native inputs and
outputs to source, compiler, flags, GPU commit, and hashes. The tests also
exercise cold initialization, both emergence thresholds, frozen/non-crop
exclusion, and all sixty carried words across a full-column restart.

A second native replay captures both crop harvest thresholds, already harvested
states, the complete native reset, and reserved-word retention: 128 rows,
7680 words, 0 differing words. `harvest_oracle.f90` embeds the pinned driver's
unmodified `checkIfHarvest` and `gecros_reinit` procedures behind its original
call gate (driver lines 1414–1420).
