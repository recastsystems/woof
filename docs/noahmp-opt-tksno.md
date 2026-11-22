# Noah-MP `opt_tksno`: snow thermal conductivity choices

Label: **re-implemented from published descriptions, not bitwise to upstream.**

## What it is

Noah-MP v4 (the WRF-lineage code WOOF-ArWen ports, WRF 4.6.1 / Noah-MP `848f54ad`)
hard-codes one snow thermal conductivity in CSNOW, k = 3.2217e-6 ρ². Noah-MP v5
made that a namelist choice with five formulas (He et al. 2023, GMD 16:5131,
Sect. 2.6: "added a new physics option for snow thermal conductivity"; the
option list and its five references are in the public NCAR Noah-MP tutorial,
He, 25 May 2023). `opt_tksno` gives our v4 port the same choice. Option 1 is
the v4 formula and the default. With the option unset or 1, the run is byte-
identical to the tree before this change.

No Noah-MP v5 source code was opened, cloned, read or quoted for this work: not
NCAR/noahmp, not the WRF 4.8.0 noahmp submodule, and no UCAR-licensed copy.
Every formula below comes from its own paper. Only the list of which five
formulas v5 offers comes from the public documentation.

## The formulas

ρ is the snow-layer bulk density (SNICE + SNLIQ) / DZSNSO in kg m⁻³, the value
CSNOW already forms. k is in W m⁻¹ K⁻¹.

| opt | source | k(ρ) | k at 250 kg m⁻³ |
|---|---|---|---|
| 1 (default) | Yen, Y.-C. (1965), CRREL Res. Rep. 175, Stieglitz form | 3.2217e-6 ρ² | 0.2014 |
| 2 | Anderson, E. A. (1976), NOAA Tech. Rep. NWS 19 | 0.02 + 2.5e-6 ρ² | 0.1763 |
| 3 | constant | 0.35 | 0.35 |
| 4 | Verseghy, D. L. (1991), Int. J. Climatol. 11:111-133 (CLASS I) | 2.576e-6 ρ² + 0.074 | 0.235 |
| 5 | Yen, Y.-C. (1981), CRREL Rep. 81-10; Douville et al. (1995), Clim. Dyn. 12:21-35 | 2.22 (ρ/1000)^1.88 | 0.1638 |

Anderson's intercept is quoted in the literature as 0.02 or 0.021; 0.02 is used
here. The difference is at most 0.001 W m⁻¹ K⁻¹.

## Where it acts, per routine

- **Land, THERMOPROP / CSNOW** (`woof/core/kernels/noahmp_tksno_options.cuh`,
  kernel `noahmp_thermoprop_tksno`). It runs after the unchanged default leaf
  `noahmp_leaf_thermoprop` on the same packed rows, only when `opt_tksno` is not
  1. It replaces DF in each snow layer (CSNOW's TKSNO) and the snow/soil
  interface DF of soil layer 1. That interface value is THERMOPROP's thickness-
  weighted blend of the layer-1 soil value with the bottom snow layer. The
  layer-1 soil value is formed again with the same TDFCND, urban and IST=2
  statements, so only the snow conductivity changes. With no snow layer
  (ISNOW = 0), THERMOPROP blends with the constant 0.35 and nothing changes.
  Heat capacity, the volumetric fractions and FACT are untouched. Every consumer
  of DF (VEGE_FLUX and BARE_FLUX ground heat, TSNOSOI) reads the same array.
- **Glacier, THERMOPROP_GLACIER / CSNOW_GLACIER**
  (`woof/core/kernels/noahmp_glacier_options.cuh`). The selected glacier unit
  takes `opt_tksno`. Value 1 keeps its original expression. Values 2-5 call
  `noahmp_tksno_selected`. The glacier snow/soil interface blend already reads
  the snow DF.
- **Host reference** (`woof/core/noahmp_tksno.py`): the float32 formulas, and
  the admission check that every entry point shares.
- **Doors**: `RunConfig.opt_tksno` (keyword-only, default 1), the Noah-MP option
  registry in `woof/config.py` (values 1-5), `&noah_mp opt_tksno` in namelist
  import, and `physics.py` to `NoahmpRuntimeParameters`. A non-default value
  routes the land and glacier steps through the resident selected-option path
  (`noahmp_surface_device.selected_options`).

## What it changes in a forecast

A higher k couples the snow surface more tightly to the ground beneath it. At
night the surface draws more heat up from the soil, so skin temperature and T2
are warmer over snow, and the soil under the snow cools and freezes deeper. A
lower k insulates more: colder nights over snow, warmer soil. The effect is
largest on clear, calm winter nights over deep, low-density snow, at high
latitude and altitude, and it builds over days, so it matters most in S2S runs.

## References

- He, C., et al. (2023). The Community Noah-MP Land Surface Modeling System
  Technical Description Version 5.0 / Modernizing the open-source community
  Noah with multi-parameterization options (Noah-MP) land surface model
  (version 5.0) with enhanced modularity, interoperability, and applicability.
  Geosci. Model Dev., 16, 5131-5151, doi:10.5194/gmd-16-5131-2023 (CC BY 4.0).
- He, C. (2023). Noah-MP tutorial, NCAR, 25 May 2023 (public slides).
- Yen, Y.-C. (1965). Effective thermal conductivity and water vapor diffusivity
  of naturally compacted snow. J. Geophys. Res., 70(8), 1821-1825.
- Anderson, E. A. (1976). A point energy and mass balance model of a snow
  cover. NOAA Tech. Rep. NWS 19.
- Verseghy, D. L. (1991). CLASS - A Canadian land surface scheme for GCMs. I.
  Soil model. Int. J. Climatol., 11, 111-133, doi:10.1002/joc.3370110202.
- Yen, Y.-C. (1981). Review of thermal properties of snow, ice and sea ice.
  CRREL Report 81-10.
- Douville, H., Royer, J.-F., Mahfouf, J.-F. (1995). A new snow
  parameterization for the Meteo-France climate model. Part I. Clim. Dyn., 12,
  21-35.
