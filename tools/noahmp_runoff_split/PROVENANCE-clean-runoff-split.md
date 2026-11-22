# Provenance: separate surface and subsurface runoff options (opt_runsrf / opt_runsub)

Label carried by the option: **re-implemented from published descriptions, not bitwise to upstream.**

## What was read

Only published descriptions and our own tree:

- He, C., et al. (2023), *Modernizing the open-source community Noah-MP land surface model (version 5.0) with
  enhanced modularity, interoperability, and applicability*, GMD 16, 5131-5151 (CC BY 4.0). Sect. 2.6 states that
  v5.0 "separated the runoff options for surface and subsurface runoff processes". Table 1 lists the options.
- He, C., et al. (2023), *The Community Noah-MP Land Surface Modeling System Technical Description Version 5.0*,
  NCAR/TN-575+STR, doi:10.5065/ew8g-yr95 (public PDF on the WRF physics references page; CC BY-NC 4.0 document,
  read as a description, nothing from it is shipped). Sect. 3.7.8 "Soil Hydrology": option tables for
  OptRunoffSurface and OptRunoffSubsurface (both 1-8, "currently tested & recommended the same option# as surface
  runoff"), Eqs. 3.7.189-3.7.338, and the namelist mapping table (OPT_RUNSRF, OPT_RUNSUB, OPT_INFDV).
- NCAR Noah-MP tutorial slides, C. He, 25 May 2023 (public PDF): names SURFACE_RUNOFF_OPTION,
  SUBSURFACE_RUNOFF_OPTION and DVIC_INFILTRATION_OPTION (1 Philip, 2 Green-Ampt, 3 Smith-Parlange).
- WRF user forum thread 27515 (WRF 4.8.0 Noah-MP namelist change: `opt_run` replaced by `opt_runsrf`/`opt_runsub`),
  cited by the program catalogue.
- Our own WRF-v4-lineage port: `woof/core/kernels/noahmp_water_options.cuh` (from WRF v4.6.1 Noah-MP 848f54ad,
  distributed by WRF under its public-domain notice).

Not opened, cloned, downloaded or quoted: any file of NCAR/noahmp, the WRF 4.8.0 `phys/noahmp` submodule, HRLDAS
source, or any other UCAR-licensed copy. The tech note's code-module names (e.g. `RunoffSubSurfaceDrainageMod.F90`)
appear in its prose; no such file was read.

## What each routine does and where it comes from

The physics of every scheme is unchanged from our v4 port. The change is which half of the pair each `OPT_RUN`
gate belongs to. `tools/noahmp_runoff_split/generate.py` cuts SOILWATER/WATER/column out of the v4 options header
and reassigns each gate; every replacement is asserted to hit exactly once.

| Gate in the v4 port | Assigned to | Published basis |
| --- | --- | --- |
| Equilibrium water table (ZWTEQ) | surface 2 **or** subsurface 2 | Eq. 3.7.204 (surface 2 reads it), Eqs. 3.7.322-3.7.327 |
| TOPMODEL baseflow `(1-FCRMAX)*4*exp(-TIMEAN)*exp(-2 ZWT)` | subsurface 2 | Eq. 3.7.328 |
| Saturated-fraction surface runoff (TOPMODEL, MMF) | surface 1, 2, 5 | Eqs. 3.7.198-3.7.204, 3.7.225-3.7.227 |
| Schaake infiltration, BATS, VIC, XAJ, dynamic VIC | surface 3, 4, 6, 7, 8 | Sect. 3.7.8.1, Eqs. 3.7.205-3.7.236 |
| Fine-step re-evaluation of surface runoff | surface 3, 6, 7, 8 | Sect. 3.7.8.2 item (1) |
| Bottom drainage flux in the Richards solve (`0`, `SLOPE*WCND`, `(1-FCRMAX)*WCND`, MMF) | subsurface | Eq. 3.7.255 (keyed on OptRunoffSubsurface), Eqs. 3.7.256-3.7.257 |
| MMF bottom store in the tridiagonal step | subsurface 5 | Eqs. 3.7.256-3.7.257 |
| Groundwater-flow soil water removal | subsurface 2 | Eqs. 3.7.329-3.7.330 |
| Low-water floor, taken from RUNSUB (or DEEPRECH for 5) | subsurface /= 1 | Eqs. 3.7.331-3.7.337 |
| Niu et al. (2007) groundwater discharge | subsurface 1 | Sect. 3.7.8.5 item 4 |
| `RUNSUB += QDRAIN` | subsurface 3, 4, 6, 7, 8 | Eq. 3.7.338 |
| MMF shallow water table | subsurface 5 | Sect. 3.7.8.5 item 5 |
| Tile drainage | both halves 3 (split refuses tile drainage) | Sect. 3.7.8.4 ("only tested & calibrated to work with runoff option=3") |

Subsurface 3, 6, 7 and 8 are one scheme (Eq. 3.7.338 with the `SLOPE*WCND` bottom flux), so a pair whose
halves are both in {3, 6, 7, 8} is the stock pair and runs the stock code.

## Choices we made where the description is silent

- **Mapping:** a stock `opt_run=N` is `(N, N)`; a missing half of a WRF 4.8.0 pair inherits `opt_run`.
- **Refused pairs**, each naming its breakage: surface 1 needs a water table that subsurface 1 or 2 updates
  (otherwise Eq. 3.7.200 reads the cold-start ZWT forever); surface 5 needs subsurface 5 (the MMF water table and
  its lateral step); TOPMODEL (1, 2) with MMF (5) mixes ZWT sign conventions (positive depth vs negative); tile
  drainage on a split pair would silently not drain; surface 8 needs `opt_infdv` 1-3 (existing rule).
- **State:** a pair carries the groundwater state of whichever half needs it (TOPMODEL ZWT/WA/WT or the MMF set).

## Tests

- `tests/test_noahmp_runoff_split.py`: the split kernel at every stock pair (N, N), both `opt_inf`, is bit-identical
  to the stock selected kernel on the 36 WATER fixture columns; a mixed pair's ZWT and RUNSUB match a hand
  computation of Eqs. 3.7.322-3.7.328 under Schaake and VIC surface runoff; resolution and refusals.
- Option-off: a bare run and a stock `opt_run` run are byte-identical before and after the change (see the port's
  RESULT.md).
